'use strict';

const crypto = require('crypto');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  GetCommand,
  TransactWriteCommand
} = require('@aws-sdk/lib-dynamodb');
const { resolveAwsCredentials } = require('./aws-credentials');

const memoryStore = new Map();
let cachedDocClient = null;
let cachedClientKey = '';
const CHATBOT_STATIC_CREDENTIAL_SETS = Object.freeze([
  Object.freeze({
    name: 'chatbot',
    accessKeyId: 'CHATBOT_AWS_ACCESS_KEY_ID',
    secretAccessKey: 'CHATBOT_AWS_SECRET_ACCESS_KEY',
    sessionToken: 'CHATBOT_AWS_SESSION_TOKEN'
  }),
  Object.freeze({
    name: 'default',
    accessKeyId: 'AWS_ACCESS_KEY_ID',
    secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
    sessionToken: 'AWS_SESSION_TOKEN'
  })
]);

function numberEnv(key, fallback) {
  const raw = process.env[key];
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function getLimitConfig() {
  return {
    minSecondsBetweenQueries: numberEnv('CHATBOT_MIN_SECONDS_BETWEEN_QUERIES', 8),
    windowSeconds: numberEnv('CHATBOT_WINDOW_SECONDS', 600),
    windowLimit: numberEnv('CHATBOT_WINDOW_LIMIT', 8),
    dailyLimit: numberEnv('CHATBOT_DAILY_LIMIT', 40),
    globalDailyLimit: numberEnv('CHATBOT_GLOBAL_DAILY_LIMIT', 250),
    ttlDays: numberEnv('CHATBOT_RATE_LIMIT_TTL_DAYS', 3)
  };
}

function boolEnv(key, fallback = false) {
  const raw = String(process.env[key] || '').trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(raw)) return true;
  if (['0', 'false', 'no', 'off'].includes(raw)) return false;
  return fallback;
}

function isProductionRuntime() {
  return process.env.VERCEL_ENV === 'production';
}

function requiresDdbRateLimit() {
  return boolEnv('CHATBOT_REQUIRE_DDB', isProductionRuntime());
}

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function getRateLimitTable() {
  return pickEnv(['CHATBOT_DDB_TABLE', 'CHATBOT_DDB_TABLE_NAME']);
}

function getRegion() {
  return pickEnv(['CHATBOT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
}

function getAwsCredentialConfig(region) {
  return resolveAwsCredentials({
    service: 'chatbot-ddb',
    region,
    roleArnEnvKeys: ['CHATBOT_DDB_AWS_ROLE_ARN'],
    staticCredentialSets: CHATBOT_STATIC_CREDENTIAL_SETS
  });
}

function getDocClient() {
  const region = getRegion();
  const auth = getAwsCredentialConfig(region);
  const key = `${region}:${auth.cacheKey}`;
  if (cachedDocClient && cachedClientKey === key) return cachedDocClient;

  const client = new DynamoDBClient({ region, credentials: auth.credentials });
  cachedDocClient = DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedClientKey = key;
  return cachedDocClient;
}

function getClientIp(req) {
  const forwarded = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
  const real = String(req.headers['x-real-ip'] || '').trim();
  const socket = req.socket && req.socket.remoteAddress ? String(req.socket.remoteAddress) : '';
  return forwarded || real || socket || 'unknown';
}

function getActorHash(req) {
  const salt = pickEnv(['CHATBOT_HASH_SALT']) || pickEnv(['VERCEL_PROJECT_PRODUCTION_URL', 'VERCEL_URL']) || 'local-chatbot-salt';
  if (!pickEnv(['CHATBOT_HASH_SALT']) && requiresDdbRateLimit()) {
    const err = new Error('CHATBOT_HASH_SALT is not configured');
    err.code = 'CHATBOT_HASH_SALT_MISSING';
    throw err;
  }

  const ip = getClientIp(req);
  return crypto
    .createHmac('sha256', salt)
    .update(ip)
    .digest('hex')
    .slice(0, 32);
}

function todayKey(now = Date.now()) {
  return new Date(now).toISOString().slice(0, 10);
}

function windowKey(now, windowSeconds) {
  return String(Math.floor(now / 1000 / windowSeconds));
}

function ttlSeconds(now, days) {
  return Math.floor(now / 1000) + Math.max(1, days) * 86400;
}

function getMemoryItem(pk, sk) {
  const key = `${pk}|${sk}`;
  const item = memoryStore.get(key) || null;
  if (item && item.ttl <= Math.floor(Date.now() / 1000)) {
    memoryStore.delete(key);
    return null;
  }
  return item;
}

function updateMemoryCount(pk, sk, ttl, now) {
  const key = `${pk}|${sk}`;
  const item = memoryStore.get(key) || { pk, sk, count: 0 };
  item.count = Number(item.count || 0) + 1;
  item.ttl = ttl;
  item.updatedAt = now;
  memoryStore.set(key, item);
  return item;
}

async function getItem(tableName, pk, sk) {
  if (!tableName) return getMemoryItem(pk, sk);
  const result = await getDocClient().send(new GetCommand({
    TableName: tableName,
    Key: { pk, sk },
    ConsistentRead: true
  }));
  return result.Item || null;
}

function countUpdate(tableName, pk, sk, ttl, now, limit) {
  return { Update: {
    TableName: tableName,
    Key: { pk, sk },
    ...(limit ? { ConditionExpression: 'attribute_not_exists(#count) OR #count < :limit' } : {}),
    UpdateExpression: 'SET #ttl = :ttl, #updatedAt = :now ADD #count :one',
    ExpressionAttributeNames: {
      '#ttl': 'ttl',
      '#updatedAt': 'updatedAt',
      '#count': 'count'
    },
    ExpressionAttributeValues: {
      ':ttl': ttl,
      ':now': now,
      ':one': 1,
      ...(limit ? { ':limit': limit } : {})
    },
    ReturnValuesOnConditionCheckFailure: 'ALL_OLD'
  } };
}

function metaUpdate(tableName, actorPk, now, ttl, config, challengePassed) {
  return { Update: {
    TableName: tableName,
    Key: { pk: actorPk, sk: 'META' },
    ...(!challengePassed ? { ConditionExpression: 'attribute_not_exists(#lastQueryAt) OR #lastQueryAt <= :latest' } : {}),
    UpdateExpression: 'SET #lastQueryAt = :now, #ttl = :ttl, #updatedAt = :now',
    ExpressionAttributeNames: {
      '#lastQueryAt': 'lastQueryAt',
      '#ttl': 'ttl',
      '#updatedAt': 'updatedAt'
    },
    ExpressionAttributeValues: {
      ':now': now,
      ':ttl': ttl,
      ...(!challengePassed ? { ':latest': now - config.minSecondsBetweenQueries * 1000 } : {})
    },
    ReturnValuesOnConditionCheckFailure: 'ALL_OLD'
  } };
}

function limitPayload(reason, retryAfter, config, challengeRequired = false) {
  return {
    ok: false,
    error: reason,
    retryAfter,
    challengeRequired,
    limits: {
      minSecondsBetweenQueries: config.minSecondsBetweenQueries,
      windowSeconds: config.windowSeconds,
      windowLimit: config.windowLimit,
      dailyLimit: config.dailyLimit
    }
  };
}

async function checkChatbotRateLimit(req, body = {}, options = {}) {
  const config = getLimitConfig();
  const tableName = getRateLimitTable();
  if (!tableName && requiresDdbRateLimit()) {
    const err = new Error('CHATBOT_DDB_TABLE is not configured');
    err.code = 'CHATBOT_RATE_LIMIT_STORE_MISSING';
    throw err;
  }

  const now = Date.now();
  const actorHash = getActorHash(req);
  const ttl = ttlSeconds(now, config.ttlDays);
  const actorPk = `CHATBOT#ACTOR#${actorHash}`;
  const globalPk = 'CHATBOT#GLOBAL';
  const challengePassed = options.challengePassed === true;

  const meta = await getItem(tableName, actorPk, 'META');
  const lastQueryAt = Number(meta && meta.lastQueryAt) || 0;
  const elapsedSeconds = lastQueryAt ? Math.floor((now - lastQueryAt) / 1000) : Infinity;
  if (!challengePassed && elapsedSeconds < config.minSecondsBetweenQueries) {
    return {
      allowed: false,
      actorHash,
      statusCode: 429,
      payload: limitPayload(
        'Please wait before sending another question.',
        Math.max(1, config.minSecondsBetweenQueries - elapsedSeconds),
        config,
        true
      )
    };
  }

  const currentWindow = windowKey(now, config.windowSeconds);
  const currentDay = todayKey(now);
  const windowSk = `WINDOW#${currentWindow}`;
  const daySk = `DAY#${currentDay}`;
  const denied = (index) => {
    const reasons = [
      ['Please wait before sending another question.', Math.ceil(config.minSecondsBetweenQueries), true],
      ['Too many questions in a short period.', config.windowSeconds, true],
      ['Daily question limit reached.', 86400, false],
      ['The site-wide daily chatbot limit has been reached.', 86400, false]
    ];
    const [reason, retryAfter, challengeRequired] = reasons[index];
    return {
      allowed: false,
      actorHash,
      statusCode: 429,
      payload: limitPayload(reason, retryAfter, config, challengeRequired)
    };
  };

  if (!tableName) {
    // No awaits between admission and reservation: parallel local requests
    // cannot partially spend another actor's global budget.
    const latestQueryAt = Number(getMemoryItem(actorPk, 'META')?.lastQueryAt) || 0;
    if (!challengePassed && latestQueryAt > now - config.minSecondsBetweenQueries * 1000) return denied(0);
    const counts = [getMemoryItem(actorPk, windowSk), getMemoryItem(actorPk, daySk), getMemoryItem(globalPk, daySk)];
    if (!challengePassed && Number(counts[0]?.count || 0) >= config.windowLimit) return denied(1);
    if (Number(counts[1]?.count || 0) >= config.dailyLimit) return denied(2);
    if (Number(counts[2]?.count || 0) >= config.globalDailyLimit) return denied(3);
    updateMemoryCount(actorPk, windowSk, ttl, now);
    updateMemoryCount(actorPk, daySk, ttl, now);
    updateMemoryCount(globalPk, daySk, ttl, now);
    memoryStore.set(`${actorPk}|META`, { pk: actorPk, sk: 'META', lastQueryAt: now, ttl, updatedAt: now });
  } else {
    const command = new TransactWriteCommand({ TransactItems: [
      metaUpdate(tableName, actorPk, now, ttl, config, challengePassed),
      countUpdate(tableName, actorPk, windowSk, ttl, now, challengePassed ? null : config.windowLimit),
      countUpdate(tableName, actorPk, daySk, ttl, now, config.dailyLimit),
      countUpdate(tableName, globalPk, daySk, ttl, now, config.globalDailyLimit)
    ] });
    for (let attempt = 0; ; attempt += 1) {
      try {
        await getDocClient().send(command);
        break;
      } catch (err) {
        const reasons = err?.CancellationReasons || [];
        const rejected = reasons.findIndex((reason) => reason?.Code === 'ConditionalCheckFailed');
        if (rejected >= 0) return denied(rejected);
        const conflict = err?.name === 'TransactionConflictException' || reasons.some((reason) => reason?.Code === 'TransactionConflict');
        if (!conflict || attempt >= 3) throw err;
        await new Promise((resolve) => setTimeout(resolve, 10 * (attempt + 1)));
      }
    }
  }

  const [windowItem, dayItem, globalDayItem] = await Promise.all([
    getItem(tableName, actorPk, windowSk),
    getItem(tableName, actorPk, daySk),
    getItem(tableName, globalPk, daySk)
  ]);
  return {
    allowed: true,
    actorHash,
    config,
    counts: {
      window: Number(windowItem.count || 0),
      daily: Number(dayItem.count || 0),
      globalDaily: Number(globalDayItem.count || 0)
    }
  };
}

module.exports = {
  checkChatbotRateLimit,
  getActorHash,
  getClientIp,
  getLimitConfig,
  getRateLimitTable,
  isProductionRuntime,
  requiresDdbRateLimit,
  _memoryStore: memoryStore,
  _setDocClientForTests(client) { cachedDocClient = client; cachedClientKey = client ? `${getRegion()}:${getAwsCredentialConfig(getRegion()).cacheKey}` : ''; }
};
