'use strict';

const crypto = require('crypto');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  GetCommand,
  UpdateCommand
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
  return boolEnv('CHATBOT_REQUIRE_DDB', false);
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

function getActorHash(req, body = {}) {
  const salt = pickEnv(['CHATBOT_HASH_SALT']) || pickEnv(['VERCEL_PROJECT_PRODUCTION_URL', 'VERCEL_URL']) || 'local-chatbot-salt';
  if (!pickEnv(['CHATBOT_HASH_SALT']) && requiresDdbRateLimit()) {
    const err = new Error('CHATBOT_HASH_SALT is not configured');
    err.code = 'CHATBOT_HASH_SALT_MISSING';
    throw err;
  }

  const ip = getClientIp(req);
  const session = String(body.conversationId || req.headers['x-chatbot-session'] || '').slice(0, 80);
  return crypto
    .createHash('sha256')
    .update(`${salt || 'local-chatbot-salt'}:${ip}:${session}`)
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
  return memoryStore.get(`${pk}|${sk}`) || null;
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
    Key: { pk, sk }
  }));
  return result.Item || null;
}

async function updateCount(tableName, pk, sk, ttl, now) {
  if (!tableName) return updateMemoryCount(pk, sk, ttl, now);
  const result = await getDocClient().send(new UpdateCommand({
    TableName: tableName,
    Key: { pk, sk },
    UpdateExpression: 'SET #ttl = :ttl, #updatedAt = :now ADD #count :one',
    ExpressionAttributeNames: {
      '#ttl': 'ttl',
      '#updatedAt': 'updatedAt',
      '#count': 'count'
    },
    ExpressionAttributeValues: {
      ':ttl': ttl,
      ':now': now,
      ':one': 1
    },
    ReturnValues: 'ALL_NEW'
  }));
  return result.Attributes || { count: 1 };
}

async function setLastQuery(tableName, actorHash, now, ttl) {
  const pk = `CHATBOT#ACTOR#${actorHash}`;
  const sk = 'META';
  if (!tableName) {
    memoryStore.set(`${pk}|${sk}`, { pk, sk, lastQueryAt: now, ttl, updatedAt: now });
    return;
  }
  await getDocClient().send(new UpdateCommand({
    TableName: tableName,
    Key: { pk, sk },
    UpdateExpression: 'SET #lastQueryAt = :now, #ttl = :ttl, #updatedAt = :now',
    ExpressionAttributeNames: {
      '#lastQueryAt': 'lastQueryAt',
      '#ttl': 'ttl',
      '#updatedAt': 'updatedAt'
    },
    ExpressionAttributeValues: {
      ':now': now,
      ':ttl': ttl
    }
  }));
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
  const actorHash = getActorHash(req, body);
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
  const windowItem = await updateCount(tableName, actorPk, `WINDOW#${currentWindow}`, ttl, now);
  const dayItem = await updateCount(tableName, actorPk, `DAY#${currentDay}`, ttl, now);
  const globalDayItem = await updateCount(tableName, globalPk, `DAY#${currentDay}`, ttl, now);

  if (!challengePassed && Number(windowItem.count || 0) > config.windowLimit) {
    return {
      allowed: false,
      actorHash,
      statusCode: 429,
      payload: limitPayload(
        'Too many questions in a short period.',
        config.windowSeconds,
        config,
        true
      )
    };
  }

  if (Number(dayItem.count || 0) > config.dailyLimit) {
    return {
      allowed: false,
      actorHash,
      statusCode: 429,
      payload: limitPayload('Daily question limit reached.', 86400, config, false)
    };
  }

  if (Number(globalDayItem.count || 0) > config.globalDailyLimit) {
    return {
      allowed: false,
      actorHash,
      statusCode: 429,
      payload: limitPayload('The site-wide daily chatbot limit has been reached.', 86400, config, false)
    };
  }

  await setLastQuery(tableName, actorHash, now, ttl);
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
  _memoryStore: memoryStore
};
