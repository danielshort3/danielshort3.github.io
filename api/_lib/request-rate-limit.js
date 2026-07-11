'use strict';

const crypto = require('crypto');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { getAwsClientConfig } = require('./aws-credentials');
const { clientIp } = require('./http-boundary');

const memoryStore = new Map();
const documentClients = new Map();

function positiveNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function booleanValue(value, fallback = false) {
  const normalized = String(value || '').trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
  return fallback;
}

function actorHash(req, options = {}) {
  const salt = String(options.salt || 'local-request-rate-limit').trim();
  return crypto
    .createHash('sha256')
    .update(`${salt}:${clientIp(req)}`)
    .digest('hex')
    .slice(0, 32);
}

function getDocumentClient(options) {
  const region = options.region || 'us-east-2';
  const aws = getAwsClientConfig(options.workload, { region });
  const key = `${options.workload}:${region}:${aws.cacheKey}`;
  if (documentClients.has(key)) return documentClients.get(key);

  const client = DynamoDBDocumentClient.from(new DynamoDBClient(aws.clientConfig), {
    marshallOptions: { removeUndefinedValues: true }
  });
  documentClients.set(key, client);
  return client;
}

function pruneMemory(nowSeconds) {
  if (memoryStore.size < 5000) return;
  for (const [key, value] of memoryStore.entries()) {
    if (Number(value.ttl || 0) <= nowSeconds) memoryStore.delete(key);
  }
}

function updateMemoryCount(pk, sk, ttl, now) {
  const key = `${pk}|${sk}`;
  const current = memoryStore.get(key) || { count: 0 };
  const next = {
    pk,
    sk,
    count: Number(current.count || 0) + 1,
    ttl,
    updatedAt: now
  };
  memoryStore.set(key, next);
  return next;
}

async function updateCount(options, pk, sk, ttl, now) {
  if (!options.tableName) return updateMemoryCount(pk, sk, ttl, now);
  const client = options.documentClient || getDocumentClient(options);
  const result = await client.send(new UpdateCommand({
    TableName: options.tableName,
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

function retryAfterForWindow(nowSeconds, windowSeconds) {
  const remainder = nowSeconds % windowSeconds;
  return Math.max(1, windowSeconds - remainder);
}

async function consumeRequestLimit(req, options = {}) {
  const namespace = String(options.namespace || 'request').replace(/[^A-Za-z0-9#:_-]/g, '').slice(0, 120);
  const windowSeconds = Math.floor(positiveNumber(options.windowSeconds, 600));
  const windowLimit = Math.floor(positiveNumber(options.windowLimit, 30));
  const dailyLimit = Math.floor(positiveNumber(options.dailyLimit, 200));
  const globalDailyLimit = Math.floor(positiveNumber(options.globalDailyLimit, 3000));
  const ttlDays = Math.floor(positiveNumber(options.ttlDays, 3));

  if (!options.tableName && options.requireTable) {
    const err = new Error('Rate-limit table is not configured');
    err.code = 'RATE_LIMIT_TABLE_MISSING';
    throw err;
  }

  const now = Number.isFinite(options.now) ? options.now : Date.now();
  const nowSeconds = Math.floor(now / 1000);
  const ttl = nowSeconds + ttlDays * 86400;
  const actor = actorHash(req, options);
  const date = new Date(now).toISOString().slice(0, 10);
  const window = Math.floor(nowSeconds / windowSeconds);
  const actorPk = `${namespace.toUpperCase()}#ACTOR#${actor}`;
  const globalNamespace = String(options.globalNamespace || namespace)
    .replace(/[^A-Za-z0-9#:_-]/g, '')
    .slice(0, 120);
  const globalPk = `${globalNamespace.toUpperCase()}#GLOBAL`;

  pruneMemory(nowSeconds);
  const [windowItem, dayItem, globalDayItem] = await Promise.all([
    updateCount(options, actorPk, `WINDOW#${window}`, ttl, now),
    updateCount(options, actorPk, `DAY#${date}`, ttl, now),
    updateCount(options, globalPk, `DAY#${date}`, ttl, now)
  ]);

  const counts = {
    window: Number(windowItem.count || 0),
    daily: Number(dayItem.count || 0),
    globalDaily: Number(globalDayItem.count || 0)
  };
  if (counts.window > windowLimit) {
    return {
      allowed: false,
      statusCode: 429,
      retryAfter: retryAfterForWindow(nowSeconds, windowSeconds),
      reason: 'Too many requests in a short period.',
      counts
    };
  }
  if (counts.daily > dailyLimit) {
    return {
      allowed: false,
      statusCode: 429,
      retryAfter: 86400,
      reason: 'Daily request limit reached.',
      counts
    };
  }
  if (counts.globalDaily > globalDailyLimit) {
    return {
      allowed: false,
      statusCode: 429,
      retryAfter: 86400,
      reason: 'The site-wide daily request limit has been reached.',
      counts
    };
  }
  return { allowed: true, counts };
}

module.exports = {
  actorHash,
  booleanValue,
  consumeRequestLimit,
  positiveNumber,
  _memoryStore: memoryStore,
  _private: {
    documentClients,
    getDocumentClient,
    retryAfterForWindow,
    updateMemoryCount
  }
};
