/*
  DynamoDB storage for tools accounts (sessions + activity).

  Env vars required:
  - TOOLS_DDB_TABLE (or TOOLS_DDB_TABLE_NAME)
  - AWS_REGION (or AWS_DEFAULT_REGION)
  - Prefer TOOLS_AWS_ACCESS_KEY_ID / TOOLS_AWS_SECRET_ACCESS_KEY to avoid conflicts
  - Falls back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or any AWS SDK credential provider chain)
  - Optional (temporary creds only): TOOLS_AWS_SESSION_TOKEN / AWS_SESSION_TOKEN
  - Optional lifecycle: TOOLS_DDB_TTL_ATTRIBUTE (default ttl),
    TOOLS_SESSION_RETENTION_DAYS (365), TOOLS_ACTIVITY_RETENTION_DAYS (90)
  - Optional quotas: TOOLS_MAX_SESSIONS_PER_TOOL (50), TOOLS_MAX_SESSIONS_PER_USER (250)
  - Optional purge guard: TOOLS_DELETE_ALL_MAX_ITEMS (10000)
*/
'use strict';

const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DeleteCommand,
  DynamoDBDocumentClient,
  GetCommand,
  QueryCommand,
  TransactWriteCommand,
  UpdateCommand
} = require('@aws-sdk/lib-dynamodb');
const { KNOWN_TOOL_IDS } = require('./tools-api');

const PREFIX = 'USER';
const MAX_TS = 9999999999999;
const MAX_SNAPSHOT_BYTES = 300_000;
const MAX_SUMMARY_CHARS = 2_000;
const MAX_TITLE_CHARS = 120;
const MAX_NOTE_CHARS = 800;
const MAX_TAGS = 12;
const MAX_TAG_CHARS = 24;
const MAX_ACTIVITY_DATA_BYTES = 32_000;
const DEFAULT_MAX_SESSIONS_PER_TOOL = 50;
const DEFAULT_MAX_SESSIONS_PER_USER = 250;
const DEFAULT_SESSION_RETENTION_DAYS = 365;
const DEFAULT_ACTIVITY_RETENTION_DAYS = 90;
const DEFAULT_DELETE_ALL_MAX_ITEMS = 10_000;
const DEFAULT_DELETE_LOCK_SECONDS = 15 * 60;
const MAX_CURSOR_CHARS = 2_048;

function pickEnv(keys){
  for (const key of keys) {
    if (!key) continue;
    if (typeof process.env[key] === 'undefined') continue;
    const raw = String(process.env[key]);
    if (!raw.trim()) continue;
    return { key, raw };
  }
  return { key: '', raw: '' };
}

function integerEnv(name, fallback, min, max){
  const numeric = Number(process.env[name]);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(Math.max(Math.floor(numeric), min), max);
}

function getLifecycleConfig(){
  const ttlAttributeRaw = String(process.env.TOOLS_DDB_TTL_ATTRIBUTE || 'ttl').trim();
  const ttlAttribute = /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(ttlAttributeRaw) ? ttlAttributeRaw : 'ttl';
  return {
    ttlAttribute,
    sessionRetentionDays: integerEnv('TOOLS_SESSION_RETENTION_DAYS', DEFAULT_SESSION_RETENTION_DAYS, 1, 3650),
    activityRetentionDays: integerEnv('TOOLS_ACTIVITY_RETENTION_DAYS', DEFAULT_ACTIVITY_RETENTION_DAYS, 1, 3650),
    maxSessionsPerTool: integerEnv('TOOLS_MAX_SESSIONS_PER_TOOL', DEFAULT_MAX_SESSIONS_PER_TOOL, 1, 500),
    maxSessionsPerUser: integerEnv('TOOLS_MAX_SESSIONS_PER_USER', DEFAULT_MAX_SESSIONS_PER_USER, 1, 5000),
    deleteAllMaxItems: integerEnv('TOOLS_DELETE_ALL_MAX_ITEMS', DEFAULT_DELETE_ALL_MAX_ITEMS, 100, 50_000),
    deleteLockSeconds: integerEnv('TOOLS_DELETE_LOCK_SECONDS', DEFAULT_DELETE_LOCK_SECONDS, 60, 86_400)
  };
}

function expirySeconds(nowMs, days){
  return Math.floor(nowMs / 1000) + (days * 24 * 60 * 60);
}

function withTtl(item, expiresAt){
  const { ttlAttribute } = getLifecycleConfig();
  return { ...item, [ttlAttribute]: expiresAt };
}

function readExpiresAt(item){
  if (!item) return 0;
  const { ttlAttribute } = getLifecycleConfig();
  return Number(item[ttlAttribute]) || 0;
}

function isExpired(item, nowSeconds = Math.floor(Date.now() / 1000)){
  const expiresAt = readExpiresAt(item);
  return expiresAt > 0 && expiresAt <= nowSeconds;
}

function createStoreError(code, message){
  const err = new Error(message);
  err.code = code;
  return err;
}

function currentVersion(item){
  const version = Number(item?.version);
  return Number.isInteger(version) && version > 0 ? version : 1;
}

function normalizeExpectedVersion(value){
  if (typeof value === 'undefined' || value === null || value === '') return null;
  const version = Number(value);
  return Number.isInteger(version) && version >= 0 ? version : null;
}

function assertExpectedVersion(record, expectedVersion){
  const expected = normalizeExpectedVersion(expectedVersion);
  if (expected === null) return;
  const actual = record ? currentVersion(record) : 0;
  if (actual !== expected) {
    throw createStoreError('VERSION_CONFLICT', `Session changed since it was loaded (current version ${actual}).`);
  }
}

function encodeCursor(key){
  if (!key?.pk || !key?.sk) return '';
  return Buffer.from(JSON.stringify({ pk: key.pk, sk: key.sk }), 'utf8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function decodeCursor(value, expectedPk, expectedPrefix){
  const cursor = String(value || '').trim();
  if (!cursor) return undefined;
  if (cursor.length > MAX_CURSOR_CHARS) throw createStoreError('INVALID_CURSOR', 'Invalid pagination cursor.');
  try {
    const padded = cursor.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(cursor.length / 4) * 4, '=');
    const parsed = JSON.parse(Buffer.from(padded, 'base64').toString('utf8'));
    const pk = String(parsed?.pk || '');
    const sk = String(parsed?.sk || '');
    if (pk !== expectedPk || !sk.startsWith(expectedPrefix)) throw new Error('cursor scope mismatch');
    return { pk, sk };
  } catch {
    throw createStoreError('INVALID_CURSOR', 'Invalid pagination cursor.');
  }
}

function attachNextCursor(items, result){
  const nextCursor = encodeCursor(result?.LastEvaluatedKey);
  Object.defineProperty(items, 'nextCursor', {
    configurable: true,
    enumerable: false,
    value: nextCursor
  });
  return items;
}

function getAwsCredentialsFromEnv(){
  const accessKeyIdEnv = pickEnv(['TOOLS_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKeyEnv = pickEnv(['TOOLS_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionTokenEnv = pickEnv(['TOOLS_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);

  const accessKeyId = accessKeyIdEnv.raw.trim();
  const secretAccessKey = secretAccessKeyEnv.raw.trim();
  const sessionToken = sessionTokenEnv.raw.trim();

  if (!accessKeyId || !secretAccessKey) return null;

  const creds = { accessKeyId, secretAccessKey };
  if (sessionToken && accessKeyId.startsWith('ASIA')) creds.sessionToken = sessionToken;
  return creds;
}

function getRequiredEnv(){
  const tableName = pickEnv(['TOOLS_DDB_TABLE', 'TOOLS_DDB_TABLE_NAME']).raw.trim();
  const region =
    (process.env.AWS_REGION ? String(process.env.AWS_REGION).trim() : '') ||
    (process.env.AWS_DEFAULT_REGION ? String(process.env.AWS_DEFAULT_REGION).trim() : '');

  if (!tableName) {
    const err = new Error('TOOLS_DDB_TABLE is not configured');
    err.code = 'DDB_ENV_MISSING';
    throw err;
  }
  if (!region) {
    const err = new Error('AWS_REGION is not configured');
    err.code = 'DDB_ENV_MISSING';
    throw err;
  }
  return { tableName, region };
}

let cachedDocClient = null;
let cachedClientKey = '';

function getDocClient(){
  const { region } = getRequiredEnv();
  const creds = getAwsCredentialsFromEnv();
  const key = `${region}:${creds ? creds.accessKeyId : 'default'}`;
  if (cachedDocClient && cachedClientKey === key) return cachedDocClient;

  const client = new DynamoDBClient({ region, credentials: creds || undefined });
  cachedDocClient = DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedClientKey = key;
  return cachedDocClient;
}

function randomBase64Url(size = 18){
  const crypto = require('crypto');
  return crypto.randomBytes(size).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function byteLength(value){
  try {
    return Buffer.byteLength(String(value), 'utf8');
  } catch {
    return String(value || '').length;
  }
}

function normalizeSummary(value){
  const summary = String(value || '').trim();
  if (!summary) return '';
  return summary.length > MAX_SUMMARY_CHARS ? summary.slice(0, MAX_SUMMARY_CHARS) : summary;
}

function normalizeTitle(value){
  const title = String(value || '').trim();
  if (!title) return '';
  return title.length > MAX_TITLE_CHARS ? title.slice(0, MAX_TITLE_CHARS).trimEnd() : title;
}

function normalizeNote(value){
  const note = String(value || '').trim();
  if (!note) return '';
  return note.length > MAX_NOTE_CHARS ? note.slice(0, MAX_NOTE_CHARS).trimEnd() : note;
}

function normalizeTags(value){
  const raw = Array.isArray(value)
    ? value.map(v => String(v || '').trim())
    : String(value || '')
      .split(/[,\n]/g)
      .map(v => String(v || '').trim());

  const seen = new Set();
  const tags = [];
  raw.forEach((tag) => {
    const cleaned = tag.replace(/\s+/g, ' ').trim();
    if (!cleaned) return;
    const clipped = cleaned.length > MAX_TAG_CHARS ? cleaned.slice(0, MAX_TAG_CHARS).trimEnd() : cleaned;
    const key = clipped.toLowerCase();
    if (!key) return;
    if (seen.has(key)) return;
    seen.add(key);
    tags.push(clipped);
  });

  return tags.slice(0, MAX_TAGS);
}

function ensureSnapshotOk(snapshot){
  const raw = JSON.stringify(snapshot || {});
  if (byteLength(raw) > MAX_SNAPSHOT_BYTES) {
    const err = new Error(`Snapshot too large (max ${MAX_SNAPSHOT_BYTES} bytes).`);
    err.code = 'SNAPSHOT_TOO_LARGE';
    throw err;
  }
}

function ensureActivityDataOk(data){
  if (!data || typeof data !== 'object') return;
  const raw = JSON.stringify(data);
  if (byteLength(raw) > MAX_ACTIVITY_DATA_BYTES) {
    throw createStoreError('ACTIVITY_TOO_LARGE', `Activity data is too large (max ${MAX_ACTIVITY_DATA_BYTES} bytes).`);
  }
}

function clampTimestampMs(value){
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return Date.now();
  return Math.min(Math.floor(numeric), MAX_TS);
}

function reverseTimestampKey(ts){
  const safeTs = clampTimestampMs(ts);
  const rev = Math.max(0, MAX_TS - safeTs);
  return String(rev).padStart(13, '0');
}

function pkUser(sub){
  return `${PREFIX}#${sub}`;
}

function pkTool(sub, toolId){
  return `${PREFIX}#${sub}#TOOL#${toolId}`;
}

function pkUserSessions(sub){
  return `${PREFIX}#${sub}#SESSIONS`;
}

function pkToolSessions(sub, toolId){
  return `${PREFIX}#${sub}#TOOLSESSIONS#${toolId}`;
}

function pkUserActivity(sub){
  return `${PREFIX}#${sub}#ACTIVITY`;
}

function pkToolActivity(sub, toolId){
  return `${PREFIX}#${sub}#ACTIVITY#${toolId}`;
}

function skToolMeta(toolId){
  return `TOOL#${toolId}`;
}

function skUserMeta(){
  return 'META';
}

function userDeletionGuard(tableName, sub){
  const { deleteLockSeconds } = getLifecycleConfig();
  return {
    ConditionCheck: {
      TableName: tableName,
      Key: { pk: pkUser(sub), sk: skUserMeta() },
      ConditionExpression: 'attribute_not_exists(deletingAt) OR deletingAt < :staleBefore',
      ExpressionAttributeValues: {
        ':staleBefore': Date.now() - (deleteLockSeconds * 1000)
      }
    }
  };
}

function skSession(sessionId){
  return `SESSION#${sessionId}`;
}

function skUserSessionIndex(revTsKey, toolId, sessionId){
  return `UPDATED#${revTsKey}#${toolId}#${sessionId}`;
}

function skToolSessionIndex(revTsKey, sessionId){
  return `UPDATED#${revTsKey}#${sessionId}`;
}

function skUserActivityEvent(revTsKey, toolId, eventId){
  return `TS#${revTsKey}#${toolId}#${eventId}`;
}

function skToolActivityEvent(revTsKey, eventId){
  return `TS#${revTsKey}#${eventId}`;
}

function toSessionMeta(item){
  if (!item) return null;
  const toolId = String(item.toolId || '').trim();
  const sessionId = String(item.sessionId || '').trim();
  if (!toolId || !sessionId) return null;
  return {
    toolId,
    sessionId,
    createdAt: Number(item.createdAt) || 0,
    updatedAt: Number(item.updatedAt) || 0,
    outputSummary: String(item.outputSummary || '').trim(),
    title: String(item.title || '').trim(),
    note: String(item.note || '').trim(),
    tags: Array.isArray(item.tags) ? item.tags.map(v => String(v || '').trim()).filter(Boolean) : [],
    pinned: Boolean(item.pinned),
    version: currentVersion(item),
    expiresAt: readExpiresAt(item)
  };
}

async function getSession({ sub, toolId, sessionId }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
  }));
  const item = result && result.Item ? result.Item : null;
  if (!item || isExpired(item)) return null;
  return {
    toolId: String(item.toolId || toolId).trim(),
    sessionId: String(item.sessionId || sessionId).trim(),
    createdAt: Number(item.createdAt) || 0,
    updatedAt: Number(item.updatedAt) || 0,
    outputSummary: String(item.outputSummary || '').trim(),
    title: String(item.title || '').trim(),
    note: String(item.note || '').trim(),
    tags: Array.isArray(item.tags) ? item.tags.map(v => String(v || '').trim()).filter(Boolean) : [],
    pinned: Boolean(item.pinned),
    version: currentVersion(item),
    expiresAt: readExpiresAt(item),
    snapshot: item.snapshot && typeof item.snapshot === 'object' ? item.snapshot : {}
  };
}

async function countPartitionItems({ pk, prefix }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const { ttlAttribute } = getLifecycleConfig();
  const nowSeconds = Math.floor(Date.now() / 1000);
  let count = 0;
  let startKey;
  do {
    const result = await client.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
      ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk', '#ttl': ttlAttribute },
      ExpressionAttributeValues: { ':pk': pk, ':prefix': prefix, ':now': nowSeconds },
      FilterExpression: 'attribute_not_exists(#ttl) OR #ttl > :now',
      ExclusiveStartKey: startKey,
      ConsistentRead: true,
      Select: 'COUNT'
    }));
    count += Number(result?.Count) || 0;
    startKey = result?.LastEvaluatedKey || undefined;
  } while (startKey);
  return count;
}

async function reconcileCounter({ key, attribute, observed, entityType, toolId }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const result = await client.send(new GetCommand({ TableName: tableName, Key: key }));
  const record = result?.Item || null;
  const hasCount = Number.isFinite(Number(record?.[attribute]));
  const priorCount = hasCount ? Number(record[attribute]) : 0;
  if (hasCount && priorCount === observed) return;

  const names = { '#count': attribute, '#entityType': 'entityType' };
  const values = { ':observed': observed, ':entityType': entityType };
  let condition = 'attribute_not_exists(#count)';
  if (hasCount) {
    condition = '#count = :priorCount';
    values[':priorCount'] = priorCount;
  }
  let updateExpression = 'SET #count = :observed, #entityType = if_not_exists(#entityType, :entityType)';
  if (toolId) {
    names['#toolId'] = 'toolId';
    values[':toolId'] = toolId;
    updateExpression += ', #toolId = :toolId';
  }

  try {
    await client.send(new UpdateCommand({
      TableName: tableName,
      Key: key,
      ConditionExpression: condition,
      UpdateExpression: updateExpression,
      ExpressionAttributeNames: names,
      ExpressionAttributeValues: values
    }));
  } catch (err) {
    if (err?.name !== 'ConditionalCheckFailedException') throw err;
  }
}

async function prepareNewSessionQuota({ sub, toolId }){
  const config = getLifecycleConfig();
  const [toolCount, userCount] = await Promise.all([
    countPartitionItems({ pk: pkToolSessions(sub, toolId), prefix: 'UPDATED#' }),
    countPartitionItems({ pk: pkUserSessions(sub), prefix: 'UPDATED#' })
  ]);

  if (toolCount >= config.maxSessionsPerTool) {
    throw createStoreError('SESSION_QUOTA_EXCEEDED', `This tool has reached its ${config.maxSessionsPerTool}-session account limit.`);
  }
  if (userCount >= config.maxSessionsPerUser) {
    throw createStoreError('SESSION_QUOTA_EXCEEDED', `This account has reached its ${config.maxSessionsPerUser}-session limit.`);
  }

  await Promise.all([
    reconcileCounter({
      key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
      attribute: 'sessionCount',
      observed: toolCount,
      entityType: 'tool_meta',
      toolId
    }),
    reconcileCounter({
      key: { pk: pkUser(sub), sk: skUserMeta() },
      attribute: 'sessionCount',
      observed: userCount,
      entityType: 'user_meta'
    })
  ]);

  return config;
}

async function saveSession({ sub, toolId, sessionId, snapshot, outputSummary, expectedVersion }){
  ensureSnapshotOk(snapshot);
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const now = Date.now();
  const nextSessionId = sessionId || randomBase64Url(18);

  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { pk: pkTool(sub, toolId), sk: skSession(nextSessionId) }
  }));
  const existing = result && result.Item ? result.Item : null;
  if (existing && isExpired(existing)) {
    throw createStoreError('SESSION_EXPIRED', 'This saved session has expired. Start a new session.');
  }
  assertExpectedVersion(existing, expectedVersion);

  const lifecycle = existing ? getLifecycleConfig() : await prepareNewSessionQuota({ sub, toolId });
  const expiresAt = expirySeconds(now, lifecycle.sessionRetentionDays);
  const version = existing ? currentVersion(existing) + 1 : 1;

  const createdAt = Number(existing?.createdAt) || now;
  const priorToolIndexSk = String(existing?.toolIndexSk || '').trim();
  const priorUserIndexSk = String(existing?.userIndexSk || '').trim();

  const title = normalizeTitle(existing?.title);
  const note = normalizeNote(existing?.note);
  const tags = normalizeTags(existing?.tags);
  const pinned = Boolean(existing?.pinned);

  const revTsKey = reverseTimestampKey(now);
  const nextToolIndexSk = skToolSessionIndex(revTsKey, nextSessionId);
  const nextUserIndexSk = skUserSessionIndex(revTsKey, toolId, nextSessionId);

  const summary = normalizeSummary(outputSummary);

  const sessionItem = withTtl({
    pk: pkTool(sub, toolId),
    sk: skSession(nextSessionId),
    entityType: 'session',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary,
    title: title || undefined,
    note: note || undefined,
    tags: tags.length ? tags : undefined,
    pinned,
    version,
    snapshot: snapshot || {},
    toolIndexSk: nextToolIndexSk,
    userIndexSk: nextUserIndexSk
  }, expiresAt);

  const toolIndexItem = withTtl({
    pk: pkToolSessions(sub, toolId),
    sk: nextToolIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary,
    title: title || undefined,
    note: note || undefined,
    tags: tags.length ? tags : undefined,
    pinned,
    version
  }, expiresAt);

  const userIndexItem = withTtl({
    pk: pkUserSessions(sub),
    sk: nextUserIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary,
    title: title || undefined,
    note: note || undefined,
    tags: tags.length ? tags : undefined,
    pinned,
    version
  }, expiresAt);

  const isNewSession = !existing;
  const sessionPut = {
    TableName: tableName,
    Item: sessionItem,
    ConditionExpression: existing
      ? 'attribute_exists(sessionId) AND (attribute_not_exists(#version) OR #version = :currentVersion)'
      : 'attribute_not_exists(sessionId)'
  };
  if (existing) {
    sessionPut.ExpressionAttributeNames = { '#version': 'version' };
    sessionPut.ExpressionAttributeValues = { ':currentVersion': currentVersion(existing) };
  }

  const transactItems = [
    { Put: sessionPut },
    { Put: { TableName: tableName, Item: toolIndexItem } },
    { Put: { TableName: tableName, Item: userIndexItem } }
  ];

  if (isNewSession) {
    transactItems.push(
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
          ConditionExpression: 'attribute_exists(sessionCount) AND sessionCount < :limit',
          UpdateExpression: 'SET toolId = :toolId, firstUsedAt = if_not_exists(firstUsedAt, :now), lastUsedAt = :now ADD sessionCount :one',
          ExpressionAttributeValues: {
            ':toolId': toolId,
            ':now': now,
            ':one': 1,
            ':limit': lifecycle.maxSessionsPerTool
          }
        }
      },
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skUserMeta() },
          ConditionExpression: 'attribute_exists(sessionCount) AND sessionCount < :limit AND (attribute_not_exists(deletingAt) OR deletingAt < :staleBefore)',
          UpdateExpression: 'SET entityType = if_not_exists(entityType, :entityType), lastUsedAt = :now ADD sessionCount :one',
          ExpressionAttributeValues: {
            ':entityType': 'user_meta',
            ':now': now,
            ':one': 1,
            ':limit': lifecycle.maxSessionsPerUser,
            ':staleBefore': now - (lifecycle.deleteLockSeconds * 1000)
          }
        }
      }
    );
  } else {
    transactItems.push(
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
          UpdateExpression: 'SET toolId = :toolId, firstUsedAt = if_not_exists(firstUsedAt, :now), lastUsedAt = :now',
          ExpressionAttributeValues: { ':toolId': toolId, ':now': now }
        }
      },
      userDeletionGuard(tableName, sub)
    );
  }

  if (priorToolIndexSk && priorToolIndexSk !== nextToolIndexSk) {
    transactItems.push({
      Delete: {
        TableName: tableName,
        Key: { pk: pkToolSessions(sub, toolId), sk: priorToolIndexSk }
      }
    });
  }

  if (priorUserIndexSk && priorUserIndexSk !== nextUserIndexSk) {
    transactItems.push({
      Delete: {
        TableName: tableName,
        Key: { pk: pkUserSessions(sub), sk: priorUserIndexSk }
      }
    });
  }

  try {
    await client.send(new TransactWriteCommand({ TransactItems: transactItems }));
  } catch (err) {
    if (err?.name === 'TransactionCanceledException') {
      throw createStoreError('WRITE_CONFLICT', 'Session changed or an account session limit was reached. Reload and try again.');
    }
    throw err;
  }

  return {
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary,
    version,
    expiresAt
  };
}

async function deleteSession({ sub, toolId, sessionId, expectedVersion }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
  }));
  const record = result && result.Item ? result.Item : null;
  if (!record) return false;
  assertExpectedVersion(record, expectedVersion);
  const lifecycle = getLifecycleConfig();

  const toolIndexSk = String(record?.toolIndexSk || '').trim();
  const userIndexSk = String(record?.userIndexSk || '').trim();

  const buildDeleteItems = () => {
    const items = [
      {
        Delete: {
          TableName: tableName,
          Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) },
          ConditionExpression: 'attribute_exists(sessionId) AND (attribute_not_exists(#version) OR #version = :currentVersion)',
          ExpressionAttributeNames: { '#version': 'version' },
          ExpressionAttributeValues: { ':currentVersion': currentVersion(record) }
        }
      },
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
          ConditionExpression: 'attribute_not_exists(#sessionCount) OR #sessionCount > :zero',
          UpdateExpression: 'SET #sessionCount = if_not_exists(#sessionCount, :one) - :one',
          ExpressionAttributeNames: { '#sessionCount': 'sessionCount' },
          ExpressionAttributeValues: { ':one': 1, ':zero': 0 }
        }
      },
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skUserMeta() },
          ConditionExpression: '(attribute_not_exists(#sessionCount) OR #sessionCount > :zero) AND (attribute_not_exists(deletingAt) OR deletingAt < :staleBefore)',
          UpdateExpression: 'SET #sessionCount = if_not_exists(#sessionCount, :one) - :one',
          ExpressionAttributeNames: { '#sessionCount': 'sessionCount' },
          ExpressionAttributeValues: {
            ':one': 1,
            ':zero': 0,
            ':staleBefore': Date.now() - (lifecycle.deleteLockSeconds * 1000)
          }
        }
      }
    ];

    if (toolIndexSk) {
      items.push({
        Delete: {
          TableName: tableName,
          Key: { pk: pkToolSessions(sub, toolId), sk: toolIndexSk }
        }
      });
    }

    if (userIndexSk) {
      items.push({
        Delete: {
          TableName: tableName,
          Key: { pk: pkUserSessions(sub), sk: userIndexSk }
        }
      });
    }
    return items;
  };

  const transactDelete = () => client.send(new TransactWriteCommand({ TransactItems: buildDeleteItems() }));
  try {
    await transactDelete();
  } catch (err) {
    if (err?.name !== 'TransactionCanceledException') throw err;

    const latest = await client.send(new GetCommand({
      TableName: tableName,
      Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
    }));
    if (!latest?.Item) return true;
    if (currentVersion(latest.Item) !== currentVersion(record)) {
      throw createStoreError('WRITE_CONFLICT', 'Session changed before it could be deleted. Reload and try again.');
    }

    const [toolCount, userCount] = await Promise.all([
      countPartitionItems({ pk: pkToolSessions(sub, toolId), prefix: 'UPDATED#' }),
      countPartitionItems({ pk: pkUserSessions(sub), prefix: 'UPDATED#' })
    ]);
    await Promise.all([
      reconcileCounter({
        key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
        attribute: 'sessionCount',
        observed: toolCount,
        entityType: 'tool_meta',
        toolId
      }),
      reconcileCounter({
        key: { pk: pkUser(sub), sk: skUserMeta() },
        attribute: 'sessionCount',
        observed: userCount,
        entityType: 'user_meta'
      })
    ]);
    try {
      await transactDelete();
    } catch (retryErr) {
      if (retryErr?.name === 'TransactionCanceledException') {
        throw createStoreError('WRITE_CONFLICT', 'Session changed or account deletion is in progress. Reload and try again.');
      }
      throw retryErr;
    }
  }

  return true;
}

async function updateSessionMeta({ sub, toolId, sessionId, title, note, tags, pinned, expectedVersion }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
  }));
  const record = result && result.Item ? result.Item : null;
  if (!record || isExpired(record)) return null;
  assertExpectedVersion(record, expectedVersion);
  const version = currentVersion(record) + 1;
  const expiresAt = readExpiresAt(record);

  const nextTitle = (typeof title !== 'undefined') ? normalizeTitle(title) : normalizeTitle(record.title);
  const nextNote = (typeof note !== 'undefined') ? normalizeNote(note) : normalizeNote(record.note);
  const nextTags = (typeof tags !== 'undefined') ? normalizeTags(tags) : normalizeTags(record.tags);
  const nextPinned = (typeof pinned !== 'undefined') ? Boolean(pinned) : Boolean(record.pinned);

  const toolIndexSk = String(record.toolIndexSk || '').trim() || skToolSessionIndex(reverseTimestampKey(record.updatedAt), sessionId);
  const userIndexSk = String(record.userIndexSk || '').trim() || skUserSessionIndex(reverseTimestampKey(record.updatedAt), toolId, sessionId);

  const setParts = ['#pinned = :pinned', '#version = :nextVersion'];
  const removeParts = [];
  const names = {
    '#title': 'title',
    '#note': 'note',
    '#tags': 'tags',
    '#pinned': 'pinned',
    '#version': 'version'
  };
  const values = {
    ':pinned': nextPinned,
    ':nextVersion': version,
    ':currentVersion': currentVersion(record)
  };

  if (nextTitle) {
    setParts.push('#title = :title');
    values[':title'] = nextTitle;
  } else {
    removeParts.push('#title');
  }

  if (nextNote) {
    setParts.push('#note = :note');
    values[':note'] = nextNote;
  } else {
    removeParts.push('#note');
  }

  if (nextTags.length) {
    setParts.push('#tags = :tags');
    values[':tags'] = nextTags;
  } else {
    removeParts.push('#tags');
  }

  const expressionParts = [];
  if (setParts.length) expressionParts.push(`SET ${setParts.join(', ')}`);
  if (removeParts.length) expressionParts.push(`REMOVE ${removeParts.join(', ')}`);

  const updateExpression = expressionParts.join(' ');

  const createdAt = Number(record.createdAt) || 0;
  const updatedAt = Number(record.updatedAt) || 0;
  const outputSummary = String(record.outputSummary || '').trim();

  const toolIndexItemBase = {
    pk: pkToolSessions(sub, toolId),
    sk: toolIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId,
    createdAt,
    updatedAt,
    outputSummary,
    title: nextTitle || undefined,
    note: nextNote || undefined,
    tags: nextTags.length ? nextTags : undefined,
    pinned: nextPinned,
    version
  };
  const toolIndexItem = expiresAt ? withTtl(toolIndexItemBase, expiresAt) : toolIndexItemBase;

  const userIndexItemBase = {
    pk: pkUserSessions(sub),
    sk: userIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId,
    createdAt,
    updatedAt,
    outputSummary,
    title: nextTitle || undefined,
    note: nextNote || undefined,
    tags: nextTags.length ? nextTags : undefined,
    pinned: nextPinned,
    version
  };
  const userIndexItem = expiresAt ? withTtl(userIndexItemBase, expiresAt) : userIndexItemBase;

  try {
    await client.send(new TransactWriteCommand({
      TransactItems: [
        {
          Update: {
            TableName: tableName,
            Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) },
            ConditionExpression: 'attribute_exists(sessionId) AND (attribute_not_exists(#version) OR #version = :currentVersion)',
            UpdateExpression: updateExpression,
            ExpressionAttributeNames: names,
            ExpressionAttributeValues: values
          }
        },
        { Put: { TableName: tableName, Item: toolIndexItem } },
        { Put: { TableName: tableName, Item: userIndexItem } },
        userDeletionGuard(tableName, sub)
      ]
    }));
  } catch (err) {
    if (err?.name === 'TransactionCanceledException') {
      throw createStoreError('WRITE_CONFLICT', 'Session changed before its details could be saved. Reload and try again.');
    }
    throw err;
  }

  return {
    toolId,
    sessionId,
    createdAt,
    updatedAt,
    outputSummary,
    title: nextTitle,
    note: nextNote,
    tags: nextTags,
    pinned: nextPinned,
    version,
    expiresAt
  };
}

async function listSessions({ sub, toolId, limit, cursor }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Math.min(Math.max(Number(limit) || 20, 1), 50);

  const pk = pkToolSessions(sub, toolId);
  const result = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': pk,
      ':prefix': 'UPDATED#'
    },
    ScanIndexForward: true,
    Limit: safeLimit,
    ExclusiveStartKey: decodeCursor(cursor, pk, 'UPDATED#')
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  return attachNextCursor(items.filter(item => !isExpired(item)).map(toSessionMeta).filter(Boolean), result);
}

async function listRecentSessions(sub, limit, cursor){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Math.min(Math.max(Number(limit) || 25, 1), 100);

  const pk = pkUserSessions(sub);
  const result = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': pk,
      ':prefix': 'UPDATED#'
    },
    ScanIndexForward: true,
    Limit: safeLimit,
    ExclusiveStartKey: decodeCursor(cursor, pk, 'UPDATED#')
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  return attachNextCursor(items.filter(item => !isExpired(item)).map(toSessionMeta).filter(Boolean), result);
}

async function logActivity({ sub, toolId, type, summary, data }){
  ensureActivityDataOk(data);
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const now = Date.now();
  const { activityRetentionDays } = getLifecycleConfig();
  const expiresAt = expirySeconds(now, activityRetentionDays);
  const event = {
    id: randomBase64Url(18),
    toolId,
    type: String(type || '').trim(),
    ts: now,
    summary: normalizeSummary(summary),
    data: data && typeof data === 'object' ? data : undefined,
    expiresAt
  };

  const revTsKey = reverseTimestampKey(now);
  await client.send(new TransactWriteCommand({
    TransactItems: [
      {
        Put: {
          TableName: tableName,
          Item: withTtl({
            pk: pkUserActivity(sub),
            sk: skUserActivityEvent(revTsKey, toolId, event.id),
            entityType: 'activity',
            ...event
          }, expiresAt)
        }
      },
      {
        Put: {
          TableName: tableName,
          Item: withTtl({
            pk: pkToolActivity(sub, toolId),
            sk: skToolActivityEvent(revTsKey, event.id),
            entityType: 'activity',
            ...event
          }, expiresAt)
        }
      },
      {
        Update: {
          TableName: tableName,
          Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
          UpdateExpression: 'SET toolId = :toolId, firstUsedAt = if_not_exists(firstUsedAt, :now), lastUsedAt = :now ADD activityCount :one',
          ExpressionAttributeValues: {
            ':toolId': toolId,
            ':now': now,
            ':one': 1
          }
        }
      },
      userDeletionGuard(tableName, sub)
    ]
  }));

  return event;
}

async function listActivity({ sub, toolId, limit, cursor }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Math.min(Math.max(Number(limit) || 50, 1), 200);

  const pk = toolId ? pkToolActivity(sub, toolId) : pkUserActivity(sub);
  const result = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': pk,
      ':prefix': 'TS#'
    },
    ScanIndexForward: true,
    Limit: safeLimit,
    ExclusiveStartKey: decodeCursor(cursor, pk, 'TS#')
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  const events = items
    .filter(item => !isExpired(item))
    .map(item => ({
      id: String(item.id || '').trim(),
      toolId: String(item.toolId || '').trim(),
      type: String(item.type || '').trim(),
      ts: Number(item.ts) || 0,
      summary: String(item.summary || '').trim(),
      data: item.data && typeof item.data === 'object' ? item.data : undefined,
      expiresAt: readExpiresAt(item)
    }))
    .filter(evt => evt.toolId && evt.ts);
  return attachNextCursor(events, result);
}

async function listUserTools(sub){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const tools = [];
  let startKey;
  do {
    const result = await client.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
      ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
      ExpressionAttributeValues: {
        ':pk': pkUser(sub),
        ':prefix': 'TOOL#'
      },
      ExclusiveStartKey: startKey
    }));
    const items = (result && Array.isArray(result.Items)) ? result.Items : [];
    items.forEach((item) => {
      const toolId = String(item.toolId || '').trim();
      if (toolId) tools.push(toolId);
    });
    startKey = result && result.LastEvaluatedKey ? result.LastEvaluatedKey : undefined;
  } while (startKey);

  return [...new Set(tools)].sort();
}

async function getToolMeta(sub, toolId){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { pk: pkUser(sub), sk: skToolMeta(toolId) }
  }));
  const item = result && result.Item ? result.Item : null;
  if (!item) return null;
  return {
    toolId: String(item.toolId || toolId).trim(),
    firstUsedAt: Number(item.firstUsedAt) || 0,
    lastUsedAt: Number(item.lastUsedAt) || 0,
    sessionCount: Number(item.sessionCount) || 0,
    activityCount: Number(item.activityCount) || 0
  };
}

async function queryPartitionRecords(pk, maxItems){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const records = [];
  let startKey;
  do {
    const result = await client.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: '#pk = :pk',
      ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk', '#toolId': 'toolId' },
      ExpressionAttributeValues: { ':pk': pk },
      ProjectionExpression: '#pk, #sk, #toolId',
      ExclusiveStartKey: startKey,
      ConsistentRead: true
    }));
    const items = Array.isArray(result?.Items) ? result.Items : [];
    records.push(...items);
    if (records.length > maxItems) {
      throw createStoreError('DELETE_TOO_LARGE', `Account data exceeds the ${maxItems}-item delete limit. Contact support for an assisted purge.`);
    }
    startKey = result?.LastEvaluatedKey || undefined;
  } while (startKey);
  return records;
}

async function deleteKeysInBatches(keys, markerKey, deletionStartedAt){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const commands = [];
  for (let offset = 0; offset < keys.length; offset += 99) {
    const batch = keys.slice(offset, offset + 99);
    commands.push(() => client.send(new TransactWriteCommand({
      TransactItems: [
        {
          ConditionCheck: {
            TableName: tableName,
            Key: markerKey,
            ConditionExpression: 'deletingAt = :deletingAt',
            ExpressionAttributeValues: { ':deletingAt': deletionStartedAt }
          }
        },
        ...batch.map(Key => ({ Delete: { TableName: tableName, Key } }))
      ]
    })));
  }
  for (let offset = 0; offset < commands.length; offset += 4) {
    await Promise.all(commands.slice(offset, offset + 4).map(run => run()));
  }
}

async function deleteAllUserData({ sub }){
  const { deleteAllMaxItems, deleteLockSeconds } = getLifecycleConfig();
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const deletionStartedAt = Date.now();
  const markerKey = { pk: pkUser(sub), sk: skUserMeta() };
  try {
    await client.send(new UpdateCommand({
      TableName: tableName,
      Key: markerKey,
      ConditionExpression: 'attribute_not_exists(deletingAt) OR deletingAt < :staleBefore',
      UpdateExpression: 'SET entityType = if_not_exists(entityType, :entityType), deletingAt = :deletingAt',
      ExpressionAttributeValues: {
        ':entityType': 'user_meta',
        ':deletingAt': deletionStartedAt,
        ':staleBefore': deletionStartedAt - (deleteLockSeconds * 1000)
      }
    }));
  } catch (err) {
    if (err?.name === 'ConditionalCheckFailedException') {
      throw createStoreError('DELETE_IN_PROGRESS', 'Account data deletion is already in progress.');
    }
    throw err;
  }

  let completed = false;
  try {
    const rootPartitions = [pkUser(sub), pkUserSessions(sub), pkUserActivity(sub)];
    const rootResults = await Promise.all(rootPartitions.map(pk => queryPartitionRecords(pk, deleteAllMaxItems)));
    const records = rootResults.flat();
    const toolIds = new Set(KNOWN_TOOL_IDS);
    records.forEach((item) => {
      const toolId = String(item?.toolId || '').trim();
      if (toolId) toolIds.add(toolId);
      const sk = String(item?.sk || '');
      const metaMatch = /^TOOL#([a-z0-9][a-z0-9-]*)$/.exec(sk);
      if (metaMatch) toolIds.add(metaMatch[1]);
    });

    for (const toolId of toolIds) {
      const partitions = [pkTool(sub, toolId), pkToolSessions(sub, toolId), pkToolActivity(sub, toolId)];
      for (const pk of partitions) {
        const remaining = deleteAllMaxItems - records.length;
        if (remaining <= 0) {
          throw createStoreError('DELETE_TOO_LARGE', `Account data exceeds the ${deleteAllMaxItems}-item delete limit. Contact support for an assisted purge.`);
        }
        records.push(...await queryPartitionRecords(pk, remaining));
      }
    }

    const unique = new Map();
    records.forEach((item) => {
      const pk = String(item?.pk || '');
      const sk = String(item?.sk || '');
      if (!pk || !sk) return;
      unique.set(`${pk}\u0000${sk}`, { pk, sk });
    });
    const markerId = `${markerKey.pk}\u0000${markerKey.sk}`;
    unique.delete(markerId);
    const keys = [...unique.values()];
    if (keys.length + 1 > deleteAllMaxItems) {
      throw createStoreError('DELETE_TOO_LARGE', `Account data exceeds the ${deleteAllMaxItems}-item delete limit. Contact support for an assisted purge.`);
    }

    await deleteKeysInBatches(keys, markerKey, deletionStartedAt);
    await client.send(new DeleteCommand({
      TableName: tableName,
      Key: markerKey,
      ConditionExpression: 'deletingAt = :deletingAt',
      ExpressionAttributeValues: { ':deletingAt': deletionStartedAt }
    }));
    completed = true;
    return { deletedCount: keys.length + 1, toolCount: toolIds.size };
  } finally {
    if (!completed) {
      try {
        await client.send(new UpdateCommand({
          TableName: tableName,
          Key: markerKey,
          ConditionExpression: 'deletingAt = :deletingAt',
          UpdateExpression: 'REMOVE deletingAt',
          ExpressionAttributeValues: { ':deletingAt': deletionStartedAt }
        }));
      } catch {}
    }
  }
}

module.exports = {
  MAX_SNAPSHOT_BYTES,
  DEFAULT_MAX_SESSIONS_PER_TOOL,
  DEFAULT_MAX_SESSIONS_PER_USER,
  saveSession,
  listSessions,
  getSession,
  deleteSession,
  updateSessionMeta,
  logActivity,
  listActivity,
  deleteAllUserData,
  listUserTools,
  getToolMeta,
  listRecentSessions
};
