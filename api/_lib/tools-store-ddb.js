/*
  DynamoDB storage for tools accounts (sessions + activity).

  Env vars required:
  - TOOLS_DDB_TABLE (or TOOLS_DDB_TABLE_NAME)
  - AWS_REGION (or AWS_DEFAULT_REGION)
  - Prefer TOOLS_AWS_ACCESS_KEY_ID / TOOLS_AWS_SECRET_ACCESS_KEY to avoid conflicts
  - Falls back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or any AWS SDK credential provider chain)
  - Optional (temporary creds only): TOOLS_AWS_SESSION_TOKEN / AWS_SESSION_TOKEN
*/
'use strict';

const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  GetCommand,
  QueryCommand,
  TransactWriteCommand,
  UpdateCommand
} = require('@aws-sdk/lib-dynamodb');

const PREFIX = 'USER';
const MAX_TS = 9999999999999;
const MAX_SNAPSHOT_BYTES = 300_000;
const MAX_SUMMARY_CHARS = 2_000;

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

function ensureSnapshotOk(snapshot){
  const raw = JSON.stringify(snapshot || {});
  if (byteLength(raw) > MAX_SNAPSHOT_BYTES) {
    const err = new Error(`Snapshot too large (max ${MAX_SNAPSHOT_BYTES} bytes).`);
    err.code = 'SNAPSHOT_TOO_LARGE';
    throw err;
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
    outputSummary: String(item.outputSummary || '').trim()
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
  if (!item) return null;
  return {
    toolId: String(item.toolId || toolId).trim(),
    sessionId: String(item.sessionId || sessionId).trim(),
    createdAt: Number(item.createdAt) || 0,
    updatedAt: Number(item.updatedAt) || 0,
    outputSummary: String(item.outputSummary || '').trim(),
    snapshot: item.snapshot && typeof item.snapshot === 'object' ? item.snapshot : {}
  };
}

async function saveSession({ sub, toolId, sessionId, snapshot, outputSummary }){
  ensureSnapshotOk(snapshot);
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const now = Date.now();
  const nextSessionId = sessionId || randomBase64Url(18);

  let existing = null;
  try {
    const result = await client.send(new GetCommand({
      TableName: tableName,
      Key: { pk: pkTool(sub, toolId), sk: skSession(nextSessionId) }
    }));
    existing = result && result.Item ? result.Item : null;
  } catch {}

  const createdAt = Number(existing?.createdAt) || now;
  const priorToolIndexSk = String(existing?.toolIndexSk || '').trim();
  const priorUserIndexSk = String(existing?.userIndexSk || '').trim();

  const revTsKey = reverseTimestampKey(now);
  const nextToolIndexSk = skToolSessionIndex(revTsKey, nextSessionId);
  const nextUserIndexSk = skUserSessionIndex(revTsKey, toolId, nextSessionId);

  const summary = normalizeSummary(outputSummary);

  const sessionItem = {
    pk: pkTool(sub, toolId),
    sk: skSession(nextSessionId),
    entityType: 'session',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary,
    snapshot: snapshot || {},
    toolIndexSk: nextToolIndexSk,
    userIndexSk: nextUserIndexSk
  };

  const toolIndexItem = {
    pk: pkToolSessions(sub, toolId),
    sk: nextToolIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary
  };

  const userIndexItem = {
    pk: pkUserSessions(sub),
    sk: nextUserIndexSk,
    entityType: 'session_index',
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary
  };

  const isNewSession = !existing;
  const metaUpdate = {
    Update: {
      TableName: tableName,
      Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
      UpdateExpression: 'SET toolId = :toolId, firstUsedAt = if_not_exists(firstUsedAt, :now), lastUsedAt = :now ADD sessionCount :inc',
      ExpressionAttributeValues: {
        ':toolId': toolId,
        ':now': now,
        ':inc': isNewSession ? 1 : 0
      }
    }
  };

  const transactItems = [
    { Put: { TableName: tableName, Item: sessionItem } },
    { Put: { TableName: tableName, Item: toolIndexItem } },
    { Put: { TableName: tableName, Item: userIndexItem } },
    metaUpdate
  ];

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

  await client.send(new TransactWriteCommand({ TransactItems: transactItems }));

  return {
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: summary
  };
}

async function deleteSession({ sub, toolId, sessionId }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const existing = await getSession({ sub, toolId, sessionId });
  if (!existing) return false;

  let record;
  try {
    const result = await client.send(new GetCommand({
      TableName: tableName,
      Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
    }));
    record = result && result.Item ? result.Item : null;
  } catch {
    record = null;
  }

  const toolIndexSk = String(record?.toolIndexSk || '').trim();
  const userIndexSk = String(record?.userIndexSk || '').trim();

  const transactItems = [
    {
      Delete: {
        TableName: tableName,
        Key: { pk: pkTool(sub, toolId), sk: skSession(sessionId) }
      }
    }
  ];

  if (toolIndexSk) {
    transactItems.push({
      Delete: {
        TableName: tableName,
        Key: { pk: pkToolSessions(sub, toolId), sk: toolIndexSk }
      }
    });
  }

  if (userIndexSk) {
    transactItems.push({
      Delete: {
        TableName: tableName,
        Key: { pk: pkUserSessions(sub), sk: userIndexSk }
      }
    });
  }

  await client.send(new TransactWriteCommand({ TransactItems: transactItems }));

  try {
    await client.send(new UpdateCommand({
      TableName: tableName,
      Key: { pk: pkUser(sub), sk: skToolMeta(toolId) },
      ConditionExpression: 'attribute_exists(sessionCount) AND sessionCount > :zero',
      UpdateExpression: 'ADD sessionCount :negOne',
      ExpressionAttributeValues: {
        ':negOne': -1,
        ':zero': 0
      }
    }));
  } catch {}

  return true;
}

async function listSessions({ sub, toolId, limit }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Math.min(Math.max(Number(limit) || 20, 1), 50);

  const result = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': pkToolSessions(sub, toolId),
      ':prefix': 'UPDATED#'
    },
    ScanIndexForward: true,
    Limit: safeLimit
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  return items.map(toSessionMeta).filter(Boolean);
}

async function listRecentSessions(sub, limit){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Math.min(Math.max(Number(limit) || 25, 1), 100);

  const result = await client.send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': pkUserSessions(sub),
      ':prefix': 'UPDATED#'
    },
    ScanIndexForward: true,
    Limit: safeLimit
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  return items.map(toSessionMeta).filter(Boolean);
}

async function logActivity({ sub, toolId, type, summary, data }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const now = Date.now();
  const event = {
    id: randomBase64Url(18),
    toolId,
    type: String(type || '').trim(),
    ts: now,
    summary: normalizeSummary(summary),
    data: data && typeof data === 'object' ? data : undefined
  };

  const revTsKey = reverseTimestampKey(now);
  await client.send(new TransactWriteCommand({
    TransactItems: [
      {
        Put: {
          TableName: tableName,
          Item: {
            pk: pkUserActivity(sub),
            sk: skUserActivityEvent(revTsKey, toolId, event.id),
            entityType: 'activity',
            ...event
          }
        }
      },
      {
        Put: {
          TableName: tableName,
          Item: {
            pk: pkToolActivity(sub, toolId),
            sk: skToolActivityEvent(revTsKey, event.id),
            entityType: 'activity',
            ...event
          }
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
      }
    ]
  }));

  return event;
}

async function listActivity({ sub, toolId, limit }){
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
    Limit: safeLimit
  }));

  const items = (result && Array.isArray(result.Items)) ? result.Items : [];
  return items
    .map(item => ({
      id: String(item.id || '').trim(),
      toolId: String(item.toolId || '').trim(),
      type: String(item.type || '').trim(),
      ts: Number(item.ts) || 0,
      summary: String(item.summary || '').trim(),
      data: item.data && typeof item.data === 'object' ? item.data : undefined
    }))
    .filter(evt => evt.toolId && evt.ts);
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

module.exports = {
  MAX_SNAPSHOT_BYTES,
  saveSession,
  listSessions,
  getSession,
  deleteSession,
  logActivity,
  listActivity,
  listUserTools,
  getToolMeta,
  listRecentSessions
};
