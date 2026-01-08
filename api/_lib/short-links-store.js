/*
  DynamoDB storage for short links.

  Env vars required:
  - SHORTLINKS_DDB_TABLE
  - AWS_REGION (or AWS_DEFAULT_REGION)
  - Prefer SHORTLINKS_AWS_ACCESS_KEY_ID / SHORTLINKS_AWS_SECRET_ACCESS_KEY to avoid conflicts
  - Falls back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or any AWS SDK credential provider chain)
  - Optional (temporary creds only): SHORTLINKS_AWS_SESSION_TOKEN / AWS_SESSION_TOKEN

  Optional click log table:
  - SHORTLINKS_DDB_CLICKS_TABLE
*/
'use strict';

const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  DeleteCommand,
  GetCommand,
  PutCommand,
  QueryCommand,
  ScanCommand,
  UpdateCommand
} = require('@aws-sdk/lib-dynamodb');

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
  const accessKeyIdEnv = pickEnv(['SHORTLINKS_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKeyEnv = pickEnv(['SHORTLINKS_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionTokenEnv = pickEnv(['SHORTLINKS_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);

  const accessKeyId = accessKeyIdEnv.raw.trim();
  const secretAccessKey = secretAccessKeyEnv.raw.trim();
  const sessionToken = sessionTokenEnv.raw.trim();

  if (!accessKeyId || !secretAccessKey) return null;

  const creds = { accessKeyId, secretAccessKey };
  if (sessionToken && accessKeyId.startsWith('ASIA')) creds.sessionToken = sessionToken;
  return creds;
}

function getAwsCredentialEnvInfo(){
  const crypto = require('crypto');

  const accessKeyIdEnv = pickEnv(['SHORTLINKS_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKeyEnv = pickEnv(['SHORTLINKS_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionTokenEnv = pickEnv(['SHORTLINKS_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);

  const accessKeyIdRaw = accessKeyIdEnv.raw;
  const secretAccessKeyRaw = secretAccessKeyEnv.raw;
  const sessionTokenRaw = sessionTokenEnv.raw;

  const accessKeyId = accessKeyIdRaw.trim();
  const secretAccessKey = secretAccessKeyRaw.trim();
  const sessionToken = sessionTokenRaw.trim();

  const sessionTokenUsed = !!(sessionToken && accessKeyId && accessKeyId.startsWith('ASIA'));
  const secretFingerprint = secretAccessKey
    ? crypto.createHash('sha256').update(secretAccessKey, 'utf8').digest('hex').slice(0, 12)
    : '';

  return {
    accessKeyId,
    accessKeyIdSource: accessKeyIdEnv.key,
    secretSource: secretAccessKeyEnv.key,
    sessionTokenSource: sessionTokenEnv.key,
    accessKeyConfigured: !!accessKeyId,
    secretConfigured: !!secretAccessKey,
    sessionTokenConfigured: !!sessionToken,
    sessionTokenUsed,
    sessionTokenIgnored: !!sessionToken && !sessionTokenUsed,
    accessKeyTrimmed: accessKeyIdRaw !== accessKeyId,
    secretTrimmed: secretAccessKeyRaw !== secretAccessKey,
    sessionTokenTrimmed: sessionTokenRaw !== sessionToken,
    accessKeyLength: accessKeyIdRaw.length,
    secretLength: secretAccessKeyRaw.length,
    sessionTokenLength: sessionTokenRaw.length,
    secretFingerprint
  };
}

function getRequiredEnv(){
  const tableName = process.env.SHORTLINKS_DDB_TABLE ? String(process.env.SHORTLINKS_DDB_TABLE).trim() : '';
  const region =
    (process.env.AWS_REGION ? String(process.env.AWS_REGION).trim() : '') ||
    (process.env.AWS_DEFAULT_REGION ? String(process.env.AWS_DEFAULT_REGION).trim() : '');

  if (!tableName) {
    const err = new Error('SHORTLINKS_DDB_TABLE is not configured');
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

function getClicksTableName(){
  return process.env.SHORTLINKS_DDB_CLICKS_TABLE ? String(process.env.SHORTLINKS_DDB_CLICKS_TABLE).trim() : '';
}

function sanitizeValue(value, maxLen){
  const raw = typeof value === 'string' ? value : '';
  if (!raw) return '';
  const cleaned = raw.replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  if (cleaned.length <= maxLen) return cleaned;
  return cleaned.slice(0, maxLen);
}

let cachedDocClient = null;
let cachedKey = '';

function getDocClient(){
  const { region } = getRequiredEnv();
  const creds = getAwsCredentialsFromEnv();
  const key = `${region}:${creds ? creds.accessKeyId : 'default'}`;
  if (cachedDocClient && cachedKey === key) return cachedDocClient;

  const client = new DynamoDBClient({ region, credentials: creds || undefined });
  cachedDocClient = DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedKey = key;
  return cachedDocClient;
}

async function getLink(slug){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const result = await client.send(new GetCommand({
    TableName: tableName,
    Key: { slug }
  }));
  return result && result.Item ? result.Item : null;
}

async function upsertLink({ slug, destination, permanent, expiresAt, updatedAt }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const setExpressions = [
    'destination = :destination',
    'permanent = :permanent',
    'updatedAt = :updatedAt',
    'disabled = if_not_exists(disabled, :disabled)',
    'createdAt = if_not_exists(createdAt, :createdAt)',
    'clicks = if_not_exists(clicks, :zero)'
  ];

  const removeExpressions = [];
  const values = {
    ':destination': destination,
    ':permanent': !!permanent,
    ':updatedAt': updatedAt,
    ':disabled': false,
    ':createdAt': updatedAt,
    ':zero': 0
  };

  if (typeof expiresAt !== 'undefined') {
    const numericExpiresAt = Number(expiresAt);
    if (Number.isFinite(numericExpiresAt) && numericExpiresAt > 0) {
      setExpressions.push('expiresAt = :expiresAt');
      values[':expiresAt'] = Math.floor(numericExpiresAt);
    } else {
      removeExpressions.push('expiresAt');
    }
  }

  const updateExpression = `SET ${setExpressions.join(', ')}${removeExpressions.length ? ` REMOVE ${removeExpressions.join(', ')}` : ''}`;
  const result = await client.send(new UpdateCommand({
    TableName: tableName,
    Key: { slug },
    UpdateExpression: updateExpression,
    ExpressionAttributeValues: values,
    ReturnValues: 'ALL_NEW'
  }));
  return result && result.Attributes ? result.Attributes : null;
}

async function setLinkDisabled({ slug, disabled, updatedAt }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const result = await client.send(new UpdateCommand({
    TableName: tableName,
    Key: { slug },
    ConditionExpression: 'attribute_exists(slug)',
    UpdateExpression: 'SET disabled = :disabled, updatedAt = :updatedAt',
    ExpressionAttributeValues: {
      ':disabled': !!disabled,
      ':updatedAt': updatedAt
    },
    ReturnValues: 'ALL_NEW'
  }));
  return result && result.Attributes ? result.Attributes : null;
}

async function deleteLink(slug){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  await client.send(new DeleteCommand({
    TableName: tableName,
    Key: { slug }
  }));
}

async function incrementClicks(slug){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  await client.send(new UpdateCommand({
    TableName: tableName,
    Key: { slug },
    UpdateExpression: 'SET clicks = if_not_exists(clicks, :zero) + :one',
    ExpressionAttributeValues: {
      ':zero': 0,
      ':one': 1
    }
  }));
}

async function listLinks(){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();

  const items = [];
  let startKey;
  do {
    const result = await client.send(new ScanCommand({
      TableName: tableName,
      ExclusiveStartKey: startKey
    }));
    if (result && Array.isArray(result.Items)) items.push(...result.Items);
    startKey = result && result.LastEvaluatedKey ? result.LastEvaluatedKey : undefined;
  } while (startKey);

  return items;
}

async function recordClick(event){
  const clicksTableName = getClicksTableName();
  if (!clicksTableName) return;
  const client = getDocClient();

  const slug = typeof event.slug === 'string' ? event.slug : '';
  const clickId = typeof event.clickId === 'string' ? event.clickId : '';
  const clickedAt = typeof event.clickedAt === 'string' ? event.clickedAt : '';
  if (!slug || !clickId || !clickedAt) return;

  const item = {
    slug,
    clickId,
    clickedAt,
    destination: sanitizeValue(event.destination, 2048),
    statusCode: Number.isFinite(Number(event.statusCode)) ? Number(event.statusCode) : undefined,
    host: sanitizeValue(event.host, 255),
    referer: sanitizeValue(event.referer, 2048),
    userAgent: sanitizeValue(event.userAgent, 768),
    country: sanitizeValue(event.country, 64),
    region: sanitizeValue(event.region, 128),
    city: sanitizeValue(event.city, 128)
  };

  const ttlDays = 90;
  item.expiresAt = Math.floor((Date.now() + ttlDays * 24 * 60 * 60 * 1000) / 1000);

  await client.send(new PutCommand({
    TableName: clicksTableName,
    Item: item
  }));
}

async function listClicks({ slug, limit }){
  const clicksTableName = getClicksTableName();
  if (!clicksTableName) {
    const err = new Error('SHORTLINKS_DDB_CLICKS_TABLE is not configured');
    err.code = 'DDB_CLICKS_ENV_MISSING';
    throw err;
  }
  const client = getDocClient();
  const safeLimit = Number.isFinite(Number(limit)) ? Math.max(1, Math.min(500, Number(limit))) : 100;

  const result = await client.send(new QueryCommand({
    TableName: clicksTableName,
    KeyConditionExpression: 'slug = :slug',
    ExpressionAttributeValues: { ':slug': slug },
    ScanIndexForward: false,
    Limit: safeLimit
  }));

  return result && Array.isArray(result.Items) ? result.Items : [];
}

module.exports = {
  getAwsCredentialsFromEnv,
  getAwsCredentialEnvInfo,
  getRequiredEnv,
  getLink,
  upsertLink,
  setLinkDisabled,
  deleteLink,
  incrementClicks,
  listLinks,
  recordClick,
  listClicks
};
