/*
  DynamoDB storage for short links.

  Env vars required:
  - SHORTLINKS_DDB_TABLE
  - AWS_REGION (or AWS_DEFAULT_REGION)
  - Prefer SHORTLINKS_AWS_ROLE_ARN for Vercel OIDC credentials
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
  TransactWriteCommand,
  UpdateCommand
} = require('@aws-sdk/lib-dynamodb');
const { resolveAwsCredentials } = require('./aws-credentials');
const {
  SET_KEY_PREFIX,
  BATCH_KEY_PREFIX,
  buildSetRecordKey,
  buildBatchRecordKey,
  isInternalRecordSlug,
  normalizeSlugLower
} = require('./short-links');

const SLUG_RESERVATION_PREFIX = '__slug_lower__/';
const CLICK_RETENTION_DAYS = 90;
const CLICK_TTL_ATTRIBUTE = 'expiresAt';
const SHORTLINKS_STATIC_CREDENTIAL_SETS = Object.freeze([
  Object.freeze({
    name: 'short-links',
    accessKeyId: 'SHORTLINKS_AWS_ACCESS_KEY_ID',
    secretAccessKey: 'SHORTLINKS_AWS_SECRET_ACCESS_KEY',
    sessionToken: 'SHORTLINKS_AWS_SESSION_TOKEN'
  }),
  Object.freeze({
    name: 'default',
    accessKeyId: 'AWS_ACCESS_KEY_ID',
    secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
    sessionToken: 'AWS_SESSION_TOKEN'
  })
]);

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

function getAwsCredentialConfig(region){
  return resolveAwsCredentials({
    service: 'short-links',
    region,
    roleArnEnvKeys: ['SHORTLINKS_AWS_ROLE_ARN'],
    staticCredentialSets: SHORTLINKS_STATIC_CREDENTIAL_SETS
  });
}

function getAwsCredentialsFromEnv(region){
  return getAwsCredentialConfig(region).credentials || null;
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

function buildSlugReservationKey(slug){
  const lower = normalizeSlugLower(slug);
  return lower ? `${SLUG_RESERVATION_PREFIX}${lower}` : '';
}

function createSlugConflictError(slug){
  const err = new Error(`Slug conflicts with an existing link: ${slug}`);
  err.code = 'SLUG_CONFLICT';
  err.statusCode = 409;
  return err;
}

function isSlugConflictError(err){
  if (!err) return false;
  if (err.code === 'SLUG_CONFLICT') return true;
  if (err.name !== 'TransactionCanceledException') return false;
  if (Array.isArray(err.CancellationReasons)) {
    return err.CancellationReasons.some(reason => reason && reason.Code === 'ConditionalCheckFailed');
  }
  return /ConditionalCheckFailed/i.test(String(err.message || ''));
}

function encodeCursor(lastEvaluatedKey){
  if (!lastEvaluatedKey || typeof lastEvaluatedKey.slug !== 'string') return '';
  return Buffer.from(JSON.stringify({ slug: lastEvaluatedKey.slug }), 'utf8').toString('base64url');
}

function decodeCursor(cursor){
  const raw = typeof cursor === 'string' ? cursor.trim() : '';
  if (!raw) return undefined;
  try {
    const parsed = JSON.parse(Buffer.from(raw, 'base64url').toString('utf8'));
    if (!parsed || typeof parsed.slug !== 'string' || !parsed.slug) throw new Error('Invalid cursor');
    return { slug: parsed.slug };
  } catch {
    const err = new Error('Invalid pagination cursor');
    err.code = 'INVALID_CURSOR';
    err.statusCode = 400;
    throw err;
  }
}

let cachedDocClient = null;
let cachedKey = '';

function getDocClient(){
  const { region } = getRequiredEnv();
  const auth = getAwsCredentialConfig(region);
  const key = `${region}:${auth.cacheKey}`;
  if (cachedDocClient && cachedKey === key) return cachedDocClient;

  const client = new DynamoDBClient({ region, credentials: auth.credentials });
  cachedDocClient = DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedKey = key;
  return cachedDocClient;
}

function isLinkEntity(item){
  if (!item || typeof item !== 'object') return false;
  const slug = typeof item.slug === 'string' ? item.slug : '';
  if (!slug || isInternalRecordSlug(slug)) return false;
  return !item.entityType || item.entityType === 'link';
}

function isSetTemplateEntity(item){
  if (!item || typeof item !== 'object') return false;
  const slug = typeof item.slug === 'string' ? item.slug : '';
  return item.entityType === 'setTemplate' && slug.startsWith(SET_KEY_PREFIX);
}

function isGeneratedBatchEntity(item){
  if (!item || typeof item !== 'object') return false;
  const slug = typeof item.slug === 'string' ? item.slug : '';
  return item.entityType === 'generatedBatch' && slug.startsWith(BATCH_KEY_PREFIX);
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

async function getSlugReservation(slug){
  const key = buildSlugReservationKey(slug);
  if (!key) return null;
  return getLink(key);
}

async function getLinkWithLegacyFallback(slug){
  const exact = await getLink(slug);
  if (isLinkEntity(exact)) return exact;

  const lower = normalizeSlugLower(slug);
  if (!lower) return null;

  const reservation = await getSlugReservation(lower);
  const canonicalSlug = reservation && typeof reservation.canonicalSlug === 'string'
    ? reservation.canonicalSlug
    : '';
  if (canonicalSlug && canonicalSlug !== slug) {
    const canonical = await getLink(canonicalSlug);
    if (isLinkEntity(canonical)) return canonical;
  }

  if (lower === slug) return null;

  const legacy = await getLink(lower);
  return isLinkEntity(legacy) ? legacy : null;
}

async function putItem(item){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  await client.send(new PutCommand({
    TableName: tableName,
    Item: item
  }));
  return item;
}

async function upsertLink({ slug, destination, permanent, expiresAt, updatedAt, metadata }){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const setExpressions = [
    'entityType = :entityType',
    'slugLower = :slugLower',
    'destination = :destination',
    'permanent = :permanent',
    'updatedAt = :updatedAt',
    'disabled = if_not_exists(disabled, :disabled)',
    'createdAt = if_not_exists(createdAt, :createdAt)',
    'clicks = if_not_exists(clicks, :zero)'
  ];

  const removeExpressions = [];
  const values = {
    ':entityType': 'link',
    ':slugLower': normalizeSlugLower(slug),
    ':destination': destination,
    ':permanent': !!permanent,
    ':updatedAt': updatedAt,
    ':disabled': false,
    ':createdAt': updatedAt,
    ':zero': 0
  };

  if (permanent) {
    removeExpressions.push('expiresAt');
  } else if (typeof expiresAt !== 'undefined') {
    const numericExpiresAt = Number(expiresAt);
    if (Number.isFinite(numericExpiresAt) && numericExpiresAt > 0) {
      setExpressions.push('expiresAt = :expiresAt');
      values[':expiresAt'] = Math.floor(numericExpiresAt);
    } else {
      removeExpressions.push('expiresAt');
    }
  }

  const optionalTextFields = [
    ['label', 160],
    ['templateId', 64],
    ['templateTitle', 160],
    ['batchId', 64],
    ['batchTitle', 160],
    ['contextType', 32],
    ['contextEntryId', 96],
    ['contextCompany', 160],
    ['contextTitle', 160]
  ];

  optionalTextFields.forEach(([field, maxLen]) => {
    const cleaned = sanitizeValue(metadata && metadata[field], maxLen);
    if (cleaned) {
      setExpressions.push(`${field} = :${field}`);
      values[`:${field}`] = cleaned;
    } else {
      removeExpressions.push(field);
    }
  });

  const updateExpression = `SET ${setExpressions.join(', ')}${removeExpressions.length ? ` REMOVE ${removeExpressions.join(', ')}` : ''}`;
  const reservationKey = buildSlugReservationKey(slug);
  if (!reservationKey) throw createSlugConflictError(slug);

  try {
    await client.send(new TransactWriteCommand({
      TransactItems: [
        {
          Update: {
            TableName: tableName,
            Key: { slug: reservationKey },
            ConditionExpression: 'attribute_not_exists(#slug) OR #canonicalSlug = :canonicalSlug',
            UpdateExpression: 'SET #entityType = :reservationType, #slugLower = :slugLower, #canonicalSlug = :canonicalSlug, #createdAt = if_not_exists(#createdAt, :updatedAt), #updatedAt = :updatedAt',
            ExpressionAttributeNames: {
              '#slug': 'slug',
              '#entityType': 'entityType',
              '#slugLower': 'slugLower',
              '#canonicalSlug': 'canonicalSlug',
              '#createdAt': 'createdAt',
              '#updatedAt': 'updatedAt'
            },
            ExpressionAttributeValues: {
              ':reservationType': 'slugReservation',
              ':slugLower': normalizeSlugLower(slug),
              ':canonicalSlug': slug,
              ':updatedAt': updatedAt
            }
          }
        },
        {
          Update: {
            TableName: tableName,
            Key: { slug },
            UpdateExpression: updateExpression,
            ExpressionAttributeValues: values
          }
        }
      ]
    }));
  } catch (err) {
    if (isSlugConflictError(err)) throw createSlugConflictError(slug);
    throw err;
  }

  return getLink(slug);
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
  const reservationKey = buildSlugReservationKey(slug);
  if (!reservationKey || isInternalRecordSlug(slug)) {
    await client.send(new DeleteCommand({
      TableName: tableName,
      Key: { slug }
    }));
    return;
  }

  await client.send(new TransactWriteCommand({
    TransactItems: [
      {
        Delete: {
          TableName: tableName,
          Key: { slug }
        }
      },
      {
        Delete: {
          TableName: tableName,
          Key: { slug: reservationKey },
          ConditionExpression: 'attribute_not_exists(#slug) OR #canonicalSlug = :canonicalSlug',
          ExpressionAttributeNames: {
            '#slug': 'slug',
            '#canonicalSlug': 'canonicalSlug'
          },
          ExpressionAttributeValues: { ':canonicalSlug': slug }
        }
      }
    ]
  }));
}

async function incrementClicks(slug){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  await client.send(new UpdateCommand({
    TableName: tableName,
    Key: { slug },
    ConditionExpression: 'attribute_exists(#slug) AND (attribute_not_exists(#entityType) OR #entityType = :linkType)',
    UpdateExpression: 'SET clicks = if_not_exists(clicks, :zero) + :one',
    ExpressionAttributeNames: {
      '#slug': 'slug',
      '#entityType': 'entityType'
    },
    ExpressionAttributeValues: {
      ':zero': 0,
      ':one': 1,
      ':linkType': 'link'
    }
  }));
}

async function listAllItems(){
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

async function listLinks(){
  const items = await listAllItems();
  return items.filter(isLinkEntity);
}

async function listLinksPage({ limit, cursor } = {}){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Number.isFinite(Number(limit)) ? Math.max(1, Math.min(500, Math.floor(Number(limit)))) : 100;
  const items = [];
  let startKey = decodeCursor(cursor);
  let lastEvaluatedKey;

  do {
    const remaining = safeLimit - items.length;
    const result = await client.send(new ScanCommand({
      TableName: tableName,
      ExclusiveStartKey: startKey,
      Limit: remaining,
      FilterExpression: 'attribute_not_exists(#entityType) OR #entityType = :linkType',
      ExpressionAttributeNames: { '#entityType': 'entityType' },
      ExpressionAttributeValues: { ':linkType': 'link' }
    }));
    if (result && Array.isArray(result.Items)) items.push(...result.Items.filter(isLinkEntity));
    lastEvaluatedKey = result && result.LastEvaluatedKey ? result.LastEvaluatedKey : undefined;
    startKey = lastEvaluatedKey;
  } while (items.length < safeLimit && startKey);

  return {
    items,
    nextCursor: encodeCursor(lastEvaluatedKey)
  };
}

async function findLinkByLowerSlug(slug){
  const target = normalizeSlugLower(slug);
  if (!target) return null;
  const items = await listLinks();
  return items.find(item => normalizeSlugLower(item.slug) === target) || null;
}

async function getSetTemplate(setId){
  return getLink(buildSetRecordKey(setId));
}

async function listSetTemplates(){
  const items = await listAllItems();
  return items
    .filter(isSetTemplateEntity)
    .sort((a, b) => String(a.title || '').localeCompare(String(b.title || '')));
}

async function listSetTemplatesPage({ limit, cursor } = {}){
  const { tableName } = getRequiredEnv();
  const client = getDocClient();
  const safeLimit = Number.isFinite(Number(limit)) ? Math.max(1, Math.min(200, Math.floor(Number(limit)))) : 50;
  const items = [];
  let startKey = decodeCursor(cursor);
  let lastEvaluatedKey;

  do {
    const remaining = safeLimit - items.length;
    const result = await client.send(new ScanCommand({
      TableName: tableName,
      ExclusiveStartKey: startKey,
      Limit: remaining,
      FilterExpression: '#entityType = :entityType',
      ExpressionAttributeNames: { '#entityType': 'entityType' },
      ExpressionAttributeValues: { ':entityType': 'setTemplate' }
    }));
    if (result && Array.isArray(result.Items)) items.push(...result.Items.filter(isSetTemplateEntity));
    lastEvaluatedKey = result && result.LastEvaluatedKey ? result.LastEvaluatedKey : undefined;
    startKey = lastEvaluatedKey;
  } while (items.length < safeLimit && startKey);

  return {
    items,
    nextCursor: encodeCursor(lastEvaluatedKey)
  };
}

async function saveSetTemplate(template){
  return putItem(template);
}

async function deleteSetTemplate(setId){
  return deleteLink(buildSetRecordKey(setId));
}

async function saveGeneratedBatch(batch){
  return putItem(batch);
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
    path: sanitizeValue(event.path, 2048),
    refererHost: sanitizeValue(event.refererHost, 255),
    userAgent: sanitizeValue(event.userAgent, 768),
    country: sanitizeValue(event.country, 64),
    region: sanitizeValue(event.region, 128),
    city: sanitizeValue(event.city, 128),
    timezone: sanitizeValue(event.timezone, 128)
  };

  item[CLICK_TTL_ATTRIBUTE] = Math.floor((Date.now() + CLICK_RETENTION_DAYS * 24 * 60 * 60 * 1000) / 1000);

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
  getAwsCredentialConfig,
  getAwsCredentialsFromEnv,
  getAwsCredentialEnvInfo,
  getRequiredEnv,
  getLink,
  getLinkWithLegacyFallback,
  putItem,
  upsertLink,
  setLinkDisabled,
  deleteLink,
  incrementClicks,
  listAllItems,
  listLinks,
  listLinksPage,
  findLinkByLowerSlug,
  getSetTemplate,
  listSetTemplates,
  listSetTemplatesPage,
  saveSetTemplate,
  deleteSetTemplate,
  saveGeneratedBatch,
  isSetTemplateEntity,
  isGeneratedBatchEntity,
  recordClick,
  listClicks,
  CLICK_RETENTION_DAYS,
  CLICK_TTL_ATTRIBUTE,
  SLUG_RESERVATION_PREFIX,
  buildSlugReservationKey,
  isSlugConflictError
};
