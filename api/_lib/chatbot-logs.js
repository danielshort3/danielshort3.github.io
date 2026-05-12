'use strict';

const crypto = require('crypto');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  GetCommand,
  PutCommand,
  QueryCommand
} = require('@aws-sdk/lib-dynamodb');

let cachedDocClient = null;
let cachedClientKey = '';

function pickEnv(keys) {
  for (const key of keys) {
    const raw = process.env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function numberEnv(key, fallback) {
  const value = Number(process.env[key]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function getLogConfig() {
  return {
    tableName: pickEnv(['CHATBOT_DDB_TABLE', 'CHATBOT_DDB_TABLE_NAME']),
    ttlDays: numberEnv('CHATBOT_LOG_TTL_DAYS', 30)
  };
}

function getRegion() {
  return pickEnv(['CHATBOT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
}

function getAwsCredentialsFromEnv() {
  const accessKeyId = pickEnv(['CHATBOT_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKey = pickEnv(['CHATBOT_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionToken = pickEnv(['CHATBOT_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);
  if (!accessKeyId || !secretAccessKey) return null;
  return {
    accessKeyId,
    secretAccessKey,
    ...(sessionToken ? { sessionToken } : {})
  };
}

function getDocClient() {
  const region = getRegion();
  const credentials = getAwsCredentialsFromEnv();
  const key = `${region}:${credentials ? credentials.accessKeyId : 'default'}`;
  if (cachedDocClient && cachedClientKey === key) return cachedDocClient;

  const client = new DynamoDBClient({ region, credentials: credentials || undefined });
  cachedDocClient = DynamoDBDocumentClient.from(client, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedClientKey = key;
  return cachedDocClient;
}

function requireTableName() {
  const { tableName } = getLogConfig();
  if (!tableName) {
    const err = new Error('CHATBOT_DDB_TABLE is not configured');
    err.code = 'CHATBOT_LOG_STORE_MISSING';
    throw err;
  }
  return tableName;
}

function ttlSeconds(now, days) {
  return Math.floor(now / 1000) + Math.max(1, days) * 86400;
}

function reverseTimestampKey(ts) {
  const safe = Number.isFinite(Number(ts)) && Number(ts) > 0 ? Math.floor(Number(ts)) : Date.now();
  const maxTs = 9_999_999_999_999;
  return String(Math.max(0, maxTs - safe)).padStart(13, '0');
}

function sanitizeText(value, maxLen) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  return text.length > maxLen ? text.slice(0, maxLen) : text;
}

function normalizePageContext(value) {
  const input = value && typeof value === 'object' ? value : {};
  return {
    url: sanitizeText(input.url, 512),
    title: sanitizeText(input.title, 240)
  };
}

function sanitizeSources(value, maxItems = 8) {
  return (Array.isArray(value) ? value : [])
    .map((item) => ({
      id: sanitizeText(item && item.id, 120),
      title: sanitizeText(item && item.title, 160),
      url: sanitizeText(item && item.url, 512),
      category: sanitizeText(item && item.category, 80),
      score: Number.isFinite(Number(item && item.score)) ? Number(item.score) : undefined
    }))
    .filter((item) => item.title || item.url)
    .slice(0, maxItems);
}

function sanitizeLinks(value, maxItems = 8) {
  return (Array.isArray(value) ? value : [])
    .map((item) => ({
      title: sanitizeText(item && item.title, 160),
      url: sanitizeText(item && item.url, 512),
      reason: sanitizeText(item && item.reason, 180)
    }))
    .filter((item) => item.title || item.url)
    .slice(0, maxItems);
}

function requestMetadata(req) {
  const headers = req && req.headers ? req.headers : {};
  return {
    origin: sanitizeText(headers.origin, 512),
    referer: sanitizeText(headers.referer || headers.referrer, 1024),
    userAgent: sanitizeText(headers['user-agent'], 768),
    country: sanitizeText(headers['x-vercel-ip-country'], 64),
    region: sanitizeText(headers['x-vercel-ip-country-region'], 128),
    city: sanitizeText(headers['x-vercel-ip-city'], 128)
  };
}

function summarizeLog(record) {
  if (!record) return null;
  const currentPage = record.currentPage && typeof record.currentPage === 'object'
    ? record.currentPage
    : normalizePageContext(record.pageContext);
  return {
    logId: String(record.logId || '').trim(),
    createdAt: String(record.createdAt || '').trim(),
    ts: Number(record.ts) || 0,
    status: String(record.status || '').trim(),
    conversationId: String(record.conversationId || '').trim(),
    actorHash: String(record.actorHash || '').trim(),
    currentPage,
    question: sanitizeText(record.question, 1000),
    answerPreview: sanitizeText(record.answerPreview || record.answer, 260),
    confidence: Number.isFinite(Number(record.confidence)) ? Number(record.confidence) : 0,
    sourceCount: Number.isFinite(Number(record.sourceCount)) ? Number(record.sourceCount) : 0,
    suggestedLinkCount: Number.isFinite(Number(record.suggestedLinkCount)) ? Number(record.suggestedLinkCount) : 0,
    latencyMs: Number.isFinite(Number(record.latencyMs)) ? Number(record.latencyMs) : 0,
    error: sanitizeText(record.error, 240)
  };
}

async function recordChatbotLog(input = {}) {
  const { tableName, ttlDays } = getLogConfig();
  if (!tableName) return null;

  const now = Number.isFinite(Number(input.ts)) && Number(input.ts) > 0 ? Math.floor(Number(input.ts)) : Date.now();
  const createdAt = new Date(now).toISOString();
  const logId = crypto.randomBytes(10).toString('hex');
  const ttl = ttlSeconds(now, ttlDays);
  const sources = sanitizeSources(input.sources);
  const suggestedLinks = sanitizeLinks(input.suggestedLinks);
  const retrievalChunks = sanitizeSources(input.retrievalChunks, 12);
  const question = sanitizeText(input.question, 1200);
  const answer = sanitizeText(input.answer, 8000);
  const status = sanitizeText(input.status, 80) || 'unknown';
  const pageContext = normalizePageContext(input.pageContext);

  const detailItem = {
    pk: `CHATBOT#LOG#${logId}`,
    sk: 'DETAIL',
    entityType: 'chatbotLog',
    logId,
    createdAt,
    ts: now,
    ttl,
    status,
    actorHash: sanitizeText(input.actorHash, 80),
    conversationId: sanitizeText(input.conversationId, 120),
    question,
    answer,
    error: sanitizeText(input.error, 500),
    pageContext,
    request: requestMetadata(input.req),
    sources,
    suggestedLinks,
    confidence: Number.isFinite(Number(input.confidence)) ? Number(input.confidence) : 0,
    skippedModel: Boolean(input.skippedModel),
    model: sanitizeText(input.model, 120),
    usage: input.usage && typeof input.usage === 'object' ? input.usage : undefined,
    latencyMs: Number.isFinite(Number(input.latencyMs)) ? Math.max(0, Math.floor(Number(input.latencyMs))) : 0,
    rateLimit: input.rateLimit && typeof input.rateLimit === 'object' ? input.rateLimit : undefined,
    retrieval: {
      confident: Boolean(input.retrievalConfident),
      bestScore: Number.isFinite(Number(input.retrievalBestScore)) ? Number(input.retrievalBestScore) : 0,
      queryTerms: Array.isArray(input.queryTerms) ? input.queryTerms.map((term) => sanitizeText(term, 64)).filter(Boolean).slice(0, 40) : [],
      chunks: retrievalChunks
    },
    sourceCount: sources.length,
    suggestedLinkCount: suggestedLinks.length
  };

  const summaryItem = {
    pk: 'CHATBOT#LOGS',
    sk: `TS#${reverseTimestampKey(now)}#${logId}`,
    entityType: 'chatbotLogIndex',
    ttl,
    ...summarizeLog(detailItem)
  };

  const client = getDocClient();
  await Promise.all([
    client.send(new PutCommand({ TableName: tableName, Item: detailItem })),
    client.send(new PutCommand({ TableName: tableName, Item: summaryItem }))
  ]);

  return { logId, summary: summarizeLog(detailItem) };
}

async function listChatbotLogs(options = {}) {
  const tableName = requireTableName();
  const limit = Math.max(1, Math.min(200, Number(options.limit) || 50));
  const result = await getDocClient().send(new QueryCommand({
    TableName: tableName,
    KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
    ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
    ExpressionAttributeValues: {
      ':pk': 'CHATBOT#LOGS',
      ':prefix': 'TS#'
    },
    ScanIndexForward: true,
    Limit: limit
  }));
  const items = Array.isArray(result.Items) ? result.Items : [];
  return items.map(summarizeLog).filter(Boolean);
}

async function getChatbotLog(logId) {
  const id = sanitizeText(logId, 120);
  if (!/^[a-f0-9]{20}$/i.test(id)) return null;
  const tableName = requireTableName();
  const result = await getDocClient().send(new GetCommand({
    TableName: tableName,
    Key: { pk: `CHATBOT#LOG#${id}`, sk: 'DETAIL' }
  }));
  return result.Item || null;
}

function getAdminToken() {
  return pickEnv(['CHATBOT_ADMIN_TOKEN']);
}

function isAdminRequest(req) {
  const configured = getAdminToken();
  if (!configured) return false;
  const headers = req && req.headers ? req.headers : {};
  const auth = headers.authorization ? String(headers.authorization) : '';
  const headerToken = headers['x-admin-token'] || headers['x-chatbot-admin-token'];
  let provided = headerToken ? String(headerToken).trim() : '';
  if (!provided && auth.toLowerCase().startsWith('bearer ')) {
    provided = auth.slice(7).trim();
  }
  if (!provided) return false;

  const a = Buffer.from(provided);
  const b = Buffer.from(configured);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

module.exports = {
  getAdminToken,
  getChatbotLog,
  getLogConfig,
  isAdminRequest,
  listChatbotLogs,
  recordChatbotLog,
  summarizeLog
};
