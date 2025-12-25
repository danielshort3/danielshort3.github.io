const { randomUUID } = require('crypto');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  PutCommand,
  UpdateCommand,
  DeleteCommand,
  QueryCommand
} = require('@aws-sdk/lib-dynamodb');
const { S3Client, PutObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

const dynamo = DynamoDBDocumentClient.from(new DynamoDBClient({}), {
  marshallOptions: {
    removeUndefinedValues: true,
    convertEmptyValues: true
  }
});
const s3 = new S3Client({});

const {
  APPLICATIONS_TABLE,
  ATTACHMENTS_BUCKET,
  ALLOWED_ORIGINS = '',
  PRESIGN_TTL_SECONDS = '900'
} = process.env;

const allowedOrigins = ALLOWED_ORIGINS.split(',').map(origin => origin.trim()).filter(Boolean);
const presignTtl = Math.max(parseInt(PRESIGN_TTL_SECONDS, 10) || 900, 60);

const INTERVIEW_STATUSES = new Set([
  'screening',
  'screen',
  'phone screen',
  'interview',
  'panel',
  'onsite',
  'assessment'
]);
const OFFER_STATUSES = new Set(['offer', 'accepted']);
const REJECTION_STATUSES = new Set(['rejected', 'declined', 'no response', 'ghosted']);

const httpError = (statusCode, message) => {
  const err = new Error(message);
  err.statusCode = statusCode;
  return err;
};

const buildHeaders = (origin) => ({
  'Access-Control-Allow-Origin': origin || '*',
  'Access-Control-Allow-Methods': 'GET,POST,PATCH,DELETE,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,Authorization',
  'Access-Control-Max-Age': '86400',
  'Content-Type': 'application/json'
});

const resolveCorsOrigin = (origin) => {
  if (!allowedOrigins.length) return origin || '*';
  if (allowedOrigins.includes('*')) return origin || '*';
  if (origin && allowedOrigins.includes(origin)) return origin;
  return allowedOrigins[0];
};

const parseBody = (event = {}) => {
  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body || '', 'base64').toString('utf8')
    : (event.body || '');
  if (!rawBody) return {};
  try {
    return JSON.parse(rawBody);
  } catch {
    return {};
  }
};

const nowIso = () => new Date().toISOString();

const normalizeStatus = (value) => {
  if (!value) return 'Applied';
  return value
    .toString()
    .trim()
    .split(/\s+/)
    .map(word => word ? word[0].toUpperCase() + word.slice(1).toLowerCase() : '')
    .join(' ');
};

const normalizeUrl = (value) => {
  const trimmed = (value || '').toString().trim();
  if (!trimmed) return '';
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  return `https://${trimmed}`;
};

const normalizeAttachments = (userId, attachments) => {
  if (attachments === undefined || attachments === null) return [];
  if (!Array.isArray(attachments)) {
    throw httpError(400, 'attachments must be an array.');
  }
  const now = nowIso();
  const safe = [];
  for (const attachment of attachments) {
    if (!attachment || typeof attachment !== 'object') continue;
    const key = (attachment.key || '').toString().trim();
    const filename = (attachment.filename || '').toString().trim();
    if (!key || !filename) continue;
    if (!key.startsWith(`${userId}/`)) {
      throw httpError(400, 'Invalid attachment key.');
    }
    safe.push({
      key,
      filename,
      contentType: (attachment.contentType || '').toString().trim(),
      kind: (attachment.kind || '').toString().trim(),
      uploadedAt: (attachment.uploadedAt || now).toString().trim()
    });
    if (safe.length >= 12) break;
  }
  return safe;
};

const normalizePath = (event) => {
  const rawPath = event.rawPath || event.path || '';
  const stage = event.requestContext?.stage;
  if (stage && rawPath.startsWith(`/${stage}/`)) {
    return rawPath.slice(stage.length + 1);
  }
  return rawPath;
};

const parseDate = (value) => {
  if (!value || typeof value !== 'string') return null;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const formatDate = (date) => date.toISOString().slice(0, 10);

const getRange = (query = {}) => {
  const end = parseDate(query.end) || new Date();
  const start = parseDate(query.start) || new Date(end.getTime() - 89 * 86400000);
  const safeStart = start > end ? end : start;
  const maxRangeMs = 365 * 86400000;
  if (end.getTime() - safeStart.getTime() > maxRangeMs) {
    return {
      start: new Date(end.getTime() - maxRangeMs),
      end
    };
  }
  return { start: safeStart, end };
};

const getUserId = (event) => {
  const claims = event?.requestContext?.authorizer?.jwt?.claims || {};
  const userId = claims.sub || claims.username || claims['cognito:username'];
  if (!userId) throw httpError(401, 'Unauthorized.');
  return userId;
};

const queryApplications = async (userId, { range, limit, scanForward, recordType } = {}) => {
  const params = {
    TableName: APPLICATIONS_TABLE,
    KeyConditionExpression: 'userId = :userId',
    ExpressionAttributeValues: { ':userId': userId },
    ScanIndexForward: scanForward !== false
  };
  const filters = [];
  const names = {};
  if (Number.isFinite(limit) && limit > 0) {
    params.Limit = Math.max(1, Math.floor(limit));
  }
  if (range) {
    filters.push('#appliedDate BETWEEN :start AND :end');
    names['#appliedDate'] = 'appliedDate';
    params.ExpressionAttributeValues[':start'] = formatDate(range.start);
    params.ExpressionAttributeValues[':end'] = formatDate(range.end);
  }
  if (recordType) {
    names['#recordType'] = 'recordType';
    params.ExpressionAttributeValues[':recordType'] = recordType;
    if (recordType === 'application') {
      filters.push('(attribute_not_exists(#recordType) OR #recordType = :recordType)');
    } else {
      filters.push('#recordType = :recordType');
    }
  }
  if (filters.length) {
    params.FilterExpression = filters.join(' AND ');
    params.ExpressionAttributeNames = names;
  }

  let items = [];
  let lastKey;
  do {
    const result = await dynamo.send(new QueryCommand({ ...params, ExclusiveStartKey: lastKey }));
    items = items.concat(result.Items || []);
    lastKey = result.LastEvaluatedKey;
    if (params.Limit) break;
  } while (lastKey);
  return items;
};

const buildDailySeries = (items, range) => {
  const start = new Date(range.start);
  const end = new Date(range.end);
  const series = [];
  const counts = new Map();
  let cursor = new Date(start);
  while (cursor <= end) {
    const key = formatDate(cursor);
    counts.set(key, 0);
    cursor = new Date(cursor.getTime() + 86400000);
  }
  items.forEach((item) => {
    const date = item.appliedDate;
    if (counts.has(date)) {
      counts.set(date, counts.get(date) + 1);
    }
  });
  counts.forEach((count, date) => series.push({ date, count }));
  return series;
};

const buildStatusBreakdown = (items) => {
  const counts = new Map();
  items.forEach((item) => {
    const status = normalizeStatus(item.status || 'Applied');
    counts.set(status, (counts.get(status) || 0) + 1);
  });
  return Array.from(counts.entries())
    .map(([status, count]) => ({ status, count }))
    .sort((a, b) => b.count - a.count);
};

const buildSummary = (items) => {
  let interviews = 0;
  let offers = 0;
  let rejections = 0;
  items.forEach((item) => {
    const status = (item.status || '').toString().trim().toLowerCase();
    if (INTERVIEW_STATUSES.has(status)) interviews += 1;
    if (OFFER_STATUSES.has(status)) offers += 1;
    if (REJECTION_STATUSES.has(status)) rejections += 1;
  });
  return {
    totalApplications: items.length,
    interviews,
    offers,
    rejections
  };
};

const handleCreateApplication = async (userId, payload = {}) => {
  const company = (payload.company || '').toString().trim();
  const title = (payload.title || '').toString().trim();
  const appliedDate = (payload.appliedDate || '').toString().trim();
  if (!company || !title || !appliedDate) {
    throw httpError(400, 'Company, title, and appliedDate are required.');
  }
  const parsedDate = parseDate(appliedDate);
  if (!parsedDate) throw httpError(400, 'appliedDate must be YYYY-MM-DD.');
  const status = normalizeStatus(payload.status || 'Applied');
  const attachments = normalizeAttachments(userId, payload.attachments);
  const now = nowIso();
  const applicationId = `APP#${Date.now()}#${randomUUID()}`;
  const item = {
    userId,
    applicationId,
    recordType: 'application',
    company,
    title,
    appliedDate: formatDate(parsedDate),
    status,
    notes: (payload.notes || '').toString().trim(),
    statusHistory: [{ status, date: now }],
    createdAt: now,
    updatedAt: now
  };
  if (attachments.length) item.attachments = attachments;
  await dynamo.send(new PutCommand({
    TableName: APPLICATIONS_TABLE,
    Item: item
  }));
  return item;
};

const handleCreateProspect = async (userId, payload = {}) => {
  const company = (payload.company || '').toString().trim();
  const title = (payload.title || '').toString().trim();
  const jobUrl = normalizeUrl(payload.jobUrl || payload.url);
  if (!company || !title || !jobUrl) {
    throw httpError(400, 'Company, title, and jobUrl are required.');
  }
  const status = normalizeStatus(payload.status || 'Active');
  const now = nowIso();
  const prospectId = `PROSPECT#${Date.now()}#${randomUUID()}`;
  const item = {
    userId,
    applicationId: prospectId,
    recordType: 'prospect',
    company,
    title,
    jobUrl,
    location: (payload.location || '').toString().trim(),
    source: (payload.source || '').toString().trim(),
    status,
    notes: (payload.notes || '').toString().trim(),
    createdAt: now,
    updatedAt: now
  };
  await dynamo.send(new PutCommand({
    TableName: APPLICATIONS_TABLE,
    Item: item
  }));
  return item;
};

const handleUpdateProspect = async (userId, applicationId, payload = {}) => {
  if (!applicationId) throw httpError(400, 'Missing prospectId.');
  const updates = [];
  const names = {};
  const values = { ':updatedAt': nowIso() };

  const addField = (key, value) => {
    if (value === undefined || value === null) return;
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  };

  addField('company', payload.company?.toString().trim());
  addField('title', payload.title?.toString().trim());
  if (payload.jobUrl || payload.url) {
    const jobUrl = normalizeUrl(payload.jobUrl || payload.url);
    if (!jobUrl) throw httpError(400, 'jobUrl is required.');
    addField('jobUrl', jobUrl);
  }
  addField('location', payload.location?.toString().trim());
  addField('source', payload.source?.toString().trim());
  addField('notes', payload.notes?.toString().trim());
  if (payload.status) {
    addField('status', normalizeStatus(payload.status));
  }

  updates.push('#updatedAt = :updatedAt');
  names['#updatedAt'] = 'updatedAt';

  const result = await dynamo.send(new UpdateCommand({
    TableName: APPLICATIONS_TABLE,
    Key: { userId, applicationId },
    UpdateExpression: `SET ${updates.join(', ')}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
    ReturnValues: 'ALL_NEW'
  }));
  return result.Attributes;
};

const handleUpdateApplication = async (userId, applicationId, payload = {}) => {
  if (!applicationId) throw httpError(400, 'Missing applicationId.');
  const updates = [];
  const names = {};
  const values = { ':updatedAt': nowIso() };
  const pushStatus = payload.status ? normalizeStatus(payload.status) : null;

  const addField = (key, value) => {
    if (value === undefined || value === null) return;
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  };

  addField('company', payload.company?.toString().trim());
  addField('title', payload.title?.toString().trim());
  if (payload.appliedDate) {
    const parsedDate = parseDate(payload.appliedDate);
    if (!parsedDate) throw httpError(400, 'appliedDate must be YYYY-MM-DD.');
    addField('appliedDate', formatDate(parsedDate));
  }
  addField('notes', payload.notes?.toString().trim());
  if (payload.attachments) {
    const attachments = normalizeAttachments(userId, payload.attachments);
    addField('attachments', attachments);
  }
  if (pushStatus) {
    addField('status', pushStatus);
    names['#statusHistory'] = 'statusHistory';
    values[':empty'] = [];
    values[':statusEntry'] = [{ status: pushStatus, date: values[':updatedAt'] }];
    updates.push('#statusHistory = list_append(if_not_exists(#statusHistory, :empty), :statusEntry)');
  }

  updates.push('#updatedAt = :updatedAt');
  names['#updatedAt'] = 'updatedAt';

  const result = await dynamo.send(new UpdateCommand({
    TableName: APPLICATIONS_TABLE,
    Key: { userId, applicationId },
    UpdateExpression: `SET ${updates.join(', ')}`,
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values,
    ReturnValues: 'ALL_NEW'
  }));
  return result.Attributes;
};

const handleDeleteApplication = async (userId, applicationId) => {
  if (!applicationId) throw httpError(400, 'Missing applicationId.');
  await dynamo.send(new DeleteCommand({
    TableName: APPLICATIONS_TABLE,
    Key: { userId, applicationId }
  }));
  return { ok: true };
};

const handlePresign = async (userId, payload = {}) => {
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const applicationId = (payload.applicationId || '').toString().trim();
  const filename = (payload.filename || '').toString().trim();
  const contentType = (payload.contentType || 'application/octet-stream').toString().trim();
  if (!applicationId || !filename) {
    throw httpError(400, 'applicationId and filename are required.');
  }
  const safeName = filename.replace(/[^a-zA-Z0-9._-]+/g, '-');
  const key = `${userId}/${applicationId}/${Date.now()}-${safeName}`;
  const uploadUrl = await getSignedUrl(
    s3,
    new PutObjectCommand({
      Bucket: ATTACHMENTS_BUCKET,
      Key: key,
      ContentType: contentType
    }),
    { expiresIn: presignTtl }
  );
  return {
    uploadUrl,
    key,
    bucket: ATTACHMENTS_BUCKET,
    expiresIn: presignTtl
  };
};

const handlePresignDownload = async (userId, payload = {}) => {
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const key = (payload.key || '').toString().trim();
  if (!key) throw httpError(400, 'key is required.');
  if (!key.startsWith(`${userId}/`)) throw httpError(403, 'Unauthorized.');
  const downloadUrl = await getSignedUrl(
    s3,
    new GetObjectCommand({
      Bucket: ATTACHMENTS_BUCKET,
      Key: key
    }),
    { expiresIn: presignTtl }
  );
  return {
    downloadUrl,
    key,
    expiresIn: presignTtl
  };
};

exports.handler = async (event) => {
  const requestOrigin = event?.headers?.origin || event?.headers?.Origin || '';
  const corsOrigin = resolveCorsOrigin(requestOrigin);
  const method = event.requestContext?.http?.method || event.httpMethod || '';
  const path = normalizePath(event);
  const routeKey = event.routeKey || `${method} ${path}`;

  if (method === 'OPTIONS') {
    return { statusCode: 204, headers: buildHeaders(corsOrigin) };
  }

  try {
    const userId = getUserId(event);
    if (!APPLICATIONS_TABLE) throw httpError(500, 'Applications table not configured.');

    if (routeKey.startsWith('GET /api/analytics/summary')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const summary = buildSummary(items);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          ...summary,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey.startsWith('GET /api/analytics/applications-over-time')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const series = buildDailySeries(items, range);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          series,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey.startsWith('GET /api/analytics/status-breakdown')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const statuses = buildStatusBreakdown(items);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          statuses,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey.startsWith('GET /api/analytics/calendar')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const days = buildDailySeries(items, range);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          days,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey === 'GET /api/applications') {
      const limit = parseInt(event.queryStringParameters?.limit || '0', 10) || 0;
      const items = await queryApplications(userId, { limit, scanForward: false, recordType: 'application' });
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({ items })
      };
    }

    if (routeKey === 'GET /api/prospects') {
      const limit = parseInt(event.queryStringParameters?.limit || '0', 10) || 0;
      const items = await queryApplications(userId, { limit, scanForward: false, recordType: 'prospect' });
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({ items })
      };
    }

    if (routeKey === 'POST /api/applications') {
      const payload = parseBody(event);
      const item = await handleCreateApplication(userId, payload);
      return {
        statusCode: 201,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(item)
      };
    }

    if (routeKey === 'POST /api/prospects') {
      const payload = parseBody(event);
      const item = await handleCreateProspect(userId, payload);
      return {
        statusCode: 201,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(item)
      };
    }

    if (routeKey.startsWith('PATCH /api/prospects')) {
      const id = path.split('/').pop();
      const payload = parseBody(event);
      const updated = await handleUpdateProspect(userId, id, payload);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(updated)
      };
    }

    if (routeKey.startsWith('PATCH /api/applications')) {
      const id = path.split('/').pop();
      const payload = parseBody(event);
      const updated = await handleUpdateApplication(userId, id, payload);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(updated)
      };
    }

    if (routeKey.startsWith('DELETE /api/applications')) {
      const id = path.split('/').pop();
      const result = await handleDeleteApplication(userId, id);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(result)
      };
    }

    if (routeKey === 'POST /api/attachments/presign') {
      const payload = parseBody(event);
      const data = await handlePresign(userId, payload);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(data)
      };
    }

    if (routeKey === 'POST /api/attachments/download') {
      const payload = parseBody(event);
      const data = await handlePresignDownload(userId, payload);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(data)
      };
    }

    return {
      statusCode: 404,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: 'Route not found.' })
    };
  } catch (err) {
    const statusCode = err.statusCode || 500;
    return {
      statusCode,
      headers: buildHeaders(corsOrigin),
      body: JSON.stringify({ error: err.message || 'Request failed.' })
    };
  }
};
