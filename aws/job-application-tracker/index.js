const { createHash, randomUUID } = require('crypto');
const { once } = require('events');
const { PassThrough } = require('stream');
const { Upload } = require('@aws-sdk/lib-storage');
const { createPresignedPost } = require('@aws-sdk/s3-presigned-post');
const archiver = require('archiver');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
  UpdateCommand,
  DeleteCommand,
  QueryCommand
} = require('@aws-sdk/lib-dynamodb');
const {
  S3Client,
  GetObjectCommand,
  HeadObjectCommand,
  DeleteObjectsCommand,
  PutObjectTaggingCommand
} = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

const dynamo = DynamoDBDocumentClient.from(new DynamoDBClient({}), {
  marshallOptions: {
    removeUndefinedValues: true,
    convertEmptyValues: true
  }
});
const s3 = new S3Client({
  requestChecksumCalculation: 'when_required',
  responseChecksumValidation: 'when_required'
});

const {
  APPLICATIONS_TABLE,
  ATTACHMENTS_BUCKET,
  ALLOWED_ORIGINS = '',
  PRESIGN_TTL_SECONDS = '900',
  MAX_ATTACHMENT_BYTES = '10485760',
  MAX_ATTACHMENT_COUNT = '12',
  MAX_TAGS = '12',
  MAX_CUSTOM_FIELDS = '12',
  MAX_EXPORT_APPLICATIONS = '1000',
  MAX_EXPORT_ATTACHMENTS = '50',
  MAX_EXPORT_BYTES = '52428800',
  MAX_EXPORT_METADATA_BYTES = '8388608'
} = process.env;

const allowedOrigins = ALLOWED_ORIGINS.split(',').map(origin => origin.trim()).filter(Boolean);
const presignTtl = Math.max(parseInt(PRESIGN_TTL_SECONDS, 10) || 900, 60);
const maxAttachmentBytes = Math.max(parseInt(MAX_ATTACHMENT_BYTES, 10) || 10485760, 1048576);
const maxAttachmentCount = Math.max(parseInt(MAX_ATTACHMENT_COUNT, 10) || 12, 1);
const maxTags = Math.max(parseInt(MAX_TAGS, 10) || 12, 1);
const maxCustomFields = Math.max(parseInt(MAX_CUSTOM_FIELDS, 10) || 12, 1);
const maxExportApplications = Math.min(Math.max(parseInt(MAX_EXPORT_APPLICATIONS, 10) || 1000, 1), 1000);
const maxExportAttachments = Math.min(Math.max(parseInt(MAX_EXPORT_ATTACHMENTS, 10) || 50, 1), 50);
const maxExportBytes = Math.min(Math.max(parseInt(MAX_EXPORT_BYTES, 10) || 52428800, 1048576), 52428800);
const maxExportMetadataBytes = Math.min(
  Math.max(parseInt(MAX_EXPORT_METADATA_BYTES, 10) || 8388608, 65536),
  8388608
);
const MAX_LIST_PAGE_SIZE = 500;
const MAX_QUERY_PAGES = 25;
const MAX_INTERNAL_QUERY_LIMIT = 10_000;
const MAX_INTERNAL_QUERY_BYTES = 8 * 1024 * 1024;
const MAX_ANALYTICS_QUERY_BYTES = 4 * 1024 * 1024;
const MAX_CURSOR_CHARS = 2_048;
const STAGING_TAGS = [{ Key: 'purpose', Value: 'staging' }];
const ATTACHMENT_TAGS = [{ Key: 'purpose', Value: 'attachment' }];
const STAGING_TAG_XML = '<Tagging><TagSet><Tag><Key>purpose</Key><Value>staging</Value></Tag></TagSet></Tagging>';
const MAX_TAG_LENGTH = 36;
const MAX_CUSTOM_FIELD_KEY_LENGTH = 40;
const MAX_CUSTOM_FIELD_VALUE_LENGTH = 180;

const INTERVIEW_STATUSES = new Set([
  'screening',
  'screen',
  'phone screen',
  'interview',
  'panel',
  'onsite',
  'assessment'
]);
const SCREENING_STATUSES = new Set([
  'screening',
  'screen',
  'phone screen',
  'recruiter screen',
  'technical screen',
  'assessment'
]);
const OFFER_STATUSES = new Set(['offer', 'accepted']);
const REJECTION_STATUSES = new Set(['rejected', 'declined', 'no response', 'ghosted', 'withdrawn']);
const FOLLOW_UP_DAYS = {
  applied: 7,
  screening: 5,
  interview: 3,
  active: 5,
  interested: 3
};

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
  if (attachments.length > maxAttachmentCount) {
    throw httpError(400, `A maximum of ${maxAttachmentCount} attachments is allowed.`);
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
    const rawSize = attachment.size;
    const size = rawSize === undefined || rawSize === null || rawSize === ''
      ? null
      : Number(rawSize);
    if (size !== null && !Number.isSafeInteger(size)) {
      throw httpError(400, 'Attachment size must be an integer number of bytes.');
    }
    if (Number.isSafeInteger(size)) {
      if (size <= 0 || size > maxAttachmentBytes) {
        throw httpError(400, `Attachment size exceeds ${maxAttachmentBytes} bytes.`);
      }
    }
    safe.push({
      key,
      filename,
      contentType: (attachment.contentType || '').toString().trim(),
      kind: (attachment.kind || '').toString().trim(),
      uploadedAt: (attachment.uploadedAt || now).toString().trim(),
      ...(Number.isSafeInteger(size) ? { size } : {}),
      ...(String(attachment.etag || '').trim() ? { etag: String(attachment.etag).trim().replace(/^"|"$/g, '') } : {})
    });
  }
  return safe;
};

const verifyUploadedAttachments = async (attachments) => {
  if (!attachments.length) return [];
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  return Promise.all(attachments.map(async (attachment) => {
    let object;
    try {
      object = await s3.send(new HeadObjectCommand({
        Bucket: ATTACHMENTS_BUCKET,
        Key: attachment.key
      }));
    } catch {
      throw httpError(400, `Uploaded attachment not found: ${attachment.filename}`);
    }
    const actualSize = Number(object?.ContentLength);
    if (!Number.isFinite(actualSize) || actualSize <= 0 || actualSize > maxAttachmentBytes) {
      throw httpError(400, `Uploaded attachment exceeds ${maxAttachmentBytes} bytes.`);
    }
    if (Number.isFinite(Number(attachment.size)) && actualSize !== Number(attachment.size)) {
      throw httpError(400, `Uploaded attachment size mismatch: ${attachment.filename}`);
    }
    const expectedType = String(attachment.contentType || '').trim().toLowerCase();
    const actualType = String(object?.ContentType || '').trim().toLowerCase();
    if (expectedType && actualType && expectedType !== actualType) {
      throw httpError(400, `Uploaded attachment type mismatch: ${attachment.filename}`);
    }
    return {
      ...attachment,
      size: actualSize,
      ...(String(object?.ETag || '').trim()
        ? { etag: String(object.ETag).trim().replace(/^"|"$/g, '') }
        : {})
    };
  }));
};

const setAttachmentPurpose = async (attachments, purpose) => {
  if (!attachments.length) return;
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const tags = purpose === 'staging' ? STAGING_TAGS : ATTACHMENT_TAGS;
  for (const attachment of attachments) {
    await s3.send(new PutObjectTaggingCommand({
      Bucket: ATTACHMENTS_BUCKET,
      Key: attachment.key,
      Tagging: { TagSet: tags }
    }));
  }
};

const restageAttachmentsQuietly = async (attachments) => {
  if (!attachments.length) return;
  try {
    await setAttachmentPurpose(attachments, 'staging');
  } catch (err) {
    console.warn('Unable to restore staging tags after a failed attachment save.', err?.message || err);
  }
};

const deleteAttachmentKeys = async (keys) => {
  const uniqueKeys = Array.from(new Set((keys || []).filter(Boolean)));
  if (!uniqueKeys.length) return { deleted: 0, errors: [] };
  if (!ATTACHMENTS_BUCKET) return { deleted: 0, errors: uniqueKeys };
  try {
    const result = await s3.send(new DeleteObjectsCommand({
      Bucket: ATTACHMENTS_BUCKET,
      Delete: {
        Objects: uniqueKeys.map(Key => ({ Key })),
        Quiet: true
      }
    }));
    const errors = (result?.Errors || []).map(entry => String(entry?.Key || '')).filter(Boolean);
    return { deleted: uniqueKeys.length - errors.length, errors };
  } catch {
    return { deleted: 0, errors: uniqueKeys };
  }
};

const normalizePath = (event) => {
  const rawPath = event.rawPath || event.path || '';
  const stage = event.requestContext?.stage;
  if (stage && rawPath.startsWith(`/${stage}/`)) {
    const stripped = rawPath.slice(stage.length + 1);
    if (stripped.length > 1 && stripped.endsWith('/')) {
      return stripped.slice(0, -1);
    }
    return stripped;
  }
  if (rawPath.length > 1 && rawPath.endsWith('/')) {
    return rawPath.slice(0, -1);
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

const parseDateValue = (value) => {
  if (!value) return null;
  const parsed = parseDate(value);
  if (parsed) return parsed;
  const iso = new Date(value);
  return Number.isNaN(iso.getTime()) ? null : iso;
};

const normalizeTags = (tags) => {
  if (tags === undefined || tags === null) return [];
  let values = [];
  if (Array.isArray(tags)) {
    values = tags;
  } else if (typeof tags === 'string') {
    values = tags.split(/[;,]+/);
  } else {
    throw httpError(400, 'tags must be an array or string.');
  }
  const cleaned = [];
  const seen = new Set();
  values.forEach((tag) => {
    if (cleaned.length >= maxTags) return;
    const trimmed = (tag || '').toString().trim();
    if (!trimmed) return;
    const clipped = trimmed.slice(0, MAX_TAG_LENGTH);
    const key = clipped.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    cleaned.push(clipped);
  });
  return cleaned;
};

const normalizeCustomFields = (fields) => {
  if (fields === undefined || fields === null) return {};
  let entries = [];
  if (Array.isArray(fields)) {
    entries = fields.map(item => [item?.key, item?.value]);
  } else if (typeof fields === 'string') {
    try {
      const parsed = JSON.parse(fields);
      if (parsed && typeof parsed === 'object') {
        entries = Object.entries(parsed);
      }
    } catch {
      throw httpError(400, 'customFields must be an object or JSON string.');
    }
  } else if (typeof fields === 'object') {
    entries = Object.entries(fields);
  } else {
    throw httpError(400, 'customFields must be an object.');
  }

  const cleaned = {};
  let count = 0;
  entries.forEach(([key, value]) => {
    if (count >= maxCustomFields) return;
    const safeKey = (key || '').toString().trim().slice(0, MAX_CUSTOM_FIELD_KEY_LENGTH);
    if (!safeKey) return;
    const safeValue = (value ?? '').toString().trim().slice(0, MAX_CUSTOM_FIELD_VALUE_LENGTH);
    if (!safeValue) return;
    cleaned[safeKey] = safeValue;
    count += 1;
  });
  return cleaned;
};

const normalizeViewFilters = (filters) => {
  if (!filters || typeof filters !== 'object') return {};
  const allowKeys = [
    'query',
    'type',
    'statusGroup',
    'status',
    'source',
    'batch',
    'location',
    'start',
    'end',
    'tags',
    'sort',
    'view'
  ];
  const cleaned = {};
  allowKeys.forEach((key) => {
    if (filters[key] === undefined) return;
    if (key === 'tags') {
      cleaned.tags = normalizeTags(filters.tags);
      return;
    }
    const value = (filters[key] || '').toString().trim();
    if (value) cleaned[key] = value;
  });
  return cleaned;
};

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

const getCursorScope = ({ range, scanForward, recordType } = {}) => createHash('sha256')
  .update(JSON.stringify({
    recordType: recordType || '',
    start: range ? formatDate(range.start) : '',
    end: range ? formatDate(range.end) : '',
    scanForward: scanForward !== false
  }))
  .digest('hex')
  .slice(0, 16);

const encodeCursor = (key, scope) => {
  if (!key || typeof key !== 'object') return '';
  return Buffer.from(JSON.stringify({ ...key, scope }), 'utf8').toString('base64url');
};

const decodeCursor = (value, userId, scope) => {
  const raw = String(value || '').trim();
  if (!raw) return null;
  if (raw.length > MAX_CURSOR_CHARS) throw httpError(400, 'Invalid cursor.');
  try {
    const parsed = JSON.parse(Buffer.from(raw, 'base64url').toString('utf8'));
    if (
      !parsed ||
      parsed.userId !== userId ||
      !String(parsed.applicationId || '').trim() ||
      parsed.scope !== scope
    ) {
      throw new Error('Cursor scope mismatch.');
    }
    return { userId: parsed.userId, applicationId: parsed.applicationId };
  } catch {
    throw httpError(400, 'Invalid cursor.');
  }
};

const queryApplications = async (userId, { range, limit, scanForward, recordType, cursor, withCursor = false } = {}) => {
  const params = {
    TableName: APPLICATIONS_TABLE,
    KeyConditionExpression: 'userId = :userId',
    ExpressionAttributeValues: { ':userId': userId },
    ScanIndexForward: scanForward !== false
  };
  const filters = [];
  const names = {};
  const hasExplicitLimit = Number.isFinite(limit) && limit > 0;
  const targetLimit = hasExplicitLimit
    ? Math.min(withCursor ? MAX_LIST_PAGE_SIZE : MAX_INTERNAL_QUERY_LIMIT, Math.max(1, Math.floor(limit)))
    : (withCursor ? MAX_LIST_PAGE_SIZE : MAX_INTERNAL_QUERY_LIMIT);
  const resultByteLimit = hasExplicitLimit || withCursor ? MAX_INTERNAL_QUERY_BYTES : MAX_ANALYTICS_QUERY_BYTES;
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

  const cursorScope = getCursorScope({ range, scanForward, recordType });
  let items = [];
  let lastKey = decodeCursor(cursor, userId, cursorScope);
  let pages = 0;
  let resultBytes = 0;
  do {
    const remaining = targetLimit ? Math.max(1, targetLimit - items.length) : undefined;
    const result = await dynamo.send(new QueryCommand({
      ...params,
      ...(remaining ? { Limit: remaining } : {}),
      ...(lastKey ? { ExclusiveStartKey: lastKey } : {})
    }));
    const pageItems = result.Items || [];
    resultBytes += Buffer.byteLength(JSON.stringify(pageItems), 'utf8');
    if (resultBytes > resultByteLimit) {
      throw httpError(413, 'Too much application data to process at once. Narrow the date range.');
    }
    items = items.concat(pageItems);
    lastKey = result.LastEvaluatedKey;
    pages += 1;
    if (targetLimit && items.length >= targetLimit) break;
  } while (lastKey && pages < MAX_QUERY_PAGES);
  if (lastKey && !withCursor && !hasExplicitLimit) {
    throw httpError(413, 'Too many records to process at once. Narrow the date range.');
  }
  const result = {
    items: targetLimit ? items.slice(0, targetLimit) : items,
    nextCursor: lastKey ? encodeCursor(lastKey, cursorScope) : ''
  };
  return withCursor ? result : result.items;
};

const getApplication = async (userId, applicationId) => {
  if (!applicationId) throw httpError(400, 'Missing applicationId.');
  const result = await dynamo.send(new GetCommand({
    TableName: APPLICATIONS_TABLE,
    Key: { userId, applicationId }
  }));
  if (!result.Item) throw httpError(404, 'Application not found.');
  return result.Item;
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

const getEntryType = (item) => (item && item.recordType === 'prospect' ? 'prospect' : 'application');

const buildStatusTimeline = (entry = {}) => {
  const history = Array.isArray(entry.statusHistory) ? entry.statusHistory : [];
  const timeline = history
    .map(item => ({
      status: (item?.status || '').toString().trim(),
      date: parseDateValue(item?.date)
    }))
    .filter(item => item.status && item.date);

  if (!timeline.length) {
    const status = (entry.status || '').toString().trim();
    const fallbackDate = parseDateValue(entry.appliedDate || entry.captureDate || entry.updatedAt || entry.createdAt);
    if (status && fallbackDate) {
      timeline.push({ status, date: fallbackDate });
    }
  } else {
    const currentStatus = (entry.status || '').toString().trim();
    const last = timeline[timeline.length - 1];
    if (currentStatus && currentStatus !== last.status) {
      const fallbackDate = parseDateValue(entry.updatedAt || entry.createdAt || entry.appliedDate || entry.captureDate);
      if (fallbackDate) {
        timeline.push({ status: currentStatus, date: fallbackDate });
      }
    }
  }

  return timeline.sort((a, b) => a.date - b.date);
};

const getStageKey = (status = '') => {
  const key = status.toString().trim().toLowerCase();
  if (!key) return null;
  if (REJECTION_STATUSES.has(key)) return 'rejected';
  if (OFFER_STATUSES.has(key)) return 'offer';
  if (INTERVIEW_STATUSES.has(key)) return 'interview';
  if (SCREENING_STATUSES.has(key)) return 'screening';
  if (key === 'applied') return 'applied';
  return 'applied';
};

const getStageLabel = (stage) => {
  switch (stage) {
    case 'screening':
      return 'Screening';
    case 'interview':
      return 'Interview';
    case 'offer':
      return 'Offer';
    case 'rejected':
      return 'Rejected';
    default:
      return 'Applied';
  }
};

const buildFunnel = (items = []) => {
  const stageKeys = ['applied', 'screening', 'interview', 'offer', 'rejected'];
  const counts = stageKeys.reduce((acc, key) => ({ ...acc, [key]: 0 }), {});
  items.forEach((entry) => {
    const reached = new Set(['applied']);
    const timeline = buildStatusTimeline(entry);
    timeline.forEach((item) => {
      const stage = getStageKey(item.status);
      if (stage) reached.add(stage);
    });
    reached.forEach((stage) => {
      if (counts[stage] !== undefined) counts[stage] += 1;
    });
  });
  const base = counts.applied || items.length || 0;
  const stages = stageKeys.map(key => ({
    stage: getStageLabel(key),
    count: counts[key] || 0,
    rate: base ? (counts[key] / base) * 100 : 0
  }));
  const conversions = [
    ['applied', 'screening'],
    ['screening', 'interview'],
    ['interview', 'offer']
  ].map(([from, to]) => ({
    from: getStageLabel(from),
    to: getStageLabel(to),
    rate: counts[from] ? (counts[to] / counts[from]) * 100 : 0
  }));
  return { stages, conversions };
};

const buildTimeInStage = (items = []) => {
  const stageKeys = ['applied', 'screening', 'interview', 'offer'];
  const durations = stageKeys.reduce((acc, key) => ({ ...acc, [key]: [] }), {});
  items.forEach((entry) => {
    const timeline = buildStatusTimeline(entry);
    if (timeline.length < 2) return;
    for (let i = 0; i < timeline.length - 1; i += 1) {
      const current = timeline[i];
      const next = timeline[i + 1];
      const stage = getStageKey(current.status);
      if (!stage || !durations[stage] || !current.date || !next.date) continue;
      const diff = (next.date.getTime() - current.date.getTime()) / 86400000;
      if (Number.isFinite(diff) && diff >= 0) durations[stage].push(diff);
    }
  });
  const stages = stageKeys.map((key) => {
    const values = durations[key] || [];
    if (!values.length) {
      return { stage: getStageLabel(key), avgDays: null, medianDays: null, count: 0 };
    }
    const sorted = [...values].sort((a, b) => a - b);
    const sum = sorted.reduce((acc, val) => acc + val, 0);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return {
      stage: getStageLabel(key),
      avgDays: sum / sorted.length,
      medianDays: median,
      count: sorted.length
    };
  });
  return { stages, entriesAnalyzed: items.length };
};

const addDays = (value, days) => {
  if (!value && value !== 0) return null;
  const parsed = value instanceof Date ? value : parseDateValue(value);
  if (!parsed) return null;
  const due = new Date(parsed.getTime());
  due.setUTCDate(due.getUTCDate() + days);
  return due;
};

const deriveStatusDate = (entry) => {
  const timeline = buildStatusTimeline(entry);
  if (!timeline.length) return null;
  return timeline[timeline.length - 1].date || null;
};

const buildNextAction = (verb, dueDate) => {
  if (!verb) return { label: 'Follow up soon', tone: '', dueDate: null };
  if (!dueDate) return { label: `${verb} soon`, tone: '', dueDate: null };
  const dayMs = 86400000;
  const todayKey = Date.UTC(new Date().getUTCFullYear(), new Date().getUTCMonth(), new Date().getUTCDate());
  const dueKey = Date.UTC(dueDate.getUTCFullYear(), dueDate.getUTCMonth(), dueDate.getUTCDate());
  const diffDays = Math.round((dueKey - todayKey) / dayMs);
  const label = diffDays <= 0 ? `${verb} now` : `${verb} by ${formatDate(dueDate)}`;
  const tone = diffDays <= 0 ? 'danger' : diffDays <= 2 ? 'warning' : '';
  return { label, tone, dueDate };
};

const buildFollowUpAction = (entry) => {
  const entryType = getEntryType(entry);
  const statusKey = (entry?.status || '').toString().trim().toLowerCase();
  if (entryType === 'prospect') {
    if (statusKey === 'rejected' || statusKey === 'inactive') return null;
    const verb = statusKey === 'interested' ? 'Apply' : 'Review';
    const followUpDate = parseDateValue(entry.followUpDate);
    if (followUpDate) {
      return { ...buildNextAction(verb, followUpDate), source: 'manual' };
    }
    const baseDate = entry.captureDate || deriveStatusDate(entry);
    const offset = FOLLOW_UP_DAYS[statusKey] ?? 5;
    return { ...buildNextAction(verb, addDays(baseDate, offset)), source: 'auto' };
  }
  if (statusKey === 'offer') return { label: 'Review offer', tone: 'success', dueDate: null, source: 'auto' };
  if (statusKey === 'rejected' || statusKey === 'withdrawn') return null;
  const followUpDate = parseDateValue(entry.followUpDate);
  if (followUpDate) {
    return { ...buildNextAction('Follow up', followUpDate), source: 'manual' };
  }
  const baseDate = deriveStatusDate(entry) || entry.appliedDate;
  const offset = FOLLOW_UP_DAYS[statusKey] ?? 7;
  return { ...buildNextAction('Follow up', addDays(baseDate, offset)), source: 'auto' };
};

const sanitizeFilename = (value, fallback = 'attachment') => {
  const cleaned = (value || '')
    .toString()
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned || fallback;
};

const escapeCsv = (value) => {
  const text = (value ?? '').toString();
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
};

const buildExportData = (items, userId) => {
  const attachments = [];
  const exportItems = items.map((item) => {
    const applicationId = (item.applicationId || '').toString();
    const folder = sanitizeFilename(applicationId || 'application');
    const mappedAttachments = (Array.isArray(item.attachments) ? item.attachments : []).map((attachment, index) => {
      const key = (attachment?.key || '').toString().trim();
      const filename = (attachment?.filename || '').toString().trim();
      const safeFilename = sanitizeFilename(filename || `attachment-${index + 1}`);
      const exportPath = `attachments/${folder}/${String(index + 1).padStart(2, '0')}-${safeFilename}`;
      const size = Number.isFinite(Number(attachment?.size)) ? Number(attachment.size) : 0;
      const etag = String(attachment?.etag || '').trim().replace(/^"|"$/g, '');
      if (key && key.startsWith(`${userId}/`)) {
        attachments.push({ key, exportPath, size, etag });
      }
      return {
        key,
        filename,
        size,
        etag,
        contentType: (attachment?.contentType || '').toString().trim(),
        kind: (attachment?.kind || '').toString().trim(),
        uploadedAt: (attachment?.uploadedAt || '').toString().trim(),
        exportPath
      };
    });
    return {
      applicationId,
      company: (item.company || '').toString().trim(),
      title: (item.title || '').toString().trim(),
      jobUrl: (item.jobUrl || '').toString().trim(),
      location: (item.location || '').toString().trim(),
      source: (item.source || '').toString().trim(),
      batch: (item.batch || '').toString().trim(),
      postingDate: (item.postingDate || '').toString().trim(),
      captureDate: (item.captureDate || '').toString().trim(),
      appliedDate: (item.appliedDate || '').toString().trim(),
      status: (item.status || '').toString().trim(),
      notes: (item.notes || '').toString().trim(),
      tags: Array.isArray(item.tags) ? item.tags : [],
      followUpDate: (item.followUpDate || '').toString().trim(),
      followUpNote: (item.followUpNote || '').toString().trim(),
      customFields: item.customFields && typeof item.customFields === 'object' ? item.customFields : {},
      createdAt: (item.createdAt || '').toString().trim(),
      updatedAt: (item.updatedAt || '').toString().trim(),
      statusHistory: Array.isArray(item.statusHistory) ? item.statusHistory : [],
      attachments: mappedAttachments
    };
  });
  return { exportItems, attachments };
};

const buildExportCsv = (items) => {
  const headers = [
    'company',
    'title',
    'jobUrl',
    'location',
    'source',
    'postingDate',
    'captureDate',
    'appliedDate',
    'status',
    'batch',
    'notes',
    'tags',
    'followUpDate',
    'followUpNote',
    'customFields',
    'attachments'
  ];
  const rows = [headers.join(',')];
  items.forEach((item) => {
    const attachmentList = (Array.isArray(item.attachments) ? item.attachments : [])
      .map(attachment => attachment.exportPath || attachment.filename)
      .filter(Boolean)
      .join(' | ');
    const tags = Array.isArray(item.tags) ? item.tags.join(';') : '';
    const customFields = item.customFields ? JSON.stringify(item.customFields) : '';
    const values = [
      item.company,
      item.title,
      item.jobUrl,
      item.location,
      item.source,
      item.postingDate,
      item.captureDate,
      item.appliedDate,
      item.status,
      item.batch,
      item.notes,
      tags,
      item.followUpDate,
      item.followUpNote,
      customFields,
      attachmentList
    ];
    rows.push(values.map(escapeCsv).join(','));
  });
  return rows.join('\n');
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
  const postingDateRaw = payload.postingDate;
  const captureDateRaw = payload.captureDate;
  const statusDateRaw = payload.statusDate;
  const followUpDateRaw = payload.followUpDate;
  const jobUrl = normalizeUrl(payload.jobUrl || payload.url);
  const location = (payload.location || '').toString().trim();
  const source = (payload.source || '').toString().trim();
  const batch = (payload.batch || '').toString().trim();
  const tags = normalizeTags(payload.tags);
  const customFields = normalizeCustomFields(payload.customFields);
  const followUpNote = (payload.followUpNote || '').toString().trim();
  const requestedAttachments = normalizeAttachments(userId, payload.attachments);
  if (requestedAttachments.length) {
    throw httpError(400, 'Create the application before uploading attachments.');
  }
  const attachments = [];
  const now = nowIso();
  let statusHistoryDate = now;
  if (statusDateRaw !== undefined && statusDateRaw !== null && statusDateRaw !== '') {
    const parsedStatusDate = parseDate(statusDateRaw);
    if (!parsedStatusDate) throw httpError(400, 'statusDate must be YYYY-MM-DD.');
    statusHistoryDate = formatDate(parsedStatusDate);
  } else if (status.toLowerCase() === 'applied') {
    statusHistoryDate = formatDate(parsedDate);
  }
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
    statusHistory: [{ status, date: statusHistoryDate }],
    createdAt: now,
    updatedAt: now
  };
  if (postingDateRaw !== undefined && postingDateRaw !== null && postingDateRaw !== '') {
    const postingDate = parseDate(postingDateRaw);
    if (!postingDate) throw httpError(400, 'postingDate must be YYYY-MM-DD.');
    item.postingDate = formatDate(postingDate);
  }
  if (captureDateRaw !== undefined && captureDateRaw !== null && captureDateRaw !== '') {
    const captureDate = parseDate(captureDateRaw);
    if (!captureDate) throw httpError(400, 'captureDate must be YYYY-MM-DD.');
    item.captureDate = formatDate(captureDate);
  }
  if (jobUrl) item.jobUrl = jobUrl;
  if (location) item.location = location;
  if (source) item.source = source;
  if (batch) item.batch = batch;
  if (followUpDateRaw !== undefined && followUpDateRaw !== null && followUpDateRaw !== '') {
    const followUpDate = parseDate(followUpDateRaw);
    if (!followUpDate) throw httpError(400, 'followUpDate must be YYYY-MM-DD.');
    item.followUpDate = formatDate(followUpDate);
  }
  if (followUpNote) item.followUpNote = followUpNote;
  if (tags.length) item.tags = tags;
  if (Object.keys(customFields).length) item.customFields = customFields;
  if (attachments.length) item.attachments = attachments;
  await dynamo.send(new PutCommand({
    TableName: APPLICATIONS_TABLE,
    Item: item,
    ConditionExpression: 'attribute_not_exists(applicationId)'
  }));
  if (attachments.length) {
    try {
      await setAttachmentPurpose(attachments, 'attachment');
    } catch (err) {
      await dynamo.send(new DeleteCommand({
        TableName: APPLICATIONS_TABLE,
        Key: { userId, applicationId },
        ConditionExpression: 'attribute_exists(applicationId)'
      })).catch(() => {});
      await restageAttachmentsQuietly(attachments);
      throw httpError(502, `Application attachments could not be finalized: ${err.message || 'tagging failed'}`);
    }
  }
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
  const postingDateRaw = payload.postingDate;
  const captureDateRaw = payload.captureDate;
  const statusDateRaw = payload.statusDate;
  const followUpDateRaw = payload.followUpDate;
  const batch = (payload.batch || '').toString().trim();
  const tags = normalizeTags(payload.tags);
  const customFields = normalizeCustomFields(payload.customFields);
  const followUpNote = (payload.followUpNote || '').toString().trim();
  let captureDate = new Date();
  if (captureDateRaw) {
    const parsedCapture = parseDate(captureDateRaw);
    if (!parsedCapture) throw httpError(400, 'captureDate must be YYYY-MM-DD.');
    captureDate = parsedCapture;
  }
  const now = nowIso();
  let statusHistoryDate = now;
  if (statusDateRaw !== undefined && statusDateRaw !== null && statusDateRaw !== '') {
    const parsedStatusDate = parseDate(statusDateRaw);
    if (!parsedStatusDate) throw httpError(400, 'statusDate must be YYYY-MM-DD.');
    statusHistoryDate = formatDate(parsedStatusDate);
  }
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
    captureDate: formatDate(captureDate),
    statusHistory: [{ status, date: statusHistoryDate }],
    createdAt: now,
    updatedAt: now
  };
  if (postingDateRaw !== undefined && postingDateRaw !== null && postingDateRaw !== '') {
    const postingDate = parseDate(postingDateRaw);
    if (!postingDate) throw httpError(400, 'postingDate must be YYYY-MM-DD.');
    item.postingDate = formatDate(postingDate);
  }
  if (batch) item.batch = batch;
  if (followUpDateRaw !== undefined && followUpDateRaw !== null && followUpDateRaw !== '') {
    const followUpDate = parseDate(followUpDateRaw);
    if (!followUpDate) throw httpError(400, 'followUpDate must be YYYY-MM-DD.');
    item.followUpDate = formatDate(followUpDate);
  }
  if (followUpNote) item.followUpNote = followUpNote;
  if (tags.length) item.tags = tags;
  if (Object.keys(customFields).length) item.customFields = customFields;
  await dynamo.send(new PutCommand({
    TableName: APPLICATIONS_TABLE,
    Item: item
  }));
  return item;
};

const handleCreateView = async (userId, payload = {}) => {
  const name = (payload.name || '').toString().trim();
  if (!name) throw httpError(400, 'name is required.');
  const filters = normalizeViewFilters(payload.filters || {});
  const now = nowIso();
  const viewId = `VIEW#${Date.now()}#${randomUUID()}`;
  const item = {
    userId,
    applicationId: viewId,
    recordType: 'view',
    name: name.slice(0, 80),
    filters,
    createdAt: now,
    updatedAt: now
  };
  await dynamo.send(new PutCommand({
    TableName: APPLICATIONS_TABLE,
    Item: item
  }));
  return item;
};

const handleListViews = async (userId) => {
  const items = await queryApplications(userId, { recordType: 'view', scanForward: true });
  return items.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
};

const handleDeleteView = async (userId, viewId) => {
  if (!viewId) throw httpError(400, 'Missing viewId.');
  const item = await getApplication(userId, viewId);
  if (item.recordType !== 'view') throw httpError(400, 'Not a saved view.');
  await dynamo.send(new DeleteCommand({
    TableName: APPLICATIONS_TABLE,
    Key: { userId, applicationId: viewId }
  }));
  return { ok: true };
};

const handleUpdateProspect = async (userId, applicationId, payload = {}) => {
  if (!applicationId) throw httpError(400, 'Missing prospectId.');
  const updates = [];
  const removeFields = [];
  const names = {};
  const values = { ':updatedAt': nowIso() };
  const pushStatus = payload.status ? normalizeStatus(payload.status) : null;
  let statusHistoryDate = values[':updatedAt'];
  if (payload.statusDate !== undefined) {
    if (!pushStatus) throw httpError(400, 'statusDate requires status.');
    if (payload.statusDate) {
      const parsedStatusDate = parseDate(payload.statusDate);
      if (!parsedStatusDate) throw httpError(400, 'statusDate must be YYYY-MM-DD.');
      statusHistoryDate = formatDate(parsedStatusDate);
    }
  }

  const addField = (key, value) => {
    if (value === undefined || value === null) return;
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  };
  const addOptionalField = (key, value) => {
    if (value === undefined) return;
    const trimmed = (value || '').toString().trim();
    const nameKey = `#${key}`;
    names[nameKey] = key;
    if (!trimmed) {
      removeFields.push(nameKey);
      return;
    }
    const valueKey = `:${key}`;
    values[valueKey] = trimmed;
    updates.push(`${nameKey} = ${valueKey}`);
  };
  const addDateField = (key, value) => {
    if (value === undefined) return;
    if (value === null || value === '') {
      const nameKey = `#${key}`;
      names[nameKey] = key;
      removeFields.push(nameKey);
      return;
    }
    const parsed = parseDate(value);
    if (!parsed) throw httpError(400, `${key} must be YYYY-MM-DD.`);
    addField(key, formatDate(parsed));
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
  addOptionalField('batch', payload.batch);
  addDateField('postingDate', payload.postingDate);
  addDateField('captureDate', payload.captureDate);
  addDateField('followUpDate', payload.followUpDate);
  addOptionalField('followUpNote', payload.followUpNote);
  addField('notes', payload.notes?.toString().trim());
  if (payload.tags !== undefined) {
    const tags = normalizeTags(payload.tags);
    names['#tags'] = 'tags';
    if (tags.length) {
      values[':tags'] = tags;
      updates.push('#tags = :tags');
    } else {
      removeFields.push('#tags');
    }
  }
  if (payload.customFields !== undefined) {
    const customFields = normalizeCustomFields(payload.customFields);
    names['#customFields'] = 'customFields';
    if (Object.keys(customFields).length) {
      values[':customFields'] = customFields;
      updates.push('#customFields = :customFields');
    } else {
      removeFields.push('#customFields');
    }
  }
  if (pushStatus) {
    addField('status', pushStatus);
    names['#statusHistory'] = 'statusHistory';
    values[':empty'] = [];
    values[':statusEntry'] = [{ status: pushStatus, date: statusHistoryDate }];
    updates.push('#statusHistory = list_append(if_not_exists(#statusHistory, :empty), :statusEntry)');
  }

  updates.push('#updatedAt = :updatedAt');
  names['#updatedAt'] = 'updatedAt';

  let updateExpression = `SET ${updates.join(', ')}`;
  if (removeFields.length) {
    updateExpression += ` REMOVE ${removeFields.join(', ')}`;
  }

  names['#recordType'] = 'recordType';
  values[':expectedRecordType'] = 'prospect';
  let result;
  try {
    result = await dynamo.send(new UpdateCommand({
      TableName: APPLICATIONS_TABLE,
      Key: { userId, applicationId },
      UpdateExpression: updateExpression,
      ConditionExpression: 'attribute_exists(applicationId) AND #recordType = :expectedRecordType',
      ExpressionAttributeNames: names,
      ExpressionAttributeValues: values,
      ReturnValues: 'ALL_NEW'
    }));
  } catch (err) {
    if (err?.name === 'ConditionalCheckFailedException') throw httpError(404, 'Prospect not found.');
    throw err;
  }
  return result.Attributes;
};

const handleUpdateApplication = async (userId, applicationId, payload = {}) => {
  if (!applicationId) throw httpError(400, 'Missing applicationId.');
  const existingItem = await getApplication(userId, applicationId);
  if (existingItem.recordType && existingItem.recordType !== 'application') {
    throw httpError(404, 'Application not found.');
  }
  const updates = [];
  const removeFields = [];
  const names = {};
  const expectedUpdatedAt = String(payload.expectedUpdatedAt || '').trim().slice(0, 64);
  let nextUpdatedAt = nowIso();
  if (expectedUpdatedAt && nextUpdatedAt === expectedUpdatedAt) {
    nextUpdatedAt = new Date(Date.parse(expectedUpdatedAt) + 1).toISOString();
  }
  const values = { ':updatedAt': nextUpdatedAt };
  if (payload.attachments !== undefined && !expectedUpdatedAt) {
    throw httpError(428, 'expectedUpdatedAt is required when changing attachments.');
  }
  let verifiedAttachments = null;
  let newlyAttached = [];
  const pushStatus = payload.status ? normalizeStatus(payload.status) : null;
  let statusHistoryDate = values[':updatedAt'];
  if (payload.statusDate !== undefined) {
    if (!pushStatus) throw httpError(400, 'statusDate requires status.');
    if (payload.statusDate) {
      const parsedStatusDate = parseDate(payload.statusDate);
      if (!parsedStatusDate) throw httpError(400, 'statusDate must be YYYY-MM-DD.');
      statusHistoryDate = formatDate(parsedStatusDate);
    }
  }

  const addField = (key, value) => {
    if (value === undefined || value === null) return;
    const nameKey = `#${key}`;
    const valueKey = `:${key}`;
    names[nameKey] = key;
    values[valueKey] = value;
    updates.push(`${nameKey} = ${valueKey}`);
  };
  const addOptionalField = (key, value) => {
    if (value === undefined) return;
    const trimmed = (value || '').toString().trim();
    const nameKey = `#${key}`;
    names[nameKey] = key;
    if (!trimmed) {
      removeFields.push(nameKey);
      return;
    }
    const valueKey = `:${key}`;
    values[valueKey] = trimmed;
    updates.push(`${nameKey} = ${valueKey}`);
  };
  const addDateField = (key, value) => {
    if (value === undefined) return;
    if (value === null || value === '') {
      const nameKey = `#${key}`;
      names[nameKey] = key;
      removeFields.push(nameKey);
      return;
    }
    const parsed = parseDate(value);
    if (!parsed) throw httpError(400, `${key} must be YYYY-MM-DD.`);
    addField(key, formatDate(parsed));
  };

  addField('company', payload.company?.toString().trim());
  addField('title', payload.title?.toString().trim());
  if (payload.appliedDate) {
    const parsedDate = parseDate(payload.appliedDate);
    if (!parsedDate) throw httpError(400, 'appliedDate must be YYYY-MM-DD.');
    addField('appliedDate', formatDate(parsedDate));
  }
  addDateField('postingDate', payload.postingDate);
  addDateField('captureDate', payload.captureDate);
  addDateField('followUpDate', payload.followUpDate);
  if (payload.jobUrl !== undefined || payload.url !== undefined) {
    const jobUrl = normalizeUrl(payload.jobUrl || payload.url);
    addField('jobUrl', jobUrl);
  }
  addField('location', payload.location?.toString().trim());
  addField('source', payload.source?.toString().trim());
  addOptionalField('batch', payload.batch);
  addOptionalField('followUpNote', payload.followUpNote);
  addField('notes', payload.notes?.toString().trim());
  if (payload.tags !== undefined) {
    const tags = normalizeTags(payload.tags);
    names['#tags'] = 'tags';
    if (tags.length) {
      values[':tags'] = tags;
      updates.push('#tags = :tags');
    } else {
      removeFields.push('#tags');
    }
  }
  if (payload.customFields !== undefined) {
    const customFields = normalizeCustomFields(payload.customFields);
    names['#customFields'] = 'customFields';
    if (Object.keys(customFields).length) {
      values[':customFields'] = customFields;
      updates.push('#customFields = :customFields');
    } else {
      removeFields.push('#customFields');
    }
  }
  if (payload.attachments !== undefined) {
    const attachments = normalizeAttachments(userId, payload.attachments);
    verifiedAttachments = await verifyUploadedAttachments(attachments);
    const existingAttachments = Array.isArray(existingItem.attachments) ? existingItem.attachments : [];
    const existingKeys = new Set(existingAttachments.map(attachment => attachment?.key).filter(Boolean));
    newlyAttached = verifiedAttachments.filter(attachment => !existingKeys.has(attachment.key));
    const expectedStagingPrefix = `${userId}/staging/${applicationId}/`;
    if (newlyAttached.some(attachment => !attachment.key.startsWith(expectedStagingPrefix))) {
      throw httpError(400, 'New attachments must use a staging upload created for this application.');
    }
    if (verifiedAttachments.length) {
      try {
        await setAttachmentPurpose(verifiedAttachments, 'attachment');
      } catch (err) {
        await restageAttachmentsQuietly(newlyAttached);
        throw httpError(502, `Application attachments could not be finalized: ${err.message || 'tagging failed'}`);
      }
    }
    addField('attachments', verifiedAttachments);
  }
  if (pushStatus) {
    addField('status', pushStatus);
    names['#statusHistory'] = 'statusHistory';
    values[':empty'] = [];
    values[':statusEntry'] = [{ status: pushStatus, date: statusHistoryDate }];
    updates.push('#statusHistory = list_append(if_not_exists(#statusHistory, :empty), :statusEntry)');
  }

  updates.push('#updatedAt = :updatedAt');
  names['#updatedAt'] = 'updatedAt';

  let updateExpression = `SET ${updates.join(', ')}`;
  if (removeFields.length) {
    updateExpression += ` REMOVE ${removeFields.join(', ')}`;
  }

  names['#recordType'] = 'recordType';
  values[':expectedRecordType'] = 'application';
  if (expectedUpdatedAt) values[':expectedUpdatedAt'] = expectedUpdatedAt;
  const conditionExpression = expectedUpdatedAt
    ? 'attribute_exists(applicationId) AND (attribute_not_exists(#recordType) OR #recordType = :expectedRecordType) AND #updatedAt = :expectedUpdatedAt'
    : 'attribute_exists(applicationId) AND (attribute_not_exists(#recordType) OR #recordType = :expectedRecordType)';
  let result;
  try {
    result = await dynamo.send(new UpdateCommand({
      TableName: APPLICATIONS_TABLE,
      Key: { userId, applicationId },
      UpdateExpression: updateExpression,
      ConditionExpression: conditionExpression,
      ExpressionAttributeNames: names,
      ExpressionAttributeValues: values,
      ReturnValues: 'ALL_NEW'
    }));
  } catch (err) {
    await restageAttachmentsQuietly(newlyAttached);
    if (err?.name === 'ConditionalCheckFailedException') {
      if (expectedUpdatedAt) throw httpError(409, 'Application changed in another session. Refresh and try again.');
      throw httpError(404, 'Application not found.');
    }
    throw err;
  }
  if (verifiedAttachments) {
    const nextKeys = new Set(verifiedAttachments.map(attachment => attachment.key));
    const removedKeys = (Array.isArray(existingItem.attachments) ? existingItem.attachments : [])
      .map(attachment => String(attachment?.key || '').trim())
      .filter(key => key.startsWith(`${userId}/`) && !nextKeys.has(key));
    const cleanup = await deleteAttachmentKeys(removedKeys);
    if (cleanup.errors.length) {
      await restageAttachmentsQuietly(cleanup.errors.map(key => ({ key })));
      console.warn(`Unable to clean up ${cleanup.errors.length} replaced attachment(s).`);
    }
  }
  return result.Attributes;
};

const handleDeleteApplication = async (userId, applicationId) => {
  if (!applicationId) throw httpError(400, 'Missing applicationId.');
  let deleted;
  try {
    deleted = await dynamo.send(new DeleteCommand({
      TableName: APPLICATIONS_TABLE,
      Key: { userId, applicationId },
      ConditionExpression: 'attribute_exists(applicationId) AND (attribute_not_exists(#recordType) OR #recordType = :application)',
      ExpressionAttributeNames: { '#recordType': 'recordType' },
      ExpressionAttributeValues: { ':application': 'application' },
      ReturnValues: 'ALL_OLD'
    }));
  } catch (err) {
    if (err?.name === 'ConditionalCheckFailedException') throw httpError(404, 'Application not found.');
    throw err;
  }
  const item = deleted.Attributes || {};
  const attachmentKeys = (Array.isArray(item.attachments) ? item.attachments : [])
    .map(attachment => String(attachment?.key || '').trim())
    .filter(key => key && key.startsWith(`${userId}/`));
  const cleanup = await deleteAttachmentKeys(attachmentKeys);
  if (cleanup.errors.length) {
    await restageAttachmentsQuietly(cleanup.errors.map(key => ({ key })));
  }
  return {
    ok: true,
    deletedAttachments: cleanup.deleted,
    cleanupPending: cleanup.errors.length > 0
  };
};

const handleDeleteProspect = async (userId, prospectId) => {
  if (!prospectId) throw httpError(400, 'Missing prospectId.');
  try {
    await dynamo.send(new DeleteCommand({
      TableName: APPLICATIONS_TABLE,
      Key: { userId, applicationId: prospectId },
      ConditionExpression: 'attribute_exists(applicationId) AND #recordType = :prospect',
      ExpressionAttributeNames: { '#recordType': 'recordType' },
      ExpressionAttributeValues: { ':prospect': 'prospect' }
    }));
  } catch (err) {
    if (err?.name === 'ConditionalCheckFailedException') throw httpError(404, 'Prospect not found.');
    throw err;
  }
  return { ok: true };
};

const handlePresign = async (userId, payload = {}) => {
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const applicationId = (payload.applicationId || '').toString().trim();
  const filename = (payload.filename || '').toString().trim();
  const contentType = (payload.contentType || 'application/octet-stream').toString().trim();
  const size = Number(payload.size);
  if (!applicationId || !filename) {
    throw httpError(400, 'applicationId and filename are required.');
  }
  const application = await getApplication(userId, applicationId);
  if (application.recordType && application.recordType !== 'application') throw httpError(404, 'Application not found.');
  if ((Array.isArray(application.attachments) ? application.attachments.length : 0) >= maxAttachmentCount) {
    throw httpError(400, `A maximum of ${maxAttachmentCount} attachments is allowed.`);
  }
  if (!Number.isSafeInteger(size) || size <= 0) {
    throw httpError(400, 'size is required.');
  }
  if (size > maxAttachmentBytes) {
    throw httpError(400, `Attachment exceeds ${maxAttachmentBytes} bytes.`);
  }
  const safeName = sanitizeFilename(filename, 'attachment').slice(0, 180);
  const key = `${userId}/staging/${applicationId}/${randomUUID()}-${safeName}`;
  const fields = {
    'Content-Type': contentType,
    'success_action_status': '201',
    tagging: STAGING_TAG_XML
  };
  const presigned = await createPresignedPost(s3, {
    Bucket: ATTACHMENTS_BUCKET,
    Key: key,
    Fields: fields,
    Conditions: [
      { 'Content-Type': contentType },
      { 'success_action_status': '201' },
      { tagging: STAGING_TAG_XML },
      ['content-length-range', size, size]
    ],
    Expires: presignTtl
  });
  return {
    uploadUrl: presigned.url,
    uploadMethod: 'POST',
    fields: presigned.fields,
    key,
    bucket: ATTACHMENTS_BUCKET,
    expiresIn: presignTtl,
    maxBytes: maxAttachmentBytes,
    maxCount: maxAttachmentCount
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

const getExportRange = (payload = {}) => {
  const startValue = payload.start;
  const endValue = payload.end;
  if (startValue && !parseDate(startValue)) throw httpError(400, 'start must be YYYY-MM-DD.');
  if (endValue && !parseDate(endValue)) throw httpError(400, 'end must be YYYY-MM-DD.');
  return getRange({
    start: startValue,
    end: endValue
  });
};

const exportRevision = (values) => createHash('sha256')
  .update(JSON.stringify(values || []))
  .digest('hex')
  .slice(0, 12);

const validateExportBudget = (applicationCount, attachmentCount) => {
  if (applicationCount > maxExportApplications) {
    throw httpError(413, `Export exceeds ${maxExportApplications} applications. Narrow the date range.`);
  }
  if (attachmentCount > maxExportAttachments) {
    throw httpError(413, `Export exceeds ${maxExportAttachments} attachments. Narrow the date range.`);
  }
};

const validateExportMetadata = (...values) => {
  const bytes = values.reduce((total, value) => total + Buffer.byteLength(String(value || ''), 'utf8'), 0);
  if (bytes > maxExportMetadataBytes) {
    throw httpError(413, `Export metadata exceeds ${maxExportMetadataBytes} bytes. Narrow the date range.`);
  }
  if (bytes > maxExportBytes) {
    throw httpError(413, `Export input exceeds ${maxExportBytes} bytes. Narrow the date range.`);
  }
  return bytes;
};

const preflightExportItemsMetadata = (items) => {
  let bytes = 0;
  for (const item of items) {
    bytes += Buffer.byteLength(JSON.stringify(item), 'utf8');
    if (bytes > maxExportMetadataBytes) {
      throw httpError(413, `Export metadata exceeds ${maxExportMetadataBytes} bytes. Narrow the date range.`);
    }
  }
  return bytes;
};

const isMissingObjectError = (err) => {
  const status = Number(err?.$metadata?.httpStatusCode);
  return status === 403 || status === 404 || err?.name === 'NotFound' || err?.name === 'NoSuchKey';
};

const inspectExportAttachments = async (attachments, metadataBytes = 0) => {
  const available = [];
  const missing = [];
  const signature = [];
  let attachmentBytes = 0;
  for (const attachment of attachments) {
    let object;
    try {
      object = await s3.send(new HeadObjectCommand({
        Bucket: ATTACHMENTS_BUCKET,
        Key: attachment.key
      }));
    } catch (err) {
      if (!isMissingObjectError(err)) throw err;
      missing.push(attachment);
      signature.push([attachment.key, 'missing']);
      continue;
    }
    const size = Number(object?.ContentLength);
    if (!Number.isFinite(size) || size <= 0 || size > maxAttachmentBytes) {
      throw httpError(413, `Attachment ${attachment.key} exceeds the per-file export limit.`);
    }
    attachmentBytes += size;
    if (metadataBytes + attachmentBytes > maxExportBytes) {
      throw httpError(413, `Export input exceeds ${maxExportBytes} bytes. Narrow the date range.`);
    }
    const ifMatch = String(object?.ETag || '').trim();
    const etag = ifMatch.replace(/^"|"$/g, '');
    available.push({ ...attachment, size, etag, ifMatch });
    signature.push([attachment.key, size, etag]);
  }
  return { available, missing, signature, attachmentBytes };
};

const startArchiveUpload = (key, metadata) => {
  const uploadStream = new PassThrough();
  const archive = archiver('zip', { zlib: { level: 6 } });
  archive.pipe(uploadStream);
  const uploader = new Upload({
    client: s3,
    leavePartsOnError: false,
    params: {
      Bucket: ATTACHMENTS_BUCKET,
      Key: key,
      Body: uploadStream,
      ContentType: 'application/zip',
      Tagging: 'purpose=export',
      Metadata: metadata
    }
  });
  const uploadResult = uploader.done().then(
    () => ({ error: null }),
    error => ({ error })
  );
  const archiveResult = new Promise((resolve) => {
    let settled = false;
    const finish = (error = null) => {
      if (settled) return;
      settled = true;
      resolve({ error });
    };
    archive.on('warning', err => {
      if (err.code !== 'ENOENT') finish(err);
    });
    archive.on('error', finish);
    uploadStream.on('error', finish);
    uploadStream.on('finish', () => finish());
    uploadStream.on('close', () => finish());
  });
  return { archive, uploadStream, uploader, uploadResult, archiveResult };
};

const appendBodyToArchive = async (archive, body, name) => {
  if (!body || typeof body[Symbol.asyncIterator] !== 'function') {
    throw new Error(`Attachment body is not streamable: ${name}`);
  }
  const entry = new PassThrough();
  archive.append(entry, { name });
  try {
    for await (const chunk of body) {
      if (!entry.write(chunk)) await once(entry, 'drain');
    }
    entry.end();
  } catch (err) {
    entry.destroy();
    if (typeof body.destroy === 'function') body.destroy();
    throw err;
  }
};

const abortArchiveUpload = async (context, error) => {
  try { context.archive.abort(); } catch {}
  if (!context.uploadStream.destroyed) context.uploadStream.destroy(error);
  try { await context.uploader.abort(); } catch {}
  await Promise.all([context.uploadResult, context.archiveResult]);
};

const finishArchiveUpload = async (context) => {
  await context.archive.finalize();
  const [upload, archived] = await Promise.all([context.uploadResult, context.archiveResult]);
  if (archived.error) throw archived.error;
  if (upload.error) throw upload.error;
};

const getCachedExport = async (key) => {
  try {
    const object = await s3.send(new HeadObjectCommand({ Bucket: ATTACHMENTS_BUCKET, Key: key }));
    return object || null;
  } catch (err) {
    const status = Number(err?.$metadata?.httpStatusCode);
    if (status === 404 || err?.name === 'NotFound' || err?.name === 'NoSuchKey') return null;
    throw err;
  }
};

const signExportDownload = async (key) => getSignedUrl(
  s3,
  new GetObjectCommand({ Bucket: ATTACHMENTS_BUCKET, Key: key }),
  { expiresIn: presignTtl }
);

const handleCreateExport = async (userId, range) => {
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const items = await queryApplications(userId, {
    range,
    recordType: 'application',
    limit: maxExportApplications + 1
  });
  const sorted = [...items].sort((a, b) => (a.appliedDate || '').localeCompare(b.appliedDate || ''));
  const { exportItems, attachments } = buildExportData(sorted, userId);
  validateExportBudget(exportItems.length, attachments.length);
  preflightExportItemsMetadata(exportItems);
  const exportPayload = {
    generatedAt: nowIso(),
    start: formatDate(range.start),
    end: formatDate(range.end),
    totalApplications: exportItems.length,
    items: exportItems
  };
  const exportJson = JSON.stringify(exportPayload, null, 2);
  const csv = buildExportCsv(exportItems);
  const metadataBytes = validateExportMetadata(exportJson, csv);
  const inspected = await inspectExportAttachments(attachments, metadataBytes);
  const revision = exportRevision([
    sorted.map(item => [item.applicationId, item.updatedAt, item.attachments]),
    inspected.signature
  ]);
  const key = `${userId}/exports/job-applications-${formatDate(range.start)}-to-${formatDate(range.end)}-${revision}.zip`;
  const cached = await getCachedExport(key);
  if (cached) {
    return {
      downloadUrl: await signExportDownload(key),
      key,
      expiresIn: presignTtl,
      start: exportPayload.start,
      end: exportPayload.end,
      totalApplications: exportItems.length,
      attachmentsExported: Number(cached.Metadata?.attachmentsexported) || 0,
      attachmentsMissing: Number(cached.Metadata?.attachmentsmissing) || 0,
      inputBytes: Number(cached.Metadata?.inputbytes) || metadataBytes,
      cached: true
    };
  }
  const inputBytes = metadataBytes + inspected.attachmentBytes;
  const context = startArchiveUpload(key, {
    totalapplications: String(exportItems.length),
    attachmentsexported: String(inspected.available.length),
    attachmentsmissing: String(inspected.missing.length),
    inputbytes: String(inputBytes)
  });
  try {
    context.archive.append(exportJson, { name: 'applications.json' });
    context.archive.append(csv, { name: 'applications.csv' });
    for (const attachment of inspected.available) {
      const response = await s3.send(new GetObjectCommand({
        Bucket: ATTACHMENTS_BUCKET,
        Key: attachment.key,
        ...(attachment.ifMatch ? { IfMatch: attachment.ifMatch } : {})
      }));
      await appendBodyToArchive(context.archive, response.Body, attachment.exportPath);
    }
    await finishArchiveUpload(context);
  } catch (err) {
    await abortArchiveUpload(context, err);
    if (err?.statusCode) throw err;
    throw httpError(502, `Unable to create export: ${err.message || 'archive failed'}`);
  }

  const downloadUrl = await signExportDownload(key);
  return {
    downloadUrl,
    key,
    expiresIn: presignTtl,
    start: exportPayload.start,
    end: exportPayload.end,
    totalApplications: exportItems.length,
    attachmentsExported: inspected.available.length,
    attachmentsMissing: inspected.missing.length,
    inputBytes,
    cached: false
  };
};

const handleCreateAttachmentZip = async (userId, payload = {}) => {
  if (!ATTACHMENTS_BUCKET) throw httpError(500, 'Attachments bucket not configured.');
  const applicationId = (payload.applicationId || '').toString().trim();
  if (!applicationId) throw httpError(400, 'applicationId is required.');
  const item = await getApplication(userId, applicationId);
  if (item.recordType && item.recordType !== 'application') throw httpError(404, 'Application not found.');
  const attachments = normalizeAttachments(userId, Array.isArray(item.attachments) ? item.attachments : []);
  if (!attachments.length) throw httpError(400, 'No attachments to zip.');
  validateExportBudget(1, attachments.length);

  const label = sanitizeFilename(
    [item.company, item.title].filter(Boolean).join('-'),
    sanitizeFilename(applicationId, 'entry')
  );
  const metadataBytes = validateExportMetadata(JSON.stringify(attachments.map(attachment => ({
    key: attachment.key,
    filename: attachment.filename
  }))));
  const inspected = await inspectExportAttachments(attachments, metadataBytes);
  const revision = exportRevision([item.applicationId, item.updatedAt, inspected.signature]);
  const key = `${userId}/exports/attachments-${sanitizeFilename(applicationId, 'entry')}-${revision}.zip`;
  const cached = await getCachedExport(key);
  if (cached) {
    return {
      downloadUrl: await signExportDownload(key),
      key,
      expiresIn: presignTtl,
      attachmentsExported: Number(cached.Metadata?.attachmentsexported) || 0,
      attachmentsMissing: Number(cached.Metadata?.attachmentsmissing) || 0,
      inputBytes: Number(cached.Metadata?.inputbytes) || metadataBytes,
      cached: true
    };
  }
  const inputBytes = metadataBytes + inspected.attachmentBytes;
  const context = startArchiveUpload(key, {
    attachmentsexported: String(inspected.available.length),
    attachmentsmissing: String(inspected.missing.length),
    inputbytes: String(inputBytes)
  });
  try {
    for (let index = 0; index < inspected.available.length; index += 1) {
      const attachment = inspected.available[index];
      const attachmentKey = (attachment?.key || '').toString().trim();
      const filename = (attachment?.filename || '').toString().trim();
      const safeFilename = sanitizeFilename(filename, `attachment-${index + 1}`);
      const exportPath = `${label}/${String(index + 1).padStart(2, '0')}-${safeFilename}`;
      const response = await s3.send(new GetObjectCommand({
        Bucket: ATTACHMENTS_BUCKET,
        Key: attachmentKey,
        ...(attachment.ifMatch ? { IfMatch: attachment.ifMatch } : {})
      }));
      await appendBodyToArchive(context.archive, response.Body, exportPath);
    }
    await finishArchiveUpload(context);
  } catch (err) {
    await abortArchiveUpload(context, err);
    if (err?.statusCode) throw err;
    throw httpError(502, `Unable to create attachment archive: ${err.message || 'archive failed'}`);
  }

  const downloadUrl = await signExportDownload(key);
  return {
    downloadUrl,
    key,
    expiresIn: presignTtl,
    attachmentsExported: inspected.available.length,
    attachmentsMissing: inspected.missing.length,
    inputBytes,
    cached: false
  };
};

const handleFollowUps = async (userId, query = {}) => {
  const range = getRange({ start: query.start, end: query.end });
  const includeOverdue = query.includeOverdue !== 'false';
  const [applications, prospects] = await Promise.all([
    queryApplications(userId, { recordType: 'application' }),
    queryApplications(userId, { recordType: 'prospect' })
  ]);
  const followUps = [];
  const items = applications.concat(prospects);
  items.forEach((entry) => {
    const action = buildFollowUpAction(entry);
    if (!action || !action.dueDate) return;
    const dueDate = action.dueDate;
    if (dueDate > range.end) return;
    const isOverdue = dueDate < range.start;
    if (isOverdue && !includeOverdue) return;
    followUps.push({
      applicationId: entry.applicationId,
      entryType: getEntryType(entry),
      company: (entry.company || '').toString().trim(),
      title: (entry.title || '').toString().trim(),
      status: (entry.status || '').toString().trim(),
      jobUrl: (entry.jobUrl || '').toString().trim(),
      followUpDate: (entry.followUpDate || '').toString().trim(),
      followUpNote: (entry.followUpNote || '').toString().trim(),
      dueDate: formatDate(dueDate),
      actionLabel: action.label,
      actionTone: action.tone,
      actionSource: action.source,
      overdue: isOverdue
    });
  });
  followUps.sort((a, b) => a.dueDate.localeCompare(b.dueDate));
  return {
    items: followUps,
    start: formatDate(range.start),
    end: formatDate(range.end),
    total: followUps.length
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

    if (routeKey.startsWith('GET /api/analytics/dashboard')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          start: formatDate(range.start),
          end: formatDate(range.end),
          summary: buildSummary(items),
          timeline: { series: buildDailySeries(items, range) },
          statuses: { statuses: buildStatusBreakdown(items) },
          calendar: { days: buildDailySeries(items, range) },
          funnel: buildFunnel(items),
          timeInStage: buildTimeInStage(items),
          applications: { items }
        })
      };
    }

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

    if (routeKey.startsWith('GET /api/analytics/funnel')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const data = buildFunnel(items);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          ...data,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey.startsWith('GET /api/analytics/time-in-stage')) {
      const range = getRange(event.queryStringParameters || {});
      const items = await queryApplications(userId, { range, recordType: 'application' });
      const data = buildTimeInStage(items);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({
          ...data,
          start: formatDate(range.start),
          end: formatDate(range.end)
        })
      };
    }

    if (routeKey.startsWith('GET /api/analytics/followups')) {
      const data = await handleFollowUps(userId, event.queryStringParameters || {});
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(data)
      };
    }

    if (routeKey === 'GET /api/applications') {
      const query = event.queryStringParameters || {};
      const requestedLimit = parseInt(query.limit || '', 10);
      const limit = Number.isFinite(requestedLimit) && requestedLimit > 0
        ? Math.min(requestedLimit, MAX_LIST_PAGE_SIZE)
        : MAX_LIST_PAGE_SIZE;
      const range = (query.start || query.end) ? getRange(query) : null;
      const result = await queryApplications(userId, {
        limit,
        scanForward: false,
        recordType: 'application',
        range,
        cursor: query.cursor,
        withCursor: true
      });
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(result)
      };
    }

    if (routeKey === 'GET /api/views') {
      const items = await handleListViews(userId);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify({ items })
      };
    }

    if (routeKey === 'GET /api/prospects') {
      const query = event.queryStringParameters || {};
      const requestedLimit = parseInt(query.limit || '', 10);
      const limit = Number.isFinite(requestedLimit) && requestedLimit > 0
        ? Math.min(requestedLimit, MAX_LIST_PAGE_SIZE)
        : MAX_LIST_PAGE_SIZE;
      const result = await queryApplications(userId, {
        limit,
        scanForward: false,
        recordType: 'prospect',
        cursor: query.cursor,
        withCursor: true
      });
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(result)
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

    if (routeKey === 'POST /api/views') {
      const payload = parseBody(event);
      const item = await handleCreateView(userId, payload);
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

    if (routeKey.startsWith('DELETE /api/views')) {
      const id = path.split('/').pop();
      const result = await handleDeleteView(userId, id);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(result)
      };
    }

    if (routeKey === 'POST /api/exports' || (method === 'POST' && path === '/api/exports')) {
      const payload = parseBody(event);
      const range = getExportRange(payload);
      const data = await handleCreateExport(userId, range);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(data)
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

    if (routeKey.startsWith('DELETE /api/prospects')) {
      const id = path.split('/').pop();
      const result = await handleDeleteProspect(userId, id);
      return {
        statusCode: 200,
        headers: buildHeaders(corsOrigin),
        body: JSON.stringify(result)
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

    if (routeKey === 'POST /api/attachments/zip') {
      const payload = parseBody(event);
      const data = await handleCreateAttachmentZip(userId, payload);
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
