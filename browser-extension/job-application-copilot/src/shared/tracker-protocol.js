export const TRACKER_CHANNEL = 'danielshort.job-application-copilot';
export const TRACKER_PAGE_SOURCE = 'danielshort.job-tracker';
export const TRACKER_EXTENSION_SOURCE = 'danielshort.job-application-copilot.extension';
export const TRACKER_PROTOCOL_VERSION = 1;

export const TRACKER_MESSAGE_TYPES = Object.freeze({
  MANIFEST_REQUEST: 'capture-manifest-request',
  MANIFEST: 'capture-manifest',
  FILE_CHUNK_REQUEST: 'capture-file-chunk-request',
  FILE_CHUNK: 'capture-file-chunk',
  COMPLETE_ACK: 'capture-complete-ack',
  COMPLETE: 'capture-complete',
  DISMISSED: 'capture-dismissed'
});

export const TRACKER_INTERNAL_MESSAGE_TYPES = Object.freeze({
  BEGIN: 'tracker-capture-begin'
});

export const TRACKER_TRANSFER_LIMITS = Object.freeze({
  chunkSize: 262144,
  maxFiles: 2,
  maxFileBytes: 10 * 1024 * 1024,
  maxNonceLength: 128,
  minNonceLength: 16
});

const COMMON_KEYS = [
  'channel',
  'source',
  'protocolVersion',
  'type',
  'captureId',
  'requestId',
  'channelNonce'
];
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu;
const NONCE_PATTERN = /^[A-Za-z0-9_.~-]+$/u;
const SHA256_PATTERN = /^(?:[0-9a-f]{64}|[A-Za-z0-9_-]{43}|[A-Za-z0-9+/]{43}=)$/iu;
const MIME_PATTERN = /^[a-z0-9][a-z0-9!#$&^_.+-]*\/[a-z0-9][a-z0-9!#$&^_.+-]*$/iu;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/u;
const STANDARD_BASE64_PATTERN = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;

const isPlainObject = (value) => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value)
  && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

const fail = (message) => {
  throw new Error(`Invalid tracker protocol message: ${message}`);
};

const exactKeys = (value, expected, label) => {
  if (!isPlainObject(value)) fail(`${label} must be a plain object.`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    fail(`${label} has unexpected or missing keys.`);
  }
};

const boundedString = (value, label, { min = 0, max = 500 } = {}) => {
  if (typeof value !== 'string' || value.length < min || value.length > max) {
    fail(`${label} must be a string between ${min} and ${max} characters.`);
  }
  return value;
};

const uuid = (value, label) => {
  if (typeof value !== 'string' || !UUID_PATTERN.test(value)) fail(`${label} must be a UUID.`);
  return value;
};

const nonce = (value) => {
  boundedString(value, 'channelNonce', {
    min: TRACKER_TRANSFER_LIMITS.minNonceLength,
    max: TRACKER_TRANSFER_LIMITS.maxNonceLength
  });
  if (!NONCE_PATTERN.test(value)) fail('channelNonce contains unsupported characters.');
  return value;
};

const isoTimestamp = (value, label) => {
  boundedString(value, label, { min: 20, max: 40 });
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime()) || parsed.toISOString() !== value) {
    fail(`${label} must be a canonical ISO timestamp.`);
  }
  return value;
};

const optionalDate = (value, label) => {
  boundedString(value, label, { max: 10 });
  if (value && !DATE_PATTERN.test(value)) fail(`${label} must use YYYY-MM-DD.`);
};

const validateJob = (job) => {
  exactKeys(job, [
    'company',
    'title',
    'jobUrl',
    'location',
    'source',
    'postingDate',
    'appliedDate',
    'status',
    'tags'
  ], 'payload.job');
  boundedString(job.company, 'job.company', { min: 1, max: 200 });
  boundedString(job.title, 'job.title', { min: 1, max: 200 });
  boundedString(job.location, 'job.location', { max: 240 });
  boundedString(job.source, 'job.source', { max: 120 });
  boundedString(job.status, 'job.status', { min: 1, max: 20 });
  if (!['Applied', 'Screening', 'Interview', 'Offer', 'Rejected', 'Withdrawn'].includes(job.status)) {
    fail('job.status is not supported by the tracker.');
  }
  optionalDate(job.postingDate, 'job.postingDate');
  optionalDate(job.appliedDate, 'job.appliedDate');
  boundedString(job.jobUrl, 'job.jobUrl', { max: 2048 });
  if (job.jobUrl) {
    let parsed;
    try {
      parsed = new URL(job.jobUrl);
    } catch {
      fail('job.jobUrl must be a valid URL.');
    }
    if (!['http:', 'https:'].includes(parsed.protocol)) fail('job.jobUrl must use HTTP or HTTPS.');
  }
  if (!Array.isArray(job.tags) || job.tags.length > 20) fail('job.tags must be an array with at most 20 items.');
  const seen = new Set();
  job.tags.forEach((tag, index) => {
    boundedString(tag, `job.tags[${index}]`, { min: 1, max: 36 });
    const normalized = tag.toLocaleLowerCase('en-US');
    if (seen.has(normalized)) fail('job.tags must be unique without regard to case.');
    seen.add(normalized);
  });
  return job;
};

const validateManifestFile = (file, index) => {
  exactKeys(file, [
    'fileId',
    'name',
    'type',
    'kind',
    'size',
    'sha256',
    'totalChunks',
    'chunkSize'
  ], `payload.files[${index}]`);
  uuid(file.fileId, `files[${index}].fileId`);
  boundedString(file.name, `files[${index}].name`, { min: 1, max: 255 });
  boundedString(file.type, `files[${index}].type`, { min: 3, max: 200 });
  if (!MIME_PATTERN.test(file.type)) fail(`files[${index}].type must be a MIME type.`);
  boundedString(file.kind, `files[${index}].kind`, { min: 1, max: 64 });
  if (!['resume', 'cover-letter'].includes(file.kind)) fail(`files[${index}].kind is not supported.`);
  const lowerName = file.name.toLocaleLowerCase('en-US');
  const isPdf = file.type === 'application/pdf' && lowerName.endsWith('.pdf');
  const isDocx = file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    && lowerName.endsWith('.docx');
  if (!isPdf && !isDocx) fail(`files[${index}] must be a PDF or DOCX file with a matching extension.`);
  boundedString(file.sha256, `files[${index}].sha256`, { min: 43, max: 64 });
  if (!SHA256_PATTERN.test(file.sha256)) fail(`files[${index}].sha256 is invalid.`);
  if (!Number.isSafeInteger(file.size) || file.size < 1 || file.size > TRACKER_TRANSFER_LIMITS.maxFileBytes) {
    fail(`files[${index}].size is outside the supported range.`);
  }
  if (file.chunkSize !== TRACKER_TRANSFER_LIMITS.chunkSize) fail(`files[${index}].chunkSize is invalid.`);
  if (file.totalChunks !== Math.ceil(file.size / TRACKER_TRANSFER_LIMITS.chunkSize)) {
    fail(`files[${index}].totalChunks does not match the file size.`);
  }
  return file;
};

const validateManifestPayloadInternal = (payload) => {
  exactKeys(payload, ['issuedAt', 'expiresAt', 'job', 'files'], 'payload');
  isoTimestamp(payload.issuedAt, 'payload.issuedAt');
  isoTimestamp(payload.expiresAt, 'payload.expiresAt');
  if (new Date(payload.expiresAt) <= new Date(payload.issuedAt)) fail('payload.expiresAt must be after issuedAt.');
  validateJob(payload.job);
  if (!Array.isArray(payload.files) || payload.files.length > TRACKER_TRANSFER_LIMITS.maxFiles) {
    fail(`payload.files must contain at most ${TRACKER_TRANSFER_LIMITS.maxFiles} files.`);
  }
  const fileIds = new Set();
  const fileKinds = new Set();
  payload.files.forEach((file, index) => {
    validateManifestFile(file, index);
    if (fileIds.has(file.fileId)) fail('payload.files contains a duplicate fileId.');
    if (fileKinds.has(file.kind)) fail('payload.files contains a duplicate file kind.');
    fileIds.add(file.fileId);
    fileKinds.add(file.kind);
  });
  return payload;
};

const validateCommon = (message, expectedSource) => {
  boundedString(message.channel, 'channel', { min: 1, max: 100 });
  if (message.channel !== TRACKER_CHANNEL) fail('channel is not recognized.');
  if (message.source !== expectedSource) fail('source is not allowed for this message type.');
  if (message.protocolVersion !== TRACKER_PROTOCOL_VERSION) fail('protocolVersion is not supported.');
  uuid(message.captureId, 'captureId');
  uuid(message.requestId, 'requestId');
  nonce(message.channelNonce);
};

export const validateTrackerWindowMessage = (message, { expectedSource } = {}) => {
  if (!isPlainObject(message)) fail('message must be a plain object.');
  const source = expectedSource || message.source;
  const type = message.type;
  if (!Object.values(TRACKER_MESSAGE_TYPES).includes(type)) fail('type is not recognized.');

  if (type === TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST) {
    exactKeys(message, [...COMMON_KEYS, 'requestedAt'], 'message');
    if (source !== TRACKER_PAGE_SOURCE) fail('manifest requests must come from the tracker page.');
    validateCommon(message, TRACKER_PAGE_SOURCE);
    isoTimestamp(message.requestedAt, 'requestedAt');
  } else if (type === TRACKER_MESSAGE_TYPES.MANIFEST) {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_EXTENSION_SOURCE) fail('manifests must come from the extension.');
    validateCommon(message, TRACKER_EXTENSION_SOURCE);
    validateManifestPayloadInternal(message.payload);
  } else if (type === TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST) {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_PAGE_SOURCE) fail('chunk requests must come from the tracker page.');
    validateCommon(message, TRACKER_PAGE_SOURCE);
    exactKeys(message.payload, ['fileId', 'chunkIndex', 'chunkSize'], 'payload');
    uuid(message.payload.fileId, 'payload.fileId');
    if (!Number.isSafeInteger(message.payload.chunkIndex) || message.payload.chunkIndex < 0) {
      fail('payload.chunkIndex must be a non-negative integer.');
    }
    if (message.payload.chunkSize !== TRACKER_TRANSFER_LIMITS.chunkSize) fail('payload.chunkSize is invalid.');
  } else if (type === TRACKER_MESSAGE_TYPES.FILE_CHUNK) {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_EXTENSION_SOURCE) fail('chunks must come from the extension.');
    validateCommon(message, TRACKER_EXTENSION_SOURCE);
    exactKeys(message.payload, ['fileId', 'chunkIndex', 'totalChunks', 'data'], 'payload');
    uuid(message.payload.fileId, 'payload.fileId');
    if (!Number.isSafeInteger(message.payload.chunkIndex) || message.payload.chunkIndex < 0) {
      fail('payload.chunkIndex must be a non-negative integer.');
    }
    if (!Number.isSafeInteger(message.payload.totalChunks)
      || message.payload.totalChunks < 1
      || message.payload.chunkIndex >= message.payload.totalChunks) {
      fail('payload.totalChunks is invalid.');
    }
    boundedString(message.payload.data, 'payload.data', {
      min: 4,
      max: Math.ceil(TRACKER_TRANSFER_LIMITS.chunkSize / 3) * 4
    });
    if (!STANDARD_BASE64_PATTERN.test(message.payload.data)) fail('payload.data must be standard base64.');
  } else if (type === TRACKER_MESSAGE_TYPES.COMPLETE) {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_PAGE_SOURCE) fail('completion must come from the tracker page.');
    validateCommon(message, TRACKER_PAGE_SOURCE);
    exactKeys(message.payload, ['applicationId', 'completedAt'], 'payload');
    boundedString(message.payload.applicationId, 'payload.applicationId', { min: 1, max: 256 });
    isoTimestamp(message.payload.completedAt, 'payload.completedAt');
  } else if (type === TRACKER_MESSAGE_TYPES.COMPLETE_ACK) {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_EXTENSION_SOURCE) fail('completion acknowledgements must come from the extension.');
    validateCommon(message, TRACKER_EXTENSION_SOURCE);
    exactKeys(message.payload, ['accepted', 'applicationId', 'acknowledgedAt', 'reason'], 'payload');
    if (typeof message.payload.accepted !== 'boolean') fail('payload.accepted must be a boolean.');
    boundedString(message.payload.applicationId, 'payload.applicationId', { min: 1, max: 256 });
    isoTimestamp(message.payload.acknowledgedAt, 'payload.acknowledgedAt');
    boundedString(message.payload.reason, 'payload.reason', { max: 160 });
    if (message.payload.accepted && message.payload.reason) {
      fail('accepted completion acknowledgements cannot include a reason.');
    }
    if (!message.payload.accepted && !message.payload.reason) {
      fail('rejected completion acknowledgements require a reason.');
    }
  } else {
    exactKeys(message, [...COMMON_KEYS, 'payload'], 'message');
    if (source !== TRACKER_PAGE_SOURCE) fail('dismissal must come from the tracker page.');
    validateCommon(message, TRACKER_PAGE_SOURCE);
    exactKeys(message.payload, ['dismissedAt'], 'payload');
    isoTimestamp(message.payload.dismissedAt, 'payload.dismissedAt');
  }
  return message;
};

export const validateTrackerBeginMessage = (message) => {
  exactKeys(message, ['type', 'payload'], 'internal message');
  if (message.type !== TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN) fail('internal message type is not recognized.');
  exactKeys(message.payload, ['captureId', 'issuedAt', 'expiresAt', 'job', 'files'], 'internal payload');
  uuid(message.payload.captureId, 'payload.captureId');
  isoTimestamp(message.payload.issuedAt, 'payload.issuedAt');
  isoTimestamp(message.payload.expiresAt, 'payload.expiresAt');
  if (new Date(message.payload.expiresAt) <= new Date(message.payload.issuedAt)) {
    fail('payload.expiresAt must be after issuedAt.');
  }
  validateJob(message.payload.job);
  if (!Array.isArray(message.payload.files) || message.payload.files.length > TRACKER_TRANSFER_LIMITS.maxFiles) {
    fail(`payload.files must contain at most ${TRACKER_TRANSFER_LIMITS.maxFiles} files.`);
  }
  const fileIds = new Set();
  const fileKinds = new Set();
  message.payload.files.forEach((file, index) => {
    exactKeys(file, ['fileId', 'recordId', 'kind'], `payload.files[${index}]`);
    uuid(file.fileId, `files[${index}].fileId`);
    boundedString(file.recordId, `files[${index}].recordId`, { min: 1, max: 200 });
    boundedString(file.kind, `files[${index}].kind`, { min: 1, max: 64 });
    if (!['resume', 'cover-letter'].includes(file.kind)) fail(`files[${index}].kind is not supported.`);
    if (fileIds.has(file.fileId)) fail('payload.files contains a duplicate fileId.');
    if (fileKinds.has(file.kind)) fail('payload.files contains a duplicate file kind.');
    fileIds.add(file.fileId);
    fileKinds.add(file.kind);
  });
  return message;
};

export const isTrackerPageUrl = (value) => {
  try {
    const url = new URL(value);
    return url.protocol === 'https:'
      && url.hostname === 'www.danielshort.me'
      && url.pathname === '/tools/job-application-tracker'
      && !url.username
      && !url.password;
  } catch {
    return false;
  }
};

export const createChannelNonce = () => {
  const bytes = crypto.getRandomValues(new Uint8Array(24));
  let binary = '';
  bytes.forEach(value => { binary += String.fromCharCode(value); });
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/u, '');
};

export const validateManifestPayload = validateManifestPayloadInternal;
