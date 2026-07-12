/*
  Amazon Transcribe endpoints for the file transcription tool.

  Required env vars:
  - TRANSCRIBE_UPLOAD_BUCKET
  - TRANSCRIBE_SIGNING_SECRET
  - AWS_REGION (or AWS_DEFAULT_REGION)
  - Prefer TRANSCRIBE_AWS_ACCESS_KEY_ID / TRANSCRIBE_AWS_SECRET_ACCESS_KEY
*/
'use strict';

const crypto = require('crypto');
const { DeleteObjectCommand, HeadObjectCommand, S3Client } = require('@aws-sdk/client-s3');
const { createPresignedPost } = require('@aws-sdk/s3-presigned-post');
const {
  DeleteTranscriptionJobCommand,
  GetTranscriptionJobCommand,
  StartTranscriptionJobCommand,
  TranscribeClient
} = require('@aws-sdk/client-transcribe');
const { getBearerToken, readJson, sendJson } = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');
const {
  RUN_STATE,
  getRun,
  markJobCreated,
  markRunTerminal,
  refundRun,
  reserveRun
} = require('../transcribe-ledger');

const SUPPORTED_FORMATS = new Set(['mp3', 'mp4', 'wav', 'flac', 'ogg', 'webm', 'm4a', 'amr']);
const VIDEO_FORMATS = new Set(['mp4', 'webm']);
const MIN_DURATION_SECONDS = 15;
const DEFAULT_PRICE_PER_SECOND = 0.0004;
const DEFAULT_MAX_FILES_PER_RUN = 10;
const DEFAULT_MAX_FILE_BYTES = 500 * 1024 * 1024;
const DEFAULT_MAX_TOTAL_COST_USD = 10;
const DEFAULT_UPLOAD_TTL_SECONDS = 15 * 60;
const DEFAULT_TOKEN_TTL_SECONDS = 2 * 60 * 60;
const DEFAULT_RUN_TOKEN_TTL_SECONDS = 24 * 60 * 60;
const AWS_MAX_MEDIA_DURATION_SECONDS = 28_800;
const DEFAULT_DAILY_COST_LIMIT_USD = 25;
const DEFAULT_DAILY_FILE_LIMIT = 50;
const DEFAULT_GLOBAL_DAILY_COST_LIMIT_USD = 100;
const DEFAULT_GLOBAL_DAILY_FILE_LIMIT = 200;
const DEFAULT_MAX_CONCURRENT = 2;
const DEFAULT_LEDGER_START_LEASE_SECONDS = 5 * 60;
const DEFAULT_LEDGER_LEASE_SECONDS = 12 * 60 * 60;
const DEFAULT_LEDGER_RENEWAL_SECONDS = 60;
const DEFAULT_LEDGER_RUN_RETENTION_DAYS = 45;
const DEFAULT_LEDGER_DAY_RETENTION_DAYS = 45;
const DEFAULT_TRANSCRIPT_FETCH_TIMEOUT_MS = 12_000;
const DEFAULT_MAX_TRANSCRIPT_FETCH_BYTES = 25 * 1024 * 1024;
const DEFAULT_MAX_TRANSCRIPT_BYTES = 3 * 1024 * 1024;
const MAX_LEDGER_CONCURRENT = 10;
const MAX_TRANSCRIPT_FETCH_TIMEOUT_MS = 30_000;
const MAX_TRANSCRIPT_FETCH_BYTES = 50 * 1024 * 1024;
const MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024;
const MAX_FILENAME_CHARS = 120;

let cachedS3Client = null;
let cachedTranscribeClient = null;
let cachedClientKey = '';

function pickEnv(keys){
  for (const key of keys) {
    if (!key) continue;
    if (typeof process.env[key] === 'undefined') continue;
    const raw = String(process.env[key]);
    if (!raw.trim()) continue;
    return raw.trim();
  }
  return '';
}

function getAwsCredentialsFromEnv(){
  const accessKeyId = pickEnv(['TRANSCRIBE_AWS_ACCESS_KEY_ID', 'TOOLS_AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID']);
  const secretAccessKey = pickEnv(['TRANSCRIBE_AWS_SECRET_ACCESS_KEY', 'TOOLS_AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY']);
  const sessionToken = pickEnv(['TRANSCRIBE_AWS_SESSION_TOKEN', 'TOOLS_AWS_SESSION_TOKEN', 'AWS_SESSION_TOKEN']);
  if (!accessKeyId || !secretAccessKey) return null;
  const creds = { accessKeyId, secretAccessKey };
  if (sessionToken && accessKeyId.startsWith('ASIA')) creds.sessionToken = sessionToken;
  return creds;
}

function isProductionRuntime(){
  return String(process.env.VERCEL_ENV || '').trim().toLowerCase() === 'production' ||
    String(process.env.NODE_ENV || '').trim().toLowerCase() === 'production';
}

function getLedgerMode(){
  const requested = pickEnv(['TRANSCRIBE_LEDGER_MODE']).toLowerCase();
  if (requested && !['required', 'disabled'].includes(requested)) {
    const err = new Error('TRANSCRIBE_LEDGER_MODE must be "required" or "disabled".');
    err.code = 'TRANSCRIBE_ENV_MISSING';
    throw err;
  }
  if (requested === 'disabled' && !isProductionRuntime()) return 'disabled';
  return 'required';
}

function getConfig(){
  const region = pickEnv(['TRANSCRIBE_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']);
  const bucket = pickEnv(['TRANSCRIBE_UPLOAD_BUCKET']);
  const prefix = (pickEnv(['TRANSCRIBE_UPLOAD_PREFIX']) || 'tools-transcribe/').replace(/^\/+/, '').replace(/\/?$/, '/');
  const signingSecret = pickEnv(['TRANSCRIBE_SIGNING_SECRET']);
  const languageCode = pickEnv(['TRANSCRIBE_LANGUAGE_CODE']) || 'en-US';
  const pricePerSecond = positiveNumber(process.env.TRANSCRIBE_PRICE_PER_SECOND, DEFAULT_PRICE_PER_SECOND);
  const maxFilesPerRun = positiveInteger(process.env.TRANSCRIBE_MAX_FILES_PER_RUN, DEFAULT_MAX_FILES_PER_RUN);
  const maxFileBytes = positiveInteger(process.env.TRANSCRIBE_MAX_FILE_BYTES, DEFAULT_MAX_FILE_BYTES);
  const maxTotalCostUsd = positiveNumber(process.env.TRANSCRIBE_MAX_TOTAL_COST_USD, DEFAULT_MAX_TOTAL_COST_USD);
  const uploadTtlSeconds = positiveInteger(process.env.TRANSCRIBE_UPLOAD_TTL_SECONDS, DEFAULT_UPLOAD_TTL_SECONDS);
  const tokenTtlSeconds = positiveInteger(process.env.TRANSCRIBE_TOKEN_TTL_SECONDS, DEFAULT_TOKEN_TTL_SECONDS);
  const runTokenTtlSeconds = positiveInteger(process.env.TRANSCRIBE_RUN_TOKEN_TTL_SECONDS, DEFAULT_RUN_TOKEN_TTL_SECONDS);
  const ledgerMode = getLedgerMode();
  const ledgerEnabled = ledgerMode !== 'disabled';
  const ledgerTable = pickEnv(['TRANSCRIBE_DDB_TABLE', 'TOOLS_DDB_TABLE', 'TOOLS_DDB_TABLE_NAME']);
  const ledgerTtlAttributeRaw = pickEnv(['TRANSCRIBE_DDB_TTL_ATTRIBUTE', 'TOOLS_DDB_TTL_ATTRIBUTE']) || 'ttl';
  const ledgerTtlAttribute = /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(ledgerTtlAttributeRaw) ? ledgerTtlAttributeRaw : 'ttl';
  const ledgerDailyCostLimitUsd = positiveNumber(process.env.TRANSCRIBE_DAILY_COST_LIMIT_USD, DEFAULT_DAILY_COST_LIMIT_USD);
  const ledgerReservationCostUsd = calculateCostUsd(AWS_MAX_MEDIA_DURATION_SECONDS, pricePerSecond);
  const ledgerDailyFileLimit = positiveInteger(process.env.TRANSCRIBE_DAILY_FILE_LIMIT, DEFAULT_DAILY_FILE_LIMIT);
  const ledgerGlobalDailyCostLimitUsd = positiveNumber(
    process.env.TRANSCRIBE_GLOBAL_DAILY_COST_LIMIT_USD,
    DEFAULT_GLOBAL_DAILY_COST_LIMIT_USD
  );
  const ledgerGlobalDailyFileLimit = positiveInteger(
    process.env.TRANSCRIBE_GLOBAL_DAILY_FILE_LIMIT,
    DEFAULT_GLOBAL_DAILY_FILE_LIMIT
  );
  const ledgerMaxConcurrent = Math.min(
    positiveInteger(process.env.TRANSCRIBE_MAX_CONCURRENT, DEFAULT_MAX_CONCURRENT),
    MAX_LEDGER_CONCURRENT
  );
  const ledgerLeaseSeconds = positiveInteger(process.env.TRANSCRIBE_LEDGER_LEASE_SECONDS, DEFAULT_LEDGER_LEASE_SECONDS);
  const ledgerRenewalSeconds = positiveInteger(process.env.TRANSCRIBE_LEDGER_RENEWAL_SECONDS, DEFAULT_LEDGER_RENEWAL_SECONDS);
  const ledgerStartLeaseSeconds = positiveInteger(process.env.TRANSCRIBE_LEDGER_START_LEASE_SECONDS, DEFAULT_LEDGER_START_LEASE_SECONDS);
  const ledgerRunRetentionDays = positiveInteger(process.env.TRANSCRIBE_LEDGER_RUN_RETENTION_DAYS, DEFAULT_LEDGER_RUN_RETENTION_DAYS);
  const ledgerDayRetentionDays = positiveInteger(process.env.TRANSCRIBE_LEDGER_DAY_RETENTION_DAYS, DEFAULT_LEDGER_DAY_RETENTION_DAYS);
  const transcriptFetchTimeoutMs = Math.min(
    positiveInteger(process.env.TRANSCRIBE_TRANSCRIPT_FETCH_TIMEOUT_MS, DEFAULT_TRANSCRIPT_FETCH_TIMEOUT_MS),
    MAX_TRANSCRIPT_FETCH_TIMEOUT_MS
  );
  const maxTranscriptFetchBytes = Math.min(
    positiveInteger(process.env.TRANSCRIBE_MAX_TRANSCRIPT_FETCH_BYTES, DEFAULT_MAX_TRANSCRIPT_FETCH_BYTES),
    MAX_TRANSCRIPT_FETCH_BYTES
  );
  const maxTranscriptBytes = Math.min(
    positiveInteger(process.env.TRANSCRIBE_MAX_TRANSCRIPT_BYTES, DEFAULT_MAX_TRANSCRIPT_BYTES),
    MAX_TRANSCRIPT_BYTES
  );

  if (!region) {
    const err = new Error('AWS_REGION is not configured.');
    err.code = 'TRANSCRIBE_ENV_MISSING';
    throw err;
  }
  if (!bucket) {
    const err = new Error('TRANSCRIBE_UPLOAD_BUCKET is not configured.');
    err.code = 'TRANSCRIBE_ENV_MISSING';
    throw err;
  }
  if (!signingSecret || signingSecret.length < 24) {
    const err = new Error('TRANSCRIBE_SIGNING_SECRET is not configured.');
    err.code = 'TRANSCRIBE_ENV_MISSING';
    throw err;
  }
  if (ledgerEnabled && !ledgerTable) {
    const err = new Error('TRANSCRIBE_DDB_TABLE or TOOLS_DDB_TABLE is not configured.');
    err.code = 'TRANSCRIBE_ENV_MISSING';
    throw err;
  }

  return {
    region,
    bucket,
    prefix,
    signingSecret,
    languageCode,
    pricePerSecond,
    maxFilesPerRun,
    maxFileBytes,
    maxTotalCostUsd,
    uploadTtlSeconds,
    tokenTtlSeconds,
    runTokenTtlSeconds,
    ledgerEnabled,
    ledgerTable,
    ledgerTtlAttribute,
    ledgerDailyCostLimitUsd,
    ledgerDailyCostLimitMicros: Math.round(ledgerDailyCostLimitUsd * 1_000_000),
    ledgerReservationCostUsd,
    ledgerDailyFileLimit,
    ledgerGlobalDailyCostLimitMicros: Math.round(ledgerGlobalDailyCostLimitUsd * 1_000_000),
    ledgerGlobalDailyFileLimit,
    ledgerMaxConcurrent,
    ledgerStartLeaseSeconds,
    ledgerLeaseSeconds,
    ledgerRenewalSeconds,
    ledgerRunRetentionDays,
    ledgerDayRetentionDays,
    transcriptFetchTimeoutMs,
    maxTranscriptFetchBytes,
    maxTranscriptBytes,
    awsCredentials: getAwsCredentialsFromEnv()
  };
}

function getPublicConfig(){
  const region = pickEnv(['TRANSCRIBE_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
  const bucket = pickEnv(['TRANSCRIBE_UPLOAD_BUCKET']);
  const signingSecret = pickEnv(['TRANSCRIBE_SIGNING_SECRET']);
  let ledgerMode = 'required';
  let ledgerConfigValid = true;
  try {
    ledgerMode = getLedgerMode();
  } catch {
    ledgerConfigValid = false;
  }
  const ledgerTable = pickEnv(['TRANSCRIBE_DDB_TABLE', 'TOOLS_DDB_TABLE', 'TOOLS_DDB_TABLE_NAME']);
  const ledgerEnabled = ledgerMode !== 'disabled';
  return {
    region,
    languageCode: pickEnv(['TRANSCRIBE_LANGUAGE_CODE']) || 'en-US',
    pricePerSecond: positiveNumber(process.env.TRANSCRIBE_PRICE_PER_SECOND, DEFAULT_PRICE_PER_SECOND),
    maxFilesPerRun: positiveInteger(process.env.TRANSCRIBE_MAX_FILES_PER_RUN, DEFAULT_MAX_FILES_PER_RUN),
    maxFileBytes: positiveInteger(process.env.TRANSCRIBE_MAX_FILE_BYTES, DEFAULT_MAX_FILE_BYTES),
    maxTotalCostUsd: positiveNumber(process.env.TRANSCRIBE_MAX_TOTAL_COST_USD, DEFAULT_MAX_TOTAL_COST_USD),
    dailyCostLimitUsd: positiveNumber(process.env.TRANSCRIBE_DAILY_COST_LIMIT_USD, DEFAULT_DAILY_COST_LIMIT_USD),
    reservationCostUsd: calculateCostUsd(
      AWS_MAX_MEDIA_DURATION_SECONDS,
      positiveNumber(process.env.TRANSCRIBE_PRICE_PER_SECOND, DEFAULT_PRICE_PER_SECOND)
    ),
    maxServiceDurationSeconds: AWS_MAX_MEDIA_DURATION_SECONDS,
    dailyFileLimit: positiveInteger(process.env.TRANSCRIBE_DAILY_FILE_LIMIT, DEFAULT_DAILY_FILE_LIMIT),
    maxConcurrent: Math.min(positiveInteger(process.env.TRANSCRIBE_MAX_CONCURRENT, DEFAULT_MAX_CONCURRENT), MAX_LEDGER_CONCURRENT),
    configured: Boolean(ledgerConfigValid && region && bucket && signingSecret && signingSecret.length >= 24 && (!ledgerEnabled || ledgerTable))
  };
}

function positiveNumber(value, fallback){
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function positiveInteger(value, fallback){
  const n = Number.parseInt(String(value || ''), 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function getClients(){
  const config = getConfig();
  const creds = config.awsCredentials;
  const key = `${config.region}:${creds ? creds.accessKeyId : 'default'}`;
  if (cachedS3Client && cachedTranscribeClient && cachedClientKey === key) {
    return { s3: cachedS3Client, transcribe: cachedTranscribeClient, config };
  }
  cachedS3Client = new S3Client({
    region: config.region,
    credentials: creds || undefined,
    requestChecksumCalculation: 'WHEN_REQUIRED',
    responseChecksumValidation: 'WHEN_REQUIRED'
  });
  cachedTranscribeClient = new TranscribeClient({ region: config.region, credentials: creds || undefined });
  cachedClientKey = key;
  return { s3: cachedS3Client, transcribe: cachedTranscribeClient, config };
}

function base64UrlEncode(value){
  return Buffer.from(value).toString('base64url');
}

function base64UrlDecode(value){
  return Buffer.from(String(value || ''), 'base64url').toString('utf8');
}

function signToken(payload, secret){
  const body = base64UrlEncode(JSON.stringify(payload));
  const sig = crypto.createHmac('sha256', secret).update(body).digest('base64url');
  return `${body}.${sig}`;
}

function verifyToken(token, secret, expectedType){
  const raw = String(token || '').trim();
  const [body, sig] = raw.split('.');
  if (!body || !sig) {
    const err = new Error('Invalid token.');
    err.code = 'TOKEN_INVALID';
    throw err;
  }
  const expected = crypto.createHmac('sha256', secret).update(body).digest('base64url');
  const sigBuffer = Buffer.from(sig);
  const expectedBuffer = Buffer.from(expected);
  if (sigBuffer.length !== expectedBuffer.length || !crypto.timingSafeEqual(sigBuffer, expectedBuffer)) {
    const err = new Error('Invalid token.');
    err.code = 'TOKEN_INVALID';
    throw err;
  }
  let payload;
  try {
    payload = JSON.parse(base64UrlDecode(body));
  } catch {
    const err = new Error('Invalid token.');
    err.code = 'TOKEN_INVALID';
    throw err;
  }
  if (!payload || payload.type !== expectedType) {
    const err = new Error('Invalid token type.');
    err.code = 'TOKEN_INVALID';
    throw err;
  }
  if (!Number.isFinite(Number(payload.exp)) || Number(payload.exp) < Math.floor(Date.now() / 1000)) {
    const err = new Error('Token expired.');
    err.code = 'TOKEN_EXPIRED';
    throw err;
  }
  return payload;
}

async function requireUser(req, res){
  const token = getBearerToken(req);
  if (!token) {
    sendJson(res, 401, { ok: false, error: 'Sign in before transcribing files.' });
    return null;
  }
  try {
    const claims = await verifyCognitoIdToken(token);
    const sub = String(claims?.sub || '').trim();
    if (!sub) throw new Error('Missing subject.');
    return {
      sub,
      email: String(claims?.email || '').trim()
    };
  } catch (err) {
    if (err && err.code === 'COGNITO_ENV_MISSING') {
      sendJson(res, 503, { ok: false, error: err.message });
      return null;
    }
    sendJson(res, 401, { ok: false, error: 'Sign in before transcribing files.' });
    return null;
  }
}

function safeFilename(value){
  const raw = String(value || '').trim().replace(/\\/g, '/').split('/').pop() || 'media';
  const cleaned = raw.replace(/[^a-zA-Z0-9._ -]+/g, '_').replace(/\s+/g, ' ').trim();
  const clipped = cleaned.length > MAX_FILENAME_CHARS ? cleaned.slice(0, MAX_FILENAME_CHARS).trim() : cleaned;
  return clipped || 'media';
}

function safeS3Name(value){
  return safeFilename(value).replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'media';
}

function extensionFromName(value){
  const name = safeFilename(value).toLowerCase();
  const match = name.match(/\.([a-z0-9]+)$/);
  return match ? match[1] : '';
}

function normalizeContentType(value){
  const raw = String(value || '').split(';', 1)[0].trim().toLowerCase();
  if (!raw) return 'application/octet-stream';
  if (raw.length > 120) return 'application/octet-stream';
  return raw;
}

function calculateBillableSeconds(durationSeconds){
  const duration = Number(durationSeconds);
  if (!Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(MIN_DURATION_SECONDS, Math.ceil(duration));
}

function calculateCostUsd(durationSeconds, pricePerSecond){
  const billableSeconds = calculateBillableSeconds(durationSeconds);
  return Number((billableSeconds * pricePerSecond).toFixed(6));
}

function buildJobName(quote, secret){
  const identity = [
    String(quote && quote.sub || ''),
    String(quote && quote.bucket || ''),
    String(quote && quote.key || ''),
    String(quote && quote.iat || '')
  ].join('\n');
  const token = crypto.createHmac('sha256', secret).update(identity).digest('hex');
  return `site-transcribe-${token}`;
}

function buildQuoteHash(quote, secret){
  const canonical = [
    String(quote && quote.sub || ''),
    String(quote && quote.bucket || ''),
    String(quote && quote.key || ''),
    String(quote && quote.bytes || ''),
    String(quote && quote.billableSeconds || ''),
    String(quote && quote.estimatedCostUsd || ''),
    String(quote && quote.iat || '')
  ].join('\n');
  return crypto.createHmac('sha256', secret).update(canonical).digest('hex');
}

function isMissingObjectError(err){
  const status = Number(err && err.$metadata && err.$metadata.httpStatusCode);
  const name = String(err && (err.name || err.Code || err.code) || '').toLowerCase();
  return status === 404 || name === 'notfound' || name === 'nosuchkey';
}

function isDefinitiveStartRejection(err){
  const status = Number(err && err.$metadata && err.$metadata.httpStatusCode);
  const name = String(err && (err.name || err.Code || err.code) || '');
  if (name === 'ConflictException') return false;
  return [400, 401, 403, 429].includes(status) || [
    'AccessDeniedException',
    'BadRequestException',
    'CredentialsProviderError',
    'LimitExceededException',
    'ThrottlingException',
    'TooManyRequestsException',
    'UnauthorizedException',
    'ValidationException'
  ].includes(name);
}

async function inspectUploadedObject(s3, config, quote){
  let object;
  try {
    object = await s3.send(new HeadObjectCommand({
      Bucket: quote.bucket,
      Key: quote.key
    }));
  } catch (err) {
    const error = new Error(isMissingObjectError(err)
      ? 'Uploaded file was not found. Upload it again.'
      : 'Unable to verify the uploaded file.');
    error.code = isMissingObjectError(err) ? 'UPLOAD_NOT_FOUND' : 'UPLOAD_CHECK_FAILED';
    throw error;
  }

  const expectedBytes = Number(quote.bytes);
  const actualBytes = Number(object && object.ContentLength);
  if (!Number.isSafeInteger(actualBytes) || actualBytes <= 0 || actualBytes !== expectedBytes || actualBytes > config.maxFileBytes) {
    const error = new Error('Uploaded file size does not match the signed quote. Upload it again.');
    error.code = 'UPLOAD_SIZE_MISMATCH';
    throw error;
  }

  const expectedContentType = normalizeContentType(quote.contentType);
  const actualContentType = normalizeContentType(object && object.ContentType);
  if (actualContentType !== expectedContentType) {
    const error = new Error('Uploaded file type does not match the signed quote. Upload it again.');
    error.code = 'UPLOAD_TYPE_MISMATCH';
    throw error;
  }

  return { bytes: actualBytes, contentType: actualContentType };
}

function transcriptionJobMatches(job, jobName, config, quote){
  const expectedMediaUri = `s3://${quote.bucket}/${quote.key}`;
  return Boolean(
    job &&
    String(job.TranscriptionJobName || '') === jobName &&
    String(job.Media && job.Media.MediaFileUri || '') === expectedMediaUri &&
    String(job.MediaFormat || '').toLowerCase() === String(quote.format || '').toLowerCase() &&
    String(job.LanguageCode || '') === config.languageCode
  );
}

async function getExistingTranscriptionJob(transcribe, jobName){
  try {
    const out = await transcribe.send(new GetTranscriptionJobCommand({ TranscriptionJobName: jobName }));
    return out && out.TranscriptionJob || null;
  } catch (err) {
    const status = Number(err && err.$metadata && err.$metadata.httpStatusCode);
    const name = String(err && (err.name || err.Code || err.code) || '').toLowerCase();
    if (status === 404 || name === 'notfoundexception') return null;
    throw err;
  }
}

function buildRunPayload(quote, jobName, config){
  const quoteIat = Number(quote.iat);
  const now = Math.floor(Date.now() / 1000);
  return {
    ...quote,
    type: 'run',
    jobName,
    startedAt: quoteIat * 1000,
    iat: quoteIat,
    exp: Math.max(Number(quote.exp) || 0, now + config.runTokenTtlSeconds)
  };
}

function sendStartResult(res, runToken, jobName, quote, status, idempotent){
  sendJson(res, 200, {
    ok: true,
    runToken,
    jobName,
    status: status || 'QUEUED',
    idempotent: Boolean(idempotent),
    estimatedCostUsd: quote.estimatedCostUsd,
    billableSeconds: quote.billableSeconds
  });
}

function sendPendingConfirmation(res, runToken, jobName, quote, message){
  sendJson(res, 202, {
    ok: true,
    runToken,
    jobName,
    status: 'PENDING_CONFIRMATION',
    idempotent: true,
    estimatedCostUsd: quote.estimatedCostUsd,
    billableSeconds: quote.billableSeconds,
    message: message || 'AWS is still confirming the transcription job.'
  });
}

function publicConfig(config){
  return {
    ok: true,
    service: 'Amazon Transcribe',
    configured: config.configured !== false,
    region: config.region,
    languageCode: config.languageCode,
    pricePerSecond: config.pricePerSecond,
    pricePerMinute: Number((config.pricePerSecond * 60).toFixed(6)),
    minDurationSeconds: MIN_DURATION_SECONDS,
    minBillableSeconds: MIN_DURATION_SECONDS,
    maxFilesPerRun: config.maxFilesPerRun,
    maxFileBytes: config.maxFileBytes,
    maxTotalCostUsd: config.maxTotalCostUsd,
    dailyCostLimitUsd: config.dailyCostLimitUsd,
    reservationCostUsd: config.reservationCostUsd,
    maxServiceDurationSeconds: config.maxServiceDurationSeconds,
    dailyFileLimit: config.dailyFileLimit,
    maxConcurrent: config.maxConcurrent,
    supportedFormats: Array.from(SUPPORTED_FORMATS).sort()
  };
}

async function handleConfig(req, res){
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }
  sendJson(res, 200, publicConfig(getPublicConfig()));
}

async function handlePresign(req, res){
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const user = await requireUser(req, res);
  if (!user) return;

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON payload.' });
    return;
  }

  let clients;
  try {
    clients = getClients();
  } catch (err) {
    sendJson(res, err.code === 'TRANSCRIBE_ENV_MISSING' ? 503 : 500, {
      ok: false,
      error: err.message || 'Transcribe configuration is unavailable.'
    });
    return;
  }
  const { s3, config } = clients;

  const filename = safeFilename(body.filename || body.name || 'media');
  const format = extensionFromName(filename);
  const contentType = normalizeContentType(body.contentType || body.content_type);
  const bytes = Number(body.bytes || body.size || 0);
  const durationSeconds = Number(body.durationSeconds || body.duration_seconds || 0);

  if (!SUPPORTED_FORMATS.has(format)) {
    sendJson(res, 400, { ok: false, error: `Unsupported file format: ${format || 'unknown'}.` });
    return;
  }
  if (!Number.isSafeInteger(bytes) || bytes <= 0) {
    sendJson(res, 400, { ok: false, error: 'File size is required.' });
    return;
  }
  if (bytes > config.maxFileBytes) {
    sendJson(res, 413, { ok: false, error: `File exceeds ${config.maxFileBytes} bytes.` });
    return;
  }
  if (!Number.isFinite(durationSeconds) || durationSeconds < MIN_DURATION_SECONDS) {
    sendJson(res, 400, { ok: false, error: `File must be at least ${MIN_DURATION_SECONDS} seconds long.` });
    return;
  }

  const billableSeconds = calculateBillableSeconds(durationSeconds);
  const estimatedCostUsd = calculateCostUsd(durationSeconds, config.pricePerSecond);
  if (estimatedCostUsd > config.maxTotalCostUsd) {
    sendJson(res, 400, { ok: false, error: `Estimated cost exceeds $${config.maxTotalCostUsd}.` });
    return;
  }

  const now = Math.floor(Date.now() / 1000);
  const safeName = safeS3Name(filename);
  const key = `${config.prefix}${user.sub}/${Date.now()}-${crypto.randomBytes(10).toString('hex')}-${safeName}`;
  const quote = {
    type: 'quote',
    sub: user.sub,
    filename,
    format,
    contentType,
    bytes,
    durationSeconds: Number(durationSeconds.toFixed(3)),
    billableSeconds,
    estimatedCostUsd,
    bucket: config.bucket,
    key,
    iat: now,
    exp: now + config.tokenTtlSeconds
  };
  const quoteToken = signToken(quote, config.signingSecret);

  try {
    const uploadTagging = '<Tagging><TagSet><Tag><Key>tool</Key><Value>amazon-transcribe</Value></Tag><Tag><Key>retention</Key><Value>temporary</Value></Tag></TagSet></Tagging>';
    const upload = await createPresignedPost(s3, {
      Bucket: config.bucket,
      Key: key,
      Expires: Math.min(config.uploadTtlSeconds, 3600),
      Fields: {
        'Content-Type': contentType,
        tagging: uploadTagging
      },
      Conditions: [
        ['eq', '$Content-Type', contentType],
        ['eq', '$tagging', uploadTagging],
        ['content-length-range', bytes, bytes]
      ]
    });
    sendJson(res, 200, {
      ok: true,
      uploadUrl: upload.url,
      method: 'POST',
      fields: upload.fields,
      headers: {},
      quoteToken,
      quote: {
        filename,
        format,
        durationSeconds: quote.durationSeconds,
        billableSeconds,
        estimatedCostUsd
      }
    });
  } catch {
    sendJson(res, 500, { ok: false, error: 'Unable to prepare upload.' });
  }
}

async function handleStart(req, res){
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const user = await requireUser(req, res);
  if (!user) return;

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON payload.' });
    return;
  }

  let clients;
  try {
    clients = getClients();
  } catch (err) {
    sendJson(res, err.code === 'TRANSCRIBE_ENV_MISSING' ? 503 : 500, {
      ok: false,
      error: err.message || 'Transcribe configuration is unavailable.'
    });
    return;
  }
  const { s3, transcribe, config } = clients;

  let quote;
  try {
    quote = verifyToken(body.quoteToken, config.signingSecret, 'quote');
  } catch (err) {
    sendJson(res, err.code === 'TOKEN_EXPIRED' ? 410 : 400, { ok: false, error: err.message || 'Invalid quote.' });
    return;
  }
  const userPrefix = `${config.prefix}${user.sub}/`;
  if (quote.sub !== user.sub || quote.bucket !== config.bucket || !String(quote.key || '').startsWith(userPrefix)) {
    sendJson(res, 403, { ok: false, error: 'Quote does not belong to the signed-in user.' });
    return;
  }
  const quoteBytes = Number(quote.bytes);
  const quoteIat = Number(quote.iat);
  const quoteExp = Number(quote.exp);
  const quoteCost = Number(quote.estimatedCostUsd);
  if (
    !Number.isSafeInteger(quoteBytes) || quoteBytes <= 0 || quoteBytes > config.maxFileBytes ||
    !SUPPORTED_FORMATS.has(String(quote.format || '').toLowerCase()) ||
    !Number.isFinite(quoteIat) || !Number.isFinite(quoteExp) || quoteExp <= quoteIat ||
    !Number.isFinite(quoteCost) || quoteCost < 0
  ) {
    sendJson(res, 400, { ok: false, error: 'Quote contains invalid upload details.' });
    return;
  }
  if (quoteCost > config.maxTotalCostUsd) {
    sendJson(res, 400, { ok: false, error: `Estimated cost exceeds $${config.maxTotalCostUsd}.` });
    return;
  }

  const jobName = buildJobName(quote, config.signingSecret);
  const quoteHash = buildQuoteHash(quote, config.signingSecret);
  const runPayload = buildRunPayload(quote, jobName, config);
  const runToken = signToken(runPayload, config.signingSecret);
  try {
    const recordedRun = await getRun({ config, sub: user.sub, jobName });
    if (recordedRun && String(recordedRun.quoteHash || '') !== quoteHash) {
      sendJson(res, 409, { ok: false, error: 'A conflicting transcription reservation already exists.' });
      return;
    }
    const recordedState = String(recordedRun?.state || '').toUpperCase();
    if ([RUN_STATE.RUNNING, RUN_STATE.COMPLETED, RUN_STATE.FAILED, RUN_STATE.MISSING].includes(recordedState)) {
      sendStartResult(res, runToken, jobName, quote, recordedState, true);
      return;
    }
  } catch {
    sendJson(res, 503, { ok: false, error: 'Unable to check the transcription usage ledger. Retry this request.' });
    return;
  }

  try {
    await inspectUploadedObject(s3, config, quote);
  } catch (err) {
    if (err && (err.code === 'UPLOAD_SIZE_MISMATCH' || err.code === 'UPLOAD_TYPE_MISMATCH')) {
      await Promise.allSettled([
        s3.send(new DeleteObjectCommand({ Bucket: quote.bucket, Key: quote.key }))
      ]);
    }
    sendJson(res, err && err.code === 'UPLOAD_CHECK_FAILED' ? 502 : (err && err.code === 'UPLOAD_NOT_FOUND' ? 400 : 409), {
      ok: false,
      error: err && err.message || 'Unable to verify the uploaded file.'
    });
    return;
  }

  const attemptId = crypto.randomBytes(18).toString('base64url');
  let reservation;
  try {
    reservation = await reserveRun({
      config,
      sub: user.sub,
      jobName,
      quoteHash,
      // Browser-probed duration is useful UX but not a trust boundary. Reserve
      // the AWS hard maximum duration so a direct caller cannot understate cost.
      costUsd: config.ledgerReservationCostUsd,
      attemptId
    });
  } catch (err) {
    const code = String(err && err.code || '');
    const statusCode = code === 'LEDGER_RUN_CONFLICT'
      ? 409
      : ([
          'LEDGER_DAILY_COST_LIMIT',
          'LEDGER_DAILY_FILE_LIMIT',
          'LEDGER_GLOBAL_COST_LIMIT',
          'LEDGER_GLOBAL_FILE_LIMIT',
          'LEDGER_CONCURRENCY_LIMIT'
        ].includes(code) ? 429 : 503);
    sendJson(res, statusCode, {
      ok: false,
      error: err && err.message || 'Unable to reserve transcription capacity. Retry this request.'
    });
    return;
  }

  const mediaFileUri = `s3://${quote.bucket}/${quote.key}`;
  const reservationState = String(reservation?.record?.state || '').toUpperCase();
  if (reservation?.idempotent && [RUN_STATE.RUNNING, RUN_STATE.COMPLETED, RUN_STATE.FAILED, RUN_STATE.MISSING].includes(reservationState)) {
    sendStartResult(res, runToken, jobName, quote, reservationState, true);
    return;
  }
  if (!reservation?.startAllowed) {
    try {
      const existingJob = await getExistingTranscriptionJob(transcribe, jobName);
      if (existingJob && transcriptionJobMatches(existingJob, jobName, config, quote)) {
        await markJobCreated({ config, sub: user.sub, jobName, quoteHash });
        sendStartResult(
          res,
          runToken,
          jobName,
          quote,
          String(existingJob.TranscriptionJobStatus || 'QUEUED').toUpperCase(),
          true
        );
        return;
      }
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to confirm the existing transcription job. Retry this request.' });
      return;
    }
    sendJson(res, 409, {
      ok: false,
      error: 'This transcription job is already being created. Retry in a moment.'
    });
    return;
  }

  try {
    await transcribe.send(new StartTranscriptionJobCommand({
      TranscriptionJobName: jobName,
      LanguageCode: config.languageCode,
      MediaFormat: quote.format,
      Media: { MediaFileUri: mediaFileUri },
      Tags: [
        { Key: 'tool', Value: 'amazon-transcribe' },
        { Key: 'user', Value: crypto.createHmac('sha256', config.signingSecret).update(user.sub).digest('hex').slice(0, 32) }
      ]
    }));
    try {
      await markJobCreated({ config, sub: user.sub, jobName, quoteHash });
    } catch {
      sendPendingConfirmation(
        res,
        runToken,
        jobName,
        quote,
        'The transcription job started and its usage record is still being confirmed.'
      );
      return;
    }
    sendStartResult(res, runToken, jobName, quote, 'QUEUED', false);
  } catch (err) {
    let existingJob = null;
    try {
      existingJob = await getExistingTranscriptionJob(transcribe, jobName);
    } catch {
      sendPendingConfirmation(res, runToken, jobName, quote, 'AWS is still confirming the transcription job state.');
      return;
    }

    if (existingJob) {
      if (!transcriptionJobMatches(existingJob, jobName, config, quote)) {
        sendJson(res, 409, { ok: false, error: 'A conflicting transcription job already exists.' });
        return;
      }
      try {
        await markJobCreated({ config, sub: user.sub, jobName, quoteHash });
      } catch {
        sendPendingConfirmation(
          res,
          runToken,
          jobName,
          quote,
          'The transcription job exists and its usage record is still being confirmed.'
        );
        return;
      }
      sendStartResult(
        res,
        runToken,
        jobName,
        quote,
        String(existingJob.TranscriptionJobStatus || 'QUEUED').toUpperCase(),
        true
      );
      return;
    }

    if (err && err.name === 'ConflictException') {
      sendPendingConfirmation(res, runToken, jobName, quote, 'The transcription job is already being created.');
      return;
    }

    if (!isDefinitiveStartRejection(err)) {
      sendPendingConfirmation(
        res,
        runToken,
        jobName,
        quote,
        'AWS did not confirm the start response; polling will reconcile the reserved job.'
      );
      return;
    }

    try {
      await refundRun({
        config,
        sub: user.sub,
        jobName,
        quoteHash,
        attemptId,
        reason: String(err && err.name || 'START_FAILED')
      });
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to reconcile the failed transcription reservation. Retry this request.' });
      return;
    }

    if (err && err.name === 'BadRequestException') {
      await Promise.allSettled([
        s3.send(new DeleteObjectCommand({ Bucket: quote.bucket, Key: quote.key }))
      ]);
    }
    sendJson(res, err && err.name === 'BadRequestException' ? 400 : 500, {
      ok: false,
      error: err && err.name === 'BadRequestException'
        ? (err.message || 'Unable to start transcription job.')
        : 'Unable to start transcription job. Retry this request.'
    });
  }
}

async function cleanupRun({ s3, transcribe, run, deleteJob = false }){
  // Retain the deterministic job name through the quote lifetime so it remains
  // an AWS-enforced replay barrier. A later scheduled cleanup may opt in to deletion.
  const tasks = [
    s3.send(new DeleteObjectCommand({ Bucket: run.bucket, Key: run.key }))
  ];
  if (deleteJob) {
    tasks.push(transcribe.send(new DeleteTranscriptionJobCommand({ TranscriptionJobName: run.jobName })));
  }
  const results = await Promise.allSettled(tasks);
  return {
    uploadDeleted: results[0] && results[0].status === 'fulfilled',
    jobDeleted: deleteJob && results[1] && results[1].status === 'fulfilled',
    jobRetained: !deleteJob
  };
}

function extractTranscript(data){
  const transcripts = data && data.results && Array.isArray(data.results.transcripts)
    ? data.results.transcripts
    : [];
  const joined = transcripts
    .map((item) => String(item && item.transcript || '').trim())
    .filter(Boolean)
    .join('\n\n')
    .trim();
  return joined;
}

function isTrustedTranscriptUrl(value){
  try {
    const url = new URL(String(value || ''));
    const hostname = url.hostname.toLowerCase();
    return url.protocol === 'https:' && (
      hostname === 'amazonaws.com' ||
      hostname.endsWith('.amazonaws.com') ||
      hostname === 'amazonaws.com.cn' ||
      hostname.endsWith('.amazonaws.com.cn')
    );
  } catch {
    return false;
  }
}

async function fetchTranscriptData(uri, config){
  if (!isTrustedTranscriptUrl(uri)) throw new Error('Transcript URL is not trusted.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.transcriptFetchTimeoutMs);
  try {
    let currentUrl = String(uri);
    let transcriptRes;
    for (let redirects = 0; redirects <= 3; redirects += 1) {
      if (!isTrustedTranscriptUrl(currentUrl)) throw new Error('Transcript URL is not trusted.');
      transcriptRes = await fetch(currentUrl, {
        signal: controller.signal,
        redirect: 'manual'
      });
      if (![301, 302, 303, 307, 308].includes(transcriptRes.status)) break;
      const location = transcriptRes.headers.get('location');
      if (!location || redirects >= 3) throw new Error('Transcript redirect could not be followed safely.');
      currentUrl = new URL(location, currentUrl).toString();
    }
    if (!transcriptRes.ok) throw new Error(`Transcript fetch failed (${transcriptRes.status}).`);
    if (!isTrustedTranscriptUrl(transcriptRes.url || currentUrl)) throw new Error('Transcript redirect is not trusted.');
    const contentLength = Number(transcriptRes.headers.get('content-length'));
    if (Number.isFinite(contentLength) && contentLength > config.maxTranscriptFetchBytes) {
      throw new Error('Transcript response is too large.');
    }
    if (!transcriptRes.body || typeof transcriptRes.body.getReader !== 'function') {
      throw new Error('Transcript response could not be streamed safely.');
    }

    const reader = transcriptRes.body.getReader();
    const chunks = [];
    let bytes = 0;
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      const chunk = Buffer.from(part.value);
      bytes += chunk.length;
      if (bytes > config.maxTranscriptFetchBytes) {
        controller.abort();
        throw new Error('Transcript response is too large.');
      }
      chunks.push(chunk);
    }
    return JSON.parse(Buffer.concat(chunks, bytes).toString('utf8'));
  } finally {
    clearTimeout(timer);
  }
}

function isMissingTranscriptionJobError(err){
  const status = Number(err && err.$metadata && err.$metadata.httpStatusCode);
  const name = String(err && (err.name || err.Code || err.code) || '').toLowerCase();
  return status === 404 || name === 'notfoundexception';
}

async function getTranscriptionJobForStatus(transcribe, jobName){
  const retryDelays = [0, 250, 750];
  let lastError;
  for (const delayMs of retryDelays) {
    if (delayMs) await new Promise(resolve => setTimeout(resolve, delayMs));
    try {
      const out = await transcribe.send(new GetTranscriptionJobCommand({ TranscriptionJobName: jobName }));
      return out.TranscriptionJob;
    } catch (err) {
      lastError = err;
      if (!isMissingTranscriptionJobError(err)) throw err;
    }
  }
  throw lastError;
}

async function handleStatus(req, res){
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  const user = await requireUser(req, res);
  if (!user) return;

  let clients;
  try {
    clients = getClients();
  } catch (err) {
    sendJson(res, err.code === 'TRANSCRIBE_ENV_MISSING' ? 503 : 500, {
      ok: false,
      error: err.message || 'Transcribe configuration is unavailable.'
    });
    return;
  }
  const { s3, transcribe, config } = clients;

  const token = String((req.query && (req.query.run || req.query.runToken)) || '').trim();
  let run;
  try {
    run = verifyToken(token, config.signingSecret, 'run');
  } catch (err) {
    sendJson(res, err.code === 'TOKEN_EXPIRED' ? 410 : 400, { ok: false, error: err.message || 'Invalid run.' });
    return;
  }
  if (
    run.sub !== user.sub ||
    run.bucket !== config.bucket ||
    !String(run.key || '').startsWith(`${config.prefix}${user.sub}/`)
  ) {
    sendJson(res, 403, { ok: false, error: 'Run does not belong to the signed-in user.' });
    return;
  }
  const quoteHash = buildQuoteHash(run, config.signingSecret);

  let job;
  try {
    job = await getTranscriptionJobForStatus(transcribe, run.jobName);
  } catch (err) {
    if (!isMissingTranscriptionJobError(err)) {
      sendJson(res, 502, { ok: false, error: 'Unable to check the transcription job. Retry this request.' });
      return;
    }
    let ledgerRun;
    try {
      ledgerRun = await getRun({ config, sub: user.sub, jobName: run.jobName });
      const nowSeconds = Math.floor(Date.now() / 1000);
      const ledgerState = String(ledgerRun?.state || '').toUpperCase();
      const startAttemptExpiresAt = Number(ledgerRun?.startAttemptExpiresAt) || 0;
      const lastConfirmedAt = Math.max(
        Number(ledgerRun?.lastJobSeenAt) || 0,
        Number(ledgerRun?.jobCreatedAt) || 0
      );
      const withinStartGrace = ledgerState === RUN_STATE.RESERVED && startAttemptExpiresAt > nowSeconds;
      const withinSeenGrace = Boolean(ledgerRun?.jobCreated && lastConfirmedAt && nowSeconds - lastConfirmedAt < 60);
      if (withinStartGrace || withinSeenGrace) {
        sendJson(res, 200, {
          ok: true,
          status: withinStartGrace ? 'PENDING_CONFIRMATION' : 'IN_PROGRESS',
          estimatedCostUsd: run.estimatedCostUsd,
          billableSeconds: run.billableSeconds,
          durationSeconds: run.durationSeconds
        });
        return;
      }
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to confirm the missing transcription job. Retry this request.' });
      return;
    }
    if (
      String(ledgerRun?.state || '').toUpperCase() === RUN_STATE.RESERVED &&
      String(ledgerRun?.startAttemptId || '').trim()
    ) {
      try {
        await refundRun({
          config,
          sub: user.sub,
          jobName: run.jobName,
          quoteHash,
          attemptId: String(ledgerRun.startAttemptId),
          reason: 'JOB_NOT_FOUND_AFTER_START_GRACE'
        });
      } catch {
        sendJson(res, 503, { ok: false, error: 'Unable to reconcile the missing transcription reservation. Retry this request.' });
        return;
      }
      await cleanupRun({ s3, transcribe, run });
      sendJson(res, 404, { ok: false, status: RUN_STATE.MISSING, error: 'Transcription job was not found.' });
      return;
    }
    try {
      await markRunTerminal({
        config,
        sub: user.sub,
        jobName: run.jobName,
        quoteHash,
        status: RUN_STATE.MISSING
      });
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to finalize the missing transcription job. Retry this request.' });
      return;
    }
    await cleanupRun({ s3, transcribe, run });
    sendJson(res, 404, { ok: false, status: RUN_STATE.MISSING, error: 'Transcription job was not found.' });
    return;
  }

  try {
    await markJobCreated({ config, sub: user.sub, jobName: run.jobName, quoteHash });
  } catch {
    sendJson(res, 503, { ok: false, error: 'Unable to confirm the transcription usage lease. Retry this request.' });
    return;
  }

  const status = String(job && job.TranscriptionJobStatus || '').toUpperCase();
  if (status === 'COMPLETED') {
    try {
      await markRunTerminal({ config, sub: user.sub, jobName: run.jobName, quoteHash, status: RUN_STATE.COMPLETED });
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to finalize the completed transcription job. Retry this request.' });
      return;
    }
    const uri = String(job?.Transcript?.TranscriptFileUri || '').trim();
    let transcript = '';
    try {
      const data = await fetchTranscriptData(uri, config);
      transcript = extractTranscript(data);
      if (Buffer.byteLength(JSON.stringify(transcript), 'utf8') > config.maxTranscriptBytes) {
        const sizeError = new Error('Transcript is too large to return in one response.');
        sizeError.code = 'TRANSCRIPT_OUTPUT_TOO_LARGE';
        throw sizeError;
      }
    } catch (err) {
      await cleanupRun({ s3, transcribe, run });
      sendJson(res, err?.code === 'TRANSCRIPT_OUTPUT_TOO_LARGE' ? 413 : 500, {
        ok: false,
        error: err?.code === 'TRANSCRIPT_OUTPUT_TOO_LARGE'
          ? err.message
          : 'Unable to fetch completed transcript.'
      });
      return;
    }
    const cleanup = await cleanupRun({ s3, transcribe, run });
    sendJson(res, 200, {
      ok: true,
      status,
      transcript,
      costUsd: run.estimatedCostUsd,
      billableSeconds: run.billableSeconds,
      durationSeconds: run.durationSeconds,
      cleanedUp: cleanup.uploadDeleted,
      uploadCleanedUp: cleanup.uploadDeleted,
      jobRetained: cleanup.jobRetained
    });
    return;
  }

  if (status === 'FAILED') {
    try {
      await markRunTerminal({ config, sub: user.sub, jobName: run.jobName, quoteHash, status: RUN_STATE.FAILED });
    } catch {
      sendJson(res, 503, { ok: false, error: 'Unable to finalize the failed transcription job. Retry this request.' });
      return;
    }
    const cleanup = await cleanupRun({ s3, transcribe, run });
    sendJson(res, 200, {
      ok: true,
      status,
      error: String(job?.FailureReason || 'Transcription failed.'),
      costUsd: run.estimatedCostUsd,
      billableSeconds: run.billableSeconds,
      durationSeconds: run.durationSeconds,
      cleanedUp: cleanup.uploadDeleted,
      uploadCleanedUp: cleanup.uploadDeleted,
      jobRetained: cleanup.jobRetained
    });
    return;
  }

  sendJson(res, 200, {
    ok: true,
    status: status || 'IN_PROGRESS',
    estimatedCostUsd: run.estimatedCostUsd,
    billableSeconds: run.billableSeconds,
    durationSeconds: run.durationSeconds
  });
}

module.exports = async (req, res, segments = []) => {
  const action = String(Array.isArray(segments) ? segments[0] : '').trim().toLowerCase();
  if (action === 'config' || !action) return handleConfig(req, res);
  if (action === 'presign') return handlePresign(req, res);
  if (action === 'start') return handleStart(req, res);
  if (action === 'status') return handleStatus(req, res);
  sendJson(res, 404, { ok: false, error: 'Not Found' });
};
