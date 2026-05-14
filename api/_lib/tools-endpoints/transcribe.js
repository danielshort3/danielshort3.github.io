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
const { DeleteObjectCommand, PutObjectCommand, S3Client } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const {
  DeleteTranscriptionJobCommand,
  GetTranscriptionJobCommand,
  StartTranscriptionJobCommand,
  TranscribeClient
} = require('@aws-sdk/client-transcribe');
const { getBearerToken, readJson, sendJson } = require('../tools-api');
const { verifyCognitoIdToken } = require('../cognito-jwt');

const SUPPORTED_FORMATS = new Set(['mp3', 'mp4', 'wav', 'flac', 'ogg', 'webm', 'm4a', 'amr']);
const VIDEO_FORMATS = new Set(['mp4', 'webm']);
const MIN_DURATION_SECONDS = 15;
const DEFAULT_PRICE_PER_SECOND = 0.0004;
const DEFAULT_MAX_FILES_PER_RUN = 10;
const DEFAULT_MAX_FILE_BYTES = 500 * 1024 * 1024;
const DEFAULT_MAX_TOTAL_COST_USD = 10;
const DEFAULT_UPLOAD_TTL_SECONDS = 15 * 60;
const DEFAULT_TOKEN_TTL_SECONDS = 2 * 60 * 60;
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
    tokenTtlSeconds
  };
}

function getPublicConfig(){
  const region = pickEnv(['TRANSCRIBE_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
  const bucket = pickEnv(['TRANSCRIBE_UPLOAD_BUCKET']);
  const signingSecret = pickEnv(['TRANSCRIBE_SIGNING_SECRET']);
  return {
    region,
    languageCode: pickEnv(['TRANSCRIBE_LANGUAGE_CODE']) || 'en-US',
    pricePerSecond: positiveNumber(process.env.TRANSCRIBE_PRICE_PER_SECOND, DEFAULT_PRICE_PER_SECOND),
    maxFilesPerRun: positiveInteger(process.env.TRANSCRIBE_MAX_FILES_PER_RUN, DEFAULT_MAX_FILES_PER_RUN),
    maxFileBytes: positiveInteger(process.env.TRANSCRIBE_MAX_FILE_BYTES, DEFAULT_MAX_FILE_BYTES),
    maxTotalCostUsd: positiveNumber(process.env.TRANSCRIBE_MAX_TOTAL_COST_USD, DEFAULT_MAX_TOTAL_COST_USD),
    configured: Boolean(region && bucket && signingSecret && signingSecret.length >= 24)
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
  const creds = getAwsCredentialsFromEnv();
  const key = `${config.region}:${creds ? creds.accessKeyId : 'default'}`;
  if (cachedS3Client && cachedTranscribeClient && cachedClientKey === key) {
    return { s3: cachedS3Client, transcribe: cachedTranscribeClient, config };
  }
  cachedS3Client = new S3Client({ region: config.region, credentials: creds || undefined });
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

function buildJobName(){
  const token = crypto.randomBytes(12).toString('hex');
  return `site-transcribe-${Date.now()}-${token}`.slice(0, 200);
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
  if (!Number.isFinite(bytes) || bytes <= 0) {
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
    const uploadUrl = await getSignedUrl(
      s3,
      new PutObjectCommand({
        Bucket: config.bucket,
        Key: key,
        ContentType: contentType
      }),
      { expiresIn: Math.min(config.uploadTtlSeconds, 3600) }
    );
    sendJson(res, 200, {
      ok: true,
      uploadUrl,
      method: 'PUT',
      headers: { 'Content-Type': contentType },
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
  if (quote.sub !== user.sub || quote.bucket !== config.bucket || !String(quote.key || '').startsWith(config.prefix)) {
    sendJson(res, 403, { ok: false, error: 'Quote does not belong to the signed-in user.' });
    return;
  }
  if (Number(quote.estimatedCostUsd) > config.maxTotalCostUsd) {
    sendJson(res, 400, { ok: false, error: `Estimated cost exceeds $${config.maxTotalCostUsd}.` });
    return;
  }

  const now = Math.floor(Date.now() / 1000);
  const jobName = buildJobName();
  const runPayload = {
    ...quote,
    type: 'run',
    jobName,
    startedAt: Date.now(),
    iat: now,
    exp: now + config.tokenTtlSeconds
  };
  const runToken = signToken(runPayload, config.signingSecret);

  try {
    await transcribe.send(new StartTranscriptionJobCommand({
      TranscriptionJobName: jobName,
      LanguageCode: config.languageCode,
      MediaFormat: quote.format,
      Media: { MediaFileUri: `s3://${quote.bucket}/${quote.key}` },
      Tags: [
        { Key: 'tool', Value: 'amazon-transcribe' },
        { Key: 'user', Value: user.sub.slice(0, 200) }
      ]
    }));
    sendJson(res, 200, {
      ok: true,
      runToken,
      jobName,
      status: 'QUEUED',
      estimatedCostUsd: quote.estimatedCostUsd,
      billableSeconds: quote.billableSeconds
    });
  } catch (err) {
    await Promise.allSettled([
      s3.send(new DeleteObjectCommand({ Bucket: quote.bucket, Key: quote.key }))
    ]);
    sendJson(res, 500, {
      ok: false,
      error: err && err.name === 'BadRequestException'
        ? (err.message || 'Unable to start transcription job.')
        : 'Unable to start transcription job.'
    });
  }
}

async function cleanupRun({ s3, transcribe, run }){
  await Promise.allSettled([
    s3.send(new DeleteObjectCommand({ Bucket: run.bucket, Key: run.key })),
    transcribe.send(new DeleteTranscriptionJobCommand({ TranscriptionJobName: run.jobName }))
  ]);
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
  if (run.sub !== user.sub || run.bucket !== config.bucket || !String(run.key || '').startsWith(config.prefix)) {
    sendJson(res, 403, { ok: false, error: 'Run does not belong to the signed-in user.' });
    return;
  }

  let job;
  try {
    const out = await transcribe.send(new GetTranscriptionJobCommand({ TranscriptionJobName: run.jobName }));
    job = out.TranscriptionJob;
  } catch (err) {
    sendJson(res, 404, { ok: false, error: 'Transcription job was not found.' });
    return;
  }

  const status = String(job && job.TranscriptionJobStatus || '').toUpperCase();
  if (status === 'COMPLETED') {
    const uri = String(job?.Transcript?.TranscriptFileUri || '').trim();
    let transcript = '';
    try {
      const transcriptRes = await fetch(uri);
      if (!transcriptRes.ok) throw new Error(`Transcript fetch failed (${transcriptRes.status}).`);
      const data = await transcriptRes.json();
      transcript = extractTranscript(data);
    } catch {
      sendJson(res, 500, { ok: false, error: 'Unable to fetch completed transcript.' });
      return;
    }
    await cleanupRun({ s3, transcribe, run });
    sendJson(res, 200, {
      ok: true,
      status,
      transcript,
      costUsd: run.estimatedCostUsd,
      billableSeconds: run.billableSeconds,
      durationSeconds: run.durationSeconds,
      cleanedUp: true
    });
    return;
  }

  if (status === 'FAILED') {
    await cleanupRun({ s3, transcribe, run });
    sendJson(res, 200, {
      ok: true,
      status,
      error: String(job?.FailureReason || 'Transcription failed.'),
      costUsd: run.estimatedCostUsd,
      billableSeconds: run.billableSeconds,
      durationSeconds: run.durationSeconds,
      cleanedUp: true
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
