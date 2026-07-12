'use strict';

const { once } = require('events');
const {
  LambdaClient,
  InvokeWithResponseStreamCommand
} = require('@aws-sdk/client-lambda');
const { resolveAwsCredentials } = require('./aws-credentials');
const {
  checkChatbotRateLimit
} = require('./chatbot-rate-limit');

const SITE_ORIGIN = 'https://www.danielshort.me';
const DEFAULT_REGION = 'us-east-2';
const MAX_JSON_BODY_BYTES = 8 * 1024;
const MAX_PROMPT_CHARS = 1200;
const MAX_PROMPT_BYTES = 6 * 1024;
const MAX_STREAM_BYTES = 512 * 1024;
const MAX_PRELUDE_BYTES = 16 * 1024;
const MAX_EVENT_LINE_BYTES = 256 * 1024;
const MAX_TOKEN_EVENT_BYTES = 32 * 1024;
const MAX_FOLLOWUP_CONTEXT_BYTES = 4 * 1024;
const DEFAULT_TIMEOUT_MS = 35_000;
const MIN_TIMEOUT_MS = 5_000;
const MAX_TIMEOUT_MS = 55_000;
const RESPONSE_STREAM_DELIMITER = Buffer.alloc(8);

const CHATBOT_STATIC_CREDENTIAL_SETS = Object.freeze([
  Object.freeze({
    name: 'chatbot',
    accessKeyId: 'CHATBOT_AWS_ACCESS_KEY_ID',
    secretAccessKey: 'CHATBOT_AWS_SECRET_ACCESS_KEY',
    sessionToken: 'CHATBOT_AWS_SESSION_TOKEN'
  }),
  Object.freeze({
    name: 'default',
    accessKeyId: 'AWS_ACCESS_KEY_ID',
    secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
    sessionToken: 'AWS_SESSION_TOKEN'
  })
]);

let cachedLambdaClient = null;
let cachedLambdaClientKey = '';
let lambdaClientFactoryOverride = null;
let rateLimiterOverride = null;

function pickEnv(keys, env = process.env) {
  for (const key of Array.isArray(keys) ? keys : []) {
    const raw = env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function isProductionRuntime(env = process.env) {
  return String(env.VERCEL_ENV || '').trim().toLowerCase() === 'production';
}

function configError(code, message) {
  const err = new Error(message);
  err.code = code;
  err.statusCode = 503;
  return err;
}

function requestError(code, message, statusCode) {
  const err = new Error(message);
  err.code = code;
  err.statusCode = statusCode;
  return err;
}

function clampInteger(value, fallback, min, max) {
  const number = Number.parseInt(String(value || ''), 10);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(min, Math.min(max, number));
}

function parseFunctionArn(value, env = process.env) {
  const arn = String(value || '').trim();
  const match = arn.match(/^arn:(aws(?:-[a-z]+)*):lambda:([a-z0-9-]+):(\d{12}):function:([A-Za-z0-9-_]+):([A-Za-z0-9-_]+)$/);
  if (!match) {
    throw configError(
      'CHATBOT_STREAM_FUNCTION_ARN_INVALID',
      'CHATBOT_STREAM_FUNCTION_ARN must be a qualified Lambda alias ARN.'
    );
  }
  const parsed = {
    arn,
    partition: match[1],
    region: match[2],
    accountId: match[3],
    functionName: match[4],
    alias: match[5]
  };
  if (isProductionRuntime(env) && parsed.alias !== 'live') {
    throw configError(
      'CHATBOT_STREAM_ALIAS_INVALID',
      'Production chatbot streaming must invoke the live Lambda alias.'
    );
  }
  return parsed;
}

function getRuntimeConfig(env = process.env) {
  const functionArn = pickEnv(['CHATBOT_STREAM_FUNCTION_ARN'], env);
  if (!functionArn) {
    throw configError(
      'CHATBOT_STREAM_FUNCTION_ARN_MISSING',
      'CHATBOT_STREAM_FUNCTION_ARN is not configured.'
    );
  }
  const target = parseFunctionArn(functionArn, env);
  const configuredRegion = pickEnv([
    'CHATBOT_STREAM_AWS_REGION',
    'CHATBOT_AWS_REGION',
    'AWS_REGION',
    'AWS_DEFAULT_REGION'
  ], env) || DEFAULT_REGION;
  if (configuredRegion !== target.region) {
    throw configError(
      'CHATBOT_STREAM_REGION_MISMATCH',
      'The chatbot stream Lambda ARN and configured AWS region do not match.'
    );
  }
  const auth = resolveAwsCredentials({
    env,
    service: 'chatbot-stream',
    region: target.region,
    roleArnEnvKeys: [
      'CHATBOT_STREAM_AWS_ROLE_ARN',
      'DEMO_INVOKE_AWS_ROLE_ARN'
    ],
    staticCredentialSets: CHATBOT_STATIC_CREDENTIAL_SETS
  });
  if (isProductionRuntime(env) && auth.source !== 'oidc') {
    throw configError(
      'CHATBOT_STREAM_OIDC_REQUIRED',
      'Production chatbot streaming requires Vercel OIDC role credentials.'
    );
  }
  return {
    functionArn: target.arn,
    alias: target.alias,
    region: target.region,
    credentials: auth.credentials,
    authSource: auth.source,
    clientKey: `${target.region}:${auth.cacheKey}`,
    timeoutMs: clampInteger(
      env.CHATBOT_STREAM_TIMEOUT_MS,
      DEFAULT_TIMEOUT_MS,
      MIN_TIMEOUT_MS,
      MAX_TIMEOUT_MS
    ),
    maxStreamBytes: MAX_STREAM_BYTES
  };
}

function createLambdaClient(config) {
  if (lambdaClientFactoryOverride) return lambdaClientFactoryOverride(config);
  if (cachedLambdaClient && cachedLambdaClientKey === config.clientKey) {
    return cachedLambdaClient;
  }
  cachedLambdaClient = new LambdaClient({
    region: config.region,
    credentials: config.credentials,
    maxAttempts: 1
  });
  cachedLambdaClientKey = config.clientKey;
  return cachedLambdaClient;
}

function requestOrigin(req) {
  return String(req?.headers?.origin || '').trim();
}

function expectedOrigin(req) {
  const headers = req?.headers || {};
  const host = String(headers['x-forwarded-host'] || headers.host || '')
    .split(',')[0]
    .trim();
  if (!host) return '';
  const forwardedProto = String(headers['x-forwarded-proto'] || '')
    .split(',')[0]
    .trim();
  const proto = forwardedProto
    || (/^(localhost|127\.0\.0\.1|\[::1\])(?::|$)/i.test(host) ? 'http' : 'https');
  return `${proto}://${host}`;
}

function isAllowedOrigin(req, env = process.env) {
  const origin = requestOrigin(req);
  const fetchSite = String(req?.headers?.['sec-fetch-site'] || '').trim().toLowerCase();
  if (!origin || (fetchSite && fetchSite !== 'same-origin')) return false;
  if (origin !== expectedOrigin(req)) return false;
  if (isProductionRuntime(env)) return origin === SITE_ORIGIN;
  try {
    const url = new URL(origin);
    if (String(env.VERCEL_ENV || '').trim().toLowerCase() === 'preview') {
      return url.protocol === 'https:' && url.hostname.endsWith('.vercel.app');
    }
    return ['localhost', '127.0.0.1', '::1'].includes(url.hostname)
      || origin === SITE_ORIGIN;
  } catch {
    return false;
  }
}

function assertJsonContentType(req) {
  const contentType = String(req?.headers?.['content-type'] || '').toLowerCase();
  if (!/^application\/json(?:\s*;|$)/.test(contentType)) {
    throw requestError(
      'UNSUPPORTED_MEDIA_TYPE',
      'Content-Type must be application/json.',
      415
    );
  }
}

function assertBodySize(raw) {
  if (Buffer.byteLength(raw, 'utf8') > MAX_JSON_BODY_BYTES) {
    throw requestError('BODY_TOO_LARGE', 'Request body is too large.', 413);
  }
}

async function readJsonBody(req) {
  const contentLength = Number(req?.headers?.['content-length']);
  if (Number.isFinite(contentLength) && contentLength > MAX_JSON_BODY_BYTES) {
    throw requestError('BODY_TOO_LARGE', 'Request body is too large.', 413);
  }
  if (req?.body && typeof req.body === 'object' && !Buffer.isBuffer(req.body)) {
    const raw = JSON.stringify(req.body);
    assertBodySize(raw);
    return req.body;
  }
  if (typeof req?.body === 'string') {
    assertBodySize(req.body);
    try {
      return JSON.parse(req.body || '{}');
    } catch {
      throw requestError('INVALID_JSON', 'Request body must be valid JSON.', 400);
    }
  }

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += bytes.length;
    if (size > MAX_JSON_BODY_BYTES) {
      throw requestError('BODY_TOO_LARGE', 'Request body is too large.', 413);
    }
    chunks.push(bytes);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  try {
    return JSON.parse(raw || '{}');
  } catch {
    throw requestError('INVALID_JSON', 'Request body must be valid JSON.', 400);
  }
}

function normalizePayload(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw requestError('INVALID_BODY', 'Request body must be a JSON object.', 400);
  }
  if (typeof value.prompt !== 'string') {
    throw requestError('PROMPT_REQUIRED', 'Prompt is required.', 400);
  }
  const prompt = value.prompt.replace(/\r\n?/g, '\n').trim();
  if (!prompt) throw requestError('PROMPT_REQUIRED', 'Prompt is required.', 400);
  if (/\0|[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(prompt)) {
    throw requestError('PROMPT_INVALID', 'Prompt contains unsupported characters.', 400);
  }
  if (prompt.length > MAX_PROMPT_CHARS || Buffer.byteLength(prompt, 'utf8') > MAX_PROMPT_BYTES) {
    throw requestError(
      'PROMPT_TOO_LONG',
      `Prompt must be ${MAX_PROMPT_CHARS} characters or fewer.`,
      413
    );
  }
  const followupContext = normalizeFollowupContext(value.followup_context);
  return {
    prompt,
    ...(followupContext ? { followup_context: followupContext } : {})
  };
}

function boundedString(value, field, maxChars, options = {}) {
  if (typeof value === 'undefined' || value === null || value === '') return '';
  if (typeof value !== 'string') {
    throw requestError('FOLLOWUP_CONTEXT_INVALID', `${field} must be a string.`, 400);
  }
  const text = value.replace(/\r\n?/g, '\n').trim();
  if (text.length > maxChars || /\0|[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(text)) {
    throw requestError('FOLLOWUP_CONTEXT_INVALID', `${field} is invalid.`, 400);
  }
  if (options.httpsUrl && text) {
    try {
      const url = new URL(text);
      if (url.protocol !== 'https:' || url.username || url.password) throw new Error('invalid');
      url.hash = '';
      return url.toString();
    } catch {
      throw requestError('FOLLOWUP_CONTEXT_INVALID', `${field} must be an HTTPS URL.`, 400);
    }
  }
  return text;
}

function boundedStringArray(value, field, maxItems, maxChars, options = {}) {
  if (typeof value === 'undefined' || value === null) return [];
  if (!Array.isArray(value) || value.length > maxItems) {
    throw requestError('FOLLOWUP_CONTEXT_INVALID', `${field} is invalid.`, 400);
  }
  return value.map((item, index) => boundedString(
    item,
    `${field}[${index}]`,
    maxChars,
    options
  )).filter(Boolean);
}

function normalizeFollowupContext(value) {
  if (typeof value === 'undefined' || value === null) return null;
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw requestError('FOLLOWUP_CONTEXT_INVALID', 'followup_context must be an object.', 400);
  }
  const source = boundedString(value.source, 'followup_context.source', 40);
  if (source !== 'recommended_followup') {
    throw requestError(
      'FOLLOWUP_CONTEXT_INVALID',
      'followup_context.source is invalid.',
      400
    );
  }
  const normalized = {
    source,
    prompt: boundedString(value.prompt, 'followup_context.prompt', 240),
    previous_question: boundedString(
      value.previous_question,
      'followup_context.previous_question',
      300
    ),
    previous_answer: boundedString(
      value.previous_answer,
      'followup_context.previous_answer',
      650
    ),
    previous_intent: boundedString(
      value.previous_intent,
      'followup_context.previous_intent',
      80
    ),
    previous_category: boundedString(
      value.previous_category,
      'followup_context.previous_category',
      80
    ),
    previous_route: boundedString(
      value.previous_route,
      'followup_context.previous_route',
      160
    ),
    source_labels: boundedStringArray(
      value.source_labels,
      'followup_context.source_labels',
      10,
      120
    ),
    source_urls: boundedStringArray(
      value.source_urls,
      'followup_context.source_urls',
      10,
      500,
      { httpsUrl: true }
    )
  };
  if (Buffer.byteLength(JSON.stringify(normalized), 'utf8') > MAX_FOLLOWUP_CONTEXT_BYTES) {
    throw requestError(
      'FOLLOWUP_CONTEXT_TOO_LARGE',
      'followup_context is too large.',
      413
    );
  }
  return normalized;
}

function buildLambdaEvent(body) {
  return {
    version: '2.0',
    routeKey: '$default',
    rawPath: '/',
    rawQueryString: '',
    headers: {
      accept: 'application/x-ndjson',
      'content-type': 'application/json',
      'user-agent': 'danielshort-chatbot-stream-proxy/1.0'
    },
    requestContext: {
      http: {
        method: 'POST',
        path: '/',
        protocol: 'HTTP/1.1'
      }
    },
    body: JSON.stringify(body),
    isBase64Encoded: false
  };
}

function rateLimitRequest(req) {
  const headers = { ...(req?.headers || {}) };
  delete headers['x-chatbot-session'];
  return {
    headers,
    socket: req?.socket || null
  };
}

async function consumeRateLimit(req) {
  const limiter = rateLimiterOverride || checkChatbotRateLimit;
  return limiter(rateLimitRequest(req), {}, { challengePassed: false });
}

function sendJson(res, statusCode, payload, extraHeaders = {}) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  for (const [key, value] of Object.entries(extraHeaders)) res.setHeader(key, value);
  res.end(JSON.stringify(payload));
}

function rateHeaders(rate) {
  const config = rate?.config || {};
  const limit = Number(config.windowLimit || rate?.payload?.limits?.windowLimit || 0);
  const count = Number(rate?.counts?.window || 0);
  const headers = {};
  if (limit > 0) {
    headers['X-RateLimit-Limit'] = String(limit);
    headers['X-RateLimit-Remaining'] = String(Math.max(0, limit - count));
  }
  return headers;
}

function validateStreamEvent(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw requestError('CHATBOT_STREAM_INVALID_EVENT', 'Invalid chatbot stream event.', 502);
  }
  const type = String(value.type || '').trim();
  if (type === 'token') {
    const text = typeof value.text === 'string' ? value.text : '';
    if (!text || Buffer.byteLength(text, 'utf8') > MAX_TOKEN_EVENT_BYTES) {
      throw requestError('CHATBOT_STREAM_INVALID_TOKEN', 'Invalid chatbot stream token.', 502);
    }
    return { value: { type: 'token', text }, terminal: false };
  }
  if (type === 'meta' || type === 'done') {
    const serialized = JSON.stringify(value);
    if (Buffer.byteLength(serialized, 'utf8') > MAX_EVENT_LINE_BYTES) {
      throw requestError('CHATBOT_STREAM_EVENT_TOO_LARGE', 'Chatbot stream event is too large.', 502);
    }
    return { value, terminal: type === 'done' };
  }
  if (type === 'error') {
    return {
      value: { type: 'error', error: 'Chatbot stream is temporarily unavailable.' },
      terminal: true,
      failed: true
    };
  }
  throw requestError('CHATBOT_STREAM_UNKNOWN_EVENT', 'Unknown chatbot stream event.', 502);
}

async function writeChunk(res, chunk) {
  if (res.destroyed || res.writableEnded) return false;
  if (res.write(chunk) !== false) return true;
  await once(res, 'drain');
  return !(res.destroyed || res.writableEnded);
}

async function writeEventLine(res, line) {
  if (!line.trim()) return { terminal: false };
  if (Buffer.byteLength(line, 'utf8') > MAX_EVENT_LINE_BYTES) {
    throw requestError('CHATBOT_STREAM_EVENT_TOO_LARGE', 'Chatbot stream event is too large.', 502);
  }
  let parsed;
  try {
    parsed = JSON.parse(line);
  } catch {
    throw requestError('CHATBOT_STREAM_INVALID_JSON', 'Invalid chatbot stream JSON.', 502);
  }
  const event = validateStreamEvent(parsed);
  await writeChunk(res, `${JSON.stringify(event.value)}\n`);
  return event;
}

function validateStreamPrelude(raw) {
  if (!raw.length || raw.length > MAX_PRELUDE_BYTES) {
    throw requestError('CHATBOT_STREAM_INVALID_PRELUDE', 'Invalid chatbot stream prelude.', 502);
  }
  let metadata;
  try {
    metadata = JSON.parse(raw.toString('utf8'));
  } catch {
    throw requestError('CHATBOT_STREAM_INVALID_PRELUDE', 'Invalid chatbot stream prelude.', 502);
  }
  const statusCode = Number(metadata?.statusCode);
  if (!Number.isInteger(statusCode) || statusCode < 200 || statusCode >= 300) {
    throw requestError('CHATBOT_STREAM_UPSTREAM_STATUS', 'Chatbot stream was rejected.', 502);
  }
  const headers = metadata?.headers;
  if (!headers || typeof headers !== 'object' || Array.isArray(headers)) {
    throw requestError('CHATBOT_STREAM_INVALID_PRELUDE', 'Invalid chatbot stream prelude.', 502);
  }
  const contentTypeEntry = Object.entries(headers).find(([key]) => (
    String(key).toLowerCase() === 'content-type'
  ));
  const contentType = String(contentTypeEntry?.[1] || '').toLowerCase();
  if (!contentType.startsWith('application/x-ndjson')) {
    throw requestError('CHATBOT_STREAM_INVALID_CONTENT_TYPE', 'Invalid chatbot stream type.', 502);
  }
  return { statusCode, contentType };
}

async function forwardLambdaEventStream(res, eventStream, config, controller) {
  if (!eventStream || typeof eventStream[Symbol.asyncIterator] !== 'function') {
    throw requestError('CHATBOT_STREAM_MISSING', 'Chatbot stream is unavailable.', 502);
  }
  const decoder = new TextDecoder();
  let buffer = '';
  let totalBytes = 0;
  let preludeBuffer = Buffer.alloc(0);
  let preludeComplete = false;
  let sawDone = false;
  let sawError = false;
  let invokeError = null;

  for await (const envelope of eventStream) {
    if (envelope?.PayloadChunk?.Payload) {
      let payload = Buffer.from(envelope.PayloadChunk.Payload);
      totalBytes += payload.byteLength;
      if (totalBytes > config.maxStreamBytes) {
        controller.abort();
        throw requestError('CHATBOT_STREAM_TOO_LARGE', 'Chatbot stream is too large.', 502);
      }
      if (!preludeComplete) {
        preludeBuffer = Buffer.concat([preludeBuffer, payload]);
        const delimiterIndex = preludeBuffer.indexOf(RESPONSE_STREAM_DELIMITER);
        if (delimiterIndex < 0) {
          if (preludeBuffer.length > MAX_PRELUDE_BYTES) {
            throw requestError(
              'CHATBOT_STREAM_INVALID_PRELUDE',
              'Invalid chatbot stream prelude.',
              502
            );
          }
          continue;
        }
        validateStreamPrelude(preludeBuffer.subarray(0, delimiterIndex));
        payload = preludeBuffer.subarray(delimiterIndex + RESPONSE_STREAM_DELIMITER.length);
        preludeBuffer = Buffer.alloc(0);
        preludeComplete = true;
        if (!payload.length) continue;
      }
      buffer += decoder.decode(payload, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const result = await writeEventLine(res, line);
        sawDone = sawDone || (result.terminal && !result.failed);
        sawError = sawError || Boolean(result.failed);
      }
    }
    if (envelope?.InvokeComplete?.ErrorCode) {
      invokeError = String(envelope.InvokeComplete.ErrorCode);
    }
  }

  if (!preludeComplete) {
    throw requestError('CHATBOT_STREAM_PRELUDE_MISSING', 'Chatbot stream prelude is missing.', 502);
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    const result = await writeEventLine(res, buffer);
    sawDone = sawDone || (result.terminal && !result.failed);
    sawError = sawError || Boolean(result.failed);
  }
  if (invokeError || (!sawDone && !sawError)) {
    throw requestError('CHATBOT_STREAM_INCOMPLETE', 'Chatbot stream ended unexpectedly.', 502);
  }
  return { totalBytes, sawDone, sawError };
}

function setStreamHeaders(res, rate, response) {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store, no-transform');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('X-Accel-Buffering', 'no');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  const executedVersion = String(response?.ExecutedVersion || '');
  if (/^\d+$/.test(executedVersion)) {
    res.setHeader('X-Chatbot-Lambda-Version', executedVersion);
  }
  for (const [key, value] of Object.entries(rateHeaders(rate))) res.setHeader(key, value);
  if (typeof res.flushHeaders === 'function') res.flushHeaders();
}

async function writeTerminalError(res) {
  if (res.destroyed || res.writableEnded) return;
  await writeChunk(res, `${JSON.stringify({
    type: 'error',
    error: 'Chatbot stream is temporarily unavailable.'
  })}\n`);
}

function statusForInvokeError(err) {
  if (err?.name === 'AbortError' || err?.code === 'CHATBOT_STREAM_TIMEOUT') return 504;
  if (['TooManyRequestsException', 'ThrottlingException'].includes(String(err?.name || ''))) return 503;
  return Number(err?.statusCode) || 502;
}

async function handleChatbotStream(req, res) {
  if (!isAllowedOrigin(req)) {
    sendJson(res, 403, { ok: false, error: 'Origin not allowed.' });
    return;
  }
  if (String(req.method || '').toUpperCase() !== 'POST') {
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' }, { Allow: 'POST' });
    return;
  }

  let body;
  try {
    assertJsonContentType(req);
    body = normalizePayload(await readJsonBody(req));
  } catch (err) {
    sendJson(res, err?.statusCode || 400, { ok: false, error: err?.message || 'Invalid request.' });
    return;
  }

  let config;
  let lambda;
  try {
    config = getRuntimeConfig();
    lambda = createLambdaClient(config);
  } catch {
    sendJson(res, 503, { ok: false, error: 'Chatbot stream is not configured.' });
    return;
  }

  let rate;
  try {
    rate = await consumeRateLimit(req);
  } catch {
    sendJson(res, 503, { ok: false, error: 'Chatbot request protection is unavailable.' });
    return;
  }
  const headers = rateHeaders(rate);
  if (!rate?.allowed) {
    const retryAfter = Math.max(1, Number(rate?.payload?.retryAfter || 60));
    headers['Retry-After'] = String(retryAfter);
    sendJson(res, rate?.statusCode || 429, {
      ok: false,
      error: rate?.payload?.error || 'Too many chatbot requests.',
      retryAfter
    }, headers);
    return;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.timeoutMs);
  const closeHandler = () => {
    if (!res.writableEnded) controller.abort();
  };
  if (typeof res.once === 'function') res.once('close', closeHandler);

  let response;
  try {
    response = await lambda.send(new InvokeWithResponseStreamCommand({
      FunctionName: config.functionArn,
      InvocationType: 'RequestResponse',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(buildLambdaEvent(body)), 'utf8')
    }), { abortSignal: controller.signal });
    if (Number(response?.StatusCode) !== 200) {
      throw requestError('CHATBOT_STREAM_INVOKE_FAILED', 'Chatbot invocation failed.', 502);
    }
    setStreamHeaders(res, rate, response);
    await forwardLambdaEventStream(res, response.EventStream, config, controller);
  } catch (err) {
    if (res.headersSent) {
      await writeTerminalError(res).catch(() => {});
    } else {
      const statusCode = statusForInvokeError(err);
      const retryHeaders = statusCode === 503 ? { 'Retry-After': '10' } : {};
      sendJson(res, statusCode, {
        ok: false,
        error: statusCode === 504
          ? 'Chatbot stream timed out.'
          : 'Chatbot stream is temporarily unavailable.'
      }, { ...headers, ...retryHeaders });
      return;
    }
  } finally {
    clearTimeout(timeout);
    if (typeof res.off === 'function') res.off('close', closeHandler);
  }
  if (!res.writableEnded) res.end();
}

function setLambdaClientFactoryForTests(factory) {
  lambdaClientFactoryOverride = factory || null;
  cachedLambdaClient = null;
  cachedLambdaClientKey = '';
}

function setRateLimiterForTests(limiter) {
  rateLimiterOverride = limiter || null;
}

module.exports = {
  handleChatbotStream,
  _internal: {
    MAX_JSON_BODY_BYTES,
    MAX_FOLLOWUP_CONTEXT_BYTES,
    MAX_PRELUDE_BYTES,
    MAX_PROMPT_BYTES,
    MAX_PROMPT_CHARS,
    MAX_STREAM_BYTES,
    buildLambdaEvent,
    expectedOrigin,
    forwardLambdaEventStream,
    getRuntimeConfig,
    isAllowedOrigin,
    normalizePayload,
    normalizeFollowupContext,
    parseFunctionArn,
    rateLimitRequest,
    setLambdaClientFactoryForTests,
    setRateLimiterForTests,
    validateStreamPrelude,
    validateStreamEvent,
    writeEventLine
  }
};
