'use strict';

const { SESv2Client, SendEmailCommand } = require('@aws-sdk/client-sesv2');
const { AWS_WORKLOADS, getAwsClientConfig } = require('./aws-credentials');
const {
  applyBoundaryHeaders,
  httpError,
  isSameOriginRequest,
  readJsonBody,
  sendJson
} = require('./http-boundary');
const { consumeRequestLimit, positiveNumber } = require('./request-rate-limit');

const DEFAULT_UPSTREAM = 'https://muee4eg6ze.execute-api.us-east-2.amazonaws.com/prod/contact';
const MAX_BODY_BYTES = 16_000;
const LEGACY_TIMEOUT_MS = 15_000;
const sesClients = new Map();

function pickEnv(env, keys) {
  for (const key of keys) {
    const value = env && typeof env[key] !== 'undefined' ? String(env[key]).trim() : '';
    if (value) return value;
  }
  return '';
}

function sanitizeInline(value, maxLength) {
  return String(value || '')
    .replace(/[\u0000-\u001f\u007f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxLength);
}

function sanitizeMessage(value) {
  return String(value || '').replace(/\u0000/g, '').trim();
}

function isEmail(value) {
  return value.length <= 254
    && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
    && !/[\r\n]/.test(value);
}

function validateContactPayload(payload) {
  const rawName = String(payload && payload.name || '').trim();
  const rawEmail = String(payload && payload.email || '').trim();
  const rawMessage = sanitizeMessage(payload && payload.message);
  const honey = sanitizeInline(payload && payload.company, 500);
  const name = sanitizeInline(rawName, 200);
  const email = sanitizeInline(rawEmail, 254);

  if (honey) return { valid: true, honeypot: true, value: { name, email, message: rawMessage } };
  const valid = Boolean(
    rawName
    && rawName.length <= 200
    && name
    && rawEmail
    && rawEmail.length <= 254
    && isEmail(email)
    && rawMessage
    && rawMessage.length <= 4000
  );
  return {
    valid,
    honeypot: false,
    value: { name, email, message: rawMessage }
  };
}

function resolveDeliveryMode(env) {
  const explicit = String(env.CONTACT_DELIVERY_MODE || '').trim().toLowerCase();
  if (explicit) {
    if (!['legacy', 'ses'].includes(explicit)) {
      throw httpError(503, 'CONTACT_DELIVERY_MODE_INVALID', 'Contact service is not configured');
    }
    return explicit;
  }
  return String(env.AWS_AUTH_MODE || '').trim().toLowerCase() === 'oidc' ? 'ses' : 'legacy';
}

function contactEmailConfig(env) {
  const source = pickEnv(env, ['CONTACT_SENDER_EMAIL', 'SENDER_EMAIL']);
  const recipients = pickEnv(env, ['CONTACT_RECIPIENT_EMAIL', 'RECIPIENT_EMAIL'])
    .split(',')
    .map((value) => sanitizeInline(value, 254).toLowerCase())
    .filter(Boolean);
  if (!isEmail(source) || !recipients.length || recipients.length > 10 || recipients.some((email) => !isEmail(email))) {
    throw httpError(503, 'CONTACT_EMAIL_CONFIG_INVALID', 'Contact service is not configured');
  }
  return {
    source,
    recipients,
    configurationSet: sanitizeInline(env.CONTACT_SES_CONFIGURATION_SET, 64),
    subjectPrefix: sanitizeInline(env.CONTACT_EMAIL_SUBJECT_PREFIX || 'Website contact', 80)
  };
}

function buildSesInput(payload, env) {
  const config = contactEmailConfig(env);
  const subject = `${config.subjectPrefix}: ${payload.name}`.slice(0, 200);
  const input = {
    FromEmailAddress: config.source,
    Destination: { ToAddresses: config.recipients },
    ReplyToAddresses: [payload.email],
    Content: {
      Simple: {
        Subject: { Data: subject, Charset: 'UTF-8' },
        Body: {
          Text: {
            Data: `Name: ${payload.name}\nEmail: ${payload.email}\n\nMessage:\n${payload.message}`,
            Charset: 'UTF-8'
          }
        }
      }
    }
  };
  if (config.configurationSet) input.ConfigurationSetName = config.configurationSet;
  return input;
}

function getSesClient(env) {
  const region = pickEnv(env, ['CONTACT_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']) || 'us-east-2';
  const aws = getAwsClientConfig(AWS_WORKLOADS.CONTACT, { region });
  const key = `${region}:${aws.cacheKey}`;
  if (sesClients.has(key)) return sesClients.get(key);
  const client = new SESv2Client(aws.clientConfig);
  sesClients.set(key, client);
  return client;
}

async function sendThroughSes(payload, env) {
  const input = buildSesInput(payload, env);
  await getSesClient(env).send(new SendEmailCommand(input));
}

function safeUpstreamError(data) {
  const raw = data && typeof data.error === 'string' ? data.error : '';
  const safe = sanitizeInline(raw, 200);
  return safe || 'Unable to send message.';
}

async function sendThroughLegacy(payload, env, fetchImpl) {
  const upstream = pickEnv(env, ['CONTACT_FORM_UPSTREAM']) || DEFAULT_UPSTREAM;
  let url;
  try {
    url = new URL(upstream);
  } catch {
    throw httpError(503, 'CONTACT_UPSTREAM_INVALID', 'Contact service is not configured');
  }
  if (url.protocol !== 'https:') {
    throw httpError(503, 'CONTACT_UPSTREAM_INVALID', 'Contact service is not configured');
  }

  const response = await fetchImpl(url.toString(), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
    ...(typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
      ? { signal: AbortSignal.timeout(LEGACY_TIMEOUT_MS) }
      : {})
  });
  const raw = await response.text().catch(() => '');
  let data = {};
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = {};
    }
  }
  if (!response.ok) {
    const statusCode = response.status >= 400 && response.status < 500 ? response.status : 502;
    return { statusCode, payload: { ok: false, error: safeUpstreamError(data) } };
  }
  return { statusCode: 200, payload: { ok: true } };
}

async function checkContactRateLimit(req, env) {
  return consumeRequestLimit(req, {
    namespace: 'contact',
    windowSeconds: positiveNumber(env.CONTACT_RATE_LIMIT_WINDOW_SECONDS, 3600),
    windowLimit: positiveNumber(env.CONTACT_RATE_LIMIT_MAX, 5),
    dailyLimit: positiveNumber(env.CONTACT_DAILY_LIMIT, 20),
    globalDailyLimit: positiveNumber(env.CONTACT_GLOBAL_DAILY_LIMIT, 200),
    ttlDays: 3,
    salt: pickEnv(env, ['CONTACT_HASH_SALT', 'VERCEL_PROJECT_PRODUCTION_URL', 'VERCEL_URL']) || 'local-contact'
  });
}

function createContactHandler(dependencies = {}) {
  const env = dependencies.env || process.env;
  const rateLimit = dependencies.checkRateLimit || checkContactRateLimit;
  const sendSes = dependencies.sendSes || sendThroughSes;
  const fetchImpl = dependencies.fetch || globalThis.fetch;

  return async function contactHandler(req, res) {
    const boundaryOptions = {
      env,
      allowedOriginsEnv: 'CONTACT_ALLOWED_ORIGINS',
      methods: ['POST', 'OPTIONS']
    };
    applyBoundaryHeaders(req, res, boundaryOptions);
    if (!isSameOriginRequest(req, boundaryOptions)) {
      sendJson(res, 403, { ok: false, error: 'Origin not allowed' });
      return;
    }
    if (String(req.method || '').toUpperCase() === 'OPTIONS') {
      res.statusCode = 204;
      res.end();
      return;
    }
    if (String(req.method || '').toUpperCase() !== 'POST') {
      res.setHeader('Allow', 'POST, OPTIONS');
      sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
      return;
    }

    let body;
    try {
      body = await readJsonBody(req, MAX_BODY_BYTES);
    } catch (err) {
      sendJson(res, err.statusCode || 400, { ok: false, error: err.message || 'Invalid JSON body' });
      return;
    }

    const validation = validateContactPayload(body.value);
    if (validation.honeypot) {
      sendJson(res, 200, { ok: true });
      return;
    }

    let limit;
    try {
      limit = await rateLimit(req, env);
    } catch (err) {
      console.error('Contact rate-limit check failed', { code: err && err.code, name: err && err.name });
      sendJson(res, 503, { ok: false, error: 'Contact service unavailable' });
      return;
    }
    if (!limit.allowed) {
      res.setHeader('Retry-After', String(limit.retryAfter || 60));
      sendJson(res, 429, { ok: false, error: 'Too many messages. Please try again later.' });
      return;
    }

    if (!validation.valid) {
      sendJson(res, 400, { ok: false, error: 'Please provide a valid name, email, and message.' });
      return;
    }

    try {
      const mode = resolveDeliveryMode(env);
      if (mode === 'ses') {
        await sendSes(validation.value, env);
        sendJson(res, 200, { ok: true });
        return;
      }
      const result = await sendThroughLegacy(validation.value, env, fetchImpl);
      sendJson(res, result.statusCode, result.payload);
    } catch (err) {
      const isTimeout = err && (err.name === 'TimeoutError' || err.name === 'AbortError');
      const isConfig = err && Number(err.statusCode) === 503;
      console.error('Contact delivery failed', { code: err && err.code, name: err && err.name });
      sendJson(res, isConfig ? 503 : (isTimeout ? 504 : 502), {
        ok: false,
        error: isConfig ? 'Contact service is not configured' : 'Unable to send message right now.'
      });
    }
  };
}

module.exports = {
  buildSesInput,
  checkContactRateLimit,
  contactEmailConfig,
  createContactHandler,
  resolveDeliveryMode,
  sendThroughLegacy,
  sendThroughSes,
  validateContactPayload,
  _private: {
    DEFAULT_UPSTREAM,
    LEGACY_TIMEOUT_MS,
    MAX_BODY_BYTES,
    getSesClient,
    isEmail,
    sanitizeInline,
    sanitizeMessage,
    sesClients
  }
};
