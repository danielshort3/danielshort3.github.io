'use strict';

const crypto = require('crypto');
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { resolveAwsCredentials } = require('./aws-credentials');

const SITE_ORIGIN = 'https://www.danielshort.me';
const DEFAULT_REGION = 'us-east-2';
const DEFAULT_AUDIENCE = 'sts.amazonaws.com';
const MAX_JSON_BODY_BYTES = 512 * 1024;
const MAX_JSON_RESPONSE_BYTES = 3 * 1024 * 1024;
const VERCEL_ROUTE_QUERY_KEYS = new Set([
  'slug',
  '...slug',
  '[...slug]',
  'demo',
  'action',
  'state',
  'x-vercel-protection-bypass',
  'x-vercel-set-bypass-cookie'
]);

const ENABLED_PROXY_MODES = new Set(['oidc', 'iam', 'private', 'role', 'enabled', 'true']);
const DISABLED_PROXY_MODES = new Set(['disabled', 'off', 'false']);
const DEMO_STATIC_CREDENTIAL_SETS = Object.freeze([
  Object.freeze({
    name: 'demo',
    accessKeyId: 'DEMO_AWS_ACCESS_KEY_ID',
    secretAccessKey: 'DEMO_AWS_SECRET_ACCESS_KEY',
    sessionToken: 'DEMO_AWS_SESSION_TOKEN'
  }),
  Object.freeze({
    name: 'default',
    accessKeyId: 'AWS_ACCESS_KEY_ID',
    secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
    sessionToken: 'AWS_SESSION_TOKEN'
  })
]);

const RATE_LIMITS = Object.freeze({
  health: { limit: 30, windowSeconds: 60 },
  warmup: { limit: 6, windowSeconds: 600 },
  inference: { limit: 60, windowSeconds: 600 },
  data: { limit: 120, windowSeconds: 600 }
});

const DEMO_MANIFEST = Object.freeze({
  shape: {
    envKey: 'DEMO_SHAPE_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 512 * 1024, 512 * 1024, 120_000),
      route('predict', ['POST'], '/predict', 'inference', 512 * 1024, 1024 * 1024, 120_000)
    ]
  },
  'digit-generator': {
    envKey: 'DEMO_DIGIT_GENERATOR_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 32 * 1024, 512 * 1024, 120_000),
      route('generate', ['POST'], '/generate', 'inference', 32 * 1024, 2 * 1024 * 1024, 120_000)
    ]
  },
  handwriting: {
    envKey: 'DEMO_HANDWRITING_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 512 * 1024, 512 * 1024, 120_000),
      route('score', ['POST'], '/score', 'inference', 512 * 1024, 1024 * 1024, 120_000)
    ]
  },
  minesweeper: {
    envKey: 'DEMO_MINESWEEPER_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 32 * 1024, 512 * 1024, 120_000),
      route('solve', ['POST'], '/solve', 'inference', 32 * 1024, 2 * 1024 * 1024, 120_000)
    ]
  },
  nonogram: {
    envKey: 'DEMO_NONOGRAM_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 32 * 1024, 512 * 1024, 120_000),
      route('solve', ['POST'], '/solve', 'inference', 32 * 1024, 2 * 1024 * 1024, 120_000)
    ]
  },
  'pizza-tips': {
    envKey: 'DEMO_PIZZA_TIPS_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('warmup', ['POST'], '/warmup', 'warmup', 32 * 1024, 512 * 1024, 60_000),
      route('predict', ['POST'], '/predict', 'inference', 32 * 1024, 1024 * 1024, 60_000)
    ]
  },
  'smart-sentence': {
    envKey: 'DEMO_SMART_SENTENCE_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('rank', ['POST'], '/rank', 'inference', 24 * 1024, 1024 * 1024, 60_000)
    ]
  },
  'covid-outbreak': {
    envKey: 'DEMO_COVID_OUTBREAK_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('meta', ['GET', 'HEAD'], '/meta', 'data', 0, 512 * 1024, 30_000),
      route('states', ['GET', 'HEAD'], '/states', 'data', 0, 512 * 1024, 30_000, ['date']),
      dynamicRoute(/^state\/([A-Za-z]{2})$/, ['GET', 'HEAD'], (match) => `/state/${match[1].toUpperCase()}`, 'data', 0, 512 * 1024, 30_000, ['date'])
    ]
  },
  'target-empty-package': {
    envKey: 'DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('data', ['GET', 'HEAD'], '/data', 'data', 0, 2 * 1024 * 1024, 45_000)
    ]
  },
  'retail-loss-sales': {
    envKey: 'DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN',
    routes: [
      route('health', ['GET', 'HEAD'], '/health', 'health', 0, 256 * 1024, 25_000),
      route('data', ['GET', 'HEAD'], '/data', 'data', 0, 1024 * 1024, 45_000)
    ]
  }
});

const memoryRateStore = new Map();
let cachedClients = null;
let clientFactoryOverride = null;

function route(path, methods, upstreamPath, rateClass, maxBodyBytes, maxResponseBytes, timeoutMs, queryKeys = []) {
  return Object.freeze({
    path,
    methods: Object.freeze(methods.slice()),
    upstreamPath,
    rateClass,
    maxBodyBytes,
    maxResponseBytes,
    timeoutMs,
    queryKeys: Object.freeze(queryKeys.slice())
  });
}

function dynamicRoute(pattern, methods, upstreamPath, rateClass, maxBodyBytes, maxResponseBytes, timeoutMs, queryKeys = []) {
  return Object.freeze({
    pattern,
    methods: Object.freeze(methods.slice()),
    upstreamPath,
    rateClass,
    maxBodyBytes,
    maxResponseBytes,
    timeoutMs,
    queryKeys: Object.freeze(queryKeys.slice())
  });
}

function pickEnv(keys, env = process.env) {
  for (const key of keys) {
    const raw = env[key];
    if (typeof raw === 'string' && raw.trim()) return raw.trim();
  }
  return '';
}

function boolEnv(key, fallback, env = process.env) {
  const raw = String(env[key] || '').trim().toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(raw)) return true;
  if (['0', 'false', 'no', 'off'].includes(raw)) return false;
  return fallback;
}

function isProductionRuntime(env = process.env) {
  return String(env.VERCEL_ENV || '').trim().toLowerCase() === 'production';
}

function proxyMode(env = process.env, functionArn = '') {
  const raw = String(env.DEMO_PROXY_MODE || '').trim().toLowerCase();
  if (DISABLED_PROXY_MODES.has(raw)) return { enabled: false, code: 'DEMO_PROXY_DISABLED' };
  if (raw && !ENABLED_PROXY_MODES.has(raw)) return { enabled: false, code: 'DEMO_PROXY_MODE_INVALID' };
  if (!functionArn) return { enabled: false, code: 'DEMO_PROXY_CONFIG_MISSING' };
  return { enabled: true, code: raw || 'auto' };
}

function normalizeSegments(values) {
  const list = Array.isArray(values) ? values : String(values || '').split('/');
  const decoded = [];
  for (const value of list) {
    if (!value) continue;
    let segment;
    try {
      segment = decodeURIComponent(String(value));
    } catch {
      return [];
    }
    if (!segment || segment === '.' || segment === '..' || /[\\/\0]/.test(segment)) return [];
    decoded.push(segment);
  }
  return decoded;
}

function getRequestUrl(req) {
  try {
    return new URL(req.url || '/', SITE_ORIGIN);
  } catch {
    return new URL('/', SITE_ORIGIN);
  }
}

function getQueryEntries(req) {
  const url = getRequestUrl(req);
  return Array.from(url.searchParams.entries()).filter(([key]) => !VERCEL_ROUTE_QUERY_KEYS.has(key));
}

function validateQuery(entries, allowedKeys) {
  const allowed = new Set(allowedKeys || []);
  const output = {};
  for (const [key, value] of entries) {
    if (!allowed.has(key) || Object.prototype.hasOwnProperty.call(output, key)) {
      const safeKey = String(key || '').replace(/[^A-Za-z0-9_.-]/g, '').slice(0, 64) || 'unknown';
      return { ok: false, error: `Unsupported query parameter: ${safeKey}.` };
    }
    if (key === 'date' && !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
      return { ok: false, error: 'Invalid date query parameter.' };
    }
    if (Buffer.byteLength(value, 'utf8') > 128) {
      return { ok: false, error: 'Query parameter is too long.' };
    }
    output[key] = value;
  }
  return { ok: true, value: output };
}

function resolveDemoRoute(segments, method, queryEntries = []) {
  const normalized = normalizeSegments(segments);
  if (normalized.length < 2) return { ok: false, status: 404, error: 'Not Found' };
  const demoId = normalized[0];
  const demo = DEMO_MANIFEST[demoId];
  if (!demo) return { ok: false, status: 404, error: 'Not Found' };

  const requestPath = normalized.slice(1).join('/');
  let matchedRoute = null;
  let match = null;
  for (const candidate of demo.routes) {
    if (candidate.path === requestPath) {
      matchedRoute = candidate;
      break;
    }
    if (candidate.pattern) {
      const dynamicMatch = requestPath.match(candidate.pattern);
      if (dynamicMatch) {
        matchedRoute = candidate;
        match = dynamicMatch;
        break;
      }
    }
  }
  if (!matchedRoute) return { ok: false, status: 404, error: 'Not Found' };

  const normalizedMethod = String(method || 'GET').toUpperCase();
  if (!matchedRoute.methods.includes(normalizedMethod)) {
    return {
      ok: false,
      status: 405,
      error: 'Method Not Allowed',
      allow: [...new Set([...matchedRoute.methods, 'OPTIONS'])].join(', ')
    };
  }

  const query = validateQuery(queryEntries, matchedRoute.queryKeys);
  if (!query.ok) return { ok: false, status: 400, error: query.error };
  const upstreamPath = typeof matchedRoute.upstreamPath === 'function'
    ? matchedRoute.upstreamPath(match)
    : matchedRoute.upstreamPath;

  return {
    ok: true,
    demoId,
    demo,
    route: matchedRoute,
    method: normalizedMethod,
    upstreamPath,
    query: query.value
  };
}

function sendJson(res, statusCode, payload, extraHeaders = {}) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  for (const [key, value] of Object.entries(extraHeaders)) res.setHeader(key, value);
  res.end(JSON.stringify(payload));
}

function requestOrigin(req) {
  return String(req?.headers?.origin || '').trim();
}

function expectedOrigin(req) {
  const headers = req?.headers || {};
  const host = String(headers['x-forwarded-host'] || headers.host || '').split(',')[0].trim();
  if (!host) return '';
  const proto = String(headers['x-forwarded-proto'] || '').split(',')[0].trim()
    || (/^(localhost|127\.0\.0\.1|\[::1\])(?::|$)/i.test(host) ? 'http' : 'https');
  return `${proto}://${host}`;
}

function isAllowedOrigin(req) {
  const fetchSite = String(req?.headers?.['sec-fetch-site'] || '').trim().toLowerCase();
  if (fetchSite === 'cross-site') return false;
  const origin = requestOrigin(req);
  if (!origin) return true;
  return origin === expectedOrigin(req);
}

function createBodyTooLargeError(maxBytes) {
  const err = new Error(`JSON request body exceeds ${maxBytes} bytes.`);
  err.code = 'BODY_TOO_LARGE';
  err.statusCode = 413;
  return err;
}

function assertBodyWithinLimit(value, maxBytes) {
  const raw = typeof value === 'string' ? value : JSON.stringify(value || {});
  if (Buffer.byteLength(raw, 'utf8') > maxBytes) throw createBodyTooLargeError(maxBytes);
  return raw;
}

async function readJsonBody(req, maxBytes) {
  if (!maxBytes) return {};
  const contentType = String(req?.headers?.['content-type'] || '').toLowerCase();
  if (contentType && !contentType.includes('application/json')) {
    const err = new Error('Content-Type must be application/json.');
    err.statusCode = 415;
    throw err;
  }
  const declared = Number(req?.headers?.['content-length']);
  if (Number.isFinite(declared) && declared > maxBytes) throw createBodyTooLargeError(maxBytes);

  if (req.body && typeof req.body === 'object') {
    assertBodyWithinLimit(req.body, maxBytes);
    if (Array.isArray(req.body)) throw new Error('JSON request body must be an object.');
    return req.body;
  }
  if (typeof req.body === 'string') {
    assertBodyWithinLimit(req.body, maxBytes);
    const parsed = req.body ? JSON.parse(req.body) : {};
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('JSON request body must be an object.');
    return parsed;
  }

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > maxBytes) throw createBodyTooLargeError(maxBytes);
    chunks.push(buffer);
  }
  if (!chunks.length) return {};
  const parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('JSON request body must be an object.');
  return parsed;
}

function getClientIp(req) {
  const headers = req?.headers || {};
  const forwarded = String(headers['x-forwarded-for'] || '').split(',')[0].trim();
  const real = String(headers['x-real-ip'] || '').trim();
  const direct = String(req?.socket?.remoteAddress || '').trim();
  return forwarded || real || direct || 'unknown';
}

function getActorHash(req, env = process.env) {
  const salt = pickEnv(['DEMO_HASH_SALT'], env) || 'local-demo-salt';
  return crypto.createHmac('sha256', salt).update(getClientIp(req)).digest('hex').slice(0, 32);
}

function getRuntimeConfig(env = process.env, functionArn = '') {
  const region = pickEnv(['DEMO_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION'], env) || DEFAULT_REGION;
  const roleArn = pickEnv(['DEMO_INVOKE_AWS_ROLE_ARN'], env);
  const tableName = pickEnv(['DEMO_RATE_LIMIT_TABLE'], env);
  const hashSalt = pickEnv(['DEMO_HASH_SALT'], env);
  const audience = pickEnv(['AWS_OIDC_AUDIENCE'], env) || DEFAULT_AUDIENCE;
  const requireDdb = boolEnv('DEMO_REQUIRE_DDB_RATE_LIMIT', isProductionRuntime(env), env);
  const mode = proxyMode(env, functionArn);
  if (!mode.enabled) {
    const err = new Error('Demo proxy is unavailable.');
    err.code = mode.code;
    err.statusCode = 503;
    throw err;
  }
  if (requireDdb && (!tableName || !hashSalt)) {
    const err = new Error('Demo proxy protection is not configured.');
    err.code = 'DEMO_RATE_LIMIT_CONFIG_MISSING';
    err.statusCode = 503;
    throw err;
  }
  const auth = resolveAwsCredentials({
    env,
    service: 'demo-invoke',
    region,
    roleArnEnvKeys: ['DEMO_INVOKE_AWS_ROLE_ARN'],
    staticCredentialSets: DEMO_STATIC_CREDENTIAL_SETS
  });
  const vercelEnvironment = String(env.VERCEL_ENV || '').trim().toLowerCase();
  if (['production', 'preview'].includes(vercelEnvironment) && auth.source === 'default') {
    const err = new Error('Demo proxy AWS authentication is not configured.');
    err.code = 'DEMO_AWS_AUTH_MISSING';
    err.statusCode = 503;
    throw err;
  }
  return {
    region,
    roleArn,
    tableName,
    hashSalt,
    audience: auth.audience || audience,
    requireDdb,
    auth
  };
}

function createClients(config) {
  if (clientFactoryOverride) return clientFactoryOverride(config);
  const key = `${config.region}:${config.auth.cacheKey}`;
  if (cachedClients && cachedClients.key === key) return cachedClients;
  const credentials = config.auth.credentials;
  const lambda = new LambdaClient({ region: config.region, credentials });
  const ddb = DynamoDBDocumentClient.from(
    new DynamoDBClient({ region: config.region, credentials }),
    { marshallOptions: { removeUndefinedValues: true } }
  );
  cachedClients = { key, lambda, ddb };
  return cachedClients;
}

function rateLimitConfig(rateClass) {
  return RATE_LIMITS[rateClass] || RATE_LIMITS.inference;
}

function conditionalCheckFailed(err) {
  return err && (err.name === 'ConditionalCheckFailedException' || err.Code === 'ConditionalCheckFailedException');
}

function consumeMemoryRateLimit(key, limit, expiresAt) {
  const current = memoryRateStore.get(key);
  const entry = !current || current.expiresAt <= Date.now()
    ? { count: 0, expiresAt }
    : current;
  if (entry.count >= limit) return { allowed: false, count: entry.count };
  entry.count += 1;
  memoryRateStore.set(key, entry);
  if (memoryRateStore.size > 5_000) {
    const now = Date.now();
    for (const [storedKey, stored] of memoryRateStore) {
      if (stored.expiresAt <= now) memoryRateStore.delete(storedKey);
    }
  }
  return { allowed: true, count: entry.count };
}

async function consumeRateLimit(req, resolved, config, clients) {
  const policy = rateLimitConfig(resolved.route.rateClass);
  const nowMs = Date.now();
  const nowSeconds = Math.floor(nowMs / 1000);
  const windowId = Math.floor(nowSeconds / policy.windowSeconds);
  const resetAtSeconds = (windowId + 1) * policy.windowSeconds;
  const retryAfterSeconds = Math.max(1, resetAtSeconds - nowSeconds);
  const actorHash = getActorHash(req, { ...process.env, DEMO_HASH_SALT: config.hashSalt || '' });
  const pk = `DEMO#ACTOR#${actorHash}`;
  const sk = `${resolved.demoId.toUpperCase()}#${resolved.route.rateClass.toUpperCase()}#${windowId}`;

  if (!config.tableName) {
    const memory = consumeMemoryRateLimit(`${pk}|${sk}`, policy.limit, resetAtSeconds * 1000);
    return { ...memory, limit: policy.limit, retryAfterSeconds };
  }

  try {
    const result = await clients.ddb.send(new UpdateCommand({
      TableName: config.tableName,
      Key: { pk, sk },
      UpdateExpression: 'SET #ttl = :ttl, #updatedAt = :now ADD #count :one',
      ConditionExpression: 'attribute_not_exists(#count) OR #count < :limit',
      ExpressionAttributeNames: {
        '#ttl': 'ttl',
        '#updatedAt': 'updatedAt',
        '#count': 'count'
      },
      ExpressionAttributeValues: {
        ':ttl': resetAtSeconds + 86400,
        ':now': nowMs,
        ':one': 1,
        ':limit': policy.limit
      },
      ReturnValues: 'ALL_NEW'
    }));
    return {
      allowed: true,
      count: Number(result?.Attributes?.count || 1),
      limit: policy.limit,
      retryAfterSeconds
    };
  } catch (err) {
    if (conditionalCheckFailed(err)) {
      return { allowed: false, count: policy.limit, limit: policy.limit, retryAfterSeconds };
    }
    if (config.requireDdb) throw err;
    const memory = consumeMemoryRateLimit(`${pk}|${sk}`, policy.limit, resetAtSeconds * 1000);
    return { ...memory, limit: policy.limit, retryAfterSeconds };
  }
}

function buildLambdaEvent(req, resolved, body) {
  const query = resolved.query || {};
  const rawQueryString = new URLSearchParams(query).toString();
  const requestId = crypto.randomUUID();
  const hasBody = resolved.route.maxBodyBytes > 0 && resolved.method !== 'HEAD';
  return {
    version: '2.0',
    routeKey: '$default',
    rawPath: resolved.upstreamPath,
    rawQueryString,
    headers: {
      accept: 'application/json',
      'content-type': 'application/json',
      origin: SITE_ORIGIN,
      'x-request-id': requestId
    },
    queryStringParameters: Object.keys(query).length ? query : undefined,
    requestContext: {
      accountId: 'anonymous',
      apiId: 'demo-proxy',
      domainName: 'www.danielshort.me',
      domainPrefix: 'www',
      http: {
        method: resolved.method === 'HEAD' ? 'GET' : resolved.method,
        path: resolved.upstreamPath,
        protocol: 'HTTP/1.1',
        sourceIp: '0.0.0.0',
        userAgent: 'danielshort-demo-proxy'
      },
      requestId,
      routeKey: '$default',
      stage: '$default',
      time: new Date().toUTCString(),
      timeEpoch: Date.now()
    },
    body: hasBody ? JSON.stringify(body || {}) : null,
    isBase64Encoded: false
  };
}

function decodeLambdaPayload(payload, maxResponseBytes = MAX_JSON_RESPONSE_BYTES) {
  const buffer = Buffer.from(payload || []);
  if (buffer.length > maxResponseBytes) {
    const err = new Error('Demo response exceeded the configured limit.');
    err.code = 'DEMO_RESPONSE_TOO_LARGE';
    err.statusCode = 502;
    throw err;
  }
  const raw = buffer.toString('utf8');
  if (!raw) {
    const err = new Error('Demo service returned an empty response.');
    err.code = 'DEMO_EMPTY_RESPONSE';
    err.statusCode = 502;
    throw err;
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    const err = new Error('Demo service returned invalid JSON.');
    err.code = 'DEMO_INVALID_RESPONSE';
    err.statusCode = 502;
    throw err;
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    const err = new Error('Demo service returned an invalid response.');
    err.code = 'DEMO_INVALID_RESPONSE';
    err.statusCode = 502;
    throw err;
  }

  if (!Object.prototype.hasOwnProperty.call(parsed, 'statusCode')) {
    const bodyText = JSON.stringify(parsed);
    if (Buffer.byteLength(bodyText, 'utf8') > maxResponseBytes) {
      const err = new Error('Demo response exceeded the configured limit.');
      err.code = 'DEMO_RESPONSE_TOO_LARGE';
      err.statusCode = 502;
      throw err;
    }
    return { statusCode: 200, bodyText };
  }

  const statusCode = Number(parsed.statusCode);
  if (!Number.isInteger(statusCode) || statusCode < 100 || statusCode > 599) {
    const err = new Error('Demo service returned an invalid status code.');
    err.code = 'DEMO_INVALID_RESPONSE';
    err.statusCode = 502;
    throw err;
  }
  let bodyText = '';
  if (typeof parsed.body === 'string') {
    bodyText = parsed.isBase64Encoded
      ? Buffer.from(parsed.body, 'base64').toString('utf8')
      : parsed.body;
  } else if (parsed.body !== undefined && parsed.body !== null) {
    bodyText = JSON.stringify(parsed.body);
  }
  if (bodyText && statusCode !== 204) {
    try {
      JSON.parse(bodyText);
    } catch {
      const err = new Error('Demo service returned an invalid JSON body.');
      err.code = 'DEMO_INVALID_RESPONSE';
      err.statusCode = 502;
      throw err;
    }
  }
  if (Buffer.byteLength(bodyText, 'utf8') > maxResponseBytes) {
    const err = new Error('Demo response exceeded the configured limit.');
    err.code = 'DEMO_RESPONSE_TOO_LARGE';
    err.statusCode = 502;
    throw err;
  }
  return { statusCode, bodyText };
}

async function invokeLambda(req, resolved, body, functionArn, config, clients) {
  const event = buildLambdaEvent(req, resolved, body);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), resolved.route.timeoutMs);
  let response;
  try {
    response = await clients.lambda.send(new InvokeCommand({
      FunctionName: functionArn,
      InvocationType: 'RequestResponse',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(event), 'utf8')
    }), { abortSignal: controller.signal });
  } catch (err) {
    if (err?.name === 'AbortError' || controller.signal.aborted) {
      const timeoutError = new Error('Demo service timed out.');
      timeoutError.code = 'DEMO_TIMEOUT';
      timeoutError.statusCode = 504;
      throw timeoutError;
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
  if (response?.FunctionError) {
    const err = new Error('Demo service invocation failed.');
    err.code = 'DEMO_FUNCTION_ERROR';
    err.statusCode = 502;
    throw err;
  }
  return decodeLambdaPayload(response?.Payload, resolved.route.maxResponseBytes);
}

function sanitizeUpstreamBody(statusCode, bodyText) {
  if (statusCode < 500 || !bodyText) return bodyText;
  return JSON.stringify({ ok: false, error: 'Demo service is temporarily unavailable.' });
}

async function handleDemoRequest(req, res, segments) {
  if (!isAllowedOrigin(req)) {
    sendJson(res, 403, { ok: false, error: 'Origin not allowed.' });
    return;
  }
  if (String(req.method || '').toUpperCase() === 'OPTIONS') {
    res.statusCode = 204;
    res.setHeader('Allow', 'GET, HEAD, POST, OPTIONS');
    res.setHeader('Cache-Control', 'no-store');
    res.end();
    return;
  }

  const resolved = resolveDemoRoute(segments, req.method, getQueryEntries(req));
  if (!resolved.ok) {
    const headers = resolved.allow ? { Allow: resolved.allow } : {};
    sendJson(res, resolved.status, { ok: false, error: resolved.error }, headers);
    return;
  }

  const functionArn = pickEnv([resolved.demo.envKey]);
  let config;
  let clients;
  try {
    config = getRuntimeConfig(process.env, functionArn);
    clients = createClients(config);
  } catch (err) {
    sendJson(res, err?.statusCode || 503, { ok: false, error: 'Demo proxy is unavailable.' });
    return;
  }

  let rate;
  try {
    rate = await consumeRateLimit(req, resolved, config, clients);
  } catch {
    sendJson(res, 503, { ok: false, error: 'Demo request protection is unavailable.' });
    return;
  }
  const rateHeaders = {
    'X-RateLimit-Limit': String(rate.limit),
    'X-RateLimit-Remaining': String(Math.max(0, rate.limit - Number(rate.count || 0)))
  };
  if (!rate.allowed) {
    rateHeaders['Retry-After'] = String(rate.retryAfterSeconds);
    sendJson(res, 429, {
      ok: false,
      error: 'Too many demo requests. Please wait and try again.',
      retryAfter: rate.retryAfterSeconds
    }, rateHeaders);
    return;
  }

  let body = {};
  if (resolved.route.maxBodyBytes > 0) {
    try {
      body = await readJsonBody(req, Math.min(resolved.route.maxBodyBytes, MAX_JSON_BODY_BYTES));
    } catch (err) {
      sendJson(res, err?.statusCode || 400, { ok: false, error: err?.message || 'Invalid JSON body.' }, rateHeaders);
      return;
    }
  }

  let upstream;
  try {
    upstream = await invokeLambda(req, resolved, body, functionArn, config, clients);
  } catch (err) {
    const statusCode = err?.statusCode || 502;
    sendJson(res, statusCode, {
      ok: false,
      error: statusCode === 504 ? 'Demo service timed out.' : 'Demo service is unavailable.'
    }, rateHeaders);
    return;
  }

  res.statusCode = upstream.statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  for (const [key, value] of Object.entries(rateHeaders)) res.setHeader(key, value);
  if (resolved.method === 'HEAD' || upstream.statusCode === 204) {
    res.end();
    return;
  }
  res.end(sanitizeUpstreamBody(upstream.statusCode, upstream.bodyText));
}

function setClientFactoryForTests(factory) {
  clientFactoryOverride = factory || null;
  cachedClients = null;
}

module.exports = {
  handleDemoRequest,
  _internal: {
    DEFAULT_AUDIENCE,
    DEMO_MANIFEST,
    RATE_LIMITS,
    assertBodyWithinLimit,
    buildLambdaEvent,
    consumeRateLimit,
    decodeLambdaPayload,
    getActorHash,
    getQueryEntries,
    getRuntimeConfig,
    isAllowedOrigin,
    memoryRateStore,
    normalizeSegments,
    proxyMode,
    resolveDemoRoute,
    setClientFactoryForTests,
    validateQuery
  }
};
