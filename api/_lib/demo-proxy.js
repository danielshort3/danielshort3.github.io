'use strict';

const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');
const { AWS_WORKLOADS, getAwsClientConfig } = require('./aws-credentials');
const {
  applyBoundaryHeaders,
  clientIp,
  headerValue,
  httpError,
  isSameOriginRequest,
  readJsonBody,
  sendJson
} = require('./http-boundary');
const {
  booleanValue,
  consumeRequestLimit,
  positiveNumber
} = require('./request-rate-limit');

const MAX_RESPONSE_BYTES = 6 * 1024 * 1024;
const INTERNAL_DEMO_ORIGIN = 'https://danielshort.me';
const lambdaClients = new Map();

function operation(path, methods, options = {}) {
  return Object.freeze({
    path,
    methods: Object.freeze(methods),
    maxBodyBytes: options.maxBodyBytes || 0,
    timeoutMs: options.timeoutMs || 30_000,
    rateCategory: options.rateCategory || '',
    query: Object.freeze(options.query || {})
  });
}

const HEALTH = operation('/health', ['GET', 'HEAD'], { timeoutMs: 12_000 });
const WARMUP_HEAVY = operation('/warmup', ['POST'], {
  maxBodyBytes: 128_000,
  timeoutMs: 60_000,
  rateCategory: 'heavy'
});
const DATE_QUERY = Object.freeze({
  date: (value) => /^\d{4}-\d{2}-\d{2}$/.test(value)
});
const STATE_DATE_QUERY = Object.freeze({
  state: (value) => /^[A-Za-z]{2}$/.test(value),
  ...DATE_QUERY
});

const DEMO_REGISTRY = Object.freeze({
  shape: Object.freeze({
    functionArnEnv: 'DEMO_SHAPE_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_SHAPE_LEGACY_URL',
    legacyUrl: 'https://zcosyfxhs3sntpwzo3qayki2he0kdxkw.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: operation('/warmup', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      predict: operation('/predict', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      classify: operation('/classify', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' })
    })
  }),
  'smart-sentence': Object.freeze({
    functionArnEnv: 'DEMO_SMART_SENTENCE_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_SMART_SENTENCE_LEGACY_URL',
    legacyUrl: 'https://7aa3jt3tzkinheprc5ds52bihq0itutx.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      // The image performs an expensive first forward pass after cold start.
      // Warm it through the fixed /rank route before enabling the query form.
      warmup: operation('/rank', ['POST'], { maxBodyBytes: 24_000, timeoutMs: 55_000, rateCategory: 'heavy' }),
      rank: operation('/rank', ['POST'], { maxBodyBytes: 24_000, timeoutMs: 50_000, rateCategory: 'heavy' })
    })
  }),
  nonogram: Object.freeze({
    functionArnEnv: 'DEMO_NONOGRAM_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_NONOGRAM_LEGACY_URL',
    legacyUrl: 'https://gw676aqgd4lpcoxngfvnfki5ce0urcfg.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: WARMUP_HEAVY,
      solve: operation('/solve', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      predict: operation('/predict', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' })
    })
  }),
  handwriting: Object.freeze({
    functionArnEnv: 'DEMO_HANDWRITING_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_HANDWRITING_LEGACY_URL',
    legacyUrl: 'https://wklw5cb75voj4idwbnigdt6p7a0zrxwv.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: operation('/warmup', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      score: operation('/score', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 1_500_000, timeoutMs: 60_000, rateCategory: 'heavy' })
    })
  }),
  'digit-generator': Object.freeze({
    functionArnEnv: 'DEMO_DIGIT_GENERATOR_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_DIGIT_GENERATOR_LEGACY_URL',
    legacyUrl: 'https://w7jbsr55oggohong42ii6wo3jq0bvkvq.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: WARMUP_HEAVY,
      generate: operation('/generate', ['POST'], { maxBodyBytes: 64_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      sample: operation('/sample', ['POST'], { maxBodyBytes: 64_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 64_000, timeoutMs: 60_000, rateCategory: 'heavy' })
    })
  }),
  'covid-outbreak': Object.freeze({
    functionArnEnv: 'DEMO_COVID_OUTBREAK_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_COVID_OUTBREAK_LEGACY_URL',
    legacyUrl: 'https://lv4inwnj6yyo3kfdajpk2v5eda0hrtao.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      meta: operation('/meta', ['GET'], { timeoutMs: 30_000, rateCategory: 'data' }),
      states: operation('/states', ['GET'], { timeoutMs: 30_000, rateCategory: 'data', query: DATE_QUERY }),
      query: operation('/query', ['GET', 'POST'], {
        maxBodyBytes: 32_000,
        timeoutMs: 30_000,
        rateCategory: 'data',
        query: STATE_DATE_QUERY
      })
    }),
    dynamicOperation(value) {
      const match = /^state\/([A-Za-z]{2})$/.exec(value);
      if (!match) return null;
      const state = match[1].toUpperCase();
      return {
        operation: `state/${state}`,
        rule: operation(`/state/${state}`, ['GET'], { timeoutMs: 30_000, rateCategory: 'data', query: DATE_QUERY })
      };
    }
  }),
  'pizza-tips': Object.freeze({
    functionArnEnv: 'DEMO_PIZZA_TIPS_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_PIZZA_TIPS_LEGACY_URL',
    legacyUrl: 'https://2d6lrg4ozy564xymwi2epjbqty0nvhje.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: WARMUP_HEAVY,
      predict: operation('/predict', ['POST'], { maxBodyBytes: 64_000, timeoutMs: 45_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 64_000, timeoutMs: 45_000, rateCategory: 'heavy' })
    })
  }),
  'target-empty-package': Object.freeze({
    functionArnEnv: 'DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_TARGET_EMPTY_PACKAGE_LEGACY_URL',
    legacyUrl: 'https://e7axe2ymmx2kadovfvvji23wkq0xmybn.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      data: operation('/data', ['GET'], { timeoutMs: 30_000, rateCategory: 'data' }),
      meta: operation('/meta', ['GET'], { timeoutMs: 30_000, rateCategory: 'data' })
    })
  }),
  'retail-loss-sales': Object.freeze({
    functionArnEnv: 'DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_RETAIL_LOSS_SALES_LEGACY_URL',
    legacyUrl: 'https://oafstar74okqvfqaxayeici3ry0yikkx.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      data: operation('/data', ['GET'], { timeoutMs: 30_000, rateCategory: 'data' }),
      meta: operation('/meta', ['GET'], { timeoutMs: 30_000, rateCategory: 'data' })
    })
  }),
  minesweeper: Object.freeze({
    functionArnEnv: 'DEMO_MINESWEEPER_FUNCTION_ARN',
    legacyUrlEnv: 'DEMO_MINESWEEPER_LEGACY_URL',
    legacyUrl: 'https://jnvd3mdbyb5f44yh4afzsvqlwy0mdtzy.lambda-url.us-east-2.on.aws/',
    operations: Object.freeze({
      health: HEALTH,
      warmup: WARMUP_HEAVY,
      solve: operation('/solve', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      predict: operation('/predict', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' }),
      invoke: operation('/', ['POST'], { maxBodyBytes: 128_000, timeoutMs: 60_000, rateCategory: 'heavy' })
    })
  })
});

function slugSegments(req) {
  const querySlug = req && req.query && (
    typeof req.query.slug !== 'undefined' ? req.query.slug : req.query['...slug']
  );
  const rawSegments = Array.isArray(querySlug)
    ? querySlug
    : (typeof querySlug === 'string' ? querySlug.split('/') : null);
  if (rawSegments) return rawSegments.map((value) => String(value).trim()).filter(Boolean);

  try {
    const url = new URL(req.url, 'https://example.invalid');
    const match = /^\/api\/demos\/(.+)$/.exec(url.pathname);
    return match ? match[1].split('/').map((value) => decodeURIComponent(value)).filter(Boolean) : [];
  } catch {
    return [];
  }
}

function resolveDemoRoute(req) {
  const segments = slugSegments(req);
  if (segments.length < 2 || segments.length > 4) return null;
  const project = String(segments.shift() || '').toLowerCase();
  if (!/^[a-z0-9-]{2,40}$/.test(project)) return null;
  const config = DEMO_REGISTRY[project];
  if (!config) return null;

  const requestedOperation = segments.join('/');
  if (!requestedOperation || !/^[A-Za-z0-9/-]{1,80}$/.test(requestedOperation)) return null;
  const exactKey = requestedOperation.toLowerCase();
  if (config.operations[exactKey]) {
    return { project, operation: exactKey, config, rule: config.operations[exactKey] };
  }
  const dynamic = typeof config.dynamicOperation === 'function'
    ? config.dynamicOperation(requestedOperation)
    : null;
  if (!dynamic) return null;
  return { project, operation: dynamic.operation, config, rule: dynamic.rule };
}

function validateQuery(req, rule) {
  const validators = rule.query || {};
  const allowedKeys = new Set(Object.keys(validators));
  const input = new URL(req.url || '/', 'https://example.invalid').searchParams;
  // Vercel's frameworkless catch-all router exposes the matched path through
  // an internal query key. It is routing metadata, not caller input.
  input.delete('slug');
  input.delete('...slug');
  const output = new URLSearchParams();
  for (const key of new Set(input.keys())) {
    const values = input.getAll(key);
    if (!allowedKeys.has(key) || values.length !== 1 || !validators[key](values[0])) {
      throw httpError(400, 'DEMO_QUERY_INVALID', `Invalid query parameter: ${key}`);
    }
    output.set(key, values[0]);
  }
  return output;
}

function resolveProxyMode(env) {
  const explicit = String(env.DEMO_PROXY_MODE || '').trim().toLowerCase();
  if (explicit) {
    if (!['legacy', 'lambda'].includes(explicit)) {
      throw httpError(503, 'DEMO_PROXY_MODE_INVALID', 'Demo proxy is not configured');
    }
    return explicit;
  }
  return String(env.AWS_AUTH_MODE || '').trim().toLowerCase() === 'oidc' ? 'lambda' : 'legacy';
}

function isLiveAliasArn(value) {
  return /^arn:(?:aws|aws-us-gov|aws-cn):lambda:[a-z0-9-]+:\d{12}:function:[A-Za-z0-9-_]{1,64}:live$/.test(value);
}

function resolveFunctionArn(route, env) {
  const arn = String(env[route.config.functionArnEnv] || '').trim();
  if (!isLiveAliasArn(arn)) {
    throw httpError(503, 'DEMO_FUNCTION_ARN_INVALID', `${route.config.functionArnEnv} must be a :live-qualified Lambda ARN`);
  }
  return arn;
}

function filteredEventHeaders(req) {
  // The outer proxy has already enforced same-origin access. Use a fixed trusted
  // origin for legacy demo containers that still perform their own CORS check;
  // never forward a caller-controlled Origin header into the Lambda event.
  const headers = { origin: INTERNAL_DEMO_ORIGIN };
  for (const name of ['accept', 'content-type', 'user-agent', 'x-request-id']) {
    const value = headerValue(req, name);
    if (value) headers[name] = value.slice(0, 1000);
  }
  return headers;
}

function queryObject(searchParams) {
  const output = {};
  for (const [key, value] of searchParams.entries()) output[key] = value;
  return Object.keys(output).length ? output : undefined;
}

function buildLambdaEvent(req, route, rawBody, searchParams) {
  const requestMethod = String(req.method || 'GET').toUpperCase();
  const method = requestMethod === 'HEAD' ? 'GET' : requestMethod;
  const rawQueryString = searchParams.toString();
  const headers = filteredEventHeaders(req);
  return {
    version: '2.0',
    routeKey: `${method} ${route.rule.path}`,
    rawPath: route.rule.path,
    rawQueryString,
    headers,
    queryStringParameters: queryObject(searchParams),
    requestContext: {
      http: {
        method,
        path: route.rule.path,
        protocol: 'HTTP/1.1',
        sourceIp: clientIp(req),
        userAgent: headers['user-agent'] || ''
      }
    },
    body: rawBody || null,
    isBase64Encoded: false
  };
}

function getLambdaClient(env) {
  const region = String(env.DEMO_AWS_REGION || env.AWS_REGION || env.AWS_DEFAULT_REGION || 'us-east-2').trim();
  const aws = getAwsClientConfig(AWS_WORKLOADS.DEMO_INVOKE, { region });
  const key = `${region}:${aws.cacheKey}`;
  if (lambdaClients.has(key)) return lambdaClients.get(key);
  const client = new LambdaClient(aws.clientConfig);
  lambdaClients.set(key, client);
  return client;
}

function decodeLambdaPayload(response) {
  if (response && response.FunctionError) {
    const err = new Error('Lambda function reported an invocation error');
    err.code = 'DEMO_LAMBDA_FUNCTION_ERROR';
    throw err;
  }
  const raw = Buffer.from(response && response.Payload || []).toString('utf8');
  if (!raw) throw httpError(502, 'DEMO_LAMBDA_EMPTY', 'Demo service returned an empty response');

  let envelope;
  try {
    envelope = JSON.parse(raw);
  } catch {
    throw httpError(502, 'DEMO_LAMBDA_RESPONSE_INVALID', 'Demo service returned an invalid response');
  }

  if (!envelope || typeof envelope !== 'object' || typeof envelope.statusCode === 'undefined') {
    const body = Buffer.from(JSON.stringify(envelope));
    return { statusCode: 200, contentType: 'application/json; charset=utf-8', body };
  }

  const statusCode = Number(envelope.statusCode);
  if (!Number.isInteger(statusCode) || statusCode < 100 || statusCode > 599) {
    throw httpError(502, 'DEMO_LAMBDA_STATUS_INVALID', 'Demo service returned an invalid response');
  }
  let body;
  if (envelope.isBase64Encoded) {
    body = Buffer.from(String(envelope.body || ''), 'base64');
  } else if (typeof envelope.body === 'string') {
    body = Buffer.from(envelope.body, 'utf8');
  } else {
    body = Buffer.from(JSON.stringify(envelope.body ?? {}));
  }
  const headers = envelope.headers && typeof envelope.headers === 'object' ? envelope.headers : {};
  const contentType = String(headers['content-type'] || headers['Content-Type'] || 'application/json; charset=utf-8');
  return { statusCode, contentType, body };
}

async function invokeLambda(route, event, env) {
  const functionArn = resolveFunctionArn(route, env);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), route.rule.timeoutMs);
  try {
    const response = await getLambdaClient(env).send(new InvokeCommand({
      FunctionName: functionArn,
      InvocationType: 'RequestResponse',
      LogType: 'None',
      Payload: Buffer.from(JSON.stringify(event))
    }), { abortSignal: controller.signal });
    return decodeLambdaPayload(response);
  } finally {
    clearTimeout(timeout);
  }
}

function legacyUrl(route, env) {
  const configured = String(env[route.config.legacyUrlEnv] || '').trim();
  const raw = configured || route.config.legacyUrl;
  let base;
  try {
    base = new URL(raw);
  } catch {
    throw httpError(503, 'DEMO_LEGACY_URL_INVALID', 'Demo proxy is not configured');
  }
  if (base.protocol !== 'https:' || !/^[a-z0-9]+\.lambda-url\.[a-z0-9-]+\.on\.aws$/.test(base.hostname)) {
    throw httpError(503, 'DEMO_LEGACY_URL_INVALID', 'Demo proxy is not configured');
  }
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  return base;
}

async function fetchLegacy(route, event, env, fetchImpl) {
  const base = legacyUrl(route, env);
  const path = route.rule.path === '/' ? '' : route.rule.path.replace(/^\//, '');
  const url = new URL(path, base);
  url.search = event.rawQueryString;
  const headers = { Accept: 'application/json' };
  if (event.body) headers['Content-Type'] = 'application/json';
  const response = await fetchImpl(url.toString(), {
    method: event.requestContext.http.method,
    headers,
    ...(event.body ? { body: event.body } : {}),
    ...(typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
      ? { signal: AbortSignal.timeout(route.rule.timeoutMs) }
      : {})
  });
  const contentLength = Number(response.headers.get('content-length'));
  if (Number.isFinite(contentLength) && contentLength > MAX_RESPONSE_BYTES) {
    throw httpError(502, 'DEMO_RESPONSE_TOO_LARGE', 'Demo service response too large');
  }
  const body = Buffer.from(await response.arrayBuffer());
  return {
    statusCode: response.status,
    contentType: response.headers.get('content-type') || 'application/json; charset=utf-8',
    body
  };
}

function isJsonBody(body) {
  if (!body || body.length === 0) return true;
  try {
    JSON.parse(body.toString('utf8'));
    return true;
  } catch {
    return false;
  }
}

function sendProxyResponse(req, res, result) {
  const statusCode = Number(result && result.statusCode) || 502;
  const body = Buffer.from(result && result.body || []);
  if (statusCode >= 500) {
    sendJson(res, statusCode === 504 ? 504 : 502, { ok: false, error: 'Demo service unavailable' });
    return;
  }
  if (body.length > MAX_RESPONSE_BYTES || !isJsonBody(body)) {
    sendJson(res, 502, { ok: false, error: 'Demo service returned an invalid response' });
    return;
  }
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  if (String(req.method || '').toUpperCase() === 'HEAD' || statusCode === 204) {
    res.end();
    return;
  }
  res.end(body);
}

async function checkDemoRateLimit(req, route, env) {
  if (!route.rule.rateCategory) return { allowed: true };
  const hostedOidc = ['preview', 'production'].includes(String(env.VERCEL_ENV || '').toLowerCase())
    && String(env.AWS_AUTH_MODE || '').toLowerCase() === 'oidc';
  const requireTable = booleanValue(env.DEMO_REQUIRE_DDB_RATE_LIMIT, hostedOidc);
  const tableName = String(env.DEMO_RATE_LIMIT_TABLE || env.DEMO_RATE_LIMIT_TABLE_NAME || '').trim();
  const salt = String(env.DEMO_HASH_SALT || '').trim();
  if (requireTable && !salt) {
    const err = new Error('DEMO_HASH_SALT is not configured');
    err.code = 'DEMO_HASH_SALT_MISSING';
    throw err;
  }
  const heavy = route.rule.rateCategory === 'heavy';
  return consumeRequestLimit(req, {
    namespace: `demo#${route.project}`,
    globalNamespace: 'demo',
    windowSeconds: positiveNumber(env.DEMO_RATE_LIMIT_WINDOW_SECONDS, 600),
    windowLimit: positiveNumber(
      heavy ? env.DEMO_HEAVY_WINDOW_LIMIT : env.DEMO_DATA_WINDOW_LIMIT,
      heavy ? 40 : 120
    ),
    dailyLimit: positiveNumber(
      heavy ? env.DEMO_HEAVY_DAILY_LIMIT : env.DEMO_DATA_DAILY_LIMIT,
      heavy ? 200 : 1000
    ),
    globalDailyLimit: positiveNumber(env.DEMO_GLOBAL_DAILY_LIMIT, 3000),
    ttlDays: positiveNumber(env.DEMO_RATE_LIMIT_TTL_DAYS, 3),
    tableName,
    requireTable,
    salt: salt || String(env.VERCEL_PROJECT_PRODUCTION_URL || env.VERCEL_URL || 'local-demo'),
    workload: AWS_WORKLOADS.DEMO_INVOKE,
    region: String(env.DEMO_AWS_REGION || env.AWS_REGION || env.AWS_DEFAULT_REGION || 'us-east-2').trim()
  });
}

function createDemoProxyHandler(dependencies = {}) {
  const env = dependencies.env || process.env;
  const rateLimit = dependencies.checkRateLimit || checkDemoRateLimit;
  const lambdaInvoker = dependencies.invokeLambda || invokeLambda;
  const legacyFetcher = dependencies.fetchLegacy || fetchLegacy;
  const fetchImpl = dependencies.fetch || globalThis.fetch;

  return async function demoProxyHandler(req, res) {
    const boundaryOptions = {
      env,
      allowedOriginsEnv: 'DEMO_ALLOWED_ORIGINS',
      methods: ['GET', 'HEAD', 'POST', 'OPTIONS']
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

    const route = resolveDemoRoute(req);
    if (!route) {
      sendJson(res, 404, { ok: false, error: 'Demo operation not found' });
      return;
    }
    const requestMethod = String(req.method || 'GET').toUpperCase();
    if (!route.rule.methods.includes(requestMethod)) {
      res.setHeader('Allow', [...route.rule.methods, 'OPTIONS'].join(', '));
      sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
      return;
    }

    let searchParams;
    let rawBody = '';
    try {
      searchParams = validateQuery(req, route.rule);
      if (requestMethod === 'POST') {
        const body = await readJsonBody(req, route.rule.maxBodyBytes, { allowEmpty: true });
        rawBody = JSON.stringify(body.value);
      } else if (Number(headerValue(req, 'content-length')) > 0) {
        throw httpError(400, 'DEMO_BODY_NOT_ALLOWED', 'Request body is not allowed');
      }
    } catch (err) {
      sendJson(res, err.statusCode || 400, { ok: false, error: err.message || 'Invalid request' });
      return;
    }

    let limit;
    try {
      limit = await rateLimit(req, route, env);
    } catch (err) {
      console.error('Demo rate-limit check failed', {
        project: route.project,
        operation: route.operation,
        code: err && err.code,
        name: err && err.name
      });
      sendJson(res, 503, { ok: false, error: 'Demo service unavailable' });
      return;
    }
    if (!limit.allowed) {
      res.setHeader('Retry-After', String(limit.retryAfter || 60));
      sendJson(res, 429, { ok: false, error: limit.reason || 'Too many demo requests. Please try again later.' });
      return;
    }

    let mode = '';
    try {
      mode = resolveProxyMode(env);
      const event = buildLambdaEvent(req, route, rawBody, searchParams);
      const result = mode === 'lambda'
        ? await lambdaInvoker(route, event, env)
        : await legacyFetcher(route, event, env, fetchImpl);
      sendProxyResponse(req, res, result);
    } catch (err) {
      const timeout = err && (err.name === 'TimeoutError' || err.name === 'AbortError');
      const config = err && Number(err.statusCode) === 503;
      console.error('Demo proxy invocation failed', {
        project: route.project,
        operation: route.operation,
        mode,
        code: err && err.code,
        name: err && err.name
      });
      sendJson(res, config ? 503 : (timeout ? 504 : 502), {
        ok: false,
        error: config ? 'Demo proxy is not configured' : (timeout ? 'Demo service timed out' : 'Demo service unavailable')
      });
    }
  };
}

module.exports = {
  DEMO_REGISTRY,
  buildLambdaEvent,
  checkDemoRateLimit,
  createDemoProxyHandler,
  decodeLambdaPayload,
  fetchLegacy,
  invokeLambda,
  isLiveAliasArn,
  legacyUrl,
  resolveDemoRoute,
  resolveFunctionArn,
  resolveProxyMode,
  sendProxyResponse,
  validateQuery,
  _private: {
    INTERNAL_DEMO_ORIGIN,
    MAX_RESPONSE_BYTES,
    filteredEventHeaders,
    getLambdaClient,
    isJsonBody,
    lambdaClients,
    operation,
    queryObject,
    slugSegments
  }
};
