'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const {
  buildSesInput,
  createContactHandler,
  resolveDeliveryMode,
  validateContactPayload
} = require('../api/_lib/contact-service');
const {
  buildLambdaEvent,
  createDemoProxyHandler,
  decodeLambdaPayload,
  isLiveAliasArn,
  resolveDemoRoute,
  resolveFunctionArn,
  validateQuery
} = require('../api/_lib/demo-proxy');
const { consumeRequestLimit, _memoryStore } = require('../api/_lib/request-rate-limit');

let checks = 0;

function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function request(options = {}) {
  const body = typeof options.body === 'undefined' ? '' : options.body;
  const chunks = typeof body === 'string' || Buffer.isBuffer(body) ? [body] : [];
  const req = Readable.from(chunks);
  req.method = options.method || 'GET';
  req.url = options.url || '/';
  req.headers = {
    host: 'www.danielshort.me',
    'x-forwarded-proto': 'https',
    'user-agent': 'boundary-test',
    ...(options.headers || {})
  };
  req.query = options.query || {};
  req.socket = { remoteAddress: '203.0.113.10' };
  if (options.parsedBody) req.body = options.parsedBody;
  return req;
}

function response() {
  return {
    statusCode: 200,
    headers: {},
    body: Buffer.alloc(0),
    setHeader(name, value) {
      this.headers[String(name).toLowerCase()] = value;
    },
    getHeader(name) {
      return this.headers[String(name).toLowerCase()];
    },
    end(value) {
      this.body = typeof value === 'undefined'
        ? Buffer.alloc(0)
        : (Buffer.isBuffer(value) ? value : Buffer.from(String(value)));
      this.ended = true;
    }
  };
}

function jsonBody(res) {
  return res.body.length ? JSON.parse(res.body.toString('utf8')) : null;
}

async function contactTests() {
  const validation = validateContactPayload({
    name: 'Daniel\r\nBcc: ignored@example.com',
    email: 'visitor@example.com',
    message: 'Hello',
    company: ''
  });
  check(validation.valid, 'valid contact payload should pass');
  check(!validation.value.name.includes('\n'), 'contact name should strip header newlines');
  check(validateContactPayload({ company: 'bot', name: '', email: '', message: '' }).honeypot, 'honeypot should short-circuit');
  check(!validateContactPayload({ name: 'A', email: 'bad', message: 'Hello' }).valid, 'invalid email should fail');

  const sesInput = buildSesInput(validation.value, {
    CONTACT_SENDER_EMAIL: 'verified@example.com',
    CONTACT_RECIPIENT_EMAIL: 'owner@example.com,backup@example.com',
    CONTACT_EMAIL_SUBJECT_PREFIX: 'Portfolio'
  });
  check(sesInput.FromEmailAddress === 'verified@example.com', 'SES input should use verified sender');
  check(sesInput.Destination.ToAddresses.length === 2, 'SES input should support configured recipients');
  check(sesInput.ReplyToAddresses[0] === 'visitor@example.com', 'SES input should reply to visitor');
  check(resolveDeliveryMode({ AWS_AUTH_MODE: 'oidc' }) === 'ses', 'OIDC should default contact delivery to SES');
  check(resolveDeliveryMode({ AWS_AUTH_MODE: 'legacy' }) === 'legacy', 'legacy auth should preserve upstream delivery');

  let delivered = null;
  const handler = createContactHandler({
    env: {
      CONTACT_DELIVERY_MODE: 'ses',
      CONTACT_SENDER_EMAIL: 'verified@example.com',
      CONTACT_RECIPIENT_EMAIL: 'owner@example.com'
    },
    checkRateLimit: async () => ({ allowed: true }),
    sendSes: async (payload) => { delivered = payload; }
  });
  const req = request({
    method: 'POST',
    url: '/api/contact',
    headers: { origin: 'https://www.danielshort.me', 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'Visitor', email: 'visitor@example.com', message: 'Hello', company: '' })
  });
  const res = response();
  await handler(req, res);
  check(res.statusCode === 200 && jsonBody(res).ok === true, 'valid contact request should preserve success contract');
  check(delivered && delivered.email === 'visitor@example.com', 'valid contact request should call SES delivery');
  check(res.headers['cache-control'] === 'no-store', 'contact response should disable caching');

  const blockedRes = response();
  await handler(request({
    method: 'POST',
    url: '/api/contact',
    headers: { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site', 'content-type': 'application/json' },
    body: '{}'
  }), blockedRes);
  check(blockedRes.statusCode === 403, 'cross-site contact request should be rejected');

  let honeypotDeliveries = 0;
  const honeypotHandler = createContactHandler({
    env: { CONTACT_DELIVERY_MODE: 'ses' },
    checkRateLimit: async () => { throw new Error('honeypot should not rate limit'); },
    sendSes: async () => { honeypotDeliveries += 1; }
  });
  const honeypotRes = response();
  await honeypotHandler(request({
    method: 'POST',
    url: '/api/contact',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ company: 'spam company' })
  }), honeypotRes);
  check(honeypotRes.statusCode === 200 && jsonBody(honeypotRes).ok, 'honeypot should return indistinguishable success');
  check(honeypotDeliveries === 0, 'honeypot should not deliver email');

  const limitedHandler = createContactHandler({
    env: { CONTACT_DELIVERY_MODE: 'ses' },
    checkRateLimit: async () => ({ allowed: false, retryAfter: 90 }),
    sendSes: async () => { throw new Error('rate-limited request should not deliver'); }
  });
  const limitedRes = response();
  await limitedHandler(request({
    method: 'POST',
    url: '/api/contact',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'Visitor', email: 'visitor@example.com', message: 'Hello' })
  }), limitedRes);
  check(limitedRes.statusCode === 429, 'contact rate limit should return 429');
  check(limitedRes.headers['retry-after'] === '90', 'contact rate limit should set Retry-After');
}

async function demoTests() {
  const vercelConfig = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'vercel.json'), 'utf8'));
  const rewrites = vercelConfig.rewrites || [];
  check(rewrites.some((rule) => rule.source === '/api/demos/:project/:operation' && rule.destination.includes('%2F')), 'Vercel should encode two-part demo routes into the single catch-all segment');
  check(rewrites.some((rule) => rule.source === '/api/demos/:project/:first/:second' && rule.destination.includes('%2F:first%2F:second')), 'Vercel should encode three-part dynamic demo routes');
  check(rewrites.some((rule) => rule.source === '/api/sentence-demo/:operation' && rule.destination.includes('smart-sentence%2F')), 'legacy Smart Sentence routes should share the consolidated demo function');
  const covidReq = request({
    method: 'GET',
    url: '/api/demos/covid-outbreak/state/co?date=2026-01-03&...slug=covid-outbreak%2Fstate%2Fco'
  });
  const covidRoute = resolveDemoRoute(covidReq);
  check(covidRoute && covidRoute.project === 'covid-outbreak', 'known demo project should resolve');
  check(covidRoute.operation === 'state/CO', 'dynamic state operation should be normalized');
  check(covidRoute.rule.path === '/state/CO', 'dynamic operation should build a fixed Lambda path');
  check(validateQuery(covidReq, covidRoute.rule).get('date') === '2026-01-03', 'allowed demo query should pass');
  check(!validateQuery(covidReq, covidRoute.rule).has('...slug'), 'Vercel catch-all routing metadata should not reach the Lambda query');
  const legacySmartRoute = resolveDemoRoute(request({
    method: 'GET',
    url: '/api/sentence-demo/health',
    query: { '...slug': 'smart-sentence/health' }
  }));
  check(legacySmartRoute && legacySmartRoute.project === 'smart-sentence' && legacySmartRoute.operation === 'health', 'Vercel internal catch-all metadata should resolve the consolidated legacy route');
  assert.throws(
    () => validateQuery(request({ url: '/api/demos/covid-outbreak/states?refresh=true' }), resolveDemoRoute(request({ url: '/api/demos/covid-outbreak/states?refresh=true' })).rule),
    /Invalid query parameter/,
    'cache refresh query must not be public'
  );
  checks += 1;
  check(resolveDemoRoute(request({ url: '/api/demos/arbitrary/invoke' })) === null, 'arbitrary demo project should not resolve');
  check(resolveDemoRoute(request({ url: '/api/demos/shape/delete' })) === null, 'unknown demo operation should not resolve');
  const smartWarmupRoute = resolveDemoRoute(request({ method: 'POST', url: '/api/demos/smart-sentence/warmup' }));
  check(smartWarmupRoute && smartWarmupRoute.rule.path === '/rank', 'Smart Sentence warmup should map only to its fixed rank route');
  check(smartWarmupRoute.rule.timeoutMs === 55_000, 'Smart Sentence cold warmup should finish before Vercel maxDuration');

  const shapeRoute = resolveDemoRoute(request({ method: 'POST', url: '/api/demos/shape/predict' }));
  const event = buildLambdaEvent(request({
    method: 'POST',
    url: '/api/demos/shape/predict',
    headers: {
      authorization: 'Bearer secret',
      cookie: 'session=secret',
      'content-type': 'application/json',
      'x-request-id': 'test-id'
    }
  }), shapeRoute, '{"b64":"abc"}', new URLSearchParams());
  check(event.rawPath === '/predict' && event.requestContext.http.method === 'POST', 'Lambda event should preserve allowlisted operation');
  check(!event.headers.authorization && !event.headers.cookie, 'Lambda event should strip auth and cookies');
  check(event.headers.origin === 'https://danielshort.me', 'Lambda event should use a fixed trusted internal origin');
  check(event.body === '{"b64":"abc"}', 'Lambda event should preserve validated JSON body');

  const goodArn = 'arn:aws:lambda:us-east-2:123456789012:function:shape-demo:live';
  check(isLiveAliasArn(goodArn), 'live-qualified Lambda ARN should pass');
  check(!isLiveAliasArn('arn:aws:lambda:us-east-2:123456789012:function:shape-demo'), 'unqualified Lambda ARN should fail');
  check(resolveFunctionArn(shapeRoute, { DEMO_SHAPE_FUNCTION_ARN: goodArn }) === goodArn, 'route should select only its fixed function env');

  const decoded = decodeLambdaPayload({
    Payload: Buffer.from(JSON.stringify({
      statusCode: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ ok: true, prediction: 'circle' })
    }))
  });
  check(decoded.statusCode === 200 && JSON.parse(decoded.body.toString()).prediction === 'circle', 'Lambda envelope should decode safely');
  assert.throws(
    () => decodeLambdaPayload({ FunctionError: 'Unhandled', Payload: Buffer.from('{}') }),
    /invocation error/,
    'Lambda FunctionError must fail closed'
  );
  checks += 1;

  let invocation = null;
  const handler = createDemoProxyHandler({
    env: { DEMO_PROXY_MODE: 'lambda' },
    checkRateLimit: async () => ({ allowed: true }),
    invokeLambda: async (route, lambdaEvent) => {
      invocation = { route, lambdaEvent };
      return {
        statusCode: 200,
        contentType: 'application/json',
        body: Buffer.from(JSON.stringify({ ok: true, result: 7 }))
      };
    }
  });
  const invokeRes = response();
  await handler(request({
    method: 'POST',
    url: '/api/demos/digit-generator/generate',
    headers: { origin: 'https://www.danielshort.me', 'content-type': 'application/json' },
    body: JSON.stringify({ digit: 7 })
  }), invokeRes);
  check(invokeRes.statusCode === 200 && jsonBody(invokeRes).result === 7, 'allowlisted demo invocation should proxy JSON response');
  check(invocation.route.project === 'digit-generator', 'handler should invoke only resolved registry project');
  check(JSON.parse(invocation.lambdaEvent.body).digit === 7, 'handler should forward validated JSON payload');

  const unknownRes = response();
  await handler(request({ method: 'POST', url: '/api/demos/digit-generator/admin', body: '{}' }), unknownRes);
  check(unknownRes.statusCode === 404, 'unknown operation should return 404 before invocation');

  const originRes = response();
  await handler(request({
    method: 'POST',
    url: '/api/demos/digit-generator/generate',
    headers: { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site', 'content-type': 'application/json' },
    body: '{}'
  }), originRes);
  check(originRes.statusCode === 403, 'cross-site demo request should be rejected');

  const maskedHandler = createDemoProxyHandler({
    env: { DEMO_PROXY_MODE: 'lambda' },
    checkRateLimit: async () => ({ allowed: true }),
    invokeLambda: async () => ({
      statusCode: 500,
      contentType: 'application/json',
      body: Buffer.from(JSON.stringify({ error: 'internal stack and bucket name' }))
    })
  });
  const maskedRes = response();
  await maskedHandler(request({ method: 'GET', url: '/api/demos/retail-loss-sales/data' }), maskedRes);
  check(maskedRes.statusCode === 502, 'upstream 5xx should become a safe gateway error');
  check(!maskedRes.body.toString().includes('bucket name'), 'upstream internal errors should be masked');

  _memoryStore.clear();
  const rateReq = request({ url: '/api/demos/shape/predict' });
  const rateOptions = {
    namespace: 'demo#shape',
    windowSeconds: 60,
    windowLimit: 1,
    dailyLimit: 5,
    globalDailyLimit: 10,
    salt: 'test-salt',
    now: Date.UTC(2026, 0, 1)
  };
  check((await consumeRequestLimit(rateReq, rateOptions)).allowed, 'first request should pass memory rate limit');
  const denied = await consumeRequestLimit(rateReq, rateOptions);
  check(!denied.allowed && denied.statusCode === 429, 'second request should hit configured memory rate limit');
}

async function main() {
  await contactTests();
  await demoTests();
  process.stdout.write(`AWS contact/demo boundary tests passed (${checks} checks).\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
