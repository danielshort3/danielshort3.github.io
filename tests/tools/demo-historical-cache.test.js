'use strict';

const assert = require('assert');
const proxy = require('../../api/_lib/demo-proxy');

function snapshot(demo, value = 10) {
  if (demo === 'target-empty-package') {
    return {
      meta: { recordCount: 1, startDate: '2023-01-01', endDate: '2023-01-01' },
      rows: [{ datetime: '2023-01-01T12:00:00', value, employee: 'Employee_1', location: 'Area_1', condition: 'Condition_1', department: '1' }]
    };
  }
  return {
    meta: { generatedAt: '2023-01-01T00:00:00Z', salesStore: 'Store_1', incidentYear: 2023, shortageYear: 2023, currency: 'USD' },
    sales: {
      store: 'Store_1', rows: 1,
      weekly: [{ week: '2023-01-01', sales: value }], monthly: [{ month: '2023-01', sales: value }],
      departments: [{ department: '1', sales: value }], boycott: []
    },
    incidents: { rows: 1, year: 2023, monthly: [{ month: '2023-01', incidents: 1 }], stores: [{ store: 'Store_1', incidents: 1 }], regions: [], formats: [], states: [] },
    inventory: { rows: 1, year: 2023, years: [{ year: 2023, avgShortagePercent: .01 }], stores: [] },
    emptyPackages: { rows: 1, monthly: [{ month: '2023-01', count: 1, estimatedValue: value }], employees: [], areas: [], conditions: [] }
  };
}

function lambdaResponse(payload, statusCode = 200) {
  return { Payload: Buffer.from(JSON.stringify({ statusCode, body: JSON.stringify(payload) })) };
}

async function main() {
  const settings = {
    DEMO_PROXY_MODE: 'iam',
    DEMO_REQUIRE_DDB_RATE_LIMIT: 'false',
    DEMO_RATE_LIMIT_TABLE: '',
    DEMO_INVOKE_AWS_ROLE_ARN: '',
    AWS_AUTH_MODE: 'auto',
    VERCEL_ENV: 'development',
    DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:retail:live',
    DEMO_TARGET_EMPTY_PACKAGE_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:packages:live',
    DEMO_PIZZA_TIPS_FUNCTION_ARN: 'arn:aws:lambda:us-east-2:123456789012:function:pizza:live'
  };
  const previous = Object.fromEntries(Object.keys(settings).map(key => [key, process.env[key]]));
  const originalNow = Date.now;
  Object.assign(process.env, settings);
  let calls = 0;
  let status = 200;
  let payload;
  const deferredResponses = [];
  proxy._internal.memoryRateStore.clear();
  const installClients = () => proxy._internal.setClientFactoryForTests(() => ({
    lambda: {
      async send(command) {
        calls += 1;
        if (deferredResponses.length) return deferredResponses.shift()(command);
        const demo = command.input.FunctionName.includes(':packages:') ? 'target-empty-package' : 'retail-loss-sales';
        return lambdaResponse(payload === undefined ? snapshot(demo) : payload, status);
      }
    }
  }));
  installClients();
  async function request(demo, action = 'data', method = 'GET', extraHeaders = {}) {
    const res = {
      headers: {},
      setHeader(name, value) { this.headers[name] = value; },
      end(body) { this.body = body ? JSON.parse(body) : null; }
    };
    await proxy.handleDemoRequest({
      method, url: `/api/demos/${demo}/${action}`, body: {},
      headers: { host: 'localhost', 'content-type': 'application/json', ...extraHeaders }
    }, res, [demo, action]);
    return res;
  }
  try {
    const first = await request('retail-loss-sales');
    const second = await request('retail-loss-sales');
    assert.equal(calls, 1, 'Repeated historical reads should reuse the successful snapshot');
    assert.deepEqual(second.body, first.body);
    assert.match(second.headers['Cache-Control'], /^private, max-age=\d+$/);
    assert(Number(second.headers['X-RateLimit-Remaining']) < Number(first.headers['X-RateLimit-Remaining']),
      'Cache hits must still consume the visitor rate limit');

    const denied = await request('retail-loss-sales', 'data', 'GET', { origin: 'https://evil.example', 'sec-fetch-site': 'cross-site' });
    assert.equal(denied.statusCode, 403, 'A cached response must not bypass the origin guard');
    assert.equal(denied.headers['Cache-Control'], 'no-store');
    process.env.DEMO_PROXY_MODE = 'off';
    assert.equal((await request('retail-loss-sales')).statusCode, 503, 'Disabling the proxy must also disable cached data');
    process.env.DEMO_PROXY_MODE = 'iam';

    await request('target-empty-package');
    assert.equal(calls, 2, 'Different historical demos must not share cache entries');
    const beforeHealth = calls;
    assert.equal((await request('retail-loss-sales', 'health')).headers['Cache-Control'], 'no-store');
    await request('retail-loss-sales', 'health');
    assert.equal(calls, beforeHealth + 2, 'Health checks must reach the backend each time');
    await request('pizza-tips', 'predict', 'POST');
    assert.equal((await request('pizza-tips', 'predict', 'POST')).headers['Cache-Control'], 'no-store');

    Date.now = () => originalNow() + 301000;
    const beforeExpiry = calls;
    await request('retail-loss-sales');
    assert.equal(calls, beforeExpiry + 1, 'Expired historical data must be refreshed');
    process.env.DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN = 'arn:aws:lambda:us-east-2:123456789012:function:retail:replacement';
    await request('retail-loss-sales');
    assert.equal(calls, beforeExpiry + 2, 'A different upstream must not reuse the previous snapshot');

    Date.now = () => originalNow() + 602000;
    status = 503;
    const failed = await request('retail-loss-sales');
    const beforeRetry = calls;
    assert.equal(failed.headers['Cache-Control'], 'no-store');
    status = 200;
    payload = { ok: false, error: 'Data unavailable' };
    assert.equal((await request('retail-loss-sales')).headers['Cache-Control'], 'no-store');
    payload = snapshot('retail-loss-sales', 20);
    const recovered = await request('retail-loss-sales');
    assert.equal(calls, beforeRetry + 2, 'HTTP errors and error envelopes must both remain retryable');
    assert.equal(recovered.body.sales.weekly[0].sales, 20);

    for (const demo of ['retail-loss-sales', 'target-empty-package']) {
      const malformedSnapshots = [null, [], {}, { status: 'ok' }, { meta: {}, rows: {} }];
      if (demo === 'retail-loss-sales') {
        malformedSnapshots.push({ ...snapshot(demo), sales: {} });
      } else {
        malformedSnapshots.push({ ...snapshot(demo), meta: [] });
      }
      for (const malformed of malformedSnapshots) {
        installClients();
        payload = malformed;
        const beforeMalformed = calls;
        assert.equal((await request(demo)).headers['Cache-Control'], 'no-store', 'Malformed 200 snapshots cannot enter browser caches');
        payload = snapshot(demo);
        assert.match((await request(demo)).headers['Cache-Control'], /^private, max-age=\d+$/);
        assert.equal(calls, beforeMalformed + 2, 'A malformed 200 snapshot must not block a successful retry');
      }
    }

    for (const badResponse of [
      lambdaResponse({ error: 'Data unavailable' }, 503),
      lambdaResponse({ ok: false, error: 'Data unavailable' }),
      lambdaResponse({})
    ]) {
      installClients();
      let finishSuccess;
      let finishFailure;
      deferredResponses.push(
        () => new Promise((resolve) => { finishSuccess = resolve; }),
        () => new Promise((resolve) => { finishFailure = resolve; })
      );
      const successfulRequest = request('retail-loss-sales');
      const failedRequest = request('retail-loss-sales');
      await new Promise((resolve) => setImmediate(resolve));
      finishSuccess(lambdaResponse(snapshot('retail-loss-sales')));
      assert.match((await successfulRequest).headers['Cache-Control'], /^private, max-age=\d+$/);
      finishFailure(badResponse);
      const concurrentFailure = await failedRequest;
      assert.equal(concurrentFailure.headers['Cache-Control'], 'no-store', 'A concurrent failure cannot inherit another response\'s cache headers');
      const afterRace = calls;
      const stillCached = await request('retail-loss-sales');
      assert.equal(calls, afterRace, 'A concurrent failure must preserve the successful snapshot');
      assert.deepEqual(stillCached.body, snapshot('retail-loss-sales'));
    }

    installClients();
    payload = undefined;
    const aliasBase = 'arn:aws:lambda:us-east-2:123456789012:function:retail:';
    for (let alias = 0; alias < 5; alias += 1) {
      process.env.DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN = aliasBase + 'version' + alias;
      await request('retail-loss-sales');
    }
    const beforeEvictionCheck = calls;
    process.env.DEMO_RETAIL_LOSS_SALES_FUNCTION_ARN = aliasBase + 'version0';
    await request('retail-loss-sales');
    assert.equal(calls, beforeEvictionCheck + 1, 'The cache evicts the oldest snapshot once four entries are stored');
    console.log('Historical demo cache: reuse, expiry, isolation, guards, malformed snapshots, concurrent failures and bounds passed.');
  } finally {
    Date.now = originalNow;
    proxy._internal.setClientFactoryForTests(null);
    proxy._internal.memoryRateStore.clear();
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
