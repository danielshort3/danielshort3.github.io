'use strict';

const assert = require('assert');
const fs = require('fs');
const http = require('http');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { once } = require('events');
const { handleLocalChatbotRequest } = require('../../build/lib/local-chatbot-proxy');
const { createLocalServer } = require('../../build/dev');

const nativeFetch = global.fetch;
const UPSTREAM = 'https://k8bys9gicf.execute-api.us-east-2.amazonaws.com/prod';

async function withServer(server, run) {
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  try {
    await run(`http://127.0.0.1:${server.address().port}`);
  } finally {
    server.closeAllConnections();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
}

function jsonResponse(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json' }
  });
}

async function testProxyRequests() {
  const calls = [];
  let respond = () => jsonResponse({ status: 'READY' });
  const fetchImpl = async (url, options) => {
    calls.push({ url: String(url), options });
    return respond(url, options);
  };
  const server = http.createServer((req, res) => {
    handleLocalChatbotRequest(req, res, { fetchImpl }).catch((error) => {
      res.statusCode = 500;
      res.end(error.message);
    });
  });

  await withServer(server, async (origin) => {
    const request = async (suffix, options = {}) => {
      const response = await nativeFetch(`${origin}/api/chatbot-demo/${suffix}`, options);
      const text = await response.text();
      return { response, text, body: JSON.parse(text) };
    };
    const post = (body) => ({
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });

    for (const [suffix, upstreamPath, options] of [
      ['bedrock/status', '/bedrock/status', {}],
      ['qwen/status', '/status', {}],
      ['qwen/warmup', '/warmup', post({})],
      ['qwen/submit', '/submit', post({ prompt: 'Plan a day in Grand Junction.' })],
      ['qwen/result?jobId=job-123', '/result?jobId=job-123', {}],
      ['qwen/result?outputUri=s3%3A%2F%2Ftest-bucket%2Fresults%2Fanswer.json', '/result?outputUri=s3%3A%2F%2Ftest-bucket%2Fresults%2Fanswer.json', {}]
    ]) {
      const result = await request(suffix, options);
      assert.strictEqual(result.response.status, 200, `${suffix} should be available locally`);
      assert.deepStrictEqual(result.body, { status: 'READY' });
      assert(result.response.headers.get('cache-control').includes('no-store'));
      const call = calls.at(-1);
      assert.strictEqual(call.url, UPSTREAM + upstreamPath, 'The bridge must select the fixed AWS route');
      assert.strictEqual(call.options.method, options.method || 'GET');
      assert.strictEqual(call.options.redirect, 'error', 'Upstream redirects must not escape the fixed destination');
      if (options.body) assert.deepStrictEqual(JSON.parse(call.options.body), JSON.parse(options.body));
    }

    await request('qwen/submit', {
      ...post({ prompt: 'Another question' }),
      headers: {
        'content-type': 'application/json',
        origin,
        authorization: 'Bearer browser-only-secret',
        cookie: 'session=browser-only-secret',
        'x-private-header': 'browser-only-secret'
      }
    });
    const forwardedHeaders = Object.fromEntries(new Headers(calls.at(-1).options.headers).entries());
    assert.strictEqual(forwardedHeaders['content-type'], 'application/json');
    for (const header of ['origin', 'authorization', 'cookie', 'x-private-header']) {
      assert(!Object.hasOwn(forwardedHeaders, header), `${header} must stay out of upstream requests`);
    }

    const invalidRequests = [
      ['bedrock/submit', {}, 404],
      ['unknown/status', {}, 404],
      ['qwen/status/extra', {}, 404],
      ['qwen/status', post({}), 405],
      ['qwen/submit', {}, 405],
      ['qwen/status?endpoint=https%3A%2F%2Fexample.com', {}, 400],
      ['qwen/result', {}, 400],
      ['qwen/result?jobId=', {}, 400],
      ['qwen/result?jobId=a&jobId=b', {}, 400],
      ['qwen/result?jobId=a&outputUri=s3%3A%2F%2Fb%2Fkey', {}, 400],
      ['qwen/result?jobId=a&unexpected=value', {}, 400],
      [`qwen/result?jobId=${'a'.repeat(129)}`, {}, 400],
      [`qwen/result?outputUri=${'a'.repeat(2049)}`, {}, 400],
      ['qwen/status', { headers: { origin: 'https://other.example' } }, 403],
      ['qwen/status', { headers: { origin: origin.replace('127.0.0.1', 'localhost') } }, 403],
      ['qwen/status', { headers: { 'sec-fetch-site': 'cross-site' } }, 403],
      ['qwen/submit', { method: 'POST', headers: { 'content-type': 'text/plain' }, body: '{}' }, 415],
      ['qwen/submit', { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{' }, 400],
      ['qwen/submit', post([]), 400],
      ['qwen/submit', post(null), 400],
      ['qwen/submit', post({ prompt: 'a'.repeat(8192) }), 413]
    ];
    for (const [suffix, options, status] of invalidRequests) {
      const before = calls.length;
      const result = await request(suffix, options);
      assert.strictEqual(result.response.status, status, `${suffix} should reject invalid requests`);
      assert.strictEqual(calls.length, before, 'Rejected requests must not reach AWS');
      if (status === 405) assert(result.response.headers.get('allow'), 'Method rejection must advertise allowed methods');
    }

    respond = () => jsonResponse({ error: 'Please wait.' }, 429);
    const limited = await request('bedrock/status');
    assert.strictEqual(limited.response.status, 429, 'Real upstream status must survive the bridge');
    assert.deepStrictEqual(limited.body, { error: 'Please wait.' });

    for (const failure of [
      () => { throw new Error('private-upstream-secret network details'); },
      () => new Response('<html>private-upstream-secret</html>', { status: 502 }),
      () => jsonResponse({ data: 'a'.repeat(1024 * 1024) })
    ]) {
      respond = failure;
      const failed = await request('bedrock/status');
      assert.strictEqual(failed.response.status, 502, 'Bad upstream responses should fail with a gateway error');
      assert(!failed.text.includes('private-upstream-secret'), 'Gateway errors must not expose upstream diagnostics');
      assert(failed.text.length < 2048, 'Oversized upstream data must not reach the browser');
    }
  });
}

function testLocalEndpointSelection() {
  const html = fs.readFileSync(path.join(__dirname, '../../demos/chatbot-demo.html'), 'utf8');
  const start = html.indexOf('function resolveBackendApiUrl(');
  assert(start >= 0, 'The chatbot must select a local control endpoint');
  const end = html.indexOf('\n    }', start);
  assert(end > start, 'The endpoint selection helper must have a complete body');
  const source = html.slice(start, end + '\n    }'.length);
  for (const hostname of ['localhost', '127.0.0.1', '[::1]', 'www.danielshort.me']) {
    const resolutionCalls = [];
    const env = {
      window: { location: { hostname, origin: `http://${hostname}:4181`, search: '?endpoint=https://stale.example' } },
      resolveEndpoint: (options) => {
        resolutionCalls.push(options);
        return 'https://stale.example/';
      }
    };
    vm.runInNewContext(source, env);
    for (const backend of ['qwen', 'bedrock']) {
      const resolved = env.resolveBackendApiUrl(backend, 'https://runtime-override.example', `stored.${backend}`);
      if (hostname === 'www.danielshort.me') {
        assert.strictEqual(resolved.replace(/\/$/, ''), 'https://stale.example', 'Production endpoint selection must retain its existing resolver');
        assert.strictEqual(resolutionCalls.at(-1).defaultUrl, 'https://runtime-override.example');
        assert.strictEqual(resolutionCalls.at(-1).storageKey, `stored.${backend}`);
      } else {
        assert.strictEqual(resolved.replace(/\/$/, ''), `/api/chatbot-demo/${backend}`, 'Localhost must ignore stale query, storage and runtime endpoint overrides');
        assert.strictEqual(resolutionCalls.length, 0, 'Local endpoint selection must bypass external overrides');
      }
    }
  }
}

async function testLocalServerIntegration() {
  const tempRoot = path.resolve(os.tmpdir());
  const envDir = fs.mkdtempSync(path.join(tempRoot, 'website-chatbot-proxy-test-'));
  const calls = [];
  const previousFetch = global.fetch;
  global.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    return jsonResponse({ status: 'READY', backend: 'bedrock' });
  };
  try {
    await withServer(createLocalServer({ envDir }), async (origin) => {
      const response = await nativeFetch(`${origin}/api/chatbot-demo/bedrock/status`);
      assert.strictEqual(response.status, 200, 'The normal development server must route the local chatbot bridge');
      assert.deepStrictEqual(await response.json(), { status: 'READY', backend: 'bedrock' });
      assert.strictEqual(calls.length, 1);
      assert.strictEqual(calls[0].url, UPSTREAM + '/bedrock/status');
      const rejected = await nativeFetch(`${origin}/api/chatbot-demo/bedrock/status?endpoint=https://example.com`);
      assert.strictEqual(rejected.status, 400, 'The normal server must preserve bridge route validation');
      await rejected.text();
      assert.strictEqual(calls.length, 1);
    });
  } finally {
    global.fetch = previousFetch;
    const resolved = path.resolve(envDir);
    assert.strictEqual(path.dirname(resolved), tempRoot, 'Cleanup must remain within the test temporary directory');
    assert(path.basename(resolved).startsWith('website-chatbot-proxy-test-'));
    fs.rmSync(resolved, { recursive: true, force: true });
  }
}

(async () => {
  await testProxyRequests();
  testLocalEndpointSelection();
  await testLocalServerIntegration();
  console.log('Local chatbot proxy: fixed routes, origin checks, bounded JSON, upstream errors, local endpoint selection and development server routing passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
