'use strict';

const assert = require('assert');
const { EventEmitter } = require('events');
const {
  handleChatbotStream,
  _internal
} = require('../api/_lib/chatbot-stream-proxy');

const TEST_FUNCTION_ARN = 'arn:aws:lambda:us-east-2:123456789012:function:VGJBedrockStream:live';
const TEST_ROLE_ARN = 'arn:aws:iam::123456789012:role/website-demo-invoke-production';

class MockResponse extends EventEmitter {
  constructor() {
    super();
    this.statusCode = 200;
    this.headers = new Map();
    this.chunks = [];
    this.headersSent = false;
    this.writableEnded = false;
    this.destroyed = false;
  }

  setHeader(name, value) {
    this.headers.set(String(name).toLowerCase(), String(value));
  }

  getHeader(name) {
    return this.headers.get(String(name).toLowerCase());
  }

  flushHeaders() {
    this.headersSent = true;
  }

  write(value) {
    this.headersSent = true;
    this.chunks.push(Buffer.from(String(value)));
    return true;
  }

  end(value) {
    if (typeof value !== 'undefined') this.write(value);
    this.headersSent = true;
    this.writableEnded = true;
  }

  bodyText() {
    return Buffer.concat(this.chunks).toString('utf8');
  }
}

function request(overrides = {}) {
  return {
    method: 'POST',
    headers: {
      origin: 'https://www.danielshort.me',
      host: 'www.danielshort.me',
      'x-forwarded-proto': 'https',
      'x-forwarded-for': '203.0.113.9',
      'sec-fetch-site': 'same-origin',
      'content-type': 'application/json',
      'x-chatbot-session': 'attacker-controlled-session',
      ...(overrides.headers || {})
    },
    body: overrides.body || { prompt: 'Plan a Grand Junction weekend.' },
    socket: { remoteAddress: '203.0.113.9' },
    ...overrides,
    headers: {
      origin: 'https://www.danielshort.me',
      host: 'www.danielshort.me',
      'x-forwarded-proto': 'https',
      'x-forwarded-for': '203.0.113.9',
      'sec-fetch-site': 'same-origin',
      'content-type': 'application/json',
      'x-chatbot-session': 'attacker-controlled-session',
      ...(overrides.headers || {})
    }
  };
}

async function* streamEvents(lines) {
  const prelude = Buffer.from(JSON.stringify({
    statusCode: 200,
    headers: {
      'Cache-Control': 'no-store',
      'Content-Type': 'application/x-ndjson; charset=utf-8'
    }
  }), 'utf8');
  const delimiter = Buffer.alloc(8);
  const payload = Buffer.from(lines.join('\n') + '\n', 'utf8');
  const split = Math.max(1, Math.floor(payload.length / 2));
  yield {
    PayloadChunk: {
      Payload: Buffer.concat([prelude, delimiter.subarray(0, 3)])
    }
  };
  yield {
    PayloadChunk: {
      Payload: Buffer.concat([delimiter.subarray(3), payload.subarray(0, split)])
    }
  };
  yield { PayloadChunk: { Payload: payload.subarray(split) } };
  yield { InvokeComplete: {} };
}

async function run() {
  const previousEnv = {};
  const envPatch = {
    VERCEL_ENV: 'production',
    AWS_AUTH_MODE: 'oidc',
    CHATBOT_STREAM_FUNCTION_ARN: TEST_FUNCTION_ARN,
    CHATBOT_STREAM_AWS_ROLE_ARN: TEST_ROLE_ARN,
    CHATBOT_AWS_REGION: 'us-east-2',
    CHATBOT_STREAM_TIMEOUT_MS: '35000'
  };
  for (const [key, value] of Object.entries(envPatch)) {
    previousEnv[key] = process.env[key];
    process.env[key] = value;
  }

  let invocationInput = null;
  let invocationCount = 0;
  let rateRequest = null;
  _internal.setLambdaClientFactoryForTests(() => ({
    async send(command) {
      invocationCount += 1;
      invocationInput = command.input;
      return {
        StatusCode: 200,
        ExecutedVersion: '7',
        EventStream: streamEvents([
          JSON.stringify({ type: 'meta', backend: 'bedrock', context_count: 2 }),
          JSON.stringify({ type: 'token', text: 'Hello ' }),
          JSON.stringify({ type: 'token', text: 'Grand Junction.' }),
          JSON.stringify({
            type: 'done',
            data: { answer: 'Hello Grand Junction.' }
          })
        ])
      };
    }
  }));
  _internal.setRateLimiterForTests(async (req) => {
    rateRequest = req;
    return {
      allowed: true,
      config: { windowLimit: 8 },
      counts: { window: 1 }
    };
  });

  try {
    const parsed = _internal.parseFunctionArn(TEST_FUNCTION_ARN, { VERCEL_ENV: 'production' });
    assert.strictEqual(parsed.alias, 'live');
    assert.throws(
      () => _internal.parseFunctionArn(TEST_FUNCTION_ARN.replace(':live', ''), { VERCEL_ENV: 'production' }),
      err => err.code === 'CHATBOT_STREAM_FUNCTION_ARN_INVALID'
    );
    assert.throws(
      () => _internal.parseFunctionArn(TEST_FUNCTION_ARN.replace(':live', ':preview'), { VERCEL_ENV: 'production' }),
      err => err.code === 'CHATBOT_STREAM_ALIAS_INVALID'
    );

    assert(_internal.isAllowedOrigin(request(), process.env));
    assert(!_internal.isAllowedOrigin(request({
      headers: {
        origin: 'https://evil.example',
        'sec-fetch-site': 'cross-site'
      }
    }), process.env));

    const normalized = _internal.normalizePayload({
      prompt: '  Plan a day downtown.  ',
      followup_context: {
        source: 'recommended_followup',
        prompt: 'Add an evening stop',
        previous_question: 'Plan a day downtown',
        previous_answer: 'Start near Main Street.',
        previous_intent: 'itinerary',
        previous_category: 'planning',
        previous_route: 'bedrock',
        source_labels: ['Downtown Grand Junction'],
        source_urls: ['https://www.visitgrandjunction.com/things-to-do/downtown/'],
        secret: 'not-forwarded'
      },
      top_k: 1000
    });
    assert.strictEqual(normalized.prompt, 'Plan a day downtown.');
    assert.strictEqual(normalized.followup_context.previous_answer, 'Start near Main Street.');
    assert.strictEqual(normalized.followup_context.source_urls.length, 1);
    assert(!Object.prototype.hasOwnProperty.call(normalized.followup_context, 'secret'));
    assert.throws(
      () => _internal.normalizePayload({ prompt: 'x'.repeat(_internal.MAX_PROMPT_CHARS + 1) }),
      err => err.code === 'PROMPT_TOO_LONG' && err.statusCode === 413
    );

    const res = new MockResponse();
    await handleChatbotStream(request({
      body: {
        prompt: 'Plan a Grand Junction weekend.',
        followup_context: {
          source: 'recommended_followup',
          prompt: 'Make it family-friendly',
          previous_question: 'Plan a Grand Junction weekend.',
          previous_answer: 'Use Grand Junction as the base.',
          previous_intent: 'itinerary',
          previous_category: 'planning',
          previous_route: 'bedrock',
          source_labels: ['Visit Grand Junction'],
          source_urls: ['https://www.visitgrandjunction.com/']
        }
      }
    }), res);
    assert.strictEqual(res.statusCode, 200);
    assert.strictEqual(res.getHeader('content-type'), 'application/x-ndjson; charset=utf-8');
    assert.strictEqual(res.getHeader('cache-control'), 'no-store, no-transform');
    assert.strictEqual(res.getHeader('x-chatbot-lambda-version'), '7');
    assert.strictEqual(res.getHeader('x-ratelimit-limit'), '8');
    assert.strictEqual(res.getHeader('x-ratelimit-remaining'), '7');
    assert.strictEqual(invocationCount, 1);
    assert.strictEqual(invocationInput.FunctionName, TEST_FUNCTION_ARN);
    const lambdaEvent = JSON.parse(Buffer.from(invocationInput.Payload).toString('utf8'));
    assert.strictEqual(lambdaEvent.version, '2.0');
    assert.strictEqual(lambdaEvent.requestContext.http.method, 'POST');
    const lambdaBody = JSON.parse(lambdaEvent.body);
    assert.strictEqual(lambdaBody.prompt, 'Plan a Grand Junction weekend.');
    assert.strictEqual(lambdaBody.followup_context.prompt, 'Make it family-friendly');
    assert.strictEqual(lambdaBody.followup_context.previous_answer, 'Use Grand Junction as the base.');
    assert(!Object.prototype.hasOwnProperty.call(rateRequest.headers, 'x-chatbot-session'));
    const streamed = res.bodyText().trim().split('\n').map(line => JSON.parse(line));
    assert.deepStrictEqual(streamed.map(event => event.type), ['meta', 'token', 'token', 'done']);
    assert.strictEqual(streamed[3].data.answer, 'Hello Grand Junction.');

    const crossOriginRes = new MockResponse();
    await handleChatbotStream(request({
      headers: {
        origin: 'https://evil.example',
        'sec-fetch-site': 'cross-site'
      }
    }), crossOriginRes);
    assert.strictEqual(crossOriginRes.statusCode, 403);
    assert.strictEqual(invocationCount, 1);

    _internal.setRateLimiterForTests(async () => ({
      allowed: false,
      statusCode: 429,
      payload: {
        error: 'Please wait before sending another question.',
        retryAfter: 8,
        limits: { windowLimit: 8 }
      }
    }));
    const limitedRes = new MockResponse();
    await handleChatbotStream(request(), limitedRes);
    assert.strictEqual(limitedRes.statusCode, 429);
    assert.strictEqual(limitedRes.getHeader('retry-after'), '8');
    assert.strictEqual(invocationCount, 1);

    const errorRes = new MockResponse();
    const controller = new AbortController();
    await _internal.forwardLambdaEventStream(
      errorRes,
      streamEvents([JSON.stringify({ type: 'error', error: 'sensitive upstream details' })]),
      { maxStreamBytes: _internal.MAX_STREAM_BYTES },
      controller
    );
    assert(!errorRes.bodyText().includes('sensitive upstream details'));
    assert(errorRes.bodyText().includes('temporarily unavailable'));

    let oversizedError = null;
    try {
      await _internal.forwardLambdaEventStream(
        new MockResponse(),
        (async function* oversized() {
          yield { PayloadChunk: { Payload: Buffer.alloc(_internal.MAX_STREAM_BYTES + 1, 120) } };
        })(),
        { maxStreamBytes: _internal.MAX_STREAM_BYTES },
        new AbortController()
      );
    } catch (err) {
      oversizedError = err;
    }
    assert.strictEqual(oversizedError?.code, 'CHATBOT_STREAM_TOO_LARGE');

    console.log('chatbot-stream-proxy: 28 checks passed');
  } finally {
    _internal.setLambdaClientFactoryForTests(null);
    _internal.setRateLimiterForTests(null);
    for (const [key, value] of Object.entries(previousEnv)) {
      if (typeof value === 'undefined') delete process.env[key];
      else process.env[key] = value;
    }
  }
}

run().catch(err => {
  console.error(err);
  process.exitCode = 1;
});
