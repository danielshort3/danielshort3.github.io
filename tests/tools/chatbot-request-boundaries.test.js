'use strict';

const assert = require('node:assert/strict');
const { Readable } = require('node:stream');
const { test } = require('node:test');
const handler = require('../../api/chatbot');
const { readJson } = require('../../api/_lib/json-body');
const { readJson: toolsReadJson } = require('../../api/_lib/tools-api');
const { readJson: chatbotReadJson, normalizeHistory } = handler._private;
const limit = 24_000;

function request(body, kind, headers = {}){
  const raw = typeof body === 'string' ? body : JSON.stringify(body);
  const req = kind === 'stream' ? Readable.from([Buffer.from(raw)]) : {};
  req.headers = { origin: 'https://www.danielshort.me', ...headers };
  req.method = 'POST';
  req.url = '/api/chatbot';
  if (kind === 'object') req.body = JSON.parse(raw);
  if (kind === 'string') req.body = raw;
  if (kind === 'buffer') req.body = Buffer.from(raw);
  return req;
}

function bodyOfSize(bytes){
  const empty = JSON.stringify({ message: '', text: '' });
  return { message: '', text: 'x'.repeat(bytes - Buffer.byteLength(empty)) };
}

for (const kind of ['object', 'string', 'buffer', 'stream']) {
  test(`${kind}: accepts below/at limit and rejects above limit`, async () => {
    for (const bytes of [limit - 1, limit]) {
      assert.deepEqual(await chatbotReadJson(request(bodyOfSize(bytes), kind)), bodyOfSize(bytes));
    }
    await assert.rejects(chatbotReadJson(request(bodyOfSize(limit + 1), kind)), { code: 'BODY_TOO_LARGE' });
    await assert.rejects(chatbotReadJson(request({ message: '', text: '😀'.repeat(7000) }, kind)), { code: 'BODY_TOO_LARGE' });
  });

  test(`${kind}: HTTP rejects oversize before inference, independent of declared length`, async () => {
    const req = request(bodyOfSize(100_032), kind, { 'content-length': '12' });
    const res = { headers: {}, setHeader(key, value){ this.headers[key] = value; }, end(value){ this.body = value; } };
    await handler(req, res);
    assert.equal(res.statusCode, 413);
    assert.equal(res.headers['Cache-Control'], 'no-store');
  });
}

test('declared oversized bodies are rejected without consuming the stream', async () => {
  await assert.rejects(chatbotReadJson({ headers: { 'content-length': '24001' } }), { code: 'BODY_TOO_LARGE' });
});

test('multibyte codepoints split between chunks are not corrupted', async () => {
  const expected = { message: 'a😀éz' };
  const raw = Buffer.from(JSON.stringify(expected));
  const req = Readable.from(Array.from(raw, byte => Buffer.from([byte])));
  req.headers = {};
  assert.deepEqual(await chatbotReadJson(req), expected);
});

test('invalid JSON and non-object input are rejected', async () => {
  for (const value of ['{bad', 'null', '[]', '42', '"message"']) {
    await assert.rejects(chatbotReadJson(request(value, 'string')));
  }
});

test('history has an input cap and discards older turns before normalization', async () => {
  await assert.rejects(chatbotReadJson(request({ history: Array(33).fill({}) }, 'object')), /at most 32/);
  await assert.rejects(chatbotReadJson(request({ history: 'not-an-array' }, 'object')), /at most 32/);
  const older = { get text(){ throw Error('discarded history must not be read'); } };
  const latest = Array.from({ length: 8 }, (_, i) => ({ role: 'user', text: `${i}` }));
  assert.deepEqual(normalizeHistory([older, ...latest]), latest);
});

test('tools and chatbot use the same reader with different limits', async () => {
  assert.equal(toolsReadJson, readJson);
  const body = bodyOfSize(25_000);
  assert.deepEqual(await toolsReadJson(request(body, 'object')), body);
  await assert.rejects(chatbotReadJson(request(body, 'object')), { code: 'BODY_TOO_LARGE' });
});

test('aborted streams reject rather than remain pending', async () => {
  const req = new Readable({ read(){} });
  req.headers = {};
  const reading = readJson(req);
  req.emit('aborted');
  await assert.rejects(reading, /aborted/);
  req.destroy();
});
