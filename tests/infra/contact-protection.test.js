'use strict';

const assert = require('node:assert/strict');
const { createSignedHeaders, verifySignedRequest, validatePayload } = require('../../api/_lib/contact-protection');
const { createHandler } = require('../../aws/contact-function');
const createAtomicStore = require('../helpers/atomic-ddb');
const secret = 'offline-contact-protection-secret-at-least-32-bytes';
const time = Date.UTC(2026, 8, 29, 12);
const origin = 'https://www.danielshort.me';
const env = { CONTACT_PROXY_SECRET: secret, CONTACT_RATE_LIMIT_TABLE: 'offline-contact', SENDER_EMAIL: 'sender@example.com', RECIPIENT_EMAIL: 'owner@example.com' };
const payload = { name: 'Offline test', email: 'offline@example.com', message: 'No real mail is sent.' };

function event(ip = '203.0.113.4', body = JSON.stringify(payload)) {
  return { body, headers: createSignedHeaders({ body, clientIp: ip, origin, now: time }, secret), requestContext: { http: { method: 'POST' } } };
}

async function run() {
  assert.equal(validatePayload({ ...payload, name: null }), null);
  assert.equal(validatePayload({ ...payload, email: [] }), null);
  assert.equal(validatePayload({ ...payload, message: 1 }), null);
  assert.equal(validatePayload({ ...payload, company: {} }), null);
  const signed = event();
  assert(verifySignedRequest(signed, secret, time));
  assert.equal(verifySignedRequest({ ...signed, body: JSON.stringify({ ...payload, message: 'Changed body' }) }, secret, time), null);
  assert.equal(verifySignedRequest(signed, secret, time + 301000), null);
  assert.equal(verifySignedRequest({ ...signed, headers: { ...signed.headers, 'X-Contact-Actor': 'f'.repeat(64) } }, secret, time), null);
  const store = createAtomicStore();
  let sent = 0;
  const handler = createHandler({ env, now: () => time, ddb: store, ses: { async send() { sent++; return {}; } } });
  const direct = { ...signed, headers: { origin }, body: JSON.stringify(payload) };
  assert.equal((await handler(direct)).statusCode, 403, 'Direct AWS callers cannot bypass the proxy');
  assert.equal((await handler({ ...signed, headers: { ...signed.headers, Origin: 'https://foreign.example' } })).statusCode, 403);
  assert.equal((await handler(event('203.0.113.4', JSON.stringify({ ...payload, name: null })))).statusCode, 400);
  assert.equal(sent, 0, 'Invalid requests never call SES');
  const replies = await Promise.all(Array.from({ length: 10 }, () => handler(event())));
  assert.equal(replies.filter((reply) => reply.statusCode === 200).length, 2, 'Concurrent attempts share a durable per-client quota');
  assert.equal(sent, 2);
  const replay = event('203.0.113.5');
  assert.equal((await handler(replay)).statusCode, 200);
  assert.equal((await handler(replay)).statusCode, 429, 'A captured signed request sends at most once');
  const unavailable = createHandler({ env, now: () => time, ddb: { async send() { throw new Error('offline failure'); } }, ses: { async send() { throw new Error('must not send'); } } });
  assert.equal((await unavailable(event('203.0.113.6'))).statusCode, 503, 'Protection failure blocks SES');
  const globalStore = createAtomicStore();
  let globalSent = 0;
  const global = createHandler({ env: { ...env, CONTACT_GLOBAL_DAILY_LIMIT: '3' }, now: () => time, ddb: globalStore, ses: { async send() { globalSent++; } } });
  const globalReplies = await Promise.all(Array.from({ length: 10 }, (_, index) => global(event(`198.51.100.${index}`))));
  assert.equal(globalReplies.filter((reply) => reply.statusCode === 200).length, 3);
  assert.equal(globalSent, 3, 'Atomic global quota caps actual sends');
  const rawRecords = JSON.stringify([...globalStore.items.values()]);
  assert(!rawRecords.includes(payload.email) && !rawRecords.includes(payload.message) && !rawRecords.includes('198.51.100.'), 'Quota records retain no message, email or raw address');
  console.log('Contact protection passed: trusted proxy, body/IP-bound signatures, malformed fields, replay, concurrent quotas and fail-closed sending. SES and DynamoDB mocked.');
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
