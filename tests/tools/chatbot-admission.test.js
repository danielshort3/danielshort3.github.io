'use strict';

const assert = require('node:assert/strict');
const limiter = require('../../api/_lib/chatbot-rate-limit');
const createAtomicStore = require('../helpers/atomic-ddb');
const settings = {
  CHATBOT_REQUIRE_DDB: 'false', CHATBOT_DDB_TABLE: '', CHATBOT_DDB_TABLE_NAME: '',
  CHATBOT_DAILY_LIMIT: '2', CHATBOT_GLOBAL_DAILY_LIMIT: '4', CHATBOT_WINDOW_LIMIT: '2',
  CHATBOT_HASH_SALT: 'offline-admission-salt', CHATBOT_DDB_AWS_ROLE_ARN: '', AWS_AUTH_MODE: 'auto', VERCEL_ENV: 'development'
};
const previous = Object.fromEntries(Object.keys(settings).map((name) => [name, process.env[name]]));
const realNow = Date.now;
let time = Date.UTC(2026, 8, 29, 12);
Date.now = () => time;
Object.assign(process.env, settings);
const request = (ip = '203.0.113.1') => ({ headers: { 'x-forwarded-for': ip } });

async function run() {
  try {
    assert.equal(limiter.getActorHash(request(), { conversationId: 'first-conversation' }), limiter.getActorHash(request(), { conversationId: 'different-conversation' }));
    assert.notEqual(limiter.getActorHash(request()), limiter.getActorHash(request('203.0.113.2')));
    for (const backend of ['memory', 'ddb']) {
      limiter._memoryStore.clear();
      const store = createAtomicStore();
      process.env.CHATBOT_DDB_TABLE = backend === 'ddb' ? 'offline-table' : '';
      limiter._setDocClientForTests(backend === 'ddb' ? store : null);
      let concurrent = await Promise.all(Array.from({ length: 12 }, (_, index) => limiter.checkChatbotRateLimit(request(), { conversationId: `rotation-${index}` })));
      assert.equal(concurrent.filter((result) => result.allowed).length, 1, `${backend}: simultaneous IDs share the cooldown`);
      time += 9000;
      assert.equal((await limiter.checkChatbotRateLimit(request(), { conversationId: 'next-id' })).allowed, true);
      time += 9000;
      for (let index = 0; index < 12; index++) {
        const rejected = await limiter.checkChatbotRateLimit(request(), { conversationId: `new-id-${index}` });
        assert.equal(rejected.allowed, false);
      }
      assert.equal((await limiter.checkChatbotRateLimit(request('203.0.113.2'))).allowed, true, `${backend}: rejected actor must not spend the global budget`);
      assert.equal((await limiter.checkChatbotRateLimit(request('203.0.113.3'))).allowed, true);
      assert.equal((await limiter.checkChatbotRateLimit(request('203.0.113.4'))).allowed, false, `${backend}: admitted work still obeys the global limit`);
      time += 86400000;
      concurrent = await Promise.all(Array.from({ length: 12 }, (_, index) => limiter.checkChatbotRateLimit(request(`198.51.100.${index}`), {}, { challengePassed: true })));
      assert.equal(concurrent.filter((result) => result.allowed).length, 4, `${backend}: concurrent admission cannot exceed global cap`);
      time += 86400000;
    }
    process.env.VERCEL_ENV = 'production';
    process.env.CHATBOT_REQUIRE_DDB = '';
    process.env.CHATBOT_DDB_TABLE = '';
    await assert.rejects(limiter.checkChatbotRateLimit(request()), { code: 'CHATBOT_RATE_LIMIT_STORE_MISSING' });
    console.log('Chatbot admission passed: identity rotation, cooldown races, actor rejection, atomic global budget and production fail-closed. No network calls.');
  } finally {
    Date.now = realNow;
    for (const [name, value] of Object.entries(previous)) {
      if (typeof value === 'undefined') delete process.env[name]; else process.env[name] = value;
    }
    limiter._setDocClientForTests(null);
    limiter._memoryStore.clear();
  }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
