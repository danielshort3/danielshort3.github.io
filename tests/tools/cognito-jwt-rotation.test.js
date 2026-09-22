'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../../api/_lib/cognito-jwt.js'), 'utf8');
const issuer = 'https://cognito-idp.us-east-2.amazonaws.com/test-pool';
const clientId = 'test-client';
const pairs = ['old', 'new'].map(kid => {
  const pair = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
  return { ...pair, kid, jwk: { ...pair.publicKey.export({ format: 'jwk' }), kid, use: 'sig', alg: 'RS256' } };
});

function harness(){
  const state = { now: 1_800_000_000_000, calls: 0, keys: [pairs[0].jwk], fetch: null, signals: [] };
  const sandbox = {
    module: { exports: {} }, require, Buffer, AbortController,
    process: { env: { TOOLS_COGNITO_ISSUER: issuer, TOOLS_COGNITO_CLIENT_ID: clientId } },
    Date: class extends Date { static now(){ return state.now; } },
    setTimeout: fn => setTimeout(fn, 30), clearTimeout,
    fetch: async (url, options) => {
      assert.equal(url, `${issuer}/.well-known/jwks.json`);
      state.calls++;
      state.signals.push(options.signal);
      return state.fetch ? state.fetch() : { ok: true, json: async () => ({ keys: state.keys }) };
    }
  };
  vm.runInNewContext(source, sandbox, { filename: 'cognito-jwt.js' });
  state.verify = sandbox.module.exports.verifyCognitoIdToken;
  state.token = (key = pairs[0], claims = {}, header = {}) => {
    const encode = value => Buffer.from(JSON.stringify(value)).toString('base64url');
    const input = `${encode({ alg: 'RS256', kid: key.kid, ...header })}.${encode({
      iss: issuer, aud: clientId, token_use: 'id', sub: 'test-user',
      exp: Math.floor(state.now / 1000) + 3600, ...claims
    })}`;
    return `${input}.${crypto.sign('RSA-SHA256', Buffer.from(input), key.privateKey).toString('base64url')}`;
  };
  return state;
}

const code = expected => error => error.code === expected;

test('a newly published key is accepted during a fresh cache', async () => {
  const h = harness();
  await h.verify(h.token());
  await h.verify(h.token());
  assert.equal(h.calls, 1);
  h.keys = pairs.map(pair => pair.jwk);
  assert.equal((await h.verify(h.token(pairs[1]))).sub, 'test-user');
  assert.equal(h.calls, 2);
});

test('cold and rotation refreshes are shared by concurrent verifiers', async () => {
  const h = harness();
  await Promise.all(Array.from({ length: 20 }, () => h.verify(h.token())));
  assert.equal(h.calls, 1);
  h.keys = pairs.map(pair => pair.jwk);
  await Promise.all(Array.from({ length: 20 }, () => h.verify(h.token(pairs[1]))));
  assert.equal(h.calls, 2);
});

test('random unknown kids share an issuer cooldown and cannot trigger a refresh storm', async () => {
  const h = harness();
  await h.verify(h.token());
  await Promise.all(Array.from({ length: 20 }, (_, i) =>
    assert.rejects(h.verify(h.token(pairs[1], {}, { kid: `unknown-${i}` })), code('JWT_KID_UNKNOWN'))));
  assert.equal(h.calls, 2);
  await assert.rejects(h.verify(h.token(pairs[1])), code('JWT_KID_UNKNOWN'));
  assert.equal(h.calls, 2);
  await h.verify(h.token());
  h.now += 30_001;
  h.keys = pairs.map(pair => pair.jwk);
  await h.verify(h.token(pairs[1]));
  assert.equal(h.calls, 3);
});

test('normal cache expiry still refreshes keys', async () => {
  const h = harness();
  await h.verify(h.token());
  h.now += 600_001;
  await h.verify(h.token());
  assert.equal(h.calls, 2);
});

for (const stage of ['headers', 'body']) {
  test(`the deadline bounds stalled ${stage} and aborts the request`, async () => {
    const h = harness();
    h.fetch = stage === 'headers'
      ? () => new Promise(() => {})
      : () => ({ ok: true, json: () => new Promise(() => {}) });
    await assert.rejects(h.verify(h.token()), code('JWKS_FETCH_TIMEOUT'));
    assert.equal(h.signals[0].aborted, true);
    h.fetch = null;
    await h.verify(h.token());
    assert.equal(h.calls, 2, 'failed in-flight requests must be cleared');
  });
}

test('failed forced refresh is bounded and does not poison known cached keys', async () => {
  const h = harness();
  await h.verify(h.token());
  h.fetch = () => ({ ok: false, status: 503 });
  await assert.rejects(h.verify(h.token(pairs[1])), code('JWKS_FETCH_FAILED'));
  await h.verify(h.token());
  await assert.rejects(h.verify(h.token(pairs[1])), code('JWT_KID_UNKNOWN'));
  assert.equal(h.calls, 2);
  h.now += 30_001;
  h.fetch = null;
  h.keys = pairs.map(pair => pair.jwk);
  await h.verify(h.token(pairs[1]));
});

test('malformed key sets are rejected rather than cached', async () => {
  const h = harness();
  h.keys = [{ kid: 'old', kty: 'EC' }];
  await assert.rejects(h.verify(h.token()), code('JWKS_INVALID'));
  h.keys = [pairs[0].jwk];
  await h.verify(h.token());
  assert.equal(h.calls, 2);
});

test('rotation keeps issuer, audience, expiry, algorithm, use and signature checks', async () => {
  const h = harness();
  for (const [claims, header, expected] of [
    [{ iss: 'https://untrusted.invalid' }, {}, 'JWT_ISS'],
    [{ aud: 'wrong-client' }, {}, 'JWT_AUD'],
    [{ exp: 1 }, {}, 'JWT_EXP'],
    [{ token_use: 'access' }, {}, 'JWT_USE'],
    [{}, { alg: 'none' }, 'JWT_ALG'],
    [{}, { kid: '' }, 'JWT_KID']
  ]) await assert.rejects(h.verify(h.token(pairs[0], claims, header)), code(expected));
  assert.equal(h.calls, 0, 'invalid claims must not trigger network requests');
  await assert.rejects(h.verify(h.token(pairs[1], {}, { kid: 'old' })), code('JWT_SIG'));
  assert.equal(h.calls, 1);
});
