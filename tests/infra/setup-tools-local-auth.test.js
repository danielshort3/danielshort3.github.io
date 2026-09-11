'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { DEFAULTS, parseOptions, callbackUrls, buildUpdateInput, main } = require('../../scripts/setup-tools-local-auth.js');

const localUrls = callbackUrls([4181]);
const currentClient = {
  UserPoolId: DEFAULTS.userPoolId,
  ClientId: DEFAULTS.clientId,
  ClientSecret: 'never-print-this-client-secret',
  CreationDate: '2026-01-01T00:00:00Z',
  LastModifiedDate: '2026-01-02T00:00:00Z',
  ClientName: 'website-client',
  CallbackURLs: ['https://www.danielshort.me/tools/dashboard', ...localUrls],
  LogoutURLs: ['https://www.danielshort.me/tools/dashboard'],
  SupportedIdentityProviders: ['COGNITO', 'Google'],
  AllowedOAuthFlows: ['code'],
  AllowedOAuthScopes: ['openid', 'email', 'profile'],
  AllowedOAuthFlowsUserPoolClient: true,
  RefreshTokenValidity: 30,
  AccessTokenValidity: 60,
  TokenValidityUnits: { AccessToken: 'minutes', RefreshToken: 'days' },
  EnableTokenRevocation: true
};
const skeleton = Object.fromEntries(Object.keys(currentClient)
  .filter(key => !['ClientSecret', 'CreationDate', 'LastModifiedDate'].includes(key)).map(key => [key, null]));
const clone = value => JSON.parse(JSON.stringify(value));

function fixture(overrides = {}) {
  const state = { client: clone(currentClient), calls: [], logs: [], reads: 0, updates: 0, inputPath: null, ...overrides };
  state.run = (executable, args, commandOptions) => {
    assert.equal(commandOptions.windowsHide, true);
    state.calls.push(args);
    if (args[0] === 'sts') return JSON.stringify({ Account: state.account || DEFAULTS.expectedAccount });
    if (args[1] === 'describe-user-pool-client') {
      state.reads += 1;
      state.onRead?.(state);
      return JSON.stringify({ UserPoolClient: state.client });
    }
    assert.equal(args[1], 'update-user-pool-client');
    if (args.includes('--generate-cli-skeleton')) return JSON.stringify(skeleton);
    state.updates += 1;
    state.inputPath = args[args.indexOf('--cli-input-json') + 1].slice('file://'.length);
    const raw = fs.readFileSync(state.inputPath, 'utf8');
    assert.ok(!raw.includes(currentClient.ClientSecret));
    state.written = JSON.parse(raw);
    if (state.failUpdate) throw new Error(`AWS response contains ${currentClient.ClientSecret}`);
    state.client = clone(state.written);
    return '{}';
  };
  state.execute = apply => main(['--ports', '4181', ...(apply ? ['--apply'] : [])], state.run, message => state.logs.push(message));
  return state;
}

test('preview reports missing logout URLs even when all callbacks are already registered', () => {
  assert.equal(parseOptions([]).apply, false);
  const state = fixture();
  state.execute(false);
  const preview = JSON.parse(state.logs[0]);
  assert.deepEqual(preview.callbackAdditions, []);
  assert.deepEqual(preview.logoutAdditions, localUrls);
  assert.equal(state.updates, 0);
  assert.ok(!state.logs.join('\n').includes(currentClient.ClientSecret));
});

test('merges both URL lists while preserving production URLs, Google, and unrelated nested settings', () => {
  const source = clone(currentClient);
  source.CallbackURLs = [currentClient.CallbackURLs[0]];
  const before = clone(source);
  const input = buildUpdateInput(source, skeleton, localUrls, DEFAULTS);
  assert.deepEqual(input.CallbackURLs, currentClient.CallbackURLs);
  assert.deepEqual(input.LogoutURLs, [currentClient.LogoutURLs[0], ...localUrls]);
  const expected = { ...before, CallbackURLs: input.CallbackURLs, LogoutURLs: input.LogoutURLs };
  for (const field of ['ClientSecret', 'CreationDate', 'LastModifiedDate']) delete expected[field];
  assert.deepEqual(input, expected);
  assert.deepEqual(source, before);
  const absentLists = clone(source);
  delete absentLists.CallbackURLs;
  delete absentLists.LogoutURLs;
  const unchanged = buildUpdateInput(absentLists, skeleton, [], DEFAULTS);
  assert.equal(Object.hasOwn(unchanged, 'CallbackURLs'), false);
  assert.equal(Object.hasOwn(unchanged, 'LogoutURLs'), false);
});

test('apply fills logout-only gaps and verifies cleanup and idempotency', () => {
  const state = fixture();
  state.execute(true);
  assert.equal(state.updates, 1);
  assert.equal(state.reads, 3);
  assert.deepEqual(state.written.CallbackURLs, currentClient.CallbackURLs);
  assert.deepEqual(state.written.LogoutURLs, [currentClient.LogoutURLs[0], ...localUrls]);
  assert.equal(fs.existsSync(state.inputPath), false);
  assert.equal(fs.existsSync(path.dirname(state.inputPath)), false);
  state.execute(true);
  assert.equal(state.updates, 1, 'A repeated apply must not update the client again.');
});

test('both Cognito URL quotas are checked independently', () => {
  for (const field of ['CallbackURLs', 'LogoutURLs']) {
    const source = clone(currentClient);
    source[field] = Array.from({ length: 100 }, (_, index) => `https://example.com/${index}`);
    assert.throws(() => buildUpdateInput(source, skeleton, localUrls, DEFAULTS), new RegExp(`${field}.*100-URL`));
  }
});

test('fails closed on wrong account or unknown mutable fields', () => {
  const state = fixture({ account: '000000000000' });
  assert.throws(() => state.execute(true), /account mismatch/);
  assert.equal(state.calls.length, 1);
  assert.throws(() => buildUpdateInput({ ...currentClient, NewSetting: true }, skeleton, localUrls, DEFAULTS), /unsupported client fields: NewSetting/);
});

test('concurrent changes are not hidden by merging requested URLs', () => {
  for (const edit of [
    client => { client.LogoutURLs.push(localUrls[0]); },
    client => { client.CallbackURLs.pop(); },
    client => { client.SupportedIdentityProviders = ['COGNITO']; }
  ]) {
    const state = fixture({ onRead(current) { if (current.reads === 2) edit(current.client); } });
    assert.throws(() => state.execute(true), /changed during preview/);
    assert.equal(state.updates, 0);
  }
});

test('readback detects missing logout URLs or unrelated settings changes', () => {
  for (const edit of [
    client => { client.LogoutURLs.pop(); },
    client => { client.RefreshTokenValidity = 7; }
  ]) {
    const state = fixture({ onRead(current) { if (current.reads === 3) edit(current.client); } });
    assert.throws(() => state.execute(true), /verification found a settings difference/);
    assert.equal(state.updates, 1);
    assert.equal(fs.existsSync(state.inputPath), false);
  }
});

test('failed updates redact AWS response details and remove temporary input', () => {
  const state = fixture({ failUpdate: true });
  assert.throws(() => state.execute(true), error => {
    assert.match(error.message, /no AWS response body was printed/);
    assert.ok(!error.message.includes(currentClient.ClientSecret));
    return true;
  });
  assert.equal(fs.existsSync(state.inputPath), false);
  assert.equal(fs.existsSync(path.dirname(state.inputPath)), false);
});
