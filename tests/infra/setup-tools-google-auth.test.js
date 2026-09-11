'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { DEFAULTS, parseOptions, buildUpdateInput, main } = require('../../scripts/setup-tools-google-auth.js');

const GOOGLE_CLIENT_ID = '123-example.apps.googleusercontent.com';
const options = { ...DEFAULTS, googleClientId: GOOGLE_CLIENT_ID };
const args = ['--google-client-id', GOOGLE_CLIENT_ID];
const originalClient = {
  UserPoolId: DEFAULTS.userPoolId,
  ClientId: DEFAULTS.clientId,
  ClientName: 'existing-client',
  ClientSecret: 'must-never-be-logged-or-written',
  CreationDate: '2026-01-01T00:00:00Z',
  LastModifiedDate: '2026-01-02T00:00:00Z',
  SupportedIdentityProviders: ['COGNITO', 'OtherProvider'],
  AllowedOAuthFlowsUserPoolClient: true,
  AllowedOAuthFlows: ['code'],
  AllowedOAuthScopes: ['email', 'openid', 'profile'],
  CallbackURLs: ['https://example.com/callback', 'http://localhost:4173/tools/dashboard'],
  LogoutURLs: ['https://example.com/logout'],
  DefaultRedirectURI: 'https://example.com/callback',
  ExplicitAuthFlows: ['ALLOW_USER_SRP_AUTH', 'ALLOW_REFRESH_TOKEN_AUTH'],
  RefreshTokenValidity: 45,
  AccessTokenValidity: 30,
  IdTokenValidity: 30,
  TokenValidityUnits: { RefreshToken: 'days', AccessToken: 'minutes', IdToken: 'minutes' },
  EnableTokenRevocation: true,
  PreventUserExistenceErrors: 'ENABLED',
  AuthSessionValidity: 5,
  RefreshTokenRotation: { Feature: 'DISABLED' }
};
const originalProvider = {
  UserPoolId: DEFAULTS.userPoolId,
  ProviderName: 'Google',
  ProviderType: 'Google',
  ClientId: GOOGLE_CLIENT_ID,
  Scopes: 'openid email profile',
  AttributeMapping: { email: 'email', email_verified: 'email_verified', name: 'name', username: 'sub' }
};
const skeleton = Object.fromEntries(Object.keys(originalClient)
  .filter(key => !['ClientSecret', 'CreationDate', 'LastModifiedDate'].includes(key)).map(key => [key, null]));
skeleton.WriteAttributes = [];
const clone = value => JSON.parse(JSON.stringify(value));

function fixture(overrides = {}) {
  const state = {
    client: clone(originalClient), provider: clone(originalProvider), calls: [], messages: [],
    clientReads: 0, providerReads: 0, updates: 0, inputPath: null, ...overrides
  };
  state.run = (executable, commandArgs, commandOptions) => {
    assert.equal(commandOptions.windowsHide, true);
    assert.deepEqual(commandOptions.stdio, ['ignore', 'pipe', 'pipe']);
    state.calls.push(commandArgs);
    const [service, operation] = commandArgs;
    if (service === 'sts') return JSON.stringify({ Account: state.account || DEFAULTS.expectedAccount });
    if (operation === 'describe-user-pool-client') {
      state.clientReads += 1;
      state.onClientRead?.(state);
      return JSON.stringify({ UserPoolClient: state.client });
    }
    if (operation === 'describe-identity-provider') {
      state.providerReads += 1;
      state.onProviderRead?.(state);
      assert.ok(commandArgs.includes('--query'));
      assert.ok(!commandArgs[commandArgs.indexOf('--query') + 1].includes('client_secret'));
      return JSON.stringify(state.provider);
    }
    assert.equal(operation, 'update-user-pool-client');
    if (commandArgs.includes('--generate-cli-skeleton')) return JSON.stringify(state.skeleton || skeleton);
    state.updates += 1;
    state.inputPath = commandArgs[commandArgs.indexOf('--cli-input-json') + 1].slice('file://'.length);
    const inputText = fs.readFileSync(state.inputPath, 'utf8');
    assert.ok(!inputText.includes(originalClient.ClientSecret));
    state.written = JSON.parse(inputText);
    if (state.failUpdate) throw new Error('AWS secret-bearing error must-never-be-logged-or-written');
    state.client = { ...state.written, LastModifiedDate: '2026-02-01T00:00:00Z' };
    return '{}';
  };
  state.execute = apply => main(apply ? [...args, '--apply'] : args, state.run, message => state.messages.push(message));
  return state;
}

test('setup requires expected Google client ID and remains read-only by default', () => {
  assert.equal(parseOptions(args).apply, false);
  assert.equal(parseOptions([...args, '--apply']).apply, true);
  assert.equal(parseOptions(['--help']).help, true);
  assert.throws(() => parseOptions([]), /google-client-id/);
  assert.throws(() => parseOptions([...args, '--google-client-secret', 'secret']), /Invalid option/);
  const state = fixture();
  state.execute(false);
  assert.equal(state.updates, 0);
  assert.equal(state.clientReads, 1);
  assert.ok(state.messages.join('\n').includes('CognitoIdentityProviders=COGNITO,OtherProvider,Google'));
  assert.ok(!state.messages.join('\n').includes(originalClient.ClientSecret));
});

test('Google addition preserves existing providers, callback URLs, nested settings, and source data', () => {
  const source = clone(originalClient);
  const input = buildUpdateInput(source, skeleton, originalProvider, options);
  assert.deepEqual(input.SupportedIdentityProviders, ['COGNITO', 'OtherProvider', 'Google']);
  const expected = clone(originalClient);
  for (const field of ['ClientSecret', 'CreationDate', 'LastModifiedDate']) delete expected[field];
  expected.SupportedIdentityProviders.push('Google');
  assert.deepEqual(input, expected);
  assert.deepEqual(source, originalClient);
});

test('fails closed for an unexpected AWS account or mismatched Cognito resource', () => {
  const state = fixture({ account: '000000000000' });
  assert.throws(() => state.execute(true), /account mismatch/);
  assert.equal(state.calls.length, 1);
  assert.throws(() => buildUpdateInput({ ...originalClient, ClientId: 'different' }, skeleton, originalProvider, options), /does not match/);
  assert.throws(() => buildUpdateInput(originalClient, skeleton, { ...originalProvider, UserPoolId: 'different' }, options), /does not match/);
  assert.throws(() => buildUpdateInput(originalClient, skeleton, { ...originalProvider, ClientId: 'different' }, options), /different OAuth client/);
});

test('does not discard fields unknown to the installed AWS CLI', () => {
  assert.throws(() => buildUpdateInput({ ...originalClient, FutureSetting: true }, skeleton, originalProvider, options), /unsupported client fields: FutureSetting/);
});

test('refuses unusable provider scopes, email mappings, or restricted mapped-attribute permissions', () => {
  assert.throws(() => buildUpdateInput(originalClient, skeleton, { ...originalProvider, Scopes: 'profile' }, options), /openid and email/);
  assert.throws(() => buildUpdateInput(originalClient, skeleton, { ...originalProvider, AttributeMapping: {} }, options), /map the Google email/);
  assert.throws(() => buildUpdateInput({ ...originalClient, WriteAttributes: ['email'] }, skeleton, originalProvider, options), /email_verified, name/);
  const input = buildUpdateInput({ ...originalClient, WriteAttributes: ['email', 'email_verified', 'name'] }, skeleton, originalProvider, options);
  assert.deepEqual(input.WriteAttributes, ['email', 'email_verified', 'name']);
});

test('does not silently change email sign-in or authorization flow configuration', () => {
  assert.throws(() => buildUpdateInput({ ...originalClient, SupportedIdentityProviders: [] }, skeleton, originalProvider, options), /COGNITO email sign-in/);
  assert.throws(() => buildUpdateInput({ ...originalClient, AllowedOAuthFlows: ['implicit'] }, skeleton, originalProvider, options), /authorization-code flow/);
});

test('apply verifies readback and removes its temporary input', () => {
  const state = fixture();
  state.execute(true);
  assert.equal(state.updates, 1);
  assert.equal(state.clientReads, 3);
  assert.equal(state.providerReads, 3);
  assert.deepEqual(state.written, buildUpdateInput(originalClient, skeleton, originalProvider, options));
  assert.equal(fs.existsSync(state.inputPath), false);
  assert.equal(fs.existsSync(path.dirname(state.inputPath)), false);
  assert.ok(state.messages.at(-1).includes('verified unchanged'));
});

test('already-enabled Google is an idempotent no-op', () => {
  const client = clone(originalClient);
  client.SupportedIdentityProviders.push('Google');
  const state = fixture({ client });
  state.execute(true);
  assert.equal(state.updates, 0);
  assert.ok(state.messages.at(-1).includes('already enabled'));
});

test('rejects concurrent client changes, including another operator adding Google', () => {
  for (const edit of [
    client => { client.RefreshTokenValidity = 10; },
    client => { client.SupportedIdentityProviders.push('Google'); }
  ]) {
    const state = fixture({ onClientRead(current) { if (current.clientReads === 2) edit(current.client); } });
    assert.throws(() => state.execute(true), /changed during preview/);
    assert.equal(state.updates, 0);
  }
});

test('rejects concurrent Google provider changes before updating the client', () => {
  const state = fixture({ onProviderRead(current) { if (current.providerReads === 2) current.provider.ClientId = 'changed'; } });
  assert.throws(() => state.execute(true), /changed during preview/);
  assert.equal(state.updates, 0);
});

test('readback detects a lost provider or an unrelated changed setting', () => {
  for (const edit of [
    client => { client.RefreshTokenValidity = 10; },
    client => { client.SupportedIdentityProviders = ['COGNITO']; }
  ]) {
    const state = fixture({ onClientRead(current) { if (current.clientReads === 3) edit(current.client); } });
    assert.throws(() => state.execute(true), /readback found a settings difference/);
    assert.equal(state.updates, 1);
    assert.equal(fs.existsSync(state.inputPath), false);
  }
});

test('AWS mutation failures redact response details and still clean up input files', () => {
  const state = fixture({ failUpdate: true });
  assert.throws(() => state.execute(true), error => {
    assert.match(error.message, /no AWS response body was printed/);
    assert.ok(!error.message.includes(originalClient.ClientSecret));
    return true;
  });
  assert.equal(fs.existsSync(state.inputPath), false);
  assert.equal(fs.existsSync(path.dirname(state.inputPath)), false);
});

test('CloudFormation requires an explicit provider list and keeps the external provider unmanaged', () => {
  const template = fs.readFileSync(path.join(__dirname, '../../aws/job-application-tracker/template.yaml'), 'utf8');
  const parameter = template.match(/^  CognitoIdentityProviders:\r?\n([\s\S]*?)(?=^  \w+:)/m)?.[1];
  assert.ok(parameter, 'The provider-list parameter must exist.');
  assert.match(parameter, /Type: CommaDelimitedList/);
  assert.doesNotMatch(parameter, /Default:/, 'A new template adoption must not silently choose COGNITO alone.');
  assert.match(template, /SupportedIdentityProviders: !Ref CognitoIdentityProviders/);
  assert.match(template, /KeepCognitoEmailSignIn:[\s\S]*?Fn::Contains:[\s\S]*?!Ref CognitoIdentityProviders[\s\S]*?- COGNITO/);
  assert.doesNotMatch(template, /Type: AWS::Cognito::UserPoolIdentityProvider/);
});
