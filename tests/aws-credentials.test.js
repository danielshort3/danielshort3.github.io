'use strict';

const assert = require('node:assert/strict');
const {
  DEFAULT_OIDC_AUDIENCE,
  resolveAwsCredentials
} = require('../api/_lib/aws-credentials');

const ROLE_ARN = 'arn:aws:iam::123456789012:role/website-tools-state-test';
const STATIC_SETS = [
  {
    name: 'service',
    accessKeyId: 'SERVICE_AWS_ACCESS_KEY_ID',
    secretAccessKey: 'SERVICE_AWS_SECRET_ACCESS_KEY',
    sessionToken: 'SERVICE_AWS_SESSION_TOKEN'
  },
  {
    name: 'default',
    accessKeyId: 'AWS_ACCESS_KEY_ID',
    secretAccessKey: 'AWS_SECRET_ACCESS_KEY',
    sessionToken: 'AWS_SESSION_TOKEN'
  }
];

function resolve(env, overrides = {}){
  return resolveAwsCredentials({
    env,
    service: 'tools-state',
    region: 'us-east-2',
    roleArnEnvKeys: ['SERVICE_AWS_ROLE_ARN'],
    staticCredentialSets: STATIC_SETS,
    ...overrides
  });
}

function expectCode(code, fn){
  assert.throws(fn, err => err && err.code === code);
}

function run(){
  const defaultConfig = resolve({});
  assert.equal(defaultConfig.source, 'default');
  assert.equal(defaultConfig.credentials, undefined);

  const serviceStatic = resolve({
    SERVICE_AWS_ACCESS_KEY_ID: 'AKIASERVICE',
    SERVICE_AWS_SECRET_ACCESS_KEY: 'service-secret',
    AWS_ACCESS_KEY_ID: 'AKIAGENERIC',
    AWS_SECRET_ACCESS_KEY: 'generic-secret'
  });
  assert.equal(serviceStatic.source, 'static');
  assert.equal(serviceStatic.staticCredentialSource, 'service');
  assert.equal(serviceStatic.credentials.accessKeyId, 'AKIASERVICE');

  const temporaryStatic = resolve({
    SERVICE_AWS_ACCESS_KEY_ID: 'ASIATEMPORARY',
    SERVICE_AWS_SECRET_ACCESS_KEY: 'service-secret',
    SERVICE_AWS_SESSION_TOKEN: 'temporary-token'
  });
  assert.equal(temporaryStatic.credentials.sessionToken, 'temporary-token');

  const longTermStatic = resolve({
    SERVICE_AWS_ACCESS_KEY_ID: 'AKIALONGTERM',
    SERVICE_AWS_SECRET_ACCESS_KEY: 'service-secret',
    SERVICE_AWS_SESSION_TOKEN: 'stale-token'
  });
  assert.equal(longTermStatic.credentials.sessionToken, undefined);

  let providerOptions = null;
  const oidcConfig = resolve({
    SERVICE_AWS_ROLE_ARN: ROLE_ARN,
    AWS_OIDC_AUDIENCE: 'sts.amazonaws.com',
    SERVICE_AWS_ACCESS_KEY_ID: 'incomplete-static-key'
  }, {
    providerFactory(options){
      providerOptions = options;
      return async () => ({ accessKeyId: 'test', secretAccessKey: 'test' });
    }
  });
  assert.equal(oidcConfig.source, 'oidc');
  assert.equal(typeof oidcConfig.credentials, 'function');
  assert.equal(providerOptions.roleArn, ROLE_ARN);
  assert.equal(providerOptions.audience, DEFAULT_OIDC_AUDIENCE);
  assert.equal(providerOptions.clientConfig.region, 'us-east-2');
  assert.equal(providerOptions.roleSessionName, 'vercel-tools-state');

  expectCode('AWS_OIDC_ROLE_MISSING', () => resolve({ AWS_AUTH_MODE: 'oidc' }));
  expectCode('AWS_OIDC_ROLE_INVALID', () => resolve({ SERVICE_AWS_ROLE_ARN: 'not-an-arn' }));
  expectCode('AWS_AUTH_MODE_INVALID', () => resolve({ AWS_AUTH_MODE: 'sometimes' }));
  expectCode('AWS_STATIC_CREDENTIALS_INCOMPLETE', () => resolve({
    SERVICE_AWS_ACCESS_KEY_ID: 'incomplete-static-key'
  }));
  expectCode('AWS_STATIC_SESSION_TOKEN_MISSING', () => resolve({
    SERVICE_AWS_ACCESS_KEY_ID: 'ASIATEMPORARY',
    SERVICE_AWS_SECRET_ACCESS_KEY: 'temporary-secret'
  }));
  expectCode('AWS_STATIC_CREDENTIALS_MISSING', () => resolve({
    AWS_AUTH_MODE: 'static',
    SERVICE_AWS_ROLE_ARN: ROLE_ARN
  }));

  const forcedStatic = resolve({
    AWS_AUTH_MODE: 'static',
    SERVICE_AWS_ROLE_ARN: ROLE_ARN,
    AWS_ACCESS_KEY_ID: 'AKIAGENERIC',
    AWS_SECRET_ACCESS_KEY: 'generic-secret'
  });
  assert.equal(forcedStatic.source, 'static');
  assert.equal(forcedStatic.staticCredentialSource, 'default');

  process.stdout.write('aws-credentials tests passed\n');
}

run();
