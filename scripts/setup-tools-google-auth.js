#!/usr/bin/env node
'use strict';

// Enable an existing Google provider without resetting unrelated Cognito client settings.
// Google credentials belong in the provider configuration, never this script or its input file.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { DEFAULTS: LOCAL_DEFAULTS, sameSettings } = require('./setup-tools-local-auth.js');

const DEFAULTS = Object.freeze({
  region: LOCAL_DEFAULTS.region,
  userPoolId: LOCAL_DEFAULTS.userPoolId,
  clientId: LOCAL_DEFAULTS.clientId,
  expectedAccount: LOCAL_DEFAULTS.expectedAccount
});
const READ_ONLY_FIELDS = new Set(['ClientSecret', 'CreationDate', 'LastModifiedDate']);
const PROVIDER_QUERY = 'IdentityProvider.{UserPoolId:UserPoolId,ProviderName:ProviderName,ProviderType:ProviderType,ClientId:ProviderDetails.client_id,Scopes:ProviderDetails.authorize_scopes,AttributeMapping:AttributeMapping}';

function parseOptions(args) {
  const options = { ...DEFAULTS, apply: false };
  const flags = {
    '--region': 'region', '--user-pool-id': 'userPoolId', '--client-id': 'clientId',
    '--expected-account': 'expectedAccount', '--profile': 'profile', '--aws-cli': 'awsCli',
    '--google-client-id': 'googleClientId'
  };
  for (let index = 0; index < args.length; index += 1) {
    const flag = args[index];
    if (flag === '--apply') { options.apply = true; continue; }
    if (flag === '--help' || flag === '-h') { options.help = true; continue; }
    const key = flags[flag];
    if (!key || !args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`Invalid option: ${flag}`);
    options[key] = args[++index];
  }
  if (!/^\d{12}$/.test(options.expectedAccount)) throw new Error('Expected account must contain 12 digits.');
  if (!options.help && !/^[a-zA-Z0-9-]+\.apps\.googleusercontent\.com$/.test(options.googleClientId || '')) {
    throw new Error('--google-client-id must be the expected Google OAuth web client ID.');
  }
  return options;
}

function validateProvider(provider, options) {
  if (!provider || provider.UserPoolId !== options.userPoolId || provider.ProviderName !== 'Google' || provider.ProviderType !== 'Google') {
    throw new Error('The existing Google provider does not match the requested Cognito pool.');
  }
  if (provider.ClientId !== options.googleClientId) throw new Error('The Google provider uses a different OAuth client ID. Nothing changed.');
  const scopes = new Set(String(provider.Scopes || '').split(/[\s,]+/));
  if (!scopes.has('openid') || !(scopes.has('email') || scopes.has('https://www.googleapis.com/auth/userinfo.email'))) {
    throw new Error('The Google provider must request openid and email scopes before it can be enabled.');
  }
  if (provider.AttributeMapping?.email !== 'email') throw new Error('The Google provider must map the Google email claim to Cognito email.');
}

function clientSettings(current, skeleton, options) {
  if (!current || current.ClientId !== options.clientId || current.UserPoolId !== options.userPoolId) {
    throw new Error('The described Cognito client does not match the requested pool and client.');
  }
  const unsupported = Object.keys(current).filter(key => !READ_ONLY_FIELDS.has(key) && !Object.hasOwn(skeleton, key));
  if (unsupported.length) throw new Error(`Update AWS CLI before proceeding; unsupported client fields: ${unsupported.join(', ')}`);
  return Object.fromEntries(Object.entries(current).filter(([key]) => !READ_ONLY_FIELDS.has(key)));
}

function buildUpdateInput(current, skeleton, provider, options) {
  validateProvider(provider, options);
  const input = clientSettings(current, skeleton, options);
  if (!input.SupportedIdentityProviders?.includes('COGNITO')) throw new Error('The client must already support COGNITO email sign-in. Nothing changed.');
  if (!input.AllowedOAuthFlowsUserPoolClient || !input.AllowedOAuthFlows?.includes('code') || !input.AllowedOAuthScopes?.includes('openid')) {
    throw new Error('The client must already support the OAuth authorization-code flow and openid scope. Nothing changed.');
  }
  if (input.WriteAttributes) {
    const missing = Object.keys(provider.AttributeMapping || {}).filter(attribute => attribute !== 'username' && !input.WriteAttributes.includes(attribute));
    if (missing.length) throw new Error(`The client cannot write mapped Google attributes: ${missing.join(', ')}. Review attribute permissions first.`);
  }
  input.SupportedIdentityProviders = [...new Set([...input.SupportedIdentityProviders, 'Google'])];
  return input;
}

function main(args, runCommand = execFileSync, log = console.log) {
  const options = parseOptions(args);
  if (options.help) {
    log('Usage: node scripts/setup-tools-google-auth.js --google-client-id CLIENT.apps.googleusercontent.com [--apply] [--profile NAME]');
    log('Optional: --region REGION --user-pool-id POOL --client-id CLIENT --expected-account ACCOUNT --aws-cli PATH');
    log('Without --apply, validates an existing Google provider and previews the app-client change. Does not create providers or accept secrets.');
    return;
  }
  const windowsAws = 'C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe';
  const executable = options.awsCli || (process.platform === 'win32' && fs.existsSync(windowsAws) ? windowsAws : 'aws');
  const commonArgs = ['--region', options.region, '--output', 'json', '--no-cli-pager'];
  if (options.profile) commonArgs.push('--profile', options.profile);
  const aws = (service, operation, extra = []) => {
    try {
      return JSON.parse(runCommand(executable, [service, operation, ...extra, ...commonArgs], {
        encoding: 'utf8', windowsHide: true, maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, AWS_PAGER: '', AWS_CLI_AUTO_PROMPT: 'off' },
        stdio: ['ignore', 'pipe', 'pipe']
      }));
    } catch {
      throw new Error(`AWS CLI ${service} ${operation} failed. Check credentials, region, provider setup, and permissions; no AWS response body was printed.`);
    }
  };
  const identity = aws('sts', 'get-caller-identity');
  if (identity.Account !== options.expectedAccount) throw new Error('AWS account mismatch. Nothing changed.');
  const identifiers = ['--user-pool-id', options.userPoolId, '--client-id', options.clientId];
  const readClient = () => aws('cognito-idp', 'describe-user-pool-client', identifiers).UserPoolClient;
  // The query intentionally excludes ProviderDetails.client_secret before stdout reaches Node.
  const readProvider = () => aws('cognito-idp', 'describe-identity-provider', [
    '--user-pool-id', options.userPoolId, '--provider-name', 'Google', '--query', PROVIDER_QUERY
  ]);
  const current = readClient();
  const provider = readProvider();
  const skeleton = aws('cognito-idp', 'update-user-pool-client', ['--generate-cli-skeleton', 'input']);
  const input = buildUpdateInput(current, skeleton, provider, options);
  const additions = input.SupportedIdentityProviders.filter(name => !current.SupportedIdentityProviders.includes(name));
  log(JSON.stringify({
    mode: options.apply ? 'apply' : 'preview', account: identity.Account,
    region: options.region, userPoolId: options.userPoolId, clientId: options.clientId,
    providerAdditions: additions, supportedIdentityProviders: input.SupportedIdentityProviders,
    cloudFormationParameter: `CognitoIdentityProviders=${input.SupportedIdentityProviders.join(',')}`,
    preservedSettings: Object.keys(input).filter(key => key !== 'SupportedIdentityProviders')
  }, null, 2));
  if (!options.apply || !additions.length) {
    log(additions.length ? 'Preview only. Add --apply to enable Google for this app client.' : 'Google is already enabled. No update needed.');
    return;
  }
  const latest = clientSettings(readClient(), skeleton, options);
  const latestProvider = readProvider();
  if (!sameSettings(clientSettings(current, skeleton, options), latest) || !sameSettings(provider, latestProvider)) {
    throw new Error('The Cognito client or Google provider changed during preview. Run this command again to use current settings.');
  }
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tools-google-auth-'));
  const inputPath = path.join(tempDir, 'update-client.json');
  try {
    fs.writeFileSync(inputPath, JSON.stringify(input), { encoding: 'utf8', mode: 0o600 });
    aws('cognito-idp', 'update-user-pool-client', ['--cli-input-json', `file://${inputPath}`]);
  } finally {
    if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
    fs.rmdirSync(tempDir);
  }
  const verified = clientSettings(readClient(), skeleton, options);
  if (!sameSettings(input, verified) || !sameSettings(provider, readProvider())) {
    throw new Error('Cognito accepted the update, but readback found a settings difference. Review the app client and Google provider.');
  }
  log('Google enabled. COGNITO sign-in and all other supported app-client settings verified unchanged.');
}

module.exports = { DEFAULTS, parseOptions, validateProvider, clientSettings, buildUpdateInput, main };
if (require.main === module) {
  try { main(process.argv.slice(2)); } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
