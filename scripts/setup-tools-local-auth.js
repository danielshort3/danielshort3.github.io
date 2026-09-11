#!/usr/bin/env node
'use strict';

// Cognito updates reset omitted settings. Preserve the described client and add local sign-in/sign-out URLs.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { isDeepStrictEqual } = require('node:util');

const DEFAULTS = Object.freeze({
  region: 'us-east-2',
  userPoolId: 'us-east-2_y80EG3pKd',
  clientId: '78oo663obb0t28u63u9bqn00o9',
  expectedAccount: '886623862678',
  ports: [3000, 4173, 4181]
});
const READ_ONLY_FIELDS = new Set(['ClientSecret', 'CreationDate', 'LastModifiedDate']);
const UNORDERED_FIELDS = new Set([
  'CallbackURLs', 'LogoutURLs', 'AllowedOAuthScopes', 'AllowedOAuthFlows', 'ExplicitAuthFlows',
  'SupportedIdentityProviders', 'ReadAttributes', 'WriteAttributes'
]);

function normalizeSettings(input) {
  return Object.fromEntries(Object.entries(input).map(([key, value]) => [
    key, UNORDERED_FIELDS.has(key) && Array.isArray(value) ? value.slice().sort() : value
  ]));
}

function sameSettings(left, right) {
  return isDeepStrictEqual(normalizeSettings(left), normalizeSettings(right));
}

function parseOptions(args) {
  const options = { ...DEFAULTS, apply: false };
  const flags = {
    '--region': 'region', '--user-pool-id': 'userPoolId', '--client-id': 'clientId',
    '--expected-account': 'expectedAccount', '--profile': 'profile', '--aws-cli': 'awsCli'
  };
  for (let index = 0; index < args.length; index += 1) {
    const flag = args[index];
    if (flag === '--apply') { options.apply = true; continue; }
    if (flag === '--help' || flag === '-h') { options.help = true; continue; }
    if (flag === '--ports') {
      const value = args[++index];
      if (!value) throw new Error('--ports requires a comma-separated list.');
      options.ports = [...new Set(value.split(',').map(Number))];
      if (options.ports.some(port => !Number.isInteger(port) || port < 1 || port > 65535)) {
        throw new Error('Each port must be an integer from 1 through 65535.');
      }
      continue;
    }
    const key = flags[flag];
    if (!key || !args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`Invalid option: ${flag}`);
    options[key] = args[++index];
  }
  if (!/^\d{12}$/.test(options.expectedAccount)) throw new Error('Expected account must contain 12 digits.');
  return options;
}

function callbackUrls(ports) {
  return ports.flatMap(port => ['localhost', '127.0.0.1'].map(host => `http://${host}:${port}/tools/dashboard`));
}

function buildUpdateInput(current, skeleton, urls, options) {
  if (!current || current.ClientId !== options.clientId || current.UserPoolId !== options.userPoolId) {
    throw new Error('The described Cognito client does not match the requested pool and client.');
  }
  const unsupported = Object.keys(current).filter(key => !READ_ONLY_FIELDS.has(key) && !Object.hasOwn(skeleton, key));
  if (unsupported.length) throw new Error(`Update AWS CLI before proceeding; unsupported client fields: ${unsupported.join(', ')}`);
  const input = Object.fromEntries(Object.entries(current).filter(([key]) => !READ_ONLY_FIELDS.has(key)));
  for (const field of ['CallbackURLs', 'LogoutURLs']) {
    if (urls.length) {
      if (!Object.hasOwn(skeleton, field)) throw new Error(`Update AWS CLI before proceeding; unsupported client field: ${field}`);
      input[field] = [...new Set([...(current[field] || []), ...urls])];
    }
    if (input[field]?.length > 100) throw new Error(`The merged ${field} list exceeds Cognito's 100-URL limit.`);
  }
  return input;
}

function main(args, runCommand = execFileSync, log = console.log) {
  const options = parseOptions(args);
  if (options.help) {
    log('Usage: node scripts/setup-tools-local-auth.js [--apply] [--ports 3000,4173,4181] [--profile NAME]');
    log('Optional: --region REGION --user-pool-id POOL --client-id CLIENT --expected-account ACCOUNT --aws-cli PATH');
    log('Without --apply, reads AWS configuration and prints local callback and sign-out URL additions only.');
    return;
  }
  const windowsAws = 'C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe';
  const executable = options.awsCli || (process.platform === 'win32' && fs.existsSync(windowsAws) ? windowsAws : 'aws');
  const commonArgs = ['--region', options.region, '--output', 'json', '--no-cli-pager'];
  if (options.profile) commonArgs.push('--profile', options.profile);
  const aws = (service, operation, extra = []) => {
    try {
      const output = runCommand(executable, [service, operation, ...extra, ...commonArgs], {
        encoding: 'utf8', windowsHide: true, maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, AWS_PAGER: '', AWS_CLI_AUTO_PROMPT: 'off' },
        stdio: ['ignore', 'pipe', 'pipe']
      });
      return JSON.parse(output);
    } catch {
      throw new Error(`AWS CLI ${service} ${operation} failed. Check CLI availability, credentials, region, and permissions; no AWS response body was printed.`);
    }
  };
  const identity = aws('sts', 'get-caller-identity');
  if (identity.Account !== options.expectedAccount) {
    throw new Error(`AWS account mismatch: expected ${options.expectedAccount}, received ${identity.Account || 'unknown'}. Nothing changed.`);
  }
  const identifiers = ['--user-pool-id', options.userPoolId, '--client-id', options.clientId];
  const current = aws('cognito-idp', 'describe-user-pool-client', identifiers).UserPoolClient;
  const skeleton = aws('cognito-idp', 'update-user-pool-client', ['--generate-cli-skeleton', 'input']);
  const input = buildUpdateInput(current, skeleton, callbackUrls(options.ports), options);
  const callbackAdditions = input.CallbackURLs.filter(url => !(current.CallbackURLs || []).includes(url));
  const logoutAdditions = input.LogoutURLs.filter(url => !(current.LogoutURLs || []).includes(url));
  const hasAdditions = callbackAdditions.length > 0 || logoutAdditions.length > 0;
  log(JSON.stringify({
    mode: options.apply ? 'apply' : 'preview', account: identity.Account,
    region: options.region, userPoolId: options.userPoolId, clientId: options.clientId,
    callbackAdditions, logoutAdditions,
    existingCallbackCount: (current.CallbackURLs || []).length,
    existingLogoutCount: (current.LogoutURLs || []).length,
    preservedSettings: Object.keys(input).filter(key => !['CallbackURLs', 'LogoutURLs'].includes(key))
  }, null, 2));
  if (!options.apply || !hasAdditions) {
    log(hasAdditions ? 'Preview only. Add --apply to register these exact callback and sign-out URLs.' : 'All requested callback and sign-out URLs are already registered. No update needed.');
    return;
  }
  // Re-read immediately before writing so a concurrent settings edit is never silently overwritten.
  const latest = aws('cognito-idp', 'describe-user-pool-client', identifiers).UserPoolClient;
  const latestInput = buildUpdateInput(latest, skeleton, [], options);
  const originalInput = buildUpdateInput(current, skeleton, [], options);
  if (!sameSettings(originalInput, latestInput)) throw new Error('The Cognito client changed during preview. Run this command again to use its current settings.');
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tools-local-auth-'));
  const inputPath = path.join(tempDir, 'update-client.json');
  try {
    fs.writeFileSync(inputPath, JSON.stringify(input), { encoding: 'utf8', mode: 0o600 });
    aws('cognito-idp', 'update-user-pool-client', ['--cli-input-json', `file://${inputPath}`]);
  } finally {
    if (fs.existsSync(inputPath)) fs.unlinkSync(inputPath);
    fs.rmdirSync(tempDir);
  }
  const verified = aws('cognito-idp', 'describe-user-pool-client', identifiers).UserPoolClient;
  const verifiedInput = buildUpdateInput(verified, skeleton, [], options);
  if (!sameSettings(input, verifiedInput)) {
    throw new Error('Cognito accepted the update, but verification found a settings difference. Review the app client before continuing.');
  }
  log('Local callback and sign-out URLs registered. Existing URLs and all other supported client settings verified unchanged.');
}

module.exports = { DEFAULTS, parseOptions, callbackUrls, buildUpdateInput, sameSettings, main };
if (require.main === module) {
  try { main(process.argv.slice(2)); } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
