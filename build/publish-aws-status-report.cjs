'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { loadStatusCatalog } = require('../api/_lib/aws-status-catalog');

const REGION = 'us-east-2';
const HOUR_MS = 60 * 60_000;
const DAY_MS = 24 * HOUR_MS;
const CI_MAX_REPORT_AGE_MS = 30 * 60_000;
const CI_REPORT_LIFETIME_MS = 3 * HOUR_MS;
const CI_MAX_SESSION_MS = 15 * 60_000;
const MAX_REPORT_BYTES = 128 * 1024;
const STATUS = new Map([['pass', 'operational'], ['fail', 'degraded'], ['unknown', 'unknown']]);
const DEFAULT_PROFILE = 'website-admin';
const REPORT_FUNCTION_NAME = 'website-aws-status-report';
const ACCOUNT = '886623862678';
const ASSUMED_ROLE_PREFIX = `arn:aws:sts::${ACCOUNT}:assumed-role/`;
const CI_ROLE_PREFIX = `${ASSUMED_ROLE_PREFIX}website-aws-status-report-github/`;
const SESSION_ENV = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'];
const USAGE = 'Usage: node build/publish-aws-status-report.cjs <report.json> ' +
  '[--profile <temporary-sso-profile> | --github-oidc [<catalog-root>]]';

function prepareReport(report, now = Date.now(), { githubOidc = false, catalogRoot } = {}) {
  const checked = Date.parse(report?.checkedAt);
  const maxAge = githubOidc ? CI_MAX_REPORT_AGE_MS : DAY_MS;
  const maxClockSkew = githubOidc ? 0 : 60_000;
  if (report?.schemaVersion !== 1 || report.environment !== 'local' ||
    !Array.isArray(report.checks) || report.checks.length === 0 || report.checks.length > 70 ||
    !Number.isFinite(checked) || checked > now + maxClockSkew || checked < now - maxAge) {
    throw new Error('Expected a recent local status report with 1–70 checks.');
  }

  const checkedAt = new Date(checked).toISOString();
  if (githubOidc && report.checkedAt !== checkedAt) throw new Error('Expected a UTC browser report timestamp.');
  const catalogDir = catalogRoot ? path.resolve(catalogRoot, 'content') : undefined;
  const ids = new Set(loadStatusCatalog(catalogDir).map((item) => item.id));
  const expiresAt = new Date(checked + (githubOidc ? CI_REPORT_LIFETIME_MS : 7 * DAY_MS)).toISOString();
  const checks = report.checks.map((check) => {
    if (!ids.delete(check?.id) || !STATUS.has(check.status)) {
      throw new Error('The report contains a duplicate or unrecognized catalog check.');
    }
    if (githubOidc && check.method !== 'browser') {
      throw new Error('GitHub publishing requires browser results.');
    }
    return {
      id: check.id,
      state: STATUS.get(check.status),
      lastChecked: checkedAt,
      source: check.method === 'browser' ? 'browser' : 'local',
      expiresAt
    };
  });
  if (githubOidc && ids.size) throw new Error('GitHub publishing requires the entire catalog.');
  return { checks };
}

function invokeCli(args, env, executable) {
  const result = spawnSync(executable, args, {
    env, encoding: 'utf8', windowsHide: true, timeout: 30_000, maxBuffer: 128 * 1024
  });
  if (result.status !== 0 || result.error) throw new Error('AWS rejected the request or did not respond.');
  return result.stdout;
}

function githubSession(env, now) {
  const expiration = Date.parse(env.AWS_STATUS_GITHUB_CREDENTIAL_EXPIRATION);
  if (!/^ASIA[A-Z0-9]+$/.test(env.AWS_ACCESS_KEY_ID || '') ||
    !SESSION_ENV.every((name) => /^\S+$/.test(env[name] || '')) ||
    !Number.isFinite(expiration) || expiration <= now || expiration > now + CI_MAX_SESSION_MS) {
    throw new Error('Use a current 15-minute GitHub OIDC session.');
  }
  return Object.fromEntries(SESSION_ENV.map((name) => [name, env[name]]));
}

function parseJson(value, errorMessage) {
  try {
    return JSON.parse(value);
  } catch {
    throw new Error(errorMessage);
  }
}

function main(argv, runCli = invokeCli) {
  const [file, flag, value] = argv;
  const githubOidc = flag === '--github-oidc';
  const validArgs = githubOidc ? argv.length === 2 || (argv.length === 3 && value)
    : argv.length === 1 || (argv.length === 3 && flag === '--profile');
  if (!file || !validArgs) throw new Error(USAGE);
  const profile = githubOidc ? DEFAULT_PROFILE : value || DEFAULT_PROFILE;
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(profile)) throw new Error('Invalid AWS profile.');
  if (fs.statSync(file).size > MAX_REPORT_BYTES) throw new Error('The report exceeds 128 KB.');
  const now = Date.now();
  const report = parseJson(fs.readFileSync(file, 'utf8'), 'Invalid status report JSON.');
  const payload = prepareReport(report, now, { githubOidc, catalogRoot: githubOidc ? value : undefined });
  const credentials = githubOidc ? githubSession(process.env, now) : null;
  const env = { ...process.env, AWS_CLI_AUTO_PROMPT: 'off', AWS_PAGER: '' };
  if (githubOidc) {
    for (const name of Object.keys(env)) if (/^AWS_/i.test(name)) delete env[name];
    Object.assign(env, credentials, {
      AWS_CLI_AUTO_PROMPT: 'off', AWS_PAGER: '', AWS_EC2_METADATA_DISABLED: 'true'
    });
  } else {
    for (const name of [...SESSION_ENV, 'AWS_SECURITY_TOKEN', 'AWS_PROFILE', 'AWS_DEFAULT_PROFILE',
      'AWS_ROLE_ARN', 'AWS_WEB_IDENTITY_TOKEN_FILE']) delete env[name];
  }

  const profileFlags = githubOidc ? [] : ['--profile', profile];
  const flags = [...profileFlags, '--region', REGION, '--output', 'json', '--no-cli-pager'];
  const executable = githubOidc ? 'aws' : process.env.AWS_STATUS_AWS_CLI || 'aws';
  const functionArn = `arn:aws:lambda:${REGION}:${ACCOUNT}:function:${REPORT_FUNCTION_NAME}`;

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aws-status-report-'));
  const output = path.join(dir, 'response.json');
  const input = path.join(dir, 'request.json');
  const emptyConfig = path.join(dir, 'aws-config');
  try {
    if (githubOidc) {
      fs.writeFileSync(emptyConfig, '', { flag: 'wx' });
      env.AWS_CONFIG_FILE = emptyConfig;
      env.AWS_SHARED_CREDENTIALS_FILE = emptyConfig;
    }
    const identity = parseJson(runCli(['sts', 'get-caller-identity', ...flags], env, executable),
      'AWS returned an invalid caller identity.');
    if (identity?.Account !== ACCOUNT || typeof identity.Arn !== 'string' ||
      !identity.Arn.startsWith(ASSUMED_ROLE_PREFIX)) {
      throw new Error('Use a temporary role session in the website AWS account.');
    }
    if (githubOidc && (!identity.Arn.startsWith(CI_ROLE_PREFIX) ||
      !/^[\w+=,.@-]{2,64}$/.test(identity.Arn.slice(CI_ROLE_PREFIX.length)))) {
      throw new Error('Use the AWS Status GitHub reporter role.');
    }
    fs.writeFileSync(input, JSON.stringify(payload), { flag: 'wx' });
    const result = parseJson(runCli([
      'lambda', 'invoke', '--function-name', githubOidc ? `${functionArn}:browser-ci` : functionArn,
      '--payload', `fileb://${input}`, ...flags, output
    ], env, executable), 'AWS returned an invalid invocation response.');
    if (result.StatusCode !== 200 || result.FunctionError) throw new Error('The status reporter rejected the upload.');
    const receipt = parseJson(fs.readFileSync(output, 'utf8'), 'The status reporter returned an invalid receipt.');
    if (receipt.ok !== true || receipt.accepted !== payload.checks.length) {
      throw new Error('The status reporter did not acknowledge all checks.');
    }
    console.log(`Published ${receipt.accepted} check results.`);
  } finally {
    for (const file of [input, output, emptyConfig]) {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    }
    fs.rmdirSync(dir);
  }
}

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { main, prepareReport };
