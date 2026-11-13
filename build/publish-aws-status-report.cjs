'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { loadStatusCatalog } = require('../api/_lib/aws-status-catalog');

const REGION = 'us-east-2';
const DAY_MS = 24 * 60 * 60_000;
const MAX_REPORT_BYTES = 128 * 1024;
const STATUS = new Map([['pass', 'operational'], ['fail', 'degraded'], ['unknown', 'unknown']]);
const DEFAULT_PROFILE = 'website-admin';
const REPORT_FUNCTION_NAME = 'website-aws-status-report';
const ACCOUNT = '886623862678';

function prepareReport(report, now = Date.now()) {
  const checked = Date.parse(report?.checkedAt);
  if (report?.schemaVersion !== 1 || report.environment !== 'local' ||
    !Array.isArray(report.checks) || report.checks.length === 0 || report.checks.length > 70 ||
    !Number.isFinite(checked) || checked > now + 60_000 || checked < now - DAY_MS) {
    throw new Error('Expected a recent local status report with 1–70 checks.');
  }

  const ids = new Set(loadStatusCatalog().map((item) => item.id));
  const checkedAt = new Date(checked).toISOString();
  const expiresAt = new Date(checked + 7 * DAY_MS).toISOString();
  const checks = report.checks.map((check) => {
    if (!ids.delete(check?.id) || !STATUS.has(check.status)) {
      throw new Error('The report contains a duplicate or unrecognized catalog check.');
    }
    return {
      id: check.id,
      state: STATUS.get(check.status),
      lastChecked: checkedAt,
      source: check.method === 'browser' ? 'browser' : 'local',
      expiresAt
    };
  });
  return { checks };
}

function invokeCli(args, env) {
  const result = spawnSync(process.env.AWS_STATUS_AWS_CLI || 'aws', args, {
    env, encoding: 'utf8', windowsHide: true, timeout: 30_000, maxBuffer: 128 * 1024
  });
  if (result.status !== 0 || result.error) throw new Error('AWS rejected the request or did not respond.');
  return result.stdout;
}

function main(argv) {
  const [file, flag, profileInput] = argv;
  if (!file || (argv.length !== 1 && (argv.length !== 3 || flag !== '--profile'))) {
    throw new Error('Usage: node build/publish-aws-status-report.cjs <report.json> [--profile <temporary-sso-profile>]');
  }
  const profile = profileInput || DEFAULT_PROFILE;
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(profile)) throw new Error('Invalid AWS profile.');
  if (fs.statSync(file).size > MAX_REPORT_BYTES) throw new Error('The report exceeds 128 KB.');
  const payload = prepareReport(JSON.parse(fs.readFileSync(file, 'utf8')));
  const env = { ...process.env, AWS_CLI_AUTO_PROMPT: 'off', AWS_PAGER: '' };
  for (const name of ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_SECURITY_TOKEN',
    'AWS_PROFILE', 'AWS_DEFAULT_PROFILE', 'AWS_ROLE_ARN', 'AWS_WEB_IDENTITY_TOKEN_FILE']) delete env[name];

  const flags = ['--profile', profile, '--region', REGION, '--output', 'json', '--no-cli-pager'];
  const identity = JSON.parse(invokeCli(['sts', 'get-caller-identity', ...flags], env));
  if (identity.Account !== ACCOUNT || typeof identity.Arn !== 'string' ||
    !identity.Arn.startsWith(`arn:aws:sts::${ACCOUNT}:assumed-role/`)) {
    throw new Error('Use a temporary role session in the website AWS account.');
  }

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aws-status-report-'));
  const output = path.join(dir, 'response.json');
  const input = path.join(dir, 'request.json');
  try {
    fs.writeFileSync(input, JSON.stringify(payload), { flag: 'wx' });
    const result = JSON.parse(invokeCli([
      'lambda', 'invoke', '--function-name', `arn:aws:lambda:${REGION}:${identity.Account}:function:${REPORT_FUNCTION_NAME}`,
      '--payload', `fileb://${input}`, ...flags, output
    ], env));
    if (result.StatusCode !== 200 || result.FunctionError) throw new Error('The status reporter rejected the upload.');
    const receipt = JSON.parse(fs.readFileSync(output, 'utf8'));
    if (receipt.ok !== true || receipt.accepted !== payload.checks.length) {
      throw new Error('The status reporter did not acknowledge all checks.');
    }
    console.log(`Published ${receipt.accepted} check results.`);
  } finally {
    for (const file of [input, output]) {
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

module.exports = { prepareReport };
