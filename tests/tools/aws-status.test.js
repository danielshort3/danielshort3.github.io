'use strict';

const assert = require('node:assert/strict');
const { randomBytes } = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');
const { createAwsStatusHandler } = require('../../api/_lib/tools-endpoints/aws-status');
const { loadStatusCatalog } = require('../../api/_lib/aws-status-catalog');
const { sanitizeSnapshot } = require('../../api/_lib/aws-status-snapshot');
const { COOKIE_NAME, createSessionFromClaims } = require('../../api/_lib/tools-auth-session');
const { main: publishReport, prepareReport } = require('../../build/publish-aws-status-report.cjs');

const CHECKED_AT = '2026-11-11T23:00:00.000Z';
const SNAPSHOT = {
  ok: true,
  checkedAt: CHECKED_AT,
  region: 'us-east-2',
  days: 90,
  connections: [{
    id: 'tracker',
    days: [{
      date: '2026-11-11', state: 'degraded', from: '2026-11-11T22:05:00Z', to: '2026-11-11T22:09:00Z',
      issue: 'Tracker errors exceeded the threshold.', alarmActions: ['private-sns-topic']
    }],
    hours: [{
      start: '2026-11-11T22:00:00.000Z', state: 'degraded', from: '2026-11-11T22:05:00Z',
      to: '2026-11-11T22:09:00Z', observedAt: '2026-11-11T22:10:00Z',
      issue: 'Tracker errors exceeded the threshold.', alarmActions: ['private-sns-topic']
    }]
  }],
  checks: [
    { id: 'tool:text-compare', state: 'operational', source: 'browser', lastChecked: CHECKED_AT, private: 'private-input' },
    { id: 'project:smartSentence', state: 'degraded', source: 'production', lastChecked: '2026-11-09T10:00:00Z' },
    { id: 'game:roulette', state: 'outage', source: 'local', lastChecked: '2026-11-11T22:00:00Z' },
    { id: 'tool:transcribe', state: 'degraded', source: 'production', lastChecked: '2026-11-11T21:00:00Z' },
    { id: 'tool:transcribe', state: 'operational', source: 'browser', lastChecked: CHECKED_AT },
    { id: 'tool:unknown-tool', state: 'operational', source: 'local', lastChecked: CHECKED_AT }
  ]
};

function createRequestHandler(t, options) {
  t.mock.method(Date, 'now', () => Date.parse(CHECKED_AT));
  const secret = process.env.TOOLS_SESSION_SECRETS;
  process.env.TOOLS_SESSION_SECRETS = randomBytes(32).toString('base64url');
  t.after(() => {
    if (secret === undefined) delete process.env.TOOLS_SESSION_SECRETS;
    else process.env.TOOLS_SESSION_SECRETS = secret;
  });
  const handler = createAwsStatusHandler(options);
  return async ({ claims, method = 'GET', headers = {} } = {}) => {
    const responseHeaders = {};
    let body;
    const response = {
      setHeader: (name, value) => { responseHeaders[name] = value; },
      end: (value) => { body = JSON.parse(value); }
    };
    if (claims) {
      headers.cookie = `${COOKIE_NAME}=${createSessionFromClaims(claims).value}`;
    }
    await handler({ method, headers }, response);
    return { status: response.statusCode, headers: responseHeaders, body };
  };
}

const adminClaims = { sub: 'admin-test', 'cognito:groups': ['admin'] };

test('only a verified admin can cause an AWS read, even after the cache is populated', async (t) => {
  let reads = 0;
  const request = createRequestHandler(t, { readStatus: () => { reads++; return SNAPSHOT; } });
  assert.equal((await request()).status, 401);
  assert.equal((await request({ claims: { sub: 'ordinary-test' } })).status, 403);
  assert.equal((await request({ claims: adminClaims, method: 'POST' })).status, 405);
  assert.equal((await request({ claims: adminClaims, headers: { 'sec-fetch-site': 'cross-site' } })).status, 403);
  assert.equal((await request({ claims: adminClaims, headers: { origin: 'https://foreign.example', host: 'localhost' } })).status, 403);
  assert.equal(reads, 0);

  const [first, second] = await Promise.all([
    request({ claims: adminClaims }), request({ claims: adminClaims })
  ]);
  assert.equal(reads, 1);
  assert.equal(first.status, 200);
  assert.deepEqual(first, second);
  assert.equal(first.headers['Cache-Control'], 'no-store');
  assert.equal(first.body.days, 90);
  assert.equal(first.body.connections.length, 7);
  assert.equal(first.body.connections.find((item) => item.id === 'tracker').days[0].state, 'degraded');
  assert.equal(first.body.connections.find((item) => item.id === 'tracker').hours[0].state, 'degraded');
  assert.equal(first.body.connections.find((item) => item.id === 'contact').days.length, 0);
  const expectedCatalog = loadStatusCatalog();
  assert.deepEqual(first.body.catalog.map(({ id, label, group }) => ({ id, label, group })), expectedCatalog);
  assert(expectedCatalog.some((item) => item.group === 'Tools'));
  assert(expectedCatalog.some((item) => item.group === 'Projects'));
  assert(expectedCatalog.some((item) => item.group === 'Games'));
  assert.equal(first.body.catalog.find((item) => item.id === 'tool:text-compare').source, 'browser');
  assert.equal(first.body.catalog.find((item) => item.id === 'project:smartSentence').state, 'unknown');
  assert.equal(first.body.catalog.find((item) => item.id === 'game:roulette').state, 'outage');
  assert.equal(first.body.catalog.find((item) => item.id === 'tool:transcribe').state, 'degraded');
  assert(!first.body.catalog.some((item) => item.id === 'tool:unknown-tool'));
  assert(!JSON.stringify(first.body).includes('private-'));
  assert.equal((await request({ claims: { sub: 'ordinary-test' } })).status, 403);
  assert.equal((await request({ claims: adminClaims })).body.cached, true);
  assert.equal(reads, 1);
});

test('hourly history keeps valid observations ordered and handles older reader responses', () => {
  const earlier = {
    start: '2026-11-11T21:00:00.000Z', state: 'unknown', from: '2026-11-11T21:00:00Z',
    to: '2026-11-11T22:00:00Z', observedAt: '2026-11-11T22:00:00Z', issue: ' No check recorded. '
  };
  const { hours: [incident], ...daily } = SNAPSHOT.connections[0];
  const sanitize = (connection) => sanitizeSnapshot({ ...SNAPSHOT, connections: [connection] }, [])
    .connections.find(({ id }) => id === 'tracker');
  const output = sanitize({ ...daily, hours: [incident, earlier, {
    ...incident, start: '2026-11-11T22:30:00.000Z'
  }, {
    ...incident, start: '2026-11-11T20:00:00.000Z', from: '2026-11-11T20:55:00Z', to: '2026-11-11T21:05:00Z'
  }, { ...incident, state: 'unknown' }, null] });
  assert.deepEqual(output.hours, [
    { start: earlier.start, state: earlier.state, from: earlier.from, to: earlier.to, issue: 'No check recorded.' },
    { start: incident.start, state: incident.state, from: incident.from, to: incident.to, issue: incident.issue,
      observedAt: incident.observedAt }
  ]);
  assert.equal(output.days[0].state, 'degraded');
  assert.equal(Object.hasOwn(sanitize(daily), 'hours'), false);
  assert.equal(Object.hasOwn(sanitize({ ...daily, hours: Array(73).fill(incident) }), 'hours'), false);
});

test('publishing strips fixture content and rejects forged or duplicate catalog entries', () => {
  const report = { schemaVersion: 1, environment: 'local', checkedAt: CHECKED_AT,
    checks: [{ id: 'tool:text-compare', status: 'pass', method: 'browser', reason: 'private fixture text' }] };
  const payload = prepareReport(report, Date.parse(CHECKED_AT));
  assert.deepEqual(payload, { checks: [{
    id: 'tool:text-compare', state: 'operational', lastChecked: CHECKED_AT,
    source: 'browser', expiresAt: '2026-11-18T23:00:00.000Z'
  }] });
  assert.throws(() => prepareReport({ ...report, checks: [...report.checks, report.checks[0]] }, Date.parse(CHECKED_AT)), /duplicate/);
  assert.throws(() => prepareReport({ ...report, checks: [{ id: 'tool:unknown-tool', status: 'pass' }] }, Date.parse(CHECKED_AT)), /unrecognized/);
});

function browserReport() {
  return {
    schemaVersion: 1, environment: 'local', checkedAt: CHECKED_AT,
    checks: loadStatusCatalog().map(({ id }, index) => ({
      id, status: ['pass', 'fail', 'unknown'][index % 3], method: 'browser', reason: 'private fixture'
    }))
  };
}

test('GitHub publishing validates full browser coverage and limits report freshness', () => {
  const now = Date.parse(CHECKED_AT);
  const report = browserReport();
  const options = { githubOidc: true };
  const payload = prepareReport(report, now, options);
  assert.equal(payload.checks.length, loadStatusCatalog().length);
  assert.deepEqual(payload.checks[0], {
    id: report.checks[0].id, state: 'operational', lastChecked: CHECKED_AT,
    source: 'browser', expiresAt: '2026-11-12T02:00:00.000Z'
  });
  assert.deepEqual(payload.checks.slice(1, 3).map(({ state }) => state), ['degraded', 'unknown']);
  assert(!JSON.stringify(payload).includes('private fixture'));
  assert.throws(() => prepareReport({ ...report, checks: report.checks.slice(1) }, now, options), /entire catalog/);
  for (const method of ['local', 'production', undefined]) {
    const checks = [{ ...report.checks[0], method }, ...report.checks.slice(1)];
    assert.throws(() => prepareReport({ ...report, checks }, now, options), /browser results/);
  }
  assert.throws(() => prepareReport(report, now + 30 * 60_000 + 1, options), /recent/);
  assert.throws(() => prepareReport(report, now - 1, options), /recent/);
  assert.throws(() => prepareReport({ ...report, checkedAt: CHECKED_AT.replace('.000', '') }, now, options), /UTC/);
  assert.equal(prepareReport(report, now + 30 * 60_000, options).checks.length, payload.checks.length);
});

test('GitHub publishing reads the deployed checkout catalog from JSON', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aws-status-catalog-test-'));
  const content = path.join(root, 'content');
  const directories = ['tools', 'projects', 'pages'].map((name) => path.join(content, name));
  for (const directory of directories) fs.mkdirSync(directory, { recursive: true });
  const games = path.join(directories[2], 'games.json');
  fs.writeFileSync(games, JSON.stringify({ games: [{ id: 'deployed-game', title: 'Deployed game' }] }));
  t.after(() => {
    fs.unlinkSync(games);
    for (const directory of [...directories, content, root]) fs.rmdirSync(directory);
  });
  const report = {
    ...browserReport(), checks: [{ id: 'game:deployed-game', status: 'pass', method: 'browser' }]
  };
  const now = Date.parse(CHECKED_AT);
  assert.equal(prepareReport(report, now, { githubOidc: true, catalogRoot: root }).checks[0].id, 'game:deployed-game');
  assert.throws(() => prepareReport(report, now, { githubOidc: true }), /unrecognized/);
});

test('publisher uses the exact GitHub role and alias and preserves the manual profile', (t) => {
  const now = Date.parse(CHECKED_AT);
  t.mock.method(Date, 'now', () => now);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'aws-status-publisher-test-'));
  const file = path.join(dir, 'report.json');
  const report = browserReport();
  fs.writeFileSync(file, JSON.stringify(report));
  t.after(() => { fs.unlinkSync(file); fs.rmdirSync(dir); });

  const expirationKey = 'AWS_STATUS_GITHUB_CREDENTIAL_EXPIRATION';
  const testEnv = {
    AWS_ACCESS_KEY_ID: 'ASIAEXAMPLE',
    AWS_SECRET_ACCESS_KEY: 'example-secret',
    AWS_SESSION_TOKEN: 'example-session',
    AWS_PROFILE: 'other-profile',
    AWS_ENDPOINT_URL: 'https://example.invalid',
    AWS_STATUS_AWS_CLI: 'example-cli',
    [expirationKey]: new Date(now + 15 * 60_000).toISOString()
  };
  const before = Object.fromEntries(Object.keys(testEnv).map((key) => [key, process.env[key]]));
  Object.assign(process.env, testEnv);
  t.after(() => {
    for (const [key, value] of Object.entries(before)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  const account = '886623862678';
  const githubRole = `arn:aws:sts::${account}:assumed-role/website-aws-status-report-github/GitHubActions`;
  const functionArn = `arn:aws:lambda:us-east-2:${account}:function:website-aws-status-report`;
  let caller = { Account: account, Arn: githubRole };
  let mode = 'github';
  let invocations = 0;
  function runCli(args, env, executable) {
    if (mode === 'github') {
      assert.equal(executable, 'aws');
      assert(!args.includes('--profile'));
      assert.equal(env.AWS_SESSION_TOKEN, testEnv.AWS_SESSION_TOKEN);
      assert.equal(env.AWS_ACCESS_KEY_ID, testEnv.AWS_ACCESS_KEY_ID);
      assert.equal(env.AWS_SECRET_ACCESS_KEY, testEnv.AWS_SECRET_ACCESS_KEY);
      assert.equal(env.AWS_PROFILE, undefined);
      assert.equal(env.AWS_ENDPOINT_URL, undefined);
      assert.equal(env.AWS_EC2_METADATA_DISABLED, 'true');
      assert.equal(fs.readFileSync(env.AWS_CONFIG_FILE, 'utf8'), '');
      assert.equal(env.AWS_SHARED_CREDENTIALS_FILE, env.AWS_CONFIG_FILE);
    } else {
      assert.equal(executable, 'example-cli');
      assert.equal(args[args.indexOf('--profile') + 1], 'explicit-sso');
      assert.equal(env.AWS_ACCESS_KEY_ID, undefined);
      assert.equal(env.AWS_SECRET_ACCESS_KEY, undefined);
      assert.equal(env.AWS_SESSION_TOKEN, undefined);
    }
    assert.equal(args[args.indexOf('--region') + 1], 'us-east-2');
    if (args[0] === 'sts') return JSON.stringify(caller);
    assert.deepEqual(args.slice(0, 2), ['lambda', 'invoke']);
    assert.equal(args[args.indexOf('--function-name') + 1],
      mode === 'github' ? `${functionArn}:browser-ci` : functionArn);
    const payloadFile = args[args.indexOf('--payload') + 1].slice('fileb://'.length);
    const payload = JSON.parse(fs.readFileSync(payloadFile, 'utf8'));
    assert.deepEqual(payload, prepareReport(report, now, { githubOidc: mode === 'github' }));
    invocations++;
    fs.writeFileSync(args.at(-1), JSON.stringify({ ok: true, accepted: payload.checks.length }));
    return JSON.stringify({ StatusCode: 200 });
  }

  const githubArgs = [file, '--github-oidc'];
  process.env[expirationKey] = new Date(now + 15 * 60_000 + 1).toISOString();
  assert.throws(() => publishReport(githubArgs, runCli), /15-minute/);
  process.env[expirationKey] = testEnv[expirationKey];
  delete process.env.AWS_SESSION_TOKEN;
  assert.throws(() => publishReport(githubArgs, runCli), /15-minute/);
  process.env.AWS_SESSION_TOKEN = testEnv.AWS_SESSION_TOKEN;
  caller = { Account: account, Arn: githubRole.replace('-github/', '-github-extra/') };
  assert.throws(() => publishReport(githubArgs, runCli), /reporter role/);
  caller = { Account: '123456789012', Arn: githubRole.replace(account, '123456789012') };
  assert.throws(() => publishReport(githubArgs, runCli), /website AWS account/);
  assert.equal(invocations, 0);
  caller = { Account: account, Arn: githubRole };
  publishReport(githubArgs, runCli);
  mode = 'manual';
  caller = { Account: account, Arn: githubRole.replace('website-aws-status-report-github', 'WebsiteAdmin') };
  publishReport([file, '--profile', 'explicit-sso'], runCli);
  assert.equal(invocations, 2);
});

test('homepage reachability preserves fuller results and still surfaces observed failures', () => {
  const id = 'project:website';
  const catalog = [{ id, label: 'Website', group: 'Projects' }];
  const now = Date.parse(CHECKED_AT);
  const browser = { id, state: 'outage', source: 'browser', lastChecked: CHECKED_AT };
  const head = { id, state: 'operational', source: 'production', method: 'homepage-head', lastChecked: CHECKED_AT };
  const full = { id, state: 'degraded', source: 'production', lastChecked: '2026-11-11T22:00:00Z' };
  const status = (...checks) => sanitizeSnapshot({ ...SNAPSHOT, checks }, catalog, now).catalog[0];

  assert.equal(status(browser, head).state, 'outage');
  assert.equal(status({ ...browser, state: 'operational' }, { ...head, state: 'outage' }).state, 'outage');
  assert.equal(status(full, head).state, 'degraded');
  assert.equal(status(...Array(210).fill(browser), head).state, 'outage');
  assert.throws(() => status(...Array(211).fill(browser), head), /invalid AWS history snapshot/i);
});

test('invalid and failed sources never appear as all-clear status', async (t) => {
  const broken = createRequestHandler(t, {
    readStatus: () => ({ ...SNAPSHOT, checkedAt: null, connections: [{ id: 'tracker', days: [{ state: 'bad' }] }], checks: [] })
  });
  const result = await broken({ claims: adminClaims });
  assert.equal(result.status, 200);
  assert(result.body.connections.every((item) => item.days.length === 0));
  assert(result.body.catalog.every((item) => item.state === 'unknown' && item.lastChecked === null));
  assert.equal(result.body.checkedAt, null);

  const missing = createAwsStatusHandler({ readStatus: () => { throw new Error('private role ARN'); } });
  let body;
  const response = { setHeader() {}, end: (value) => { body = JSON.parse(value); } };
  const cookie = `${COOKIE_NAME}=${createSessionFromClaims(adminClaims).value}`;
  await missing({ method: 'GET', headers: { cookie } }, response);
  assert.equal(response.statusCode, 503);
  assert.equal(body.error, 'AWS status is temporarily unavailable.');
});
