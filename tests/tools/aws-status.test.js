'use strict';

const assert = require('node:assert/strict');
const { randomBytes } = require('node:crypto');
const { test } = require('node:test');
const { createAwsStatusHandler } = require('../../api/_lib/tools-endpoints/aws-status');
const { loadStatusCatalog } = require('../../api/_lib/aws-status-catalog');
const { sanitizeSnapshot } = require('../../api/_lib/aws-status-snapshot');
const { COOKIE_NAME, createSessionFromClaims } = require('../../api/_lib/tools-auth-session');
const { prepareReport } = require('../../build/publish-aws-status-report.cjs');

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
