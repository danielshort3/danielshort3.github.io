'use strict';

const assert = require('node:assert/strict');
const { generateKeyPairSync, verify } = require('node:crypto');
const test = require('node:test');
const { dispatchFallback } = require('./fallback-dispatcher.cjs');

const NOW = Date.parse('2026-11-14T13:47:12Z');
const SLOT = '2026-11-14T13:00:00Z';
const SCHEDULE = 'arn:aws:scheduler:us-east-2:123456789012:schedule/test/fallback';
const WORKFLOW = '/repos/danielshort3/danielshort3.github.io/actions/workflows/416880337';
const REPOSITORY_ID = 698339955;
const TOKEN = 'test-only-installation-token';
const { privateKey, publicKey } = generateKeyPairSync('rsa', { modulusLength: 2048 });
const CREDENTIAL = JSON.stringify({
  clientId: 'Iv1.test-client-id',
  installationId: 123,
  privateKeyPem: privateKey.export({ type: 'pkcs8', format: 'pem' })
});

function reply(status, value) {
  return { status, json: async () => value };
}

function tokenReply(overrides = {}) {
  return reply(201, {
    token: TOKEN,
    expires_at: '2026-11-14T14:47:12Z',
    permissions: { actions: 'write', metadata: 'read' },
    repositories: [{ id: REPOSITORY_ID }],
    ...overrides
  });
}

function run(overrides = {}) {
  return {
    id: 456,
    head_branch: 'main',
    event: 'schedule',
    status: 'completed',
    created_at: '2026-11-14T13:29:00Z',
    updated_at: '2026-11-14T13:33:00Z',
    display_title: 'Projects & Tools status',
    ...overrides
  };
}

function jobs(observed = '2026-11-14T13:32:00Z', publication = 'success') {
  return { jobs: [
    { name: 'check', steps: [{ name: 'Run the catalog browser checks', conclusion: 'success', completed_at: observed }] },
    { name: 'publish / publish', steps: [{ name: 'Publish the browser outcomes', conclusion: publication }] }
  ] };
}

function fixture({ responses = [], event = {}, credential = CREDENTIAL } = {}) {
  const requests = [];
  let reads = 0;
  const execute = () => dispatchFallback({
    event: { scheduleArn: SCHEDULE, scheduledTime: '2026-11-14T13:47:00Z', ...event },
    now: NOW,
    expectedScheduleArn: SCHEDULE,
    fetchRequest: async (url, options) => {
      requests.push({ url: new URL(url), ...options });
      const response = responses[requests.length - 1];
      if (response instanceof Error) throw response;
      return response;
    },
    readCredential: async () => {
      reads += 1;
      return credential;
    }
  });
  return { execute, requests, credentialReads: () => reads };
}

test('requests temporary repository Actions access and dispatches the fixed workflow for an uncovered hour', async () => {
  const { execute, requests } = fixture({ responses: [
    tokenReply(), reply(200, { workflow_runs: [] }), reply(200, { workflow_run_id: 789 })
  ] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'dispatched', runId: 789 });
  assert.deepEqual(requests.map(({ method, url }) => [method, url.origin, url.pathname]), [
    ['POST', 'https://api.github.com', '/app/installations/123/access_tokens'],
    ['GET', 'https://api.github.com', WORKFLOW + '/runs'],
    ['POST', 'https://api.github.com', WORKFLOW + '/dispatches']
  ]);
  for (const { headers, redirect, signal } of requests) {
    assert.equal(headers['X-GitHub-Api-Version'], '2026-03-10');
    assert.equal(redirect, 'error');
    assert.ok(signal instanceof AbortSignal);
  }
  const [header, payload, signature] = requests[0].headers.Authorization.slice('Bearer '.length).split('.');
  assert.deepEqual(JSON.parse(Buffer.from(header, 'base64url')), { alg: 'RS256', typ: 'JWT' });
  assert.deepEqual(JSON.parse(Buffer.from(payload, 'base64url')), {
    iss: 'Iv1.test-client-id', iat: NOW / 1000 - 60, exp: NOW / 1000 + 5 * 60
  });
  assert.ok(verify('RSA-SHA256', Buffer.from(`${header}.${payload}`), publicKey, Buffer.from(signature, 'base64url')));
  assert.deepEqual(JSON.parse(requests[0].body), {
    repository_ids: [REPOSITORY_ID], permissions: { actions: 'write' }
  });
  assert.equal(requests[1].url.searchParams.get('branch'), 'main');
  assert.equal(requests[1].url.searchParams.get('created'), '>=2026-11-14T12:00:00.000Z');
  assert.equal(requests[2].headers.Authorization, 'Bearer ' + TOKEN);
  assert.deepEqual(JSON.parse(requests[2].body), { ref: 'main', inputs: { fallback_slot: SLOT } });
});

test('avoids dispatch after a successful sample and publication even when the workflow concluded failure', async () => {
  const { execute, requests } = fixture({ responses: [
    tokenReply(), reply(200, { workflow_runs: [run({ conclusion: 'failure' })] }), reply(200, jobs())
  ] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'hour-already-published' });
  assert.equal(requests[2].url.pathname, '/repos/danielshort3/danielshort3.github.io/actions/runs/456/jobs');
  assert.equal(requests[2].url.searchParams.get('filter'), 'latest');
  assert.equal(requests.length, 3);
});

test('queues a fallback when a pending native run has yet to publish', async () => {
  const { execute, requests } = fixture({ responses: [
    tokenReply(), reply(200, { workflow_runs: [run({ status: 'queued' })] }),
    reply(200, { workflow_run_id: 789 })
  ] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'dispatched', runId: 789 });
  assert.equal(requests.length, 3);
});

test('deduplicates against a pending fallback for the same slot', async () => {
  const { execute, requests } = fixture({ responses: [tokenReply(), reply(200, { workflow_runs: [run({
    event: 'workflow_dispatch', display_title: 'Hourly status fallback ' + SLOT, status: 'queued'
  })] })] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'fallback-already-pending' });
  assert.equal(requests.length, 2);
});

test('can recover from a completed same-slot fallback that did not publish', async () => {
  const { execute, requests } = fixture({ responses: [tokenReply(), reply(200, { workflow_runs: [run({
    event: 'workflow_dispatch', display_title: 'Hourly status fallback ' + SLOT, conclusion: 'failure'
  })] }), reply(200, jobs(undefined, 'failure')), reply(200, { workflow_run_id: 789 })] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'dispatched', runId: 789 });
  assert.equal(requests.length, 4);
});

test('dispatches when a previous sample was older or publication did not complete', async () => {
  const { execute, requests } = fixture({ responses: [
    tokenReply(), reply(200, { workflow_runs: [
      run({ head_branch: 'feature', status: 'in_progress' }),
      run({ status: 'in_progress', created_at: '2026-11-14T13:06:00Z' }),
      run({ id: 567 }), run({ id: 678 })
    ] }), reply(200, jobs('2026-11-14T12:59:59Z')), reply(200, jobs(undefined, 'skipped')),
    reply(200, { workflow_run_id: 789 })
  ] });
  assert.deepEqual(await execute(), { slot: SLOT, status: 'dispatched', runId: 789 });
  assert.equal(requests.length, 5);
});

test('rejects malformed, wrong-source, or stale events before accessing the credential', async () => {
  for (const event of [
    { scheduleArn: SCHEDULE + '-wrong' }, { scheduledTime: 'broken' },
    { scheduledTime: '2026-11-14T13:46:00Z' }, { scheduledTime: '2026-11-14T12:47:00Z' },
    { scheduledTime: '2026-11-14T14:47:00Z' }
  ]) {
    const { execute, credentialReads, requests } = fixture({ event });
    await assert.rejects(execute(), /scheduler event|intended time window/i);
    assert.equal(credentialReads(), 0);
    assert.equal(requests.length, 0);
  }
});

test('rejects token responses with extra privileges, wrong repository, or unexpected token lifetime', async () => {
  for (const token of [
    tokenReply({ permissions: { actions: 'write', contents: 'write' } }),
    tokenReply({ repositories: [{ id: 1 }] }),
    tokenReply({ repositories: null }),
    tokenReply({ expires_at: '2026-11-14T15:47:12Z' })
  ]) {
    const { execute, requests } = fixture({ responses: [token] });
    await assert.rejects(execute(), { message: 'GitHub installation token was not restricted as expected.' });
    assert.equal(requests.length, 1);
  }
});

test('keeps request errors and response contents out of failure messages', async () => {
  const limit = fixture({ responses: [tokenReply(), reply(429, { message: TOKEN })] });
  await assert.rejects(limit.execute(), { message: 'GitHub run list returned HTTP 429.' });
  const timeout = fixture({ responses: [tokenReply(), new Error(TOKEN)] });
  await assert.rejects(timeout.execute(), { message: 'GitHub run list request did not complete.' });
  const badCredential = fixture({ credential: TOKEN });
  await assert.rejects(badCredential.execute(), { message: 'GitHub App parameter is not valid JSON.' });
});

test('requires a workflow run acknowledgment and fails closed on incomplete run data', async () => {
  const missingAck = fixture({ responses: [tokenReply(), reply(200, { workflow_runs: [] }), reply(200, {})] });
  await assert.rejects(missingAck.execute(), { message: 'GitHub dispatch did not acknowledge a workflow run.' });
  const missingList = fixture({ responses: [tokenReply(), reply(200, {})] });
  await assert.rejects(missingList.execute(), { message: 'GitHub run list was incomplete.' });
  assert.equal(missingList.requests.length, 2);
});
