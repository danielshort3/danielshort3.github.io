'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { verifyProductionRelease, REQUIRED_JOBS } = require('../../build/verify-production-release.cjs');
const sha = 'a'.repeat(40);
const env = { VERCEL_ENV: 'production', VERCEL_GIT_COMMIT_SHA: sha, VERCEL_GIT_COMMIT_REF: 'main' };
const run = { id: 42, head_sha: sha, head_branch: 'main', event: 'push', path: '.github/workflows/ci.yml', status: 'completed', conclusion: 'success' };
const jobs = REQUIRED_JOBS.map(name => ({ name, head_sha: sha, status: 'completed', conclusion: 'success' }));
const ok = body => ({ ok: true, json: async () => body });
const verify = options => verifyProductionRelease({ env, timeoutMs: 0, log() {}, ...options });
test('local and preview builds do not claim approval or query GitHub', async () => {
  for (const environment of [{}, { VERCEL_ENV: 'preview' }]) {
    assert.deepEqual(await verify({ env: environment, fetchImpl() { throw Error('must not fetch'); } }), { skipped: true });
  }
});
test('production metadata is mandatory and identifies main in this repository', async () => {
  for (const invalid of [{ VERCEL_ENV: 'production' }, { ...env, VERCEL_GIT_COMMIT_REF: 'other' },
    { ...env, VERCEL_GIT_REPO_SLUG: 'other' }, { VERCEL: '1' }]) {
    await assert.rejects(verify({ env: invalid }), /requires|metadata|environment/);
  }
});
test('accepts successful main push CI and all exact-commit jobs', async () => {
  const result = await verify({ fetchImpl: async url => ok(url.includes('/jobs?') ? { jobs } : { workflow_runs: [run] }) });
  assert.deepEqual(result, { sha, runId: 42 });
});
test('waits for pending CI instead of treating it as a pass', async () => {
  let time = 0, requests = 0;
  const result = await verify({ timeoutMs: 100, pollMs: 10, now: () => time, sleep: async ms => { time += ms; },
    fetchImpl: async url => ok(url.includes('/jobs?') ? { jobs } : { workflow_runs: [{ ...run, status: ++requests === 1 ? 'in_progress' : 'completed' }] }) });
  assert.equal(result.runId, 42); assert.equal(requests, 2);
});
test('failed, skipped, neutral and cancelled runs block production', async () => {
  for (const conclusion of ['failure', 'cancelled', 'skipped', 'neutral', 'timed_out']) {
    await assert.rejects(verify({ fetchImpl: async () => ok({ workflow_runs: [{ ...run, conclusion }] }) }), /remains blocked/);
  }
});
test('other commits, PR events and workflows cannot approve production', async () => {
  for (const other of [{ ...run, head_sha: 'b'.repeat(40) }, { ...run, event: 'pull_request' }, { ...run, path: 'other.yml' }]) {
    await assert.rejects(verify({ fetchImpl: async () => ok({ workflow_runs: [other] }) }), /Timed out/);
  }
});
test('every required job must be present once, complete, successful and current', async () => {
  for (const name of REQUIRED_JOBS) {
    for (const patch of [null, { conclusion: 'skipped' }, { head_sha: 'b'.repeat(40) }, { status: 'in_progress' }]) {
      const changed = jobs.flatMap(job => job.name !== name ? [job] : patch ? [{ ...job, ...patch }] : []);
      await assert.rejects(verify({ fetchImpl: async url => ok(url.includes('/jobs?') ? { jobs: changed } : { workflow_runs: [run] }) }), /required check/);
    }
  }
});
test('missing evidence, pagination, API and network errors fail closed', async () => {
  await assert.rejects(verify({ fetchImpl: async () => ({ ok: false, status: 403 }) }), /403/);
  await assert.rejects(verify({ fetchImpl: async () => { throw Error('offline'); } }), /offline/);
  await assert.rejects(verify({ fetchImpl: async () => ok({}) }), /Timed out/);
  await assert.rejects(verify({ fetchImpl: async url => ok(url.includes('/jobs?') ? { jobs, total_count: 101 } : { workflow_runs: [run] }) }), /Incomplete/);
});
