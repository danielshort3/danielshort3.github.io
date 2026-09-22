'use strict';
// Read-only approval of the exact production commit, not a previous green build.
const REPOSITORY = 'danielshort3/danielshort3.github.io';
const REQUIRED_JOBS = Object.freeze([
  'dependency-security (.)',
  'dependency-security (aws/job-application-tracker)',
  'dependency-security (browser-extension/job-application-copilot)',
  'build-and-test', 'release-quality', 'mobile-performance', 'release-ready'
]);

async function verifyProductionRelease({ env = process.env, fetchImpl = globalThis.fetch,
  now = Date.now, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)),
  timeoutMs = 20 * 60 * 1000, pollMs = 45000, log = console.log } = {}) {
  if (env.VERCEL_ENV !== 'production' && env.VERCEL_TARGET_ENV !== 'production') {
    if (env.VERCEL && !env.VERCEL_ENV && !env.VERCEL_TARGET_ENV) {
      throw Error('Vercel environment is missing; refusing to infer production approval.');
    }
    log('[release-gate] Local/preview build; no production approval is granted.');
    return { skipped: true };
  }
  const sha = String(env.VERCEL_GIT_COMMIT_SHA || '');
  if (!/^[a-f0-9]{40}$/.test(sha) || env.VERCEL_GIT_COMMIT_REF !== 'main') {
    throw Error('Production requires a full commit SHA and VERCEL_GIT_COMMIT_REF=main.');
  }
  if ((env.VERCEL_GIT_REPO_OWNER && env.VERCEL_GIT_REPO_OWNER !== 'danielshort3') ||
      (env.VERCEL_GIT_REPO_SLUG && env.VERCEL_GIT_REPO_SLUG !== 'danielshort3.github.io')) {
    throw Error('Production repository metadata does not match.');
  }
  async function get(urlPath) {
    const response = await fetchImpl(`https://api.github.com/repos/${REPOSITORY}${urlPath}`, {
      signal: AbortSignal.timeout(10000),
      headers: { Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'danielshort-production-release-gate' }
    });
    if (!response.ok) throw Error(`GitHub verification failed (${response.status}); production remains blocked.`);
    return response.json();
  }
  const deadline = now() + timeoutMs;
  do {
    const data = await get(`/actions/workflows/ci.yml/runs?branch=main&event=push&head_sha=${sha}&per_page=10`);
    const run = (data.workflow_runs || []).filter(item => item.head_sha === sha && item.head_branch === 'main'
      && item.event === 'push' && item.path === '.github/workflows/ci.yml')
      .sort((a, b) => b.id - a.id || b.run_attempt - a.run_attempt)[0];
    if (run?.status === 'completed') {
      if (run.conclusion !== 'success') throw Error(`CI run ${run.id} concluded ${run.conclusion}; production remains blocked.`);
      const report = await get(`/actions/runs/${run.id}/jobs?filter=latest&per_page=100`);
      if (!Array.isArray(report.jobs) || Number(report.total_count) > report.jobs.length) {
        throw Error('Incomplete job evidence; production remains blocked.');
      }
      for (const name of REQUIRED_JOBS) {
        const matches = report.jobs.filter(job => job.name === name);
        if (matches.length !== 1 || matches[0].head_sha !== sha || matches[0].status !== 'completed'
            || matches[0].conclusion !== 'success') throw Error(`Missing, stale, or unsuccessful required check: ${name}`);
      }
      log(`[release-gate] Approved ${sha}; all required checks passed in CI run ${run.id}.`);
      return { sha, runId: run.id };
    }
    if (now() >= deadline) break;
    log(`[release-gate] CI pending for ${sha.slice(0, 12)}; production is not approved.`);
    await sleep(Math.min(pollMs, Math.max(0, deadline - now())));
  } while (now() <= deadline);
  throw Error('Timed out waiting for exact-commit CI. Redeploy only after its required checks pass.');
}
module.exports = { verifyProductionRelease, REQUIRED_JOBS };
if (require.main === module) verifyProductionRelease().catch(error => {
  console.error(`[release-gate] ${error.message}`); process.exitCode = 1;
});
