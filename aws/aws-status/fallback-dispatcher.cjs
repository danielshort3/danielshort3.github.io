'use strict';

const { sign } = require('node:crypto');

const REPOSITORY_ID = 698339955;
const WORKFLOW_ID = 416880337;
const API_ROOT = 'https://api.github.com';
const ACTIONS_PATH = '/repos/danielshort3/danielshort3.github.io/actions';
const WORKFLOW_PATH = `${ACTIONS_PATH}/workflows/${WORKFLOW_ID}`;
const MINUTE = 60_000;
const MAX_SCHEDULE_LATENESS = 12 * MINUTE;
const GITHUB_TIMEOUT = 5_000;
const PENDING_STATES = new Set(['requested', 'waiting', 'pending', 'queued', 'in_progress']);
const ISO_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/;

class DispatchError extends Error {}

function utcHour(time) {
  return new Date(time).toISOString().slice(0, 13) + ':00:00Z';
}

function fallbackTitle(slot) {
  return 'Hourly status fallback ' + slot;
}

function getSlot(event, now, expectedScheduleArn) {
  const scheduled = event?.scheduledTime;
  if (!expectedScheduleArn || event?.scheduleArn !== expectedScheduleArn || !ISO_TIMESTAMP.test(scheduled)) {
    throw new DispatchError('Unexpected scheduler event.');
  }
  const time = Date.parse(scheduled);
  if (!Number.isFinite(time) || utcHour(time).slice(0, 13) !== scheduled.slice(0, 13) ||
      new Date(time).getUTCMinutes() !== 47 || time > now + MINUTE || now - time > MAX_SCHEDULE_LATENESS) {
    throw new DispatchError('Scheduler event is outside its intended time window.');
  }
  return utcHour(time);
}

function makeAppJwt(value, now) {
  let app;
  try {
    app = JSON.parse(value);
  } catch {
    throw new DispatchError('GitHub App parameter is not valid JSON.');
  }
  if (!app || typeof app.clientId !== 'string' || !/^[A-Za-z0-9._-]{8,120}$/.test(app.clientId) ||
      !Number.isSafeInteger(app.installationId) || app.installationId < 1 ||
      typeof app.privateKeyPem !== 'string' || !/^-----BEGIN (?:RSA )?PRIVATE KEY-----\r?\n/.test(app.privateKeyPem)) {
    throw new DispatchError('GitHub App parameter has invalid fields.');
  }
  const encode = (data) => Buffer.from(JSON.stringify(data)).toString('base64url');
  const seconds = Math.floor(now / 1000);
  const unsigned = [
    encode({ alg: 'RS256', typ: 'JWT' }),
    encode({ iss: app.clientId, iat: seconds - 60, exp: seconds + 5 * 60 })
  ].join('.');
  try {
    const signature = sign('RSA-SHA256', Buffer.from(unsigned), app.privateKeyPem).toString('base64url');
    return { jwt: unsigned + '.' + signature, installationId: app.installationId };
  } catch {
    throw new DispatchError('GitHub App key could not sign a request.');
  }
}

async function github(fetchRequest, token, path, { method = 'GET', expected = 200, body, label }) {
  let response;
  try {
    response = await fetchRequest(API_ROOT + path, {
      method,
      headers: {
        Accept: 'application/vnd.github+json',
        Authorization: 'Bearer ' + token,
        'Content-Type': 'application/json',
        'User-Agent': 'website-status-hourly-fallback',
        'X-GitHub-Api-Version': '2026-03-10'
      },
      redirect: 'error',
      signal: AbortSignal.timeout(GITHUB_TIMEOUT),
      body: body === undefined ? undefined : JSON.stringify(body)
    });
  } catch {
    throw new DispatchError('GitHub ' + label + ' request did not complete.');
  }
  if (response.status !== expected) {
    throw new DispatchError('GitHub ' + label + ' returned HTTP ' + response.status + '.');
  }
  try {
    return await response.json();
  } catch {
    throw new DispatchError('GitHub ' + label + ' returned invalid JSON.');
  }
}

async function getInstallationToken(fetchRequest, { jwt, installationId }, now) {
  const receipt = await github(fetchRequest, jwt, `/app/installations/${installationId}/access_tokens`, {
    method: 'POST', expected: 201, label: 'installation token',
    body: { repository_ids: [REPOSITORY_ID], permissions: { actions: 'write' } }
  });
  const expires = Date.parse(receipt?.expires_at);
  if (typeof receipt?.token !== 'string' || !receipt.token || !Number.isFinite(expires) ||
      expires <= now || expires > now + 61 * MINUTE ||
      receipt.permissions?.actions !== 'write' ||
      Object.entries(receipt.permissions).some(([key, value]) =>
        key !== 'actions' && (key !== 'metadata' || value !== 'read')) ||
      !Array.isArray(receipt.repositories) || receipt.repositories.length !== 1 ||
      receipt.repositories[0]?.id !== REPOSITORY_ID) {
    throw new DispatchError('GitHub installation token was not restricted as expected.');
  }
  return receipt.token;
}

function coveredByJobs(jobs, slot) {
  const stepSucceeded = (job, step, condition = () => true) =>
    job?.steps?.some((entry) => entry.name === step && entry.conclusion === 'success' && condition(entry));
  return jobs.some((job) => job.name === 'check' &&
      stepSucceeded(job, 'Run the catalog browser checks', ({ completed_at }) => {
        const completed = Date.parse(completed_at);
        return ISO_TIMESTAMP.test(completed_at) && Number.isFinite(completed) && utcHour(completed) === slot;
      })) &&
    jobs.some((job) => job.name === 'publish / publish' && stepSucceeded(job, 'Publish the browser outcomes'));
}

async function priorRun(fetchRequest, token, slot) {
  const since = new Date(Date.parse(slot) - MINUTE * 60).toISOString();
  const query = new URLSearchParams({ branch: 'main', created: '>=' + since, per_page: '20' });
  const listing = await github(fetchRequest, token, WORKFLOW_PATH + '/runs?' + query, { label: 'run list' });
  if (!Array.isArray(listing?.workflow_runs)) throw new DispatchError('GitHub run list was incomplete.');
  const runs = listing.workflow_runs.filter((run) => Number.isSafeInteger(run?.id) && run.id > 0 &&
    run.head_branch === 'main' && ['schedule', 'workflow_dispatch'].includes(run.event));
  // A queued native run might never publish. The workflow serializes the backup with any native run.
  const pendingFallback = runs.some((run) => PENDING_STATES.has(run.status) &&
    run.event === 'workflow_dispatch' && run.display_title === fallbackTitle(slot));
  if (pendingFallback) return 'fallback-already-pending';
  // Only inspect completed runs whose observation could belong to this hour.
  const candidates = runs.filter((run) => run.status === 'completed' && Date.parse(run.updated_at) >= Date.parse(slot));
  const jobsByRun = await Promise.all(candidates.map(async (run) => {
    const result = await github(fetchRequest, token, `${ACTIONS_PATH}/runs/${run.id}/jobs?filter=latest&per_page=20`, {
      label: 'run jobs'
    });
    if (!Array.isArray(result?.jobs)) throw new DispatchError('GitHub run jobs were incomplete.');
    return result.jobs;
  }));
  return jobsByRun.some((jobs) => coveredByJobs(jobs, slot)) ? 'hour-already-published' : null;
}

async function dispatchFallback({ event, now, expectedScheduleArn, fetchRequest, readCredential }) {
  const slot = getSlot(event, now, expectedScheduleArn);
  const credential = await readCredential();
  const token = await getInstallationToken(fetchRequest, makeAppJwt(credential, now), now);
  const existing = await priorRun(fetchRequest, token, slot);
  if (existing) return { slot, status: existing };

  const receipt = await github(fetchRequest, token, WORKFLOW_PATH + '/dispatches', {
    method: 'POST', label: 'workflow dispatch',
    body: { ref: 'main', inputs: { fallback_slot: slot } }
  });
  if (!Number.isSafeInteger(receipt?.workflow_run_id) || receipt.workflow_run_id < 1) {
    throw new DispatchError('GitHub dispatch did not acknowledge a workflow run.');
  }
  return { slot, status: 'dispatched', runId: receipt.workflow_run_id };
}

async function handler(event) {
  try {
    const result = await dispatchFallback({
      event,
      now: Date.now(),
      expectedScheduleArn: process.env.STATUS_FALLBACK_SCHEDULE_ARN,
      fetchRequest: fetch,
      readCredential: async () => {
        const { SSMClient, GetParameterCommand } = require('@aws-sdk/client-ssm');
        const client = new SSMClient({ maxAttempts: 2 });
        const result = await client.send(new GetParameterCommand({
          Name: process.env.STATUS_FALLBACK_GITHUB_PARAMETER,
          WithDecryption: true
        }), { abortSignal: AbortSignal.timeout(GITHUB_TIMEOUT) });
        if (result.Parameter?.Type !== 'SecureString' || !result.Parameter.Value) {
          throw new DispatchError('GitHub App requires a SecureString parameter.');
        }
        return result.Parameter.Value;
      }
    });
    console.log(JSON.stringify(result));
    return result;
  } catch (error) {
    throw new Error(error instanceof DispatchError ? error.message : 'Hourly status fallback could not complete.');
  }
}

module.exports = { handler, dispatchFallback };
