'use strict';

const { InvokeCommand, LambdaClient } = require('@aws-sdk/client-lambda');
const { resolveAwsCredentials } = require('../aws-credentials');
const { isToolsAdminClaims } = require('../short-links');
const { sendJson } = require('../tools-api');
const { assertSameOriginRequest, authenticateToolsRequest } = require('../tools-auth-session');

const { loadStatusCatalog } = require('../aws-status-catalog');
const { sanitizeSnapshot } = require('../aws-status-snapshot');

const REGION = 'us-east-2';
const CACHE_MS = 60_000;
const MAX_PAYLOAD_BYTES = 384 * 1024;

async function readAwsStatus() {
  const functionArn = String(process.env.AWS_STATUS_FUNCTION_ARN || '').trim();
  const roleArn = String(process.env.AWS_STATUS_AWS_ROLE_ARN || '').trim();
  const fn = /^arn:aws:lambda:us-east-2:(\d{12}):function:website-aws-status$/.exec(functionArn);
  if (process.env.VERCEL_ENV !== 'production' || !fn ||
    roleArn !== `arn:aws:iam::${fn[1]}:role/website-aws-status-vercel`) {
    throw new Error('AWS status is unconfigured.');
  }

  const auth = resolveAwsCredentials({
    service: 'aws-status',
    region: REGION,
    roleArnEnvKeys: ['AWS_STATUS_AWS_ROLE_ARN']
  });
  if (auth.source !== 'oidc') throw new Error('Temporary AWS status access is required.');

  const client = new LambdaClient({ region: REGION, credentials: auth.credentials, maxAttempts: 1 });
  try {
    const response = await client.send(new InvokeCommand({
      FunctionName: functionArn,
      InvocationType: 'RequestResponse',
      Payload: Buffer.from('{}')
    }), { abortSignal: AbortSignal.timeout(15_000) });
    if (response.StatusCode !== 200 || response.FunctionError ||
      !response.Payload?.length || response.Payload.length > MAX_PAYLOAD_BYTES) {
      throw new Error('AWS status collector returned an invalid response.');
    }
    return JSON.parse(Buffer.from(response.Payload).toString('utf8'));
  } finally {
    client.destroy();
  }
}

function createAwsStatusHandler({ readStatus = readAwsStatus } = {}) {
  let cached;
  let pending;
  let catalog;

  return async (req, res) => {
    if (req.method !== 'GET') {
      res.setHeader('Allow', 'GET');
      return sendJson(res, 405, { ok: false, error: 'Method not allowed.' });
    }
    const fetchSite = String(req.headers?.['sec-fetch-site'] || '');
    if (fetchSite && fetchSite !== 'same-origin' && fetchSite !== 'none') {
      return sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
    }

    let claims;
    try {
      assertSameOriginRequest(req, { checkReadOrigin: true });
      ({ claims } = await authenticateToolsRequest(req));
    } catch (error) {
      if (error.code === 'AUTH_ORIGIN_MISMATCH') {
        return sendJson(res, 403, { ok: false, error: 'Same-origin request required.' });
      }
      return sendJson(res, 401, { ok: false, error: 'Sign in to view AWS status.' });
    }
    if (!claims?.sub || !isToolsAdminClaims(claims)) {
      return sendJson(res, 403, { ok: false, error: 'Administrator access required.' });
    }

    try {
      if (cached && Date.now() < cached.expiresAt) {
        return sendJson(res, 200, { ...cached.snapshot, cached: true });
      }
      pending ||= Promise.resolve().then(readStatus).then((snapshot) => {
        catalog ||= loadStatusCatalog();
        return sanitizeSnapshot(snapshot, catalog);
      }).then((snapshot) => {
        cached = { snapshot, expiresAt: Date.now() + CACHE_MS };
        return snapshot;
      }).finally(() => { pending = undefined; });
      const snapshot = await pending;
      return sendJson(res, 200, { ...snapshot, cached: false });
    } catch {
      return sendJson(res, 503, { ok: false, error: 'AWS status is temporarily unavailable.' });
    }
  };
}

module.exports = createAwsStatusHandler();
module.exports.createAwsStatusHandler = createAwsStatusHandler;
