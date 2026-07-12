/*
  Admin health endpoint for short links storage: /api/short-links/health
  - Requires SHORTLINKS_ADMIN_TOKEN (same as the dashboard).
  - Returns sanitized DynamoDB connectivity diagnostics.
*/
'use strict';

const {
  DynamoDBClient,
  DescribeTableCommand,
  DescribeTimeToLiveCommand
} = require('@aws-sdk/client-dynamodb');
const {
  CLICK_RETENTION_DAYS,
  CLICK_TTL_ATTRIBUTE,
  getAwsCredentialConfig,
  getAwsCredentialEnvInfo,
  getRequiredEnv
} = require('../_lib/short-links-store');
const { getAdminToken, isAdminRequest, sendJson } = require('../_lib/short-links');

function maskAccessKeyId(value){
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw) return '';
  if (raw.length <= 8) return `${raw.slice(0, 2)}…${raw.slice(-2)}`;
  return `${raw.slice(0, 4)}…${raw.slice(-4)}`;
}

module.exports = async (req, res) => {
  const adminToken = getAdminToken();
  if (!adminToken) {
    sendJson(res, 503, { ok: false, error: 'SHORTLINKS_ADMIN_TOKEN is not configured' });
    return;
  }
  if (!isAdminRequest(req)) {
    sendJson(res, 401, { ok: false, error: 'Unauthorized' });
    return;
  }

  if (req.method !== 'GET') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET');
    sendJson(res, 405, { ok: false, error: 'Method Not Allowed' });
    return;
  }

  let env;
  try {
    env = getRequiredEnv();
  } catch (err) {
    sendJson(res, 503, { ok: false, error: err && err.message ? err.message : 'Short links backend misconfigured' });
    return;
  }

  let auth;
  try {
    auth = getAwsCredentialConfig(env.region);
  } catch (err) {
    sendJson(res, 503, {
      ok: false,
      error: err && err.message ? err.message : 'AWS credentials are not configured'
    });
    return;
  }

  const envInfo = getAwsCredentialEnvInfo();
  const staticCredentialsInUse = auth.source === 'static';
  const creds = {
    authMode: auth.authMode,
    credentialSource: auth.source,
    roleArnConfigured: auth.roleArnConfigured,
    roleArnSource: auth.roleArnSource,
    oidcAudienceConfigured: Boolean(auth.audience),
    accessKeyId: staticCredentialsInUse ? maskAccessKeyId(envInfo.accessKeyId) : '',
    accessKeyIdSource: staticCredentialsInUse ? envInfo.accessKeyIdSource : '',
    sessionTokenConfigured: staticCredentialsInUse && envInfo.sessionTokenConfigured,
    sessionTokenUsed: staticCredentialsInUse && envInfo.sessionTokenUsed,
    sessionTokenIgnored: staticCredentialsInUse && envInfo.sessionTokenIgnored,
    accessKeyTrimmed: staticCredentialsInUse && envInfo.accessKeyTrimmed,
    secretTrimmed: staticCredentialsInUse && envInfo.secretTrimmed,
    sessionTokenTrimmed: staticCredentialsInUse && envInfo.sessionTokenTrimmed
  };
  const client = new DynamoDBClient({ region: env.region, credentials: auth.credentials });

  try {
    const result = await client.send(new DescribeTableCommand({ TableName: env.tableName }));
    const table = result && result.Table ? result.Table : null;
    const clicksTableName = process.env.SHORTLINKS_DDB_CLICKS_TABLE
      ? String(process.env.SHORTLINKS_DDB_CLICKS_TABLE).trim()
      : '';

    let clicksTable = null;
    let clicksError = null;
    let clicksTtl = null;
    let clicksTtlError = null;
    if (clicksTableName) {
      try {
        const clicksResult = await client.send(new DescribeTableCommand({ TableName: clicksTableName }));
        clicksTable = clicksResult && clicksResult.Table ? clicksResult.Table : null;
      } catch (err) {
        clicksError = {
          name: err && err.name ? err.name : '',
          code: err && err.code ? err.code : '',
          message: err && err.message ? err.message : ''
        };
      }
      try {
        const ttlResult = await client.send(new DescribeTimeToLiveCommand({ TableName: clicksTableName }));
        clicksTtl = ttlResult && ttlResult.TimeToLiveDescription ? ttlResult.TimeToLiveDescription : null;
      } catch (err) {
        clicksTtlError = {
          name: err && err.name ? err.name : '',
          code: err && err.code ? err.code : '',
          message: err && err.message ? err.message : ''
        };
      }
    }

    const ttlStatus = clicksTtl && clicksTtl.TimeToLiveStatus ? clicksTtl.TimeToLiveStatus : '';
    const ttlAttribute = clicksTtl && clicksTtl.AttributeName ? clicksTtl.AttributeName : '';
    const ttlConfigured = ttlStatus === 'ENABLED' && ttlAttribute === CLICK_TTL_ATTRIBUTE;

    sendJson(res, 200, {
      ok: true,
      aws: {
        region: env.region,
        ...creds
      },
      table: {
        name: table && table.TableName ? table.TableName : env.tableName,
        status: table && table.TableStatus ? table.TableStatus : '',
        billingMode: table && table.BillingModeSummary ? table.BillingModeSummary.BillingMode : ''
      },
      clicks: {
        configured: !!clicksTableName,
        retention: {
          days: CLICK_RETENTION_DAYS,
          ttlAttribute: CLICK_TTL_ATTRIBUTE,
          ttlConfigured,
          ttlStatus,
          configuredAttribute: ttlAttribute,
          note: ttlConfigured
            ? `Click events expire after ${CLICK_RETENTION_DAYS} days.`
            : `Enable DynamoDB TTL on ${CLICK_TTL_ATTRIBUTE} to enforce ${CLICK_RETENTION_DAYS}-day retention.`
        },
        table: clicksTableName ? {
          name: clicksTable && clicksTable.TableName ? clicksTable.TableName : clicksTableName,
          status: clicksTable && clicksTable.TableStatus ? clicksTable.TableStatus : '',
          billingMode: clicksTable && clicksTable.BillingModeSummary ? clicksTable.BillingModeSummary.BillingMode : ''
        } : null,
        error: clicksError,
        ttlError: clicksTtlError
      }
    });
  } catch (err) {
    sendJson(res, 502, {
      ok: false,
      error: 'DynamoDB backend unavailable',
      details: {
        name: err && err.name ? err.name : '',
        code: err && err.code ? err.code : '',
        message: err && err.message ? err.message : ''
      },
      aws: {
        region: env.region,
        table: env.tableName,
        ...creds
      }
    });
  }
};
