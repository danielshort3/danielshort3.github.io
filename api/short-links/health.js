/*
  Admin health endpoint for short links storage: /api/short-links/health
  - Requires a verified Cognito admins-group token (legacy token only during rollback mode).
  - Returns sanitized DynamoDB connectivity diagnostics.
*/
'use strict';

const { DynamoDBClient, DescribeTableCommand } = require('@aws-sdk/client-dynamodb');
const {
  AWS_WORKLOADS,
  describeAwsAuth,
  getAwsClientConfig
} = require('../_lib/aws-credentials');
const { getRequiredEnv } = require('../_lib/short-links-store');
const { authorizeShortLinksAdmin, sendJson } = require('../_lib/short-links');

function safeAwsError(err){
  return {
    name: err && err.name ? String(err.name).slice(0, 120) : '',
    code: err && err.code ? String(err.code).slice(0, 120) : ''
  };
}

module.exports = async (req, res) => {
  const admin = await authorizeShortLinksAdmin(req);
  if (!admin.authorized) {
    sendJson(res, admin.statusCode, { ok: false, error: admin.error });
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

  let aws;
  let auth;
  try {
    aws = getAwsClientConfig(AWS_WORKLOADS.SHORT_LINKS, { region: env.region });
    auth = describeAwsAuth(AWS_WORKLOADS.SHORT_LINKS, { region: env.region });
  } catch (err) {
    sendJson(res, 503, {
      ok: false,
      error: 'Short links AWS authentication is not configured',
      details: safeAwsError(err)
    });
    return;
  }
  const client = new DynamoDBClient(aws.clientConfig);

  try {
    const result = await client.send(new DescribeTableCommand({ TableName: env.tableName }));
    const table = result && result.Table ? result.Table : null;
    const clicksTableName = process.env.SHORTLINKS_DDB_CLICKS_TABLE
      ? String(process.env.SHORTLINKS_DDB_CLICKS_TABLE).trim()
      : '';

    let clicksTable = null;
    let clicksError = null;
    if (clicksTableName) {
      try {
        const clicksResult = await client.send(new DescribeTableCommand({ TableName: clicksTableName }));
        clicksTable = clicksResult && clicksResult.Table ? clicksResult.Table : null;
      } catch (err) {
        clicksError = safeAwsError(err);
      }
    }

    sendJson(res, 200, {
      ok: true,
      aws: {
        region: env.region,
        ...auth
      },
      table: {
        name: table && table.TableName ? table.TableName : env.tableName,
        status: table && table.TableStatus ? table.TableStatus : '',
        billingMode: table && table.BillingModeSummary ? table.BillingModeSummary.BillingMode : ''
      },
      clicks: {
        configured: !!clicksTableName,
        table: clicksTableName ? {
          name: clicksTable && clicksTable.TableName ? clicksTable.TableName : clicksTableName,
          status: clicksTable && clicksTable.TableStatus ? clicksTable.TableStatus : '',
          billingMode: clicksTable && clicksTable.BillingModeSummary ? clicksTable.BillingModeSummary.BillingMode : ''
        } : null,
        error: clicksError
      }
    });
  } catch (err) {
    sendJson(res, 502, {
      ok: false,
      error: 'DynamoDB backend unavailable',
      details: safeAwsError(err),
      aws: {
        region: env.region,
        table: env.tableName,
        ...auth
      }
    });
  }
};
