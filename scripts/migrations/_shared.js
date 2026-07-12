'use strict';

const crypto = require('crypto');
const {
  DescribeTableCommand,
  DescribeTimeToLiveCommand,
  DynamoDBClient
} = require('@aws-sdk/client-dynamodb');
const {
  DynamoDBDocumentClient,
  ScanCommand
} = require('@aws-sdk/lib-dynamodb');

const VALID_ENVIRONMENTS = new Set(['production', 'preview']);

function migrationError(code, message){
  const err = new Error(message);
  err.code = code;
  return err;
}

function readEnv(name, env = process.env){
  if (!name || typeof env[name] === 'undefined') return '';
  return String(env[name]).trim();
}

function parseOptionArgs(argv, config = {}){
  const booleanFlags = new Set(config.booleanFlags || []);
  const valueFlags = new Set(config.valueFlags || []);
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = String(argv[index] || '');
    if (!token.startsWith('--')) throw migrationError('ARG_INVALID', `Unexpected argument: ${token}`);
    const equalsAt = token.indexOf('=');
    const flag = equalsAt >= 0 ? token.slice(0, equalsAt) : token;
    if (booleanFlags.has(flag)) {
      if (equalsAt >= 0) throw migrationError('ARG_INVALID', `${flag} does not accept a value.`);
      options[flag.slice(2)] = true;
      continue;
    }
    if (!valueFlags.has(flag)) throw migrationError('ARG_INVALID', `Unknown option: ${flag}`);
    const value = equalsAt >= 0 ? token.slice(equalsAt + 1) : argv[++index];
    if (typeof value === 'undefined' || !String(value).trim()) {
      throw migrationError('ARG_INVALID', `${flag} requires a value.`);
    }
    options[flag.slice(2)] = String(value).trim();
  }
  return options;
}

function requireEnvironment(value){
  const environment = String(value || '').trim().toLowerCase();
  if (!VALID_ENVIRONMENTS.has(environment)) {
    throw migrationError('ENVIRONMENT_REQUIRED', '--environment must be production or preview.');
  }
  return environment;
}

function resolveTarget(options = {}){
  const environment = requireEnvironment(options.environment);
  const env = options.env || process.env;
  const tableBaseEnv = String(options.tableBaseEnv || '').trim();
  const regionEnvKeys = Array.isArray(options.regionEnvKeys) ? options.regionEnvKeys : [];
  const environmentTableKey = `${tableBaseEnv}_${environment.toUpperCase()}`;
  const tableName = String(options.table || '').trim()
    || readEnv(environmentTableKey, env)
    || readEnv(tableBaseEnv, env);
  if (!/^[A-Za-z0-9_.-]{3,255}$/.test(tableName)) {
    throw migrationError(
      'TABLE_REQUIRED',
      `Provide --table or set ${environmentTableKey} (or ${tableBaseEnv}).`
    );
  }

  const region = String(options.region || '').trim()
    || regionEnvKeys.map(key => readEnv(key, env)).find(Boolean)
    || '';
  if (!/^[a-z]{2}(?:-gov)?-[a-z]+-\d$/.test(region)) {
    throw migrationError('REGION_REQUIRED', 'Provide --region or configure an AWS region environment variable.');
  }
  return { environment, tableName, region };
}

function createClients(region){
  const base = new DynamoDBClient({ region });
  return {
    base,
    document: DynamoDBDocumentClient.from(base, {
      marshallOptions: { removeUndefinedValues: true }
    })
  };
}

function targetHash(tableName){
  return crypto.createHash('sha256').update(String(tableName || ''), 'utf8').digest('hex').slice(0, 12);
}

async function assertTableKeySchema(client, tableName, expectedKeys){
  const result = await client.send(new DescribeTableCommand({ TableName: tableName }));
  const table = result?.Table;
  const actual = Array.isArray(table?.KeySchema)
    ? table.KeySchema.map(entry => `${entry.KeyType}:${entry.AttributeName}`).sort()
    : [];
  const expected = expectedKeys.slice().sort();
  if (actual.length !== expected.length || actual.some((value, index) => value !== expected[index])) {
    throw migrationError('TABLE_SCHEMA_MISMATCH', 'The target table key schema does not match this migration.');
  }
  return {
    status: String(table?.TableStatus || ''),
    itemCountEstimate: Number(table?.ItemCount) || 0
  };
}

async function getTtlStatus(client, tableName){
  const result = await client.send(new DescribeTimeToLiveCommand({ TableName: tableName }));
  const ttl = result?.TimeToLiveDescription || {};
  return {
    status: String(ttl.TimeToLiveStatus || ''),
    attribute: String(ttl.AttributeName || '')
  };
}

async function scanAll(client, input, onProgress){
  const items = [];
  let lastEvaluatedKey;
  let pages = 0;
  do {
    const result = await client.send(new ScanCommand({
      ...input,
      ExclusiveStartKey: lastEvaluatedKey,
      ConsistentRead: true
    }));
    items.push(...(Array.isArray(result?.Items) ? result.Items : []));
    pages += 1;
    lastEvaluatedKey = result?.LastEvaluatedKey || undefined;
    if (typeof onProgress === 'function') onProgress({ pages, scanned: items.length });
  } while (lastEvaluatedKey);
  return { items, pages };
}

function safeError(err){
  const name = String(err?.name || 'Error').replace(/[^A-Za-z0-9_.-]/g, '').slice(0, 80) || 'Error';
  const code = String(err?.code || '').replace(/[^A-Za-z0-9_.-]/g, '').slice(0, 80);
  return code && code !== name ? `${name} (${code})` : name;
}

function printTarget(prefix, target, apply){
  process.stdout.write(
    `[${prefix}] mode=${apply ? 'apply' : 'dry-run'} environment=${target.environment} ` +
    `region=${target.region} tableHash=${targetHash(target.tableName)}\n`
  );
}

function requireApplyGuards(options){
  if (!options.apply) return;
  if (!options['backup-confirmed']) {
    throw migrationError(
      'BACKUP_CONFIRMATION_REQUIRED',
      '--apply also requires --backup-confirmed after a current backup or export is verified.'
    );
  }
}

module.exports = {
  assertTableKeySchema,
  createClients,
  getTtlStatus,
  migrationError,
  parseOptionArgs,
  printTarget,
  readEnv,
  requireApplyGuards,
  resolveTarget,
  safeError,
  scanAll,
  targetHash
};
