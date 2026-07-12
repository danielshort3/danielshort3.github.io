#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const { UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const {
  assertTableKeySchema,
  createClients,
  getTtlStatus,
  migrationError,
  parseOptionArgs,
  printTarget,
  requireApplyGuards,
  resolveTarget,
  safeError,
  scanAll
} = require('./_shared');

const PREFIX = 'tools-ttl-backfill';
const DAY_SECONDS = 24 * 60 * 60;
const MAX_TS = 9_999_999_999_999;
const SESSION_RETENTION_DAYS = 365;
const ACTIVITY_RETENTION_DAYS = 90;

function help(){
  return [
    'Backfill Tools-account TTL while keeping account/tool metadata durable.',
    '',
    'Policy: sessions and session indexes = 365 days; activity = 90 days; metadata = no TTL.',
    '',
    'Dry run:',
    '  npm run migrate:tools-ttl -- --environment production --table <table> --region <region>',
    '',
    'Apply after a reviewed dry run, verified backup, and enabled DynamoDB TTL:',
    '  npm run migrate:tools-ttl -- --environment production --table <table> --region <region> --apply --backup-confirmed',
    '',
    'Table env fallbacks:',
    '  TOOLS_DDB_TABLE_PRODUCTION, TOOLS_DDB_TABLE_PREVIEW, TOOLS_DDB_TABLE',
    '',
    'Options: --environment, --table, --region, --ttl-attribute, --page-size, --progress-every, --apply, --backup-confirmed, --help'
  ].join('\n');
}

function positiveInteger(value, fallback, min, max){
  if (typeof value === 'undefined') return fallback;
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw migrationError('ARG_INVALID', `Expected an integer from ${min} to ${max}.`);
  }
  return parsed;
}

function opaqueItemId(item){
  const key = `${String(item?.pk || '')}\u0000${String(item?.sk || '')}`;
  return crypto.createHash('sha256').update(key, 'utf8').digest('hex').slice(0, 12);
}

function keyShape(item){
  const pk = typeof item?.pk === 'string' ? item.pk : '';
  const sk = typeof item?.sk === 'string' ? item.sk : '';
  if (/^USER#[^#]+#TOOL#[^#]+$/.test(pk) && /^SESSION#/.test(sk)) return 'session';
  if (
    (/^USER#[^#]+#SESSIONS$/.test(pk) || /^USER#[^#]+#TOOLSESSIONS#[^#]+$/.test(pk)) &&
    /^UPDATED#/.test(sk)
  ) return 'session_index';
  if (/^USER#[^#]+#ACTIVITY(?:#[^#]+)?$/.test(pk) && /^TS#/.test(sk)) return 'activity';
  if (/^USER#[^#]+$/.test(pk) && (sk === 'META' || /^TOOL#[^#]+$/.test(sk))) return 'metadata';
  return 'other';
}

function classifyItem(item){
  const entityType = typeof item?.entityType === 'string' ? item.entityType.trim() : '';
  const shape = keyShape(item);
  if (!entityType) return { kind: shape, legacy: shape !== 'other' };
  if (entityType === 'session') return { kind: shape === 'session' ? 'session' : 'invalid', legacy: false };
  if (entityType === 'session_index') return { kind: shape === 'session_index' ? 'session_index' : 'invalid', legacy: false };
  if (entityType === 'activity') return { kind: shape === 'activity' ? 'activity' : 'invalid', legacy: false };
  if (entityType === 'user_meta' || entityType === 'tool_meta') {
    return { kind: shape === 'metadata' ? 'metadata' : 'invalid', legacy: false };
  }
  return { kind: 'other', legacy: false };
}

function toTimestampMs(value){
  if (value instanceof Date) return value.getTime();
  if (typeof value === 'number' || (typeof value === 'string' && /^\d+(?:\.\d+)?$/.test(value.trim()))) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric <= 0) return 0;
    return numeric < 100_000_000_000 ? Math.floor(numeric * 1000) : Math.floor(numeric);
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Date.parse(value.trim());
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
  }
  return 0;
}

function reverseTimestampFromKey(sk){
  const match = /^(?:UPDATED|TS)#(\d{13})(?:#|$)/.exec(String(sk || ''));
  if (!match) return 0;
  const reversed = Number(match[1]);
  if (!Number.isSafeInteger(reversed) || reversed < 0 || reversed > MAX_TS) return 0;
  return MAX_TS - reversed;
}

function timestampFor(item, kind, nowMs){
  const candidates = kind === 'activity'
    ? ['ts', 'updatedAt', 'createdAt']
    : ['updatedAt', 'createdAt'];
  for (const attribute of candidates) {
    if (!Object.prototype.hasOwnProperty.call(item, attribute)) continue;
    const timestampMs = toTimestampMs(item[attribute]);
    if (timestampMs > 0 && timestampMs <= nowMs + 5 * 60 * 1000) {
      return { timestampMs, attribute, rawValue: item[attribute], derived: false, absentAttributes: [] };
    }
    return null;
  }
  const derived = reverseTimestampFromKey(item?.sk);
  if (derived > 0 && derived <= nowMs + 5 * 60 * 1000) {
    return { timestampMs: derived, attribute: '', rawValue: undefined, derived: true, absentAttributes: candidates };
  }
  return null;
}

function ttlIsPresent(item, ttlAttribute){
  return Object.prototype.hasOwnProperty.call(item, ttlAttribute) && item[ttlAttribute] !== null;
}

function analyzeItems(items, ttlAttribute, nowMs = Date.now()){
  const summary = {
    scanned: items.length,
    sessions: 0,
    sessionIndexes: 0,
    activities: 0,
    metadata: 0,
    otherRecords: 0,
    legacyManagedRecords: 0,
    unchanged: 0,
    ttlSetsPlanned: 0,
    ttlRemovalsPlanned: 0,
    desiredTtlAlreadyExpired: 0,
    invalidManagedRecords: 0
  };
  const invalidIds = [];
  const plan = [];
  for (const item of items) {
    const classification = classifyItem(item);
    const kind = classification.kind;
    if (kind === 'other') {
      summary.otherRecords += 1;
      continue;
    }
    if (kind === 'invalid') {
      summary.invalidManagedRecords += 1;
      invalidIds.push(opaqueItemId(item));
      continue;
    }
    if (classification.legacy) summary.legacyManagedRecords += 1;

    if (kind === 'metadata') {
      summary.metadata += 1;
      if (!ttlIsPresent(item, ttlAttribute)) {
        summary.unchanged += 1;
        continue;
      }
      summary.ttlRemovalsPlanned += 1;
      plan.push({
        action: 'remove',
        kind,
        key: { pk: item.pk, sk: item.sk },
        priorTtl: item[ttlAttribute],
        priorTtlPresent: true,
        entityTypePresent: Object.prototype.hasOwnProperty.call(item, 'entityType'),
        entityType: item.entityType
      });
      continue;
    }

    if (kind === 'session') summary.sessions += 1;
    if (kind === 'session_index') summary.sessionIndexes += 1;
    if (kind === 'activity') summary.activities += 1;
    const timestamp = timestampFor(item, kind, nowMs);
    if (!timestamp) {
      summary.invalidManagedRecords += 1;
      invalidIds.push(opaqueItemId(item));
      continue;
    }
    const retentionDays = kind === 'activity' ? ACTIVITY_RETENTION_DAYS : SESSION_RETENTION_DAYS;
    const desiredTtl = Math.floor(timestamp.timestampMs / 1000) + retentionDays * DAY_SECONDS;
    if (desiredTtl <= Math.floor(nowMs / 1000)) summary.desiredTtlAlreadyExpired += 1;
    const priorTtlPresent = ttlIsPresent(item, ttlAttribute);
    const priorTtl = priorTtlPresent ? item[ttlAttribute] : undefined;
    if (priorTtlPresent && typeof priorTtl === 'number' && Number.isFinite(priorTtl) && priorTtl === desiredTtl) {
      summary.unchanged += 1;
      continue;
    }
    summary.ttlSetsPlanned += 1;
    plan.push({
      action: 'set',
      kind,
      key: { pk: item.pk, sk: item.sk },
      desiredTtl,
      priorTtl,
      priorTtlPresent,
      entityTypePresent: Object.prototype.hasOwnProperty.call(item, 'entityType'),
      entityType: item.entityType,
      timestamp
    });
  }
  return { summary, plan, invalidIds: invalidIds.sort() };
}

function assertSafeAnalysis(analysis){
  if (analysis.summary.invalidManagedRecords) {
    throw migrationError('PREFLIGHT_FAILED', 'Managed records with invalid keys or timestamps were found; no writes were attempted.');
  }
}

function addSnapshotConditions(entry, ttlAttribute){
  const names = { '#pk': 'pk', '#sk': 'sk', '#ttl': ttlAttribute, '#entityType': 'entityType' };
  const values = {};
  const conditions = ['attribute_exists(#pk)', 'attribute_exists(#sk)'];
  if (entry.priorTtlPresent) {
    values[':priorTtl'] = entry.priorTtl;
    conditions.push('#ttl = :priorTtl');
  } else {
    conditions.push('attribute_not_exists(#ttl)');
  }
  if (entry.entityTypePresent) {
    values[':priorEntityType'] = entry.entityType;
    conditions.push('#entityType = :priorEntityType');
  } else {
    conditions.push('attribute_not_exists(#entityType)');
  }
  if (entry.action === 'set' && entry.timestamp) {
    if (entry.timestamp.derived) {
      entry.timestamp.absentAttributes.forEach((attribute, index) => {
        const name = `#timestamp${index}`;
        names[name] = attribute;
        conditions.push(`attribute_not_exists(${name})`);
      });
    } else {
      names['#timestamp'] = entry.timestamp.attribute;
      values[':priorTimestamp'] = entry.timestamp.rawValue;
      conditions.push('#timestamp = :priorTimestamp');
    }
  }
  return { names, values, condition: conditions.join(' AND ') };
}

async function applyEntry(client, tableName, ttlAttribute, entry){
  const snapshot = addSnapshotConditions(entry, ttlAttribute);
  if (entry.action === 'remove') {
    await client.send(new UpdateCommand({
      TableName: tableName,
      Key: entry.key,
      UpdateExpression: 'REMOVE #ttl',
      ConditionExpression: snapshot.condition,
      ExpressionAttributeNames: snapshot.names,
      ExpressionAttributeValues: snapshot.values
    }));
    return;
  }
  await client.send(new UpdateCommand({
    TableName: tableName,
    Key: entry.key,
    UpdateExpression: 'SET #ttl = :desiredTtl',
    ConditionExpression: snapshot.condition,
    ExpressionAttributeNames: snapshot.names,
    ExpressionAttributeValues: { ...snapshot.values, ':desiredTtl': entry.desiredTtl }
  }));
}

function validateTtlAttribute(value){
  const attribute = String(value || 'ttl').trim();
  if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(attribute)) {
    throw migrationError('TTL_ATTRIBUTE_INVALID', '--ttl-attribute must be a valid DynamoDB attribute name.');
  }
  if (['pk', 'sk', 'entityType', 'createdAt', 'updatedAt', 'ts'].includes(attribute)) {
    throw migrationError('TTL_ATTRIBUTE_INVALID', 'The TTL attribute conflicts with a structural attribute.');
  }
  return attribute;
}

async function run(argv = process.argv.slice(2), env = process.env){
  const options = parseOptionArgs(argv, {
    booleanFlags: ['--apply', '--backup-confirmed', '--help'],
    valueFlags: ['--environment', '--table', '--region', '--ttl-attribute', '--page-size', '--progress-every']
  });
  if (options.help) {
    process.stdout.write(`${help()}\n`);
    return { help: true };
  }
  requireApplyGuards(options);
  const target = resolveTarget({
    ...options,
    env,
    tableBaseEnv: 'TOOLS_DDB_TABLE',
    regionEnvKeys: ['TOOLS_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']
  });
  const ttlAttribute = validateTtlAttribute(options['ttl-attribute'] || env.TOOLS_DDB_TTL_ATTRIBUTE || 'ttl');
  const pageSize = positiveInteger(options['page-size'], 100, 1, 1000);
  const progressEvery = positiveInteger(options['progress-every'], 500, 1, 100_000);
  printTarget(PREFIX, target, Boolean(options.apply));
  process.stdout.write(`[${PREFIX}] policy sessions=${SESSION_RETENTION_DAYS}d activity=${ACTIVITY_RETENTION_DAYS}d metadata=durable ttlAttribute=${ttlAttribute}\n`);

  const clients = createClients(target.region);
  const table = await assertTableKeySchema(clients.base, target.tableName, ['HASH:pk', 'RANGE:sk']);
  const ttl = await getTtlStatus(clients.base, target.tableName);
  process.stdout.write(
    `[${PREFIX}] tableStatus=${table.status} itemCountEstimate=${table.itemCountEstimate} ` +
    `ttlStatus=${ttl.status || 'UNKNOWN'} ttlAttributeMatches=${ttl.attribute === ttlAttribute}\n`
  );
  if (options.apply && (!['ENABLED', 'ENABLING'].includes(ttl.status) || ttl.attribute !== ttlAttribute)) {
    throw migrationError('TTL_NOT_READY', 'Enable DynamoDB TTL on the configured attribute before applying the backfill.');
  }

  let lastProgress = 0;
  const names = {
    '#pk': 'pk',
    '#sk': 'sk',
    '#entityType': 'entityType',
    '#ttl': ttlAttribute,
    '#createdAt': 'createdAt',
    '#updatedAt': 'updatedAt',
    '#ts': 'ts'
  };
  const scan = await scanAll(clients.document, {
    TableName: target.tableName,
    Limit: pageSize,
    ProjectionExpression: '#pk, #sk, #entityType, #ttl, #createdAt, #updatedAt, #ts',
    ExpressionAttributeNames: names
  }, ({ pages, scanned }) => {
    if (scanned - lastProgress >= progressEvery) {
      lastProgress = scanned;
      process.stdout.write(`[${PREFIX}] scanProgress pages=${pages} scanned=${scanned}\n`);
    }
  });

  const analysis = analyzeItems(scan.items, ttlAttribute);
  process.stdout.write(`[${PREFIX}] summary ${JSON.stringify(analysis.summary)}\n`);
  if (analysis.invalidIds.length) {
    process.stdout.write(`[${PREFIX}] invalidRecordHashes=${analysis.invalidIds.join(',')}\n`);
  }
  assertSafeAnalysis(analysis);
  if (!options.apply) {
    process.stdout.write(`[${PREFIX}] dry-run complete; rerun with --apply --backup-confirmed to write ${analysis.plan.length} conditional updates.\n`);
    return analysis;
  }

  let applied = 0;
  for (const entry of analysis.plan) {
    await applyEntry(clients.document, target.tableName, ttlAttribute, entry);
    applied += 1;
    if (applied % Math.min(progressEvery, 100) === 0 || applied === analysis.plan.length) {
      process.stdout.write(`[${PREFIX}] applyProgress applied=${applied} total=${analysis.plan.length}\n`);
    }
  }
  process.stdout.write(
    `[${PREFIX}] apply complete applied=${applied} ttlSets=${analysis.summary.ttlSetsPlanned} ` +
    `ttlRemovals=${analysis.summary.ttlRemovalsPlanned}; rerun the dry-run to verify zero pending updates.\n`
  );
  return { ...analysis, applied };
}

if (require.main === module) {
  run().catch(err => {
    process.stderr.write(`[${PREFIX}] failed=${safeError(err)}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  ACTIVITY_RETENTION_DAYS,
  SESSION_RETENTION_DAYS,
  _internal: {
    addSnapshotConditions,
    analyzeItems,
    applyEntry,
    assertSafeAnalysis,
    classifyItem,
    keyShape,
    reverseTimestampFromKey,
    timestampFor,
    toTimestampMs,
    validateTtlAttribute
  },
  run
};
