#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const { UpdateTimeToLiveCommand } = require('@aws-sdk/client-dynamodb');
const { TransactWriteCommand, UpdateCommand } = require('@aws-sdk/lib-dynamodb');
const { isInternalRecordSlug } = require('../../api/_lib/short-links');
const {
  assertTableKeySchema,
  createClients,
  getTtlStatus,
  migrationError,
  parseOptionArgs,
  requireApplyGuards,
  resolveTarget,
  safeError,
  scanAll,
  targetHash
} = require('./_shared');

const PREFIX = 'short-links-clicks-reconcile';
const BASELINE_CLICK_ID = '__historical_baseline__';
const BASELINE_ENTITY_TYPE = 'clickBaseline';
const EVENT_ENTITY_TYPE = 'clickEvent';
const TTL_ATTRIBUTE = 'expiresAt';

function help(){
  return [
    'Reconcile Short Links aggregate clicks with durable click history.',
    '',
    'This migration removes expiresAt from click-table rows and creates one honest',
    'summary baseline per live slug. A baseline records only the aggregate/detail gap;',
    'it never invents individual event details.',
    '',
    'Dry run:',
    '  npm run migrate:short-links-clicks -- --environment production --links-table <links-table> --clicks-table <clicks-table> --region <region>',
    '',
    'Apply reconciliation and disable TTL after reviewing a dry run and verifying a backup:',
    '  npm run migrate:short-links-clicks -- --environment production --links-table <links-table> --clicks-table <clicks-table> --region <region> --apply --backup-confirmed --traffic-paused --disable-ttl',
    '',
    'Table env fallbacks:',
    '  SHORTLINKS_DDB_TABLE_PRODUCTION, SHORTLINKS_DDB_TABLE_PREVIEW, SHORTLINKS_DDB_TABLE',
    '  SHORTLINKS_DDB_CLICKS_TABLE_PRODUCTION, SHORTLINKS_DDB_CLICKS_TABLE_PREVIEW, SHORTLINKS_DDB_CLICKS_TABLE',
    '',
    'Options: --environment, --links-table, --clicks-table, --region, --page-size,',
    '  --progress-every, --apply, --backup-confirmed, --traffic-paused,',
    '  --disable-ttl, --help'
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

function hasOwn(item, attribute){
  return Object.prototype.hasOwnProperty.call(item || {}, attribute);
}

function opaqueSlugId(value){
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex').slice(0, 12);
}

function isCount(value){
  return Number.isSafeInteger(value) && value >= 0;
}

function validIsoTimestamp(value){
  if (typeof value !== 'string' || !value.trim()) return false;
  return Number.isFinite(Date.parse(value));
}

function isConditionalConflict(err){
  const name = String(err?.name || err?.code || '');
  return name === 'ConditionalCheckFailedException' || name === 'TransactionCanceledException';
}

function isLiveLink(item){
  const slug = typeof item?.slug === 'string' ? item.slug.trim() : '';
  const entityType = typeof item?.entityType === 'string' ? item.entityType.trim() : '';
  return Boolean(slug && !isInternalRecordSlug(slug) && (!entityType || entityType === 'link'));
}

function classifyBaseline(item){
  const isBaselineKey = item?.clickId === BASELINE_CLICK_ID;
  const isBaselineType = item?.entityType === BASELINE_ENTITY_TYPE;
  if (!isBaselineKey && !isBaselineType) return { baseline: false, valid: true };
  const hasEventDetail = [
    'clickedAt',
    'destination',
    'statusCode',
    'host',
    'path',
    'referer',
    'refererHost',
    'userAgent',
    'country',
    'region',
    'city',
    'timezone',
    'latitude',
    'longitude'
  ]
    .some(attribute => hasOwn(item, attribute));
  const valid = Boolean(
    isBaselineKey &&
    isBaselineType &&
    isCount(item.historicalClicks) &&
    isCount(item.recordedEventCount) &&
    isCount(item.aggregateClicks) &&
    item.historicalClicks + item.recordedEventCount === item.aggregateClicks &&
    (!hasOwn(item, 'reconciledAt') || validIsoTimestamp(item.reconciledAt)) &&
    !hasEventDetail
  );
  return { baseline: true, valid };
}

function analyzeTables(linkItems, clickItems){
  const links = new Map();
  const eventsBySlug = new Map();
  const baselinesBySlug = new Map();
  const ttlPlan = [];
  const invalidIds = [];
  const mismatchIds = [];
  const summary = {
    linksScanned: linkItems.length,
    clickRowsScanned: clickItems.length,
    liveLinks: 0,
    ignoredLinkRecords: 0,
    detailedEvents: 0,
    baselineRows: 0,
    orphanEventSlugs: 0,
    orphanBaselineSlugs: 0,
    invalidRecords: 0,
    aggregateMismatches: 0,
    ttlRemovalsPlanned: 0,
    baselineCreatesPlanned: 0,
    baselineUpdatesPlanned: 0,
    baselinesUnchanged: 0,
    historicalClicksRepresented: 0,
    recordedClicksRepresented: 0,
    aggregateClicksRepresented: 0
  };

  for (const item of linkItems) {
    if (!isLiveLink(item)) {
      summary.ignoredLinkRecords += 1;
      continue;
    }
    const slug = item.slug.trim();
    const aggregatePresent = hasOwn(item, 'clicks');
    const aggregateClicks = aggregatePresent ? item.clicks : 0;
    if (links.has(slug) || !isCount(aggregateClicks)) {
      summary.invalidRecords += 1;
      invalidIds.push(opaqueSlugId(slug));
      continue;
    }
    links.set(slug, {
      slug,
      aggregateClicks,
      aggregatePresent,
      entityTypePresent: hasOwn(item, 'entityType'),
      entityType: item.entityType
    });
    summary.liveLinks += 1;
  }

  for (const item of clickItems) {
    const slug = typeof item?.slug === 'string' ? item.slug.trim() : '';
    const clickId = typeof item?.clickId === 'string' ? item.clickId.trim() : '';
    if (!slug || !clickId) {
      summary.invalidRecords += 1;
      invalidIds.push(opaqueSlugId(`${slug}\u0000${clickId}`));
      continue;
    }

    const baseline = classifyBaseline(item);
    if (baseline.baseline) {
      summary.baselineRows += 1;
      if (!baseline.valid || baselinesBySlug.has(slug)) {
        summary.invalidRecords += 1;
        invalidIds.push(opaqueSlugId(slug));
      } else {
        baselinesBySlug.set(slug, item);
      }
    } else {
      const entityType = typeof item.entityType === 'string' ? item.entityType.trim() : '';
      if ((entityType && entityType !== EVENT_ENTITY_TYPE) || !validIsoTimestamp(item.clickedAt)) {
        summary.invalidRecords += 1;
        invalidIds.push(opaqueSlugId(`${slug}\u0000${clickId}`));
      } else {
        eventsBySlug.set(slug, (eventsBySlug.get(slug) || 0) + 1);
        summary.detailedEvents += 1;
      }
    }

    if (hasOwn(item, TTL_ATTRIBUTE)) {
      ttlPlan.push({
        key: { slug, clickId },
        priorExpiresAt: item[TTL_ATTRIBUTE],
        entityTypePresent: hasOwn(item, 'entityType'),
        entityType: item.entityType
      });
      summary.ttlRemovalsPlanned += 1;
    }
  }

  const baselinePlan = [];
  for (const link of links.values()) {
    const recordedEventCount = eventsBySlug.get(link.slug) || 0;
    if (recordedEventCount > link.aggregateClicks) {
      summary.aggregateMismatches += 1;
      mismatchIds.push(opaqueSlugId(link.slug));
      continue;
    }
    const historicalClicks = link.aggregateClicks - recordedEventCount;
    const prior = baselinesBySlug.get(link.slug) || null;
    const desired = {
      historicalClicks,
      recordedEventCount,
      aggregateClicks: link.aggregateClicks
    };
    summary.historicalClicksRepresented += historicalClicks;
    summary.recordedClicksRepresented += recordedEventCount;
    summary.aggregateClicksRepresented += link.aggregateClicks;

    const matches = Boolean(
      prior &&
      prior.historicalClicks === desired.historicalClicks &&
      prior.recordedEventCount === desired.recordedEventCount &&
      prior.aggregateClicks === desired.aggregateClicks &&
      validIsoTimestamp(prior.reconciledAt)
    );
    if (matches) {
      summary.baselinesUnchanged += 1;
      continue;
    }
    if (prior) summary.baselineUpdatesPlanned += 1;
    else summary.baselineCreatesPlanned += 1;
    baselinePlan.push({ link, prior, desired });
  }

  const clickSlugs = new Set([...eventsBySlug.keys(), ...baselinesBySlug.keys()]);
  for (const slug of clickSlugs) {
    if (links.has(slug)) continue;
    if (eventsBySlug.has(slug)) summary.orphanEventSlugs += 1;
    if (baselinesBySlug.has(slug)) summary.orphanBaselineSlugs += 1;
  }

  return {
    summary,
    ttlPlan,
    baselinePlan,
    invalidIds: [...new Set(invalidIds)].sort(),
    mismatchIds: [...new Set(mismatchIds)].sort()
  };
}

function assertSafeAnalysis(analysis){
  if (analysis.summary.invalidRecords || analysis.summary.aggregateMismatches) {
    throw migrationError(
      'PREFLIGHT_FAILED',
      'Invalid records or aggregate/detail mismatches were found; no writes were attempted.'
    );
  }
}

function requireTrafficPause(options){
  if (options.apply && !options['traffic-paused']) {
    throw migrationError(
      'TRAFFIC_PAUSE_REQUIRED',
      '--apply requires --traffic-paused after redirect and click writes are paused.'
    );
  }
}

function ttlApplyAction(ttl, options){
  const status = String(ttl?.status || '');
  const attribute = String(ttl?.attribute || '');
  if (!options.apply) return 'dry-run';
  if (status === 'ENABLED') {
    if (attribute !== TTL_ATTRIBUTE) {
      throw migrationError('TTL_ATTRIBUTE_MISMATCH', 'The click table uses an unexpected TTL attribute; no changes were made.');
    }
    if (!options['disable-ttl']) {
      throw migrationError('TTL_MUST_BE_DISABLED', 'Rerun with --disable-ttl before removing expiration attributes.');
    }
    return 'request-disable';
  }
  if (status === 'ENABLING') {
    throw migrationError('TTL_TRANSITION_PENDING', 'TTL is still enabling. Wait for ENABLED, then rerun with --disable-ttl.');
  }
  if (status === 'DISABLING') return 'apply-during-disable';
  if (status !== 'DISABLED') {
    throw migrationError('TTL_STATUS_UNKNOWN', 'The click-table TTL status could not be verified as DISABLED.');
  }
  return 'ready';
}

async function requestTtlDisable(client, tableName){
  await client.send(new UpdateTimeToLiveCommand({
    TableName: tableName,
    TimeToLiveSpecification: {
      AttributeName: TTL_ATTRIBUTE,
      Enabled: false
    }
  }));
}

async function applyTtlRemoval(client, tableName, entry){
  const names = {
    '#slug': 'slug',
    '#clickId': 'clickId',
    '#expiresAt': TTL_ATTRIBUTE,
    '#entityType': 'entityType'
  };
  const values = { ':priorExpiresAt': entry.priorExpiresAt };
  const conditions = [
    'attribute_exists(#slug)',
    'attribute_exists(#clickId)',
    '#expiresAt = :priorExpiresAt'
  ];
  if (entry.entityTypePresent) {
    values[':priorEntityType'] = entry.entityType;
    conditions.push('#entityType = :priorEntityType');
  } else {
    conditions.push('attribute_not_exists(#entityType)');
  }
  await client.send(new UpdateCommand({
    TableName: tableName,
    Key: entry.key,
    UpdateExpression: 'REMOVE #expiresAt',
    ConditionExpression: conditions.join(' AND '),
    ExpressionAttributeNames: names,
    ExpressionAttributeValues: values
  }));
}

function baselineCondition(entry, names, values){
  if (!entry.prior) return 'attribute_not_exists(#clickId)';
  values[':priorEntityType'] = entry.prior.entityType;
  values[':priorHistoricalClicks'] = entry.prior.historicalClicks;
  values[':priorRecordedEventCount'] = entry.prior.recordedEventCount;
  values[':priorAggregateClicks'] = entry.prior.aggregateClicks;
  const conditions = [
    '#entityType = :priorEntityType',
    '#historicalClicks = :priorHistoricalClicks',
    '#recordedEventCount = :priorRecordedEventCount',
    '#aggregateClicks = :priorAggregateClicks'
  ];
  if (hasOwn(entry.prior, 'reconciledAt')) {
    values[':priorReconciledAt'] = entry.prior.reconciledAt;
    conditions.push('#reconciledAt = :priorReconciledAt');
  } else {
    conditions.push('attribute_not_exists(#reconciledAt)');
  }
  return conditions.join(' AND ');
}

async function applyBaseline(client, linksTableName, clicksTableName, entry, reconciledAt){
  const linkNames = { '#slug': 'slug', '#entityType': 'entityType', '#clicks': 'clicks' };
  const linkValues = {};
  const linkConditions = ['attribute_exists(#slug)'];
  if (entry.link.entityTypePresent) {
    linkValues[':priorLinkEntityType'] = entry.link.entityType;
    linkConditions.push('#entityType = :priorLinkEntityType');
  } else {
    linkConditions.push('attribute_not_exists(#entityType)');
  }
  if (entry.link.aggregatePresent) {
    linkValues[':priorLinkClicks'] = entry.link.aggregateClicks;
    linkConditions.push('#clicks = :priorLinkClicks');
  } else {
    linkConditions.push('attribute_not_exists(#clicks)');
  }

  const baselineNames = {
    '#clickId': 'clickId',
    '#entityType': 'entityType',
    '#historicalClicks': 'historicalClicks',
    '#recordedEventCount': 'recordedEventCount',
    '#aggregateClicks': 'aggregateClicks',
    '#reconciledAt': 'reconciledAt',
    '#expiresAt': TTL_ATTRIBUTE
  };
  const baselineValues = {
    ':entityType': BASELINE_ENTITY_TYPE,
    ':historicalClicks': entry.desired.historicalClicks,
    ':recordedEventCount': entry.desired.recordedEventCount,
    ':aggregateClicks': entry.desired.aggregateClicks,
    ':reconciledAt': reconciledAt
  };
  const condition = baselineCondition(entry, baselineNames, baselineValues);

  const conditionCheck = {
    TableName: linksTableName,
    Key: { slug: entry.link.slug },
    ConditionExpression: linkConditions.join(' AND '),
    ExpressionAttributeNames: linkNames
  };
  if (Object.keys(linkValues).length) conditionCheck.ExpressionAttributeValues = linkValues;

  await client.send(new TransactWriteCommand({
    TransactItems: [
      { ConditionCheck: conditionCheck },
      {
        Update: {
          TableName: clicksTableName,
          Key: { slug: entry.link.slug, clickId: BASELINE_CLICK_ID },
          ConditionExpression: condition,
          UpdateExpression: 'SET #entityType = :entityType, #historicalClicks = :historicalClicks, #recordedEventCount = :recordedEventCount, #aggregateClicks = :aggregateClicks, #reconciledAt = :reconciledAt REMOVE #expiresAt',
          ExpressionAttributeNames: baselineNames,
          ExpressionAttributeValues: baselineValues
        }
      }
    ]
  }));
}

function resolveTargets(options, env){
  const common = {
    environment: options.environment,
    region: options.region,
    env,
    regionEnvKeys: ['SHORTLINKS_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']
  };
  const links = resolveTarget({
    ...common,
    table: options['links-table'],
    tableBaseEnv: 'SHORTLINKS_DDB_TABLE'
  });
  const clicks = resolveTarget({
    ...common,
    table: options['clicks-table'],
    tableBaseEnv: 'SHORTLINKS_DDB_CLICKS_TABLE'
  });
  if (links.region !== clicks.region || links.environment !== clicks.environment) {
    throw migrationError('TARGET_MISMATCH', 'Links and clicks targets must use the same environment and region.');
  }
  if (links.tableName === clicks.tableName) {
    throw migrationError('TARGET_MISMATCH', 'Links and clicks must be separate DynamoDB tables.');
  }
  return { links, clicks };
}

async function run(argv = process.argv.slice(2), env = process.env){
  const options = parseOptionArgs(argv, {
    booleanFlags: [
      '--apply',
      '--backup-confirmed',
      '--traffic-paused',
      '--disable-ttl',
      '--help'
    ],
    valueFlags: ['--environment', '--links-table', '--clicks-table', '--region', '--page-size', '--progress-every']
  });
  if (options.help) {
    process.stdout.write(`${help()}\n`);
    return { help: true };
  }
  requireApplyGuards(options);
  requireTrafficPause(options);
  const targets = resolveTargets(options, env);
  const pageSize = positiveInteger(options['page-size'], 100, 1, 1000);
  const progressEvery = positiveInteger(options['progress-every'], 500, 1, 100_000);
  process.stdout.write(
    `[${PREFIX}] mode=${options.apply ? 'apply' : 'dry-run'} environment=${targets.links.environment} ` +
    `region=${targets.links.region} linksTableHash=${targetHash(targets.links.tableName)} ` +
    `clicksTableHash=${targetHash(targets.clicks.tableName)}\n`
  );

  const clients = createClients(targets.links.region);
  const linksTable = await assertTableKeySchema(clients.base, targets.links.tableName, ['HASH:slug']);
  const clicksTable = await assertTableKeySchema(
    clients.base,
    targets.clicks.tableName,
    ['HASH:slug', 'RANGE:clickId']
  );
  const ttl = await getTtlStatus(clients.base, targets.clicks.tableName);
  process.stdout.write(
    `[${PREFIX}] linksStatus=${linksTable.status} linksItemCountEstimate=${linksTable.itemCountEstimate} ` +
    `clicksStatus=${clicksTable.status} clicksItemCountEstimate=${clicksTable.itemCountEstimate} ` +
    `ttlStatus=${ttl.status || 'UNKNOWN'} ttlAttribute=${ttl.attribute || 'none'}\n`
  );

  let linksProgress = 0;
  const linksScan = await scanAll(clients.document, {
    TableName: targets.links.tableName,
    Limit: pageSize,
    ConsistentRead: true,
    ProjectionExpression: '#slug, #entityType, #clicks',
    ExpressionAttributeNames: {
      '#slug': 'slug',
      '#entityType': 'entityType',
      '#clicks': 'clicks'
    }
  }, ({ pages, scanned }) => {
    if (scanned - linksProgress >= progressEvery) {
      linksProgress = scanned;
      process.stdout.write(`[${PREFIX}] linksScanProgress pages=${pages} scanned=${scanned}\n`);
    }
  });

  let clicksProgress = 0;
  const clicksScan = await scanAll(clients.document, {
    TableName: targets.clicks.tableName,
    Limit: pageSize,
    ConsistentRead: true,
    ProjectionExpression: '#slug, #clickId, #entityType, #clickedAt, #historicalClicks, #recordedEventCount, #aggregateClicks, #reconciledAt, #expiresAt, #destination, #statusCode, #host, #path, #referer, #refererHost, #userAgent, #country, #region, #city, #timezone, #latitude, #longitude',
    ExpressionAttributeNames: {
      '#slug': 'slug',
      '#clickId': 'clickId',
      '#entityType': 'entityType',
      '#clickedAt': 'clickedAt',
      '#historicalClicks': 'historicalClicks',
      '#recordedEventCount': 'recordedEventCount',
      '#aggregateClicks': 'aggregateClicks',
      '#reconciledAt': 'reconciledAt',
      '#expiresAt': TTL_ATTRIBUTE,
      '#destination': 'destination',
      '#statusCode': 'statusCode',
      '#host': 'host',
      '#path': 'path',
      '#referer': 'referer',
      '#refererHost': 'refererHost',
      '#userAgent': 'userAgent',
      '#country': 'country',
      '#region': 'region',
      '#city': 'city',
      '#timezone': 'timezone',
      '#latitude': 'latitude',
      '#longitude': 'longitude'
    }
  }, ({ pages, scanned }) => {
    if (scanned - clicksProgress >= progressEvery) {
      clicksProgress = scanned;
      process.stdout.write(`[${PREFIX}] clicksScanProgress pages=${pages} scanned=${scanned}\n`);
    }
  });

  const analysis = analyzeTables(linksScan.items, clicksScan.items);
  process.stdout.write(`[${PREFIX}] summary ${JSON.stringify(analysis.summary)}\n`);
  if (analysis.invalidIds.length) {
    process.stdout.write(`[${PREFIX}] invalidRecordHashes=${analysis.invalidIds.join(',')}\n`);
  }
  if (analysis.mismatchIds.length) {
    process.stdout.write(`[${PREFIX}] aggregateMismatchHashes=${analysis.mismatchIds.join(',')}\n`);
  }
  assertSafeAnalysis(analysis);
  if (!options.apply) {
    process.stdout.write(
      `[${PREFIX}] dry-run complete; no TTL or row changes were made. Review the summary before apply.\n`
    );
    return { ...analysis, ttl };
  }

  const ttlAction = ttlApplyAction(ttl, options);
  let ttlDisableRequested = false;
  if (ttlAction === 'request-disable') {
    await requestTtlDisable(clients.base, targets.clicks.tableName);
    ttlDisableRequested = true;
    process.stdout.write(
      `[${PREFIX}] TTL disable requested; removing expiration attributes immediately so pending rows are no longer eligible for TTL deletion.\n`
    );
  }
  if (ttlAction === 'apply-during-disable') {
    process.stdout.write(
      `[${PREFIX}] TTL is DISABLING; continuing row reconciliation to protect remaining click records.\n`
    );
  }

  let ttlRemovalsApplied = 0;
  for (const entry of analysis.ttlPlan) {
    try {
      await applyTtlRemoval(clients.document, targets.clicks.tableName, entry);
    } catch (err) {
      if (isConditionalConflict(err)) {
        throw migrationError(
          'ROW_CHANGED_RETRY_REQUIRED',
          'A click row changed after the scan. Rerun the dry run before retrying.'
        );
      }
      throw err;
    }
    ttlRemovalsApplied += 1;
    if (
      ttlRemovalsApplied % Math.min(progressEvery, 100) === 0 ||
      ttlRemovalsApplied === analysis.ttlPlan.length
    ) {
      process.stdout.write(
        `[${PREFIX}] ttlProgress applied=${ttlRemovalsApplied} total=${analysis.ttlPlan.length}\n`
      );
    }
  }

  const reconciledAt = new Date().toISOString();
  let baselinesApplied = 0;
  for (const entry of analysis.baselinePlan) {
    try {
      await applyBaseline(
        clients.document,
        targets.links.tableName,
        targets.clicks.tableName,
        entry,
        reconciledAt
      );
    } catch (err) {
      if (isConditionalConflict(err)) {
        throw migrationError(
          'CONCURRENT_CLICK_RETRY_REQUIRED',
          'The link aggregate or baseline changed after the scan. Rerun the dry run before retrying.'
        );
      }
      throw err;
    }
    baselinesApplied += 1;
    if (
      baselinesApplied % Math.min(progressEvery, 100) === 0 ||
      baselinesApplied === analysis.baselinePlan.length
    ) {
      process.stdout.write(
        `[${PREFIX}] baselineProgress applied=${baselinesApplied} total=${analysis.baselinePlan.length}\n`
      );
    }
  }

  const applied = ttlRemovalsApplied + baselinesApplied;
  process.stdout.write(
    `[${PREFIX}] apply complete applied=${applied} ttlRemovals=${ttlRemovalsApplied} ` +
    `baselines=${baselinesApplied}; rerun the dry run and require all planned counts to be zero.\n`
  );
  return {
    ...analysis,
    ttl,
    ttlDisableRequested,
    ttlTransitionPending: ttlDisableRequested || ttlAction === 'apply-during-disable',
    applied,
    ttlRemovalsApplied,
    baselinesApplied
  };
}

if (require.main === module) {
  run().catch(err => {
    process.stderr.write(`[${PREFIX}] failed=${safeError(err)}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  BASELINE_CLICK_ID,
  BASELINE_ENTITY_TYPE,
  EVENT_ENTITY_TYPE,
  TTL_ATTRIBUTE,
  _internal: {
    analyzeTables,
    applyBaseline,
    applyTtlRemoval,
    assertSafeAnalysis,
    classifyBaseline,
    isConditionalConflict,
    opaqueSlugId,
    requestTtlDisable,
    requireTrafficPause,
    resolveTargets,
    ttlApplyAction
  },
  run
};
