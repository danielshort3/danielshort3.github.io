#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const { TransactWriteCommand } = require('@aws-sdk/lib-dynamodb');
const {
  isInternalRecordSlug,
  normalizeSlugLower
} = require('../../api/_lib/short-links');
const {
  assertTableKeySchema,
  createClients,
  migrationError,
  parseOptionArgs,
  printTarget,
  requireApplyGuards,
  resolveTarget,
  safeError,
  scanAll
} = require('./_shared');

const PREFIX = 'short-links-reservations';
const RESERVATION_PREFIX = '__slug_lower__/';

function help(){
  return [
    'Backfill case-insensitive Short Links slug reservations.',
    '',
    'Dry run:',
    '  npm run migrate:short-links-reservations -- --environment production --table <table> --region <region>',
    '',
    'Apply after a reviewed dry run and verified backup:',
    '  npm run migrate:short-links-reservations -- --environment production --table <table> --region <region> --apply --backup-confirmed',
    '',
    'Table env fallbacks:',
    '  SHORTLINKS_DDB_TABLE_PRODUCTION, SHORTLINKS_DDB_TABLE_PREVIEW, SHORTLINKS_DDB_TABLE',
    '',
    'Options: --environment, --table, --region, --page-size, --progress-every, --apply, --backup-confirmed, --help'
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

function opaqueSlugId(value){
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex').slice(0, 12);
}

function analyzeItems(items){
  const linksByLower = new Map();
  const reservationsByLower = new Map();
  const summary = {
    scanned: items.length,
    links: 0,
    internalRecords: 0,
    otherRecords: 0,
    reservations: 0,
    alreadyReserved: 0,
    missingReservations: 0,
    orphanReservations: 0,
    collisions: 0,
    reservationConflicts: 0,
    invalidLinks: 0,
    malformedReservations: 0
  };

  for (const item of items) {
    const slug = typeof item?.slug === 'string' ? item.slug.trim() : '';
    const entityType = typeof item?.entityType === 'string' ? item.entityType.trim() : '';
    if (!slug) {
      summary.otherRecords += 1;
      continue;
    }

    if (slug.startsWith(RESERVATION_PREFIX)) {
      summary.reservations += 1;
      const lower = slug.slice(RESERVATION_PREFIX.length);
      const canonicalSlug = typeof item?.canonicalSlug === 'string' ? item.canonicalSlug.trim() : '';
      const valid = Boolean(
        lower &&
        entityType === 'slugReservation' &&
        canonicalSlug &&
        normalizeSlugLower(canonicalSlug) === lower &&
        (!item.slugLower || String(item.slugLower).trim() === lower)
      );
      if (!valid || reservationsByLower.has(lower)) {
        summary.malformedReservations += 1;
        continue;
      }
      reservationsByLower.set(lower, { canonicalSlug });
      continue;
    }

    if (isInternalRecordSlug(slug)) {
      summary.internalRecords += 1;
      continue;
    }
    if (entityType && entityType !== 'link') {
      summary.otherRecords += 1;
      continue;
    }

    const lower = normalizeSlugLower(slug);
    if (!lower) {
      summary.invalidLinks += 1;
      continue;
    }
    summary.links += 1;
    const group = linksByLower.get(lower) || [];
    group.push(slug);
    linksByLower.set(lower, group);
  }

  const collisionIds = [];
  const conflictIds = [];
  const plan = [];
  for (const [lower, slugs] of linksByLower) {
    if (slugs.length > 1) {
      summary.collisions += 1;
      collisionIds.push(opaqueSlugId(lower));
      continue;
    }
    const canonicalSlug = slugs[0];
    const reservation = reservationsByLower.get(lower);
    if (!reservation) {
      summary.missingReservations += 1;
      plan.push({ lower, canonicalSlug, reservationKey: `${RESERVATION_PREFIX}${lower}` });
      continue;
    }
    if (reservation.canonicalSlug !== canonicalSlug) {
      summary.reservationConflicts += 1;
      conflictIds.push(opaqueSlugId(lower));
      continue;
    }
    summary.alreadyReserved += 1;
  }

  for (const lower of reservationsByLower.keys()) {
    if (!linksByLower.has(lower)) summary.orphanReservations += 1;
  }

  return {
    summary,
    plan,
    collisionIds: collisionIds.sort(),
    conflictIds: conflictIds.sort()
  };
}

function assertSafeAnalysis(analysis){
  const { summary } = analysis;
  if (
    summary.collisions ||
    summary.reservationConflicts ||
    summary.invalidLinks ||
    summary.malformedReservations
  ) {
    throw migrationError('PREFLIGHT_FAILED', 'Collision or record validation failed; no writes were attempted.');
  }
}

async function createReservation(client, tableName, entry, now){
  await client.send(new TransactWriteCommand({
    TransactItems: [
      {
        ConditionCheck: {
          TableName: tableName,
          Key: { slug: entry.canonicalSlug },
          ConditionExpression: 'attribute_exists(#slug) AND (attribute_not_exists(#entityType) OR #entityType = :linkType)',
          ExpressionAttributeNames: {
            '#slug': 'slug',
            '#entityType': 'entityType'
          },
          ExpressionAttributeValues: { ':linkType': 'link' }
        }
      },
      {
        Update: {
          TableName: tableName,
          Key: { slug: entry.reservationKey },
          ConditionExpression: 'attribute_not_exists(#slug) OR (#entityType = :reservationType AND #canonicalSlug = :canonicalSlug)',
          UpdateExpression: 'SET #entityType = :reservationType, #slugLower = :slugLower, #canonicalSlug = :canonicalSlug, #createdAt = if_not_exists(#createdAt, :now), #updatedAt = :now',
          ExpressionAttributeNames: {
            '#slug': 'slug',
            '#entityType': 'entityType',
            '#slugLower': 'slugLower',
            '#canonicalSlug': 'canonicalSlug',
            '#createdAt': 'createdAt',
            '#updatedAt': 'updatedAt'
          },
          ExpressionAttributeValues: {
            ':reservationType': 'slugReservation',
            ':slugLower': entry.lower,
            ':canonicalSlug': entry.canonicalSlug,
            ':now': now
          }
        }
      }
    ]
  }));
}

function printAnalysis(analysis){
  const summary = analysis.summary;
  process.stdout.write(`[${PREFIX}] summary ${JSON.stringify(summary)}\n`);
  if (analysis.collisionIds.length) {
    process.stdout.write(`[${PREFIX}] collisionHashes=${analysis.collisionIds.join(',')}\n`);
  }
  if (analysis.conflictIds.length) {
    process.stdout.write(`[${PREFIX}] reservationConflictHashes=${analysis.conflictIds.join(',')}\n`);
  }
}

async function run(argv = process.argv.slice(2), env = process.env){
  const options = parseOptionArgs(argv, {
    booleanFlags: ['--apply', '--backup-confirmed', '--help'],
    valueFlags: ['--environment', '--table', '--region', '--page-size', '--progress-every']
  });
  if (options.help) {
    process.stdout.write(`${help()}\n`);
    return { help: true };
  }
  requireApplyGuards(options);
  const target = resolveTarget({
    ...options,
    env,
    tableBaseEnv: 'SHORTLINKS_DDB_TABLE',
    regionEnvKeys: ['SHORTLINKS_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']
  });
  const pageSize = positiveInteger(options['page-size'], 100, 1, 1000);
  const progressEvery = positiveInteger(options['progress-every'], 500, 1, 100_000);
  printTarget(PREFIX, target, Boolean(options.apply));

  const clients = createClients(target.region);
  const table = await assertTableKeySchema(clients.base, target.tableName, ['HASH:slug']);
  process.stdout.write(`[${PREFIX}] tableStatus=${table.status} itemCountEstimate=${table.itemCountEstimate}\n`);
  let lastProgress = 0;
  const scan = await scanAll(clients.document, {
    TableName: target.tableName,
    Limit: pageSize,
    ProjectionExpression: '#slug, #entityType, #canonicalSlug, #slugLower',
    ExpressionAttributeNames: {
      '#slug': 'slug',
      '#entityType': 'entityType',
      '#canonicalSlug': 'canonicalSlug',
      '#slugLower': 'slugLower'
    }
  }, ({ pages, scanned }) => {
    if (scanned - lastProgress >= progressEvery) {
      lastProgress = scanned;
      process.stdout.write(`[${PREFIX}] scanProgress pages=${pages} scanned=${scanned}\n`);
    }
  });

  const analysis = analyzeItems(scan.items);
  printAnalysis(analysis);
  assertSafeAnalysis(analysis);
  if (!options.apply) {
    process.stdout.write(`[${PREFIX}] dry-run complete; rerun with --apply --backup-confirmed to write ${analysis.plan.length} reservations.\n`);
    return analysis;
  }

  const now = Date.now();
  let applied = 0;
  for (const entry of analysis.plan) {
    await createReservation(clients.document, target.tableName, entry, now);
    applied += 1;
    if (applied % Math.min(progressEvery, 100) === 0 || applied === analysis.plan.length) {
      process.stdout.write(`[${PREFIX}] applyProgress applied=${applied} total=${analysis.plan.length}\n`);
    }
  }
  process.stdout.write(`[${PREFIX}] apply complete applied=${applied} unchanged=${analysis.summary.alreadyReserved}\n`);
  return { ...analysis, applied };
}

if (require.main === module) {
  run().catch(err => {
    process.stderr.write(`[${PREFIX}] failed=${safeError(err)}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  RESERVATION_PREFIX,
  _internal: {
    analyzeItems,
    assertSafeAnalysis,
    createReservation,
    opaqueSlugId
  },
  run
};
