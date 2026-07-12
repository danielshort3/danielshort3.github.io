'use strict';

const assert = require('node:assert/strict');
const shortLinksMigration = require('../scripts/migrations/short-links-reservations');
const toolsTtlMigration = require('../scripts/migrations/tools-ttl-backfill');
const { requireApplyGuards, resolveTarget } = require('../scripts/migrations/_shared');

async function run(){
  const now = Date.UTC(2026, 6, 11, 12, 0, 0);
  const shortAnalysis = shortLinksMigration._internal.analyzeItems([
    { slug: 'Alpha', entityType: 'link' },
    { slug: 'Bravo', entityType: 'link' },
    {
      slug: '__slug_lower__/bravo',
      entityType: 'slugReservation',
      slugLower: 'bravo',
      canonicalSlug: 'Bravo'
    },
    { slug: '__set__/saved', entityType: 'setTemplate' }
  ]);
  assert.equal(shortAnalysis.summary.links, 2);
  assert.equal(shortAnalysis.summary.alreadyReserved, 1);
  assert.equal(shortAnalysis.summary.missingReservations, 1);
  assert.deepEqual(shortAnalysis.plan[0], {
    lower: 'alpha',
    canonicalSlug: 'Alpha',
    reservationKey: '__slug_lower__/alpha'
  });

  const collision = shortLinksMigration._internal.analyzeItems([
    { slug: 'MixedCase' },
    { slug: 'mixedcase' }
  ]);
  assert.equal(collision.summary.collisions, 1);
  assert.throws(
    () => shortLinksMigration._internal.assertSafeAnalysis(collision),
    err => err?.code === 'PREFLIGHT_FAILED'
  );

  const conflict = shortLinksMigration._internal.analyzeItems([
    { slug: 'Current' },
    {
      slug: '__slug_lower__/current',
      entityType: 'slugReservation',
      slugLower: 'current',
      canonicalSlug: 'CURRENT'
    }
  ]);
  assert.equal(conflict.summary.reservationConflicts, 1);

  let reservationCommand = null;
  await shortLinksMigration._internal.createReservation({
    async send(command){
      reservationCommand = command;
    }
  }, 'short-links-table', {
    lower: 'alpha',
    canonicalSlug: 'Alpha',
    reservationKey: '__slug_lower__/alpha'
  }, now);
  assert.equal(reservationCommand.input.TransactItems.length, 2);
  assert.match(
    reservationCommand.input.TransactItems[1].Update.ConditionExpression,
    /canonicalSlug/
  );

  const sessionUpdatedAt = now - 24 * 60 * 60 * 1000;
  const activityAt = now - 2 * 24 * 60 * 60 * 1000;
  const reversed = String(9_999_999_999_999 - sessionUpdatedAt).padStart(13, '0');
  const ttlAnalysis = toolsTtlMigration._internal.analyzeItems([
    {
      pk: 'USER#user-1#TOOL#text-compare',
      sk: 'SESSION#session-1',
      entityType: 'session',
      updatedAt: sessionUpdatedAt
    },
    {
      pk: 'USER#user-1#SESSIONS',
      sk: `UPDATED#${reversed}#text-compare#session-1`
    },
    {
      pk: 'USER#user-1#ACTIVITY',
      sk: 'TS#0000000000001#text-compare#event-1',
      entityType: 'activity',
      ts: activityAt
    },
    {
      pk: 'USER#user-1',
      sk: 'META',
      entityType: 'user_meta',
      ttl: Math.floor(now / 1000) + 100
    },
    {
      pk: 'TRANSCRIBE#GLOBAL',
      sk: 'DAY#2026-07-11',
      entityType: 'transcribe_global_day',
      ttl: Math.floor(now / 1000) + 100
    }
  ], 'ttl', now);
  assert.equal(ttlAnalysis.summary.sessions, 1);
  assert.equal(ttlAnalysis.summary.sessionIndexes, 1);
  assert.equal(ttlAnalysis.summary.activities, 1);
  assert.equal(ttlAnalysis.summary.metadata, 1);
  assert.equal(ttlAnalysis.summary.otherRecords, 1);
  assert.equal(ttlAnalysis.summary.ttlSetsPlanned, 3);
  assert.equal(ttlAnalysis.summary.ttlRemovalsPlanned, 1);
  assert.equal(ttlAnalysis.summary.invalidManagedRecords, 0);
  assert.equal(
    ttlAnalysis.plan.find(entry => entry.kind === 'session').desiredTtl,
    Math.floor(sessionUpdatedAt / 1000) + 365 * 86400
  );
  assert.equal(
    ttlAnalysis.plan.find(entry => entry.kind === 'activity').desiredTtl,
    Math.floor(activityAt / 1000) + 90 * 86400
  );

  let ttlCommand = null;
  const sessionPlan = ttlAnalysis.plan.find(entry => entry.kind === 'session');
  await toolsTtlMigration._internal.applyEntry({
    async send(command){
      ttlCommand = command;
    }
  }, 'tools-table', 'ttl', sessionPlan);
  assert.equal(ttlCommand.input.UpdateExpression, 'SET #ttl = :desiredTtl');
  assert.match(ttlCommand.input.ConditionExpression, /#timestamp = :priorTimestamp/);

  const invalidTtl = toolsTtlMigration._internal.analyzeItems([{
    pk: 'USER#user-1#TOOL#text-compare',
    sk: 'SESSION#missing-time',
    entityType: 'session'
  }], 'ttl', now);
  assert.equal(invalidTtl.summary.invalidManagedRecords, 1);
  assert.throws(
    () => toolsTtlMigration._internal.assertSafeAnalysis(invalidTtl),
    err => err?.code === 'PREFLIGHT_FAILED'
  );

  assert.throws(
    () => requireApplyGuards({ apply: true }),
    err => err?.code === 'BACKUP_CONFIRMATION_REQUIRED'
  );
  const target = resolveTarget({
    environment: 'preview',
    tableBaseEnv: 'TOOLS_DDB_TABLE',
    regionEnvKeys: ['AWS_REGION'],
    env: {
      TOOLS_DDB_TABLE_PREVIEW: 'preview-tools-table',
      AWS_REGION: 'us-east-2'
    }
  });
  assert.equal(target.tableName, 'preview-tools-table');
  assert.equal(target.environment, 'preview');

  process.stdout.write('aws-data-migrations tests passed\n');
}

run().catch(err => {
  process.stderr.write(`${err?.stack || err}\n`);
  process.exitCode = 1;
});
