'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const shortLinksClicksMigration = require('../scripts/migrations/short-links-clicks-reconcile');
const shortLinksMigration = require('../scripts/migrations/short-links-reservations');
const transcribeHistoryMigration = require('../scripts/migrations/transcribe-history-backfill');
const toolsTtlMigration = require('../scripts/migrations/tools-ttl-backfill');
const { requireApplyGuards, resolveTarget } = require('../scripts/migrations/_shared');

async function run(){
  const clickMigrationSource = fs.readFileSync(
    require.resolve('../scripts/migrations/short-links-clicks-reconcile'),
    'utf8'
  );
  assert.equal(
    (clickMigrationSource.match(/ConsistentRead: true/g) || []).length,
    2,
    'click reconciliation should strongly read both the link and click table scans'
  );
  const transcribeMigrationSource = fs.readFileSync(
    require.resolve('../scripts/migrations/transcribe-history-backfill'),
    'utf8'
  );
  assert.match(transcribeMigrationSource, /ConsistentRead: true/);
  assert.match(transcribeMigrationSource, /IfNoneMatch: '\*'/);
  assert.doesNotMatch(transcribeMigrationSource, /ListTranscriptionJobsCommand/);
  assert.doesNotMatch(transcribeMigrationSource, /--subject/);
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

  const clickAnalysis = shortLinksClicksMigration._internal.analyzeTables([
    { slug: 'Alpha', entityType: 'link', clicks: 5 },
    { slug: 'Bravo', entityType: 'link', clicks: 2 },
    { slug: 'Charlie', entityType: 'link' },
    { slug: '__slug_lower__/alpha', entityType: 'slugReservation' }
  ], [
    {
      slug: 'Alpha',
      clickId: '2026-07-10T10:00:00.000Z#a1',
      clickedAt: '2026-07-10T10:00:00.000Z',
      expiresAt: 1_800_000_000
    },
    {
      slug: 'Alpha',
      clickId: '2026-07-11T10:00:00.000Z#a2',
      entityType: 'clickEvent',
      clickedAt: '2026-07-11T10:00:00.000Z'
    },
    {
      slug: 'Bravo',
      clickId: '2026-07-10T11:00:00.000Z#b1',
      clickedAt: '2026-07-10T11:00:00.000Z',
      expiresAt: 1_800_000_001
    },
    {
      slug: 'Bravo',
      clickId: '2026-07-11T11:00:00.000Z#b2',
      clickedAt: '2026-07-11T11:00:00.000Z'
    },
    {
      slug: 'Orphan',
      clickId: '2026-07-09T11:00:00.000Z#o1',
      clickedAt: '2026-07-09T11:00:00.000Z',
      expiresAt: 1_800_000_002
    }
  ]);
  assert.equal(clickAnalysis.summary.liveLinks, 3);
  assert.equal(clickAnalysis.summary.detailedEvents, 5);
  assert.equal(clickAnalysis.summary.orphanEventSlugs, 1);
  assert.equal(clickAnalysis.summary.ttlRemovalsPlanned, 3);
  assert.equal(clickAnalysis.summary.baselineCreatesPlanned, 3);
  assert.equal(clickAnalysis.summary.historicalClicksRepresented, 3);
  assert.equal(clickAnalysis.summary.recordedClicksRepresented, 4);
  assert.equal(clickAnalysis.summary.aggregateClicksRepresented, 7);
  assert.deepEqual(
    clickAnalysis.baselinePlan.find(entry => entry.link.slug === 'Alpha').desired,
    { historicalClicks: 3, recordedEventCount: 2, aggregateClicks: 5 }
  );
  assert.deepEqual(
    clickAnalysis.baselinePlan.find(entry => entry.link.slug === 'Charlie').desired,
    { historicalClicks: 0, recordedEventCount: 0, aggregateClicks: 0 }
  );

  const reconciledAt = '2026-07-11T12:00:00.000Z';
  const idempotentClickAnalysis = shortLinksClicksMigration._internal.analyzeTables([
    { slug: 'Alpha', entityType: 'link', clicks: 5 }
  ], [
    {
      slug: 'Alpha',
      clickId: '2026-07-10T10:00:00.000Z#a1',
      clickedAt: '2026-07-10T10:00:00.000Z'
    },
    {
      slug: 'Alpha',
      clickId: '2026-07-11T10:00:00.000Z#a2',
      clickedAt: '2026-07-11T10:00:00.000Z'
    },
    {
      slug: 'Alpha',
      clickId: shortLinksClicksMigration.BASELINE_CLICK_ID,
      entityType: shortLinksClicksMigration.BASELINE_ENTITY_TYPE,
      historicalClicks: 3,
      recordedEventCount: 2,
      aggregateClicks: 5,
      reconciledAt
    }
  ]);
  assert.equal(idempotentClickAnalysis.summary.baselinesUnchanged, 1);
  assert.equal(idempotentClickAnalysis.baselinePlan.length, 0);

  const runtimeBaselineAnalysis = shortLinksClicksMigration._internal.analyzeTables([
    { slug: 'Alpha', entityType: 'link', clicks: 5 }
  ], [{
    slug: 'Alpha',
    clickId: shortLinksClicksMigration.BASELINE_CLICK_ID,
    entityType: shortLinksClicksMigration.BASELINE_ENTITY_TYPE,
    historicalClicks: 5,
    recordedEventCount: 0,
    aggregateClicks: 5,
    createdAt: reconciledAt,
    updatedAt: reconciledAt
  }]);
  assert.equal(runtimeBaselineAnalysis.summary.invalidRecords, 0);
  assert.equal(runtimeBaselineAnalysis.summary.baselineUpdatesPlanned, 1);
  assert.equal(
    shortLinksClicksMigration._internal.classifyBaseline({
      slug: 'Alpha',
      clickId: shortLinksClicksMigration.BASELINE_CLICK_ID,
      entityType: shortLinksClicksMigration.BASELINE_ENTITY_TYPE,
      historicalClicks: 5,
      recordedEventCount: 0,
      aggregateClicks: 5,
      destination: 'https://example.com'
    }).valid,
    false
  );

  const excessEvents = shortLinksClicksMigration._internal.analyzeTables([
    { slug: 'Alpha', clicks: 0 }
  ], [{
    slug: 'Alpha',
    clickId: '2026-07-11T10:00:00.000Z#a1',
    clickedAt: '2026-07-11T10:00:00.000Z'
  }]);
  assert.equal(excessEvents.summary.aggregateMismatches, 1);
  assert.throws(
    () => shortLinksClicksMigration._internal.assertSafeAnalysis(excessEvents),
    err => err?.code === 'PREFLIGHT_FAILED'
  );

  const fabricatedBaselineDetail = shortLinksClicksMigration._internal.analyzeTables([
    { slug: 'Alpha', clicks: 1 }
  ], [{
    slug: 'Alpha',
    clickId: shortLinksClicksMigration.BASELINE_CLICK_ID,
    entityType: shortLinksClicksMigration.BASELINE_ENTITY_TYPE,
    historicalClicks: 1,
    recordedEventCount: 0,
    aggregateClicks: 1,
    reconciledAt,
    clickedAt: reconciledAt
  }]);
  assert.equal(fabricatedBaselineDetail.summary.invalidRecords, 1);

  let clickTtlCommand = null;
  await shortLinksClicksMigration._internal.applyTtlRemoval({
    async send(command){
      clickTtlCommand = command;
    }
  }, 'clicks-table', clickAnalysis.ttlPlan[0]);
  assert.equal(clickTtlCommand.input.UpdateExpression, 'REMOVE #expiresAt');
  assert.match(clickTtlCommand.input.ConditionExpression, /#expiresAt = :priorExpiresAt/);

  let baselineCommand = null;
  await shortLinksClicksMigration._internal.applyBaseline({
    async send(command){
      baselineCommand = command;
    }
  }, 'links-table', 'clicks-table', clickAnalysis.baselinePlan[0], reconciledAt);
  assert.equal(baselineCommand.input.TransactItems.length, 2);
  assert.match(
    baselineCommand.input.TransactItems[0].ConditionCheck.ConditionExpression,
    /#clicks = :priorLinkClicks/
  );
  assert.equal(
    baselineCommand.input.TransactItems[1].Update.ExpressionAttributeValues[':historicalClicks'],
    3
  );
  assert.equal(
    baselineCommand.input.TransactItems[1].Update.Key.clickId,
    shortLinksClicksMigration.BASELINE_CLICK_ID
  );

  assert.equal(
    shortLinksClicksMigration._internal.ttlApplyAction(
      { status: 'ENABLED', attribute: 'expiresAt' },
      { apply: true, 'disable-ttl': true }
    ),
    'request-disable'
  );
  assert.equal(
    shortLinksClicksMigration._internal.ttlApplyAction(
      { status: 'DISABLED', attribute: '' },
      { apply: true }
    ),
    'ready'
  );
  assert.equal(
    shortLinksClicksMigration._internal.ttlApplyAction(
      { status: 'DISABLING', attribute: 'expiresAt' },
      { apply: true }
    ),
    'apply-during-disable'
  );
  assert.throws(
    () => shortLinksClicksMigration._internal.requireTrafficPause({ apply: true }),
    err => err?.code === 'TRAFFIC_PAUSE_REQUIRED'
  );

  let disableCommand = null;
  await shortLinksClicksMigration._internal.requestTtlDisable({
    async send(command){
      disableCommand = command;
    }
  }, 'clicks-table');
  assert.deepEqual(disableCommand.input.TimeToLiveSpecification, {
    AttributeName: 'expiresAt',
    Enabled: false
  });

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

  const backfillNow = Math.floor(Date.UTC(2026, 6, 16, 12, 0, 0) / 1000);
  const backfillJobA = `site-transcribe-${'a'.repeat(64)}`;
  const backfillJobB = `site-transcribe-${'b'.repeat(64)}`;
  const backfillJobC = `site-transcribe-${'c'.repeat(64)}`;
  const backfillJobD = `site-transcribe-${'d'.repeat(64)}`;
  const backfillConfig = {
    subject: 'owner-123',
    bucket: 'private-transcribe-bucket',
    prefix: 'tools-transcribe/',
    retentionDays: 90,
    completedAfter: backfillNow - 4 * 86400,
    completedBefore: backfillNow + 3600,
    expectedCount: 2,
    nowSeconds: backfillNow
  };
  const existingHistoryKey = `tools-transcribe/${backfillConfig.subject}/history/${backfillJobB}.txt`;
  const backfillAnalysis = transcribeHistoryMigration._internal.classifyRunItems([
    {
      pk: `TRANSCRIBE#${backfillConfig.subject}`,
      sk: `RUN#${backfillJobA}`,
      entityType: 'transcribe_run',
      state: 'COMPLETED',
      jobName: backfillJobA,
      quoteHash: '1'.repeat(64),
      terminalAt: backfillNow - 2 * 86400
    },
    {
      pk: `TRANSCRIBE#${backfillConfig.subject}`,
      sk: `RUN#${backfillJobB}`,
      entityType: 'transcribe_run',
      state: 'COMPLETED',
      jobName: backfillJobB,
      quoteHash: '2'.repeat(64),
      terminalAt: backfillNow - 86400,
      historySk: `HISTORY#0000000000001#${backfillJobB}`,
      transcriptObjectKey: existingHistoryKey,
      transcriptSha256: '3'.repeat(64),
      historyCreatedAt: (backfillNow - 86400) * 1000,
      historyExpiresAt: backfillNow + 89 * 86400
    },
    {
      pk: `TRANSCRIBE#${backfillConfig.subject}`,
      sk: `RUN#${backfillJobC}`,
      entityType: 'transcribe_run',
      state: 'COMPLETED',
      jobName: backfillJobC,
      quoteHash: '4'.repeat(64),
      terminalAt: backfillNow - 3600,
      historyDeletedAt: backfillNow - 1800
    },
    {
      pk: `TRANSCRIBE#${backfillConfig.subject}`,
      sk: `RUN#${backfillJobD}`,
      entityType: 'transcribe_run',
      state: 'COMPLETED',
      jobName: backfillJobD,
      quoteHash: '5'.repeat(64),
      terminalAt: backfillNow - 8 * 86400
    },
    {
      pk: `TRANSCRIBE#${backfillConfig.subject}`,
      sk: 'RUN#refunded',
      entityType: 'transcribe_run',
      state: 'REFUNDED',
      jobName: 'refunded',
      terminalAt: backfillNow - 1800
    }
  ], backfillConfig);
  assert.equal(backfillAnalysis.summary.queried, 5);
  assert.equal(backfillAnalysis.summary.completed, 4);
  assert.equal(backfillAnalysis.summary.candidates, 1);
  assert.equal(backfillAnalysis.summary.alreadyBackfilled, 1);
  assert.equal(backfillAnalysis.summary.deleted, 1);
  assert.equal(backfillAnalysis.summary.outsideWindow, 1);
  assert.equal(backfillAnalysis.summary.nonCompleted, 1);
  assert.equal(backfillAnalysis.summary.invalid, 0);
  assert.equal(backfillAnalysis.summary.eligible, 2);
  transcribeHistoryMigration._internal.assertSafeAnalysis(backfillAnalysis, backfillConfig);
  assert.throws(
    () => transcribeHistoryMigration._internal.assertSafeAnalysis(
      backfillAnalysis,
      { ...backfillConfig, expectedCount: 3 }
    ),
    err => err?.code === 'EXPECTED_COUNT_MISMATCH'
  );
  assert.equal(
    transcribeHistoryMigration._internal.deriveFilenameFromMediaUri(
      `s3://${backfillConfig.bucket}/${backfillConfig.prefix}${backfillConfig.subject}/1784100000000-${'f'.repeat(20)}-meeting_audio.mp3`,
      backfillConfig
    ),
    'meeting_audio.mp3'
  );
  assert.throws(
    () => transcribeHistoryMigration._internal.deriveFilenameFromMediaUri(
      `s3://${backfillConfig.bucket}/${backfillConfig.prefix}other-owner/1784100000000-${'f'.repeat(20)}-meeting_audio.mp3`,
      backfillConfig
    ),
    err => err?.code === 'OWNERSHIP_MISMATCH'
  );
  const resolvedBackfill = transcribeHistoryMigration._internal.resolveBackfillConfig({
    environment: 'production',
    table: 'tools-production-table',
    bucket: 'private-transcribe-bucket',
    region: 'us-east-2',
    'expected-count': '2',
    'completed-after': '2026-07-13T00:00:00.000Z',
    'completed-before': '2026-07-17T00:00:00.000Z'
  }, {
    TRANSCRIBE_BACKFILL_SUB: backfillConfig.subject
  }, backfillNow);
  assert.equal(resolvedBackfill.subject, backfillConfig.subject);
  assert.equal(resolvedBackfill.prefix, 'tools-transcribe/');
  assert.equal(resolvedBackfill.expectedCount, 2);
  assert.throws(
    () => transcribeHistoryMigration._internal.resolveBackfillConfig({
      environment: 'production',
      table: 'tools-production-table',
      bucket: 'private-transcribe-bucket',
      region: 'us-east-2',
      'expected-count': '2',
      'completed-after': '2026-07-13T00:00:00.000Z',
      'completed-before': '2026-07-17T00:00:00.000Z'
    }, {}, backfillNow),
    err => err?.code === 'SUBJECT_REQUIRED'
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
