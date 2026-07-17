#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const {
  DeleteObjectCommand,
  GetObjectCommand,
  GetObjectTaggingCommand,
  PutObjectCommand,
  S3Client
} = require('@aws-sdk/client-s3');
const {
  GetTranscriptionJobCommand,
  TranscribeClient
} = require('@aws-sdk/client-transcribe');
const {
  GetCommand,
  QueryCommand
} = require('@aws-sdk/lib-dynamodb');
const transcribeEndpoint = require('../../api/_lib/tools-endpoints/transcribe');
const {
  getRun,
  getTranscriptHistory,
  persistTranscriptHistoryMetadata
} = require('../../api/_lib/transcribe-ledger');
const {
  assertTableKeySchema,
  createClients,
  getTtlStatus,
  migrationError,
  parseOptionArgs,
  readEnv,
  requireApplyGuards,
  resolveTarget,
  safeError,
  targetHash
} = require('./_shared');

const PREFIX = 'transcribe-history-backfill';
const DAY_SECONDS = 24 * 60 * 60;
const MAX_GUARD_WINDOW_DAYS = 7;
const MAX_BACKFILL_AGE_DAYS = 14;
const DEFAULT_HISTORY_RETENTION_DAYS = 90;
const DEFAULT_TRANSCRIPT_FETCH_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_TRANSCRIPT_FETCH_BYTES = 25 * 1024 * 1024;
const DEFAULT_MAX_TRANSCRIPT_BYTES = 3 * 1024 * 1024;
const JOB_NAME_PATTERN = /^site-transcribe-[a-f0-9]{64}$/;
const UPLOAD_NAME_PATTERN = /^\d{13}-[a-f0-9]{20}-(.+)$/;
const SUBJECT_PATTERN = /^[A-Za-z0-9-]{1,128}$/;
const HISTORY_FIELDS = [
  'historySk',
  'transcriptObjectKey',
  'transcriptSha256',
  'historyCreatedAt',
  'historyExpiresAt'
];
const {
  analyzeTranscriptCoverage,
  extractTranscript,
  fetchTranscriptData,
  safeFilename,
  transcriptHistoryObjectKey
} = transcribeEndpoint._internal;

function help(){
  return [
    'Backfill private transcript history for a narrow, verified set of completed jobs.',
    '',
    'The Cognito subject must be provided through TRANSCRIBE_BACKFILL_SUB; it is never',
    'accepted as a command-line argument or printed. Dry runs fetch and validate every',
    'candidate transcript but do not write S3 or DynamoDB.',
    '',
    'Dry run:',
    '  npm run migrate:transcribe-history -- --environment production --table <table> --bucket <bucket> --region <region> --expected-count <count> --completed-after <ISO> --completed-before <ISO>',
    '',
    'Apply after reviewing the dry run and confirming a current backup:',
    '  npm run migrate:transcribe-history -- --environment production --table <table> --bucket <bucket> --region <region> --expected-count <count> --completed-after <ISO> --completed-before <ISO> --apply --backup-confirmed',
    '',
    'Environment fallbacks:',
    '  TOOLS_DDB_TABLE_PRODUCTION, TOOLS_DDB_TABLE_PREVIEW, TOOLS_DDB_TABLE',
    '  TRANSCRIBE_UPLOAD_BUCKET_PRODUCTION, TRANSCRIBE_UPLOAD_BUCKET_PREVIEW, TRANSCRIBE_UPLOAD_BUCKET',
    '  TRANSCRIBE_UPLOAD_PREFIX, TRANSCRIBE_HISTORY_RETENTION_DAYS, AWS_REGION',
    '',
    'Options: --environment, --table, --bucket, --region, --prefix, --retention-days,',
    '  --expected-count, --completed-after, --completed-before, --apply,',
    '  --backup-confirmed, --help'
  ].join('\n');
}

function integerOption(value, name, fallback, min, max){
  if ((typeof value === 'undefined' || value === '') && typeof fallback !== 'undefined') return fallback;
  const parsed = Number.parseInt(String(value || ''), 10);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw migrationError('ARG_INVALID', `${name} must be an integer from ${min} to ${max}.`);
  }
  return parsed;
}

function timestampOption(value, name){
  const raw = String(value || '').trim();
  const timestampMs = Date.parse(raw);
  if (!raw || !Number.isFinite(timestampMs)) {
    throw migrationError('ARG_INVALID', `${name} must be an ISO-8601 timestamp.`);
  }
  return Math.floor(timestampMs / 1000);
}

function normalizeBucket(value){
  const bucket = String(value || '').trim();
  if (!/^(?=.{3,63}$)(?!\d+\.\d+\.\d+\.\d+$)[a-z0-9](?:[a-z0-9.-]*[a-z0-9])$/.test(bucket) || bucket.includes('..')) {
    throw migrationError('BUCKET_REQUIRED', 'Provide a valid S3 bucket name with --bucket or the environment fallback.');
  }
  return bucket;
}

function normalizePrefix(value){
  const raw = String(value || 'tools-transcribe/').trim().replace(/\\/g, '/');
  if (!raw || raw.startsWith('/') || raw.includes('..') || raw.includes('//') || !/^[A-Za-z0-9/_-]+\/?$/.test(raw)) {
    throw migrationError('PREFIX_INVALID', 'The upload prefix is invalid.');
  }
  return raw.endsWith('/') ? raw : `${raw}/`;
}

function resolveBucket(options, env){
  const environment = String(options.environment || '').trim().toUpperCase();
  return normalizeBucket(
    options.bucket ||
    readEnv(`TRANSCRIBE_UPLOAD_BUCKET_${environment}`, env) ||
    readEnv('TRANSCRIBE_UPLOAD_BUCKET', env)
  );
}

function resolveBackfillConfig(options, env = process.env, nowSeconds = Math.floor(Date.now() / 1000)){
  const target = resolveTarget({
    ...options,
    env,
    tableBaseEnv: 'TOOLS_DDB_TABLE',
    regionEnvKeys: ['TOOLS_AWS_REGION', 'AWS_REGION', 'AWS_DEFAULT_REGION']
  });
  const subject = readEnv('TRANSCRIBE_BACKFILL_SUB', env);
  if (!SUBJECT_PATTERN.test(subject)) {
    throw migrationError('SUBJECT_REQUIRED', 'Set TRANSCRIBE_BACKFILL_SUB to the exact Cognito subject before running this migration.');
  }
  const bucket = resolveBucket(options, env);
  const prefix = normalizePrefix(options.prefix || readEnv('TRANSCRIBE_UPLOAD_PREFIX', env));
  const retentionDays = integerOption(
    options['retention-days'] || readEnv('TRANSCRIBE_HISTORY_RETENTION_DAYS', env),
    '--retention-days',
    DEFAULT_HISTORY_RETENTION_DAYS,
    1,
    365
  );
  const expectedCount = integerOption(options['expected-count'], '--expected-count', undefined, 1, 1000);
  const completedAfter = timestampOption(options['completed-after'], '--completed-after');
  const completedBefore = timestampOption(options['completed-before'], '--completed-before');
  if (completedAfter >= completedBefore) {
    throw migrationError('DATE_GUARD_INVALID', '--completed-after must be earlier than --completed-before.');
  }
  if (completedBefore - completedAfter > MAX_GUARD_WINDOW_DAYS * DAY_SECONDS) {
    throw migrationError('DATE_GUARD_INVALID', `The guarded completion window cannot exceed ${MAX_GUARD_WINDOW_DAYS} days.`);
  }
  if (completedAfter < nowSeconds - MAX_BACKFILL_AGE_DAYS * DAY_SECONDS || completedBefore > nowSeconds + DAY_SECONDS) {
    throw migrationError(
      'DATE_GUARD_INVALID',
      `The guarded completion window must cover only the last ${MAX_BACKFILL_AGE_DAYS} days and cannot extend more than one day ahead.`
    );
  }
  const ttlAttribute = String(readEnv('TRANSCRIBE_LEDGER_TTL_ATTRIBUTE', env) || 'ttl').trim();
  if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(ttlAttribute)) {
    throw migrationError('TTL_ATTRIBUTE_INVALID', 'TRANSCRIBE_LEDGER_TTL_ATTRIBUTE is invalid.');
  }
  return {
    ...target,
    subject,
    bucket,
    prefix,
    retentionDays,
    expectedCount,
    completedAfter,
    completedBefore,
    ttlAttribute,
    nowSeconds
  };
}

function opaqueOwnerHash(subject){
  return crypto.createHash('sha256').update(String(subject || ''), 'utf8').digest('hex').slice(0, 12);
}

function hasOwn(record, field){
  return Object.prototype.hasOwnProperty.call(record || {}, field);
}

function validHistoryLink(record, config){
  const expectedKey = transcriptHistoryObjectKey(config, config.subject, record.jobName);
  return Boolean(
    String(record.historySk || '').startsWith('HISTORY#') &&
    String(record.transcriptObjectKey || '') === expectedKey &&
    /^[a-f0-9]{64}$/.test(String(record.transcriptSha256 || '')) &&
    Number.isSafeInteger(Number(record.historyCreatedAt)) &&
    Number(record.historyCreatedAt) > 0 &&
    Number.isSafeInteger(Number(record.historyExpiresAt)) &&
    Number(record.historyExpiresAt) > config.nowSeconds
  );
}

function classifyRunItems(items, config){
  const summary = {
    queried: items.length,
    completed: 0,
    outsideWindow: 0,
    nonCompleted: 0,
    deleted: 0,
    expired: 0,
    alreadyBackfilled: 0,
    candidates: 0,
    invalid: 0,
    eligible: 0
  };
  const candidates = [];
  const existing = [];
  for (const record of items) {
    if (String(record?.state || '') !== 'COMPLETED') {
      summary.nonCompleted += 1;
      continue;
    }
    summary.completed += 1;
    const terminalAt = Number(record.terminalAt);
    if (!Number.isSafeInteger(terminalAt) || terminalAt <= 0) {
      summary.invalid += 1;
      continue;
    }
    if (terminalAt < config.completedAfter || terminalAt >= config.completedBefore) {
      summary.outsideWindow += 1;
      continue;
    }
    const jobName = String(record.jobName || '');
    const structurallyValid = record.pk === `TRANSCRIBE#${config.subject}` &&
      record.sk === `RUN#${jobName}` &&
      record.entityType === 'transcribe_run' &&
      JOB_NAME_PATTERN.test(jobName) &&
      /^[a-f0-9]{64}$/.test(String(record.quoteHash || ''));
    if (!structurallyValid) {
      summary.invalid += 1;
      continue;
    }
    if (Number(record.historyDeletedAt) > 0) {
      summary.deleted += 1;
      continue;
    }
    if (terminalAt + config.retentionDays * DAY_SECONDS <= config.nowSeconds) {
      summary.expired += 1;
      continue;
    }
    const historyFieldCount = HISTORY_FIELDS.filter(field => hasOwn(record, field)).length;
    if (historyFieldCount === 0) {
      candidates.push(record);
      summary.candidates += 1;
      summary.eligible += 1;
      continue;
    }
    if (historyFieldCount === HISTORY_FIELDS.length && validHistoryLink(record, config)) {
      existing.push(record);
      summary.alreadyBackfilled += 1;
      summary.eligible += 1;
      continue;
    }
    summary.invalid += 1;
  }
  candidates.sort((a, b) => Number(a.terminalAt) - Number(b.terminalAt) || String(a.jobName).localeCompare(String(b.jobName)));
  existing.sort((a, b) => Number(a.terminalAt) - Number(b.terminalAt) || String(a.jobName).localeCompare(String(b.jobName)));
  return { summary, candidates, existing };
}

function assertSafeAnalysis(analysis, config){
  if (analysis.summary.invalid > 0) {
    throw migrationError('PREFLIGHT_FAILED', 'Invalid completed run records were found inside the guarded window; no writes were attempted.');
  }
  if (analysis.summary.eligible !== config.expectedCount) {
    throw migrationError(
      'EXPECTED_COUNT_MISMATCH',
      `The guarded set contains ${analysis.summary.eligible} eligible runs, not the required ${config.expectedCount}; no writes were attempted.`
    );
  }
}

async function queryRunItems(client, tableName, subject){
  const items = [];
  let lastEvaluatedKey;
  do {
    const result = await client.send(new QueryCommand({
      TableName: tableName,
      KeyConditionExpression: '#pk = :pk AND begins_with(#sk, :prefix)',
      ExpressionAttributeNames: { '#pk': 'pk', '#sk': 'sk' },
      ExpressionAttributeValues: {
        ':pk': `TRANSCRIBE#${subject}`,
        ':prefix': 'RUN#'
      },
      ConsistentRead: true,
      ExclusiveStartKey: lastEvaluatedKey
    }));
    items.push(...(Array.isArray(result?.Items) ? result.Items : []));
    lastEvaluatedKey = result?.LastEvaluatedKey || undefined;
  } while (lastEvaluatedKey);
  return items;
}

function deriveFilenameFromMediaUri(mediaUri, config){
  const ownerRoot = `s3://${config.bucket}/${config.prefix}${config.subject}/`;
  const value = String(mediaUri || '');
  if (!value.startsWith(ownerRoot)) {
    throw migrationError('OWNERSHIP_MISMATCH', 'A transcription job does not point to the guarded owner upload prefix.');
  }
  const relativeKey = value.slice(ownerRoot.length);
  if (!relativeKey || relativeKey.includes('/')) {
    throw migrationError('MEDIA_KEY_INVALID', 'A transcription job has an invalid upload key.');
  }
  const match = UPLOAD_NAME_PATTERN.exec(relativeKey);
  if (!match) {
    throw migrationError('MEDIA_KEY_INVALID', 'A transcription job upload key does not match the current safe naming format.');
  }
  return safeFilename(match[1]);
}

async function inspectCandidate(record, clients, config){
  const result = await clients.transcribe.send(new GetTranscriptionJobCommand({
    TranscriptionJobName: record.jobName
  }));
  const job = result?.TranscriptionJob;
  if (
    !job ||
    String(job.TranscriptionJobName || '') !== record.jobName ||
    String(job.TranscriptionJobStatus || '').toUpperCase() !== 'COMPLETED'
  ) {
    throw migrationError('JOB_STATE_MISMATCH', 'A guarded transcription job is missing or is no longer completed.');
  }
  const filename = deriveFilenameFromMediaUri(job?.Media?.MediaFileUri, config);
  const transcriptData = await fetchTranscriptData(job?.Transcript?.TranscriptFileUri, {
    transcriptFetchTimeoutMs: DEFAULT_TRANSCRIPT_FETCH_TIMEOUT_MS,
    maxTranscriptFetchBytes: DEFAULT_MAX_TRANSCRIPT_FETCH_BYTES
  });
  const transcript = extractTranscript(transcriptData);
  const body = Buffer.from(transcript, 'utf8');
  if (body.length > DEFAULT_MAX_TRANSCRIPT_BYTES) {
    throw migrationError('TRANSCRIPT_OUTPUT_TOO_LARGE', 'A guarded transcript is too large for private history storage.');
  }
  const coverage = analyzeTranscriptCoverage(transcriptData, 0);
  return {
    record,
    filename,
    transcript,
    body,
    bytes: body.length,
    sha256: crypto.createHash('sha256').update(body).digest('hex'),
    objectKey: transcriptHistoryObjectKey(config, config.subject, record.jobName),
    coverage
  };
}

async function boundedBody(body, maxBytes){
  if (!body) throw migrationError('HISTORY_OBJECT_INVALID', 'A private history object has no body.');
  if (Buffer.isBuffer(body) || body instanceof Uint8Array) {
    const result = Buffer.from(body);
    if (result.length > maxBytes) throw migrationError('HISTORY_OBJECT_INVALID', 'A private history object is too large.');
    return result;
  }
  if (typeof body[Symbol.asyncIterator] === 'function') {
    const chunks = [];
    let bytes = 0;
    for await (const rawChunk of body) {
      const chunk = Buffer.from(rawChunk);
      bytes += chunk.length;
      if (bytes > maxBytes) throw migrationError('HISTORY_OBJECT_INVALID', 'A private history object is too large.');
      chunks.push(chunk);
    }
    return Buffer.concat(chunks, bytes);
  }
  if (typeof body.transformToByteArray === 'function') {
    const result = Buffer.from(await body.transformToByteArray());
    if (result.length > maxBytes) throw migrationError('HISTORY_OBJECT_INVALID', 'A private history object is too large.');
    return result;
  }
  throw migrationError('HISTORY_OBJECT_INVALID', 'A private history object could not be read safely.');
}

function isMissingObject(err){
  const name = String(err?.name || err?.code || '');
  const status = Number(err?.$metadata?.httpStatusCode);
  return ['NoSuchKey', 'NotFound', 'NoSuchBucket'].includes(name) || status === 404;
}

function isPreconditionFailure(err){
  const name = String(err?.name || err?.code || '');
  return name === 'PreconditionFailed' || Number(err?.$metadata?.httpStatusCode) === 412;
}

async function verifyHistoryObject(s3, config, objectKey, expectedSha256, expectedBytes){
  const result = await s3.send(new GetObjectCommand({ Bucket: config.bucket, Key: objectKey }));
  const declaredBytes = Number(result?.ContentLength);
  if (Number.isFinite(declaredBytes) && declaredBytes !== expectedBytes) {
    throw migrationError('HISTORY_OBJECT_CONFLICT', 'An existing private history object has an unexpected size.');
  }
  const body = await boundedBody(result?.Body, DEFAULT_MAX_TRANSCRIPT_BYTES);
  const actualSha256 = crypto.createHash('sha256').update(body).digest('hex');
  const metadataSha256 = String(result?.Metadata?.sha256 || '').toLowerCase();
  if (body.length !== expectedBytes || actualSha256 !== expectedSha256 || metadataSha256 !== expectedSha256) {
    throw migrationError('HISTORY_OBJECT_CONFLICT', 'An existing private history object failed its integrity check.');
  }
  if (String(result?.ContentType || '').toLowerCase() !== 'text/plain; charset=utf-8') {
    throw migrationError('HISTORY_OBJECT_CONFLICT', 'An existing private history object has an unexpected content type.');
  }
  if (String(result?.CacheControl || '').toLowerCase() !== 'private, no-store') {
    throw migrationError('HISTORY_OBJECT_CONFLICT', 'An existing private history object has an unexpected cache policy.');
  }
  const tagsResult = await s3.send(new GetObjectTaggingCommand({ Bucket: config.bucket, Key: objectKey }));
  const tags = new Map((tagsResult?.TagSet || []).map(tag => [String(tag.Key || ''), String(tag.Value || '')]));
  if (tags.get('tool') !== 'amazon-transcribe' || tags.get('retention') !== 'history') {
    throw migrationError('HISTORY_OBJECT_CONFLICT', 'An existing private history object has an unexpected retention policy.');
  }
  return true;
}

async function historyObjectExists(s3, config, item){
  try {
    await verifyHistoryObject(s3, config, item.objectKey, item.sha256, item.bytes);
    return true;
  } catch (err) {
    if (isMissingObject(err)) return false;
    throw err;
  }
}

async function putHistoryObject(s3, config, item){
  if (await historyObjectExists(s3, config, item)) return { created: false };
  try {
    await s3.send(new PutObjectCommand({
      Bucket: config.bucket,
      Key: item.objectKey,
      Body: item.body,
      ContentLength: item.bytes,
      ContentType: 'text/plain; charset=utf-8',
      CacheControl: 'private, no-store',
      Tagging: 'tool=amazon-transcribe&retention=history',
      Metadata: { sha256: item.sha256 },
      IfNoneMatch: '*'
    }));
    await verifyHistoryObject(s3, config, item.objectKey, item.sha256, item.bytes);
    return { created: true };
  } catch (err) {
    if (isPreconditionFailure(err)) {
      await verifyHistoryObject(s3, config, item.objectKey, item.sha256, item.bytes);
      return { created: false };
    }
    throw err;
  }
}

function ledgerConfig(config){
  return {
    region: config.region,
    ledgerTable: config.tableName,
    ledgerTtlAttribute: config.ttlAttribute,
    ledgerEnabled: true,
    historyRetentionDays: config.retentionDays,
    awsCredentialsCacheKey: `history-backfill:${targetHash(config.tableName)}:${opaqueOwnerHash(config.subject)}`
  };
}

async function metadataMatches(item, config){
  const current = await getRun({
    config: ledgerConfig(config),
    sub: config.subject,
    jobName: item.record.jobName
  });
  return Boolean(
    current &&
    String(current.historySk || '').startsWith('HISTORY#') &&
    String(current.transcriptObjectKey || '') === item.objectKey &&
    String(current.transcriptSha256 || '') === item.sha256 &&
    Number(current.transcriptBytes) === item.bytes
  );
}

async function verifyHistoryMetadata(documentClient, item, config){
  const history = await getTranscriptHistory({
    config: ledgerConfig(config),
    sub: config.subject,
    jobName: item.record.jobName
  });
  if (
    !history ||
    history.transcriptObjectKey !== item.objectKey ||
    history.transcriptSha256 !== item.sha256 ||
    Number(history.transcriptBytes) !== item.bytes
  ) {
    throw migrationError('POST_WRITE_VERIFY_FAILED', 'Committed private history metadata could not be verified.');
  }
  const indexResult = await documentClient.send(new GetCommand({
    TableName: config.tableName,
    Key: { pk: `TRANSCRIBE#${config.subject}`, sk: history.historySk },
    ConsistentRead: true
  }));
  if (indexResult?.Item?.entityType !== 'transcribe_history' || indexResult?.Item?.jobName !== item.record.jobName) {
    throw migrationError('POST_WRITE_VERIFY_FAILED', 'Committed private history index metadata could not be verified.');
  }
  await verifyHistoryObject(config.s3Client, config, item.objectKey, item.sha256, item.bytes);
  return history;
}

async function applyCandidate(item, clients, config){
  const objectResult = await putHistoryObject(clients.s3, config, item);
  let result;
  try {
    result = await persistTranscriptHistoryMetadata({
      config: ledgerConfig(config),
      sub: config.subject,
      jobName: item.record.jobName,
      quoteHash: item.record.quoteHash,
      filename: item.filename,
      transcriptObjectKey: item.objectKey,
      transcriptBytes: item.bytes,
      transcriptSha256: item.sha256,
      historyStatus: 'COMPLETED',
      durationSeconds: 0,
      billableSeconds: 0,
      coverageStatus: 'OK',
      transcriptEndSeconds: item.coverage.transcriptEndSeconds,
      transcriptGapSeconds: 0,
      coverageRatio: 1
    });
    if (!result?.persisted) {
      const err = migrationError('HISTORY_PERSIST_REJECTED', 'The guarded run no longer permits private history.');
      err.persistRejected = true;
      throw err;
    }
  } catch (err) {
    const committed = await metadataMatches(item, config).catch(() => false);
    if (committed) return { committed: true, recoveredAfterError: true };
    if (objectResult.created) {
      await clients.s3.send(new DeleteObjectCommand({ Bucket: config.bucket, Key: item.objectKey })).catch(() => {});
    }
    throw err;
  }
  await verifyHistoryMetadata(clients.document, item, { ...config, s3Client: clients.s3 });
  return { committed: true, recoveredAfterError: false };
}

async function verifyExistingHistory(record, clients, config){
  const history = await getTranscriptHistory({
    config: ledgerConfig(config),
    sub: config.subject,
    jobName: record.jobName
  });
  if (!history) throw migrationError('EXISTING_HISTORY_INVALID', 'Existing private history metadata could not be verified.');
  const item = {
    record,
    objectKey: history.transcriptObjectKey,
    sha256: history.transcriptSha256,
    bytes: Number(history.transcriptBytes) || 0
  };
  await verifyHistoryMetadata(clients.document, item, { ...config, s3Client: clients.s3 });
}

async function run(argv = process.argv.slice(2), env = process.env){
  const options = parseOptionArgs(argv, {
    booleanFlags: ['--apply', '--backup-confirmed', '--help'],
    valueFlags: [
      '--environment',
      '--table',
      '--bucket',
      '--region',
      '--prefix',
      '--retention-days',
      '--expected-count',
      '--completed-after',
      '--completed-before'
    ]
  });
  if (options.help) {
    process.stdout.write(`${help()}\n`);
    return { help: true };
  }
  requireApplyGuards(options);
  const config = resolveBackfillConfig(options, env);
  process.stdout.write(
    `[${PREFIX}] mode=${options.apply ? 'apply' : 'dry-run'} environment=${config.environment} ` +
    `region=${config.region} tableHash=${targetHash(config.tableName)} ` +
    `bucketHash=${targetHash(config.bucket)} ownerHash=${opaqueOwnerHash(config.subject)}\n`
  );
  process.stdout.write(
    `[${PREFIX}] guard expected=${config.expectedCount} windowSeconds=${config.completedBefore - config.completedAfter} ` +
    `retentionDays=${config.retentionDays}\n`
  );

  const ddb = createClients(config.region);
  const clients = {
    document: ddb.document,
    s3: new S3Client({ region: config.region }),
    transcribe: new TranscribeClient({ region: config.region })
  };
  const table = await assertTableKeySchema(ddb.base, config.tableName, ['HASH:pk', 'RANGE:sk']);
  const ttl = await getTtlStatus(ddb.base, config.tableName);
  process.stdout.write(
    `[${PREFIX}] tableStatus=${table.status} ttlStatus=${ttl.status || 'UNKNOWN'} ` +
    `ttlAttributeMatches=${ttl.attribute === config.ttlAttribute}\n`
  );
  if (table.status !== 'ACTIVE') {
    throw migrationError('TABLE_NOT_ACTIVE', 'The target table must be active before this migration runs.');
  }
  if (!['ENABLED', 'ENABLING'].includes(ttl.status) || ttl.attribute !== config.ttlAttribute) {
    throw migrationError('TTL_NOT_READY', 'DynamoDB TTL must be enabled on the configured attribute before this migration runs.');
  }

  const records = await queryRunItems(clients.document, config.tableName, config.subject);
  const analysis = classifyRunItems(records, config);
  process.stdout.write(`[${PREFIX}] ledgerSummary ${JSON.stringify(analysis.summary)}\n`);
  assertSafeAnalysis(analysis, config);

  for (const record of analysis.existing) {
    await verifyExistingHistory(record, clients, config);
  }

  const plan = [];
  let transcriptBytes = 0;
  for (const record of analysis.candidates) {
    const item = await inspectCandidate(record, clients, config);
    plan.push(item);
    transcriptBytes += item.bytes;
  }
  process.stdout.write(
    `[${PREFIX}] transcriptPreflight validated=${plan.length} existingVerified=${analysis.existing.length} ` +
    `totalUtf8Bytes=${transcriptBytes}\n`
  );

  if (!options.apply) {
    process.stdout.write(
      `[${PREFIX}] dry-run complete; rerun with --apply --backup-confirmed to write ${plan.length} private histories.\n`
    );
    return { analysis, planned: plan.length, transcriptBytes };
  }

  let applied = 0;
  let recoveredAfterError = 0;
  for (const item of plan) {
    const result = await applyCandidate(item, clients, config);
    applied += 1;
    if (result.recoveredAfterError) recoveredAfterError += 1;
    process.stdout.write(`[${PREFIX}] applyProgress applied=${applied} total=${plan.length}\n`);
  }
  process.stdout.write(
    `[${PREFIX}] apply complete applied=${applied} alreadyPresent=${analysis.existing.length} ` +
    `recoveredAfterError=${recoveredAfterError}; rerun the dry-run to verify zero pending writes.\n`
  );
  return { analysis, applied, recoveredAfterError, transcriptBytes };
}

if (require.main === module) {
  run().catch(err => {
    process.stderr.write(`[${PREFIX}] failed=${safeError(err)}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  DEFAULT_HISTORY_RETENTION_DAYS,
  _internal: {
    assertSafeAnalysis,
    classifyRunItems,
    deriveFilenameFromMediaUri,
    historyObjectExists,
    normalizeBucket,
    normalizePrefix,
    opaqueOwnerHash,
    putHistoryObject,
    resolveBackfillConfig,
    verifyHistoryObject
  },
  run
};
