'use strict';

const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const {
  DeleteCommand,
  DynamoDBDocumentClient,
  GetCommand,
  TransactWriteCommand,
  UpdateCommand
} = require('@aws-sdk/lib-dynamodb');

const DAY_SECONDS = 24 * 60 * 60;
const RUN_STATE = Object.freeze({
  RESERVED: 'RESERVED',
  RUNNING: 'RUNNING',
  REFUNDED: 'REFUNDED',
  COMPLETED: 'COMPLETED',
  FAILED: 'FAILED',
  MISSING: 'MISSING'
});
const TERMINAL_STATES = new Set([RUN_STATE.REFUNDED, RUN_STATE.COMPLETED, RUN_STATE.FAILED, RUN_STATE.MISSING]);

let cachedClient = null;
let cachedClientKey = '';

function ledgerError(code, message){
  const err = new Error(message);
  err.code = code;
  return err;
}

function getClient(config){
  const credentials = config.awsCredentials || undefined;
  const key = [config.region, config.awsCredentialsCacheKey || 'default'].join(':');
  if (cachedClient && cachedClientKey === key) return cachedClient;
  const base = new DynamoDBClient({
    region: config.region,
    credentials
  });
  cachedClient = DynamoDBDocumentClient.from(base, {
    marshallOptions: { removeUndefinedValues: true }
  });
  cachedClientKey = key;
  return cachedClient;
}

function userPk(sub){
  return `TRANSCRIBE#${String(sub || '').trim()}`;
}

function runKey(sub, jobName){
  return { pk: userPk(sub), sk: `RUN#${jobName}` };
}

function dayKey(sub, day){
  return { pk: userPk(sub), sk: `DAY#${day}` };
}

function globalDayKey(day){
  return { pk: 'TRANSCRIBE#GLOBAL', sk: `DAY#${day}` };
}

function slotKey(sub, slotNumber){
  return { pk: userPk(sub), sk: `SLOT#${String(slotNumber).padStart(3, '0')}` };
}

function utcDay(nowSeconds){
  return new Date(nowSeconds * 1000).toISOString().slice(0, 10);
}

function startOfUtcDaySeconds(day){
  return Math.floor(Date.parse(`${day}T00:00:00.000Z`) / 1000);
}

function isConditionalFailure(err){
  if (err?.name === 'ConditionalCheckFailedException') return true;
  if (err?.name !== 'TransactionCanceledException') return false;
  const reasons = Array.isArray(err?.CancellationReasons) ? err.CancellationReasons : [];
  if (!reasons.length) return false;
  const meaningful = reasons.map(reason => String(reason?.Code || '')).filter(code => code && code !== 'None');
  return meaningful.length > 0 && meaningful.every(code => code === 'ConditionalCheckFailed');
}

function toCostMicros(value){
  const cost = Number(value);
  if (!Number.isFinite(cost) || cost < 0) throw ledgerError('LEDGER_INVALID_COST', 'The transcription cost reservation is invalid.');
  return Math.round(cost * 1_000_000);
}

function isEnabled(config){
  return config?.ledgerEnabled !== false;
}

function ttlItem(config, item, expiresAt){
  return { ...item, [config.ledgerTtlAttribute || 'ttl']: expiresAt };
}

async function getRun({ config, sub, jobName }){
  if (!isEnabled(config)) return null;
  const out = await getClient(config).send(new GetCommand({
    TableName: config.ledgerTable,
    Key: runKey(sub, jobName),
    ConsistentRead: true
  }));
  return out?.Item || null;
}

function ensureMatchingRun(record, quoteHash){
  if (!record) return;
  if (String(record.quoteHash || '') !== String(quoteHash || '')) {
    throw ledgerError('LEDGER_RUN_CONFLICT', 'A conflicting transcription reservation already exists.');
  }
}

function reservationTransaction({ config, sub, jobName, quoteHash, costMicros, day, now, slotNumber, existingRefunded, attemptId }){
  const leaseExpiresAt = now + config.ledgerLeaseSeconds;
  const runExpiresAt = now + (config.ledgerRunRetentionDays * DAY_SECONDS);
  const dayExpiresAt = startOfUtcDaySeconds(day) + ((config.ledgerDayRetentionDays + 1) * DAY_SECONDS);
  const remainingCostMicros = config.ledgerDailyCostLimitMicros - costMicros;
  const remainingFiles = config.ledgerDailyFileLimit - 1;
  const globalRemainingCostMicros = config.ledgerGlobalDailyCostLimitMicros - costMicros;
  const globalRemainingFiles = config.ledgerGlobalDailyFileLimit - 1;
  const runItem = ttlItem(config, {
    ...runKey(sub, jobName),
    entityType: 'transcribe_run',
    jobName,
    quoteHash,
    state: RUN_STATE.RESERVED,
    jobCreated: false,
    day,
    costMicros,
    fileCount: 1,
    slotNumber,
    reservedAt: now,
    updatedAt: now,
    leaseExpiresAt,
    startAttemptId: attemptId,
    startAttemptExpiresAt: now + config.ledgerStartLeaseSeconds
  }, runExpiresAt);

  const runWrite = existingRefunded
    ? {
        Update: {
          TableName: config.ledgerTable,
          Key: runKey(sub, jobName),
          ConditionExpression: '#state = :refunded AND quoteHash = :quoteHash',
          UpdateExpression: 'SET #state = :reserved, jobCreated = :false, #day = :day, costMicros = :cost, fileCount = :one, slotNumber = :slot, reservedAt = :now, updatedAt = :now, leaseExpiresAt = :lease, startAttemptId = :attemptId, startAttemptExpiresAt = :attemptExpiresAt, #ttl = :expiresAt REMOVE terminalAt, terminalStatus, refundedAt, refundReason',
          ExpressionAttributeNames: { '#state': 'state', '#day': 'day', '#ttl': config.ledgerTtlAttribute || 'ttl' },
          ExpressionAttributeValues: {
            ':refunded': RUN_STATE.REFUNDED,
            ':reserved': RUN_STATE.RESERVED,
            ':quoteHash': quoteHash,
            ':false': false,
            ':day': day,
            ':cost': costMicros,
            ':one': 1,
            ':slot': slotNumber,
            ':now': now,
            ':lease': leaseExpiresAt,
            ':attemptId': attemptId,
            ':attemptExpiresAt': now + config.ledgerStartLeaseSeconds,
            ':expiresAt': runExpiresAt
          }
        }
      }
    : {
        Put: {
          TableName: config.ledgerTable,
          Item: runItem,
          ConditionExpression: 'attribute_not_exists(pk) AND attribute_not_exists(sk)'
        }
      };

  return {
    leaseExpiresAt,
    runItem,
    transactItems: [
      runWrite,
      {
        Put: {
          TableName: config.ledgerTable,
          Item: ttlItem(config, {
            ...slotKey(sub, slotNumber),
            entityType: 'transcribe_slot',
            slotNumber,
            ownerJobName: jobName,
            quoteHash,
            acquiredAt: now,
            leaseExpiresAt
          }, leaseExpiresAt + DAY_SECONDS),
          ConditionExpression: 'attribute_not_exists(pk) OR leaseExpiresAt < :now OR ownerJobName = :jobName',
          ExpressionAttributeValues: {
            ':now': now,
            ':jobName': jobName
          }
        }
      },
      {
        Update: {
          TableName: config.ledgerTable,
          Key: dayKey(sub, day),
          ConditionExpression: '(attribute_not_exists(reservedCostMicros) OR reservedCostMicros <= :remainingCost) AND (attribute_not_exists(fileCount) OR fileCount <= :remainingFiles)',
          UpdateExpression: 'SET entityType = :entityType, #day = :day, updatedAt = :now, #ttl = :expiresAt ADD reservedCostMicros :cost, fileCount :one',
          ExpressionAttributeNames: { '#day': 'day', '#ttl': config.ledgerTtlAttribute || 'ttl' },
          ExpressionAttributeValues: {
            ':entityType': 'transcribe_day',
            ':day': day,
            ':now': now,
            ':expiresAt': dayExpiresAt,
            ':remainingCost': remainingCostMicros,
            ':remainingFiles': remainingFiles,
            ':cost': costMicros,
            ':one': 1
          }
        }
      },
      {
        Update: {
          TableName: config.ledgerTable,
          Key: globalDayKey(day),
          ConditionExpression: '(attribute_not_exists(reservedCostMicros) OR reservedCostMicros <= :remainingCost) AND (attribute_not_exists(fileCount) OR fileCount <= :remainingFiles)',
          UpdateExpression: 'SET entityType = :entityType, #day = :day, updatedAt = :now, #ttl = :expiresAt ADD reservedCostMicros :cost, fileCount :one',
          ExpressionAttributeNames: { '#day': 'day', '#ttl': config.ledgerTtlAttribute || 'ttl' },
          ExpressionAttributeValues: {
            ':entityType': 'transcribe_global_day',
            ':day': day,
            ':now': now,
            ':expiresAt': dayExpiresAt,
            ':remainingCost': globalRemainingCostMicros,
            ':remainingFiles': globalRemainingFiles,
            ':cost': costMicros,
            ':one': 1
          }
        }
      }
    ]
  };
}

async function classifyReservationFailure({ config, sub, jobName, quoteHash, day, costMicros }){
  const existing = await getRun({ config, sub, jobName });
  if (existing && String(existing.state || '') !== RUN_STATE.REFUNDED) {
    ensureMatchingRun(existing, quoteHash);
    return { reserved: false, idempotent: true, startAllowed: false, record: existing };
  }

  const [out, globalOut] = await Promise.all([
    getClient(config).send(new GetCommand({
      TableName: config.ledgerTable,
      Key: dayKey(sub, day),
      ConsistentRead: true
    })),
    getClient(config).send(new GetCommand({
      TableName: config.ledgerTable,
      Key: globalDayKey(day),
      ConsistentRead: true
    }))
  ]);
  const record = out?.Item || {};
  const globalRecord = globalOut?.Item || {};
  if ((Number(record.reservedCostMicros) || 0) + costMicros > config.ledgerDailyCostLimitMicros) {
    throw ledgerError('LEDGER_DAILY_COST_LIMIT', 'Your daily transcription cost limit has been reached.');
  }
  if ((Number(record.fileCount) || 0) + 1 > config.ledgerDailyFileLimit) {
    throw ledgerError('LEDGER_DAILY_FILE_LIMIT', 'Your daily transcription file limit has been reached.');
  }
  if ((Number(globalRecord.reservedCostMicros) || 0) + costMicros > config.ledgerGlobalDailyCostLimitMicros) {
    throw ledgerError('LEDGER_GLOBAL_COST_LIMIT', 'The transcription service has reached its daily cost limit.');
  }
  if ((Number(globalRecord.fileCount) || 0) + 1 > config.ledgerGlobalDailyFileLimit) {
    throw ledgerError('LEDGER_GLOBAL_FILE_LIMIT', 'The transcription service has reached its daily file limit.');
  }
  throw ledgerError('LEDGER_CONCURRENCY_LIMIT', 'All transcription slots are busy. Wait for an active job to finish and try again.');
}

async function claimStaleStartAttempt({ config, sub, jobName, quoteHash, attemptId, record, now }){
  const currentExpiry = Number(record?.startAttemptExpiresAt) || 0;
  if (currentExpiry >= now) return null;
  const slotNumber = Number(record?.slotNumber);
  if (!Number.isInteger(slotNumber) || slotNumber <= 0) return null;
  const attemptExpiresAt = now + config.ledgerStartLeaseSeconds;
  const leaseExpiresAt = now + config.ledgerLeaseSeconds;
  try {
    await getClient(config).send(new TransactWriteCommand({
      TransactItems: [
        {
          Update: {
            TableName: config.ledgerTable,
            Key: runKey(sub, jobName),
            ConditionExpression: 'quoteHash = :quoteHash AND #state = :reserved AND (attribute_not_exists(startAttemptExpiresAt) OR startAttemptExpiresAt < :now)',
            UpdateExpression: 'SET startAttemptId = :attemptId, startAttemptExpiresAt = :attemptExpiresAt, leaseExpiresAt = :lease, updatedAt = :now',
            ExpressionAttributeNames: { '#state': 'state' },
            ExpressionAttributeValues: {
              ':quoteHash': quoteHash,
              ':reserved': RUN_STATE.RESERVED,
              ':now': now,
              ':attemptId': attemptId,
              ':attemptExpiresAt': attemptExpiresAt,
              ':lease': leaseExpiresAt
            }
          }
        },
        {
          Update: {
            TableName: config.ledgerTable,
            Key: slotKey(sub, slotNumber),
            ConditionExpression: 'ownerJobName = :jobName',
            UpdateExpression: 'SET leaseExpiresAt = :lease, #ttl = :expiresAt',
            ExpressionAttributeNames: { '#ttl': config.ledgerTtlAttribute || 'ttl' },
            ExpressionAttributeValues: {
              ':jobName': jobName,
              ':lease': leaseExpiresAt,
              ':expiresAt': leaseExpiresAt + DAY_SECONDS
            }
          }
        }
      ]
    }));
    return getRun({ config, sub, jobName });
  } catch (err) {
    if (!isConditionalFailure(err)) throw err;
    return null;
  }
}

async function reserveRun({ config, sub, jobName, quoteHash, costUsd, attemptId }){
  if (!isEnabled(config)) {
    return {
      reserved: true,
      idempotent: false,
      startAllowed: true,
      record: { jobName, quoteHash, state: RUN_STATE.RESERVED, ledgerDisabled: true }
    };
  }
  if (!config.ledgerTable) throw ledgerError('LEDGER_NOT_CONFIGURED', 'The transcription usage ledger is not configured.');

  const now = Math.floor(Date.now() / 1000);
  const day = utcDay(now);
  const costMicros = toCostMicros(costUsd);
  if (costMicros > config.ledgerDailyCostLimitMicros) {
    throw ledgerError('LEDGER_DAILY_COST_LIMIT', 'This transcription exceeds your daily transcription cost limit.');
  }
  if (costMicros > config.ledgerGlobalDailyCostLimitMicros) {
    throw ledgerError('LEDGER_GLOBAL_COST_LIMIT', 'This transcription exceeds the service daily cost limit.');
  }

  let existing = await getRun({ config, sub, jobName });
  ensureMatchingRun(existing, quoteHash);
  if (existing && String(existing.state || '') !== RUN_STATE.REFUNDED) {
    if (String(existing.state || '') === RUN_STATE.RESERVED) {
      const claimed = await claimStaleStartAttempt({ config, sub, jobName, quoteHash, attemptId, record: existing, now });
      if (claimed) return { reserved: false, idempotent: true, startAllowed: true, record: claimed };
    }
    return { reserved: false, idempotent: true, startAllowed: false, record: existing };
  }

  for (let slotNumber = 1; slotNumber <= config.ledgerMaxConcurrent; slotNumber += 1) {
    const transaction = reservationTransaction({
      config,
      sub,
      jobName,
      quoteHash,
      costMicros,
      day,
      now,
      slotNumber,
      existingRefunded: Boolean(existing),
      attemptId
    });
    try {
      await getClient(config).send(new TransactWriteCommand({ TransactItems: transaction.transactItems }));
      return {
        reserved: true,
        idempotent: false,
        startAllowed: true,
        record: transaction.runItem
      };
    } catch (err) {
      if (!isConditionalFailure(err)) throw err;
      existing = await getRun({ config, sub, jobName });
      ensureMatchingRun(existing, quoteHash);
      if (existing && String(existing.state || '') !== RUN_STATE.REFUNDED) {
        return { reserved: false, idempotent: true, startAllowed: false, record: existing };
      }
    }
  }

  return classifyReservationFailure({ config, sub, jobName, quoteHash, day, costMicros });
}

async function markJobCreated({ config, sub, jobName, quoteHash }){
  if (!isEnabled(config)) return null;
  const client = getClient(config);
  const now = Math.floor(Date.now() / 1000);
  const current = await getRun({ config, sub, jobName });
  ensureMatchingRun(current, quoteHash);
  const currentState = String(current?.state || '');
  if (TERMINAL_STATES.has(currentState)) return current;
  if (
    currentState === RUN_STATE.RUNNING &&
    Number(current?.lastJobSeenAt) > now - Math.max(1, Number(config.ledgerRenewalSeconds) || 60)
  ) return current;
  const leaseExpiresAt = now + config.ledgerLeaseSeconds;
  let updated;
  try {
    const out = await client.send(new UpdateCommand({
      TableName: config.ledgerTable,
      Key: runKey(sub, jobName),
      ConditionExpression: 'quoteHash = :quoteHash AND (#state = :reserved OR #state = :running)',
      UpdateExpression: 'SET #state = :running, jobCreated = :true, jobCreatedAt = if_not_exists(jobCreatedAt, :now), lastJobSeenAt = :now, updatedAt = :now, leaseExpiresAt = :lease REMOVE startAttemptId, startAttemptExpiresAt',
      ExpressionAttributeNames: { '#state': 'state' },
      ExpressionAttributeValues: {
        ':quoteHash': quoteHash,
        ':reserved': RUN_STATE.RESERVED,
        ':running': RUN_STATE.RUNNING,
        ':true': true,
        ':now': now,
        ':lease': leaseExpiresAt
      },
      ReturnValues: 'ALL_NEW'
    }));
    updated = out?.Attributes || null;
  } catch (err) {
    if (!isConditionalFailure(err)) throw err;
    const existing = await getRun({ config, sub, jobName });
    ensureMatchingRun(existing, quoteHash);
    const existingState = String(existing?.state || '');
    const recentlySeen = existingState === RUN_STATE.RUNNING &&
      Number(existing?.lastJobSeenAt) > now - Math.max(1, Number(config.ledgerRenewalSeconds) || 60);
    if (!existing || (!TERMINAL_STATES.has(existingState) && !recentlySeen)) throw err;
    return existing;
  }

  const slotNumber = Number(updated?.slotNumber);
  if (Number.isInteger(slotNumber) && slotNumber > 0) {
    try {
      await client.send(new UpdateCommand({
        TableName: config.ledgerTable,
        Key: slotKey(sub, slotNumber),
        ConditionExpression: 'ownerJobName = :jobName',
        UpdateExpression: 'SET leaseExpiresAt = :lease, #ttl = :expiresAt',
        ExpressionAttributeNames: { '#ttl': config.ledgerTtlAttribute || 'ttl' },
        ExpressionAttributeValues: {
          ':jobName': jobName,
          ':lease': leaseExpiresAt,
          ':expiresAt': leaseExpiresAt + DAY_SECONDS
        }
      }));
    } catch (err) {
      if (!isConditionalFailure(err)) throw err;
    }
  }
  return updated;
}

async function renewRunLease({ config, sub, jobName, quoteHash }){
  if (!isEnabled(config)) return;
  const record = await getRun({ config, sub, jobName });
  ensureMatchingRun(record, quoteHash);
  if (!record || ![RUN_STATE.RESERVED, RUN_STATE.RUNNING].includes(String(record.state || ''))) return;
  const now = Math.floor(Date.now() / 1000);
  const leaseExpiresAt = now + config.ledgerLeaseSeconds;
  const slotNumber = Number(record.slotNumber);
  const operations = [
    getClient(config).send(new UpdateCommand({
      TableName: config.ledgerTable,
      Key: runKey(sub, jobName),
      ConditionExpression: 'quoteHash = :quoteHash AND (#state = :reserved OR #state = :running)',
      UpdateExpression: 'SET leaseExpiresAt = :lease, lastJobSeenAt = :now, updatedAt = :now',
      ExpressionAttributeNames: { '#state': 'state' },
      ExpressionAttributeValues: {
        ':quoteHash': quoteHash,
        ':reserved': RUN_STATE.RESERVED,
        ':running': RUN_STATE.RUNNING,
        ':lease': leaseExpiresAt,
        ':now': now
      }
    }))
  ];
  if (Number.isInteger(slotNumber) && slotNumber > 0) {
    operations.push(getClient(config).send(new UpdateCommand({
      TableName: config.ledgerTable,
      Key: slotKey(sub, slotNumber),
      ConditionExpression: 'ownerJobName = :jobName',
      UpdateExpression: 'SET leaseExpiresAt = :lease, #ttl = :expiresAt',
      ExpressionAttributeNames: { '#ttl': config.ledgerTtlAttribute || 'ttl' },
      ExpressionAttributeValues: {
        ':jobName': jobName,
        ':lease': leaseExpiresAt,
        ':expiresAt': leaseExpiresAt + DAY_SECONDS
      }
    })));
  }
  const results = await Promise.allSettled(operations);
  if (results[0]?.status === 'rejected') {
    const reason = results[0].reason;
    if (!isConditionalFailure(reason)) throw reason;
    const latest = await getRun({ config, sub, jobName });
    ensureMatchingRun(latest, quoteHash);
    if (!latest || !TERMINAL_STATES.has(String(latest.state || ''))) throw reason;
  }
  if (results[1]?.status === 'rejected' && !isConditionalFailure(results[1].reason)) {
    throw results[1].reason;
  }
}

async function deleteOwnedSlot(config, sub, jobName, slotNumber){
  if (!Number.isInteger(slotNumber) || slotNumber <= 0) return;
  try {
    await getClient(config).send(new DeleteCommand({
      TableName: config.ledgerTable,
      Key: slotKey(sub, slotNumber),
      ConditionExpression: 'ownerJobName = :jobName',
      ExpressionAttributeValues: { ':jobName': jobName }
    }));
  } catch (err) {
    if (!isConditionalFailure(err)) throw err;
  }
}

async function markRunTerminal({ config, sub, jobName, quoteHash, status }){
  if (!isEnabled(config)) return null;
  const terminalStatus = String(status || '').toUpperCase();
  if (![RUN_STATE.COMPLETED, RUN_STATE.FAILED, RUN_STATE.MISSING].includes(terminalStatus)) {
    throw ledgerError('LEDGER_INVALID_STATUS', 'The transcription terminal status is invalid.');
  }
  const record = await getRun({ config, sub, jobName });
  ensureMatchingRun(record, quoteHash);
  if (!record) return null;
  if (String(record.state || '') === RUN_STATE.REFUNDED) return record;
  if (TERMINAL_STATES.has(String(record.state || ''))) {
    await deleteOwnedSlot(config, sub, jobName, Number(record.slotNumber));
    return record;
  }

  const now = Math.floor(Date.now() / 1000);
  let updated = record;
  try {
    const out = await getClient(config).send(new UpdateCommand({
      TableName: config.ledgerTable,
      Key: runKey(sub, jobName),
      ConditionExpression: 'quoteHash = :quoteHash AND #state <> :refunded',
      UpdateExpression: 'SET #state = :status, terminalStatus = :status, terminalAt = :now, updatedAt = :now REMOVE leaseExpiresAt',
      ExpressionAttributeNames: { '#state': 'state' },
      ExpressionAttributeValues: {
        ':quoteHash': quoteHash,
        ':refunded': RUN_STATE.REFUNDED,
        ':status': terminalStatus,
        ':now': now
      },
      ReturnValues: 'ALL_NEW'
    }));
    updated = out?.Attributes || record;
  } catch (err) {
    if (!isConditionalFailure(err)) throw err;
    updated = await getRun({ config, sub, jobName });
  }
  await deleteOwnedSlot(config, sub, jobName, Number(updated?.slotNumber || record.slotNumber));
  return updated;
}

async function refundRun({ config, sub, jobName, quoteHash, attemptId, reason }){
  if (!isEnabled(config)) return null;
  const record = await getRun({ config, sub, jobName });
  ensureMatchingRun(record, quoteHash);
  if (!record || String(record.state || '') === RUN_STATE.REFUNDED) return record;
  if (record.jobCreated || String(record.state || '') !== RUN_STATE.RESERVED) {
    throw ledgerError('LEDGER_REFUND_UNSAFE', 'The transcription reservation cannot be refunded because an AWS job may exist.');
  }

  const now = Math.floor(Date.now() / 1000);
  const costMicros = Number(record.costMicros) || 0;
  try {
    await getClient(config).send(new TransactWriteCommand({
      TransactItems: [
        {
          Update: {
            TableName: config.ledgerTable,
            Key: runKey(sub, jobName),
            ConditionExpression: 'quoteHash = :quoteHash AND #state = :reserved AND startAttemptId = :attemptId AND (attribute_not_exists(jobCreated) OR jobCreated = :false)',
            UpdateExpression: 'SET #state = :refunded, terminalStatus = :refunded, terminalAt = :now, refundedAt = :now, refundReason = :reason, updatedAt = :now REMOVE leaseExpiresAt, startAttemptId, startAttemptExpiresAt',
            ExpressionAttributeNames: { '#state': 'state' },
            ExpressionAttributeValues: {
              ':quoteHash': quoteHash,
              ':attemptId': attemptId,
              ':reserved': RUN_STATE.RESERVED,
              ':refunded': RUN_STATE.REFUNDED,
              ':false': false,
              ':now': now,
              ':reason': String(reason || 'AWS_JOB_NOT_CREATED').slice(0, 120)
            }
          }
        },
        {
          Update: {
            TableName: config.ledgerTable,
            Key: dayKey(sub, record.day),
            ConditionExpression: 'reservedCostMicros >= :cost AND fileCount >= :one',
            UpdateExpression: 'SET updatedAt = :now ADD reservedCostMicros :negativeCost, fileCount :negativeOne',
            ExpressionAttributeValues: {
              ':cost': costMicros,
              ':one': 1,
              ':now': now,
              ':negativeCost': -costMicros,
              ':negativeOne': -1
            }
          }
        },
        {
          Update: {
            TableName: config.ledgerTable,
            Key: globalDayKey(record.day),
            ConditionExpression: 'reservedCostMicros >= :cost AND fileCount >= :one',
            UpdateExpression: 'SET updatedAt = :now ADD reservedCostMicros :negativeCost, fileCount :negativeOne',
            ExpressionAttributeValues: {
              ':cost': costMicros,
              ':one': 1,
              ':now': now,
              ':negativeCost': -costMicros,
              ':negativeOne': -1
            }
          }
        }
      ]
    }));
  } catch (err) {
    if (!isConditionalFailure(err)) throw err;
    const latest = await getRun({ config, sub, jobName });
    ensureMatchingRun(latest, quoteHash);
    if (String(latest?.state || '') !== RUN_STATE.REFUNDED) throw err;
  }
  await deleteOwnedSlot(config, sub, jobName, Number(record.slotNumber));
  return getRun({ config, sub, jobName });
}

module.exports = {
  RUN_STATE,
  getRun,
  markJobCreated,
  markRunTerminal,
  refundRun,
  renewRunLease,
  reserveRun,
  _internal: {
    dayKey,
    globalDayKey,
    reservationTransaction,
    runKey,
    slotKey,
    toCostMicros,
    utcDay
  }
};
