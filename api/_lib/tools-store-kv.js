/*
  KV-backed storage for tools accounts (sessions + activity).
*/
'use strict';

const { kvCall, kvGet, kvSet, kvDel, kvSadd, kvSmembers } = require('./kv');

const PREFIX = 'tools';
const MAX_SNAPSHOT_BYTES = 300_000;
const MAX_SUMMARY_CHARS = 2_000;
const MAX_ACTIVITY_EVENTS = 1_000;

function userToolsKey(sub){
  return `${PREFIX}:user:${sub}:tools`;
}

function toolMetaKey(sub, toolId){
  return `${PREFIX}:user:${sub}:tool:${toolId}:meta`;
}

function toolSessionsKey(sub, toolId){
  return `${PREFIX}:user:${sub}:tool:${toolId}:sessions`;
}

function toolActivityKey(sub, toolId){
  return `${PREFIX}:user:${sub}:tool:${toolId}:activity`;
}

function userSessionsKey(sub){
  return `${PREFIX}:user:${sub}:sessions`;
}

function userActivityKey(sub){
  return `${PREFIX}:user:${sub}:activity`;
}

function sessionKey(sub, toolId, sessionId){
  return `${PREFIX}:user:${sub}:tool:${toolId}:session:${sessionId}`;
}

function randomBase64Url(size = 16){
  const nodeCrypto = require('crypto');
  return nodeCrypto.randomBytes(size).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function byteLength(value){
  try {
    return Buffer.byteLength(String(value), 'utf8');
  } catch {
    return String(value || '').length;
  }
}

function normalizeSummary(value){
  const summary = String(value || '').trim();
  if (!summary) return '';
  return summary.length > MAX_SUMMARY_CHARS ? summary.slice(0, MAX_SUMMARY_CHARS) : summary;
}

function ensureSnapshotOk(snapshot){
  const raw = JSON.stringify(snapshot || {});
  if (byteLength(raw) > MAX_SNAPSHOT_BYTES) {
    const err = new Error(`Snapshot too large (max ${MAX_SNAPSHOT_BYTES} bytes).`);
    err.code = 'SNAPSHOT_TOO_LARGE';
    throw err;
  }
  return raw;
}

async function upsertToolMeta(sub, toolId, updates){
  const key = toolMetaKey(sub, toolId);
  let current = null;
  try {
    const raw = await kvGet(key);
    current = raw ? JSON.parse(raw) : null;
  } catch {}

  const now = Date.now();
  const next = {
    toolId,
    firstUsedAt: current?.firstUsedAt || now,
    lastUsedAt: now,
    sessionCount: Number.isFinite(current?.sessionCount) ? current.sessionCount : 0,
    activityCount: Number.isFinite(current?.activityCount) ? current.activityCount : 0,
    ...(updates || {})
  };
  await kvSet(key, JSON.stringify(next));
  return next;
}

async function saveSession({ sub, toolId, sessionId, snapshot, outputSummary }){
  const snapshotJson = ensureSnapshotOk(snapshot);
  const now = Date.now();
  const existingKey = sessionId ? sessionKey(sub, toolId, sessionId) : '';
  let createdAt = now;
  if (existingKey) {
    try {
      const raw = await kvGet(existingKey);
      const existing = raw ? JSON.parse(raw) : null;
      if (existing?.createdAt) createdAt = Number(existing.createdAt) || createdAt;
    } catch {}
  }

  const nextSessionId = sessionId || randomBase64Url(18);
  const record = {
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: normalizeSummary(outputSummary),
    snapshot: snapshot || {}
  };

  await kvSet(sessionKey(sub, toolId, nextSessionId), JSON.stringify(record));
  await kvCall('zadd', toolSessionsKey(sub, toolId), now, nextSessionId);
  await kvCall('zadd', userSessionsKey(sub), now, `${toolId}:${nextSessionId}`);
  await kvSadd(userToolsKey(sub), toolId);

  let sessionCount = 0;
  try {
    sessionCount = Number(await kvCall('zcard', toolSessionsKey(sub, toolId))) || 0;
  } catch {}
  await upsertToolMeta(sub, toolId, { sessionCount });

  return record;
}

async function listSessions({ sub, toolId, limit }){
  const safeLimit = Math.min(Math.max(Number(limit) || 20, 1), 50);
  const ids = await kvCall('zrevrange', toolSessionsKey(sub, toolId), 0, safeLimit - 1);
  const sessionIds = Array.isArray(ids) ? ids : [];
  const records = [];
  for (const sessionId of sessionIds) {
    try {
      const raw = await kvGet(sessionKey(sub, toolId, sessionId));
      if (!raw) continue;
      const record = JSON.parse(raw);
      if (record && record.sessionId) records.push(record);
    } catch {}
  }
  return records;
}

async function getSession({ sub, toolId, sessionId }){
  const raw = await kvGet(sessionKey(sub, toolId, sessionId));
  if (!raw) return null;
  return JSON.parse(raw);
}

async function deleteSession({ sub, toolId, sessionId }){
  await kvDel(sessionKey(sub, toolId, sessionId));
  try {
    await kvCall('zrem', toolSessionsKey(sub, toolId), sessionId);
  } catch {}
  try {
    await kvCall('zrem', userSessionsKey(sub), `${toolId}:${sessionId}`);
  } catch {}
  let sessionCount = 0;
  try {
    sessionCount = Number(await kvCall('zcard', toolSessionsKey(sub, toolId))) || 0;
  } catch {}
  await upsertToolMeta(sub, toolId, { sessionCount });
  return true;
}

async function logActivity({ sub, toolId, type, summary, data }){
  const now = Date.now();
  const event = {
    id: randomBase64Url(18),
    toolId,
    type: String(type || '').trim(),
    ts: now,
    summary: normalizeSummary(summary),
    data: data && typeof data === 'object' ? data : undefined
  };

  await kvCall('zadd', toolActivityKey(sub, toolId), now, JSON.stringify(event));
  await kvCall('zadd', userActivityKey(sub), now, JSON.stringify(event));
  await kvSadd(userToolsKey(sub), toolId);

  try {
    const toolCount = Number(await kvCall('zcard', toolActivityKey(sub, toolId))) || 0;
    if (toolCount > MAX_ACTIVITY_EVENTS) {
      await kvCall('zremrangebyrank', toolActivityKey(sub, toolId), 0, toolCount - MAX_ACTIVITY_EVENTS - 1);
    }
  } catch {}
  try {
    const userCount = Number(await kvCall('zcard', userActivityKey(sub))) || 0;
    if (userCount > MAX_ACTIVITY_EVENTS) {
      await kvCall('zremrangebyrank', userActivityKey(sub), 0, userCount - MAX_ACTIVITY_EVENTS - 1);
    }
  } catch {}

  let activityCount = 0;
  try {
    activityCount = Number(await kvCall('zcard', toolActivityKey(sub, toolId))) || 0;
  } catch {}
  await upsertToolMeta(sub, toolId, { activityCount });

  return event;
}

async function listActivity({ sub, toolId, limit }){
  const safeLimit = Math.min(Math.max(Number(limit) || 50, 1), 200);
  const key = toolId ? toolActivityKey(sub, toolId) : userActivityKey(sub);
  const items = await kvCall('zrevrange', key, 0, safeLimit - 1);
  const rawEvents = Array.isArray(items) ? items : [];
  const events = [];
  rawEvents.forEach((raw) => {
    try {
      const event = JSON.parse(raw);
      if (event && event.ts) events.push(event);
    } catch {}
  });
  return events;
}

async function listUserTools(sub){
  const tools = await kvSmembers(userToolsKey(sub));
  return tools.filter(Boolean).sort();
}

async function getToolMeta(sub, toolId){
  try {
    const raw = await kvGet(toolMetaKey(sub, toolId));
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

async function listRecentSessions(sub, limit){
  const safeLimit = Math.min(Math.max(Number(limit) || 25, 1), 100);
  const items = await kvCall('zrevrange', userSessionsKey(sub), 0, safeLimit - 1);
  const members = Array.isArray(items) ? items : [];
  const records = [];
  for (const member of members) {
    const parts = String(member || '').split(':');
    const toolId = parts[0] || '';
    const sessionId = parts.slice(1).join(':') || '';
    if (!toolId || !sessionId) continue;
    try {
      const raw = await kvGet(sessionKey(sub, toolId, sessionId));
      if (!raw) continue;
      const record = JSON.parse(raw);
      if (record && record.sessionId) records.push(record);
    } catch {}
  }
  return records;
}

module.exports = {
  MAX_SNAPSHOT_BYTES,
  userToolsKey,
  toolMetaKey,
  toolSessionsKey,
  sessionKey,
  saveSession,
  listSessions,
  getSession,
  deleteSession,
  logActivity,
  listActivity,
  listUserTools,
  getToolMeta,
  listRecentSessions
};
