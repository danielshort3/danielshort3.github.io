/*
  KV-backed storage for tools accounts (sessions + activity).
*/
'use strict';

const { kvCall, kvGet, kvSet, kvDel, kvSadd, kvSmembers } = require('./kv');

const PREFIX = 'tools';
const MAX_SNAPSHOT_BYTES = 300_000;
const MAX_SUMMARY_CHARS = 2_000;
const MAX_ACTIVITY_EVENTS = 1_000;
const MAX_TITLE_CHARS = 120;
const MAX_NOTE_CHARS = 800;
const MAX_TAGS = 12;
const MAX_TAG_CHARS = 24;

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

function normalizeTitle(value){
  const title = String(value || '').trim();
  if (!title) return '';
  return title.length > MAX_TITLE_CHARS ? title.slice(0, MAX_TITLE_CHARS).trimEnd() : title;
}

function normalizeNote(value){
  const note = String(value || '').trim();
  if (!note) return '';
  return note.length > MAX_NOTE_CHARS ? note.slice(0, MAX_NOTE_CHARS).trimEnd() : note;
}

function normalizeTags(value){
  const raw = Array.isArray(value)
    ? value.map(v => String(v || '').trim())
    : String(value || '')
      .split(/[,\n]/g)
      .map(v => String(v || '').trim());

  const seen = new Set();
  const tags = [];
  raw.forEach((tag) => {
    const cleaned = tag.replace(/\s+/g, ' ').trim();
    if (!cleaned) return;
    const clipped = cleaned.length > MAX_TAG_CHARS ? cleaned.slice(0, MAX_TAG_CHARS).trimEnd() : cleaned;
    const key = clipped.toLowerCase();
    if (!key) return;
    if (seen.has(key)) return;
    seen.add(key);
    tags.push(clipped);
  });

  return tags.slice(0, MAX_TAGS);
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
  let title = '';
  let note = '';
  let tags = [];
  let pinned = false;
  if (existingKey) {
    try {
      const raw = await kvGet(existingKey);
      const existing = raw ? JSON.parse(raw) : null;
      if (existing?.createdAt) createdAt = Number(existing.createdAt) || createdAt;
      title = normalizeTitle(existing?.title);
      note = normalizeNote(existing?.note);
      tags = normalizeTags(existing?.tags);
      pinned = Boolean(existing?.pinned);
    } catch {}
  }

  const nextSessionId = sessionId || randomBase64Url(18);
  const record = {
    toolId,
    sessionId: nextSessionId,
    createdAt,
    updatedAt: now,
    outputSummary: normalizeSummary(outputSummary),
    title: title || undefined,
    note: note || undefined,
    tags: tags.length ? tags : undefined,
    pinned,
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

async function updateSessionMeta({ sub, toolId, sessionId, title, note, tags, pinned }){
  const key = sessionKey(sub, toolId, sessionId);
  let record;
  try {
    const raw = await kvGet(key);
    record = raw ? JSON.parse(raw) : null;
  } catch {
    record = null;
  }
  if (!record) return null;

  const next = {
    ...record,
    title: (typeof title !== 'undefined') ? normalizeTitle(title) : normalizeTitle(record.title),
    note: (typeof note !== 'undefined') ? normalizeNote(note) : normalizeNote(record.note),
    tags: (typeof tags !== 'undefined') ? normalizeTags(tags) : normalizeTags(record.tags),
    pinned: (typeof pinned !== 'undefined') ? Boolean(pinned) : Boolean(record.pinned)
  };

  if (!next.title) delete next.title;
  if (!next.note) delete next.note;
  if (!Array.isArray(next.tags) || !next.tags.length) delete next.tags;

  await kvSet(key, JSON.stringify(next));
  return next;
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
  updateSessionMeta,
  listSessions,
  getSession,
  deleteSession,
  logActivity,
  listActivity,
  listUserTools,
  getToolMeta,
  listRecentSessions
};
