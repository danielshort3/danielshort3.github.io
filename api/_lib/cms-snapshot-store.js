'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const {
  normalizeCmsCollection,
  normalizeCmsDocumentId
} = require('./cms-content-model');

const root = path.resolve(__dirname, '..', '..');
const SNAPSHOT_DIR = path.join(root, 'content', 'cms-library', 'snapshots');
const MAX_SNAPSHOT_BYTES = 620_000;
const MAX_SNAPSHOTS_PER_DOCUMENT = 24;

function hashContent(value) {
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex');
}

function writeFileAtomic(absPath, value) {
  const dir = path.dirname(absPath);
  const base = path.basename(absPath);
  const tempPath = path.join(dir, `.${base}.${process.pid}.${Date.now()}.${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tempPath, value, { encoding: 'utf8', flag: 'wx' });
    fs.renameSync(tempPath, absPath);
  } catch (err) {
    try {
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    } catch {}
    throw err;
  }
}

function createSnapshotError(code, message) {
  const err = new Error(message);
  err.code = code;
  return err;
}

function getSnapshotPath(id) {
  const normalizedId = normalizeCmsDocumentId(id);
  if (!normalizedId) return null;
  const absPath = path.join(SNAPSHOT_DIR, `${normalizedId}.json`);
  const relPath = path.relative(root, absPath).replace(/\\/g, '/');
  if (!relPath.startsWith('content/cms-library/snapshots/')) return null;
  return { absPath, relPath, id: normalizedId };
}

function normalizeSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)) {
    throw createSnapshotError('CMS_INVALID_SNAPSHOT', 'CMS snapshot must be a JSON object');
  }
  const collection = normalizeCmsCollection(snapshot.collection);
  const documentId = normalizeCmsDocumentId(snapshot.documentId || snapshot.id);
  const snapshotId = normalizeCmsDocumentId(snapshot.snapshotId || snapshot.id);
  if (!collection || !documentId || !snapshotId) {
    throw createSnapshotError('CMS_INVALID_SNAPSHOT', 'CMS snapshot has an invalid collection, document id, or snapshot id');
  }
  return {
    snapshotId,
    collection,
    documentId,
    relPath: String(snapshot.relPath || '').trim(),
    reason: String(snapshot.reason || 'before-save').trim() || 'before-save',
    createdAt: String(snapshot.createdAt || new Date().toISOString()),
    revisionId: String(snapshot.revisionId || ''),
    document: snapshot.document && typeof snapshot.document === 'object' && !Array.isArray(snapshot.document)
      ? snapshot.document
      : {}
  };
}

function readSnapshot(absPath) {
  const snapshot = normalizeSnapshot(JSON.parse(fs.readFileSync(absPath, 'utf8')));
  const stat = fs.statSync(absPath);
  return {
    ...snapshot,
    relPath: snapshot.relPath || path.relative(root, absPath).replace(/\\/g, '/'),
    filePath: path.relative(root, absPath).replace(/\\/g, '/'),
    updatedAt: Math.round(stat.mtimeMs),
    updatedAtIso: new Date(stat.mtimeMs).toISOString()
  };
}

function listSnapshots(filters = {}) {
  const collection = normalizeCmsCollection(filters.collection);
  const documentId = normalizeCmsDocumentId(filters.id || filters.documentId);
  if (!fs.existsSync(SNAPSHOT_DIR)) return [];
  return fs.readdirSync(SNAPSHOT_DIR)
    .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
    .map((name) => {
      try {
        return readSnapshot(path.join(SNAPSHOT_DIR, name));
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .filter((snapshot) => !collection || snapshot.collection === collection)
    .filter((snapshot) => !documentId || snapshot.documentId === documentId)
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
}

function pruneSnapshotsForDocument(collection, documentId) {
  const snapshots = listSnapshots({ collection, id: documentId });
  snapshots.slice(MAX_SNAPSHOTS_PER_DOCUMENT).forEach((snapshot) => {
    const target = getSnapshotPath(snapshot.snapshotId);
    if (!target) return;
    try {
      fs.unlinkSync(target.absPath);
    } catch {}
  });
}

function createDocumentSnapshot({ collection, id, relPath, document, revisionId = '', reason = 'before-save' }) {
  const normalizedCollection = normalizeCmsCollection(collection);
  const normalizedId = normalizeCmsDocumentId(id);
  if (!normalizedCollection || !normalizedId || !document || typeof document !== 'object' || Array.isArray(document)) return null;

  const createdAt = new Date().toISOString();
  const stamp = createdAt.replace(/[-:.TZ]/g, '').slice(0, 14);
  const digest = hashContent(JSON.stringify(document)).slice(0, 10);
  const snapshotId = normalizeCmsDocumentId(`${normalizedCollection}-${normalizedId}-${stamp}-${digest}`);
  const snapshot = normalizeSnapshot({
    snapshotId,
    collection: normalizedCollection,
    documentId: normalizedId,
    relPath,
    reason,
    createdAt,
    revisionId,
    document
  });
  const target = getSnapshotPath(snapshot.snapshotId);
  if (!target) throw createSnapshotError('CMS_INVALID_SNAPSHOT', 'Invalid CMS snapshot path');
  const raw = `${JSON.stringify(snapshot, null, 2)}\n`;
  if (Buffer.byteLength(raw, 'utf8') > MAX_SNAPSHOT_BYTES) {
    throw createSnapshotError('CMS_SNAPSHOT_TOO_LARGE', `CMS snapshot is too large (max ${MAX_SNAPSHOT_BYTES} bytes)`);
  }
  fs.mkdirSync(path.dirname(target.absPath), { recursive: true });
  writeFileAtomic(target.absPath, raw);
  pruneSnapshotsForDocument(snapshot.collection, snapshot.documentId);
  return readSnapshot(target.absPath);
}

function getSnapshot(snapshotId) {
  const target = getSnapshotPath(snapshotId);
  if (!target || !fs.existsSync(target.absPath)) return null;
  return readSnapshot(target.absPath);
}

function createJsonDiffSummary(before, after) {
  const beforeText = JSON.stringify(before || {}, null, 2).split('\n');
  const afterText = JSON.stringify(after || {}, null, 2).split('\n');
  const max = Math.max(beforeText.length, afterText.length);
  let changedLines = 0;
  for (let index = 0; index < max; index += 1) {
    if (beforeText[index] !== afterText[index]) changedLines += 1;
  }
  return {
    changedLines,
    beforeLines: beforeText.length,
    afterLines: afterText.length
  };
}

module.exports = {
  createDocumentSnapshot,
  createJsonDiffSummary,
  getSnapshot,
  listSnapshots
};
