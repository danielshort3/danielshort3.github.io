'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const {
  CMS_COLLECTION_NAMES,
  getCollectionConfig,
  listFileContentRecords,
  normalizeCmsCollection,
  normalizeCmsDocumentId,
  validateCmsDocumentInput
} = require('./cms-content-model');
const {
  createDocumentSnapshot
} = require('./cms-snapshot-store');

const root = path.resolve(__dirname, '..', '..');

function hashContent(value) {
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex');
}

function getDocumentPath({ collection, id }) {
  const normalizedCollection = normalizeCmsCollection(collection);
  const normalizedId = normalizeCmsDocumentId(id);
  if (!normalizedCollection || !normalizedId) return null;

  const config = getCollectionConfig(normalizedCollection);
  if (!config) return null;

  const absDir = path.join(root, config.relDir);
  const absPath = path.join(absDir, `${normalizedId}.json`);
  const relPath = path.relative(root, absPath).replace(/\\/g, '/');
  const relContentDir = path.relative(root, path.join(root, 'content')).replace(/\\/g, '/');

  if (!relPath || relPath.startsWith('..') || path.isAbsolute(relPath)) return null;
  if (!relPath.startsWith(`${relContentDir}/`)) return null;

  return { absPath, relPath };
}

function getFileRevision(absPath) {
  try {
    return hashContent(fs.readFileSync(absPath, 'utf8'));
  } catch (err) {
    if (err && err.code === 'ENOENT') return '';
    throw err;
  }
}

function toContentRecord(record) {
  const filePath = record && record.relPath ? path.join(root, record.relPath) : '';
  let stat = null;
  let revisionId = '';
  try {
    stat = filePath ? fs.statSync(filePath) : null;
    revisionId = stat && stat.isFile() ? getFileRevision(filePath) : '';
  } catch {}

  const updatedAt = stat ? Math.round(stat.mtimeMs) : 0;
  return {
    collection: record.collection,
    id: record.id,
    relPath: record.relPath || '',
    document: record.document,
    updatedAt,
    updatedAtIso: updatedAt ? new Date(updatedAt).toISOString() : '',
    updatedBy: 'local file',
    revisionId
  };
}

function createRevisionConflictError() {
  const err = new Error('CMS document changed on disk. Refresh before saving again.');
  err.code = 'CMS_REVISION_CONFLICT';
  throw err;
}

function assertExpectedRevision(absPath, expectedRevisionId) {
  if (typeof expectedRevisionId !== 'string') return;

  const exists = fs.existsSync(absPath);
  if (!expectedRevisionId && exists) createRevisionConflictError();
  if (!exists) {
    if (expectedRevisionId) createRevisionConflictError();
    return;
  }

  const currentRevisionId = getFileRevision(absPath);
  if (currentRevisionId !== expectedRevisionId) createRevisionConflictError();
}

function writeFileAtomic(absPath, value) {
  const dir = path.dirname(absPath);
  const base = path.basename(absPath);
  const nonce = crypto.randomBytes(6).toString('hex');
  const tempPath = path.join(dir, `.${base}.${process.pid}.${Date.now()}.${nonce}.tmp`);

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

function readJson(absPath) {
  return JSON.parse(fs.readFileSync(absPath, 'utf8'));
}

function listAllRecords() {
  return listFileContentRecords(root).map(toContentRecord);
}

async function listAllCurrentDocuments() {
  const grouped = {};
  CMS_COLLECTION_NAMES.forEach((collection) => {
    grouped[collection] = [];
  });

  listAllRecords().forEach((record) => {
    grouped[record.collection].push(record);
  });

  return grouped;
}

async function listCurrentDocuments(collection) {
  const normalizedCollection = normalizeCmsCollection(collection);
  if (!normalizedCollection) return [];
  return listAllRecords()
    .filter((record) => record.collection === normalizedCollection)
    .sort((a, b) => a.id.localeCompare(b.id));
}

async function getCurrentDocument({ collection, id }) {
  const normalizedCollection = normalizeCmsCollection(collection);
  const normalizedId = normalizeCmsDocumentId(id);
  if (!normalizedCollection || !normalizedId) return null;
  const records = await listCurrentDocuments(normalizedCollection);
  return records.find((record) => record.id === normalizedId) || null;
}

async function saveCurrentDocument({ collection, id, document, expectedRevisionId }) {
  const input = validateCmsDocumentInput({ collection, id, document });
  const target = getDocumentPath(input);
  if (!target) {
    const err = new Error('Invalid CMS document path');
    err.code = 'CMS_INVALID_DOCUMENT_ID';
    throw err;
  }

  fs.mkdirSync(path.dirname(target.absPath), { recursive: true });
  assertExpectedRevision(target.absPath, expectedRevisionId);
  if (fs.existsSync(target.absPath)) {
    createDocumentSnapshot({
      collection: input.collection,
      id: input.id,
      relPath: target.relPath,
      document: readJson(target.absPath),
      revisionId: getFileRevision(target.absPath),
      reason: 'before-save'
    });
  }
  writeFileAtomic(target.absPath, `${JSON.stringify(input.document, null, 2)}\n`);

  return toContentRecord({
    collection: input.collection,
    id: input.id,
    relPath: target.relPath,
    document: input.document
  });
}

module.exports = {
  getCurrentDocument,
  getDocumentPath,
  listAllCurrentDocuments,
  listCurrentDocuments,
  saveCurrentDocument
};
