'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { normalizeCmsDocumentId } = require('./cms-content-model');

const root = path.resolve(__dirname, '..', '..');
const LIBRARY_TYPES = {
  template: 'templates',
  section: 'sections',
  draft: 'drafts'
};
const LIBRARY_TYPE_NAMES = Object.keys(LIBRARY_TYPES);
const MAX_LIBRARY_ITEM_BYTES = 420_000;

function normalizeLibraryType(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (LIBRARY_TYPES[raw]) return raw;
  const singular = raw.replace(/s$/, '');
  return LIBRARY_TYPES[singular] ? singular : '';
}

function getLibraryDir(type) {
  const normalizedType = normalizeLibraryType(type);
  if (!normalizedType) return null;
  return path.join(root, 'content', 'cms-library', LIBRARY_TYPES[normalizedType]);
}

function getLibraryPath(type, id) {
  const normalizedType = normalizeLibraryType(type);
  const normalizedId = normalizeCmsDocumentId(id);
  const dir = getLibraryDir(normalizedType);
  if (!normalizedType || !normalizedId || !dir) return null;
  const absPath = path.join(dir, `${normalizedId}.json`);
  const relPath = path.relative(root, absPath).replace(/\\/g, '/');
  if (!relPath.startsWith(`content/cms-library/${LIBRARY_TYPES[normalizedType]}/`)) return null;
  return { absPath, relPath, type: normalizedType, id: normalizedId };
}

function hashContent(value) {
  return crypto.createHash('sha256').update(String(value || ''), 'utf8').digest('hex');
}

function readJson(absPath) {
  return JSON.parse(fs.readFileSync(absPath, 'utf8'));
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

function createLibraryError(code, message) {
  const err = new Error(message);
  err.code = code;
  return err;
}

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw createLibraryError('CMS_INVALID_LIBRARY_ITEM', `${label} must be a JSON object`);
  }
}

function normalizeStringArray(value) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item || '').trim()).filter(Boolean).slice(0, 24);
}

function normalizeLibraryItem(type, item) {
  const normalizedType = normalizeLibraryType(type);
  assertPlainObject(item, 'CMS library item');
  const id = normalizeCmsDocumentId(item.id);
  if (!normalizedType) throw createLibraryError('CMS_INVALID_LIBRARY_TYPE', 'Invalid CMS library type');
  if (!id) throw createLibraryError('CMS_INVALID_LIBRARY_ID', 'Invalid CMS library item id');

  const base = {
    id,
    name: String(item.name || item.title || id).trim() || id,
    description: String(item.description || '').trim(),
    folder: String(item.folder || 'General').trim() || 'General',
    tags: normalizeStringArray(item.tags),
    updatedAt: item.updatedAt || new Date().toISOString()
  };

  if (normalizedType === 'template') {
    assertPlainObject(item.page, 'CMS template page');
    return { ...base, page: item.page };
  }

  if (normalizedType === 'section') {
    assertPlainObject(item.section, 'CMS saved section');
    return {
      ...base,
      locks: {
        editable: item.locks && item.locks.editable === false ? false : true,
        lockPosition: Boolean(item.locks && item.locks.lockPosition),
        lockRemoval: Boolean(item.locks && item.locks.lockRemoval),
        lockText: Boolean(item.locks && item.locks.lockText),
        lockMedia: Boolean(item.locks && item.locks.lockMedia)
      },
      section: item.section
    };
  }

  assertPlainObject(item.document, 'CMS draft document');
  return {
    ...base,
    source: item.source && typeof item.source === 'object' && !Array.isArray(item.source) ? item.source : {},
    pagePath: String(item.pagePath || '').trim(),
    createdAt: item.createdAt || new Date().toISOString(),
    document: item.document
  };
}

function toLibraryRecord(type, absPath, item) {
  let stat = null;
  try {
    stat = fs.statSync(absPath);
  } catch {}
  return {
    type: normalizeLibraryType(type),
    id: item.id,
    relPath: path.relative(root, absPath).replace(/\\/g, '/'),
    item,
    updatedAt: stat ? Math.round(stat.mtimeMs) : 0,
    updatedAtIso: stat ? new Date(stat.mtimeMs).toISOString() : '',
    revisionId: stat ? hashContent(fs.readFileSync(absPath, 'utf8')) : ''
  };
}

function listLibraryItems(type) {
  const normalizedType = normalizeLibraryType(type);
  if (!normalizedType) throw createLibraryError('CMS_INVALID_LIBRARY_TYPE', 'Invalid CMS library type');
  const dir = getLibraryDir(normalizedType);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
    .sort((a, b) => a.localeCompare(b))
    .map((fileName) => {
      const absPath = path.join(dir, fileName);
      const item = normalizeLibraryItem(normalizedType, readJson(absPath));
      return toLibraryRecord(normalizedType, absPath, item);
    });
}

function listAllLibraryItems() {
  return LIBRARY_TYPE_NAMES.reduce((acc, type) => {
    acc[`${LIBRARY_TYPES[type]}`] = listLibraryItems(type);
    return acc;
  }, {});
}

function saveLibraryItem(type, item) {
  const normalized = normalizeLibraryItem(type, {
    ...item,
    updatedAt: new Date().toISOString()
  });
  const target = getLibraryPath(type, normalized.id);
  if (!target) throw createLibraryError('CMS_INVALID_LIBRARY_ID', 'Invalid CMS library path');
  const raw = `${JSON.stringify(normalized, null, 2)}\n`;
  if (Buffer.byteLength(raw, 'utf8') > MAX_LIBRARY_ITEM_BYTES) {
    throw createLibraryError('CMS_LIBRARY_ITEM_TOO_LARGE', `CMS library item is too large (max ${MAX_LIBRARY_ITEM_BYTES} bytes)`);
  }
  fs.mkdirSync(path.dirname(target.absPath), { recursive: true });
  writeFileAtomic(target.absPath, raw);
  return toLibraryRecord(target.type, target.absPath, normalized);
}

function deleteLibraryItem(type, id) {
  const target = getLibraryPath(type, id);
  if (!target) throw createLibraryError('CMS_INVALID_LIBRARY_ID', 'Invalid CMS library path');
  try {
    fs.unlinkSync(target.absPath);
    return true;
  } catch (err) {
    if (err && err.code === 'ENOENT') return false;
    throw err;
  }
}

module.exports = {
  LIBRARY_TYPE_NAMES,
  deleteLibraryItem,
  listAllLibraryItems,
  listLibraryItems,
  normalizeLibraryType,
  saveLibraryItem
};
