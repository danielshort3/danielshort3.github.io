'use strict';

const fs = require('fs');
const path = require('path');

const CMS_COLLECTIONS = [
  { name: 'site', label: 'Site', relDir: path.join('content', 'site'), sortKey: 'id' },
  { name: 'pages', label: 'Pages', relDir: path.join('content', 'pages'), sortKey: 'id' },
  { name: 'audiences', label: 'Audiences', relDir: path.join('content', 'audiences'), sortKey: 'key' },
  { name: 'resumes', label: 'Resumes', relDir: path.join('content', 'resumes'), sortKey: 'key' },
  { name: 'projects', label: 'Projects', relDir: path.join('content', 'projects'), sortKey: 'id' },
  { name: 'tools', label: 'Tools', relDir: path.join('content', 'tools'), sortKey: 'slug' }
];

const CMS_COLLECTION_NAMES = CMS_COLLECTIONS.map((collection) => collection.name);
const CMS_COLLECTION_SET = new Set(CMS_COLLECTION_NAMES);
const SITE_DOCUMENT_IDS = ['settings', 'navigation', 'footer'];
const MAX_DOCUMENT_BYTES = 360_000;

function getCollectionConfig(value) {
  const name = normalizeCmsCollection(value);
  return CMS_COLLECTIONS.find((collection) => collection.name === name) || null;
}

function normalizeCmsCollection(value) {
  const name = String(value || '').trim();
  return CMS_COLLECTION_SET.has(name) ? name : '';
}

function normalizeCmsDocumentId(value) {
  const id = String(value || '').trim();
  if (!id || id.length > 128) return '';
  return /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id) ? id : '';
}

function byteLength(value) {
  try {
    return Buffer.byteLength(String(value), 'utf8');
  } catch {
    return String(value || '').length;
  }
}

function assertPlainObject(document, label) {
  if (!document || typeof document !== 'object' || Array.isArray(document)) {
    throw new Error(`${label} must be a JSON object`);
  }
}

function createShapeError(message) {
  const err = new Error(message);
  err.code = 'CMS_INVALID_DOCUMENT_SHAPE';
  return err;
}

function assertStringField(document, field, label) {
  if (typeof document[field] !== 'string' || !document[field].trim()) {
    throw createShapeError(`${label} must include a non-empty "${field}" string`);
  }
}

function assertArrayField(document, field, label) {
  if (!Array.isArray(document[field])) {
    throw createShapeError(`${label} must include a "${field}" array`);
  }
}

function assertObjectField(document, field, label) {
  if (!document[field] || typeof document[field] !== 'object' || Array.isArray(document[field])) {
    throw createShapeError(`${label} must include a "${field}" object`);
  }
}

function assertMatchingStringField(document, field, expected, label) {
  assertStringField(document, field, label);
  if (document[field] !== expected) {
    throw createShapeError(`${label} "${field}" must match document id "${expected}"`);
  }
}

function validateSiteDocumentShape(id, document) {
  if (id === 'settings') {
    ['siteOrigin', 'siteName', 'ownerName', 'email'].forEach((field) => {
      assertStringField(document, field, 'site/settings');
    });
    assertObjectField(document, 'ogImage', 'site/settings');
    return;
  }

  if (id === 'navigation') {
    ['brand', 'portfolio', 'resume', 'contact', 'search'].forEach((field) => {
      assertObjectField(document, field, 'site/navigation');
    });
    return;
  }

  if (id === 'footer') {
    assertArrayField(document, 'columns', 'site/footer');
  }
}

function validateCmsDocumentShape({ collection, id, document }) {
  const label = `${collection}/${id}`;

  if (collection === 'site') {
    validateSiteDocumentShape(id, document);
    return;
  }

  if (collection === 'pages') {
    assertMatchingStringField(document, 'id', id, label);
    ['template', 'outputPath', 'title', 'canonicalPath', 'description'].forEach((field) => {
      assertStringField(document, field, label);
    });
    if (document.template === 'raw-body') assertStringField(document, 'bodyHtml', label);
    if (document.template === 'visual-page') assertArrayField(document, 'sections', label);
    return;
  }

  if (collection === 'audiences') {
    assertMatchingStringField(document, 'key', id, label);
    ['label', 'homePath', 'portfolioPath', 'resumePath'].forEach((field) => {
      assertStringField(document, field, label);
    });
    assertObjectField(document, 'page', label);
    return;
  }

  if (collection === 'resumes') {
    assertMatchingStringField(document, 'key', id, label);
    assertStringField(document, 'audience', label);
    assertObjectField(document, 'digitalPage', label);
    assertObjectField(document, 'pdfPage', label);
    return;
  }

  if (collection === 'projects') {
    assertMatchingStringField(document, 'id', id, label);
    ['title', 'subtitle', 'image'].forEach((field) => {
      assertStringField(document, field, label);
    });
    return;
  }

  if (collection === 'tools') {
    assertMatchingStringField(document, 'slug', id, label);
    ['title', 'href', 'categoryId', 'summary'].forEach((field) => {
      assertStringField(document, field, label);
    });
  }
}

function validateCmsDocumentInput({ collection, id, document }) {
  const normalizedCollection = normalizeCmsCollection(collection);
  const normalizedId = normalizeCmsDocumentId(id);

  if (!normalizedCollection) {
    const err = new Error('Invalid CMS collection');
    err.code = 'CMS_INVALID_COLLECTION';
    throw err;
  }
  if (!normalizedId) {
    const err = new Error('Invalid CMS document id');
    err.code = 'CMS_INVALID_DOCUMENT_ID';
    throw err;
  }
  if (normalizedCollection === 'site' && !SITE_DOCUMENT_IDS.includes(normalizedId)) {
    const err = new Error('Invalid site document id');
    err.code = 'CMS_INVALID_DOCUMENT_ID';
    throw err;
  }

  assertPlainObject(document, 'CMS document');
  validateCmsDocumentShape({
    collection: normalizedCollection,
    id: normalizedId,
    document
  });

  const raw = JSON.stringify(document);
  if (byteLength(raw) > MAX_DOCUMENT_BYTES) {
    const err = new Error(`CMS document is too large (max ${MAX_DOCUMENT_BYTES} bytes)`);
    err.code = 'CMS_DOCUMENT_TOO_LARGE';
    throw err;
  }

  return {
    collection: normalizedCollection,
    id: normalizedId,
    document
  };
}

function readJson(absPath) {
  const raw = fs.readFileSync(absPath, 'utf8');
  return JSON.parse(raw);
}

function listJsonFiles(absDir) {
  if (!fs.existsSync(absDir)) return [];
  return fs.readdirSync(absDir)
    .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
    .sort((a, b) => a.localeCompare(b));
}

function listFileContentRecords(root) {
  const records = [];

  CMS_COLLECTIONS.forEach((collectionConfig) => {
    const absDir = path.join(root, collectionConfig.relDir);
    listJsonFiles(absDir).forEach((fileName) => {
      const id = normalizeCmsDocumentId(path.basename(fileName, '.json'));
      if (!id) {
        throw new Error(`Invalid CMS document file name: ${path.join(collectionConfig.relDir, fileName)}`);
      }
      const relPath = path.join(collectionConfig.relDir, fileName);
      const document = readJson(path.join(root, relPath));
      assertPlainObject(document, relPath);
      records.push({
        collection: collectionConfig.name,
        id,
        relPath,
        document
      });
    });
  });

  return records;
}

function groupContentRecords(records) {
  const grouped = {};
  CMS_COLLECTION_NAMES.forEach((name) => {
    grouped[name] = [];
  });

  (Array.isArray(records) ? records : []).forEach((record) => {
    const collection = normalizeCmsCollection(record && record.collection);
    const id = normalizeCmsDocumentId(record && record.id);
    if (!collection || !id) return;
    const document = record.document;
    assertPlainObject(document, `${collection}/${id}`);
    grouped[collection].push({
      collection,
      id,
      relPath: record.relPath || '',
      document,
      updatedAt: record.updatedAt || 0,
      updatedAtIso: record.updatedAtIso || '',
      updatedBy: record.updatedBy || '',
      revisionId: record.revisionId || ''
    });
  });

  return grouped;
}

function keyBy(items, keyName) {
  return items.reduce((acc, item) => {
    const key = String(item && item[keyName] ? item[keyName] : '').trim();
    if (!key) return acc;
    acc[key] = item;
    return acc;
  }, {});
}

function sortByOrderThenId(items, idKey = 'id') {
  return [...items].sort((a, b) => {
    const orderA = Number.isFinite(Number(a && a.order)) ? Number(a.order) : Number.MAX_SAFE_INTEGER;
    const orderB = Number.isFinite(Number(b && b.order)) ? Number(b.order) : Number.MAX_SAFE_INTEGER;
    if (orderA !== orderB) return orderA - orderB;
    const keyA = String((a && (a[idKey] || a.key || a.slug || a.title)) || '');
    const keyB = String((b && (b[idKey] || b.key || b.slug || b.title)) || '');
    return keyA.localeCompare(keyB);
  });
}

function sortDocumentsForCollection(collection, documents) {
  if (collection === 'pages') {
    return [...documents];
  }
  if (collection === 'audiences' || collection === 'resumes') {
    return sortByOrderThenId(documents, 'key');
  }
  if (collection === 'projects') {
    return sortByOrderThenId(documents, 'id');
  }
  if (collection === 'tools') {
    return sortByOrderThenId(documents, 'slug');
  }
  return [...documents];
}

function getDocumentByRecordId(records, id) {
  const found = (Array.isArray(records) ? records : []).find((record) => record.id === id);
  return found && found.document && typeof found.document === 'object' ? found.document : {};
}

function recordsToSiteContent(records) {
  const grouped = groupContentRecords(records);
  const siteRecords = grouped.site || [];

  const pages = sortDocumentsForCollection('pages', (grouped.pages || []).map((record) => record.document));
  const audiences = sortDocumentsForCollection('audiences', (grouped.audiences || []).map((record) => record.document));
  const resumes = sortDocumentsForCollection('resumes', (grouped.resumes || []).map((record) => record.document));
  const projects = sortDocumentsForCollection('projects', (grouped.projects || []).map((record) => record.document));
  const tools = sortDocumentsForCollection('tools', (grouped.tools || []).map((record) => record.document));

  return {
    site: {
      settings: getDocumentByRecordId(siteRecords, 'settings'),
      navigation: getDocumentByRecordId(siteRecords, 'navigation'),
      footer: getDocumentByRecordId(siteRecords, 'footer')
    },
    pages,
    pagesById: keyBy(pages, 'id'),
    audiences,
    audiencesByKey: keyBy(audiences, 'key'),
    resumes,
    resumesByKey: keyBy(resumes, 'key'),
    projects,
    projectsById: keyBy(projects, 'id'),
    tools
  };
}

function loadFileSiteContent(root) {
  return recordsToSiteContent(listFileContentRecords(root));
}

function stableStringify(value) {
  const seen = new WeakSet();
  const normalize = (input) => {
    if (!input || typeof input !== 'object') return input;
    if (seen.has(input)) return null;
    seen.add(input);
    if (Array.isArray(input)) return input.map((item) => normalize(item));
    return Object.keys(input)
      .sort((a, b) => a.localeCompare(b))
      .reduce((acc, key) => {
        acc[key] = normalize(input[key]);
        return acc;
      }, {});
  };
  return JSON.stringify(normalize(value), null, 2);
}

module.exports = {
  CMS_COLLECTIONS,
  CMS_COLLECTION_NAMES,
  MAX_DOCUMENT_BYTES,
  SITE_DOCUMENT_IDS,
  getCollectionConfig,
  groupContentRecords,
  keyBy,
  listFileContentRecords,
  loadFileSiteContent,
  normalizeCmsCollection,
  normalizeCmsDocumentId,
  recordsToSiteContent,
  sortByOrderThenId,
  stableStringify,
  validateCmsDocumentInput
};
