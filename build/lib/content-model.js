'use strict';

const fs = require('fs');
const path = require('path');

const COLLECTIONS = [
  { name: 'site', relDir: path.join('content', 'site') },
  { name: 'pages', relDir: path.join('content', 'pages') },
  { name: 'audiences', relDir: path.join('content', 'audiences') },
  { name: 'resumes', relDir: path.join('content', 'resumes') },
  { name: 'projects', relDir: path.join('content', 'projects') },
  { name: 'tools', relDir: path.join('content', 'tools') }
];
const COLLECTION_NAMES = COLLECTIONS.map((collection) => collection.name);
const COLLECTION_SET = new Set(COLLECTION_NAMES);

function assertPlainObject(document, label) {
  if (!document || typeof document !== 'object' || Array.isArray(document)) {
    throw new Error(`${label} must be a JSON object`);
  }
}

function isDocumentId(value) {
  const id = String(value || '').trim();
  return id.length <= 128 && /^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(id);
}

function listJsonFiles(absDir) {
  if (!fs.existsSync(absDir)) return [];
  return fs.readdirSync(absDir)
    .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
    .sort((a, b) => a.localeCompare(b));
}

function listFileContentRecords(root) {
  const records = [];
  COLLECTIONS.forEach((collection) => {
    const absDir = path.join(root, collection.relDir);
    listJsonFiles(absDir).forEach((fileName) => {
      const id = path.basename(fileName, '.json');
      const relPath = path.join(collection.relDir, fileName);
      if (!isDocumentId(id)) throw new Error(`Invalid content document file name: ${relPath}`);
      const document = JSON.parse(fs.readFileSync(path.join(root, relPath), 'utf8'));
      assertPlainObject(document, relPath);
      records.push({ collection: collection.name, id, relPath, document });
    });
  });
  return records;
}

function groupContentRecords(records) {
  const grouped = {};
  COLLECTION_NAMES.forEach((name) => { grouped[name] = []; });
  (Array.isArray(records) ? records : []).forEach((record) => {
    const collection = String(record && record.collection || '').trim();
    const id = String(record && record.id || '').trim();
    if (!COLLECTION_SET.has(collection) || !isDocumentId(id)) return;
    assertPlainObject(record.document, `${collection}/${id}`);
    grouped[collection].push({ collection, id, relPath: record.relPath || '', document: record.document });
  });
  return grouped;
}

function keyBy(items, keyName) {
  return items.reduce((acc, item) => {
    const key = String(item && item[keyName] ? item[keyName] : '').trim();
    if (key) acc[key] = item;
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
  if (collection === 'audiences' || collection === 'resumes') return sortByOrderThenId(documents, 'key');
  if (collection === 'projects') return sortByOrderThenId(documents, 'id');
  if (collection === 'tools') return sortByOrderThenId(documents, 'slug');
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

module.exports = {
  COLLECTIONS,
  groupContentRecords,
  listFileContentRecords,
  loadFileSiteContent,
  recordsToSiteContent,
  sortByOrderThenId
};
