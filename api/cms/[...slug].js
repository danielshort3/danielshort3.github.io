/*
  Local CMS API router: /api/cms/content

  This endpoint is intentionally local-only. It edits repo JSON files under
  content/ during development; deployed hosts should not expose a writeable CMS.
*/
'use strict';

const fs = require('fs');
const path = require('path');
const {
  sendJson,
  readJson
} = require('../_lib/tools-api');
const {
  CMS_COLLECTIONS,
  CMS_COLLECTION_NAMES,
  listFileContentRecords,
  loadFileSiteContent,
  normalizeCmsCollection,
  normalizeCmsDocumentId,
  validateCmsDocumentInput
} = require('../_lib/cms-content-model');
const {
  getWidgetDefinitions,
  renderVisualPageBody
} = require('../_lib/cms-widgets');
const {
  getCurrentDocument,
  listAllCurrentDocuments,
  listCurrentDocuments,
  saveCurrentDocument
} = require('../_lib/cms-file-store');
const {
  deleteLibraryItem,
  listAllLibraryItems,
  listLibraryItems,
  normalizeLibraryType,
  saveLibraryItem
} = require('../_lib/cms-library-store');
const {
  createJsonDiffSummary,
  getSnapshot,
  listSnapshots
} = require('../_lib/cms-snapshot-store');
const {
  renderFooter,
  renderFullPage,
  renderGamesDirectoryBody,
  renderHeader,
  renderToolsDirectoryBody
} = require('../../build/lib/cms-renderers');
const {
  renderProjectPage
} = require('../../build/generate-project-pages');

const pickQuery = (value) => Array.isArray(value) ? value[0] : value;
const root = path.resolve(__dirname, '..', '..');

function getEndpointFromRequest(req) {
  const querySlug = req.query && req.query.slug;
  if (Array.isArray(querySlug)) return querySlug[0] || '';
  if (typeof querySlug === 'string') return querySlug;
  try {
    const url = new URL(req.url, 'http://localhost');
    const match = url.pathname.match(/\/api\/cms\/(.+)$/);
    if (!match) return '';
    const raw = decodeURIComponent(match[1]);
    return raw.split('/')[0] || '';
  } catch {
    return '';
  }
}

function getQuery(req) {
  if (req.query && typeof req.query === 'object') return req.query;
  try {
    const url = new URL(req.url, 'http://localhost');
    return Object.fromEntries(url.searchParams.entries());
  } catch {
    return {};
  }
}

function getFirstHeaderValue(value) {
  if (Array.isArray(value)) return value[0] || '';
  return value || '';
}

function normalizeHostName(value) {
  const raw = String(getFirstHeaderValue(value) || '').trim().toLowerCase();
  if (!raw) return '';
  if (raw.startsWith('[')) {
    const end = raw.indexOf(']');
    return end > 0 ? raw.slice(1, end) : raw;
  }
  return raw.split(':')[0];
}

function isLocalHostValue(value) {
  const host = normalizeHostName(value);
  return host === 'localhost'
    || host === '127.0.0.1'
    || host === '::1';
}

function allowsPrivateHostAccess() {
  const value = String(process.env.CMS_ALLOW_PRIVATE_HOSTS || '').trim().toLowerCase();
  return value === '1' || value === 'true' || value === 'yes';
}

function isPrivateIpv4Host(value) {
  const host = normalizeHostName(value);
  const parts = host.split('.').map((part) => Number(part));
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }
  return parts[0] === 10
    || (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
    || (parts[0] === 192 && parts[1] === 168);
}

function isAllowedHostValue(value) {
  return isLocalHostValue(value)
    || (allowsPrivateHostAccess() && isPrivateIpv4Host(value));
}

function isLocalOriginValue(value) {
  const raw = String(getFirstHeaderValue(value) || '').trim();
  if (!raw) return true;
  if (raw === 'null') return false;
  try {
    const url = new URL(raw);
    return (url.protocol === 'http:' || url.protocol === 'https:')
      && isAllowedHostValue(url.host);
  } catch {
    return false;
  }
}

function isLocalRequest(req) {
  const host = req.headers && req.headers.host;
  if (host && !isAllowedHostValue(host)) return false;
  if (req.headers && !isLocalOriginValue(req.headers.origin)) return false;
  if (isAllowedHostValue(host)) return true;

  const remote = req.socket && req.socket.remoteAddress;
  return remote === '127.0.0.1'
    || remote === '::1'
    || remote === '::ffff:127.0.0.1';
}

function handleStorageError(res, err) {
  if (err && (
    err.code === 'CMS_INVALID_COLLECTION' ||
    err.code === 'CMS_INVALID_DOCUMENT_ID' ||
    err.code === 'CMS_INVALID_DOCUMENT_SHAPE' ||
    err.code === 'CMS_DOCUMENT_TOO_LARGE' ||
    err.code === 'CMS_INVALID_LIBRARY_TYPE' ||
    err.code === 'CMS_INVALID_LIBRARY_ID' ||
    err.code === 'CMS_INVALID_LIBRARY_ITEM' ||
    err.code === 'CMS_LIBRARY_ITEM_TOO_LARGE' ||
    err.code === 'CMS_INVALID_SNAPSHOT' ||
    err.code === 'CMS_SNAPSHOT_TOO_LARGE'
  )) {
    sendJson(res, 400, { ok: false, error: err.message });
    return true;
  }
  if (err && err.code === 'CMS_REVISION_CONFLICT') {
    sendJson(res, 409, { ok: false, error: err.message });
    return true;
  }
  return false;
}

function serializeContentRecord(record) {
  if (!record) return null;
  return {
    collection: record.collection,
    id: record.id,
    relPath: record.relPath || '',
    document: record.document,
    updatedAt: record.updatedAt,
    updatedAtIso: record.updatedAtIso,
    updatedBy: record.updatedBy || '',
    revisionId: record.revisionId || ''
  };
}

function serializeLibraryRecord(record) {
  if (!record) return null;
  return {
    type: record.type,
    id: record.id,
    relPath: record.relPath || '',
    item: record.item,
    updatedAt: record.updatedAt,
    updatedAtIso: record.updatedAtIso,
    revisionId: record.revisionId || ''
  };
}

function getAudienceLabel(content, audienceKey) {
  const audience = content.audiencesByKey[audienceKey];
  if (audience && audience.brandNavPrimary) return audience.brandNavPrimary;
  const defaultAudience = content.audiencesByKey[content.site.settings.defaultAudience];
  return defaultAudience && defaultAudience.brandNavPrimary
    ? defaultAudience.brandNavPrimary
    : 'Data Analytics';
}

function preparePreviewPage(page, content) {
  if (!page || typeof page !== 'object' || Array.isArray(page)) {
    const err = new Error('Preview page must be a JSON object');
    err.code = 'CMS_INVALID_DOCUMENT_SHAPE';
    throw err;
  }

  if (page.template === 'tools-directory') {
    return {
      ...page,
      bodyHtml: renderToolsDirectoryBody(page, content.tools)
    };
  }

  if (page.template === 'games-directory') {
    return {
      ...page,
      bodyHtml: renderGamesDirectoryBody(page)
    };
  }

  if (page.template === 'visual-page') {
    return {
      ...page,
      bodyHtml: renderVisualPageBody(page)
    };
  }

  if (typeof page.bodyHtml !== 'string') {
    const err = new Error('Preview page must include bodyHtml');
    err.code = 'CMS_INVALID_DOCUMENT_SHAPE';
    throw err;
  }

  return page;
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function mergePlainObjects(target, source) {
  if (!isPlainObject(source)) return target;
  Object.entries(source).forEach(([key, value]) => {
    if (isPlainObject(value)) {
      if (!isPlainObject(target[key])) target[key] = {};
      mergePlainObjects(target[key], value);
      return;
    }
    target[key] = value;
  });
  return target;
}

function getPreviewSite(content, overrides) {
  const site = {
    settings: { ...(content.site.settings || {}) },
    navigation: JSON.parse(JSON.stringify(content.site.navigation || {})),
    footer: JSON.parse(JSON.stringify(content.site.footer || {}))
  };
  if (isPlainObject(overrides)) {
    ['settings', 'navigation', 'footer'].forEach((key) => {
      if (isPlainObject(overrides[key])) mergePlainObjects(site[key], overrides[key]);
    });
  }
  return site;
}

async function handleContent(req, res) {
  if (req.method === 'GET') {
    const query = getQuery(req);
    const collection = normalizeCmsCollection(pickQuery(query.collection));
    const id = normalizeCmsDocumentId(pickQuery(query.id));

    try {
      if (collection && id) {
        const record = await getCurrentDocument({ collection, id });
        if (!record) {
          sendJson(res, 404, { ok: false, error: 'CMS document not found' });
          return;
        }
        sendJson(res, 200, { ok: true, document: serializeContentRecord(record), collections: CMS_COLLECTIONS });
        return;
      }

      if (collection) {
        const documents = await listCurrentDocuments(collection);
        sendJson(res, 200, {
          ok: true,
          collection,
          collections: CMS_COLLECTIONS,
          documents: documents.map(serializeContentRecord)
        });
        return;
      }

      const grouped = await listAllCurrentDocuments();
      const content = {};
      CMS_COLLECTION_NAMES.forEach((name) => {
        content[name] = (grouped[name] || []).map(serializeContentRecord);
      });
      sendJson(res, 200, { ok: true, collections: CMS_COLLECTIONS, content });
      return;
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS file storage unavailable' });
      return;
    }
  }

  if (req.method === 'PUT') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }

    try {
      const input = validateCmsDocumentInput({
        collection: body && body.collection,
        id: body && body.id,
        document: body && body.document
      });
      const record = await saveCurrentDocument({
        ...input,
        expectedRevisionId: typeof (body && body.expectedRevisionId) === 'string'
          ? body.expectedRevisionId
          : undefined
      });
      sendJson(res, 200, { ok: true, document: serializeContentRecord(record) });
      return;
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS file storage unavailable' });
      return;
    }
  }

  sendJson(res, 405, { ok: false, error: 'Method not allowed' });
}

async function handlePreview(req, res) {
  if (req.method !== 'POST') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
    return;
  }

  try {
    const content = loadFileSiteContent(root);
    const site = getPreviewSite(content, body && body.site);
    const page = preparePreviewPage(body && body.page, content);
    const html = renderFullPage({
      settings: site.settings,
      navigation: site.navigation,
      footer: site.footer,
      projectsById: content.projectsById,
      pagesById: content.pagesById,
      tools: content.tools,
      page,
      audienceLabel: getAudienceLabel(content, page.audienceKey)
    });
    sendJson(res, 200, { ok: true, html, warnings: [] });
  } catch (err) {
    if (handleStorageError(res, err)) return;
    sendJson(res, 500, { ok: false, error: 'Unable to render CMS preview' });
  }
}

function getProjectPreviewList(content, project) {
  const projectId = String(project && project.id ? project.id : '').trim();
  const projects = (Array.isArray(content.projects) ? content.projects : []).map((item) => {
    return String(item && item.id ? item.id : '').trim() === projectId ? project : item;
  });
  if (projectId && !projects.some((item) => String(item && item.id ? item.id : '').trim() === projectId)) {
    projects.push(project);
  }
  return projects;
}

function getToolPreviewList(content, tool) {
  const slug = String(tool && tool.slug ? tool.slug : '').trim();
  const previewTool = {
    ...tool,
    hidden: false,
    visibility: 'public'
  };
  const tools = (Array.isArray(content.tools) ? content.tools : []).map((item) => {
    return String(item && item.slug ? item.slug : '').trim() === slug ? previewTool : item;
  });
  if (slug && !tools.some((item) => String(item && item.slug ? item.slug : '').trim() === slug)) {
    tools.push(previewTool);
  }
  return tools;
}

function decorateToolPreviewHtml(html, tool) {
  const slug = String(tool && tool.slug ? tool.slug : '').trim();
  if (!slug) return html;
  const marker = [
    '<style>',
    '.cms-tool-preview-highlight{outline:4px solid rgba(47,125,225,.92)!important;outline-offset:5px!important;box-shadow:0 0 0 8px rgba(47,125,225,.16)!important;}',
    '</style>',
    '<script>',
    '(function(){',
    `var slug=${JSON.stringify(slug)};`,
    'function focusTool(){',
    'var link=document.querySelector(".tool-card a[href=\\"tools/"+slug+"\\"],.tool-card a[href$=\\"/"+slug+"\\"]");',
    'var card=link&&link.closest(".tool-card");',
    'if(!card)return;',
    'card.hidden=false;',
    'card.classList.add("cms-tool-preview-highlight");',
    'card.scrollIntoView({block:"center",inline:"nearest"});',
    '}',
    'if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",focusTool,{once:true});else setTimeout(focusTool,60);',
    '}());',
    '</script>'
  ].join('');
  return String(html || '').includes('</body>')
    ? String(html).replace('</body>', `${marker}</body>`)
    : `${html}${marker}`;
}

function getProjectsById(projects) {
  return (Array.isArray(projects) ? projects : []).reduce((acc, project) => {
    const id = String(project && project.id ? project.id : '').trim();
    if (id) acc[id] = project;
    return acc;
  }, {});
}

function hydrateProjectPreviewChrome(html, { content, site, projects }) {
  const headerHtml = renderHeader({
    settings: site.settings,
    navigation: site.navigation,
    projectsById: getProjectsById(projects),
    pagesById: content.pagesById,
    tools: content.tools,
    audienceLabel: getAudienceLabel(content, site.settings && site.settings.defaultAudience)
  });
  const footerHtml = renderFooter({
    footer: site.footer,
    year: new Date().getFullYear()
  });
  return String(html || '')
    .replace('<header id="combined-header-nav"></header>', headerHtml)
    .replace(/<footer>[\s\S]*?<\/footer>/, footerHtml);
}

async function handleProjectPreview(req, res) {
  if (req.method !== 'POST') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
    return;
  }

  try {
    const project = body && body.project;
    if (!project || typeof project !== 'object' || Array.isArray(project) || !String(project.id || '').trim()) {
      sendJson(res, 400, { ok: false, error: 'Preview project must include an id' });
      return;
    }
    const content = loadFileSiteContent(root);
    const site = getPreviewSite(content, body && body.site);
    const projects = getProjectPreviewList(content, project);
    const projectIndex = projects.findIndex((item) => String(item && item.id ? item.id : '').trim() === String(project.id).trim());
    const html = hydrateProjectPreviewChrome(renderProjectPage(project, {
      projects,
      index: projectIndex
    }), { content, site, projects });
    sendJson(res, 200, { ok: true, html, warnings: [] });
  } catch (err) {
    sendJson(res, 500, { ok: false, error: 'Unable to render project preview' });
  }
}

async function handleToolPreview(req, res) {
  if (req.method !== 'POST') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
    return;
  }

  try {
    const tool = body && body.tool;
    if (!tool || typeof tool !== 'object' || Array.isArray(tool) || !String(tool.slug || '').trim()) {
      sendJson(res, 400, { ok: false, error: 'Preview tool must include a slug' });
      return;
    }
    const content = loadFileSiteContent(root);
    const site = getPreviewSite(content, body && body.site);
    const tools = getToolPreviewList(content, tool);
    const page = content.pagesById.tools || (Array.isArray(content.pages) ? content.pages.find((item) => item.template === 'tools-directory') : null);
    if (!page) {
      sendJson(res, 404, { ok: false, error: 'Tools directory page is not available' });
      return;
    }
    const previewContent = {
      ...content,
      tools
    };
    const preparedPage = preparePreviewPage(page, previewContent);
    const html = decorateToolPreviewHtml(renderFullPage({
      settings: site.settings,
      navigation: site.navigation,
      footer: site.footer,
      projectsById: content.projectsById,
      pagesById: content.pagesById,
      tools,
      page: preparedPage,
      audienceLabel: getAudienceLabel(content, site.settings && site.settings.defaultAudience)
    }), tool);
    sendJson(res, 200, { ok: true, html, warnings: [] });
  } catch (err) {
    if (handleStorageError(res, err)) return;
    sendJson(res, 500, { ok: false, error: 'Unable to render tool preview' });
  }
}

async function handleWidgets(req, res) {
  if (req.method !== 'GET') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }
  sendJson(res, 200, { ok: true, widgets: getWidgetDefinitions() });
}

function normalizeAssetReference(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  let next = raw
    .replace(/^url\((["']?)(.*?)\1\)$/i, '$2')
    .replace(/^https?:\/\/[^/]+\/+/i, '')
    .replace(/^\/+/, '');
  try {
    next = decodeURIComponent(next);
  } catch {}
  return next.split(/[?#]/)[0].replace(/\\/g, '/');
}

function readPngDimensions(buffer) {
  if (buffer.length < 24 || buffer.toString('ascii', 1, 4) !== 'PNG') return null;
  return {
    width: buffer.readUInt32BE(16),
    height: buffer.readUInt32BE(20)
  };
}

function readGifDimensions(buffer) {
  if (buffer.length < 10 || buffer.toString('ascii', 0, 3) !== 'GIF') return null;
  return {
    width: buffer.readUInt16LE(6),
    height: buffer.readUInt16LE(8)
  };
}

function readJpegDimensions(buffer) {
  if (buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) return null;
  let offset = 2;
  while (offset + 9 < buffer.length) {
    if (buffer[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = buffer[offset + 1];
    if (marker === 0xd8 || marker === 0xd9) {
      offset += 2;
      continue;
    }
    const length = buffer.readUInt16BE(offset + 2);
    if (!length || offset + 2 + length > buffer.length) return null;
    if ((marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) || (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf)) {
      return {
        width: buffer.readUInt16BE(offset + 7),
        height: buffer.readUInt16BE(offset + 5)
      };
    }
    offset += 2 + length;
  }
  return null;
}

function readWebpDimensions(buffer) {
  if (buffer.length < 30 || buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WEBP') return null;
  const type = buffer.toString('ascii', 12, 16);
  if (type === 'VP8X' && buffer.length >= 30) {
    return {
      width: 1 + buffer.readUIntLE(24, 3),
      height: 1 + buffer.readUIntLE(27, 3)
    };
  }
  if (type === 'VP8L' && buffer.length >= 25 && buffer[20] === 0x2f) {
    const bits = buffer.readUInt32LE(21);
    return {
      width: 1 + (bits & 0x3fff),
      height: 1 + ((bits >> 14) & 0x3fff)
    };
  }
  if (type === 'VP8 ' && buffer.length >= 30 && buffer[23] === 0x9d && buffer[24] === 0x01 && buffer[25] === 0x2a) {
    return {
      width: buffer.readUInt16LE(26) & 0x3fff,
      height: buffer.readUInt16LE(28) & 0x3fff
    };
  }
  return null;
}

function readSvgDimensions(absPath) {
  try {
    const raw = fs.readFileSync(absPath, 'utf8').slice(0, 4096);
    const width = raw.match(/\bwidth=["']?([0-9.]+)/i);
    const height = raw.match(/\bheight=["']?([0-9.]+)/i);
    if (width && height) {
      return {
        width: Math.round(Number(width[1])) || 0,
        height: Math.round(Number(height[1])) || 0
      };
    }
    const viewBox = raw.match(/\bviewBox=["'][^"']*?\s([0-9.]+)\s+([0-9.]+)["']/i);
    if (viewBox) {
      return {
        width: Math.round(Number(viewBox[1])) || 0,
        height: Math.round(Number(viewBox[2])) || 0
      };
    }
  } catch {}
  return null;
}

function readImageDimensions(absPath) {
  const ext = path.extname(absPath).toLowerCase();
  if (ext === '.svg') return readSvgDimensions(absPath);
  try {
    const buffer = fs.readFileSync(absPath);
    if (ext === '.png') return readPngDimensions(buffer);
    if (ext === '.gif') return readGifDimensions(buffer);
    if (ext === '.jpg' || ext === '.jpeg') return readJpegDimensions(buffer);
    if (ext === '.webp') return readWebpDimensions(buffer);
  } catch {}
  return null;
}

function getDocumentDisplayTitle(record) {
  const document = record && record.document ? record.document : {};
  return String(document.title || document.label || document.siteName || document.slug || document.key || record.id || '').trim();
}

function addAssetUsage(assetMap, assetPath, record, fieldPath, options = {}) {
  const normalized = normalizeAssetReference(assetPath);
  const asset = assetMap.get(normalized);
  if (!asset) return;
  const altText = String(options.altText || '').trim();
  asset.usageCount += 1;
  if (altText) asset.altTexts.push(altText);
  if (options.missingAlt) asset.missingAlt = true;
  if (asset.usedIn.length < 16) {
    asset.usedIn.push({
      collection: record.collection,
      id: record.id,
      title: getDocumentDisplayTitle(record),
      fieldPath,
      missingAlt: Boolean(options.missingAlt),
      altText
    });
  }
}

function getSiblingAltText(parent, key) {
  if (!parent || typeof parent !== 'object' || Array.isArray(parent)) return '';
  const candidates = [
    `${key}Alt`,
    key.replace(/(?:Src|Image)$/i, 'Alt'),
    'imageAlt',
    'logoAlt',
    'alt',
    'label',
    'title'
  ];
  for (const candidate of candidates) {
    if (typeof parent[candidate] === 'string' && parent[candidate].trim()) return parent[candidate].trim();
  }
  return '';
}

function scanHtmlImages(html, record, fieldPath, assetMap) {
  String(html || '').replace(/<img\b[^>]*>/gi, (tag) => {
    const src = tag.match(/\bsrc=["']([^"']+)["']/i);
    if (!src) return tag;
    const alt = tag.match(/\balt=["']([^"']*)["']/i);
    const altText = alt ? alt[1].trim() : '';
    addAssetUsage(assetMap, src[1], record, fieldPath, {
      altText,
      missingAlt: !altText
    });
    return tag;
  });
}

function scanAssetString(value, record, fieldPath, parent, key, assetMap) {
  const raw = String(value || '');
  if (!raw) return;
  if (/<img\b/i.test(raw)) scanHtmlImages(raw, record, fieldPath, assetMap);
  const exact = normalizeAssetReference(raw);
  if (assetMap.has(exact)) {
    const altText = getSiblingAltText(parent, key);
    addAssetUsage(assetMap, exact, record, fieldPath, {
      altText,
      missingAlt: !altText && /(?:image|thumb|logo|src|icon)/i.test(String(key || ''))
    });
    return;
  }
  const matches = raw.match(/(?:^|["'(=\s])\/?(img\/[A-Za-z0-9._~:/?#@!$&%+\-,;=]+)/g) || [];
  matches.forEach((match) => {
    const assetPath = normalizeAssetReference(match.replace(/^[^i]*(img\/)/, 'img/'));
    if (!assetMap.has(assetPath)) return;
    const altText = getSiblingAltText(parent, key);
    addAssetUsage(assetMap, assetPath, record, fieldPath, {
      altText,
      missingAlt: !altText && /(?:image|thumb|logo|src|icon|html)/i.test(String(key || fieldPath || ''))
    });
  });
}

function scanRecordForAssetUsage(record, assetMap) {
  const visit = (value, fieldPath, parent, key) => {
    if (typeof value === 'string') {
      scanAssetString(value, record, fieldPath, parent, key, assetMap);
      return;
    }
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${fieldPath}.${index}`, value, String(index)));
      return;
    }
    Object.entries(value).forEach(([entryKey, entryValue]) => {
      visit(entryValue, fieldPath ? `${fieldPath}.${entryKey}` : entryKey, value, entryKey);
    });
  };
  visit(record.document, '', null, '');
}

function collectMediaAssets() {
  const mediaRoot = path.join(root, 'img');
  const allowedExtensions = new Set(['.gif', '.jpg', '.jpeg', '.png', '.svg', '.webp']);
  const assets = [];
  const walk = (absDir) => {
    if (!fs.existsSync(absDir)) return;
    fs.readdirSync(absDir, { withFileTypes: true }).forEach((entry) => {
      if (entry.name.startsWith('.')) return;
      const absPath = path.join(absDir, entry.name);
      if (entry.isDirectory()) {
        walk(absPath);
        return;
      }
      if (!entry.isFile() || !allowedExtensions.has(path.extname(entry.name).toLowerCase())) return;
      const stat = fs.statSync(absPath);
      const dimensions = readImageDimensions(absPath) || {};
      assets.push({
        path: path.relative(root, absPath).split(path.sep).join('/'),
        size: stat.size,
        updatedAt: stat.mtimeMs,
        width: dimensions.width || 0,
        height: dimensions.height || 0,
        usageCount: 0,
        usedIn: [],
        altTexts: [],
        missingAlt: false
      });
    });
  };
  walk(mediaRoot);
  return assets.sort((a, b) => a.path.localeCompare(b.path));
}

function listMediaAssets() {
  const assets = collectMediaAssets();
  const assetMap = new Map(assets.map((asset) => [asset.path, asset]));
  listFileContentRecords(root).forEach((record) => scanRecordForAssetUsage(record, assetMap));
  assets.forEach((asset) => {
    asset.altTexts = [...new Set(asset.altTexts)].slice(0, 8);
    asset.status = asset.usageCount
      ? (asset.missingAlt ? 'missing-alt' : 'used')
      : 'unused';
  });
  return assets;
}

async function handleMedia(req, res) {
  if (req.method !== 'GET') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }
  try {
    sendJson(res, 200, { ok: true, assets: listMediaAssets() });
  } catch {
    sendJson(res, 500, { ok: false, error: 'CMS media catalog unavailable' });
  }
}

function createHealthIssue({ severity = 'info', title, detail, record, fieldPath = '', action = 'open-record' }) {
  return {
    severity,
    title,
    detail: detail || '',
    collection: record && record.collection ? record.collection : '',
    id: record && record.id ? record.id : '',
    titleText: record ? getDocumentDisplayTitle(record) : '',
    fieldPath,
    action
  };
}

function normalizeInternalHref(value) {
  const raw = String(value || '').trim();
  if (!raw || raw.startsWith('#')) return '';
  if (/^(?:https?:|mailto:|tel:|sms:|data:|blob:|javascript:)/i.test(raw)) return '';
  const withoutHash = raw.split('#')[0].split('?')[0].trim();
  if (!withoutHash || withoutHash.startsWith('#')) return '';
  return normalizeAssetReference(withoutHash);
}

function collectContentLinks(record) {
  const links = [];
  const add = (href, fieldPath) => {
    const normalized = normalizeInternalHref(href);
    if (normalized) links.push({ href: normalized, fieldPath });
  };
  const visit = (value, fieldPath, key) => {
    if (typeof value === 'string') {
      if (/href=["']/i.test(value)) {
        value.replace(/\bhref=["']([^"']+)["']/gi, (match, href) => {
          add(href, fieldPath);
          return match;
        });
      }
      if (/^(?:\/|\.?\.?\/|[A-Za-z0-9._-]+\/)/.test(value) && /(?:html?|documents\/|img\/|portfolio\/|tools\/|resume|contact|search|sitemap|privacy)/i.test(value)) {
        add(value, fieldPath);
      }
      return;
    }
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, `${fieldPath}.${index}`, String(index)));
      return;
    }
    Object.entries(value).forEach(([entryKey, entryValue]) => {
      visit(entryValue, fieldPath ? `${fieldPath}.${entryKey}` : entryKey, entryKey);
    });
  };
  visit(record.document, '', '');
  return links;
}

function buildValidInternalPaths(records) {
  const paths = new Set(['', 'index', 'index.html']);
  records.forEach((record) => {
    const document = record.document || {};
    if (record.collection === 'pages') {
      [document.canonicalPath, document.outputPath, `${record.id}`, `${record.id}.html`, `pages/${record.id}.html`]
        .forEach((item) => {
          const normalized = normalizeInternalHref(item);
          if (normalized) paths.add(normalized.replace(/\.html$/i, ''));
          if (normalized) paths.add(normalized);
        });
    }
    if (record.collection === 'projects') {
      paths.add(`portfolio/${record.id}`);
      paths.add(`portfolio/${record.id}.html`);
      paths.add(`pages/portfolio/${record.id}.html`);
    }
    if (record.collection === 'tools') {
      const href = normalizeInternalHref(document.href || `tools/${record.id}`);
      if (href) {
        paths.add(href);
        paths.add(href.replace(/\.html$/i, ''));
      }
    }
  });
  return paths;
}

function internalPathExists(href, validPaths) {
  const normalized = normalizeInternalHref(href);
  if (!normalized) return true;
  if (validPaths.has(normalized) || validPaths.has(normalized.replace(/\.html$/i, ''))) return true;
  const candidates = [
    normalized,
    `${normalized}.html`,
    path.join('pages', normalized),
    path.join('pages', `${normalized}.html`)
  ];
  if (normalized.startsWith('tools/')) {
    const slug = normalized.replace(/^tools\//, '').replace(/\.html$/i, '');
    candidates.push(path.join('pages', `${slug}.html`));
  }
  return candidates.some((candidate) => fs.existsSync(path.join(root, candidate)));
}

function getPageLikeDocuments(record) {
  const document = record.document || {};
  if (record.collection === 'pages') return [{ page: document, path: '' }];
  if (record.collection === 'audiences' && document.page) return [{ page: document.page, path: 'page' }];
  if (record.collection === 'resumes') {
    return [
      document.digitalPage ? { page: document.digitalPage, path: 'digitalPage' } : null,
      document.pdfPage ? { page: document.pdfPage, path: 'pdfPage' } : null
    ].filter(Boolean);
  }
  return [];
}

function countLegacySections(page) {
  if (!page || typeof page !== 'object') return 0;
  if (page.template === 'raw-body') return 1;
  if (!Array.isArray(page.sections)) return 0;
  return page.sections.filter((section) => section && (section.type === 'legacy-html' || section.html || (section.props && section.props.html))).length;
}

function buildCmsHealthReport() {
  const records = listFileContentRecords(root);
  const issues = [];
  const validPaths = buildValidInternalPaths(records);
  const mediaAssets = listMediaAssets();

  records.forEach((record) => {
    getPageLikeDocuments(record).forEach(({ page, path: pagePath }) => {
      if (!String(page.title || '').trim()) {
        issues.push(createHealthIssue({ severity: 'error', title: 'Missing page title', detail: 'Add a browser title before publishing.', record, fieldPath: `${pagePath}.title` }));
      }
      if (!String(page.description || '').trim()) {
        issues.push(createHealthIssue({ severity: 'error', title: 'Missing meta description', detail: 'Add a search snippet for this page.', record, fieldPath: `${pagePath}.description` }));
      } else if (String(page.description).trim().length < 70) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Short meta description', detail: 'The description is likely too short for a useful search/social preview.', record, fieldPath: `${pagePath}.description` }));
      }
      if (!String(page.canonicalPath || '').trim()) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Missing canonical path', detail: 'Add a canonical URL path for social previews and SEO.', record, fieldPath: `${pagePath}.canonicalPath` }));
      }
      const legacyCount = countLegacySections(page);
      if (legacyCount > 0) {
        issues.push(createHealthIssue({ severity: 'info', title: 'Legacy HTML section', detail: `${legacyCount} section${legacyCount === 1 ? '' : 's'} still use raw HTML. Convert important sections to widgets over time.`, record, fieldPath: pagePath || 'page' }));
      }
    });

    if (record.collection === 'projects') {
      const project = record.document || {};
      if (!String(project.imageAlt || '').trim()) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Project image uses fallback alt text', detail: 'Add explicit alt text for the main project image.', record, fieldPath: 'imageAlt' }));
      }
      if (!Number(project.imageWidth) || !Number(project.imageHeight)) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Project image dimensions missing', detail: 'Set imageWidth and imageHeight to reduce layout shift and improve social previews.', record, fieldPath: 'imageWidth' }));
      }
      ['problem'].forEach((field) => {
        if (!String(project[field] || '').trim()) {
          issues.push(createHealthIssue({ severity: 'warning', title: `Missing project ${field}`, detail: 'Portfolio case studies work better with a clear problem statement.', record, fieldPath: field }));
        }
      });
      ['actions', 'results', 'resources', 'caseStudy'].forEach((field) => {
        if (!Array.isArray(project[field]) || !project[field].length) {
          issues.push(createHealthIssue({ severity: 'warning', title: `Missing project ${field}`, detail: 'Add structured portfolio detail for recruiters and hiring managers.', record, fieldPath: field }));
        }
      });
      if (!Array.isArray(project.audiences) || !project.audiences.length) {
        issues.push(createHealthIssue({ severity: 'info', title: 'Project has no audience tags', detail: 'Audience tags help preview the portfolio for analytics, tourism, and data science roles.', record, fieldPath: 'audiences' }));
      }
    }

    if (record.collection === 'tools') {
      const tool = record.document || {};
      if (tool.visibility === 'public' && tool.hidden) {
        issues.push(createHealthIssue({ severity: 'info', title: 'Public tool hidden from directory', detail: 'This tool is public but hidden from the generated tools directory.', record, fieldPath: 'hidden' }));
      }
      if (String(tool.href || '').trim() && !internalPathExists(tool.href, validPaths)) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Tool href may be missing', detail: `Could not find a local route or file for ${tool.href}.`, record, fieldPath: 'href' }));
      }
    }

    collectContentLinks(record).forEach((link) => {
      if (!internalPathExists(link.href, validPaths)) {
        issues.push(createHealthIssue({ severity: 'warning', title: 'Internal link may be broken', detail: `Could not find a local route or file for ${link.href}.`, record, fieldPath: link.fieldPath }));
      }
    });
  });

  mediaAssets
    .filter((asset) => asset.usageCount > 0 && asset.missingAlt)
    .slice(0, 80)
    .forEach((asset) => {
      const usage = asset.usedIn.find((item) => item.missingAlt) || asset.usedIn[0] || {};
      issues.push(createHealthIssue({
        severity: 'warning',
        title: 'Image usage missing alt text',
        detail: asset.path,
        record: usage.collection && usage.id ? { collection: usage.collection, id: usage.id, document: { title: usage.title || usage.id } } : null,
        fieldPath: usage.fieldPath || ''
      }));
    });

  const counts = issues.reduce((acc, issue) => {
    acc[issue.severity] = (acc[issue.severity] || 0) + 1;
    return acc;
  }, { error: 0, warning: 0, info: 0 });

  const deployReady = !counts.error;
  return {
    generatedAt: new Date().toISOString(),
    deployReady,
    counts: {
      records: records.length,
      mediaAssets: mediaAssets.length,
      unusedMedia: mediaAssets.filter((asset) => !asset.usageCount).length,
      errors: counts.error || 0,
      warnings: counts.warning || 0,
      info: counts.info || 0
    },
    summary: deployReady
      ? (counts.warning ? 'No blocking CMS issues. Review warnings before deploying.' : 'CMS content is ready for a manual build and deploy.')
      : 'Fix blocking CMS issues before deploying.',
    checklist: [
      { id: 'blocking-issues', label: 'No blocking errors', complete: !counts.error },
      { id: 'metadata', label: 'Pages have titles and descriptions', complete: !issues.some((issue) => /title|description/i.test(issue.title) && issue.severity === 'error') },
      { id: 'media-alt', label: 'Used images have alt text', complete: !mediaAssets.some((asset) => asset.usageCount > 0 && asset.missingAlt) },
      { id: 'snapshots', label: 'Local save history is enabled', complete: true }
    ],
    issues: issues.slice(0, 240)
  };
}

async function handleHealth(req, res) {
  if (req.method !== 'GET') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }
  try {
    sendJson(res, 200, { ok: true, health: buildCmsHealthReport() });
  } catch {
    sendJson(res, 500, { ok: false, error: 'CMS health report unavailable' });
  }
}

function serializeSnapshotRecord(snapshot, includeDocument = false) {
  if (!snapshot) return null;
  const base = {
    snapshotId: snapshot.snapshotId,
    collection: snapshot.collection,
    documentId: snapshot.documentId,
    relPath: snapshot.relPath || '',
    filePath: snapshot.filePath || '',
    reason: snapshot.reason || '',
    createdAt: snapshot.createdAt || '',
    revisionId: snapshot.revisionId || '',
    updatedAt: snapshot.updatedAt || 0,
    updatedAtIso: snapshot.updatedAtIso || ''
  };
  if (includeDocument) base.document = snapshot.document;
  return base;
}

async function handleSnapshots(req, res) {
  if (req.method === 'GET') {
    const query = getQuery(req);
    const limit = Math.max(1, Math.min(80, Number(pickQuery(query.limit)) || 24));
    try {
      const snapshots = listSnapshots({
        collection: pickQuery(query.collection),
        id: pickQuery(query.id)
      }).slice(0, limit);
      sendJson(res, 200, {
        ok: true,
        snapshots: snapshots.map((snapshot) => serializeSnapshotRecord(snapshot, pickQuery(query.includeDocument) === 'true'))
      });
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS snapshots unavailable' });
    }
    return;
  }

  if (req.method === 'POST') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }
    try {
      const snapshot = getSnapshot(body && (body.snapshotId || body.id));
      if (!snapshot) {
        sendJson(res, 404, { ok: false, error: 'CMS snapshot not found' });
        return;
      }
      const current = await getCurrentDocument({
        collection: snapshot.collection,
        id: snapshot.documentId
      });
      sendJson(res, 200, {
        ok: true,
        snapshot: serializeSnapshotRecord(snapshot, true),
        current: serializeContentRecord(current),
        diff: createJsonDiffSummary(snapshot.document, current && current.document)
      });
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS snapshot unavailable' });
    }
    return;
  }

  sendJson(res, 405, { ok: false, error: 'Method not allowed' });
}

async function handleLibrary(req, res) {
  const query = getQuery(req);
  const type = normalizeLibraryType(pickQuery(query.type));

  if (req.method === 'GET') {
    try {
      if (type) {
        const items = listLibraryItems(type).map(serializeLibraryRecord);
        sendJson(res, 200, { ok: true, type, items });
        return;
      }
      const grouped = listAllLibraryItems();
      sendJson(res, 200, {
        ok: true,
        library: {
          templates: grouped.templates.map(serializeLibraryRecord),
          sections: grouped.sections.map(serializeLibraryRecord),
          drafts: grouped.drafts.map(serializeLibraryRecord)
        }
      });
      return;
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS library unavailable' });
      return;
    }
  }

  if (req.method === 'PUT') {
    let body;
    try {
      body = await readJson(req);
    } catch {
      sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
      return;
    }
    try {
      const bodyType = normalizeLibraryType((body && body.type) || type);
      const record = saveLibraryItem(bodyType, body && body.item);
      sendJson(res, 200, { ok: true, item: serializeLibraryRecord(record) });
      return;
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS library unavailable' });
      return;
    }
  }

  if (req.method === 'DELETE') {
    const id = normalizeCmsDocumentId(pickQuery(query.id));
    try {
      if (!type || !id) {
        sendJson(res, 400, { ok: false, error: 'Library type and id are required' });
        return;
      }
      const deleted = deleteLibraryItem(type, id);
      sendJson(res, 200, { ok: true, deleted });
      return;
    } catch (err) {
      if (handleStorageError(res, err)) return;
      sendJson(res, 500, { ok: false, error: 'CMS library unavailable' });
      return;
    }
  }

  sendJson(res, 405, { ok: false, error: 'Method not allowed' });
}

function getWslHostAddress() {
  try {
    const resolv = fs.readFileSync('/etc/resolv.conf', 'utf8');
    const match = resolv.match(/^nameserver\s+([^\s]+)/m);
    return match && isPrivateIpv4Host(match[1]) ? match[1] : '';
  } catch {
    return '';
  }
}

function ipv4FromLittleEndianHex(value) {
  const raw = String(value || '').trim();
  if (!/^[0-9a-f]{8}$/i.test(raw)) return '';
  const ip = raw.match(/../g)
    .map((part) => Number.parseInt(part, 16))
    .reverse()
    .join('.');
  return isPrivateIpv4Host(ip) ? ip : '';
}

function getWslDefaultGatewayAddress() {
  try {
    const routeTable = fs.readFileSync('/proc/net/route', 'utf8');
    const route = routeTable
      .split(/\r?\n/)
      .slice(1)
      .map((line) => line.trim().split(/\s+/))
      .find((fields) => fields[1] === '00000000' && fields[2]);
    return route ? ipv4FromLittleEndianHex(route[2]) : '';
  } catch {
    return '';
  }
}

function getOllamaBaseUrlCandidates() {
  const configured = String(process.env.CMS_OLLAMA_URL || process.env.OLLAMA_BASE_URL || '').trim();
  if (configured) return [configured.replace(/\/+$/, '')];

  const candidates = ['http://127.0.0.1:11434', 'http://localhost:11434'];
  const wslGateway = getWslDefaultGatewayAddress();
  if (wslGateway) candidates.push(`http://${wslGateway}:11434`);
  const wslHost = getWslHostAddress();
  if (wslHost) candidates.push(`http://${wslHost}:11434`);
  candidates.push('http://host.docker.internal:11434');
  return [...new Set(candidates.map((item) => item.replace(/\/+$/, '')))];
}

async function requestOllama(pathname, options = {}) {
  let lastError = null;
  for (const baseUrl of getOllamaBaseUrlCandidates()) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 8_000);
    try {
      const response = await fetch(`${baseUrl}${pathname}`, {
        method: options.method || 'GET',
        headers: options.headers || undefined,
        body: options.body,
        signal: controller.signal
      });
      const text = await response.text();
      return { baseUrl, response, text };
    } catch (err) {
      lastError = err;
    } finally {
      clearTimeout(timeout);
    }
  }
  const err = lastError || new Error('Unable to reach local Ollama');
  err.attemptedBaseUrls = getOllamaBaseUrlCandidates();
  throw err;
}

function normalizeOllamaModelInfo(model) {
  const name = String((model && (model.name || model.model)) || '').trim();
  if (!name) return null;
  return {
    name,
    modifiedAt: model.modified_at || model.modifiedAt || '',
    size: Number(model.size) || 0,
    digest: String(model.digest || ''),
    details: isPlainObject(model.details) ? model.details : {}
  };
}

async function handleOllamaModels(req, res) {
  if (req.method !== 'GET') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }

  try {
    const { baseUrl: resolvedBaseUrl, response, text } = await requestOllama('/api/tags', {
      timeoutMs: 8_000
    });
    if (!response.ok) {
      sendJson(res, response.status, { ok: false, error: `Ollama model list failed (${response.status}): ${text.slice(0, 300)}` });
      return;
    }
    const data = text ? JSON.parse(text) : {};
    const models = (Array.isArray(data.models) ? data.models : [])
      .map(normalizeOllamaModelInfo)
      .filter(Boolean)
      .sort((a, b) => a.name.localeCompare(b.name));
    sendJson(res, 200, {
      ok: true,
      baseUrl: resolvedBaseUrl,
      models
    });
  } catch (err) {
    const message = err && err.name === 'AbortError'
      ? `Ollama model list timed out. Tried: ${getOllamaBaseUrlCandidates().join(', ')}.`
      : `Unable to reach local Ollama. Tried: ${getOllamaBaseUrlCandidates().join(', ')}. Start Ollama, then refresh the model list.`;
    sendJson(res, 502, { ok: false, error: message, models: [], attemptedBaseUrls: getOllamaBaseUrlCandidates() });
  }
}

function extractJsonObject(text) {
  const raw = String(text || '').trim()
    .replace(/^```(?:json)?/i, '')
    .replace(/```$/i, '')
    .trim();
  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start >= 0 && end > start) {
      return JSON.parse(raw.slice(start, end + 1));
    }
    throw new Error('Ollama response was not valid JSON');
  }
}

function createOllamaMessages(prompt, context) {
  return [
    {
      role: 'system',
      content: [
        'You edit a local website through a JSON-based CMS.',
        'Return ONLY valid JSON, with no markdown and no prose outside the JSON object.',
        'Allowed response shape:',
        '{"reply":"short human summary","edits":{"metadata":{},"section":{"id":"","label":"","html":"","enabled":true},"page":{},"navigation":{},"footer":{}}}',
        'Use metadata for page title, description, canonicalPath, ogTitle, ogDescription, robots, twitter fields, themeColor, and similar page head fields.',
        'Use section only when changing the selected page section. Preserve valid HTML.',
        'Use navigation for global header changes and footer for global footer changes. Return only changed fields where practical.'
      ].join('\n')
    },
    {
      role: 'user',
      content: JSON.stringify({
        request: prompt,
        cmsContext: context
      }, null, 2)
    }
  ];
}

async function handleOllama(req, res) {
  if (req.method !== 'POST') {
    sendJson(res, 405, { ok: false, error: 'Method not allowed' });
    return;
  }

  let body;
  try {
    body = await readJson(req);
  } catch {
    sendJson(res, 400, { ok: false, error: 'Invalid JSON body' });
    return;
  }

  const prompt = String(body && body.prompt ? body.prompt : '').trim();
  if (!prompt) {
    sendJson(res, 400, { ok: false, error: 'Ollama prompt is required' });
    return;
  }

  const model = String(body && body.model ? body.model : process.env.CMS_OLLAMA_MODEL || 'llama3.1').trim();

  try {
    const { response, text } = await requestOllama('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: false,
        messages: createOllamaMessages(prompt, body && body.context ? body.context : {})
      }),
      timeoutMs: 90_000
    });
    if (!response.ok) {
      sendJson(res, response.status, { ok: false, error: `Ollama request failed (${response.status}): ${text.slice(0, 300)}` });
      return;
    }
    const data = text ? JSON.parse(text) : {};
    const content = data && data.message && typeof data.message.content === 'string'
      ? data.message.content
      : (typeof data.response === 'string' ? data.response : '');
    let parsed;
    try {
      parsed = extractJsonObject(content);
    } catch {
      sendJson(res, 200, {
        ok: true,
        reply: content || 'Ollama returned a response, but it did not include JSON edits.',
        edits: {}
      });
      return;
    }
    sendJson(res, 200, {
      ok: true,
      reply: parsed.reply || 'Ollama edit ready.',
      edits: isPlainObject(parsed.edits) ? parsed.edits : parsed
    });
  } catch (err) {
    const message = err && err.name === 'AbortError'
      ? 'Ollama request timed out.'
      : `Unable to reach local Ollama. Tried: ${getOllamaBaseUrlCandidates().join(', ')}. Start Ollama and make sure the selected model is installed.`;
    sendJson(res, 502, { ok: false, error: message, attemptedBaseUrls: getOllamaBaseUrlCandidates() });
  }
}

module.exports = async (req, res) => {
  if (!isLocalRequest(req)) {
    sendJson(res, 403, { ok: false, error: 'CMS editing is available only on localhost.' });
    return;
  }

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.end();
    return;
  }

  const endpoint = getEndpointFromRequest(req);
  if (endpoint === 'content') return handleContent(req, res);
  if (endpoint === 'preview') return handlePreview(req, res);
  if (endpoint === 'preview-project') return handleProjectPreview(req, res);
  if (endpoint === 'preview-tool') return handleToolPreview(req, res);
  if (endpoint === 'widgets') return handleWidgets(req, res);
  if (endpoint === 'media') return handleMedia(req, res);
  if (endpoint === 'health') return handleHealth(req, res);
  if (endpoint === 'snapshots') return handleSnapshots(req, res);
  if (endpoint === 'library') return handleLibrary(req, res);
  if (endpoint === 'ollama-models') return handleOllamaModels(req, res);
  if (endpoint === 'ollama') return handleOllama(req, res);

  sendJson(res, 404, { ok: false, error: 'Not Found' });
};
