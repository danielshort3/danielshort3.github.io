#!/usr/bin/env node
'use strict';

/*
  Inject shared head metadata into site HTML files.

  Responsibilities:
  - Normalizes baseline SEO, Open Graph, and Twitter metadata.
  - Injects connected, route-aware JSON-LD for the public site.
  - Keeps authored and generated pages aligned with content/site/settings.json.

  No external deps.
*/

const fs = require('fs');
const path = require('path');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');

const root = path.resolve(__dirname, '..');

function readJsonFile(absPath, fallback = {}) {
  try {
    const parsed = JSON.parse(fs.readFileSync(absPath, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : fallback;
  } catch {
    return fallback;
  }
}

function loadJsonRecords(relDir) {
  const absDir = path.join(root, relDir);
  try {
    return fs.readdirSync(absDir)
      .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
      .sort()
      .map((name) => readJsonFile(path.join(absDir, name), null))
      .filter(Boolean);
  } catch {
    return [];
  }
}

function sortRecords(records) {
  return [...(Array.isArray(records) ? records : [])].sort((a, b) => {
    const orderA = Number.isFinite(Number(a && a.order)) ? Number(a.order) : Number.MAX_SAFE_INTEGER;
    const orderB = Number.isFinite(Number(b && b.order)) ? Number(b.order) : Number.MAX_SAFE_INTEGER;
    if (orderA !== orderB) return orderA - orderB;
    return String(a && (a.title || a.id || a.slug) || '').localeCompare(String(b && (b.title || b.id || b.slug) || ''));
  });
}

const SITE_SETTINGS = Object.freeze(readJsonFile(path.join(root, 'content', 'site', 'settings.json')));
const SITE_ORIGIN = String(SITE_SETTINGS.siteOrigin || 'https://www.danielshort.me').replace(/\/+$/, '');
const OWNER_NAME = String(SITE_SETTINGS.ownerName || SITE_SETTINGS.siteName || 'Daniel Short').trim();
const SITE_NAME = String(SITE_SETTINGS.siteName || OWNER_NAME).trim();
const SITE_LANGUAGE = String(SITE_SETTINGS.language || 'en-US').trim();
const OG_LOCALE = String(SITE_SETTINGS.locale || 'en_US').trim();
const TWITTER_SITE = String(SITE_SETTINGS.twitterSite || '@danielshort3').trim();
const TWITTER_CREATOR = String(SITE_SETTINGS.twitterCreator || TWITTER_SITE).trim();
const PROFILE_IMAGE = String(SITE_SETTINGS.profileImage || `${SITE_ORIGIN}/img/hero/head-avatar-384.jpg`).trim();
const SAME_AS = Object.freeze((Array.isArray(SITE_SETTINGS.sameAs) ? SITE_SETTINGS.sameAs : [])
  .map((value) => String(value || '').trim())
  .filter(Boolean));
const DEFAULT_OG_IMAGE = Object.freeze({
  url: String(SITE_SETTINGS.ogImage && SITE_SETTINGS.ogImage.url || `${SITE_ORIGIN}/img/brand/07-website-hero-light-version.png`).trim(),
  width: String(SITE_SETTINGS.ogImage && SITE_SETTINGS.ogImage.width || '1672').trim(),
  height: String(SITE_SETTINGS.ogImage && SITE_SETTINGS.ogImage.height || '941').trim(),
  type: String(SITE_SETTINGS.ogImage && SITE_SETTINGS.ogImage.type || 'image/png').trim(),
  alt: String(SITE_SETTINGS.ogImage && SITE_SETTINGS.ogImage.alt || 'Daniel Short portfolio preview').trim()
});
const LEGACY_SHARED_OG_IMAGES = new Set([
  `${SITE_ORIGIN}/img/hero/head.png`,
  `${SITE_ORIGIN}/img/brand/10-github-readme-portfolio-banner.svg`
]);
const GAME_PAGE = readJsonFile(path.join(root, 'content', 'pages', 'games.json'));
const GAME_RECORDS = Object.freeze(sortRecords(Array.isArray(GAME_PAGE.games) ? GAME_PAGE.games : [])
  .filter((game) => game && !game.hidden && !game.noindex && (game.href || game.id)));
const GAME_BY_PATH = new Map(GAME_RECORDS.map((game) => [
  normalizePathname(game.href || `/games/${game.id}`),
  game
]));
const PROJECT_RECORDS = Object.freeze(sortRecords(loadJsonRecords(path.join('content', 'projects')))
  .filter((project) => project && project.id && project.published !== false && !project.hidden && !project.noindex));
const TOOL_RECORDS = Object.freeze(sortRecords(loadJsonRecords(path.join('content', 'tools')))
  .filter((tool) => {
    const visibility = String(tool && tool.visibility || 'public').trim().toLowerCase();
    return tool && tool.slug && !tool.hidden && !tool.noindex && visibility === 'public';
  }));
const noindexPathnames = loadNoindexPathnamesFromVercel(root);
const CSS_MANIFEST_PATH = path.join(root, 'dist', 'styles-manifest.json');
const CSS_MANIFEST = Object.freeze(loadCssManifest());
const BASE_STYLESHEET_FALLBACK = 'dist/styles.css';
const TOOLS_STYLESHEET_FALLBACK = 'dist/styles-tools.css';
const BASE_STYLESHEET_HREF = resolveManagedStylesheetHref(BASE_STYLESHEET_FALLBACK, CSS_MANIFEST.file);
const TOOLS_STYLESHEET_HREF = resolveManagedStylesheetHref(TOOLS_STYLESHEET_FALLBACK, CSS_MANIFEST.toolsFile);

const ROUTE_COMPONENT_STYLES_PATH = path.join(root, 'build', 'route-component-styles.json');
const ROUTE_COMPONENT_STYLES = Object.freeze(loadRouteComponentStyles());

function loadCssManifest() {
  let raw;
  try {
    raw = fs.readFileSync(CSS_MANIFEST_PATH, 'utf8');
  } catch {
    return {};
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {};
  }

  return parsed;
}

function resolveManagedStylesheetHref(fallbackHref, manifestFile) {
  const relPath = String(manifestFile || '').trim();
  if (!relPath) return fallbackHref;
  return `dist/${relPath.replace(/^dist\//i, '')}`;
}

function stylesheetCandidates(...hrefs) {
  const seen = new Set();
  return hrefs.filter((href) => {
    const value = String(href || '').trim();
    if (!value || seen.has(value)) return false;
    seen.add(value);
    return true;
  });
}

function loadRouteComponentStyles() {
  let raw;
  try {
    raw = fs.readFileSync(ROUTE_COMPONENT_STYLES_PATH, 'utf8');
  } catch {
    return {};
  }

  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {};
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {};
  }

  const normalized = {};
  Object.entries(parsed).forEach(([pathname, hrefs]) => {
    if (typeof pathname !== 'string' || !pathname.startsWith('/')) return;
    if (pathname.endsWith('*') && pathname.length < 2) return;
    if (!Array.isArray(hrefs)) return;
    const validHrefs = hrefs
      .map((href) => String(href || '').trim())
      .filter(Boolean);
    if (!validHrefs.length) return;
    normalized[pathname] = validHrefs;
  });

  return normalized;
}

function getRouteComponentStyles(pathname) {
  const rawPath = String(pathname || '').trim();
  if (!rawPath) return [];

  const matches = [];
  Object.entries(ROUTE_COMPONENT_STYLES).forEach(([pattern, hrefs]) => {
    if (!Array.isArray(hrefs) || !hrefs.length) return;
    if (pattern === rawPath) {
      matches.push(...hrefs);
      return;
    }
    if (!pattern.endsWith('*')) return;
    const prefix = pattern.slice(0, -1);
    if (!prefix || !rawPath.startsWith(prefix)) return;
    matches.push(...hrefs);
  });

  return [...new Set(matches)];
}
function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function write(relPath, contents) {
  fs.writeFileSync(path.join(root, relPath), contents, 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function walkHtmlFiles(dirRelPath) {
  const start = path.join(root, dirRelPath);
  if (!fs.existsSync(start)) return [];
  const htmlFiles = [];
  const stack = [start];
  while (stack.length) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    entries.forEach((entry) => {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        return;
      }
      if (entry.isFile() && entry.name.endsWith('.html')) {
        htmlFiles.push(full);
      }
    });
  }
  return htmlFiles.sort();
}

function relFromRoot(absPath) {
  return path.relative(root, absPath).replace(/\\/g, '/');
}

function listRootHtmlFiles() {
  let entries;
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.html'))
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function sliceHead(html) {
  if (!html) return null;
  const openMatch = /<head\b[^>]*>/i.exec(html);
  if (!openMatch) return null;
  const openIndex = openMatch.index;
  const openEnd = openIndex + openMatch[0].length;
  const closeIndex = html.search(/<\/head>/i);
  if (closeIndex < 0 || closeIndex < openEnd) return null;
  return {
    openIndex,
    openEnd,
    closeIndex,
    inner: html.slice(openEnd, closeIndex)
  };
}

function hasTag(headInner, re) {
  re.lastIndex = 0;
  return re.test(headInner);
}

function getMetaContent(headInner, { name, property }) {
  const attr = name ? `name="${name}"` : `property="${property}"`;
  const re = new RegExp(`<meta\\s+[^>]*${attr}[^>]*content="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(headInner);
  return match ? String(match[1] || '').trim() : '';
}

function getCanonicalHref(headInner) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(headInner);
  return match ? String(match[1] || '').trim() : '';
}

function getTitleText(headInner) {
  const match = /<title>([^<]+)<\/title>/i.exec(headInner);
  return match ? String(match[1] || '').trim() : '';
}

function escapeHtmlAttribute(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function upsertMetaTag(headInner, attr, key, content) {
  const safeAttr = String(attr || '').trim();
  const safeKey = String(key || '').trim();
  const safeContent = String(content || '').trim();
  if (!safeAttr || !safeKey || !safeContent) return headInner;
  const matcher = new RegExp(`<meta\\b[^>]*\\b${escapeRegExp(safeAttr)}=["']${escapeRegExp(safeKey)}["'][^>]*>`, 'i');
  const tag = `<meta ${safeAttr}="${escapeHtmlAttribute(safeKey)}" content="${escapeHtmlAttribute(safeContent)}">`;
  if (matcher.test(headInner)) return headInner.replace(matcher, tag);
  return `${String(headInner || '').trimEnd()}\n  ${tag}\n`;
}

function inferImageMime(imageUrl) {
  const pathname = String(imageUrl || '').split(/[?#]/)[0].toLowerCase();
  if (pathname.endsWith('.png')) return 'image/png';
  if (pathname.endsWith('.jpg') || pathname.endsWith('.jpeg')) return 'image/jpeg';
  if (pathname.endsWith('.webp')) return 'image/webp';
  if (pathname.endsWith('.gif')) return 'image/gif';
  return '';
}

function toAbsoluteSiteUrl(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    return new URL(raw, `${SITE_ORIGIN}/`).href;
  } catch {
    return '';
  }
}

function getHeadPathname(headInner) {
  return toPathname(getCanonicalHref(headInner));
}

function replaceLegacySharedOgImage(headInner) {
  const currentImage = getMetaContent(headInner, { property: 'og:image' });
  if (!LEGACY_SHARED_OG_IMAGES.has(currentImage)) return headInner;

  let next = String(headInner || '');
  LEGACY_SHARED_OG_IMAGES.forEach((legacyUrl) => {
    next = next.split(legacyUrl).join(DEFAULT_OG_IMAGE.url);
  });
  next = upsertMetaTag(next, 'property', 'og:image:width', DEFAULT_OG_IMAGE.width);
  next = upsertMetaTag(next, 'property', 'og:image:height', DEFAULT_OG_IMAGE.height);
  next = upsertMetaTag(next, 'property', 'og:image:type', DEFAULT_OG_IMAGE.type);
  next = upsertMetaTag(next, 'property', 'og:image:alt', DEFAULT_OG_IMAGE.alt);
  next = upsertMetaTag(next, 'name', 'twitter:image', DEFAULT_OG_IMAGE.url);
  next = upsertMetaTag(next, 'name', 'twitter:image:alt', DEFAULT_OG_IMAGE.alt);
  return next;
}

function ensureGameSocialMetadata(headInner) {
  const pathname = getHeadPathname(headInner);
  const game = GAME_BY_PATH.get(pathname);
  if (!game) return headInner;

  const title = String(game.title || getTitleText(headInner) || 'Browser game').trim();
  const pageTitle = getTitleText(headInner) || `${title} | ${OWNER_NAME}`;
  const description = String(game.summary || '').trim();
  const canonical = getCanonicalHref(headInner);
  const image = toAbsoluteSiteUrl(game.image) || DEFAULT_OG_IMAGE.url;
  const imageWidth = String(game.imageWidth || (image === DEFAULT_OG_IMAGE.url ? DEFAULT_OG_IMAGE.width : '')).trim();
  const imageHeight = String(game.imageHeight || (image === DEFAULT_OG_IMAGE.url ? DEFAULT_OG_IMAGE.height : '')).trim();
  const imageAlt = String(game.imageAlt || `${title} browser game preview`).trim();

  let next = headInner;
  if (description) next = upsertMetaTag(next, 'name', 'description', description);
  next = upsertMetaTag(next, 'property', 'og:title', pageTitle);
  next = upsertMetaTag(next, 'property', 'og:site_name', SITE_NAME);
  if (description) next = upsertMetaTag(next, 'property', 'og:description', description);
  if (canonical) next = upsertMetaTag(next, 'property', 'og:url', canonical);
  next = upsertMetaTag(next, 'property', 'og:image', image);
  if (imageWidth) next = upsertMetaTag(next, 'property', 'og:image:width', imageWidth);
  if (imageHeight) next = upsertMetaTag(next, 'property', 'og:image:height', imageHeight);
  next = upsertMetaTag(next, 'property', 'og:image:type', inferImageMime(image) || DEFAULT_OG_IMAGE.type);
  next = upsertMetaTag(next, 'property', 'og:image:alt', imageAlt);
  next = upsertMetaTag(next, 'property', 'og:type', 'website');
  next = upsertMetaTag(next, 'name', 'twitter:card', 'summary_large_image');
  next = upsertMetaTag(next, 'name', 'twitter:site', TWITTER_SITE);
  return next;
}

function ensureBaselineMetadata(headInner) {
  const pathname = getHeadPathname(headInner);
  const explicitNoindex = hasNoindexRobotsMeta(headInner);
  const routeNoindex = pathname && noindexPathnames.has(pathname);
  let next = headInner;

  next = upsertMetaTag(next, 'name', 'author', OWNER_NAME);
  if (pathname && !explicitNoindex && !routeNoindex) {
    next = upsertMetaTag(next, 'name', 'robots', 'index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1');
  }
  if (getMetaContent(next, { property: 'og:title' })) {
    next = upsertMetaTag(next, 'property', 'og:site_name', SITE_NAME);
    next = upsertMetaTag(next, 'property', 'og:locale', OG_LOCALE);
  }
  const image = getMetaContent(next, { property: 'og:image' });
  const imageType = inferImageMime(image);
  if (imageType) next = upsertMetaTag(next, 'property', 'og:image:type', imageType);
  if (getMetaContent(next, { name: 'twitter:title' }) || getMetaContent(next, { property: 'og:title' })) {
    next = upsertMetaTag(next, 'name', 'twitter:site', TWITTER_SITE);
    next = upsertMetaTag(next, 'name', 'twitter:creator', TWITTER_CREATOR);
  }
  return next;
}

function findLineInsertionPoint(headInner) {
  const candidates = [
    /(^([ \t]*)<meta\s+name="twitter:site"[^>]*>\s*$)/mi,
    /(^([ \t]*)<meta\s+name="twitter:card"[^>]*>\s*$)/mi,
    /(^([ \t]*)<meta\s+property="og:type"[^>]*>\s*$)/mi,
    /(^([ \t]*)<meta\s+property="og:image:alt"[^>]*>\s*$)/mi,
    /(^([ \t]*)<meta\s+property="og:image"[^>]*>\s*$)/mi,
    /(^([ \t]*)<meta\s+property="og:url"[^>]*>\s*$)/mi
  ];
  for (const re of candidates) {
    re.lastIndex = 0;
    const match = re.exec(headInner);
    if (!match) continue;
    return { index: match.index + match[1].length, indent: match[2] || '' };
  }
  return { index: headInner.length, indent: '  ' };
}

function injectAfterMetaLine(headInner, metaProperty, lines) {
  const re = new RegExp(`(^([ \\t]*)<meta\\s+property="${metaProperty.replace(/[-/\\^$*+?.()|[\\]{}]/g, '\\\\$&')}"[^>]*>\\s*$)`, 'mi');
  const match = re.exec(headInner);
  if (!match) return headInner;
  const indent = match[2] || '';
  const insertion = lines.map((line) => `${indent}${line}`).join('\n');
  return headInner.slice(0, match.index + match[1].length) + '\n' + insertion + headInner.slice(match.index + match[1].length);
}

function ensureTwitterMeta(headInner) {
  const ogTitle = getMetaContent(headInner, { property: 'og:title' });
  const ogDesc = getMetaContent(headInner, { property: 'og:description' }) || getMetaContent(headInner, { name: 'description' });
  const ogImage = getMetaContent(headInner, { property: 'og:image' });
  const ogAlt = getMetaContent(headInner, { property: 'og:image:alt' });

  const hasTwitterTitle = hasTag(headInner, /<meta\b[^>]*\bname="twitter:title"/i);
  const hasTwitterDesc = hasTag(headInner, /<meta\b[^>]*\bname="twitter:description"/i);
  const hasTwitterImage = hasTag(headInner, /<meta\b[^>]*\bname="twitter:image"/i);
  const hasTwitterAlt = hasTag(headInner, /<meta\b[^>]*\bname="twitter:image:alt"/i);

  const lines = [];
  if (!hasTwitterTitle && ogTitle) lines.push(`<meta name="twitter:title" content="${ogTitle}">`);
  if (!hasTwitterDesc && ogDesc) lines.push(`<meta name="twitter:description" content="${ogDesc}">`);
  if (!hasTwitterImage && ogImage) lines.push(`<meta name="twitter:image" content="${ogImage}">`);
  if (!hasTwitterAlt && (ogAlt || (ogImage === DEFAULT_OG_IMAGE.url ? DEFAULT_OG_IMAGE.alt : ''))) {
    const alt = ogAlt || (ogImage === DEFAULT_OG_IMAGE.url ? DEFAULT_OG_IMAGE.alt : '');
    if (alt) lines.push(`<meta name="twitter:image:alt" content="${alt}">`);
  }
  if (!lines.length) return headInner;

  const { index, indent } = findLineInsertionPoint(headInner);
  const block = lines.map((line) => `${indent}${line}`).join('\n');
  return headInner.slice(0, index) + '\n' + block + headInner.slice(index);
}

function ensureSharedOgImageDimensions(headInner) {
  const ogImage = getMetaContent(headInner, { property: 'og:image' });
  if (ogImage !== DEFAULT_OG_IMAGE.url) return headInner;

  const hasWidth = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:width"/i);
  const hasHeight = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:height"/i);
  const hasAlt = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:alt"/i);

  if (hasWidth && hasHeight && hasAlt) return headInner;

  const lines = [];
  if (!hasWidth) lines.push(`<meta property="og:image:width" content="${DEFAULT_OG_IMAGE.width}">`);
  if (!hasHeight) lines.push(`<meta property="og:image:height" content="${DEFAULT_OG_IMAGE.height}">`);
  if (!hasAlt) lines.push(`<meta property="og:image:alt" content="${DEFAULT_OG_IMAGE.alt}">`);

  if (!lines.length) return headInner;
  return injectAfterMetaLine(headInner, 'og:image', lines);
}

function toPathname(urlValue) {
  const raw = String(urlValue || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizePathname(url.pathname || '/');
  } catch {
    return '';
  }
}

function needsToolsStyles(pathname) {
  return pathname === '/tools'
    || pathname.startsWith('/tools/')
    || pathname === '/short-links'
    || pathname === '/games/ocean-wave-simulation';
}

function ensureStylesheetLink(headInner, href, preferredAfterHrefs = []) {
  const safeHref = String(href || '').trim();
  if (!safeHref) return headInner;
  const escapedHref = safeHref.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  if (hasTag(headInner, new RegExp(`<link\\b[^>]*\\bhref="${escapedHref}"[^>]*>`, 'i'))) return headInner;

  for (const afterHref of preferredAfterHrefs) {
    const safeAfter = String(afterHref || '').trim();
    if (!safeAfter) continue;
    const escapedAfter = safeAfter.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    const re = new RegExp(`(^([ \\t]*)<link\\s+[^>]*href="${escapedAfter}"[^>]*>\\s*$)`, 'mi');
    const match = re.exec(headInner);
    if (!match) continue;
    const indent = match[2] || '  ';
    const insert = `\n${indent}<link rel="stylesheet" href="${safeHref}">`;
    return headInner.slice(0, match.index + match[1].length)
      + insert
      + headInner.slice(match.index + match[1].length);
  }

  return `${headInner.trimEnd()}\n  <link rel="stylesheet" href="${safeHref}">\n`;
}

function replaceManagedStylesheetLinks(headInner) {
  return String(headInner || '')
    .replace(/href="dist\/styles(?:\.[0-9a-f]{8})?\.css"/gi, `href="${BASE_STYLESHEET_HREF}"`)
    .replace(/href="dist\/styles-tools(?:\.[0-9a-f]{8})?\.css"/gi, `href="${TOOLS_STYLESHEET_HREF}"`);
}

function ensureToolsStylesheet(headInner) {
  const canonical = getCanonicalHref(headInner);
  const pathname = toPathname(canonical);
  if (!needsToolsStyles(pathname)) return headInner;
  return ensureStylesheetLink(
    headInner,
    TOOLS_STYLESHEET_HREF,
    stylesheetCandidates(BASE_STYLESHEET_HREF, BASE_STYLESHEET_FALLBACK)
  );
}

function ensureRouteComponentStylesheet(headInner) {
  const canonical = getCanonicalHref(headInner);
  const pathname = toPathname(canonical);
  const hrefs = getRouteComponentStyles(pathname);
  if (!Array.isArray(hrefs) || !hrefs.length) return headInner;

  return hrefs.reduce(
    (nextHead, href) => ensureStylesheetLink(
      nextHead,
      href,
      stylesheetCandidates(
        TOOLS_STYLESHEET_HREF,
        TOOLS_STYLESHEET_FALLBACK,
        BASE_STYLESHEET_HREF,
        BASE_STYLESHEET_FALLBACK
      )
    ),
    headInner
  );
}

function dedupeMeta(headInner, attr, value) {
  const escapedValue = String(value || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const escapedAttr = String(attr || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const re = new RegExp(`<meta\\b[^>]*\\b${escapedAttr}="${escapedValue}"`, 'i');
  const lines = String(headInner || '').split('\n');
  let seen = false;
  const out = [];
  lines.forEach((line) => {
    if (!re.test(line)) {
      out.push(line);
      return;
    }
    if (seen) return;
    seen = true;
    out.push(line);
  });
  return out.join('\n');
}

function escapeTitleSuffix(title) {
  return String(title || '').replace(/\s*\|\s*Daniel Short\s*$/i, '').trim();
}

function stripToolJsonLd(headInner) {
  return String(headInner || '').replace(/\n?[ \t]*<script\b[^>]*\bid="tool-jsonld"[^>]*>[\s\S]*?<\/script>/i, '');
}

function hasNoindexRobotsMeta(headInner) {
  return /<meta\b[^>]*\bname="robots"[^>]*\bcontent="[^"]*noindex[^"]*"[^>]*>/i.test(String(headInner || ''));
}

function decodeHtmlText(value) {
  return String(value || '')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&amp;/gi, '&');
}

function stripSiteJsonLd(headInner) {
  return String(headInner || '').replace(/\n?[ \t]*<script\b[^>]*\bid="site-jsonld"[^>]*>[\s\S]*?<\/script>/i, '');
}

function collectionItems(pathname) {
  let records = [];
  if (pathname === '/portfolio') {
    records = PROJECT_RECORDS.map((project) => ({
      name: project.title || project.id,
      url: `${SITE_ORIGIN}/portfolio/${encodeURIComponent(project.id)}`
    }));
  } else if (pathname === '/tools') {
    records = TOOL_RECORDS.map((tool) => ({
      name: tool.title || tool.slug,
      url: toAbsoluteSiteUrl(tool.href || `/tools/${tool.slug}`)
    }));
  } else if (pathname === '/games') {
    records = GAME_RECORDS.map((game) => ({
      name: game.title || game.id,
      url: toAbsoluteSiteUrl(game.href || `/games/${game.id}`)
    }));
  }

  return records
    .filter((item) => item.name && item.url)
    .map((item, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: String(item.name),
      url: item.url
    }));
}

function breadcrumbNode(pathname, canonicalUrl, pageName) {
  if (!pathname || pathname === '/') return null;
  const segmentLabels = {
    portfolio: 'Portfolio',
    tools: 'Tools',
    games: 'Games'
  };
  const segments = pathname.split('/').filter(Boolean);
  const items = [
    { '@type': 'ListItem', position: 1, name: 'Home', item: `${SITE_ORIGIN}/` }
  ];
  let route = '';
  segments.forEach((segment, index) => {
    route += `/${segment}`;
    const isLast = index === segments.length - 1;
    items.push({
      '@type': 'ListItem',
      position: index + 2,
      name: isLast ? pageName : (segmentLabels[segment] || segment.replace(/-/g, ' ')),
      item: isLast ? canonicalUrl : `${SITE_ORIGIN}${route}`
    });
  });
  return {
    '@type': 'BreadcrumbList',
    '@id': `${canonicalUrl}#breadcrumb`,
    itemListElement: items
  };
}

function ensureSiteJsonLd(headInner) {
  const withoutExisting = stripSiteJsonLd(headInner);
  if (hasNoindexRobotsMeta(withoutExisting)) return withoutExisting;

  const canonicalUrl = getCanonicalHref(withoutExisting);
  const pathname = toPathname(canonicalUrl);
  if (!canonicalUrl || !pathname || noindexPathnames.has(pathname)) return withoutExisting;

  const title = decodeHtmlText(getTitleText(withoutExisting));
  const name = escapeTitleSuffix(title) || SITE_NAME;
  const description = decodeHtmlText(
    getMetaContent(withoutExisting, { name: 'description' })
    || getMetaContent(withoutExisting, { property: 'og:description' })
  );
  if (!title || !description) return withoutExisting;

  const image = getMetaContent(withoutExisting, { property: 'og:image' }) || DEFAULT_OG_IMAGE.url;
  const websiteId = `${SITE_ORIGIN}/#website`;
  const personId = `${SITE_ORIGIN}/#person`;
  const webpageId = `${canonicalUrl}#webpage`;
  const person = {
    '@type': 'Person',
    '@id': personId,
    name: OWNER_NAME,
    url: `${SITE_ORIGIN}/`,
    image: PROFILE_IMAGE,
    ...(SAME_AS.length ? { sameAs: SAME_AS } : {})
  };
  const website = {
    '@type': 'WebSite',
    '@id': websiteId,
    url: `${SITE_ORIGIN}/`,
    name: SITE_NAME,
    publisher: { '@id': personId },
    inLanguage: SITE_LANGUAGE
  };
  const pageType = pathname === '/'
    ? 'ProfilePage'
    : (['/portfolio', '/tools', '/games'].includes(pathname)
      ? 'CollectionPage'
      : (pathname === '/contact' ? 'ContactPage' : 'WebPage'));
  const page = {
    '@type': pageType,
    '@id': webpageId,
    url: canonicalUrl,
    name: title,
    description,
    isPartOf: { '@id': websiteId },
    about: { '@id': personId },
    inLanguage: SITE_LANGUAGE,
    primaryImageOfPage: {
      '@type': 'ImageObject',
      url: image
    }
  };
  const graph = [website, person, page];

  if (pathname === '/') {
    page.mainEntity = { '@id': personId };
  }

  const items = collectionItems(pathname);
  if (items.length) {
    const itemListId = `${canonicalUrl}#itemlist`;
    page.mainEntity = { '@id': itemListId };
    graph.push({
      '@type': 'ItemList',
      '@id': itemListId,
      numberOfItems: items.length,
      itemListElement: items
    });
  }

  const game = GAME_BY_PATH.get(pathname);
  if (game) {
    const gameId = `${canonicalUrl}#game`;
    const gameImage = toAbsoluteSiteUrl(game.image) || image;
    page.mainEntity = { '@id': gameId };
    graph.push({
      '@type': ['VideoGame', 'WebApplication'],
      '@id': gameId,
      name: String(game.title || name),
      description: String(game.summary || description),
      url: canonicalUrl,
      image: gameImage,
      creator: { '@id': personId },
      applicationCategory: 'GameApplication',
      operatingSystem: 'Any',
      gamePlatform: 'Web browser',
      playMode: 'SinglePlayer',
      isAccessibleForFree: true,
      ...(Array.isArray(game.tags) && game.tags.length ? { genre: game.tags } : {})
    });
  } else if (pathname.startsWith('/tools/') && pathname !== '/tools') {
    page.mainEntity = { '@id': `${canonicalUrl}#app` };
  } else if (pathname.startsWith('/portfolio/') && pathname !== '/portfolio') {
    page.mainEntity = { '@id': `${canonicalUrl}#project` };
  }

  const breadcrumb = breadcrumbNode(pathname, canonicalUrl, name);
  if (breadcrumb) {
    page.breadcrumb = { '@id': breadcrumb['@id'] };
    graph.push(breadcrumb);
  }

  const serialized = JSON.stringify({ '@context': 'https://schema.org', '@graph': graph }).replace(/</g, '\\u003c');
  const block = [
    '  <script type="application/ld+json" id="site-jsonld">',
    `    ${serialized}`,
    '  </script>'
  ].join('\n');
  return `${withoutExisting.trimEnd()}\n${block}\n`;
}

function ensureToolJsonLd(headInner) {
  const withoutExisting = stripToolJsonLd(headInner);
  if (hasNoindexRobotsMeta(withoutExisting)) return withoutExisting;

  const canonical = getCanonicalHref(withoutExisting);
  if (!canonical || !canonical.startsWith(`${SITE_ORIGIN}/tools/`)) return withoutExisting;
  if (canonical === `${SITE_ORIGIN}/tools`) return withoutExisting;
  const pathname = toPathname(canonical);
  if (pathname && noindexPathnames.has(pathname)) return withoutExisting;

  const title = decodeHtmlText(escapeTitleSuffix(getMetaContent(withoutExisting, { property: 'og:title' }) || getTitleText(withoutExisting)));
  const description = decodeHtmlText(getMetaContent(withoutExisting, { name: 'description' }) || getMetaContent(withoutExisting, { property: 'og:description' }));
  if (!title || !description) return withoutExisting;

  const image = getMetaContent(withoutExisting, { property: 'og:image' }) || DEFAULT_OG_IMAGE.url;

  const json = {
    '@context': 'https://schema.org',
    '@type': 'WebApplication',
    '@id': `${canonical}#app`,
    name: title,
    description,
    url: canonical,
    image,
    applicationCategory: 'UtilitiesApplication',
    operatingSystem: 'Any',
    isAccessibleForFree: true,
    creator: {
      '@type': 'Person',
      '@id': `${SITE_ORIGIN}/#person`,
      name: OWNER_NAME,
      url: `${SITE_ORIGIN}/`
    }
  };

  const serialized = JSON.stringify(json);

  const indent = '  ';
  const block = [
    `${indent}<script type="application/ld+json" id="tool-jsonld">`,
    `${indent}  ${serialized}`,
    `${indent}</script>`
  ].join('\n');

  return withoutExisting.trimEnd() + '\n' + block + '\n';
}

function processHtml(html) {
  const head = sliceHead(html);
  if (!head) return { html, changed: false };

  let inner = head.inner;
  inner = replaceManagedStylesheetLinks(inner);
  inner = dedupeMeta(inner, 'property', 'og:image:width');
  inner = dedupeMeta(inner, 'property', 'og:image:height');
  inner = dedupeMeta(inner, 'property', 'og:image:type');
  inner = dedupeMeta(inner, 'property', 'og:image:alt');
  inner = dedupeMeta(inner, 'property', 'og:site_name');
  inner = dedupeMeta(inner, 'property', 'og:locale');
  inner = dedupeMeta(inner, 'name', 'author');
  inner = dedupeMeta(inner, 'name', 'robots');
  inner = dedupeMeta(inner, 'name', 'twitter:site');
  inner = dedupeMeta(inner, 'name', 'twitter:creator');
  inner = dedupeMeta(inner, 'name', 'twitter:title');
  inner = dedupeMeta(inner, 'name', 'twitter:description');
  inner = dedupeMeta(inner, 'name', 'twitter:image');
  inner = dedupeMeta(inner, 'name', 'twitter:image:alt');
  inner = replaceLegacySharedOgImage(inner);
  inner = ensureGameSocialMetadata(inner);
  inner = ensureSharedOgImageDimensions(inner);
  inner = ensureBaselineMetadata(inner);
  inner = ensureTwitterMeta(inner);
  inner = ensureToolsStylesheet(inner);
  inner = ensureRouteComponentStylesheet(inner);
  inner = ensureToolJsonLd(inner);
  inner = ensureSiteJsonLd(inner);
  inner = dedupeMeta(inner, 'property', 'og:image:width');
  inner = dedupeMeta(inner, 'property', 'og:image:height');
  inner = dedupeMeta(inner, 'property', 'og:image:type');
  inner = dedupeMeta(inner, 'property', 'og:image:alt');
  inner = dedupeMeta(inner, 'property', 'og:site_name');
  inner = dedupeMeta(inner, 'property', 'og:locale');
  inner = dedupeMeta(inner, 'name', 'author');
  inner = dedupeMeta(inner, 'name', 'robots');
  inner = dedupeMeta(inner, 'name', 'twitter:site');
  inner = dedupeMeta(inner, 'name', 'twitter:creator');
  inner = dedupeMeta(inner, 'name', 'twitter:title');
  inner = dedupeMeta(inner, 'name', 'twitter:description');
  inner = dedupeMeta(inner, 'name', 'twitter:image');
  inner = dedupeMeta(inner, 'name', 'twitter:image:alt');

  const changed = inner !== head.inner;
  if (!changed) return { html, changed: false };
  const next = html.slice(0, head.openEnd) + inner + html.slice(head.closeIndex);
  return { html: next, changed: next !== html };
}

function main() {
  const rootHtmlFiles = listRootHtmlFiles();
  const pagesHtmlFiles = walkHtmlFiles('pages');
  const targets = [...rootHtmlFiles, ...pagesHtmlFiles];

  let updated = 0;
  let skipped = 0;

  targets.forEach((absPath) => {
    const relPath = relFromRoot(absPath);
    if (relPath === 'public' || relPath.startsWith('public/')) return;
    if (relPath === 'node_modules' || relPath.startsWith('node_modules/')) return;
    if (!exists(relPath)) return;

    const html = read(relPath);
    const processed = processHtml(html);
    if (!processed.changed) {
      skipped += 1;
      return;
    }
    write(relPath, processed.html);
    updated += 1;
  });

  process.stdout.write(`[inject-head-metadata] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
