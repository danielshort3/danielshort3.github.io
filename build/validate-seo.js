#!/usr/bin/env node
'use strict';

/*
  Validate the source-of-truth SEO contracts for the static site.

  The site keeps a few root HTML copies in sync with pages/*.html. Those files
  are intentionally evaluated as one document after grouping by canonical URL.
  This script has no runtime dependencies so it can run in CI before a build.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const SITE_ORIGIN = 'https://www.danielshort.me';
const EXPECTED_SITEMAP_URL_COUNT = 38;
const EXPECTED_BULK_REDIRECTS_PATH = 'seo-bulk-redirects.json';
const PROFESSIONAL_AUDIENCE_KEYS = ['analytics', 'data-science', 'tourism'];
const GOOGLEBOT_USER_AGENT = 'Googlebot';
const DOCUMENT_CRAWL_SAMPLE = '/documents/Resume.pdf';
const RASTER_IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const NON_SITEMAP_UTILITY_SOURCES = new Set([
  '404.html',
  'dshort.html',
  'pages/contributions.html',
  'pages/search.html',
  'pages/sitemap-pretty.html'
]);

const DIRECTORY_RESULT_CONTRACTS = [
  { file: 'pages/portfolio.html', pathPrefix: '/portfolio/', label: 'Portfolio' },
  { file: 'pages/tools.html', pathPrefix: '/tools/', label: 'Tools' },
  { file: 'pages/games.html', pathPrefix: '/games/', label: 'Games' }
];

const REQUIRED_ALIAS_REDIRECTS = [
  ['/index.html', '/'],
  ['/portfolio/:project.html', '/portfolio/:project'],
  ['/tools/:tool.html', '/tools/:tool'],
  ['/games/:game.html', '/games/:game'],
  ['/:slug.html', '/:slug'],
  ['/word-frequency', '/tools/word-frequency'],
  ['/word-frequency.html', '/tools/word-frequency'],
  ['/oxford-comma-checker', '/tools/oxford-comma-checker'],
  ['/oxford-comma-checker.html', '/tools/oxford-comma-checker'],
  ['/nbsp-cleaner', '/tools/nbsp-cleaner'],
  ['/nbsp-cleaner.html', '/tools/nbsp-cleaner'],
  ['/project-starfall', '/games/project-starfall'],
  ['/project-starfall.html', '/games/project-starfall'],
  ['/tools/ocean-wave-simulation', '/games/ocean-wave-simulation'],
  ['/tools/ocean-wave-simulation.html', '/games/ocean-wave-simulation'],
  ['/projects', '/portfolio'],
  ['/projects/:project', '/portfolio/:project'],
  ['/projects/:project.html', '/portfolio/:project']
];

const ONE_HOP_REDIRECT_CASES = [
  ['/index.html', '/'],
  ['/portfolio/retailStore.html', '/portfolio/retailStore'],
  ['/tools/text-compare.html', '/tools/text-compare'],
  ['/games/project-starfall.html', '/games/project-starfall'],
  ['/contact.html', '/contact'],
  ['/tools/whisper-transcribe-monitor.html', '/tools/transcribe'],
  ['/demos/stellar-dogfight-demo.html', '/games/stellar-dogfight'],
  ['/demos/roulette-double-zero-demo.html', '/games/roulette'],
  ['/probability-engine.html', '/games/probability-engine'],
  ['/word-frequency.html', '/tools/word-frequency'],
  ['/oxford-comma-checker.html', '/tools/oxford-comma-checker'],
  ['/nbsp-cleaner.html', '/tools/nbsp-cleaner'],
  ['/project-starfall.html', '/games/project-starfall'],
  ['/tools/ocean-wave-simulation.html', '/games/ocean-wave-simulation'],
  ['/projects/retailStore.html', '/portfolio/retailStore'],
  ['/pages/portfolio', '/portfolio'],
  ['/pages/portfolio.html', '/portfolio'],
  ['/pages/background-remover', '/tools/background-remover'],
  ['/pages/background-remover.html', '/tools/background-remover'],
  ['/pages/word-frequency', '/tools/word-frequency'],
  ['/pages/word-frequency.html', '/tools/word-frequency'],
  ['/destination-analytics.html', '/tourism'],
  ['/pages/destination-analytics', '/tourism'],
  ['/pages/destination-analytics.html', '/tourism'],
  ['/contributions.html', '/tourism'],
  ['/pages/contributions', '/tourism'],
  ['/pages/contributions.html', '/tourism']
];

const errors = [];

function report(scope, message) {
  errors.push(`${scope}: ${message}`);
}

function toPosix(filePath) {
  return String(filePath || '').replace(/\\/g, '/');
}

function readRequired(relativePath) {
  const absolutePath = path.join(root, relativePath);
  try {
    return fs.readFileSync(absolutePath, 'utf8');
  } catch (error) {
    report(relativePath, `could not be read (${error.code || error.message})`);
    return '';
  }
}

function decodeEntities(value) {
  const named = {
    amp: '&',
    apos: "'",
    gt: '>',
    hellip: '\u2026',
    lt: '<',
    nbsp: ' ',
    quot: '"'
  };
  return String(value || '').replace(/&(#x[0-9a-f]+|#\d+|[a-z][a-z0-9]+);/gi, (match, entity) => {
    if (entity.charAt(0) === '#') {
      const isHex = entity.charAt(1).toLowerCase() === 'x';
      const number = Number.parseInt(entity.slice(isHex ? 2 : 1), isHex ? 16 : 10);
      if (Number.isFinite(number)) {
        try {
          return String.fromCodePoint(number);
        } catch (_) {
          return match;
        }
      }
      return match;
    }
    return Object.prototype.hasOwnProperty.call(named, entity.toLowerCase())
      ? named[entity.toLowerCase()]
      : match;
  });
}

function normalizeText(value) {
  return decodeEntities(String(value || '').replace(/<[^>]*>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function parseAttributes(source) {
  const attributes = {};
  const matcher = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
  let match;
  while ((match = matcher.exec(String(source || '')))) {
    const name = String(match[1] || '').toLowerCase();
    if (!name || Object.prototype.hasOwnProperty.call(attributes, name)) continue;
    const rawValue = match[2] !== undefined
      ? match[2]
      : match[3] !== undefined
        ? match[3]
        : match[4] !== undefined
          ? match[4]
          : '';
    attributes[name] = decodeEntities(rawValue);
  }
  return attributes;
}

function collectTagAttributes(html, tagName) {
  const tags = [];
  const matcher = new RegExp(`<${tagName}\\b([^>]*)>`, 'gi');
  let match;
  while ((match = matcher.exec(String(html || '')))) {
    tags.push(parseAttributes(match[1]));
  }
  return tags;
}

function collectHtmlFiles(directory, relativeDirectory = '') {
  const files = [];
  const absoluteDirectory = path.join(root, directory);
  let entries = [];
  try {
    entries = fs.readdirSync(absoluteDirectory, { withFileTypes: true });
  } catch (error) {
    report(directory, `could not list HTML sources (${error.code || error.message})`);
    return files;
  }

  entries.forEach((entry) => {
    const relativePath = path.join(relativeDirectory, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectHtmlFiles(path.join(directory, entry.name), relativePath));
      return;
    }
    if (entry.isFile() && entry.name.toLowerCase().endsWith('.html')) {
      files.push(toPosix(path.join(directory, entry.name)));
    }
  });
  return files;
}

function listSeoHtmlSources() {
  let rootEntries = [];
  try {
    rootEntries = fs.readdirSync(root, { withFileTypes: true });
  } catch (error) {
    report('HTML sources', `could not list project root (${error.code || error.message})`);
  }
  const rootFiles = rootEntries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.html'))
    .map((entry) => entry.name);
  return [...rootFiles, ...collectHtmlFiles('pages')].sort();
}

function normalizeRoutePathname(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const pathname = (raw.startsWith('/') ? raw : `/${raw}`)
    .split('#')[0]
    .split('?')[0];
  let normalized = pathname.replace(/\/+$/, '') || '/';
  if (normalized !== '/' && normalized.toLowerCase().endsWith('.html')) {
    normalized = normalized.slice(0, -5) || '/';
  }
  return normalized;
}

function parseCanonical(value, scope, validatePreferredForm = true) {
  let parsed;
  try {
    parsed = new URL(String(value || ''));
  } catch (_) {
    report(scope, `canonical URL is invalid: ${value || '(empty)'}`);
    return null;
  }
  if (validatePreferredForm) {
    if (parsed.origin !== SITE_ORIGIN || parsed.protocol !== 'https:') {
      report(scope, `canonical must use the preferred HTTPS origin ${SITE_ORIGIN}: ${value}`);
    }
    if (parsed.search || parsed.hash || parsed.username || parsed.password) {
      report(scope, `canonical must not contain credentials, a query, or a fragment: ${value}`);
    }
    if (parsed.pathname !== '/' && parsed.pathname.endsWith('/')) {
      report(scope, `canonical must use the site's non-trailing-slash form: ${value}`);
    }
    if (parsed.pathname.toLowerCase().endsWith('.html')) {
      report(scope, `canonical must use a clean URL instead of .html: ${value}`);
    }
  }
  return parsed;
}

function parseHtmlDocument(relativePath) {
  const html = readRequired(relativePath);
  const withoutComments = html.replace(/<!--[\s\S]*?-->/g, '');
  const links = collectTagAttributes(withoutComments, 'link');
  const metas = collectTagAttributes(withoutComments, 'meta');
  const canonicalValues = links
    .filter((attributes) => String(attributes.rel || '').toLowerCase().split(/\s+/).includes('canonical'))
    .map((attributes) => String(attributes.href || '').trim());
  const titleValues = [];
  const titleMatcher = /<title\b[^>]*>([\s\S]*?)<\/title>/gi;
  let titleMatch;
  while ((titleMatch = titleMatcher.exec(withoutComments))) {
    titleValues.push(normalizeText(titleMatch[1]));
  }

  const jsonLdBlocks = [];
  const scriptMatcher = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let scriptMatch;
  while ((scriptMatch = scriptMatcher.exec(withoutComments))) {
    const attributes = parseAttributes(scriptMatch[1]);
    if (String(attributes.type || '').trim().toLowerCase() === 'application/ld+json') {
      jsonLdBlocks.push(scriptMatch[2].trim());
    }
  }

  const robotsValues = metas
    .filter((attributes) => String(attributes.name || '').trim().toLowerCase() === 'robots')
    .map((attributes) => String(attributes.content || '').toLowerCase());

  if (canonicalValues.length !== 1) {
    report(relativePath, `expected exactly one canonical link, found ${canonicalValues.length}`);
  }

  const canonical = canonicalValues.length === 1
    ? parseCanonical(canonicalValues[0], relativePath, false)
    : null;

  return {
    relativePath,
    html,
    metas,
    canonical: canonical ? canonical.href : '',
    canonicalUrl: canonical,
    titleValues,
    h1Count: (withoutComments.match(/<h1\b/gi) || []).length,
    jsonLdBlocks,
    metaNoindex: robotsValues.some((value) => /(?:^|[,\s])noindex(?:$|[,\s])/.test(value))
  };
}

function getMetaValues(document, attributeName, key) {
  const expected = String(key || '').toLowerCase();
  return document.metas
    .filter((attributes) => String(attributes[attributeName] || '').trim().toLowerCase() === expected)
    .map((attributes) => String(attributes.content || '').trim());
}

function requireSingleMeta(document, attributeName, key) {
  const values = getMetaValues(document, attributeName, key);
  if (values.length !== 1) {
    report(document.canonical || document.relativePath, `expected exactly one ${key} meta tag in ${document.relativePath}, found ${values.length}`);
    return '';
  }
  if (!normalizeText(values[0])) {
    report(document.canonical || document.relativePath, `${key} must not be empty in ${document.relativePath}`);
  }
  return values[0];
}

function hasRobotsHeaderDirective(rule, directive) {
  const matcher = new RegExp(`(?:^|[,\\s])${directive}(?:$|[,\\s])`, 'i');
  return Array.isArray(rule && rule.headers) && rule.headers.some((header) => (
    String(header && header.key || '').trim().toLowerCase() === 'x-robots-tag'
      && matcher.test(String(header && header.value || ''))
  ));
}

function hasNoindexHeader(rule) {
  return hasRobotsHeaderDirective(rule, 'noindex');
}

function hasNoindexNofollowHeader(rule) {
  return hasNoindexHeader(rule) && hasRobotsHeaderDirective(rule, 'nofollow');
}

function isExactRouteSource(source) {
  return Boolean(source) && !/[:*()]/.test(String(source));
}

function isUnconditionalRule(rule) {
  return !(Array.isArray(rule && rule.has) && rule.has.length)
    && !(Array.isArray(rule && rule.missing) && rule.missing.length);
}

function chooseRepresentative(documents) {
  return [...documents].sort((left, right) => {
    const leftScore = left.relativePath.startsWith('pages/') ? 0 : 1;
    const rightScore = right.relativePath.startsWith('pages/') ? 0 : 1;
    return leftScore - rightScore || left.relativePath.localeCompare(right.relativePath);
  })[0];
}

function isUtilityOnlyGroup(documents) {
  return documents.length > 0
    && documents.every((document) => NON_SITEMAP_UTILITY_SOURCES.has(document.relativePath));
}

function validateJsonLd(document) {
  if (!document.jsonLdBlocks.length) {
    report(document.canonical, `expected JSON-LD in ${document.relativePath}`);
    return;
  }

  document.jsonLdBlocks.forEach((block, index) => {
    if (!block) {
      report(document.canonical, `JSON-LD block ${index + 1} is empty in ${document.relativePath}`);
      return;
    }
    let parsed;
    try {
      parsed = JSON.parse(block);
    } catch (error) {
      report(document.canonical, `JSON-LD block ${index + 1} is invalid JSON in ${document.relativePath}: ${error.message}`);
      return;
    }

    const roots = Array.isArray(parsed) ? parsed : [parsed];
    const hasSchemaContext = roots.some((entry) => {
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return false;
      const contexts = Array.isArray(entry['@context']) ? entry['@context'] : [entry['@context']];
      return contexts.some((context) => /^https?:\/\/schema\.org\/?$/i.test(String(context || '')));
    });
    const containsType = (value) => {
      if (Array.isArray(value)) return value.some(containsType);
      if (!value || typeof value !== 'object') return false;
      if (typeof value['@type'] === 'string' && value['@type'].trim()) return true;
      return Object.values(value).some(containsType);
    };

    if (!hasSchemaContext) {
      report(document.canonical, `JSON-LD block ${index + 1} needs a schema.org @context in ${document.relativePath}`);
    }
    if (!containsType(parsed)) {
      report(document.canonical, `JSON-LD block ${index + 1} needs at least one @type in ${document.relativePath}`);
    }
  });
}

function validateSocialImage(document, imageValue, label) {
  let imageUrl;
  try {
    imageUrl = new URL(imageValue);
  } catch (_) {
    report(document.canonical, `${label} must be an absolute URL in ${document.relativePath}: ${imageValue}`);
    return;
  }

  if (imageUrl.protocol !== 'https:') {
    report(document.canonical, `${label} must use HTTPS in ${document.relativePath}: ${imageValue}`);
  }

  const extension = path.posix.extname(imageUrl.pathname).toLowerCase();
  if (!RASTER_IMAGE_EXTENSIONS.has(extension)) {
    report(document.canonical, `${label} must be PNG, JPEG, or WebP (not SVG/AVIF) in ${document.relativePath}: ${imageValue}`);
  }

  if (imageUrl.origin === SITE_ORIGIN) {
    let decodedPathname = imageUrl.pathname;
    try {
      decodedPathname = decodeURIComponent(decodedPathname);
    } catch (_) {}
    const imagePath = path.join(root, ...decodedPathname.split('/').filter(Boolean));
    if (!fs.existsSync(imagePath)) {
      report(document.canonical, `same-origin social image does not exist: ${toPosix(path.relative(root, imagePath))}`);
    }
  }

  const canonicalPathname = document.canonicalUrl && document.canonicalUrl.pathname;
  const projectMatch = canonicalPathname && /^\/portfolio\/([^/]+)$/.exec(canonicalPathname);
  if (projectMatch) {
    const imageStem = path.posix.basename(imageUrl.pathname, extension);
    if (!imageUrl.pathname.startsWith('/img/projects/') || imageStem !== projectMatch[1]) {
      report(document.canonical, `${label} must use the matching img/projects/${projectMatch[1]} raster asset in ${document.relativePath}`);
    }
  }
}

function validateIndexableDocument(document) {
  const scope = document.canonical || document.relativePath;
  parseCanonical(document.canonical, scope);
  if (document.titleValues.length !== 1 || !document.titleValues[0]) {
    report(scope, `expected exactly one non-empty title in ${document.relativePath}, found ${document.titleValues.length}`);
  }
  const title = document.titleValues.length === 1 ? document.titleValues[0] : '';
  const description = requireSingleMeta(document, 'name', 'description');

  if (document.h1Count !== 1) {
    report(scope, `expected exactly one h1 in ${document.relativePath}, found ${document.h1Count}`);
  }

  requireSingleMeta(document, 'property', 'og:title');
  requireSingleMeta(document, 'property', 'og:site_name');
  requireSingleMeta(document, 'property', 'og:description');
  const ogUrl = requireSingleMeta(document, 'property', 'og:url');
  const ogImage = requireSingleMeta(document, 'property', 'og:image');
  requireSingleMeta(document, 'property', 'og:image:alt');
  const ogImageWidth = requireSingleMeta(document, 'property', 'og:image:width');
  const ogImageHeight = requireSingleMeta(document, 'property', 'og:image:height');
  requireSingleMeta(document, 'property', 'og:type');

  requireSingleMeta(document, 'name', 'twitter:card');
  requireSingleMeta(document, 'name', 'twitter:site');
  requireSingleMeta(document, 'name', 'twitter:title');
  requireSingleMeta(document, 'name', 'twitter:description');
  const twitterImage = requireSingleMeta(document, 'name', 'twitter:image');
  requireSingleMeta(document, 'name', 'twitter:image:alt');

  if (ogUrl && ogUrl !== document.canonical) {
    report(scope, `og:url must exactly match the canonical in ${document.relativePath}`);
  }
  if (ogImageWidth && (!/^\d+$/.test(ogImageWidth) || Number(ogImageWidth) <= 0)) {
    report(scope, `og:image:width must be a positive integer in ${document.relativePath}`);
  }
  if (ogImageHeight && (!/^\d+$/.test(ogImageHeight) || Number(ogImageHeight) <= 0)) {
    report(scope, `og:image:height must be a positive integer in ${document.relativePath}`);
  }
  if (ogImage) validateSocialImage(document, ogImage, 'og:image');
  if (twitterImage) validateSocialImage(document, twitterImage, 'twitter:image');

  validateJsonLd(document);
  return { title, description };
}

function validateUniqueMetadata(records, field) {
  const byValue = new Map();
  records.forEach((record) => {
    const normalized = normalizeText(record[field]).toLowerCase();
    if (!normalized) return;
    if (!byValue.has(normalized)) byValue.set(normalized, []);
    byValue.get(normalized).push(record.canonical);
  });
  byValue.forEach((canonicals) => {
    if (canonicals.length > 1) {
      report(`Unique ${field}`, `${canonicals.join(', ')} share the same ${field}`);
    }
  });
}

function parseSitemapLocations(xml) {
  const locations = [];
  const matcher = /<loc>\s*([^<]+?)\s*<\/loc>/gi;
  let match;
  while ((match = matcher.exec(String(xml || '')))) {
    locations.push(decodeEntities(match[1]).trim());
  }
  return locations;
}

function validateSitemap(groups, exactNoindexRoutes) {
  const locations = parseSitemapLocations(readRequired('sitemap.xml'));
  const sitemapSet = new Set();

  if (locations.length !== EXPECTED_SITEMAP_URL_COUNT) {
    report('sitemap.xml', `expected ${EXPECTED_SITEMAP_URL_COUNT} URLs, found ${locations.length}`);
  }

  locations.forEach((location) => {
    if (sitemapSet.has(location)) {
      report('sitemap.xml', `duplicate URL: ${location}`);
    }
    sitemapSet.add(location);
    const parsed = parseCanonical(location, 'sitemap.xml');
    if (!parsed) return;
    const pathname = normalizeRoutePathname(parsed.pathname);
    if (pathname === '/documents' || pathname.startsWith('/documents/')) {
      report('sitemap.xml', `document URL must not be included: ${location}`);
    }
    if (exactNoindexRoutes.has(pathname)) {
      report('sitemap.xml', `noindex route must not be included: ${location}`);
    }
    const group = groups.get(location);
    if (group && !group.indexable) {
      report('sitemap.xml', `HTML robots metadata marks this URL noindex: ${location}`);
    }
  });

  const expectedSet = new Set(
    [...groups.values()]
      .filter((group) => group.indexable && !group.utilityOnly)
      .map((group) => group.canonical)
  );
  const missing = [...expectedSet].filter((canonical) => !sitemapSet.has(canonical)).sort();
  const unexpected = [...sitemapSet].filter((canonical) => !expectedSet.has(canonical)).sort();
  if (missing.length) {
    report('sitemap.xml', `missing ${missing.length} indexable canonical(s): ${missing.join(', ')}`);
  }
  if (unexpected.length) {
    report('sitemap.xml', `contains ${unexpected.length} URL(s) that are not indexable canonical HTML pages: ${unexpected.join(', ')}`);
  }
  return sitemapSet;
}

function conditionUsesUserAgent(condition) {
  return String(condition && condition.type || '').toLowerCase() === 'header'
    && String(condition && condition.key || '').toLowerCase() === 'user-agent';
}

function findRouteRule(rules, source, destination, requirePermanent = false) {
  return rules.find((rule) => (
    String(rule && rule.source || '') === source
      && String(rule && rule.destination || '') === destination
      && (!requirePermanent || rule.permanent === true)
  ));
}

function escapeRegex(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function matchRouteSource(source, pathname) {
  const rawSource = String(source || '');
  const names = [];
  let pattern = '^';
  let offset = 0;
  const tokenMatcher = /:([a-zA-Z][a-zA-Z0-9_]*)(\*)?/g;
  let token;
  while ((token = tokenMatcher.exec(rawSource))) {
    pattern += escapeRegex(rawSource.slice(offset, token.index));
    names.push(token[1]);
    pattern += token[2] ? '(.*)' : '([^/]+)';
    offset = tokenMatcher.lastIndex;
  }
  pattern += `${escapeRegex(rawSource.slice(offset))}$`;

  let match;
  try {
    match = new RegExp(pattern).exec(String(pathname || ''));
  } catch (_) {
    return null;
  }
  if (!match) return null;
  return names.reduce((params, name, index) => {
    params[name] = match[index + 1] || '';
    return params;
  }, {});
}

function interpolateRouteDestination(destination, params) {
  return String(destination || '').replace(/:([a-zA-Z][a-zA-Z0-9_]*)(\*)?/g, (_match, name) => (
    Object.prototype.hasOwnProperty.call(params, name) ? params[name] : ''
  ));
}

function findFirstUnconditionalRedirect(redirects, pathname) {
  for (const redirect of redirects) {
    if (!isUnconditionalRule(redirect)) continue;
    const params = matchRouteSource(redirect.source, pathname);
    if (!params) continue;
    return {
      redirect,
      destination: interpolateRouteDestination(redirect.destination, params)
    };
  }
  return null;
}

function cleanUrlRedirectDestination(pathname) {
  const value = String(pathname || '');
  if (!value.toLowerCase().endsWith('.html')) return '';
  const withoutExtension = value.slice(0, -5) || '/';
  if (withoutExtension.toLowerCase() === '/index') return '/';
  if (withoutExtension.toLowerCase().endsWith('/index')) {
    return withoutExtension.slice(0, -6) || '/';
  }
  return withoutExtension;
}

function findBulkRedirect(bulkRedirects, pathname) {
  const normalized = String(pathname || '').toLowerCase();
  const redirect = bulkRedirects.find((candidate) => (
    String(candidate && candidate.source || '').toLowerCase() === normalized
  ));
  return redirect
    ? { redirect, destination: String(redirect.destination || ''), kind: 'bulk redirect' }
    : null;
}

function findFirstPlatformRedirect(vercel, redirects, bulkRedirects, pathname) {
  const bulk = findBulkRedirect(bulkRedirects, pathname);
  if (bulk) return bulk;

  if (vercel && vercel.cleanUrls === true) {
    const destination = cleanUrlRedirectDestination(pathname);
    if (destination) {
      return {
        redirect: { source: pathname, destination },
        destination,
        kind: 'cleanUrls redirect'
      };
    }
  }

  const configured = findFirstUnconditionalRedirect(redirects, pathname);
  return configured ? { ...configured, kind: 'deployment redirect' } : null;
}

function resolveDeploymentRedirectDestination(redirects, pathname) {
  let current = String(pathname || '');
  const visited = new Set();
  while (current && !/^https?:\/\//i.test(current) && !visited.has(current)) {
    visited.add(current);
    const redirect = findFirstUnconditionalRedirect(redirects, current);
    if (!redirect) return current;
    current = redirect.destination;
  }
  if (visited.has(current)) {
    report('vercel.json', `redirect cycle detected while resolving ${pathname}`);
  }
  return current;
}

function addExpectedBulkRedirect(expected, source, destination) {
  const existing = expected.get(source);
  if (existing && existing !== destination) {
    report(EXPECTED_BULK_REDIRECTS_PATH, `${source} has conflicting expected destinations ${existing} and ${destination}`);
    return;
  }
  expected.set(source, destination);
}

function buildExpectedBulkRedirects(documents, redirects) {
  const expected = new Map();

  documents
    .filter((document) => document.relativePath.startsWith('pages/') && document.canonicalUrl)
    .forEach((document) => {
      const relative = document.relativePath.slice('pages/'.length, -'.html'.length);
      const destination = resolveDeploymentRedirectDestination(redirects, document.canonicalUrl.pathname);
      const htmlSource = `/pages/${relative}.html`;
      const cleanSource = `/pages/${relative}`;
      addExpectedBulkRedirect(expected, htmlSource, destination);

      const current = findFirstUnconditionalRedirect(redirects, cleanSource);
      if (!current || current.destination !== destination) {
        addExpectedBulkRedirect(expected, cleanSource, destination);
      }

      if (relative.startsWith('portfolio/') && !relative.slice('portfolio/'.length).includes('/')) {
        const project = relative.slice('portfolio/'.length);
        addExpectedBulkRedirect(expected, `/projects/${project}.html`, destination);
      }
    });

  redirects.forEach((redirect) => {
    if (!isUnconditionalRule(redirect) || !isExactRouteSource(redirect.source)) return;
    const source = String(redirect.source || '');
    const destination = resolveDeploymentRedirectDestination(redirects, redirect.destination);
    const cleanDestination = cleanUrlRedirectDestination(source);
    if (cleanDestination && cleanDestination !== destination) {
      addExpectedBulkRedirect(expected, source, destination);
    }

    const firstConfigured = findFirstUnconditionalRedirect(redirects, source);
    if (firstConfigured && firstConfigured.redirect !== redirect && firstConfigured.destination !== destination) {
      addExpectedBulkRedirect(expected, source, destination);
    }
  });

  return expected;
}

function validateBulkRedirectManifest(vercel, redirects, documents) {
  const configuredPath = String(vercel && vercel.bulkRedirectsPath || '');
  if (configuredPath !== EXPECTED_BULK_REDIRECTS_PATH) {
    report('vercel.json', `bulkRedirectsPath must be ${EXPECTED_BULK_REDIRECTS_PATH}`);
  }

  const parsed = readJsonRequired(EXPECTED_BULK_REDIRECTS_PATH);
  if (!Array.isArray(parsed)) {
    report(EXPECTED_BULK_REDIRECTS_PATH, 'must contain a JSON array');
    return [];
  }

  const actual = new Map();
  parsed.forEach((redirect, index) => {
    const scope = `${EXPECTED_BULK_REDIRECTS_PATH} entry ${index + 1}`;
    const source = String(redirect && redirect.source || '');
    const destination = String(redirect && redirect.destination || '');
    if (!source.startsWith('/') || /[:*()?#]/.test(source)) {
      report(scope, `source must be one exact internal pathname: ${source || '(empty)'}`);
    }
    if (!destination.startsWith('/') || /[?#]/.test(destination)) {
      report(scope, `destination must be one clean internal pathname: ${destination || '(empty)'}`);
    }
    if (Number(redirect && redirect.statusCode) !== 308) {
      report(scope, 'statusCode must be 308');
    }
    if (redirect && redirect.preserveQueryParams !== true) {
      report(scope, 'preserveQueryParams must be true');
    }
    if (actual.has(source)) {
      report(scope, `duplicate source ${source}`);
    }
    actual.set(source, destination);
  });

  const expected = buildExpectedBulkRedirects(documents, redirects);
  expected.forEach((destination, source) => {
    if (!actual.has(source)) {
      report(EXPECTED_BULK_REDIRECTS_PATH, `missing priority redirect ${source} -> ${destination}`);
      return;
    }
    if (actual.get(source) !== destination) {
      report(EXPECTED_BULK_REDIRECTS_PATH, `${source} must redirect directly to ${destination}, found ${actual.get(source)}`);
    }
  });
  actual.forEach((destination, source) => {
    if (!expected.has(source)) {
      report(EXPECTED_BULK_REDIRECTS_PATH, `unexpected redirect outside the known canonical aliases: ${source} -> ${destination}`);
    }
    const downstreamBulk = actual.get(destination);
    const downstreamConfigured = findFirstUnconditionalRedirect(redirects, destination);
    if (downstreamBulk || downstreamConfigured) {
      const next = downstreamBulk || downstreamConfigured.destination;
      report(EXPECTED_BULK_REDIRECTS_PATH, `${source} still requires another redirect hop: ${destination} -> ${next}`);
    }
  });

  return parsed;
}

function validateOneHopRedirects(vercel, redirects, bulkRedirects) {
  ONE_HOP_REDIRECT_CASES.forEach(([requestPath, expectedDestination]) => {
    const first = findFirstPlatformRedirect(vercel, redirects, bulkRedirects, requestPath);
    if (!first) {
      report('vercel.json', `missing canonical redirect for ${requestPath}`);
      return;
    }
    if (first.destination !== expectedDestination) {
      report('vercel.json', `canonical redirect for ${requestPath} must go directly to ${expectedDestination}, found ${first.destination}`);
      return;
    }
    const second = findFirstPlatformRedirect(vercel, redirects, bulkRedirects, first.destination);
    if (second) {
      report('vercel.json', `canonical redirect for ${requestPath} requires another hop: ${first.destination} -> ${second.destination}`);
    }
  });
}

function hasQueryCondition(rule, key, value) {
  return Array.isArray(rule && rule.has) && rule.has.some((condition) => (
    String(condition && condition.type || '').toLowerCase() === 'query'
      && String(condition && condition.key || '') === key
      && String(condition && condition.value || '') === value
  ));
}

function validateConditionalNoindexHeaders(headers) {
  const queryVariants = [
    ['mode', 'professional'],
    ...PROFESSIONAL_AUDIENCE_KEYS.map((audience) => ['audience', audience])
  ];
  queryVariants.forEach(([key, value]) => {
    ['/', '/:path*'].forEach((source) => {
      const rule = headers.find((candidate) => (
        String(candidate && candidate.source || '') === source
          && hasQueryCondition(candidate, key, value)
      ));
      if (!rule || !hasNoindexNofollowHeader(rule)) {
        report('vercel.json', `missing query-specific X-Robots-Tag noindex, nofollow for ${source}?${key}=${value}`);
        return;
      }
      const conditions = [
        ...(Array.isArray(rule.has) ? rule.has : []),
        ...(Array.isArray(rule.missing) ? rule.missing : [])
      ];
      if (conditions.some(conditionUsesUserAgent)) {
        report('vercel.json', `query noindex coverage must not vary by user-agent: ${source}?${key}=${value}`);
      }
    });
  });
}

function parseRobotsGroups(contents) {
  const groups = [];
  let agents = [];
  let directives = [];
  const flush = () => {
    if (agents.length) groups.push({ agents, directives });
    agents = [];
    directives = [];
  };

  String(contents || '').split(/\r?\n/).forEach((line) => {
    const normalized = line.replace(/#.*$/, '').trim();
    if (!normalized) return;
    const match = normalized.match(/^([^:]+):\s*(.*)$/);
    if (!match) return;
    const name = match[1].trim().toLowerCase();
    const value = match[2].trim();
    if (name === 'user-agent') {
      if (directives.length) flush();
      agents.push(value.toLowerCase());
      return;
    }
    if ((name === 'allow' || name === 'disallow') && agents.length) {
      directives.push({ type: name, value });
    }
  });
  flush();
  return groups;
}

function robotsPatternMatches(pattern, pathname) {
  const value = String(pattern || '');
  if (!value) return false;
  const endAnchored = value.endsWith('$');
  const source = endAnchored ? value.slice(0, -1) : value;
  const regexSource = escapeRegex(source).replace(/\\\*/g, '.*');
  try {
    return new RegExp(`^${regexSource}${endAnchored ? '$' : ''}`).test(String(pathname || '/'));
  } catch (_) {
    return false;
  }
}

function isRobotsAllowed(contents, userAgent, pathname) {
  const normalizedAgent = String(userAgent || '').toLowerCase();
  const matchedGroups = parseRobotsGroups(contents)
    .map((group) => {
      const matches = group.agents
        .filter((agent) => agent === '*' || normalizedAgent.includes(agent))
        .map((agent) => (agent === '*' ? 0 : agent.length));
      return matches.length ? { group, specificity: Math.max(...matches) } : null;
    })
    .filter(Boolean);
  if (!matchedGroups.length) return true;
  const bestSpecificity = Math.max(...matchedGroups.map((entry) => entry.specificity));
  const matchingDirectives = matchedGroups
    .filter((entry) => entry.specificity === bestSpecificity)
    .flatMap((entry) => entry.group.directives)
    .filter((directive) => robotsPatternMatches(directive.value, pathname))
    .map((directive) => ({
      ...directive,
      specificity: directive.value.replace(/[*$]/g, '').length
    }))
    .sort((left, right) => (
      right.specificity - left.specificity
        || (left.type === 'allow' ? -1 : 1)
    ));
  return !matchingDirectives.length || matchingDirectives[0].type === 'allow';
}

function validateGooglebotAccessibility(robots, sitemapSet) {
  const paths = [
    DOCUMENT_CRAWL_SAMPLE,
    ...[...sitemapSet].map((location) => {
      try {
        return new URL(location).pathname;
      } catch (_) {
        return '';
      }
    }).filter(Boolean)
  ];
  [...new Set(paths)].forEach((pathname) => {
    if (!isRobotsAllowed(robots, GOOGLEBOT_USER_AGENT, pathname)) {
      report('robots.txt', `${GOOGLEBOT_USER_AGENT} must be allowed to crawl ${pathname} so response noindex/canonical signals can be read`);
    }
  });
}

function ruleCoversPrefix(rule, prefix) {
  if (!isUnconditionalRule(rule) || !hasNoindexHeader(rule)) return false;
  const source = String(rule.source || '');
  return source === prefix
    || source.startsWith(`${prefix}/`)
    || source.startsWith(`${prefix}(`)
    || source.startsWith(`${prefix}:`)
    || source.startsWith(`${prefix}*`);
}

function validateVercelAndRobots(vercel, sitemapSet, documents) {
  const rewrites = Array.isArray(vercel.rewrites) ? vercel.rewrites : [];
  const redirects = Array.isArray(vercel.redirects) ? vercel.redirects : [];
  const headers = Array.isArray(vercel.headers) ? vercel.headers : [];
  const bulkRedirects = validateBulkRedirectManifest(vercel, redirects, documents);

  rewrites.forEach((rewrite) => {
    const conditions = [
      ...(Array.isArray(rewrite && rewrite.has) ? rewrite.has : []),
      ...(Array.isArray(rewrite && rewrite.missing) ? rewrite.missing : [])
    ];
    if (conditions.some(conditionUsesUserAgent)) {
      report('vercel.json', `content must not be rewritten by user-agent: ${rewrite.source} -> ${rewrite.destination}`);
    }
  });

  validateConditionalNoindexHeaders(headers);

  [
    ['/ai/:path*', '/dist/ai-pages/:path*']
  ].forEach(([source, destination]) => {
    if (!findRouteRule(rewrites, source, destination)) {
      report('vercel.json', `missing rewrite ${source} -> ${destination}`);
    }
  });

  [
    ['/pages/:path*', '/:path*'],
    ['/dist/ai-pages/:path*', '/ai/:path*']
  ].forEach(([source, destination]) => {
    if (!findRouteRule(redirects, source, destination, true)) {
      report('vercel.json', `missing permanent redirect ${source} -> ${destination}`);
    }
  });

  ['/pages', '/ai', '/dist/ai-pages'].forEach((prefix) => {
    if (!headers.some((rule) => ruleCoversPrefix(rule, prefix))) {
      report('vercel.json', `missing unconditional X-Robots-Tag noindex coverage for ${prefix}/`);
    }
  });

  const documentsHeader = headers.find((rule) => (
    String(rule && rule.source || '') === '/documents/:path*'
      && isUnconditionalRule(rule)
      && hasNoindexNofollowHeader(rule)
  ));
  if (!documentsHeader) {
    report('vercel.json', 'missing unconditional X-Robots-Tag noindex, nofollow coverage for /documents/:path*');
  }

  REQUIRED_ALIAS_REDIRECTS.forEach(([source, destination]) => {
    if (!findRouteRule(redirects, source, destination, true)) {
      report('vercel.json', `missing permanent alias redirect ${source} -> ${destination}`);
    }
  });
  validateOneHopRedirects(vercel, redirects, bulkRedirects);

  if (!findRouteRule(rewrites, '/portfolio/:project', '/pages/portfolio/:project')) {
    report('vercel.json', 'missing project-page rewrite /portfolio/:project -> /pages/portfolio/:project');
  }

  redirects.forEach((redirect) => {
    const source = String(redirect && redirect.source || '');
    const destination = String(redirect && redirect.destination || '');
    if (source !== '/projects' && source.startsWith('/projects/') && destination === '/portfolio') {
      report('vercel.json', `project alias loses its project id: ${source} -> ${destination}`);
    }
  });

  const robots = readRequired('robots.txt');
  const sitemapDirective = new RegExp(`^\\s*Sitemap:\\s*${SITE_ORIGIN.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\/sitemap\\.xml\\s*$`, 'im');
  if (!sitemapDirective.test(robots)) {
    report('robots.txt', `missing Sitemap: ${SITE_ORIGIN}/sitemap.xml`);
  }

  const disallowMatcher = /^\s*Disallow:\s*([^#\r\n]*)/gim;
  let disallowMatch;
  while ((disallowMatch = disallowMatcher.exec(robots))) {
    const rule = String(disallowMatch[1] || '').trim().toLowerCase();
    if (/^\/(?:pages|ai|dist\/ai-pages)(?:[/\s*$]|$)/.test(rule)) {
      report('robots.txt', `must not block redirectable/noindex content path: ${rule}`);
    }
  }
  validateGooglebotAccessibility(robots, sitemapSet);
}

function extractElementWithAttribute(html, attributeName) {
  const openingMatcher = /<([a-z][a-z0-9:-]*)\b([^>]*)>/gi;
  let opening;
  while ((opening = openingMatcher.exec(String(html || '')))) {
    const attributes = parseAttributes(opening[2]);
    if (Object.prototype.hasOwnProperty.call(attributes, attributeName.toLowerCase())) break;
  }
  if (!opening) return '';
  const tagName = opening[1];
  const tagMatcher = new RegExp(`</?${tagName}\\b[^>]*>`, 'gi');
  tagMatcher.lastIndex = opening.index;
  let depth = 0;
  let tagMatch;
  while ((tagMatch = tagMatcher.exec(html))) {
    if (tagMatch[0].startsWith('</')) {
      depth -= 1;
      if (depth === 0) return html.slice(opening.index, tagMatcher.lastIndex);
    } else if (!tagMatch[0].endsWith('/>')) {
      depth += 1;
    }
  }
  return '';
}

function validateDirectoryResultAnchors() {
  DIRECTORY_RESULT_CONTRACTS.forEach((contract) => {
    const html = readRequired(contract.file);
    if (
      !/<\/main>/i.test(html)
      || !/<footer\b[\s\S]*?<\/footer>/i.test(html)
      || !/<\/body>\s*<\/html>\s*$/i.test(html)
    ) {
      report(contract.file, 'incomplete HTML document shell');
      return;
    }
    const container = extractElementWithAttribute(html, 'data-portfolio-results');
    if (!container) {
      report(contract.file, 'missing a complete raw HTML [data-portfolio-results] container');
      return;
    }

    const openingTags = [];
    const openingTagMatcher = /<([a-z][a-z0-9:-]*)\b([^>]*)>/gi;
    let openingTagMatch;
    while ((openingTagMatch = openingTagMatcher.exec(container))) {
      openingTags.push({
        name: openingTagMatch[1].toLowerCase(),
        attributes: parseAttributes(openingTagMatch[2])
      });
    }

    const resultItemCount = openingTags.filter((tag) => (
      String(tag.attributes.role || '').toLowerCase() === 'listitem'
    )).length;
    const crawlablePaths = new Set();
    openingTags.filter((tag) => tag.name === 'a').forEach((tag) => {
      const href = String(tag.attributes.href || '').trim();
      if (!href || href.startsWith('#') || /^(?:javascript|data|mailto|tel):/i.test(href)) return;
      let url;
      try {
        url = new URL(href, `${SITE_ORIGIN}/`);
      } catch (_) {
        return;
      }
      if (url.origin === SITE_ORIGIN && url.pathname.startsWith(contract.pathPrefix)) {
        crawlablePaths.add(url.pathname);
      }
    });

    if (!resultItemCount) {
      report(contract.file, `${contract.label} raw results container has no list items`);
    }
    if (!crawlablePaths.size) {
      report(contract.file, `${contract.label} raw results container has no crawlable ${contract.pathPrefix} anchors`);
    } else if (crawlablePaths.size < resultItemCount) {
      report(contract.file, `${contract.label} raw results expose ${resultItemCount} list items but only ${crawlablePaths.size} unique crawlable anchors`);
    }
  });
}

function readJsonRequired(relativePath) {
  const contents = readRequired(relativePath);
  try {
    return JSON.parse(contents);
  } catch (error) {
    report(relativePath, `invalid JSON: ${error.message}`);
    return null;
  }
}

function validateProfessionalPortfolioPath(scope, audienceKey, field, value) {
  let url;
  try {
    url = new URL(String(value || ''), SITE_ORIGIN);
  } catch (_) {
    report(scope, `${field} is not a valid URL: ${value || '(empty)'}`);
    return;
  }
  if (url.origin !== SITE_ORIGIN || url.pathname !== '/portfolio') {
    report(scope, `${field} must target the clean /portfolio path`);
  }
  if (url.searchParams.get('audience') !== audienceKey) {
    report(scope, `${field} must include audience=${audienceKey}`);
  }
  if (url.searchParams.get('mode') !== 'professional') {
    report(scope, `${field} must include mode=professional`);
  }
}

function validateProfessionalLinkGeneration() {
  PROFESSIONAL_AUDIENCE_KEYS.forEach((audienceKey) => {
    const sourcePath = `content/audiences/${audienceKey}.json`;
    const audience = readJsonRequired(sourcePath);
    if (!audience) return;
    ['portfolioPath', 'portfolioAllPath'].forEach((field) => {
      validateProfessionalPortfolioPath(sourcePath, audienceKey, field, audience[field]);
    });
  });

  try {
    const generatedPath = path.join(root, 'js', 'common', 'audience-config.js');
    delete require.cache[require.resolve(generatedPath)];
    const generated = require(generatedPath);
    PROFESSIONAL_AUDIENCE_KEYS.forEach((audienceKey) => {
      const audience = generated && generated.audiences && generated.audiences[audienceKey];
      if (!audience) {
        report('js/common/audience-config.js', `missing generated ${audienceKey} audience configuration`);
        return;
      }
      ['portfolioPath', 'portfolioAllPath'].forEach((field) => {
        validateProfessionalPortfolioPath('js/common/audience-config.js', audienceKey, field, audience[field]);
      });
    });
  } catch (error) {
    report('js/common/audience-config.js', `could not load generated audience configuration (${error.message})`);
  }

  const footer = readJsonRequired('content/site/footer.json');
  const professionalColumns = footer && footer.navVariants && footer.navVariants.professional;
  const portfolioLinks = (Array.isArray(professionalColumns) ? professionalColumns : [])
    .flatMap((column) => (Array.isArray(column && column.links) ? column.links : []))
    .filter((link) => String(link && link.label || '').trim().toLowerCase() === 'portfolio');
  if (portfolioLinks.length !== 1 || String(portfolioLinks[0] && portfolioLinks[0].href || '') !== 'portfolio') {
    report('content/site/footer.json', 'shared static footer must use the clean portfolio URL; runtime audience navigation adds professional query state');
  }
}

function validateProfessionalMarkupLinks(documents) {
  documents.forEach((document) => {
    collectTagAttributes(document.html, 'a').forEach((attributes) => {
      const href = String(attributes.href || '').trim();
      if (!href) return;
      let url;
      try {
        url = new URL(href, SITE_ORIGIN);
      } catch (_) {
        return;
      }
      if (url.origin !== SITE_ORIGIN) return;
      const audience = String(url.searchParams.get('audience') || '').toLowerCase();
      if (!PROFESSIONAL_AUDIENCE_KEYS.includes(audience)) return;
      if (url.searchParams.get('mode') !== 'professional') {
        report(document.relativePath, `professional audience link must include mode=professional: ${href}`);
      }
    });
  });
}

function main() {
  let vercel = {};
  try {
    vercel = JSON.parse(readRequired('vercel.json'));
  } catch (error) {
    report('vercel.json', `invalid JSON: ${error.message}`);
  }

  const headerRules = Array.isArray(vercel.headers) ? vercel.headers : [];
  const exactNoindexRoutes = new Set(
    headerRules
      .filter((rule) => isUnconditionalRule(rule) && isExactRouteSource(rule.source) && hasNoindexHeader(rule))
      .map((rule) => normalizeRoutePathname(rule.source))
      .filter(Boolean)
  );

  const documents = listSeoHtmlSources().map(parseHtmlDocument);
  const groups = new Map();
  documents.forEach((document) => {
    if (!document.canonical || !document.canonicalUrl) return;
    if (!groups.has(document.canonical)) {
      groups.set(document.canonical, {
        canonical: document.canonical,
        documents: []
      });
    }
    groups.get(document.canonical).documents.push(document);
  });

  const metadataRecords = [];
  groups.forEach((group) => {
    const pathname = normalizeRoutePathname(new URL(group.canonical).pathname);
    const headerNoindex = exactNoindexRoutes.has(pathname);
    group.indexable = !headerNoindex && group.documents.some((document) => !document.metaNoindex);
    group.utilityOnly = isUtilityOnlyGroup(group.documents);
    group.representative = chooseRepresentative(group.documents);
    if (!group.indexable) return;
    const metadata = validateIndexableDocument(group.representative);
    metadataRecords.push({
      canonical: group.canonical,
      title: metadata.title,
      description: metadata.description
    });
  });

  validateUniqueMetadata(metadataRecords, 'title');
  validateUniqueMetadata(metadataRecords, 'description');
  validateProfessionalMarkupLinks(documents);
  const sitemapSet = validateSitemap(groups, exactNoindexRoutes);
  validateVercelAndRobots(vercel, sitemapSet, documents);
  validateDirectoryResultAnchors();
  validateProfessionalLinkGeneration();

  if (errors.length) {
    process.stderr.write(`SEO validation failed with ${errors.length} issue(s):\n`);
    [...new Set(errors)].sort().forEach((error) => {
      process.stderr.write(`- ${error}\n`);
    });
    process.exitCode = 1;
    return;
  }

  const indexableCount = [...groups.values()].filter((group) => group.indexable).length;
  process.stdout.write(
    `SEO validation passed: ${documents.length} source HTML files, ${groups.size} unique canonicals, `
      + `${indexableCount} indexable canonicals, ${sitemapSet.size} sitemap URLs.\n`
  );
}

main();
