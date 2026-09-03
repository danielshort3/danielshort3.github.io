'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const SITE_ORIGIN = 'https://www.danielshort.me';
const REALM_QUERY_KEYS = new Set(['audience', 'mode']);

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

function readJson(relativePath) {
  return JSON.parse(read(relativePath));
}

function exists(relativePath) {
  return fs.existsSync(path.join(ROOT, relativePath));
}

function toPosix(relativePath) {
  return String(relativePath || '').replace(/\\/g, '/');
}

function walkFiles(relativeDir, extension) {
  const start = path.join(ROOT, relativeDir);
  if (!fs.existsSync(start)) return [];
  const files = [];
  const stack = [start];
  while (stack.length) {
    const current = stack.pop();
    fs.readdirSync(current, { withFileTypes: true }).forEach((entry) => {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(full);
      if (entry.isFile() && entry.name.toLowerCase().endsWith(extension)) files.push(full);
    });
  }
  return files.sort();
}

function decodeHtmlAttribute(value) {
  return String(value || '')
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&#x27;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>');
}

function tagAttribute(tag, name) {
  const escaped = String(name).replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const match = new RegExp(`\\s${escaped}\\s*=\\s*(["'])([\\s\\S]*?)\\1`, 'i').exec(tag);
  return decodeHtmlAttribute(match ? match[2] : '');
}

function extractNavigationTags(html) {
  const tags = [];
  const pattern = /<(?:a|area|form|button|input)\b[^>]*>/gi;
  let match;
  while ((match = pattern.exec(String(html || '')))) {
    const tag = match[0];
    const tagName = /^<([a-z]+)/i.exec(tag)?.[1]?.toLowerCase() || '';
    const attribute = tagName === 'form'
      ? 'action'
      : (tagName === 'button' || tagName === 'input' ? 'formaction' : 'href');
    const target = tagAttribute(tag, attribute);
    if (!target) continue;
    tags.push({ tag, target });
  }
  return tags;
}

function canonicalUrlForHtml(html) {
  const tag = /<link\b[^>]*\brel=["']canonical["'][^>]*>/i.exec(String(html || ''))?.[0] || '';
  const href = tagAttribute(tag, 'href');
  try {
    return new URL(href || '/', SITE_ORIGIN);
  } catch (_) {
    return new URL('/', SITE_ORIGIN);
  }
}

function documentBaseUrl(html) {
  const canonical = canonicalUrlForHtml(html);
  const tag = /<base\b[^>]*>/i.exec(String(html || ''))?.[0] || '';
  const href = tagAttribute(tag, 'href');
  try {
    return new URL(href || canonical.href, canonical);
  } catch (_) {
    return canonical;
  }
}

function extractHomepageReferences() {
  const html = read('index.html');
  const baseUrl = documentBaseUrl(html);
  const references = extractNavigationTags(html).map(({ target }) => ({
    target,
    baseUrl,
    source: 'index.html'
  }));
  ['about', 'projects', 'tools', 'games', 'contact'].forEach((category) => {
    references.push({
      target: `/#${category}`,
      baseUrl,
      source: `index.html category state ${category}`
    });
  });
  return references;
}

function extractManagedShellEntryReferences() {
  const rootHtmlFiles = fs.readdirSync(ROOT, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.html'))
    .map((entry) => path.join(ROOT, entry.name));
  const candidates = [...rootHtmlFiles, ...walkFiles('pages', '.html')];
  const references = [];

  candidates.forEach((filePath) => {
    const html = fs.readFileSync(filePath, 'utf8');
    if (!html.includes('data-personal-accordion-shell')) return;
    const relativePath = toPosix(path.relative(ROOT, filePath));
    references.push({
      target: canonicalUrlForHtml(html).href,
      baseUrl: new URL('/', SITE_ORIGIN),
      source: `${relativePath} canonical shell entry`
    });
  });

  return references;
}

function extractSearchIndexReferences(assert) {
  const index = readJson('dist/search-index.json');
  assert(index && Array.isArray(index.pages), 'dist/search-index.json should contain a pages array after build');
  return index.pages.map((entry, indexPosition) => {
    assert(entry && typeof entry.url === 'string' && entry.url.trim(),
      `Search index entry ${indexPosition + 1} should contain a URL`);
    return {
      target: entry.url,
      baseUrl: new URL('/', SITE_ORIGIN),
      source: `dist/search-index.json entry ${indexPosition + 1}`
    };
  });
}

function extractToolCatalogReferences(assert) {
  const source = read('js/accounts/tools-account-ui.js');
  const start = source.indexOf('const TOOL_CATALOG = {');
  const end = source.indexOf('\n  };', start);
  assert(start !== -1 && end !== -1, 'tools-account-ui.js should expose the TOOL_CATALOG literal');
  const catalogSource = source.slice(start, end);
  const references = [];
  const entryCount = (catalogSource.match(/^\s{4}["'][^"']+["']\s*:\s*\{/gm) || []).length;
  const pattern = /["']([^"']+)["']\s*:\s*\{[^{}]*?\bhref\s*:\s*["']([^"']+)["']/g;
  let match;
  while ((match = pattern.exec(catalogSource))) {
    references.push({
      target: match[2],
      baseUrl: new URL('/', SITE_ORIGIN),
      source: `TOOL_CATALOG reopen href for ${match[1]}`
    });
  }
  assert(references.length > 0, 'TOOL_CATALOG should expose at least one reopen href');
  assert(references.length === entryCount,
    'Every TOOL_CATALOG entry should expose a statically auditable reopen href');
  return references;
}

function extractPublishedLiveDemoReferences(assert) {
  const references = [];
  walkFiles('content/projects', '.json').forEach((filePath) => {
    const relativePath = toPosix(path.relative(ROOT, filePath));
    const project = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    if (!project || project.published === false) return;
    (Array.isArray(project.resources) ? project.resources : []).forEach((resource) => {
      if (String(resource && resource.label || '').trim().toLowerCase() !== 'live demo') return;
      assert(typeof resource.url === 'string' && resource.url.trim(),
        `${relativePath} Live Demo resource should contain a URL`);
      references.push({
        target: resource.url,
        baseUrl: new URL('/', SITE_ORIGIN),
        source: `${relativePath} Live Demo`
      });
    });
  });
  assert(references.length > 0, 'Published projects should expose at least one Live Demo resource');
  return references;
}

function hasNonPersonalRealmEscape(url) {
  return [...url.searchParams.keys()].some((key) => {
    if (!REALM_QUERY_KEYS.has(String(key).toLowerCase())) return false;
    return url.searchParams.getAll(key).some((value) => String(value).trim().toLowerCase() !== 'personal');
  });
}

function isExemptTarget(rawTarget, url) {
  const raw = String(rawTarget || '').trim();
  if (/^(?:mailto|tel):/i.test(raw)) return true;
  if (/^(?:data|blob|javascript):/i.test(raw)) return true;
  if (url.origin !== SITE_ORIGIN) return true;

  const pathname = url.pathname.toLowerCase();
  if (pathname === '/robots.txt' || pathname === '/sitemap.xml') return true;
  if (/^\/(?:api)(?:\/|$)/i.test(pathname)) return true;
  if (/^\/(?:documents)(?:\/|$)/i.test(pathname)) return true;
  if (
    /^\/(?:assets|css|data|dist|fonts|img|js|models|vendor)(?:\/|$)/i.test(pathname) &&
    !/\.html?$/i.test(pathname)
  ) return true;
  return /\.(?:avif|bmp|css|csv|docx?|gif|ico|jpe?g|json|map|md|mp3|mp4|pdf|png|svg|txt|webm|webp|woff2?|xlsx?|xml|zip)$/i.test(pathname);
}

function normalizeReference(reference, assert) {
  const rawTarget = String(reference.target || '').trim();
  if (!rawTarget) return null;
  if (/^(?:mailto|tel|data|blob|javascript):/i.test(rawTarget)) return null;

  let url;
  try {
    url = new URL(rawTarget, reference.baseUrl || new URL('/', SITE_ORIGIN));
  } catch (_) {
    assert(false, `${reference.source} contains an invalid navigation target: ${rawTarget}`);
    return null;
  }
  if (url.origin !== SITE_ORIGIN) return null;
  assert(!hasNonPersonalRealmEscape(url),
    `${reference.source} must not escape the personal theme with a non-personal audience/mode value: ${url.href}`);
  if (isExemptTarget(rawTarget, url)) return null;
  return { ...reference, url };
}

function matchRoutePattern(pattern, pathname) {
  const patternParts = String(pattern || '').split('/').filter(Boolean);
  const pathParts = String(pathname || '').split('/').filter(Boolean);
  const params = {};
  let pathIndex = 0;

  for (let patternIndex = 0; patternIndex < patternParts.length; patternIndex += 1) {
    const part = patternParts[patternIndex];
    const dynamic = /^:([A-Za-z0-9_]+)(\*)?(.*)$/.exec(part);
    if (!dynamic) {
      if (pathParts[pathIndex] !== part) return null;
      pathIndex += 1;
      continue;
    }

    const [, name, wildcard, suffix] = dynamic;
    if (wildcard) {
      let value = pathParts.slice(pathIndex).join('/');
      if (suffix) {
        if (!value.endsWith(suffix)) return null;
        value = value.slice(0, -suffix.length);
      }
      params[name] = value;
      pathIndex = pathParts.length;
      continue;
    }

    const valuePart = pathParts[pathIndex];
    if (valuePart == null || (suffix && !valuePart.endsWith(suffix))) return null;
    params[name] = suffix ? valuePart.slice(0, -suffix.length) : valuePart;
    pathIndex += 1;
  }

  return pathIndex === pathParts.length ? params : null;
}

function substituteRouteParams(destination, params) {
  return String(destination || '').replace(/:([A-Za-z0-9_]+)\*?/g, (_, name) => (
    Object.prototype.hasOwnProperty.call(params, name) ? params[name] : ''
  ));
}

function ruleConditionsMatch(rule, url) {
  if (Array.isArray(rule.missing) && rule.missing.length) return false;
  if (!Array.isArray(rule.has) || !rule.has.length) return true;
  return rule.has.every((condition) => {
    if (!condition || condition.type !== 'query') return false;
    if (!url.searchParams.has(condition.key)) return false;
    if (!condition.value) return true;
    try {
      return new RegExp(`^(?:${condition.value})$`).test(url.searchParams.get(condition.key) || '');
    } catch (_) {
      return false;
    }
  });
}

function matchingRule(rules, url) {
  for (const rule of Array.isArray(rules) ? rules : []) {
    if (!ruleConditionsMatch(rule, url)) continue;
    const params = matchRoutePattern(rule.source, url.pathname);
    if (params) return { rule, params };
  }
  return null;
}

function sourceFileForInternalPath(pathname) {
  const normalized = decodeURIComponent(String(pathname || '/')).replace(/^\/+/, '');
  if (!normalized) return exists('index.html') ? 'index.html' : '';
  const candidates = [];
  if (normalized.toLowerCase().endsWith('.html')) candidates.push(normalized);
  else {
    candidates.push(`${normalized}.html`);
    candidates.push(path.join(normalized, 'index.html'));
  }
  return candidates.map(toPosix).find((candidate) => exists(candidate)) || '';
}

function resolveHtmlDestination(initialUrl, vercel, assert, source) {
  let publicUrl = new URL(initialUrl.href);
  for (let redirects = 0; redirects < 8; redirects += 1) {
    const match = matchingRule(vercel.redirects, publicUrl);
    if (!match) break;
    const destination = substituteRouteParams(match.rule.destination, match.params);
    publicUrl = new URL(destination, publicUrl);
    assert(publicUrl.origin === SITE_ORIGIN,
      `${source} should not redirect an internal destination off-origin: ${publicUrl.href}`);
    assert(!hasNonPersonalRealmEscape(publicUrl),
      `${source} should not redirect through a non-personal audience/mode value: ${publicUrl.href}`);
  }

  const rewriteMatch = matchingRule(vercel.rewrites, publicUrl);
  const storageUrl = rewriteMatch
    ? new URL(substituteRouteParams(rewriteMatch.rule.destination, rewriteMatch.params), SITE_ORIGIN)
    : publicUrl;
  assert(storageUrl.origin === SITE_ORIGIN,
    `${source} should resolve to same-origin storage: ${storageUrl.href}`);
  assert(!hasNonPersonalRealmEscape(storageUrl),
    `${source} should not rewrite through a non-personal audience/mode value: ${storageUrl.href}`);

  return {
    publicUrl,
    storageUrl,
    sourceFile: sourceFileForInternalPath(storageUrl.pathname),
    rewrite: rewriteMatch ? rewriteMatch.rule : null
  };
}

function bodyTagForHtml(html) {
  return /<body\b[^>]*>/i.exec(String(html || ''))?.[0] || '';
}

function assertHomepageContract(html, sourceFile, assert) {
  const body = bodyTagForHtml(html);
  assert(/\bdata-page=["']home["']/i.test(body) && /\bdata-audience=["']personal["']/i.test(body),
    `${sourceFile} should remain the personal homepage`);
  assert(/(?:^|\s)home-pattern-page(?:\s|$)/i.test(tagAttribute(body, 'class')),
    `${sourceFile} should retain the new personal homepage theme`);
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function assertCompactPersonalShell(html, sourceFile, publicUrl, themedStylesheet, assert) {
  const body = bodyTagForHtml(html);
  const shellTags = html.match(/<(?:section|main|div)\b[^>]*\bdata-personal-accordion-shell(?:\s|=|>)[^>]*>/gi) || [];
  assert(shellTags.length === 1,
    `${publicUrl.pathname} resolves to ${sourceFile}, which should contain exactly one personal accordion shell`);
  assert(/(?:^|\s)personal-accordion-page(?:\s|$)/i.test(tagAttribute(body, 'class')),
    `${sourceFile} should carry the personal-accordion-page body class`);
  assert(tagAttribute(body, 'data-personal-chrome') === 'compact',
    `${sourceFile} should use compact personal chrome`);
  assert(tagAttribute(body, 'data-audience') === 'personal',
    `${sourceFile} should remain in the personal audience`);

  const category = tagAttribute(body, 'data-personal-category');
  assert(['about', 'projects', 'tools', 'games', 'contact'].includes(category),
    `${sourceFile} should declare a recognized personal category`);
  assert(tagAttribute(shellTags[0] || '', 'data-personal-active-category') === category,
    `${sourceFile} shell should identify the body category as active`);

  const activeRailTags = (html.match(/<[a-z][^>]*\bdata-personal-rail-active=["']true["'][^>]*>/gi) || []);
  assert(activeRailTags.length === 1,
    `${sourceFile} should contain exactly one active personal rail marker`);
  assert(new RegExp(`(?:^|\\s)personal-accordion__rail--${escapeRegExp(category)}(?:\\s|$)`, 'i')
    .test(tagAttribute(activeRailTags[0] || '', 'class')),
  `${sourceFile} active rail should match its ${category} category`);

  const categoryMarkerTags = (html.match(/<[a-z][^>]*\bdata-personal-category-marker=["'][^"']+["'][^>]*>/gi) || []);
  assert(categoryMarkerTags.length === 1 &&
    tagAttribute(categoryMarkerTags[0] || '', 'data-personal-category-marker') === category,
  `${sourceFile} should contain exactly one category identity marker for ${category}`);

  const backTags = (html.match(/<a\b[^>]*>/gi) || []).filter((tag) => (
    /(?:^|\s)personal-accordion__back(?:\s|$)/i.test(tagAttribute(tag, 'class'))
  ));
  assert(backTags.length === 1 && tagAttribute(backTags[0], 'href'),
    `${sourceFile} should contain exactly one linked personal accordion back control`);

  const compactFooters = (html.match(/<footer\b[^>]*>/gi) || []).filter((tag) => (
    /(?:^|\s)footer--personal-compact(?:\s|$)/i.test(tagAttribute(tag, 'class'))
  ));
  assert(compactFooters.length === 1,
    `${sourceFile} should contain exactly one compact personal footer`);

  const themedStylesheets = (html.match(/<link\b[^>]*>/gi) || []).filter((tag) => {
    const rel = tagAttribute(tag, 'rel').toLowerCase().split(/\s+/);
    if (!rel.includes('stylesheet')) return false;
    try {
      return new URL(tagAttribute(tag, 'href'), documentBaseUrl(html)).pathname === `/${themedStylesheet}`;
    } catch (_) {
      return false;
    }
  });
  assert(themedStylesheets.length === 1,
    `${sourceFile} should load the built personal accordion stylesheet exactly once`);
}

function assertPersonalDestination(reference, vercel, themedStylesheet, assert) {
  const resolved = resolveHtmlDestination(reference.url, vercel, assert, reference.source);
  assert(resolved.sourceFile,
    `${reference.source} points to ${reference.url.pathname}, which does not resolve to a source HTML file`);
  const html = read(resolved.sourceFile);
  if (resolved.publicUrl.pathname === '/') {
    assertHomepageContract(html, resolved.sourceFile, assert);
  } else {
    assertCompactPersonalShell(html, resolved.sourceFile, resolved.publicUrl, themedStylesheet, assert);
  }
  assert(!toPosix(resolved.sourceFile).startsWith('pages/professional/'),
    `${reference.source} should not resolve into a professional audience snapshot`);
  return resolved;
}

function canonicalLiveDemoReferences(liveDemoReferences, assert) {
  const canonicalDemos = new Map();
  liveDemoReferences.forEach((reference) => {
    const normalized = normalizeReference(reference, assert);
    if (!normalized) return;
    const canonicalUrl = new URL(normalized.url.href);
    canonicalUrl.pathname = canonicalUrl.pathname.replace(/\.html$/i, '');
    canonicalDemos.set(`${canonicalUrl.pathname}${canonicalUrl.search}`, {
      target: canonicalUrl.href,
      baseUrl: new URL('/', SITE_ORIGIN),
      source: `${reference.source} canonical destination`
    });
  });
  assert(canonicalDemos.size > 0, 'At least one published Live Demo should use the canonical site origin');
  return [...canonicalDemos.values()];
}

function runPersonalThemeContinuityTests({ assert }) {
  const vercel = readJson('vercel.json');
  const stylesManifest = readJson('dist/styles-manifest.json');
  assert(Array.isArray(vercel.redirects) && Array.isArray(vercel.rewrites),
    'vercel.json should expose redirects and rewrites for continuity resolution');
  assert(typeof stylesManifest.personalAccordionFile === 'string' && stylesManifest.personalAccordionFile,
    'dist/styles-manifest.json should identify the built personal accordion stylesheet');
  const themedStylesheet = `dist/${stylesManifest.personalAccordionFile}`;

  assert(!hasNonPersonalRealmEscape(new URL('/?audience=personal&mode=personal', SITE_ORIGIN)),
    'Explicit personal audience/mode query values should remain valid');
  assert(hasNonPersonalRealmEscape(new URL('/?audience=professional', SITE_ORIGIN)) &&
    hasNonPersonalRealmEscape(new URL('/?mode=professional', SITE_ORIGIN)),
  'Non-personal audience/mode query values should be rejected');

  const homepageReferences = extractHomepageReferences();
  const managedShellReferences = extractManagedShellEntryReferences();
  const searchReferences = extractSearchIndexReferences(assert);
  const toolCatalogReferences = extractToolCatalogReferences(assert);
  const liveDemoReferences = extractPublishedLiveDemoReferences(assert);
  const canonicalDemoReferences = canonicalLiveDemoReferences(liveDemoReferences, assert);
  const pending = [
    {
      target: '/',
      baseUrl: new URL('/', SITE_ORIGIN),
      source: 'personal homepage entry'
    },
    ...homepageReferences,
    ...managedShellReferences,
    ...searchReferences,
    ...toolCatalogReferences,
    ...liveDemoReferences,
    ...canonicalDemoReferences
  ];

  const visited = new Set();
  let destinationCount = 0;
  while (pending.length) {
    const reference = pending.shift();
    const normalized = normalizeReference(reference, assert);
    if (!normalized) continue;
    const key = `${normalized.url.pathname}${normalized.url.search}`;
    if (visited.has(key)) continue;
    visited.add(key);

    const resolved = assertPersonalDestination(normalized, vercel, themedStylesheet, assert);
    const html = read(resolved.sourceFile);
    const baseUrl = documentBaseUrl(html);
    extractNavigationTags(html).forEach(({ target }) => {
      pending.push({
        target,
        baseUrl,
        source: `${resolved.sourceFile} reachable navigation`
      });
    });
    destinationCount += 1;
  }
  assert(destinationCount > 0,
    'Personal navigation discovery should find same-origin HTML destinations');

  canonicalDemoReferences.forEach((reference) => {
    const normalized = normalizeReference(reference, assert);
    assert(normalized && visited.has(`${normalized.url.pathname}${normalized.url.search}`),
      `${reference.source} should be checked by the themed destination crawl`);
  });

  const stellar = resolveHtmlDestination(
    new URL('/games/stellar-dogfight', SITE_ORIGIN),
    vercel,
    assert,
    'Stellar Dogfight continuity contract'
  );
  assert(stellar.sourceFile, 'Stellar Dogfight should resolve to an HTML source file');
  const stellarBody = bodyTagForHtml(read(stellar.sourceFile));
  assert(tagAttribute(stellarBody, 'data-personal-fit') === 'immersive',
    'Stellar Dogfight should retain immersive fit');
  assert(tagAttribute(stellarBody, 'data-personal-chrome') === 'compact',
    'Stellar Dogfight should retain compact personal chrome even with immersive fit');
}

module.exports = runPersonalThemeContinuityTests;

if (require.main === module) {
  runPersonalThemeContinuityTests({
    assert(condition, message) {
      if (!condition) throw new Error(message);
    }
  });
  process.stdout.write('Personal theme continuity tests passed.\n');
}
