#!/usr/bin/env node
'use strict';

/*
  Generate deterministic, JS-free page digests for AI retrieval agents.

  The output is intentionally static and reviewable:
  - llms.txt
  - dist/ai-digest-manifest.json
  - dist/ai-pages/<canonical-route>.html
*/

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'dist', 'ai-pages');
const manifestPath = path.join(root, 'dist', 'ai-digest-manifest.json');
const llmsPath = path.join(root, 'llms.txt');
const SITE_ORIGIN = 'https://www.danielshort.me';
const MAX_SOURCE_CHARS = 8000;
const MAX_SUMMARY_CHARS = 520;
const MAX_FACTS = 10;
const MAX_EVIDENCE = 8;
const MAX_BODY_POINTS = 6;
const MAX_LINKS = 14;

const excludedPathPatterns = [
  /^\/admin(?:\/|$)/i,
  /^\/api(?:\/|$)/i,
  /^\/ai(?:\/|$)/i,
  /^\/pages(?:\/|$)/i,
  /^\/search$/i,
  /^\/sitemap-pretty$/i,
  /^\/(?:analytics|data-science|tourism|destination-analytics|contributions)$/i,
  /^\/(?:resume|resume-pdf)$/i,
  /^\/resume(?:-[a-z-]+)?(?:-pdf)?$/i,
  /^\/tools\/(?:dashboard|short-links|ga4-utm-performance|job-application-tracker|transcribe|whisper-transcribe-monitor)$/i
];

const scoreTerms = [
  'analytics',
  'automation',
  'dashboard',
  'forecast',
  'insight',
  'kpi',
  'python',
  'reporting',
  'sql',
  'tableau',
  'workflow'
];

function sleepSync(ms) {
  const waitMs = Math.max(0, Number(ms) || 0);
  if (!waitMs) return;
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, waitMs);
}

function removeWithRetries(target) {
  const transientCodes = new Set(['EBUSY', 'ENOTEMPTY', 'EPERM']);
  let lastError = null;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      fs.rmSync(target, {
        recursive: true,
        force: true,
        maxRetries: 3,
        retryDelay: 100
      });
      return;
    } catch (err) {
      lastError = err;
      if (!transientCodes.has(err && err.code)) throw err;
      sleepSync(120 * (attempt + 1));
    }
  }
  throw lastError;
}

function ensureCleanDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
  fs.readdirSync(dirPath).forEach((entry) => {
    removeWithRetries(path.join(dirPath, entry));
  });
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function decodeHtml(value) {
  return String(value ?? '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => {
      const code = Number.parseInt(hex, 16);
      if (!Number.isFinite(code)) return '';
      try { return String.fromCodePoint(code); } catch { return ''; }
    })
    .replace(/&#(\d+);/g, (_, dec) => {
      const code = Number.parseInt(dec, 10);
      if (!Number.isFinite(code)) return '';
      try { return String.fromCodePoint(code); } catch { return ''; }
    });
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeWhitespace(value) {
  return decodeHtml(String(value ?? '')
    .replace(/\u00a0/g, ' ')
    .replace(/\s+/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function cleanText(value) {
  return normalizeWhitespace(String(value || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ')
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<[^>]+>/g, ' '));
}

function uniqueList(values, maxItems) {
  const seen = new Set();
  const out = [];
  (Array.isArray(values) ? values : []).forEach((value) => {
    const text = normalizeWhitespace(value);
    if (!text) return;
    const key = text.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push(text);
  });
  return Number.isFinite(maxItems) ? out.slice(0, maxItems) : out;
}

function normalizeTextArray(value) {
  if (!value) return [];
  if (Array.isArray(value)) return uniqueList(value);
  const text = normalizeWhitespace(value);
  return text ? [text] : [];
}

function trimToSentence(value, maxChars) {
  const text = normalizeWhitespace(value);
  if (!text || text.length <= maxChars) return text;
  const sliced = text.slice(0, maxChars);
  const sentenceEnd = Math.max(sliced.lastIndexOf('. '), sliced.lastIndexOf('! '), sliced.lastIndexOf('? '));
  if (sentenceEnd > 120) return sliced.slice(0, sentenceEnd + 1).trim();
  return sliced.replace(/\s+\S*$/, '').trim();
}

function stripSiteSuffix(title) {
  return normalizeWhitespace(title)
    .replace(/\s*\|\s*Daniel Short\s*$/i, '')
    .replace(/\s*-\s*Daniel Short\s*$/i, '')
    .trim();
}

function extractTitle(html) {
  const match = /<title>([^<]+)<\/title>/i.exec(String(html || ''));
  return stripSiteSuffix(match ? match[1] : '');
}

function extractMeta(html, attr, key) {
  const safeAttr = String(attr || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const safeKey = String(key || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const re = new RegExp(`<meta\\b[^>]*\\b${safeAttr}="${safeKey}"[^>]*\\bcontent="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(String(html || ''));
  return normalizeWhitespace(match ? match[1] : '');
}

function extractDescription(html) {
  return extractMeta(html, 'name', 'description') || extractMeta(html, 'property', 'og:description');
}

function extractCanonical(html) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return normalizeWhitespace(match ? match[1] : '');
}

function extractRobots(html) {
  const match = /<meta\s+[^>]*name="robots"[^>]*content="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return normalizeWhitespace(match ? match[1] : '');
}

function isNoindex(html) {
  return extractRobots(html).toLowerCase().includes('noindex');
}

function walkFiles(dirPath, predicate) {
  if (!fs.existsSync(dirPath)) return [];
  const files = [];
  const stack = [dirPath];
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
      if (entry.isFile() && (!predicate || predicate(full))) files.push(full);
    });
  }
  return files.sort();
}

function listRootHtmlFiles() {
  return fs.readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.html'))
    .map((entry) => path.join(root, entry.name))
    .sort();
}

function relFromRoot(absPath) {
  return path.relative(root, absPath).replace(/\\/g, '/');
}

function readJsonFile(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function readJsonRel(relPath) {
  return readJsonFile(path.join(root, relPath));
}

function loadJsonRecords(relDir) {
  const dirPath = path.join(root, relDir);
  return walkFiles(dirPath, (filePath) => filePath.endsWith('.json'))
    .map((filePath) => ({
      absPath: filePath,
      relPath: relFromRoot(filePath),
      data: readJsonFile(filePath)
    }))
    .filter((record) => record.data && typeof record.data === 'object');
}

function slugifyId(value) {
  const slug = normalizeWhitespace(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || 'section';
}

function textHash(value) {
  return sourceHash(String(value || ''));
}

function routeToAiUrl(urlPath) {
  const normalized = normalizePathname(urlPath);
  if (!normalized || normalized === '/') return `${SITE_ORIGIN}/ai/index`;
  return `${SITE_ORIGIN}/ai${normalized}`;
}

function isPublicVisibility(value) {
  const visibility = normalizeWhitespace(value).toLowerCase();
  return !visibility || visibility === 'public';
}

function loadPublicToolSlugs() {
  const slugs = new Set();
  walkFiles(path.join(root, 'content', 'tools'), (filePath) => filePath.endsWith('.json')).forEach((filePath) => {
    try {
      const tool = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const slug = normalizeWhitespace(tool && tool.slug);
      const visibility = normalizeWhitespace(tool && tool.visibility).toLowerCase();
      if (!slug || tool.hidden || tool.noindex || !isPublicVisibility(visibility)) return;
      slugs.add(slug);
    } catch {}
  });
  return slugs;
}

function toPathFromCanonical(canonical) {
  const raw = normalizeWhitespace(canonical);
  if (!raw) return '';
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizePathname(url.pathname || '/');
  } catch {
    return '';
  }
}

function toPathFromRelFile(relPath, publicToolSlugs) {
  const safe = String(relPath || '').replace(/\\/g, '/');
  if (safe === 'index.html') return '/';
  if (safe.startsWith('pages/portfolio/') && safe.endsWith('.html')) {
    const id = safe.replace(/^pages\/portfolio\//, '').replace(/\.html$/, '');
    return id ? `/portfolio/${encodeURIComponent(id)}` : '';
  }
  if (safe.startsWith('pages/') && safe.endsWith('.html')) {
    const slug = safe.replace(/^pages\//, '').replace(/\.html$/, '');
    if (!slug) return '';
    if (publicToolSlugs && publicToolSlugs.has(slug)) return `/tools/${slug}`;
    return `/${slug}`;
  }
  if (safe.endsWith('.html') && !safe.includes('/')) {
    const slug = safe.replace(/\.html$/, '');
    return slug ? `/${slug}` : '';
  }
  return '';
}

function routeCategory(urlPath) {
  if (urlPath === '/portfolio' || urlPath.startsWith('/portfolio/')) return 'Portfolio';
  if (urlPath === '/tools' || urlPath.startsWith('/tools/')) return 'Tools';
  if (urlPath === '/games' || urlPath.startsWith('/games/')) return 'Games';
  if (['/', '/contact'].includes(urlPath)) return 'Core';
  return 'Page';
}

function shouldExcludeUrl(urlPath, html, noindexPathnames, override) {
  const normalized = normalizePathname(urlPath);
  if (!normalized) return true;
  if (override && override.exclude === true) return true;
  if (isNoindex(html)) return true;
  if (noindexPathnames.has(normalized)) return true;
  return excludedPathPatterns.some((pattern) => pattern.test(normalized));
}

function stripIndexNoise(html) {
  return String(html || '')
    .replace(/<article\b[^>]*\bdata-tools-visibility="[^"]+"[^>]*>[\s\S]*?<\/article>/gi, ' ')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ');
}

function extractMainRegion(html) {
  const stripped = stripIndexNoise(html);
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(stripped);
  let region = mainMatch ? mainMatch[1] : stripped;
  region = region
    .replace(/<header\b[^>]*>[\s\S]*?<\/header>/gi, ' ')
    .replace(/<nav\b[^>]*>[\s\S]*?<\/nav>/gi, ' ')
    .replace(/<footer\b[^>]*>[\s\S]*?<\/footer>/gi, ' ')
    .replace(/<form\b[^>]*>[\s\S]*?<\/form>/gi, ' ')
    .replace(/<button\b[^>]*>[\s\S]*?<\/button>/gi, ' ')
    .replace(/<template\b[^>]*>[\s\S]*?<\/template>/gi, ' ');
  return region;
}

function extractCleanMainText(html) {
  return cleanText(extractMainRegion(html)).slice(0, MAX_SOURCE_CHARS);
}

function extractTagTexts(region, tagNames) {
  const tags = Array.isArray(tagNames) ? tagNames.join('|') : String(tagNames || '');
  if (!tags) return [];
  const re = new RegExp(`<(?:${tags})\\b[^>]*>([\\s\\S]*?)<\\/(?:${tags})>`, 'gi');
  const values = [];
  let match;
  while ((match = re.exec(String(region || '')))) {
    const text = cleanText(match[1]);
    if (text) values.push(text);
  }
  return values;
}

function extractKeywords(html) {
  const keywords = [];
  const re = /<span\b[^>]*\bclass="[^"]*\b(?:project-tag|tool-pill|resume-chip)\b[^"]*"[^>]*>([\s\S]*?)<\/span>/gi;
  let match;
  while ((match = re.exec(String(html || '')))) {
    const text = cleanText(match[1]);
    if (text) keywords.push(text);
  }
  return uniqueList(keywords, 30);
}

function parseAttributes(raw) {
  const attrs = {};
  const re = /([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"([^"]*)"/g;
  let match;
  while ((match = re.exec(String(raw || '')))) {
    attrs[String(match[1]).toLowerCase()] = decodeHtml(match[2]);
  }
  return attrs;
}

function normalizeHref(href) {
  const raw = normalizeWhitespace(href);
  if (!raw || raw.startsWith('#') || /^javascript:/i.test(raw)) return '';
  if (/^(?:mailto:|tel:)/i.test(raw)) return raw;
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin === SITE_ORIGIN) {
      const pathname = normalizePathname(url.pathname || '/');
      if (/^\/(?:pages|api|ai)(?:\/|$)/i.test(pathname)) return '';
      return `${SITE_ORIGIN}${pathname}${url.search || ''}${url.hash || ''}`;
    }
    return url.toString();
  } catch {
    return '';
  }
}

function extractLinks(region) {
  const links = [];
  const re = /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = re.exec(String(region || '')))) {
    const attrs = parseAttributes(match[1]);
    const href = normalizeHref(attrs.href || '');
    const label = cleanText(match[2]);
    if (!href || !label) continue;
    if (label.length > 120) continue;
    links.push({ label, url: href });
  }

  const seen = new Set();
  return links.filter((link) => {
    const key = `${link.label.toLowerCase()}|${link.url.toLowerCase()}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  }).slice(0, MAX_LINKS);
}

function splitSentences(text) {
  return normalizeWhitespace(text)
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/g)
    .map((part) => part.trim())
    .filter(Boolean);
}

function isNoiseText(text) {
  const value = normalizeWhitespace(text);
  if (!value) return true;
  if (value.length > 220) return true;
  if (/\b(?:Resume Portfolio Contact|View credential Certification|Send a Message|Clear form|Required|Scroll for)\b/i.test(value)) return true;
  if ((value.match(/\b(?:Resume|Portfolio|Contact|Certification|Project Examples|Work Experience)\b/g) || []).length >= 3) return true;
  return false;
}

function hasEvidenceSignal(text) {
  return /(?:\b\d[\d,.]*\b|%|\$|\b(?:hours?|annually|gpa|certif|degree|reduced|improved|saved|cut|built|created|modeled|validated|deployed|forecast|dashboard|sql|python|tableau)\b)/i.test(String(text || ''));
}

function scoreFact(text) {
  const value = String(text || '');
  let score = 0;
  if (/\d/.test(value)) score += 4;
  if (/%|\$|\+/.test(value)) score += 2;
  if (/\b(?:built|created|modeled|validated|deployed|reduced|improved|saved|cut|identified|supported|translated|designed)\b/i.test(value)) score += 3;
  scoreTerms.forEach((term) => {
    if (new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(value)) score += 1;
  });
  if (value.length < 35 || value.length > 280) score -= 3;
  return score;
}

function selectFacts({ override, listItems, paragraphs, sentences }) {
  const candidates = [
    ...normalizeTextArray(override && override.facts),
    ...listItems.filter(hasEvidenceSignal),
    ...sentences.filter(hasEvidenceSignal),
    ...paragraphs.filter(hasEvidenceSignal)
  ].filter((text) => !isNoiseText(text));
  const ranked = uniqueList(candidates)
    .map((text, index) => ({ text, index, score: scoreFact(text) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map((item) => item.text);
  return uniqueList(ranked, MAX_FACTS);
}

function selectEvidence({ override, facts, listItems, sentences }) {
  const candidates = [
    ...normalizeTextArray(override && override.evidence),
    ...facts,
    ...listItems,
    ...sentences
  ].filter((text) => /(?:\d|%|\$|\+)/.test(String(text || '')) && !isNoiseText(text));
  return uniqueList(candidates, MAX_EVIDENCE);
}

function selectBodyPoints({ override, paragraphs, sentences, facts }) {
  const factSet = new Set(facts.map((value) => value.toLowerCase()));
  const body = [
    ...normalizeTextArray(override && override.body),
    ...paragraphs,
    ...sentences
  ].filter((text) => {
    if (isNoiseText(text)) return false;
    const normalized = text.toLowerCase();
    if (factSet.has(normalized)) return false;
    return text.length >= 45 && text.length <= 260;
  });
  return uniqueList(body, MAX_BODY_POINTS);
}

function sourceHash(html) {
  return crypto.createHash('sha256').update(String(html || '')).digest('hex').slice(0, 16);
}

function routeToOutputRel(urlPath) {
  const normalized = normalizePathname(urlPath);
  const withoutLeading = normalized.replace(/^\/+/, '') || 'index';
  const safeParts = withoutLeading.split('/').map((part) => {
    let decoded = part;
    try { decoded = decodeURIComponent(part); } catch {}
    return decoded.replace(/[^A-Za-z0-9._-]/g, '-').replace(/^-+|-+$/g, '') || 'page';
  });
  return `${safeParts.join('/')}.html`;
}

function renderList(items) {
  if (!items || !items.length) return '';
  return [
    '<ul>',
    ...items.map((item) => `  <li>${escapeHtml(item)}</li>`),
    '</ul>'
  ].join('\n');
}

function renderLinks(links) {
  if (!links || !links.length) return '';
  return [
    '<ul>',
    ...links.map((link) => {
      const description = normalizeWhitespace(link.description || '');
      const suffix = description ? `: ${escapeHtml(description)}` : '';
      return `  <li><a href="${escapeHtml(link.url)}">${escapeHtml(link.label)}</a>${suffix}</li>`;
    }),
    '</ul>'
  ].join('\n');
}

function normalizeSection(section) {
  if (!section || typeof section !== 'object') return null;
  const title = normalizeWhitespace(section.title || section.heading || '');
  if (!title) return null;
  const paragraphs = uniqueList(normalizeTextArray(section.paragraphs || section.text || section.summary), 6);
  const items = uniqueList(normalizeTextArray(section.items || section.facts || section.bullets), 16);
  const links = normalizeStructuredLinks(section.links || []);
  const level = Math.min(6, Math.max(2, Number(section.level) || 2));
  if (!paragraphs.length && !items.length && !links.length) return null;
  return { title, paragraphs, items, links, level };
}

function renderSection(section, index) {
  const normalized = normalizeSection(section);
  if (!normalized) return '';
  const id = `${slugifyId(normalized.title)}-${index + 1}`;
  const headingTag = `h${normalized.level}`;
  const parts = [
    `<section aria-labelledby="${escapeHtml(id)}">`,
    `  <${headingTag} id="${escapeHtml(id)}">${escapeHtml(normalized.title)}</${headingTag}>`
  ];
  normalized.paragraphs.forEach((paragraph) => {
    parts.push(`  <p>${escapeHtml(paragraph)}</p>`);
  });
  if (normalized.items.length) parts.push(renderList(normalized.items));
  if (normalized.links.length) parts.push(renderLinks(normalized.links));
  parts.push('</section>');
  return parts.join('\n');
}

function renderDigest(page) {
  const introParagraphs = uniqueList(normalizeTextArray(page.introParagraphs || []), 6)
    .map((paragraph) => `      <p>${escapeHtml(paragraph)}</p>`)
    .join('\n');
  const sections = (page.sections || [])
    .map(renderSection)
    .filter(Boolean);

  return `<!DOCTYPE html>
<html lang="en" data-ai-digest="true">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(page.title)} | AI Digest</title>
  <link rel="canonical" href="${escapeHtml(page.canonicalUrl)}">
  <meta name="description" content="${escapeHtml(page.description || page.summary || page.title)}">
  <meta name="generator" content="Daniel Short deterministic AI digest">
  <meta name="source-path" content="${escapeHtml(page.sourcePath)}">
  <meta name="source-hash" content="${escapeHtml(page.sourceHash)}">
</head>
<body>
  <main id="main" data-ai-digest="true" data-canonical-url="${escapeHtml(page.canonicalUrl)}">
    <article>
      <h1>${escapeHtml(page.title)}</h1>
${introParagraphs}
${sections.join('\n')}
    </article>
  </main>
</body>
</html>
`;
}

function coerceAiDigest(raw) {
  if (!raw || typeof raw !== 'object') return null;
  return {
    ...(raw.exclude === true ? { exclude: true } : {}),
    summary: normalizeWhitespace(raw.summary || ''),
    facts: normalizeTextArray(raw.facts),
    evidence: normalizeTextArray(raw.evidence),
    body: normalizeTextArray(raw.body),
    links: Array.isArray(raw.links) ? raw.links : []
  };
}

function mergeOverrides(base, next) {
  if (!next) return base;
  const current = base || {};
  return {
    exclude: current.exclude === true || next.exclude === true,
    summary: next.summary || current.summary || '',
    facts: uniqueList([...(current.facts || []), ...(next.facts || [])]),
    evidence: uniqueList([...(current.evidence || []), ...(next.evidence || [])]),
    body: uniqueList([...(current.body || []), ...(next.body || [])]),
    links: [...(current.links || []), ...(next.links || [])]
  };
}

function addOverride(overrides, route, raw) {
  const normalized = normalizePathname(route);
  const override = coerceAiDigest(raw);
  if (!normalized || !override) return;
  overrides.set(normalized, mergeOverrides(overrides.get(normalized), override));
}

function loadAiDigestOverrides() {
  const overrides = new Map();
  walkFiles(path.join(root, 'content'), (filePath) => filePath.endsWith('.json')).forEach((filePath) => {
    let parsed;
    try {
      parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch {
      return;
    }

    const relPath = relFromRoot(filePath);
    const dirname = path.dirname(relPath).replace(/\\/g, '/');
    const basename = path.basename(relPath, '.json');
    const routes = [];

    if (parsed && parsed.canonicalPath) routes.push(parsed.canonicalPath);
    if (parsed && parsed.page && parsed.page.canonicalPath) routes.push(parsed.page.canonicalPath);
    if (parsed && parsed.digitalPage && parsed.digitalPage.canonicalPath) routes.push(parsed.digitalPage.canonicalPath);

    if (dirname === 'content/tools') {
      const slug = normalizeWhitespace(parsed.slug || basename);
      if (slug) routes.push(`/tools/${slug}`);
    }
    if (dirname === 'content/projects') {
      const id = normalizeWhitespace(parsed.id || basename);
      if (id) routes.push(`/portfolio/${id}`);
    }
    if (dirname === 'content/pages' && parsed.id) {
      routes.push(`/${normalizeWhitespace(parsed.id)}`);
    }

    routes.forEach((route) => {
      addOverride(overrides, route, parsed.aiDigest);
      if (parsed.page) addOverride(overrides, route, parsed.page.aiDigest);
      if (parsed.digitalPage) addOverride(overrides, route, parsed.digitalPage.aiDigest);
    });
  });
  return overrides;
}

function normalizeOverrideLinks(links) {
  if (!Array.isArray(links)) return [];
  return links.map((link) => {
    if (!link || typeof link !== 'object') return null;
    const label = normalizeWhitespace(link.label || link.title || '');
    const url = normalizeHref(link.url || link.href || '');
    if (!label || !url) return null;
    return { label, url };
  }).filter(Boolean);
}

function normalizeStructuredLinks(links) {
  if (!Array.isArray(links)) return [];
  return links.map((link) => {
    if (!link || typeof link !== 'object') return null;
    const label = normalizeWhitespace(link.label || link.title || '');
    const url = normalizeHref(link.url || link.href || '');
    const description = normalizeWhitespace(link.description || link.summary || '');
    if (!label || !url) return null;
    return { label, url, description };
  }).filter(Boolean);
}

function getClassBlocks(html, tagName, className) {
  const safeTag = String(tagName || '').replace(/[^A-Za-z0-9-]/g, '');
  const safeClass = String(className || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  if (!safeTag || !safeClass) return [];
  const re = new RegExp(`<${safeTag}\\b[^>]*\\bclass="[^"]*\\b${safeClass}\\b[^"]*"[^>]*>([\\s\\S]*?)<\\/${safeTag}>`, 'gi');
  const blocks = [];
  let match;
  while ((match = re.exec(String(html || '')))) blocks.push(match[1]);
  return blocks;
}

function getFirstTagText(html, tagName, className) {
  const blocks = getClassBlocks(html, tagName, className);
  return blocks.length ? cleanText(blocks[0]) : '';
}

function extractHtmlLinks(html, maxItems = MAX_LINKS) {
  return normalizeStructuredLinks(extractLinks(html).slice(0, maxItems));
}

function projectSummary(project) {
  const subtitle = normalizeWhitespace(project && project.subtitle);
  const problem = normalizeWhitespace(project && project.problem);
  const summary = [subtitle, problem].filter(Boolean).join(': ');
  return trimToSentence(summary || normalizeWhitespace(project && project.notes), MAX_SUMMARY_CHARS);
}

function readableLabel(value) {
  return normalizeWhitespace(value)
    .replace(/,(\S)/g, ', $1');
}

function pageSectionLabels(page) {
  return ((page && page.sections) || [])
    .map((section) => readableLabel(section && section.label))
    .filter((label) => label && !/^(?:Jump to section|Show jump menu)$/i.test(label));
}

function findPageSectionLabel(page, patterns, fallback) {
  const labels = pageSectionLabels(page);
  const regexes = (Array.isArray(patterns) ? patterns : [patterns]).filter(Boolean);
  return labels.find((label) => regexes.some((pattern) => pattern.test(label))) || fallback;
}

function createStructuredPage(urlPath, fields) {
  const normalizedUrl = normalizePathname(urlPath);
  if (!normalizedUrl) return null;
  const sourceText = fields.sourceText || '';
  return {
    url: normalizedUrl,
    title: normalizeWhitespace(fields.title),
    description: normalizeWhitespace(fields.description || fields.summary),
    summary: trimToSentence(fields.summary || fields.description, MAX_SUMMARY_CHARS),
    category: normalizeWhitespace(fields.category || routeCategory(normalizedUrl)),
    sourcePath: normalizeWhitespace(fields.sourcePath || ''),
    sourceHash: sourceText ? textHash(sourceText) : '',
    sections: (fields.sections || []).map(normalizeSection).filter(Boolean),
    introParagraphs: normalizeTextArray(fields.introParagraphs || []),
    keywords: uniqueList(fields.keywords || [], 30),
    links: normalizeStructuredLinks(fields.links || [])
  };
}

function buildProjectStructuredPage(record) {
  const project = record.data;
  const id = normalizeWhitespace(project.id || path.basename(record.relPath, '.json'));
  if (!id || project.hidden || project.noindex) return null;
  const urlPath = `/portfolio/${id}`;
  const resources = normalizeStructuredLinks(project.resources || []);
  const links = resources.length ? resources : [];
  return createStructuredPage(urlPath, {
    title: project.title,
    description: projectSummary(project),
    summary: projectSummary(project),
    category: 'Portfolio',
    sourcePath: record.relPath,
    sourceText: JSON.stringify(project),
    keywords: [...(project.tools || []), ...(project.concepts || []), ...(project.audiences || [])],
    links,
    sections: [
      {
        title: 'STAR Summary',
        items: uniqueList([
          project.problem ? `Situation: ${project.problem}` : '',
          ...normalizeTextArray(project.role).map((item) => `Task: ${item}`),
          ...normalizeTextArray(project.actions).map((item) => `Action: ${item}`),
          ...normalizeTextArray(project.results).map((item) => `Result: ${item}`)
        ].filter(Boolean), 16)
      },
      { title: 'Notes', paragraphs: [project.notes].filter(Boolean) },
      { title: 'Links', links }
    ]
  });
}

function extractResumeDetails(record) {
  const resume = record.data || {};
  const page = resume.digitalPage || {};
  const html = (page.sections || [])
    .map((section) => section && section.props && section.props.html ? section.props.html : '')
    .join('\n');
  const summary = getFirstTagText(html, 'p', 'resume-summary') || page.description || '';

  const skills = getClassBlocks(html, 'div', 'resume-skill-group')
    .map((block) => {
      const label = cleanText((/<h3\b[^>]*>([\s\S]*?)<\/h3>/i.exec(block) || [])[1] || '');
      const items = extractTagTexts(block, ['li']).filter(Boolean);
      return label && items.length ? `${label}: ${items.join(', ')}` : '';
    })
    .filter(Boolean);

  const experience = getClassBlocks(html, 'article', 'resume-role')
    .map((block) => {
      const title = cleanText((/<h3\b[^>]*>([\s\S]*?)<\/h3>/i.exec(block) || [])[1] || '');
      const company = getFirstTagText(block, 'p', 'resume-role-company');
      const dates = getFirstTagText(block, 'p', 'resume-role-dates');
      const bullets = extractTagTexts(block, ['li']).filter(Boolean);
      const heading = [title, company, dates].filter(Boolean).join(' - ');
      return { heading, bullets };
    })
    .filter((role) => role.heading || role.bullets.length);

  const education = getClassBlocks(html, 'a', 'resume-education-item')
    .map(cleanText)
    .filter(Boolean);

  const selectedProjects = getClassBlocks(html, 'li', 'resume-project')
    .map((block) => {
      const title = cleanText((/<a\b[^>]*>([\s\S]*?)<\/a>/i.exec(block) || [])[1] || '');
      const meta = getFirstTagText(block, 'p', 'resume-project-meta');
      return [title, meta].filter(Boolean).join(': ');
    })
    .filter(Boolean);

  const links = extractHtmlLinks(html, 10)
    .filter((link) => !/^tel:/i.test(link.url));

  return { summary, skills, experience, education, selectedProjects, links, html };
}

function buildResumeStructuredPage(record) {
  const resume = record.data || {};
  const page = resume.digitalPage || {};
  const urlPath = normalizeWhitespace(page.canonicalPath);
  if (!urlPath || page.robots && String(page.robots).toLowerCase().includes('noindex')) return null;
  const details = extractResumeDetails(record);
  const experienceItems = [];
  details.experience.forEach((role) => {
    if (role.heading) experienceItems.push(role.heading);
    role.bullets.forEach((bullet) => experienceItems.push(bullet));
  });
  return createStructuredPage(urlPath, {
    title: stripSiteSuffix(page.title || `${resume.audience || resume.key || 'Resume'} Resume`),
    description: page.description,
    summary: details.summary || page.description,
    category: 'Resume',
    sourcePath: record.relPath,
    sourceText: JSON.stringify(record.data),
    keywords: [resume.audience, resume.key, 'resume', 'experience', 'skills'],
    links: details.links,
    sections: [
      { title: 'Summary', paragraphs: [details.summary || page.description].filter(Boolean) },
      { title: 'Skills', items: details.skills },
      { title: 'Experience', items: experienceItems },
      { title: 'Education', items: details.education },
      { title: 'Selected Projects', items: details.selectedProjects },
      { title: 'Links', links: details.links }
    ]
  });
}

function extractProofItemsFromAudience(audience) {
  const html = ((audience.page && audience.page.sections) || [])
    .map((section) => section && section.props && section.props.html ? section.props.html : '')
    .join('\n');
  const items = [];
  const re = /<span\b[^>]*\bclass="[^"]*\bhome-proof-value\b[^"]*"[^>]*>([\s\S]*?)<\/span>\s*<span\b[^>]*\bclass="[^"]*\bhome-proof-label\b[^"]*"[^>]*>([\s\S]*?)<\/span>/gi;
  let match;
  while ((match = re.exec(html))) {
    const value = cleanText(match[1]);
    const label = cleanText(match[2]);
    if (value && label) items.push(`${value} ${label}`);
  }
  return uniqueList(items, 8);
}

function buildAudienceStructuredPage(record, projectsById, resumeDetailsByKey) {
  const audience = record.data || {};
  const page = audience.page || {};
  const urlPath = normalizeWhitespace(page.canonicalPath || audience.homePath);
  if (!urlPath) return null;
  const key = normalizeWhitespace(audience.key || path.basename(record.relPath, '.json'));
  const resumeDetails = resumeDetailsByKey.get(key);
  const pageHtml = (page.sections || [])
    .map((section) => section && section.props && section.props.html ? section.props.html : '')
    .join('\n');
  const heroLabel = pageSectionLabels(page)[0] || stripSiteSuffix(page.title || audience.label);
  const heroTagline = getFirstTagText(pageHtml, 'p', 'hero-tagline');
  const heroStatus = getFirstTagText(pageHtml, 'p', 'hero-status');
  const resultsLabel = findPageSectionLabel(page, [/Results/i, /Impact/i], 'Results');
  const projectsLabel = findPageSectionLabel(page, [/Project Examples/i], 'Project Examples');
  const experienceLabel = findPageSectionLabel(page, [/Work Experience/i], 'Work Experience');
  const skillsLabel = findPageSectionLabel(page, [/Skills in Practice/i], 'Skills in Practice');
  const credentialsLabel = findPageSectionLabel(page, [/Certifications.*Degrees/i], 'Certifications & Degrees');
  const contactLabel = findPageSectionLabel(page, [/Open to/i, /Send a Message/i], 'Send a Message');
  const featuredProjects = (audience.featuredProjectIds || [])
    .map((id) => projectsById.get(id))
    .filter(Boolean);
  const featuredItems = featuredProjects.map((project) => {
    return [project.title, project.subtitle].filter(Boolean).join(': ');
  });
  const experienceItems = resumeDetails
    ? resumeDetails.experience.slice(0, 3).map((role) => {
      const bullet = role.bullets && role.bullets[0] ? ` - ${role.bullets[0]}` : '';
      return `${role.heading}${bullet}`;
    })
    : [];
  const links = [
    { label: audience.resumeNavTitle || `${audience.label} Resume`, url: audience.resumePath },
    { label: audience.portfolioTitle || `${audience.label} Portfolio`, url: audience.portfolioPath },
    { label: 'Contact Daniel Short', url: '/contact' }
  ];
  return createStructuredPage(urlPath, {
    title: stripSiteSuffix(page.title || audience.label),
    description: page.description,
    summary: page.description,
    category: 'Core',
    sourcePath: record.relPath,
    sourceText: JSON.stringify(record.data),
    keywords: [audience.label, audience.shortLabel, 'hiring', 'portfolio', 'resume'],
    links,
    sections: [
      { title: heroLabel, paragraphs: [heroTagline, heroStatus].filter(Boolean) },
      { title: resultsLabel, items: extractProofItemsFromAudience(audience) },
      { title: projectsLabel, items: featuredItems },
      { title: experienceLabel, items: experienceItems },
      { title: skillsLabel, items: resumeDetails ? resumeDetails.skills : [] },
      { title: credentialsLabel, items: resumeDetails ? resumeDetails.education : [] },
      { title: contactLabel, links }
    ]
  });
}

function buildToolsDirectoryStructuredPage(pageRecord, toolRecords) {
  const page = pageRecord && pageRecord.data;
  if (!page) return null;
  const categories = Array.isArray(page.categories) ? page.categories : [];
  const publicTools = toolRecords
    .map((record) => record.data)
    .filter((tool) => tool && !tool.hidden && !tool.noindex && isPublicVisibility(tool.visibility))
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));
  const sections = [
    { title: 'Summary', paragraphs: [page.description].filter(Boolean) }
  ];
  categories.forEach((category) => {
    const tools = publicTools.filter((tool) => tool.categoryId === category.id);
    if (!tools.length) return;
    sections.push({
      title: category.title,
      paragraphs: [category.description].filter(Boolean),
      links: tools.map((tool) => ({
        label: tool.title,
        url: tool.href || `/tools/${tool.slug}`,
        description: tool.summary
      }))
    });
  });
  return createStructuredPage(page.canonicalPath, {
    title: stripSiteSuffix(page.title),
    description: page.description,
    summary: page.description,
    category: 'Tools',
    sourcePath: pageRecord.relPath,
    sourceText: JSON.stringify(page),
    keywords: ['tools', 'utilities', 'privacy-first'],
    links: publicTools.map((tool) => ({ label: tool.title, url: tool.href || `/tools/${tool.slug}`, description: tool.summary })),
    sections
  });
}

function buildGamesDirectoryStructuredPage(pageRecord) {
  const page = pageRecord && pageRecord.data;
  if (!page) return null;
  const games = (Array.isArray(page.games) ? page.games : [])
    .filter((game) => game && !game.hidden && !game.noindex && (game.href || game.id))
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));
  const links = games.map((game) => ({
    label: game.title,
    url: game.href || `/games/${game.id}`,
    description: game.summary
  }));
  return createStructuredPage(page.canonicalPath, {
    title: stripSiteSuffix(page.title),
    description: page.description,
    summary: page.heroLead || page.description,
    category: 'Games',
    sourcePath: pageRecord.relPath,
    sourceText: JSON.stringify(page),
    keywords: ['games', 'simulations', ...games.flatMap((game) => Array.isArray(game.tags) ? game.tags : [])],
    links,
    sections: [
      { title: page.heroTitle || 'Games', paragraphs: [page.heroLead || page.description].filter(Boolean), links }
    ]
  });
}

function buildToolStructuredPage(record, categoriesById) {
  const tool = record.data || {};
  const slug = normalizeWhitespace(tool.slug || path.basename(record.relPath, '.json'));
  if (!slug || tool.hidden || tool.noindex || !isPublicVisibility(tool.visibility)) return null;
  const pills = (tool.pills || []).map((pill) => normalizeWhitespace(pill && pill.label)).filter(Boolean);
  const category = categoriesById.get(tool.categoryId);
  const isLocal = pills.some((pill) => /^local$/i.test(pill)) || /\b(?:locally|local|no uploads)\b/i.test(tool.summary || '');
  const privacy = isLocal
    ? 'Designed for local, browser-based use where supported; avoid sending sensitive inputs unless the tool explicitly states it uses a backend.'
    : 'May use an account, API, or server-side compute depending on the tool workflow.';
  const links = [{ label: tool.title, url: tool.href || `/tools/${slug}`, description: tool.summary }];
  return createStructuredPage(`/tools/${slug}`, {
    title: tool.title,
    description: tool.summary,
    summary: tool.summary,
    category: 'Tools',
    sourcePath: record.relPath,
    sourceText: JSON.stringify(tool),
    keywords: [...pills, category && category.title].filter(Boolean),
    links,
    sections: [
      { title: 'Summary', paragraphs: [tool.summary].filter(Boolean) },
      { title: 'What It Does', items: [tool.summary].filter(Boolean) },
      { title: 'Privacy And Runtime', items: [privacy] },
      { title: 'Use Cases', items: pills.filter((pill) => !/^local$/i.test(pill)) },
      { title: 'Links', links }
    ]
  });
}

function buildPortfolioStructuredPage(projectRecords) {
  const projects = projectRecords
    .map((record) => record.data)
    .filter((project) => project && project.id && !project.hidden && !project.noindex)
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));
  const links = projects.map((project) => ({
    label: project.title,
    url: `/portfolio/${project.id}`,
    description: projectSummary(project)
  }));
  return createStructuredPage('/portfolio', {
    title: 'Portfolio',
    description: 'Project library of data projects, software experiments, tools, and demos by Daniel Short.',
    summary: 'Project library of data projects, software experiments, tools, and demos by Daniel Short.',
    category: 'Portfolio',
    sourcePath: 'content/projects/*.json',
    sourceText: JSON.stringify(projects),
    keywords: ['portfolio', 'projects', 'tools', 'experiments', 'data'],
    links,
    sections: [
      { title: 'Summary', paragraphs: ['Project library of data projects, software experiments, tools, and demos by Daniel Short.'] },
      { title: 'Featured Projects', links: links.slice(0, 12) }
    ]
  });
}

function loadStructuredPages() {
  const structured = new Map();
  const projectRecords = loadJsonRecords('content/projects');
  const toolRecords = loadJsonRecords('content/tools');
  const toolsPageRecord = {
    relPath: 'content/pages/tools.json',
    data: readJsonRel('content/pages/tools.json')
  };
  const gamesPageRecord = {
    relPath: 'content/pages/games.json',
    data: readJsonRel('content/pages/games.json')
  };
  const categoriesById = new Map(((toolsPageRecord.data && toolsPageRecord.data.categories) || [])
    .map((category) => [category.id, category]));
  projectRecords.forEach((record) => {
    const page = buildProjectStructuredPage(record);
    if (page) structured.set(page.url, page);
  });

  const portfolioPage = buildPortfolioStructuredPage(projectRecords);
  if (portfolioPage) structured.set(portfolioPage.url, portfolioPage);

  const toolsPage = buildToolsDirectoryStructuredPage(toolsPageRecord, toolRecords);
  if (toolsPage) structured.set(toolsPage.url, toolsPage);

  const gamesPage = buildGamesDirectoryStructuredPage(gamesPageRecord);
  if (gamesPage) structured.set(gamesPage.url, gamesPage);

  toolRecords.forEach((record) => {
    const page = buildToolStructuredPage(record, categoriesById);
    if (page) structured.set(page.url, page);
  });

  return structured;
}

function applyStructuredPage(basePage, structuredPage) {
  if (!structuredPage) return basePage;
  return {
    ...basePage,
    title: structuredPage.title || basePage.title,
    description: structuredPage.description || basePage.description,
    summary: structuredPage.summary || basePage.summary,
    category: structuredPage.category || basePage.category,
    sourcePath: structuredPage.sourcePath || basePage.sourcePath,
    sourceHash: structuredPage.sourceHash || basePage.sourceHash,
    sections: basePage.sections && basePage.sections.length ? basePage.sections : structuredPage.sections,
    introParagraphs: basePage.introParagraphs && basePage.introParagraphs.length ? basePage.introParagraphs : structuredPage.introParagraphs,
    keywords: uniqueList([...(structuredPage.keywords || []), ...(basePage.keywords || [])], 30),
    links: normalizeStructuredLinks([...(structuredPage.links || []), ...(basePage.links || [])])
  };
}

function extractSectionsFromRegion(region) {
  const sections = [];
  const introParagraphs = [];
  let current = null;
  const pushCurrent = () => {
    const normalized = normalizeSection(current);
    if (normalized) sections.push(normalized);
    current = null;
  };
  const tokenRe = /<(h[1-6]|p|li|a)\b([^>]*)>([\s\S]*?)<\/\1>/gi;
  let match;
  while ((match = tokenRe.exec(String(region || '')))) {
    const tag = String(match[1] || '').toLowerCase();
    const text = cleanText(match[3]);
    if (!text || isNoiseText(text)) continue;

    if (/^h[1-6]$/.test(tag)) {
      if (tag === 'h1') continue;
      pushCurrent();
      current = {
        title: text,
        paragraphs: [],
        items: [],
        links: [],
        level: Math.min(6, Math.max(2, Number(tag.slice(1)) || 2))
      };
      continue;
    }

    if (tag === 'p') {
      if (text.length < 20) continue;
      if (current) current.paragraphs.push(text);
      else introParagraphs.push(text);
      continue;
    }

    if (tag === 'li') {
      if (!current || text.length < 12 || text.length > 320) continue;
      current.items.push(text);
      continue;
    }

    if (tag === 'a' && current) {
      const attrs = parseAttributes(match[2]);
      const href = normalizeHref(attrs.href || '');
      if (!href || text.length > 140) continue;
      current.links.push({ label: text, url: href });
    }
  }
  pushCurrent();
  return {
    introParagraphs: uniqueList(introParagraphs, 6),
    sections: sections.map(normalizeSection).filter(Boolean)
  };
}

function buildDigestPage({ html, relPath, urlPath, override, generatedAt }) {
  const region = extractMainRegion(html);
  const mainText = extractCleanMainText(html);
  if (!mainText || mainText.length < 80) return null;

  const headings = uniqueList(extractTagTexts(region, ['h1', 'h2', 'h3']), 14);
  const listItems = uniqueList(extractTagTexts(region, ['li']), 80).filter((item) => item.length >= 20 && item.length <= 280);
  const paragraphs = uniqueList(extractTagTexts(region, ['p']), 80).filter((item) => item.length >= 35 && item.length <= 340);
  const sentences = splitSentences(mainText).filter((item) => item.length >= 35 && item.length <= 300);
  const extractedSections = extractSectionsFromRegion(region);

  const title = extractTitle(html) || headings[0] || urlPath;
  const description = extractDescription(html);
  const summary = trimToSentence(
    override && override.summary
      ? override.summary
      : description || paragraphs[0] || sentences.slice(0, 2).join(' '),
    MAX_SUMMARY_CHARS
  );
  const facts = selectFacts({ override, listItems, paragraphs, sentences });
  const evidence = selectEvidence({ override, facts, listItems, sentences });
  const bodyPoints = selectBodyPoints({ override, paragraphs, sentences, facts });
  const links = [
    ...normalizeOverrideLinks(override && override.links),
    ...extractLinks(region)
  ];
  const dedupedLinks = [];
  const seenLinks = new Set();
  links.forEach((link) => {
    const key = `${link.label.toLowerCase()}|${link.url.toLowerCase()}`;
    if (seenLinks.has(key)) return;
    seenLinks.add(key);
    dedupedLinks.push(link);
  });

  const keywords = uniqueList([
    ...extractKeywords(html),
    ...headings.filter((heading) => heading.length <= 80)
  ], 30);
  const canonicalUrl = `${SITE_ORIGIN}${urlPath}`;
  const aiUrl = routeToAiUrl(urlPath);
  const outputRelPath = routeToOutputRel(urlPath);

  return {
    url: urlPath,
    title,
    description,
    summary,
    category: routeCategory(urlPath),
    sourcePath: relPath,
    sourceHash: sourceHash(html),
    canonicalUrl,
    aiUrl,
    outputPath: `dist/ai-pages/${outputRelPath}`,
    outputRelPath,
    generatedAt,
    facts,
    evidence,
    bodyPoints,
    introParagraphs: extractedSections.introParagraphs.length
      ? extractedSections.introParagraphs
      : paragraphs.slice(0, 4),
    sections: extractedSections.sections,
    keywords,
    links: dedupedLinks.slice(0, MAX_LINKS)
  };
}

function buildDigests() {
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const publicToolSlugs = loadPublicToolSlugs();
  const overrides = loadAiDigestOverrides();
  const structuredPages = loadStructuredPages();
  const candidates = [
    ...listRootHtmlFiles(),
    ...walkFiles(path.join(root, 'pages'), (filePath) => filePath.endsWith('.html')),
    ...walkFiles(path.join(root, 'demos'), (filePath) => filePath.endsWith('.html'))
  ];
  const generatedAt = new Date().toISOString();
  const pagesByUrl = new Map();

  candidates.forEach((absPath) => {
    const relPath = relFromRoot(absPath);
    if (!relPath || relPath.startsWith('public/') || relPath.startsWith('node_modules/')) return;

    let html = '';
    try {
      html = fs.readFileSync(absPath, 'utf8');
    } catch {
      return;
    }

    const urlPath = toPathFromCanonical(extractCanonical(html)) || toPathFromRelFile(relPath, publicToolSlugs);
    const normalizedUrl = normalizePathname(urlPath);
    const override = overrides.get(normalizedUrl);
    if (shouldExcludeUrl(normalizedUrl, html, noindexPathnames, override)) return;

    let page = buildDigestPage({ html, relPath, urlPath: normalizedUrl, override, generatedAt });
    if (!page) return;
    page = applyStructuredPage(page, structuredPages.get(normalizedUrl));

    const previous = pagesByUrl.get(normalizedUrl);
    const previousScore = previous ? previous.facts.length + previous.evidence.length + previous.bodyPoints.length : -1;
    const nextScore = page.facts.length + page.evidence.length + page.bodyPoints.length;
    if (!previous || nextScore >= previousScore) pagesByUrl.set(normalizedUrl, page);
  });

  return [...pagesByUrl.values()].sort((a, b) => a.url.localeCompare(b.url));
}

function escapeMarkdown(value) {
  return normalizeWhitespace(value)
    .replace(/\\/g, '\\\\')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]');
}

function trimLlmsDescription(value, maxChars = 240) {
  const text = normalizeWhitespace(value);
  const finish = (description) => {
    const cleaned = normalizeWhitespace(description).replace(/\b(?:with|and|or|the|a|an|to|of|for|that|could|would|should)$/i, '').trim();
    return cleaned && /[.!?)]$/.test(cleaned) ? cleaned : `${cleaned.replace(/[,:;]+$/, '')}.`;
  };
  if (!text) return '';
  if (text.length <= maxChars) return finish(text);
  const sliced = text.slice(0, maxChars);
  const sentenceEnd = Math.max(sliced.lastIndexOf('. '), sliced.lastIndexOf('! '), sliced.lastIndexOf('? '));
  if (sentenceEnd > 80) return sliced.slice(0, sentenceEnd + 1).trim();
  const trimmed = sliced.replace(/\s+\S*$/, '').replace(/[,:;]+$/, '').trim();
  return trimmed ? finish(trimmed) : '';
}

function llmsLine(page) {
  const label = escapeMarkdown(page.title || page.url);
  const url = page.aiUrl || routeToAiUrl(page.url);
  const description = trimLlmsDescription(page.summary || page.description || '');
  return `- [${label}](${url})${description ? `: ${description}` : ''}`;
}

function llmsSection(title, pages) {
  const lines = uniqueList((pages || []).filter(Boolean).map(llmsLine));
  if (!lines.length) return '';
  return [`## ${title}`, '', ...lines].join('\n');
}

function renderLlmsTxt(pages) {
  const byUrl = new Map((pages || []).map((page) => [page.url, page]));
  const pick = (urls) => urls.map((url) => byUrl.get(url)).filter(Boolean);
  const portfolioUrls = [
    '/portfolio',
    '/portfolio/retailStore',
    '/portfolio/targetEmptyPackage',
    '/portfolio/pizzaDashboard',
    '/portfolio/deliveryTip',
    '/portfolio/ufoDashboard',
    '/portfolio/chatbotLora',
    '/portfolio/smartSentence'
  ];
  const toolUrls = [
    '/tools',
    '/tools/text-compare',
    '/tools/utm-batch-builder',
    '/tools/word-frequency',
    '/tools/qr-code-generator',
    '/tools/image-optimizer'
  ];
  const gameUrls = [
    '/games',
    '/games/stellar-dogfight',
    '/games/roulette',
    '/games/probability-engine',
    '/games/project-starfall',
    '/games/ocean-wave-simulation'
  ];
  const optionalPages = [
    ...pick(['/privacy', '/sitemap'])
  ];
  const sections = [
    llmsSection('Start Here', pick(['/', '/portfolio', '/tools', '/games', '/contact'])),
    llmsSection('Projects', pick(portfolioUrls)),
    llmsSection('Tools', pick(toolUrls)),
    llmsSection('Games', pick(gameUrls)),
    llmsSection('Optional', optionalPages)
  ].filter(Boolean);

  return [
    '# Daniel Short',
    '',
    '> Personal website for Daniel Short. This file prioritizes stable AI-readable entry points for projects, tools, experiments, and contact information.',
    '',
    'Canonical site: https://www.danielshort.me/',
    'AI-readable pages use stable /ai/ URLs and canonicalize back to their public human-facing pages.',
    '',
    sections.join('\n\n'),
    ''
  ].join('\n');
}

function writeOutputs(pages) {
  ensureCleanDir(outDir);

  pages.forEach((page) => {
    const target = path.join(outDir, page.outputRelPath);
    ensureDir(path.dirname(target));
    fs.writeFileSync(target, renderDigest(page), 'utf8');
  });

  const routes = {};
  pages.forEach((page) => {
    routes[page.url] = {
      title: page.title,
      outputPath: page.outputPath,
      canonicalUrl: page.canonicalUrl,
      aiUrl: page.aiUrl,
      sourcePath: page.sourcePath,
      sourceHash: page.sourceHash
    };
  });

  const manifest = {
    version: 1,
    generatedAt: pages[0] ? pages[0].generatedAt : new Date().toISOString(),
    origin: SITE_ORIGIN,
    pages: pages.map((page) => ({
      url: page.url,
      title: page.title,
      description: page.description,
      category: page.category,
      canonicalUrl: page.canonicalUrl,
      aiUrl: page.aiUrl,
      outputPath: page.outputPath,
      sourcePath: page.sourcePath,
      sourceHash: page.sourceHash,
      summary: page.summary,
      keywords: page.keywords
    })),
    routes
  };

  ensureDir(path.dirname(manifestPath));
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  fs.writeFileSync(llmsPath, renderLlmsTxt(pages), 'utf8');
}

function main() {
  const pages = buildDigests();
  writeOutputs(pages);
  process.stdout.write(`[ai-digests] Wrote llms.txt, dist/ai-digest-manifest.json, and ${pages.length} AI page digest(s)\n`);
}

main();
