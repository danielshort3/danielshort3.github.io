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
let nonpublicCatalogRoutesCache = null;

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
    .replace(/<\/?(?:a|abbr|b|cite|code|em|i|mark|small|span|strong|sub|sup|time)\b[^>]*>/gi, '')
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

function fallbackMainTextForRoute(urlPath) {
  const normalized = normalizePathname(urlPath);
  if (normalized !== '/') return '';
  return [
    'Daniel Short personal website with portfolio projects, browser tools, games, and contact information.',
    'The homepage introduces Daniel Short and links to featured projects, browser tools, games, and contact information.'
  ].join(' ');
}

function isPublicVisibility(value) {
  const visibility = normalizeWhitespace(value).toLowerCase();
  return !visibility || visibility === 'public';
}

function isPublicProject(project) {
  return Boolean(project && project.id && project.published !== false && !project.hidden &&
    !project.noindex && isPublicVisibility(project.visibility));
}

function isPublicTool(tool) {
  return Boolean(tool && tool.slug && tool.published !== false && !tool.hidden &&
    !tool.noindex && isPublicVisibility(tool.visibility));
}

function loadPublicToolSlugs() {
  const slugs = new Set();
  walkFiles(path.join(root, 'content', 'tools'), (filePath) => filePath.endsWith('.json')).forEach((filePath) => {
    try {
      const tool = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const slug = normalizeWhitespace(tool && tool.slug);
      const visibility = normalizeWhitespace(tool && tool.visibility).toLowerCase();
      if (!slug || !isPublicTool({ ...tool, slug, visibility })) return;
      slugs.add(slug);
    } catch {}
  });
  return slugs;
}

function loadNonpublicCatalogRoutes() {
  if (nonpublicCatalogRoutesCache) return nonpublicCatalogRoutesCache;
  const routes = new Set();
  loadJsonRecords('content/projects').forEach((record) => {
    const project = record.data;
    const id = normalizeWhitespace(project.id || path.basename(record.relPath, '.json'));
    if (id && !isPublicProject({ ...project, id })) routes.add(`/portfolio/${id}`);
  });
  loadJsonRecords('content/tools').forEach((record) => {
    const tool = record.data;
    const slug = normalizeWhitespace(tool.slug || path.basename(record.relPath, '.json'));
    if (slug && !isPublicTool({ ...tool, slug })) routes.add(`/tools/${slug}`);
  });
  nonpublicCatalogRoutesCache = routes;
  return routes;
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
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ');
}

function pruneHiddenMarkup(html) {
  const source = String(html || '');
  const tokens = /<!--[\s\S]*?-->|<![^>]*>|<\/?[A-Za-z][^>]*>/g;
  const voidTags = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);
  const stack = [];
  const output = [];
  let cursor = 0;
  let match;
  while ((match = tokens.exec(source))) {
    if (!stack.some((entry) => entry.hidden)) output.push(source.slice(cursor, match.index));
    const token = match[0];
    cursor = tokens.lastIndex;
    if (/^<!/.test(token)) continue;
    const tagMatch = /^<\/?([A-Za-z][\w:-]*)\b/.exec(token);
    if (!tagMatch) continue;
    const tag = tagMatch[1].toLowerCase();
    if (/^<\//.test(token)) {
      const index = stack.findLastIndex((entry) => entry.tag === tag);
      if (index < 0) continue;
      const hidden = stack[index].hidden;
      stack.splice(index);
      if (!hidden && !stack.some((entry) => entry.hidden)) output.push(token);
      continue;
    }
    const attributes = token.slice(tagMatch[0].length, -1);
    const attrs = parseAttributes(attributes);
    const hidden = stack.some((entry) => entry.hidden) ||
      /(?:^|\s)(?:hidden|inert)(?:\s|=|$)/i.test(attributes) ||
      /^(?:true|1)$/i.test(attrs['aria-hidden'] || '') ||
      /^(?:status|alert)$/i.test(attrs.role || '') ||
      /^(?:polite|assertive)$/i.test(attrs['aria-live'] || '') ||
      /(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden)(?:\s*[;!]|$)/i.test(attrs.style || '') ||
      /\bsitemap-(?:counts|generated)\b/i.test(attrs.class || '') ||
      Boolean(attrs['data-tools-visibility'] && !isPublicVisibility(attrs['data-tools-visibility']));
    if (!hidden) output.push(token);
    if (!voidTags.has(tag) && !/\/\s*>$/.test(token)) stack.push({ tag, hidden });
  }
  if (!stack.some((entry) => entry.hidden)) output.push(source.slice(cursor));
  return output.join('');
}

function extractMainRegion(html) {
  const stripped = stripIndexNoise(html);
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(stripped);
  let region = pruneHiddenMarkup(mainMatch ? mainMatch[1] : stripped);
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
  if (/^(?:mailto:|tel:)/i.test(raw)) return raw.split(/[?#]/, 1)[0];
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return '';
    if (url.origin === SITE_ORIGIN) {
      const pathname = normalizePathname(url.pathname || '/');
      if (/^\/(?:pages|api|ai|admin|professional)(?:\/|$)/i.test(pathname)) return '';
      if (excludedPathPatterns.some((pattern) => pattern.test(pathname)) ||
        loadNonpublicCatalogRoutes().has(pathname)) return '';
      return `${SITE_ORIGIN}${pathname}`;
    }
    return `${url.origin}${url.pathname}`;
  } catch {
    return '';
  }
}

function linkLabel(html) {
  const heading = /<h[2-6]\b[^>]*>([\s\S]*?)<\/h[2-6]>/i.exec(String(html || ''));
  return cleanText(heading ? heading[1] : html);
}

function extractLinks(region) {
  const links = [];
  const re = /<a\b([^>]*)>([\s\S]*?)<\/a>/gi;
  let match;
  while ((match = re.exec(String(region || '')))) {
    const attrs = parseAttributes(match[1]);
    const href = normalizeHref(attrs.href || '');
    const label = linkLabel(match[2]);
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
  if (value.length > 900) return true;
  if (/^(?:Resume Portfolio Contact|View credential Certification|Send a Message|Clear form|Required|Scroll for)[.!?]?$/i.test(value)) return true;
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
  const items = uniqueList(normalizeTextArray(section.items || section.facts || section.bullets));
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
  <meta name="robots" content="noindex, follow">
  <meta name="generator" content="Daniel Short deterministic AI digest">
  <meta name="source-path" content="${escapeHtml(page.sourcePath)}">
  <meta name="source-hash" content="${escapeHtml(page.sourceHash)}">
</head>
<body>
  <main id="main" data-ai-digest="true" data-canonical-url="${escapeHtml(page.canonicalUrl)}">
    <article>
      <h1>${escapeHtml(page.title)}</h1>
      <p><a href="${escapeHtml(page.canonicalUrl)}">View the full page</a></p>
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
  return trimToSentence(
    project && (project.metaDescription || project.subtitle || project.problem || project.notes),
    MAX_SUMMARY_CHARS
  );
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
  if (!isPublicProject({ ...project, id })) return null;
  const urlPath = `/portfolio/${id}`;
  const resources = normalizeStructuredLinks(project.resources || []);
  const links = resources.length ? resources : [{ label: 'View project', url: urlPath }];
  const demo = project.embed || {};
  const instructions = project.demoInstructions || {};
  const demoSection = normalizeSection({
    title: demo.heading || 'Project Preview',
    paragraphs: [demo.description, instructions.lead].filter(Boolean),
    items: normalizeTextArray(instructions.bullets)
  });
  return createStructuredPage(urlPath, {
    title: project.title,
    description: project.metaDescription || projectSummary(project),
    summary: projectSummary(project),
    category: 'Portfolio',
    sourcePath: record.relPath,
    sourceText: JSON.stringify(project),
    introParagraphs: [project.subtitle].filter(Boolean),
    keywords: [...(project.tools || []), ...(project.concepts || []), ...(project.audiences || [])],
    links,
    sections: [
      demoSection,
      {
        title: 'STAR Summary',
        items: [
          project.problem ? `Situation: ${project.problem}` : '',
          project.task ? `Task: ${project.task}` : '',
          ...normalizeTextArray(project.actions).map((item) => `Action: ${item}`),
          ...normalizeTextArray(project.results).map((item) => `Result: ${item}`)
        ].filter(Boolean)
      },
      { title: 'Additional Context', paragraphs: [project.notes].filter(Boolean) },
      { title: 'Links', links }
    ].filter(Boolean)
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
    .filter(isPublicTool)
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
    sourceText: JSON.stringify({ page, tools: publicTools }),
    keywords: ['tools', 'utilities', 'privacy-first'],
    links: publicTools.map((tool) => ({ label: tool.title, url: tool.href || `/tools/${tool.slug}`, description: tool.summary })),
    sections
  });
}

function buildGamesDirectoryStructuredPage(pageRecord) {
  const page = pageRecord && pageRecord.data;
  if (!page) return null;
  const games = (Array.isArray(page.games) ? page.games : [])
    .filter((game) => game && game.published !== false && !game.hidden && !game.noindex &&
      isPublicVisibility(game.visibility) && (game.href || game.id))
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
  if (!isPublicTool({ ...tool, slug })) return null;
  const pills = (tool.pills || []).map((pill) => normalizeWhitespace(pill && pill.label)).filter(Boolean);
  const category = categoriesById.get(tool.categoryId);
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
      { title: 'What It Does', paragraphs: [tool.summary].filter(Boolean) },
      { title: 'Inputs', items: normalizeTextArray(tool.inputs) },
      { title: 'Outputs', items: normalizeTextArray(tool.outputs) },
      { title: 'Privacy and runtime', paragraphs: [tool.privacy].filter(Boolean) },
      { title: 'Links', links }
    ]
  });
}

function buildPortfolioStructuredPage(projectRecords) {
  const projects = projectRecords
    .map((record) => record.data)
    .filter(isPublicProject)
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
      { title: 'Projects', links }
    ]
  });
}

function buildGameStructuredPage(game, pageRecord) {
  if (!game || game.published === false || game.hidden || game.noindex ||
    !isPublicVisibility(game.visibility) || !(game.href || game.id)) return null;
  const rawPath = String(game.href || `games/${game.id}`).trim();
  const urlPath = `/${rawPath.replace(/^\/+/, '').replace(/\.html$/i, '')}`;
  const title = normalizeWhitespace(game.title || game.id || 'Browser game');
  const summary = normalizeWhitespace(game.summary || '');
  if (!title || !summary) return null;
  const tags = normalizeTextArray(game.tags);
  const links = [{ label: title, url: urlPath, description: summary }];
  return createStructuredPage(urlPath, {
    title,
    description: summary,
    summary,
    category: 'Games',
    sourcePath: pageRecord.relPath,
    sourceText: JSON.stringify(game),
    keywords: ['browser game', ...tags],
    links,
    sections: [
      { title: 'Overview', paragraphs: [summary] },
      ...(tags.length ? [{ title: 'System Focus', items: tags }] : []),
      { title: 'Play', links }
    ]
  });
}

function buildPersonalHomeStructuredPage(audienceRecord, projectRecords, toolRecords, gamesPageRecord) {
  const audience = audienceRecord && audienceRecord.data;
  const page = audience && audience.page;
  if (!page) return null;
  const accordion = (page.sections || []).find(section => section.type === 'home-accordion' && section.enabled !== false);
  const categories = new Map((accordion?.props?.categories || []).map(category => [category.id, category]));
  const about = categories.get('about') || {};
  const story = about.aboutStory || {};
  const connections = Array.isArray(story.connections) ? story.connections : [];
  const personalLinks = connections.length
    ? connections.filter(connection => connection.project?.href).map(connection => ({
        label: connection.project.title,
        url: connection.project.href,
        description: connection.project.summary
      }))
    : (about.featuredItems || []).map(item => ({ label: item.title, url: item.href, description: item.summary }));
  const introLinks = [
    about.primaryAction && { label: about.primaryAction.label, url: about.primaryAction.href },
    about.currentWork && { label: about.currentWork.text, url: about.currentWork.href }
  ].filter(Boolean);
  const contact = categories.get('contact') || {};

  const projects = projectRecords
    .map((record) => record.data)
    .filter(isPublicProject)
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));
  const tools = toolRecords
    .map((record) => record.data)
    .filter(isPublicTool)
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));
  const gamesPage = gamesPageRecord && gamesPageRecord.data;
  const games = (Array.isArray(gamesPage && gamesPage.games) ? gamesPage.games : [])
    .filter((game) => game && game.published !== false && !game.hidden && !game.noindex &&
      isPublicVisibility(game.visibility) && (game.href || game.id))
    .sort((a, b) => (a.order || 999) - (b.order || 999) || normalizeWhitespace(a.title).localeCompare(normalizeWhitespace(b.title)));

  const projectLinks = projects.map((project) => ({
    label: project.title,
    url: `/portfolio/${project.id}`,
    description: projectSummary(project)
  }));
  const toolLinks = tools.map((tool) => ({
    label: tool.title,
    url: tool.href || `/tools/${tool.slug}`,
    description: tool.summary
  }));
  const gameLinks = games.map((game) => ({
    label: game.title,
    url: game.href || `/games/${game.id}`,
    description: game.summary
  }));

  return createStructuredPage(page.canonicalPath || '/', {
    title: stripSiteSuffix(page.title || audience.label || 'Daniel Short'),
    description: page.description,
    summary: page.description,
    category: 'Core',
    sourcePath: audienceRecord.relPath,
    sourceText: JSON.stringify({ audience, projects, tools, games }),
    introParagraphs: [about.lead, about.context].filter(Boolean),
    keywords: ['projects', 'tools', 'games', 'machine learning', 'data analytics', 'browser experiments'],
    links: [
      ...personalLinks,
      { label: 'Projects', url: '/portfolio', description: `${projects.length} projects in the project library.` },
      { label: 'Tools', url: '/tools', description: `${tools.length} practical browser tools.` },
      { label: 'Games', url: '/games', description: `${games.length} browser games and simulations.` },
      ...projectLinks.slice(0, 8),
      ...toolLinks.slice(0, 6),
      ...gameLinks.slice(0, 5)
    ],
    sections: [
      {
        title: about.title || 'About Daniel',
        links: introLinks
      },
      {
        title: story.title || about.featuredTitle || 'Featured projects',
        paragraphs: connections.map(connection => [connection.title, connection.description].filter(Boolean).join(': ')),
        links: personalLinks
      },
      {
        title: categories.get('projects')?.title || 'Projects',
        paragraphs: [categories.get('projects')?.lead, `${projects.length} published projects in the project library.`].filter(Boolean),
        links: projectLinks
      },
      {
        title: categories.get('tools')?.title || 'Tools',
        paragraphs: [categories.get('tools')?.lead, `${tools.length} public tools in the tool library.`].filter(Boolean),
        links: toolLinks
      },
      {
        title: categories.get('games')?.title || 'Games',
        paragraphs: [categories.get('games')?.lead, `${games.length} games in the game library.`].filter(Boolean),
        links: gameLinks
      },
      {
        title: contact.title || 'Contact',
        paragraphs: [contact.lead].filter(Boolean),
        links: (contact.items || []).map(item => ({ label: item.title, url: item.href, description: item.summary }))
      }
    ]
  });
}

function sitemapRouteLabel(urlPath) {
  if (urlPath === '/') return 'Home';
  return urlPath.slice(1).split('/').map((part) => {
    const words = part.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[-_]+/g, ' ');
    return words.charAt(0).toUpperCase() + words.slice(1);
  }).join(' / ');
}

function buildSitemapStructuredPage(structuredPages) {
  const sitemapPath = path.join(root, 'sitemap.xml');
  if (!fs.existsSync(sitemapPath)) return null;
  const xml = fs.readFileSync(sitemapPath, 'utf8');
  const urls = uniqueList([...xml.matchAll(/<loc>([^<]+)<\/loc>/gi)]
    .map((match) => toPathFromCanonical(decodeHtml(match[1])))
    .filter(Boolean));
  if (!urls.length) return null;
  const groups = [
    ['Main pages', urls.filter((url) => routeCategory(url) === 'Core' || routeCategory(url) === 'Page')],
    ['Projects', urls.filter((url) => routeCategory(url) === 'Portfolio')],
    ['Tools', urls.filter((url) => routeCategory(url) === 'Tools')],
    ['Games', urls.filter((url) => routeCategory(url) === 'Games')]
  ];
  const labels = new Map(urls.map((url) => [url, structuredPages.get(url)?.title || sitemapRouteLabel(url)]));
  return createStructuredPage('/sitemap', {
    title: 'Sitemap',
    description: 'Browse indexable public pages across Daniel Short’s projects, tools, games, and site information.',
    summary: 'Browse indexable public pages across Daniel Short’s projects, tools, games, and site information.',
    category: 'Page',
    sourcePath: 'sitemap.xml',
    sourceText: JSON.stringify({ xml, labels: [...labels] }),
    introParagraphs: [`Browse ${urls.length} indexable public pages. The XML sitemap is the source for this list.`],
    links: [{ label: 'XML sitemap', url: '/sitemap.xml' }],
    sections: groups.map(([title, routes]) => ({
      title,
      links: routes.map((url) => ({ label: labels.get(url), url }))
    }))
  });
}

function loadStructuredPages() {
  const structured = new Map();
  const projectRecords = loadJsonRecords('content/projects');
  const toolRecords = loadJsonRecords('content/tools');
  const personalAudienceRecord = {
    relPath: 'content/audiences/personal.json',
    data: readJsonRel('content/audiences/personal.json')
  };
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

  const personalHomePage = buildPersonalHomeStructuredPage(personalAudienceRecord, projectRecords, toolRecords, gamesPageRecord);
  if (personalHomePage) structured.set(personalHomePage.url, personalHomePage);

  const toolsPage = buildToolsDirectoryStructuredPage(toolsPageRecord, toolRecords);
  if (toolsPage) structured.set(toolsPage.url, toolsPage);

  const gamesPage = buildGamesDirectoryStructuredPage(gamesPageRecord);
  if (gamesPage) structured.set(gamesPage.url, gamesPage);

  ((gamesPageRecord.data && gamesPageRecord.data.games) || []).forEach((game) => {
    const page = buildGameStructuredPage(game, gamesPageRecord);
    if (page) structured.set(page.url, page);
  });

  toolRecords.forEach((record) => {
    const page = buildToolStructuredPage(record, categoriesById);
    if (page) structured.set(page.url, page);
  });

  const sitemapPage = buildSitemapStructuredPage(structured);
  if (sitemapPage) structured.set(sitemapPage.url, sitemapPage);

  return structured;
}

function applyStructuredPage(basePage, structuredPage) {
  if (!structuredPage) return basePage;
  // Authored catalog data is the source of truth. Scraped HTML can contain
  // hidden controls, status text, or cards unavailable to public visitors.
  return {
    ...basePage,
    title: structuredPage.title || basePage.title,
    description: structuredPage.description || basePage.description,
    summary: structuredPage.summary || basePage.summary,
    category: structuredPage.category || basePage.category,
    sourcePath: structuredPage.sourcePath || basePage.sourcePath,
    sourceHash: structuredPage.sourceHash || basePage.sourceHash,
    sections: structuredPage.sections,
    introParagraphs: structuredPage.introParagraphs,
    keywords: structuredPage.keywords,
    links: structuredPage.links,
    facts: [],
    evidence: [],
    bodyPoints: []
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
  const tokenRe = /<(h[1-6]|summary|p|li|a)\b([^>]*)>([\s\S]*?)<\/\1>/gi;
  let match;
  while ((match = tokenRe.exec(String(region || '')))) {
    const tag = String(match[1] || '').toLowerCase();
    const text = cleanText(match[3]);
    if (!text || isNoiseText(text)) continue;

    if (/^h[1-6]$/.test(tag) || tag === 'summary') {
      if (tag === 'h1') continue;
      pushCurrent();
      current = {
        title: text,
        paragraphs: [],
        items: [],
        links: [],
        level: tag === 'summary' ? 2 : Math.min(6, Math.max(2, Number(tag.slice(1)) || 2))
      };
      continue;
    }

    if (tag === 'p') {
      const inlineLinks = extractLinks(match[3]);
      if (current) current.links.push(...inlineLinks);
      const prose = cleanText(match[3].replace(/<a\b[^>]*>[\s\S]*?<\/a>/gi, ' '));
      if (inlineLinks.length && !prose) continue;
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
      const label = linkLabel(match[3]);
      if (!href || !label || label.length > 140) continue;
      current.links.push({ label, url: href });
    }
  }
  pushCurrent();
  return {
    introParagraphs: uniqueList(introParagraphs, 6),
    sections: sections.map(normalizeSection).filter(Boolean)
  };
}

function buildDigestPage({ html, relPath, urlPath, override, generatedAt, structuredPage }) {
  const region = extractMainRegion(html);
  let mainText = extractCleanMainText(html);
  if (mainText.length < 80) {
    mainText = [mainText, fallbackMainTextForRoute(urlPath)].filter(Boolean).join(' ');
  }
  if (mainText.length < 80 && structuredPage) {
    // Form-led tools can have no prose left after removing interactive inputs.
    // Use their authored public metadata without exposing form values.
    mainText = [
      mainText,
      structuredPage.summary,
      ...(structuredPage.introParagraphs || []),
      ...(structuredPage.sections || []).flatMap((section) => [...(section.paragraphs || []), ...(section.items || [])])
    ].filter(Boolean).join(' ').slice(0, MAX_SOURCE_CHARS);
  }
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
    ...extractKeywords(region),
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
      : extractedSections.sections.length ? [] : paragraphs.slice(0, 4),
    sections: extractedSections.sections,
    keywords,
    links: dedupedLinks.slice(0, MAX_LINKS)
  };
}

function buildDigests() {
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const publicToolSlugs = loadPublicToolSlugs();
  const nonpublicCatalogRoutes = loadNonpublicCatalogRoutes();
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
    if (relPath.startsWith('pages/professional/')) return;

    let html = '';
    try {
      html = fs.readFileSync(absPath, 'utf8');
    } catch {
      return;
    }

    const urlPath = toPathFromCanonical(extractCanonical(html)) || toPathFromRelFile(relPath, publicToolSlugs);
    const normalizedUrl = normalizePathname(urlPath);
    const override = overrides.get(normalizedUrl);
    if (nonpublicCatalogRoutes.has(normalizedUrl) ||
      shouldExcludeUrl(normalizedUrl, html, noindexPathnames, override)) return;

    const structuredPage = structuredPages.get(normalizedUrl);
    let page = buildDigestPage({ html, relPath, urlPath: normalizedUrl, override, generatedAt, structuredPage });
    if (!page) return;
    page = applyStructuredPage(page, structuredPage);

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
  const url = page.canonicalUrl || `${SITE_ORIGIN}${page.url}`;
  const description = trimLlmsDescription(page.summary || page.description || '');
  const aiUrl = page.aiUrl || routeToAiUrl(page.url);
  return `- [${label}](${url})${description ? `: ${description}` : ''} ([AI summary](${aiUrl}))`;
}

function llmsSection(title, pages) {
  const lines = uniqueList((pages || []).filter(Boolean).map(llmsLine));
  if (!lines.length) return '';
  return [`## ${title}`, '', ...lines].join('\n');
}

function renderLlmsTxt(pages) {
  const byUrl = new Map((pages || []).map((page) => [page.url, page]));
  const assigned = new Set();
  const pick = (urls) => urls.map((url) => byUrl.get(url)).filter((page) => {
    if (!page || assigned.has(page.url)) return false;
    assigned.add(page.url);
    return true;
  });
  const startPages = pick(['/', '/portfolio', '/tools', '/games', '/contact']);
  const projectPages = pick((pages || []).filter((page) => page.category === 'Portfolio').map((page) => page.url));
  const toolPages = pick((pages || []).filter((page) => page.category === 'Tools').map((page) => page.url));
  const gamePages = pick((pages || []).filter((page) => page.category === 'Games').map((page) => page.url));
  const otherPages = pick((pages || []).map((page) => page.url));
  const sections = [
    llmsSection('Start Here', startPages),
    llmsSection('Projects', projectPages),
    llmsSection('Tools', toolPages),
    llmsSection('Games', gamePages),
    llmsSection('Other Pages', otherPages)
  ].filter(Boolean);

  return [
    '# Daniel Short',
    '',
    '> Supplemental, AI-readable summaries of Daniel Short\'s projects, tools, experiments, and contact information.',
    '',
    'Canonical site: https://www.danielshort.me/',
    'Each main link opens the authoritative public page. Its adjacent AI summary link opens a shorter, optional /ai/ page that canonicalizes back to that public page. AI platforms may use either version; this file does not change what the main URL serves.',
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
