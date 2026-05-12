#!/usr/bin/env node
'use strict';

/*
  Generate deterministic, JS-free page digests for AI retrieval agents.

  The output is intentionally static and reviewable:
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
const SITE_ORIGIN = 'https://www.danielshort.me';
const MAX_SOURCE_CHARS = 8000;
const MAX_SUMMARY_CHARS = 520;
const MAX_FACTS = 10;
const MAX_EVIDENCE = 8;
const MAX_BODY_POINTS = 6;
const MAX_LINKS = 14;

const excludedPathPatterns = [
  /^\/$/i,
  /^\/admin(?:\/|$)/i,
  /^\/api(?:\/|$)/i,
  /^\/ai(?:\/|$)/i,
  /^\/pages(?:\/|$)/i,
  /^\/search$/i,
  /^\/sitemap-pretty$/i,
  /^\/(?:resume|resume-pdf)$/i,
  /^\/resume-(?:analytics|data-science|tourism)-pdf$/i,
  /^\/tools\/(?:dashboard|short-links|ga4-utm-performance|whisper-transcribe-monitor)$/i
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

function ensureCleanDir(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
  fs.mkdirSync(dirPath, { recursive: true });
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

function loadPublicToolSlugs() {
  const slugs = new Set();
  walkFiles(path.join(root, 'content', 'tools'), (filePath) => filePath.endsWith('.json')).forEach((filePath) => {
    try {
      const tool = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const slug = normalizeWhitespace(tool && tool.slug);
      const visibility = normalizeWhitespace(tool && tool.visibility).toLowerCase();
      if (!slug || tool.hidden || tool.noindex || visibility === 'admin') return;
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
  if (urlPath.includes('resume')) return 'Resume';
  if (urlPath.startsWith('/games/')) return 'Games';
  if (['/analytics', '/data-science', '/tourism', '/contact'].includes(urlPath)) return 'Core';
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
    .replace(/<article\b[^>]*\bdata-tools-visibility="admin"[^>]*>[\s\S]*?<\/article>/gi, ' ')
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
    ...links.map((link) => `  <li><a href="${escapeHtml(link.url)}">${escapeHtml(link.label)}</a></li>`),
    '</ul>'
  ].join('\n');
}

function renderDigest(page) {
  const generatedAt = page.generatedAt;
  const sections = [];
  if (page.summary) {
    sections.push(`<section aria-labelledby="summary">
  <h2 id="summary">Summary</h2>
  <p>${escapeHtml(page.summary)}</p>
</section>`);
  }
  if (page.facts.length) {
    sections.push(`<section aria-labelledby="key-facts">
  <h2 id="key-facts">Key Facts</h2>
${renderList(page.facts)}
</section>`);
  }
  if (page.evidence.length) {
    sections.push(`<section aria-labelledby="evidence">
  <h2 id="evidence">Evidence And Outcomes</h2>
${renderList(page.evidence)}
</section>`);
  }
  if (page.bodyPoints.length) {
    sections.push(`<section aria-labelledby="context">
  <h2 id="context">Context</h2>
${renderList(page.bodyPoints)}
</section>`);
  }
  if (page.keywords.length) {
    sections.push(`<section aria-labelledby="topics">
  <h2 id="topics">Topics</h2>
  <p>${escapeHtml(page.keywords.join(', '))}</p>
</section>`);
  }
  if (page.links.length) {
    sections.push(`<section aria-labelledby="links">
  <h2 id="links">Relevant Links</h2>
${renderLinks(page.links)}
</section>`);
  }

  return `<!DOCTYPE html>
<html lang="en" data-ai-digest="true">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${escapeHtml(page.title)} | AI Digest</title>
  <link rel="canonical" href="${escapeHtml(page.canonicalUrl)}">
  <meta name="description" content="${escapeHtml(page.description || page.summary || page.title)}">
  <meta name="generator" content="Daniel Short deterministic AI digest">
</head>
<body>
  <main id="main" data-ai-digest="true" data-canonical-url="${escapeHtml(page.canonicalUrl)}">
    <article>
      <header>
        <p>AI digest for <a href="${escapeHtml(page.canonicalUrl)}">${escapeHtml(page.canonicalUrl)}</a></p>
        <h1>${escapeHtml(page.title)}</h1>
        <p>Page type: ${escapeHtml(page.category)}</p>
      </header>
${sections.join('\n')}
      <footer>
        <h2>Source Metadata</h2>
        <ul>
          <li>Canonical URL: <a href="${escapeHtml(page.canonicalUrl)}">${escapeHtml(page.canonicalUrl)}</a></li>
          <li>Source path: ${escapeHtml(page.sourcePath)}</li>
          <li>Source hash: ${escapeHtml(page.sourceHash)}</li>
          <li>Generated at: ${escapeHtml(generatedAt)}</li>
        </ul>
      </footer>
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

function buildDigestPage({ html, relPath, urlPath, override, generatedAt }) {
  const region = extractMainRegion(html);
  const mainText = extractCleanMainText(html);
  if (!mainText || mainText.length < 80) return null;

  const headings = uniqueList(extractTagTexts(region, ['h1', 'h2', 'h3']), 14);
  const listItems = uniqueList(extractTagTexts(region, ['li']), 80).filter((item) => item.length >= 20 && item.length <= 280);
  const paragraphs = uniqueList(extractTagTexts(region, ['p']), 80).filter((item) => item.length >= 35 && item.length <= 340);
  const sentences = splitSentences(mainText).filter((item) => item.length >= 35 && item.length <= 300);

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
    outputPath: `dist/ai-pages/${outputRelPath}`,
    outputRelPath,
    generatedAt,
    facts,
    evidence,
    bodyPoints,
    keywords,
    links: dedupedLinks.slice(0, MAX_LINKS)
  };
}

function buildDigests() {
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const publicToolSlugs = loadPublicToolSlugs();
  const overrides = loadAiDigestOverrides();
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

    const page = buildDigestPage({ html, relPath, urlPath: normalizedUrl, override, generatedAt });
    if (!page) return;

    const previous = pagesByUrl.get(normalizedUrl);
    const previousScore = previous ? previous.facts.length + previous.evidence.length + previous.bodyPoints.length : -1;
    const nextScore = page.facts.length + page.evidence.length + page.bodyPoints.length;
    if (!previous || nextScore >= previousScore) pagesByUrl.set(normalizedUrl, page);
  });

  return [...pagesByUrl.values()].sort((a, b) => a.url.localeCompare(b.url));
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
}

function main() {
  const pages = buildDigests();
  writeOutputs(pages);
  process.stdout.write(`[ai-digests] Wrote dist/ai-digest-manifest.json and ${pages.length} AI page digest(s)\n`);
}

main();
