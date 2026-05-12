#!/usr/bin/env node
'use strict';

/*
  Generate a citeable, dependency-free knowledge database for the embedded site
  chatbot. The output is intentionally simple JSON so the serverless API can
  retrieve context cheaply without a hosted vector database.
*/

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');

const root = path.resolve(__dirname, '..');
const outPath = path.join(root, 'dist', 'chatbot-knowledge.json');
const SITE_ORIGIN = 'https://www.danielshort.me';
const MAX_CHUNK_CHARS = 900;
const MIN_CHUNK_CHARS = 260;
const MAX_PAGE_CHUNKS = 8;

const excludedPathPatterns = [
  /^\/(?:chatbot-demo|.*-demo)(?:\/|$)/i,
  /^\/games(?:\/|$)/i,
  /^\/tools(?:\/|$)/i,
  /^\/privacy$/i,
  /^\/sitemap(?:-pretty)?$/i,
  /^\/resume(?:-pdf)?$/i,
  /^\/resume-(?:analytics|data-science|tourism)-pdf$/i
];

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

function stripSiteSuffix(title) {
  return String(title || '')
    .replace(/\s*\|\s*Daniel Short\s*$/i, '')
    .replace(/\s*-\s*Daniel Short\s*$/i, '')
    .trim();
}

function extractTitle(html) {
  const match = /<title>([^<]+)<\/title>/i.exec(String(html || ''));
  return stripSiteSuffix(decodeHtml(match ? match[1] : ''));
}

function extractMeta(html, attr, key) {
  const safeAttr = String(attr || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const safeKey = String(key || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const re = new RegExp(`<meta\\b[^>]*\\b${safeAttr}="${safeKey}"[^>]*\\bcontent="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractDescription(html) {
  return extractMeta(html, 'name', 'description') || extractMeta(html, 'property', 'og:description');
}

function extractCanonical(html) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractRobots(html) {
  const match = /<meta\s+[^>]*name="robots"[^>]*content="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function isNoindex(html) {
  return extractRobots(html).toLowerCase().includes('noindex');
}

function walkHtmlFiles(dirRelPath) {
  const start = path.join(root, dirRelPath);
  if (!fs.existsSync(start)) return [];
  const files = [];
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
      if (entry.isFile() && entry.name.endsWith('.html')) files.push(full);
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

function toPathFromCanonical(canonical) {
  const raw = String(canonical || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizePathname(url.pathname || '/');
  } catch {
    return '';
  }
}

function toPathFromRelFile(relPath) {
  const safe = String(relPath || '').replace(/\\/g, '/');
  if (safe === 'index.html') return '/';
  if (safe.startsWith('pages/portfolio/') && safe.endsWith('.html')) {
    const id = safe.replace(/^pages\/portfolio\//, '').replace(/\.html$/, '');
    return id ? `/portfolio/${encodeURIComponent(id)}` : '';
  }
  if (safe.startsWith('pages/') && safe.endsWith('.html')) {
    const slug = safe.replace(/^pages\//, '').replace(/\.html$/, '');
    return slug ? `/${slug}` : '';
  }
  if (safe.endsWith('.html') && !safe.includes('/')) {
    const slug = safe.replace(/\.html$/, '');
    return slug ? `/${slug}` : '';
  }
  return '';
}

function categoryForPath(urlPath) {
  if (urlPath === '/portfolio' || urlPath.startsWith('/portfolio/')) return 'Portfolio';
  if (urlPath.includes('resume')) return 'Resume';
  if (['/analytics', '/data-science', '/tourism', '/contact'].includes(urlPath)) return 'Core';
  return 'Pages';
}

function audienceForPath(urlPath) {
  if (urlPath.includes('tourism') || urlPath.includes('destination')) return 'tourism';
  if (urlPath.includes('data-science')) return 'data-science';
  if (urlPath.includes('analytics')) return 'analytics';
  return '';
}

function extractKeywords(html) {
  const keywords = [];
  const tagRe = /<span\b[^>]*\bclass="[^"]*\b(?:project-tag|tool-pill|resume-chip)\b[^"]*"[^>]*>([\s\S]*?)<\/span>/gi;
  let match;
  while ((match = tagRe.exec(String(html || '')))) {
    const text = cleanText(match[1]);
    if (text) keywords.push(text);
  }
  return [...new Set(keywords)].slice(0, 30);
}

function cleanText(value) {
  return decodeHtml(String(value || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<[^>]+>/g, ' '))
    .replace(/\s+/g, ' ')
    .trim();
}

function extractMainText(html) {
  const stripped = String(html || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ');
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(stripped);
  let region = mainMatch ? mainMatch[1] : stripped;
  region = region
    .replace(/<header\b[^>]*>[\s\S]*?<\/header>/gi, ' ')
    .replace(/<nav\b[^>]*>[\s\S]*?<\/nav>/gi, ' ')
    .replace(/<footer\b[^>]*>[\s\S]*?<\/footer>/gi, ' ')
    .replace(/<button\b[^>]*>[\s\S]*?<\/button>/gi, ' ')
    .replace(/<form\b[^>]*>[\s\S]*?<\/form>/gi, ' ');
  return cleanText(region);
}

function splitIntoSentences(text) {
  return String(text || '')
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/g)
    .map((part) => part.trim())
    .filter(Boolean);
}

function chunkText(text) {
  const sentences = splitIntoSentences(text);
  const chunks = [];
  let current = '';

  sentences.forEach((sentence) => {
    const candidate = current ? `${current} ${sentence}` : sentence;
    if (candidate.length <= MAX_CHUNK_CHARS) {
      current = candidate;
      return;
    }
    if (current.length >= MIN_CHUNK_CHARS) chunks.push(current);
    current = sentence.length > MAX_CHUNK_CHARS ? sentence.slice(0, MAX_CHUNK_CHARS) : sentence;
  });

  if (current.length >= 120) chunks.push(current);
  if (!chunks.length && text) chunks.push(text.slice(0, MAX_CHUNK_CHARS));
  return chunks.slice(0, MAX_PAGE_CHUNKS);
}

function makeId(input) {
  return crypto.createHash('sha256').update(String(input || '')).digest('hex').slice(0, 16);
}

function isExcludedUrl(urlPath, noindexPathnames) {
  const normalized = normalizePathname(urlPath);
  if (!normalized) return true;
  if (noindexPathnames.has(normalized)) return true;
  return excludedPathPatterns.some((pattern) => pattern.test(normalized));
}

function buildKnowledge() {
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const files = [...listRootHtmlFiles(), ...walkHtmlFiles('pages')];
  const pagesByUrl = new Map();

  files.forEach((absPath) => {
    const sourcePath = relFromRoot(absPath);
    let html = '';
    try {
      html = fs.readFileSync(absPath, 'utf8');
    } catch {
      return;
    }
    if (isNoindex(html)) return;

    const url = toPathFromCanonical(extractCanonical(html)) || toPathFromRelFile(sourcePath);
    if (isExcludedUrl(url, noindexPathnames)) return;

    const title = extractTitle(html) || url;
    const description = extractDescription(html);
    const text = extractMainText(html);
    if (!text || text.length < 160) return;

    const entry = {
      url,
      title,
      description,
      sourcePath,
      category: categoryForPath(url),
      audience: audienceForPath(url),
      keywords: extractKeywords(html),
      text
    };

    const previous = pagesByUrl.get(url);
    if (!previous || entry.text.length > previous.text.length) pagesByUrl.set(url, entry);
  });

  const chunks = [];
  [...pagesByUrl.values()]
    .sort((a, b) => a.url.localeCompare(b.url))
    .forEach((page) => {
      chunkText([page.description, page.text].filter(Boolean).join(' ')).forEach((text, index) => {
        chunks.push({
          id: makeId(`${page.url}:${index}:${text}`),
          pageId: makeId(page.url),
          url: page.url,
          title: page.title,
          category: page.category,
          audience: page.audience,
          sourcePath: page.sourcePath,
          keywords: page.keywords,
          text
        });
      });
    });

  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    origin: SITE_ORIGIN,
    pages: [...pagesByUrl.values()].map(({ text, ...page }) => page),
    chunks
  };
}

function main() {
  ensureDir(path.dirname(outPath));
  const knowledge = buildKnowledge();
  fs.writeFileSync(outPath, JSON.stringify(knowledge, null, 2) + '\n', 'utf8');
  process.stdout.write(`[chatbot-knowledge] Wrote dist/chatbot-knowledge.json (${knowledge.pages.length} pages, ${knowledge.chunks.length} chunks)\n`);
}

main();
