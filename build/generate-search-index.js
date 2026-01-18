#!/usr/bin/env node
'use strict';

/*
  Generate a lightweight, dependency-free site search index.

  - Reads canonical URLs from sitemap.xml
  - Extracts title/description/keywords from local HTML pages
  - Writes dist/search-index.json (copied to public/ during build)
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const sitemapPath = path.join(root, 'sitemap.xml');
const toolsIndexPath = path.join(root, 'pages', 'tools.html');
const outPath = path.join(root, 'dist', 'search-index.json');
const SITE_ORIGIN = 'https://danielshort.me';

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function ensureDir(relDir) {
  fs.mkdirSync(path.join(root, relDir), { recursive: true });
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
  const t = String(title || '').trim();
  if (!t) return '';
  return t
    .replace(/\s*\|\s*Daniel Short\s*$/i, '')
    .replace(/\s*-\s*Daniel Short\s*$/i, '')
    .trim();
}

function extractTitle(html) {
  const match = /<title>([^<]+)<\/title>/i.exec(String(html || ''));
  return stripSiteSuffix(decodeHtml(match ? match[1] : ''));
}

function stripOwnerPrefix(title, urlPath) {
  const t = String(title || '').trim();
  if (!t) return '';
  if (String(urlPath || '') === '/') return t;
  return t.replace(/^Daniel Short\s*-\s*/i, '').trim();
}

function extractMeta(html, attr, key) {
  const safeAttr = String(attr || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const safeKey = String(key || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const re = new RegExp(`<meta\\b[^>]*\\b${safeAttr}="${safeKey}"[^>]*\\bcontent="([^"]*)"[^>]*>`, 'i');
  const match = re.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractDescription(html) {
  const d = extractMeta(html, 'name', 'description') || extractMeta(html, 'property', 'og:description');
  return String(d || '').trim();
}

function extractProjectTags(html) {
  const tags = [];
  const re = /<span\b[^>]*\bclass="project-tag"[^>]*>([^<]+)<\/span>/gi;
  let match;
  while ((match = re.exec(String(html || '')))) {
    const tag = decodeHtml(match[1]).replace(/\s+/g, ' ').trim();
    if (!tag) continue;
    tags.push(tag);
  }
  return [...new Set(tags)];
}

function loadToolKeywords() {
  const map = new Map();
  if (!exists('pages/tools.html')) return map;
  let html;
  try { html = read('pages/tools.html'); } catch { return map; }

  const cards = String(html).split('<article class="tool-card">').slice(1);
  cards.forEach((card) => {
    const hrefMatch = /href="tools\/([^"#?]+)"/i.exec(card);
    if (!hrefMatch) return;
    const slug = String(hrefMatch[1] || '').trim();
    if (!slug) return;

    const pills = [];
    const pillRe = /<span\b[^>]*\bclass="tool-pill[^"]*"[^>]*>([^<]+)<\/span>/gi;
    let match;
    while ((match = pillRe.exec(card))) {
      const pill = decodeHtml(match[1]).replace(/\s+/g, ' ').trim();
      if (!pill) continue;
      pills.push(pill);
    }
    map.set(slug, [...new Set(pills)]);
  });

  return map;
}

function extractSitemapUrls(xml) {
  const urls = new Set();
  const re = /<loc>([^<]+)<\/loc>/g;
  let match;
  while ((match = re.exec(String(xml || '')))) {
    const loc = String(match[1] || '').trim();
    if (!loc) continue;
    if (!loc.startsWith(SITE_ORIGIN)) continue;
    urls.add(loc);
  }
  return [...urls].sort();
}

function toPath(loc) {
  const raw = String(loc || '').trim();
  if (!raw) return '';
  const stripped = raw.replace(SITE_ORIGIN, '') || '/';
  const withSlash = stripped.startsWith('/') ? stripped : `/${stripped}`;
  return withSlash === '' ? '/' : withSlash;
}

function categoryForPath(urlPath) {
  if (urlPath === '/portfolio' || urlPath.startsWith('/portfolio/')) return 'Portfolio';
  if (urlPath === '/tools' || urlPath.startsWith('/tools/')) return 'Tools';
  return 'Pages';
}

function sourceFileForPath(urlPath) {
  if (urlPath === '/') return 'index.html';
  if (urlPath === '/portfolio') return 'pages/portfolio.html';
  if (urlPath.startsWith('/portfolio/')) {
    const id = urlPath.replace(/^\/portfolio\//, '');
    return `pages/portfolio/${id}.html`;
  }
  if (urlPath === '/tools') return 'pages/tools.html';
  if (urlPath === '/tools/dashboard') return 'pages/tools-dashboard.html';
  if (urlPath.startsWith('/tools/')) {
    const slug = urlPath.replace(/^\/tools\//, '');
    return `pages/${slug}.html`;
  }
  const slug = urlPath.replace(/^\/+/, '');
  return `pages/${slug}.html`;
}

function main() {
  if (!fs.existsSync(sitemapPath)) {
    process.stderr.write('[search-index] Missing sitemap.xml; run build first.\n');
    process.exit(1);
  }

  const sitemapXml = fs.readFileSync(sitemapPath, 'utf8');
  const locs = extractSitemapUrls(sitemapXml);
  const toolKeywords = loadToolKeywords();

  const pages = [];
  locs.forEach((loc) => {
    const urlPath = toPath(loc);
    const sourceFile = sourceFileForPath(urlPath);
    if (!exists(sourceFile)) return;

    let html;
    try { html = read(sourceFile); } catch { return; }

    const category = categoryForPath(urlPath);
    const title = stripOwnerPrefix(extractTitle(html) || urlPath, urlPath) || urlPath;
    const description = extractDescription(html);

    let keywords = [];
    if (category === 'Portfolio' && urlPath.startsWith('/portfolio/')) {
      keywords = extractProjectTags(html);
    }
    if (category === 'Tools' && urlPath.startsWith('/tools/')) {
      const slug = urlPath.replace(/^\/tools\//, '');
      keywords = toolKeywords.get(slug) || [];
    }

    const entry = {
      url: urlPath,
      title,
      ...(description ? { description } : {}),
      category,
      ...(keywords.length ? { keywords } : {})
    };
    pages.push(entry);
  });

  pages.sort((a, b) => {
    const cat = String(a.category).localeCompare(String(b.category));
    if (cat !== 0) return cat;
    return String(a.title).localeCompare(String(b.title));
  });

  ensureDir('dist');
  fs.writeFileSync(
    outPath,
    JSON.stringify({ generatedAt: new Date().toISOString(), pages }, null, 2) + '\n',
    'utf8'
  );

  process.stdout.write(`[search-index] Wrote dist/search-index.json (${pages.length} pages)\n`);
}

main();
