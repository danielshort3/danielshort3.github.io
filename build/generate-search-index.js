#!/usr/bin/env node
'use strict';

/*
  Generate a lightweight, dependency-free site search index.

  - Scans root HTML + pages/** for indexable pages
  - Uses canonical URLs when present (preferred)
  - Extracts title/description/keywords (+ short main content excerpt)
  - Writes dist/search-index.json (copied to public/ during build)
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
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

function relFromRoot(absPath) {
  return path.relative(root, absPath).replace(/\\/g, '/');
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

function extractCanonical(html) {
  const match = /<link\s+[^>]*rel="canonical"[^>]*href="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function extractRobots(html) {
  const match = /<meta\s+[^>]*name="robots"[^>]*content="([^"]+)"[^>]*>/i.exec(String(html || ''));
  return decodeHtml(match ? match[1] : '').trim();
}

function isNoindex(html) {
  const robots = extractRobots(html).toLowerCase();
  return robots.includes('noindex');
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

function normalizeUrlPath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  if (raw === '/') return '/';
  return raw.replace(/\/+$/, '');
}

function toPathFromCanonical(canonical) {
  const raw = String(canonical || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, SITE_ORIGIN);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizeUrlPath(url.pathname || '/');
  } catch {
    return '';
  }
}

	function toPathFromRelFile(relPath, toolKeywords) {
	  const safe = String(relPath || '').replace(/\\/g, '/');
	  if (safe === 'index.html') return '/';
	  if (safe.startsWith('pages/portfolio/') && safe.endsWith('.html')) {
	    const id = safe.replace(/^pages\/portfolio\//, '').replace(/\.html$/, '');
	    return id ? `/portfolio/${encodeURIComponent(id)}` : '';
	  }
	  if (safe.startsWith('pages/') && safe.endsWith('.html')) {
	    const slug = safe.replace(/^pages\//, '').replace(/\.html$/, '');
	    if (!slug) return '';
	    if (toolKeywords && typeof toolKeywords.has === 'function' && toolKeywords.has(slug)) {
	      return `/tools/${slug}`;
	    }
	    return `/${slug}`;
	  }
	  if (safe.endsWith('.html') && !safe.includes('/')) {
	    const slug = safe.replace(/\.html$/, '');
	    return slug ? `/${slug}` : '';
	  }
	  return '';
	}

function categoryForPath(urlPath) {
  if (urlPath === '/portfolio' || urlPath.startsWith('/portfolio/')) return 'Portfolio';
  if (urlPath === '/tools' || urlPath.startsWith('/tools/')) return 'Tools';
  return 'Pages';
}

function stripIndexNoise(html) {
  let out = String(html || '');
  out = out.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ');
  out = out.replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ');
  return out;
}

function extractIndexableText(html) {
  const stripped = stripIndexNoise(html);
  const mainMatch = /<main\b[^>]*>([\s\S]*?)<\/main>/i.exec(stripped);
  let region = mainMatch ? mainMatch[1] : stripped;
  region = region.replace(/<header\b[^>]*>[\s\S]*?<\/header>/gi, ' ');
  region = region.replace(/<nav\b[^>]*>[\s\S]*?<\/nav>/gi, ' ');
  region = region.replace(/<footer\b[^>]*>[\s\S]*?<\/footer>/gi, ' ');
  region = region.replace(/<[^>]+>/g, ' ');
  const text = decodeHtml(region).replace(/\s+/g, ' ').trim();
  if (!text) return '';
  const maxChars = 2200;
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).replace(/\s+\S*$/, '') + '…';
}

function main() {
  const toolKeywords = loadToolKeywords();

  const candidates = [
    ...listRootHtmlFiles(),
    ...walkHtmlFiles('pages')
  ];

  const byUrl = new Map();

  candidates.forEach((absPath) => {
    const relPath = relFromRoot(absPath);
    if (!relPath) return;
    if (relPath === 'public' || relPath.startsWith('public/')) return;
    if (relPath === 'node_modules' || relPath.startsWith('node_modules/')) return;
    if (!relPath.endsWith('.html')) return;

    let html;
    try { html = read(relPath); } catch { return; }
    if (isNoindex(html)) return;

	    const canonical = extractCanonical(html);
	    const urlPath = toPathFromCanonical(canonical) || toPathFromRelFile(relPath, toolKeywords);
	    if (!urlPath || !urlPath.startsWith('/')) return;

    const category = categoryForPath(urlPath);
    const title = stripOwnerPrefix(extractTitle(html) || urlPath, urlPath) || urlPath;
    const description = extractDescription(html);
    const content = extractIndexableText(html);

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
      ...(keywords.length ? { keywords } : {}),
      ...(content ? { content } : {})
    };

    const prev = byUrl.get(urlPath);
    if (!prev) {
      byUrl.set(urlPath, entry);
      return;
    }

    const prevScore = (prev.description ? prev.description.length : 0) + (prev.content ? prev.content.length : 0);
    const nextScore = (entry.description ? entry.description.length : 0) + (entry.content ? entry.content.length : 0);
    if (nextScore > prevScore) byUrl.set(urlPath, entry);
  });

  const pages = [...byUrl.values()];

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
