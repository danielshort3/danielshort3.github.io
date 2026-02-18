#!/usr/bin/env node
'use strict';

/*
  Inject shared head metadata into site HTML files.

  Currently:
  - Ensures Twitter card tags exist (title/description/image/alt) based on Open Graph tags.
  - Adds og:image width/height for the shared headshot image.
  - Injects JSON-LD WebApplication data for /tools/* pages.

  No external deps.
*/

const fs = require('fs');
const path = require('path');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');

const root = path.resolve(__dirname, '..');

const SHARED_OG_IMAGE = 'https://www.danielshort.me/img/hero/head.jpg';
const SHARED_OG_IMAGE_WIDTH = '558';
const SHARED_OG_IMAGE_HEIGHT = '558';
const SHARED_OG_IMAGE_ALT = 'Portrait photo of Daniel Short';
const SITE_ORIGIN = 'https://www.danielshort.me';
const noindexPathnames = loadNoindexPathnamesFromVercel(root);

const ROUTE_COMPONENT_STYLES = Object.freeze({
  '/short-links': ['css/components/short-links.css'],
  '/tools/short-links': ['css/components/short-links.css'],
  '/games/ocean-wave-simulation': ['css/components/ocean-wave-simulation.css'],
  '/tools/word-frequency': ['css/components/word-frequency.css'],
  '/tools/text-compare': ['css/components/text-compare.css'],
  '/tools/button-css-lab': ['css/components/button-css-lab.css'],
  '/tools/point-of-view-checker': ['css/components/point-of-view-checker.css'],
  '/tools/oxford-comma-checker': ['css/components/oxford-comma-checker.css'],
  '/tools/nbsp-cleaner': ['css/components/nbsp-cleaner.css'],
  '/tools/background-remover': ['css/components/background-remover.css'],
  '/tools/image-optimizer': ['css/components/image-optimizer.css'],
  '/tools/qr-code-generator': ['css/components/qr-code-generator.css'],
  '/tools/screen-recorder': ['css/components/screen-recorder.css'],
  '/tools/job-application-tracker': ['css/components/job-application-tracker.css'],
  '/tools/utm-batch-builder': ['css/components/utm-batch-builder.css'],
  '/tools/whisper-transcribe-monitor': ['css/components/whisper-transcribe-monitor.css'],
  '/tools/ga4-utm-performance': ['css/components/ga4-utm-performance.css']
});

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
  if (!hasTwitterAlt && (ogAlt || (ogImage === SHARED_OG_IMAGE ? SHARED_OG_IMAGE_ALT : ''))) {
    const alt = ogAlt || (ogImage === SHARED_OG_IMAGE ? SHARED_OG_IMAGE_ALT : '');
    if (alt) lines.push(`<meta name="twitter:image:alt" content="${alt}">`);
  }
  if (!lines.length) return headInner;

  const { index, indent } = findLineInsertionPoint(headInner);
  const block = lines.map((line) => `${indent}${line}`).join('\n');
  return headInner.slice(0, index) + '\n' + block + headInner.slice(index);
}

function ensureSharedOgImageDimensions(headInner) {
  const ogImage = getMetaContent(headInner, { property: 'og:image' });
  if (ogImage !== SHARED_OG_IMAGE) return headInner;

  const hasWidth = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:width"/i);
  const hasHeight = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:height"/i);
  const hasAlt = hasTag(headInner, /<meta\b[^>]*\bproperty="og:image:alt"/i);

  if (hasWidth && hasHeight && hasAlt) return headInner;

  const lines = [];
  if (!hasWidth) lines.push(`<meta property="og:image:width" content="${SHARED_OG_IMAGE_WIDTH}">`);
  if (!hasHeight) lines.push(`<meta property="og:image:height" content="${SHARED_OG_IMAGE_HEIGHT}">`);
  if (!hasAlt) lines.push(`<meta property="og:image:alt" content="${SHARED_OG_IMAGE_ALT}">`);

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

function ensureToolsStylesheet(headInner) {
  const canonical = getCanonicalHref(headInner);
  const pathname = toPathname(canonical);
  if (!needsToolsStyles(pathname)) return headInner;
  return ensureStylesheetLink(headInner, 'dist/styles-tools.css', ['dist/styles.css']);
}

function ensureRouteComponentStylesheet(headInner) {
  const canonical = getCanonicalHref(headInner);
  const pathname = toPathname(canonical);
  const hrefs = ROUTE_COMPONENT_STYLES[pathname];
  if (!Array.isArray(hrefs) || !hrefs.length) return headInner;

  return hrefs.reduce(
    (nextHead, href) => ensureStylesheetLink(nextHead, href, ['dist/styles-tools.css', 'dist/styles.css']),
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

function ensureToolJsonLd(headInner) {
  const withoutExisting = stripToolJsonLd(headInner);
  if (hasNoindexRobotsMeta(withoutExisting)) return withoutExisting;

  const canonical = getCanonicalHref(withoutExisting);
  if (!canonical || !canonical.startsWith(`${SITE_ORIGIN}/tools/`)) return withoutExisting;
  if (canonical === `${SITE_ORIGIN}/tools`) return withoutExisting;
  const pathname = toPathname(canonical);
  if (pathname && noindexPathnames.has(pathname)) return withoutExisting;

  const title = escapeTitleSuffix(getMetaContent(withoutExisting, { property: 'og:title' }) || getTitleText(withoutExisting));
  const description = getMetaContent(withoutExisting, { name: 'description' }) || getMetaContent(withoutExisting, { property: 'og:description' });
  if (!title || !description) return withoutExisting;

  const image = getMetaContent(withoutExisting, { property: 'og:image' }) || SHARED_OG_IMAGE;

  const json = {
    '@context': 'https://schema.org',
    '@type': 'WebApplication',
    name: title,
    description,
    url: canonical,
    image,
    applicationCategory: 'UtilitiesApplication',
    operatingSystem: 'Any',
    isAccessibleForFree: true,
    creator: {
      '@type': 'Person',
      name: 'Daniel Short',
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
  inner = dedupeMeta(inner, 'property', 'og:image:width');
  inner = dedupeMeta(inner, 'property', 'og:image:height');
  inner = dedupeMeta(inner, 'property', 'og:image:alt');
  inner = dedupeMeta(inner, 'name', 'twitter:title');
  inner = dedupeMeta(inner, 'name', 'twitter:description');
  inner = dedupeMeta(inner, 'name', 'twitter:image');
  inner = dedupeMeta(inner, 'name', 'twitter:image:alt');
  inner = ensureSharedOgImageDimensions(inner);
  inner = ensureTwitterMeta(inner);
  inner = ensureToolsStylesheet(inner);
  inner = ensureRouteComponentStylesheet(inner);
  inner = ensureToolJsonLd(inner);
  inner = dedupeMeta(inner, 'property', 'og:image:width');
  inner = dedupeMeta(inner, 'property', 'og:image:height');
  inner = dedupeMeta(inner, 'property', 'og:image:alt');
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
