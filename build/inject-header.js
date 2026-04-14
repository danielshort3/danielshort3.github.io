#!/usr/bin/env node
'use strict';

/*
  Inject a shared header/navigation into site HTML files.

  This replaces the runtime-injected header so:
  - Primary navigation is available without JavaScript.
  - Crawlers see consistent internal links.

  No external deps.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const headerTemplatePath = path.join(root, 'build', 'templates', 'header.partial.html');
const audienceApi = require(path.join(root, 'js', 'common', 'audience-config.js'));
const normalizeAudience = audienceApi && typeof audienceApi.normalizeAudience === 'function'
  ? audienceApi.normalizeAudience
  : ((value) => String(value || '').trim().toLowerCase() || 'analytics');
const getAudience = audienceApi && typeof audienceApi.getAudience === 'function'
  ? audienceApi.getAudience
  : (() => ({ brandNavPrimary: 'Data Analytics' }));
const detectAudienceFromPath = audienceApi && typeof audienceApi.detectAudienceFromPath === 'function'
  ? audienceApi.detectAudienceFromPath
  : (() => null);

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function write(relPath, contents) {
  fs.writeFileSync(path.join(root, relPath), contents, 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function loadHeaderTemplate() {
  const raw = fs.readFileSync(headerTemplatePath, 'utf8');
  return raw.trim();
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

function indentBlock(block, indent) {
  return block
    .split('\n')
    .map((line) => `${indent}${line}`.trimEnd())
    .join('\n');
}

function normalizeIndent(indent) {
  const raw = String(indent || '');
  if (!raw) return '';

  // Guard against runaway whitespace bloat if a file already contains an
  // abnormally long indent prefix (historically caused multi-megabyte HTML).
  if (raw.length > 24) return '  ';

  return raw;
}

function replaceHeader(html, headerHtml) {
  // Include the leading indent so we don't duplicate it on each build run.
  const headerRe = /^([\t ]*)<header\b[^>]*\bid=["']combined-header-nav["'][^>]*>[\s\S]*?<\/header>/im;
  const match = headerRe.exec(html);
  if (!match) return { html, changed: false };

  const indent = normalizeIndent(match[1]);
  const replacement = indentBlock(headerHtml, indent);
  const next = html.replace(headerRe, replacement);
  return { html: next, changed: next !== html };
}

function relPathToRoute(relPath) {
  const normalized = String(relPath || '').replace(/\\/g, '/');
  if (!normalized || normalized === 'index.html') return '/';
  if (normalized.startsWith('pages/')) {
    return `/${normalized.slice('pages/'.length).replace(/\.html$/i, '')}`;
  }
  return `/${normalized.replace(/\.html$/i, '')}`;
}

function detectAudienceForFile(html, relPath) {
  const bodyAudienceMatch = html.match(/\bdata-audience=(["'])([^"']+)\1/i);
  if (bodyAudienceMatch && bodyAudienceMatch[2]) {
    return normalizeAudience(bodyAudienceMatch[2]);
  }
  return detectAudienceFromPath(relPathToRoute(relPath));
}

function withAudienceBranding(headerHtml, audienceKey) {
  const audience = getAudience(audienceKey);
  const label = String(audience && audience.brandNavPrimary ? audience.brandNavPrimary : 'Data Analytics');
  return headerHtml.replace(
    /(<span class="brand-tagline-chunk" data-brand-tagline-primary="true">)([\s\S]*?)(<\/span>)/i,
    `$1${label}$3`
  );
}

function main() {
  const headerHtml = loadHeaderTemplate();

  const rootHtmlFiles = listRootHtmlFiles();
  const pagesHtmlFiles = walkHtmlFiles('pages');
  const targets = [...rootHtmlFiles, ...pagesHtmlFiles];

  let updated = 0;
  let skipped = 0;

  targets.forEach((absPath) => {
    const relPath = relFromRoot(absPath);

    // Only process real site pages.
    if (relPath === 'public' || relPath.startsWith('public/')) return;
    if (relPath === 'node_modules' || relPath.startsWith('node_modules/')) return;

    if (!exists(relPath)) return;
    const html = read(relPath);
    const audienceKey = detectAudienceForFile(html, relPath);
    const pageHeaderHtml = withAudienceBranding(headerHtml, audienceKey);
    const replaced = replaceHeader(html, pageHeaderHtml);
    if (!replaced.changed) {
      skipped += 1;
      return;
    }
    write(relPath, replaced.html);
    updated += 1;
  });

  process.stdout.write(`[inject-header] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
