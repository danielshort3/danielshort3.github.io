#!/usr/bin/env node
'use strict';

/*
  Keep internal site navigation in the same tab while preserving new-tab behavior
  for demos, documents, and external destinations.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const SITE_ORIGIN = 'https://www.danielshort.me';

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

function shouldKeepNewTab(href) {
  const raw = String(href || '').trim();
  if (!raw || raw.startsWith('#') || raw.startsWith('mailto:') || raw.startsWith('tel:') || raw.startsWith('javascript:')) {
    return true;
  }

  let pathname = raw;
  if (/^https?:\/\//i.test(raw)) {
    try {
      const url = new URL(raw, SITE_ORIGIN);
      if (url.origin !== SITE_ORIGIN) return true;
      pathname = url.pathname || '/';
    } catch {
      return true;
    }
  } else {
    pathname = raw.startsWith('/') ? raw : `/${raw.replace(/^\.?\//, '')}`;
  }

  if (/^\/documents\//i.test(pathname)) return true;
  if (/\/(?:demos\/|[^/]*-demo(?:\.html)?)(?:$|[?#/])/i.test(pathname)) return true;
  if (/^\/games\/(?:stellar-dogfight|slot-machine|roulette|probability-engine)(?:\.html)?(?:$|[?#])/i.test(pathname)) return true;
  return false;
}

function normalizeRelAttribute(tag) {
  return tag.replace(/\s+rel="([^"]*)"/i, (match, value) => {
    const kept = String(value || '')
      .split(/\s+/)
      .map((token) => token.trim())
      .filter(Boolean)
      .filter((token) => !/^(noopener|noreferrer)$/i.test(token));
    return kept.length ? ` rel="${kept.join(' ')}"` : '';
  });
}

function processHtml(html) {
  const next = String(html || '').replace(/<a\b[^>]*\bhref="([^"]+)"[^>]*>/gi, (tag, href) => {
    if (shouldKeepNewTab(href)) return tag;
    if (!/\btarget="_blank"/i.test(tag)) return tag;

    let normalized = tag.replace(/\s+target="_blank"/i, '');
    normalized = normalizeRelAttribute(normalized);
    return normalized;
  });

  return { html: next, changed: next !== html };
}

function main() {
  const targets = [...listRootHtmlFiles(), ...walkHtmlFiles('pages')];
  let updated = 0;
  let skipped = 0;

  targets.forEach((absPath) => {
    const relPath = relFromRoot(absPath);
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

  process.stdout.write(`[normalize-internal-links] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
