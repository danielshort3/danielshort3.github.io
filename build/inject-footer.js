#!/usr/bin/env node
'use strict';

/*
  Inject a shared footer into site HTML files.

  This replaces the runtime-injected footer so:
  - The footer is available without JavaScript.
  - Privacy settings / Do Not Sell links are always present.
  - Sitemap is discoverable from every page.

  No external deps.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const footerTemplatePath = path.join(root, 'build', 'templates', 'footer.partial.html');

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function write(relPath, contents) {
  fs.writeFileSync(path.join(root, relPath), contents, 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function loadFooterTemplate() {
  const year = new Date().getFullYear();
  const raw = fs.readFileSync(footerTemplatePath, 'utf8');
  return raw.replace(/__YEAR__/g, String(year)).trim();
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

function replaceFooter(html, footerHtml) {
  const footerRe = /<footer\b[^>]*>[\s\S]*?<\/footer>/i;
  const match = footerRe.exec(html);
  if (!match) return { html, changed: false };

  const startIndex = match.index;
  const lineStart = html.lastIndexOf('\n', startIndex) + 1;
  const indent = html.slice(lineStart, startIndex).match(/^[\t ]*/)?.[0] || '';

  const replacement = indentBlock(footerHtml, indent);
  const next = html.replace(footerRe, replacement);
  return { html: next, changed: next !== html };
}

function main() {
  const footerHtml = loadFooterTemplate();

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
    const replaced = replaceFooter(html, footerHtml);
    if (!replaced.changed) {
      skipped += 1;
      return;
    }
    write(relPath, replaced.html);
    updated += 1;
  });

  process.stdout.write(`[inject-footer] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
