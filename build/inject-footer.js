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
const FOOTER_AUDIENCES = ['personal', 'analytics', 'data-science', 'tourism'];

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function write(relPath, contents) {
  fs.writeFileSync(path.join(root, relPath), contents, 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function loadFooterTemplates() {
  const year = new Date().getFullYear();
  const templates = new Map();
  FOOTER_AUDIENCES.forEach((audience) => {
    const templatePath = audience === 'personal'
      ? footerTemplatePath
      : path.join(root, 'build', 'templates', `footer.${audience}.partial.html`);
    if (!fs.existsSync(templatePath)) return;
    const raw = fs.readFileSync(templatePath, 'utf8');
    templates.set(audience, raw.replace(/__YEAR__/g, String(year)).trim());
  });
  if (!templates.has('personal')) {
    throw new Error('Personal footer template is missing');
  }
  return templates;
}

function detectAudience(html) {
  const bodyMatch = String(html || '').match(/<body\b[^>]*\bdata-audience\s*=\s*["']([^"']+)["']/i);
  const audience = String(bodyMatch && bodyMatch[1] || 'personal').trim().toLowerCase();
  return FOOTER_AUDIENCES.includes(audience) ? audience : 'personal';
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

function replaceFooter(html, footerHtml) {
  // Replace the footer plus any previously injected shell widgets that sit
  // between the footer and the closing scripts/body so the step stays idempotent.
  const footerRe = /^([\t ]*)<footer\b[^>]*>[\s\S]*?<\/footer>(?:[\s\S]*?(?=^[\t ]*<script\b|^[\t ]*<\/body>|(?![\s\S])))?/im;
  const match = footerRe.exec(html);
  if (!match) return { html, changed: false };

  const indent = normalizeIndent(match[1]);
  const replacement = `${indentBlock(footerHtml, indent)}\n`;
  const next = html.replace(footerRe, replacement);
  return { html: next, changed: next !== html };
}

function insertMissingPersonalAccordionFooter(html, footerHtml) {
  if (!html.includes('data-personal-accordion-shell') || /<footer\b/i.test(html)) {
    return { html, changed: false };
  }

  const shellEnd = html.indexOf('<!-- personal-accordion-shell:end -->');
  if (shellEnd === -1) return { html, changed: false };
  const afterShell = shellEnd + '<!-- personal-accordion-shell:end -->'.length;
  const scriptOffset = html.slice(afterShell).search(/<script\b/i);
  const bodyEnd = html.lastIndexOf('</body>');
  const insertionPoint = scriptOffset === -1 ? bodyEnd : afterShell + scriptOffset;
  if (insertionPoint < 0) return { html, changed: false };

  const before = html.slice(0, insertionPoint).replace(/[\t ]+$/g, '');
  const separator = /\r?\n$/.test(before) ? '' : '\n';
  const next = `${before}${separator}${footerHtml}\n${html.slice(insertionPoint)}`;
  return { html: next, changed: true };
}

function main() {
  const footerTemplates = loadFooterTemplates();

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
    const audience = detectAudience(html);
    const footerHtml = footerTemplates.get(audience) || footerTemplates.get('personal');
    const replaced = replaceFooter(html, footerHtml);
    const rendered = replaced.changed
      ? replaced
      : insertMissingPersonalAccordionFooter(html, footerHtml);
    if (!rendered.changed) {
      skipped += 1;
      return;
    }
    write(relPath, rendered.html);
    updated += 1;
  });

  process.stdout.write(`[inject-footer] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
