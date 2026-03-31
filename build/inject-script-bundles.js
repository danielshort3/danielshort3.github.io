#!/usr/bin/env node
'use strict';

/*
  Replace stable/raw script tags in HTML with hashed dist bundle references.
  This keeps authored HTML simple while letting builds serve immutable bundles.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const manifestPath = path.join(root, 'dist', 'scripts-manifest.json');
const manifest = loadManifest();

const managedHrefs = {
  shell: resolveHref('site-shell.js', manifest.shell),
  consent: resolveHref('site-consent.js', manifest.consent),
  home: resolveHref('site-home.js', manifest.home),
  contact: resolveHref('site-contact.js', manifest.contact),
  search: resolveHref('site-search.js', manifest.search),
  contributions: resolveHref('site-contributions.js', manifest.contributions),
  sitemap: resolveHref('site-sitemap.js', manifest.sitemap),
  privacy: resolveHref('site-privacy.js', manifest.privacy),
  toolsAccount: resolveHref('site-tools-account.js', manifest.toolsAccount),
  toolsLanding: resolveHref('site-tools-landing.js', manifest.toolsLanding)
};

function loadManifest() {
  try {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  } catch {
    return {};
  }
}

function resolveHref(fallbackName, manifestName) {
  const rel = String(manifestName || '').trim();
  if (!rel) return `dist/${fallbackName}`;
  return `dist/${rel.replace(/^dist\//i, '')}`;
}

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

function lineIndent(line) {
  const match = /^(\s*)/.exec(String(line || ''));
  return match ? match[1] : '';
}

function isManagedLine(trimmed, baseName) {
  return new RegExp(`^<script\\s+defer\\s+src="dist\\/${baseName}(?:\\.[0-9a-f]{8})?\\.js"(?:\\s+[^>]*)?><\\/script>$`, 'i').test(trimmed);
}

function processHtml(html, relPath) {
  const lines = String(html || '').split(/\r?\n/);
  const out = [];

  let shellInserted = false;
  let consentInserted = false;
  let contributionsInserted = false;
  let toolsInserted = false;

  const isToolsLanding = relPath === 'pages/tools.html';

  lines.forEach((line) => {
    const trimmed = line.trim();
    const indent = lineIndent(line);

    if (
      /^<script\s+defer\s+src="js\/common\/common\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/navigation\/navigation\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/animations\/animations\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-shell')
    ) {
      if (!shellInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.shell}"></script>`);
        shellInserted = true;
      }
      return;
    }

    if (
      /^<script\s+src="js\/privacy\/config\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/privacy\/consent_manager\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-consent')
    ) {
      if (!consentInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.consent}"></script>`);
        consentInserted = true;
      }
      return;
    }

    if (
      /^<script\s+defer\s+src="js\/contributions\/contributions-data\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/contributions\/contributions\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/contributions\/carousel\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-contributions')
    ) {
      if (!contributionsInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.contributions}"></script>`);
        contributionsInserted = true;
      }
      return;
    }

    if (
      /^<script\s+defer\s+src="js\/accounts\/tools-config\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/accounts\/tools-auth\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/accounts\/tools-state\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/accounts\/tools-account-ui\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-tools-account')
      || /^<script\s+defer\s+src="dist\/site-tools-landing(?:\.[0-9a-f]{8})?\.js"(?:\s+[^>]*)?><\/script>$/i.test(trimmed)
    ) {
      if (!toolsInserted) {
        if (isToolsLanding) {
          out.push(`${indent}<script defer src="${managedHrefs.toolsLanding}" data-tools-account-src="${managedHrefs.toolsAccount}"></script>`);
        } else {
          out.push(`${indent}<script defer src="${managedHrefs.toolsAccount}"></script>`);
        }
        toolsInserted = true;
      }
      return;
    }

    if (/^<script\s+defer\s+src="js\/common\/certifications-modal\.js"><\/script>$/i.test(trimmed) || isManagedLine(trimmed, 'site-home')) {
      out.push(`${indent}<script defer src="${managedHrefs.home}"></script>`);
      return;
    }

    if (/^<script\s+defer\s+src="js\/forms\/contact\.js"><\/script>$/i.test(trimmed) || isManagedLine(trimmed, 'site-contact')) {
      out.push(`${indent}<script defer src="${managedHrefs.contact}"></script>`);
      return;
    }

    if (/^<script\s+defer\s+src="js\/search\/site-search\.js"><\/script>$/i.test(trimmed) || isManagedLine(trimmed, 'site-search')) {
      out.push(`${indent}<script defer src="${managedHrefs.search}"></script>`);
      return;
    }

    if (/^<script\s+defer\s+src="js\/sitemap\/sitemap-page\.js"><\/script>$/i.test(trimmed) || isManagedLine(trimmed, 'site-sitemap')) {
      out.push(`${indent}<script defer src="${managedHrefs.sitemap}"></script>`);
      return;
    }

    if (/^<script\s+defer\s+src="js\/privacy\/privacy-preferences\.js"><\/script>$/i.test(trimmed) || isManagedLine(trimmed, 'site-privacy')) {
      out.push(`${indent}<script defer src="${managedHrefs.privacy}"></script>`);
      return;
    }

    out.push(line);
  });

  const next = out.join('\n');
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
    const processed = processHtml(html, relPath);
    if (!processed.changed) {
      skipped += 1;
      return;
    }
    write(relPath, processed.html);
    updated += 1;
  });

  process.stdout.write(`[inject-script-bundles] Updated ${updated} file(s); skipped ${skipped}.\n`);
}

main();
