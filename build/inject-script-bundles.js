#!/usr/bin/env node
'use strict';

/*
  Replace stable/raw script tags in HTML with hashed dist bundle references.
  This keeps authored HTML simple while letting builds serve immutable bundles.
*/

const fs = require('fs');
const path = require('path');
const {
  finalizePersonalRouteDocument,
  validatePersonalRouteDocument
} = require('./lib/personal-accordion-shell');

const root = path.resolve(__dirname, '..');
const manifestPath = path.join(root, 'dist', 'scripts-manifest.json');
const manifest = loadManifest();
const TRANSIENT_WRITE_ERROR_CODES = new Set(['EACCES', 'EBUSY', 'EPERM', 'UNKNOWN']);
const WRITE_RETRY_DELAYS_MS = Object.freeze([25, 50, 100, 200, 400, 800]);
const writeRetrySignal = new Int32Array(new SharedArrayBuffer(4));

const managedHrefs = {
  shell: resolveHref('site-shell.js', manifest.shell),
  home: resolveHref('site-home.js', manifest.home),
  consent: resolveHref('site-consent.js', manifest.consent),
  contact: resolveHref('site-contact.js', manifest.contact),
  search: resolveHref('site-search.js', manifest.search),
  contributions: resolveHref('site-contributions.js', manifest.contributions),
  sitemap: resolveHref('site-sitemap.js', manifest.sitemap),
  privacy: resolveHref('site-privacy.js', manifest.privacy),
  toolsAccount: resolveHref('site-tools-account.js', manifest.toolsAccount),
  toolsLanding: resolveHref('site-tools-landing.js', manifest.toolsLanding),
  projectStarfall: resolveHref('project-starfall.js', manifest.projectStarfall)
};
const SITE_SHELL_BUNDLE_PATTERN = /<script\b[^>]*\bsrc=(["'])\/?dist\/site-shell(?:\.[0-9a-f]{8})?\.js\1[^>]*>\s*<\/script>/gi;

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
  const filePath = path.join(root, relPath);
  for (let attempt = 0; ; attempt += 1) {
    try {
      fs.writeFileSync(filePath, contents, 'utf8');
      return;
    } catch (error) {
      const canRetry = TRANSIENT_WRITE_ERROR_CODES.has(error && error.code) &&
        attempt < WRITE_RETRY_DELAYS_MS.length;
      if (!canRetry) throw error;
      Atomics.wait(writeRetrySignal, 0, 0, WRITE_RETRY_DELAYS_MS[attempt]);
    }
  }
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

function isVisitorFacingSiteDocument(html, relPath = '') {
  const normalizedPath = String(relPath || '').replace(/\\/g, '/');
  const isManagedVisitorPath = normalizedPath && (
    !normalizedPath.includes('/') || normalizedPath.startsWith('pages/')
  );
  if (!isManagedVisitorPath || normalizedPath.startsWith('admin/') || normalizedPath.startsWith('demos/')) {
    return false;
  }

  const source = String(html || '');
  return /<html\b/i.test(source) &&
    /<head\b/i.test(source) &&
    /<main\b/i.test(source) &&
    (
      /<header\b[^>]*\bid=(["'])combined-header-nav\1/i.test(source) ||
      /\bdata-personal-accordion-shell\b/i.test(source) ||
      /<body\b[^>]*\bclass=(["'])[^"']*\bsite-page\b[^"']*\1/i.test(source)
    );
}

function isSiteShellBundleLine(trimmed) {
  return /^<script\b[^>]*\bsrc=(["'])\/?dist\/site-shell(?:\.[0-9a-f]{8})?\.js\1[^>]*>\s*<\/script>$/i.test(trimmed);
}

function ensureSiteShellBundleInHead(lines) {
  const source = (Array.isArray(lines) ? lines : []).join('\n');
  const headOpen = /<head\b[^>]*>/i.exec(source);
  if (!headOpen) return Array.isArray(lines) ? lines.slice() : [];
  const headStart = headOpen.index + headOpen[0].length;
  const headEnd = source.indexOf('</head>', headStart);
  if (headEnd === -1) return Array.isArray(lines) ? lines.slice() : [];

  const canonicalScript = `<script defer src="${managedHrefs.shell}"></script>`;
  let keptShell = false;
  let headInner = source.slice(headStart, headEnd).replace(SITE_SHELL_BUNDLE_PATTERN, () => {
    if (keptShell) return '';
    keptShell = true;
    return canonicalScript;
  });

  if (!keptShell) {
    const personalStylesheet = /^([ \t]*)<link\b[^>]*href="(?:\/?dist\/)?styles-personal-accordion(?:\.[0-9a-f]{8})?\.css"[^>]*>[ \t]*$/im.exec(headInner);
    if (personalStylesheet) {
      const scriptLine = `${personalStylesheet[1]}${canonicalScript}\n`;
      headInner = headInner.slice(0, personalStylesheet.index) + scriptLine + headInner.slice(personalStylesheet.index);
    } else {
      headInner = `${headInner.trimEnd()}\n  ${canonicalScript}\n`;
    }
  }

  const beforeHead = source.slice(0, headStart).replace(SITE_SHELL_BUNDLE_PATTERN, '');
  const afterHead = source.slice(headEnd).replace(SITE_SHELL_BUNDLE_PATTERN, '');
  return `${beforeHead}${headInner}${afterHead}`.split('\n');
}

function insertManagedScript(lines, scriptLine) {
  const next = Array.isArray(lines) ? lines.slice() : [];
  const scriptIndex = next.findIndex((line) => /^\s*<script\b/i.test(String(line || '')));
  if (scriptIndex !== -1) {
    const indent = lineIndent(next[scriptIndex]);
    next.splice(scriptIndex, 0, `${indent}${scriptLine}`);
    return next;
  }

  const bodyIndex = next.findIndex((line) => /^\s*<\/body>\s*$/i.test(String(line || '')));
  if (bodyIndex !== -1) {
    const indent = lineIndent(next[bodyIndex]);
    next.splice(bodyIndex, 0, `${indent}${scriptLine}`);
    return next;
  }

  next.push(scriptLine);
  return next;
}

function insertManagedScriptBefore(lines, scriptLine, predicate) {
  const next = Array.isArray(lines) ? lines.slice() : [];
  const insertIndex = next.findIndex((line) => predicate(String(line || '').trim()));
  if (insertIndex !== -1) {
    const indent = lineIndent(next[insertIndex]);
    next.splice(insertIndex, 0, `${indent}${scriptLine}`);
    return next;
  }
  return insertManagedScript(next, scriptLine);
}

function processHtml(html, relPath) {
  const lines = String(html || '').split(/\r?\n/);
  const out = [];

  let shellInserted = false;
  let homeInserted = false;
  let consentInserted = false;
  let contributionsInserted = false;
  let toolsInserted = false;
  let projectStarfallInserted = false;

  const isToolsLanding = relPath === 'pages/tools.html';
  const isProjectStarfall = relPath === 'pages/games/project-starfall.html';
  const isIsolatedCaptureSurface = relPath === 'pages/job-application-tracker.html';

  lines.forEach((line) => {
    const trimmed = line.trim();
    const indent = lineIndent(line);

    if (isProjectStarfall && (
      /^<script\s+defer\s+src="js\/games\/project-starfall\/.+\.js(?:\?[^"']*)?"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'project-starfall')
    )) {
      return;
    }

    if (isProjectStarfall && /^<script\s+defer\s+src="js\/vendor\/pixi\.min\.js(?:\?[^"']*)?"><\/script>$/i.test(trimmed)) {
      out.push(line);
      if (!projectStarfallInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.projectStarfall}"></script>`);
        projectStarfallInserted = true;
      }
      return;
    }

    if (
      /^<script\s+defer\s+src="js\/common\/common\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/navigation\/navigation\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/animations\/animations\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-shell')
      || isSiteShellBundleLine(trimmed)
    ) {
      if (!shellInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.shell}"></script>`);
        shellInserted = true;
      }
      return;
    }

    if (
      /^<script\s+defer\s+src="js\/home\/category-accordion\.js(?:\?[^"']*)?"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-home')
    ) {
      if (relPath === 'index.html' && !homeInserted) {
        out.push(`${indent}<script defer src="${managedHrefs.home}" data-tools-account-src="${managedHrefs.toolsAccount}"></script>`);
        homeInserted = true;
      }
      return;
    }

    if (
      /^<script\s+src="js\/privacy\/config\.js"><\/script>$/i.test(trimmed)
      || /^<script\s+defer\s+src="js\/privacy\/consent_manager\.js"><\/script>$/i.test(trimmed)
      || isManagedLine(trimmed, 'site-consent')
    ) {
      if (isIsolatedCaptureSurface) return;
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

  let normalized = out;
  const needsShellBundle = isVisitorFacingSiteDocument(html, relPath);
  const needsConsentBundle = !isIsolatedCaptureSurface && (
    needsShellBundle
      || !relPath.includes('/')
      || relPath.startsWith('pages/')
      || relPath.startsWith('demos/')
  );
  if (needsShellBundle && !shellInserted) {
    normalized = insertManagedScript(normalized, `<script defer src="${managedHrefs.shell}"></script>`);
  }
  if (relPath === 'index.html' && !homeInserted) {
    normalized = insertManagedScript(
      normalized,
      `<script defer src="${managedHrefs.home}" data-tools-account-src="${managedHrefs.toolsAccount}"></script>`
    );
  }
  if (needsConsentBundle && !normalized.some((line) => isManagedLine(String(line || '').trim(), 'site-consent'))) {
    normalized = insertManagedScript(normalized, `<script defer src="${managedHrefs.consent}"></script>`);
  }

  const requiredPageBundles = [];
  if (relPath === 'pages/contact.html') requiredPageBundles.push(['site-contact', managedHrefs.contact]);
  if (relPath === 'pages/search.html') requiredPageBundles.push(['site-search', managedHrefs.search]);
  if (relPath === 'pages/contributions.html') requiredPageBundles.push(['site-contributions', managedHrefs.contributions]);
  if (relPath === 'pages/sitemap-pretty.html' || relPath === 'pages/sitemap.html') {
    requiredPageBundles.push(['site-sitemap', managedHrefs.sitemap]);
  }
  if (relPath === 'pages/privacy.html') requiredPageBundles.push(['site-privacy', managedHrefs.privacy]);

  requiredPageBundles.forEach(([baseName, href]) => {
    const hasBundle = normalized.some((line) => isManagedLine(String(line || '').trim(), baseName));
    if (hasBundle) return;
    normalized = insertManagedScriptBefore(
      normalized,
      `<script defer src="${href}"></script>`,
      (trimmed) => isManagedLine(trimmed, 'site-consent') || /^<script\s+src="js\/privacy\/config\.js"><\/script>$/i.test(trimmed) || /^<script\s+defer\s+src="js\/privacy\/consent_manager\.js"><\/script>$/i.test(trimmed)
    );
  });

  // The homepage owns a lightweight, lazy account loader in the site-home
  // bundle so opening the Tools library does not eagerly pay for account UI.
  const needsToolsAccountBundle = relPath !== 'index.html' &&
    (/data-tools-account="dock"/i.test(html) || /data-tools-account="dock-inner"/i.test(html));
  if (needsToolsAccountBundle && !toolsInserted) {
    const toolsScript = relPath === 'pages/tools.html'
      ? `<script defer src="${managedHrefs.toolsLanding}" data-tools-account-src="${managedHrefs.toolsAccount}"></script>`
      : `<script defer src="${managedHrefs.toolsAccount}"></script>`;
    normalized = insertManagedScriptBefore(
      normalized,
      toolsScript,
      (trimmed) => isManagedLine(trimmed, 'site-consent') || /^<script\s+src="js\/privacy\/config\.js"><\/script>$/i.test(trimmed) || /^<script\s+defer\s+src="js\/privacy\/consent_manager\.js"><\/script>$/i.test(trimmed)
    );
  }

  if (isProjectStarfall && !projectStarfallInserted) {
    normalized = insertManagedScriptBefore(
      normalized,
      `<script defer src="${managedHrefs.projectStarfall}"></script>`,
      (trimmed) => isManagedLine(trimmed, 'site-shell') || isManagedLine(trimmed, 'site-consent')
    );
  }

  if (needsShellBundle) {
    normalized = ensureSiteShellBundleInHead(normalized);
  }

  const next = normalized.join('\n').replace(/^[ \t]+$/gm, '');
  const finalized = finalizePersonalRouteDocument(next, { home: relPath === 'index.html' });
  if (/\bid="site-route-manifest"/i.test(finalized)) validatePersonalRouteDocument(finalized);
  return { html: finalized, changed: finalized !== html };
}

function main() {
  const targets = [...listRootHtmlFiles(), ...walkHtmlFiles('pages'), ...walkHtmlFiles('demos')];
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

if (require.main === module) main();

module.exports = {
  ensureSiteShellBundleInHead,
  isVisitorFacingSiteDocument,
  processHtml
};
