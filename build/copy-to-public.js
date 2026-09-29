#!/usr/bin/env node
/*
  Prepare Vercel output: copy static site into ./public
  - Copies root HTML files and selected assets directories
  - Ensures ./public exists and is clean
*/
const fs = require('fs');
const path = require('path');
const { BROWSER_PROJECT_DOCUMENTS } = require('./lib/browser-project-assets');
const { copyAppUpdateFeeds } = require('./lib/app-update-feeds.cjs');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'public');
const cssManifestPath = path.join(root, 'dist', 'styles-manifest.json');
const jsManifestPath = path.join(root, 'dist', 'scripts-manifest.json');
const includeStarfallBackups = /^(1|true|yes)$/i.test(String(process.env.PUBLIC_INCLUDE_STARFALL_BACKUPS || '').trim());

function log(msg){
  process.stdout.write(msg + '\n');
}

function sleepSync(ms){
  const waitMs = Math.max(0, Number(ms) || 0);
  if (!waitMs) return;
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, waitMs);
}

function removeWithRetries(target){
  const transientCodes = new Set(['EBUSY', 'ENOTEMPTY', 'EPERM']);
  let lastError = null;
  for (let attempt = 0; attempt < 12; attempt += 1) {
    try {
      fs.rmSync(target, {
        recursive: true,
        force: true,
        maxRetries: 5,
        retryDelay: 150
      });
      return;
    } catch (err) {
      lastError = err;
      if (!transientCodes.has(err && err.code)) throw err;
      sleepSync(180 * (attempt + 1));
    }
  }
  throw lastError;
}

function ensureCleanDir(dir){
  fs.mkdirSync(dir, { recursive: true });
  fs.readdirSync(dir).forEach((entry) => {
    removeWithRetries(path.join(dir, entry));
  });
}

function copyFile(src, dest){
  const transientCodes = new Set(['EBUSY', 'ENOENT', 'EPERM']);
  let lastError = null;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      fs.mkdirSync(path.dirname(dest), { recursive: true });
      fs.copyFileSync(src, dest);
      return;
    } catch (err) {
      lastError = err;
      if (!transientCodes.has(err && err.code)) throw err;
      sleepSync(120 * (attempt + 1));
    }
  }
  throw lastError;
}

function shouldSkipPublicCopy(absPath) {
  const rel = path.relative(root, absPath).replace(/\\/g, '/');
  return rel === 'img/project-starfall/review'
    || rel.startsWith('img/project-starfall/review/')
    || (!includeStarfallBackups && (rel === 'img/project-starfall/backups' || rel.startsWith('img/project-starfall/backups/')))
    || /^img\/project-starfall\/(?:.+\/)?source(?:\/|$)/.test(rel);
}

function copyDir(src, dest){
  if (!fs.existsSync(src) || shouldSkipPublicCopy(src)) return;

  let entries;
  try {
    entries = fs.readdirSync(src, { withFileTypes: true });
  } catch {
    return;
  }

  fs.mkdirSync(dest, { recursive: true });
  entries.forEach((entry) => {
    const entrySrc = path.join(src, entry.name);
    if (shouldSkipPublicCopy(entrySrc)) return;
    const entryDest = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(entrySrc, entryDest);
      return;
    }
    if (entry.isFile()) copyFile(entrySrc, entryDest);
  });
}

function isSafeDistArtifactName(name) {
  const value = String(name || '').trim();
  if (!value) return false;
  if (value.includes('/') || value.includes('\\')) return false;
  if (value.includes('..')) return false;
  return true;
}

function collectDistArtifacts(cssManifest, jsManifest) {
  const artifacts = new Set([
    'ai-digest-manifest.json',
    'styles.css',
    'styles-home.css',
    'styles-workbench.css',
    'styles-tools.css',
    'styles-manifest.json',
    'scripts-manifest.json',
    'chatbot-knowledge.json',
    'search-index.json',
    'shortlinks-destinations.json'
  ]);

  if (cssManifest && typeof cssManifest === 'object') {
    Object.values(cssManifest).forEach((value) => {
      if (typeof value === 'string') artifacts.add(value);
    });
  }

  if (jsManifest && typeof jsManifest === 'object') {
    Object.values(jsManifest).forEach((value) => {
      if (typeof value === 'string') artifacts.add(value);
    });
  }

  [
    'site-shell.js',
    'site-home.js',
    'site-consent.js',
    'site-contact.js',
    'site-search.js',
    'site-sitemap.js',
    'site-privacy.js',
    'site-tools-account.js',
    'site-tools-landing.js'
  ].forEach((fileName) => artifacts.add(fileName));

  return [...artifacts]
    .map((name) => String(name || '').trim())
    .filter(isSafeDistArtifactName)
    .sort();
}

function copyDistArtifacts(cssManifest, jsManifest) {
  const sourceDir = path.join(root, 'dist');
  const destinationDir = path.join(outDir, 'dist');
  fs.mkdirSync(destinationDir, { recursive: true });

  const artifacts = collectDistArtifacts(cssManifest, jsManifest);
  let copied = 0;
  let missing = 0;

  artifacts.forEach((name) => {
    const src = path.join(sourceDir, name);
    let stat;
    try {
      stat = fs.statSync(src);
    } catch {
      missing += 1;
      return;
    }
    if (!stat.isFile()) {
      missing += 1;
      return;
    }
    copyFile(src, path.join(destinationDir, name));
    copied += 1;
  });

  log(`Copied ${copied} dist artifact(s)${missing ? `; ${missing} missing.` : '.'}`);
}

function listRootHtmlFiles(base){
  return fs.readdirSync(base)
    .filter(f => f.endsWith('.html'))
    .map(f => path.join(base, f));
}

function readJson(filePath) {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function listHtmlFilesRecursive(dirPath) {
  const results = [];
  const stack = [dirPath];
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
      if (entry.isFile() && entry.name.endsWith('.html')) results.push(full);
    });
  }
  return results;
}

function resolveApprovedDocuments(sourceRoot) {
  return BROWSER_PROJECT_DOCUMENTS.map((relPath) => {
    const src = path.join(sourceRoot, relPath);
    let stat;
    try {
      stat = fs.statSync(src);
    } catch {
      throw new Error(`Missing browser-project asset: ${relPath}`);
    }
    if (!stat.isFile()) throw new Error(`Missing browser-project asset: ${relPath}`);
    return { relPath, src, size: stat.size };
  });
}

function copyApprovedDocuments(documents, destinationRoot) {
  documents.forEach(({ relPath, src }) => {
    copyFile(src, path.join(destinationRoot, relPath));
  });
  const totalBytes = documents.reduce((sum, document) => sum + document.size, 0);
  log(`Copied ${documents.length} approved documents (${Math.round(totalBytes / 1024)} KB).`);
}

function rewriteCssLinksInHtml(html, cssHrefs) {
  if (!html) return html;
  let next = html;
  if (cssHrefs.base) {
    next = next.replace(/href=(["'])dist\/styles\.css\1/g, `href="dist/${cssHrefs.base}"`);
  }
  if (cssHrefs.home) {
    next = next.replace(/href=(["'])dist\/styles-home\.css\1/g, `href="dist/${cssHrefs.home}"`);
  }
  if (cssHrefs.workbench) {
    next = next.replace(/href=(["'])dist\/styles-workbench\.css\1/g, `href="dist/${cssHrefs.workbench}"`);
  }
  if (cssHrefs.tools) {
    next = next.replace(/href=(["'])dist\/styles-tools\.css\1/g, `href="dist/${cssHrefs.tools}"`);
  }
  return next;
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeHtmlAttribute(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function decodeHtmlAttribute(value) {
  return String(value || '')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

function getHtmlAttribute(tag, name) {
  const matcher = new RegExp(`${escapeRegExp(name)}\\s*=\\s*(["'])(.*?)\\1`, 'i');
  const match = matcher.exec(String(tag || ''));
  return match ? decodeHtmlAttribute(match[2]) : '';
}

function setHtmlAttribute(tag, name, value) {
  const attrName = escapeRegExp(name);
  const escaped = escapeHtmlAttribute(value);
  const matcher = new RegExp(`\\s${attrName}\\s*=\\s*(["'])[^"']*\\1`, 'i');
  if (matcher.test(tag)) {
    return tag.replace(matcher, ` ${name}="${escaped}"`);
  }
  return tag.replace(/>$/, ` ${name}="${escaped}">`);
}

function normalizeGoogleMapsZoom(value) {
  const zoom = Number(value);
  if (!Number.isFinite(zoom)) return 10;
  return Math.max(0, Math.min(21, Math.round(zoom)));
}

function readGoogleMapsApiKey() {
  const envKey = String(process.env.GOOGLE_MAPS_API_KEY || process.env.GOOGLE_MAPS_EMBED_API_KEY || '').trim();
  if (envKey) return envKey;

  const keyPath = path.join(root, 'google_maps_api_key.txt');
  try {
    return fs.readFileSync(keyPath, 'utf8').trim();
  } catch {
    return '';
  }
}

function buildGoogleMapsEmbedUrl(apiKey, address, zoom) {
  const params = new URLSearchParams();
  params.set('key', apiKey);
  params.set('q', String(address || 'Grand Junction, CO').trim() || 'Grand Junction, CO');
  params.set('zoom', String(normalizeGoogleMapsZoom(zoom)));
  return `https://www.google.com/maps/embed/v1/place?${params.toString()}`;
}

function rewriteGoogleMapsEmbedsInHtml(html, apiKey) {
  let count = 0;
  const next = String(html || '').replace(/<iframe\b[^>]*\bdata-google-maps-iframe\b[^>]*>/gi, (tag) => {
    const address = getHtmlAttribute(tag, 'data-google-maps-address') || 'Grand Junction, CO';
    const zoom = getHtmlAttribute(tag, 'data-google-maps-zoom') || 10;
    count += 1;
    return setHtmlAttribute(tag, 'src', buildGoogleMapsEmbedUrl(apiKey, address, zoom));
  });
  return { html: next, count };
}

function rewriteGoogleMapsEmbedsInPublic() {
  const apiKey = readGoogleMapsApiKey();
  const publicHtmlFiles = listHtmlFilesRecursive(outDir);
  let iframeCount = 0;
  let fileCount = 0;

  publicHtmlFiles.forEach((filePath) => {
    let html;
    try {
      html = fs.readFileSync(filePath, 'utf8');
    } catch {
      return;
    }
    if (!html.includes('data-google-maps-iframe')) return;

    if (!apiKey) {
      iframeCount += (html.match(/\bdata-google-maps-iframe\b/g) || []).length;
      return;
    }

    const rewritten = rewriteGoogleMapsEmbedsInHtml(html, apiKey);
    iframeCount += rewritten.count;
    if (rewritten.count && rewritten.html !== html) {
      try {
        fs.writeFileSync(filePath, rewritten.html, 'utf8');
        fileCount += 1;
      } catch {}
    }
  });

  if (!iframeCount) return;
  if (!apiKey) {
    log(`Google Maps API key not found; left ${iframeCount} fallback map iframe(s) in public/.`);
    return;
  }
  log(`Rewrote ${iframeCount} Google Maps iframe(s) with Maps Embed API URLs in ${fileCount} public HTML file(s).`);
}

function copyStatic(){
  // Fail before clearing public/ if a clean checkout is missing a required
  // self-hosted project download; never silently fall back to an AWS URL.
  const approvedDocuments = resolveApprovedDocuments(root);
  ensureCleanDir(outDir);
  const cssManifest = readJson(cssManifestPath);
  const jsManifest = readJson(jsManifestPath);

  // Keep Contact behind its clean-URL rewrite. A root contact.html file
  // takes precedence when Vercel clean URLs are enabled.
  const htmlFiles = listRootHtmlFiles(root);
  htmlFiles.forEach(src => {
    const rel = path.relative(root, src);
    if (rel === 'contact.html') return;
    copyFile(src, path.join(outDir, rel));
  });

  // Copy selected root-level static files if present
  const rootFiles = ['robots.txt', 'sitemap.xml', 'sitemap.xsl', 'llms.txt', 'favicon.ico', 'manifest.json', 'sw.js'];
  rootFiles.forEach(name => {
    const src = path.join(root, name);
    if (fs.existsSync(src)) copyFile(src, path.join(outDir, name));
  });

  // Copy asset and content directories used by the site, including the nested
  // demos/data snapshots and the browser calculation scripts under js/demos.
  // Dist artifacts are handled separately via an explicit whitelist.
  // Recursive img copying includes library icons, img/games/ocean HDR skies, and img/social cards.
  // Recorded ocean soundscapes are loaded on demand from audio/ocean.
  const dirs = ['img', 'js', 'css', 'pages', 'demos', 'audio'];
  dirs.forEach(d => copyDir(path.join(root, d), path.join(outDir, d)));
  copyApprovedDocuments(approvedDocuments, outDir);
  copyDistArtifacts(cssManifest, jsManifest);
  // Publish only the explicit native catalog, never other build files.
  copyFile(path.join(root, 'dist', 'app-content', 'v1', 'catalog.json'), path.join(outDir, 'app-content', 'v1', 'catalog.json'));
  log(`Copied ${copyAppUpdateFeeds(root, outDir)} approved Android update feed(s).`);
  copyDir(path.join(root, 'dist', 'ai-pages'), path.join(outDir, 'dist', 'ai-pages'));
  // Keep generated legacy files behind their permanent redirects.
  removeWithRetries(path.join(outDir, 'js', 'common', 'audience-config.js'));
  removeWithRetries(path.join(outDir, 'pages', 'destination-analytics.html'));

  // Rewrite public HTML to reference the hashed CSS bundle (better caching).
  const cssHrefs = {
    base: cssManifest && typeof cssManifest.file === 'string' ? cssManifest.file : null,
    home: cssManifest && typeof cssManifest.homeFile === 'string' ? cssManifest.homeFile : null,
    workbench: cssManifest && typeof cssManifest.workbenchFile === 'string' ? cssManifest.workbenchFile : null,
    tools: cssManifest && typeof cssManifest.toolsFile === 'string' ? cssManifest.toolsFile : null
  };
  if (!cssHrefs.base) {
    log('No CSS manifest found; leaving dist/styles.css references intact.');
  } else {
    const publicHtmlFiles = listHtmlFilesRecursive(outDir);
    let rewrote = 0;
    publicHtmlFiles.forEach((filePath) => {
      let html;
      try {
        html = fs.readFileSync(filePath, 'utf8');
      } catch {
        return;
      }
      const next = rewriteCssLinksInHtml(html, cssHrefs);
      if (next !== html) {
        try {
          fs.writeFileSync(filePath, next, 'utf8');
          rewrote++;
        } catch {}
      }
    });
    const rewrittenTargets = [`dist/${cssHrefs.base}`];
    if (cssHrefs.home) rewrittenTargets.push(`dist/${cssHrefs.home}`);
    if (cssHrefs.workbench) rewrittenTargets.push(`dist/${cssHrefs.workbench}`);
    if (cssHrefs.tools) rewrittenTargets.push(`dist/${cssHrefs.tools}`);
    log(`Rewrote CSS links in ${rewrote} HTML files to ${rewrittenTargets.join(', ')}`);
  }
  rewriteGoogleMapsEmbedsInPublic();
}

if (require.main === module) {
  copyStatic();
  log('Prepared public/ output for Vercel');
}

module.exports = { resolveApprovedDocuments, copyApprovedDocuments };
