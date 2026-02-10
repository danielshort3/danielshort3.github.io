#!/usr/bin/env node
/*
  Prepare Vercel output: copy static site into ./public
  - Copies root HTML files and selected assets directories
  - Ensures ./public exists and is clean
*/
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'public');
const cssManifestPath = path.join(root, 'dist', 'styles-manifest.json');
const textExtensions = new Set([
  '.css',
  '.html',
  '.js',
  '.json',
  '.md',
  '.mjs',
  '.txt',
  '.ts',
  '.tsx',
  '.xml',
  '.xsl',
  '.yaml',
  '.yml'
]);
const scanSkipDirs = new Set([
  '.git',
  '.vercel',
  'Project Submission',
  'Slot-Machine-v4',
  'aws',
  'dist',
  'documents',
  'img',
  'node_modules',
  'public'
]);

function log(msg){
  process.stdout.write(msg + '\n');
}

function ensureCleanDir(dir){
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}

function copyFile(src, dest){
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

function copyDir(src, dest){
  // Node >=16: cpSync is available
  if (fs.existsSync(src)) {
    fs.cpSync(src, dest, { recursive: true });
  }
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

function listTextFilesRecursive(dirPath) {
  const files = [];
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
        if (scanSkipDirs.has(entry.name)) return;
        stack.push(full);
        return;
      }
      if (!entry.isFile()) return;
      const ext = path.extname(entry.name).toLowerCase();
      if (!textExtensions.has(ext)) return;
      files.push(full);
    });
  }
  return files;
}

function collectReferencedDocumentFiles() {
  const matches = new Set();
  const sources = listTextFilesRecursive(root);
  const matcher = /(?:^|["'(=\s])\/?(documents\/[A-Za-z0-9._\-/%]+)(?=$|["')#?\s<])/g;

  sources.forEach((filePath) => {
    let body = '';
    try {
      body = fs.readFileSync(filePath, 'utf8');
    } catch {
      return;
    }
    let match;
    while ((match = matcher.exec(body))) {
      const raw = String(match[1] || '').trim();
      if (!raw) continue;
      let decoded = raw;
      try {
        decoded = decodeURIComponent(raw);
      } catch {}
      decoded = decoded.replace(/\\/g, '/');
      if (!decoded.startsWith('documents/')) continue;
      if (decoded.includes('..')) continue;
      const abs = path.join(root, decoded);
      let stat;
      try {
        stat = fs.statSync(abs);
      } catch {
        continue;
      }
      if (!stat.isFile()) continue;
      matches.add(decoded);
    }
  });

  // Keep baseline resume exports available even if scan misses a reference.
  ['documents/Resume.pdf', 'documents/Resume.docx'].forEach((relPath) => {
    const abs = path.join(root, relPath);
    if (!fs.existsSync(abs)) return;
    matches.add(relPath);
  });

  return [...matches].sort();
}

function copyReferencedDocuments() {
  const relPaths = collectReferencedDocumentFiles();
  if (!relPaths.length) {
    copyDir(path.join(root, 'documents'), path.join(outDir, 'documents'));
    log('No referenced documents detected; copied full documents/ directory.');
    return;
  }

  let totalBytes = 0;
  relPaths.forEach((relPath) => {
    const src = path.join(root, relPath);
    const dest = path.join(outDir, relPath);
    copyFile(src, dest);
    try {
      totalBytes += fs.statSync(src).size;
    } catch {}
  });
  log(`Copied ${relPaths.length} referenced documents (${Math.round(totalBytes / 1024)} KB).`);
}

function rewriteCssLinksInHtml(html, cssHrefs) {
  if (!html) return html;
  let next = html;
  if (cssHrefs.base) {
    next = next.replace(/href=(["'])dist\/styles\.css\1/g, `href="dist/${cssHrefs.base}"`);
  }
  if (cssHrefs.tools) {
    next = next.replace(/href=(["'])dist\/styles-tools\.css\1/g, `href="dist/${cssHrefs.tools}"`);
  }
  return next;
}

function copyStatic(){
  ensureCleanDir(outDir);

  // Copy all root-level HTML files
  const htmlFiles = listRootHtmlFiles(root);
  htmlFiles.forEach(src => {
    const rel = path.relative(root, src);
    copyFile(src, path.join(outDir, rel));
  });

  // Copy selected root-level static files if present
  const rootFiles = ['robots.txt', 'sitemap.xml', 'sitemap.xsl', 'favicon.ico'];
  rootFiles.forEach(name => {
    const src = path.join(root, name);
    if (fs.existsSync(src)) copyFile(src, path.join(outDir, name));
  });

  // Copy asset and content directories used by the site
  const dirs = ['img', 'js', 'css', 'dist', 'pages', 'demos', 'slot-config'];
  dirs.forEach(d => copyDir(path.join(root, d), path.join(outDir, d)));
  copyReferencedDocuments();

  // Rewrite public HTML to reference the hashed CSS bundle (better caching).
  const manifest = readJson(cssManifestPath);
  const cssHrefs = {
    base: manifest && typeof manifest.file === 'string' ? manifest.file : null,
    tools: manifest && typeof manifest.toolsFile === 'string' ? manifest.toolsFile : null
  };
  if (!cssHrefs.base) {
    log('No CSS manifest found; leaving dist/styles.css references intact.');
    return;
  }
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
  if (cssHrefs.tools) rewrittenTargets.push(`dist/${cssHrefs.tools}`);
  log(`Rewrote CSS links in ${rewrote} HTML files to ${rewrittenTargets.join(', ')}`);
}

copyStatic();
log('Prepared public/ output for Vercel');
