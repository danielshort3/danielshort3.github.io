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

function rewriteCssLinksInHtml(html, cssHref) {
  if (!html || !cssHref) return html;
  return html.replace(/href=(["'])dist\/styles\.css\1/g, `href="dist/${cssHref}"`);
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
  const dirs = ['img', 'js', 'css', 'documents', 'dist', 'pages', 'demos', 'slot-config'];
  dirs.forEach(d => copyDir(path.join(root, d), path.join(outDir, d)));

  // Rewrite public HTML to reference the hashed CSS bundle (better caching).
  const manifest = readJson(cssManifestPath);
  const hashedFile = manifest && typeof manifest.file === 'string' ? manifest.file : null;
  const cssHref = hashedFile || null;
  if (!cssHref) {
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
    const next = rewriteCssLinksInHtml(html, cssHref);
    if (next !== html) {
      try {
        fs.writeFileSync(filePath, next, 'utf8');
        rewrote++;
      } catch {}
    }
  });
  log(`Rewrote CSS links in ${rewrote} HTML files to dist/${cssHref}`);
}

copyStatic();
log('Prepared public/ output for Vercel');
