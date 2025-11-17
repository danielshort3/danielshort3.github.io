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

function copyStatic(){
  ensureCleanDir(outDir);

  // Copy all root-level HTML files
  const htmlFiles = listRootHtmlFiles(root);
  htmlFiles.forEach(src => {
    const rel = path.relative(root, src);
    copyFile(src, path.join(outDir, rel));
  });

  // Copy selected root-level static files if present
  const rootFiles = ['robots.txt', 'sitemap.xml', 'favicon.ico'];
  rootFiles.forEach(name => {
    const src = path.join(root, name);
    if (fs.existsSync(src)) copyFile(src, path.join(outDir, name));
  });

  // Copy asset and content directories used by the site
  const dirs = ['img', 'js', 'css', 'documents', 'dist', 'pages', 'demos', 'slot-config'];
  dirs.forEach(d => copyDir(path.join(root, d), path.join(outDir, d)));
}

copyStatic();
log('Prepared public/ output for Vercel');
