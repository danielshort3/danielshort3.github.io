#!/usr/bin/env node
/*
  Prepare Vercel output: copy static site into ./public
  - Copies root HTML files and selected assets directories
  - Ensures ./public exists and is clean
*/
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const contributionsData = require('../js/contributions/contributions-data.js');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'public');
const generatedDir = path.join(root, 'documents', 'contributions', 'generated');

function log(msg){
  process.stdout.write(msg + '\n');
}

function ensureCleanDir(dir){
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}

function ensureDir(dir){
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

function formatItemLine(item){
  const meta = [];
  if (item.year && item.year !== 'Earlier') meta.push(item.year);
  if (item.quarter) meta.push(item.quarter);
  if (item.focus) meta.push(item.focus);
  const metaLine = meta.length ? ` (${meta.join(' · ')})` : '';
  const lines = [`- ${item.title}${metaLine}`];
  if (item.link) lines.push(`  Link: ${item.link}`);
  if (item.pdf) lines.push(`  PDF: ${item.pdf}`);
  return lines.join('\n');
}

function buildContributionDownloads(){
  if (!Array.isArray(contributionsData)) return;
  ensureDir(generatedDir);
  contributionsData.forEach(section => {
    const dl = section.download;
    if (!dl || !dl.file) return;

    const tempFiles = [];

    if (dl.includeLinkIndex) {
      const linkPath = path.join(generatedDir, `${section.id}-links.txt`);
      const lines = [
        `${section.heading} – Link Index`,
        '',
        ...section.items.map(formatItemLine)
      ];
      ensureDir(path.dirname(linkPath));
      fs.writeFileSync(linkPath, lines.join('\n'), 'utf8');
      tempFiles.push(linkPath);
    }

    if (Array.isArray(dl.summaryNotes) && dl.summaryNotes.length){
      const summaryPath = path.join(generatedDir, `${section.id}-summary.txt`);
      const lines = [
        `${section.heading} – Talking Points`,
        '',
        ...dl.summaryNotes.map(note => `- ${note}`)
      ];
      ensureDir(path.dirname(summaryPath));
      fs.writeFileSync(summaryPath, lines.join('\n'), 'utf8');
      tempFiles.push(summaryPath);
    }

    if (Array.isArray(dl.extraFiles)) {
      dl.extraFiles.forEach(rel => {
        const abs = path.join(root, rel);
        if (fs.existsSync(abs)) tempFiles.push(abs);
      });
    }

    if (!tempFiles.length) return;

    const output = path.join(root, dl.file);
    ensureDir(path.dirname(output));
    const args = [path.join(root, 'build', 'create_zip.py'), output, ...tempFiles];
    const result = spawnSync('python3', args, { stdio: 'inherit' });
    if (result.status !== 0) {
      throw new Error(`Failed to build download pack for ${section.id}`);
    }
  });
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
  const dirs = ['img', 'js', 'css', 'documents', 'dist', 'pages', 'demos'];
  dirs.forEach(d => copyDir(path.join(root, d), path.join(outDir, d)));
}

buildContributionDownloads();
copyStatic();
log('Prepared public/ output for Vercel');
