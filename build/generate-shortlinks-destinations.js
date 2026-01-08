#!/usr/bin/env node
'use strict';

/*
  Generates a JSON manifest of internal site destinations for the Short Links admin tool.

  Output:
  - dist/shortlinks-destinations.json
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const outputPath = path.join(root, 'dist', 'shortlinks-destinations.json');

function readFileSafe(filePath){
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch {
    return '';
  }
}

function listHtmlFiles(dir){
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter(name => name.endsWith('.html'))
    .map(name => path.join(dir, name));
}

function extractTitle(html){
  const match = html.match(/<title[^>]*>([^<]*)<\/title>/i);
  return match ? String(match[1]).trim() : '';
}

function extractCanonicalHref(html){
  const direct = html.match(/<link[^>]*rel=["']canonical["'][^>]*href=["']([^"']+)["'][^>]*>/i);
  if (direct) return String(direct[1]).trim();
  const flipped = html.match(/<link[^>]*href=["']([^"']+)["'][^>]*rel=["']canonical["'][^>]*>/i);
  if (flipped) return String(flipped[1]).trim();
  return '';
}

function decodeHtmlEntities(value){
  return String(value || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'");
}

function normalizeLabel(title){
  let label = String(title || '').trim();
  label = decodeHtmlEntities(label);
  label = label.replace(/\s*\|\s*Daniel Short\s*$/i, '').trim();
  label = label.replace(/\s*-\s*Daniel Short\s*$/i, '').trim();
  label = label.replace(/^Daniel Short\s*-\s*/i, '').trim();
  label = label.replace(/\s*\(\s*Daniel\s*Short\s*\)\s*$/i, '').trim();
  return label || title || '';
}

function classifyGroup(pathname){
  const p = String(pathname || '');
  if (p.startsWith('/tools/')) return 'Tools';
  if (p === '/tools') return 'Tools';
  if (p.startsWith('/portfolio/')) return 'Portfolio';
  if (p === '/portfolio') return 'Portfolio';
  if (p.includes('-demo')) return 'Demos';
  return 'Pages';
}

function loadDemoRoutes(){
  const vercelPath = path.join(root, 'vercel.json');
  const raw = readFileSafe(vercelPath);
  if (!raw) return [];

  let config;
  try {
    config = JSON.parse(raw);
  } catch {
    return [];
  }

  const rewrites = Array.isArray(config.rewrites) ? config.rewrites : [];
  const demos = rewrites
    .filter(rule => rule && typeof rule.source === 'string' && typeof rule.destination === 'string')
    .filter(rule => rule.destination.startsWith('/demos/'))
    .filter(rule => !rule.source.includes(':') && !rule.source.includes('('));

  const byDestination = new Map();
  demos.forEach(rule => {
    const dest = rule.destination;
    const source = rule.source;
    if (!byDestination.has(dest)) byDestination.set(dest, []);
    byDestination.get(dest).push(source);
  });

  const routes = [];
  for (const [dest, sources] of byDestination.entries()) {
    const clean = sources.find(src => !src.endsWith('.html')) || sources[0];
    const slug = dest.replace(/^\/demos\//, '');
    const filePath = path.join(root, 'demos', `${slug}.html`);
    const html = readFileSafe(filePath);
    const title = normalizeLabel(extractTitle(html)) || slug;
    routes.push({ path: clean, label: title, group: 'Demos' });
  }

  return routes;
}

function buildManifest(){
  const files = [];
  files.push(...listHtmlFiles(root));
  files.push(...listHtmlFiles(path.join(root, 'pages')));
  files.push(...listHtmlFiles(path.join(root, 'pages', 'portfolio')));

  const seen = new Map();
  files.forEach(filePath => {
    const html = readFileSafe(filePath);
    if (!html) return;
    const canonical = extractCanonicalHref(html);
    if (!canonical) return;

    let pathname = '';
    try {
      const url = new URL(canonical);
      pathname = url.pathname || '';
    } catch {
      return;
    }
    if (!pathname) return;
    if (pathname === '/404.html') return;

    const label = normalizeLabel(extractTitle(html)) || pathname;
    const group = classifyGroup(pathname);
    const finalLabel = pathname === '/' ? 'Home' : label;
    if (!seen.has(pathname)) seen.set(pathname, { path: pathname, label: finalLabel, group });
  });

  loadDemoRoutes().forEach(route => {
    if (!seen.has(route.path)) {
      seen.set(route.path, route);
    }
  });

  const groupOrder = ['Pages', 'Tools', 'Portfolio', 'Demos'];
  const pages = Array.from(seen.values())
    .sort((a, b) => {
      const aGroup = groupOrder.indexOf(a.group);
      const bGroup = groupOrder.indexOf(b.group);
      if (aGroup !== bGroup) return aGroup - bGroup;
      const aLabel = String(a.label || '');
      const bLabel = String(b.label || '');
      const byLabel = aLabel.localeCompare(bLabel);
      if (byLabel !== 0) return byLabel;
      return String(a.path || '').localeCompare(String(b.path || ''));
    });

  return {
    generatedAt: new Date().toISOString(),
    origin: 'https://danielshort.me',
    pages
  };
}

function writeManifest(){
  const manifest = buildManifest();
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(manifest, null, 2) + '\n');
  process.stdout.write(`Wrote ${path.relative(root, outputPath)} (${manifest.pages.length} destinations)\n`);
}

writeManifest();
