#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const repositoryRoot = path.resolve(__dirname, '..');
const ORIGIN = 'https://www.danielshort.me';

function attributes(tag) {
  return Object.fromEntries([...tag.matchAll(/(?:^|\s)([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)]
    .map((match) => [match[1].toLowerCase(), (match[2] ?? match[3]).replace(/&amp;/g, '&')]));
}

function declaredAssets(html, route) {
  const source = html.replace(/<!--[\s\S]*?-->/g, '');
  const baseTag = source.match(/<base\b[^>]*>/i);
  const base = new URL(baseTag ? attributes(baseTag[0]).href || route : route, ORIGIN);
  const assets = new Map();
  const external = new Set();
  for (const match of source.matchAll(/<(script|link)\b[^>]*>/gi)) {
    const attrs = attributes(match[0]);
    const reference = match[1].toLowerCase() === 'script' ? attrs.src
      : String(attrs.rel || '').split(/\s+/).includes('stylesheet') && attrs.href;
    if (!reference) continue;
    const url = new URL(reference, base);
    if (url.origin !== ORIGIN) {
      external.add(url.href);
      continue;
    }
    // Fragments do not create a separate request; different query versions do.
    url.hash = '';
    assets.set(url.href, { url: `${url.pathname}${url.search}`, file: decodeURIComponent(url.pathname).replace(/^\//, '') });
  }
  return { assets: [...assets.values()], external: [...external].sort() };
}

function readAsset(outputRoot, relative) {
  const absolute = path.resolve(outputRoot, relative);
  const inside = path.relative(path.resolve(outputRoot), absolute);
  if (!inside || inside.startsWith('..') || path.isAbsolute(inside)) throw new Error(`Unsafe performance asset path: ${relative}`);
  const bytes = fs.readFileSync(absolute);
  return { bytes: bytes.length, gzipBytes: zlib.gzipSync(bytes, { level: 9 }).length };
}

function measureRoute(outputRoot, definition) {
  const html = fs.readFileSync(path.resolve(outputRoot, definition.file), 'utf8');
  const references = declaredAssets(html, definition.route);
  const assets = references.assets.map((asset) => ({ url: asset.url, ...readAsset(outputRoot, asset.file) }));
  const document = readAsset(outputRoot, definition.file);
  const total = assets.reduce((sum, asset) => ({ bytes: sum.bytes + asset.bytes, gzipBytes: sum.gzipBytes + asset.gzipBytes }), document);
  return { route: definition.route, category: definition.category, document, assets, external: references.external,
    ...total, maxGzipBytes: definition.maxGzipBytes, passed: total.gzipBytes <= definition.maxGzipBytes };
}

function measureCatalog(outputRoot, definition) {
  const directory = path.join(outputRoot, definition.directory);
  const files = fs.readdirSync(directory).filter((file) => /\.png$/i.test(file)).sort();
  if (!files.length) throw new Error(`No original catalog artwork in ${definition.directory}`);
  let pngBytes = 0;
  let webpBytes = 0;
  for (const file of files) {
    pngBytes += fs.statSync(path.join(directory, file)).size;
    webpBytes += fs.statSync(path.join(directory, file.replace(/\.png$/i, '.webp'))).size;
  }
  return { directory: definition.directory, count: files.length, pngBytes, webpBytes,
    savingsPercent: Math.round((1 - webpBytes / pngBytes) * 1000) / 10,
    maxWebpBytes: definition.maxWebpBytes, passed: webpBytes <= definition.maxWebpBytes };
}

function createReport(outputRoot, budgets) {
  const routes = budgets.routes.map((route) => measureRoute(outputRoot, route));
  const catalogs = budgets.catalogs.map((catalog) => measureCatalog(outputRoot, catalog));
  return { version: 1, generatedAt: new Date().toISOString(),
    scope: 'Sum of each initial HTML document and directly declared first-party stylesheet/script files, gzip level 9. This is a deterministic build-size estimate, not a browser waterfall or Core Web Vitals measurement.',
    exclusions: ['Dynamically imported or fetched scripts/models/game assets', 'Fonts and images in page-load totals (complete icon catalogs are reported separately)', 'Third-party requests, caching, HTTP headers, network and execution time'],
    rationale: budgets.rationale, passed: [...routes, ...catalogs].every((entry) => entry.passed), routes, catalogs };
}

function toMarkdown(report) {
  const rows = report.routes.map((entry) => `| ${entry.route} | ${entry.category} | ${(entry.gzipBytes / 1000).toFixed(1)} | ${(entry.maxGzipBytes / 1000).toFixed(0)} | ${entry.passed ? 'Pass' : 'Over budget'} |`);
  return ['# Website delivery budget report', '', report.scope, '',
    '| Route | Category | Estimated gzip KB | Limit KB | Result |', '|---|---|---:|---:|---|', ...rows, '',
    '## Approved catalog artwork', '', 'Catalog totals are not initial-page transfers. PNG fallbacks remain available.', '',
    '| Catalog | Icons | PNG KB | WebP KB | Smaller |', '|---|---:|---:|---:|---:|',
    ...report.catalogs.map((entry) => `| ${entry.directory} | ${entry.count} | ${(entry.pngBytes / 1000).toFixed(1)} | ${(entry.webpBytes / 1000).toFixed(1)} | ${entry.savingsPercent}% |`), '',
    '## Scope and limits', '', ...report.exclusions.map((item) => `- Excludes ${item.charAt(0).toLowerCase()}${item.slice(1)}.`), '', report.rationale, ''].join('\n');
}

function main() {
  const args = process.argv.slice(2);
  const outputArgument = args.indexOf('--output');
  const output = outputArgument >= 0 ? args[outputArgument + 1] : 'tmp/performance/report.json';
  if (!output) throw new Error('--output requires a report path.');
  const budgets = JSON.parse(fs.readFileSync(path.join(__dirname, 'performance-budgets.json'), 'utf8'));
  const report = createReport(path.join(repositoryRoot, 'public'), budgets);
  const target = path.resolve(repositoryRoot, output);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, JSON.stringify(report, null, 2) + '\n');
  fs.writeFileSync(target.replace(/\.json$/i, '') + '.md', toMarkdown(report));
  for (const route of report.routes) console.log(`${route.passed ? 'PASS' : 'FAIL'} ${route.route}: ${(route.gzipBytes / 1000).toFixed(1)} KB estimated gzip / ${route.maxGzipBytes / 1000} KB budget`);
  console.log(`Performance report: ${target}`);
  if (process.env.GITHUB_STEP_SUMMARY) fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, toMarkdown(report));
  if (args.includes('--check') && !report.passed) process.exitCode = 1;
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = { declaredAssets, readAsset, measureRoute, measureCatalog, createReport, toMarkdown };
