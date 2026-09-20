#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Assets = require('../js/games/project-starfall/engine/assets.js');
const ROOT = path.resolve(__dirname, '..');
const DIR = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1');
const BASELINE = path.join(DIR, 'baseline.json');
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function filesIn(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    return entry.isDirectory() ? filesIn(file) : [file];
  });
}
function relative(file) { return path.relative(ROOT, file).replace(/\\/g, '/'); }
function activePaths() {
  return Array.from(new Set(Assets.collectAssetPaths(Data).concat([
    'img/project-starfall/ui/start-screen.avif',
    'img/project-starfall/ui/start-screen.webp'
  ]))).sort();
}
fs.mkdirSync(DIR, { recursive: true });
if (process.argv.includes('--snapshot')) {
  if (fs.existsSync(BASELINE)) throw new Error('Baseline is immutable; refusing to replace it.');
  const record = {
    created: new Date().toISOString(),
    source: 'Live runtime registry; generated start-screen alternatives included.',
    assets: activePaths().map((file) => ({ path: file, sha256: hash(path.join(ROOT, file)) })),
    protectedSessionFiles: filesIn(path.join(ROOT, 'output/starfall-animation-samples'))
      .filter((file) => /\.(png|gif|webp)$/i.test(file))
      .map((file) => ({ path: relative(file), sha256: hash(file) }))
  };
  fs.writeFileSync(BASELINE, `${JSON.stringify(record, null, 2)}\n`);
  console.log(`Saved ${record.assets.length} active assets and ${record.protectedSessionFiles.length} protected session images.`);
} else {
  const baseline = JSON.parse(fs.readFileSync(BASELINE, 'utf8'));
  const currentPaths = new Set(activePaths());
  const inventoryPath = path.join(DIR, 'enemies/inventory.json');
  const inventory = fs.existsSync(inventoryPath) ? JSON.parse(fs.readFileSync(inventoryPath, 'utf8')) : { items: [] };
  const replacements = new Map(inventory.items.filter((item) => item.replacement).map((item) => [item.replacement.oldSheet, item.replacement]));
  const missing = [], unchanged = [], changed = [], retired = [], protectedFailures = [], unaccounted = [];
  for (const item of baseline.assets) {
    const replacement = replacements.get(item.path);
    if (!currentPaths.has(item.path)) {
      if (replacement && currentPaths.has(replacement.sheet) && fs.existsSync(path.join(ROOT, replacement.sheet)) && hash(path.join(ROOT, replacement.sheet)) === replacement.sheetSha256) {
        retired.push({ path: item.path, replacement: replacement.sheet, sha256: replacement.sheetSha256 });
      } else {
        unaccounted.push(item.path);
      }
      continue;
    }
    const file = path.join(ROOT, item.path);
    if (!fs.existsSync(file)) missing.push(item.path);
    else (hash(file) === item.sha256 ? unchanged : changed).push(item.path);
  }
  for (const item of baseline.protectedSessionFiles) {
    const file = path.join(ROOT, item.path);
    if (!fs.existsSync(file) || hash(file) !== item.sha256) protectedFailures.push(item.path);
  }
  const originalPaths = new Set(baseline.assets.map((item) => item.path));
  const added = Array.from(currentPaths).filter((file) => !originalPaths.has(file));
  for (const file of added) if (!fs.existsSync(path.join(ROOT, file))) missing.push(file);
  const publicVerified = process.argv.includes('--public');
  const publicFailures = publicVerified ? Array.from(currentPaths).filter((file) => {
    const sourceFile = path.join(ROOT, file), publicFile = path.join(ROOT, 'public', file);
    return !fs.existsSync(sourceFile) || !fs.existsSync(publicFile) || hash(sourceFile) !== hash(publicFile);
  }) : [];
  const report = { checked: new Date().toISOString(), total: baseline.assets.length, activeTotal: currentPaths.size, changed, retired, added, unchanged, missing, unaccounted, protectedFailures, publicVerified, publicFailures };
  fs.writeFileSync(path.join(DIR, 'coverage.json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ total: report.total, activeTotal: currentPaths.size, changed: changed.length, retired: retired.length, added: added.length, unchanged: unchanged.length, missing, unaccounted, protectedFailures, publicVerified, publicFailures }, null, 2));
  if (missing.length || unaccounted.length || protectedFailures.length || publicFailures.length || (process.argv.includes('--complete') && unchanged.length)) process.exitCode = 1;
}
