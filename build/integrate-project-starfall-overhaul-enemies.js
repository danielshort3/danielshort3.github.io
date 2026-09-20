#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const ROOT = path.resolve(__dirname, '..');
const SOURCE = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies');
const inventory = JSON.parse(fs.readFileSync(path.join(SOURCE, 'inventory.json'), 'utf8'));
const registrations = JSON.parse(fs.readFileSync(path.join(SOURCE, 'enemy-registration.json'), 'utf8'));
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT, file))).digest('hex');
for (const item of inventory.items) {
  const replacement = item.replacement;
  if (!replacement || !registrations[item.fileId]) throw new Error(`Import and review ${item.fileId} before activating this migration.`);
  if (hash(replacement.sheet) !== replacement.sheetSha256) throw new Error(`Unreviewed output changed: ${replacement.sheet}`);
}
const animationFile = path.join(ROOT, 'js/games/project-starfall/data/animations.js');
let source = fs.readFileSync(animationFile, 'utf8');
const entries = Object.keys(registrations).sort().map((id) => `      '${id}': Object.freeze(${JSON.stringify(registrations[id])})`);
const block = `// BEGIN ILLUSTRATED ENEMY REGISTRATION\n    const ILLUSTRATED_ENEMY_REGISTRATIONS = Object.freeze({\n${entries.join(',\n')}\n    });\n    // END ILLUSTRATED ENEMY REGISTRATION`;
source = source.replace(/\/\/ BEGIN ILLUSTRATED ENEMY REGISTRATION[\s\S]*?\/\/ END ILLUSTRATED ENEMY REGISTRATION/, block);
source = source.replace(/assets\[enemyId\] = COMPACT_ENEMY_ANIMATION_FILE_IDS\[enemyId\][\s\S]*?: makeEnemyAnimationAsset\(ENEMY_ANIMATION_FILE_IDS\[enemyId\], enemyId\);/, 'assets[enemyId] = makeEnemyAnimationAsset(ENEMY_ANIMATION_FILE_IDS[enemyId], enemyId);');
fs.writeFileSync(animationFile, source);
const manifestFile = path.join(ROOT, 'asset-sources/project-starfall/asset-generation-manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
Object.assign(manifest.contracts.enemies, {
  artVersion: 'illustrated-v1',
  sourceFolder: 'asset-sources/project-starfall/overhaul-v1/enemies',
  sourcePattern: '<enemy-file-id>/source.png',
  animationPattern: '<enemy-file-id>-sheet.png',
  frameSize: 160,
  columns: 6,
  sheetWidth: 960,
  sheetHeight: 1280,
  processor: 'build/process-project-starfall-overhaul-enemies.js --enemy <file-id> --import',
  inventory: 'asset-sources/project-starfall/overhaul-v1/enemies/inventory.json'
});
delete manifest.contracts.enemies.sourceGuideColor;
delete manifest.contracts.enemies.defaultChroma;
delete manifest.contracts.enemies.greenSubjectChroma;
fs.writeFileSync(manifestFile, `${JSON.stringify(manifest, null, 2)}\n`);
execFileSync(process.execPath, [path.join(__dirname, 'generate-project-starfall-enemy-hurtboxes.js')], { cwd: ROOT, stdio: 'inherit' });
console.log(`Activated ${inventory.items.length} reviewed enemy atlases with per-identity registration.`);
