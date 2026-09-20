#!/usr/bin/env node
'use strict';
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { readActors, packCell, writeAtlas, sha256 } = require('./lib/starfall-overhaul-images');
const Data = require('../js/games/project-starfall/project-starfall-data');
const ROOT = path.resolve(__dirname, '..');
const SOURCE = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/fx');
const record = (file) => ({ path: path.relative(ROOT, file).replace(/\\/g, '/'), sha256: sha256(file) });
async function gridFrames(file, columns, rows, size, padding = 16) {
  const meta = await sharp(file).metadata(), frames = [];
  for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
    const left = Math.round(column * meta.width / columns), top = Math.round(row * meta.height / rows);
    const width = Math.round((column + 1) * meta.width / columns) - left, height = Math.round((row + 1) * meta.height / rows) - top;
    const content = await sharp(file).extract({ left, top, width, height }).resize(size - padding * 2, size - padding * 2, { fit: 'contain', background: '#00000000' }).png().toBuffer();
    frames.push(await sharp({ create: { width: size, height: size, channels: 4, background: '#00000000' } }).composite([{ input: content, left: padding, top: padding }]).png().toBuffer());
  }
  return frames;
}
async function processPortals() {
  const source = await readActors(path.join(SOURCE, 'portals.png'), 6, 3);
  const outputs = [];
  for (const [row, id] of ['standard', 'boss', 'locked'].entries()) {
    const frames = [];
    for (const cell of source.cells.slice(row * 6, row * 6 + 6)) frames.push(await packCell(source, cell, { scale: 0.47, baseline: 150 }));
    const output = path.join(ROOT, Data.PORTAL_ANIMATION_ASSETS[id].sheet);
    await writeAtlas(frames, output, 6, 1); outputs.push(record(output));
  }
  return outputs;
}
async function processGlobalFx() {
  const frames = await gridFrames(path.join(SOURCE, 'global-fx-clean.png'), 6, 6, 160, 19);
  const outputs = [];
  for (const [row, id] of ['slash', 'cast', 'arrowRelease', 'partyBuff', 'impact', 'defeatBurst'].entries()) {
    const output = path.join(ROOT, Data.FX_ANIMATION_ASSETS[id].sheet);
    await writeAtlas(frames.slice(row * 6, row * 6 + 6), output, 6, 1); outputs.push(record(output));
  }
  return outputs;
}
async function main() {
  const outputs = [...await processPortals(), ...await processGlobalFx()];
  const knives = await gridFrames(path.join(SOURCE, 'bandit-knife.png'), 3, 1, 64, 5);
  const output = path.join(ROOT, Data.ENEMY_PROJECTILE_ANIMATION_ASSETS.banditThrower.sheet);
  // Rigid projectile body: its travel and orientation are driven by the runtime.
  // A changing hand-drawn blade silhouette would read as shape drift.
  await writeAtlas([knives[0], knives[0], knives[0]], output, 3, 1, 64);
  outputs.push(record(output));
  for (const animation of Object.values(Data.SKILL_FX_ANIMATION_ASSETS).concat(Object.values(Data.BASIC_ATTACK_FX_ANIMATION_ASSETS), Object.values(Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS))) outputs.push(record(path.join(ROOT, animation.sheet)));
  fs.writeFileSync(path.join(SOURCE, 'ledger.json'), `${JSON.stringify({ outputs, sources: ['portals.png', 'global-fx-clean.png', 'bandit-knife.png'].map((name) => record(path.join(SOURCE, name))), nativeSource: 'build/lib/starfall-combat-language-art.js', rejectedSource: 'global-fx.png (dark matte on bright scenery)' }, null, 2)}\n`);
  console.log(`Recorded ${outputs.length} effect and portal outputs.`);
}
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
module.exports = { main, processPortals, processGlobalFx };
