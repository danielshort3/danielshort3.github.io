#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { readActors, packCell, writeAtlas, sha256 } = require('./lib/starfall-overhaul-images');
const ROOT = path.resolve(__dirname, '..');
const SOURCE = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/players');
const REVIEW = path.join(ROOT, 'output/starfall-animation-samples/review-v2');
const ROWS = ['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat'];
const SOURCES = {
  idle: ['player-idle-climb.png', 0, 0.425],
  run: ['player-run-study.png', 0, 0.36, true],
  jump: ['player-jump-fall.png', 0, 0.46],
  fall: ['player-jump-fall.png', 8, 0.46],
  climb: ['player-idle-climb.png', 8, 0.50],
  basic: ['player-strike-study.png', 0, 0.35, true],
  skill: ['player-skill-party.png', 0, 0.47],
  party: ['player-skill-party.png', 8, 0.47],
  hit: ['player-hit-defeat.png', 0, 0.445],
  defeat: ['player-hit-defeat.png', 8, 0.445]
};
// Authored body landmarks in each isolated pose. These move with the drawing;
// they never determine sprite scale. Near/far limbs remain separate sockets.
const HANDS = {
  idle: [[0.86, 0.70]],
  run: [[0.86, 0.59], [0.87, 0.58], [0.85, 0.59], [0.84, 0.55], [0.95, 0.60], [0.92, 0.61], [0.96, 0.60], [0.95, 0.56]],
  jump: [[0.72, 0.83], [0.88, 0.66], [0.92, 0.28], [0.86, 0.49], [0.84, 0.46], [0.92, 0.50], [0.94, 0.52], [0.96, 0.52]],
  fall: [[0.95, 0.53], [0.88, 0.55], [0.91, 0.57], [0.84, 0.63], [0.84, 0.64], [0.94, 0.65], [0.93, 0.66], [0.91, 0.66]],
  climb: [[0.90, 0.08], [0.92, 0.31], [0.90, 0.10], [0.91, 0.06], [0.12, 0.12], [0.14, 0.33], [0.13, 0.13], [0.15, 0.09]],
  basic: [[0.85, 0.58], [0.88, 0.51], [0.90, 0.45], [0.98, 0.47], [0.97, 0.47], [0.89, 0.52], [0.82, 0.58], [0.86, 0.58]],
  skill: [[0.88, 0.64], [0.88, 0.63], [0.78, 0.56], [0.79, 0.55], [0.94, 0.45], [0.97, 0.48], [0.83, 0.59], [0.88, 0.64]],
  party: [[0.88, 0.64], [0.73, 0.65], [0.78, 0.55], [0.75, 0.55], [0.96, 0.53], [0.97, 0.53], [0.88, 0.72], [0.87, 0.64]],
  hit: [[0.88, 0.60], [0.90, 0.55], [0.90, 0.62], [0.89, 0.63], [0.86, 0.60], [0.78, 0.65], [0.87, 0.61], [0.87, 0.61]],
  defeat: [[0.87, 0.72], [0.80, 0.77], [0.74, 0.82], [0.86, 0.90], [0.74, 0.91], [0.63, 0.87], [0.64, 0.85], [0.65, 0.86]]
};
function poseAttachments(row, index, cell, packed, buckle) {
  const b = cell.bounds, t = packed.transform;
  const point = (x, y, angle = 0) => [Math.round(t.left + b.width * x * t.scale), Math.round(t.top + b.height * y * t.scale), angle];
  const hands = HANDS[row], hand = hands[Math.min(index, hands.length - 1)];
  const prone = row === 'defeat' && index >= 5;
  const main = point(hand[0], hand[1]);
  const off = point(prone ? 0.47 : 0.22, prone ? 0.81 : row === 'party' ? hand[1] : 0.66);
  const head = point(prone ? 0.80 : row === 'climb' ? 0.50 : 0.60, prone ? 0.43 : 0.25, prone ? 80 : 0);
  const torso = buckle ? [Math.round(t.left + (buckle.x - b.left) * t.scale - 4), Math.round(t.top + (buckle.y - b.top) * t.scale - 13), 30, 34, prone ? 80 : 0] : [...point(0.49, 0.64).slice(0, 2), 30, 34, prone ? 80 : 0];
  const feet = prone ? [point(0.10, 0.83), point(0.26, 0.85)] : row === 'run' ? [point(index % 4 === 1 ? 0.49 : 0.83, 0.94, 15), point(0.16, 0.84, -10)] : [point(0.71, 0.95), point(0.29, 0.95)];
  const angles = row === 'basic' ? [-62, -34, -7, 8, 25, 42, 62, 62] : row === 'run' ? [30, 24, 35, 40, 30, 24, 35, 40] : [62];
  const stowed = row === 'climb' || row === 'party';
  const options = { weapon: stowed ? [torso[0] - 22, torso[1] + 13, 78] : [main[0], main[1], angles[Math.min(index, angles.length - 1)]], weaponMode: prone ? 'dropped' : stowed ? 'stowed' : 'held' };
  return { torso, head, main, off, feet, options };
}
function goldCenter(source, bounds) {
  let sumX = 0, sumY = 0, count = 0;
  for (let y = Math.floor(bounds.top + bounds.height * 0.46); y < bounds.top + bounds.height * 0.83; y += 1) {
    for (let x = bounds.left; x < bounds.left + bounds.width; x += 1) {
      const i = (y * source.info.width + x) * 4;
      const [r, g, b, a] = source.data.subarray(i, i + 4);
      if (a < 128 || r < 150 || g < 110 || g - b < 45 || r / Math.max(1, b) < 2.1 || g / r < 0.67) continue;
      sumX += x; sumY += y; count += 1;
    }
  }
  return count > 2 ? { x: sumX / count, y: sumY / count, count } : null;
}
async function main() {
  const cache = new Map(), frames = [], metadata = {}, attachments = {};
  for (const row of ROWS) {
    const [name, start, scale, protectedStudy] = SOURCES[row];
    const file = path.join(protectedStudy ? REVIEW : SOURCE, name);
    if (!cache.has(file)) cache.set(file, await readActors(file, 4, protectedStudy ? 2 : 4));
    const source = cache.get(file);
    metadata[row] = [];
    attachments[row] = [];
    for (let index = 0; index < 8; index += 1) {
      const cell = source.cells[start + index];
      const buckle = goldCenter(source, cell.bounds);
      // The buckle fixes horizontal registration through reaching poses.
      // Climbing and prone poses use authored silhouette landmarks instead.
      let rootX = buckle ? buckle.x - 6 / scale : cell.bounds.left + cell.bounds.width * 0.50;
      if (row === 'defeat' && index >= 5) rootX = cell.bounds.left + cell.bounds.width * 0.5;
      if (row === 'climb') rootX = cell.bounds.left + cell.bounds.width * 0.5;
      const airOffset = row === 'run' ? [0, 0, 2, 5, 0, 0, 2, 5][index] : 0;
      const packed = await packCell(source, cell, { scale, rootX, baseline: 150 - airOffset });
      frames.push(packed);
      attachments[row].push(poseAttachments(row, index, cell, packed, buckle));
      metadata[row].push({ source: path.relative(ROOT, file).replace(/\\/g, '/'), sourceRect: cell.bounds, transform: packed.transform, buckle, registration: { originX: 80, groundY: 150, authoredBodyHeight: 140 } });
    }
  }
  const output = path.join(ROOT, 'img/project-starfall/animations/players/generic-player-sheet.png');
  await writeAtlas(frames, output, 8, 10);
  const portrait = path.join(ROOT, 'img/project-starfall/characters/generic-player.png');
  await sharp(frames[0].png).resize(320, 320).png().toFile(portrait);
  const fox = await readActors(path.join(SOURCE, 'starfall-fox.png'), 6, 6);
  const foxFrames = [];
  for (const cell of fox.cells) foxFrames.push(await packCell(fox, cell, { scale: 0.61, rootX: cell.bounds.left + cell.bounds.width * 0.6, baseline: 148 }));
  const foxOutput = path.join(ROOT, 'img/project-starfall/animations/pets/starfall-fox-sheet.png');
  await writeAtlas(foxFrames, foxOutput, 6, 6);
  fs.writeFileSync(path.join(SOURCE, 'registration.json'), `${JSON.stringify(metadata, null, 2)}\n`);
  const attachmentPath = path.join(ROOT, 'js/games/project-starfall/engine/equipment-attachments.js');
  let attachmentCode = fs.readFileSync(attachmentPath, 'utf8');
  const generated = '// BEGIN ILLUSTRATED PLAYER SOCKETS\nconst OVERHAUL_ATTACHMENTS = Object.freeze({\n' + ROWS.map((row) => `  ${row}: Object.freeze([\n` + attachments[row].map((pose) => `    A(${[pose.torso, pose.head, pose.main, pose.off, pose.feet, pose.options].map(JSON.stringify).join(', ')})`).join(',\n') + '\n  ])').join(',\n') + '\n});\n// END ILLUSTRATED PLAYER SOCKETS';
  if (attachmentCode.includes('// BEGIN ILLUSTRATED PLAYER SOCKETS')) attachmentCode = attachmentCode.replace(/\/\/ BEGIN ILLUSTRATED PLAYER SOCKETS[\s\S]*?\/\/ END ILLUSTRATED PLAYER SOCKETS/, generated);
  else attachmentCode = attachmentCode.replace('function getEquipmentAttachment(row, frame) {', `${generated}\n\nfunction getEquipmentAttachment(row, frame) {`);
  fs.writeFileSync(attachmentPath, attachmentCode);
  const ledger = { source: 'Built-in image generation; session run and strike sources preserved byte-for-byte.', layout: 'player 8x10 cells160; pet6x6 cells160', outputs: [output, portrait, foxOutput].map((file) => ({ path: path.relative(ROOT, file).replace(/\\/g, '/'), sha256: sha256(file) })), sources: [...cache.keys(), fox.file].map((file) => ({ path: path.relative(ROOT, file).replace(/\\/g, '/'), sha256: sha256(file) })) };
  fs.writeFileSync(path.join(SOURCE, 'ledger.json'), `${JSON.stringify(ledger, null, 2)}\n`);
  console.log('Exported player80poses, portrait and fox36poses with explicit registration and fixed per-action scales.');
}
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
module.exports = { main, goldCenter };
