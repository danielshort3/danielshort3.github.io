#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Visuals = require('../js/games/project-starfall/engine/visuals.js');
const Rig = require('../js/games/project-starfall/engine/equipment-attachments.js');
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'output/starfall-overhaul-review');
const SOURCE = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1');
const baseline = JSON.parse(fs.readFileSync(path.join(SOURCE, 'baseline.json'), 'utf8'));
const baselineHashes = new Map(baseline.assets.map((r) => [r.path, r.sha256]));
const hash = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');
const relative = (file) => path.relative(OUT, path.join(ROOT, file)).replace(/\\/g, '/');
const title = (id) => id.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[-_]/g, ' ').replace(/\b\w/g, (s) => s.toUpperCase());
const escape = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function moduleAtHead(file) {
  const context = { module: { exports: {} }, console };
  vm.runInNewContext(execFileSync('git', ['show', `HEAD:${file}`], { cwd: ROOT, encoding: 'utf8', maxBuffer: 10e6 }), context, { filename: file });
  return context.module.exports;
}

function assetUrl(file) {
  if (!fs.existsSync(path.join(ROOT, file))) throw new Error(`Review asset missing: ${file}`);
  return `${relative(file)}?v=${hash(fs.readFileSync(path.join(ROOT, file))).slice(0, 12)}`;
}

function assertBaseline(file, original) {
  const expected = baselineHashes.get(original);
  if (!expected || hash(fs.readFileSync(file)) !== expected) throw new Error(`Before image does not match baseline: ${original}`);
}

function makeClip(animation, height, registration, rig) {
  const actions = {};
  for (const [id, config] of Object.entries(animation.states)) {
    const sequence = Array.isArray(config.sequence) ? config.sequence : Array.from({ length: config.frames }, (_, i) => i);
    const holdsMs = sequence.map((frame) => 1000 * Math.max(1, config.holds?.[frame] || 1) / Math.max(1, config.fps));
    if (config.loop && config.loopDelay) holdsMs[holdsMs.length - 1] += config.loopDelay * 1000;
    actions[id] = {
      row: config.row, sequence, holdsMs, durationMs: holdsMs.reduce((a, b) => a + b, 0), loop: !!config.loop,
      registrations: sequence.map((frame) => rig ? rig.getPlayerSpriteRegistration(id, frame, registration) : registration)
    };
  }
  return { image: assetUrl(animation.sheet), frameWidth: animation.frameWidth, frameHeight: animation.frameHeight, height, actions };
}

function applyRepresentativeTell(clip, enemy) {
  const tell = clip.actions.telegraph;
  if (!tell) return;
  const behavior = enemy.behavior;
  const durationMs = behavior === 'boss' ? 1000 : behavior === 'charger' ? 750 : ['thrower', 'turret', 'ranged'].includes(behavior) ? 540 : 420;
  const commitmentMs = Math.min(durationMs, behavior === 'boss' ? 300 : 200);
  tell.holdsMs = tell.sequence.map((_, i) => i === tell.sequence.length - 1 ? commitmentMs : (durationMs - commitmentMs) / Math.max(1, tell.sequence.length - 1));
  tell.durationMs = durationMs;
  tell.representative = true;
  tell.commitmentMs = commitmentMs;
}

async function contacts(name, entries, options = {}) {
  const columns = options.columns || 8;
  const width = options.width || 112;
  const height = options.height || 108;
  const imageHeight = height - 26;
  const composites = [];
  for (let i = 0; i < entries.length; i++) {
    const [label, file] = entries[i];
    const left = i % columns * width;
    const top = Math.floor(i / columns) * height;
    const thumb = await sharp(path.join(ROOT, file)).resize(width - 12, imageHeight - 10, { fit: 'contain', background: '#eef0eb' }).flatten({ background: '#eef0eb' }).png().toBuffer();
    composites.push({ input: thumb, left: left + 6, top: top + 5 });
    const lines = label.length > 20 ? [label.slice(0, 20), label.slice(20, 40)] : [label];
    const labelSvg = `<svg width="${width}" height="26" xmlns="http://www.w3.org/2000/svg"><style>text{font:10px Arial;fill:#35413c}</style>${lines.map((s, n) => `<text x="${width / 2}" y="${10 + n * 11}" text-anchor="middle">${escape(s)}</text>`).join('')}</svg>`;
    composites.push({ input: Buffer.from(labelSvg), left, top: top + imageHeight });
  }
  if (!entries.length) return null;
  const filename = `${name}.webp`;
  await sharp({ create: { width: columns * width, height: Math.ceil(entries.length / columns) * height, channels: 3, background: '#eef0eb' } }).composite(composites).webp({ quality: 90 }).toFile(path.join(OUT, filename));
  return { name: title(name), image: filename, count: entries.length, entries: entries.map(([label, file]) => ({ label, image: assetUrl(file) })) };
}

async function main() {
  fs.mkdirSync(path.join(OUT, 'before'), { recursive: true });
  const playerPath = 'img/project-starfall/animations/players/generic-player-sheet.png';
  const beforePlayer = path.join(OUT, 'before/generic-player-sheet.png');
  if (!fs.existsSync(beforePlayer)) fs.writeFileSync(beforePlayer, execFileSync('git', ['show', `HEAD:${playerPath}`], { cwd: ROOT, maxBuffer: 20e6 }));
  assertBaseline(beforePlayer, playerPath);
  const contractsPath = path.join(OUT, 'before/animation-contracts.json');
  let originalContracts;
  if (fs.existsSync(contractsPath)) originalContracts = JSON.parse(fs.readFileSync(contractsPath, 'utf8'));
  else {
    const originalAnimations = moduleAtHead('js/games/project-starfall/data/animations.js').createAnimationData({ CLASS_FILE_IDS: Data.CLASS_FILE_IDS });
    const originalRig = moduleAtHead('js/games/project-starfall/engine/equipment-attachments.js');
    const registrations = {};
    for (const [id, config] of Object.entries(originalAnimations.GENERIC_PLAYER_ANIMATION_ASSET.states)) registrations[id] = Array.from({ length: config.frames }, (_, frame) => originalRig.getPlayerSpriteRegistration(id, frame, { originX: 80, groundY: 154, authoredBodyHeight: 143 }));
    originalContracts = { commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim(), player: originalAnimations.GENERIC_PLAYER_ANIMATION_ASSET, enemies: originalAnimations.ENEMY_ANIMATION_ASSETS, registrations };
    fs.writeFileSync(contractsPath, JSON.stringify(originalContracts, null, 2) + '\n');
  }
  const originalRig = { getPlayerSpriteRegistration: (id, frame, fallback) => originalContracts.registrations[id]?.[frame] || fallback };
  const oldPlayer = { ...originalContracts.player, sheet: path.relative(ROOT, beforePlayer).replace(/\\/g, '/') };
  const actors = [{ id: 'player', name: 'Adventurer', family: 'Player', before: makeClip(oldPlayer, 56, { originX: 80, groundY: 154, authoredBodyHeight: 143 }, originalRig), after: makeClip(Data.GENERIC_PLAYER_ANIMATION_ASSET, 56, Visuals.PLAYER_SPRITE_REGISTRATION || { originX: 80, groundY: 150, authoredBodyHeight: 140 }, Rig) }];
  const inventory = JSON.parse(fs.readFileSync(path.join(SOURCE, 'enemies/inventory.json'), 'utf8'));
  for (const enemy of Data.ENEMIES) {
    const item = inventory.items.find((r) => r.enemies.some((e) => e.id === enemy.id));
    if (!item) throw new Error(`Missing original identity: ${enemy.id}`);
    const original = item.animations.find((r) => r.sheet === originalContracts.enemies[enemy.id]?.sheet) || item.animations[0];
    const oldFile = `asset-sources/project-starfall/overhaul-v1/enemies/originals/${original.sheet}`;
    assertBaseline(path.join(ROOT, oldFile), original.sheet);
    const next = Data.ENEMY_ANIMATION_ASSETS[enemy.id];
    const height = Visuals.getEnemySpriteRenderProfile(enemy).size;
    const oldRegistration = { originX: 64, groundY: 118, authoredBodyHeight: 102 };
    const after = makeClip(next, height, next.registration || oldRegistration);
    applyRepresentativeTell(after, enemy);
    actors.push({ id: enemy.id, name: enemy.name, family: enemy.family, before: makeClip({ ...original, sheet: oldFile }, height, oldRegistration), after });
  }
  const scenery = JSON.parse(fs.readFileSync(path.join(SOURCE, 'scenery/ledger.json'), 'utf8'));
  const icons = JSON.parse(fs.readFileSync(path.join(SOURCE, 'icons/ledger.json'), 'utf8'));
  const backgroundEntries = scenery.assets.filter((r) => r.group === 'backgrounds').map((r) => [title(r.id), r.outputPath]);
  const backgrounds = backgroundEntries.map(([name, file]) => ({ name, image: assetUrl(file) }));
  const scenerySheets = [await contacts('backgrounds', backgroundEntries, { columns: 5, width: 240, height: 142 })];
  for (const group of ['terrain', 'props', 'ramps']) {
    const kits = Array.from(new Map(scenery.assets.filter((r) => r.group === group).map((r) => [r.kit, r])).values());
    scenerySheets.push(await contacts(group, kits.map((r) => [title(r.kit), r.outputPath]), { columns: 3, width: 320, height: group === 'terrain' ? 186 : 144 }));
  }
  scenerySheets.push(await contacts('landmarks-and-stations', scenery.assets.filter((r) => ['structures', 'stations', 'world-map'].includes(r.group)).map((r) => [title(r.id), r.outputPath]), { columns: 3, width: 320, height: 220 }));
  const iconGroups = new Map();
  for (const file of Object.keys(icons.outputs)) {
    const group = file.includes('/skills/') ? 'skills' : file.includes('/cards/') ? 'cards' : file.includes('/items/') ? 'items' : file.includes('/equipment') ? 'equipment' : 'interface';
    if (!iconGroups.has(group)) iconGroups.set(group, []);
    iconGroups.get(group).push([title(path.basename(file, path.extname(file))), file]);
  }
  const iconSheets = [];
  for (const [group, entries] of iconGroups) iconSheets.push(await contacts(`icons-${group}`, entries));
  const data = { actors, backgrounds, counts: { enemyIds: Data.ENEMIES.length, playerActions: Object.keys(actors[0].after.actions).length, scenery: scenery.assets.length, icons: Object.keys(icons.outputs).length }, scenerySheets, iconSheets };
  fs.writeFileSync(path.join(OUT, 'review-data.json'), JSON.stringify(data, null, 2) + '\n');
  const template = fs.readFileSync(path.join(ROOT, 'build/templates/starfall-overhaul-review.template.html'), 'utf8');
  fs.writeFileSync(path.join(OUT, 'index.html'), template.replace('__REVIEW_DATA__', JSON.stringify(data).replace(/</g, '\\u003c')));
  console.log(`Generated review: ${actors.length} actors, ${data.counts.playerActions} player actions, ${scenerySheets.length + iconSheets.length} static contact sheets. Every before image matches the captured baseline.`);
}
main().catch((e) => { console.error(e); process.exit(1); });
