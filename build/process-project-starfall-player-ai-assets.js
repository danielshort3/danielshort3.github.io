#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const Data = require('../js/games/project-starfall/project-starfall-data.js');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_PATH = path.join(ROOT, 'img/project-starfall/characters/source/ai-class-sheet.png');
const CHARACTER_DIR = path.join(ROOT, 'img/project-starfall/characters');
const PLAYER_SHEET_DIR = path.join(ROOT, 'img/project-starfall/animations/players');
const EQUIPMENT_DIR = path.join(ROOT, 'img/project-starfall/equipment-layers');
const CHARACTER_SIZE = 320;
const FRAME_SIZE = 160;
const SHEET_COLS = 6;
const PLAYER_ROWS = Object.freeze(['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat']);

function parseAssetFrame(assetPath) {
  const value = String(assetPath || '').trim();
  const hashIndex = value.indexOf('#');
  if (hashIndex < 0) return { path: value };
  const assetPathOnly = value.slice(0, hashIndex);
  const fragment = value.slice(hashIndex + 1);
  const match = fragment.match(/(?:^|[;&])(?:frame|xywh)=([0-9.,-]+)/);
  if (!match) return { path: assetPathOnly };
  const parts = match[1].split(',').map((part) => Number(part));
  return {
    path: assetPathOnly,
    frame: {
      left: Math.max(0, Math.floor(parts[0]) || 0),
      top: Math.max(0, Math.floor(parts[1]) || 0),
      width: Math.max(1, Math.floor(parts[2]) || 1),
      height: Math.max(1, Math.floor(parts[3]) || 1)
    }
  };
}

async function readItemIcon(iconPath) {
  const descriptor = parseAssetFrame(iconPath);
  const absolutePath = path.join(ROOT, descriptor.path || '');
  if (!descriptor.path || !fs.existsSync(absolutePath)) return null;
  let pipeline = sharp(absolutePath).ensureAlpha();
  if (descriptor.frame) pipeline = pipeline.extract(descriptor.frame);
  return pipeline.png().toBuffer();
}

const SOURCE_CLASS_CELLS = Object.freeze({
  fighter: Object.freeze({ row: 0, col: 0 }),
  mage: Object.freeze({ row: 0, col: 1 }),
  archer: Object.freeze({ row: 0, col: 2 }),
  guardian: Object.freeze({ row: 1, col: 0 }),
  berserker: Object.freeze({ row: 1, col: 1 }),
  fireMage: Object.freeze({ row: 1, col: 2 }),
  runeMage: Object.freeze({ row: 2, col: 0 }),
  sniper: Object.freeze({ row: 2, col: 1 }),
  trapper: Object.freeze({ row: 2, col: 2 })
});

const DERIVED_CLASS_SOURCES = Object.freeze({
  duelist: Object.freeze({ source: 'fighter', hue: 28, saturation: 1.06, brightness: 1.06 }),
  stormMage: Object.freeze({ source: 'mage', hue: 190, saturation: 1.12, brightness: 1.08 }),
  beastArcher: Object.freeze({ source: 'archer', hue: 72, saturation: 1.04, brightness: 1.02 })
});

const CLASS_ACCENTS = Object.freeze({
  fighter: '#f0c36a',
  mage: '#8bd7ff',
  archer: '#ffe16a',
  guardian: '#d5ecff',
  berserker: '#ffb35c',
  duelist: '#ffffff',
  fireMage: '#ffd36b',
  runeMage: '#b8fff2',
  stormMage: '#ffffff',
  sniper: '#fff0a6',
  trapper: '#b7c3ca',
  beastArcher: '#fff0a6'
});

function ensureDirs() {
  [CHARACTER_DIR, PLAYER_SHEET_DIR, EQUIPMENT_DIR].forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
}

function sourceCellRect(meta, cell) {
  const cellW = Math.floor(meta.width / 3);
  const cellH = Math.floor(meta.height / 3);
  const left = Math.max(0, Math.round(cell.col * cellW));
  const top = Math.max(0, Math.round(cell.row * cellH));
  const width = cell.col === 2 ? meta.width - left : cellW;
  const height = cell.row === 2 ? meta.height - top : cellH;
  return { left, top, width, height };
}

function removeChromaKey(raw) {
  const data = Buffer.from(raw.data);
  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const isMagentaKey = r > 170 && b > 145 && g < 95;
    const isGreenKey = g > 170 && r < 95 && b < 95;
    if (isMagentaKey || isGreenKey) {
      data[index + 3] = 0;
      continue;
    }
    if (r > 150 && b > 130 && g < 120) {
      data[index] = Math.max(0, r - 18);
      data[index + 2] = Math.max(0, b - 18);
    }
  }
  return sharp(data, { raw: raw.info }).png().toBuffer();
}

async function extractClassSprite(sourceMeta, cell) {
  const rect = sourceCellRect(sourceMeta, cell);
  const raw = await sharp(SOURCE_PATH)
    .extract(rect)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return sharp(await removeChromaKey(raw)).trim({ threshold: 8 }).png().toBuffer();
}

async function deriveSprite(sprite, transform) {
  if (!transform) return sprite;
  let pipeline = sharp(sprite);
  if (transform.hue || transform.brightness || transform.saturation) {
    pipeline = pipeline.modulate({
      hue: Number(transform.hue || 0),
      brightness: Number(transform.brightness || 1),
      saturation: Number(transform.saturation || 1)
    });
  }
  return pipeline.png().toBuffer();
}

async function placeOnCanvas(sprite, width, height, options) {
  const maxWidth = Math.max(1, Number(options && options.maxWidth || width));
  const maxHeight = Math.max(1, Number(options && options.maxHeight || height));
  let pipeline = sharp(sprite).resize(Math.round(maxWidth), Math.round(maxHeight), {
    fit: 'inside',
    withoutEnlargement: false,
    kernel: 'lanczos3',
    background: { r: 0, g: 0, b: 0, alpha: 0 }
  });
  if (options && options.rotate) {
    pipeline = pipeline.rotate(Number(options.rotate || 0), {
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    });
  }
  if (options && options.tint) {
    pipeline = pipeline.modulate({
      brightness: Number(options.tint.brightness || 1),
      saturation: Number(options.tint.saturation || 1)
    });
  }
  let rendered = await pipeline.png().toBuffer();
  let meta = await sharp(rendered).metadata();
  if (meta.width > width || meta.height > height) {
    rendered = await sharp(rendered)
      .resize(width, height, {
        fit: 'inside',
        withoutEnlargement: true,
        kernel: 'lanczos3',
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toBuffer();
    meta = await sharp(rendered).metadata();
  }
  const left = Math.max(0, Math.min(width - meta.width, Math.round((width - meta.width) / 2 + Number(options && options.dx || 0))));
  const top = Math.max(0, Math.min(height - meta.height, Math.round(height - meta.height - Number(options && options.bottom || 0) + Number(options && options.dy || 0))));
  return sharp({
    create: {
      width,
      height,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite([{ input: rendered, left, top }])
    .png({ compressionLevel: 9 })
    .toBuffer();
}

function framePose(stateId, frame, classId) {
  const phase = frame / Math.max(1, SHEET_COLS - 1);
  const wave = Math.sin(phase * Math.PI * 2);
  const bounce = Math.abs(wave);
  const pose = { dx: 0, dy: 0, rotate: 0, scale: 1, bottom: 12 };
  if (stateId === 'idle') {
    pose.dy = -bounce * 3;
    pose.rotate = wave * 1.6;
  } else if (stateId === 'run') {
    pose.dx = wave * 9;
    pose.dy = -bounce * 8;
    pose.rotate = wave * 5;
  } else if (stateId === 'jump') {
    pose.dy = -18 - Math.sin(phase * Math.PI) * 22;
    pose.rotate = -5 + frame * 2.1;
  } else if (stateId === 'fall') {
    pose.dy = -10 + frame * 5;
    pose.rotate = 8 - frame * 1.4;
  } else if (stateId === 'climb') {
    pose.dx = wave * 5;
    pose.dy = -frame * 4;
    pose.rotate = wave * 7;
  } else if (stateId === 'basic') {
    pose.dx = frame < 3 ? frame * 8 : (5 - frame) * 8;
    pose.dy = -bounce * 5;
    pose.rotate = -9 + frame * 4.2;
  } else if (stateId === 'skill') {
    pose.dx = Math.cos(phase * Math.PI * 2) * 5;
    pose.dy = -8 - Math.sin(phase * Math.PI) * 8;
    pose.rotate = wave * 8;
    pose.scale = 1.02 + bounce * 0.05;
  } else if (stateId === 'party') {
    pose.dy = -bounce * 6;
    pose.rotate = wave * 4;
    pose.scale = 0.98 + bounce * 0.04;
  } else if (stateId === 'hit') {
    pose.dx = frame % 2 ? -8 : 7;
    pose.rotate = frame % 2 ? -10 : 8;
  } else if (stateId === 'defeat') {
    pose.dx = -8 + frame * 4;
    pose.dy = 18 + frame * 4;
    pose.rotate = 52 + frame * 8;
    pose.scale = Math.max(0.76, 1 - frame * 0.04);
    pose.bottom = 5;
  }
  if (classId === 'sniper' || classId === 'beastArcher') pose.dx += 2;
  return pose;
}

function effectOverlay(stateId, frame, classId) {
  if (stateId !== 'skill' && stateId !== 'party') return null;
  const accent = CLASS_ACCENTS[classId] || '#ffe16a';
  const alpha = stateId === 'skill' ? 0.72 : 0.48;
  const radius = stateId === 'skill' ? 34 + frame * 4 : 44 + frame * 2;
  const y = stateId === 'skill' ? 78 : 104;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${FRAME_SIZE}" height="${FRAME_SIZE}" viewBox="0 0 ${FRAME_SIZE} ${FRAME_SIZE}">
    <ellipse cx="82" cy="${y}" rx="${radius}" ry="${Math.round(radius * 0.42)}" fill="none" stroke="${accent}" stroke-width="5" stroke-opacity="${alpha}"/>
    <circle cx="${104 + frame * 3}" cy="${52 + frame % 3 * 8}" r="${6 + frame % 2 * 3}" fill="${accent}" fill-opacity="${alpha}"/>
    <circle cx="${48 - frame * 2}" cy="${68 + frame % 2 * 9}" r="${4 + frame % 3}" fill="#ffffff" fill-opacity="${alpha * 0.7}"/>
  </svg>`;
  return Buffer.from(svg);
}

async function createFrame(sprite, stateId, frame, classId) {
  const pose = framePose(stateId, frame, classId);
  const base = await placeOnCanvas(sprite, FRAME_SIZE, FRAME_SIZE, {
    maxWidth: 112 * pose.scale,
    maxHeight: 140 * pose.scale,
    dx: pose.dx,
    dy: pose.dy,
    bottom: pose.bottom,
    rotate: pose.rotate,
    tint: stateId === 'hit' && frame % 2 ? { brightness: 1.35, saturation: 0.7 } : null
  });
  const overlay = effectOverlay(stateId, frame, classId);
  if (!overlay) return base;
  return sharp(base)
    .composite([{ input: overlay, left: 0, top: 0 }])
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function writeCharacterPortrait(classId, fileId, sprite) {
  const output = await placeOnCanvas(sprite, CHARACTER_SIZE, CHARACTER_SIZE, {
    maxWidth: 246,
    maxHeight: 284,
    bottom: 16
  });
  const outPath = path.join(CHARACTER_DIR, `${fileId}.png`);
  await sharp(output).toFile(outPath);
  return path.relative(ROOT, outPath).replace(/\\/g, '/');
}

async function writePlayerSheet(classId, fileId, sprite) {
  const cells = [];
  for (const stateId of PLAYER_ROWS) {
    for (let frame = 0; frame < SHEET_COLS; frame += 1) {
      cells.push(await createFrame(sprite, stateId, frame, classId));
    }
  }
  const composites = cells.map((input, index) => ({
    input,
    left: (index % SHEET_COLS) * FRAME_SIZE,
    top: Math.floor(index / SHEET_COLS) * FRAME_SIZE
  }));
  const outPath = path.join(PLAYER_SHEET_DIR, `${fileId}-sheet.png`);
  await sharp({
    create: {
      width: SHEET_COLS * FRAME_SIZE,
      height: PLAYER_ROWS.length * FRAME_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(outPath);
  return path.relative(ROOT, outPath).replace(/\\/g, '/');
}

function equipmentPose(layer, stateId, frame) {
  const phase = frame / Math.max(1, SHEET_COLS - 1);
  const wave = Math.sin(phase * Math.PI * 2);
  const base = { dx: 26, dy: -58, rotate: 0, size: 48 };
  if (layer === 'chest') return { dx: 0, dy: -78 + Math.abs(wave) * -3, rotate: wave * 2, size: 54 };
  if (layer === 'boots') return { dx: wave * 5, dy: -14, rotate: wave * 5, size: 42 };
  if (layer === 'head') return { dx: 1 + wave * 2, dy: -114 + Math.abs(wave) * -2, rotate: wave * 3, size: 42 };
  if (layer === 'gloves') return { dx: 28 + wave * 5, dy: -57 + Math.abs(wave) * -3, rotate: -8 + wave * 8, size: 34 };
  if (layer === 'aura') return { dx: 0, dy: -72, rotate: frame * 7, size: 76 };
  if (layer === 'offhand') return { dx: -28 + wave * 3, dy: -58, rotate: -10 + wave * 4, size: 50 };
  if (stateId === 'basic') return { dx: 36 + frame * 5, dy: -64 + wave * 6, rotate: -24 + frame * 10, size: 64 };
  if (stateId === 'skill') return { dx: 28 + wave * 8, dy: -70, rotate: frame * 14, size: 62 };
  if (stateId === 'run') return { dx: 26 + wave * 8, dy: -56 + Math.abs(wave) * -4, rotate: wave * 10, size: 56 };
  if (stateId === 'defeat') return { dx: 18 + frame * 3, dy: -20, rotate: 55 + frame * 7, size: 58 };
  return base;
}

async function writeEquipmentFrame(iconBuffer, layer, stateId, frame) {
  const pose = equipmentPose(layer, stateId, frame);
  let item = await sharp(iconBuffer)
    .resize(pose.size, pose.size, {
      fit: 'inside',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .rotate(pose.rotate, { background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toBuffer();
  let meta = await sharp(item).metadata();
  if (meta.width > FRAME_SIZE || meta.height > FRAME_SIZE) {
    item = await sharp(item)
      .resize(FRAME_SIZE, FRAME_SIZE, {
        fit: 'inside',
        withoutEnlargement: true,
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toBuffer();
    meta = await sharp(item).metadata();
  }
  const left = Math.max(0, Math.min(FRAME_SIZE - meta.width, Math.round(FRAME_SIZE / 2 + pose.dx - meta.width / 2)));
  const top = Math.max(0, Math.min(FRAME_SIZE - meta.height, Math.round(FRAME_SIZE + pose.dy - meta.height / 2)));
  return sharp({
    create: {
      width: FRAME_SIZE,
      height: FRAME_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite([{ input: item, left, top }])
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function writeEquipmentSheet(itemId, visual) {
  const iconPath = Data.ITEM_ASSETS && (Data.ITEM_ASSETS[visual.assetId] || Data.ITEM_ASSETS[itemId]);
  const iconBuffer = await readItemIcon(iconPath);
  if (!iconBuffer) return null;
  const cells = [];
  for (const stateId of PLAYER_ROWS) {
    for (let frame = 0; frame < SHEET_COLS; frame += 1) {
      cells.push(await writeEquipmentFrame(iconBuffer, visual.layer, stateId, frame));
    }
  }
  const composites = cells.map((input, index) => ({
    input,
    left: (index % SHEET_COLS) * FRAME_SIZE,
    top: Math.floor(index / SHEET_COLS) * FRAME_SIZE
  }));
  const outPath = path.join(ROOT, visual.animation.sheet);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  await sharp({
    create: {
      width: SHEET_COLS * FRAME_SIZE,
      height: PLAYER_ROWS.length * FRAME_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(outPath);
  return path.relative(ROOT, outPath).replace(/\\/g, '/');
}

async function buildClassSprites() {
  if (!fs.existsSync(SOURCE_PATH)) {
    throw new Error(`Missing Project Starfall AI class source sheet: ${path.relative(ROOT, SOURCE_PATH)}`);
  }
  const sourceMeta = await sharp(SOURCE_PATH).metadata();
  const sprites = {};
  for (const [classId, cell] of Object.entries(SOURCE_CLASS_CELLS)) {
    sprites[classId] = await extractClassSprite(sourceMeta, cell);
  }
  for (const [classId, transform] of Object.entries(DERIVED_CLASS_SOURCES)) {
    sprites[classId] = await deriveSprite(sprites[transform.source], transform);
  }
  return sprites;
}

async function validatePng(filePath, width, height) {
  const meta = await sharp(filePath).metadata();
  if (meta.width !== width || meta.height !== height) {
    throw new Error(`${path.relative(ROOT, filePath)} is ${meta.width}x${meta.height}; expected ${width}x${height}`);
  }
}

async function generateAll() {
  ensureDirs();
  const sprites = await buildClassSprites();
  const generated = [];
  for (const [classId, fileId] of Object.entries(Data.CLASS_FILE_IDS || {})) {
    const sprite = sprites[classId];
    if (!sprite) throw new Error(`Missing AI class sprite for ${classId}`);
    generated.push(await writeCharacterPortrait(classId, fileId, sprite));
    generated.push(await writePlayerSheet(classId, fileId, sprite));
  }
  for (const [itemId, visual] of Object.entries(Data.EQUIPMENT_VISUALS || {})) {
    const generatedSheet = await writeEquipmentSheet(itemId, visual);
    if (generatedSheet) generated.push(generatedSheet);
  }
  generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
}

async function validateAll() {
  await validatePng(path.join(PLAYER_SHEET_DIR, 'generic-player-sheet.png'), SHEET_COLS * FRAME_SIZE, PLAYER_ROWS.length * FRAME_SIZE);
  for (const fileId of Object.values(Data.CLASS_FILE_IDS || {})) {
    await validatePng(path.join(CHARACTER_DIR, `${fileId}.png`), CHARACTER_SIZE, CHARACTER_SIZE);
    await validatePng(path.join(PLAYER_SHEET_DIR, `${fileId}-sheet.png`), SHEET_COLS * FRAME_SIZE, PLAYER_ROWS.length * FRAME_SIZE);
  }
  for (const visual of Object.values(Data.EQUIPMENT_VISUALS || {})) {
    await validatePng(path.join(ROOT, visual.animation.sheet), SHEET_COLS * FRAME_SIZE, PLAYER_ROWS.length * FRAME_SIZE);
  }
  console.log(`Validated generic paper-doll sheet, ${Object.keys(Data.CLASS_FILE_IDS || {}).length} class portraits/player sheets, and ${Object.keys(Data.EQUIPMENT_VISUALS || {}).length} equipment layer sheets`);
}

async function main() {
  if (process.argv.includes('--validate')) {
    await validateAll();
    return;
  }
  await generateAll();
  await validateAll();
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exit(1);
  });
}
