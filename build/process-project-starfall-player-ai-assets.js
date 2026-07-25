#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const Data = require('../js/games/project-starfall/project-starfall-data.js');
const EquipmentAttachments = require('../js/games/project-starfall/engine/equipment-attachments.js');
const {
  GUIDE_LINE_HEX,
  detectGuideGrid,
  getGridCellRect,
  isGuidePixelRgba
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_ROOT = path.join(ROOT, 'asset-sources/project-starfall/players');
const CLASS_SOURCE_DIR = path.join(SOURCE_ROOT, 'classes');
const REFERENCE_PATH = path.join(SOURCE_ROOT, 'starfall-chibi-equipment-reference.png');
const BASE_SPRITE_PATH = path.join(SOURCE_ROOT, 'plain-adventurer-base.png');
const APPROVED_ANIMATION_SOURCE_PATH = path.join(SOURCE_ROOT, 'generic-player-v2-generated-source.png');
const REVIEW_CONTACT_SHEET_PATH = path.join(SOURCE_ROOT, 'plain-adventurer-review.png');
const CHARACTER_DIR = path.join(ROOT, 'img/project-starfall/characters');
const PLAYER_SHEET_DIR = path.join(ROOT, 'img/project-starfall/animations/players');

const STYLE_REFERENCE_SOURCE = '/home/sd205521/.codex/generated_images/019e9e04-d798-7ce3-89ea-0fac77c06988/ig_0d2f4bc617b8d0f0016a257b3533108194b32a4850b76a73b2.png';

const CHARACTER_SIZE = 320;
const FRAME_SIZE = 160;
const GUIDE_WIDTH = 1;
const SHEET_COLS = 6;
const PLAYER_ROWS = Object.freeze((Data.PLAYER_ANIMATION_ROWS || ['idle', 'run', 'jump', 'fall', 'climb', 'basic', 'skill', 'party', 'hit', 'defeat']).slice());
const PLAYER_ROW_INDEX = Object.freeze(PLAYER_ROWS.reduce((rows, rowId, rowIndex) => {
  rows[rowId] = rowIndex;
  return rows;
}, {}));
const EQUIPMENT_VISUALS_BY_FILE_ID = Object.freeze(Object.values(Data.EQUIPMENT_VISUALS || {}).reduce((visuals, visual) => {
  if (visual && visual.fileId && !visuals[visual.fileId]) visuals[visual.fileId] = visual;
  return visuals;
}, {}));
const EQUIPMENT_PREVIEW_PADDING = 128;

const REFERENCE_FIRST_ADVENTURER_CROP = Object.freeze({
  left: 55,
  top: 165,
  width: 260,
  height: 500
});

const HEAD_PART_DEF = Object.freeze({
  predicate: (x, y) => y <= 170 && x <= 186
});

const SOURCE_PART_DEFS = Object.freeze({
  head: Object.freeze({
    predicate: (x, y) => y <= 170 && x <= 186,
    pivot: Object.freeze([94, 118])
  }),
  backLeg: Object.freeze({
    predicate: (x, y) => y >= 254 && x >= 96 && x <= 168,
    pivot: Object.freeze([126, 267])
  }),
  frontLeg: Object.freeze({
    predicate: (x, y) => y >= 254 && x >= 34 && x <= 103,
    pivot: Object.freeze([72, 267])
  }),
  backArm: Object.freeze({
    predicate: (x, y) => y >= 160 && y <= 292 && x >= 124,
    pivot: Object.freeze([139, 177])
  }),
  torso: Object.freeze({
    predicate: (x, y) => y >= 142 && y <= 296 && x >= 48 && x <= 150,
    pivot: Object.freeze([96, 220])
  }),
  frontArm: Object.freeze({
    predicate: (x, y) => y >= 146 && y <= 294 && x <= 78,
    pivot: Object.freeze([61, 176])
  })
});

const SOURCE_PART_ORDER = Object.freeze(['backLeg', 'frontLeg', 'backArm', 'torso', 'head', 'frontArm']);

const CLASS_COMBAT_STYLES = Object.freeze({
  fighter: 'melee',
  guardian: 'melee',
  berserker: 'melee',
  duelist: 'melee',
  mage: 'magic',
  fireMage: 'magic',
  runeMage: 'magic',
  stormMage: 'magic',
  archer: 'bow',
  sniper: 'bow',
  trapper: 'bow',
  beastArcher: 'bow'
});

const BODY_COLORS = Object.freeze({
  outline: '#11191f',
  outlineSoft: '#1b2630',
  skin: '#f6c28d',
  skinShade: '#c98250',
  shirt: '#efe3c8',
  shirtShade: '#c99b68',
  shirtShadow: '#8a6546',
  pants: '#1c2525',
  pantsLight: '#32403b',
  belt: '#5b3b25',
  buckle: '#d9a75a',
  boot: '#6b3d22',
  bootDark: '#2b1c16',
  bootLight: '#a66b35'
});

const BODY_ANCHOR = Object.freeze({
  headX: 80,
  headTop: 17,
  shoulderBackX: 97,
  shoulderFrontX: 53,
  shoulderY: 83,
  hipBackX: 83,
  hipFrontX: 65,
  hipY: 112,
  bootY: 141
});

function ensureDirs() {
  [
    SOURCE_ROOT,
    CLASS_SOURCE_DIR,
    CHARACTER_DIR,
    PLAYER_SHEET_DIR
  ].forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
}

function copyReferenceIfAvailable() {
  if (fs.existsSync(REFERENCE_PATH)) return;
  if (!fs.existsSync(STYLE_REFERENCE_SOURCE)) return;
  fs.mkdirSync(path.dirname(REFERENCE_PATH), { recursive: true });
  fs.copyFileSync(STYLE_REFERENCE_SOURCE, REFERENCE_PATH);
}

function toRepoPath(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function attr(attrs) {
  return attrs ? ` ${attrs}` : '';
}

function rect(x, y, w, h, color, attrs) {
  return `<rect x="${Math.round(x)}" y="${Math.round(y)}" width="${Math.max(1, Math.round(w))}" height="${Math.max(1, Math.round(h))}" fill="${color}"${attr(attrs)}/>`;
}

function ellipse(cx, cy, rx, ry, color, attrs) {
  return `<ellipse cx="${Math.round(cx)}" cy="${Math.round(cy)}" rx="${Math.max(1, Math.round(rx))}" ry="${Math.max(1, Math.round(ry))}" fill="${color}"${attr(attrs)}/>`;
}

function polygon(points, color, attrs) {
  const pointList = points.map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).join(' ');
  return `<polygon points="${pointList}" fill="${color}"${attr(attrs)}/>`;
}

function svgPath(data, color, attrs) {
  return `<path d="${data}" fill="none" stroke="${color}" stroke-linecap="round" stroke-linejoin="round"${attr(attrs)}/>`;
}

function svg(width, height, body) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="crispEdges">${body}</svg>`);
}

function makeGridOverlay(width, height) {
  const parts = [];
  for (let col = 0; col <= SHEET_COLS; col += 1) {
    parts.push(rect(col * (FRAME_SIZE + GUIDE_WIDTH), 0, GUIDE_WIDTH, height, GUIDE_LINE_HEX));
  }
  for (let row = 0; row <= PLAYER_ROWS.length; row += 1) {
    parts.push(rect(0, row * (FRAME_SIZE + GUIDE_WIDTH), width, GUIDE_WIDTH, GUIDE_LINE_HEX));
  }
  return svg(width, height, parts.join(''));
}

function makeLabelSvg(text, width, height) {
  const safeText = String(text || '').replace(/[<>&]/g, '');
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <text x="0" y="22" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#d9e8f2">${safeText}</text>
  </svg>`);
}

function isReferenceBackgroundPixel(data, index) {
  const offset = index * 4;
  const r = data[offset];
  const g = data[offset + 1];
  const b = data[offset + 2];
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const brightness = (r + g + b) / 3;
  return brightness >= 10 && brightness <= 92 && max - min <= 42;
}

function removeReferenceBackground(raw, width, height) {
  const data = Buffer.from(raw);
  const background = new Uint8Array(width * height);
  const queue = [];
  const push = (x, y) => {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const index = y * width + x;
    if (background[index]) return;
    if (!isReferenceBackgroundPixel(data, index)) return;
    background[index] = 1;
    queue.push(index);
  };

  for (let x = 0; x < width; x += 1) {
    push(x, 0);
    push(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    push(0, y);
    push(width - 1, y);
  }
  for (let index = 0; index < queue.length; index += 1) {
    const pixel = queue[index];
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    push(x + 1, y);
    push(x - 1, y);
    push(x, y + 1);
    push(x, y - 1);
  }

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (background[pixel]) {
      data[offset] = 0;
      data[offset + 1] = 0;
      data[offset + 2] = 0;
      data[offset + 3] = 0;
    } else if (data[offset + 3] > 0) {
      data[offset + 3] = 255;
    }
  }
  return data;
}

function removeSmallAlphaComponents(raw, width, height, minPixels) {
  const data = Buffer.from(raw);
  const visited = new Uint8Array(width * height);
  const threshold = Number.isFinite(minPixels) ? minPixels : 500;
  for (let start = 0; start < width * height; start += 1) {
    if (visited[start] || data[start * 4 + 3] <= 20) continue;
    const queue = [start];
    const pixels = [start];
    visited[start] = 1;
    for (let index = 0; index < queue.length; index += 1) {
      const pixel = queue[index];
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (!dx && !dy) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (visited[next] || data[next * 4 + 3] <= 20) continue;
          visited[next] = 1;
          queue.push(next);
          pixels.push(next);
        }
      }
    }
    if (pixels.length >= threshold) continue;
    pixels.forEach((pixel) => {
      const offset = pixel * 4;
      data[offset] = 0;
      data[offset + 1] = 0;
      data[offset + 2] = 0;
      data[offset + 3] = 0;
    });
  }
  return data;
}

function getAlphaBounds(raw, width, height, threshold) {
  let count = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  const alphaThreshold = threshold == null ? 20 : Number(threshold);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const alpha = raw[(y * width + x) * 4 + 3];
      if (alpha <= alphaThreshold) continue;
      count += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (!count) return null;
  return Object.freeze({
    count,
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2
  });
}

async function writePlainAdventurerBaseSprite() {
  copyReferenceIfAvailable();
  if (!fs.existsSync(REFERENCE_PATH)) {
    throw new Error(`Missing Project Starfall player reference: ${toRepoPath(REFERENCE_PATH)}`);
  }
  const decoded = await sharp(REFERENCE_PATH)
    .extract(REFERENCE_FIRST_ADVENTURER_CROP)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const raw = removeSmallAlphaComponents(
    removeReferenceBackground(decoded.data, decoded.info.width, decoded.info.height),
    decoded.info.width,
    decoded.info.height,
    500
  );
  const bounds = getAlphaBounds(raw, decoded.info.width, decoded.info.height, 20);
  if (!bounds) throw new Error('Could not isolate the plain adventurer from the reference image');
  await sharp(raw, { raw: { width: decoded.info.width, height: decoded.info.height, channels: 4 } })
    .extract({
      left: bounds.minX,
      top: bounds.minY,
      width: bounds.width,
      height: bounds.height
    })
    .png({ compressionLevel: 9 })
    .toFile(BASE_SPRITE_PATH);
  return BASE_SPRITE_PATH;
}

async function extractMaskedPart(name, definition, raw, width, height) {
  const output = Buffer.alloc(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= 20 || !definition.predicate(x, y)) continue;
      output[offset] = raw[offset];
      output[offset + 1] = raw[offset + 1];
      output[offset + 2] = raw[offset + 2];
      output[offset + 3] = raw[offset + 3];
    }
  }
  const bounds = getAlphaBounds(output, width, height, 20);
  if (!bounds) throw new Error(`Could not extract player rig part: ${name}`);
  const buffer = await sharp(output, { raw: { width, height, channels: 4 } })
    .extract({
      left: bounds.minX,
      top: bounds.minY,
      width: bounds.width,
      height: bounds.height
    })
    .png({ compressionLevel: 9 })
    .toBuffer();
  return Object.freeze({
    name,
    order: definition.order,
    buffer,
    width: bounds.width,
    height: bounds.height,
    sourceLeft: bounds.minX,
    sourceTop: bounds.minY,
    pivotX: definition.pivot ? definition.pivot[0] - bounds.minX : bounds.width / 2,
    pivotY: definition.pivot ? definition.pivot[1] - bounds.minY : bounds.height / 2,
    bounds
  });
}

async function buildPlayerRig() {
  const decoded = await sharp(BASE_SPRITE_PATH).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const width = decoded.info.width;
  const height = decoded.info.height;
  const raw = Buffer.from(decoded.data);
  const parts = {};
  for (const [name, definition] of Object.entries(SOURCE_PART_DEFS)) {
    parts[name] = await extractMaskedPart(name, definition, raw, width, height);
  }
  return Object.freeze({
    width,
    height,
    body: Object.freeze({
      buffer: await sharp(BASE_SPRITE_PATH).png({ compressionLevel: 9 }).toBuffer(),
      width,
      height
    }),
    head: parts.head,
    parts: Object.freeze(parts)
  });
}

function makePose(config) {
  return Object.assign({
    x: 0,
    y: 0,
    height: 128,
    shadow: true,
    shadowScale: 1,
    shadowAlpha: 0.24,
    aura: '',
    overlay: '',
    sourceBody: false,
    sourceParts: true,
    parts: {}
  }, config || {});
}

function getClassCombatStyle(classId) {
  return CLASS_COMBAT_STYLES[classId] || 'melee';
}

function makeConnectedPose(config) {
  const settings = config || {};
  return makePose(Object.assign({}, settings, {
    sourceBody: true,
    sourceParts: false,
    parts: {
      torso: Object.assign({}, settings.torso || {})
    }
  }));
}

function getPose(rowId, frame, combatStyle) {
  const style = combatStyle || 'melee';
  const phase = frame / Math.max(1, SHEET_COLS - 1);
  const wave = Math.sin(phase * Math.PI * 2);
  const bounce = Math.abs(wave);

  if (rowId === 'idle') {
    return makePose({
      sourceBody: true,
      overlay: 'idle',
      shadowAlpha: 0.22,
      parts: {
        head: {},
        torso: {},
        backArm: {},
        frontArm: {},
        backLeg: { dy: 0 },
        frontLeg: { dy: 0 }
      }
    });
  }

  if (rowId === 'run') {
    const cycle = [
      { x: -4, y: 2, rotate: -3, overlay: 'runStepA' },
      { x: -2, y: -1, rotate: -1, overlay: 'runPassA' },
      { x: 1, y: 0, rotate: 1, overlay: 'runStepB' },
      { x: 4, y: 2, rotate: 2, overlay: 'runStepB' },
      { x: 2, y: -1, rotate: 1, overlay: 'runPassB' },
      { x: -2, y: 0, rotate: -1, overlay: 'runStepA' }
    ][frame] || {};
    return makeConnectedPose({
      x: cycle.x || 0,
      y: cycle.y || 0,
      torso: { rotate: cycle.rotate || 0 },
      shadowScale: frame % 3 === 1 ? 0.94 : 1.05,
      shadowAlpha: 0.22,
      overlay: cycle.overlay
    });
  }

  if (rowId === 'jump') {
    const cycle = [
      { y: 4, rotate: -2, shadowScale: 0.96, shadowAlpha: 0.23, overlay: 'landingDust' },
      { y: -6, rotate: -1, shadowScale: 0.84, shadowAlpha: 0.15 },
      { y: -13, rotate: 0, shadowScale: 0.7, shadowAlpha: 0.06 },
      { y: -15, rotate: 1, shadowScale: 0.64, shadowAlpha: 0.05 },
      { y: -8, rotate: 0, shadowScale: 0.82, shadowAlpha: 0.14 },
      { y: 3, rotate: -1, shadowScale: 0.98, shadowAlpha: 0.22, overlay: 'landingDust' }
    ][frame] || {};
    return makeConnectedPose({
      y: cycle.y || 0,
      torso: { rotate: cycle.rotate || 0 },
      shadowScale: cycle.shadowScale,
      shadowAlpha: cycle.shadowAlpha == null ? 0.16 : cycle.shadowAlpha,
      overlay: cycle.overlay || ''
    });
  }

  if (rowId === 'fall') {
    const cycle = [
      { y: -14, rotate: 1, shadowScale: 0.72, shadowAlpha: 0.09 },
      { y: -9, rotate: 1, shadowScale: 0.78, shadowAlpha: 0.12 },
      { y: -4, rotate: 0, shadowScale: 0.84, shadowAlpha: 0.16 },
      { y: 1, rotate: -1, shadowScale: 0.9, shadowAlpha: 0.2 },
      { y: 4, rotate: -1, shadowScale: 0.97, shadowAlpha: 0.23 },
      { y: 6, rotate: 0, shadowScale: 1.04, shadowAlpha: 0.24, overlay: 'landingDust' }
    ][frame] || {};
    return makeConnectedPose({
      y: cycle.y || 0,
      torso: { rotate: cycle.rotate || 0 },
      shadowScale: cycle.shadowScale,
      shadowAlpha: cycle.shadowAlpha,
      overlay: cycle.overlay || ''
    });
  }

  if (rowId === 'climb') {
    const cycle = [
      { x: -2, y: -2, rotate: -2 },
      { x: -1, y: -5, rotate: -1 },
      { x: 1, y: -3, rotate: 1 },
      { x: 2, y: -1, rotate: 2 },
      { x: 1, y: -4, rotate: 1 },
      { x: -1, y: -5, rotate: -1 }
    ][frame] || {};
    return makeConnectedPose({
      x: cycle.x || 0,
      y: cycle.y || 0,
      torso: { rotate: cycle.rotate || 0 },
      shadow: false,
      overlay: 'climb'
    });
  }

  if (rowId === 'basic') {
    const cycles = {
      melee: [
        { x: -3, y: 0, torso: { rotate: -3 }, head: { rotate: -1 }, frontArm: { dx: -5, dy: -4, rotate: -24 }, backArm: { dx: -3, dy: 1, rotate: -13 }, frontLeg: { dx: -2, dy: 1, rotate: -8 }, backLeg: { dx: 2, rotate: 7 }, overlay: 'meleeWindup' },
        { x: -6, y: -1, torso: { rotate: -5 }, head: { rotate: -2 }, frontArm: { dx: -8, dy: -5, rotate: -34 }, backArm: { dx: -4, dy: 0, rotate: -18 }, frontLeg: { dx: -3, dy: 1, rotate: -11 }, backLeg: { dx: 3, rotate: 9 }, overlay: 'meleeWindup' },
        { x: 6, y: -1, torso: { rotate: 4 }, head: { dx: 1, rotate: 1 }, frontArm: { dx: 12, dy: -5, rotate: 34 }, backArm: { dx: 5, dy: -1, rotate: 18 }, frontLeg: { dx: 7, dy: 1, rotate: 19 }, backLeg: { dx: -5, dy: 1, rotate: -15 }, overlay: 'meleeStrike' },
        { x: 8, y: 0, torso: { rotate: 5 }, head: { dx: 1, rotate: 1 }, frontArm: { dx: 15, dy: -2, rotate: 38 }, backArm: { dx: 5, rotate: 19 }, frontLeg: { dx: 8, dy: 2, rotate: 20 }, backLeg: { dx: -5, dy: 1, rotate: -16 }, overlay: 'meleeStrike' },
        { x: 3, y: 0, torso: { rotate: 2 }, head: { rotate: 1 }, frontArm: { dx: 4, dy: 0, rotate: 10 }, backArm: { dx: 2, dy: 1, rotate: 5 }, frontLeg: { dx: 3, rotate: 6 }, backLeg: { dx: -2, rotate: -5 }, overlay: 'meleeRecover' },
        { x: 0, y: 0, torso: {}, head: {}, frontArm: {}, backArm: {}, frontLeg: {}, backLeg: {}, overlay: 'meleeRecover' }
      ],
      bow: [
        { x: -4, y: 0, torso: { rotate: -2 }, head: { rotate: -1 }, frontArm: { dx: -3, dy: -4, rotate: -18 }, backArm: { dx: 8, dy: -3, rotate: 24 }, frontLeg: { dx: -2, rotate: -5 }, backLeg: { dx: 2, rotate: 5 }, overlay: 'bowDraw' },
        { x: -5, y: -1, torso: { rotate: -3 }, head: { rotate: -1 }, frontArm: { dx: -5, dy: -6, rotate: -24 }, backArm: { dx: 11, dy: -4, rotate: 32 }, frontLeg: { dx: -3, rotate: -7 }, backLeg: { dx: 3, rotate: 7 }, overlay: 'bowDraw' },
        { x: 1, y: -1, torso: { rotate: 1 }, head: { dx: 1 }, frontArm: { dx: 8, dy: -5, rotate: 18 }, backArm: { dx: -3, dy: -4, rotate: -15 }, frontLeg: { dx: 2, rotate: 5 }, backLeg: { dx: -2, rotate: -5 }, overlay: 'bowRelease' },
        { x: 3, y: 0, torso: { rotate: 2 }, head: { dx: 1 }, frontArm: { dx: 11, dy: -4, rotate: 21 }, backArm: { dx: -4, dy: -2, rotate: -11 }, frontLeg: { dx: 3, rotate: 6 }, backLeg: { dx: -3, rotate: -6 }, overlay: 'bowArrow' },
        { x: 1, y: 0, torso: { rotate: 1 }, head: {}, frontArm: { dx: 4, dy: -1, rotate: 8 }, backArm: { dx: 0, dy: -1, rotate: 0 }, frontLeg: {}, backLeg: {}, overlay: 'bowRecover' },
        { x: 0, y: 0, torso: {}, head: {}, frontArm: {}, backArm: {}, frontLeg: {}, backLeg: {}, overlay: 'bowRecover' }
      ],
      magic: [
        { x: -2, y: 0, torso: { rotate: -1 }, head: { dy: -1 }, frontArm: { dx: -2, dy: -9, rotate: -27 }, backArm: { dx: 3, dy: -7, rotate: 21 }, frontLeg: { dx: -1 }, backLeg: { dx: 1 }, overlay: 'magicCharge' },
        { x: -2, y: -2, torso: { rotate: -1 }, head: { dy: -2 }, frontArm: { dx: -3, dy: -13, rotate: -34 }, backArm: { dx: 4, dy: -10, rotate: 27 }, frontLeg: { dx: -1 }, backLeg: { dx: 1 }, overlay: 'magicCharge' },
        { x: 3, y: -2, torso: { rotate: 2 }, head: { dx: 1, dy: -1 }, frontArm: { dx: 12, dy: -8, rotate: 30 }, backArm: { dx: 4, dy: -6, rotate: 12 }, frontLeg: { dx: 2, rotate: 5 }, backLeg: { dx: -2, rotate: -5 }, overlay: 'magicCast' },
        { x: 4, y: -1, torso: { rotate: 2 }, head: { dx: 1 }, frontArm: { dx: 14, dy: -6, rotate: 34 }, backArm: { dx: 4, dy: -4, rotate: 14 }, frontLeg: { dx: 2, rotate: 5 }, backLeg: { dx: -2, rotate: -5 }, overlay: 'magicBolt' },
        { x: 1, y: 0, torso: { rotate: 1 }, head: {}, frontArm: { dx: 5, dy: -2, rotate: 10 }, backArm: {}, frontLeg: {}, backLeg: {}, overlay: 'magicFade' },
        { x: 0, y: 0, torso: {}, head: {}, frontArm: {}, backArm: {}, frontLeg: {}, backLeg: {}, overlay: 'magicFade' }
      ]
    };
    const cycle = (cycles[style] || cycles.melee)[frame] || {};
    return makeConnectedPose({
      x: cycle.x || 0,
      y: cycle.y || 0,
      torso: cycle.torso || {},
      shadowScale: frame >= 2 && frame <= 3 ? 1.18 : 1,
      overlay: cycle.overlay || ''
    });
  }

  if (rowId === 'skill') {
    const cycles = {
      melee: [
        { y: 0, aura: 'skillCharge', frontArm: { dx: -4, dy: -10, rotate: -30 }, backArm: { dx: 3, dy: -8, rotate: 24 }, head: { dy: -1 }, overlay: 'meleeSkillCharge' },
        { y: -3, aura: 'skillCharge', frontArm: { dx: -5, dy: -13, rotate: -36 }, backArm: { dx: 4, dy: -10, rotate: 28 }, head: { dy: -2 }, overlay: 'meleeSkillCharge' },
        { y: -4, aura: 'skillCast', frontArm: { dx: 15, dy: -7, rotate: 34 }, backArm: { dx: 5, dy: -5, rotate: 16 }, torso: { rotate: 3 }, overlay: 'meleeSkillImpact' },
        { y: -4, aura: 'skillCast', frontArm: { dx: 18, dy: -5, rotate: 38 }, backArm: { dx: 5, dy: -4, rotate: 17 }, torso: { rotate: 3 }, overlay: 'meleeSkillImpact' },
        { y: -2, aura: 'skillRelease', frontArm: { dx: 8, dy: -1, rotate: 16 }, backArm: { dx: 2, dy: -1, rotate: 6 }, overlay: 'meleeSkillFade' },
        { y: 0, aura: 'skillRelease', frontArm: {}, backArm: {}, torso: {}, head: {}, overlay: 'meleeSkillFade' }
      ],
      bow: [
        { y: 0, aura: 'skillCharge', frontArm: { dx: -3, dy: -7, rotate: -24 }, backArm: { dx: 10, dy: -6, rotate: 33 }, head: { dy: -1 }, overlay: 'bowSkillDraw' },
        { y: -2, aura: 'skillCharge', frontArm: { dx: -5, dy: -9, rotate: -29 }, backArm: { dx: 13, dy: -7, rotate: 39 }, head: { dy: -2 }, overlay: 'bowSkillDraw' },
        { y: -3, aura: 'skillCast', frontArm: { dx: 10, dy: -7, rotate: 20 }, backArm: { dx: -3, dy: -6, rotate: -17 }, torso: { rotate: 2 }, overlay: 'bowSkillVolley' },
        { y: -3, aura: 'skillCast', frontArm: { dx: 12, dy: -6, rotate: 22 }, backArm: { dx: -4, dy: -5, rotate: -15 }, torso: { rotate: 2 }, overlay: 'bowSkillVolley' },
        { y: -1, aura: 'skillRelease', frontArm: { dx: 5, dy: -2, rotate: 9 }, backArm: {}, overlay: 'bowSkillFade' },
        { y: 0, aura: 'skillRelease', frontArm: {}, backArm: {}, torso: {}, head: {}, overlay: 'bowSkillFade' }
      ],
      magic: [
        { y: 0, aura: 'skillCharge', frontArm: { dx: -3, dy: -12, rotate: -34 }, backArm: { dx: 4, dy: -10, rotate: 28 }, head: { dy: -1 }, overlay: 'magicSkillCharge' },
        { y: -4, aura: 'skillCharge', frontArm: { dx: -4, dy: -16, rotate: -40 }, backArm: { dx: 5, dy: -13, rotate: 34 }, head: { dy: -2 }, overlay: 'magicSkillCharge' },
        { y: -5, aura: 'skillCast', frontArm: { dx: 16, dy: -10, rotate: 34 }, backArm: { dx: 5, dy: -7, rotate: 16 }, torso: { rotate: 3 }, overlay: 'magicSkillCast' },
        { y: -5, aura: 'skillCast', frontArm: { dx: 19, dy: -9, rotate: 38 }, backArm: { dx: 6, dy: -6, rotate: 18 }, torso: { rotate: 3 }, overlay: 'magicSkillCast' },
        { y: -2, aura: 'skillRelease', frontArm: { dx: 10, dy: -3, rotate: 16 }, backArm: { dx: 2, dy: -1, rotate: 6 }, overlay: 'magicSkillRelease' },
        { y: 0, aura: 'skillRelease', frontArm: {}, backArm: {}, torso: {}, head: {}, overlay: 'magicSkillRelease' }
      ]
    };
    const cycle = (cycles[style] || cycles.melee)[frame] || {};
    return makeConnectedPose({
      y: cycle.y || 0,
      aura: cycle.aura,
      torso: cycle.torso || {},
      overlay: cycle.overlay || '',
      shadowAlpha: 0.2
    });
  }

  if (rowId === 'party') {
    const up = frame < 3;
    return makeConnectedPose({
      y: -Math.round(bounce * 3),
      aura: 'party',
      shadowAlpha: 0.2,
      torso: { rotate: up ? -1 : 1 },
      overlay: up ? 'partyHandsUp' : 'partyHandsDown'
    });
  }

  if (rowId === 'hit') {
    const cycle = [
      { x: 4, y: -1, rotate: 4 },
      { x: -8, y: 1, rotate: -7 },
      { x: 5, y: 0, rotate: 5 },
      { x: -5, y: 1, rotate: -4 },
      { x: 2, y: 0, rotate: 2 },
      { x: 0, y: 0, rotate: 0 }
    ][frame] || {};
    return makeConnectedPose({
      x: cycle.x,
      y: cycle.y,
      torso: { rotate: cycle.rotate },
      overlay: frame < 4 ? 'hit' : '',
      shadowScale: 1.02
    });
  }

  if (rowId === 'defeat') {
    const cycle = [
      { x: 0, y: 6, rotate: -6 },
      { x: 1, y: 13, rotate: -22 },
      { x: 4, y: 23, rotate: -46 },
      { x: 7, y: 29, rotate: -68 },
      { x: 9, y: 32, rotate: -84 },
      { x: 9, y: 33, rotate: -90 }
    ][frame] || {};
    return makePose({
      x: cycle.x,
      y: cycle.y,
      sourceBody: true,
      shadow: frame < 2,
      shadowAlpha: 0.16,
      parts: {
        torso: { rotate: cycle.rotate }
      }
    });
  }

  return makePose();
}

function makeShadowSvg(pose) {
  if (!pose.shadow) return null;
  const width = Math.max(32, Math.round(58 * Math.max(0.65, pose.shadowScale || 1)));
  const alpha = pose.shadowAlpha == null ? 0.24 : pose.shadowAlpha;
  return svg(FRAME_SIZE, FRAME_SIZE, ellipse(80 + Number(pose.x || 0) * 0.35, 149, width / 2, 5, '#071015', `opacity="${alpha}"`));
}

function makeAuraSvg(kind, frame) {
  if (kind === 'skillCharge') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      ellipse(80, 82, 34 + frame * 3, 22, '#55b8ff', 'opacity="0.16"'),
      rect(112, 50 - frame * 2, 4, 16, '#d9f6ff', 'opacity="0.55"'),
      rect(48, 64 - frame, 3, 12, '#7bdff2', 'opacity="0.42"')
    ].join(''));
  }
  if (kind === 'skillCast' || kind === 'skillRelease') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      ellipse(96, 82, 48 + frame * 2, 24, '#55b8ff', 'opacity="0.2"'),
      rect(120 + frame, 62, 28, 4, '#9bdfff', 'opacity="0.68"'),
      rect(133 + frame, 51, 4, 26, '#ffffff', 'opacity="0.52"')
    ].join(''));
  }
  if (kind === 'party') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      ellipse(80, 91, 48, 31, '#f6c85f', 'opacity="0.18"'),
      ellipse(80, 91, 36 + frame * 2, 19, '#7bdff2', 'opacity="0.13"'),
      rect(119 - frame, 53, 4, 14, '#ffe16a', 'opacity="0.55"'),
      rect(45 + frame, 62, 3, 12, '#fff0a6', 'opacity="0.4"')
    ].join(''));
  }
  return null;
}

function makeMeleeOverlay(kind, frame) {
  const flash = kind.includes('Skill') ? '#ffe18a' : '#d9f6ff';
  const blade = kind.includes('Skill') ? '#fff2b8' : '#d6e5ec';
  if (kind.includes('Windup') || kind.includes('Charge')) {
    return [
      svgPath('M 56 96 L 92 54', blade, 'stroke-width="5" opacity="0.92"'),
      svgPath('M 52 101 L 68 92', '#5b3b25', 'stroke-width="4" opacity="0.9"'),
      rect(47, 101, 14, 5, BODY_COLORS.buckle, 'opacity="0.86"')
    ].join('');
  }
  if (kind.includes('Strike') || kind.includes('Impact')) {
    const reach = frame >= 3 ? 8 : 0;
    return [
      svgPath(`M ${72 + reach} 90 L ${139 + reach} 70`, blade, 'stroke-width="6" opacity="0.96"'),
      svgPath(`M ${77 + reach} 95 L ${115 + reach} 84`, '#5b3b25', 'stroke-width="4" opacity="0.92"'),
      svgPath(`M ${84 + reach} 56 C ${109 + reach} 54 ${132 + reach} 66 ${142 + reach} 82`, flash, 'stroke-width="5" opacity="0.46"'),
      svgPath(`M ${90 + reach} 64 C ${112 + reach} 63 ${130 + reach} 72 ${138 + reach} 86`, '#ffffff', 'stroke-width="2" opacity="0.6"')
    ].join('');
  }
  return [
    svgPath('M 82 93 L 112 87', blade, 'stroke-width="4" opacity="0.42"'),
    rect(108, 80, 12, 3, flash, 'opacity="0.2"')
  ].join('');
}

function makeBowOverlay(kind, frame) {
  const bow = '#875229';
  const bowLight = '#c0833f';
  const string = '#dbe8ec';
  const arrow = '#edf9ff';
  const draw = kind.includes('Draw');
  const release = kind.includes('Release') || kind.includes('Arrow') || kind.includes('Volley');
  const power = kind.includes('Skill');
  const arrowX = release ? 94 + frame * 3 : 91;
  return [
    svgPath('M 100 62 C 126 76 126 105 100 123', bow, 'stroke-width="5" opacity="0.96"'),
    svgPath('M 103 67 C 121 79 121 101 103 117', bowLight, 'stroke-width="2" opacity="0.85"'),
    svgPath(draw ? 'M 101 64 L 86 92 L 101 121' : 'M 101 64 L 104 92 L 101 121', string, 'stroke-width="2" opacity="0.82"'),
    svgPath(`M ${draw ? 75 : arrowX} 91 L ${draw ? 117 : arrowX + 34} 91`, arrow, 'stroke-width="3" opacity="0.95"'),
    polygon([[draw ? 115 : arrowX + 32, 87], [draw ? 124 : arrowX + 42, 91], [draw ? 115 : arrowX + 32, 95]], arrow, 'opacity="0.94"'),
    release ? svgPath(`M ${arrowX - 8} 86 L ${arrowX + 18} 86`, power ? '#ffe18a' : '#9bdfff', 'stroke-width="2" opacity="0.54"') : '',
    release && power ? svgPath(`M ${arrowX - 3} 96 L ${arrowX + 23} 101`, '#ffe18a', 'stroke-width="2" opacity="0.46"') : ''
  ].join('');
}

function makeMagicOverlay(kind, frame) {
  const staff = kind.includes('Skill') ? '#d4a85a' : '#a66b35';
  const glow = kind.includes('Skill') ? '#b58cff' : '#7bdff2';
  const white = '#f4fbff';
  const cast = kind.includes('Cast') || kind.includes('Bolt') || kind.includes('Release');
  const charge = kind.includes('Charge');
  const orbX = cast ? 104 + frame * 3 : 100;
  const orbY = cast ? 70 : 76 - frame;
  return [
    svgPath('M 62 113 L 113 55', staff, 'stroke-width="5" opacity="0.92"'),
    svgPath('M 67 111 L 116 58', '#f2d58b', 'stroke-width="2" opacity="0.55"'),
    ellipse(orbX, orbY, charge ? 8 + frame : 12, charge ? 8 + frame : 10, glow, cast ? 'opacity="0.72"' : 'opacity="0.46"'),
    ellipse(orbX, orbY, charge ? 3 + frame : 5, charge ? 3 + frame : 4, white, 'opacity="0.72"'),
    cast ? svgPath(`M ${orbX + 8} ${orbY} C ${orbX + 18} ${orbY - 8} ${orbX + 28} ${orbY - 4} ${orbX + 34} ${orbY - 13}`, glow, 'stroke-width="4" opacity="0.56"') : '',
    cast ? rect(130, 56 + frame, 4, 18, white, 'opacity="0.52"') : '',
    charge ? rect(48 + frame * 2, 63, 3, 12, glow, 'opacity="0.42"') : ''
  ].join('');
}

function makeOverlaySvg(kind, frame) {
  if (kind === 'idle') {
    const alpha = [0.02, 0.28, 0.1, 0.24, 0.03, 0.18][frame] || 0.08;
    return svg(FRAME_SIZE, FRAME_SIZE, [
      rect(59, 84, 8, 28, '#fff7dc', `opacity="${alpha}"`),
      rect(72, 82, 4, 32, '#f6d09d', `opacity="${Math.max(0.02, alpha * 0.72)}"`),
      rect(91, 89, 5, 24, '#fff0d3', `opacity="${Math.max(0.02, alpha * 0.62)}"`),
      rect(52, 95, 14, 5, '#fff7dc', `opacity="${Math.max(0.02, alpha * 0.55)}"`)
    ].join(''));
  }
  if (kind && kind.startsWith('run')) {
    const left = kind.endsWith('A') ? 50 : 88;
    const right = kind.endsWith('A') ? 84 : 54;
    const pass = kind.includes('Pass');
    return svg(FRAME_SIZE, FRAME_SIZE, [
      ellipse(left, 148, pass ? 8 : 13, 3, '#0a171c', 'opacity="0.34"'),
      ellipse(right, 150, pass ? 12 : 7, 3, '#0a171c', 'opacity="0.22"'),
      pass ? '' : rect(left - 9, 143, 16, 2, '#9cc2c8', 'opacity="0.18"')
    ].join(''));
  }
  if (kind === 'landingDust') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      ellipse(63, 149, 14, 3, '#9cc2c8', 'opacity="0.14"'),
      ellipse(96, 149, 12, 3, '#9cc2c8', 'opacity="0.12"')
    ].join(''));
  }
  if (kind.startsWith('melee')) {
    return svg(FRAME_SIZE, FRAME_SIZE, makeMeleeOverlay(kind, frame));
  }
  if (kind.startsWith('bow')) {
    return svg(FRAME_SIZE, FRAME_SIZE, makeBowOverlay(kind, frame));
  }
  if (kind.startsWith('magic')) {
    return svg(FRAME_SIZE, FRAME_SIZE, makeMagicOverlay(kind, frame));
  }
  if (kind === 'strike') {
    const offset = frame === 2 ? 0 : 8;
    return svg(FRAME_SIZE, FRAME_SIZE, [
      rect(103 + offset, 73, 40, 5, '#ffffff', 'opacity="0.5"'),
      rect(112 + offset, 66, 29, 3, '#9bdfff', 'opacity="0.64"')
    ].join(''));
  }
  if (kind === 'spark') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      rect(119 + frame, 64, 24, 4, '#9bdfff', 'opacity="0.68"'),
      rect(129 + frame, 55, 4, 23, '#ffffff', 'opacity="0.55"')
    ].join(''));
  }
  if (kind === 'climb') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      rect(43, 34, 3, 110, '#c8d5dc', 'opacity="0.22"'),
      rect(124, 34, 3, 110, '#c8d5dc', 'opacity="0.22"')
    ].join(''));
  }
  if (kind === 'hit') {
    return svg(FRAME_SIZE, FRAME_SIZE, [
      rect(108 + frame * 2, 54, 16, 3, '#ff7d5c', 'opacity="0.6"'),
      rect(116 + frame, 48, 3, 15, '#ffd1c2', 'opacity="0.45"')
    ].join(''));
  }
  return null;
}

function getPartTransform(pose, partName) {
  const part = pose.parts && pose.parts[partName] || {};
  return {
    dx: Number(pose.x || 0) + Number(part.dx || 0),
    dy: Number(pose.y || 0) + Number(part.dy || 0),
    rotate: Number(part.rotate || 0),
    scaleX: Number(part.scaleX || 1),
    scaleY: Number(part.scaleY || 1)
  };
}

function makeSvgTransform(transform, pivotX, pivotY) {
  const dx = Number(transform.dx || 0);
  const dy = Number(transform.dy || 0);
  const rotate = Number(transform.rotate || 0);
  const scaleY = Number(transform.scaleY || 1);
  const transforms = [];
  if (dx || dy) transforms.push(`translate(${Math.round(dx)} ${Math.round(dy)})`);
  if (rotate) transforms.push(`rotate(${Math.round(rotate)} ${Math.round(pivotX)} ${Math.round(pivotY)})`);
  if (scaleY !== 1) {
    transforms.push(`translate(0 ${Math.round(pivotY)}) scale(1 ${scaleY.toFixed(3)}) translate(0 ${Math.round(-pivotY)})`);
  }
  return transforms.length ? ` transform="${transforms.join(' ')}"` : '';
}

function svgGroup(transform, pivotX, pivotY, body, attrs) {
  return `<g${makeSvgTransform(transform, pivotX, pivotY)}${attr(attrs)}>${body}</g>`;
}

function drawLeg(transform, side) {
  const front = side === 'front';
  const x = front ? BODY_ANCHOR.hipFrontX : BODY_ANCHOR.hipBackX;
  const y = front ? BODY_ANCHOR.hipY : BODY_ANCHOR.hipY + 1;
  const pivotX = x + 7;
  const pivotY = y + 3;
  const shade = front ? BODY_COLORS.pants : BODY_COLORS.outlineSoft;
  const boot = front ? BODY_COLORS.boot : BODY_COLORS.bootDark;
  const highlightOpacity = front ? 'opacity="0.58"' : 'opacity="0.22"';
  const body = [
    polygon([[x - 3, y - 1], [x + 14, y - 1], [x + 15, y + 28], [x + 11, y + 32], [x, y + 31], [x - 4, y + 8]], BODY_COLORS.outline),
    polygon([[x, y + 1], [x + 11, y + 1], [x + 12, y + 25], [x + 9, y + 28], [x + 2, y + 28], [x, y + 8]], shade),
    polygon([[x + 3, y + 2], [x + 7, y + 2], [x + 8, y + 25], [x + 4, y + 25]], BODY_COLORS.pantsLight, highlightOpacity),
    polygon([[x - 5, y + 26], [x + 15, y + 26], [x + 19, y + 32], [x + 15, y + 39], [x - 3, y + 39], [x - 6, y + 35]], BODY_COLORS.outline),
    polygon([[x - 2, y + 28], [x + 13, y + 28], [x + 16, y + 32], [x + 13, y + 36], [x - 2, y + 36]], boot),
    rect(x + 2, y + 30, 8, 4, BODY_COLORS.bootLight, highlightOpacity)
  ].join('');
  return svgGroup(transform, pivotX, pivotY, body, front ? '' : 'opacity="0.86"');
}

function drawArm(transform, side) {
  const front = side === 'front';
  const x = front ? BODY_ANCHOR.shoulderFrontX : BODY_ANCHOR.shoulderBackX;
  const y = front ? BODY_ANCHOR.shoulderY + 1 : BODY_ANCHOR.shoulderY + 5;
  const pivotX = x + 6;
  const pivotY = y + 4;
  const sleeve = front ? BODY_COLORS.shirt : BODY_COLORS.shirtShade;
  const skin = front ? BODY_COLORS.skin : BODY_COLORS.skinShade;
  const body = [
    polygon([[x - 3, y - 1], [x + 12, y], [x + 12, y + 22], [x + 7, y + 29], [x - 2, y + 26], [x - 4, y + 7]], BODY_COLORS.outline),
    polygon([[x, y + 1], [x + 9, y + 2], [x + 9, y + 20], [x + 6, y + 24], [x, y + 23], [x - 1, y + 7]], sleeve),
    polygon([[x + 1, y + 4], [x + 5, y + 4], [x + 5, y + 18], [x + 2, y + 20]], '#ffffff', 'opacity="0.32"'),
    polygon([[x - 3, y + 23], [x + 9, y + 23], [x + 14, y + 30], [x + 10, y + 37], [x - 2, y + 36], [x - 7, y + 31]], BODY_COLORS.outline),
    polygon([[x, y + 25], [x + 8, y + 25], [x + 11, y + 30], [x + 8, y + 34], [x, y + 33], [x - 3, y + 30]], skin),
    rect(x + 2, y + 27, 4, 3, '#ffd7a9', front ? 'opacity="0.62"' : 'opacity="0.24"')
  ].join('');
  return svgGroup(transform, pivotX, pivotY, body, front ? '' : 'opacity="0.76"');
}

function drawTorso(transform) {
  const pivotX = 80;
  const pivotY = 100;
  const body = [
    polygon([[57, 76], [101, 76], [108, 112], [97, 128], [62, 128], [51, 112]], BODY_COLORS.outline),
    polygon([[61, 80], [97, 80], [102, 111], [94, 123], [65, 123], [56, 111]], BODY_COLORS.shirt),
    polygon([[62, 80], [75, 80], [70, 121], [57, 111]], '#ffffff', 'opacity="0.42"'),
    polygon([[91, 82], [98, 84], [101, 111], [94, 123]], BODY_COLORS.shirtShade),
    polygon([[76, 76], [88, 76], [90, 86], [74, 87]], BODY_COLORS.skinShade),
    polygon([[70, 79], [76, 79], [93, 122], [86, 123]], BODY_COLORS.shirtShadow, 'opacity="0.85"'),
    rect(56, 111, 45, 8, BODY_COLORS.belt),
    rect(78, 110, 8, 10, BODY_COLORS.buckle),
    polygon([[61, 119], [99, 119], [95, 134], [65, 134]], BODY_COLORS.pants),
    polygon([[67, 123], [94, 123], [92, 128], [68, 128]], BODY_COLORS.pantsLight, 'opacity="0.3"')
  ].join('');
  return svgGroup(transform, pivotX, pivotY, body);
}

function makeBodySvg(pose) {
  const parts = [
    drawLeg(getPartTransform(pose, 'backLeg'), 'back'),
    drawLeg(getPartTransform(pose, 'frontLeg'), 'front'),
    drawArm(getPartTransform(pose, 'backArm'), 'back'),
    drawTorso(getPartTransform(pose, 'torso')),
    drawArm(getPartTransform(pose, 'frontArm'), 'front')
  ];
  return svg(FRAME_SIZE, FRAME_SIZE, parts.join(''));
}

async function makeSourceBodyComposite(body, pose) {
  const transform = getPartTransform(pose, 'torso');
  const targetHeight = 128;
  const targetWidth = Math.max(1, Math.round(body.width * targetHeight / body.height));
  let pipeline = sharp(body.buffer).resize(targetWidth, targetHeight, {
    fit: 'fill',
    kernel: 'nearest',
    background: { r: 0, g: 0, b: 0, alpha: 0 }
  });
  if (transform.rotate) {
    pipeline = pipeline.rotate(transform.rotate, { background: { r: 0, g: 0, b: 0, alpha: 0 } });
  }
  const buffer = await pipeline.png({ compressionLevel: 9 }).toBuffer();
  const meta = await sharp(buffer).metadata();
  const baseLeft = 80 - targetWidth / 2;
  const baseTop = 150 - targetHeight;
  const centerX = baseLeft + targetWidth / 2 + transform.dx;
  const centerY = baseTop + targetHeight / 2 + transform.dy;
  return {
    input: buffer,
    left: Math.round(centerX - meta.width / 2),
    top: Math.round(centerY - meta.height / 2)
  };
}

async function makeSourcePartComposite(rig, partName, pose) {
  const part = rig.parts && rig.parts[partName];
  if (!part) return null;
  const transform = getPartTransform(pose, partName);
  const bodyTargetHeight = 128;
  const bodyTargetWidth = Math.max(1, Math.round(rig.width * bodyTargetHeight / rig.height));
  const scale = bodyTargetHeight / rig.height;
  const targetWidth = Math.max(1, Math.round(part.width * scale * transform.scaleX));
  const targetHeight = Math.max(1, Math.round(part.height * scale * transform.scaleY));
  let pipeline = sharp(part.buffer).resize(targetWidth, targetHeight, {
    fit: 'fill',
    kernel: 'nearest',
    background: { r: 0, g: 0, b: 0, alpha: 0 }
  });
  if (transform.rotate) {
    pipeline = pipeline.rotate(transform.rotate, { background: { r: 0, g: 0, b: 0, alpha: 0 } });
  }
  const buffer = await pipeline.png({ compressionLevel: 9 }).toBuffer();
  const meta = await sharp(buffer).metadata();
  const baseLeft = 80 - bodyTargetWidth / 2;
  const baseTop = 150 - bodyTargetHeight;
  const centerX = baseLeft + (part.sourceLeft + part.width / 2) * scale + transform.dx;
  const centerY = baseTop + (part.sourceTop + part.height / 2) * scale + transform.dy;
  return {
    input: buffer,
    left: Math.round(centerX - meta.width / 2),
    top: Math.round(centerY - meta.height / 2)
  };
}

async function makeSourcePartComposites(rig, pose) {
  const composites = [];
  for (const partName of SOURCE_PART_ORDER) {
    const composite = await makeSourcePartComposite(rig, partName, pose);
    if (composite) composites.push(composite);
  }
  return composites;
}

async function makeHeadComposite(head, pose) {
  const transform = getPartTransform(pose, 'head');
  const targetHeight = 66;
  const targetWidth = Math.max(1, Math.round(head.width * targetHeight / head.height * transform.scaleX));
  const scaledHeight = Math.max(1, Math.round(targetHeight * transform.scaleY));
  let pipeline = sharp(head.buffer).resize(targetWidth, scaledHeight, {
    fit: 'fill',
    kernel: 'nearest',
    background: { r: 0, g: 0, b: 0, alpha: 0 }
  });
  if (transform.rotate) {
    pipeline = pipeline.rotate(transform.rotate, { background: { r: 0, g: 0, b: 0, alpha: 0 } });
  }
  const buffer = await pipeline.png({ compressionLevel: 9 }).toBuffer();
  const meta = await sharp(buffer).metadata();
  const baseLeft = BODY_ANCHOR.headX - targetWidth / 2;
  const baseTop = BODY_ANCHOR.headTop;
  const centerX = baseLeft + targetWidth / 2 + transform.dx;
  const centerY = baseTop + scaledHeight / 2 + transform.dy;
  return {
    input: buffer,
    left: Math.round(centerX - meta.width / 2),
    top: Math.round(centerY - meta.height / 2)
  };
}

async function renderPlayerFrame(rig, rowId, frame, combatStyle) {
  const pose = getPose(rowId, frame, combatStyle);
  const composites = [];
  const shadow = makeShadowSvg(pose);
  const aura = makeAuraSvg(pose.aura, frame);
  const overlay = makeOverlaySvg(pose.overlay, frame);
  if (shadow) composites.push({ input: shadow, left: 0, top: 0 });
  if (aura) composites.push({ input: aura, left: 0, top: 0 });
  if (pose.sourceBody) {
    composites.push(await makeSourceBodyComposite(rig.body, pose));
  } else if (pose.sourceParts) {
    composites.push(...await makeSourcePartComposites(rig, pose));
  } else {
    composites.push({ input: makeBodySvg(pose), left: 0, top: 0 });
    composites.push(await makeHeadComposite(rig.head, pose));
  }
  if (overlay) composites.push({ input: overlay, left: 0, top: 0 });
  const rendered = await sharp({
    create: {
      width: FRAME_SIZE,
      height: FRAME_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const cleaned = removeSmallAlphaComponents(rendered.data, rendered.info.width, rendered.info.height, 8);
  return sharp(cleaned, { raw: { width: rendered.info.width, height: rendered.info.height, channels: 4 } })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function renderPlayerFrames(combatStyle) {
  const rig = await buildPlayerRig();
  const frames = [];
  for (const rowId of PLAYER_ROWS) {
    for (let frame = 0; frame < SHEET_COLS; frame += 1) {
      frames.push(await renderPlayerFrame(rig, rowId, frame, combatStyle));
    }
  }
  return frames;
}

async function writeSourceSheet(frames, destination) {
  const width = SHEET_COLS * (FRAME_SIZE + GUIDE_WIDTH) + GUIDE_WIDTH;
  const height = PLAYER_ROWS.length * (FRAME_SIZE + GUIDE_WIDTH) + GUIDE_WIDTH;
  const composites = frames.map((input, index) => ({
    input,
    left: GUIDE_WIDTH + (index % SHEET_COLS) * (FRAME_SIZE + GUIDE_WIDTH),
    top: GUIDE_WIDTH + Math.floor(index / SHEET_COLS) * (FRAME_SIZE + GUIDE_WIDTH)
  }));
  composites.push({ input: makeGridOverlay(width, height), left: 0, top: 0 });
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  await sharp({
    create: {
      width,
      height,
      channels: 4,
      background: { r: 255, g: 0, b: 255, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(destination);
  return toRepoPath(destination);
}

async function writeAllClassSourceSheets(genericFrames) {
  const generated = [];
  const genericSource = path.join(CLASS_SOURCE_DIR, 'generic-player-source.png');
  generated.push(await writeSourceSheet(genericFrames, genericSource));
  for (const fileId of Object.values(Data.CLASS_FILE_IDS || {})) {
    const destination = path.join(CLASS_SOURCE_DIR, `${fileId}-source.png`);
    generated.push(await writeSourceSheet(genericFrames, destination));
  }
  return generated;
}

function isChromaKeyPixel(raw, offset) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const directMagenta = r > 190 && b > 180 && g < 120;
  const blendedMagenta = r > 140 && b > 140 && g < 130 && Math.abs(r - b) < 96 && Math.max(r, b) - g > 70;
  const directGreen = g > 190 && r < 120 && b < 120;
  return directMagenta || blendedMagenta || directGreen;
}

function sanitizeSourceCell(raw, width, height) {
  const output = Buffer.from(raw);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const guideFringe = x < 3 || x >= width - 3 || y < 3 || y >= height - 3;
    if (guideFringe || isGuidePixelRgba(output, offset, GUIDE_LINE_HEX) || isChromaKeyPixel(output, offset)) {
      output[offset] = 0;
      output[offset + 1] = 0;
      output[offset + 2] = 0;
      output[offset + 3] = 0;
    }
  }
  return output;
}

function copyCellRaw(raw, sheetWidth, rectDef) {
  const output = Buffer.alloc(rectDef.w * rectDef.h * 4);
  for (let y = 0; y < rectDef.h; y += 1) {
    const sourceStart = ((rectDef.y + y) * sheetWidth + rectDef.x) * 4;
    const sourceEnd = sourceStart + rectDef.w * 4;
    raw.copy(output, y * rectDef.w * 4, sourceStart, sourceEnd);
  }
  return sanitizeSourceCell(output, rectDef.w, rectDef.h);
}

async function readSourceCells(sourcePath, label) {
  const decoded = await sharp(sourcePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const width = decoded.info.width;
  const height = decoded.info.height;
  const raw = Buffer.from(decoded.data);
  const grid = detectGuideGrid(raw, width, height, {
    columns: SHEET_COLS,
    rows: PLAYER_ROWS.length,
    guideColor: GUIDE_LINE_HEX,
    label
  });
  const cells = [];
  for (let row = 0; row < PLAYER_ROWS.length; row += 1) {
    for (let col = 0; col < SHEET_COLS; col += 1) {
      const rectDef = getGridCellRect(grid, row, col);
      const cellRaw = copyCellRaw(raw, width, rectDef);
      let cell = sharp(cellRaw, { raw: { width: rectDef.w, height: rectDef.h, channels: 4 } });
      if (rectDef.w !== FRAME_SIZE || rectDef.h !== FRAME_SIZE) {
        cell = cell.resize(FRAME_SIZE, FRAME_SIZE, {
          fit: 'contain',
          kernel: 'nearest',
          background: { r: 0, g: 0, b: 0, alpha: 0 }
        });
      }
      cells.push(await cell.png({ compressionLevel: 9 }).toBuffer());
    }
  }
  return cells;
}

async function writeRuntimeSheet(cells, destination) {
  const composites = cells.map((input, index) => ({
    input,
    left: (index % SHEET_COLS) * FRAME_SIZE,
    top: Math.floor(index / SHEET_COLS) * FRAME_SIZE
  }));
  fs.mkdirSync(path.dirname(destination), { recursive: true });
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
    .toFile(destination);
  return toRepoPath(destination);
}

async function writePortrait(cellBuffer, destination) {
  const portrait = await sharp(cellBuffer)
    .resize({ height: 260, fit: 'inside', kernel: 'nearest' })
    .png({ compressionLevel: 9 })
    .toBuffer();
  const meta = await sharp(portrait).metadata();
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  await sharp({
    create: {
      width: CHARACTER_SIZE,
      height: CHARACTER_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite([{
      input: portrait,
      left: Math.round((CHARACTER_SIZE - meta.width) / 2),
      top: Math.max(0, CHARACTER_SIZE - meta.height - 24)
    }])
    .png({ compressionLevel: 9 })
    .toFile(destination);
  return toRepoPath(destination);
}

async function processClassSource(fileId, sourcePath, characterPath, sheetPath) {
  const cells = await readSourceCells(sourcePath, `${fileId} player source sheet`);
  return [
    await writePortrait(cells[0], characterPath),
    await writeRuntimeSheet(cells, sheetPath)
  ];
}

async function makeRuntimeScalePreview(frameBuffer) {
  const runtimeSprite = await sharp(frameBuffer)
    .resize(66, 118, {
      fit: 'contain',
      kernel: 'nearest',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .png({ compressionLevel: 9 })
    .toBuffer();
  const meta = await sharp(runtimeSprite).metadata();
  return sharp({
    create: {
      width: 120,
      height: 154,
      channels: 4,
      background: { r: 12, g: 14, b: 17, alpha: 1 }
    }
  })
    .composite([{
      input: runtimeSprite,
      left: Math.round((120 - meta.width) / 2),
      top: Math.round((154 - meta.height) / 2)
    }])
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function readEquipmentFrame(fileId, row, frame) {
  const visual = EQUIPMENT_VISUALS_BY_FILE_ID[fileId];
  const state = PLAYER_ROWS[row] || 'idle';
  const parts = EquipmentAttachments.resolveEquipmentAtlasParts(visual, state, frame);
  const layers = (await Promise.all(parts.map(async (part) => {
    const frameDef = part.frame || {};
    const sheetPath = path.join(ROOT, String(frameDef.sheet || ''));
    if (!frameDef.sheet || !fs.existsSync(sheetPath)) return null;
    const frameWidth = Math.max(1, Number(frameDef.frameWidth || 128));
    const frameHeight = Math.max(1, Number(frameDef.frameHeight || frameWidth));
    const scaleX = Math.max(0.05, Number(part.scaleX || 1));
    const scaleY = Math.max(0.05, Number(part.scaleY || 1));
    const width = Math.max(1, Math.round(frameWidth * scaleX));
    const height = Math.max(1, Math.round(frameHeight * scaleY));
    const input = await sharp(sheetPath)
      .extract({
        left: Math.max(0, Number(frameDef.frameIndex || 0)) * frameWidth,
        top: Math.max(0, Number(frameDef.row || 0)) * frameHeight,
        width: frameWidth,
        height: frameHeight
      })
      .resize(width, height, { fit: 'fill', kernel: 'nearest' })
      .ensureAlpha()
      .png({ compressionLevel: 9 })
      .toBuffer();
    return {
      input,
      left: EQUIPMENT_PREVIEW_PADDING + Math.round(Number(part.socket.x || 0) - Number(part.pivot.x || 0) * scaleX),
      top: EQUIPMENT_PREVIEW_PADDING + Math.round(Number(part.socket.y || 0) - Number(part.pivot.y || 0) * scaleY),
      order: Number(part.order || 0)
    };
  }))).filter(Boolean).sort((a, b) => a.order - b.order);
  if (!layers.length) return null;
  const paddedSize = FRAME_SIZE + EQUIPMENT_PREVIEW_PADDING * 2;
  const padded = await sharp({
    create: {
      width: paddedSize,
      height: paddedSize,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(layers.map(({ input, left, top }) => ({ input, left, top })))
    .png({ compressionLevel: 9 })
    .toBuffer();
  return sharp(padded)
    .extract({
      left: EQUIPMENT_PREVIEW_PADDING,
      top: EQUIPMENT_PREVIEW_PADDING,
      width: FRAME_SIZE,
      height: FRAME_SIZE
    })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function makeEquippedCompositePreview(frameBuffer) {
  const equipmentFileIds = [
    'stitched-vest',
    'traveler-boots',
    'fieldguard-helm',
    'training-sword'
  ];
  const equipmentFrames = (await Promise.all(equipmentFileIds.map((fileId) => (
    readEquipmentFrame(fileId, 0, 0)
  )))).filter(Boolean);
  if (!equipmentFrames.length) return makeRuntimeScalePreview(frameBuffer);
  const composite = await sharp({
    create: {
      width: FRAME_SIZE,
      height: FRAME_SIZE,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
    })
    .composite([
      { input: frameBuffer, left: 0, top: 0 },
      ...equipmentFrames.map((input) => ({ input, left: 0, top: 0 }))
    ])
    .png({ compressionLevel: 9 })
    .toBuffer();
  return makeRuntimeScalePreview(composite);
}

async function writeReviewContactSheet(frames) {
  const referenceCrop = await sharp(REFERENCE_PATH)
    .extract(REFERENCE_FIRST_ADVENTURER_CROP)
    .resize({ height: 220, fit: 'inside' })
    .png()
    .toBuffer();
  const basePreview = await sharp(frames[0])
    .resize({ height: 220, fit: 'inside', kernel: 'nearest' })
    .png()
    .toBuffer();
  const runtimePreview = await makeRuntimeScalePreview(frames[0]);
  const equippedPreview = await makeEquippedCompositePreview(frames[0]);
  const frameThumbs = await Promise.all(frames.map((input) => sharp(input)
    .resize(86, 86, { fit: 'contain', kernel: 'nearest', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toBuffer()));
  const composites = [
    { input: referenceCrop, left: 30, top: 34 },
    { input: basePreview, left: 250, top: 34 },
    { input: runtimePreview, left: 470, top: 58 },
    { input: equippedPreview, left: 610, top: 58 },
    { input: makeLabelSvg('reference', 150, 28), left: 30, top: 260 },
    { input: makeLabelSvg('generated idle', 180, 28), left: 250, top: 260 },
    { input: makeLabelSvg('runtime scale', 180, 28), left: 470, top: 260 },
    { input: makeLabelSvg('weapon + armor', 150, 28), left: 610, top: 260 }
  ];
  const rowTop = 308;
  const rowHeight = 96;
  PLAYER_ROWS.forEach((rowId, rowIndex) => {
    const top = rowTop + rowIndex * rowHeight;
    composites.push({ input: makeLabelSvg(rowId, 110, 28), left: 30, top: top + 34 });
    for (let frame = 0; frame < SHEET_COLS; frame += 1) {
      composites.push({
        input: frameThumbs[rowIndex * SHEET_COLS + frame],
        left: 150 + frame * 92,
        top: top + 6
      });
    }
  });
  await sharp({
    create: {
      width: 760,
      height: 1320,
      channels: 4,
      background: { r: 31, g: 34, b: 38, alpha: 1 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(REVIEW_CONTACT_SHEET_PATH);
  return toRepoPath(REVIEW_CONTACT_SHEET_PATH);
}

async function generateAll() {
  ensureDirs();
  if (!fs.existsSync(APPROVED_ANIMATION_SOURCE_PATH)) {
    throw new Error(`Missing approved player animation source: ${toRepoPath(APPROVED_ANIMATION_SOURCE_PATH)}`);
  }
  const generated = [];
  const frames = await readSourceCells(APPROVED_ANIMATION_SOURCE_PATH, 'approved weaponless player animation source');
  generated.push(await writeReviewContactSheet(frames));
  generated.push(...await writeAllClassSourceSheets(frames));
  generated.push(...await processClassSource(
    'generic-player',
    path.join(CLASS_SOURCE_DIR, 'generic-player-source.png'),
    path.join(CHARACTER_DIR, 'generic-player.png'),
    path.join(PLAYER_SHEET_DIR, 'generic-player-sheet.png')
  ));
  for (const fileId of Object.values(Data.CLASS_FILE_IDS || {})) {
    generated.push(...await processClassSource(
      fileId,
      path.join(CLASS_SOURCE_DIR, `${fileId}-source.png`),
      path.join(CHARACTER_DIR, `${fileId}.png`),
      path.join(PLAYER_SHEET_DIR, `${fileId}-sheet.png`)
    ));
  }
  generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
  await validateAll();
}

async function validatePngDimensions(filePath, width, height) {
  const meta = await sharp(filePath).metadata();
  if (meta.width !== width || meta.height !== height) {
    throw new Error(`${toRepoPath(filePath)} is ${meta.width}x${meta.height}; expected ${width}x${height}`);
  }
}

async function validateSourceSheet(filePath, label) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing ${label}: ${toRepoPath(filePath)}`);
  const decoded = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  detectGuideGrid(Buffer.from(decoded.data), decoded.info.width, decoded.info.height, {
    columns: SHEET_COLS,
    rows: PLAYER_ROWS.length,
    guideColor: GUIDE_LINE_HEX,
    label
  });
}

async function getPngInfo(filePath) {
  return sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
}

function getFrameRaw(sheetRaw, sheetWidth, rowIndex, frameIndex) {
  const output = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  const x0 = frameIndex * FRAME_SIZE;
  const y0 = rowIndex * FRAME_SIZE;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceStart = ((y0 + y) * sheetWidth + x0) * 4;
    const sourceEnd = sourceStart + FRAME_SIZE * 4;
    sheetRaw.copy(output, y * FRAME_SIZE * 4, sourceStart, sourceEnd);
  }
  return output;
}

function hasVisibleChroma(raw, width, height) {
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (raw[offset + 3] <= 12) continue;
    if (isChromaKeyPixel(raw, offset) || isGuidePixelRgba(raw, offset, GUIDE_LINE_HEX)) return true;
  }
  return false;
}

function getVisibleColorBuckets(raw, width, height) {
  const buckets = new Set();
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (raw[offset + 3] <= 20) continue;
    const r = raw[offset] >> 4;
    const g = raw[offset + 1] >> 4;
    const b = raw[offset + 2] >> 4;
    buckets.add(`${r}:${g}:${b}`);
  }
  return buckets.size;
}

function countFramePixelDiff(raw, width, rowIndex, firstFrame, secondFrame, threshold) {
  let count = 0;
  const y0 = rowIndex * FRAME_SIZE;
  const xA = firstFrame * FRAME_SIZE;
  const xB = secondFrame * FRAME_SIZE;
  const diffThreshold = threshold == null ? 32 : Number(threshold);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const a = ((y0 + y) * width + xA + x) * 4;
      const b = ((y0 + y) * width + xB + x) * 4;
      const delta = Math.abs(raw[a] - raw[b]) +
        Math.abs(raw[a + 1] - raw[b + 1]) +
        Math.abs(raw[a + 2] - raw[b + 2]) +
        Math.abs(raw[a + 3] - raw[b + 3]);
      if (delta >= diffThreshold) count += 1;
    }
  }
  return count;
}

function countFramePixelDiffBetweenSheets(firstRaw, secondRaw, width, rowId, frameIndex, threshold) {
  let count = 0;
  const rowIndex = PLAYER_ROW_INDEX[rowId];
  const y0 = rowIndex * FRAME_SIZE;
  const x0 = frameIndex * FRAME_SIZE;
  const diffThreshold = threshold == null ? 32 : Number(threshold);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const offset = ((y0 + y) * width + x0 + x) * 4;
      const delta = Math.abs(firstRaw[offset] - secondRaw[offset]) +
        Math.abs(firstRaw[offset + 1] - secondRaw[offset + 1]) +
        Math.abs(firstRaw[offset + 2] - secondRaw[offset + 2]) +
        Math.abs(firstRaw[offset + 3] - secondRaw[offset + 3]);
      if (delta >= diffThreshold) count += 1;
    }
  }
  return count;
}

function getRegionAlphaArea(raw, x0, y0, width, height, threshold) {
  const alphaThreshold = threshold == null ? 20 : Number(threshold);
  let count = 0;
  for (let y = y0; y < y0 + height; y += 1) {
    for (let x = x0; x < x0 + width; x += 1) {
      if (x < 0 || x >= FRAME_SIZE || y < 0 || y >= FRAME_SIZE) continue;
      if (raw[(y * FRAME_SIZE + x) * 4 + 3] > alphaThreshold) count += 1;
    }
  }
  return count;
}

function getFrameEdgeAlphaArea(raw, edge, threshold) {
  const alphaThreshold = threshold == null ? 20 : Number(threshold);
  let count = 0;
  if (edge === 'top') {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      if (raw[x * 4 + 3] > alphaThreshold) count += 1;
    }
    return count;
  }
  const x = edge === 'right' ? FRAME_SIZE - 1 : 0;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    if (raw[(y * FRAME_SIZE + x) * 4 + 3] > alphaThreshold) count += 1;
  }
  return count;
}

function getFrameStats(sheetRaw, sheetWidth, rowId, frame) {
  const rowIndex = PLAYER_ROW_INDEX[rowId];
  const frameRaw = getFrameRaw(sheetRaw, sheetWidth, rowIndex, frame);
  const bounds = getAlphaBounds(frameRaw, FRAME_SIZE, FRAME_SIZE, 20);
  return {
    raw: frameRaw,
    bounds,
    upperArea: getRegionAlphaArea(frameRaw, 0, 0, FRAME_SIZE, 86, 20),
    midArea: getRegionAlphaArea(frameRaw, 0, 55, FRAME_SIZE, 65, 20),
    lowerArea: getRegionAlphaArea(frameRaw, 0, 94, FRAME_SIZE, 66, 20),
    rightActionArea: getRegionAlphaArea(frameRaw, 102, 45, 58, 70, 20),
    leftActionArea: getRegionAlphaArea(frameRaw, 0, 45, 58, 70, 20)
  };
}

function assertSemantic(condition, message) {
  if (!condition) throw new Error(message);
}

function maxBoundsDelta(stats, key) {
  const values = stats.map((item) => item.bounds && item.bounds[key]).filter((value) => Number.isFinite(value));
  return Math.max(...values) - Math.min(...values);
}

function validateActionSemantics(decoded, label) {
  const raw = decoded.data;
  const width = decoded.info.width;
  const idle = Array.from({ length: SHEET_COLS }, (_, frame) => getFrameStats(raw, width, 'idle', frame));
  assertSemantic(maxBoundsDelta(idle, 'width') <= 4 && maxBoundsDelta(idle, 'height') <= 4, `${label} idle should not resize between frames`);
  assertSemantic(maxBoundsDelta(idle, 'maxY') <= 4, `${label} idle should keep feet planted`);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.idle, 0, 1, 36) >= 120, `${label} idle needs subtle breathing motion`);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.idle, 0, 1, 36) <= 7200, `${label} idle motion is too large for an idle row`);

  const run0 = getFrameStats(raw, width, 'run', 0);
  const run1 = getFrameStats(raw, width, 'run', 1);
  const run2 = getFrameStats(raw, width, 'run', 2);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.run, 0, 1, 40) >= 900, `${label} run should not use identical frames`);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.run, 0, 2, 40) >= 1400, `${label} run should alternate stride poses`);
  assertSemantic(Math.abs((run0.bounds && run0.bounds.centerX || 0) - (run2.bounds && run2.bounds.centerX || 0)) >= 2 ||
    Math.abs(run0.lowerArea - run2.lowerArea) >= 120,
    `${label} run needs visible lower-body stride changes`);
  assertSemantic(Math.abs(run0.upperArea - run1.upperArea) >= 80 || countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.run, 1, 2, 40) >= 900,
    `${label} run needs arm/body counter motion`);

  const jump0 = getFrameStats(raw, width, 'jump', 0);
  const jump2 = getFrameStats(raw, width, 'jump', 2);
  const jump5 = getFrameStats(raw, width, 'jump', 5);
  assertSemantic(jump2.bounds.minY <= jump0.bounds.minY - 10, `${label} jump should clearly lift off the ground`);
  assertSemantic(jump5.bounds.maxY >= jump2.bounds.maxY + 8, `${label} jump should return toward landing`);

  const fall0 = getFrameStats(raw, width, 'fall', 0);
  const fall3 = getFrameStats(raw, width, 'fall', 3);
  const fall5 = getFrameStats(raw, width, 'fall', 5);
  assertSemantic(fall5.bounds.centerY >= fall3.bounds.centerY + 15, `${label} fall should descend after the aerial brace`);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.fall, 0, 1, 40) >= 700, `${label} fall needs bracing motion`);

  const climb0 = getFrameStats(raw, width, 'climb', 0);
  const climb1 = getFrameStats(raw, width, 'climb', 1);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.climb, 0, 1, 40) >= 900, `${label} climb should alternate reach/pull frames`);
  assertSemantic(Math.abs(climb0.upperArea - climb1.upperArea) >= 80 || Math.abs(climb0.lowerArea - climb1.lowerArea) >= 80,
    `${label} climb needs distinct arm and leg positions`);

  const basic0 = getFrameStats(raw, width, 'basic', 0);
  const basic2 = getFrameStats(raw, width, 'basic', 2);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.basic, 0, 2, 40) >= 1400, `${label} basic attack should include windup and strike`);
  assertSemantic(basic2.rightActionArea >= basic0.rightActionArea + 20 ||
    Math.abs((basic2.bounds.centerX || 0) - (basic0.bounds.centerX || 0)) >= 4,
    `${label} basic attack should move the striking side forward`);

  const skill0 = getFrameStats(raw, width, 'skill', 0);
  const skill3 = getFrameStats(raw, width, 'skill', 3);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.skill, 0, 3, 40) >= 1400, `${label} skill should charge then cast/release`);
  assertSemantic(skill3.rightActionArea >= skill0.rightActionArea + 40, `${label} skill cast should push visible action forward`);

  const party0 = getFrameStats(raw, width, 'party', 0);
  const party3 = getFrameStats(raw, width, 'party', 3);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.party, 0, 3, 40) >= 800, `${label} party buff should pulse or change pose`);
  assertSemantic(party0.midArea > idle[0].midArea || party3.midArea > idle[0].midArea,
    `${label} party buff should include visible channel/buff art`);

  const hit0 = getFrameStats(raw, width, 'hit', 0);
  const hit1 = getFrameStats(raw, width, 'hit', 1);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.hit, 0, 1, 40) >= 650, `${label} hit should recoil between frames`);
  assertSemantic(Math.abs(hit0.bounds.centerX - hit1.bounds.centerX) >= 1 ||
    Math.abs(hit0.bounds.height - hit1.bounds.height) >= 6,
    `${label} hit should visibly stagger or compress under impact`);

  const defeat0 = getFrameStats(raw, width, 'defeat', 0);
  const defeat5 = getFrameStats(raw, width, 'defeat', 5);
  assertSemantic(countFramePixelDiff(raw, width, PLAYER_ROW_INDEX.defeat, 0, 5, 40) >= 1200, `${label} defeat should not use a static standing frame`);
  assertSemantic(defeat5.bounds.width >= defeat0.bounds.width + 12, `${label} defeat final frame should slump wider than standing`);
  assertSemantic(defeat5.bounds.centerY >= defeat0.bounds.centerY + 7, `${label} defeat final frame should slump downward`);
}

async function validateRuntimeSheet(filePath, label) {
  await validatePngDimensions(filePath, SHEET_COLS * FRAME_SIZE, PLAYER_ROWS.length * FRAME_SIZE);
  const decoded = await getPngInfo(filePath);
  if (hasVisibleChroma(decoded.data, decoded.info.width, decoded.info.height)) {
    throw new Error(`${label} contains visible guide or chroma-key pixels`);
  }
  for (const rowId of PLAYER_ROWS) {
    const rowIndex = PLAYER_ROW_INDEX[rowId];
    const representativeFrame = rowId === 'run' ? 2 : rowId === 'defeat' ? 5 : 0;
    const frameRaw = getFrameRaw(decoded.data, decoded.info.width, rowIndex, representativeFrame);
    const stats = getAlphaBounds(frameRaw, FRAME_SIZE, FRAME_SIZE, 20);
    if (!stats) throw new Error(`${label} ${rowId} frame has no visible art`);
    for (let frame = 0; frame < SHEET_COLS; frame += 1) {
      const actionRaw = getFrameRaw(decoded.data, decoded.info.width, rowIndex, frame);
      const clippedEdge = getFrameEdgeAlphaArea(actionRaw, 'left', 20) ||
        getFrameEdgeAlphaArea(actionRaw, 'right', 20) ||
        getFrameEdgeAlphaArea(actionRaw, 'top', 20);
      if (clippedEdge) {
        throw new Error(`${label} ${rowId} frame ${frame} touches a cell edge and may be clipped`);
      }
    }
    if (rowId === 'defeat') {
      if (stats.count < 1600 || stats.width < 70 || stats.height < 30) {
        throw new Error(`${label} ${rowId} frame is too small (${stats.width}x${stats.height}, ${stats.count}px)`);
      }
    } else if (stats.count < 2300 || stats.width < 52 || stats.height < 100) {
      throw new Error(`${label} ${rowId} frame is too small (${stats.width}x${stats.height}, ${stats.count}px)`);
    }
  }
  validateActionSemantics(decoded, label);
  return decoded;
}

async function validatePlainAdventurerLikeness() {
  if (!fs.existsSync(BASE_SPRITE_PATH)) throw new Error(`Missing source-backed base sprite: ${toRepoPath(BASE_SPRITE_PATH)}`);
  const base = await getPngInfo(BASE_SPRITE_PATH);
  const baseStats = getAlphaBounds(base.data, base.info.width, base.info.height, 20);
  if (!baseStats || baseStats.height < 300 || baseStats.width < 145 || baseStats.count < 30000) {
    throw new Error(`Plain adventurer source crop is too small (${baseStats && baseStats.width}x${baseStats && baseStats.height})`);
  }
  if (getVisibleColorBuckets(base.data, base.info.width, base.info.height) < 120) {
    throw new Error('Plain adventurer source crop is too flat; expected painterly pixel shading from the reference');
  }
}

async function validateAll() {
  if (!fs.existsSync(REFERENCE_PATH)) {
    throw new Error(`Missing Project Starfall player style reference: ${toRepoPath(REFERENCE_PATH)}`);
  }
  await validatePlainAdventurerLikeness();
  await validateSourceSheet(APPROVED_ANIMATION_SOURCE_PATH, 'approved weaponless player animation source');
  await validateSourceSheet(path.join(CLASS_SOURCE_DIR, 'generic-player-source.png'), 'generic player source sheet');
  await validateRuntimeSheet(path.join(PLAYER_SHEET_DIR, 'generic-player-sheet.png'), 'generic player runtime sheet');
  await validatePngDimensions(path.join(CHARACTER_DIR, 'generic-player.png'), CHARACTER_SIZE, CHARACTER_SIZE);
  for (const [classId, fileId] of Object.entries(Data.CLASS_FILE_IDS || {})) {
    await validateSourceSheet(path.join(CLASS_SOURCE_DIR, `${fileId}-source.png`), `${classId} player source sheet`);
    await validatePngDimensions(path.join(CHARACTER_DIR, `${fileId}.png`), CHARACTER_SIZE, CHARACTER_SIZE);
    await validateRuntimeSheet(path.join(PLAYER_SHEET_DIR, `${fileId}-sheet.png`), `${classId} player runtime sheet`);
  }
  await validatePngDimensions(REVIEW_CONTACT_SHEET_PATH, 760, 1320);
  console.log(`Validated generated weaponless player art: ${Object.keys(Data.CLASS_FILE_IDS || {}).length + 1} registered player sheets with semantic action rows; equipped items remain separate runtime layers`);
}

async function main(argv) {
  const args = Array.isArray(argv) ? argv : process.argv.slice(2);
  if (args.includes('--validate')) {
    await validateAll();
    return;
  }
  await generateAll();
}

module.exports = {
  main,
  validateAll,
  generateAll
};

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exit(1);
  });
}
