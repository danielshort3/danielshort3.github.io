#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const {
  GUIDE_LINE_HEX,
  detectGuideGrid,
  getGridCellRect
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/skills/source');
const BASE_DIR = path.join(ROOT, 'img/project-starfall/skills/base');
const ADVANCED_DIR = path.join(ROOT, 'img/project-starfall/skills/advanced');
const CELL_SIZE = 256;

const SHEETS = Object.freeze([
  Object.freeze({
    source: 'fighter-base-sheet.png',
    columns: 3,
    rows: 2,
    outputDir: BASE_DIR,
    skills: Object.freeze([
      'fighter-heavy-strike',
      'fighter-dash-slash',
      'fighter-guard',
      'fighter-ground-slam',
      'fighter-power-break',
      'fighter-momentum-burst'
    ])
  }),
  Object.freeze({
    source: 'mage-base-sheet.png',
    columns: 3,
    rows: 2,
    outputDir: BASE_DIR,
    skills: Object.freeze([
      'mage-magic-bolt',
      'mage-blink',
      'mage-arcane-burst',
      'mage-mana-shield',
      'mage-spell-mark',
      'mage-energy-release'
    ])
  }),
  Object.freeze({
    source: 'archer-base-sheet.png',
    columns: 3,
    rows: 2,
    outputDir: BASE_DIR,
    skills: Object.freeze([
      'archer-quick-shot',
      'archer-roll-shot',
      'archer-marked-shot',
      'archer-piercing-arrow',
      'archer-eagle-stance',
      'archer-focused-volley'
    ])
  }),
  Object.freeze({
    source: 'guardian-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'guardian'),
    skills: Object.freeze([
      'guardian-shield-bash',
      'guardian-shield-dash',
      'guardian-impact-guard',
      'guardian-oath-barrier',
      'guardian-retaliation-wave',
      'guardian-hold-the-line',
      'guardian-verdict',
      'guardian-shield-wall'
    ])
  }),
  Object.freeze({
    source: 'berserker-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'berserker'),
    skills: Object.freeze([
      'berserker-blood-cleave',
      'berserker-rage-surge',
      'berserker-reckless-leap',
      'berserker-crimson-recovery',
      'berserker-pain-to-power',
      'berserker-last-stand',
      'berserker-war-cry'
    ])
  }),
  Object.freeze({
    source: 'fire-mage-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'fire-mage'),
    skills: Object.freeze([
      'fire-mage-fireball',
      'fire-mage-flame-trail',
      'fire-mage-burning-mark',
      'fire-mage-heat-vent',
      'fire-mage-wildfire',
      'fire-mage-inferno-burst',
      'fire-mage-ignition-aura'
    ])
  }),
  Object.freeze({
    source: 'rune-mage-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'rune-mage'),
    skills: Object.freeze([
      'rune-mage-rune-mark',
      'rune-mage-rune-blink',
      'rune-mage-ground-glyph',
      'rune-mage-arcane-link',
      'rune-mage-rune-detonation',
      'rune-mage-mana-seal',
      'rune-mage-grand-inscription',
      'rune-mage-rune-circle'
    ])
  }),
  Object.freeze({
    source: 'sniper-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'sniper'),
    skills: Object.freeze([
      'sniper-aimed-shot',
      'sniper-combat-roll',
      'sniper-weak-point-mark',
      'sniper-steady-breath',
      'sniper-pierce-armor',
      'sniper-execution-shot',
      'sniper-one-perfect-shot',
      'sniper-eagle-eye'
    ])
  }),
  Object.freeze({
    source: 'trapper-advanced-sheet.png',
    columns: 4,
    rows: 2,
    outputDir: path.join(ADVANCED_DIR, 'trapper'),
    skills: Object.freeze([
      'trapper-snare-trap',
      'trapper-grapple-dash',
      'trapper-spike-trap',
      'trapper-lure-shot',
      'trapper-tripwire',
      'trapper-detonate',
      'trapper-kill-zone',
      'trapper-tactical-field'
    ])
  }),
  Object.freeze({
    source: 'duelist-storm-beast-advanced-sheet.png',
    columns: 3,
    rows: 3,
    outputDir: ADVANCED_DIR,
    skills: Object.freeze([
      'duelist/duelist-quick-cut',
      'duelist/duelist-flash-step',
      'duelist/duelist-rallying-flourish',
      'storm-mage/storm-mage-chain-bolt',
      'storm-mage/storm-mage-static-shift',
      'storm-mage/storm-mage-stormfront',
      'beast-archer/beast-archer-companion-strike',
      'beast-archer/beast-archer-pounce-roll',
      'beast-archer/beast-archer-pack-call'
    ])
  })
]);

const MASTERY_ICONS = Object.freeze([
  Object.freeze({ file: 'fighter-damage-mastery', outputDir: BASE_DIR, color: '#f25f4c', accent: '#ffd166' }),
  Object.freeze({ file: 'mage-damage-mastery', outputDir: BASE_DIR, color: '#4f8cff', accent: '#e7fbff' }),
  Object.freeze({ file: 'archer-damage-mastery', outputDir: BASE_DIR, color: '#3aa76d', accent: '#fff0a6' }),
  Object.freeze({ file: 'guardian-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'guardian'), color: '#68a9ff', accent: '#d5ecff' }),
  Object.freeze({ file: 'berserker-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'berserker'), color: '#ef3d55', accent: '#ffbe55' }),
  Object.freeze({ file: 'duelist-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'duelist'), color: '#f0c36a', accent: '#ffffff' }),
  Object.freeze({ file: 'fire-mage-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'fire-mage'), color: '#ff8a3d', accent: '#ffe16a' }),
  Object.freeze({ file: 'rune-mage-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'rune-mage'), color: '#28c7b7', accent: '#b8fff2' }),
  Object.freeze({ file: 'storm-mage-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'storm-mage'), color: '#7bdff2', accent: '#ffffff' }),
  Object.freeze({ file: 'sniper-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'sniper'), color: '#c9b35c', accent: '#ffffff' }),
  Object.freeze({ file: 'trapper-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'trapper'), color: '#b07a47', accent: '#dbffe6' }),
  Object.freeze({ file: 'beast-archer-damage-mastery', outputDir: path.join(ADVANCED_DIR, 'beast-archer'), color: '#78b26a', accent: '#fff0a6' })
]);

function getOnlyMode(args) {
  const equalsArg = args.find((arg) => String(arg || '').startsWith('--only='));
  if (equalsArg) return equalsArg.split('=').slice(1).join('=').trim();
  const onlyIndex = args.indexOf('--only');
  if (onlyIndex >= 0) return String(args[onlyIndex + 1] || '').trim();
  return '';
}

function colorDistance(raw, offset, color) {
  return Math.abs(raw[offset] - color.r) +
    Math.abs(raw[offset + 1] - color.g) +
    Math.abs(raw[offset + 2] - color.b);
}

function isChromaKey(raw, offset) {
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  return r > 150 && b > 150 && g < 130 && Math.abs(r - b) < 100;
}

function makeTransparent(raw, width, height) {
  const corners = [
    0,
    (width - 1) * 4,
    ((height - 1) * width) * 4,
    ((height * width) - 1) * 4
  ].map((offset) => ({
    r: raw[offset],
    g: raw[offset + 1],
    b: raw[offset + 2]
  }));
  const visited = new Uint8Array(width * height);
  const queue = [];

  function isLikelyBackground(offset) {
    if (raw[offset + 3] < 10 || isChromaKey(raw, offset)) return true;
    return corners.some((color) => colorDistance(raw, offset, color) <= 120);
  }

  function enqueue(x, y) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const pixel = y * width + x;
    if (visited[pixel]) return;
    const offset = pixel * 4;
    if (!isLikelyBackground(offset)) return;
    visited[pixel] = 1;
    queue.push(pixel);
  }

  for (let x = 0; x < width; x += 1) {
    enqueue(x, 0);
    enqueue(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    enqueue(0, y);
    enqueue(width - 1, y);
  }

  for (let index = 0; index < queue.length; index += 1) {
    const pixel = queue[index];
    const offset = pixel * 4;
    raw[offset + 3] = 0;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    enqueue(x + 1, y);
    enqueue(x - 1, y);
    enqueue(x, y + 1);
    enqueue(x, y - 1);
  }

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (isChromaKey(raw, offset)) raw[offset + 3] = 0;
  }
}

function getAlphaBounds(raw, width, height) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= 20) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) return null;
  return { minX, minY, maxX, maxY };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function recenterVisiblePixels(raw, width, height) {
  const bounds = getAlphaBounds(raw, width, height);
  if (!bounds) return;
  const centerX = (bounds.minX + bounds.maxX + 1) / 2;
  const centerY = (bounds.minY + bounds.maxY + 1) / 2;
  const shiftX = clamp(Math.round(width / 2 - centerX), -bounds.minX, width - 1 - bounds.maxX);
  const shiftY = clamp(Math.round(height / 2 - centerY), -bounds.minY, height - 1 - bounds.maxY);
  if (!shiftX && !shiftY) return;
  const source = Buffer.from(raw);
  raw.fill(0);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceOffset = (y * width + x) * 4;
      if (!source[sourceOffset + 3]) continue;
      const targetX = x + shiftX;
      const targetY = y + shiftY;
      if (targetX < 0 || targetY < 0 || targetX >= width || targetY >= height) continue;
      const targetOffset = (targetY * width + targetX) * 4;
      raw[targetOffset] = source[sourceOffset];
      raw[targetOffset + 1] = source[sourceOffset + 1];
      raw[targetOffset + 2] = source[sourceOffset + 2];
      raw[targetOffset + 3] = source[sourceOffset + 3];
    }
  }
}

async function processCell(sheet, index) {
  const col = index % sheet.columns;
  const row = Math.floor(index / sheet.columns);
  const rect = getGridCellRect(sheet.grid, row, col);
  const { data, info } = await sharp(sheet.sourcePath)
    .extract({ left: rect.x, top: rect.y, width: rect.w, height: rect.h })
    .resize(CELL_SIZE, CELL_SIZE, {
      fit: 'contain',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  makeTransparent(data, info.width, info.height);
  recenterVisiblePixels(data, info.width, info.height);
  return sharp(data, { raw: info }).png({ compressionLevel: 9 }).toBuffer();
}

async function processSheet(sheet) {
  const sourcePath = path.join(SOURCE_DIR, sheet.source);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing skill icon sheet source: ${path.relative(ROOT, sourcePath)}`);
  }
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(data, info.width, info.height, {
    columns: sheet.columns,
    rows: sheet.rows,
    label: `skill icon sheet ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}`,
    guideColor: GUIDE_LINE_HEX
  });
  const sheetRuntime = Object.assign({}, sheet, { sourcePath, grid });
  const generated = [];
  fs.mkdirSync(sheet.outputDir, { recursive: true });
  for (let index = 0; index < sheet.skills.length; index += 1) {
    const skillPath = sheet.skills[index];
    const destination = path.join(sheet.outputDir, `${skillPath}.png`);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    const iconBuffer = await processCell(sheetRuntime, index);
    await sharp(iconBuffer).toFile(destination);
    generated.push(path.relative(ROOT, destination).replace(/\\/g, '/'));
  }
  return generated;
}

function makeMasterySvg(icon) {
  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${CELL_SIZE}" height="${CELL_SIZE}" viewBox="0 0 ${CELL_SIZE} ${CELL_SIZE}">
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="44%">
      <stop offset="0%" stop-color="${icon.accent}" stop-opacity="0.95"/>
      <stop offset="58%" stop-color="${icon.color}" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="${icon.color}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="core" x1="48" y1="214" x2="208" y2="42" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="${icon.color}"/>
      <stop offset="100%" stop-color="${icon.accent}"/>
    </linearGradient>
  </defs>
  <circle cx="128" cy="128" r="88" fill="url(#glow)"/>
  <path d="M128 34l58 94-58 94-58-94z" fill="url(#core)" stroke="#ffffff" stroke-opacity="0.72" stroke-width="8" stroke-linejoin="round"/>
  <path d="M128 66l36 62-36 62-36-62z" fill="none" stroke="#102033" stroke-opacity="0.72" stroke-width="10" stroke-linejoin="round"/>
  <path d="M89 142l39-74 39 74h-23l-16-31-16 31z" fill="#ffffff" fill-opacity="0.94"/>
  <path d="M83 164h90" stroke="${icon.accent}" stroke-width="12" stroke-linecap="round"/>
  <path d="M98 186h60" stroke="#ffffff" stroke-opacity="0.88" stroke-width="9" stroke-linecap="round"/>
</svg>
`;
}

async function processMasteryIcons() {
  const generated = [];
  for (const icon of MASTERY_ICONS) {
    const destination = path.join(icon.outputDir, `${icon.file}.png`);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    await sharp(Buffer.from(makeMasterySvg(icon))).png({ compressionLevel: 9 }).toFile(destination);
    generated.push(path.relative(ROOT, destination).replace(/\\/g, '/'));
  }
  return generated;
}

async function main() {
  const only = getOnlyMode(process.argv.slice(2));
  if (only && only !== 'mastery') {
    throw new Error(`Unsupported --only mode: ${only}`);
  }
  const generated = [];
  if (!only) {
    for (const sheet of SHEETS) {
      generated.push(...await processSheet(sheet));
    }
  }
  if (!only || only === 'mastery') {
    generated.push(...await processMasteryIcons());
  }
  generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
