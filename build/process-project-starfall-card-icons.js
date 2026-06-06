#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  GUIDE_LINE_HEX,
  detectGuideGrid,
  getGridCellRect,
  isGuidePixelRgba
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const STARFALL_ROOT = path.join(ROOT, 'img/project-starfall');
const SOURCE_FILE = path.join(STARFALL_ROOT, 'cards/source/ai-card-icons.png');
const OUTPUT_DIR = path.join(STARFALL_ROOT, 'cards/icons');
const ICON_SIZE = 64;
const COLUMNS = 7;
const ROWS = 3;
const MIN_VISIBLE_PIXELS = 48;

const CARD_ICON_IDS = Object.freeze((Data.CARD_DEFINITIONS || []).map((card) => card.id));

function rel(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function isKeyPixel(raw, offset) {
  const alpha = raw[offset + 3] == null ? 255 : raw[offset + 3];
  if (alpha < 8) return true;
  if (isGuidePixelRgba(raw, offset, GUIDE_LINE_HEX)) return true;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const magentaDistance = Math.abs(r - 255) + Math.abs(g - 0) + Math.abs(b - 255);
  if (magentaDistance <= 110 && r > 172 && b > 172 && g < 124) return true;
  return r > 188 && b > 164 && g < 118 && r - g > 92 && b - g > 76;
}

function cleanCell(raw, width, height) {
  const cleaned = Buffer.from(raw);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let visiblePixels = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (isKeyPixel(cleaned, offset)) {
        cleaned[offset] = 0;
        cleaned[offset + 1] = 0;
        cleaned[offset + 2] = 0;
        cleaned[offset + 3] = 0;
        continue;
      }
      if (cleaned[offset + 3] <= 12) continue;
      visiblePixels += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (visiblePixels < MIN_VISIBLE_PIXELS || maxX < minX || maxY < minY) {
    throw new Error(`Card icon cell has too little visible art (${visiblePixels} pixels)`);
  }

  const pad = 4;
  const x = Math.max(0, minX - pad);
  const y = Math.max(0, minY - pad);
  const right = Math.min(width - 1, maxX + pad);
  const bottom = Math.min(height - 1, maxY + pad);
  return {
    buffer: cleaned,
    width,
    height,
    bounds: {
      left: x,
      top: y,
      width: Math.max(1, right - x + 1),
      height: Math.max(1, bottom - y + 1)
    },
    visiblePixels
  };
}

async function processCardIcon(sourceImage, grid, cardId, index) {
  const row = Math.floor(index / COLUMNS);
  const col = index % COLUMNS;
  const rect = getGridCellRect(grid, row, col, 2);
  const source = await sourceImage
    .clone()
    .extract({ left: rect.x, top: rect.y, width: rect.w, height: rect.h })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const cleaned = cleanCell(source.data, source.info.width, source.info.height);
  const outputFile = path.join(OUTPUT_DIR, `${cardId}.png`);
  await sharp(cleaned.buffer, {
    raw: {
      width: cleaned.width,
      height: cleaned.height,
      channels: 4
    }
  })
    .extract(cleaned.bounds)
    .resize(ICON_SIZE, ICON_SIZE, {
      fit: 'contain',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .png()
    .toFile(outputFile);
  return outputFile;
}

async function detectSourceGrid(sourceFile) {
  const source = sharp(sourceFile).ensureAlpha();
  const raw = await source.clone().raw().toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(raw.data, raw.info.width, raw.info.height, {
    columns: COLUMNS,
    rows: ROWS,
    label: 'Project Starfall card icon sheet',
    minCoverage: 0.32
  });
  return { source, grid };
}

async function processCardIcons() {
  if (!fs.existsSync(SOURCE_FILE)) {
    throw new Error(`Missing card icon source sheet: ${rel(SOURCE_FILE)}`);
  }
  if (CARD_ICON_IDS.length !== COLUMNS * ROWS) {
    throw new Error(`Card icon sheet expects ${COLUMNS * ROWS} cards, found ${CARD_ICON_IDS.length}`);
  }
  ensureDir(OUTPUT_DIR);
  const { source, grid } = await detectSourceGrid(SOURCE_FILE);
  const outputs = [];
  for (let index = 0; index < CARD_ICON_IDS.length; index += 1) {
    outputs.push(await processCardIcon(source, grid, CARD_ICON_IDS[index], index));
  }
  return outputs;
}

async function validateIconFile(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing card icon: ${rel(filePath)}`);
  const image = sharp(filePath).ensureAlpha();
  const metadata = await image.metadata();
  if (metadata.width !== ICON_SIZE || metadata.height !== ICON_SIZE) {
    throw new Error(`${rel(filePath)} must be ${ICON_SIZE}x${ICON_SIZE}; received ${metadata.width}x${metadata.height}`);
  }
  const raw = await image.raw().toBuffer({ resolveWithObject: true });
  let visiblePixels = 0;
  for (let pixel = 0; pixel < raw.info.width * raw.info.height; pixel += 1) {
    if (raw.data[pixel * 4 + 3] > 12) visiblePixels += 1;
  }
  if (visiblePixels < MIN_VISIBLE_PIXELS) {
    throw new Error(`${rel(filePath)} has too little visible art (${visiblePixels} pixels)`);
  }
  return true;
}

async function validateCardIcons() {
  if (fs.existsSync(SOURCE_FILE)) await detectSourceGrid(SOURCE_FILE);
  await Promise.all(CARD_ICON_IDS.map((cardId) => validateIconFile(path.join(OUTPUT_DIR, `${cardId}.png`))));
  return true;
}

async function main() {
  const args = new Set(process.argv.slice(2));
  if (args.has('--help')) {
    process.stdout.write([
      'Usage: node build/process-project-starfall-card-icons.js [--validate]',
      '',
      `Processes ${rel(SOURCE_FILE)} into transparent card icons under ${rel(OUTPUT_DIR)}.`
    ].join('\n') + '\n');
    return;
  }
  if (args.has('--validate')) {
    await validateCardIcons();
    process.stdout.write(`Validated ${CARD_ICON_IDS.length} Project Starfall card icon(s).\n`);
    return;
  }
  const outputs = await processCardIcons();
  process.stdout.write(`Processed ${outputs.length} Project Starfall card icon(s) into ${rel(OUTPUT_DIR)}.\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${error && error.stack || error}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  CARD_ICON_IDS,
  processCardIcons,
  validateCardIcons
};
