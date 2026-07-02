#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'img/project-starfall/environment/ramps');
const SOURCE_PATH = path.join(ROOT, 'img/project-starfall/environment/source/ramp-imagegen-source.png');
const CELL = 128;
const COLUMNS = 4;
const WIDTH = CELL * COLUMNS;
const HEIGHT = CELL;
const RAMP_ATLAS_SCHEMA = 'ramps-v1';
const SOURCE_ROWS = 10;
const SOURCE_COLUMNS = 2;
const RAMP_SOURCE_THEME_ROWS = [
  { row: 9, tokens: ['bandit', 'duelist', 'sniper'] },
  { row: 8, tokens: ['eclipse', 'rift'] },
  { row: 7, tokens: ['astral', 'rune'] },
  { row: 6, tokens: ['storm'] },
  { row: 5, tokens: ['frost', 'glacier', 'rime'] },
  { row: 4, tokens: ['cinder', 'ember', 'ashglass', 'fire', 'berserker'] },
  { row: 3, tokens: ['rust', 'gear', 'quarry', 'titan', 'deepcore', 'guardian'] },
  { row: 2, tokens: ['thorn', 'bramble', 'trapper'] },
  { row: 1, tokens: ['greenroot', 'meadow', 'beast'] }
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setPixel(raw, width, x, y, rgba) {
  const offset = (y * width + x) * 4;
  raw[offset] = rgba[0];
  raw[offset + 1] = rgba[1];
  raw[offset + 2] = rgba[2];
  raw[offset + 3] = rgba[3];
}

function isSourceKeyPixel(r, g, b) {
  return g >= 178 && r <= 126 && b <= 132 && g - Math.max(r, b) >= 82;
}

function getImagegenSourceRow(themeId) {
  const id = String(themeId || '').toLowerCase();
  const match = RAMP_SOURCE_THEME_ROWS.find((entry) => entry.tokens.some((token) => id.includes(token)));
  return match ? match.row : 0;
}

function getImagegenSourceColumn(direction) {
  return direction === 'down-right' ? 1 : 0;
}

function findImagegenPanelBounds(source, row, column) {
  const regionLeft = Math.floor(column * source.width / SOURCE_COLUMNS);
  const regionRight = Math.ceil((column + 1) * source.width / SOURCE_COLUMNS);
  const regionTop = Math.floor(row * source.height / SOURCE_ROWS);
  const regionBottom = Math.ceil((row + 1) * source.height / SOURCE_ROWS);
  let minX = regionRight;
  let minY = regionBottom;
  let maxX = regionLeft;
  let maxY = regionTop;
  let count = 0;
  for (let y = regionTop; y < regionBottom; y += 1) {
    for (let x = regionLeft; x < regionRight; x += 1) {
      const offset = (y * source.width + x) * 4;
      if (source.raw[offset + 3] < 16) continue;
      if (isSourceKeyPixel(source.raw[offset], source.raw[offset + 1], source.raw[offset + 2])) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      count += 1;
    }
  }
  if (!count) {
    throw new Error(`No visible imagegen ramp panel found for source row ${row}, column ${column}`);
  }
  const padX = 12;
  const padY = 8;
  const left = clamp(minX - padX, regionLeft, regionRight - 1);
  const top = clamp(minY - padY, regionTop, regionBottom - 1);
  const right = clamp(maxX + padX, left + 1, regionRight);
  const bottom = clamp(maxY + padY, top + 1, regionBottom);
  return {
    left,
    top,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top)
  };
}

function cleanImagegenPixel(rgba, x, y, variant) {
  const r = rgba[0];
  const g = rgba[1];
  const b = rgba[2];
  if (rgba[3] < 16 || isSourceKeyPixel(r, g, b)) return [0, 0, 0, 0];
  const cleaned = [r, g, b, rgba[3]];
  const greenExcess = cleaned[1] - Math.max(cleaned[0], cleaned[2]);
  if (greenExcess > 34 && cleaned[1] > 132) {
    cleaned[1] = clamp(Math.round(cleaned[1] - greenExcess * 0.78), 0, 255);
  }
  if (variant) {
    const relief = (x * 13 + y * 7) % 29 < 9 ? 1.055 : 0.965;
    cleaned[0] = clamp(Math.round(cleaned[0] * relief), 0, 255);
    cleaned[1] = clamp(Math.round(cleaned[1] * relief), 0, 255);
    cleaned[2] = clamp(Math.round(cleaned[2] * relief), 0, 255);
  }
  if (x < 3 || x > CELL - 4 || y < 3 || y > CELL - 4) cleaned[3] = 0;
  return cleaned;
}

async function readImagegenSource() {
  if (!fs.existsSync(SOURCE_PATH)) {
    throw new Error(`Missing Project Starfall imagegen ramp source: ${SOURCE_PATH}`);
  }
  const image = sharp(SOURCE_PATH).ensureAlpha();
  const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
  if (info.width < 512 || info.height < 512) {
    throw new Error(`${SOURCE_PATH} should be a large 10-row imagegen ramp source sheet`);
  }
  return { raw: data, width: info.width, height: info.height };
}

async function createImagegenRampPanel(source, row, direction, variant) {
  const column = getImagegenSourceColumn(direction);
  const bounds = findImagegenPanelBounds(source, row, column);
  const { data } = await sharp(SOURCE_PATH)
    .ensureAlpha()
    .extract(bounds)
    .resize(CELL, CELL, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
    .raw()
    .toBuffer({ resolveWithObject: true });
  const panel = Buffer.alloc(CELL * CELL * 4, 0);
  let visible = 0;
  for (let y = 0; y < CELL; y += 1) {
    for (let x = 0; x < CELL; x += 1) {
      const offset = (y * CELL + x) * 4;
      const cleaned = cleanImagegenPixel([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3]
      ], x, y, variant);
      if (cleaned[3] > 24) visible += 1;
      panel[offset] = cleaned[0];
      panel[offset + 1] = cleaned[1];
      panel[offset + 2] = cleaned[2];
      panel[offset + 3] = cleaned[3];
    }
  }
  if (visible < CELL * CELL * 0.12) {
    throw new Error(`Imagegen ramp panel row ${row} ${direction} did not retain enough visible art`);
  }
  return panel;
}

function drawImagegenRampCell(target, panel, cellIndex) {
  const cellX = cellIndex * CELL;
  for (let y = 0; y < CELL; y += 1) {
    for (let x = 0; x < CELL; x += 1) {
      const sourceOffset = (y * CELL + x) * 4;
      setPixel(target, WIDTH, cellX + x, y, [
        panel[sourceOffset],
        panel[sourceOffset + 1],
        panel[sourceOffset + 2],
        panel[sourceOffset + 3]
      ]);
    }
  }
}

async function generateRampAtlas(themeId, terrainAsset, imagegenSource) {
  const target = Buffer.alloc(WIDTH * HEIGHT * 4, 0);
  if (!terrainAsset || !terrainAsset.path) throw new Error(`Missing terrain asset for ${themeId}`);
  const row = getImagegenSourceRow(themeId);
  const upPanel = await createImagegenRampPanel(imagegenSource, row, 'up-right', false);
  const downPanel = await createImagegenRampPanel(imagegenSource, row, 'down-right', false);
  const upVariantPanel = await createImagegenRampPanel(imagegenSource, row, 'up-right', true);
  const downVariantPanel = await createImagegenRampPanel(imagegenSource, row, 'down-right', true);
  drawImagegenRampCell(target, upPanel, 0);
  drawImagegenRampCell(target, downPanel, 1);
  drawImagegenRampCell(target, upVariantPanel, 2);
  drawImagegenRampCell(target, downVariantPanel, 3);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const outPath = path.join(OUT_DIR, `${themeId}.png`);
  await sharp(target, { raw: { width: WIDTH, height: HEIGHT, channels: 4 } }).png().toFile(outPath);
  return outPath;
}

async function validateRampAtlas(themeId) {
  const filePath = path.join(OUT_DIR, `${themeId}.png`);
  if (!fs.existsSync(filePath)) throw new Error(`Missing ramp atlas: ${filePath}`);
  const image = sharp(filePath).ensureAlpha();
  const metadata = await image.metadata();
  if (metadata.width !== WIDTH || metadata.height !== HEIGHT) {
    throw new Error(`${filePath} should be a ${WIDTH}x${HEIGHT} ${RAMP_ATLAS_SCHEMA} atlas`);
  }
  const { data } = await image.raw().toBuffer({ resolveWithObject: true });
  const corners = [
    data[3],
    data[(WIDTH - 1) * 4 + 3],
    data[((HEIGHT - 1) * WIDTH) * 4 + 3],
    data[((HEIGHT - 1) * WIDTH + WIDTH - 1) * 4 + 3]
  ];
  if (corners.some((alpha) => alpha > 0)) throw new Error(`${filePath} should have transparent corners`);
}

async function main() {
  const validateOnly = process.argv.includes('--validate');
  const data = require('../js/games/project-starfall/project-starfall-data.js');
  const terrainAssets = data.ENVIRONMENT_ASSETS && data.ENVIRONMENT_ASSETS.terrain || {};
  const themeIds = Object.keys(terrainAssets);
  if (!themeIds.length) throw new Error('No Project Starfall terrain assets found');
  if (validateOnly) {
    for (const themeId of themeIds) await validateRampAtlas(themeId);
    console.log(`validated ${themeIds.length} Project Starfall ramp atlases`);
    return;
  }
  const imagegenSource = await readImagegenSource();
  for (const themeId of themeIds) await generateRampAtlas(themeId, terrainAssets[themeId], imagegenSource);
  for (const themeId of themeIds) await validateRampAtlas(themeId);
  console.log(`generated ${themeIds.length} Project Starfall ramp atlases from imagegen source`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exit(1);
  });
}
