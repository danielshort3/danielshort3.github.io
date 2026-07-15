#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const {
  GUIDE_LINE_HEX,
  detectGuideGrid,
  getGridCellRect,
  isGuidePixelRgba
} = require('./project-starfall-sheet-grid.js');
const { processSemanticSkillFx } = require('./generate-project-starfall-combat-fx.js');

const ROOT = path.resolve(__dirname, '..');
const Data = require('../js/games/project-starfall/project-starfall-data.js');

const STARFALL_ROOT = path.join(ROOT, 'img/project-starfall');
const BACKUP_ROOT = path.join(STARFALL_ROOT, 'backups/procedural');
const KEY_COLOR = Object.freeze({ r: 0, g: 255, b: 0 });
const GUIDE_COLOR = Object.freeze({ r: 0, g: 255, b: 255 });
const TRANSPARENT = Object.freeze({ r: 0, g: 0, b: 0, alpha: 0 });
const SOURCE_COMPONENT_MIN_AREA = 28;
const FINAL_COMPONENT_MIN_AREA = 8;

const SOURCE_FILES = Object.freeze({
  menu: path.join(STARFALL_ROOT, 'ui/source/ai-menu-icons-sheet.png'),
  coupons: path.join(STARFALL_ROOT, 'items/source/ai-items-rate-coupons.png'),
  portals: path.join(STARFALL_ROOT, 'animations/portals/source/ai-portals-sheet.png'),
  globalFx: path.join(STARFALL_ROOT, 'animations/fx/source/ai-global-fx-sheet.png'),
  mageProjectiles: path.join(STARFALL_ROOT, 'animations/combat-fx/projectiles/source/ai-mage-projectile-rows.png'),
  structures: path.join(STARFALL_ROOT, 'environment/structures/source/ai-town-landmarks.png')
});

const MENU_ICON_ORDER = Object.freeze([
  'character',
  'equipment',
  'partyPanel',
  'inventory',
  'skills',
  'quests',
  'worldmap',
  'monsters',
  'shop',
  'upgrade',
  'cashShop',
  'beta',
  'guide',
  'log',
  'settings',
  'keybinds',
  'admin',
  'logout'
]);

const RATE_COUPON_ORDER = Object.freeze([
  'xp_coupon_1_2_1h',
  'xp_coupon_1_5_1h',
  'xp_coupon_2_0_1h',
  'drop_coupon_1_2_1h',
  'drop_coupon_1_5_1h',
  'drop_coupon_2_0_1h'
]);

const PORTAL_ROWS = Object.freeze([
  Object.freeze({ file: 'standard-sheet.png', row: 0 }),
  Object.freeze({ file: 'boss-sheet.png', row: 1 }),
  Object.freeze({ file: 'locked-sheet.png', row: 2 })
]);

const FX_ROWS = Object.freeze([
  Object.freeze({ id: 'slash', file: 'slash-sheet.png', row: 0 }),
  Object.freeze({ id: 'cast', file: 'cast-sheet.png', row: 1 }),
  Object.freeze({ id: 'arrowRelease', file: 'arrow-release-sheet.png', row: 2 }),
  Object.freeze({ id: 'partyBuff', file: 'party-buff-sheet.png', row: 3 }),
  Object.freeze({ id: 'impact', file: 'impact-sheet.png', row: 4 }),
  Object.freeze({ id: 'defeatBurst', file: 'defeat-burst-sheet.png', row: 5 })
]);

const MAP_DERIVATIONS = Object.freeze([
  Object.freeze({ target: 'maps/frostfen-outskirts.webp', source: 'maps/source/safe-zones/frostfen-camp.png', brightness: 1.04, saturation: 0.92, hue: 196 }),
  Object.freeze({ target: 'maps/glacier-spine.webp', source: 'maps/source/safe-zones/frostfen-camp.png', brightness: 1.08, saturation: 0.86, hue: 206 }),
  Object.freeze({ target: 'maps/rimewarden-sanctum.webp', source: 'maps/source/safe-zones/frostfen-camp.png', brightness: 0.78, saturation: 1.08, hue: 218 }),
  Object.freeze({ target: 'maps/brambleking-court.webp', source: 'maps/source/field/bramble-depths.png', brightness: 0.78, saturation: 1.16, hue: 330 }),
  Object.freeze({ target: 'maps/titan-foundry.webp', source: 'maps/source/field/gearworks-vault.png', brightness: 0.86, saturation: 0.9, hue: 42 }),
  Object.freeze({ target: 'maps/deepcore-core.webp', source: 'maps/source/field/oreback-quarry.png', brightness: 0.8, saturation: 0.94, hue: 165 }),
  Object.freeze({ target: 'maps/emberjaw-furnace.webp', source: 'maps/source/field/emberjaw-lair.png', brightness: 0.86, saturation: 1.2, hue: 14 }),
  Object.freeze({ target: 'maps/rimewarden-vault.webp', source: 'maps/source/safe-zones/frostfen-camp.png', brightness: 0.9, saturation: 1.05, hue: 204 }),
  Object.freeze({ target: 'maps/stormbreak-aerie.webp', source: 'maps/source/field/stormbreak-cliffs.png', brightness: 0.92, saturation: 1.02, hue: 220 }),
  Object.freeze({ target: 'maps/astral-stacks.webp', source: 'maps/source/field/astral-archive.png', brightness: 0.82, saturation: 1.12, hue: 262 }),
  Object.freeze({ target: 'maps/eclipse-throne.webp', source: 'maps/source/field/eclipse-frontier.png', brightness: 0.76, saturation: 1.08, hue: 286 }),
  Object.freeze({ target: 'maps/trials/guardian-trial.webp', source: 'maps/source/field/starfall-crossing.png', brightness: 1.02, saturation: 0.9, hue: 205 }),
  Object.freeze({ target: 'maps/trials/berserker-trial.webp', source: 'maps/source/field/cinder-hollow.png', brightness: 0.9, saturation: 1.18, hue: 352 }),
  Object.freeze({ target: 'maps/trials/duelist-trial.webp', source: 'maps/source/field/bandit-ridge-camp.png', brightness: 1.02, saturation: 0.98, hue: 30 }),
  Object.freeze({ target: 'maps/trials/fire-mage-trial.webp', source: 'maps/source/field/emberjaw-lair.png', brightness: 1.04, saturation: 1.22, hue: 18 }),
  Object.freeze({ target: 'maps/trials/rune-mage-trial.webp', source: 'maps/source/field/astral-archive.png', brightness: 1.04, saturation: 1.08, hue: 174 }),
  Object.freeze({ target: 'maps/trials/storm-mage-trial.webp', source: 'maps/source/field/stormbreak-cliffs.png', brightness: 1.0, saturation: 1.0, hue: 208 }),
  Object.freeze({ target: 'maps/trials/sniper-trial.webp', source: 'maps/source/field/bandit-ridge-camp.png', brightness: 1.04, saturation: 0.88, hue: 44 }),
  Object.freeze({ target: 'maps/trials/trapper-trial.webp', source: 'maps/source/field/thornpath-thicket.png', brightness: 0.92, saturation: 1.12, hue: 342 }),
  Object.freeze({ target: 'maps/trials/beast-archer-trial.webp', source: 'maps/source/field/greenroot-meadow.png', brightness: 1.04, saturation: 1.06, hue: 96 })
]);

const BASIC_FX_ROWS = Object.freeze([1, 2, 4, 0]);
const ENEMY_FX_ROWS = Object.freeze([1, 0, 2, 3, 4]);
const ENEMY_FX_HUE_SLOT_COUNT = 60;
const MAGE_PROJECTILE_SOURCE_ROWS = Object.freeze({
  magic: 0,
  fire: 1,
  rune: 2,
  lightning: 3
});
let globalFxGridPromise = null;
const globalFxRowCache = new Map();
const mageProjectileRowCache = new Map();
const sourceGridCache = new Map();

function rel(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function backupPathFor(filePath) {
  const relative = path.relative(STARFALL_ROOT, filePath);
  return path.join(BACKUP_ROOT, relative);
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function assertExists(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing ${label}: ${rel(filePath)}`);
  }
}

function backupOutput(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const backupPath = backupPathFor(filePath);
  if (fs.existsSync(backupPath)) return false;
  ensureDir(backupPath);
  fs.copyFileSync(filePath, backupPath);
  return true;
}

function colorDistanceSq(r, g, b, color) {
  const dr = r - color.r;
  const dg = g - color.g;
  const db = b - color.b;
  return dr * dr + dg * dg + db * db;
}

function shouldRemovePixel(r, g, b) {
  if (g >= 205 && r <= 95 && b <= 95 && g - Math.max(r, b) >= 92) return true;
  if (colorDistanceSq(r, g, b, KEY_COLOR) <= 42 * 42) return true;
  return colorDistanceSq(r, g, b, GUIDE_COLOR) <= 46 * 46;
}

function isBackgroundKeyPixel(raw, offset) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  if (shouldRemovePixel(r, g, b)) return true;
  return isGuidePixelRgba(raw, offset, GUIDE_LINE_HEX);
}

function softenGreenFringe(raw, offset) {
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  if (g > Math.max(r, b) + 34 && r < 160 && b < 180) {
    raw[offset + 1] = Math.min(g, Math.max(r, b) + 24);
  }
}

function clearBackgroundKeys(raw, width, height) {
  removeEdgeConnectedKeys(raw, width, height);
  for (let offset = 0; offset < raw.length; offset += 4) {
    if (raw[offset + 3] <= 0) continue;
    if (isBackgroundKeyPixel(raw, offset)) {
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    } else {
      softenGreenFringe(raw, offset);
    }
  }
  return raw;
}

function clearResidualGreenBackdrop(raw) {
  const data = Buffer.from(raw);
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3];
    if (alpha <= 0) continue;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const darkGreenChroma = g >= 8 && g <= 140 && r <= 90 && b <= 110 && g - Math.max(r, b) >= 3;
    const greenDominant = g > Math.max(r, b) + 10 && g > 18 && r < 92 && b < 92;
    const muddyKey = g > r * 1.12 && g > b * 1.12 && r < 72 && b < 72;
    if (!darkGreenChroma && !greenDominant && !muddyKey) continue;
    data[offset] = 0;
    data[offset + 1] = 0;
    data[offset + 2] = 0;
    data[offset + 3] = 0;
  }
  return data;
}

function thickenProjectileAlphaVertically(raw, width, height, radius) {
  const safeRadius = Math.max(0, Math.round(Number(radius || 0)));
  if (!safeRadius) return raw;
  const source = Buffer.from(raw);
  const data = Buffer.from(raw);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const sourceOffset = (y * width + x) * 4;
      const alpha = source[sourceOffset + 3];
      if (alpha <= 24) continue;
      for (let dy = -safeRadius; dy <= safeRadius; dy += 1) {
        if (!dy) continue;
        const targetY = y + dy;
        if (targetY < 0 || targetY >= height) continue;
        const targetOffset = (targetY * width + x) * 4;
        const spreadAlpha = Math.round(alpha * (dy < 0 ? 0.72 : 0.66));
        if (data[targetOffset + 3] >= spreadAlpha) continue;
        data[targetOffset] = source[sourceOffset];
        data[targetOffset + 1] = source[sourceOffset + 1];
        data[targetOffset + 2] = source[sourceOffset + 2];
        data[targetOffset + 3] = spreadAlpha;
      }
    }
  }
  return data;
}

function removeEdgeConnectedKeys(raw, width, height) {
  const visited = new Uint8Array(width * height);
  const queue = [];
  const add = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const index = y * width + x;
    if (visited[index]) return;
    const offset = index * 4;
    if (!isBackgroundKeyPixel(raw, offset)) return;
    visited[index] = 1;
    queue.push(index);
  };
  for (let x = 0; x < width; x += 1) {
    add(x, 0);
    add(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    add(0, y);
    add(width - 1, y);
  }
  for (let head = 0; head < queue.length; head += 1) {
    const index = queue[head];
    const x = index % width;
    const y = Math.floor(index / width);
    const offset = index * 4;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
    add(x + 1, y);
    add(x - 1, y);
    add(x, y + 1);
    add(x, y - 1);
  }
  return raw;
}

function getAlphaBounds(raw, width, height, alphaThreshold) {
  const threshold = Number(alphaThreshold == null ? 16 : alphaThreshold);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= threshold) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  return maxX >= minX && maxY >= minY
    ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }
    : null;
}

function findAlphaComponents(raw, width, height, alphaThreshold) {
  const threshold = Number(alphaThreshold == null ? 0 : alphaThreshold);
  const visited = new Uint8Array(width * height);
  const queue = [];
  const component = [];
  const components = [];
  for (let start = 0; start < width * height; start += 1) {
    if (visited[start] || raw[start * 4 + 3] <= threshold) continue;
    queue.length = 0;
    component.length = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    visited[start] = 1;
    queue.push(start);
    for (let head = 0; head < queue.length; head += 1) {
      const index = queue[head];
      const x = index % width;
      const y = Math.floor(index / width);
      component.push(index);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (!dx && !dy) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const next = ny * width + nx;
          if (visited[next] || raw[next * 4 + 3] <= threshold) continue;
          visited[next] = 1;
          queue.push(next);
        }
      }
    }
    components.push({
      pixels: component.slice(),
      area: component.length,
      minX,
      minY,
      maxX,
      maxY,
      touchesEdge: minX <= 1 || minY <= 1 || maxX >= width - 2 || maxY >= height - 2
    });
  }
  return components.sort((a, b) => b.area - a.area);
}

function removeSmallAlphaComponents(raw, width, height, minArea) {
  const areaFloor = Math.max(1, Number(minArea || 1));
  for (let index = 0; index < width * height; index += 1) {
    if (raw[index * 4 + 3] <= 8) raw[index * 4 + 3] = 0;
  }
  findAlphaComponents(raw, width, height, 0).forEach((component) => {
    if (component.area >= areaFloor) return;
    component.pixels.forEach((index) => {
      const offset = index * 4;
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    });
  });
  return raw;
}

function isThinEdgeFragment(component, width, height, edgeBand) {
  const band = Math.max(4, Math.round(Number(edgeBand || 12)));
  const boxWidth = component.maxX - component.minX + 1;
  const boxHeight = component.maxY - component.minY + 1;
  const nearHorizontalEdge = component.minY <= band || component.maxY >= height - band - 1;
  const nearVerticalEdge = component.minX <= band || component.maxX >= width - band - 1;
  if (boxHeight <= 2 && boxWidth >= 16 && nearHorizontalEdge) return true;
  return boxWidth <= 2 && boxHeight >= 16 && nearVerticalEdge;
}

function removeThinEdgeFragments(raw, width, height, edgeBand) {
  findAlphaComponents(raw, width, height, 16).forEach((component) => {
    if (component.area > Math.max(320, width * height * 0.012)) return;
    if (!isThinEdgeFragment(component, width, height, edgeBand)) return;
    component.pixels.forEach((index) => {
      const offset = index * 4;
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    });
  });
  return raw;
}

async function loadSourceGrid(sourcePath, columns, rows, label) {
  assertExists(sourcePath, 'AI source sheet');
  const cacheKey = `${sourcePath}:${columns}:${rows}`;
  if (sourceGridCache.has(cacheKey)) return sourceGridCache.get(cacheKey);
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(data, info.width, info.height, {
    columns,
    rows,
    label: label || rel(sourcePath),
    guideColor: GUIDE_LINE_HEX
  });
  const source = {
    sourcePath,
    columns,
    rows,
    width: info.width,
    height: info.height,
    grid
  };
  sourceGridCache.set(cacheKey, source);
  return source;
}

async function extractGuideCell(source, index, inset) {
  const col = index % source.columns;
  const row = Math.floor(index / source.columns);
  const rect = getGridCellRect(source.grid, row, col, inset == null ? 3 : inset);
  const { data, info } = await sharp(source.sourcePath)
    .extract({ left: rect.x, top: rect.y, width: rect.w, height: rect.h })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const raw = removeSmallAlphaComponents(clearBackgroundKeys(Buffer.from(data), info.width, info.height), info.width, info.height, SOURCE_COMPONENT_MIN_AREA);
  return {
    raw,
    width: info.width,
    height: info.height
  };
}

async function prepareCellImage(cell, options) {
  const settings = options || {};
  const outputWidth = Math.max(1, Math.round(Number(settings.outputWidth || settings.outputSize || 64)));
  const outputHeight = Math.max(1, Math.round(Number(settings.outputHeight || settings.outputSize || outputWidth)));
  const safeMargin = Math.max(0, Math.round(Number(settings.safeMargin == null ? 6 : settings.safeMargin)));
  const cropPadding = Math.max(0, Math.round(Number(settings.cropPadding == null ? 4 : settings.cropPadding)));
  const placement = settings.placement || 'center';
  const bounds = getAlphaBounds(cell.raw, cell.width, cell.height, settings.alphaThreshold == null ? 16 : settings.alphaThreshold);
  if (!bounds) {
    return sharp({
      create: {
        width: outputWidth,
        height: outputHeight,
        channels: 4,
        background: TRANSPARENT
      }
    }).png({ compressionLevel: 9 }).toBuffer();
  }
  const cropLeft = Math.max(0, bounds.x - cropPadding);
  const cropTop = Math.max(0, bounds.y - cropPadding);
  const cropRight = Math.min(cell.width, bounds.x + bounds.w + cropPadding);
  const cropBottom = Math.min(cell.height, bounds.y + bounds.h + cropPadding);
  const cropWidth = Math.max(1, cropRight - cropLeft);
  const cropHeight = Math.max(1, cropBottom - cropTop);
  const maxWidth = Math.max(1, outputWidth - safeMargin * 2);
  const maxHeight = Math.max(1, outputHeight - safeMargin * 2);
  const resized = await sharp(cell.raw, {
    raw: { width: cell.width, height: cell.height, channels: 4 }
  })
    .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
    .resize(maxWidth, maxHeight, {
      fit: 'inside',
      kernel: sharp.kernel.lanczos3,
      background: TRANSPARENT
    })
    .png()
    .toBuffer();
  const metadata = await sharp(resized).metadata();
  const left = Math.round((outputWidth - metadata.width) / 2);
  const top = placement === 'bottom'
    ? Math.max(safeMargin, outputHeight - safeMargin - metadata.height)
    : Math.round((outputHeight - metadata.height) / 2);
  const composed = await sharp({
    create: {
      width: outputWidth,
      height: outputHeight,
      channels: 4,
      background: TRANSPARENT
    }
  })
    .composite([{ input: resized, left, top }])
    .raw()
    .toBuffer();
  let cleaned = removeThinEdgeFragments(
    removeSmallAlphaComponents(clearBackgroundKeys(composed, outputWidth, outputHeight), outputWidth, outputHeight, FINAL_COMPONENT_MIN_AREA),
    outputWidth,
    outputHeight,
    safeMargin + 8
  );
  if (settings.aggressiveGreenKey) {
    cleaned = removeSmallAlphaComponents(clearResidualGreenBackdrop(cleaned), outputWidth, outputHeight, FINAL_COMPONENT_MIN_AREA);
  }
  if (settings.verticalDilate) {
    cleaned = thickenProjectileAlphaVertically(cleaned, outputWidth, outputHeight, settings.verticalDilate);
  }
  return sharp(cleaned, {
    raw: { width: outputWidth, height: outputHeight, channels: 4 }
  }).png({ compressionLevel: 9 }).toBuffer();
}

async function processMenuIcons(written) {
  const columns = 6;
  const cellSize = 64;
  const source = await loadSourceGrid(SOURCE_FILES.menu, columns, 3, 'AI menu icon sheet');
  for (let index = 0; index < MENU_ICON_ORDER.length; index += 1) {
    const id = MENU_ICON_ORDER[index];
    const target = path.join(ROOT, Data.MENU_ICON_ASSETS[id]);
    const cell = await extractGuideCell(source, index, 3);
    backupOutput(target);
    ensureDir(target);
    await sharp(await prepareCellImage(cell, {
      outputSize: cellSize,
      safeMargin: 6,
      placement: 'center'
    })).toFile(target);
    written.push(rel(target));
  }
}

async function processRateCoupons(written) {
  const columns = 3;
  const rows = 2;
  const cellSize = 64;
  const source = await loadSourceGrid(SOURCE_FILES.coupons, columns, rows, 'AI rate coupon sheet');
  const composites = [];
  for (let index = 0; index < RATE_COUPON_ORDER.length; index += 1) {
    const cell = await extractGuideCell(source, index, 3);
    composites.push({
      input: await prepareCellImage(cell, {
        outputSize: cellSize,
        safeMargin: 6,
        placement: 'center'
      }),
      left: (index % columns) * cellSize,
      top: Math.floor(index / columns) * cellSize
    });
  }
  const target = path.join(STARFALL_ROOT, 'items/sheets/ai-items-rate-coupons-sheet.png');
  backupOutput(target);
  ensureDir(target);
  await sharp({
    create: {
      width: columns * cellSize,
      height: rows * cellSize,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(target);
  written.push(rel(target));
}

async function processRowSheet(sourcePath, columns, rows, cellSize, row, target, options) {
  const settings = options || {};
  const source = await loadSourceGrid(sourcePath, columns, rows, settings.label || `AI row sheet ${rel(sourcePath)}`);
  const composites = [];
  for (let frame = 0; frame < columns; frame += 1) {
    const cell = await extractGuideCell(source, row * columns + frame, settings.inset == null ? 3 : settings.inset);
    composites.push({
      input: await prepareCellImage(cell, {
        outputSize: cellSize,
        safeMargin: settings.safeMargin == null ? 8 : settings.safeMargin,
        placement: settings.placement || 'center',
        cropPadding: settings.cropPadding == null ? 4 : settings.cropPadding
      }),
      left: frame * cellSize,
      top: 0
    });
  }
  backupOutput(target);
  ensureDir(target);
  await sharp({
    create: {
      width: columns * cellSize,
      height: cellSize,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(target);
}

async function processPortals(written) {
  for (const entry of PORTAL_ROWS) {
    const target = path.join(STARFALL_ROOT, 'animations/portals', entry.file);
    await processRowSheet(SOURCE_FILES.portals, 6, 3, 160, entry.row, target, {
      label: 'AI portal animation sheet',
      safeMargin: 8,
      placement: 'bottom',
      cropPadding: 3
    });
    written.push(rel(target));
  }
}

async function processGlobalFx(written) {
  for (const entry of FX_ROWS) {
    const target = path.join(STARFALL_ROOT, 'animations/fx', entry.file);
    await processRowSheet(SOURCE_FILES.globalFx, 6, 6, 160, entry.row, target, {
      label: 'AI global combat FX sheet',
      safeMargin: 8,
      placement: 'center',
      cropPadding: 4
    });
    written.push(rel(target));
  }
}

async function getGlobalFxGrid() {
  if (!globalFxGridPromise) {
    globalFxGridPromise = loadSourceGrid(SOURCE_FILES.globalFx, 6, 6, 'AI global combat FX sheet');
  }
  return globalFxGridPromise;
}

async function extractFxCell(rowIndex, frame, options) {
  const settings = options || {};
  const source = await getGlobalFxGrid();
  const cell = await extractGuideCell(source, rowIndex * 6 + frame, settings.inset == null ? 3 : settings.inset);
  return prepareCellImage(cell, {
    outputWidth: settings.outputWidth || settings.outputSize || 160,
    outputHeight: settings.outputHeight || settings.outputSize || 160,
    safeMargin: settings.safeMargin == null ? 8 : settings.safeMargin,
    placement: settings.placement || 'center',
    cropPadding: settings.cropPadding == null ? 4 : settings.cropPadding
  });
}

async function extractFxRow(rowIndex) {
  if (globalFxRowCache.has(rowIndex)) return globalFxRowCache.get(rowIndex);
  const cells = [];
  for (let frame = 0; frame < 6; frame += 1) {
    cells.push(await extractFxCell(rowIndex, frame, {
      outputSize: 160,
      safeMargin: 8,
      placement: 'center'
    }));
  }
  globalFxRowCache.set(rowIndex, cells);
  return cells;
}

async function extractMageProjectileRow(rowIndex) {
  if (mageProjectileRowCache.has(rowIndex)) return mageProjectileRowCache.get(rowIndex);
  const source = await loadSourceGrid(SOURCE_FILES.mageProjectiles, 6, 4, 'AI mage projectile row sheet');
  const cells = [];
  for (let frame = 0; frame < 6; frame += 1) {
    const cell = await extractGuideCell(source, rowIndex * 6 + frame, 3);
    cells.push(await prepareCellImage(cell, {
      outputSize: 160,
      safeMargin: 10,
      placement: 'center',
      cropPadding: 4,
      aggressiveGreenKey: true,
      verticalDilate: 5
    }));
  }
  mageProjectileRowCache.set(rowIndex, cells);
  return cells;
}

async function tintBuffer(buffer, hue, saturation, brightness) {
  return sharp(buffer)
    .modulate({
      hue: Math.round(Number(hue || 0)),
      saturation: Math.max(0.2, Number(saturation || 1)),
      brightness: Math.max(0.2, Number(brightness || 1))
    })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

function hashIdentity32(value) {
  const text = String(value || '');
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function buildEnemyFxIdentityMap(enemyIds) {
  const ids = Array.from(new Set((enemyIds || []).map((id) => String(id || '')).filter(Boolean))).sort();
  if (ids.length > ENEMY_FX_HUE_SLOT_COUNT) {
    throw new Error(`Enemy combat FX palette supports ${ENEMY_FX_HUE_SLOT_COUNT} unique hue slots, got ${ids.length} enemies`);
  }
  const identities = new Map();
  const usedHueSlots = new Set();

  for (const enemyId of ids) {
    const hash = hashIdentity32(`project-starfall:enemy-fx:${enemyId}`);
    let hueSlot = hash % ENEMY_FX_HUE_SLOT_COUNT;
    let probes = 0;
    while (usedHueSlots.has(hueSlot)) {
      hueSlot = (hueSlot + 23) % ENEMY_FX_HUE_SLOT_COUNT;
      probes += 1;
      if (probes >= ENEMY_FX_HUE_SLOT_COUNT) throw new Error('Enemy combat FX palette exhausted all available hue slots');
    }
    usedHueSlots.add(hueSlot);
    identities.set(enemyId, Object.freeze({
      hue: hueSlot * (360 / ENEMY_FX_HUE_SLOT_COUNT),
      saturation: 0.88 + ((hash >>> 9) % 9) * 0.04,
      brightness: 0.92 + ((hash >>> 17) % 7) * 0.025
    }));
  }

  return identities;
}

function getSkillById(skillId) {
  return (Data.SKILLS || []).find((skill) => skill && skill.id === skillId) || null;
}

function getMageProjectileTypeForSkill(skill) {
  if (!skill || !skill.targeting) return null;
  const mode = skill.targeting.mode;
  if (mode !== 'projectile' && mode !== 'chain') return null;
  const explicitType = skill.targeting.projectileType;
  if (Object.prototype.hasOwnProperty.call(MAGE_PROJECTILE_SOURCE_ROWS, explicitType)) return explicitType;
  if (skill.ownerClass === 'fireMage') return 'fire';
  if (skill.ownerClass === 'runeMage') return 'rune';
  if (skill.ownerClass === 'stormMage') return 'lightning';
  if (skill.ownerClass === 'mage') return 'magic';
  return null;
}

function makeMageProjectileRowOverride(projectileType) {
  if (!Object.prototype.hasOwnProperty.call(MAGE_PROJECTILE_SOURCE_ROWS, projectileType)) return null;
  return {
    preserveColor: true,
    cells: () => extractMageProjectileRow(MAGE_PROJECTILE_SOURCE_ROWS[projectileType])
  };
}

async function composeRows(rowIndexes, target, options) {
  const settings = options || {};
  const rowOverrides = settings.rowOverrides || {};
  const composites = [];
  for (let row = 0; row < rowIndexes.length; row += 1) {
    const rowOverride = rowOverrides[row] || null;
    const cells = rowOverride ? await rowOverride.cells() : await extractFxRow(rowIndexes[row]);
    for (let frame = 0; frame < cells.length; frame += 1) {
      const input = rowOverride && rowOverride.preserveColor
        ? cells[frame]
        : await tintBuffer(cells[frame], settings.hue, settings.saturation, settings.brightness);
      composites.push({
        input,
        left: frame * 160,
        top: row * 160
      });
    }
  }
  backupOutput(target);
  ensureDir(target);
  const width = 960;
  const height = rowIndexes.length * 160;
  const composed = await sharp({
    create: {
      width,
      height,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .raw()
    .toBuffer();
  await sharp(clearResidualGreenBackdrop(composed), {
    raw: { width, height, channels: 4 }
  })
    .png({ compressionLevel: 9 })
    .toFile(target);
}

function getEnemyCombatFxEntries() {
  return Object.entries(Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {});
}

function getEnemyFxSelection(argv) {
  const args = argv || process.argv.slice(2);
  const inline = args.find((argument) => String(argument).startsWith('--enemy='));
  const optionIndex = args.indexOf('--enemy');
  const value = inline
    ? String(inline).slice('--enemy='.length)
    : optionIndex >= 0 ? args[optionIndex + 1] : '';
  return String(value || '')
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean);
}

function getSkillFxSelection(argv) {
  const args = argv || process.argv.slice(2);
  const inline = args.find((argument) => String(argument).startsWith('--skill='));
  const optionIndex = args.indexOf('--skill');
  const value = inline
    ? String(inline).slice('--skill='.length)
    : optionIndex >= 0 ? args[optionIndex + 1] : '';
  return String(value || '')
    .split(',')
    .map((id) => id.trim())
    .filter(Boolean);
}

async function processEnemyCombatFx(written, onlyEnemyIds) {
  const enemyEntries = getEnemyCombatFxEntries();
  const identities = buildEnemyFxIdentityMap(enemyEntries.map(([enemyId]) => enemyId));
  const selectedIds = new Set(onlyEnemyIds || []);
  if (selectedIds.size) {
    const knownIds = new Set(enemyEntries.map(([enemyId]) => enemyId));
    const unknownIds = Array.from(selectedIds).filter((enemyId) => !knownIds.has(enemyId));
    if (unknownIds.length) throw new Error(`Unknown enemy combat FX id(s): ${unknownIds.join(', ')}`);
  }

  for (const [enemyId, animation] of enemyEntries) {
    if (selectedIds.size && !selectedIds.has(enemyId)) continue;
    const identity = identities.get(enemyId);
    const target = path.join(ROOT, animation.sheet);
    await composeRows(ENEMY_FX_ROWS, target, identity);
    written.push(rel(target));
  }
}

async function processDerivedCombatFx(written) {
  const basicEntries = Object.entries(Data.BASIC_ATTACK_FX_ANIMATION_ASSETS || {});
  for (const [classId, animation] of basicEntries) {
    const target = path.join(ROOT, animation.sheet);
    const projectileOverride = classId === 'mage'
      ? makeMageProjectileRowOverride('magic')
      : null;
    await composeRows(BASIC_FX_ROWS, target, {
      hue: classId === 'mage' ? 244 : classId === 'archer' ? 118 : 28,
      saturation: classId === 'mage' ? 1.12 : 1,
      brightness: 1,
      rowOverrides: projectileOverride ? { 1: projectileOverride } : null
    });
    written.push(rel(target));
  }

  await processEnemyCombatFx(written);

  const skillResult = await processSemanticSkillFx();
  skillResult.outputPaths.forEach((target) => written.push(rel(target)));

  const projectile = Data.ENEMY_PROJECTILE_ANIMATION_ASSETS && Data.ENEMY_PROJECTILE_ANIMATION_ASSETS.banditThrower;
  if (projectile && projectile.sheet) {
    const target = path.join(ROOT, projectile.sheet);
    backupOutput(target);
    ensureDir(target);
    await sharp({
      create: {
        width: 192,
        height: 64,
        channels: 4,
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      }
    })
      .composite(await Promise.all([0, 2, 4].map(async (sourceIndex, frame) => ({
        input: await tintBuffer(await extractFxCell(2, sourceIndex, {
          outputSize: 64,
          safeMargin: 8,
          placement: 'center'
        }), 34, 0.82, 0.96),
        left: frame * 64,
        top: 0
      }))))
      .png({ compressionLevel: 9 })
      .toFile(target);
    written.push(rel(target));
  }
}

async function processBasicMageCombatFx(written) {
  const animation = Data.BASIC_ATTACK_FX_ANIMATION_ASSETS && Data.BASIC_ATTACK_FX_ANIMATION_ASSETS.mage;
  if (!animation || !animation.sheet) throw new Error('Missing basic mage combat FX animation asset');
  const target = path.join(ROOT, animation.sheet);
  await composeRows(BASIC_FX_ROWS, target, {
    hue: 244,
    saturation: 1.12,
    brightness: 1,
    rowOverrides: { 1: makeMageProjectileRowOverride('magic') }
  });
  written.push(rel(target));
  await validateAnimationSheet(animation, { minMargin: 8 });
  await validateHorizontalProjectileRow(target, 'Basic mage', 1);
}

async function processMageProjectileCombatFx(written) {
  await processBasicMageCombatFx(written);
  const skillEntries = Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {});
  const projectileEntries = skillEntries.filter(([skillId]) => getMageProjectileTypeForSkill(getSkillById(skillId)));
  if (projectileEntries.length) {
    const skillResult = await processSemanticSkillFx({ skillIds: projectileEntries.map(([skillId]) => skillId) });
    skillResult.outputPaths.forEach((target) => written.push(rel(target)));
  }
  for (const [skillId, animation] of projectileEntries) {
    const target = path.join(ROOT, animation.sheet);
    await validateAnimationSheet(animation, { minMargin: 8 });
    await validateHorizontalProjectileRow(target, `Skill ${skillId}`, 1);
  }
}

async function processStructures(written) {
  const target = path.join(STARFALL_ROOT, 'environment/structures/town-landmarks.png');
  const columns = 4;
  const rows = 2;
  const cellSize = 256;
  const source = await loadSourceGrid(SOURCE_FILES.structures, columns, rows, 'AI town landmark sheet');
  const composites = [];
  for (let index = 0; index < columns * rows; index += 1) {
    const cell = await extractGuideCell(source, index, 3);
    composites.push({
      input: await prepareCellImage(cell, {
        outputSize: cellSize,
        safeMargin: 12,
        placement: 'bottom',
        cropPadding: 5
      }),
      left: (index % columns) * cellSize,
      top: Math.floor(index / columns) * cellSize
    });
  }
  backupOutput(target);
  ensureDir(target);
  await sharp({
    create: {
      width: columns * cellSize,
      height: rows * cellSize,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9 })
    .toFile(target);
  written.push(rel(target));
}

async function processMaps(written) {
  for (const entry of MAP_DERIVATIONS) {
    const source = path.join(STARFALL_ROOT, entry.source);
    const target = path.join(STARFALL_ROOT, entry.target);
    assertExists(source, 'AI map source');
    backupOutput(target);
    ensureDir(target);
    await sharp(source)
      .resize(1280, 640, { fit: 'cover', position: 'center', kernel: sharp.kernel.lanczos3 })
      .modulate({
        brightness: Number(entry.brightness || 1),
        saturation: Number(entry.saturation || 1),
        hue: Math.round(Number(entry.hue || 0))
      })
      .webp({ quality: 86 })
      .toFile(target);
    written.push(rel(target));
  }
}

async function validatePng(filePath, width, height) {
  const metadata = await sharp(filePath).metadata();
  if (metadata.width !== width || metadata.height !== height) {
    throw new Error(`${rel(filePath)} is ${metadata.width}x${metadata.height}; expected ${width}x${height}`);
  }
}

function hasEdgeAlpha(raw, width, height, alphaThreshold) {
  const threshold = Number(alphaThreshold == null ? 32 : alphaThreshold);
  for (let x = 0; x < width; x += 1) {
    if (raw[x * 4 + 3] > threshold) return true;
    if (raw[((height - 1) * width + x) * 4 + 3] > threshold) return true;
  }
  for (let y = 1; y < height - 1; y += 1) {
    if (raw[(y * width) * 4 + 3] > threshold) return true;
    if (raw[(y * width + width - 1) * 4 + 3] > threshold) return true;
  }
  return false;
}

function hasGuideOrKeyArtifactNearEdge(raw, width, height, edgeWidth) {
  const band = Math.max(1, Math.round(Number(edgeWidth || 4)));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (x >= band && y >= band && x < width - band && y < height - band) continue;
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= 18) continue;
      if (isBackgroundKeyPixel(raw, offset)) return true;
    }
  }
  return false;
}

async function validateCellMargins(filePath, columns, rows, cellWidth, cellHeight, options) {
  const settings = options || {};
  await validatePng(filePath, columns * cellWidth, rows * cellHeight);
  const minMargin = Math.max(0, Number(settings.minMargin == null ? 4 : settings.minMargin));
  const alphaThreshold = Number(settings.alphaThreshold == null ? 24 : settings.alphaThreshold);
  const maxCenterOffsetX = settings.maxCenterOffsetX == null ? null : Number(settings.maxCenterOffsetX);
  const maxCenterOffsetY = settings.maxCenterOffsetY == null ? null : Number(settings.maxCenterOffsetY);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const { data } = await sharp(filePath)
        .extract({ left: col * cellWidth, top: row * cellHeight, width: cellWidth, height: cellHeight })
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const context = `${rel(filePath)} cell ${row + 1}:${col + 1}`;
      const bounds = getAlphaBounds(data, cellWidth, cellHeight, alphaThreshold);
      if (!bounds) throw new Error(`${context} has no visible pixels`);
      if (hasEdgeAlpha(data, cellWidth, cellHeight, 32)) {
        throw new Error(`${context} has visible pixels touching the cell edge`);
      }
      if (hasGuideOrKeyArtifactNearEdge(data, cellWidth, cellHeight, Math.max(4, minMargin))) {
        throw new Error(`${context} has residual guide/key-colored pixels near the cell edge`);
      }
      const thinFragment = findAlphaComponents(data, cellWidth, cellHeight, 16)
        .find((component) =>
          component.area <= Math.max(320, cellWidth * cellHeight * 0.012) &&
          isThinEdgeFragment(component, cellWidth, cellHeight, minMargin + 8));
      if (thinFragment) {
        throw new Error(`${context} has a detached thin edge fragment (${thinFragment.minX},${thinFragment.minY})-(${thinFragment.maxX},${thinFragment.maxY})`);
      }
      const margin = Math.min(
        bounds.x,
        bounds.y,
        cellWidth - bounds.x - bounds.w,
        cellHeight - bounds.y - bounds.h
      );
      if (margin < minMargin) {
        throw new Error(`${context} has only ${margin}px visible margin; expected at least ${minMargin}px`);
      }
      if (maxCenterOffsetX != null) {
        const offsetX = Math.abs(bounds.x + bounds.w / 2 - cellWidth / 2);
        if (offsetX > maxCenterOffsetX) {
          throw new Error(`${context} is horizontally off center by ${offsetX.toFixed(1)}px; expected <= ${maxCenterOffsetX}px`);
        }
      }
      if (maxCenterOffsetY != null) {
        const offsetY = Math.abs(bounds.y + bounds.h / 2 - cellHeight / 2);
        if (offsetY > maxCenterOffsetY) {
          throw new Error(`${context} is vertically off center by ${offsetY.toFixed(1)}px; expected <= ${maxCenterOffsetY}px`);
        }
      }
    }
  }
}

function getFrameAlphaStats(raw, width, height, alphaThreshold) {
  const threshold = Number(alphaThreshold == null ? 24 : alphaThreshold);
  let count = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= threshold) continue;
      count += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      sumX += x;
      sumY += y;
    }
  }
  if (!count) return null;
  const meanX = sumX / count;
  const meanY = sumY / count;
  let varX = 0;
  let covXY = 0;
  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= threshold) continue;
      const dx = x - meanX;
      varX += dx * dx;
      covXY += dx * (y - meanY);
    }
  }
  const slope = varX > 1 ? covXY / varX : 0;
  return {
    count,
    slope,
    angle: Math.atan(slope) * 180 / Math.PI,
    bounds: {
      x: minX,
      y: minY,
      w: maxX - minX + 1,
      h: maxY - minY + 1,
      centerX: (minX + maxX + 1) / 2,
      centerY: (minY + maxY + 1) / 2
    }
  };
}

async function validateHorizontalProjectileRow(filePath, label, rowIndex) {
  await validatePng(filePath, 960, 640);
  const frameSize = 160;
  const row = Number(rowIndex == null ? 1 : rowIndex);
  for (let frame = 0; frame < 6; frame += 1) {
    const { data } = await sharp(filePath)
      .extract({ left: frame * frameSize, top: row * frameSize, width: frameSize, height: frameSize })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const stats = getFrameAlphaStats(data, frameSize, frameSize, 24);
    const context = `${label} projectile frame ${frame + 1}`;
    if (!stats) throw new Error(`${context} has no visible pixels`);
    if (stats.bounds.w < 48 || stats.bounds.h < 24) {
      throw new Error(`${context} should have readable projectile art, got ${stats.bounds.w}x${stats.bounds.h}`);
    }
    if (Math.abs(stats.slope) > 0.16) {
      throw new Error(`${context} is tilted by ${stats.angle.toFixed(1)} degrees; expected a straight horizontal row`);
    }
    const centerOffsetY = Math.abs(stats.bounds.centerY - frameSize / 2);
    if (centerOffsetY > 24) {
      throw new Error(`${context} is vertically off center by ${centerOffsetY.toFixed(1)}px`);
    }
  }
}

async function validateAnimationSheet(animation, options) {
  if (!animation || !animation.sheet) return;
  const filePath = path.join(ROOT, animation.sheet);
  const frameWidth = Number(animation.frameWidth || 160);
  const frameHeight = Number(animation.frameHeight || frameWidth);
  const metadata = await sharp(filePath).metadata();
  if (metadata.width % frameWidth || metadata.height % frameHeight) {
    throw new Error(`${rel(filePath)} dimensions must be divisible by ${frameWidth}x${frameHeight}`);
  }
  await validateCellMargins(filePath, metadata.width / frameWidth, metadata.height / frameHeight, frameWidth, frameHeight, options);
}

async function validateEnemyCombatFxUniqueness() {
  const enemyEntries = getEnemyCombatFxEntries();
  const identities = buildEnemyFxIdentityMap(enemyEntries.map(([enemyId]) => enemyId));
  if (identities.size !== enemyEntries.length) {
    throw new Error(`Enemy combat FX palette contains ${identities.size} identities for ${enemyEntries.length} enemies`);
  }

  const identityKeys = new Map();
  for (const [enemyId, identity] of identities) {
    const key = `${identity.hue}|${identity.saturation.toFixed(3)}|${identity.brightness.toFixed(3)}`;
    if (identityKeys.has(key)) {
      throw new Error(`Enemy combat FX palette collision: ${identityKeys.get(key)} and ${enemyId} both use ${key}`);
    }
    identityKeys.set(key, enemyId);
  }

  const outputHashes = new Map();
  for (const [enemyId, animation] of enemyEntries) {
    await validateAnimationSheet(animation, { minMargin: 8 });
    const filePath = path.join(ROOT, animation.sheet);
    const pixels = await sharp(filePath).ensureAlpha().raw().toBuffer();
    const digest = crypto.createHash('sha256').update(pixels).digest('hex');
    if (outputHashes.has(digest)) {
      throw new Error(`Enemy combat FX output collision: ${outputHashes.get(digest)} and ${enemyId} have identical rendered pixels`);
    }
    outputHashes.set(digest, enemyId);
  }

  return Object.freeze({ identities: identities.size, outputs: outputHashes.size });
}

async function validateOutputs() {
  await Promise.all(MENU_ICON_ORDER.map((id) => validateCellMargins(path.join(ROOT, Data.MENU_ICON_ASSETS[id]), 1, 1, 64, 64, {
    minMargin: 6,
    maxCenterOffsetX: 7,
    maxCenterOffsetY: 7
  })));
  await validateCellMargins(path.join(STARFALL_ROOT, 'items/sheets/ai-items-rate-coupons-sheet.png'), 3, 2, 64, 64, {
    minMargin: 6,
    maxCenterOffsetX: 7,
    maxCenterOffsetY: 8
  });
  await Promise.all(PORTAL_ROWS.map((entry) => validateCellMargins(path.join(STARFALL_ROOT, 'animations/portals', entry.file), 6, 1, 160, 160, {
    minMargin: 8
  })));
  await Promise.all(FX_ROWS.map((entry) => validateCellMargins(path.join(STARFALL_ROOT, 'animations/fx', entry.file), 6, 1, 160, 160, {
    minMargin: 8
  })));
  await Promise.all(Object.values(Data.BASIC_ATTACK_FX_ANIMATION_ASSETS || {}).map((animation) => validateAnimationSheet(animation, { minMargin: 8 })));
  await validateEnemyCombatFxUniqueness();
  await Promise.all(Object.values(Data.SKILL_FX_ANIMATION_ASSETS || {}).map((animation) => validateAnimationSheet(animation, { minMargin: 8 })));
  if (Data.BASIC_ATTACK_FX_ANIMATION_ASSETS && Data.BASIC_ATTACK_FX_ANIMATION_ASSETS.mage) {
    await validateHorizontalProjectileRow(path.join(ROOT, Data.BASIC_ATTACK_FX_ANIMATION_ASSETS.mage.sheet), 'Basic mage', 1);
  }
  await Promise.all(Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {}).map(([skillId, animation]) => {
    const skill = getSkillById(skillId);
    if (!getMageProjectileTypeForSkill(skill)) return Promise.resolve();
    return validateHorizontalProjectileRow(path.join(ROOT, animation.sheet), `Skill ${skillId}`, 1);
  }));
  await Promise.all(Object.values(Data.ENEMY_PROJECTILE_ANIMATION_ASSETS || {}).map((animation) => validateAnimationSheet(animation, { minMargin: 6 })));
  await validateCellMargins(path.join(STARFALL_ROOT, 'environment/structures/town-landmarks.png'), 4, 2, 256, 256, {
    minMargin: 8
  });
  await Promise.all(MAP_DERIVATIONS.map(async (entry) => {
    const metadata = await sharp(path.join(STARFALL_ROOT, entry.target)).metadata();
    if (metadata.width !== 1280 || metadata.height !== 640) {
      throw new Error(`${entry.target} is ${metadata.width}x${metadata.height}; expected 1280x640`);
    }
  }));
}

async function main() {
  const args = process.argv.slice(2);
  const validateOnly = process.argv.includes('--validate');
  if (validateOnly) {
    await validateOutputs();
    console.log('Validated Project Starfall AI visual sweep outputs.');
    return;
  }

  const written = [];
  const selectedSkillIds = getSkillFxSelection(args);
  if (process.argv.includes('--only-skill-fx') || selectedSkillIds.length) {
    const result = await processSemanticSkillFx({ skillIds: selectedSkillIds });
    result.outputPaths.forEach((target) => process.stdout.write(`Processed ${rel(target)}\n`));
    return;
  }

  if (process.argv.includes('--only-mage-projectiles')) {
    await processMageProjectileCombatFx(written);
    written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
    return;
  }

  if (process.argv.includes('--only-basic-mage')) {
    await processBasicMageCombatFx(written);
    written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
    return;
  }

  if (process.argv.includes('--only-enemy-fx')) {
    await processEnemyCombatFx(written, getEnemyFxSelection());
    await validateEnemyCombatFxUniqueness();
    written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
    return;
  }

  await processMenuIcons(written);
  await processRateCoupons(written);
  await processPortals(written);
  await processGlobalFx(written);
  await processDerivedCombatFx(written);
  await processStructures(written);
  await processMaps(written);
  await validateOutputs();
  written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exitCode = 1;
  });
}

module.exports = {
  MAP_DERIVATIONS,
  SOURCE_FILES,
  buildEnemyFxIdentityMap,
  getSkillFxSelection,
  processEnemyCombatFx,
  processDerivedCombatFx,
  processMageProjectileCombatFx,
  validateEnemyCombatFxUniqueness,
  validateOutputs
};
