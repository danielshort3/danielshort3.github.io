#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const Data = require('../js/games/project-starfall/project-starfall-data.js');

const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/animations/combat-fx/skills/source');
const OUTPUT_DIR = path.join(ROOT, 'img/project-starfall/animations/combat-fx/skills');
const FRAME_SIZE = 160;
const COLUMNS = 6;
const ROWS = 4;
const SHEET_WIDTH = FRAME_SIZE * COLUMNS;
const SHEET_HEIGHT = FRAME_SIZE * ROWS;
const SAFETY_GUTTER = 8;
const KEY_COLORS = Object.freeze([
  Object.freeze({ r: 0, g: 255, b: 0 }),
  Object.freeze({ r: 255, g: 0, b: 255 })
]);
const GUIDE_COLOR = Object.freeze({ r: 0, g: 255, b: 255 });

function usage() {
  return [
    'Usage: node build/process-project-starfall-ai-skill-fx.js [--validate] [--strict] [--only <skill-id>]',
    'Processes image-generated skill FX sheets from img/project-starfall/animations/combat-fx/skills/source/.'
  ].join('\n');
}

function getArgValue(args, flag) {
  const equalsArg = args.find((arg) => String(arg || '').startsWith(`${flag}=`));
  if (equalsArg) return equalsArg.split('=').slice(1).join('=').trim();
  const index = args.indexOf(flag);
  return index >= 0 ? String(args[index + 1] || '').trim() : '';
}

function hasFlag(args, flag) {
  return args.includes(flag);
}

function normalizeId(value) {
  return String(value || '')
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/_/g, '-')
    .replace(/[^a-z0-9-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
}

function getSkillEntries(onlyId) {
  const requestedId = normalizeId(onlyId);
  return Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {})
    .map(([skillId, animation]) => {
      const outputPath = path.join(ROOT, animation.sheet);
      const fileName = path.basename(outputPath);
      return {
        skillId,
        fileId: path.basename(fileName, '.png'),
        sourcePath: path.join(SOURCE_DIR, fileName),
        outputPath
      };
    })
    .filter((entry) => !requestedId || normalizeId(entry.skillId) === requestedId || normalizeId(entry.fileId) === requestedId);
}

function distanceSq(color, key) {
  const dr = color.r - key.r;
  const dg = color.g - key.g;
  const db = color.b - key.b;
  return dr * dr + dg * dg + db * db;
}

/**
 * Source-background classifier. The generated source contract forbids green in
 * the effect, so a green-dominant pixel is safe to key. Cyan/teal effects are
 * retained because the green and blue channels remain close together.
 */
function isKeyPixel(r, g, b) {
  if (KEY_COLORS.some((key) => distanceSq({ r, g, b }, key) <= 48 * 48)) return true;
  if (g >= 115 && g - r >= 45 && g - b >= 45) return true;
  return r >= 135 && b >= 135 && r - g >= 60 && b - g >= 60;
}

/**
 * Runtime-output validator for the actual chroma colors. This is deliberately
 * narrower than the source-background classifier: archer, trapper, and beast
 * effects legitimately use natural greens that must remain visible.
 */
function isOutputKeyPixel(r, g, b) {
  return KEY_COLORS.some((key) => distanceSq({ r, g, b }, key) <= 48 * 48);
}

function isGuideColor(r, g, b) {
  return distanceSq({ r, g, b }, GUIDE_COLOR) <= 42 * 42;
}

function isGuideLineCoordinate(value, cellSize, total) {
  if (value <= 1 || value >= total - 2) return true;
  const local = value % cellSize;
  return local <= 1 || local >= cellSize - 2;
}

function isGuidePixel(r, g, b, x, y, width = SHEET_WIDTH, height = SHEET_HEIGHT) {
  if (!isGuideColor(r, g, b)) return false;
  return isGuideLineCoordinate(x, FRAME_SIZE, width) ||
    isGuideLineCoordinate(y, FRAME_SIZE, height);
}

function clearPixel(data, offset) {
  data[offset] = 0;
  data[offset + 1] = 0;
  data[offset + 2] = 0;
  data[offset + 3] = 0;
}

function clearTransparentRgb(data) {
  for (let offset = 0; offset < data.length; offset += 4) {
    if (data[offset + 3] === 0) clearPixel(data, offset);
  }
  return data;
}

function despillKeyGreen(data, offset) {
  const r = data[offset];
  const g = data[offset + 1];
  const b = data[offset + 2];
  if (g > Math.max(r, b) + 30 && r < 170 && b < 185) {
    data[offset + 1] = Math.min(g, Math.max(r, b) + 26);
  }
}

function getVisibleBounds(data, width = FRAME_SIZE, height = FRAME_SIZE) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let visiblePixels = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (data[offset + 3] === 0) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      visiblePixels += 1;
    }
  }
  if (!visiblePixels) return null;
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    visiblePixels
  };
}

function copyFrameFromSheet(data, sheetWidth, row, column) {
  const frame = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceStart = (((row * FRAME_SIZE + y) * sheetWidth) + column * FRAME_SIZE) * 4;
    const destinationStart = y * FRAME_SIZE * 4;
    data.copy(frame, destinationStart, sourceStart, sourceStart + FRAME_SIZE * 4);
  }
  return frame;
}

function copyFrameToSheet(frame, data, sheetWidth, row, column) {
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceStart = y * FRAME_SIZE * 4;
    const destinationStart = (((row * FRAME_SIZE + y) * sheetWidth) + column * FRAME_SIZE) * 4;
    frame.copy(data, destinationStart, sourceStart, sourceStart + FRAME_SIZE * 4);
  }
}

function getConnectedBackgroundMask(data) {
  const pixelCount = FRAME_SIZE * FRAME_SIZE;
  const queued = new Uint8Array(pixelCount);
  const mask = new Uint8Array(pixelCount);
  const queue = new Int32Array(pixelCount);
  let head = 0;
  let tail = 0;

  function enqueue(x, y) {
    const index = y * FRAME_SIZE + x;
    if (queued[index]) return;
    const offset = index * 4;
    if (data[offset + 3] === 0 || isKeyPixel(data[offset], data[offset + 1], data[offset + 2])) {
      queued[index] = 1;
      queue[tail] = index;
      tail += 1;
    }
  }

  for (let x = 0; x < FRAME_SIZE; x += 1) {
    enqueue(x, 0);
    enqueue(x, FRAME_SIZE - 1);
  }
  for (let y = 1; y < FRAME_SIZE - 1; y += 1) {
    enqueue(0, y);
    enqueue(FRAME_SIZE - 1, y);
  }

  while (head < tail) {
    const index = queue[head];
    head += 1;
    if (mask[index]) continue;
    mask[index] = 1;
    const x = index % FRAME_SIZE;
    const y = Math.floor(index / FRAME_SIZE);
    if (x > 0) enqueue(x - 1, y);
    if (x < FRAME_SIZE - 1) enqueue(x + 1, y);
    if (y > 0) enqueue(x, y - 1);
    if (y < FRAME_SIZE - 1) enqueue(x, y + 1);
  }
  return mask;
}

function removeFrameBackground(data) {
  const backgroundMask = getConnectedBackgroundMask(data);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const pixel = y * FRAME_SIZE + x;
      const offset = pixel * 4;
      const alpha = data[offset + 3];
      if (!alpha) {
        clearPixel(data, offset);
        continue;
      }
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      if (backgroundMask[pixel] || isKeyPixel(r, g, b) || isGuidePixel(r, g, b, x, y, FRAME_SIZE, FRAME_SIZE)) {
        clearPixel(data, offset);
      } else {
        despillKeyGreen(data, offset);
      }
    }
  }
  return data;
}

function isGroundAlignedFrame(row, bounds) {
  if (row === ROWS - 1) return true;
  return row === ROWS - 2 && bounds.width >= bounds.height * 1.45 && bounds.maxY >= FRAME_SIZE / 2;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

async function fitFrameWithinGutter(frame, row) {
  const bounds = getVisibleBounds(frame);
  if (!bounds) return frame;
  const innerSize = FRAME_SIZE - SAFETY_GUTTER * 2;
  const scale = Math.min(1, innerSize / bounds.width, innerSize / bounds.height);
  const targetWidth = Math.max(1, Math.min(innerSize, Math.floor(bounds.width * scale)));
  const targetHeight = Math.max(1, Math.min(innerSize, Math.floor(bounds.height * scale)));

  const crop = Buffer.alloc(bounds.width * bounds.height * 4);
  for (let y = 0; y < bounds.height; y += 1) {
    const sourceStart = ((bounds.minY + y) * FRAME_SIZE + bounds.minX) * 4;
    const destinationStart = y * bounds.width * 4;
    frame.copy(crop, destinationStart, sourceStart, sourceStart + bounds.width * 4);
  }

  let art = crop;
  if (targetWidth !== bounds.width || targetHeight !== bounds.height) {
    art = (await sharp(crop, {
      raw: { width: bounds.width, height: bounds.height, channels: 4 }
    })
      .resize(targetWidth, targetHeight, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
      .raw()
      .toBuffer());
  }
  clearTransparentRgb(art);

  const centerX = (bounds.minX + bounds.maxX + 1) / 2;
  const centerY = (bounds.minY + bounds.maxY + 1) / 2;
  const maximumLeft = FRAME_SIZE - SAFETY_GUTTER - targetWidth;
  const maximumTop = FRAME_SIZE - SAFETY_GUTTER - targetHeight;
  const left = clamp(Math.round(centerX - targetWidth / 2), SAFETY_GUTTER, maximumLeft);
  const top = isGroundAlignedFrame(row, bounds)
    ? maximumTop
    : clamp(Math.round(centerY - targetHeight / 2), SAFETY_GUTTER, maximumTop);
  const output = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceStart = y * targetWidth * 4;
    const destinationStart = ((top + y) * FRAME_SIZE + left) * 4;
    art.copy(output, destinationStart, sourceStart, sourceStart + targetWidth * 4);
  }
  return output;
}

async function processFrame(frame, row) {
  removeFrameBackground(frame);
  const fitted = await fitFrameWithinGutter(frame, row);
  // Lanczos can reintroduce one-alpha chroma samples along a resized edge.
  // Key those interpolation samples, then re-register once without scaling so
  // ground-aligned rows retain a stable bottom-center anchor.
  removeFrameBackground(fitted);
  const registered = await fitFrameWithinGutter(fitted, row);
  clearTransparentRgb(registered);
  return registered;
}

async function loadNormalizedSource(sourcePath) {
  return sharp(sourcePath)
    .resize(SHEET_WIDTH, SHEET_HEIGHT, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
}

function validateSheetData(data, width, height, fileLabel = 'skill FX sheet') {
  if (width !== SHEET_WIDTH || height !== SHEET_HEIGHT) {
    throw new Error(`${fileLabel} should be ${SHEET_WIDTH}x${SHEET_HEIGHT}, got ${width}x${height}`);
  }
  if (data.length !== width * height * 4) {
    throw new Error(`${fileLabel} has an unexpected raw pixel buffer size`);
  }

  let visiblePixels = 0;
  let transparentRgbPixels = 0;
  let keyResiduePixels = 0;
  const frameHashes = new Map();
  const frames = [];

  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLUMNS; column += 1) {
      const frame = copyFrameFromSheet(data, width, row, column);
      const bounds = getVisibleBounds(frame);
      const frameName = `row ${row + 1}, frame ${column + 1}`;
      if (!bounds) throw new Error(`${fileLabel} has an empty ${frameName}`);
      if (bounds.minX < SAFETY_GUTTER || bounds.minY < SAFETY_GUTTER ||
          bounds.maxX >= FRAME_SIZE - SAFETY_GUTTER || bounds.maxY >= FRAME_SIZE - SAFETY_GUTTER) {
        throw new Error(`${fileLabel} ${frameName} violates the ${SAFETY_GUTTER}px safety gutter ` +
          `(${bounds.minX},${bounds.minY})-(${bounds.maxX},${bounds.maxY})`);
      }
      const hash = crypto.createHash('sha256').update(frame).digest('hex');
      if (frameHashes.has(hash)) {
        throw new Error(`${fileLabel} has exact duplicate frames: ${frameHashes.get(hash)} and ${frameName}`);
      }
      frameHashes.set(hash, frameName);
      visiblePixels += bounds.visiblePixels;
      frames.push({ row, column, bounds, hash });
    }
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const alpha = data[offset + 3];
      if (!alpha && (r || g || b)) transparentRgbPixels += 1;
      if ((alpha || r || g || b) &&
          (isOutputKeyPixel(r, g, b) || isGuidePixel(r, g, b, x, y, width, height))) {
        keyResiduePixels += 1;
      }
    }
  }

  if (keyResiduePixels) {
    throw new Error(`${fileLabel} contains ${keyResiduePixels} visible or hidden chroma-key/guide pixel(s)`);
  }
  if (transparentRgbPixels) {
    throw new Error(`${fileLabel} contains ${transparentRgbPixels} transparent pixel(s) with hidden RGB data`);
  }
  if (visiblePixels < 1200) {
    throw new Error(`${fileLabel} has too little visible FX art`);
  }
  return { visiblePixels, transparentRgbPixels, keyResiduePixels, frames };
}

async function writeTransparentSheet(sourcePath, outputPath) {
  const { data, info } = await loadNormalizedSource(sourcePath);
  const output = Buffer.alloc(SHEET_WIDTH * SHEET_HEIGHT * 4);
  for (let row = 0; row < ROWS; row += 1) {
    for (let column = 0; column < COLUMNS; column += 1) {
      const sourceFrame = copyFrameFromSheet(data, info.width, row, column);
      const frame = await processFrame(sourceFrame, row);
      copyFrameToSheet(frame, output, SHEET_WIDTH, row, column);
    }
  }
  clearTransparentRgb(output);
  validateSheetData(output, SHEET_WIDTH, SHEET_HEIGHT, path.relative(ROOT, outputPath));
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  await sharp(output, {
    raw: { width: SHEET_WIDTH, height: SHEET_HEIGHT, channels: 4 }
  })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(outputPath);
}

async function getPngStats(filePath) {
  const metadata = await sharp(filePath).metadata();
  if (metadata.width !== SHEET_WIDTH || metadata.height !== SHEET_HEIGHT) {
    throw new Error(`${path.relative(ROOT, filePath)} should be ${SHEET_WIDTH}x${SHEET_HEIGHT}, got ${metadata.width}x${metadata.height}`);
  }
  const { data, info } = await sharp(filePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return validateSheetData(data, info.width, info.height, path.relative(ROOT, filePath));
}

async function processSkillFxSheets(options = {}) {
  const entries = getSkillEntries(options.only);
  if (!entries.length) {
    throw new Error(options.only ? `Unknown skill FX id: ${options.only}` : 'No skill FX animation assets found.');
  }

  let processed = 0;
  let validated = 0;
  let missingSources = 0;
  const missing = [];

  for (const entry of entries) {
    if (!fs.existsSync(entry.sourcePath)) {
      missingSources += 1;
      missing.push(path.relative(ROOT, entry.sourcePath));
      if (options.strict) continue;
    } else if (!options.validateOnly) {
      await writeTransparentSheet(entry.sourcePath, entry.outputPath);
      processed += 1;
    }
    await getPngStats(entry.outputPath);
    validated += 1;
  }

  if (missingSources && options.strict) {
    throw new Error(`Missing ${missingSources} source-backed skill FX sheet(s):\n${missing.join('\n')}`);
  }

  return {
    processed,
    validated,
    missingSources,
    total: entries.length
  };
}

async function main() {
  const args = process.argv.slice(2);
  if (hasFlag(args, '--help') || hasFlag(args, '-h')) {
    console.log(usage());
    return;
  }
  const result = await processSkillFxSheets({
    validateOnly: hasFlag(args, '--validate'),
    strict: hasFlag(args, '--strict'),
    only: getArgValue(args, '--only')
  });
  const action = hasFlag(args, '--validate') ? 'Validated' : 'Processed';
  console.log(`${action} ${result.validated} Project Starfall skill FX sheet(s); ${result.processed} source-backed sheet(s) written, ${result.missingSources} existing runtime sheet(s) reused.`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error.message || error);
    process.exitCode = 1;
  });
}

module.exports = {
  SOURCE_DIR,
  OUTPUT_DIR,
  FRAME_SIZE,
  COLUMNS,
  ROWS,
  SHEET_WIDTH,
  SHEET_HEIGHT,
  SAFETY_GUTTER,
  clearTransparentRgb,
  fitFrameWithinGutter,
  getPngStats,
  getSkillEntries,
  getVisibleBounds,
  isGuidePixel,
  isGroundAlignedFrame,
  isKeyPixel,
  isOutputKeyPixel,
  processFrame,
  processSkillFxSheets,
  removeFrameBackground,
  validateSheetData,
  writeTransparentSheet
};
