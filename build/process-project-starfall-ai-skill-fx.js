#!/usr/bin/env node
'use strict';

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

function isKeyPixel(r, g, b) {
  if (g >= 170 && r <= 95 && b <= 105 && g - Math.max(r, b) >= 80) return true;
  if (g >= 8 && g <= 140 && r <= 90 && b <= 110 && g - Math.max(r, b) >= 3) return true;
  return KEY_COLORS.some((key) => distanceSq({ r, g, b }, key) <= 28 * 28);
}

function isGuideLineCoordinate(value, cellSize, total) {
  if (value <= 1 || value >= total - 2) return true;
  const local = value % cellSize;
  return local <= 1 || local >= cellSize - 2;
}

function isGuidePixel(r, g, b, x, y) {
  if (distanceSq({ r, g, b }, GUIDE_COLOR) > 34 * 34) return false;
  return isGuideLineCoordinate(x, FRAME_SIZE, SHEET_WIDTH) ||
    isGuideLineCoordinate(y, FRAME_SIZE, SHEET_HEIGHT);
}

function despillKeyGreen(data, offset) {
  const r = data[offset];
  const g = data[offset + 1];
  const b = data[offset + 2];
  if (g > Math.max(r, b) + 38 && r < 140 && b < 170) {
    data[offset + 1] = Math.min(g, Math.max(r, b) + 34);
  }
}

async function loadNormalizedSource(sourcePath) {
  return sharp(sourcePath)
    .resize(SHEET_WIDTH, SHEET_HEIGHT, { fit: 'fill', kernel: sharp.kernel.lanczos3 })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
}

async function writeTransparentSheet(sourcePath, outputPath) {
  const { data, info } = await loadNormalizedSource(sourcePath);
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3];
    if (!alpha) continue;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const pixel = offset / 4;
    const x = pixel % info.width;
    const y = Math.floor(pixel / info.width);
    if (isKeyPixel(r, g, b) || isGuidePixel(r, g, b, x, y)) {
      data[offset + 3] = 0;
    } else {
      despillKeyGreen(data, offset);
    }
  }
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  await sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels
    }
  })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(outputPath);
}

async function getPngStats(filePath) {
  const metadata = await sharp(filePath).metadata();
  if (metadata.width !== SHEET_WIDTH || metadata.height !== SHEET_HEIGHT) {
    throw new Error(`${path.relative(ROOT, filePath)} should be ${SHEET_WIDTH}x${SHEET_HEIGHT}, got ${metadata.width}x${metadata.height}`);
  }
  const { data } = await sharp(filePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  let visiblePixels = 0;
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3];
    if (alpha > 0) visiblePixels += 1;
  }
  if (visiblePixels < 1200) {
    throw new Error(`${path.relative(ROOT, filePath)} has too little visible FX art`);
  }
  return { visiblePixels };
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
  SHEET_WIDTH,
  SHEET_HEIGHT,
  processSkillFxSheets
};
