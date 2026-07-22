#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_PATH = path.join(ROOT, 'img/project-starfall/environment/source/starfall-crossing-structures-v1-keyed.png');
const ATLAS_PATH = path.join(ROOT, 'img/project-starfall/environment/structures/town-landmarks.png');
const CELL_SIZE = 256;
const COLUMN_COUNT = 4;
const LEGACY_ROW_COUNT = 2;
const OUTPUT_ROW_COUNT = 3;
const MAGENTA_NEAR_DISTANCE = 24;
const MAGENTA_OPAQUE_DISTANCE = 118;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

async function removeMagentaKey(sourcePath) {
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  for (let offset = 0; offset < data.length; offset += 4) {
    const red = data[offset];
    const green = data[offset + 1];
    const blue = data[offset + 2];
    const distance = Math.sqrt((255 - red) ** 2 + green ** 2 + (255 - blue) ** 2);
    const distanceAlpha = clamp((distance - MAGENTA_NEAR_DISTANCE) / (MAGENTA_OPAQUE_DISTANCE - MAGENTA_NEAR_DISTANCE), 0, 1);
    const magentaMinimum = Math.min(red, blue);
    const magentaSimilarity = magentaMinimum > 0 ? 1 - Math.abs(red - blue) / Math.max(red, blue) : 0;
    const magentaDominance = magentaMinimum - green;
    const spillAlpha = magentaSimilarity > 0.68 && magentaMinimum > 15 && magentaDominance > 18
      ? clamp((38 - magentaDominance) / 20, 0, 1)
      : 1;
    const alpha = Math.min(distanceAlpha, spillAlpha);
    if (alpha <= 0) {
      data[offset] = 0;
      data[offset + 1] = 0;
      data[offset + 2] = 0;
      data[offset + 3] = 0;
      continue;
    }
    if (alpha < 1) {
      data[offset] = clamp(Math.round((red - 255 * (1 - alpha)) / alpha), 0, 255);
      data[offset + 1] = clamp(Math.round(green / alpha), 0, 255);
      data[offset + 2] = clamp(Math.round((blue - 255 * (1 - alpha)) / alpha), 0, 255);
    }
    data[offset + 3] = Math.round(alpha * 255);
  }
  return sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  }).png().toBuffer();
}

async function createStructureCell(keyedBuffer, index, sourceWidth, sourceHeight) {
  const left = Math.round(index * sourceWidth / COLUMN_COUNT);
  const right = Math.round((index + 1) * sourceWidth / COLUMN_COUNT);
  const cropped = await sharp(keyedBuffer)
    .extract({ left, top: 0, width: right - left, height: sourceHeight })
    .png()
    .toBuffer();
  const trimmed = await sharp(cropped)
    .trim({ background: { r: 0, g: 0, b: 0, alpha: 0 }, threshold: 8 })
    .png()
    .toBuffer();
  return sharp(trimmed)
    .resize(CELL_SIZE - 16, CELL_SIZE - 12, {
      fit: 'contain',
      position: 'bottom',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .extend({
      top: 6,
      bottom: 6,
      left: 8,
      right: 8,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .png()
    .toBuffer();
}

async function buildAtlas() {
  if (!fs.existsSync(SOURCE_PATH)) throw new Error(`Missing keyed Crossing structure source: ${SOURCE_PATH}`);
  if (!fs.existsSync(ATLAS_PATH)) throw new Error(`Missing town landmark atlas: ${ATLAS_PATH}`);
  const sourceMetadata = await sharp(SOURCE_PATH).metadata();
  const keyedBuffer = await removeMagentaKey(SOURCE_PATH);
  const legacyBuffer = await sharp(ATLAS_PATH)
    .extract({ left: 0, top: 0, width: CELL_SIZE * COLUMN_COUNT, height: CELL_SIZE * LEGACY_ROW_COUNT })
    .png()
    .toBuffer();
  const cells = [];
  for (let index = 0; index < COLUMN_COUNT; index += 1) {
    cells.push(await createStructureCell(keyedBuffer, index, sourceMetadata.width, sourceMetadata.height));
  }
  const composites = [{ input: legacyBuffer, left: 0, top: 0 }];
  cells.forEach((input, index) => {
    composites.push({ input, left: index * CELL_SIZE, top: LEGACY_ROW_COUNT * CELL_SIZE });
  });
  await sharp({
    create: {
      width: CELL_SIZE * COLUMN_COUNT,
      height: CELL_SIZE * OUTPUT_ROW_COUNT,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(ATLAS_PATH);
  return ATLAS_PATH;
}

async function validateAtlas() {
  const image = sharp(ATLAS_PATH).ensureAlpha();
  const metadata = await image.metadata();
  const expectedWidth = CELL_SIZE * COLUMN_COUNT;
  const expectedHeight = CELL_SIZE * OUTPUT_ROW_COUNT;
  if (metadata.width !== expectedWidth || metadata.height !== expectedHeight) {
    throw new Error(`Town landmarks should be ${expectedWidth}x${expectedHeight}, got ${metadata.width}x${metadata.height}`);
  }
  const { data, info } = await image.raw().toBuffer({ resolveWithObject: true });
  for (let cellIndex = 8; cellIndex <= 11; cellIndex += 1) {
    const cellX = (cellIndex % COLUMN_COUNT) * CELL_SIZE;
    const cellY = Math.floor(cellIndex / COLUMN_COUNT) * CELL_SIZE;
    let opaquePixels = 0;
    for (let y = cellY; y < cellY + CELL_SIZE; y += 1) {
      for (let x = cellX; x < cellX + CELL_SIZE; x += 1) {
        if (data[(y * info.width + x) * 4 + 3] > 24) opaquePixels += 1;
      }
    }
    if (opaquePixels < 1800) throw new Error(`Town landmark cell ${cellIndex} is unexpectedly empty (${opaquePixels} opaque pixels)`);
  }
  return { width: metadata.width, height: metadata.height, cells: 12 };
}

async function main() {
  if (!process.argv.includes('--validate')) await buildAtlas();
  const result = await validateAtlas();
  process.stdout.write(`Validated Crossing town landmarks ${result.width}x${result.height} (${result.cells} cells)\n`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  removeMagentaKey,
  buildAtlas,
  validateAtlas
};
