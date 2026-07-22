#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/maps/source/field');
const SAFE_ZONE_SOURCE_DIR = path.join(ROOT, 'img/project-starfall/maps/source/safe-zones');
const BOSS_SOURCE_DIR = path.join(ROOT, 'img/project-starfall/maps/source/boss');
const MAP_DIR = path.join(ROOT, 'img/project-starfall/maps');
const TILE_WIDTH = 1280;
const TILE_HEIGHT = 640;
const PANORAMA_WIDTH = 2560;
const WRAP_OFFSET_X = TILE_WIDTH / 2;
const EDGE_FEATHER_PX = 360;
const EDGE_LOCK_COLUMNS = 2;
const WEBP_QUALITY = 88;
const EDGE_SCORE_LIMIT = 44;
const MIDPOINT_SCORE_LIMIT = 64;
const HALF_REPEAT_MIN_DELTA = 4;
const PANORAMA_OUTPUTS = Object.freeze(new Set(['starfall-crossing.webp', 'eclipse-throne-v2.webp']));

const SOURCE_BACKED_MAPS = Object.freeze([
  ['starfall-crossing.webp', 'starfall-crossing-fractured-observatory-v1.png'],
  ['greenroot-meadow.webp', 'greenroot-meadow.png'],
  ['thornpath-thicket.webp', 'thornpath-thicket.png'],
  ['bramble-depths.webp', 'bramble-depths.png'],
  ['rustcoil-ruins.webp', 'rustcoil-ruins.png'],
  ['gearworks-vault.webp', 'gearworks-vault.png'],
  ['cinder-hollow.webp', 'cinder-hollow.png'],
  ['emberjaw-lair.webp', 'emberjaw-lair.png'],
  ['bandit-ridge-camp.webp', 'bandit-ridge-camp.png'],
  ['oreback-quarry.webp', 'oreback-quarry.png'],
  ['ashglass-pass.webp', 'ashglass-pass.png'],
  ['stormbreak-cliffs.webp', 'stormbreak-cliffs.png'],
  ['astral-archive.webp', 'astral-archive.png'],
  ['eclipse-frontier.webp', 'eclipse-frontier.png'],
  ['eclipse-throne-v2.webp', 'eclipse-throne-totality-v1.png', BOSS_SOURCE_DIR],
  ['endless-rift.webp', 'endless-rift.png'],
  ['rustcoil-outpost.webp', 'rustcoil-outpost.png', SAFE_ZONE_SOURCE_DIR],
  ['cinder-refuge.webp', 'cinder-refuge.png', SAFE_ZONE_SOURCE_DIR],
  ['frostfen-camp.webp', 'frostfen-camp.png', SAFE_ZONE_SOURCE_DIR],
  ['stormbreak-haven.webp', 'stormbreak-haven.png', SAFE_ZONE_SOURCE_DIR],
  ['astral-observatory.webp', 'astral-observatory.png', SAFE_ZONE_SOURCE_DIR]
]);

function getRequestedTargets() {
  const onlyIndex = process.argv.indexOf('--only');
  if (onlyIndex < 0) return null;
  const requested = String(process.argv[onlyIndex + 1] || '')
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return requested.length ? new Set(requested) : null;
}

function smoothStep(value) {
  const t = Math.max(0, Math.min(1, Number(value) || 0));
  return t * t * (3 - 2 * t);
}

async function loadSourceTile(sourcePath) {
  return sharp(sourcePath)
    .resize(TILE_WIDTH, TILE_HEIGHT, { fit: 'cover', position: 'center' })
    .ensureAlpha()
    .raw()
    .toBuffer();
}

function makeSeamlessWrapTile(sourcePixels) {
  const channels = 4;
  const output = Buffer.alloc(TILE_WIDTH * TILE_HEIGHT * channels);
  for (let y = 0; y < TILE_HEIGHT; y += 1) {
    for (let x = 0; x < TILE_WIDTH; x += 1) {
      const wrappedX = (x + WRAP_OFFSET_X) % TILE_WIDTH;
      const original = (y * TILE_WIDTH + x) * channels;
      const wrapped = (y * TILE_WIDTH + wrappedX) * channels;
      const target = (y * TILE_WIDTH + x) * channels;
      const edgeDistance = Math.min(x, TILE_WIDTH - 1 - x);
      const originalWeight = smoothStep(edgeDistance / EDGE_FEATHER_PX);
      const wrappedWeight = 1 - originalWeight;
      output[target] = Math.round(sourcePixels[wrapped] * wrappedWeight + sourcePixels[original] * originalWeight);
      output[target + 1] = Math.round(sourcePixels[wrapped + 1] * wrappedWeight + sourcePixels[original + 1] * originalWeight);
      output[target + 2] = Math.round(sourcePixels[wrapped + 2] * wrappedWeight + sourcePixels[original + 2] * originalWeight);
      output[target + 3] = 255;
    }
  }
  return output;
}

function lockWrapEdges(pixels) {
  const channels = 4;
  for (let y = 0; y < TILE_HEIGHT; y += 1) {
    for (let x = 0; x < EDGE_LOCK_COLUMNS; x += 1) {
      const left = (y * TILE_WIDTH + x) * channels;
      const right = (y * TILE_WIDTH + TILE_WIDTH - 1 - x) * channels;
      for (let channel = 0; channel < 3; channel += 1) {
        const value = Math.round((pixels[left + channel] + pixels[right + channel]) / 2);
        pixels[left + channel] = value;
        pixels[right + channel] = value;
      }
      pixels[left + 3] = 255;
      pixels[right + 3] = 255;
    }
  }
  return pixels;
}

async function writeMapBackground(outputName, sourceName, sourceDir) {
  const sourcePath = path.join(sourceDir || SOURCE_DIR, sourceName);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing Project Starfall map source image: ${sourcePath}`);
  }
  fs.mkdirSync(MAP_DIR, { recursive: true });
  const outputPath = path.join(MAP_DIR, outputName);
  if (PANORAMA_OUTPUTS.has(outputName)) {
    await sharp(sourcePath)
      .resize(PANORAMA_WIDTH, TILE_HEIGHT, { fit: 'cover', position: 'center' })
      .webp({ quality: WEBP_QUALITY, effort: 4 })
      .toFile(outputPath);
    return path.relative(ROOT, outputPath).replace(/\\/g, '/');
  }
  const sourcePixels = await loadSourceTile(sourcePath);
  const tilePixels = lockWrapEdges(makeSeamlessWrapTile(sourcePixels));
  await sharp(tilePixels, {
    raw: { width: TILE_WIDTH, height: TILE_HEIGHT, channels: 4 }
  }).webp({
    quality: WEBP_QUALITY,
    effort: 4
  }).toFile(outputPath);
  return path.relative(ROOT, outputPath).replace(/\\/g, '/');
}

function seamScore(pixels, width, height, xA, xB) {
  const channels = 4;
  let total = 0;
  for (let y = 0; y < height; y += 1) {
    const a = (y * width + xA) * channels;
    const b = (y * width + xB) * channels;
    total += Math.abs(pixels[a] - pixels[b]);
    total += Math.abs(pixels[a + 1] - pixels[b + 1]);
    total += Math.abs(pixels[a + 2] - pixels[b + 2]);
  }
  return total / height;
}

function halfRepeatDelta(pixels, width, height) {
  const channels = 4;
  const half = Math.floor(width / 2);
  let total = 0;
  let count = 0;
  for (let y = 0; y < height; y += 4) {
    for (let x = 0; x < half; x += 4) {
      const a = (y * width + x) * channels;
      const b = (y * width + x + half) * channels;
      total += Math.abs(pixels[a] - pixels[b]);
      total += Math.abs(pixels[a + 1] - pixels[b + 1]);
      total += Math.abs(pixels[a + 2] - pixels[b + 2]);
      count += 3;
    }
  }
  return count ? total / count : 0;
}

async function validateMapBackground(outputName) {
  const outputPath = path.join(MAP_DIR, outputName);
  if (!fs.existsSync(outputPath)) throw new Error(`Missing generated map background: ${outputPath}`);
  const image = sharp(outputPath).ensureAlpha();
  const metadata = await image.metadata();
  const panorama = PANORAMA_OUTPUTS.has(outputName);
  const expectedWidth = panorama ? PANORAMA_WIDTH : TILE_WIDTH;
  if (metadata.width !== expectedWidth || metadata.height !== TILE_HEIGHT) {
    throw new Error(`${outputName} should be ${expectedWidth}x${TILE_HEIGHT}, got ${metadata.width}x${metadata.height}`);
  }
  if (panorama) return { outputName, panorama: true, width: expectedWidth, height: TILE_HEIGHT };
  const pixels = await image.raw().toBuffer();
  const edge = seamScore(pixels, TILE_WIDTH, TILE_HEIGHT, 0, TILE_WIDTH - 1);
  const midpoint = seamScore(pixels, TILE_WIDTH, TILE_HEIGHT, WRAP_OFFSET_X - 1, WRAP_OFFSET_X);
  const halfDelta = halfRepeatDelta(pixels, TILE_WIDTH, TILE_HEIGHT);
  if (edge > EDGE_SCORE_LIMIT) {
    throw new Error(`${outputName} left/right seam score ${edge.toFixed(1)} exceeds ${EDGE_SCORE_LIMIT}`);
  }
  if (midpoint > MIDPOINT_SCORE_LIMIT) {
    throw new Error(`${outputName} midpoint seam score ${midpoint.toFixed(1)} exceeds ${MIDPOINT_SCORE_LIMIT}`);
  }
  if (halfDelta < HALF_REPEAT_MIN_DELTA) {
    throw new Error(`${outputName} still resembles a repeated half-tile; half delta ${halfDelta.toFixed(1)} below ${HALF_REPEAT_MIN_DELTA}`);
  }
  return { outputName, edge, midpoint, halfDelta };
}

async function validateAllMapBackgrounds() {
  const results = [];
  for (const [outputName] of SOURCE_BACKED_MAPS) {
    results.push(await validateMapBackground(outputName));
  }
  return results;
}

async function main() {
  if (process.argv.includes('--validate')) {
    const results = await validateAllMapBackgrounds();
    results.forEach((result) => {
      if (result.panorama) {
        process.stdout.write(`Validated ${result.outputName} panorama=${result.width}x${result.height}\n`);
        return;
      }
      process.stdout.write(`Validated ${result.outputName} edge=${result.edge.toFixed(1)} midpoint=${result.midpoint.toFixed(1)} halfDelta=${result.halfDelta.toFixed(1)}\n`);
    });
    return;
  }
  const requestedTargets = getRequestedTargets();
  const generated = [];
  for (const [outputName, sourceName, sourceDir] of SOURCE_BACKED_MAPS) {
    const targetId = outputName.replace(/\.webp$/i, '');
    if (requestedTargets && !requestedTargets.has(targetId) && !requestedTargets.has(outputName.toLowerCase())) continue;
    generated.push(await writeMapBackground(outputName, sourceName, sourceDir));
  }
  generated.forEach((file) => process.stdout.write(`Generated ${file}\n`));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

module.exports = {
  BOSS_SOURCE_DIR,
  SOURCE_BACKED_MAPS,
  PANORAMA_OUTPUTS,
  makeSeamlessWrapTile,
  validateAllMapBackgrounds
};
