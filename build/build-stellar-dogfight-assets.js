#!/usr/bin/env node
'use strict';

/*
  Generate raster gameplay art for Stellar Dogfight.
  The SVG sources remain the editable masters; PNGs are used by the active
  renderer so fullscreen play does not repeatedly pay SVG filter costs.
*/

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const manifestPath = path.join(root, 'js', 'games', 'stellar-dogfight', 'art.js');
const rasterDir = path.join(root, 'img', 'games', 'stellar-dogfight', 'raster');

function readArtManifest() {
  const source = fs.readFileSync(manifestPath, 'utf8');
  const sandbox = { window: {} };
  vm.runInNewContext(source, sandbox, { filename: manifestPath });
  const manifest = sandbox.window.STELLAR_DOGFIGHT_ART;
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Missing STELLAR_DOGFIGHT_ART manifest');
  }
  return manifest;
}

function collectAssets(manifest) {
  const sections = ['sprites', 'effects', 'backgrounds'];
  return sections.flatMap((section) => Object.entries(manifest[section] || {}).map(([id, definition]) => ({
    id,
    section,
    definition
  }))).filter((asset) => asset.definition && asset.definition.src);
}

function getRasterPath(src) {
  const sourceName = path.basename(src, path.extname(src));
  return path.join(rasterDir, `${sourceName}.png`);
}

async function renderAsset(asset) {
  const sourcePath = path.join(root, asset.definition.src);
  const outputPath = getRasterPath(asset.definition.src);
  const frames = Math.max(1, asset.definition.frames || 1);
  const scale = asset.section === 'backgrounds' ? 1 : 2;
  const targetWidth = Math.max(1, Math.round((asset.definition.width || 128) * frames * scale));
  const targetHeight = Math.max(1, Math.round((asset.definition.height || 128) * scale));

  await sharp(sourcePath, { density: 144 })
    .resize(targetWidth, targetHeight, { fit: 'fill' })
    .png({
      compressionLevel: 9,
      adaptiveFiltering: true
    })
    .toFile(outputPath);

  return path.relative(root, outputPath).replace(/\\/g, '/');
}

async function main() {
  fs.mkdirSync(rasterDir, { recursive: true });
  const manifest = readArtManifest();
  const assets = collectAssets(manifest);
  const outputs = [];

  for (const asset of assets) {
    outputs.push(await renderAsset(asset));
  }

  process.stdout.write(`Generated ${outputs.length} Stellar Dogfight raster asset(s) in ${path.relative(root, rasterDir)}\n`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
