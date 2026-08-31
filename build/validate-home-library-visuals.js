#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const previewRoot = path.join(root, 'img', 'home-previews');

const HOME_LIBRARY_VISUALS = {
  projects: {
    smartSentence: 'semantic-retrieval',
    chatbotLora: 'grounded-chat',
    shapeClassifier: 'shape-classification',
    ufoDashboard: 'sighting-report',
    covidAnalysis: 'hospital-decision-tree',
    targetEmptyPackage: 'package-anomaly',
    handwritingRating: 'digit-legibility',
    digitGenerator: 'synthetic-digit-generation',
    sheetMusicUpscale: 'music-restoration',
    deliveryTip: 'delivery-tip-inputs',
    retailStore: 'retail-etl',
    pizza: 'pizza-regression-inputs',
    babynames: 'name-preference-learning',
    pizzaDashboard: 'delivery-operations-inputs',
    nonogram: 'nonogram-model',
    website: 'site-accordion'
  },
  tools: {
    'text-compare': 'paired-differences',
    'nbsp-cleaner': 'spacing-cleanup',
    'oxford-comma-checker': 'list-punctuation',
    'point-of-view-checker': 'narrative-person',
    'word-frequency': 'token-frequency',
    'utm-batch-builder': 'parameter-batch',
    'qr-code-generator': 'qr-customization',
    'image-optimizer': 'image-compression',
    'background-remover': 'background-transparency',
    'screen-recorder': 'screen-capture'
  },
  games: {
    'stellar-dogfight': 'space-combat-key-art',
    roulette: 'double-zero-roulette',
    'probability-engine': 'probability-branching',
    'project-starfall': 'fractured-world-key-art',
    stormbreak: 'olympian-storm-key-art',
    'ocean-wave-simulation': 'wave-parameter-study'
  }
};

function previewPath(category, id) {
  return path.join(previewRoot, category, `${id}.webp`);
}

async function validatePreview(category, id) {
  const filePath = previewPath(category, id);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing homepage preview: ${path.relative(root, filePath)}`);
  }

  const metadata = await sharp(filePath).metadata();
  if (metadata.format !== 'webp' || metadata.width !== 640 || metadata.height !== 360) {
    throw new Error(`Homepage preview must be a 640x360 WebP: ${path.relative(root, filePath)}`);
  }

  const buffer = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

async function main() {
  const hashes = [];
  let count = 0;

  for (const [category, visuals] of Object.entries(HOME_LIBRARY_VISUALS)) {
    const categoryDir = path.join(previewRoot, category);
    const actualFiles = fs.readdirSync(categoryDir, { withFileTypes: true })
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .sort();
    const expectedFiles = Object.keys(visuals).map((id) => `${id}.webp`).sort();

    if (JSON.stringify(actualFiles) !== JSON.stringify(expectedFiles)) {
      throw new Error(`Unexpected homepage preview files in ${category}`);
    }

    for (const id of Object.keys(visuals)) {
      hashes.push(await validatePreview(category, id));
      count += 1;
    }
  }

  if (new Set(hashes).size !== hashes.length) {
    throw new Error('Homepage preview images must have unique visual content');
  }

  process.stdout.write(`[home-library-visuals] Validated ${count} generated previews.\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`[home-library-visuals] ${error && error.message ? error.message : error}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  HOME_LIBRARY_VISUALS,
  previewPath,
  validatePreview
};
