#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const previewRoot = path.join(root, 'img', 'home-previews');
const publicPreviewRoot = path.join(root, 'public', 'img', 'home-previews');

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

function normalizedRelativePath(value) {
  return String(value || '').split(path.sep).join('/');
}

function expectedPreviewTree() {
  const directories = Object.keys(HOME_LIBRARY_VISUALS).sort();
  const files = directories.flatMap((category) => Object.keys(HOME_LIBRARY_VISUALS[category])
    .map((id) => `${category}/${id}.webp`))
    .sort();
  return { directories, files };
}

function listPreviewTree(baseDir) {
  if (!fs.existsSync(baseDir)) {
    throw new Error(`Missing homepage preview directory: ${path.relative(root, baseDir)}`);
  }

  const directories = [];
  const files = [];
  const pending = [{ absolutePath: baseDir, relativePath: '' }];

  while (pending.length) {
    const current = pending.pop();
    const entries = fs.readdirSync(current.absolutePath, { withFileTypes: true });
    entries.forEach((entry) => {
      const relativePath = normalizedRelativePath(path.join(current.relativePath, entry.name));
      const absolutePath = path.join(current.absolutePath, entry.name);
      if (entry.isDirectory()) {
        directories.push(relativePath);
        pending.push({ absolutePath, relativePath });
        return;
      }
      if (entry.isFile()) files.push(relativePath);
    });
  }

  return {
    directories: directories.sort(),
    files: files.sort()
  };
}

function validateCatalogMappings() {
  const dataPath = path.join(root, 'js', 'home', 'home-library-data.js');
  delete require.cache[require.resolve(dataPath)];
  const libraryData = require(dataPath);

  Object.entries(HOME_LIBRARY_VISUALS).forEach(([category, visuals]) => {
    const items = libraryData[category]?.items || [];
    const expectedIds = Object.keys(visuals);
    const actualIds = items.map((item) => item.id);
    if (JSON.stringify(actualIds) !== JSON.stringify(expectedIds)) {
      throw new Error(`Homepage ${category} preview manifest is out of sync with the generated catalog`);
    }
    items.forEach((item) => {
      const expectedImage = `/img/home-previews/${category}/${item.id}.webp`;
      if (item.image !== expectedImage) {
        throw new Error(`Unexpected homepage preview mapping for ${category}/${item.id}`);
      }
    });
  });
}

async function validatePreviewAt(baseDir, category, id) {
  const filePath = path.join(baseDir, category, `${id}.webp`);
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

async function validatePreview(category, id) {
  return validatePreviewAt(previewRoot, category, id);
}

async function validatePreviewTree(baseDir, label) {
  const expectedTree = expectedPreviewTree();
  const actualTree = listPreviewTree(baseDir);
  if (JSON.stringify(actualTree.directories) !== JSON.stringify(expectedTree.directories) ||
    JSON.stringify(actualTree.files) !== JSON.stringify(expectedTree.files)) {
    throw new Error(`Unexpected homepage preview tree in ${label}`);
  }

  const hashes = new Map();
  for (const [category, visuals] of Object.entries(HOME_LIBRARY_VISUALS)) {
    for (const id of Object.keys(visuals)) {
      const relativePath = `${category}/${id}.webp`;
      hashes.set(relativePath, await validatePreviewAt(baseDir, category, id));
    }
  }

  if (new Set(hashes.values()).size !== hashes.size) {
    throw new Error('Homepage preview images must have unique visual content');
  }

  return hashes;
}

function validateMatchingHashes(sourceHashes, deployedHashes) {
  for (const [relativePath, sourceHash] of sourceHashes) {
    if (deployedHashes.get(relativePath) !== sourceHash) {
      throw new Error(`Deployed homepage preview differs from source: ${relativePath}`);
    }
  }
}

async function main() {
  const validatePublic = process.argv.slice(2).includes('--public');
  validateCatalogMappings();
  const sourceHashes = await validatePreviewTree(previewRoot, 'img/home-previews');

  if (validatePublic) {
    const deployedHashes = await validatePreviewTree(publicPreviewRoot, 'public/img/home-previews');
    validateMatchingHashes(sourceHashes, deployedHashes);
    process.stdout.write(`[home-library-visuals] Validated ${sourceHashes.size} source and deployed previews.\n`);
    return;
  }

  process.stdout.write(`[home-library-visuals] Validated ${sourceHashes.size} generated previews.\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`[home-library-visuals] ${error && error.message ? error.message : error}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  HOME_LIBRARY_VISUALS,
  expectedPreviewTree,
  listPreviewTree,
  previewPath,
  validateCatalogMappings,
  validateMatchingHashes,
  validatePreview,
  validatePreviewTree
};
