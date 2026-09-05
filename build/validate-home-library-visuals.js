#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const previewRoot = path.join(root, 'img', 'home-previews');
const publicPreviewRoot = path.join(root, 'public', 'img', 'home-previews');

const GENERATED_HOME_LIBRARY_VISUALS = {
  games: {
    'stellar-dogfight': 'space-combat-key-art',
    roulette: 'double-zero-roulette',
    'probability-engine': 'probability-branching',
    stormbreak: 'olympian-storm-key-art',
    'ocean-wave-simulation': 'wave-parameter-study'
  }
};

// These superseded AI previews stay in the asset tree for history, but library data must not reference them.
const RETAINED_TOOL_PREVIEW_IDS = [
  'text-compare',
  'nbsp-cleaner',
  'oxford-comma-checker',
  'point-of-view-checker',
  'word-frequency',
  'utm-batch-builder',
  'qr-code-generator',
  'image-optimizer',
  'background-remover',
  'screen-recorder'
];

const RETAINED_PROJECT_PREVIEW_IDS = [
  'smartSentence',
  'chatbotLora',
  'shapeClassifier',
  'ufoDashboard',
  'covidAnalysis',
  'targetEmptyPackage',
  'handwritingRating',
  'digitGenerator',
  'sheetMusicUpscale',
  'deliveryTip',
  'retailStore',
  'pizza',
  'babynames',
  'pizzaDashboard',
  'nonogram',
  'website'
];

// Project Starfall is no longer public, but its generated preview remains with
// the archived source assets and must never re-enter the games catalog.
const RETAINED_GAME_PREVIEW_IDS = ['project-starfall'];

function previewPath(category, id) {
  return path.join(previewRoot, category, `${id}.webp`);
}

function normalizedRelativePath(value) {
  return String(value || '').split(path.sep).join('/');
}

function projectLibraryAsset(image) {
  const normalized = normalizedRelativePath(String(image || '').trim()).replace(/[?#].*$/, '');
  if (!normalized || /^(?:[a-z]+:)?\/\//i.test(normalized)) return '';
  const rooted = normalized.startsWith('/') ? normalized : `/${normalized}`;
  const extension = path.posix.extname(rooted);
  const basename = extension ? rooted.slice(0, -extension.length) : rooted;
  return `${basename}-640.webp`;
}

function loadPublishedProjects() {
  const projectsRoot = path.join(root, 'content', 'projects');
  return fs.readdirSync(projectsRoot)
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => JSON.parse(fs.readFileSync(path.join(projectsRoot, fileName), 'utf8')))
    .filter((project) => project && project.id && project.published !== false)
    .sort((left, right) => {
      const leftOrder = Number.isFinite(Number(left.order)) ? Number(left.order) : Number.MAX_SAFE_INTEGER;
      const rightOrder = Number.isFinite(Number(right.order)) ? Number(right.order) : Number.MAX_SAFE_INTEGER;
      return leftOrder - rightOrder || String(left.id).localeCompare(String(right.id));
    });
}

function loadToolIcons() {
  const toolsRoot = path.join(root, 'content', 'tools');
  return fs.readdirSync(toolsRoot)
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => JSON.parse(fs.readFileSync(path.join(toolsRoot, fileName), 'utf8')))
    .map((tool) => ({
      id: String(tool.slug || '').trim(),
      image: `/${normalizedRelativePath(tool.iconImage).replace(/^\/+/, '')}`,
      public: Boolean(tool.href) && String(tool.visibility || 'public').trim().toLowerCase() === 'public' &&
        !tool.hidden && !tool.noindex
    }));
}

function expectedPreviewTree() {
  const previewInventory = {
    projects: Object.fromEntries(RETAINED_PROJECT_PREVIEW_IDS.map((id) => [id, true])),
    tools: Object.fromEntries(RETAINED_TOOL_PREVIEW_IDS.map((id) => [id, true])),
    ...GENERATED_HOME_LIBRARY_VISUALS,
    games: {
      ...GENERATED_HOME_LIBRARY_VISUALS.games,
      ...Object.fromEntries(RETAINED_GAME_PREVIEW_IDS.map((id) => [id, true]))
    }
  };
  const directories = Object.keys(previewInventory).sort();
  const files = directories.flatMap((category) => Object.keys(previewInventory[category])
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

  const projects = loadPublishedProjects();
  const projectItems = libraryData.projects?.items || [];
  if (JSON.stringify(projectItems.map((item) => item.id)) !==
    JSON.stringify(projects.map((project) => String(project.id)))) {
    throw new Error('Homepage projects preview catalog is out of sync with published project content');
  }
  projects.forEach((project) => {
    const id = String(project.id).trim();
    const expectedImage = projectLibraryAsset(project.image);
    if (expectedImage !== `/img/projects/${id}-640.webp`) {
      throw new Error(`Canonical project image does not derive the expected optimized asset for ${id}`);
    }
    const item = projectItems.find((entry) => entry.id === id);
    if (!item || item.image !== expectedImage || item.imageAlt !== '') {
      throw new Error(`Unexpected original project preview mapping for projects/${id}`);
    }
  });

  const tools = loadToolIcons();
  const toolIcons = new Map(tools.map((tool) => [tool.id, tool.image]));
  const toolItems = libraryData.tools?.items || [];
  if (JSON.stringify(toolItems.map((item) => item.id).sort()) !==
    JSON.stringify(tools.filter((tool) => tool.public).map((tool) => tool.id).sort())) {
    throw new Error('Homepage tools icon catalog is out of sync with public tool content');
  }
  toolItems.forEach((item) => {
    if (!toolIcons.has(item.id) || item.image !== toolIcons.get(item.id) || item.imageAlt !== '') {
      throw new Error(`Unexpected original tool icon mapping for tools/${item.id}`);
    }
  });

  Object.entries(GENERATED_HOME_LIBRARY_VISUALS).forEach(([category, visuals]) => {
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

  return projects;
}

async function validateProjectAssetAt(baseDir, project) {
  const assetPath = projectLibraryAsset(project.image);
  const filePath = path.join(baseDir, assetPath.replace(/^\/+/, ''));
  if (!assetPath || !fs.existsSync(filePath)) {
    throw new Error(`Missing original project preview: ${path.relative(root, filePath)}`);
  }

  const metadata = await sharp(filePath).metadata();
  if (metadata.format !== 'webp' || metadata.width !== 640 || !Number(metadata.height)) {
    throw new Error(`Original project preview must be a 640px-wide WebP: ${path.relative(root, filePath)}`);
  }

  const buffer = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

async function validateProjectAssets(baseDir, projects) {
  const hashes = new Map();
  for (const project of projects) {
    const assetPath = projectLibraryAsset(project.image);
    hashes.set(assetPath, await validateProjectAssetAt(baseDir, project));
  }
  return hashes;
}

async function validateToolIconAssets(baseDir) {
  const hashes = new Map();
  for (const tool of loadToolIcons()) {
    if (!tool.id || tool.image !== `/img/tools/icons/${tool.id}.png`) {
      throw new Error(`Unexpected canonical tool icon path for ${tool.id || 'unnamed tool'}`);
    }
    const filePath = path.join(baseDir, tool.image.replace(/^\/+/, ''));
    if (!fs.existsSync(filePath)) {
      throw new Error(`Missing original tool icon: ${path.relative(root, filePath)}`);
    }
    const metadata = await sharp(filePath).metadata();
    if (metadata.format !== 'png' || !Number(metadata.width) || !Number(metadata.height)) {
      throw new Error(`Original tool icon must be a valid PNG: ${path.relative(root, filePath)}`);
    }
    hashes.set(tool.image, crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex'));
  }
  return hashes;
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
  for (const [category, visuals] of Object.entries(GENERATED_HOME_LIBRARY_VISUALS)) {
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
  const projects = validateCatalogMappings();
  const projectSourceHashes = await validateProjectAssets(root, projects);
  const toolSourceHashes = await validateToolIconAssets(root);
  const sourceHashes = await validatePreviewTree(previewRoot, 'img/home-previews');

  if (validatePublic) {
    const projectDeployedHashes = await validateProjectAssets(path.join(root, 'public'), projects);
    const toolDeployedHashes = await validateToolIconAssets(path.join(root, 'public'));
    const deployedHashes = await validatePreviewTree(publicPreviewRoot, 'public/img/home-previews');
    validateMatchingHashes(projectSourceHashes, projectDeployedHashes);
    validateMatchingHashes(toolSourceHashes, toolDeployedHashes);
    validateMatchingHashes(sourceHashes, deployedHashes);
    process.stdout.write(`[home-library-visuals] Validated ${projectSourceHashes.size} original project previews, ${toolSourceHashes.size} original tool icons, and ${sourceHashes.size} source and deployed generated previews.\n`);
    return;
  }

  process.stdout.write(`[home-library-visuals] Validated ${projectSourceHashes.size} original project previews, ${toolSourceHashes.size} original tool icons, and ${sourceHashes.size} generated previews.\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`[home-library-visuals] ${error && error.message ? error.message : error}\n`);
    process.exitCode = 1;
  });
}

module.exports = {
  GENERATED_HOME_LIBRARY_VISUALS,
  RETAINED_GAME_PREVIEW_IDS,
  RETAINED_PROJECT_PREVIEW_IDS,
  RETAINED_TOOL_PREVIEW_IDS,
  expectedPreviewTree,
  listPreviewTree,
  previewPath,
  projectLibraryAsset,
  validateCatalogMappings,
  validateMatchingHashes,
  validatePreview,
  validateProjectAssets,
  validateToolIconAssets,
  validatePreviewTree
};
