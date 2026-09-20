#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const crypto = require('crypto');
const { drawItem } = require('./project-starfall-equipment-illustrations.js');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Attachments = require('../js/games/project-starfall/engine/equipment-attachments.js');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT_DIR = path.join(ROOT, 'img/project-starfall/equipment-atlases');
const CELL_SIZE = Attachments.ATLAS_CELL_SIZE;
const ANGLES = Attachments.ATLAS_ANGLE_SETS.weapon;
const BOW_STATES = Object.freeze(['rest', 'draw', 'release']);
const EXPECTED_VISUAL_COUNT = 85;
const ALPHA_THRESHOLD = 8;
const MIN_VISIBLE_PIXELS = 8;
const RECOGNIZED_KINDS = Object.freeze([
  'sword',
  'axe',
  'wand',
  'staff',
  'bow',
  'chest',
  'boots',
  'head',
  'gloves',
  'ring',
  'amulet',
  'shield',
  'grip',
  'core',
  'focus',
  'scope',
  'kit'
]);
const RECOGNIZED_KIND_SET = new Set(RECOGNIZED_KINDS);

function svg(width, height, body) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">${body}</svg>`);
}

function rect(x, y, width, height, color, extra) {
  return `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${color}"${extra || ''}/>`;
}

function ellipse(cx, cy, rx, ry, color, extra) {
  return `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="${color}"${extra || ''}/>`;
}

function polygon(points, color, extra) {
  return `<polygon points="${points}" fill="${color}"${extra || ''}/>`;
}

function linePath(d, color, width, extra) {
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="square" stroke-linejoin="miter"${extra || ''}/>`;
}

function group(children) {
  return Array.isArray(children) ? children.join('') : String(children || '');
}

function equipmentColor(item, keys, fallback) {
  for (const key of keys) {
    if (item[key]) return item[key];
  }
  return fallback;
}

function buildEquipmentDefinitions() {
  const visuals = Object.values(Data.EQUIPMENT_VISUALS || {});
  const styles = Data.PLAYER_RIGS && Data.PLAYER_RIGS.fighter && Data.PLAYER_RIGS.fighter.equipmentVisuals || {};
  const definitions = visuals.map((visual) => {
    const style = Object.assign({}, styles[visual.id] || {});
    if (style.shine && !style.bright) style.bright = style.shine;
    if (style.grip && !style.haft && (style.kind === 'axe' || visual.kind === 'axe')) style.haft = style.grip;
    return Object.assign({}, style, {
      id: visual.id,
      fileId: visual.fileId || String(visual.id || '').replace(/_/g, '-'),
      kind: style.kind || visual.kind || '',
      slot: visual.slot || '',
      atlas: visual.atlas || null
    });
  });

  if (definitions.length !== EXPECTED_VISUAL_COUNT) {
    throw new Error(`ProjectStarfallData exposes ${definitions.length} equipment visuals; expected ${EXPECTED_VISUAL_COUNT}`);
  }

  const ids = new Set();
  const fileIds = new Set();
  for (const item of definitions) {
    if (!item.id || !item.fileId) throw new Error('Every equipment visual must define an id and fileId');
    if (ids.has(item.id)) throw new Error(`Duplicate equipment visual id: ${item.id}`);
    if (fileIds.has(item.fileId)) throw new Error(`Duplicate equipment visual fileId: ${item.fileId}`);
    if (!RECOGNIZED_KIND_SET.has(item.kind)) {
      throw new Error(`Unrecognized equipment kind "${item.kind || '(empty)'}" for ${item.id}`);
    }
    ids.add(item.id);
    fileIds.add(item.fileId);
  }
  return definitions;
}

const EQUIPMENT = Object.freeze(buildEquipmentDefinitions());

function atlasRows(item) {
  return item.kind === 'bow' ? BOW_STATES : Object.freeze(['default']);
}

function atlasAngles(item) {
  if (item.atlas && Array.isArray(item.atlas.angles) && item.atlas.angles.length) {
    return item.atlas.angles;
  }
  return Attachments.getEquipmentAngleSet(item.kind);
}

function makeAtlas(item) {
  const rows = atlasRows(item);
  const angles = atlasAngles(item);
  const cells = rows.map((state, rowIndex) => angles.map((angle, columnIndex) => {
    const body = drawItem(item, state, rowIndex + '-' + columnIndex);
    return `<svg x="${columnIndex * CELL_SIZE}" y="${rowIndex * CELL_SIZE}" width="${CELL_SIZE}" height="${CELL_SIZE}" viewBox="0 0 ${CELL_SIZE} ${CELL_SIZE}" overflow="hidden"><g transform="translate(${CELL_SIZE / 2} ${CELL_SIZE / 2}) rotate(${angle})">${body}</g></svg>`;
  }).join('')).join('');
  return svg(angles.length * CELL_SIZE, rows.length * CELL_SIZE, cells);
}

function atlasPath(item) {
  return path.join(OUTPUT_DIR, `${item.fileId}-atlas.png`);
}

async function writeAtlas(item) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const destination = atlasPath(item);
  const source = makeAtlas(item);
  const sourceDir = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/icons/equipment');
  fs.mkdirSync(sourceDir, { recursive: true });
  fs.writeFileSync(path.join(sourceDir, item.fileId + '.svg'), source);
  await sharp(source).png({ compressionLevel: 9 }).toFile(destination);
  return destination;
}

function cellVisibleAlpha(raw, width, row, column) {
  const x0 = column * CELL_SIZE;
  const y0 = row * CELL_SIZE;
  let visible = 0;
  for (let y = 0; y < CELL_SIZE; y += 1) {
    for (let x = 0; x < CELL_SIZE; x += 1) {
      if (raw[(((y0 + y) * width) + x0 + x) * 4 + 3] > ALPHA_THRESHOLD) visible += 1;
    }
  }
  return visible;
}

function cellEdgeHasAlpha(raw, width, row, column) {
  const x0 = column * CELL_SIZE;
  const y0 = row * CELL_SIZE;
  for (let offset = 0; offset < CELL_SIZE; offset += 1) {
    if (raw[((y0 * width) + x0 + offset) * 4 + 3] > ALPHA_THRESHOLD) return true;
    if (raw[(((y0 + CELL_SIZE - 1) * width) + x0 + offset) * 4 + 3] > ALPHA_THRESHOLD) return true;
    if (raw[(((y0 + offset) * width) + x0) * 4 + 3] > ALPHA_THRESHOLD) return true;
    if (raw[(((y0 + offset) * width) + x0 + CELL_SIZE - 1) * 4 + 3] > ALPHA_THRESHOLD) return true;
  }
  return false;
}

async function validateAtlas(item) {
  const filePath = atlasPath(item);
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing equipment atlas: ${path.relative(ROOT, filePath)}`);
  }
  const rows = atlasRows(item);
  const angles = atlasAngles(item);
  const expectedWidth = angles.length * CELL_SIZE;
  const expectedHeight = rows.length * CELL_SIZE;
  const decoded = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  if (decoded.info.width !== expectedWidth || decoded.info.height !== expectedHeight) {
    throw new Error(`${path.relative(ROOT, filePath)} is ${decoded.info.width}x${decoded.info.height}; expected ${expectedWidth}x${expectedHeight}`);
  }
  for (let row = 0; row < rows.length; row += 1) {
    for (let column = 0; column < angles.length; column += 1) {
      if (cellVisibleAlpha(decoded.data, decoded.info.width, row, column) < MIN_VISIBLE_PIXELS) {
        throw new Error(`${path.relative(ROOT, filePath)} ${rows[row]} angle ${angles[column]} is empty`);
      }
      if (cellEdgeHasAlpha(decoded.data, decoded.info.width, row, column)) {
        throw new Error(`${path.relative(ROOT, filePath)} ${rows[row]} angle ${angles[column]} touches a cell edge`);
      }
    }
  }
}

function validateOutputFileCount() {
  const files = fs.existsSync(OUTPUT_DIR)
    ? fs.readdirSync(OUTPUT_DIR).filter((file) => file.endsWith('-atlas.png'))
    : [];
  const expected = new Set(EQUIPMENT.map((item) => `${item.fileId}-atlas.png`));
  const unexpected = files.filter((file) => !expected.has(file));
  if (files.length !== EQUIPMENT.length || unexpected.length) {
    throw new Error(`Equipment atlas directory contains ${files.length} atlas PNGs; expected exactly ${EQUIPMENT.length}${unexpected.length ? ` (unexpected: ${unexpected.join(', ')})` : ''}`);
  }
}

function parseArguments(argv) {
  const args = Array.from(argv || []);
  const itemIndex = args.indexOf('--item');
  const itemId = itemIndex >= 0 ? String(args[itemIndex + 1] || '').trim() : '';
  const all = args.includes('--all') || itemIndex < 0;
  const validateOnly = args.includes('--validate');
  const recognized = new Set(['--item', '--all', '--validate']);
  const unknown = args.filter((arg, index) => {
    if (itemIndex >= 0 && index === itemIndex + 1) return false;
    return !recognized.has(arg);
  });
  if (unknown.length) throw new Error(`Unsupported argument${unknown.length === 1 ? '' : 's'}: ${unknown.join(', ')}`);
  if (itemIndex >= 0 && !itemId) throw new Error('--item requires an equipment visual id or fileId');
  if (itemIndex >= 0 && args.includes('--all')) throw new Error('Use either --item <id> or --all, not both');
  return { all, itemId, validateOnly };
}

function selectTargets(options) {
  if (options.all) return EQUIPMENT;
  const normalized = String(options.itemId || '').toLowerCase();
  const item = EQUIPMENT.find((candidate) => candidate.id.toLowerCase() === normalized || candidate.fileId.toLowerCase() === normalized);
  if (!item) throw new Error(`Unknown equipment visual: ${options.itemId}`);
  return Object.freeze([item]);
}

async function main(argv) {
  const options = parseArguments(argv == null ? process.argv.slice(2) : argv);
  const targets = selectTargets(options);

  const ledgerPath = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/icons/ledger.json');
  const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const relative = file => path.relative(ROOT, file).replace(/\\/g, '/');
  const ledger = fs.existsSync(ledgerPath) ? JSON.parse(fs.readFileSync(ledgerPath)) : null;
  if (ledger && !options.validateOnly) {
    ledger.equipmentBaseline ||= {};
    for (const item of targets) if (!ledger.equipmentBaseline[relative(atlasPath(item))]) ledger.equipmentBaseline[relative(atlasPath(item))] = digest(atlasPath(item));
    fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
  }
  if (!options.validateOnly) {
    for (const item of targets) {
      const destination = await writeAtlas(item);
      process.stdout.write(`Generated ${path.relative(ROOT, destination).replace(/\\/g, '/')} (${item.kind}, ${atlasAngles(item).length} angles${item.kind === 'bow' ? ` x ${BOW_STATES.length} states` : ''})\n`);
    }
  }

  for (const item of targets) {
    await validateAtlas(item);
  }
  if (options.all) validateOutputFileCount();
  if (ledger && !options.validateOnly) {
    ledger.equipmentOutputs ||= {};
    for (const item of targets) {
      const source = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/icons/equipment', item.fileId + '.svg');
      ledger.equipmentOutputs[relative(atlasPath(item))] = { sha256: digest(atlasPath(item)), source: relative(source), sourceHash: digest(source), kind: item.kind, angles: atlasAngles(item), rows: atlasRows(item), cellSize: CELL_SIZE, tool: 'native editable SVG; substantial illustrated geometry replacement', generator: 'build/generate-project-starfall-equipment-atlases.js', illustrationSource: 'build/project-starfall-equipment-illustrations.js' };
    }
    fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
  }
  process.stdout.write(`Validated ${targets.length} equipment atlas${targets.length === 1 ? '' : 'es'} from ${EQUIPMENT.length} recognized ProjectStarfallData visuals\n`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

module.exports = Object.freeze({
  ANGLES,
  BOW_STATES,
  CELL_SIZE,
  EQUIPMENT,
  RECOGNIZED_KINDS,
  atlasAngles,
  atlasPath,
  main,
  makeAtlas,
  validateAtlas
});
