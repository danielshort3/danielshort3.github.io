#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
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
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="crispEdges">${body}</svg>`);
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

function drawSword(item) {
  const blade = equipmentColor(item, ['blade', 'metal'], '#c9d5dd');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#f4fbff');
  const grip = equipmentColor(item, ['grip', 'haft', 'dark'], '#765039');
  return group([
    rect(-37, -4, 18, 8, grip),
    rect(-22, -9, 6, 18, bright),
    rect(-17, -6, 50, 12, blade),
    rect(-12, -4, 40, 3, bright),
    polygon('33,-6 44,0 33,6', bright)
  ]);
}

function drawAxe(item) {
  const haft = equipmentColor(item, ['haft', 'grip', 'dark'], '#765039');
  const blade = equipmentColor(item, ['blade', 'metal'], '#c9d5dd');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#f4fbff');
  return group([
    rect(-39, -4, 70, 8, haft),
    rect(-34, -7, 10, 14, haft),
    polygon('22,-20 42,-15 47,-5 37,1 22,-3', blade),
    polygon('24,-20 42,-15 45,-10 24,-12', bright),
    polygon('22,3 37,-1 47,5 42,15 22,20', blade)
  ]);
}

function drawWand(item, isStaff) {
  const rod = equipmentColor(item, ['rod', 'haft', 'grip'], '#765039');
  const gem = equipmentColor(item, ['gem', 'core', 'bright'], '#c6f4ff');
  const glow = equipmentColor(item, ['glow', 'accent', 'bright'], '#8bd7ff');
  const start = isStaff ? -43 : -32;
  const end = isStaff ? 36 : 27;
  return group([
    rect(start, -4, end - start, 8, rod),
    rect(end - 3, -9, 13, 18, gem),
    rect(end + 1, -5, 6, 6, '#ffffff', ' opacity="0.72"'),
    rect(end - 10, -16, 31, 32, glow, ' opacity="0.18"')
  ]);
}

function drawBow(item, state) {
  const height = item.long ? 88 : 78;
  const half = height / 2;
  const wood = equipmentColor(item, ['wood', 'leather', 'haft'], '#8a5c2f');
  const string = equipmentColor(item, ['string', 'bright', 'trim'], '#f5efd6');
  const arrow = equipmentColor(item, ['arrow', 'accent', 'bright'], '#ffe16a');
  const pull = state === 'draw' ? -22 : state === 'release' ? -7 : 0;
  const arrowStart = state === 'draw' ? -30 : state === 'release' ? -14 : -10;
  const arrowEnd = state === 'release' ? 45 : 36;
  return group([
    linePath(`M 0 ${-half} Q 15 ${-half * 0.58} 10 0 Q 15 ${half * 0.58} 0 ${half}`, wood, 6),
    linePath(`M 1 ${-half + 2} L ${pull} 0 L 1 ${half - 2}`, string, 2),
    rect(arrowStart, -2, arrowEnd - arrowStart, 4, arrow, state === 'rest' ? ' opacity="0.74"' : ''),
    polygon(`${arrowEnd},-5 ${arrowEnd + 9},0 ${arrowEnd},5`, arrow),
    state === 'draw' ? polygon(`${arrowStart},0 ${arrowStart + 8},-6 ${arrowStart + 6},0 ${arrowStart + 8},6`, string) : ''
  ]);
}

function drawChest(item) {
  const cloth = equipmentColor(item, ['cloth', 'metal', 'leather', 'core'], '#6d7682');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright', 'accent'], '#d6a86d');
  const stitch = equipmentColor(item, ['stitch', 'dark', 'sole'], '#392b2b');
  return group([
    polygon('-27,-28 -13,-38 -6,-30 6,-30 13,-38 27,-28 22,31 0,39 -22,31', cloth),
    polygon('-27,-28 -13,-38 -6,-30 -14,-19 -23,-16', trim),
    polygon('27,-28 13,-38 6,-30 14,-19 23,-16', trim),
    rect(-3, -25, 6, 57, stitch),
    rect(-20, 26, 40, 6, trim)
  ]);
}

function drawBoots(item) {
  const leather = equipmentColor(item, ['leather', 'cloth', 'metal'], '#6f412b');
  const sole = equipmentColor(item, ['sole', 'dark', 'edge'], '#2b1d1b');
  const buckle = equipmentColor(item, ['buckle', 'trim', 'bright'], '#d6a14a');
  return group([
    rect(-12, -28, 23, 45, leather),
    rect(-14, 12, 35, 10, sole),
    polygon('-14,17 19,17 26,25 -14,25', sole),
    rect(-9, -3, 17, 6, buckle)
  ]);
}

function drawHead(item) {
  const metal = equipmentColor(item, ['metal', 'cloth', 'leather'], '#667481');
  const dark = equipmentColor(item, ['dark', 'sole', 'stitch'], '#29313a');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright', 'accent'], '#d3dde5');
  return group([
    polygon('-35,17 -31,-14 -20,-31 0,-39 20,-31 31,-14 35,17 24,29 -24,29', dark),
    polygon('-29,12 -25,-12 -16,-25 0,-32 16,-25 25,-12 29,12', metal),
    rect(-30, 10, 60, 8, trim),
    rect(-5, -38, 10, 30, trim),
    polygon('17,-17 34,-8 24,-3', trim)
  ]);
}

function drawGloves(item, grip) {
  const dark = equipmentColor(item, ['dark', 'leather', 'cloth'], '#4b3429');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#8b6a52');
  const edge = equipmentColor(item, ['edge', 'bright', 'buckle'], '#d3dde5');
  const cuff = grip ? equipmentColor(item, ['metal', 'cloth'], metal) : dark;
  return group([
    polygon('-16,-5 -3,-18 13,-13 16,5 5,20 -12,13', dark),
    polygon('-10,-5 -1,-13 9,-9 10,4 2,12 -8,8', metal),
    rect(-16, 7, 17, 10, cuff),
    polygon('8,-13 18,-9 10,-3', edge)
  ]);
}

function drawRing(item) {
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#f7d879');
  const glow = equipmentColor(item, ['glow', 'accent', 'bright'], '#ffe16a');
  return group([
    `<ellipse cx="0" cy="5" rx="19" ry="14" fill="none" stroke="${metal}" stroke-width="9"/>`,
    polygon('-9,-14 0,-25 9,-14 0,-5', glow),
    rect(-3, -21, 6, 6, '#ffffff', ' opacity="0.68"')
  ]);
}

function drawAmulet(item) {
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#d9b967');
  const gem = equipmentColor(item, ['gem', 'glow', 'accent'], '#8bd7ff');
  return group([
    linePath('M -29 -29 Q 0 5 29 -29', metal, 5),
    polygon('0,-1 14,14 0,32 -14,14', metal),
    polygon('0,5 9,15 0,26 -9,15', gem),
    rect(-3, 8, 6, 7, '#ffffff', ' opacity="0.66"')
  ]);
}

function drawShield(item) {
  const face = equipmentColor(item, ['face', 'cloth', 'metal'], '#466d91');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#d5ecff');
  const metal = equipmentColor(item, ['metal', 'dark'], '#6fa8d9');
  return group([
    polygon('-37,-38 37,-38 33,22 0,43 -33,22', trim),
    polygon('-30,-31 30,-31 26,17 0,35 -26,17', face),
    rect(-5, -28, 10, 56, metal),
    rect(-25, -7, 50, 11, metal),
    rect(-3, -24, 6, 7, trim)
  ]);
}

function drawCore(item) {
  const core = equipmentColor(item, ['core', 'gem', 'metal'], '#ff6b35');
  const glow = equipmentColor(item, ['glow', 'bright', 'accent'], '#ffc15e');
  const dark = equipmentColor(item, ['dark', 'cloth'], '#8b2635');
  return group([
    polygon('0,-39 27,-19 34,15 13,37 -18,34 -36,8 -27,-23', dark),
    polygon('0,-29 22,-12 23,14 7,28 -17,24 -26,5 -19,-17', core),
    polygon('0,-19 14,-6 10,13 -6,19 -15,4 -10,-10', glow),
    rect(-4, -12, 8, 9, '#ffffff', ' opacity="0.64"')
  ]);
}

function drawFocus(item) {
  const core = equipmentColor(item, ['core', 'gem', 'metal'], '#28c7b7');
  const glow = equipmentColor(item, ['glow', 'bright', 'accent'], '#b8fff2');
  const dark = equipmentColor(item, ['dark', 'cloth'], '#146b72');
  return group([
    polygon('0,-42 31,-12 25,27 0,42 -25,27 -31,-12', dark),
    polygon('0,-32 22,-9 17,20 0,31 -17,20 -22,-9', glow),
    polygon('0,-20 14,0 0,20 -14,0', core),
    rect(-4, -12, 8, 9, '#ffffff', ' opacity="0.68"')
  ]);
}

function drawScope(item) {
  const metal = equipmentColor(item, ['metal', 'dark'], '#4b5663');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#d8c25f');
  const lens = equipmentColor(item, ['lens', 'glow', 'accent'], '#ffe16a');
  return group([
    rect(-39, -11, 68, 22, metal),
    rect(-27, -17, 13, 34, trim),
    rect(23, -16, 15, 32, trim),
    ellipse(35, 0, 8, 16, lens),
    rect(-18, 11, 19, 10, metal),
    rect(-15, 18, 13, 7, trim),
    rect(31, -8, 4, 9, '#ffffff', ' opacity="0.62"')
  ]);
}

function drawKit(item) {
  const leather = equipmentColor(item, ['leather', 'cloth', 'dark'], '#8a5a36');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#b7c3ca');
  const cord = equipmentColor(item, ['cord', 'edge', 'stitch'], '#3f2c24');
  return group([
    polygon('-37,-28 27,-28 36,-15 32,33 -29,33 -39,18', cord),
    rect(-32, -22, 57, 47, leather),
    rect(-27, -17, 47, 10, metal),
    rect(-5, -8, 10, 24, cord),
    rect(-28, 17, 49, 7, metal),
    linePath('M 25 -14 Q 43 0 31 25', cord, 6),
    rect(27, 18, 12, 17, metal)
  ]);
}

function drawItem(item, state) {
  if (item.kind === 'sword') return drawSword(item);
  if (item.kind === 'axe') return drawAxe(item);
  if (item.kind === 'wand') return drawWand(item, false);
  if (item.kind === 'staff') return drawWand(item, true);
  if (item.kind === 'bow') return drawBow(item, state);
  if (item.kind === 'chest') return drawChest(item);
  if (item.kind === 'boots') return drawBoots(item);
  if (item.kind === 'head') return drawHead(item);
  if (item.kind === 'gloves') return drawGloves(item, false);
  if (item.kind === 'ring') return drawRing(item);
  if (item.kind === 'amulet') return drawAmulet(item);
  if (item.kind === 'shield') return drawShield(item);
  if (item.kind === 'grip') return drawGloves(item, true);
  if (item.kind === 'core') return drawCore(item);
  if (item.kind === 'focus') return drawFocus(item);
  if (item.kind === 'scope') return drawScope(item);
  if (item.kind === 'kit') return drawKit(item);
  throw new Error(`Cannot render unrecognized equipment kind: ${item.kind}`);
}

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
    const body = drawItem(item, state);
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
  await sharp(makeAtlas(item)).png({ compressionLevel: 9 }).toFile(destination);
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
