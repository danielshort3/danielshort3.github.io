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
const ATLAS_VERSION = 'v2';
const EXPECTED_VISUAL_COUNT = 85;
const ALPHA_THRESHOLD = 8;
const MIN_VISIBLE_PIXELS = 8;
const STARFRONT_PROFILE_ID = 'fractured-starfront';
const MIN_STARFRONT_PALETTE_COLORS = 4;
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

function isStarfrontEquipment(item) {
  return String(item && item.profile || '').trim() === STARFRONT_PROFILE_ID;
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
  const starfront = isStarfrontEquipment(item);
  const blade = equipmentColor(item, ['blade', 'metal'], '#82919b');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#d7e2e7');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'shadow'], '#263139');
  const grip = equipmentColor(item, ['grip', 'haft', 'leather', 'dark'], '#3b3432');
  const accent = equipmentColor(item, ['accent', 'glow'], starfront ? '#54c9da' : bright);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : grip);
  const tip = Math.max(29, Math.min(43, Number(item.bladeLength || (starfront ? 35 : 41))));
  const half = Math.max(3, Math.min(6, Number(item.bladeHalfWidth || (starfront ? 4 : 5))));
  return group([
    polygon(`-38,0 -34,-6 -29,-5 -29,5 -34,6`, dark),
    rect(-34, -4, 18, 8, dark),
    rect(-32, -2, 14, 4, grip),
    rect(-29, -3, 3, 6, ember),
    rect(-23, -3, 3, 6, accent),
    polygon(`-19,-9 -14,-6 -14,6 -19,9 -22,5 -22,-5`, dark),
    rect(-19, -6, 4, 12, bright),
    polygon(`-15,${-half - 1} ${tip - 6},${-half} ${tip},0 ${tip - 6},${half} -15,${half + 1}`, dark),
    polygon(`-13,${-half + 1} ${tip - 7},${-half + 1} ${tip - 2},0 ${tip - 7},${half - 1} -13,${half - 1}`, blade),
    polygon(`-10,${-half + 1} ${tip - 8},${-half + 1} ${tip - 13},${-half + 3} -10,${-half + 3}`, bright, ' opacity="0.76"'),
    rect(-9, -1, Math.max(12, tip - 17), 2, accent, ' opacity="0.82"')
  ]);
}

function drawAxe(item) {
  const starfront = isStarfrontEquipment(item);
  const haft = equipmentColor(item, ['haft', 'grip', 'leather', 'dark'], '#3b3432');
  const blade = equipmentColor(item, ['blade', 'metal'], '#82919b');
  const bright = equipmentColor(item, ['bright', 'shine', 'edge'], '#d7e2e7');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'shadow'], '#263139');
  const accent = equipmentColor(item, ['accent', 'glow'], starfront ? '#54c9da' : bright);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : haft);
  return group([
    rect(-37, -4, 65, 8, dark),
    rect(-35, -2, 60, 4, haft),
    rect(-30, -3, 4, 6, ember),
    rect(15, -4, 6, 8, accent),
    polygon('17,-19 31,-22 42,-15 45,-5 35,1 18,-3', dark),
    polygon('20,-16 31,-19 39,-14 41,-7 33,-3 20,-6', blade),
    polygon('22,-15 31,-18 37,-14 24,-11', bright, ' opacity="0.72"'),
    polygon('18,3 35,-1 45,5 42,15 31,22 17,19', dark),
    polygon('20,6 33,3 41,7 38,14 30,18 20,16', blade),
    polygon('24,6 35,4 39,7 34,9', accent, ' opacity="0.78"'),
    rect(-39, -6, 7, 12, dark)
  ]);
}

function drawWand(item, isStaff) {
  const starfront = isStarfrontEquipment(item);
  const rod = equipmentColor(item, ['rod', 'haft', 'grip'], '#3b3432');
  const gem = equipmentColor(item, ['gem', 'core', 'bright'], '#a8edf3');
  const glow = equipmentColor(item, ['glow', 'accent', 'bright'], '#54c9da');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'shadow'], '#263139');
  const metal = equipmentColor(item, ['metal', 'blade', 'trim'], '#7f8f99');
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : rod);
  const start = isStaff ? -42 : -31;
  const end = isStaff ? 34 : starfront ? 22 : 27;
  const headRadius = isStaff ? 12 : starfront ? 9 : 11;
  return group([
    ellipse(end + 2, 0, headRadius + 3, headRadius + 3, glow, ' opacity="0.09"'),
    rect(start, -4, end - start, 8, dark),
    rect(start + 2, -2, end - start - 2, 4, rod),
    rect(start + 8, -3, 5, 6, ember),
    rect(end - 10, -4, 6, 8, glow),
    polygon(`${end - 3},${-headRadius} ${end + headRadius - 2},${-headRadius + 3} ${end + headRadius + 2},0 ${end + headRadius - 2},${headRadius - 3} ${end - 3},${headRadius} ${end - 8},0`, dark),
    polygon(`${end - 1},${-headRadius + 3} ${end + headRadius - 4},${-headRadius + 5} ${end + headRadius - 1},0 ${end + headRadius - 4},${headRadius - 5} ${end - 1},${headRadius - 3} ${end - 5},0`, metal),
    polygon(`${end + 1},-5 ${end + 7},0 ${end + 1},5 ${end - 2},0`, gem),
    rect(end + 1, -4, 3, 4, '#ffffff', ' opacity="0.72"')
  ]);
}

function drawBow(item, state) {
  const starfront = isStarfrontEquipment(item);
  const height = item.long ? 78 : starfront ? 66 : 72;
  const half = height / 2;
  const wood = equipmentColor(item, ['wood', 'leather', 'haft'], '#485057');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'shadow'], '#263139');
  const metal = equipmentColor(item, ['metal', 'trim'], '#7f8f99');
  const string = equipmentColor(item, ['string', 'bright', 'trim'], '#a8edf3');
  const arrow = equipmentColor(item, ['arrow', 'ember', 'warm', 'accent'], '#cf7049');
  const accent = equipmentColor(item, ['accent', 'glow'], starfront ? '#54c9da' : arrow);
  const pull = state === 'draw' ? -18 : state === 'release' ? -5 : 5;
  const arrowStart = state === 'draw' ? -26 : state === 'release' ? -11 : -7;
  const arrowEnd = state === 'release' ? 39 : 33;
  const arrowArt = state === 'rest' ? '' : group([
    rect(arrowStart, -1.5, arrowEnd - arrowStart, 3, metal),
    polygon(`${arrowEnd},-4 ${arrowEnd + 8},0 ${arrowEnd},4`, arrow),
    state === 'draw' ? polygon(`${arrowStart},0 ${arrowStart + 7},-5 ${arrowStart + 5},0 ${arrowStart + 7},5`, string) : ''
  ]);
  return group([
    linePath(`M 0 ${-half} Q 17 ${-half * 0.57} 10 0 Q 17 ${half * 0.57} 0 ${half}`, dark, 8),
    linePath(`M 1 ${-half + 2} Q 15 ${-half * 0.54} 10 0 Q 15 ${half * 0.54} 1 ${half - 2}`, wood, 4),
    linePath(`M 2 ${-half + 2} L ${pull} 0 L 2 ${half - 2}`, string, 2),
    rect(6, -8, 8, 16, dark),
    rect(8, -6, 4, 12, metal),
    rect(8, -2, 4, 4, accent),
    polygon(`-2,${-half + 4} 3,${-half} 5,${-half + 6}`, arrow),
    polygon(`-2,${half - 4} 3,${half} 5,${half - 6}`, arrow),
    arrowArt
  ]);
}

function drawChest(item) {
  const starfront = isStarfrontEquipment(item);
  const cloth = equipmentColor(item, ['cloth', 'metal', 'leather', 'core'], '#48545e');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#84949e');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'sole'], '#202a32');
  const accent = equipmentColor(item, ['accent', 'glow', 'stitch'], starfront ? '#54c9da' : trim);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : trim);
  return group([
    polygon('-23,-25 -12,-33 -5,-27 5,-27 12,-33 23,-25 20,23 11,30 0,34 -11,30 -20,23', dark),
    polygon('-19,-23 -11,-28 -5,-23 -5,25 -11,26 -16,21', cloth),
    polygon('19,-23 11,-28 5,-23 5,25 11,26 16,21', cloth),
    polygon('-5,-23 0,-19 5,-23 5,25 0,29 -5,25', trim, ' opacity="0.68"'),
    polygon('-20,-23 -12,-31 -6,-25 -11,-16 -19,-14', trim),
    polygon('20,-23 12,-31 6,-25 11,-16 19,-14', trim),
    linePath('M -13 -10 L -10 20 L 0 27 L 10 20 L 13 -10', accent, 2),
    rect(-3, 4, 6, 3, ember),
    rect(-16, 20, 32, 4, dark),
    rect(-6, 20, 12, 4, trim)
  ]);
}

function drawBoots(item) {
  const starfront = isStarfrontEquipment(item);
  const leather = equipmentColor(item, ['leather', 'cloth', 'metal'], '#48545e');
  const sole = equipmentColor(item, ['sole', 'darkMetal', 'dark', 'edge'], '#202a32');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#84949e');
  const accent = equipmentColor(item, ['accent', 'glow', 'buckle'], starfront ? '#54c9da' : metal);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : leather);
  return group([
    polygon('-9,-18 7,-18 8,2 14,5 17,10 14,14 -9,14 -11,8 -8,1', sole),
    polygon('-6,-15 5,-15 6,4 12,7 13,10 -6,10 -8,6 -5,0', leather),
    polygon('-6,-15 5,-15 5,-9 -6,-7', metal, ' opacity="0.72"'),
    rect(-7, 9, 21, 4, sole),
    polygon('7,5 12,7 13,10 6,10', metal),
    rect(-5, -3, 10, 2, accent),
    rect(-4, -13, 4, 3, ember)
  ]);
}

function drawHead(item) {
  const starfront = isStarfrontEquipment(item);
  const metal = equipmentColor(item, ['metal', 'cloth', 'leather'], '#71808a');
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'sole', 'stitch'], '#202a32');
  const trim = equipmentColor(item, ['trim', 'edge', 'bright'], '#a9bac4');
  const accent = equipmentColor(item, ['accent', 'glow'], starfront ? '#54c9da' : trim);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : metal);
  return group([
    polygon('-23,4 -22,-10 -13,-23 0,-29 13,-23 22,-10 23,4 16,9 -16,9', dark),
    polygon('-19,0 -18,-9 -10,-19 0,-24 10,-19 18,-9 19,0 13,3 -13,3', metal),
    polygon('-18,-8 -10,-19 0,-24 0,-16 -12,-8', trim, ' opacity="0.62"'),
    rect(-18, -1, 36, 4, accent),
    polygon('-19,2 -12,4 -12,14 -18,18 -21,11', dark),
    polygon('19,2 12,4 12,14 18,18 21,11', dark),
    rect(-3, -27, 6, 11, trim),
    rect(-2, -25, 4, 4, ember)
  ]);
}

function drawGloves(item, grip) {
  const starfront = isStarfrontEquipment(item);
  const dark = equipmentColor(item, ['darkMetal', 'dark', 'leather', 'cloth'], '#202a32');
  const metal = equipmentColor(item, ['metal', 'trim', 'bright'], '#71808a');
  const edge = equipmentColor(item, ['edge', 'bright'], '#a9bac4');
  const accent = equipmentColor(item, ['accent', 'glow', 'buckle'], starfront ? '#54c9da' : edge);
  const ember = equipmentColor(item, ['ember', 'warm'], starfront ? '#cf7049' : metal);
  const cuff = grip ? equipmentColor(item, ['metal', 'cloth'], metal) : dark;
  return group([
    polygon('-10,-3 -2,-13 8,-10 11,3 4,12 -8,8', dark),
    polygon('-6,-3 -1,-9 5,-7 7,2 2,7 -5,5', metal),
    rect(-10, 5, 11, 7, cuff),
    rect(-8, 6, 8, 2, accent),
    polygon('5,-9 12,-6 7,-2', edge),
    rect(-3, -8, 3, 3, ember)
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
  return path.join(OUTPUT_DIR, `${item.fileId}-atlas-${ATLAS_VERSION}.png`);
}

function legacyAtlasPath(item) {
  return path.join(OUTPUT_DIR, `${item.fileId}-atlas.png`);
}

async function writeAtlas(item) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const destination = atlasPath(item);
  await sharp(makeAtlas(item)).png({ compressionLevel: 9 }).toFile(destination);
  const legacyPath = legacyAtlasPath(item);
  if (legacyPath !== destination && fs.existsSync(legacyPath)) fs.unlinkSync(legacyPath);
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

function parseHexColor(value) {
  const hex = String(value || '').trim().replace(/^#/, '');
  if (!/^[0-9a-f]{6}$/i.test(hex)) return null;
  return {
    r: Number.parseInt(hex.slice(0, 2), 16),
    g: Number.parseInt(hex.slice(2, 4), 16),
    b: Number.parseInt(hex.slice(4, 6), 16)
  };
}

function getCellVisualStats(raw, width, row, column) {
  const x0 = column * CELL_SIZE;
  const y0 = row * CELL_SIZE;
  let minX = CELL_SIZE;
  let minY = CELL_SIZE;
  let maxX = -1;
  let maxY = -1;
  const opaqueColors = new Set();
  for (let y = 0; y < CELL_SIZE; y += 1) {
    for (let x = 0; x < CELL_SIZE; x += 1) {
      const offset = (((y0 + y) * width) + x0 + x) * 4;
      const alpha = raw[offset + 3];
      if (alpha <= ALPHA_THRESHOLD) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      if (alpha >= 240) opaqueColors.add(`${raw[offset]},${raw[offset + 1]},${raw[offset + 2]}`);
    }
  }
  return {
    bounds: maxX >= minX && maxY >= minY
      ? { width: maxX - minX + 1, height: maxY - minY + 1 }
      : null,
    opaqueColors
  };
}

function cellContainsColor(raw, width, row, column, value, tolerance) {
  const color = parseHexColor(value);
  if (!color) return false;
  const delta = Math.max(0, Number(tolerance || 0));
  const x0 = column * CELL_SIZE;
  const y0 = row * CELL_SIZE;
  for (let y = 0; y < CELL_SIZE; y += 1) {
    for (let x = 0; x < CELL_SIZE; x += 1) {
      const offset = (((y0 + y) * width) + x0 + x) * 4;
      if (raw[offset + 3] <= ALPHA_THRESHOLD) continue;
      if (Math.abs(raw[offset] - color.r) <= delta &&
        Math.abs(raw[offset + 1] - color.g) <= delta &&
        Math.abs(raw[offset + 2] - color.b) <= delta) return true;
    }
  }
  return false;
}

function validateStarfrontArtDirection(item, decoded, angles) {
  if (!isStarfrontEquipment(item)) return;
  const neutralColumn = angles.reduce((bestIndex, angle, index) => (
    Math.abs(Number(angle || 0)) < Math.abs(Number(angles[bestIndex] || 0)) ? index : bestIndex
  ), 0);
  const stats = getCellVisualStats(decoded.data, decoded.info.width, 0, neutralColumn);
  if (!stats.bounds) throw new Error(`${item.id} Starfront profile has no neutral silhouette`);
  if (stats.opaqueColors.size < MIN_STARFRONT_PALETTE_COLORS) {
    throw new Error(`${item.id} Starfront profile uses ${stats.opaqueColors.size} opaque colors; expected at least ${MIN_STARFRONT_PALETTE_COLORS}`);
  }
  ['accent', 'ember'].forEach((key) => {
    if (!item[key] || !cellContainsColor(decoded.data, decoded.info.width, 0, neutralColumn, item[key], 3)) {
      throw new Error(`${item.id} Starfront profile should render its ${key} palette color`);
    }
  });
  const neutralExtent = Math.max(stats.bounds.width, stats.bounds.height);
  if (item.maxNeutralExtent && neutralExtent > Number(item.maxNeutralExtent)) {
    throw new Error(`${item.id} neutral silhouette is ${neutralExtent}px; expected no more than ${item.maxNeutralExtent}px`);
  }
  if (item.minNeutralExtent && neutralExtent < Number(item.minNeutralExtent)) {
    throw new Error(`${item.id} neutral silhouette is ${neutralExtent}px; expected at least ${item.minNeutralExtent}px`);
  }
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
  validateStarfrontArtDirection(item, decoded, angles);
}

function validateOutputFileCount() {
  const files = fs.existsSync(OUTPUT_DIR)
    ? fs.readdirSync(OUTPUT_DIR).filter((file) => /-atlas(?:-v\d+)?\.png$/i.test(file))
    : [];
  const expected = new Set(EQUIPMENT.map((item) => `${item.fileId}-atlas-${ATLAS_VERSION}.png`));
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
  ATLAS_VERSION,
  BOW_STATES,
  CELL_SIZE,
  EQUIPMENT,
  RECOGNIZED_KINDS,
  STARFRONT_PROFILE_ID,
  atlasAngles,
  atlasPath,
  isStarfrontEquipment,
  main,
  makeAtlas,
  validateAtlas
});
