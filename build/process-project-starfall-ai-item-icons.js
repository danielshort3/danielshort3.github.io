#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const {
  GUIDE_LINE_HEX,
  detectGuideGrid,
  getGridCellRect
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const STARFALL_ROOT = path.join(ROOT, 'img/project-starfall');
const SOURCE_DIR = path.join(STARFALL_ROOT, 'items/source');
const OUTPUT_DIR = path.join(STARFALL_ROOT, 'items/sheets');
const BACKUP_ROOT = path.join(STARFALL_ROOT, 'backups/procedural');
const ICON_SIZE = 64;
const KEY_COLOR = Object.freeze({ r: 0, g: 255, b: 0 });
const SOURCE_COMPONENT_MIN_AREA = 80;
const FINAL_COMPONENT_MIN_AREA = 10;
const RATE_COUPON_OUTPUT_FILE = 'ai-items-rate-coupons-sheet.png';
const POTION_TIER_SOURCE_FILE = 'ai-items-potion-tiers.png';
const POTION_TIER_OUTPUT_FILE = 'ai-items-potion-tiers-sheet.png';
const RATE_COUPON_ITEMS = Object.freeze([
  Object.freeze({ id: 'xp_coupon_1_2_1h', type: 'xp', tier: 0, main: '#54a5ff', accent: '#cbe8ff', foil: '#c98b45' }),
  Object.freeze({ id: 'xp_coupon_1_5_1h', type: 'xp', tier: 1, main: '#6f7dff', accent: '#e1e7ff', foil: '#d7dde7' }),
  Object.freeze({ id: 'xp_coupon_2_0_1h', type: 'xp', tier: 2, main: '#9d62ff', accent: '#fff0a6', foil: '#f6c44f' }),
  Object.freeze({ id: 'drop_coupon_1_2_1h', type: 'drop', tier: 0, main: '#52c77a', accent: '#dbffe6', foil: '#c98b45' }),
  Object.freeze({ id: 'drop_coupon_1_5_1h', type: 'drop', tier: 1, main: '#20bda2', accent: '#b8fff2', foil: '#d7dde7' }),
  Object.freeze({ id: 'drop_coupon_2_0_1h', type: 'drop', tier: 2, main: '#20b6ff', accent: '#fff0a6', foil: '#f6c44f' })
]);
const POTION_TIER_ITEMS = Object.freeze([
  Object.freeze({ id: 'minor_health_potion', kind: 'hp', tier: 0, main: '#d94848', accent: '#ffd0c6', metal: '#b98648' }),
  Object.freeze({ id: 'standard_health_potion', kind: 'hp', tier: 1, main: '#e84848', accent: '#ffe1d6', metal: '#c9d3dc' }),
  Object.freeze({ id: 'greater_health_potion', kind: 'hp', tier: 2, main: '#ff4b58', accent: '#fff0a6', metal: '#f0c85a' }),
  Object.freeze({ id: 'superior_health_potion', kind: 'hp', tier: 3, main: '#ff375f', accent: '#fff4d5', metal: '#ffe083' }),
  Object.freeze({ id: 'minor_resource_tonic', kind: 'mp', tier: 0, main: '#4f80ff', accent: '#d6edff', metal: '#b98648' }),
  Object.freeze({ id: 'standard_resource_tonic', kind: 'mp', tier: 1, main: '#6e70ff', accent: '#e5e9ff', metal: '#c9d3dc' }),
  Object.freeze({ id: 'greater_resource_tonic', kind: 'mp', tier: 2, main: '#8662ff', accent: '#fff0a6', metal: '#f0c85a' }),
  Object.freeze({ id: 'superior_resource_tonic', kind: 'mp', tier: 3, main: '#a349ff', accent: '#fff4d5', metal: '#ffe083' }),
  Object.freeze({ id: 'camp_ration', kind: 'ration', tier: 0, main: '#b88448', accent: '#ffe6b8', metal: '#78603f' }),
  Object.freeze({ id: 'field_ration', kind: 'ration', tier: 1, main: '#c89451', accent: '#fff0cf', metal: '#9fa8a8' }),
  Object.freeze({ id: 'expedition_ration', kind: 'ration', tier: 2, main: '#d39a4f', accent: '#fff0a6', metal: '#d8bc68' }),
  Object.freeze({ id: 'hero_ration', kind: 'ration', tier: 3, main: '#e5a952', accent: '#fff4d5', metal: '#ffe083' })
]);

function rel(filePath) {
  return path.relative(ROOT, filePath).replace(/\\/g, '/');
}

function backupPathFor(filePath) {
  return path.join(BACKUP_ROOT, path.relative(STARFALL_ROOT, filePath));
}

function ensureFileDir(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function backupOutput(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const backupPath = backupPathFor(filePath);
  if (fs.existsSync(backupPath)) return false;
  ensureFileDir(backupPath);
  fs.copyFileSync(filePath, backupPath);
  return true;
}

function ensureRuntimeBackup(filePath) {
  if (!fs.existsSync(filePath)) return false;
  const backupPath = backupPathFor(filePath);
  if (fs.existsSync(backupPath)) return false;
  ensureFileDir(backupPath);
  fs.copyFileSync(filePath, backupPath);
  return true;
}

const SHEETS = Object.freeze([
  Object.freeze({
    source: 'ai-items-consumables-materials.png',
    cols: 5,
    rows: 5,
    outputRows: 4,
    items: Object.freeze([
      'coins',
      Object.freeze({ id: 'town_return_scroll', sourceIndex: 3 }),
      Object.freeze({ id: 'guard_tonic', sourceIndex: 5 }),
      Object.freeze({ id: 'swiftstep_oil', sourceIndex: 6 }),
      Object.freeze({ id: 'magnet_charm', sourceIndex: 7 }),
      Object.freeze({ id: 'pet_whistle', sourceIndex: 8 }),
      Object.freeze({ id: 'cube_fragment', sourceIndex: 14 }),
      Object.freeze({ id: 'base_skill_manual', sourceIndex: 15 }),
      Object.freeze({ id: 'advanced_skill_manual', sourceIndex: 16 }),
      Object.freeze({ id: 'skill_reset_scroll', sourceIndex: 17 }),
      Object.freeze({ id: 'admin_worldwright_console', sourceIndex: 18 }),
      Object.freeze({ id: 'upgrade_dust', sourceIndex: 19 }),
      Object.freeze({ id: 'upgrade_catalyst', sourceIndex: 20 }),
      Object.freeze({ id: 'warding_scroll', sourceIndex: 21 }),
      Object.freeze({ id: 'refinement_core', sourceIndex: 22 }),
      Object.freeze({ id: 'gel_drop', sourceIndex: 23 }),
      Object.freeze({ id: 'ore_chunks', sourceIndex: 24 })
    ])
  }),
  Object.freeze({
    source: POTION_TIER_SOURCE_FILE,
    cols: 4,
    rows: 3,
    items: Object.freeze(POTION_TIER_ITEMS.map((item) => item.id))
  }),
  Object.freeze({
    source: 'ai-items-mob-materials-core.png',
    cols: 5,
    rows: 4,
    maxComponents: 18,
    items: Object.freeze([
      'dew_bead',
      'moss_hide',
      'thorn_fiber',
      'vine_fiber',
      'bristle_hide',
      'briar_antler',
      'dust_claw',
      'clockwork_scrap',
      'charged_coil',
      'scrap_plate',
      'ember_dust',
      'ash_carapace',
      'molten_fang',
      'cinder_gland',
      'bandit_cloth',
      'throwing_knife_scrap',
      'glow_spores',
      'bramble_crown',
      'titan_core',
      'colossus_ore'
    ])
  }),
  Object.freeze({
    source: 'ai-items-mob-materials-late.png',
    cols: 5,
    rows: 4,
    maxComponents: 18,
    items: Object.freeze([
      'emberjaw_badge',
      'rime_shard',
      'frozen_hide',
      'glacier_core',
      'snowglare_dust',
      'icebloom_petal',
      'gale_feather',
      'storm_fletching',
      'thunder_horn',
      'cloud_silk',
      'runic_page',
      'lumen_plate',
      'void_dust',
      'eclipse_silk',
      'rift_splinter',
      'rimewarden_sigil',
      'stormbreak_plume',
      'archivist_index',
      'sovereign_corona'
    ])
  }),
  Object.freeze({
    source: 'ai-items-coin-stacks.png',
    cols: 4,
    rows: 1,
    grid: 'uniform',
    background: 'checker',
    allowEdgeArt: true,
    rects: Object.freeze([
      Object.freeze({ x: 20, y: 118, w: 430, h: 520 }),
      Object.freeze({ x: 450, y: 130, w: 540, h: 520 }),
      Object.freeze({ x: 930, y: 90, w: 590, h: 560 }),
      Object.freeze({ x: 1470, y: 60, w: 670, h: 610 })
    ]),
    items: Object.freeze([
      'coins_small',
      'coins_medium',
      'coins_large',
      'coins_huge'
    ])
  }),
  Object.freeze({
    source: 'ai-items-rate-coupons.png',
    cols: 3,
    rows: 2,
    items: Object.freeze(RATE_COUPON_ITEMS.map((item) => item.id))
  }),
  Object.freeze({
    source: 'ai-items-slot-prisms-plinko.png',
    cols: 3,
    rows: 3,
    items: Object.freeze([
      'equipment_slot_coupon',
      'usable_slot_coupon',
      'etc_slot_coupon',
      'card_slot_coupon',
      'potential_cube',
      'preservation_cube',
      'plinko_ball_basic',
      'plinko_ball_polished',
      'plinko_ball_meteor'
    ])
  }),
  Object.freeze({
    source: 'ai-items-shop-boss-forest.png',
    cols: 5,
    rows: 5,
    items: Object.freeze([
      'training_sword',
      'training_wand',
      'training_bow',
      'copper_sword',
      'birch_wand',
      'simple_bow',
      'stitched_vest',
      'traveler_boots',
      'plain_ring',
      'iron_sword',
      'iron_axe',
      'apprentice_staff',
      'oak_longbow',
      'guardian_tower_shield',
      'berserker_war_grip',
      'ember_core',
      'rune_etched_focus',
      'deadeye_scope',
      'trap_kit',
      'thorncrown_greatsword',
      'thornroot_staff',
      'briarstring_longbow',
      'briar_crown',
      'barkplate_harness',
      'grasping_thorn_gloves'
    ])
  }),
  Object.freeze({
    source: 'ai-items-world-drops.png',
    cols: 5,
    rows: 4,
    grid: 'uniform',
    background: 'checker',
    allowEdgeArt: true,
    items: Object.freeze([
      'adventurer_cutlass',
      'balanced_focus',
      'wanderer_charm',
      'fieldguard_helm',
      'trailwoven_gloves',
      'vanguard_blade',
      'bulwark_plate',
      'breaker_gauntlets',
      'sentinel_greaves',
      'starglass_staff',
      'runewoven_robes',
      'channeler_gloves',
      'aetherstep_boots',
      'ranger_recurve',
      'pathfinder_leathers',
      'deadeye_wraps',
      'windrunner_boots'
    ])
  }),
  Object.freeze({
    source: 'ai-items-boss-core-storm.png',
    cols: 5,
    rows: 5,
    items: Object.freeze([
      'rootstep_greaves',
      'emberjaw_cleaver',
      'magma_scepter',
      'cindercoil_bow',
      'ashen_jaw_helm',
      'furnaceplate',
      'lavaforged_gauntlets',
      'scorchtrail_boots',
      'gearcleaver',
      'chrono_staff',
      'ratchet_repeater',
      'titan_visor',
      'clockplate_harness',
      'gyro_gauntlets',
      'springstep_boots',
      'colossus_maul',
      'geode_scepter',
      'oreline_greatbow',
      'deepcore_helm',
      'bedrock_plate',
      'quarry_fists',
      'stonewake_boots',
      'stormtalon_saber',
      'cloudspine_rod',
      'skybreaker_bow'
    ])
  }),
  Object.freeze({
    source: 'ai-items-boss-astral-eclipse.png',
    cols: 5,
    rows: 4,
    items: Object.freeze([
      'rocfeather_mask',
      'tempest_mantle',
      'lightning_grip_gloves',
      'gale_boots',
      'index_blade',
      'starbound_codex',
      'cometstring_bow',
      'archivist_crown',
      'astral_robes',
      'scribe_gloves',
      'orbit_boots',
      'eclipse_edge',
      'umbral_starstaff',
      'corona_longbow',
      'sovereign_crown',
      'eclipse_plate',
      'penumbra_gloves',
      'sunfall_boots'
    ])
  })
]);

function keyDistance(r, g, b) {
  const dr = r - KEY_COLOR.r;
  const dg = g - KEY_COLOR.g;
  const db = b - KEY_COLOR.b;
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function isChromaKeyPixel(r, g, b) {
  const distance = keyDistance(r, g, b);
  return distance <= 118 ||
    (g >= 220 && r <= 80 && b <= 80) ||
    (g >= 200 && g - Math.max(r, b) >= 132);
}

function isChromaFringePixel(r, g, b) {
  return g >= 150 && g - Math.max(r, b) >= 56 && keyDistance(r, g, b) <= 205;
}

function getAlphaBounds(raw, width, height, alphaThreshold) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (raw[offset + 3] <= alphaThreshold) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  return maxX >= minX && maxY >= minY
    ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }
    : null;
}

function removeChromaKey(raw, width, height) {
  const output = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i += 1) {
    const src = i * 3;
    const dst = i * 4;
    const r = raw[src];
    const g = raw[src + 1];
    const b = raw[src + 2];
    const key = isChromaKeyPixel(r, g, b);
    const fringe = !key && isChromaFringePixel(r, g, b);
    output[dst] = key ? 0 : r;
    output[dst + 1] = key ? 0 : fringe ? Math.min(g, Math.max(r, b) + 18) : g;
    output[dst + 2] = key ? 0 : b;
    output[dst + 3] = key ? 0 : 255;
  }
  return output;
}

function findAlphaComponents(raw, width, height, alphaThreshold) {
  const visited = new Uint8Array(width * height);
  const queue = [];
  const component = [];
  const components = [];
  const minAlpha = Number(alphaThreshold == null ? 0 : alphaThreshold);
  for (let start = 0; start < width * height; start += 1) {
    if (visited[start] || raw[start * 4 + 3] <= minAlpha) continue;
    queue.length = 0;
    component.length = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    visited[start] = 1;
    queue.push(start);
    for (let head = 0; head < queue.length; head += 1) {
      const index = queue[head];
      const x = index % width;
      const y = Math.floor(index / width);
      component.push(index);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          if (!dx && !dy) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const next = ny * width + nx;
          if (visited[next] || raw[next * 4 + 3] <= minAlpha) continue;
          visited[next] = 1;
          queue.push(next);
        }
      }
    }
    components.push({
      pixels: component.slice(),
      area: component.length,
      minX,
      minY,
      maxX,
      maxY,
      touchesEdge: minX <= 1 || minY <= 1 || maxX >= width - 2 || maxY >= height - 2
    });
  }
  return components.sort((a, b) => b.area - a.area);
}

function removeSmallAlphaComponents(raw, width, height, minArea) {
  for (let index = 0; index < width * height; index += 1) {
    if (raw[index * 4 + 3] <= 32) raw[index * 4 + 3] = 0;
  }
  findAlphaComponents(raw, width, height, 0).forEach((component) => {
    if (component.area >= minArea) return;
    component.pixels.forEach((index) => {
      raw[index * 4 + 3] = 0;
    });
  });
  return raw;
}

function removeSmallEdgeComponents(raw, width, height) {
  const components = findAlphaComponents(raw, width, height, 32);
  const largestArea = components.length ? components[0].area : 0;
  const maxEdgeFragmentArea = Math.max(400, largestArea * 0.16);
  components.forEach((component) => {
    if (!component.touchesEdge || component.area > maxEdgeFragmentArea) return;
    component.pixels.forEach((index) => {
      const offset = index * 4;
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    });
  });
  return raw;
}

function purgeVisibleChromaKeyRgba(raw, width, height) {
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (raw[offset + 3] <= 0) continue;
    const r = raw[offset];
    const g = raw[offset + 1];
    const b = raw[offset + 2];
    if (isChromaKeyPixel(r, g, b)) {
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    } else if (isChromaFringePixel(r, g, b)) {
      raw[offset + 1] = Math.min(g, Math.max(r, b) + 18);
    }
  }
  return raw;
}

function isCheckerBackgroundPixel(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const average = (r + g + b) / 3;
  return average >= 132 && max - min <= 70;
}

function removeCheckerBackground(raw, width, height) {
  const output = Buffer.from(raw);
  const visited = new Uint8Array(width * height);
  const queue = [];
  const addIfBackground = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const index = y * width + x;
    if (visited[index]) return;
    const offset = index * 4;
    if (output[offset + 3] <= 0 || !isCheckerBackgroundPixel(output[offset], output[offset + 1], output[offset + 2])) return;
    visited[index] = 1;
    queue.push(index);
  };
  for (let x = 0; x < width; x += 1) {
    addIfBackground(x, 0);
    addIfBackground(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    addIfBackground(0, y);
    addIfBackground(width - 1, y);
  }
  for (let head = 0; head < queue.length; head += 1) {
    const index = queue[head];
    const x = index % width;
    const y = Math.floor(index / width);
    const offset = index * 4;
    output[offset] = 0;
    output[offset + 1] = 0;
    output[offset + 2] = 0;
    output[offset + 3] = 0;
    addIfBackground(x + 1, y);
    addIfBackground(x - 1, y);
    addIfBackground(x, y + 1);
    addIfBackground(x, y - 1);
  }
  return output;
}

function validateCellComponents(raw, width, height, context, options) {
  const settings = options || {};
  const maxComponents = Math.max(1, Number(settings.maxComponents || 8));
  const components = findAlphaComponents(raw, width, height, 32)
    .filter((component) => component.area >= SOURCE_COMPONENT_MIN_AREA);
  if (!components.length) {
    throw new Error(`${context} has no visible item pixels after chroma-key cleanup`);
  }
  const edgeComponent = components.find((component) => component.touchesEdge);
  if (edgeComponent && !settings.allowEdgeArt) {
    throw new Error(`${context} has visible art touching a cell edge; regenerate the source sheet with more padding and no cross-cell spillover`);
  }
  if (components.length > maxComponents) {
    throw new Error(`${context} has ${components.length} separated visible components; regenerate with centered item art and no stray fragments`);
  }
}

function getSheetCellRect(sheet, row, col) {
  const manualRect = Array.isArray(sheet.rects) ? sheet.rects[row * sheet.cols + col] : null;
  if (manualRect) {
    return {
      x: Math.max(0, Math.round(Number(manualRect.x || 0))),
      y: Math.max(0, Math.round(Number(manualRect.y || 0))),
      w: Math.max(1, Math.round(Number(manualRect.w || 1))),
      h: Math.max(1, Math.round(Number(manualRect.h || 1)))
    };
  }
  if (sheet.grid === 'uniform') {
    const x = Math.round(col * sheet.width / sheet.cols);
    const y = Math.round(row * sheet.height / sheet.rows);
    const right = Math.round((col + 1) * sheet.width / sheet.cols);
    const bottom = Math.round((row + 1) * sheet.height / sheet.rows);
    return {
      x,
      y,
      w: Math.max(1, right - x),
      h: Math.max(1, bottom - y)
    };
  }
  return getGridCellRect(sheet.grid, row, col, 2);
}

async function extractCell(sheet, index) {
  const col = index % sheet.cols;
  const row = Math.floor(index / sheet.cols);
  const id = sheet.items[index];
  const context = `${sheet.source} ${id} row ${row + 1} col ${col + 1}`;
  const rect = getSheetCellRect(sheet, row, col);
  const extractor = sharp(sheet.sourcePath).extract({ left: rect.x, top: rect.y, width: rect.w, height: rect.h });
  const { data, info } = await (sheet.background === 'checker'
    ? extractor.ensureAlpha()
    : extractor.removeAlpha())
    .raw()
    .toBuffer({ resolveWithObject: true });
  const raw = sheet.background === 'checker'
    ? removeSmallEdgeComponents(removeSmallAlphaComponents(removeCheckerBackground(data, info.width, info.height), info.width, info.height, SOURCE_COMPONENT_MIN_AREA), info.width, info.height)
    : removeSmallAlphaComponents(removeChromaKey(data, info.width, info.height), info.width, info.height, SOURCE_COMPONENT_MIN_AREA);
  validateCellComponents(raw, info.width, info.height, context, {
    allowEdgeArt: !!sheet.allowEdgeArt,
    maxComponents: sheet.maxComponents
  });
  return {
    raw,
    width: info.width,
    height: info.height
  };
}

function getSheetOutputFile(source) {
  return `${path.basename(source, path.extname(source))}-sheet.png`;
}

function getSheetOutputColumns(sheet) {
  return Math.max(1, Math.floor(Number(sheet && (sheet.outputCols || sheet.cols) || 1)) || 1);
}

function getSheetOutputRows(sheet) {
  return Math.max(1, Math.floor(Number(sheet && (sheet.outputRows || sheet.rows) || 1)) || 1);
}

function getSheetItemId(entry) {
  if (typeof entry === 'string') return entry;
  return entry && typeof entry.id === 'string' ? entry.id : '';
}

function getSheetItemSourceIndex(entry, fallbackIndex) {
  if (entry && typeof entry === 'object' && Number.isFinite(Number(entry.sourceIndex))) {
    return Math.max(0, Math.floor(Number(entry.sourceIndex)));
  }
  return fallbackIndex;
}

function couponShape(tag, attributes, children) {
  const attrs = Object.entries(attributes || {})
    .filter(([, value]) => value !== null && value !== undefined && value !== false)
    .map(([key, value]) => `${key}="${String(value).replace(/"/g, '&quot;')}"`)
    .join(' ');
  const open = `<${tag}${attrs ? ` ${attrs}` : ''}`;
  return children ? `${open}>${children}</${tag}>` : `${open}/>`;
}

function drawPotionBottleIcon(icon) {
  const tier = Math.max(0, Math.floor(Number(icon.tier || 0)));
  const bodyWidth = 22 + tier * 3;
  const bodyHeight = 31 + tier * 2;
  const bodyX = Math.round((ICON_SIZE - bodyWidth) / 2);
  const bodyY = 22 - Math.min(4, tier);
  const neckWidth = Math.max(10, bodyWidth - 13);
  const neckX = Math.round((ICON_SIZE - neckWidth) / 2);
  const capY = bodyY - 9;
  const halo = tier * 2;
  return [
    couponShape('ellipse', { cx: 32, cy: 56, rx: 19 + halo, ry: 5, fill: '#122033', 'fill-opacity': 0.24 }),
    couponShape('rect', { x: neckX, y: capY + 6, width: neckWidth, height: 12, rx: 3, fill: icon.main, stroke: '#1c2b3a', 'stroke-width': 3 }),
    couponShape('rect', { x: neckX - 3, y: capY, width: neckWidth + 6, height: 8, rx: 3, fill: icon.metal, stroke: '#1c2b3a', 'stroke-width': 2 }),
    couponShape('path', {
      d: `M${bodyX + 5} ${bodyY + 6} C${bodyX + 1} ${bodyY + 13} ${bodyX} ${bodyY + bodyHeight - 7} ${bodyX + 6} ${bodyY + bodyHeight} L${bodyX + bodyWidth - 6} ${bodyY + bodyHeight} C${bodyX + bodyWidth} ${bodyY + bodyHeight - 7} ${bodyX + bodyWidth - 1} ${bodyY + 13} ${bodyX + bodyWidth - 5} ${bodyY + 6} Z`,
      fill: icon.main,
      stroke: '#1c2b3a',
      'stroke-width': 3,
      'stroke-linejoin': 'round'
    }),
    couponShape('path', {
      d: `M${bodyX + 7} ${bodyY + bodyHeight - 11} C${bodyX + 14} ${bodyY + bodyHeight - 5} ${bodyX + bodyWidth - 10} ${bodyY + bodyHeight - 6} ${bodyX + bodyWidth - 5} ${bodyY + bodyHeight - 13} L${bodyX + bodyWidth - 6} ${bodyY + bodyHeight - 4} L${bodyX + 6} ${bodyY + bodyHeight - 4} Z`,
      fill: icon.accent,
      'fill-opacity': 0.55
    }),
    couponShape('path', {
      d: `M${bodyX + 9} ${bodyY + 11} C${bodyX + 12} ${bodyY + 6} ${bodyX + 15} ${bodyY + 6} ${bodyX + 18} ${bodyY + 8}`,
      fill: 'none',
      stroke: '#ffffff',
      'stroke-width': 3,
      'stroke-linecap': 'round',
      'stroke-opacity': 0.74
    }),
    tier >= 1 ? couponShape('circle', { cx: bodyX + bodyWidth - 2, cy: bodyY + 12, r: 4 + tier, fill: icon.metal, stroke: icon.accent, 'stroke-width': 1 }) : '',
    tier >= 2 ? couponShape('path', { d: `M${bodyX + 2} ${bodyY + bodyHeight - 1} L${bodyX + 10} ${bodyY + bodyHeight + 7} L${bodyX + 14} ${bodyY + bodyHeight - 1} Z`, fill: icon.metal }) : '',
    tier >= 2 ? couponShape('path', { d: `M${bodyX + bodyWidth - 14} ${bodyY + bodyHeight - 1} L${bodyX + bodyWidth - 10} ${bodyY + bodyHeight + 7} L${bodyX + bodyWidth - 2} ${bodyY + bodyHeight - 1} Z`, fill: icon.metal }) : '',
    tier >= 3 ? couponShape('circle', { cx: 48, cy: 14, r: 4, fill: icon.accent, 'fill-opacity': 0.9 }) : '',
    tier >= 3 ? couponShape('circle', { cx: 16, cy: 48, r: 4, fill: icon.accent, 'fill-opacity': 0.78 }) : ''
  ].join('');
}

function drawRationIcon(icon) {
  const tier = Math.max(0, Math.floor(Number(icon.tier || 0)));
  const packY = 19 - Math.min(3, tier);
  const packWidth = 34 + tier * 3;
  const packHeight = 29 + tier;
  const packX = Math.round((ICON_SIZE - packWidth) / 2);
  const strapColor = tier >= 2 ? icon.metal : '#665341';
  return [
    couponShape('ellipse', { cx: 32, cy: 56, rx: 22 + tier, ry: 5, fill: '#122033', 'fill-opacity': 0.22 }),
    couponShape('path', {
      d: `M${packX + 5} ${packY + 7} C${packX + 9} ${packY - 1} ${packX + packWidth - 9} ${packY - 1} ${packX + packWidth - 5} ${packY + 7} L${packX + packWidth - 2} ${packY + packHeight - 5} C${packX + packWidth - 7} ${packY + packHeight + 3} ${packX + 7} ${packY + packHeight + 3} ${packX + 2} ${packY + packHeight - 5} Z`,
      fill: icon.main,
      stroke: '#1c2b3a',
      'stroke-width': 3,
      'stroke-linejoin': 'round'
    }),
    couponShape('path', {
      d: `M${packX + 7} ${packY + 12} C${packX + 16} ${packY + 17} ${packX + packWidth - 15} ${packY + 17} ${packX + packWidth - 7} ${packY + 12}`,
      fill: 'none',
      stroke: icon.accent,
      'stroke-width': 4,
      'stroke-linecap': 'round',
      'stroke-opacity': 0.82
    }),
    couponShape('rect', { x: packX + 5, y: packY + 24, width: packWidth - 10, height: 7, rx: 3, fill: strapColor, 'fill-opacity': 0.92 }),
    couponShape('circle', { cx: 32, cy: packY + 27, r: 5 + Math.min(2, tier), fill: icon.accent, stroke: '#1c2b3a', 'stroke-width': 2 }),
    couponShape('path', { d: `M${packX + 11} ${packY + 6} L${packX + 17} ${packY + 1} L${packX + 22} ${packY + 7}`, fill: 'none', stroke: strapColor, 'stroke-width': 3, 'stroke-linecap': 'round' }),
    couponShape('path', { d: `M${packX + packWidth - 22} ${packY + 7} L${packX + packWidth - 17} ${packY + 1} L${packX + packWidth - 11} ${packY + 6}`, fill: 'none', stroke: strapColor, 'stroke-width': 3, 'stroke-linecap': 'round' }),
    tier >= 1 ? couponShape('rect', { x: packX + packWidth - 10, y: packY + 15, width: 7, height: 10, rx: 2, fill: icon.accent, 'fill-opacity': 0.78 }) : '',
    tier >= 2 ? couponShape('path', { d: `M${packX - 1} ${packY + 20} L${packX - 7} ${packY + 14} L${packX - 3} ${packY + 28} Z`, fill: icon.metal }) : '',
    tier >= 3 ? couponShape('circle', { cx: 48, cy: 14, r: 4, fill: icon.accent, 'fill-opacity': 0.9 }) : ''
  ].join('');
}

function drawPotionTierIcon(icon) {
  return icon.kind === 'ration' ? drawRationIcon(icon) : drawPotionBottleIcon(icon);
}

function makePotionTierSheetSvg() {
  const body = POTION_TIER_ITEMS.map((icon, index) => {
    const x = (index % 4) * ICON_SIZE;
    const y = Math.floor(index / 4) * ICON_SIZE;
    return `<g transform="translate(${x} ${y})">${drawPotionTierIcon(icon)}</g>`;
  }).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${ICON_SIZE * 4}" height="${ICON_SIZE * 3}" viewBox="0 0 ${ICON_SIZE * 4} ${ICON_SIZE * 3}">${body}</svg>`;
}

async function writePotionTierBackupSheet(destination) {
  const backupPath = backupPathFor(destination);
  if (fs.existsSync(backupPath)) return false;
  ensureFileDir(backupPath);
  await sharp(Buffer.from(makePotionTierSheetSvg()))
    .png({ compressionLevel: 9 })
    .toFile(backupPath);
  return true;
}

function drawCouponSparkles(icon) {
  const sparkCount = 2 + icon.tier;
  const positions = [
    [12, 10],
    [48, 49],
    [18, 48],
    [49, 12]
  ];
  return Array.from({ length: sparkCount }, (_, index) => {
    const position = positions[index % positions.length];
    const x = position[0];
    const y = position[1];
    return [
      couponShape('rect', { x: x + 2, y, width: 2, height: 9, fill: icon.accent, 'fill-opacity': 0.9 }),
      couponShape('rect', { x: x - 1, y: y + 3, width: 8, height: 2, fill: icon.accent, 'fill-opacity': 0.9 })
    ].join('');
  }).join('');
}

function drawXpCouponIcon(icon) {
  const halo = icon.tier * 3;
  const ribbonY = 46 - icon.tier;
  return [
    couponShape('ellipse', { cx: 32, cy: 56, rx: 24 + halo, ry: 5, fill: '#122033', 'fill-opacity': 0.22 }),
    couponShape('rect', { x: 13, y: 13, width: 38, height: 42, rx: 5, fill: '#14263a' }),
    couponShape('rect', { x: 17, y: 16, width: 30, height: 36, rx: 4, fill: icon.main }),
    couponShape('rect', { x: 20, y: 20, width: 24, height: 6, rx: 2, fill: icon.accent, 'fill-opacity': 0.72 }),
    couponShape('path', { d: 'M32 27 L36 35 L45 36 L38 42 L40 51 L32 46 L24 51 L26 42 L19 36 L28 35 Z', fill: icon.foil, stroke: icon.accent, 'stroke-width': icon.tier >= 2 ? 2 : 1 }),
    couponShape('path', { d: `M20 ${ribbonY} L44 ${ribbonY} L39 ${ribbonY + 7} L25 ${ribbonY + 7} Z`, fill: icon.foil, 'fill-opacity': 0.92 }),
    couponShape('path', { d: `M28 ${38 - icon.tier * 2} L32 ${32 - icon.tier * 3} L36 ${38 - icon.tier * 2} M32 ${32 - icon.tier * 3} L32 44`, fill: 'none', stroke: icon.accent, 'stroke-width': 3, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-opacity': 0.95 }),
    icon.tier >= 1 ? couponShape('circle', { cx: 48, cy: 14, r: 5 + icon.tier, fill: icon.foil, 'fill-opacity': 0.74 }) : '',
    icon.tier >= 2 ? couponShape('circle', { cx: 16, cy: 48, r: 5, fill: icon.accent, 'fill-opacity': 0.82 }) : '',
    drawCouponSparkles(icon)
  ].join('');
}

function drawDropCouponIcon(icon) {
  const halo = icon.tier * 2;
  return [
    couponShape('ellipse', { cx: 32, cy: 56, rx: 24 + halo, ry: 5, fill: '#122033', 'fill-opacity': 0.22 }),
    couponShape('rect', { x: 14, y: 15, width: 36, height: 39, rx: 6, fill: '#102b31' }),
    couponShape('path', { d: 'M21 26 C23 14 41 14 43 26 L49 48 C43 56 21 56 15 48 Z', fill: icon.main, stroke: '#122033', 'stroke-width': 3, 'stroke-linejoin': 'round' }),
    couponShape('rect', { x: 22, y: 24, width: 20, height: 5, rx: 2, fill: icon.foil }),
    couponShape('circle', { cx: 30, cy: 42, r: 9, fill: icon.foil }),
    couponShape('path', { d: 'M27 42 L31 37 L35 42 L31 47 Z', fill: icon.accent, 'fill-opacity': 0.95 }),
    couponShape('circle', { cx: 44, cy: 39, r: 6 + icon.tier, fill: icon.accent, 'fill-opacity': 0.78 }),
    couponShape('circle', { cx: 17, cy: 35, r: 4 + icon.tier, fill: icon.foil, 'fill-opacity': 0.82 }),
    icon.tier >= 1 ? couponShape('path', { d: 'M47 20 L54 13 L53 24 Z', fill: icon.accent, 'fill-opacity': 0.82 }) : '',
    icon.tier >= 2 ? couponShape('path', { d: 'M11 22 L19 10 L20 27 Z', fill: icon.accent, 'fill-opacity': 0.76 }) : '',
    drawCouponSparkles(icon)
  ].join('');
}

function makeRateCouponSheetSvg() {
  const body = RATE_COUPON_ITEMS.map((icon, index) => {
    const x = (index % 3) * ICON_SIZE;
    const y = Math.floor(index / 3) * ICON_SIZE;
    const art = icon.type === 'drop' ? drawDropCouponIcon(icon) : drawXpCouponIcon(icon);
    return `<g transform="translate(${x} ${y})">${art}</g>`;
  }).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${ICON_SIZE * 3}" height="${ICON_SIZE * 2}" viewBox="0 0 ${ICON_SIZE * 3} ${ICON_SIZE * 2}">${body}</svg>`;
}

async function writeRateCouponSheet(assets) {
  RATE_COUPON_ITEMS.forEach((item) => {
    if (!assets[item.id]) throw new Error(`Missing ITEM_ASSETS mapping for ${item.id}`);
  });
  const destination = path.join(OUTPUT_DIR, RATE_COUPON_OUTPUT_FILE);
  backupOutput(destination);
  ensureFileDir(destination);
  await sharp(Buffer.from(makeRateCouponSheetSvg()))
    .png({ compressionLevel: 9 })
    .toFile(destination);
  ensureRuntimeBackup(destination);
  return rel(destination);
}

async function renderIcon(cell, context) {
  const bounds = getAlphaBounds(cell.raw, cell.width, cell.height, 16);
  if (!bounds) throw new Error(`No visible item pixels after chroma key removal for ${context}`);
  const padding = 8;
  const cropLeft = Math.max(0, bounds.x - padding);
  const cropTop = Math.max(0, bounds.y - padding);
  const cropRight = Math.min(cell.width, bounds.x + bounds.w + padding);
  const cropBottom = Math.min(cell.height, bounds.y + bounds.h + padding);
  const cropWidth = Math.max(1, cropRight - cropLeft);
  const cropHeight = Math.max(1, cropBottom - cropTop);
  const fitSize = Math.max(1, Math.round(ICON_SIZE * 0.84));
  const transparent = { r: 255, g: 255, b: 255, alpha: 0 };
  const itemBuffer = await sharp(cell.raw, {
    raw: { width: cell.width, height: cell.height, channels: 4 }
  })
    .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
    .resize(fitSize, fitSize, {
      fit: 'inside',
      kernel: 'lanczos3',
      background: transparent
    })
    .png()
    .toBuffer();
  const itemMeta = await sharp(itemBuffer).metadata();
  const composed = await sharp({
    create: {
      width: ICON_SIZE,
      height: ICON_SIZE,
      channels: 4,
      background: transparent
    }
  })
    .composite([{
      input: itemBuffer,
      left: Math.round((ICON_SIZE - itemMeta.width) / 2),
      top: Math.round((ICON_SIZE - itemMeta.height) / 2)
    }])
    .png({ compressionLevel: 9 })
    .toBuffer();
  const { data, info } = await sharp(composed)
    .raw()
    .toBuffer({ resolveWithObject: true });
  const cleaned = await normalizeIconFill(
    removeSmallAlphaComponents(purgeVisibleChromaKeyRgba(data, info.width, info.height), info.width, info.height, FINAL_COMPONENT_MIN_AREA),
    info.width,
    info.height
  );
  return sharp(cleaned, {
    raw: { width: info.width, height: info.height, channels: 4 }
  })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function normalizeIconFill(raw, width, height) {
  const bounds = getAlphaBounds(raw, width, height, 16);
  if (!bounds) return raw;
  const maxVisible = Math.max(bounds.w, bounds.h);
  const minimumVisible = Math.round(ICON_SIZE * 0.76);
  if (maxVisible >= minimumVisible) return raw;
  const cropPadding = 2;
  const cropLeft = Math.max(0, bounds.x - cropPadding);
  const cropTop = Math.max(0, bounds.y - cropPadding);
  const cropRight = Math.min(width, bounds.x + bounds.w + cropPadding);
  const cropBottom = Math.min(height, bounds.y + bounds.h + cropPadding);
  const cropWidth = Math.max(1, cropRight - cropLeft);
  const cropHeight = Math.max(1, cropBottom - cropTop);
  const targetMax = Math.min(Math.round(ICON_SIZE * 0.86), Math.max(minimumVisible, maxVisible));
  const scale = targetMax / Math.max(1, maxVisible);
  const resizedWidth = Math.max(1, Math.round(cropWidth * scale));
  const resizedHeight = Math.max(1, Math.round(cropHeight * scale));
  const transparent = { r: 255, g: 255, b: 255, alpha: 0 };
  const resized = await sharp(raw, {
    raw: { width, height, channels: 4 }
  })
    .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
    .resize(resizedWidth, resizedHeight, {
      fit: 'fill',
      kernel: 'lanczos3',
      background: transparent
    })
    .png()
    .toBuffer();
  const composed = await sharp({
    create: {
      width,
      height,
      channels: 4,
      background: transparent
    }
  })
    .composite([{
      input: resized,
      left: Math.round((width - resizedWidth) / 2),
      top: Math.round((height - resizedHeight) / 2)
    }])
    .raw()
    .toBuffer();
  return removeSmallAlphaComponents(purgeVisibleChromaKeyRgba(composed, width, height), width, height, FINAL_COMPONENT_MIN_AREA);
}

async function main() {
  const data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
  const assets = data.ITEM_ASSETS || {};
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const written = [];
  for (const sheet of SHEETS) {
    const sourcePath = path.join(SOURCE_DIR, sheet.source);
    if (!fs.existsSync(sourcePath)) throw new Error(`Missing AI item source sheet: ${path.relative(ROOT, sourcePath)}`);
    const { data, info } = await sharp(sourcePath)
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const grid = sheet.grid === 'uniform'
      ? 'uniform'
      : detectGuideGrid(data, info.width, info.height, {
          columns: sheet.cols,
          rows: sheet.rows,
          label: `AI item icon sheet ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}`,
          guideColor: GUIDE_LINE_HEX
        });
    const sheetRuntime = Object.assign({}, sheet, {
      sourcePath,
      width: info.width,
      height: info.height,
      grid
    });
    const composites = [];
    const outputCols = getSheetOutputColumns(sheet);
    for (let index = 0; index < sheet.items.length; index += 1) {
      const entry = sheet.items[index];
      const id = getSheetItemId(entry);
      const sourceIndex = getSheetItemSourceIndex(entry, index);
      const assetPath = assets[id];
      if (!assetPath) throw new Error(`Missing ITEM_ASSETS mapping for ${id}`);
      const cell = await extractCell(sheetRuntime, sourceIndex);
      const icon = await renderIcon(cell, id);
      composites.push({
        input: icon,
        left: (index % outputCols) * ICON_SIZE,
        top: Math.floor(index / outputCols) * ICON_SIZE
      });
    }
    const outputFile = getSheetOutputFile(sheet.source);
    const destination = path.join(OUTPUT_DIR, outputFile);
    backupOutput(destination);
    ensureFileDir(destination);
    await sharp({
      create: {
        width: outputCols * ICON_SIZE,
        height: getSheetOutputRows(sheet) * ICON_SIZE,
        channels: 4,
        background: { r: 255, g: 255, b: 255, alpha: 0 }
      }
    })
      .composite(composites)
      .png({ compressionLevel: 9 })
      .toFile(destination);
    if (sheet.source === POTION_TIER_SOURCE_FILE) {
      await writePotionTierBackupSheet(destination);
    } else {
      ensureRuntimeBackup(destination);
    }
    written.push(rel(destination));
  }
  const coveredIds = new Set(SHEETS.reduce((ids, sheet) => {
    (sheet.items || []).forEach((entry) => {
      const id = getSheetItemId(entry);
      if (id) ids.push(id);
    });
    return ids;
  }, []));
  RATE_COUPON_ITEMS.forEach((item) => coveredIds.add(item.id));
  coveredIds.add('stat_reset_scroll');
  const uniqueWritten = new Set(written);
  const missing = Object.keys(assets)
    .filter((id) => !coveredIds.has(id));
  if (missing.length) {
    throw new Error(`AI item sheets did not cover ITEM_ASSETS ids: ${missing.join(', ')}`);
  }
  const missingSheets = Object.entries(assets)
    .filter(([, assetPath]) => String(assetPath || '').startsWith('img/project-starfall/items/sheets/'))
    .filter(([, assetPath]) => !uniqueWritten.has(String(assetPath).split('#')[0]))
    .map(([id]) => id);
  if (missingSheets.length) {
    throw new Error(`ITEM_ASSETS references sheets that were not generated: ${missingSheets.join(', ')}`);
  }
  written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
