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
const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/items/source');
const OUTPUT_DIR = path.join(ROOT, 'img/project-starfall/items');
const ICON_SIZE = 64;
const KEY_COLOR = Object.freeze({ r: 0, g: 255, b: 0 });
const SOURCE_COMPONENT_MIN_AREA = 80;
const FINAL_COMPONENT_MIN_AREA = 10;

const SHEETS = Object.freeze([
  Object.freeze({
    source: 'ai-items-consumables-materials.png',
    cols: 5,
    rows: 5,
    items: Object.freeze([
      'coins',
      'minor_health_potion',
      'minor_resource_tonic',
      'town_return_scroll',
      'camp_ration',
      'guard_tonic',
      'swiftstep_oil',
      'magnet_charm',
      'pet_whistle',
      'equipment_slot_coupon',
      'usable_slot_coupon',
      'etc_slot_coupon',
      'potential_cube',
      'preservation_cube',
      'cube_fragment',
      'base_skill_manual',
      'advanced_skill_manual',
      'skill_reset_scroll',
      'admin_worldwright_console',
      'upgrade_dust',
      'upgrade_catalyst',
      'warding_scroll',
      'refinement_core',
      'gel_drop',
      'ore_chunks'
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

function validateCellComponents(raw, width, height, context) {
  const components = findAlphaComponents(raw, width, height, 32)
    .filter((component) => component.area >= SOURCE_COMPONENT_MIN_AREA);
  if (!components.length) {
    throw new Error(`${context} has no visible item pixels after chroma-key cleanup`);
  }
  const edgeComponent = components.find((component) => component.touchesEdge);
  if (edgeComponent) {
    throw new Error(`${context} has visible art touching a cell edge; regenerate the source sheet with more padding and no cross-cell spillover`);
  }
  if (components.length > 8) {
    throw new Error(`${context} has ${components.length} separated visible components; regenerate with one centered item and no stray fragments`);
  }
}

async function extractCell(sheet, index) {
  const col = index % sheet.cols;
  const row = Math.floor(index / sheet.cols);
  const id = sheet.items[index];
  const context = `${sheet.source} ${id} row ${row + 1} col ${col + 1}`;
  const rect = getGridCellRect(sheet.grid, row, col, 2);
  const { data, info } = await sharp(sheet.sourcePath)
    .extract({ left: rect.x, top: rect.y, width: rect.w, height: rect.h })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const raw = removeSmallAlphaComponents(removeChromaKey(data, info.width, info.height), info.width, info.height, SOURCE_COMPONENT_MIN_AREA);
  validateCellComponents(raw, info.width, info.height, context);
  return {
    raw,
    width: info.width,
    height: info.height
  };
}

async function writeIcon(cell, destination) {
  const bounds = getAlphaBounds(cell.raw, cell.width, cell.height, 16);
  if (!bounds) throw new Error(`No visible item pixels after chroma key removal for ${destination}`);
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
  await sharp(cleaned, {
    raw: { width: info.width, height: info.height, channels: 4 }
  })
    .png({ compressionLevel: 9 })
    .toFile(destination);
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
  const data = require(path.join(ROOT, 'js/project-starfall/project-starfall-data.js'));
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
    const grid = detectGuideGrid(data, info.width, info.height, {
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
    for (let index = 0; index < sheet.items.length; index += 1) {
      const id = sheet.items[index];
      const assetPath = assets[id];
      if (!assetPath) throw new Error(`Missing ITEM_ASSETS mapping for ${id}`);
      const destination = path.join(ROOT, assetPath);
      const cell = await extractCell(sheetRuntime, index);
      await writeIcon(cell, destination);
      written.push(path.relative(ROOT, destination).replace(/\\/g, '/'));
    }
  }
  const uniqueWritten = new Set(written);
  const missing = Object.entries(assets)
    .filter(([, assetPath]) => assetPath.startsWith('img/project-starfall/items/'))
    .filter(([, assetPath]) => !uniqueWritten.has(assetPath))
    .map(([id]) => id);
  if (missing.length) {
    throw new Error(`AI item sheets did not cover ITEM_ASSETS ids: ${missing.join(', ')}`);
  }
  written.forEach((file) => process.stdout.write(`Processed ${file}\n`));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
