#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/environment/source');
const TERRAIN_DIR = path.join(ROOT, 'img/project-starfall/environment/terrain');
const PROP_DIR = path.join(ROOT, 'img/project-starfall/environment/props');
const SOURCE_PATH = path.join(SOURCE_DIR, 'imagegen-environment-source.png');
const KEYED_SOURCE_PATH = path.join(SOURCE_DIR, 'imagegen-environment-source-keyed.png');
const CELL = 64;

const MAP_THEME_IDS = Object.freeze([
  'starfall-crossing',
  'rustcoil-outpost',
  'cinder-refuge',
  'frostfen-camp',
  'stormbreak-haven',
  'astral-observatory',
  'greenroot-meadow',
  'thornpath-thicket',
  'bramble-depths',
  'rustcoil-ruins',
  'gearworks-vault',
  'cinder-hollow',
  'emberjaw-lair',
  'bandit-ridge-camp',
  'oreback-quarry',
  'ashglass-pass',
  'frostfen-outskirts',
  'glacier-spine',
  'rimewarden-sanctum',
  'stormbreak-cliffs',
  'astral-archive',
  'eclipse-frontier',
  'endless-rift',
  'brambleking-court',
  'titan-foundry',
  'deepcore-core',
  'emberjaw-furnace',
  'rimewarden-vault',
  'stormbreak-aerie',
  'astral-stacks',
  'eclipse-throne'
]);

const TRIAL_THEME_IDS = Object.freeze([
  'guardian-trial',
  'berserker-trial',
  'duelist-trial',
  'fire-mage-trial',
  'rune-mage-trial',
  'storm-mage-trial',
  'sniper-trial',
  'trapper-trial',
  'beast-archer-trial'
]);

const THEME_IDS = Object.freeze([...MAP_THEME_IDS, ...TRIAL_THEME_IDS]);

const TERRAIN_SOURCES = Object.freeze({
  town: { x: 22, y: 18, w: 676, h: 124 },
  meadow: { x: 18, y: 162, w: 650, h: 126 },
  thorn: { x: 18, y: 304, w: 645, h: 94 },
  gearworks: { x: 18, y: 398, w: 680, h: 88 },
  cinder: { x: 18, y: 480, w: 678, h: 92 },
  frost: { x: 18, y: 578, w: 678, h: 90 },
  ruins: { x: 18, y: 676, w: 620, h: 92 },
  astral: { x: 18, y: 758, w: 650, h: 108 },
  rift: { x: 18, y: 862, w: 660, h: 90 },
  quarry: { x: 18, y: 946, w: 670, h: 78 }
});

const PROP_SOURCES = Object.freeze({
  town: {
    grass: { x: 1100, y: 82, w: 108, h: 58 },
    bush: { x: 1268, y: 82, w: 86, h: 64 },
    tree: { x: 1448, y: 16, w: 76, h: 104 },
    rock: { x: 1450, y: 106, w: 86, h: 64 },
    flower: { x: 1182, y: 84, w: 80, h: 54 },
    small: { x: 736, y: 54, w: 58, h: 98 },
    tall: { x: 800, y: 58, w: 70, h: 94 },
    crate: { x: 930, y: 50, w: 70, h: 78 },
    crystal: { x: 1478, y: 122, w: 48, h: 64 },
    vine: { x: 1210, y: 40, w: 74, h: 92 },
    sign: { x: 1164, y: 70, w: 86, h: 62 },
    glow: { x: 1388, y: 40, w: 80, h: 96 }
  },
  meadow: {
    grass: { x: 1100, y: 160, w: 80, h: 70 },
    bush: { x: 850, y: 148, w: 72, h: 74 },
    tree: { x: 692, y: 122, w: 96, h: 132 },
    rock: { x: 944, y: 166, w: 58, h: 54 },
    flower: { x: 1170, y: 150, w: 72, h: 80 },
    small: { x: 895, y: 154, w: 64, h: 82 },
    tall: { x: 780, y: 112, w: 86, h: 136 },
    crate: { x: 1266, y: 162, w: 56, h: 60 },
    crystal: { x: 1168, y: 150, w: 78, h: 82 },
    vine: { x: 1010, y: 136, w: 72, h: 96 },
    sign: { x: 1455, y: 162, w: 66, h: 72 },
    glow: { x: 1185, y: 164, w: 60, h: 58 }
  },
  thorn: {
    grass: { x: 1015, y: 280, w: 90, h: 74 },
    bush: { x: 900, y: 266, w: 90, h: 94 },
    tree: { x: 700, y: 252, w: 76, h: 128 },
    rock: { x: 1110, y: 274, w: 92, h: 78 },
    flower: { x: 1184, y: 278, w: 78, h: 82 },
    small: { x: 1210, y: 274, w: 70, h: 86 },
    tall: { x: 790, y: 252, w: 86, h: 122 },
    crate: { x: 1262, y: 282, w: 58, h: 64 },
    crystal: { x: 1116, y: 270, w: 90, h: 82 },
    vine: { x: 822, y: 252, w: 82, h: 124 },
    sign: { x: 1444, y: 276, w: 76, h: 92 },
    glow: { x: 1358, y: 260, w: 70, h: 106 }
  },
  gearworks: {
    grass: { x: 1122, y: 380, w: 76, h: 58 },
    bush: { x: 1310, y: 380, w: 94, h: 72 },
    tree: { x: 835, y: 356, w: 70, h: 112 },
    rock: { x: 1395, y: 382, w: 80, h: 58 },
    flower: { x: 1475, y: 380, w: 60, h: 66 },
    small: { x: 898, y: 354, w: 78, h: 102 },
    tall: { x: 1030, y: 352, w: 70, h: 110 },
    crate: { x: 1184, y: 378, w: 58, h: 62 },
    crystal: { x: 1390, y: 372, w: 82, h: 78 },
    vine: { x: 760, y: 350, w: 66, h: 118 },
    sign: { x: 1468, y: 354, w: 64, h: 98 },
    glow: { x: 704, y: 352, w: 116, h: 108 }
  },
  cinder: {
    grass: { x: 1376, y: 482, w: 52, h: 70 },
    bush: { x: 977, y: 470, w: 62, h: 96 },
    tree: { x: 975, y: 464, w: 70, h: 118 },
    rock: { x: 708, y: 466, w: 88, h: 86 },
    flower: { x: 1170, y: 478, w: 54, h: 72 },
    small: { x: 810, y: 455, w: 66, h: 108 },
    tall: { x: 888, y: 456, w: 76, h: 110 },
    crate: { x: 1226, y: 482, w: 62, h: 70 },
    crystal: { x: 712, y: 455, w: 90, h: 100 },
    vine: { x: 888, y: 456, w: 76, h: 110 },
    sign: { x: 1210, y: 462, w: 74, h: 100 },
    glow: { x: 1065, y: 480, w: 88, h: 80 }
  },
  frost: {
    grass: { x: 1110, y: 592, w: 70, h: 64 },
    bush: { x: 690, y: 588, w: 90, h: 80 },
    tree: { x: 694, y: 560, w: 88, h: 118 },
    rock: { x: 855, y: 585, w: 92, h: 66 },
    flower: { x: 1168, y: 580, w: 62, h: 84 },
    small: { x: 930, y: 570, w: 92, h: 86 },
    tall: { x: 1230, y: 550, w: 82, h: 118 },
    crate: { x: 1080, y: 586, w: 62, h: 64 },
    crystal: { x: 1254, y: 548, w: 72, h: 118 },
    vine: { x: 842, y: 582, w: 96, h: 70 },
    sign: { x: 1200, y: 552, w: 70, h: 112 },
    glow: { x: 1320, y: 564, w: 60, h: 96 }
  },
  ruins: {
    grass: { x: 828, y: 672, w: 84, h: 74 },
    bush: { x: 785, y: 668, w: 90, h: 80 },
    tree: { x: 690, y: 654, w: 88, h: 112 },
    rock: { x: 995, y: 680, w: 70, h: 60 },
    flower: { x: 826, y: 670, w: 78, h: 72 },
    small: { x: 960, y: 648, w: 74, h: 104 },
    tall: { x: 1038, y: 642, w: 66, h: 112 },
    crate: { x: 1118, y: 690, w: 64, h: 54 },
    crystal: { x: 689, y: 758, w: 64, h: 90 },
    vine: { x: 903, y: 652, w: 70, h: 112 },
    sign: { x: 1262, y: 656, w: 76, h: 88 },
    glow: { x: 1068, y: 648, w: 70, h: 104 }
  },
  astral: {
    grass: { x: 772, y: 790, w: 62, h: 62 },
    bush: { x: 686, y: 760, w: 78, h: 88 },
    tree: { x: 1082, y: 776, w: 70, h: 86 },
    rock: { x: 1130, y: 776, w: 82, h: 82 },
    flower: { x: 868, y: 754, w: 82, h: 96 },
    small: { x: 1018, y: 750, w: 64, h: 100 },
    tall: { x: 940, y: 746, w: 78, h: 108 },
    crate: { x: 1210, y: 780, w: 56, h: 68 },
    crystal: { x: 804, y: 760, w: 72, h: 92 },
    vine: { x: 1168, y: 748, w: 66, h: 104 },
    sign: { x: 1200, y: 736, w: 78, h: 116 },
    glow: { x: 890, y: 742, w: 88, h: 108 }
  },
  rift: {
    grass: { x: 1062, y: 874, w: 90, h: 76 },
    bush: { x: 1122, y: 866, w: 92, h: 84 },
    tree: { x: 1240, y: 846, w: 76, h: 122 },
    rock: { x: 1015, y: 878, w: 70, h: 78 },
    flower: { x: 790, y: 862, w: 70, h: 96 },
    small: { x: 872, y: 862, w: 62, h: 100 },
    tall: { x: 930, y: 848, w: 66, h: 116 },
    crate: { x: 1410, y: 874, w: 84, h: 74 },
    crystal: { x: 820, y: 852, w: 84, h: 112 },
    vine: { x: 1020, y: 848, w: 74, h: 112 },
    sign: { x: 1340, y: 842, w: 80, h: 118 },
    glow: { x: 698, y: 848, w: 90, h: 106 }
  },
  quarry: {
    grass: { x: 1250, y: 968, w: 74, h: 50 },
    bush: { x: 690, y: 940, w: 82, h: 84 },
    tree: { x: 992, y: 914, w: 80, h: 110 },
    rock: { x: 1188, y: 958, w: 88, h: 62 },
    flower: { x: 1328, y: 954, w: 82, h: 68 },
    small: { x: 835, y: 950, w: 70, h: 72 },
    tall: { x: 1000, y: 916, w: 72, h: 108 },
    crate: { x: 1195, y: 932, w: 70, h: 84 },
    crystal: { x: 688, y: 918, w: 92, h: 106 },
    vine: { x: 1010, y: 918, w: 72, h: 106 },
    sign: { x: 1070, y: 918, w: 70, h: 106 },
    glow: { x: 1388, y: 954, w: 80, h: 68 }
  }
});

const THEME_CONFIGS = Object.freeze({
  'starfall-crossing': { terrain: 'town', props: 'town', shift: 0.02, hue: 0, brightness: 1.03, saturation: 1.03 },
  'rustcoil-outpost': { terrain: 'gearworks', props: 'gearworks', shift: 0.08, hue: 12, brightness: 0.96, saturation: 0.92 },
  'cinder-refuge': { terrain: 'cinder', props: 'cinder', shift: 0.12, hue: 6, brightness: 1.04, saturation: 1.08 },
  'frostfen-camp': { terrain: 'frost', props: 'frost', shift: 0.02, hue: 0, brightness: 1.05, saturation: 0.92 },
  'stormbreak-haven': { terrain: 'ruins', props: 'town', shift: 0.18, hue: 196, brightness: 1.02, saturation: 0.94 },
  'astral-observatory': { terrain: 'astral', props: 'astral', shift: 0.06, hue: 184, brightness: 1.05, saturation: 1.08 },
  'greenroot-meadow': { terrain: 'meadow', props: 'meadow', shift: 0.0, hue: 0, brightness: 1.06, saturation: 1.08 },
  'thornpath-thicket': { terrain: 'thorn', props: 'thorn', shift: 0.04, hue: 348, brightness: 0.96, saturation: 1.12 },
  'bramble-depths': { terrain: 'thorn', props: 'thorn', shift: 0.42, hue: 330, brightness: 0.82, saturation: 1.2 },
  'rustcoil-ruins': { terrain: 'ruins', props: 'gearworks', shift: 0.28, hue: 25, brightness: 0.9, saturation: 0.82 },
  'gearworks-vault': { terrain: 'gearworks', props: 'gearworks', shift: 0.48, hue: 190, brightness: 0.92, saturation: 1.04 },
  'cinder-hollow': { terrain: 'cinder', props: 'cinder', shift: 0.38, hue: 350, brightness: 0.82, saturation: 1.14 },
  'emberjaw-lair': { terrain: 'cinder', props: 'cinder', shift: 0.62, hue: 22, brightness: 0.92, saturation: 1.2 },
  'bandit-ridge-camp': { terrain: 'meadow', props: 'town', shift: 0.5, hue: 28, brightness: 0.98, saturation: 0.94 },
  'oreback-quarry': { terrain: 'quarry', props: 'quarry', shift: 0.0, hue: 0, brightness: 0.98, saturation: 0.98 },
  'ashglass-pass': { terrain: 'cinder', props: 'quarry', shift: 0.72, hue: 320, brightness: 0.94, saturation: 0.9 },
  'frostfen-outskirts': { terrain: 'frost', props: 'meadow', shift: 0.34, hue: 185, brightness: 1.03, saturation: 0.78 },
  'glacier-spine': { terrain: 'frost', props: 'frost', shift: 0.52, hue: 204, brightness: 1.08, saturation: 0.86 },
  'rimewarden-sanctum': { terrain: 'astral', props: 'frost', shift: 0.36, hue: 198, brightness: 1.02, saturation: 0.72 },
  'stormbreak-cliffs': { terrain: 'ruins', props: 'frost', shift: 0.5, hue: 205, brightness: 0.96, saturation: 0.88 },
  'astral-archive': { terrain: 'astral', props: 'town', shift: 0.42, hue: 220, brightness: 0.96, saturation: 0.98 },
  'eclipse-frontier': { terrain: 'rift', props: 'rift', shift: 0.14, hue: 35, brightness: 0.9, saturation: 1.05 },
  'endless-rift': { terrain: 'rift', props: 'rift', shift: 0.46, hue: 278, brightness: 0.86, saturation: 1.2 },
  'brambleking-court': { terrain: 'thorn', props: 'thorn', shift: 0.72, hue: 12, brightness: 0.9, saturation: 1.16 },
  'titan-foundry': { terrain: 'gearworks', props: 'gearworks', shift: 0.7, hue: 18, brightness: 0.88, saturation: 1.04 },
  'deepcore-core': { terrain: 'quarry', props: 'quarry', shift: 0.38, hue: 172, brightness: 0.9, saturation: 1.14 },
  'emberjaw-furnace': { terrain: 'cinder', props: 'gearworks', shift: 0.84, hue: 20, brightness: 0.9, saturation: 1.24 },
  'rimewarden-vault': { terrain: 'frost', props: 'astral', shift: 0.78, hue: 204, brightness: 1.04, saturation: 0.86 },
  'stormbreak-aerie': { terrain: 'ruins', props: 'astral', shift: 0.74, hue: 205, brightness: 0.98, saturation: 1.0 },
  'astral-stacks': { terrain: 'astral', props: 'astral', shift: 0.76, hue: 245, brightness: 0.94, saturation: 1.08 },
  'eclipse-throne': { terrain: 'rift', props: 'astral', shift: 0.7, hue: 290, brightness: 0.82, saturation: 1.18 },
  'guardian-trial': { terrain: 'town', props: 'town', shift: 0.2, hue: 205, brightness: 1.03, saturation: 0.92 },
  'berserker-trial': { terrain: 'cinder', props: 'cinder', shift: 0.2, hue: 350, brightness: 0.94, saturation: 1.18 },
  'duelist-trial': { terrain: 'town', props: 'meadow', shift: 0.46, hue: 28, brightness: 1.04, saturation: 0.98 },
  'fire-mage-trial': { terrain: 'cinder', props: 'cinder', shift: 0.3, hue: 18, brightness: 1.02, saturation: 1.2 },
  'rune-mage-trial': { terrain: 'astral', props: 'astral', shift: 0.2, hue: 170, brightness: 1.04, saturation: 1.08 },
  'storm-mage-trial': { terrain: 'ruins', props: 'frost', shift: 0.62, hue: 205, brightness: 1.02, saturation: 0.98 },
  'sniper-trial': { terrain: 'meadow', props: 'town', shift: 0.68, hue: 42, brightness: 1.02, saturation: 0.9 },
  'trapper-trial': { terrain: 'thorn', props: 'thorn', shift: 0.18, hue: 345, brightness: 0.94, saturation: 1.12 },
  'beast-archer-trial': { terrain: 'meadow', props: 'meadow', shift: 0.28, hue: 96, brightness: 1.02, saturation: 1.06 }
});

function ensureDirs() {
  [SOURCE_DIR, TERRAIN_DIR, PROP_DIR].forEach((dir) => fs.mkdirSync(dir, { recursive: true }));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function safeCrop(crop, width, height) {
  const x = clamp(Math.round(crop.x || 0), 0, Math.max(0, width - 1));
  const y = clamp(Math.round(crop.y || 0), 0, Math.max(0, height - 1));
  const w = clamp(Math.round(crop.w || CELL), 1, Math.max(1, width - x));
  const h = clamp(Math.round(crop.h || CELL), 1, Math.max(1, height - y));
  return { left: x, top: y, width: w, height: h };
}

async function createKeyedSourceBuffer(sourcePath) {
  const source = await sharp(sourcePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const data = Buffer.from(source.data);
  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const isHardKey = r < 32 && g > 190 && b < 32;
    const isSoftKey = r < 46 && g > 170 && b < 46;
    if (isHardKey || isSoftKey) {
      data[index + 3] = 0;
      continue;
    }
    if (g > 150 && g > r * 1.9 && g > b * 1.9) {
      data[index + 1] = Math.max(r, b);
    }
  }
  return sharp(data, { raw: source.info }).png().toBuffer();
}

function themeSourceConfig(id) {
  return THEME_CONFIGS[id] || THEME_CONFIGS['greenroot-meadow'];
}

function themeStructureKey(id) {
  const config = themeSourceConfig(id);
  return [
    config.terrain,
    config.props,
    Number(config.shift || 0).toFixed(2)
  ].join(':');
}

function validateThemeConfigCoverage() {
  const missing = THEME_IDS.filter((id) => !THEME_CONFIGS[id]);
  if (missing.length) {
    throw new Error(`Missing imagegen environment theme configs: ${missing.join(', ')}`);
  }
  const seen = new Map();
  THEME_IDS.forEach((id) => {
    const key = themeStructureKey(id);
    const existing = seen.get(key);
    if (existing) {
      throw new Error(`${id} reuses ${existing}'s structural environment source key (${key})`);
    }
    seen.set(key, id);
  });
}

function terrainCrop(source, config, kind) {
  const maxShift = Math.max(0, source.w - 118);
  const shiftedX = source.x + Math.round(maxShift * Number(config.shift || 0));
  const base = {
    top: { x: shiftedX + 18, y: source.y + 2, w: 118, h: 64 },
    body: { x: shiftedX + 82, y: source.y + Math.round(source.h * 0.34), w: 118, h: 64 },
    left: { x: source.x, y: source.y + 2, w: 92, h: Math.min(88, source.h) },
    right: { x: source.x + Math.max(0, source.w - 100), y: source.y + 2, w: 96, h: Math.min(88, source.h) },
    underside: { x: shiftedX + 130, y: source.y + Math.max(0, source.h - 56), w: 118, h: 56 },
    topAlt: { x: shiftedX + 250, y: source.y + 2, w: 118, h: 64 },
    cap: { x: shiftedX + 390, y: source.y + Math.round(source.h * 0.28), w: 98, h: 64 },
    detail: { x: shiftedX + 500, y: source.y + Math.round(source.h * 0.08), w: 90, h: 60 }
  };
  return base[kind] || base.top;
}

async function buildTerrainCell(keyedBuffer, sourceMeta, crop, config, options) {
  let pipeline = sharp(keyedBuffer)
    .extract(safeCrop(crop, sourceMeta.width, sourceMeta.height))
    .resize(CELL, CELL, {
      fit: options && options.fit || 'cover',
      position: options && options.position || 'center',
      kernel: sharp.kernel.nearest,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    });
  if (config.hue || config.brightness || config.saturation) {
    pipeline = pipeline.modulate({
      hue: Number(config.hue || 0),
      brightness: Number(config.brightness || 1),
      saturation: Number(config.saturation || 1)
    });
  }
  return pipeline.png().toBuffer();
}

async function buildSpriteCell(keyedBuffer, sourceMeta, crop, config, scale) {
  const maxSize = Math.max(24, Math.round(CELL * Number(scale || 0.92)));
  const safe = safeCrop(crop, sourceMeta.width, sourceMeta.height);
  let spritePipeline = sharp(keyedBuffer).extract(safe);
  try {
    spritePipeline = spritePipeline.trim({ threshold: 8 });
    if (config.hue || config.brightness || config.saturation) {
      spritePipeline = spritePipeline.modulate({
        hue: Number(config.hue || 0),
        brightness: Number(config.brightness || 1),
        saturation: Number(config.saturation || 1)
      });
    }
    const sprite = await spritePipeline
      .resize(maxSize, maxSize, {
        fit: 'inside',
        withoutEnlargement: false,
        kernel: sharp.kernel.nearest,
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toBuffer();
    return placeSpriteOnCell(sprite);
  } catch (error) {
    let fallback = sharp(keyedBuffer).extract(safe);
    if (config.hue || config.brightness || config.saturation) {
      fallback = fallback.modulate({
        hue: Number(config.hue || 0),
        brightness: Number(config.brightness || 1),
        saturation: Number(config.saturation || 1)
      });
    }
    const sprite = await fallback
      .resize(maxSize, maxSize, {
        fit: 'inside',
        withoutEnlargement: false,
        kernel: sharp.kernel.nearest,
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toBuffer();
    return placeSpriteOnCell(sprite);
  }
}

async function placeSpriteOnCell(sprite) {
  const meta = await sharp(sprite).metadata();
  const left = Math.max(0, Math.floor((CELL - meta.width) / 2));
  const top = Math.max(0, CELL - meta.height - 2);
  return sharp({
    create: {
      width: CELL,
      height: CELL,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite([{ input: sprite, left, top }])
    .png()
    .toBuffer();
}

async function writeAtlas(outPath, width, height, cells) {
  const composites = cells.map((input, index) => ({
    input,
    left: (index % (width / CELL)) * CELL,
    top: Math.floor(index / (width / CELL)) * CELL
  }));
  await sharp({
    create: {
      width,
      height,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite(composites)
    .png()
    .toFile(outPath);
}

async function generateTerrainAtlas(keyedBuffer, sourceMeta, id) {
  const config = themeSourceConfig(id);
  const source = TERRAIN_SOURCES[config.terrain] || TERRAIN_SOURCES.meadow;
  const cells = await Promise.all([
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'top'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'body'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'left'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'right'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'underside'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'topAlt'), config),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'cap'), config, { fit: 'contain' }),
    buildTerrainCell(keyedBuffer, sourceMeta, terrainCrop(source, config, 'detail'), config, { fit: 'contain' })
  ]);
  await writeAtlas(path.join(TERRAIN_DIR, `${id}.png`), 256, 128, cells);
}

async function generatePropAtlas(keyedBuffer, sourceMeta, id) {
  const config = themeSourceConfig(id);
  const props = PROP_SOURCES[config.props] || PROP_SOURCES.meadow;
  const cells = await Promise.all([
    buildSpriteCell(keyedBuffer, sourceMeta, props.grass, config, 0.78),
    buildSpriteCell(keyedBuffer, sourceMeta, props.bush, config, 0.9),
    buildSpriteCell(keyedBuffer, sourceMeta, props.tree, config, 0.98),
    buildSpriteCell(keyedBuffer, sourceMeta, props.rock, config, 0.86),
    buildSpriteCell(keyedBuffer, sourceMeta, props.flower, config, 0.78),
    buildSpriteCell(keyedBuffer, sourceMeta, props.small, config, 0.86),
    buildSpriteCell(keyedBuffer, sourceMeta, props.tall, config, 0.98),
    buildSpriteCell(keyedBuffer, sourceMeta, props.crate, config, 0.86),
    buildSpriteCell(keyedBuffer, sourceMeta, props.crystal, config, 0.96),
    buildSpriteCell(keyedBuffer, sourceMeta, props.vine, config, 0.98),
    buildSpriteCell(keyedBuffer, sourceMeta, props.sign, config, 0.92),
    buildSpriteCell(keyedBuffer, sourceMeta, props.glow, config, 0.76)
  ]);
  await writeAtlas(path.join(PROP_DIR, `${id}.png`), 384, 128, cells);
}

async function validatePng(filePath, expectedWidth, expectedHeight, requireTransparentCorners) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing ${path.relative(ROOT, filePath)}`);
  const image = await sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const { width, height, channels } = image.info;
  if (width !== expectedWidth || height !== expectedHeight) {
    throw new Error(`${path.relative(ROOT, filePath)} is ${width}x${height}; expected ${expectedWidth}x${expectedHeight}`);
  }
  if (channels < 4) throw new Error(`${path.relative(ROOT, filePath)} is missing alpha`);
  let visible = 0;
  for (let index = 3; index < image.data.length; index += channels) {
    if (image.data[index] > 12) visible += 1;
  }
  if (visible < expectedWidth * expectedHeight * 0.08) {
    throw new Error(`${path.relative(ROOT, filePath)} has too little visible sprite coverage`);
  }
  if (requireTransparentCorners) {
    const cornerIndexes = [
      3,
      (expectedWidth - 1) * channels + 3,
      ((expectedHeight - 1) * expectedWidth) * channels + 3,
      ((expectedHeight * expectedWidth) - 1) * channels + 3
    ];
    const opaqueCorner = cornerIndexes.some((index) => image.data[index] > 12);
    if (opaqueCorner) throw new Error(`${path.relative(ROOT, filePath)} has opaque prop atlas corners`);
  }
}

async function generateAll() {
  ensureDirs();
  validateThemeConfigCoverage();
  if (!fs.existsSync(SOURCE_PATH)) {
    throw new Error(`Missing imagegen source: ${path.relative(ROOT, SOURCE_PATH)}`);
  }
  const keyedBuffer = await createKeyedSourceBuffer(SOURCE_PATH);
  await fs.promises.writeFile(KEYED_SOURCE_PATH, keyedBuffer);
  const sourceMeta = await sharp(keyedBuffer).metadata();
  for (const id of THEME_IDS) {
    await generateTerrainAtlas(keyedBuffer, sourceMeta, id);
    await generatePropAtlas(keyedBuffer, sourceMeta, id);
  }
  console.log(`Generated ${THEME_IDS.length} imagegen terrain atlases and ${THEME_IDS.length} prop atlases from ${path.relative(ROOT, SOURCE_PATH)}`);
}

async function validateAll() {
  validateThemeConfigCoverage();
  if (!fs.existsSync(SOURCE_PATH)) throw new Error(`Missing ${path.relative(ROOT, SOURCE_PATH)}`);
  for (const id of THEME_IDS) {
    await validatePng(path.join(TERRAIN_DIR, `${id}.png`), 256, 128, false);
    await validatePng(path.join(PROP_DIR, `${id}.png`), 384, 128, true);
  }
  console.log(`Validated ${THEME_IDS.length} imagegen environment terrain atlases and ${THEME_IDS.length} prop atlases`);
}

async function main() {
  if (process.argv.includes('--validate')) {
    await validateAll();
    return;
  }
  await generateAll();
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack || error);
    process.exit(1);
  });
}
