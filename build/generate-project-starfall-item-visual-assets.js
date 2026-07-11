#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const MANIFEST_PATH = 'img/project-starfall/items/item-visual-manifest.json';
const ASSET_ROOT = 'img/project-starfall';
const PROCEDURAL_BACKUP_ROOT = `${ASSET_ROOT}/backups/procedural`;
const DATA = require('../js/games/project-starfall/project-starfall-data.js');
const MANIFEST = JSON.parse(fs.readFileSync(path.join(ROOT, MANIFEST_PATH), 'utf8'));

function fullPath(filePath) {
  return path.join(ROOT, filePath);
}

function ensureDir(filePath) {
  fs.mkdirSync(path.dirname(fullPath(filePath)), { recursive: true });
}

function ensureProceduralBackup(filePath) {
  const relativePath = path.relative(ASSET_ROOT, filePath).replace(/\\/g, '/');
  if (!relativePath || relativePath.startsWith('../')) return false;
  const backupPath = `${PROCEDURAL_BACKUP_ROOT}/${relativePath}`;
  if (fs.existsSync(fullPath(backupPath))) return false;
  ensureDir(backupPath);
  fs.copyFileSync(fullPath(filePath), fullPath(backupPath));
  return true;
}

function kebabFromId(itemId) {
  return String(itemId || '')
    .replace(/_/g, '-')
    .replace(/[^a-zA-Z0-9-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase();
}

function materialAssetId(materialId) {
  return String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase();
}

function getGeneratedConfig() {
  return MANIFEST.generatedAssets || {};
}

function getRarityVisual(rarity) {
  return MANIFEST.rarities && MANIFEST.rarities[rarity] || MANIFEST.rarities.Common;
}

function buildDefinitionMap() {
  const definitions = new Map();
  const add = (id, definition) => {
    if (!id || definitions.has(id)) return;
    definitions.set(id, definition);
  };

  (DATA.CONSUMABLE_ITEMS || []).forEach((item) => add(item.id, Object.assign({ kind: 'consumable' }, item)));
  (DATA.MATERIAL_ITEMS || []).forEach((item) => {
    const assetId = item.assetId || materialAssetId(item.materialId || item.id);
    const definition = Object.assign({ kind: 'material', id: assetId }, item);
    add(assetId, definition);
    add(item.materialId || item.id, definition);
  });
  []
    .concat(DATA.SHOP_ITEMS || [])
    .concat(DATA.RANDOM_EQUIPMENT_ITEMS || [])
    .concat(DATA.BOSS_EQUIPMENT_ITEMS || [])
    .forEach((item) => {
      const definition = Object.assign({ kind: 'equipment' }, item);
      add(item.id, definition);
      if (item.assetId) add(item.assetId, definition);
      if (item.visualId) add(item.visualId, definition);
    });
  (DATA.CARD_DEFINITIONS || []).forEach((card) => add(card.id, Object.assign({ kind: 'card' }, card)));
  return definitions;
}

function svgBuffer(svg) {
  return Buffer.from(svg.trim());
}

function ringSvg(width, height, color, alpha, ring) {
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
      <defs>
        <radialGradient id="aura" cx="50%" cy="48%" r="45%">
          <stop offset="0%" stop-color="${color}" stop-opacity="${Math.min(0.26, alpha * 0.32)}"/>
          <stop offset="58%" stop-color="${color}" stop-opacity="${Math.min(0.16, alpha * 0.2)}"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="${width}" height="${height}" fill="transparent"/>
      <ellipse cx="${width / 2}" cy="${Math.round(height * 0.82)}" rx="${Math.round(width * 0.26)}" ry="5" fill="#050713" opacity="0.34"/>
      <circle cx="${width / 2}" cy="${Math.round(height * 0.46)}" r="${Math.round(width * 0.35)}" fill="url(#aura)"/>
      <circle cx="${width / 2}" cy="${Math.round(height * 0.46)}" r="${Math.round(width * 0.29)}" fill="none" stroke="${color}" stroke-width="${ring}" opacity="${Math.min(0.46, alpha * 0.5)}"/>
    </svg>
  `;
}

function accentColorForItem(item) {
  const id = String(item && item.id || '').toLowerCase();
  const source = String(item && item.source || '').toLowerCase();
  const combined = `${id} ${source}`;
  if (/rustcoil|gear|clock|coil|scrap/.test(combined)) return '#3aa6c8';
  if (/cinder|ember|ash|furnace|lava|magma|scorch/.test(combined)) return '#ff6b2f';
  if (/frost|rime|snow|ice|glacier/.test(combined)) return '#68a9ff';
  if (/storm|thunder|gale|cloud|tempest|lightning/.test(combined)) return '#dff8ff';
  if (/astral|star|comet|orbit|index|lens/.test(combined)) return '#6edcff';
  if (/duelist/.test(combined)) return '#fff4d5';
  if (/beast/.test(combined)) return '#8fcf6a';
  return '#ffbe55';
}

function directIconAccentSvg(item, color) {
  const slot = String(item && item.slot || '').toLowerCase();
  const symbol = {
    weapon: '<path d="M12 52 L50 14 L55 19 L17 57 Z" fill="#050713" opacity="0.24"/><path d="M43 10 L57 7 L54 21 Z" fill="#fff7d5" opacity="0.55"/>',
    head: '<path d="M43 11 L56 17 L52 28 L42 25 Z" fill="#050713" opacity="0.24"/>',
    chest: '<path d="M44 13 L57 22 L54 38 L42 31 Z" fill="#050713" opacity="0.24"/>',
    gloves: '<circle cx="49" cy="18" r="8" fill="#050713" opacity="0.22"/>',
    boots: '<path d="M42 18 C50 20 55 25 55 34 L42 34 Z" fill="#050713" opacity="0.24"/>',
    ring: '<circle cx="49" cy="20" r="8" fill="none" stroke="#050713" stroke-width="4" opacity="0.28"/>',
    amulet: '<path d="M49 10 L58 22 L49 34 L40 22 Z" fill="#050713" opacity="0.24"/>',
    offhand: '<path d="M49 9 L58 15 L55 29 L49 36 L43 29 L40 15 Z" fill="#050713" opacity="0.24"/>'
  }[slot] || '<circle cx="49" cy="20" r="9" fill="#050713" opacity="0.24"/>';
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
      <defs>
        <radialGradient id="accent" cx="72%" cy="34%" r="30%">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.48"/>
          <stop offset="70%" stop-color="${color}" stop-opacity="0.16"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="64" height="64" fill="transparent"/>
      <circle cx="46" cy="21" r="17" fill="url(#accent)"/>
      <path d="M53 8 L57 15 L53 22 L49 15 Z" fill="${color}" opacity="0.86"/>
      <circle cx="53" cy="15" r="2.5" fill="#fff7d5" opacity="0.82"/>
      ${symbol}
    </svg>
  `;
}

async function stripLowAlphaPixels(buffer, alphaCutoff = 24) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  for (let offset = 0; offset < data.length; offset += 4) {
    if (data[offset + 3] <= alphaCutoff) {
      data[offset + 3] = 0;
    }
  }
  return sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  }).png().toBuffer();
}

async function createDirectIconVariant(sourcePath, outputPath, item) {
  const accent = accentColorForItem(item);
  const baseIcon = await sharp(fullPath(sourcePath))
    .resize(56, 56, { fit: 'contain', withoutEnlargement: true })
    .png()
    .toBuffer();
  ensureDir(outputPath);
  const rendered = await sharp(svgBuffer('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="transparent"/></svg>'))
    .png()
    .composite([
      { input: svgBuffer(directIconAccentSvg(item, accent)), left: 0, top: 0 },
      { input: baseIcon, left: 4, top: 4 }
    ])
    .png()
    .toBuffer();
  const cleaned = await stripLowAlphaPixels(rendered);
  await sharp(cleaned).png().toFile(fullPath(outputPath));
  ensureProceduralBackup(outputPath);
}

async function writePngFromSvg(relativePath, svg) {
  ensureDir(relativePath);
  await sharp(svgBuffer(svg)).png().toFile(fullPath(relativePath));
}

async function createPickupSprite(sourcePath, outputPath, rarity) {
  const visual = getRarityVisual(rarity || 'Common');
  const color = visual.color || '#d8e5ec';
  const alpha = Number(visual.alpha || 0.45);
  const ring = Number(visual.ring || 1.4);
  const base = sharp(svgBuffer(ringSvg(64, 64, color, alpha, ring))).png();
  const icon = await sharp(fullPath(sourcePath))
    .resize(48, 48, { fit: 'contain', withoutEnlargement: true })
    .png()
    .toBuffer();
  ensureDir(outputPath);
  await base
    .composite([{ input: icon, left: 8, top: 5 }])
    .png()
    .toFile(fullPath(outputPath));
}

function rarityFrameSvg(rarity, visual) {
  const color = visual.color || '#d8e5ec';
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
      <rect width="64" height="64" fill="transparent"/>
      <rect x="4" y="4" width="56" height="56" rx="8" fill="none" stroke="#050713" stroke-width="4" opacity="0.5"/>
      <rect x="5" y="5" width="54" height="54" rx="7" fill="none" stroke="${color}" stroke-width="2.5" opacity="0.95"/>
      <path d="M12 6 L22 6 L6 22 L6 12 Z" fill="${color}" opacity="0.45"/>
      <path d="M58 42 L58 52 L52 58 L42 58 Z" fill="${color}" opacity="0.34"/>
      <circle cx="50" cy="14" r="2" fill="#fff7d5" opacity="0.72"/>
      <title>${rarity}</title>
    </svg>
  `;
}

function markerSvg(category, color) {
  const shape = {
    weapon: '<path d="M9 24 L23 10 L27 14 L13 28 Z M21 8 L28 4 L30 6 L25 13 Z" />',
    armor: '<path d="M16 4 L27 9 L25 22 L16 30 L7 22 L5 9 Z" />',
    accessory: '<circle cx="16" cy="17" r="8" fill="none" stroke-width="4"/><circle cx="16" cy="8" r="3"/>',
    potion: '<path d="M12 5 H20 V10 L24 16 V27 H8 V16 L12 10 Z" />',
    food: '<path d="M8 18 C8 10 16 6 24 12 C24 22 18 28 10 26 C8 24 8 21 8 18 Z" />',
    scroll: '<path d="M8 9 C8 6 11 5 14 7 H24 V25 H11 C8 25 7 22 9 20 V11 C8 11 8 10 8 9 Z" />',
    coupon: '<path d="M5 10 H27 V15 C24 15 24 19 27 19 V24 H5 V19 C8 19 8 15 5 15 Z" />',
    tool: '<circle cx="16" cy="16" r="8" fill="none" stroke-width="4"/><path d="M16 2 V8 M16 24 V30 M2 16 H8 M24 16 H30" fill="none" stroke-width="3"/>',
    currency: '<circle cx="12" cy="18" r="7"/><circle cx="20" cy="14" r="7" opacity="0.75"/>',
    crafting_material: '<path d="M16 3 L27 16 L18 29 L6 20 Z" />',
    upgrade_material: '<path d="M16 3 L27 14 L16 30 L5 14 Z M16 8 L21 15 L16 23 L11 15 Z" fill-rule="evenodd" />',
    star_card: '<path d="M16 3 L20 12 L30 12 L22 18 L25 28 L16 22 L7 28 L10 18 L2 12 L12 12 Z" />',
    monster_card: '<rect x="7" y="4" width="18" height="24" rx="3"/><circle cx="16" cy="15" r="5" fill="#050713" opacity="0.36"/>',
    quest_item: '<circle cx="11" cy="11" r="5" fill="none" stroke-width="4"/><path d="M15 15 L28 28 M22 22 L25 19 M25 25 L28 22" fill="none" stroke-width="4"/>',
    boss_drop: '<path d="M5 25 L7 9 L14 17 L16 6 L19 17 L26 9 L28 25 Z" />'
  }[category] || '<circle cx="16" cy="16" r="10"/>';
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
      <rect width="32" height="32" fill="transparent"/>
      <circle cx="16" cy="16" r="15" fill="#050713" opacity="0.72"/>
      <g fill="${color}" stroke="${color}" stroke-linejoin="round" stroke-linecap="round">${shape}</g>
      <circle cx="24" cy="8" r="2" fill="#fff7d5" opacity="0.72"/>
    </svg>
  `;
}

function vfxSheetSvg(frameCount, color, mode) {
  const width = frameCount * 64;
  const frames = [];
  for (let i = 0; i < frameCount; i += 1) {
    const x = i * 64;
    const progress = frameCount <= 1 ? 1 : i / (frameCount - 1);
    const ringRadius = 10 + progress * 20;
    const opacity = Math.max(0.08, 0.85 - progress * 0.7);
    const sparkle = mode === 'sparkle' || mode === 'boss';
    const crown = mode === 'boss';
    frames.push(`
      <g transform="translate(${x} 0)">
        <circle cx="32" cy="32" r="${ringRadius.toFixed(1)}" fill="none" stroke="${color}" stroke-width="${(3 - progress * 1.6).toFixed(2)}" opacity="${opacity.toFixed(2)}"/>
        <circle cx="32" cy="32" r="${(5 + progress * 4).toFixed(1)}" fill="${color}" opacity="${Math.max(0.05, 0.25 - progress * 0.12).toFixed(2)}"/>
        ${sparkle ? `<path d="M${18 + progress * 7} 17 L${21 + progress * 7} 24 L${28 + progress * 7} 27 L${21 + progress * 7} 30 L${18 + progress * 7} 37 L${15 + progress * 7} 30 L${8 + progress * 7} 27 L${15 + progress * 7} 24 Z" fill="#fff7d5" opacity="${(0.75 - progress * 0.35).toFixed(2)}"/>` : ''}
        ${crown ? `<path d="M20 42 L22 29 L29 36 L32 24 L35 36 L42 29 L44 42 Z" fill="${color}" opacity="${(0.55 - progress * 0.2).toFixed(2)}"/>` : ''}
      </g>
    `);
  }
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="64" viewBox="0 0 ${width} 64">
      <rect width="${width}" height="64" fill="transparent"/>
      ${frames.join('\n')}
    </svg>
  `;
}

function animatedPickupSvg(frameCount, color, mode) {
  const width = frameCount * 64;
  const frames = [];
  for (let i = 0; i < frameCount; i += 1) {
    const x = i * 64;
    const progress = frameCount <= 1 ? 0 : i / (frameCount - 1);
    const bob = Math.sin(progress * Math.PI * 2) * 3;
    const pulse = 0.35 + Math.sin(progress * Math.PI * 2) * 0.16;
    const centerY = 30 + bob;
    const object = {
      currency: `<circle cx="27" cy="${centerY + 4}" r="9" fill="${color}"/><circle cx="37" cy="${centerY}" r="9" fill="#fff0a6" opacity="0.82"/>`,
      quest: `<circle cx="25" cy="${centerY - 3}" r="7" fill="none" stroke="${color}" stroke-width="4"/><path d="M30 ${centerY + 2} L43 ${centerY + 15} M37 ${centerY + 9} L41 ${centerY + 5}" stroke="${color}" stroke-width="4" stroke-linecap="round"/>`,
      equipment: `<path d="M18 ${centerY + 14} L41 ${centerY - 9} L46 ${centerY - 4} L23 ${centerY + 19} Z" fill="${color}"/><path d="M35 ${centerY - 13} L48 ${centerY - 18} L43 ${centerY - 5} Z" fill="#fff7d5" opacity="0.7"/>`,
      boss: `<path d="M17 ${centerY + 14} L20 ${centerY - 7} L29 ${centerY + 4} L32 ${centerY - 12} L36 ${centerY + 4} L45 ${centerY - 7} L48 ${centerY + 14} Z" fill="${color}"/><circle cx="32" cy="${centerY + 5}" r="5" fill="#fff7d5" opacity="0.65"/>`
    }[mode] || `<path d="M32 ${centerY - 16} L46 ${centerY} L32 ${centerY + 16} L18 ${centerY} Z" fill="${color}"/>`;
    frames.push(`
      <g transform="translate(${x} 0)">
        <ellipse cx="32" cy="53" rx="15" ry="4" fill="#050713" opacity="0.28"/>
        <circle cx="32" cy="${centerY}" r="${18 + pulse * 10}" fill="none" stroke="${color}" stroke-width="2" opacity="${pulse.toFixed(2)}"/>
        ${object}
      </g>
    `);
  }
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="64" viewBox="0 0 ${width} 64">
      <rect width="${width}" height="64" fill="transparent"/>
      ${frames.join('\n')}
    </svg>
  `;
}

async function generatePickups(definitions) {
  const generated = getGeneratedConfig().pickupSprites || {};
  const folder = generated.folder || 'img/project-starfall/items/pickups';
  let count = 0;
  for (const [itemId, assetPath] of Object.entries(DATA.ITEM_ASSETS || {})) {
    const definition = definitions.get(itemId) || {};
    const rarity = definition.rarity || 'Common';
    await createPickupSprite(assetPath, `${folder}/pickup-${kebabFromId(itemId)}.png`, rarity);
    count += 1;
  }
  for (const [cardId, assetPath] of Object.entries(DATA.CARD_ASSETS || {})) {
    const definition = definitions.get(cardId) || {};
    await createPickupSprite(assetPath, `${folder}/pickup-card-${kebabFromId(cardId)}.png`, definition.rarity || 'Common');
    count += 1;
  }
  return count;
}

async function generateDirectIconVariants() {
  const candidates = (DATA.SHOP_ITEMS || []).filter((item) =>
    item &&
    item.assetId &&
    item.assetId !== item.id &&
    DATA.ITEM_ASSETS &&
    DATA.ITEM_ASSETS[item.id] &&
    DATA.ITEM_ASSETS[item.assetId]
  );
  let count = 0;
  for (const item of candidates) {
    await createDirectIconVariant(
      DATA.ITEM_ASSETS[item.assetId],
      DATA.ITEM_ASSETS[item.id],
      item
    );
    count += 1;
  }
  return count;
}

async function generateRarityFrames() {
  const generated = getGeneratedConfig().rarityFrames || {};
  const folder = generated.folder || 'img/project-starfall/items/rarity-frames';
  const rarities = Object.entries(DATA.ITEM_RARITY_VISUALS || {});
  for (const [rarity, visual] of rarities) {
    await writePngFromSvg(`${folder}/frame-rarity-${kebabFromId(rarity)}.png`, rarityFrameSvg(rarity, visual));
  }
  return rarities.length;
}

async function generateCategoryMarkers() {
  const generated = getGeneratedConfig().categoryMarkers || {};
  const folder = generated.folder || 'img/project-starfall/ui/item-overlays';
  const categories = Object.keys(MANIFEST.categories || {});
  const colors = ['#68a9ff', '#ffbe55', '#74d680', '#c794ff', '#d8e5ec'];
  for (let i = 0; i < categories.length; i += 1) {
    await writePngFromSvg(`${folder}/marker-category-${kebabFromId(categories[i])}.png`, markerSvg(categories[i], colors[i % colors.length]));
  }
  return categories.length;
}

async function generateVfx() {
  const generated = getGeneratedConfig().itemVfx || {};
  const folder = generated.folder || 'img/project-starfall/animations/item-vfx';
  const rarities = Object.entries(DATA.ITEM_RARITY_VISUALS || {});
  for (const [rarity, visual] of rarities) {
    await writePngFromSvg(`${folder}/vfx-rarity-${kebabFromId(rarity)}-drop.png`, vfxSheetSvg(6, visual.color, 'ring'));
  }
  await writePngFromSvg(`${folder}/vfx-currency-sparkle.png`, vfxSheetSvg(4, '#ffbe55', 'sparkle'));
  await writePngFromSvg(`${folder}/vfx-quest-pulse.png`, vfxSheetSvg(8, '#6edcff', 'sparkle'));
  await writePngFromSvg(`${folder}/vfx-boss-drop-reveal.png`, vfxSheetSvg(12, '#ffbe55', 'boss'));
  return rarities.length + 3;
}

async function generateAnimatedPickups() {
  const generated = getGeneratedConfig().animatedPickups || {};
  const folder = generated.folder || 'img/project-starfall/items/pickups/animated';
  const sheets = [
    ['pickup-common-idle-sheet.png', 4, '#d8e5ec', 'common'],
    ['pickup-rare-glow-sheet.png', 6, '#68a9ff', 'common'],
    ['pickup-quest-pulse-sheet.png', 8, '#6edcff', 'quest'],
    ['pickup-currency-sparkle-sheet.png', 4, '#ffbe55', 'currency'],
    ['pickup-equipment-reveal-sheet.png', 8, '#c794ff', 'equipment'],
    ['pickup-boss-drop-sheet.png', 12, '#ffbe55', 'boss']
  ];
  for (const [fileName, frames, color, mode] of sheets) {
    await writePngFromSvg(`${folder}/${fileName}`, animatedPickupSvg(frames, color, mode));
  }
  return sheets.length;
}

async function main() {
  const definitions = buildDefinitionMap();
  const counts = {
    directIconVariants: await generateDirectIconVariants(),
    pickupSprites: await generatePickups(definitions),
    rarityFrames: await generateRarityFrames(),
    categoryMarkers: await generateCategoryMarkers(),
    itemVfxSheets: await generateVfx(),
    animatedPickupSheets: await generateAnimatedPickups()
  };
  process.stdout.write([
    'Generated Project Starfall item visual assets:',
    `- ${counts.directIconVariants} direct equipment icon variants`,
    `- ${counts.pickupSprites} pickup sprites`,
    `- ${counts.rarityFrames} rarity frames`,
    `- ${counts.categoryMarkers} category markers`,
    `- ${counts.itemVfxSheets} item VFX sheets`,
    `- ${counts.animatedPickupSheets} animated pickup sheets`
  ].join('\n'));
  process.stdout.write('\n');
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
