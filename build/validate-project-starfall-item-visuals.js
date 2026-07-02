#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const MANIFEST_RELATIVE_PATH = 'img/project-starfall/items/item-visual-manifest.json';

function readJson(relativePath, issues) {
  const filePath = path.join(ROOT, relativePath);
  if (!fs.existsSync(filePath)) {
    issues.push(`${relativePath} is missing`);
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch (error) {
    issues.push(`${relativePath} is not valid JSON: ${error.message}`);
    return null;
  }
}

function readPngDimensions(relativePath) {
  const filePath = path.join(ROOT, relativePath);
  const bytes = fs.readFileSync(filePath);
  if (bytes[0] !== 0x89 || bytes.toString('ascii', 1, 4) !== 'PNG') {
    throw new Error(`${relativePath} should be a PNG`);
  }
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20),
    bitDepth: bytes[24],
    colorType: bytes[25]
  };
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

function fileExists(relativePath) {
  return fs.existsSync(path.join(ROOT, relativePath));
}

function pathIsDirectory(relativePath) {
  const filePath = path.join(ROOT, relativePath);
  return fs.existsSync(filePath) && fs.statSync(filePath).isDirectory();
}

function getItemVisualData(data) {
  const catalogById = new Map();
  const definitions = [];
  const addDefinition = (id, definition) => {
    if (!id) return;
    const existing = catalogById.get(id);
    if (existing && existing.kind !== definition.kind) return;
    catalogById.set(id, definition);
  };

  (data.CONSUMABLE_ITEMS || []).forEach((item) => {
    const definition = Object.assign({ kind: 'consumable' }, item);
    definitions.push(definition);
    addDefinition(item.id, definition);
  });

  (data.MATERIAL_ITEMS || []).forEach((item) => {
    const assetId = item.assetId || materialAssetId(item.materialId || item.id);
    const definition = Object.assign({ kind: 'material', id: assetId, sourceMaterialId: item.materialId || item.id }, item);
    definitions.push(definition);
    addDefinition(assetId, definition);
    addDefinition(item.materialId || item.id, definition);
  });

  []
    .concat(data.SHOP_ITEMS || [])
    .concat(data.RANDOM_EQUIPMENT_ITEMS || [])
    .concat(data.BOSS_EQUIPMENT_ITEMS || [])
    .forEach((item) => {
      const definition = Object.assign({
        kind: item.source && /boss/i.test(item.source) ? 'boss_equipment' : 'equipment'
      }, item);
      definitions.push(definition);
      addDefinition(item.id, definition);
      if (item.assetId) addDefinition(item.assetId, definition);
      if (item.visualId) addDefinition(item.visualId, definition);
    });

  (data.CARD_DEFINITIONS || []).forEach((card) => {
    const definition = Object.assign({ kind: 'card', category: 'monster_card' }, card);
    definitions.push(definition);
    addDefinition(card.id, definition);
  });

  return { catalogById, definitions };
}

function arrayIncludesAny(value, values) {
  const normalizedValue = String(value || '').toLowerCase();
  return (values || []).some((entry) => normalizedValue.includes(String(entry || '').toLowerCase()));
}

function arrayHasSuffix(value, suffixes) {
  const normalizedValue = String(value || '').toLowerCase();
  return (suffixes || []).some((suffix) => normalizedValue.endsWith(String(suffix || '').toLowerCase()));
}

function matchRule(itemId, rules) {
  const normalizedId = String(itemId || '').toLowerCase();
  return (rules || []).find((rule) => {
    if ((rule.ids || []).map((id) => String(id).toLowerCase()).includes(normalizedId)) return true;
    if (arrayIncludesAny(normalizedId, rule.includes)) return true;
    if (arrayHasSuffix(normalizedId, rule.suffixes)) return true;
    return false;
  }) || null;
}

function inferCategory(itemId, definition, manifest) {
  const visualClassification = manifest.visualClassification || {};
  const slotCategories = visualClassification.slotCategories || {};
  const categories = manifest.categories || {};
  if (definition && definition.category && categories[definition.category]) return definition.category;
  if (definition && definition.slot && slotCategories[definition.slot]) return slotCategories[definition.slot];
  if (definition && definition.kind === 'card') return 'monster_card';
  if (definition && definition.kind === 'material') {
    const upgradeRule = matchRule(itemId, (visualClassification.categoryRules || []).filter((rule) => rule.category === 'upgrade_material'));
    return upgradeRule ? 'upgrade_material' : 'crafting_material';
  }
  const rule = matchRule(itemId, visualClassification.categoryRules);
  return rule ? rule.category : '';
}

function inferTheme(itemId, definition, manifest) {
  const visualClassification = manifest.visualClassification || {};
  const fromRule = matchRule(itemId, visualClassification.themeRules);
  if (fromRule) return fromRule.theme;
  if (definition && Array.isArray(definition.tags)) {
    const tagMatch = definition.tags.map((tag) => String(tag).toLowerCase()).find((tag) => manifest.themes && manifest.themes[tag]);
    if (tagMatch) return tagMatch;
  }
  return '';
}

function inferRarity(definition, manifest) {
  const rarity = definition && definition.rarity || manifest.visualClassification && manifest.visualClassification.defaultRarity;
  return String(rarity || '');
}

function validateFolders(manifest, issues) {
  Object.entries(manifest.folders || {}).forEach(([folderId, folder]) => {
    const relativePath = folder && folder.path;
    if (!folder || !relativePath) {
      issues.push(`Manifest folder ${folderId} should declare a path`);
      return;
    }
    const isFileReference = /\.[a-z0-9]+$/i.test(relativePath);
    if (isFileReference) {
      if (!fileExists(relativePath)) issues.push(`Manifest folder file reference ${relativePath} is missing`);
      return;
    }
    if (!pathIsDirectory(relativePath)) {
      issues.push(`Manifest folder ${relativePath} is missing`);
      return;
    }
    if (folder.required && !fileExists(path.join(relativePath, 'README.md')) &&
      ['runtimeIcons', 'runtimeSheets', 'sourceSheets'].indexOf(folderId) < 0) {
      issues.push(`Manifest folder ${relativePath} should include README.md`);
    }
  });
}

function validateRarities(data, manifest, issues) {
  const manifestRarities = manifest.rarities || {};
  Object.entries(data.ITEM_RARITY_VISUALS || {}).forEach(([rarity, visual]) => {
    const manifestRarity = manifestRarities[rarity];
    if (!manifestRarity) {
      issues.push(`${rarity} rarity is implemented but missing from item visual manifest`);
      return;
    }
    if (manifestRarity.color !== visual.color) {
      issues.push(`${rarity} rarity color should match runtime ${visual.color}`);
    }
    ['glow', 'alpha', 'ring'].forEach((field) => {
      if (Number(manifestRarity[field]) !== Number(visual[field])) {
        issues.push(`${rarity} rarity ${field} should match runtime ${visual[field]}`);
      }
    });
  });
  if (!manifestRarities.Mythic || manifestRarities.Mythic.status !== 'future') {
    issues.push('Mythic should remain marked as a future item rarity until runtime support exists');
  }
}

function validateItemAssets(data, manifest, issues) {
  const { catalogById } = getItemVisualData(data);
  const iconContract = manifest.runtime && manifest.runtime.inventoryIcon || {};
  const cardContract = manifest.runtime && manifest.runtime.cardIcon || {};
  const categories = manifest.categories || {};
  const themes = manifest.themes || {};
  const rarities = manifest.rarities || {};
  const requiredIconWidth = Number(iconContract.width || 0);
  const requiredIconHeight = Number(iconContract.height || 0);

  Object.entries(data.ITEM_ASSETS || {}).forEach(([itemId, assetPath]) => {
    if (!assetPath || typeof assetPath !== 'string') {
      issues.push(`${itemId} should have an icon path`);
      return;
    }
    if (!assetPath.startsWith(`${iconContract.folder}/`)) {
      issues.push(`${itemId} icon path should stay under ${iconContract.folder}`);
    }
    if (!assetPath.endsWith(`${kebabFromId(itemId)}.png`)) {
      issues.push(`${itemId} icon filename should be ${kebabFromId(itemId)}.png`);
    }
    if (!fileExists(assetPath)) {
      issues.push(`${itemId} icon file is missing at ${assetPath}`);
      return;
    }
    try {
      const dimensions = readPngDimensions(assetPath);
      if (dimensions.width !== requiredIconWidth || dimensions.height !== requiredIconHeight) {
        issues.push(`${itemId} icon should be ${requiredIconWidth}x${requiredIconHeight}, got ${dimensions.width}x${dimensions.height}`);
      }
    } catch (error) {
      issues.push(error.message);
    }

    const definition = catalogById.get(itemId);
    const category = inferCategory(itemId, definition, manifest);
    const theme = inferTheme(itemId, definition, manifest);
    const rarity = inferRarity(definition, manifest);
    if (!category || !categories[category]) issues.push(`${itemId} should resolve to a valid visual category`);
    if (!theme || !themes[theme]) issues.push(`${itemId} should resolve to a valid visual theme`);
    if (!rarity || !rarities[rarity]) issues.push(`${itemId} should resolve to a valid rarity or explicit default`);
  });

  Object.entries(data.CARD_ASSETS || {}).forEach(([cardId, assetPath]) => {
    if (!assetPath || typeof assetPath !== 'string') {
      issues.push(`${cardId} should have a card icon path`);
      return;
    }
    if (!assetPath.startsWith(`${cardContract.folder}/`)) {
      issues.push(`${cardId} card icon path should stay under ${cardContract.folder}`);
    }
    if (!fileExists(assetPath)) {
      issues.push(`${cardId} card icon file is missing at ${assetPath}`);
      return;
    }
    try {
      const dimensions = readPngDimensions(assetPath);
      if (dimensions.width !== Number(cardContract.width) || dimensions.height !== Number(cardContract.height)) {
        issues.push(`${cardId} card icon should be ${cardContract.width}x${cardContract.height}, got ${dimensions.width}x${dimensions.height}`);
      }
    } catch (error) {
      issues.push(error.message);
    }
    const definition = catalogById.get(cardId);
    const category = inferCategory(cardId, definition, manifest);
    const theme = inferTheme(cardId, definition, manifest) || 'card_collection';
    const rarity = inferRarity(definition, manifest);
    if (!category || !categories[category]) issues.push(`${cardId} should resolve to a valid card visual category`);
    if (!theme || !themes[theme]) issues.push(`${cardId} should resolve to a valid card visual theme`);
    if (!rarity || !rarities[rarity]) issues.push(`${cardId} should resolve to a valid card rarity`);
  });
}

function validatePromptsAndDocs(manifest, issues) {
  const guidePath = manifest.sourceOfTruth || 'ITEM_VISUAL_DESIGN_GUIDE.md';
  if (!fileExists(guidePath)) {
    issues.push(`${guidePath} is missing`);
    return;
  }
  const guide = fs.readFileSync(path.join(ROOT, guidePath), 'utf8');
  [
    'img/project-starfall/items/item-visual-manifest.json',
    'npm run validate:project-starfall-item-visuals',
    'img/project-starfall/items/pickups/',
    'img/project-starfall/animations/item-vfx/',
    'asset-sources/project-starfall/prompts/item-visual-prompts.md'
  ].forEach((needle) => {
    if (!guide.includes(needle)) issues.push(`${guidePath} should reference ${needle}`);
  });

  const promptTemplates = manifest.promptTemplates || {};
  Object.entries(promptTemplates).forEach(([promptId, prompt]) => {
    const promptPath = String(prompt && prompt.path || '').split('#')[0];
    if (!promptPath) {
      issues.push(`Prompt template ${promptId} should declare a path`);
      return;
    }
    if (!fileExists(promptPath)) issues.push(`Prompt template ${promptId} file is missing at ${promptPath}`);
  });

  const promptDocPath = 'asset-sources/project-starfall/prompts/item-visual-prompts.md';
  if (fileExists(promptDocPath)) {
    const promptDoc = fs.readFileSync(path.join(ROOT, promptDocPath), 'utf8');
    ['Master Item Art Prompt', 'Master Negative Prompt', 'World Pickup Sprite Prompt', 'Animated Pickup Sprite Sheet Prompt'].forEach((heading) => {
      if (!promptDoc.includes(heading)) issues.push(`${promptDocPath} should include ${heading}`);
    });
  }
}

function validateRuntimePolicy(manifest, issues) {
  const worldPickup = manifest.runtime && manifest.runtime.worldPickup || {};
  if (worldPickup.currentMode !== 'reuse_inventory_icon') {
    issues.push('World pickup currentMode should remain reuse_inventory_icon until renderer data support is added');
  }
  if (!worldPickup.generatedFolder || !pathIsDirectory(worldPickup.generatedFolder)) {
    issues.push('World pickup generatedFolder should exist');
  }
  if (Number(worldPickup.tierAuraInnerIconPx) > Number(worldPickup.tierAuraBoxPx)) {
    issues.push('tierAuraInnerIconPx should fit inside tierAuraBoxPx');
  }
  if (Number(worldPickup.stackableInnerIconPx) > Number(worldPickup.stackableBoxPx)) {
    issues.push('stackableInnerIconPx should fit inside stackableBoxPx');
  }
}

function validateGeneratedImage(relativePath, expectedWidth, expectedHeight, label, issues) {
  if (!fileExists(relativePath)) {
    issues.push(`${label} is missing at ${relativePath}`);
    return;
  }
  try {
    const dimensions = readPngDimensions(relativePath);
    if (dimensions.width !== expectedWidth || dimensions.height !== expectedHeight) {
      issues.push(`${label} should be ${expectedWidth}x${expectedHeight}, got ${dimensions.width}x${dimensions.height}`);
    }
    if (dimensions.bitDepth !== 8 || ![4, 6].includes(dimensions.colorType)) {
      issues.push(`${label} should be an 8-bit alpha PNG`);
    }
  } catch (error) {
    issues.push(error.message);
  }
}

function validateGeneratedAssets(data, manifest, issues) {
  const generatedAssets = manifest.generatedAssets || {};
  const pickupConfig = generatedAssets.pickupSprites || {};
  const pickupFolder = pickupConfig.folder || '';
  const pickupWidth = Number(pickupConfig.width || 0);
  const pickupHeight = Number(pickupConfig.height || 0);
  if (pickupFolder) {
    Object.keys(data.ITEM_ASSETS || {}).forEach((itemId) => {
      validateGeneratedImage(
        `${pickupFolder}/pickup-${kebabFromId(itemId)}.png`,
        pickupWidth,
        pickupHeight,
        `${itemId} generated pickup sprite`,
        issues
      );
    });
    Object.keys(data.CARD_ASSETS || {}).forEach((cardId) => {
      validateGeneratedImage(
        `${pickupFolder}/pickup-card-${kebabFromId(cardId)}.png`,
        pickupWidth,
        pickupHeight,
        `${cardId} generated card pickup sprite`,
        issues
      );
    });
  }

  const rarityFrameConfig = generatedAssets.rarityFrames || {};
  Object.keys(data.ITEM_RARITY_VISUALS || {}).forEach((rarity) => {
    validateGeneratedImage(
      `${rarityFrameConfig.folder}/frame-rarity-${kebabFromId(rarity)}.png`,
      Number(rarityFrameConfig.width || 0),
      Number(rarityFrameConfig.height || 0),
      `${rarity} generated rarity frame`,
      issues
    );
  });

  const markerConfig = generatedAssets.categoryMarkers || {};
  Object.keys(manifest.categories || {}).forEach((category) => {
    validateGeneratedImage(
      `${markerConfig.folder}/marker-category-${kebabFromId(category)}.png`,
      Number(markerConfig.width || 0),
      Number(markerConfig.height || 0),
      `${category} generated category marker`,
      issues
    );
  });

  const vfxConfig = generatedAssets.itemVfx || {};
  Object.keys(data.ITEM_RARITY_VISUALS || {}).forEach((rarity) => {
    const frameCount = Number(vfxConfig.rarityDropFrames || 0);
    validateGeneratedImage(
      `${vfxConfig.folder}/vfx-rarity-${kebabFromId(rarity)}-drop.png`,
      Number(vfxConfig.width || 0) * frameCount,
      Number(vfxConfig.height || 0),
      `${rarity} generated drop VFX sheet`,
      issues
    );
  });
  (vfxConfig.sheets || []).forEach((sheet) => {
    validateGeneratedImage(
      `${vfxConfig.folder}/${sheet.file}`,
      Number(vfxConfig.width || 0) * Number(sheet.frames || 0),
      Number(vfxConfig.height || 0),
      `${sheet.file} generated item VFX sheet`,
      issues
    );
  });

  const animatedConfig = generatedAssets.animatedPickups || {};
  (animatedConfig.sheets || []).forEach((sheet) => {
    validateGeneratedImage(
      `${animatedConfig.folder}/${sheet.file}`,
      Number(animatedConfig.width || 0) * Number(sheet.frames || 0),
      Number(animatedConfig.height || 0),
      `${sheet.file} generated animated pickup sheet`,
      issues
    );
  });
}

function validateProjectStarfallItemVisuals(data, options) {
  const settings = options || {};
  const issues = [];
  const warnings = [];
  const manifest = readJson(MANIFEST_RELATIVE_PATH, issues);
  const starfallData = data || require('../js/games/project-starfall/project-starfall-data.js');
  if (!manifest) return { issues, warnings, manifest: null };

  validateFolders(manifest, issues);
  validateRarities(starfallData, manifest, issues);
  validateItemAssets(starfallData, manifest, issues);
  validatePromptsAndDocs(manifest, issues);
  validateRuntimePolicy(manifest, issues);
  validateGeneratedAssets(starfallData, manifest, issues);

  const metadataFields = manifest.metadataFields || {};
  [
    'icon_path',
    'pickup_sprite_path',
    'animated_pickup_path',
    'rarity',
    'category',
    'theme',
    'motif',
    'visual_tags',
    'vfx_path',
    'ui_frame',
    'tooltip_icon',
    'world_pickup_scale',
    'inventory_scale',
    'drop_effect',
    'pickup_effect'
  ].forEach((field) => {
    if (!metadataFields[field]) issues.push(`Manifest metadataFields should include ${field}`);
  });

  if (settings.includeWarnings) {
    []
      .concat(starfallData.SHOP_ITEMS || [])
      .concat(starfallData.RANDOM_EQUIPMENT_ITEMS || [])
      .concat(starfallData.BOSS_EQUIPMENT_ITEMS || [])
      .filter((item) => {
        const directAsset = starfallData.ITEM_ASSETS && starfallData.ITEM_ASSETS[item.id];
        return item.assetId && item.assetId !== item.id && item.asset !== directAsset;
      })
      .forEach((item) => {
        warnings.push(`${item.id} shares icon asset ${item.assetId}; replace with direct art when finalizing that item identity`);
      });
  }

  return { issues, warnings, manifest };
}

if (require.main === module) {
  const result = validateProjectStarfallItemVisuals(null, { includeWarnings: process.argv.includes('--warnings') });
  if (result.warnings.length) {
    console.warn(result.warnings.join('\n'));
  }
  if (result.issues.length) {
    console.error(result.issues.join('\n'));
    process.exit(1);
  }
  console.log('Project Starfall item visual validation passed.');
}

module.exports = {
  validateProjectStarfallItemVisuals
};
