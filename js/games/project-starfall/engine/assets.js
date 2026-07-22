(function initProjectStarfallEngineAssets(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreAssets = (typeof require === 'function' ? require('../core/assets.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getAssetSourcePath = CoreAssets.createAssetSourcePathGetter
    ? CoreAssets.createAssetSourcePathGetter({ cache: false, includeSheetSize: true })
    : function getAssetSourcePathFallback(assetPath) {
        return String(assetPath || '').split('#')[0].trim();
      };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatAssetPreviewLabel(value) {
    const text = normalizeId(value);
    if (!text) return 'Asset';
    return text
      .replace(/[_-]+/g, ' ')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
      .replace(/\s+/g, ' ')
      .trim()
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function getAnimationPreviewMeta(animation) {
    if (!animation || !animation.sheet || !animation.states) return null;
    const states = Object.keys(animation.states).map((id) => {
      const state = animation.states[id] || {};
      const meta = {
        id,
        label: formatAssetPreviewLabel(id),
        row: Number(state.row || 0),
        frames: Math.max(1, Number(state.frames || 1) || 1),
        fps: Math.max(1, Number(state.fps || 1) || 1),
        loop: !!state.loop,
        loopDelay: Math.max(0, Number(state.loopDelay || 0) || 0)
      };
      if (Array.isArray(state.sequence) && state.sequence.length) {
        meta.sequence = state.sequence.slice();
      }
      return meta;
    });
    return {
      sheet: normalizeId(animation.sheet),
      frameWidth: Math.max(1, Number(animation.frameWidth || 160) || 160),
      frameHeight: Math.max(1, Number(animation.frameHeight || 160) || 160),
      states
    };
  }

  function makeEquipmentAtlasPreviewAnimation(atlas) {
    if (!atlas || !atlas.sheet) return null;
    const frameCount = Math.max(1, Array.isArray(atlas.angles) ? atlas.angles.length : 8);
    const variants = Array.isArray(atlas.variants) && atlas.variants.length
      ? atlas.variants
      : ['default'];
    const states = variants.reduce((result, variant, row) => {
      const id = normalizeId(variant) || `variant${row + 1}`;
      result[id] = {
        row,
        frames: frameCount,
        fps: 4,
        loop: true,
        loopDelay: 0.25
      };
      return result;
    }, {});
    return {
      sheet: atlas.sheet,
      frameWidth: Math.max(1, Number(atlas.frameWidth || 128) || 128),
      frameHeight: Math.max(1, Number(atlas.frameHeight || atlas.frameWidth || 128) || 128),
      states
    };
  }

  function normalizeAssetSourceType(value) {
    return normalizeId(value).toLowerCase() === 'procedural' ? 'procedural' : 'ai';
  }

  function getAssetPreviewSourceLabel(sourceType) {
    return normalizeAssetSourceType(sourceType) === 'ai' ? 'AI-generated' : 'Procedural';
  }

  function getAssetBackupPath(assetPath, data, fallbackData) {
    const path = normalizeId(getAssetSourcePath(assetPath));
    const backupPaths = data && data.ASSET_BACKUP_PATHS || fallbackData && fallbackData.ASSET_BACKUP_PATHS || {};
    return path && backupPaths[path] ? normalizeId(backupPaths[path]) : '';
  }

  function imageReady(image) {
    return !!(image && image.complete && image.naturalWidth > 0);
  }

  function getResolvedAssetImage(runtime, assetPath, fallbackData) {
    if (!runtime) return null;
    const path = normalizeId(getAssetSourcePath(assetPath));
    if (!path) return null;
    if (imageReady(runtime.assets && runtime.assets[path])) return runtime.assets[path];
    const backupPath = getAssetBackupPath(path, runtime.data, fallbackData);
    if (backupPath && runtime.failedAssets && runtime.failedAssets[path] && imageReady(runtime.assets && runtime.assets[backupPath])) {
      return runtime.assets[backupPath];
    }
    return null;
  }

  function addAssetPath(paths, path) {
    const assetPath = normalizeId(getAssetSourcePath(path));
    if (assetPath) paths.push(assetPath);
  }

  function addAssetPathBackup(paths, path, data, fallbackData) {
    const backupPath = getAssetBackupPath(path, data, fallbackData);
    if (backupPath) paths.push(backupPath);
  }

  function collectAnimationAssetPaths(value, paths) {
    if (!value || typeof value !== 'object') return;
    Object.entries(value).forEach(([key, child]) => {
      if (key === 'sheet' && typeof child === 'string') {
        addAssetPath(paths, child);
        return;
      }
      collectAnimationAssetPaths(child, paths);
    });
  }

  function collectEnvironmentAssetPaths(value, paths) {
    if (!value || typeof value !== 'object') return;
    Object.entries(value).forEach(([key, child]) => {
      if (key === 'path' && typeof child === 'string') {
        addAssetPath(paths, child);
        return;
      }
      collectEnvironmentAssetPaths(child, paths);
    });
  }

  function collectMapEnvironmentAssetPaths(map, data, paths) {
    const source = data || {};
    const profile = map && (map.environment ||
      source.MAP_ENVIRONMENT_PROFILES && source.MAP_ENVIRONMENT_PROFILES[map.id] ||
      source.MAP_ENVIRONMENT_PROFILES && source.MAP_ENVIRONMENT_PROFILES.greenrootMeadow);
    const assets = source.ENVIRONMENT_ASSETS || {};
    ['terrain', 'props', 'ramps'].forEach((kind) => {
      const id = profile && (profile[kind] || (kind === 'ramps' ? profile.terrain : ''));
      const asset = id && assets[kind] && assets[kind][id];
      addAssetPath(paths, asset && asset.path);
    });
    collectEnvironmentAssetPaths(source.ENVIRONMENT_STRUCTURE_ASSETS, paths);
  }

  function collectAssetPaths(data, fallbackData) {
    const source = data || fallbackData || {};
    const paths = [];
    addAssetPath(paths, source.WORLD_MAP_ATLAS && source.WORLD_MAP_ATLAS.asset);
    addAssetPath(paths, source.GENERIC_PLAYER_ASSET);
    collectAnimationAssetPaths(source.GENERIC_PLAYER_ANIMATION_ASSET, paths);
    collectAnimationAssetPaths(source.PET_ANIMATION_ASSET, paths);
    addAssetPath(paths, source.CHARACTER_SLOT_PEDESTAL_ASSET);
    Object.values(source.UI_ASSETS || {}).forEach((assetPath) => addAssetPath(paths, assetPath));
    Object.values(source.MENU_ICON_ASSETS || {}).forEach((assetPath) => addAssetPath(paths, assetPath));
    Object.values(source.BASE_CLASSES || {}).forEach((classData) => addAssetPath(paths, classData.asset));
    Object.values(source.ADVANCED_CLASSES || {}).forEach((classData) => addAssetPath(paths, classData.asset));
    (source.ENEMIES || []).forEach((enemy) => addAssetPath(paths, enemy.asset));
    (source.MAPS || []).forEach((map) => {
      addAssetPath(paths, map.asset);
      (map.stations || []).forEach((station) => addAssetPath(paths, station.asset));
    });
    Object.values(source.CLASS_TRIAL_ASSETS || {}).forEach((assetPath) => addAssetPath(paths, assetPath));
    collectEnvironmentAssetPaths(source.ENVIRONMENT_ASSETS, paths);
    collectEnvironmentAssetPaths(source.ENVIRONMENT_STRUCTURE_ASSETS, paths);
    Object.values(source.ITEM_ASSETS || {}).forEach((assetPath) => addAssetPath(paths, assetPath));
    Object.values(source.CARD_ASSETS || {}).forEach((assetPath) => addAssetPath(paths, assetPath));
    (source.SHOP_ITEMS || []).forEach((item) => addAssetPath(paths, item.asset));
    (source.RANDOM_EQUIPMENT_ITEMS || []).forEach((item) => addAssetPath(paths, item.asset));
    (source.BOSS_EQUIPMENT_ITEMS || []).forEach((item) => addAssetPath(paths, item.asset));
    (source.SKILLS || []).forEach((skill) => addAssetPath(paths, skill.iconAsset));
    collectAnimationAssetPaths(source.ANIMATION_ASSETS, paths);
    collectAnimationAssetPaths(source.EQUIPMENT_VISUALS, paths);
    if (source.PRELOAD_ASSET_BACKUPS) {
      Object.keys(source.ASSET_BACKUP_PATHS || {}).forEach((assetPath) => addAssetPathBackup(paths, assetPath, source, fallbackData));
    }
    return Array.from(new Set(paths));
  }

  function createAssetLoadProgress(total, settled, loaded, failed, complete) {
    const assetTotal = Math.max(0, Math.floor(Number(total || 0)));
    const assetSettled = clamp(Math.floor(Number(settled || 0)), 0, assetTotal);
    const done = !!complete || assetTotal <= 0 || assetSettled >= assetTotal;
    return {
      total: assetTotal,
      settled: assetSettled,
      loaded: Math.max(0, Math.floor(Number(loaded || 0))),
      failed: Math.max(0, Math.floor(Number(failed || 0))),
      percent: done ? 100 : clamp(Math.max(assetSettled > 0 ? 1 : 0, Math.round((assetSettled / Math.max(1, assetTotal)) * 100)), 0, 99),
      complete: done
    };
  }

  function createAssetPreviewCollector() {
    const entries = new Map();
    const add = (entry) => {
      const path = normalizeId(entry && entry.path || entry && entry.animation && entry.animation.sheet);
      if (!path) return null;
      const entrySourceType = normalizeAssetSourceType(entry && entry.sourceType);
      let asset = entries.get(path);
      if (!asset) {
        asset = {
          id: path,
          path,
          label: formatAssetPreviewLabel(entry.label || path.split('/').pop() || path),
          labels: [],
          category: normalizeId(entry.category) || 'Other',
          categories: [],
          kind: entry.kind || 'image',
          tags: [],
          sourceIds: [],
          sourceType: entrySourceType,
          sourceLabel: getAssetPreviewSourceLabel(entrySourceType),
          animation: null
        };
        entries.set(path, asset);
      }
      if (entrySourceType === 'ai' && asset.sourceType !== 'ai' && asset.sourceType !== 'procedural') {
        asset.sourceType = 'ai';
        asset.sourceLabel = getAssetPreviewSourceLabel('ai');
      }
      const label = formatAssetPreviewLabel(entry.label || asset.label);
      if (label && !asset.labels.includes(label)) asset.labels.push(label);
      if (!asset.label || asset.label === 'Asset') asset.label = label;
      const category = normalizeId(entry.category) || 'Other';
      if (category && !asset.categories.includes(category)) asset.categories.push(category);
      if (asset.category === 'Other' && category !== 'Other') asset.category = category;
      (entry.tags || []).map(normalizeId).filter(Boolean).forEach((tag) => {
        if (!asset.tags.includes(tag)) asset.tags.push(tag);
      });
      const sourceId = normalizeId(entry.sourceId);
      if (sourceId && !asset.sourceIds.includes(sourceId)) asset.sourceIds.push(sourceId);
      const animation = getAnimationPreviewMeta(entry.animation);
      if (animation) {
        asset.kind = 'animation';
        asset.animation = animation;
      }
      return asset;
    };
    return {
      add,
      list() {
        return Array.from(entries.values())
          .map((entry) => Object.assign({}, entry, {
            labels: entry.labels.slice(),
            categories: entry.categories.slice(),
            tags: entry.tags.slice(),
            sourceIds: entry.sourceIds.slice(),
            animation: entry.animation ? Object.assign({}, entry.animation, {
              states: entry.animation.states.map((state) => Object.assign({}, state))
            }) : null
          }))
          .sort((a, b) => `${a.category}:${a.label}:${a.path}`.localeCompare(`${b.category}:${b.label}:${b.path}`));
      },
      has(path) {
        return entries.has(normalizeId(path));
      }
    };
  }

  function collectAssetPreviewAnimations(value, category, labelPrefix, collector, sourceType = 'ai') {
    if (!value || typeof value !== 'object') return;
    if (value.sheet && value.states) {
      collector.add({
        path: value.sheet,
        label: labelPrefix,
        category,
        kind: 'animation',
        animation: value,
        sourceType,
        tags: [labelPrefix]
      });
      return;
    }
    Object.entries(value).forEach(([key, child]) => {
      collectAssetPreviewAnimations(child, category, labelPrefix ? `${labelPrefix} ${key}` : key, collector, sourceType);
    });
  }

  function collectAssetPreviewEnvironment(value, category, labelPrefix, collector, sourceType) {
    if (!value || typeof value !== 'object') return;
    if (typeof value.path === 'string') {
      collector.add({
        path: value.path,
        label: value.name || value.id || labelPrefix,
        category,
        kind: 'image',
        sourceId: value.id || labelPrefix,
        sourceType,
        tags: [labelPrefix]
      });
    }
    Object.entries(value).forEach(([key, child]) => {
      if (key === 'path') return;
      collectAssetPreviewEnvironment(child, category, labelPrefix ? `${labelPrefix} ${key}` : key, collector, sourceType);
    });
  }

  function createAssetPreviewCatalog(data, options) {
    const source = data || {};
    const settings = options || {};
    const collector = createAssetPreviewCollector();
    const add = (path, label, category, addOptions) => collector.add(Object.assign({
      path,
      label,
      category,
      kind: 'image',
      sourceId: label
    }, addOptions || {}));

    add(source.WORLD_MAP_ATLAS && source.WORLD_MAP_ATLAS.asset, 'World Map Atlas', 'Maps', { tags: ['world map', 'atlas'], sourceType: 'ai' });
    add(source.GENERIC_PLAYER_ASSET, 'Generic Player Portrait', 'Players', { sourceType: 'ai' });
    add(source.CHARACTER_SLOT_PEDESTAL_ASSET, 'Character Slot Pedestal', 'UI', { sourceType: 'ai', tags: ['character select', 'pedestal'] });
    Object.entries(source.UI_ASSETS || {}).forEach(([id, path]) => add(path, id, 'UI', { sourceType: 'ai' }));
    Object.entries(source.MENU_ICON_ASSETS || {}).forEach(([id, path]) => add(path, `${id} menu icon`, 'UI', { tags: ['menu icon'], sourceType: 'ai' }));
    Object.entries(source.CLASS_ASSETS || {}).forEach(([id, path]) => add(path, `${id} class portrait`, 'Players', { sourceId: id, sourceType: 'ai' }));
    Object.values(source.BASE_CLASSES || {}).forEach((classData) => {
      add(classData.asset, `${classData.name || classData.id} portrait`, 'Players', { sourceId: classData.id, sourceType: 'ai' });
      if (classData.animation) collector.add({ path: classData.animation.sheet, label: `${classData.name || classData.id} animation`, category: 'Players', kind: 'animation', animation: classData.animation, sourceId: classData.id, sourceType: 'ai' });
    });
    Object.values(source.ADVANCED_CLASSES || {}).forEach((classData) => {
      add(classData.asset, `${classData.name || classData.id} portrait`, 'Players', { sourceId: classData.id, sourceType: 'ai' });
      if (classData.animation) collector.add({ path: classData.animation.sheet, label: `${classData.name || classData.id} animation`, category: 'Players', kind: 'animation', animation: classData.animation, sourceId: classData.id, sourceType: 'ai' });
    });
    Object.entries(source.PLAYER_ANIMATION_ASSETS || {}).forEach(([id, animation]) => collector.add({ path: animation && animation.sheet, label: `${id} player animation`, category: 'Players', kind: 'animation', animation, sourceId: id, sourceType: 'ai' }));
    if (source.PET_ANIMATION_ASSET) {
      collector.add({
        path: source.PET_ANIMATION_ASSET.sheet,
        label: 'Starfall Fox pet animation',
        category: 'Pets',
        kind: 'animation',
        animation: source.PET_ANIMATION_ASSET,
        sourceId: 'starfallFox',
        sourceType: 'ai',
        tags: ['pet', 'companion']
      });
    }
    Object.entries(source.EQUIPMENT_VISUALS || {}).forEach(([id, visual]) => {
      const atlasAnimation = makeEquipmentAtlasPreviewAnimation(visual && visual.atlas);
      const animation = atlasAnimation || visual && visual.animation;
      collector.add({
        path: animation && animation.sheet,
        label: `${visual && visual.name || id} equipment visual`,
        category: 'Equipment',
        kind: 'animation',
        animation,
        sourceId: id,
        sourceType: 'procedural',
        tags: [visual && visual.layer, visual && visual.kind, atlasAnimation ? 'atlas' : 'sheet']
      });
    });
    Object.entries(source.ENEMY_ANIMATION_ASSETS || {}).forEach(([id, animation]) => collector.add({ path: animation && animation.sheet, label: `${id} enemy animation`, category: 'Enemies', kind: 'animation', animation, sourceId: id, sourceType: 'ai' }));
    Object.entries(source.ENEMY_PROJECTILE_ANIMATION_ASSETS || {}).forEach(([id, animation]) => collector.add({ path: animation && animation.sheet, label: `${id} enemy projectile`, category: 'Enemies', kind: 'animation', animation, sourceId: id, sourceType: 'ai', tags: ['projectile'] }));
    (source.ENEMIES || []).forEach((enemy) => {
      add(enemy.asset, `${enemy.name || enemy.id} enemy`, 'Enemies', { sourceId: enemy.id, sourceType: 'ai', tags: [enemy.behavior, enemy.mapRole] });
      if (enemy.animation) collector.add({ path: enemy.animation.sheet, label: `${enemy.name || enemy.id} animation`, category: 'Enemies', kind: 'animation', animation: enemy.animation, sourceId: enemy.id, sourceType: 'ai' });
    });
    Object.entries(source.MAP_ASSETS || {}).forEach(([id, path]) => add(path, `${id} map background`, 'Backgrounds', { sourceId: id, sourceType: 'ai', tags: ['map background'] }));
    (source.MAPS || []).forEach((map) => {
      add(map.asset, `${map.name || map.id} background`, 'Backgrounds', { sourceId: map.id, sourceType: 'ai', tags: [map.region, map.biome, 'map'] });
      (map.stations || []).forEach((station) => add(station.asset, `${station.name || station.id} station`, 'Environment', { sourceId: station.id, sourceType: 'ai', tags: ['station', map.id] }));
    });
    Object.entries(source.CLASS_TRIAL_ASSETS || {}).forEach(([id, path]) => add(path, `${id} trial background`, 'Backgrounds', { sourceId: id, sourceType: 'ai', tags: ['class trial'] }));
    collectAssetPreviewEnvironment(source.ENVIRONMENT_ASSETS, 'Environment', 'environment', collector, 'ai');
    collectAssetPreviewEnvironment(source.ENVIRONMENT_STRUCTURE_ASSETS, 'Environment', 'environment structures', collector, 'ai');
    Object.entries(source.ITEM_ASSETS || {}).forEach(([id, path]) => add(path, `${id} item`, 'Items', { sourceId: id, sourceType: 'ai' }));
    Object.entries(source.CARD_ASSETS || {}).forEach(([id, path]) => {
      const card = (source.CARD_DEFINITIONS || []).find((definition) => definition && definition.id === id) || {};
      add(path, `${card.name || id} card icon`, 'Cards', { sourceId: id, sourceType: 'ai', tags: [card.rarity, ...(card.tags || [])] });
    });
    (source.SHOP_ITEMS || []).forEach((item) => add(item.asset || source.ITEM_ASSETS && source.ITEM_ASSETS[item.id], `${item.name || item.id} shop item`, 'Items', { sourceId: item.id, sourceType: 'ai', tags: ['shop'] }));
    (source.RANDOM_EQUIPMENT_ITEMS || []).forEach((item) => add(item.asset || source.ITEM_ASSETS && source.ITEM_ASSETS[item.id], `${item.name || item.id} equipment`, 'Equipment', { sourceId: item.id, sourceType: 'ai', tags: [item.slot, item.rarity] }));
    (source.BOSS_EQUIPMENT_ITEMS || []).forEach((item) => add(item.asset || source.ITEM_ASSETS && source.ITEM_ASSETS[item.id], `${item.name || item.id} boss equipment`, 'Equipment', { sourceId: item.id, sourceType: 'ai', tags: [item.slot, item.rarity, 'boss'] }));
    (source.SKILLS || []).forEach((skill) => add(skill.iconAsset, `${skill.name || skill.id} skill icon`, 'Skills', { sourceId: skill.id, sourceType: 'ai', tags: [skill.owner, skill.type] }));
    collectAssetPreviewAnimations(source.FX_ANIMATION_ASSETS, 'Combat FX', 'fx', collector);
    collectAssetPreviewAnimations(source.SKILL_FX_ANIMATION_ASSETS, 'Combat FX', 'skill fx', collector);
    collectAssetPreviewAnimations(source.BASIC_ATTACK_FX_ANIMATION_ASSETS, 'Combat FX', 'basic attack fx', collector);
    collectAssetPreviewAnimations(source.ENEMY_COMBAT_FX_ANIMATION_ASSETS, 'Combat FX', 'enemy combat fx', collector);
    collectAssetPreviewAnimations(source.ENEMY_PROJECTILE_ANIMATION_ASSETS, 'Combat FX', 'enemy projectile fx', collector);
    collectAssetPreviewAnimations(source.PORTAL_ANIMATION_ASSETS, 'Portals', 'portal', collector);
    (Array.isArray(settings.assetPaths) ? settings.assetPaths : []).forEach((path) => {
      if (!collector.has(path)) add(path, path.split('/').pop() || path, 'Other', { sourceId: path });
    });
    return collector.list();
  }

  const api = {
    formatAssetPreviewLabel,
    getAnimationPreviewMeta,
    makeEquipmentAtlasPreviewAnimation,
    normalizeAssetSourceType,
    getAssetPreviewSourceLabel,
    getAssetBackupPath,
    getResolvedAssetImage,
    addAssetPath,
    addAssetPathBackup,
    collectAssetPaths,
    collectAnimationAssetPaths,
    collectEnvironmentAssetPaths,
    collectMapEnvironmentAssetPaths,
    imageReady,
    createAssetLoadProgress,
    createAssetPreviewCollector,
    collectAssetPreviewAnimations,
    collectAssetPreviewEnvironment,
    createAssetPreviewCatalog
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.assets = Object.assign({}, modules.assets || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
