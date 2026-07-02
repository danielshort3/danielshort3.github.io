(function initProjectStarfallEngineMapModifiers(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const seededUnit = CoreMath.seededUnit || function seededUnitFallback(seed, salt) {
    let hash = 2166136261;
    const text = `${String(seed || '')}:${String(salt || '')}`;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    hash ^= hash << 13;
    hash ^= hash >>> 17;
    hash ^= hash << 5;
    return ((hash >>> 0) % 10000) / 10000;
  };
  const seededPick = CoreMath.seededPick || function seededPickFallback(items, seed, salt) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return '';
    return options[Math.floor(seededUnit(seed, salt) * options.length) % options.length];
  };

  function getMapModifierData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createMapModifierState(value, options) {
    const data = getMapModifierData(options);
    const source = value && typeof value === 'object' ? value : {};
    const validModifierIds = new Set((data.MAP_MODIFIERS || []).map((modifier) => modifier.id));
    const activeByMapId = {};
    Object.entries(source.activeByMapId || source.activeByMap || {}).forEach(([mapId, ids]) => {
      const key = normalizeId(mapId);
      if (!key || !getById(data.MAPS || [], key)) return;
      const list = (Array.isArray(ids) ? ids : [ids])
        .map(normalizeId)
        .filter((id) => validModifierIds.has(id));
      if (list.length) activeByMapId[key] = Array.from(new Set(list)).slice(0, 2);
    });
    const unlockedIds = Array.isArray(source.unlockedIds)
      ? source.unlockedIds.map(normalizeId).filter((id) => validModifierIds.has(id))
      : (data.MAP_MODIFIERS || []).map((modifier) => modifier.id);
    return {
      activeByMapId,
      unlockedIds: Array.from(new Set(unlockedIds))
    };
  }

  function getMapModifierType(map) {
    if (!map) return 'field';
    if (map.id === 'endlessRift' || map.endlessScaling) return 'rift';
    if (map.isDungeon) return 'dungeon';
    return 'field';
  }

  function createDefaultMapModifierIds(map, playerLevel, options) {
    const data = getMapModifierData(options);
    if (!map || map.safeZone) return [];
    const type = getMapModifierType(map);
    const source = data.MAP_MODIFIERS || [];
    const optionsForType = source.filter((modifier) => {
      const types = Array.isArray(modifier.mapTypes) ? modifier.mapTypes : [];
      return !types.length || types.includes(type);
    });
    if (!optionsForType.length) return [];
    const level = Number(playerLevel || 1);
    const seed = `${map.id}:${type}:${level}`;
    const first = seededPick(optionsForType, seed, 'primary');
    const secondOptions = optionsForType.filter((modifier) => modifier && modifier.id !== (first && first.id));
    const second = type === 'rift' || seededUnit(seed, 'secondary') > 0.62 ? seededPick(secondOptions, seed, 'secondary') : null;
    return [first, second].filter(Boolean).map((modifier) => modifier.id);
  }

  function getMapModifiersByIds(ids, options) {
    const data = getMapModifierData(options);
    const sourceIds = Array.isArray(ids) ? ids : [];
    const modifiers = [];
    const limit = Math.min(2, sourceIds.length);
    for (let index = 0; index < limit; index += 1) {
      const modifier = getById(data.MAP_MODIFIERS || [], sourceIds[index]);
      if (modifier) modifiers.push(modifier);
    }
    return modifiers;
  }

  function getMapModifierBonus(ids, key, options) {
    const modifiers = getMapModifiersByIds(ids, options);
    let sum = 0;
    for (let index = 0; index < modifiers.length; index += 1) {
      sum += Number(modifiers[index] && modifiers[index][key] || 0);
    }
    return sum;
  }

  function getMapModifierScale(ids, key, options) {
    const modifiers = getMapModifiersByIds(ids, options);
    let scale = 1;
    for (let index = 0; index < modifiers.length; index += 1) {
      scale *= Number(modifiers[index] && modifiers[index][key] || 1);
    }
    return scale;
  }

  function addRoundedStats(target, source) {
    Object.entries(source || {}).forEach(([key, value]) => {
      const amount = Math.round(Number(value || 0) || 0);
      if (amount) target[key] = (target[key] || 0) + amount;
    });
    return target;
  }

  function createMapModifierPlayerBonuses(ids, options) {
    const modifiers = getMapModifiersByIds(ids, options);
    const stats = {};
    for (let index = 0; index < modifiers.length; index += 1) {
      addRoundedStats(stats, modifiers[index].statBonuses || {});
    }
    return stats;
  }

  function getMapModifierSnapshotCacheKey(context) {
    const settings = context || {};
    const mapId = normalizeId(settings.mapId);
    const mapModifiers = createMapModifierState(settings.mapModifiers, settings);
    const activeByMapId = mapModifiers.activeByMapId || {};
    const stored = Array.isArray(activeByMapId[mapId]) ? activeByMapId[mapId].map(normalizeId).filter(Boolean).join(',') : '';
    const rift = settings.rift && typeof settings.rift === 'object' ? settings.rift : {};
    const mechanicEntry = rift.mapMechanics && rift.mapMechanics.byMapId && rift.mapMechanics.byMapId[mapId] || {};
    const revisions = settings.revisions || {};
    return [
      Number(revisions.world || 0),
      Number(revisions.guide || 0),
      mapId,
      Math.max(1, Number(settings.player && settings.player.level || 1) || 1),
      stored,
      Math.max(1, Number(rift.tier || 1) || 1),
      Math.max(1, Number(rift.bestTier || rift.tier || 1) || 1),
      Math.max(0, Number(rift.score || 0) || 0),
      Array.isArray(rift.mutationIds) ? rift.mutationIds.map(normalizeId).filter(Boolean).join(',') : '',
      normalizeId(mechanicEntry.activeSectionId),
      Math.round(Number(mechanicEntry.progress || 0) * 100) / 100,
      Number(mechanicEntry.completedCycles || 0),
      Number(mechanicEntry.eventCount || 0),
      Number(mechanicEntry.objectiveCount || 0),
      Number(mechanicEntry.surgeCount || 0),
      Number(mechanicEntry.antiCampStacks || 0),
      Math.round(Number(mechanicEntry.rewardScale || 1) * 100) / 100
    ].join('|');
  }

  function createMapModifierSnapshot(map, activeModifiers, riftSnapshot, mapMechanicSnapshot, options) {
    const settings = options || {};
    return {
      mapId: map && map.id || settings.mapId,
      mapType: getMapModifierType(map),
      active: (activeModifiers || []).map((modifier) => Object.assign({}, modifier)),
      rift: riftSnapshot,
      mapMechanic: mapMechanicSnapshot
    };
  }

  const api = {
    createMapModifierState,
    getMapModifierType,
    createDefaultMapModifierIds,
    getMapModifiersByIds,
    getMapModifierBonus,
    getMapModifierScale,
    createMapModifierPlayerBonuses,
    getMapModifierSnapshotCacheKey,
    createMapModifierSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.mapModifiers = Object.assign({}, modules.mapModifiers || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
