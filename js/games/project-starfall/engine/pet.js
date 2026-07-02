(function initProjectStarfallEnginePet(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };

  const PET_HP_POTION_ID = 'minor_health_potion';
  const PET_MP_POTION_ID = 'minor_resource_tonic';
  const PET_THRESHOLD_MIN = 5;
  const PET_THRESHOLD_MAX = 95;
  const PET_THRESHOLD_STEP = 5;
  const PET_DEFAULT_HP_THRESHOLD = 45;
  const PET_DEFAULT_MP_THRESHOLD = 35;
  const PET_RARITY_ORDER = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic', 'Relic']);

  function getPetData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function normalizePetThreshold(value, fallback) {
    const raw = Number(value);
    const base = Number.isFinite(raw) ? raw : Number(fallback);
    const stepped = Math.round(base / PET_THRESHOLD_STEP) * PET_THRESHOLD_STEP;
    return clamp(stepped, PET_THRESHOLD_MIN, PET_THRESHOLD_MAX);
  }

  let petPotionOptionsCache = null;

  function getPetPotionOptions(kind, options) {
    const data = getPetData(options);
    const id = normalizeId(kind);
    const source = data.CONSUMABLE_ITEMS || [];
    let cache = petPotionOptionsCache;
    if (!cache || cache.source !== source || cache.length !== source.length) {
      const hp = [];
      const mp = [];
      source.forEach((item) => {
        if (!item) return;
        if (item.hpPercent || item.hpFlat) hp.push(item);
        if (item.resourcePercent || item.resourceFlat) mp.push(item);
      });
      cache = { source, length: source.length, hp, mp };
      petPotionOptionsCache = cache;
    }
    if (id === 'hp') return cache.hp;
    if (id === 'mp') return cache.mp;
    return [];
  }

  function normalizePetPotionId(value, kind, options) {
    const settings = options || {};
    const id = normalizeId(value);
    const optionsForKind = getPetPotionOptions(kind, settings);
    if (optionsForKind.some((item) => item.id === id)) return id;
    const fallback = normalizeId(kind) === 'mp'
      ? normalizeId(settings.mpPotionId) || PET_MP_POTION_ID
      : normalizeId(settings.hpPotionId) || PET_HP_POTION_ID;
    if (optionsForKind.some((item) => item.id === fallback)) return fallback;
    return optionsForKind[0] ? optionsForKind[0].id : '';
  }

  function normalizePetLootFilters(value, options) {
    const settings = options || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : PET_RARITY_ORDER;
    const source = value && typeof value === 'object' ? value : {};
    const rarity = rarityOrder.includes(source.minEquipmentRarity) ? source.minEquipmentRarity : 'Common';
    return {
      currency: Object.prototype.hasOwnProperty.call(source, 'currency') ? !!source.currency : true,
      consumables: Object.prototype.hasOwnProperty.call(source, 'consumables') ? !!source.consumables : true,
      materials: Object.prototype.hasOwnProperty.call(source, 'materials') ? !!source.materials : true,
      cards: Object.prototype.hasOwnProperty.call(source, 'cards') ? !!source.cards : true,
      equipment: Object.prototype.hasOwnProperty.call(source, 'equipment') ? !!source.equipment : true,
      minEquipmentRarity: rarity
    };
  }

  function createPetState(value, options) {
    const settings = options || {};
    const source = value && typeof value === 'object' ? value : {};
    return {
      unlocked: !!source.unlocked,
      autoLoot: Object.prototype.hasOwnProperty.call(source, 'autoLoot') ? !!source.autoLoot : true,
      autoHp: Object.prototype.hasOwnProperty.call(source, 'autoHp') ? !!source.autoHp : true,
      autoMp: Object.prototype.hasOwnProperty.call(source, 'autoMp') ? !!source.autoMp : true,
      hpPotionId: normalizePetPotionId(source.hpPotionId, 'hp', settings),
      mpPotionId: normalizePetPotionId(source.mpPotionId, 'mp', settings),
      hpThreshold: normalizePetThreshold(source.hpThreshold, PET_DEFAULT_HP_THRESHOLD),
      mpThreshold: normalizePetThreshold(source.mpThreshold, PET_DEFAULT_MP_THRESHOLD),
      lootFilters: normalizePetLootFilters(source.lootFilters, settings),
      starterWhistleGranted: !!source.starterWhistleGranted
    };
  }

  function createPetRuntime() {
    return {
      loot: 0,
      hp: 0,
      mp: 0,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      facing: 1,
      grounded: false,
      groundedPlatformId: '',
      groundedPlatformIndex: -1,
      climbing: false,
      climbableId: '',
      pathTargetPlatformIndex: -1,
      nextJumpAt: 0,
      airRouteUntil: 0,
      airRouteVx: 0,
      dropThroughUntil: 0,
      dropThroughPlatformId: '',
      dropThroughPlatformIndex: -1,
      lastX: 0,
      stuckTime: 0,
      initialized: false,
      mode: 'follow',
      targetDropUid: '',
      lootRoutePlatformId: '',
      lootRoutePlatformIndex: -1,
      animationState: 'idle',
      animationStartedAt: 0,
      walkCycle: 0,
      hopStartedAt: 0,
      hopDuration: 0,
      hopHeight: 0,
      teleportUntil: 0
    };
  }

  const api = {
    PET_HP_POTION_ID,
    PET_MP_POTION_ID,
    PET_THRESHOLD_MIN,
    PET_THRESHOLD_MAX,
    PET_THRESHOLD_STEP,
    PET_DEFAULT_HP_THRESHOLD,
    PET_DEFAULT_MP_THRESHOLD,
    normalizePetThreshold,
    getPetPotionOptions,
    normalizePetPotionId,
    normalizePetLootFilters,
    createPetState,
    createPetRuntime
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.pet = Object.assign({}, modules.pet || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
