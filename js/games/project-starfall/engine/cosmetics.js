(function initProjectStarfallEngineCosmetics(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    const source = Array.isArray(items) ? items : [];
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] && source[index].id === id) return source[index];
    }
    return null;
  };

  function getCosmeticData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getCosmeticSlot(cosmetic) {
    return normalizeId(cosmetic && cosmetic.slot) || 'aura';
  }

  function getCosmeticById(cosmeticId, options) {
    const data = getCosmeticData(options);
    return getById(data.COSMETICS || [], normalizeId(cosmeticId));
  }

  function getCosmeticUnlockedIds(cosmetics) {
    return cosmetics && Array.isArray(cosmetics.unlockedIds) ? cosmetics.unlockedIds : [];
  }

  function getCosmeticEquippedBySlot(cosmetics) {
    return cosmetics && cosmetics.equippedBySlot && typeof cosmetics.equippedBySlot === 'object' ? cosmetics.equippedBySlot : {};
  }

  function createCosmeticState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const unlockedIds = Array.isArray(source.unlockedIds)
      ? source.unlockedIds.map(normalizeId).filter(Boolean)
      : [];
    const unlocked = new Set(unlockedIds);
    const equippedBySlot = {};
    Object.entries(source.equippedBySlot && typeof source.equippedBySlot === 'object' ? source.equippedBySlot : {}).forEach(([slot, cosmeticId]) => {
      const slotId = normalizeId(slot);
      const equippedId = normalizeId(cosmeticId);
      const cosmetic = equippedId ? getCosmeticById(equippedId, options) : null;
      if (slotId && cosmetic && unlocked.has(cosmetic.id) && getCosmeticSlot(cosmetic) === slotId) equippedBySlot[slotId] = cosmetic.id;
    });
    const equippedId = normalizeId(source.equippedId);
    const legacyCosmetic = equippedId && unlocked.has(equippedId) ? getCosmeticById(equippedId, options) : null;
    if (legacyCosmetic) {
      const legacySlot = getCosmeticSlot(legacyCosmetic);
      if (!equippedBySlot[legacySlot]) equippedBySlot[legacySlot] = legacyCosmetic.id;
    }
    return {
      unlockedIds: Array.from(unlocked),
      equippedId: equippedBySlot.aura || '',
      equippedBySlot
    };
  }

  function createCosmeticUnlockPlan(cosmetic, cosmetics) {
    if (!cosmetic) return null;
    const unlockedIds = getCosmeticUnlockedIds(cosmetics);
    const equippedBySlot = getCosmeticEquippedBySlot(cosmetics);
    const slot = getCosmeticSlot(cosmetic);
    const shouldEquipSlot = !equippedBySlot[slot];
    return {
      cosmeticId: cosmetic.id,
      slot,
      shouldUnlock: !unlockedIds.includes(cosmetic.id),
      shouldEquipSlot,
      equippedId: slot === 'aura' && shouldEquipSlot ? cosmetic.id : equippedBySlot.aura || '',
      toast: `${cosmetic.name} unlocked.`
    };
  }

  function createCosmeticEquipPlan(cosmetic, cosmetics) {
    if (!cosmetic || !getCosmeticUnlockedIds(cosmetics).includes(cosmetic.id)) {
      return { ok: false, toast: 'Cosmetic is not unlocked.' };
    }
    const equippedBySlot = getCosmeticEquippedBySlot(cosmetics);
    const slot = getCosmeticSlot(cosmetic);
    return {
      ok: true,
      cosmeticId: cosmetic.id,
      slot,
      equippedId: slot === 'aura' ? cosmetic.id : equippedBySlot.aura || '',
      toast: `${cosmetic.name} equipped.`
    };
  }

  function createCosmeticPurchasePlan(cosmetic, cosmetics, currency) {
    if (!cosmetic) return { ok: false, missing: true };
    if (cosmetic.seasonReward) return { ok: false, toast: `${cosmetic.name} is a season reward.` };
    if (cosmetic.cashShopOnly) return { ok: false, toast: `${cosmetic.name} is available in the Cash Shop.` };
    if (getCosmeticUnlockedIds(cosmetics).includes(cosmetic.id)) {
      return { ok: false, alreadyOwned: true, equipId: cosmetic.id };
    }
    const cost = Math.max(0, Number(cosmetic.cost || 0));
    if (currency < cost) return { ok: false, cost, toast: 'Not enough coins.' };
    const equippedBySlot = getCosmeticEquippedBySlot(cosmetics);
    const slot = getCosmeticSlot(cosmetic);
    return {
      ok: true,
      cosmeticId: cosmetic.id,
      slot,
      cost,
      equippedId: slot === 'aura' ? cosmetic.id : equippedBySlot.aura || '',
      toast: `${cosmetic.name} purchased and equipped.`
    };
  }

  function getCosmeticSnapshotCacheKey(cosmetics, options) {
    const data = getCosmeticData(options);
    const source = cosmetics && typeof cosmetics === 'object' ? cosmetics : {};
    const unlocked = Array.isArray(source.unlockedIds)
      ? source.unlockedIds.map(normalizeId).filter(Boolean).sort()
      : [];
    const equippedBySlot = source.equippedBySlot && typeof source.equippedBySlot === 'object' ? source.equippedBySlot : {};
    const equipped = Object.keys(equippedBySlot).sort().map((slot) => `${normalizeId(slot)}:${normalizeId(equippedBySlot[slot])}`);
    return [
      data.COSMETICS || [],
      (data.COSMETICS || []).length,
      unlocked.join(','),
      equipped.join(','),
      normalizeId(source.equippedId)
    ];
  }

  function createCosmeticSnapshot(cosmetics, options) {
    const data = getCosmeticData(options);
    const source = cosmetics && typeof cosmetics === 'object' ? cosmetics : {};
    const unlocked = new Set(source.unlockedIds || []);
    const equippedBySlot = source.equippedBySlot && typeof source.equippedBySlot === 'object' ? source.equippedBySlot : {};
    return {
      equippedId: equippedBySlot.aura || source.equippedId || '',
      equippedBySlot: Object.assign({}, equippedBySlot),
      cosmetics: (data.COSMETICS || []).map((cosmetic) => Object.assign({}, cosmetic, {
        unlocked: unlocked.has(cosmetic.id),
        equipped: equippedBySlot[getCosmeticSlot(cosmetic)] === cosmetic.id
      }))
    };
  }

  function getEquippedCosmetic(slot, cosmetics, options) {
    const slotId = normalizeId(slot) || 'aura';
    const source = cosmetics && typeof cosmetics === 'object' ? cosmetics : {};
    const equippedBySlot = source.equippedBySlot && typeof source.equippedBySlot === 'object' ? source.equippedBySlot : {};
    const cosmeticId = normalizeId(equippedBySlot[slotId]);
    return cosmeticId ? getCosmeticById(cosmeticId, options) : null;
  }

  const api = {
    getCosmeticSlot,
    getCosmeticById,
    createCosmeticState,
    createCosmeticUnlockPlan,
    createCosmeticEquipPlan,
    createCosmeticPurchasePlan,
    getCosmeticSnapshotCacheKey,
    createCosmeticSnapshot,
    getEquippedCosmetic
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.cosmetics = Object.assign({}, modules.cosmetics || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
