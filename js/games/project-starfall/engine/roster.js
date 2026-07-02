(function initProjectStarfallEngineRoster(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getRosterData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getRosterTraitSlotCount(options) {
    const data = getRosterData(options);
    return Math.max(1, Number(data.ROSTER_TRAIT_SLOTS || 2));
  }

  function createRosterState(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const activeTraitIds = Array.isArray(source.activeTraitIds)
      ? source.activeTraitIds.map(normalizeId).filter(Boolean)
      : [];
    const unlockedTraitIds = Array.isArray(source.unlockedTraitIds)
      ? source.unlockedTraitIds.map(normalizeId).filter(Boolean)
      : [];
    return {
      activeTraitIds: Array.from(new Set(activeTraitIds)).slice(0, getRosterTraitSlotCount(options)),
      unlockedTraitIds: Array.from(new Set(unlockedTraitIds))
    };
  }

  function createRosterUnlockSyncState(roster, player, dungeons, options) {
    const data = getRosterData(options);
    const state = createRosterState(roster, options);
    const activePlayer = player || {};
    const dungeonState = dungeons && typeof dungeons === 'object' ? dungeons : {};
    const completedDungeonIds = Array.isArray(dungeonState.completedDungeonIds) ? dungeonState.completedDungeonIds : [];
    const unlocked = new Set(state.unlockedTraitIds);
    (data.ROSTER_TRAITS || []).forEach((trait) => {
      if (trait.sourceAdvancedId && trait.sourceAdvancedId === activePlayer.advancedClassId) unlocked.add(trait.id);
      if (trait.sourceDungeonId && completedDungeonIds.includes(trait.sourceDungeonId)) unlocked.add(trait.id);
    });
    state.unlockedTraitIds = Array.from(unlocked);
    state.activeTraitIds = state.activeTraitIds
      .filter((traitId) => unlocked.has(traitId))
      .slice(0, getRosterTraitSlotCount(options));
    return state;
  }

  function createRosterTraitBonuses(roster, options) {
    const data = getRosterData(options);
    const state = roster && typeof roster === 'object' ? roster : createRosterState(null, options);
    const unlocked = new Set(state.unlockedTraitIds || []);
    return (state.activeTraitIds || []).reduce((stats, traitId) => {
      if (!unlocked.has(traitId)) return stats;
      const trait = getById(data.ROSTER_TRAITS || [], traitId);
      Object.entries(trait && trait.statBonuses || {}).forEach(([key, value]) => {
        stats[key] = Number(stats[key] || 0) + Number(value || 0);
      });
      return stats;
    }, {});
  }

  function createRosterSnapshot(roster, options) {
    const data = getRosterData(options);
    const state = roster && typeof roster === 'object' ? roster : createRosterState(null, options);
    const unlocked = new Set(state.unlockedTraitIds || []);
    const active = new Set(state.activeTraitIds || []);
    return {
      slots: getRosterTraitSlotCount(options),
      activeTraitIds: (state.activeTraitIds || []).slice(),
      unlockedTraitIds: (state.unlockedTraitIds || []).slice(),
      traits: (data.ROSTER_TRAITS || []).map((trait) => Object.assign({}, trait, {
        unlocked: unlocked.has(trait.id),
        active: active.has(trait.id)
      }))
    };
  }

  function createRosterTraitTogglePlan(trait, roster, options) {
    if (!trait) return { ok: false, reason: 'missing' };
    const state = roster && typeof roster === 'object' ? roster : createRosterState(null, options);
    const unlocked = new Set(state.unlockedTraitIds || []);
    if (!unlocked.has(trait.id)) {
      return {
        ok: false,
        reason: 'locked',
        toast: `${trait.name} is not unlocked yet.`
      };
    }
    const active = state.activeTraitIds || [];
    if (active.includes(trait.id)) {
      return {
        ok: true,
        activeTraitIds: active.filter((id) => id !== trait.id),
        toast: `${trait.name} removed from roster traits.`
      };
    }
    return {
      ok: true,
      activeTraitIds: active.concat(trait.id).slice(-getRosterTraitSlotCount(options)),
      toast: `${trait.name} activated.`
    };
  }

  function createRosterSynergySnapshot(roster, options) {
    const data = getRosterData(options);
    const state = roster && typeof roster === 'object' ? roster : createRosterState(null, options);
    const unlocked = new Set(state.unlockedTraitIds || []);
    const active = new Set(state.activeTraitIds || []);
    return {
      synergies: (data.ROSTER_SYNERGIES || []).map((synergy) => {
        const required = Array.isArray(synergy.requiredTraitIds) ? synergy.requiredTraitIds : [];
        const unlockedAll = required.every((traitId) => unlocked.has(traitId));
        const activeAll = required.every((traitId) => active.has(traitId));
        return Object.assign({}, synergy, { unlocked: unlockedAll, active: activeAll });
      })
    };
  }

  function addRoundedStats(target, source) {
    Object.entries(source || {}).forEach(([key, value]) => {
      const amount = Math.round(Number(value || 0) || 0);
      if (amount) target[key] = (target[key] || 0) + amount;
    });
    return target;
  }

  function createRosterSynergyBonuses(snapshot) {
    return (snapshot && snapshot.synergies || []).reduce((stats, synergy) => {
      if (synergy.active) addRoundedStats(stats, synergy.statBonuses || {});
      return stats;
    }, {});
  }

  const api = {
    createRosterState,
    createRosterUnlockSyncState,
    createRosterTraitBonuses,
    createRosterSnapshot,
    createRosterTraitTogglePlan,
    createRosterSynergySnapshot,
    createRosterSynergyBonuses
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.roster = Object.assign({}, modules.roster || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
