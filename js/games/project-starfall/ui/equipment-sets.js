(function initProjectStarfallUiEquipmentSets(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiLookups = (typeof require === 'function' ? require('./lookups.js') : null) || UiModules.lookups || {};

  function getEquipmentSet(setId, options) {
    const settings = options || {};
    if (settings.getEquipmentSet) return settings.getEquipmentSet(setId);
    if (UiLookups.getEquipmentSet) return UiLookups.getEquipmentSet(setId, settings);
    const id = String(setId || '');
    const lookup = settings.equipmentSetsById || {};
    return lookup[id] || null;
  }

  function getEquippedSetPieceCount(snapshot, setId, projectedItem) {
    const id = String(setId || '');
    if (!id || !snapshot || !snapshot.state) return 0;
    let count = 0;
    Object.entries(snapshot.state.equipment || {}).forEach(([slot, item]) => {
      if (projectedItem && projectedItem.slot === slot) {
        if (projectedItem.setId === id) count += 1;
        return;
      }
      if (item && item.setId === id) count += 1;
    });
    if (projectedItem && projectedItem.setId === id && !snapshot.state.equipment[projectedItem.slot]) count += 1;
    return count;
  }

  function getSetPieceTotal(setId, options) {
    const id = String(setId || '');
    if (!id) return 0;
    const data = options && options.data || Data;
    return (data.BOSS_EQUIPMENT_ITEMS || []).filter((item) => item && item.setId === id).length;
  }

  function getSetBonusDetails(snapshot, item, options) {
    const settings = options || {};
    const compactStats = settings.compactStats || function compactStatsFallback() {
      return '';
    };
    const set = getEquipmentSet(item && item.setId, settings);
    if (!set) return null;
    const currentCount = getEquippedSetPieceCount(snapshot, set.id);
    const projectedCount = getEquippedSetPieceCount(snapshot, set.id, item);
    const totalPieces = getSetPieceTotal(set.id, settings) || currentCount || projectedCount || 0;
    const maxBonusPieces = Math.max(0, ...(set.pieceBonuses || []).map((bonus) => Number(bonus.pieces || 0)));
    return {
      set,
      currentCount,
      projectedCount,
      totalPieces,
      maxBonusPieces,
      rows: (set.pieceBonuses || []).map((bonus) => {
        const pieces = Number(bonus.pieces || 0);
        const status = currentCount >= pieces ? 'active' : projectedCount >= pieces ? 'projected' : 'next';
        return {
          pieces,
          stats: bonus.stats || {},
          status,
          statusLabel: status === 'active' ? 'Active' : status === 'projected' ? 'If equipped' : 'Next',
          summary: compactStats(bonus.stats || {}, 3) || 'No bonus'
        };
      })
    };
  }

  function formatSetBonusDetailsLabel(details) {
    if (!details || !details.set) return '';
    const total = details.totalPieces || '?';
    const rows = (details.rows || [])
      .map((row) => `${row.pieces}pc ${row.statusLabel}: ${row.summary}`)
      .join(' | ');
    return `${details.set.name} set. Currently equipped: ${details.currentCount}/${total}. If equipped: ${details.projectedCount}/${total}. ${rows}`;
  }

  function createEquipmentSetUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      data: settings.data || Data,
      equipmentSetsById: settings.equipmentSetsById || {},
      getEquipmentSet: settings.getEquipmentSet,
      compactStats: settings.compactStats
    });
    const mergeOptions = (override) => Object.assign({}, helperOptions, override || {});
    return Object.freeze({
      getEquipmentSet: (setId, override) => getEquipmentSet(setId, mergeOptions(override)),
      getEquippedSetPieceCount,
      getSetPieceTotal: (setId, override) => getSetPieceTotal(setId, mergeOptions(override)),
      getSetBonusDetails: (snapshot, item, override) => getSetBonusDetails(snapshot, item, mergeOptions(override)),
      formatSetBonusDetailsLabel
    });
  }

  const api = {
    getEquipmentSet,
    getEquippedSetPieceCount,
    getSetPieceTotal,
    getSetBonusDetails,
    formatSetBonusDetailsLabel,
    createEquipmentSetUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.equipmentSets = Object.assign({}, modules.equipmentSets || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
