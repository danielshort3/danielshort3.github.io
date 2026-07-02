(function initProjectStarfallEngineClassMastery(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getClassMasteryData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createClassMasteryState(value, options) {
    const data = getClassMasteryData(options);
    const source = value && typeof value === 'object' ? value : {};
    const trackIds = new Set((data.CLASS_MASTERY_TRACKS || []).map((track) => track.classId));
    const xpByClassId = {};
    const levelByClassId = {};
    Object.entries(source.xpByClassId || {}).forEach(([classId, amount]) => {
      const key = normalizeId(classId);
      if (trackIds.has(key)) xpByClassId[key] = Math.max(0, Math.floor(Number(amount) || 0));
    });
    Object.entries(source.levelByClassId || {}).forEach(([classId, level]) => {
      const key = normalizeId(classId);
      if (trackIds.has(key)) levelByClassId[key] = Math.max(1, Math.floor(Number(level) || 1));
    });
    (data.CLASS_MASTERY_TRACKS || []).forEach((track) => {
      if (!track || !track.classId) return;
      if (!Object.prototype.hasOwnProperty.call(xpByClassId, track.classId)) xpByClassId[track.classId] = 0;
      if (!Object.prototype.hasOwnProperty.call(levelByClassId, track.classId)) levelByClassId[track.classId] = 1;
    });
    return { xpByClassId, levelByClassId };
  }

  function getClassMasteryTrack(classId, options) {
    const data = getClassMasteryData(options);
    const id = normalizeId(classId);
    return (data.CLASS_MASTERY_TRACKS || []).find((track) => track && track.classId === id) || null;
  }

  function getClassMasteryLevelForXp(track, xp) {
    const amount = Math.max(0, Number(xp || 0));
    const milestones = track && track.milestones || [];
    let level = 1;
    for (let index = 0; index < milestones.length; index += 1) {
      const milestone = milestones[index];
      if (amount >= Number(milestone && milestone.xp || 0)) {
        level = Math.max(level, Number(milestone && milestone.level || 1));
      }
    }
    return level;
  }

  function addRoundedStats(target, source) {
    Object.entries(source || {}).forEach(([key, value]) => {
      const amount = Math.round(Number(value || 0) || 0);
      if (amount) target[key] = (target[key] || 0) + amount;
    });
    return target;
  }

  function createClassMasteryXpPlan(amount, classId, player, mastery, options) {
    const activePlayer = player || {};
    const state = createClassMasteryState(mastery, options);
    const id = normalizeId(classId || activePlayer.advancedClassId || activePlayer.classId);
    const track = getClassMasteryTrack(id, options);
    if (!track) return { ok: false, reason: 'missing' };
    const before = Math.max(1, Number(state.levelByClassId[id] || 1));
    const nextXp = Math.max(0, Number(state.xpByClassId[id] || 0)) + Math.max(0, Number(amount || 0));
    const nextLevel = getClassMasteryLevelForXp(track, nextXp);
    return {
      ok: true,
      classId: id,
      nextXp,
      beforeLevel: before,
      nextLevel,
      toast: nextLevel > before ? `${track.name} level ${nextLevel}.` : ''
    };
  }

  function createClassMasteryBonuses(player, mastery, options) {
    const activePlayer = player || {};
    const state = mastery && typeof mastery === 'object' ? mastery : createClassMasteryState(null, options);
    const ids = [activePlayer.classId, activePlayer.advancedClassId].map(normalizeId).filter(Boolean);
    return ids.reduce((stats, classId) => {
      const track = getClassMasteryTrack(classId, options);
      const level = Math.max(1, Number(state.levelByClassId && state.levelByClassId[classId] || 1));
      (track && track.milestones || []).forEach((milestone) => {
        if (level >= Number(milestone.level || 1)) addRoundedStats(stats, milestone.statBonuses || {});
      });
      return stats;
    }, {});
  }

  function getClassMasterySnapshotCacheKey(player, mastery, revisionKey, options) {
    const data = getClassMasteryData(options);
    const activePlayer = player || {};
    const state = createClassMasteryState(mastery, options);
    const xpByClassId = state.xpByClassId || {};
    const levelByClassId = state.levelByClassId || {};
    const ids = Array.from(new Set((data.CLASS_MASTERY_TRACKS || [])
      .map((track) => normalizeId(track && track.classId))
      .filter(Boolean))).sort();
    return [
      revisionKey,
      normalizeId(activePlayer.classId),
      normalizeId(activePlayer.advancedClassId),
      ids.map((id) => `${id}:${Math.max(0, Number(xpByClassId[id] || 0) || 0)}:${Math.max(1, Number(levelByClassId[id] || 1) || 1)}`).join(',')
    ].join('|');
  }

  function createClassMasterySnapshot(player, mastery, options) {
    const data = getClassMasteryData(options);
    const activePlayer = player || {};
    const state = createClassMasteryState(mastery, options);
    const currentIds = new Set([activePlayer.classId, activePlayer.advancedClassId].map(normalizeId).filter(Boolean));
    return {
      tracks: (data.CLASS_MASTERY_TRACKS || []).map((track) => {
        const xp = Math.max(0, Number(state.xpByClassId[track.classId] || 0));
        const level = getClassMasteryLevelForXp(track, xp);
        let next = null;
        const milestones = track.milestones || [];
        for (let index = 0; index < milestones.length; index += 1) {
          const milestone = milestones[index];
          if (Number(milestone && milestone.level || 1) > level) {
            next = milestone;
            break;
          }
        }
        const bonuses = [];
        for (let index = 0; index < milestones.length; index += 1) {
          const milestone = milestones[index];
          if (level >= Number(milestone && milestone.level || 1)) bonuses.push(milestone.statBonuses || {});
        }
        return Object.assign({}, track, {
          current: currentIds.has(track.classId),
          xp,
          level,
          nextXp: next ? Number(next.xp || 0) : 0,
          bonuses
        });
      })
    };
  }

  const api = {
    createClassMasteryState,
    getClassMasteryTrack,
    getClassMasteryLevelForXp,
    createClassMasteryXpPlan,
    createClassMasteryBonuses,
    getClassMasterySnapshotCacheKey,
    createClassMasterySnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.classMastery = Object.assign({}, modules.classMastery || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
