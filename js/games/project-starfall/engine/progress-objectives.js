(function initProjectStarfallEngineProgressObjectives(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const MAP_KILL_QUEST_GROWTH_PER_COMPLETION = 0.1;
  const MAP_KILL_QUEST_MAX_GROWTH_COMPLETIONS = 5;
  const MAP_KILL_QUEST_MAX_GOAL_MULTIPLIER = 1.5;

  function normalizeNonNegativeInteger(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.floor(number)));
  }

  function getObjectiveKey(objective, index) {
    return normalizeId(objective && objective.id) || `${normalizeId(objective && objective.type) || 'objective'}_${index}`;
  }

  function getObjectiveGoal(objective) {
    if (!objective) return 1;
    if (objective.type === 'level') return Math.max(1, Number(objective.level || objective.count || 1));
    return Math.max(1, Number(objective.count || 1));
  }

  function createProgressEntry(value) {
    const source = value && typeof value === 'object' ? value : {};
    const objectiveValues = source.objectiveValues && typeof source.objectiveValues === 'object' ? source.objectiveValues : {};
    return {
      objectiveValues: Object.keys(objectiveValues).reduce((values, key) => {
        values[key] = Math.max(0, Number(objectiveValues[key]) || 0);
        return values;
      }, {}),
      completedAt: Number(source.completedAt || 0)
    };
  }

  function createProgressTrialInstanceState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const trialId = normalizeId(source.trialId);
    return {
      active: !!source.active && !!trialId,
      trialId,
      returnMapId: normalizeId(source.returnMapId) || 'starfallCrossing',
      returnX: Number.isFinite(Number(source.returnX)) ? Number(source.returnX) : 1845,
      returnY: Number.isFinite(Number(source.returnY)) ? Number(source.returnY) : 360,
      startedAt: Number(source.startedAt || 0)
    };
  }

  function getProgressTrialInstanceStateCreator(options) {
    const settings = options || {};
    if (typeof settings.createTrialInstanceState === 'function') return settings.createTrialInstanceState;
    const modules = global.ProjectStarfallEngineModules || {};
    return modules.classTrials && modules.classTrials.createTrialInstanceState
      ? modules.classTrials.createTrialInstanceState
      : createProgressTrialInstanceState;
  }

  function createProgressState(value, options) {
    const createTrialInstanceState = getProgressTrialInstanceStateCreator(options);
    const source = value && typeof value === 'object' ? value : {};
    const questProgress = {};
    Object.entries(source.questProgress || {}).forEach(([id, entry]) => {
      questProgress[id] = createProgressEntry(entry);
    });
    const trialProgress = {};
    Object.entries(source.trialProgress || {}).forEach(([id, entry]) => {
      trialProgress[id] = createProgressEntry(entry);
    });
    const completedTrials = {};
    Object.entries(source.completedTrials || {}).forEach(([id, done]) => {
      if (done) completedTrials[id] = true;
    });
    const completedQuestIds = Array.isArray(source.completedQuestIds)
      ? source.completedQuestIds.map(normalizeId).filter(Boolean)
      : [];
    const claimedQuestIds = Array.isArray(source.claimedQuestIds)
      ? source.claimedQuestIds.map(normalizeId).filter(Boolean)
      : completedQuestIds;
    return {
      activeQuestId: normalizeId(source.activeQuestId),
      completedQuestIds: Array.from(new Set(completedQuestIds)),
      claimedQuestIds: Array.from(new Set(claimedQuestIds)),
      questProgress,
      activeTrialId: normalizeId(source.activeTrialId),
      trialProgress,
      completedTrials,
      trialInstance: createTrialInstanceState(source.trialInstance)
    };
  }

  function getProgressRootKey(kind) {
    return kind === 'trial'
      ? 'trialProgress'
      : kind === 'accomplishment'
        ? 'accomplishmentProgress'
        : 'questProgress';
  }

  function ensureProgressEntry(progress, kind, id) {
    const rootKey = getProgressRootKey(kind);
    progress[rootKey] = progress[rootKey] && typeof progress[rootKey] === 'object' ? progress[rootKey] : {};
    if (!progress[rootKey][id]) progress[rootKey][id] = createProgressEntry();
    return progress[rootKey][id];
  }

  function objectiveMatchesEvent(objective, type, payload, options) {
    if (!objective || objective.type !== type) return false;
    const settings = options || {};
    const data = payload || {};
    const getPotentialTierRank = settings.getPotentialTierRank || function getPotentialTierRankFallback() {
      return -1;
    };
    if (type === 'travel') return !objective.mapId || objective.mapId === data.mapId;
    if (type === 'interact') {
      if (objective.stationId && objective.stationId !== data.stationId) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'talk') {
      if (objective.npcId && objective.npcId !== data.npcId) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'defeat') {
      if (objective.enemyId && objective.enemyId !== data.enemyId) return false;
      if (objective.family && objective.family !== data.family) return false;
      if (objective.role && objective.role !== data.role) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'loot') {
      if (objective.kind && objective.kind !== data.kind) return false;
      if (objective.materialId && objective.materialId !== data.materialId) return false;
      if (objective.consumableId && objective.consumableId !== data.consumableId) return false;
      if (objective.cardId && objective.cardId !== data.cardId) return false;
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'equip') {
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      if (objective.slot && objective.slot !== data.slot) return false;
      return true;
    }
    if (type === 'rankSkill') {
      if (objective.skillId && objective.skillId !== data.skillId) return false;
      if (objective.owner && objective.owner !== data.owner) return false;
      if (objective.minRank && Number(data.rank || 0) < Number(objective.minRank || 0)) return false;
      return true;
    }
    if (type === 'useSkill') {
      if (objective.skillId && objective.skillId !== data.skillId) return false;
      if (objective.owner && objective.owner !== data.owner) return false;
      return true;
    }
    if (type === 'useConsumable') {
      if (objective.consumableId && objective.consumableId !== data.consumableId) return false;
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      return true;
    }
    if (type === 'partyFind') {
      if (objective.minMembers && Number(data.count || 0) < Number(objective.minMembers || 0)) return false;
      return true;
    }
    if (type === 'plinkoDrop') {
      if (objective.ballId && objective.ballId !== data.ballId) return false;
      if (objective.tier && objective.tier !== data.tier) return false;
      return true;
    }
    if (type === 'upgrade') {
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      if (objective.outcome && objective.outcome !== data.outcome) return false;
      if (objective.rolledOutcome && objective.rolledOutcome !== data.rolledOutcome) return false;
      if (objective.minUpgrade && Number(data.upgrade || 0) < Number(objective.minUpgrade || 0)) return false;
      return true;
    }
    if (type === 'itemPotential') {
      const lineCount = Array.isArray(data.lines) ? data.lines.length : Math.max(0, Number(data.lineCount || 0) || 0);
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      if (objective.tier && objective.tier !== data.tier) return false;
      if (objective.minTier && getPotentialTierRank(data.tier) < getPotentialTierRank(objective.minTier)) return false;
      if (objective.minLines && lineCount < Number(objective.minLines || 0)) return false;
      return true;
    }
    if (type === 'itemPotentialLineUpgrade') {
      if (objective.itemId && objective.itemId !== data.itemId) return false;
      if (Object.prototype.hasOwnProperty.call(objective, 'success') && !!objective.success !== !!data.success) return false;
      if (objective.minLineCount && Number(data.lineCount || 0) < Number(objective.minLineCount || 0)) return false;
      return true;
    }
    if (type === 'questClaim') {
      if (objective.questId && objective.questId !== data.questId) return false;
      if (objective.chainId && objective.chainId !== data.chainId) return false;
      return true;
    }
    if (type === 'dailyLoginClaim') return true;
    if (type === 'trialComplete') return !objective.advancedId || objective.advancedId === data.advancedId;
    if (type === 'dungeonComplete') {
      if (objective.dungeonId && objective.dungeonId !== data.dungeonId) return false;
      if (objective.bossId && objective.bossId !== data.bossId) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'defeatBoss') {
      if (objective.bossId && objective.bossId !== data.bossId) return false;
      if (objective.enemyId && objective.enemyId !== data.bossId) return false;
      if (objective.mapId && objective.mapId !== data.mapId) return false;
      return true;
    }
    if (type === 'advancedClass') {
      if (objective.advancedId && objective.advancedId !== data.advancedId) return false;
      if (objective.baseClass && objective.baseClass !== data.baseClass) return false;
      return true;
    }
    if (type === 'level') return true;
    return true;
  }

  function createObjectiveStatuses(entry, objectives) {
    const progress = entry || createProgressEntry();
    return (objectives || []).map((objective, index) => {
      const key = getObjectiveKey(objective, index);
      const goal = getObjectiveGoal(objective);
      const value = Math.min(goal, Math.max(0, Number(progress.objectiveValues[key]) || 0));
      return Object.assign({}, objective, {
        key,
        value,
        goal,
        complete: value >= goal
      });
    });
  }

  function applyProgressEventToEntry(progress, kind, entryData, type, payload, options) {
    if (!entryData) return false;
    const settings = options || {};
    const entry = ensureProgressEntry(progress, kind, entryData.id);
    let changed = false;
    (entryData.objectives || []).forEach((objective, index) => {
      if (!objectiveMatchesEvent(objective, type, payload, settings)) return;
      const key = getObjectiveKey(objective, index);
      const goal = getObjectiveGoal(objective);
      const before = Math.max(0, Number(entry.objectiveValues[key]) || 0);
      const next = type === 'level'
        ? Math.max(before, Number(payload && payload.level || 0))
        : Math.min(goal, before + Math.max(1, Number(payload && payload.count || 1)));
      if (next !== before) {
        entry.objectiveValues[key] = next;
        if (Array.isArray(settings.changes)) {
          settings.changes.push({
            objective,
            key,
            before,
            value: next,
            goal
          });
        }
        changed = true;
      }
    });
    return changed;
  }

  function addObjectiveTypes(types, entries) {
    (entries || []).forEach((entry) => {
      (entry && entry.objectives || []).forEach((objective) => {
        const type = normalizeId(objective && objective.type);
        if (type) types.add(type);
      });
    });
    return types;
  }

  function getProgressObjectiveData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createMapKillQuestState(value, options) {
    const data = getProgressObjectiveData(options);
    const source = value && typeof value === 'object' ? value : {};
    return (data.MAPS || []).reduce((quests, map) => {
      if (!map || map.safeZone) return quests;
      const entry = source[map.id] && typeof source[map.id] === 'object' ? source[map.id] : {};
      const progress = normalizeNonNegativeInteger(entry.progress);
      const completedAt = Number(entry.completedAt || 0);
      quests[map.id] = {
        active: !!entry.active || progress > 0 || completedAt > 0,
        progress,
        completions: normalizeNonNegativeInteger(entry.completions),
        completedAt,
        lastCompletedAt: Number(entry.lastCompletedAt || 0)
      };
      return quests;
    }, {});
  }

  function getMapKillQuestGoalProfile(baseGoal, completions) {
    const base = Math.max(1, normalizeNonNegativeInteger(baseGoal) || 1);
    const completionCount = normalizeNonNegativeInteger(completions);
    const masteryTier = Math.min(MAP_KILL_QUEST_MAX_GROWTH_COMPLETIONS, completionCount);
    const goalMultiplier = Math.min(
      MAP_KILL_QUEST_MAX_GOAL_MULTIPLIER,
      1 + masteryTier * MAP_KILL_QUEST_GROWTH_PER_COMPLETION
    );
    return {
      baseGoal: base,
      completions: completionCount,
      masteryTier,
      maxMasteryTier: MAP_KILL_QUEST_MAX_GROWTH_COMPLETIONS,
      goalMultiplier,
      maxGoalMultiplier: MAP_KILL_QUEST_MAX_GOAL_MULTIPLIER,
      goal: Math.max(1, Math.ceil(base * goalMultiplier - 1e-9)),
      capped: completionCount >= MAP_KILL_QUEST_MAX_GROWTH_COMPLETIONS
    };
  }

  function getMapKillQuestGoal(baseGoal, completions) {
    return getMapKillQuestGoalProfile(baseGoal, completions).goal;
  }

  function createQuestGuideState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const type = ['quest', 'trial', 'dungeon', 'mapKill', 'map'].includes(source.type) ? source.type : '';
    return {
      type,
      id: type ? normalizeId(source.id) : ''
    };
  }

  function createProgressObjectiveTypeSet(options) {
    const data = getProgressObjectiveData(options);
    const types = new Set();
    addObjectiveTypes(types, data.QUESTS || []);
    addObjectiveTypes(types, data.CLASS_TRIALS || []);
    addObjectiveTypes(types, data.SEASONS || []);
    addObjectiveTypes(types, data.ACCOMPLISHMENTS || []);
    return types;
  }

  function createObjectiveTypeEntryMap(entries) {
    const objectiveMap = new Map();
    (entries || []).forEach((entry) => {
      if (!entry) return;
      const objectiveTypes = new Set((entry.objectives || [])
        .map((objective) => normalizeId(objective && objective.type))
        .filter(Boolean));
      objectiveTypes.forEach((objectiveType) => {
        if (!objectiveMap.has(objectiveType)) objectiveMap.set(objectiveType, []);
        objectiveMap.get(objectiveType).push(entry);
      });
    });
    return objectiveMap;
  }

  function getObjectiveTypeEntries(objectiveMap, type) {
    const id = normalizeId(type);
    if (!id || !objectiveMap) return [];
    return objectiveMap.get(id) || [];
  }

  function createEventStepMap(steps) {
    const eventMap = new Map();
    (steps || []).forEach((step) => {
      const eventType = normalizeId(step && step.event);
      if (!eventType) return;
      if (!eventMap.has(eventType)) eventMap.set(eventType, []);
      eventMap.get(eventType).push(step);
    });
    return eventMap;
  }

  function getEventStepsByType(eventMap, type) {
    const id = normalizeId(type);
    if (!id || !eventMap) return [];
    return eventMap.get(id) || [];
  }

  const api = {
    getObjectiveKey,
    getObjectiveGoal,
    createProgressEntry,
    createProgressState,
    ensureProgressEntry,
    objectiveMatchesEvent,
    createObjectiveStatuses,
    applyProgressEventToEntry,
    addObjectiveTypes,
    createMapKillQuestState,
    getMapKillQuestGoalProfile,
    getMapKillQuestGoal,
    createQuestGuideState,
    createProgressObjectiveTypeSet,
    createObjectiveTypeEntryMap,
    getObjectiveTypeEntries,
    createEventStepMap,
    getEventStepsByType
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.progressObjectives = Object.assign({}, modules.progressObjectives || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
