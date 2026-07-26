(function initProjectStarfallEngineWeeklyRoutes(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreTime = (typeof require === 'function' ? require('../core/time.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const WEEKLY_ROUTE_DAY_MS = Number(CoreTime.WEEKLY_ROUTE_DAY_MS || 24 * 60 * 60 * 1000);
  const WEEKLY_ROUTE_WEEK_MS = Number(CoreTime.WEEKLY_ROUTE_WEEK_MS || 7 * WEEKLY_ROUTE_DAY_MS);
  const getWeeklyRouteWeekStartMs = CoreTime.getWeeklyRouteWeekStartMs || function getWeeklyRouteWeekStartMsFallback(nowMs) {
    const timestamp = Number.isFinite(Number(nowMs)) ? Number(nowMs) : Date.now();
    const date = new Date(timestamp);
    const utcDayStart = Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
    const daysSinceMonday = (date.getUTCDay() + 6) % 7;
    return utcDayStart - daysSinceMonday * WEEKLY_ROUTE_DAY_MS;
  };
  const getWeeklyRouteWeekKey = CoreTime.getWeeklyRouteWeekKey || function getWeeklyRouteWeekKeyFallback(nowMs) {
    return new Date(getWeeklyRouteWeekStartMs(nowMs)).toISOString().slice(0, 10);
  };
  const getWeeklyRouteResetAt = CoreTime.getWeeklyRouteResetAt || function getWeeklyRouteResetAtFallback(nowMs) {
    return getWeeklyRouteWeekStartMs(nowMs) + WEEKLY_ROUTE_WEEK_MS;
  };

  const WEEKLY_ROUTE_KIND_EVENT_TYPES = Object.freeze({
    mapHunt: 'mapHuntClaim',
    mapMechanic: 'mapMechanicComplete',
    dungeon: 'dungeonComplete'
  });

  const DEFAULT_WEEKLY_STAR_ROUTES_CONFIG = Object.freeze({
    id: 'weekly_star_routes',
    unlockSeasonId: 'beta_foundations',
    completionGoal: 3,
    eventKeyLimit: 32,
    slots: Object.freeze([
      Object.freeze({ id: 'field_a', kind: 'mapHunt', eventType: 'mapHuntClaim', fallbackKinds: Object.freeze([]) }),
      Object.freeze({ id: 'field_b', kind: 'mapHunt', eventType: 'mapHuntClaim', fallbackKinds: Object.freeze([]) }),
      Object.freeze({ id: 'challenge', kind: 'mapMechanic', eventType: 'mapMechanicComplete', fallbackKinds: Object.freeze(['mapHunt']) }),
      Object.freeze({ id: 'dungeon', kind: 'dungeon', eventType: 'dungeonComplete', fallbackKinds: Object.freeze(['mapHunt']) })
    ]),
    reward: Object.freeze({
      currency: 400,
      starTokens: 75
    })
  });

  function getWeeklyRouteConfig(options) {
    const settings = options || {};
    const data = settings.data || global.ProjectStarfallData || {};
    return settings.config || data.WEEKLY_STAR_ROUTES_CONFIG || DEFAULT_WEEKLY_STAR_ROUTES_CONFIG;
  }

  function getWeeklyRouteSlots(config) {
    const source = config && Array.isArray(config.slots) && config.slots.length
      ? config.slots
      : DEFAULT_WEEKLY_STAR_ROUTES_CONFIG.slots;
    const seen = new Set();
    return source.reduce((slots, value) => {
      const slot = value && typeof value === 'object' ? value : {};
      const id = normalizeId(slot.id);
      const kind = normalizeId(slot.kind);
      if (!id || seen.has(id) || !WEEKLY_ROUTE_KIND_EVENT_TYPES[kind]) return slots;
      seen.add(id);
      const fallbackKinds = Array.isArray(slot.fallbackKinds)
        ? slot.fallbackKinds.map(normalizeId).filter((entry) => WEEKLY_ROUTE_KIND_EVENT_TYPES[entry])
        : [];
      slots.push({
        id,
        kind,
        eventType: normalizeId(slot.eventType) || WEEKLY_ROUTE_KIND_EVENT_TYPES[kind],
        fallbackKinds: Array.from(new Set(fallbackKinds))
      });
      return slots;
    }, []);
  }

  function normalizeWeekStartMs(value) {
    const timestamp = Math.max(0, Number(value || 0) || 0);
    return timestamp ? getWeeklyRouteWeekStartMs(timestamp) : 0;
  }

  function normalizeWeeklyRouteAssignment(value, slotsById) {
    const source = value && typeof value === 'object' ? value : {};
    const slotId = normalizeId(source.slotId);
    const slot = slotsById.get(slotId);
    const kind = normalizeId(source.kind);
    const targetId = normalizeId(source.targetId);
    if (!slot || !targetId || !WEEKLY_ROUTE_KIND_EVENT_TYPES[kind]) return null;
    if (kind !== slot.kind && !slot.fallbackKinds.includes(kind)) return null;
    const mapId = normalizeId(source.mapId);
    return {
      id: `${slotId}:${kind}:${targetId}`,
      slotId,
      kind,
      type: WEEKLY_ROUTE_KIND_EVENT_TYPES[kind],
      targetId,
      mapId,
      label: String(source.label || source.name || '').trim(),
      summary: String(source.summary || '').trim(),
      guideType: normalizeId(source.guideType),
      guideId: normalizeId(source.guideId),
      recommendedMin: Math.max(0, Math.floor(Number(source.recommendedMin || 0) || 0)),
      recommendedMax: Math.max(0, Math.floor(Number(source.recommendedMax || 0) || 0)),
      levelRequirement: Math.max(0, Math.floor(Number(source.levelRequirement || 0) || 0)),
      goal: Math.max(1, Math.floor(Number(source.goal || 1) || 1))
    };
  }

  function createWeeklyRouteState(value, options) {
    const config = getWeeklyRouteConfig(options);
    const slots = getWeeklyRouteSlots(config);
    const slotsById = new Map(slots.map((slot) => [slot.id, slot]));
    const source = value && typeof value === 'object' ? value : {};
    const seenSlots = new Set();
    const assignments = (Array.isArray(source.assignments) ? source.assignments : [])
      .map((assignment) => normalizeWeeklyRouteAssignment(assignment, slotsById))
      .filter((assignment) => {
        if (!assignment || seenSlots.has(assignment.slotId)) return false;
        seenSlots.add(assignment.slotId);
        return true;
      })
      .sort((left, right) => {
        return slots.findIndex((slot) => slot.id === left.slotId) - slots.findIndex((slot) => slot.id === right.slotId);
      });
    const assignmentsById = new Map(assignments.map((assignment) => [assignment.id, assignment]));
    const objectiveValues = Object.entries(source.objectiveValues && typeof source.objectiveValues === 'object'
      ? source.objectiveValues
      : {}).reduce((values, entry) => {
      const id = normalizeId(entry[0]);
      const assignment = assignmentsById.get(id);
      if (assignment) {
        values[id] = Math.min(
          assignment.goal,
          Math.max(0, Math.floor(Number(entry[1] || 0) || 0))
        );
      }
      return values;
    }, {});
    const eventKeyLimit = Math.max(1, Math.floor(Number(config.eventKeyLimit || 32) || 32));
    const creditedEventKeys = Array.from(new Set((Array.isArray(source.creditedEventKeys) ? source.creditedEventKeys : [])
      .map(normalizeId)
      .filter(Boolean)))
      .slice(-eventKeyLimit);
    return {
      version: 1,
      unlocked: !!source.unlocked,
      weekStartMs: normalizeWeekStartMs(source.weekStartMs),
      assignments,
      objectiveValues,
      creditedEventKeys,
      rewardGrantedWeekStartMs: normalizeWeekStartMs(source.rewardGrantedWeekStartMs),
      completedWeekCount: Math.max(0, Math.floor(Number(source.completedWeekCount || 0) || 0))
    };
  }

  function isWeeklyRouteUnlocked(seasonState, options) {
    const settings = options || {};
    const config = getWeeklyRouteConfig(settings);
    const source = seasonState && typeof seasonState === 'object' ? seasonState : {};
    const weekly = source.weeklyRoutes && typeof source.weeklyRoutes === 'object'
      ? source.weeklyRoutes
      : source;
    if (weekly.unlocked || settings.unlocked === true || settings.unlockSeasonComplete === true) return true;
    const unlockSeasonId = normalizeId(config.unlockSeasonId) || 'beta_foundations';
    if ((Array.isArray(source.claimedRewardIds) ? source.claimedRewardIds : []).map(normalizeId).includes(unlockSeasonId)) {
      return true;
    }
    const snapshot = settings.seasonSnapshot && typeof settings.seasonSnapshot === 'object'
      ? settings.seasonSnapshot
      : null;
    const activeSeasonId = normalizeId(snapshot && snapshot.activeSeason && snapshot.activeSeason.id);
    return !!(snapshot && snapshot.complete && (!activeSeasonId || activeSeasonId === unlockSeasonId));
  }

  function stableWeeklyRouteHash(value) {
    const source = String(value || '');
    let hash = 2166136261;
    for (let index = 0; index < source.length; index += 1) {
      hash ^= source.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function normalizeWeeklyRouteCandidate(value, kind) {
    const source = value && typeof value === 'object' ? value : {};
    const stringValue = typeof value === 'string' ? normalizeId(value) : '';
    let targetId = stringValue || normalizeId(source.targetId || source.id);
    if (kind === 'mapHunt') targetId = normalizeId(source.targetId || source.mapId || source.id || stringValue);
    if (kind === 'mapMechanic') targetId = normalizeId(source.targetId || source.mapMechanicId || source.id || stringValue);
    if (kind === 'dungeon') targetId = normalizeId(source.targetId || source.dungeonId || source.id || stringValue);
    if (!targetId) return null;
    return {
      kind,
      targetId,
      mapId: normalizeId(source.mapId || (kind === 'mapHunt' ? targetId : '')),
      label: String(source.label || source.name || '').trim(),
      summary: String(source.summary || '').trim(),
      guideType: normalizeId(source.guideType),
      guideId: normalizeId(source.guideId),
      recommendedMin: Math.max(0, Math.floor(Number(source.recommendedMin || 0) || 0)),
      recommendedMax: Math.max(0, Math.floor(Number(source.recommendedMax || 0) || 0)),
      levelRequirement: Math.max(0, Math.floor(Number(source.levelRequirement || 0) || 0))
    };
  }

  function getWeeklyRouteCandidatePool(candidates, kind) {
    const source = candidates && typeof candidates === 'object' ? candidates : {};
    const keys = kind === 'mapHunt'
      ? ['field', 'mapHunts']
      : kind === 'mapMechanic'
        ? ['mechanic', 'mapMechanics']
        : ['dungeon', 'dungeons'];
    const values = keys.reduce((entries, key) => {
      return entries.concat(Array.isArray(source[key]) ? source[key] : []);
    }, []);
    const seen = new Set();
    return values
      .map((candidate) => normalizeWeeklyRouteCandidate(candidate, kind))
      .filter((candidate) => {
        const keyValue = candidate ? `${kind}:${candidate.targetId}` : '';
        if (!candidate || seen.has(keyValue)) return false;
        seen.add(keyValue);
        return true;
      })
      .sort((left, right) => {
        return left.targetId.localeCompare(right.targetId) || left.mapId.localeCompare(right.mapId);
      });
  }

  function selectWeeklyRouteCandidate(pool, weekKey, slotId, usedTargetKeys, usedMapIds) {
    const available = pool.filter((candidate) => !usedTargetKeys.has(`${candidate.kind}:${candidate.targetId}`));
    if (!available.length) return null;
    const unusedMaps = available.filter((candidate) => !candidate.mapId || !usedMapIds.has(candidate.mapId));
    const selectionPool = unusedMaps.length ? unusedMaps : available;
    const index = stableWeeklyRouteHash(`${weekKey}:${slotId}:${selectionPool[0].kind}`) % selectionPool.length;
    return selectionPool[index];
  }

  function createWeeklyRouteAssignments(candidates, options) {
    const receivedOptionsOnly = !options && candidates && typeof candidates === 'object' && candidates.candidates;
    const settings = receivedOptionsOnly ? candidates : options || {};
    const candidateSource = receivedOptionsOnly ? candidates.candidates : candidates || settings.candidates || {};
    const config = getWeeklyRouteConfig(settings);
    const slots = getWeeklyRouteSlots(config);
    const weekStartMs = normalizeWeekStartMs(settings.weekStartMs || settings.nowMs || Date.now());
    const weekKey = normalizeId(settings.weekKey) || getWeeklyRouteWeekKey(weekStartMs);
    const existingBySlot = new Map();
    const slotsById = new Map(slots.map((slot) => [slot.id, slot]));
    (Array.isArray(settings.existingAssignments) ? settings.existingAssignments : []).forEach((value) => {
      const assignment = normalizeWeeklyRouteAssignment(value, slotsById);
      if (assignment && !existingBySlot.has(assignment.slotId)) existingBySlot.set(assignment.slotId, assignment);
    });
    const pools = {
      mapHunt: getWeeklyRouteCandidatePool(candidateSource, 'mapHunt'),
      mapMechanic: getWeeklyRouteCandidatePool(candidateSource, 'mapMechanic'),
      dungeon: getWeeklyRouteCandidatePool(candidateSource, 'dungeon')
    };
    const usedTargetKeys = new Set();
    const usedMapIds = new Set();
    const assignments = [];
    slots.forEach((slot) => {
      const existing = existingBySlot.get(slot.id);
      if (existing) {
        assignments.push(existing);
        usedTargetKeys.add(`${existing.kind}:${existing.targetId}`);
        if (existing.mapId) usedMapIds.add(existing.mapId);
        return;
      }
      const kinds = [slot.kind].concat(slot.fallbackKinds);
      let selected = null;
      for (let index = 0; index < kinds.length && !selected; index += 1) {
        selected = selectWeeklyRouteCandidate(pools[kinds[index]] || [], weekKey, slot.id, usedTargetKeys, usedMapIds);
      }
      if (!selected) return;
      const assignment = {
        id: `${slot.id}:${selected.kind}:${selected.targetId}`,
        slotId: slot.id,
        kind: selected.kind,
        type: WEEKLY_ROUTE_KIND_EVENT_TYPES[selected.kind],
        targetId: selected.targetId,
        mapId: selected.mapId,
        label: selected.label,
        summary: selected.summary,
        guideType: selected.guideType,
        guideId: selected.guideId,
        recommendedMin: selected.recommendedMin,
        recommendedMax: selected.recommendedMax,
        levelRequirement: selected.levelRequirement,
        goal: 1
      };
      assignments.push(assignment);
      usedTargetKeys.add(`${assignment.kind}:${assignment.targetId}`);
      if (assignment.mapId) usedMapIds.add(assignment.mapId);
    });
    return assignments;
  }

  function reconcileWeeklyRouteState(value, candidates, options) {
    const receivedOptionsOnly = !options && candidates && typeof candidates === 'object' && (
      candidates.candidates ||
      Object.prototype.hasOwnProperty.call(candidates, 'nowMs') ||
      Object.prototype.hasOwnProperty.call(candidates, 'config') ||
      Object.prototype.hasOwnProperty.call(candidates, 'data') ||
      Object.prototype.hasOwnProperty.call(candidates, 'seasonState') ||
      Object.prototype.hasOwnProperty.call(candidates, 'unlockSeasonComplete')
    );
    const settings = receivedOptionsOnly ? candidates : options || {};
    const candidateSource = receivedOptionsOnly ? candidates.candidates || {} : candidates || settings.candidates || {};
    const config = getWeeklyRouteConfig(settings);
    const sourceIsSeasonState = !!(value && typeof value === 'object' && value.weeklyRoutes);
    const seasonState = settings.seasonState || (sourceIsSeasonState ? value : null);
    const initial = createWeeklyRouteState(sourceIsSeasonState ? value.weeklyRoutes : value, settings);
    const next = createWeeklyRouteState(initial, settings);
    const computedWeekStartMs = getWeeklyRouteWeekStartMs(settings.nowMs);
    const effectiveWeekStartMs = Math.max(
      computedWeekStartMs,
      next.weekStartMs,
      next.rewardGrantedWeekStartMs
    );
    const clockGuarded = computedWeekStartMs < effectiveWeekStartMs;
    const unlocked = next.unlocked || settings.unlocked === true || isWeeklyRouteUnlocked(seasonState, settings);
    next.unlocked = unlocked;
    let rolledOver = false;
    let initialized = false;
    let replacedAssignmentIds = [];
    if (unlocked) {
      initialized = !next.weekStartMs;
      rolledOver = !!next.weekStartMs && effectiveWeekStartMs > next.weekStartMs;
      if (initialized || rolledOver) {
        next.weekStartMs = effectiveWeekStartMs;
        next.assignments = createWeeklyRouteAssignments(candidateSource, Object.assign({}, settings, {
          config,
          weekStartMs: effectiveWeekStartMs,
          existingAssignments: []
        }));
        next.objectiveValues = {};
        next.creditedEventKeys = [];
      } else {
        const validate = typeof settings.isAssignmentValid === 'function' ? settings.isAssignmentValid : null;
        const preserved = next.assignments.filter((assignment) => !validate || validate(assignment) !== false);
        replacedAssignmentIds = next.assignments
          .filter((assignment) => !preserved.includes(assignment))
          .map((assignment) => assignment.id);
        next.assignments = createWeeklyRouteAssignments(candidateSource, Object.assign({}, settings, {
          config,
          weekStartMs: effectiveWeekStartMs,
          existingAssignments: preserved
        }));
        const assignmentIds = new Set(next.assignments.map((assignment) => assignment.id));
        next.objectiveValues = Object.entries(next.objectiveValues).reduce((values, entry) => {
          if (assignmentIds.has(entry[0])) values[entry[0]] = entry[1];
          return values;
        }, {});
      }
    }
    const changed = JSON.stringify(initial) !== JSON.stringify(next);
    return {
      state: next,
      changed,
      initialized,
      rolledOver,
      clockGuarded,
      replacedAssignmentIds,
      weekStartMs: next.weekStartMs,
      weekKey: next.weekStartMs ? getWeeklyRouteWeekKey(next.weekStartMs) : '',
      resetAt: next.weekStartMs ? getWeeklyRouteResetAt(next.weekStartMs) : 0
    };
  }

  function weeklyRouteAssignmentMatchesEvent(assignment, type, payload) {
    if (!assignment || assignment.type !== type) return false;
    const data = payload && typeof payload === 'object' ? payload : {};
    if (type === 'mapHuntClaim') return assignment.targetId === normalizeId(data.mapId);
    if (type === 'mapMechanicComplete') {
      if (assignment.targetId !== normalizeId(data.mapMechanicId)) return false;
      return !assignment.mapId || assignment.mapId === normalizeId(data.mapId);
    }
    if (type === 'dungeonComplete') {
      if (!normalizeId(data.runId)) return false;
      if (assignment.targetId !== normalizeId(data.dungeonId)) return false;
      return !assignment.mapId || assignment.mapId === normalizeId(data.mapId);
    }
    return false;
  }

  function getWeeklyRouteCompletion(state, config) {
    const completeAssignmentIds = state.assignments
      .filter((assignment) => {
        return Math.max(0, Number(state.objectiveValues[assignment.id] || 0)) >= assignment.goal;
      })
      .map((assignment) => assignment.id);
    const completionGoal = Math.max(1, Math.floor(Number(config.completionGoal || 3) || 3));
    return {
      completeAssignmentIds,
      completionCount: completeAssignmentIds.length,
      completionGoal,
      complete: completeAssignmentIds.length >= completionGoal
    };
  }

  function cloneWeeklyRouteReward(reward) {
    const source = reward && typeof reward === 'object' ? reward : {};
    const result = {
      currency: Math.max(0, Math.floor(Number(source.currency || 0) || 0)),
      starTokens: Math.max(0, Math.floor(Number(source.starTokens || 0) || 0))
    };
    const materials = Object.entries(source.materials && typeof source.materials === 'object' ? source.materials : {})
      .reduce((values, entry) => {
        const id = normalizeId(entry[0]);
        const count = Math.max(0, Math.floor(Number(entry[1] || 0) || 0));
        if (id && count) values[id] = count;
        return values;
      }, {});
    if (Object.keys(materials).length) result.materials = materials;
    return result;
  }

  function createWeeklyRouteEventPlan(value, type, payload, options) {
    const settings = options || {};
    const config = getWeeklyRouteConfig(settings);
    const reconciliation = reconcileWeeklyRouteState(value, settings);
    const state = reconciliation.state;
    const eventType = normalizeId(type);
    const eventPayload = payload && typeof payload === 'object' ? payload : {};
    const eventKey = normalizeId(eventPayload.eventKey);
    const beforeCompletion = getWeeklyRouteCompletion(state, config);
    const result = {
      state,
      changed: reconciliation.changed,
      credited: false,
      assignmentId: '',
      eventKey,
      reason: '',
      completionCount: beforeCompletion.completionCount,
      completionGoal: beforeCompletion.completionGoal,
      completeAssignmentIds: beforeCompletion.completeAssignmentIds,
      rewardGranted: false,
      reward: null,
      reconciliation
    };
    if (!state.unlocked) {
      result.reason = 'locked';
      return result;
    }
    const requiredAssignmentCount = Math.max(1, getWeeklyRouteSlots(config).length);
    if (state.assignments.length < requiredAssignmentCount) {
      result.reason = 'routes-needed';
      return result;
    }
    if (reconciliation.initialized && settings.skipInitialEventCredit) {
      result.reason = 'initialized';
      return result;
    }
    if (!eventKey) {
      result.reason = 'missing-event-key';
      return result;
    }
    if (state.creditedEventKeys.includes(eventKey)) {
      result.reason = 'duplicate-event';
      return result;
    }
    const assignment = state.assignments.find((entry) => weeklyRouteAssignmentMatchesEvent(entry, eventType, eventPayload));
    if (!assignment) {
      result.reason = 'no-match';
      return result;
    }
    const beforeValue = Math.max(0, Number(state.objectiveValues[assignment.id] || 0));
    if (beforeValue >= assignment.goal) {
      result.reason = 'already-complete';
      return result;
    }
    state.objectiveValues[assignment.id] = Math.min(
      assignment.goal,
      beforeValue + Math.max(1, Math.floor(Number(eventPayload.count || 1) || 1))
    );
    const eventKeyLimit = Math.max(1, Math.floor(Number(config.eventKeyLimit || 32) || 32));
    state.creditedEventKeys = state.creditedEventKeys.concat(eventKey).slice(-eventKeyLimit);
    result.changed = true;
    result.credited = true;
    result.assignmentId = assignment.id;
    result.reason = 'credited';
    const completion = getWeeklyRouteCompletion(state, config);
    result.completionCount = completion.completionCount;
    result.completeAssignmentIds = completion.completeAssignmentIds;
    if (completion.complete && state.rewardGrantedWeekStartMs !== state.weekStartMs) {
      state.rewardGrantedWeekStartMs = state.weekStartMs;
      state.completedWeekCount += 1;
      result.rewardGranted = true;
      result.reward = cloneWeeklyRouteReward(config.reward);
    }
    return result;
  }

  function createWeeklyRouteSnapshot(value, options) {
    const settings = options || {};
    const config = getWeeklyRouteConfig(settings);
    const source = value && value.state && typeof value.state === 'object' ? value.state : value;
    const state = createWeeklyRouteState(source, settings);
    const completion = getWeeklyRouteCompletion(state, config);
    const weekStartMs = state.weekStartMs;
    const resetAt = weekStartMs ? getWeeklyRouteResetAt(weekStartMs) : 0;
    const nowMs = Number.isFinite(Number(settings.nowMs)) ? Number(settings.nowMs) : Date.now();
    return {
      id: normalizeId(config.id) || 'weekly_star_routes',
      name: String(config.name || 'Weekly Star Routes'),
      unlocked: state.unlocked,
      weekStartMs,
      weekKey: weekStartMs ? getWeeklyRouteWeekKey(weekStartMs) : '',
      resetAt,
      remainingMs: resetAt ? Math.max(0, resetAt - nowMs) : 0,
      clockGuarded: !!weekStartMs && getWeeklyRouteWeekStartMs(nowMs) < weekStartMs,
      completionCount: completion.completionCount,
      completionGoal: completion.completionGoal,
      complete: completion.complete,
      rewardGranted: !!weekStartMs && state.rewardGrantedWeekStartMs === weekStartMs,
      reward: cloneWeeklyRouteReward(config.reward),
      completedWeekCount: state.completedWeekCount,
      assignments: state.assignments.map((assignment) => {
        const value = Math.min(assignment.goal, Math.max(0, Number(state.objectiveValues[assignment.id] || 0)));
        return Object.assign({}, assignment, {
          value,
          complete: value >= assignment.goal
        });
      })
    };
  }

  const api = {
    createWeeklyRouteState,
    getWeeklyRouteWeekStartMs,
    getWeeklyRouteWeekKey,
    getWeeklyRouteResetAt,
    createWeeklyRouteAssignments,
    reconcileWeeklyRouteState,
    createWeeklyRouteEventPlan,
    createWeeklyRouteSnapshot,
    isWeeklyRouteUnlocked
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.weeklyRoutes = Object.assign({}, modules.weeklyRoutes || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
