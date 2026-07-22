(function initProjectStarfallEngineRiftCounterplay(global) {
  'use strict';

  const COUNTERPLAY_CONFIG = Object.freeze({
    interrupt_echo: Object.freeze({ label: 'Break the Echo', duration: 4.5 }),
    control_spacing: Object.freeze({ label: 'Control the Spread', duration: 5 }),
    break_guard: Object.freeze({ label: 'Shatter the Guard', duration: 5 }),
    rotate_hazards: Object.freeze({ label: 'Rotate the Heat', duration: 5 }),
    break_focus: Object.freeze({ label: 'Break Their Focus', duration: 4.5 }),
    burst_window: Object.freeze({ label: 'Choose the Burst Window', duration: 4 })
  });
  const COUNTERPLAY_IDS = Object.freeze(Object.keys(COUNTERPLAY_CONFIG));
  const MOVEMENT_WINDOW_SECONDS = 3;
  const MOVEMENT_DISTANCE_GOAL = 240;
  const CONTROL_HIT_WINDOW_SECONDS = 0.85;
  const CONTROL_MIN_SPACING = 70;
  const CONTROL_MAX_SPACING = 420;
  const GUARD_PRESSURE_WINDOW_SECONDS = 2.75;
  const GUARD_DAMAGE_GOAL = 0.14;
  const FOCUS_FOLLOWUP_SECONDS = 2.4;
  const RESOURCE_HOLD_SECONDS = 1.25;
  const BURST_FOLLOWUP_SECONDS = 2;

  function normalizeId(value) {
    return String(value || '').trim();
  }

  function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : Number(fallback || 0);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, finiteNumber(value, min)));
  }

  function createRiftCounterplayState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const activeUntil = {};
    const lastActivatedAt = {};
    COUNTERPLAY_IDS.forEach((id) => {
      const until = Math.max(0, finiteNumber(source.activeUntil && source.activeUntil[id], 0));
      const activatedAt = Math.max(0, finiteNumber(source.lastActivatedAt && source.lastActivatedAt[id], 0));
      if (until > 0) activeUntil[id] = until;
      if (activatedAt > 0) lastActivatedAt[id] = activatedAt;
    });
    const recentHits = (Array.isArray(source.recentHits) ? source.recentHits : []).slice(-8).map((hit) => ({
      enemyKey: normalizeId(hit && hit.enemyKey),
      x: finiteNumber(hit && hit.x, 0),
      platformIndex: Math.floor(finiteNumber(hit && hit.platformIndex, -1)),
      at: Math.max(0, finiteNumber(hit && hit.at, 0))
    })).filter((hit) => hit.enemyKey);
    const guard = source.guardPressure && typeof source.guardPressure === 'object' ? source.guardPressure : {};
    return {
      revision: Math.max(0, Math.floor(finiteNumber(source.revision, 0))),
      mapId: normalizeId(source.mapId),
      activeUntil,
      lastActivatedAt,
      movementWindowStartedAt: finiteNumber(source.movementWindowStartedAt, -1),
      movementDistance: Math.max(0, finiteNumber(source.movementDistance, 0)),
      laneChangedAt: finiteNumber(source.laneChangedAt, -1),
      recentHits,
      guardPressure: {
        enemyKey: normalizeId(guard.enemyKey),
        hitCount: Math.max(0, Math.floor(finiteNumber(guard.hitCount, 0))),
        damageFraction: Math.max(0, finiteNumber(guard.damageFraction, 0)),
        lastHitAt: finiteNumber(guard.lastHitAt, -1)
      },
      highResourceStartedAt: finiteNumber(source.highResourceStartedAt, -1),
      burstPrimedUntil: Math.max(0, finiteNumber(source.burstPrimedUntil, 0))
    };
  }

  function cloneRiftCounterplayState(value) {
    const state = createRiftCounterplayState(value);
    return Object.assign({}, state, {
      activeUntil: Object.assign({}, state.activeUntil),
      lastActivatedAt: Object.assign({}, state.lastActivatedAt),
      recentHits: state.recentHits.map((hit) => Object.assign({}, hit)),
      guardPressure: Object.assign({}, state.guardPressure)
    });
  }

  function getActiveRiftCounterplayIds(value, now) {
    const state = createRiftCounterplayState(value);
    const time = finiteNumber(now, 0);
    return COUNTERPLAY_IDS.filter((id) => Number(state.activeUntil[id] || 0) > time);
  }

  function activateCounterplay(state, id, now, allowedIds) {
    const config = COUNTERPLAY_CONFIG[id];
    if (!config || allowedIds && !allowedIds.has(id) || Number(state.activeUntil[id] || 0) > now) return false;
    state.activeUntil[id] = now + config.duration;
    state.lastActivatedAt[id] = now;
    state.revision += 1;
    return true;
  }

  function pruneExpiredCounterplay(state, now) {
    let changed = false;
    COUNTERPLAY_IDS.forEach((id) => {
      if (!Number(state.activeUntil[id] || 0) || Number(state.activeUntil[id]) > now) return;
      delete state.activeUntil[id];
      changed = true;
    });
    if (changed) state.revision += 1;
    return changed;
  }

  function observeResource(state, event, now) {
    const maxResource = Math.max(0, finiteNumber(event.maxResource, 0));
    const resource = Math.max(0, finiteNumber(event.resource, 0));
    if (!maxResource || resource / maxResource < 0.72) {
      state.highResourceStartedAt = -1;
      return;
    }
    if (state.highResourceStartedAt < 0) state.highResourceStartedAt = now;
  }

  function reduceMovement(state, event, now, activatedIds, allowedIds) {
    const dx = Math.abs(finiteNumber(event.dx, 0));
    const dy = Math.abs(finiteNumber(event.dy, 0));
    const laneDelta = Math.max(dy, Math.abs(finiteNumber(event.laneDelta, 0)));
    const fromPlatformIndex = Math.floor(finiteNumber(event.fromPlatformIndex, -1));
    const toPlatformIndex = Math.floor(finiteNumber(event.toPlatformIndex, -1));
    if (fromPlatformIndex >= 0 && toPlatformIndex >= 0 && fromPlatformIndex !== toPlatformIndex && laneDelta >= 28) {
      state.laneChangedAt = now;
    }
    if (!event.inCombat || dx < 4) return;
    if (state.movementWindowStartedAt < 0 || now - state.movementWindowStartedAt > MOVEMENT_WINDOW_SECONDS) {
      state.movementWindowStartedAt = now;
      state.movementDistance = 0;
    }
    state.movementDistance += dx;
    if (state.movementDistance < MOVEMENT_DISTANCE_GOAL) return;
    if (activateCounterplay(state, 'rotate_hazards', now, allowedIds)) activatedIds.push('rotate_hazards');
    state.movementWindowStartedAt = -1;
    state.movementDistance = 0;
  }

  function reduceSkillSpent(state, event, now) {
    const maxResource = Math.max(0, finiteNumber(event.maxResource, 0));
    const resourceBefore = Math.max(0, finiteNumber(event.resourceBefore, 0));
    const cost = Math.max(0, finiteNumber(event.cost, 0));
    const heldLongEnough = state.highResourceStartedAt >= 0 && now - state.highResourceStartedAt >= RESOURCE_HOLD_SECONDS;
    if (!event.offensive || !maxResource || cost < 5 || resourceBefore / maxResource < 0.72 || !heldLongEnough) {
      return;
    }
    state.burstPrimedUntil = now + BURST_FOLLOWUP_SECONDS;
    state.highResourceStartedAt = -1;
  }

  function reduceDirectHit(state, event, now, activatedIds, allowedIds) {
    const enemyKey = normalizeId(event.enemyKey);
    if (!enemyKey) return;
    if (event.telegraphing && activateCounterplay(state, 'interrupt_echo', now, allowedIds)) activatedIds.push('interrupt_echo');

    const currentHit = {
      enemyKey,
      x: finiteNumber(event.x, 0),
      platformIndex: Math.floor(finiteNumber(event.platformIndex, -1)),
      at: now
    };
    state.recentHits = state.recentHits.filter((hit) => now - Number(hit.at || 0) <= CONTROL_HIT_WINDOW_SECONDS);
    const controlledPair = state.recentHits.find((hit) => {
      if (hit.enemyKey === currentHit.enemyKey) return false;
      if (hit.platformIndex < 0 || currentHit.platformIndex < 0 || hit.platformIndex !== currentHit.platformIndex) return false;
      const spacing = Math.abs(hit.x - currentHit.x);
      return spacing >= CONTROL_MIN_SPACING && spacing <= CONTROL_MAX_SPACING;
    });
    state.recentHits.push(currentHit);
    state.recentHits = state.recentHits.slice(-8);
    if (controlledPair && activateCounterplay(state, 'control_spacing', now, allowedIds)) {
      activatedIds.push('control_spacing');
      state.recentHits = [];
    }

    const maxHealth = Math.max(1, finiteNumber(event.maxHealth, 1));
    const damageFraction = Math.max(0, finiteNumber(event.damage, 0)) / maxHealth;
    const guard = state.guardPressure;
    if (guard.enemyKey !== enemyKey || guard.lastHitAt < 0 || now - guard.lastHitAt > GUARD_PRESSURE_WINDOW_SECONDS) {
      state.guardPressure = {
        enemyKey,
        hitCount: 1,
        damageFraction,
        lastHitAt: now
      };
    } else {
      guard.hitCount += 1;
      guard.damageFraction += damageFraction;
      guard.lastHitAt = now;
    }
    const pressure = state.guardPressure;
    if (pressure.hitCount >= 2 && pressure.damageFraction >= GUARD_DAMAGE_GOAL || pressure.hitCount >= 4) {
      if (activateCounterplay(state, 'break_guard', now, allowedIds)) activatedIds.push('break_guard');
      state.guardPressure = { enemyKey: '', hitCount: 0, damageFraction: 0, lastHitAt: -1 };
    }

    if (state.laneChangedAt >= 0 && now - state.laneChangedAt <= FOCUS_FOLLOWUP_SECONDS) {
      if (activateCounterplay(state, 'break_focus', now, allowedIds)) activatedIds.push('break_focus');
      state.laneChangedAt = -1;
    }

    if (event.skillId && Number(state.burstPrimedUntil || 0) > now) {
      if (activateCounterplay(state, 'burst_window', now, allowedIds)) activatedIds.push('burst_window');
      state.burstPrimedUntil = 0;
    }
  }

  function reduceRiftCounterplay(value, event) {
    const action = event && typeof event === 'object' ? event : {};
    const now = Math.max(0, finiteNumber(action.now, 0));
    if (action.type === 'reset') {
      const reset = createRiftCounterplayState({ mapId: action.mapId });
      reset.revision = Math.max(0, Math.floor(finiteNumber(value && value.revision, 0))) + 1;
      return { state: reset, activatedIds: [], expiredIds: getActiveRiftCounterplayIds(value, now) };
    }
    const previousState = createRiftCounterplayState(value);
    const previousIds = COUNTERPLAY_IDS.filter((id) => Number(previousState.activeUntil[id] || 0) > 0);
    const state = cloneRiftCounterplayState(value);
    const activatedIds = [];
    const allowedIds = Array.isArray(action.allowedIds)
      ? new Set(action.allowedIds.map(normalizeId).filter((id) => COUNTERPLAY_CONFIG[id]))
      : null;
    pruneExpiredCounterplay(state, now);
    if (allowedIds) {
      let removedDisallowed = false;
      COUNTERPLAY_IDS.forEach((id) => {
        if (!state.activeUntil[id] || allowedIds.has(id)) return;
        delete state.activeUntil[id];
        removedDisallowed = true;
      });
      if (removedDisallowed) state.revision += 1;
    }
    if (action.type === 'tick' || action.type === 'move') observeResource(state, action, now);
    if (action.type === 'move') reduceMovement(state, action, now, activatedIds, allowedIds);
    if (action.type === 'skill_spent') reduceSkillSpent(state, action, now);
    if (action.type === 'direct_hit') reduceDirectHit(state, action, now, activatedIds, allowedIds);
    if (action.type === 'damage_taken') {
      state.movementWindowStartedAt = -1;
      state.movementDistance = 0;
      state.laneChangedAt = -1;
    }
    const activeIds = getActiveRiftCounterplayIds(state, now);
    const activeSet = new Set(activeIds);
    const expiredIds = previousIds.filter((id) => !activeSet.has(id));
    return { state, activatedIds, expiredIds, activeIds };
  }

  function createRiftCounterplaySnapshot(value, now) {
    const state = createRiftCounterplayState(value);
    const time = Math.max(0, finiteNumber(now, 0));
    const activeIds = getActiveRiftCounterplayIds(state, time);
    return {
      revision: state.revision,
      activeIds,
      active: activeIds.map((id) => ({
        id,
        label: COUNTERPLAY_CONFIG[id].label,
        remaining: Math.max(0, Number(state.activeUntil[id] || 0) - time),
        duration: COUNTERPLAY_CONFIG[id].duration
      })),
      progress: {
        movementDistance: Math.min(MOVEMENT_DISTANCE_GOAL, Math.round(state.movementDistance)),
        movementDistanceGoal: MOVEMENT_DISTANCE_GOAL,
        guardHits: state.guardPressure.hitCount,
        guardDamageProgress: clamp(state.guardPressure.damageFraction / GUARD_DAMAGE_GOAL, 0, 1),
        focusFollowupReady: state.laneChangedAt >= 0 && time - state.laneChangedAt <= FOCUS_FOLLOWUP_SECONDS,
        burstFollowupReady: Number(state.burstPrimedUntil || 0) > time,
        resourceHoldProgress: state.highResourceStartedAt >= 0
          ? clamp((time - state.highResourceStartedAt) / RESOURCE_HOLD_SECONDS, 0, 1)
          : 0
      }
    };
  }

  const api = {
    COUNTERPLAY_CONFIG,
    COUNTERPLAY_IDS,
    MOVEMENT_DISTANCE_GOAL,
    createRiftCounterplayState,
    getActiveRiftCounterplayIds,
    reduceRiftCounterplay,
    createRiftCounterplaySnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.riftCounterplay = Object.assign({}, modules.riftCounterplay || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
