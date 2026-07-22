(function initProjectStarfallEngineCombatFeedback(global) {
  'use strict';

  const BASIC_HITSTOP_MS = 45;
  const REDUCED_BASIC_HITSTOP_MS = 12;
  const CAMERA_IMPULSE_DURATION_MS = 112;
  const CAMERA_IMPULSE_STRENGTH_PX = 3.25;
  const ENEMY_HIT_REACTION_DURATION_MS = 138;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeDirection(value) {
    return Number(value) < 0 ? -1 : 1;
  }

  function createCombatFeedbackState(value) {
    const source = value && typeof value === 'object' ? value : {};
    return {
      hitstopRemainingMs: Math.max(0, Number(source.hitstopRemainingMs || 0)),
      impulseElapsedMs: Math.max(0, Number(source.impulseElapsedMs || 0)),
      impulseDurationMs: Math.max(0, Number(source.impulseDurationMs || 0)),
      impulseStrengthPx: Math.max(0, Number(source.impulseStrengthPx || 0)),
      impulseDirection: normalizeDirection(source.impulseDirection),
      serial: Math.max(0, Math.floor(Number(source.serial || 0)))
    };
  }

  function getBasicHitFeedbackProfile(options) {
    const settings = options || {};
    const reducedEffects = !!settings.reducedEffects;
    const criticalScale = settings.critical ? 1.12 : 1;
    return {
      reducedEffects,
      hitstopMs: reducedEffects
        ? REDUCED_BASIC_HITSTOP_MS
        : Math.round(BASIC_HITSTOP_MS * criticalScale),
      impulseDurationMs: reducedEffects ? 0 : CAMERA_IMPULSE_DURATION_MS,
      impulseStrengthPx: reducedEffects ? 0 : CAMERA_IMPULSE_STRENGTH_PX * criticalScale,
      enemyReactionDurationMs: reducedEffects ? 92 : ENEMY_HIT_REACTION_DURATION_MS,
      enemyRecoilPx: reducedEffects ? 0 : 6.5 * criticalScale,
      enemySquash: reducedEffects ? 0 : 0.1,
      enemyFlashAlpha: reducedEffects ? 0.2 : 0.56
    };
  }

  function triggerBasicHitFeedback(state, options) {
    const current = createCombatFeedbackState(state);
    const settings = options || {};
    const profile = getBasicHitFeedbackProfile(settings);
    const shouldRefreshImpulse = profile.impulseStrengthPx >= current.impulseStrengthPx ||
      current.impulseElapsedMs >= current.impulseDurationMs;
    return {
      hitstopRemainingMs: Math.max(current.hitstopRemainingMs, profile.hitstopMs),
      impulseElapsedMs: shouldRefreshImpulse ? 0 : current.impulseElapsedMs,
      impulseDurationMs: shouldRefreshImpulse ? profile.impulseDurationMs : current.impulseDurationMs,
      impulseStrengthPx: shouldRefreshImpulse ? profile.impulseStrengthPx : current.impulseStrengthPx,
      impulseDirection: shouldRefreshImpulse ? normalizeDirection(settings.direction) : current.impulseDirection,
      serial: current.serial + 1
    };
  }

  function advanceCombatFeedback(state, frameMs) {
    const current = createCombatFeedbackState(state);
    const elapsedMs = Math.max(0, Number(frameMs || 0));
    const hitstopConsumedMs = Math.min(current.hitstopRemainingMs, elapsedMs);
    const simulationMs = Math.max(0, elapsedMs - hitstopConsumedMs);
    const next = createCombatFeedbackState({
      hitstopRemainingMs: current.hitstopRemainingMs - hitstopConsumedMs,
      impulseElapsedMs: current.impulseElapsedMs + simulationMs,
      impulseDurationMs: current.impulseDurationMs,
      impulseStrengthPx: current.impulseStrengthPx,
      impulseDirection: current.impulseDirection,
      serial: current.serial
    });
    if (!next.impulseDurationMs || next.impulseElapsedMs >= next.impulseDurationMs) {
      next.impulseElapsedMs = next.impulseDurationMs;
      next.impulseStrengthPx = 0;
    }
    return {
      state: next,
      hitstopConsumedMs,
      simulationScale: elapsedMs > 0 ? clamp(simulationMs / elapsedMs, 0, 1) : 1
    };
  }

  function getCameraImpulseOffset(state) {
    const current = createCombatFeedbackState(state);
    if (!current.impulseStrengthPx || !current.impulseDurationMs || current.impulseElapsedMs >= current.impulseDurationMs) {
      return { x: 0, y: 0 };
    }
    const progress = clamp(current.impulseElapsedMs / current.impulseDurationMs, 0, 1);
    const envelope = (1 - progress) * (1 - progress);
    const horizontalWave = Math.cos(progress * Math.PI * 5);
    const verticalWave = Math.sin(progress * Math.PI * 4);
    return {
      x: current.impulseDirection * current.impulseStrengthPx * envelope * horizontalWave,
      y: -current.impulseStrengthPx * 0.22 * envelope * verticalWave
    };
  }

  function createEnemyHitReaction(options) {
    const settings = options || {};
    const profile = getBasicHitFeedbackProfile(settings);
    return {
      startedAtMs: Math.max(0, Number(settings.startedAtMs || 0)),
      durationMs: profile.enemyReactionDurationMs,
      direction: normalizeDirection(settings.direction),
      recoilPx: profile.enemyRecoilPx,
      squash: profile.enemySquash,
      flashAlpha: profile.enemyFlashAlpha,
      reducedEffects: profile.reducedEffects
    };
  }

  function getEnemyHitReactionState(reaction, nowMs) {
    const source = reaction && typeof reaction === 'object' ? reaction : null;
    const durationMs = Math.max(0, Number(source && source.durationMs || 0));
    const elapsedMs = Math.max(0, Number(nowMs || 0) - Number(source && source.startedAtMs || 0));
    if (!source || !durationMs || elapsedMs >= durationMs) {
      return { active: false, translateX: 0, scaleX: 1, scaleY: 1, flashAlpha: 0, progress: 1 };
    }
    const progress = clamp(elapsedMs / durationMs, 0, 1);
    const envelope = (1 - progress) * (1 - progress);
    const squashWave = Math.cos(progress * Math.PI * 3);
    const squashAmount = Number(source.squash || 0) * envelope * squashWave;
    const scaleY = clamp(1 - squashAmount, 0.86, 1.08);
    const recoilPx = Number(source.recoilPx || 0);
    return {
      active: true,
      translateX: recoilPx ? -normalizeDirection(source.direction) * recoilPx * envelope : 0,
      scaleX: clamp(2 - scaleY, 0.92, 1.14),
      scaleY,
      flashAlpha: clamp(Number(source.flashAlpha || 0) * envelope, 0, 1),
      progress
    };
  }

  function applyEnemyHitReactionToBox(box, reactionState) {
    const source = box || {};
    const state = reactionState || {};
    const width = Math.max(0, Number(source.w || 0));
    const height = Math.max(0, Number(source.h || 0));
    const nextWidth = width * Math.max(0.01, Number(state.scaleX || 1));
    const nextHeight = height * Math.max(0.01, Number(state.scaleY || 1));
    return {
      x: Number(source.x || 0) + Number(state.translateX || 0) + (width - nextWidth) / 2,
      y: Number(source.y || 0) + height - nextHeight,
      w: nextWidth,
      h: nextHeight
    };
  }

  const api = {
    BASIC_HITSTOP_MS,
    REDUCED_BASIC_HITSTOP_MS,
    CAMERA_IMPULSE_DURATION_MS,
    CAMERA_IMPULSE_STRENGTH_PX,
    ENEMY_HIT_REACTION_DURATION_MS,
    createCombatFeedbackState,
    getBasicHitFeedbackProfile,
    triggerBasicHitFeedback,
    advanceCombatFeedback,
    getCameraImpulseOffset,
    createEnemyHitReaction,
    getEnemyHitReactionState,
    applyEnemyHitReactionToBox
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.combatFeedback = Object.assign({}, modules.combatFeedback || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
