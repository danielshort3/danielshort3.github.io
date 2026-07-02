(function initProjectStarfallEngineVisuals(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const rectOverlapsBox = CoreMath.rectOverlapsBox || function rectOverlapsBoxFallback(rect, x, y, w, h) {
    return !!rect && x < rect.x + rect.w && x + w > rect.x && y < rect.y + rect.h && y + h > rect.y;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getEffectDrawBox(effect) {
    if (!effect) return { x: 0, y: 0, w: 0, h: 0 };
    if (effect.type === 'chainLine') {
      const x1 = Number(effect.x || 0);
      const y1 = Number(effect.y || 0);
      const x2 = Number(effect.x2 == null ? x1 : effect.x2);
      const y2 = Number(effect.y2 == null ? y1 : effect.y2);
      const pad = 72;
      return {
        x: Math.min(x1, x2) - pad,
        y: Math.min(y1, y2) - pad,
        w: Math.abs(x2 - x1) + pad * 2,
        h: Math.abs(y2 - y1) + pad * 2
      };
    }
    if (effect.type === 'telegraph') {
      const pad = 32;
      return {
        x: Number(effect.x || 0) - pad,
        y: Number(effect.y || 0) - pad,
        w: Math.max(1, Number(effect.w || 0)) + pad * 2,
        h: Math.max(1, Number(effect.h || 0)) + pad * 2
      };
    }
    if (effect.type === 'damageSplat') {
      return {
        x: Number(effect.x || 0) - 96,
        y: Number(effect.y || 0) - 58,
        w: 192,
        h: 116
      };
    }
    const radius = Math.max(34, Number(effect.r || effect.radius || 46));
    const width = Math.max(radius * 2, Number(effect.w || 0));
    const height = Math.max(radius * 2, Number(effect.h || 0));
    const x = Number(effect.x || 0);
    const y = Number(effect.y || 0);
    return {
      x: x - width / 2 - radius * 0.5,
      y: y - height / 2 - radius * 0.5,
      w: width + radius,
      h: height + radius
    };
  }

  function isEffectVisible(effect, viewBox) {
    if (!effect || !viewBox) return false;
    const box = getEffectDrawBox(effect);
    return rectOverlapsBox(viewBox, box.x, box.y, box.w, box.h);
  }

  function getEnemyDrawBox(enemy) {
    if (!enemy) return { x: 0, y: 0, w: 0, h: 0 };
    return {
      x: Number(enemy.x || 0) - 160,
      y: Number(enemy.y || 0) - 180,
      w: Math.max(1, Number(enemy.w || 0)) + 320,
      h: Math.max(1, Number(enemy.h || 0)) + 320
    };
  }

  function getProjectileDrawBox(projectile) {
    if (!projectile) return { x: 0, y: 0, w: 0, h: 0 };
    const radius = Math.max(28, Number(projectile.r || projectile.radius || projectile.size || 0));
    return {
      x: Number(projectile.x || 0) - radius,
      y: Number(projectile.y || 0) - radius,
      w: Math.max(1, Number(projectile.w || 0)) + radius * 2,
      h: Math.max(1, Number(projectile.h || 0)) + radius * 2
    };
  }

  function getVisibleEnemiesForDraw(enemies, viewBox, target) {
    const sourceEnemies = enemies || [];
    if (!viewBox && !Array.isArray(target)) return sourceEnemies;
    const visible = Array.isArray(target) ? target : [];
    visible.length = 0;
    for (let index = 0; index < sourceEnemies.length; index += 1) {
      const enemy = sourceEnemies[index];
      if (!enemy) continue;
      const box = getEnemyDrawBox(enemy);
      if (!viewBox || rectOverlapsBox(viewBox, box.x, box.y, box.w, box.h)) visible.push(enemy);
    }
    return visible;
  }

  function getVisibleProjectilesForDraw(projectiles, viewBox, target) {
    const sourceProjectiles = projectiles || [];
    if (!viewBox && !Array.isArray(target)) return sourceProjectiles;
    const visible = Array.isArray(target) ? target : [];
    visible.length = 0;
    for (let index = 0; index < sourceProjectiles.length; index += 1) {
      const projectile = sourceProjectiles[index];
      if (!projectile) continue;
      if (!viewBox) {
        visible.push(projectile);
        continue;
      }
      const box = getProjectileDrawBox(projectile);
      if (rectOverlapsBox(viewBox, box.x, box.y, box.w, box.h)) visible.push(projectile);
    }
    return visible;
  }

  function createVisualBudgetStats(options) {
    const settings = options || {};
    return {
      reducedEffects: !!settings.reducedEffects,
      damageNumbers: settings.damageNumbers || 'normal',
      visualQualityLevel: 'normal',
      effectBudget: Number(settings.effectBudget || 260),
      damageSplatBudget: Number(settings.damageSplatBudget || 220),
      worldEffectsDrawn: 0,
      worldEffectsVisible: 0,
      worldEffectsSkippedOffscreen: 0,
      worldEffectsSkippedBudget: 0,
      damageSplatsDrawn: 0,
      damageSplatsVisible: 0,
      damageSplatsSkippedOffscreen: 0,
      damageSplatsSkippedBudget: 0,
      simplifiedDamageSplats: false
    };
  }

  function createVisualFrameStats(samples, options) {
    const settings = options || {};
    const sourceSamples = Array.isArray(samples) ? samples : [];
    const sampleCount = sourceSamples.length;
    const sampleLimit = Math.max(1, Math.floor(Number(settings.sampleLimit || 45) || 45));
    const averageValue = typeof settings.averageValue === 'function'
      ? settings.averageValue
      : (values) => {
        const list = (values || []).filter((value) => Number.isFinite(Number(value)));
        if (!list.length) return 0;
        return list.reduce((sum, value) => sum + Number(value), 0) / list.length;
      };
    const percentileValue = typeof settings.percentileValue === 'function'
      ? settings.percentileValue
      : (values, percentile) => {
        const list = (values || [])
          .map((value) => Number(value))
          .filter(Number.isFinite)
          .sort((a, b) => a - b);
        if (!list.length) return 0;
        const index = Math.max(0, Math.min(list.length - 1, Math.ceil(list.length * percentile) - 1));
        return list[index];
      };
    const frameMsValues = [];
    const start = Math.max(0, sampleCount - sampleLimit);
    for (let index = start; index < sampleCount; index += 1) {
      const value = Number(sourceSamples[index] && sourceSamples[index].frameMs || 0);
      if (value > 0) frameMsValues.push(value);
    }
    return frameMsValues.length ? {
      averageFrameMs: averageValue(frameMsValues),
      p95FrameMs: percentileValue(frameMsValues, 0.95)
    } : { averageFrameMs: 0, p95FrameMs: 0 };
  }

  function createAdaptiveVisualQualityState(totalEffects, totalDamageSplats, options) {
    const settings = options || {};
    const frameStats = settings.frameStats || {};
    const enemies = settings.enemies || [];
    const projectiles = settings.projectiles || [];
    let liveEnemies = 0;
    for (let index = 0; index < enemies.length; index += 1) {
      const enemy = enemies[index];
      if (enemy && Number(enemy.hp || 0) > 0) liveEnemies += 1;
    }
    const projectileCount = Number(projectiles.length || 0);
    const effectPressure = Number(totalEffects || 0) >= Number(settings.effectPressureCount || 520);
    const damagePressure = Number(totalDamageSplats || 0) >= Number(settings.damageSplatPressureCount || 260);
    const combatPressure = liveEnemies >= Number(settings.enemyPressureCount || 58) ||
      projectileCount >= Number(settings.projectilePressureCount || 90) ||
      effectPressure ||
      damagePressure;
    const framePressure = Number(frameStats.averageFrameMs || 0) >= Number(settings.averageFrameMs || 20) ||
      Number(frameStats.p95FrameMs || 0) >= Number(settings.p95FrameMs || 28);
    const userReducedEffects = !!settings.userReducedEffects;
    const reduced = !!(userReducedEffects || combatPressure || framePressure);
    return {
      level: reduced ? 'reduced' : 'normal',
      reduceEffects: reduced,
      simplifyDamageSplats: userReducedEffects || combatPressure || damagePressure || framePressure,
      hideFarHpBars: reduced && (combatPressure || framePressure),
      framePressure,
      combatPressure,
      averageFrameMs: Number(frameStats.averageFrameMs || 0),
      p95FrameMs: Number(frameStats.p95FrameMs || 0),
      liveEnemies,
      projectileCount,
      totalEffects: Number(totalEffects || 0),
      totalDamageSplats: Number(totalDamageSplats || 0)
    };
  }

  function getWorldEffectDrawBudget(totalEffects, quality, options) {
    const settings = options || {};
    const visualQuality = quality || {};
    const effectPressureCount = Number(settings.effectPressureCount || 520);
    const normalBudget = Number(settings.normalBudget || 260);
    const reducedBudget = Number(settings.reducedBudget || 120);
    const pressureBudget = Number(settings.pressureBudget || 72);
    const pressure = Number(totalEffects || 0) >= effectPressureCount;
    if (visualQuality.combatPressure || visualQuality.framePressure) return pressureBudget;
    return visualQuality.reduceEffects || pressure ? reducedBudget : normalBudget;
  }

  function getDamageSplatDrawBudget(totalSplats, mode, quality, options) {
    const settings = options || {};
    const visualQuality = quality || {};
    const pressureCount = Number(settings.pressureCount || 260);
    const normalBudget = Number(settings.normalBudget || 220);
    const reducedBudget = Number(settings.reducedBudget || 140);
    const pressureBudget = Number(settings.pressureBudget || 48);
    const minimalBudget = Number(settings.minimalBudget || 32);
    if (mode === 'minimal') return minimalBudget;
    if (visualQuality.reduceEffects || Number(totalSplats || 0) >= pressureCount) return pressureBudget;
    if (mode === 'reduced') return reducedBudget;
    return normalBudget;
  }

  function getWorldEffectPriority(effect) {
    const type = effect && effect.type || '';
    let score = 0;
    if (type === 'lootPickup' || type === 'upgradeResult' || type === 'potentialCubeResult') score = 800;
    else if (type === 'recoveryPulse') score = 650;
    else if (type === 'skillImpact' || type === 'shockBurst') score = 500;
    else if (type === 'slash' || type === 'cast' || type === 'arrowRelease') score = 420;
    else if (type === 'chainLine') score = 320;
    else if (type === 'field') score = 140;
    score += Math.max(0, Number(effect && effect.ttl || 0)) * 10;
    return score;
  }

  function getDamageSplatPriority(effect) {
    if (!effect) return 0;
    let score = 0;
    const targetType = effect.targetType || '';
    if (targetType === 'player' || targetType === 'xp') score += 1200;
    if (effect.critical) score += 700;
    if (effect.isBossDamage) score += 500;
    if (effect.isTickDamage) score -= 180;
    score -= Math.max(0, Number(effect.lineIndex || 0)) * 24;
    const amount = effect.amount ? Math.abs(Number(effect.amount || 0)) : 0;
    if (amount > 0) score += Math.min(240, Math.log10(amount + 1) * 42);
    const visibleAge = Math.max(0, Number(effect.age || 0));
    score += Math.max(0, 120 - visibleAge * 60);
    return score;
  }

  function isBudgetedEffectEntryWorse(a, b) {
    if (a.priority !== b.priority) return a.priority < b.priority;
    return a.index < b.index;
  }

  function siftBudgetedEffectEntryUp(heap, index) {
    let current = index;
    while (current > 0) {
      const parent = Math.floor((current - 1) / 2);
      if (!isBudgetedEffectEntryWorse(heap[current], heap[parent])) break;
      const swap = heap[parent];
      heap[parent] = heap[current];
      heap[current] = swap;
      current = parent;
    }
  }

  function siftBudgetedEffectEntryDown(heap, index) {
    let current = index;
    while (true) {
      const left = current * 2 + 1;
      const right = left + 1;
      let worst = current;
      if (left < heap.length && isBudgetedEffectEntryWorse(heap[left], heap[worst])) worst = left;
      if (right < heap.length && isBudgetedEffectEntryWorse(heap[right], heap[worst])) worst = right;
      if (worst === current) break;
      const swap = heap[current];
      heap[current] = heap[worst];
      heap[worst] = swap;
      current = worst;
    }
  }

  function selectBudgetedEffectEntries(entries, skippedOffscreen, budget, getPriority, options) {
    const settings = options || {};
    const sourceEntries = entries;
    const drawBudget = Math.max(0, Math.floor(Number(budget) || 0));
    const visibleCount = sourceEntries.length;
    let selectedEntries = sourceEntries;
    let skippedBudget = 0;
    if (drawBudget && visibleCount > drawBudget) {
      skippedBudget = visibleCount - drawBudget;
      selectedEntries = [];
      const priorityMode = getPriority === 'damageSplat' || getPriority === 'world' ? getPriority : '';
      const priorityFn = priorityMode ? null : getPriority;
      const worldPriority = settings.getWorldEffectPriority || getWorldEffectPriority;
      const damageSplatPriority = settings.getDamageSplatPriority || getDamageSplatPriority;
      for (let index = 0; index < sourceEntries.length; index += 1) {
        const entry = sourceEntries[index];
        entry.priority = priorityMode === 'damageSplat'
          ? damageSplatPriority(entry.effect)
          : priorityMode === 'world'
            ? worldPriority(entry.effect)
            : priorityFn(entry.effect);
        if (selectedEntries.length < drawBudget) {
          selectedEntries.push(entry);
          siftBudgetedEffectEntryUp(selectedEntries, selectedEntries.length - 1);
        } else if (isBudgetedEffectEntryWorse(selectedEntries[0], entry)) {
          selectedEntries[0] = entry;
          siftBudgetedEffectEntryDown(selectedEntries, 0);
        }
      }
      selectedEntries.sort((a, b) => a.index - b.index);
    }
    const effects = new Array(selectedEntries.length);
    for (let index = 0; index < selectedEntries.length; index += 1) effects[index] = selectedEntries[index].effect;
    return {
      effects,
      visible: visibleCount,
      skippedOffscreen,
      skippedBudget,
      budget: drawBudget || visibleCount
    };
  }

  function getBudgetedVisibleEffects(effects, type, viewBox, budget, options) {
    const settings = options || {};
    const entries = [];
    let skippedOffscreen = 0;
    const sourceEffects = effects || [];
    const effectVisible = settings.isEffectVisible || isEffectVisible;
    for (let index = 0; index < sourceEffects.length; index += 1) {
      const effect = sourceEffects[index];
      if (!effect || effect.type !== type) continue;
      if (!effectVisible(effect, viewBox)) {
        skippedOffscreen += 1;
        continue;
      }
      entries.push({ effect, index });
    }
    return selectBudgetedEffectEntries(entries, skippedOffscreen, budget, type === 'damageSplat' ? 'damageSplat' : 'world', settings);
  }

  function getVisibleWorldEffects(effects, viewBox, budget, options) {
    const settings = options || {};
    const entries = [];
    let skippedOffscreen = 0;
    const sourceEffects = effects || [];
    const effectVisible = settings.isEffectVisible || isEffectVisible;
    for (let index = 0; index < sourceEffects.length; index += 1) {
      const effect = sourceEffects[index];
      if (!effect || !(effect.type !== 'damageSplat' && effect.type !== 'levelUpBurst')) continue;
      if (!effectVisible(effect, viewBox)) {
        skippedOffscreen += 1;
        continue;
      }
      entries.push({ effect, index });
    }
    return selectBudgetedEffectEntries(
      entries,
      skippedOffscreen,
      budget,
      'world',
      settings
    );
  }

  function createVisualDrawListState(effects, worldViewBox, splatViewBox, options) {
    const settings = options || {};
    const sourceEffects = effects || [];
    const effectVisible = settings.isEffectVisible || isEffectVisible;
    const worldEntries = [];
    const splatEntries = [];
    const levelUpBursts = [];
    let damageSplatCount = 0;
    let worldSkippedOffscreen = 0;
    let splatSkippedOffscreen = 0;
    for (let index = 0; index < sourceEffects.length; index += 1) {
      const effect = sourceEffects[index];
      if (!effect) continue;
      if (effect.type === 'damageSplat') {
        damageSplatCount += 1;
        if (effectVisible(effect, splatViewBox)) splatEntries.push({ effect, index });
        else splatSkippedOffscreen += 1;
        continue;
      }
      if (effect.type === 'levelUpBurst') {
        if (effectVisible(effect, worldViewBox)) levelUpBursts.push(effect);
        continue;
      }
      if (effectVisible(effect, worldViewBox)) worldEntries.push({ effect, index });
      else worldSkippedOffscreen += 1;
    }
    const visualQuality = typeof settings.getAdaptiveVisualQuality === 'function'
      ? settings.getAdaptiveVisualQuality(sourceEffects.length, damageSplatCount)
      : settings.visualQuality || {};
    const damageMode = typeof settings.getDamageNumberMode === 'function'
      ? settings.getDamageNumberMode()
      : settings.damageMode || 'normal';
    const worldBudget = getWorldEffectDrawBudget(sourceEffects.length, visualQuality, {
      effectPressureCount: settings.effectPressureCount,
      normalBudget: settings.worldEffectNormalBudget,
      reducedBudget: settings.worldEffectReducedBudget,
      pressureBudget: settings.worldEffectPressureBudget
    });
    const splatBudget = getDamageSplatDrawBudget(damageSplatCount, damageMode, visualQuality, {
      pressureCount: settings.damageSplatPressureCount,
      normalBudget: settings.damageSplatNormalBudget,
      reducedBudget: settings.damageSplatReducedBudget,
      pressureBudget: settings.damageSplatPressureBudget,
      minimalBudget: settings.damageSplatMinimalBudget
    });
    const world = selectBudgetedEffectEntries(
      worldEntries,
      worldSkippedOffscreen,
      worldBudget,
      'world',
      settings
    );
    const splats = selectBudgetedEffectEntries(
      splatEntries,
      splatSkippedOffscreen,
      splatBudget,
      'damageSplat',
      settings
    );
    const qualitySnapshot = Object.assign({}, visualQuality, {
      damageNumbers: damageMode,
      worldEffectBudget: world.budget,
      damageSplatBudget: splats.budget,
      simplifyDamageSplats: !!(visualQuality.simplifyDamageSplats || damageSplatCount >= Number(settings.damageSplatPressureCount || 260)),
      reduceEffects: !!(visualQuality.reduceEffects || world.effects.length < worldEntries.length)
    });
    return {
      worldEffects: world.effects,
      damageSplats: splats.effects,
      levelUpBursts,
      visualQuality: qualitySnapshot,
      visualBudgetStats: {
        reducedEffects: qualitySnapshot.reduceEffects,
        damageNumbers: damageMode,
        visualQualityLevel: qualitySnapshot.level,
        effectBudget: world.budget,
        damageSplatBudget: splats.budget,
        worldEffectsDrawn: world.effects.length,
        worldEffectsVisible: world.visible,
        worldEffectsSkippedOffscreen: world.skippedOffscreen,
        worldEffectsSkippedBudget: world.skippedBudget,
        damageSplatsDrawn: splats.effects.length,
        damageSplatsVisible: splats.visible,
        damageSplatsSkippedOffscreen: splats.skippedOffscreen,
        damageSplatsSkippedBudget: splats.skippedBudget,
        simplifiedDamageSplats: qualitySnapshot.simplifyDamageSplats
      }
    };
  }

  function createAnimationFrameSourceRect(frameDef, options) {
    if (!frameDef) return null;
    const settings = options || {};
    const frameWidth = Number(frameDef.frameWidth) || Number(settings.frameWidth) || 160;
    const frameHeight = Number(frameDef.frameHeight) || Number(settings.frameHeight) || 160;
    return {
      sourceX: Math.max(0, Number(frameDef.frameIndex || 0)) * frameWidth,
      sourceY: Math.max(0, Number(frameDef.row || 0)) * frameHeight,
      frameWidth,
      frameHeight
    };
  }

  function createAnimationFrameDrawState(frameDef, x, y, width, height, facing, options) {
    const source = createAnimationFrameSourceRect(frameDef, options);
    if (!source) return null;
    return Object.assign({}, source, {
      translateX: x + width / 2,
      translateY: y + height / 2,
      scaleX: (Number(facing) || 1) < 0 ? -1 : 1,
      scaleY: 1,
      drawX: -width / 2,
      drawY: -height / 2,
      drawWidth: width,
      drawHeight: height
    });
  }

  function createRendererAnimationFrame(animation, frame) {
    if (!frame || !animation || !animation.sheet) return null;
    return {
      sheet: animation.sheet,
      row: Number(frame.row || 0),
      frameIndex: Number(frame.frameIndex || 0),
      frameWidth: Number(frame.frameWidth || animation.frameWidth || 160),
      frameHeight: Number(frame.frameHeight || animation.frameHeight || 160)
    };
  }

  function createRendererAnimationFrameForIndex(animation, state, frameIndex, options) {
    if (!animation || !animation.sheet || !animation.states) return null;
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const getAnimationFrameUnits = typeof settings.getAnimationFrameUnits === 'function'
      ? settings.getAnimationFrameUnits
      : (frameDef) => frameDef || {};
    const stateId = normalize(state);
    const frameDef = stateId && animation.states[stateId];
    if (!frameDef) return null;
    const units = getAnimationFrameUnits(frameDef);
    const frameCount = Math.max(1, Number(units && units.frameCount || frameDef.frames || 1) || 1);
    return {
      sheet: animation.sheet,
      row: Number(frameDef.row || 0),
      frameIndex: clamp(Math.floor(Number(frameIndex || 0) || 0), 0, frameCount - 1),
      frameWidth: Number(animation.frameWidth || 160),
      frameHeight: Number(animation.frameHeight || 160)
    };
  }

  function getCombatFxAnimationState(animation, requestedState, options) {
    if (!animation || !animation.states) return '';
    const settings = options || {};
    const normalize = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeId;
    const state = normalize(requestedState);
    if (state && animation.states[state]) return state;
    if (animation.states.impact) return 'impact';
    if (animation.states.projectile) return 'projectile';
    if (animation.states.cast) return 'cast';
    return Object.keys(animation.states)[0] || '';
  }

  function createCombatFxAnimationFrame(animation, requestedState, elapsedSeconds, options) {
    const settings = options || {};
    const stateId = getCombatFxAnimationState(animation, requestedState, settings);
    if (!stateId || !animation || !animation.states || !animation.sheet) return null;
    const frameDef = animation.states[stateId];
    if (!frameDef) return null;
    const elapsed = Math.max(0, Number(elapsedSeconds || 0));
    const getWeightedAnimationFrameIndex = typeof settings.getWeightedAnimationFrameIndex === 'function'
      ? settings.getWeightedAnimationFrameIndex
      : (sourceFrameDef, sourceElapsed) => {
        const frames = Math.max(1, Number(sourceFrameDef && sourceFrameDef.frames || 1) || 1);
        const fps = Math.max(1, Number(sourceFrameDef && sourceFrameDef.fps || 1) || 1);
        return Math.floor(Math.max(0, Number(sourceElapsed || 0)) * fps) % frames;
      };
    return {
      sheet: animation.sheet,
      row: Number(frameDef.row || 0),
      frameIndex: getWeightedAnimationFrameIndex(frameDef, elapsed),
      frameWidth: Number(animation.frameWidth || 160),
      frameHeight: Number(animation.frameHeight || 160)
    };
  }

  function createTimedCombatFxAnimationFrame(animation, requestedState, ttl, duration, options) {
    const settings = options || {};
    const stateId = getCombatFxAnimationState(animation, requestedState, settings);
    const frameDef = stateId && animation && animation.states ? animation.states[stateId] : null;
    const getAnimationDuration = typeof settings.getAnimationDuration === 'function'
      ? settings.getAnimationDuration
      : (sourceFrameDef) => Number(sourceFrameDef && sourceFrameDef.duration || 0);
    const fallbackDuration = frameDef
      ? getAnimationDuration(frameDef)
      : 0.35;
    const resolvedDuration = Math.max(0.01, Number(duration || fallbackDuration) || fallbackDuration);
    const elapsed = clamp(resolvedDuration - Number(ttl || 0), 0, resolvedDuration);
    if (typeof settings.createCombatFxAnimationFrame === 'function') {
      return settings.createCombatFxAnimationFrame(animation, stateId, elapsed);
    }
    return createCombatFxAnimationFrame(animation, stateId, elapsed, settings);
  }

  function getEffectCombatFxState(effect, fallback) {
    if (!effect) return fallback || 'impact';
    if (effect.combatFxState) return effect.combatFxState;
    if (effect.type === 'skillCast' || effect.type === 'cast' || effect.type === 'arrowRelease') return 'cast';
    if (effect.type === 'skillArea' || effect.type === 'field' || effect.type === 'shockBurst') return 'area';
    if (effect.type === 'slash') return effect.enemyFxId ? 'melee' : 'trail';
    if (effect.type === 'bossPhase' || effect.type === 'telegraph') return 'telegraph';
    return fallback || 'impact';
  }

  function getProjectileCombatFxSize(projectile) {
    const explicitSize = Number(projectile && (projectile.projectileVisualSize || projectile.visualSize) || 0);
    if (explicitSize > 0) return explicitSize;
    const width = Number(projectile && projectile.w || 0);
    const height = Number(projectile && projectile.h || 0);
    const radius = Number(projectile && (projectile.r || projectile.radius) || 0);
    const type = String(projectile && projectile.type || '');
    const base = projectile && projectile.owner === 'enemy'
      ? 88
      : type === 'arrow'
        ? 92
        : type === 'lightning'
          ? 104
          : 96;
    return Math.max(base, radius * 3.2, Math.max(width, height) * 4.2);
  }

  function createProjectileCombatFxDrawState(projectile, options) {
    if (!projectile) return null;
    const settings = options || {};
    const getSize = typeof settings.getProjectileCombatFxSize === 'function'
      ? settings.getProjectileCombatFxSize
      : getProjectileCombatFxSize;
    const totalTtl = Math.max(0.01, Number(projectile.totalTtl || projectile.ttl || 0.6));
    return {
      centerX: Number(projectile.x || 0) + Number(projectile.w || 0) / 2,
      centerY: Number(projectile.y || 0) + Number(projectile.h || 0) / 2,
      angle: Math.atan2(Number(projectile.vy || 0), Number(projectile.vx || 1)),
      totalTtl,
      ttl: Number(projectile.ttl || 0),
      size: getSize(projectile),
      state: projectile.combatFxState || 'projectile',
      alpha: settings.alpha == null ? 0.96 : Number(settings.alpha)
    };
  }

  function createEffectCombatFxDrawState(effect, state) {
    if (!effect || effect.type === 'chainLine') return null;
    const duration = Math.max(0.01, Number(effect.duration || effect.ttl || 0.36));
    const ttl = Number(effect.ttl || 0);
    const alpha = clamp(ttl / duration, 0, 1);
    const radius = Math.max(18, Number(effect.r || effect.radius || 42));
    const resolvedState = state || getEffectCombatFxState(effect);
    const size = resolvedState === 'area'
      ? Math.max(124, radius * 2.15)
      : resolvedState === 'melee' || resolvedState === 'trail'
        ? Math.max(96, radius * 2.2)
        : Math.max(86, radius * 1.85);
    return {
      state: resolvedState,
      duration,
      ttl,
      alpha,
      radius,
      size,
      elapsed: Number.isFinite(Number(effect.visualAge)) ? Number(effect.visualAge) : null,
      x: effect.x,
      y: effect.y,
      facing: Number(effect.facing || 1)
    };
  }

  function createFxAnimationDrawState(effect, animation, options) {
    if (!effect || !animation || !animation.states) return null;
    const settings = options || {};
    const state = animation.states[effect.type] ? effect.type : Object.keys(animation.states)[0];
    const frameDef = animation.states[state];
    if (!frameDef) return null;
    const getAnimationDuration = typeof settings.getAnimationDuration === 'function'
      ? settings.getAnimationDuration
      : (sourceFrameDef) => Number(sourceFrameDef && sourceFrameDef.duration || 0);
    const getWeightedAnimationFrameIndex = typeof settings.getWeightedAnimationFrameIndex === 'function'
      ? settings.getWeightedAnimationFrameIndex
      : (sourceFrameDef, sourceElapsed) => {
        const frames = Math.max(1, Number(sourceFrameDef && sourceFrameDef.frames || 1) || 1);
        const fps = Math.max(1, Number(sourceFrameDef && sourceFrameDef.fps || 1) || 1);
        return Math.floor(Math.max(0, Number(sourceElapsed || 0)) * fps) % frames;
      };
    const duration = Math.max(0.01, Number(effect.duration) || Number(effect.ttl) || getAnimationDuration(frameDef));
    const elapsed = clamp(duration - Number(effect.ttl || 0), 0, duration);
    const frameIndex = getWeightedAnimationFrameIndex(frameDef, elapsed);
    const alpha = clamp(Number(effect.ttl || 0) / duration, 0, 1);
    const size = Math.max(42, Number(effect.r || 48) * (effect.type === 'partyBuff' || effect.type === 'defeatBurst' ? 2 : 1.65));
    return {
      state,
      frameDef,
      frame: Object.assign({}, frameDef, {
        frameIndex,
        frameWidth: animation.frameWidth,
        frameHeight: animation.frameHeight
      }),
      duration,
      elapsed,
      frameIndex,
      alpha,
      size,
      x: effect.x,
      y: effect.y,
      facing: effect.facing || 1
    };
  }

  function createLootPickupEffectDrawState(effect) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 0.26));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const startX = Number(effect.x || 0);
    const startY = Number(effect.y || 0);
    const targetX = Number(effect.targetX || startX);
    const targetY = Number(effect.targetY || startY);
    return {
      duration,
      progress,
      ease,
      startX,
      startY,
      targetX,
      targetY,
      x: startX + (targetX - startX) * ease,
      y: startY + (targetY - startY) * ease - Math.sin(progress * Math.PI) * 22,
      size: Math.max(12, Number(effect.size || 32) * (1 - progress * 0.28)),
      alpha: clamp(1 - progress * 0.86, 0, 1)
    };
  }

  function createUpgradeResultEffectDrawState(effect) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 1.15));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const alpha = clamp(1 - Math.max(0, progress - 0.72) / 0.28, 0, 1);
    const success = effect.outcome === 'success';
    const destroy = effect.outcome === 'destroy';
    const color = effect.color || (success ? '#34c759' : destroy ? '#7f1d1d' : '#d64545');
    const accent = effect.accentColor || '#ffffff';
    const bob = success
      ? -Math.sin(progress * Math.PI) * 18
      : Math.sin(progress * Math.PI * 8) * (1 - progress) * 8;
    return {
      duration,
      progress,
      alpha,
      success,
      destroy,
      color,
      accent,
      bob,
      x: Number(effect.x || 0) + (destroy ? Math.sin(progress * Math.PI * 10) * (1 - progress) * 6 : 0),
      y: Number(effect.y || 0) - progress * 22 + bob,
      ringRadius: 28 + Math.sin(progress * Math.PI) * 5,
      symbol: success ? '+' : destroy ? 'X' : 'v',
      text: String(effect.text || '').toUpperCase(),
      itemName: String(effect.itemName || '').slice(0, 28),
      textFill: success ? '#eaffea' : '#fff0f0'
    };
  }

  function createPotentialCubeResultEffectDrawState(effect) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 1.35));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const alpha = clamp(1 - Math.max(0, progress - 0.82) / 0.18, 0, 1);
    const color = effect.color || '#9e72ff';
    const accent = effect.accentColor || '#ff8fc2';
    const pulse = Math.sin(progress * Math.PI);
    const size = 24 + pulse * 9;
    const rings = [];
    for (let ring = 0; ring < 3; ring += 1) {
      const ringProgress = clamp(progress * 1.18 - ring * 0.14, 0, 1);
      if (ringProgress <= 0) continue;
      rings.push({
        index: ring,
        progress: ringProgress,
        alpha: alpha * (0.5 - ring * 0.1) * (1 - ringProgress * 0.62),
        color: ring % 2 ? accent : color,
        lineWidth: ring === 0 ? 3 : 1.6,
        radius: 26 + ringProgress * (46 + ring * 10)
      });
    }
    const motes = [];
    for (let index = 0; index < 10; index += 1) {
      const angle = progress * Math.PI * 2 + index * Math.PI / 3;
      const moteRadius = 22 + progress * 36 + (index % 3) * 4;
      motes.push({
        index,
        angle,
        radius: moteRadius,
        x: Math.cos(angle) * moteRadius,
        y: Math.sin(angle) * moteRadius * 0.72,
        drawRadius: index % 3 === 0 ? 3.2 : 2.3,
        color: index % 2 ? accent : '#ffffff',
        alpha: alpha * (0.35 + (1 - progress) * 0.5)
      });
    }
    const streaks = [];
    for (let streak = -1; streak <= 1; streak += 1) {
      streaks.push({
        index: streak,
        fromX: -52,
        fromY: streak * 10 + 18 - progress * 36,
        toX: 52,
        toY: streak * 10 - 18 + progress * 36
      });
    }
    return {
      duration,
      progress,
      alpha,
      color,
      accent,
      x: Number(effect.x || 0),
      y: Number(effect.y || 0) - pulse * 24 - progress * 12,
      pulse,
      size,
      rings,
      motes,
      streaks,
      streakAlpha: alpha * 0.46 * pulse,
      rotation: progress * Math.PI * 1.5,
      squareX: -size / 2,
      squareY: -size / 2,
      squareSize: size,
      innerSquareX: -size / 4,
      innerSquareY: -size / 4,
      innerSquareSize: size / 2,
      text: String(effect.text || 'ATTUNEMENT').toUpperCase(),
      itemName: String(effect.itemName || '').slice(0, 28)
    };
  }

  function createPetTeleportEffectDrawState(effect, options) {
    if (!effect) return null;
    const settings = options || {};
    const duration = Math.max(0.01, Number(effect.duration || settings.duration || 0.42));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const pulse = Math.sin(progress * Math.PI);
    const color = effect.color || '#7bdff2';
    const accent = effect.accentColor || '#ffe16a';
    const createGate = (x, y, reverse) => {
      const motes = [];
      for (let index = 0; index < 6; index += 1) {
        const angle = progress * Math.PI * 2 + index * Math.PI * 2 / 6;
        motes.push({
          index,
          angle,
          x: Math.cos(angle) * (18 + pulse * 8),
          y: Math.sin(angle) * (28 + pulse * 8),
          w: 3,
          h: 3
        });
      }
      return {
        x: Number(x || 0),
        y: Number(y || 0),
        translateX: Number(x || 0),
        translateY: Number(y || 0) - 20,
        reverse: !!reverse,
        alpha: clamp(0.18 + pulse * 0.72, 0, 1),
        outerRadiusX: 15 + pulse * 10,
        outerRadiusY: 29 + pulse * 12,
        outerRotation: reverse ? -progress * 1.8 : progress * 1.8,
        innerRadiusX: 7 + pulse * 7,
        innerRadiusY: 22 + pulse * 8,
        innerRotation: reverse ? progress * 2.4 : -progress * 2.4,
        motes
      };
    };
    const startX = Number(effect.x || 0);
    const startY = Number(effect.y || 0);
    const endX = Number(effect.x2 || effect.x || 0);
    const endY = Number(effect.y2 || effect.y || 0);
    return {
      duration,
      progress,
      pulse,
      color,
      accent,
      gates: [
        createGate(startX, startY, true),
        createGate(endX, endY, false)
      ],
      linkAlpha: pulse * 0.45,
      link: {
        fromX: startX,
        fromY: startY - 20,
        toX: endX,
        toY: endY - 20,
        dash: [6, 8]
      }
    };
  }

  function createBossPhaseEffectDrawState(effect) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 1.05));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const alpha = clamp(Number(effect.ttl || 0) / duration, 0, 1);
    const radius = Math.max(72, Number(effect.r || 110));
    const color = effect.color || '#ffbe55';
    const accent = effect.accentColor || '#ffffff';
    const pulse = Math.sin(progress * Math.PI);
    const motes = [];
    for (let index = 0; index < 8; index += 1) {
      const angle = progress * Math.PI * 2 + index * Math.PI / 4;
      motes.push({
        index,
        angle,
        x: Math.cos(angle) * radius * 0.48,
        y: Math.sin(angle) * radius * 0.2,
        w: 4,
        h: 4
      });
    }
    return {
      duration,
      progress,
      alpha,
      radius,
      color,
      accent,
      pulse,
      translateX: Number(effect.x || 0),
      translateY: Number(effect.y || 0),
      compositeOperation: 'lighter',
      shadowBlur: 24 * alpha,
      outerEllipse: {
        radiusX: radius * (0.58 + progress * 0.38),
        radiusY: radius * (0.24 + progress * 0.18),
        rotation: 0
      },
      innerEllipse: {
        radiusX: radius * (0.34 + progress * 0.32),
        radiusY: radius * (0.14 + progress * 0.12),
        rotation: 0
      },
      motes,
      labelAlpha: alpha * clamp(0.35 + pulse, 0, 1),
      text: String(effect.text || 'PHASE').toUpperCase().slice(0, 24),
      textY: -radius * 0.36,
      subtitle: effect.subtitle ? String(effect.subtitle || '').slice(0, 46) : '',
      subtitleY: -radius * 0.21
    };
  }

  function createRuneFieldTimerBarDrawState(effect, color, lifeRatio, options) {
    if (!effect || Number(effect.ttl || 0) <= 0) return null;
    const settings = options || {};
    const radius = Math.max(1, Number(effect.r || 80));
    const duration = Math.max(0.01, Number(effect.duration || effect.baseDuration || effect.ttl || 1));
    const fillRatio = clamp(lifeRatio, 0, 1);
    const barW = clamp(radius * 1.18, 72, 164);
    const barH = 6;
    const x = Number(effect.x || 0) - barW / 2;
    const y = Number(effect.y || 0) - Math.max(28, radius * 0.26);
    const currentTime = Number(settings.nowSeconds);
    const refillPulseSeconds = Math.max(0.01, Number(settings.refillPulseSeconds || 0.35));
    const lastExtension = Math.max(0, Number(effect.lastExtensionAmount || 0));
    let refillPulse = null;
    if (lastExtension > 0 && Number.isFinite(currentTime)) {
      const pulseAge = currentTime - Number(effect.lastExtendedAt || 0);
      if (pulseAge >= 0 && pulseAge < refillPulseSeconds) {
        const pulseAlpha = 0.78 * (1 - pulseAge / refillPulseSeconds);
        const pulseW = Math.max(5, barW * clamp(lastExtension / duration, 0.035, 0.45));
        const pulseX = x + Math.max(0, barW * fillRatio - pulseW);
        refillPulse = {
          age: pulseAge,
          alpha: pulseAlpha,
          x: pulseX,
          y,
          w: Math.min(pulseW, x + barW - pulseX),
          h: barH
        };
      }
    }
    return {
      radius,
      duration,
      fillRatio,
      color,
      x,
      y,
      w: barW,
      h: barH,
      fillW: barW * fillRatio,
      outline: {
        x: x - 0.5,
        y: y - 0.5,
        w: barW + 1,
        h: barH + 1
      },
      refillPulse
    };
  }

  function createRuneFieldGroundVisualDrawState(effect, color, lifeRatio, options) {
    if (!effect || Number(effect.ttl || 0) <= 0) return null;
    const settings = options || {};
    const drawAura = settings.aura !== false;
    const drawPulse = settings.pulse !== false;
    const radius = Math.max(1, Number(effect.r || 80));
    const fieldHeight = Math.max(22, Number(effect.verticalTolerance || radius * 0.18));
    const x = Number(effect.x || 0);
    const y = Number(effect.y || 0);
    let aura = null;
    if (drawAura) {
      aura = {
        alpha: clamp(0.42 + lifeRatio * 0.36, 0, 0.82),
        fillAlpha: 0.12 + lifeRatio * 0.14,
        strokeAlpha: 0.26 + lifeRatio * 0.36,
        lineWidth: 2.2,
        shadowColor: color,
        shadowBlur: 8 + lifeRatio * 12,
        fillEllipse: { x, y, radiusX: radius, radiusY: fieldHeight, rotation: 0 },
        strokeEllipse: { x, y, radiusX: radius * 0.98, radiusY: fieldHeight * 0.98, rotation: 0 }
      };
    }
    let pulse = null;
    let impact = null;
    if (drawPulse) {
      const progress = clamp(Number(effect.pulseProgress || 0), 0, 1);
      if (progress > 0.02) {
        const pulseAlpha = Math.sin(progress * Math.PI);
        pulse = {
          progress,
          pulseAlpha,
          alpha: clamp(0.18 + pulseAlpha * 0.7, 0, 0.88),
          strokeAlpha: 0.22 + pulseAlpha * 0.54,
          lineWidth: 1.4 + pulseAlpha * 3.2,
          shadowColor: color,
          shadowBlur: 10 + pulseAlpha * 18,
          ellipse: {
            x,
            y,
            radiusX: Math.max(10, radius * progress),
            radiusY: Math.max(8, fieldHeight * progress),
            rotation: 0
          }
        };
      }
      const currentTime = Number(settings.nowSeconds);
      const pulseAge = currentTime - Number(effect.lastPulseAt || 0);
      if (Number(effect.lastPulseHits || 0) > 0 && pulseAge >= 0 && pulseAge < 0.16) {
        impact = {
          age: pulseAge,
          alpha: 0.46 * (1 - pulseAge / 0.16),
          strokeAlpha: 0.72,
          lineWidth: 3,
          ellipse: { x, y, radiusX: radius, radiusY: fieldHeight, rotation: 0 }
        };
      }
    }
    return {
      drawAura,
      drawPulse,
      radius,
      fieldHeight,
      x,
      y,
      color,
      lifeRatio,
      aura,
      pulse,
      impact
    };
  }

  function createShockBurstEffectDrawState(effect, visual) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 0.72));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const alpha = clamp(effect.ttl, 0, 1);
    const color = visual && visual.color || effect.color || '#d8f6ff';
    const accent = visual && visual.accent || effect.accentColor || '#7aa7ff';
    const radius = Number(effect.r || 46);
    const bodyW = Math.max(22, Number(effect.w || 36));
    const bodyH = Math.max(34, Number(effect.h || 58));
    const pulseIndex = Number(effect.pulseIndex || 0);
    const flicker = 0.72 + Math.sin((progress + pulseIndex) * Math.PI * 8) * 0.18;
    const x = effect.x;
    const y = effect.y;
    const bolts = [];
    for (let index = 0; index < 6; index += 1) {
      const side = index % 2 ? -1 : 1;
      const top = y - bodyH * 0.48 + index * bodyH * 0.16;
      const startX = x + side * (bodyW * 0.34 + Math.sin(progress * 8 + index) * 5);
      bolts.push({
        index,
        side,
        startX,
        top,
        midX: startX + side * (10 + index % 3 * 4),
        midY: top + 9,
        endX: startX - side * 4,
        endY: top + 18
      });
    }
    const sparks = [];
    for (let index = 0; index < 7; index += 1) {
      const angle = progress * Math.PI * 2 + index * Math.PI * 2 / 7;
      const sparkRadius = radius * (0.38 + (index % 3) * 0.12);
      sparks.push({
        index,
        angle,
        radius: sparkRadius,
        x: x + Math.cos(angle) * sparkRadius,
        y: y + Math.sin(angle) * sparkRadius * 0.72,
        w: 3,
        h: 3
      });
    }
    return {
      duration,
      progress,
      alpha,
      color,
      accent,
      radius,
      bodyW,
      bodyH,
      pulseIndex,
      flicker,
      bodyAlpha: alpha * flicker,
      bodyEllipse: {
        x,
        y,
        radiusX: bodyW * 0.78 + progress * 11,
        radiusY: bodyH * 0.58 + progress * 9,
        rotation: 0
      },
      bolts,
      sparks
    };
  }

  function createChainLineEffectDrawState(effect, visual) {
    if (!effect) return null;
    const duration = Math.max(0.01, Number(effect.duration || 0.24));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const color = visual && visual.color || effect.color || '#d8f6ff';
    const accent = visual && visual.accent || effect.accentColor || '#7aa7ff';
    const dx = Number(effect.x2 || effect.x) - Number(effect.x || 0);
    const dy = Number(effect.y2 || effect.y) - Number(effect.y || 0);
    const length = Math.max(1, Math.hypot(dx, dy));
    const tx = dx / length;
    const ty = dy / length;
    const nx = -ty;
    const ny = tx;
    const head = clamp(0.08 + progress * 1.08, 0.08, 1);
    const tail = clamp(head - 0.42, 0, 1);
    const pulseIndex = Number(effect.pulseIndex || 0);
    const points = [];
    for (let index = 0; index <= 5; index += 1) {
      const localT = index / 5;
      const t = tail + (head - tail) * localT;
      const jag = (index === 0 || index === 5) ? 0 : (index % 2 ? 1 : -1) * (8 + progress * 8 + pulseIndex * 1.5);
      points.push({
        index,
        localT,
        t,
        jag,
        x: effect.x + dx * t + nx * jag,
        y: effect.y + dy * t + ny * jag
      });
    }
    const branches = points.slice(1, -1).map((point, index) => {
      const side = index % 2 ? -1 : 1;
      return {
        index,
        side,
        fromX: point.x,
        fromY: point.y,
        toX: point.x + nx * side * (16 + progress * 10) + tx * 12,
        toY: point.y + ny * side * (16 + progress * 10) + ty * 12
      };
    });
    const pulseAlpha = Math.max(0.18, Math.sin(progress * Math.PI) * 0.86);
    const headPoint = points[points.length - 1] || { x: effect.x, y: effect.y };
    return {
      duration,
      progress,
      color,
      accent,
      dx,
      dy,
      length,
      tx,
      ty,
      nx,
      ny,
      head,
      tail,
      pulseIndex,
      pulseAlpha,
      shadowBlur: 18 * pulseAlpha,
      points,
      branches,
      headPoint,
      headRadius: 3.5 + progress * 2.5
    };
  }

  function createRecoveryPulseEffectDrawState(effect) {
    if (!effect) return null;
    const alpha = clamp(effect.ttl, 0, 1);
    const duration = Math.max(0.01, Number(effect.duration || 0.5));
    const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
    const color = effect.color || '#72e6c9';
    const accent = effect.accentColor || '#ffffff';
    const radius = Math.max(14, Number(effect.r || 42));
    const ringW = radius * (1.05 + progress * 0.65);
    const ringH = Math.max(16, radius * 0.36 * (1 + progress * 0.28));
    const x = effect.x;
    const y = effect.y;
    const rays = [];
    for (let index = -1; index <= 1; index += 1) {
      const gx = x + index * radius * 0.22;
      rays.push({
        index,
        fromX: gx,
        fromY: y - ringH * 0.68 - progress * 8,
        toX: gx + index * 4,
        toY: y - ringH * 1.22 - progress * 13
      });
    }
    return {
      alpha,
      duration,
      progress,
      color,
      accent,
      radius,
      ringW,
      ringH,
      drawAlpha: alpha * (1 - progress * 0.25),
      fillAlpha: 0.16,
      strokeAlpha: 0.74,
      accentStrokeAlpha: 0.68 * (1 - progress * 0.35),
      fillEllipse: { x, y, radiusX: ringW, radiusY: ringH, rotation: 0 },
      strokeEllipse: { x, y, radiusX: ringW * 0.82, radiusY: ringH * 0.9, rotation: 0 },
      lineWidth: 2.4,
      accentLineWidth: 1.6,
      rays
    };
  }

  function createFloatingNumberEffectDrawState(effect) {
    if (!effect) return null;
    const alpha = clamp(effect.ttl, 0, 1);
    return {
      alpha,
      color: effect.color || '#ffffff',
      font: 'bold 16px system-ui',
      textAlign: 'center',
      text: effect.text,
      x: effect.x,
      y: effect.y - (1 - alpha) * 24
    };
  }

  function createFieldEffectDrawState(effect) {
    if (!effect) return null;
    const color = effect.color || '#28c7b7';
    const duration = Math.max(0.01, Number(effect.duration || effect.baseDuration || effect.ttl || 1));
    const lifeRatio = clamp(Number(effect.ttl || 0) / duration, 0, 1);
    return {
      color,
      duration,
      lifeRatio,
      runeField: !!effect.runeField,
      fillAlpha: 0.33,
      ellipse: {
        x: effect.x,
        y: effect.y,
        radiusX: effect.r,
        radiusY: 22,
        rotation: 0
      }
    };
  }

  function createTelegraphEffectDrawState(effect) {
    if (!effect) return null;
    return {
      fillStyle: `${effect.color || '#ef5b4c'}88`,
      x: effect.x,
      y: effect.y,
      w: effect.w,
      h: effect.h
    };
  }

  function createFallbackCircleEffectDrawState(effect) {
    if (!effect) return null;
    return {
      strokeStyle: effect.color || '#ffffff',
      lineWidth: 4,
      arc: {
        x: effect.x,
        y: effect.y,
        radius: effect.r || 36,
        startAngle: 0,
        endAngle: Math.PI * 2
      }
    };
  }

  function createAttractScreenDrawState(width, height) {
    const centerX = width / 2;
    const centerY = height / 2;
    return {
      background: {
        fillStyle: 'rgba(9,31,59,0.52)',
        x: 0,
        y: 0,
        w: width,
        h: height
      },
      textFillStyle: '#ffffff',
      textAlign: 'center',
      title: {
        text: 'Project Starfall',
        font: '700 28px system-ui',
        x: centerX,
        y: centerY - 12
      },
      subtitle: {
        text: 'Choose Fighter, Mage, or Archer to start the prototype.',
        font: '16px system-ui',
        x: centerX,
        y: centerY + 22
      }
    };
  }

  function createProjectileRendererAnimationFrame(projectile, animation, options) {
    if (!animation) return null;
    const settings = options || {};
    const createFrame = typeof settings.createCombatFxAnimationFrame === 'function'
      ? settings.createCombatFxAnimationFrame
      : (sourceAnimation, state, elapsed) => createCombatFxAnimationFrame(sourceAnimation, state, elapsed, settings);
    const elapsed = Math.max(0, Number(projectile && projectile.totalTtl || projectile && projectile.ttl || 0) - Number(projectile && projectile.ttl || 0));
    return createFrame(animation, projectile && projectile.combatFxState || 'projectile', elapsed);
  }

  function createEffectRendererAnimationFrame(effect, animation, options) {
    if (!animation) return null;
    const settings = options || {};
    const getFxState = typeof settings.getEffectCombatFxState === 'function'
      ? settings.getEffectCombatFxState
      : getEffectCombatFxState;
    const createFrame = typeof settings.createCombatFxAnimationFrame === 'function'
      ? settings.createCombatFxAnimationFrame
      : (sourceAnimation, state, elapsed) => createCombatFxAnimationFrame(sourceAnimation, state, elapsed, settings);
    const createTimedFrame = typeof settings.createTimedCombatFxAnimationFrame === 'function'
      ? settings.createTimedCombatFxAnimationFrame
      : (sourceAnimation, state, ttl, duration) => createTimedCombatFxAnimationFrame(sourceAnimation, state, ttl, duration, settings);
    const state = getFxState(effect);
    if (Number.isFinite(Number(effect && effect.visualAge))) {
      return createFrame(animation, state, Number(effect.visualAge));
    }
    return createTimedFrame(
      animation,
      state,
      effect && effect.ttl,
      effect && effect.duration
    );
  }

  const api = {
    getEffectDrawBox,
    isEffectVisible,
    getEnemyDrawBox,
    getProjectileDrawBox,
    getVisibleEnemiesForDraw,
    getVisibleProjectilesForDraw,
    createVisualBudgetStats,
    createVisualFrameStats,
    createAdaptiveVisualQualityState,
    getWorldEffectDrawBudget,
    getDamageSplatDrawBudget,
    getWorldEffectPriority,
    getDamageSplatPriority,
    isBudgetedEffectEntryWorse,
    siftBudgetedEffectEntryUp,
    siftBudgetedEffectEntryDown,
    selectBudgetedEffectEntries,
    getBudgetedVisibleEffects,
    getVisibleWorldEffects,
    createVisualDrawListState,
    createAnimationFrameSourceRect,
    createAnimationFrameDrawState,
    createRendererAnimationFrame,
    createRendererAnimationFrameForIndex,
    getCombatFxAnimationState,
    createCombatFxAnimationFrame,
    createTimedCombatFxAnimationFrame,
    getEffectCombatFxState,
    getProjectileCombatFxSize,
    createProjectileCombatFxDrawState,
    createEffectCombatFxDrawState,
    createFxAnimationDrawState,
    createLootPickupEffectDrawState,
    createUpgradeResultEffectDrawState,
    createPotentialCubeResultEffectDrawState,
    createPetTeleportEffectDrawState,
    createBossPhaseEffectDrawState,
    createRuneFieldTimerBarDrawState,
    createRuneFieldGroundVisualDrawState,
    createShockBurstEffectDrawState,
    createChainLineEffectDrawState,
    createRecoveryPulseEffectDrawState,
    createFloatingNumberEffectDrawState,
    createFieldEffectDrawState,
    createTelegraphEffectDrawState,
    createFallbackCircleEffectDrawState,
    createAttractScreenDrawState,
    createProjectileRendererAnimationFrame,
    createEffectRendererAnimationFrame
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.visuals = Object.assign({}, modules.visuals || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
