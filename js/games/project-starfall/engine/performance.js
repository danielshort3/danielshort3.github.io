(function initProjectStarfallEnginePerformance(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreSettings = (typeof require === 'function' ? require('../core/settings.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const hashString = CoreMath.hashString || function hashStringFallback(value) {
    const text = String(value || '');
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  };
  const seededUnit = CoreMath.seededUnit || function seededUnitFallback(seed, salt) {
    let value = hashString(`${seed}:${salt}`);
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return ((value >>> 0) % 10000) / 10000;
  };
  const normalizePerformanceDebugModeSetting = CoreSettings.normalizePerformanceDebugMode || function normalizePerformanceDebugModeFallback(value, modes, fallback) {
    const source = Array.isArray(modes) ? modes : [];
    const defaultValue = String(fallback || 'off');
    const mode = String(value || defaultValue);
    return source.includes(mode) ? mode : defaultValue;
  };

  const PERFORMANCE_DEBUG_MODES = Object.freeze(['off', 'fps', 'breakdown']);
  const PERFORMANCE_DEBUG_WINDOW_MS = 10000;
  const PERFORMANCE_DEBUG_SAMPLE_LIMIT = 3600;
  const PERFORMANCE_DEBUG_REPORT_FRAME_LIMIT = 60;
  const PERFORMANCE_DEBUG_SLOW_FRAME_MS = 33.34;
  const PERFORMANCE_BENCHMARK_SCENARIO_ID = 'intense-combat';
  const PERFORMANCE_BENCHMARK_MAP_ID = 'performanceBenchmarkArena';
  const PERFORMANCE_BENCHMARK_MAP_NAME = 'Benchmark Arena';
  const PERFORMANCE_BENCHMARK_SCENE_SEED = `${PERFORMANCE_BENCHMARK_SCENARIO_ID}:dedicated-arena:v1`;
  const PERFORMANCE_BENCHMARK_CHARACTER = Object.freeze({
    classId: 'mage',
    advancedClassId: 'stormMage',
    name: 'Benchmark Echo',
    level: 120
  });
  const PERFORMANCE_BENCHMARK_ENEMY_IDS = Object.freeze([
    'vineSnapper',
    'coilSentry',
    'emberWisp',
    'stormboundArcher',
    'indexScribe',
    'lumenSentinel',
    'voidMote',
    'eclipseDuelist'
  ]);
  const PERFORMANCE_BENCHMARK_WARMUP_MS = 2000;
  const PERFORMANCE_BENCHMARK_SAMPLE_MS = 8000;
  const PERFORMANCE_BENCHMARK_COUNTS = Object.freeze({
    enemies: 72,
    projectiles: 120,
    effects: 180,
    damageSplats: 160
  });
  const PERFORMANCE_BENCHMARK_MAXED_SKILL_IDS = Object.freeze([
    'mage_magic_bolt',
    'mage_arcane_burst',
    'storm_mage_chain_bolt',
    'storm_mage_stormfront'
  ]);
  const PERFORMANCE_DEBUG_COUNTS_CACHE_MS = 120;

  function normalizePerformanceDebugMode(value, fallback) {
    return normalizePerformanceDebugModeSetting(value, PERFORMANCE_DEBUG_MODES, fallback || 'off');
  }

  function createPerformanceDebugRuntime() {
    return {
      samples: [],
      currentFrame: null,
      pendingUiPhases: {},
      uiActions: [],
      overlayStats: {},
      lastSnapshot: null,
      lastSnapshotAt: 0,
      disabledSummarySnapshot: null,
      disabledSummarySnapshotAt: 0
    };
  }

  function createPerformanceFrameState(timestamp, frameMs, delta, options) {
    const settings = options || {};
    const frame = {
      timestamp: Number(timestamp) || Number(settings.fallbackTimestampMs || 0),
      capturedAt: Number(settings.capturedAt || 0),
      frameMs: Math.max(0, Number(frameMs) || 0),
      deltaMs: Math.max(0, Number(delta || 0) * 1000),
      update: {},
      draw: {},
      counts: {}
    };
    const pendingUiPhases = settings.pendingUiPhases || {};
    Object.entries(pendingUiPhases).forEach(([phaseId, ms]) => {
      frame.update[phaseId] = Number(frame.update[phaseId] || 0) + Math.max(0, Number(ms) || 0);
    });
    return frame;
  }

  function recordPerformancePhaseState(runtime, group, name, ms) {
    if (!runtime) return false;
    const frame = runtime.currentFrame;
    const groupId = group === 'draw' ? 'draw' : 'update';
    const phaseId = String(name || '').trim();
    if (!phaseId) return false;
    if (!frame) {
      if (group === 'ui' || phaseId.startsWith('ui:')) {
        runtime.pendingUiPhases = runtime.pendingUiPhases || {};
        runtime.pendingUiPhases[phaseId] = Number(runtime.pendingUiPhases[phaseId] || 0) + Math.max(0, Number(ms) || 0);
        return true;
      }
      return false;
    }
    frame[groupId] = frame[groupId] || {};
    frame[groupId][phaseId] = Number(frame[groupId][phaseId] || 0) + Math.max(0, Number(ms) || 0);
    return true;
  }

  function finishPerformanceFrameState(runtime, counts, options) {
    if (!runtime) return null;
    const settings = options || {};
    const frame = runtime.currentFrame;
    if (!frame) return null;
    frame.counts = Object.assign({}, counts || {}, frame.counts || {});
    runtime.samples = Array.isArray(runtime.samples) ? runtime.samples : [];
    runtime.samples.push(frame);
    const latestTimestamp = Number(frame.timestamp || 0);
    if (latestTimestamp) {
      const windowMs = Number(settings.windowMs == null ? PERFORMANCE_DEBUG_WINDOW_MS : settings.windowMs);
      const reportFrameLimit = Number(settings.reportFrameLimit || PERFORMANCE_DEBUG_REPORT_FRAME_LIMIT);
      const cutoff = latestTimestamp - windowMs;
      while (runtime.samples.length > reportFrameLimit &&
        Number(runtime.samples[0] && runtime.samples[0].timestamp || 0) < cutoff) {
        runtime.samples.shift();
      }
    }
    const sampleLimit = Number(settings.sampleLimit || PERFORMANCE_DEBUG_SAMPLE_LIMIT);
    if (runtime.samples.length > sampleLimit) {
      runtime.samples.splice(0, runtime.samples.length - sampleLimit);
    }
    runtime.currentFrame = null;
    return frame;
  }

  function recordPerformanceUiActionState(runtime, record, options) {
    if (!runtime) return null;
    const settings = options || {};
    const entry = Object.assign({
      actionId: '',
      totalMs: 0,
      snapshotCalls: 0,
      drawRequests: 0,
      panelCacheHits: 0,
      panelCacheMisses: 0,
      capturedAt: Number(settings.capturedAt || 0)
    }, record || {});
    const maxActions = Math.max(1, Math.floor(Number(settings.maxActions || 24) || 24));
    runtime.uiActions = Array.isArray(runtime.uiActions) ? runtime.uiActions : [];
    runtime.uiActions.unshift(entry);
    if (runtime.uiActions.length > maxActions) runtime.uiActions.length = maxActions;
    return entry;
  }

  function recordPerformanceOverlayStatsState(runtime, stats) {
    if (!runtime) return null;
    runtime.overlayStats = Object.assign({}, runtime.overlayStats || {}, stats || {});
    const frame = runtime.currentFrame;
    if (frame) frame.counts = Object.assign({}, frame.counts || {}, runtime.overlayStats);
    return runtime.overlayStats;
  }

  function resetPerformanceDebugRuntimeState(runtime) {
    if (!runtime) return null;
    runtime.samples = [];
    runtime.lastSnapshot = null;
    runtime.lastSnapshotAt = 0;
    runtime.disabledSummarySnapshot = null;
    runtime.disabledSummarySnapshotAt = 0;
    runtime.countsCache = null;
    return runtime;
  }

  function createPerformanceDebugCountsState(options) {
    const settings = options || {};
    const state = settings.state || {};
    const player = settings.player || state.player || {};
    const overlay = settings.overlay || {};
    const visuals = settings.visuals || {};
    const runtime = settings.runtime || {};
    const enemies = Array.isArray(settings.enemies) ? settings.enemies : [];
    const projectiles = settings.projectiles || [];
    const effects = Array.isArray(settings.effects) ? settings.effects : [];
    let liveEnemyCount = 0;
    let damageSplatCount = 0;
    for (let index = 0; index < enemies.length; index += 1) {
      const enemy = enemies[index];
      if (enemy && Number(enemy.hp || 0) > 0) liveEnemyCount += 1;
    }
    for (let index = 0; index < effects.length; index += 1) {
      const effect = effects[index];
      if (effect && effect.type === 'damageSplat') damageSplatCount += 1;
    }
    const heldSkillIds = settings.heldSkillIds;
    const heldSkillRetryAt = settings.heldSkillRetryAt;
    return {
      mapId: state.mapId || '',
      channelId: settings.channelId || '',
      classId: player.advancedClassId || player.classId || '',
      level: Number(player.level || 1),
      frameRateLimit: settings.frameRateLimit,
      skippedRafFrames: Number(settings.skippedFrameCount || 0),
      renderBackend: settings.renderBackend || 'Canvas 2D',
      renderBackendStatus: settings.renderBackendStatus || 'canvas',
      worldWidth: Number(runtime.worldWidth || 0),
      enemies: liveEnemyCount,
      enemyActors: enemies.length,
      projectiles: Number(projectiles.length || 0),
      effects: effects.length,
      damageSplats: damageSplatCount,
      heldSkills: heldSkillIds && heldSkillIds.size ? heldSkillIds.size : 0,
      heldSkillRetries: heldSkillRetryAt && typeof heldSkillRetryAt === 'object' ? Object.keys(heldSkillRetryAt).length : 0,
      drawnEffects: Number(visuals.worldEffectsDrawn || 0),
      culledEffects: Number(visuals.worldEffectsSkippedOffscreen || 0) + Number(visuals.worldEffectsSkippedBudget || 0),
      drawnDamageSplats: Number(visuals.damageSplatsDrawn || 0),
      culledDamageSplats: Number(visuals.damageSplatsSkippedOffscreen || 0) + Number(visuals.damageSplatsSkippedBudget || 0),
      effectDrawBudget: Number(visuals.effectBudget || 0),
      damageSplatDrawBudget: Number(visuals.damageSplatBudget || 0),
      damageNumberMode: String(visuals.damageNumbers || settings.damageNumberMode),
      visualQuality: String(visuals.visualQualityLevel || settings.visualQuality || 'normal'),
      reducedEffects: !!visuals.reducedEffects,
      simplifiedDamageSplats: !!visuals.simplifiedDamageSplats,
      lootDrops: Number(settings.lootDrops || 0),
      skillObjects: Array.isArray(player.activeSkillObjects) ? player.activeSkillObjects.length : 0,
      openWindows: Number(overlay.openWindows || 0),
      canvasHitRegions: Number(overlay.canvasHitRegions || 0),
      commandOpen: !!overlay.commandOpen,
      openPanelIds: String(overlay.openPanelIds || ''),
      hoverPointerMoves: Number(overlay.hoverPointerMoves || 0),
      hoverTargetChanges: Number(overlay.hoverTargetChanges || 0),
      hoverForcedDraws: Number(overlay.hoverForcedDraws || 0),
      hoverCoalescedSkips: Number(overlay.hoverCoalescedSkips || 0),
      activeHoverType: String(overlay.activeHoverType || ''),
      activeHoverKey: String(overlay.activeHoverKey || ''),
      canvasDrawRequests: Number(overlay.canvasDrawRequests || 0),
      canvasImmediateDraws: Number(overlay.canvasImmediateDraws || 0),
      canvasDeferredDraws: Number(overlay.canvasDeferredDraws || 0),
      canvasSkippedRunningDraws: Number(overlay.canvasSkippedRunningDraws || 0),
      overlayCacheHits: Number(overlay.overlayCacheHits || 0),
      overlayCacheMisses: Number(overlay.overlayCacheMisses || 0),
      panelCacheHits: Number(overlay.panelCacheHits || 0),
      panelCacheMisses: Number(overlay.panelCacheMisses || 0),
      panelCachePrewarmQueued: Number(overlay.panelCachePrewarmQueued || 0),
      panelCachePrewarmCompleted: Number(overlay.panelCachePrewarmCompleted || 0),
      panelCachePrewarmSkipped: Number(overlay.panelCachePrewarmSkipped || 0),
      windowDragMoves: Number(overlay.windowDragMoves || 0),
      activeDragPanel: String(overlay.activeDragPanel || '')
    };
  }

  function clonePlainValue(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function averagePerformanceValue(values) {
    const list = (values || []).filter((value) => Number.isFinite(Number(value)));
    if (!list.length) return 0;
    return list.reduce((sum, value) => sum + Number(value), 0) / list.length;
  }

  function percentilePerformanceValue(values, percentile) {
    const list = (values || [])
      .map((value) => Number(value))
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    if (!list.length) return 0;
    const index = clamp(Math.ceil(list.length * percentile) - 1, 0, list.length - 1);
    return list[index];
  }

  function summarizePerformancePhases(samples, group) {
    const keys = new Set();
    (samples || []).forEach((sample) => {
      Object.keys(sample && sample[group] || {}).forEach((key) => keys.add(key));
    });
    return Array.from(keys).map((name) => {
      const values = (samples || []).map((sample) => Number(sample && sample[group] && sample[group][name]) || 0);
      return {
        name,
        averageMs: averagePerformanceValue(values),
        maxMs: Math.max(0, ...values),
        lastMs: values.length ? values[values.length - 1] : 0
      };
    }).sort((a, b) => b.averageMs - a.averageMs || b.maxMs - a.maxMs || a.name.localeCompare(b.name));
  }

  function getPerformanceWindowSamples(samples, options) {
    const settings = options || {};
    const sampleLimit = Math.max(1, Math.floor(Number(settings.sampleLimit || PERFORMANCE_DEBUG_SAMPLE_LIMIT) || PERFORMANCE_DEBUG_SAMPLE_LIMIT));
    const windowMs = Math.max(0, Number(settings.windowMs == null ? PERFORMANCE_DEBUG_WINDOW_MS : settings.windowMs) || 0);
    const list = (samples || []).filter(Boolean);
    const newest = list.length ? Number(list[list.length - 1].timestamp || 0) : 0;
    if (!newest || !windowMs) return list.slice(-sampleLimit);
    const cutoff = newest - windowMs;
    return list.filter((sample) => Number(sample.timestamp || 0) >= cutoff).slice(-sampleLimit);
  }

  function summarizePerformanceCounts(samples, key) {
    const values = (samples || [])
      .map((sample) => sample && sample.counts ? Number(sample.counts[key]) : 0)
      .filter(Number.isFinite);
    return {
      average: averagePerformanceValue(values),
      max: Math.max(0, ...values),
      last: values.length ? values[values.length - 1] : 0
    };
  }

  function getPerformanceFrameTopPhase(sample) {
    const phases = [];
    ['update', 'draw'].forEach((group) => {
      const phaseMap = sample && sample[group] || {};
      Object.keys(phaseMap).forEach((name) => {
        if (name === 'updateTotal' || name === 'drawTotal') return;
        phases.push({ group, name, ms: Number(phaseMap[name]) || 0 });
      });
    });
    phases.sort((a, b) => b.ms - a.ms || a.name.localeCompare(b.name));
    return phases[0] || null;
  }

  function createDisabledPerformanceDebugSnapshot(samples, options) {
    const settings = options || {};
    const sampleList = Array.isArray(samples) ? samples : [];
    const last = sampleList[sampleList.length - 1] || null;
    return {
      mode: 'off',
      enabled: false,
      summaryOnly: true,
      sampleCount: sampleList.length,
      sampleWindow: 0,
      sampleWindowMs: settings.windowMs || PERFORMANCE_DEBUG_WINDOW_MS,
      observedWindowMs: 0,
      currentFps: last && last.frameMs > 0 ? 1000 / last.frameMs : 0,
      averageFps: 0,
      frameMs: last ? Number(last.frameMs || 0) : 0,
      averageFrameMs: 0,
      slowFrameThresholdMs: settings.slowFrameMs || PERFORMANCE_DEBUG_SLOW_FRAME_MS,
      updatePhases: [],
      drawPhases: [],
      snapshotPhases: [],
      mapGeometryPhases: [],
      topPhases: [],
      suspectedBottleneck: null,
      recentUiActions: [],
      counts: {},
      countSummary: {},
      spikeFrames: [],
      recentFrames: []
    };
  }

  function createPerformanceDebugSnapshot(samples, options) {
    const settings = options || {};
    const sampleList = Array.isArray(samples) ? samples.slice() : [];
    const recent = getPerformanceWindowSamples(sampleList, {
      windowMs: settings.windowMs == null ? PERFORMANCE_DEBUG_WINDOW_MS : settings.windowMs,
      sampleLimit: settings.sampleLimit || PERFORMANCE_DEBUG_SAMPLE_LIMIT
    });
    const frameMsValues = recent.map((sample) => Number(sample.frameMs) || 0).filter((value) => value > 0);
    const deltaMsValues = recent.map((sample) => Number(sample.deltaMs) || 0).filter((value) => value > 0);
    const averageFrameMs = averagePerformanceValue(frameMsValues);
    const last = sampleList[sampleList.length - 1] || null;
    const firstRecent = recent[0] || null;
    const observedWindowMs = firstRecent && last
      ? Math.max(0, Number(last.timestamp || 0) - Number(firstRecent.timestamp || 0))
      : 0;
    const currentFps = last && last.frameMs > 0 ? 1000 / last.frameMs : 0;
    const averageFps = averageFrameMs > 0 ? 1000 / averageFrameMs : 0;
    const updatePhases = summarizePerformancePhases(recent, 'update');
    const drawPhases = summarizePerformancePhases(recent, 'draw');
    const snapshotPhases = drawPhases.filter((phase) => String(phase.name || '').startsWith('overlaySnapshot:'));
    const mapGeometryPhases = drawPhases.filter((phase) => String(phase.name || '').startsWith('mapGeometry:'));
    const unaccountedMsValues = recent.map((sample) => Math.max(0,
      Number(sample.frameMs || 0) -
      Number(sample.update && sample.update.updateTotal || 0) -
      Number(sample.draw && sample.draw.drawTotal || 0)));
    const updateMsValues = recent.map((sample) => Number(sample.update && sample.update.updateTotal || 0));
    const drawMsValues = recent.map((sample) => Number(sample.draw && sample.draw.drawTotal || 0));
    const subsystemPhases = updatePhases
      .filter((phase) => phase.name !== 'updateTotal')
      .map((phase) => Object.assign({ group: 'update' }, phase))
      .concat(drawPhases
        .filter((phase) => phase.name !== 'drawTotal')
        .map((phase) => Object.assign({ group: 'draw' }, phase)))
      .sort((a, b) => b.averageMs - a.averageMs || b.maxMs - a.maxMs || a.name.localeCompare(b.name));
    const reportFrameLimit = Math.max(0, Math.floor(Number(settings.reportFrameLimit || PERFORMANCE_DEBUG_REPORT_FRAME_LIMIT) || 0));
    return {
      mode: settings.mode || 'off',
      enabled: settings.mode !== 'off',
      sampleCount: sampleList.length,
      sampleWindow: recent.length,
      sampleWindowMs: settings.windowMs || PERFORMANCE_DEBUG_WINDOW_MS,
      observedWindowMs,
      currentFps,
      averageFps,
      frameMs: last ? Number(last.frameMs || 0) : 0,
      averageFrameMs,
      minFrameMs: frameMsValues.length ? Math.min(...frameMsValues) : 0,
      p50FrameMs: percentilePerformanceValue(frameMsValues, 0.5),
      p95FrameMs: percentilePerformanceValue(frameMsValues, 0.95),
      maxFrameMs: Math.max(0, ...frameMsValues),
      averageDeltaMs: averagePerformanceValue(deltaMsValues),
      p95DeltaMs: percentilePerformanceValue(deltaMsValues, 0.95),
      maxDeltaMs: Math.max(0, ...deltaMsValues),
      slowFrameThresholdMs: settings.slowFrameMs || PERFORMANCE_DEBUG_SLOW_FRAME_MS,
      over16FrameCount: frameMsValues.filter((value) => value >= 16.67).length,
      slowFrameCount: frameMsValues.filter((value) => value >= (settings.slowFrameMs || PERFORMANCE_DEBUG_SLOW_FRAME_MS)).length,
      over50FrameCount: frameMsValues.filter((value) => value >= 50).length,
      over100FrameCount: frameMsValues.filter((value) => value >= 100).length,
      updateMs: averagePerformanceValue(updateMsValues),
      p95UpdateMs: percentilePerformanceValue(updateMsValues, 0.95),
      maxUpdateMs: Math.max(0, ...updateMsValues),
      drawMs: averagePerformanceValue(drawMsValues),
      p95DrawMs: percentilePerformanceValue(drawMsValues, 0.95),
      maxDrawMs: Math.max(0, ...drawMsValues),
      overlayMs: averagePerformanceValue(recent.map((sample) => {
        const draw = sample.draw || {};
        return Number(draw.overlaySnapshot || 0) + Number(draw.overlayDraw || 0);
      })),
      unaccountedMs: averagePerformanceValue(unaccountedMsValues),
      p95UnaccountedMs: percentilePerformanceValue(unaccountedMsValues, 0.95),
      maxUnaccountedMs: Math.max(0, ...unaccountedMsValues),
      updatePhases,
      drawPhases,
      snapshotPhases,
      mapGeometryPhases,
      topPhases: subsystemPhases.slice(0, 8),
      suspectedBottleneck: subsystemPhases[0] || null,
      recentUiActions: (settings.uiActions || []).slice(0, 8).map((entry) => clonePlainValue(entry)),
      counts: last && last.counts ? Object.assign({}, last.counts) : Object.assign({}, settings.counts || {}),
      countSummary: {
        enemies: summarizePerformanceCounts(recent, 'enemies'),
        projectiles: summarizePerformanceCounts(recent, 'projectiles'),
        effects: summarizePerformanceCounts(recent, 'effects'),
        heldSkills: summarizePerformanceCounts(recent, 'heldSkills'),
        drawnEffects: summarizePerformanceCounts(recent, 'drawnEffects'),
        culledEffects: summarizePerformanceCounts(recent, 'culledEffects'),
        drawnDamageSplats: summarizePerformanceCounts(recent, 'drawnDamageSplats'),
        culledDamageSplats: summarizePerformanceCounts(recent, 'culledDamageSplats'),
        lootDrops: summarizePerformanceCounts(recent, 'lootDrops'),
        skillObjects: summarizePerformanceCounts(recent, 'skillObjects'),
        openWindows: summarizePerformanceCounts(recent, 'openWindows'),
        canvasHitRegions: summarizePerformanceCounts(recent, 'canvasHitRegions'),
        canvasDrawRequests: summarizePerformanceCounts(recent, 'canvasDrawRequests'),
        hoverPointerMoves: summarizePerformanceCounts(recent, 'hoverPointerMoves'),
        windowDragMoves: summarizePerformanceCounts(recent, 'windowDragMoves')
      },
      spikeFrames: recent.slice()
        .sort((a, b) => Number(b.frameMs || 0) - Number(a.frameMs || 0))
        .slice(0, 10)
        .map((sample) => {
          const topPhase = getPerformanceFrameTopPhase(sample);
          return {
            ageMs: last ? Math.max(0, Number(last.timestamp || 0) - Number(sample.timestamp || 0)) : 0,
            frameMs: Number(sample.frameMs || 0),
            deltaMs: Number(sample.deltaMs || 0),
            updateMs: Number(sample.update && sample.update.updateTotal || 0),
            drawMs: Number(sample.draw && sample.draw.drawTotal || 0),
            overlayMs: Number(sample.draw && (Number(sample.draw.overlaySnapshot || 0) + Number(sample.draw.overlayDraw || 0)) || 0),
            unaccountedMs: Math.max(0,
              Number(sample.frameMs || 0) -
              Number(sample.update && sample.update.updateTotal || 0) -
              Number(sample.draw && sample.draw.drawTotal || 0)),
            counts: Object.assign({}, sample.counts || {}),
            topPhase
          };
        }),
      recentFrames: sampleList.slice(-reportFrameLimit).map((sample) => ({
        frameMs: Number(sample.frameMs || 0),
        updateMs: Number(sample.update && sample.update.updateTotal || 0),
        drawMs: Number(sample.draw && sample.draw.drawTotal || 0),
        overlayMs: Number(sample.draw && (Number(sample.draw.overlaySnapshot || 0) + Number(sample.draw.overlayDraw || 0)) || 0),
        unaccountedMs: Math.max(0,
          Number(sample.frameMs || 0) -
          Number(sample.update && sample.update.updateTotal || 0) -
          Number(sample.draw && sample.draw.drawTotal || 0)),
        counts: Object.assign({}, sample.counts || {})
      }))
    };
  }

  function createPerformanceDebugReport(debug, options) {
    const settings = options || {};
    const snapshot = debug || {};
    const counts = snapshot.counts || {};
    const countSummary = snapshot.countSummary || {};
    const player = settings.player || {};
    const adminSettings = settings.adminSettings || {};
    const formatMs = typeof settings.formatPerformanceMs === 'function'
      ? settings.formatPerformanceMs
      : (value) => `${(Number(value) || 0).toFixed(2)}ms`;
    const formatFps = typeof settings.formatPerformanceFps === 'function'
      ? settings.formatPerformanceFps
      : (value) => `${Math.round(Number(value) || 0)} fps`;
    const normalizeAdminRate = typeof settings.normalizeAdminRate === 'function'
      ? settings.normalizeAdminRate
      : (value) => Math.max(1, Math.round(Number(value) || 1));
    const userAgent = settings.userAgent || (typeof navigator !== 'undefined' && navigator.userAgent ? navigator.userAgent : 'unknown');
    const canvasWidth = settings.canvasWidth;
    const canvasHeight = settings.canvasHeight;
    const phaseLine = (phase) => `${phase.group || ''}${phase.group ? ':' : ''}${phase.name} avg ${formatMs(phase.averageMs)} max ${formatMs(phase.maxMs)}`.trim();
    const topPhaseLine = (phase) => phase ? `${phase.group}:${phase.name} ${formatMs(phase.ms)}` : 'none';
    const countLine = (label, summary) => `${label} avg ${Number(summary && summary.average || 0).toFixed(1)} max ${Number(summary && summary.max || 0)}`;
    const observedSeconds = Number(snapshot.observedWindowMs || 0) / 1000;
    const configuredSeconds = Number(snapshot.sampleWindowMs || settings.windowMs || PERFORMANCE_DEBUG_WINDOW_MS) / 1000;
    const topPhases = snapshot.topPhases || [];
    const snapshotPhases = snapshot.snapshotPhases || [];
    const mapGeometryPhases = snapshot.mapGeometryPhases || [];
    const updatePhases = snapshot.updatePhases || [];
    const drawPhases = snapshot.drawPhases || [];
    const spikeFrames = snapshot.spikeFrames || [];
    const recentFrames = snapshot.recentFrames || [];
    const lines = [
      'Project Starfall Performance Debug',
      `Captured: ${new Date().toISOString()}`,
      `User agent: ${userAgent}`,
      `Mode: ${snapshot.mode}`,
      `Map: ${settings.mapId} | Class: ${player.advancedClassId || player.classId || 'none'} | Level: ${Number(player.level || 1)}`,
      `Canvas: ${canvasWidth}x${canvasHeight} | World width: ${counts.worldWidth || 0}`,
      `Renderer: ${counts.renderBackend || 'Canvas 2D'} | Status: ${counts.renderBackendStatus || 'canvas'}`,
      `Admin rates: XP ${normalizeAdminRate(adminSettings.xpRate)}x, Drop ${normalizeAdminRate(adminSettings.dropRate)}x`,
      `Visual settings: Damage numbers ${counts.damageNumberMode || 'normal'}, Reduced FX ${counts.reducedEffects ? 'on' : 'off'}, drawn splats ${counts.drawnDamageSplats || 0}/${counts.damageSplats || 0}, culled splats ${counts.culledDamageSplats || 0}, drawn effects ${counts.drawnEffects || 0}/${counts.effects || 0}, culled effects ${counts.culledEffects || 0}`,
      `Sample window: last ${configuredSeconds.toFixed(1)}s, observed ${observedSeconds.toFixed(2)}s, frames ${snapshot.sampleWindow}/${snapshot.sampleCount}`,
      `FPS: current ${formatFps(snapshot.currentFps)}, average ${formatFps(snapshot.averageFps)}, p95 ${formatMs(snapshot.p95FrameMs)}, max ${formatMs(snapshot.maxFrameMs)}, slow frames ${snapshot.slowFrameCount}/${snapshot.sampleWindow}`,
      `Frame pacing: target ${counts.frameRateLimit ? `${counts.frameRateLimit} fps` : 'display'}, skipped rAF ${counts.skippedRafFrames || 0}, min ${formatMs(snapshot.minFrameMs)}, avg ${formatMs(snapshot.averageFrameMs)}, p50 ${formatMs(snapshot.p50FrameMs)}, p95 ${formatMs(snapshot.p95FrameMs)}, max ${formatMs(snapshot.maxFrameMs)}, >=16.7ms ${snapshot.over16FrameCount || 0}, >=33.3ms ${snapshot.slowFrameCount || 0}, >=50ms ${snapshot.over50FrameCount || 0}, >=100ms ${snapshot.over100FrameCount || 0}`,
      `Simulation delta: avg ${formatMs(snapshot.averageDeltaMs)}, p95 ${formatMs(snapshot.p95DeltaMs)}, max ${formatMs(snapshot.maxDeltaMs)}`,
      `Totals: update avg ${formatMs(snapshot.updateMs)} p95 ${formatMs(snapshot.p95UpdateMs)} max ${formatMs(snapshot.maxUpdateMs)}, draw avg ${formatMs(snapshot.drawMs)} p95 ${formatMs(snapshot.p95DrawMs)} max ${formatMs(snapshot.maxDrawMs)}, overlay ${formatMs(snapshot.overlayMs)}, unaccounted avg ${formatMs(snapshot.unaccountedMs)} p95 ${formatMs(snapshot.p95UnaccountedMs)} max ${formatMs(snapshot.maxUnaccountedMs)}`,
      `Main-thread gap: avg ${formatMs(snapshot.unaccountedMs)} p95 ${formatMs(snapshot.p95UnaccountedMs)} max ${formatMs(snapshot.maxUnaccountedMs)} outside measured update/draw, usually browser, GC, or UI work`,
      `Counts: enemies ${counts.enemies}/${counts.enemyActors}, projectiles ${counts.projectiles}, effects ${counts.effects}, damage splats ${counts.damageSplats}, held skills ${counts.heldSkills || 0}, held retries ${counts.heldSkillRetries || 0}, loot ${counts.lootDrops}, skill objects ${counts.skillObjects}, windows ${counts.openWindows}, hit regions ${counts.canvasHitRegions}`,
      `Peak counts: ${countLine('enemies', countSummary.enemies)}, ${countLine('projectiles', countSummary.projectiles)}, ${countLine('effects', countSummary.effects)}, ${countLine('held skills', countSummary.heldSkills)}, ${countLine('drawn splats', countSummary.drawnDamageSplats)}, ${countLine('culled splats', countSummary.culledDamageSplats)}, ${countLine('loot', countSummary.lootDrops)}, ${countLine('hit regions', countSummary.canvasHitRegions)}, ${countLine('draw requests', countSummary.canvasDrawRequests)}`,
      `Hover: moves ${counts.hoverPointerMoves || 0}, target changes ${counts.hoverTargetChanges || 0}, forced draws ${counts.hoverForcedDraws || 0}, skipped draws ${counts.hoverCoalescedSkips || 0}, active ${counts.activeHoverType || 'none'}, panels ${counts.openPanelIds || 'none'}`,
      `Canvas draw requests: total ${counts.canvasDrawRequests || 0}, immediate ${counts.canvasImmediateDraws || 0}, deferred ${counts.canvasDeferredDraws || 0}, skipped while running ${counts.canvasSkippedRunningDraws || 0}, window drag moves ${counts.windowDragMoves || 0}, active drag ${counts.activeDragPanel || 'none'}`,
      `Overlay cache: hits ${counts.overlayCacheHits || 0}, misses ${counts.overlayCacheMisses || 0}`,
      `Panel cache: hits ${counts.panelCacheHits || 0}, misses ${counts.panelCacheMisses || 0}, prewarm queued ${counts.panelCachePrewarmQueued || 0}, completed ${counts.panelCachePrewarmCompleted || 0}, skipped ${counts.panelCachePrewarmSkipped || 0}`,
      String(counts.openPanelIds || '').includes('admin') ? 'Measurement note: admin panel is open and may add overlay/UI cost.' : 'Measurement note: admin panel not reported open.',
      `Likely bottleneck: ${snapshot.suspectedBottleneck ? phaseLine(snapshot.suspectedBottleneck) : 'not enough samples'}`,
      '',
      'Top phases:',
      ...(topPhases.length ? topPhases.map(phaseLine) : ['not enough samples']),
      '',
      'Snapshot phases:',
      ...(snapshotPhases.length ? snapshotPhases.slice(0, 12).map((phase) => phaseLine(Object.assign({ group: 'draw' }, phase))) : ['not enough samples']),
      '',
      'Map geometry phases:',
      ...(mapGeometryPhases.length ? mapGeometryPhases.slice(0, 12).map((phase) => phaseLine(Object.assign({ group: 'draw' }, phase))) : ['not enough samples']),
      '',
      'Update phases:',
      ...(updatePhases.filter((phase) => phase.name !== 'updateTotal').slice(0, 14).map((phase) => phaseLine(Object.assign({ group: 'update' }, phase))) || []),
      '',
      'Draw phases:',
      ...(drawPhases.filter((phase) => phase.name !== 'drawTotal').slice(0, 14).map((phase) => phaseLine(Object.assign({ group: 'draw' }, phase))) || []),
      '',
      `Largest frame spikes (${spikeFrames.length}):`,
      ...(spikeFrames.length ? spikeFrames.map((frame, index) => `${index + 1}. ${formatMs(frame.frameMs)} age ${formatMs(frame.ageMs)} delta ${formatMs(frame.deltaMs)} update ${formatMs(frame.updateMs)} draw ${formatMs(frame.drawMs)} overlay ${formatMs(frame.overlayMs)} unaccounted ${formatMs(frame.unaccountedMs)} top ${topPhaseLine(frame.topPhase)} enemies ${frame.counts.enemies || 0} held ${frame.counts.heldSkills || 0} loot ${frame.counts.lootDrops || 0}`) : ['not enough samples']),
      '',
      `Recent frames (${recentFrames.length}):`,
      ...recentFrames.map((frame, index) => `${index + 1}. frame ${formatMs(frame.frameMs)} update ${formatMs(frame.updateMs)} draw ${formatMs(frame.drawMs)} overlay ${formatMs(frame.overlayMs)} unaccounted ${formatMs(frame.unaccountedMs)} enemies ${frame.counts.enemies || 0} held ${frame.counts.heldSkills || 0} loot ${frame.counts.lootDrops || 0}`)
    ];
    return lines.join('\n');
  }

  function getBenchmarkSkillDefinition(skillId, options) {
    const settings = options || {};
    if (typeof settings.getSkillDefinitionById === 'function') return settings.getSkillDefinitionById(skillId);
    const data = settings.data || global.ProjectStarfallData || {};
    return (data.SKILLS || []).find((skill) => skill && skill.id === skillId) || null;
  }

  function getPerformanceBenchmarkPlatform(runtime, options) {
    const settings = options || {};
    const sourceRuntime = runtime || settings.runtime || {};
    const platforms = (sourceRuntime.platforms || []).filter((platform) => platform && platform.w > 120);
    if (!platforms.length) {
      return {
        x: 0,
        y: settings.playfieldHeight,
        w: sourceRuntime.worldWidth || settings.playfieldWidth,
        h: settings.solidPlatformHeight,
        index: 0,
        id: 'benchmark_ground'
      };
    }
    return platforms.find((platform) => platform.index === 0) || platforms[0];
  }

  function getPerformanceBenchmarkEnemyIds(options) {
    const settings = options || {};
    const data = settings.data || {};
    const enemies = data.ENEMIES || [];
    const enemyIds = Array.isArray(settings.enemyIds) ? settings.enemyIds : PERFORMANCE_BENCHMARK_ENEMY_IDS;
    const getById = typeof settings.getById === 'function'
      ? settings.getById
      : (items, id) => (items || []).find((item) => item && item.id === id) || null;
    const dedicatedIds = enemyIds.filter((enemyId) => {
      const enemyData = getById(enemies, enemyId);
      return enemyData && enemyData.behavior !== 'boss';
    });
    const fallbackIds = enemies
      .filter((enemyData) => enemyData && enemyData.id && enemyData.behavior !== 'boss')
      .map((enemyData) => enemyData.id)
      .slice(0, 8);
    return (dedicatedIds.length ? dedicatedIds : fallbackIds).filter(Boolean);
  }

  function createPerformanceBenchmarkScenePlan(options) {
    const settings = options || {};
    const platform = settings.platform || getPerformanceBenchmarkPlatform(settings.runtime, settings);
    const player = settings.player || {};
    const stats = settings.stats || {};
    const character = settings.character || PERFORMANCE_BENCHMARK_CHARACTER;
    const getPlatformSurfaceY = typeof settings.getPlatformSurfaceY === 'function'
      ? settings.getPlatformSurfaceY
      : (sourcePlatform) => Number(sourcePlatform && sourcePlatform.y || 0);
    const viewportWidth = settings.viewportWidth;
    const preferredAnchorX = 1760;
    const anchorX = clamp(
      preferredAnchorX,
      platform.x + Math.min(140, platform.w / 2),
      platform.x + Math.max(Math.min(platform.w - 140, platform.w / 2), 80)
    );
    const playerWidth = Number(player.w || 48);
    const playerHeight = Number(player.h || 54);
    const centeredX = clamp(anchorX - playerWidth / 2, platform.x + 20, platform.x + platform.w - playerWidth - 20);
    const playerSurfaceY = getPlatformSurfaceY(platform, centeredX + playerWidth / 2);
    return {
      platform,
      anchorX,
      centeredX,
      playerSurfaceY,
      playerState: {
        x: centeredX,
        y: playerSurfaceY - playerHeight,
        vx: 0,
        vy: 0,
        grounded: true,
        groundedPlatformId: platform.id || '',
        groundedPlatformIndex: platform.index || 0,
        climbing: false,
        climbableId: '',
        activeStation: '',
        activePortalId: '',
        activeQuestNpcId: '',
        activeShopVendorId: '',
        hp: stats.maxHp,
        mp: stats.maxMp,
        resource: stats.secondaryResourceMax,
        combatLockUntil: 0,
        movementLockUntil: 0
      },
      scene: {
        mapId: settings.mapId || PERFORMANCE_BENCHMARK_MAP_ID,
        mapName: settings.mapName || PERFORMANCE_BENCHMARK_MAP_NAME,
        characterId: character.advancedClassId || character.classId,
        characterName: character.name,
        characterLevel: character.level,
        seed: settings.sceneSeed || PERFORMANCE_BENCHMARK_SCENE_SEED,
        anchorX,
        anchorY: playerSurfaceY - playerHeight / 2,
        platformIndex: platform.index || 0,
        platformY: playerSurfaceY,
        viewportWidth,
        enemyIds: Array.isArray(settings.enemyIds) ? settings.enemyIds : []
      }
    };
  }

  function createPerformanceBenchmarkProjectile(benchmark, index, options) {
    const settings = options || {};
    const runtime = settings.runtime || {};
    const scene = benchmark && benchmark.scene || {};
    const viewWidth = settings.viewWidth || settings.playfieldWidth || 1280;
    const playfieldWidth = settings.playfieldWidth || viewWidth;
    const playfieldHeight = settings.playfieldHeight || 640;
    const worldWidth = Number(runtime.worldWidth || playfieldWidth);
    const platformY = Number(scene.platformY || playfieldHeight);
    const anchorX = Number(scene.anchorX || 520);
    const angle = seededUnit(scene.seed, `projectileAngle:${index}`) * Math.PI * 2;
    const radius = 180 + seededUnit(scene.seed, `projectileRadius:${index}`) * Math.max(260, Number(scene.viewportWidth || viewWidth) * 0.42);
    const startX = clamp(anchorX + Math.cos(angle) * radius, 24, Math.max(48, worldWidth - 48));
    const startY = clamp(platformY - 210 + Math.sin(angle) * 110, 96, Math.max(120, platformY - 24));
    const targetX = anchorX + (seededUnit(scene.seed, `projectileTargetX:${index}`) - 0.5) * 180;
    const targetY = platformY - 120 + (seededUnit(scene.seed, `projectileTargetY:${index}`) - 0.5) * 90;
    const dx = targetX - startX;
    const dy = targetY - startY;
    const distance = Math.hypot(dx, dy) || 1;
    const speed = 180 + seededUnit(scene.seed, `projectileSpeed:${index}`) * 160;
    const type = index % 3 === 0 ? 'knife' : index % 3 === 1 ? 'firebolt' : 'thorn';
    return {
      owner: 'enemy',
      type,
      x: startX,
      y: startY,
      vx: dx / distance * speed,
      vy: dy / distance * speed,
      w: type === 'knife' ? 22 : 18,
      h: type === 'knife' ? 10 : 12,
      damage: 1,
      ttl: 1.8 + seededUnit(scene.seed, `projectileTtl:${index}`) * 1.7,
      totalTtl: 3.5,
      pierce: 0,
      benchmark: true
    };
  }

  function createPerformanceBenchmarkEffect(benchmark, index, options) {
    const settings = options || {};
    const runtime = settings.runtime || {};
    const scene = benchmark && benchmark.scene || {};
    const viewWidth = settings.viewWidth || settings.playfieldWidth || 1280;
    const playfieldWidth = settings.playfieldWidth || viewWidth;
    const playfieldHeight = settings.playfieldHeight || 640;
    const worldWidth = Number(runtime.worldWidth || playfieldWidth);
    const platformY = Number(scene.platformY || playfieldHeight);
    const anchorX = Number(scene.anchorX || 520);
    const x = clamp(anchorX + (seededUnit(scene.seed, `effectX:${index}`) - 0.5) * Math.max(520, Number(scene.viewportWidth || viewWidth) * 0.72), 24, Math.max(48, worldWidth - 48));
    const y = clamp(platformY - 96 + (seededUnit(scene.seed, `effectY:${index}`) - 0.5) * 180, 90, Math.max(120, platformY - 8));
    const colors = ['#7bdff2', '#ffbe55', '#ff7b3a', '#66d79a', '#c794ff'];
    const color = colors[index % colors.length];
    const duration = 1.6 + seededUnit(scene.seed, `effectDuration:${index}`) * 2.2;
    if (index % 5 === 0) {
      return { type: 'chainLine', x, y, x2: x + 80 + seededUnit(scene.seed, `lineX:${index}`) * 180, y2: y - 80 + seededUnit(scene.seed, `lineY:${index}`) * 150, ttl: duration, duration, color, accentColor: '#ffffff', pulseIndex: index % 7, benchmark: true };
    }
    if (index % 5 === 1) {
      return { type: 'field', x, y: platformY - 6, r: 78 + seededUnit(scene.seed, `field:${index}`) * 60, ttl: duration, duration, color, damage: 0, slow: true, targetCap: 0, benchmark: true };
    }
    if (index % 5 === 2) {
      return { type: 'telegraph', x: x - 72, y: platformY - 12, w: 138 + seededUnit(scene.seed, `telegraph:${index}`) * 160, h: 9, ttl: duration, duration, color, benchmark: true };
    }
    if (index % 5 === 3) {
      return { type: 'shockBurst', x, y, w: 50, h: 64, r: 54, ttl: duration, duration, color, accentColor: '#ffffff', pulseIndex: index % 9, benchmark: true };
    }
    return { type: index % 2 ? 'slash' : 'cast', x, y, r: 38 + seededUnit(scene.seed, `action:${index}`) * 34, ttl: duration, duration, color, accentColor: '#ffffff', visualKind: index % 2 ? 'slash' : 'cast', benchmark: true };
  }

  function createPerformanceBenchmarkDamageSplat(benchmark, index, options) {
    const settings = options || {};
    const scene = benchmark && benchmark.scene || {};
    const enemies = Array.isArray(settings.enemies) ? settings.enemies : [];
    const player = settings.player || {};
    const target = enemies.length ? enemies[index % enemies.length] : player;
    const formatInteger = typeof settings.formatIntegerWithCommas === 'function'
      ? settings.formatIntegerWithCommas
      : (value) => String(Math.round(Number(value) || 0));
    const damageSplatDuration = Number(settings.damageSplatDuration || 1.05);
    const duration = damageSplatDuration + seededUnit(scene.seed, `splatDuration:${index}`) * 0.7;
    const lineCount = 3 + index % 4;
    return {
      type: 'damageSplat',
      text: formatInteger(180 + index * 17 % 5200),
      subtext: '',
      x: Number(target.x || 0) + Number(target.w || 40) / 2 + (seededUnit(scene.seed, `splatX:${index}`) - 0.5) * 44,
      y: Number(target.y || 0) - 8 - (index % 5) * 6,
      vx: (seededUnit(scene.seed, `splatVx:${index}`) - 0.5) * 18,
      vy: -76 - seededUnit(scene.seed, `splatVy:${index}`) * 28,
      ttl: duration,
      duration,
      age: seededUnit(scene.seed, `splatAge:${index}`) * 0.18,
      delay: 0,
      lineIndex: index % lineCount,
      lineCount,
      lineGroupId: `benchmark-${Math.floor(index / lineCount)}`,
      stacked: true,
      color: index % 7 === 0 ? '#ffef7a' : '#fff4c7',
      stroke: 'rgba(9, 31, 59, 0.9)',
      burstColor: index % 7 === 0 ? 'rgba(255, 93, 93, 0.62)' : 'rgba(255, 225, 106, 0.5)',
      critical: index % 7 === 0,
      benchmark: true
    };
  }

  function createPerformanceBenchmarkLoadPlan(benchmark, projectiles, effects, fillAll, options) {
    if (!benchmark || !benchmark.active || !benchmark.scene) return null;
    const settings = options || {};
    const target = benchmark.counts || settings.counts || PERFORMANCE_BENCHMARK_COUNTS;
    const projectileLimit = fillAll ? target.projectiles : 24;
    const effectLimit = fillAll ? target.effects : 32;
    const damageSplatLimit = fillAll ? target.damageSplats : 34;
    const projectileList = Array.isArray(projectiles) ? projectiles : [];
    let effectCount = 0;
    let damageSplatCount = 0;
    (effects || []).forEach((effect) => {
      if (!effect) return;
      if (effect.type === 'damageSplat') damageSplatCount += 1;
      else effectCount += 1;
    });
    let projectileAdds = 0;
    for (let added = 0; projectileList.length + added < target.projectiles && added < projectileLimit; added += 1) {
      projectileAdds += 1;
    }
    let effectAdds = 0;
    for (let added = 0; effectCount + added < target.effects && added < effectLimit; added += 1) {
      effectAdds += 1;
    }
    let damageSplatAdds = 0;
    for (let added = 0; damageSplatCount + added < target.damageSplats && added < damageSplatLimit; added += 1) {
      damageSplatAdds += 1;
    }
    return {
      target,
      projectileLimit,
      effectLimit,
      damageSplatLimit,
      projectileCount: projectileList.length,
      effectCount,
      damageSplatCount,
      projectileAdds,
      effectAdds,
      damageSplatAdds
    };
  }

  function createPerformanceBenchmarkEnemySpawnPlan(benchmark, options) {
    const settings = options || {};
    const scene = benchmark && benchmark.scene || {};
    const enemyIds = Array.isArray(settings.enemyIds) ? settings.enemyIds : scene.enemyIds || [];
    if (!enemyIds.length) return { spawns: [] };
    const data = settings.data || {};
    const enemies = data.ENEMIES || [];
    const getById = typeof settings.getById === 'function'
      ? settings.getById
      : (items, id) => (items || []).find((item) => item && item.id === id) || null;
    const getPlatformSurfaceY = typeof settings.getPlatformSurfaceY === 'function'
      ? settings.getPlatformSurfaceY
      : (platform) => Number(platform && platform.y || 0);
    const platforms = (settings.platforms || []).filter((platform) => platform && platform.w > 120);
    const fallbackPlatform = settings.fallbackPlatform || getPerformanceBenchmarkPlatform(settings.runtime, settings);
    const basePlatformIndex = Math.max(0, platforms.findIndex((platform) => platform.index === scene.platformIndex));
    const targetEnemyCount = benchmark && benchmark.counts && benchmark.counts.enemies;
    const playerLevel = settings.playerLevel;
    const spawns = [];
    for (let index = 0; index < targetEnemyCount; index += 1) {
      const platform = platforms[(basePlatformIndex + Math.floor(index / 18)) % platforms.length] || fallbackPlatform;
      const enemyId = enemyIds[index % enemyIds.length];
      const enemyData = getById(enemies, enemyId);
      if (!enemyData || !platform) continue;
      const lane = index % 18;
      const side = lane % 2 ? -1 : 1;
      const spread = 86 + Math.floor(lane / 2) * 42 + Math.floor(index / 18) * 22;
      const jitter = (seededUnit(scene.seed, `enemy:${index}`) - 0.5) * 18;
      const x = clamp(Number(scene.anchorX || 520) + side * spread + jitter, platform.x + 18, platform.x + platform.w - 72);
      const surfaceY = getPlatformSurfaceY(platform, x);
      const nativeLevelMin = Number(enemyData.levelRange && enemyData.levelRange[0] || 1);
      const nativeLevelMax = Number(enemyData.levelRange && enemyData.levelRange[1] || nativeLevelMin);
      const level = clamp(Math.max(nativeLevelMin, Number(playerLevel || nativeLevelMin)), nativeLevelMin, nativeLevelMax);
      spawns.push({
        index,
        enemyId,
        enemyData,
        createOptions: { x, y: surfaceY, platformIndex: platform.index, platformId: platform.id },
        level,
        x,
        surfaceY,
        flyerY: Math.max(150, surfaceY - 180 - (index % 4) * 18),
        vx: side * (10 + index % 5),
        vy: 0,
        facing: side > 0 ? -1 : 1,
        grounded: enemyData.behavior !== 'flyer',
        groundedPlatformId: platform.id || '',
        groundedPlatformIndex: platform.index || 0,
        aggroTargetKind: 'player',
        aggroTargetId: 'player',
        attackCd: 0.15 + seededUnit(scene.seed, `attack:${index}`) * 0.6,
        animation: index % 3 ? 'move' : 'idle'
      });
    }
    return {
      spawns,
      enemyIds,
      basePlatformIndex
    };
  }

  function createPerformanceBenchmarkCharacterState(options) {
    const settings = options || {};
    const createInitialState = typeof settings.createInitialState === 'function' ? settings.createInitialState : null;
    if (!createInitialState) return null;
    const data = settings.data || global.ProjectStarfallData || {};
    const character = settings.character || PERFORMANCE_BENCHMARK_CHARACTER;
    const mapId = settings.mapId || PERFORMANCE_BENCHMARK_MAP_ID;
    const getDefaultCharacterLookId = typeof settings.getDefaultCharacterLookId === 'function'
      ? settings.getDefaultCharacterLookId
      : function getDefaultCharacterLookIdFallback() { return ''; };
    const createAdminSettings = typeof settings.createAdminSettings === 'function'
      ? settings.createAdminSettings
      : function createAdminSettingsFallback(value) { return Object.assign({}, value || {}); };
    const createDefaultRanks = typeof settings.createDefaultRanks === 'function'
      ? settings.createDefaultRanks
      : function createDefaultRanksFallback() { return {}; };
    const getDefaultSkillRank = typeof settings.getDefaultSkillRank === 'function'
      ? settings.getDefaultSkillRank
      : function getDefaultSkillRankFallback(skill) { return Math.max(0, Math.min(1, Number(skill && skill.maxRank || 1))); };
    const state = createInitialState(character.classId, {
      name: character.name,
      lookId: getDefaultCharacterLookId()
    });
    if (!state || typeof state !== 'object') return state;
    state.mapId = mapId;
    state.player = state.player && typeof state.player === 'object' ? state.player : {};
    state.player.advancedClassId = character.advancedClassId;
    state.player.level = character.level;
    state.player.xp = 0;
    state.player.skillPoints = 0;
    state.player.baseSkillPoints = 0;
    state.player.advancedSkillPoints = 0;
    state.player.currency = 0;
    state.inventory = [];
    state.equipment = {};
    state.lootDrops = [];
    state.waveByMap = {};
    state.log = ['Dedicated performance benchmark scene.'];
    state.session = state.session && typeof state.session === 'object' ? state.session : {};
    state.session.selectedPanel = 'admin';
    state.adminSettings = createAdminSettings(Object.assign({}, settings.currentAdminSettings || {}, {
      performanceDebugMode: 'breakdown'
    }));
    state.skills = createDefaultRanks(character.classId);
    (data.SKILLS || []).forEach((skill) => {
      if (!skill || skill.owner !== character.advancedClassId) return;
      state.skills[skill.id] = getDefaultSkillRank(skill);
    });
    const maxedSkillIds = Array.isArray(settings.maxedSkillIds)
      ? settings.maxedSkillIds
      : PERFORMANCE_BENCHMARK_MAXED_SKILL_IDS;
    maxedSkillIds.forEach((skillId) => {
      const skill = getBenchmarkSkillDefinition(skillId, settings);
      if (skill) state.skills[skillId] = Math.max(Number(state.skills[skillId] || 0), Math.max(1, Number(skill.maxRank || 10) || 10));
    });
    return state;
  }

  function createPerformanceBenchmarkResult(benchmark, debug, options) {
    const settings = options || {};
    const scene = benchmark && benchmark.scene || {};
    const counts = debug && debug.counts || {};
    const character = settings.character || PERFORMANCE_BENCHMARK_CHARACTER;
    const topPhases = (debug && debug.topPhases || []).slice(0, 8).map((phase) => ({
      group: phase.group || '',
      name: phase.name || '',
      averageMs: Number(phase.averageMs || 0),
      maxMs: Number(phase.maxMs || 0),
      count: Number(phase.count || 0)
    }));
    return {
      scenarioId: settings.scenarioId || PERFORMANCE_BENCHMARK_SCENARIO_ID,
      capturedAt: new Date().toISOString(),
      userAgent: typeof navigator !== 'undefined' && navigator.userAgent ? navigator.userAgent : 'unknown',
      mapId: scene.mapId || settings.currentMapId || PERFORMANCE_BENCHMARK_MAP_ID,
      mapName: scene.mapName || settings.mapName || PERFORMANCE_BENCHMARK_MAP_NAME,
      characterId: scene.characterId || character.advancedClassId || character.classId,
      characterName: scene.characterName || character.name,
      characterLevel: Number(scene.characterLevel || character.level),
      sceneSeed: scene.seed || settings.sceneSeed || PERFORMANCE_BENCHMARK_SCENE_SEED,
      dedicatedScene: true,
      warmupMs: Number(settings.warmupMs || PERFORMANCE_BENCHMARK_WARMUP_MS),
      sampleMs: Number(settings.sampleMs || PERFORMANCE_BENCHMARK_SAMPLE_MS),
      observedWindowMs: Number(debug && debug.observedWindowMs || 0),
      sampleFrames: Number(debug && debug.sampleWindow || 0),
      averageFps: Number(debug && debug.averageFps || 0),
      currentFps: Number(debug && debug.currentFps || 0),
      averageFrameMs: Number(debug && debug.averageFrameMs || 0),
      p50FrameMs: Number(debug && debug.p50FrameMs || 0),
      p95FrameMs: Number(debug && debug.p95FrameMs || 0),
      maxFrameMs: Number(debug && debug.maxFrameMs || 0),
      slowFrameCount: Number(debug && debug.slowFrameCount || 0),
      over50FrameCount: Number(debug && debug.over50FrameCount || 0),
      updateMs: Number(debug && debug.updateMs || 0),
      p95UpdateMs: Number(debug && debug.p95UpdateMs || 0),
      drawMs: Number(debug && debug.drawMs || 0),
      p95DrawMs: Number(debug && debug.p95DrawMs || 0),
      overlayMs: Number(debug && debug.overlayMs || 0),
      unaccountedMs: Number(debug && debug.unaccountedMs || 0),
      renderer: counts.renderBackend || 'Canvas 2D',
      rendererStatus: counts.renderBackendStatus || 'canvas',
      visualQuality: counts.visualQuality || 'normal',
      counts: {
        enemies: counts.enemies || 0,
        enemyActors: counts.enemyActors || 0,
        projectiles: counts.projectiles || 0,
        effects: counts.effects || 0,
        damageSplats: counts.damageSplats || 0,
        drawnEffects: counts.drawnEffects || 0,
        drawnDamageSplats: counts.drawnDamageSplats || 0,
        culledEffects: counts.culledEffects || 0,
        culledDamageSplats: counts.culledDamageSplats || 0
      },
      targetCounts: Object.assign({}, benchmark && benchmark.counts || PERFORMANCE_BENCHMARK_COUNTS),
      topPhases,
      suspectedBottleneck: topPhases[0] || null
    };
  }

  function createPerformanceBenchmarkReport(result, options) {
    const settings = options || {};
    const character = settings.character || PERFORMANCE_BENCHMARK_CHARACTER;
    const formatPerformanceMs = typeof settings.formatPerformanceMs === 'function'
      ? settings.formatPerformanceMs
      : function formatPerformanceMsFallback(value) { return `${(Number(value) || 0).toFixed(2)}ms`; };
    const formatPerformanceFps = typeof settings.formatPerformanceFps === 'function'
      ? settings.formatPerformanceFps
      : function formatPerformanceFpsFallback(value) { return `${Math.round(Number(value) || 0)} fps`; };
    if (!result) {
      return [
        'Project Starfall Performance Benchmark',
        'No completed benchmark result is available yet.'
      ].join('\n');
    }
    const phaseLine = (phase) => `${phase.group}:${phase.name} avg ${formatPerformanceMs(phase.averageMs)} max ${formatPerformanceMs(phase.maxMs)}`;
    const counts = result.counts || {};
    const targetCounts = result.targetCounts || {};
    const topPhases = Array.isArray(result.topPhases) ? result.topPhases : [];
    const lines = [
      'Project Starfall Performance Benchmark',
      `Scenario: ${result.scenarioId}`,
      `Captured: ${result.capturedAt}`,
      `User agent: ${result.userAgent}`,
      `Scene: ${result.mapName || settings.mapName || PERFORMANCE_BENCHMARK_MAP_NAME} (${result.mapId})`,
      `Character: ${result.characterName || character.name} Lv ${Number(result.characterLevel || character.level)}`,
      `Seed: ${result.sceneSeed || settings.sceneSeed || PERFORMANCE_BENCHMARK_SCENE_SEED}`,
      `Renderer: ${result.renderer} | Status: ${result.rendererStatus} | Visual quality: ${result.visualQuality}`,
      `Window: warmup ${(result.warmupMs / 1000).toFixed(1)}s, sample ${(result.sampleMs / 1000).toFixed(1)}s, observed ${(result.observedWindowMs / 1000).toFixed(2)}s, frames ${result.sampleFrames}`,
      `FPS: average ${formatPerformanceFps(result.averageFps)}, current ${formatPerformanceFps(result.currentFps)}, p95 frame ${formatPerformanceMs(result.p95FrameMs)}, max ${formatPerformanceMs(result.maxFrameMs)}, slow frames ${result.slowFrameCount}`,
      `Frame pacing: avg ${formatPerformanceMs(result.averageFrameMs)}, p50 ${formatPerformanceMs(result.p50FrameMs)}, p95 ${formatPerformanceMs(result.p95FrameMs)}, max ${formatPerformanceMs(result.maxFrameMs)}, >=50ms ${result.over50FrameCount}`,
      `Totals: update avg ${formatPerformanceMs(result.updateMs)} p95 ${formatPerformanceMs(result.p95UpdateMs)}, draw avg ${formatPerformanceMs(result.drawMs)} p95 ${formatPerformanceMs(result.p95DrawMs)}, overlay ${formatPerformanceMs(result.overlayMs)}, unaccounted ${formatPerformanceMs(result.unaccountedMs)}`,
      `Counts: enemies ${counts.enemies}/${counts.enemyActors}, projectiles ${counts.projectiles}, effects ${counts.effects}, damage splats ${counts.damageSplats}, drawn effects ${counts.drawnEffects}, drawn splats ${counts.drawnDamageSplats}`,
      `Target counts: enemies ${targetCounts.enemies}, projectiles ${targetCounts.projectiles}, effects ${targetCounts.effects}, damage splats ${targetCounts.damageSplats}`,
      `Likely bottleneck: ${result.suspectedBottleneck ? phaseLine(result.suspectedBottleneck) : 'not enough samples'}`,
      '',
      'Top phases:',
      ...(topPhases.length ? topPhases.map(phaseLine) : ['not enough samples'])
    ];
    return lines.join('\n');
  }

  function createPerformanceBenchmarkSnapshot(benchmark, result, nowMs, options) {
    const settings = options || {};
    const character = settings.character || PERFORMANCE_BENCHMARK_CHARACTER;
    const scenarioId = settings.scenarioId || PERFORMANCE_BENCHMARK_SCENARIO_ID;
    const mapId = settings.mapId || PERFORMANCE_BENCHMARK_MAP_ID;
    const mapName = settings.mapName || PERFORMANCE_BENCHMARK_MAP_NAME;
    const warmupMs = Number(settings.warmupMs || PERFORMANCE_BENCHMARK_WARMUP_MS);
    const sampleMs = Number(settings.sampleMs || PERFORMANCE_BENCHMARK_SAMPLE_MS);
    if (!benchmark || !benchmark.active) {
      return {
        active: false,
        phase: result ? 'complete' : 'idle',
        scenarioId,
        mapId: result && result.mapId || mapId,
        mapName: result && result.mapName || mapName,
        characterName: result && result.characterName || character.name,
        characterLevel: result && result.characterLevel || character.level,
        warmupMs,
        sampleMs,
        progress: result ? 1 : 0,
        result
      };
    }
    const currentNowMs = Number(nowMs || 0);
    const totalMs = warmupMs + sampleMs;
    const elapsedMs = Math.max(0, currentNowMs - Number(benchmark.startedAtMs || currentNowMs));
    return {
      active: true,
      phase: benchmark.phase || 'warmup',
      scenarioId: benchmark.scenarioId || scenarioId,
      mapId: benchmark.scene && benchmark.scene.mapId || mapId,
      mapName: benchmark.scene && benchmark.scene.mapName || mapName,
      characterName: benchmark.scene && benchmark.scene.characterName || character.name,
      characterLevel: benchmark.scene && benchmark.scene.characterLevel || character.level,
      warmupMs,
      sampleMs,
      elapsedMs,
      remainingMs: Math.max(0, (benchmark.phase === 'sample' ? benchmark.sampleUntilMs : benchmark.warmupUntilMs) - currentNowMs),
      progress: clamp(elapsedMs / Math.max(1, totalMs), 0, 1),
      result: null
    };
  }

  function createPerformanceBenchmarkCompleteState(benchmark, result, options) {
    const settings = options || {};
    return {
      active: false,
      phase: 'complete',
      scenarioId: settings.scenarioId || PERFORMANCE_BENCHMARK_SCENARIO_ID,
      startedAtMs: Number(benchmark && benchmark.startedAtMs || 0),
      warmupMs: Number(settings.warmupMs || PERFORMANCE_BENCHMARK_WARMUP_MS),
      sampleMs: Number(settings.sampleMs || PERFORMANCE_BENCHMARK_SAMPLE_MS),
      progress: 1,
      result
    };
  }

  function createPerformanceBenchmarkStartState(nowMs, captured, options) {
    const settings = options || {};
    const startedAtMs = Number(nowMs || 0);
    const warmupMs = Number(settings.warmupMs || PERFORMANCE_BENCHMARK_WARMUP_MS);
    const sampleMs = Number(settings.sampleMs || PERFORMANCE_BENCHMARK_SAMPLE_MS);
    return {
      active: true,
      phase: 'warmup',
      scenarioId: settings.scenarioId || PERFORMANCE_BENCHMARK_SCENARIO_ID,
      startedAtMs,
      warmupUntilMs: startedAtMs + warmupMs,
      sampleStartedAtMs: 0,
      sampleUntilMs: startedAtMs + warmupMs + sampleMs,
      warmupMs,
      sampleMs,
      counts: Object.assign({}, settings.counts || PERFORMANCE_BENCHMARK_COUNTS),
      captured,
      scene: null,
      projectileCursor: 0,
      effectCursor: 0,
      damageSplatCursor: 0,
      result: null
    };
  }

  function updatePerformanceBenchmarkPhaseState(benchmark, nowMs, options) {
    if (!benchmark || !benchmark.active) {
      return {
        active: false,
        phaseChanged: false,
        complete: false
      };
    }
    const settings = options || {};
    const currentNowMs = Number(nowMs || 0);
    const sampleMs = Object.prototype.hasOwnProperty.call(settings, 'sampleMs')
      ? Number(settings.sampleMs)
      : PERFORMANCE_BENCHMARK_SAMPLE_MS;
    let phaseChanged = false;
    if (benchmark.phase === 'warmup' && currentNowMs >= benchmark.warmupUntilMs) {
      benchmark.phase = 'sample';
      benchmark.sampleStartedAtMs = currentNowMs;
      benchmark.sampleUntilMs = currentNowMs + sampleMs;
      phaseChanged = true;
    }
    return {
      active: true,
      phaseChanged,
      complete: benchmark.phase === 'sample' && currentNowMs >= benchmark.sampleUntilMs
    };
  }

  function capturePerformanceBenchmarkState(options) {
    const settings = options || {};
    const clonePlain = typeof settings.clonePlain === 'function' ? settings.clonePlain : clonePlainValue;
    const createPetRuntime = typeof settings.createPetRuntime === 'function' ? settings.createPetRuntime : function createPetRuntimeFallback() {
      return {};
    };
    const cloneValue = (value, fallback) => value == null ? fallback : clonePlain(value);
    return {
      state: clonePlain(settings.state),
      runtime: cloneValue(settings.runtime, null),
      enemies: clonePlain(settings.enemies || []),
      projectiles: clonePlain(settings.projectiles || []),
      effects: clonePlain(settings.effects || []),
      chainPulses: clonePlain(settings.chainPulses || []),
      petRuntime: cloneValue(settings.petRuntime, createPetRuntime()),
      camera: cloneValue(settings.camera, { x: 0, y: 0 }),
      input: Object.assign({}, settings.input || {}),
      heldSkillIds: Array.from(settings.heldSkillIds || []),
      heldSkillRetryAt: Object.assign({}, settings.heldSkillRetryAt || {}),
      nextHeldLootAt: Number(settings.nextHeldLootAt || 0),
      bossIntroSummary: cloneValue(settings.bossIntroSummary, null),
      bossClearSummary: cloneValue(settings.bossClearSummary, null)
    };
  }

  function createPerformanceBenchmarkRestoreState(captured, options) {
    if (!captured || !captured.state) return null;
    const settings = options || {};
    const clonePlain = typeof settings.clonePlain === 'function' ? settings.clonePlain : clonePlainValue;
    const createMapRuntime = typeof settings.createMapRuntime === 'function' ? settings.createMapRuntime : function createMapRuntimeFallback() {
      return {};
    };
    const createPetRuntime = typeof settings.createPetRuntime === 'function' ? settings.createPetRuntime : function createPetRuntimeFallback() {
      return {};
    };
    const state = clonePlain(captured.state);
    return {
      state,
      runtime: captured.runtime ? clonePlain(captured.runtime) : createMapRuntime(state.mapId, settings.viewport),
      enemies: clonePlain(captured.enemies || []),
      projectiles: clonePlain(captured.projectiles || []),
      effects: clonePlain(captured.effects || []),
      chainPulses: clonePlain(captured.chainPulses || []),
      petRuntime: captured.petRuntime ? clonePlain(captured.petRuntime) : createPetRuntime(),
      camera: captured.camera ? clonePlain(captured.camera) : { x: 0, y: 0 },
      input: Object.assign({ left: false, right: false, up: false, jump: false, down: false, attack: false, loot: false }, captured.input || {}),
      heldSkillIds: new Set(captured.heldSkillIds || []),
      heldSkillRetryAt: Object.assign({}, captured.heldSkillRetryAt || {}),
      nextHeldLootAt: Number(captured.nextHeldLootAt || 0),
      bossIntroSummary: captured.bossIntroSummary ? clonePlain(captured.bossIntroSummary) : null,
      bossClearSummary: captured.bossClearSummary ? clonePlain(captured.bossClearSummary) : null
    };
  }

  const api = {
    PERFORMANCE_DEBUG_MODES,
    PERFORMANCE_DEBUG_WINDOW_MS,
    PERFORMANCE_DEBUG_SAMPLE_LIMIT,
    PERFORMANCE_DEBUG_REPORT_FRAME_LIMIT,
    PERFORMANCE_DEBUG_SLOW_FRAME_MS,
    PERFORMANCE_BENCHMARK_SCENARIO_ID,
    PERFORMANCE_BENCHMARK_MAP_ID,
    PERFORMANCE_BENCHMARK_MAP_NAME,
    PERFORMANCE_BENCHMARK_SCENE_SEED,
    PERFORMANCE_BENCHMARK_CHARACTER,
    PERFORMANCE_BENCHMARK_ENEMY_IDS,
    PERFORMANCE_BENCHMARK_WARMUP_MS,
    PERFORMANCE_BENCHMARK_SAMPLE_MS,
    PERFORMANCE_BENCHMARK_COUNTS,
    PERFORMANCE_BENCHMARK_MAXED_SKILL_IDS,
    PERFORMANCE_DEBUG_COUNTS_CACHE_MS,
    normalizePerformanceDebugMode,
    createPerformanceDebugRuntime,
    createPerformanceFrameState,
    recordPerformancePhaseState,
    finishPerformanceFrameState,
    recordPerformanceUiActionState,
    recordPerformanceOverlayStatsState,
    resetPerformanceDebugRuntimeState,
    createPerformanceDebugCountsState,
    averagePerformanceValue,
    percentilePerformanceValue,
    summarizePerformancePhases,
    getPerformanceWindowSamples,
    summarizePerformanceCounts,
    getPerformanceFrameTopPhase,
    createDisabledPerformanceDebugSnapshot,
    createPerformanceDebugSnapshot,
    createPerformanceDebugReport,
    getPerformanceBenchmarkPlatform,
    getPerformanceBenchmarkEnemyIds,
    createPerformanceBenchmarkScenePlan,
    createPerformanceBenchmarkProjectile,
    createPerformanceBenchmarkEffect,
    createPerformanceBenchmarkDamageSplat,
    createPerformanceBenchmarkLoadPlan,
    createPerformanceBenchmarkEnemySpawnPlan,
    createPerformanceBenchmarkCharacterState,
    createPerformanceBenchmarkResult,
    createPerformanceBenchmarkReport,
    createPerformanceBenchmarkSnapshot,
    createPerformanceBenchmarkCompleteState,
    createPerformanceBenchmarkStartState,
    updatePerformanceBenchmarkPhaseState,
    capturePerformanceBenchmarkState,
    createPerformanceBenchmarkRestoreState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.performance = Object.assign({}, modules.performance || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
