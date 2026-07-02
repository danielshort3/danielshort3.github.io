(function initProjectStarfallEngineGeneratedMaps(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const DEFAULT_PERFORMANCE_BENCHMARK_MAP_ID = 'performanceBenchmarkArena';
  const DEFAULT_PERFORMANCE_BENCHMARK_MAP_NAME = 'Benchmark Arena';

  function getGeneratedMapData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createClassTrialInstanceMap(trial, options) {
    const data = getGeneratedMapData(options);
    const id = normalizeId(trial && trial.id) || 'class_trial';
    const enemyIds = [];
    (trial && trial.objectives || []).forEach((objective) => {
      if (!objective || objective.type !== 'defeat' || !objective.enemyId) return;
      const count = Math.max(1, Math.floor(Number(objective.count || 1) || 1));
      for (let index = 0; index < count; index += 1) enemyIds.push(objective.enemyId);
    });
    const paletteByClass = {
      fighter: ['#8e5f3f', '#d8b46f', '#4d5968'],
      mage: ['#5b67b7', '#83d7ff', '#3d335f'],
      archer: ['#4f8f65', '#d3c96b', '#40563f']
    };
    return {
      id: `trial_instance_${id}`,
      name: `${trial && trial.title || 'Class Trial'} Instance`,
      levelRange: [20, 25],
      safeZone: false,
      isTrialInstance: true,
      trialId: id,
      scaleEnemies: true,
      waveMax: 0,
      waveDelay: 0,
      palette: paletteByClass[trial && trial.baseClass] || ['#686f86', '#bfd7ff', '#2f3443'],
      asset: data.CLASS_TRIAL_ASSETS && data.CLASS_TRIAL_ASSETS[id] || '',
      environment: data.MAP_ENVIRONMENT_PROFILES && data.MAP_ENVIRONMENT_PROFILES[id] || null,
      purpose: 'Instanced job advancement trial arena.',
      enemies: enemyIds,
      platforms: [
        { x: 0, y: 520, w: 3600, h: 80, terrainVisual: { kind: 'ground', segments: [] } },
        { x: 260, y: 456, w: 760, h: 22, terrainVisual: { kind: 'solidLane', segments: [] } },
        { x: 1140, y: 456, y2: 386, w: 360, h: 24, shape: 'slope', terrainVisual: { kind: 'slope', segments: [] } },
        { x: 1620, y: 456, w: 820, h: 22, terrainVisual: { kind: 'solidLane', segments: [] } },
        { x: 760, y: 318, w: 760, h: 22, terrainVisual: { kind: 'solidLane', segments: [] } },
        { x: 1660, y: 318, w: 700, h: 22, terrainVisual: { kind: 'solidLane', segments: [] } },
        { x: 1060, y: 180, w: 900, h: 22, terrainVisual: { kind: 'island', segments: [] } },
        { x: 1420, y: 386, w: 220, h: 22, terrainVisual: { kind: 'connector', segments: [] } },
        { x: 980, y: 249, y2: 180, w: 300, h: 24, shape: 'slope', terrainVisual: { kind: 'slope', segments: [] } },
        { x: 1960, y: 249, y2: 180, w: 300, h: 24, shape: 'slope', terrainVisual: { kind: 'slope', segments: [] } },
        { x: 2480, y: 366, w: 280, h: 20, terrainVisual: { kind: 'hop', segments: [] } }
      ],
      climbables: [
        { id: `trial_${id}_rope_1_mid`, x: 1100, y: 318, w: 28, h: 138 },
        { id: `trial_${id}_rope_2_mid`, x: 1760, y: 318, w: 28, h: 138 },
        { id: `trial_${id}_rope_1_high`, x: 1220, y: 180, w: 28, h: 138 },
        { id: `trial_${id}_rope_2_high`, x: 2040, y: 180, w: 28, h: 138 }
      ],
      spawnPoints: [
        { x: 700, platformIndex: 1, weight: 2 },
        { x: 1260, platformIndex: 2, weight: 2 },
        { x: 1880, platformIndex: 3, weight: 2 },
        { x: 1160, platformIndex: 4, weight: 2 },
        { x: 2020, platformIndex: 5, weight: 2 },
        { x: 1500, platformIndex: 6, weight: 1 },
        { x: 2580, platformIndex: 10, weight: 2 }
      ],
      stations: [],
      questNpcs: []
    };
  }

  function createPerformanceBenchmarkMap(options) {
    const settings = options || {};
    const data = getGeneratedMapData(settings);
    const enemyIds = Array.isArray(settings.enemyIds) ? settings.enemyIds : [];
    return {
      id: settings.mapId || DEFAULT_PERFORMANCE_BENCHMARK_MAP_ID,
      name: settings.mapName || DEFAULT_PERFORMANCE_BENCHMARK_MAP_NAME,
      levelRange: [1, 999],
      safeZone: true,
      isTrialInstance: true,
      benchmarkMap: true,
      scaleEnemies: true,
      waveMax: 0,
      waveDelay: 0,
      palette: ['#22314d', '#7bdff2', '#182033'],
      asset: data.CLASS_TRIAL_ASSETS && data.CLASS_TRIAL_ASSETS.storm_mage_trial || '',
      environment: data.MAP_ENVIRONMENT_PROFILES && data.MAP_ENVIRONMENT_PROFILES.storm_mage_trial || null,
      purpose: 'Dedicated performance benchmark arena.',
      enemies: enemyIds.slice(),
      platforms: [
        [0, 520, 4200, 80],
        [260, 456, 1120, 22],
        [1720, 456, 1180, 22],
        [560, 318, 1080, 22],
        [2020, 318, 980, 22],
        [1080, 180, 1060, 22],
        [1420, 386, 240, 22],
        [1280, 249, 240, 22],
        [2240, 249, 240, 22]
      ],
      climbables: [
        { id: 'benchmark_rope_low_left', x: 780, y: 318, w: 28, h: 138 },
        { id: 'benchmark_rope_low_right', x: 2280, y: 318, w: 28, h: 138 },
        { id: 'benchmark_rope_high_left', x: 1300, y: 180, w: 28, h: 138 },
        { id: 'benchmark_rope_high_right', x: 2060, y: 180, w: 28, h: 138 }
      ],
      spawnPoints: [
        { x: 660, platformIndex: 1, weight: 2 },
        { x: 1180, platformIndex: 3, weight: 2 },
        { x: 1860, platformIndex: 2, weight: 2 },
        { x: 2480, platformIndex: 4, weight: 2 },
        { x: 1600, platformIndex: 5, weight: 1 },
        { x: 2360, platformIndex: 8, weight: 1 }
      ],
      stations: [],
      questNpcs: []
    };
  }

  const api = {
    createClassTrialInstanceMap,
    createPerformanceBenchmarkMap
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.generatedMaps = Object.assign({}, modules.generatedMaps || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
