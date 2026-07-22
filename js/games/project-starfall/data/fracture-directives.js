(function initProjectStarfallDataFractureDirectives(global) {
  'use strict';

  const FRACTURE_DIRECTIVE_SEASON_ID = 'beta_foundations';
  const FRACTURE_DIRECTIVE_MAX_STABILIZATION = 3;
  const FRACTURE_DIRECTIVE_REWARDS = Object.freeze({
    currency: 300,
    starTokens: 180,
    materials: Object.freeze({
      upgradeDust: 10,
      upgradeCatalyst: 1
    })
  });
  const GREENROOT_ATLAS_ROUTE = Object.freeze([
    'greenrootMeadow',
    'thornpathThicket',
    'banditRidgeCamp',
    'brambleDepths'
  ]);
  const GREENROOT_STABILIZATION = Object.freeze({
    kind: 'atlasVisual',
    visualOnly: true,
    maxSeals: FRACTURE_DIRECTIVE_MAX_STABILIZATION,
    label: 'Greenroot relay stabilization'
  });

  const FRACTURE_DIRECTIVES = Object.freeze([
    Object.freeze({
      id: 'greenroot_relay_survey',
      seasonId: FRACTURE_DIRECTIVE_SEASON_ID,
      name: 'Greenroot Relay Survey',
      summary: 'Recalibrate the forest relay by reading enemy pressure across each surviving Greenroot approach.',
      playstyle: 'fieldSurvey',
      areaId: 'greenroot',
      routeId: 'forest',
      minLevel: 25,
      estimatedMinutes: 75.6,
      mapIds: GREENROOT_ATLAS_ROUTE,
      requiredMapIds: Object.freeze(['greenrootMeadow', 'thornpathThicket', 'banditRidgeCamp']),
      objectives: Object.freeze([
        Object.freeze({ id: 'directive_greenroot_meadow_survey', type: 'defeat', mapId: 'greenrootMeadow', count: 72, label: 'Defeat 72 enemies in Starfall Verge' }),
        Object.freeze({ id: 'directive_greenroot_thornpath_survey', type: 'defeat', mapId: 'thornpathThicket', count: 72, label: 'Defeat 72 enemies in Thornpath Thicket' }),
        Object.freeze({ id: 'directive_greenroot_ridge_survey', type: 'defeat', mapId: 'banditRidgeCamp', count: 72, label: 'Defeat 72 enemies in Bandit Ridge Camp' })
      ]),
      rewards: FRACTURE_DIRECTIVE_REWARDS,
      stabilization: GREENROOT_STABILIZATION
    }),
    Object.freeze({
      id: 'greenroot_echo_breach',
      seasonId: FRACTURE_DIRECTIVE_SEASON_ID,
      name: 'Bramble Echo Breach',
      summary: 'Break repeated Brambleking echoes so the relay can learn and seal the crown resonance.',
      playstyle: 'bossBreach',
      areaId: 'greenroot',
      routeId: 'forest',
      minLevel: 25,
      estimatedMinutes: 70,
      mapIds: GREENROOT_ATLAS_ROUTE,
      requiredMapIds: Object.freeze(['brambleDepths']),
      objectives: Object.freeze([
        Object.freeze({ id: 'directive_greenroot_bramble_breach', type: 'defeatBoss', bossId: 'brambleking', count: 7, label: 'Defeat 7 Brambleking echoes' })
      ]),
      rewards: FRACTURE_DIRECTIVE_REWARDS,
      stabilization: GREENROOT_STABILIZATION
    }),
    Object.freeze({
      id: 'greenroot_expedition_relay',
      seasonId: FRACTURE_DIRECTIVE_SEASON_ID,
      name: 'Bramble Depths Relay',
      summary: 'Escort five calibration runs through Bramble Depths and reopen its fractured expedition lane.',
      playstyle: 'dungeonExpedition',
      areaId: 'greenroot',
      routeId: 'forest',
      minLevel: 25,
      estimatedMinutes: 75,
      mapIds: GREENROOT_ATLAS_ROUTE,
      requiredMapIds: Object.freeze(['brambleDepths']),
      requiredDungeonIds: Object.freeze(['bramble_depths']),
      objectives: Object.freeze([
        Object.freeze({ id: 'directive_greenroot_depths_relay', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 5, label: 'Clear Bramble Depths 5 times' })
      ]),
      rewards: FRACTURE_DIRECTIVE_REWARDS,
      stabilization: GREENROOT_STABILIZATION
    })
  ]);

  const api = {
    FRACTURE_DIRECTIVE_SEASON_ID,
    FRACTURE_DIRECTIVE_MAX_STABILIZATION,
    FRACTURE_DIRECTIVE_REWARDS,
    FRACTURE_DIRECTIVES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.fractureDirectives = Object.assign({}, modules.fractureDirectives || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
