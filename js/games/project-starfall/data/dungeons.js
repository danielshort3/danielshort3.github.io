(function initProjectStarfallDataDungeons(global) {
  'use strict';

  function createEncounterFlow(config) {
    const source = config || {};
    return Object.freeze({
      id: source.id,
      bossIntroDelaySeconds: Number(source.bossIntroDelaySeconds || 2.6),
      bossHpScale: Math.max(1, Number(source.bossHpScale || 1) || 1),
      beats: Object.freeze((source.beats || []).map((beat) => Object.freeze({
        id: beat.id,
        kind: beat.kind,
        name: beat.name,
        summary: beat.summary,
        sectionIds: Object.freeze((beat.sectionIds || []).slice()),
        spawnGroupIds: Object.freeze((beat.spawnGroupIds || []).slice()),
        enemyIds: Object.freeze((beat.enemyIds || []).slice()),
        bossIds: Object.freeze((beat.bossIds || []).slice()),
        gateX: Math.max(0, Number(beat.gateX || 0))
      })))
    });
  }

  const DUNGEONS = Object.freeze([
    Object.freeze({
      id: 'bramble_depths',
      name: 'Bramble Depths',
      summary: 'An early MVP Alpha dungeon built around control, fire payoff, and vertical add pressure.',
      mapId: 'brambleDepths',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'brambleking',
      requiresAdvancedClass: true,
      encounterFlow: createEncounterFlow({
        id: 'bramble_depths_route',
        bossIntroDelaySeconds: 2.6,
        bossHpScale: 6,
        beats: [
          {
            id: 'break_root_gate',
            kind: 'combat',
            name: 'Break the Root Gate',
            summary: 'Clear the ridge return and break the living seal blocking the root lanes.',
            sectionIds: ['brambleDepths_ridge_return'],
            enemyIds: ['thornSprout', 'vineSnapper', 'thornSprout', 'mossback'],
            gateX: 1200
          },
          {
            id: 'purge_bramble_seal',
            kind: 'combat',
            name: 'Purge the Bramble Seal',
            summary: 'Defeat the root-lane defenders and open the approach to the court.',
            sectionIds: ['brambleDepths_root_lanes'],
            enemyIds: ['briarStag', 'glowcapHealer', 'vineSnapper', 'thornSprout'],
            gateX: 3200
          },
          {
            id: 'challenge_brambleking',
            kind: 'boss',
            name: 'Challenge the Brambleking',
            summary: 'Enter the court gate and break the Brambleking crown.',
            sectionIds: ['brambleDepths_court_gate'],
            bossIds: ['brambleking']
          }
        ]
      }),
      rewards: Object.freeze({ xp: 360, currency: 180, materials: Object.freeze({ upgradeDust: 5, gelDrop: 2 }) })
    }),
    Object.freeze({
      id: 'emberjaw_lair',
      name: 'Emberjaw Lair',
      summary: 'A compact party-style dungeon that culminates in the Emberjaw Golem boss fight.',
      mapId: 'emberjawLair',
      levelRequirement: 25,
      recommendedPartySize: 4,
      bossId: 'emberjawGolem',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 420, currency: 220, materials: Object.freeze({ upgradeDust: 6, upgradeCatalyst: 1 }) })
    }),
    Object.freeze({
      id: 'gearworks_vault',
      name: 'Gearworks Vault',
      summary: 'A late prototype dungeon that checks armor break, add control, and boss uptime.',
      mapId: 'gearworksVault',
      levelRequirement: 35,
      recommendedPartySize: 4,
      bossId: 'quarryColossus',
      bossIds: Object.freeze(['clockworkTitan', 'quarryColossus']),
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 620, currency: 320, materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 3 }) })
    }),
    Object.freeze({
      id: 'rimewarden_sanctum',
      name: 'Rimewarden Sanctum',
      summary: 'A Frostfen dungeon built around slick footing, frost flyers, and Rimewarden arena control.',
      mapId: 'rimewardenSanctum',
      levelRequirement: 58,
      recommendedPartySize: 4,
      bossId: 'rimewarden',
      requiresAdvancedClass: true,
      rewards: Object.freeze({ xp: 760, currency: 380, materials: Object.freeze({ upgradeDust: 9, upgradeCatalyst: 3, refinementCore: 1 }) })
    })
  ]);

  const api = {
    DUNGEONS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.dungeons = Object.assign({}, modules.dungeons || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
