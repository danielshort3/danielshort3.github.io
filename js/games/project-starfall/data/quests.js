(function initProjectStarfallDataQuests(global) {
  'use strict';

  function defaultFreezeQuestReward(reward) {
    const source = reward || {};
    const frozen = Object.assign({}, source);
    ['materials', 'consumables', 'items', 'timedBuffs', 'permanentStats'].forEach((key) => {
      if (frozen[key] && typeof frozen[key] === 'object') frozen[key] = Object.freeze(frozen[key]);
    });
    return Object.freeze(frozen);
  }

  function getQuestStatUpgradePoints(config) {
    const reward = config && config.rewards || {};
    if (reward.statUpgradePoints != null) return Math.max(0, Math.floor(Number(reward.statUpgradePoints) || 0));
    return config && config.chainId === 'boss_echoes' ? 2 : 1;
  }

  function createQuestData(options) {
    const settings = options || {};
    const freezeQuestReward = typeof settings.freezeQuestReward === 'function'
      ? settings.freezeQuestReward
      : defaultFreezeQuestReward;

    function quest(config) {
      const rewards = Object.assign({}, config.rewards || {}, {
        statUpgradePoints: getQuestStatUpgradePoints(config)
      });
      return Object.freeze(Object.assign({}, config, {
        objectives: Object.freeze((config.objectives || []).map((objective) => Object.freeze(objective))),
        rewards: freezeQuestReward(rewards)
      }));
    }

    const QUESTS = Object.freeze([
      Object.freeze({
        id: 'first_steps',
        title: 'First Expedition',
        summary: 'Reach Starfall Verge, defeat its first frontier threats, and recover a dropped field material.',
        objectives: Object.freeze([
          Object.freeze({ id: 'travel_greenroot', type: 'travel', mapId: 'greenrootMeadow', count: 1, label: 'Reach Starfall Verge' }),
          Object.freeze({ id: 'defeat_glassbacks_intro', type: 'defeat', enemyId: 'glassback', count: 3, label: 'Defeat 3 Glassbacks' }),
          Object.freeze({ id: 'loot_drop', type: 'loot', count: 1, label: 'Loot 1 dropped item' })
        ]),
        rewards: Object.freeze({ xp: 90, currency: 60, materials: Object.freeze({ upgradeDust: 2 }), statUpgradePoints: 1 }),
        nextQuestId: 'field_scout'
      }),
      Object.freeze({
        id: 'field_scout',
        title: 'Thornpath Field Scout',
        summary: 'Push into the thicket and prove you can handle mixed melee and ranged packs.',
        objectives: Object.freeze([
          Object.freeze({ id: 'travel_thornpath', type: 'travel', mapId: 'thornpathThicket', count: 1, label: 'Travel to Thornpath Thicket' }),
          Object.freeze({ id: 'defeat_mossbacks', type: 'defeat', enemyId: 'mossback', count: 2, label: 'Defeat 2 Mossbacks' }),
          Object.freeze({ id: 'defeat_thorns', type: 'defeat', enemyId: 'thornSprout', count: 2, label: 'Defeat 2 Thorn Sprouts' })
        ]),
        rewards: Object.freeze({ xp: 180, currency: 90, materials: Object.freeze({ upgradeDust: 3, gelDrop: 1 }), statUpgradePoints: 1 }),
        nextQuestId: 'trial_ready'
      }),
      Object.freeze({
        id: 'trial_ready',
        title: 'Ready for Advancement',
        summary: 'Reach the trial tier and complete any branch trial before choosing an advanced class.',
        requiredLevel: 20,
        objectives: Object.freeze([
          Object.freeze({ id: 'reach_20', type: 'level', level: 20, label: 'Reach Level 20' }),
          Object.freeze({ id: 'complete_trial', type: 'trialComplete', count: 1, label: 'Complete any class trial' })
        ]),
        rewards: Object.freeze({ xp: 260, currency: 140, materials: Object.freeze({ upgradeDust: 4, upgradeCatalyst: 1 }), statUpgradePoints: 1 }),
        nextQuestId: 'emberjaw_lair'
      }),
      Object.freeze({
        id: 'emberjaw_lair',
        title: 'Emberjaw Vertical Slice',
        summary: 'Enter the first dungeon, test advanced-class power, and defeat the Emberjaw Golem.',
        objectives: Object.freeze([
          Object.freeze({ id: 'reach_25', type: 'level', level: 25, label: 'Reach Level 25' }),
          Object.freeze({ id: 'clear_emberjaw', type: 'dungeonComplete', dungeonId: 'emberjaw_lair', count: 1, label: 'Clear Emberjaw Lair' })
        ]),
        rewards: Object.freeze({ xp: 520, currency: 260, materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 3 }), statUpgradePoints: 1 }),
        nextQuestId: ''
      }),
      quest({
        id: 'greenroot_samples',
        chainId: 'greenroot_relief',
        title: 'Verge Field Samples',
        summary: 'Recover stable star-glass so the repair crew can reinforce the frontier beacon.',
        requiredQuestIds: ['first_steps'],
        objectives: [
          { id: 'defeat_glassbacks', type: 'defeat', enemyId: 'glassback', mapId: 'greenrootMeadow', count: 4, label: 'Defeat 4 Glassbacks' },
          { id: 'collect_star_glass', type: 'loot', materialId: 'starGlassChip', count: 2, label: 'Collect 2 Star-glass Chips' }
        ],
        rewards: { xp: 140, currency: 80, materials: { upgradeDust: 3, starGlassChip: 1 }, consumables: { minor_health_potion: 2 } }
      }),
      quest({
        id: 'ridge_courier',
        chainId: 'thornpath_ridge',
        title: 'Courier to the Ridge',
        summary: 'Carry Thornpath Scout orders to the Ridge Watch before the bandits settle in.',
        requiredQuestIds: ['field_scout'],
        objectives: [
          { id: 'thin_vines', type: 'defeat', enemyId: 'vineSnapper', mapId: 'thornpathThicket', count: 3, label: 'Defeat 3 Vine Snappers' },
          { id: 'talk_ridge_watch', type: 'talk', npcId: 'ridge_watch', mapId: 'banditRidgeCamp', count: 1, label: 'Report to Ridge Watch' }
        ],
        rewards: { xp: 260, currency: 130, materials: { upgradeDust: 4 }, consumables: { camp_ration: 1 } }
      }),
      quest({
        id: 'ridge_cleanup',
        chainId: 'thornpath_ridge',
        title: 'Ridge Cleanup',
        summary: 'Break the bandit camp foothold and recover upgrade supplies from the ridge.',
        requiredQuestIds: ['ridge_courier'],
        requiredLevel: 18,
        objectives: [
          { id: 'defeat_cutters', type: 'defeat', enemyId: 'banditCutter', mapId: 'banditRidgeCamp', count: 6, label: 'Defeat 6 Bandit Cutters' },
          { id: 'defeat_throwers', type: 'defeat', enemyId: 'banditThrower', mapId: 'banditRidgeCamp', count: 4, label: 'Defeat 4 Bandit Throwers' },
          { id: 'recover_upgrade_dust', type: 'loot', materialId: 'upgradeDust', count: 2, label: 'Recover 2 Upgrade Dust' }
        ],
        rewards: { xp: 520, currency: 240, materials: { upgradeDust: 7 }, consumables: { guard_tonic: 1 } }
      }),
      quest({
        id: 'bramble_crown_report',
        chainId: 'thornpath_ridge',
        title: 'Bramble Crown Report',
        summary: 'Push through Bramble Depths and bring proof that the old root crown can be contained.',
        requiredQuestIds: ['ridge_cleanup'],
        requiredLevel: 25,
        objectives: [
          { id: 'clear_bramble_depths', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 1, label: 'Clear Bramble Depths' }
        ],
        rewards: { xp: 820, currency: 420, materials: { upgradeDust: 10, upgradeCatalyst: 2 }, consumables: { base_skill_manual: 1 } }
      }),
      quest({
        id: 'rustcoil_relay',
        chainId: 'rustcoil_front',
        title: 'Rustcoil Relay',
        summary: 'Meet the Rustcoil Surveyor and open a safer path into the construct ruins.',
        requiredQuestIds: ['field_scout'],
        requiredLevel: 12,
        objectives: [
          { id: 'talk_surveyor', type: 'talk', npcId: 'ruins_surveyor', mapId: 'rustcoilRuins', count: 1, label: 'Speak with the Rustcoil Surveyor' }
        ],
        rewards: { xp: 300, currency: 150, materials: { upgradeDust: 5 }, consumables: { minor_resource_tonic: 2 } }
      }),
      quest({
        id: 'rustcoil_reclamation',
        chainId: 'rustcoil_front',
        title: 'Rustcoil Reclamation',
        summary: 'Disable the first construct patrols and recover upgrade dust from the ruined machinery.',
        requiredQuestIds: ['rustcoil_relay'],
        objectives: [
          { id: 'defeat_ratchets', type: 'defeat', enemyId: 'rustRatchet', mapId: 'rustcoilRuins', count: 5, label: 'Defeat 5 Rust Ratchets' },
          { id: 'defeat_clockbugs', type: 'defeat', enemyId: 'clockbug', mapId: 'rustcoilRuins', count: 4, label: 'Defeat 4 Clockbugs' },
          { id: 'collect_rust_dust', type: 'loot', materialId: 'upgradeDust', count: 3, label: 'Collect 3 Upgrade Dust' }
        ],
        rewards: { xp: 620, currency: 300, materials: { upgradeDust: 10, upgradeCatalyst: 1 }, consumables: { guard_tonic: 1 } }
      }),
      quest({
        id: 'quarry_contract',
        chainId: 'rustcoil_front',
        title: 'Oreback Quarry Contract',
        summary: 'Help the quarry crew reclaim ore lanes and ship raw ore back to Rustcoil Outpost.',
        requiredQuestIds: ['rustcoil_reclamation'],
        requiredLevel: 24,
        objectives: [
          { id: 'defeat_orebacks', type: 'defeat', enemyId: 'orebackBeetle', mapId: 'orebackQuarry', count: 6, label: 'Defeat 6 Oreback Beetles' },
          { id: 'collect_ore_chunks', type: 'loot', materialId: 'oreChunks', count: 4, label: 'Collect 4 Ore Chunks' }
        ],
        rewards: { xp: 880, currency: 430, materials: { oreChunks: 8, upgradeCatalyst: 2 }, consumables: { camp_ration: 2 } }
      }),
      quest({
        id: 'gearworks_vault_report',
        chainId: 'rustcoil_front',
        title: 'Gearworks Vault Report',
        summary: 'Clear the Gearworks Vault and brief the Cinder Envoy on what the constructs were guarding.',
        requiredQuestIds: ['quarry_contract'],
        requiredLevel: 35,
        objectives: [
          { id: 'clear_gearworks', type: 'dungeonComplete', dungeonId: 'gearworks_vault', count: 1, label: 'Clear Gearworks Vault' },
          { id: 'talk_cinder_envoy', type: 'talk', npcId: 'cinder_envoy', mapId: 'cinderRefuge', count: 1, label: 'Report to the Cinder Envoy' }
        ],
        rewards: { xp: 1220, currency: 650, materials: { upgradeDust: 14, upgradeCatalyst: 3 }, consumables: { potential_cube: 1 } }
      }),
      quest({
        id: 'cinder_dispatch',
        chainId: 'cinder_front',
        title: 'Cinder Dispatch',
        summary: 'Carry the refuge orders into Cinder Hollow and find the pathfinder watching the furnace roads.',
        requiredQuestIds: ['trial_ready'],
        requiredLevel: 25,
        objectives: [
          { id: 'talk_pathfinder', type: 'talk', npcId: 'cinder_pathfinder', mapId: 'cinderHollow', count: 1, label: 'Speak with the Cinder Pathfinder' }
        ],
        rewards: { xp: 520, currency: 280, materials: { upgradeDust: 6, upgradeCatalyst: 1 }, consumables: { minor_health_potion: 2, minor_resource_tonic: 2 } }
      }),
      quest({
        id: 'cinder_samples',
        chainId: 'cinder_front',
        title: 'Cinder Samples',
        summary: 'Gather volatile cinder samples while cutting down the fast volcanic packs.',
        requiredQuestIds: ['cinder_dispatch'],
        objectives: [
          { id: 'defeat_lava_ticks', type: 'defeat', enemyId: 'lavaTick', mapId: 'cinderHollow', count: 6, label: 'Defeat 6 Lava Ticks' },
          { id: 'defeat_spitters', type: 'defeat', enemyId: 'cinderSpitter', mapId: 'cinderHollow', count: 4, label: 'Defeat 4 Cinder Spitters' },
          { id: 'collect_catalyst', type: 'loot', materialId: 'upgradeCatalyst', count: 1, label: 'Collect 1 Upgrade Catalyst' }
        ],
        rewards: { xp: 820, currency: 420, materials: { upgradeDust: 10, upgradeCatalyst: 2 }, consumables: { swiftstep_oil: 1 } }
      }),
      quest({
        id: 'emberjaw_report',
        chainId: 'cinder_front',
        title: 'Emberjaw Report',
        summary: 'Confirm Emberjaw Lair is contained and return the report to the refuge.',
        requiredQuestIds: ['cinder_samples'],
        requiredLevel: 25,
        objectives: [
          { id: 'clear_emberjaw_lair', type: 'dungeonComplete', dungeonId: 'emberjaw_lair', count: 1, label: 'Clear Emberjaw Lair' },
          { id: 'report_cinder_envoy', type: 'talk', npcId: 'cinder_envoy', mapId: 'cinderRefuge', count: 1, label: 'Report to the Cinder Envoy' }
        ],
        rewards: { xp: 980, currency: 520, materials: { upgradeDust: 12, upgradeCatalyst: 3 }, consumables: { advanced_skill_manual: 1 } }
      }),
      quest({
        id: 'ashglass_crossing',
        chainId: 'cinder_front',
        title: 'Ashglass Crossing',
        summary: 'Open the ashglass route by meeting the courier and clearing the elite glass trail.',
        requiredQuestIds: ['emberjaw_report'],
        requiredLevel: 40,
        objectives: [
          { id: 'talk_ashglass_courier', type: 'talk', npcId: 'ashglass_courier', mapId: 'ashglassPass', count: 1, label: 'Meet the Ashglass Courier' },
          { id: 'clear_ashglass_wisps', type: 'defeat', enemyId: 'emberWisp', mapId: 'ashglassPass', count: 6, label: 'Defeat 6 Ember Wisps in Ashglass Pass' },
          { id: 'collect_ash_catalysts', type: 'loot', materialId: 'upgradeCatalyst', count: 2, label: 'Collect 2 Upgrade Catalysts' }
        ],
        rewards: { xp: 1400, currency: 760, materials: { upgradeDust: 16, upgradeCatalyst: 4 }, consumables: { guard_tonic: 2 } }
      }),
      quest({
        id: 'frostfen_relay',
        chainId: 'frostfen_front',
        title: 'Frostfen Relay',
        summary: 'Carry the Ashglass route report to Frostfen and meet the tracker at the frozen outskirts.',
        requiredQuestIds: ['ashglass_crossing'],
        requiredLevel: 45,
        objectives: [
          { id: 'talk_frostfen_tracker', type: 'talk', npcId: 'frostfen_tracker', mapId: 'frostfenOutskirts', count: 1, label: 'Speak with the Frostfen Tracker' }
        ],
        rewards: { xp: 1100, currency: 620, materials: { upgradeDust: 12, cubeFragment: 1 }, consumables: { camp_ration: 2 } }
      }),
      quest({
        id: 'frostfen_field_notes',
        chainId: 'frostfen_front',
        title: 'Frostfen Field Notes',
        summary: 'Document frost packs and recover prism shards for the quartermaster.',
        requiredQuestIds: ['frostfen_relay'],
        objectives: [
          { id: 'defeat_shardlings', type: 'defeat', enemyId: 'shardling', mapId: 'frostfenOutskirts', count: 7, label: 'Defeat 7 Shardlings' },
          { id: 'defeat_scouts', type: 'defeat', enemyId: 'frostlingScout', mapId: 'frostfenOutskirts', count: 5, label: 'Defeat 5 Frostling Scouts' },
          { id: 'collect_prism_shard', type: 'loot', materialId: 'cubeFragment', count: 1, label: 'Recover 1 Prism Shard' }
        ],
        rewards: { xp: 1580, currency: 860, materials: { upgradeDust: 18, cubeFragment: 2 }, consumables: { potential_cube: 1 } }
      }),
      quest({
        id: 'glacier_cartography',
        chainId: 'frostfen_front',
        title: 'Glacier Cartography',
        summary: 'Map the glacier ridge, break sentinel positions, and return the chart to Frostfen Camp.',
        requiredQuestIds: ['frostfen_field_notes'],
        requiredLevel: 52,
        objectives: [
          { id: 'defeat_sentinels', type: 'defeat', enemyId: 'glacierSentinel', mapId: 'glacierSpine', count: 5, label: 'Defeat 5 Glacier Sentinels' },
          { id: 'defeat_brutes', type: 'defeat', enemyId: 'rimebackBrute', mapId: 'glacierSpine', count: 5, label: 'Defeat 5 Rimeback Brutes' },
          { id: 'return_quartermaster', type: 'talk', npcId: 'frostfen_quartermaster', mapId: 'frostfenCamp', count: 1, label: 'Return the chart to Frostfen Camp' }
        ],
        rewards: { xp: 1950, currency: 1040, materials: { upgradeCatalyst: 4, cubeFragment: 2 }, consumables: { preservation_cube: 1 } }
      }),
      quest({
        id: 'rimewarden_sanctum_report',
        chainId: 'frostfen_front',
        title: 'Rimewarden Sanctum Report',
        summary: 'Clear the sanctum and prepare the highland route for Stormbreak support.',
        requiredQuestIds: ['glacier_cartography'],
        requiredLevel: 58,
        objectives: [
          { id: 'clear_rimewarden', type: 'dungeonComplete', dungeonId: 'rimewarden_sanctum', count: 1, label: 'Clear Rimewarden Sanctum' }
        ],
        rewards: { xp: 2400, currency: 1300, materials: { upgradeCatalyst: 5, cubeFragment: 3, refinementCore: 1 }, consumables: { advanced_skill_manual: 1 } }
      }),
      quest({
        id: 'stormbreak_orders',
        chainId: 'stormbreak_front',
        title: 'Stormbreak Orders',
        summary: 'Meet the cliff scout and prepare lightning rods for the upper routes.',
        requiredQuestIds: ['rimewarden_sanctum_report'],
        requiredLevel: 60,
        objectives: [
          { id: 'talk_stormbreak_scout', type: 'talk', npcId: 'stormbreak_scout', mapId: 'stormbreakCliffs', count: 1, label: 'Speak with the Stormbreak Scout' }
        ],
        rewards: { xp: 1800, currency: 980, materials: { cubeFragment: 2 }, consumables: { swiftstep_oil: 2 } }
      }),
      quest({
        id: 'stormbreak_rods',
        chainId: 'stormbreak_front',
        title: 'Stormbreak Rods',
        summary: 'Clear storm packs and recover prism shards to tune the lightning rods.',
        requiredQuestIds: ['stormbreak_orders'],
        objectives: [
          { id: 'defeat_harriers', type: 'defeat', enemyId: 'galeHarrier', mapId: 'stormbreakCliffs', count: 7, label: 'Defeat 7 Gale Harriers' },
          { id: 'defeat_archers', type: 'defeat', enemyId: 'stormboundArcher', mapId: 'stormbreakCliffs', count: 6, label: 'Defeat 6 Stormbound Archers' },
          { id: 'collect_rod_shards', type: 'loot', materialId: 'cubeFragment', count: 2, label: 'Collect 2 Prism Shards' }
        ],
        rewards: { xp: 2600, currency: 1450, materials: { upgradeCatalyst: 5, cubeFragment: 4 }, consumables: { potential_cube: 2 } }
      }),
      quest({
        id: 'astral_liaison',
        chainId: 'astral_front',
        title: 'Astral Liaison',
        summary: 'Carry Stormbreak findings to the observatory and coordinate with the astral scribe.',
        requiredQuestIds: ['stormbreak_rods'],
        requiredLevel: 70,
        objectives: [
          { id: 'talk_observatory_liaison', type: 'talk', npcId: 'observatory_liaison', mapId: 'astralObservatory', count: 1, label: 'Report to the Observatory Liaison' },
          { id: 'talk_astral_scribe', type: 'talk', npcId: 'astral_scribe', mapId: 'astralArchive', count: 1, label: 'Meet the Astral Scribe' }
        ],
        rewards: { xp: 2200, currency: 1200, materials: { cubeFragment: 3 }, consumables: { camp_ration: 3 } }
      }),
      quest({
        id: 'astral_indexing',
        chainId: 'astral_front',
        title: 'Astral Indexing',
        summary: 'Rebuild damaged archive indices by defeating living entries and recovering prism shards.',
        requiredQuestIds: ['astral_liaison'],
        objectives: [
          { id: 'defeat_scribes', type: 'defeat', enemyId: 'indexScribe', mapId: 'astralArchive', count: 8, label: 'Defeat 8 Index Scribes' },
          { id: 'defeat_sentinels', type: 'defeat', enemyId: 'lumenSentinel', mapId: 'astralArchive', count: 6, label: 'Defeat 6 Lumen Sentinels' },
          { id: 'collect_archive_shards', type: 'loot', materialId: 'cubeFragment', count: 3, label: 'Collect 3 Prism Shards' }
        ],
        rewards: { xp: 3400, currency: 1900, materials: { cubeFragment: 6, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
      }),
      quest({
        id: 'eclipse_frontier_message',
        chainId: 'eclipse_front',
        title: 'Message to Eclipse Frontier',
        summary: 'Carry the observatory warning to the frontier envoy before the rift pressure builds.',
        requiredQuestIds: ['astral_indexing'],
        requiredLevel: 85,
        objectives: [
          { id: 'talk_eclipse_envoy', type: 'talk', npcId: 'eclipse_envoy', mapId: 'eclipseFrontier', count: 1, label: 'Deliver the message to the Eclipse Envoy' }
        ],
        rewards: { xp: 2600, currency: 1500, materials: { cubeFragment: 4 }, consumables: { advanced_skill_manual: 1 } }
      }),
      quest({
        id: 'eclipse_frontier_patrol',
        chainId: 'eclipse_front',
        title: 'Eclipse Frontier Patrol',
        summary: 'Hold the outer frontier by thinning elite duelists and recovering enough prism shards to reinforce the line.',
        requiredQuestIds: ['eclipse_frontier_message'],
        objectives: [
          { id: 'defeat_duelists', type: 'defeat', enemyId: 'eclipseDuelist', mapId: 'eclipseFrontier', count: 9, label: 'Defeat 9 Eclipse Duelists' },
          { id: 'defeat_void_motes', type: 'defeat', enemyId: 'voidMote', mapId: 'eclipseFrontier', count: 7, label: 'Defeat 7 Void Motes' },
          { id: 'collect_frontier_shards', type: 'loot', materialId: 'cubeFragment', count: 4, label: 'Collect 4 Prism Shards' }
        ],
        rewards: { xp: 4300, currency: 2400, materials: { cubeFragment: 8, refinementCore: 2 }, consumables: { potential_cube: 2, preservation_cube: 1 } }
      }),
      quest({
        id: 'rift_watch',
        chainId: 'eclipse_front',
        title: 'Rift Watch',
        summary: 'Test the endless rift front and recover refined cores from the most unstable enemies.',
        requiredQuestIds: ['eclipse_frontier_patrol'],
        requiredLevel: 100,
        objectives: [
          { id: 'defeat_aberrations', type: 'defeat', enemyId: 'riftAberration', mapId: 'endlessRift', count: 8, label: 'Defeat 8 Rift Aberrations' },
          { id: 'collect_rift_cores', type: 'loot', materialId: 'refinementCore', count: 1, label: 'Recover 1 Refinement Core' }
        ],
        rewards: { xp: 5600, currency: 3200, materials: { cubeFragment: 10, refinementCore: 3 }, consumables: { preservation_cube: 2 } }
      }),
      quest({
        id: 'brambleking_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Brambleking Court',
        summary: 'Enter the Brambleking echo room and defeat the crowned root before it spreads again.',
        requiredQuestIds: ['bramble_crown_report'],
        requiredLevel: 32,
        objectives: [
          { id: 'reach_bramble_court', type: 'travel', mapId: 'bramblekingCourt', count: 1, label: 'Enter Brambleking Court' },
          { id: 'defeat_brambleking_echo', type: 'defeatBoss', bossId: 'brambleking', mapId: 'bramblekingCourt', count: 1, label: 'Defeat Brambleking' }
        ],
        rewards: { xp: 1800, currency: 760, materials: { upgradeCatalyst: 3, cubeFragment: 2 }, consumables: { potential_cube: 1 } }
      }),
      quest({
        id: 'titan_foundry_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Titan Foundry',
        summary: 'Challenge the Clockwork Titan in its foundry echo and recover the exposed core readings.',
        requiredQuestIds: ['gearworks_vault_report'],
        requiredLevel: 50,
        objectives: [
          { id: 'reach_titan_foundry', type: 'travel', mapId: 'titanFoundry', count: 1, label: 'Enter Titan Foundry' },
          { id: 'defeat_titan_echo', type: 'defeatBoss', bossId: 'clockworkTitan', mapId: 'titanFoundry', count: 1, label: 'Defeat Clockwork Titan' }
        ],
        rewards: { xp: 2600, currency: 1200, materials: { upgradeCatalyst: 4, cubeFragment: 3 }, consumables: { potential_cube: 1 } }
      }),
      quest({
        id: 'deepcore_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Deepcore Core',
        summary: 'Fight the Quarry Colossus inside the deepcore echo and break its ore armor.',
        requiredQuestIds: ['gearworks_vault_report'],
        requiredLevel: 60,
        objectives: [
          { id: 'reach_deepcore', type: 'travel', mapId: 'deepcoreCore', count: 1, label: 'Enter Deepcore Core' },
          { id: 'defeat_colossus_echo', type: 'defeatBoss', bossId: 'quarryColossus', mapId: 'deepcoreCore', count: 1, label: 'Defeat Quarry Colossus' }
        ],
        rewards: { xp: 3200, currency: 1550, materials: { upgradeCatalyst: 5, cubeFragment: 3, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
      }),
      quest({
        id: 'emberjaw_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Emberjaw Furnace',
        summary: 'Face Emberjaw in the furnace echo and use its overheat windows to end the fight cleanly.',
        requiredQuestIds: ['emberjaw_report'],
        requiredLevel: 42,
        objectives: [
          { id: 'reach_emberjaw_furnace', type: 'travel', mapId: 'emberjawFurnace', count: 1, label: 'Enter Emberjaw Furnace' },
          { id: 'defeat_emberjaw_echo', type: 'defeatBoss', bossId: 'emberjawGolem', mapId: 'emberjawFurnace', count: 1, label: 'Defeat Emberjaw Golem' }
        ],
        rewards: { xp: 2300, currency: 1080, materials: { upgradeCatalyst: 4, cubeFragment: 3 }, consumables: { potential_cube: 1 } }
      }),
      quest({
        id: 'rimewarden_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Rimewarden Vault',
        summary: 'Enter the Rimewarden echo and hold position through whiteout lanes and frost rings.',
        requiredQuestIds: ['rimewarden_sanctum_report'],
        requiredLevel: 66,
        objectives: [
          { id: 'reach_rimewarden_vault', type: 'travel', mapId: 'rimewardenVault', count: 1, label: 'Enter Rimewarden Vault' },
          { id: 'defeat_rimewarden_echo', type: 'defeatBoss', bossId: 'rimewarden', mapId: 'rimewardenVault', count: 1, label: 'Defeat Rimewarden' }
        ],
        rewards: { xp: 3800, currency: 1900, materials: { cubeFragment: 5, refinementCore: 1 }, consumables: { preservation_cube: 1 } }
      }),
      quest({
        id: 'stormbreak_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Stormbreak Aerie',
        summary: 'Fight Aurelion in the aerie echo and manage lightning rods through divebomb phases.',
        requiredQuestIds: ['stormbreak_rods'],
        requiredLevel: 76,
        objectives: [
          { id: 'reach_stormbreak_aerie', type: 'travel', mapId: 'stormbreakAerie', count: 1, label: 'Enter Stormbreak Aerie' },
          { id: 'defeat_roc_echo', type: 'defeatBoss', bossId: 'stormbreakRoc', mapId: 'stormbreakAerie', count: 1, label: 'Defeat Aurelion' }
        ],
        rewards: { xp: 4500, currency: 2300, materials: { cubeFragment: 6, refinementCore: 1 }, consumables: { potential_cube: 2 } }
      }),
      quest({
        id: 'astral_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Astral Stacks',
        summary: 'Challenge the Astral Archivist echo and keep skill variety high through mirrored pages.',
        requiredQuestIds: ['astral_indexing'],
        requiredLevel: 88,
        objectives: [
          { id: 'reach_astral_stacks', type: 'travel', mapId: 'astralStacks', count: 1, label: 'Enter Astral Stacks' },
          { id: 'defeat_archivist_echo', type: 'defeatBoss', bossId: 'astralArchivist', mapId: 'astralStacks', count: 1, label: 'Defeat the Astral Archivist' }
        ],
        rewards: { xp: 5400, currency: 2800, materials: { cubeFragment: 8, refinementCore: 2 }, consumables: { preservation_cube: 1 } }
      }),
      quest({
        id: 'eclipse_echo',
        chainId: 'boss_echoes',
        title: 'Echo: Eclipse Throne',
        summary: 'Enter the sovereign echo and survive solar and lunar stance swaps through totality.',
        requiredQuestIds: ['eclipse_frontier_patrol'],
        requiredLevel: 100,
        objectives: [
          { id: 'reach_eclipse_throne', type: 'travel', mapId: 'eclipseThrone', count: 1, label: 'Enter Eclipse Throne' },
          { id: 'defeat_sovereign_echo', type: 'defeatBoss', bossId: 'eclipseSovereign', mapId: 'eclipseThrone', count: 1, label: 'Defeat Eclipse Sovereign' }
        ],
        rewards: { xp: 6800, currency: 3600, materials: { cubeFragment: 10, refinementCore: 3 }, consumables: { preservation_cube: 2, advanced_skill_manual: 1 } }
      })
    ]);

    return Object.freeze({ QUESTS });
  }

  const defaultQuestData = createQuestData();
  const api = Object.assign({
    defaultFreezeQuestReward,
    getQuestStatUpgradePoints,
    createQuestData
  }, defaultQuestData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.quests = Object.assign({}, modules.quests || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
