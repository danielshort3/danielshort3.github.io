(function initProjectStarfallDataAccomplishments(global) {
  'use strict';

  const ACCOMPLISHMENTS = Object.freeze([
    Object.freeze({
      id: 'first_cache',
      title: 'First Field Cache',
      summary: 'Loot your first handful of field drops.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'loot_any', type: 'loot', count: 5, label: 'Loot 5 ground drops' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ minor_health_potion: 2, minor_resource_tonic: 2 }),
        materials: Object.freeze({ upgradeDust: 2 })
      })
    }),
    Object.freeze({
      id: 'first_fit',
      title: 'First Fit',
      summary: 'Equip a replacement item and start reading gear stats.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'equip_any', type: 'equip', count: 1, label: 'Equip 1 item' })
      ]),
      rewards: Object.freeze({
        currency: 60,
        consumables: Object.freeze({ camp_ration: 1 }),
        materials: Object.freeze({ upgradeDust: 2 })
      })
    }),
    Object.freeze({
      id: 'skill_spark',
      title: 'Skill Spark',
      summary: 'Spend your first skill point and feel the class kit open.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'rank_first_skill', type: 'rankSkill', count: 1, label: 'Rank 1 skill' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ minor_resource_tonic: 3 }),
        materials: Object.freeze({ upgradeDust: 2 })
      })
    }),
    Object.freeze({
      id: 'town_services',
      title: 'Town Services',
      summary: 'Check the shop and storage loops before committing to a route.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'open_shop', type: 'interact', stationId: 'shop', count: 1, label: 'Open a shop' }),
        Object.freeze({ id: 'open_storage', type: 'interact', stationId: 'storage', count: 1, label: 'Open storage' })
      ]),
      rewards: Object.freeze({
        currency: 120,
        consumables: Object.freeze({ town_return_scroll: 1 }),
        materials: Object.freeze({ upgradeDust: 3 })
      })
    }),
    Object.freeze({
      id: 'pet_assist',
      title: 'Pet Assist',
      summary: 'Unlock Pet Assist so routine loot and potion support feel smoother.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'use_pet_whistle', type: 'useConsumable', consumableId: 'pet_whistle', count: 1, label: 'Use a Pet Whistle' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ minor_health_potion: 3, minor_resource_tonic: 3, magnet_charm: 1 })
      })
    }),
    Object.freeze({
      id: 'greenroot_initiate',
      title: 'Greenroot Initiate',
      summary: 'Finish the first guide chain and collect the starter route payout.',
      category: 'Onboarding',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'claim_first_steps', type: 'questClaim', questId: 'first_steps', count: 1, label: 'Claim First Steps' })
      ]),
      rewards: Object.freeze({
        currency: 160,
        materials: Object.freeze({ upgradeDust: 4, gelDrop: 2 }),
        consumables: Object.freeze({ equipment_slot_coupon: 1 })
      })
    }),
    Object.freeze({
      id: 'slime_sweeper',
      title: 'Slime Sweeper I',
      summary: 'Clear enough Slimelets to make Greenroot safer.',
      category: 'Combat',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'slimelets', type: 'defeat', enemyId: 'slimelet', count: 20, label: 'Defeat 20 Slimelets' })
      ]),
      rewards: Object.freeze({
        items: Object.freeze([Object.freeze({ itemId: 'plain_ring', rarity: 'Uncommon', upgrade: 1, source: 'Slime Sweeper accomplishment' })]),
        timedBuffs: Object.freeze([Object.freeze({ buffId: 'guardTonic', duration: 30 })])
      })
    }),
    Object.freeze({
      id: 'slime_sweeper_ii',
      title: 'Slime Sweeper II',
      summary: 'Keep Greenroot farming productive beyond the first route task.',
      category: 'Combat',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'slimelets_120', type: 'defeat', enemyId: 'slimelet', count: 120, label: 'Defeat 120 Slimelets' })
      ]),
      rewards: Object.freeze({
        currency: 260,
        materials: Object.freeze({ upgradeDust: 10, gelDrop: 6 }),
        cards: Object.freeze([Object.freeze({ cardId: 'gel_spark', rank: 1, source: 'Slime Sweeper II accomplishment' })])
      })
    }),
    Object.freeze({
      id: 'slime_sweeper_iii',
      title: 'Slime Sweeper III',
      summary: 'Turn a starter monster into a long grind target with a real payoff.',
      category: 'Combat',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'slimelets_500', type: 'defeat', enemyId: 'slimelet', count: 500, label: 'Defeat 500 Slimelets' })
      ]),
      rewards: Object.freeze({
        starTokens: 6,
        materials: Object.freeze({ upgradeDust: 24, gelDrop: 16, greenStarCard: 2 }),
        timedBuffs: Object.freeze([Object.freeze({ buffId: 'magnetCharm', duration: 60 })])
      })
    }),
    Object.freeze({
      id: 'ooze_researcher',
      title: 'Ooze Researcher',
      summary: 'Farm the broader ooze family for cards and upgrade stock.',
      category: 'Combat',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'ooze_kills', type: 'defeat', family: 'Ooze', count: 200, label: 'Defeat 200 Ooze enemies' })
      ]),
      rewards: Object.freeze({
        currency: 320,
        materials: Object.freeze({ upgradeDust: 14, gelDrop: 10, dewBead: 6 }),
        cards: Object.freeze([Object.freeze({ cardId: 'gel_spark', rank: 1, quantity: 2, source: 'Ooze Researcher accomplishment' })])
      })
    }),
    Object.freeze({
      id: 'battle_hardened',
      title: 'Battle Hardened',
      summary: 'Let every route, dungeon, and farm session push one long combat bar forward.',
      category: 'Combat',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'all_kills_1000', type: 'defeat', count: 1000, label: 'Defeat 1,000 enemies' })
      ]),
      rewards: Object.freeze({
        starTokens: 12,
        materials: Object.freeze({ upgradeDust: 40, upgradeCatalyst: 6, blueStarCard: 3 }),
        consumables: Object.freeze({ drop_coupon_1_5_1h: 1 })
      })
    }),
    Object.freeze({
      id: 'greenroot_pathfinder',
      title: 'Greenroot Pathfinder',
      summary: 'Push through the first forest route from meadow to ridge.',
      category: 'Exploration',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'greenroot_kills', type: 'defeat', mapId: 'greenrootMeadow', count: 18, label: 'Defeat 18 enemies in Greenroot Meadow' }),
        Object.freeze({ id: 'thornpath_kills', type: 'defeat', mapId: 'thornpathThicket', count: 24, label: 'Defeat 24 enemies in Thornpath Thicket' }),
        Object.freeze({ id: 'ridge_kills', type: 'defeat', mapId: 'banditRidgeCamp', count: 24, label: 'Defeat 24 enemies in Bandit Ridge Camp' })
      ]),
      rewards: Object.freeze({
        items: Object.freeze([Object.freeze({ itemId: 'traveler_boots', rarity: 'Rare', upgrade: 2, source: 'Greenroot Pathfinder accomplishment' })]),
        materials: Object.freeze({ upgradeDust: 8, gelDrop: 4, thornFiber: 4 })
      })
    }),
    Object.freeze({
      id: 'regional_envoy',
      title: 'Regional Envoy',
      summary: 'Reach each major town hub and learn the campaign route shape.',
      category: 'Exploration',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'visit_rustcoil', type: 'travel', mapId: 'rustcoilOutpost', count: 1, label: 'Visit Rustcoil Outpost' }),
        Object.freeze({ id: 'visit_cinder', type: 'travel', mapId: 'cinderRefuge', count: 1, label: 'Visit Cinder Refuge' }),
        Object.freeze({ id: 'visit_frostfen', type: 'travel', mapId: 'frostfenCamp', count: 1, label: 'Visit Frostfen Camp' }),
        Object.freeze({ id: 'visit_stormbreak', type: 'travel', mapId: 'stormbreakHaven', count: 1, label: 'Visit Stormbreak Haven' }),
        Object.freeze({ id: 'visit_astral', type: 'travel', mapId: 'astralObservatory', count: 1, label: 'Visit Astral Observatory' })
      ]),
      rewards: Object.freeze({
        currency: 520,
        consumables: Object.freeze({ town_return_scroll: 3, swiftstep_oil: 2 }),
        materials: Object.freeze({ upgradeDust: 14 })
      })
    }),
    Object.freeze({
      id: 'route_veteran',
      title: 'Route Veteran',
      summary: 'Complete every current field-route kill line at least once.',
      category: 'Exploration',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'forest_route', type: 'defeat', mapId: 'banditRidgeCamp', count: 24, label: 'Finish the Forest route fields' }),
        Object.freeze({ id: 'ruins_route', type: 'defeat', mapId: 'orebackQuarry', count: 26, label: 'Finish the Ruins route fields' }),
        Object.freeze({ id: 'cinder_route', type: 'defeat', mapId: 'cinderHollow', count: 30, label: 'Finish the Cinder route field' }),
        Object.freeze({ id: 'frostfen_route', type: 'defeat', mapId: 'glacierSpine', count: 34, label: 'Finish the Frostfen route fields' }),
        Object.freeze({ id: 'ascension_route', type: 'defeat', mapId: 'endlessRift', count: 44, label: 'Finish the Ascension route fields' })
      ]),
      rewards: Object.freeze({
        starTokens: 14,
        materials: Object.freeze({ upgradeDust: 42, upgradeCatalyst: 7, purpleStarCard: 1 }),
        permanentStats: Object.freeze({ speed: 5 })
      })
    }),
    Object.freeze({
      id: 'horizon_cartographer',
      title: 'Horizon Cartographer',
      summary: 'Touch the far-field landmarks that define the current world map.',
      category: 'Exploration',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'reach_bramble_court', type: 'travel', mapId: 'bramblekingCourt', count: 1, label: 'Reach Brambleking Court' }),
        Object.freeze({ id: 'reach_titan_foundry', type: 'travel', mapId: 'titanFoundry', count: 1, label: 'Reach Titan Foundry' }),
        Object.freeze({ id: 'reach_rimewarden_vault', type: 'travel', mapId: 'rimewardenVault', count: 1, label: 'Reach Rimewarden Vault' }),
        Object.freeze({ id: 'reach_stormbreak_aerie', type: 'travel', mapId: 'stormbreakAerie', count: 1, label: 'Reach Stormbreak Aerie' }),
        Object.freeze({ id: 'reach_eclipse_throne', type: 'travel', mapId: 'eclipseThrone', count: 1, label: 'Reach Eclipse Throne' })
      ]),
      rewards: Object.freeze({
        starTokens: 22,
        cosmeticId: 'crossing_cape',
        consumables: Object.freeze({ drop_coupon_1_5_1h: 1, xp_coupon_1_5_1h: 1 })
      })
    }),
    Object.freeze({
      id: 'upgrade_apprentice',
      title: 'Upgrade Apprentice',
      summary: 'Make several upgrade attempts and learn the gear risk loop.',
      category: 'Crafting',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'upgrade_attempts', type: 'upgrade', count: 5, label: 'Attempt 5 item upgrades' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 1 })
      })
    }),
    Object.freeze({
      id: 'forge_journeyman',
      title: 'Forge Journeyman',
      summary: 'Keep upgrading past the tutorial dust sink and reach a useful plus level.',
      category: 'Crafting',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'upgrade_attempts_25', type: 'upgrade', count: 25, label: 'Attempt 25 item upgrades' }),
        Object.freeze({ id: 'reach_plus_4', type: 'upgrade', minUpgrade: 4, count: 1, label: 'Upgrade any item to +4' })
      ]),
      rewards: Object.freeze({
        currency: 440,
        materials: Object.freeze({ upgradeDust: 20, upgradeCatalyst: 3, wardingScroll: 1 })
      })
    }),
    Object.freeze({
      id: 'forge_veteran',
      title: 'Forge Veteran',
      summary: 'Chase meaningful gear growth without making one roll decide the save.',
      category: 'Crafting',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'upgrade_attempts_100', type: 'upgrade', count: 100, label: 'Attempt 100 item upgrades' }),
        Object.freeze({ id: 'reach_plus_8', type: 'upgrade', minUpgrade: 8, count: 1, label: 'Upgrade any item to +8' })
      ]),
      rewards: Object.freeze({
        starTokens: 10,
        materials: Object.freeze({ upgradeDust: 48, upgradeCatalyst: 8, wardingScroll: 3, refinementCore: 1 }),
        permanentStats: Object.freeze({ power: 1 })
      })
    }),
    Object.freeze({
      id: 'first_attunement',
      title: 'First Attunement',
      summary: 'Roll a first Attunement Prism and start reading potential lines.',
      category: 'Crafting',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'roll_potential', type: 'itemPotential', count: 1, label: 'Attune 1 item' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ cubeFragment: 3 }),
        consumables: Object.freeze({ potential_cube: 1 })
      })
    }),
    Object.freeze({
      id: 'prism_journeyman',
      title: 'Prism Journeyman',
      summary: 'Make attunement a regular part of the gear loop.',
      category: 'Crafting',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'potential_rolls_25', type: 'itemPotential', count: 25, label: 'Attune 25 items' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ cubeFragment: 14, blueStarCard: 1 }),
        consumables: Object.freeze({ potential_cube: 2, preservation_cube: 1 })
      })
    }),
    Object.freeze({
      id: 'lineworker',
      title: 'Lineworker',
      summary: 'Use Line Catalysts to push more value out of favorite gear.',
      category: 'Crafting',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'line_attempts_5', type: 'itemPotentialLineUpgrade', count: 5, label: 'Attempt 5 attunement line upgrades' }),
        Object.freeze({ id: 'three_lines', type: 'itemPotentialLineUpgrade', minLineCount: 3, count: 1, label: 'Reach 3 attunement lines' })
      ]),
      rewards: Object.freeze({
        starTokens: 8,
        materials: Object.freeze({ cubeFragment: 24, refinementCore: 1 }),
        consumables: Object.freeze({ line_catalyst: 2 })
      })
    }),
    Object.freeze({
      id: 'relic_attuner',
      title: 'Relic Attuner',
      summary: 'Land a Relic or better attunement tier on any item.',
      category: 'Crafting',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'relic_potential', type: 'itemPotential', minTier: 'relic', count: 1, label: 'Roll Relic or better attunement' })
      ]),
      rewards: Object.freeze({
        starTokens: 18,
        materials: Object.freeze({ cubeFragment: 40, refinementCore: 2, orangeStarCard: 1 }),
        permanentStats: Object.freeze({ mpMax: 20, resourceGain: 1 })
      })
    }),
    Object.freeze({
      id: 'first_card',
      title: 'First Card',
      summary: 'Pick up a monster card and open the collection chase.',
      category: 'Collection',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'loot_card', type: 'loot', kind: 'card', count: 1, label: 'Loot 1 card' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ card_slot_coupon: 1 }),
        materials: Object.freeze({ whiteStarCard: 3 })
      })
    }),
    Object.freeze({
      id: 'card_binder',
      title: 'Card Binder',
      summary: 'Build a useful stack of duplicate cards for deck upgrades.',
      category: 'Collection',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'loot_cards_25', type: 'loot', kind: 'card', count: 25, label: 'Loot 25 cards' })
      ]),
      rewards: Object.freeze({
        starTokens: 6,
        materials: Object.freeze({ greenStarCard: 4, blueStarCard: 2 }),
        cards: Object.freeze([Object.freeze({ cardId: 'mimic_cache', rank: 1, source: 'Card Binder accomplishment' })])
      })
    }),
    Object.freeze({
      id: 'material_runner',
      title: 'Material Runner',
      summary: 'Collect enough Upgrade Dust to keep several gear experiments moving.',
      category: 'Collection',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'upgrade_dust_50', type: 'loot', materialId: 'upgradeDust', count: 50, label: 'Loot 50 Upgrade Dust' })
      ]),
      rewards: Object.freeze({
        currency: 260,
        materials: Object.freeze({ upgradeDust: 16, upgradeCatalyst: 2 })
      })
    }),
    Object.freeze({
      id: 'plinko_first_drop',
      title: 'First Starfall Drop',
      summary: 'Drop a Plinko ball and learn the prize-board side loop.',
      category: 'Collection',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'drop_plinko_ball', type: 'plinkoDrop', count: 1, label: 'Drop 1 Plinko ball' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ plinko_ball_basic: 2 }),
        materials: Object.freeze({ whiteStarCard: 2 })
      })
    }),
    Object.freeze({
      id: 'plinko_regular',
      title: 'Plinko Regular',
      summary: 'Use the Plinko board as a steady secondary grind target.',
      category: 'Collection',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'drop_plinko_25', type: 'plinkoDrop', count: 25, label: 'Drop 25 Plinko balls' })
      ]),
      rewards: Object.freeze({
        starTokens: 5,
        consumables: Object.freeze({ plinko_ball_polished: 2, plinko_ball_meteor: 1 }),
        materials: Object.freeze({ greenStarCard: 3, blueStarCard: 1 })
      })
    }),
    Object.freeze({
      id: 'trial_proven',
      title: 'Trial Proven',
      summary: 'Complete any class trial and prove a branch path is ready.',
      category: 'Class',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'complete_trial', type: 'trialComplete', count: 1, label: 'Complete 1 class trial' })
      ]),
      rewards: Object.freeze({
        currency: 240,
        consumables: Object.freeze({ base_skill_manual: 1 }),
        materials: Object.freeze({ upgradeDust: 8 })
      })
    }),
    Object.freeze({
      id: 'advanced_path',
      title: 'Advanced Path',
      summary: 'Choose any advanced class after completing its trial.',
      category: 'Class',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'advanced_class', type: 'advancedClass', count: 1, label: 'Choose an advanced class' })
      ]),
      rewards: Object.freeze({
        currency: 300,
        consumables: Object.freeze({ equipment_slot_coupon: 1, advanced_skill_manual: 1 }),
        permanentStats: Object.freeze({ mpMax: 20, resourceGain: 1 })
      })
    }),
    Object.freeze({
      id: 'party_ready',
      title: 'Party Ready',
      summary: 'Use the simulated party finder and bring allies into the field loop.',
      category: 'Class',
      tier: 'Bronze',
      objectives: Object.freeze([
        Object.freeze({ id: 'find_party', type: 'partyFind', count: 1, label: 'Find a prototype party' })
      ]),
      rewards: Object.freeze({
        consumables: Object.freeze({ guard_tonic: 2, swiftstep_oil: 1 }),
        materials: Object.freeze({ upgradeDust: 4 })
      })
    }),
    Object.freeze({
      id: 'level_20_candidate',
      title: 'Level 20 Candidate',
      summary: 'Reach the class-trial level band.',
      category: 'Class',
      tier: 'Silver',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_20', type: 'level', level: 20, label: 'Reach Level 20' })
      ]),
      rewards: Object.freeze({
        currency: 260,
        consumables: Object.freeze({ base_skill_manual: 1 }),
        materials: Object.freeze({ upgradeDust: 8, upgradeCatalyst: 1 })
      })
    }),
    Object.freeze({
      id: 'level_40_pathbreaker',
      title: 'Level 40 Pathbreaker',
      summary: 'Reach the mid-route gearing band.',
      category: 'Class',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_40', type: 'level', level: 40, label: 'Reach Level 40' })
      ]),
      rewards: Object.freeze({
        currency: 520,
        items: Object.freeze([Object.freeze({ itemId: 'sentinel_greaves', rarity: 'Rare', upgrade: 3, source: 'Level 40 Pathbreaker accomplishment' })]),
        consumables: Object.freeze({ advanced_skill_manual: 1 })
      })
    }),
    Object.freeze({
      id: 'level_60_vanguard',
      title: 'Level 60 Vanguard',
      summary: 'Reach the level 60 specialization milestone.',
      category: 'Class',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_cap', type: 'level', level: 60, label: 'Reach Level 60' })
      ]),
      rewards: Object.freeze({
        currency: 800,
        cosmeticId: 'vault_spark',
        permanentStats: Object.freeze({ hp: 60, mpMax: 25, power: 2, defense: 2 })
      })
    }),
    Object.freeze({
      id: 'level_100_legend',
      title: 'Level 100 Legend',
      summary: 'Push past the visible specialization band into long-term leveling.',
      category: 'Class',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_100', type: 'level', level: 100, label: 'Reach Level 100' })
      ]),
      rewards: Object.freeze({
        starTokens: 20,
        materials: Object.freeze({ upgradeDust: 60, refinementCore: 2, orangeStarCard: 1 }),
        consumables: Object.freeze({ xp_coupon_1_5_1h: 1 })
      })
    }),
    Object.freeze({
      id: 'bramblebreaker',
      title: 'Bramblebreaker',
      summary: 'Clear the Bramble Depths route boss dungeon.',
      category: 'Dungeon',
      tier: 'Gold',
      objectives: Object.freeze([
        Object.freeze({ id: 'bramble_depths', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 1, label: 'Clear Bramble Depths' })
      ]),
      rewards: Object.freeze({
        currency: 420,
        materials: Object.freeze({ upgradeDust: 10, upgradeCatalyst: 2 }),
        permanentStats: Object.freeze({ hp: 40, defense: 2 })
      })
    }),
    Object.freeze({
      id: 'dungeon_runner',
      title: 'Dungeon Runner',
      summary: 'Repeat dungeon clears enough that rewards and mechanics become familiar.',
      category: 'Dungeon',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'dungeon_clears_10', type: 'dungeonComplete', count: 10, label: 'Clear 10 dungeons' })
      ]),
      rewards: Object.freeze({
        starTokens: 10,
        materials: Object.freeze({ upgradeDust: 36, upgradeCatalyst: 6, refinementCore: 1 }),
        consumables: Object.freeze({ drop_coupon_1_2_1h: 1 })
      })
    }),
    Object.freeze({
      id: 'dungeon_master',
      title: 'Dungeon Master',
      summary: 'Clear every current dungeon at least once.',
      category: 'Dungeon',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'clear_bramble', type: 'dungeonComplete', dungeonId: 'bramble_depths', count: 1, label: 'Clear Bramble Depths' }),
        Object.freeze({ id: 'clear_emberjaw', type: 'dungeonComplete', dungeonId: 'emberjaw_lair', count: 1, label: 'Clear Emberjaw Lair' }),
        Object.freeze({ id: 'clear_gearworks', type: 'dungeonComplete', dungeonId: 'gearworks_vault', count: 1, label: 'Clear Gearworks Vault' }),
        Object.freeze({ id: 'clear_rimewarden', type: 'dungeonComplete', dungeonId: 'rimewarden_sanctum', count: 1, label: 'Clear Rimewarden Sanctum' })
      ]),
      rewards: Object.freeze({
        starTokens: 24,
        materials: Object.freeze({ upgradeDust: 64, upgradeCatalyst: 10, refinementCore: 3, purpleStarCard: 2 }),
        consumables: Object.freeze({ equipment_slot_coupon: 1, card_slot_coupon: 1 })
      })
    }),
    Object.freeze({
      id: 'bossbreaker',
      title: 'Bossbreaker',
      summary: 'Defeat several area bosses across the route chain.',
      category: 'Boss',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'bosses', type: 'defeatBoss', count: 3, label: 'Defeat 3 bosses' })
      ]),
      rewards: Object.freeze({
        materials: Object.freeze({ upgradeDust: 16, upgradeCatalyst: 4 }),
        timedBuffs: Object.freeze([Object.freeze({ buffId: 'swiftstepOil', duration: 45 })]),
        permanentStats: Object.freeze({ power: 3, crit: 1 })
      })
    }),
    Object.freeze({
      id: 'rimewarden_breaker',
      title: 'Rimewarden Breaker',
      summary: 'Bring the Frostfen dungeon boss down and claim its route prestige.',
      category: 'Boss',
      tier: 'Relic',
      objectives: Object.freeze([
        Object.freeze({ id: 'clear_rimewarden', type: 'dungeonComplete', dungeonId: 'rimewarden_sanctum', count: 1, label: 'Clear Rimewarden Sanctum' })
      ]),
      rewards: Object.freeze({
        starTokens: 8,
        materials: Object.freeze({ upgradeDust: 18, upgradeCatalyst: 3, rimewardenSigil: 1 }),
        permanentStats: Object.freeze({ hp: 50, defense: 2 })
      })
    }),
    Object.freeze({
      id: 'boss_reaper',
      title: 'Boss Reaper',
      summary: 'Make repeated boss clears a long-tail grind goal.',
      category: 'Boss',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'bosses_25', type: 'defeatBoss', count: 25, label: 'Defeat 25 bosses' })
      ]),
      rewards: Object.freeze({
        starTokens: 20,
        materials: Object.freeze({ upgradeDust: 72, upgradeCatalyst: 12, refinementCore: 3, orangeStarCard: 1 }),
        permanentStats: Object.freeze({ power: 3, crit: 1 })
      })
    }),
    Object.freeze({
      id: 'starfall_master',
      title: 'Starfall Master',
      summary: 'Tie leveling, bosses, collection, and attunement into one aspirational chase.',
      category: 'Mastery',
      tier: 'Mythic',
      objectives: Object.freeze([
        Object.freeze({ id: 'level_100', type: 'level', level: 100, label: 'Reach Level 100' }),
        Object.freeze({ id: 'bosses_50', type: 'defeatBoss', count: 50, label: 'Defeat 50 bosses' }),
        Object.freeze({ id: 'cards_75', type: 'loot', kind: 'card', count: 75, label: 'Loot 75 cards' }),
        Object.freeze({ id: 'plinko_50', type: 'plinkoDrop', count: 50, label: 'Drop 50 Plinko balls' }),
        Object.freeze({ id: 'relic_attune', type: 'itemPotential', minTier: 'relic', count: 1, label: 'Roll Relic or better attunement' })
      ]),
      rewards: Object.freeze({
        starTokens: 35,
        cosmeticId: 'ember_trim',
        materials: Object.freeze({ upgradeDust: 100, upgradeCatalyst: 16, refinementCore: 4, orangeStarCard: 2 }),
        consumables: Object.freeze({ xp_coupon_2_0_1h: 1, drop_coupon_2_0_1h: 1 }),
        permanentStats: Object.freeze({ hp: 100, mpMax: 40, power: 4, defense: 4 })
      })
    })
  ]);

  const api = {
    ACCOMPLISHMENTS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.accomplishments = Object.assign({}, modules.accomplishments || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
