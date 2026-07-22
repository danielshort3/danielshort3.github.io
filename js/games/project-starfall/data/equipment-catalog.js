(function initProjectStarfallDataEquipmentCatalog(global) {
  'use strict';

  function defaultAttachEquipmentItemAssets(item) {
    return Object.freeze(Object.assign({}, item || {}));
  }

  function createEquipmentCatalogData(options) {
    const settings = options || {};
    const attachEquipmentItemAssets = typeof settings.attachEquipmentItemAssets === 'function'
      ? settings.attachEquipmentItemAssets
      : defaultAttachEquipmentItemAssets;

    const SHOP_ITEMS = Object.freeze([
      { id: 'training_sword', name: 'Training Sword', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'fighter', stats: { power: 8 }, source: 'Starter Outfitter' },
      { id: 'training_wand', name: 'Training Wand', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'mage', stats: { power: 8 }, source: 'Starter Outfitter' },
      { id: 'training_bow', name: 'Training Bow', slot: 'weapon', rarity: 'Common', cost: 0, level: 1, classId: 'archer', stats: { power: 8 }, source: 'Starter Outfitter' },
      { id: 'copper_sword', name: 'Copper Sword', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'fighter', stats: { power: 16, speed: 4 }, source: 'Starter Outfitter' },
      { id: 'birch_wand', name: 'Birch Wand', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'mage', stats: { power: 15, speed: 4 }, source: 'Starter Outfitter' },
      { id: 'simple_bow', name: 'Simple Bow', slot: 'weapon', rarity: 'Common', cost: 85, level: 5, classId: 'archer', stats: { power: 15, resourceGain: 2 }, source: 'Starter Outfitter' },
      { id: 'stitched_vest', name: 'Stitched Vest', slot: 'chest', rarity: 'Common', cost: 70, level: 5, classId: 'any', stats: { defense: 6, hp: 24 }, source: 'Starter Outfitter' },
      { id: 'traveler_boots', name: 'Traveler Boots', slot: 'boots', rarity: 'Uncommon', cost: 90, level: 5, classId: 'any', stats: { speed: 12, defense: 2 }, source: 'Starter Outfitter' },
      { id: 'plain_ring', name: 'Plain Ring', slot: 'ring', rarity: 'Common', cost: 65, level: 5, classId: 'any', stats: { hp: 18 }, source: 'Starter Outfitter' },
      { id: 'iron_sword', name: 'Iron Sword', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'fighter', stats: { power: 34, speed: 4 }, source: 'Weapon Smith' },
      { id: 'iron_axe', name: 'Iron Axe', slot: 'weapon', rarity: 'Uncommon', cost: 320, level: 15, classId: 'fighter', stats: { power: 39, armorBreak: 4, speed: -4 }, source: 'Weapon Smith' },
      { id: 'apprentice_staff', name: 'Apprentice Staff', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'mage', stats: { power: 32, areaDamage: 4 }, source: 'Weapon Smith' },
      { id: 'oak_longbow', name: 'Oak Longbow', slot: 'weapon', rarity: 'Uncommon', cost: 260, level: 15, classId: 'archer', stats: { power: 31, critDamage: 5, range: 24 }, source: 'Weapon Smith' },
      { id: 'guardian_tower_shield', name: 'Guardian Tower Shield', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'guardian', stats: { defense: 40, hp: 180, block: 6 }, source: 'Class Supplier' },
      { id: 'berserker_war_grip', name: 'Berserker War Grip', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'berserker', stats: { power: 18, resourceGain: 6 }, source: 'Class Supplier' },
      { id: 'duelist_parry_medal', name: 'Duelist Parry Medal', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'duelist', stats: { power: 10, speed: 12, crit: 4 }, source: 'Class Supplier', visualId: 'deadeye_scope', assetId: 'deadeye_scope' },
      { id: 'ember_core', name: 'Ember Core', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'fireMage', stats: { power: 16, burnDamage: 8, resourceMax: 5 }, source: 'Class Supplier' },
      { id: 'rune_etched_focus', name: 'Rune-Etched Focus', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'runeMage', stats: { power: 14, runeDuration: 8, resourceGain: 4 }, source: 'Class Supplier' },
      { id: 'storm_charge_focus', name: 'Storm Charge Focus', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'stormMage', stats: { areaDamage: 7, resourceGain: 5, speed: 6 }, source: 'Class Supplier', visualId: 'rune_etched_focus', assetId: 'rune_etched_focus' },
      { id: 'deadeye_scope', name: 'Deadeye Scope', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'sniper', stats: { crit: 6, weakPointDuration: 8 }, source: 'Class Supplier' },
      { id: 'trap_kit', name: 'Trap Kit', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'trapper', stats: { trapSpeed: 10, trapDamage: 7 }, source: 'Class Supplier' },
      { id: 'beast_bond_charm', name: 'Beast Bond Charm', slot: 'offhand', rarity: 'Rare', cost: 520, level: 25, classId: 'beastArcher', stats: { hp: 95, resourceGain: 5, avoid: 3 }, source: 'Class Supplier', visualId: 'wanderer_charm', assetId: 'wanderer_charm' },
      { id: 'rustcoil_field_helm', name: 'Rustcoil Field Helm', slot: 'head', rarity: 'Common', cost: 210, level: 15, classId: 'any', stats: { hp: 54, defense: 8 }, source: 'Rustcoil Armorer', visualId: 'fieldguard_helm', assetId: 'fieldguard_helm' },
      { id: 'rustcoil_work_vest', name: 'Rustcoil Work Vest', slot: 'chest', rarity: 'Uncommon', cost: 280, level: 15, classId: 'any', stats: { hp: 80, defense: 12 }, source: 'Rustcoil Armorer', visualId: 'stitched_vest', assetId: 'stitched_vest' },
      { id: 'rustcoil_grip_gloves', name: 'Rustcoil Grip Gloves', slot: 'gloves', rarity: 'Common', cost: 230, level: 15, classId: 'any', stats: { power: 7, armorBreak: 2 }, source: 'Rustcoil Armorer', visualId: 'trailwoven_gloves', assetId: 'trailwoven_gloves' },
      { id: 'cinder_steel_sword', name: 'Cinder Steel Sword', slot: 'weapon', rarity: 'Uncommon', cost: 620, level: 25, classId: 'fighter', stats: { power: 52, armorBreak: 5, hp: 36 }, source: 'Cinder Weapon Shop', visualId: 'iron_sword', assetId: 'iron_sword' },
      { id: 'cinder_steel_scepter', name: 'Cinder Steel Scepter', slot: 'weapon', rarity: 'Uncommon', cost: 620, level: 25, classId: 'mage', stats: { power: 50, areaDamage: 6, mpMax: 30 }, source: 'Cinder Weapon Shop', visualId: 'apprentice_staff', assetId: 'apprentice_staff' },
      { id: 'cinder_ashwood_bow', name: 'Cinder Ashwood Bow', slot: 'weapon', rarity: 'Uncommon', cost: 620, level: 25, classId: 'archer', stats: { power: 49, critDamage: 8, range: 30 }, source: 'Cinder Weapon Shop', visualId: 'oak_longbow', assetId: 'oak_longbow' },
      { id: 'cinder_reinforced_mail', name: 'Cinder Reinforced Mail', slot: 'chest', rarity: 'Uncommon', cost: 560, level: 25, classId: 'any', stats: { hp: 130, defense: 18, burnDamage: 3 }, source: 'Cinder Armorer', visualId: 'furnaceplate', assetId: 'furnaceplate' },
      { id: 'cinder_forge_boots', name: 'Cinder Forge Boots', slot: 'boots', rarity: 'Uncommon', cost: 470, level: 25, classId: 'any', stats: { speed: 16, defense: 7, avoid: 2 }, source: 'Cinder Armorer', visualId: 'scorchtrail_boots', assetId: 'scorchtrail_boots' },
      { id: 'cinder_ember_band', name: 'Cinder Ember Band', slot: 'ring', rarity: 'Rare', cost: 720, level: 25, classId: 'any', stats: { power: 8, hp: 48, burnDamage: 4 }, source: 'Cinder Special Shop', visualId: 'plain_ring', assetId: 'plain_ring' },
      { id: 'frostfen_silver_saber', name: 'Frostfen Silver Saber', slot: 'weapon', rarity: 'Rare', cost: 1250, level: 40, classId: 'fighter', stats: { power: 78, speed: 8, block: 3 }, source: 'Frostfen Weapon Shop', visualId: 'thorncrown_greatsword', assetId: 'thorncrown_greatsword' },
      { id: 'frostfen_moonlit_staff', name: 'Frostfen Moonlit Staff', slot: 'weapon', rarity: 'Rare', cost: 1250, level: 40, classId: 'mage', stats: { power: 75, mpMax: 60, resourceGain: 6 }, source: 'Frostfen Weapon Shop', visualId: 'thornroot_staff', assetId: 'thornroot_staff' },
      { id: 'frostfen_frostpine_bow', name: 'Frostfen Frostpine Bow', slot: 'weapon', rarity: 'Rare', cost: 1250, level: 40, classId: 'archer', stats: { power: 73, range: 46, crit: 5 }, source: 'Frostfen Weapon Shop', visualId: 'briarstring_longbow', assetId: 'briarstring_longbow' },
      { id: 'frostfen_iceguard_coat', name: 'Frostfen Iceguard Coat', slot: 'chest', rarity: 'Rare', cost: 1120, level: 40, classId: 'any', stats: { hp: 210, defense: 28, avoid: 3 }, source: 'Frostfen Armorer', visualId: 'barkplate_harness', assetId: 'barkplate_harness' },
      { id: 'frostfen_snowstep_boots', name: 'Frostfen Snowstep Boots', slot: 'boots', rarity: 'Rare', cost: 940, level: 40, classId: 'any', stats: { speed: 24, defense: 11, avoid: 5 }, source: 'Frostfen Armorer', visualId: 'rootstep_greaves', assetId: 'rootstep_greaves' },
      { id: 'frostfen_rime_ring', name: 'Frostfen Rime Ring', slot: 'ring', rarity: 'Rare', cost: 980, level: 40, classId: 'any', stats: { hp: 92, defense: 10, mpMax: 34 }, source: 'Frostfen Special Shop', visualId: 'plain_ring', assetId: 'plain_ring' },
      { id: 'stormbreak_stormforged_blade', name: 'Stormforged Blade', slot: 'weapon', rarity: 'Rare', cost: 2200, level: 55, classId: 'fighter', stats: { power: 102, speed: 14, crit: 5 }, source: 'Stormbreak Weapon Shop', visualId: 'stormtalon_saber', assetId: 'stormtalon_saber' },
      { id: 'stormbreak_thunder_rod', name: 'Thunder Rod', slot: 'weapon', rarity: 'Rare', cost: 2200, level: 55, classId: 'mage', stats: { power: 98, areaDamage: 12, resourceGain: 7 }, source: 'Stormbreak Weapon Shop', visualId: 'cloudspine_rod', assetId: 'cloudspine_rod' },
      { id: 'stormbreak_gale_longbow', name: 'Gale Longbow', slot: 'weapon', rarity: 'Rare', cost: 2200, level: 55, classId: 'archer', stats: { power: 96, range: 62, critDamage: 15 }, source: 'Stormbreak Weapon Shop', visualId: 'skybreaker_bow', assetId: 'skybreaker_bow' },
      { id: 'stormbreak_tempest_mantle', name: 'Stormbreak Tempest Mantle', slot: 'chest', rarity: 'Rare', cost: 1960, level: 55, classId: 'any', stats: { hp: 260, defense: 34, areaDamage: 8 }, source: 'Stormbreak Armorer', visualId: 'tempest_mantle', assetId: 'tempest_mantle' },
      { id: 'stormbreak_cloudrunner_boots', name: 'Cloudrunner Boots', slot: 'boots', rarity: 'Rare', cost: 1680, level: 55, classId: 'any', stats: { speed: 34, avoid: 8, range: 14 }, source: 'Stormbreak Armorer', visualId: 'gale_boots', assetId: 'gale_boots' },
      { id: 'stormbreak_lightning_charm', name: 'Lightning Charm', slot: 'amulet', rarity: 'Rare', cost: 1800, level: 55, classId: 'any', stats: { crit: 5, speed: 12, resourceGain: 5 }, source: 'Stormbreak Special Shop', visualId: 'wanderer_charm', assetId: 'wanderer_charm' },
      { id: 'astral_index_blade', name: 'Astral Index Blade', slot: 'weapon', rarity: 'Rare', cost: 3600, level: 70, classId: 'fighter', stats: { power: 124, resourceGain: 9, crit: 8 }, source: 'Astral Weapon Shop', visualId: 'index_blade', assetId: 'index_blade' },
      { id: 'astral_star_lens_staff', name: 'Star Lens Staff', slot: 'weapon', rarity: 'Rare', cost: 3600, level: 70, classId: 'mage', stats: { power: 121, mpMax: 100, areaDamage: 16 }, source: 'Astral Weapon Shop', visualId: 'umbral_starstaff', assetId: 'umbral_starstaff' },
      { id: 'astral_comet_bow', name: 'Cometstring Training Bow', slot: 'weapon', rarity: 'Rare', cost: 3600, level: 70, classId: 'archer', stats: { power: 118, range: 80, crit: 10 }, source: 'Astral Weapon Shop', visualId: 'cometstring_bow', assetId: 'cometstring_bow' },
      { id: 'astral_starwoven_robes', name: 'Starwoven Robes', slot: 'chest', rarity: 'Rare', cost: 3200, level: 70, classId: 'any', stats: { hp: 310, mpMax: 90, defense: 38 }, source: 'Astral Armorer', visualId: 'astral_robes', assetId: 'astral_robes' },
      { id: 'astral_orbitstep_boots', name: 'Orbitstep Boots', slot: 'boots', rarity: 'Rare', cost: 2700, level: 70, classId: 'any', stats: { speed: 32, range: 24, avoid: 7 }, source: 'Astral Armorer', visualId: 'orbit_boots', assetId: 'orbit_boots' },
      { id: 'astral_lens_amulet', name: 'Star Lens Amulet', slot: 'amulet', rarity: 'Rare', cost: 3000, level: 70, classId: 'any', stats: { mpMax: 80, resourceMax: 12, crit: 5 }, source: 'Astral Special Shop', visualId: 'wanderer_charm', assetId: 'wanderer_charm' }
    ].map(attachEquipmentItemAssets));

    const RANDOM_EQUIPMENT_ITEMS = Object.freeze([
      { id: 'adventurer_cutlass', name: 'Adventurer Cutlass', slot: 'weapon', rarity: 'Common', level: 8, classId: 'fighter', stats: { power: 18, speed: 3 }, source: 'World drop', visualId: 'copper_sword', assetId: 'adventurer_cutlass', dropOnly: true },
      { id: 'balanced_focus', name: 'Balanced Focus', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'mage', stats: { power: 32, resourceGain: 3 }, source: 'World drop', visualId: 'birch_wand', assetId: 'balanced_focus', dropOnly: true },
      { id: 'wanderer_charm', name: 'Wanderer Charm', slot: 'amulet', rarity: 'Uncommon', level: 12, classId: 'any', stats: { hp: 36, resourceGain: 2 }, source: 'World drop', assetId: 'wanderer_charm', dropOnly: true },
      { id: 'fieldguard_helm', name: 'Fieldguard Helm', slot: 'head', rarity: 'Common', level: 10, classId: 'any', stats: { hp: 34, defense: 5 }, source: 'World drop', assetId: 'fieldguard_helm', dropOnly: true },
      { id: 'trailwoven_gloves', name: 'Trailwoven Gloves', slot: 'gloves', rarity: 'Common', level: 14, classId: 'any', stats: { power: 6, speed: 5 }, source: 'World drop', assetId: 'trailwoven_gloves', dropOnly: true },

      { id: 'vanguard_blade', name: 'Vanguard Blade', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'fighter', stats: { power: 38, hp: 42, armorBreak: 3 }, source: 'World drop', visualId: 'iron_sword', assetId: 'vanguard_blade', dropOnly: true },
      { id: 'bulwark_plate', name: 'Bulwark Plate', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'fighter', stats: { hp: 105, defense: 16, block: 3 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'bulwark_plate', dropOnly: true },
      { id: 'breaker_gauntlets', name: 'Breaker Gauntlets', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'fighter', stats: { power: 18, armorBreak: 8, defense: 6 }, source: 'World drop', assetId: 'breaker_gauntlets', dropOnly: true },
      { id: 'sentinel_greaves', name: 'Sentinel Greaves', slot: 'boots', rarity: 'Rare', level: 34, classId: 'fighter', stats: { hp: 84, defense: 13, block: 4 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'sentinel_greaves', dropOnly: true },

      { id: 'starglass_staff', name: 'Starglass Staff', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'mage', stats: { power: 36, mpMax: 36, areaDamage: 4 }, source: 'World drop', visualId: 'apprentice_staff', assetId: 'starglass_staff', dropOnly: true },
      { id: 'runewoven_robes', name: 'Runewoven Robes', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'mage', stats: { mpMax: 70, defense: 9, resourceGain: 4 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'runewoven_robes', dropOnly: true },
      { id: 'channeler_gloves', name: 'Channeler Gloves', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'mage', stats: { power: 15, areaDamage: 7, resourceMax: 8 }, source: 'World drop', assetId: 'channeler_gloves', dropOnly: true },
      { id: 'aetherstep_boots', name: 'Aetherstep Boots', slot: 'boots', rarity: 'Rare', level: 34, classId: 'mage', stats: { speed: 20, mpMax: 48, resourceGain: 5 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'aetherstep_boots', dropOnly: true },

      { id: 'ranger_recurve', name: 'Ranger Recurve', slot: 'weapon', rarity: 'Uncommon', level: 18, classId: 'archer', stats: { power: 35, range: 34, crit: 3 }, source: 'World drop', visualId: 'oak_longbow', assetId: 'ranger_recurve', dropOnly: true },
      { id: 'pathfinder_leathers', name: 'Pathfinder Leathers', slot: 'chest', rarity: 'Uncommon', level: 20, classId: 'archer', stats: { hp: 64, defense: 10, speed: 9 }, source: 'World drop', visualId: 'stitched_vest', assetId: 'pathfinder_leathers', dropOnly: true },
      { id: 'deadeye_wraps', name: 'Deadeye Wraps', slot: 'gloves', rarity: 'Rare', level: 28, classId: 'archer', stats: { power: 14, critDamage: 10, range: 18 }, source: 'World drop', assetId: 'deadeye_wraps', dropOnly: true },
      { id: 'windrunner_boots', name: 'Windrunner Boots', slot: 'boots', rarity: 'Rare', level: 34, classId: 'archer', stats: { speed: 28, avoid: 6, crit: 4 }, source: 'World drop', visualId: 'traveler_boots', assetId: 'windrunner_boots', dropOnly: true }
    ].map(attachEquipmentItemAssets));

    const BOSS_EQUIPMENT_SOURCES = Object.freeze([
      Object.freeze({ bossId: 'brambleking', name: 'Brambleking, Crowned Root', level: 35, rarity: 'Epic', setId: 'thorncrown_regalia', dropChance: 0.1 }),
      Object.freeze({ bossId: 'emberjawGolem', name: 'Emberjaw Prime', level: 45, rarity: 'Epic', setId: 'furnaceheart_arsenal', dropChance: 0.1 }),
      Object.freeze({ bossId: 'clockworkTitan', name: 'Clockwork Titan Mk II', level: 55, rarity: 'Epic', setId: 'titanwork_aegis', dropChance: 0.1 }),
      Object.freeze({ bossId: 'quarryColossus', name: 'Quarry Colossus, Deepcore Awakened', level: 65, rarity: 'Relic', setId: 'deepcore_colossus', dropChance: 0.06 }),
      Object.freeze({ bossId: 'stormbreakRoc', name: 'Aurelion, Stormbreak Roc', level: 78, rarity: 'Relic', setId: 'stormcaller_tempest', dropChance: 0.06 }),
      Object.freeze({ bossId: 'astralArchivist', name: 'The Astral Archivist', level: 92, rarity: 'Relic', setId: 'astral_index', dropChance: 0.06 }),
      Object.freeze({ bossId: 'eclipseSovereign', name: 'Eclipse Sovereign', level: 105, rarity: 'Relic', setId: 'eclipse_paragon', dropChance: 0.06 })
    ]);

    const DROP_ECONOMY = Object.freeze({
      normalDropChance: 0.08,
      eliteDropChance: 0.45,
      bossLootChance: 0.7,
      equipmentLevelLeeway: 8,
      bossPity: Object.freeze({
        epicStart: 8,
        epicStep: 0.01,
        epicMax: 0.1,
        relicStart: 10,
        relicStep: 0.0075,
        relicMax: 0.09
      }),
  	    globalRareChance: Object.freeze({
  	      normal: 0.0025,
  	      elite: 0.0075,
  	      boss: 0.015,
  	      specialElite: 0.02
  	    }),
  		    dropTableChances: Object.freeze({
  		      coins: 0.5,
  		      potions: 0.12,
  		      equipment: 0.05,
  		      bonusMaterials: 0.06,
  		      cards: 0.035,
  		      plinkoBalls: 0.04
  		    }),
  		    dropTableCaps: Object.freeze({
  		      coins: 0.95,
  		      potions: 0.6,
  		      equipment: 0.4,
  		      bonusMaterials: 0.4,
  		      cards: 0.25,
  		      plinkoBalls: 0.65,
  		      rareValuables: 0.1
  		    }),
  	    classWeights: Object.freeze({
  	      currentClass: 4,
  	      universal: 3,
  	      offClass: 1
  	    }),
      bossPieceWeights: Object.freeze({
        missing: 6,
        duplicate: 1
      }),
      lootWeights: Object.freeze({
        equipment: 7,
        rareEquipment: 11,
        primaryEtc: 40,
        secondaryEtc: 9,
        upgradeDust: 6,
        upgradeDustBoosted: 9,
        currency: 42,
        currencyBurst: 58,
        healthPotion: 20,
        resourceTonic: 20,
        campRation: 12,
        townReturnScroll: 5,
        guardTonic: 8,
        swiftstepOil: 8,
  	      magnetCharm: 6,
  	      xpCoupon12: 2,
  	      dropCoupon12: 2,
  	      eliteXpCoupon15: 2,
  	      eliteDropCoupon15: 2,
  	      bossXpCoupon20: 1,
  	      bossDropCoupon20: 1,
  	      mimicXpCoupon20: 2,
  	      mimicDropCoupon20: 2,
  	      skillManual: 4,
  	      eliteSkillManual: 7,
  	      skillReset: 1,
        eliteSkillReset: 2,
        attunementPrism: 1,
        eliteAttunementPrism: 2,
        echoPrism: 1,
        eliteEchoPrism: 1,
        gelDrop: 40,
        oreChunks: 40,
        upgradeCatalyst: 5,
        wardingScroll: 1,
        eliteWardingScroll: 2,
        refinementCore: 1,
        eliteRefinementCore: 2,
  	      equipmentSlotCoupon: 1,
  	      eliteEquipmentSlotCoupon: 2,
  	      mimicEquipmentSlotCoupon: 3,
  	      usableSlotCoupon: 1,
  	      eliteUsableSlotCoupon: 2,
  	      mimicUsableSlotCoupon: 3,
  	      etcSlotCoupon: 1,
  	      eliteEtcSlotCoupon: 2,
  	      mimicEtcSlotCoupon: 3,
  	      cardSlotCoupon: 1,
  	      eliteCardSlotCoupon: 2,
  	      mimicCardSlotCoupon: 3
  	    })
  	  });

    function bossGearItem(config) {
      const source = BOSS_EQUIPMENT_SOURCES.find((entry) => entry.setId === config.setId) || {};
      return attachEquipmentItemAssets(Object.assign({
        rarity: source.rarity || 'Epic',
        level: source.level || 35,
        cost: 0,
        source: `${source.name || 'Boss'} drop`,
        bossId: source.bossId || config.bossId || '',
        dropOnly: true
      }, config));
    }

    const BOSS_EQUIPMENT_ITEMS = Object.freeze([
      bossGearItem({ id: 'thorncrown_greatsword', name: 'Thorncrown Greatsword', slot: 'weapon', classId: 'fighter', setId: 'thorncrown_regalia', visualId: 'iron_sword', stats: { power: 58, armorBreak: 7, hp: 80 } }),
      bossGearItem({ id: 'thornroot_staff', name: 'Thornroot Staff', slot: 'weapon', classId: 'mage', setId: 'thorncrown_regalia', visualId: 'apprentice_staff', stats: { power: 56, areaDamage: 8, resourceGain: 4 } }),
      bossGearItem({ id: 'briarstring_longbow', name: 'Briarstring Longbow', slot: 'weapon', classId: 'archer', setId: 'thorncrown_regalia', visualId: 'oak_longbow', stats: { power: 55, range: 38, crit: 5 } }),
      bossGearItem({ id: 'briar_crown', name: 'Briar Crown', slot: 'head', classId: 'any', setId: 'thorncrown_regalia', stats: { hp: 70, defense: 8, resourceGain: 3 } }),
      bossGearItem({ id: 'barkplate_harness', name: 'Barkplate Harness', slot: 'chest', classId: 'any', setId: 'thorncrown_regalia', visualId: 'stitched_vest', stats: { hp: 145, defense: 18 } }),
      bossGearItem({ id: 'grasping_thorn_gloves', name: 'Grasping Thorn Gloves', slot: 'gloves', classId: 'any', setId: 'thorncrown_regalia', stats: { power: 11, armorBreak: 6, crit: 3 } }),
      bossGearItem({ id: 'rootstep_greaves', name: 'Rootstep Greaves', slot: 'boots', classId: 'any', setId: 'thorncrown_regalia', visualId: 'traveler_boots', stats: { speed: 18, defense: 8, avoid: 3 } }),

      bossGearItem({ id: 'emberjaw_cleaver', name: 'Emberjaw Cleaver', slot: 'weapon', classId: 'fighter', setId: 'furnaceheart_arsenal', visualId: 'iron_axe', stats: { power: 72, burnDamage: 10, armorBreak: 8 } }),
      bossGearItem({ id: 'magma_scepter', name: 'Magma Scepter', slot: 'weapon', classId: 'mage', setId: 'furnaceheart_arsenal', visualId: 'apprentice_staff', stats: { power: 70, burnDamage: 16, areaDamage: 9 } }),
      bossGearItem({ id: 'cindercoil_bow', name: 'Cindercoil Bow', slot: 'weapon', classId: 'archer', setId: 'furnaceheart_arsenal', visualId: 'oak_longbow', stats: { power: 68, burnDamage: 8, critDamage: 12, range: 34 } }),
      bossGearItem({ id: 'ashen_jaw_helm', name: 'Ashen Jaw Helm', slot: 'head', classId: 'any', setId: 'furnaceheart_arsenal', stats: { power: 10, burnDamage: 8, defense: 8 } }),
      bossGearItem({ id: 'furnaceplate', name: 'Furnaceplate', slot: 'chest', classId: 'any', setId: 'furnaceheart_arsenal', visualId: 'stitched_vest', stats: { hp: 170, defense: 22, burnDamage: 6 } }),
      bossGearItem({ id: 'lavaforged_gauntlets', name: 'Lavaforged Gauntlets', slot: 'gloves', classId: 'any', setId: 'furnaceheart_arsenal', stats: { power: 16, critDamage: 10, burnDamage: 6 } }),
      bossGearItem({ id: 'scorchtrail_boots', name: 'Scorchtrail Boots', slot: 'boots', classId: 'any', setId: 'furnaceheart_arsenal', visualId: 'traveler_boots', stats: { speed: 22, areaDamage: 5, avoid: 4 } }),

      bossGearItem({ id: 'gearcleaver', name: 'Gearcleaver', slot: 'weapon', classId: 'fighter', setId: 'titanwork_aegis', visualId: 'iron_axe', stats: { power: 86, armorBreak: 15, block: 4 } }),
      bossGearItem({ id: 'chrono_staff', name: 'Chrono Staff', slot: 'weapon', classId: 'mage', setId: 'titanwork_aegis', visualId: 'apprentice_staff', stats: { power: 82, resourceGain: 8, armorBreak: 8, areaDamage: 8 } }),
      bossGearItem({ id: 'ratchet_repeater', name: 'Ratchet Repeater', slot: 'weapon', classId: 'archer', setId: 'titanwork_aegis', visualId: 'oak_longbow', stats: { power: 80, crit: 8, armorBreak: 8, range: 42 } }),
      bossGearItem({ id: 'titan_visor', name: 'Titan Visor', slot: 'head', classId: 'any', setId: 'titanwork_aegis', stats: { defense: 18, crit: 4, armorBreak: 5 } }),
      bossGearItem({ id: 'clockplate_harness', name: 'Clockplate Harness', slot: 'chest', classId: 'any', setId: 'titanwork_aegis', visualId: 'stitched_vest', stats: { hp: 240, defense: 34, block: 4 } }),
      bossGearItem({ id: 'gyro_gauntlets', name: 'Gyro Gauntlets', slot: 'gloves', classId: 'any', setId: 'titanwork_aegis', stats: { power: 20, armorBreak: 9, resourceGain: 4 } }),
      bossGearItem({ id: 'springstep_boots', name: 'Springstep Boots', slot: 'boots', classId: 'any', setId: 'titanwork_aegis', visualId: 'traveler_boots', stats: { speed: 26, defense: 12, avoid: 5 } }),

      bossGearItem({ id: 'colossus_maul', name: 'Colossus Maul', slot: 'weapon', classId: 'fighter', setId: 'deepcore_colossus', visualId: 'iron_axe', stats: { power: 106, armorBreak: 20, critDamage: 14, speed: -4 } }),
      bossGearItem({ id: 'geode_scepter', name: 'Geode Scepter', slot: 'weapon', classId: 'mage', setId: 'deepcore_colossus', visualId: 'apprentice_staff', stats: { power: 102, areaDamage: 15, armorBreak: 12, resourceMax: 12 } }),
      bossGearItem({ id: 'oreline_greatbow', name: 'Oreline Greatbow', slot: 'weapon', classId: 'archer', setId: 'deepcore_colossus', visualId: 'oak_longbow', stats: { power: 100, critDamage: 22, armorBreak: 12, range: 50 } }),
      bossGearItem({ id: 'deepcore_helm', name: 'Deepcore Helm', slot: 'head', classId: 'any', setId: 'deepcore_colossus', stats: { hp: 140, defense: 24, block: 4 } }),
      bossGearItem({ id: 'bedrock_plate', name: 'Bedrock Plate', slot: 'chest', classId: 'any', setId: 'deepcore_colossus', visualId: 'stitched_vest', stats: { hp: 330, defense: 46, armorBreak: 6 } }),
      bossGearItem({ id: 'quarry_fists', name: 'Quarry Fists', slot: 'gloves', classId: 'any', setId: 'deepcore_colossus', stats: { power: 28, armorBreak: 12, critDamage: 10 } }),
      bossGearItem({ id: 'stonewake_boots', name: 'Stonewake Boots', slot: 'boots', classId: 'any', setId: 'deepcore_colossus', visualId: 'traveler_boots', stats: { speed: 18, defense: 22, hp: 90 } }),

      bossGearItem({ id: 'stormtalon_saber', name: 'Stormtalon Saber', slot: 'weapon', classId: 'fighter', setId: 'stormcaller_tempest', visualId: 'iron_sword', stats: { power: 122, speed: 22, crit: 10, avoid: 5 } }),
      bossGearItem({ id: 'cloudspine_rod', name: 'Cloudspine Rod', slot: 'weapon', classId: 'mage', setId: 'stormcaller_tempest', visualId: 'apprentice_staff', stats: { power: 118, areaDamage: 18, speed: 18, resourceGain: 8 } }),
      bossGearItem({ id: 'skybreaker_bow', name: 'Skybreaker Bow', slot: 'weapon', classId: 'archer', setId: 'stormcaller_tempest', visualId: 'oak_longbow', stats: { power: 116, range: 72, crit: 12, critDamage: 18 } }),
      bossGearItem({ id: 'rocfeather_mask', name: 'Rocfeather Mask', slot: 'head', classId: 'any', setId: 'stormcaller_tempest', stats: { speed: 18, crit: 5, avoid: 6 } }),
      bossGearItem({ id: 'tempest_mantle', name: 'Tempest Mantle', slot: 'chest', classId: 'any', setId: 'stormcaller_tempest', visualId: 'stitched_vest', stats: { hp: 260, defense: 34, areaDamage: 8 } }),
      bossGearItem({ id: 'lightning_grip_gloves', name: 'Lightning-Grip Gloves', slot: 'gloves', classId: 'any', setId: 'stormcaller_tempest', stats: { power: 32, crit: 7, resourceGain: 6 } }),
      bossGearItem({ id: 'gale_boots', name: 'Gale Boots', slot: 'boots', classId: 'any', setId: 'stormcaller_tempest', visualId: 'traveler_boots', stats: { speed: 38, avoid: 9, range: 24 } }),

      bossGearItem({ id: 'index_blade', name: 'Index Blade', slot: 'weapon', classId: 'fighter', setId: 'astral_index', visualId: 'iron_sword', stats: { power: 138, resourceGain: 12, crit: 10, range: 18 } }),
      bossGearItem({ id: 'starbound_codex', name: 'Starbound Codex', slot: 'weapon', classId: 'mage', setId: 'astral_index', visualId: 'apprentice_staff', stats: { power: 136, mpMax: 120, resourceMax: 24, areaDamage: 20 } }),
      bossGearItem({ id: 'cometstring_bow', name: 'Cometstring Bow', slot: 'weapon', classId: 'archer', setId: 'astral_index', visualId: 'oak_longbow', stats: { power: 132, range: 88, crit: 13, resourceGain: 10 } }),
      bossGearItem({ id: 'archivist_crown', name: 'Archivist Crown', slot: 'head', classId: 'any', setId: 'astral_index', stats: { mpMax: 80, resourceMax: 14, crit: 6 } }),
      bossGearItem({ id: 'astral_robes', name: 'Astral Robes', slot: 'chest', classId: 'any', setId: 'astral_index', visualId: 'stitched_vest', stats: { hp: 280, mpMax: 100, defense: 36, areaDamage: 8 } }),
      bossGearItem({ id: 'scribe_gloves', name: 'Scribe Gloves', slot: 'gloves', classId: 'any', setId: 'astral_index', stats: { power: 36, resourceGain: 9, areaDamage: 8 } }),
      bossGearItem({ id: 'orbit_boots', name: 'Orbit Boots', slot: 'boots', classId: 'any', setId: 'astral_index', visualId: 'traveler_boots', stats: { speed: 30, range: 30, avoid: 7 } }),

      bossGearItem({ id: 'eclipse_edge', name: 'Eclipse Edge', slot: 'weapon', classId: 'fighter', setId: 'eclipse_paragon', visualId: 'iron_sword', stats: { power: 160, crit: 14, critDamage: 28, resourceGain: 10 } }),
      bossGearItem({ id: 'umbral_starstaff', name: 'Umbral Starstaff', slot: 'weapon', classId: 'mage', setId: 'eclipse_paragon', visualId: 'apprentice_staff', stats: { power: 156, areaDamage: 26, crit: 10, resourceMax: 26 } }),
      bossGearItem({ id: 'corona_longbow', name: 'Corona Longbow', slot: 'weapon', classId: 'archer', setId: 'eclipse_paragon', visualId: 'oak_longbow', stats: { power: 154, range: 100, crit: 16, critDamage: 32 } }),
      bossGearItem({ id: 'sovereign_crown', name: 'Sovereign Crown', slot: 'head', classId: 'any', setId: 'eclipse_paragon', stats: { power: 28, crit: 8, defense: 20 } }),
      bossGearItem({ id: 'eclipse_plate', name: 'Eclipse Plate', slot: 'chest', classId: 'any', setId: 'eclipse_paragon', visualId: 'stitched_vest', stats: { hp: 460, defense: 60, power: 18 } }),
      bossGearItem({ id: 'penumbra_gloves', name: 'Penumbra Gloves', slot: 'gloves', classId: 'any', setId: 'eclipse_paragon', stats: { power: 42, critDamage: 22, resourceGain: 8 } }),
      bossGearItem({ id: 'sunfall_boots', name: 'Sunfall Boots', slot: 'boots', classId: 'any', setId: 'eclipse_paragon', visualId: 'traveler_boots', stats: { speed: 34, avoid: 10, areaDamage: 10 } })
    ]);

    return Object.freeze({
      SHOP_ITEMS,
      RANDOM_EQUIPMENT_ITEMS,
      BOSS_EQUIPMENT_SOURCES,
      DROP_ECONOMY,
      BOSS_EQUIPMENT_ITEMS
    });
  }

  const defaultEquipmentCatalogData = createEquipmentCatalogData();

  const api = Object.assign({
    defaultAttachEquipmentItemAssets,
    createEquipmentCatalogData
  }, defaultEquipmentCatalogData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentCatalog = Object.assign({}, modules.equipmentCatalog || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
