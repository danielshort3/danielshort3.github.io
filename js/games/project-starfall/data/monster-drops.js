(function initProjectStarfallDataMonsterDrops(global) {
  'use strict';

  function monsterDropEntry(type, id, weight, options) {
    const settings = options || {};
    const entry = {
      type,
      weight: Math.max(1, Math.round(Number(weight || 1)))
    };
    if (type === 'material') entry.materialId = id;
    else if (type === 'equipment') entry.itemId = id;
    else if (type === 'consumable') entry.consumableId = id;
    else if (type === 'card') entry.cardId = id;
    else entry.id = id;
    if (settings.minQuantity != null) entry.minQuantity = Math.max(1, Math.floor(Number(settings.minQuantity) || 1));
    if (settings.maxQuantity != null) entry.maxQuantity = Math.max(entry.minQuantity || 1, Math.floor(Number(settings.maxQuantity) || 1));
    if (settings.rarity) entry.rarity = settings.rarity;
    if (settings.groupId) entry.groupId = settings.groupId;
    return Object.freeze(entry);
  }

  const materialDrop = (id, weight, options) => monsterDropEntry('material', id, weight, options);
  const equipmentDrop = (id, weight, options) => monsterDropEntry('equipment', id, weight, options);
  const cardDrop = (id, weight, options) => monsterDropEntry('card', id, weight, options);
  const STAR_CARD_MATERIAL_IDS = Object.freeze({
    white: 'whiteStarCard',
    green: 'greenStarCard',
    blue: 'blueStarCard',
    purple: 'purpleStarCard',
    orange: 'orangeStarCard'
  });
  const STAR_CARD_DROP_TABLES = Object.freeze({
    early: Object.freeze([
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.white, weight: 5 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.green, weight: 1 })
    ]),
    mid: Object.freeze([
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.white, weight: 4 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.green, weight: 3 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.blue, weight: 1 })
    ]),
    late: Object.freeze([
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.green, weight: 4 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.blue, weight: 3 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.purple, weight: 1 })
    ]),
    boss: Object.freeze([
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.green, weight: 4 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.blue, weight: 3 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.purple, weight: 2 }),
      Object.freeze({ materialId: STAR_CARD_MATERIAL_IDS.orange, weight: 1 })
    ])
  });

  const MONSTER_EQUIPMENT_DROP_GROUPS = Object.freeze({
    training: Object.freeze(['adventurer_cutlass', 'birch_wand', 'simple_bow', 'fieldguard_helm', 'trailwoven_gloves']),
    traveler: Object.freeze(['traveler_boots', 'wanderer_charm']),
    focus: Object.freeze(['balanced_focus', 'starglass_staff', 'wanderer_charm', 'channeler_gloves']),
    guard: Object.freeze(['fieldguard_helm', 'bulwark_plate', 'sentinel_greaves']),
    sharp: Object.freeze(['ranger_recurve', 'deadeye_wraps', 'vanguard_blade']),
    steel: Object.freeze(['vanguard_blade', 'iron_sword', 'iron_axe']),
    support: Object.freeze(['wanderer_charm', 'runewoven_robes', 'channeler_gloves']),
    rustcoil: Object.freeze(['balanced_focus', 'breaker_gauntlets', 'sentinel_greaves', 'vanguard_blade']),
    cinder: Object.freeze(['ember_core', 'breaker_gauntlets', 'channeler_gloves', 'aetherstep_boots']),
    frost: Object.freeze(['aetherstep_boots', 'sentinel_greaves', 'windrunner_boots', 'runewoven_robes']),
    storm: Object.freeze(['windrunner_boots', 'deadeye_wraps', 'aetherstep_boots', 'ranger_recurve']),
    astral: Object.freeze(['starglass_staff', 'runewoven_robes', 'channeler_gloves', 'aetherstep_boots']),
    eclipse: Object.freeze(['breaker_gauntlets', 'sentinel_greaves', 'deadeye_wraps', 'windrunner_boots']),
    rare: Object.freeze(['breaker_gauntlets', 'sentinel_greaves', 'channeler_gloves', 'deadeye_wraps', 'windrunner_boots'])
  });

  function equipmentDrops(groupId, weight) {
    return (MONSTER_EQUIPMENT_DROP_GROUPS[groupId] || []).map((itemId) => equipmentDrop(itemId, weight));
  }

  const MONSTER_CARD_DROP_GROUPS = Object.freeze({
    forest: Object.freeze(['gel_spark', 'mossguard_oath', 'thorn_focus']),
    beast: Object.freeze(['bristle_charge', 'mossguard_oath', 'hunter_tempo']),
    construct: Object.freeze(['clockwork_patience', 'rustcoil_lens', 'titan_gearheart']),
    cinder: Object.freeze(['ember_glint', 'ashflare_core', 'stormbreak_plume']),
    bandit: Object.freeze(['hunter_tempo', 'storm_fletching', 'rift_splinter']),
    frost: Object.freeze(['frost_thread', 'mossguard_oath', 'cloudcall_vellum']),
    storm: Object.freeze(['storm_fletching', 'cloudcall_vellum', 'stormbreak_plume']),
    astral: Object.freeze(['astral_index', 'cloudcall_vellum', 'archivist_star']),
    eclipse: Object.freeze(['rift_splinter', 'eclipse_corona', 'archivist_star']),
    mimic: Object.freeze(['mimic_cache', 'rift_splinter', 'titan_gearheart']),
    bossForest: Object.freeze(['bramble_heart', 'mossguard_oath', 'thorn_focus']),
    bossConstruct: Object.freeze(['titan_gearheart', 'clockwork_patience', 'rustcoil_lens']),
    bossCinder: Object.freeze(['ashflare_core', 'ember_glint', 'stormbreak_plume']),
    bossFrost: Object.freeze(['frost_thread', 'bramble_heart', 'cloudcall_vellum']),
    bossStorm: Object.freeze(['stormbreak_plume', 'storm_fletching', 'cloudcall_vellum']),
    bossAstral: Object.freeze(['archivist_star', 'astral_index', 'cloudcall_vellum']),
    bossEclipse: Object.freeze(['eclipse_corona', 'rift_splinter', 'archivist_star'])
  });

  function cardDrops(groupId, weight) {
    return (MONSTER_CARD_DROP_GROUPS[groupId] || []).map((cardId) => cardDrop(cardId, weight, { groupId }));
  }

  function getStarCardDropTier(cardEntries) {
    const groups = new Set((cardEntries || []).map((entry) => entry && entry.groupId).filter(Boolean));
    if (['bossForest', 'bossConstruct', 'bossCinder', 'bossFrost', 'bossStorm', 'bossAstral', 'bossEclipse'].some((groupId) => groups.has(groupId))) return 'boss';
    if (['astral', 'eclipse', 'mimic'].some((groupId) => groups.has(groupId))) return 'late';
    if (['construct', 'cinder', 'bandit', 'frost', 'storm'].some((groupId) => groups.has(groupId))) return 'mid';
    return 'early';
  }

  function starCardDrops(tierId) {
    const table = STAR_CARD_DROP_TABLES[tierId] || STAR_CARD_DROP_TABLES.early;
    return table.map((entry) => materialDrop(entry.materialId, entry.weight));
  }

  function monsterDropPool(config) {
    const source = config || {};
    const cardEntries = (source.cards || []).slice();
    const starCards = source.starCards === false
      ? []
      : Array.isArray(source.starCards)
        ? source.starCards.slice()
        : cardEntries.length ? starCardDrops(getStarCardDropTier(cardEntries)) : [];
    return Object.freeze({
      materials: Object.freeze((source.materials || []).slice().concat(starCards)),
      equipment: Object.freeze((source.equipment || []).slice()),
      consumables: Object.freeze((source.consumables || []).slice()),
      cards: Object.freeze(cardEntries),
      currencyWeight: Math.max(0, Math.round(Number(source.currencyWeight == null ? 14 : source.currencyWeight))),
      globalRareEligible: source.globalRareEligible !== false,
      basicConsumables: source.basicConsumables !== false
    });
  }

  const MONSTER_DROP_POOLS = Object.freeze({
    glassback: monsterDropPool({ materials: [materialDrop('starGlassChip', 48, { minQuantity: 1, maxQuantity: 2 })], equipment: equipmentDrops('training', 3), cards: cardDrops('forest', 2), currencyWeight: 10 }),
    riftLantern: monsterDropPool({ materials: [materialDrop('lanternCore', 38), materialDrop('starGlassChip', 14)], equipment: equipmentDrops('focus', 3), cards: cardDrops('forest', 2), currencyWeight: 11 }),
    faultSkitter: monsterDropPool({ materials: [materialDrop('starGlassChip', 24)], equipment: equipmentDrops('training', 2), cards: cardDrops('forest', 2), currencyWeight: 8 }),
    slimelet: monsterDropPool({ materials: [materialDrop('gelDrop', 48, { minQuantity: 1, maxQuantity: 2 })], equipment: equipmentDrops('training', 3), cards: cardDrops('forest', 3), currencyWeight: 10 }),
    dewSlime: monsterDropPool({ materials: [materialDrop('dewBead', 46), materialDrop('gelDrop', 18, { minQuantity: 1, maxQuantity: 2 })], equipment: equipmentDrops('training', 3), cards: cardDrops('forest', 3), currencyWeight: 10 }),
    mossback: monsterDropPool({ materials: [materialDrop('mossHide', 48)], equipment: equipmentDrops('guard', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    thornSprout: monsterDropPool({ materials: [materialDrop('thornFiber', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    vineSnapper: monsterDropPool({ materials: [materialDrop('vineFiber', 44), materialDrop('upgradeDust', 7)], equipment: equipmentDrops('focus', 3), cards: cardDrops('forest', 4), currencyWeight: 12 }),
    bristleBoar: monsterDropPool({ materials: [materialDrop('bristleHide', 46)], equipment: equipmentDrops('traveler', 4), cards: cardDrops('beast', 4), currencyWeight: 13 }),
    briarStag: monsterDropPool({ materials: [materialDrop('briarAntler', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('traveler', 4), cards: cardDrops('beast', 4), currencyWeight: 14 }),
    dustImp: monsterDropPool({ materials: [materialDrop('dustClaw', 44), materialDrop('upgradeDust', 9)], equipment: equipmentDrops('sharp', 2), cards: cardDrops('bandit', 3), currencyWeight: 12 }),
    clockbug: monsterDropPool({ materials: [materialDrop('clockworkScrap', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('construct', 3), currencyWeight: 13 }),
    rustRatchet: monsterDropPool({ materials: [materialDrop('clockworkScrap', 36), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('rustcoil', 3), cards: cardDrops('construct', 4), currencyWeight: 13 }),
    coilSentry: monsterDropPool({ materials: [materialDrop('chargedCoil', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('focus', 3), cards: cardDrops('construct', 4), currencyWeight: 14 }),
    scrapWarden: monsterDropPool({ materials: [materialDrop('scrapPlate', 42)], equipment: equipmentDrops('rustcoil', 4).concat(equipmentDrops('guard', 2)), cards: cardDrops('construct', 4), currencyWeight: 15 }),
    emberWisp: monsterDropPool({ materials: [materialDrop('emberDust', 48)], equipment: equipmentDrops('focus', 3), cards: cardDrops('cinder', 3), currencyWeight: 13 }),
    ashCrawler: monsterDropPool({ materials: [materialDrop('ashCarapace', 42), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('cinder', 3), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    lavaTick: monsterDropPool({ materials: [materialDrop('moltenFang', 42), materialDrop('emberDust', 20)], equipment: equipmentDrops('cinder', 3), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    cinderSpitter: monsterDropPool({ materials: [materialDrop('cinderGland', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('focus', 2).concat(equipmentDrops('cinder', 3)), cards: cardDrops('cinder', 4), currencyWeight: 14 }),
    banditCutter: monsterDropPool({ materials: [materialDrop('banditCloth', 46)], equipment: equipmentDrops('steel', 4), cards: cardDrops('bandit', 4), currencyWeight: 16 }),
    banditCutterDirect: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterReference: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterHybrid: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditCutterPuppet: monsterDropPool({ globalRareEligible: false, basicConsumables: false, currencyWeight: 0 }),
    banditThrower: monsterDropPool({ materials: [materialDrop('throwingKnifeScrap', 46)], equipment: equipmentDrops('sharp', 4), cards: cardDrops('bandit', 4), currencyWeight: 16 }),
    orebackBeetle: monsterDropPool({ materials: [materialDrop('oreChunks', 48, { minQuantity: 1, maxQuantity: 3 }), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('guard', 2), cards: cardDrops('construct', 3), currencyWeight: 12 }),
    glowcapHealer: monsterDropPool({ materials: [materialDrop('glowSpores', 46)], equipment: equipmentDrops('support', 4), cards: cardDrops('forest', 3).concat(cardDrops('frost', 2)), currencyWeight: 13 }),
    crackedMimic: monsterDropPool({ materials: [materialDrop('upgradeCatalyst', 10), materialDrop('wardingScroll', 4), materialDrop('refinementCore', 3)], equipment: equipmentDrops('rare', 8), cards: cardDrops('mimic', 7), currencyWeight: 48 }),
    brambleking: monsterDropPool({ materials: [materialDrop('brambleCrown', 46), materialDrop('upgradeCatalyst', 8)], cards: cardDrops('bossForest', 8), currencyWeight: 22 }),
    clockworkTitan: monsterDropPool({ materials: [materialDrop('titanCore', 46), materialDrop('upgradeCatalyst', 8)], cards: cardDrops('bossConstruct', 8), currencyWeight: 22 }),
    quarryColossus: monsterDropPool({ materials: [materialDrop('colossusOre', 46), materialDrop('upgradeCatalyst', 7), materialDrop('wardingScroll', 3)], cards: cardDrops('bossConstruct', 8), currencyWeight: 24 }),
    emberjawGolem: monsterDropPool({ materials: [materialDrop('emberjawBadge', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 3)], cards: cardDrops('bossCinder', 8), currencyWeight: 24 }),
    frostlingScout: monsterDropPool({ materials: [materialDrop('rimeShard', 46)], equipment: equipmentDrops('frost', 3), cards: cardDrops('frost', 3), currencyWeight: 14 }),
    shardling: monsterDropPool({ materials: [materialDrop('rimeShard', 42), materialDrop('upgradeDust', 8)], equipment: equipmentDrops('frost', 3), cards: cardDrops('frost', 3), currencyWeight: 14 }),
    rimebackBrute: monsterDropPool({ materials: [materialDrop('frozenHide', 44), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('frost', 2), cards: cardDrops('frost', 4), currencyWeight: 15 }),
    glacierSentinel: monsterDropPool({ materials: [materialDrop('glacierCore', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('frost', 4), cards: cardDrops('frost', 4), currencyWeight: 15 }),
    snowglareWisp: monsterDropPool({ materials: [materialDrop('snowglareDust', 46)], equipment: equipmentDrops('focus', 4), cards: cardDrops('frost', 3).concat(cardDrops('astral', 2)), currencyWeight: 14 }),
    icebloomOracle: monsterDropPool({ materials: [materialDrop('icebloomPetal', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('support', 4), cards: cardDrops('frost', 3).concat(cardDrops('astral', 2)), currencyWeight: 14 }),
    galeHarrier: monsterDropPool({ materials: [materialDrop('galeFeather', 44)], equipment: equipmentDrops('storm', 4).concat(equipmentDrops('focus', 2)), cards: cardDrops('storm', 4), currencyWeight: 15 }),
    stormboundArcher: monsterDropPool({ materials: [materialDrop('stormFletching', 44)], equipment: equipmentDrops('storm', 4).concat(equipmentDrops('sharp', 2)), cards: cardDrops('storm', 4), currencyWeight: 15 }),
    thunderRam: monsterDropPool({ materials: [materialDrop('thunderHorn', 42), materialDrop('upgradeCatalyst', 4)], equipment: equipmentDrops('traveler', 3).concat(equipmentDrops('storm', 3)), cards: cardDrops('storm', 4), currencyWeight: 16 }),
    cloudcallAcolyte: monsterDropPool({ materials: [materialDrop('cloudSilk', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('support', 4), cards: cardDrops('storm', 3).concat(cardDrops('astral', 3)), currencyWeight: 15 }),
    indexScribe: monsterDropPool({ materials: [materialDrop('runicPage', 44)], equipment: equipmentDrops('astral', 4).concat(equipmentDrops('focus', 2)), cards: cardDrops('astral', 4), currencyWeight: 16 }),
    lumenSentinel: monsterDropPool({ materials: [materialDrop('lumenPlate', 42)], equipment: equipmentDrops('astral', 4).concat(equipmentDrops('guard', 2)), cards: cardDrops('astral', 4), currencyWeight: 16 }),
    voidMote: monsterDropPool({ materials: [materialDrop('voidDust', 44), materialDrop('cubeFragment', 5)], equipment: equipmentDrops('focus', 3), cards: cardDrops('astral', 3).concat(cardDrops('eclipse', 2)), currencyWeight: 16 }),
    eclipseDuelist: monsterDropPool({ materials: [materialDrop('eclipseSilk', 42)], equipment: equipmentDrops('eclipse', 4).concat(equipmentDrops('sharp', 2)), cards: cardDrops('eclipse', 4), currencyWeight: 18 }),
    riftAberration: monsterDropPool({ materials: [materialDrop('riftSplinter', 40), materialDrop('upgradeCatalyst', 6), materialDrop('wardingScroll', 3), materialDrop('refinementCore', 3)], equipment: equipmentDrops('rare', 7), cards: cardDrops('mimic', 4).concat(cardDrops('eclipse', 5)), currencyWeight: 26 }),
    rimewarden: monsterDropPool({ materials: [materialDrop('rimewardenSigil', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 3)], cards: cardDrops('bossFrost', 8), currencyWeight: 24 }),
    stormbreakRoc: monsterDropPool({ materials: [materialDrop('stormbreakPlume', 46), materialDrop('upgradeCatalyst', 7), materialDrop('wardingScroll', 3)], cards: cardDrops('bossStorm', 8), currencyWeight: 24 }),
    astralArchivist: monsterDropPool({ materials: [materialDrop('archivistIndex', 46), materialDrop('upgradeCatalyst', 7), materialDrop('cubeFragment', 4)], cards: cardDrops('bossAstral', 8), currencyWeight: 26 }),
    eclipseSovereign: monsterDropPool({ materials: [materialDrop('sovereignCorona', 46), materialDrop('upgradeCatalyst', 7), materialDrop('refinementCore', 4)], cards: cardDrops('bossEclipse', 8), currencyWeight: 28 })
  });

  const api = {
    monsterDropEntry,
    materialDrop,
    equipmentDrop,
    cardDrop,
    equipmentDrops,
    cardDrops,
    getStarCardDropTier,
    starCardDrops,
    monsterDropPool,
    STAR_CARD_MATERIAL_IDS,
    STAR_CARD_DROP_TABLES,
    MONSTER_EQUIPMENT_DROP_GROUPS,
    MONSTER_CARD_DROP_GROUPS,
    MONSTER_DROP_POOLS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.monsterDrops = Object.assign({}, modules.monsterDrops || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
