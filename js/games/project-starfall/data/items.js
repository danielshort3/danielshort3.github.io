(function initProjectStarfallDataItems(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const ASSET_ROOT = DataAssets.ASSET_ROOT;

  const ITEM_ICON_ROOT = `${ASSET_ROOT}/items`;
  const ITEM_ICON_FILE_ROOT = `${ITEM_ICON_ROOT}/icons`;

  function itemIconFileName(itemId) {
    return String(itemId || '')
      .replace(/_/g, '-')
      .replace(/[^a-zA-Z0-9-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')
      .toLowerCase();
  }

  function itemIconAssetPath(itemId) {
    return `${ITEM_ICON_FILE_ROOT}/${itemIconFileName(itemId)}.png`;
  }

  function itemSheetAssets(sheetFile, columns, itemIds) {
    return Object.freeze((itemIds || []).reduce((assets, id) => {
      assets[id] = itemIconAssetPath(id);
      return assets;
    }, {}));
  }

  function materialAssetId(materialId) {
    return String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase();
  }

  function materialItem(materialId, name, icon, rarity, options) {
    const settings = options || {};
    return Object.freeze({
      id: materialId,
      materialId,
      assetId: settings.assetId || materialAssetId(materialId),
      name,
      icon,
      rarity: rarity || 'Common',
      starterQuantity: Math.max(0, Math.floor(Number(settings.starterQuantity || 0) || 0)),
      primaryDrop: settings.primaryDrop !== false,
      genericDrop: !!settings.genericDrop,
      dropLabels: Object.freeze((settings.dropLabels || [name]).slice())
    });
  }

  const MATERIAL_ITEMS = Object.freeze([
    materialItem('upgradeDust', 'Upgrade Dust', 'UD', 'Uncommon', { starterQuantity: 6, primaryDrop: false }),
    materialItem('upgradeCatalyst', 'Upgrade Catalyst', 'UC', 'Rare', { primaryDrop: false, dropLabels: ['Upgrade Catalyst', 'Rare catalyst'] }),
	    materialItem('wardingScroll', 'Warding Scroll', 'WS', 'Rare', { primaryDrop: false }),
	    materialItem('refinementCore', 'Refinement Core', 'RC', 'Epic', { primaryDrop: false }),
	    materialItem('cubeFragment', 'Prism Shard', 'PS', 'Rare', { primaryDrop: false, dropLabels: ['Prism Shards', 'Prism Shard'] }),
	    materialItem('whiteStarCard', 'White Star Card', 'WSC', 'Common', { assetId: 'white_star_card', primaryDrop: false, dropLabels: ['White Star Card', 'White Star Cards'] }),
    materialItem('greenStarCard', 'Green Star Card', 'GSC', 'Uncommon', { assetId: 'green_star_card', primaryDrop: false, dropLabels: ['Green Star Card', 'Green Star Cards'] }),
    materialItem('blueStarCard', 'Blue Star Card', 'BSC', 'Rare', { assetId: 'blue_star_card', primaryDrop: false, dropLabels: ['Blue Star Card', 'Blue Star Cards'] }),
    materialItem('purpleStarCard', 'Purple Star Card', 'PSC', 'Epic', { assetId: 'purple_star_card', primaryDrop: false, dropLabels: ['Purple Star Card', 'Purple Star Cards'] }),
    materialItem('orangeStarCard', 'Orange Star Card', 'OSC', 'Relic', { assetId: 'orange_star_card', primaryDrop: false, dropLabels: ['Orange Star Card', 'Orange Star Cards'] }),
    materialItem('starGlassChip', 'Star-glass Chip', 'SGC', 'Common', { assetId: 'rime_shard' }),
    materialItem('lanternCore', 'Lantern Core', 'LNC', 'Uncommon', { assetId: 'void_dust' }),
    materialItem('gelDrop', 'Gel Drop', 'GD', 'Common', { genericDrop: true }),
    materialItem('oreChunks', 'Ore Chunks', 'OR', 'Common', { genericDrop: true }),
    materialItem('dewBead', 'Dew Bead', 'DEW', 'Common'),
    materialItem('mossHide', 'Moss Hide', 'MOS', 'Common'),
    materialItem('thornFiber', 'Thorn Fiber', 'THN', 'Common'),
    materialItem('vineFiber', 'Vine Fiber', 'VIN', 'Common'),
    materialItem('bristleHide', 'Bristle Hide', 'BRI', 'Common'),
    materialItem('briarAntler', 'Briar Antler', 'ANT', 'Uncommon'),
    materialItem('dustClaw', 'Dust Claw', 'CLW', 'Common'),
    materialItem('clockworkScrap', 'Clockwork Scrap', 'CLK', 'Common'),
    materialItem('chargedCoil', 'Charged Coil', 'COI', 'Uncommon'),
    materialItem('scrapPlate', 'Scrap Plate', 'SCP', 'Uncommon'),
    materialItem('emberDust', 'Ember Dust', 'EMB', 'Common'),
    materialItem('ashCarapace', 'Ash Carapace', 'ASH', 'Common'),
    materialItem('moltenFang', 'Molten Fang', 'FNG', 'Common'),
    materialItem('cinderGland', 'Cinder Gland', 'CIN', 'Uncommon'),
    materialItem('banditCloth', 'Bandit Cloth', 'BAN', 'Common'),
    materialItem('throwingKnifeScrap', 'Throwing Knife Scrap', 'TKS', 'Common'),
    materialItem('glowSpores', 'Glow Spores', 'GLW', 'Common'),
    materialItem('brambleCrown', 'Bramble Crown', 'BRC', 'Rare'),
    materialItem('titanCore', 'Titan Core', 'TIT', 'Rare'),
    materialItem('colossusOre', 'Colossus Ore', 'COL', 'Rare'),
    materialItem('emberjawBadge', 'Emberjaw Badge', 'EJB', 'Rare'),
    materialItem('rimeShard', 'Rime Shard', 'RIM', 'Common'),
    materialItem('frozenHide', 'Frozen Hide', 'FRZ', 'Common'),
    materialItem('glacierCore', 'Glacier Core', 'GLC', 'Uncommon'),
    materialItem('snowglareDust', 'Snowglare Dust', 'SNO', 'Common'),
    materialItem('icebloomPetal', 'Icebloom Petal', 'ICE', 'Common'),
    materialItem('galeFeather', 'Gale Feather', 'GAL', 'Common'),
    materialItem('stormFletching', 'Storm Fletching', 'FLT', 'Common'),
    materialItem('thunderHorn', 'Thunder Horn', 'THU', 'Uncommon'),
    materialItem('cloudSilk', 'Cloud Silk', 'CLD', 'Common'),
    materialItem('runicPage', 'Runic Page', 'RUN', 'Common'),
    materialItem('lumenPlate', 'Lumen Plate', 'LUM', 'Uncommon'),
    materialItem('voidDust', 'Void Dust', 'VOI', 'Common'),
    materialItem('eclipseSilk', 'Eclipse Silk', 'ECL', 'Uncommon'),
    materialItem('riftSplinter', 'Rift Splinter', 'RFT', 'Rare'),
    materialItem('rimewardenSigil', 'Rimewarden Sigil', 'RWS', 'Rare'),
    materialItem('stormbreakPlume', 'Stormbreak Plume', 'SBP', 'Rare'),
    materialItem('archivistIndex', 'Archivist Index', 'IDX', 'Rare'),
    materialItem('sovereignCorona', 'Sovereign Corona', 'COR', 'Rare')
  ]);

  const ITEM_ASSETS = Object.freeze(Object.assign({},
    itemSheetAssets('ai-items-star-cards-sheet.png', 5, [
      'white_star_card',
      'green_star_card',
      'blue_star_card',
      'purple_star_card',
      'orange_star_card'
    ]),
    itemSheetAssets('ai-items-consumables-materials-sheet.png', 5, [
      'coins',
      'town_return_scroll',
      'guard_tonic',
      'swiftstep_oil',
      'magnet_charm',
      'pet_whistle',
      'cube_fragment',
      'base_skill_manual',
      'advanced_skill_manual',
      'skill_reset_scroll',
      'stat_reset_scroll',
      'admin_worldwright_console',
      'upgrade_dust',
      'upgrade_catalyst',
      'warding_scroll',
      'refinement_core',
      'gel_drop',
      'ore_chunks',
      'line_catalyst'
    ]),
    itemSheetAssets('ai-items-potion-tiers-sheet.png', 4, [
      'minor_health_potion',
      'standard_health_potion',
      'greater_health_potion',
      'superior_health_potion',
      'minor_resource_tonic',
      'standard_resource_tonic',
      'greater_resource_tonic',
      'superior_resource_tonic',
      'camp_ration',
      'field_ration',
      'expedition_ration',
      'hero_ration'
    ]),
    itemSheetAssets('ai-items-mob-materials-core-sheet.png', 5, [
      'dew_bead',
      'moss_hide',
      'thorn_fiber',
      'vine_fiber',
      'bristle_hide',
      'briar_antler',
      'dust_claw',
      'clockwork_scrap',
      'charged_coil',
      'scrap_plate',
      'ember_dust',
      'ash_carapace',
      'molten_fang',
      'cinder_gland',
      'bandit_cloth',
      'throwing_knife_scrap',
      'glow_spores',
      'bramble_crown',
      'titan_core',
      'colossus_ore'
    ]),
    itemSheetAssets('ai-items-mob-materials-late-sheet.png', 5, [
      'emberjaw_badge',
      'rime_shard',
      'frozen_hide',
      'glacier_core',
      'snowglare_dust',
      'icebloom_petal',
      'gale_feather',
      'storm_fletching',
      'thunder_horn',
      'cloud_silk',
      'runic_page',
      'lumen_plate',
      'void_dust',
      'eclipse_silk',
      'rift_splinter',
      'rimewarden_sigil',
      'stormbreak_plume',
      'archivist_index',
      'sovereign_corona'
    ]),
    itemSheetAssets('ai-items-coin-stacks-sheet.png', 4, [
      'coins_small',
      'coins_medium',
      'coins_large',
      'coins_huge'
    ]),
    itemSheetAssets('ai-items-shop-boss-forest-sheet.png', 5, [
      'training_sword',
      'training_wand',
      'training_bow',
      'copper_sword',
      'birch_wand',
      'simple_bow',
      'stitched_vest',
      'traveler_boots',
      'plain_ring',
      'iron_sword',
      'iron_axe',
      'apprentice_staff',
      'oak_longbow',
      'guardian_tower_shield',
      'berserker_war_grip',
      'ember_core',
      'rune_etched_focus',
      'deadeye_scope',
      'trap_kit',
      'thorncrown_greatsword',
      'thornroot_staff',
      'briarstring_longbow',
      'briar_crown',
      'barkplate_harness',
      'grasping_thorn_gloves'
    ]),
    itemSheetAssets('ai-items-world-drops-sheet.png', 5, [
      'adventurer_cutlass',
      'balanced_focus',
      'wanderer_charm',
      'fieldguard_helm',
      'trailwoven_gloves',
      'vanguard_blade',
      'bulwark_plate',
      'breaker_gauntlets',
      'sentinel_greaves',
      'starglass_staff',
      'runewoven_robes',
      'channeler_gloves',
      'aetherstep_boots',
      'ranger_recurve',
      'pathfinder_leathers',
      'deadeye_wraps',
      'windrunner_boots'
    ]),
    itemSheetAssets('ai-items-boss-core-storm-sheet.png', 5, [
      'rootstep_greaves',
      'emberjaw_cleaver',
      'magma_scepter',
      'cindercoil_bow',
      'ashen_jaw_helm',
      'furnaceplate',
      'lavaforged_gauntlets',
      'scorchtrail_boots',
      'gearcleaver',
      'chrono_staff',
      'ratchet_repeater',
      'titan_visor',
      'clockplate_harness',
      'gyro_gauntlets',
      'springstep_boots',
      'colossus_maul',
      'geode_scepter',
      'oreline_greatbow',
      'deepcore_helm',
      'bedrock_plate',
      'quarry_fists',
      'stonewake_boots',
      'stormtalon_saber',
      'cloudspine_rod',
      'skybreaker_bow'
    ]),
	    itemSheetAssets('ai-items-boss-astral-eclipse-sheet.png', 5, [
	      'rocfeather_mask',
	      'tempest_mantle',
	      'lightning_grip_gloves',
	      'gale_boots',
      'index_blade',
      'starbound_codex',
      'cometstring_bow',
      'archivist_crown',
      'astral_robes',
      'scribe_gloves',
      'orbit_boots',
      'eclipse_edge',
      'umbral_starstaff',
      'corona_longbow',
      'sovereign_crown',
      'eclipse_plate',
	      'penumbra_gloves',
	      'sunfall_boots'
	    ]),
    itemSheetAssets('generated-regional-shop-icons', 0, [
      'duelist_parry_medal',
      'storm_charge_focus',
      'beast_bond_charm',
      'rustcoil_field_helm',
      'rustcoil_work_vest',
      'rustcoil_grip_gloves',
      'cinder_steel_sword',
      'cinder_steel_scepter',
      'cinder_ashwood_bow',
      'cinder_reinforced_mail',
      'cinder_forge_boots',
      'cinder_ember_band',
      'frostfen_silver_saber',
      'frostfen_moonlit_staff',
      'frostfen_frostpine_bow',
      'frostfen_iceguard_coat',
      'frostfen_snowstep_boots',
      'frostfen_rime_ring',
      'stormbreak_stormforged_blade',
      'stormbreak_thunder_rod',
      'stormbreak_gale_longbow',
      'stormbreak_tempest_mantle',
      'stormbreak_cloudrunner_boots',
      'stormbreak_lightning_charm',
      'astral_index_blade',
      'astral_star_lens_staff',
      'astral_comet_bow',
      'astral_starwoven_robes',
      'astral_orbitstep_boots',
      'astral_lens_amulet'
    ]),
    itemSheetAssets('ai-items-rate-coupons-sheet.png', 3, [
      'xp_coupon_1_2_1h',
      'xp_coupon_1_5_1h',
      'xp_coupon_2_0_1h',
      'drop_coupon_1_2_1h',
      'drop_coupon_1_5_1h',
      'drop_coupon_2_0_1h'
    ]),
	    itemSheetAssets('ai-items-slot-prisms-plinko-sheet.png', 3, [
	      'equipment_slot_coupon',
	      'usable_slot_coupon',
	      'etc_slot_coupon',
	      'card_slot_coupon',
	      'potential_cube',
	      'preservation_cube',
	      'plinko_ball_basic',
	      'plinko_ball_polished',
	      'plinko_ball_meteor'
	    ])
		  ));

  const ITEM_SHEET_BACKUP_ASSETS = Object.freeze([
    'ai-items-star-cards-sheet.png',
    'ai-items-consumables-materials-sheet.png',
    'ai-items-potion-tiers-sheet.png',
    'ai-items-mob-materials-core-sheet.png',
    'ai-items-mob-materials-late-sheet.png',
    'ai-items-coin-stacks-sheet.png',
    'ai-items-shop-boss-forest-sheet.png',
    'ai-items-world-drops-sheet.png',
    'ai-items-boss-core-storm-sheet.png',
    'ai-items-boss-astral-eclipse-sheet.png',
    'ai-items-rate-coupons-sheet.png',
    'ai-items-slot-prisms-plinko-sheet.png'
  ].map((fileName) => `${ITEM_ICON_ROOT}/sheets/${fileName}`));

  const ITEM_RARITY_VISUALS = Object.freeze({
    Common: Object.freeze({ color: '#d8e5ec', glow: 7, alpha: 0.5, ring: 1.4 }),
    Uncommon: Object.freeze({ color: '#74d680', glow: 11, alpha: 0.7, ring: 1.8 }),
    Rare: Object.freeze({ color: '#68a9ff', glow: 15, alpha: 0.82, ring: 2.1 }),
    Epic: Object.freeze({ color: '#c794ff', glow: 19, alpha: 0.92, ring: 2.4, pulse: 0.16 }),
    Relic: Object.freeze({ color: '#ffbe55', glow: 23, alpha: 0.98, ring: 2.7, pulse: 0.2 })
  });

  const api = {
    ITEM_ASSETS,
    ITEM_SHEET_BACKUP_ASSETS,
    ITEM_RARITY_VISUALS,
    MATERIAL_ITEMS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.items = Object.assign({}, modules.items || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
