(function initProjectStarfallDataCommerce(global) {
  'use strict';

  const MARKET_LISTINGS = Object.freeze([
    Object.freeze({ id: 'dust_cache', name: 'Upgrade Dust Cache', summary: 'A repeatable material bundle for early upgrade attempts.', cost: 180, reward: Object.freeze({ materials: Object.freeze({ upgradeDust: 8 }) }) }),
    Object.freeze({ id: 'catalyst_cache', name: 'Catalyst Cache', summary: 'Chance boosters for risky upgrade attempts.', cost: 320, reward: Object.freeze({ materials: Object.freeze({ upgradeCatalyst: 2 }) }) }),
	    Object.freeze({ id: 'warding_scroll_offer', name: 'Warding Scroll', summary: 'Protects gear from one selected destroy-risk upgrade attempt.', cost: 420, reward: Object.freeze({ materials: Object.freeze({ wardingScroll: 1 }) }) }),
	    Object.freeze({ id: 'refinement_core_offer', name: 'Refinement Core', summary: 'Enhances one selected successful upgrade attempt.', cost: 560, reward: Object.freeze({ materials: Object.freeze({ refinementCore: 1 }) }) }),
	    Object.freeze({ id: 'field_supply_crate', name: 'Field Supply Crate', summary: 'Consumables for longer map sessions and dungeon attempts.', cost: 220, reward: Object.freeze({ consumables: Object.freeze({ camp_ration: 2, minor_resource_tonic: 2 }) }) }),
	    Object.freeze({ id: 'lesser_xp_coupon_offer', name: 'Lesser XP Coupon', summary: 'A one-hour 1.2x XP boost for focused leveling sessions.', cost: 480, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_1_2_1h: 1 }) }) }),
	    Object.freeze({ id: 'greater_xp_coupon_offer', name: 'Greater XP Coupon', summary: 'A one-hour 1.5x XP boost for dungeon and boss routing.', cost: 1250, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_1_5_1h: 1 }) }) }),
	    Object.freeze({ id: 'radiant_xp_coupon_offer', name: 'Radiant XP Coupon', summary: 'A one-hour 2.0x XP boost for high-effort progression pushes.', cost: 3000, reward: Object.freeze({ consumables: Object.freeze({ xp_coupon_2_0_1h: 1 }) }) }),
	    Object.freeze({ id: 'lesser_drop_coupon_offer', name: 'Lesser Drop Coupon', summary: 'A one-hour 1.2x monster drop chance boost.', cost: 520, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_1_2_1h: 1 }) }) }),
	    Object.freeze({ id: 'greater_drop_coupon_offer', name: 'Greater Drop Coupon', summary: 'A one-hour 1.5x monster drop chance boost.', cost: 1350, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_1_5_1h: 1 }) }) }),
	    Object.freeze({ id: 'radiant_drop_coupon_offer', name: 'Radiant Drop Coupon', summary: 'A one-hour 2.0x monster drop chance boost.', cost: 3200, reward: Object.freeze({ consumables: Object.freeze({ drop_coupon_2_0_1h: 1 }) }) }),
	    Object.freeze({ id: 'equipment_slot_coupon_offer', name: 'Equipment Coupon Offer', summary: 'A rare account-service purchase that expands gear storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ equipment_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'usable_slot_coupon_offer', name: 'Usable Coupon Offer', summary: 'A rare account-service purchase that expands usable storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ usable_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'etc_slot_coupon_offer', name: 'Etc Coupon Offer', summary: 'A rare account-service purchase that expands material storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ etc_slot_coupon: 1 }) }) }),
    Object.freeze({ id: 'card_slot_coupon_offer', name: 'Card Coupon Offer', summary: 'A rare account-service purchase that expands card storage.', cost: 650, once: true, reward: Object.freeze({ consumables: Object.freeze({ card_slot_coupon: 1 }) }) })
	  ]);

  const COSMETICS = Object.freeze([
    Object.freeze({ id: 'crossing_cape', name: 'Crossing Cape', slot: 'aura', icon: 'CPE', cost: 160, summary: 'A clean starter cape cosmetic for town and field play.' }),
    Object.freeze({ id: 'ember_trim', name: 'Ember Trim', slot: 'aura', icon: 'EMB', cost: 240, summary: 'A warm orange combat accent inspired by Emberjaw Lair.' }),
    Object.freeze({ id: 'vault_spark', name: 'Vault Spark', slot: 'aura', icon: 'VLT', cost: 280, summary: 'A blue-white gearwork spark cosmetic from construct regions.' }),
    Object.freeze({ id: 'ember_impact_splat', name: 'Ember Impact', slot: 'damageSplat', icon: 'EIM', cost: 0, cashShopOnly: true, summary: 'Hot orange damage numbers with ember flares and slash sparks.', damageSplatStyle: Object.freeze({ id: 'ember_impact', variant: 'ember', color: '#ffe16a', criticalColor: '#fff27a', stroke: 'rgba(94, 26, 26, 0.94)', burstColor: '#ff5d5d', accentColor: '#ff8a3d', ringColor: '#fff27a', slashColor: '#ff3d2e', shardColor: '#ffbe55', secondaryShardColor: '#ff5d5d' }) }),
    Object.freeze({ id: 'vault_arc_splat', name: 'Vault Arc', slot: 'damageSplat', icon: 'VAR', cost: 0, cashShopOnly: true, summary: 'Gearwork blue damage numbers with electric arcs and clean rings.', damageSplatStyle: Object.freeze({ id: 'vault_arc', variant: 'vault', color: '#d9f8ff', criticalColor: '#f4fdff', stroke: 'rgba(10, 48, 82, 0.94)', burstColor: '#7bdff2', accentColor: '#ffffff', ringColor: '#9cf6ff', slashColor: '#68a9ff', shardColor: '#7bdff2', secondaryShardColor: '#ffffff' }) }),
    Object.freeze({ id: 'frost_shatter_splat', name: 'Frost Shatter', slot: 'damageSplat', icon: 'FRS', cost: 0, cashShopOnly: true, summary: 'Crisp cyan damage numbers with icy shards and a sharp pop.', damageSplatStyle: Object.freeze({ id: 'frost_shatter', variant: 'frost', color: '#e9fbff', criticalColor: '#ffffff', stroke: 'rgba(22, 61, 92, 0.94)', burstColor: '#a8f1ff', accentColor: '#f7fbff', ringColor: '#b8e6ff', slashColor: '#7bdff2', shardColor: '#e9fbff', secondaryShardColor: '#7bdff2' }) }),
    Object.freeze({ id: 'astral_prism_splat', name: 'Astral Prism', slot: 'damageSplat', icon: 'APR', cost: 0, cashShopOnly: true, summary: 'Violet-gold damage numbers with prism flashes and orbiting shards.', damageSplatStyle: Object.freeze({ id: 'astral_prism', variant: 'astral', color: '#fff0a6', criticalColor: '#fff7d6', stroke: 'rgba(50, 31, 90, 0.94)', burstColor: '#c794ff', accentColor: '#ffe16a', ringColor: '#c794ff', slashColor: '#8f8cff', shardColor: '#ffe16a', secondaryShardColor: '#c794ff' }) }),
    Object.freeze({ id: 'stormcall_strike_splat', name: 'Stormcall Strike', slot: 'damageSplat', icon: 'STM', cost: 0, cashShopOnly: true, summary: 'Yellow-blue damage numbers with fast lightning slashes.', damageSplatStyle: Object.freeze({ id: 'stormcall_strike', variant: 'storm', color: '#fff7b8', criticalColor: '#ffffff', stroke: 'rgba(30, 45, 91, 0.94)', burstColor: '#68a9ff', accentColor: '#ffe16a', ringColor: '#7bdff2', slashColor: '#ffe16a', shardColor: '#68a9ff', secondaryShardColor: '#fff7b8' }) }),
    Object.freeze({ id: 'verdant_bloom_splat', name: 'Verdant Bloom', slot: 'damageSplat', icon: 'VBL', cost: 0, cashShopOnly: true, summary: 'Green-gold damage numbers with petal arcs and a soft bloom.', damageSplatStyle: Object.freeze({ id: 'verdant_bloom', variant: 'verdant', color: '#e6ffd1', criticalColor: '#f7ffe6', stroke: 'rgba(30, 73, 43, 0.94)', burstColor: '#8ec878', accentColor: '#fff0a6', ringColor: '#68d58d', slashColor: '#8ec878', shardColor: '#e6ffd1', secondaryShardColor: '#68d58d' }) }),
    Object.freeze({ id: 'starlit_checkin', name: 'Starlit Check-In', slot: 'aura', icon: 'CHK', cost: 0, dailyReward: true, summary: 'A soft blue aura for players who keep returning to Starfall routes.' }),
    Object.freeze({ id: 'constellation_trail', name: 'Constellation Trail', slot: 'aura', icon: 'CON', cost: 0, dailyReward: true, summary: 'A long-haul attendance aura with small star sparks around movement.' }),
    Object.freeze({ id: 'comet_year_splat', name: 'Comet Year', slot: 'damageSplat', icon: '365', cost: 0, dailyReward: true, summary: 'Prestige comet damage numbers for a full year of claimed daily rewards.', damageSplatStyle: Object.freeze({ id: 'comet_year', variant: 'comet', color: '#f9f0bd', criticalColor: '#ffffff', stroke: 'rgba(30, 42, 78, 0.94)', burstColor: '#7bdff2', accentColor: '#ffbe55', ringColor: '#b785ff', slashColor: '#68a9ff', shardColor: '#fff3cf', secondaryShardColor: '#7bdff2' }) }),
    Object.freeze({ id: 'founder_spark', name: 'Founder Spark', slot: 'aura', icon: 'FND', cost: 0, summary: 'First-clear prestige aura for completing Fracture Watch weekly operations.', seasonReward: true })
  ]);

  const CASH_SHOP_CATEGORIES = Object.freeze([
    Object.freeze({ id: 'featured', label: 'Featured' }),
    Object.freeze({ id: 'cosmetics', label: 'Cosmetics' }),
    Object.freeze({ id: 'effects', label: 'Effects' }),
    Object.freeze({ id: 'buffs', label: 'Buffs' }),
    Object.freeze({ id: 'owned', label: 'Owned' })
  ]);

  const CASH_SHOP_ITEMS = Object.freeze([
    Object.freeze({ id: 'crossing_cape_cash', name: 'Crossing Cape', category: 'cosmetics', featured: true, kind: 'cosmetic', cosmeticId: 'crossing_cape', icon: 'CPE', price: 80, summary: 'A clean starter cape cosmetic for town and field play.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'ember_trim_cash', name: 'Ember Trim', category: 'effects', featured: true, kind: 'cosmetic', cosmeticId: 'ember_trim', icon: 'EMB', price: 120, summary: 'A warm orange combat accent inspired by Emberjaw Lair.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'vault_spark_cash', name: 'Vault Spark', category: 'effects', kind: 'cosmetic', cosmeticId: 'vault_spark', icon: 'VLT', price: 160, summary: 'A blue-white gearwork spark cosmetic from construct regions.', tags: Object.freeze(['Cosmetic']) }),
    Object.freeze({ id: 'ember_impact_splat_cash', name: 'Ember Impact', category: 'effects', kind: 'cosmetic', cosmeticId: 'ember_impact_splat', icon: 'EIM', price: 95, summary: 'Hot orange damage numbers with ember flares and slash sparks.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'vault_arc_splat_cash', name: 'Vault Arc', category: 'effects', kind: 'cosmetic', cosmeticId: 'vault_arc_splat', icon: 'VAR', price: 110, summary: 'Gearwork blue damage numbers with electric arcs and clean rings.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'frost_shatter_splat_cash', name: 'Frost Shatter', category: 'effects', kind: 'cosmetic', cosmeticId: 'frost_shatter_splat', icon: 'FRS', price: 125, summary: 'Crisp cyan damage numbers with icy shards and a sharp pop.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'astral_prism_splat_cash', name: 'Astral Prism', category: 'effects', kind: 'cosmetic', cosmeticId: 'astral_prism_splat', icon: 'APR', price: 145, summary: 'Violet-gold damage numbers with prism flashes and orbiting shards.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'stormcall_strike_splat_cash', name: 'Stormcall Strike', category: 'effects', kind: 'cosmetic', cosmeticId: 'stormcall_strike_splat', icon: 'STM', price: 160, summary: 'Yellow-blue damage numbers with fast lightning slashes.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'verdant_bloom_splat_cash', name: 'Verdant Bloom', category: 'effects', kind: 'cosmetic', cosmeticId: 'verdant_bloom_splat', icon: 'VBL', price: 105, summary: 'Green-gold damage numbers with petal arcs and a soft bloom.', tags: Object.freeze(['Cosmetic', 'Damage Splat']) }),
    Object.freeze({ id: 'guard_tonic_pack', name: 'Guard Tonic Pack', category: 'buffs', featured: true, kind: 'buffBundle', icon: 'GRD', price: 45, weeklyLimit: 3, summary: 'Three normal Guard Tonics. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ guard_tonic: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'swiftstep_oil_pack', name: 'Swiftstep Oil Pack', category: 'buffs', kind: 'buffBundle', icon: 'SFT', price: 45, weeklyLimit: 3, summary: 'Three normal Swiftstep Oils. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ swiftstep_oil: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'magnet_charm_pack', name: 'Magnet Charm Pack', category: 'buffs', kind: 'buffBundle', icon: 'MAG', price: 60, weeklyLimit: 3, summary: 'Two normal Magnet Charms. Identical to field-earned versions.', reward: Object.freeze({ consumables: Object.freeze({ magnet_charm: 2 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) }),
    Object.freeze({ id: 'field_ration_crate', name: 'Field Ration Crate', category: 'buffs', kind: 'buffBundle', icon: 'RAT', price: 35, weeklyLimit: 3, summary: 'Three Camp Rations for longer field sessions.', reward: Object.freeze({ consumables: Object.freeze({ camp_ration: 3 }) }), earnableSources: Object.freeze(['Field drops', 'Market crates', 'Accomplishments']), tags: Object.freeze(['Earnable In Game']) })
  ]);

  const SEASONS = Object.freeze([
    Object.freeze({
      id: 'beta_foundations',
      name: 'Fracture Watch: Weekly Operations',
      active: true,
      cadence: 'weekly',
      resetDayUtc: 1,
      resetHourUtc: 0,
      summary: 'A low-pressure weekly route through field combat, bosses, and dungeons. Progress resets Monday at 00:00 UTC.',
      objectives: Object.freeze([
        Object.freeze({ id: 'field_patrol', type: 'defeat', count: 60, label: 'Defeat 60 enemies' }),
        Object.freeze({ id: 'field_bosses', type: 'defeatBoss', count: 2, label: 'Defeat 2 bosses' }),
        Object.freeze({ id: 'dungeon_clears', type: 'dungeonComplete', count: 2, label: 'Clear 2 dungeons' })
      ]),
      rewards: Object.freeze({ currency: 300, starTokens: 180, materials: Object.freeze({ upgradeDust: 10, upgradeCatalyst: 1 }) }),
      firstCompletionRewards: Object.freeze({ cosmeticId: 'founder_spark' })
    })
  ]);

  const api = {
    MARKET_LISTINGS,
    COSMETICS,
    CASH_SHOP_CATEGORIES,
    CASH_SHOP_ITEMS,
    SEASONS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.commerce = Object.assign({}, modules.commerce || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
