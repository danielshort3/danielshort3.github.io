(function initProjectStarfallDataConsumables(global) {
  'use strict';

  function rateCouponItem(config) {
    const multiplier = Number(config && config.multiplier || 1);
    const type = String(config && config.type || '').trim();
    const typeLabel = type === 'drop' ? 'Drop' : 'XP';
    const tierLabel = multiplier >= 2 ? 'Radiant' : multiplier >= 1.5 ? 'Greater' : 'Lesser';
    return Object.freeze({
      id: `${type}_coupon_${multiplier.toFixed(1).replace('.', '_')}_1h`,
      name: `${tierLabel} ${typeLabel} Coupon`,
      icon: type === 'drop' ? 'DRP' : 'XP',
      rarity: config && config.rarity || (multiplier >= 2 ? 'Epic' : multiplier >= 1.5 ? 'Rare' : 'Uncommon'),
      effect: `Increase ${type === 'drop' ? 'monster drop chance' : 'XP gains'} by ${multiplier.toFixed(1)}x for 1 hour.`,
      buffId: `${type}Coupon${Math.round(multiplier * 10)}`,
      buffDuration: 3600,
      rateBuffType: type,
      rateMultiplier: multiplier
    });
  }

  const RATE_COUPON_ITEMS = Object.freeze([
    rateCouponItem({ type: 'xp', multiplier: 1.2, rarity: 'Uncommon' }),
    rateCouponItem({ type: 'xp', multiplier: 1.5, rarity: 'Rare' }),
    rateCouponItem({ type: 'xp', multiplier: 2, rarity: 'Epic' }),
    rateCouponItem({ type: 'drop', multiplier: 1.2, rarity: 'Uncommon' }),
    rateCouponItem({ type: 'drop', multiplier: 1.5, rarity: 'Rare' }),
    rateCouponItem({ type: 'drop', multiplier: 2, rarity: 'Epic' })
  ]);

  const PLINKO_BALLS = Object.freeze([
    Object.freeze({
      id: 'plinko_ball_basic',
      name: 'Starfall Ball',
      icon: 'PLK',
      rarity: 'Uncommon',
      effect: 'Drop it into Starfall Plinko for a modest reward roll.',
      plinkoBall: true,
      plinkoTier: 'basic',
      cost: 250,
      pity: 1,
      pityTarget: 100
    }),
    Object.freeze({
      id: 'plinko_ball_polished',
      name: 'Polished Starfall Ball',
      icon: 'PLK+',
      rarity: 'Rare',
      effect: 'Drop it into Starfall Plinko for stronger material, coupon, and gear odds.',
      plinkoBall: true,
      plinkoTier: 'polished',
      cost: 1200,
      pity: 1,
      pityTarget: 100
    }),
    Object.freeze({
      id: 'plinko_ball_meteor',
      name: 'Meteor Starfall Ball',
      icon: 'MET',
      rarity: 'Epic',
      effect: 'Drop it into Starfall Plinko for premium reward odds and 100-ball pity.',
      plinkoBall: true,
      plinkoTier: 'meteor',
      cost: 6000,
      pity: 1,
      pityTarget: 100
    })
  ]);

  const CONSUMABLE_ITEMS = Object.freeze([
    Object.freeze({ id: 'minor_health_potion', name: 'Minor Health Potion', icon: 'HP', rarity: 'Common', effect: 'Restore 60 HP.', hpFlat: 60 }),
    Object.freeze({ id: 'minor_resource_tonic', name: 'Minor MP Tonic', icon: 'MP', rarity: 'Common', effect: 'Restore 45 MP.', resourceFlat: 45 }),
    Object.freeze({ id: 'camp_ration', name: 'Camp Ration', icon: 'RAT', rarity: 'Common', effect: 'Restore 40 HP and 25 MP.', hpFlat: 40, resourceFlat: 25 }),
    Object.freeze({ id: 'standard_health_potion', name: 'Standard Health Potion', icon: 'HP+', rarity: 'Uncommon', effect: 'Restore 150 HP.', hpFlat: 150 }),
    Object.freeze({ id: 'standard_resource_tonic', name: 'Standard MP Tonic', icon: 'MP+', rarity: 'Uncommon', effect: 'Restore 90 MP.', resourceFlat: 90 }),
    Object.freeze({ id: 'field_ration', name: 'Field Ration', icon: 'RAT+', rarity: 'Uncommon', effect: 'Restore 100 HP and 55 MP.', hpFlat: 100, resourceFlat: 55 }),
    Object.freeze({ id: 'greater_health_potion', name: 'Greater Health Potion', icon: 'GHP', rarity: 'Rare', effect: 'Restore 320 HP.', hpFlat: 320 }),
    Object.freeze({ id: 'greater_resource_tonic', name: 'Greater MP Tonic', icon: 'GMP', rarity: 'Rare', effect: 'Restore 180 MP.', resourceFlat: 180 }),
    Object.freeze({ id: 'expedition_ration', name: 'Expedition Ration', icon: 'EXR', rarity: 'Rare', effect: 'Restore 220 HP and 110 MP.', hpFlat: 220, resourceFlat: 110 }),
    Object.freeze({ id: 'superior_health_potion', name: 'Superior Health Potion', icon: 'SHP', rarity: 'Epic', effect: 'Restore 560 HP.', hpFlat: 560 }),
    Object.freeze({ id: 'superior_resource_tonic', name: 'Superior MP Tonic', icon: 'SMP', rarity: 'Epic', effect: 'Restore 320 MP.', resourceFlat: 320 }),
    Object.freeze({ id: 'hero_ration', name: 'Hero Ration', icon: 'HER', rarity: 'Epic', effect: 'Restore 380 HP and 190 MP.', hpFlat: 380, resourceFlat: 190 }),
    Object.freeze({ id: 'town_return_scroll', name: 'Town Return Scroll', icon: 'TWN', effect: 'Return to the nearest regional town.', returnMapId: 'starfallCrossing', dynamicTownReturn: true }),
    Object.freeze({ id: 'guard_tonic', name: 'Guard Tonic', icon: 'GRD', effect: 'Take reduced damage for 12 seconds.', buffId: 'guardTonic', buffDuration: 12 }),
    Object.freeze({ id: 'swiftstep_oil', name: 'Swiftstep Oil', icon: 'SPD', effect: 'Move faster for 12 seconds.', buffId: 'swiftstepOil', buffDuration: 12 }),
    Object.freeze({ id: 'magnet_charm', name: 'Magnet Charm', icon: 'MAG', effect: 'Increase loot pickup reach for 30 seconds.', buffId: 'magnetCharm', buffDuration: 30 }),
    ...RATE_COUPON_ITEMS,
    ...PLINKO_BALLS,
    Object.freeze({ id: 'pet_whistle', name: 'Pet Whistle', icon: 'PET', effect: 'Permanently unlock Pet Assist automation.', petUnlock: true }),
    Object.freeze({ id: 'equipment_slot_coupon', name: 'Equipment Slot Coupon', icon: 'EQP', effect: 'Expands the Equipment inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'equipment' }),
    Object.freeze({ id: 'usable_slot_coupon', name: 'Usable Slot Coupon', icon: 'USE', effect: 'Expands the Usable inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'usable' }),
    Object.freeze({ id: 'etc_slot_coupon', name: 'Etc Slot Coupon', icon: 'ETC', effect: 'Expands the Etc inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'etc' }),
    Object.freeze({ id: 'card_slot_coupon', name: 'Card Slot Coupon', icon: 'CRD', effect: 'Expands the Cards inventory tab by 36 slots.', inventorySectionCoupon: true, inventorySectionTab: 'cards' }),
    Object.freeze({ id: 'potential_cube', name: 'Attunement Prism', icon: 'PRS', rarity: 'Rare', effect: 'Retunes active attunement lines on selected gear. Tiers improve line quality, not line count.', potentialCube: true }),
    Object.freeze({ id: 'preservation_cube', name: 'Echo Prism', icon: 'ECHO', rarity: 'Epic', effect: 'Retunes active attunement lines once, then lets you keep the current attunement or apply the new one.', preservationCube: true }),
    Object.freeze({ id: 'line_catalyst', name: 'Line Catalyst', icon: 'LC', rarity: 'Epic', effect: 'Attempts to unlock another active attunement line on selected gear. Failure hides one active line when possible.', lineCatalyst: true }),
    Object.freeze({ id: 'admin_worldwright_console', name: 'Worldwright Console', icon: 'ADM', rarity: 'Relic', effect: 'Admin-only limit testing console for spawning mobs, granting items, and editing gear.', adminOnly: true, opensAdminConsole: true }),
    Object.freeze({ id: 'base_skill_manual', name: 'Base SP Manual', icon: 'BSP', effect: 'Grants 1 Base SP until the base job can max every skill except one.', skillPointPool: 'baseSkillPoints', skillPointAmount: 1 }),
    Object.freeze({ id: 'advanced_skill_manual', name: 'Advanced SP Manual', icon: 'ASP', effect: 'Grants 1 Advanced SP until the advanced job can max every skill except one.', skillPointPool: 'advancedSkillPoints', skillPointAmount: 1 }),
    Object.freeze({ id: 'skill_reset_scroll', name: 'SP Reset Scroll', icon: 'RST', effect: 'Reset all skill ranks and refund spent SP to the matching job pools.', resetSkillPoints: true }),
    Object.freeze({ id: 'stat_reset_scroll', name: 'Stat Reset Scroll', icon: 'AP', effect: 'Reset all Stat Upgrade Point allocations.', resetStatUpgrades: true })
  ]);

  const api = {
    rateCouponItem,
    RATE_COUPON_ITEMS,
    PLINKO_BALLS,
    CONSUMABLE_ITEMS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.consumables = Object.assign({}, modules.consumables || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
