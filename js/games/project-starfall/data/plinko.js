(function initProjectStarfallDataPlinko(global) {
  'use strict';

  const PLINKO_BOUNCE_COUNT = 8;
  const PLINKO_ACTIVE_DROP_LIMIT = 0;
  const PLINKO_PITY_TARGET = 100;
  const PLINKO_SLOT_PROBABILITIES = Object.freeze([1, 8, 28, 56, 70, 56, 28, 8, 1]);
  const PLINKO_SLOT_DENOMINATOR = 256;
  const PLINKO_SLOT_PROBABILITY_TABLES = Object.freeze({
    5: Object.freeze([1, 4, 6, 4, 1]),
    7: Object.freeze([1, 6, 15, 20, 15, 6, 1]),
    9: PLINKO_SLOT_PROBABILITIES
  });

  function getPlinkoSlotProbabilities(slotCount) {
    const count = Math.max(1, Math.floor(Number(slotCount || 9) || 9));
    return PLINKO_SLOT_PROBABILITY_TABLES[count] || PLINKO_SLOT_PROBABILITIES;
  }

  function getPlinkoSlotDenominator(slotCount) {
    return getPlinkoSlotProbabilities(slotCount).reduce((sum, value) => sum + value, 0);
  }

  function plinkoSlot(config, slotCount) {
    const index = Math.max(0, Math.floor(Number(config.index || 0) || 0));
    const probabilities = getPlinkoSlotProbabilities(slotCount);
    const kind = config.kind || (config.teleport ? 'teleport' : 'reward');
    return Object.freeze(Object.assign({}, config, {
      index,
      kind,
      teleport: kind === 'teleport' || !!config.teleport,
      probability: probabilities[index] || 1,
      probabilityDenominator: getPlinkoSlotDenominator(slotCount),
      reward: Object.freeze(config.reward || {})
    }));
  }

  function plinkoBoard(boardId, slots, pityReward, options) {
    const settings = options || {};
    const normalizedSlots = slots || [];
    const slotCount = normalizedSlots.length;
    return Object.freeze({
      id: boardId,
      tier: settings.tier || boardId,
      stage: settings.stage || 'main',
      title: settings.title || `${boardId} Plinko`,
      slotCount,
      slots: Object.freeze(normalizedSlots.map((slot, index) => plinkoSlot(Object.assign({ boardId, index }, slot), slotCount))),
      pityReward: Object.freeze(pityReward || {})
    });
  }

  const PLINKO_BOARD_SLOTS = Object.freeze([
    Object.freeze({ id: 'jackpot_left', label: 'Left Jackpot', tone: '#f0648f', jackpot: true }),
    Object.freeze({ id: 'edge_left', label: 'Edge Prize', tone: '#ff9f5a' }),
    Object.freeze({ id: 'rare_left', label: 'Rare Prize', tone: '#b785ff' }),
    Object.freeze({ id: 'uncommon_left', label: 'Bonus Prize', tone: '#71d99b' }),
    Object.freeze({ id: 'center', label: 'Common Prize', tone: '#8bc7ff', common: true }),
    Object.freeze({ id: 'uncommon_right', label: 'Bonus Prize', tone: '#71d99b' }),
    Object.freeze({ id: 'rare_right', label: 'Rare Prize', tone: '#b785ff' }),
    Object.freeze({ id: 'edge_right', label: 'Edge Prize', tone: '#ff9f5a' }),
    Object.freeze({ id: 'jackpot_right', label: 'Right Jackpot', tone: '#f0648f', jackpot: true })
  ]);

  const PLINKO_BOARDS = Object.freeze({
    basic: plinkoBoard('basic', [
      { id: 'basic_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'basic_bonus', rewardFamily: 'teleport' },
      { id: 'basic_prism_cache', label: 'Prism Cache', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'basic', reward: { materials: { cubeFragment: 10 }, consumables: { potential_cube: 1 } } },
      { id: 'basic_card', label: 'Starter Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'common', reward: { cards: { gel_spark: 1 } } },
      { id: 'basic_coupon', label: 'Rate Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'basic', reward: { consumables: { xp_coupon_1_2_1h: 1 } } },
      { id: 'basic_dust', label: 'Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 14 } } },
      { id: 'basic_catalyst', label: 'Catalyst', tone: '#71d99b', rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 18, upgradeCatalyst: 1 } } },
      { id: 'basic_cutlass', label: 'Gear', tone: '#b785ff', rewardFamily: 'gear', rewardTier: 'uncommon', reward: { items: [{ itemId: 'adventurer_cutlass', rarity: 'Uncommon' }] } },
      { id: 'basic_slot_coupon', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { usable_slot_coupon: 1 } } },
      { id: 'basic_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'basic_bonus', rewardFamily: 'teleport' }
    ], { currency: 900, materials: { upgradeDust: 30, upgradeCatalyst: 2 }, consumables: { drop_coupon_1_2_1h: 1 } }, { tier: 'basic', stage: 'main', title: 'Starfall Board' }),
    basic_bonus: plinkoBoard('basic_bonus', [
      { id: 'basic_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'basic_apex', rewardFamily: 'teleport' },
      { id: 'basic_bonus_coin_pack', label: 'Charm Cache', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'uncommon', reward: { items: [{ itemId: 'wanderer_charm', rarity: 'Uncommon' }], materials: { upgradeCatalyst: 2 } } },
      { id: 'basic_bonus_card', label: 'Wild Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'uncommon', reward: { cards: { vinebinder_loop: 1 } } },
      { id: 'basic_bonus_materials', label: 'Dust Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 36, cubeFragment: 6 } } },
      { id: 'basic_bonus_prisms', label: 'Prism Pair', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'basic', reward: { consumables: { potential_cube: 1, preservation_cube: 1 } } },
      { id: 'basic_bonus_slot_coupon', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { equipment_slot_coupon: 1 } } },
      { id: 'basic_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'basic_apex', rewardFamily: 'teleport' }
    ], { currency: 1600, materials: { upgradeDust: 44, upgradeCatalyst: 3 }, consumables: { xp_coupon_1_5_1h: 1 } }, { tier: 'basic', stage: 'bonus', title: 'Starfall Bonus Board' }),
    basic_apex: plinkoBoard('basic_apex', [
      { id: 'basic_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'rare', reward: { currency: 4200, items: [{ itemId: 'vanguard_blade', rarity: 'Rare', upgrade: 1 }], consumables: { drop_coupon_1_5_1h: 1 } } },
      { id: 'basic_apex_prism_cache', label: 'Prism Cache', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'basic', reward: { materials: { cubeFragment: 18 }, consumables: { potential_cube: 2 } } },
      { id: 'basic_apex_bundle', label: 'Apex Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'basic', reward: { materials: { upgradeDust: 62, upgradeCatalyst: 3 } } },
      { id: 'basic_apex_slot_coupon', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'basic', reward: { consumables: { usable_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'basic_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'rare', reward: { currency: 4600, items: [{ itemId: 'runewoven_robes', rarity: 'Rare', upgrade: 1 }], consumables: { xp_coupon_1_5_1h: 1 } } }
    ], { currency: 2400, materials: { upgradeDust: 60, upgradeCatalyst: 4 }, consumables: { drop_coupon_1_5_1h: 1 } }, { tier: 'basic', stage: 'apex', title: 'Starfall Apex Board' }),
    polished: plinkoBoard('polished', [
      { id: 'polished_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'polished_bonus', rewardFamily: 'teleport' },
      { id: 'polished_rare_gear', label: 'Rare Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'rare', reward: { items: [{ itemId: 'vanguard_blade', rarity: 'Rare', upgrade: 1 }] } },
      { id: 'polished_card', label: 'Rare Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'rare', reward: { cards: { mimic_cache: 1 } } },
      { id: 'polished_coupon', label: 'Rate Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'rare', reward: { consumables: { xp_coupon_1_5_1h: 1 } } },
      { id: 'polished_dust', label: 'Dust Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 48, upgradeCatalyst: 1 } } },
      { id: 'polished_core', label: 'Core Bundle', tone: '#71d99b', rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeCatalyst: 4, refinementCore: 1 } } },
      { id: 'polished_prism', label: 'Prism', tone: '#b785ff', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 1, preservation_cube: 1 } } },
      { id: 'polished_storage', label: 'Slot Coupon', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'polished_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'polished_bonus', rewardFamily: 'teleport' }
    ], { currency: 3200, materials: { upgradeCatalyst: 6, cubeFragment: 14, refinementCore: 1 }, consumables: { xp_coupon_1_5_1h: 1 } }, { tier: 'polished', stage: 'main', title: 'Polished Board' }),
    polished_bonus: plinkoBoard('polished_bonus', [
      { id: 'polished_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'polished_apex', rewardFamily: 'teleport' },
      { id: 'polished_bonus_currency', label: 'Gear Trove', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'rare', reward: { items: [{ itemId: 'thornroot_staff', rarity: 'Rare', upgrade: 1 }], materials: { upgradeCatalyst: 6 } } },
      { id: 'polished_bonus_card', label: 'Rare Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'rare', reward: { cards: { astral_index: 1 } } },
      { id: 'polished_bonus_materials', label: 'Core Bundle', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 82, upgradeCatalyst: 5, cubeFragment: 16 } } },
      { id: 'polished_bonus_prism', label: 'Prism Trove', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 2, preservation_cube: 1 } } },
      { id: 'polished_bonus_slot_coupon', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 1, usable_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'polished_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'polished_apex', rewardFamily: 'teleport' }
    ], { currency: 5200, materials: { upgradeCatalyst: 9, cubeFragment: 24, refinementCore: 2 }, consumables: { drop_coupon_1_5_1h: 1 } }, { tier: 'polished', stage: 'bonus', title: 'Polished Bonus Board' }),
    polished_apex: plinkoBoard('polished_apex', [
      { id: 'polished_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'epic', reward: { currency: 11500, items: [{ itemId: 'thorncrown_greatsword', rarity: 'Epic', upgrade: 2 }], consumables: { drop_coupon_2_0_1h: 1 } } },
      { id: 'polished_apex_prism', label: 'Prism Trove', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'rare', reward: { consumables: { potential_cube: 3, preservation_cube: 1 } } },
      { id: 'polished_apex_materials', label: 'Apex Core', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'rare', reward: { materials: { upgradeDust: 120, upgradeCatalyst: 8, refinementCore: 2 } } },
      { id: 'polished_apex_slot_coupon', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'rare', reward: { consumables: { equipment_slot_coupon: 2, card_slot_coupon: 1 } } },
      { id: 'polished_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'epic', reward: { currency: 12500, items: [{ itemId: 'briarstring_longbow', rarity: 'Epic', upgrade: 2 }], consumables: { xp_coupon_2_0_1h: 1 } } }
    ], { currency: 7800, materials: { upgradeCatalyst: 12, cubeFragment: 30, refinementCore: 3 }, consumables: { xp_coupon_2_0_1h: 1 } }, { tier: 'polished', stage: 'apex', title: 'Polished Apex Board' }),
    meteor: plinkoBoard('meteor', [
      { id: 'meteor_left_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'meteor_bonus', rewardFamily: 'teleport' },
      { id: 'meteor_epic_gear', label: 'Epic Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'thorncrown_greatsword', rarity: 'Epic', upgrade: 2 }] } },
      { id: 'meteor_epic_card', label: 'Epic Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'epic', reward: { cards: { rift_splinter: 1 } } },
      { id: 'meteor_coupon', label: 'Power Coupon', tone: '#71d99b', rewardFamily: 'rate_coupon', rewardTier: 'epic', reward: { consumables: { xp_coupon_2_0_1h: 1, drop_coupon_1_5_1h: 1 } } },
      { id: 'meteor_dust', label: 'Meteor Dust', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 72 } } },
      { id: 'meteor_gear_cache', label: 'Meteor Gear Cache', tone: '#71d99b', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'stormtalon_saber', rarity: 'Epic', upgrade: 1 }] } },
      { id: 'meteor_prism', label: 'Prism Trove', tone: '#b785ff', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 3, preservation_cube: 1 } } },
      { id: 'meteor_slot_pack', label: 'Slot Pack', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 1, usable_slot_coupon: 1, etc_slot_coupon: 1, card_slot_coupon: 1 } } },
      { id: 'meteor_right_gate', label: 'Bonus Gate', tone: '#5bd7ff', kind: 'teleport', nextBoardId: 'meteor_bonus', rewardFamily: 'teleport' }
    ], { currency: 12500, materials: { upgradeCatalyst: 18, cubeFragment: 36, refinementCore: 4 }, consumables: { drop_coupon_2_0_1h: 1 }, cards: { stormbreak_plume: 1 } }, { tier: 'meteor', stage: 'main', title: 'Meteor Board' }),
    meteor_bonus: plinkoBoard('meteor_bonus', [
      { id: 'meteor_bonus_left_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'meteor_apex', rewardFamily: 'teleport' },
      { id: 'meteor_bonus_currency', label: 'Meteor Gear', tone: '#ff9f5a', rewardFamily: 'gear', rewardTier: 'epic', reward: { items: [{ itemId: 'stormtalon_saber', rarity: 'Epic', upgrade: 2 }], materials: { upgradeCatalyst: 18, refinementCore: 4 } } },
      { id: 'meteor_bonus_card', label: 'Epic Card', tone: '#b785ff', rewardFamily: 'card', rewardTier: 'epic', reward: { cards: { stormbreak_plume: 1 } } },
      { id: 'meteor_bonus_materials', label: 'Core Trove', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 110, upgradeCatalyst: 8, cubeFragment: 20, refinementCore: 2 } } },
      { id: 'meteor_bonus_prism', label: 'Prism Vault', tone: '#71d99b', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 4, preservation_cube: 2 } } },
      { id: 'meteor_bonus_slot_pack', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 2, usable_slot_coupon: 1, etc_slot_coupon: 1, card_slot_coupon: 2 } } },
      { id: 'meteor_bonus_right_gate', label: 'Apex Gate', tone: '#80f0ff', kind: 'teleport', nextBoardId: 'meteor_apex', rewardFamily: 'teleport' }
    ], { currency: 18000, materials: { upgradeCatalyst: 24, cubeFragment: 52, refinementCore: 6 }, consumables: { drop_coupon_2_0_1h: 1 } }, { tier: 'meteor', stage: 'bonus', title: 'Meteor Bonus Board' }),
    meteor_apex: plinkoBoard('meteor_apex', [
      { id: 'meteor_apex_left_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'relic', reward: { currency: 42000, items: [{ itemId: 'eclipse_edge', rarity: 'Relic', upgrade: 4 }], consumables: { drop_coupon_2_0_1h: 2 } } },
      { id: 'meteor_apex_prism', label: 'Prism Vault', tone: '#ff9f5a', rewardFamily: 'prism', rewardTier: 'epic', reward: { consumables: { potential_cube: 6, preservation_cube: 3 } } },
      { id: 'meteor_apex_materials', label: 'Apex Trove', tone: '#8bc7ff', common: true, rewardFamily: 'materials', rewardTier: 'epic', reward: { materials: { upgradeDust: 170, upgradeCatalyst: 12, cubeFragment: 32, refinementCore: 4 } } },
      { id: 'meteor_apex_slot_pack', label: 'Slot Vault', tone: '#ff9f5a', rewardFamily: 'slot_coupon', rewardTier: 'epic', reward: { consumables: { equipment_slot_coupon: 2, usable_slot_coupon: 2, etc_slot_coupon: 2, card_slot_coupon: 2 } } },
      { id: 'meteor_apex_right_prize', label: 'Apex Prize', tone: '#f0648f', jackpot: true, rewardFamily: 'gear', rewardTier: 'relic', reward: { currency: 46000, items: [{ itemId: 'corona_longbow', rarity: 'Relic', upgrade: 4 }], consumables: { xp_coupon_2_0_1h: 2 } } }
    ], { currency: 30000, materials: { upgradeCatalyst: 32, cubeFragment: 80, refinementCore: 9 }, consumables: { drop_coupon_2_0_1h: 1 }, cards: { eclipse_corona: 1 } }, { tier: 'meteor', stage: 'apex', title: 'Meteor Apex Board' })
  });

  const PLINKO_REWARD_TABLES = PLINKO_BOARDS;

  const api = {
    PLINKO_BOUNCE_COUNT,
    PLINKO_ACTIVE_DROP_LIMIT,
    PLINKO_PITY_TARGET,
    PLINKO_SLOT_PROBABILITIES,
    PLINKO_SLOT_DENOMINATOR,
    PLINKO_SLOT_PROBABILITY_TABLES,
    PLINKO_BOARD_SLOTS,
    PLINKO_BOARDS,
    PLINKO_REWARD_TABLES
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.plinko = Object.assign({}, modules.plinko || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
