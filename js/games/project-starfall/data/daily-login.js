(function initProjectStarfallDataDailyLogin(global) {
  'use strict';

  function defaultFreezeQuestReward(reward) {
    const source = reward || {};
    const frozen = Object.assign({}, source);
    ['materials', 'consumables', 'items', 'timedBuffs', 'permanentStats'].forEach((key) => {
      if (frozen[key] && typeof frozen[key] === 'object') frozen[key] = Object.freeze(frozen[key]);
    });
    return Object.freeze(frozen);
  }

  function createDailyLoginData(options) {
    const settings = options || {};
    const freezeQuestReward = typeof settings.freezeQuestReward === 'function'
      ? settings.freezeQuestReward
      : defaultFreezeQuestReward;

    const DAILY_LOGIN_REWARDS = Object.freeze([
      Object.freeze({
        id: 'daily_field_supplies',
        day: 1,
        name: 'Field Supplies',
        summary: 'Potions and dust for the next short route.',
        reward: freezeQuestReward({ consumables: { minor_health_potion: 3, minor_resource_tonic: 2 }, materials: { upgradeDust: 4 } })
      }),
      Object.freeze({
        id: 'daily_upgrade_sparks',
        day: 2,
        name: 'Upgrade Sparks',
        summary: 'A small upgrade bundle for early gear attempts.',
        reward: freezeQuestReward({ currency: 120, materials: { upgradeDust: 8, upgradeCatalyst: 1 } })
      }),
      Object.freeze({
        id: 'daily_route_boost',
        day: 3,
        name: 'Route Boost',
        summary: 'A light buff kit for focused field grinding.',
        reward: freezeQuestReward({ consumables: { guard_tonic: 1, swiftstep_oil: 1, xp_coupon_1_2_1h: 1 } })
      }),
      Object.freeze({
        id: 'daily_material_run',
        day: 4,
        name: 'Material Run',
        summary: 'General route materials that stay useful while upgrading.',
        reward: freezeQuestReward({ materials: { upgradeDust: 10, gelDrop: 4, oreChunks: 4 } })
      }),
      Object.freeze({
        id: 'daily_card_spark',
        day: 5,
        name: 'Card Spark',
        summary: 'Star Card materials and a starter card for collection progress.',
        reward: freezeQuestReward({ materials: { whiteStarCard: 3, greenStarCard: 1 }, cards: [{ cardId: 'gel_spark', rank: 1, quantity: 1 }] })
      }),
      Object.freeze({
        id: 'daily_prism_prep',
        day: 6,
        name: 'Prism Prep',
        summary: 'Attunement support without skipping the gear chase.',
        reward: freezeQuestReward({ materials: { cubeFragment: 5, upgradeCatalyst: 2 }, consumables: { potential_cube: 1 } })
      }),
      Object.freeze({
        id: 'daily_weekly_cache',
        day: 7,
        name: 'Weekly Star Cache',
        summary: 'A notable weekly bundle with tokens, Plinko, and upgrade support.',
        reward: freezeQuestReward({ currency: 420, starTokens: 35, materials: { upgradeDust: 18, upgradeCatalyst: 2, refinementCore: 1 }, consumables: { plinko_ball_basic: 3, drop_coupon_1_2_1h: 1 } })
      })
    ]);

    const DAILY_LOGIN_MILESTONES = Object.freeze([
      Object.freeze({
        id: 'login_7_day_route',
        days: 7,
        name: 'Seven-Day Route',
        summary: 'A first-week loyalty cache for keeping the route warm.',
        reward: freezeQuestReward({ starTokens: 60, materials: { upgradeDust: 24, upgradeCatalyst: 3 }, consumables: { plinko_ball_basic: 4 } })
      }),
      Object.freeze({
        id: 'login_14_day_starlit',
        days: 14,
        name: 'Starlit Fortnight',
        summary: 'A cosmetic attendance spark and collection materials.',
        reward: freezeQuestReward({ starTokens: 80, cosmeticId: 'starlit_checkin', materials: { greenStarCard: 3, blueStarCard: 1, cubeFragment: 8 } })
      }),
      Object.freeze({
        id: 'login_30_day_veteran',
        days: 30,
        name: 'Monthly Route Veteran',
        summary: 'A practical long-session bundle with a rare card reward.',
        reward: freezeQuestReward({ starTokens: 140, materials: { upgradeDust: 48, refinementCore: 2, blueStarCard: 2 }, consumables: { potential_cube: 2, preservation_cube: 1 }, cards: [{ cardId: 'mimic_cache', rank: 1, quantity: 1 }] })
      }),
      Object.freeze({
        id: 'login_60_day_constellation',
        days: 60,
        name: 'Constellation Habit',
        summary: 'A long-term cosmetic and bounded account-style utility bundle.',
        reward: freezeQuestReward({ starTokens: 180, cosmeticId: 'constellation_trail', materials: { cubeFragment: 20, refinementCore: 3 }, consumables: { usable_slot_coupon: 1, etc_slot_coupon: 1 } })
      }),
      Object.freeze({
        id: 'login_90_day_horizon',
        days: 90,
        name: 'Horizon Regular',
        summary: 'A deep-grind support bundle with no best-in-slot shortcut.',
        reward: freezeQuestReward({ starTokens: 240, materials: { upgradeCatalyst: 12, refinementCore: 4, purpleStarCard: 2 }, consumables: { preservation_cube: 2, line_catalyst: 1 }, cards: [{ cardId: 'bramble_heart', rank: 1, quantity: 1 }] })
      }),
      Object.freeze({
        id: 'login_180_day_astral',
        days: 180,
        name: 'Astral Half-Year',
        summary: 'Prestige materials and a small finite stat keepsake for committed players.',
        reward: freezeQuestReward({ starTokens: 420, materials: { cubeFragment: 36, refinementCore: 6, orangeStarCard: 1 }, consumables: { advanced_skill_manual: 1, equipment_slot_coupon: 1 }, permanentStats: { hp: 40, defense: 2 } })
      }),
      Object.freeze({
        id: 'login_365_day_comet',
        days: 365,
        name: 'Comet Year',
        summary: 'Aspirational year-long cosmetic prestige with a modest permanent keepsake.',
        reward: freezeQuestReward({ starTokens: 900, cosmeticId: 'comet_year_splat', materials: { cubeFragment: 60, refinementCore: 10, orangeStarCard: 3 }, consumables: { card_slot_coupon: 1, preservation_cube: 3, line_catalyst: 2 }, permanentStats: { power: 3, mpMax: 35 } })
      })
    ]);

    return Object.freeze({
      DAILY_LOGIN_REWARDS,
      DAILY_LOGIN_MILESTONES
    });
  }

  const defaultDailyLoginData = createDailyLoginData();
  const api = Object.assign({
    createDailyLoginData,
    defaultFreezeQuestReward
  }, defaultDailyLoginData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.dailyLogin = Object.assign({}, modules.dailyLogin || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
