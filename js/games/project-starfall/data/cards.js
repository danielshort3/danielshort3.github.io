(function initProjectStarfallDataCards(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataAssets = (typeof require === 'function' ? require('./assets.js') : null) || DataModules.assets || {};
  const CARD_ICON_ROOT = DataAssets.CARD_ICON_ROOT;

  const CARD_DEFINITIONS = Object.freeze([
    Object.freeze({ id: 'gel_spark', name: 'Gel Spark', icon: 'GS', rarity: 'Common', tags: Object.freeze(['Ooze', 'Starter']), summary: 'Small HP and resource-flow card for early grinding.', baseStats: Object.freeze({ hp: 24, resourceGain: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'mossguard_oath', name: 'Mossguard Oath', icon: 'MO', rarity: 'Common', tags: Object.freeze(['Guard', 'Forest']), summary: 'Steady defense for safer platform routes.', baseStats: Object.freeze({ hp: 18, defense: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'thorn_focus', name: 'Thorn Focus', icon: 'TF', rarity: 'Common', tags: Object.freeze(['Focus', 'Forest']), summary: 'Low-rarity power with a light mobbing bonus.', baseStats: Object.freeze({ power: 2, areaDamage: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'bristle_charge', name: 'Bristle Charge', icon: 'BC', rarity: 'Common', tags: Object.freeze(['Beast', 'Mobility']), summary: 'Movement and power for aggressive farming lanes.', baseStats: Object.freeze({ speed: 8, power: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'clockwork_patience', name: 'Clockwork Patience', icon: 'CP', rarity: 'Common', tags: Object.freeze(['Construct', 'Guard']), summary: 'A defensive card for learning boss timings.', baseStats: Object.freeze({ defense: 2, block: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'ember_glint', name: 'Ember Glint', icon: 'EG', rarity: 'Common', tags: Object.freeze(['Cinder', 'Burn']), summary: 'Entry burn support with a flat power bump.', baseStats: Object.freeze({ burnDamage: 2, power: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'hunter_tempo', name: 'Hunter Tempo', icon: 'HT', rarity: 'Common', tags: Object.freeze(['Bandit', 'Crit']), summary: 'Crit and movement for active combat routes.', baseStats: Object.freeze({ crit: 1, speed: 6 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'frost_thread', name: 'Frost Thread', icon: 'FT', rarity: 'Common', tags: Object.freeze(['Frost', 'Guard']), summary: 'Early percent HP with enough defense to matter.', baseStats: Object.freeze({ maxHpPercent: 1, defense: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'vinebinder_loop', name: 'Vinebinder Loop', icon: 'VL', rarity: 'Uncommon', tags: Object.freeze(['Forest', 'Mobbing']), summary: 'Area pressure with smoother resource returns.', baseStats: Object.freeze({ areaDamage: 3, resourceGain: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'rustcoil_lens', name: 'Rustcoil Lens', icon: 'RL', rarity: 'Uncommon', tags: Object.freeze(['Construct', 'Break']), summary: 'Break-focused card for armored monsters.', baseStats: Object.freeze({ armorBreak: 3, defensePercent: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'ashflare_core', name: 'Ashflare Core', icon: 'AC', rarity: 'Uncommon', tags: Object.freeze(['Cinder', 'Burn']), summary: 'Burn damage with a light percent power bonus.', baseStats: Object.freeze({ burnDamage: 5, powerPercent: 1 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'storm_fletching', name: 'Storm Fletching', icon: 'SF', rarity: 'Uncommon', tags: Object.freeze(['Storm', 'Ranged']), summary: 'Range and crit for safer ranged grinding.', baseStats: Object.freeze({ range: 12, crit: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'cloudcall_vellum', name: 'Cloudcall Vellum', icon: 'CV', rarity: 'Uncommon', tags: Object.freeze(['Support', 'Resource']), summary: 'Resource uptime for skill-heavy rotations.', baseStats: Object.freeze({ mpMax: 20, resourceGainPercent: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'astral_index', name: 'Astral Index', icon: 'AI', rarity: 'Rare', tags: Object.freeze(['Astral', 'Crit']), summary: 'Precision card for builds that scale crit damage.', baseStats: Object.freeze({ critDamage: 8, power: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'bramble_heart', name: 'Bramble Heart', icon: 'BH', rarity: 'Rare', tags: Object.freeze(['Boss', 'Guard']), summary: 'Boss-farm survival card from deep forest routes.', baseStats: Object.freeze({ hp: 60, maxHpPercent: 2, defense: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'titan_gearheart', name: 'Titan Gearheart', icon: 'TG', rarity: 'Rare', tags: Object.freeze(['Boss', 'Break']), summary: 'Break and block package for construct bosses.', baseStats: Object.freeze({ armorBreak: 5, defensePercent: 2, block: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'mimic_cache', name: 'Mimic Cache', icon: 'MC', rarity: 'Rare', tags: Object.freeze(['Treasure', 'Hybrid']), summary: 'Flexible rare card with crit and resource gain.', baseStats: Object.freeze({ crit: 3, resourceGainPercent: 3 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'rift_splinter', name: 'Rift Splinter', icon: 'RS', rarity: 'Epic', tags: Object.freeze(['Rift', 'Damage']), summary: 'High-end damage card for rift and boss pushes.', baseStats: Object.freeze({ attackDamagePercent: 3, critDamage: 10 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'stormbreak_plume', name: 'Stormbreak Plume', icon: 'SP', rarity: 'Epic', tags: Object.freeze(['Storm', 'Mobility']), summary: 'Fast ranged card for evasive farming routes.', baseStats: Object.freeze({ speed: 16, crit: 4, range: 18 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'archivist_star', name: 'Archivist Star', icon: 'AS', rarity: 'Relic', tags: Object.freeze(['Astral', 'Relic']), summary: 'Relic resource and power card from late-route bosses.', baseStats: Object.freeze({ mpMax: 45, resourceGainPercent: 5, powerPercent: 2 }), rankScale: 0.35, maxRank: 5 }),
    Object.freeze({ id: 'eclipse_corona', name: 'Eclipse Corona', icon: 'EC', rarity: 'Relic', tags: Object.freeze(['Eclipse', 'Relic']), summary: 'Relic damage card for the strongest single-target loadouts.', baseStats: Object.freeze({ attackDamagePercent: 4, powerPercent: 2, critDamage: 12 }), rankScale: 0.35, maxRank: 5 })
  ]);

  const CARD_ASSETS = Object.freeze(CARD_DEFINITIONS.reduce((assets, card) => {
    assets[card.id] = `${CARD_ICON_ROOT}/${card.id}.png`;
    return assets;
  }, {}));

  const api = {
    CARD_DEFINITIONS,
    CARD_ASSETS
  };

  const modules = global.ProjectStarfallDataModules || {};
  modules.cards = Object.assign({}, modules.cards || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
