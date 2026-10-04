(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersCollectionContent = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const CARDS = [
    ['trail-courier', 'Trail Courier', 'greenway', 'common', 'journey', { coins: .08 }, 'Trail delivery coins +8%'],
    ['trail-cartographer', 'Trail Cartographer', 'greenway', 'rare', 'journey', { maps: .12 }, 'Map output +12%'],
    ['trail-stag', 'Silver Stag', 'greenway', 'epic', 'journey', { travel: .1, cargo: .05 }, 'Trail travel +10%; new voyage cargo +5%'],
    ['quarry-mole', 'Quarry Mole', 'quarry', 'common', 'industry', { picks: .1 }, 'Quarry extraction capacity +10%'],
    ['quarry-hauler', 'Stone Hauler', 'quarry', 'rare', 'industry', { haul: .15 }, 'Quarry hauling capacity +15%'],
    ['quarry-salamander', 'Ember Salamander', 'quarry', 'epic', 'industry', { smelt: .12, oreYield: .05 }, 'Refining capacity +12%; ore yield +5%'],
    ['tower-scribe', 'Tower Scribe', 'watchtower', 'common', 'discovery', { knowledge: .1 }, 'Knowledge output +10%'],
    ['tower-signalist', 'Signal Keeper', 'watchtower', 'rare', 'discovery', { haul: .08, maps: .08 }, 'Quarry hauling +8%; map output +8%'],
    ['tower-astronomer', 'Star Astronomer', 'watchtower', 'legendary', 'discovery', { research: .15, maps: .1 }, 'Commission research +15%; map output +10%'],
    ['workshop-tinker', 'Workshop Tinker', 'workshop', 'common', 'industry', { assembly: .1 }, 'Workshop assembly capacity +10%'],
    ['workshop-smith', 'Alloy Smith', 'workshop', 'rare', 'industry', { oreYield: .08, workshopYield: .08 }, 'Ore yield and manufacturing yield +8%'],
    ['workshop-clockwork', 'Clockwork Helper', 'workshop', 'epic', 'industry', { oreSaving: .12, assembly: .08 }, 'Workshop ore per item −12%; assembly capacity +8%'],
    ['ruins-delver', 'Ruins Delver', 'ruins', 'common', 'discovery', { delving: .12 }, 'Ruins delving capacity +12%'],
    ['ruins-restorer', 'Relic Restorer', 'ruins', 'rare', 'discovery', { recovery: .12, artifacts: .08 }, 'Recovery capacity +12%; restored artifact support +8%'],
    ['ruins-oracle', 'Moss Oracle', 'ruins', 'epic', 'discovery', { interpretation: .12, research: .1 }, 'Interpretation capacity +12%; commission research +10%'],
    ['harbor-deckhand', 'Harbor Deckhand', 'harbor', 'common', 'journey', { cargo: .1 }, 'New voyage cargo +10%'],
    ['harbor-navigator', 'Ocean Navigator', 'harbor', 'rare', 'journey', { voyage: .15 }, 'Funded voyage travel capacity +15%'],
    ['harbor-leviathan', 'Gentle Leviathan', 'harbor', 'legendary', 'journey', { cargo: .16, voyage: .08 }, 'New voyage cargo +16%; funded voyage travel +8%']
  ].map(([id, name, area, rarity, tag, effects, description]) => ({ id, artId: id, name, area, rarity, tag, effects, description }));
  const GEAR = [
    ['quarry-pick', 'Quarry Pick', 'tool', 'quarry', 'common', { ore: 30, coins: 250 }, { picks: .08 }, { picks: .012 }, 'Extraction capacity'],
    ['clockwork-wrench', 'Clockwork Wrench', 'tool', 'workshop', 'rare', { ore: 250, knowledge: 150 }, { assembly: .08 }, { assembly: .012 }, 'Assembly capacity'],
    ['survey-hood', 'Survey Hood', 'head', 'watchtower', 'common', { ore: 40, coins: 500 }, { knowledge: .06, maps: .04 }, { knowledge: .009, maps: .006 }, 'Knowledge and maps'],
    ['captain-hat', 'Captain’s Hat', 'head', 'harbor', 'rare', { provisions: 1500, maps: 600 }, { cargo: .08 }, { cargo: .012 }, 'New voyage cargo'],
    ['porter-coat', 'Porter’s Coat', 'coat', 'quarry', 'common', { ore: 45, coins: 600 }, { haul: .08 }, { haul: .012 }, 'Hauling capacity'],
    ['scholar-robe', 'Scholar’s Robe', 'coat', 'ruins', 'rare', { herbs: 600, knowledge: 500 }, { research: .08 }, { research: .012 }, 'Commission research'],
    ['trail-boots', 'Trail Boots', 'boots', 'greenway', 'common', { ore: 35, coins: 300 }, { travel: .08 }, { travel: .012 }, 'Trail travel'],
    ['deck-boots', 'Deck Boots', 'boots', 'harbor', 'rare', { provisions: 1200, ore: 800 }, { voyage: .08 }, { voyage: .012 }, 'Funded voyage travel']
  ].map(([id, name, slot, area, rarity, costs, effects, perPoint, description]) => ({ id, artId: id, name, slot, area, rarity, costs, effects, perPoint, description, slots: 6 }));
  const SCROLLS = [
    { id: 'steady', name: 'Steady Scroll', success: 1, points: 1, weight: 50 },
    { id: 'bold', name: 'Bold Scroll', success: .6, points: 2, weight: 32 },
    { id: 'brilliant', name: 'Brilliant Scroll', success: .15, points: 5, weight: 15 },
    { id: 'restoration', name: 'Restoration Scroll', success: 1, points: 0, weight: 3 }
  ];
  const RARITIES = ['common', 'rare', 'epic', 'legendary'];
  const WEIGHTS = { common: 62, rare: 28, epic: 9, legendary: 1 };
  const INK = { common: 1, rare: 3, epic: 9, legendary: 27 };
  const RANK_SCALE = [0, 1, 1.2, 1.36, 1.48, 1.56];
  const FUSION = [0, 2, 3, 5, 8];
  const SLOTS = ['tool', 'head', 'coat', 'boots'];
  return { CARDS, GEAR, SCROLLS, RARITIES, WEIGHTS, INK, RANK_SCALE, FUSION, SLOTS };
});
