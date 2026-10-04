'use strict';

// Reproducible eligible-clock budget, not a natural area-unlock/pacing simulation.
// All six areas are deliberately available to measure the full rarity pool.
// Run: node tests/games/wayfarers-guild-collection-budget.cjs [output.json]
const fs = require('node:fs');
const { Core, clone, mature } = require('./helpers/wayfarers-progression.cjs');
const K = require('../../js/games/wayfarers-guild/collections.js');
const base = mature(), horizons = [3600, 86400, 7 * 86400, 28 * 86400];
const samples = Object.fromEntries(horizons.map(seconds => [seconds, []]));
for (let seed = 1; seed <= 64; seed += 1) {
  const state = clone(base);
  state.collection = K.create(seed * 104729);
  for (const kind of ['cards', 'equipment']) Core.act(state, { type: 'collection-unlock', kind });
  let elapsed = 0;
  for (const seconds of horizons) {
    while (elapsed < seconds) {
      const dt = Math.min(seconds - elapsed, K.nextEvent(state));
      K.tick(state, dt); elapsed += dt;
    }
    const x = clone(state.collection), rankByRarity = {};
    for (const [id, item] of Object.entries(x.cards)) {
      while (item.rank < 5 && item.copies >= K.Content.FUSION[item.rank]) { item.copies -= K.Content.FUSION[item.rank]; item.rank += 1; }
      const rarity = K.Content.CARDS.find(card => card.id === id).rarity;
      (rankByRarity[rarity] ||= []).push(item.rank);
    }
    samples[seconds].push({ cards: x.cardFinds, caches: x.scrollFinds, owned: Object.keys(x.cards).length, restoration: x.scrolls.restoration,
      scrolls: x.scrolls, ranks: Object.fromEntries(Object.entries(rankByRarity).map(([rarity, ranks]) => [rarity, ranks.reduce((a, b) => a + b, 0) / ranks.length])) });
  }
}
function distribution(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return { min: sorted[0], median: sorted[Math.floor(sorted.length / 2)], mean: values.reduce((a, b) => a + b, 0) / values.length, max: sorted.at(-1) };
}
const report = {
  method: '64 fixed seeds; exact collection event clock; all areas available from time zero; no purchased currency; no reset, clock or RNG rewind; no active bonuses. Fusion statistics greedily spend duplicates, never Ink. This is an acquisition upper-bound pool witness, not natural progression duration.',
  horizons: Object.fromEntries(horizons.map(seconds => [seconds, {
    hours: seconds / 3600,
    ...Object.fromEntries(['cards', 'caches', 'owned', 'restoration'].map(key => [key, distribution(samples[seconds].map(sample => sample[key]))])),
    scrolls: Object.fromEntries(K.Content.SCROLLS.map(def => [def.id, distribution(samples[seconds].map(sample => sample.scrolls[def.id]))])),
    meanFusedRanks: Object.fromEntries(K.Content.RARITIES.map(rarity => [rarity, distribution(samples[seconds].map(sample => sample.ranks[rarity]).filter(Number.isFinite))]))
  }])),
  economics: {
    expectedCachesPerDay: 12, expectedRestorationPerDay: .36, initialRestoration: 1,
    steady: { expectedPointsFromSixSlots: 6, maximumPoints: 6, expectedFailuresToSixSuccesses: 0 },
    bold: { expectedPointsFromSixSlots: 7.2, maximumPoints: 12, expectedFailuresToSixSuccesses: 4 },
    brilliant: { expectedPointsFromSixSlots: 4.5, maximumPoints: 30, expectedFailuresToSixSuccesses: 34 },
    perfectBrilliantItemExpectedRestorationDaysAfterStarter: 33 / .36,
    perfectBrilliantFourItemsExpectedRestorationDaysAfterStarter: 135 / .36,
    reforge: { restorationCost: 3, originalRecipeMultiplier: 10, gate: 'Workshop discovered', effect: 'All successes and failures removed together; owned/equipped base item remains.', expectedFirstAffordabilityDaysIncludingStarter: 2 / .36 },
    note: 'Expectation, not guarantee. Perfect-item estimates preserve successful points using individual failure restoration; they are not optimal Reforge bounds. Restoration is never sold or craftable. Workshop Reforge is a separate costly confirmed action removing all enhancements together. Guaranteed scrolls remain the reliable path.'
  }
};
const json = JSON.stringify(report, null, 2) + '\n';
if (process.argv[2]) fs.writeFileSync(process.argv[2], json);
else process.stdout.write(json);
