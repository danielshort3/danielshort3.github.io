(function (root, factory) {
  'use strict';
  const api = factory(typeof module === 'object' && module.exports ? require('./numbers.js') : root.WayfarersNumbers);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersProgressionModifiers = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (N) {
  'use strict';

  // Core computes the canonical crew, charm, relic, meal, preparation and route
  // effects, including supply shortages and verified account entitlements. Reuse
  // those ratios for physical project work; never apply them to the wallet twice.
  function ratio(value, baseline) {
    if (!(baseline > 0) || value === undefined) return 1;
    const result = N.toNumber(N.div(value, baseline));
    return Number.isFinite(result) ? Math.max(0, result) : 1e12;
  }
  function apply(state, raw, ownership, options) {
    const rates = options && options.rates;
    if (!rates) return raw;
    const areas = Object.fromEntries(Object.entries(raw.areas).map(([id, value]) => [id, { ...value }]));
    const project = raw.areas[state.expedition.projectArea];
    const travel = ratio(rates.travel, project.work + project.finale);
    const output = id => ratio(rates.gain[id], raw.gain[id]);
    const scale = (id, keys, multiplier) => {
      if (!areas[id]) return;
      for (const key of keys) if (typeof areas[id][key] === 'number') areas[id][key] *= multiplier;
    };
    scale('greenway', ['work', 'finale', 'travel'], travel);
    scale('quarry', ['work', 'finale'], output('ore'));
    scale('watchtower', ['work', 'finale', 'research', 'repair', 'beacon'], output('knowledge'));
    scale('workshop', ['work', 'finale'], output('provisions'));
    scale('ruins', ['work', 'finale'], output('herbs'));
    scale('ruins', ['research'], output('knowledge'));
    scale('harbor', ['work', 'finale', 'travel', 'voyagePace'], travel);
    scale('harbor', ['research'], output('maps'));
    if (areas.harbor && travel > 0) areas.harbor.duration /= travel;
    const researchRate = Object.values(areas).reduce((sum, area) => sum + (area.research || 0), 0) * .6;
    return { ...raw, areas, researchRate };
  }
  return { apply };
});
