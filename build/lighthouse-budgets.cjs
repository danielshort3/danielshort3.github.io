'use strict';
// Staged mobile laboratory ceiling, not a real-user Core Web Vitals pass.
// Keep the independent, tighter per-route regression limits unchanged.
const ABSOLUTE = Object.freeze({ lcp: 3500, tbt: 200, cls: 0.1 });
function assertBudgets(result, before) {
  if (!before) throw Error(`Missing reviewed baseline: ${result.route}`);
  for (const [key, limit] of Object.entries(ABSOLUTE)) {
    const value = result.median?.[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw Error(`${result.route}: invalid ${key} measurement`);
    }
    if (value > limit) throw Error(`${result.route}: ${key} ${value} exceeds absolute lab budget ${limit}`);
  }
  for (const [key, floor] of [['lcp', 250], ['tbt', 50]]) {
    if (typeof before[key] !== 'number' || !Number.isFinite(before[key]) || before[key] < 0) {
      throw Error(`${result.route}: invalid ${key} baseline`);
    }
    if (result.median[key] > before[key] * 1.2 + floor) {
      throw Error(`${result.route}: ${key} regressed beyond baseline tolerance`);
    }
  }
}
module.exports = { ABSOLUTE, assertBudgets };
