'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const { ABSOLUTE, assertBudgets } = require('../../build/lighthouse-budgets.cjs');
const baseline = { lcp: 3000, tbt: 100, cls: 0 };
const check = median => assertBudgets({ route: '/', median }, baseline);
test('absolute ceilings hold even when a slower baseline would allow a pass', () => {
  for (const key of Object.keys(ABSOLUTE)) {
    assert.throws(() => check({ lcp: 3000, tbt: 100, cls: 0, [key]: ABSOLUTE[key] + .001 }), /absolute/);
  }
});
test('existing relative tolerances are preserved', () => {
  assert.throws(() => assertBudgets({ route: '/portfolio/website', median: { lcp: 3000, tbt: 79.5, cls: 0 } }, { lcp: 3000, tbt: 19.5 }), /regressed/);
  assert.doesNotThrow(() => check({ lcp: 3200, tbt: 120, cls: 0 }));
});
test('missing, NaN, infinite and negative results cannot pass', () => {
  for (const key of Object.keys(ABSOLUTE)) for (const value of [undefined, null, NaN, Infinity, -1]) {
    assert.throws(() => check({ lcp: 3000, tbt: 100, cls: 0, [key]: value }), /invalid/);
  }
  assert.throws(() => assertBudgets({ route: '/', median: baseline }), /Missing/);
});
