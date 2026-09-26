'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { compareLighthouseRoute } = require('../../build/lighthouse-regression.cjs');

const before = { lcp: 3029.0256, cls: 0, tbt: 19.5 };

function result(candidateTbt, referenceTbt) {
  const runs = candidateTbt.map(tbt => ({ tbt }));
  return {
    route: '/portfolio/website',
    runs,
    referenceRuns: referenceTbt && referenceTbt.map(tbt => ({ tbt })),
    median: { cls: 0, lcp: 3100, tbt: [...candidateTbt].sort((a, b) => a - b)[1] }
  };
}

test('matched fast and slow runners keep the reviewed 73.4 ms TBT limit', () => {
  const fast = compareLighthouseRoute(before, result([8, 9.5, 11], [8, 9.5, 11]));
  const slow = compareLighthouseRoute(before, result([72.5, 74.5, 76], [72.5, 74.5, 76]));
  assert.equal(fast.limits.tbt, 73.4);
  assert.equal(slow.limits.tbt, 73.4);
  assert.equal(fast.adjustedTbt, 19.5);
  assert.equal(slow.adjustedTbt, 19.5);
  assert.deepEqual(slow.failures, []);
});

test('a genuine 60 ms paired regression fails even on a slow runner', () => {
  const comparison = compareLighthouseRoute(before, result([132.5, 134.5, 136], [72.5, 74.5, 76]));
  assert.equal(comparison.tbtDelta, 60);
  assert.equal(comparison.adjustedTbt, 79.5);
  assert.match(comparison.failures[0], /tbt regressed/);
});

test('local mode keeps the raw TBT gate', () => {
  const comparison = compareLighthouseRoute(before, result([72.5, 74.5, 76]));
  assert.equal(comparison.adjustedTbt, 74.5);
  assert.match(comparison.failures[0], /tbt regressed/);
});

test('absolute LCP and CLS gates still fail in paired mode', () => {
  const candidate = result([72.5, 74.5, 76], [72.5, 74.5, 76]);
  candidate.median.lcp = 3900;
  candidate.median.cls = 0.11;
  const comparison = compareLighthouseRoute(before, candidate);
  assert.equal(comparison.failures.length, 2);
  assert.match(comparison.failures[0], /CLS/);
  assert.match(comparison.failures[1], /lcp/);
});

test('a missing reference pair is an error', () => {
  assert.throws(() => compareLighthouseRoute(before, result([75, 76, 77], [75, 76])), /three matched/);
});
