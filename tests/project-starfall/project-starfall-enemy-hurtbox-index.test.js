'use strict';

const assert = require('assert');
const { performance } = require('perf_hooks');
const masks = require('../../js/games/project-starfall/data/enemy-hurtboxes.js');
const hurtboxes = require('../../js/games/project-starfall/engine/enemy-hurtboxes.js');

// Deliberately retain a linear oracle independent of the candidate index. These
// are the original world/source-space comparisons, including strict rectangle
// edges, inclusive circle tangency and floating-point transform order.
function overlaps(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function referenceRect(hurtbox, rect) {
  if (!hurtbox || !rect || !(rect.w > 0) || !(rect.h > 0) || !overlaps(hurtbox.bounds, rect)) return false;
  const x1 = (rect.x - hurtbox.originX) / hurtbox.scaleX;
  const x2 = (rect.x + rect.w - hurtbox.originX) / hurtbox.scaleX;
  const y1 = (rect.y - hurtbox.originY) / hurtbox.scaleY;
  const y2 = (rect.y + rect.h - hurtbox.originY) / hurtbox.scaleY;
  const source = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1) };
  return hurtbox.mask.rectangles.some(candidate => overlaps(candidate, source));
}

function referenceCircle(hurtbox, x, y, radius) {
  if (!hurtbox || !Number.isFinite(x) || !Number.isFinite(y) || !(radius >= 0)) return false;
  const bounds = hurtbox.bounds;
  const nearestX = Math.max(bounds.x, Math.min(x, bounds.x + bounds.w));
  const nearestY = Math.max(bounds.y, Math.min(y, bounds.y + bounds.h));
  const radiusSquared = radius * radius;
  if ((nearestX - x) ** 2 + (nearestY - y) ** 2 > radiusSquared) return false;
  return hurtbox.mask.rectangles.some(rect => {
    const x1 = hurtbox.originX + rect.x * hurtbox.scaleX;
    const x2 = hurtbox.originX + (rect.x + rect.w) * hurtbox.scaleX;
    const y1 = hurtbox.originY + rect.y * hurtbox.scaleY;
    const y2 = hurtbox.originY + (rect.y + rect.h) * hurtbox.scaleY;
    const closestX = Math.max(Math.min(x1, x2), Math.min(x, Math.max(x1, x2)));
    const closestY = Math.max(Math.min(y1, y2), Math.min(y, Math.max(y1, y2)));
    return (closestX - x) ** 2 + (closestY - y) ** 2 <= radiusSquared;
  });
}

function createCases() {
  let seed = 0x6e624eb7;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const cases = [];
  let frames = 0;
  for (const [sheetPath, sheet] of Object.entries(masks.sheets)) {
    const animation = { sheet: sheetPath, frameWidth: sheet.frameWidth, frameHeight: sheet.frameHeight };
    for (let row = 0; row < sheet.rows; row += 1) {
      for (let frameIndex = 0; frameIndex < sheet.columns; frameIndex += 1) {
        for (const facing of [-1, 1]) {
          const frame = { row, frameIndex, frameWidth: sheet.frameWidth, frameHeight: sheet.frameHeight };
          const box = { x: random() * 1000 - 500, y: random() * 500 - 250, w: 76 + random() * 102, h: 76 + random() * 102 };
          const hurtbox = hurtboxes.createEnemyHurtbox(animation, frame, box, facing);
          assert(hurtbox);
          const bounds = hurtbox.bounds;
          for (let probe = 0; probe < 24; probe += 1) {
            const x = bounds.x + (random() * 1.4 - 0.2) * bounds.w;
            const y = bounds.y + (random() * 1.4 - 0.2) * bounds.h;
            const radius = probe % 6 === 0 ? 0 : random() * (probe % 3 ? 4 : 70);
            const rect = { x, y, w: 0.01 + random() * (probe % 3 ? 6 : 100), h: 0.01 + random() * (probe % 3 ? 6 : 100) };
            cases.push({ hurtbox, rect, x, y, radius });
          }
          // Boundaries and tangencies are especially sensitive to candidate
          // pruning: include edges at band seams and fractional world scales.
          for (let index = 0; index < hurtbox.mask.rectangles.length; index += 17) {
            const source = hurtbox.mask.rectangles[index];
            const x = hurtbox.originX + source.x * hurtbox.scaleX;
            const y = hurtbox.originY + source.y * hurtbox.scaleY;
            cases.push({ hurtbox, rect: { x, y, w: 0.000001, h: 0.000001 }, x, y, radius: 0 });
            const edgeY = hurtbox.originY + (source.y + source.h) * hurtbox.scaleY;
            cases.push({ hurtbox, rect: { x, y: edgeY, w: 0.000001, h: 0.000001 }, x, y: edgeY, radius: 0 });
          }
        }
        frames += 1;
      }
    }
  }
  return { cases, frames };
}

function runBenchmark(cases) {
  const timings = { rectangle: { linear: [], indexed: [] }, circle: { linear: [], indexed: [] } };
  const sample = (fn, circle) => {
    const start = performance.now();
    let hits = 0;
    if (circle) {
      for (const query of cases) hits += Number(fn(query.hurtbox, query.x, query.y, query.radius));
    } else {
      for (const query of cases) hits += Number(fn(query.hurtbox, query.rect));
    }
    return { milliseconds: performance.now() - start, hits };
  };
  const methods = [
    ['rectangle', false, referenceRect, hurtboxes.intersectsRect],
    ['circle', true, referenceCircle, hurtboxes.intersectsCircle]
  ];
  for (let iteration = 0; iteration < 7; iteration += 1) {
    for (const [label, circle, linear, indexed] of methods) {
      const order = iteration % 2 ? [['indexed', indexed], ['linear', linear]] : [['linear', linear], ['indexed', indexed]];
      let expectedHits;
      for (const [name, fn] of order) {
        const result = sample(fn, circle);
        if (expectedHits == null) expectedHits = result.hits;
        else assert.strictEqual(result.hits, expectedHits);
        if (iteration > 1) timings[label][name].push(result.milliseconds);
      }
    }
  }
  const median = values => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
  return Object.fromEntries(Object.entries(timings).map(([name, values]) => {
    const linearMs = median(values.linear);
    const indexedMs = median(values.indexed);
    return [name, { queries: cases.length, linearMs, indexedMs, speedup: linearMs / indexedMs }];
  }));
}

function main() {
  const { cases, frames } = createCases();
  for (let index = 0; index < cases.length; index += 1) {
    const query = cases[index];
    assert.strictEqual(hurtboxes.intersectsRect(query.hurtbox, query.rect), referenceRect(query.hurtbox, query.rect), `rectangle parity ${index}`);
    assert.strictEqual(hurtboxes.intersectsCircle(query.hurtbox, query.x, query.y, query.radius), referenceCircle(query.hurtbox, query.x, query.y, query.radius), `circle parity ${index}`);
  }
  const hurtbox = cases[0].hurtbox;
  for (const radius of [-1, NaN, Infinity]) {
    assert.strictEqual(hurtboxes.intersectsCircle(hurtbox, 0, 0, radius), referenceCircle(hurtbox, 0, 0, radius));
  }
  // The public draw transform also permits a collapsed or inverted box when no
  // registration is supplied. Broad-phase indexing must retain those results.
  const sheetPath = Object.keys(masks.sheets)[0];
  const animation = { sheet: sheetPath, frameWidth: 160, frameHeight: 160 };
  for (const height of [0, -86, 0.01]) {
    const collapsed = hurtboxes.createEnemyHurtbox(animation, { row: 0, frameIndex: 0 }, { x: 0, y: 0, w: 86, h: height }, 1);
    for (let y = -90; y < 4; y += 2) {
      for (let x = 0; x < 86; x += 2) {
        for (const radius of [0, 1, 20, '1', '20', true, null, [], [20]]) {
          assert.strictEqual(hurtboxes.intersectsCircle(collapsed, x, y, radius), referenceCircle(collapsed, x, y, radius), `collapsed/inverted circle ${height}/${x}/${y}/${radius}`);
        }
      }
    }
  }
  console.log(`Project Starfall hurtbox index passed: ${frames} production frames, both facings, ${cases.length * 2} exact linear-oracle comparisons.`);
  if (process.argv.includes('--benchmark')) console.log(JSON.stringify(runBenchmark(cases), null, 2));
}

if (require.main === module) main();
module.exports = { createCases, referenceRect, referenceCircle, runBenchmark };
