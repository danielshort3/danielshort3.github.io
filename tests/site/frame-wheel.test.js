'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/navigation/site-frame.js'), 'utf8');
const handlers = source.slice(source.indexOf('  function hasFrameInteractionLayer('), source.indexOf('  function adopt('));

function setup() {
  const body = { nodeType: 1, closest: () => null, classList: { contains: () => false } };
  const viewport = {
    scrollTop: 100, scrollHeight: 1500, clientHeight: 500,
    closest: () => null, contains: () => false,
    scrollTo({ top, behavior }) {
      assert.equal(behavior, 'instant', 'wheel deltas must not queue smooth animations');
      this.scrollTop = top;
    }
  };
  const context = {
    document: { body, documentElement: {}, querySelector: () => null, querySelectorAll: () => [] },
    frame: { isConnected: true }, viewport, current: { fit: 'viewport' }, compactQuery: { matches: false },
    geometry: null, held: null, wipeMotion: null, wipeClosed: false,
    getComputedStyle: () => ({ overflowY: 'auto', lineHeight: '24px' })
  };
  vm.createContext(context);
  vm.runInContext(handlers, context);
  return {
    context,
    wheel(overrides = {}) {
      const event = {
        target: body, deltaY: 120, deltaX: 0, deltaMode: 0, cancelable: true, defaultPrevented: false,
        preventDefault() { this.defaultPrevented = true; }, ...overrides
      };
      context.scrollFromFrameChrome(event);
      return { top: viewport.scrollTop, prevented: event.defaultPrevented };
    }
  };
}

assert.deepEqual(setup().wheel(), { top: 220, prevented: true }, 'pixel wheel deltas retain their distance');
assert.deepEqual(setup().wheel({ deltaY: 3, deltaMode: 1 }), { top: 172, prevented: true }, 'line deltas use the viewport line height');
assert.deepEqual(setup().wheel({ deltaY: 1, deltaMode: 2 }), { top: 600, prevented: true }, 'page deltas use the viewport height');
const normalLine = setup();
normalLine.context.getComputedStyle = () => ({ overflowY: 'auto', lineHeight: 'normal' });
assert.deepEqual(normalLine.wheel({ deltaY: 3, deltaMode: 1 }), { top: 148, prevented: true }, 'normal line height has a finite fallback');

for (const overrides of [
  { ctrlKey: true }, { metaKey: true }, { shiftKey: true }, { altKey: true },
  { deltaX: 200 }, { deltaX: 120 }, { deltaY: 0 }, { deltaY: NaN },
  { cancelable: false }, { defaultPrevented: true }
]) {
  assert.deepEqual(setup().wheel(overrides), { top: 100, prevented: Boolean(overrides.defaultPrevented) },
    `the frame leaves this gesture alone: ${JSON.stringify(overrides)}`);
}

const edges = setup();
assert.deepEqual(edges.wheel({ deltaY: -200 }), { top: 0, prevented: true }, 'upward scrolling clamps at the top');
assert.deepEqual(edges.wheel({ deltaY: -200 }), { top: 0, prevented: false }, 'a gesture at the top remains unclaimed');
assert.deepEqual(edges.wheel({ deltaY: 2000 }), { top: 1000, prevented: true }, 'downward scrolling clamps at the bottom');
assert.deepEqual(edges.wheel(), { top: 1000, prevented: false }, 'a gesture at the bottom remains unclaimed');

for (const field of ['geometry', 'held', 'wipeMotion', 'wipeClosed']) {
  const moving = setup();
  moving.context[field] = true;
  assert.deepEqual(moving.wheel(), { top: 100, prevented: false }, `${field} suppresses forwarding during navigation`);
}

console.log('Frame wheel units, gesture ownership, boundaries, and navigation guards passed.');
