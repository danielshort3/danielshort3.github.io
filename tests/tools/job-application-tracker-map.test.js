'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'js/tools/job-application-tracker.js'), 'utf8');
const mapSource = fs.readFileSync(path.join(root, 'img/ui/us-map.svg'), 'utf8');
const start = source.indexOf('  const loadMap = async () => {');
const end = source.indexOf('\n  const updateMap =', start);
assert(start >= 0 && end > start, 'The map loading function must remain independently testable');

function createHarness(markup, ok = true) {
  const requests = [];
  const parsed = [];
  const mounted = [];
  const errors = [];
  const attributes = new Map([['width', '959'], ['height', '593']]);
  const classes = new Set();
  const svg = {
    classList: { add: (name) => classes.add(name) },
    getAttribute: (name) => attributes.get(name) || null,
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
    querySelector: () => null
  };
  const state = { mapLoaded: false, mapSvg: null };
  const els = {
    mapContainer: {
      dataset: { jobtrackMapSrc: 'img/ui/us-map.svg' },
      innerHTML: 'Loading map',
      appendChild: (node) => mounted.push(node)
    },
    mapPlaceholder: { textContent: '' }
  };
  const loadMap = vm.runInNewContext(`${source.slice(start, end)}\nloadMap;`, {
    state,
    els,
    console: { error: (...args) => errors.push(args) },
    fetch: async (url) => {
      requests.push(url);
      return { ok, text: async () => markup };
    },
    DOMParser: class {
      parseFromString(text, type) {
        // Chromium checks SVG styles while parsing, before they are mounted.
        assert.doesNotMatch(text, /<style\b/i, 'Strict tracker CSP rejects styles at DOMParser time');
        assert.equal(type, 'image/svg+xml');
        parsed.push(text);
        return { querySelector: (selector) => selector === 'svg' ? svg : null };
      }
    }
  });
  return { loadMap, state, els, svg, attributes, classes, requests, parsed, mounted, errors };
}

test('current map asset uses SVG presentation attributes and preserves state geometry', () => {
  assert.doesNotMatch(mapSource, /<style\b|\bstyle=/i, 'The self-hosted map must not require inline stylesheet execution');
  assert.match(mapSource, /<g class="state" fill="#D0D0D0">/);
  assert.match(mapSource, /<g class="borders" fill="none" stroke="#FFFFFF" stroke-width="1">/);
  assert.match(mapSource, /<path class="separator1" fill="none" stroke="#B0B0B0" stroke-width="2"/);
  const states = new Set([...mapSource.matchAll(/\bclass="([a-z]{2})"/g)].map((match) => match[1]));
  assert(states.size >= 50, 'The map must retain geometry for every state');
  for (const state of ['ca', 'co', 'ny', 'tx']) assert(states.has(state));
  const css = fs.readFileSync(path.join(root, 'css/components/job-application-tracker.css'), 'utf8');
  assert.match(css, /\.jobtrack-map-svg \.state path,/);
  assert.match(css, /fill:var\(--jobtrack-heat-0\)/, 'The external tracker stylesheet must own the heatmap colors');
  assert.match(css, /\.jobtrack-map-svg \.borders,/);
});

test('map parsing removes every legacy stylesheet before strict CSP can reject it', async () => {
  const legacyStyles = '<style type="text/css">.state { fill: #D0D0D0; }</style><STYLE>.borders { stroke: #fff; }</STYLE >';
  for (const markup of [mapSource, mapSource.replace('</svg>', `${legacyStyles}</svg>`)]) {
    const harness = createHarness(markup);
    assert.equal(await harness.loadMap(), harness.svg, 'A current or cached legacy map must load');
    assert.equal(harness.errors.length, 0);
    assert.equal(harness.parsed.length, 1);
    assert.equal(harness.parsed[0], mapSource, 'Removing cached styles must preserve all map geometry and titles');
    assert.equal(harness.mounted[0], harness.svg);
    assert(harness.classes.has('jobtrack-map-svg'));
    assert.equal(harness.attributes.get('viewBox'), '0 0 959 593');
    assert.equal(harness.attributes.get('preserveAspectRatio'), 'xMidYMid meet');
    assert(!harness.attributes.has('width') && !harness.attributes.has('height'));
    assert.equal(harness.state.mapLoaded, true);
    assert.equal(await harness.loadMap(), harness.svg);
    assert.equal(harness.requests.length, 1, 'Subsequent chart updates must reuse the loaded map');
  }
});

test('map fetch failure remains recoverable without mounting incomplete content', async () => {
  const harness = createHarness('', false);
  assert.equal(await harness.loadMap(), null);
  assert.equal(harness.state.mapLoaded, false);
  assert.equal(harness.mounted.length, 0);
  assert.equal(harness.els.mapPlaceholder.textContent, 'Unable to load map.');
});
