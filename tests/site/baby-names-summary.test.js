'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../demos/baby-names-demo.html'), 'utf8');
const scripts = [...source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
const inlineScript = scripts.at(-1)[1].replace(/\n    init\(\);\s*$/, '');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../../demos/data/baby-names.json'), 'utf8'));

function createHarness(initialResponse = data) {
  const elements = new Map();
  const requests = [];
  let response = initialResponse;
  let focused = null;
  function element(id, dataset = {}) {
    const listeners = new Map();
    const classes = new Set();
    const attributes = new Map();
    let html = '';
    let children = [];
    return {
      id, dataset, textContent: '', value: '', hidden: false, disabled: false, writes: 0,
      classList: { toggle(name, active) { if (active) classes.add(name); else classes.delete(name); } },
      get innerHTML() { return html; },
      set innerHTML(value) {
        html = value;
        this.writes += 1;
        if (id === 'rating-bars') children = [...value.matchAll(/data-rating="(\d+)"/g)].map((match) => element(`rating-${match[1]}`, { rating: match[1] }));
      },
      addEventListener(type, callback) { listeners.set(type, callback); },
      emit(type, event = {}) { return listeners.get(type)?.({ target: this, ...event }); },
      setAttribute(name, value) { attributes.set(name, value); },
      getAttribute(name) { return attributes.get(name); },
      querySelectorAll() { return children; },
      querySelector(selector) { return children.find((child) => selector.includes(`"${child.dataset.rating}"`)); },
      closest() { return this.dataset.rating ? this : null; },
      focus() { focused = id; },
      scrollIntoView() {}
    };
  }
  function getElement(id) {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  }
  const segments = ['F', 'M', 'all'].map((sex) => element(`sex-${sex}`, { sex }));
  const tabs = ['recommendations', 'ratings'].map((tab) => element(`tab-${tab}`, { tab }));
  const window = { addEventListener() {} };
  window.self = window;
  window.top = window;
  const context = vm.createContext({
    window,
    document: {
      getElementById: getElement,
      querySelectorAll(selector) { return selector === '.segment-btn' ? segments : selector === '.tab-btn' ? tabs : []; }
    },
    fetch: async (...args) => {
      requests.push(args);
      if (response instanceof Error) throw response;
      return { ok: true, json: async () => response };
    },
    requestAnimationFrame(callback) { callback(); },
    console
  });
  vm.runInContext(inlineScript, context, { filename: 'baby-names-demo.html' });
  return {
    getElement, requests, segments, tabs,
    evaluate: (code) => vm.runInContext(code, context),
    respondWith(value) { response = value; },
    focused: () => focused
  };
}

async function run() {
  const harness = createHarness();
  await harness.evaluate('init()');
  assert.strictEqual(harness.evaluate('appState.sex'), 'F', 'Girls remain the initial dataset');
  assert.strictEqual(harness.getElement('stat-rated').textContent, '1,237');
  assert.strictEqual(harness.getElement('stat-average').textContent, '2.5');
  assert.strictEqual(harness.getElement('stat-median').textContent, '2.0');
  assert.strictEqual(harness.getElement('stat-high-rated').textContent, '14');
  assert.strictEqual(harness.getElement('stat-high-share').textContent, '1.1% of rating entries');
  assert.strictEqual(harness.requests[0].length, 1, 'Static data uses normal browser caching');
  assert(harness.getElement('meta-source').textContent.includes('2024'));
  assert(harness.getElement('meta-generated').textContent.includes('2025-12-21'));
  assert.strictEqual((harness.getElement('top-rated').innerHTML.match(/<li>/g) || []).length, 5);
  assert.strictEqual(harness.evaluate('appState.pageCount'), 3, 'All 50 girl recommendations occupy three pages');
  assert(source.indexOf('id="stat-rated"') < source.indexOf('id="explorer-panel"'), 'Summary precedes the name explorer');
  assert(!source.includes('id="rating-detail-list"'), 'No empty selected-rating panel remains');

  const originalChartWrites = harness.getElement('rating-bars').writes;
  const originalFavoritesWrites = harness.getElement('top-rated').writes;
  const ratingEight = harness.getElement('rating-bars').querySelector('[data-rating="8"]');
  harness.getElement('rating-bars').emit('click', { target: ratingEight });
  assert.strictEqual(harness.evaluate('appState.tab'), 'ratings');
  assert.strictEqual(harness.evaluate('appState.selectedRating'), 8);
  assert(harness.evaluate('getFilteredItems().every((row) => row.rating === 8)'));
  assert.strictEqual(harness.getElement('rating-filter').hidden, false);
  assert.strictEqual(ratingEight.getAttribute('aria-pressed'), 'true');
  assert.strictEqual(harness.getElement('rating-bars').writes, originalChartWrites, 'Selecting a rating preserves chart nodes and keyboard focus');
  assert.strictEqual(harness.getElement('top-rated').writes, originalFavoritesWrites, 'Selecting a rating does not rebuild the summary');
  assert.strictEqual(harness.getElement('stat-rated').textContent, '1,237');
  let prevented = false;
  harness.getElement('rating-bars').emit('keydown', { target: ratingEight, key: 'ArrowRight', preventDefault() { prevented = true; } });
  assert(prevented && harness.focused() === 'rating-9', 'Arrow keys move focus to the adjacent bar');
  harness.getElement('rating-bars').emit('keydown', { target: ratingEight, key: 'Home', preventDefault() {} });
  assert.strictEqual(harness.focused(), 'rating-1');
  harness.getElement('rating-filter').emit('click');
  assert.strictEqual(harness.evaluate('appState.selectedRating'), null);
  assert.strictEqual(harness.getElement('rating-filter').hidden, true);
  assert.strictEqual(harness.focused(), 'name-search');

  const visited = [];
  while (true) {
    visited.push(...JSON.parse(harness.evaluate('JSON.stringify(getFilteredItems().slice((appState.page - 1) * PAGE_SIZE, appState.page * PAGE_SIZE).map((row) => row.name))')));
    const displayedCount = (harness.getElement('name-list').innerHTML.match(/class="name-item"/g) || []).length;
    assert(displayedCount > 0 && displayedCount <= 20);
    if (harness.getElement('next-page').disabled) break;
    harness.getElement('next-page').emit('click');
  }
  assert.strictEqual(visited.length, data.ratings.F.length, 'Paging reaches every rated girl name');
  assert.deepStrictEqual(visited.slice().sort(), data.ratings.F.map((row) => row.name).sort(), 'Paging preserves every source entry, including repeated names');
  assert.strictEqual(harness.evaluate('appState.page'), 62);
  harness.getElement('name-search').value = 'no-such-name-123';
  harness.getElement('name-search').emit('input');
  assert.strictEqual(harness.evaluate('appState.page'), 1, 'Search resets pagination');
  assert.strictEqual(harness.getElement('list-count').textContent, 'No matching names.');
  assert.strictEqual(harness.getElement('pagination').hidden, true);
  assert.strictEqual(harness.getElement('stat-rated').textContent, '1,237', 'Search does not redefine summary statistics');
  harness.getElement('name-search').value = 'Alice';
  harness.getElement('name-search').emit('input');
  assert(harness.evaluate('getFilteredItems().every((row) => row.name.toLowerCase().includes("alice"))'));
  harness.getElement('name-search').value = '';
  harness.getElement('name-search').emit('input');

  harness.evaluate('setSex("all")');
  assert.strictEqual(harness.getElement('stat-rated').textContent, '2,262');
  assert.strictEqual(harness.getElement('stat-high-rated').textContent, '26');
  assert.strictEqual(harness.getElement('stat-high-share').textContent, '1.1% of rating entries');
  assert(harness.evaluate('Math.abs(getDerived("all").stats.average - 2.513262599469496) < 0.000001'));
  assert(harness.evaluate('getDerived("F") === getDerived("F")'), 'Derived statistics and sorted lists are cached');
  harness.evaluate('selectRating(10); setTab("recommendations")');
  assert.strictEqual(harness.evaluate('appState.selectedRating'), null, 'Recommendations clear the actual-rating filter');
  assert.strictEqual(harness.evaluate('appState.pageCount'), 5, 'All 100 recommendations remain reachable');
  harness.evaluate('setSex("M")');
  assert.strictEqual(harness.getElement('stat-rated').textContent, '1,025');
  assert.strictEqual(harness.getElement('stat-high-rated').textContent, '12');
  assert.strictEqual(harness.getElement('stat-high-share').textContent, '1.2% of rating entries');
  const median = harness.evaluate('computeStats([{ name: "A", rating: 2 }, { name: "B", rating: 8 }]).median');
  assert.strictEqual(median, 5, 'Even-sized datasets average their middle ratings');

  const empty = createHarness({ ratings: { F: [], M: [] }, recommendations: { F: [], M: [] }, meta: {} });
  await empty.evaluate('init()');
  assert.strictEqual(empty.getElement('stat-rated').textContent, '0');
  assert.strictEqual(empty.getElement('stat-average').textContent, '—');
  assert.strictEqual(empty.getElement('stat-high-share').textContent, 'No ratings available');
  assert.strictEqual(empty.getElement('pagination').hidden, true);

  const unavailable = createHarness(new Error('Offline'));
  await unavailable.evaluate('init()');
  assert.strictEqual(unavailable.getElement('retry-demo').hidden, false);
  assert.strictEqual(unavailable.getElement('name-search').disabled, true);
  assert(unavailable.getElement('name-list').innerHTML.includes('Unable to load'));
  unavailable.respondWith(data);
  await unavailable.getElement('retry-demo').emit('click');
  assert.strictEqual(unavailable.getElement('retry-demo').hidden, true);
  assert.strictEqual(unavailable.getElement('name-search').disabled, false);
  assert.strictEqual(unavailable.getElement('stat-rated').textContent, '1,237');

  process.stdout.write('Baby names: summary scope, statistics, chart selection, keyboard focus, complete pagination, empty data, and retry passed.\n');
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
