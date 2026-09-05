'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const demo = fs.readFileSync(path.join(__dirname, '../../demos/pizza-tips-demo.html'), 'utf8');
const source = demo.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const metadata = fs.readFileSync(path.join(__dirname, '../../js/demos/pizza-tips-meta.js'), 'utf8');
const flush = () => new Promise((resolve) => setImmediate(resolve));

function prediction(payload, tip = 5.25) {
  return {
    predictions: { tip: { value: tip, interval: { level: payload.confidenceLevel, low: 1, high: 9 } } },
    warnings: [],
    heatmap: payload.grid ? { min: 1, max: 9, points: [{ lat: 33.1, lon: -96.8, tip }] } : null
  };
}

function createHarness() {
  const elements = new Map();
  const requests = [];
  const pendingResponses = [];
  const heatRenders = [];
  const createElement = () => ({
    dataset: {}, style: {}, children: [], value: '', textContent: '', hidden: false, open: false,
    listeners: new Map(),
    addEventListener(type, callback) { this.listeners.set(type, callback); },
    emit(type) { return this.listeners.get(type)?.({ target: this, preventDefault() {} }); },
    append(...children) { this.children.push(...children); },
    appendChild(child) { this.children.push(child); },
    querySelector() { return createElement(); },
    reportValidity() { return true; }
  });
  const getElement = (id) => {
    if (!elements.has(id)) elements.set(id, createElement());
    return elements.get(id);
  };
  const window = {
    DemoAws: {
      resolveEndpoint: () => '/api/demos/pizza-tips/',
      listCandidates: () => ['/api/demos/pizza-tips/'],
      rememberEndpoint() {},
      healthJson: async () => ({ ok: true }),
      warmupJson: async () => ({ ok: true }),
      retryRequest: (operation) => operation(),
      postWithFallback: async (url, paths, payload) => {
        requests.push(payload);
        if (pendingResponses.length) return pendingResponses.shift()(payload);
        return prediction(payload);
      }
    }
  };
  const map = { fitBounds() {}, invalidateSize() {}, panTo() {}, on() {} };
  const layer = () => ({ addTo() { return this; }, setLatLng() {}, on() {} });
  const context = vm.createContext({
    window,
    document: { getElementById: getElement, createElement, querySelectorAll: () => [] },
    console: { error() {} },
    L: {
      map: () => map, tileLayer: layer, marker: layer, geoJSON: layer,
      heatLayer: () => ({ ...layer(), setLatLngs(points) { heatRenders.push(points); } }),
      control: () => ({ addTo() { this.onAdd(); } }),
      DomUtil: { create: createElement }, DomEvent: { disableClickPropagation() {} }
    }
  });
  vm.runInContext(metadata, context);
  vm.runInContext(source, context, { filename: 'pizza-tips-demo.html' });
  return { requests, pendingResponses, heatRenders, getElement, evaluate: (code) => vm.runInContext(code, context) };
}

async function run() {
  assert(demo.indexOf('id="tip-amount"') < demo.indexOf('id="scenario-form"'), 'Summary must precede the scenario controls');
  const harness = createHarness();
  await flush();
  const { getElement, requests, evaluate, pendingResponses, heatRenders } = harness;
  assert.strictEqual(requests.length, 1, 'The computed example makes one prediction request');
  assert(!requests[0].grid, 'The initial prediction must not compute the 576-point grid');
  assert.strictEqual(getElement('tip-amount').textContent, '$5.25');
  assert.strictEqual(getElement('tip-percent').textContent, '15.0%');
  assert(getElement('result-scenario').textContent.startsWith('Computed example'));
  assert(getElement('tip-interval').textContent.includes('80% uncertainty interval'));

  getElement('location-details').open = true;
  getElement('location-details').emit('toggle');
  await flush();
  assert.strictEqual(requests.length, 1, 'Opening the location map alone does not calculate a grid');
  getElement('heatmap-details').open = true;
  getElement('heatmap-details').emit('toggle');
  await flush();
  assert.strictEqual(requests.length, 2);
  assert.strictEqual(requests[1].grid.rows * requests[1].grid.cols, 576);
  getElement('heatmap-details').open = false;
  getElement('heatmap-details').emit('toggle');
  getElement('heatmap-details').open = true;
  getElement('heatmap-details').emit('toggle');
  await flush();
  assert.strictEqual(requests.length, 2, 'Reopening an unchanged comparison uses its cached grid');

  getElement('cost').value = '50';
  getElement('scenario-form').emit('input');
  assert.strictEqual(getElement('estimate-summary').dataset.state, 'stale');
  assert.strictEqual(heatRenders.at(-1).length, 0, 'Changing a scenario clears the old grid immediately');
  assert.strictEqual(requests.length, 2, 'Changing a scenario does not request an estimate until submitted');
  await evaluate('requestEstimate()');
  await flush();
  assert.strictEqual(getElement('tip-percent').textContent, '10.5%');
  assert.strictEqual(requests.filter((payload) => payload.grid).length, 2, 'An updated cost requires a new comparison grid');

  getElement('confidence').value = '.95';
  getElement('scenario-form').emit('change');
  await evaluate('requestEstimate()');
  await flush();
  assert.strictEqual(requests.filter((payload) => payload.grid).length, 2, 'Interval confidence does not alter grid predictions');
  assert(getElement('tip-interval').textContent.includes('95% uncertainty interval'));
  evaluate('setLatLon(META.cityCenters.Plano.latitude, META.cityCenters.Plano.longitude)');
  assert.strictEqual(heatRenders.at(-1).length, 0, 'Location changes also hide the previous grid pending an updated estimate');
  await evaluate('requestEstimate()');
  await flush();
  assert.strictEqual(requests.filter((payload) => payload.grid).length, 2, 'The same area grid is reusable after changing the selected point');

  let finishOldEstimate;
  pendingResponses.push((payload) => new Promise((resolve) => { finishOldEstimate = () => resolve(prediction(payload, 99)); }));
  const oldRequest = evaluate('requestEstimate()');
  await flush();
  getElement('cost').value = '60';
  getElement('scenario-form').emit('input');
  await evaluate('requestEstimate()');
  await flush();
  finishOldEstimate();
  await oldRequest;
  assert.strictEqual(getElement('tip-amount').textContent, '$5.25', 'Late responses cannot overwrite the current estimate');

  getElement('heatmap-details').open = false;
  getElement('heatmap-details').emit('toggle');
  getElement('cost').value = '70';
  getElement('scenario-form').emit('input');
  await evaluate('requestEstimate()');
  let finishOldGrid;
  pendingResponses.push((payload) => new Promise((resolve) => { finishOldGrid = () => resolve(prediction(payload, 88)); }));
  getElement('heatmap-details').open = true;
  getElement('heatmap-details').emit('toggle');
  await flush();
  getElement('cost').value = '80';
  getElement('scenario-form').emit('input');
  finishOldGrid();
  await flush();
  assert.strictEqual(heatRenders.at(-1).length, 0, 'A late grid response cannot restore a stale scenario overlay');

  await evaluate('requestEstimate()');
  await flush();
  getElement('cost').value = '0';
  getElement('scenario-form').emit('input');
  await evaluate('requestEstimate()');
  await flush();
  assert.strictEqual(getElement('tip-percent').textContent, '--');
  assert(getElement('tip-percent-interval').textContent.includes('above $0'), 'Zero-cost scenarios never display an infinite percentage or old interval');

  pendingResponses.push(() => { throw new Error('Network unavailable'); });
  await evaluate('requestEstimate()');
  assert.strictEqual(getElement('estimate-summary').dataset.state, 'error');
  assert.strictEqual(getElement('predict').disabled, false, 'A failed estimate remains retryable');
  await evaluate('requestEstimate()');
  assert.strictEqual(getElement('estimate-summary').dataset.state, 'current');

  getElement('heatmap-details').open = false;
  getElement('heatmap-details').emit('toggle');
  getElement('cost').value = '90';
  getElement('scenario-form').emit('input');
  await evaluate('requestEstimate()');
  pendingResponses.push(() => { throw new Error('Map unavailable'); });
  getElement('heatmap-details').open = true;
  getElement('heatmap-details').emit('toggle');
  await flush();
  assert.strictEqual(getElement('estimate-summary').dataset.state, 'current', 'A map failure must retain the successful estimate');
  assert.strictEqual(getElement('heatmap-retry').hidden, false);
  getElement('heatmap-retry').emit('click');
  await flush();
  assert.strictEqual(getElement('heatmap-retry').hidden, true);
  assert(heatRenders.at(-1).length > 0, 'The map retry restores the overlay');
  getElement('reset').emit('click');
  await flush();
  assert.strictEqual(getElement('cost').value, 35);
  assert.strictEqual(getElement('tip-percent').textContent, '15.0%');
  assert(getElement('result-scenario').textContent.startsWith('Computed example'));
  console.log('Pizza summary, grid caching, stale responses, uncertainty, and retry checks passed.');
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
