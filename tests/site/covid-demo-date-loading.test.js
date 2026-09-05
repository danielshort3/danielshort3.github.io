'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const demo = fs.readFileSync(path.join(__dirname, '../../demos/covid-outbreak-demo.html'), 'utf8');
const inlineScript = demo.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
const dates = ['2021-01-01', '2021-01-02', '2021-01-03'];
const payload = (date) => ({ states: [{ id: 'CO', name: date, inMap: true }], hotspots: [] });

function createHarness() {
  const elements = new Map();
  const responses = new Map();
  const requests = [];
  const renders = [];
  let reloads = 0;
  const getElement = (id) => {
    if (!elements.has(id)) {
      const listeners = new Map();
      elements.set(id, {
        dataset: {},
        textContent: '--',
        value: '',
        hidden: false,
        disabled: false,
        addEventListener(type, callback) { listeners.set(type, callback); },
        emit(type) { return listeners.get(type)?.({ target: this }); }
      });
    }
    return elements.get(id);
  };
  const window = {
    addEventListener() {},
    location: { reload() { reloads += 1; } },
    DemoAws: {
      resolveEndpoint: () => '/api/demos/covid-outbreak/',
      joinUrl: (base, suffix) => base + suffix,
      retryRequest: (operation) => operation(),
      getJson: async (url) => {
        requests.push(url);
        if (url.endsWith('/meta')) return { dates };
        const date = new URL(url, 'https://example.test').searchParams.get('date');
        const response = responses.get(date);
        if (response instanceof Error) throw response;
        if (response) return response;
        return payload(date);
      }
    }
  };
  const context = vm.createContext({
    window,
    document: {
      getElementById: getElement,
      querySelectorAll: () => [],
      documentElement: {},
      body: {}
    },
    parent: { postMessage() {} },
    requestAnimationFrame: (callback) => callback(),
    setInterval() {},
    console,
    recordRender: (date) => renders.push(date)
  });
  vm.runInContext(inlineScript.replace(/\n    init\(\);\s*$/, ''), context, { filename: 'covid-outbreak-demo.html' });
  vm.runInContext(`
    renderMap = async () => {};
    updateMap = () => recordRender(appState.activeDate);
    updateHotspots = () => {};
    updateTerritories = () => {};
    selectState = () => {};
  `, context);
  return {
    getElement,
    responses,
    requests,
    renders,
    evaluate: (code) => vm.runInContext(code, context),
    reloadCount: () => reloads
  };
}

async function flushPromises() {
  await new Promise((resolve) => setImmediate(resolve));
}

async function run() {
  const harness = createHarness();
  await harness.evaluate('init()');
  const slider = harness.getElement('date-slider');
  const retry = harness.getElement('retry-demo');
  const initialDateLabel = harness.getElement('date-value').textContent;
  const initialRenders = harness.renders.length;

  harness.responses.set(dates[0], new Error('Failed to fetch'));
  slider.value = '0';
  slider.emit('input');
  await flushPromises();
  assert.strictEqual(harness.evaluate('appState.activeDate'), dates[2], 'A failed slider request must retain the rendered date');
  assert.strictEqual(harness.getElement('date-value').textContent, initialDateLabel, 'Date labels must still describe the visible map');
  assert.strictEqual(slider.value, '2', 'The slider must return to the last rendered date');
  assert.strictEqual(harness.renders.length, initialRenders, 'A failed request must not redraw with missing or stale data');
  assert.strictEqual(harness.getElement('connection-pill').dataset.state, 'err');
  assert(harness.getElement('connection-meta').textContent.includes('Still showing'), 'The failure state must explain retained data');
  assert.strictEqual(retry.hidden, false, 'The failed date must offer a retry');

  harness.responses.delete(dates[0]);
  await retry.emit('click');
  assert.strictEqual(harness.evaluate('appState.activeDate'), dates[0], 'Retry must load the failed date, not the restored slider date');
  assert.strictEqual(slider.value, '0');
  assert.strictEqual(retry.hidden, true);
  assert.strictEqual(harness.getElement('connection-pill').dataset.state, 'ok');
  assert.strictEqual(harness.reloadCount(), 0, 'A date retry must preserve the current demo session');

  let resolveOlder;
  harness.responses.set(dates[1], new Promise((resolve) => { resolveOlder = resolve; }));
  const older = harness.evaluate('handleDateChange(1)');
  const newer = harness.evaluate('handleDateChange(2)');
  await newer;
  resolveOlder(payload(dates[1]));
  await older;
  assert.strictEqual(harness.evaluate('appState.activeDate'), dates[2], 'An older response must not overwrite the newer selection');
  assert.strictEqual(slider.value, '2');

  const initiallyUnavailable = createHarness();
  initiallyUnavailable.responses.set(dates[2], new Error('Service unavailable'));
  await initiallyUnavailable.evaluate('init()');
  assert.strictEqual(initiallyUnavailable.evaluate('appState.activeDate'), null);
  assert(initiallyUnavailable.getElement('connection-meta').textContent.includes('No date has loaded yet.'));
  initiallyUnavailable.responses.delete(dates[2]);
  await initiallyUnavailable.getElement('retry-demo').emit('click');
  assert.strictEqual(initiallyUnavailable.evaluate('appState.activeDate'), dates[2], 'Retry must also recover the initial date load');

  process.stdout.write('COVID date loading: failure, retry, initial failure, and out-of-order responses passed.\n');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
