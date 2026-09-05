'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const readDemo = (name) => fs.readFileSync(path.join(__dirname, '../../demos', `${name}-demo.html`), 'utf8');

function createHarness(name, getJson = async () => ({})) {
  const elements = new Map();
  const getElement = (id) => {
    if (!elements.has(id)) {
      const listeners = new Map();
      elements.set(id, {
        dataset: {}, style: {}, textContent: '', innerHTML: '', value: '', disabled: false,
        classList: { toggle() {} },
        setAttribute() {},
        addEventListener(type, callback) { listeners.set(type, callback); },
        emit(type) { return listeners.get(type)?.({ target: this }); }
      });
    }
    return elements.get(id);
  };
  const metricButtons = ['value', 'count'].map((metric) => {
    const button = getElement(`metric-${metric}`);
    button.dataset.metric = metric;
    return button;
  });
  const context = vm.createContext({
    window: {
      addEventListener() {},
      DemoAws: {
        resolveEndpoint: () => '/api/demo/', joinUrl: (base, suffix) => base + suffix,
        retryRequest: (operation) => operation(), getJson
      }
    },
    document: {
      getElementById: getElement,
      querySelectorAll: (selector) => selector === '[data-metric]' ? metricButtons : [],
      documentElement: {}, body: {}
    },
    parent: { postMessage() {} },
    getComputedStyle: () => ({ getPropertyValue: () => '#005fed' }),
    requestAnimationFrame: (callback) => callback(),
    setInterval() {}, console
  });
  const script = readDemo(name).match(/<script type="module">([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script.replace(/\n    init\(\);\s*$/, ''), context);
  return { getElement, evaluate: (code) => vm.runInContext(code, context) };
}

async function run() {
  for (const name of ['retail-loss-sales', 'target-empty-package', 'covid-outbreak']) {
    const source = readDemo(name);
    const ids = [...source.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
    assert.strictEqual(new Set(ids).size, ids.length, `${name} should retain unique markup IDs`);
    for (const match of source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
      if (match[1].trim()) new vm.Script(match[1], { filename: `${name}-demo.html` });
    }
  }

  const retail = readDemo('retail-loss-sales');
  assert(retail.indexOf('<section class="panel kpi-panel">') < retail.indexOf('<section class="panel sales-panel">'),
    'Retail summary must precede its first chart in reading and mobile order');
  const targetMarkup = readDemo('target-empty-package');
  const summary = targetMarkup.slice(targetMarkup.indexOf('<section class="panel exec-panel"'), targetMarkup.indexOf('<section class="panel filters-panel"'));
  assert.strictEqual((summary.match(/data-kpi=/g) || []).length, 4, 'Empty-package summary should expose four existing metrics');
  assert(summary.includes('Recorded retail value'), 'Record values must not be presented as cash recovered');

  const target = createHarness('target-empty-package');
  target.evaluate(`
    state.rows = [
      { value: 100, location: 'A', department: '1', employee: 'X', date: new Date('2021-01-10T12:00:00') },
      { value: 200, location: 'A', department: '1', employee: 'X', date: new Date('2021-04-10T12:00:00') },
      { value: 300, location: 'B', department: '2', employee: 'Y', date: new Date('2021-04-11T12:00:00') }
    ];
    dataReady = true;
    renderAll();
  `);
  assert.strictEqual(target.getElement('kpi-total').textContent, '$600');
  assert.strictEqual(target.getElement('kpi-avg').textContent, '$200.00');
  assert.strictEqual(target.getElement('kpi-qoq').textContent, '+400%');
  target.getElement('filter-location').value = 'A';
  target.getElement('filter-location').emit('change');
  assert.strictEqual(target.getElement('kpi-total').textContent, '$300', 'Filters must invalidate cached aggregates');
  assert.strictEqual(target.getElement('kpi-avg').textContent, '$150.00');
  assert.strictEqual(target.getElement('kpi-incidents').textContent, '2');
  assert(!target.getElement('breakdown-list').innerHTML.includes('>B<'), 'Breakdown must follow the same filtered records');
  target.getElement('metric-count').emit('click');
  assert.strictEqual(target.getElement('kpi-total').textContent, '$300', 'Chart metric changes must preserve the summary population');
  assert(target.getElement('trend-meta').textContent.includes('incidents'), 'Count toggle must update the visible chart metric');
  target.getElement('filter-location').value = 'Missing';
  target.getElement('filter-location').emit('change');
  assert.strictEqual(target.getElement('kpi-incidents').textContent, '0');
  assert.strictEqual(target.getElement('kpi-avg').textContent, '--', 'An empty selection must not invent an average');

  const pending = new Map();
  const requests = [];
  const covid = createHarness('covid-outbreak', (url) => {
    requests.push(url);
    return new Promise((resolve) => pending.set(url, resolve));
  });
  covid.evaluate(`
    updateMap = () => {};
    updateDrivers = () => {};
    const renderedHistories = [];
    updateTrend = (history) => renderedHistories.push(history.map((point) => point.date));
    appState.endpoint = '/api/demo/';
    appState.activeDate = '2021-01-02';
    appState.states = new Map([
      ['CO', { id: 'CO', name: 'Colorado', prob: .1, icuUtilization: .6 }],
      ['NY', { id: 'NY', name: 'New York', prob: .2, icuUtilization: .7 }]
    ]);
    setServerReady(true);
    updateStateOptions(Array.from(appState.states.values()));
  `);
  const selector = covid.getElement('state-select');
  assert(!selector.disabled && selector.innerHTML.includes('Colorado'), 'Available states must be selectable without using the map');
  selector.value = 'CO';
  selector.emit('change');
  selector.value = 'NY';
  selector.emit('change');
  pending.get('/api/demo/state/NY')({ history: [{ date: '2021-01-01' }, { date: '2021-01-03' }] });
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(covid.getElement('state-title').textContent, 'New York');
  assert.strictEqual(covid.evaluate('JSON.stringify(renderedHistories.at(-1))'), '["2021-01-01"]',
    'Historical trend must exclude dates later than the selected summary date');
  pending.get('/api/demo/state/CO')({ history: [{ date: '2021-01-02' }] });
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(covid.evaluate('JSON.stringify(renderedHistories.at(-1))'), '["2021-01-01"]',
    'A late previous-state response must not replace the current trend');
  selector.emit('change');
  await new Promise((resolve) => setImmediate(resolve));
  assert.strictEqual(requests.length, 2, 'Revisiting the selected state should reuse its downloaded history');
  console.log('Analytics summaries: markup, filters, chart toggles, state selection, and historical response ordering passed.');
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
