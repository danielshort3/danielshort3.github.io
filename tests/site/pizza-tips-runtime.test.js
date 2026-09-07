'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const createPredictor = require('../../js/demos/pizza-tips-predictor.js');
const { handler } = require('../../aws/pizza-tips-predict/index.js');
const legacyModel = require('../../aws/pizza-tips-predict/model.json');
const assets = path.join(__dirname, '../../js/demos');
const readAsset = (name) => fs.readFileSync(path.join(assets, name), 'utf8');
const plain = (value) => JSON.parse(JSON.stringify(value));
const flush = () => new Promise((resolve) => setImmediate(resolve));
const metadataContext = vm.createContext({ window: {} });
vm.runInContext(readAsset('pizza-tips-meta.js'), metadataContext);
vm.runInContext(readAsset('pizza-tips-model.js'), metadataContext);
const metadata = metadataContext.window.PizzaTipsMeta;
const config = metadataContext.window.PizzaTipsModel;
const browserModel = {
  ...config,
  categories: { ...config.categories, city: { ...config.categories.city, boundaries: metadata.cityBoundaries } }
};
const predictor = createPredictor(browserModel);
const example = {
  latitude: 33.110288, longitude: -96.824164,
  cost: 35, housing: 'Residential', orderHour: 18, confidenceLevel: 0.8
};

function runtimeHarness({ workerEnabled = true } = {}) {
  const workers = [];
  const timers = new Map();
  let timerId = 0;
  class LocalWorker {
    constructor(url) {
      assert.strictEqual(url, '/js/demos/pizza-tips-worker.js');
      this.messages = [];
      workers.push(this);
      const self = { createPizzaTipsPredictor: createPredictor };
      self.postMessage = (data) => setImmediate(() => {
        if (!this.terminated) this.onmessage({ data: structuredClone(data) });
      });
      this.scope = vm.createContext({
        self,
        importScripts(url) { assert.strictEqual(url, 'pizza-tips-predictor.js'); }
      });
      vm.runInContext(readAsset('pizza-tips-worker.js'), this.scope);
    }
    postMessage(data) {
      this.messages.push(data);
      const copied = structuredClone(data);
      setImmediate(() => {
        if (!this.terminated) this.scope.self.onmessage({ data: copied });
      });
    }
    terminate() { this.terminated = true; }
  }
  const window = {
    PizzaTipsMeta: metadata, PizzaTipsModel: config, createPizzaTipsPredictor: createPredictor,
    Worker: workerEnabled ? LocalWorker : undefined,
    setTimeout(callback) { timers.set(++timerId, callback); return timerId; },
    clearTimeout(id) { timers.delete(id); }
  };
  const context = vm.createContext({
    window,
    fetch() { throw new Error('Predictions must never use a network API'); }
  });
  vm.runInContext(readAsset('pizza-tips-runtime.js'), context);
  return { predict: window.PizzaTipsRuntime.predict, workers, timers };
}

async function run() {
  assert(!config.categories.city.boundaries, 'The lightweight config must not duplicate the polygons');
  assert(fs.statSync(path.join(assets, 'pizza-tips-model.js')).size < 10000);
  assert.deepStrictEqual(plain(metadata.cityBoundaries), legacyModel.categories.city.boundaries);
  for (const key of ['inputFeatures', 'targets', 'bounds', 'ranges', 'metrics', 'coefficients']) {
    assert.deepStrictEqual(plain(config[key]), legacyModel[key], `Browser ${key} must match the saved training model`);
  }
  assert.deepStrictEqual(plain(config.categories.housing), legacyModel.categories.housing);
  assert.deepStrictEqual(plain(config.categories.city.values), legacyModel.categories.city.values);
  assert.strictEqual(config.categories.city.baseline, legacyModel.categories.city.baseline);
  assert.strictEqual(metadata.housingBaseline, config.categories.housing.baseline);
  const builder = fs.readFileSync(path.join(__dirname, '../../build/build-pizza-tips-model.js'), 'utf8');
  let generatedModel;
  vm.runInNewContext(builder.slice(builder.indexOf('const browserModel ='), builder.indexOf('console.log(`Model written')), {
    model: legacyModel,
    BROWSER_MODEL_PATH: 'browser-model.js',
    fs: { writeFileSync(filename, content) { generatedModel = content; } }
  });
  assert.strictEqual(generatedModel, readAsset('pizza-tips-model.js'), 'Retraining must emit the same browser config format and parameters');
  const browserScope = vm.createContext({});
  vm.runInContext(readAsset('pizza-tips-predictor.js'), browserScope);
  assert.deepStrictEqual(plain(browserScope.createPizzaTipsPredictor(browserModel).predict(example)), plain(predictor.predict(example)));
  const cases = [
    example,
    { ...example, cost: 0 },
    { ...example, cost: -100 },
    { ...example, cost: 500, orderHour: 26.9 },
    { ...example, cost: 'bad' },
    { ...example, latitude: 0, longitude: 0 },
    { ...example, latitude: null },
    { ...example, housing: 'unknown' },
    { ...example, confidenceLevel: '91%' },
    { ...example, confidenceLevel: 'bad' },
    { lat: example.latitude, lng: example.longitude, orderCost: '35', housingType: ' residential ', orderTime: '18:45', confidence: '85%' },
    {}, null,
    { ...example, grid: { rows: 24, cols: 24 } },
    { ...example, grid: { rows: 1, cols: 60 } },
    { ...example, grid: { rows: 10, cols: 10, bounds: { latMin: 0, latMax: 1, lonMin: 0, lonMax: 1 } } },
    { ...example, grid: { rows: 10, cols: 10, bounds: { latMin: 33.2, latMax: 33.01, lonMin: -96.71, lonMax: -96.93 } } }
  ];
  for (const center of Object.values(metadata.cityCenters)) {
    for (const housing of metadata.housingOptions) {
      for (const confidenceLevel of [0.8, 0.85, 0.9, 0.95]) {
        cases.push({ ...example, latitude: center.latitude, longitude: center.longitude, housing, confidenceLevel });
      }
    }
  }
  // Include points exactly on and just beside real polygon vertices to preserve edge behavior.
  for (const feature of metadata.cityBoundaries.features) {
    const polygon = feature.geometry.type === 'Polygon' ? feature.geometry.coordinates : feature.geometry.coordinates[0];
    const [longitude, latitude] = polygon[0][0];
    for (const delta of [-1e-7, 0, 1e-7]) {
      cases.push({ ...example, latitude: latitude + delta, longitude: longitude + delta });
    }
  }
  for (const payload of cases) {
    const legacy = await handler({ httpMethod: 'POST', path: '/predict', body: JSON.stringify(payload) });
    assert.deepStrictEqual(plain(predictor.predict(payload)), {
      statusCode: legacy.statusCode, data: JSON.parse(legacy.body)
    }, `Browser calculation must match the preserved Lambda: ${JSON.stringify(payload)}`);
  }
  const result = predictor.predict(example).data;
  assert.strictEqual(result.predictions.tip.value, 5.960513454791636, 'Known saved-model estimate remains unchanged');
  assert.strictEqual(result.predictions.tip.interval.low, 3.1327520371573865);
  assert.strictEqual(result.predictions.tip.interval.high, 10.723119877199238);

  const runtime = runtimeHarness();
  assert.deepStrictEqual(plain(await runtime.predict(example)), plain(result));
  assert.strictEqual(runtime.workers.length, 0, 'The initial estimate does not start a grid worker');
  const gridPayload = { ...example, grid: { rows: 24, cols: 24 } };
  const [grid, secondGrid] = await Promise.all([
    runtime.predict(gridPayload), runtime.predict({ ...gridPayload, cost: 0 })
  ]);
  assert.deepStrictEqual(plain(grid), plain(predictor.predict(gridPayload).data));
  assert.deepStrictEqual(plain(secondGrid), plain(predictor.predict({ ...gridPayload, cost: 0 }).data));
  assert.strictEqual(runtime.workers.length, 1, 'Grid requests reuse their initialized worker');
  assert.strictEqual(runtime.workers[0].messages.filter(({ type }) => type === 'init').length, 1);
  assert.strictEqual(runtime.timers.size, 0, 'Completed grid requests release their timers');

  const failedGrid = runtime.predict(gridPayload);
  const rejection = assert.rejects(failedGrid, /could not be calculated/);
  runtime.workers[0].onerror({ preventDefault() {} });
  await rejection;
  assert.strictEqual(runtime.timers.size, 0);
  assert.deepStrictEqual(plain(await runtime.predict(gridPayload)), plain(grid));
  assert.strictEqual(runtime.workers.length, 2, 'Retry creates a fresh worker after a failure');

  const timedOutGrid = runtime.predict(gridPayload);
  const timeoutRejection = assert.rejects(timedOutGrid, /took too long/);
  runtime.timers.values().next().value();
  await timeoutRejection;
  assert.deepStrictEqual(plain(await runtime.predict(example)), plain(result), 'A grid failure leaves ordinary estimates available');
  assert.deepStrictEqual(plain(await runtime.predict(gridPayload)), plain(grid), 'A timed-out grid can be retried');
  const fallback = runtimeHarness({ workerEnabled: false });
  assert.deepStrictEqual(plain(await fallback.predict(gridPayload)), plain(grid), 'Browsers without workers can still calculate a grid');
  await assert.rejects(runtime.predict({ ...example, latitude: 0, longitude: 0 }), /outside supported/);
  await flush();
  console.log(`Pizza local runtime passed ${cases.length} Lambda parity cases, worker isolation, failure recovery, and no-worker fallback.`);
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
