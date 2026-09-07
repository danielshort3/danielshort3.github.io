'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { createLoader } = require('../../js/demos/dashboard-data');

const ROOT = path.resolve(__dirname, '../..');
const DATA = path.join(ROOT, 'demos', 'data');

async function run() {
  const manifest = JSON.parse(fs.readFileSync(path.join(DATA, 'dashboard-manifest.json'), 'utf8'));
  for (const entry of manifest.files) {
    const bytes = fs.readFileSync(path.join(DATA, entry.path));
    assert.equal(bytes.length, entry.bytes, `${entry.path}: published dataset size changed`);
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), entry.sha256,
      `${entry.path}: refresh the provenance manifest when changing historical data`);
  }

  const requests = [];
  const loader = createLoader(async (url, options) => {
    requests.push(url);
    assert(url.startsWith('/demos/data/'), 'Dashboard reads must stay within bundled site data');
    assert.equal(options.credentials, 'omit', 'Public datasets do not need account credentials');
    const bytes = fs.readFileSync(path.join(ROOT, url));
    return { ok: true, json: async () => JSON.parse(bytes) };
  });

  const target = await loader.loadDashboardData('target-empty-package');
  assert.equal(target.rows.length, 5955, 'Migration must preserve every published empty-package record');
  for (const row of target.rows) {
    assert.match(row.employee, /^Employee-\d+$/);
    assert.match(row.location, /^Location-\d+$/);
    assert.equal(typeof row.value, 'number');
  }
  const retail = await loader.loadDashboardData('retail-loss-sales');
  assert.equal(retail.incidents.stores.length, 2016, 'Keep all stores available for ranking and region filters');
  assert.equal(retail.sales.weekly.length, 238, 'Keep the complete published sales trend');
  assert.equal(retail.emptyPackages.monthly.length, 31);
  for (const store of retail.incidents.stores) assert.match(store.store, /^Store_\d+$/);
  for (const row of retail.emptyPackages.employees) assert.match(row.employee, /^Employee(?:_?ID)?[_-]?\d+$/i);

  const meta = await loader.loadCovidMeta();
  assert.equal(meta.dates.length, 345, 'Every date offered by the public dashboard must remain available');
  assert.equal(meta.states.length, 54, 'Preserve state and territory selection');
  assert.equal(meta.latest, meta.dates.at(-1));
  const dateRows = new Map();
  for (const date of meta.dates) {
    const payload = await loader.loadCovidDate(date);
    assert(payload.states.length > 0, `${date}: empty map data`);
    dateRows.set(date, new Map(payload.states.map((row) => [row.id, row])));
    for (const hotspot of payload.hotspots) assert(dateRows.get(date).has(hotspot.id));
  }
  for (const state of meta.states) {
    const payload = await loader.loadCovidState(state.id);
    for (const point of payload.history) {
      const summary = dateRows.get(point.date)?.get(state.id);
      assert(summary, `${state.id}/${point.date}: trend point must have a matching map summary`);
      assert.equal(point.prob, summary.prob, 'Historical trend and selected-date score must agree');
    }
  }
  assert.equal(requests.length, manifest.files.length, 'Manifest must cover every loadable dataset');
  assert.throws(() => loader.loadDashboardData('../private'), /Unknown dashboard/);
  assert.throws(() => loader.loadCovidDate('../private'), /Invalid date/);
  assert.throws(() => loader.loadCovidState('../private'), /Invalid state/);

  let attempts = 0;
  const retryable = createLoader(async () => {
    attempts += 1;
    return attempts === 1
      ? { ok: false, status: 404 }
      : { ok: true, json: async () => meta };
  });
  await assert.rejects(retryable.loadCovidMeta(), /404/);
  assert.equal(attempts, 1, 'Missing static data should fail immediately without cloud warm-up retries');
  assert.deepEqual(await retryable.loadCovidMeta(), meta, 'Retry must recover without a cached rejected response');
  const invalid = createLoader(async () => ({ ok: true, json: async () => ({}) }));
  await assert.rejects(invalid.loadDashboardData('target-empty-package'), /incomplete or invalid/);

  for (const name of ['target-empty-package', 'retail-loss-sales', 'covid-outbreak']) {
    const html = fs.readFileSync(path.join(ROOT, 'demos', `${name}-demo.html`), 'utf8');
    assert(!/DemoAws|aws-client\.js|\/api\/demos\/|WARM_RETRY|resolveEndpoint/.test(html), `${name}: cloud runtime wiring remains`);
    assert(html.includes('js/demos/dashboard-data.js'));
    for (const match of html.matchAll(/<script[^>]+src="(js\/demos\/vendor\/[^"?]+)"/g)) {
      assert(fs.existsSync(path.join(ROOT, match[1])), `${name}: missing visualization library`);
    }
  }
  console.log('Local dashboards: complete published datasets, anonymous identifiers, map/trend consistency, static loading and recovery passed.');
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
