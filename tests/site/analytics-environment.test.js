'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/analytics/environment.js'), 'utf8');
function load(href, storage = new Map()) {
  const window = {
    location: { href },
    sessionStorage: {
      getItem: (key) => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: (key) => storage.delete(key)
    }
  };
  vm.runInNewContext(source, { window, URL });
  return window;
}

for (const url of ['https://www.danielshort.me/', 'https://danielshort.me/portfolio']) {
  const window = load(url);
  assert.strictEqual(window.SiteAnalyticsEnvironment.enabled, true);
  assert.strictEqual(window.dataLayer[0].analytics_debug, undefined, 'Visitors must not receive debug_mode=false');
  assert.strictEqual(window.dataLayer[0].traffic_type, undefined);
}
for (const url of ['http://localhost:4181/', 'http://127.0.0.1:4181/', 'https://website-preview.vercel.app/', 'https://www.danielshort.me.evil.example/', 'file:///test/index.html']) {
  assert.strictEqual(load(url).SiteAnalyticsEnvironment.enabled, false, `${url} must not load production analytics by default`);
}
const storage = new Map();
const qa = load('http://127.0.0.1:4181/?analytics_debug=1', storage);
assert.strictEqual(qa.SiteAnalyticsEnvironment.enabled, true);
assert.strictEqual(qa.dataLayer[0].analytics_debug, true);
assert.strictEqual(qa.dataLayer[0].traffic_type, 'internal');
assert.strictEqual(load('http://127.0.0.1:4181/tools', storage).SiteAnalyticsEnvironment.debug, true, 'Debug persists across full-page navigation in a test tab');
assert.strictEqual(load('http://127.0.0.1:4181/?analytics_debug=0', storage).SiteAnalyticsEnvironment.enabled, false, 'Explicit reset ends debug collection');
assert.strictEqual(load('https://www.danielshort.me/?utm_source=QA').dataLayer[0].traffic_type, 'internal', 'QA campaign traffic must be classified before the Google tag loads');
const blockedStorage = load('https://www.danielshort.me/');
blockedStorage.sessionStorage.getItem = () => { throw new Error('Storage denied'); };
vm.runInNewContext(source, { window: blockedStorage, URL });
assert.strictEqual(blockedStorage.SiteAnalyticsEnvironment.enabled, true, 'Unavailable storage must not block normal production collection');
console.log('Analytics environment tests passed.');
