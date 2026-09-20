'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/analytics/web-vitals.js', 'utf8').replace(/^import .*;\r?\n/, '');
function harness({ consent = false, enabled = true, dnt = '' } = {}) {
  const listeners = {}, callbacks = [], sent = [];
  const window = {
    SiteAnalyticsEnvironment: { enabled },
    consentAPI: { get: () => ({ analytics: consent }) },
    addEventListener: (name, fn) => { listeners[name] = fn; },
    sendWebVital: metric => sent.push(metric)
  };
  vm.runInNewContext(source, { window, navigator: { doNotTrack: dnt }, onCLS: fn => callbacks.push(fn), onINP: fn => callbacks.push(fn), onLCP: fn => callbacks.push(fn) });
  return { callbacks, sent, consent: value => listeners['consent-changed']({ detail: { categories: { analytics: value } } }) };
}
const test = harness();
assert.equal(test.callbacks.length, 0);
test.consent(true);
assert.equal(test.callbacks.length, 3);
test.callbacks[0]({ name: 'LCP', value: 123, rating: 'good' });
assert.equal(test.sent.length, 1);
test.consent(false);
test.callbacks.forEach(fn => fn({ name: 'CLS', value: .1, rating: 'good' }));
assert.equal(test.sent.length, 1, 'Withdrawal must suppress callbacks');
test.consent(true);
assert.equal(test.callbacks.length, 3, 'No duplicate document observers');
test.callbacks[0]({ name: 'LCP', value: 123, rating: 'good' });
assert.equal(test.sent.length, 1, 'Do not replay data from the withdrawn period');
assert.equal(harness({ consent: true, enabled: false }).callbacks.length, 0);
assert.equal(harness({ consent: true, dnt: '1' }).callbacks.length, 0);
console.log('Web vital consent, withdrawal, DNT, environment and observer lifetime passed.');
