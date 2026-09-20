'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('js/common/session-drafts.js', 'utf8');
let now = 10000000;
const storage = {};
Object.defineProperties(storage, {
  getItem: { value: key => storage[key] || null },
  setItem: { value: (key, value) => { storage[key] = value; } },
  removeItem: { value: key => { delete storage[key]; } }
});
const window = { sessionStorage: storage };
vm.runInNewContext(source, { window, TextEncoder, Date: { now: () => now } });
const drafts = window.SiteSessionDrafts;
assert(drafts.write('tools:text-compare', { text: 'before' }));
assert.equal(drafts.read('tools:text-compare').text, 'before');
now += 2 * 60 * 60 * 1000;
assert.equal(drafts.read('tools:text-compare'), null, 'Expire after inactivity');
storage.unrelated = 'preserve';
for (let i = 0; i < 6; i++) { now++; drafts.write(`tools:${i}`, { text: 'a'.repeat(900000) }); }
assert.equal(drafts.read('tools:0'), null, 'Evict oldest first');
assert(drafts.read('tools:5'));
assert.equal(storage.unrelated, 'preserve');
assert.equal(drafts.write('tools:5', { text: 'a'.repeat(1024 * 1024) }), false);
assert.equal(drafts.read('tools:5'), null, 'Oversize must not resurrect an older value');
drafts.removePrefix('tools:');
assert.equal(Object.keys(storage).length, 1, 'Auth isolation leaves unrelated storage');
Object.defineProperty(window, 'sessionStorage', { get() { throw Error('denied'); } });
assert.equal(drafts.read('x'), null);
assert.equal(drafts.write('x', {}), false);
drafts.remove('x');
console.log('Session draft limits, expiry, isolation, and storage denial passed.');
