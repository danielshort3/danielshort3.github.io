'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const E = require('../../js/games/wayfarers-guild/expeditions.js');
const Skills = require('../../js/games/wayfarers-guild/area-skills.js');
const old = require('./fixtures/wayfarers-v4-state.json');
const expected = require('./fixtures/wayfarers-v4-expected.json');
const clone = value => JSON.parse(JSON.stringify(value));
const envelope = (state, version = state.schemaVersion, savedAt = state.lastUpdate) => JSON.stringify({ format: Storage.FORMAT, version, savedAt, state });
const latestVersion = state => Object.assign(clone(state), { schemaVersion: Core.VERSION, collection: Core.createState(state.createdAt).collection, areaSkills: Skills.initial(state, true) });
function fixture(initial = {}, time = old.lastUpdate) {
  const values = new Map(Object.entries(initial));
  const writes = [];
  let failAt = null;
  const storage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => { if (key === failAt) throw Error('Full'); writes.push(key); values.set(key, value); }
  };
  return { values, writes, storage, store: Storage.createStore({ storage, now: () => time }), fail: key => { failAt = key; } };
}
function advance(state, seconds) {
  let left = seconds;
  while (left > 0) { const result = Core.advance(state, left); assert.ok(result.seconds > 0); left = result.pendingSeconds || 0; }
}

test('released v4 migrates without resetting, paying rewards, changing choices or mutating its source', () => {
  const input = clone(old), before = clone(input);
  const state = Core.migrateState(input);
  assert.deepEqual(input, before);
  assert.deepEqual(state, latestVersion(old));
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  assert.deepEqual(Core.getRates(state), expected.rates);
  assert.deepEqual(clone(E.view(state).cards.map(({ id, level, cap, cost }) => ({ id, level, cap, cost }))), expected.cards);
});

for (const seconds of [120, 28800]) {
  test('released economy and reward schedule are unchanged after ' + seconds + ' seconds', () => {
    const state = Core.migrateState(clone(old));
    advance(state, seconds);
    assert.deepEqual(state, latestVersion(expected['after' + seconds]));
    assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  });
}

test('old source bytes remain the backup after first schema5 write and reloading is idempotent', () => {
  const text = envelope(old), f = fixture({ [Storage.SAVE_KEY]: text });
  const loaded = f.store.load({ deferOffline: true });
  assert.equal(loaded.status, 'loaded');
  assert.equal(loaded.state.schemaVersion, Core.VERSION);
  assert.equal(f.writes.length, 0);
  assert.ok(f.store.save(loaded.state).ok);
  assert.equal(f.values.get(Storage.BACKUP_KEY), text);
  assert.equal(JSON.parse(f.values.get(Storage.SAVE_KEY)).version, Core.VERSION);
  assert.deepEqual(f.store.load({ deferOffline: true }).state, loaded.state);
});

test('failed migration writes and malformed/future envelopes cannot sacrifice an existing save', () => {
  const text = envelope(old), f = fixture({ [Storage.SAVE_KEY]: text });
  const loaded = f.store.load({ deferOffline: true });
  f.fail(Storage.BACKUP_KEY);
  assert.equal(f.store.save(loaded.state).ok, false);
  assert.equal(f.values.get(Storage.SAVE_KEY), text);
  assert.equal(f.values.has(Storage.BACKUP_KEY), false);
  f.fail(Storage.SAVE_KEY);
  assert.equal(f.store.save(loaded.state).ok, false);
  assert.equal(f.values.get(Storage.SAVE_KEY), text);
  assert.equal(f.values.get(Storage.BACKUP_KEY), text);
  for (const bad of [envelope(old, 5), envelope(old, 6), envelope({ ...old, expedition: {} }), envelope({ ...old, schemaVersion: 5 }, 4)]) {
    const protectedStore = fixture({ [Storage.SAVE_KEY]: bad });
    const result = protectedStore.store.load({ deferOffline: true });
    assert.equal(result.canSave, false);
    assert.equal(protectedStore.store.save(Core.createState(old.lastUpdate)).ok, false);
    assert.equal(protectedStore.values.get(Storage.SAVE_KEY), bad);
  }
});

test('native bootstrap restores v4 then mirrors only canonical schema5 saved by the game', () => {
  const text = envelope(old), f = fixture();
  const messages = [];
  const window = {
    WayfarersStorage: { ...Storage }, WayfarersNativeCheckpoint: { text },
    WayfarersAndroid: { postMessage: message => messages.push(JSON.parse(message)) }
  };
  const context = { window, localStorage: f.storage, document: { querySelector: () => null } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../mobile/android/wayfarers/web/checkpoint.js'), 'utf8'), context);
  assert.equal(f.values.get(Storage.SAVE_KEY), text);
  const store = window.WayfarersStorage.createStore({ storage: f.storage, now: () => old.lastUpdate });
  const loaded = store.load({ deferOffline: true });
  assert.equal(loaded.state.schemaVersion, Core.VERSION);
  assert.ok(store.save(loaded.state).ok);
  assert.equal(messages.length, 1);
  assert.equal(JSON.parse(messages[0].text).version, Core.VERSION);
  assert.equal(window.WayfarersCheckpoint.confirmed(), false);
  window.WayfarersAndroid.onmessage({ data: JSON.stringify({ type: 'checkpoint', ok: true, requestId: messages[0].requestId }) });
  assert.equal(window.WayfarersCheckpoint.confirmed(), true);
  assert.equal(f.values.get(Storage.BACKUP_KEY), text);
});
