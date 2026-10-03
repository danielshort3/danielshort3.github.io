'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const Skills = require('../../js/games/wayfarers-guild/area-skills.js');
const old = require('./fixtures/wayfarers-v6-network.json');
const clone = value => JSON.parse(JSON.stringify(value));
const envelope = state => JSON.stringify({ format: Storage.FORMAT, version: state.schemaVersion, savedAt: state.lastUpdate, state });
function fixture(text) {
  const values = new Map(text ? [[Storage.SAVE_KEY, text]] : []), writes = [];
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => { writes.push(key); values.set(key, value); } };
  return { values, writes, store: Storage.createStore({ storage, now: () => old.lastUpdate }) };
}

test('released schema6 is migrated additively without changing any earned state or activating techniques', () => {
  const before = clone(old), state = Core.migrateState(old);
  assert.deepEqual(old, before);
  assert.deepEqual(Core.validateState(state), { valid: true, errors: [] });
  const retained = clone(state);
  delete retained.areaSkills;
  retained.schemaVersion = 6;
  assert.deepEqual(retained, old);
  assert.deepEqual(state.areaSkills.unlocked, []);
  assert.ok(Object.values(state.areaSkills.ranks).every(rank => rank === 0));
  assert.ok(Object.values(state.areaSkills.configs).every(mode => mode === 'off'));
  assert.equal(Skills.throughput(Skills.value(state, 'express-routes')), 1);
  for (const [areaId, area] of Object.entries(state.expedition.areas)) {
    const foundations = require('../../js/games/wayfarers-guild/progression-content.js').AREAS.find(item => item.id === areaId).tracks.slice(0, 3);
    assert.ok(foundations.filter(track => area.learned.includes(track.id)).every(track => Skills.foundationRequirement(state, areaId, track.id).learned));
  }
});

test('schema6 storage retains exact previous bytes as backup and schema7 round-trips idempotently', () => {
  const text = envelope(old), f = fixture(text), loaded = f.store.load({ deferOffline: true });
  assert.equal(loaded.status, 'loaded');
  assert.equal(loaded.state.schemaVersion, 7);
  assert.equal(f.writes.length, 0);
  assert.ok(f.store.save(loaded.state).ok);
  assert.equal(f.values.get(Storage.BACKUP_KEY), text);
  assert.equal(JSON.parse(f.values.get(Storage.SAVE_KEY)).version, 7);
  assert.deepEqual(f.store.load({ deferOffline: true }).state, loaded.state);
  const exported = f.store.export(loaded.state);
  assert.ok(exported.ok);
  assert.deepEqual(f.store.inspectImport(exported.text).state, loaded.state);
});

test('malformed schema6 and schema7 saves remain protected rather than overwritten by a fresh guild', () => {
  const invalidOld = [clone(old), clone(old), clone(old)];
  delete invalidOld[0].resources.coins;
  invalidOld[1].unexpected = true;
  invalidOld[2].areaSkills = Skills.initial(old);
  const invalidNew = [Core.migrateState(old), Core.migrateState(old), Core.migrateState(old)];
  invalidNew[0].areaSkills.ranks['express-routes'] = 11;
  invalidNew[1].areaSkills.configs['express-routes'] = 'unknown';
  delete invalidNew[2].areaSkills.runtime;
  for (const bad of invalidOld.concat(invalidNew)) {
    const text = envelope(bad), f = fixture(text), result = f.store.load({ deferOffline: true });
    assert.equal(result.canSave, false);
    assert.equal(f.store.save(Core.createState(old.lastUpdate)).ok, false);
    assert.equal(f.store.inspectImport(text).ok, false);
    assert.equal(f.store.replaceImport(text).ok, false);
    assert.equal(f.values.get(Storage.SAVE_KEY), text);
    assert.equal(f.writes.length, 0);
  }
});

test('released expedition2 remains inactive for all new skills until its existing reset boundary', () => {
  const legacy = require('./fixtures/wayfarers-v5-retained.json').state;
  const state = Core.migrateState(legacy);
  assert.equal(state.expedition.version, 2);
  assert.equal(Skills.active(state), false);
  assert.deepEqual(Skills.view(state).items, []);
  assert.deepEqual(state.expedition, legacy.expedition);
  assert.ok(Core.validateState(Core.normalizeState(state, state.lastUpdate)).valid);
});
