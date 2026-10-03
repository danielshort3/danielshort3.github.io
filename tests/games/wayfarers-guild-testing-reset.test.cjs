'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');

function memory() {
  const data = new Map();
  let writes = 0, failAt = 0;
  return { data, getItem: key => data.get(key) ?? null,
    setItem(key, value) { writes++; if (writes === failAt) throw new Error('Full disk'); data.set(key, String(value)); },
    removeItem: key => data.delete(key), failAfter(n) { failAt = writes + n; }, allow() { failAt = 0; } };
}
function setup() {
  const storage = memory(); let time = 1000;
  const make = () => Storage.createStore({ storage, now: () => time, crypto });
  const store = make(), state = store.load().state;
  Core.advance(state, 500);
  assert.equal(store.save(state).ok, true);
  assert.equal(store.save(state).ok, true);
  time = 2000000;
  storage.setItem('unrelated-site-data', 'keep');
  storage.setItem('wayfarers-guild-quiet', 'true');
  return { storage, store, make, state };
}

test('complete reset replaces main and recovery with the same empty opening, retaining unrelated data', () => {
  const { storage, store, make } = setup();
  const result = store.resetForTesting();
  assert.equal(result.ok, true);
  assert.equal(result.state.createdAt, 2000000);
  assert.deepEqual(Core.validateState(result.state), { valid: true, errors: [] });
  assert.equal(result.state.lifetime.refits, 0);
  assert.deepEqual(result.state.collection.cards, {});
  assert.equal(storage.getItem(Storage.SAVE_KEY), storage.getItem(Storage.BACKUP_KEY));
  assert.equal(JSON.parse(storage.getItem(Storage.RESET_KEY)).text, null);
  assert.equal(storage.getItem('unrelated-site-data'), 'keep');
  assert.equal(storage.getItem('wayfarers-guild-quiet'), 'true');
  assert.equal(make().load().state.createdAt, result.state.createdAt);
  storage.removeItem(Storage.SAVE_KEY);
  assert.equal(make().load().state.createdAt, result.state.createdAt, 'backup cannot restore erased guild');
});

for (const failAt of [1, 2, 3, 4, 5, 6]) test('reset write failure ' + failAt + ' recovers without an old-guild resurrection or reroll', () => {
  const { storage, store, make } = setup();
  const before = storage.getItem(Storage.SAVE_KEY);
  storage.failAfter(failAt);
  const result = store.resetForTesting();
  assert.equal(result.ok, false);
  if (failAt === 1) {
    assert.equal(storage.getItem(Storage.SAVE_KEY), before);
    assert.equal(storage.getItem(Storage.RESET_KEY), null);
    return;
  }
  const journal = JSON.parse(storage.getItem(Storage.RESET_KEY));
  storage.allow();
  const reopened = make();
  const loaded = reopened.load();
  assert.equal(loaded.status, 'reset-pending');
  assert.equal(loaded.state.createdAt, JSON.parse(journal.text).state.createdAt);
  assert.equal(reopened.save(loaded.state).ok, false, 'cannot play through an incomplete reset');
  const retry = reopened.resetForTesting();
  assert.equal(retry.ok, true);
  assert.equal(JSON.parse(storage.getItem(Storage.RESET_KEY)).id, journal.id);
  assert.deepEqual(retry.state, JSON.parse(journal.text).state);
  assert.equal(storage.getItem(Storage.BACKUP_KEY), journal.text);
});

test('stale tabs cannot overwrite, import or reset the new generation', () => {
  const { storage, store, make, state } = setup();
  const stale = make(); stale.load();
  const oldExport = stale.export(state).text;
  assert.equal(store.resetForTesting().ok, true);
  const snapshot = Array.from(storage.data);
  assert.equal(stale.save(state).status, 'conflict');
  assert.equal(stale.replaceImport(oldExport).status, 'conflict');
  assert.equal(stale.resetForTesting().status, 'conflict');
  assert.deepEqual(Array.from(storage.data), snapshot);
});

for (const mirrorKey of [Storage.SAVE_KEY, Storage.BACKUP_KEY]) test('a failed compatibility mirror cannot roll back a committed tutorial reward: ' + mirrorKey, () => {
  const { storage, store, make } = setup();
  const reset = store.resetForTesting();
  assert(reset.ok);
  const state = reset.state;
  assert(Core.act(state, { type: 'onboarding-visit', id: 'greenway' }).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.practiceAction).ok);
  assert(store.save(state).ok);
  const write = storage.setItem;
  storage.setItem = (key, value) => {
    if (key === mirrorKey) throw new Error('Compatibility mirror unavailable');
    write(key, value);
  };
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  assert.equal(store.save(state).ok, true, 'The canonical generation holds the committed completion');
  assert.equal(store.save(state).ok, true, 'A subsequent save does not conflict with its own write');
  const reopened = make().load().state;
  assert.equal(reopened.onboarding.progress.greenway, 3);
  assert.equal(Core.Numbers.toNumber(reopened.resources.coins), 12);
  assert.equal(Core.act(reopened, { type: 'onboarding-next', id: 'greenway', stepId: 'next-step' }).ok, false);
  assert.equal(Core.Numbers.toNumber(reopened.resources.coins), 12);
});

test('a failed canonical generation write remains retryable without claiming a tutorial reward', () => {
  const { storage, store, make } = setup();
  const reset = store.resetForTesting();
  const state = reset.state;
  assert(reset.ok);
  assert(Core.act(state, { type: 'onboarding-visit', id: 'greenway' }).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.practiceAction).ok);
  assert(store.save(state).ok);
  const before = JSON.parse(JSON.stringify(state));
  const write = storage.setItem;
  const canonical = Storage.SAVE_KEY + '-generation-' + store.pendingReset().id;
  storage.setItem = (key, value) => {
    if (key === canonical) throw new Error('Canonical save unavailable');
    write(key, value);
  };
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  assert.equal(store.save(state).ok, false);
  assert.equal(make().load().state.onboarding.practice.progress.greenway, 2);
  storage.setItem = write;
  assert(store.save(before).ok, 'The app can save its rolled-back step before another explicit attempt');
  assert(Core.act(before, Core.getView(before).onboarding.active.inspectAction).ok);
  assert(store.save(before).ok);
  assert.equal(Core.Numbers.toNumber(make().load().state.resources.coins), 12);
});

test('a post-write fence read failure keeps the committed tutorial result and retries without conflict', () => {
  const { storage, store, make } = setup();
  const state = store.resetForTesting().state;
  assert(Core.act(state, { type: 'onboarding-visit', id: 'greenway' }).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  assert(Core.act(state, Core.getView(state).onboarding.active.practiceAction).ok);
  assert(store.save(state).ok);
  const read = storage.getItem;
  let markerReads = 0;
  storage.getItem = key => {
    if (key === Storage.RESET_KEY && ++markerReads === 2) throw new Error('Transient final fence read');
    return read(key);
  };
  assert(Core.act(state, Core.getView(state).onboarding.active.inspectAction).ok);
  const saved = store.save(state);
  assert.equal(saved.ok, false);
  assert.equal(saved.committed, true, 'Caller must retain this exact pending result, not roll it back');
  assert.equal(saved.status, 'unavailable');
  assert(store.save(state).ok);
  assert.equal(make().load().state.onboarding.progress.greenway, 3);
  assert.equal(Core.Numbers.toNumber(make().load().state.resources.coins), 12);
});

test('a reset during the final save fence still blocks the old generation despite its committed write', () => {
  const { storage, store, make } = setup();
  const state = store.resetForTesting().state;
  assert(store.save(state).ok);
  const oldGeneration = store.pendingReset().id;
  const write = storage.setItem;
  let replacement;
  storage.setItem = (key, value) => {
    write(key, value);
    if (!replacement && key === Storage.SAVE_KEY + '-generation-' + oldGeneration) {
      replacement = { pending: true };
      const other = make(); other.load(); replacement = other.resetForTesting();
      assert(replacement.ok);
    }
  };
  Core.advance(state, 5);
  const saved = store.save(state);
  assert.equal(saved.ok, false);
  assert.equal(saved.status, 'conflict');
  assert.equal(saved.committed, true);
  assert.equal(store.save(state).status, 'conflict');
  assert.equal(make().load().state.createdAt, replacement.state.createdAt);
  assert.notEqual(replacement.state.createdAt, state.createdAt);
});

test('two resets at an identical clock time have distinct generations and fresh identities', () => {
  const { store } = setup();
  const first = store.resetForTesting();
  const firstId = store.pendingReset().id;
  const second = store.resetForTesting();
  assert.equal(second.ok, true);
  assert.notEqual(store.pendingReset().id, firstId);
  assert.notEqual(first.state.createdAt, second.state.createdAt);
});

test('corrupt reset journal fails closed rather than recovering an erased backup', () => {
  const { storage, make } = setup();
  storage.setItem(Storage.RESET_KEY, '{broken');
  const next = make();
  const before = Array.from(storage.data);
  assert.equal(next.load().canSave, false);
  assert.equal(next.save(Core.createState(999)).ok, false);
  assert.equal(next.resetForTesting().ok, false);
  assert.deepEqual(Array.from(storage.data), before);
});

test('transient reset-marker read failure cannot authorize a temporary empty guild to overwrite progress', () => {
  const { storage, make } = setup(); const old = storage.getItem(Storage.SAVE_KEY);
  const get = storage.getItem; let reject = true;
  storage.getItem = key => { if (key === Storage.RESET_KEY && reject) { reject = false; throw new Error('Temporarily unavailable'); } return get(key); };
  const next = make(); const loaded = next.load();
  assert.equal(loaded.canSave, false);
  assert.equal(next.save(loaded.state).ok, false);
  assert.equal(storage.getItem(Storage.SAVE_KEY), old);
});

test('an old save already between read and write cannot erase confirmed purchases made after reset', () => {
  const { storage, store, make, state } = setup();
  const stale = make(); stale.load();
  const get = storage.getItem; let interleave = true, freshIdentity;
  storage.getItem = key => {
    const previous = get(key);
    if (key === Storage.SAVE_KEY && interleave) {
      interleave = false;
      const reset = store.resetForTesting(); assert.equal(reset.ok, true);
      freshIdentity = reset.state.createdAt;
      Core.advance(reset.state, 10);
      const purchase = Core.getView(reset.state).expedition.cards.find(card => card.visible !== false);
      assert.equal(Core.act(reset.state, purchase.action).ok, true);
      assert.equal(store.save(reset.state).ok, true);
    }
    return previous;
  };
  assert.equal(stale.save(state).status, 'conflict');
  const recovered = make().load();
  assert.equal(recovered.state.createdAt, freshIdentity);
  assert.equal(recovered.state.expedition.areas.greenway.ranks.boots, 1);
});

function checkpointHarness(storage, native, host) {
  const api = Object.assign({}, Storage);
  const bridge = { postMessage(text) { host(JSON.parse(text), bridge); } };
  const context = { WayfarersStorage: api, WayfarersNativeCheckpoint: native, WayfarersAndroid: bridge,
    localStorage: storage, document: { querySelector: () => null }, console, setTimeout, clearTimeout };
  context.window = context;
  vm.runInNewContext(fs.readFileSync(require.resolve('../../mobile/android/wayfarers/web/checkpoint.js'), 'utf8'), context);
  const store = api.createStore({ storage, now: () => 2000000, crypto });
  return { context, store, bridge };
}

test('native reset waits for acknowledgment and fences snapshots while pending', async () => {
  const { storage } = setup(); let resetMessage, response;
  const { context, store } = checkpointHarness(storage, null, (message, bridge) => {
    if (message.type === 'reset-guild') { resetMessage = message; response = bridge; }
  });
  store.load();
  const pending = store.resetForTesting();
  assert.equal(context.WayfarersCheckpoint.snapshot(), '');
  assert.equal(store.save(Core.createState(2000000)).ok, false);
  assert.equal(resetMessage.previousGeneration, '');
  response.onmessage({ data: JSON.stringify({ type: 'reset-guild', requestId: resetMessage.requestId, ok: true }) });
  assert.equal((await pending).ok, true);
  assert.equal(context.WayfarersCheckpoint.generation(), resetMessage.generation);
  assert.equal(JSON.parse(storage.getItem(Storage.RESET_KEY)).text, null);
});

test('native failure keeps one pending seed and Retry sends the identical reset transaction', async () => {
  const { storage } = setup(); const requests = [];
  let accept = false;
  const { store } = checkpointHarness(storage, null, (message, bridge) => {
    if (message.type === 'reset-guild') {
      requests.push(message);
      queueMicrotask(() => bridge.onmessage({ data: JSON.stringify({ type: 'reset-guild', requestId: message.requestId, ok: accept }) }));
    }
  });
  store.load();
  assert.equal((await store.resetForTesting()).ok, false);
  accept = true;
  assert.equal((await store.resetForTesting()).ok, true);
  assert.equal(requests[0].text, requests[1].text);
  assert.equal(requests[0].generation, requests[1].generation);
});

test('bootstrap with a pending reset never copies the old native guild over either browser save', () => {
  const { storage, store } = setup();
  const old = storage.getItem(Storage.SAVE_KEY);
  assert.equal(store.resetForTesting({ deferCommit: true }).ok, true);
  const fresh = storage.getItem(Storage.SAVE_KEY);
  const { context, store: next } = checkpointHarness(storage, { text: old, generation: '' }, () => {});
  assert.equal(storage.getItem(Storage.SAVE_KEY), fresh);
  assert.equal(storage.getItem(Storage.BACKUP_KEY), fresh);
  assert.equal(next.load().status, 'reset-pending');
  assert.equal(context.WayfarersCheckpoint.snapshot(), '');
});

test('a lost native reset acknowledgment recovers later new-guild progress instead of the initial reset snapshot', () => {
  const { storage, store } = setup();
  store.resetForTesting({ deferCommit: true });
  const journal = store.pendingReset();
  const progressed = JSON.parse(journal.text);
  progressed.savedAt += 100;
  Core.advance(progressed.state, .1);
  const newer = JSON.stringify(progressed);
  const { store: next } = checkpointHarness(storage, { text: newer, generation: journal.id, previousGeneration: '' }, () => {});
  assert.equal(next.load().state.lastUpdate, progressed.state.lastUpdate);
  assert.equal(storage.getItem(Storage.SAVE_KEY), newer);
  assert.equal(storage.getItem(Storage.BACKUP_KEY), newer);
});

test('native bootstrap compares the canonical new generation rather than a stale compatibility mirror', () => {
  const { storage, store } = setup();
  const old = storage.getItem(Storage.SAVE_KEY);
  const reset = store.resetForTesting();
  const nativeText = storage.getItem(Storage.SAVE_KEY);
  Core.advance(reset.state, 10);
  assert.equal(Core.act(reset.state, Core.getView(reset.state).expedition.cards.find(card => card.visible !== false).action).ok, true);
  assert.equal(store.save(reset.state).ok, true);
  storage.setItem(Storage.SAVE_KEY, old);
  const { store: next } = checkpointHarness(storage, { text: nativeText, generation: store.pendingReset().id, previousGeneration: '' }, () => {});
  assert.equal(next.load().state.expedition.areas.greenway.ranks.boots, 1);
});
