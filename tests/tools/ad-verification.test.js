'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../../js/demos/ad-verification-core.js');
const { createPlayer } = require('../../js/demos/ad-verification-player.js');
const root = path.resolve(__dirname, '../..');
const delay = () => new Promise((resolve) => setTimeout(resolve, 2));
let ledger;
let proof;
test.before(async () => {
  ledger = await core.createLedger();
  for (const draft of core.createPlan('mixed', 1, true).events) ledger.commit(await ledger.prepare(draft));
  proof = ledger.snapshot();
});
async function modified(change) {
  const blocks = core.clone(proof.blocks);
  await change(blocks);
  return core.verifyChain(blocks, proof.trust);
}
test('SHA-256 matches a published test vector; JSON ordering is stable', async () => {
  assert.equal(await core.hash('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(core.canonical({ z: 1, a: true }), core.canonical({ a: true, z: 1 }));
  for (const value of [Infinity, undefined, () => {}]) assert.throws(() => core.canonical(value));
});
test('a mixed group contains exactly two shared blocks and twelve interleaved events', async () => {
  assert.equal(proof.blocks.length, 14);
  assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
  const events = proof.blocks.map((block) => block.transactions[0].event);
  assert.deepEqual(events.map((event) => event.type), ['campaign', 'purchase', 'ad', 'ad', 'ad', 'ad', 'website', 'website', 'destination', 'destination', 'summary', 'summary', 'summary', 'summary']);
  assert.equal(new Set(events.map((event) => event.campaignId)).size, 1);
});
test('all five scenarios omit nonexistent visit events rather than append dummy blocks', () => {
  const counts = { mixed: [2, 2, 14], none: [0, 0, 10], website: [4, 0, 14], destination: [0, 4, 14], both: [4, 4, 18] };
  for (const [scenario, [web, visit, count]] of Object.entries(counts)) {
    const { events } = core.createPlan(scenario, 1, true);
    assert.equal(events.length, count);
    assert.equal(events.filter((event) => event.type === 'website').length, web);
    assert.equal(events.filter((event) => event.type === 'destination').length, visit);
    assert.equal(events.filter((event) => event.type === 'summary').length, 4);
  }
  assert.throws(() => core.createPlan('__proto__', 1));
});
test('traveler pointers skip intervening people, while block hashes remain one chain', () => {
  const events = proof.blocks.map((block) => block.transactions[0].event);
  assert.equal(events[6].previousTravelerEvent, 'event-3');
  assert.equal(events[8].travelerId, 'T003');
  assert.equal(events[8].previousTravelerEvent, 'event-5');
  proof.blocks.slice(1).forEach((block, index) => assert.equal(block.header.previousHash, proof.blocks[index].hash));
});
test('new groups append without rewriting old blocks or recreating the campaign', async () => {
  const prefix = JSON.stringify(proof.blocks);
  for (const draft of core.createPlan('none', 2).events) ledger.commit(await ledger.prepare(draft));
  const next = ledger.snapshot();
  assert.equal(next.blocks.length, 22);
  assert.equal(JSON.stringify(next.blocks.slice(0, 14)), prefix);
  assert.equal((await core.verifyChain(next.blocks, next.trust)).valid, true);
});
test('candidates are not committed until the player calls commit', async () => {
  const local = await core.createLedger();
  const ticket = await local.prepare(core.createPlan('mixed', 1, true).events[0]);
  assert.equal(local.snapshot().blocks.length, 0);
  local.commit(ticket);
  assert.equal(local.snapshot().blocks.length, 1);
  assert.throws(() => local.commit(ticket));
  local.dispose();
});
test('a caller cannot replace a verified candidate by editing its public ticket', async () => {
  const local = await core.createLedger();
  const ticket = await local.prepare(core.createPlan('mixed', 1, true).events[0]);
  ticket.block.transactions[0].event.data.budget = 2;
  local.commit(ticket);
  assert.equal(local.snapshot().blocks[0].transactions[0].event.data.budget, 100000);
});
test('editing any committed event fails; later records are dependent, not falsely edited', async () => {
  for (let index = 0; index < proof.blocks.length; index += 1) {
    const result = await modified((blocks) => { blocks[index].transactions[0].event.data.note = 'changed'; });
    assert.equal(result.valid, false);
    assert.equal(result.blocks[index].state, 'changed');
    assert.ok(result.blocks.slice(0, index).every((block) => block.state === 'verified'));
    assert.ok(result.blocks.slice(index + 1).every((block) => block.state === 'dependent'));
  }
});
test('restoring exact original bytes verifies without new signatures', async () => {
  const before = JSON.stringify(proof);
  await modified((blocks) => { blocks[2].transactions[0].event.data.impressions = 12; });
  assert.equal((await core.verifyChain(core.clone(proof.blocks), proof.trust)).valid, true);
  assert.equal(JSON.stringify(proof), before);
});
test('rehashing a rewritten chain cannot recreate approvals or the checkpoint', async () => {
  const result = await modified(async (blocks) => {
    blocks[2].transactions[0].event.data.impressions = 99;
    for (let index = 0; index < blocks.length; index += 1) {
      blocks[index].header.merkleRoot = await core.merkleRoot(blocks[index].transactions);
      blocks[index].header.previousHash = index ? blocks[index - 1].hash : core.ZERO;
      blocks[index].hash = await core.headerHash(blocks[index].header);
    }
  });
  assert.equal(result.valid, false);
  assert.equal(result.blocks[2].checks.merkleRoot, true);
  assert.equal(result.blocks[2].checks.eventSignature, false);
  assert.equal(result.blocks[2].checks.approvals, false);
  assert.equal(result.headValid, false);
});
test('duplicate, missing and forged approvals fail', async () => {
  assert.equal((await modified((blocks) => { blocks[0].approvals[1] = core.clone(blocks[0].approvals[0]); })).valid, false);
  assert.equal((await modified((blocks) => blocks[0].approvals.pop())).valid, false);
  assert.equal((await modified((blocks) => { blocks[0].approvals[0].signature = '00'.repeat(64); })).valid, false);
});
test('deletion, truncation, reordering, replay, and malformed structure fail', async () => {
  for (const change of [(blocks) => blocks.pop(), (blocks) => blocks.splice(2, 1), (blocks) => blocks.reverse(),
    (blocks) => blocks.push(core.clone(blocks[2])), (blocks) => { blocks[0] = null; }]) assert.equal((await modified(change)).valid, false);
  for (const value of [null, {}, [null]]) assert.equal((await core.verifyChain(value, proof.trust)).valid, false);
  assert.equal((await core.verifyChain(proof.blocks, {})).valid, false);
});
test('public proof contains no private key material and wrong trusted keys fail', async () => {
  for (const jwk of Object.values(proof.trust.publicKeys)) assert.equal(jwk.d, undefined);
  assert.ok(!JSON.stringify(proof).includes('privateKey'));
  const local = await core.createLedger();
  const trust = core.clone(proof.trust);
  trust.publicKeys.publisher = local.snapshot().trust.publicKeys.publisher;
  assert.equal((await core.verifyChain(proof.blocks, trust)).valid, false);
});
test('invalid business paths cannot be approved, including visits without an ad', async () => {
  const local = await core.createLedger();
  await assert.rejects(local.prepare({ type: 'destination', travelerId: 'T001', group: 1, data: {} }));
  await assert.rejects(local.prepare({ type: '__proto__' }));
});

// Manual RAF, real Node Web Crypto: exercise timing without wall-clock guesses.
function harness(extra = {}) {
  let time = 0;
  let callback;
  let handle = 0;
  const events = [];
  const player = createPlayer({ requestFrame: (fn) => { callback = fn; return ++handle; }, cancelFrame: () => { callback = null; },
    onChange: (event, snapshot) => { if (event.kind !== 'frame') events.push({ ...event, snapshot }); }, ...extra });
  async function advance(milliseconds) {
    const count = Math.ceil(milliseconds / 16);
    for (let index = 0; index < count; index += 1) {
      time += 16;
      const next = callback; callback = null;
      if (next) next(time);
      await delay();
    }
  }
  return { player, advance, events };
}
test('pause freezes progress and commits; resume preserves the same event', async () => {
  const { player, advance } = harness();
  await player.reset(); player.setContinuous(false); player.play(); await advance(450);
  const before = player.snapshot(); player.pause(); await advance(1000);
  assert.equal(player.snapshot().active.progress, before.active.progress);
  assert.equal(player.snapshot().sequence, before.sequence);
  player.play(); await advance(1800);
  assert.equal(player.snapshot().sequence, 1);
  assert.equal(player.snapshot().proof.blocks[0].transactions[0].event.id, before.active.id);
  player.destroy();
});
test('1x, 2x, and 4x share the exact event order and single commit boundary', async () => {
  for (const speed of [1, 2, 4]) {
    const { player, advance, events } = harness();
    await player.reset(); player.setContinuous(false); player.setSpeed(speed); player.play();
    for (let limit = 0; player.snapshot().running && limit < 60; limit += 1) await advance(1000);
    const state = player.snapshot();
    assert.equal(state.running, false);
    assert.equal(state.sequence, 14);
    const commits = events.filter((event) => event.kind === 'commit');
    assert.deepEqual(commits.map((event) => event.completed.id), Array.from({ length: 14 }, (_, index) => 'event-' + (index + 1)));
    commits.forEach((event, index) => {
      assert.equal(event.snapshot.active, null);
      assert.equal(event.snapshot.proof.blocks.length, index + 1);
      assert.equal(event.block.transactions[0].event.id, event.completed.id);
      assert.deepEqual(event.block.transactions[0].event.data, event.completed.draft.data);
    });
    player.destroy();
  }
});
test('slow signing holds both views before completion and cannot commit while paused', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({ createLedger: async () => {
    const local = await core.createLedger();
    return { ...local, prepare: async (draft) => { await gate; return local.prepare(draft); } };
  } });
  await h.player.reset(); h.player.play(); await h.advance(2600);
  assert.equal(h.player.snapshot().active.progress, .78);
  assert.equal(h.player.snapshot().sequence, 0);
  h.player.pause(); release(); await h.advance(500);
  assert.equal(h.player.snapshot().sequence, 0);
  h.player.play(); await h.advance(600);
  assert.equal(h.player.snapshot().sequence, 1);
  h.player.destroy();
});
test('reset invalidates an in-flight signature request and leaves the new chain empty', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  let calls = 0;
  const h = harness({ createLedger: async () => {
    const local = await core.createLedger(); calls += 1;
    return calls === 1 ? { ...local, prepare: async (draft) => { await gate; return local.prepare(draft); } } : local;
  } });
  await h.player.reset(); h.player.play(); await h.advance(120); await h.player.reset(); release(); await h.advance(300);
  assert.equal(h.player.snapshot().proof.blocks.length, 0);
  assert.equal(h.player.snapshot().error, '');
  assert.equal(h.player.snapshot().active, null);
  h.player.destroy();
});
test('continuous playback appends a new scenario to the same campaign', async () => {
  const { player, advance } = harness();
  await player.reset(); player.setSpeed(4); player.play();
  while (player.snapshot().sequence < 5) await advance(200);
  player.setScenario('none');
  while (player.snapshot().group < 2) await advance(200);
  player.setContinuous(false);
  while (player.snapshot().running) await advance(400);
  const state = player.snapshot();
  assert.equal(state.sequence, 22);
  assert.equal(state.groups[0].scenario, 'mixed');
  assert.equal(state.groups[1].scenario, 'none');
  assert.equal((await core.verifyChain(state.proof.blocks, state.proof.trust)).valid, true);
  player.destroy();
});
test('unlisted flags, metadata, and privacy boundaries remain intact', () => {
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /name="robots" content="noindex, nofollow, noarchive"/);
  assert.match(html, /<main id="main"/);
  assert.match(html, /skip-link/);
  assert.doesNotMatch(html, /Visit Grand Junction|Foursquare|Viant|Cadent/);
  for (const name of ['ad-verification.js', 'ad-verification-core.js', 'ad-verification-player.js']) {
    const source = fs.readFileSync(path.join(root, 'js/demos', name), 'utf8');
    assert.doesNotMatch(source, /\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage|navigator\.geolocation/);
  }
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/adVerification.json'), 'utf8'));
  assert.equal(metadata.published, false); assert.equal(metadata.hidden, true); assert.equal(metadata.noindex, true);
});
test('built assets are present but absent from public discovery indexes', (t) => {
  const publicRoot = path.join(root, 'public');
  if (!fs.existsSync(publicRoot)) { t.skip('Full website build is checked by CI.'); return; }
  for (const file of ['demos/ad-verification.html', 'js/demos/ad-verification-core.js', 'js/demos/ad-verification-player.js', 'js/demos/ad-verification.js', 'css/components/ad-verification.css', 'img/projects/ad-verification-travelers.webp']) assert.ok(fs.statSync(path.join(publicRoot, file)).size > 0);
  assert.match(fs.readFileSync(path.join(publicRoot, 'demos/ad-verification.html'), 'utf8'), /name="robots"[^>]*noindex/);
  for (const file of ['sitemap.xml', 'dist/search-index.json', 'app-content/v1/catalog.json']) {
    const target = path.join(publicRoot, file);
    if (fs.existsSync(target)) assert.doesNotMatch(fs.readFileSync(target, 'utf8'), /adVerification|demos\/ad-verification|Cedar Valley/);
  }
});
