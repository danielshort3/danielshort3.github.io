'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../../js/demos/ad-verification-core');
const { createPlayer } = require('../../js/demos/ad-verification-player');
const root = path.resolve(__dirname, '../..');
const wait = () => new Promise((resolve) => setTimeout(resolve, 1));
function harness(extra = {}) {
  let callback = null;
  let time = 0;
  let overlap = 0;
  const commits = [];
  const replacements = [];
  let mixedStages = false;
  const player = createPlayer({ ...extra, requestFrame: (fn) => { callback = fn; return 1; }, cancelFrame: () => { callback = null; }, onChange: (event, state) => {
    const active = state.lanes.filter((lane) => lane.phase === 'recording');
    overlap = Math.max(overlap, active.length);
    if (new Set(active.map((lane) => lane.type)).size > 1) mixedStages = true;
    if (state.writer?.travelerId) {
      const lane = state.lanes.find((item) => item.id === state.writer.travelerId);
      assert.equal(lane.key, state.writer.key);
      assert.equal(lane.phase, 'verifying');
      assert.equal(lane.progress, state.writer.progress);
    }
    if (event.block) {
      const record = event.block.transactions[0].event;
      const lane = state.lanes.find((item) => item.id === record.travelerId);
      if (record.travelerId && lane) assert.deepEqual(lane.records[record.type], { height: state.count, key: record.key });
      assert.equal(state.proof.blocks.length, state.count);
      commits.push({ record, count: state.count, time: state.time });
    }
    for (const change of event.changes || []) if (change.type === 'enter' && change.replaced) replacements.push({ ...change, others: state.lanes.filter((lane) => lane.slot !== change.slot).map((lane) => lane.id), time: state.time });
    if (extra.onChange) extra.onChange(event, state);
  } });
  async function advance(ms) {
    for (let elapsed = 0; elapsed < ms; elapsed += 32) {
      time += 32; const next = callback; callback = null; if (next) next(time); await wait();
    }
  }
  async function until(predicate, max = 160000) {
    for (let elapsed = 0; elapsed < max; elapsed += 128) { if (predicate(player.snapshot(false))) return; await advance(128); }
    assert.fail('Simulation did not reach the expected state: ' + JSON.stringify(player.snapshot(false)));
  }
  return { player, advance, until, commits, replacements, evidence: () => ({ overlap, mixedStages }) };
}
let proof;
test.before(async () => {
  const h = harness(); await h.player.reset(); h.player.setContinuous(false); h.player.setSpeed(4); h.player.play();
  await h.until((s) => !s.running); proof = h.player.snapshot().proof; h.player.destroy();
});
async function modified(change) { const blocks = core.clone(proof.blocks); await change(blocks); return core.verifyChain(blocks, proof.trust); }
test('known SHA-256 vector and canonical finite JSON', async () => {
  assert.equal(await core.hash('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(core.canonical({ a: 1, b: true }), core.canonical({ b: true, a: 1 }));
  assert.throws(() => core.canonical({ value: Infinity })); assert.throws(() => core.canonical(undefined));
});
test('five independent travelers produce one valid campaign without group fields', async () => {
  assert.equal(proof.blocks.length, 17);
  assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
  assert.ok(proof.blocks.every((block) => !Object.hasOwn(block.transactions[0].event, 'group')));
  assert.equal(proof.blocks.filter((block) => block.transactions[0].event.type === 'campaign').length, 1);
});
test('every scenario omits nonexistent visit records', async () => {
  for (const [scenario, counts] of Object.entries({ none: [0, 0, 12], website: [5, 0, 17], destination: [0, 5, 17], both: [5, 5, 22] })) {
    const h = harness(); await h.player.reset(); h.player.setScenario(scenario); h.player.setContinuous(false); h.player.setSpeed(4); h.player.play();
    await h.until((s) => !s.running);
    const result = h.player.snapshot().proof;
    const events = result.blocks.map((block) => block.transactions[0].event);
    assert.equal(events.filter((event) => event.type === 'website').length, counts[0]);
    assert.equal(events.filter((event) => event.type === 'destination').length, counts[1]);
    assert.equal(events.length, counts[2]);
    assert.equal(events.filter((event) => event.type === 'summary').length, 5);
    assert.equal((await core.verifyChain(result.blocks, result.trust)).valid, true);
    h.player.destroy();
  }
});
test('different travelers AND different stages overlap at every speed', async () => {
  for (const speed of [1, 2, 4]) {
    const h = harness(); await h.player.reset(); h.player.setContinuous(false); h.player.setSpeed(speed); h.player.play();
    await h.until((s) => !s.running);
    assert.ok(h.evidence().overlap >= 3);
    assert.equal(h.evidence().mixedStages, true);
    assert.equal(h.commits.length, 17);
    assert.equal(new Set(h.commits.map((item) => item.record.key)).size, 17);
    h.player.destroy();
  }
});
test('each completed traveler is replaced alone while other lanes keep progressing', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play();
  await h.until((s) => s.admitted >= 8);
  h.player.pause();
  assert.ok(h.replacements.length >= 3);
  assert.ok(h.replacements[0].others.some((id) => ['T001', 'T002', 'T003', 'T004', 'T005'].includes(id)));
  for (const replacement of h.replacements) {
    const summary = h.commits.find((item) => item.record.key === replacement.replaced + '/summary');
    assert.ok(summary); assert.ok(replacement.time >= summary.time + 1200);
  }
  const before = h.player.snapshot().proof;
  h.player.play(); await h.until((s) => s.count >= before.blocks.length + 5); h.player.pause();
  const after = h.player.snapshot().proof;
  assert.deepEqual(after.blocks.slice(0, before.blocks.length), before.blocks);
  assert.equal(after.trust.chainId, before.trust.chainId);
  assert.equal((await core.verifyChain(after.blocks, after.trust)).valid, true);
  h.player.destroy();
});
test('scenario changes affect only new arrivals', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play();
  await h.until((s) => s.admitted === 5);
  const paths = h.player.snapshot(false).lanes.map((lane) => [lane.id, lane.path]);
  h.player.setScenario('none');
  assert.deepEqual(h.player.snapshot(false).lanes.map((lane) => [lane.id, lane.path]), paths);
  await h.until((s) => s.admitted >= 6);
  assert.equal(h.player.snapshot(false).lanes.find((lane) => lane.number === 6).path, 'none');
  h.player.destroy();
});
test('pause freezes ALL lane progress, replacement, queue and block writer', async () => {
  const h = harness(); await h.player.reset(); h.player.play();
  await h.until((s) => s.lanes.some((lane) => lane.phase === 'recording') && s.writer?.travelerId);
  h.player.pause(); const before = h.player.snapshot(false); await h.advance(2000);
  assert.deepEqual(h.player.snapshot(false), before);
  h.player.play(); await h.until((s) => s.count > before.count); h.player.destroy();
});
test('delayed signing does not stop other travelers measuring or commit before approval', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({ createLedger: async () => {
    const local = await core.createLedger();
    return { ...local, prepare: async (draft) => { if (draft.key === 'T001/ad') await gate; return local.prepare(draft); } };
  } });
  await h.player.reset(); h.player.play(); await h.until((s) => s.queue.length >= 4);
  assert.equal(h.player.snapshot(false).count, 2);
  assert.equal(h.player.snapshot(false).writer.progress, .8);
  h.player.pause(); release(); await h.advance(300); assert.equal(h.player.snapshot(false).count, 2);
  h.player.play(); await h.until((s) => s.count > 2); h.player.destroy();
});
test('reset discards pending work and prevents stale arrivals and commits', async () => {
  let release; const gate = new Promise((resolve) => { release = resolve; }); let calls = 0;
  const h = harness({ createLedger: async () => {
    const local = await core.createLedger(); calls += 1;
    return calls === 1 ? { ...local, prepare: async (draft) => { await gate; return local.prepare(draft); } } : local;
  } });
  await h.player.reset(); h.player.play(); await h.advance(200); await h.player.reset(); release(); await h.advance(400);
  const s = h.player.snapshot(false); assert.equal(s.count, 0); assert.equal(s.admitted, 0); assert.equal(s.writer, null); assert.equal(s.error, '');
  h.player.destroy();
});
test('capacity reservation lets every admitted traveler finish before the cap', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play();
  await h.until((s) => s.capacityClosed && !s.running, 2400000);
  const s = h.player.snapshot();
  assert.ok(s.count <= core.MAX_BLOCKS); assert.ok(s.count >= core.MAX_BLOCKS - 4);
  assert.equal(s.completed, s.admitted); assert.ok(s.lanes.every((lane) => lane.phase === 'done'));
  assert.equal((await core.verifyChain(s.proof.blocks, s.proof.trust)).valid, true);
  h.player.destroy();
});
test('editing each record invalidates its proof; untouched descendants depend on it', async () => {
  for (let index = 0; index < proof.blocks.length; index += 1) {
    const result = await modified((blocks) => { blocks[index].transactions[0].event.data.note = 'edited'; });
    assert.equal(result.valid, false); assert.equal(result.blocks[index].state, 'changed');
    assert.ok(result.blocks.slice(index + 1).every((row) => row.state === 'dependent'));
  }
});
test('exact original bytes restore successfully without new signatures', async () => {
  const copy = JSON.stringify(proof);
  await modified((blocks) => { blocks[3].transactions[0].event.data.impressions = 12; });
  assert.equal((await core.verifyChain(core.clone(proof.blocks), proof.trust)).valid, true); assert.equal(JSON.stringify(proof), copy);
});
test('rehashing cannot recreate signatures, approvals or original checkpoint', async () => {
  const result = await modified(async (blocks) => {
    blocks[2].transactions[0].event.data.impressions = 9;
    for (let index = 0; index < blocks.length; index += 1) {
      blocks[index].header.merkleRoot = await core.merkleRoot(blocks[index].transactions);
      blocks[index].header.previousHash = index ? blocks[index - 1].hash : core.ZERO;
      blocks[index].hash = await core.headerHash(blocks[index].header);
    }
  });
  assert.equal(result.valid, false); assert.equal(result.blocks[2].checks.merkleRoot, true); assert.equal(result.blocks[2].checks.eventSignature, false); assert.equal(result.blocks[2].checks.approvals, false); assert.equal(result.headValid, false);
});
test('missing, duplicate or forged approvals fail', async () => {
  for (const change of [(blocks) => blocks[0].approvals.pop(), (blocks) => { blocks[0].approvals[1] = core.clone(blocks[0].approvals[0]); }, (blocks) => { blocks[0].approvals[0].signature = '00'.repeat(64); }]) assert.equal((await modified(change)).valid, false);
});
test('truncation, deletion, reordering, replay and malformed records fail closed', async () => {
  for (const change of [(blocks) => blocks.pop(), (blocks) => blocks.splice(3, 1), (blocks) => blocks.reverse(), (blocks) => blocks.push(core.clone(blocks[3])), (blocks) => { blocks[1] = null; }]) assert.equal((await modified(change)).valid, false);
  for (const value of [null, {}, [null]]) assert.equal((await core.verifyChain(value, proof.trust)).valid, false);
  assert.equal((await core.verifyChain(proof.blocks, {})).valid, false);
});
test('trusted registry rejects another session; private keys are never exported', async () => {
  for (const key of Object.values(proof.trust.publicKeys)) assert.equal(key.d, undefined);
  const local = await core.createLedger(); const trust = core.clone(proof.trust); trust.publicKeys.publisher = local.snapshot().trust.publicKeys.publisher;
  assert.equal((await core.verifyChain(proof.blocks, trust)).valid, false); local.dispose();
});
test('candidates cannot append early, twice, after retirement, or with edited ticket bytes', async () => {
  const local = await core.createLedger();
  const draft = { key: 'campaign', type: 'campaign', travelerId: null, observedAtMs: 0, data: { name: 'Original' } };
  const ticket = await local.prepare(draft); assert.equal(local.snapshot().blocks.length, 0);
  ticket.block.transactions[0].event.data.name = 'Forged'; local.commit(ticket);
  assert.equal(local.snapshot().blocks[0].transactions[0].event.data.name, 'Original'); assert.throws(() => local.commit(ticket));
  local.dispose(); await assert.rejects(local.prepare(draft));
});
test('invalid traveler path is not approved', async () => {
  const local = await core.createLedger();
  await assert.rejects(local.prepare({ key: 'T001/destination', type: 'destination', travelerId: 'T001', observedAtMs: 0, data: {} })); local.dispose();
  assert.throws(() => core.createTraveler('__proto__', 1));
});
test('no visitor tracking, group controls, or public publication flags', () => {
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /name="robots" content="noindex, nofollow, noarchive"/); assert.match(html, /<main id="main"/); assert.match(html, /skip-link/);
  assert.doesNotMatch(html, /data-group|Visit Grand Junction|Foursquare|Viant|Cadent/);
  for (const file of ['ad-verification-core.js', 'ad-verification-player.js', 'ad-verification.js']) assert.doesNotMatch(fs.readFileSync(path.join(root, 'js/demos', file), 'utf8'), /\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage|navigator\.geolocation/);
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/adVerification.json'), 'utf8'));
  assert.equal(metadata.published, false); assert.equal(metadata.hidden, true); assert.equal(metadata.noindex, true);
});
test('built files remain noindex and absent from discovery', (t) => {
  const publicRoot = path.join(root, 'public');
  if (!fs.existsSync(publicRoot)) { t.skip('Requires full site build, run in CI.'); return; }
  assert.match(fs.readFileSync(path.join(publicRoot, 'demos/ad-verification.html'), 'utf8'), /name="robots"[^>]*noindex/);
  for (const file of ['js/demos/ad-verification-core.js', 'js/demos/ad-verification-player.js', 'js/demos/ad-verification.js', 'css/components/ad-verification.css', 'img/projects/ad-verification-travelers.webp']) assert.ok(fs.statSync(path.join(publicRoot, file)).size > 0);
  for (const file of ['sitemap.xml', 'dist/search-index.json', 'app-content/v1/catalog.json']) {
    const target = path.join(publicRoot, file); if (fs.existsSync(target)) assert.doesNotMatch(fs.readFileSync(target, 'utf8'), /adVerification|demos\/ad-verification|Cedar Valley/);
  }
});
