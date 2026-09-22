'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../../js/demos/ad-verification-core');
const { createPlayer, HOLD_MS } = require('../../js/demos/ad-verification-player');
const root = path.resolve(__dirname, '../..');
const wait = () => new Promise((resolve) => setTimeout(resolve, 1));
function harness(extra = {}) {
  let callback = null;
  let time = 0;
  const evidence = { maxParallel: 0, mixedStages: false, commits: [], replacements: [] };
  const player = createPlayer({ ...extra, requestFrame: (fn) => { callback = fn; return 1; }, cancelFrame: () => { callback = null; }, onChange: (event, state) => {
    const active = state.lanes.filter((lane) => lane.phase === 'recording');
    evidence.maxParallel = Math.max(evidence.maxParallel, active.length);
    if (new Set(active.map((lane) => lane.type)).size > 1) evidence.mixedStages = true;
    if (state.writer?.travelerId) {
      const lane = state.lanes.find((item) => item.id === state.writer.travelerId);
      assert.equal(lane.key, state.writer.key); assert.equal(lane.phase, 'verifying'); assert.equal(lane.progress, state.writer.progress);
    }
    if (event.block) {
      const record = event.block.transactions[0].event;
      const lane = state.lanes.find((item) => item.id === record.travelerId);
      if (record.travelerId) assert.deepEqual(lane.records[record.type], { height: state.count, key: record.key });
      assert.equal(state.proof.blocks.length, state.count);
      evidence.commits.push({ record, count: state.count, time: state.time });
    }
    for (const change of event.changes || []) if (change.type === 'enter' && change.replaced) evidence.replacements.push({ ...change, time: state.time, others: state.lanes.filter((lane) => lane.slot !== change.slot).map((lane) => lane.id) });
  } });
  async function advance(ms) {
    for (let elapsed = 0; elapsed < ms; elapsed += 32) { time += 32; const next = callback; callback = null; if (next) next(time); await wait(); }
  }
  async function until(predicate, max = 240000) {
    for (let elapsed = 0; elapsed < max; elapsed += 128) { if (predicate(player.snapshot(false))) return; await advance(128); }
    assert.fail('Expected state not reached: ' + JSON.stringify(player.snapshot(false)));
  }
  return { player, advance, until, evidence };
}
let proof;
test.before(async () => {
  const h = harness(); await h.player.reset(); h.player.setContinuous(false); h.player.setSpeed(4); h.player.play();
  await h.until((s) => !s.running); proof = h.player.snapshot().proof; h.player.destroy();
});
async function modified(change) { const blocks = core.clone(proof.blocks); await change(blocks); return core.verifyChain(blocks, proof.trust); }
const eventsOf = (value) => value.blocks.map((block) => block.transactions[0].event);
test('SHA-256 known vector; canonical JSON; no non-finite input', async () => {
  assert.equal(await core.hash('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(core.canonical({ a: 1, b: true }), core.canonical({ b: true, a: 1 }));
  assert.throws(() => core.canonical({ value: Infinity })); assert.throws(() => core.canonical(undefined));
});
test('mixed campaign separates reported visits and independently computed attribution', async () => {
  assert.equal(proof.blocks.length, 21);
  const result = await core.verifyResults(proof.blocks, proof.trust);
  assert.equal(result.report.valid, true);
  assert.deepEqual(result.totals, { exposures: 5, websiteVisits: 3, reportedVisits: 3, attributedVisits: 2, completed: 5 });
  assert.equal(eventsOf(proof).filter((event) => event.type === 'attribution' && !event.data.credited).length, 1);
  assert.ok(eventsOf(proof).every((event) => !Object.hasOwn(event, 'group')));
});
test('attribution references actual earlier ad and visit hashes', () => {
  for (const event of eventsOf(proof).filter((event) => event.type === 'attribution')) {
    for (const [field, type] of [['exposure', 'ad'], ['visit', 'destination']]) {
      const ref = event.data[field]; const block = proof.blocks[ref.blockHeight - 1];
      assert.equal(ref.blockHash, block.hash); assert.equal(ref.eventId, block.transactions[0].event.id);
      assert.equal(block.transactions[0].event.type, type); assert.equal(block.transactions[0].event.travelerId, event.travelerId);
    }
  }
});
test('a destination-only traveler receives credit without website evidence', () => {
  const event = eventsOf(proof).find((item) => item.type === 'attribution' && item.travelerId === 'T004');
  assert.equal(event.data.credited, true);
  assert.equal(eventsOf(proof).some((item) => item.travelerId === 'T004' && item.type === 'website'), false);
});
test('a late visit is validly recorded but not credited', async () => {
  const index = eventsOf(proof).findIndex((event) => event.type === 'attribution' && !event.data.credited);
  const event = eventsOf(proof)[index];
  assert.equal(event.data.elapsedDays, 35); assert.equal(event.data.withinWindow, false);
  assert.equal((await core.verifyChain(proof.blocks, proof.trust)).blocks[index].state, 'verified');
});
test('30-day boundary is inclusive; late and pre-exposure visits are not credited', () => {
  const ad = core.clone(proof.blocks.find((block) => block.transactions[0].event.key === 'T001/ad'));
  const visit = core.clone(proof.blocks.find((block) => block.transactions[0].event.key === 'T001/destination'));
  const start = ad.transactions[0].event.data.exampleDay;
  for (const [days, credited] of [[0, true], [30, true], [31, false], [-1, false]]) {
    visit.transactions[0].event.data.exampleDay = start + days;
    assert.equal(core.attributionFor([ad, visit], 'T001').credited, credited);
  }
});
test('missing, wrong-traveler or cross-campaign evidence cannot produce credit', () => {
  assert.throws(() => core.attributionFor([], 'T001')); assert.throws(() => core.attributionFor(proof.blocks, 'T999'));
  const blocks = core.clone(proof.blocks); blocks.find((block) => block.transactions[0].event.key === 'T001/destination').transactions[0].event.campaignId = 'another-campaign';
  assert.throws(() => core.attributionFor(blocks, 'T001'));
});
test('the same visit cannot be counted twice', () => {
  assert.equal(core.attributionFor(proof.blocks, 'T001').notPreviouslyCounted, false);
  assert.equal(core.attributionFor(proof.blocks, 'T001').credited, false);
});
test('all selectable scenarios produce valid, distinct outcome totals', async () => {
  const cases = { none: [12, 0, 0, 0], website: [17, 5, 0, 0], destination: [22, 0, 5, 5], both: [27, 5, 5, 5], late: [27, 5, 5, 0] };
  for (const [scenario, [count, web, visit, attributed]] of Object.entries(cases)) {
    const h = harness(); await h.player.reset(); h.player.setScenario(scenario); h.player.setContinuous(false); h.player.setSpeed(4); h.player.play();
    await h.until((s) => !s.running); const next = h.player.snapshot().proof;
    const result = await core.verifyResults(next.blocks, next.trust);
    assert.equal(next.blocks.length, count); assert.equal(result.report.valid, true);
    assert.equal(result.totals.websiteVisits, web); assert.equal(result.totals.reportedVisits, visit); assert.equal(result.totals.attributedVisits, attributed);
    h.player.destroy();
  }
});
test('people and different stages overlap at 1x, 2x and 4x', async () => {
  for (const speed of [1, 2, 4]) {
    const h = harness(); await h.player.reset(); h.player.setContinuous(false); h.player.setSpeed(speed); h.player.play();
    await h.until((s) => !s.running); assert.ok(h.evidence.maxParallel >= 3); assert.equal(h.evidence.mixedStages, true);
    assert.equal(h.evidence.commits.length, 21); h.player.destroy();
  }
});
test('individual replacements preserve other lanes and append-only history', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play(); await h.until((s) => s.admitted >= 8); h.player.pause();
  assert.ok(h.evidence.replacements.length >= 3);
  for (const replacement of h.evidence.replacements) {
    const summary = h.evidence.commits.find((item) => item.record.key === replacement.replaced + '/summary');
    assert.ok(summary); assert.ok(replacement.time >= summary.time + HOLD_MS);
  }
  const before = h.player.snapshot().proof; h.player.play(); await h.until((s) => s.count > before.blocks.length + 3); h.player.pause();
  const after = h.player.snapshot().proof; assert.deepEqual(after.blocks.slice(0, before.blocks.length), before.blocks);
  assert.equal((await core.verifyChain(after.blocks, after.trust)).valid, true); h.player.destroy();
});
test('mix changes affect new arrivals only; pause freezes every lane', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play(); await h.until((s) => s.admitted === 5);
  const before = h.player.snapshot(false).lanes.map((lane) => [lane.id, lane.path]); h.player.setScenario('none');
  assert.deepEqual(h.player.snapshot(false).lanes.map((lane) => [lane.id, lane.path]), before);
  await h.until((s) => s.admitted >= 6); assert.equal(h.player.snapshot(false).lanes.find((lane) => lane.number === 6).path, 'none');
  h.player.pause(); const paused = h.player.snapshot(false); await h.advance(1500); assert.deepEqual(h.player.snapshot(false), paused); h.player.destroy();
});
test('slow signing allows other observations but no early or paused commit', async () => {
  let release; const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({ createLedger: async () => { const local = await core.createLedger(); return { ...local, prepare: async (draft) => { if (draft.key === 'T001/ad') await gate; return local.prepare(draft); } }; } });
  await h.player.reset(); h.player.play(); await h.until((s) => s.queue.length >= 4);
  assert.equal(h.player.snapshot(false).count, 2); assert.equal(h.player.snapshot(false).writer.progress, .8);
  h.player.pause(); release(); await h.advance(300); assert.equal(h.player.snapshot(false).count, 2);
  h.player.play(); await h.until((s) => s.count > 2); h.player.destroy();
});
test('reset cancels pending cryptography with no stale block or arrival', async () => {
  let release; const gate = new Promise((resolve) => { release = resolve; }); let calls = 0;
  const h = harness({ createLedger: async () => { const local = await core.createLedger(); calls += 1; return calls === 1 ? { ...local, prepare: async (draft) => { await gate; return local.prepare(draft); } } : local; } });
  await h.player.reset(); h.player.play(); await h.advance(200); await h.player.reset(); release(); await h.advance(400);
  const s = h.player.snapshot(false); assert.equal(s.count, 0); assert.equal(s.admitted, 0); assert.equal(s.error, ''); h.player.destroy();
});
test('capacity reserves attribution and closing records so everyone finishes', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play(); await h.until((s) => s.capacityClosed && !s.running, 2400000);
  const s = h.player.snapshot(); assert.ok(s.count <= core.MAX_BLOCKS); assert.ok(s.count >= core.MAX_BLOCKS - 5); assert.equal(s.completed, s.admitted);
  assert.equal((await core.verifyChain(s.proof.blocks, s.proof.trust)).valid, true); h.player.destroy();
});
test('editing attribution credit, rule, time or evidence reference fails', async () => {
  const index = eventsOf(proof).findIndex((event) => event.type === 'attribution');
  for (const mutate of [(data) => { data.credited = !data.credited; }, (data) => { data.windowDays = 60; }, (data) => { data.elapsedDays = 2; }, (data) => { data.exposure.blockHash = '0'.repeat(64); }, (data) => { data.visit.blockHeight = 1; }]) {
    const result = await modified((blocks) => mutate(blocks[index].transactions[0].event.data));
    assert.equal(result.valid, false); assert.equal(result.blocks[index].checks.attribution, false);
  }
});
test('invalid proofs return no totals; exact original bytes verify again', async () => {
  for (const index of [0, 2, 6, proof.blocks.length - 1]) {
    const blocks = core.clone(proof.blocks); blocks[index].transactions[0].event.data.note = 'edited';
    const result = await core.verifyResults(blocks, proof.trust); assert.equal(result.report.valid, false); assert.equal(result.totals, null);
    assert.equal(result.report.blocks[index].state, 'changed');
  }
  assert.equal((await core.verifyResults(core.clone(proof.blocks), proof.trust)).report.valid, true);
});
test('rehashing cannot restore signatures, approvals or original checkpoint', async () => {
  const result = await modified(async (blocks) => {
    blocks[2].transactions[0].event.data.publisher = 'Changed publisher';
    for (let index = 0; index < blocks.length; index += 1) {
      blocks[index].header.merkleRoot = await core.merkleRoot(blocks[index].transactions);
      blocks[index].header.previousHash = index ? blocks[index - 1].hash : core.ZERO;
      blocks[index].hash = await core.headerHash(blocks[index].header);
    }
  });
  assert.equal(result.valid, false); assert.equal(result.blocks[2].checks.eventSignature, false); assert.equal(result.blocks[2].checks.approvals, false); assert.equal(result.headValid, false);
});
test('missing/duplicate/forged approvals and malformed or replayed chains fail', async () => {
  const changes = [(blocks) => blocks[0].approvals.pop(), (blocks) => { blocks[0].approvals[1] = core.clone(blocks[0].approvals[0]); },
    (blocks) => { blocks[0].approvals[0].signature = '00'.repeat(64); }, (blocks) => blocks.pop(), (blocks) => blocks.reverse(), (blocks) => blocks.splice(4, 1), (blocks) => blocks.push(core.clone(blocks[2]))];
  for (const change of changes) assert.equal((await modified(change)).valid, false);
  for (const value of [null, {}, [null]]) assert.equal((await core.verifyChain(value, proof.trust)).valid, false);
});
test('keys stay private; replacing the trusted registry invalidates signatures', async () => {
  for (const key of Object.values(proof.trust.publicKeys)) assert.equal(key.d, undefined);
  const ledger = await core.createLedger(); const trust = core.clone(proof.trust); trust.publicKeys.publisher = ledger.snapshot().trust.publicKeys.publisher;
  assert.equal((await core.verifyChain(proof.blocks, trust)).valid, false); ledger.dispose();
});
test('a verified ticket cannot be changed or committed twice', async () => {
  const ledger = await core.createLedger(); const draft = { key: 'campaign', type: 'campaign', travelerId: null, observedAtMs: 0, data: { name: 'Original' } };
  const ticket = await ledger.prepare(draft); assert.equal(ledger.snapshot().blocks.length, 0);
  ticket.block.transactions[0].event.data.name = 'Forged'; ledger.commit(ticket); assert.equal(ledger.snapshot().blocks[0].transactions[0].event.data.name, 'Original');
  assert.throws(() => ledger.commit(ticket)); ledger.dispose(); await assert.rejects(ledger.prepare(draft));
});
test('a claimed credit without evidence cannot be approved', async () => {
  const ledger = await core.createLedger();
  await assert.rejects(ledger.prepare({ key: 'T001/attribution', type: 'attribution', travelerId: 'T001', observedAtMs: 0, data: { credited: true } }));
  ledger.dispose(); assert.throws(() => core.createTraveler('__proto__', 1));
});
test('unlisted design and privacy contract remains intact', () => {
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /name="robots" content="noindex, nofollow, noarchive"/); assert.match(html, /<main id="main"/); assert.match(html, /skip-link/);
  assert.doesNotMatch(html, /data-group|tamper-proof|Visit Grand Junction|Viant|Foursquare|Horizon Media/);
  for (const file of ['ad-verification-core.js', 'ad-verification-player.js', 'ad-verification.js']) assert.doesNotMatch(fs.readFileSync(path.join(root, 'js/demos', file), 'utf8'), /\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage|navigator\.geolocation/);
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/adVerification.json'), 'utf8'));
  assert.equal(metadata.published, false); assert.equal(metadata.hidden, true); assert.equal(metadata.noindex, true);
});
test('built assets are present, noindex, and excluded from discovery', (t) => {
  const pub = path.join(root, 'public'); if (!fs.existsSync(pub)) { t.skip('Requires full website build.'); return; }
  assert.match(fs.readFileSync(path.join(pub, 'demos/ad-verification.html'), 'utf8'), /name="robots"[^>]*noindex/);
  for (const file of ['js/demos/ad-verification-core.js', 'js/demos/ad-verification-player.js', 'js/demos/ad-verification.js', 'css/components/ad-verification.css', 'img/projects/ad-verification-portraits.webp', 'img/projects/ad-verification-mark.webp', 'img/projects/ad-verification-mountains.webp']) assert.ok(fs.statSync(path.join(pub, file)).size > 0);
  for (const file of ['sitemap.xml', 'dist/search-index.json', 'app-content/v1/catalog.json']) if (fs.existsSync(path.join(pub, file))) assert.doesNotMatch(fs.readFileSync(path.join(pub, file), 'utf8'), /adVerification|demos\/ad-verification|Cedar Valley/);
});
