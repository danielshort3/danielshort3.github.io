'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../../js/demos/ad-verification-core');
const { createPlayer, HOLD_MS } = require('../../js/demos/ad-verification-player');
const root = path.resolve(__dirname, '../..');
const yieldToCrypto = () => new Promise((resolve) => setTimeout(resolve, 1));
function harness(options = {}) {
  let fn = null;
  let time = 0;
  const evidence = { parallel: 0, mixed: 0, batches: [], replacements: 0, old: [] };
  const player = createPlayer({ ...options, requestFrame: (callback) => { fn = callback; return 1; }, cancelFrame: () => { fn = null; }, onChange: (event, state) => {
    const active = state.lanes.filter((lane) => lane.active);
    evidence.parallel = Math.max(evidence.parallel, active.length);
    if (new Set(active.map((lane) => lane.active)).size > 1) evidence.mixed += 1;
    state.lanes.forEach((lane, index) => {
      const old = evidence.old[index];
      if (lane.id && old?.id && lane.id !== old.id) {
        assert.equal(old.phase, 'done'); assert.ok(old.recorded.end); evidence.replacements += 1;
      }
    });
    evidence.old = state.lanes;
    if (event.kind === 'block') {
      for (const signed of event.block.records) {
        const name = player.describe(signed.receipt.id)?.traveler;
        if (name) {
          const lane = state.lanes.find((item) => item.id === name);
          assert.equal(lane.recorded[signed.receipt.type].id, signed.receipt.id);
          assert.equal(lane.recorded[signed.receipt.type].block, event.block.header.height);
        }
      }
      evidence.batches.push(event.block.records.length);
    }
  } });
  async function advance(ms) {
    for (let elapsed = 0; elapsed < ms; elapsed += 32) {
      time += 32; const next = fn; fn = null; if (next) next(time); await yieldToCrypto();
    }
  }
  async function until(predicate, max = 2000000) {
    for (let elapsed = 0; elapsed < max; elapsed += 128) {
      if (predicate(player.snapshot(false))) return;
      await advance(128);
    }
    assert.fail('Expected state not reached: ' + JSON.stringify(player.snapshot(false)));
  }
  return { player, evidence, advance, until };
}
async function campaign(scenario = 'mixed', speed = 4) {
  const h = harness(); await h.player.reset(); h.player.setScenario(scenario);
  h.player.setSpeed(speed); h.player.setContinuous(false); h.player.play();
  await h.until((s) => s.completed === 5 && !s.running);
  return h;
}
async function minimal() {
  const session = await core.createSession();
  await session.append([
    await session.issue('campaign', {}, { rule: core.RULE.id, window: core.RULE.days }),
    await session.issue('purchase', {}, { authorized: true })
  ]);
  const ad = await session.observe('ad', 'T001', 2);
  const visit = await session.observe('visit', 'T001', 8);
  await session.append([ad, visit]);
  return { session, ad, visit };
}
let baseline;
test.before(async () => { baseline = await campaign(); });
test.after(() => baseline.player.destroy());
const sourceProof = () => baseline.player.snapshot().proof;
test('known SHA-256 vector and deterministic finite canonical JSON', async () => {
  assert.equal(await core.hash('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.equal(core.canonical({ z: 1, a: true }), core.canonical({ a: true, z: 1 }));
  assert.throws(() => core.canonical({ a: Infinity })); assert.throws(() => core.canonical(undefined));
});
test('multiple receipts from overlapping travelers enter real batched blocks', async () => {
  const proof = sourceProof();
  assert.equal(core.receipts(proof.blocks).length, 21);
  assert.ok(proof.blocks.length < 21);
  assert.ok(baseline.evidence.batches.some((n) => n > 1));
  assert.ok(baseline.evidence.parallel >= 3); assert.ok(baseline.evidence.mixed > 0);
  const checked = await core.verifyProof(proof); assert.equal(checked.valid, true);
  assert.deepEqual(checked.totals, { exposures: 5, websites: 3, visits: 3, attributed: 2, corrections: 0 });
});
test('exports contain no private evidence, synthetic traveler identifiers or private keys', () => {
  const proof = sourceProof(); const json = JSON.stringify(proof);
  assert.doesNotMatch(json, /"traveler"|T00\d|hiking-guides|"place"|"salt"|"privateKey"/);
  for (const key of Object.values(proof.trust.keys)) assert.equal(key.d, undefined);
  assert.ok(core.receipts(proof.blocks).every((r) => /^[a-f0-9]{64}$/.test(r.evidenceDigest)));
});
test('salted evidence fingerprints differ even for identical provider observations', async () => {
  const session = await core.createSession();
  const first = await session.observe('ad', 'T001', 1); const second = await session.observe('ad', 'T001', 1);
  assert.notEqual(first.receipt.evidenceDigest, second.receipt.evidenceDigest); session.dispose();
});
test('record verification and authorized evidence checking are different operations', async () => {
  const decision = core.receipts(sourceProof().blocks).find((r) => r.type === 'attribution' && r.data.credited);
  const checked = await baseline.player.audit(decision.id);
  assert.equal(checked.record, true); assert.equal(checked.evidence, 'checked'); assert.equal(checked.calculation.days, 6);
  await baseline.player.action('withhold', decision.id, true);
  const unavailable = await baseline.player.audit(decision.id);
  assert.deepEqual(unavailable, { record: true, evidence: 'unavailable' });
  assert.equal((await core.verifyProof(sourceProof())).valid, true);
  await baseline.player.action('withhold', decision.id, false);
  assert.equal((await baseline.player.audit(decision.id)).evidence, 'checked');
});
test('missing supporting evidence also prevents calculation reproduction', async () => {
  const decision = core.receipts(sourceProof().blocks).find((r) => r.type === 'attribution');
  await baseline.player.action('withhold', decision.refs[0], true);
  assert.equal((await baseline.player.audit(decision.id)).evidence, 'unavailable');
  await baseline.player.action('withhold', decision.refs[0], false);
});
test('a correctly signed false claim remains a claim; evidence audit catches the calculation mismatch', async () => {
  const { session, ad, visit } = await minimal();
  const wrong = await session.issue('attribution', { traveler: 'T001' }, { credited: false, rule: core.RULE.id, window: core.RULE.days }, [ad.receipt.id, visit.receipt.id]);
  await session.append([wrong]);
  assert.equal((await core.verifyProof(session.snapshot())).valid, true);
  assert.equal((await session.audit(wrong.receipt.id)).evidence, 'mismatch'); session.dispose();
});
test('matching rule includes day 30, excludes late/pre-exposure visits, and requires matching evidence', () => {
  const ad = { type: 'ad', traveler: 'T001', campaign: 'c', day: 3 };
  for (const [days, credited] of [[0, true], [30, true], [31, false], [-1, false]]) {
    assert.equal(core.decision(ad, { ...ad, type: 'visit', day: 3 + days }).credited, credited);
  }
  assert.throws(() => core.decision(ad, { ...ad, type: 'visit', traveler: 'T002' }));
  assert.throws(() => core.decision(ad, { ...ad, type: 'visit', campaign: 'other' }));
});
test('destination-only attribution requires no website receipt; late visits are not credited', async () => {
  const all = core.receipts(sourceProof().blocks).filter((r) => r.type === 'attribution');
  const results = await Promise.all(all.map((r) => baseline.player.audit(r.id)));
  assert.ok(results.some((a) => a.calculation?.days === 35 && !a.calculation.credited));
  const direct = all.find((r) => baseline.player.describe(r.id).traveler === 'T004');
  assert.equal(direct.data.credited, true);
  assert.equal(direct.refs.length, 2);
});
test('signed reports are exact historical snapshots and edited report totals fail', async () => {
  const { session, ad, visit } = await minimal();
  await session.append([await session.observe('attribution', 'T001', 9, { ad: ad.receipt.id, visit: visit.receipt.id })]);
  const report = await session.report(); const original = session.snapshot();
  assert.equal(report.records[0].receipt.data.totals.attributed, 1);
  const edited = core.clone(original); edited.blocks.at(-1).records[0].receipt.data.totals.attributed = 4;
  const result = await core.verifyProof(edited); assert.equal(result.valid, false); assert.equal(result.totals, null);
  assert.equal((await core.verifyProof(original)).valid, true); session.dispose();
});
test('authorized corrections append; the original report remains byte-identical and current total changes once', async () => {
  const { session, ad, visit } = await minimal();
  const signed = await session.observe('attribution', 'T001', 9, { ad: ad.receipt.id, visit: visit.receipt.id });
  await session.append([signed]); await session.report(); const before = session.snapshot();
  await session.correct(signed.receipt.id); const after = session.snapshot();
  assert.deepEqual(after.blocks.slice(0, before.blocks.length), before.blocks);
  assert.equal((await core.verifyProof(after)).totals.attributed, 0);
  await assert.rejects(session.correct(signed.receipt.id));
  const updated = await session.report(); assert.equal(updated.records[0].receipt.data.totals.attributed, 0);
  assert.equal((await core.verifyProof(session.snapshot())).valid, true); session.dispose();
});
test('replayed visits and unauthorized correction targets cannot be accepted', async () => {
  const { session, ad, visit } = await minimal();
  await session.append([await session.observe('attribution', 'T001', 9, { ad: ad.receipt.id, visit: visit.receipt.id })]);
  await assert.rejects(session.append([await session.observe('attribution', 'T001', 9, { ad: ad.receipt.id, visit: visit.receipt.id })]));
  await assert.rejects(session.append([await session.issue('correction', {}, { credited: false, reason: 'duplicate-visit' }, [ad.receipt.id])]));
  session.dispose();
});
test('three real local copies detect divergence without changing the others or canonical history', async () => {
  const { session } = await minimal(); const original = session.snapshot();
  await session.manageCopy(1, 'alter');
  assert.deepEqual(session.copyStates().map((c) => c.status), ['Up to date', 'Mismatch', 'Up to date']);
  assert.deepEqual(session.snapshot(), original);
  await session.manageCopy(1, 'restore'); assert.ok(session.copyStates().every((c) => c.status === 'Up to date'));
  session.dispose();
});
test('a valid lagging copy is Behind, not current or corrupted; catch-up is verified', async () => {
  const { session } = await minimal(); await session.manageCopy(1, 'pause'); await session.report();
  assert.equal(session.copyStates()[1].status, 'Behind');
  assert.equal(session.copyStates()[0].status, 'Up to date');
  await session.manageCopy(1, 'restore'); assert.equal(session.copyStates()[1].status, 'Up to date'); session.dispose();
});
test('all outcomes and all three speeds produce consistent receipts and totals', async () => {
  for (const [scenario, attributed] of Object.entries({ none: 0, website: 0, destination: 5, both: 5, late: 0 })) {
    const h = await campaign(scenario); const result = await core.verifyProof(h.player.snapshot().proof);
    assert.equal(result.valid, true); assert.equal(result.totals.attributed, attributed); assert.equal(h.player.snapshot(false).completed, 5); h.player.destroy();
  }
  for (const speed of [1, 2]) {
    const h = await campaign('mixed', speed); assert.ok(h.evidence.parallel >= 3);
    assert.equal((await core.verifyProof(h.player.snapshot().proof)).totals.attributed, 2); h.player.destroy();
  }
});
test('people can progress to later observations while the first batch is still awaiting cryptography', async () => {
  let release; const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({ createSession: async () => {
    const session = await core.createSession();
    return { ...session, prepare: async (records) => { if (records.some((r) => r.receipt.type === 'ad')) await gate; return session.prepare(records); } };
  } });
  await h.player.reset(); h.player.play(); await h.until((s) => s.queued >= 8);
  const state = h.player.snapshot(false); assert.equal(state.blockCount, 1);
  assert.ok(state.lanes.some((lane) => lane.observed.website && !lane.recorded.ad));
  h.player.pause(); const paused = h.player.snapshot(false); release(); await h.advance(500);
  assert.deepEqual(h.player.snapshot(false), paused);
  h.player.play(); await h.until((s) => s.blockCount > 1); h.player.destroy();
});
test('pause freezes all visible clocks and reset rejects stale source work', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.play();
  await h.until((s) => s.writer && s.admitted === 5); h.player.pause();
  const paused = h.player.snapshot(false); await h.advance(2000); assert.deepEqual(h.player.snapshot(false), paused);
  await h.player.reset(); await h.advance(1000);
  assert.equal(h.player.snapshot(false).recordCount, 0); assert.equal(h.player.snapshot(false).admitted, 0); h.player.destroy();
});
test('individual replacement and new-arrival mix do not rewrite existing history', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.setContinuous(true); h.player.play();
  await h.until((s) => s.admitted === 5); const paths = h.player.snapshot(false).lanes.map((l) => l.path);
  h.player.setScenario('none'); assert.deepEqual(h.player.snapshot(false).lanes.map((l) => l.path), paths);
  await h.until((s) => s.admitted >= 8); h.player.pause(); assert.ok(h.evidence.replacements >= 3);
  const before = h.player.snapshot().proof; h.player.play(); await h.until((s) => s.blockCount >= before.blocks.length + 2); h.player.pause();
  const after = h.player.snapshot().proof; assert.deepEqual(after.blocks.slice(0, before.blocks.length), before.blocks);
  assert.equal((await core.verifyProof(after)).valid, true); h.player.destroy();
});
test('manual review cancels a pending candidate without losing or duplicating its receipts', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.setContinuous(false); h.player.play();
  await h.until((s) => s.writer && s.recordCount >= 5); await h.player.action('report');
  h.player.play(); await h.until((s) => !s.running && s.completed === 5);
  const proof = h.player.snapshot().proof;
  assert.equal(new Set(core.receipts(proof.blocks).map((r) => r.id)).size, 22);
  assert.equal((await core.verifyProof(proof)).valid, true); h.player.destroy();
});
test('capacity reservations finish every admitted traveler and preserve review headroom', async () => {
  const h = harness(); await h.player.reset(); h.player.setSpeed(4); h.player.setContinuous(true); h.player.play();
  await h.until((s) => s.limited && !s.running, 3000000);
  const state = h.player.snapshot(false);
  assert.ok(state.recordCount <= core.LIMIT - 12); assert.equal(state.completed, state.admitted);
  await h.player.action('report'); assert.equal((await core.verifyProof(h.player.snapshot().proof)).valid, true); h.player.destroy();
});
test('signature, role, batch membership, ordering, missing blocks and checkpoint edits fail closed', async () => {
  const mutations = [
    (p) => { p.blocks[0].records[0].receipt.data.window = 99; },
    (p) => { p.blocks[0].records[0].signature = '00'.repeat(64); },
    (p) => { p.blocks[1].records[0].receipt.source = 'advertiser'; },
    (p) => { p.blocks[0].approvals[1] = core.clone(p.blocks[0].approvals[0]); },
    (p) => { p.blocks[0].approvals.pop(); }, (p) => { p.blocks.pop(); },
    (p) => { p.blocks.reverse(); }, (p) => { p.blocks[0].records.reverse(); }
  ];
  for (const change of mutations) { const p = sourceProof(); change(p); assert.equal((await core.verifyProof(p)).valid, false); }
  for (const p of [null, {}, { version: 5, blocks: [null], trust: {} }]) assert.equal((await core.verifyProof(p)).valid, false);
});
test('rebuilding hashes cannot forge provider signatures or validator approvals', async () => {
  const p = sourceProof(); p.blocks[1].records[0].receipt.data.count = 9;
  for (let i = 0; i < p.blocks.length; i += 1) {
    p.blocks[i].header.previous = p.blocks[i - 1]?.hash || core.ZERO;
    p.blocks[i].header.root = await core.merkleRoot(p.blocks[i].records);
    p.blocks[i].hash = await core.headerHash(p.blocks[i].header);
  }
  p.trust.head = p.blocks.at(-1).hash;
  assert.equal((await core.verifyProof(p)).valid, false);
});
test('candidate tickets use private verified bytes and cannot append twice or after replica mutation', async () => {
  const { session } = await minimal();
  const record = await session.observe('website', 'T001', 4);
  const ticket = await session.prepare([record]); ticket.block.records[0].receipt.data.count = 99;
  session.commit(ticket); assert.equal(session.snapshot().blocks.at(-1).records[0].receipt.data.count, 1);
  assert.throws(() => session.commit(ticket));
  const next = await session.prepare([await session.observe('ad', 'T002', 4)]);
  await session.manageCopy(1, 'pause'); assert.throws(() => session.commit(next)); session.dispose();
});
test('unlisted and no-tracking source contracts', () => {
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /noindex, nofollow, noarchive/); assert.match(html, /<main id="main"/); assert.match(html, /skip-link/);
  assert.doesNotMatch(html, /data-group|tamper-proof|Visit Grand Junction/);
  for (const file of ['ad-verification-core.js', 'ad-verification-player.js', 'ad-verification.js']) assert.doesNotMatch(fs.readFileSync(path.join(root, 'js/demos', file), 'utf8'), /\bfetch\s*\(|sendBeacon|XMLHttpRequest|localStorage|sessionStorage|navigator\.geolocation/);
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/adVerification.json'), 'utf8'));
  assert.equal(metadata.published, false); assert.equal(metadata.hidden, true); assert.equal(metadata.noindex, true);
});
test('built publication excludes the unlisted project from discovery', (t) => {
  const pub = path.join(root, 'public'); if (!fs.existsSync(pub)) { t.skip('Run the full site build for publication checks.'); return; }
  assert.match(fs.readFileSync(path.join(pub, 'demos/ad-verification.html'), 'utf8'), /noindex, nofollow, noarchive/);
  for (const file of ['js/demos/ad-verification-core.js', 'js/demos/ad-verification-player.js', 'js/demos/ad-verification.js', 'css/components/ad-verification.css']) assert.ok(fs.statSync(path.join(pub, file)).size > 0);
  for (const file of ['sitemap.xml', 'dist/search-index.json', 'app-content/v1/catalog.json']) if (fs.existsSync(path.join(pub, file))) assert.doesNotMatch(fs.readFileSync(path.join(pub, file), 'utf8'), /adVerification|demos\/ad-verification|Cedar Valley/);
});
