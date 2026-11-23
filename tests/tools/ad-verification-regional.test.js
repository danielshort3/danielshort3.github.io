'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const core = require('../../js/demos/ad-verification-core');
const regional = require('../../js/demos/ad-verification-regional');
const campaign = 'regional-' + 'a'.repeat(24);
const token = 'c'.repeat(64);
const zero = '0'.repeat(64);
const totals = (views, observations, matched) => ({ views, observations, matched });
const copy = (item) => JSON.parse(JSON.stringify(item));
const flip = (hex) => (hex[0] === 'a' ? 'b' : 'a') + hex.slice(1);
const hex = (bytes) => Buffer.from(bytes).toString('hex');
const isValid = async (proof) => (await regional.verifyProof(proof)).valid;
let completed;
let snapshots;
let proofs;
let sourcePairs;
let sourceKeys;

test.before(async () => {
  completed = await regional.createSession();
  snapshots = [completed.snapshot()];
  proofs = [completed.proof()];
  for (let i = 0; i < 3; i += 1) {
    snapshots.push(await completed.nextBatch());
    proofs.push(completed.proof());
  }
  const names = ['field-guide', 'weekend-reader', 'downtown', 'interchange', 'outsider'];
  const pairs = await Promise.all(names.map(() =>
    webcrypto.subtle.generateKey({ name: 'ECDSA', namedCurve: 'P-256' }, false, ['sign', 'verify'])));
  sourcePairs = Object.fromEntries(names.map((name, i) => [name, pairs[i]]));
  sourceKeys = Object.fromEntries(names.filter((name) => name !== 'outsider').map((name) => [name, sourcePairs[name].publicKey]));
});

test('signed receiving-app views arrive before the destination evidence and every sealed prefix verifies', async () => {
  assert.deepEqual(snapshots.map((s) => s.totals), [totals(0, 0, 0), totals(6, 0, 0), totals(6, 5, 1), totals(6, 8, 2)]);
  assert.deepEqual(snapshots.map((s) => s.completedBatches), [0, 1, 2, 3]);
  assert.ok(snapshots.every((s) => s.totalBatches === 3));
  assert.deepEqual(snapshots[1].journeys.map((j) => j.views), [2, 1, 0, 1, 1, 1]);
  assert.ok(snapshots[1].journeys.every((j) => !j.stops.length));
  const travelerViews = snapshots[1].journeys[0].viewReceipts;
  assert.deepEqual(travelerViews.map((v) => [v.app, v.day, v.visiblePercent, v.seconds]), [
    ['Field Guide app', 1, 100, 4], ['Weekend Reader app', 2, 85, 3]
  ]);
  assert.ok(travelerViews.every((v) => v.id && v.creative && v.placement));
  for (const [i, proof] of proofs.entries()) {
    assert.equal(proof.checkpoint.height, i);
    assert.equal(await isValid(proof), true);
  }
  const { batches } = proofs[3];
  assert.deepEqual(batches.map((b) => b.previousHash), [zero, batches[0].hash, batches[1].hash]);
  assert.equal(proofs[3].checkpoint.head, batches[2].hash);
  assert.deepEqual(await completed.audit(), { valid: true, matched: 2 });
});

test('multiple signed views and three regional stops count once across batches and an exact retry is discarded', async () => {
  const [traveler, boundary, missing, transit, late, unlinked] = snapshots[3].journeys;
  assert.deepEqual(traveler.stops.map((s) => [s.label, s.day]), [
    ['Downtown district', 8], ['Regional trailhead', 9], ['Lodging district', 10]
  ]);
  assert.ok(traveler.stops.every((s) => s.id && s.source && s.dwellMinutes > 0));
  assert.match(traveler.tokenPreview, /^[a-f0-9]{12}$/);
  assert.ok(traveler.viewReceipts.concat(traveler.stops).every((receipt) => receipt.tokenPreview === traveler.tokenPreview));
  assert.equal(traveler.outcome, 'Matched');
  assert.equal(boundary.stops[0].day - boundary.viewReceipts[0].day, 30);
  assert.equal(boundary.outcome, 'Matched');
  assert.equal(missing.views, 0);
  assert.equal(missing.outcome, 'Missing signed view');
  assert.equal(transit.outcome, 'Transit only');
  assert.equal(late.outcome, 'Outside window');
  assert.equal(unlinked.outcome, 'Unlinked');
  assert.equal(unlinked.tokenPreview, '');
  assert.ok(unlinked.viewReceipts.concat(unlinked.stops).every((receipt) => receipt.tokenPreview === ''));
  assert.deepEqual(snapshots.map((s) => s.rejectedReplays), [0, 0, 0, 1]);
  assert.deepEqual(await completed.nextBatch(), snapshots[3]);
});

test('public exports are allowlisted aggregate fields, with no source receipt, identifier, observation or private key', () => {
  const proof = proofs[3];
  function keysAre(object, keys) {
    assert.deepEqual(Object.keys(object).sort(), keys.sort());
  }
  const publicFields = ['version', 'campaign', 'region', 'rule', 'validators', 'checkpoint', 'batches'];
  for (const prefix of proofs) keysAre(prefix, publicFields);
  assert.deepEqual(proofs[0].batches, []);
  assert.equal(proofs[0].checkpoint.head, zero);
  keysAre(proof.validators, ['network', 'reviewer']);
  assert.ok(Object.values(proof.validators).every((k) => /^04[a-f0-9]{128}$/.test(k)));
  keysAre(proof.checkpoint, ['height', 'head', 'signature']);
  for (const block of proof.batches) {
    keysAre(block, ['number', 'fingerprint', 'previousHash', 'hash', 'totals', 'approvals']);
    keysAre(block.totals, ['views', 'observations', 'matched']);
    keysAre(block.approvals, ['network', 'reviewer']);
    assert.match(block.fingerprint, /^[a-f0-9]{64}$/);
  }
  const json = JSON.stringify(proofs);
  const privateValues = snapshots[3].journeys.flatMap((j) => [
    j.tokenPreview, j.label, ...j.viewReceipts.flatMap((v) => [v.id, v.app, v.placement, v.creative]),
    ...j.stops.flatMap((s) => [s.id, s.label, s.source])
  ]);
  for (const value of privateValues.filter(Boolean)) assert.equal(json.includes(value), false, value);
  assert.doesNotMatch(json, /"token"|"device"|"day"|"minute"|"journeys"|"signedAt"|"privateKey"|"location"|"issuer"/);
});

test('every altered signed field, swapped validator, broken link, truncated tip or unapproved public field is rejected', async () => {
  const changes = [
    (p) => { p.batches[0].fingerprint = flip(p.batches[0].fingerprint); },
    (p) => { p.batches[2].totals.matched = 3; },
    (p) => { p.batches[1].previousHash = zero; },
    (p) => { p.batches[0].hash = flip(p.batches[0].hash); },
    (p) => { p.batches[0].number = 2; },
    (p) => { p.batches.reverse(); },
    (p) => { p.batches.pop(); },
    (p) => { p.batches.pop(); p.checkpoint.height = 2; p.checkpoint.head = p.batches[1].hash; },
    (p) => { p.checkpoint.signature = flip(p.checkpoint.signature); },
    (p) => { p.batches[1].approvals.network = p.batches[1].approvals.reviewer; },
    (p) => { p.validators.network = p.validators.reviewer; },
    (p) => { [p.validators.network, p.validators.reviewer] = [p.validators.reviewer, p.validators.network]; },
    (p) => { p.campaign = campaign; },
    (p) => { p.region = 'Elsewhere'; },
    (p) => { p.rule += '-different'; },
    (p) => { p.batches[0].receiptRows = []; }
  ];
  for (const edit of changes) {
    const edited = copy(proofs[3]);
    edit(edited);
    assert.equal(await isValid(edited), false);
  }
  for (const malformed of [null, {}, ['proof'], { ...proofs[3], validators: null }]) {
    assert.equal(await isValid(malformed), false);
  }
});

test('the private audit pins this campaign, the current checkpoint and both enrolled validators', async () => {
  assert.equal(await isValid(proofs[2]), true);
  assert.deepEqual(await completed.audit(proofs[2]), { valid: false, matched: 0 });
  const other = await regional.createSession();
  for (let i = 0; i < 3; i += 1) await other.nextBatch();
  assert.equal(await isValid(other.proof()), true);
  assert.notEqual(other.proof().campaign, proofs[3].campaign);
  assert.notDeepEqual(other.proof().validators, proofs[3].validators);
  assert.notEqual(other.snapshot().journeys[0].tokenPreview, snapshots[3].journeys[0].tokenPreview);
  assert.notEqual(other.proof().batches[0].fingerprint, proofs[3].batches[0].fingerprint);
  assert.deepEqual(await completed.audit(other.proof()), { valid: false, matched: 0 });
  const changed = completed.proof();
  changed.batches[2].totals.matched += 1;
  assert.deepEqual(await completed.audit(changed), { valid: false, matched: 0 });
  assert.deepEqual(await completed.audit(), { valid: true, matched: 2 });
});

// Independently issue synthetic input to exercise the same signature and policy checks used by the private audit.
function statement(issuer, day, minute = 720) {
  const type = ['field-guide', 'weekend-reader'].includes(issuer) ? 'view' : 'stop';
  const base = {
    id: hex(webcrypto.getRandomValues(new Uint8Array(16))), campaign, region: 'High Country',
    issuer, type, day, minute, token, authorized: true
  };
  const data = type === 'view' ? {
    received: true, viewable: true, visiblePercent: 90, seconds: 2, creative: 'Test creative', placement: 'Test placement'
  } : { dwellMinutes: 18 };
  return { ...base, ...data };
}
async function signed(event, signer = event.issuer) {
  const text = 'REGIONAL/source/v1\n' + core.canonical(event);
  const signature = await webcrypto.subtle.sign({ name: 'ECDSA', hash: 'SHA-256' },
    sourcePairs[signer].privateKey, new TextEncoder().encode(text));
  return { statement: event, signature: hex(signature) };
}
const review = (batches) => regional.reviewEvidence(campaign, sourceKeys, batches);

test('the signed demo rule requires at least half the ad area for one continuous second', async () => {
  assert.ok(proofs.every((proof) => proof.rule.includes('confirmed-view/min-50pct/continuous-1s/')));
  const view = statement('field-guide', 1);
  const stop = await signed(statement('downtown', 9));
  for (const [visiblePercent, seconds] of [[49, 10], [49.99, 2], [100, 0], [50, 0.99]]) {
    const receipt = await signed({ ...view, visiblePercent, seconds });
    await assert.rejects(review([[receipt, stop]]), /source receipt/);
  }
  for (const [visiblePercent, seconds] of [[50, 1], [50.5, 1.25]]) {
    const receipt = await signed({ ...view, visiblePercent, seconds });
    assert.deepEqual((await review([[receipt, stop]]))[0].totals, totals(1, 1, 1));
  }
});

test('private auditing checks role, app receipt, signature, currency of campaign, grant and trust directory', async () => {
  const view = statement('field-guide', 1);
  const stop = statement('downtown', 4);
  const goodView = await signed(view);
  const goodStop = await signed(stop);
  const good = await review([[goodView], [goodStop]]);
  assert.deepEqual(good.map((entry) => entry.totals), [totals(1, 0, 0), totals(1, 1, 1)]);
  for (const edit of [
    { day: 2 }, { visiblePercent: 63 }, { seconds: 5 }, { creative: 'Changed creative' },
    { token: 'b'.repeat(64) }, { authorized: false, token: null }
  ]) {
    const altered = { statement: { ...view, ...edit }, signature: goodView.signature };
    await assert.rejects(review([[altered, goodStop]]), /signature/);
  }
  const badStop = { statement: { ...stop, dwellMinutes: 25 }, signature: goodStop.signature };
  await assert.rejects(review([[goodView, badStop]]), /signature/);
  for (const invalid of [
    await signed(view, 'outsider'), await signed(stop, 'field-guide'),
    await signed({ ...view, issuer: 'downtown' }), await signed({ ...stop, issuer: 'weekend-reader' }),
    await signed({ ...view, received: false }), await signed({ ...view, viewable: false }),
    await signed({ ...view, visiblePercent: 0 }), await signed({ ...view, seconds: 0 }),
    await signed({ ...view, authorized: false }), await signed({ ...view, token: null }),
    await signed({ ...stop, campaign: 'regional-' + 'b'.repeat(24) }), await signed({ ...stop, region: 'Other region' })
  ]) await assert.rejects(review([[invalid]]));
  await assert.rejects(regional.reviewEvidence(campaign, {}, [[goodView]]), /signature/);
  await assert.rejects(review([[goodView], [goodStop, copy(goodView)]]), /Repeated source receipt/);
  await assert.rejects(review([[goodView, await signed({ ...stop, id: view.id })]]), /Repeated source receipt/);
});

test('joining compares only authorized tokens and signed occurrence times, including both 30-day boundaries', async () => {
  const view = await signed(statement('field-guide', 1));
  for (const [day, minute, expected] of [[1, 719, 0], [1, 720, 1], [1, 721, 1], [31, 720, 1], [31, 721, 0]]) {
    const stop = await signed(statement('downtown', day, minute));
    const summary = await review([[view, stop]]);
    assert.equal(summary[0].totals.matched, expected);
  }
  const stop = statement('downtown', 4);
  for (const changes of [{ authorized: false, token: null }, { token: 'b'.repeat(64) }, { issuer: 'interchange' }]) {
    const summary = await review([[view, await signed({ ...stop, ...changes })]]);
    assert.deepEqual(summary[0].totals, totals(1, 1, 0));
  }
  const denied = await signed({ ...statement('weekend-reader', 1), authorized: false, token: null });
  assert.deepEqual((await review([[denied, await signed(stop)]]))[0].totals, totals(1, 1, 0));
  assert.deepEqual((await review([[await signed(stop)], [view]])).map((entry) => entry.totals), [totals(0, 1, 0), totals(1, 1, 1)]);
  const newerView = await signed(statement('weekend-reader', 8));
  const laterStop = await signed(statement('downtown', 36));
  assert.deepEqual((await review([[view, newerView], [laterStop]]))[1].totals, totals(2, 1, 1));
});

test('the private HMAC requires explicit authorization and binds an enrolled credential to the campaign and secret key', async () => {
  const algorithm = { name: 'HMAC', hash: 'SHA-256', length: 256 };
  const [key, otherKey] = await Promise.all([false, false].map((extractable) =>
    webcrypto.subtle.generateKey(algorithm, extractable, ['sign'])));
  assert.equal(key.extractable, false);
  await assert.rejects(webcrypto.subtle.exportKey('raw', key));
  const credential = '1'.repeat(64);
  const match = regional.deriveCampaignToken;
  const actual = await match(key, campaign, credential, true);
  const text = 'REGIONAL/join/v1\n' + core.canonical({ campaign, credential });
  assert.equal(actual, hex(await webcrypto.subtle.sign('HMAC', key, new TextEncoder().encode(text))));
  assert.equal(await match(key, campaign, credential, true), actual);
  assert.notEqual(await match(otherKey, campaign, credential, true), actual);
  assert.notEqual(await match(key, 'regional-' + 'd'.repeat(24), credential, true), actual);
  assert.notEqual(await match(key, campaign, '2'.repeat(64), true), actual);
  for (const grant of [false, null, undefined, 'yes']) assert.equal(await match(key, campaign, credential, grant), null);
  await assert.rejects(match(key, campaign, null, true), /enrollment/);
});

test('simultaneous submissions coalesce and snapshots, public exports and audit revisions are isolated', async () => {
  const session = await regional.createSession();
  const first = session.nextBatch();
  const second = session.nextBatch();
  assert.equal(first, second);
  assert.equal(session.snapshot().completedBatches, 0);
  const startingAudit = session.audit();
  await Promise.all([first, second]);
  assert.deepEqual(await startingAudit, { valid: true, matched: 0 });
  assert.equal(session.snapshot().completedBatches, 1);
  const privateCopy = session.snapshot();
  privateCopy.journeys[0].stops.push({ label: 'Changed' });
  privateCopy.totals.views = 98;
  const publicCopy = session.proof();
  publicCopy.validators.network = '';
  publicCopy.batches.length = 0;
  assert.deepEqual(session.snapshot().totals, totals(6, 0, 0));
  assert.equal(session.snapshot().journeys[0].stops.length, 0);
  assert.equal(await isValid(session.proof()), true);
});

function browserEngine(crypto) {
  const context = vm.createContext({ TextEncoder, crypto });
  for (const moduleName of ['ad-verification-core', 'ad-verification-regional']) {
    const file = path.resolve(__dirname, '../../js/demos', moduleName + '.js');
    vm.runInContext(fs.readFileSync(file, 'utf8'), context, { filename: file });
  }
  return context.AdVerificationRegional;
}

test('missing Web Crypto fails closed and a transient signing failure can be retried without a partial batch', async () => {
  const unavailable = browserEngine(undefined);
  await assert.rejects(unavailable.createSession(), /Web Crypto/);
  assert.equal((await unavailable.verifyProof(proofs[3])).valid, false);
  let fail = false;
  const subtle = new Proxy(webcrypto.subtle, { get(target, method) {
    if (method === 'sign') return (...args) => {
      if (fail) {
        fail = false;
        throw new Error('Simulated signer outage.');
      }
      return target.sign(...args);
    };
    return target[method].bind(target);
  } });
  const simulated = browserEngine({ subtle, getRandomValues: (array) => webcrypto.getRandomValues(array) });
  const session = await simulated.createSession();
  fail = true;
  await assert.rejects(session.nextBatch(), /signer outage/);
  assert.equal(session.snapshot().completedBatches, 0);
  assert.equal(session.proof().batches.length, 0);
  await session.nextBatch();
  assert.equal(session.snapshot().completedBatches, 1);
  assert.equal((await session.audit()).valid, true);
});
