'use strict';
const assert = require('node:assert/strict');
const { test, before } = require('node:test');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
let core;
let session;
before(async () => {
  core = await import('../../js/demos/ad-verification-core.mjs');
  session = await core.createDemo();
});
const clone = () => structuredClone(session.chain);
const check = chain => core.verifyChain(chain, session.trust);

test('canonical serialization is independent of object key order', () => {
  assert.equal(core.canonical({ z: 3, a: { y: false, x: 'a' } }), core.canonical({ a: { x: 'a', y: false }, z: 3 }));
  for (const input of [undefined, NaN, Infinity, 2n, new Date(), { value: undefined }]) assert.throws(() => core.canonical(input));
  let deep = {};
  for (let index = 0; index < 15; index += 1) deep = { deep };
  assert.throws(() => core.canonical(deep));
});
test('fingerprint is genuine SHA-256, deterministic and sensitive to content', async () => {
  const data = { impressions: 12500 };
  assert.equal(await core.fingerprint(data), createHash('sha256').update(core.canonical(data)).digest('hex'));
  assert.notEqual(await core.fingerprint(data), await core.fingerprint({ impressions: 13000 }));
});
test('Merkle tree uses domain-separated leaves and duplicates odd nodes', async () => {
  const txs = [{ a: 1 }, { a: 2 }, { a: 3 }];
  assert.equal(await core.merkleRoot(txs), await core.merkleRoot([...txs, txs[2]]));
  assert.notEqual(await core.merkleRoot(txs), await core.merkleRoot(txs.slice().reverse()));
  await assert.rejects(core.merkleRoot([]));
  await assert.rejects(core.merkleRoot(Array(33).fill({ a: 1 })));
});
test('all four genuine signed blocks and local approvals verify', async () => {
  const result = await check(session.chain);
  assert.equal(result.valid, true);
  assert.equal(result.count, 4);
  for (const block of result.blocks) {
    for (const key of ['hashValid', 'signatureValid', 'approvalsValid', 'linkValid', 'sequenceValid', 'trusted']) assert.equal(block[key], true);
  }
});
test('only public CryptoKeys leave the builder; no private-key serialization', () => {
  assert.equal(Object.keys(session).join(','), 'chain,trust');
  for (const key of [...Object.values(session.trust.signers), ...Object.values(session.trust.validators)]) assert.equal(key.type, 'public');
  assert(!JSON.stringify(session).includes('privateKey'));
});
test('publisher tamper is detected without mutating baseline', async () => {
  const tampered = core.simulateChange(session.chain);
  assert.equal(session.chain[2].transactions[0].payload.details.impressions, 12500);
  assert.equal(tampered[2].transactions[0].payload.details.impressions, 13000);
  const result = await check(tampered);
  assert.equal(result.valid, false);
  assert.equal(result.count, 2);
  assert.equal(result.blocks[2].hashValid, false);
  assert.equal(result.blocks[2].signatureValid, false);
  assert.equal(result.blocks[2].approvalsValid, false);
  assert.equal(result.blocks[3].signatureValid, true, 'Downstream signature can still be internally valid.');
  assert.equal(result.blocks[3].linkValid, false);
  assert.equal(result.blocks[3].trusted, false, 'An intact downstream signature does not restore ancestry.');
  assert.equal((await check(clone())).valid, true, 'Restoring original signed records succeeds.');
});
test('recalculating all hashes cannot repair missing signatures', async () => {
  const tampered = core.simulateChange(session.chain);
  let previous = session.trust.genesisHash;
  for (const block of tampered) {
    block.header.previousHash = previous;
    block.header.merkleRoot = await core.merkleRoot(block.transactions);
    block.hash = await core.blockHash(block.header);
    previous = block.hash;
  }
  const result = await check(tampered);
  assert.equal(result.valid, false);
  assert.equal(result.blocks[2].hashValid, true);
  assert.equal(result.blocks[2].signatureValid, false);
  assert.equal(result.blocks[2].approvalsValid, false);
  assert(result.blocks[3].reasons.some(reason => reason.includes('checkpoint')));
});
test('changing the previous-block link is detected', async () => {
  const tampered = clone();
  tampered[1].header.previousHash = '0'.repeat(64);
  assert.equal((await check(tampered)).blocks[1].linkValid, false);
});
test('deleting, truncating, duplicating, appending or reordering blocks fails', async () => {
  for (const chain of [[], clone().slice(0, 3), clone().slice(1), [...clone(), clone()[0]], [clone()[0], clone()[0], ...clone().slice(2)], clone().reverse()]) {
    assert.equal((await check(chain)).valid, false);
  }
});
test('fewer than three approvals or duplicate/unknown validators fail', async () => {
  for (const mode of ['missing', 'duplicate', 'unknown', 'bad-signature']) {
    const tampered = clone();
    const approvals = tampered[1].approvals;
    if (mode === 'missing') approvals.pop();
    if (mode === 'duplicate') approvals[1] = structuredClone(approvals[0]);
    if (mode === 'unknown') approvals[1].validator = 'unknown';
    if (mode === 'bad-signature') approvals[1].signature = '00'.repeat(64);
    assert.equal((await check(tampered)).blocks[1].approvalsValid, false, mode);
  }
});
test('signer role and campaign are bound into the signature and sequence', async () => {
  for (const field of ['signer', 'campaignId', 'event', 'id']) {
    const tampered = clone();
    tampered[1].transactions[0].payload[field] = 'forged';
    const result = await check(tampered);
    assert.equal(result.blocks[1].signatureValid, false);
    assert.equal(result.blocks[1].sequenceValid, false);
  }
});
test('a fresh legitimate chain cannot replace another session checkpoint', async () => {
  const other = await core.createDemo();
  assert.notEqual(other.trust.sessionId, session.trust.sessionId);
  assert.equal((await check(other.chain)).valid, false);
});
test('invalid signature encoding and forged payload keys fail closed', async () => {
  const tampered = clone();
  tampered[0].transactions[0].signature = 'not-hex';
  tampered[0].transactions[0].publicKey = session.trust.signers.publisher;
  assert.equal((await check(tampered)).valid, false);
});
test('malformed and oversized input fails closed rather than showing verified', async () => {
  for (const input of [null, {}, [null], Array(4).fill({}), Array(4).fill({ header: {}, transactions: [] })]) assert.equal((await check(input)).valid, false);
  const oversized = clone();
  oversized[0].transactions[0].payload.details.extra = 'x'.repeat(70000);
  assert.equal((await check(oversized)).valid, false);
  await assert.rejects(core.verifyChain(clone(), null));
});
test('missing Web Crypto fails with a clear error, never a fake hash fallback', async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  try {
    Object.defineProperty(globalThis, 'crypto', { configurable: true, value: {} });
    await assert.rejects(core.createDemo(), /Web Crypto/);
    await assert.rejects(check(clone()), /Web Crypto/);
  } finally {
    Object.defineProperty(globalThis, 'crypto', descriptor);
  }
});
test('page is unlisted/noindex, fictional, dependency-free and network-isolated', () => {
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification-demo.html'), 'utf8');
  assert.match(html, /name="robots" content="noindex, nofollow"/);
  assert.match(html, /connect-src 'none'/);
  assert.match(html, /fictional Cedar Valley Tourism/);
  assert.match(html, /simulated events and validators/);
  assert.match(html, /class="skip-link"/);
  assert.match(html, /<main id="main">/);
  assert.match(html, /<dialog/);
  assert.doesNotMatch(html, /Visit Grand Junction|\bVGJ\b|google-analytics|googletagmanager/);
  for (const name of ['ad-verification-core.mjs', 'ad-verification-demo.mjs']) {
    const source = fs.readFileSync(path.join(root, 'js/demos', name), 'utf8');
    assert.doesNotMatch(source, /\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\b|\blocalStorage\b|\bsessionStorage\b|\.innerHTML\s*=/);
  }
});
test('built discovery outputs never surface the unlisted demo', context => {
  const files = ['sitemap.xml', 'dist/search-index.json', 'dist/chatbot-knowledge.json', 'dist/shortlinks-destinations.json', 'dist/app-content/v1/catalog.json', 'js/portfolio/projects-data.js'];
  const existing = files.filter(name => fs.existsSync(path.join(root, name)));
  if (!existing.length && process.env.AV_REQUIRE_BUILD !== '1') { context.skip('Full repository build is checked separately in CI.'); return; }
  for (const file of existing) assert.doesNotMatch(fs.readFileSync(path.join(root, file), 'utf8'), /ad-verification-demo|Cedar Valley Tourism/, file);
  if (process.env.AV_REQUIRE_BUILD === '1') {
    for (const name of files.slice(0, 5)) assert(fs.existsSync(path.join(root, name)), `Build output must exist: ${name}`);
    assert(fs.existsSync(path.join(root, 'public/demos/ad-verification-demo.html')));
    assert(fs.existsSync(path.join(root, 'public/js/demos/ad-verification-core.mjs')));
  }
});
