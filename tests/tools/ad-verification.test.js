const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../../js/demos/ad-verification-core.js');
const root = path.resolve(__dirname, '../..');
let demo;
test.before(async () => { demo = await core.createDemo(); });

async function check(change) {
  const blocks = core.clone(demo.blocks);
  await change(blocks);
  return core.verifyChain(blocks, demo.trust);
}

test('SHA-256 matches a known test vector', async () => {
  assert.equal(await core.hash('abc'), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
});
test('canonical serialization is key-order independent and rejects non-JSON values', () => {
  assert.equal(core.canonical({ z: 1, a: { y: true, x: 'ok' } }), core.canonical({ a: { x: 'ok', y: true }, z: 1 }));
  assert.throws(() => core.canonical({ x: Infinity }));
  assert.throws(() => core.canonical({ x: undefined }));
});
test('all four original blocks and their approvals verify', async () => {
  const result = await core.verifyChain(demo.blocks, demo.trust);
  assert.equal(result.valid, true);
  assert.equal(result.blocks.length, 4);
  assert.ok(result.blocks.every((row) => Object.values(row.checks).every(Boolean)));
});
test('no private keys are returned or serialized', () => {
  assert.deepEqual(Object.keys(demo).sort(), ['blocks', 'trust']);
  assert.equal(Object.keys(demo.trust.publicKeys).length, 7);
  for (const key of Object.values(demo.trust.publicKeys)) assert.equal(key.d, undefined);
  assert.ok(!JSON.stringify(demo).includes('privateKey'));
});
test('publisher edit is detected and the next block is marked dependent', async () => {
  const result = await core.verifyChain(core.simulateChange(demo.blocks), demo.trust);
  assert.equal(result.valid, false);
  assert.deepEqual(result.blocks.map((row) => row.state), ['verified', 'verified', 'changed', 'dependent']);
  assert.equal(result.blocks[2].checks.eventSignature, false);
  assert.equal(result.blocks[2].checks.merkleRoot, false);
  assert.notEqual(result.blocks[2].computedHash, demo.blocks[2].hash);
  assert.equal(demo.blocks[2].transactions[0].event.data.impressions, 10000);
});
test('restoration uses the same original signed bytes and verifies', async () => {
  await core.verifyChain(core.simulateChange(demo.blocks), demo.trust);
  const restored = core.clone(demo.blocks);
  assert.equal((await core.verifyChain(restored, demo.trust)).valid, true);
  assert.equal(restored[2].transactions[0].signature, demo.blocks[2].transactions[0].signature);
});
test('corrupted event signature fails', async () => {
  const result = await check((blocks) => { blocks[1].transactions[0].signature = '00'.repeat(64); });
  assert.equal(result.valid, false);
  assert.equal(result.blocks[1].checks.eventSignature, false);
});
test('a fake validator signature fails', async () => {
  const result = await check((blocks) => { blocks[0].approvals[0].signature = '00'.repeat(64); });
  assert.equal(result.valid, false);
  assert.equal(result.blocks[0].checks.approvals, false);
});
test('duplicate approvals cannot satisfy the approval policy', async () => {
  const result = await check((blocks) => { blocks[0].approvals[1] = core.clone(blocks[0].approvals[0]); });
  assert.equal(result.blocks[0].checks.approvals, false);
  assert.equal(result.valid, false);
});
test('missing or unknown validator approvals fail closed', async () => {
  assert.equal((await check((blocks) => blocks[0].approvals.pop())).valid, false);
  assert.equal((await check((blocks) => { blocks[0].approvals[0].validator = 'intruder'; })).valid, false);
});
test('recalculating all hashes does not recreate signatures or the original checkpoint', async () => {
  const result = await check(async (blocks) => {
    blocks[2].transactions[0].event.data.impressions = 12500;
    for (let index = 0; index < blocks.length; index += 1) {
      blocks[index].header.merkleRoot = await core.merkleRoot(blocks[index].transactions);
      if (index) blocks[index].header.previousHash = blocks[index - 1].hash;
      blocks[index].hash = await core.headerHash(blocks[index].header);
    }
  });
  assert.equal(result.valid, false);
  assert.equal(result.blocks[2].checks.merkleRoot, true);
  assert.equal(result.blocks[2].checks.eventSignature, false);
  assert.equal(result.blocks[2].checks.approvals, false);
  assert.equal(result.headValid, false);
});
test('truncation is detected against the original length and head checkpoint', async () => {
  const result = await check((blocks) => blocks.pop());
  assert.equal(result.valid, false);
  assert.equal(result.lengthValid, false);
  assert.equal(result.headValid, false);
});
test('empty, reordered, deleted and duplicated blocks fail', async () => {
  assert.equal((await core.verifyChain([], demo.trust)).valid, false);
  assert.equal((await check((blocks) => blocks.reverse())).valid, false);
  assert.equal((await check((blocks) => blocks.splice(1, 1))).valid, false);
  assert.equal((await check((blocks) => blocks.push(core.clone(blocks[3])))).valid, false);
});
test('records from a different session cannot be replayed into this chain', async () => {
  const other = await core.createDemo();
  const result = await check((blocks) => { blocks[0] = other.blocks[0]; });
  assert.equal(result.valid, false);
  assert.equal(result.blocks[0].checks.structure, false);
});
test('changing the public-key registry invalidates the original signatures', async () => {
  const other = await core.createDemo();
  const trust = core.clone(demo.trust);
  trust.publicKeys.publisher = other.trust.publicKeys.publisher;
  assert.equal((await core.verifyChain(demo.blocks, trust)).valid, false);
});
test('malformed input never produces a valid result', async () => {
  for (const value of [null, {}, [null], [{ header: {} }]]) assert.equal((await core.verifyChain(value, demo.trust)).valid, false);
  assert.equal((await core.verifyChain(demo.blocks, {})).valid, false);
  assert.equal((await check((blocks) => { blocks[0].extra = 'unsigned metadata'; })).valid, false);
});
test('transaction order changes the Merkle root', async () => {
  const a = demo.blocks[0].transactions[0];
  const b = demo.blocks[1].transactions[0];
  assert.notEqual(await core.merkleRoot([a, b, a]), await core.merkleRoot([b, a, a]));
  assert.equal(await core.merkleRoot([a]), demo.blocks[0].header.merkleRoot);
});
test('preview is unlisted, noindex and uses no real destination or vendor data', () => {
  const project = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/adVerification.json'), 'utf8'));
  assert.equal(project.published, false);
  assert.equal(project.hidden, true);
  assert.equal(project.noindex, true);
  const html = fs.readFileSync(path.join(root, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /name="robots" content="noindex, nofollow, noarchive"/);
  assert.match(html, /Cedar Valley Tourism/);
  assert.match(html, /class="av-skip skip-link"/);
  assert.match(html, /<main id="main"/);
  assert.doesNotMatch(html, /Visit Grand Junction|Foursquare|Viant|Cadent|gtag|googletagmanager/);
});
test('runtime does not send campaign data or retain signing keys in storage', () => {
  for (const filename of ['ad-verification.js', 'ad-verification-core.js']) {
    const source = fs.readFileSync(path.join(root, 'js/demos', filename), 'utf8');
    assert.doesNotMatch(source, /\bfetch\s*\(|XMLHttpRequest|WebSocket|sendBeacon|localStorage|sessionStorage/);
  }
});
test('generated public assets retain noindex and stay out of discovery indexes when present', (t) => {
  const publicRoot = path.join(root, 'public');
  if (!fs.existsSync(publicRoot)) { t.skip('Run npm run build to check publication artifacts.'); return; }
  const html = fs.readFileSync(path.join(publicRoot, 'demos/ad-verification.html'), 'utf8');
  assert.match(html, /name="robots"[^>]*noindex/);
  for (const file of ['js/demos/ad-verification-core.js', 'js/demos/ad-verification.js', 'css/components/ad-verification.css', 'img/projects/ad-verification-landscape.webp']) {
    assert.ok(fs.statSync(path.join(publicRoot, file)).size > 0, file);
  }
  for (const file of ['sitemap.xml', 'dist/search-index.json', 'app-content/v1/catalog.json']) {
    const target = path.join(publicRoot, file);
    if (fs.existsSync(target)) assert.doesNotMatch(fs.readFileSync(target, 'utf8'), /adVerification|demos\/ad-verification|Cedar Valley/);
  }
});
