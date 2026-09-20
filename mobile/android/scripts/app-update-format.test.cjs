'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const zlib = require('node:zlib');
const { encodePatch, applyPatch, sha256, MAX_APK_BYTES, MAX_OPERATIONS } = require('./app-update-format.cjs');
const { parseArgs, validateAssetUrl, validateManifest, prepareBundle, writeBundle } = require('./prepare-app-update.cjs');

const gzip = bytes => zlib.gzipSync(bytes, { level: 9, mtime: 0 });
function modifiedPatch(base, target, mutate) {
  const data = zlib.gunzipSync(encodePatch(base, target, { blockBytes: 4 }));
  return gzip(mutate(data) || data);
}

test('round trips insertions, deletions, moves and changed bytes at random offsets', () => {
  for (let i = 0; i < 25; i += 1) {
    const base = crypto.randomBytes(90000 + i * 7);
    const target = Buffer.concat([base.subarray(25000, 45000), crypto.randomBytes(i + 1), base.subarray(0, 20000), base.subarray(47000)]);
    const patch = encodePatch(base, target, { blockBytes: 1024 });
    assert.deepEqual(applyPatch(base, patch), target);
    assert.ok(patch.length < target.length / 3);
  }
});

test('default block matching finds bytes after an unaligned insertion', () => {
  const base = crypto.randomBytes(4 * 32768);
  const target = Buffer.concat([Buffer.from('inserted'), base]);
  const patch = encodePatch(base, target);
  assert.ok(patch.length < 512);
  assert.deepEqual(applyPatch(base, patch), target);
});

test('small and entirely different inputs use bounded literals', () => {
  for (const [base, target] of [['a', 'b'], ['old text', 'entirely new content'], ['abc', 'abc']]) {
    assert.deepEqual(applyPatch(Buffer.from(base), encodePatch(Buffer.from(base), Buffer.from(target))), Buffer.from(target));
  }
});

test('gzip output is deterministic and fixture uses exact cross-platform bytes', () => {
  const base = Buffer.from('abcdefghijklmno');
  const target = Buffer.from('abcdINSERTefghijklmno!');
  const patch = encodePatch(base, target, { blockBytes: 4 });
  assert.deepEqual(patch, encodePatch(base, target, { blockBytes: 4 }));
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'app-update-protocol-fixture.json'), 'utf8'));
  assert.equal(base.toString('base64'), fixture.baseBase64);
  assert.equal(target.toString('base64'), fixture.targetBase64);
  assert.equal(zlib.gunzipSync(patch).toString('hex'), fixture.uncompressedPatchHex);
  assert.deepEqual(applyPatch(base, Buffer.from(fixture.patchBase64, 'base64')), target);
});

test('rejects modified base and malformed or corrupt gzip', () => {
  const patch = encodePatch(Buffer.from('original'), Buffer.from('target'));
  assert.throws(() => applyPatch(Buffer.from('modified'), patch), /Base APK/);
  assert.throws(() => applyPatch(Buffer.from('original'), patch.subarray(0, patch.length - 3)));
  const corrupt = Buffer.from(patch);
  corrupt[corrupt.length - 5] ^= 128;
  assert.throws(() => applyPatch(Buffer.from('original'), corrupt));
});

test('rejects invalid magic, hashes, sizes, operations, truncation and trailing data', () => {
  const base = Buffer.from('abcdefghijklmnop');
  const target = Buffer.from('abcdefghijklmnop');
  const checks = [
    data => { data[0] = 0; },
    data => { data[8] ^= 1; },
    data => { data[40] ^= 1; },
    data => { data.writeBigInt64BE(-1n, 72); },
    data => { data.writeBigInt64BE(BigInt(MAX_APK_BYTES) + 1n, 80); },
    data => { data.writeBigInt64BE(0n, 80); },
    data => { data[88] = 6; },
    data => { data.writeBigInt64BE(-1n, 89); },
    data => { data.writeBigInt64BE(1000n, 89); },
    data => { data.writeUInt32BE(0, 97); },
    data => { data.writeUInt32BE(0xffffffff, 97); },
    data => data.subarray(0, 99),
    data => Buffer.concat([data, Buffer.from([0])])
  ];
  for (const mutate of checks) assert.throws(() => applyPatch(base, modifiedPatch(base, target, mutate)));
});

test('rejects missing output and more than 100000 operations', () => {
  const base = Buffer.from('a');
  const target = Buffer.alloc(MAX_OPERATIONS + 1, 'a');
  const original = zlib.gunzipSync(encodePatch(base, target));
  assert.throws(() => applyPatch(base, gzip(Buffer.concat([original.subarray(0, 88), Buffer.from([255])]))), /Target APK/);
  const literal = Buffer.from([1, 0, 0, 0, 1, 97]);
  assert.throws(() => applyPatch(base, gzip(Buffer.concat([original.subarray(0, 88), ...Array(MAX_OPERATIONS + 1).fill(literal), Buffer.from([255])]))), /too many/);
});

test('validates supported HTTPS destinations and explicit CLI arguments', () => {
  assert.equal(validateAssetUrl('https://www.danielshort.me/app-updates/'), 'https://www.danielshort.me/app-updates/');
  for (const value of ['http://danielshort.me/a', 'https://other.example/a', 'https://github.com/other/repo/releases/download/v1/', 'https://danielshort.me/not-updates/a', 'https://user@danielshort.me/app-updates/a', 'https://danielshort.me/app-updates/a?q=1', 'https://danielshort.me:444/app-updates/a', 'https://danielshort.me/app-updates/x/../y', 'https://danielshort.me/app-updates/%2e%2e/y', 'https://danielshort.me/app-updates/%2fy', 'https://danielshort.me/app-updates/%5cy', 'https://danielshort.me/app-updates/%25y', `https://danielshort.me/app-updates/${'x'.repeat(8192)}`]) {
    assert.throws(() => validateAssetUrl(value));
  }
  assert.throws(() => parseArgs(['--wat', 'x']));
  assert.throws(() => parseArgs(['--apk', 'one', '--apk', 'two']));
});

const signer = 'a'.repeat(64);
function fakeApk(bytes, versionCode, overrides = {}) {
  return { bytes, versionCode, versionName: `0.${versionCode}.0`, minSdk: 26, packageName: 'me.danielshort.app.debug', sha256: sha256(bytes), size: bytes.length, signerSha256: signer, ...overrides };
}
function bundleOptions() {
  return { apk: 'target', base: ['base'], channel: 'review', 'base-url': 'https://github.com/danielshort3/danielshort3.github.io/releases/download/android-v0.3.0-review/' };
}

test('prepares verifiable immutable assets and deduplicates approved bases', () => {
  const base = fakeApk(crypto.randomBytes(100000), 2);
  const target = fakeApk(Buffer.concat([Buffer.from('change'), base.bytes]), 3);
  const options = { ...bundleOptions(), base: ['base', 'base'] };
  const { manifest, artifacts } = prepareBundle(options, name => name === 'target' ? target : base);
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.releases.length, 2);
  assert.equal(manifest.patches.length, 1);
  const delta = artifacts.get(new URL(manifest.patches[0].url).pathname.split('/').at(-1));
  assert.deepEqual(applyPatch(base.bytes, delta), target.bytes);
  assert.equal(sha256(delta), manifest.patches[0].sha256);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'app-update-test-'));
  try {
    writeBundle(directory, artifacts);
    writeBundle(directory, artifacts);
    assert.throws(() => writeBundle(directory, new Map([['latest.json', Buffer.from('different')]])), /overwrite/);
    assert.throws(() => writeBundle(directory, new Map([['../escape', Buffer.from('x')]])), /filename/);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('rejects incorrect package, signer and non-increasing versions', () => {
  const base = fakeApk(Buffer.from('old'), 2);
  const target = fakeApk(Buffer.from('new'), 3);
  for (const change of [{ packageName: 'other.app' }, { signerSha256: 'b'.repeat(64) }, { versionCode: 3 }]) {
    assert.throws(() => prepareBundle(bundleOptions(), name => name === 'target' ? target : { ...base, ...change }));
  }
});

test('retains historical known hashes without assuming version codes identify bytes', () => {
  const base = fakeApk(crypto.randomBytes(100000), 2);
  const other = fakeApk(crypto.randomBytes(100000), 2);
  const target = fakeApk(Buffer.concat([base.bytes, Buffer.from('new')]), 3);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'app-update-history-'));
  const previous = path.join(directory, 'previous.json');
  const previousFeed = { schemaVersion: 1, channel: 'review', packageName: base.packageName, latest: { versionCode: 2, versionName: '0.2.0', minSdk: 26, signerSha256: signer, apk: { url: 'https://www.danielshort.me/app-updates/review/base.apk', sha256: base.sha256, size: base.size } }, releases: [base, other].map(({ versionCode, sha256: hash, size, signerSha256 }) => ({ versionCode, sha256: hash, size, signerSha256 })), patches: [] };
  try {
    fs.writeFileSync(previous, JSON.stringify(previousFeed));
    const { manifest } = prepareBundle({ ...bundleOptions(), 'previous-manifest': previous }, name => name === 'target' ? target : base);
    assert.equal(manifest.releases.length, 3);
    assert.equal(manifest.releases.filter(item => item.versionCode === 2).length, 2);
    previousFeed.releases[0].signerSha256 = 'b'.repeat(64);
    fs.writeFileSync(previous, JSON.stringify(previousFeed));
    assert.throws(() => prepareBundle({ ...bundleOptions(), 'previous-manifest': previous }, name => name === 'target' ? target : base));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('manifest validation rejects unsupported schema, channel, bounds and inconsistent patch links', () => {
  const base = fakeApk(crypto.randomBytes(100000), 2);
  const target = fakeApk(Buffer.concat([base.bytes, Buffer.from('new')]), 3);
  const { manifest } = prepareBundle(bundleOptions(), name => name === 'target' ? target : base);
  assert.equal(validateManifest(manifest), manifest);
  const mutations = [
    value => { value.schemaVersion = 2; },
    value => { value.channel = 'other'; },
    value => { value.latest.apk.size += 1; },
    value => { value.latest.apk.url = 'https://evil.example/a.apk'; },
    value => { value.latest.signerSha256 = 'b'.repeat(64); },
    value => { value.latest.versionName = 'x'.repeat(81); },
    value => { value.latest.versionName = '0.3.0\n'; },
    value => { value.latest.versionCode = 2; },
    value => { value.releases = []; },
    value => { value.releases.push(value.releases[0]); },
    value => { value.releases = Array(201).fill(value.releases[0]); },
    value => { value.patches[0].toSha256 = base.sha256; },
    value => { value.patches[0].fromSha256 = target.sha256; },
    value => { value.patches[0].size = target.size; },
    value => { value.patches[0].format = 'other'; },
    value => { value.patches.push(value.patches[0]); },
    value => { value.patches = Array(201).fill(value.patches[0]); },
    value => { value.extra = 'x'.repeat(512 * 1024); }
  ];
  for (const mutate of mutations) {
    const copy = structuredClone(manifest);
    mutate(copy);
    assert.throws(() => validateManifest(copy));
  }
});
