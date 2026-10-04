'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const content = require('./prepare-guild-content.cjs');

const BASE_URL = 'https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-content-v6/';
const sampleFiles = () => new Map([['wayfarers/index.html', Buffer.from('<!doctype html><title>Guild</title>')], ['wayfarers/game.css', Buffer.from('body{margin:0}')], ['wayfarers/numbers.js', Buffer.from('window.guild=1;')]]);
const keys = crypto.generateKeyPairSync('ec', { namedCurve: 'prime256v1' });

function sampleManifest(changes = {}) {
  const { archive, records } = content.writeStoreZip(sampleFiles());
  return { schemaVersion: 1, packageName: content.PACKAGE_NAME, contentVersion: 6, label: '0.14.0.1', nativeApi: 1, minAppVersionCode: 18, saveSchema: 8,
    archive: { url: `${BASE_URL}archive.zip`, sha256: content.sha256(archive), size: archive.length }, records, ...changes };
}

function signLegacyManifest(manifest) {
  const payload = Buffer.from(JSON.stringify(manifest), 'utf8');
  const signature = crypto.sign('sha256', payload, { key: keys.privateKey, dsaEncoding: 'der' });
  return Buffer.from(JSON.stringify({ payload: payload.toString('base64'), signature: signature.toString('base64') }));
}

function withTemporary(fn) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'guild-content-test-'));
  try { return fn(directory); } finally { fs.rmSync(directory, { recursive: true, force: true }); }
}

function fakeBundle(directory, files = sampleFiles()) {
  const records = [];
  for (const [filename, bytes] of files) {
    fs.mkdirSync(path.dirname(path.join(directory, filename)), { recursive: true });
    fs.writeFileSync(path.join(directory, filename), bytes);
    records.push({ path: filename, sha256: content.sha256(bytes), source: 'test' });
  }
  return records;
}

test('explicit CLI enforces positive version, required options and approved immutable release URLs', () => {
  assert.throws(() => content.validateAssetUrl(BASE_URL + 'nested/archive.zip'));
  const argv = ['--key', 'key.pem', '--output', 'release', '--base-url', BASE_URL, '--version', '6', '--label', '0.14.0.1', '--previous-envelope', 'prior.json'];
  assert.equal(content.parseArgs(argv).version, 6);
  for (const args of [argv.concat(['--wat', 'x']), argv.concat(['--version', '7']), argv.slice(0, 8), argv.slice(0, -2), argv.map(value => value === '6' ? '5' : value), argv.map(value => value === '6' ? '6.5' : value)]) assert.throws(() => content.parseArgs(args));
  for (const url of ['http://github.com/danielshort3/danielshort3.github.io/releases/download/v1/', 'https://github.com/other/repo/releases/download/v1/', `${BASE_URL}?q=1`, `${BASE_URL}#x`, `${BASE_URL}../x`, `${BASE_URL}%2e%2e/x`, `${BASE_URL}%2fx`, `${BASE_URL}%5cx`, `${BASE_URL}%25x`, BASE_URL.replace('github.com', 'github.com:443'), `${BASE_URL}a b`]) assert.throws(() => content.validateAssetUrl(url), url);
  assert.throws(() => content.parseArgs(argv.map(value => value === BASE_URL ? BASE_URL.slice(0, -1) : value)), /end with/);
});

test('inventory excludes APK-owned bridge scripts and unknown, traversal or ambiguous paths', () => {
  for (const filename of ['wayfarers/native-checkpoint.js', 'wayfarers/checkpoint.js', 'wayfarers/android.js', 'wayfarers/bundle-manifest.json', '../escape.js', 'wayfarers/../evil.js', 'wayfarers/foo/bar.js', 'wayfarers/other.html', 'assets/wayfarers/index.html', 'wayfarers/evil.JS', 'wayfarers/foo%2fbar.js', 'wayfarers/x\\y.js', 'img/other/a.png', `wayfarers/${'x'.repeat(200)}.js`]) assert.throws(() => content.validatePath(filename), filename);
  for (const filename of ['wayfarers/index.html', 'wayfarers/game.css', 'wayfarers/android.css', 'wayfarers/area-skills.js', 'img/wayfarers-guild/c-skill-mining.webp']) assert.equal(content.validatePath(filename), filename);
});

test('deterministic store ZIP uses UTF-8 regular files, fixed timestamps and exact CRC/lengths', () => {
  const files = sampleFiles();
  const first = content.writeStoreZip(files);
  assert.deepEqual(first.archive, content.writeStoreZip(new Map([...files].reverse())).archive);
  assert.equal(content.crc32(Buffer.from('123456789')), 0xcbf43926);
  let offset = 0;
  for (const record of first.records) {
    const data = first.archive;
    assert.equal(data.readUInt32LE(offset), 0x04034b50);
    assert.equal(data.readUInt16LE(offset + 6), 0x0800);
    assert.equal(data.readUInt16LE(offset + 8), 0);
    assert.equal(data.readUInt16LE(offset + 10), 0);
    assert.equal(data.readUInt16LE(offset + 12), 0x0021);
    assert.equal(data.readUInt32LE(offset + 14), content.crc32(files.get(record.path)));
    assert.equal(data.readUInt32LE(offset + 18), record.size);
    assert.equal(data.readUInt32LE(offset + 22), record.size);
    const nameLength = data.readUInt16LE(offset + 26);
    assert.equal(data.subarray(offset + 30, offset + 30 + nameLength).toString('utf8'), record.path);
    const bytes = data.subarray(offset + 30 + nameLength, offset + 30 + nameLength + record.size);
    assert.deepEqual(bytes, files.get(record.path));
    assert.equal(content.sha256(bytes), record.sha256);
    offset += 30 + nameLength + record.size;
  }
  assert.equal(first.archive.readUInt32LE(offset), 0x02014b50);
  assert.equal(first.archive.readUInt32LE(first.archive.length - 22), 0x06054b50);
  assert.equal(first.archive.readUInt16LE(first.archive.length - 14), files.size);
  assert.equal(first.archive.readUInt32LE(first.archive.length - 6), offset);
});

test('ZIP and manifest reject empty, excessive, duplicate or unsupported file inventories', () => {
  assert.throws(() => content.writeStoreZip(new Map()), /inventory/);
  assert.throws(() => content.writeStoreZip(new Map([...sampleFiles(), ['wayfarers/empty.js', Buffer.alloc(0)]])), /nonempty/);
  assert.throws(() => content.writeStoreZip(new Map([...sampleFiles(), ['wayfarers/large.js', Buffer.alloc(content.MAX_FILE_BYTES + 1)]])), /8 MiB/);
  assert.throws(() => content.writeStoreZip(new Map([...sampleFiles(), ['wayfarers/Game.css', Buffer.from('x')]])), /Duplicate/);
  assert.throws(() => content.writeStoreZip(new Map(Array.from({ length: content.MAX_FILES + 1 }, (_, index) => [`wayfarers/file${index}.js`, Buffer.from('x')]))), /inventory/);
  assert.throws(() => content.writeStoreZip(new Map(Array.from({ length: 4 }, (_, index) => [`wayfarers/file${index}.js`, Buffer.alloc(content.MAX_FILE_BYTES)]))), /32 MiB/);
  const manifest = sampleManifest();
  assert.throws(() => content.validateManifest({ ...manifest, records: [] }), /inventory/);
  assert.throws(() => content.validateManifest({ ...manifest, records: [manifest.records[0], manifest.records[0]] }), /Duplicate/);
  assert.throws(() => content.validateManifest({ ...manifest, records: manifest.records.filter(record => record.path !== 'wayfarers/index.html') }), /include/);
  for (const change of [{ extra: true }, { packageName: 'other.app' }, { schemaVersion: 2 }, { nativeApi: 2 }, { saveSchema: 7 }, { minAppVersionCode: 16 }, { minAppVersionCode: 17 }, { minAppVersionCode: 19 }, { label: '' }, { label: 'x\n' }, { contentVersion: 5 }, { contentVersion: 6.5 }, { contentVersion: 2147483648 }]) assert.throws(() => content.validateManifest({ ...manifest, ...change }), /identity/);
  assert.throws(() => content.validateManifest({ ...manifest, archive: { ...manifest.archive, sha256: 'x' } }), /archive/);
});

test('ECDSA DER signature verifies exact UTF-8 payload; tampering and other public keys fail', () => {
  const manifest = sampleManifest({ label: 'Guild – 0.13.0.1' });
  const signed = content.signManifest(manifest, keys.privateKey);
  const verified = content.verifyEnvelope(signed, keys.publicKey);
  assert.deepEqual(verified.manifest, manifest);
  assert.equal(verified.payload.toString('utf8'), JSON.stringify(manifest));
  const otherKey = crypto.generateKeyPairSync('ec', { namedCurve: 'prime256v1' });
  assert.throws(() => content.verifyEnvelope(signed, otherKey.publicKey), /signature/);
  const envelope = JSON.parse(signed);
  const changed = Buffer.from(envelope.payload, 'base64');
  changed[10] ^= 1;
  assert.throws(() => content.verifyEnvelope(Buffer.from(JSON.stringify({ ...envelope, payload: changed.toString('base64') })), keys.publicKey), /signature/);
  const signature = Buffer.from(envelope.signature, 'base64');
  signature[signature.length - 1] ^= 1;
  assert.throws(() => content.verifyEnvelope(Buffer.from(JSON.stringify({ ...envelope, signature: signature.toString('base64') })), keys.publicKey), /signature/);
  assert.throws(() => content.verifyEnvelope(Buffer.from(JSON.stringify({ ...envelope, extra: 1 })), keys.publicKey), /envelope/);
  assert.throws(() => content.verifyEnvelope(Buffer.from(JSON.stringify({ ...envelope, payload: `${envelope.payload}\n` })), keys.publicKey), /base64/);
  assert.throws(() => content.verifyEnvelope(Buffer.alloc(content.MAX_ENVELOPE_BYTES + 1), keys.publicKey), /512 KiB/);
});

test('canonical snapshot validates source hash and rejects changed, empty or unknown files', () => {
  withTemporary(directory => {
    const files = new Map([...sampleFiles(), ['wayfarers/android.js', Buffer.from('native')]]);
    const captured = content.canonicalFiles(directory, destination => fakeBundle(destination, files));
    assert.equal(captured.size, 3);
    assert.equal(captured.has('wayfarers/android.js'), false);
    assert.throws(() => content.canonicalFiles(directory, destination => {
      const records = fakeBundle(destination);
      fs.writeFileSync(path.join(destination, records[0].path), 'changed');
      return records;
    }), /changed/);
    assert.throws(() => content.canonicalFiles(directory, destination => fakeBundle(destination, new Map([...sampleFiles(), ['other/evil.js', Buffer.from('x')]]))), /Unknown/);
    assert.throws(() => content.canonicalFiles(directory, destination => fakeBundle(destination, new Map([...sampleFiles(), ['wayfarers/empty.js', Buffer.alloc(0)]]))), /size/);
  });
});

test('builder signs canonical complete content, preserves prior-version monotonicity and never publishes', () => {
  withTemporary(directory => {
    const key = path.join(directory, 'key.pem');
    fs.writeFileSync(key, keys.privateKey.export({ type: 'pkcs8', format: 'pem' }));
    const legacy = path.join(directory, 'legacy.json');
    fs.writeFileSync(legacy, signLegacyManifest(sampleManifest({ contentVersion: 2, label: '0.12.0.1', minAppVersionCode: 16, saveSchema: 7 })));
    assert.throws(() => content.signManifest(sampleManifest({ contentVersion: 2, minAppVersionCode: 16, saveSchema: 7 }), keys.privateKey), /identity/);
    assert.throws(() => content.verifyEnvelope(fs.readFileSync(legacy), keys.publicKey), /identity/);
    const options = { key, version: 6, label: '0.14.0.1', 'base-url': BASE_URL, 'previous-envelope': legacy };
    const release = content.prepareBundle(options, fakeBundle);
    assert.equal(release.publicKeySpki, keys.publicKey.export({ type: 'spki', format: 'der' }).toString('base64'));
    assert.equal(release.manifest.records.length, sampleFiles().size);
    assert.equal(release.manifest.minAppVersionCode, 18);
    assert.equal(release.manifest.saveSchema, 8);
    assert.equal(release.manifest.nativeApi, 1);
    assert.throws(() => content.prepareBundle({ ...options, version: 5 }, fakeBundle), /baseline/);
    assert.throws(() => content.prepareBundle({ ...options, 'previous-envelope': undefined }, fakeBundle), /previous/);
    const incompatiblePrior = path.join(directory, 'incompatible-prior.json');
    fs.writeFileSync(incompatiblePrior, signLegacyManifest(sampleManifest({ contentVersion: 2, minAppVersionCode: 15, saveSchema: 6 })));
    assert.throws(() => content.prepareBundle({ ...options, 'previous-envelope': incompatiblePrior }, fakeBundle), /identity/);
    const signed = release.artifacts.get('latest-content.json');
    assert.deepEqual(content.verifyEnvelope(signed, keys.publicKey).manifest, release.manifest);
    const archive = release.artifacts.get(new URL(release.manifest.archive.url).pathname.split('/').at(-1));
    assert.equal(content.sha256(archive), release.manifest.archive.sha256);
    assert.equal(archive.length, release.manifest.archive.size);
    const prior = path.join(directory, 'prior.json');
    fs.writeFileSync(prior, signed);
    assert.throws(() => content.prepareBundle({ ...options, 'previous-envelope': prior }, fakeBundle), /increase/);
    assert.equal(content.prepareBundle({ ...options, version: 7, 'previous-envelope': prior }, fakeBundle).manifest.contentVersion, 7);
    const releasedSeventeen = path.join(directory, 'released-seventeen.json');
    fs.writeFileSync(releasedSeventeen, signLegacyManifest(sampleManifest({ contentVersion: 4, label: '0.13.0.1', minAppVersionCode: 17, saveSchema: 8 })));
    assert.throws(() => content.verifyEnvelope(fs.readFileSync(releasedSeventeen), keys.publicKey), /identity/);
    assert.equal(content.prepareBundle({ ...options, 'previous-envelope': releasedSeventeen }, fakeBundle).manifest.contentVersion, 6);
    const output = path.join(directory, 'release');
    content.writeBundle(output, release.artifacts);
    content.writeBundle(output, release.artifacts);
    const original = fs.readFileSync(path.join(output, 'latest-content.json'));
    assert.throws(() => content.writeBundle(output, new Map([['new.json', Buffer.from('new')], ['latest-content.json', Buffer.from('changed')]])), /overwrite/);
    assert.equal(fs.existsSync(path.join(output, 'new.json')), false);
    assert.deepEqual(fs.readFileSync(path.join(output, 'latest-content.json')), original);
    assert.throws(() => content.writeBundle(output, new Map([['../outside', Buffer.from('x')]])), /filename/);
    assert.throws(() => content.writeBundle(path.resolve(__dirname, 'release-output'), release.artifacts), /outside/);
    const other = crypto.generateKeyPairSync('rsa', { modulusLength: 2048 });
    fs.writeFileSync(key, other.privateKey.export({ type: 'pkcs8', format: 'pem' }));
    assert.throws(() => content.loadSigningKey(key), /P-256/);
  });
});

test('shared fixture proves Java-compatible SPKI, DER signature and exact stored ZIP bytes', () => {
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, 'guild-content-protocol-fixture.json'), 'utf8'));
  const key = crypto.createPublicKey({ key: Buffer.from(fixture.publicKeySpkiBase64, 'base64'), format: 'der', type: 'spki' });
  const signed = Buffer.from(fixture.envelopeBase64, 'base64');
  assert.throws(() => content.verifyEnvelope(signed, key), /identity/);
  const manifest = content.verifyEnvelope(signed, key, { allowPrevious: true }).manifest;
  const archive = Buffer.from(fixture.archiveBase64, 'base64');
  assert.equal(content.sha256(archive), manifest.archive.sha256);
  assert.equal(archive.length, manifest.archive.size);
  assert.deepEqual(archive, content.writeStoreZip(new Map(fixture.files.map(file => [file.path, Buffer.from(file.bytesBase64, 'base64')]))).archive);
});
