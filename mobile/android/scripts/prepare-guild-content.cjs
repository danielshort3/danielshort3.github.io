#!/usr/bin/env node
'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { bundle } = require('../../../build/bundle-wayfarers-android.cjs');

const ROOT = path.resolve(__dirname, '../../..');
const PACKAGE_NAME = 'me.danielshort.wayfarers';
const BUNDLED_CONTENT_VERSION = 3;
const MIN_APP_VERSION_CODE = 17;
const SAVE_SCHEMA = 8;
const MAX_ARCHIVE_BYTES = 32 * 1024 * 1024;
const MAX_FILE_BYTES = 8 * 1024 * 1024;
const MAX_FILES = 512;
const MAX_PAYLOAD_BYTES = 256 * 1024;
const MAX_ENVELOPE_BYTES = 512 * 1024;
const HASH = /^[a-f0-9]{64}$/;
const RESERVED_PATHS = new Set(['wayfarers/native-checkpoint.js', 'wayfarers/checkpoint.js', 'wayfarers/android.js', 'wayfarers/bundle-manifest.json']);
const BASE_RELEASE = 'https://github.com/danielshort3/danielshort3.github.io/releases/download/';
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function isWithin(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

function validatePath(value) {
  if (typeof value !== 'string' || value.length > 200 || RESERVED_PATHS.has(value) ||
      !(/^(?:wayfarers\/[A-Za-z0-9][A-Za-z0-9._-]*\.(?:html|js|css|json)|img\/wayfarers-guild\/[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|webp|jpg|jpeg|gif|json))$/.test(value)) ||
      (value.endsWith('.html') && value !== 'wayfarers/index.html')) {
    throw new Error(`Unknown or APK-owned content path: ${String(value)}`);
  }
  return value;
}

function validateAssetUrl(value) {
  if (typeof value !== 'string' || value.length > 8192 || !value.startsWith(BASE_RELEASE) || /[\\\u0000-\u0020\u007f-\u009f]/.test(value)) throw new Error('Invalid content release URL');
  const rawPath = value.slice('https://github.com'.length);
  if (/%2f|%5c|%25/i.test(rawPath)) throw new Error('Invalid encoded content URL path');
  const decoded = decodeURIComponent(rawPath);
  if (decoded.split('/').some(part => part === '.' || part === '..')) throw new Error('Invalid content URL path');
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.port || url.username || url.password || url.search || url.hash ||
      !url.pathname.startsWith('/danielshort3/danielshort3.github.io/releases/download/') ||
      !/^[A-Za-z0-9][A-Za-z0-9._-]{0,100}\/(?:[A-Za-z0-9][A-Za-z0-9._-]{0,180})?$/.test(url.pathname.slice('/danielshort3/danielshort3.github.io/releases/download/'.length))) throw new Error('Content assets must use the approved public GitHub HTTPS release URL');
  return url.href;
}

function validateManifest(manifest, { allowPrevious = false } = {}) {
  const current = manifest?.minAppVersionCode === MIN_APP_VERSION_CODE && manifest?.saveSchema === SAVE_SCHEMA && manifest?.contentVersion > BUNDLED_CONTENT_VERSION;
  const previous = allowPrevious && manifest?.minAppVersionCode === 16 && manifest?.saveSchema === 7 && manifest?.contentVersion >= 2;
  if (!manifest || Object.keys(manifest).length !== 9 || manifest.schemaVersion !== 1 || manifest.packageName !== PACKAGE_NAME ||
      !Number.isSafeInteger(manifest.contentVersion) || manifest.contentVersion < 2 || manifest.contentVersion > 2147483647 ||
      typeof manifest.label !== 'string' || !manifest.label.trim() || manifest.label.length > 80 || /[\u0000-\u001f\u007f-\u009f]/.test(manifest.label) ||
      manifest.nativeApi !== 1 || (!current && !previous)) throw new Error('Invalid or incompatible Guild content identity');
  if (!manifest.archive || Object.keys(manifest.archive).length !== 3 || !HASH.test(manifest.archive.sha256) || !Number.isSafeInteger(manifest.archive.size) ||
      manifest.archive.size < 1 || manifest.archive.size > MAX_ARCHIVE_BYTES) throw new Error('Invalid content archive');
  validateAssetUrl(manifest.archive.url);
  if (!Array.isArray(manifest.records) || manifest.records.length < 2 || manifest.records.length > MAX_FILES) throw new Error('Invalid content inventory');
  const seen = new Set();
  let total = 0;
  for (const record of manifest.records) {
    if (!record || Object.keys(record).length !== 3 || !HASH.test(record.sha256) || !Number.isSafeInteger(record.size) || record.size < 1 || record.size > MAX_FILE_BYTES) throw new Error('Invalid content file record');
    validatePath(record.path);
    if (seen.has(record.path) || seen.has(record.path.toLowerCase())) throw new Error('Duplicate content path');
    seen.add(record.path.toLowerCase());
    total += record.size;
    if (total > MAX_ARCHIVE_BYTES) throw new Error('Expanded content exceeds 32 MiB');
  }
  if (!seen.has('wayfarers/index.html') || !seen.has('wayfarers/game.css')) throw new Error('Content must include index.html and game.css');
  if (Buffer.byteLength(JSON.stringify(manifest), 'utf8') > MAX_PAYLOAD_BYTES) throw new Error('Content manifest exceeds 256 KiB');
  return manifest;
}

function parseArgs(argv) {
  const result = {};
  const known = new Set(['key', 'output', 'base-url', 'version', 'label', 'previous-envelope']);
  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i]?.replace(/^--/, '');
    if (!argv[i]?.startsWith('--') || !known.has(name) || !argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error('Usage: --key PRIVATE_PEM --output EXTERNAL_DIR --base-url HTTPS_RELEASE_URL/ --version INTEGER --label LABEL [--previous-envelope FILE]');
    if (result[name] !== undefined) throw new Error(`Duplicate --${name}`);
    result[name] = argv[i + 1];
  }
  for (const name of ['key', 'output', 'base-url', 'version', 'label']) if (!result[name]) throw new Error(`Missing --${name}`);
  if (!/^[1-9]\d*$/.test(result.version) || !Number.isSafeInteger(Number(result.version)) || Number(result.version) <= BUNDLED_CONTENT_VERSION || Number(result.version) > 2147483647) throw new Error(`Content version must be an Android integer above the bundled baseline ${BUNDLED_CONTENT_VERSION}`);
  result.version = Number(result.version);
  if (!result['previous-envelope']) throw new Error('Require --previous-envelope from the published signed release');
  if (!result['base-url'].endsWith('/')) throw new Error('--base-url must end with /');
  result['base-url'] = validateAssetUrl(result['base-url']);
  return result;
}

function loadSigningKey(filename) {
  const absolute = path.resolve(filename);
  if (isWithin(ROOT, absolute) || isWithin(ROOT, fs.realpathSync(absolute))) throw new Error('Keep the content signing private key outside the repository');
  const stat = fs.statSync(absolute);
  if (!stat.isFile() || stat.size < 1 || stat.size > 16 * 1024) throw new Error('Invalid content signing key file');
  const privateKey = crypto.createPrivateKey(fs.readFileSync(absolute));
  if (privateKey.asymmetricKeyType !== 'ec' || privateKey.asymmetricKeyDetails?.namedCurve !== 'prime256v1') throw new Error('Content signing requires an EC P-256 private key');
  const publicKey = crypto.createPublicKey(privateKey);
  const publicKeySpki = publicKey.export({ type: 'spki', format: 'der' }).toString('base64');
  return { privateKey, publicKey, publicKeySpki };
}

function decodeBase64(value, maximum, name) {
  if (typeof value !== 'string' || value.length < 1 || value.length > Math.ceil(maximum / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) throw new Error(`Invalid ${name} base64`);
  const decoded = Buffer.from(value, 'base64');
  if (decoded.length < 1 || decoded.length > maximum || decoded.toString('base64') !== value) throw new Error(`Invalid ${name} bytes`);
  return decoded;
}

function verifyEnvelope(bytes, publicKey, options = {}) {
  const encoded = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (encoded.length > MAX_ENVELOPE_BYTES) throw new Error('Signed content envelope exceeds 512 KiB');
  const envelope = JSON.parse(encoded.toString('utf8'));
  if (!envelope || Object.keys(envelope).length !== 2 || !Object.hasOwn(envelope, 'payload') || !Object.hasOwn(envelope, 'signature')) throw new Error('Invalid signed content envelope');
  const payload = decodeBase64(envelope.payload, MAX_PAYLOAD_BYTES, 'payload');
  const signature = decodeBase64(envelope.signature, 128, 'signature');
  if (!crypto.verify('sha256', payload, { key: publicKey, dsaEncoding: 'der' }, signature)) throw new Error('Content signature verification failed');
  const text = payload.toString('utf8');
  if (!Buffer.from(text, 'utf8').equals(payload)) throw new Error('Content manifest is not exact UTF-8');
  return { manifest: validateManifest(JSON.parse(text), options), payload, envelope };
}

function signManifest(manifest, privateKey) {
  validateManifest(manifest);
  const payload = Buffer.from(JSON.stringify(manifest), 'utf8');
  const signature = crypto.sign('sha256', payload, { key: privateKey, dsaEncoding: 'der' });
  const bytes = Buffer.from(`${JSON.stringify({ payload: payload.toString('base64'), signature: signature.toString('base64') }, null, 2)}\n`, 'utf8');
  verifyEnvelope(bytes, crypto.createPublicKey(privateKey));
  return bytes;
}

const CRC_TABLE = Array.from({ length: 256 }, (_, value) => {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) crc = (crc & 1) ? (0xedb88320 ^ (crc >>> 1)) : (crc >>> 1);
  return crc >>> 0;
});
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function writeStoreZip(files) {
  if (!(files instanceof Map) || files.size < 2 || files.size > MAX_FILES) throw new Error('Invalid ZIP file inventory');
  const local = [];
  const central = [];
  const records = [];
  const seen = new Set();
  let offset = 0;
  for (const filename of [...files.keys()].sort()) {
    validatePath(filename);
    if (seen.has(filename.toLowerCase())) throw new Error('Duplicate ZIP path');
    seen.add(filename.toLowerCase());
    const bytes = files.get(filename);
    if (!Buffer.isBuffer(bytes) || bytes.length < 1 || bytes.length > MAX_FILE_BYTES) throw new Error('Content files must be nonempty and no larger than 8 MiB');
    const name = Buffer.from(filename, 'utf8');
    const checksum = crc32(bytes);
    const header = Buffer.alloc(30);
    header.writeUInt32LE(0x04034b50, 0);
    header.writeUInt16LE(20, 4);
    header.writeUInt16LE(0x0800, 6);
    header.writeUInt16LE(0x0021, 12); // 1980-01-01, identical across release machines.
    header.writeUInt32LE(checksum, 14);
    header.writeUInt32LE(bytes.length, 18);
    header.writeUInt32LE(bytes.length, 22);
    header.writeUInt16LE(name.length, 26);
    local.push(header, name, bytes);
    const directory = Buffer.alloc(46);
    directory.writeUInt32LE(0x02014b50, 0);
    directory.writeUInt16LE(20, 4);
    directory.writeUInt16LE(20, 6);
    directory.writeUInt16LE(0x0800, 8);
    directory.writeUInt16LE(0x0021, 14);
    directory.writeUInt32LE(checksum, 16);
    directory.writeUInt32LE(bytes.length, 20);
    directory.writeUInt32LE(bytes.length, 24);
    directory.writeUInt16LE(name.length, 28);
    directory.writeUInt32LE(offset, 42);
    central.push(directory, name);
    offset += header.length + name.length + bytes.length;
    records.push({ path: filename, sha256: sha256(bytes), size: bytes.length });
    if (offset > MAX_ARCHIVE_BYTES) throw new Error('Content ZIP exceeds 32 MiB');
  }
  const directoryBytes = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.size, 8);
  end.writeUInt16LE(files.size, 10);
  end.writeUInt32LE(directoryBytes.length, 12);
  end.writeUInt32LE(offset, 16);
  const archive = Buffer.concat([...local, directoryBytes, end]);
  if (archive.length > MAX_ARCHIVE_BYTES) throw new Error('Content ZIP exceeds 32 MiB');
  return { archive, records };
}

function canonicalFiles(directory, bundleGame = bundle) {
  const sources = bundleGame(directory);
  if (!Array.isArray(sources) || sources.length < 2 || sources.length > MAX_FILES + RESERVED_PATHS.size) throw new Error('Invalid canonical bundle inventory');
  const files = new Map();
  for (const record of sources) {
    if (!record || typeof record.path !== 'string') throw new Error('Invalid canonical bundle record');
    if (RESERVED_PATHS.has(record.path)) continue;
    validatePath(record.path);
    if (files.has(record.path)) throw new Error('Duplicate canonical bundle path');
    const filename = path.join(directory, record.path);
    if (!isWithin(path.resolve(directory), fs.realpathSync(filename)) || !fs.lstatSync(filename).isFile()) throw new Error('Canonical bundle path must be a regular file inside staging');
    const stat = fs.statSync(filename);
    if (stat.size < 1 || stat.size > MAX_FILE_BYTES) throw new Error('Canonical file size is invalid');
    const bytes = fs.readFileSync(filename);
    if (!HASH.test(record.sha256) || sha256(bytes) !== record.sha256) throw new Error('Canonical bundle changed before signing');
    files.set(record.path, bytes);
  }
  return files;
}

function prepareBundle(options, bundleGame = bundle) {
  if (!Number.isSafeInteger(options.version) || options.version <= BUNDLED_CONTENT_VERSION || options.version > 2147483647) throw new Error(`Content version must be an Android integer above the bundled baseline ${BUNDLED_CONTENT_VERSION}`);
  if (!options['previous-envelope']) throw new Error('Require the previous published signed envelope');
  const baseUrl = validateAssetUrl(options['base-url']);
  if (!baseUrl.endsWith('/')) throw new Error('--base-url must end with /');
  const key = loadSigningKey(options.key);
  const staging = fs.mkdtempSync(path.join(os.tmpdir(), 'wayfarers-content-bundle-'));
  try {
    const { archive, records } = writeStoreZip(canonicalFiles(staging, bundleGame));
    const archiveHash = sha256(archive);
    const archiveName = `Wayfarers-content-v${options.version}-${archiveHash.slice(0, 16)}.zip`;
    const manifest = validateManifest({ schemaVersion: 1, packageName: PACKAGE_NAME, contentVersion: options.version, label: options.label,
      nativeApi: 1, minAppVersionCode: MIN_APP_VERSION_CODE, saveSchema: SAVE_SCHEMA,
      archive: { url: new URL(archiveName, baseUrl).href, sha256: archiveHash, size: archive.length }, records });
    if (options['previous-envelope']) {
      const filename = path.resolve(options['previous-envelope']);
      if (fs.statSync(filename).size > MAX_ENVELOPE_BYTES) throw new Error('Previous signed envelope is too large');
      const previous = verifyEnvelope(fs.readFileSync(filename), key.publicKey, { allowPrevious: true }).manifest;
      if (previous.contentVersion >= manifest.contentVersion) throw new Error('Content version must increase beyond the previously published signed release');
    }
    const envelope = signManifest(manifest, key.privateKey);
    const envelopeName = `content-v${manifest.contentVersion}-${archiveHash.slice(0, 16)}.json`;
    const artifacts = new Map([[archiveName, archive], [envelopeName, envelope], ['latest-content.json', envelope]]);
    artifacts.set('CONTENT-SHA256SUMS.txt', Buffer.from([...artifacts].map(([name, bytes]) => `${sha256(bytes)}  ${name}\n`).join(''), 'utf8'));
    return { manifest, artifacts, publicKeySpki: key.publicKeySpki };
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

function writeBundle(directory, artifacts) {
  const output = path.resolve(directory);
  if (isWithin(ROOT, output)) throw new Error('Stage release artifacts outside the repository');
  fs.mkdirSync(output, { recursive: true });
  if (isWithin(ROOT, fs.realpathSync(output))) throw new Error('Release staging must not resolve inside the repository');
  for (const [name, bytes] of artifacts) {
    if (typeof name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name) || path.basename(name) !== name || !Buffer.isBuffer(bytes)) throw new Error('Invalid content artifact filename');
    const destination = path.join(output, name);
    if (fs.existsSync(destination) && (!fs.lstatSync(destination).isFile() || !fs.readFileSync(destination).equals(bytes))) throw new Error(`Refusing to overwrite a different artifact: ${name}`);
  }
  for (const [name, bytes] of artifacts) {
    const destination = path.join(output, name);
    if (!fs.existsSync(destination)) fs.writeFileSync(destination, bytes, { flag: 'wx' });
  }
  return output;
}

if (require.main === module) {
  try {
    const options = parseArgs(process.argv.slice(2));
    const release = prepareBundle(options);
    const output = writeBundle(options.output, release.artifacts);
    process.stdout.write(`Prepared signed Guild content ${release.manifest.label} (${release.manifest.contentVersion})\n${release.manifest.records.length} verified files; ${release.manifest.archive.size} archive bytes\nPublic key SPKI: ${release.publicKeySpki}\nOutput: ${output}\nLocal staging only. No files were published.\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { PACKAGE_NAME, BUNDLED_CONTENT_VERSION, MIN_APP_VERSION_CODE, SAVE_SCHEMA, MAX_ARCHIVE_BYTES, MAX_FILE_BYTES, MAX_FILES, MAX_PAYLOAD_BYTES, MAX_ENVELOPE_BYTES, RESERVED_PATHS,
  sha256, crc32, validatePath, validateAssetUrl, validateManifest, parseArgs, loadSigningKey, signManifest, verifyEnvelope, writeStoreZip, canonicalFiles, prepareBundle, writeBundle };
