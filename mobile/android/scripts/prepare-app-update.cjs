#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { MAX_APK_BYTES, sha256, encodePatch, applyPatch } = require('./app-update-format.cjs');

const HASH = /^[a-f0-9]{64}$/;
const CHANNEL_PACKAGES = { review: 'me.danielshort.app.debug', stable: 'me.danielshort.app' };

function validateAssetUrl(value) {
  if (typeof value !== 'string' || value.length > 8192 || !value.startsWith('https://') || /[\\\u0000-\u0020\u007f-\u009f]/.test(value)) throw new Error('Invalid update URL');
  const rawPath = value.match(/^https:\/\/[^/?#]+([^?#]*)/)?.[1] || '';
  const decodedPath = decodeURIComponent(rawPath);
  if (!decodedPath.startsWith('/') || decodedPath.includes('\\') || decodedPath.split('/').some(part => part === '.' || part === '..') || /%2f|%5c|%25/i.test(rawPath)) throw new Error('Invalid update URL path');
  const url = new URL(value);
  const allowed = (['danielshort.me', 'www.danielshort.me'].includes(url.hostname) && url.pathname.startsWith('/app-updates/')) ||
    (url.hostname === 'github.com' && url.pathname.startsWith('/danielshort3/danielshort3.github.io/releases/download/'));
  if (url.protocol !== 'https:' || url.port || url.username || url.password || url.hash || url.search || !allowed) {
    throw new Error('Assets must use an approved public HTTPS release URL');
  }
  return url.href;
}

function parseArgs(argv) {
  const result = { base: [] };
  const known = new Set(['apk', 'base', 'previous-manifest', 'output', 'base-url', 'channel']);
  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i]?.replace(/^--/, '');
    if (!argv[i]?.startsWith('--') || !known.has(name) || !argv[i + 1] || argv[i + 1].startsWith('--')) {
      throw new Error('Usage: --apk FILE [--base FILE ...] [--previous-manifest FILE] --output DIR --base-url HTTPS_URL --channel review|stable');
    }
    if (name === 'base') result.base.push(argv[i + 1]);
    else if (result[name]) throw new Error(`Duplicate --${name}`);
    else result[name] = argv[i + 1];
  }
  for (const name of ['apk', 'output', 'base-url', 'channel']) if (!result[name]) throw new Error(`Missing --${name}`);
  if (!CHANNEL_PACKAGES[result.channel]) throw new Error('Channel must be review or stable');
  result['base-url'] = validateAssetUrl(result['base-url']);
  if (!result['base-url'].endsWith('/')) throw new Error('--base-url must end with /');
  return result;
}

function sdkTools(environment = process.env) {
  const sdk = environment.ANDROID_HOME || environment.ANDROID_SDK_ROOT || path.join(os.homedir(), 'AppData', 'Local', 'Android', 'Sdk');
  const tools = path.join(sdk, 'build-tools', '36.1.0');
  const executable = process.platform === 'win32' ? '.exe' : '';
  const java = environment.JAVA_HOME ? path.join(environment.JAVA_HOME, 'bin', `java${executable}`) : `java${executable}`;
  const aapt = path.join(tools, `aapt${executable}`);
  const signer = path.join(tools, 'lib', 'apksigner.jar');
  if (!fs.existsSync(aapt) || !fs.existsSync(signer)) throw new Error('Android SDK Build Tools 36.1.0 are required; set ANDROID_HOME and JAVA_HOME');
  return { java, aapt, signer };
}

function inspectApk(filename, tools = sdkTools()) {
  const absolute = path.resolve(filename);
  const stat = fs.statSync(absolute);
  if (!stat.isFile() || stat.size < 1 || stat.size > MAX_APK_BYTES) throw new Error('APK must be a file no larger than 256 MiB');
  const bytes = fs.readFileSync(absolute);
  const signed = execFileSync(tools.java, ['-jar', tools.signer, 'verify', '--verbose', '--print-certs', absolute], { encoding: 'utf8', maxBuffer: 1024 * 1024 });
  const certificates = [...signed.matchAll(/Signer #\d+ certificate SHA-256 digest: ([a-fA-F0-9]{64})/g)].map(match => match[1].toLowerCase());
  if (certificates.length !== 1) throw new Error('A single verified signing identity is required');
  const badging = execFileSync(tools.aapt, ['dump', 'badging', absolute], { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 });
  const pkg = badging.match(/^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'/m);
  const minSdk = Number(badging.match(/^sdkVersion:'(\d+)'/m)?.[1]);
  const versionCode = Number(pkg?.[2]);
  if (!pkg || !Number.isSafeInteger(versionCode) || versionCode < 1 || !Number.isSafeInteger(minSdk) || minSdk < 1) {
    throw new Error('APK package, version and SDK metadata could not be verified');
  }
  if (/\bsplit='[^']+'/.test(badging.split('\n')[0])) throw new Error('Use a complete APK, not an APK split');
  const confirmed = fs.statSync(absolute);
  if (confirmed.size !== bytes.length || sha256(fs.readFileSync(absolute)) !== sha256(bytes)) {
    throw new Error('APK changed during verification; finish the build before preparing updates');
  }
  return { bytes, packageName: pkg[1], versionCode, versionName: pkg[3], minSdk, signerSha256: certificates[0], sha256: sha256(bytes), size: bytes.length };
}

function releaseRecord(apk) {
  return { versionCode: apk.versionCode, sha256: apk.sha256, size: apk.size, signerSha256: apk.signerSha256 };
}

function validateRelease(record, signerSha256, latestVersion) {
  if (!record || !Number.isSafeInteger(record.versionCode) || record.versionCode < 1 || record.versionCode > latestVersion ||
      !HASH.test(record.sha256) || !Number.isSafeInteger(record.size) || record.size < 1 || record.size > MAX_APK_BYTES ||
      record.signerSha256 !== signerSha256) throw new Error('Invalid or incompatible previously approved release');
  return releaseRecord(record);
}

function validateManifest(manifest) {
  if (!manifest || manifest.schemaVersion !== 1 || !CHANNEL_PACKAGES[manifest.channel] ||
      manifest.packageName !== CHANNEL_PACKAGES[manifest.channel]) throw new Error('Invalid update manifest identity');
  const latest = manifest.latest;
  if (!latest || !Number.isSafeInteger(latest.versionCode) || latest.versionCode < 1 ||
      typeof latest.versionName !== 'string' || !latest.versionName.trim() || latest.versionName.length > 80 || /[\u0000-\u001f\u007f-\u009f]/.test(latest.versionName) ||
      !Number.isSafeInteger(latest.minSdk) || latest.minSdk < 1 || latest.minSdk > 1000 || !HASH.test(latest.signerSha256)) {
    throw new Error('Invalid latest release metadata');
  }
  function asset(item) {
    if (!item || !HASH.test(item.sha256) || !Number.isSafeInteger(item.size) || item.size < 1 || item.size > MAX_APK_BYTES) throw new Error('Invalid release asset');
    validateAssetUrl(item.url);
  }
  asset(latest.apk);
  if (!Array.isArray(manifest.releases) || manifest.releases.length < 1 || manifest.releases.length > 200 ||
      !Array.isArray(manifest.patches) || manifest.patches.length > 200) throw new Error('Invalid release inventory');
  const releases = new Map();
  for (const record of manifest.releases) {
    validateRelease(record, latest.signerSha256, latest.versionCode);
    if (releases.has(record.sha256)) throw new Error('Duplicate approved APK hash');
    releases.set(record.sha256, record);
  }
  const target = releases.get(latest.apk.sha256);
  if (!target || target.versionCode !== latest.versionCode || target.size !== latest.apk.size) throw new Error('Latest APK is missing its exact approved release record');
  const sources = new Set();
  for (const patch of manifest.patches) {
    asset(patch);
    const base = releases.get(patch.fromSha256);
    if (patch.format !== 'dsupd1-gzip' || patch.toSha256 !== latest.apk.sha256 || !base ||
        base.versionCode >= latest.versionCode || sources.has(patch.fromSha256) || patch.size >= latest.apk.size) throw new Error('Invalid patch relationship');
    sources.add(patch.fromSha256);
  }
  if (Buffer.byteLength(`${JSON.stringify(manifest, null, 2)}\n`, 'utf8') > 512 * 1024) throw new Error('Manifest exceeds 512 KiB');
  return manifest;
}

function prepareBundle(options, inspect = inspectApk) {
  const target = inspect(options.apk);
  const expectedPackage = CHANNEL_PACKAGES[options.channel];
  if (!expectedPackage || target.packageName !== expectedPackage) throw new Error('APK does not match the selected channel package');
  const baseUrl = validateAssetUrl(options['base-url']);
  if (!baseUrl.endsWith('/')) throw new Error('--base-url must end with /');
  const manifest = { schemaVersion: 1, channel: options.channel, packageName: target.packageName, latest: {
    versionCode: target.versionCode, versionName: target.versionName, minSdk: target.minSdk,
    apk: {}, signerSha256: target.signerSha256
  }, releases: [], patches: [] };
  const records = new Map();
  if (options['previous-manifest']) {
    const previousPath = path.resolve(options['previous-manifest']);
    if (fs.statSync(previousPath).size > 512 * 1024) throw new Error('Previous manifest is too large');
    const previous = JSON.parse(fs.readFileSync(previousPath, 'utf8'));
    validateManifest(previous);
    if (previous.schemaVersion !== 1 || previous.channel !== manifest.channel || previous.packageName !== target.packageName ||
        !Array.isArray(previous.releases) || previous.releases.length > 200 || !previous.latest || previous.latest.versionCode > target.versionCode) {
      throw new Error('Previous manifest is incompatible or would downgrade the channel');
    }
    for (const item of previous.releases) {
      const valid = validateRelease(item, target.signerSha256, target.versionCode);
      const prior = records.get(valid.sha256);
      if (prior && JSON.stringify(prior) !== JSON.stringify(valid)) throw new Error('Conflicting release identity');
      records.set(valid.sha256, valid);
    }
    if (previous.latest.versionCode === target.versionCode && previous.latest.apk?.sha256 !== target.sha256) {
      throw new Error('A different published target must increment versionCode');
    }
  }
  const artifacts = new Map();
  const apkName = `Daniel-Short-${options.channel}-v${target.versionCode}-${target.sha256.slice(0, 16)}.apk`;
  artifacts.set(apkName, target.bytes);
  manifest.latest.apk = { url: new URL(apkName, baseUrl).href, sha256: target.sha256, size: target.size };
  for (const filename of options.base || []) {
    const base = inspect(filename);
    if (base.packageName !== target.packageName || base.signerSha256 !== target.signerSha256) throw new Error('Base and target APK identities differ');
    if (base.versionCode >= target.versionCode) throw new Error('Target versionCode must be greater than every base');
    records.set(base.sha256, releaseRecord(base));
    if (manifest.patches.some(patch => patch.fromSha256 === base.sha256)) continue;
    const delta = encodePatch(base.bytes, target.bytes);
    if (!applyPatch(base.bytes, delta).equals(target.bytes)) throw new Error('Generated patch failed its exact reconstruction check');
    if (delta.length >= target.size || delta.length > MAX_APK_BYTES) continue;
    const patchHash = sha256(delta);
    const name = `patch-${base.sha256.slice(0, 16)}-${target.sha256.slice(0, 16)}-${patchHash.slice(0, 16)}.dsupd.gz`;
    artifacts.set(name, delta);
    manifest.patches.push({ fromSha256: base.sha256, toSha256: target.sha256, url: new URL(name, baseUrl).href, sha256: patchHash, size: delta.length, format: 'dsupd1-gzip' });
  }
  records.set(target.sha256, releaseRecord(target));
  if (records.size > 200) throw new Error('Too many approved release records');
  manifest.releases = [...records.values()].sort((a, b) => a.versionCode - b.versionCode || a.sha256.localeCompare(b.sha256));
  manifest.patches.sort((a, b) => a.fromSha256.localeCompare(b.fromSha256));
  validateManifest(manifest);
  artifacts.set('latest.json', Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`));
  artifacts.set('SHA256SUMS.txt', Buffer.from([...artifacts].map(([name, bytes]) => `${sha256(bytes)}  ${name}\n`).join('')));
  return { manifest, artifacts };
}

function writeBundle(directory, artifacts) {
  const output = path.resolve(directory);
  fs.mkdirSync(output, { recursive: true });
  // Check every destination before writing any bytes; releases are immutable.
  for (const [name, bytes] of artifacts) {
    if (path.basename(name) !== name || name === '.' || name === '..') throw new Error('Invalid artifact filename');
    const destination = path.join(output, name);
    if (fs.existsSync(destination) && !fs.readFileSync(destination).equals(bytes)) throw new Error(`Refusing to overwrite different artifact: ${name}`);
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
    const { manifest, artifacts } = prepareBundle(options);
    const output = writeBundle(options.output, artifacts);
    process.stdout.write(`Prepared ${manifest.latest.versionName} (${manifest.latest.versionCode}) for ${manifest.packageName}\n${manifest.patches.length} verified patches; ${manifest.releases.length} approved APK hashes\nOutput: ${output}\nLocal staging only. No files were published.\n`);
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
}

module.exports = { CHANNEL_PACKAGES, validateAssetUrl, validateManifest, parseArgs, inspectApk, prepareBundle, writeBundle };
