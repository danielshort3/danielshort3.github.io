import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { zipSync } from 'fflate';
import { validateExtensionArchive } from '../scripts/validate-package.mjs';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const encoder = new TextEncoder();

const fakePng = (size) => {
  const bytes = new Uint8Array(24);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, size);
  view.setUint32(20, size);
  return bytes;
};

const archiveFixture = async (target) => {
  const manifest = JSON.parse(await readFile(path.join(packageDir, 'manifest.json'), 'utf8'));
  if (target === 'store') delete manifest.key;
  const files = {
    'manifest.json': encoder.encode(JSON.stringify(manifest)),
    'background/service-worker.js': encoder.encode(''),
    'sidepanel/sidepanel.html': encoder.encode(''),
    'content/tracker-bridge.js': encoder.encode(''),
    'THIRD_PARTY_NOTICES.txt': encoder.encode('mammoth@1.12.0\npdfjs-dist@6.1.200\n')
  };
  for (const [size, filename] of Object.entries(manifest.icons)) files[filename] = fakePng(Number(size));
  return zipSync(files);
};

test('manifest declares runtime icon sizes, homepage, and supported Chrome floor', async () => {
  const manifest = JSON.parse(await readFile(path.join(packageDir, 'manifest.json'), 'utf8'));
  assert.equal(manifest.minimum_chrome_version, '120');
  assert.equal(manifest.homepage_url, 'https://www.danielshort.me/job-application-copilot');
  assert.deepEqual(Object.keys(manifest.icons), ['16', '32', '48', '128']);
  assert.deepEqual(manifest.action.default_icon, manifest.icons);
  for (const filename of Object.values(manifest.icons)) {
    assert.ok((await readFile(path.join(packageDir, 'src', filename))).length > 24);
  }
});

test('store archive validator requires a keyless manifest and packaged notices', async () => {
  const storeArchive = await archiveFixture('store');
  const result = validateExtensionArchive(storeArchive, { target: 'store' });
  assert.equal(Object.hasOwn(result.manifest, 'key'), false);
  assert.ok(result.paths.includes('THIRD_PARTY_NOTICES.txt'));

  const devArchive = await archiveFixture('dev');
  assert.throws(
    () => validateExtensionArchive(devArchive, { target: 'store' }),
    /must not contain a key/u
  );
  assert.doesNotThrow(() => validateExtensionArchive(devArchive, { target: 'dev' }));
});

test('packaging creates a separate Store artifact without mutating the dev manifest', async () => {
  const source = await readFile(path.join(packageDir, 'scripts/package.mjs'), 'utf8');
  assert.match(source, /const packagedManifest = structuredClone\(manifest\)/u);
  assert.match(source, /if \(target === 'store'\) delete packagedManifest\.key/u);
  assert.match(source, /chrome-web-store\.zip/u);
  assert.match(source, /validateExtensionArchive\(archiveBytes, \{ target \}\)/u);
});
