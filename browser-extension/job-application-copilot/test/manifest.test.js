import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const extensionIdFromKey = (key) => [...createHash('sha256').update(Buffer.from(key, 'base64')).digest().subarray(0, 16)]
  .flatMap(byte => [byte >> 4, byte & 15])
  .map(nibble => String.fromCharCode('a'.charCodeAt(0) + nibble))
  .join('');

const sourceFiles = async (directory) => {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async entry => {
    const target = path.join(directory, entry.name);
    return entry.isDirectory() ? sourceFiles(target) : [target];
  }));
  return nested.flat();
};

test('manifest retains its stable ID and least-privilege surface', async () => {
  const manifest = JSON.parse(await readFile(path.join(packageDir, 'manifest.json'), 'utf8'));
  assert.equal(extensionIdFromKey(manifest.key), 'jigajpmnbiofgmgcnmdeechgibpjlfop');
  assert.deepEqual([...manifest.permissions].sort(), ['activeTab', 'scripting', 'sidePanel', 'storage']);
  assert.deepEqual(manifest.host_permissions, ['http://127.0.0.1:11434/*']);
  assert.deepEqual(manifest.content_scripts[0].js, ['content/tracker-bridge.js']);
  assert.deepEqual(manifest.content_scripts[0].matches, [
    'https://www.danielshort.me/tools/job-application-tracker*'
  ]);
  assert.equal(
    manifest.content_security_policy.extension_pages,
    "default-src 'self'; base-uri 'none'; object-src 'none'; frame-src 'none'; script-src 'self'; style-src 'self'; worker-src 'self'; connect-src http://127.0.0.1:11434"
  );
  assert.doesNotMatch(manifest.content_security_policy.extension_pages, /unsafe-|https?:\/\/(?!127\.0\.0\.1:11434)/u);
});

test('foundation source contains no cloud-model key placeholder', async () => {
  const files = (await sourceFiles(path.join(packageDir, 'src'))).filter(file => file.endsWith('.js'));
  const source = (await Promise.all(files.map(file => readFile(file, 'utf8')))).join('\n');
  assert.doesNotMatch(source, /OPENAI_API_KEY|api\.openai\.com/iu);
});

test('toolbar action uses the custom activeTab launch path', async () => {
  const serviceWorker = await readFile(path.join(packageDir, 'src/background/service-worker.js'), 'utf8');
  assert.match(serviceWorker, /setPanelBehavior\(\{ openPanelOnActionClick: false \}\)/u);
  assert.doesNotMatch(serviceWorker, /openPanelOnActionClick: true/u);
  assert.match(serviceWorker, /chrome\.action\.onClicked\.addListener/u);
  assert.match(serviceWorker, /toolbarActionLauncher\.launch\(tab\)/u);
});
