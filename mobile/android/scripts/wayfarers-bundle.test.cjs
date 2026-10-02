'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const { bundle, MODULES } = require('../../../build/bundle-wayfarers-android.cjs');
const { prepareBundle, parseArgs, validateManifest } = require('./prepare-app-update.cjs');
const { sha256, applyPatch } = require('./app-update-format.cjs');
const ROOT = path.resolve(__dirname, '../../..');

test('standalone assets use the canonical game bytes and contain no website cache or network scripts', () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), 'wayfarers-bundle-'));
  const records = bundle(output);
  const html = fs.readFileSync(path.join(output, 'wayfarers/index.html'), 'utf8');
  assert.match(html, /<main id="main" class="wg-app" data-wayfarers-guild>/);
  assert.match(html, /<body class="wayfarers-guild-page">/);
  assert.match(html, /connect-src 'none'/);
  assert.doesNotMatch(html, /serviceWorker|service-worker|site-consent|analytics|https:\/\/(?!appassets)/);
  assert.doesNotMatch(html, /personal-accordion|personal-game-header/);
  for (const name of MODULES) {
    const record = records.find(record => record.path === `wayfarers/${name}.js`);
    assert.equal(record.sha256, sha256(fs.readFileSync(path.join(ROOT, record.source))));
    assert.deepEqual(fs.readFileSync(path.join(output, record.path)), fs.readFileSync(path.join(ROOT, record.source)));
  }
  assert.equal(records.find(record => record.path === 'wayfarers/game.css').sha256,
    sha256(fs.readFileSync(path.join(ROOT, 'css/games/wayfarers-guild.css'))));
  for (const image of ['living-trail.webp', 'living-room.webp', 'living-mine.webp', 'actors.png', 'props.png', 'realms.png', 'ui-icons.png']) {
    assert.ok(records.some(record => record.path === `img/wayfarers-guild/${image}`), image);
  }
});

test('Wayfarers release profile retains strict identity and reconstructs a separate signed product', () => {
  const options = parseArgs(['--apk', 'candidate', '--base', 'baseline', '--output', 'outside-repo',
    '--base-url', 'https://github.com/danielshort3/danielshort3.github.io/releases/download/wayfarers-guild-v0.1.1/', '--channel', 'wayfarers']);
  const base = Buffer.alloc(65536, 42);
  const target = Buffer.concat([base, Buffer.from('candidate')]);
  const inspect = file => {
    const bytes = file === 'baseline' ? base : target;
    return { bytes, packageName: 'me.danielshort.wayfarers', versionCode: file === 'baseline' ? 1 : 2,
      versionName: file === 'baseline' ? '0.1.0' : '0.1.1', minSdk: 26, signerSha256: 'ab'.repeat(32), sha256: sha256(bytes), size: bytes.length };
  };
  const { manifest, artifacts } = prepareBundle(options, inspect);
  assert.equal(manifest.channel, 'wayfarers');
  assert.equal(manifest.packageName, 'me.danielshort.wayfarers');
  assert.ok(manifest.latest.apk.url.includes('/Wayfarers-Guild-v2-'));
  validateManifest(manifest);
  const patch = manifest.patches[0];
  assert.deepEqual(applyPatch(base, artifacts.get(new URL(patch.url).pathname.split('/').pop())), target);
  assert.throws(() => prepareBundle(options, file => ({ ...inspect(file), packageName: 'me.danielshort.app' })), /selected channel package/);
  assert.throws(() => validateManifest({ ...manifest, channel: 'stable' }), /manifest identity/);
});

test('native save export never hands off stale textarea contents when canonical export fails', () => {
  for (const succeeds of [true, false]) {
    const callbacks = {}, timers = [], messages = [];
    const input = { value: 'previous pasted import' };
    const button = { dataset: { export: 'download' } };
    const options = { setAttribute() {}, addEventListener() {} };
    const document = {
      documentElement: { style: { setProperty() {} } },
      querySelector: selector => selector === '.wg-exit' ? options : selector === '#wg-save-text' ? input : {},
      addEventListener: (kind, callback) => { callbacks[kind] = callback; }
    };
    const window = { innerHeight: 800, addEventListener() {}, WayfarersAndroid: { postMessage: text => messages.push(JSON.parse(text)) } };
    vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/web/android.js'), 'utf8'), {
      window, document, MutationObserver: class { observe() {} }, setTimeout: callback => timers.push(callback)
    });
    callbacks.click({ target: { closest: () => button } });
    assert.equal(button.dataset.export, 'text');
    assert.equal(input.value, '');
    // Canonical app.exportSave writes the textarea only after store.export succeeds.
    if (succeeds) input.value = 'fresh verified export';
    timers.forEach(callback => callback());
    assert.equal(button.dataset.export, 'download');
    assert.deepEqual(messages, succeeds ? [{ type: 'export', text: 'fresh verified export' }] : []);
    assert.equal(input.value, succeeds ? 'fresh verified export' : 'previous pasted import');
  }
});

test('standalone Settings removes the browser file picker and its separate label', () => {
  let mutation;
  const file = { id: 'wg-save-file', hidden: false }, label = { hidden: false };
  const nativeButtons = [];
  const body = {
    querySelector: selector => selector === '#wg-save-text' ? {} : selector === 'input[type="file"]' ? file : selector === 'label[for="wg-save-file"]' ? label : null,
    prepend: actions => { nativeButtons.push(...actions.children); }
  };
  const options = { setAttribute() {}, addEventListener() {} };
  const document = {
    documentElement: { style: { setProperty() {} } },
    querySelector: selector => selector === '.wg-exit' ? options : body,
    addEventListener() {},
    createElement: () => ({ children: [], dataset: {}, addEventListener() {}, append(node) { this.children.push(node); } })
  };
  vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/web/android.js'), 'utf8'), {
    document, window: { innerHeight: 800, addEventListener() {} },
    MutationObserver: class { constructor(callback) { mutation = callback; } observe() {} }
  });
  mutation();
  assert.equal(file.hidden, true);
  assert.equal(label.hidden, true);
  assert.deepEqual(nativeButtons.map(button => button.textContent), ['App updates', 'Open save file']);
});
