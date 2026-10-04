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

test('standalone sideload installer survives the main app Play-source removal without duplicate classes', () => {
  const gradle = fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/build.gradle.kts'), 'utf8');
  const adapter = fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/src/main/java/me/danielshort/app/updates/AutomaticAppInstaller.kt'), 'utf8');
  assert.match(gradle, /exclude\("me\/danielshort\/app\/updates\/AutomaticAppInstaller\.kt"\)/);
  assert.match(gradle, /exclude\("me\/danielshort\/app\/updates\/AutomaticInstallReceiver\.kt"\)/);
  assert.match(adapter, /class AutomaticAppInstaller/);
  assert.match(adapter, /AutomaticInstallEngine\(/);
  assert.doesNotMatch(adapter, /SiteApplication/);
});

test('standalone assets use the canonical game bytes and contain no website cache or network scripts', () => {
  const output = fs.mkdtempSync(path.join(os.tmpdir(), 'wayfarers-bundle-'));
  const records = bundle(output);
  const html = fs.readFileSync(path.join(output, 'wayfarers/index.html'), 'utf8');
  assert.match(html, /<main id="main" class="wg-app" data-wayfarers-guild>/);
  assert.match(html, /<body class="wayfarers-guild-page">/);
  assert.match(html, /connect-src 'none'/);
  assert.doesNotMatch(html, /serviceWorker|service-worker|site-consent|analytics|https:\/\/(?!appassets)/);
  assert.doesNotMatch(html, /personal-accordion|personal-game-header/);
  assert.ok(html.indexOf('wayfarers/persistence.js') < html.indexOf('wayfarers/checkpoint.js'));
  assert.ok(html.indexOf('wayfarers/native-checkpoint.js') < html.indexOf('wayfarers/checkpoint.js'));
  assert.ok(html.indexOf('wayfarers/checkpoint.js') < html.indexOf('wayfarers/app.js'));
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

function checkpointHarness(main, native) {
  const persistence = require('../../../js/games/wayfarers-guild/persistence.js');
  const values = new Map(main == null ? [] : [[persistence.SAVE_KEY, main]]);
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
  const messages = [];
  const window = { WayfarersStorage: { ...persistence }, WayfarersNativeCheckpoint: native,
    WayfarersAndroid: { postMessage: text => messages.push(JSON.parse(text)) } };
  vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/web/checkpoint.js'), 'utf8'), { window, localStorage: storage });
  return { window, storage, messages, values, key: persistence.SAVE_KEY };
}

function checkpointEnvelope(createdAt, savedAt, boots = 0) {
  const core = require('../../../js/games/wayfarers-guild/core.js');
  const state = core.createState(createdAt);
  state.lastUpdate = savedAt;
  state.upgrades.boots = boots;
  return JSON.stringify({ format: 'wayfarers-guild-save', version: state.schemaVersion, savedAt, state });
}

test('native checkpoint recovers only a validated latest same-guild save or a reviewed replacement', () => {
  const old = checkpointEnvelope(1000, 2000);
  const recent = checkpointEnvelope(1000, 3000, 1);
  const replacement = checkpointEnvelope(4000, 5000, 2);
  assert.equal(checkpointHarness(old, { text: recent }).values.get('wayfarers-guild-save-v1'), recent);
  assert.equal(checkpointHarness(null, { text: recent }).values.get('wayfarers-guild-save-v1'), recent);
  assert.equal(checkpointHarness(recent, { text: old }).values.get('wayfarers-guild-save-v1'), recent);
  assert.equal(checkpointHarness(old, { text: replacement }).values.get('wayfarers-guild-save-v1'), old);
  assert.equal(checkpointHarness(old, { text: replacement, replacesCreatedAt: 1000 }).values.get('wayfarers-guild-save-v1'), replacement);
  assert.equal(checkpointHarness(old, { text: '{}' }).values.get('wayfarers-guild-save-v1'), old);
  const unsupported = JSON.stringify({ ...JSON.parse(old), version: 99 });
  assert.equal(checkpointHarness(unsupported, { text: recent }).values.get('wayfarers-guild-save-v1'), unsupported);
});

test('native checkpoint mirrors successful canonical saves and explicitly reviewed imports', () => {
  const old = checkpointEnvelope(1000, 2000);
  const h = checkpointHarness(old, null);
  const store = h.window.WayfarersStorage.createStore({ storage: h.storage, now: () => 3000 });
  const loaded = store.load();
  loaded.state.upgrades.boots = 1;
  assert.equal(store.save(loaded.state).ok, true);
  assert.equal(h.messages.length, 1);
  assert.equal(JSON.parse(h.messages[0].text).state.upgrades.boots, 1);
  assert.equal(h.window.WayfarersCheckpoint.confirmed(), false);
  h.window.WayfarersAndroid.onmessage({ data: JSON.stringify({ type: 'checkpoint', requestId: 1, ok: true }) });
  assert.equal(h.window.WayfarersCheckpoint.confirmed(), true);
  assert.equal(store.save({}).ok, false);
  assert.equal(h.messages.length, 1);
  assert.equal(store.replaceImport(checkpointEnvelope(4000, 5000, 2)).persisted, true);
  assert.equal(h.messages[1].replacesCreatedAt, 1000);
  assert.equal(JSON.parse(h.messages[1].text).state.createdAt, 4000);
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

test('native content readiness waits for every visible real station and its durable checkpoint', () => {
  let interval, cleared = 0, flushes = 0, confirmed = false;
  const messages = [];
  const canvas = (top, status) => ({ dataset: { sceneStatus: status },
    getBoundingClientRect: () => ({ top, bottom: top + 300, left: 0, right: 390, width: 390, height: 300 }) });
  const first = canvas(0, 'ready'), second = canvas(300, 'loading'), offscreen = canvas(650, 'loading');
  const world = { getBoundingClientRect: () => ({ top: 0, bottom: 600, left: 0, right: 390 }),
    querySelectorAll: () => [first, second, offscreen] };
  const options = { setAttribute() {}, addEventListener() {} };
  const document = { documentElement: { style: { setProperty() {} } }, addEventListener() {},
    querySelector: selector => selector === '.wg-exit' ? options : selector === '[data-wx-station-world]' ? world :
      selector === '[data-wx-canvas]' ? { dataset: { sceneStatus: 'ready' } } : null };
  const window = { innerWidth: 390, innerHeight: 800, addEventListener() {},
    WayfarersContent: { version: 3, documentToken: 'new-document', recoveryToken: '' },
    WayfarersAndroid: { postMessage: text => messages.push(JSON.parse(text)) },
    WayfarersUI: { contentReady: () => true, flushForContentUpdate: () => { flushes++; return true; } },
    WayfarersCheckpoint: { confirmed: () => confirmed, snapshot: () => 'schema8 durable save', generation: () => '' } };
  vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'mobile/android/wayfarers/web/android.js'), 'utf8'), {
    window, document, MutationObserver: class { observe() {} },
    setInterval: callback => { interval = callback; return 1; }, clearInterval: () => { cleared++; }
  });
  interval();
  assert.equal(flushes, 0, 'The hidden legacy ready canvas cannot certify a loading station');
  second.dataset.sceneStatus = 'error'; interval();
  assert.equal(messages.length, 0, 'A failed visible image cannot commit executable content');
  second.dataset.sceneStatus = 'ready'; interval();
  assert.equal(flushes, 1, 'A loading offscreen station must not prevent lazy rendering');
  interval(); assert.equal(messages.length, 0, 'A painted scene still needs a durable native save');
  confirmed = true; interval();
  assert.deepEqual(messages, [{ type: 'content-ready', documentToken: 'new-document', version: 3,
    recoveryToken: '', text: 'schema8 durable save', generation: '' }]);
  assert.equal(cleared, 1);
});
