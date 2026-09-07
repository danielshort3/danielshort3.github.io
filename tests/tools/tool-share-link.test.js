'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/tool-share-link.js'), 'utf8');

class Element {
  constructor(properties = {}) {
    Object.assign(this, { dataset: {}, listeners: new Map(), textContent: '', value: '', checked: false, type: '', id: '', name: '' }, properties);
  }

  addEventListener(type, callback, options = {}) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push({ callback, once: options.once });
  }

  dispatchEvent(event) {
    const entries = [...(this.listeners.get(event.type) || [])];
    entries.forEach((entry) => {
      if (entry.once) this.listeners.set(event.type, this.listeners.get(event.type).filter((item) => item !== entry));
      entry.callback(event);
    });
  }

  setAttribute(name, value) { this[name] = value; }
}

function input(id, type, value, properties = {}) {
  return new Element({ id, type, value, tagName: type === 'select-one' ? 'SELECT' : 'INPUT', ...properties });
}

function harness({ url = 'https://example.test/tools/screen-recorder', explicit = true, readyState = 'interactive', softRoute = false } = {}) {
  const controls = [];
  const inserted = [];
  const clipboard = [];
  const timers = [];
  const cleanup = [];
  const button = explicit ? new Element({ textContent: 'Share settings', dataset: { toolShareLink: '' } }) : null;
  const scope = new Element({ parentNode: {} });
  const primary = new Element({ parentNode: scope });
  primary.insertAdjacentElement = (position, element) => inserted.push({ position, element });
  scope.querySelectorAll = (selector) => selector.includes(':not(')
    ? controls.filter((control) => !['file', 'password', 'hidden', 'submit', 'button'].includes(control.type))
    : controls;
  scope.querySelector = (selector) => {
    if (selector.includes('data-tool-share-link')) return button || inserted[0]?.element || null;
    if (selector.startsWith('button[')) return primary;
    return scope.querySelectorAll(selector)[0] || null;
  };
  const document = new Element({ readyState, currentScript: softRoute ? { dataset: { siteRouteOwnedScript: 'tools:screen-recorder' } } : null });
  document.querySelector = (selector) => selector === 'main' ? scope : selector.includes('data-tools-account') ? {} : null;
  document.createElement = () => new Element();
  const window = new Element({ location: new URL(url), SiteRoutes: { addCleanup: (callback) => cleanup.push(callback) } });
  const context = vm.createContext({
    document, window, URL, URLSearchParams, TextEncoder, TextDecoder, Uint8Array, Event,
    btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
    atob: (value) => Buffer.from(value, 'base64').toString('binary'),
    setTimeout: (callback) => { timers.push(callback); return timers.length; },
    navigator: { clipboard: { writeText: async (value) => { clipboard.push(value); } } },
    history: { replaceState: (state, title, value) => { window.location = new URL(value, window.location); } }
  });
  return {
    controls, inserted, clipboard, button, timers, document, window,
    run: () => vm.runInContext(source, context),
    emit: (type) => {
      document.dispatchEvent(new Event(type));
      if (type === 'DOMContentLoaded') window.dispatchEvent(new Event(type));
    },
    copy: async () => {
      (button || inserted[0].element).dispatchEvent(new Event('click'));
      await Promise.resolve();
      return clipboard[clipboard.length - 1];
    }
  };
}

function addSettings(env, selected) {
  const controls = {
    audio: input('screenrec-audio', 'checkbox', 'on', { checked: selected }),
    mic: input('screenrec-mic', 'checkbox', 'on', { checked: !selected }),
    fps: input('screenrec-fps', 'select-one', selected ? '30' : '15'),
    secret: input('private-token', 'password', 'do-not-share'),
    radioA: input('mode-a', 'radio', 'a', { name: 'mode', checked: selected }),
    radioB: input('mode-b', 'radio', 'b', { name: 'mode', checked: !selected })
  };
  env.controls.push(...Object.values(controls));
  return controls;
}

function addFormats(env, selected) {
  const controls = {
    auto: input('format-auto', 'checkbox', 'auto', { name: 'screenrec-format', checked: !selected }),
    mp4: input('format-mp4', 'checkbox', 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', { name: 'screenrec-format', checked: selected }),
    webm: input('format-webm', 'checkbox', 'video/webm;codecs=vp9', { name: 'screenrec-format', checked: selected }),
    png: input('image-png', 'checkbox', 'image/png', { name: 'screenrec-image-format', checked: selected })
  };
  env.controls.push(...Object.values(controls));
  return controls;
}

async function run() {
  const original = harness({});
  addSettings(original, true);
  original.run();
  original.run();
  assert.strictEqual(original.button.listeners.get('click'), undefined, 'deferred scripts must finish before share setup');
  addFormats(original, true);
  original.emit('DOMContentLoaded');
  assert.strictEqual(original.inserted.length, 0, 'an explicit share button should retain its authored position');
  assert.strictEqual(original.button.listeners.get('click').length, 1, 'an existing share button should only be wired once');
  const sharedUrl = await original.copy();
  const payload = JSON.parse(Buffer.from(new URL(sharedUrl).searchParams.get('s'), 'base64url').toString('utf8'));
  assert.strictEqual(original.clipboard.length, 1);
  assert.deepStrictEqual(payload['screenrec-format'], ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/webm;codecs=vp9']);
  assert.deepStrictEqual(payload['screenrec-mic'], []);
  assert.strictEqual(payload['private-token'], undefined, 'password values must remain excluded');
  original.timers.forEach((callback) => callback());
  assert.strictEqual(original.button.textContent, 'Share settings', 'copy feedback must restore an explicit button label');

  const restored = harness({ url: sharedUrl });
  const restoredSettings = addSettings(restored, false);
  restored.run();
  const restoredFormats = addFormats(restored, false);
  let changed = 0;
  restoredFormats.mp4.addEventListener('change', () => { changed += 1; });
  restored.emit('DOMContentLoaded');
  assert.strictEqual(restoredSettings.audio.checked, true);
  assert.strictEqual(restoredSettings.mic.checked, false);
  assert.strictEqual(restoredSettings.fps.value, '30');
  assert.strictEqual(restoredSettings.radioA.checked, true, 'named radios with their own IDs must restore by group');
  assert.strictEqual(restoredSettings.radioB.checked, false);
  assert.strictEqual(restoredFormats.auto.checked, false);
  assert.strictEqual(restoredFormats.mp4.checked, true, 'codec commas must not split a checkbox value');
  assert.strictEqual(restoredFormats.webm.checked, true);
  assert.strictEqual(restoredFormats.png.checked, true);
  assert.strictEqual(changed, 1, 'restored dynamic controls must notify their tool listeners');
  assert.strictEqual(restored.window.location.search, '');

  const soft = harness({ url: sharedUrl, readyState: 'complete', softRoute: true });
  addSettings(soft, false);
  soft.run();
  soft.emit('DOMContentLoaded');
  assert(soft.window.location.search.includes('s='), 'soft-route state must survive until later scripts create format controls');
  const softFormats = addFormats(soft, false);
  soft.emit('site:route-mounted');
  assert.strictEqual(softFormats.auto.checked, false);
  assert.strictEqual(softFormats.mp4.checked, true);
  assert.strictEqual(softFormats.png.checked, true);
  assert.strictEqual(soft.button.listeners.get('click').length, 1);

  const generic = harness({ explicit: false, readyState: 'complete' });
  generic.controls.push(input('message', 'text', 'Example'));
  generic.run();
  generic.run();
  assert.strictEqual(generic.inserted.length, 1, 'other tools should keep one automatically inserted button');
  assert.strictEqual(generic.inserted[0].position, 'afterend');
  assert.strictEqual(generic.inserted[0].element.textContent, 'Copy link to my inputs');
  assert((await generic.copy()).includes('?s='));

  const legacyPayload = Buffer.from(JSON.stringify({ v: 1, 'screenrec-audio': 'on', 'screenrec-mic': '', 'screenrec-format': 'video/webm;codecs=vp9' })).toString('base64url');
  const legacy = harness({ url: `https://example.test/tools/screen-recorder?s=${legacyPayload}` });
  const legacySettings = addSettings(legacy, false);
  const legacyFormats = addFormats(legacy, false);
  legacy.run();
  legacy.emit('DOMContentLoaded');
  assert.strictEqual(legacySettings.audio.checked, true, 'legacy checkbox strings should remain readable');
  assert.strictEqual(legacySettings.mic.checked, false);
  assert.strictEqual(legacyFormats.webm.checked, true);
  console.log('Tool share link tests passed.');
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
