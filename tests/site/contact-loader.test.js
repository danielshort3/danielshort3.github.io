'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/common/common.js'), 'utf8');
const loaderSource = source.slice(source.indexOf('  function loadScriptOnce(src)'), source.indexOf('  window.requestContactModal = requestContactModal;'));
const turn = () => new Promise((resolve) => setImmediate(resolve));

function createHarness() {
  const scripts = [];
  const warnings = [];
  const navigations = [];
  const fields = Object.fromEntries(['name', 'email', 'message'].map((name) => ['contact-' + name, { value: '' }]));
  const modal = { closest: () => null };
  let opened = 0;
  const document = {
    head: { appendChild: (script) => scripts.push(script) },
    createElement: () => ({}),
    getElementById: (id) => id === 'contact-modal' ? modal : fields[id] || null
  };
  const window = {
    location: { hash: '', assign: (href) => navigations.push(href) },
    openContactModal: () => { opened += 1; }
  };
  vm.runInNewContext(loaderSource + '\nwindow.requestContactModal = requestContactModal;', {
    window, document, location: window.location,
    console: { warn: (...args) => warnings.push(args) },
    loadedScripts: new Map(),
    CONTACT_MODAL_ID: 'contact-modal',
    CONTACT_MODAL_SCRIPT: 'js/forms/contact.js',
    storeContactOrigin() {}
  }, { filename: 'js/common/common.js:contact-loader' });
  return { window, scripts, warnings, navigations, fields, opened: () => opened };
}

async function runContactLoaderTests() {
  const failing = createHarness();
  // Leave the click-style call unobserved until the next turn: the original
  // discarded .then(open) rejection must fail this process, not be swallowed by a test await.
  const failedRequest = failing.window.requestContactModal({ message: 'A draft' });
  failing.scripts[0].onerror();
  await turn();
  await failedRequest;
  assert.equal(failing.warnings.length, 1, 'a failed contact bundle must keep its diagnostic warning');
  assert.deepEqual(failing.navigations, ['/contact'], 'a failed contact bundle must offer the full contact page');
  assert.equal(failing.opened(), 0, 'load failure must not open an uninitialized modal');
  assert.equal(failing.window.location.hash, '', 'load failure must not claim the modal opened through its hash');
  assert.equal(failing.fields['contact-message'].value, '', 'load failure must not pretend prefill was applied');

  const working = createHarness();
  const request = working.window.requestContactModal({ name: 'Taylor', message: 'Project question' });
  working.scripts[0].onload();
  await request;
  assert.equal(working.opened(), 1, 'a loaded controller must open the requested contact modal');
  assert.equal(working.fields['contact-name'].value, 'Taylor');
  assert.equal(working.fields['contact-message'].value, 'Project question');
  assert.deepEqual(working.navigations, [], 'successful contact loading must stay on the current page');
  assert.equal(working.warnings.length, 0);

  working.window.__contactModalReady = true;
  working.window.requestContactModal();
  assert.equal(working.opened(), 2, 'an already prepared contact modal should still open immediately');
  assert.equal(working.scripts.length, 1, 'an already prepared controller must not load another bundle');
}

module.exports = runContactLoaderTests;

if (require.main === module) {
  runContactLoaderTests()
    .then(() => console.log('Contact loader tests passed (12 assertions).'))
    .catch((error) => { console.error(error); process.exitCode = 1; });
}
