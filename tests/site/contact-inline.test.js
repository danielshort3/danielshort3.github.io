'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { createHarness } = require('./mobile-contact.test');
const { render } = require('../../js/forms/contact-markup');
const { renderVisualPageBody } = require('../../build/lib/section-renderers');
const page = require('../../content/pages/contact.json');
const turn = () => new Promise(resolve => setImmediate(resolve));

async function main() {
  const inlineMarkup = render({ inline: true });
  const modalMarkup = render();
  const generated = renderVisualPageBody(page);
  assert.equal((generated.match(/id="contact-form"/g) || []).length, 1);
  assert(!generated.includes('id="contact-modal"') && !generated.includes('<iframe'));
  assert(generated.includes('https://github.com/danielshort3') && generated.includes('Open in Maps'));
  for (const markup of [inlineMarkup, modalMarkup]) {
    assert(markup.includes('data-contact-error-list') && markup.includes('Used to reply to your message.'));
    assert(markup.includes('maxlength="200"') && markup.includes('maxlength="254"') && markup.includes('maxlength="4000"'));
    assert(!markup.includes('reply shortly'));
  }
  const browserGlobal = { window: {}, module: { exports: {} } };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../js/forms/contact-markup.js'), 'utf8'), browserGlobal);
  assert.equal(browserGlobal.window.SiteContactMarkup, browserGlobal.module.exports,
    'the bundled CommonJS branch must also expose its browser renderer');

  const h = createHarness();
  h.document.documentElement = h.document.createElement('html');
  h.storedDrafts.set('contact:personal', { name: 'Recovered name', email: 'old@example.com', message: 'Recovered note' });
  const scene = h.createScene({ inline: true });
  h.evaluate();
  h.document.dispatch('DOMContentLoaded');
  const controller = h.window.initializeContactModal(scene.root);
  assert(controller.inline && scene.modal.parentElement === scene.main);
  assert.equal(h.records.size, 0, 'inline mounting must not initialize modal background isolation');
  assert.equal(h.document.activeElement, h.document.body, 'ordinary inline mounting must not steal focus');
  assert.equal(scene.fields.message.value, 'Recovered note');
  assert.equal(h.document.querySelectorAll('#contact-form').length, 1);
  h.window.openContactModal();
  assert.equal(h.document.activeElement, scene.fields.name);
  assert(!h.document.body.classList.contains('modal-open') && !scene.root.inert);
  h.document.dispatch('keydown', { key: 'Escape' });
  assert(!scene.modal.hidden && scene.fields.message.value === 'Recovered note');

  Object.values(scene.fields).forEach(field => { field.value = ''; });
  scene.form.dispatch('submit');
  assert.equal(h.requests.length, 0);
  assert.equal(h.document.activeElement, scene.errors);
  assert(!scene.errors.hidden && scene.errorList.children.length === 3);
  const nameLink = scene.errorList.children[0].children[0];
  assert.equal(nameLink.getAttribute('href'), '#contact-name');
  scene.errors.dispatch('click', { target: nameLink });
  assert.equal(h.document.activeElement, scene.fields.name);
  scene.fields.name.value = 'Example Person';
  scene.fields.name.dispatch('input');
  assert.equal(scene.errorList.children.length, 2);
  assert.equal(scene.fields.name.getAttribute('aria-invalid'), null);
  scene.fields.email.value = 'visitor@';
  scene.fields.email.validity.valid = false;
  scene.fields.message.value = 'Keep this note after a failure.';
  scene.form.dispatch('submit');
  assert.equal(h.requests.length, 0);
  assert.equal(scene.errorList.children.length, 1);
  assert(scene.errorList.children[0].children[0].textContent.includes('valid email'));
  assert.equal(scene.fields.message.value, 'Keep this note after a failure.');

  scene.fields.email.value = 'example@example.com';
  scene.fields.email.validity.valid = true;
  let resolveRequest;
  h.window.fetch = (url, options) => {
    h.requests.push({ url, ...options });
    return new Promise(resolve => { resolveRequest = resolve; });
  };
  scene.form.dispatch('submit');
  scene.form.dispatch('submit');
  assert.equal(h.requests.length, 1, 'duplicate clicks cannot submit twice');
  assert(scene.submit.disabled && scene.fields.message.readOnly);
  assert(scene.errors.hidden);
  resolveRequest({ ok: false, status: 503, json: async () => ({ error: 'Unavailable' }) });
  await turn();
  assert.equal(scene.fields.message.value, 'Keep this note after a failure.');
  assert.equal(h.storedDrafts.get('contact:personal').message, scene.fields.message.value);
  assert.equal(scene.label.textContent, 'Retry sending');
  assert.equal(h.document.activeElement.id, 'contact-status');
  assert(!scene.submit.disabled && !scene.fields.message.readOnly);
  assert(scene.success.hidden);
  scene.form.dispatch('submit');
  assert.equal(h.requests.length, 2, 'only an explicit retry sends a second request');
  resolveRequest({ ok: true, status: 200, json: async () => ({ ok: true }) });
  await turn();
  assert(scene.form.hidden && !scene.success.hidden);
  assert.equal(h.document.activeElement, scene.success);
  assert(Object.values(scene.fields).every(field => field.value === ''));
  assert(!h.storedDrafts.has('contact:personal'), 'only confirmed success clears the saved contact draft');
  controller.dispose();
  assert(scene.modal.parentElement === scene.main && scene.modal.isConnected,
    'inline cleanup leaves its source node in place without a portal or orphan');

  const modal = h.createScene();
  h.window.initializeContactModal(modal.root);
  modal.opener.dispatch('click');
  modal.form.dispatch('submit');
  assert.equal(h.document.activeElement, modal.fields.name, 'the retained modal keeps blank-submit focus on Name');
  console.log('Inline contact tests passed: shared markup, validation links, no modal isolation, drafts, explicit retry and confirmed success. No messages sent.');
}

module.exports = main;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
