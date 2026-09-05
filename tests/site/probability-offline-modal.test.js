'use strict';

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const read = (file) => fs.readFileSync(path.join(__dirname, '../..', file), 'utf8');

function createHarness() {
  let document;
  class Node {
    constructor(tag, classes = '') {
      this.tagName = tag.toUpperCase();
      this.nodeType = tag === '#comment' ? 8 : tag === '#document' ? 9 : 1;
      this.parentNode = null;
      this.childNodes = [];
      this.attributes = new Map();
      this.listeners = new Map();
      this.hidden = false;
      this.inert = false;
      const names = new Set(classes.split(' ').filter(Boolean));
      this.classList = {
        contains: (name) => names.has(name),
        add: (...values) => values.forEach((name) => names.add(name)),
        remove: (...values) => values.forEach((name) => names.delete(name)),
        toggle: (name, force = !names.has(name)) => { if (force) names.add(name); else names.delete(name); return force; }
      };
    }
    get children() { return this.childNodes.filter((node) => node.nodeType === 1); }
    get parentElement() { return this.parentNode?.nodeType === 1 ? this.parentNode : null; }
    get isConnected() { return this === document || Boolean(this.parentNode?.isConnected); }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    hasAttribute(name) { return this.attributes.has(name); }
    removeAttribute(name) { this.attributes.delete(name); }
    toggleAttribute(name, force = !this.hasAttribute(name)) { if (force) this.setAttribute(name, ''); else this.removeAttribute(name); }
    appendChild(node) { node.remove(); this.childNodes.push(node); node.parentNode = this; return node; }
    before(node) {
      const parent = this.parentNode;
      node.remove();
      parent.childNodes.splice(parent.childNodes.indexOf(this), 0, node);
      node.parentNode = parent;
    }
    replaceWith(node) { this.before(node); this.remove(); }
    remove() {
      if (this.parentNode) this.parentNode.childNodes.splice(this.parentNode.childNodes.indexOf(this), 1);
      this.parentNode = null;
    }
    querySelector(selector) {
      const all = this.children.flatMap((node) => [node, ...node.descendants()]);
      if (selector === '.modal.active') return all.find((node) => node.classList.contains('modal') && node.classList.contains('active')) || null;
      return null;
    }
    descendants() { return this.children.flatMap((node) => [node, ...node.descendants()]); }
    addEventListener(type, callback, options) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Map());
      this.listeners.get(type).set(callback, options);
    }
    removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
    dispatchEvent(event) {
      for (const [callback, options] of [...this.listeners.get(event.type) || []]) {
        if (options?.once) this.removeEventListener(event.type, callback);
        callback.call(this, event);
      }
      return !event.defaultPrevented;
    }
    focus() { document.activeElement = this; }
  }
  document = new Node('#document');
  document.documentElement = document.appendChild(new Node('html'));
  document.body = document.documentElement.appendChild(new Node('body'));
  document.body.dataset = {};
  document.readyState = 'loading';
  document.createComment = () => new Node('#comment');
  const header = document.body.appendChild(new Node('header'));
  const protectedNode = document.body.appendChild(new Node('aside'));
  protectedNode.inert = true;
  protectedNode.setAttribute('inert', '');
  protectedNode.setAttribute('aria-hidden', 'false');
  const owner = document.body.appendChild(new Node('main'));
  const opener = owner.appendChild(new Node('button'));
  const modal = owner.appendChild(new Node('div', 'modal hidden'));
  modal.setAttribute('aria-hidden', 'true');
  const claim = modal.appendChild(new Node('button'));
  document.activeElement = opener;
  const window = new Node('window');
  window.EventTarget = Node;
  window.location = { href: 'https://example.test/games/probability-engine' };
  const pending = new Map();
  window.SiteMotion = {
    presence(node, open, options) {
      this.cancel(node);
      if (open) {
        node.hidden = false;
        node.classList.add(options.className);
        return Promise.resolve(true);
      }
      node.classList.remove(options.className);
      return new Promise((resolve) => pending.set(node, { options, resolve }));
    },
    cancel(node) { const motion = pending.get(node); pending.delete(node); motion?.resolve(false); }
  };
  class Credits {
    constructor(value) { this.value = value; }
    gt(value) { return this.value > value; }
    add(other) { return new Credits(this.value + other.value); }
    static zero() { return new Credits(0); }
  }
  const state = { wallet: new Credits(100), runCash: new Credits(50), lifetimeCash: new Credits(200), offline: { pendingGain: new Credits(7), awaySeconds: 60 } };
  let renders = 0;
  const context = vm.createContext({
    window, document, HTMLElement: Node, URL, AbortController, CustomEvent, Event, console,
    MutationObserver: class { observe() {} },
    dom: { offlineModal: modal, claimOfflineButton: claim },
    stateRef: { current: state }, Big: Credits, addLog() {}, formatBig: (value) => value.value,
    renderAll: () => { renders += 1; }
  });
  vm.runInContext(read('js/navigation/site-route-runtime.js'), context);
  vm.runInContext(read('js/common/modal-accessibility.js'), context);
  const app = read('js/games/probability-engine/app.js');
  const start = app.indexOf('const offlineModalAccessibility =');
  const end = app.indexOf('function validateImportedSave', start);
  const claimStart = app.indexOf('  dom.claimOfflineButton.addEventListener("click"');
  const claimEnd = app.indexOf('  window.addEventListener("beforeunload"', claimStart);
  if (start < 0 || end < start || claimStart < 0 || claimEnd < claimStart) throw new Error('Offline modal controller or claim handler not found.');
  const code = `(() => { let lastFocusedBeforeOffline = null; ${app.slice(start, end)} ${app.slice(claimStart, claimEnd)} return { openOfflineModal, closeOfflineModal }; })()`;
  const runtime = window.SiteRoutes;
  const id = 'games:probability-engine';
  const mount = async () => {
    runtime.ensureLegacyRoute(id, { scripts: [] });
    const controller = runtime.runInScope(id, () => vm.runInContext(code, context));
    await runtime.mount(id, { navigationType: 'load', root: owner });
    return controller;
  };
  const finishClose = () => {
    const motion = pending.get(modal);
    pending.delete(modal);
    motion?.options.onFinish?.();
    motion?.resolve(true);
  };
  return { document, window, header, protectedNode, owner, opener, modal, claim, state, runtime, mount, finishClose, renders: () => renders };
}

module.exports = async function runProbabilityOfflineModalTests({ assert }) {
  const h = createHarness();
  let modal = await h.mount();
  modal.openOfflineModal();
  assert(h.modal.parentElement === h.document.body && !h.modal.hidden && h.modal.classList.contains('active'), 'offline dialog must escape the clipped route viewport and become visible');
  assert(h.document.activeElement === h.claim && h.header.inert && h.owner.inert, 'opening must focus Claim and isolate persistent navigation');
  assert(h.document.body.classList.contains('modal-open'), 'the shared controller must lock background scrolling');
  modal.closeOfflineModal();
  assert(h.modal.parentElement === h.document.body && !h.modal.hidden && h.owner.inert, 'normal close must preserve the visible dialog and isolation until exit completes');
  h.finishClose();
  assert(h.modal.hidden && h.modal.classList.contains('hidden') && h.modal.parentElement === h.owner, 'completed close must restore the original owned node and hide it');
  assert(h.document.activeElement === h.opener && !h.header.inert && !h.owner.inert, 'completed close must restore focus and shared navigation');
  assert(h.protectedNode.inert && h.protectedNode.getAttribute('aria-hidden') === 'false', 'dialog cleanup must preserve preexisting inert and aria state');

  modal.openOfflineModal();
  modal.closeOfflineModal();
  h.owner.remove();
  h.header.focus();
  await h.runtime.unmount();
  h.finishClose();
  assert(h.modal.parentElement === h.owner && !h.modal.isConnected && h.modal.hidden, 'leaving during close must restore the dialog to its detached route owner');
  assert(!h.header.inert && !h.document.body.classList.contains('modal-open') && h.document.activeElement === h.header, 'route disposal must immediately release persistent UI without stealing destination focus');
  assert(h.state.wallet.value === 100 && h.state.offline.pendingGain.value === 7 && h.renders() === 0, 'route disposal must never invoke Claim or mutate offline earnings');

  h.document.body.appendChild(h.owner);
  h.opener.focus();
  modal = await h.mount();
  modal.openOfflineModal();
  h.claim.dispatchEvent(new Event('click'));
  assert(h.state.wallet.value === 107 && h.state.runCash.value === 57 && h.state.lifetimeCash.value === 207 && h.state.offline.pendingGain.value === 0, 'a remounted Claim must award pending gains exactly once');
  assert(h.renders() === 1, 'legacy unmount must remove the prior Claim listener before remount binds it again');
  h.finishClose();
  modal.openOfflineModal();
  await h.runtime.unmount();
  assert(h.modal.parentElement === h.owner && h.modal.hidden && !h.header.inert && !h.owner.inert, 'leaving an open dialog must restore all shared controls synchronously');
};

if (require.main === module) {
  module.exports({ assert: require('node:assert/strict') }).then(() => {
    console.log('Probability Engine offline dialog lifecycle passed.');
  }).catch((error) => { console.error(error); process.exitCode = 1; });
}
