'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const source = fs.readFileSync(path.join(__dirname, '../../js/common/certifications-modal.js'), 'utf8');

function createHarness() {
  let document;
  class Node {
    constructor(tag, attributes = {}) {
      this.tagName = tag.toUpperCase();
      this.attributes = attributes;
      this.children = [];
      this.parentNode = null;
      this.listeners = new Map();
      this.focusCount = 0;
      this.hidden = false;
      const classes = new Set((attributes.class || '').split(' ').filter(Boolean));
      this.classList = {
        contains: (name) => classes.has(name),
        add: (name) => classes.add(name),
        remove: (name) => classes.delete(name)
      };
    }
    get parentElement() { return this.parentNode; }
    get isConnected() { return this === document || Boolean(this.parentNode?.isConnected); }
    matches(selector) {
      if (this.tagName.startsWith('#')) return false;
      return selector.split(',').some((raw) => {
        const match = raw.trim();
        if (match.startsWith('#')) return this.attributes.id === match.slice(1);
        if (match.startsWith('.')) return this.classList.contains(match.slice(1));
        if (match.startsWith('[')) return Object.hasOwn(this.attributes, match.slice(1, -1));
        return this.tagName === match.toUpperCase();
      });
    }
    querySelectorAll(selector) {
      return this.children.flatMap((child) => [child, ...child.querySelectorAll(selector)]).filter((child) => child.matches(selector));
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) {
      for (let node = this; node; node = node.parentElement) if (node.matches(selector)) return node;
      return null;
    }
    contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
    appendChild(node) {
      node.remove();
      this.children.push(node);
      node.parentNode = this;
      return node;
    }
    before(node) {
      node.remove();
      this.parentNode.children.splice(this.parentNode.children.indexOf(this), 0, node);
      node.parentNode = this.parentNode;
    }
    replaceWith(node) { this.before(node); this.remove(); }
    remove() {
      if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
      this.parentNode = null;
    }
    focus() { document.activeElement = this; this.focusCount += 1; }
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(callback);
    }
    removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
    dispatch(type, details = {}) {
      [...this.listeners.get(type) || []].forEach((callback) => callback({ type, target: this, preventDefault() {}, ...details }));
    }
    count(type) { return this.listeners.get(type)?.size || 0; }
  }
  document = new Node('#document');
  document.body = document.appendChild(new Node('body'));
  document.activeElement = document.body;
  document.readyState = 'loading';
  document.createComment = () => new Node('#comment');
  const add = (parent, tag, attributes) => parent.appendChild(new Node(tag, attributes));
  const createScene = ({ bodyModal = false } = {}) => {
    const root = add(document.body, 'section', { 'data-site-route-content': '' });
    const main = add(root, 'main');
    const opener = add(main, 'button', { 'data-cert-modal-open': '' });
    const modal = add(bodyModal ? document.body : main, 'div', { id: 'certifications-modal', class: 'modal' });
    const content = add(modal, 'div', { class: 'modal-content' });
    const close = add(content, 'button', { class: 'modal-close' });
    const credential = add(content, 'a');
    return { root, main, opener, modal, content, close, credential };
  };
  const window = new Node('window');
  window.location = new URL('https://example.test/analytics?audience=analytics#proof');
  window.history = { replaceState: (state, title, next) => { window.location = new URL(next, window.location.href); } };
  const records = new Map();
  const release = () => document.body.children.forEach((node) => { node.inert = false; });
  window.createModalAccessibility = (modal) => {
    const record = { pending: null, disposed: false };
    records.set(modal, record);
    modal.hidden = true;
    const api = {
      show() { record.pending = null; modal.hidden = false; },
      isolateBackground() { document.body.children.forEach((node) => { node.inert = node !== modal; }); },
      hide({ onFinish, immediate = false } = {}) {
        modal.classList.remove('active');
        const complete = () => {
          if (record.pending !== complete) return;
          record.pending = null;
          modal.hidden = true;
          release();
          document.body.classList.remove('modal-open');
          onFinish?.();
        };
        record.pending = complete;
        if (immediate) complete();
      },
      dispose() { record.disposed = true; api.hide({ immediate: true }); }
    };
    return api;
  };
  const evaluate = () => vm.runInNewContext(source, { window, document, URL, URLSearchParams }, { filename: 'js/common/certifications-modal.js' });
  return { document, window, add, createScene, evaluate, records };
}

module.exports = function runCertificationPortalTests({ assert }) {
  const h = createHarness();
  const scene = h.createScene();
  h.evaluate();
  assert(scene.modal.parentElement === scene.main, 'credential markup should remain in its source location until initialization');
  h.document.dispatch('DOMContentLoaded');
  const record = h.records.get(scene.modal);
  assert(scene.modal.parentElement === h.document.body && h.document.querySelectorAll('#certifications-modal').length === 1,
    'initialization must portal the existing credential dialog without creating duplicate IDs');
  assert(scene.credential.parentElement === scene.content, 'portaling must retain original credential links and content');
  h.document.dispatch('site:route-mounted', { detail: { root: scene.root } });
  h.document.dispatch('site:content-updated', { detail: { root: scene.root } });
  assert(h.records.get(scene.modal) === record && scene.opener.count('click') === 1,
    'direct initialization and repeated route events must reuse the portaled controller');
  scene.opener.focus();
  scene.opener.dispatch('click');
  assert(scene.root.inert && !scene.modal.inert && !scene.modal.hidden,
    'opening must isolate the whole route while keeping the body-level dialog usable');
  assert(h.window.location.search.includes('modal=certifications') && h.window.location.hash === '#proof',
    'opening the portaled dialog must preserve audience and hash state');
  scene.close.dispatch('click');
  const stale = record.pending;
  assert(!scene.modal.hidden && scene.content.count('keydown') === 1 && scene.root.inert,
    'closing must retain rendered content, focus trap, and background isolation until animation completes');
  const priorFocus = scene.opener.focusCount;
  scene.opener.dispatch('click');
  stale();
  assert(!scene.modal.hidden && scene.opener.focusCount === priorFocus && scene.content.count('keydown') === 1,
    'rapid reopening must cancel stale close cleanup without recreating the dialog');
  scene.close.dispatch('click');
  record.pending();
  assert(scene.modal.hidden && !scene.root.inert && h.document.activeElement === scene.opener,
    'completed closing must release the route and restore the original opener');
  assert(h.window.location.search === '?audience=analytics' && h.window.location.hash === '#proof',
    'closing must remove only the modal query parameter');
  scene.opener.dispatch('click');
  scene.close.dispatch('click');
  h.document.dispatch('site:route-unmounted', { detail: { root: scene.root } });
  assert(record.disposed && !record.pending && !scene.content.count('keydown') && !scene.opener.count('click'),
    'route disposal during exit must settle the animation and remove all controller listeners');
  assert(scene.modal.parentElement === scene.main && !scene.root.inert,
    'disposing a connected route must restore its source markup and release background isolation');
  h.document.dispatch('site:route-mounted', { detail: { root: scene.root } });
  assert(scene.modal.parentElement === h.document.body && scene.opener.count('click') === 1,
    'a restored route must remount one functional dialog');
  scene.root.remove();
  h.document.dispatch('site:route-unmounted', { detail: { root: scene.root } });
  assert(!scene.modal.isConnected && h.document.querySelectorAll('#certifications-modal').length === 0,
    'unmounting a removed route must remove its portal rather than orphaning it in the body');
  const replacement = h.createScene();
  h.document.dispatch('site:route-mounted', { detail: { root: replacement.root } });
  replacement.root.remove();
  const next = h.createScene();
  h.document.dispatch('site:route-mounted', { detail: { root: next.root } });
  assert(!replacement.modal.isConnected && next.modal.parentElement === h.document.body && h.document.querySelectorAll('#certifications-modal').length === 1,
    'mounting after a missed unmount must dispose the disconnected owner before adopting a new dialog');
  const unrelated = h.add(h.document.body, 'section', { 'data-site-route-content': '' });
  h.document.dispatch('site:route-mounted', { detail: { root: unrelated } });
  assert(next.modal.parentElement === next.main && !next.opener.count('click'),
    'a route without credentials must not adopt the previous route body portal');
  next.root.remove();
  unrelated.remove();
  const bodyScene = h.createScene({ bodyModal: true });
  h.document.dispatch('site:content-updated', { detail: { root: h.document } });
  bodyScene.root.remove();
  h.document.dispatch('site:route-unmounted', { detail: { root: bodyScene.root } });
  assert(!bodyScene.modal.isConnected, 'a body-level dialog owned by a removed route must also be disposed');

  const noAccessibility = createHarness();
  const plain = noAccessibility.createScene();
  delete noAccessibility.window.createModalAccessibility;
  noAccessibility.evaluate();
  noAccessibility.document.dispatch('DOMContentLoaded');
  assert(plain.modal.parentElement === plain.main && !plain.opener.count('click'),
    'missing accessibility initialization must leave source markup intact');
};

if (require.main === module) {
  let count = 0;
  module.exports({ assert(condition, message) { count += 1; require('assert').ok(condition, message); } });
  process.stdout.write(`Certification portal tests passed (${count} assertions).\n`);
}
