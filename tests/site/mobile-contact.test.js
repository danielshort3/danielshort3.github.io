'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/forms/contact.js'), 'utf8');

function createHarness() {
  let document;
  class Node {
    constructor(tag = 'div', attributes = {}) {
      this.tagName = tag.toUpperCase();
      this.attributes = { ...attributes };
      this.children = [];
      this.parentNode = null;
      this.dataset = {};
      this.listeners = new Map();
      this.hidden = false;
      this.inert = false;
      this.disabled = false;
      this.value = '';
      this.textContent = '';
      this.focusCount = 0;
      const classes = new Set((attributes.class || '').split(' ').filter(Boolean));
      this.classList = {
        contains: (name) => classes.has(name),
        add: (...names) => names.forEach((name) => classes.add(name)),
        remove: (...names) => names.forEach((name) => classes.delete(name)),
        toggle(name, force) {
          const enabled = force === undefined ? !classes.has(name) : force;
          if (enabled) classes.add(name);
          else classes.delete(name);
          return enabled;
        }
      };
    }
    get parentElement() { return this.parentNode?.tagName === '#DOCUMENT' ? null : this.parentNode; }
    get isConnected() { return this === document || Boolean(this.parentNode?.isConnected); }
    get id() { return this.attributes.id || ''; }
    get type() { return this.attributes.type || ''; }
    getAttribute(name) {
      if (name.startsWith('data-')) {
        const key = name.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
        if (Object.hasOwn(this.dataset, key)) return this.dataset[key];
      }
      if (name === 'hidden' && this.hidden) return '';
      if (name === 'inert' && this.inert) return '';
      return Object.hasOwn(this.attributes, name) ? this.attributes[name] : null;
    }
    hasAttribute(name) { return this.getAttribute(name) !== null; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    matches(selector) {
      return selector.split(',').some((part) => {
        let candidate = part.trim();
        if (this.tagName.startsWith('#')) return false;
        const excluded = candidate.match(/:not\(([^)]+)\)/);
        if (excluded && this.matches(excluded[1])) return false;
        candidate = candidate.replace(/:not\([^)]+\)/g, '');
        const id = candidate.match(/#([\w-]+)/);
        if (id && this.id !== id[1]) return false;
        const tag = candidate.match(/^[a-z][\w-]*/i);
        if (tag && this.tagName !== tag[0].toUpperCase()) return false;
        if ([...candidate.matchAll(/\.([\w-]+)/g)].some((match) => !this.classList.contains(match[1]))) return false;
        return [...candidate.matchAll(/\[([\w-]+)(?:="([^"]*)")?\]/g)].every((match) =>
          match[2] === undefined ? this.hasAttribute(match[1]) : this.getAttribute(match[1]) === match[2]);
      });
    }
    querySelectorAll(selector) {
      return this.children.flatMap((node) => [node, ...node.querySelectorAll(selector)]).filter((node) => node.matches(selector));
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    closest(selector) {
      for (let current = this; current; current = current.parentElement) if (current.matches(selector)) return current;
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
      const parent = this.parentNode;
      node.remove();
      parent.children.splice(parent.children.indexOf(this), 0, node);
      node.parentNode = parent;
    }
    replaceWith(node) {
      this.before(node);
      this.remove();
    }
    remove() {
      if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1);
      this.parentNode = null;
    }
    addEventListener(type, callback) {
      if (!this.listeners.has(type)) this.listeners.set(type, new Set());
      this.listeners.get(type).add(callback);
    }
    removeEventListener(type, callback) { this.listeners.get(type)?.delete(callback); }
    dispatch(type, details = {}) {
      const event = { type, target: this, preventDefault() {}, ...details };
      [...this.listeners.get(type) || []].forEach((callback) => callback(event));
    }
    focus() { document.activeElement = this; this.focusCount += 1; }
    reset() { this.querySelectorAll('input,textarea').forEach((node) => { node.value = ''; }); }
  }

  document = new Node('#document');
  document.body = document.appendChild(new Node('body'));
  document.activeElement = document.body;
  document.readyState = 'loading';
  document.createComment = () => new Node('#comment');
  document.getElementById = (id) => document.querySelector('#' + id);
  const add = (parent, tag, attributes) => parent.appendChild(new Node(tag, attributes));
  const createScene = ({ injected = false } = {}) => {
    const root = add(document.body, 'section', { 'data-site-route-content': '' });
    const main = add(root, 'main', { id: 'main' });
    const opener = add(main, 'button', { id: 'contact-form-toggle' });
    const modal = add(injected ? document.body : main, 'div', { id: 'contact-modal', class: 'modal' });
    if (injected) modal.dataset.contactModalInjected = 'true';
    const content = add(modal, 'div', { class: 'modal-content' });
    const close = add(content, 'button', { class: 'modal-close' });
    const body = add(content, 'div', { class: 'modal-body' });
    const form = add(body, 'form', { id: 'contact-form', action: '/api/contact' });
    const fields = {};
    for (const name of ['name', 'email', 'message']) {
      const field = add(form, 'div', { class: 'form-field' });
      fields[name] = add(field, name === 'message' ? 'textarea' : 'input', { id: 'contact-' + name, name, type: name === 'email' ? 'email' : 'text' });
      fields[name].validity = { valid: true };
      add(field, 'span', { id: 'contact-' + name + '-required' });
    }
    add(form, 'p', { id: 'contact-status' });
    add(form, 'div', { id: 'contact-alt' });
    add(form, 'button', { 'data-contact-reset': '' });
    add(form, 'button', { type: 'submit' });
    const success = add(body, 'div', { id: 'contact-success' });
    add(success, 'button', { 'data-contact-new': '' });
    return { root, main, opener, modal, content, close, form, fields };
  };
  const records = new Map();
  const registrations = new Map();
  const requests = [];
  const window = new Node('window');
  window.location = { href: 'https://example.test/contact', pathname: '/contact', hash: '' };
  window.requestAnimationFrame = (callback) => callback();
  window.setTimeout = () => 1;
  window.clearTimeout = () => {};
  window.fetch = (url, options) => {
    requests.push({ url, ...options });
    return new Promise(() => {});
  };
  window.SiteRoutes = { register: (name, lifecycle) => registrations.set(name, lifecycle) };
  window.SiteMotion = { swap: (body, update) => update() };
  window.createModalAccessibility = (modal) => {
    const record = { pending: null, disposed: 0 };
    records.set(modal, record);
    modal.hidden = true;
    const restore = () => document.body.children.forEach((node) => { node.inert = false; });
    const controller = {
      show() {
        record.pending = null;
        modal.hidden = false;
        modal.inert = false;
        modal.dataset.motionState = 'open';
      },
      isolateBackground() {
        document.body.children.forEach((node) => { node.inert = node !== modal; });
      },
      hide({ onFinish, immediate = false } = {}) {
        modal.classList.remove('active');
        modal.dataset.motionState = 'closing';
        const complete = () => {
          if (record.pending !== complete) return;
          record.pending = null;
          restore();
          onFinish?.();
          modal.hidden = true;
          modal.inert = true;
          modal.dataset.motionState = 'closed';
          if (!document.querySelector('.modal.active')) document.body.classList.remove('modal-open');
        };
        record.pending = complete;
        if (immediate) complete();
      },
      dispose() {
        record.disposed += 1;
        controller.hide({ immediate: true });
      }
    };
    return controller;
  };
  const evaluate = () => vm.runInNewContext(source, {
    window,
    document,
    sessionStorage: { getItem: () => null, removeItem() {} },
    fetch: window.fetch,
    AbortController,
    FormData: class {
      constructor(form) { this.form = form; }
      get(name) { return this.form.querySelector('[name="' + name + '"]')?.value || ''; }
    },
    console
  }, { filename: 'js/forms/contact.js' });
  return { document, window, createScene, evaluate, records, registrations, requests };
}

module.exports = function runMobileContactTests({ assert }) {
  const prepared = createHarness();
  const preparedScene = prepared.createScene();
  prepared.document.body.dataset.siteRouteModule = 'page:content';
  prepared.evaluate();
  prepared.document.dispatch('DOMContentLoaded');
  assert(preparedScene.modal.parentElement === preparedScene.main && !prepared.window.__contactModalReady,
    'preparing the contact controller on another route must not portal or initialize its modal');
  prepared.window.initializeContactModal(preparedScene.root);
  assert(preparedScene.modal.parentElement === prepared.document.body,
    'an explicit contact request must still initialize a prepared controller');
  const h = createHarness();
  const scene = h.createScene();
  h.evaluate();
  assert(scene.modal.parentElement === scene.main, 'contact markup must stay in the route until JavaScript initialization');
  h.document.dispatch('DOMContentLoaded');
  assert(scene.modal.parentElement === h.document.body, 'initialized contact dialog should leave isolated route stacking contexts');
  assert(h.document.querySelectorAll('#contact-modal').length === 1, 'portaling must move the existing dialog rather than duplicate it');
  const controller = h.window.initializeContactModal(scene.root);
  assert(controller?.modal === scene.modal && controller.ownerRoot === scene.root, 'route mounts must find their previously portaled controller');
  const cleanups = [];
  h.registrations.get('contact:contact').mount({ root: scene.root, cleanup: (callback) => cleanups.push(callback) });
  assert(cleanups.length === 1 && scene.opener.listeners.get('click').size === 1,
    'direct initialization followed by route mount should register cleanup without duplicate listeners');

  scene.opener.focus();
  scene.opener.dispatch('click');
  scene.fields.message.value = 'Draft retained while the modal reverses.';
  assert(scene.modal.classList.contains('active') && h.document.activeElement === scene.content && scene.root.inert,
    'portaled dialog should focus itself and isolate the complete route background');
  scene.close.dispatch('click');
  const staleClose = h.records.get(scene.modal).pending;
  assert(!scene.modal.hidden && h.document.body.classList.contains('modal-open'), 'dialog exit should retain visibility and body scroll lock');
  assert(h.window.initializeContactModal(scene.root) === controller, 'remounting during exit must preserve its pending controller');
  h.window.openContactModal();
  staleClose();
  assert(scene.modal.classList.contains('active') && !scene.modal.hidden && scene.fields.message.value.includes('Draft retained'),
    'reopening the portaled form should preserve entered text and ignore stale close cleanup');
  scene.close.dispatch('click');
  h.records.get(scene.modal).pending();
  assert(scene.modal.hidden && h.document.activeElement === scene.opener && !scene.root.inert,
    'completed close should restore the original route opener and background accessibility');

  scene.opener.dispatch('click');
  scene.close.dispatch('click');
  cleanups[0]();
  assert(scene.modal.parentElement === scene.main && scene.modal.hidden, 'route disposal should restore the hidden source node to its placeholder');
  assert(!scene.content.listeners.get('keydown')?.size && !scene.opener.listeners.get('click')?.size,
    'disposing during exit must remove both its focus trap and trigger listeners');
  assert(!h.document.body.classList.contains('modal-open') && !h.window.openContactModal,
    'route disposal must release locks and its public open callback');
  scene.root.remove();
  assert(h.document.querySelectorAll('#contact-modal').length === 0, 'removing the disposed route must leave no orphaned body dialog');

  const next = h.createScene();
  const nextController = h.window.initializeContactModal(next.root);
  assert(nextController !== controller && next.modal.parentElement === h.document.body, 'a new route should mount and portal its own source dialog');
  assert(next.fields.message.value.includes('Draft retained'), 'contact draft text should survive an accepted route departure and remount');
  nextController.dispose();
  next.root.remove();
  h.document.body.dataset.audience = 'tourism';
  const professional = h.createScene();
  const professionalController = h.window.initializeContactModal(professional.root);
  assert(professional.fields.message.value === '', 'contact drafts must not cross audience boundaries');
  professionalController.dispose();
  professional.root.remove();
  h.document.body.dataset.audience = 'personal';
  h.document.body.appendChild(next.root);
  h.window.initializeContactModal(next.root);
  next.root.remove();
  const replacement = h.createScene();
  const replacementController = h.window.initializeContactModal(replacement.root);
  assert(!next.modal.isConnected && h.document.querySelectorAll('#contact-modal').length === 1,
    'mounting after a detached owner must remove the stale portal before adopting the next dialog');
  h.document.dispatch('site:route-unmounted', { detail: { root: replacement.root } });
  assert(replacement.modal.parentElement === replacement.main && !h.window.openContactModal,
    'route-unmounted should dispose the matching portaled controller even outside explicit route cleanup');
  assert(replacementController !== nextController, 'route replacement must not reuse the previous form controller');
  replacement.root.remove();

  const injected = h.createScene({ injected: true });
  const injectedController = h.window.initializeContactModal(h.document);
  assert(injectedController.ownerRoot === injected.root && injected.modal.parentElement === h.document.body,
    'already injected body dialogs should be owned by the current route without another portal');
  h.document.dispatch('site:route-unmounted', { detail: { root: injected.root } });
  assert(!injected.modal.isConnected, 'route disposal should remove its dynamically injected body dialog');
  injected.root.remove();

  const sending = h.createScene();
  const sendingController = h.window.initializeContactModal(sending.root);
  sending.fields.name.value = 'Test User';
  sending.fields.email.value = 'test@example.test';
  sending.fields.message.value = 'Pending submission';
  sending.opener.dispatch('click');
  sending.form.dispatch('submit');
  assert(h.requests.length === 1 && sendingController.sending && !h.registrations.get('contact:contact').beforeLeave(),
    'portaling must preserve the pending-submission route veto');
  let prevented = false;
  h.document.dispatch('site:route-before-leave', { preventDefault() { prevented = true; } });
  assert(prevented && !h.window.SiteContact.canLeave(), 'a pending global contact submission must veto departure from every route family');
  sendingController.dispose();
  assert(h.requests[0].signal.aborted && !h.window.__contactModalReady, 'disposal must abort an in-flight submission and clear modal readiness');
};

if (require.main === module) {
  let assertions = 0;
  module.exports({ assert(condition, message) {
    if (!condition) throw new Error(message);
    assertions += 1;
  } });
  console.log(`Mobile contact lifecycle tests passed (${assertions} assertions).`);
}
