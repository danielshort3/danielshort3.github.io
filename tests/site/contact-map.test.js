'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..', '..');
const MAP_SOURCE = 'https://www.google.com/maps?q=Delta%2C%20CO&output=embed';

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(listener);
    },
    emit(type) { [...(listeners.get(type) || [])].forEach((listener) => listener()); }
  };
}

function element(tagName = 'div') {
  const attributes = new Map();
  const classes = new Set();
  const node = {
    tagName: tagName.toUpperCase(),
    dataset: {},
    style: {},
    childNodes: [],
    parentNode: null,
    hidden: false,
    inert: false,
    clientLeft: 0,
    clientTop: 0,
    clientWidth: 940,
    clientHeight: 300,
    scrollLeft: 0,
    scrollTop: 0,
    bounds: { left: 0, top: 0 },
    srcWrites: [],
    srcConnections: [],
    loadedDetachments: 0,
    classList: {
      contains: (name) => classes.has(name),
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name)
    },
    get isConnected() { return this.connectedRoot || Boolean(this.parentNode?.isConnected); },
    get firstChild() { return this.childNodes[0] || null; },
    getAttribute: (name) => attributes.get(name) ?? null,
    hasAttribute: (name) => attributes.has(name),
    setAttribute(name, value) {
      attributes.set(name, String(value));
      if (name === 'src') {
        this.srcWrites.push(String(value));
        this.srcConnections.push(this.isConnected);
      }
    },
    removeAttribute: (name) => attributes.delete(name),
    append(child) {
      child.remove();
      this.childNodes.push(child);
      child.parentNode = this;
    },
    insertBefore(child, reference) {
      child.remove();
      const index = this.childNodes.indexOf(reference);
      if (index < 0) this.childNodes.push(child);
      else this.childNodes.splice(index, 0, child);
      child.parentNode = this;
    },
    remove() {
      if (!this.parentNode) return;
      if (this.isConnected) {
        const recordDetach = (child) => {
          if (child.tagName === 'IFRAME' && child.hasAttribute('src')) child.loadedDetachments += 1;
          child.childNodes.forEach(recordDetach);
        };
        recordDetach(this);
      }
      this.parentNode.childNodes = this.parentNode.childNodes.filter((child) => child !== this);
      this.parentNode = null;
    },
    querySelector(selector) {
      const matches = (child) => selector === '[data-contact-map-slot]'
        ? child.hasAttribute('data-contact-map-slot')
        : selector === 'iframe[data-home-contact-map-src]' && child.tagName === 'IFRAME' &&
          child.hasAttribute('data-home-contact-map-src');
      for (const child of this.childNodes) {
        if (matches(child)) return child;
        const match = child.querySelector(selector);
        if (match) return match;
      }
      return null;
    },
    closest(selector) {
      const selectors = selector.split(',').map((value) => value.trim());
      for (let current = this; current; current = current.parentNode) {
        if (selectors.some((value) =>
          (value === '[hidden]' && current.hidden) ||
          (value === '[inert]' && current.inert) ||
          (value === '[aria-hidden="true"]' && current.getAttribute('aria-hidden') === 'true'))) return current;
      }
      return null;
    },
    getBoundingClientRect() { return this.bounds; }
  };
  return node;
}

function createContactBody() {
  const body = element('main');
  const slot = element();
  slot.setAttribute('data-contact-map-slot', '');
  slot.bounds = { left: 260, top: 500 };
  slot.clientLeft = 1;
  slot.clientTop = 1;
  const iframe = element('iframe');
  iframe.setAttribute('data-home-contact-map-src', MAP_SOURCE);
  slot.append(iframe);
  body.append(slot);
  return { body, slot, iframe };
}

function createHarness() {
  let nextTask = 0;
  const tasks = new Map();
  const resizeObservers = [];
  const root = element();
  root.connectedRoot = true;
  const viewport = element();
  viewport.bounds = { left: 60, top: 80 };
  viewport.scrollTop = 120;
  root.append(viewport);
  const initial = createContactBody();
  viewport.append(initial.body);
  let state = { body: initial.body, category: 'about', home: true, view: 'overview' };
  const document = Object.assign(eventTarget(), { createElement: element });
  const window = Object.assign(eventTarget(), {
    requestAnimationFrame(callback) {
      const id = ++nextTask;
      tasks.set(id, callback);
      return id;
    },
    SiteFrame: {
      current: () => state,
      viewport: () => viewport,
      root: () => root
    }
  });
  const observer = class {
    constructor(callback) { this.callback = callback; this.targets = []; }
    observe(target) { this.targets.push(target); }
    disconnect() { this.targets = []; }
  };
  const context = {
    window,
    document,
    ResizeObserver: class extends observer {
      constructor(callback) { super(callback); resizeObservers.push(this); }
    },
    MutationObserver: observer
  };
  const source = fs.readFileSync(path.join(ROOT, 'js/common/contact-map.js'), 'utf8');
  vm.runInNewContext(source, context);
  const frameSource = fs.readFileSync(path.join(ROOT, 'js/navigation/site-frame.js'), 'utf8');
  const replacement = frameSource.match(/  function replaceRouteBody\(nextBody\) \{[\s\S]*?\n  \}/);
  if (!replacement) throw new Error('Missing persistent route-body replacement');
  const replaceRouteBody = vm.runInNewContext(`(${replacement[0].trim()})`, { window, viewport });
  const flush = () => {
    for (let iteration = 0; tasks.size && iteration < 10; iteration += 1) {
      const pending = [...tasks.values()];
      tasks.clear();
      pending.forEach((callback) => callback());
    }
    if (tasks.size) throw new Error('Contact map updates did not settle');
  };
  const refresh = () => { window.ContactMap.refresh(); flush(); };
  return {
    ...initial, root, viewport, window, document, flush, refresh, replaceRouteBody,
    state: () => state,
    setState(next) { state = { ...state, ...next }; },
    host: () => viewport.childNodes.find((node) => node.hasAttribute('data-persistent-contact-map')),
    resize() { resizeObservers.forEach((entry) => entry.callback()); flush(); },
    rerun: () => vm.runInNewContext(source, context)
  };
}

module.exports = function runContactMapTests({ assert }) {
  const map = createHarness();
  map.flush();
  assert(!map.host() && map.iframe.srcWrites.length === 0,
    'The Contact map should not create a live map host or assign a URL before Contact is selected');

  map.setState({ category: 'contact' });
  map.viewport.inert = true;
  map.refresh();
  const host = map.host();
  assert(host?.isConnected && host.dataset.mapActive === 'true' && !host.hidden && host.inert &&
    map.iframe.srcWrites.length === 1 && map.iframe.srcConnections[0],
  'Selecting Contact during an incoming transition should prepare the connected map behind the clip while keeping it unfocusable');
  map.viewport.inert = false;
  map.refresh();
  assert(host?.isConnected && host.childNodes[0] === map.iframe && map.iframe.srcWrites.length === 1 &&
    map.iframe.srcWrites[0] === MAP_SOURCE && map.iframe.srcConnections[0] &&
    host.dataset.mapActive === 'true' && !host.hidden && !host.inert,
  'The first active Contact view should connect the original blank iframe before assigning its single map URL');
  assert(host.style.left === '201px' && host.style.top === '541px' &&
    host.style.width === '940px' && host.style.height === '300px',
  'The retained map should align with the map slot content box inside a scrolled viewport');

  const activeGeometry = JSON.stringify(host.style);
  map.viewport.inert = true;
  map.refresh();
  assert(host.dataset.mapActive === 'true' && !host.hidden && host.inert &&
    JSON.stringify(host.style) === activeGeometry && map.iframe.srcWrites.length === 1,
  'Closing Contact should preserve its painted map and dimensions under the viewport clip while disabling interaction');
  map.root.classList.add('site-frame--moving');
  map.slot.clientWidth = 600;
  map.slot.clientHeight = 240;
  map.resize();
  assert(host.dataset.mapActive === 'true' && !host.hidden && host.inert &&
    host.style.width === '600px' && host.style.height === '240px' &&
    host.style.left === '201px' && host.style.top === '541px' && map.iframe.loadedDetachments === 0,
  'Moving frame geometry should keep the existing map painted and aligned to its changing slot dimensions');
  map.viewport.inert = false;
  map.refresh();
  assert(host.dataset.mapActive === 'true' && !host.hidden && !host.inert,
    'The moving-frame class alone should not hide the map or keep an interactive viewport map inert');
  map.root.classList.remove('site-frame--moving');
  map.slot.clientWidth = 940;
  map.slot.clientHeight = 300;
  map.resize();

  map.setState({ category: 'tools' });
  map.refresh();
  assert(!host.dataset.mapActive && !host.hidden && host.inert &&
    JSON.stringify(host.style) === activeGeometry && map.iframe.isConnected && map.iframe.loadedDetachments === 0,
  'Leaving Contact should park the map without interaction, layout collapse, or detachment of its browsing context');
  map.setState({ category: 'contact' });
  map.refresh();
  assert(map.host() === host && host.dataset.mapActive === 'true' && !host.hidden && map.iframe.srcWrites.length === 1,
    'Returning to the Contact tab should reveal the same iframe without assigning src again');

  map.slot.hidden = true;
  map.refresh();
  assert(!host.dataset.mapActive && host.inert && !host.hidden,
    'An explicitly hidden Contact slot should park the map even when the active category is Contact');
  map.slot.hidden = false;
  map.body.setAttribute('aria-hidden', 'true');
  map.refresh();
  assert(!host.dataset.mapActive && host.inert && !host.hidden,
    'An aria-hidden Contact ancestor should park the map without collapsing the iframe layout');
  map.body.removeAttribute('aria-hidden');
  map.refresh();
  assert(host.dataset.mapActive === 'true' && !host.inert && map.iframe.srcWrites.length === 1,
    'Revealing Contact after an explicit hidden state should restore the existing map without a new load');

  map.viewport.scrollTop = 300;
  map.slot.bounds.top = 320;
  map.refresh();
  assert(host.style.top === '541px',
    'Scrolling the viewport should retain the map content position while its viewport rectangle changes');
  map.slot.clientWidth = 320;
  map.slot.clientHeight = 280;
  map.resize();
  assert(host.style.width === '320px' && host.style.height === '280px',
    'Resizing the active Contact slot should resize the retained map without loading another document');

  const standalone = createContactBody();
  standalone.slot.bounds = { left: 92, top: 260 };
  map.setState({ body: standalone.body, home: false, view: 'detail' });
  map.replaceRouteBody(standalone.body);
  assert(map.viewport.childNodes.includes(host) && map.iframe.isConnected && !host.dataset.mapActive && !host.hidden && host.inert,
    'Replacing a route body should retain the connected map host and hide it until the new Contact slot is ready');
  map.flush();
  assert(map.host() === host && host.childNodes[0] === map.iframe && standalone.iframe.parentNode === null &&
    standalone.iframe.srcWrites.length === 0 && host.dataset.mapActive === 'true' && !host.hidden &&
    host.style.left === '33px' && host.style.top === '481px',
  'A fresh standalone Contact route should discard its blank duplicate and reuse the first map in its new slot');
  assert(map.viewport.firstChild === standalone.body && map.viewport.childNodes.indexOf(host) > 0,
    'New route content should remain before the persistent iframe in DOM reading and keyboard order');

  const otherBody = element('main');
  map.setState({ category: 'projects', body: otherBody });
  map.replaceRouteBody(otherBody);
  map.flush();
  assert(!host.dataset.mapActive && !host.hidden && host.inert && map.iframe.isConnected,
    'Navigating from Contact to an individual project should retain an inactive connected map');

  const returningHome = createContactBody();
  map.setState({ category: 'contact', body: returningHome.body, home: true, view: 'overview' });
  map.replaceRouteBody(returningHome.body);
  map.flush();
  assert(host.dataset.mapActive === 'true' && !host.hidden && map.iframe.srcWrites.length === 1 && map.iframe.loadedDetachments === 0 &&
    returningHome.iframe.srcWrites.length === 0 && returningHome.iframe.parentNode === null,
  'Returning to Contact through a newly rendered homepage should preserve the original loaded iframe across route replacements');

  const incompleteContact = createContactBody();
  incompleteContact.iframe.remove();
  map.setState({ body: incompleteContact.body });
  map.replaceRouteBody(incompleteContact.body);
  map.flush();
  assert(!host.dataset.mapActive && !host.hidden && host.inert && map.iframe.isConnected,
    'A Contact placeholder without a map source should keep the previous map hidden instead of showing stale map content');
  map.setState({ body: returningHome.body });
  map.replaceRouteBody(returningHome.body);
  map.flush();
  assert(host.dataset.mapActive === 'true' && !host.hidden && map.viewport.firstChild === returningHome.body && map.iframe.srcWrites.length === 1,
    'Restoring a valid Contact body should restore map visibility and reading order without reloading it');

  map.document.emit('site:route-unmounted');
  assert(!host.dataset.mapActive && !host.hidden && host.inert,
    'Route cleanup should immediately make the retained map hidden and unfocusable');
  map.root.classList.add('site-frame--moving');
  map.viewport.inert = true;
  map.refresh();
  assert(host.dataset.mapActive === 'true' && !host.hidden && host.inert &&
    host.style.width === '940px' && host.style.height === '300px',
  'Refreshing a valid Contact slot during the next transition should restore its paint and dimensions while leaving interaction disabled');
  map.root.classList.remove('site-frame--moving');
  map.viewport.inert = false;
  map.refresh();
  map.rerun();
  map.refresh();
  assert(host.dataset.mapActive === 'true' && !host.hidden && !host.inert &&
    map.iframe.srcWrites.length === 1 && map.iframe.loadedDetachments === 0 &&
    map.viewport.childNodes.filter((node) => node.hasAttribute('data-persistent-contact-map')).length === 1,
  'Repeated runtime initialization and transition completion should preserve exactly one loaded map browsing context');
};
