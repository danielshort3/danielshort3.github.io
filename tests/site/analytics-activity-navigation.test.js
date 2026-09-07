'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/analytics/activity-events.js'), 'utf8');

class TestElement {
  constructor(selectors = [], parent = null) {
    this.selectors = new Set(selectors);
    this.parent = parent;
    this.children = [];
    if (parent) parent.children.push(this);
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = {};
    this.isConnected = true;
    this.hidden = false;
    this.value = '';
  }

  addEventListener(type, listener) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(listener);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  dispatch(type, detail = {}) {
    for (const listener of this.listeners.get(type) || []) listener({ type, target: this, ...detail });
  }

  matches(selector) {
    return selector.split(',').some((part) => this.selectors.has(part.trim()));
  }

  closest(selector) {
    return this.matches(selector) ? this : this.parent?.closest(selector) || null;
  }

  contains(element) {
    return element === this || this.children.some((child) => child.contains(element));
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    const matches = [];
    for (const child of this.children) {
      if (child.matches(selector)) matches.push(child);
      matches.push(...child.querySelectorAll(selector));
    }
    return matches;
  }

  getAttribute(name) {
    return this.attributes[name] || null;
  }

  hasAttribute(name) {
    return Object.hasOwn(this.attributes, name);
  }

  disconnect() {
    this.isConnected = false;
    this.children.forEach((child) => child.disconnect());
  }
}

function createHarness(initialPath = '/', initialConsent = true) {
  let consent = initialConsent;
  let main = new TestElement(['main']);
  let nextTimer = 1;
  const timers = new Map();
  const observers = [];
  const events = [];
  const document = new TestElement();
  document.visibilityState = 'visible';
  document.body = { dataset: { page: 'home' }, classList: { contains: () => false } };
  document.querySelector = (selector) => selector === 'main' ? main : main.querySelector(selector);
  document.querySelectorAll = (selector) => main.querySelectorAll(selector);
  const window = new TestElement();
  window.location = new URL(initialPath, 'https://www.danielshort.me');
  window.consentAPI = { get: () => ({ analytics: consent }) };
  window.gaEvent = (name, params) => {
    if (!consent) return false;
    events.push({ name, params });
    return true;
  };
  window.setTimeout = (callback, delay) => {
    const id = nextTimer++;
    timers.set(id, { callback, delay });
    return id;
  };
  window.clearTimeout = (id) => timers.delete(id);
  window.trackProjectView = (id) => events.push({ name: 'project_view', params: { project_id: id } });
  class IntersectionObserver {
    constructor(callback) {
      this.callback = callback;
      this.disconnected = false;
      observers.push(this);
    }

    observe(target) { this.target = target; }
    disconnect() { this.disconnected = true; }
    visible() { this.callback([{ target: this.target, isIntersecting: true, intersectionRatio: 1 }]); }
  }
  const harness = {
    window, document, events, timers, observers,
    main: () => main,
    start: () => vm.runInNewContext(source, {
      window, document, URLSearchParams, IntersectionObserver,
      Element: TestElement, HTMLInputElement: TestElement
    }, { filename: 'activity-events.js' }),
    route(url, nextMain = new TestElement(['main'])) {
      if (main !== nextMain) main.disconnect();
      main = nextMain;
      window.location = new URL(url, window.location);
      window.dispatch('site:route-complete', { detail: { url: window.location.href } });
    },
    setConsent(value) {
      consent = value;
      window.dispatch('consent-changed', { detail: { analytics: value } });
    },
    flush(delay) {
      for (const [id, timer] of Array.from(timers)) {
        if (timer.delay !== delay || !timers.has(id)) continue;
        timers.delete(id);
        timer.callback();
      }
    },
    click(target) { document.dispatch('click', { target }); },
    ofType(name) { return events.filter((event) => event.name === name); }
  };
  return harness;
}

function addDirectory(main) {
  const root = new TestElement(['[data-portfolio-workbench]'], main);
  root.dataset.directoryWorkbench = 'portfolio';
  const results = new TestElement(['[data-portfolio-results]'], root);
  results.scrollHeight = 1000;
  results.clientHeight = 200;
  results.scrollTop = 500;
  const input = new TestElement(['[data-portfolio-search]'], root);
  return { root, results, input };
}

function addProjectResource(main) {
  const link = new TestElement(['.project-link[href]'], main);
  link.attributes.href = 'https://github.com/example/project';
  return link;
}

const context = createHarness();
context.start();
context.route('/portfolio/alpha');
context.click(addProjectResource(context.main()));
context.route('/portfolio/beta');
context.click(addProjectResource(context.main()));
assert.deepEqual(context.ofType('select_content').map((event) => event.params.content_id), ['alpha', 'beta'],
  'project resource clicks must read the current route after each soft navigation');

context.route('/tools/image-optimizer');
context.document.dispatch('tools:run-start', { detail: { toolId: 'image-optimizer', action: 'optimize' } });
context.route('/tools/text-compare');
context.document.dispatch('tools:run-complete', { detail: { toolId: 'image-optimizer' } });
context.document.dispatch('tools:run-start', { detail: { toolId: 'text-compare', action: 'compare' } });
context.document.dispatch('tools:run-complete', { detail: { toolId: 'text-compare' } });
assert.deepEqual(context.ofType('tool_run_complete').map((event) => event.params.tool_id), ['text-compare'],
  'tool runs use current tool context and cannot complete a run from a previous route');

for (const game of ['stormbreak', 'roulette']) {
  context.route(`/games/${game}`);
  context.document.dispatch('pointerdown', { target: context.main(), pointerType: 'mouse' });
  context.document.dispatch('pointerdown', { target: context.main(), pointerType: 'mouse' });
}
assert.deepEqual(context.ofType('game_session_start').map((event) => event.params.game_id), ['stormbreak', 'roulette'],
  'game session starts are emitted once per current game route');

const directory = createHarness('/portfolio');
const oldDirectory = addDirectory(directory.main());
directory.start();
directory.window.dispatch('site:route-complete');
directory.window.dispatch('site:route-complete');
assert.equal(oldDirectory.results.listeners.get('scroll').size, 1, 'repeated route completion must not duplicate scroll listeners');
oldDirectory.results.dispatch('scroll');
directory.window.dispatch('site:route-complete');
oldDirectory.results.dispatch('scroll');
assert.equal(directory.ofType('directory_depth_reached').length, 1, 'refreshing the same route retains its once-only depth milestone');
oldDirectory.input.value = 'private search';
directory.document.dispatch('input', { target: oldDirectory.input });
const nextMain = new TestElement(['main']);
const nextDirectory = addDirectory(nextMain);
directory.route('/tools', nextMain);
assert.equal(oldDirectory.results.listeners.get('scroll').size, 0, 'old directory scroll listeners are disposed');
directory.flush(650);
assert.equal(directory.ofType('directory_search').length, 0, 'a delayed search from a departed route must be canceled');
oldDirectory.results.dispatch('scroll');
nextDirectory.results.dispatch('scroll');
assert.equal(directory.ofType('directory_depth_reached').length, 2, 'newly mounted directories receive fresh depth tracking');

const proof = createHarness('/portfolio/alpha', false);
new TestElement(['.project-star'], proof.main());
proof.start();
assert.equal(proof.observers.length, 0, 'case-study dwell time must start after consent');
proof.setConsent(true);
assert.equal(proof.observers.length, 1, 'granting consent observes an already visible case study');
const oldObserver = proof.observers[0];
oldObserver.visible();
const betaMain = new TestElement(['main']);
new TestElement(['.project-star'], betaMain);
proof.route('/portfolio/beta', betaMain);
assert.equal(oldObserver.disconnected, true, 'route changes disconnect the previous case-study observer');
oldObserver.visible();
proof.flush(5000);
assert.equal(proof.ofType('case_study_engaged').length, 0, 'stale observer callbacks and timers cannot attribute old proof to a new route');
const betaObserver = proof.observers.at(-1);
betaObserver.visible();
proof.setConsent(false);
assert.equal(betaObserver.disconnected, true, 'revoking consent disconnects the active proof observer');
proof.flush(5000);
assert.equal(proof.ofType('case_study_engaged').length, 0, 'revocation cancels pending engagement');
proof.setConsent(true);
proof.observers.at(-1).visible();
proof.flush(5000);
assert.deepEqual(proof.ofType('case_study_engaged').map((event) => event.params.project_id), ['beta'],
  'regranting consent re-arms dwell tracking for the current case study');
proof.window.dispatch('site:route-complete');
proof.observers.at(-1).visible();
proof.flush(5000);
assert.equal(proof.ofType('case_study_engaged').length, 1, 'same-route refresh must not duplicate completed engagement');

console.log('Analytics activity navigation tests passed.');
