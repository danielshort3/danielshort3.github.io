'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..', '..');
const TRANSITION_KEY = 'sitePageTransition';
const FOCUS_KEY = 'sitePageTransitionFocus';

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

function createClassList(initial = []) {
  const values = new Set(initial);
  return {
    add(...names) {
      names.forEach((name) => values.add(name));
    },
    contains(name) {
      return values.has(name);
    },
    remove(...names) {
      names.forEach((name) => values.delete(name));
    },
    toggle(name, force) {
      const enabled = typeof force === 'boolean' ? force : !values.has(name);
      if (enabled) values.add(name);
      else values.delete(name);
      return enabled;
    },
    values
  };
}

function createScheduler() {
  let now = 0;
  let nextId = 1;
  const timers = [];
  const frames = [];

  function clearTimeout(timerId) {
    const timer = timers.find((entry) => entry.id === timerId);
    if (timer) timer.cancelled = true;
  }

  function setTimeout(callback, delay = 0) {
    const id = nextId;
    nextId += 1;
    timers.push({
      callback,
      cancelled: false,
      due: now + Math.max(0, Number(delay) || 0),
      id
    });
    return id;
  }

  function advanceBy(duration) {
    const target = now + Math.max(0, Number(duration) || 0);
    let safety = 200;
    while (safety > 0) {
      const next = timers
        .filter((entry) => !entry.cancelled && entry.due <= target)
        .sort((left, right) => left.due - right.due || left.id - right.id)[0];
      if (!next) break;
      next.cancelled = true;
      now = next.due;
      next.callback();
      safety -= 1;
    }
    now = target;
  }

  function flushFrames(limit = 40) {
    let remaining = limit;
    while (frames.length && remaining > 0) {
      const pending = frames.splice(0);
      pending.forEach(({ callback }) => callback(now));
      remaining -= 1;
    }
  }

  return {
    advanceBy,
    cancelAnimationFrame(frameId) {
      const frame = frames.find((entry) => entry.id === frameId);
      if (frame) frame.cancelled = true;
    },
    clearTimeout,
    flushFrames,
    nextTimerDelay() {
      const next = timers
        .filter((entry) => !entry.cancelled)
        .sort((left, right) => left.due - right.due || left.id - right.id)[0];
      return next ? Math.max(0, next.due - now) : null;
    },
    pendingTimerCount() {
      return timers.filter((entry) => !entry.cancelled).length;
    },
    requestAnimationFrame(callback) {
      const id = nextId;
      nextId += 1;
      frames.push({ callback, cancelled: false, id });
      return id;
    },
    setTimeout
  };
}

function createFocusTarget(document) {
  const attributes = new Map();
  return {
    attributes,
    addEventListener() {},
    closest() {
      return null;
    },
    focus(options) {
      this.focusedWith = options || null;
      document.activeElement = this;
    },
    focusedWith: null,
    hasAttribute(name) {
      return attributes.has(name);
    },
    matches() {
      return false;
    },
    removeAttribute(name) {
      attributes.delete(name);
    },
    setAttribute(name, value) {
      attributes.set(name, String(value));
    }
  };
}

function createLink(config = {}, personalShell = null) {
  const dataset = {};
  if (config.pageTransition !== undefined) dataset.pageTransition = config.pageTransition;
  if (config.personalTransition !== false) {
    dataset.personalTransition = config.personalTransition || 'detail';
  }
  return {
    dataset,
    closest(selector) {
      if (selector === 'a[href]') return config.nonAnchor ? null : this;
      if (selector === '[data-contact-modal-link]') return config.contactModal ? this : null;
      if (selector.includes('[data-home-accordion]')) {
        return config.personalSurface === false ? null : personalShell;
      }
      return null;
    },
    getAttribute(name) {
      if (name === 'href') return config.href === undefined ? '/tools/text-compare' : config.href;
      if (name === 'target') return config.target || '';
      return null;
    },
    hasAttribute(name) {
      return name === 'download' && Boolean(config.download);
    }
  };
}

function runPageTransitionRuntime(source, noJsSource, options = {}) {
  const currentUrl = new URL(options.url || 'https://www.danielshort.me/tools');
  const scheduler = createScheduler();
  const assigned = [];
  const appended = [];
  const navigationEvents = [];
  const documentListeners = new Map();
  const windowListeners = new Map();
  const stored = new Map(Object.entries(options.storage || {}));
  const htmlClassList = createClassList(options.htmlClasses || []);
  const bodyClassList = createClassList(options.bodyClasses || []);
  const rootDataset = { ...(options.rootDataset || {}) };
  const shell = options.personalShell === false ? null : {
    dataset: {},
    querySelector() {
      return null;
    }
  };
  const body = {
    classList: bodyClassList,
    dataset: {
      personalAccordionView: options.view || 'library',
      personalCategory: options.category || 'tools'
    }
  };
  const document = {
    activeElement: body,
    baseURI: currentUrl.href,
    body,
    documentElement: {
      classList: htmlClassList,
      dataset: rootDataset
    },
    fullscreenElement: options.fullscreen ? {} : null,
    head: {
      appendChild(node) {
        appended.push(node);
      }
    },
    pointerLockElement: options.pointerLock ? {} : null,
    readyState: 'complete',
    addEventListener(type, listener) {
      const listeners = documentListeners.get(type) || [];
      listeners.push(listener);
      documentListeners.set(type, listeners);
    },
    createElement() {
      return { dataset: {} };
    },
    createEvent() {
      return {
        initCustomEvent(type, bubbles, cancelable, detail) {
          this.type = type;
          this.detail = detail;
        }
      };
    },
    dispatchEvent(event) {
      navigationEvents.push(event);
    },
    querySelector(selector) {
      if (selector === '[data-home-accordion]') return options.homeRoot || null;
      if (selector === '[data-personal-accordion-shell]') return shell;
      if (selector.includes('dialog[open]')) return options.blockingLayer ? {} : null;
      if (options.focusTarget && (
        selector.includes('data-personal-accordion-view') ||
        selector.includes('data-home-library-view')
      )) return focusTarget;
      if (selector === 'meta[name="theme-color"]') return null;
      return null;
    }
  };
  const focusTarget = createFocusTarget(document);
  currentUrl.assign = (href) => assigned.push(String(href));
  currentUrl.replace = () => {};
  const history = {
    state: options.historyState || null,
    replaceState(state) {
      this.state = state;
    }
  };
  const sessionStorage = {
    getItem(key) {
      return stored.has(key) ? stored.get(key) : null;
    },
    removeItem(key) {
      stored.delete(key);
    },
    setItem(key, value) {
      stored.set(key, String(value));
    }
  };
  const window = {
    document,
    history,
    location: currentUrl,
    performance: {
      getEntriesByType() {
        return [{ type: options.navigationType || 'navigate' }];
      }
    },
    sessionStorage,
    addEventListener(type, listener) {
      const listeners = windowListeners.get(type) || [];
      listeners.push(listener);
      windowListeners.set(type, listeners);
    },
    cancelAnimationFrame: scheduler.cancelAnimationFrame,
    clearTimeout: scheduler.clearTimeout,
    matchMedia() {
      return { matches: Boolean(options.reducedMotion) };
    },
    requestAnimationFrame: scheduler.requestAnimationFrame,
    setTimeout: scheduler.setTimeout
  };
  const cssTransitionSupport = options.cssTransitionSupport ?? Boolean(options.nativeTransitions);
  const pageswapSupport = options.pageswapSupport ?? Boolean(options.nativeTransitions);
  const pagerevealSupport = options.pagerevealSupport ?? Boolean(options.nativeTransitions);
  if (pageswapSupport) window.onpageswap = null;
  if (pagerevealSupport) window.onpagereveal = null;
  class RuntimeCustomEvent {
    constructor(type, init) {
      this.type = type;
      this.detail = init?.detail;
    }
  }
  const context = {
    CSS: { supports: () => cssTransitionSupport },
    CustomEvent: RuntimeCustomEvent,
    Date,
    JSON,
    Number,
    Object,
    Set,
    URL,
    document,
    window
  };

  if (options.runNoJs) vm.runInNewContext(noJsSource, context);
  const afterNoJs = {
    classes: new Set(htmlClassList.values),
    dataset: { ...rootDataset }
  };
  vm.runInNewContext(source, context);

  const defaultLink = createLink(options.link || {}, shell);
  function dispatchDocument(type, event) {
    (documentListeners.get(type) || []).forEach((listener) => listener(event));
  }
  function dispatchWindow(type, event) {
    (windowListeners.get(type) || []).forEach((listener) => listener(event));
  }
  function click(overrides = {}) {
    const link = overrides.link || defaultLink;
    const event = {
      altKey: false,
      button: 0,
      ctrlKey: false,
      defaultPrevented: false,
      metaKey: false,
      preventDefaultCalls: 0,
      shiftKey: false,
      target: link,
      preventDefault() {
        this.defaultPrevented = true;
        this.preventDefaultCalls += 1;
      },
      ...overrides
    };
    if (overrides.contactHandled) event.__contactHandled = true;
    dispatchDocument('click', event);
    return event;
  }

  return {
    afterNoJs,
    appended,
    assigned,
    bodyClassList,
    click,
    document,
    focusTarget,
    history,
    htmlClassList,
    navigationEvents,
    pointerDown(link = defaultLink) {
      dispatchDocument('pointerdown', { target: link });
    },
    rootDataset,
    scheduler,
    stored,
    window,
    firePageshow(persisted = false) {
      dispatchWindow('pageshow', { persisted });
    },
    firePagereveal(viewTransition = null) {
      dispatchWindow('pagereveal', { viewTransition });
    }
  };
}

function transitionPayload(target, overrides = {}) {
  return JSON.stringify({
    category: 'tools',
    direction: 'forward',
    fromView: 'library',
    mode: 'personal',
    target,
    toView: 'detail',
    transport: 'fallback',
    ts: Date.now(),
    ...overrides
  });
}

function assertNotIntercepted(assert, pageTransitionJs, noJs, config, eventOverrides, label) {
  const runtime = runPageTransitionRuntime(pageTransitionJs, noJs, {
    link: config
  });
  const event = runtime.click(eventOverrides);
  assert(event.preventDefaultCalls === 0 && runtime.assigned.length === 0 &&
    !runtime.stored.has(TRANSITION_KEY), label);
}

module.exports = function runPageTransitionTests({ assert }) {
  const pageTransitionJs = read('js/navigation/page-transitions.js');
  const noJs = read('js/common/no-js.js');
  const transitionCss = read('css/components/page-transitions.css');

  assert(transitionCss.includes('#combined-header-nav > .nav {\n    view-transition-name: site-header;') &&
    transitionCss.includes('.mobile-site-masthead {\n      view-transition-name: site-header;') &&
    transitionCss.includes('#combined-header-nav > .nav {\n      view-transition-name: none;'),
  'desktop and mobile mastheads should each remain anchored during their matching viewport transitions');
  assert(transitionCss.includes(':root:has(.home-accordion.is-view-changing)::view-transition-group(personal-shell),') &&
    transitionCss.includes(':root:has(.home-accordion.is-view-changing)::view-transition-group(personal-panel),') &&
    transitionCss.includes(':root:has(.home-accordion.is-view-changing)::view-transition-group(personal-content)') &&
    !transitionCss.includes('scrollbar-gutter: stable;'),
  'same-document homepage transitions should keep snapshots at their settled geometry without introducing a temporary document gutter');

  const fallback = runPageTransitionRuntime(pageTransitionJs, noJs);
  fallback.pointerDown();
  const fallbackClick = fallback.click();
  const fallbackPayload = JSON.parse(fallback.stored.get(TRANSITION_KEY) || 'null');
  const exitDelay = fallback.scheduler.nextTimerDelay();
  assert(fallbackClick.defaultPrevented &&
    fallback.appended.some((node) => node.dataset.prefetch === 'page-transition') &&
    fallback.htmlClassList.contains('site-page-transition-out') &&
    fallback.bodyClassList.contains('site-page-transition-out') &&
    fallback.assigned.length === 0,
  'fallback personal navigation should prefetch, lock interaction, and cover the current page before assigning');
  assert(fallbackPayload &&
    fallbackPayload.target === 'https://www.danielshort.me/tools/text-compare' &&
    fallbackPayload.mode === 'personal' &&
    fallbackPayload.category === 'tools' &&
    fallbackPayload.fromView === 'library' &&
    fallbackPayload.toView === 'detail' &&
    fallbackPayload.direction === 'forward' &&
    fallbackPayload.transport === 'fallback',
  'fallback personal navigation should persist semantic transition context for the incoming document');
  assert(fallback.rootDataset.siteTransitionMode === 'personal' &&
    fallback.rootDataset.siteTransitionCategory === 'tools' &&
    fallback.rootDataset.siteTransitionDirection === 'forward',
  'fallback navigation should expose its semantic transition context to the outgoing CSS veil');
  assert(typeof exitDelay === 'number' && exitDelay >= 100 && exitDelay <= 300,
    'fallback navigation should allow a short bounded cover phase before document assignment');
  fallback.scheduler.advanceBy(exitDelay - 1);
  assert(fallback.assigned.length === 0 && fallback.htmlClassList.contains('site-page-transition-out'),
    'fallback navigation should keep the current document covered until the exit phase finishes');
  fallback.scheduler.advanceBy(1);
  assert(fallback.assigned[0] === 'https://www.danielshort.me/tools/text-compare',
    'fallback navigation should assign the intended clean URL only after the cover phase');

  [
    {
      category: 'projects',
      fromView: 'overview',
      href: '/portfolio',
      label: 'homepage to project library',
      toView: 'library',
      direction: 'forward',
      url: 'https://www.danielshort.me/#projects'
    },
    {
      category: 'projects',
      fromView: 'library',
      href: '/portfolio/babynames',
      label: 'project library to detail',
      toView: 'detail',
      direction: 'forward',
      url: 'https://www.danielshort.me/portfolio'
    },
    {
      category: 'projects',
      fromView: 'detail',
      href: '/portfolio',
      label: 'project detail to library',
      personalTransition: 'collapse',
      toView: 'library',
      direction: 'back',
      url: 'https://www.danielshort.me/portfolio/babynames'
    },
    {
      category: 'tools',
      fromView: 'library',
      href: '/tools/text-compare',
      label: 'tool library to detail',
      toView: 'detail',
      direction: 'forward',
      url: 'https://www.danielshort.me/tools'
    },
    {
      category: 'games',
      fromView: 'library',
      href: '/games/stormbreak',
      label: 'game library to detail',
      toView: 'detail',
      direction: 'forward',
      url: 'https://www.danielshort.me/games'
    },
    {
      category: 'contact',
      fromView: 'overview',
      href: '/contact',
      label: 'homepage to contact detail',
      toView: 'detail',
      direction: 'forward',
      url: 'https://www.danielshort.me/#contact'
    },
    {
      category: 'about',
      fromView: 'overview',
      href: '/privacy',
      label: 'homepage to utility detail',
      toView: 'detail',
      direction: 'forward',
      url: 'https://www.danielshort.me/#about'
    },
    {
      category: 'tools',
      destinationCategory: 'games',
      fromView: 'library',
      href: '/games',
      label: 'cross-category library navigation',
      toView: 'library',
      direction: 'cross',
      url: 'https://www.danielshort.me/tools'
    }
  ].forEach((route) => {
    const runtime = runPageTransitionRuntime(pageTransitionJs, noJs, {
      category: route.category,
      link: {
        href: route.href,
        personalTransition: route.personalTransition || 'detail'
      },
      url: route.url,
      view: route.fromView
    });
    runtime.click();
    const payload = JSON.parse(runtime.stored.get(TRANSITION_KEY) || 'null');
    assert(payload?.mode === 'personal' &&
      payload.category === (route.destinationCategory || route.category) &&
      payload.fromView === route.fromView && payload.toView === route.toView &&
      payload.direction === route.direction,
    `${route.label} should produce the correct semantic fallback descriptor`);
  });

  const rapid = runPageTransitionRuntime(pageTransitionJs, noJs);
  const firstRapidClick = rapid.click();
  const timersAfterFirstClick = rapid.scheduler.pendingTimerCount();
  const secondRapidClick = rapid.click();
  assert(firstRapidClick.defaultPrevented && secondRapidClick.defaultPrevented &&
    rapid.scheduler.pendingTimerCount() === timersAfterFirstClick &&
    rapid.navigationEvents.filter((event) => event.type === 'site:navigation-start').length === 1,
  'rapid repeated clicks should share one navigation lock, one event, and one pending assignment');
  rapid.scheduler.advanceBy(300);
  assert(rapid.assigned.length === 1,
    'rapid repeated clicks should never commit the same navigation more than once');

  const nativeRuntime = runPageTransitionRuntime(pageTransitionJs, noJs, {
    nativeTransitions: true
  });
  const nativeClick = nativeRuntime.click();
  assert(!nativeClick.defaultPrevented && nativeRuntime.assigned.length === 0 &&
    nativeRuntime.navigationEvents.some((event) => event.type === 'site:navigation-start') &&
    nativeRuntime.stored.has(FOCUS_KEY) &&
    JSON.parse(nativeRuntime.stored.get(TRANSITION_KEY) || 'null')?.mode === 'personal' &&
    JSON.parse(nativeRuntime.stored.get(TRANSITION_KEY) || 'null')?.transport === 'native' &&
    !nativeRuntime.htmlClassList.contains('site-page-transition-out'),
  'complete cross-document View Transition support should leave navigation to the browser while retaining incoming semantics without arming the fallback exit veil');

  [
    { cssTransitionSupport: false, pageswapSupport: true, pagerevealSupport: true },
    { cssTransitionSupport: true, pageswapSupport: false, pagerevealSupport: true },
    { cssTransitionSupport: true, pageswapSupport: true, pagerevealSupport: false }
  ].forEach((support, index) => {
    const partial = runPageTransitionRuntime(pageTransitionJs, noJs, support);
    const event = partial.click();
    assert(event.defaultPrevented && partial.htmlClassList.contains('site-page-transition-out') &&
      partial.stored.has(TRANSITION_KEY),
    `partial native transition support case ${index + 1} should use the fully controlled fallback`);
  });

  const reduced = runPageTransitionRuntime(pageTransitionJs, noJs, {
    reducedMotion: true
  });
  const reducedClick = reduced.click();
  assert(reducedClick.defaultPrevented &&
    reduced.assigned[0] === 'https://www.danielshort.me/tools/text-compare' &&
    reduced.stored.has(FOCUS_KEY) && !reduced.stored.has(TRANSITION_KEY) &&
    !reduced.htmlClassList.contains('site-page-transition-out') &&
    reduced.scheduler.pendingTimerCount() === 0,
  'reduced-motion navigation should assign immediately, preserve focus intent, and store no visual transition payload');

  const nativeReduced = runPageTransitionRuntime(pageTransitionJs, noJs, {
    nativeTransitions: true,
    reducedMotion: true
  });
  const nativeReducedClick = nativeReduced.click();
  assert(nativeReducedClick.defaultPrevented &&
    nativeReduced.assigned[0] === 'https://www.danielshort.me/tools/text-compare' &&
    !nativeReduced.stored.has(TRANSITION_KEY) &&
    !nativeReduced.htmlClassList.contains('site-page-transition-out') &&
    !nativeReduced.rootDataset.siteTransitionMode,
  'native-capable reduced-motion navigation should bypass all visual transition state and assign immediately');

  const incomingUrl = 'https://www.danielshort.me/tools/text-compare';
  const incoming = runPageTransitionRuntime(pageTransitionJs, noJs, {
    runNoJs: true,
    storage: {
      [TRANSITION_KEY]: transitionPayload(incomingUrl)
    },
    url: incomingUrl,
    view: 'detail'
  });
  assert(incoming.afterNoJs.classes.has('site-page-transition-preload') &&
    incoming.afterNoJs.dataset.siteTransitionMode === 'personal' &&
    incoming.afterNoJs.dataset.siteTransitionCategory === 'tools' &&
    incoming.afterNoJs.dataset.siteTransitionDirection === 'forward',
  'the head bootstrap should synchronously preload the incoming veil and restore its semantic CSS context');
  assert(incoming.htmlClassList.contains('site-page-transition-preload') &&
    !incoming.htmlClassList.contains('site-page-transition-in'),
  'incoming hydration should retain the preload cover until the first animation frame');
  incoming.scheduler.flushFrames();
  assert(!incoming.htmlClassList.contains('site-page-transition-preload') &&
    incoming.htmlClassList.contains('site-page-transition-in'),
  'incoming hydration should atomically replace preload with the entry phase');
  incoming.firePageshow(false);
  assert(incoming.htmlClassList.contains('site-page-transition-in'),
  'an ordinary pageshow should not cancel an active incoming transition');

  const nativeIncoming = runPageTransitionRuntime(pageTransitionJs, noJs, {
    runNoJs: true,
    storage: {
      [TRANSITION_KEY]: transitionPayload(incomingUrl, { transport: 'native' })
    },
    url: incomingUrl,
    view: 'detail'
  });
  assert(nativeIncoming.afterNoJs.classes.has('site-page-transition-native-preload') &&
    !nativeIncoming.afterNoJs.classes.has('site-page-transition-preload') &&
    nativeIncoming.rootDataset.siteTransitionTransport === 'native',
  'native incoming navigation should arm only its native handoff cover in the parser-blocking bootstrap');
  const overlappingNativeClick = nativeIncoming.click({
    link: createLink({ href: '/tools/word-frequency' }, {})
  });
  assert(overlappingNativeClick.defaultPrevented &&
    !nativeIncoming.stored.has(TRANSITION_KEY),
  'an incoming native reveal should reject overlapping navigation until its handoff finishes');
  nativeIncoming.firePagereveal(null);
  nativeIncoming.scheduler.flushFrames();
  assert(!nativeIncoming.htmlClassList.contains('site-page-transition-native-preload') &&
    !nativeIncoming.rootDataset.siteTransitionMode &&
    !nativeIncoming.rootDataset.siteTransitionTransport,
  'the early pagereveal bridge should clear native handoff state without starting the fallback entry animation');

  const slowIncoming = runPageTransitionRuntime(pageTransitionJs, noJs, {
    runNoJs: true,
    storage: {
      [TRANSITION_KEY]: transitionPayload(incomingUrl, { ts: Date.now() - 10000 })
    },
    url: incomingUrl,
    view: 'detail'
  });
  assert(slowIncoming.afterNoJs.classes.has('site-page-transition-preload'),
    'a matching cold navigation should retain its cover beyond four seconds instead of flashing on arrival');

  const bfcache = runPageTransitionRuntime(pageTransitionJs, noJs, {
    focusTarget: true,
    historyState: { personalCategory: 'tools', personalView: 'library' },
    htmlClasses: ['site-page-transition-preload', 'site-page-transition-in'],
    url: 'https://www.danielshort.me/tools',
    view: 'library'
  });
  bfcache.firePageshow(true);
  bfcache.scheduler.flushFrames();
  assert(!bfcache.htmlClassList.contains('site-page-transition-preload') &&
    !bfcache.htmlClassList.contains('site-page-transition-in') &&
    !bfcache.htmlClassList.contains('site-page-transition-out') &&
    bfcache.focusTarget.focusedWith?.preventScroll === true,
  'BFCache restoration should clear stale transition phases and restore focus for the current personal state');

  const staleIncoming = runPageTransitionRuntime(pageTransitionJs, noJs, {
    runNoJs: true,
    storage: {
      [TRANSITION_KEY]: transitionPayload('https://www.danielshort.me/portfolio', {
        ts: Date.now() - 60000
      })
    },
    url: incomingUrl,
    view: 'detail'
  });
  assert(!staleIncoming.afterNoJs.classes.has('site-page-transition-preload') &&
    !staleIncoming.htmlClassList.contains('site-page-transition-preload'),
  'stale or target-mismatched transition storage should never leave the incoming page veiled');

  const neutral = runPageTransitionRuntime(pageTransitionJs, noJs, {
    link: {
      href: '/unclassified-page',
      personalSurface: false,
      personalTransition: false
    },
    personalShell: false,
    url: 'https://www.danielshort.me/unclassified-origin'
  });
  neutral.click();
  const neutralPayload = JSON.parse(neutral.stored.get(TRANSITION_KEY) || 'null');
  assert(neutralPayload?.mode === 'neutral' && neutralPayload.category === 'neutral' &&
    neutralPayload.fromView === 'detail' && neutralPayload.toView === 'detail' &&
    neutralPayload.direction === 'replace',
  'same-origin routes without personal semantics should use a neutral full-viewport transition payload');

  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { metaKey: true },
    'meta-click navigation should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { ctrlKey: true },
    'control-click navigation should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { shiftKey: true },
    'shift-click navigation should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { altKey: true },
    'alt-click navigation should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { button: 1 },
    'non-primary click navigation should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { defaultPrevented: true },
    'previously handled click navigation should remain untouched');
  assertNotIntercepted(assert, pageTransitionJs, noJs, {}, { contactHandled: true },
    'contact modal click navigation should remain untouched after its handler runs');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { target: '_blank' }, {},
    'new-tab targets should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { download: true }, {},
    'download links should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { pageTransition: 'false' }, {},
    'links that opt out of page transitions should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: '#tools' }, {},
    'hash-only links should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: '/tools#filters' }, {},
    'same-document hash changes should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: 'mailto:test@example.com' }, {},
    'mail links should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: 'tel:+15555550100' }, {},
    'telephone links should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: 'https://example.com/tools' }, {},
    'cross-origin links should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { href: '/documents/resume.pdf' }, {},
    'non-document assets should remain browser-native');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { contactModal: true }, {},
    'contact modal links should remain owned by the modal controller');
  assertNotIntercepted(assert, pageTransitionJs, noJs, { nonAnchor: true }, {},
    'clicks outside anchors should not begin navigation transitions');

  const blocked = runPageTransitionRuntime(pageTransitionJs, noJs, {
    blockingLayer: true
  });
  const blockedClick = blocked.click();
  assert(!blockedClick.defaultPrevented && !blocked.stored.has(TRANSITION_KEY),
    'an open interaction layer should retain control instead of starting a page transition underneath it');
};
