'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const analyticsCode = fs.readFileSync(path.join(root, 'js/analytics/ga4-events.js'), 'utf8');
const homeCode = fs.readFileSync(path.join(root, 'js/home/category-accordion.js'), 'utf8');

function eventTarget() {
  const handlers = new Map();
  return {
    addEventListener(name, fn) {
      if (!handlers.has(name)) handlers.set(name, new Set());
      handlers.get(name).add(fn);
    },
    removeEventListener(name, fn) { handlers.get(name)?.delete(fn); },
    dispatchEvent(event) {
      [...(handlers.get(event.type) || [])].forEach(fn => fn(event));
      return true;
    }
  };
}

function harness({ url = '/', page = 'home', audience = 'personal', consent = true, embedded = false, enabled = true } = {}) {
  const window = eventTarget();
  window.location = new URL(url, 'https://www.danielshort.me/');
  window.self = window;
  window.top = embedded ? { location: { origin: window.location.origin } } : window;
  window.dataLayer = [];
  window.SiteAnalyticsEnvironment = { enabled };
  window.consentAPI = { get: () => ({ categories: { analytics: consent } }) };
  window.innerHeight = 600;
  window.scrollY = 0;
  window.getComputedStyle = element => ({ overflowY: element.overflowY || 'visible' });
  const document = Object.assign(eventTarget(), {
    title: 'Daniel Short',
    baseURI: window.location.href,
    readyState: 'complete',
    body: { dataset: { page, audience } },
    documentElement: { scrollHeight: 1600 },
    querySelector: () => null,
    querySelectorAll: () => []
  });
  const context = vm.createContext({ window, document, URL, URLSearchParams, CustomEvent, console, getComputedStyle: () => ({ visibility: 'visible' }) });
  vm.runInContext(analyticsCode, context);
  const complete = (url, title = 'Route title', source = 'router') => {
    const previousUrl = window.location.href;
    window.location = new URL(url, window.location);
    document.title = title;
    window.dispatchEvent(new CustomEvent('site:route-complete', { detail: { url: window.location.href, previousUrl, title, source } }));
  };
  const pageviews = () => window.dataLayer.filter(item => item.event === 'virtual_page_view');
  return { window, document, context, complete, pageviews };
}

async function run() {
  const consentEntry = fs.readFileSync(path.join(root, 'build/entries/site-consent.entry.js'), 'utf8');
  assert(consentEntry.indexOf("import '../../js/analytics/ga4-events.js'") < consentEntry.indexOf("import '../../js/privacy/consent_manager.js'"),
    'analytics must bind its context listener before consent can start GTM');
  for (const savedConsent of [false, true]) {
    const startup = harness({ url: '/contact?audience=tourism', page: 'contact', audience: 'tourism', consent: false });
    delete startup.window.consentAPI;
    const values = new Map(savedConsent ? [['pcz_consent_v1', JSON.stringify({ categories: { analytics: true } })]] : []);
    const scripts = new Map();
    startup.context.localStorage = {
      getItem: key => values.get(key) || null,
      setItem: (key, value) => values.set(key, value),
      removeItem: key => values.delete(key)
    };
    startup.context.navigator = { onLine: true, language: 'en-US' };
    startup.context.location = startup.window.location;
    startup.document.readyState = savedConsent ? 'complete' : 'loading';
    startup.document.body.removeAttribute = () => {};
    startup.document.getElementById = id => id.startsWith('pcz-consent-') ? {} : scripts.get(id) || null;
    startup.document.createElement = () => ({});
    startup.document.head = { appendChild: script => scripts.set(script.id, script) };
    vm.runInContext(fs.readFileSync(path.join(root, 'js/privacy/config.js'), 'utf8'), startup.context);
    vm.runInContext(fs.readFileSync(path.join(root, 'js/privacy/consent_manager.js'), 'utf8'), startup.context);
    if (!savedConsent) startup.window.consentAPI.set({ analytics: true });
    const queued = startup.window.dataLayer;
    const startIndex = queued.findIndex(item => item.event === 'gtm.js');
    const contextIndex = queued.findIndex(item => item.page_id === 'contact' && item.audience === 'tourism');
    assert(startIndex >= 0 && contextIndex >= 0 && contextIndex < startIndex,
      `${savedConsent ? 'saved' : 'new'} consent must queue page context before the GTM startup event`);
    assert.equal(queued.filter(item => item.event === 'gtm.js').length, 1);
    assert.equal(startup.pageviews().length, 0, 'starting GTM never duplicates the initial view');
  }

  const env = harness();
  assert.equal(env.pageviews().length, 0, 'the Google tag owns the initial page view');
  assert.equal(env.window.dataLayer[0].page_id, 'home', 'initial Google tag variables receive current context');
  env.complete('/#projects');
  env.complete('/#tools');
  assert.equal(env.pageviews().length, 0, 'homepage section changes are not page views');
  env.complete('/portfolio?search=private%40example.com&token=private#work', 'Portfolio');
  assert.equal(env.pageviews().length, 1);
  let event = env.pageviews()[0];
  assert.equal(event.page_id, 'portfolio', 'the homepage library URL overrides the persistent home body');
  assert.equal(event.page_location, 'https://www.danielshort.me/portfolio');
  assert.equal(event.page_referrer, 'https://www.danielshort.me/');
  assert.equal(event.page_title, 'Portfolio');
  env.complete('/portfolio?sort=recent#different');
  assert.equal(env.pageviews().length, 1, 'query filters and hashes do not duplicate semantic routes');
  env.window.gaEvent('select_content', { item_id: 'project' });
  assert.equal(env.window.dataLayer.at(-1).page_id, 'portfolio', 'custom events use the same route-aware context');
  env.window.gaEvent('game_milestone', { game_id: 'stormbreak', milestone_id: 'first_victory', input_type: 'keyboard', private_detail: 'private' });
  assert.equal(env.window.dataLayer.at(-1).activity_label, 'stormbreak');
  assert.equal(env.window.dataLayer.at(-1).activity_detail, 'first_victory', 'GTM curated activity fields retain the milestone identifier');
  assert(!JSON.stringify(env.window.dataLayer.at(-1)).includes('private'), 'milestone context remains allowlisted');
  env.complete('/portfolio/chatbotLora', 'Chatbot');
  assert.equal(env.pageviews().at(-1).page_id, 'project');
  env.complete('/portfolio', 'Portfolio', 'pop');
  assert.equal(env.pageviews().length, 3, 'Back returning to a previous route counts once');
  assert.equal(env.pageviews().at(-1).page_referrer, 'https://www.danielshort.me/portfolio/chatbotLora');

  for (const [url, audience, pageId] of [
    ['/portfolio?audience=analytics&token=private', 'analytics', 'portfolio'],
    ['/contact?audience=data_science', 'data-science', 'contact'],
    ['/portfolio?audience=tourism-analytics', 'tourism', 'portfolio'],
    ['/data-science', 'data-science', 'data-science'],
    ['/resume-tourism-pdf', 'tourism', 'resume-tourism-pdf'],
    ['/professional/analytics/contact', 'analytics', 'contact'],
    ['/contact?mode=professional', 'analytics', 'contact']
  ]) {
    const audienceEnv = harness();
    audienceEnv.complete(url);
    const measured = audienceEnv.pageviews()[0];
    assert.equal(measured.audience, audience, url);
    assert.equal(measured.page_id, pageId, url);
    assert(!JSON.stringify(measured).includes('private'), 'only approved audience parameters may survive');
  }

  const variant = harness({ url: '/portfolio', page: 'portfolio' });
  variant.complete('/portfolio?audience=analytics');
  variant.complete('/portfolio?audience=analytics&sort=date');
  variant.complete('/portfolio?audience=personal');
  assert.equal(variant.pageviews().length, 2, 'audience variants count, same-variant filter changes do not');
  variant.complete('/contact', 'Contact private@example.com');
  assert.equal(variant.pageviews().at(-1).page_title, 'contact', 'unsafe titles fall back to the safe page identifier');
  variant.complete('/portfolio/private%40example.com');
  assert.equal(variant.pageviews().length, 3, 'encoded email paths cannot enter measured URLs');

  const denied = harness({ consent: false });
  denied.complete('/portfolio');
  assert.equal(denied.window.dataLayer.length, 0);
  denied.window.dispatchEvent(new CustomEvent('consent-changed', { detail: { categories: { analytics: true } } }));
  denied.complete('/portfolio');
  assert.equal(denied.pageviews().length, 0, 'granting consent does not replay earlier route changes');
  denied.complete('/contact');
  assert.equal(denied.pageviews().length, 1);
  denied.window.dispatchEvent(new CustomEvent('consent-changed', { detail: { analytics: false } }));
  denied.complete('/tools');
  assert.equal(denied.pageviews().length, 1, 'withdrawal immediately stops page views');
  for (const options of [{ embedded: true }, { enabled: false }]) {
    const gated = harness(options);
    gated.complete('/portfolio');
    assert.equal(gated.window.gaEvent('select_content', { item_id: 'test' }), false);
    assert.equal(gated.window.dataLayer.length, 0, 'embedded and disabled environments never queue analytics');
  }

  const scroll = harness();
  scroll.window.scrollY = 600;
  scroll.window.dispatchEvent(new Event('scroll'));
  scroll.window.dispatchEvent(new Event('scroll'));
  scroll.complete('/contact');
  scroll.window.dispatchEvent(new Event('scroll'));
  assert.equal(scroll.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 2, 'scroll milestone resets for the new route');

  const framed = harness();
  const viewport = { isConnected: true, clientHeight: 400, scrollHeight: 1400, scrollTop: 499, overflowY: 'auto' };
  framed.window.SiteFrame = { viewport: () => viewport };
  const depths = () => framed.window.dataLayer.filter(item => item.event === 'scroll_depth');
  const frameScroll = target => framed.document.dispatchEvent({ type: 'scroll', target });
  framed.window.scrollY = 900;
  framed.window.dispatchEvent({ type: 'scroll', target: framed.document });
  frameScroll({ scrollTop: 900, scrollHeight: 1000, clientHeight: 100 });
  frameScroll(viewport);
  assert.equal(depths().length, 0, 'document and nested tool scrolls cannot stand in for active frame depth');
  viewport.scrollTop = 500;
  frameScroll(viewport);
  frameScroll(viewport);
  assert.equal(depths().length, 1, 'the frame sends 50 percent exactly once');
  framed.complete('/?sort=recent#tools');
  frameScroll(viewport);
  assert.equal(depths().length, 1, 'hash and filter changes do not reset the milestone');
  framed.complete('/portfolio');
  frameScroll(viewport);
  assert.equal(depths().length, 2, 'semantic route completion resets frame depth');
  assert.equal(depths()[1].page_id, 'portfolio');

  framed.complete('/contact');
  viewport.overflowY = 'visible';
  frameScroll(viewport);
  assert.equal(depths().length, 2, 'a frame that no longer owns scrolling cannot emit a milestone');
  framed.window.dispatchEvent({ type: 'scroll', target: framed.document });
  assert.equal(depths().length, 3, 'responsive document scrolling is measured instead');
  framed.complete('/tools');
  framed.document.documentElement.scrollHeight = 600;
  framed.window.dispatchEvent({ type: 'scroll', target: framed.document });
  assert.equal(depths().length, 3, 'a page without scroll range has no scroll milestone');

  const duringNavigation = harness();
  duringNavigation.window.scrollY = 600;
  duringNavigation.window.SiteNavigation = { isNavigating: () => true };
  duringNavigation.window.dispatchEvent(new Event('scroll'));
  duringNavigation.window.SiteNavigation.isNavigating = () => false;
  duringNavigation.window.location = new URL('/contact', duringNavigation.window.location);
  duringNavigation.window.dispatchEvent(new Event('scroll'));
  assert.equal(duringNavigation.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 0,
    'navigation geometry and provisional routes are not engagement');
  duringNavigation.complete('/contact');
  duringNavigation.window.dispatchEvent(new Event('scroll'));
  assert.equal(duringNavigation.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 1);

  for (const options of [{ consent: false }, { enabled: false }, { embedded: true }]) {
    const gatedScroll = harness(options);
    const gatedViewport = { ...viewport, overflowY: 'auto' };
    gatedScroll.window.SiteFrame = { viewport: () => gatedViewport };
    gatedScroll.document.dispatchEvent({ type: 'scroll', target: gatedViewport });
    gatedScroll.window.scrollY = 900;
    gatedScroll.window.dispatchEvent(new Event('scroll'));
    assert.equal(gatedScroll.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 0,
      'scroll collection respects consent, local environment, and embedding');
  }
  const scrollConsent = harness({ consent: false });
  scrollConsent.window.scrollY = 600;
  scrollConsent.window.dispatchEvent(new Event('scroll'));
  scrollConsent.window.dispatchEvent(new CustomEvent('consent-changed', { detail: { analytics: true } }));
  assert.equal(scrollConsent.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 0,
    'granting consent does not replay earlier scrolling');
  scrollConsent.window.dispatchEvent(new Event('scroll'));
  scrollConsent.window.dispatchEvent(new CustomEvent('consent-changed', { detail: { analytics: false } }));
  scrollConsent.complete('/contact');
  scrollConsent.window.dispatchEvent(new Event('scroll'));
  assert.equal(scrollConsent.window.dataLayer.filter(item => item.event === 'scroll_depth').length, 1,
    'a new consented scroll can count but withdrawal immediately stops collection');

  // Exercise the actual homepage controller, including its history restoration,
  // rather than relying only on synthetic route-completion events.
  const home = harness();
  const frameRoot = Object.assign(eventTarget(), { dataset: { frameCategory: 'about', frameView: 'overview' } });
  const item = { querySelector: selector => selector === '[data-home-library-view]' ? { dataset: { homeLibraryRendered: 'true' } } : null };
  const state = { title: 'Home', canonical: home.window.location.href, category: 'about', view: 'overview', items: new Map(['about', 'projects', 'tools', 'games'].map(id => [id, item])) };
  const homeViewport = { scrollTop: 0 };
  home.window.SiteFrame = {
    homeState: () => state,
    tabs: () => new Map(),
    root: () => frameRoot,
    viewport: () => homeViewport,
    showHome: async (category, view) => {
      state.category = category;
      state.view = view;
      frameRoot.dataset.frameCategory = category;
      frameRoot.dataset.frameView = view;
      return true;
    }
  };
  home.window.history = {
    state: {},
    pushState(value, title, url) { this.state = value; home.window.location = new URL(url); },
    replaceState(value, title, url) { this.state = value; home.window.location = new URL(url); }
  };
  vm.runInContext(homeCode, home.context);
  await new Promise(setImmediate);
  assert.equal(home.pageviews().length, 0);
  frameRoot.dispatchEvent({ type: 'click', preventDefault() {}, target: { closest: selector => selector === '[data-home-library-open]' ? { dataset: { homeLibraryOpen: 'projects' } } : null } });
  await new Promise(setImmediate);
  assert.equal(home.pageviews().length, 1, 'homepage library click completes a measured route');
  assert.equal(home.pageviews()[0].page_title, 'Data & Machine Learning Portfolio | Daniel Short');
  home.window.location = new URL('https://www.danielshort.me/#projects');
  home.window.history.state = { homePanel: 'projects', homeView: 'overview' };
  home.window.dispatchEvent(new Event('popstate'));
  await new Promise(setImmediate);
  assert.equal(home.pageviews().length, 2, 'homepage browser Back completes a measured route');
  assert.equal(home.pageviews()[1].page_id, 'home');
  home.window.location = new URL('https://www.danielshort.me/#tools');
  home.window.dispatchEvent(new Event('hashchange'));
  await new Promise(setImmediate);
  assert.equal(home.pageviews().length, 2, 'homepage category browsing remains uncounted');
  console.log('Analytics route measurement tests passed.');
}

run().catch(error => { console.error(error); process.exitCode = 1; });
