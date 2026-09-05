'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const audiences = require('../../js/common/audience-config');
const source = fs.readFileSync(path.join(__dirname, '../../js/navigation/header-breadcrumbs.js'), 'utf8');

module.exports = function runHeaderBreadcrumbsTests({ assert }) {
  const listeners = new Map();
  const animations = [];
  const motionListeners = [];
  const reducedMotion = { matches: false, addEventListener: (_, listener) => motionListeners.push(listener) };
  let page;
  class Element {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.children = [];
      this.dataset = {};
      this.attributes = {};
      this.style = { setProperty: (key, value) => { this.attributes[key] = value; } };
    }
    append(...nodes) {
      nodes.forEach((node) => this.children.push(...(node.tagName === '#FRAGMENT' ? node.children : [node])));
    }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name] || null; }
    getClientRects() { return [{}]; }
    animate(frames, options) {
      const animation = { frames, options, cancelled: false, cancel() { this.cancelled = true; } };
      animations.push(animation);
      return animation;
    }
  }
  const nav = new Element('nav');
  const list = new Element('ol');
  nav.querySelector = () => list;
  const document = {
    readyState: 'loading',
    body: { dataset: {} },
    title: '',
    createElement: (tag) => new Element(tag),
    createDocumentFragment: () => new Element('#fragment'),
    addEventListener(name, listener) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(listener);
    },
    querySelector(selector) {
      if (selector === '[data-header-breadcrumbs]') return nav;
      if (selector === '[data-site-route-manifest]') return { textContent: JSON.stringify(page.manifest) };
      if (selector === '[data-site-route-body], [data-personal-detail-content]') {
        return { querySelector: () => page.heading ? { textContent: page.heading } : null };
      }
      if (selector === '[data-site-route-toolbar] .personal-accordion__back') {
        return page.back ? { getAttribute: (name) => page.back[name] || null } : null;
      }
      return null;
    }
  };
  const window = {
    location: { href: 'https://www.danielshort.me/' },
    getSiteAudienceConfig: audiences.getAudience,
    matchMedia: () => reducedMotion,
    SiteMotion: { duration: () => reducedMotion.matches ? 0 : 160 },
    SiteFrame: { current: () => page.frame || null, root: () => null }
  };
  const context = vm.createContext({ window, document, URL, getComputedStyle: () => ({ getPropertyValue: () => '' }) });
  const emit = (name) => (listeners.get(name) || []).forEach((listener) => listener({ type: name }));
  const setPage = (options) => {
    page = { path: '/', category: 'about', view: 'detail', audience: 'personal', id: 'page', ...options };
    page.manifest = { path: page.manifestPath || page.path, id: page.id, category: page.category, view: page.view, navigation: page.hard ? 'hard' : 'soft' };
    document.body.dataset = {
      audience: page.audience, page: page.page || page.id, siteRouteCategory: page.category,
      siteRouteView: page.view, siteRouteNavigation: page.manifest.navigation
    };
    document.title = page.title || `${page.heading || 'Page'} | Daniel Short`;
    window.location.href = `https://www.danielshort.me${page.path}${page.query || ''}`;
  };
  const crumbs = () => list.children.map((item) => {
    const label = item.children[item.children.length - 1];
    return { label: label.textContent, href: label.href, tag: label.tagName, current: label.attributes['aria-current'], hard: label.dataset.navigation };
  });
  const expect = (labels, hrefs = []) => {
    const actual = crumbs();
    assert(JSON.stringify(actual.map((crumb) => crumb.label)) === JSON.stringify(labels), `breadcrumb labels should be ${labels.join(' > ')}`);
    assert(JSON.stringify(actual.slice(0, -1).map((crumb) => crumb.href)) === JSON.stringify(hrefs), 'ancestors should preserve their intended audience and hierarchy URLs');
    assert(actual.at(-1).tag === 'SPAN' && actual.at(-1).current === 'page' && !actual.at(-1).href, 'the current location should be a nonlink with aria-current=page');
    assert(actual.slice(0, -1).every((crumb) => crumb.tag === 'A' && !crumb.current), 'only ancestors should be native links');
    assert(list.children.slice(1).every((item) => item.children[0].attributes['aria-hidden'] === 'true'), 'decorative separators should be hidden from assistive technology');
  };

  setPage({ id: 'home', view: 'overview' });
  vm.runInContext(source, context);
  assert(!listeners.has('site:route-change'), 'initialization should wait for the DOM');
  document.readyState = 'complete';
  emit('DOMContentLoaded');
  assert(nav.hidden && list.children.length === 0, 'home overview should leave the header uncluttered');

  setPage({ id: 'home', path: '/games', manifestPath: '/', category: 'games', view: 'library', frame: { category: 'games', view: 'library' } });
  emit('home:category-change');
  expect(['Home', 'Games'], ['/']);
  assert(nav.attributes['--breadcrumb-accent'] === '#c94b0a', 'the breadcrumb should use the active category color');
  setPage({ path: '/games/stormbreak', category: 'games', heading: 'Stormbreak', title: 'Stormbreak: Idle Olympus | Daniel Short' });
  emit('site:route-change');
  expect(['Home', 'Games', 'Stormbreak'], ['/', '/games']);

  setPage({ path: '/baby-names-demo', category: 'projects', page: 'project-demo', title: 'Baby Name Predictor Demo | Daniel Short', back: { href: '/portfolio/babynames', 'aria-label': 'Back to Baby Name Predictor' } });
  emit('site:route-change');
  expect(['Home', 'Projects', 'Baby Name Predictor', 'Demo'], ['/', '/portfolio', '/portfolio/babynames']);
  page.back = { href: '/?view=library#projects', 'aria-label': 'Back to project library' };
  emit('site:route-change');
  expect(['Home', 'Projects', 'Baby Name Predictor Demo'], ['/', '/portfolio']);

  setPage({ path: '/portfolio', audience: 'analytics', category: 'projects', heading: 'Project Library' });
  emit('site:route-change');
  expect(['Home', 'Projects'], ['/analytics']);
  setPage({ path: '/portfolio/retailStore', audience: 'analytics', category: 'projects', heading: 'Store-Level Loss & Sales ETL' });
  emit('site:route-change');
  expect(['Home', 'Projects', 'Store-Level Loss & Sales ETL'], ['/analytics', '/portfolio?audience=analytics']);
  setPage({ path: '/resume-data-science', audience: 'data-science', category: 'resume', heading: 'Daniel Short' });
  emit('site:route-change');
  expect(['Home', 'Resume'], ['/data-science']);
  setPage({ path: '/resume-tourism-pdf', audience: 'tourism', category: 'resume', heading: 'PDF Preview' });
  emit('site:route-change');
  expect(['Home', 'Resume', 'PDF Preview'], ['/tourism', '/resume-tourism']);
  setPage({ path: '/contact', audience: 'analytics', category: 'contact', heading: "Let's Connect" });
  emit('site:route-change');
  expect(['Home', 'Contact'], ['/analytics']);
  setPage({ path: '/search', audience: 'analytics', query: '?audience=analytics&q=private-query', heading: 'Search' });
  emit('site:route-change');
  expect(['Home', 'Search'], ['/analytics']);
  assert(!JSON.stringify(crumbs()).includes('private-query'), 'search text must not enter the header or ancestor URLs');
  setPage({ path: '/privacy', heading: 'Privacy & Analytics' });
  emit('site:route-change');
  expect(['Home', 'Privacy & Analytics'], ['/']);

  for (const [id, heading] of [['background-remover', 'Background Remover'], ['transcribe', 'File Transcriber'], ['job-application-tracker', 'Job Application Tracker']]) {
    setPage({ path: `/tools/${id}`, category: 'tools', heading, hard: true });
    emit('site:route-change');
    expect(['Home', 'Tools', heading], ['/', '/tools']);
    assert(crumbs().slice(0, -1).every((crumb) => crumb.hard === 'hard'), 'standalone tool ancestors must retain hard navigation boundaries');
  }

  const committed = JSON.stringify(crumbs());
  setPage({ path: '/contact', category: 'contact', heading: "Let's Connect" });
  emit('site:navigation-start');
  emit('site:route-mounted');
  emit('site:route-navigation-error');
  assert(JSON.stringify(crumbs()) === committed, 'pending, vetoed and failed destinations must not replace the committed breadcrumb');
  emit('site:route-change');
  expect(['Home', 'Contact'], ['/']);
  assert(animations.every((animation) => animation.options.duration === 160 && animation.frames.every((frame) => !('opacity' in frame))), 'text updates should use the shared duration without flashing or hiding content');
  assert(animations.slice(0, -1).every((animation) => animation.cancelled), 'a rapid committed update should cancel the superseded text animation');
  reducedMotion.matches = true;
  motionListeners.forEach((listener) => listener());
  assert(animations.at(-1).cancelled, 'enabling reduced motion should settle an active breadcrumb animation');
  const animationCount = animations.length;
  setPage({ path: '/tools/example', category: 'tools', heading: 'A <B> & C' });
  emit('site:route-change');
  expect(['Home', 'Tools', 'A <B> & C'], ['/', '/tools']);
  assert(animations.length === animationCount, 'reduced motion should render the destination immediately');
  assert(list.children.at(-1).children.at(-1).children.length === 0, 'page headings should be inserted as text, never interpreted as markup');
  vm.runInContext(source, context);
  assert(listeners.get('site:route-change').length === 1 && listeners.get('home:category-change').length === 1, 'duplicate shell evaluation must not attach duplicate breadcrumb listeners');
  setPage({ path: '/analytics', audience: 'analytics', heading: 'Daniel Short' });
  emit('site:route-change');
  assert(nav.hidden && list.children.length === 0, 'professional landing pages should also hide the breadcrumb');
  setPage({ id: 'home', view: 'overview', category: 'contact', path: '/', query: '#contact' });
  emit('home:category-change');
  assert(nav.hidden && list.children.length === 0, 'returning to any inline overview tab should clear the trail');
};

if (require.main === module) {
  module.exports({ assert: require('assert') });
  process.stdout.write('Header breadcrumb tests passed.\n');
}
