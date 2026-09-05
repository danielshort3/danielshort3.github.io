'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const renderer = require('../../build/lib/cms-renderers.js');
const audienceApi = require('../../js/common/audience-config.js');
const root = path.resolve(__dirname, '..', '..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');

function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`Missing function ${name}`);
  let depth = 0;
  for (let index = source.indexOf('{', start); index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}' && --depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`Unclosed function ${name}`);
}

function createNode(document) {
  const attributes = new Map();
  const listeners = new Map();
  const classes = new Set();
  const node = {
    dataset: {},
    children: [],
    value: '',
    classList: {
      contains: (value) => classes.has(value),
      toggle(value, enabled) { if (enabled) classes.add(value); else classes.delete(value); }
    },
    setAttribute: (name, value) => attributes.set(name, String(value)),
    getAttribute: (name) => attributes.get(name) ?? null,
    addEventListener(name, callback) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(callback);
    },
    fire(name, properties = {}) {
      const event = { target: node, preventDefault() { this.defaultPrevented = true; }, ...properties };
      (listeners.get(name) || []).forEach((callback) => callback(event));
      return event;
    },
    appendChild(child) { child.parent = node; node.children.push(child); },
    remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); },
    focus() { document.activeElement = node; node.fire('focus'); },
    contains(target) { return target === node || node.children.some((child) => child.contains(target)); },
    querySelector(selector) {
      return node.queries?.[selector]
        || (selector === '[data-search-audience]' && node.children.find((child) => 'searchAudience' in child.dataset))
        || null;
    }
  };
  return node;
}

module.exports = function runCompactSiteHeaderTests({ assert }) {
  const settings = JSON.parse(read('content/site/settings.json'));
  const navigation = JSON.parse(read('content/site/navigation.json'));
  const footer = JSON.parse(read('content/site/footer.json'));
  ['personal', 'analytics', 'data-science', 'tourism'].forEach((key) => {
    const audience = audienceApi.getAudience(key);
    const html = renderer.renderHeader({ settings, navigation, audience });
    assert(html.includes('data-site-shell-header') && html.includes('class="brand"') && html.includes('role="search"'),
      `${key} should render the shared brand/search header`);
    assert(!/primary-menu|nav-toggle|nav-dropdown|nav-item|burger|data-resume-home-link|data-contact-modal-link/.test(html),
      `${key} should leave category navigation in the page rails rather than a header menu`);
    assert((html.match(/<a\b/g) || []).length === 1 && (html.match(/<button\b/g) || []).length === 1,
      `${key} header should initially expose its brand and search before a deeper route fills the breadcrumb`);
    assert(html.includes('aria-label="Breadcrumb" data-header-breadcrumbs hidden') && html.includes('data-header-breadcrumb-list'),
      `${key} header should reserve a hidden breadcrumb landmark for deeper pages`);
    assert(html.includes(`href="${key === 'personal' ? '/' : audience.homePath.replace(/^\//, '')}"`),
      `${key} brand should return to its own home`);
    assert(key === 'personal' ? !html.includes('name="audience"') : html.includes(`name="audience" value="${key}"`),
      `${key} search should preserve its audience without advertising other audiences`);
    const footerHtml = renderer.renderFooter({ footer, year: 2026, audience });
    assert(footerHtml.includes('footer--personal-compact') && footerHtml.includes('aria-label="Footer utility"') && footerHtml.includes('© 2026 Daniel Short'),
      `${key} should use the same compact utility footer and copyright`);
    assert(!/speed-dial|footer-identity|footer-nav|data-footer-realm|data-resume-home-link|portfolio\?audience=/.test(footerHtml),
      `${key} footer should not recreate a floating menu, category links, or professional discovery links`);
    ['Email', 'GitHub', 'Privacy', 'Cookie settings'].forEach((label) => {
      assert(footerHtml.includes(`>${label}</`), `${key} footer should retain ${label}`);
    });
  });

  const source = read('js/navigation/navigation.js');
  const document = {};
  Object.assign(document, createNode(document));
  document.createElement = () => createNode(document);
  const form = createNode(document);
  const input = createNode(document);
  const button = createNode(document);
  form.appendChild(input);
  form.appendChild(button);
  form.queries = { '.nav-search-input': input, '.nav-search-button': button };
  const frames = [];
  const media = { matches: true, addEventListener() {} };
  const context = vm.createContext({
    document,
    window: { matchMedia: () => media },
    requestAnimationFrame: (callback) => frames.push(callback),
    host: { querySelector: () => form },
    form
  });
  vm.runInContext(`${extractFunction(source, 'setupHeaderSearch')}\n${extractFunction(source, 'syncSearchAudience')}\nsetupHeaderSearch(host);`, context);
  assert(input.tabIndex === -1 && input.getAttribute('aria-hidden') === 'true' && button.getAttribute('aria-label') === 'Open search',
    'collapsed header search should expose only its button to keyboard and assistive technology');
  assert(form.fire('submit').defaultPrevented && form.classList.contains('is-expanded') && input.tabIndex === 0,
    'first search activation should reveal the search field without navigating');
  document.activeElement = input;
  form.fire('keydown', { key: 'Escape' });
  frames.splice(0).forEach((callback) => callback());
  assert(!form.classList.contains('is-expanded') && document.activeElement === button,
    'Escape during search entry should close it and cancel any stale queued input focus');
  form.fire('submit');
  frames.splice(0).forEach((callback) => callback());
  input.value = 'calendar';
  assert(document.activeElement === input && !form.fire('submit').defaultPrevented,
    'expanded search with a query should submit normally');
  // The search results page clears and blurs its header input before this form receives Escape.
  document.activeElement = document;
  form.fire('keydown', { key: 'Escape', target: input });
  assert(document.activeElement === button && !form.classList.contains('is-expanded'),
    'header Escape should restore its button even when a search-page input handler already blurred the field');
  form.fire('submit');
  frames.splice(0).forEach((callback) => callback());
  document.fire('pointerdown', { target: createNode(document) });
  assert(!form.classList.contains('is-expanded') && input.tabIndex === -1,
    'outside pointer input should dismiss search and remove the field from the tab order');
  vm.runInContext("syncSearchAudience(form, { key: 'analytics' }); syncSearchAudience(form, { key: 'tourism' });", context);
  assert(form.children.filter((child) => child.name === 'audience').length === 1 && form.querySelector('[data-search-audience]').value === 'tourism',
    'route changes should update one hidden search audience without duplicating it');
  vm.runInContext("syncSearchAudience(form, { key: 'personal' });", context);
  assert(!form.querySelector('[data-search-audience]'), 'returning to personal pages should remove the professional search parameter');

  const location = { href: 'https://www.danielshort.me/portfolio?audience=analytics', pathname: '/portfolio', search: '?audience=analytics' };
  const audienceContext = vm.createContext({
    location, URL, URLSearchParams,
    document: { body: { dataset: { audience: 'personal', siteRealm: 'personal' } } },
    window: { location, SITE_AUDIENCE_CONFIG: audienceApi, sessionStorage: { getItem: () => 'tourism', setItem() {} } }
  });
  vm.runInContext(extractFunction(source, 'getNavigationContext'), audienceContext);
  assert(vm.runInContext('getNavigationContext().activeAudience.key', audienceContext) === 'analytics',
    'an explicit professional route should control header context over generated personal markup and prior session state');
  location.pathname = '/'; location.search = ''; location.href = 'https://www.danielshort.me/';
  assert(vm.runInContext('getNavigationContext().entryHome', audienceContext) === '/' && vm.runInContext('getNavigationContext().activeAudience.key', audienceContext) === 'personal',
    'the public homepage should retain personal identity after visiting professional pages');

  assert(!/setupDropdown|setupMobileSiteDock|buildResumeNavHtml/.test(source + read('js/common/site-realm.js')),
    'runtime audience changes should never recreate header dropdowns or a bottom navigation dock');
  assert(!/initSpeedDial|data-speed-dial/.test(read('js/common/common.js')),
    'shared page initialization should not recreate or bind a floating contact navigation menu');
  assert(source.includes("document.addEventListener('site:route-change', syncHeaderContext)"),
    'the persistent masthead should update context after a committed route change');
};

if (require.main === module) {
  module.exports({ assert: require('assert') });
  process.stdout.write('Compact site header tests passed.\n');
}
