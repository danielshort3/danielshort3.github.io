'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function classList(initial = []) {
  const values = new Set(initial);
  return {
    add: (...names) => names.forEach((name) => values.add(name)),
    remove: (...names) => names.forEach((name) => values.delete(name)),
    contains: (name) => values.has(name)
  };
}

const root = { classList: classList(['site-realm-professional']) };
const body = {
  dataset: { audience: 'analytics', siteRealm: 'professional', siteRealmHome: '/analytics' },
  classList: classList(['professional-home-page'])
};
const searchInput = { removed: false, remove() { this.removed = true; } };
const homeLink = { href: '/analytics', setAttribute(name, value) { this[name] = value; } };
const robots = { removed: false, remove() { this.removed = true; } };
const document = {
  readyState: 'complete',
  documentElement: root,
  body,
  head: { querySelector: () => robots },
  querySelectorAll(selector) {
    if (selector === 'input[data-search-audience]') return [searchInput];
    if (selector === '[data-entry-home-link="true"]') return [homeLink];
    return [];
  },
  addEventListener() {}
};
const window = {
  location: new URL('https://example.test/portfolio?audience=analytics&sort=recent#projects'),
  history: {
    state: null,
    replaceState(state, title, next) {
      window.location = new URL(next, window.location);
    }
  }
};

const source = fs.readFileSync(path.join(__dirname, '../../js/common/site-realm.js'), 'utf8');
vm.runInNewContext(source, { window, document, URL, Set });

assert.strictEqual(window.location.pathname + window.location.search + window.location.hash,
  '/portfolio?sort=recent#projects', 'legacy audience query must be removed without losing normal query or hash');
assert.strictEqual(body.dataset.audience, 'personal');
assert.strictEqual(body.dataset.siteRealm, 'personal');
assert(!Object.prototype.hasOwnProperty.call(body.dataset, 'siteRealmHome'));
assert(root.classList.contains('site-realm-personal') && !root.classList.contains('site-realm-professional'));
assert(!body.classList.contains('professional-home-page'));
assert(searchInput.removed && robots.removed && homeLink.href === '/');
assert(window.SITE_REALM === 'personal' && window.SITE_AUDIENCE === 'personal');
assert(window.getSiteRealm() === 'personal' && window.getSiteAudience() === 'personal');
assert(window.isProfessionalRealm() === false);

window.location = new URL('https://example.test/contact?mode=career&subject=hello');
window.SiteRealm.sync();
assert.strictEqual(window.location.pathname + window.location.search, '/contact?subject=hello',
  'legacy mode query must be removed from canonical pages');

window.location = new URL('https://example.test/portfolio/babynames?audience=tourism');
window.SiteRealm.sync();
assert.strictEqual(window.location.pathname + window.location.search, '/portfolio/babynames',
  'project details must use the canonical route');

process.stdout.write('Canonical site realm passed.\n');
