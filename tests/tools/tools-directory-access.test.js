'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { loadSiteContent } = require('../../build/lib/content-loader');
const { buildToolsLibraryItems } = require('../../build/generate-personal-accordion-pages');
const { buildHomeLibraryData } = require('../../build/generate-cms-artifacts');
const { renderPersonalLibraryMain } = require('../../build/lib/personal-accordion-shell');

const root = path.resolve(__dirname, '../..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');

async function run() {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };
  const content = loadSiteContent(root);
  const items = buildToolsLibraryItems(content);
  // Compare both production render paths without writing generated artifacts.
  const homeItems = buildHomeLibraryData(content).tools.items;
  const restricted = items.filter((item) => item.visibility !== 'public');
  check(restricted.length === 5, 'The dedicated Tools library must retain all five account and admin tools.');
  ['short-links', 'campaign-creative-tracker', 'ga4-utm-performance'].forEach((id) => {
    check(items.some((item) => item.id === id && item.visibility === 'admin'), `${id} must retain its admin restriction.`);
  });
  ['job-application-tracker', 'transcribe'].forEach((id) => {
    check(items.some((item) => item.id === id && item.visibility === 'authed'), `${id} must remain available to signed-in accounts.`);
  });
  const html = renderPersonalLibraryMain({ category: 'tools', items });
  const expectedGroups = {
    'Start here': ['text-compare', 'qr-code-generator', 'screen-recorder'],
    Text: ['nbsp-cleaner', 'oxford-comma-checker', 'point-of-view-checker', 'word-frequency'],
    Images: ['image-optimizer', 'background-remover'],
    Links: ['utm-batch-builder'],
    'Account tools': ['job-application-tracker', 'transcribe'],
    'Admin tools': ['short-links', 'campaign-creative-tracker', 'ga4-utm-performance']
  };
  const groups = [...html.matchAll(/<section class="home-library__group" aria-label="([^"]+)">([\s\S]*?)<\/section>/g)]
    .map((match) => ({
      name: match[1],
      html: match[2],
      cards: [...match[2].matchAll(/<li\b([^>]*)>([\s\S]*?)<\/li>/g)]
    }));
  check(groups.map((group) => group.name).join('|') === Object.keys(expectedGroups).join('|'),
    'Dedicated tools should lead with the selected public tools, followed by subject groups, account tools, and admin tools.');
  check(JSON.stringify(items.slice(0, 3).map((item) => item.id)) === JSON.stringify(expectedGroups['Start here']),
    'The first three tools must be Text Compare, QR Code Generator, and Screen Recorder in that order.');
  groups.forEach((group) => {
    const cardIds = group.cards.map((match) => /data-content-id="([^"]+)"/.exec(match[2])?.[1]);
    check(JSON.stringify([...cardIds].sort()) === JSON.stringify([...expectedGroups[group.name]].sort()),
      `${group.name} must contain exactly the tools appropriate to that group.`);
    const heading = /<h2\b([^>]*)>([^<]+)<\/h2>/.exec(group.html);
    check(heading?.[2] === group.name && /\bid="home-library-tools-group-\d+"/.test(heading[1]) &&
      /\btabindex="-1"/.test(heading[1]) && group.html.includes(`aria-label="${group.name}"`),
      `${group.name} must retain a meaningful heading and named list.`);
    const isRestricted = ['Account tools', 'Admin tools'].includes(group.name);
    check(group.cards.every((match) => /\bhidden\b/.test(match[1]) === isRestricted),
      `${group.name} must ${isRestricted ? 'hide every card before authentication' : 'show its public cards without authentication'}.`);
  });
  const publicItems = items.filter((item) => item.visibility === 'public');
  check(publicItems.length === homeItems.length && publicItems.every((item, index) =>
    item.id === homeItems[index].id && item.group === homeItems[index].group),
  'Direct Tools navigation and generated HOME_LIBRARY_DATA must have identical public group order, card order, and membership.');
  check(!homeItems.some((item) => ['Account tools', 'Admin tools'].includes(item.group)),
    'The homepage library must not expose restricted group headings.');
  check(/\.home-library__group:not\(:has\(\.home-library__item:not\(\[hidden\]\)\)\)\s*\{[^}]*display:\s*none\s*;/
    .test(read('css/components/home-library.css')),
  'A group whose cards are all hidden must stay hidden until account visibility reveals a child.');
  const libraryCards = [...html.matchAll(/<li\b([^>]*)>([\s\S]*?)<\/li>/g)];
  const restrictedCards = libraryCards.filter((match) => /data-tools-visibility=/.test(match[1]));
  check(restrictedCards.length === restricted.length, 'Restricted cards must exist in generated HTML so hydration can reveal them.');
  check(restrictedCards.every((match) => /\bhidden\b/.test(match[1]) && /aria-hidden="true"/.test(match[1])), 'Restricted cards must be hidden before authentication, including without JavaScript.');
  items.forEach((item) => {
    const card = libraryCards.find((match) => match[2].includes(`data-content-id="${item.id}"`));
    const accessLabels = [...card[2].matchAll(/<span class="home-library__access">([\s\S]*?)<\/span>/g)];
    if (item.visibility === 'admin') {
      check(accessLabels.length === 1 && accessLabels[0][1].replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim() === 'Admin access',
        `${item.title} must show one clear Admin access label on its card.`);
      check(/<svg\b[^>]*aria-hidden="true"/.test(accessLabels[0][1]),
        `${item.title} must keep its key icon decorative so the text names the access level.`);
    } else {
      check(accessLabels.length === 0 && !card[2].includes('Admin access'),
        `${item.title} must not imply administrator access for a ${item.visibility} tool.`);
    }
  });
  check(restricted.every((item) => !require('../../js/home/home-library-data').tools.items.some((publicItem) => publicItem.id === item.id)), 'The homepage showcase must remain public-only.');
  const fixtureItems = buildToolsLibraryItems({ tools: [
    { slug: 'listed', href: 'tools/listed', visibility: 'public' },
    { slug: 'hidden-public', href: 'tools/hidden-public', visibility: 'public', hidden: true },
    { slug: 'noindex-public', href: 'tools/noindex-public', visibility: 'public', noindex: true },
    { slug: 'restricted', href: 'tools/restricted', visibility: 'admin', hidden: true, noindex: true },
    { slug: 'invalid', href: 'tools/invalid', visibility: 'unexpected' }
  ] });
  check(fixtureItems.map((item) => item.id).sort().join(',') === 'listed,restricted', 'Public hidden/noindex entries stay excluded while restricted entries retain their account visibility.');

  const listeners = new Map();
  const stored = new Map();
  const storage = { getItem: (key) => stored.get(key) || null, setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key) };
  const makeCard = (visibility) => {
    const attributes = new Map([['hidden', ''], ['aria-hidden', 'true']]);
    return {
      dataset: { toolsVisibility: visibility },
      hasAttribute: (name) => attributes.has(name),
      setAttribute: (name, value) => attributes.set(name, value),
      removeAttribute: (name) => attributes.delete(name)
    };
  };
  const cards = { public: makeCard('public'), account: makeCard('authed'), admin: makeCard('admin'), unknown: makeCard('unexpected') };
  const document = {
    body: { dataset: { page: 'tools' } },
    querySelector: () => null,
    querySelectorAll: (selector) => selector === '[data-tools-visibility]' ? Object.values(cards) : [],
    addEventListener: (name, callback) => {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(callback);
    },
    removeEventListener: (name, callback) => listeners.get(name)?.delete(callback),
    dispatchEvent: (event) => [...(listeners.get(event.type) || [])].forEach((callback) => callback(event))
  };
  const window = { location: { origin: 'http://127.0.0.1:4173', pathname: '/tools', search: '', hash: '', assign() {} }, addEventListener: () => {} };
  const cleanups = [];
  const context = {
    window, document, localStorage: storage, sessionStorage: storage, console,
    URL, URLSearchParams, Date, Set, Map, atob,
    CustomEvent: class { constructor(type) { this.type = type; } },
    cleanText: (value) => String(value || '').trim(),
    disposed: false,
    registerCleanup: (callback) => cleanups.push(callback)
  };
  vm.createContext(context);
  vm.runInContext(read('js/accounts/tools-config.js'), context);
  vm.runInContext(read('js/accounts/tools-auth.js'), context);
  const accountSource = read('js/accounts/tools-account-ui.js');
  const visibilityStart = accountSource.indexOf('  const getToolsVisibilityContext =');
  const visibilityEnd = accountSource.indexOf('  const ensureToolsHero =', visibilityStart);
  assert(visibilityStart >= 0 && visibilityEnd > visibilityStart);
  vm.runInContext(accountSource.slice(visibilityStart, visibilityEnd), context);
  const mountStart = accountSource.indexOf('      const applyToolsAccountVisibility =');
  const mountEnd = accountSource.indexOf("      if (page !== 'tools'", mountStart);
  assert(mountStart >= 0 && mountEnd > mountStart);
  vm.runInContext(accountSource.slice(mountStart, mountEnd), context);

  check(!cards.public.hasAttribute('hidden') && cards.account.hasAttribute('hidden') && cards.admin.hasAttribute('hidden'), 'Initial signed-out mount must hide account and admin cards.');
  check(cards.unknown.hasAttribute('hidden'), 'Unknown visibility rules must fail closed.');
  const updateAuth = (claims, expired = false) => {
    storage.setItem('toolsAuth', JSON.stringify({ sessionOnly: true, expiresAt: Date.now() + (expired ? -1 : 3600000), claims }));
    document.dispatchEvent({ type: 'tools:auth-changed' });
  };
  updateAuth({ sub: 'regular-account', email: 'reader@example.com', email_verified: true });
  check(!cards.account.hasAttribute('hidden') && cards.admin.hasAttribute('hidden'), 'Ordinary signed-in accounts see account tools without admin tools.');
  updateAuth({ sub: 'unverified-owner', email: 'danielshort3@gmail.com', email_verified: false });
  check(cards.admin.hasAttribute('hidden'), 'An unverified owner email must not reveal admin tools.');
  updateAuth({ sub: 'owner', email: 'danielshort3@gmail.com', email_verified: true });
  check(!cards.admin.hasAttribute('hidden') && !cards.admin.hasAttribute('aria-hidden'), 'Verified configured administrator email must reveal accessible cards after the auth event.');
  updateAuth({ sub: 'group-admin', email: 'group-admin@example.com', 'cognito:groups': ['admins'] });
  check(!cards.admin.hasAttribute('hidden'), 'Existing administrator group entitlement must continue to work.');
  updateAuth({ sub: 'expired-owner', email: 'danielshort3@gmail.com', email_verified: true }, true);
  check(cards.account.hasAttribute('hidden') && cards.admin.hasAttribute('hidden'), 'Expired sessions must hide restricted cards.');

  stored.clear();
  context.fetch = async () => ({ ok: true, json: async () => ({
    expiresAt: Math.floor(Date.now() / 1000) + 3600,
    user: { sub: 'restored-owner', email: 'danielshort3@gmail.com', emailVerified: true, groups: [] }
  }) });
  await window.ToolsAuth.ensureFreshAuth();
  document.dispatchEvent({ type: 'tools:auth-changed' });
  check(!cards.account.hasAttribute('hidden') && !cards.admin.hasAttribute('hidden'), 'Restored cookie sessions must reveal the same authorized cards as token sessions.');
  await window.ToolsAuth.signOut();
  document.dispatchEvent({ type: 'tools:auth-changed' });
  check(cards.account.hasAttribute('hidden') && cards.admin.hasAttribute('hidden'), 'Sign-out must hide previously revealed restricted cards.');
  cleanups.forEach((cleanup) => cleanup());
  check(listeners.get('tools:auth-changed').size === 0, 'Route cleanup must remove the visibility listener.');
  return checks;
}

module.exports = run;
if (require.main === module) run().then((checks) => console.log(`Tools directory access: ${checks} checks passed.`)).catch((error) => { console.error(error); process.exitCode = 1; });
