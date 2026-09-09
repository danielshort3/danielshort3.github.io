'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const framePolicy = require('../../js/navigation/site-frame-policy');
const frameSource = fs.readFileSync(path.join(__dirname, '../../js/navigation/site-frame.js'), 'utf8');
const accordionSource = fs.readFileSync(path.join(__dirname, '../../js/home/category-accordion.js'), 'utf8');
const order = ['about', 'projects', 'tools', 'games', 'contact'];
const helpers = frameSource.slice(frameSource.indexOf('  function compactStack('), frameSource.indexOf('  function transitionFrame('));
const context = vm.createContext({ personalOrder: order, current: { fit: 'viewport' }, document: { body: { getBoundingClientRect: () => ({ bottom: 1300 }) } }, window: { scrollY: 200, innerHeight: 844 } });
vm.runInContext(helpers, context);
let checks = 0;
const check = (value, message) => { assert(value, message); checks += 1; };

const descriptionContext = vm.createContext({ framePolicy });
vm.runInContext(frameSource.slice(frameSource.indexOf('  function describe('), frameSource.indexOf('  function ensureTab(')), descriptionContext);
for (const audience of ['personal', 'analytics', 'data-science', 'tourism']) {
  for (const category of order) {
    for (const fit of [undefined, 'document', 'viewport', 'unknown']) {
      const source = { dataset: { personalActiveCategory: category }, querySelector: () => null, querySelectorAll: () => [] };
      const scope = { body: { dataset: { audience, personalFit: fit } },
        querySelector: (selector) => selector === '[data-personal-accordion-shell]' ? source : null };
      check(descriptionContext.describe(scope, { view: 'library' }).fit === 'viewport',
        `${audience} ${category} routes normalize ${JSON.stringify(fit)} metadata to the standard desktop frame.`);
    }
  }
}
const immersiveSource = { dataset: {}, querySelector: () => null, querySelectorAll: () => [] };
check(descriptionContext.describe({ body: { dataset: { personalFit: 'immersive' } },
  querySelector: (selector) => selector === '[data-personal-accordion-shell]' ? immersiveSource : null }).fit === 'immersive',
  'Explicitly immersive route metadata remains exempt from the bounded frame.');

const layoutTabs = new Map([...order, 'resume'].map(id => [id, { dataset: {},
  classList: { states: new Map(), toggle(name, active) { this.states.set(name, active); } },
  style: {}, attributes: new Map(), setAttribute(name, value) { this.attributes.set(name, value); },
  removeAttribute(name) { this.attributes.delete(name); } }]));
const configurationContext = vm.createContext({ framePolicy, tabs: layoutTabs, colors: {}, personalOrder: order,
  professionalOrder: ['about', 'projects', 'resume', 'contact'], compactQuery: { matches: false },
  ensureTab: id => layoutTabs.get(id), stage: { style: {} }, slot: { style: {} },
  welcome: {}, panel: { setAttribute() {} }, body: {},
  frame: { dataset: {}, classList: { toggle() {} }, toggleAttribute() {}, style: { setProperty() {} } } });
vm.runInContext(frameSource.slice(frameSource.indexOf('  function configure('), frameSource.indexOf('  function capture(')), configurationContext);
const staleSnapshot = { audience: 'tourism', category: 'projects', view: 'detail', home: false, fit: 'document' };
configurationContext.configure(staleSnapshot);
check(staleSnapshot.fit === 'viewport' && configurationContext.frame.dataset.frameFit === 'viewport',
  'Reconfiguring a restored snapshot normalizes stale document fit in both state and rendered attributes.');
for (const compact of [false, true]) {
  configurationContext.compactQuery.matches = compact;
  configurationContext.configure({ audience: 'personal', category: '', view: 'closed', home: true });
  check(!configurationContext.welcome.hidden && !configurationContext.welcome.inert && configurationContext.welcome.id === 'main',
    'A folded homepage exposes its welcome as the accessible main landmark.');
  check(configurationContext.body.hidden && configurationContext.body.inert && configurationContext.panel.hidden &&
    configurationContext.body.id !== 'main', 'A folded homepage hides its retained category body without duplicating the main landmark.');
  check(order.every(id => !layoutTabs.get(id).hidden && layoutTabs.get(id).tabIndex === 0 &&
    layoutTabs.get(id).attributes.get('aria-expanded') === 'false' &&
    !layoutTabs.get(id).classList.states.get('is-active')), 'All five folded tabs stay available with no expanded or active selection.');
  check(configurationContext.stage.style.gridTemplateColumns === (compact ? 'minmax(0, 1fr)' : 'repeat(5, minmax(0, 1fr))') &&
    order.every((id, index) => layoutTabs.get(id).style.gridArea === (compact ? `${index + 1} / 1` : `1 / ${index + 1}`)),
  'The folded navigation uses one seamless mobile column or five adjacent desktop rails.');
  configurationContext.configure({ audience: 'personal', category: 'about', view: 'overview', home: true });
  check(configurationContext.welcome.hidden && configurationContext.welcome.inert && configurationContext.body.id === 'main' &&
    !configurationContext.body.hidden && !configurationContext.body.inert && !configurationContext.panel.hidden,
  'Reopening About restores the retained category body and hides the resting welcome.');
}

let adoptedCommit;
const hardManifest = { id: 'tools:transcribe', path: '/tools/transcribe', navigation: 'hard' };
const hardContext = vm.createContext({
  frame: null, welcome: null, stage: null, panel: null, slot: null, canvas: null, toolbar: null, viewport: null, loading: null, lastWidth: 0,
  document: { querySelector: selector => selector === '[data-site-route-manifest]' ? { textContent: JSON.stringify(hardManifest) }
    : selector === '[data-site-route-content]' ? { replaceWith: node => { node.isConnected = true; } } : null },
  describe: (scope, manifest) => ({ manifest, tabSources: [] }),
  make: () => ({ append() {}, setAttribute() {}, clientWidth: 1412 }),
  commit: (description, options) => { adoptedCommit = { description, options }; }
});
vm.runInContext(frameSource.slice(frameSource.indexOf('  function adopt('), frameSource.indexOf('  function refresh(')), hardContext);
const hardFrame = hardContext.adopt();
check(hardFrame?.isConnected && adoptedCommit.options.original && adoptedCommit.options.animate === false,
  'A hard-navigation tool mounts the same visual frame around its original document content.');
check(adoptedCommit.description.manifest.navigation === 'hard' && hardContext.adopt() === hardFrame,
  'Shared frame adoption preserves the hard route manifest and remains idempotent.');
hardContext.frame = null;
hardContext.describe = () => { throw new Error('No shared shell'); };
check(hardContext.adopt() === null, 'Raw document surfaces without a shared shell remain outside frame adoption.');

const routerSource = fs.readFileSync(path.join(__dirname, '../../js/navigation/page-transitions.js'), 'utf8');
let currentManifest = hardManifest;
let isBoundary = true;
const lifecycleContext = vm.createContext({ window: { location: { href: 'https://example.test/tools/transcribe' } }, document: {},
  resolveUrl: value => new URL(value), readRouteManifest: () => currentManifest, isHardBoundary: () => isBoundary });
vm.runInContext(routerSource.slice(routerSource.indexOf('  function isCurrentRouteSoft('), routerSource.indexOf('  function getEligibleLinkUrl(')), lifecycleContext);
check(!lifecycleContext.isCurrentRouteSoft(), 'A mounted hard-navigation document cannot opt into soft route transitions.');
currentManifest = { navigation: 'soft' };
check(!lifecycleContext.isCurrentRouteSoft(), 'A hard URL boundary cannot be bypassed by stale soft route metadata.');
isBoundary = false;
check(lifecycleContext.isCurrentRouteSoft(), 'Ordinary shared-frame routes retain their existing soft lifecycle.');

function layout(category, gap, view = 'overview') {
  let y = 62;
  const tabs = new Map();
  let slot;
  for (const id of order) {
    const height = view === 'overview' ? (id === category ? 54 : 48) : (id === category ? 78 : 0);
    tabs.set(id, height ? { x: 0, y, width: 390, height } : { x: 0, y: 0, width: 0, height: 0 });
    y += height;
    if (id === category) { slot = { x: 0, y, width: 390, height: gap, padding: '4px' }; y += gap; }
  }
  return { category, home: true, view, audience: 'personal', compact: true,
    frame: { x: 0, y: 62, width: 390, height: y - 62, borderWidth: '0px' }, tabs, slot, scroll: { y: 200 } };
}
const mix = (first, last, progress) => ({ x: first.x + (last.x - first.x) * progress,
  y: first.y + (last.y - first.y) * progress, width: first.width + (last.width - first.width) * progress,
  height: first.height + (last.height - first.height) * progress });
function sample(plan, progress) {
  const first = progress <= .5 ? plan.first : plan.middle;
  const last = progress <= .5 ? plan.middle : plan.last;
  return new Map(plan.ids.map((id) => [id, mix(first.get(id), last.get(id), progress <= .5 ? progress * 2 : (progress - .5) * 2)]));
}
function assertPacked(plan, label) {
  for (let step = 0; step <= 40; step += 1) {
    const rows = [...sample(plan, step / 40).values()].filter((row) => row.height > .001);
    const gaps = rows.slice(1).map((row, index) => row.y - rows[index].y - rows[index].height);
    check(gaps.every((gap) => gap >= -.001), `${label}: rows must never overlap`);
    check(gaps.filter((gap) => gap > .001).length <= 1, `${label}: only the selected content gap may separate rows`);
  }
  const packed = [...plan.middle.values()];
  check(packed.slice(1).every((row, index) => Math.abs(row.y - packed[index].y - packed[index].height) < .001), `${label}: midpoint rows must be contiguous`);
  check(plan.closing.height === 0 && plan.opening.height === 0, `${label}: move the gap only while it is closed`);
}

for (const from of order) {
  for (const to of order.filter((id) => id !== from)) {
    const first = layout(from, 516);
    const last = layout(to, 640);
    const plan = context.compactStack(first, last, last);
    check(Boolean(plan), `${from} to ${to} must use compact stack motion`);
    assertPacked(plan, `${from} to ${to}`);
    // Retarget from the actual geometry in either phase, including a destination
    // category whose gap has not opened yet.
    for (const fraction of [.2, .7]) {
      const rows = sample(plan, fraction);
      const owner = fraction < .5 ? from : to;
      const row = rows.get(owner);
      const slotHeight = fraction < .5 ? first.slot.height * (1 - fraction * 2) : last.slot.height * (fraction - .5) * 2;
      const interrupted = { ...last, tabs: rows, slot: { ...last.slot, y: row.y + row.height, height: slotHeight } };
      const reverse = context.compactStack(interrupted, first, first);
      assertPacked(reverse, `${from} to ${to} reversed at ${fraction}`);
    }
  }
  const first = layout(from, 520);
  const library = layout(from, 900, 'library');
  assertPacked(context.compactStack(first, library, library), `${from} library entry`);
  assertPacked(context.compactStack(library, first, first), `${from} library return`);
}
const desktop = { ...layout('about', 500), compact: false };
check(context.compactStack(desktop, layout('projects', 600), layout('projects', 600)) === null, 'desktop outward motion must keep its existing geometry path');
check(context.compactStack(layout('about', 500), layout('about', 600), layout('about', 600)) === null, 'same-section resize must not close and reopen content');
check(context.resolveScrollTarget(layout('projects', 640), { top: 9999 }) === 656, 'scroll destination must clamp to the final natural document height');
check(context.resolveScrollTarget(layout('projects', 640), { top: -50 }) === 0, 'negative scroll targets must clamp at the document start');
check(context.resolveScrollTarget(layout('projects', 640), { category: 'projects', offset: 62 }) === 248, 'category alignment must subtract the visible header exactly once');
check(context.resolveScrollTarget(desktop, { top: 0 }) === null, 'viewport homepage scroll orchestration must not change desktop document scrolling');
context.current = { fit: 'document' };
check(context.resolveScrollTarget(desktop, { top: 320 }) === null, 'legacy fit metadata cannot route standard desktop scroll history to the document');
context.current = { fit: 'viewport' };
context.document.documentElement = {};
context.getComputedStyle = (node) => node === context.document.documentElement ? { scrollPaddingTop: '62px' } : { scrollMarginTop: '80px' };
const hashTarget = { isConnected: true, getBoundingClientRect: () => ({ top: 350 }) };
check(context.resolveScrollTarget(layout('projects', 640), { target: hashTarget, top: 0, offset: 62 }) === 470,
  'hash destinations must use their measured layout and the larger authored/header offset without counting the header twice');
hashTarget.isConnected = false;
check(context.resolveScrollTarget(layout('projects', 640), { target: hashTarget, top: 50 }) === 50,
  'a disconnected hash target must fall back to saved history scroll');

const properties = new Map([['min-height', { value: '95vh', priority: 'important' }]]);
const scrollCalls = [];
const flowContext = vm.createContext({ flowReservation: null,
  document: { documentElement: { scrollHeight: 1600, style: {
    getPropertyValue: (key) => properties.get(key)?.value || '', getPropertyPriority: (key) => properties.get(key)?.priority || '',
    setProperty: (key, value, priority) => properties.set(key, { value, priority }), removeProperty: (key) => properties.delete(key)
  } } }, window: { scrollX: 0, scrollY: 700, innerHeight: 844, scrollTo: (value) => scrollCalls.push(value) }
});
vm.runInContext(frameSource.slice(frameSource.indexOf('  function reserveFlow('), frameSource.indexOf('  function guardLayout(')), flowContext);
flowContext.reserveFlow();
check(properties.get('min-height').value === '1600px', 'a shrinking document must retain the departing scroll range during motion');
flowContext.reserveFlow();
flowContext.releaseFlow({ x: 0, y: 200 });
check(properties.get('min-height').value === '95vh' && properties.get('min-height').priority === 'important', 'retargeting must restore the original inline minimum and priority, not its temporary reservation');
check(scrollCalls.length === 1 && scrollCalls[0].top === 200 && scrollCalls[0].behavior === 'instant', 'settlement must apply the final scroll target without starting another animation');
flowContext.releaseFlow();
check(properties.get('min-height').value === '95vh', 'repeated cancellation cleanup must leave the original inline minimum intact');

let copiedIcon;
const tab = { firstElementChild: { replaceChildren: (icon) => { copiedIcon = icon; } }, children: [{}, { textContent: 'Projects' }], setAttribute() {}, style: { setProperty() {} } };
const iconContext = vm.createContext({ tabs: new Map([['projects', tab]]), colors: { projects: '#155dfc' },
  document: { importNode: (source) => ({ attributes: new Map(source.attributes), hasAttribute(name) { return this.attributes.has(name); }, setAttribute(name, value) { this.attributes.set(name, value); } }) }
});
vm.runInContext(frameSource.slice(frameSource.indexOf('  function ensureTab('), frameSource.indexOf('  function loadContent(')), iconContext);
const authoredIcon = { attributes: new Map([['viewBox', '0 0 24 24']]) };
const iconSource = { querySelector: (selector) => selector === 'svg' ? authoredIcon : null, getAttribute: () => null };
iconContext.ensureTab('projects', iconSource);
check(copiedIcon.attributes.get('width') === '24' && copiedIcon.attributes.get('height') === '24', 'newly adopted tab SVGs must have intrinsic dimensions before CSS arrives');
check(copiedIcon.attributes.get('fill') === 'none' && copiedIcon.attributes.get('stroke') === 'currentColor', 'tab SVGs must render as outlined symbols without relying on loaded stylesheets');
authoredIcon.attributes.set('stroke-width', '2.5');
iconContext.ensureTab('projects', iconSource);
check(copiedIcon.attributes.get('stroke-width') === '2.5', 'fallback SVG styling must preserve explicit authored geometry');

const headerCode = accordionSource.slice(accordionSource.indexOf('    function headerBottom('), accordionSource.indexOf('    async function select('));
let headers = [{ visible: false, bottom: 0 }, { visible: true, bottom: 62 }];
const headerContext = vm.createContext({ document: { querySelectorAll: () => headers.map((header) => ({ getClientRects: () => header.visible ? [{}] : [], getBoundingClientRect: () => header })) }, getComputedStyle: () => ({ visibility: 'visible' }) });
vm.runInContext(headerCode, headerContext);
check(headerContext.headerBottom() === 62, 'portrait tabs must measure the visible masthead, not the hidden desktop wrapper');
headers = [{ visible: true, bottom: 61 }, { visible: false, bottom: 0 }];
check(headerContext.headerBottom() === 61, 'short landscape tabs must measure the visible fixed navigation bar');

async function runAsyncChecks() {
  const homeOverview = { hasAttribute: () => false };
  const homeLibrary = { hasAttribute: (name) => name === 'data-home-library-view' };
  const homeItem = { querySelector: () => ({ children: [homeOverview, homeLibrary] }) };
  const homeHeading = { tagName: 'H1' };
  const libraryBackButtons = new Map(['projects', 'tools', 'games'].map(category => [category, { category }]));
  const homeToolbar = { childNodes: [], hidden: true, replaceChildren(...children) { this.childNodes = children; } };
  let mountedChildren;
  const releasedFits = [];
  const homeContext = vm.createContext({
    window: {},
    current: { home: true, view: 'overview', fit: 'viewport', manifest: {}, heading: homeHeading,
      libraryBackButtons,
      items: new Map(order.map(category => [category, category === 'tools' ? homeItem : { querySelector: () => ({ children: [] }) }])) },
    framePolicy, localSequence: 0, desiredTarget: null, held: null,
    toolbar: homeToolbar, viewport: {
      childNodes: [],
      get firstChild() { return this.childNodes[0] || null; },
      insertBefore(node, reference) {
        node.remove?.();
        const index = reference ? this.childNodes.indexOf(reference) : this.childNodes.length;
        this.childNodes.splice(index, 0, node);
        node.parentNode = this;
        node.remove = () => {
          this.childNodes.splice(this.childNodes.indexOf(node), 1);
          node.parentNode = null;
        };
      }
    }, capture: () => ({}), transition() {}, setLoading() {}, wipe: () => Promise.resolve(true),
    body: { replaceChildren: (...children) => { mountedChildren = children; } },
    release: () => { releasedFits.push(homeContext.current.fit); return Promise.resolve(true); }
  });
  vm.runInContext(frameSource.slice(frameSource.indexOf('  function updateHomeToolbar('), frameSource.indexOf('  function commit(')), homeContext);
  vm.runInContext(frameSource.slice(frameSource.indexOf('  function snapshot('), frameSource.indexOf('  function showHome(')), homeContext);
  vm.runInContext(frameSource.slice(frameSource.indexOf('  function showHome('), frameSource.indexOf('  function hasFrameInteractionLayer(')), homeContext);
  check(await homeContext.showHome('tools', 'library', { animate: false }), 'Opening a home library commits the selected view.');
  check(homeContext.current.fit === 'viewport' && releasedFits[0] === 'viewport', 'The inline tool library retains the same desktop frame before it is released.');
  check(homeOverview.hidden && homeOverview.inert && !homeLibrary.hidden && !homeLibrary.inert, 'Opening a library exposes only its library content.');
  check(mountedChildren[0] === homeHeading && mountedChildren[1] === homeItem, 'Changing home views preserves the page heading before its content.');
  check(!homeToolbar.hidden && homeToolbar.childNodes[0] === libraryBackButtons.get('tools'),
    'A home library uses its original back control in the shared toolbar.');
  const savedLibrary = homeContext.snapshot();
  check(await homeContext.showHome('tools', 'overview', { animate: false }), 'Returning from a library commits the overview.');
  check(homeContext.current.fit === 'viewport' && releasedFits[1] === 'viewport', 'Returning home restores the desktop viewport frame.');
  check(!homeOverview.hidden && !homeOverview.inert && homeLibrary.hidden && homeLibrary.inert, 'Returning home restores the overview content.');
  check(homeToolbar.hidden && homeToolbar.childNodes.length === 0,
    'Returning to an overview hides the toolbar without losing the stored back control.');
  for (const category of order) {
    for (const view of ['library', 'overview']) {
      homeContext.current.fit = 'document';
      check(await homeContext.showHome(category, view, { animate: false }), `Opening ${category} ${view} commits the selected view.`);
      check(homeContext.current.fit === 'viewport' && releasedFits.at(-1) === 'viewport',
        `${category} ${view} restores the standard frame from stale history metadata.`);
      check(homeContext.desiredTarget.category === category && homeContext.desiredTarget.fit === 'viewport' &&
        mountedChildren[1] === homeContext.current.items.get(category), `${category} ${view} mounts the intended content without changing frame policy.`);
      check(homeToolbar.hidden === (view !== 'library' || !libraryBackButtons.has(category)) &&
        (homeToolbar.hidden || homeToolbar.childNodes[0] === libraryBackButtons.get(category)),
        `${category} ${view} keeps the correct original back control across repeated switches.`);
    }
  }
  const detailBack = { category: 'detail' };
  const savedDetail = { description: { home: false, view: 'detail' }, body: {}, toolbar: [detailBack] };
  homeContext.restore(savedDetail, { animate: false });
  check(homeToolbar.childNodes[0] === detailBack && !homeToolbar.hidden,
    'Restoring a detail keeps the canonical route toolbar.');
  homeContext.restore(savedLibrary, { animate: false });
  check(homeToolbar.childNodes[0] === libraryBackButtons.get('tools') && !homeToolbar.hidden,
    'Restoring a home library snapshot recovers its original back control after a detail route.');
  check(await homeContext.showHome('tools', 'overview', { animate: false }) && homeToolbar.hidden,
    'A restored home library can close normally through the same toolbar state.');

  libraryBackButtons.forEach((back) => { back.hasAttribute = (name) => name === 'data-page-masthead-parent'; });
  check(await homeContext.showHome('tools', 'library', { animate: false }) &&
    homeToolbar.hidden && homeToolbar.childNodes.length === 0,
  'Integrated library parent controls remain inside their headers instead of creating a second toolbar.');
  const integratedSnapshot = homeContext.snapshot();
  homeContext.restore(savedDetail, { animate: false });
  homeContext.restore(integratedSnapshot, { animate: false });
  check(homeToolbar.hidden && homeToolbar.childNodes.length === 0 &&
    homeContext.current.libraryBackButtons.get('tools') === libraryBackButtons.get('tools'),
  'Restoring an integrated library preserves the original parent control and keeps the extra toolbar hidden.');
  const retainedChildren = mountedChildren;
  check(await homeContext.showHome('', 'closed', { animate: false }) && homeContext.current.category === '' &&
    homeContext.current.view === 'closed' && mountedChildren === retainedChildren,
  'Folding all categories retains the mounted content instead of replacing it with an empty body.');
  check(homeToolbar.hidden && homeToolbar.childNodes.length === 0, 'A folded homepage has no stale library back control.');
  check(await homeContext.showHome('tools', 'overview', { animate: false }) && mountedChildren[1] === homeItem,
    'Reopening a folded category reuses its original content object.');

  const animations = [];
  const timers = new Map();
  let nextTimer = 1;
  let milliseconds = 0;
  const motionContext = vm.createContext({
    viewport: { style: {}, inert: true, animate: () => {
      const animation = { canceled: false, cancel() { this.canceled = true; } };
      animation.finished = new Promise(resolve => { animation.complete = resolve; });
      animations.push(animation);
      return animation;
    } },
    window: { setTimeout(callback) { const id = nextTimer++; timers.set(id, callback); return id; }, clearTimeout: id => timers.delete(id) },
    current: { view: 'overview' }, geometry: null, wipeMotion: null, wipeClosed: true, compactQuery: { matches: true }, duration: () => milliseconds,
    getComputedStyle: () => ({ clipPath: 'inset(0% 0% 50% 0%)' })
  });
  vm.runInContext(frameSource.slice(frameSource.indexOf('  function wipe('), frameSource.indexOf('  function setLoading(')), motionContext);
  let ready;
  let exposed = false;
  motionContext.geometry = { opening: new Promise(resolve => { ready = resolve; }) };
  const pendingReveal = motionContext.wipe(true).then(value => { exposed = value; });
  await Promise.resolve();
  check(!exposed && motionContext.viewport.inert, 'Incoming content stays closed while the departing mobile accordion gap collapses.');
  ready(true);
  await pendingReveal;
  check(exposed && !motionContext.viewport.inert, 'Incoming content reveals once its own mobile accordion gap opens.');
  motionContext.viewport.inert = true;
  motionContext.geometry.opening = Promise.resolve(false);
  check(await motionContext.wipe(true) === false && motionContext.viewport.inert, 'Superseded accordion geometry cannot reveal stale content.');
  motionContext.geometry = null;
  check(await motionContext.wipe(true), 'Reduced-motion reveal completes when no accordion midpoint is pending.');
  check(!motionContext.viewport.inert && motionContext.viewport.style.clipPath === 'inset(0% 0% 0% 0%)', 'Revealed content is interactive and unclipped.');
  await motionContext.wipe(false);
  check(motionContext.viewport.inert, 'Closing the content prevents interaction while a destination is prepared.');

  milliseconds = 160;
  const controller = new AbortController();
  const canceled = motionContext.wipe(false, { signal: controller.signal });
  controller.abort();
  check(await canceled === false && animations[0].canceled, 'Aborting a wipe cancels its animation and resolves the pending operation.');
  check(timers.size === 0 && motionContext.wipeMotion === null, 'Aborted reveals clean up timers and ownership.');
  const superseded = motionContext.wipe(false);
  const revealed = motionContext.wipe(true);
  check(await superseded === false && animations[1].canceled, 'A newer reveal cancels the previous closing animation.');
  animations[2].complete();
  check(await revealed === true && !motionContext.viewport.inert, 'The current reveal completes with interactive content.');
  check(timers.size === 0 && motionContext.wipeMotion === null, 'Completed reveals clean up timers and ownership.');
  motionContext.current.view = 'closed';
  check(await motionContext.wipe(true) && motionContext.viewport.inert && motionContext.wipeClosed &&
    motionContext.viewport.style.clipPath === 'inset(0% 0% 100% 0%)',
  'A router reveal cannot expose or enable the retained category body while all tabs are folded.');
  console.log(`Vertical accordion geometry, compact retargeting, scroll history, and reveal cleanup: ${checks} checks passed.`);
}
runAsyncChecks().catch((error) => { console.error(error); process.exitCode = 1; });
