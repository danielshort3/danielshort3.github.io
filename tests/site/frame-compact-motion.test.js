'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const frameSource = fs.readFileSync(path.join(__dirname, '../../js/navigation/site-frame.js'), 'utf8');
const accordionSource = fs.readFileSync(path.join(__dirname, '../../js/home/category-accordion.js'), 'utf8');
const order = ['about', 'projects', 'tools', 'games', 'contact'];
const helpers = frameSource.slice(frameSource.indexOf('  function compactStack('), frameSource.indexOf('  function transitionFrame('));
const context = vm.createContext({ personalOrder: order, document: { body: { getBoundingClientRect: () => ({ bottom: 1300 }) } }, window: { scrollY: 200, innerHeight: 844 } });
vm.runInContext(helpers, context);
let checks = 0;
const check = (value, message) => { assert(value, message); checks += 1; };

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
check(context.resolveScrollTarget(desktop, { top: 0 }) === null, 'compact scroll orchestration must not change desktop scrolling');
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
  let ready;
  let exposed = false;
  const motionContext = vm.createContext({
    viewport: { animate: null, style: {}, inert: true }, geometry: { opening: new Promise((resolve) => { ready = resolve; }) },
    wipeMotion: null, wipeClosed: true, compactQuery: { matches: true }, duration: () => 0,
    getComputedStyle: () => ({ clipPath: 'inset(0% 0% 100% 0%)' })
  });
  vm.runInContext(frameSource.slice(frameSource.indexOf('  function wipe('), frameSource.indexOf('  function setLoading(')), motionContext);
  const reveal = motionContext.wipe(true).then((value) => { exposed = value; });
  await Promise.resolve();
  check(!exposed && motionContext.viewport.inert, 'incoming content must stay closed while the previous compact gap collapses');
  ready(true);
  await reveal;
  check(exposed && !motionContext.viewport.inert, 'incoming content may reveal when its own gap starts opening');
  motionContext.viewport.inert = true;
  motionContext.geometry.opening = Promise.resolve(false);
  check(await motionContext.wipe(true) === false && motionContext.viewport.inert, 'superseded compact geometry must not reveal stale content');
  console.log(`Compact stack, retargeting, scroll targets, header measurement, and reveal gating: ${checks} checks passed.`);
}
runAsyncChecks().catch((error) => { console.error(error); process.exitCode = 1; });
