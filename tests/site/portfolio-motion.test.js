const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.resolve(__dirname, '../../js/portfolio/portfolio.js'), 'utf8');
const css = fs.readFileSync(path.resolve(__dirname, '../../css/components/portfolio-workbench.css'), 'utf8');

const element = () => {
  const classes = new Set();
  return {
    hidden: false,
    dataset: {},
    focused: 0,
    attributes: new Map(),
    classList: {
      contains: (name) => classes.has(name),
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle(name, enabled) {
        if (enabled) classes.add(name);
        else classes.delete(name);
      }
    },
    setAttribute(name, value) { this.attributes.set(name, value); },
    removeAttribute(name) { this.attributes.delete(name); },
    toggleAttribute(name, enabled) {
      if (enabled) this.attributes.set(name, '');
      else this.attributes.delete(name);
    },
    querySelectorAll: () => [],
    querySelector: () => null,
    focus() { this.focused += 1; }
  };
};

const motion = () => {
  const calls = [];
  const pending = new Map();
  return {
    calls,
    swaps: 0,
    presence(target, open, options = {}) {
      const call = { target, open, options, finish: () => options.onFinish?.() };
      calls.push(call);
      pending.set(target, call);
      target.classList.toggle(options.className || 'is-open', open);
      return Promise.resolve(true);
    },
    finish(target) {
      const call = pending.get(target);
      pending.delete(target);
      call?.finish();
    },
    swap(target, update) {
      this.swaps += 1;
      update();
      return Promise.resolve(true);
    }
  };
};

module.exports = function runPortfolioMotionTests({ assert }) {
  const compactQuery = source.match(/const PORTFOLIO_COMPACT_QUERY = '([^']+)';/)?.[1];
  assert(compactQuery === '(max-width: 820px), (max-height: 480px) and (pointer: coarse)', 'compact workbenches must include short touch landscape without changing fine-pointer desktop layouts');
  assert((css.match(/@media \(max-width: 820px\), \(max-height: 480px\) and \(pointer: coarse\)/g) || []).length === 3, 'all compact workbench CSS blocks must match the shared interaction query');
  assert(!source.includes("window.matchMedia('(min-width: 821px)')"), 'filter dialogs must derive desktop state from the same compact query as Quick view');
  const sheetSource = source.slice(source.indexOf('  const setSheetOpen = (open) => {'), source.indexOf('  if (openButton) {'));
  const root = element();
  const filterPanel = element();
  const children = [element(), element()];
  children.forEach((child) => { child.hidden = true; });
  const sheetMotion = motion();
  const frames = [];
  const sheetContext = vm.createContext({
    root,
    filterPanel,
    children,
    document: { body: element() },
    window: { SiteMotion: sheetMotion, requestAnimationFrame: (callback) => frames.push(callback) },
    bindings: {
      frame: (callback) => frames.push(callback),
      media: (query, callback) => query.addEventListener('change', callback)
    },
    backgroundIsolation: { isolate() {}, restore() {} },
    syncToggleLabel() {},
    toggleButton: element(),
    triggerButton: element()
  });
  vm.runInContext(`
    let sheetRevision = 0;
    let desktopMode = false;
    const setExpandableHidden = (hidden) => children.forEach((child) => { child.hidden = hidden; });
    ${sheetSource}
    globalThis.setSheetOpen = setSheetOpen;
    globalThis.setDesktop = (desktop) => { desktopMode = desktop; setSheetOpen(false); };
  `, sheetContext);
  sheetContext.setSheetOpen(true);
  assert(children.every((child) => !child.hidden), 'filter contents must be available before the sheet enters');
  sheetContext.setSheetOpen(false);
  const closingSheet = sheetMotion.calls.at(-1);
  assert(children.every((child) => !child.hidden), 'filter contents must stay rendered throughout the sheet exit');
  closingSheet.finish();
  assert(children.every((child) => child.hidden), 'filter contents must become hidden after exit completes');
  sheetContext.setSheetOpen(true);
  sheetContext.setSheetOpen(false);
  const interruptedClose = sheetMotion.calls.at(-1);
  sheetContext.setSheetOpen(true);
  interruptedClose.finish();
  assert(children.every((child) => !child.hidden) && filterPanel.classList.contains('is-open'), 'a stale close must not hide a reopened filter sheet');
  sheetContext.setSheetOpen(false);
  sheetContext.setDesktop(true);
  sheetMotion.calls.at(-1).finish();
  assert(children.every((child) => !child.hidden) && !filterPanel.hidden, 'desktop resize must preserve visible filters despite pending mobile cleanup');
  frames.splice(0).forEach((callback) => callback());
  assert(sheetContext.toggleButton.focused === 0 && sheetContext.triggerButton.focused === 0, 'stale mobile focus callbacks must not run after switching to desktop');

  const mobileSheetSource = source.slice(source.indexOf('function setupPortfolioMobileFilterSheet('));
  const viewportSource = mobileSheetSource.slice(mobileSheetSource.indexOf('  if (window.matchMedia) {'), mobileSheetSource.indexOf('  const syncSortControls = () => {'));
  let viewportChanged;
  const compactMedia = {
    matches: true,
    addEventListener(event, callback) { viewportChanged = callback; }
  };
  sheetContext.PORTFOLIO_COMPACT_QUERY = compactQuery;
  sheetContext.window.matchMedia = (query) => {
    assert(query === compactQuery, 'filter sheet viewport mode must consume the shared query');
    return compactMedia;
  };
  vm.runInContext(viewportSource, sheetContext);
  assert(filterPanel.attributes.get('role') === 'dialog', 'short touch landscape must use the filter dialog lifecycle');
  sheetContext.setSheetOpen(true);
  compactMedia.matches = false;
  viewportChanged();
  assert(!filterPanel.attributes.has('role') && !filterPanel.attributes.has('inert') && children.every((child) => !child.hidden), 'switching to a desktop pointer/layout must restore ordinary visible filters');
  compactMedia.matches = true;
  viewportChanged();
  assert(filterPanel.attributes.get('role') === 'dialog' && children.every((child) => child.hidden), 'switching back to compact mode must restore closed dialog semantics');

  const workbenchSource = source.slice(source.indexOf('function buildPortfolioWorkbench('));
  const rendererSource = workbenchSource.slice(workbenchSource.indexOf('  const renderInspector = ('), workbenchSource.indexOf('  const isMobileSelectionCard = ('));
  const renderedInspector = element();
  const rendererContext = vm.createContext({
    inspector: renderedInspector,
    directoryKind: 'projects',
    isDirectoryWorkbench: false,
    isAudienceScopedView: true,
    itemSingular: 'project',
    summaryTitle: 'Problem',
    highlightsTitle: 'Outcome',
    approachTitle: 'Approach',
    stackTitle: 'Tools',
    ctaLabel: 'View case study',
    toList: (value) => Array.isArray(value) ? value : [],
    getProjectFocuses: () => [],
    getPrimaryFormat: () => 'Case study',
    getProjectHref: () => '/portfolio/test',
    escapeHtml: (value) => String(value || ''),
    unique: (values) => [...new Set(values)],
    chipMarkup: () => ''
  });
  vm.runInContext(`${rendererSource}\nrenderInspector({ id: 'test', title: 'Test project', problem: 'Readable project details' });`, rendererContext);
  const bodyStart = renderedInspector.innerHTML.indexOf('<div class="portfolio-inspector__body">');
  assert(bodyStart > renderedInspector.innerHTML.indexOf('data-portfolio-inspector-close'), 'Quick view Close must stay outside the independently scrolling body');
  assert(bodyStart < renderedInspector.innerHTML.indexOf('Readable project details') && bodyStart < renderedInspector.innerHTML.indexOf('View case study'), 'project details and case-study link must share the scrollable body');
  const selectionSource = workbenchSource.slice(workbenchSource.indexOf('  const renderSelection = ('), workbenchSource.indexOf('  const clearSelection = ('));
  const inspector = element();
  const inspectorRoot = element();
  const inspectorMotion = motion();
  const state = { selectedId: null };
  const selectionContext = vm.createContext({
    root: inspectorRoot,
    inspector,
    resultHost: element(),
    state,
    allProjects: [{ id: 'first' }, { id: 'second' }],
    mobileSelectionOverlayEnabled: true,
    window: { SiteMotion: inspectorMotion },
    syncInspectorDialogState() {},
    renderInspector(project) { inspector.content = project?.id || 'empty'; }
  });
  vm.runInContext(`
    let inspectorRevision = 0;
    let renderedInspectorId;
    let mobile = true;
    const isMobileSelectionCard = () => mobile;
    ${selectionSource}
    globalThis.renderSelection = renderSelection;
    globalThis.setMobile = (value) => { mobile = value; };
  `, selectionContext);
  state.selectedId = 'first';
  selectionContext.renderSelection();
  assert(inspector.content === 'first' && inspector.classList.contains('is-open'), 'Quick view must render its selected project before entering');
  state.selectedId = null;
  selectionContext.renderSelection();
  const closeInspector = inspectorMotion.calls.at(-1);
  assert(inspector.content === 'first' && inspectorRoot.classList.contains('is-inspector-closing'), 'Quick view must retain the previous project and layout through exit');
  state.selectedId = 'second';
  selectionContext.renderSelection();
  closeInspector.finish();
  assert(inspector.content === 'second' && inspectorRoot.classList.contains('has-selected-project'), 'interrupted Quick view cleanup must not erase the newly selected project');
  state.selectedId = null;
  selectionContext.renderSelection();
  inspectorMotion.calls.at(-1).finish();
  assert(inspector.content === 'empty' && !inspectorRoot.classList.contains('has-selected-project'), 'completed Quick view exit must clear retained content and layout');
  selectionContext.setMobile(false);
  state.selectedId = 'first';
  selectionContext.renderSelection();
  const swaps = inspectorMotion.swaps;
  state.selectedId = 'second';
  selectionContext.renderSelection();
  assert(inspectorMotion.swaps === swaps + 1 && inspector.content === 'second', 'desktop project changes must use the height-preserving content transition');
  selectionContext.renderSelection();
  assert(inspectorMotion.swaps === swaps + 1, 'unchanged selection renders must not restart the content transition');
};

if (require.main === module) {
  module.exports({ assert: require('assert') });
  process.stdout.write('Portfolio motion lifecycle checks passed.\n');
}
