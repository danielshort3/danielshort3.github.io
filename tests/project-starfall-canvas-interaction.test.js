'use strict';

const assert = require('assert');

global.ProjectStarfallData = require('../js/games/project-starfall/data/index.js');
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function pointInRegion(point, region) {
  return point.x >= region.x &&
    point.x <= region.x + region.w &&
    point.y >= region.y &&
    point.y <= region.y + region.h;
}

function createPendingInteractionFixture(staleRegion, currentRegion) {
  const ui = Object.create(ProjectStarfallUi.prototype);
  const point = { x: 44, y: 44 };
  let draws = 0;
  let selectedInventoryTab = '';
  let selectedWorldMapId = '';
  let closedPanelId = '';

  Object.assign(ui, {
    itemContextMenu: null,
    isCommandOpen: false,
    openWindows: [currentRegion.panelId],
    canvasHitRegions: [staleRegion],
    canvasDownRegion: null,
    canvasDownRegionModalBypass: false,
    canvasDrag: null,
    minimapDrag: null,
    questTrackerDrag: null,
    combatMetricsDrag: null,
    upgradePromptDrag: null,
    potentialPromptDrag: null,
    shardCraftPromptDrag: null,
    questPromptDrag: null,
    dropQuantityPromptDrag: null,
    adminNumberPromptDrag: null,
    confirmPromptDrag: null,
    canvasBindDrag: null,
    canvasGearDrag: null,
    canvasInventoryDrag: null,
    canvasSliderDrag: null,
    plinkoDropHold: null,
    selectedBindActionId: '',
    monsterGuideSearchFocused: false,
    pendingUiRefresh: { draw: true },
    pendingUiRefreshFrame: 17,
    pendingCanvasDrawFrame: 0,
    pendingCanvasDrawForce: false,
    pendingRunningCanvasDraw: false,
    pendingRunningCanvasDrawForce: false,
    pendingRunningCanvasDrawFrameId: -1,
    canvasDrawStats: { requests: 0, immediate: 0, deferred: 0, skippedWhileRunning: 0 },
    windowDragStats: { moves: 0, activePanel: '' },
    elements: {
      canvas: {
        width: 1280,
        height: 806
      }
    }
  });

  ui.engine = {
    running: true,
    draw() {
      draws += 1;
      ui.pendingRunningCanvasDraw = false;
      ui.pendingRunningCanvasDrawForce = false;
      ui.pendingRunningCanvasDrawFrameId = -1;
      ui.canvasHitRegions = [currentRegion];
    },
    selectWorldMapNode(mapId) {
      selectedWorldMapId = mapId;
    }
  };
  ui.getCanvasPoint = () => point;
  ui.findCanvasRegion = (candidate, filter) => {
    const region = ui.canvasHitRegions
      .slice()
      .reverse()
      .find((entry) => pointInRegion(candidate, entry) && (!filter || filter(entry)));
    return region || null;
  };
  ui.findCanvasDragSourceRegion = () => null;
  ui.getCanvasInventoryDragPayload = () => null;
  ui.getCanvasGearUid = () => '';
  ui.getDropCandidateForCanvasRegion = () => null;
  ui.raiseWindow = () => {};
  ui.selectInventoryTab = (tabId) => {
    selectedInventoryTab = tabId;
  };
  ui.closePanel = (panelId) => {
    closedPanelId = panelId;
    ui.openWindows = ui.openWindows.filter((id) => id !== panelId);
  };
  ui.flushUiRefresh = function flushUiRefreshForTest() {
    this.pendingUiRefresh = null;
    this.pendingRunningCanvasDraw = true;
    return true;
  };

  return {
    ui,
    point,
    getDraws: () => draws,
    getSelectedInventoryTab: () => selectedInventoryTab,
    getSelectedWorldMapId: () => selectedWorldMapId,
    getClosedPanelId: () => closedPanelId
  };
}

function runImmediatePointerDownTest(staleRegion, currentRegion, label) {
  const fixture = createPendingInteractionFixture(staleRegion, currentRegion);
  let prevented = false;
  fixture.ui.handleCanvasPointerDown({
    clientX: fixture.point.x,
    clientY: fixture.point.y,
    preventDefault() {
      prevented = true;
    }
  });

  check(fixture.getDraws() === 1, `${label} should synchronously refresh a pending canvas draw before hit testing`);
  check(fixture.ui.pendingUiRefresh === null, `${label} should flush the queued UI refresh`);
  check(fixture.ui.canvasDownRegion && fixture.ui.canvasDownRegion.type === currentRegion.type,
    `${label} should capture the current panel region instead of the stale region`);
  check(fixture.ui.canvasDownRegion && (
    fixture.ui.canvasDownRegion.tabId === currentRegion.tabId ||
    fixture.ui.canvasDownRegion.mapId === currentRegion.mapId
  ), `${label} should preserve the intended control metadata`);
  check(prevented === false, `${label} should retain the normal control pointer-down behavior`);

  fixture.ui.handleCanvasPointerUp({
    clientX: fixture.point.x,
    clientY: fixture.point.y,
    preventDefault() {}
  });
  if (currentRegion.type === 'inventory-tab') {
    check(fixture.getSelectedInventoryTab() === currentRegion.tabId,
      `${label} should select the intended inventory tab on pointer release`);
  } else {
    check(fixture.getSelectedWorldMapId() === currentRegion.mapId,
      `${label} should select the intended world-map node on pointer release`);
  }
  check(fixture.getClosedPanelId() === '', `${label} should not dismiss the open panel`);
  check(fixture.ui.openWindows.includes(currentRegion.panelId), `${label} should preserve the open panel`);
}

runImmediatePointerDownTest(
  { type: 'close-window', panelId: 'inventory', x: 20, y: 20, w: 60, h: 60 },
  { type: 'inventory-tab', panelId: 'inventory', tabId: 'usable', priority: 80, x: 20, y: 20, w: 60, h: 60 },
  'immediate inventory-tab click'
);

runImmediatePointerDownTest(
  { type: 'menu-panel', panelId: 'worldmap', source: 'command-menu', x: 20, y: 20, w: 60, h: 60 },
  { type: 'world-map-node', panelId: 'worldmap', mapId: 'thornpathThicket', x: 20, y: 20, w: 60, h: 60 },
  'immediate world-map click'
);

const steadyFixture = createPendingInteractionFixture(
  { type: 'inventory-tab', panelId: 'inventory', tabId: 'usable', x: 20, y: 20, w: 60, h: 60 },
  { type: 'inventory-tab', panelId: 'inventory', tabId: 'etc', x: 20, y: 20, w: 60, h: 60 }
);
steadyFixture.ui.pendingUiRefresh = null;
steadyFixture.ui.pendingUiRefreshFrame = 0;
steadyFixture.ui.pendingRunningCanvasDraw = false;
steadyFixture.ui.handleCanvasPointerDown({ clientX: 44, clientY: 44, preventDefault() {} });
check(steadyFixture.getDraws() === 0,
  'steady-state canvas clicks should not force a synchronous draw');
check(steadyFixture.ui.canvasDownRegion && steadyFixture.ui.canvasDownRegion.tabId === 'usable',
  'steady-state canvas clicks should keep using the current hit regions');

const staleMenuUi = Object.create(ProjectStarfallUi.prototype);
let staleMenuToggleCount = 0;
Object.assign(staleMenuUi, {
  isCommandOpen: false,
  monsterGuideSearchFocused: false,
  toggleCommandPanel() {},
  togglePanel() {
    staleMenuToggleCount += 1;
  },
  handleAction() {}
});
staleMenuUi.executeCanvasRegion({ type: 'menu-panel', panelId: 'worldmap', source: 'command-menu' });
check(staleMenuToggleCount === 0,
  'a stale command-menu region should not toggle a panel after the command menu has closed');

staleMenuUi.executeCanvasRegion({ type: 'menu-panel', panelId: 'worldmap', source: 'hud-quick' });
check(staleMenuToggleCount === 1,
  'a persistent HUD quick button should remain actionable while the command menu is closed');

staleMenuUi.isCommandOpen = true;
staleMenuUi.executeCanvasRegion({ type: 'menu-panel', panelId: 'worldmap', source: 'command-menu' });
check(staleMenuToggleCount === 2,
  'a current command-menu region should keep its normal panel-toggle behavior');

console.log(`Project Starfall canvas interaction checks passed: ${checks}`);
