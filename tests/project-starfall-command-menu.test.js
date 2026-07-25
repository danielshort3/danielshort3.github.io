'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const hud = require('../js/games/project-starfall/ui/hud.js');
const input = require('../js/games/project-starfall/ui/input.js');
const panels = require('../js/games/project-starfall/ui/panels.js');

global.ProjectStarfallData = require('../js/games/project-starfall/data/index.js');
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function flattenGroups(groups) {
  return (groups || []).flatMap((group) => group.items || []);
}

function sorted(values) {
  return values.slice().sort();
}

const snapshot = {
  dailyLogin: { claimable: true },
  channel: {
    currentId: 'ch3',
    channels: Array.from({ length: 8 }, (_, index) => ({
      id: `ch${index + 1}`,
      label: `Ch. ${index + 1}`,
      current: index === 2
    }))
  }
};

const getGroups = (pageId) => hud.getCanvasMenuGroups(snapshot, { pageId });
const rootItems = flattenGroups(getGroups('root'));
const adventureItems = flattenGroups(getGroups('adventure'));
const channelItems = flattenGroups(getGroups('channels'));
const settingsItems = flattenGroups(getGroups('settings'));

check(rootItems.length === 9,
  'the first-tier command menu should stay compact at nine stable choices');
check(rootItems.filter((item) => item.pageId).length === 3,
  'the first tier should expose exactly three secondary categories');
check(JSON.stringify(rootItems.filter((item) => item.pageId).map((item) => item.pageId)) ===
  JSON.stringify(['adventure', 'channels', 'settings']),
  'secondary categories should keep a stable Adventure, Channels, Settings order');
check(JSON.stringify(rootItems.filter((item) => item.panel).map((item) => item.panel)) ===
  JSON.stringify(['character', 'inventory', 'skills', 'quests', 'worldmap', 'daily']),
  'frequent character and progression panels should remain one click from the first tier');
check(rootItems.some((item) => item.panel === 'daily' && item.label === 'Daily Reward!'),
  'claimable daily rewards should keep their visible first-tier alert');
check(!rootItems.some((item) => item.action === 'changeChannel'),
  'individual channels should no longer crowd the first tier');

const allPageItems = rootItems.concat(adventureItems, settingsItems);
const reachablePanels = sorted(Array.from(new Set(allPageItems.filter((item) => item.panel).map((item) => item.panel))));
const expectedPanels = sorted([
  'character', 'equipment', 'partyPanel', 'inventory', 'skills', 'quests',
  'worldmap', 'monsters', 'shop', 'upgrade', 'plinko', 'daily', 'cashShop',
  'beta', 'settings', 'keybinds', 'admin', 'guide', 'log'
]);
check(JSON.stringify(reachablePanels) === JSON.stringify(expectedPanels),
  'the two-tier hierarchy should preserve every existing panel destination');
check(settingsItems.some((item) => item.action === 'fullscreen'),
  'Focus / Fullscreen should remain available on the Settings & Help page');
check(adventureItems.length === 8,
  'Adventure & Gear should remain focused while preserving all gear, party, shop, and activity choices');
check(settingsItems.length === 6,
  'Settings & Help should preserve options, help, logs, and admin access without crowding the root');

check(channelItems.length === 8 &&
  channelItems.every((item) => item.action === 'changeChannel'),
  'the Channels page should preserve all eight channel destinations');
check(channelItems.filter((item) => item.selected).length === 1 &&
  channelItems.find((item) => item.selected).channelId === 'ch3',
  'the Channels page should preserve exactly one current-channel marker');

check(hud.normalizeCanvasMenuPageId('settings') === 'settings',
  'known command-menu page ids should be preserved');
check(hud.normalizeCanvasMenuPageId('unknown') === 'root',
  'unknown command-menu page ids should fall back safely to the root');

const rootFooter = hud.getCanvasMenuFooterAction('root');
const subpageFooter = hud.getCanvasMenuFooterAction('channels');
check(rootFooter.action === 'load' && rootFooter.label === 'Logout',
  'the root footer should retain the existing logout action');
check(subpageFooter.pageId === 'root' && subpageFooter.back === true,
  'secondary pages should replace Logout with an in-menu Back action');

['root', 'adventure', 'channels', 'settings'].forEach((pageId) => {
  const footer = hud.getCanvasMenuFooterAction(pageId);
  const layout = hud.getCanvasMenuLayout(1280, 806, 720, getGroups(pageId), { footer, pageId });
  check(layout.h < 300,
    `${pageId} should fit in a compact parchment window under 300px tall`);
});

const baseCacheOptions = {
  commandOpen: true,
  commandMenuPage: 'root',
  openWindows: [],
  windowState: {}
};
const overlaySnapshot = { state: { mapId: 'greenrootMeadow', player: { classId: 'guardian' } } };
const rootCacheKey = hud.getCanvasOverlayCacheKey(1280, 806, overlaySnapshot, baseCacheOptions);
const adventureCacheKey = hud.getCanvasOverlayCacheKey(1280, 806, overlaySnapshot, {
  ...baseCacheOptions,
  commandMenuPage: 'adventure'
});
check(rootCacheKey !== adventureCacheKey,
  'overlay cache keys should change with the active command-menu page');

const pageAction = panels.getCommandMenuRegionAction({
  type: 'menu-page',
  pageId: 'channels',
  source: 'command-menu'
});
check(pageAction.handled && pageAction.type === 'navigateCommandMenu' && pageAction.pageId === 'channels',
  'menu-page regions should navigate inside the open command menu');
check(input.getEscapeMenuInputAction({
  isCommandOpen: true,
  commandMenuPage: 'channels'
}).action === 'backCommandMenuPage',
  'Escape should return from a secondary page before closing the command menu');
check(input.getEscapeMenuInputAction({
  isCommandOpen: true,
  commandMenuPage: 'root'
}).action === 'closeCommandPanel',
  'Escape should close the command menu from its root page');

const drawnRegions = [];
const rowUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(rowUi, {
  drawCanvasUiPanel() {},
  drawCanvasMenuIcon() {},
  getCanvasMenuIconId(item) {
    return item.iconId;
  },
  getPrimaryKeyLabel() {
    return '';
  },
  drawCanvasText() {},
  addCanvasRegion(region) {
    drawnRegions.push(region);
  }
});
rowUi.drawCanvasMenuRow({}, {
  label: 'Channels',
  pageId: 'channels',
  iconId: 'worldmap'
}, 10, 20, 100, 24);
check(drawnRegions.length === 1 &&
  drawnRegions[0].type === 'menu-page' &&
  drawnRegions[0].source === 'command-menu',
  'secondary category rows should emit guarded command-menu page regions');

let draws = 0;
let openedPanel = '';
let changedChannel = '';
const ui = Object.create(ProjectStarfallUi.prototype);
Object.assign(ui, {
  itemContextMenu: null,
  isCommandOpen: false,
  commandMenuPage: 'settings',
  monsterGuideSearchFocused: false,
  openWindows: [],
  dropQuantityPrompt: null,
  adminNumberPrompt: null,
  confirmPrompt: null,
  pendingInventoryDrop: null,
  gearPickerContext: null,
  potentialPromptOpen: false,
  shardCraftPromptOpen: false,
  upgradePromptOpen: false,
  closeItemContextMenu() {},
  clearHoldInputs() {},
  renderCommandPanel() {},
  queueUiRefresh() {},
  requestCanvasDraw() {
    draws += 1;
  },
  togglePanel(panelId) {
    openedPanel = panelId;
  },
  handleAction() {},
  isFocusModeActive() {
    return false;
  },
  engine: {
    changeChannel(channelId) {
      changedChannel = channelId;
    }
  },
  readEngineSnapshot() {}
});

ui.toggleCommandPanel(true);
check(ui.isCommandOpen && ui.commandMenuPage === 'root',
  'opening the command menu should always start from the root');
ui.executeCanvasRegion({ type: 'menu-page', pageId: 'adventure', source: 'command-menu' });
check(ui.isCommandOpen && ui.commandMenuPage === 'adventure' && draws === 1,
  'opening a secondary page should keep the command menu open and request a fresh draw');
ui.handleEscapeMenuKey();
check(ui.isCommandOpen && ui.commandMenuPage === 'root',
  'the first Escape on a secondary page should return to the root');
ui.handleEscapeMenuKey();
check(!ui.isCommandOpen && ui.commandMenuPage === 'root',
  'the next Escape should close the root menu and leave it reset');

ui.executeCanvasRegion({ type: 'menu-page', pageId: 'settings', source: 'command-menu' });
check(ui.commandMenuPage === 'root',
  'stale secondary-page regions should be ignored after the command menu closes');

ui.toggleCommandPanel(true);
ui.executeCanvasRegion({ type: 'menu-panel', panelId: 'inventory', source: 'command-menu' });
check(!ui.isCommandOpen && openedPanel === 'inventory',
  'direct panel choices should keep their existing close-then-open behavior');

ui.toggleCommandPanel(true);
ui.executeCanvasRegion({
  type: 'menu-action',
  action: 'changeChannel',
  channelId: 'ch6',
  source: 'command-menu'
});
check(!ui.isCommandOpen && changedChannel === 'ch6',
  'channel selection should keep its existing close-then-switch behavior');

const uiSource = fs.readFileSync(path.join(__dirname, '..', 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
check(uiSource.includes("commandMenuPage: this.commandMenuPage || 'root'") &&
  uiSource.includes("this.commandMenuPage || 'root',"),
  'both modular and fallback overlay cache paths should include the active command-menu page');

console.log(`Project Starfall command menu checks passed: ${checks}`);
