'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const hud = require('../js/games/project-starfall/ui/hud.js');
const canvasHover = require('../js/games/project-starfall/ui/canvas-hover.js');
const canvasRegions = require('../js/games/project-starfall/ui/canvas-regions.js');

global.ProjectStarfallData = require('../js/games/project-starfall/data/index.js');
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

const skillActions = [
  { id: 'skill:alpha', type: 'skill', skillId: 'alpha', label: 'Alpha Arc' },
  { id: 'skill:beta', type: 'skill', skillId: 'beta', label: 'Beta Burst' },
  { id: 'skill:gamma', type: 'skill', skillId: 'gamma', label: 'Gamma Guard' },
  { id: 'item:potion', type: 'item', itemId: 'potion', label: 'Potion' },
  { id: 'skill:passive', type: 'skill', skillId: 'passive', label: 'Passive', passive: true },
  { id: 'skill:locked', type: 'skill', skillId: 'locked', label: 'Locked', usable: false },
  { id: 'skill:zero', type: 'skill', skillId: 'zero', label: 'Zero Rank' },
  { id: 'skill:beta-copy', type: 'skill', skillId: 'beta', label: 'Duplicate Beta' },
  { id: 'skill:delta', type: 'skill', skillId: 'delta', label: 'Delta Dive' },
  { id: 'skill:epsilon', type: 'skill', skillId: 'epsilon', label: 'Epsilon Edge' },
  { id: 'skill:zeta', type: 'skill', skillId: 'zeta', label: 'Zeta Zone' },
  { id: 'skill:eta', type: 'skill', skillId: 'eta', label: 'Eta Echo' },
  { id: 'skill:theta', type: 'skill', skillId: 'theta', label: 'Theta Throw' }
];
const keybinds = {
  'skill:alpha': [],
  'skill:beta': ['Digit2'],
  'skill:gamma': ['Digit1']
};
const cooldowns = [
  { skillId: 'beta', remaining: 0.92, baseCooldown: 4 },
  { skillId: 'gamma', remaining: 0.5, baseCooldown: 8 }
];
const ranks = {
  alpha: 1,
  beta: 2,
  gamma: 1,
  zero: 0,
  delta: 1,
  epsilon: 1,
  zeta: 1,
  eta: 1,
  theta: 1
};
const options = {
  skillRanks: ranks,
  formatKeyCode(code) {
    return code.replace('Digit', '');
  }
};

const sourceBefore = JSON.stringify({ skillActions, keybinds, cooldowns, ranks });
const entries = hud.getCanvasCombatBeltEntries(skillActions, keybinds, cooldowns, options);

check(entries.length === 6,
  'the combat belt should always reserve six cells');
check(entries.every((entry, index) => entry.slotIndex === index),
  'each fixed belt cell should expose its stable slot index');
check(JSON.stringify(entries.slice(0, 5).map((entry) => entry.skillId)) ===
  JSON.stringify(['beta', 'gamma', 'alpha', 'delta', 'epsilon']),
  'bound skills should come first while preserving stable order within bound and unbound groups');
check(entries.slice(0, 5).every((entry) => entry.kind === 'skill'),
  'visible usable skills should be distinguishable as skill cells');
check(!entries.some((entry) => ['passive', 'locked', 'zero'].includes(entry.skillId)),
  'passive, unusable, and explicitly zero-rank actions should be filtered out');
check(entries.filter((entry) => entry.skillId === 'beta').length === 1,
  'duplicate skill actions should not create duplicate belt cells');
check(entries[0].bound && entries[0].keyCode === 'Digit2' && entries[0].keyLabel === '2',
  'bound skill cells should retain cloned binding and display-key metadata');
check(entries[0].cooldownRemaining === 0.92 && entries[0].cooldownBucket === 4 && !entries[0].ready,
  'skill cells should expose cooldown readiness and quarter-second buckets');
check(entries[5].kind === 'overflow' && entries[5].overflowCount === 3 && entries[5].label === '+3',
  'more than six usable skills should reserve the final cell for the hidden-skill count');
check(entries[5].region.type === 'hud-skill-overflow' && entries[5].panelId === 'skills',
  'the overflow cell should carry the Skills opener region metadata');
check(JSON.stringify({ skillActions, keybinds, cooldowns, ranks }) === sourceBefore,
  'deriving belt entries should not mutate actions, bindings, cooldowns, or ranks');

const lightweightSnapshotUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(lightweightSnapshotUi, {
  bindActionCache: new Map(),
  snapshot: {
    state: {
      player: {
        classId: 'fighter',
        advancedClassId: '',
        baseSkillPoints: 0,
        advancedSkillPoints: 0
      },
      skills: {
        fighter_heavy_strike: 1
      }
    },
    domainRevisions: {
      skills: 1,
      player: 1
    }
  }
});
const lightweightActions = lightweightSnapshotUi.getSkillBindActions();
check(lightweightActions.some((action) => action.skillId === 'fighter_heavy_strike'),
  'a lightweight HUD snapshot should derive ranked class skills before Skills has ever been opened');

const shortEntries = hud.getCanvasCombatBeltEntries(
  skillActions.slice(0, 3),
  keybinds,
  cooldowns,
  options
);
check(shortEntries.length === 6 &&
  shortEntries.filter((entry) => entry.kind === 'skill').length === 3 &&
  shortEntries.filter((entry) => entry.kind === 'empty').length === 3,
  'a short skill list should keep the six-cell footprint with explicit empty cells');
check(!shortEntries.some((entry) => entry.kind === 'overflow'),
  'the belt should not show an overflow opener when all usable skills fit');
check(shortEntries[0].region.type === 'hud-skill' &&
  shortEntries[0].region.skillId === 'beta' &&
  shortEntries[0].region.actionId === 'skill:beta',
  'visible skills should carry one-click HUD region metadata');

const cacheState = hud.getCanvasCombatBeltCacheState(
  skillActions,
  keybinds,
  cooldowns,
  options
);
const clonedCacheState = hud.getCanvasCombatBeltCacheState(
  JSON.parse(JSON.stringify(skillActions)),
  JSON.parse(JSON.stringify(keybinds)),
  JSON.parse(JSON.stringify(cooldowns)),
  { skillRanks: JSON.parse(JSON.stringify(ranks)), formatKeyCode: options.formatKeyCode }
);
check(cacheState.key === clonedCacheState.key &&
  cacheState.key === hud.getCanvasCombatBeltCacheKey(skillActions, keybinds, cooldowns, options),
  'equivalent belt state should produce a deterministic public cache key');
check(cacheState.cooldownBucketSeconds === 0.25 &&
  cacheState.skills.some((entry) => entry.skillId === 'beta' && entry.rank === 2),
  'cache metadata should expose the cooldown cadence and supplied skill ranks');

const rankChangedKey = hud.getCanvasCombatBeltCacheKey(
  skillActions,
  keybinds,
  cooldowns,
  { ...options, skillRanks: { ...ranks, beta: 3 } }
);
check(rankChangedKey !== cacheState.key,
  'changing a supplied skill rank should invalidate the belt cache');

const bindingChangedKey = hud.getCanvasCombatBeltCacheKey(
  skillActions,
  { ...keybinds, 'skill:beta': ['Digit3'] },
  cooldowns,
  options
);
check(bindingChangedKey !== cacheState.key,
  'changing a skill binding should invalidate the belt cache');

const sameCooldownBucketKey = hud.getCanvasCombatBeltCacheKey(
  skillActions,
  keybinds,
  cooldowns.map((cooldown) => cooldown.skillId === 'beta'
    ? { ...cooldown, remaining: 0.99 }
    : cooldown),
  options
);
check(sameCooldownBucketKey === cacheState.key,
  'cooldown movement inside one quarter-second bucket should reuse the belt cache');

const nextCooldownBucketKey = hud.getCanvasCombatBeltCacheKey(
  skillActions,
  keybinds,
  cooldowns.map((cooldown) => cooldown.skillId === 'beta'
    ? { ...cooldown, remaining: 0.74 }
    : cooldown),
  options
);
check(nextCooldownBucketKey !== cacheState.key,
  'crossing a quarter-second boundary should invalidate the belt cache');

const skillAction = hud.getHudRegionAction({
  type: 'hud-skill',
  skillId: 'beta',
  actionId: 'skill:beta'
});
check(skillAction.handled &&
  skillAction.type === 'activateSkill' &&
  skillAction.skillId === 'beta' &&
  skillAction.actionId === 'skill:beta',
  'skill regions should resolve to one-click skill activation');
const overflowAction = hud.getHudRegionAction({ type: 'hud-skill-overflow' });
check(overflowAction.handled &&
  overflowAction.type === 'openPanel' &&
  overflowAction.panelId === 'skills',
  'overflow regions should always resolve to the Skills panel');
check(!hud.getHudRegionAction({ type: 'hud-skill', skillId: '' }).handled,
  'malformed skill regions should remain safely unhandled');
check(hud.getHudRegionAction({ type: 'station-prompt', action: 'talk' }).type === 'stationPrompt' &&
  hud.getHudRegionAction({ type: 'minimap-toggle' }).type === 'toggleMinimap' &&
  hud.getHudRegionAction({ type: 'quest-tracker-toggle' }).type === 'toggleQuestTracker',
  'new belt regions should preserve all existing HUD region actions');

const layout = hud.getCanvasCombatBeltLayout(entries, { x: 10, y: 20 });
check(layout.maxCells === 6 && layout.slotSize === 36 && layout.gap === 4,
  'the default belt layout should use six 36px cells with 4px gaps');
check(layout.w === 236 &&
  layout.maxWidth === 236 &&
  layout.fitsApproximateFootprint &&
  layout.w <= hud.CANVAS_COMBAT_BELT_APPROXIMATE_FOOTPRINT,
  'the full six-cell belt should fit within the current approximate HUD footprint');
check(layout.cells.length === 6 &&
  layout.cells[0].x === 10 &&
  layout.cells[1].x === 50 &&
  layout.cells[5].x === 210,
  'belt cells should use deterministic fixed-width placement');
check(layout.cells[0].region.x === 10 &&
  layout.cells[0].region.y === 20 &&
  layout.cells[0].region.w === 36 &&
  layout.cells[0].region.h === 36,
  'interactive belt cells should expose their final hit-region rectangle');

[1024, 1280].forEach((width) => {
  const hudLayout = hud.getCanvasStatusHudLayout(
    { x: 0, y: 620, w: width, h: 84 },
    entries,
    { quickSize: 36, quickGap: 4 }
  );
  const beltLayout = hud.getCanvasCombatBeltLayout(entries, {
    x: hudLayout.quickX,
    y: hudLayout.menuY
  });
  check(beltLayout.x >= hudLayout.contentRight &&
    beltLayout.x + beltLayout.w < hudLayout.menu.x,
  `the ${width}px HUD should keep the combat belt between meters and Menu without overlap`);
});

const hoverRegions = [{
  type: 'hud-skill',
  skillId: 'beta',
  actionId: 'skill:beta',
  x: 10,
  y: 20,
  w: 36,
  h: 36
}];
const hudSkillHover = canvasHover.getCanvasHoverTargetAt({ x: 20, y: 30 }, {
  openWindows: [],
  findHoverRegion(filter) {
    return hoverRegions.find(filter) || null;
  }
});
check(hudSkillHover.type === 'skill' &&
  hudSkillHover.key === 'hudSkill:beta' &&
  hudSkillHover.sourcePanel === 'hud' &&
  hudSkillHover.skillId === 'beta',
  'combat-belt skills should expose a HUD-specific rich skill tooltip without opening a panel');
check(canvasRegions.isHudTooltipCanvasRegion(hoverRegions[0]) &&
  canvasRegions.isHudTooltipCanvasRegion({ type: 'hud-skill-overflow' }),
  'skill and overflow belt cells should keep HUD hover tracking active');

const panelToHudHoverUpdate = canvasHover.getCanvasHoverTargetUpdate(
  { type: 'skill', key: 'skill:beta', sourcePanel: 'skills', skillId: 'beta' },
  { x: 20, y: 30 },
  {
    hoverKey: 'belt:beta',
    getCanvasHoverTargetAt() {
      return hudSkillHover;
    }
  }
);
check(panelToHudHoverUpdate.changed &&
  panelToHudHoverUpdate.target.sourcePanel === 'hud',
  'moving the same skill from a panel to the belt should refresh its HUD casting context');

const firstHoverCacheKey = canvasHover.getCanvasHoverRegionCacheKey({
  canvasHitRegions: hoverRegions,
  combatBeltKey: 'beta:Digit2'
});
const reorderedHoverCacheKey = canvasHover.getCanvasHoverRegionCacheKey({
  canvasHitRegions: hoverRegions,
  combatBeltKey: 'gamma:Digit1'
});
check(firstHoverCacheKey !== reorderedHoverCacheKey,
  'reordering belt skills should invalidate the hover-region cache even when region count is unchanged');
const refreshedStationaryHover = canvasHover.getCanvasHoverRefreshForDraw(
  { x: 20, y: 30 },
  {
    x: 20,
    y: 30,
    key: firstHoverCacheKey,
    target: hudSkillHover
  },
  {
    hoverKey: reorderedHoverCacheKey,
    getCanvasHoverTargetAt() {
      return { type: 'skill', key: 'hudSkill:gamma', sourcePanel: 'hud', skillId: 'gamma' };
    }
  }
);
check(!refreshedStationaryHover.reused &&
  refreshedStationaryHover.target.skillId === 'gamma',
  'a stationary pointer should resolve the newly reordered belt skill instead of reusing stale hover state');

const drawnRegions = [];
const drawnActions = [];
const drawUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(drawUi, {
  canvasHoverTarget: { type: '', key: '' },
  drawCanvasHudSocket() {},
  drawActionIcon(ctx, action) {
    drawnActions.push(action);
  },
  drawRoundRect() {},
  drawCanvasText() {},
  drawCanvasMenuIcon() {},
  addCanvasRegion(region) {
    drawnRegions.push(region);
  },
  getBindableAction(actionId) {
    return { id: actionId, type: 'skill', skillId: actionId.replace('skill:', ''), label: actionId };
  }
});
drawUi.drawCanvasCombatBeltCell({}, layout.cells[0]);
drawUi.drawCanvasCombatBeltCell({}, layout.cells[5]);
check(drawnActions.length === 1 && drawnActions[0].skillId === 'beta',
  'rendering a skill cell should reuse the existing skill icon/cooldown renderer');
check(drawnRegions.some((region) => region.type === 'hud-skill' && region.skillId === 'beta') &&
  drawnRegions.some((region) => region.type === 'hud-skill-overflow' && region.tooltipTitle),
  'rendered skill and overflow cells should publish clickable, descriptive hit regions');

let activatedSkill = '';
let openedPanel = '';
const interactionUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(interactionUi, {
  monsterGuideSearchFocused: false,
  isCommandOpen: false,
  openWindows: [],
  activateSkillSelection(skillId) {
    activatedSkill = skillId;
    return true;
  },
  togglePanel(panelId) {
    openedPanel = panelId;
  }
});
interactionUi.executeCanvasRegion({ type: 'hud-skill', skillId: 'beta', actionId: 'skill:beta' });
check(activatedSkill === 'beta',
  'a single combat-belt click should activate its skill exactly once');
interactionUi.executeCanvasRegion({ type: 'hud-skill-overflow', panelId: 'skills' });
check(openedPanel === 'skills',
  'the overflow cell should open Skills without changing any key bindings');

const quickActions = hud.getCanvasHudQuickActions(
  ['inventory', 'equipment', 'skills', 'worldmap'],
  {
    getBindableAction(panelId) {
      return {
        panel: panelId,
        label: `${panelId} Popup`
      };
    }
  }
);
check(JSON.stringify(quickActions.map((action) => action.panel)) ===
  JSON.stringify(['inventory', 'equipment', 'skills', 'worldmap']),
  'the combat belt helpers should preserve all four existing utility destinations');

const helpers = hud.createHudCombatBeltUiHelpers();
check(Object.isFrozen(helpers) &&
  helpers.getCanvasCombatBeltEntries === hud.getCanvasCombatBeltEntries &&
  helpers.getCanvasCombatBeltCacheKey === hud.getCanvasCombatBeltCacheKey,
  'the HUD module should expose a frozen combat-belt helper group');

const uiSource = fs.readFileSync(
  path.join(__dirname, '..', 'js/games/project-starfall/project-starfall-ui.js'),
  'utf8'
);
check(uiSource.includes('const combatBeltKey = this.getCanvasCombatBeltCacheKey') &&
  uiSource.includes('hudLayout.quickButtons.forEach(({ action, x, y, size }) =>') &&
  uiSource.includes("this.drawCanvasText(ctx, 'Combat Skills'"),
  'the live HUD should render the belt and include its binding/rank/cooldown state in the overlay cache');

console.log(`Project Starfall combat belt checks passed: ${checks}`);
