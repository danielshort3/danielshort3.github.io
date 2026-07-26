'use strict';

const data = require('../js/games/project-starfall/data/index.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const keybindings = require('../js/games/project-starfall/ui/keybindings.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

global.ProjectStarfallData = data;
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');

let checks = 0;
const failures = [];
function check(condition, message) {
  checks += 1;
  if (!condition) failures.push(message);
}

const nearbyDrop = {
  uid: 'drop_first_loot_prompt',
  x: 164,
  y: 606,
  w: 34,
  h: 34,
  item: {
    uid: 'first_loot_prompt',
    id: 'upgrade_dust',
    kind: 'material',
    materialId: 'upgradeDust',
    name: 'Upgrade Dust',
    rarity: 'Common',
    quantity: 1
  }
};

function createWorldPromptSnapshot(playerOverrides, snapshotOverrides) {
  return Object.assign({
    state: {
      player: Object.assign({
        activeStation: '',
        activePortalId: '',
        activeQuestNpcId: ''
      }, playerOverrides || {})
    },
    map: { stations: [] },
    portals: [],
    questNpcs: { npcs: [] },
    onboarding: {
      hidden: false,
      nextStep: { id: 'loot_drop', title: 'Pick up loot' }
    },
    nearbyLoot: nearbyDrop
  }, snapshotOverrides || {});
}

const defaultKeybinds = keybindings.createDefaultKeybinds();
const defaultLootLabel = keybindings.getPrimaryKeyLabel(defaultKeybinds, 'loot');
check(defaultLootLabel === 'Z', 'the existing default loot binding should remain Z');

const reboundKeybinds = Object.assign({}, defaultKeybinds, { loot: ['KeyG'] });
const reboundLootLabel = keybindings.getPrimaryKeyLabel(reboundKeybinds, 'loot');
check(reboundLootLabel === 'G', 'the keybinding system should expose a rebound loot key label');

const defaultLootPrompt = hud.getStationPromptContext(
  createWorldPromptSnapshot(),
  { keyLabels: { loot: defaultLootLabel } }
);
check(!!defaultLootPrompt, 'a reachable nearby drop should publish a contextual world prompt');
check(defaultLootPrompt && defaultLootPrompt.promptAction === 'loot',
  'the nearby-drop prompt should expose the loot action');
check(defaultLootPrompt && defaultLootPrompt.loot === nearbyDrop,
  'the nearby-drop prompt should retain the reachable drop it describes');
check(defaultLootPrompt && defaultLootPrompt.target !== nearbyDrop &&
  defaultLootPrompt.target.x === nearbyDrop.x - 29 &&
  defaultLootPrompt.target.y === nearbyDrop.y - 48 &&
  defaultLootPrompt.target.w === 58 &&
  defaultLootPrompt.target.h === 58,
  'the nearby-drop prompt should anchor to the actual non-equipment loot draw bounds');
check(defaultLootPrompt && defaultLootPrompt.title === 'Upgrade Dust',
  'the nearby-drop prompt should identify the item being collected');
check(defaultLootPrompt && defaultLootPrompt.hint.includes('Z') && /loot|pick up/i.test(defaultLootPrompt.hint),
  'the nearby-drop prompt should explain collection with the default loot key');
check(defaultLootPrompt && /loot|drop/i.test(defaultLootPrompt.kindLabel),
  'the nearby-drop prompt should identify itself as loot rather than a station');

const reboundLootPrompt = hud.getStationPromptContext(
  createWorldPromptSnapshot(),
  { keyLabels: { loot: reboundLootLabel } }
);
check(reboundLootPrompt && reboundLootPrompt.hint.includes('G'),
  'the nearby-drop prompt should reflect the player rebound loot key');
check(reboundLootPrompt && !reboundLootPrompt.hint.includes('Z'),
  'the nearby-drop prompt should not keep advertising Z after loot is rebound');

const unboundLootPrompt = hud.getStationPromptContext(
  createWorldPromptSnapshot(),
  { keyLabels: { loot: 'Unbound' } }
);
check(unboundLootPrompt && /click pick up/i.test(unboundLootPrompt.hint),
  'an unbound loot action should advertise the clickable prompt instead of an unusable key');
check(hud.getStationPromptContext(createWorldPromptSnapshot({}, {
  onboarding: {
    hidden: true,
    nextStep: { id: 'loot_drop', title: 'Pick up loot' }
  }
}), {
  keyLabels: { loot: 'Z' }
}) === null, 'hiding the journey guide should also suppress its contextual loot prompt');

['equipment', 'card'].forEach((kind) => {
  const tieredDrop = Object.assign({}, nearbyDrop, {
    uid: `drop_first_loot_${kind}`,
    item: Object.assign({}, nearbyDrop.item, {
      uid: `first_loot_${kind}`,
      kind
    })
  });
  const prompt = hud.getStationPromptContext(createWorldPromptSnapshot({}, {
    nearbyLoot: tieredDrop
  }), {
    keyLabels: { loot: 'Z' }
  });
  check(prompt && prompt.target.x === tieredDrop.x - 22 &&
    prompt.target.y === tieredDrop.y - 36 &&
    prompt.target.w === 44 &&
    prompt.target.h === 44,
  `${kind} loot prompts should anchor to the compact tier-aura draw bounds`);
});

[
  {
    label: 'station',
    snapshot: createWorldPromptSnapshot(
      { activeStation: 'greenroot_workbench' },
      { map: { stations: [{ id: 'greenroot_workbench', name: 'Greenroot Workbench', x: 120, y: 560, w: 80, h: 60 }] } }
    ),
    action: 'interact',
    targetId: 'greenroot_workbench'
  },
  {
    label: 'portal',
    snapshot: createWorldPromptSnapshot(
      { activePortalId: 'greenroot_return' },
      { portals: [{ id: 'greenroot_return', label: 'Starfall Crossing', x: 120, y: 520, w: 100, h: 100 }] }
    ),
    action: 'portal',
    targetId: 'greenroot_return'
  },
  {
    label: 'NPC',
    snapshot: createWorldPromptSnapshot(
      { activeQuestNpcId: 'greenroot_guide' },
      {
        questNpcs: {
          npcs: [{
            id: 'greenroot_guide',
            name: 'Greenroot Guide',
            x: 120,
            y: 560,
            w: 52,
            h: 72,
            iconStates: [{ action: 'accept', questId: 'first_steps', icon: '!' }]
          }]
        }
      }
    ),
    action: 'npcTalk',
    targetId: 'greenroot_guide'
  }
].forEach((fixture) => {
  const prompt = hud.getStationPromptContext(fixture.snapshot, { keyLabels: { loot: 'Z' } });
  check(prompt && prompt.promptAction === fixture.action,
    `an active ${fixture.label} prompt should retain priority over nearby loot`);
  check(prompt && prompt.target && prompt.target.id === fixture.targetId,
    `an active ${fixture.label} prompt should keep its original target`);
});

const overlappingPrompt = hud.getStationPromptContext(createWorldPromptSnapshot(
  {
    activeStation: 'greenroot_workbench',
    activePortalId: 'greenroot_return',
    activeQuestNpcId: 'greenroot_guide'
  },
  {
    map: {
      stations: [{ id: 'greenroot_workbench', name: 'Greenroot Workbench', x: 120, y: 560, w: 80, h: 60 }]
    },
    portals: [{ id: 'greenroot_return', label: 'Starfall Crossing', x: 120, y: 520, w: 100, h: 100 }],
    questNpcs: {
      npcs: [{
        id: 'greenroot_guide',
        name: 'Greenroot Guide',
        x: 120,
        y: 560,
        w: 52,
        h: 72,
        iconStates: [{ action: 'accept', questId: 'first_steps', icon: '!' }]
      }]
    }
  }
), {
  keyLabels: { moveUp: 'Up', npcTalk: 'Y', interact: 'F', loot: 'Z' }
});
check(overlappingPrompt && overlappingPrompt.promptAction === 'npcTalk' &&
  overlappingPrompt.target && overlappingPrompt.target.id === 'greenroot_guide',
'overlapping world interactions should pair the highest-priority NPC label, target, and action');

const stationPortalPrompt = hud.getStationPromptContext(createWorldPromptSnapshot(
  { activeStation: 'greenroot_workbench', activePortalId: 'greenroot_return' },
  {
    map: {
      stations: [{ id: 'greenroot_workbench', name: 'Greenroot Workbench', x: 120, y: 560, w: 80, h: 60 }]
    },
    portals: [{ id: 'greenroot_return', label: 'Starfall Crossing', x: 120, y: 520, w: 100, h: 100 }]
  }
), {
  keyLabels: { moveUp: 'Up', interact: 'F', loot: 'Z' }
});
check(stationPortalPrompt && stationPortalPrompt.promptAction === 'interact' &&
  stationPortalPrompt.target && stationPortalPrompt.target.id === 'greenroot_workbench',
'overlapping station and portal interactions should pair the station label, target, and action');

check(hud.getStationPromptContext(createWorldPromptSnapshot({}, { nearbyLoot: null }), {
  keyLabels: { loot: 'Z' }
}) === null, 'the contextual prompt should disappear when no interactive target or reachable loot remains');
check(hud.getStationPromptContext(createWorldPromptSnapshot({}, {
  onboarding: {
    hidden: false,
    nextStep: { id: 'open_inventory', title: 'Review the item grid' }
  }
}), {
  keyLabels: { loot: 'Z' }
}) === null, 'the contextual loot prompt should stay suppressed after the first-loot onboarding step');

const lootPromptBox = defaultLootPrompt
  ? hud.getStationPromptLayout(1280, 720, 620, defaultLootPrompt.target, {
    camera: { x: 0, y: 0, zoom: 1 }
  })
  : null;
const lootPromptMetadata = defaultLootPrompt && lootPromptBox
  ? hud.getStationPromptRenderMetadata(defaultLootPrompt, lootPromptBox)
  : null;
check(lootPromptMetadata && lootPromptMetadata.region.action === 'loot',
  'the rendered nearby-drop prompt should retain its loot action in the canvas hit region');
check(lootPromptMetadata && lootPromptMetadata.titleText.value === 'Upgrade Dust',
  'the rendered nearby-drop prompt should retain the item name');
check(defaultLootPrompt && lootPromptBox &&
  lootPromptBox.y + lootPromptBox.h <= defaultLootPrompt.target.y - 8,
  'the parchment prompt should stay visibly above the loot sprite instead of overlapping it');

const hudLootAction = hud.getHudRegionAction({ type: 'station-prompt', action: 'loot' });
check(hudLootAction.handled && hudLootAction.type === 'stationPrompt' && hudLootAction.action === 'loot',
  'the shared HUD region router should preserve the loot action');

const lootGuideSnapshot = {
  progress: {},
  onboarding: {
    nextStep: {
      id: 'loot_drop',
      title: 'Pick up loot',
      summary: 'Hold the loot key near a dropped item to collect coins, gear, or materials.'
    },
    activePhase: { title: 'First Steps', completeCount: 8, total: 9 },
    total: 9
  }
};
const defaultLootGuide = hud.getQuestTrackerEntries(lootGuideSnapshot, {
  keyLabels: { loot: defaultLootLabel }
})[0];
const reboundLootGuide = hud.getQuestTrackerEntries(lootGuideSnapshot, {
  keyLabels: { loot: reboundLootLabel }
})[0];
const unboundLootGuide = hud.getQuestTrackerEntries(lootGuideSnapshot, {
  keyLabels: { loot: 'Unbound' }
})[0];
check(defaultLootGuide && defaultLootGuide.objectives[0].label.includes('Hold Z'),
  'the First Steps tracker should name the default loot key');
check(reboundLootGuide && reboundLootGuide.objectives[0].label.includes('Hold G'),
  'the First Steps tracker should update when Loot is rebound');
check(reboundLootGuide && !reboundLootGuide.objectives[0].label.includes('loot key'),
  'the First Steps tracker should replace the generic loot-key wording');
check(unboundLootGuide && /click the nearby prompt/i.test(unboundLootGuide.objectives[0].label) &&
  /bind Loot in Keybinds/i.test(unboundLootGuide.objectives[0].label),
'the unbound First Steps tracker should explain both the working click fallback and rebinding');

const renderedRegions = [];
const renderedText = [];
const promptUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(promptUi, {
  snapshot: createWorldPromptSnapshot(),
  openWindows: [],
  keybinds: reboundKeybinds,
  addCanvasRegion(region) {
    renderedRegions.push(region);
  },
  drawCanvasUiPanel() {},
  drawCanvasText(ctx, value) {
    renderedText.push(String(value || ''));
  }
});
promptUi.drawCanvasStationPrompt({}, 1280, 720, 620);
check(renderedRegions.some((region) => region.type === 'station-prompt' && region.action === 'loot'),
  'the real canvas prompt renderer should publish a clickable loot region');
check(renderedText.includes('Upgrade Dust'),
  'the real canvas prompt renderer should draw the nearby item name');
check(renderedText.some((value) => value.includes('G') && /loot|pick up/i.test(value)),
  'the real canvas prompt renderer should draw the rebound loot key');

let routedCanvasAction = '';
let canvasFocusCount = 0;
const clickUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(clickUi, {
  monsterGuideSearchFocused: false,
  isCommandOpen: false,
  handleAction(action) {
    routedCanvasAction = action;
    return true;
  },
  startPortalTransition() {
    return false;
  },
  focusCanvas() {
    canvasFocusCount += 1;
  }
});
clickUi.executeCanvasRegion({ type: 'station-prompt', action: 'loot' });
check(routedCanvasAction === 'loot',
  'clicking the canvas loot prompt should dispatch the same loot action as the bound key');
check(canvasFocusCount === 1,
  'clicking the canvas loot prompt should return focus to gameplay');

let stationInteractionCount = 0;
let portalTransitionCount = 0;
const overlappingInteractionUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(overlappingInteractionUi, {
  mapTransition: null,
  openWindows: [],
  questPrompt: null,
  engine: {
    state: {
      player: {
        activeStation: 'shop',
        activePortalId: 'greenroot_return'
      }
    },
    interact() {
      stationInteractionCount += 1;
      return true;
    },
    lastInteractionOpenedQuestPrompt: false,
    lastInteractionOpenedPanel: false
  },
  profileUiAction(actionName, callback) {
    return callback();
  },
  recordControlOnboardingEvent() {},
  getActiveStationPanelId() {
    return 'shop';
  },
  startPortalTransition() {
    portalTransitionCount += 1;
    return true;
  }
});
check(overlappingInteractionUi.handleAction('interact') &&
  stationInteractionCount === 1 &&
  portalTransitionCount === 0,
'generic interaction should use an overlapping station before attempting portal travel');
overlappingInteractionUi.engine.state.player.activeStation = '';
overlappingInteractionUi.getActiveStationPanelId = () => '';
stationInteractionCount = 0;
check(overlappingInteractionUi.handleAction('interact') &&
  stationInteractionCount === 0 &&
  portalTransitionCount === 1,
'generic interaction should still start portal travel when no station is active');

const questEngine = createProjectStarfallEngine(null, data);
check(questEngine.chooseClass('fighter'), 'the first-loot fixture should create a playable character');
check(questEngine.changeMap('greenrootMeadow'), 'the first-loot fixture should enter Greenroot Meadow');
check(questEngine.startQuest('first_steps'), 'the first-loot fixture should activate First Steps');
const firstLootStepIndex = data.ONBOARDING_STEPS.findIndex((step) => step.id === 'loot_drop');
questEngine.state.onboarding.completedIds = data.ONBOARDING_STEPS
  .slice(0, firstLootStepIndex)
  .map((step) => step.id);
questEngine.onboardingSnapshotCache = null;

const player = questEngine.state.player;
player.activeStation = '';
player.activePortalId = '';
player.activeQuestNpcId = '';
const questDrop = questEngine.dropLootItem({
  uid: 'first_steps_loot_item',
  id: 'upgrade_dust',
  kind: 'material',
  materialId: 'upgradeDust',
  name: 'Upgrade Dust',
  rarity: 'Common',
  quantity: 1
}, null, {
  landX: player.x + player.w / 2,
  landY: player.y + player.h
});
questDrop.airborne = false;
questDrop.x = player.x + player.w / 2;
questDrop.y = player.y + player.h;
questDrop.vx = 0;
questDrop.vy = 0;
questDrop.settledAt = Date.now();
questEngine.invalidateLootDropCaches();

const lootObjectiveBefore = questEngine.getProgressSnapshot().activeQuest.objectives
  .find((objective) => objective.id === 'loot_drop');
const reachableQuestDrop = questEngine.findReachableLoot(100);
const questPromptBefore = hud.getStationPromptContext({
  state: questEngine.state,
  map: { stations: questEngine.runtime.stations },
  portals: questEngine.runtime.portals,
  questNpcs: questEngine.getQuestNpcSnapshot(),
  onboarding: questEngine.getOnboardingSnapshot(),
  nearbyLoot: reachableQuestDrop
}, { keyLabels: { loot: 'Z' } });
check(questEngine.getOnboardingSnapshot().nextStep &&
  questEngine.getOnboardingSnapshot().nextStep.id === 'loot_drop',
'the deterministic fixture should place the journey guide on its first-loot step');
check(lootObjectiveBefore && lootObjectiveBefore.value === 0 && !lootObjectiveBefore.complete,
  'First Steps should begin with its mandatory loot objective incomplete');
check(reachableQuestDrop === questDrop,
  'the engine should publish the nearest reachable First Steps drop');
check(questPromptBefore && questPromptBefore.promptAction === 'loot',
  'the published First Steps drop should produce the contextual loot prompt');
check(questEngine.lootNearestDrop(100),
  'the prompt action should collect the same reachable drop through the real engine');

const lootObjectiveAfter = questEngine.getProgressSnapshot().activeQuest.objectives
  .find((objective) => objective.id === 'loot_drop');
const onboardingAfterLoot = questEngine.getOnboardingSnapshot();
const questPromptAfter = hud.getStationPromptContext({
  state: questEngine.state,
  map: { stations: questEngine.runtime.stations },
  portals: questEngine.runtime.portals,
  questNpcs: questEngine.getQuestNpcSnapshot(),
  onboarding: questEngine.getOnboardingSnapshot(),
  nearbyLoot: questEngine.findReachableLoot(100)
}, { keyLabels: { loot: 'Z' } });
check(lootObjectiveAfter && lootObjectiveAfter.value === 1 && lootObjectiveAfter.complete,
  'collecting from the prompt should advance and complete the First Steps loot objective');
check(!onboardingAfterLoot.nextStep || onboardingAfterLoot.nextStep.id !== 'loot_drop',
  'collecting the first drop should advance the journey guide beyond its first-loot step');
check(questPromptAfter === null,
  'the contextual loot prompt should disappear immediately after the reachable drop is collected');

if (failures.length) {
  console.error(`Project Starfall first-loot prompt checks failed: ${failures.length}/${checks}`);
  failures.forEach((message, index) => console.error(`${index + 1}. ${message}`));
  process.exitCode = 1;
} else {
  console.log(`Project Starfall first-loot prompt checks passed: ${checks}`);
}
