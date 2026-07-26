'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const onboardingData = require('../js/games/project-starfall/data/onboarding.js');
const starfallData = require('../js/games/project-starfall/data/index.js');
const mapBuilders = require('../js/games/project-starfall/data/map-builders.js');
const mapLayouts = require('../js/games/project-starfall/data/map-layouts.js');
const mapTown = require('../js/games/project-starfall/data/map-town.js');
const engineState = require('../js/games/project-starfall/engine/state.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const canvasHover = require('../js/games/project-starfall/ui/canvas-hover.js');
const canvasWindows = require('../js/games/project-starfall/ui/canvas-windows.js');
const questUi = require('../js/games/project-starfall/ui/quests.js');
const resourceWidgets = require('../js/games/project-starfall/ui/resource-widgets.js');

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), 'utf8');
}

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

const reconciled = engineState.reconcileOnboardingState(
  { hidden: false, completedIds: [] },
  { classId: 'fighter', advancedClassId: 'guardian' },
  onboardingData
);
check(reconciled.completedIds.includes('choose_class'), 'restored class saves should reconcile the class-selection milestone');
check(reconciled.completedIds.includes('choose_advanced'), 'restored advanced-class saves should reconcile the advancement milestone');

const onboarding = engineState.createOnboardingSnapshot(reconciled, onboardingData.ONBOARDING_STEPS);
check(onboarding.nextStep && onboarding.nextStep.id === 'learn_move', 'restored characters should advance past class selection');
check(onboarding.activePhase && onboarding.activePhase.title === 'First Steps' &&
  onboarding.activePhase.completeCount === 1 &&
  onboarding.activePhase.total === 9,
  'onboarding should expose focused first-phase progress');
check(onboarding.phases.map((phase) => phase.title).join('|') === 'First Steps|Gear & Skills|Workshop & Party|Advancement',
  'onboarding should be organized into four readable phases');

const restoredEngine = createProjectStarfallEngine(null, starfallData);
restoredEngine.state.player.classId = 'fighter';
restoredEngine.state.player.advancedClassId = 'guardian';
restoredEngine.state.onboarding = { hidden: false, completedIds: [] };
const restoredOnboarding = restoredEngine.getOnboardingSnapshot();
check(restoredOnboarding.nextStep && restoredOnboarding.nextStep.id === 'learn_move' &&
  restoredOnboarding.activePhase && restoredOnboarding.activePhase.title === 'First Steps',
  'the real engine snapshot should reconcile restored saves before rendering onboarding');

const trackerEntries = hud.getQuestTrackerEntries({ onboarding, progress: {} });
check(trackerEntries[0] && trackerEntries[0].title === 'First Steps 1/9: Move through town',
  'quest tracking should name the active onboarding phase instead of a flat 21-step guide');

const crossing = starfallData.MAPS.find((map) => map.id === 'starfallCrossing');
const greenroot = starfallData.MAPS.find((map) => map.id === 'greenrootMeadow');
check(crossing && crossing.questNpcs.some((npc) => npc.id === 'crossing_trail_guide' &&
  npc.x === 560 &&
  npc.platformIndex === 0 &&
  npc.questIds.includes('first_steps')),
  'Starfall Crossing should offer First Steps before the player travels');
check(greenroot && greenroot.questNpcs.some((npc) => npc.id === 'greenroot_guide' && npc.questIds.includes('first_steps')),
  'Greenroot should retain the First Steps handoff and reward route');

const crossingGateLandmark = crossing.townScene.rearStructures.find((entry) => entry.label === 'Greenroot Gate');
const crossingGate = crossing.portals.find((portal) => portal.id === 'crossing_greenroot');
const crossingRouteEngine = createProjectStarfallEngine(null, starfallData);
crossingRouteEngine.chooseClass('fighter');
const runtimeCrossingGate = crossingRouteEngine.runtime.portals.find((portal) => portal.id === 'crossing_greenroot');
const runtimeCrossingNpcs = crossingRouteEngine.runtime.questNpcs;
const crossingGateCenter = runtimeCrossingGate.x + runtimeCrossingGate.w / 2;
const crossingGateVisualW = Math.max(96, runtimeCrossingGate.w * 1.82);
const crossingGateVisualLeft = crossingGateCenter - crossingGateVisualW / 2;
const crossingGateVisualRight = crossingGateCenter + crossingGateVisualW / 2;
check(crossingGateLandmark && crossingGate && runtimeCrossingGate &&
  Math.abs(crossingGateCenter - (crossingGateLandmark.x + crossingGateLandmark.w / 2)) <= 1,
  'the functional Greenroot Gate should remain centered beneath its authored town arch');
check(runtimeCrossingNpcs.every((npc) =>
  npc.x + npc.w <= crossingGateVisualLeft || npc.x >= crossingGateVisualRight
), 'the Greenroot Gate artwork should remain visually separate from every town NPC');

const crossingPlayer = crossingRouteEngine.state.player;
crossingPlayer.x = crossingGateCenter - crossingPlayer.w / 2;
crossingPlayer.y = runtimeCrossingGate.y + runtimeCrossingGate.h - crossingPlayer.h;
crossingRouteEngine.activeInteractionTargetCache = null;
const crossingGateTarget = crossingRouteEngine.updateActiveStation();
check(crossingGateTarget.portalId === 'crossing_greenroot' && !crossingGateTarget.questNpcId,
  'standing at the Greenroot Gate should activate travel without also targeting a quest NPC');
const crossingGatePrompt = hud.getStationPromptContext({
  state: { player: crossingPlayer },
  map: { stations: crossingRouteEngine.runtime.stations },
  portals: crossingRouteEngine.runtime.portals,
  questNpcs: { npcs: runtimeCrossingNpcs }
}, { keyLabels: { moveUp: 'Up' } });
check(crossingGatePrompt &&
  crossingGatePrompt.title === 'Greenroot Gate' &&
  crossingGatePrompt.promptAction === 'portal' &&
  crossingGatePrompt.kindLabel === 'Portal' &&
  crossingGatePrompt.hint === 'Up Travel' &&
  crossingGatePrompt.target.id === 'crossing_greenroot',
  'the aligned gate should publish one coherent Greenroot travel prompt');

const assignedNpc = mapTown.assignTownQuestNpcs([{ id: 'placed_guide', x: 560, platformIndex: 0 }])[0];
check(assignedNpc.x === 560 && assignedNpc.platformIndex === 0,
  'town quest placement should preserve authored ground positions');

const rimewardenLayout = mapLayouts.DUNGEON_ARENA_SKELETONS.rimewardenSanctum;
const rimewardenPlatforms = mapBuilders.makeDungeonArenaPlatforms(4800, 'rimewardenSanctum');
const firstRimewardenSlope = rimewardenPlatforms.find((platform) => platform.shape === 'slope');
check(rimewardenLayout.entryClearance >= 400 &&
  firstRimewardenSlope &&
  firstRimewardenSlope.x >= rimewardenLayout.entryClearance,
  'Rimewarden Sanctum should reserve a readable flat arrival apron before its first ramp');

const guardian = starfallData.ADVANCED_CLASSES.guardian;
const guardianWidget = resourceWidgets.getResourceWidgetData({
  state: {
    player: {
      classId: 'fighter',
      advancedClassId: 'guardian',
      hp: 100,
      resource: 80,
      activeSkillObjects: [],
      classMechanics: { guardianImpact: 45 }
    }
  },
  stats: { maxHp: 100, secondaryResourceMax: 100 },
  advancedData: guardian,
  enemies: [],
  activeBuffs: []
});
check(guardian.resourceName === 'Resolve' && guardian.classMechanicName === 'Stored Impact',
  'Guardian generic skill energy and defensive class mechanic should have distinct explicit labels');
check(guardianWidget.label === 'Stored Impact' &&
  guardianWidget.value === 45 &&
  guardianWidget.max === 120 &&
  guardianWidget.detail === '45/120 impact stored',
  'Guardian defensive pressure should remain clearly labeled as Stored Impact');

const engineSource = read('js/games/project-starfall/project-starfall-engine.js');
check(!/guardianImpact[^;\n]*,\s*0,\s*140\)/.test(engineSource),
  'all Guardian Stored Impact mutations should use the documented 120 cap');
check(engineState.createClassMechanicsState({ guardianImpact: 999 }).guardianImpact === 120,
  'Guardian Stored Impact restore normalization should enforce the same 120 cap');

const uiSource = read('js/games/project-starfall/project-starfall-ui.js');
check(uiSource.includes('this.runMapTransition(restoredMapId, restoredMap && restoredMap.name, () => this.enterActiveCharacter())'),
  'restored characters should preload their saved map before the game stage is revealed');

const characterSelectCss = read('css/games/project-starfall/character-select.css');
check(/\.project-starfall-character-preview\s*\{[^}]*display:\s*none;/s.test(characterSelectCss) &&
  /\.project-starfall-character-slot\.has-preview-frame \.project-starfall-character-art-fallback\s*\{[^}]*opacity:\s*1;/s.test(characterSelectCss),
  'character selection should show the existing playful class art instead of the blocky procedural preview');
check(characterSelectCss.includes('img/project-starfall/maps/starfall-crossing.webp') &&
  !characterSelectCss.includes('img/project-starfall/ui/character-select-screen.png') &&
  /\.project-starfall-character-create-modal\s*\{[^}]*box-sizing:\s*border-box;[^}]*overflow-x:\s*hidden;/s.test(characterSelectCss),
  'character selection should reuse the playful in-game town art and avoid horizontal modal overflow');

const loadingCss = read('css/games/project-starfall/loading.css');
check(loadingCss.includes('img/project-starfall/maps/starfall-crossing.webp') &&
  !loadingCss.includes('img/project-starfall/ui/splash-screen.png') &&
  !loadingCss.includes('img/project-starfall/ui/start-screen.png'),
  'loading and start screens should transition through the same playful world art as gameplay');

const page = read('pages/games/project-starfall.html');
check(page.includes('<title>Project Starfall | Daniel Short</title>') &&
  page.includes('<h1>Project Starfall</h1>') &&
  page.includes('Choose a compact chibi hero') &&
  !page.includes('Project Starfall Prototype') &&
  !page.includes('static browser prototype'),
  'the public game page should present Starfall as a cohesive playable game rather than a placeholder prototype');

const gamesDirectory = read('content/pages/games.json');
check(gamesDirectory.includes('"image": "img/project-starfall/maps/starfall-crossing.webp"') &&
  !gamesDirectory.includes('"image": "img/project-starfall/ui/splash-screen.png"'),
  'the games directory should preview Starfall with its playful in-game world art');

const gdd = read('project_starfall_gdd_v0_5.md');
check(gdd.includes("keep Starfall's compact chibi cast original, playful, and readable") &&
  !gdd.includes('Using a chibi visual style that feels too close'),
  'the design direction should preserve an original playful chibi identity');

const spatialBossHud = hud.getCanvasBossEncounterHudMetadata({
  active: true,
  bossName: 'Astral Archivist',
  hpRatio: 0.74,
  phaseName: 'Opened Index',
  phaseIndex: 0,
  phaseCount: 3,
  pendingActionLabel: 'MEMORY SEAL',
  pendingActionProgress: 0.42,
  pendingSpatialSectionLabel: 'Left Archive Seal',
  color: '#c794ff',
  accentColor: '#64d9c5'
}, 1280);
check(spatialBossHud &&
  spatialBossHud.box.h === 90 &&
  spatialBossHud.actionText.value === 'MEMORY SEAL 42%' &&
  spatialBossHud.spatialText.value === 'CALL · Left Archive Seal' &&
  spatialBossHud.spatialBand,
  'spatial boss calls should add a readable location band without replacing the compact boss HUD');

const standardBossHud = hud.getCanvasBossEncounterHudMetadata({
  active: true,
  bossName: 'Brambleking',
  hpRatio: 1,
  phaseName: 'Root Court',
  phaseIndex: 0,
  phaseCount: 3
}, 1280);
check(standardBossHud &&
  standardBossHud.box.h === 72 &&
  !standardBossHud.spatialText &&
  !standardBossHud.spatialBand,
  'boss encounters without a live spatial call should retain the original compact HUD height');

const pixiRendererSource = read('js/games/project-starfall/project-starfall-renderer-pixi.js');
check(pixiRendererSource.includes('renderBossHazardEffect(effect, simplified)') &&
  pixiRendererSource.includes("if (type === 'bossHazard')") &&
  pixiRendererSource.includes('effect.spatialSectionLabel'),
  'the active Pixi renderer should preserve authored boss hazard shapes and their spatial labels');
check(['bramble', 'gear', 'core', 'furnace', 'storm', 'rime', 'astral', 'eclipse']
  .every((variant) => pixiRendererSource.includes(`variant === '${variant}'`)),
  'Pixi boss hazards should retain the authored biome motifs already present in the Canvas renderer');
check(pixiRendererSource.indexOf("id.includes('rune')") < pixiRendererSource.indexOf("id.includes('stair')"),
  'rune stair identifiers should receive the existing playful Astral palette before generic stair styling');
check(pixiRendererSource.includes("const boss = enemy.behavior === 'boss';") &&
  pixiRendererSource.includes("boss && enemy.id === 'astralArchivist'") &&
  pixiRendererSource.includes("this.drawShape('world', 'ring'") &&
  pixiRendererSource.includes("boss ? 1 : enemy.telegraph > 0 ? 0.78 : 1"),
  'the Archivist should keep its encounter-colored ground anchor below hazards without removing normal enemy telegraph fading');

const bossUiSource = read('js/games/project-starfall/project-starfall-ui.js');
const hudCssSource = read('css/games/project-starfall/hud.css');
check(bossUiSource.includes('project-starfall-boss-hud-call') &&
  bossUiSource.indexOf('CALL · ${spatialSectionLabel}') < bossUiSource.indexOf('project-starfall-boss-hud-call') &&
  hudCssSource.includes('.project-starfall-boss-hud-call'),
  'the DOM boss HUD should reserve a dedicated location-first band for spatial calls');

const standardHudVisibility = canvasWindows.getCanvasBackgroundHudVisibility(['inventory']);
const worldMapHudVisibility = canvasWindows.getCanvasBackgroundHudVisibility(['worldmap']);
const stackedWorldMapHudVisibility = canvasWindows.getCanvasBackgroundHudVisibility(['worldmap', 'inventory']);
const canvasWindowHelpers = canvasWindows.createCanvasWindowUiHelpers();
const canvasHoverHelpers = canvasHover.createCanvasHoverUiHelpers();
const questHelpers = questUi.createQuestUiHelpers();
check(standardHudVisibility.minimap && standardHudVisibility.questTracker && standardHudVisibility.riftTracker,
  'ordinary windows should preserve the normal background HUD');
check(!worldMapHudVisibility.minimap && !worldMapHudVisibility.questTracker && !worldMapHudVisibility.riftTracker &&
  !stackedWorldMapHudVisibility.minimap && !stackedWorldMapHudVisibility.questTracker && !stackedWorldMapHudVisibility.riftTracker,
  'the world map should remain the dominant window even when another panel is stacked with it');
check(canvasWindowHelpers.getCanvasBackgroundHudVisibility === canvasWindows.getCanvasBackgroundHudVisibility &&
  canvasHoverHelpers.getWorldMapNodeTooltipMetadata === canvasHover.getWorldMapNodeTooltipMetadata &&
  questHelpers.getWorldMapDetailPresentation === questUi.getWorldMapDetailPresentation,
  'the live UI helper groups should expose the same world-map hierarchy contracts as direct tests');

const selectedMapNode = {
  selected: true,
  mapId: 'starfallCrossing',
  name: 'Starfall Crossing',
  areaName: 'Starfall Crossing',
  layoutRoleLabel: 'Town Hub',
  levelRange: [1, 99],
  current: true
};
const selectedMapRegion = Object.assign({
  type: 'world-map-node',
  mapId: selectedMapNode.mapId,
  x: 0,
  y: 0,
  w: 36,
  h: 36
}, canvasHover.getWorldMapNodeTooltipMetadata(selectedMapNode));
const selectedMapHover = canvasHover.getCanvasHoverTargetAt({ x: 18, y: 18 }, {
  openWindows: ['worldmap'],
  findHoverRegion: (filter) => filter(selectedMapRegion) ? selectedMapRegion : null
});
const selectedMapAction = questUi.getWorldProgressRegionAction(selectedMapRegion);
check(!Object.prototype.hasOwnProperty.call(selectedMapRegion, 'tooltipTitle') &&
  selectedMapHover.type === '' &&
  selectedMapAction.handled &&
  selectedMapAction.type === 'selectWorldMapNode' &&
  selectedMapAction.mapId === selectedMapNode.mapId,
  'a selected map node should keep its click target without repeating the detail card in a tooltip');

const availableMapNode = Object.assign({}, selectedMapNode, {
  selected: false,
  current: false,
  mapId: 'greenrootMeadow',
  name: 'Greenroot Meadow',
  areaName: 'Greenroot',
  layoutRoleLabel: 'Starter Field',
  levelRange: [1, 8]
});
const availableMapRegion = Object.assign({
  type: 'world-map-node',
  mapId: availableMapNode.mapId,
  x: 0,
  y: 0,
  w: 36,
  h: 36
}, canvasHover.getWorldMapNodeTooltipMetadata(availableMapNode));
const availableMapHover = canvasHover.getCanvasHoverTargetAt({ x: 18, y: 18 }, {
  openWindows: ['worldmap'],
  findHoverRegion: (filter) => filter(availableMapRegion) ? availableMapRegion : null
});
check(availableMapRegion.tooltipTitle === 'Greenroot Meadow' &&
  availableMapHover.type === 'worldMapNode' &&
  availableMapHover.title === 'Greenroot Meadow',
  'unselected map nodes should retain concise discovery tooltips');

const worldMapDetail = questUi.getWorldMapDetailPresentation({
  name: 'Starfall Crossing',
  areaName: 'Starfall Crossing',
  layoutRoleLabel: 'Town Hub',
  levelRange: [1, 99],
  current: true,
  routeStage: 'Town Hub',
  mapRoadName: 'Crossing Plaza',
  enemyLabel: 'intentionally ignored',
  dropLabel: 'intentionally ignored'
}, {
  contextText: 'Landmark: central meteor plaza'
});
const detailFontSizes = [
  questUi.WORLD_MAP_DETAIL_LAYOUT.metaFont,
  questUi.WORLD_MAP_DETAIL_LAYOUT.statusFont,
  questUi.WORLD_MAP_DETAIL_LAYOUT.contextFont
].map((font) => Number((String(font).match(/(\d+)px/) || [0, 0])[1]));
check(worldMapDetail.title === 'Starfall Crossing' &&
  worldMapDetail.meta === 'Starfall Crossing - Town Hub - Lv 1-99' &&
  worldMapDetail.status === 'Current location - Town Hub - Crossing Plaza' &&
  worldMapDetail.context === 'Landmark: central meteor plaza',
  'the selected map card should expose one compact location, level, route-status, and context hierarchy');
check(questUi.WORLD_MAP_DETAIL_LAYOUT.height <= 120 &&
  detailFontSizes.every((fontSize) => fontSize >= 12),
  'the compact map card should reserve readable metadata type instead of stacking tiny analytics rows');

console.log(`Project Starfall playful polish checks passed: ${checks}`);
