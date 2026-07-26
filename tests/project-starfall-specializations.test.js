'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
global.ProjectStarfallData = require('../js/games/project-starfall/data/index.js');

const data = global.ProjectStarfallData;
const specializationEngine = require('../js/games/project-starfall/engine/specializations.js');
const skillModifiers = require('../js/games/project-starfall/engine/skill-modifiers.js');
const statMetadata = require('../js/games/project-starfall/ui/stat-metadata.js');
const combatFormulas = require('../js/games/project-starfall/engine/combat-formulas.js');
const hud = require('../js/games/project-starfall/ui/hud.js');
const panels = require('../js/games/project-starfall/ui/panels.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { ProjectStarfallUi } = require('../js/games/project-starfall/project-starfall-ui.js');

let checks = 0;
function check(condition, message) {
  assert(condition, message);
  checks += 1;
}

function read(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), 'utf8');
}

function getSkill(skillId) {
  return (data.SKILLS || []).find((skill) => skill && skill.id === skillId);
}

function createAdvancedEngine(baseClassId, advancedClassId, level) {
  const engine = createProjectStarfallEngine(null, data);
  check(engine.chooseClass(baseClassId), `setup should choose ${baseClassId}`);
  engine.state.player.advancedClassId = advancedClassId;
  engine.state.player.level = Number(level || 60);
  engine.state.mapId = 'starfallCrossing';
  engine.recalculateVitals();
  return engine;
}

const advancedIds = Object.keys(data.ADVANCED_CLASSES || {});
const routeIds = new Set();
const routeCountByAdvancedId = {};
const skillsById = new Map((data.SKILLS || []).map((skill) => [skill.id, skill]));

check(data.SPECIALIZATIONS.length === advancedIds.length * 2,
  'specialization data should expose exactly two routes per advanced class');
data.SPECIALIZATIONS.forEach((specialization) => {
  routeCountByAdvancedId[specialization.advancedId] = (routeCountByAdvancedId[specialization.advancedId] || 0) + 1;
  check(!routeIds.has(specialization.id), `specialization id ${specialization.id} should be unique`);
  routeIds.add(specialization.id);
  ['name', 'badge', 'role', 'summary', 'mechanic', 'tradeoff'].forEach((key) => {
    check(typeof specialization[key] === 'string' && specialization[key].trim().length > 0,
      `${specialization.id} should provide ${key}`);
  });
  check(Number(specialization.levelRequirement) === Number(data.SPECIALIZATION_LEVEL),
    `${specialization.id} should use the shared specialization level`);
  check(Object.keys(specialization.statBonuses || {}).length >= 2,
    `${specialization.id} should provide a meaningful stat bundle`);
  check(Object.keys(specialization.skillModifiers || {}).length >= 1,
    `${specialization.id} should alter at least one authored skill`);
  Object.keys(specialization.skillModifiers || {}).forEach((skillId) => {
    const skill = skillsById.get(skillId);
    check(!!skill, `${specialization.id} should reference an existing skill: ${skillId}`);
    check(skill && skill.owner === specialization.advancedId,
      `${specialization.id} should only modify skills owned by ${specialization.advancedId}`);
  });
});
advancedIds.forEach((advancedId) => {
  check(routeCountByAdvancedId[advancedId] === 2,
    `${advancedId} should offer exactly two specialization routes`);
});
check(Number(data.SPECIALIZATION_RESPEC_COST) === 0,
  'specialization switching should remain free in safe towns');

const specializationScores = data.SPECIALIZATIONS.map((specialization) =>
  Object.entries(specialization.statBonuses || {}).reduce((total, [key, value]) =>
    total + Number(value || 0) * Number(statMetadata.STAT_SCORE_WEIGHTS[key] || 1), 0));
const minSpecializationScore = Math.min(...specializationScores);
const maxSpecializationScore = Math.max(...specializationScores);
check(minSpecializationScore >= 46 && maxSpecializationScore <= 50,
  'specialization stat bundles should stay inside the reviewed power band');
check(maxSpecializationScore - minSpecializationScore <= 3,
  'specialization stat bundles should not create a dominant static route');

const mergedModifier = skillModifiers.mergeSkillModifiers(
  { damageScale: 1.1, cooldownScale: 0.9, resourceCostScale: 0.95, markDuration: 5, extraLines: 1 },
  { damageScale: 1.02, cooldownScale: 0.94, resourceCostScale: 1.04, markDuration: 6, extraLines: 1 }
);
check(Math.abs(mergedModifier.damageScale - 1.122) < 0.000001,
  'generic and specialization damage scales should multiply');
check(Math.abs(mergedModifier.cooldownScale - 0.846) < 0.000001,
  'generic and specialization cooldown scales should multiply');
check(Math.abs(mergedModifier.resourceCostScale - 0.988) < 0.000001,
  'generic and specialization resource costs should multiply');
check(mergedModifier.markDuration === 6 && mergedModifier.extraLines === 2,
  'duration and line metadata should merge without discarding either source');

const guardianRoutes = data.SPECIALIZATIONS.filter((specialization) => specialization.advancedId === 'guardian');
const helperPlayer = { advancedClassId: 'guardian', level: 60, currency: 1000 };
const helperState = specializationEngine.createSpecializationState(null, { data });
check(specializationEngine.getSpecializationLockReason(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: false
}) === 'Return to a town to choose a path.',
'specialization selection should require a safe town');
check(specializationEngine.getSpecializationLockReason(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: true,
  trialActive: true
}) === 'Finish the active class trial before choosing a path.',
'active class trials should block specialization changes');
check(specializationEngine.getSpecializationLockReason(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: true,
  dungeonActive: true
}) === 'Finish or leave the active dungeon before choosing a path.',
'active dungeons should block specialization changes');
check(specializationEngine.getSpecializationLockReason(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: true,
  riftActive: true
}) === 'Bank or end the Rift run before choosing a path.',
'active Rift runs should block specialization changes');
const pendingPlan = specializationEngine.createSpecializationChoicePlan(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: true
});
check(!pendingPlan.ok && pendingPlan.requiresConfirmation && pendingPlan.cost === 0,
  'the first route should be free but require explicit confirmation');
const confirmedPlan = specializationEngine.createSpecializationChoicePlan(guardianRoutes[0], helperPlayer, {
  data,
  specializations: helperState,
  safeZone: true,
  confirmed: true
});
check(confirmedPlan.ok && confirmedPlan.specializationId === guardianRoutes[0].id,
  'a confirmed safe-town choice should produce an atomic mutation plan');

const engine = createAdvancedEngine('fighter', 'guardian', 60);
const firstRoute = guardianRoutes.find((route) => route.id === 'guardian_aegis_captain');
const secondRoute = guardianRoutes.find((route) => route.id === 'guardian_impact_marshal');
const initialCurrency = engine.state.player.currency;
const initialStats = engine.getStats();
const initialSnapshot = engine.getSpecializationSnapshot();
check(initialSnapshot.selectionPending && initialSnapshot.specializations.filter((route) => route.available).length === 2,
  'a level-60 advanced character should see two pending routes');
check(!engine.chooseSpecialization(firstRoute.id) && !engine.getActiveSpecialization(),
  'an unconfirmed engine call should not mutate specialization state');
check(engine.chooseSpecialization(firstRoute.id, { confirmed: true }),
  'a confirmed first route should activate in town');
check(engine.getActiveSpecialization().id === firstRoute.id &&
  engine.getSpecializationSnapshot().selectionPending === false,
  'the active route should be reflected in engine and snapshot state');
const aegisStats = engine.getStats();
check(aegisStats.maxHp >= initialStats.maxHp + Number(firstRoute.statBonuses.hp || 0) &&
  aegisStats.defense >= initialStats.defense + Number(firstRoute.statBonuses.defense || 0),
  'the active specialization should refresh and apply its stat bonuses');
const impactGuardModifier = engine.getSkillModifierForSkill(getSkill('guardian_impact_guard'));
check(impactGuardModifier && Math.abs(impactGuardModifier.cooldownScale - 0.94) < 0.000001,
  'the active specialization should overlay authored skill behavior');
engine.state.mapId = 'greenrootMeadow';
check(!engine.chooseSpecialization(secondRoute.id, { confirmed: true }) &&
  engine.getActiveSpecialization().id === firstRoute.id,
  'field switching should be rejected without changing the active route');
engine.state.mapId = 'starfallCrossing';
check(engine.chooseSpecialization(secondRoute.id, { confirmed: true }) &&
  engine.getActiveSpecialization().id === secondRoute.id,
  'free town switching should replace the old route atomically');
check(engine.state.player.currency === initialCurrency,
  'free town switching should not consume currency');
check(!engine.getSkillModifierForSkill(getSkill('guardian_impact_guard')),
  'switching routes should remove the old route behavior');
const shieldBashModifier = engine.getSkillModifierForSkill(getSkill('guardian_shield_bash'));
check(shieldBashModifier && shieldBashModifier.breakScale === 1.08,
  'switching routes should apply the new route behavior');

const saved = engine.serialize();
const restoredEngine = createProjectStarfallEngine(null, data);
check(restoredEngine.restore(saved) &&
  restoredEngine.getActiveSpecialization() &&
  restoredEngine.getActiveSpecialization().id === secondRoute.id,
  'specialization choices should survive serialization and restore');
const legacyPayload = JSON.parse(JSON.stringify(saved));
legacyPayload.state.specializations = {
  selectedByAdvancedId: {
    guardian: 'guardian_aegis_captain'
  }
};
const legacyEngine = createProjectStarfallEngine(null, data);
check(legacyEngine.restore(legacyPayload) &&
  legacyEngine.getActiveSpecialization() &&
  legacyEngine.getActiveSpecialization().id === 'guardian_aegis_captain',
  'legacy one-route specialization saves should remain valid');

const levelEngine = createAdvancedEngine('fighter', 'guardian', 59);
const levelToasts = [];
levelEngine.setToastHandler((message) => levelToasts.push(message));
levelEngine.state.player.xp = combatFormulas.getLevelXp(59);
levelEngine.checkLevelUp({ noEmit: true });
check(levelEngine.state.player.level === 60 &&
  levelToasts.some((message) => message.includes('Specialization ready') && message.includes('Character > Class')),
  'crossing level 60 should point the player directly to the specialization choice');

const historicalEngine = createAdvancedEngine('fighter', 'guardian', 60);
historicalEngine.state.specializations.selectedByAdvancedId.guardian = firstRoute.id;
historicalEngine.state.specializations.selectedByAdvancedId.sniper = 'sniper_deadeye_commander';
const characterOverlay = historicalEngine.createOverlaySnapshot({ openPanels: ['character'] });
check(characterOverlay.specializations.specializations.filter((route) => route.available).length === 2,
  'the Character overlay should load only two available routes for the current class');
check((characterOverlay.roster.traits || []).length > 0,
  'the Character growth view should load its roster traits instead of an empty placeholder');
const characterUi = Object.create(ProjectStarfallUi.prototype);
Object.assign(characterUi, {
  snapshot: characterOverlay,
  getStatUpgradeState: () => ({ budget: {}, definitions: [], allocations: {} }),
  getSecondaryResourceName: () => 'Guard'
});
const characterContext = characterUi.getCharacterPanelContext();
const specializationMarkup = characterUi.renderCharacterSpecializationPanel(characterContext);
check((specializationMarkup.match(/data-starfall-specialization=/g) || []).length === 2,
  'historical choices from other classes should not leak into the Character route cards');

const promptUi = Object.create(ProjectStarfallUi.prototype);
let promptConfig = null;
let promptMutation = null;
let promptToast = '';
Object.assign(promptUi, {
  snapshot: {
    specializations: Object.assign({}, initialSnapshot, {
      specializations: initialSnapshot.specializations
    })
  },
  engine: {
    chooseSpecialization(specializationId, options) {
      promptMutation = { specializationId, options };
      return true;
    }
  },
  showToast(message) {
    promptToast = message;
  },
  openConfirmPrompt(config) {
    promptConfig = config;
    return true;
  }
});
check(promptUi.openSpecializationChoicePrompt(firstRoute.id) && promptConfig && !promptMutation,
  'choosing a route in the UI should open a prompt without mutating state');
check(promptConfig.onConfirm() && promptMutation &&
  promptMutation.specializationId === firstRoute.id &&
  promptMutation.options.confirmed === true,
  'only prompt confirmation should call the engine with explicit confirmation');
promptUi.snapshot.specializations.specializations = promptUi.snapshot.specializations.specializations.map((route) =>
  Object.assign({}, route, { lockedReason: route.id === secondRoute.id ? 'Return to a town to choose a path.' : route.lockedReason }));
check(!promptUi.openSpecializationChoicePrompt(secondRoute.id) &&
  promptToast === 'Return to a town to choose a path.',
  'locked route clicks should explain the block without opening a prompt');

const rootMenuItems = hud.getCanvasMenuGroups({
  specializations: { selectionPending: true },
  dailyLogin: { claimable: false }
}, { pageId: 'root' }).flatMap((group) => group.items || []);
check(rootMenuItems.some((item) => item.panel === 'character' && item.label === 'Character!'),
  'the command menu should keep a compact specialization-ready nudge');
check(panels.getDomPanelRenderMethod('beta') === 'renderRewardsStylePanel' &&
  panels.getDomPanelPresentation('beta').title === 'Rewards & Style',
  'the renamed rewards panel should have a dedicated DOM renderer instead of falling back to Character');
const derivedUpdate = panels.getPanelDerivedSnapshotUpdate({
  getSpecializationSnapshot: () => ({ selectedId: firstRoute.id }),
  getPlinkoSnapshot: () => ({ pity: 1 }),
  getCashShopSnapshot: () => ({ balance: 2 }),
  getMarketSnapshot: () => ({ listings: [{ id: 'market' }] }),
  getCosmeticSnapshot: () => ({ cosmetics: [{ id: 'cosmetic' }] }),
  getSeasonSnapshot: () => ({ activeSeason: { id: 'season' } })
}, ['skills', 'session', 'shop']);
check(derivedUpdate.specializations.selectedId === firstRoute.id &&
  derivedUpdate.market.listings.length === 1 &&
  derivedUpdate.cosmetics.cosmetics.length === 1 &&
  derivedUpdate.season.activeSeason.id === 'season',
  'incremental DOM refreshes should keep specialization and rewards state current');

const trapEngine = createAdvancedEngine('archer', 'trapper', 60);
const trapSkill = getSkill('trapper_spike_trap');
trapEngine.state.player.buffs.tacticalField = Date.now() / 1000 + 30;
trapEngine.invalidateStatsCache();
const trapStats = trapEngine.getStats();
const trapBasePower = trapEngine.getSkillBasePower(trapSkill, 1, trapStats);
check(trapEngine.tryUseSignatureSkill(trapSkill, 1, trapStats),
  'Tactical Field regression setup should place a trap');
const tacticalTrap = trapEngine.state.player.activeSkillObjects.slice(-1)[0];
check(tacticalTrap &&
  Math.abs(tacticalTrap.damage / (trapBasePower * 0.96) - 1.15) < 0.000001,
  'Tactical Field should apply its advertised trap bonus once, not twice');
check(tacticalTrap.armedAt - Date.now() / 1000 <= 0.06,
  'Tactical Field should retain its faster trap arming');

['fire_mage_wildfire', 'fire_mage_inferno_burst'].forEach((skillId) => {
  const fireEngine = createAdvancedEngine('mage', 'fireMage', 60);
  const fireSkill = getSkill(skillId);
  const enemy = { uid: `${skillId}-target`, x: 100, y: 100, w: 40, h: 60, hp: 1000, burning: 0, marked: 0 };
  let spreadCalls = 0;
  fireEngine.findRoleTarget = () => enemy;
  fireEngine.findEnemiesNear = () => [enemy];
  fireEngine.spreadBurnFromEnemy = () => {
    spreadCalls += 1;
    return 0;
  };
  fireEngine.hitRoleTarget = (target, skill, amount, settings) => {
    fireEngine.applyEnemyControl(target, settings);
    fireEngine.applySkillHitEffects(skill, target, null);
    return 1;
  };
  check(fireEngine.tryUseSignatureSkill(fireSkill, 1, fireEngine.getStats()) && spreadCalls === 1,
    `${skillId} should spread burn exactly once per direct target`);
});

const sniperEngine = createAdvancedEngine('archer', 'sniper', 60);
sniperEngine.damageEnemy = () => {};
sniperEngine.pushSkillImpactEffect = () => {};
sniperEngine.applySkillModifierHitEffects = () => {};
sniperEngine.applyBossBreakProgress = () => {};
['sniper_execution_shot', 'sniper_one_perfect_shot'].forEach((skillId) => {
  const weakPointSamples = [];
  const enemy = { uid: `${skillId}-target`, x: 100, y: 100, w: 40, h: 60, hp: 1000, weakPoint: 8, marked: 8 };
  sniperEngine.rollDamageResult = (amount, target) => {
    weakPointSamples.push(Number(target.weakPoint || 0));
    return { amount, critical: false };
  };
  sniperEngine.damageEnemyWithSkillLines(enemy, 100, getSkill(skillId), { lineCount: 4 });
  check(weakPointSamples.length === 4 && weakPointSamples.every((value) => value > 0) && enemy.weakPoint === 0,
    `${skillId} should cash out weak point after every damage line resolves`);
});
const aimedEnemy = { uid: 'aimed-target', x: 100, y: 100, w: 40, h: 60, hp: 1000, weakPoint: 8, marked: 8 };
sniperEngine.rollDamageResult = (amount) => ({ amount, critical: false });
sniperEngine.damageEnemyWithSkillLines(aimedEnemy, 100, getSkill('sniper_aimed_shot'), { lineCount: 1 });
check(aimedEnemy.weakPoint === 8,
  'ordinary Aimed Shot should preserve weak point for a finisher');

const engineSource = read('js/games/project-starfall/project-starfall-engine.js');
const uiSource = read('js/games/project-starfall/project-starfall-ui.js');
check(engineSource.includes("domains: ['hud', 'skills', 'session']") &&
  engineSource.includes("reason: 'specialization'") &&
  engineSource.includes("persist: true"),
  'specialization mutations should publish a focused persistent UI change');
check(uiSource.includes('renderCharacterSpecializationPanel(context)') &&
  uiSource.includes('drawCharacterSpecializationCanvas(ctx, x, y, w, context)') &&
  !uiSource.includes("'Beta Systems'"),
  'specialization presentation should live in Character > Class without the misleading beta label');

console.log(`Project Starfall specialization checks passed: ${checks}`);
