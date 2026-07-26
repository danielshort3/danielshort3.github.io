(function initProjectStarfallUiQuests(global) {
  'use strict';

  const QUEST_TAB_OPTIONS = Object.freeze([
    { id: 'quests', label: 'Quests' },
    { id: 'trials', label: 'Trials' },
    { id: 'dungeons', label: 'Dungeons' },
    { id: 'accomplishments', label: 'Accomplishments' }
  ]);

  const WORLD_MAP_DETAIL_LAYOUT = Object.freeze({
    height: 112,
    buttonWidth: 108,
    titleFont: '900 16px system-ui',
    metaFont: '850 12px system-ui',
    statusFont: '900 12px system-ui',
    contextFont: '12px system-ui',
    titleLineHeight: 18,
    metaLineHeight: 14,
    statusLineHeight: 14,
    contextLineHeight: 14
  });

  const ACCOMPLISHMENT_CATEGORY_ORDER = Object.freeze([
    'Onboarding',
    'Combat',
    'Exploration',
    'Crafting',
    'Collection',
    'Class',
    'Dungeon',
    'Boss',
    'Mastery'
  ]);

  const ACCOMPLISHMENT_TIER_ORDER = Object.freeze(['Bronze', 'Silver', 'Gold', 'Relic', 'Mythic']);

  function getAccomplishmentCategoryRank(accomplishment, options) {
    const order = options && options.categoryOrder || ACCOMPLISHMENT_CATEGORY_ORDER;
    const index = order.indexOf(accomplishment && accomplishment.category || '');
    return index < 0 ? order.length : index;
  }

  function getAccomplishmentTierRank(accomplishment, options) {
    const order = options && options.tierOrder || ACCOMPLISHMENT_TIER_ORDER;
    const index = order.indexOf(accomplishment && accomplishment.tier || '');
    return index < 0 ? order.length : index;
  }

  function sortAccomplishmentItems(items, options) {
    const source = Array.isArray(items) ? items : [];
    const settings = options || {};
    return source.slice().sort((a, b) => {
      const categoryDelta = getAccomplishmentCategoryRank(a && a.accomplishment, settings) -
        getAccomplishmentCategoryRank(b && b.accomplishment, settings);
      if (categoryDelta) return categoryDelta;
      const claimedDelta = Number(!!(a && a.accomplishment && a.accomplishment.claimed)) -
        Number(!!(b && b.accomplishment && b.accomplishment.claimed));
      if (claimedDelta) return claimedDelta;
      const tierDelta = getAccomplishmentTierRank(a && a.accomplishment, settings) -
        getAccomplishmentTierRank(b && b.accomplishment, settings);
      return tierDelta || Number(a && a.index || 0) - Number(b && b.index || 0);
    });
  }

  function getWorldMapDetailPresentation(node, options) {
    const source = node || {};
    const settings = options || {};
    const levelRange = Array.isArray(source.levelRange) ? source.levelRange : [];
    const levelLabel = levelRange.length ? `Lv ${levelRange[0]}-${levelRange[1]}` : 'Any level';
    const roleLabel = source.layoutRoleLabel || (source.dungeon ? 'Dungeon' : source.type || 'Field');
    const defaultStatus = source.current
      ? 'Current location'
      : source.lockedReason || (source.completed ? 'Route objective complete' : 'Available by portal route');
    const statusLabel = String(settings.statusText || defaultStatus || '').trim();
    const routeLabel = [source.routeStage, source.mapRoadName].filter(Boolean).join(' - ');
    return {
      title: source.name || 'Unknown area',
      meta: [source.areaName || source.region || 'Region', roleLabel, levelLabel].filter(Boolean).join(' - '),
      status: [statusLabel, routeLabel].filter(Boolean).join(' - '),
      context: String(settings.contextText || '').trim()
    };
  }

  function getWorldProgressDomAction(target) {
    const source = target || null;
    const getAttribute = source && typeof source.getAttribute === 'function'
      ? (name) => source.getAttribute(name)
      : () => null;
    const targetFarmId = getAttribute('data-starfall-target-farm');
    if (targetFarmId) return { handled: true, type: 'setTargetFarm', enemyId: targetFarmId };
    const worldMapNodeId = getAttribute('data-starfall-world-map-node');
    if (worldMapNodeId) return { handled: true, type: 'selectWorldMapNode', mapId: worldMapNodeId };
    const worldMapGuideId = getAttribute('data-starfall-world-map-guide');
    if (worldMapGuideId) return { handled: true, type: 'setWorldMapGuide', mapId: worldMapGuideId };
    return { handled: false, type: '' };
  }

  function getWorldProgressRegionAction(region) {
    const source = region || {};
    if (source.type === 'target-farm') return { handled: true, type: 'setTargetFarm', enemyId: source.enemyId };
    if (source.type === 'world-map-node') return { handled: true, type: 'selectWorldMapNode', mapId: source.mapId };
    if (source.type === 'world-map-guide') return { handled: true, type: 'setWorldMapGuide', mapId: source.mapId };
    return { handled: false, type: '' };
  }

  function getQuestPanelRegionAction(region) {
    const source = region || {};
    if (source.type === 'quest-guide') {
      return {
        handled: true,
        phase: 'prompt',
        type: 'setQuestGuideTarget',
        guideType: source.guideType,
        guideId: source.guideId
      };
    }
    if (
      source.type === 'quest-npc-accept-icon' ||
      source.type === 'quest-npc-reward-icon' ||
      source.type === 'quest-npc-talk-icon' ||
      source.type === 'quest-npc-active-icon'
    ) {
      return {
        handled: true,
        phase: 'prompt',
        type: 'openQuestNpcPrompt',
        npcId: source.npcId,
        questId: source.questId,
        action: source.action
      };
    }
    if (source.type === 'quest-prompt-close') {
      return { handled: true, phase: 'prompt', type: 'declineQuestPrompt' };
    }
    if (source.type === 'quest-prompt-accept' || source.type === 'quest-prompt-claim') {
      return { handled: true, phase: 'prompt', type: 'confirmQuestPrompt' };
    }
    if (source.type === 'quest-tab') {
      return { handled: true, phase: 'panel', type: 'setQuestTab', tabId: source.tabId };
    }
    if (source.type === 'claim-accomplishment') {
      return {
        handled: true,
        phase: 'panel',
        type: 'claimAccomplishment',
        accomplishmentId: source.accomplishmentId
      };
    }
    return { handled: false, phase: '', type: '' };
  }

  function getQuestAdventureRegionAction(region) {
    const source = region || {};
    if (source.type === 'start-trial') return { handled: true, type: 'startClassTrial', trialId: source.trialId };
    if (source.type === 'start-dungeon') return { handled: true, type: 'startDungeon', dungeonId: source.dungeonId };
    return { handled: false, type: '' };
  }

  function getQuestDerivedSnapshotUpdate(engine) {
    const source = engine || {};
    const update = {};
    if (typeof source.getProgressSnapshot === 'function') update.progress = source.getProgressSnapshot();
    if (typeof source.getRouteProgressSnapshot === 'function') update.routeProgress = source.getRouteProgressSnapshot();
    if (typeof source.getWorldMapSnapshot === 'function') update.worldMap = source.getWorldMapSnapshot();
    if (typeof source.getMapKillQuestSnapshot === 'function') update.mapKillQuest = source.getMapKillQuestSnapshot();
    if (typeof source.getQuestGuidanceSnapshot === 'function') update.questGuidance = source.getQuestGuidanceSnapshot();
    if (typeof source.getDungeonSnapshot === 'function') update.dungeon = source.getDungeonSnapshot();
    return update;
  }

  function getGuideDerivedSnapshotUpdate(engine) {
    const source = engine || {};
    const update = {};
    if (typeof source.getMapModifierSnapshot === 'function') update.mapModifiers = source.getMapModifierSnapshot();
    if (typeof source.getSkillModifierSnapshot === 'function') update.skillModifiers = source.getSkillModifierSnapshot();
    if (typeof source.getClassMasterySnapshot === 'function') update.classMastery = source.getClassMasterySnapshot();
    if (typeof source.getTargetFarmSnapshot === 'function') update.targetFarm = source.getTargetFarmSnapshot();
    if (typeof source.getRosterSynergySnapshot === 'function') update.rosterSynergies = source.getRosterSynergySnapshot();
    if (typeof source.getAdvancedGuideSnapshot === 'function') update.guide = source.getAdvancedGuideSnapshot();
    return update;
  }

  function createQuestUiHelpers(options) {
    const settings = options || {};
    function getOrderOptions(callOptions) {
      return Object.assign({}, settings, callOptions || {});
    }
    return Object.freeze({
      getAccomplishmentCategoryRank(accomplishment, callOptions) {
        return getAccomplishmentCategoryRank(accomplishment, getOrderOptions(callOptions));
      },
      getAccomplishmentTierRank(accomplishment, callOptions) {
        return getAccomplishmentTierRank(accomplishment, getOrderOptions(callOptions));
      },
      sortAccomplishmentItems(items, callOptions) {
        return sortAccomplishmentItems(items, getOrderOptions(callOptions));
      },
      getWorldMapDetailPresentation,
      getWorldProgressDomAction,
      getWorldProgressRegionAction,
      getQuestPanelRegionAction,
      getQuestAdventureRegionAction,
      getQuestDerivedSnapshotUpdate,
      getGuideDerivedSnapshotUpdate
    });
  }

  const api = {
    QUEST_TAB_OPTIONS,
    WORLD_MAP_DETAIL_LAYOUT,
    ACCOMPLISHMENT_CATEGORY_ORDER,
    ACCOMPLISHMENT_TIER_ORDER,
    createQuestUiHelpers,
    getAccomplishmentCategoryRank,
    getAccomplishmentTierRank,
    sortAccomplishmentItems,
    getWorldMapDetailPresentation,
    getWorldProgressDomAction,
    getWorldProgressRegionAction,
    getQuestPanelRegionAction,
    getQuestAdventureRegionAction,
    getQuestDerivedSnapshotUpdate,
    getGuideDerivedSnapshotUpdate
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.quests = Object.assign({}, modules.quests || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
