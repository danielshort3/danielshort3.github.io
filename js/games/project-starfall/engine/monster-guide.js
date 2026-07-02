(function initProjectStarfallEngineMonsterGuide(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    const source = Array.isArray(items) ? items : [];
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] && source[index].id === id) return source[index];
    }
    return null;
  };

  const MONSTER_GUIDE_STATE_NORMALIZED_KEY = '__projectStarfallMonsterGuideStateNormalized';
  const MONSTER_GUIDE_REGULAR_MILESTONES = Object.freeze([1, 5, 10, 25, 50, 100, 250]);
  const MONSTER_GUIDE_BOSS_MILESTONES = Object.freeze([1, 2, 3, 5, 10, 20, 50]);
  const MONSTER_GUIDE_DAMAGE_BONUSES = Object.freeze({
    insight: 0.02,
    mastery: 0.05
  });

  function markMonsterGuideStateNormalized(guide) {
    if (!guide || typeof guide !== 'object') return guide;
    try {
      Object.defineProperty(guide, MONSTER_GUIDE_STATE_NORMALIZED_KEY, {
        value: true,
        configurable: true,
        enumerable: false
      });
    } catch (error) {
      guide[MONSTER_GUIDE_STATE_NORMALIZED_KEY] = true;
    }
    return guide;
  }

  function isMonsterGuideStateNormalized(guide) {
    return !!(guide && typeof guide === 'object' && guide[MONSTER_GUIDE_STATE_NORMALIZED_KEY] === true);
  }

  function normalizeMonsterGuideBooleanMap(value) {
    const source = value && typeof value === 'object' ? value : {};
    return Object.entries(source).reduce((result, [key, active]) => {
      const id = normalizeId(key);
      if (id && active) result[id] = true;
      return result;
    }, {});
  }

  function normalizeMonsterGuideCountMap(value) {
    const source = value && typeof value === 'object' ? value : {};
    return Object.entries(source).reduce((result, [key, count]) => {
      const id = normalizeId(key);
      const amount = Math.max(0, Math.floor(Number(count) || 0));
      if (id && amount > 0) result[id] = amount;
      return result;
    }, {});
  }

  function getMonsterGuideMilestones(enemy, options) {
    const settings = options || {};
    const bossMilestones = Array.isArray(settings.bossMilestones) ? settings.bossMilestones : MONSTER_GUIDE_BOSS_MILESTONES;
    const regularMilestones = Array.isArray(settings.regularMilestones) ? settings.regularMilestones : MONSTER_GUIDE_REGULAR_MILESTONES;
    return enemy && enemy.behavior === 'boss' ? bossMilestones : regularMilestones;
  }

  function getMonsterGuideTier(enemy, kills, options) {
    const settings = options || {};
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const count = Math.max(0, Math.floor(Number(kills) || 0));
    return getMilestones(enemy).reduce((tier, milestone) => count >= milestone ? tier + 1 : tier, 0);
  }

  function createMonsterGuideMilestoneMap(enemy, kills, source, options) {
    const settings = options || {};
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const completed = normalizeMonsterGuideBooleanMap(source && source.completedMilestones);
    getMilestones(enemy).forEach((milestone) => {
      if (Math.max(0, Math.floor(Number(kills) || 0)) >= milestone) completed[String(milestone)] = true;
    });
    return completed;
  }

  function createMonsterGuideProgressEntry(enemy, source, legacyKills, options) {
    const settings = options || {};
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const getTier = typeof settings.getMonsterGuideTier === 'function'
      ? settings.getMonsterGuideTier
      : getMonsterGuideTier;
    const createMilestoneMap = typeof settings.createMonsterGuideMilestoneMap === 'function'
      ? settings.createMonsterGuideMilestoneMap
      : createMonsterGuideMilestoneMap;
    const entrySource = source && typeof source === 'object' ? source : {};
    const kills = Math.max(0, Math.floor(Number(entrySource.kills != null ? entrySource.kills : legacyKills) || 0));
    const firstSeen = Number(entrySource.firstSeen || 0);
    const firstKilled = Number(entrySource.firstKilled || 0);
    const lastKilled = Number(entrySource.lastKilled || 0);
    const dropCounts = normalizeMonsterGuideCountMap(entrySource.dropCounts || entrySource.dropsObtained);
    return {
      monsterId: enemy.id,
      sighted: !!entrySource.sighted || kills > 0 || firstSeen > 0,
      firstSeen,
      firstSeenMapId: normalizeId(entrySource.firstSeenMapId),
      firstKilled: firstKilled || (kills > 0 ? firstSeen : 0),
      lastKilled: lastKilled || firstKilled || 0,
      kills,
      researchPoints: Math.max(0, Math.floor(Number(entrySource.researchPoints) || kills)),
      researchTier: getTier(enemy, kills),
      dropsObtained: normalizeMonsterGuideBooleanMap(entrySource.dropsObtained),
      dropCounts,
      unlockedFields: normalizeMonsterGuideBooleanMap(entrySource.unlockedFields),
      completedMilestones: createMilestoneMap(enemy, kills, entrySource),
      claimedMilestoneRewards: normalizeMonsterGuideBooleanMap(entrySource.claimedMilestoneRewards),
      masteryComplete: !!entrySource.masteryComplete || kills >= Math.max(0, Number(getMilestones(enemy).slice(-1)[0] || 0)),
      dailyResearchBonusDate: String(entrySource.dailyResearchBonusDate || ''),
      dryStreaksByDropKey: normalizeMonsterGuideCountMap(entrySource.dryStreaksByDropKey),
      lastDropAt: Number(entrySource.lastDropAt || 0)
    };
  }

  function createMonsterGuideState(value, options) {
    if (isMonsterGuideStateNormalized(value)) return value;
    const settings = options || {};
    const data = settings.data || {};
    const enemies = Array.isArray(settings.enemies) ? settings.enemies : Array.isArray(data.ENEMIES) ? data.ENEMIES : [];
    const findById = typeof settings.getById === 'function' ? settings.getById : getById;
    const createProgressEntry = typeof settings.createMonsterGuideProgressEntry === 'function'
      ? settings.createMonsterGuideProgressEntry
      : createMonsterGuideProgressEntry;
    const source = value && typeof value === 'object' ? value : {};
    const sourceKills = source.killsByEnemyId && typeof source.killsByEnemyId === 'object'
      ? source.killsByEnemyId
      : source;
    const sourceEntries = source.entriesByMonsterId && typeof source.entriesByMonsterId === 'object'
      ? source.entriesByMonsterId
      : source.progressByMonsterId && typeof source.progressByMonsterId === 'object'
        ? source.progressByMonsterId
        : {};
    const killsByEnemyId = {};
    const entriesByMonsterId = {};
    enemies.forEach((enemy) => {
      if (!enemy || !enemy.id) return;
      const progress = createProgressEntry(enemy, sourceEntries[enemy.id], sourceKills[enemy.id]);
      entriesByMonsterId[enemy.id] = progress;
      killsByEnemyId[enemy.id] = progress.kills;
    });
    const selected = normalizeId(source.selectedEnemyId);
    const fallback = enemies && enemies[0] ? enemies[0].id : '';
    return markMonsterGuideStateNormalized({
      killsByEnemyId,
      entriesByMonsterId,
      selectedEnemyId: findById(enemies, selected) ? selected : fallback,
      pinnedEnemyIds: Array.isArray(source.pinnedEnemyIds)
        ? source.pinnedEnemyIds.map(normalizeId).filter((id) => findById(enemies, id))
        : [],
      regionRewardsClaimed: normalizeMonsterGuideBooleanMap(source.regionRewardsClaimed),
      familyRewardsClaimed: normalizeMonsterGuideBooleanMap(source.familyRewardsClaimed),
      globalRewardsClaimed: normalizeMonsterGuideBooleanMap(source.globalRewardsClaimed),
      accountWide: source.accountWide !== false
    });
  }

  function getMonsterGuideProgressEntry(enemyOrId, guideState, options) {
    const settings = options || {};
    const enemies = Array.isArray(settings.enemies) ? settings.enemies : [];
    const findById = typeof settings.getById === 'function' ? settings.getById : getById;
    const createProgressEntry = typeof settings.createMonsterGuideProgressEntry === 'function'
      ? settings.createMonsterGuideProgressEntry
      : createMonsterGuideProgressEntry;
    const getTier = typeof settings.getMonsterGuideTier === 'function'
      ? settings.getMonsterGuideTier
      : getMonsterGuideTier;
    const createMilestoneMap = typeof settings.createMonsterGuideMilestoneMap === 'function'
      ? settings.createMonsterGuideMilestoneMap
      : createMonsterGuideMilestoneMap;
    const id = normalizeId(enemyOrId && (enemyOrId.id || enemyOrId.monsterId) || enemyOrId);
    const enemy = findById(enemies, id);
    if (!enemy || !guideState) return null;
    const guide = guideState;
    guide.entriesByMonsterId = guide.entriesByMonsterId && typeof guide.entriesByMonsterId === 'object' ? guide.entriesByMonsterId : {};
    if (!guide.entriesByMonsterId[id]) {
      guide.entriesByMonsterId[id] = createProgressEntry(enemy, null, guide.killsByEnemyId && guide.killsByEnemyId[id]);
    }
    const progress = guide.entriesByMonsterId[id];
    const progressKills = Math.max(0, Math.floor(Number(progress.kills) || 0));
    const legacyKills = Math.max(0, Math.floor(Number(guide.killsByEnemyId && guide.killsByEnemyId[id]) || 0));
    const kills = Math.max(progressKills, legacyKills);
    progress.kills = kills;
    progress.researchTier = getTier(enemy, kills);
    progress.completedMilestones = createMilestoneMap(enemy, kills, progress);
    if (guide.killsByEnemyId && guide.killsByEnemyId[id] !== kills) guide.killsByEnemyId[id] = kills;
    return progress;
  }

  function recordMonsterGuideDefeat(enemy, guideState, options) {
    const settings = options || {};
    const enemyData = enemy && enemy.data ? enemy.data : enemy;
    if (!enemyData || !enemyData.id || !guideState) return false;
    const guide = guideState;
    const getProgressEntry = typeof settings.getMonsterGuideProgressEntry === 'function'
      ? settings.getMonsterGuideProgressEntry
      : getMonsterGuideProgressEntry;
    const getTier = typeof settings.getMonsterGuideTier === 'function'
      ? settings.getMonsterGuideTier
      : getMonsterGuideTier;
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const createMilestoneMap = typeof settings.createMonsterGuideMilestoneMap === 'function'
      ? settings.createMonsterGuideMilestoneMap
      : createMonsterGuideMilestoneMap;
    const progress = getProgressEntry(enemyData.id, guide);
    if (!progress) return false;
    const now = Number(settings.now == null ? Date.now() : settings.now);
    if (!progress.sighted) {
      progress.sighted = true;
      progress.firstSeen = now;
      progress.firstSeenMapId = normalizeId(settings.mapId);
    }
    progress.kills = Math.max(0, Math.floor(Number(progress.kills) || 0)) + 1;
    progress.researchPoints = Math.max(0, Math.floor(Number(progress.researchPoints) || 0)) + 1;
    progress.researchTier = getTier(enemyData, progress.kills);
    progress.firstKilled = progress.firstKilled || now;
    progress.lastKilled = now;
    progress.completedMilestones = createMilestoneMap(enemyData, progress.kills, progress);
    progress.masteryComplete = progress.masteryComplete || progress.kills >= Math.max(0, Number(getMilestones(enemyData).slice(-1)[0] || 0));
    guide.killsByEnemyId[enemyData.id] = progress.kills;
    if (!guide.selectedEnemyId) guide.selectedEnemyId = enemyData.id;
    return true;
  }

  function recordMonsterGuideSighted(enemy, guideState, options) {
    const settings = options || {};
    const enemyData = enemy && enemy.data ? enemy.data : enemy;
    if (!enemyData || !enemyData.id || enemyData.guide && enemyData.guide.visibility === 'debug' || !guideState) return false;
    const guide = guideState;
    const getProgressEntry = typeof settings.getMonsterGuideProgressEntry === 'function'
      ? settings.getMonsterGuideProgressEntry
      : getMonsterGuideProgressEntry;
    const progress = getProgressEntry(enemyData.id, guide);
    if (!progress || progress.sighted) return false;
    progress.sighted = true;
    progress.firstSeen = Number(settings.now == null ? Date.now() : settings.now);
    progress.firstSeenMapId = normalizeId(settings.mapId);
    if (!guide.selectedEnemyId) guide.selectedEnemyId = enemyData.id;
    return true;
  }

  function recordMonsterGuideDrop(enemy, item, guideState, options) {
    const settings = options || {};
    if (!enemy || !item || !guideState) return false;
    const getProgressEntry = typeof settings.getMonsterGuideProgressEntry === 'function'
      ? settings.getMonsterGuideProgressEntry
      : getMonsterGuideProgressEntry;
    const getDropKey = typeof settings.getMonsterGuideDropKey === 'function'
      ? settings.getMonsterGuideDropKey
      : getMonsterGuideDropKey;
    const getDropKind = typeof settings.getMonsterGuideDropKind === 'function'
      ? settings.getMonsterGuideDropKind
      : getMonsterGuideDropKind;
    const getNow = typeof settings.now === 'function'
      ? settings.now
      : function nowFallback() {
          return Date.now();
        };
    const progress = getProgressEntry(enemy.id, guideState);
    if (!progress) return false;
    const dropKey = normalizeId(settings.dropKey) || getDropKey(item);
    if (!dropKey || dropKey.endsWith(':')) return false;
    const kind = getDropKind(item);
    const quantity = kind === 'equipment' || kind === 'card'
      ? 1
      : Math.max(1, Math.floor(Number(item.quantity || item.amount || 1) || 1));
    progress.sighted = true;
    progress.firstSeen = progress.firstSeen || getNow();
    progress.firstSeenMapId = progress.firstSeenMapId || normalizeId(settings.mapId);
    progress.dropsObtained = progress.dropsObtained && typeof progress.dropsObtained === 'object' ? progress.dropsObtained : {};
    progress.dropCounts = progress.dropCounts && typeof progress.dropCounts === 'object' ? progress.dropCounts : {};
    progress.dropsObtained[dropKey] = true;
    progress.dropCounts[dropKey] = Math.max(0, Math.floor(Number(progress.dropCounts[dropKey]) || 0)) + quantity;
    progress.lastDropAt = getNow();
    return true;
  }

  function getMonsterGuideOddsUnlockState(enemy, kills, options) {
    const settings = options || {};
    const isBossMonster = typeof settings.isBossMonster === 'function'
      ? settings.isBossMonster
      : (candidate) => !!(candidate && candidate.behavior === 'boss');
    const boss = isBossMonster(enemy);
    const count = Math.max(0, Math.floor(Number(kills) || 0));
    return {
      sighted: count > 0,
      identified: count >= 1,
      approximateStats: boss ? count >= 2 : count >= 5,
      exactStats: boss ? count >= 3 : count >= 10,
      dropList: boss ? count >= 5 : count >= 25,
      exactOdds: boss ? count >= 10 : count >= 50,
      ultraRareOdds: boss ? count >= 20 : count >= 100,
      mastered: boss ? count >= 50 : count >= 250
    };
  }

  function getMonsterGuideDamageBonusForKills(enemy, kills, options) {
    const settings = options || {};
    const bonuses = settings.damageBonuses || MONSTER_GUIDE_DAMAGE_BONUSES;
    const getTier = typeof settings.getMonsterGuideTier === 'function'
      ? settings.getMonsterGuideTier
      : getMonsterGuideTier;
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const tier = getTier(enemy, kills);
    const milestones = getMilestones(enemy);
    if (Math.max(0, Math.floor(Number(kills) || 0)) >= Math.max(0, Number(milestones[milestones.length - 1] || 0))) return bonuses.mastery;
    if (tier >= 5) return bonuses.insight;
    return 0;
  }

  function getMonsterGuideStatRange(enemy, options) {
    const settings = options || {};
    const getMonsterHp = typeof settings.getMonsterHp === 'function'
      ? settings.getMonsterHp
      : function getMonsterHpFallback() {
          return 0;
        };
    const getMonsterDamage = typeof settings.getMonsterDamage === 'function'
      ? settings.getMonsterDamage
      : function getMonsterDamageFallback() {
          return 0;
        };
    const getMonsterDefense = typeof settings.getMonsterDefense === 'function'
      ? settings.getMonsterDefense
      : function getMonsterDefenseFallback() {
          return 0;
        };
    const levels = Array.isArray(enemy && enemy.levelRange) ? enemy.levelRange : [1, 1];
    const minLevel = Math.max(1, Number(levels[0]) || 1);
    const maxLevel = Math.max(minLevel, Number(levels[1]) || minLevel);
    const build = (level) => ({
      hp: getMonsterHp(level, enemy),
      damage: getMonsterDamage(level, enemy),
      defense: getMonsterDefense(level, enemy)
    });
    const low = build(minLevel);
    const high = build(maxLevel);
    return {
      level: { min: minLevel, max: maxLevel },
      hp: { min: low.hp, max: high.hp },
      damage: { min: low.damage, max: high.damage },
      defense: { min: low.defense, max: high.defense },
      speed: Math.max(0, Math.round(Number(enemy && enemy.speed || 0)))
    };
  }

  function getMonsterGuideAggroProfile(enemy) {
    const behavior = String(enemy && enemy.behavior || 'melee');
    if (behavior === 'turret') return { aggroRange: 620, attackRange: 620, note: 'Stationary thorn fire when the player enters range.' };
    if (behavior === 'thrower') return { aggroRange: 620, attackRange: 620, note: 'Kites near 280 px and throws arcing knives.' };
    if (behavior === 'flyer') return { aggroRange: 620, attackRange: 620, note: 'Floats near 340 px and casts firebolts.' };
    if (behavior === 'charger') return { aggroRange: 520, attackRange: 56, note: 'Telegraphs a charge before contact damage.' };
    if (behavior === 'healer') return { aggroRange: 520, attackRange: 56, note: 'Melee contact plus heals nearby enemies within 220 px.' };
    if (behavior === 'boss') return { aggroRange: 720, attackRange: 86, note: 'Boss arena tracking with telegraphed slams, charges, or special phases.' };
    if (behavior === 'elite') return { aggroRange: 560, attackRange: 56, note: 'Fast elite pressure with ambush movement and high reward drops.' };
    return { aggroRange: 520, attackRange: 56, note: 'Chases the player and attacks on the same combat lane.' };
  }

  function getMonsterGuideMaps(enemyId, options) {
    const settings = options || {};
    const maps = Array.isArray(settings.maps) ? settings.maps : [];
    return maps
      .filter((map) => (map.enemies || []).includes(enemyId))
      .map((map) => ({
        id: map.id,
        name: map.name,
        levelRange: Array.isArray(map.levelRange) ? map.levelRange.slice() : [],
        dungeon: !!map.isDungeon
      }));
  }

  function getMonsterGuideStaticEntryCacheKey(enemies, maps, options) {
    const settings = options || {};
    const state = settings.state || {};
    const player = state.player || {};
    const getPreferredSkillManualDropId = typeof settings.getPreferredSkillManualDropId === 'function'
      ? settings.getPreferredSkillManualDropId
      : function getPreferredSkillManualDropIdFallback() {
          return '';
        };
    const firstEnemy = enemies[0] || {};
    const lastEnemy = enemies[enemies.length - 1] || {};
    const firstMap = maps[0] || {};
    const lastMap = maps[maps.length - 1] || {};
    return [
      enemies.length,
      normalizeId(firstEnemy.id),
      normalizeId(lastEnemy.id),
      maps.length,
      normalizeId(firstMap.id),
      normalizeId(lastMap.id),
      normalizeId(player.classId),
      normalizeId(player.advancedClassId),
      getPreferredSkillManualDropId(state)
    ].join('|');
  }

  function createMonsterGuideStaticEntry(enemy, options) {
    if (!enemy) return null;
    const settings = options || {};
    const getMilestones = typeof settings.getMonsterGuideMilestones === 'function'
      ? settings.getMonsterGuideMilestones
      : getMonsterGuideMilestones;
    const getStatRange = typeof settings.getMonsterGuideStatRange === 'function'
      ? settings.getMonsterGuideStatRange
      : getMonsterGuideStatRange;
    const getAggroProfile = typeof settings.getMonsterGuideAggroProfile === 'function'
      ? settings.getMonsterGuideAggroProfile
      : getMonsterGuideAggroProfile;
    const getMaps = typeof settings.getMonsterGuideMaps === 'function'
      ? settings.getMonsterGuideMaps
      : getMonsterGuideMaps;
    const getDropInfo = typeof settings.getMonsterGuideDropInfo === 'function'
      ? settings.getMonsterGuideDropInfo
      : function getMonsterGuideDropInfoFallback() {
          return { tables: [], rows: [] };
        };
    const milestones = getMilestones(enemy).slice();
    const guide = enemy.guide || {};
    return {
      id: enemy.id,
      name: enemy.name,
      asset: enemy.asset || '',
      family: enemy.family,
      role: enemy.role,
      category: guide.category || (enemy.behavior === 'boss' ? 'boss' : enemy.behavior === 'elite' ? 'elite' : 'normal'),
      threatTier: guide.threatTier || (enemy.behavior === 'boss' ? 'Boss' : 'Field'),
      collectionExcluded: !!guide.excludedFromCollection,
      guideVisibility: guide.visibility || 'live',
      regionTags: Array.isArray(guide.regionTags) ? guide.regionTags.slice() : [],
      biomeTags: Array.isArray(guide.biomeTags) ? guide.biomeTags.slice() : [],
      behaviorTags: Array.isArray(guide.behaviorTags) ? guide.behaviorTags.slice() : [],
      weaknesses: Array.isArray(guide.weaknesses) ? guide.weaknesses.slice() : [],
      resistances: Array.isArray(guide.resistances) ? guide.resistances.slice() : [],
      statusVulnerabilities: Array.isArray(guide.statusVulnerabilities) ? guide.statusVulnerabilities.slice() : [],
      attackPatterns: Array.isArray(guide.attackPatterns) ? guide.attackPatterns.slice() : [],
      lore: guide.lore || '',
      spawnConditions: Array.isArray(guide.spawnConditions) ? guide.spawnConditions.slice() : [],
      respawnClass: guide.respawnClass || '',
      questTags: Array.isArray(guide.questTags) ? guide.questTags.slice() : [],
      behavior: enemy.behavior,
      mechanic: enemy.mechanic,
      counter: enemy.counter,
      drops: Array.isArray(enemy.drops) ? enemy.drops.slice() : [],
      maxTier: milestones.length,
      milestones,
      stats: getStatRange(enemy),
      aggro: getAggroProfile(enemy),
      maps: getMaps(enemy.id),
      dropInfo: getDropInfo(enemy)
    };
  }

  function getMonsterGuideBossSetDropRows(enemy, options) {
    const settings = options || {};
    const bossEquipmentItems = Array.isArray(settings.bossEquipmentItems) ? settings.bossEquipmentItems : [];
    const getBossEquipmentSource = typeof settings.getBossEquipmentSource === 'function'
      ? settings.getBossEquipmentSource
      : function getBossEquipmentSourceFallback() {
          return null;
        };
    const normalizeDropChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : function normalizeDropChanceFallback(value, fallback) {
          const chance = Number(value == null ? fallback : value);
          return Number.isFinite(chance) ? Math.max(0, Math.min(1, chance)) : Math.max(0, Math.min(1, Number(fallback || 0)));
        };
    const clamp = typeof settings.clamp === 'function'
      ? settings.clamp
      : function clampFallback(value, min, max) {
          return Math.max(min, Math.min(max, value));
        };
    const getRarityTier = typeof settings.getMonsterGuideRarityTier === 'function'
      ? settings.getMonsterGuideRarityTier
      : getMonsterGuideRarityTier;
    const enemyData = enemy && enemy.data ? enemy.data : enemy;
    const source = getBossEquipmentSource(enemyData && enemyData.id);
    if (!source) return [];
    return bossEquipmentItems
      .filter((item) => item && item.setId === source.setId)
      .map((item) => {
        const pieceChance = source.dropChance && bossEquipmentItems
          ? normalizeDropChance(source.dropChance, 0) / Math.max(1, bossEquipmentItems.filter((candidate) => candidate && candidate.setId === source.setId).length)
          : 0;
        return {
          id: `bossSet:${item.id}`,
          dropKey: `equipment:${item.id}`,
          kind: 'equipment',
          type: 'equipment',
          materialId: '',
          itemId: item.id,
          consumableId: '',
          cardId: '',
          label: item.name,
          categoryLabel: 'Boss Set',
          tableId: 'bossSet',
          tableLabel: `${source.name || enemyData.name || 'Boss'} set`,
          quantityMin: 1,
          quantityMax: 1,
          rarity: item.rarity || source.rarity || 'Epic',
          rarityTier: getRarityTier(item.rarity || source.rarity || 'Epic'),
          tableChance: normalizeDropChance(source.dropChance, 0),
          entryChance: pieceChance > 0 && source.dropChance > 0 ? clamp(pieceChance / source.dropChance, 0, 1) : 0,
          chancePerKill: pieceChance,
          conditions: 'Boss set roll; missing pieces are weighted higher than duplicates.',
          hiddenUntilFound: false,
          bossSetId: source.setId
        };
      });
  }

  function createMonsterGuideDropInfo(enemy, state, options) {
    const settings = options || {};
    const getDropTables = typeof settings.getMonsterDropTables === 'function'
      ? settings.getMonsterDropTables
      : function getMonsterDropTablesFallback() {
          return [];
        };
    const getDropLabel = typeof settings.getMonsterDropLabel === 'function'
      ? settings.getMonsterDropLabel
      : (entry) => getMonsterDropLabel(entry, settings);
    const getQuantityRange = typeof settings.getMonsterGuideQuantityRange === 'function'
      ? settings.getMonsterGuideQuantityRange
      : getMonsterGuideQuantityRange;
    const getDropRarity = typeof settings.getMonsterGuideDropRarity === 'function'
      ? settings.getMonsterGuideDropRarity
      : (entry) => getMonsterGuideDropRarity(entry, settings);
    const getRarityTier = typeof settings.getMonsterGuideRarityTier === 'function'
      ? settings.getMonsterGuideRarityTier
      : getMonsterGuideRarityTier;
    const getDropKey = typeof settings.getMonsterGuideDropKey === 'function'
      ? settings.getMonsterGuideDropKey
      : (entry) => getMonsterGuideDropKey(entry, settings);
    const normalizeDropWeight = typeof settings.normalizeDropWeight === 'function'
      ? settings.normalizeDropWeight
      : function normalizeDropWeightFallback(value, fallback) {
          return Math.max(1, Math.round(Number(value == null ? fallback : value) || fallback || 1));
        };
    const normalizeDropChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : function normalizeDropChanceFallback(value, fallback) {
          const chance = Number(value == null ? fallback : value);
          return Number.isFinite(chance) ? Math.max(0, Math.min(1, chance)) : Math.max(0, Math.min(1, Number(fallback || 0)));
        };
    const createDropRow = typeof settings.createMonsterGuideDropRowFromEntry === 'function'
      ? settings.createMonsterGuideDropRowFromEntry
      : (table, entry, index) => createMonsterGuideDropRowFromEntry(table, entry, index, settings);
    const getBossSetDropRows = typeof settings.getMonsterGuideBossSetDropRows === 'function'
      ? settings.getMonsterGuideBossSetDropRows
      : (target) => getMonsterGuideBossSetDropRows(target, settings);
    const tables = getDropTables(enemy, state).map((table) => {
      const entries = (table.entries || []).map((item) => {
        const range = getQuantityRange(item);
        const rarity = getDropRarity(item);
        return {
          type: item.type,
          materialId: item.materialId || '',
          itemId: item.itemId || '',
          consumableId: item.consumableId || '',
          cardId: item.cardId || '',
          label: getDropLabel(item),
          quantityMin: range.min,
          quantityMax: range.max,
          rarity,
          rarityTier: getRarityTier(rarity),
          dropKey: getDropKey(item),
          weight: normalizeDropWeight(item.weight, 1),
          chance: normalizeDropChance(item.chance, 0),
          chancePerKill: normalizeDropChance(item.chancePerKill, 0),
          sourcePool: table.id,
          tableId: table.id,
          tableLabel: table.label
        };
      });
      return {
        id: table.id,
        label: table.label,
        chance: normalizeDropChance(table.chance, 0),
        guaranteed: !!table.guaranteed,
        entries
      };
    });
    const rows = tables.reduce((allRows, table) => {
      (table.entries || []).forEach((entry, index) => {
        allRows.push(createDropRow(table, entry, allRows.length + index));
      });
      return allRows;
    }, []).concat(getBossSetDropRows(enemy));
    const expectedDropsPerKill = tables.reduce((sum, table) => sum + normalizeDropChance(table.chance, 0), 0);
    const globalRareExpectedDropsPerKill = tables
      .filter((table) => table.id === 'rareValuables')
      .reduce((sum, table) => sum + normalizeDropChance(table.chance, 0), 0);
    const successfulDropEntries = tables.reduce((entries, table) => entries.concat(table.entries || []), [])
      .sort((a, b) => Number(b.chancePerKill || 0) - Number(a.chancePerKill || 0) || a.label.localeCompare(b.label));
    return {
      expectedDropsPerKill,
      globalRareExpectedDropsPerKill,
      tables,
      rows,
      successfulDropEntries
    };
  }

  function applyMonsterGuideDropProgress(dropInfo, progress, staticEntry, unlockState, options) {
    const settings = options || {};
    const getDropKey = typeof settings.getMonsterGuideDropKey === 'function'
      ? settings.getMonsterGuideDropKey
      : getMonsterGuideDropKey;
    const getRarityTier = typeof settings.getMonsterGuideRarityTier === 'function'
      ? settings.getMonsterGuideRarityTier
      : getMonsterGuideRarityTier;
    const info = dropInfo || { tables: [], rows: [] };
    const state = unlockState || {};
    const dropCounts = progress && progress.dropCounts && typeof progress.dropCounts === 'object' ? progress.dropCounts : {};
    const rows = (info.rows || []).map((row) => {
      const dropKey = row.dropKey || getDropKey(row);
      const obtainedCount = Math.max(0, Math.floor(Number(dropCounts[dropKey]) || 0));
      const rarityTier = row.rarityTier || getRarityTier(row.rarity);
      const ultraRare = rarityTier === 'Ultra Rare';
      const oddsUnlocked = ultraRare ? !!state.ultraRareOdds : !!state.exactOdds;
      const listed = !!state.dropList || obtainedCount > 0;
      const knownState = oddsUnlocked && listed
        ? 'fully-researched'
        : obtainedCount > 0
          ? 'obtained'
          : listed
            ? 'revealed'
            : 'hidden';
      const displayLabel = knownState === 'hidden'
        ? `Unknown ${row.categoryLabel || 'drop'}`
        : row.label;
      return Object.assign({}, row, {
        dropKey,
        obtainedCount,
        knownState,
        displayLabel,
        oddsUnlocked,
        oddsHiddenReason: oddsUnlocked ? '' : ultraRare ? 'Unlocks at ultra-rare odds research.' : 'Unlocks at exact-odds research.',
        visible: listed || !!state.sighted,
        silhouette: knownState === 'hidden' || knownState === 'revealed' && obtainedCount <= 0
      });
    });
    const visibleRows = rows.filter((row) => row.visible);
    const obtainedRows = rows.filter((row) => row.obtainedCount > 0).length;
    const researchedRows = rows.filter((row) => row.knownState === 'fully-researched').length;
    const hiddenRows = rows.filter((row) => row.knownState === 'hidden').length;
    return Object.assign({}, info, {
      rows,
      visibleRows,
      totalRows: rows.length,
      discoveredRows: rows.length - hiddenRows,
      obtainedRows,
      researchedRows,
      hiddenRows,
      missingRows: Math.max(0, rows.length - obtainedRows),
      collectionComplete: rows.length === 0 || obtainedRows >= rows.length
    });
  }

  function createMonsterGuideEntrySnapshot(staticEntry, guideState, options) {
    if (!staticEntry) return null;
    const settings = options || {};
    const guide = guideState || {};
    const clamp = typeof settings.clamp === 'function'
      ? settings.clamp
      : function clampFallback(value, min, max) {
          return Math.max(min, Math.min(max, value));
        };
    const getProgressEntry = typeof settings.getMonsterGuideProgressEntry === 'function'
      ? settings.getMonsterGuideProgressEntry
      : function getMonsterGuideProgressEntryFallback() {
          return null;
        };
    const getOddsUnlockState = typeof settings.getMonsterGuideOddsUnlockState === 'function'
      ? settings.getMonsterGuideOddsUnlockState
      : getMonsterGuideOddsUnlockState;
    const applyDropProgress = typeof settings.applyMonsterGuideDropProgress === 'function'
      ? settings.applyMonsterGuideDropProgress
      : applyMonsterGuideDropProgress;
    const getDamageBonus = typeof settings.getMonsterGuideDamageBonusForKills === 'function'
      ? settings.getMonsterGuideDamageBonusForKills
      : getMonsterGuideDamageBonusForKills;
    const progress = getProgressEntry(staticEntry.id, guide);
    const kills = Math.max(0, Math.floor(Number(progress && progress.kills != null ? progress.kills : guide.killsByEnemyId && guide.killsByEnemyId[staticEntry.id]) || 0));
    const milestones = staticEntry.milestones || [];
    const tier = milestones.reduce((value, milestone) => kills >= milestone ? value + 1 : value, 0);
    const nextMilestone = milestones.find((milestone) => kills < milestone) || 0;
    const previousMilestone = milestones[Math.max(0, tier - 1)] || 0;
    const finalMilestone = Math.max(1, Number(milestones[milestones.length - 1] || 1));
    const nextProgressSpan = Math.max(1, (nextMilestone || finalMilestone) - previousMilestone);
    const sighted = !!(progress && progress.sighted) || kills > 0;
    const identified = kills >= 1;
    const unlockState = Object.assign(getOddsUnlockState(staticEntry, kills), {
      sighted,
      identified,
      mastered: tier >= milestones.length
    });
    const dropInfo = applyDropProgress(staticEntry.dropInfo, progress, staticEntry, unlockState);
    const researchProgress = milestones.length ? clamp(tier / milestones.length, 0, 1) : 1;
    const dropProgress = dropInfo.totalRows ? clamp(dropInfo.obtainedRows / dropInfo.totalRows, 0, 1) : 1;
    const sightProgress = sighted ? 1 : 0;
    const completionPercent = staticEntry.collectionExcluded
      ? 0
      : Math.round((sightProgress * 0.1 + researchProgress * 0.55 + dropProgress * 0.35) * 100);
    const rankLabels = ['Unseen', 'Sighted', 'Known', 'Tactical', 'Drop Catalog', 'Exact Odds', 'Deep Research', 'Mastery'];
    const researchRank = unlockState.mastered ? 'Mastery' : rankLabels[Math.min(rankLabels.length - 1, Math.max(0, tier))];
    const rewardTrack = [
      { milestone: 1, label: 'Name, portrait, kill counter', earned: kills >= 1 },
      { milestone: staticEntry.category === 'boss' ? 2 : 5, label: 'Early combat tactics', earned: !!unlockState.approximateStats },
      { milestone: staticEntry.category === 'boss' ? 5 : 25, label: 'Drop catalog silhouettes', earned: !!unlockState.dropList },
      { milestone: staticEntry.category === 'boss' ? 10 : 50, label: 'Exact common-to-rare odds', earned: !!unlockState.exactOdds },
      { milestone: staticEntry.category === 'boss' ? 20 : 100, label: 'Ultra-rare odds and advanced notes', earned: !!unlockState.ultraRareOdds },
      { milestone: milestones[milestones.length - 1] || 0, label: 'Mastery badge and prestige tracking', earned: !!unlockState.mastered }
    ];
    return Object.assign({}, staticEntry, {
      progress: Object.assign({}, progress, {
        dropCounts: Object.assign({}, progress && progress.dropCounts || {}),
        dropsObtained: Object.assign({}, progress && progress.dropsObtained || {})
      }),
      kills,
      tier,
      researchRank,
      sighted,
      identified,
      unlockState,
      nextMilestone,
      nextProgress: nextMilestone ? clamp((kills - previousMilestone) / nextProgressSpan, 0, 1) : 1,
      masteryProgress: clamp(kills / finalMilestone, 0, 1),
      completionPercent,
      entryCompletion: completionPercent,
      dropsDiscovered: dropInfo.discoveredRows,
      dropsObtained: dropInfo.obtainedRows,
      dropsTotal: dropInfo.totalRows,
      dropInfo,
      rewardTrack,
      rewardsEarned: [
        unlockState.dropList ? 'Drop catalog' : '',
        unlockState.exactOdds ? 'Exact odds' : '',
        unlockState.mastered ? 'Mastery badge' : ''
      ].filter(Boolean),
      masteryComplete: !!(progress && progress.masteryComplete) || unlockState.mastered,
      damageBonus: getDamageBonus(staticEntry, kills)
    });
  }

  function getMonsterGuideSnapshotKillValues(staticCache, guide) {
    const entries = staticCache && Array.isArray(staticCache.entries) ? staticCache.entries : [];
    const killsByEnemyId = guide && guide.killsByEnemyId && typeof guide.killsByEnemyId === 'object'
      ? guide.killsByEnemyId
      : {};
    const progressById = guide && guide.entriesByMonsterId && typeof guide.entriesByMonsterId === 'object'
      ? guide.entriesByMonsterId
      : {};
    const values = new Array(entries.length);
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const progress = progressById[entry && entry.id] || {};
      const dropCounts = progress.dropCounts && typeof progress.dropCounts === 'object' ? progress.dropCounts : {};
      const dropSignature = Object.keys(dropCounts).sort().map((key) => `${key}:${Math.max(0, Math.floor(Number(dropCounts[key]) || 0))}`).join(',');
      const progressKills = Math.max(0, Math.floor(Number(progress.kills) || 0));
      const legacyKills = Math.max(0, Math.floor(Number(killsByEnemyId[entry && entry.id]) || 0));
      values[index] = [
        Math.max(progressKills, legacyKills),
        progress.sighted ? 1 : 0,
        normalizeId(progress.firstSeenMapId),
        dropSignature
      ].join('|');
    }
    return values;
  }

  function isMonsterGuideSnapshotCacheCurrent(staticCache, guide, selectedId, snapshotCache, options) {
    const settings = options || {};
    const getKillValues = typeof settings.getMonsterGuideSnapshotKillValues === 'function'
      ? settings.getMonsterGuideSnapshotKillValues
      : getMonsterGuideSnapshotKillValues;
    if (!snapshotCache ||
      snapshotCache.staticCache !== staticCache ||
      snapshotCache.guide !== guide ||
      snapshotCache.killsByEnemyId !== (guide && guide.killsByEnemyId) ||
      snapshotCache.entriesByMonsterId !== (guide && guide.entriesByMonsterId) ||
      snapshotCache.selectedId !== selectedId) {
      return false;
    }
    const entries = staticCache && Array.isArray(staticCache.entries) ? staticCache.entries : [];
    const previousValues = Array.isArray(snapshotCache.killValues) ? snapshotCache.killValues : [];
    if (previousValues.length !== entries.length) return false;
    const values = getKillValues(staticCache, guide);
    for (let index = 0; index < values.length; index += 1) {
      if (previousValues[index] !== values[index]) return false;
    }
    return true;
  }

  function createMonsterGuideGroupSummaries(entries, getLabels) {
    const source = Array.isArray(entries) ? entries : [];
    const groups = new Map();
    source.forEach((entry) => {
      const labels = typeof getLabels === 'function' ? getLabels(entry) : [];
      (labels && labels.length ? labels : ['Uncategorized']).forEach((label) => {
        const key = normalizeId(label || 'Uncategorized');
        if (!groups.has(key)) groups.set(key, { id: key, name: label || 'Uncategorized', entries: [] });
        groups.get(key).entries.push(entry);
      });
    });
    return Array.from(groups.values()).map((group) => {
      const groupCompletion = group.entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.completionPercent || 0)), 0);
      return {
        id: group.id,
        name: group.name,
        total: group.entries.length,
        sighted: group.entries.filter((entry) => entry.sighted).length,
        mastered: group.entries.filter((entry) => entry.masteryComplete).length,
        dropsObtained: group.entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.dropsObtained || 0)), 0),
        dropsTotal: group.entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.dropsTotal || 0)), 0),
        completionPercent: group.entries.length ? Math.round(groupCompletion / group.entries.length) : 0
      };
    }).sort((a, b) => b.completionPercent - a.completionPercent || a.name.localeCompare(b.name));
  }

  function createMonsterGuideSnapshotSummary(entries) {
    const source = Array.isArray(entries) ? entries : [];
    const collectibleEntries = source.filter((entry) => !entry.collectionExcluded);
    const completionTotal = collectibleEntries.reduce((sum, entry) => sum + Math.max(0, Number(entry.completionPercent || 0)), 0);
    const regions = createMonsterGuideGroupSummaries(collectibleEntries, (entry) => entry.regionTags || []);
    const families = createMonsterGuideGroupSummaries(collectibleEntries, (entry) => [entry.family || entry.category || 'Monster']);
    return {
      totalEntries: source.length,
      totalCollectible: collectibleEntries.length,
      excludedEntries: source.length - collectibleEntries.length,
      sighted: collectibleEntries.filter((entry) => entry.sighted).length,
      identified: collectibleEntries.filter((entry) => entry.identified).length,
      mastered: collectibleEntries.filter((entry) => entry.masteryComplete).length,
      bosses: collectibleEntries.filter((entry) => entry.category === 'boss').length,
      bossMastered: collectibleEntries.filter((entry) => entry.category === 'boss' && entry.masteryComplete).length,
      dropsKnown: collectibleEntries.reduce((sum, entry) => sum + Math.max(0, Number(entry.dropsDiscovered || 0)), 0),
      dropsObtained: collectibleEntries.reduce((sum, entry) => sum + Math.max(0, Number(entry.dropsObtained || 0)), 0),
      dropsTotal: collectibleEntries.reduce((sum, entry) => sum + Math.max(0, Number(entry.dropsTotal || 0)), 0),
      regions,
      families,
      completionPercent: collectibleEntries.length ? Math.round(completionTotal / collectibleEntries.length) : 0
    };
  }

  function createMonsterGuideSnapshotValue(entries, selectedEnemyId, options) {
    const settings = options || {};
    const source = Array.isArray(entries) ? entries : [];
    const regularMilestones = Array.isArray(settings.regularMilestones) ? settings.regularMilestones : MONSTER_GUIDE_REGULAR_MILESTONES;
    const bossMilestones = Array.isArray(settings.bossMilestones) ? settings.bossMilestones : MONSTER_GUIDE_BOSS_MILESTONES;
    const summary = createMonsterGuideSnapshotSummary(source);
    return {
      selectedEnemyId,
      entries: source,
      summary,
      completion: summary,
      milestones: {
        regular: regularMilestones.slice(),
        boss: bossMilestones.slice()
      }
    };
  }

  function createMonsterGuideSnapshotCacheState(staticCache, guide, selectedId, killValues, value) {
    const sourceGuide = guide && typeof guide === 'object' ? guide : {};
    return {
      staticCache,
      guide,
      killsByEnemyId: sourceGuide.killsByEnemyId,
      entriesByMonsterId: sourceGuide.entriesByMonsterId,
      selectedId,
      killValues: Array.isArray(killValues) ? killValues : [],
      value
    };
  }

  function getMonsterDropLabel(drop, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const monsterGuideDropLabels = settings.monsterGuideDropLabels || {};
    const getMaterialSortMeta = typeof settings.getMaterialSortMeta === 'function'
      ? settings.getMaterialSortMeta
      : function getMaterialSortMetaFallback(materialId) {
          return { name: normalizeId(materialId) };
        };
    const getEquipmentDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : function getEquipmentDefinitionFallback() {
          return null;
        };
    const getConsumableDefinitionById = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : function getConsumableDefinitionByIdFallback() {
          return null;
        };
    const getCardDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : function getCardDefinitionFallback() {
          return null;
        };
    const entry = drop && typeof drop === 'object' ? drop : { type: drop };
    if (entry.type === 'material' && entry.materialId) {
      const material = materialDropDefinitions[entry.materialId];
      return material && material.name || getMaterialSortMeta(entry.materialId).name;
    }
    if (entry.type === 'equipment' && entry.itemId) {
      const item = getEquipmentDefinition(entry.itemId);
      return item && item.name || entry.itemId;
    }
    if (entry.type === 'consumable' && entry.consumableId) {
      const item = getConsumableDefinitionById(entry.consumableId);
      return item && item.name || entry.consumableId;
    }
    if (entry.type === 'card' && entry.cardId) {
      const card = getCardDefinition(entry.cardId);
      return card && card.name || entry.cardId;
    }
    return monsterGuideDropLabels[entry.type] || String(entry.type || 'Loot');
  }

  function getMonsterGuideDropKind(entry) {
    const raw = normalizeId(entry && (entry.kind || entry.type));
    if (raw === 'material' || raw === 'consumable' || raw === 'currency' || raw === 'card' || raw === 'equipment') return raw;
    if (entry && (entry.materialId || entry.type === 'material')) return 'material';
    if (entry && (entry.consumableId || entry.type === 'consumable')) return 'consumable';
    if (entry && (entry.cardId || entry.type === 'card')) return 'card';
    if (entry && (entry.itemId || entry.slot || entry.type === 'equipment')) return 'equipment';
    if (entry && (entry.id === 'coins' || entry.type === 'currency')) return 'currency';
    return raw || 'equipment';
  }

  function getMonsterGuideDropKey(entry, options) {
    const settings = options || {};
    const getDropKind = typeof settings.getMonsterGuideDropKind === 'function'
      ? settings.getMonsterGuideDropKind
      : getMonsterGuideDropKind;
    const kind = getDropKind(entry);
    if (kind === 'material') return `material:${normalizeId(entry && (entry.materialId || entry.id))}`;
    if (kind === 'consumable') return `consumable:${normalizeId(entry && (entry.consumableId || entry.id))}`;
    if (kind === 'card') return `card:${normalizeId(entry && (entry.cardId || entry.id))}`;
    if (kind === 'currency') return 'currency:coins';
    return `equipment:${normalizeId(entry && (entry.itemId || entry.id))}`;
  }

  function getMonsterGuideRarityTier(rarity) {
    if (rarity === 'Relic') return 'Ultra Rare';
    if (rarity === 'Epic') return 'Very Rare';
    if (rarity === 'Rare') return 'Rare';
    if (rarity === 'Uncommon') return 'Uncommon';
    return 'Common';
  }

  function getMonsterGuideDropRarity(entry, options) {
    const settings = options || {};
    const materialDropDefinitions = settings.materialDropDefinitions || {};
    const getEquipmentDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : function getEquipmentDefinitionFallback() {
          return null;
        };
    const getConsumableDefinitionById = typeof settings.getConsumableDefinitionById === 'function'
      ? settings.getConsumableDefinitionById
      : function getConsumableDefinitionByIdFallback() {
          return null;
        };
    const getCardDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : function getCardDefinitionFallback() {
          return null;
        };
    if (entry && entry.rarity) return entry.rarity;
    if (entry && entry.type === 'equipment' && entry.itemId) {
      const item = getEquipmentDefinition(entry.itemId);
      return item && item.rarity || 'Common';
    }
    if (entry && entry.type === 'consumable' && entry.consumableId) {
      const item = getConsumableDefinitionById(entry.consumableId);
      return item && item.rarity || 'Common';
    }
    if (entry && entry.type === 'card' && entry.cardId) {
      const card = getCardDefinition(entry.cardId);
      return card && card.rarity || 'Common';
    }
    if (entry && entry.type === 'material' && entry.materialId) {
      return materialDropDefinitions[entry.materialId] && materialDropDefinitions[entry.materialId].rarity || 'Common';
    }
    return 'Common';
  }

  function getMonsterGuideQuantityRange(entry) {
    const min = Math.max(1, Math.floor(Number(entry && (entry.minQuantity || entry.quantityMin || entry.quantity || entry.amount) || 1)));
    const maxSource = entry && (entry.maxQuantity || entry.quantityMax || entry.quantity || entry.amount);
    const max = Math.max(min, Math.floor(Number(maxSource || min) || min));
    return { min, max };
  }

  function createMonsterGuideDropRowFromEntry(table, entry, index, options) {
    const settings = options || {};
    const getDropRarity = typeof settings.getMonsterGuideDropRarity === 'function'
      ? settings.getMonsterGuideDropRarity
      : (dropEntry) => getMonsterGuideDropRarity(dropEntry, settings);
    const getQuantityRange = typeof settings.getMonsterGuideQuantityRange === 'function'
      ? settings.getMonsterGuideQuantityRange
      : getMonsterGuideQuantityRange;
    const normalizeDropChance = typeof settings.normalizeDropChance === 'function'
      ? settings.normalizeDropChance
      : function normalizeDropChanceFallback(value, fallback) {
          const chance = Number(value == null ? fallback : value);
          return Number.isFinite(chance) ? Math.max(0, Math.min(1, chance)) : Math.max(0, Math.min(1, Number(fallback || 0)));
        };
    const getDropKind = typeof settings.getMonsterGuideDropKind === 'function'
      ? settings.getMonsterGuideDropKind
      : getMonsterGuideDropKind;
    const getDropKey = typeof settings.getMonsterGuideDropKey === 'function'
      ? settings.getMonsterGuideDropKey
      : (dropEntry) => getMonsterGuideDropKey(dropEntry, settings);
    const getDropLabel = typeof settings.getMonsterDropLabel === 'function'
      ? settings.getMonsterDropLabel
      : (dropEntry) => getMonsterDropLabel(dropEntry, settings);
    const getRarityTier = typeof settings.getMonsterGuideRarityTier === 'function'
      ? settings.getMonsterGuideRarityTier
      : getMonsterGuideRarityTier;
    const rarity = getDropRarity(entry);
    const range = getQuantityRange(entry);
    const tableChance = normalizeDropChance(table && table.chance, 0);
    const entryChance = normalizeDropChance(entry && entry.chance, 0);
    const chancePerKill = normalizeDropChance(entry && entry.chancePerKill, tableChance * entryChance);
    const kind = getDropKind(entry);
    const row = {
      id: `${table && table.id || 'drop'}:${getDropKey(entry)}:${index}`,
      dropKey: getDropKey(entry),
      kind,
      type: entry && entry.type || kind,
      materialId: entry && entry.materialId || '',
      itemId: kind === 'equipment' ? entry && (entry.itemId || entry.id) || '' : '',
      consumableId: entry && entry.consumableId || '',
      cardId: entry && entry.cardId || '',
      label: getDropLabel(entry),
      categoryLabel: table && table.label || 'Loot',
      tableId: table && table.id || '',
      tableLabel: table && table.label || '',
      quantityMin: range.min,
      quantityMax: range.max,
      rarity,
      rarityTier: getRarityTier(rarity),
      tableChance,
      entryChance,
      chancePerKill,
      conditions: table && table.guaranteed ? 'Guaranteed on defeat.' : 'Independent table roll.',
      hiddenUntilFound: !!(entry && entry.isHiddenUntilFound),
      bossSetId: ''
    };
    return row;
  }

  const api = {
    MONSTER_GUIDE_STATE_NORMALIZED_KEY,
    MONSTER_GUIDE_REGULAR_MILESTONES,
    MONSTER_GUIDE_BOSS_MILESTONES,
    MONSTER_GUIDE_DAMAGE_BONUSES,
    markMonsterGuideStateNormalized,
    isMonsterGuideStateNormalized,
    normalizeMonsterGuideBooleanMap,
    normalizeMonsterGuideCountMap,
    createMonsterGuideMilestoneMap,
    createMonsterGuideProgressEntry,
    createMonsterGuideState,
    getMonsterGuideProgressEntry,
    recordMonsterGuideDefeat,
    recordMonsterGuideSighted,
    recordMonsterGuideDrop,
    getMonsterGuideOddsUnlockState,
    getMonsterGuideMilestones,
    getMonsterGuideTier,
    getMonsterGuideDamageBonusForKills,
    getMonsterGuideStatRange,
    getMonsterGuideAggroProfile,
    getMonsterGuideMaps,
    getMonsterGuideStaticEntryCacheKey,
    createMonsterGuideStaticEntry,
    getMonsterGuideBossSetDropRows,
    createMonsterGuideDropInfo,
    applyMonsterGuideDropProgress,
    createMonsterGuideEntrySnapshot,
    getMonsterGuideSnapshotKillValues,
    isMonsterGuideSnapshotCacheCurrent,
    createMonsterGuideGroupSummaries,
    createMonsterGuideSnapshotSummary,
    createMonsterGuideSnapshotValue,
    createMonsterGuideSnapshotCacheState,
    getMonsterDropLabel,
    getMonsterGuideDropKind,
    getMonsterGuideDropKey,
    getMonsterGuideRarityTier,
    getMonsterGuideDropRarity,
    getMonsterGuideQuantityRange,
    createMonsterGuideDropRowFromEntry
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.monsterGuide = Object.assign({}, modules.monsterGuide || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
