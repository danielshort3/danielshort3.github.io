(function initProjectStarfallEngineSkills(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getSkillsData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getSkillDefinitionsByOwner(owner, options) {
    const settings = options || {};
    if (typeof settings.getSkillDefinitionsByOwner === 'function') {
      return settings.getSkillDefinitionsByOwner(owner) || [];
    }
    const data = getSkillsData(settings);
    const ownerId = normalizeId(owner);
    return (data.SKILLS || []).filter((skill) => normalizeId(skill && skill.owner) === ownerId);
  }

  function getSkillPoolId(state, skill) {
    return skill && skill.owner === state.player.advancedClassId ? 'advancedSkillPoints' : 'baseSkillPoints';
  }

  function getTotalSkillPoints(player) {
    return Math.max(0, Number(player.baseSkillPoints || 0)) + Math.max(0, Number(player.advancedSkillPoints || 0));
  }

  function getSkillPointAwardForLevel(level) {
    return Number(level || 0) % 5 === 0 ? 4 : 3;
  }

  function getAdvancedSkillPointStartLevel(player, advancedId, options) {
    const data = getSkillsData(options);
    if (!player || !player.classId) return Number.POSITIVE_INFINITY;
    const selectedBranch = advancedId ? data.ADVANCED_CLASSES[advancedId] : data.ADVANCED_CLASSES[player.advancedClassId];
    if (selectedBranch && selectedBranch.baseClass === player.classId) {
      return Math.max(2, Math.floor(Number(selectedBranch.levelRequirement) || Number.POSITIVE_INFINITY));
    }
    const requirements = Object.values(data.ADVANCED_CLASSES)
      .filter((branch) => branch && branch.baseClass === player.classId)
      .map((branch) => Math.floor(Number(branch.levelRequirement) || 0))
      .filter((level) => level >= 2);
    return requirements.length ? Math.min(...requirements) : Number.POSITIVE_INFINITY;
  }

  function getSkillPointBudget(player, advancedId, options) {
    const level = Math.max(1, Math.floor(Number(player && player.level) || 1));
    const advancedStartLevel = getAdvancedSkillPointStartLevel(player, advancedId, options);
    const budget = { base: 2, advanced: 0 };
    for (let awardLevel = 2; awardLevel <= level; awardLevel += 1) {
      const pool = awardLevel >= advancedStartLevel ? 'advanced' : 'base';
      budget[pool] += getSkillPointAwardForLevel(awardLevel);
    }
    return budget;
  }

  function getDefaultSkillRank(skill) {
    if (!skill) return 0;
    if (skill.defaultRank != null) {
      return Math.max(0, Math.min(Number(skill.maxRank || 0), Math.floor(Number(skill.defaultRank || 0) || 0)));
    }
    return skill.prerequisites && skill.prerequisites.length ? 0 : 1;
  }

  function createDefaultRanks(classId, options) {
    const ranks = {};
    const skills = getSkillDefinitionsByOwner(classId, options);
    for (let index = 0; index < skills.length; index += 1) {
      const skill = skills[index];
      ranks[skill.id] = getDefaultSkillRank(skill);
    }
    return ranks;
  }

  function getClassSkills(state, options) {
    const owners = [state.player.classId];
    if (state.player.advancedClassId) owners.push(state.player.advancedClassId);
    const skills = [];
    for (let ownerIndex = 0; ownerIndex < owners.length; ownerIndex += 1) {
      const ownerSkills = getSkillDefinitionsByOwner(owners[ownerIndex], options);
      for (let skillIndex = 0; skillIndex < ownerSkills.length; skillIndex += 1) {
        skills.push(ownerSkills[skillIndex]);
      }
    }
    return skills;
  }

  function getRank(state, skillId) {
    return Number(state.skills[skillId] || 0);
  }

  function getSkillPointTargetForOwner(owner, options) {
    const skills = getSkillDefinitionsByOwner(owner, options);
    if (!skills.length) return 0;
    let totalUpgradeCost = 0;
    for (let index = 0; index < skills.length; index += 1) {
      const skill = skills[index];
      const cost = Math.max(0, Number(skill.maxRank || 0) - getDefaultSkillRank(skill));
      if (cost > 0) totalUpgradeCost += cost;
    }
    return Math.max(0, totalUpgradeCost);
  }

  function getSkillPointPoolOwner(state, poolId) {
    const player = state && state.player || {};
    if (poolId === 'advancedSkillPoints') return normalizeId(player.advancedClassId);
    return normalizeId(player.classId);
  }

  function getSkillPointPoolSpent(state, poolId, options) {
    const owner = getSkillPointPoolOwner(state, poolId);
    if (!owner) return 0;
    const skills = getSkillDefinitionsByOwner(owner, options);
    let spent = 0;
    for (let index = 0; index < skills.length; index += 1) {
      const skill = skills[index];
      const current = Math.max(0, Number(state.skills && state.skills[skill.id] || 0));
      spent += Math.max(0, current - getDefaultSkillRank(skill));
    }
    return spent;
  }

  function getSkillPointPoolEarned(state, poolId, options) {
    const player = state && state.player || {};
    return Math.max(0, Number(player[poolId] || 0)) + getSkillPointPoolSpent(state, poolId, options);
  }

  function getSkillPointPoolTarget(state, poolId, options) {
    return getSkillPointTargetForOwner(getSkillPointPoolOwner(state, poolId), options);
  }

  function isSkillPointPoolAtTarget(state, poolId, options) {
    const target = getSkillPointPoolTarget(state, poolId, options);
    return !target || getSkillPointPoolEarned(state, poolId, options) >= target;
  }

  function normalizeSkillPointValue(value) {
    return Math.max(0, Math.floor(Number(value) || 0));
  }

  function splitLegacySkillPoints(player, options) {
    const total = normalizeSkillPointValue(player.skillPoints);
    const budget = getSkillPointBudget(player, null, options);
    const advanced = Math.min(total, budget.advanced);
    return {
      base: Math.max(0, total - advanced),
      advanced
    };
  }

  function createSkillPointPoolReconciliation(player, advancedId, options) {
    let baseSkillPoints = normalizeSkillPointValue(player.baseSkillPoints);
    let advancedSkillPoints = normalizeSkillPointValue(player.advancedSkillPoints);
    const budget = getSkillPointBudget(player, advancedId, options);
    const baseExcess = Math.max(0, baseSkillPoints - budget.base);
    const advancedDeficit = Math.max(0, budget.advanced - advancedSkillPoints);
    const transfer = Math.min(baseExcess, advancedDeficit);
    if (transfer > 0) {
      baseSkillPoints -= transfer;
      advancedSkillPoints += transfer;
    }
    return {
      baseSkillPoints,
      advancedSkillPoints,
      skillPoints: getTotalSkillPoints({ baseSkillPoints, advancedSkillPoints }),
      transfer
    };
  }

  function createSkillPointGrantPlan(state, poolId, points, options) {
    const player = state && state.player;
    if (!player || !poolId) return { granted: 0 };
    const amount = normalizeSkillPointValue(points);
    if (!amount) return { granted: 0 };
    const target = getSkillPointPoolTarget(state, poolId, options);
    const earned = getSkillPointPoolEarned(state, poolId, options);
    const available = target ? Math.max(0, target - earned) : 0;
    const granted = Math.min(amount, available);
    if (granted <= 0) return { granted: 0 };
    const poolValue = normalizeSkillPointValue(player[poolId]) + granted;
    const projectedPlayer = Object.assign({}, player, { [poolId]: poolValue });
    return {
      poolId,
      granted,
      poolValue,
      skillPoints: getTotalSkillPoints(projectedPlayer)
    };
  }

  function createEarnedSkillPointBudgetGrantPlans(state, options) {
    const player = state && state.player;
    if (!player || !player.classId) return [];
    const budget = getSkillPointBudget(player, player.advancedClassId, options);
    const pools = [
      ['baseSkillPoints', budget.base],
      ['advancedSkillPoints', player.advancedClassId ? budget.advanced : 0]
    ];
    const plans = [];
    pools.forEach(([poolId, earnedBudget]) => {
      const target = getSkillPointPoolTarget(state, poolId, options);
      const cappedBudget = target ? Math.min(target, Math.max(0, Number(earnedBudget || 0))) : 0;
      const earned = getSkillPointPoolEarned(state, poolId, options);
      if (earned < cappedBudget) {
        plans.push({
          poolId,
          points: cappedBudget - earned
        });
      }
    });
    return plans;
  }

  function createSkillPointPoolExcessRebalancePlan(state, options) {
    const player = state && state.player;
    if (!player || !player.classId || !player.advancedClassId) return { transfer: 0 };
    const budget = getSkillPointBudget(player, player.advancedClassId, options);
    const baseTarget = Math.min(getSkillPointTargetForOwner(player.classId, options), Math.max(0, Number(budget.base || 0)));
    const advancedTarget = Math.min(getSkillPointTargetForOwner(player.advancedClassId, options), Math.max(0, Number(budget.advanced || 0)));
    const baseExcess = Math.max(0, getSkillPointPoolEarned(state, 'baseSkillPoints', options) - baseTarget);
    const advancedDeficit = Math.max(0, advancedTarget - getSkillPointPoolEarned(state, 'advancedSkillPoints', options));
    const transfer = Math.min(normalizeSkillPointValue(player.baseSkillPoints), baseExcess, advancedDeficit);
    if (transfer <= 0) return { transfer: 0 };
    const baseSkillPoints = player.baseSkillPoints - transfer;
    const advancedSkillPoints = normalizeSkillPointValue(player.advancedSkillPoints) + transfer;
    return {
      transfer,
      baseSkillPoints,
      advancedSkillPoints,
      skillPoints: getTotalSkillPoints(Object.assign({}, player, { baseSkillPoints, advancedSkillPoints }))
    };
  }

  function getSkillPrerequisiteRank(state, skillId, options) {
    const settings = options || {};
    if (typeof settings.getRank === 'function') return settings.getRank(state, skillId);
    return Number(state.skills[skillId] || 0);
  }

  function prerequisiteMet(state, prerequisite, options) {
    if (!prerequisite) return true;
    if (Array.isArray(prerequisite.anyOf)) {
      return prerequisite.anyOf.some((id) => getSkillPrerequisiteRank(state, id, options) >= Number(prerequisite.rank || 1));
    }
    if (prerequisite.skillId) {
      return getSkillPrerequisiteRank(state, prerequisite.skillId, options) >= Number(prerequisite.rank || 1);
    }
    return true;
  }

  function skillUnlocked(state, skill, options) {
    if (!skill) return false;
    if (!skill.prerequisites || !skill.prerequisites.length) return true;
    const groupedAny = skill.prerequisites.filter((item) => item.any);
    const strict = skill.prerequisites.filter((item) => !item.any);
    if (strict.some((item) => !prerequisiteMet(state, item, options))) return false;
    if (groupedAny.length) return groupedAny.some((item) => prerequisiteMet(state, item, options));
    return true;
  }

  function createPassiveSkillBonuses(state, options) {
    const settings = options || {};
    const isPassive = typeof settings.isPassiveSkill === 'function'
      ? settings.isPassiveSkill
      : (skill) => isPassiveSkillForPrimarySelection(skill, settings);
    const getSkillRank = typeof settings.getRank === 'function'
      ? settings.getRank
      : getRank;
    return getClassSkills(state, settings).reduce((bonuses, skill) => {
      if (!skill || !isPassive(skill) || !skill.passiveStats) return bonuses;
      const rank = Math.max(0, getSkillRank(state, skill.id));
      if (!rank) return bonuses;
      Object.entries(skill.passiveStats).forEach(([key, amount]) => {
        const id = normalizeId(key);
        if (!id) return;
        bonuses[id] = Number(bonuses[id] || 0) + Number(amount || 0) * rank;
      });
      return bonuses;
    }, {});
  }

  function isPassiveSkillForPrimarySelection(skill, options) {
    const settings = options || {};
    if (typeof settings.isPassiveSkill === 'function') return settings.isPassiveSkill(skill);
    return !!(skill && (skill.category === 'passive' || String(skill.type || '').toLowerCase().includes('passive') || Number(skill.cooldown || 0) <= 0 && Number(skill.resourceCost || 0) <= 0 && String(skill.description || '').toLowerCase().includes('passive')));
  }

  function getPrimarySkillCandidate(state, options) {
    const settings = options || {};
    if (!state || !state.player || !state.player.classId) return null;
    const getPartySkillId = typeof settings.getPartySkillId === 'function'
      ? settings.getPartySkillId
      : function getPartySkillIdFallback() {
        return settings.partySkillId || '';
      };
    const skills = getClassSkills(state, settings)
      .filter((skill) => getSkillPrerequisiteRank(state, skill.id, settings) > 0 && skill.id !== getPartySkillId())
      .filter((skill) => !isPassiveSkillForPrimarySelection(skill, settings))
      .filter((skill) => skillUnlocked(state, skill, settings));
    return skills.find((skill) => skill.batch === 'Advanced Skill Batch' && skill.primaryTraining)
      || skills.find((skill) => skill.batch === 'Advanced Skill Batch' && skill.resourceCost > 0)
      || skills.find((skill) => skill.resourceCost > 0)
      || skills[0]
      || null;
  }

  const api = {
    getSkillPoolId,
    getTotalSkillPoints,
    getSkillPointAwardForLevel,
    getAdvancedSkillPointStartLevel,
    getSkillPointBudget,
    getDefaultSkillRank,
    createDefaultRanks,
    getClassSkills,
    getRank,
    getSkillPointTargetForOwner,
    getSkillPointPoolOwner,
    getSkillPointPoolSpent,
    getSkillPointPoolEarned,
    getSkillPointPoolTarget,
    isSkillPointPoolAtTarget,
    normalizeSkillPointValue,
    splitLegacySkillPoints,
    createSkillPointPoolReconciliation,
    createSkillPointGrantPlan,
    createEarnedSkillPointBudgetGrantPlans,
    createSkillPointPoolExcessRebalancePlan,
    prerequisiteMet,
    skillUnlocked,
    createPassiveSkillBonuses,
    getPrimarySkillCandidate
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.skills = Object.assign({}, modules.skills || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
