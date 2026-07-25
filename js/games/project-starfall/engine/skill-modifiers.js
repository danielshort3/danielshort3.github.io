(function initProjectStarfallEngineSkillModifiers(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  function getSkillModifierData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getSkillRank(skillId, options) {
    const settings = options || {};
    if (typeof settings.getSkillRank === 'function') return Math.max(0, Number(settings.getSkillRank(skillId) || 0));
    return Math.max(0, Number(settings.skillRanks && settings.skillRanks[skillId] || 0));
  }

  function createSkillModifierState(value, options) {
    const data = getSkillModifierData(options);
    const source = value && typeof value === 'object' ? value : {};
    const validModifierIds = new Set((data.SKILL_MODIFIERS || []).map((modifier) => modifier.id));
    const validSkillIds = new Set((data.SKILLS || []).map((skill) => skill.id));
    const activeBySkillId = {};
    Object.entries(source.activeBySkillId || {}).forEach(([skillId, modifierId]) => {
      const skillKey = normalizeId(skillId);
      const modifierKey = normalizeId(modifierId);
      if (validSkillIds.has(skillKey) && validModifierIds.has(modifierKey)) activeBySkillId[skillKey] = modifierKey;
    });
    const unlockedModifierIds = Array.isArray(source.unlockedModifierIds)
      ? source.unlockedModifierIds.map(normalizeId).filter((id) => validModifierIds.has(id))
      : [];
    return {
      activeBySkillId,
      unlockedModifierIds: Array.from(new Set(unlockedModifierIds))
    };
  }

  function isSkillModifierUnlocked(modifier, player, options) {
    const data = getSkillModifierData(options);
    if (!modifier) return false;
    const activePlayer = player || {};
    if (activePlayer.level < Number(modifier.unlockLevel || 1)) return false;
    const skill = getById(data.SKILLS || [], modifier.skillId);
    if (skill && getSkillRank(skill.id, options) <= 0) return false;
    return true;
  }

  function createUnlockedSkillModifierIds(state, player, options) {
    const data = getSkillModifierData(options);
    const current = createSkillModifierState(state, options);
    const unlocked = new Set(current.unlockedModifierIds || []);
    (data.SKILL_MODIFIERS || []).forEach((modifier) => {
      if (isSkillModifierUnlocked(modifier, player, options)) unlocked.add(modifier.id);
    });
    return Array.from(unlocked);
  }

  function getSkillModifierForSkill(skill, state, unlockedIds, options) {
    const data = getSkillModifierData(options);
    if (!skill || !skill.id) return null;
    const current = createSkillModifierState(state, options);
    const unlocked = new Set((unlockedIds || []).map(normalizeId));
    const activeId = current.activeBySkillId && current.activeBySkillId[skill.id];
    const active = activeId ? getById(data.SKILL_MODIFIERS || [], activeId) : null;
    if (active && unlocked.has(active.id)) return active;
    return (data.SKILL_MODIFIERS || []).find((modifier) => modifier && modifier.skillId === skill.id && unlocked.has(modifier.id)) || null;
  }

  function createSkillModifierDamageScale(modifier, skill, enemy, options) {
    const settings = options || {};
    const now = Number(settings.nowSeconds || 0);
    let scale = modifier ? Number(modifier.damageScale || 1) : 1;
    const breakProfile = settings.breakProfile || null;
    if (breakProfile && enemy && Number(enemy.brokenUntil || 0) > now) {
      scale *= Number(breakProfile.damageTakenScale || 1);
    }
    if (enemy && skill && skill.owner === 'runeMage' && Number(enemy.runeFieldDamageTakenUntil || 0) > now) {
      scale *= Math.max(1, Number(enemy.runeFieldDamageTakenScale || 1));
    }
    if (!modifier) return scale;
    if (enemy && Number(enemy.brokenUntil || 0) > now) scale *= Number(modifier.brokenDamageScale || 1);
    if (enemy && (Number(enemy.marked || 0) > 0 || Number(enemy.weakPoint || 0) > 0)) scale *= Number(modifier.markedDamageScale || 1);
    return scale;
  }

  function isPassiveSkill(skill) {
    return !!(skill && (skill.category === 'passive' || String(skill.type || '').toLowerCase().includes('passive') || Number(skill.cooldown || 0) <= 0 && Number(skill.resourceCost || 0) <= 0 && String(skill.description || '').toLowerCase().includes('passive')));
  }

  function isBuffSkill(skill) {
    const type = String(skill && skill.type || '').toLowerCase();
    return !!(skill && (skill.category === 'buff' || (!type.includes('debuff') && type.includes('buff')) || type.includes('party') || type.includes('stance')));
  }

  function isMobilitySkill(skill) {
    const id = String(skill && skill.id || '');
    return !!(skill && (skill.category === 'mobility' || skill.movementEffect || id.includes('blink') || id.includes('dash') || id.includes('roll') || id.includes('leap')));
  }

  function isTeleportSkill(skill) {
    const movement = skill && skill.movementEffect || {};
    const id = String(skill && skill.id || '').toLowerCase();
    return !!(skill && (movement.mode === 'blink' || id.includes('blink')));
  }

  function getMovementSkillBlockReason(skill, player) {
    const effect = skill && skill.movementEffect || {};
    const isTeleport = effect.mode === 'blink' || String(skill && skill.id || '').includes('blink');
    const isAirMovement = !player.grounded && (skill && skill.movementEffect || String(skill && skill.id || '').match(/dash|roll|leap/));
    if (isTeleport && !player.grounded) return 'Teleport skills can only be used on the ground.';
    if (isAirMovement && player.airMobilitySkillId) return 'Air dash is available again after landing.';
    return '';
  }

  function isAttackSkill(skill) {
    return !!(skill && skill.category === 'attack');
  }

  function isDefensiveSkill(skill) {
    if (!skill) return false;
    const id = String(skill.id || '');
    const type = String(skill.type || '').toLowerCase();
    return id === 'fighter_guard' ||
      id === 'mage_mana_shield' ||
      id === 'guardian_impact_guard' ||
      id === 'guardian_oath_barrier' ||
      id === 'guardian_shield_wall' ||
      type.includes('defense') ||
      type.includes('stance');
  }

  function getSkillLineCount(skill) {
    if (!skill || isPassiveSkill(skill) || isMobilitySkill(skill)) return 0;
    const type = String(skill.type || '').toLowerCase();
    if (isBuffSkill(skill) || type.includes('defense') || type.includes('party') || type.includes('mobility') || type.includes('stance')) return 0;
    return Math.max(1, Math.min(8, Math.floor(Number(skill.lineCount || 1) || 1)));
  }

  function getSkillBasePower(skill, rank, stats) {
    const sourceStats = stats || {};
    const type = String(skill && skill.type || '');
    const finisher = type.includes('Finisher') || type.includes('Ultimate') ? 0.65 : 0;
    return (sourceStats.power || 1) * (1.25 + Math.max(1, Number(rank) || 1) * 0.11 + finisher);
  }

  function getSkillProjectileRange(targeting, rank, stats, classStats) {
    const sourceTargeting = targeting || {};
    const sourceStats = stats || {};
    const baseClassStats = classStats || {};
    const rangeBonus = Math.max(0, Number(sourceStats.range || 0) - Number(baseClassStats.range || 0));
    return Math.max(80, Number(sourceTargeting.range || sourceStats.range || 420) + rangeBonus + Math.max(0, Number(rank) || 0) * Number(sourceTargeting.rangePerRank || 0));
  }

  function isMageAirborneProjectileSkill(skill, player) {
    const mode = skill && skill.targeting && skill.targeting.mode;
    return !!player && player.classId === 'mage' && !player.grounded && (mode === 'projectile' || mode === 'chain');
  }

  function isMageAirborneBasicAttackBlocked(player, classData) {
    if (!player || player.classId !== 'mage' || player.grounded) return false;
    if (!classData) return true;
    return classData.weaponType !== 'melee';
  }

  function getSkillTargetCap(skill, channel, fallback) {
    const sourceSkill = skill || {};
    const caps = sourceSkill && sourceSkill.targetCaps && typeof sourceSkill.targetCaps === 'object' ? sourceSkill.targetCaps : {};
    const candidates = [caps[channel], caps.default, sourceSkill && sourceSkill.targeting && sourceSkill.targeting.maxTargets];
    for (let index = 0; index < candidates.length; index += 1) {
      if (candidates[index] === null || typeof candidates[index] === 'undefined' || candidates[index] === '') continue;
      const value = Math.floor(Number(candidates[index]));
      if (Number.isFinite(value) && value >= 0) return value;
    }
    if (channel === 'direct') return 1;
    if (channel === 'movement') return 3;
    if (channel === 'field') return 5;
    if (channel === 'burnSpread') return 3;
    if (channel === 'runeDetonation') return 6;
    if (channel === 'trapDetonate') return 8;
    if (channel === 'finisherArea') return 7;
    if (channel === 'projectileExplosion' || channel === 'area') {
      const id = String(sourceSkill && sourceSkill.id || '').toLowerCase();
      const type = String(sourceSkill && sourceSkill.type || '').toLowerCase();
      if (type.includes('finisher') || type.includes('ultimate') || id.includes('verdict') || id.includes('inferno') || id.includes('grand')) return 7;
      return 5;
    }
    const fallbackValue = Math.floor(Number(fallback));
    return Number.isFinite(fallbackValue) && fallbackValue >= 0 ? fallbackValue : 1;
  }

  function getChainTargetCount(targeting, rank) {
    const source = targeting || {};
    const base = Math.max(1, Math.floor(Number(source.chainTargets || 2) || 2));
    const every = Math.max(1, Math.floor(Number(source.chainTargetsPerRanks || 3) || 3));
    const maxTargets = Math.max(base, Math.floor(Number(source.maxChainTargets || 5) || 5));
    return Math.min(maxTargets, base + Math.floor(Math.max(0, Number(rank || 0)) / every));
  }

  function getSkillModifierSnapshotSkillIds(source) {
    const modifiers = source || [];
    return Array.from(new Set(modifiers
      .map((modifier) => normalizeId(modifier && modifier.skillId))
      .filter(Boolean))).sort();
  }

  function getSkillModifierSnapshotCacheKey(state, player, unlockedIds, revisionKey, options) {
    const data = getSkillModifierData(options);
    const activePlayer = player || {};
    const current = createSkillModifierState(state, options);
    const activeBySkillId = current.activeBySkillId || {};
    const activeKey = Object.keys(activeBySkillId)
      .sort()
      .map((skillId) => `${normalizeId(skillId)}:${normalizeId(activeBySkillId[skillId])}`)
      .join(',');
    const skillIds = (options && options.skillIds) || getSkillModifierSnapshotSkillIds(data.SKILL_MODIFIERS || []);
    const rankKey = skillIds
      .map((skillId) => `${skillId}:${getSkillRank(skillId, options)}`)
      .join(',');
    return [
      revisionKey,
      Math.max(1, Number(activePlayer.level || 1) || 1),
      normalizeId(activePlayer.classId),
      normalizeId(activePlayer.advancedClassId),
      rankKey,
      activeKey,
      (unlockedIds || []).map(normalizeId).filter(Boolean).sort().join(',')
    ].join('|');
  }

  function createSkillModifierSnapshot(state, unlockedIds, options) {
    const data = getSkillModifierData(options);
    const current = createSkillModifierState(state, options);
    const source = data.SKILL_MODIFIERS || [];
    const unlocked = new Set((unlockedIds || []).map(normalizeId));
    const activeBySkillId = current.activeBySkillId || {};
    const selectedBySkillId = {};
    for (let index = 0; index < source.length; index += 1) {
      const modifier = source[index];
      if (!modifier || !unlocked.has(modifier.id)) continue;
      const skillId = normalizeId(modifier.skillId);
      if (skillId && activeBySkillId[skillId] === modifier.id) selectedBySkillId[skillId] = modifier.id;
    }
    for (let index = 0; index < source.length; index += 1) {
      const modifier = source[index];
      if (!modifier || !unlocked.has(modifier.id)) continue;
      const skillId = normalizeId(modifier.skillId);
      if (skillId && !selectedBySkillId[skillId]) selectedBySkillId[skillId] = modifier.id;
    }
    return {
      activeBySkillId: Object.assign({}, activeBySkillId),
      modifiers: source.map((modifier) => {
        const skillId = normalizeId(modifier && modifier.skillId);
        const skill = getById(data.SKILLS || [], skillId);
        return Object.assign({}, modifier, {
          skillName: skill && skill.name || modifier.skillId,
          unlocked: unlocked.has(modifier.id),
          active: !!(skillId && selectedBySkillId[skillId] === modifier.id)
        });
      })
    };
  }

  const api = {
    createSkillModifierState,
    isSkillModifierUnlocked,
    createUnlockedSkillModifierIds,
    getSkillModifierForSkill,
    createSkillModifierDamageScale,
    isPassiveSkill,
    isBuffSkill,
    isMobilitySkill,
    isTeleportSkill,
    getMovementSkillBlockReason,
    isAttackSkill,
    isDefensiveSkill,
    getSkillLineCount,
    getSkillBasePower,
    getSkillProjectileRange,
    isMageAirborneProjectileSkill,
    isMageAirborneBasicAttackBlocked,
    getSkillTargetCap,
    getChainTargetCount,
    getSkillModifierSnapshotSkillIds,
    getSkillModifierSnapshotCacheKey,
    createSkillModifierSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.skillModifiers = Object.assign({}, modules.skillModifiers || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
