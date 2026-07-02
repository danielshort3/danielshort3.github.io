(function initProjectStarfallEngineParty(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };

  const PARTY_MAX_MEMBERS = 3;
  const PARTY_STATE_NORMALIZED_KEY = '__projectStarfallPartyStateNormalized';

  function getPartyData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getPartyMaxMembers(options) {
    const settings = options || {};
    return Math.max(1, Math.floor(Number(settings.partyMaxMembers || PARTY_MAX_MEMBERS) || PARTY_MAX_MEMBERS));
  }

  function getPartyNormalizedKey(options) {
    const settings = options || {};
    return String(settings.partyStateNormalizedKey || PARTY_STATE_NORMALIZED_KEY);
  }

  function clonePlain(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function markPartyStateNormalized(party, options) {
    if (!party || typeof party !== 'object') return party;
    const key = getPartyNormalizedKey(options);
    try {
      Object.defineProperty(party, key, {
        value: true,
        configurable: true,
        enumerable: false
      });
    } catch (error) {
      party[key] = true;
    }
    return party;
  }

  function isPartyStateNormalized(party, options) {
    return !!(party && typeof party === 'object' && party[getPartyNormalizedKey(options)] === true);
  }

  function getPartyClassData(classId, options) {
    const data = getPartyData(options);
    const id = normalizeId(classId);
    return (data.ADVANCED_CLASSES || {})[id] || (data.BASE_CLASSES || {})[id] || null;
  }

  function getPartyBaseClassId(classId, options) {
    const data = getPartyData(options);
    const id = normalizeId(classId);
    const advanced = (data.ADVANCED_CLASSES || {})[id];
    if (advanced && advanced.baseClass) return advanced.baseClass;
    return (data.BASE_CLASSES || {})[id] ? id : 'fighter';
  }

  function getPartyResourceName(player, options) {
    const data = getPartyData(options);
    const activePlayer = player || {};
    const advanced = (data.ADVANCED_CLASSES || {})[activePlayer.advancedClassId];
    if (advanced) return advanced.resourceName;
    const base = (data.BASE_CLASSES || {})[activePlayer.classId];
    return base ? base.resourceName : 'Resource';
  }

  function getPlayerClassData(player, options) {
    const data = getPartyData(options);
    const activePlayer = player || {};
    return (data.ADVANCED_CLASSES || {})[activePlayer.advancedClassId] ||
      (data.BASE_CLASSES || {})[activePlayer.classId] ||
      (data.BASE_CLASSES || {}).fighter;
  }

  function getPlayerBaseClassData(player, options) {
    const data = getPartyData(options);
    const activePlayer = player || {};
    return (data.BASE_CLASSES || {})[activePlayer.classId] || (data.BASE_CLASSES || {}).fighter;
  }

  function getPartyMemberRole(classId, fallback, options) {
    const data = getPartyData(options);
    const classData = getPartyClassData(classId, options);
    const profile = classData && classData.roleProfile;
    if (fallback) return fallback;
    if (profile && profile.primary) return String(profile.primary).split('/')[0].trim();
    const baseClass = (data.BASE_CLASSES || {})[getPartyBaseClassId(classId, options)];
    return baseClass && baseClass.weaponType === 'melee' ? 'Defense' : 'Damage';
  }

  function getPartyMemberAssist(classId, fallback) {
    if (fallback && fallback.type) return clonePlain(fallback);
    const id = normalizeId(classId);
    if (id === 'guardian' || id === 'fighter') return { type: 'shield', cooldown: 7.5, shieldPercent: 0.08, color: '#68a9ff' };
    if (id === 'trapper' || id === 'runeMage' || id === 'duelist') return { type: 'control', cooldown: 5.8, powerScale: 0.22, radius: 118, color: '#7bdff2' };
    if (id === 'sniper' || id === 'archer' || id === 'beastArcher') return { type: 'mark', cooldown: 5.2, powerScale: 0.3, radius: 40, color: '#ffe16a' };
    return { type: 'damage', cooldown: 4.8, powerScale: 0.34, radius: 92, color: '#ff8a3d' };
  }

  function getPartyAiLoadout(classId, options) {
    const data = getPartyData(options);
    const id = normalizeId(classId);
    const loadouts = data.PARTY_AI_LOADOUTS || {};
    return loadouts[id] || loadouts[getPartyBaseClassId(id, options)] || null;
  }

  function getPartySkillDefinition(skillId, options) {
    const settings = options || {};
    if (typeof settings.getSkillDefinitionById === 'function') return settings.getSkillDefinitionById(skillId);
    const data = getPartyData(options);
    return getById(data.SKILLS || [], normalizeId(skillId));
  }

  function normalizePartySkillLoadout(classId, source, options) {
    const fallback = getPartyAiLoadout(classId, options);
    const skills = Array.isArray(source) && source.length ? source : fallback && fallback.skills || [];
    return skills
      .map((entry) => {
        const record = typeof entry === 'string' ? { skillId: entry } : entry && typeof entry === 'object' ? entry : null;
        const skillId = normalizeId(record && record.skillId || record && record.id);
        if (!skillId || !getPartySkillDefinition(skillId, options)) return null;
        return {
          skillId,
          minCooldown: Math.max(0, Number(record.minCooldown || record.cooldown || 0) || 0),
          priority: Math.max(0, Number(record.priority || 0) || 0),
          support: !!record.support
        };
      })
      .filter(Boolean);
  }

  function normalizePartyEquipment(classId, source, options) {
    const fallback = getPartyAiLoadout(classId, options);
    return Object.assign({}, fallback && fallback.equipment || {}, source && typeof source === 'object' ? source : {});
  }

  function normalizePartySkillCooldowns(source) {
    return Object.entries(source && typeof source === 'object' ? source : {}).reduce((cooldowns, [key, value]) => {
      const skillId = normalizeId(key);
      if (skillId && Number.isFinite(Number(value))) cooldowns[skillId] = Number(value);
      return cooldowns;
    }, {});
  }

  function getPartyClassStatBonuses(classId, level, fallback, options) {
    if (fallback && typeof fallback === 'object') return clonePlain(fallback);
    const data = getPartyData(options);
    const id = normalizeId(classId);
    const base = (data.BASE_CLASSES || {})[getPartyBaseClassId(id, options)] || (data.BASE_CLASSES || {}).fighter || {};
    const stats = base.stats || {};
    const scale = Math.max(1, Math.floor(Number(level || 1) / 10));
    const role = getPartyMemberRole(id, null, options).toLowerCase();
    const bonuses = {
      hp: Math.round(Number(stats.hp || 150) * 0.16 + scale * 8),
      power: Math.max(1, Math.round(Number(stats.power || 18) * 0.08 + scale)),
      defense: Math.max(0, Math.round(Number(stats.defense || 5) * 0.22)),
      speed: role.includes('mobile') || id === 'duelist' ? 5 : 0
    };
    if (role.includes('boss') || id === 'sniper') bonuses.crit = 2 + scale;
    if (role.includes('control') || id === 'trapper') bonuses.trapDamage = 2 + scale;
    if (role.includes('mobbing') || id === 'fireMage' || id === 'stormMage') bonuses.areaDamage = 2 + scale;
    return bonuses;
  }

  function createPartyGroupBuffState(value) {
    const source = value && typeof value === 'object' ? value : {};
    const id = normalizeId(source.id);
    if (!id) return null;
    return {
      id,
      label: String(source.label || source.id || 'Party Buff'),
      skillId: normalizeId(source.skillId),
      memberId: normalizeId(source.memberId),
      powerScale: Number(source.powerScale || 0),
      defenseScale: Number(source.defenseScale || 0),
      crit: Number(source.crit || 0),
      expiresAt: Number(source.expiresAt || 0)
    };
  }

  function createPartyMemberState(value, index, options) {
    const data = getPartyData(options);
    const source = value && typeof value === 'object' ? value : {};
    const template = getById(data.PROTOTYPE_PARTY_MEMBERS || [], normalizeId(source.id));
    const classId = normalizeId(source.classId || (template && template.classId));
    if (!getPartyClassData(classId, options)) return null;
    const maxMembers = getPartyMaxMembers(options);
    const slot = clamp(Math.floor(Number(source.slot != null ? source.slot : index) || 0), 0, maxMembers - 1);
    const classData = getPartyClassData(classId, options) || {};
    const level = Math.max(1, Math.floor(Number(source.level || 1) || 1));
    const id = normalizeId(source.id) || `ai_${classId}_${slot}_${Date.now()}`;
    return {
      id,
      templateId: normalizeId(source.templateId || (template && template.id)),
      name: String(source.name || (template && template.name) || classData.name || 'AI Ally'),
      classId,
      baseClassId: getPartyBaseClassId(classId, options),
      level,
      role: getPartyMemberRole(classId, template && template.role, options),
      summary: String(source.summary || (template && template.summary) || `${classData.name || 'AI ally'} test companion.`),
      statBonuses: getPartyClassStatBonuses(classId, level, template && template.statBonuses, options),
      assist: getPartyMemberAssist(classId, template && template.assist),
      equipment: normalizePartyEquipment(classId, source.equipment || template && template.equipment, options),
      skillLoadout: normalizePartySkillLoadout(classId, source.skillLoadout || template && template.skillLoadout, options),
      skillCooldowns: normalizePartySkillCooldowns(source.skillCooldowns),
      slot,
      x: Number(source.x || 0),
      y: Number(source.y || 0),
      w: Math.max(28, Number(source.w || 38) || 38),
      h: Math.max(48, Number(source.h || 70) || 70),
      facing: Number(source.facing || 1) >= 0 ? 1 : -1,
      vx: Number(source.vx || 0),
      vy: Number(source.vy || 0),
      grounded: !!source.grounded,
      groundedPlatformId: normalizeId(source.groundedPlatformId),
      groundedPlatformIndex: Number.isFinite(Number(source.groundedPlatformIndex)) ? Number(source.groundedPlatformIndex) : -1,
      climbing: !!source.climbing,
      climbableId: normalizeId(source.climbableId),
      pathTargetPlatformIndex: Number.isFinite(Number(source.pathTargetPlatformIndex)) ? Number(source.pathTargetPlatformIndex) : -1,
      nextJumpAt: Number(source.nextJumpAt || 0),
      airRouteUntil: Number(source.airRouteUntil || 0),
      airRouteVx: Number(source.airRouteVx || 0),
      dropThroughUntil: Number(source.dropThroughUntil || 0),
      dropThroughPlatformId: normalizeId(source.dropThroughPlatformId),
      dropThroughPlatformIndex: Number.isFinite(Number(source.dropThroughPlatformIndex)) ? Number(source.dropThroughPlatformIndex) : -1,
      lastX: Number(source.lastX || source.x || 0),
      stuckTime: Number(source.stuckTime || 0),
      hp: Math.max(1, Number(source.hp || 1) || 1),
      maxHp: Math.max(1, Number(source.maxHp || 1) || 1),
      mode: String(source.mode || 'follow'),
      targetEnemyUid: normalizeId(source.targetEnemyUid),
      targetClaimUntil: Number(source.targetClaimUntil || 0),
      nextAttackAt: Number(source.nextAttackAt || 0),
      nextSkillAt: Number(source.nextSkillAt || source.nextAttackAt || 0),
      lastSkillId: normalizeId(source.lastSkillId),
      nextBuffAt: Number(source.nextBuffAt || 0),
      nextWanderAt: Number(source.nextWanderAt || 0),
      roamTargetX: Number(source.roamTargetX || 0),
      roamTargetY: Number(source.roamTargetY || 0),
      defeatedUntil: Number(source.defeatedUntil || 0),
      sharedXp: Math.max(0, Math.floor(Number(source.sharedXp || 0) || 0)),
      sharedDrops: Math.max(0, Math.floor(Number(source.sharedDrops || 0) || 0)),
      activeBuffs: Array.isArray(source.activeBuffs) ? source.activeBuffs.map(createPartyGroupBuffState).filter(Boolean) : [],
      animationState: String(source.animationState || 'idle'),
      animationStartedAt: Number(source.animationStartedAt || 0),
      animationLockUntil: Number(source.animationLockUntil || 0),
      animationLoop: source.animationLoop !== false
    };
  }

  function createPartyState(value, options) {
    const data = getPartyData(options);
    const source = value && typeof value === 'object' ? value : {};
    const validIds = new Set((data.PROTOTYPE_PARTY_MEMBERS || []).map((member) => member.id));
    const memberIds = Array.isArray(source.memberIds)
      ? source.memberIds.map(normalizeId).filter((id) => validIds.has(id))
      : [];
    const members = Array.isArray(source.members)
      ? source.members.map((member, index) => createPartyMemberState(member, index, options)).filter(Boolean)
      : [];
    const groupBuffs = Array.isArray(source.groupBuffs)
      ? source.groupBuffs.map(createPartyGroupBuffState).filter(Boolean)
      : [];
    return markPartyStateNormalized({
      memberIds: Array.from(new Set(memberIds)).slice(0, getPartyMaxMembers(options)),
      members: members.slice(0, getPartyMaxMembers(options)),
      groupBuffs,
      finderReadyAt: Number(source.finderReadyAt || 0),
      nextAssistAt: Number(source.nextAssistAt || 0),
      commandId: getById(data.PARTY_COMMANDS || [], normalizeId(source.commandId)) ? normalizeId(source.commandId) : 'balanced',
      commandUntil: Number(source.commandUntil || 0)
    }, options);
  }

  const api = {
    PARTY_MAX_MEMBERS,
    PARTY_STATE_NORMALIZED_KEY,
    markPartyStateNormalized,
    isPartyStateNormalized,
    createPartyState,
    getPartyClassData,
    getPartyBaseClassId,
    getPartyResourceName,
    getPlayerClassData,
    getPlayerBaseClassData,
    getPartyMemberRole,
    getPartyMemberAssist,
    getPartyAiLoadout,
    normalizePartySkillLoadout,
    normalizePartyEquipment,
    normalizePartySkillCooldowns,
    getPartyClassStatBonuses,
    createPartyMemberState,
    createPartyGroupBuffState
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.party = Object.assign({}, modules.party || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
