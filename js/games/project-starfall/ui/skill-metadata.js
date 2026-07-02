(function initProjectStarfallUiSkillMetadata(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};

  function isPassiveSkill(skill) {
    return !!(skill && (skill.category === 'passive' || String(skill.type || '').toLowerCase().includes('passive')));
  }

  function isBuffSkill(skill) {
    const type = String(skill && skill.type || '').toLowerCase();
    return !!(skill && (skill.category === 'buff' || (!type.includes('debuff') && type.includes('buff')) || type.includes('party') || type.includes('stance')));
  }

  function isMobilitySkill(skill) {
    const id = String(skill && skill.id || '');
    return !!(skill && (skill.category === 'mobility' || skill.movementEffect || id.includes('blink') || id.includes('dash') || id.includes('roll') || id.includes('leap')));
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
    if (!skill || isPassiveSkill(skill) || skill.movementEffect) return 0;
    const type = String(skill.type || '').toLowerCase();
    if (type.includes('buff') || type.includes('defense') || type.includes('party') || type.includes('mobility') || type.includes('stance')) return 0;
    return Math.max(1, Math.min(8, Math.floor(Number(skill.lineCount || 1) || 1)));
  }

  function getSkillPointPoolId(snapshot, skill) {
    return skill && skill.owner === snapshot.state.player.advancedClassId ? 'advancedSkillPoints' : 'baseSkillPoints';
  }

  function getSkillPointPoolLabel(snapshot, skill) {
    return getSkillPointPoolId(snapshot, skill) === 'advancedSkillPoints' ? 'Advanced SP' : 'Base SP';
  }

  function getSkillMpCost(skill, rank) {
    return Math.max(0, Math.round((skill.resourceCost || 0) * (1 - Math.min(Number(rank) || 0, 10) * 0.015)));
  }

  function getVisibleSkillCooldown(skill) {
    if (!skill || isPassiveSkill(skill)) return 0;
    const cooldown = Math.max(0, Number(skill.cooldown || 0) || 0);
    return cooldown >= 1 ? cooldown : 0;
  }

  function formatSkillPercent(value) {
    return `${Math.max(1, Math.round(Number(value || 0) * 100))}%`;
  }

  function getSkillOwnerLabel(owner, options) {
    const data = options && options.data || Data;
    const ownerData = ((data.BASE_CLASSES || {})[owner] || (data.ADVANCED_CLASSES || {})[owner]) || null;
    return ownerData ? ownerData.name : owner;
  }

  function getSkillOwnerSort(owner, options) {
    const data = options && options.data || Data;
    const baseOwners = Object.keys(data.BASE_CLASSES || {});
    const advancedOwners = Object.keys(data.ADVANCED_CLASSES || {});
    const baseIndex = baseOwners.indexOf(owner);
    if (baseIndex >= 0) return baseIndex;
    const advancedIndex = advancedOwners.indexOf(owner);
    if (advancedIndex >= 0) return baseOwners.length + advancedIndex;
    return 999;
  }

  function getSkillPurposeMeta(skill, options) {
    const purposeId = String(skill && skill.purpose || '');
    const purposes = options && options.skillPurposes || Data.SKILL_PURPOSES || {};
    return purposeId && purposes[purposeId] ? purposes[purposeId] : null;
  }

  function getSkillPurposeLabel(skill, options) {
    const purpose = getSkillPurposeMeta(skill, options);
    return purpose ? purpose.label : '';
  }

  function createSkillMetadataUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      data: settings.data || Data,
      skillPurposes: settings.skillPurposes || (settings.data || Data).SKILL_PURPOSES || {}
    });
    return Object.freeze({
      isPassiveSkill,
      isBuffSkill,
      isMobilitySkill,
      isDefensiveSkill,
      getSkillLineCount,
      getSkillPointPoolId,
      getSkillPointPoolLabel,
      getSkillMpCost,
      getVisibleSkillCooldown,
      formatSkillPercent,
      getSkillOwnerLabel: (owner) => getSkillOwnerLabel(owner, helperOptions),
      getSkillOwnerSort: (owner) => getSkillOwnerSort(owner, helperOptions),
      getSkillPurposeMeta: (skill) => getSkillPurposeMeta(skill, helperOptions),
      getSkillPurposeLabel: (skill) => getSkillPurposeLabel(skill, helperOptions)
    });
  }

  const api = {
    isPassiveSkill,
    isBuffSkill,
    isMobilitySkill,
    isDefensiveSkill,
    getSkillLineCount,
    getSkillPointPoolId,
    getSkillPointPoolLabel,
    getSkillMpCost,
    getVisibleSkillCooldown,
    formatSkillPercent,
    getSkillOwnerLabel,
    getSkillOwnerSort,
    getSkillPurposeMeta,
    getSkillPurposeLabel,
    createSkillMetadataUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.skillMetadata = Object.assign({}, modules.skillMetadata || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
