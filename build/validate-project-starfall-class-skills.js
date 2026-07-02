#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_DATA_PATH = path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js');
const DEFAULT_GUIDE_PATH = path.join(ROOT, 'CLASS_AND_SKILL_DESIGN_GUIDE.md');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function readText(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function unique(values) {
  return Array.from(new Set(values));
}

function sorted(values) {
  return values.slice().sort();
}

function sameSet(actual, expected) {
  const a = sorted(actual);
  const b = sorted(expected);
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function hasOwn(object, key) {
  return Object.prototype.hasOwnProperty.call(object || {}, key);
}

function validateRequiredFields(record, fields, label) {
  fields.forEach((field) => {
    assert(hasOwn(record, field), `${label} missing required field ${field}`);
    assert(record[field] !== null && record[field] !== undefined && record[field] !== '',
      `${label} has empty required field ${field}`);
  });
}

function validatePrerequisite(skill, prerequisite, skillById) {
  if (Array.isArray(prerequisite.anyOf)) {
    assert(prerequisite.anyOf.length > 0, `${skill.id} has an empty anyOf prerequisite`);
    prerequisite.anyOf.forEach((skillId) => {
      assert(skillById[skillId], `${skill.id} prerequisite references missing skill ${skillId}`);
    });
  }
  if (prerequisite.skillId) {
    assert(skillById[prerequisite.skillId], `${skill.id} prerequisite references missing skill ${prerequisite.skillId}`);
  }
  if (prerequisite.skillId || Array.isArray(prerequisite.anyOf)) {
    assert(Number(prerequisite.rank || 0) > 0, `${skill.id} prerequisite should require a positive rank`);
  }
}

function validateGuideContract(data, guideText) {
  const contract = data.CLASS_SKILL_GUIDE_CONTRACT || {};
  assert(contract.guidePath === 'CLASS_AND_SKILL_DESIGN_GUIDE.md',
    'Class/skill guide contract should point at CLASS_AND_SKILL_DESIGN_GUIDE.md');
  assert(guideText.includes('Project Starfall Class and Skill Design Guide'),
    'CLASS_AND_SKILL_DESIGN_GUIDE.md should be the Project Starfall class and skill guide');
  (contract.requiredGuideSections || []).forEach((section) => {
    assert(guideText.includes(section), `CLASS_AND_SKILL_DESIGN_GUIDE.md missing required section: ${section}`);
  });
  (contract.runtimeDataFiles || []).forEach((filePath) => {
    assert(fs.existsSync(path.join(ROOT, filePath)), `Guide contract data file missing: ${filePath}`);
  });
  (contract.runtimeHookFiles || []).forEach((filePath) => {
    assert(fs.existsSync(path.join(ROOT, filePath)), `Guide contract hook file missing: ${filePath}`);
  });
}

function validateClasses(data) {
  const baseClassIds = Object.keys(data.BASE_CLASSES || {});
  const advancedClassIds = Object.keys(data.ADVANCED_CLASSES || {});
  const expectedBase = data.CLASS_SKILL_GUIDE_CONTRACT.expectedBaseClasses || [];
  const expectedAdvanced = data.CLASS_SKILL_GUIDE_CONTRACT.expectedAdvancedClasses || [];
  assert(sameSet(baseClassIds, expectedBase), `Base class ids should be ${expectedBase.join(', ')}`);
  assert(sameSet(advancedClassIds, expectedAdvanced), `Advanced class ids should be ${expectedAdvanced.join(', ')}`);

  const allClasses = Object.assign({}, data.BASE_CLASSES || {}, data.ADVANCED_CLASSES || {});
  const classIds = Object.keys(allClasses);
  assert(classIds.length === 12, 'Project Starfall should expose exactly 12 playable class ids in the current design');
  assert(data.CLASS_RESOURCE_DEFINITIONS &&
    sameSet(Object.keys(data.CLASS_RESOURCE_DEFINITIONS), classIds),
    'Class resource definitions should cover every playable class exactly once');

  classIds.forEach((classId) => {
    const classData = allClasses[classId];
    validateRequiredFields(classData, data.CLASS_RUNTIME_REQUIRED_FIELDS || [], `class ${classId}`);
    assert(classData.id === classId, `class ${classId} id should match its map key`);
    assert(data.CLASS_ROLE_PROFILES && data.CLASS_ROLE_PROFILES[classId] === classData.roleProfile,
      `class ${classId} should attach its shared role profile`);
    assert(classData.roleProfile.primary && classData.roleProfile.secondary &&
      classData.roleProfile.specialty && classData.roleProfile.summary,
      `class ${classId} role profile should include primary, secondary, specialty, and summary`);
    const resource = data.CLASS_RESOURCE_DEFINITIONS[classId];
    assert(resource && resource.classId === classId, `class ${classId} should have a matching resource definition`);
    assert(resource.label === classData.resourceName,
      `class ${classId} resource definition label should match class resourceName`);
    ['gain', 'spend', 'feedback'].forEach((field) => {
      assert(Array.isArray(resource[field]) && resource[field].length > 0,
        `class ${classId} resource definition should include ${field}`);
    });
    assert(resource.playerDecision && resource.uiWidget,
      `class ${classId} resource definition should describe player decision and UI widget`);
  });

  advancedClassIds.forEach((classId) => {
    const branch = data.ADVANCED_CLASSES[classId];
    validateRequiredFields(branch, data.ADVANCED_CLASS_REQUIRED_FIELDS || [], `advanced class ${classId}`);
    assert(data.BASE_CLASSES[branch.baseClass], `advanced class ${classId} should reference a valid base class`);
    assert(Number(branch.levelRequirement) === 25, `advanced class ${classId} should unlock at level 25`);
  });

  return { classIds, baseClassIds, advancedClassIds };
}

function validateStatuses(data, classIds) {
  const statuses = data.STATUS_EFFECT_DEFINITIONS || {};
  const requiredStatuses = ['burn', 'mark', 'crack', 'slow', 'weakPoint', 'runeLink', 'packMark', 'stagger', 'shield', 'lure', 'haste'];
  requiredStatuses.forEach((statusId) => {
    const status = statuses[statusId];
    assert(status && status.id === statusId, `Missing status effect definition ${statusId}`);
    assert(status.category && status.playerRead && status.bossRule,
      `${statusId} status effect should describe category, player read, and boss rule`);
    assert(Array.isArray(status.sourceClasses) && status.sourceClasses.length > 0,
      `${statusId} status effect should list source classes`);
    status.sourceClasses.forEach((classId) => {
      assert(classIds.includes(classId), `${statusId} status effect references missing class ${classId}`);
    });
  });
}

function validateSkills(data, classIds, advancedClassIds) {
  const skills = data.SKILLS || [];
  const skillById = skills.reduce((lookup, skill) => {
    assert(skill && skill.id, 'Every skill should have an id');
    assert(!lookup[skill.id], `Duplicate skill id ${skill.id}`);
    lookup[skill.id] = skill;
    return lookup;
  }, {});
  const validRoleTags = new Set(data.ROLE_TAGS || []);
  const validPurposes = new Set(Object.keys(data.SKILL_PURPOSES || {}));
  const validCategories = new Set(['attack', 'mobility', 'buff', 'passive']);
  assert(skills.length >= 80, 'Project Starfall should keep the full current class skill roster');

  skills.forEach((skill) => {
    validateRequiredFields(skill, data.SKILL_RUNTIME_REQUIRED_FIELDS || [], `skill ${skill.id}`);
    assert(classIds.includes(skill.owner), `${skill.id} should reference a playable class owner`);
    assert(['Base Skill Batch', 'Advanced Skill Batch'].includes(skill.batch),
      `${skill.id} should use a known skill batch`);
    assert(validCategories.has(skill.category), `${skill.id} has invalid category ${skill.category}`);
    assert(validPurposes.has(skill.purpose), `${skill.id} has invalid purpose ${skill.purpose}`);
    assert(Array.isArray(skill.roleTags) && skill.roleTags.length > 0, `${skill.id} should declare role tags`);
    skill.roleTags.forEach((tag) => assert(validRoleTags.has(tag), `${skill.id} has invalid role tag ${tag}`));
    assert(Number.isFinite(Number(skill.maxRank)) && Number(skill.maxRank) > 0,
      `${skill.id} should declare a positive maxRank`);
    assert(Number.isFinite(Number(skill.resourceCost)) && Number(skill.resourceCost) >= 0,
      `${skill.id} should declare a non-negative MP cost`);
    assert(Number.isFinite(Number(skill.cooldown)) && Number(skill.cooldown) >= 0,
      `${skill.id} should declare a non-negative cooldown`);
    assert(Number.isFinite(Number(skill.lineCount)) && Number(skill.lineCount) >= 0,
      `${skill.id} should declare a non-negative lineCount`);
    assert(Number.isFinite(Number(skill.lineDamageScale)) && Number(skill.lineDamageScale) > 0,
      `${skill.id} should declare a positive lineDamageScale`);
    assert(skill.iconAsset && String(skill.iconAsset).endsWith('.png'), `${skill.id} should reference a PNG icon`);
    assert(String(skill.description || '').length >= 30, `${skill.id} should have a meaningful description`);
    (skill.prerequisites || []).forEach((prerequisite) => validatePrerequisite(skill, prerequisite, skillById));

    if (skill.category !== 'passive') {
      assert(Number(skill.cooldown) > 0, `${skill.id} active skill should have a cooldown`);
      assert(data.SKILL_FX_ANIMATION_ASSETS && data.SKILL_FX_ANIMATION_ASSETS[skill.id],
        `${skill.id} active skill should have a generated combat FX reference`);
      assert(skill.lineCount > 0 || skill.movementEffect || skill.targeting || skill.targetCaps ||
        skill.partyEffect || ['defense', 'buff', 'sustain', 'party skill'].some((keyword) =>
          String(skill.type || '').toLowerCase().includes(keyword)),
        `${skill.id} active skill should expose damage, movement, targeting, field, party, defense, or buff behavior`);
    } else {
      assert(skill.passiveStats, `${skill.id} passive skill should expose passiveStats`);
    }
  });

  advancedClassIds.forEach((classId) => {
    const branch = data.ADVANCED_CLASSES[classId];
    const ownerSkills = skills.filter((skill) => skill.owner === classId);
    const primaryTraining = ownerSkills.filter((skill) => skill.primaryTraining);
    assert(primaryTraining.length === 1, `${classId} should have exactly one advanced primary training skill`);
    assert(primaryTraining[0].prerequisites.length === 0, `${primaryTraining[0].id} should be immediately trainable`);
    assert(primaryTraining[0].purpose === 'trainer', `${primaryTraining[0].id} should use trainer purpose`);
    const partySkill = skillById[branch.partySkillId];
    assert(partySkill && partySkill.owner === classId, `${classId} partySkillId should point to an owned skill`);
    assert(partySkill.roleTags.includes('Party') && partySkill.partyEffect && partySkill.futurePartyEffect,
      `${partySkill.id} should document current and future party behavior`);
  });

  return { skillById };
}

function validateInheritance(data, classIds, advancedClassIds) {
  const EngineSkills = require(path.join(ROOT, 'js/games/project-starfall/engine/skills.js'));
  const getSkillDefinitionsByOwner = (owner) => (data.SKILLS || []).filter((skill) => skill.owner === owner);
  advancedClassIds.forEach((advancedId) => {
    const branch = data.ADVANCED_CLASSES[advancedId];
    const state = { player: { classId: branch.baseClass, advancedClassId: advancedId }, skills: {} };
    const inheritedSkills = EngineSkills.getClassSkills(state, { data, getSkillDefinitionsByOwner });
    const owners = new Set(inheritedSkills.map((skill) => skill.owner));
    assert(owners.has(branch.baseClass) && owners.has(advancedId),
      `${advancedId} should inherit ${branch.baseClass} base skills through getClassSkills`);
    const primary = EngineSkills.getPrimarySkillCandidate({
      player: { classId: branch.baseClass, advancedClassId: advancedId },
      skills: inheritedSkills.reduce((ranks, skill) => {
        ranks[skill.id] = skill.primaryTraining ? 1 : 0;
        return ranks;
      }, {})
    }, {
      data,
      getSkillDefinitionsByOwner,
      getPartySkillId: () => branch.partySkillId
    });
    assert(primary && primary.owner === advancedId && primary.primaryTraining,
      `${advancedId} primary skill candidate should prefer its advanced trainer`);
  });
  assert(classIds.length === Object.keys(data.CLASS_RESOURCE_DEFINITIONS || {}).length,
    'Resource definition coverage should remain aligned with playable classes after inheritance checks');
}

function validateDesignTables(data, classIds) {
  const tooltipFormat = data.CLASS_SKILL_TOOLTIP_FORMAT || {};
  assert(Array.isArray(tooltipFormat.lines) && tooltipFormat.lines.length >= 10,
    'Class/skill tooltip format should expose copy-ready lines');
  ['cost', 'cooldown', 'range', 'current', 'nextRank', 'useWhen', 'weakWhen', 'prerequisite'].forEach((concept) => {
    assert((tooltipFormat.requiredConcepts || []).includes(concept),
      `Tooltip format should require ${concept}`);
  });
  const requiredBalanceFields = ['skill', 'owner', 'tier', 'damage', 'cooldown', 'mpCost', 'classResource', 'range', 'startup', 'recovery', 'areaSize', 'crowdControlValue', 'mobilityValue', 'risk', 'bestUse', 'weakness'];
  assert(sameSet(data.CLASS_SKILL_BALANCE_TUNING_FIELDS || [], requiredBalanceFields),
    'Class/skill balance tuning fields should match the guide table');
  assert((data.CLASS_SKILL_LOADOUT_SLOTS || []).some((slot) => slot.id === 'primarySkill') &&
    (data.CLASS_SKILL_LOADOUT_SLOTS || []).some((slot) => slot.id === 'partySkill'),
    'Class/skill loadout slots should include primary and party skill slots');
  assert((data.CLASS_SKILL_DEBUG_SCENARIOS || []).length >= 6,
    'Class/skill debug scenarios should cover early, advanced, breakpoint, and endgame states');
  const encounterCases = data.CLASS_SKILL_ENCOUNTER_TEST_CASES || [];
  assert(encounterCases.length >= 5, 'Class/skill encounter test cases should cover multiple combat demands');
  const encounterCoverage = new Set();
  encounterCases.forEach((testCase) => {
    ['expectedStrongClasses', 'expectedPressureClasses'].forEach((field) => {
      (testCase[field] || []).forEach((classId) => {
        assert(classIds.includes(classId), `${testCase.id} references missing class ${classId}`);
        encounterCoverage.add(classId);
      });
    });
    assert(testCase.enemyRoles && testCase.terrain && testCase.validates,
      `${testCase.id} should describe enemy roles, terrain, and validated systems`);
  });
  classIds.forEach((classId) => {
    assert(encounterCoverage.has(classId), `${classId} should appear in class/skill encounter test coverage`);
  });
}

function validateProjectStarfallClassSkillData(data, options) {
  const settings = options || {};
  const projectData = data || require(DEFAULT_DATA_PATH);
  const guidePath = settings.guidePath || DEFAULT_GUIDE_PATH;
  const guideText = settings.guideText || readText(guidePath);

  validateGuideContract(projectData, guideText);
  const { classIds, advancedClassIds } = validateClasses(projectData);
  validateStatuses(projectData, classIds);
  validateSkills(projectData, classIds, advancedClassIds);
  validateInheritance(projectData, classIds, advancedClassIds);
  validateDesignTables(projectData, classIds);

  return {
    ok: true,
    classCount: classIds.length,
    skillCount: (projectData.SKILLS || []).length,
    statusCount: Object.keys(projectData.STATUS_EFFECT_DEFINITIONS || {}).length,
    encounterTestCount: (projectData.CLASS_SKILL_ENCOUNTER_TEST_CASES || []).length
  };
}

function main() {
  const data = require(DEFAULT_DATA_PATH);
  const result = validateProjectStarfallClassSkillData(data);
  console.log(`Project Starfall class/skill validation passed: ${result.classCount} classes, ${result.skillCount} skills, ${result.statusCount} status definitions, ${result.encounterTestCount} encounter tests.`);
}

if (require.main === module) {
  main();
}

module.exports = {
  validateProjectStarfallClassSkillData
};
