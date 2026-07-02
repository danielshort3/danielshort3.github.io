(function initProjectStarfallUiSkillPrerequisites(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const UiModules = global.ProjectStarfallUiModules || {};
  const UiLookups = (typeof require === 'function' ? require('./lookups.js') : null) || UiModules.lookups || {};

  function getSkillById(skillId, options) {
    const settings = options || {};
    if (settings.getSkillById) return settings.getSkillById(skillId);
    if (UiLookups.getSkillById) return UiLookups.getSkillById(skillId, settings);
    return null;
  }

  function getSkillCurrentRank(snapshot, skillId) {
    const ranks = snapshot && snapshot.state && snapshot.state.skills || {};
    return Number(ranks[skillId] || 0);
  }

  function createSkillPrerequisiteEntry(snapshot, skillId, rank, mode, options) {
    if (!skillId) return null;
    const requiredRank = Math.max(1, Number(rank || 1));
    const currentRank = getSkillCurrentRank(snapshot, skillId);
    return {
      skillId,
      skill: getSkillById(skillId, options),
      requiredRank,
      currentRank,
      met: currentRank >= requiredRank,
      mode: mode || 'all'
    };
  }

  function getSkillPrerequisiteGroups(snapshot, skill, options) {
    if (!skill || !skill.prerequisites || !skill.prerequisites.length) return [];
    const groups = [];
    const anyItems = [];
    skill.prerequisites.forEach((item, index) => {
      if (!item) return;
      if (item.any) {
        anyItems.push(item);
        return;
      }
      if (Array.isArray(item.anyOf)) {
        const entries = item.anyOf
          .map((skillId) => createSkillPrerequisiteEntry(snapshot, skillId, item.rank, 'any', options))
          .filter(Boolean);
        if (entries.length) {
          groups.push({
            id: `anyOf-${index}`,
            mode: 'any',
            entries,
            met: entries.some((entry) => entry.met)
          });
        }
        return;
      }
      const entry = createSkillPrerequisiteEntry(snapshot, item.skillId, item.rank, 'all', options);
      if (entry) {
        groups.push({
          id: `all-${index}`,
          mode: 'all',
          entries: [entry],
          met: entry.met
        });
      }
    });
    if (anyItems.length) {
      const entries = anyItems
        .map((item) => createSkillPrerequisiteEntry(snapshot, item.skillId, item.rank, 'any', options))
        .filter(Boolean);
      if (entries.length) {
        groups.push({
          id: 'any-group',
          mode: 'any',
          entries,
          met: entries.some((entry) => entry.met)
        });
      }
    }
    return groups;
  }

  function getSkillPrerequisiteEntries(snapshot, skill, options) {
    return getSkillPrerequisiteGroups(snapshot, skill, options).reduce((entries, group) => {
      return entries.concat(group.entries.map((entry) => Object.assign({}, entry, {
        groupId: group.id,
        groupMode: group.mode,
        groupMet: group.met
      })));
    }, []);
  }

  function getSkillPrerequisiteEntryLabel(entry, includeProgress) {
    const name = entry && entry.skill ? entry.skill.name : entry && entry.skillId ? entry.skillId : 'Unknown skill';
    const base = `${name} level ${entry ? entry.requiredRank : 1}`;
    if (!includeProgress || !entry) return base;
    return `${base} (${entry.currentRank}/${entry.requiredRank})`;
  }

  function getSkillPrerequisiteGroupLabel(group, includeProgress) {
    if (!group || !group.entries || !group.entries.length) return '';
    const labels = group.entries.map((entry) => getSkillPrerequisiteEntryLabel(entry, includeProgress));
    return group.mode === 'any' && labels.length > 1 ? `One of ${labels.join(' / ')}` : labels[0];
  }

  function getSkillPrerequisiteDetailLabels(snapshot, skill, options) {
    return getSkillPrerequisiteGroups(snapshot, skill, options)
      .map((group) => getSkillPrerequisiteGroupLabel(group, true))
      .filter(Boolean);
  }

  function getMissingPrerequisiteLabels(snapshot, skill, options) {
    if (!skill || !skill.prerequisites || !skill.prerequisites.length) return [];
    return getSkillPrerequisiteGroups(snapshot, skill, options)
      .filter((group) => !group.met)
      .map((group) => getSkillPrerequisiteGroupLabel(group));
  }

  function getFutureSkillRequirementEntries(skill, options) {
    if (!skill || !skill.id) return [];
    const settings = options || {};
    const data = settings.data || Data;
    const getSkillOwnerLabel = settings.getSkillOwnerLabel || ((owner) => owner);
    const getSkillOwnerSort = settings.getSkillOwnerSort || (() => 999);
    const byTarget = {};
    (data.SKILLS || []).forEach((target) => {
      if (!target || target.id === skill.id || !target.prerequisites || !target.prerequisites.length) return;
      target.prerequisites.forEach((item) => {
        if (!item) return;
        const ids = Array.isArray(item.anyOf) ? item.anyOf : item.skillId ? [item.skillId] : [];
        if (!ids.includes(skill.id)) return;
        const requiredRank = Math.max(1, Number(item.rank || 1));
        const current = byTarget[target.id];
        if (current && current.requiredRank >= requiredRank) return;
        byTarget[target.id] = {
          skillId: target.id,
          skill: target,
          owner: target.owner,
          ownerLabel: getSkillOwnerLabel(target.owner),
          requiredRank,
          mode: item.any || Array.isArray(item.anyOf) ? 'any' : 'all'
        };
      });
    });
    return Object.keys(byTarget)
      .map((id) => byTarget[id])
      .sort((a, b) => getSkillOwnerSort(a.owner) - getSkillOwnerSort(b.owner) ||
        a.requiredRank - b.requiredRank ||
        String(a.skill && a.skill.name || a.skillId).localeCompare(String(b.skill && b.skill.name || b.skillId)));
  }

  function getFutureSkillRequirementLabels(skill, options) {
    return getFutureSkillRequirementEntries(skill, options).map((entry) => {
      const name = entry.skill ? entry.skill.name : entry.skillId;
      return `${entry.ownerLabel}: ${name} requires this level ${entry.requiredRank}`;
    });
  }

  function createSkillPrerequisiteUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      data: settings.data || Data,
      getSkillById: typeof settings.getSkillById === 'function' ? settings.getSkillById : (skillId) => getSkillById(skillId, settings),
      getSkillOwnerLabel: typeof settings.getSkillOwnerLabel === 'function' ? settings.getSkillOwnerLabel : (owner) => owner,
      getSkillOwnerSort: typeof settings.getSkillOwnerSort === 'function' ? settings.getSkillOwnerSort : () => 999
    });
    function getOptions(extra) {
      return Object.assign({}, helperOptions, extra || {});
    }
    return Object.freeze({
      getSkillCurrentRank,
      createSkillPrerequisiteEntry: (snapshot, skillId, rank, mode, extraOptions) => createSkillPrerequisiteEntry(snapshot, skillId, rank, mode, getOptions(extraOptions)),
      getSkillPrerequisiteGroups: (snapshot, skill, extraOptions) => getSkillPrerequisiteGroups(snapshot, skill, getOptions(extraOptions)),
      getSkillPrerequisiteEntries: (snapshot, skill, extraOptions) => getSkillPrerequisiteEntries(snapshot, skill, getOptions(extraOptions)),
      getSkillPrerequisiteEntryLabel,
      getSkillPrerequisiteGroupLabel,
      getSkillPrerequisiteDetailLabels: (snapshot, skill, extraOptions) => getSkillPrerequisiteDetailLabels(snapshot, skill, getOptions(extraOptions)),
      getMissingPrerequisiteLabels: (snapshot, skill, extraOptions) => getMissingPrerequisiteLabels(snapshot, skill, getOptions(extraOptions)),
      getFutureSkillRequirementEntries: (skill, extraOptions) => getFutureSkillRequirementEntries(skill, getOptions(extraOptions)),
      getFutureSkillRequirementLabels: (skill, extraOptions) => getFutureSkillRequirementLabels(skill, getOptions(extraOptions))
    });
  }

  const api = {
    getSkillCurrentRank,
    createSkillPrerequisiteEntry,
    getSkillPrerequisiteGroups,
    getSkillPrerequisiteEntries,
    getSkillPrerequisiteEntryLabel,
    getSkillPrerequisiteGroupLabel,
    getSkillPrerequisiteDetailLabels,
    getMissingPrerequisiteLabels,
    getFutureSkillRequirementEntries,
    getFutureSkillRequirementLabels,
    createSkillPrerequisiteUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.skillPrerequisites = Object.assign({}, modules.skillPrerequisites || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
