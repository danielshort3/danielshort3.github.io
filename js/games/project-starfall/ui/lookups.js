(function initProjectStarfallUiLookups(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};

  function createSkillsById(data) {
    const source = data || Data;
    return Object.freeze((source.SKILLS || []).reduce((map, skill) => {
      const id = String(skill && skill.id || '');
      if (id) map[id] = skill;
      return map;
    }, {}));
  }

  function createSkillsByOwner(data) {
    const source = data || Data;
    return Object.freeze((source.SKILLS || []).reduce((map, skill) => {
      const owner = String(skill && skill.owner || '').trim();
      if (!owner) return map;
      if (!map[owner]) map[owner] = [];
      map[owner].push(skill);
      return map;
    }, {}));
  }

  function createCardDefinitionsById(data) {
    const source = data || Data;
    return Object.freeze((source.CARD_DEFINITIONS || []).reduce((map, definition) => {
      const id = String(definition && definition.id || '').trim();
      if (id) map[id] = definition;
      return map;
    }, {}));
  }

  function createGearTraitsById(data) {
    const source = data || Data;
    return Object.freeze((source.GEAR_TRAITS || []).reduce((map, trait) => {
      const id = String(trait && trait.id || '').trim();
      if (id) map[id] = trait;
      return map;
    }, {}));
  }

  function createEquipmentSetsById(data) {
    const source = data || Data;
    return Object.freeze((source.EQUIPMENT_SETS || []).reduce((map, set) => {
      const id = String(set && set.id || '').trim();
      if (id) map[id] = set;
      return map;
    }, {}));
  }

  function getPotentialTierFallback(data) {
    const source = data || Data;
    return (source.POTENTIAL_TIERS || [])[0] || null;
  }

  function createPotentialTiersById(data) {
    const source = data || Data;
    return Object.freeze((source.POTENTIAL_TIERS || []).reduce((map, tier) => {
      const id = String(tier && tier.id || '').trim();
      if (id) map[id] = tier;
      return map;
    }, {}));
  }

  function createPotentialTierIndexById(data) {
    const source = data || Data;
    return Object.freeze((source.POTENTIAL_TIERS || []).reduce((map, tier, index) => {
      const id = String(tier && tier.id || '').trim();
      if (id) map[id] = index;
      return map;
    }, {}));
  }

  function createPotentialLinePoolsByStat(data) {
    const source = data || Data;
    return Object.freeze((source.POTENTIAL_LINE_POOLS || []).reduce((map, pool) => {
      const stat = String(pool && pool.stat || '').trim();
      if (stat) map[stat] = pool;
      return map;
    }, {}));
  }

  function createEquipmentCatalog(data) {
    const source = data || Data;
    return Object.freeze((source.SHOP_ITEMS || []).concat(source.RANDOM_EQUIPMENT_ITEMS || [], source.BOSS_EQUIPMENT_ITEMS || []));
  }

  function createEquipmentTemplateById(catalog) {
    const source = Array.isArray(catalog) ? catalog : createEquipmentCatalog(Data);
    return Object.freeze(source.reduce((map, item) => {
      const id = String(item && item.id || '').trim();
      if (id && !map[id]) map[id] = item;
      return map;
    }, {}));
  }

  const SKILLS_BY_ID = createSkillsById(Data);
  const SKILLS_BY_OWNER = createSkillsByOwner(Data);
  const CARD_DEFINITIONS_BY_ID = createCardDefinitionsById(Data);
  const GEAR_TRAITS_BY_ID = createGearTraitsById(Data);
  const EQUIPMENT_SETS_BY_ID = createEquipmentSetsById(Data);
  const POTENTIAL_TIER_FALLBACK = getPotentialTierFallback(Data);
  const POTENTIAL_TIERS_BY_ID = createPotentialTiersById(Data);
  const POTENTIAL_TIER_INDEX_BY_ID = createPotentialTierIndexById(Data);
  const POTENTIAL_LINE_POOLS_BY_STAT = createPotentialLinePoolsByStat(Data);
  const EQUIPMENT_CATALOG = createEquipmentCatalog(Data);
  const EQUIPMENT_TEMPLATE_BY_ID = createEquipmentTemplateById(EQUIPMENT_CATALOG);

  function normalizePotentialTierId(tierId, options) {
    const id = String(tierId || '').trim();
    const data = options && options.data || Data;
    return (data.POTENTIAL_TIERS || []).some((tier) => tier && tier.id === id) ? id : '';
  }

  function getPotentialTierDefinition(tierId, options) {
    const settings = options || {};
    const id = normalizePotentialTierId(tierId, settings) || 'rare';
    const lookup = settings.potentialTiersById || POTENTIAL_TIERS_BY_ID;
    const fallback = settings.potentialTierFallback || POTENTIAL_TIER_FALLBACK;
    return lookup[id] || fallback;
  }

  function getPotentialTierRank(tierId, options) {
    const settings = options || {};
    const id = normalizePotentialTierId(tierId, settings);
    const lookup = settings.potentialTierIndexById || POTENTIAL_TIER_INDEX_BY_ID;
    return Object.prototype.hasOwnProperty.call(lookup, id) ? lookup[id] : -1;
  }

  function getPotentialLinePoolForStat(stat, options) {
    const id = String(stat || '').trim();
    const lookup = options && options.potentialLinePoolsByStat || POTENTIAL_LINE_POOLS_BY_STAT;
    return lookup[id] || null;
  }

  function getSkillById(skillId, options) {
    const lookup = options && options.skillsById || SKILLS_BY_ID;
    return lookup[String(skillId || '')] || null;
  }

  function getEquipmentTemplate(itemId, options) {
    const lookup = options && options.equipmentTemplateById || EQUIPMENT_TEMPLATE_BY_ID;
    return lookup[String(itemId || '').trim()] || null;
  }

  function getEquipmentSet(setId, options) {
    const id = String(setId || '').trim();
    const lookup = options && options.equipmentSetsById || EQUIPMENT_SETS_BY_ID;
    return lookup[id] || null;
  }

  function getGearTrait(traitId, options) {
    const id = String(traitId || '').trim();
    const lookup = options && options.gearTraitsById || GEAR_TRAITS_BY_ID;
    return lookup[id] || null;
  }

  function getCardDefinition(card, options) {
    const lookup = options && options.cardDefinitionsById || CARD_DEFINITIONS_BY_ID;
    return card && card.definition || lookup[String(card && card.cardId || '').trim()] || {};
  }

  function createLookupUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      data: settings.data || Data,
      skillsById: settings.skillsById || SKILLS_BY_ID,
      cardDefinitionsById: settings.cardDefinitionsById || CARD_DEFINITIONS_BY_ID,
      gearTraitsById: settings.gearTraitsById || GEAR_TRAITS_BY_ID,
      equipmentSetsById: settings.equipmentSetsById || EQUIPMENT_SETS_BY_ID,
      potentialTierFallback: settings.potentialTierFallback || POTENTIAL_TIER_FALLBACK,
      potentialTiersById: settings.potentialTiersById || POTENTIAL_TIERS_BY_ID,
      potentialTierIndexById: settings.potentialTierIndexById || POTENTIAL_TIER_INDEX_BY_ID,
      potentialLinePoolsByStat: settings.potentialLinePoolsByStat || POTENTIAL_LINE_POOLS_BY_STAT,
      equipmentTemplateById: settings.equipmentTemplateById || EQUIPMENT_TEMPLATE_BY_ID
    });
    return Object.freeze({
      normalizePotentialTierId: (tierId) => normalizePotentialTierId(tierId, helperOptions),
      getPotentialTierDefinition: (tierId) => getPotentialTierDefinition(tierId, helperOptions),
      getPotentialTierRank: (tierId) => getPotentialTierRank(tierId, helperOptions),
      getPotentialLinePoolForStat: (stat) => getPotentialLinePoolForStat(stat, helperOptions),
      getSkillById: (skillId) => getSkillById(skillId, helperOptions),
      getEquipmentTemplate: (itemId) => getEquipmentTemplate(itemId, helperOptions),
      getEquipmentSet: (setId) => getEquipmentSet(setId, helperOptions),
      getGearTrait: (traitId) => getGearTrait(traitId, helperOptions),
      getCardDefinition: (card) => getCardDefinition(card, helperOptions)
    });
  }

  const api = {
    SKILLS_BY_ID,
    SKILLS_BY_OWNER,
    CARD_DEFINITIONS_BY_ID,
    GEAR_TRAITS_BY_ID,
    EQUIPMENT_SETS_BY_ID,
    POTENTIAL_TIER_FALLBACK,
    POTENTIAL_TIERS_BY_ID,
    POTENTIAL_TIER_INDEX_BY_ID,
    POTENTIAL_LINE_POOLS_BY_STAT,
    EQUIPMENT_CATALOG,
    EQUIPMENT_TEMPLATE_BY_ID,
    createSkillsById,
    createSkillsByOwner,
    createCardDefinitionsById,
    createGearTraitsById,
    createEquipmentSetsById,
    getPotentialTierFallback,
    createPotentialTiersById,
    createPotentialTierIndexById,
    createPotentialLinePoolsByStat,
    createEquipmentCatalog,
    createEquipmentTemplateById,
    normalizePotentialTierId,
    getPotentialTierDefinition,
    getPotentialTierRank,
    getPotentialLinePoolForStat,
    getSkillById,
    getEquipmentTemplate,
    getEquipmentSet,
    getGearTrait,
    getCardDefinition,
    createLookupUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.lookups = Object.assign({}, modules.lookups || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
