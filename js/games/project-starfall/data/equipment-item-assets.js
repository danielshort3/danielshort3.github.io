(function initProjectStarfallDataEquipmentItemAssets(global) {
  'use strict';

  function createEquipmentItemAssetHelpers(options) {
    const settings = options || {};
    const ITEM_ASSETS = settings.ITEM_ASSETS || {};
    const EQUIPMENT_VISUALS = settings.EQUIPMENT_VISUALS || {};

    function getDefaultEquipmentVisualId(item) {
      const slot = String(item && item.slot || '').trim();
      if (slot === 'head') return 'fieldguard_helm';
      if (slot === 'gloves') return 'trailwoven_gloves';
      if (slot === 'amulet') return 'plain_ring';
      return '';
    }

    function getEquipmentVisualDefinition(item) {
      const candidates = [
        item && item.id,
        item && item.visualId,
        getDefaultEquipmentVisualId(item)
      ];
      for (const candidate of candidates) {
        const id = String(candidate || '').trim();
        if (id && EQUIPMENT_VISUALS[id]) return EQUIPMENT_VISUALS[id];
      }
      return null;
    }

    function attachEquipmentItemAssets(item) {
      const visual = getEquipmentVisualDefinition(item);
      const assetIds = [
        item.id,
        item.assetId,
        visual && visual.assetId,
        item.visualId
      ].filter(Boolean);
      const asset = assetIds.reduce((resolved, assetId) => resolved || ITEM_ASSETS[assetId] || '', '');
      return Object.freeze(Object.assign({}, item, {
        asset,
        visualId: visual ? visual.id : ''
      }));
    }

    return Object.freeze({
      getDefaultEquipmentVisualId,
      getEquipmentVisualDefinition,
      attachEquipmentItemAssets
    });
  }

  const defaultEquipmentItemAssetHelpers = createEquipmentItemAssetHelpers();

  const api = Object.freeze({
    createEquipmentItemAssetHelpers,
    getDefaultEquipmentVisualId: defaultEquipmentItemAssetHelpers.getDefaultEquipmentVisualId,
    getEquipmentVisualDefinition: defaultEquipmentItemAssetHelpers.getEquipmentVisualDefinition,
    attachEquipmentItemAssets: defaultEquipmentItemAssetHelpers.attachEquipmentItemAssets
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentItemAssets = api;
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
