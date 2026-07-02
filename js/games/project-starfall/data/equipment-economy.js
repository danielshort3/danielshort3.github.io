(function initProjectStarfallDataEquipmentEconomy(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataEquipmentItemAssets = (typeof require === 'function' ? require('./equipment-item-assets.js') : null) || DataModules.equipmentItemAssets || {};
  const DataEquipmentCatalog = (typeof require === 'function' ? require('./equipment-catalog.js') : null) || DataModules.equipmentCatalog || {};

  function createEquipmentEconomyData(options) {
    const settings = options || {};
    const ITEM_ASSETS = settings.ITEM_ASSETS || {};
    const EQUIPMENT_VISUALS = settings.EQUIPMENT_VISUALS || {};
    const createEquipmentItemAssetHelpers = settings.createEquipmentItemAssetHelpers || DataEquipmentItemAssets.createEquipmentItemAssetHelpers;
    const createEquipmentCatalogData = settings.createEquipmentCatalogData || DataEquipmentCatalog.createEquipmentCatalogData;

    const equipmentItemAssetHelpers = createEquipmentItemAssetHelpers({
      ITEM_ASSETS,
      EQUIPMENT_VISUALS
    });
    const attachEquipmentItemAssets = typeof settings.attachEquipmentItemAssets === 'function'
      ? settings.attachEquipmentItemAssets
      : equipmentItemAssetHelpers.attachEquipmentItemAssets;

    const equipmentCatalogData = createEquipmentCatalogData({
      attachEquipmentItemAssets
    });

    return Object.freeze({
      getDefaultEquipmentVisualId: equipmentItemAssetHelpers.getDefaultEquipmentVisualId,
      getEquipmentVisualDefinition: equipmentItemAssetHelpers.getEquipmentVisualDefinition,
      attachEquipmentItemAssets,
      SHOP_ITEMS: equipmentCatalogData.SHOP_ITEMS,
      RANDOM_EQUIPMENT_ITEMS: equipmentCatalogData.RANDOM_EQUIPMENT_ITEMS,
      BOSS_EQUIPMENT_SOURCES: equipmentCatalogData.BOSS_EQUIPMENT_SOURCES,
      DROP_ECONOMY: equipmentCatalogData.DROP_ECONOMY,
      BOSS_EQUIPMENT_ITEMS: equipmentCatalogData.BOSS_EQUIPMENT_ITEMS
    });
  }

  const defaultEquipmentEconomyData = createEquipmentEconomyData();
  const api = Object.assign({
    createEquipmentEconomyData
  }, defaultEquipmentEconomyData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.equipmentEconomy = Object.assign({}, modules.equipmentEconomy || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
