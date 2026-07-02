(function initProjectStarfallDataRewardContent(global) {
  'use strict';

  const DataModules = global.ProjectStarfallDataModules || {};
  const DataRewards = (typeof require === 'function' ? require('./rewards.js') : null) || DataModules.rewards || {};
  const DataQuests = (typeof require === 'function' ? require('./quests.js') : null) || DataModules.quests || {};
  const DataShopVendors = (typeof require === 'function' ? require('./shop-vendors.js') : null) || DataModules.shopVendors || {};
  const DataDailyLogin = (typeof require === 'function' ? require('./daily-login.js') : null) || DataModules.dailyLogin || {};

  function createRewardContentData(options) {
    const settings = options || {};
    const freezeQuestReward = settings.freezeQuestReward || DataRewards.freezeQuestReward;
    const getTownShopVendorId = settings.getTownShopVendorId || DataShopVendors.getTownShopVendorId;
    const createQuestData = settings.createQuestData || DataQuests.createQuestData;
    const createShopVendorData = settings.createShopVendorData || DataShopVendors.createShopVendorData;
    const createDailyLoginData = settings.createDailyLoginData || DataDailyLogin.createDailyLoginData;

    const questData = createQuestData({
      freezeQuestReward
    });
    const shopVendorData = createShopVendorData({
      freezeQuestReward,
      getTownShopVendorId
    });
    const dailyLoginData = createDailyLoginData({
      freezeQuestReward
    });

    return Object.freeze({
      freezeQuestReward,
      QUESTS: questData.QUESTS,
      SHOP_VENDOR_CATALOGS: shopVendorData.SHOP_VENDOR_CATALOGS,
      DAILY_LOGIN_REWARDS: dailyLoginData.DAILY_LOGIN_REWARDS,
      DAILY_LOGIN_MILESTONES: dailyLoginData.DAILY_LOGIN_MILESTONES
    });
  }

  const defaultRewardContentData = createRewardContentData();
  const api = Object.assign({
    createRewardContentData
  }, defaultRewardContentData);

  const modules = global.ProjectStarfallDataModules || {};
  modules.rewardContent = Object.assign({}, modules.rewardContent || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
