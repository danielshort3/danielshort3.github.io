(function initProjectStarfallEngineDailyLogin(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreTime = (typeof require === 'function' ? require('../core/time.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const getDailyLoginDateKey = CoreTime.getDailyLoginDateKey || function getDailyLoginDateKeyFallback(nowMs) {
    const date = new Date(Number.isFinite(Number(nowMs)) ? Number(nowMs) : Date.now());
    const pad = (value) => String(Math.max(0, Math.floor(Number(value || 0) || 0))).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  };
  const getDailyLoginDayDistance = CoreTime.getDailyLoginDayDistance || function getDailyLoginDayDistanceFallback(fromDateKey, toDateKey) {
    const from = Date.parse(String(fromDateKey || ''));
    const to = Date.parse(String(toDateKey || ''));
    if (!Number.isFinite(from) || !Number.isFinite(to)) return 0;
    return Math.round((to - from) / (24 * 60 * 60 * 1000));
  };
  const shiftDailyLoginDateKey = CoreTime.shiftDailyLoginDateKey || function shiftDailyLoginDateKeyFallback(dateKey, deltaDays) {
    const timestamp = Date.parse(String(dateKey || ''));
    if (!Number.isFinite(timestamp)) return '';
    return getDailyLoginDateKey(timestamp + Math.floor(Number(deltaDays || 0) || 0) * 24 * 60 * 60 * 1000);
  };

  function getDailyLoginData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function createDailyLoginState(value, options) {
    const data = getDailyLoginData(options);
    const source = value && typeof value === 'object' ? value : {};
    const rewardCount = Math.max(1, (data.DAILY_LOGIN_REWARDS || []).length || 1);
    const validMilestoneIds = new Set((data.DAILY_LOGIN_MILESTONES || []).map((milestone) => milestone.id));
    const claimedMilestoneIds = Array.isArray(source.claimedMilestoneIds)
      ? source.claimedMilestoneIds.map(normalizeId).filter((id) => validMilestoneIds.has(id))
      : [];
    const lastClaimedDateKey = /^\d{4}-\d{2}-\d{2}$/.test(String(source.lastClaimedDateKey || ''))
      ? String(source.lastClaimedDateKey)
      : '';
    return {
      lastClaimedDateKey,
      lastClaimedAt: Math.max(0, Number(source.lastClaimedAt || 0) || 0),
      totalClaimedDays: Math.max(0, Math.floor(Number(source.totalClaimedDays || 0) || 0)),
      cycleIndex: clamp(Math.floor(Number(source.cycleIndex || 0) || 0), 0, rewardCount - 1),
      streak: Math.max(0, Math.floor(Number(source.streak || 0) || 0)),
      claimedMilestoneIds: Array.from(new Set(claimedMilestoneIds))
    };
  }

  function createDailyLoginClaimBundle(state, player, options) {
    const data = getDailyLoginData(options);
    const settings = options || {};
    const source = state && typeof state === 'object' ? state : createDailyLoginState(null, options);
    const rewards = data.DAILY_LOGIN_REWARDS || [];
    const milestones = data.DAILY_LOGIN_MILESTONES || [];
    const mergeRewards = settings.mergeRewards || function mergeRewardsFallback(target, reward) {
      return Object.assign(target && typeof target === 'object' ? target : {}, reward || {});
    };
    const getInventoryBlockReason = settings.getInventoryBlockReason || function getInventoryBlockReasonFallback() {
      return '';
    };
    const todayKey = getDailyLoginDateKey(settings.nowMs);
    const lastKey = source.lastClaimedDateKey || '';
    const dayDistance = lastKey ? getDailyLoginDayDistance(lastKey, todayKey) : 1;
    const hasClass = !!(player && player.classId);
    const claimable = hasClass && (!lastKey || dayDistance > 0);
    const claimedToday = !!lastKey && dayDistance === 0;
    const rewardIndex = clamp(Math.floor(Number(source.cycleIndex || 0) || 0), 0, Math.max(0, rewards.length - 1));
    const dailyReward = rewards[rewardIndex] || rewards[0] || null;
    const nextTotal = Math.max(0, Math.floor(Number(source.totalClaimedDays || 0) || 0)) + (claimable ? 1 : 0);
    const claimedMilestones = new Set(source.claimedMilestoneIds || []);
    const claimableMilestones = claimable
      ? milestones.filter((milestone) => milestone && Number(milestone.days || 0) <= nextTotal && !claimedMilestones.has(milestone.id))
      : [];
    const combinedReward = [dailyReward].concat(claimableMilestones)
      .filter(Boolean)
      .reduce((reward, entry) => mergeRewards(reward, entry.reward || {}), {});
    let disabledReason = '';
    if (!hasClass) disabledReason = 'Choose a class to claim daily rewards.';
    else if (claimedToday) disabledReason = 'Today\'s reward is already claimed.';
    else if (lastKey && dayDistance < 0) disabledReason = 'Daily rewards unlock after the saved claim date.';
    if (!disabledReason && claimable) disabledReason = getInventoryBlockReason(combinedReward);
    const nextClaimDateKey = claimable
      ? todayKey
      : lastKey && dayDistance < 0
        ? shiftDailyLoginDateKey(lastKey, 1)
        : shiftDailyLoginDateKey(todayKey, 1);
    const nextStreak = claimable
      ? !lastKey || dayDistance > 1 ? 1 : Math.max(0, Number(source.streak || 0)) + 1
      : Math.max(0, Number(source.streak || 0));
    return {
      state: source,
      todayKey,
      lastClaimedDateKey: lastKey,
      dayDistance,
      claimable: claimable && !disabledReason,
      baseClaimable: claimable,
      claimedToday,
      disabledReason,
      nextClaimDateKey,
      nextTotalClaimedDays: nextTotal,
      nextStreak,
      rewardIndex,
      dailyReward,
      claimableMilestones,
      combinedReward
    };
  }

  function createDailyLoginSnapshot(bundle, options) {
    const data = getDailyLoginData(options);
    const settings = options || {};
    const formatRewardSummary = settings.formatRewardSummary || function formatRewardSummaryFallback() {
      return '';
    };
    const source = bundle || createDailyLoginClaimBundle(null, null, options);
    const state = source.state || createDailyLoginState(null, options);
    const rewards = data.DAILY_LOGIN_REWARDS || [];
    const milestones = data.DAILY_LOGIN_MILESTONES || [];
    const totalClaimedDays = Math.max(0, Math.floor(Number(state.totalClaimedDays || 0) || 0));
    const claimedMilestones = new Set(state.claimedMilestoneIds || []);
    const currentCycleIndex = source.rewardIndex;
    const upcomingMilestone = milestones.find((milestone) => milestone && !claimedMilestones.has(milestone.id) && Number(milestone.days || 0) > totalClaimedDays) || null;
    return {
      todayKey: source.todayKey,
      lastClaimedDateKey: source.lastClaimedDateKey,
      lastClaimedAt: Math.max(0, Number(state.lastClaimedAt || 0) || 0),
      totalClaimedDays,
      streak: Math.max(0, Math.floor(Number(state.streak || 0) || 0)),
      cycleIndex: currentCycleIndex,
      cycleDay: currentCycleIndex + 1,
      cycleLength: Math.max(1, rewards.length || 1),
      claimable: !!source.claimable,
      claimedToday: !!source.claimedToday,
      disabledReason: source.disabledReason || '',
      nextClaimDateKey: source.nextClaimDateKey,
      currentReward: source.dailyReward ? Object.assign({}, source.dailyReward, {
        rewardSummary: formatRewardSummary(source.dailyReward.reward || {})
      }) : null,
      claimRewardSummary: formatRewardSummary(source.combinedReward || {}),
      claimMilestones: source.claimableMilestones.map((milestone) => Object.assign({}, milestone, {
        rewardSummary: formatRewardSummary(milestone.reward || {})
      })),
      cycleRewards: rewards.map((reward, index) => Object.assign({}, reward, {
        index,
        active: index === currentCycleIndex,
        claimedInCycle: index < currentCycleIndex && !source.claimable,
        rewardSummary: formatRewardSummary(reward.reward || {})
      })),
      milestones: milestones.map((milestone) => {
        const days = Math.max(1, Math.floor(Number(milestone.days || 1) || 1));
        const claimed = claimedMilestones.has(milestone.id);
        const eligible = totalClaimedDays >= days;
        return Object.assign({}, milestone, {
          claimed,
          claimable: eligible && !claimed,
          progress: Math.min(days, totalClaimedDays),
          goal: days,
          progressRatio: clamp(totalClaimedDays / days, 0, 1),
          rewardSummary: formatRewardSummary(milestone.reward || {})
        });
      }),
      nextMilestone: upcomingMilestone ? Object.assign({}, upcomingMilestone, {
        progress: Math.min(Math.max(1, Number(upcomingMilestone.days || 1)), totalClaimedDays),
        goal: Math.max(1, Number(upcomingMilestone.days || 1)),
        rewardSummary: formatRewardSummary(upcomingMilestone.reward || {})
      }) : null,
      accountWideDeferred: true
    };
  }

  const api = {
    createDailyLoginState,
    createDailyLoginClaimBundle,
    createDailyLoginSnapshot
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.dailyLogin = Object.assign({}, modules.dailyLogin || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
