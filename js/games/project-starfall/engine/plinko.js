(function initProjectStarfallEnginePlinko(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    const source = Array.isArray(items) ? items : [];
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] && source[index].id === id) return source[index];
    }
    return null;
  };

  const PLINKO_BASIC_BALL_ID = 'plinko_ball_basic';
  const PLINKO_DEFAULT_BOUNCE_COUNT = 8;
  const PLINKO_PRIZE_TRAY_LIMIT = 60;
  const PLINKO_SLOT_COUPON_IDS = Object.freeze(['equipment_slot_coupon', 'usable_slot_coupon', 'etc_slot_coupon', 'card_slot_coupon']);
  const PLINKO_RATE_COUPON_IDS = Object.freeze(['xp_coupon_1_2_1h', 'xp_coupon_1_5_1h', 'xp_coupon_2_0_1h', 'drop_coupon_1_2_1h', 'drop_coupon_1_5_1h', 'drop_coupon_2_0_1h']);
  const PLINKO_POTENTIAL_CUBE_ID = 'potential_cube';
  const PLINKO_PRESERVATION_CUBE_ID = 'preservation_cube';

  function clonePlain(value) {
    if (!value || typeof value !== 'object') return value;
    return JSON.parse(JSON.stringify(value));
  }

  function getPlinkoData(options) {
    const settings = options || {};
    return settings.data || global.ProjectStarfallData || {};
  }

  function getPlinkoBounceCount(options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const fallback = Math.max(1, Number(settings.defaultBounceCount || PLINKO_DEFAULT_BOUNCE_COUNT));
    return Math.max(1, Number(settings.bounceCount || data.PLINKO_BOUNCE_COUNT || fallback));
  }

  function getPlinkoBall(ballId, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const balls = data.PLINKO_BALLS || [];
    const basicBallId = normalizeId(settings.basicBallId || PLINKO_BASIC_BALL_ID);
    return getById(balls, normalizeId(ballId)) || getById(balls, basicBallId) || balls[0] || null;
  }

  function getPlinkoSnapshotBall(state, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const getBall = typeof settings.getPlinkoBall === 'function'
      ? settings.getPlinkoBall
      : (ballId) => getPlinkoBall(ballId, settings);
    const getBallCount = typeof settings.getPlinkoBallCount === 'function'
      ? settings.getPlinkoBallCount
      : function getPlinkoBallCountFallback() {
          return 0;
        };
    const selectedBall = getBall(state && state.selectedBallId);
    if (selectedBall) return selectedBall;
    const ownedBall = (data.PLINKO_BALLS || []).find((ball) => ball && getBallCount(ball.id) > 0);
    return ownedBall || selectedBall || getBall(settings.basicBallId || PLINKO_BASIC_BALL_ID);
  }

  function getPlinkoActiveDropLimit(options) {
    const data = getPlinkoData(options);
    return Math.max(0, Math.floor(Number(data.PLINKO_ACTIVE_DROP_LIMIT || 0) || 0));
  }

  function getPlinkoPrizeTrayLimit(options) {
    const settings = options || {};
    return Math.max(0, Math.floor(Number(settings.prizeTrayLimit == null ? PLINKO_PRIZE_TRAY_LIMIT : settings.prizeTrayLimit) || 0));
  }

  function getPlinkoPityTarget(ball, options) {
    const data = getPlinkoData(options);
    return Math.max(1, Math.floor(Number(ball && ball.pityTarget || data.PLINKO_PITY_TARGET || 100) || 100));
  }

  function getPlinkoPityForBall(ballId, state, options) {
    const settings = options || {};
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const id = normalizeId(ballId) || normalizeId(settings.basicBallId || PLINKO_BASIC_BALL_ID);
    return Math.max(0, Math.floor(Number(plinkoState.pityByBall && plinkoState.pityByBall[id] || 0) || 0));
  }

  function getPlinkoBallCount(ballId, consumables) {
    const id = normalizeId(ballId);
    const source = consumables && typeof consumables === 'object' ? consumables : {};
    return id ? Math.max(0, Math.floor(Number(source[id] || 0) || 0)) : 0;
  }

  function getPlinkoReservedPrizeTrayCount(state) {
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    return (plinkoState.prizeTray || []).length + (plinkoState.pendingDrops || []).length;
  }

  function isPlinkoPrizeTrayFull(state, options) {
    const settings = options || {};
    const getLimit = typeof settings.getPlinkoPrizeTrayLimit === 'function'
      ? settings.getPlinkoPrizeTrayLimit
      : function getPlinkoPrizeTrayLimitFallback() {
          return settings.prizeTrayLimit == null ? PLINKO_PRIZE_TRAY_LIMIT : settings.prizeTrayLimit;
        };
    return getPlinkoReservedPrizeTrayCount(state) >= getLimit();
  }

  function createPlinkoBallSnapshot(ball, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const getBallCount = typeof settings.getPlinkoBallCount === 'function'
      ? settings.getPlinkoBallCount
      : function getPlinkoBallCountFallback(ballId) {
          return getPlinkoBallCount(ballId, settings.consumables || settings.ballCounts);
        };
    const getPityForBall = typeof settings.getPlinkoPityForBall === 'function'
      ? settings.getPlinkoPityForBall
      : (ballId) => getPlinkoPityForBall(ballId, settings.state, settings);
    const getPityTarget = typeof settings.getPlinkoPityTarget === 'function'
      ? settings.getPlinkoPityTarget
      : (targetBall) => getPlinkoPityTarget(targetBall, settings);
    const getReadiness = typeof settings.getPlinkoDropReadiness === 'function'
      ? settings.getPlinkoDropReadiness
      : (ballId) => getPlinkoDropReadiness(ballId, settings);
    const getAddCapacity = typeof settings.getStackableInventoryAddCapacityFromUsed === 'function'
      ? settings.getStackableInventoryAddCapacityFromUsed
      : function getStackableInventoryAddCapacityFromUsedFallback() {
          return Number.POSITIVE_INFINITY;
        };
    const itemAssets = settings.itemAssets || data.ITEM_ASSETS || {};
    const count = getBallCount(ball.id);
    const cost = Math.max(0, Math.floor(Number(ball.cost || 0) || 0));
    const maxBuyQuantity = cost > 0 ? Math.floor(Math.max(0, Number(settings.currency || 0)) / cost) : 999;
    const readiness = getReadiness(ball.id) || {};
    const canBuyOne = maxBuyQuantity > 0 && getAddCapacity('usable', ball.id, settings.usableUsedSlots, settings.usableCapacity) >= 1;
    return Object.assign({}, ball, {
      count,
      selected: settings.selectedBall && settings.selectedBall.id === ball.id,
      pity: getPityForBall(ball.id),
      pityTarget: getPityTarget(ball),
      canDrop: !!readiness.canDrop,
      disabledReason: readiness.disabledReason || '',
      canBuy: canBuyOne,
      maxBuyQuantity: canBuyOne ? Math.max(1, Math.min(999, maxBuyQuantity)) : 0,
      asset: itemAssets[ball.id] || ''
    });
  }

  function createPlinkoBallSnapshots(balls, options) {
    return (Array.isArray(balls) ? balls : []).map((ball) => createPlinkoBallSnapshot(ball, options));
  }

  function createPlinkoBallSelectionPlan(ball, state) {
    if (!ball) return null;
    const selectedBallId = normalizeId(ball.id);
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const changed = plinkoState.selectedBallId !== selectedBallId;
    return {
      selectedBallId,
      changed,
      emitUiChange: changed,
      emitDomains: ['shop', 'inventory'],
      emitReason: 'plinkoSelect',
      persist: true
    };
  }

  function createPlinkoBallPurchasePlan(ball, quantity) {
    if (!ball) return null;
    const amount = Math.max(1, Math.min(999, Math.floor(Number(quantity || 1) || 1)));
    const unitCost = Math.max(0, Math.floor(Number(ball.cost || 0) || 0));
    const cost = unitCost * amount;
    return {
      ballId: normalizeId(ball.id),
      selectedBallId: normalizeId(ball.id),
      amount,
      unitCost,
      cost,
      toastMessage: `Bought ${amount} ${ball.name}${amount === 1 ? '' : 's'}.`,
      toastOptions: { noEmit: true },
      addItemOptions: { source: 'plinko-shop', quietDebug: true },
      currencyPatch: {
        type: 'currency-count',
        currencyId: 'coins',
        delta: -cost,
        source: 'plinko-shop'
      },
      emitDomains: ['hud', 'shop', 'inventory'],
      emitReason: 'plinkoBuy',
      persist: true
    };
  }

  function createPlinkoBallPurchaseStackPatchOptions(patch) {
    return {
      domains: patch && patch.layoutChanged ? ['shop', 'inventory', 'pet'] : ['shop'],
      reason: 'plinkoBuy',
      persist: true,
      coalesceFullRefresh: true
    };
  }

  function normalizePlinkoSegment(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const bounceLimit = getPlinkoBounceCount(options);
    return {
      boardId: normalizeId(source.boardId || source.tier),
      boardTitle: String(source.boardTitle || ''),
      boardStage: String(source.boardStage || ''),
      path: Array.isArray(source.path) ? source.path.map((step) => step === 'R' ? 'R' : 'L').slice(0, bounceLimit) : [],
      slotCount: Math.max(0, Math.floor(Number(source.slotCount || 0) || 0)),
      slotId: normalizeId(source.slotId),
      slotIndex: Math.max(0, Math.floor(Number(source.slotIndex || 0) || 0)),
      slotLabel: String(source.slotLabel || ''),
      teleport: !!source.teleport,
      nextBoardId: normalizeId(source.nextBoardId),
      jackpot: !!source.jackpot,
      reward: source.reward && typeof source.reward === 'object' ? clonePlain(source.reward) : {},
      summary: String(source.summary || ''),
      durationMs: Math.max(1, Math.floor(Number(source.durationMs || 1800) || 1800))
    };
  }

  function normalizePlinkoRecentReward(value, options) {
    const source = value && typeof value === 'object' ? value : {};
    const segments = Array.isArray(source.segments)
      ? source.segments.map((segment) => normalizePlinkoSegment(segment, options)).filter((segment) => segment.boardId)
      : [];
    return {
      id: normalizeId(source.id),
      ballId: normalizeId(source.ballId),
      ballName: String(source.ballName || ''),
      boardId: normalizeId(source.boardId || (segments.length ? segments[segments.length - 1].boardId : '')),
      boardTitle: String(source.boardTitle || (segments.length ? segments[segments.length - 1].boardTitle : '')),
      boardStage: String(source.boardStage || (segments.length ? segments[segments.length - 1].boardStage : '')),
      slotId: normalizeId(source.slotId),
      slotIndex: Math.max(0, Math.floor(Number(source.slotIndex || 0) || 0)),
      slotLabel: String(source.slotLabel || ''),
      summary: String(source.summary || ''),
      jackpot: !!source.jackpot,
      pityJackpot: !!source.pityJackpot,
      teleportCount: Math.max(0, Math.floor(Number(source.teleportCount || segments.filter((segment) => segment.teleport).length) || 0)),
      inventoryFallback: !!source.inventoryFallback,
      storedInPrizeTray: !!source.storedInPrizeTray,
      segments,
      reward: source.reward && typeof source.reward === 'object' ? clonePlain(source.reward) : {},
      createdAt: Number(source.createdAt || 0)
    };
  }

  function normalizePlinkoPrizeTrayEntry(value, index, options) {
    const settings = options || {};
    const entry = normalizePlinkoRecentReward(value, settings);
    if (!entry.reward || !Object.keys(entry.reward).length) return null;
    entry.id = entry.id || `plinko_prize_${Math.max(0, Math.floor(Number(index || 0) || 0))}_${entry.slotId || 'reward'}`;
    entry.storedInPrizeTray = true;
    entry.createdAt = entry.createdAt || (typeof settings.now === 'function' ? settings.now() : Date.now());
    return entry;
  }

  function normalizePlinkoPendingDrop(value, options) {
    const settings = options || {};
    const source = value && typeof value === 'object' ? value : {};
    const id = normalizeId(source.id);
    const ball = getPlinkoBall(source.ballId, settings);
    if (!id || !ball) return null;
    let segments = Array.isArray(source.segments)
      ? source.segments.map((segment) => normalizePlinkoSegment(segment, settings)).filter((segment) => segment.boardId)
      : [];
    if (!segments.length) {
      segments = [normalizePlinkoSegment({
        boardId: source.boardId || source.tier || ball.plinkoTier || 'basic',
        boardTitle: source.boardTitle || '',
        boardStage: source.boardStage || '',
        path: source.path,
        slotId: source.slotId,
        slotIndex: source.slotIndex,
        slotLabel: source.slotLabel,
        jackpot: source.jackpot,
        reward: source.reward,
        summary: source.summary,
        durationMs: source.durationMs
      }, settings)].filter((segment) => segment.boardId);
    }
    const finalSegment = segments[segments.length - 1] || {};
    const durationMs = Math.max(1, Math.floor(Number(source.durationMs || segments.reduce((sum, segment) => sum + Math.max(1, Number(segment.durationMs || 0) || 0), 0) || 1800) || 1800));
    const normalizedReward = source.reward && typeof source.reward === 'object' && Object.keys(source.reward).length
      ? clonePlain(source.reward)
      : clonePlain(finalSegment.reward || {});
    const now = typeof settings.now === 'function' ? settings.now : Date.now;
    return {
      id,
      ballId: ball.id,
      ballName: String(source.ballName || ball.name || ''),
      ballIcon: String(source.ballIcon || ball.icon || ''),
      tier: normalizeId(source.tier || ball.plinkoTier || 'basic'),
      boardId: normalizeId(source.boardId || finalSegment.boardId),
      boardTitle: String(source.boardTitle || finalSegment.boardTitle || ''),
      boardStage: String(source.boardStage || finalSegment.boardStage || ''),
      path: Array.isArray(source.path) ? source.path.map((step) => step === 'R' ? 'R' : 'L').slice(0, getPlinkoBounceCount(settings)) : (finalSegment.path || []),
      slotId: normalizeId(source.slotId || finalSegment.slotId),
      slotIndex: Math.max(0, Math.floor(Number(source.slotIndex != null ? source.slotIndex : finalSegment.slotIndex || 0) || 0)),
      slotLabel: String(source.slotLabel || finalSegment.slotLabel || ''),
      reward: normalizedReward,
      summary: String(source.summary || ''),
      jackpot: !!source.jackpot,
      pityJackpot: !!source.pityJackpot,
      teleportCount: Math.max(0, Math.floor(Number(source.teleportCount || segments.filter((segment) => segment.teleport).length) || 0)),
      segments,
      pityBefore: Math.max(0, Math.floor(Number(source.pityBefore || 0) || 0)),
      pityAfter: Math.max(0, Math.floor(Number(source.pityAfter || 0) || 0)),
      createdAt: Math.max(1, Number(source.createdAt || now()) || now()),
      resolveAt: Math.max(1, Number(source.resolveAt || 0) || 0),
      durationMs
    };
  }

  function normalizePlinkoSlotOverrides(value) {
    const source = value && typeof value === 'object' ? value : {};
    const overrides = {};
    Object.entries(source).forEach(([key, reward]) => {
      const id = normalizeId(key);
      if (!id || !reward || typeof reward !== 'object' || !Object.keys(reward).length) return;
      overrides[id] = clonePlain(reward);
    });
    return overrides;
  }

  function createPlinkoState(value, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const source = value && typeof value === 'object' ? value : {};
    const selectedBall = getPlinkoBall(source.selectedBallId, settings);
    const activeDropLimit = Math.max(0, Math.floor(Number(settings.activeDropLimit == null ? data.PLINKO_ACTIVE_DROP_LIMIT || 0 : settings.activeDropLimit) || 0));
    const prizeTrayLimit = Math.max(0, Math.floor(Number(settings.prizeTrayLimit == null ? PLINKO_PRIZE_TRAY_LIMIT : settings.prizeTrayLimit) || 0));
    const pityByBall = {};
    (data.PLINKO_BALLS || []).forEach((ball) => {
      const legacySelected = selectedBall && selectedBall.id === ball.id ? source.pity : 0;
      pityByBall[ball.id] = clamp(Math.floor(Number(source.pityByBall && source.pityByBall[ball.id] || legacySelected || 0) || 0), 0, 99999);
    });
    return {
      selectedBallId: selectedBall ? selectedBall.id : normalizeId(settings.basicBallId || PLINKO_BASIC_BALL_ID),
      pityByBall,
      totalDrops: Math.max(0, Math.floor(Number(source.totalDrops || 0) || 0)),
      pendingDrops: Array.isArray(source.pendingDrops)
        ? (activeDropLimit > 0
          ? source.pendingDrops.map((drop) => normalizePlinkoPendingDrop(drop, settings)).filter(Boolean).slice(-activeDropLimit)
          : source.pendingDrops.map((drop) => normalizePlinkoPendingDrop(drop, settings)).filter(Boolean))
        : [],
      prizeTray: Array.isArray(source.prizeTray)
        ? source.prizeTray.map((entry, index) => normalizePlinkoPrizeTrayEntry(entry, index, settings)).filter(Boolean).slice(0, prizeTrayLimit)
        : [],
      lastRewards: Array.isArray(source.lastRewards)
        ? source.lastRewards.slice(0, 6).map((entry) => normalizePlinkoRecentReward(entry, settings))
        : [],
      slotOverrides: normalizePlinkoSlotOverrides(source.slotOverrides)
    };
  }

  function getPlinkoRewardSignature(reward) {
    return JSON.stringify(reward || {});
  }

  function getPlinkoSnapshotEntrySignature(entry, options) {
    if (!entry) return '';
    const settings = options || {};
    const getRewardSignature = typeof settings.getPlinkoRewardSignature === 'function'
      ? settings.getPlinkoRewardSignature
      : getPlinkoRewardSignature;
    return [
      normalizeId(entry.id),
      normalizeId(entry.ballId),
      normalizeId(entry.boardId),
      normalizeId(entry.slotId),
      Math.floor(Number(entry.slotIndex || 0) || 0),
      Math.floor(Number(entry.createdAt || 0) || 0),
      Math.floor(Number(entry.resolveAt || 0) || 0),
      getRewardSignature(entry.reward || {})
    ].join(':');
  }

  function getPlinkoSnapshotListSignature(entries, options) {
    return (entries || []).map((entry) => getPlinkoSnapshotEntrySignature(entry, options)).join('|');
  }

  function getPlinkoSlotOverrideKey(boardId, slotId) {
    return `${normalizeId(boardId)}:${normalizeId(slotId)}`;
  }

  function getPlinkoSlotOverrideSignature(state, options) {
    const settings = options || {};
    const getRewardSignature = typeof settings.getPlinkoRewardSignature === 'function'
      ? settings.getPlinkoRewardSignature
      : getPlinkoRewardSignature;
    const overrides = state && state.slotOverrides && typeof state.slotOverrides === 'object' ? state.slotOverrides : {};
    const keys = Object.keys(overrides).sort();
    if (!keys.length) return '';
    return keys.map((key) => `${normalizeId(key)}:${getRewardSignature(overrides[key])}`).join('|');
  }

  function formatPlinkoSlotOdds(slot, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const probability = Math.max(0, Number(slot && slot.probability || 0) || 0);
    const denominator = Math.max(1, Number(slot && slot.probabilityDenominator || settings.denominator || data.PLINKO_SLOT_DENOMINATOR || 256) || 256);
    return `${probability}/${denominator} (${(probability / denominator * 100).toFixed(2)}%)`;
  }

  function mergePlinkoReward(target, source, options) {
    const settings = options || {};
    const reward = source || {};
    const output = target || {};
    const getCardConfigs = typeof settings.getRewardCardConfigs === 'function'
      ? settings.getRewardCardConfigs
      : function getRewardCardConfigsFallback(sourceReward) {
          return Object.entries(sourceReward && sourceReward.cards || {}).map(([id, quantity]) => ({
            cardId: normalizeId(id),
            quantity
          })).filter((card) => card.cardId);
        };
    const getItemConfigs = typeof settings.getRewardItemConfigs === 'function'
      ? settings.getRewardItemConfigs
      : function getRewardItemConfigsFallback(sourceReward) {
          return Array.isArray(sourceReward && sourceReward.items) ? sourceReward.items : [];
        };
    ['xp', 'currency', 'starTokens', 'statUpgradePoints'].forEach((key) => {
      const value = Number(reward[key] || 0);
      if (value > 0) output[key] = Number(output[key] || 0) + value;
    });
    if (reward.materials) {
      output.materials = Object.assign({}, output.materials || {});
      Object.entries(reward.materials).forEach(([id, amount]) => {
        const value = Number(amount || 0);
        if (value > 0) output.materials[id] = Number(output.materials[id] || 0) + value;
      });
    }
    if (reward.consumables) {
      output.consumables = Object.assign({}, output.consumables || {});
      Object.entries(reward.consumables).forEach(([id, amount]) => {
        const value = Number(amount || 0);
        if (value > 0) output.consumables[id] = Number(output.consumables[id] || 0) + value;
      });
    }
    if (reward.cards) {
      output.cards = Object.assign({}, output.cards || {});
      output.cardConfigs = Array.isArray(output.cardConfigs) ? output.cardConfigs.slice() : [];
      getCardConfigs(reward).forEach((card) => {
        const id = normalizeId(card.cardId || card.id);
        const value = Math.max(1, Math.floor(Number(card.quantity || card.amount || 1) || 1));
        if (id) output.cards[id] = Number(output.cards[id] || 0) + value;
        if (id) output.cardConfigs.push(Object.assign({}, card, { cardId: id }));
      });
    }
    const itemConfigs = getItemConfigs(reward);
    if (itemConfigs.length) {
      output.items = (Array.isArray(output.items) ? output.items.slice() : []).concat(itemConfigs.map((item) => Object.assign({}, item)));
    }
    return output;
  }

  function getPlinkoRotationSeed(boardId, slotId, options) {
    const settings = options || {};
    const state = settings.state || {};
    const text = `${normalizeId(boardId)}:${normalizeId(slotId)}`;
    const hash = text.split('').reduce((sum, char) => sum + char.charCodeAt(0), 0);
    return hash + Math.max(0, Math.floor(Number(state.totalDrops || 0) || 0)) + (state.lastRewards || []).length;
  }

  function getPlinkoRarityFromTier(tier, fallback, options) {
    const settings = options || {};
    const rarityOrder = Array.isArray(settings.rarityOrder) ? settings.rarityOrder : ['Common', 'Uncommon', 'Rare', 'Epic', 'Relic'];
    const key = String(tier || '').toLowerCase();
    if (key === 'basic') return fallback || 'Uncommon';
    if (key === 'uncommon') return 'Uncommon';
    if (key === 'rare') return 'Rare';
    if (key === 'epic') return 'Epic';
    if (key === 'relic') return 'Relic';
    return rarityOrder.includes(fallback) ? fallback : 'Rare';
  }

  function pickPlinkoReplacement(candidates, currentId, seed) {
    const current = normalizeId(currentId);
    const options = (candidates || []).filter((candidate) => candidate && candidate.id && candidate.id !== current);
    if (options.length) return options[Math.abs(seed) % options.length];
    const fallback = (candidates || []).filter((candidate) => candidate && candidate.id);
    return fallback.length ? fallback[Math.abs(seed) % fallback.length] : null;
  }

  function getRewardItemConfigs(reward, options) {
    const settings = options || {};
    if (typeof settings.getRewardItemConfigs === 'function') return settings.getRewardItemConfigs(reward);
    const items = reward && reward.items;
    if (!items) return [];
    if (Array.isArray(items)) return items.filter(Boolean);
    return Object.entries(items).map(([itemId, value]) => {
      if (value && typeof value === 'object') return Object.assign({ itemId }, value);
      return { itemId, amount: value };
    });
  }

  function getRewardCardConfigs(reward, options) {
    const settings = options || {};
    if (typeof settings.getRewardCardConfigs === 'function') return settings.getRewardCardConfigs(reward);
    const cards = reward && reward.cards;
    if (!cards) return [];
    if (Array.isArray(cards)) return cards.filter(Boolean);
    return Object.entries(cards).map(([cardId, value]) => {
      if (value && typeof value === 'object') return Object.assign({ cardId }, value);
      return { cardId, amount: value };
    });
  }

  function getPlinkoBoards(options) {
    const data = getPlinkoData(options);
    return data.PLINKO_BOARDS || data.PLINKO_REWARD_TABLES || {};
  }

  function getPlinkoBoard(ballOrTier, options) {
    const boards = getPlinkoBoards(options);
    if (typeof ballOrTier === 'string') {
      const id = normalizeId(ballOrTier);
      if (boards[id]) return boards[id];
    }
    const tier = typeof ballOrTier === 'string'
      ? normalizeId(ballOrTier)
      : normalizeId(ballOrTier && (ballOrTier.boardId || ballOrTier.id && boards[ballOrTier.id] && ballOrTier.id || ballOrTier.plinkoTier || ballOrTier.tier)) || 'basic';
    return boards[tier] || boards.basic || null;
  }

  function getPlinkoBoardChain(ballOrTier, options) {
    const settings = options || {};
    const getBoard = typeof settings.getPlinkoBoard === 'function'
      ? settings.getPlinkoBoard
      : (target) => getPlinkoBoard(target, settings);
    const chain = [];
    const seen = new Set();
    let board = getBoard(ballOrTier);
    while (board && board.id && !seen.has(board.id) && chain.length < 4) {
      chain.push(board);
      seen.add(board.id);
      const gate = (board.slots || []).find((slot) => slot && slot.teleport && slot.nextBoardId);
      board = gate ? getBoard(gate.nextBoardId) : null;
    }
    return chain;
  }

  function createPlinkoResolvedSlot(board, slot, context, options) {
    const settings = options || {};
    const state = context && context.plinko || settings.state || {};
    const getOverrideKey = typeof settings.getPlinkoSlotOverrideKey === 'function'
      ? settings.getPlinkoSlotOverrideKey
      : getPlinkoSlotOverrideKey;
    const boardId = normalizeId(board && board.id || slot && slot.boardId);
    const slotId = normalizeId(slot && slot.id);
    const overrideKey = getOverrideKey(boardId, slotId);
    const override = state.slotOverrides && state.slotOverrides[overrideKey];
    const reward = slot && slot.teleport ? {} : clonePlain(override || slot && slot.reward || {});
    return Object.assign({}, slot || {}, {
      boardId,
      boardTitle: board && board.title || '',
      boardStage: board && board.stage || '',
      reward,
      rewardRotated: !!override
    });
  }

  function getPlinkoBoardSlots(ballOrBoardId, context, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const getBoard = typeof settings.getPlinkoBoard === 'function'
      ? settings.getPlinkoBoard
      : (target) => getPlinkoBoard(target, settings);
    const createResolvedSlot = typeof settings.createPlinkoResolvedSlot === 'function'
      ? settings.createPlinkoResolvedSlot
      : (board, slot, targetContext) => createPlinkoResolvedSlot(board, slot, targetContext, settings);
    const board = getBoard(ballOrBoardId);
    if (board && Array.isArray(board.slots) && board.slots.length) {
      return board.slots.map((slot) => createResolvedSlot(board, slot, context));
    }
    return data.PLINKO_BOARD_SLOTS || [];
  }

  function createPlinkoBoardChainSnapshotKey(selectedBall, boardChain, state, options) {
    const settings = options || {};
    const getOverrideSignature = typeof settings.getPlinkoSlotOverrideSignature === 'function'
      ? settings.getPlinkoSlotOverrideSignature
      : getPlinkoSlotOverrideSignature;
    const chainKey = (boardChain || []).map((board) => [
      normalizeId(board && (board.id || board.tier)),
      normalizeId(board && board.stage),
      Array.isArray(board && board.slots) ? board.slots.length : 0
    ].join(':')).join(',');
    return [
      normalizeId(selectedBall && selectedBall.id),
      normalizeId(selectedBall && selectedBall.plinkoTier),
      chainKey,
      getOverrideSignature(state || {})
    ].join('|');
  }

  function createPlinkoBoardChainSnapshot(selectedBoard, boardChain, context, options) {
    const settings = options || {};
    const getBoardSnapshot = typeof settings.getPlinkoBoardSnapshot === 'function'
      ? settings.getPlinkoBoardSnapshot
      : (board, targetContext) => createPlinkoBoardSnapshot(board, targetContext, settings);
    const chain = Array.isArray(boardChain) ? boardChain : [];
    const boardSnapshots = chain.map((board) => getBoardSnapshot(board, context));
    const boards = boardSnapshots.reduce((output, board) => {
      output[board.id] = board;
      return output;
    }, {});
    const selectedBoardSnapshot = selectedBoard && boards[selectedBoard.id]
      ? boards[selectedBoard.id]
      : getBoardSnapshot(selectedBoard, context);
    return {
      selectedBoard,
      boardChain: chain,
      boardSnapshots,
      boards,
      selectedBoardSnapshot
    };
  }

  function getPlinkoBoardSnapshotCacheKey(board, context, options) {
    const settings = options || {};
    const getOverrideKey = typeof settings.getPlinkoSlotOverrideKey === 'function'
      ? settings.getPlinkoSlotOverrideKey
      : getPlinkoSlotOverrideKey;
    const getRewardSignature = typeof settings.getPlinkoRewardSignature === 'function'
      ? settings.getPlinkoRewardSignature
      : getPlinkoRewardSignature;
    const state = context && context.plinko || settings.state || {};
    const boardId = normalizeId(board && (board.id || board.tier));
    const slots = Array.isArray(board && board.slots) ? board.slots : [];
    const parts = [
      boardId,
      normalizeId(board && board.stage),
      slots.length
    ];
    const overrides = state.slotOverrides && typeof state.slotOverrides === 'object' ? state.slotOverrides : {};
    slots.forEach((slot) => {
      if (!slot || !slot.id || slot.teleport) return;
      const override = overrides[getOverrideKey(boardId, slot.id)];
      if (override) parts.push(`${normalizeId(slot.id)}:${getRewardSignature(override)}`);
    });
    return parts.join('|');
  }

  function createPlinkoSlotSnapshot(slot, index, options) {
    const settings = options || {};
    const getBoard = typeof settings.getPlinkoBoard === 'function'
      ? settings.getPlinkoBoard
      : (target) => getPlinkoBoard(target, settings);
    const formatReward = typeof settings.formatRewardSummary === 'function'
      ? settings.formatRewardSummary
      : function formatRewardSummaryFallback(reward) {
          return Object.keys(reward || {}).length ? 'Reward' : 'No reward';
        };
    const formatOdds = typeof settings.formatPlinkoSlotOdds === 'function'
      ? settings.formatPlinkoSlotOdds
      : (targetSlot) => formatPlinkoSlotOdds(targetSlot, settings);
    const reward = clonePlain(slot && slot.reward || {});
    const targetBoard = slot && slot.teleport && slot.nextBoardId ? getBoard(slot.nextBoardId) : null;
    const summary = slot && slot.teleport
      ? `Teleports to ${targetBoard && targetBoard.title || 'a higher board'}`
      : formatReward(reward);
    return Object.assign({ index }, slot || {}, {
      index,
      reward,
      summary,
      tooltipLines: [
        summary,
        `Chance: ${formatOdds(slot)}`,
        slot && slot.teleport ? 'No prize here; the same ball continues on the next board.' : ''
      ].filter(Boolean)
    });
  }

  function createPlinkoBoardSnapshot(board, context, options) {
    if (!board) return null;
    const settings = options || {};
    const getSlots = typeof settings.getPlinkoBoardSlots === 'function'
      ? settings.getPlinkoBoardSlots
      : (target, targetContext) => getPlinkoBoardSlots(target, targetContext, settings);
    const createSlotSnapshot = typeof settings.createPlinkoSlotSnapshot === 'function'
      ? settings.createPlinkoSlotSnapshot
      : (slot, index) => createPlinkoSlotSnapshot(slot, index, settings);
    const formatReward = typeof settings.formatRewardSummary === 'function'
      ? settings.formatRewardSummary
      : function formatRewardSummaryFallback(reward) {
          return Object.keys(reward || {}).length ? 'Reward' : 'No reward';
        };
    return {
      id: board.id || board.tier || '',
      tier: board.tier || board.id || '',
      stage: board.stage || 'main',
      title: board.title || board.id || 'Plinko',
      slots: getSlots(board.id || board.tier, context).map((slot, index) => createSlotSnapshot(slot, index)),
      pityReward: board.pityReward ? clonePlain(board.pityReward) : {},
      pityRewardSummary: formatReward(board.pityReward || {})
    };
  }

  function getPlinkoRewardForSlot(ballOrBoardId, slotId, options) {
    const settings = options || {};
    const getSlots = typeof settings.getPlinkoBoardSlots === 'function'
      ? settings.getPlinkoBoardSlots
      : (target) => getPlinkoBoardSlots(target, null, settings);
    const slot = getSlots(ballOrBoardId).find((candidate) => candidate && candidate.id === slotId);
    return clonePlain(slot && slot.reward || {});
  }

  function getPlinkoPossibleRewardsForBall(ball, includePity, options) {
    const settings = options || {};
    const getChain = typeof settings.getPlinkoBoardChain === 'function'
      ? settings.getPlinkoBoardChain
      : (target) => getPlinkoBoardChain(target, settings);
    const getBoard = typeof settings.getPlinkoBoard === 'function'
      ? settings.getPlinkoBoard
      : (target) => getPlinkoBoard(target, settings);
    const getSlots = typeof settings.getPlinkoBoardSlots === 'function'
      ? settings.getPlinkoBoardSlots
      : (target) => getPlinkoBoardSlots(target, null, settings);
    const mergeReward = typeof settings.mergePlinkoReward === 'function'
      ? settings.mergePlinkoReward
      : (target, source) => mergePlinkoReward(target, source, settings);
    const boards = getChain(ball);
    const mainBoard = boards[0] || getBoard(ball);
    const pityReward = includePity && mainBoard && mainBoard.pityReward ? mainBoard.pityReward : null;
    return boards.flatMap((board) => getSlots(board.id).filter((slot) => !(slot && slot.teleport)).map((slot) => {
      const reward = clonePlain(slot && slot.reward || {});
      return pityReward ? mergeReward(reward, pityReward) : reward;
    }));
  }

  function getPlinkoDropReadiness(ballId, options) {
    const settings = options || {};
    const state = settings.state || {};
    const getBall = typeof settings.getPlinkoBall === 'function'
      ? settings.getPlinkoBall
      : (target) => getPlinkoBall(target, settings);
    const ball = getBall(ballId || state.selectedBallId);
    if (!ball) return { canDrop: false, disabledReason: 'No Plinko ball selected.' };
    const count = typeof settings.getPlinkoBallCount === 'function'
      ? Math.max(0, Math.floor(Number(settings.getPlinkoBallCount(ball.id) || 0) || 0))
      : Math.max(0, Math.floor(Number(settings.ballCounts && settings.ballCounts[ball.id] || 0) || 0));
    if (count <= 0) return { canDrop: false, disabledReason: `${ball.name} is out of stock.` };
    const activeDropLimit = Math.max(0, Math.floor(Number(settings.activeDropLimit == null ? 0 : settings.activeDropLimit) || 0));
    if (activeDropLimit > 0 && (state.pendingDrops || []).length >= activeDropLimit) {
      return { canDrop: false, disabledReason: 'A Plinko ball is still landing.' };
    }
    const prizeTrayFull = typeof settings.isPlinkoPrizeTrayFull === 'function'
      ? settings.isPlinkoPrizeTrayFull()
      : !!settings.prizeTrayFull;
    if (prizeTrayFull) return { canDrop: false, disabledReason: 'Claim stored Plinko prizes before dropping more balls.' };
    return { canDrop: true, disabledReason: '' };
  }

  function createPlinkoDropRequestPlan(ballId, state, options) {
    const settings = options || {};
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const getSnapshotBall = typeof settings.getPlinkoSnapshotBall === 'function'
      ? settings.getPlinkoSnapshotBall
      : (targetState) => getPlinkoSnapshotBall(targetState, settings);
    const getBall = typeof settings.getPlinkoBall === 'function'
      ? settings.getPlinkoBall
      : (targetBallId) => getPlinkoBall(targetBallId, settings);
    const getReadiness = typeof settings.getPlinkoDropReadiness === 'function'
      ? settings.getPlinkoDropReadiness
      : (targetBallId) => getPlinkoDropReadiness(targetBallId, Object.assign({}, settings, { state: plinkoState }));
    const fallbackBall = getSnapshotBall(plinkoState);
    const ball = getBall(ballId || fallbackBall && fallbackBall.id || plinkoState.selectedBallId);
    const readiness = ball ? getReadiness(ball.id) || {} : { canDrop: false, disabledReason: 'No Plinko ball selected.' };
    return {
      ball,
      readiness,
      canDrop: !!readiness.canDrop,
      disabledReason: readiness.disabledReason || '',
      toastMessage: readiness.canDrop ? '' : readiness.disabledReason || 'Cannot drop this ball yet.'
    };
  }

  function rollPlinkoPath(slotCount, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const random = typeof settings.random === 'function' ? settings.random : Math.random;
    const derivedBounces = Math.floor(Number(slotCount || 0) || 0) - 1;
    const fallbackBounces = Math.floor(Number(
      data.PLINKO_BOUNCE_COUNT || settings.defaultBounceCount || PLINKO_DEFAULT_BOUNCE_COUNT
    ) || (settings.defaultBounceCount || PLINKO_DEFAULT_BOUNCE_COUNT));
    const bounceCount = Math.max(1, derivedBounces > 0 ? derivedBounces : fallbackBounces);
    const path = [];
    let slotIndex = 0;
    for (let index = 0; index < bounceCount; index += 1) {
      const right = random() >= 0.5;
      path.push(right ? 'R' : 'L');
      if (right) slotIndex += 1;
    }
    return { path, slotIndex };
  }

  function createPlinkoDropSegment(board, slot, roll, slotIndex, nextBoard, reward, summary, tier, options) {
    const settings = options || {};
    const currentBoard = board || {};
    const targetSlot = slot || {};
    return {
      boardId: currentBoard.id || currentBoard.tier || tier,
      boardTitle: currentBoard.title || currentBoard.id || '',
      boardStage: currentBoard.stage || 'main',
      path: roll && roll.path,
      slotCount: Math.max(0, Math.floor(Number(settings.slotCount || 0) || 0)),
      slotId: targetSlot.id,
      slotIndex,
      slotLabel: targetSlot.label || targetSlot.id,
      teleport: !!targetSlot.teleport,
      nextBoardId: nextBoard && nextBoard.id || '',
      jackpot: !!targetSlot.jackpot,
      reward: clonePlain(reward || {}),
      summary,
      durationMs: targetSlot.teleport ? 2100 : targetSlot.jackpot ? 3400 : 2600
    };
  }

  function createPlinkoDropPath(ball, board, options) {
    const settings = options || {};
    const tier = normalizeId(settings.tier || ball && ball.plinkoTier) || 'basic';
    const maxHops = Math.max(1, Math.floor(Number(settings.maxHops || 3) || 3));
    const getBoard = typeof settings.getPlinkoBoard === 'function'
      ? settings.getPlinkoBoard
      : (ballOrTier) => getPlinkoBoard(ballOrTier, settings);
    const getSlots = typeof settings.getPlinkoBoardSlots === 'function'
      ? settings.getPlinkoBoardSlots
      : (ballOrBoardId, context) => getPlinkoBoardSlots(ballOrBoardId, context, settings);
    const rollPath = typeof settings.rollPlinkoPath === 'function'
      ? settings.rollPlinkoPath
      : (slotCount) => rollPlinkoPath(slotCount, settings);
    const getRewardForSlot = typeof settings.getPlinkoRewardForSlot === 'function'
      ? settings.getPlinkoRewardForSlot
      : (boardId, slotId) => getPlinkoRewardForSlot(boardId, slotId, settings);
    const formatReward = typeof settings.formatRewardSummary === 'function'
      ? settings.formatRewardSummary
      : function formatRewardSummaryFallback(reward) {
          return Object.keys(reward || {}).length ? 'Reward' : 'No reward';
        };
    const createSegment = typeof settings.createPlinkoDropSegment === 'function'
      ? settings.createPlinkoDropSegment
      : createPlinkoDropSegment;
    const segments = [];
    let currentBoard = board || getBoard(ball || tier);
    let finalSlot = null;
    let finalBoard = currentBoard;
    let teleportCount = 0;
    for (let hop = 0; currentBoard && hop < maxHops; hop += 1) {
      const slots = getSlots(currentBoard.id || currentBoard.tier);
      const slotList = Array.isArray(slots) ? slots : [];
      const roll = rollPath(slotList.length) || {};
      const slotIndex = clamp(Math.floor(Number(roll.slotIndex || 0) || 0), 0, Math.max(0, slotList.length - 1));
      const slot = slotList[slotIndex];
      if (!slot) break;
      const nextBoard = slot.teleport && slot.nextBoardId ? getBoard(slot.nextBoardId) : null;
      const reward = slot.teleport ? {} : clonePlain(slot.reward || getRewardForSlot(currentBoard.id, slot.id));
      const summary = slot.teleport
        ? `Teleports to ${nextBoard && nextBoard.title || 'a higher board'}`
        : formatReward(reward);
      segments.push(createSegment(currentBoard, slot, roll, slotIndex, nextBoard, reward, summary, tier, { slotCount: slotList.length }));
      if (slot.teleport && nextBoard) {
        teleportCount += 1;
        currentBoard = nextBoard;
        continue;
      }
      finalSlot = slot;
      finalBoard = currentBoard;
      break;
    }
    return {
      segments,
      finalSlot,
      finalBoard,
      teleportCount
    };
  }

  function createPlinkoPityProgress(ball, pityBefore, options) {
    const settings = options || {};
    const getTarget = typeof settings.getPlinkoPityTarget === 'function'
      ? settings.getPlinkoPityTarget
      : (targetBall) => getPlinkoPityTarget(targetBall, settings);
    const pityTarget = getTarget(ball);
    const before = Math.max(0, Math.floor(Number(pityBefore || 0) || 0));
    const pityGain = Math.max(1, Math.floor(Number(ball && ball.pity || 1) || 1));
    const pityTotal = before + pityGain;
    const pityJackpot = pityTotal >= pityTarget;
    const pityAfter = pityJackpot ? pityTotal % pityTarget : pityTotal;
    return {
      pityTarget,
      pityBefore: before,
      pityGain,
      pityTotal,
      pityJackpot,
      pityAfter
    };
  }

  function createPlinkoDropStartPlan(ball, state, options) {
    if (!ball) return null;
    const settings = options || {};
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const totalDrops = Math.max(0, Math.floor(Number(plinkoState.totalDrops || 0) || 0)) + 1;
    const tier = normalizeId(ball.plinkoTier) || 'basic';
    const pityBefore = getPlinkoPityForBall(ball.id, plinkoState, settings);
    const pityProgress = createPlinkoPityProgress(ball, pityBefore, settings);
    return Object.assign({
      totalDrops,
      tier
    }, pityProgress);
  }

  function createPlinkoPendingDropQueue(pendingDrops, pending, activeDropLimit) {
    const drops = Array.isArray(pendingDrops) ? pendingDrops : [];
    const nextDrops = drops.concat(pending);
    const limit = Math.max(0, Math.floor(Number(activeDropLimit || 0) || 0));
    return limit > 0 ? nextDrops.slice(-limit) : nextDrops;
  }

  function createPlinkoPendingDropEntry(ball, finalBoard, finalSlot, finalSegment, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const segments = Array.isArray(settings.segments) ? settings.segments : [];
    const now = typeof settings.now === 'function' ? settings.now() : Date.now();
    const totalDrops = Math.max(0, Math.floor(Number(settings.totalDrops || 0) || 0));
    const tier = normalizeId(settings.tier || ball && ball.plinkoTier) || 'basic';
    const durationMs = Math.max(1, Math.floor(Number(settings.durationMs || segments.reduce((sum, segment) => {
      return sum + Math.max(1, Number(segment && segment.durationMs || 0) || 0);
    }, 0)) || 0));
    return {
      id: `plinko_${now.toString(36)}_${totalDrops}`,
      ballId: normalizeId(ball && ball.id),
      ballName: String(ball && ball.name || ''),
      ballIcon: String(ball && ball.icon || ''),
      ballAsset: data.ITEM_ASSETS && ball && data.ITEM_ASSETS[ball.id] || '',
      tier,
      boardId: normalizeId(finalBoard && finalBoard.id || tier),
      boardTitle: String(finalBoard && finalBoard.title || ''),
      boardStage: String(finalBoard && finalBoard.stage || ''),
      path: Array.isArray(finalSegment && finalSegment.path) ? finalSegment.path.slice() : [],
      slotId: normalizeId(finalSlot && finalSlot.id),
      slotIndex: Math.max(0, Math.floor(Number(finalSegment && finalSegment.slotIndex || 0) || 0)),
      slotLabel: String(finalSlot && (finalSlot.label || finalSlot.id) || ''),
      summary: String(settings.summary || ''),
      jackpot: !!(finalSlot && finalSlot.jackpot || settings.pityJackpot),
      pityJackpot: !!settings.pityJackpot,
      teleportCount: Math.max(0, Math.floor(Number(settings.teleportCount || 0) || 0)),
      segments,
      pityBefore: Math.max(0, Math.floor(Number(settings.pityBefore || 0) || 0)),
      pityAfter: Math.max(0, Math.floor(Number(settings.pityAfter || 0) || 0)),
      reward: clonePlain(settings.reward || {}),
      createdAt: now,
      resolveAt: now + durationMs,
      durationMs
    };
  }

  function getPlinkoDropDuration(segments) {
    return (Array.isArray(segments) ? segments : []).reduce((sum, segment) => {
      return sum + Math.max(1, Number(segment && segment.durationMs || 0) || 0);
    }, 0);
  }

  function applyPlinkoFinalDropReward(finalSegment, finalSlot, finalBoard, mainBoard, tier, pityJackpot, options) {
    const settings = options || {};
    const getRewardForSlot = typeof settings.getPlinkoRewardForSlot === 'function'
      ? settings.getPlinkoRewardForSlot
      : (boardId, slotId) => getPlinkoRewardForSlot(boardId, slotId, settings);
    const mergeReward = typeof settings.mergePlinkoReward === 'function'
      ? settings.mergePlinkoReward
      : (target, source) => mergePlinkoReward(target, source, settings);
    const formatReward = typeof settings.formatRewardSummary === 'function'
      ? settings.formatRewardSummary
      : function formatRewardSummaryFallback(reward) {
          return Object.keys(reward || {}).length ? 'Reward' : 'No reward';
        };
    const boardId = finalBoard && finalBoard.id || tier;
    let reward = clonePlain(finalSlot && finalSlot.reward || getRewardForSlot(boardId, finalSlot && finalSlot.id));
    if (pityJackpot && mainBoard && mainBoard.pityReward) reward = mergeReward(reward, mainBoard.pityReward);
    const summary = formatReward(reward);
    const segment = finalSegment || {};
    segment.reward = clonePlain(reward);
    segment.summary = summary;
    segment.jackpot = !!(segment.jackpot || pityJackpot);
    segment.durationMs = (segment.jackpot || pityJackpot) ? 3400 : segment.durationMs;
    return { reward, summary, finalSegment: segment };
  }

  function createPlinkoDropOutcomePlan(segments, finalSlot, finalBoard, mainBoard, tier, pityJackpot, options) {
    const segmentList = Array.isArray(segments) ? segments : [];
    const finalSegment = segmentList[segmentList.length - 1] || null;
    if (!finalSlot || !finalSegment) return null;
    const resolution = applyPlinkoFinalDropReward(finalSegment, finalSlot, finalBoard, mainBoard, tier, pityJackpot, options);
    return {
      finalSegment: resolution.finalSegment || finalSegment,
      reward: resolution.reward,
      summary: resolution.summary,
      durationMs: getPlinkoDropDuration(segmentList)
    };
  }

  function createPlinkoGearReplacementReward(slot, reward, options) {
    const settings = options || {};
    const getEquipmentDefinition = typeof settings.getEquipmentDefinition === 'function'
      ? settings.getEquipmentDefinition
      : function getEquipmentDefinitionFallback() {
          return null;
        };
    const getEquipmentCatalog = typeof settings.getEquipmentCatalog === 'function'
      ? settings.getEquipmentCatalog
      : function getEquipmentCatalogFallback() {
          return [];
        };
    const getRarity = typeof settings.getPlinkoRarityFromTier === 'function'
      ? settings.getPlinkoRarityFromTier
      : (tier, fallback) => getPlinkoRarityFromTier(tier, fallback, settings);
    const getSeed = typeof settings.getPlinkoRotationSeed === 'function'
      ? settings.getPlinkoRotationSeed
      : (boardId, slotId) => getPlinkoRotationSeed(boardId, slotId, settings);
    const pickReplacement = typeof settings.pickPlinkoReplacement === 'function'
      ? settings.pickPlinkoReplacement
      : pickPlinkoReplacement;
    const output = clonePlain(reward || {});
    const itemConfigs = getRewardItemConfigs(output, settings);
    if (!itemConfigs.length) return null;
    const primaryConfig = itemConfigs[0] || {};
    const currentId = normalizeId(primaryConfig.itemId || primaryConfig.id);
    const currentDefinition = getEquipmentDefinition(currentId);
    const rarity = getRarity(slot && slot.rewardTier, primaryConfig.rarity || currentDefinition && currentDefinition.rarity || 'Rare');
    const catalog = getEquipmentCatalog().filter((item) => item && item.id);
    const sameSlot = currentDefinition && currentDefinition.slot
      ? catalog.filter((item) => item.slot === currentDefinition.slot && (item.rarity || 'Common') === rarity)
      : [];
    const sameRarity = catalog.filter((item) => (item.rarity || 'Common') === rarity);
    const candidates = sameSlot.length > 1 ? sameSlot : sameRarity.length > 1 ? sameRarity : catalog;
    const seed = getSeed(slot && slot.boardId, slot && slot.id);
    output.items = itemConfigs.map((config, index) => {
      const chosen = pickReplacement(candidates, normalizeId(config.itemId || config.id), seed + index);
      const next = Object.assign({}, config, {
        itemId: chosen && chosen.id || normalizeId(config.itemId || config.id),
        rarity: config.rarity || chosen && chosen.rarity || rarity,
        upgrade: Math.max(0, Math.floor(Number(config.upgrade || 0) || 0))
      });
      delete next.id;
      return next;
    });
    return output;
  }

  function createPlinkoCardReplacementReward(slot, reward, options) {
    const settings = options || {};
    const getCardDefinition = typeof settings.getCardDefinition === 'function'
      ? settings.getCardDefinition
      : function getCardDefinitionFallback() {
          return null;
        };
    const getCardDefinitionsByRarity = typeof settings.getCardDefinitionsByRarity === 'function'
      ? settings.getCardDefinitionsByRarity
      : function getCardDefinitionsByRarityFallback() {
          return [];
        };
    const getRarity = typeof settings.getPlinkoRarityFromTier === 'function'
      ? settings.getPlinkoRarityFromTier
      : (tier, fallback) => getPlinkoRarityFromTier(tier, fallback, settings);
    const getSeed = typeof settings.getPlinkoRotationSeed === 'function'
      ? settings.getPlinkoRotationSeed
      : (boardId, slotId) => getPlinkoRotationSeed(boardId, slotId, settings);
    const pickReplacement = typeof settings.pickPlinkoReplacement === 'function'
      ? settings.pickPlinkoReplacement
      : pickPlinkoReplacement;
    const output = clonePlain(reward || {});
    const cardConfigs = getRewardCardConfigs(output, settings);
    if (!cardConfigs.length) return null;
    const primaryConfig = cardConfigs[0] || {};
    const currentDefinition = getCardDefinition(primaryConfig.cardId || primaryConfig.id);
    const rarity = getRarity(slot && slot.rewardTier, currentDefinition && currentDefinition.rarity || 'Rare');
    const candidates = getCardDefinitionsByRarity(rarity);
    const seed = getSeed(slot && slot.boardId, slot && slot.id);
    output.cards = {};
    cardConfigs.forEach((config, index) => {
      const currentId = normalizeId(config.cardId || config.id);
      const chosen = pickReplacement(candidates, currentId, seed + index);
      const nextId = chosen && chosen.id || currentId;
      const amount = Math.max(1, Math.floor(Number(config.quantity || config.amount || 1) || 1));
      if (nextId) output.cards[nextId] = Number(output.cards[nextId] || 0) + amount;
    });
    delete output.cardConfigs;
    return output;
  }

  function createPlinkoSlotCouponReplacementReward(reward, options) {
    const settings = options || {};
    const slotCouponIds = Array.isArray(settings.slotCouponIds) ? settings.slotCouponIds : PLINKO_SLOT_COUPON_IDS;
    const output = clonePlain(reward || {});
    const nextConsumables = {};
    Object.entries(output.consumables || {}).forEach(([id, amount]) => {
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (value <= 0) return;
      const index = slotCouponIds.indexOf(id);
      const nextId = index >= 0 ? slotCouponIds[(index + 1) % slotCouponIds.length] : id;
      nextConsumables[nextId] = Number(nextConsumables[nextId] || 0) + value;
    });
    output.consumables = nextConsumables;
    return output;
  }

  function createPlinkoRateCouponReplacementReward(reward, options) {
    const settings = options || {};
    const rateCouponIds = Array.isArray(settings.rateCouponIds) ? settings.rateCouponIds : PLINKO_RATE_COUPON_IDS;
    const output = clonePlain(reward || {});
    const nextConsumables = {};
    Object.entries(output.consumables || {}).forEach(([id, amount]) => {
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (value <= 0) return;
      let nextId = id;
      if (rateCouponIds.includes(id)) {
        nextId = id.indexOf('xp_coupon_') === 0
          ? id.replace('xp_coupon_', 'drop_coupon_')
          : id.replace('drop_coupon_', 'xp_coupon_');
      }
      nextConsumables[nextId] = Number(nextConsumables[nextId] || 0) + value;
    });
    output.consumables = nextConsumables;
    return output;
  }

  function createPlinkoPrismReplacementReward(reward, options) {
    const settings = options || {};
    const potentialCubeId = normalizeId(settings.potentialCubeId || PLINKO_POTENTIAL_CUBE_ID);
    const preservationCubeId = normalizeId(settings.preservationCubeId || PLINKO_PRESERVATION_CUBE_ID);
    const output = clonePlain(reward || {});
    const nextConsumables = {};
    Object.entries(output.consumables || {}).forEach(([id, amount]) => {
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (value <= 0) return;
      const nextId = id === potentialCubeId ? preservationCubeId : id === preservationCubeId ? potentialCubeId : id;
      nextConsumables[nextId] = Number(nextConsumables[nextId] || 0) + value;
    });
    if (!Object.keys(nextConsumables).length) nextConsumables[potentialCubeId] = 1;
    output.consumables = nextConsumables;
    return output;
  }

  function createPlinkoMaterialReplacementReward(reward) {
    const output = clonePlain(reward || {});
    if (output.currency) output.currency = Math.max(1, Math.ceil(Number(output.currency || 0) * 1.06));
    if (output.materials && Object.keys(output.materials).length) {
      output.materials = Object.entries(output.materials).reduce((materials, [id, amount]) => {
        const value = Math.max(1, Math.floor(Number(amount || 0) || 0));
        materials[id] = Math.max(1, Math.ceil(value * 1.08));
        return materials;
      }, {});
    } else if (!output.currency) {
      output.materials = { upgradeDust: 16 };
    }
    return output;
  }

  function createPlinkoReplacementReward(slot, reward, options) {
    const settings = options || {};
    const getRewardSignature = typeof settings.getPlinkoRewardSignature === 'function'
      ? settings.getPlinkoRewardSignature
      : getPlinkoRewardSignature;
    const family = normalizeId(slot && slot.rewardFamily);
    const source = clonePlain(reward || {});
    let replacement = null;
    if (family === 'gear' || getRewardItemConfigs(source, settings).length) {
      replacement = createPlinkoGearReplacementReward(slot, source, settings);
    } else if (family === 'card' || getRewardCardConfigs(source, settings).length) {
      replacement = createPlinkoCardReplacementReward(slot, source, settings);
    } else if (family === 'slot_coupon') {
      replacement = createPlinkoSlotCouponReplacementReward(source, settings);
    } else if (family === 'rate_coupon') {
      replacement = createPlinkoRateCouponReplacementReward(source, settings);
    } else if (family === 'prism') {
      replacement = createPlinkoPrismReplacementReward(source, settings);
    } else {
      replacement = createPlinkoMaterialReplacementReward(source, settings);
    }
    if (!replacement || getRewardSignature(replacement) === getRewardSignature(source)) {
      replacement = createPlinkoMaterialReplacementReward(source, settings);
    }
    if (replacement && !(slot && slot.jackpot)) {
      delete replacement.currency;
      const hasPayload = Object.entries(replacement).some(([key, value]) => {
        if (key === 'currency') return Number(value || 0) > 0;
        if (Array.isArray(value)) return value.length > 0;
        if (value && typeof value === 'object') return Object.keys(value).length > 0;
        return !!value;
      });
      if (!hasPayload) replacement.materials = { upgradeDust: 16 };
    }
    return replacement && Object.keys(replacement).length ? replacement : null;
  }

  function createPlinkoSlotRewardRotation(board, slotId, reward, options) {
    const settings = options || {};
    const currentBoard = board || {};
    const targetSlotId = normalizeId(slotId);
    const baseSlot = (currentBoard.slots || []).find((slot) => slot && slot.id === targetSlotId);
    if (!baseSlot || baseSlot.teleport) return null;
    const currentReward = clonePlain(reward || {});
    if (!Object.keys(currentReward).length) return null;
    const resolveSlot = typeof settings.createPlinkoResolvedSlot === 'function'
      ? settings.createPlinkoResolvedSlot
      : (targetBoard, targetSlot) => createPlinkoResolvedSlot(targetBoard, targetSlot, settings.context || null, settings);
    const createReplacement = typeof settings.createPlinkoReplacementReward === 'function'
      ? settings.createPlinkoReplacementReward
      : (targetSlot, targetReward) => createPlinkoReplacementReward(targetSlot, targetReward, settings);
    const getRewardSignature = typeof settings.getPlinkoRewardSignature === 'function'
      ? settings.getPlinkoRewardSignature
      : getPlinkoRewardSignature;
    const getOverrideKey = typeof settings.getPlinkoSlotOverrideKey === 'function'
      ? settings.getPlinkoSlotOverrideKey
      : getPlinkoSlotOverrideKey;
    const replacement = createReplacement(resolveSlot(currentBoard, baseSlot), currentReward);
    if (!replacement || getRewardSignature(replacement) === getRewardSignature(currentReward)) return null;
    const boardId = normalizeId(currentBoard.id || settings.boardId);
    return {
      boardId,
      slotId: normalizeId(baseSlot.id),
      overrideKey: getOverrideKey(boardId, baseSlot.id),
      replacement: clonePlain(replacement)
    };
  }

  function createPlinkoRewardHistory(result, entries, options) {
    const settings = options || {};
    const limit = Math.max(0, Math.floor(Number(settings.limit == null ? 6 : settings.limit) || 0));
    const previous = Array.isArray(entries) ? entries : [];
    return [normalizePlinkoRecentReward(result, settings)].concat(previous).slice(0, limit);
  }

  function createPlinkoStoredPrizeEntry(result, reward, index, options) {
    const settings = options || {};
    const now = typeof settings.now === 'function' ? settings.now : Date.now;
    const prizeIndex = Math.max(0, Math.floor(Number(index || 0) || 0));
    return normalizePlinkoPrizeTrayEntry(Object.assign({}, result || {}, {
      id: (result && result.id ? `${result.id}_prize` : '') || `plinko_prize_${now().toString(36)}_${prizeIndex}`,
      reward: clonePlain(reward || {}),
      storedInPrizeTray: true,
      createdAt: now()
    }), prizeIndex, settings);
  }

  function createPlinkoPrizeStoragePlan(prizeTray, result, reward, limit, options) {
    const settings = options || {};
    const tray = Array.isArray(prizeTray) ? prizeTray : [];
    const maxEntries = Math.max(0, Math.floor(Number(limit || 0) || 0));
    if (tray.length >= maxEntries) return null;
    const createEntry = typeof settings.createPlinkoStoredPrizeEntry === 'function'
      ? settings.createPlinkoStoredPrizeEntry
      : (targetResult, targetReward, index) => createPlinkoStoredPrizeEntry(targetResult, targetReward, index, settings);
    const entry = createEntry(result, reward, tray.length);
    if (!entry) return null;
    return {
      index: tray.length,
      entry
    };
  }

  function createPlinkoPrizeClaimPlan(prizeTray, options) {
    const settings = options || {};
    const getBlockReason = typeof settings.getPlinkoInventoryBlockReasonForReward === 'function'
      ? settings.getPlinkoInventoryBlockReasonForReward
      : function getPlinkoInventoryBlockReasonForRewardFallback() {
          return '';
        };
    const remaining = [];
    const claimableRewards = [];
    (Array.isArray(prizeTray) ? prizeTray : []).forEach((entry) => {
      const reward = clonePlain(entry && entry.reward || {});
      if (!reward || !Object.keys(reward).length) return;
      const blocked = getBlockReason(reward, entry);
      if (blocked) {
        remaining.push(entry);
        return;
      }
      claimableRewards.push(reward);
    });
    return {
      remaining,
      claimableRewards,
      claimed: claimableRewards.length,
      remainingCount: remaining.length
    };
  }

  function createPlinkoPrizeClaimResult(plan) {
    const source = plan && typeof plan === 'object' ? plan : {};
    const claimed = Math.max(0, Math.floor(Number(source.claimed == null
      ? Array.isArray(source.claimableRewards) ? source.claimableRewards.length : 0
      : source.claimed) || 0));
    const remaining = Math.max(0, Math.floor(Number(source.remainingCount == null
      ? Array.isArray(source.remaining) ? source.remaining.length : 0
      : source.remainingCount) || 0));
    const message = claimed > 0
      ? `Claimed ${claimed} Plinko prize${claimed === 1 ? '' : 's'}.${remaining ? ` Make room for ${remaining} more.` : ''}`
      : `Make room for ${remaining} stored Plinko prize${remaining === 1 ? '' : 's'}.`;
    return {
      claimed,
      remaining,
      toastMessage: message,
      toastOptions: claimed > 0 ? { noEmit: true } : null,
      emitUiChange: claimed > 0,
      emitDomains: ['hud', 'shop', 'inventory', 'equipment', 'cards'],
      emitReason: 'plinkoPrizeClaim',
      persist: true
    };
  }

  function createPlinkoPrizeTraySnapshot(prizeTray, options) {
    const settings = options || {};
    const getBlockReason = typeof settings.getPlinkoInventoryBlockReasonForReward === 'function'
      ? settings.getPlinkoInventoryBlockReasonForReward
      : function getPlinkoInventoryBlockReasonForRewardFallback() {
          return '';
        };
    return (Array.isArray(prizeTray) ? prizeTray : []).map((entry) => {
      const snapshot = createPlinkoDropSnapshot(entry);
      const disabledReason = String(getBlockReason(snapshot.reward || {}, snapshot) || '');
      return Object.assign({}, snapshot, {
        claimable: !disabledReason,
        disabledReason
      });
    });
  }

  function createPlinkoDropSnapshot(drop, options) {
    const settings = options || {};
    const source = drop && typeof drop === 'object' ? drop : {};
    const snapshot = Object.assign({}, source, {
      reward: clonePlain(source.reward || {}),
      segments: Array.isArray(source.segments)
        ? source.segments.map((segment) => Object.assign({}, segment, { reward: clonePlain(segment && segment.reward || {}) }))
        : []
    });
    if (Object.prototype.hasOwnProperty.call(settings, 'storedInPrizeTray')) {
      snapshot.storedInPrizeTray = !!settings.storedInPrizeTray;
    }
    return snapshot;
  }

  function createPlinkoDropSnapshots(drops, options) {
    return (Array.isArray(drops) ? drops : []).map((drop) => createPlinkoDropSnapshot(drop, options));
  }

  function createPlinkoSnapshotCacheContext(state, options) {
    const settings = options || {};
    const data = getPlinkoData(settings);
    const plinko = state || {};
    const balls = data.PLINKO_BALLS || [];
    const consumables = settings.consumables || {};
    const pityByBall = plinko.pityByBall && typeof plinko.pityByBall === 'object' ? plinko.pityByBall : {};
    const getListSignature = typeof settings.getPlinkoSnapshotListSignature === 'function'
      ? settings.getPlinkoSnapshotListSignature
      : getPlinkoSnapshotListSignature;
    const getOverrideSignature = typeof settings.getPlinkoSlotOverrideSignature === 'function'
      ? settings.getPlinkoSlotOverrideSignature
      : getPlinkoSlotOverrideSignature;
    const getUsedSlots = typeof settings.getInventoryUsedSlots === 'function'
      ? settings.getInventoryUsedSlots
      : function getInventoryUsedSlotsFallback() {
          return settings.usableUsedSlots || 0;
        };
    const getCapacity = typeof settings.getInventoryCapacity === 'function'
      ? settings.getInventoryCapacity
      : function getInventoryCapacityFallback() {
          return settings.usableCapacity || 0;
        };
    const ballKey = balls.map((ball) => {
      const id = normalizeId(ball && ball.id);
      const count = getPlinkoBallCount(id, consumables);
      const pity = Math.max(0, Math.floor(Number(pityByBall[id] || 0) || 0));
      return `${id}:${count}:${pity}`;
    }).join(',');
    const usableUsedSlots = getUsedSlots('usable');
    const usableCapacity = getCapacity('usable');
    const key = [
      data.PLINKO_BALLS || [],
      balls.length,
      data.PLINKO_BOARDS || data.PLINKO_REWARD_TABLES || {},
      normalizeId(plinko.selectedBallId),
      Math.max(0, Math.floor(Number(plinko.totalDrops || 0) || 0)),
      Math.max(0, Math.floor(Number(settings.currency || 0) || 0)),
      usableUsedSlots,
      usableCapacity,
      Math.max(0, Math.floor(Number(settings.activeDropLimit || 0) || 0)),
      Math.max(0, Math.floor(Number(settings.prizeTrayLimit || 0) || 0)),
      ballKey,
      getListSignature(plinko.pendingDrops || []),
      getListSignature(plinko.prizeTray || []),
      getListSignature(plinko.lastRewards || []),
      getOverrideSignature(plinko)
    ];
    return { key, usableUsedSlots, usableCapacity };
  }

  function createPlinkoSnapshotPayload(state, options) {
    const settings = options || {};
    const plinko = state && state.plinko ? state.plinko : state || {};
    const selectedBall = settings.selectedBall || null;
    const selectedBallId = selectedBall ? selectedBall.id : settings.basicBallId || PLINKO_BASIC_BALL_ID;
    const selectedBoard = settings.selectedBoard || null;
    const selectedBoardSnapshot = settings.selectedBoardSnapshot || null;
    const boardSnapshots = Array.isArray(settings.boardSnapshots) ? settings.boardSnapshots : [];
    const prizeTray = Array.isArray(settings.prizeTray) ? settings.prizeTray : [];
    const readiness = settings.readiness || {};
    const getPityForBall = typeof settings.getPlinkoPityForBall === 'function'
      ? settings.getPlinkoPityForBall
      : (ballId) => getPlinkoPityForBall(ballId, plinko, settings);
    const getPityTarget = typeof settings.getPlinkoPityTarget === 'function'
      ? settings.getPlinkoPityTarget
      : (ball) => getPlinkoPityTarget(ball, settings);
    return {
      pity: getPityForBall(selectedBall && selectedBall.id),
      pityTarget: getPityTarget(selectedBall),
      totalDrops: Math.max(0, Math.floor(Number(plinko.totalDrops || 0) || 0)),
      selectedBallId,
      activeDropLimit: Math.max(0, Math.floor(Number(settings.activeDropLimit || 0) || 0)),
      activeDropCount: (plinko.pendingDrops || []).length,
      prizeTrayLimit: getPlinkoPrizeTrayLimit(settings),
      prizeTrayCount: prizeTray.length,
      prizeTrayFull: !!settings.prizeTrayFull,
      canClaimPrizeTray: prizeTray.some((entry) => entry && entry.claimable),
      claimablePrizeCount: prizeTray.filter((entry) => entry && entry.claimable).length,
      canDropSelected: !!(readiness && readiness.canDrop),
      dropDisabledReason: readiness && readiness.disabledReason || '',
      boardId: selectedBoardSnapshot && selectedBoardSnapshot.id || '',
      boardTitle: selectedBoardSnapshot && selectedBoardSnapshot.title || '',
      boardStage: selectedBoardSnapshot && selectedBoardSnapshot.stage || '',
      boardChain: boardSnapshots.map((board) => ({
        id: board && board.id,
        title: board && board.title,
        stage: board && board.stage,
        tier: board && board.tier
      })),
      boards: settings.boards || {},
      slots: selectedBoardSnapshot && selectedBoardSnapshot.slots || [],
      pityReward: selectedBoard && selectedBoard.pityReward ? clonePlain(selectedBoard.pityReward) : {},
      pityRewardSummary: selectedBoardSnapshot && selectedBoardSnapshot.pityRewardSummary || '',
      balls: Array.isArray(settings.balls) ? settings.balls : [],
      pendingDrops: Array.isArray(settings.pendingDrops) ? settings.pendingDrops : [],
      prizeTray,
      lastRewards: Array.isArray(settings.lastRewards) ? settings.lastRewards : []
    };
  }

  function createPlinkoDropFinalizationPlan(state, dropId) {
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const id = normalizeId(dropId);
    const pendingDrops = Array.isArray(plinkoState.pendingDrops) ? plinkoState.pendingDrops : [];
    const index = pendingDrops.findIndex((drop) => drop && drop.id === id);
    if (index < 0) return null;
    const pending = pendingDrops[index];
    return {
      id,
      index,
      pending,
      reward: clonePlain(pending && pending.reward || {})
    };
  }

  function createPlinkoFinalizedDropResult(pending, options) {
    const settings = options || {};
    const now = typeof settings.now === 'function' ? settings.now : Date.now;
    return normalizePlinkoRecentReward(Object.assign({}, pending || {}, {
      reward: clonePlain(settings.reward || {}),
      replacementReward: settings.replacementReward,
      storedInPrizeTray: !!settings.storedInPrizeTray,
      createdAt: now()
    }), settings);
  }

  function createPlinkoFinalizedDropUiPlan(result, options) {
    const settings = options || {};
    const storedInPrizeTray = !!(settings.storedInPrizeTray || result && result.storedInPrizeTray);
    const reward = settings.reward || result && result.reward || {};
    return {
      popupTitle: `${storedInPrizeTray ? 'Stored Plinko' : 'Plinko'}: ${result && result.slotLabel || ''}`,
      popupReward: reward,
      emitDomains: ['hud', 'shop', 'inventory', 'equipment', 'cards'],
      emitReason: 'plinkoDrop',
      persist: true,
      snapshotOptions: { storedInPrizeTray }
    };
  }

  function createPlinkoRewardInventoryDemand(reward, options) {
    const settings = options || {};
    const getEquipmentCount = typeof settings.getRewardEquipmentCount === 'function'
      ? settings.getRewardEquipmentCount
      : function getRewardEquipmentCountFallback() {
          return 0;
        };
    const getCardCount = typeof settings.getRewardCardCount === 'function'
      ? settings.getRewardCardCount
      : function getRewardCardCountFallback() {
          return 0;
        };
    const getCardConfigs = typeof settings.getRewardCardConfigs === 'function'
      ? settings.getRewardCardConfigs
      : function getRewardCardConfigsFallback(sourceReward) {
          return Array.isArray(sourceReward && sourceReward.cardConfigs) ? sourceReward.cardConfigs : [];
        };
    const demand = {
      equipment: getEquipmentCount(reward),
      cards: getCardCount(reward),
      cardConfigs: getCardConfigs(reward),
      usable: {},
      etc: {}
    };
    Object.entries(reward && reward.consumables || {}).forEach(([id, amount]) => {
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (value > 0) demand.usable[id] = Number(demand.usable[id] || 0) + value;
    });
    Object.entries(reward && reward.materials || {}).forEach(([id, amount]) => {
      const value = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (value > 0) demand.etc[id] = Number(demand.etc[id] || 0) + value;
    });
    return demand;
  }

  function mergePlinkoInventoryDemand(target, source) {
    const output = target || { equipment: 0, cards: 0, usable: {}, etc: {}, cardConfigs: [] };
    const demand = source || {};
    output.equipment = Number(output.equipment || 0) + Number(demand.equipment || 0);
    output.cards = Number(output.cards || 0) + Number(demand.cards || 0);
    output.cardConfigs = (Array.isArray(output.cardConfigs) ? output.cardConfigs.slice() : []).concat(Array.isArray(demand.cardConfigs) ? demand.cardConfigs : []);
    ['usable', 'etc'].forEach((tab) => {
      output[tab] = Object.assign({}, output[tab] || {});
      Object.entries(demand[tab] || {}).forEach(([id, amount]) => {
        output[tab][id] = Number(output[tab][id] || 0) + Number(amount || 0);
      });
    });
    return output;
  }

  function createPlinkoPendingInventoryDemand(state, options) {
    const settings = options || {};
    const plinkoState = state && state.plinko ? state.plinko : state || {};
    const getRewardDemand = typeof settings.getPlinkoRewardInventoryDemand === 'function'
      ? settings.getPlinkoRewardInventoryDemand
      : (reward) => createPlinkoRewardInventoryDemand(reward, settings);
    const mergeDemand = typeof settings.mergePlinkoInventoryDemand === 'function'
      ? settings.mergePlinkoInventoryDemand
      : mergePlinkoInventoryDemand;
    return (plinkoState.pendingDrops || []).reduce((demand, drop) => {
      return mergeDemand(demand, getRewardDemand(drop && drop.reward || {}));
    }, { equipment: 0, cards: 0, usable: {}, etc: {}, cardConfigs: [] });
  }

  function getPlinkoProjectedStackSlots(tabId, source, additions, options) {
    const settings = options || {};
    const normalizeTab = typeof settings.normalizeInventoryTab === 'function'
      ? settings.normalizeInventoryTab
      : normalizeId;
    const countSlots = typeof settings.countStackedInventorySlots === 'function'
      ? settings.countStackedInventorySlots
      : function countStackedInventorySlotsFallback(targetTab, targetSource, getSlotCount) {
          return Object.entries(targetSource || {}).reduce((sum, [id, count]) => sum + getSlotCount(targetTab, id, count), 0);
        };
    const getSlotCount = typeof settings.getStackSlotCount === 'function'
      ? settings.getStackSlotCount
      : function getStackSlotCountFallback(targetTab, id, count) {
          return Math.max(0, Math.floor(Number(count || 0) || 0)) > 0 ? 1 : 0;
        };
    const isVisibleConsumable = typeof settings.isVisibleInventoryConsumableId === 'function'
      ? settings.isVisibleInventoryConsumableId
      : function isVisibleInventoryConsumableIdFallback() {
          return true;
        };
    const tab = normalizeTab(tabId);
    const inventorySource = source && typeof source === 'object' ? source : {};
    let total = countSlots(tab, inventorySource, (slotTab, id, count) => getSlotCount(slotTab, id, count));
    Object.entries(additions || {}).forEach(([id, amount]) => {
      if (tab === 'usable' && !isVisibleConsumable(id)) return;
      const current = Math.max(0, Math.floor(Number(inventorySource[id]) || 0));
      const added = Math.max(0, Math.floor(Number(amount || 0) || 0));
      if (added <= 0) return;
      total -= getSlotCount(tab, id, current);
      total += getSlotCount(tab, id, current + added);
    });
    return total;
  }

  function getPlinkoInventoryBlockReasonForReward(reward, options) {
    const settings = options || {};
    const getPendingDemand = typeof settings.getPlinkoPendingInventoryDemand === 'function'
      ? settings.getPlinkoPendingInventoryDemand
      : (context) => createPlinkoPendingInventoryDemand(context, settings);
    const getRewardDemand = typeof settings.getPlinkoRewardInventoryDemand === 'function'
      ? settings.getPlinkoRewardInventoryDemand
      : (targetReward) => createPlinkoRewardInventoryDemand(targetReward, settings);
    const mergeDemand = typeof settings.mergePlinkoInventoryDemand === 'function'
      ? settings.mergePlinkoInventoryDemand
      : mergePlinkoInventoryDemand;
    const getUsedSlots = typeof settings.getInventoryUsedSlots === 'function'
      ? settings.getInventoryUsedSlots
      : function getInventoryUsedSlotsFallback() {
          return 0;
        };
    const getCapacity = typeof settings.getInventoryCapacity === 'function'
      ? settings.getInventoryCapacity
      : function getInventoryCapacityFallback() {
          return Number.POSITIVE_INFINITY;
        };
    const getCardStackSlots = typeof settings.getProjectedCardStackSlots === 'function'
      ? settings.getProjectedCardStackSlots
      : function getProjectedCardStackSlotsFallback() {
          return 0;
        };
    const getProjectedStackSlots = typeof settings.getPlinkoProjectedStackSlots === 'function'
      ? settings.getPlinkoProjectedStackSlots
      : (tabId, additions) => getPlinkoProjectedStackSlots(tabId, {}, additions, settings);
    const pending = settings.includePending === false
      ? { equipment: 0, cards: 0, usable: {}, etc: {} }
      : getPendingDemand(settings.context);
    const demand = mergeDemand(pending, getRewardDemand(reward));
    const capacity = (tabId) => {
      const value = Number(getCapacity(tabId));
      return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
    };
    if (Number(getUsedSlots('equipment') || 0) + Number(demand.equipment || 0) > capacity('equipment')) {
      return 'Make room in Equipment inventory before dropping this ball.';
    }
    if (Number(getCardStackSlots(demand.cardConfigs || []) || 0) > capacity('cards')) {
      return 'Make room in Card inventory before dropping this ball.';
    }
    if (Number(getProjectedStackSlots('usable', demand.usable) || 0) > capacity('usable')) {
      return 'Make room in Usable inventory before dropping this ball.';
    }
    if (Number(getProjectedStackSlots('etc', demand.etc) || 0) > capacity('etc')) {
      return 'Make room in Etc inventory before dropping this ball.';
    }
    return '';
  }

  const api = {
    PLINKO_BASIC_BALL_ID,
    PLINKO_DEFAULT_BOUNCE_COUNT,
    PLINKO_PRIZE_TRAY_LIMIT,
    PLINKO_SLOT_COUPON_IDS,
    PLINKO_RATE_COUPON_IDS,
    PLINKO_POTENTIAL_CUBE_ID,
    PLINKO_PRESERVATION_CUBE_ID,
    getPlinkoBall,
    getPlinkoSnapshotBall,
    getPlinkoActiveDropLimit,
    getPlinkoPrizeTrayLimit,
    getPlinkoPityTarget,
    getPlinkoPityForBall,
    getPlinkoBallCount,
    getPlinkoReservedPrizeTrayCount,
    isPlinkoPrizeTrayFull,
    createPlinkoBallSnapshot,
    createPlinkoBallSnapshots,
    createPlinkoBallSelectionPlan,
    createPlinkoBallPurchasePlan,
    createPlinkoBallPurchaseStackPatchOptions,
    normalizePlinkoSegment,
    normalizePlinkoRecentReward,
    normalizePlinkoPrizeTrayEntry,
    normalizePlinkoPendingDrop,
    normalizePlinkoSlotOverrides,
    createPlinkoState,
    getPlinkoRewardSignature,
    getPlinkoSnapshotEntrySignature,
    getPlinkoSnapshotListSignature,
    getPlinkoSlotOverrideKey,
    getPlinkoSlotOverrideSignature,
    formatPlinkoSlotOdds,
    mergePlinkoReward,
    getPlinkoRotationSeed,
    getPlinkoRarityFromTier,
    pickPlinkoReplacement,
    getPlinkoBoards,
    getPlinkoBoard,
    getPlinkoBoardChain,
    createPlinkoResolvedSlot,
    getPlinkoBoardSlots,
    createPlinkoBoardChainSnapshotKey,
    createPlinkoBoardChainSnapshot,
    getPlinkoBoardSnapshotCacheKey,
    createPlinkoSlotSnapshot,
    createPlinkoBoardSnapshot,
    getPlinkoRewardForSlot,
    getPlinkoPossibleRewardsForBall,
    getPlinkoDropReadiness,
    createPlinkoDropRequestPlan,
    rollPlinkoPath,
    createPlinkoDropSegment,
    createPlinkoDropPath,
    createPlinkoPityProgress,
    createPlinkoDropStartPlan,
    createPlinkoPendingDropQueue,
    createPlinkoPendingDropEntry,
    getPlinkoDropDuration,
    applyPlinkoFinalDropReward,
    createPlinkoDropOutcomePlan,
    createPlinkoGearReplacementReward,
    createPlinkoCardReplacementReward,
    createPlinkoSlotCouponReplacementReward,
    createPlinkoRateCouponReplacementReward,
    createPlinkoPrismReplacementReward,
    createPlinkoMaterialReplacementReward,
    createPlinkoReplacementReward,
    createPlinkoSlotRewardRotation,
    createPlinkoRewardHistory,
    createPlinkoStoredPrizeEntry,
    createPlinkoPrizeStoragePlan,
    createPlinkoPrizeClaimPlan,
    createPlinkoPrizeClaimResult,
    createPlinkoPrizeTraySnapshot,
    createPlinkoDropSnapshot,
    createPlinkoDropSnapshots,
    createPlinkoSnapshotCacheContext,
    createPlinkoSnapshotPayload,
    createPlinkoDropFinalizationPlan,
    createPlinkoFinalizedDropResult,
    createPlinkoFinalizedDropUiPlan,
    createPlinkoRewardInventoryDemand,
    mergePlinkoInventoryDemand,
    createPlinkoPendingInventoryDemand,
    getPlinkoProjectedStackSlots,
    getPlinkoInventoryBlockReasonForReward
  };

  global.ProjectStarfallEngineModules = global.ProjectStarfallEngineModules || {};
  global.ProjectStarfallEngineModules.plinko = api;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
