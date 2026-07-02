(function initProjectStarfallUiVisuals(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};

  const DEFAULT_ITEM_RARITY_VISUALS = Object.freeze({
    Common: Object.freeze({ color: '#d8e5ec', glow: 7, alpha: 0.5, ring: 1.4 }),
    Uncommon: Object.freeze({ color: '#74d680', glow: 11, alpha: 0.7, ring: 1.8 }),
    Rare: Object.freeze({ color: '#68a9ff', glow: 15, alpha: 0.82, ring: 2.1 }),
    Epic: Object.freeze({ color: '#c794ff', glow: 19, alpha: 0.92, ring: 2.4, pulse: 0.16 }),
    Relic: Object.freeze({ color: '#ffbe55', glow: 23, alpha: 0.98, ring: 2.7, pulse: 0.2 })
  });

  function getItemRarityVisuals(data) {
    const source = data || Data;
    return source.ITEM_RARITY_VISUALS || DEFAULT_ITEM_RARITY_VISUALS;
  }

  const ITEM_RARITY_VISUALS = getItemRarityVisuals(Data);
  const RARITY_ORDER = Object.freeze(['Common', 'Uncommon', 'Rare', 'Epic', 'Relic']);
  const ITEM_STAT_AURA_TIERS = Object.freeze([
    Object.freeze({ id: 'common', rarity: 'Common', minPower: 0 }),
    Object.freeze({ id: 'uncommon', rarity: 'Uncommon', minPower: 65 }),
    Object.freeze({ id: 'rare', rarity: 'Rare', minPower: 130 }),
    Object.freeze({ id: 'epic', rarity: 'Epic', minPower: 260 }),
    Object.freeze({ id: 'relic', rarity: 'Relic', minPower: 450 })
  ]);
  const ATTUNEMENT_TIER_VISUALS = Object.freeze({
    rare: Object.freeze({ color: '#68a9ff', stroke: 'rgba(47,125,214,0.64)' }),
    epic: Object.freeze({ color: '#c794ff', stroke: 'rgba(136,86,197,0.68)' }),
    relic: Object.freeze({ color: '#ffbe55', stroke: 'rgba(216,165,49,0.76)' }),
    mythic: Object.freeze({ color: '#ff6f91', stroke: 'rgba(255,111,145,0.78)' }),
    ascendant: Object.freeze({ color: '#42d6ff', stroke: 'rgba(66,214,255,0.82)' }),
    celestial: Object.freeze({ color: '#ffe16a', stroke: 'rgba(255,225,106,0.9)' })
  });
  const ATTUNEMENT_DEFAULT_ICON = Object.freeze({
    color: '#8a5d27',
    bg: '#fff0cf',
    svg: '<path d="M12 3l7 5v8l-7 5-7-5V8l7-5Z" /><path d="M12 7v10M8 12h8" />'
  });
  const ATTUNEMENT_STAT_ICON_META = Object.freeze({
    powerPercent: Object.freeze({ color: '#d64545', bg: '#fff0e8', svg: '<path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />' }),
    attackDamagePercent: Object.freeze({ color: '#b83a2f', bg: '#fff0e8', svg: '<path d="M14 4l6 6-9 9H5v-6l9-9Z" /><path d="M13 5l6 6" />' }),
    bossDamagePercent: Object.freeze({ color: '#a66a00', bg: '#fff4cf', svg: '<path d="M4 9l4 3 4-7 4 7 4-3-2 9H6L4 9Z" /><path d="M7 20h10" />' }),
    eliteDamagePercent: Object.freeze({ color: '#8856c5', bg: '#f4f0fb', svg: '<path d="M12 3l8 7-8 11-8-11 8-7Z" /><path d="M4 10h16M9 10l3 11 3-11" />' }),
    areaDamage: Object.freeze({ color: '#d87925', bg: '#fff0de', svg: '<path d="M12 4v16M4 12h16M6.5 6.5l11 11M17.5 6.5l-11 11" /><circle cx="12" cy="12" r="3" />' }),
    burnDamage: Object.freeze({ color: '#e14d2a', bg: '#fff0de', svg: '<path d="M12 21c-4 0-7-3-7-7 0-3 2-5 5-8 0 3 2 4 4 2 2 2 5 4 5 7 0 3-3 6-7 6Z" /><path d="M12 18c-2 0-3-1-3-3 0-1 1-3 3-5 0 2 3 3 3 5 0 2-1 3-3 3Z" />' }),
    armorBreak: Object.freeze({ color: '#6d5642', bg: '#f7efe6', svg: '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" /><path d="M13 5l-3 6h4l-2 6" />' }),
    maxMpPercent: Object.freeze({ color: '#2f7dd6', bg: '#e9f3ff', svg: '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11Z" /><path d="M9 15h6" />' }),
    mpRecoveryPercent: Object.freeze({ color: '#2f7dd6', bg: '#e9f3ff', svg: '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11Z" /><path d="M9 14c1.5 2 4.5 2 6 0M15 14h-3v-3" />' }),
    resourceGainPercent: Object.freeze({ color: '#2f9b72', bg: '#e5f7ed', svg: '<path d="M12 3l7 4v10l-7 4-7-4V7l7-4Z" /><path d="M12 8v8M8 12h8" />' }),
    resourceMax: Object.freeze({ color: '#2f9b72', bg: '#e5f7ed', svg: '<path d="M7 7l5-3 5 3v6l-5 3-5-3V7Z" /><path d="M5 12l7 4 7-4M5 16l7 4 7-4" />' }),
    resourceCostReductionPercent: Object.freeze({ color: '#2f9b72', bg: '#e5f7ed', svg: '<path d="M7 7h10M8 12h8M10 17h4" /><path d="M12 4v12M8 13l4 4 4-4" />' }),
    shieldStrengthPercent: Object.freeze({ color: '#177645', bg: '#e5f7ed', svg: '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" /><path d="M12 7v10M8.5 11h7" />' }),
    runeDuration: Object.freeze({ color: '#5a75d6', bg: '#eef2ff', svg: '<path d="M7 4h10M8 20h8M9 4c0 4 6 4 6 8s-6 4-6 8M15 4c0 4-6 4-6 8s6 4 6 8" />' }),
    block: Object.freeze({ color: '#1b6f5f', bg: '#e2f7f2', svg: '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" /><path d="M8 12h8" />' }),
    crit: Object.freeze({ color: '#d64545', bg: '#fff0e8', svg: '<circle cx="12" cy="12" r="7" /><circle cx="12" cy="12" r="2" /><path d="M12 2v4M12 18v4M2 12h4M18 12h4" />' }),
    weakPointDuration: Object.freeze({ color: '#d87925', bg: '#fff0de', svg: '<circle cx="12" cy="10" r="5" /><path d="M12 2v3M12 15v7M8 19h8" /><path d="M9 10h6" />' }),
    markDuration: Object.freeze({ color: '#b53f8c', bg: '#fbedf6', svg: '<path d="M12 3l3 6 6 1-4.5 4.5 1 6.5-5.5-3-5.5 3 1-6.5L3 10l6-1 3-6Z" /><path d="M12 8v6" />' }),
    cooldownRecoveryPercent: Object.freeze({ color: '#4b6bd6', bg: '#eef2ff', svg: '<circle cx="12" cy="12" r="7" /><path d="M12 7v5l3 2" /><path d="M17 4v5h-5" />' }),
    buffDurationPercent: Object.freeze({ color: '#8856c5', bg: '#f4f0fb', svg: '<path d="M12 3l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8L12 3Z" /><path d="M12 8v5" />' }),
    maxHpPercent: Object.freeze({ color: '#d64545', bg: '#fff0e8', svg: '<path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10Z" /><path d="M12 8v7M8.5 11.5h7" />' }),
    defensePercent: Object.freeze({ color: '#6d7f3a', bg: '#f1f5dd', svg: '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" /><path d="M9 9h6v7H9z" />' }),
    hpRecoveryPercent: Object.freeze({ color: '#c9364a', bg: '#fff0e8', svg: '<path d="M12 20s-7-4.5-7-10a4 4 0 0 1 7-2.5A4 4 0 0 1 19 10c0 5.5-7 10-7 10Z" /><path d="M6 13h3l1.5-3 2.5 6 1.5-3H18" />' }),
    damageReductionPercent: Object.freeze({ color: '#6d7f3a', bg: '#f1f5dd', svg: '<path d="M12 3l7 3v5c0 5-3 8-7 10-4-2-7-5-7-10V6l7-3Z" /><path d="M8 11h8M10 15h4" />' }),
    potionEffectPercent: Object.freeze({ color: '#2f9b72', bg: '#e5f7ed', svg: '<path d="M9 3h6M10 3v5l-4 7c-1 2 0 5 3 5h6c3 0 4-3 3-5l-4-7V3" /><path d="M8 15h8M10 12h4" />' }),
    critDamage: Object.freeze({ color: '#c9364a', bg: '#fff0e8', svg: '<circle cx="12" cy="12" r="7" /><path d="M12 5v14M5 12h14" /><path d="M15 3l-3 5h5l-5 8" />' }),
    damageFloor: Object.freeze({ color: '#d87925', bg: '#fff0de', svg: '<path d="M5 18h14" /><path d="M12 6v9M8 10l4-4 4 4" /><path d="M7 14h10" />' }),
    trapDamage: Object.freeze({ color: '#6d5642', bg: '#f7efe6', svg: '<path d="M5 16l4-7 3 7 3-7 4 7" /><path d="M5 16h14M8 20h8" />' }),
    trapSpeed: Object.freeze({ color: '#6d5642', bg: '#f7efe6', svg: '<path d="M5 16l4-7 3 7 3-7 4 7" /><circle cx="17" cy="7" r="4" /><path d="M17 5v2l1.5 1" />' }),
    executeDamagePercent: Object.freeze({ color: '#8f2335', bg: '#fff0e8', svg: '<path d="M5 19L19 5" /><path d="M14 4h6v6" /><path d="M5 5l14 14" />' }),
    speed: Object.freeze({ color: '#2f7dd6', bg: '#e9f3ff', svg: '<path d="M5 16h7l4-7h3" /><path d="M8 20h8M4 12h5M3 8h7" />' }),
    avoid: Object.freeze({ color: '#2f7dd6', bg: '#e9f3ff', svg: '<path d="M5 16c5 0 8-3 10-9l4-2-2 5c-2 5-6 8-12 8" /><path d="M7 10c2 1 4 1 6 0" />' }),
    skillEffectPercent: Object.freeze({ color: '#8856c5', bg: '#f4f0fb', svg: '<path d="M12 3v18M4 12h16M6.5 6.5l11 11M17.5 6.5l-11 11" /><circle cx="12" cy="12" r="4" />' }),
    mobilityCooldownPercent: Object.freeze({ color: '#4b6bd6', bg: '#eef2ff', svg: '<path d="M5 15c3-6 9-7 14-3-3 1-5 3-6 6-2-2-5-3-8-3Z" /><path d="M12 5v4l3 2" />' }),
    mobilityWindowPercent: Object.freeze({ color: '#4b6bd6', bg: '#eef2ff', svg: '<path d="M5 15c3-6 9-7 14-3-3 1-5 3-6 6-2-2-5-3-8-3Z" /><path d="M12 9l4 3-4 5-4-5 4-3Z" />' }),
    hpOnHit: Object.freeze({ color: '#c9364a', bg: '#fff0e8', svg: '<path d="M12 20s-6-4-6-9a3.5 3.5 0 0 1 6-2.3A3.5 3.5 0 0 1 18 11c0 5-6 9-6 9Z" /><path d="M4 6l16 12" />' }),
    mpOnHit: Object.freeze({ color: '#2f7dd6', bg: '#e9f3ff', svg: '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11Z" /><path d="M5 5l14 14" />' }),
    buffEffectPercent: Object.freeze({ color: '#8856c5', bg: '#f4f0fb', svg: '<path d="M12 3l2 6 6 2-6 2-2 8-2-8-6-2 6-2 2-6Z" /><path d="M17 4v4M15 6h4" />' })
  });
  const CANVAS_UI_THEME = Object.freeze({
    ink: '#211816',
    muted: '#6d5846',
    mutedBlue: '#9ccfe4',
    frame: '#102033',
    frameDeep: '#06111d',
    frameSoft: '#18344d',
    headerTop: '#13283d',
    headerBottom: '#071522',
    parchment: '#f3dfbd',
    parchmentTop: '#fff0cf',
    parchmentBottom: '#d9bb88',
    parchmentSoft: '#f8e9ca',
    parchmentDim: '#d3ba8e',
    slotParchment: '#ecd7ad',
    slotParchmentTop: '#fff0cf',
    slotParchmentBottom: '#d6b985',
    slotStar: 'rgba(170,121,55,0.13)',
    slotCorner: 'rgba(137,93,39,0.28)',
    slot: '#102133',
    slotTop: '#17364f',
    slotBottom: '#081421',
    hotbarTop: '#17344d',
    hotbarBottom: '#07111d',
    gold: '#b9822f',
    goldBright: '#d8aa55',
    goldDim: '#6d4b22',
    cyan: '#4fc6ee',
    cyanSoft: 'rgba(79,198,238,0.24)',
    red: '#cc443d',
    green: '#2f985c',
    disabledFill: '#bfae93',
    disabledStroke: '#7a6b58',
    shadow: 'rgba(0,0,0,0.42)',
    line: 'rgba(112,75,29,0.42)',
    lineSoft: 'rgba(112,75,29,0.24)',
    lineStrong: 'rgba(216,170,85,0.58)'
  });

  function normalizeVisualId(value) {
    return String(value || '').trim();
  }

  function getItemRarityVisual(item, options) {
    const visuals = options && options.itemRarityVisuals || ITEM_RARITY_VISUALS;
    const rarity = item && item.rarity && visuals[item.rarity] ? item.rarity : 'Common';
    return visuals[rarity] || visuals.Common;
  }

  function itemShouldShowTierAura(item) {
    const kind = String(item && item.kind || '').toLowerCase();
    return kind === 'equipment' || kind === 'card' || (!kind && !!(item && item.slot));
  }

  function getItemStatAuraTier(strength, options) {
    const tiers = options && options.itemStatAuraTiers || ITEM_STAT_AURA_TIERS;
    return tiers.reduce((current, candidate) => (
      Number(strength) >= candidate.minPower ? candidate : current
    ), tiers[0]);
  }

  function getItemStatAuraVisual(item, options) {
    const settings = options || {};
    const getStrength = typeof settings.getItemStrength === 'function'
      ? settings.getItemStrength
      : (candidate) => Number(candidate && candidate.strength || 0) || 0;
    const visuals = settings.itemRarityVisuals || ITEM_RARITY_VISUALS;
    const strength = getStrength(item);
    const tier = getItemStatAuraTier(strength, settings);
    const visual = visuals[tier.rarity] || visuals.Common || {};
    return Object.assign({}, visual, {
      id: tier.id,
      rarity: tier.rarity,
      strength
    });
  }

  function getItemAttunementTierId(item, options) {
    const settings = options || {};
    const normalizePotential = typeof settings.normalizeItemPotential === 'function'
      ? settings.normalizeItemPotential
      : (potential) => potential && typeof potential === 'object' ? potential : null;
    const normalizeId = typeof settings.normalizeId === 'function' ? settings.normalizeId : normalizeVisualId;
    const potential = normalizePotential(item && item.potential, item);
    return normalizeId(potential && potential.tier).toLowerCase();
  }

  function getItemAttunementBorderVisual(item, options) {
    const visuals = options && options.attunementTierVisuals || ATTUNEMENT_TIER_VISUALS;
    const tierId = getItemAttunementTierId(item, options);
    return visuals[tierId] || null;
  }

  function getAttunementStatIconMeta(stat, options) {
    const iconMeta = options && options.attunementStatIconMeta || ATTUNEMENT_STAT_ICON_META;
    const fallback = options && options.attunementDefaultIcon || ATTUNEMENT_DEFAULT_ICON;
    return iconMeta[String(stat || '')] || fallback;
  }

  function getItemVisualClassNames(snapshot, item, options) {
    if (!item) return '';
    const settings = options || {};
    const isClassBlocked = typeof settings.isItemClassBlockedForSnapshot === 'function'
      ? settings.isItemClassBlockedForSnapshot
      : () => false;
    const isRequirementBlocked = typeof settings.isItemRequirementBlockedForSnapshot === 'function'
      ? settings.isItemRequirementBlockedForSnapshot
      : () => false;
    const aura = getItemStatAuraVisual(item, settings);
    const attunementTier = getItemAttunementTierId(item, settings);
    return [
      `item-aura-${aura.id}`,
      attunementTier ? `attunement-${attunementTier}` : '',
      isClassBlocked(snapshot, item) ? 'is-class-blocked' : '',
      isRequirementBlocked(snapshot, item) ? 'is-requirement-blocked' : ''
    ].filter(Boolean).join(' ');
  }

  function getCanvasIconBadgeMetadata(label, x, y, size, fill, stroke, uiTheme) {
    const theme = uiTheme || CANVAS_UI_THEME;
    return {
      slot: {
        x,
        y,
        w: size,
        h: size,
        options: {
          fill: fill || theme.parchmentSoft,
          stroke: stroke || theme.lineSoft,
          radius: 8
        }
      },
      text: {
        value: label,
        x: x + size / 2,
        y: y + size / 2,
        color: theme.ink,
        font: '800 10px system-ui',
        align: 'center',
        baseline: 'middle'
      }
    };
  }

  function drawCanvasIconBadge(ctx, label, x, y, size, fill, stroke, options) {
    const settings = options || {};
    const drawCanvasUiSlot = settings.drawCanvasUiSlot;
    const drawCanvasText = settings.drawCanvasText;
    const badge = getCanvasIconBadgeMetadata(label, x, y, size, fill, stroke, settings.canvasUiTheme);
    if (typeof drawCanvasUiSlot !== 'function' || typeof drawCanvasText !== 'function') return false;
    drawCanvasUiSlot(badge.slot);
    drawCanvasText(badge.text);
    return true;
  }

  function createVisualUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      itemRarityVisuals: settings.itemRarityVisuals || ITEM_RARITY_VISUALS,
      itemStatAuraTiers: settings.itemStatAuraTiers || ITEM_STAT_AURA_TIERS,
      attunementTierVisuals: settings.attunementTierVisuals || ATTUNEMENT_TIER_VISUALS,
      attunementStatIconMeta: settings.attunementStatIconMeta || ATTUNEMENT_STAT_ICON_META,
      attunementDefaultIcon: settings.attunementDefaultIcon || ATTUNEMENT_DEFAULT_ICON,
      canvasUiTheme: settings.canvasUiTheme || CANVAS_UI_THEME,
      getItemStrength: settings.getItemStrength,
      normalizeId: settings.normalizeId,
      normalizeItemPotential: settings.normalizeItemPotential,
      isItemClassBlockedForSnapshot: settings.isItemClassBlockedForSnapshot,
      isItemRequirementBlockedForSnapshot: settings.isItemRequirementBlockedForSnapshot
    });
    const mergeOptions = (override) => Object.assign({}, helperOptions, override || {});
    return Object.freeze({
      getItemRarityVisuals,
      getItemRarityVisual: (item, override) => getItemRarityVisual(item, mergeOptions(override)),
      itemShouldShowTierAura,
      getItemStatAuraTier: (strength, override) => getItemStatAuraTier(strength, mergeOptions(override)),
      getItemStatAuraVisual: (item, override) => getItemStatAuraVisual(item, mergeOptions(override)),
      getItemAttunementTierId: (item, override) => getItemAttunementTierId(item, mergeOptions(override)),
      getItemAttunementBorderVisual: (item, override) => getItemAttunementBorderVisual(item, mergeOptions(override)),
      getAttunementStatIconMeta: (stat, override) => getAttunementStatIconMeta(stat, mergeOptions(override)),
      getItemVisualClassNames: (snapshot, item, override) => getItemVisualClassNames(snapshot, item, mergeOptions(override)),
      getCanvasIconBadgeMetadata: (label, x, y, size, fill, stroke, uiTheme) => getCanvasIconBadgeMetadata(label, x, y, size, fill, stroke, uiTheme || helperOptions.canvasUiTheme),
      drawCanvasIconBadge: (ctx, label, x, y, size, fill, stroke, drawOptions) => drawCanvasIconBadge(ctx, label, x, y, size, fill, stroke, Object.assign({}, drawOptions || {}, {
        canvasUiTheme: helperOptions.canvasUiTheme
      }))
    });
  }

  const api = {
    DEFAULT_ITEM_RARITY_VISUALS,
    ITEM_RARITY_VISUALS,
    RARITY_ORDER,
    ITEM_STAT_AURA_TIERS,
    ATTUNEMENT_TIER_VISUALS,
    ATTUNEMENT_DEFAULT_ICON,
    ATTUNEMENT_STAT_ICON_META,
    CANVAS_UI_THEME,
    getItemRarityVisuals,
    getItemRarityVisual,
    itemShouldShowTierAura,
    getItemStatAuraTier,
    getItemStatAuraVisual,
    getItemAttunementTierId,
    getItemAttunementBorderVisual,
    getAttunementStatIconMeta,
    getItemVisualClassNames,
    getCanvasIconBadgeMetadata,
    drawCanvasIconBadge,
    createVisualUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.visuals = Object.assign({}, modules.visuals || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
