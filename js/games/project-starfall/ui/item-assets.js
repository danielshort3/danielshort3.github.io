(function initProjectStarfallUiItemAssets(global) {
  'use strict';

  const Data = global.ProjectStarfallData || {};
  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreAssets = (typeof require === 'function' ? require('../core/assets.js') : null) || global.ProjectStarfallCore || {};
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const isLegacyIndividualItemAsset = CoreAssets.isLegacyIndividualItemAsset || function isLegacyIndividualItemAssetFallback(assetPath) {
    const path = String(assetPath || '').split('#')[0];
    return /^img\/project-starfall\/items\/(?!sheets\/|source\/)[^/]+\.png$/i.test(path);
  };

  function getItemVisualAssetId(item, options) {
    const settings = options || {};
    const data = settings.data || Data;
    const normalize = settings.normalizeId || normalizeId;
    const visuals = data.EQUIPMENT_VISUALS || {};
    const candidates = [
      item && item.visualId,
      item && item.id,
      normalize(item && item.slot) === 'head' ? 'fieldguard_helm' : '',
      normalize(item && item.slot) === 'gloves' ? 'trailwoven_gloves' : '',
      normalize(item && item.slot) === 'amulet' ? 'plain_ring' : ''
    ];
    for (const candidate of candidates) {
      const id = normalize(candidate);
      const visual = id && visuals[id];
      if (visual && visual.assetId) return visual.assetId;
    }
    return '';
  }

  function getItemAsset(item, options) {
    if (!item) return '';
    const settings = options || {};
    const data = settings.data || Data;
    const getMaterialAssetId = settings.getMaterialAssetId || ((materialId) => String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase());
    const isLegacyAsset = settings.isLegacyIndividualItemAsset || isLegacyIndividualItemAsset;
    const assets = data.ITEM_ASSETS || {};
    const kind = String(item.kind || '');
    const ids = (kind === 'currency'
      ? [
          item.assetId,
          item.visualId,
          item.id,
          item.itemId,
          'coins'
        ]
      : [
          item.id,
          item.itemId,
          item.consumableId,
          item.materialId ? getMaterialAssetId(item.materialId) : '',
          kind === 'material' && item.id ? getMaterialAssetId(item.id) : '',
          item.assetId,
          getItemVisualAssetId(item, settings),
          item.visualId
        ]).filter(Boolean);
    for (const id of ids) {
      if (assets[id]) return assets[id];
    }
    if (item.asset && !isLegacyAsset(item.asset)) return item.asset;
    return '';
  }

  function getCashShopItemAsset(item, options) {
    const settings = options || {};
    const data = settings.data || Data;
    const direct = getItemAsset(item, settings);
    if (direct) return direct;
    const consumables = item && item.reward && item.reward.consumables || {};
    const consumableId = Object.keys(consumables).find((id) => Number(consumables[id] || 0) > 0);
    return consumableId && data.ITEM_ASSETS ? data.ITEM_ASSETS[consumableId] || '' : '';
  }

  function escapeHtmlFallback(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderGearIconMarkup(item, fallback, options) {
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const renderAssetImage = settings.renderAssetImage || function renderAssetImageFallback() { return ''; };
    const getVisualClassNames = settings.getItemVisualClassNames || function getItemVisualClassNamesFallback() { return ''; };
    const getSlotMeta = settings.getSlotMeta || function getSlotMetaFallback() { return { icon: '' }; };
    const renderClassBadge = settings.renderItemClassBadge || function renderItemClassBadgeFallback() { return ''; };
    const tag = settings.tag === 'div' ? 'div' : 'span';
    const classes = [
      'project-starfall-gear-icon',
      settings.className || '',
      item ? settings.visualClasses || getVisualClassNames(settings.snapshot, item) : ''
    ].filter(Boolean).join(' ');
    const fallbackText = fallback || (item ? getSlotMeta(item.slot).icon : '');
    const content = item
      ? renderAssetImage(getItemAsset(item, settings), '', 'project-starfall-item-art') || escapeHtml(fallbackText)
      : escapeHtml(fallbackText);
    const classBadge = item && settings.showClassBadge === true ? renderClassBadge(item, settings.snapshot) : '';
    return `<${tag} class="${escapeHtml(classes)}" aria-hidden="true">${content}${settings.extra || ''}${classBadge}</${tag}>`;
  }

  function getCanvasItemAura(image, size, visual, assetPath, options) {
    const settings = options || {};
    const documentRef = settings.document || global.document;
    const drawAssetFrame = settings.drawAssetFrame;
    if (!image || !size || !documentRef || typeof drawAssetFrame !== 'function') return null;
    const normalizedSize = Math.max(1, Math.round(size));
    const color = String(visual && visual.color || '#d8e5ec');
    const key = `${assetPath || image.src || image.currentSrc || 'asset'}:${normalizedSize}:${color}`;
    const cache = settings.cache;
    if (cache && typeof cache.has === 'function' && cache.has(key)) return cache.get(key);
    const pad = Math.max(8, Math.ceil(normalizedSize * 0.22));
    const canvas = documentRef.createElement && documentRef.createElement('canvas');
    if (!canvas) return null;
    canvas.width = normalizedSize + pad * 2;
    canvas.height = normalizedSize + pad * 2;
    const auraCtx = canvas.getContext && canvas.getContext('2d');
    if (!auraCtx) return null;
    auraCtx.clearRect(0, 0, canvas.width, canvas.height);
    drawAssetFrame(auraCtx, image, assetPath, pad, pad, normalizedSize, normalizedSize);
    auraCtx.globalCompositeOperation = 'source-in';
    auraCtx.fillStyle = color;
    auraCtx.fillRect(0, 0, canvas.width, canvas.height);
    auraCtx.globalCompositeOperation = 'source-over';
    const aura = { canvas, pad };
    if (cache && typeof cache.set === 'function') {
      cache.set(key, aura);
      const maxCacheSize = Number(settings.maxCacheSize || 180);
      if (cache.size > maxCacheSize && typeof cache.keys === 'function' && typeof cache.delete === 'function') {
        cache.delete(cache.keys().next().value);
      }
      return cache.get(key);
    }
    return aura;
  }

  function drawCanvasItemAura(ctx, item, x, y, size, image, assetPath, options) {
    const settings = options || {};
    if (!item || !image) return false;
    const visual = typeof settings.getItemStatAuraVisual === 'function' ? settings.getItemStatAuraVisual(item) : {};
    const color = visual.color || '#d8e5ec';
    const aura = typeof settings.getItemAuraCanvas === 'function'
      ? settings.getItemAuraCanvas(image, size, visual, assetPath)
      : getCanvasItemAura(image, size, visual, assetPath, settings);
    if (!ctx || !aura || !aura.canvas) return false;
    const now = typeof settings.now === 'function' ? settings.now() : Date.now();
    const pulse = visual.pulse ? 0.5 + Math.sin(now / 240) * 0.5 : 0;
    const alpha = Math.min(1, Number(visual.alpha || 0.5) + pulse * Number(visual.pulse || 0));
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowColor = color;
    ctx.shadowBlur = Number(visual.glow || 7) + pulse * 6;
    ctx.drawImage(aura.canvas, x - aura.pad, y - aura.pad);
    ctx.globalAlpha = Math.max(0.24, alpha * 0.52);
    [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dx, dy]) => {
      ctx.drawImage(aura.canvas, x - aura.pad + dx, y - aura.pad + dy);
    });
    ctx.restore();
    return true;
  }

  function drawCanvasItemIcon(ctx, item, x, y, size, options) {
    const settings = options || {};
    const asset = getItemAsset(item, settings);
    const image = item && typeof settings.getAsset === 'function' ? settings.getAsset(asset) : null;
    const showAura = Object.prototype.hasOwnProperty.call(settings, 'showAura')
      ? !!settings.showAura
      : typeof settings.itemShouldShowTierAura === 'function' && settings.itemShouldShowTierAura(item);
    if (settings.frame !== false && typeof settings.drawCanvasUiSlot === 'function') {
      const visual = item && typeof settings.getItemRarityVisual === 'function' ? settings.getItemRarityVisual(item) : null;
      settings.drawCanvasUiSlot({
        x,
        y,
        w: size,
        h: size,
        options: {
          radius: Math.min(8, Math.max(5, size / 6)),
          stroke: visual && visual.color && typeof settings.colorWithAlpha === 'function'
            ? settings.colorWithAlpha(visual.color, 0.58)
            : settings.defaultLine
        }
      });
    }
    if (showAura && image && typeof settings.drawItemAura === 'function') {
      settings.drawItemAura({ item, x, y, size, image, asset });
    }
    if (image && typeof settings.drawAssetFrame === 'function') {
      const inset = settings.frame === false ? 0 : Math.max(2, Math.floor(size * 0.07));
      settings.drawAssetFrame(ctx, image, asset, x + inset, y + inset, Math.max(1, size - inset * 2), Math.max(1, size - inset * 2));
      return true;
    }
    return false;
  }

  function drawCanvasMaterialIcon(ctx, requirement, x, y, size, options) {
    const settings = options || {};
    const asset = getItemAsset(requirement, settings);
    const image = requirement && typeof settings.getAsset === 'function' ? settings.getAsset(asset) : null;
    const drawAssetFrame = settings.drawAssetFrame;
    if (!image || typeof drawAssetFrame !== 'function') return false;
    drawAssetFrame(ctx, image, asset, x, y, size, size);
    return true;
  }

  function createItemAssetUiHelpers(options) {
    const settings = options || {};
    const helperOptions = Object.freeze({
      data: settings.data || Data,
      getMaterialAssetId: settings.getMaterialAssetId || ((materialId) => String(materialId || '').replace(/([A-Z])/g, '_$1').toLowerCase()),
      isLegacyIndividualItemAsset: settings.isLegacyIndividualItemAsset || isLegacyIndividualItemAsset,
      normalizeId: settings.normalizeId || normalizeId
    });
    const mergeOptions = (override) => Object.assign({}, helperOptions, override || {});
    return Object.freeze({
      getItemVisualAssetId: (item, override) => getItemVisualAssetId(item, mergeOptions(override)),
      getItemAsset: (item, override) => getItemAsset(item, mergeOptions(override)),
      getCashShopItemAsset: (item, override) => getCashShopItemAsset(item, mergeOptions(override)),
      renderGearIconMarkup: (item, fallback, renderOptions) => renderGearIconMarkup(item, fallback, mergeOptions(renderOptions)),
      getCanvasItemAura: (image, size, visual, assetPath, drawOptions) => getCanvasItemAura(image, size, visual, assetPath, mergeOptions(drawOptions)),
      drawCanvasItemAura: (ctx, item, x, y, size, image, assetPath, drawOptions) => drawCanvasItemAura(ctx, item, x, y, size, image, assetPath, mergeOptions(drawOptions)),
      drawCanvasItemIcon: (ctx, item, x, y, size, drawOptions) => drawCanvasItemIcon(ctx, item, x, y, size, mergeOptions(drawOptions)),
      drawCanvasMaterialIcon: (ctx, requirement, x, y, size, drawOptions) => drawCanvasMaterialIcon(ctx, requirement, x, y, size, mergeOptions(drawOptions))
    });
  }

  const api = {
    getItemVisualAssetId,
    getItemAsset,
    getCashShopItemAsset,
    renderGearIconMarkup,
    getCanvasItemAura,
    drawCanvasItemAura,
    drawCanvasItemIcon,
    drawCanvasMaterialIcon,
    createItemAssetUiHelpers
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.itemAssets = Object.assign({}, modules.itemAssets || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
