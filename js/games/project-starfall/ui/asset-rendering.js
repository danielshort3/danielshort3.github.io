(function initProjectStarfallUiAssetRendering(global) {
  'use strict';

  const CoreAssets = (typeof require === 'function' ? require('../core/assets.js') : null) || global.ProjectStarfallCore || {};
  const ASSET_IMAGE_MARKUP_CACHE_LIMIT = 1024;
  const assetImageMarkupCache = new Map();
  const parseAssetFrameFallback = CoreAssets.parseAssetFrame || function parseAssetFrameFallback(assetPath) {
    const value = String(assetPath || '').trim();
    if (!value) return null;
    const hashIndex = value.indexOf('#');
    const path = hashIndex < 0 ? value : value.slice(0, hashIndex);
    const fragment = hashIndex < 0 ? '' : value.slice(hashIndex + 1);
    const match = fragment.match(/(?:^|[;&])(?:frame|xywh)=([0-9.,-]+)/);
    if (!match) return { path };
    const parts = match[1].split(',').map((part) => Number(part));
    const sx = Math.max(0, Math.floor(parts[0]) || 0);
    const sy = Math.max(0, Math.floor(parts[1]) || 0);
    const sw = Math.max(1, Math.floor(parts[2]) || 1);
    const sh = Math.max(1, Math.floor(parts[3]) || 1);
    const sheetWidth = Math.max(sw, Math.floor(parts[4]) || sw);
    const sheetHeight = Math.max(sh, Math.floor(parts[5]) || sh);
    return { path, sx, sy, sw, sh, sheetWidth, sheetHeight };
  };

  function escapeHtmlFallback(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function getAssetFrameStyle(frame, options) {
    const escapeHtml = options && options.escapeHtml || escapeHtmlFallback;
    if (!frame || !frame.sw || !frame.sh) return '';
    const sizeX = frame.sheetWidth / frame.sw * 100;
    const sizeY = frame.sheetHeight / frame.sh * 100;
    const posX = frame.sheetWidth > frame.sw ? frame.sx / (frame.sheetWidth - frame.sw) * 100 : 0;
    const posY = frame.sheetHeight > frame.sh ? frame.sy / (frame.sheetHeight - frame.sh) * 100 : 0;
    return [
      `background-image:url('${escapeHtml(frame.path).replace(/'/g, '%27')}')`,
      `background-size:${sizeX}% ${sizeY}%`,
      `background-position:${posX}% ${posY}%`
    ].join(';');
  }

  function renderAssetImage(assetPath, alt, className, options) {
    const settings = options || {};
    const escapeHtml = settings.escapeHtml || escapeHtmlFallback;
    const parseAssetFrame = settings.parseAssetFrame || parseAssetFrameFallback;
    const cache = settings.cache === false ? null : settings.cache || assetImageMarkupCache;
    const cacheLimit = Math.max(1, Math.floor(Number(settings.cacheLimit || ASSET_IMAGE_MARKUP_CACHE_LIMIT) || ASSET_IMAGE_MARKUP_CACHE_LIMIT));
    const path = String(assetPath || '').trim();
    if (!path) return '';
    const cacheKey = `${path}|${alt || ''}|${className || ''}`;
    if (cache && cache.has(cacheKey)) return cache.get(cacheKey);
    const frame = parseAssetFrame(path);
    let markup = '';
    if (frame && frame.sw && frame.sh) {
      const classes = [className || '', 'project-starfall-sprite-art'].filter(Boolean).join(' ');
      markup = `<span class="${escapeHtml(classes)}" role="img" aria-label="${escapeHtml(alt || '')}" style="${getAssetFrameStyle(frame, settings)}"></span>`;
    } else {
      markup = `<img class="${escapeHtml(className)}" src="${escapeHtml(frame && frame.path || path)}" alt="${escapeHtml(alt || '')}" loading="lazy" decoding="async">`;
    }
    if (cache) {
      if (cache.size > cacheLimit) cache.clear();
      cache.set(cacheKey, markup);
    }
    return markup;
  }

  function createAssetRenderingUiHelpers() {
    return Object.freeze({
      getAssetFrameStyle,
      renderAssetImage
    });
  }

  const api = {
    ASSET_IMAGE_MARKUP_CACHE_LIMIT,
    createAssetRenderingUiHelpers,
    getAssetFrameStyle,
    renderAssetImage
  };

  const modules = global.ProjectStarfallUiModules || {};
  modules.assetRendering = Object.assign({}, modules.assetRendering || {}, api);
  global.ProjectStarfallUiModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
