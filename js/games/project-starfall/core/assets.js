(function initProjectStarfallCoreAssets(global) {
  'use strict';

  const DEFAULT_ASSET_FRAME_CACHE_LIMIT = 512;
  const defaultAssetFrameCache = new Map();
  const CLASSIC_ASSET_REQUEST_VERSION = 'classic-eb8de4e1';
  const CLASSIC_ASSET_REQUEST_PATH = /^\/?img\/project-starfall\/(?:characters\/(?:generic-player|fighter|guardian|berserker|duelist|mage|fire-mage|rune-mage|storm-mage|archer|sniper|trapper|beast-archer)\.png|animations\/players\/(?:generic-player|fighter|guardian|berserker|duelist|mage|fire-mage|rune-mage|storm-mage|archer|sniper|trapper|beast-archer)-sheet\.png|equipment-atlases\/[^/]+-atlas\.png|environment\/structures\/town-landmarks\.png|maps\/(?:starfall-crossing|greenroot-meadow|eclipse-throne)\.webp|ui\/start-screen\.(?:png|webp|avif))$/;

  function getAssetRequestUrl(assetPath) {
    const value = String(assetPath || '').trim();
    const hashIndex = value.indexOf('#');
    const fragment = hashIndex < 0 ? '' : value.slice(hashIndex);
    const source = hashIndex < 0 ? value : value.slice(0, hashIndex);
    const queryIndex = source.indexOf('?');
    const path = queryIndex < 0 ? source : source.slice(0, queryIndex);
    if (!CLASSIC_ASSET_REQUEST_PATH.test(path)) return value;
    // Keep canonical paths and frame fragments intact; revise only network URLs.
    const parameters = queryIndex < 0 ? [] : source.slice(queryIndex + 1).split('&')
      .filter((parameter) => parameter && parameter.split('=')[0] !== 'v');
    parameters.push(`v=${CLASSIC_ASSET_REQUEST_VERSION}`);
    return `${path}?${parameters.join('&')}${fragment}`;
  }

  function getAssetFrameCacheKey(value, includeSheetSize) {
    return includeSheetSize ? value : `basic:${value}`;
  }

  function parseAssetFrame(assetPath, options) {
    const settings = options || {};
    const includeSheetSize = settings.includeSheetSize !== false;
    const cache = settings.cache === false ? null : settings.cache || defaultAssetFrameCache;
    const cacheLimit = Math.max(1, Math.floor(Number(settings.cacheLimit || DEFAULT_ASSET_FRAME_CACHE_LIMIT) || DEFAULT_ASSET_FRAME_CACHE_LIMIT));
    const value = String(assetPath || '').trim();
    if (!value) return null;
    const cacheKey = getAssetFrameCacheKey(value, includeSheetSize);
    if (cache && cache.has(cacheKey)) return cache.get(cacheKey);
    const hashIndex = value.indexOf('#');
    const path = hashIndex < 0 ? value : value.slice(0, hashIndex);
    const fragment = hashIndex < 0 ? '' : value.slice(hashIndex + 1);
    const match = fragment.match(/(?:^|[;&])(?:frame|xywh)=([0-9.,-]+)/);
    let frame = { path };
    if (match) {
      const parts = match[1].split(',').map((part) => Number(part));
      const sx = Math.max(0, Math.floor(parts[0]) || 0);
      const sy = Math.max(0, Math.floor(parts[1]) || 0);
      const sw = Math.max(1, Math.floor(parts[2]) || 1);
      const sh = Math.max(1, Math.floor(parts[3]) || 1);
      frame = { path, sx, sy, sw, sh };
      if (includeSheetSize) {
        frame.sheetWidth = Math.max(sw, Math.floor(parts[4]) || sw);
        frame.sheetHeight = Math.max(sh, Math.floor(parts[5]) || sh);
      }
    }
    if (cache) {
      if (cache.size > cacheLimit) cache.clear();
      cache.set(cacheKey, frame);
    }
    return frame;
  }

  function createAssetFrameParser(options) {
    const settings = options || {};
    return (assetPath) => parseAssetFrame(assetPath, settings);
  }

  function getAssetSourcePath(assetPath, options) {
    const frame = parseAssetFrame(assetPath, options);
    return frame && frame.path || '';
  }

  function createAssetSourcePathGetter(options) {
    const parse = createAssetFrameParser(options || {});
    return (assetPath) => {
      const frame = parse(assetPath);
      return frame && frame.path || '';
    };
  }

  function drawAssetFrame(ctx, image, assetPath, x, y, w, h, options) {
    if (!ctx || !image) return false;
    const settings = options || {};
    const parse = typeof settings.parseAssetFrame === 'function' ? settings.parseAssetFrame : parseAssetFrame;
    const frame = parse(assetPath);
    if (frame && frame.sw && frame.sh) {
      ctx.drawImage(image, frame.sx, frame.sy, frame.sw, frame.sh, x, y, w, h);
      return true;
    }
    ctx.drawImage(image, x, y, w, h);
    return true;
  }

  function createAssetFrameDrawer(options) {
    const settings = options || {};
    return (ctx, image, assetPath, x, y, w, h) => drawAssetFrame(ctx, image, assetPath, x, y, w, h, settings);
  }

  function isLegacyIndividualItemAsset(assetPath) {
    const path = String(assetPath || '').split('#')[0];
    return /^img\/project-starfall\/items\/(?!sheets\/|source\/)[^/]+\.png$/i.test(path);
  }

  function getAssetReadyCacheKey(progress) {
    return progress
      ? [progress.settled || 0, progress.loaded || 0, progress.failed || 0, progress.total || 0].join(':')
      : 'no-assets';
  }

  function isAssetPrewarmReady(progress) {
    if (!progress || !Number(progress.total || 0)) return true;
    return Number(progress.settled || 0) >= Number(progress.total || 0);
  }

  const api = {
    DEFAULT_ASSET_FRAME_CACHE_LIMIT,
    CLASSIC_ASSET_REQUEST_VERSION,
    getAssetRequestUrl,
    parseAssetFrame,
    createAssetFrameParser,
    getAssetSourcePath,
    createAssetSourcePathGetter,
    drawAssetFrame,
    createAssetFrameDrawer,
    isLegacyIndividualItemAsset,
    getAssetReadyCacheKey,
    isAssetPrewarmReady
  };

  const core = global.ProjectStarfallCore || {};
  Object.assign(core, api);
  global.ProjectStarfallCore = core;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
