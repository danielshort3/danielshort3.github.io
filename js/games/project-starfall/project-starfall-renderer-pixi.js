(function initProjectStarfallPixiRenderer(global) {
  'use strict';

  const CoreMath = (typeof require === 'function' ? require('./core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreGeometry = (typeof require === 'function' ? require('./core/geometry.js') : null) || global.ProjectStarfallCore || {};
  const CoreAssets = (typeof require === 'function' ? require('./core/assets.js') : null) || global.ProjectStarfallCore || {};
  const EngineVisuals = (typeof require === 'function' ? require('./engine/visuals.js') : null) || global.ProjectStarfallEngineModules && global.ProjectStarfallEngineModules.visuals || {};
  const hashString = CoreMath.hashString || function hashStringFallback(value) {
    const text = String(value || '');
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  };
  const seededUnit = CoreMath.seededUnit || function seededUnitFallback(seed, salt) {
    let value = hashString(`${seed}:${salt}`);
    value ^= value << 13;
    value ^= value >>> 17;
    value ^= value << 5;
    return ((value >>> 0) % 10000) / 10000;
  };
  const seededPick = CoreMath.seededPick || function seededPickFallback(items, seed, salt) {
    const options = (items || []).filter(Boolean);
    if (!options.length) return '';
    return options[Math.floor(seededUnit(seed, salt) * options.length) % options.length];
  };
  const rectsOverlap = CoreMath.rectsOverlap || function rectsOverlapFallback(a, b) {
    return !!(a && b &&
      a.x < b.x + b.w &&
      a.x + a.w > b.x &&
      a.y < b.y + b.h &&
      a.y + a.h > b.y);
  };
  const positiveModulo = CoreMath.positiveModulo || function positiveModuloFallback(value, divisor) {
    const base = Math.max(1, Number(divisor || 1));
    return ((Number(value || 0) % base) + base) % base;
  };
  const isSlopePlatform = CoreGeometry.isSlopePlatform || function isSlopePlatformFallback(platform) {
    return !!(platform && platform.shape === 'slope' && Number.isFinite(Number(platform.y2)));
  };
  const getPlatformSurfaceY = CoreGeometry.getPlatformSurfaceY || function getPlatformSurfaceYFallback(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!isSlopePlatform(platform)) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  };
  const getPlatformBottomY = CoreGeometry.getPlatformBottomY || function getPlatformBottomYFallback(platform) {
    if (!platform) return 0;
    const surfaceBottom = isSlopePlatform(platform) ? Math.max(Number(platform.y || 0), Number(platform.y2 || platform.y || 0)) : Number(platform.y || 0);
    return surfaceBottom + Math.max(1, Number(platform.h || 0));
  };
  const isRectInBounds = CoreGeometry.isRectInBounds || function isRectInBoundsFallback(rect, bounds, padding) {
    if (!rect || !bounds) return true;
    const pad = Number(padding || 0);
    return rect.x + rect.w >= Number(bounds.left || 0) - pad &&
      rect.x <= Number(bounds.right || 0) + pad &&
      rect.y + rect.h >= Number(bounds.top || 0) - pad &&
      rect.y <= Number(bounds.bottom || 0) + pad;
  };
  const isPointInBounds = CoreGeometry.isPointInBounds || function isPointInBoundsFallback(point, bounds, padding) {
    if (!point || !bounds) return true;
    const pad = Number(padding || 0);
    return Number(point.x || 0) >= Number(bounds.left || 0) - pad &&
      Number(point.x || 0) <= Number(bounds.right || 0) + pad &&
      Number(point.y || 0) >= Number(bounds.top || 0) - pad &&
      Number(point.y || 0) <= Number(bounds.bottom || 0) + pad;
  };
  const FALLBACK_COLOR = 0xffffff;
  const MAP_BACKGROUND_PARALLAX = 0.42;
  const MAP_BACKGROUND_TILE_OVERLAP_PX = 2;
  const FRAME_TEXTURE_CACHE_LIMIT = 420;
  const BASE_TEXTURE_CACHE_LIMIT = 256;
  const TRIMMED_TEXTURE_CACHE_LIMIT = 520;
  const COMPOSITE_TEXTURE_CACHE_LIMIT = 180;
  const RIG_TEXTURE_CACHE_LIMIT = 220;
  const ENVIRONMENT_TEXTURE_CACHE_LIMIT = 360;
  const MAP_SCENERY_PLACEMENT_CACHE_LIMIT = 24;
  const ASSET_FRAME_CACHE_LIMIT = CoreAssets.DEFAULT_ASSET_FRAME_CACHE_LIMIT || 512;
  const PLAYER_SPRITE_REGISTRATION = EngineVisuals.PLAYER_SPRITE_REGISTRATION || Object.freeze({
    originX: 80,
    groundY: 154,
    authoredBodyHeight: 143
  });
  const ENEMY_SPRITE_REGISTRATION = EngineVisuals.ENEMY_SPRITE_REGISTRATION || Object.freeze({
    originX: 64,
    groundY: 118,
    authoredBodyHeight: 102
  });
  const assetFrameCache = new Map();
  const MAP_DERIVED_CACHE_PATHS = Object.freeze([
    'img/project-starfall/maps/',
    'img/project-starfall/animations/enemies/',
    'img/project-starfall/animations/enemy-projectiles/',
    'img/project-starfall/enemies/',
    'img/project-starfall/environment/'
  ]);
  const RIG_TEXTURE_SCALE = 3;
  const RUNE_FIELD_REFILL_PULSE_SECONDS = 0.35;
  const HORIZONTAL_PLAYER_PROJECTILE_TYPES = Object.freeze(['magic', 'fire', 'rune', 'lightning']);
  const DEFAULT_ENVIRONMENT_TERRAIN_CELLS = Object.freeze({
    groundLeft: 0,
    groundMid: Object.freeze([1, 2]),
    groundRight: 3,
    platformLeft: 4,
    platformMid: Object.freeze([5, 6]),
    platformRight: 7,
    body: Object.freeze([8, 9, 10, 11]),
    bodyDeep: Object.freeze([12, 13, 14, 15]),
    underside: Object.freeze([16, 17, 18, 19]),
    left: 20,
    right: 21,
    cap: 22,
    detail: Object.freeze([23, 24, 25, 26]),
    undersideLong: Object.freeze([27, 28, 29, 30]),
    shadow: 31,
    top: Object.freeze([4, 5, 6, 7]),
    topAlt: Object.freeze([1, 2])
  });
  const DEFAULT_ENVIRONMENT_PROP_CELLS = Object.freeze({
    grass: 0,
    bush: 1,
    tree: 2,
    rock: 3,
    flower: 4,
    small: 5,
    tall: 6,
    crate: 7,
    crystal: 8,
    vine: 9,
    sign: 10,
    glow: 11
  });
  const DEFAULT_ENVIRONMENT_STRUCTURE_CELLS = Object.freeze({
    starfallGuildHall: 0,
    rustcoilWorkshop: 1,
    cinderForge: 2,
    frostfenLodge: 3,
    stormbreakGate: 4,
    astralObservatory: 5,
    marketAwning: 6,
    lanternArch: 7
  });
  const DEFAULT_ENVIRONMENT_REAR_PROP_KINDS = Object.freeze(['tree', 'tall', 'vine', 'crystal', 'sign']);
  const DEFAULT_ENVIRONMENT_FRONT_PROP_KINDS = Object.freeze(['grass', 'bush', 'flower', 'rock', 'small', 'crate', 'glow']);
  const DEFAULT_ENVIRONMENT_UPPER_FRONT_PROP_KINDS = Object.freeze(['grass', 'flower', 'rock', 'glow']);
  const DEFAULT_ENVIRONMENT_VISIBILITY = Object.freeze({
    maxFootOverlapPx: 0,
    combatClearancePx: 72,
    groundOnlyTallProps: true,
    upperPlatformPropScale: 0.6,
    rearDensityScale: 0.72,
    frontDensityScale: 0.26,
    maxUpperPropHeight: 20,
    maxFrontPropHeight: 32
  });
  const DEFAULT_ENVIRONMENT_TERRAIN_STYLE = Object.freeze({
    topHeight: 20,
    groundTopHeight: 24,
    platformBodyDepth: 30,
    groundBodyDepth: 64,
    overhang: 8,
    groundOverhang: 0,
    edgeWidth: 36,
    undersideHeight: 14,
    undersideJitter: 8,
    detailDensity: 0.16,
    bodyAlpha: 0.94
  });
  const DAMAGE_TEXT_POOL_STYLE = Object.freeze({
    fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    fontSize: 18,
    fontWeight: '900',
    fill: '#fff4c7',
    stroke: { color: 'rgba(9,31,59,0.92)', width: 4 },
    strokeThickness: 4,
    align: 'center'
  });

  function createRenderer(options) {
    return new ProjectStarfallPixiRenderer(options || {});
  }

  const parseAssetFrame = CoreAssets.createAssetFrameParser
    ? CoreAssets.createAssetFrameParser({ cache: assetFrameCache, cacheLimit: ASSET_FRAME_CACHE_LIMIT, includeSheetSize: false })
    : function parseAssetFrameFallback(assetPath) {
        const value = String(assetPath || '').trim();
        if (!value) return null;
        if (assetFrameCache.has(value)) return assetFrameCache.get(value);
        const hashIndex = value.indexOf('#');
        if (hashIndex < 0) {
          const frame = { path: value };
          if (assetFrameCache.size > ASSET_FRAME_CACHE_LIMIT) assetFrameCache.clear();
          assetFrameCache.set(value, frame);
          return frame;
        }
        const path = value.slice(0, hashIndex);
        const fragment = value.slice(hashIndex + 1);
        const match = fragment.match(/(?:^|[;&])(?:frame|xywh)=([0-9.,-]+)/);
        if (!match) {
          const frame = { path };
          if (assetFrameCache.size > ASSET_FRAME_CACHE_LIMIT) assetFrameCache.clear();
          assetFrameCache.set(value, frame);
          return frame;
        }
        const parts = match[1].split(',').map((part) => Number(part));
        const frame = {
          path,
          sx: Math.max(0, Math.floor(parts[0]) || 0),
          sy: Math.max(0, Math.floor(parts[1]) || 0),
          sw: Math.max(1, Math.floor(parts[2]) || 1),
          sh: Math.max(1, Math.floor(parts[3]) || 1)
        };
        if (assetFrameCache.size > ASSET_FRAME_CACHE_LIMIT) assetFrameCache.clear();
        assetFrameCache.set(value, frame);
        return frame;
      };

  const getAssetSourcePath = CoreAssets.createAssetSourcePathGetter
    ? CoreAssets.createAssetSourcePathGetter({ cache: assetFrameCache, cacheLimit: ASSET_FRAME_CACHE_LIMIT, includeSheetSize: false })
    : function getAssetSourcePathFallback(assetPath) {
        const frame = parseAssetFrame(assetPath);
        return frame && frame.path || '';
      };

  class ProjectStarfallPixiRenderer {
    constructor(options) {
      this.PIXI = options.PIXI || global.PIXI;
      this.rigRenderer = options.rigRenderer || global.ProjectStarfallRig || null;
      this.anchorCanvas = options.anchorCanvas || null;
      this.data = options.data || {};
      this.assetBackupPaths = this.data && this.data.ASSET_BACKUP_PATHS || {};
      this.assetPaths = Array.isArray(options.assetPaths) ? options.assetPaths.slice() : [];
      this.getRenderScale = typeof options.getRenderScale === 'function'
        ? options.getRenderScale
        : () => global.devicePixelRatio || 1;
      this.app = null;
      this.ready = false;
      this.failed = false;
      this.active = false;
      this.width = 0;
      this.height = 0;
      this.resolution = 1;
      this.baseTextureCacheLimit = BASE_TEXTURE_CACHE_LIMIT;
      this.textures = new Map();
      this.frameTextures = new Map();
      this.compositeTextures = new Map();
      this.trimmedTextures = new Map();
      this.rigTextures = new Map();
      this.environmentTextures = new Map();
      this.mapSceneryPlacementCache = new Map();
      this.loadingTextures = new Map();
      this.runtimeTextures = new Map();
      this.idlePrewarmQueue = [];
      this.idlePrewarmKeys = new Set();
      this.idlePrewarmScheduled = false;
      this.idlePrewarmPaused = false;
      this.idlePrewarmBatchMs = 1;
      this.idlePrewarmMaxJobsPerBatch = 1;
      this.idlePrewarmMaxPending = 2;
      this.spritePools = {};
      this.textPools = {};
      this.parent = null;
      this.backgroundGraphics = null;
      this.backgroundSprites = null;
      this.worldLayer = null;
      this.mapGraphics = null;
      this.vfxGraphics = null;
      this.entityGraphics = null;
      this.uiGraphics = null;
      this.mapSprites = null;
      this.worldSprites = null;
      this.entitySprites = null;
      this.vfxSprites = null;
      this.damageSprites = null;
      this.damageTexts = null;
      this.lastVisualQuality = { level: 'normal' };
      this.frameStats = { actorFallbacks: 0, rigDraws: 0 };
    }

    getCacheValue(cache, key) {
      if (!cache || !cache.has(key)) return null;
      const value = cache.get(key);
      cache.delete(key);
      cache.set(key, value);
      return value;
    }

    setCacheValue(cache, key, value, limit) {
      if (!cache || !key || !value) return value;
      if (cache.has(key)) {
        if (cache.get(key) === value) cache.delete(key);
        else this.deleteCacheValue(cache, key);
      }
      cache.set(key, value);
      const maxSize = Math.max(1, Number(limit || 0));
      while (cache.size > maxSize) {
        const oldest = cache.keys().next().value;
        if (oldest == null) break;
        this.deleteCacheValue(cache, oldest);
      }
      return value;
    }

    destroyTexture(texture, destroySource) {
      const whiteTexture = this.PIXI && this.PIXI.Texture && this.PIXI.Texture.WHITE;
      if (!texture || texture === whiteTexture || typeof texture.destroy !== 'function') return false;
      try {
        texture.destroy(!!destroySource);
        return true;
      } catch {
        return false;
      }
    }

    releaseCacheValue(cache, value) {
      if (!value) return false;
      if (cache === this.mapSceneryPlacementCache) return false;
      if (cache === this.compositeTextures || cache === this.rigTextures) {
        return this.destroyTexture(value.texture || value, true);
      }
      if (cache === this.trimmedTextures) {
        return value.canvas ? this.destroyTexture(value.texture || value, true) : false;
      }
      if (cache === this.runtimeTextures) return this.destroyTexture(value.texture || value, true);
      if (cache === this.frameTextures || cache === this.environmentTextures) {
        return this.destroyTexture(value.texture || value, false);
      }
      return false;
    }

    deleteCacheValue(cache, key) {
      if (!cache || !cache.has(key)) return false;
      const value = cache.get(key);
      cache.delete(key);
      this.releaseCacheValue(cache, value);
      return true;
    }

    clearOwnedCache(cache) {
      if (!cache || typeof cache.keys !== 'function') return;
      Array.from(cache.keys()).forEach((key) => this.deleteCacheValue(cache, key));
    }

    releaseBaseTexture(path, texture) {
      const src = String(path || '').trim();
      if (!src || !texture) return false;
      const shared = Array.from(this.textures.entries()).some(([otherPath, otherTexture]) => otherPath !== src && otherTexture === texture);
      if (shared) return false;
      const assets = this.PIXI && this.PIXI.Assets;
      if (assets && typeof assets.unload === 'function') {
        try {
          const result = assets.unload(src);
          if (result && typeof result.catch === 'function') result.catch(() => undefined);
          return true;
        } catch {
          // Fall through to direct destruction for renderer-owned fallback textures.
        }
      }
      return this.destroyTexture(texture, true);
    }

    deleteBaseTexture(path) {
      if (!this.textures.has(path)) return false;
      const texture = this.textures.get(path);
      this.textures.delete(path);
      this.releaseBaseTexture(path, texture);
      return true;
    }

    pruneTextureDerivativesForAsset(assetPath) {
      const path = String(assetPath || '').trim();
      if (!path) return 0;
      let pruned = 0;
      [this.trimmedTextures, this.compositeTextures, this.frameTextures, this.environmentTextures].forEach((cache) => {
        Array.from(cache.keys()).forEach((key) => {
          if (!String(key || '').includes(path)) return;
          if (this.deleteCacheValue(cache, key)) pruned += 1;
        });
      });
      return pruned;
    }

    setBaseTextureValue(path, texture, limit) {
      const src = String(path || '').trim();
      if (!src || !texture) return texture;
      if (this.textures.has(src)) {
        if (this.textures.get(src) === texture) this.textures.delete(src);
        else {
          this.pruneTextureDerivativesForAsset(src);
          this.deleteBaseTexture(src);
        }
      }
      this.textures.set(src, texture);
      const maxSize = Math.max(1, Number(limit || this.baseTextureCacheLimit || BASE_TEXTURE_CACHE_LIMIT));
      while (this.textures.size > maxSize) {
        const oldest = this.textures.keys().next().value;
        if (oldest == null) break;
        this.pruneTextureDerivativesForAsset(oldest);
        this.deleteBaseTexture(oldest);
      }
      return texture;
    }

    normalizeRetainedAssetPaths(assetPaths) {
      return new Set((assetPaths || [])
        .map((assetPath) => getAssetSourcePath(assetPath))
        .map((assetPath) => String(assetPath || '').trim())
        .filter(Boolean));
    }

    isMapDerivedCacheKey(key) {
      const value = String(key || '');
      return MAP_DERIVED_CACHE_PATHS.some((path) => value.includes(path));
    }

    cacheKeyHasRetainedAsset(key, retainedAssets) {
      const value = String(key || '');
      for (const assetPath of retainedAssets || []) {
        if (assetPath && value.includes(assetPath)) return true;
      }
      return false;
    }

    pruneCacheByRetainedAssets(cache, retainedAssets) {
      if (!cache || typeof cache.keys !== 'function') return 0;
      let pruned = 0;
      Array.from(cache.keys()).forEach((key) => {
        if (!this.isMapDerivedCacheKey(key)) return;
        if (this.cacheKeyHasRetainedAsset(key, retainedAssets)) return;
        this.deleteCacheValue(cache, key);
        pruned += 1;
      });
      return pruned;
    }

    pruneIdlePrewarmQueue(retainedAssets) {
      const before = this.idlePrewarmQueue.length;
      this.idlePrewarmQueue = this.idlePrewarmQueue.filter((entry) => {
        const key = entry && (entry.key || this.getPrewarmJobKey(entry.job));
        if (!this.isMapDerivedCacheKey(key)) return true;
        return this.cacheKeyHasRetainedAsset(key, retainedAssets);
      });
      this.idlePrewarmKeys.clear();
      this.idlePrewarmQueue.forEach((entry) => {
        if (entry && entry.key) this.idlePrewarmKeys.add(entry.key);
      });
      return before - this.idlePrewarmQueue.length;
    }

    pruneMapDerivedCaches(retainAssetPaths) {
      const retainedAssets = this.normalizeRetainedAssetPaths(retainAssetPaths);
      const frameTextures = this.pruneCacheByRetainedAssets(this.frameTextures, retainedAssets);
      const trimmedTextures = this.pruneCacheByRetainedAssets(this.trimmedTextures, retainedAssets);
      const compositeTextures = this.pruneCacheByRetainedAssets(this.compositeTextures, retainedAssets);
      const environmentTextures = this.pruneCacheByRetainedAssets(this.environmentTextures, retainedAssets);
      let baseTextures = 0;
      Array.from(this.textures.keys()).forEach((path) => {
        if (!this.isMapDerivedCacheKey(path) || retainedAssets.has(path)) return;
        if (this.deleteBaseTexture(path)) baseTextures += 1;
      });
      const idlePrewarmJobs = this.pruneIdlePrewarmQueue(retainedAssets);
      return {
        frameTextures,
        trimmedTextures,
        compositeTextures,
        environmentTextures,
        baseTextures,
        idlePrewarmJobs,
        retainedAssetCount: retainedAssets.size,
        totalPruned: frameTextures + trimmedTextures + compositeTextures + environmentTextures + baseTextures + idlePrewarmJobs
      };
    }

    async init() {
      if (!this.PIXI || !this.PIXI.Application || !this.anchorCanvas) {
        this.failed = true;
        return false;
      }
      const parent = this.anchorCanvas.parentElement;
      if (!parent) {
        this.failed = true;
        return false;
      }
      this.parent = parent;
      const rect = this.anchorCanvas.getBoundingClientRect ? this.anchorCanvas.getBoundingClientRect() : null;
      this.width = Math.max(1, Math.round(this.anchorCanvas.width || rect && rect.width || 1));
      this.height = Math.max(1, Math.round(this.anchorCanvas.height || rect && rect.height || 1));
      this.resolution = this.getSafeResolution();
      this.configurePixelArtRendering();
      this.app = new this.PIXI.Application();
      await this.app.init({
        width: this.width,
        height: this.height,
        resolution: this.resolution,
        autoDensity: true,
        autoStart: false,
        antialias: false,
        backgroundAlpha: 0,
        clearBeforeRender: true,
        powerPreference: 'high-performance',
        preference: 'webgl'
      });
      this.configurePixelArtRendering();
      if (this.app.ticker && typeof this.app.ticker.stop === 'function') this.app.ticker.stop();
      this.app.canvas.className = 'project-starfall-pixi-stage';
      this.app.canvas.setAttribute('aria-hidden', 'true');
      this.syncCanvasStyle();
      parent.insertBefore(this.app.canvas, this.anchorCanvas);
      this.setupScene();
      await this.primeTextures(this.assetPaths);
      this.ready = true;
      this.setActive(true);
      return true;
    }

    setupScene() {
      const { Container, Graphics } = this.PIXI;
      this.backgroundGraphics = new Graphics();
      this.backgroundSprites = new Container();
      this.worldLayer = new Container();
      this.mapGraphics = new Graphics();
      this.vfxGraphics = new Graphics();
      this.entityGraphics = new Graphics();
      this.uiGraphics = new Graphics();
      this.mapSprites = new Container();
      this.worldSprites = new Container();
      this.entitySprites = new Container();
      this.vfxSprites = new Container();
      this.damageSprites = new Container();
      this.damageTexts = new Container();
      this.worldLayer.addChild(
        this.mapSprites,
        this.mapGraphics,
        this.worldSprites,
        this.vfxGraphics,
        this.vfxSprites,
        this.entitySprites,
        this.entityGraphics,
        this.damageSprites,
        this.damageTexts,
        this.uiGraphics
      );
      this.createPool('background', this.backgroundSprites);
      this.createPool('map', this.mapSprites);
      this.createPool('world', this.worldSprites);
      this.createPool('vfx', this.vfxSprites);
      this.createPool('entities', this.entitySprites);
      this.createPool('damage', this.damageSprites);
      this.createTextPool('damageText', this.damageTexts);
      this.app.stage.addChild(this.backgroundGraphics, this.backgroundSprites, this.worldLayer);
    }

    configurePixelArtRendering() {
      const pixi = this.PIXI || {};
      const nearest = pixi.SCALE_MODES && pixi.SCALE_MODES.NEAREST != null ? pixi.SCALE_MODES.NEAREST : 'nearest';
      if (pixi.settings) {
        try {
          if ('ROUND_PIXELS' in pixi.settings) pixi.settings.ROUND_PIXELS = true;
          if ('SCALE_MODE' in pixi.settings) pixi.settings.SCALE_MODE = nearest;
        } catch {
          // Some Pixi builds expose read-only settings.
        }
      }
      const renderer = this.app && this.app.renderer;
      if (renderer) {
        try {
          renderer.roundPixels = true;
        } catch {
          // Optional renderer setting; safe to ignore on unsupported builds.
        }
      }
    }

    applyPixelArtTextureSettings(texture) {
      if (!texture) return texture;
      const pixi = this.PIXI || {};
      const nearest = pixi.SCALE_MODES && pixi.SCALE_MODES.NEAREST != null ? pixi.SCALE_MODES.NEAREST : 'nearest';
      [texture, texture.baseTexture, texture.source, texture.source && texture.source.style].forEach((target) => {
        if (!target) return;
        try {
          if ('scaleMode' in target) target.scaleMode = nearest;
        } catch {
          // Texture sampling controls differ by Pixi major version.
        }
      });
      return texture;
    }

    createPool(name, container) {
      const pool = { container, items: [], active: 0, previousActive: 0 };
      this.spritePools[name] = pool;
      return pool;
    }

    createTextPool(name, container) {
      const pool = { container, items: [], active: 0, previousActive: 0 };
      this.textPools[name] = pool;
      return pool;
    }

    beginPools() {
      Object.values(this.spritePools).forEach((pool) => {
        pool.previousActive = Math.max(0, Math.min(pool.items.length, Math.floor(Number(pool.active || 0) || 0)));
        pool.active = 0;
      });
      Object.values(this.textPools).forEach((pool) => {
        pool.previousActive = Math.max(0, Math.min(pool.items.length, Math.floor(Number(pool.active || 0) || 0)));
        pool.active = 0;
      });
    }

    hideUnusedSprites() {
      Object.values(this.spritePools).forEach((pool) => {
        const active = Math.max(0, Math.min(pool.items.length, Math.floor(Number(pool.active || 0) || 0)));
        const previousActive = Math.max(active, Math.min(pool.items.length, Math.floor(Number(pool.previousActive || 0) || 0)));
        for (let i = active; i < previousActive; i += 1) {
          pool.items[i].visible = false;
        }
        pool.previousActive = active;
      });
      Object.values(this.textPools).forEach((pool) => {
        const active = Math.max(0, Math.min(pool.items.length, Math.floor(Number(pool.active || 0) || 0)));
        const previousActive = Math.max(active, Math.min(pool.items.length, Math.floor(Number(pool.previousActive || 0) || 0)));
        for (let i = active; i < previousActive; i += 1) {
          pool.items[i].visible = false;
        }
        pool.previousActive = active;
      });
    }

    acquireSprite(poolName) {
      const pool = this.spritePools[poolName];
      if (!pool) return null;
      let sprite = pool.items[pool.active];
      if (!sprite) {
        sprite = new this.PIXI.Sprite(this.PIXI.Texture.WHITE);
        sprite.anchor.set(0.5);
        pool.items.push(sprite);
        pool.container.addChild(sprite);
      }
      pool.active += 1;
      sprite.visible = true;
      sprite.texture = this.PIXI.Texture.WHITE;
      sprite.position.set(0, 0);
      sprite.rotation = 0;
      sprite.scale.set(1, 1);
      sprite.anchor.set(0.5);
      sprite.alpha = 1;
      sprite.tint = FALLBACK_COLOR;
      sprite.blendMode = 'normal';
      return sprite;
    }

    createTextNode() {
      if (!this.PIXI || !this.PIXI.Text) return null;
      try {
        const text = new this.PIXI.Text({ text: '', style: Object.assign({}, DAMAGE_TEXT_POOL_STYLE) });
        text._starfallStyleKey = '';
        return text;
      } catch {
        try {
          const text = new this.PIXI.Text('', Object.assign({}, DAMAGE_TEXT_POOL_STYLE));
          text._starfallStyleKey = '';
          return text;
        } catch {
          return null;
        }
      }
    }

    acquireText(poolName) {
      const pool = this.textPools[poolName];
      if (!pool) return null;
      let text = pool.items[pool.active];
      if (!text) {
        text = this.createTextNode();
        if (!text) return null;
        if (text.anchor && typeof text.anchor.set === 'function') text.anchor.set(0.5);
        pool.items.push(text);
        pool.container.addChild(text);
      }
      pool.active += 1;
      text.visible = true;
      text.position.set(0, 0);
      text.rotation = 0;
      text.scale.set(1, 1);
      text.alpha = 1;
      if (text.anchor && typeof text.anchor.set === 'function') text.anchor.set(0.5);
      return text;
    }

    drawText(poolName, value, x, y, options) {
      const text = this.acquireText(poolName);
      if (!text) return false;
      const settings = options || {};
      const fontSize = Math.max(8, Number(settings.fontSize || DAMAGE_TEXT_POOL_STYLE.fontSize));
      const fontWeight = String(settings.fontWeight || DAMAGE_TEXT_POOL_STYLE.fontWeight);
      const fill = settings.fill || DAMAGE_TEXT_POOL_STYLE.fill;
      const strokeColor = settings.stroke || 'rgba(9,31,59,0.92)';
      const strokeWidth = Math.max(0, Number(settings.strokeWidth == null ? 4 : settings.strokeWidth));
      const styleKey = [fontSize, fontWeight, fill, strokeColor, strokeWidth].join('|');
      const nextText = String(value || '');
      if (text.text !== nextText) text.text = nextText;
      if (text._starfallStyleKey !== styleKey) {
        text.style = Object.assign({}, DAMAGE_TEXT_POOL_STYLE, {
          fontSize,
          fontWeight,
          fill,
          stroke: { color: strokeColor, width: strokeWidth },
          strokeThickness: strokeWidth
        });
        text._starfallStyleKey = styleKey;
      }
      text.position.set(Number(x || 0), Number(y || 0));
      text.alpha = clamp(settings.alpha == null ? 1 : Number(settings.alpha), 0, 1);
      text.rotation = Number(settings.rotation || 0);
      text.scale.set(Number(settings.scaleX || settings.scale || 1), Number(settings.scaleY || settings.scale || 1));
      return true;
    }

    primeTextures(paths) {
      return this.prewarmTextures(paths);
    }

    prewarmTextures(paths) {
      const loading = [];
      const seen = new Set();
      (paths || []).forEach((path) => {
        const src = String(path || '').trim();
        if (!src || seen.has(src)) return;
        seen.add(src);
        loading.push(this.loadTexture(src));
      });
      return Promise.all(loading).catch(() => []);
    }

    scheduleIdlePrewarm(jobs) {
      const queue = Array.isArray(jobs) ? jobs : [jobs];
      queue.forEach((job) => {
        if (!job) return;
        const key = this.getPrewarmJobKey(job);
        if (key && this.idlePrewarmKeys.has(key)) return;
        if (key) this.idlePrewarmKeys.add(key);
        this.idlePrewarmQueue.push({ job, key });
      });
      if (this.idlePrewarmQueue.length && !this.idlePrewarmPaused) this.scheduleIdlePrewarmPump();
    }

    setIdlePrewarmPaused(paused) {
      const next = !!paused;
      if (this.idlePrewarmPaused === next) return;
      this.idlePrewarmPaused = next;
      if (!next && this.idlePrewarmQueue.length) this.scheduleIdlePrewarmPump();
    }

    scheduleIdlePrewarmPump(delayMs) {
      if (this.idlePrewarmScheduled) return;
      this.idlePrewarmScheduled = true;
      const run = (deadline) => this.runIdlePrewarm(deadline);
      if (Number(delayMs) > 0) {
        global.setTimeout(() => run(null), Number(delayMs) || 0);
      } else if (global.requestIdleCallback) {
        global.requestIdleCallback(run);
      } else {
        global.setTimeout(() => run(null), 16);
      }
    }

    runIdlePrewarm(deadline) {
      this.idlePrewarmScheduled = false;
      if (this.idlePrewarmPaused) {
        return;
      }
      const startedAt = nowMs();
      let processed = 0;
      while (this.idlePrewarmQueue.length && processed < this.idlePrewarmMaxJobsPerBatch) {
        if (this.loadingTextures.size >= this.idlePrewarmMaxPending) break;
        const remaining = deadline && typeof deadline.timeRemaining === 'function'
          ? deadline.timeRemaining()
          : this.idlePrewarmBatchMs - (nowMs() - startedAt);
        if (remaining <= 1) break;
        const entry = this.idlePrewarmQueue.shift();
        if (!entry) break;
        this.processPrewarmJob(entry.job);
        processed += 1;
        if (entry.key) this.idlePrewarmKeys.delete(entry.key);
      }
      if (this.idlePrewarmQueue.length) {
        this.scheduleIdlePrewarmPump(this.loadingTextures.size >= this.idlePrewarmMaxPending ? 160 : 0);
      }
    }

    getPrewarmJobKey(job) {
      if (typeof job === 'string') return `texture:${job}`;
      if (!job || typeof job !== 'object') return '';
      if (job.type === 'texture') return `texture:${job.path || ''}`;
      if (job.type === 'actorFrame' || job.type === 'frame') return `frame:${this.getFrameTextureKey(job.frame)}`;
      if (job.type === 'actorComposite' || job.type === 'composite') return `composite:${job.key || ''}`;
      if (job.type === 'actorRig' || job.type === 'rig') return `rig:${job.key || job.rig && job.rig.cacheKey || ''}`;
      return `${job.type || 'job'}:${job.path || job.key || ''}`;
    }

    prewarmActorFrames(jobs) {
      (jobs || []).forEach((job) => {
        this.processPrewarmJob(job);
      });
    }

    processPrewarmJob(job) {
      if (typeof job === 'string') {
        this.loadTexture(job);
        return;
      }
      if (!job || typeof job !== 'object') return;
      if (job.type === 'texture') {
        this.loadTexture(job.path);
        return;
      }
      if (job.type === 'actorFrame' || job.type === 'frame') {
        this.prewarmFrameTexture(job.frame, job.key);
        return;
      }
      if (job.type === 'actorComposite' || job.type === 'composite') {
        this.prewarmCompositeTexture(job.frames, job.key);
        return;
      }
      if (job.type === 'actorRig' || job.type === 'rig') {
        this.prewarmRigTexture(job.rig, job.box);
      }
    }

    prewarmFrameTexture(frame, trimKey) {
      if (!frame || !frame.sheet) return;
      const cacheKey = this.getFrameTextureKey(frame);
      const existing = this.frameTextures.get(cacheKey) || this.getFrameTexture(frame);
      if (existing) {
        this.getTrimmedTexture(existing, trimKey || `frame:${cacheKey}`);
        return;
      }
      this.loadTexture(frame.sheet).then(() => {
        const texture = this.getFrameTexture(frame);
        if (texture) this.getTrimmedTexture(texture, trimKey || `frame:${cacheKey}`);
      });
    }

    prewarmCompositeTexture(frames, key) {
      const layers = (frames || []).filter((frame) => frame && frame.sheet);
      if (!layers.length) return;
      const cacheKey = key || layers.map((frame) => this.getFrameTextureKey(frame)).join('|');
      const existing = this.getCacheValue(this.compositeTextures, cacheKey);
      if (existing) {
        this.getTrimmedTexture(existing.texture || existing, `composite:${cacheKey}`);
        return;
      }
      const sheets = Array.from(new Set(layers.map((frame) => frame.sheet)));
      Promise.all(sheets.map((sheet) => this.loadTexture(sheet))).then(() => {
        const texture = this.getCompositeFrameTexture(layers, cacheKey);
        if (texture) this.getTrimmedTexture(texture, `composite:${cacheKey}`);
      });
    }

    prewarmRigTexture(rigRender, box) {
      if (!rigRender) return;
      const safeBox = Object.assign({ w: 40, h: 74 }, box || {});
      this.getRigFrameTexture(rigRender, safeBox);
    }

    loadTexture(path) {
      const src = String(path || '').trim();
      if (!src) return Promise.resolve(null);
      if (this.textures.has(src)) return Promise.resolve(this.textures.get(src));
      if (this.loadingTextures.has(src)) return this.loadingTextures.get(src);
      const promise = Promise.resolve()
        .then(() => {
          if (this.PIXI.Assets && typeof this.PIXI.Assets.load === 'function') return this.PIXI.Assets.load(src);
          return this.PIXI.Texture.from(src);
        })
        .then((texture) => {
          if (texture) this.setBaseTextureValue(src, this.applyPixelArtTextureSettings(texture));
          this.loadingTextures.delete(src);
          return texture || null;
        })
        .catch(() => {
          this.loadingTextures.delete(src);
          const backupPath = this.getAssetBackupPath(src);
          if (!backupPath || backupPath === src) return null;
          return this.loadTexture(backupPath).then((texture) => {
            if (texture) this.setBaseTextureValue(src, texture);
            return texture || null;
          });
        });
      this.loadingTextures.set(src, promise);
      return promise;
    }

    getAssetBackupPath(path) {
      const src = getAssetSourcePath(path);
      return src && this.assetBackupPaths && this.assetBackupPaths[src] || '';
    }

    getTexture(path) {
      const src = getAssetSourcePath(path);
      if (!src) return null;
      const texture = this.textures.get(src);
      if (texture) {
        this.textures.delete(src);
        this.textures.set(src, texture);
        return texture;
      }
      this.loadTexture(src);
      return null;
    }

    getAssetFrameTexture(assetPath) {
      const frame = parseAssetFrame(assetPath);
      if (!frame || !frame.sw || !frame.sh) return this.getTexture(assetPath);
      const key = `asset:${assetPath}`;
      const cached = this.getCacheValue(this.frameTextures, key);
      if (cached) return cached;
      const base = this.getTexture(frame.path);
      if (!base) return null;
      const { Rectangle, Texture } = this.PIXI;
      const source = base.source || base.baseTexture || base;
      const rectangle = new Rectangle(frame.sx, frame.sy, frame.sw, frame.sh);
      let texture = null;
      try {
        texture = new Texture({ source, frame: rectangle });
      } catch {
        try {
          texture = new Texture(source, rectangle);
        } catch {
          texture = null;
        }
      }
      if (texture) this.setCacheValue(this.frameTextures, key, texture, FRAME_TEXTURE_CACHE_LIMIT);
      return texture;
    }

    getRuntimeTexture(kind) {
      const key = String(kind || 'circle');
      if (this.runtimeTextures.has(key)) return this.runtimeTextures.get(key);
      const texture = this.createRuntimeTexture(key);
      if (texture) this.runtimeTextures.set(key, texture);
      return texture || this.PIXI.Texture.WHITE;
    }

    createRuntimeTexture(kind) {
      if (!global.document || typeof global.document.createElement !== 'function' || !this.PIXI || !this.PIXI.Texture) {
        return this.PIXI && this.PIXI.Texture ? this.PIXI.Texture.WHITE : null;
      }
      const size = kind === 'slash' ? { w: 128, h: 64 } : kind === 'diamond' ? { w: 48, h: 48 } : { w: 96, h: 96 };
      const canvas = global.document.createElement('canvas');
      canvas.width = size.w;
      canvas.height = size.h;
      const ctx = canvas.getContext('2d');
      if (!ctx) return this.PIXI.Texture.WHITE;
      ctx.clearRect(0, 0, size.w, size.h);
      if (kind === 'glow') {
        const gradient = ctx.createRadialGradient(size.w / 2, size.h / 2, 2, size.w / 2, size.h / 2, size.w / 2);
        gradient.addColorStop(0, 'rgba(255,255,255,0.92)');
        gradient.addColorStop(0.42, 'rgba(255,255,255,0.34)');
        gradient.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(size.w / 2, size.h / 2, size.w / 2, 0, Math.PI * 2);
        ctx.fill();
      } else if (kind === 'ring') {
        ctx.strokeStyle = 'rgba(255,255,255,0.94)';
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.arc(size.w / 2, size.h / 2, size.w / 2 - 7, 0, Math.PI * 2);
        ctx.stroke();
      } else if (kind === 'diamond') {
        ctx.fillStyle = 'rgba(255,255,255,0.96)';
        ctx.strokeStyle = 'rgba(255,255,255,0.96)';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(size.w / 2, 4);
        ctx.lineTo(size.w - 5, size.h / 2);
        ctx.lineTo(size.w / 2, size.h - 4);
        ctx.lineTo(5, size.h / 2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      } else if (kind === 'slash') {
        ctx.strokeStyle = 'rgba(255,255,255,0.92)';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(12, 45);
        ctx.quadraticCurveTo(58, 6, 118, 28);
        ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,0.32)';
        ctx.lineWidth = 18;
        ctx.beginPath();
        ctx.moveTo(16, 48);
        ctx.quadraticCurveTo(62, 4, 122, 30);
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.95)';
        ctx.beginPath();
        ctx.arc(size.w / 2, size.h / 2, size.w / 2 - 2, 0, Math.PI * 2);
        ctx.fill();
      }
      return this.PIXI.Texture.from(canvas);
    }

    drawSolidRect(poolName, x, y, width, height, options) {
      let drawX = Number(x || 0);
      let drawY = Number(y || 0);
      let drawW = Number(width || 0);
      let drawH = Number(height || 0);
      if (drawW < 0) {
        drawX += drawW;
        drawW = Math.abs(drawW);
      }
      if (drawH < 0) {
        drawY += drawH;
        drawH = Math.abs(drawH);
      }
      return this.drawTexture(poolName, this.PIXI.Texture.WHITE, drawX, drawY, drawW, drawH, Object.assign({
        anchorX: 0,
        anchorY: 0
      }, options || {}));
    }

    drawShape(poolName, kind, x, y, width, height, options) {
      return this.drawTexture(poolName, this.getRuntimeTexture(kind), x, y, width, height, options || {});
    }

    drawLine(poolName, x1, y1, x2, y2, width, options) {
      const dx = Number(x2 || 0) - Number(x1 || 0);
      const dy = Number(y2 || 0) - Number(y1 || 0);
      const length = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      return this.drawTexture(poolName, this.PIXI.Texture.WHITE, x1, y1, length, Math.max(1, Number(width || 1)), Object.assign({
        anchorX: 0,
        anchorY: 0.5,
        rotation: Math.atan2(dy, dx)
      }, options || {}));
    }

    drawRectOutline(poolName, x, y, width, height, strokeWidth, options) {
      const line = Math.max(1, Number(strokeWidth || 1));
      const settings = options || {};
      this.drawSolidRect(poolName, x, y, width, line, settings);
      this.drawSolidRect(poolName, x, y + height - line, width, line, settings);
      this.drawSolidRect(poolName, x, y, line, height, settings);
      this.drawSolidRect(poolName, x + width - line, y, line, height, settings);
    }

    getFrameTextureKey(frame) {
      if (!frame || !frame.sheet) return '';
      const frameWidth = Math.max(1, Number(frame.frameWidth || frame.w || 0) || 160);
      const frameHeight = Math.max(1, Number(frame.frameHeight || frame.h || 0) || 160);
      const frameIndex = Math.max(0, Number(frame.frameIndex || 0) || 0);
      const row = Math.max(0, Number(frame.row || 0) || 0);
      return `${frame.sheet}:${row}:${frameIndex}:${frameWidth}:${frameHeight}`;
    }

    getFrameTexture(frame) {
      if (!frame || !frame.sheet) return null;
      const key = this.getFrameTextureKey(frame);
      const cached = this.getCacheValue(this.frameTextures, key);
      if (cached) return cached;
      const frameWidth = Math.max(1, Number(frame.frameWidth || frame.w || 0) || 160);
      const frameHeight = Math.max(1, Number(frame.frameHeight || frame.h || 0) || 160);
      const frameIndex = Math.max(0, Number(frame.frameIndex || 0) || 0);
      const row = Math.max(0, Number(frame.row || 0) || 0);
      const base = this.getTexture(frame.sheet);
      if (!base) return null;
      const { Rectangle, Texture } = this.PIXI;
      const source = base.source || base.baseTexture || base;
      const rectangle = new Rectangle(frameIndex * frameWidth, row * frameHeight, frameWidth, frameHeight);
      let texture = null;
      try {
        texture = new Texture({ source, frame: rectangle });
      } catch {
        try {
          texture = new Texture(source, rectangle);
        } catch {
          texture = null;
        }
      }
      if (texture) this.setCacheValue(this.frameTextures, key, texture, FRAME_TEXTURE_CACHE_LIMIT);
      return texture;
    }

    getTextureSourceResource(texture) {
      if (!texture) return null;
      const source = texture.source || texture.baseTexture || texture;
      return source && (source.resource || source.image || source.canvas || source.source || null);
    }

    getTrimmedTexture(texture, key) {
      if (!texture) return null;
      const cacheKey = String(key || texture.uid || texture.label || texture.textureCacheIds && texture.textureCacheIds[0] || '');
      if (!cacheKey) return { texture, width: texture.width || 1, height: texture.height || 1 };
      const cached = this.getCacheValue(this.trimmedTextures, cacheKey);
      if (cached) return cached;
      const trimmed = this.createTrimmedTexture(texture) || {
        texture,
        width: Math.max(1, Number(texture.width || 1)),
        height: Math.max(1, Number(texture.height || 1))
      };
      return this.setCacheValue(this.trimmedTextures, cacheKey, trimmed, TRIMMED_TEXTURE_CACHE_LIMIT);
    }

    createTrimmedTexture(texture) {
      if (!texture || !global.document || typeof global.document.createElement !== 'function') return null;
      const resource = this.getTextureSourceResource(texture);
      if (!resource) return null;
      const frame = texture.frame || texture.orig || {};
      const sourceX = Math.max(0, Math.round(Number(frame.x || 0)));
      const sourceY = Math.max(0, Math.round(Number(frame.y || 0)));
      const sourceW = Math.max(1, Math.round(Number(frame.width || texture.width || 1)));
      const sourceH = Math.max(1, Math.round(Number(frame.height || texture.height || 1)));
      const scanCanvas = global.document.createElement('canvas');
      scanCanvas.width = sourceW;
      scanCanvas.height = sourceH;
      const scanCtx = scanCanvas.getContext('2d', { willReadFrequently: true });
      if (!scanCtx) return null;
      try {
        scanCtx.clearRect(0, 0, sourceW, sourceH);
        scanCtx.drawImage(resource, sourceX, sourceY, sourceW, sourceH, 0, 0, sourceW, sourceH);
        const pixels = scanCtx.getImageData(0, 0, sourceW, sourceH).data;
        let minX = sourceW;
        let minY = sourceH;
        let maxX = -1;
        let maxY = -1;
        for (let y = 0; y < sourceH; y += 1) {
          for (let x = 0; x < sourceW; x += 1) {
            const alpha = pixels[(y * sourceW + x) * 4 + 3];
            if (alpha <= 8) continue;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }
        }
        if (maxX < minX || maxY < minY) return null;
        const pad = 1;
        minX = Math.max(0, minX - pad);
        minY = Math.max(0, minY - pad);
        maxX = Math.min(sourceW - 1, maxX + pad);
        maxY = Math.min(sourceH - 1, maxY + pad);
        const trimW = Math.max(1, maxX - minX + 1);
        const trimH = Math.max(1, maxY - minY + 1);
        const canvas = global.document.createElement('canvas');
        canvas.width = trimW;
        canvas.height = trimH;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;
        ctx.drawImage(scanCanvas, minX, minY, trimW, trimH, 0, 0, trimW, trimH);
        const trimmedTexture = this.PIXI.Texture.from(canvas);
        return {
          texture: trimmedTexture,
          width: trimW,
          height: trimH,
          canvas
        };
      } catch {
        return null;
      }
    }

    getCompositeFrameTexture(frames, key) {
      const parts = (frames || []).filter((frame) => frame && frame.sheet);
      if (!parts.length || !global.document || typeof global.document.createElement !== 'function') return null;
      const cacheKey = String(key || parts.map((frame) => this.getFrameTextureKey(frame)).join('|'));
      if (!cacheKey) return null;
      const cached = this.getCacheValue(this.compositeTextures, cacheKey);
      if (cached) return cached.texture;
      const first = parts[0];
      const width = Math.max(1, Number(first.frameWidth || first.w || 0) || 160);
      const height = Math.max(1, Number(first.frameHeight || first.h || 0) || 160);
      const canvas = global.document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      try {
        parts.forEach((frameDef) => {
          const texture = this.getFrameTexture(frameDef);
          const resource = this.getTextureSourceResource(texture);
          if (!texture || !resource) throw new Error('Missing frame texture');
          const frame = texture.frame || texture.orig || {};
          const sourceX = Math.max(0, Math.round(Number(frame.x || 0)));
          const sourceY = Math.max(0, Math.round(Number(frame.y || 0)));
          const sourceW = Math.max(1, Math.round(Number(frame.width || texture.width || width)));
          const sourceH = Math.max(1, Math.round(Number(frame.height || texture.height || height)));
          ctx.drawImage(resource, sourceX, sourceY, sourceW, sourceH, 0, 0, width, height);
        });
      } catch {
        return null;
      }
      const texture = this.PIXI.Texture.from(canvas);
      this.setCacheValue(this.compositeTextures, cacheKey, { texture, canvas }, COMPOSITE_TEXTURE_CACHE_LIMIT);
      return texture;
    }

    getRigFrameTexture(rigRender, box) {
      if (!rigRender || !rigRender.rig || !this.rigRenderer || typeof this.rigRenderer.drawCharacter !== 'function') return null;
      const safeBox = box || {};
      const boxWidth = Math.max(1, Math.round(Number(safeBox.w || 40)));
      const boxHeight = Math.max(1, Math.round(Number(safeBox.h || 74)));
      const scale = Math.max(1, Math.round(Number(rigRender.textureScale || RIG_TEXTURE_SCALE) || RIG_TEXTURE_SCALE));
      const cacheKey = [
        rigRender.cacheKey || `${rigRender.state || 'idle'}:${rigRender.frameIndex || 0}`,
        `${boxWidth}x${boxHeight}`,
        `s${scale}`
      ].join('|');
      const cached = this.getCacheValue(this.rigTextures, cacheKey);
      if (cached) return cached;
      const rendered = this.createRigFrameTexture(rigRender, { w: boxWidth, h: boxHeight }, scale);
      if (!rendered) return null;
      return this.setCacheValue(this.rigTextures, cacheKey, rendered, RIG_TEXTURE_CACHE_LIMIT);
    }

    createRigFrameTexture(rigRender, box, textureScale) {
      if (!global.document || typeof global.document.createElement !== 'function' || !this.PIXI || !this.PIXI.Texture) return null;
      const rig = rigRender.rig || {};
      const boxWidth = Math.max(1, Math.round(Number(box && box.w || 40)));
      const boxHeight = Math.max(1, Math.round(Number(box && box.h || 74)));
      const bottomPad = Math.max(18, Math.round(boxHeight * 0.22));
      const logicalWidth = Math.ceil(Math.max(132, boxWidth + 118));
      const logicalHeight = Math.ceil(Math.max(156, boxHeight + 92));
      const actorX = Math.round((logicalWidth - boxWidth) / 2);
      const actorY = Math.round(logicalHeight - boxHeight - bottomPad);
      const scale = Math.max(1, Math.round(Number(textureScale || RIG_TEXTURE_SCALE) || RIG_TEXTURE_SCALE));
      const canvas = global.document.createElement('canvas');
      canvas.width = Math.ceil(logicalWidth * scale);
      canvas.height = Math.ceil(logicalHeight * scale);
      const ctx = canvas.getContext('2d');
      if (!ctx) return null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      let drawn = false;
      try {
        ctx.save();
        ctx.scale(scale, scale);
        const actor = {
          x: actorX,
          y: actorY,
          w: boxWidth,
          h: boxHeight,
          facing: 1,
          animationState: rigRender.state || 'idle'
        };
        drawn = this.rigRenderer.drawCharacter(ctx, actor, rig, {
          state: rigRender.state || 'idle',
          elapsed: Math.max(0, Number(rigRender.elapsed || 0)),
          equipment: rigRender.equipment || {},
          palette: rigRender.palette || {},
          scale: boxHeight / Math.max(1, Number(rig.height || boxHeight))
        });
      } catch {
        drawn = false;
      } finally {
        ctx.restore();
      }
      if (!drawn) return null;
      const texture = this.PIXI.Texture.from(canvas);
      return {
        texture,
        canvas,
        width: logicalWidth,
        height: logicalHeight,
        anchorX: (actorX + boxWidth / 2) / logicalWidth,
        anchorY: (actorY + boxHeight) / logicalHeight
      };
    }

    getSafeResolution() {
      return clamp(this.getRenderScale() || global.devicePixelRatio || 1, 0.35, 2);
    }

    syncCanvasStyle() {
      if (!this.app || !this.app.canvas || !this.app.canvas.style) return;
      this.app.canvas.style.width = '100%';
      this.app.canvas.style.height = '100%';
    }

    resize(width, height) {
      if (!this.app || !this.app.renderer) return;
      const safeWidth = Math.max(1, Math.round(width || this.width || 1));
      const safeHeight = Math.max(1, Math.round(height || this.height || 1));
      const resolution = this.getSafeResolution();
      if (safeWidth === this.width && safeHeight === this.height && resolution === this.resolution) {
        this.syncCanvasStyle();
        return;
      }
      this.width = safeWidth;
      this.height = safeHeight;
      this.resolution = resolution;
      this.app.renderer.resize(safeWidth, safeHeight, resolution);
      this.syncCanvasStyle();
    }

    setActive(active) {
      this.active = !!active;
      if (this.app && this.app.canvas) this.app.canvas.hidden = !this.active;
      if (this.parent && this.parent.classList) this.parent.classList.toggle('is-pixi-renderer', this.active);
    }

    renderFrame(snapshot) {
      if (!this.ready || !this.app || !this.active) return null;
      const timings = {};
      let marker = nowMs();
      this.lastVisualQuality = snapshot.visualQuality || { level: 'normal' };
      this.frameStats = { actorFallbacks: 0, rigDraws: 0 };
      this.resize(snapshot.width, snapshot.height);
      this.beginPools();
      this.clearGraphics();
      this.renderBackground(snapshot);
      marker = markTiming(timings, 'background', marker);
      this.updateWorldTransform(snapshot);
      this.renderMap(snapshot);
      marker = markTiming(timings, 'map', marker);
      this.renderWorldEffects(snapshot);
      marker = markTiming(timings, 'effects', marker);
      this.renderProjectiles(snapshot);
      marker = markTiming(timings, 'projectiles', marker);
      this.renderLoot(snapshot);
      marker = markTiming(timings, 'loot', marker);
      this.renderEnemies(snapshot);
      marker = markTiming(timings, 'enemies', marker);
      this.renderParty(snapshot);
      marker = markTiming(timings, 'party', marker);
      this.renderPet(snapshot);
      marker = markTiming(timings, 'pet', marker);
      this.renderPlayer(snapshot);
      marker = markTiming(timings, 'player', marker);
      this.renderDamageSplats(snapshot);
      marker = markTiming(timings, 'damageSplats', marker);
      this.hideUnusedSprites();
      marker = markTiming(timings, 'poolCleanup', marker);
      this.app.render();
      markTiming(timings, 'present', marker);
      return timings;
    }

    clearGraphics() {
      this.backgroundGraphics.clear();
      this.mapGraphics.clear();
      this.vfxGraphics.clear();
      this.entityGraphics.clear();
      this.uiGraphics.clear();
    }

    updateWorldTransform(snapshot) {
      const camera = snapshot.camera || {};
      const zoom = Math.max(1, Number(camera.zoom || 1));
      this.worldLayer.scale.set(zoom, zoom);
      this.worldLayer.position.set(
        -Math.round(Number(camera.x || 0) * zoom),
        -Math.round(Number(camera.y || 0) * zoom)
      );
    }

    renderBackground(snapshot) {
      const width = Math.max(1, Number(snapshot.width || this.width || 1));
      const height = Math.max(1, Number(snapshot.playfieldHeight || snapshot.height || this.height || 1));
      const map = snapshot.map || {};
      const palette = Array.isArray(map.palette) ? map.palette : [];
      const skyTop = map.id === 'cinderHollow' ? 0x2c2632 : 0xdff7ff;
      const skyMid = colorToNumber(palette[1], 0x91dbe8);
      const ground = colorToNumber(palette[0], 0x77bf65);
      this.backgroundGraphics
        .rect(0, 0, width, height * 0.58)
        .fill({ color: skyTop, alpha: 1 })
        .rect(0, height * 0.42, width, height * 0.58)
        .fill({ color: skyMid, alpha: 0.9 })
        .rect(0, height * 0.82, width, height * 0.18)
        .fill({ color: ground, alpha: 0.24 });
      const texture = this.getTexture(map.asset);
      if (!texture) {
        this.renderProceduralBackground(snapshot, width, height);
        return;
      }
      const imageWidth = Math.max(1, Number(texture.width || 1));
      const imageHeight = Math.max(1, Number(texture.height || 1));
      const drawWidth = Math.max(1, Math.round(imageWidth * (height / imageHeight)));
      const parallaxX = Number(snapshot.camera && snapshot.camera.x || 0) * MAP_BACKGROUND_PARALLAX;
      const parallaxY = Number(snapshot.camera && snapshot.camera.y || 0) * MAP_BACKGROUND_PARALLAX;
      const tileOffset = -(((parallaxX % drawWidth) + drawWidth) % drawWidth);
      const startX = Math.floor(tileOffset) - MAP_BACKGROUND_TILE_OVERLAP_PX;
      const tileWidth = drawWidth + MAP_BACKGROUND_TILE_OVERLAP_PX * 2;
      const drawY = -Math.round(Math.max(0, parallaxY * 0.16));
      for (let x = startX; x < width + MAP_BACKGROUND_TILE_OVERLAP_PX; x += drawWidth) {
        this.drawTexture('background', texture, x, drawY, tileWidth, height + Math.abs(drawY), {
          anchorX: 0,
          anchorY: 0
        });
      }
      this.backgroundGraphics
        .rect(0, 0, width, height)
        .fill({ color: map.id === 'cinderHollow' ? 0x140a18 : 0xffffff, alpha: map.id === 'cinderHollow' ? 0.22 : 0.08 });
      this.renderWorldBaseBand(snapshot, width, height, map);
    }

    renderProceduralBackground(snapshot, width, height) {
      const cameraX = Number(snapshot.camera && snapshot.camera.x || 0);
      const cameraY = Number(snapshot.camera && snapshot.camera.y || 0);
      const graphics = this.backgroundGraphics;
      graphics.fill({ color: 0xffffff, alpha: 0 });
      for (let i = 0; i < 7; i += 1) {
        const x = positiveModulo(i * 190 - cameraX * 0.18, width + 220);
        graphics.ellipse(x, 86 + (i % 3) * 40 - cameraY * 0.12, 80, 22).fill({ color: 0xfff2b4, alpha: 0.38 });
      }
      for (let i = 0; i < 6; i += 1) {
        const x = positiveModulo(i * 260 - cameraX * 0.34, width + 260);
        graphics
          .moveTo(x, height)
          .lineTo(x + 150, 230 + (i % 2) * 40 - cameraY * 0.18)
          .lineTo(x + 320, height)
          .closePath()
          .fill({ color: 0x091f3b, alpha: 0.18 });
      }
      this.renderWorldBaseBand(snapshot, width, height, snapshot.map || {});
    }

    getWorldBaseBandStyle(map) {
      const theme = this.getMapThemeId(map);
      const palette = map && map.palette || [];
      if (theme.includes('cinder') || theme.includes('ember') || theme.includes('fire')) return { color: 0x140a18, alpha: 0.88 };
      if (theme.includes('frost') || theme.includes('rime') || theme.includes('glacier')) return { color: 0xa3d9f2, alpha: 0.78 };
      if (theme.includes('astral') || theme.includes('eclipse') || theme.includes('rift') || theme.includes('rune')) return { color: 0x1d1d40, alpha: 0.86 };
      if (theme.includes('storm')) return { color: 0x2f445c, alpha: 0.82 };
      if (theme.includes('ruins') || theme.includes('gearworks') || theme.includes('quarry') || theme.includes('rust') || theme.includes('titan') || theme.includes('deepcore')) return { color: 0x4e504e, alpha: 0.84 };
      if (theme.includes('bandit') || theme.includes('ridge') || theme.includes('duelist') || theme.includes('sniper')) return { color: 0x654a30, alpha: 0.82 };
      return { color: colorToNumber(palette[0], 0x2f6848), alpha: 0.8 };
    }

    renderWorldBaseBand(snapshot, width, playfieldHeight, map) {
      const top = Math.max(0, Math.round(Number(playfieldHeight || 0)) - 2);
      const bottom = Math.max(top, Math.round(Math.min(
        Number(snapshot && snapshot.height || this.height || top),
        Number(playfieldHeight || 0) + Number(snapshot && snapshot.solidPlatformHeight || 0)
      )));
      if (bottom <= top) return;
      const style = this.getWorldBaseBandStyle(map);
      this.backgroundGraphics
        .rect(0, top, width, bottom - top)
        .fill({ color: style.color, alpha: style.alpha })
        .rect(0, top, width, 2)
        .fill({ color: 0x091f3b, alpha: 0.24 });
    }

    getEnvironmentProfile(map) {
      const profiles = this.data && this.data.MAP_ENVIRONMENT_PROFILES || {};
      return map && map.environment || profiles[map && map.id] || profiles.greenrootMeadow || null;
    }

    getEnvironmentAsset(kind, profile) {
      const group = this.data && this.data.ENVIRONMENT_ASSETS && this.data.ENVIRONMENT_ASSETS[kind];
      const id = profile && (profile[kind] || (kind === 'ramps' ? profile.terrain : ''));
      return group && id ? group[id] : null;
    }

    getEnvironmentCells(kind) {
      if (kind === 'props') return this.data && this.data.ENVIRONMENT_PROP_CELLS || DEFAULT_ENVIRONMENT_PROP_CELLS;
      return this.data && this.data.ENVIRONMENT_TERRAIN_CELLS || DEFAULT_ENVIRONMENT_TERRAIN_CELLS;
    }

    getEnvironmentTerrainStyle(profile) {
      return Object.assign(
        {},
        this.data && this.data.ENVIRONMENT_TERRAIN_STYLE_DEFAULTS || DEFAULT_ENVIRONMENT_TERRAIN_STYLE,
        profile && profile.terrainStyle || {}
      );
    }

    getEnvironmentCellList(cellIndex) {
      if (Array.isArray(cellIndex)) return cellIndex.filter((value) => Number.isFinite(Number(value)));
      if (cellIndex == null) return [];
      return [cellIndex];
    }

    pickEnvironmentCell(cellIndex, seed, salt) {
      const list = this.getEnvironmentCellList(cellIndex);
      if (!list.length) return null;
      if (list.length === 1) return list[0];
      const index = Math.min(list.length - 1, Math.floor(seededUnit(seed, salt) * list.length));
      return list[index];
    }

    getPlatformTerrainBodyDepth(platforms, platform, index, style, topH, isGround) {
      const baseDepth = Math.max(24, isGround ? Number(style.groundBodyDepth || 92) : Number(style.platformBodyDepth || 36));
      if (isGround) return baseDepth;
      const platformX = Number(platform && platform.x || 0);
      const platformY = Number(platform && platform.y || 0);
      const platformW = Math.max(1, Number(platform && platform.w || 0));
      const platformH = Math.max(1, Number(platform && platform.h || 22));
      const platformCap = Math.max(20, platformH + 16);
      let availableDepth = Infinity;
      (Array.isArray(platforms) ? platforms : []).forEach((candidate, candidateIndex) => {
        if (!candidate || candidateIndex === index) return;
        const candidateY = Number(candidate.y || 0);
        if (candidateY <= platformY + platformH) return;
        const candidateX = Number(candidate.x || 0);
        const candidateW = Math.max(1, Number(candidate.w || 0));
        const overlap = Math.min(platformX + platformW, candidateX + candidateW) - Math.max(platformX, candidateX);
        if (overlap <= 24) return;
        const candidateTopH = candidateIndex === 0 || Number(candidate.h || 0) >= 56
          ? Number(style.groundTopHeight || 28)
          : topH;
        const clearance = candidateY - candidateTopH - platformY - 8;
        if (clearance > 0) availableDepth = Math.min(availableDepth, clearance);
      });
      const cappedDepth = Math.min(baseDepth, platformCap, Number.isFinite(availableDepth) ? availableDepth : baseDepth);
      return Math.max(18, Math.round(cappedDepth));
    }

    getEnvironmentCellTexture(kind, profile, cellIndex) {
      const asset = this.getEnvironmentAsset(kind, profile);
      if (!asset || !asset.path) return null;
      const base = this.getTexture(asset.path);
      if (!base) return null;
      const size = Math.max(1, Number(asset.cellSize || 64));
      const columns = Math.max(1, Number(asset.columns || 4));
      const safeCell = Math.max(0, Number(cellIndex || 0) || 0);
      const key = `${asset.path}:${safeCell}:${size}:${columns}`;
      const cached = this.getCacheValue(this.environmentTextures, key);
      if (cached) return cached;
      const { Rectangle, Texture } = this.PIXI;
      const source = base.source || base.baseTexture || base;
      const rectangle = new Rectangle((safeCell % columns) * size, Math.floor(safeCell / columns) * size, size, size);
      let texture = null;
      try {
        texture = new Texture({ source, frame: rectangle });
      } catch {
        try {
          texture = new Texture(source, rectangle);
        } catch {
          texture = null;
        }
      }
      return texture ? this.setCacheValue(this.environmentTextures, key, texture, ENVIRONMENT_TEXTURE_CACHE_LIMIT) : null;
    }

    drawEnvironmentCell(kind, profile, cellIndex, x, y, w, h, options) {
      const safeCell = Array.isArray(cellIndex) ? cellIndex[0] : cellIndex;
      const texture = this.getEnvironmentCellTexture(kind, profile, safeCell);
      if (!texture) return false;
      const settings = options || {};
      const asset = settings.trim ? this.getEnvironmentAsset(kind, profile) : null;
      const trimKey = asset && asset.path ? `${asset.path}:${safeCell}:${asset.cellSize || 64}:${asset.columns || 4}:trimmed` : '';
      const trimmed = trimKey ? this.getTrimmedTexture(texture, trimKey) : null;
      return this.drawTexture('map', trimmed && trimmed.texture || texture, x, y, w, h, {
        anchorX: 0,
        anchorY: 0,
        alpha: settings.alpha,
        flipX: !!settings.flip
      });
    }

    drawEnvironmentTileStrip(snapshot, kind, profile, cellIndex, x, y, w, h, seed, alternateCellIndex, options) {
      const asset = this.getEnvironmentAsset(kind, profile);
      if (!asset) return false;
      const tileW = Math.max(1, Number(asset.cellSize || 64));
      const bounds = snapshot && snapshot.bounds || null;
      const leftLimit = bounds ? Number(bounds.left || 0) - tileW * 2 : x;
      const rightLimit = bounds ? Number(bounds.right || 0) + tileW * 2 : x + w;
      const firstOffset = Math.max(0, Math.floor((leftLimit - x) / tileW));
      const end = x + w;
      const tileOverlap = Math.max(0, Math.min(18, Number(options && options.overlap != null ? options.overlap : 2) || 0));
      let drew = false;
      for (let tx = x + firstOffset * tileW; tx < end && tx < rightLimit; tx += tileW) {
        const drawX = tx - (tx > x ? tileOverlap : 0);
        const drawRight = Math.min(end, tx + tileW + (tx + tileW < end ? tileOverlap : 0));
        const drawW = Math.max(0, drawRight - drawX);
        if (drawW <= 0 || drawX + drawW < leftLimit) continue;
        const salt = Math.round(tx / tileW);
        const useAlt = alternateCellIndex != null && seededUnit(seed, salt) > 0.68;
        const cell = this.pickEnvironmentCell(useAlt ? alternateCellIndex : cellIndex, seed, `cell:${salt}`);
        if (cell != null && this.drawEnvironmentCell(kind, profile, cell, drawX, y, drawW, h, options)) drew = true;
      }
      return drew;
    }

    getTerrainSurfaceCells(cells, isGround) {
      const source = cells || DEFAULT_ENVIRONMENT_TERRAIN_CELLS;
      return {
        left: isGround ? source.groundLeft : source.platformLeft,
        mid: isGround ? source.groundMid : source.platformMid,
        right: isGround ? source.groundRight : source.platformRight
      };
    }

    drawEnvironmentModularStrip(snapshot, kind, profile, leftCell, midCells, rightCell, x, y, w, h, seed, options) {
      const width = Math.max(0, Number(w || 0));
      const asset = this.getEnvironmentAsset(kind, profile);
      if (!width || !asset) return false;
      const tileW = Math.max(1, Number(asset.cellSize || 64));
      let midCell = this.pickEnvironmentCell(midCells, seed, 'mid');
      if (midCell == null) midCell = leftCell == null ? rightCell : leftCell;
      const fixedEdgeCells = !!(options && options.fullEdgeCells);
      if ((!fixedEdgeCells && width <= tileW * 1.35) || (fixedEdgeCells && width < tileW * 2) || leftCell == null || rightCell == null) {
        if (midCell == null) return false;
        return this.drawEnvironmentTileStrip(snapshot, kind, profile, midCell, x, y, width, h, `${seed}:mid`, null, options);
      }
      const edgeW = fixedEdgeCells ? tileW : Math.min(tileW, Math.max(28, Math.floor(width * 0.24)));
      let drew = false;
      if (this.drawEnvironmentCell(kind, profile, leftCell, x, y, edgeW, h, options)) drew = true;
      if (this.drawEnvironmentCell(kind, profile, rightCell, x + width - edgeW, y, edgeW, h, options)) drew = true;
      const midX = x + edgeW;
      const midW = Math.max(0, width - edgeW * 2);
      if (midW > 0 && midCell != null && this.drawEnvironmentTileStrip(snapshot, kind, profile, midCells, midX, y, midW, h, `${seed}:mid`, null, options)) drew = true;
      return drew;
    }

    drawTerrainSurface(snapshot, profile, cells, isGround, x, y, w, h, seed, options) {
      const surface = this.getTerrainSurfaceCells(cells, isGround);
      const drawOptions = Object.assign({}, options || {}, { fullEdgeCells: true });
      return this.drawEnvironmentModularStrip(snapshot, 'terrain', profile, surface.left, surface.mid, surface.right, x, y, w, h, seed, drawOptions);
    }

    drawTerrainBodyBand(snapshot, profile, cells, x, y, w, bodyH, undersideH, seed, alpha, options) {
      const safeBodyH = Math.max(0, Number(bodyH || 0));
      if (!safeBodyH) return false;
      const source = cells || DEFAULT_ENVIRONMENT_TERRAIN_CELLS;
      const drawOptions = Object.assign({ alpha }, options || {});
      let drew = this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, source.body, x, y, w, safeBodyH, `${seed}:body`, null, drawOptions);
      const safeUnderH = Math.max(0, Math.min(Number(undersideH || 0), safeBodyH));
      if (safeUnderH > 0) {
        const underY = y + safeBodyH - Math.round(safeUnderH * 0.62);
        if (this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, source.underside, x, underY, w, safeUnderH, `${seed}:under`, null, drawOptions)) drew = true;
      }
      return drew;
    }

    getMapThemeId(map) {
      const profile = this.getEnvironmentProfile(map);
      return String(profile && (profile.terrain || profile.props) || map && map.id || '').toLowerCase();
    }

    getEnvironmentVisibility(profile) {
      return profile && profile.visibility || this.data && this.data.ENVIRONMENT_READABILITY_DEFAULTS || DEFAULT_ENVIRONMENT_VISIBILITY;
    }

    getEnvironmentPropKinds(profile, layer, platformIndex) {
      const allKinds = profile && Array.isArray(profile.propKinds) ? profile.propKinds : ['grass', 'bush', 'rock'];
      const visibility = this.getEnvironmentVisibility(profile);
      if (layer === 'rear') {
        if (visibility.groundOnlyTallProps && platformIndex !== 0) return [];
        const rearKinds = this.data && this.data.ENVIRONMENT_REAR_PROP_KINDS || DEFAULT_ENVIRONMENT_REAR_PROP_KINDS;
        return allKinds.filter((kind) => rearKinds.includes(kind));
      }
      if (layer === 'front') {
        const frontKinds = platformIndex === 0
          ? this.data && this.data.ENVIRONMENT_FRONT_PROP_KINDS || DEFAULT_ENVIRONMENT_FRONT_PROP_KINDS
          : this.data && this.data.ENVIRONMENT_UPPER_FRONT_PROP_KINDS || DEFAULT_ENVIRONMENT_UPPER_FRONT_PROP_KINDS;
        return allKinds.filter((kind) => frontKinds.includes(kind));
      }
      return [];
    }

    getEnvironmentPropSize(kind, layer, platformIndex, visibility) {
      const base = {
        grass: [32, 18],
        bush: [44, 30],
        tree: [70, 82],
        rock: [36, 22],
        flower: [24, 24],
        small: [36, 28],
        tall: [34, 56],
        crate: [36, 30],
        crystal: [34, 42],
        vine: [30, 58],
        sign: [32, 38],
        glow: [34, 22]
      }[kind] || [34, 26];
      const rules = visibility || this.getEnvironmentVisibility(null);
      const layerScale = layer === 'rear' ? 0.98 : 1;
      const platformScale = platformIndex === 0 ? 1 : Number(rules.upperPlatformPropScale || 0.56);
      const w = Math.round(base[0] * layerScale * platformScale);
      let h = Math.round(base[1] * layerScale * platformScale);
      if (layer === 'front') {
        h = Math.min(h, Number(rules.maxFrontPropHeight || 38));
        if (platformIndex !== 0) h = Math.min(h, Number(rules.maxUpperPropHeight || 20));
      }
      return { w, h };
    }

    getMapDecorationBlockers(runtime) {
      const blockers = [];
      const safeRuntime = runtime || {};
      (safeRuntime.climbables || []).forEach((item) => blockers.push({ x: item.x - 36, y: item.y - 16, w: item.w + 72, h: item.h + 32 }));
      (safeRuntime.stations || []).forEach((item) => blockers.push({ x: item.x - 52, y: item.y - 90, w: item.w + 104, h: item.h + 112 }));
      (safeRuntime.portals || []).forEach((item) => blockers.push({ x: item.x - 56, y: item.y - 40, w: item.w + 112, h: item.h + 74 }));
      (safeRuntime.questNpcs || []).forEach((item) => blockers.push({ x: item.x - 44, y: item.y - 54, w: item.w + 88, h: item.h + 78 }));
      (safeRuntime.spawnPoints || []).forEach((point) => {
        const platform = safeRuntime.platforms && safeRuntime.platforms[point.platformIndex || 0];
        if (platform) blockers.push({ x: Number(point.x || 0) - 34, y: platform.y - 72, w: 68, h: 94 });
      });
      return blockers;
    }

    getSnapshotMapDecorationBlockers(snapshot, runtime) {
      if (snapshot && Array.isArray(snapshot._mapDecorationBlockers)) return snapshot._mapDecorationBlockers;
      const blockers = this.getMapDecorationBlockers(runtime);
      if (snapshot) snapshot._mapDecorationBlockers = blockers;
      return blockers;
    }

    isEnvironmentPropPlacementSafe(platform, x, y, w, h, blockers, visibility) {
      if (!platform || x < platform.x + 28 || x + w > platform.x + platform.w - 28) return false;
      const clearance = Number(visibility && visibility.combatClearancePx || 72);
      const rect = { x: x - clearance * 0.25, y: y - clearance * 0.2, w: w + clearance * 0.5, h: h + clearance * 0.35 };
      return !(blockers || []).some((blocker) => rectsOverlap(rect, blocker));
    }

    drawMapProp(profile, kind, x, y, w, h, seed, layer) {
      const cells = this.getEnvironmentCells('props');
      const cell = cells[kind] == null ? cells.grass : cells[kind];
      const alpha = layer === 'rear' ? 0.68 : 0.88;
      const flip = seededUnit(seed, 'flip') > 0.5;
      return this.drawEnvironmentCell('props', profile, cell, x, y, w, h, { alpha, flip, trim: true });
    }

    getMapSceneryRuntimeSignature(snapshot, runtime) {
      if (snapshot && snapshot._mapSceneryRuntimeSignature) return snapshot._mapSceneryRuntimeSignature;
      const safeRuntime = runtime || {};
      const rectSignature = (list, fields) => (list || [])
        .map((item) => fields.map((field) => Math.round(Number(item && item[field] || 0))).join(','))
        .join(';');
      const signature = [
        safeRuntime.id || '',
        safeRuntime.trialId || '',
        Math.round(Number(safeRuntime.worldWidth || 0)),
        Math.round(Number(safeRuntime.worldHeight || 0)),
        rectSignature(safeRuntime.platforms, ['x', 'y', 'y2', 'w', 'h']),
        rectSignature(safeRuntime.climbables, ['x', 'y', 'w', 'h']),
        rectSignature(safeRuntime.stations, ['x', 'y', 'w', 'h']),
        rectSignature(safeRuntime.portals, ['x', 'y', 'w', 'h']),
        rectSignature(safeRuntime.questNpcs, ['x', 'y', 'w', 'h']),
        rectSignature(safeRuntime.spawnPoints, ['x', 'platformIndex'])
      ].join('|');
      if (snapshot) snapshot._mapSceneryRuntimeSignature = signature;
      return signature;
    }

    getMapSceneryPlacementCacheKey(snapshot, runtime, map, profile, visibility, layer, densityScale) {
      return [
        map && map.id || '',
        layer || '',
        profile && profile.id || '',
        profile && profile.props || '',
        Number(profile && profile.density || 0),
        Number(densityScale || 0),
        Number(visibility && visibility.combatClearancePx || 0),
        Number(visibility && visibility.upperPlatformPropScale || 0),
        Number(visibility && visibility.maxFrontPropHeight || 0),
        Number(visibility && visibility.maxUpperPropHeight || 0),
        this.getMapSceneryRuntimeSignature(snapshot, runtime)
      ].join('::');
    }

    buildMapSceneryPlacements(snapshot, runtime, map, profile, visibility, layer, densityScale) {
      const blockers = this.getSnapshotMapDecorationBlockers(snapshot, runtime);
      const density = Number(profile.density || 0.5) * densityScale;
      const spacing = layer === 'rear' ? 430 : 340;
      const placements = [];
      (runtime.platforms || []).forEach((platform, platformIndex) => {
        if (!platform || platform.w < 120) return;
        const kindPool = this.getEnvironmentPropKinds(profile, layer, platformIndex);
        if (!kindPool.length) return;
        const rawCount = Math.max(1, Math.round(platform.w / spacing * density));
        const count = layer === 'rear' ? Math.min(8, rawCount) : Math.min(platformIndex === 0 ? 6 : 2, rawCount);
        for (let index = 0; index < count; index += 1) {
          const seed = `${map.id}:${layer}:${platformIndex}:${index}`;
          const kind = seededPick(kindPool, seed, 'kind') || 'grass';
          const size = this.getEnvironmentPropSize(kind, layer, platformIndex, visibility);
          const usableW = Math.max(1, platform.w - 96);
          const x = platform.x + 48 + Math.floor(usableW * ((index + 0.35 + seededUnit(seed, 'x') * 0.3) / Math.max(1, count)));
          const overlap = layer === 'front' ? 0 : 10 + Math.floor(seededUnit(seed, 'y') * 6);
          const y = platform.y - size.h + overlap;
          if (!this.isEnvironmentPropPlacementSafe(platform, x, y, size.w, size.h, blockers, visibility)) continue;
          placements.push({ kind, x, y, w: size.w, h: size.h, seed });
        }
      });
      return placements;
    }

    getMapSceneryPlacements(snapshot, runtime, map, profile, visibility, layer, densityScale) {
      if (!this.mapSceneryPlacementCache) this.mapSceneryPlacementCache = new Map();
      const key = this.getMapSceneryPlacementCacheKey(snapshot, runtime, map, profile, visibility, layer, densityScale);
      const cached = this.getCacheValue(this.mapSceneryPlacementCache, key);
      if (cached) return cached;
      const placements = this.buildMapSceneryPlacements(snapshot, runtime, map, profile, visibility, layer, densityScale);
      return this.setCacheValue(this.mapSceneryPlacementCache, key, placements, MAP_SCENERY_PLACEMENT_CACHE_LIMIT);
    }

    renderMapScenery(snapshot, map, layer) {
      const profile = this.getEnvironmentProfile(map);
      const propAsset = this.getEnvironmentAsset('props', profile);
      if (!profile || !propAsset || !propAsset.path) return;
      const visibility = this.getEnvironmentVisibility(profile);
      const densityScale = layer === 'rear'
        ? Number(visibility.rearDensityScale == null ? 0.34 : visibility.rearDensityScale)
        : Number(visibility.frontDensityScale == null ? 0 : visibility.frontDensityScale);
      if (densityScale <= 0) return;
      const runtime = snapshot.runtime || {};
      const bounds = snapshot.bounds || {};
      const placements = this.getMapSceneryPlacements(snapshot, runtime, map, profile, visibility, layer, densityScale);
      placements.forEach((placement) => {
        if (!isRectInBounds(placement, bounds, 80)) return;
        this.drawMapProp(profile, placement.kind, placement.x, placement.y, placement.w, placement.h, placement.seed, layer);
      });
    }

    getEnvironmentStructureAsset() {
      const group = this.data && this.data.ENVIRONMENT_STRUCTURE_ASSETS || {};
      return group.townLandmarks || null;
    }

    getEnvironmentStructureCellIndex(cell) {
      const cells = this.data && this.data.ENVIRONMENT_STRUCTURE_CELLS || DEFAULT_ENVIRONMENT_STRUCTURE_CELLS;
      const key = String(cell || '').trim();
      return cells[key] == null ? DEFAULT_ENVIRONMENT_STRUCTURE_CELLS.starfallGuildHall : cells[key];
    }

    getEnvironmentStructureCellTexture(cell) {
      const asset = this.getEnvironmentStructureAsset();
      if (!asset || !asset.path) return null;
      const base = this.getTexture(asset.path);
      if (!base) return null;
      const size = Math.max(1, Number(asset.cellSize || 256));
      const columns = Math.max(1, Number(asset.columns || 4));
      const safeCell = Math.max(0, Number(this.getEnvironmentStructureCellIndex(cell) || 0) || 0);
      const key = `${asset.path}:${safeCell}:${size}:${columns}`;
      const cached = this.getCacheValue(this.environmentTextures, key);
      if (cached) return cached;
      const { Rectangle, Texture } = this.PIXI;
      const source = base.source || base.baseTexture || base;
      const rectangle = new Rectangle((safeCell % columns) * size, Math.floor(safeCell / columns) * size, size, size);
      let texture = null;
      try {
        texture = new Texture({ source, frame: rectangle });
      } catch {
        try {
          texture = new Texture(source, rectangle);
        } catch {
          texture = null;
        }
      }
      return texture ? this.setCacheValue(this.environmentTextures, key, texture, ENVIRONMENT_TEXTURE_CACHE_LIMIT) : null;
    }

    drawEnvironmentStructureCell(cell, x, y, w, h, options) {
      const texture = this.getEnvironmentStructureCellTexture(cell);
      if (!texture) return false;
      const settings = options || {};
      return this.drawTexture('map', texture, x, y, w, h, {
        anchorX: 0,
        anchorY: 0,
        alpha: settings.alpha,
        flipX: !!settings.flip
      });
    }

    getGroundPlatform(runtime, snapshot) {
      const safeRuntime = runtime || {};
      return safeRuntime.platforms && safeRuntime.platforms[0] || {
        x: 0,
        y: Number(snapshot && snapshot.playfieldHeight || this.height || 0),
        w: Number(safeRuntime.worldWidth || snapshot && snapshot.width || this.width || 1)
      };
    }

    getStationForSceneEntry(runtime, entry) {
      const stationId = String(entry && entry.stationId || '').trim();
      if (!stationId) return null;
      return (runtime && runtime.stations || []).find((station) => station && station.id === stationId) || null;
    }

    renderTownStructureEntry(snapshot, runtime, entry, groundY) {
      if (!entry) return;
      const station = this.getStationForSceneEntry(runtime, entry);
      const w = Math.max(24, Number(entry.w || 0));
      const h = Math.max(24, Number(entry.h || 0));
      const x = station ? Number(station.x || 0) + Number(entry.dx || 0) : Number(entry.x || 0);
      const baseY = station ? Number(station.y || 0) + 44 : groundY;
      const y = baseY - h + Number(entry.footOffset || 0);
      if (!isRectInBounds({ x, y, w, h }, snapshot.bounds, 240)) return;
      this.drawEnvironmentStructureCell(entry.cell, x, y, w, h, {
        alpha: entry.alpha == null ? 0.96 : Number(entry.alpha)
      });
    }

    renderTownStreetProp(snapshot, runtime, profile, entry, groundPlatform, seed, blockers) {
      if (!entry || !groundPlatform) return;
      const w = Math.max(12, Number(entry.w || 32));
      const h = Math.max(12, Number(entry.h || 24));
      const x = Number(entry.x || 0);
      const y = groundPlatform.y - h + Number(entry.footOffset || 0);
      const visibility = this.getEnvironmentVisibility(profile);
      const placementBlockers = blockers || this.getSnapshotMapDecorationBlockers(snapshot, runtime);
      if (!isRectInBounds({ x, y, w, h }, snapshot.bounds, 120)) return;
      if (!this.isEnvironmentPropPlacementSafe(groundPlatform, x, y, w, h, placementBlockers, visibility)) return;
      this.drawMapProp(profile, entry.kind || 'grass', x, y, w, h, seed, 'front');
    }

    renderTownForegroundTrim(snapshot, runtime, profile, entry, groundPlatform, seedPrefix, blockers) {
      if (!entry || !groundPlatform) return;
      const startX = Math.max(groundPlatform.x + 28, Number(entry.startX || groundPlatform.x));
      const endX = Math.min(groundPlatform.x + groundPlatform.w - 28, Number(entry.endX || groundPlatform.x + groundPlatform.w));
      const every = Math.max(80, Number(entry.every || 360));
      const w = Math.max(10, Number(entry.w || 28));
      const h = Math.max(8, Number(entry.h || 18));
      const visibility = this.getEnvironmentVisibility(profile);
      const placementBlockers = blockers || this.getSnapshotMapDecorationBlockers(snapshot, runtime);
      for (let x = startX, index = 0; x <= endX; x += every, index += 1) {
        const jitter = Math.floor((seededUnit(seedPrefix, index) - 0.5) * Math.min(72, every * 0.24));
        const drawX = x + jitter;
        const y = groundPlatform.y - h + Number(entry.footOffset || 0);
        if (!isRectInBounds({ x: drawX, y, w, h }, snapshot.bounds, 120)) continue;
        if (!this.isEnvironmentPropPlacementSafe(groundPlatform, drawX, y, w, h, placementBlockers, visibility)) continue;
        this.drawMapProp(profile, entry.kind || 'grass', drawX, y, w, h, `${seedPrefix}:${index}`, 'front');
      }
    }

    renderTownStructures(snapshot, map, layer) {
      const scene = map && map.townScene;
      if (!scene) return;
      const runtime = snapshot.runtime || {};
      const groundPlatform = this.getGroundPlatform(runtime, snapshot);
      const groundY = Number(groundPlatform && groundPlatform.y || snapshot.playfieldHeight || 0);
      if (layer === 'rear') {
        (scene.rearStructures || []).forEach((entry) => this.renderTownStructureEntry(snapshot, runtime, entry, groundY));
        (scene.stationFacades || []).forEach((entry) => this.renderTownStructureEntry(snapshot, runtime, entry, groundY));
        return;
      }
      const profile = this.getEnvironmentProfile(map);
      const blockers = this.getSnapshotMapDecorationBlockers(snapshot, runtime);
      (scene.streetProps || []).forEach((entry, index) => {
        this.renderTownStreetProp(snapshot, runtime, profile, entry, groundPlatform, `${map.id}:town-prop:${index}`, blockers);
      });
      (scene.foregroundTrim || []).forEach((entry, index) => {
        this.renderTownForegroundTrim(snapshot, runtime, profile, entry, groundPlatform, `${map.id}:town-trim:${index}`, blockers);
      });
    }

    renderFieldCompositionLandmarks(snapshot, map) {
      const composition = map && map.fieldComposition;
      if (!composition) return;
      const profile = this.getEnvironmentProfile(map);
      const propAsset = this.getEnvironmentAsset('props', profile);
      if (!profile || !propAsset || !propAsset.path) return;
      const runtime = snapshot.runtime || {};
      const groundPlatform = this.getGroundPlatform(runtime, snapshot);
      const blockers = this.getSnapshotMapDecorationBlockers(snapshot, runtime);
      const visibility = this.getEnvironmentVisibility(profile);
      (composition.landmarkBands || []).forEach((band, bandIndex) => {
        const kind = band.kind || 'tree';
        const size = this.getEnvironmentPropSize(kind, 'rear', 0, visibility);
        const startX = Math.max(groundPlatform.x + 48, Number(band.x || groundPlatform.x + 48));
        const width = Math.max(120, Number(band.w || 480));
        const count = clamp(Math.floor(width / 360), 1, 4);
        for (let index = 0; index < count; index += 1) {
          const seed = `${map.id}:field-landmark:${bandIndex}:${index}`;
          const x = startX + Math.floor(width * ((index + 0.38 + seededUnit(seed, 'x') * 0.24) / Math.max(1, count)));
          const scale = 1.12 + seededUnit(seed, 'scale') * 0.34;
          const w = Math.round(size.w * scale);
          const h = Math.round(size.h * scale);
          const y = groundPlatform.y - h + 10 + Math.floor(seededUnit(seed, 'y') * 6);
          if (!isRectInBounds({ x, y, w, h }, snapshot.bounds, 180)) continue;
          if (!this.isEnvironmentPropPlacementSafe(groundPlatform, x, y, w, h, blockers, visibility)) continue;
          this.drawMapProp(profile, kind, x, y, w, h, seed, 'rear');
        }
      });
      (composition.routeSections || []).forEach((section, index) => {
        if (index <= 0) return;
        const x = Number(section.x || 0);
        const w = 34;
        const h = 42;
        const y = groundPlatform.y - h + 2;
        if (!isRectInBounds({ x, y, w, h }, snapshot.bounds, 140)) return;
        if (!this.isEnvironmentPropPlacementSafe(groundPlatform, x, y, w, h, blockers, visibility)) return;
        this.drawMapProp(profile, 'sign', x, y, w, h, `${map.id}:route-marker:${index}`, 'rear');
      });
    }

    drawPlatformFallback(graphics, map, platform, index) {
      const palette = Array.isArray(map && map.palette) ? map.palette : [];
      const bodyColor = map && map.id === 'cinderHollow' ? 0x332a2f : map && map.id === 'rustcoilRuins' ? 0x5f6872 : 0x4c8b5c;
      const topColor = colorToNumber(palette[2], 0xf3d86d);
      if (isSlopePlatform(platform)) {
        const x = Number(platform.x || 0);
        const w = Number(platform.w || 0);
        const leftY = Number(platform.y || 0);
        const rightY = Number(platform.y2 || platform.y || 0);
        const thickness = Math.max(18, Math.min(30, Number(platform.h || 24) + 4));
        const topInset = Math.max(3, Math.min(6, Math.round(thickness * 0.16)));
        graphics
          .moveTo(x, leftY - topInset)
          .lineTo(x + w, rightY - topInset)
          .lineTo(x + w, rightY + thickness)
          .lineTo(x, leftY + thickness)
          .closePath()
          .fill({ color: bodyColor, alpha: index === 0 ? 0.96 : 0.9 });
        graphics
          .moveTo(x, leftY)
          .lineTo(x + w, rightY)
          .stroke({ width: 5, color: topColor, alpha: index === 0 ? 0.92 : 0.84 });
        this.drawPlatformThemeTrim(graphics, map, platform, index);
        return;
      }
      const y = Number(platform.y || 0);
      const h = Math.max(12, Number(platform.h || 0));
      const topY = y - (index === 0 ? 10 : 7);
      const topH = index === 0 ? 16 : 12;
      graphics
        .rect(platform.x, topY, platform.w, topH)
        .fill({ color: topColor, alpha: index === 0 ? 0.92 : 0.84 })
        .rect(platform.x, y + 4, platform.w, h + (index === 0 ? 22 : 10))
        .fill({ color: bodyColor, alpha: index === 0 ? 0.96 : 0.9 })
        .rect(platform.x, topY, platform.w, 2)
        .fill({ color: 0xffffff, alpha: 0.16 });
      this.drawPlatformThemeTrim(graphics, map, platform, index);
    }

    drawPlatformThemeTrim(graphics, map, platform, index) {
      if (!platform || platform.w <= 80) return;
      const theme = this.getMapThemeId(map);
      const palette = map && map.palette || [];
      const accent = colorToNumber(palette[2] || palette[1], 0xf3d86d);
      if (isSlopePlatform(platform)) {
        graphics
          .moveTo(platform.x + 12, getPlatformSurfaceY(platform, platform.x + 12))
          .lineTo(platform.x + platform.w - 12, getPlatformSurfaceY(platform, platform.x + platform.w - 12))
          .stroke({ width: 2.5, color: accent, alpha: index === 0 ? 0.38 : 0.58 });
        for (let x = platform.x + 42; x < platform.x + platform.w - 26; x += 86) {
          graphics.rect(x, getPlatformSurfaceY(platform, x) - 3, 18, 3).fill({ color: 0xffffff, alpha: 0.16 });
        }
        return;
      }
      if (theme.includes('cinder') || theme.includes('ember') || theme.includes('fire')) {
        graphics.moveTo(platform.x + 14, platform.y + 4).lineTo(platform.x + platform.w - 14, platform.y + 4)
          .stroke({ width: 2, color: 0xff8a3d, alpha: index === 0 ? 0.38 : 0.54 });
        for (let x = platform.x + 48; x < platform.x + platform.w - 36; x += 118) {
          graphics.moveTo(x, platform.y - 2).lineTo(x + 18, platform.y + 7)
            .stroke({ width: 1.4, color: 0xffcf70, alpha: index === 0 ? 0.2 : 0.34 });
        }
      } else if (theme.includes('ruins') || theme.includes('gearworks') || theme.includes('quarry')) {
        for (let x = platform.x + 34; x < platform.x + platform.w - 20; x += 92) {
          graphics.rect(x, platform.y + 3, 5, 5).fill({ color: accent, alpha: 0.62 });
        }
        for (let x = platform.x + 78; x < platform.x + platform.w - 30; x += 128) {
          graphics.moveTo(x, platform.y - 3).lineTo(x, platform.y + 8)
            .stroke({ width: 1, color: 0x102033, alpha: 0.2 });
        }
      } else if (theme.includes('frost')) {
        graphics.moveTo(platform.x + 12, platform.y - 2).lineTo(platform.x + platform.w - 12, platform.y - 2)
          .stroke({ width: 3, color: 0xf7fbff, alpha: 0.58 });
        for (let x = platform.x + 42; x < platform.x + platform.w - 30; x += 104) {
          graphics.rect(x, platform.y - 5, 34, 3).fill({ color: 0xffffff, alpha: index === 0 ? 0.22 : 0.32 });
        }
      } else if (theme.includes('astral') || theme.includes('eclipse') || theme.includes('rift') || theme.includes('rune')) {
        for (let x = platform.x + 18; x < platform.x + platform.w - 18; x += 28) {
          graphics.rect(x, platform.y + 5, 12, 2).fill({ color: 0x7bdff2, alpha: 0.46 });
        }
        const runeColor = theme.includes('eclipse') || theme.includes('rift') ? 0xf06bff : 0xc794ff;
        for (let x = platform.x + 46; x < platform.x + platform.w - 42; x += 132) {
          graphics.rect(x, platform.y + 1, 10, 2).fill({ color: runeColor, alpha: 0.38 });
          graphics.rect(x + 4, platform.y - 3, 2, 10).fill({ color: runeColor, alpha: 0.38 });
        }
      } else if (theme.includes('storm')) {
        for (let x = platform.x + 24; x < platform.x + platform.w - 24; x += 96) {
          graphics.moveTo(x, platform.y + 4).lineTo(x + 22, platform.y + 4)
            .stroke({ width: 2, color: 0xffe16a, alpha: 0.44 });
        }
        for (let x = platform.x + 62; x < platform.x + platform.w - 28; x += 124) {
          graphics.rect(x, platform.y - 2, 4, 7).fill({ color: 0x7bdff2, alpha: 0.28 });
        }
      } else if (theme.includes('bandit') || theme.includes('ridge') || theme.includes('duelist') || theme.includes('sniper')) {
        for (let x = platform.x + 34; x < platform.x + platform.w - 22; x += 78) {
          graphics.moveTo(x, platform.y - 3).lineTo(x + 7, platform.y + 7)
            .stroke({ width: 1.5, color: 0x5b3d2d, alpha: index === 0 ? 0.32 : 0.5 });
        }
      } else {
        for (let x = platform.x + 28; x < platform.x + platform.w - 20; x += 58) {
          graphics.rect(x, platform.y - 4, 3, 6).fill({ color: 0x8ec878, alpha: index === 0 ? 0.24 : 0.36 });
          graphics.rect(x + 6, platform.y - 7, 2, 8).fill({ color: 0x8ec878, alpha: index === 0 ? 0.24 : 0.36 });
        }
      }
    }

    getPlatformTerrainVisual(platform, index) {
      const visual = platform && platform.terrainVisual;
      if (!visual || index === 0 || visual.kind === 'ground') return null;
      return visual;
    }

    drawFieldPlatformLedge(snapshot, map, platform, index, profile, cells, style, seed) {
      const ledgeOverhang = Math.max(10, Math.min(18, Number(style.overhang || 12)));
      const topH = Math.max(14, Math.min(22, Math.round(Number(style.topHeight || 24) * 0.72)));
      const ledgeH = Math.max(38, Math.min(48, topH + Math.round(Number(style.platformBodyDepth || 30) * 0.82)));
      const ledgeX = platform.x - ledgeOverhang;
      const ledgeW = platform.w + ledgeOverhang * 2;
      const ledgeY = platform.y - topH + 2;
      this.drawTerrainSurface(snapshot, profile, cells, false, ledgeX, ledgeY, ledgeW, ledgeH, `${seed}:ledge`, { overlap: 0 });
      this.drawPlatformThemeTrim(this.mapGraphics, map, platform, index);
    }

    drawFieldTerrainIsland(snapshot, map, platform, index, segment, segmentIndex, profile, cells, style, seed) {
      if (!segment || !platform) return;
      const asset = this.getEnvironmentAsset('terrain', profile);
      const tile = Math.max(1, Number(asset && asset.cellSize || 64));
      const bounds = snapshot.bounds || {};
      const platformW = Math.max(1, Number(platform.w || 0));
      const segmentX = clamp(Number(segment.x || 0), 24, Math.max(24, platformW - 104));
      const availableW = Math.max(0, platformW - segmentX - 24);
      const segmentW = clamp(Number(segment.w || 0), 96, Math.max(96, availableW));
      if (availableW < 96 || segmentW < 96) return;
      const islandSeed = `${seed}:island:${segmentIndex}`;
      const overhang = Math.max(8, Math.min(14, Number(style.overhang || 12)));
      const topH = Math.max(14, Math.min(20, Math.round(Number(style.topHeight || 24) * 0.78)));
      const bodyBaseH = clamp(Number(segment.depth || style.platformBodyDepth || 30), 20, 44);
      const undersideH = Math.max(10, Math.min(16, Number(style.undersideHeight || 18)));
      const jitter = Math.max(0, Math.min(8, Number(style.undersideJitter || 10)));
      const alpha = style.bodyAlpha == null ? 1 : Number(style.bodyAlpha);
      const x = platform.x + segmentX;
      const left = x - overhang;
      const right = x + segmentW + overhang;
      const topY = platform.y - topH + 2;
      const bodyY = platform.y;
      const leftLimit = Number(bounds.left || 0) - tile * 2;
      const rightLimit = Number(bounds.right || 0) + tile * 2;
      const shadowCell = this.pickEnvironmentCell(cells.shadow, islandSeed, 'shadow');
      if (shadowCell != null && isRectInBounds({ x: left, y: platform.y + 13, w: right - left, h: 18 }, bounds, 120)) {
        this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, shadowCell, left, platform.y + 13, right - left, 18, `${islandSeed}:shadow`);
      }
      const firstOffset = Math.max(0, Math.floor((leftLimit - left) / tile));
      for (let tx = left + firstOffset * tile; tx < right && tx < rightLimit; tx += tile) {
        const drawW = Math.min(tile, right - tx);
        if (tx + drawW < leftLimit) continue;
        const ordinal = Math.round((tx - left) / tile);
        const drop = Math.round((seededUnit(islandSeed, `drop:${ordinal}`) - 0.35) * jitter);
        const chunkH = Math.max(18, Math.min(48, bodyBaseH + drop));
        const bodyCells = seededUnit(islandSeed, `deep:${ordinal}`) > 0.64 && cells.bodyDeep ? cells.bodyDeep : cells.body;
        const bodyCell = this.pickEnvironmentCell(bodyCells, islandSeed, `body:${ordinal}`);
        if (bodyCell != null && isRectInBounds({ x: tx, y: bodyY, w: drawW, h: chunkH }, bounds, 120)) {
          this.drawEnvironmentCell('terrain', profile, bodyCell, tx, bodyY, drawW, chunkH, { alpha });
        }
        const underCell = this.pickEnvironmentCell(cells.underside, islandSeed, `under:${ordinal}`);
        const underY = bodyY + chunkH - Math.round(undersideH * 0.34);
        if (underCell != null && isRectInBounds({ x: tx, y: underY, w: drawW, h: undersideH }, bounds, 120)) {
          this.drawEnvironmentCell('terrain', profile, underCell, tx, underY, drawW, undersideH, { alpha });
        }
      }
      this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, cells.top, left, topY, right - left, topH, islandSeed, cells.topAlt);
      const edgeW = Math.max(24, Math.min(34, Number(style.edgeWidth || 42)));
      if (isRectInBounds({ x: left - edgeW * 0.34, y: topY, w: edgeW, h: bodyBaseH + topH }, bounds, 120)) {
        this.drawEnvironmentCell('terrain', profile, cells.left, left - edgeW * 0.34, topY, edgeW, bodyBaseH + topH);
      }
      if (isRectInBounds({ x: right - edgeW * 0.66, y: topY, w: edgeW, h: bodyBaseH + topH }, bounds, 120)) {
        this.drawEnvironmentCell('terrain', profile, cells.right, right - edgeW * 0.66, topY, edgeW, bodyBaseH + topH);
      }
      const detailCell = this.pickEnvironmentCell(cells.detail || cells.cap, islandSeed, 'detail');
      if (detailCell != null && segmentW >= 240) {
        const detailW = 30 + Math.round(seededUnit(islandSeed, 'detail:w') * 14);
        const detailH = 14 + Math.round(seededUnit(islandSeed, 'detail:h') * 8);
        const detailX = x + Math.round(segmentW * (0.26 + seededUnit(islandSeed, 'detail:x') * 0.48));
        const detailY = platform.y - detailH - 2;
        if (isRectInBounds({ x: detailX, y: detailY, w: detailW, h: detailH }, bounds, 80)) {
          this.drawEnvironmentCell('terrain', profile, detailCell, detailX, detailY, detailW, detailH, { alpha: 0.84 });
        }
      }
    }

    drawFieldSolidLaneBodySegment(snapshot, map, platform, segment, segmentIndex, profile, cells, style, seed) {
      if (!segment || !platform) return;
      const bounds = snapshot.bounds || {};
      const platformW = Math.max(1, Number(platform.w || 0));
      const segmentX = clamp(Number(segment.x || 0), 18, Math.max(18, platformW - 96));
      const availableW = Math.max(0, platformW - segmentX - 18);
      const segmentW = clamp(Number(segment.w || 0), 96, Math.max(96, availableW));
      if (availableW < 96 || segmentW < 96) return;
      const segmentSeed = `${seed}:solid-segment:${segmentIndex}`;
      const overhang = Math.max(4, Math.min(10, Number(style.overhang || 8)));
      const bodyBaseH = clamp(Number(segment.depth || style.platformBodyDepth || 24), 16, 38);
      const rawUndersideH = Number(style.undersideHeight == null ? 12 : style.undersideHeight);
      const undersideH = rawUndersideH <= 0 ? 0 : Math.max(8, Math.min(14, rawUndersideH));
      const alpha = style.bodyAlpha == null ? 0.94 : Number(style.bodyAlpha);
      const left = platform.x + segmentX - overhang;
      const right = platform.x + segmentX + segmentW + overhang;
      const bodyY = platform.y + 1;
      const shadowCell = this.pickEnvironmentCell(cells.shadow, segmentSeed, 'shadow');
      if (shadowCell != null && isRectInBounds({ x: left, y: platform.y + 12, w: right - left, h: 14 }, bounds, 120)) {
        this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, shadowCell, left, platform.y + 12, right - left, 14, `${segmentSeed}:shadow`);
      }
      if (isRectInBounds({ x: left, y: bodyY, w: right - left, h: bodyBaseH + undersideH }, bounds, 120)) {
        this.drawTerrainBodyBand(snapshot, profile, cells, left, bodyY, right - left, bodyBaseH, undersideH, segmentSeed, alpha);
      }
    }

    drawFieldPlatformUnderhangSegment(snapshot, map, platform, segment, segmentIndex, profile, cells, style, seed) {
      if (!segment || !platform) return;
      const bounds = snapshot.bounds || {};
      const platformW = Math.max(1, Number(platform.w || 0));
      const segmentX = clamp(Number(segment.x || 0), 18, Math.max(18, platformW - 96));
      const availableW = Math.max(0, platformW - segmentX - 18);
      const segmentW = clamp(Number(segment.w || 0), 96, Math.max(96, availableW));
      if (availableW < 96 || segmentW < 96) return;
      const segmentSeed = `${seed}:underhang:${segmentIndex}`;
      const overhang = Math.max(4, Math.min(10, Number(style.overhang || 8)));
      const left = platform.x + segmentX - overhang;
      const right = platform.x + segmentX + segmentW + overhang;
      const width = Math.max(1, right - left);
      const sourceCells = cells.undersideLong || cells.underside;
      const alpha = Math.min(style.bodyAlpha == null ? 0.86 : Number(style.bodyAlpha), 0.86);
      const maxH = Math.max(12, Math.min(24, Math.round(Number(style.platformBodyDepth || 28) * 0.58)));
      const fragmentCount = Math.max(1, Math.round(width / 150));
      for (let fragmentIndex = 0; fragmentIndex < fragmentCount; fragmentIndex += 1) {
        const ratio = fragmentCount === 1 ? 0.5 : fragmentIndex / Math.max(1, fragmentCount - 1);
        const jitter = (seededUnit(segmentSeed, `x:${fragmentIndex}`) - 0.5) * 30;
        const fragmentW = Math.min(width, 78 + seededUnit(segmentSeed, `w:${fragmentIndex}`) * 58);
        const fragmentX = clamp(left + ratio * width - fragmentW * 0.5 + jitter, left, Math.max(left, right - fragmentW));
        const fragmentH = Math.max(10, Math.round(maxH * (0.72 + seededUnit(segmentSeed, `h:${fragmentIndex}`) * 0.28)));
        const y = platform.y + 2 + Math.round(seededUnit(segmentSeed, `y:${fragmentIndex}`) * 4);
        if (isRectInBounds({ x: fragmentX, y, w: fragmentW, h: fragmentH }, bounds, 120)) {
          this.drawEnvironmentTileStrip(snapshot, 'terrain', profile, sourceCells, fragmentX, y, fragmentW, fragmentH, `${segmentSeed}:fragment:${fragmentIndex}`, null, { alpha, overlap: 0 });
        }
      }
    }

    drawFieldSolidLaneTerrain(snapshot, map, platform, index, visual, profile, cells, style, seed) {
      this.drawFieldPlatformLedge(snapshot, map, platform, index, profile, cells, style, seed);
      return true;
    }

    getRampTerrainCell(platform, index) {
      const descendingRight = Number(platform && platform.y2 || platform && platform.y || 0) > Number(platform && platform.y || 0);
      return (descendingRight ? 1 : 0) + (index % 2 ? 2 : 0);
    }

    drawRampPlatformTerrain(snapshot, map, platform, index, profile, style, seed) {
      if (!isSlopePlatform(platform)) return false;
      const asset = this.getEnvironmentAsset('ramps', profile);
      const texture = asset && asset.path ? this.getTexture(asset.path) : null;
      if (!asset || !texture) return false;
      const leftY = Number(platform.y || 0);
      const rightY = Number(platform.y2 || platform.y || 0);
      const overhang = Math.max(6, Math.min(14, Number(style && style.overhang || 8)));
      const topPad = Math.max(8, Math.min(14, Number(style && style.topHeight || 18) * 0.55));
      const bodyDepth = Math.max(22, Math.min(36, Number(style && style.platformBodyDepth || 28)));
      const drawX = Number(platform.x || 0) - overhang;
      const drawW = Math.max(1, Number(platform.w || 0) + overhang * 2);
      const drawY = Math.min(leftY, rightY) - topPad;
      const drawH = Math.max(36, Math.abs(rightY - leftY) + topPad + bodyDepth);
      const drawn = this.drawEnvironmentCell(
        'ramps',
        profile,
        this.getRampTerrainCell(platform, index),
        drawX,
        drawY,
        drawW,
        drawH,
        { alpha: style && style.bodyAlpha == null ? 1 : Number(style && style.bodyAlpha || 1) }
      );
      if (!drawn) return false;
      this.drawPlatformThemeTrim(this.mapGraphics, map, platform, index);
      return true;
    }

    drawFieldConnectorTerrain(snapshot, map, platform, index, visual, profile, cells, style, seed) {
      if (!visual || !platform) return false;
      if (visual.kind === 'connector' || visual.kind === 'hop' || visual.kind === 'island') {
        this.drawFieldPlatformLedge(snapshot, map, platform, index, profile, cells, style, seed);
        return true;
      }
      return false;
    }

    drawFieldVisualPlatformTerrain(snapshot, map, platform, index, profile, cells, style, seed) {
      const visual = this.getPlatformTerrainVisual(platform, index);
      if (isSlopePlatform(platform) || visual && visual.kind === 'slope') {
        if (!this.drawRampPlatformTerrain(snapshot, map, platform, index, profile, style, seed)) {
          this.drawPlatformFallback(this.mapGraphics, map, platform, index);
        }
        return true;
      }
      if (this.drawFieldConnectorTerrain(snapshot, map, platform, index, visual, profile, cells, style, seed)) return true;
      if (!visual) return false;
      if (visual.kind === 'solidLane' || visual.kind === 'ledge') return this.drawFieldSolidLaneTerrain(snapshot, map, platform, index, visual, profile, cells, style, seed);
      return false;
    }

    drawTiledPlatformTerrain(snapshot, map, platform, index) {
      const profile = this.getEnvironmentProfile(map);
      const asset = this.getEnvironmentAsset('terrain', profile);
      const cells = this.getEnvironmentCells('terrain');
      if (!asset || !asset.path || !this.getTexture(asset.path)) {
        this.drawPlatformFallback(this.mapGraphics, map, platform, index);
        return;
      }
      const seed = `${map.id}:platform:${index}`;
      const style = this.getEnvironmentTerrainStyle(profile);
      if (this.drawFieldVisualPlatformTerrain(snapshot, map, platform, index, profile, cells, style, seed)) return;
      const isGround = index === 0;
      const overhang = isGround ? Number(style.groundOverhang || 0) : Number(style.overhang || 12);
      const topH = isGround ? Number(style.groundTopHeight || 28) : Number(style.topHeight || 24);
      const platformList = snapshot && snapshot.runtime && Array.isArray(snapshot.runtime.platforms) ? snapshot.runtime.platforms : [];
      const bodyBaseH = this.getPlatformTerrainBodyDepth(platformList, platform, index, style, topH, isGround);
      const left = platform.x - overhang;
      const right = platform.x + platform.w + overhang;
      const topY = platform.y - topH + (isGround ? 0 : 2);
      const layerH = Math.max(topH, Math.round(topH + bodyBaseH));
      const bounds = snapshot.bounds || {};
      const terrainOverlap = 10;

      if (isRectInBounds({ x: left, y: topY, w: right - left, h: layerH }, bounds, 120)) {
        this.drawTerrainSurface(snapshot, profile, cells, isGround, left, topY, right - left, layerH, `${seed}:single`, { overlap: terrainOverlap });
      }
    }

    renderMap(snapshot) {
      const graphics = this.mapGraphics;
      const runtime = snapshot.runtime || {};
      const map = snapshot.map || {};
      const bounds = snapshot.bounds || {};
      this.renderTownStructures(snapshot, map, 'rear');
      this.renderFieldCompositionLandmarks(snapshot, map);
      this.renderMapScenery(snapshot, map, 'rear');
      (runtime.platforms || []).forEach((platform, index) => {
        if (!platform || platform.x + platform.w < bounds.left - 80 || platform.x > bounds.right + 80) return;
        this.drawTiledPlatformTerrain(snapshot, map, platform, index);
      });
      this.renderMapScenery(snapshot, map, 'front');
      this.renderTownStructures(snapshot, map, 'front');
      this.renderClimbables(graphics, runtime, map);
      this.renderPortals(graphics, runtime, snapshot);
      this.renderQuestNpcs(graphics, runtime);
      this.renderStations(graphics, runtime);
    }

    getClimbableVisualStyle(map, climbable) {
      const theme = this.getMapThemeId(map);
      const id = String(climbable && climbable.id || '').toLowerCase();
      if (id.includes('stair')) {
        return { kind: 'stair', rail: 0x5c442d, railAlpha: 0.82, rung: 0xf7d28a, rungAlpha: 0.72 };
      }
      if (theme.includes('grass') || theme.includes('thorn') || theme.includes('bramble') || theme.includes('trapper') || theme.includes('beast') || id.includes('vine')) {
        return { kind: 'vine', rail: 0x386b37, railAlpha: 0.82, rung: 0x8bbf65, rungAlpha: 0.66 };
      }
      if (theme.includes('ruins') || theme.includes('gearworks') || theme.includes('quarry') || id.includes('ladder') || id.includes('lift')) {
        return { kind: 'metal', rail: 0xd8b74a, railAlpha: 0.78, rung: 0x7a8592, rungAlpha: 0.82 };
      }
      if (theme.includes('cinder') || theme.includes('ember') || theme.includes('fire') || id.includes('chain')) {
        return { kind: 'chain', rail: 0x372a27, railAlpha: 0.9, rung: 0xff8140, rungAlpha: 0.72 };
      }
      if (theme.includes('frost')) {
        return { kind: 'frost', rail: 0xd7f3ff, railAlpha: 0.78, rung: 0x5ca8e8, rungAlpha: 0.76 };
      }
      if (theme.includes('storm')) {
        return { kind: 'storm', rail: 0x91dbe8, railAlpha: 0.78, rung: 0xffe16a, rungAlpha: 0.72 };
      }
      if (theme.includes('astral') || theme.includes('eclipse') || theme.includes('rift') || theme.includes('rune')) {
        return { kind: 'rune', rail: 0x7bdff2, railAlpha: 0.76, rung: 0xc794ff, rungAlpha: 0.78 };
      }
      return { kind: 'rope', rail: 0x5b3d2d, railAlpha: 0.76, rung: 0xf7d28a, rungAlpha: 0.68 };
    }

    renderClimbables(graphics, runtime, map) {
      (runtime.climbables || []).forEach((climbable) => {
        const style = this.getClimbableVisualStyle(map, climbable);
        const left = climbable.x + climbable.w * 0.28;
        const right = climbable.x + climbable.w * 0.72;
        const top = climbable.y;
        const bottom = climbable.y + climbable.h;
        graphics
          .moveTo(left, top)
          .lineTo(left, bottom)
          .moveTo(right, top)
          .lineTo(right, bottom)
          .stroke({ width: style.kind === 'vine' ? 5 : 4, color: style.rail, alpha: style.railAlpha });
        for (let y = top + 16; y < bottom; y += 28) {
          if (style.kind === 'chain') {
            graphics.ellipse((left + right) / 2, y, climbable.w * 0.34, 7)
              .stroke({ width: 3, color: style.rung, alpha: style.rungAlpha });
            continue;
          }
          if (style.kind === 'rune') {
            graphics
              .moveTo(left - 4, y)
              .lineTo(right + 4, y)
              .moveTo((left + right) / 2, y - 5)
              .lineTo((left + right) / 2, y + 5)
              .stroke({ width: 2, color: style.rung, alpha: style.rungAlpha });
            continue;
          }
          graphics
            .moveTo(left - 3, y)
            .lineTo(right + 3, y + (style.kind === 'vine' ? Math.sin(y * 0.05) * 2 : 0))
            .stroke({ width: 2, color: style.rung, alpha: style.rungAlpha });
        }
      });
    }

    getPortalDirection(portal, runtime) {
      const explicit = String(portal && portal.direction || '').toLowerCase();
      if (explicit === 'left' || explicit === 'west') return -1;
      if (explicit === 'right' || explicit === 'east') return 1;
      if (portal && portal.returnPortal) return -1;
      const worldWidth = Math.max(1, Number(runtime && runtime.worldWidth || 1));
      return Number(portal && portal.x || 0) + Number(portal && portal.w || 0) / 2 < worldWidth * 0.5 ? -1 : 1;
    }

    getPortalLabelRenderState(portal, runtime, snapshot) {
      const camera = snapshot && snapshot.camera || {};
      const cameraLeft = Number(camera.x || 0);
      const cameraRight = cameraLeft + Math.max(1, Number(camera.w || snapshot && snapshot.width || this.width || 1));
      const cx = Number(portal.x || 0) + Number(portal.w || 0) / 2;
      const margin = 170;
      if (cx < cameraLeft - margin || cx > cameraRight + margin) return null;
      const edgePad = portal.shopDoor ? 70 : 62;
      const x = clamp(cx, cameraLeft + edgePad, Math.max(cameraLeft + edgePad, cameraRight - edgePad));
      return {
        x,
        y: Math.max(Number(camera.y || 0) + 18, Number(portal.y || 0) - 13),
        direction: cx < x ? -1 : cx > x ? 1 : this.getPortalDirection(portal, runtime),
        edgeCue: Math.abs(cx - x) > 2
      };
    }

    drawPortalDirectionMarker(graphics, x, y, direction, color, alpha) {
      const sign = direction < 0 ? -1 : 1;
      graphics
        .moveTo(x + sign * 8, y)
        .lineTo(x - sign * 4, y - 6)
        .lineTo(x - sign * 4, y + 6)
        .closePath()
        .fill({ color, alpha });
    }

    renderPortalLabel(graphics, portal, runtime, snapshot, locked, color) {
      const labelState = this.getPortalLabelRenderState(portal, runtime, snapshot);
      if (!labelState) return;
      const vendorType = String(portal.shopVendorType || '').toLowerCase();
      const shopLabel = {
        weapon: 'Weapon',
        armor: 'Armor',
        supply: 'Supply',
        special: 'Special'
      }[vendorType];
      const rawLabel = String(shopLabel || portal.label || portal.destinationName || 'Portal').trim();
      const label = rawLabel.length > 22 ? `${rawLabel.slice(0, 21)}...` : rawLabel;
      const labelY = labelState.y - (vendorType === 'armor' || vendorType === 'special' ? 13 : 0);
      const markerX = labelState.x + labelState.direction * (Math.min(62, Math.max(30, label.length * 3.2)) + 8);
      this.drawPortalDirectionMarker(
        graphics,
        markerX,
        labelY + 5,
        labelState.direction,
        locked ? 0x8a97a5 : color,
        labelState.edgeCue ? 0.96 : 0.72
      );
      this.drawText('damageText', label, labelState.x, labelY, {
        fontSize: portal.shopDoor ? 11 : 10,
        fontWeight: '900',
        fill: locked ? '#d7dde3' : '#ffffff',
        stroke: 'rgba(9,31,59,0.96)',
        strokeWidth: 4,
        alpha: locked ? 0.78 : 0.98
      });
    }

    renderShopDoorPortal(graphics, portal, runtime, snapshot, locked, pulse) {
      const palettes = {
        weapon: [0x4e6174, 0xf25f4c, 0xffd166],
        armor: [0x5d6b7a, 0x2f9d8f, 0xd8e5ec],
        supply: [0x6c7354, 0x74d680, 0xfff3b0],
        special: [0x66549a, 0x7bdff2, 0xffe16a]
      };
      const palette = palettes[String(portal.shopVendorType || '').toLowerCase()] || [0x5e7d9f, 0xffd166, 0xd8e5ec];
      const x = Number(portal.x || 0);
      const y = Number(portal.y || 0);
      const w = Number(portal.w || 94);
      const h = Number(portal.h || 118);
      const alpha = locked ? 0.62 : 1;
      graphics.ellipse(x + w / 2, y + h + 2, w * 0.58, 8).fill({ color: 0x091f3b, alpha: 0.28 * alpha });
      const facadeW = Math.max(w, Number(portal.facadeWidth || 154));
      const facadeH = Math.max(h, Number(portal.facadeHeight || 142));
      const facadeDrawn = this.drawEnvironmentStructureCell(
        portal.facadeCell || 'marketAwning',
        x + w / 2 - facadeW / 2,
        y + h - facadeH,
        facadeW,
        facadeH,
        { alpha }
      );
      if (facadeDrawn) {
        this.renderPortalLabel(graphics, portal, runtime, snapshot, locked, palette[2]);
        return;
      }
      graphics.rect(x + 6, y + 22, w - 12, h - 20).fill({ color: palette[0], alpha });
      graphics
        .moveTo(x + 2, y + 28)
        .lineTo(x + w / 2, y)
        .lineTo(x + w - 2, y + 28)
        .closePath()
        .fill({ color: palette[1], alpha });
      graphics.rect(x + w * 0.29, y + h * 0.38, w * 0.42, h * 0.62)
        .fill({ color: locked ? 0x576575 : 0x17283a, alpha });
      graphics.rect(x + w * 0.34, y + h * 0.45, w * 0.32, 12).fill({ color: palette[2], alpha });
      graphics.circle(x + w * 0.62, y + h * 0.7, 3).fill({ color: palette[2], alpha });
      if (!locked) {
        graphics.rect(x + w * 0.29, y + h * 0.38, w * 0.42, h * 0.62)
          .stroke({ width: 2, color: palette[2], alpha: 0.55 + pulse * 0.3 });
      }
      this.renderPortalLabel(graphics, portal, runtime, snapshot, locked, palette[2]);
    }

    renderPortals(graphics, runtime, snapshot) {
      const now = Number(snapshot.nowSec || 0);
      (runtime.portals || []).forEach((portal) => {
        const cx = portal.x + portal.w / 2;
        const cy = portal.y + portal.h / 2;
        const locked = !!portal.locked;
        const color = locked ? 0x8a97a5 : portal.bossPortal ? 0xb073ff : 0x36c5ff;
        const pulse = 0.5 + Math.sin(now * 3.4 + portal.x * 0.01) * 0.5;
        if (portal.shopDoor) {
          this.renderShopDoorPortal(graphics, portal, runtime, snapshot, locked, pulse);
          return;
        }
        graphics.ellipse(cx, cy, portal.w * 0.42, portal.h * 0.48)
          .stroke({ width: locked ? 3 : 4, color, alpha: locked ? 0.72 : 0.9 });
        graphics.ellipse(cx, cy, portal.w * (0.18 + pulse * 0.05), portal.h * 0.28)
          .stroke({ width: 2, color, alpha: locked ? 0.46 : 0.64 });
        graphics.rect(portal.x + 4, portal.y + portal.h - 8, portal.w - 8, 5)
          .fill({ color: 0x091f3b, alpha: locked ? 0.62 : 0.76 });
        this.renderPortalLabel(graphics, portal, runtime, snapshot, locked, color);
      });
    }

    renderQuestNpcs(graphics, runtime) {
      (runtime.questNpcs || []).forEach((npc) => {
        const color = colorToNumber(npc.color, 0x4f7f63);
        const accent = colorToNumber(npc.accent, 0xffd166);
        const cx = npc.x + npc.w / 2;
        graphics.ellipse(cx, npc.y + npc.h - 2, npc.w * 0.62, 5).fill({ color: 0x091f3b, alpha: 0.34 });
        const npcTexture = this.getTexture(npc.asset);
        if (npcTexture && this.drawActorTexture(npcTexture, npc, {
          key: `quest-npc:${npc.asset}`,
          kind: 'party'
        })) {
          return;
        }
        graphics.rect(npc.x + 6, npc.y + 22, npc.w - 12, npc.h - 28).fill({ color, alpha: 1 });
        graphics.rect(npc.x + 8, npc.y + 30, npc.w - 16, 8).fill({ color: accent, alpha: 1 });
        graphics.circle(cx, npc.y + 15, npc.w * 0.33).fill({ color: 0xf3d4b2, alpha: 1 });
        graphics.rect(npc.x + 7, npc.y + npc.h - 18, 9, 18).fill({ color: 0x263547, alpha: 1 });
        graphics.rect(npc.x + npc.w - 16, npc.y + npc.h - 18, 9, 18).fill({ color: 0x263547, alpha: 1 });
      });
    }

    renderStations(graphics, runtime) {
      (runtime.stations || []).forEach((station) => {
        const stationTexture = this.getTexture(station.asset);
        if (stationTexture) {
          const drawWidth = station.id === 'upgrade' ? 112 : 124;
          const drawHeight = 96;
          this.drawTexture('map', stationTexture, station.x - 18, station.y - 48, drawWidth, drawHeight, {
            anchorX: 0,
            anchorY: 0
          });
          return;
        }
        const color = station.id === 'upgrade' ? 0xb76c40 : station.id === 'class' ? 0x4b7f9e : station.id === 'storage' ? 0x2f9d8f : station.id === 'slots' ? 0x8a6bcb : 0xd7a84a;
        graphics.rect(station.x, station.y, station.w, station.h).fill({ color, alpha: 1 });
        graphics.rect(station.x + 10, station.y + 10, station.w - 20, 12).fill({ color: 0xfff8dc, alpha: 1 });
      });
    }

    renderRuneFieldTimerBar(effect, x, y, radius, tint, alpha, lifeRatio, now) {
      if (!effect || Number(effect.ttl || 0) <= 0) return;
      const duration = Math.max(0.01, Number(effect.duration || effect.baseDuration || effect.ttl || 1));
      const fillRatio = clamp(lifeRatio, 0, 1);
      const barW = clamp(radius * 1.18, 72, 164);
      const barH = 6;
      const barX = x - barW / 2;
      const barY = y - Math.max(28, radius * 0.26);
      this.drawSolidRect('vfx', barX, barY, barW, barH, { tint: 0x091f3b, alpha: Math.max(0.32, alpha * 0.68) });
      this.drawSolidRect('vfx', barX, barY, barW * fillRatio, barH, { tint, alpha: Math.max(0.52, alpha * 0.9) });
      const pulseAge = Number(now || 0) - Number(effect.lastExtendedAt || 0);
      const lastExtension = Math.max(0, Number(effect.lastExtensionAmount || 0));
      if (lastExtension > 0 && pulseAge >= 0 && pulseAge < RUNE_FIELD_REFILL_PULSE_SECONDS) {
        const pulseAlpha = 0.78 * (1 - pulseAge / RUNE_FIELD_REFILL_PULSE_SECONDS);
        const pulseW = Math.max(5, barW * clamp(lastExtension / duration, 0.035, 0.45));
        const pulseX = barX + Math.max(0, barW * fillRatio - pulseW);
        this.drawSolidRect('vfx', pulseX, barY, Math.min(pulseW, barX + barW - pulseX), barH, { tint: 0xffffff, alpha: pulseAlpha });
      }
      this.drawRectOutline('vfx', barX - 1, barY - 1, barW + 2, barH + 2, 1, { tint: 0xffffff, alpha: Math.max(0.24, alpha * 0.38) });
    }

    renderRuneFieldGroundVisual(effect, x, y, radius, tint, alpha, lifeRatio, now, simplified, options) {
      if (!effect || Number(effect.ttl || 0) <= 0) return;
      const settings = options || {};
      const drawAura = settings.aura !== false;
      const drawPulse = settings.pulse !== false;
      const fieldRadius = Math.max(1, Number(radius || effect.r || 80));
      const verticalRadius = Math.max(22, Number(effect.verticalTolerance || fieldRadius * 0.18));
      const fieldW = fieldRadius * 2;
      const fieldH = verticalRadius * 2;
      if (drawAura) {
        const auraAlpha = Math.max(0.08, alpha * (0.28 + lifeRatio * 0.34));
        this.drawShape('vfx', 'glow', x, y, fieldW, fieldH, {
          tint,
          alpha: simplified ? auraAlpha * 0.55 : auraAlpha
        });
        this.drawShape('vfx', 'ring', x, y, fieldW * 0.98, fieldH * 0.98, {
          tint,
          alpha: Math.max(0.1, alpha * (0.18 + lifeRatio * 0.5))
        });
      }
      if (!drawPulse) return;
      const progress = clamp(Number(effect.pulseProgress || 0), 0, 1);
      if (progress > 0.02) {
        const pulseAlpha = Math.sin(progress * Math.PI);
        this.drawShape('vfx', 'ring', x, y, Math.max(20, fieldW * progress), Math.max(16, fieldH * progress), {
          tint: 0xffffff,
          alpha: (simplified ? 0.55 : 1) * Math.max(0.1, alpha * (0.22 + pulseAlpha * 0.58))
        });
      }
      const pulseAge = Number(now || 0) - Number(effect.lastPulseAt || 0);
      if (Number(effect.lastPulseHits || 0) > 0 && pulseAge >= 0 && pulseAge < 0.16) {
        this.drawShape('vfx', 'ring', x, y, fieldW, fieldH, {
          tint: 0xffffff,
          alpha: Math.max(0.1, alpha * 0.72 * (1 - pulseAge / 0.16))
        });
      }
    }

    renderWorldEffects(snapshot) {
      const now = Number(snapshot.nowSec || 0);
      const quality = snapshot.visualQuality || {};
      const simplified = !!quality.reduceEffects || quality.level === 'reduced';
      const effects = snapshot.worldEffects || [];
      for (let effectIndex = 0; effectIndex < effects.length; effectIndex += 1) {
        const effect = effects[effectIndex];
        if (!effect) continue;
        const type = String(effect.type || '');
        const color = colorToNumber(effect.color, type === 'shockBurst' ? 0x7bdff2 : type === 'field' ? 0x68d58d : 0xffd166);
        const ttl = Math.max(0, Number(effect.ttl || effect.life || 0));
        const alpha = clamp(Number(effect.alpha == null ? 0.7 : effect.alpha) * (ttl ? clamp(ttl, 0.18, 1) : 1), 0, 1);
        const x = Number(effect.x || 0);
        const y = Number(effect.y || 0);
        const isRuneFieldEffect = type === 'field' && effect.runeField;
        const runeFieldRadius = isRuneFieldEffect ? Math.max(8, Number(effect.r || effect.radius || 32)) : 0;
        const runeFieldDuration = isRuneFieldEffect ? Math.max(0.01, Number(effect.duration || effect.baseDuration || ttl || 1)) : 1;
        const runeFieldLifeRatio = isRuneFieldEffect ? clamp(ttl / runeFieldDuration, 0, 1) : 0;
        if (isRuneFieldEffect) {
          this.renderRuneFieldGroundVisual(effect, x, y, runeFieldRadius, color, alpha, runeFieldLifeRatio, now, simplified, { pulse: false });
        }
        if (effect.animationFrame && type !== 'chainLine') {
          const texture = this.getFrameTexture(effect.animationFrame);
          if (texture) {
            const radius = Math.max(18, Number(effect.r || effect.radius || 42));
            const size = type === 'skillArea' || type === 'field'
              ? Math.max(124, radius * 2.15)
              : type === 'slash' || type === 'impact'
                ? Math.max(96, radius * 2.2)
                : Math.max(86, radius * 1.85);
            this.drawTexture('vfx', texture, x, y, size, size, {
              alpha: Math.max(0.18, alpha),
              blendMode: 'add',
              flipX: Number(effect.facing || 1) < 0
            });
            if (type === 'field' && effect.runeField) {
              const duration = Math.max(0.01, Number(effect.duration || effect.baseDuration || ttl || 1));
              const lifeRatio = clamp(ttl / duration, 0, 1);
              this.renderRuneFieldGroundVisual(effect, x, y, radius, color, alpha, lifeRatio, now, simplified, { aura: false });
              this.renderRuneFieldTimerBar(effect, x, y, radius, color, alpha, lifeRatio, now);
            }
            continue;
          }
        }
        if (type === 'chainLine') {
          this.drawLine('vfx', x, y, Number(effect.x2 == null ? x : effect.x2), Number(effect.y2 == null ? y : effect.y2), Math.max(2, Number(effect.width || 4)), {
            tint: color,
            alpha: simplified ? 0.34 : 0.55
          });
          continue;
        }
        if (type === 'telegraph') {
          const w = Math.max(1, Number(effect.w || 0));
          const h = Math.max(1, Number(effect.h || 0));
          this.drawSolidRect('vfx', x, y, w, h, { tint: 0xff6b35, alpha: simplified ? 0.06 : 0.08 });
          this.drawRectOutline('vfx', x, y, w, h, 2, { tint: 0xffc857, alpha: simplified ? 0.28 : 0.42 });
          continue;
        }
        if (type === 'recoveryPulse') {
          const duration = Math.max(0.01, Number(effect.duration || 0.5));
          const progress = clamp(1 - ttl / duration, 0, 1);
          const radius = Math.max(14, Number(effect.r || 42));
          const ringW = radius * (2.05 + progress * 1.1);
          const ringH = Math.max(16, radius * 0.72 * (1 + progress * 0.28));
          const accent = colorToNumber(effect.accentColor, 0xffffff);
          const drawAlpha = alpha * (1 - progress * 0.25);
          this.drawShape('vfx', 'glow', x, y, ringW * 1.1, ringH * 1.2, { tint: color, alpha: simplified ? drawAlpha * 0.12 : drawAlpha * 0.2 });
          this.drawShape('vfx', 'ring', x, y, ringW * 0.82, ringH * 0.9, { tint: color, alpha: Math.max(0.14, drawAlpha * 0.72) });
          if (!simplified) {
            for (let i = -1; i <= 1; i += 1) {
              const gx = x + i * radius * 0.22;
              this.drawLine('vfx', gx, y - ringH * 0.34 - progress * 8, gx + i * 4, y - ringH * 0.62 - progress * 13, 1.6, {
                tint: accent,
                alpha: drawAlpha * 0.68
              });
            }
          }
          continue;
        }
        if (type === 'field') {
          const radius = Math.max(8, Number(effect.r || effect.radius || 32));
          const duration = Math.max(0.01, Number(effect.duration || effect.baseDuration || ttl || 1));
          const lifeRatio = clamp(ttl / duration, 0, 1);
          if (effect.runeField) {
            this.renderRuneFieldGroundVisual(effect, x, y, radius, color, alpha, lifeRatio, now, simplified, { aura: false });
            this.renderRuneFieldTimerBar(effect, x, y, radius, color, alpha, lifeRatio, now);
            continue;
          }
          const fieldW = radius * 2;
          const fieldH = Math.max(30, radius * 0.34);
          const fieldAlpha = Math.max(0.08, alpha * (0.34 + lifeRatio * 0.44));
          this.drawShape('vfx', 'glow', x, y, fieldW, fieldH, { tint: color, alpha: simplified ? fieldAlpha * 0.55 : fieldAlpha });
          this.drawShape('vfx', 'ring', x, y, fieldW * 0.94, fieldH * 1.16, { tint: color, alpha: Math.max(0.12, alpha * (0.18 + lifeRatio * 0.68)) });
          continue;
        }
        if (type === 'slash' || type === 'arrowRelease') {
          const facing = Number(effect.facing || 1) >= 0 ? 1 : -1;
          this.drawShape('vfx', 'slash', x + facing * 34, y - 10, 92, 46, {
            tint: color,
            alpha: Math.max(0.22, simplified ? alpha * 0.72 : alpha),
            flipX: facing < 0
          });
          continue;
        }
        if (type === 'cast') {
          const radius = Math.max(12, Number(effect.r || 28));
          this.drawShape('vfx', 'ring', x, y, radius * 2, radius * 2, { tint: color, alpha });
          continue;
        }
        if (type === 'lootPickup') {
          const duration = Math.max(0.01, Number(effect.duration || 0.26));
          const progress = clamp(1 - Number(effect.ttl || 0) / duration, 0, 1);
          const ease = 1 - Math.pow(1 - progress, 3);
          const targetX = Number(effect.targetX || x);
          const targetY = Number(effect.targetY || y);
          const drawX = x + (targetX - x) * ease;
          const drawY = y + (targetY - y) * ease - Math.sin(progress * Math.PI) * 22;
          const size = Math.max(12, Number(effect.size || 32) * (1 - progress * 0.28));
          const drawAlpha = clamp(1 - progress * 0.86, 0, 1) * alpha;
          const radius = Math.max(5, size * 0.56);
          this.drawShape('vfx', 'glow', drawX, drawY, radius * 2.8, radius * 2.8, { tint: color, alpha: 0.24 * drawAlpha });
          if (effect.showTierAura) this.drawShape('vfx', 'ring', drawX, drawY, radius * 1.18, radius * 1.18, { tint: color, alpha: Math.max(0.16, drawAlpha * 0.74) });
          const texture = this.getAssetFrameTexture(effect.itemAsset);
          if (texture) this.drawTexture('vfx', texture, drawX, drawY, size, size, { alpha: drawAlpha });
          else this.drawShape('vfx', 'circle', drawX, drawY, radius * 0.92, radius * 0.92, { tint: color, alpha: drawAlpha });
          continue;
        }
        const radius = Math.max(8, Number(effect.r || effect.radius || 32));
        this.drawShape('vfx', 'glow', x, y, radius * 2.2, radius * 2.2, { tint: color, alpha: simplified ? alpha * 0.08 : alpha * 0.12 });
        if (!simplified || type === 'skillImpact' || type === 'shockBurst') {
          this.drawShape('vfx', 'ring', x, y, radius * 1.44, radius * 1.44, { tint: color, alpha: Math.max(0.2, alpha * 0.78) });
        }
      }
    }

    renderProjectiles(snapshot) {
      const bounds = snapshot.bounds || {};
      (snapshot.projectiles || []).forEach((projectile) => {
        if (!projectile || !isPointInBounds(projectile, bounds, 100)) return;
        const color = colorToNumber(projectile.color, projectile.owner === 'enemy' ? 0xff6b35 : 0x2f7dd6);
        const x = Number(projectile.x || 0);
        const y = Number(projectile.y || 0);
        const vx = Number(projectile.vx || 0);
        const vy = Number(projectile.vy || 0);
        const radius = Math.max(4, Number(projectile.r || projectile.radius || 8));
        const centerX = x + Number(projectile.w || 0) / 2;
        const centerY = y + Number(projectile.h || 0) / 2;
        const horizontalProjectileFx = shouldRenderProjectileFxHorizontally(projectile);
        const trailVx = horizontalProjectileFx ? (vx || Math.sign(Number(projectile.facing || 1)) || 1) : vx;
        const trailVy = horizontalProjectileFx ? 0 : vy;
        this.drawLine('vfx', centerX, centerY, centerX - trailVx * 0.035, centerY - trailVy * 0.035, Math.max(2, radius * 0.54), {
          tint: color,
          alpha: 0.36
        });
        if (projectile.animationFrame) {
          const texture = this.getFrameTexture(projectile.animationFrame);
          if (texture) {
            const explicitSize = Number(projectile.projectileVisualSize || projectile.visualSize || 0);
            const size = explicitSize > 0
              ? explicitSize
              : Math.max(
                projectile.owner === 'enemy' ? 88 : 92,
                Number(projectile.r || projectile.radius || 0) * 3.2,
                Math.max(Number(projectile.w || 0), Number(projectile.h || 0)) * 4.2
            );
            this.drawTexture('vfx', texture, centerX, centerY, size, size, {
              alpha: 0.96,
              rotation: horizontalProjectileFx ? (vx < 0 ? Math.PI : 0) : Math.atan2(vy, vx || 1),
              blendMode: 'add'
            });
            return;
          }
        }
        this.drawShape('vfx', 'circle', centerX, centerY, radius * 2, radius * 2, { tint: color, alpha: 0.86 });
        this.drawShape('vfx', 'circle', centerX, centerY, radius * 0.92, radius * 0.92, { tint: 0xffffff, alpha: 0.42 });
      });
    }

    renderLoot(snapshot) {
      const graphics = this.entityGraphics;
      const bounds = snapshot.bounds || {};
      const now = Number(snapshot.nowSec || 0);
      (snapshot.lootDrops || []).forEach((drop) => {
        if (!drop || !drop.box || !isRectInBounds(drop.box, bounds, 80)) return;
        const box = drop.box;
        const centerX = box.x + box.w / 2;
        const centerY = box.y + box.h / 2;
        const color = colorToNumber(drop.rarityColor, 0xd8e5ec);
        const alpha = drop.otherPlayerDrop ? 0.52 : 1;
        const shadowY = Number(drop.landY || drop.y || centerY) + 14;
        graphics.ellipse(centerX, shadowY, drop.airborne ? box.w * 0.28 : box.w * 0.42, drop.airborne ? 3 : 4)
          .fill({ color: 0x091f3b, alpha: drop.airborne ? 0.28 * alpha : 0.58 * alpha });
        if (drop.showTierAura) {
          const pulse = 0.5 + Math.sin(now * 4.2 + Number(drop.seed || 0)) * 0.5;
          graphics.circle(centerX, centerY, box.w * 0.45 + pulse * 2)
            .stroke({ width: Math.max(1.5, Number(drop.rarityRing || 1.5)), color, alpha: Math.min(1, Number(drop.rarityAlpha || 0.5) + pulse * Number(drop.rarityPulse || 0)) * alpha });
          graphics.circle(centerX, centerY, box.w * 0.56 + pulse * 2.5)
            .stroke({ width: 1, color, alpha: 0.24 * alpha });
        }
        const texture = this.getAssetFrameTexture(drop.itemAsset);
        if (texture) {
          this.drawTexture('entities', texture, centerX, centerY, box.w - 8, box.h - 8, { alpha });
          return;
        }
        graphics.roundRect(box.x + 8, box.y + 8, box.w - 16, box.h - 16, 5)
          .fill({ color: drop.showTierAura ? color : 0xeef6ff, alpha: 0.88 * alpha })
          .stroke({ width: 1, color: 0x102033, alpha: 0.18 * alpha });
      });
    }

    renderEnemies(snapshot) {
      const bounds = snapshot.bounds || {};
      const now = Number(snapshot.nowSec || 0);
      const quality = snapshot.visualQuality || {};
      const hideFarHpBars = !!quality.hideFarHpBars;
      const camera = snapshot.camera || {};
      const player = snapshot.player || {};
      (snapshot.enemies || []).forEach((enemy) => {
        if (!enemy || !enemy.renderBox || !isRectInBounds(enemy.renderBox, bounds, 120)) return;
        const box = enemy.renderBox;
        const alpha = enemy.hp <= 0 ? 0.86 : enemy.telegraph > 0 ? 0.78 : 1;
        if (!this.renderActorSprite(enemy, box, alpha)) {
          this.frameStats.actorFallbacks += 1;
          const color = colorToNumber(enemy.color, 0x7fbe5d);
          if (enemy.behavior === 'flyer') this.drawShape('entities', 'circle', box.x + box.w / 2, box.y + box.h / 2, box.w, box.h, { tint: color, alpha });
          else this.drawSolidRect('entities', box.x, box.y, box.w, box.h, { tint: color, alpha });
        }
        if (enemy.marked > 0) {
          this.drawShape('damage', 'ring', enemy.x + enemy.w / 2, enemy.y - 8, 20, 20, { tint: 0xffe16a, alpha: 0.9 });
        }
        if (enemy.burning > 0) {
          this.drawShape('damage', 'glow', enemy.x + enemy.w / 2, enemy.y + enemy.h * 0.2, 26, 26, { tint: 0xff6b35, alpha: 0.55 });
        }
        this.renderQuestMarker(enemy, snapshot.questGuidance, now);
        if (Number(enemy.hpBarUntil || 0) > now || enemy.hp < enemy.maxHp) {
          const dx = enemy.x + enemy.w / 2 - Number(player.x || camera.x || 0);
          const farFromPlayer = Math.abs(dx) > Math.max(460, Number(camera.w || 0) * 0.45);
          if (hideFarHpBars && farFromPlayer && !enemy.questTarget && enemy.behavior !== 'boss') return;
          const ratio = clamp(Number(enemy.hp || 0) / Math.max(1, Number(enemy.maxHp || 1)), 0, 1);
          this.drawSolidRect('damage', enemy.x, enemy.y - 12, enemy.w, 5, { tint: 0x091f3b, alpha: 0.45 });
          this.drawSolidRect('damage', enemy.x, enemy.y - 12, enemy.w * ratio, 5, { tint: 0xff6b35, alpha: 0.86 });
        }
      });
    }

    renderQuestMarker(enemy, guidance, now) {
      const data = guidance || {};
      const targets = Array.isArray(data.targetEnemyIds) ? data.targetEnemyIds : [];
      if (!data.active || !enemy || enemy.hp <= 0 || data.recommendedMapId && data.recommendedMapId !== enemy.mapId) return;
      if (!targets.includes(enemy.id)) return;
      const pulse = (Math.sin(now * 5 + enemy.x * 0.015) + 1) / 2;
      const cx = enemy.x + enemy.w / 2;
      const cy = enemy.y - 24 - pulse * 4;
      const size = enemy.behavior === 'boss' ? 16 : 13;
      this.drawShape('damage', 'diamond', cx, cy, size * 2.1, size * 2.3, { tint: 0x102033, alpha: 0.18 });
      this.drawShape('damage', 'diamond', cx, cy, size * 1.6, size * 1.8, { tint: 0xffd166, alpha: 0.92 });
    }

    renderParty(snapshot) {
      (snapshot.partyMembers || []).forEach((member) => {
        if (!member) return;
        const box = { x: member.x, y: member.y, w: member.w, h: member.h };
        this.renderActorAtlasLayers(member, box, 1, true);
        if (!this.renderActorComposite(member, box, 1) && !this.renderActorSprite(member, box, 1) && !this.renderActorRig(member, box, 1)) {
          this.frameStats.actorFallbacks += 1;
          this.drawSolidRect('entities', member.x + member.w * 0.1, member.y + member.h * 0.24, member.w * 0.8, member.h * 0.54, {
            tint: colorToNumber(member.classColor, 0x68d58d),
            alpha: 1
          });
          this.drawShape('entities', 'circle', member.x + member.w / 2, member.y + member.h * 0.16, member.w * 0.56, member.w * 0.56, {
            tint: 0xf4d7b8,
            alpha: 1
          });
        }
        this.renderActorAtlasLayers(member, box, 1, false);
        const ratio = clamp(Number(member.hp || 0) / Math.max(1, Number(member.maxHp || 1)), 0, 1);
        this.drawSolidRect('damage', member.x + 3, member.y - 11, member.w - 6, 4, { tint: 0x091f3b, alpha: 0.55 });
        this.drawSolidRect('damage', member.x + 3, member.y - 11, (member.w - 6) * ratio, 4, { tint: 0x68d58d, alpha: 1 });
      });
    }

    renderPet(snapshot) {
      const pet = snapshot.pet;
      if (!pet || !pet.visible) return;
      const x = Number(pet.x || 0);
      const y = Number(pet.y || 0);
      const facing = Number(pet.facing || 1) >= 0 ? 1 : -1;
      if (pet.animationFrame) {
        const texture = this.getFrameTexture(pet.animationFrame);
        if (texture) {
          this.drawShape('entities', 'glow', x, y + 5, 36, 9, { tint: 0x091f3b, alpha: 0.22 });
          this.drawTexture('entities', texture, x, y - 36, 88, 88, {
            anchorX: 0.5,
            anchorY: 0.5,
            flipX: facing < 0
          });
          return;
        }
      }
      const bob = Math.sin(Number(snapshot.nowSec || 0) * 4) * 1.1;
      this.drawShape('entities', 'glow', x, y + 5, 36, 9, { tint: 0x091f3b, alpha: 0.22 });
      this.drawSolidRect('entities', x - 15 * facing, y - 33 + bob, 29 * facing, 25, { tint: 0xd59b54, alpha: 1 });
      this.drawSolidRect('entities', x - 10 * facing, y - 28 + bob, 16 * facing, 6, { tint: 0xf0c36a, alpha: 1 });
      this.drawShape('entities', 'circle', x + 12 * facing, y - 43 + bob, 22, 22, { tint: 0xd59b54, alpha: 1 });
      this.drawShape('entities', 'circle', x + 15 * facing, y - 45 + bob, 3.6, 3.6, { tint: 0x102033, alpha: 1 });
      if (pet.mode === 'loot') {
        this.drawSolidRect('entities', x + 19 * facing, y - 56 + bob, 4 * facing, 4, { tint: 0xffe16a, alpha: 0.78 });
      }
    }

    renderPlayer(snapshot) {
      const player = snapshot.player;
      if (!player || !player.visible) return;
      const box = { x: player.x, y: player.y, w: player.w, h: player.h };
      this.renderActorAtlasLayers(player, box, 1, true);
      let playerDrawn = this.renderActorComposite(player, box, 1);
      if (!playerDrawn) playerDrawn = this.renderActorSprite(player, box, 1);
      if (!playerDrawn) playerDrawn = this.renderActorRig(player, box, 1);
      if (!playerDrawn) {
        this.frameStats.actorFallbacks += 1;
        this.drawSolidRect('entities', player.x + player.w * 0.1, player.y + player.h * 0.26, player.w * 0.8, player.h * 0.52, {
          tint: colorToNumber(player.classColor, 0x2f7dd6),
          alpha: 1
        });
        this.drawShape('entities', 'circle', player.x + player.w / 2, player.y + player.h * 0.16, player.w * 0.56, player.w * 0.56, {
          tint: 0xf4d7b8,
          alpha: 1
        });
        this.drawSolidRect('entities', player.x + player.w * 0.08, player.y + player.h * 0.74, player.w * 0.3, player.h * 0.24, { tint: 0x263547, alpha: 1 });
        this.drawSolidRect('entities', player.x + player.w * 0.62, player.y + player.h * 0.74, player.w * 0.3, player.h * 0.24, { tint: 0x263547, alpha: 1 });
      }
      this.renderActorAtlasLayers(player, box, 1, false);
      if (player.shield > 0) {
        this.drawShape('damage', 'ring', player.x + player.w / 2, player.y + player.h / 2, 60, 88, { tint: 0x68a9ff, alpha: 0.65 });
      }
    }

    renderActorAtlasLayers(actor, box, alpha, behindActor) {
      if (!actor || !box) return 0;
      const layers = Array.isArray(actor.equipmentLayers) ? actor.equipmentLayers : [];
      const registration = PLAYER_SPRITE_REGISTRATION;
      const authoredBodyHeight = Math.max(1, Number(registration.authoredBodyHeight || 143));
      const bodyScale = Math.max(0.01, Number(box.h || 0) / authoredBodyHeight);
      const originX = Number.isFinite(Number(registration.originX)) ? Number(registration.originX) : 80;
      const groundY = Number.isFinite(Number(registration.groundY)) ? Number(registration.groundY) : 154;
      const facing = Number(actor.facing || 1) < 0 ? -1 : 1;
      let drawn = 0;
      layers.forEach((layer) => {
        const part = layer && layer.atlasPart;
        if (!part || !part.frame || !part.socket || !part.pivot) return;
        if ((Number(layer.order || 0) < 0) !== !!behindActor) return;
        const frame = part.frame;
        const texture = this.getFrameTexture(frame);
        if (texture) {
          const frameWidth = Math.max(1, Number(frame.frameWidth || texture.width || 128));
          const frameHeight = Math.max(1, Number(frame.frameHeight || texture.height || frameWidth));
          const scaleX = Math.max(0.05, Number(part.scaleX || 1));
          const scaleY = Math.max(0.05, Number(part.scaleY || 1));
          const worldX = Number(box.x || 0) + Number(box.w || 0) / 2 + facing * (Number(part.socket.x || 0) - originX) * bodyScale;
          const worldY = Number(box.y || 0) + Number(box.h || 0) + (Number(part.socket.y || 0) - groundY) * bodyScale;
          if (this.drawTexture('entities', texture, worldX, worldY, frameWidth * bodyScale * scaleX, frameHeight * bodyScale * scaleY, {
            alpha,
            anchorX: Number(part.pivot.x || 0) / frameWidth,
            anchorY: Number(part.pivot.y || 0) / frameHeight,
            flipX: facing < 0
          })) {
            drawn += 1;
          }
          return;
        }
        const fallbackTexture = this.getFrameTexture(layer.fallbackFrame);
        if (fallbackTexture && this.drawActorTexture(fallbackTexture, box, {
          alpha,
          flipX: facing < 0,
          kind: actor.kind || '',
          registration
        })) {
          drawn += 1;
        }
      });
      return drawn;
    }

    getActorCompositeFrames(actor) {
      if (!actor || !actor.animationFrame) return [];
      const equipmentLayers = Array.isArray(actor.equipmentLayers) ? actor.equipmentLayers : [];
      let hasEquipmentFrame = false;
      let needsSort = false;
      let previousOrder = -Infinity;
      for (let index = 0; index < equipmentLayers.length; index += 1) {
        const layer = equipmentLayers[index];
        if (!layer || !layer.frame) continue;
        const order = Number(layer.order || 0);
        if (order < previousOrder) needsSort = true;
        previousOrder = order;
        hasEquipmentFrame = true;
      }
      if (!hasEquipmentFrame) return [];
      const layers = needsSort
        ? equipmentLayers.filter((layer) => layer && layer.frame).sort((a, b) => Number(a.order || 0) - Number(b.order || 0))
        : equipmentLayers;
      const frames = [];
      let insertedBaseFrame = false;
      for (let index = 0; index < layers.length; index += 1) {
        const layer = layers[index];
        if (!layer || !layer.frame) continue;
        if (!insertedBaseFrame && Number(layer.order || 0) >= 0) {
          frames.push(actor.animationFrame);
          insertedBaseFrame = true;
        }
        frames.push(layer.frame);
      }
      if (!insertedBaseFrame) frames.push(actor.animationFrame);
      return frames;
    }

    getCompositeFrameTextureKey(frames) {
      let compositeKey = '';
      for (let index = 0; index < frames.length; index += 1) {
        const frameKey = this.getFrameTextureKey(frames[index]);
        if (!frameKey) continue;
        compositeKey += compositeKey ? `|${frameKey}` : frameKey;
      }
      return compositeKey;
    }

    renderActorComposite(actor, box, alpha) {
      const frames = this.getActorCompositeFrames(actor);
      if (!frames.length) return false;
      const compositeKey = this.getCompositeFrameTextureKey(frames);
      const compositeTexture = this.getCompositeFrameTexture(frames, compositeKey);
      if (!compositeTexture) return false;
      const kind = String(actor.kind || '');
      return this.drawActorTexture(compositeTexture, box, {
        alpha,
        flipX: Number(actor.facing || 1) < 0,
        key: `composite:${compositeKey}`,
        kind,
        registration: kind === 'player' || kind === 'party' ? PLAYER_SPRITE_REGISTRATION : null
      });
    }

    renderCriticalDamageSplatBurst(effect, metrics) {
      const progress = clamp(Number(metrics && metrics.progress || 0), 0, 1);
      const alpha = clamp(Number(metrics && metrics.alpha || 0), 0, 1);
      if (alpha <= 0) return;
      const x = Number(metrics && metrics.x || 0);
      const y = Number(metrics && metrics.y || 0);
      const glowW = Math.max(28, Number(metrics && metrics.glowW || 34));
      const glowH = Math.max(14, Number(metrics && metrics.glowH || 24));
      const tint = Number(metrics && metrics.tint || 0xfff27a);
      const burstTint = colorToNumber(effect && effect.burstColor, 0xff5d5d);
      const ringTint = colorToNumber(effect && effect.ringColor, 0xfff27a);
      const slashTint = colorToNumber(effect && effect.slashColor, 0xff5d5d);
      const shardTint = colorToNumber(effect && effect.shardColor, burstTint);
      const secondaryShardTint = colorToNumber(effect && effect.secondaryShardColor, tint);
      const simplified = !!(metrics && metrics.simplified);
      const flash = Math.sin(clamp(progress / 0.24, 0, 1) * Math.PI);
      const seed = Number(effect && effect.x || 0) * 0.011 + Number(effect && effect.y || 0) * 0.017;
      if (!simplified) {
        this.drawShape('damage', 'glow', x, y + 2, glowW * (1.82 + flash * 0.22), glowH * (2.2 + flash * 0.24), {
          tint: burstTint,
          alpha: alpha * 0.3
        });
      }
      this.drawShape('damage', 'ring', x, y + 2, glowW * (1.32 + progress * 0.44), glowH * (1.58 + flash * 0.22), {
        tint: ringTint,
        alpha: alpha * (0.72 - progress * 0.22)
      });
      if (!simplified) {
        this.drawShape('damage', 'ring', x, y + 2, glowW * (1.88 + progress * 0.36), glowH * (2.18 + progress * 0.24), {
          tint: burstTint,
          alpha: alpha * (0.32 - progress * 0.1)
        });
      }
      const slashAlpha = alpha * (simplified ? 0.46 : 0.68) * (1 - progress * 0.34);
      this.drawShape('damage', 'slash', x - glowW * 0.04, y - glowH * 0.18, glowW * 1.65, glowH * 1.36, {
        tint: slashTint,
        alpha: slashAlpha,
        rotation: -0.14
      });
      if (!simplified) {
        this.drawShape('damage', 'slash', x + glowW * 0.05, y + glowH * 0.02, glowW * 1.36, glowH * 1.1, {
          tint,
          alpha: alpha * 0.46 * (1 - progress * 0.26),
          rotation: Math.PI - 0.28
        });
      }
      if (!simplified) {
        const starAlpha = alpha * (0.28 + flash * 0.42);
        this.drawLine('damage', x - glowW * 0.78, y + 2, x + glowW * 0.78, y + 2, 2.2, {
          tint,
          alpha: starAlpha
        });
        this.drawLine('damage', x, y - glowH * 0.88, x, y + glowH * 0.62, 2.2, {
          tint,
          alpha: starAlpha
        });
      }
      const shardCount = simplified ? 2 : 8;
      for (let i = 0; i < shardCount; i += 1) {
        const angle = seed + progress * Math.PI * 0.68 + i * (Math.PI * 2 / shardCount);
        const inner = glowW * (0.48 + progress * 0.08);
        const outer = glowW * (0.72 + progress * 0.28) + (i % 3) * 2;
        const yScale = 0.42;
        this.drawLine(
          'damage',
          x + Math.cos(angle) * inner,
          y + 2 + Math.sin(angle) * inner * yScale,
          x + Math.cos(angle) * outer,
          y + 2 + Math.sin(angle) * outer * yScale,
          simplified ? 1.2 : 1.8,
          {
            tint: i % 2 ? shardTint : secondaryShardTint,
            alpha: alpha * (simplified ? 0.46 : 0.64) * (1 - progress * 0.34)
          }
        );
      }
    }

    renderDamageSplatCosmeticAccent(effect, metrics) {
      const styleId = String(effect && effect.damageSplatStyleId || '');
      if (!styleId) return;
      const alpha = clamp(Number(metrics && metrics.alpha || 0), 0, 1);
      if (alpha <= 0) return;
      const progress = clamp(Number(metrics && metrics.progress || 0), 0, 1);
      const x = Number(metrics && metrics.x || 0);
      const y = Number(metrics && metrics.y || 0);
      const variant = String(effect.damageSplatVariant || styleId);
      const glowW = Math.max(28, Number(metrics && metrics.glowW || 34));
      const glowH = Math.max(14, Number(metrics && metrics.glowH || 24));
      const simplified = !!(metrics && metrics.simplified);
      const accentTint = colorToNumber(effect.accentColor, Number(metrics && metrics.tint || 0xfff4c7));
      const ringTint = colorToNumber(effect.ringColor, accentTint);
      const slashTint = colorToNumber(effect.slashColor, accentTint);
      const shardTint = colorToNumber(effect.shardColor, accentTint);
      const secondaryShardTint = colorToNumber(effect.secondaryShardColor, accentTint);
      const flash = Math.sin(clamp(progress / 0.24, 0, 1) * Math.PI);
      if (variant.includes('vault')) {
        this.drawRectOutline('damage', x - glowW * 0.54, y + 2 - glowH * 0.48, glowW * 1.08, glowH * 0.96, 2, {
          tint: accentTint,
          alpha: alpha * (simplified ? 0.28 : 0.46) * (1 - progress * 0.22)
        });
        [-1, 1].forEach((side) => {
          this.drawLine('damage', x + side * glowW * 0.18, y - glowH * 0.7, x + side * glowW * 0.42, y - glowH * 0.08, simplified ? 1.2 : 1.8, { tint: secondaryShardTint, alpha: alpha * 0.48 });
          this.drawLine('damage', x + side * glowW * 0.42, y - glowH * 0.08, x + side * glowW * 0.1, y + glowH * 0.5, simplified ? 1.2 : 1.8, { tint: accentTint, alpha: alpha * 0.48 });
        });
      } else if (variant.includes('frost')) {
        for (let i = -2; i <= 2; i += 1) {
          const sx = x + i * glowW * 0.16;
          this.drawLine('damage', sx, y + glowH * 0.32, x + i * glowW * 0.25, y - glowH * (0.8 + Math.abs(i) * 0.06) - flash * 4, simplified ? 1.1 : 1.7, {
            tint: i % 2 ? shardTint : secondaryShardTint,
            alpha: alpha * (simplified ? 0.36 : 0.58) * (1 - progress * 0.28)
          });
        }
      } else if (variant.includes('astral')) {
        this.drawShape('damage', 'ring', x, y + 2, glowW * (1.28 + progress * 0.16), glowH * 1.42, {
          tint: ringTint,
          alpha: alpha * (simplified ? 0.26 : 0.44)
        });
        this.drawLine('damage', x - glowW * 0.82, y + 2, x + glowW * 0.82, y + 2, simplified ? 1.2 : 2, { tint: secondaryShardTint, alpha: alpha * 0.44 });
        this.drawLine('damage', x, y - glowH * 0.76, x, y + glowH * 0.64, simplified ? 1.2 : 2, { tint: secondaryShardTint, alpha: alpha * 0.44 });
      } else if (variant.includes('storm')) {
        [-1, 1].forEach((side) => {
          this.drawLine('damage', x - glowW * 0.62 * side, y - glowH * 0.82, x - glowW * 0.16 * side, y - glowH * 0.12, simplified ? 1.6 : 2.4, {
            tint: slashTint,
            alpha: alpha * (simplified ? 0.42 : 0.68) * (1 - progress * 0.3)
          });
          this.drawLine('damage', x - glowW * 0.16 * side, y - glowH * 0.12, x + glowW * 0.62 * side, y + glowH * 0.72, simplified ? 1.6 : 2.4, {
            tint: accentTint,
            alpha: alpha * (simplified ? 0.42 : 0.68) * (1 - progress * 0.3)
          });
        });
      } else if (variant.includes('verdant')) {
        const petals = simplified ? 4 : 6;
        for (let i = 0; i < petals; i += 1) {
          const angle = progress * 0.8 + i * (Math.PI * 2 / petals);
          this.drawShape('damage', 'glow', x + Math.cos(angle) * glowW * 0.58, y + 2 + Math.sin(angle) * glowH * 0.48, glowW * 0.28, glowH * 0.28, {
            tint: i % 2 ? shardTint : ringTint,
            alpha: alpha * (simplified ? 0.22 : 0.36) * (1 - progress * 0.24)
          });
        }
      } else {
        const sparks = simplified ? 5 : 8;
        for (let i = 0; i < sparks; i += 1) {
          const angle = progress * Math.PI + i * (Math.PI * 2 / sparks);
          this.drawLine(
            'damage',
            x + Math.cos(angle) * glowW * 0.58,
            y + 2 + Math.sin(angle) * glowH * 0.38,
            x + Math.cos(angle) * glowW * (0.82 + flash * 0.12),
            y + 2 + Math.sin(angle) * glowH * 0.66,
            simplified ? 1.1 : 1.7,
            { tint: i % 2 ? shardTint : accentTint, alpha: alpha * (simplified ? 0.36 : 0.58) * (1 - progress * 0.28) }
          );
        }
      }
    }

    renderDamageSplats(snapshot) {
      const quality = snapshot.visualQuality || {};
      const simplified = !!quality.simplifyDamageSplats || quality.level === 'reduced';
      const effects = snapshot.damageSplats || [];
      for (let effectIndex = 0; effectIndex < effects.length; effectIndex += 1) {
        const effect = effects[effectIndex];
        if (!effect) continue;
        const visibleAge = Number(effect.age) || 0;
        if (visibleAge < 0) continue;
        const duration = Math.max(0.01, Number(effect.duration) || 1.05);
        const progress = clamp(visibleAge / duration, 0, 1);
        const fadeIn = clamp(visibleAge / 0.08, 0, 1);
        const fadeOut = progress < 0.72 ? 1 : clamp((1 - progress) / 0.28, 0, 1);
        const alpha = fadeIn * fadeOut;
        if (alpha <= 0.01) continue;
        const text = String(effect.text || '');
        if (!text) continue;
        const stacked = !!effect.stacked && Number(effect.lineCount || 1) > 1;
        const critical = !!effect.critical;
        const color = effect.color || '#fff4c7';
        const strokeColor = effect.stroke || 'rgba(9,31,59,0.92)';
        const tint = colorToNumber(color, critical ? 0xfff4c7 : 0xffffff);
        const baseScale = Number(effect.scale) || 1;
        const popProgress = clamp(progress / 0.2, 0, 1);
        const scale = baseScale * (0.92 + clamp(progress / 0.08, 0, 1) * 0.08 + (simplified ? 0 : Math.sin(popProgress * Math.PI) * 0.14));
        const fontSize = text.length > 4 ? 13 : stacked ? 17 : 20;
        const x = Number(effect.x || 0);
        const y = Number(effect.y || 0);
        const glowW = Math.max(stacked ? 42 : 34, text.length * fontSize * 0.42 + 18);
        const glowH = stacked ? 20 : 24;
        if (effect.damageSplatStyleId && (!simplified || critical)) {
          this.renderDamageSplatCosmeticAccent(effect, {
            alpha,
            progress,
            x,
            y,
            glowW,
            glowH,
            tint,
            simplified
          });
        }
        if (critical) {
          this.renderCriticalDamageSplatBurst(effect, {
            alpha,
            progress,
            x,
            y,
            glowW,
            glowH,
            tint,
            simplified
          });
        }
        if (!simplified || critical || effect.targetType === 'player') {
          this.drawShape('damage', critical ? 'ring' : 'glow', x, y + 2, glowW * (critical ? 1.42 : 1.08), glowH * (critical ? 1.72 : 1.26), {
            tint,
            alpha: critical ? alpha * 0.62 : alpha * 0.32
          });
        }
        const needsTextBackplate = !simplified || critical || effect.targetType === 'player';
        if (needsTextBackplate) {
          this.drawSolidRect('damage', x - glowW / 2, y + 2 - glowH / 2, glowW, glowH, {
            tint: 0x091f3b,
            alpha: simplified ? alpha * 0.12 : alpha * 0.2
          });
        }
        this.drawText('damageText', text, x, y, {
          fill: color,
          stroke: strokeColor,
          strokeWidth: stacked ? 3 : 4,
          fontSize,
          scale,
          alpha
        });
        const subtext = String(effect.subtext || '');
        if (!critical && subtext && !simplified) {
          this.drawText('damageText', subtext, x, y + 15 * scale, {
            fill: effect.subtextColor || '#e9f7ff',
            stroke: strokeColor,
            strokeWidth: 2,
            fontSize: 9,
            scale: Math.max(0.8, scale * 0.9),
            alpha: alpha * 0.86
          });
        }
      }
    }

    renderActorRig(actor, box, alpha) {
      if (!actor || !actor.rigRender || !box) return false;
      const rendered = this.getRigFrameTexture(actor.rigRender, box);
      if (!rendered || !rendered.texture) return false;
      const drawX = Number(box.x || 0) + Number(box.w || 0) / 2;
      const drawY = Number(box.y || 0) + Number(box.h || 0);
      const drawn = this.drawTexture('entities', rendered.texture, drawX, drawY, rendered.width, rendered.height, {
        alpha,
        flipX: Number(actor.facing || 1) < 0,
        anchorX: rendered.anchorX,
        anchorY: rendered.anchorY
      });
      if (drawn) this.frameStats.rigDraws += 1;
      return drawn;
    }

    renderActorSprite(actor, box, alpha) {
      const frameTexture = this.getFrameTexture(actor.animationFrame);
      if (frameTexture) {
        const kind = String(actor.kind || '');
        return this.drawActorTexture(frameTexture, box, {
          alpha,
          flipX: Number(actor.facing || 1) < 0,
          key: `frame:${this.getFrameTextureKey(actor.animationFrame)}`,
          kind,
          airborne: actor.behavior === 'flyer',
          trim: actor.kind !== 'enemy',
          registration: kind === 'enemy'
            ? ENEMY_SPRITE_REGISTRATION
            : kind === 'player' || kind === 'party' ? PLAYER_SPRITE_REGISTRATION : null
        });
      }
      const assetTexture = this.getTexture(actor.asset);
      if (!assetTexture) return false;
      return this.drawActorTexture(assetTexture, box, {
        alpha,
        flipX: Number(actor.facing || 1) < 0,
        key: `asset:${actor.asset || ''}`,
        kind: actor.kind || '',
        airborne: actor.behavior === 'flyer'
      });
    }

    drawActorTexture(texture, box, options) {
      if (!texture || !box) return false;
      const settings = options || {};
      const registration = settings.registration;
      if (registration) {
        const contentW = Math.max(1, Number(texture.width || 1));
        const contentH = Math.max(1, Number(texture.height || 1));
        const originX = Number.isFinite(Number(registration.originX)) ? Number(registration.originX) : 80;
        const groundY = Number.isFinite(Number(registration.groundY)) ? Number(registration.groundY) : 154;
        const authoredBodyHeight = Math.max(1, Number(registration.authoredBodyHeight) || 143);
        const scale = Math.max(0.01, Number(box.h || 0) / authoredBodyHeight);
        return this.drawTexture(
          'entities',
          texture,
          Number(box.x || 0) + Number(box.w || 0) / 2,
          Number(box.y || 0) + Number(box.h || 0),
          contentW * scale,
          contentH * scale,
          {
            alpha: settings.alpha,
            flipX: !!settings.flipX,
            anchorX: originX / contentW,
            anchorY: groundY / contentH
          }
        );
      }
      const display = settings.trim === false
        ? { texture, width: Math.max(1, Number(texture.width || 1)), height: Math.max(1, Number(texture.height || 1)) }
        : this.getTrimmedTexture(texture, settings.key) || { texture };
      const content = display.texture || texture;
      const contentW = Math.max(1, Number(display.width || content.width || 1));
      const contentH = Math.max(1, Number(display.height || content.height || 1));
      const kind = String(settings.kind || '');
      const maxWidth = Number(box.w || 1) * (kind === 'player' ? 1.65 : kind === 'party' ? 1.45 : 1);
      const maxHeight = Number(box.h || 1) * (kind === 'player' || kind === 'party' ? 1.02 : 1);
      const scale = Math.max(0.01, Math.min(maxWidth / contentW, maxHeight / contentH));
      const drawW = contentW * scale;
      const drawH = contentH * scale;
      const anchorY = settings.airborne ? 0.5 : 1;
      const drawX = Number(box.x || 0) + Number(box.w || 0) / 2;
      const drawY = settings.airborne
        ? Number(box.y || 0) + Number(box.h || 0) / 2
        : Number(box.y || 0) + Number(box.h || 0);
      return this.drawTexture('entities', content, drawX, drawY, drawW, drawH, {
        alpha: settings.alpha,
        flipX: !!settings.flipX,
        anchorX: 0.5,
        anchorY
      });
    }

    drawTexture(poolName, texture, x, y, width, height, options) {
      if (!texture) return false;
      const sprite = this.acquireSprite(poolName);
      if (!sprite) return false;
      const settings = options || {};
      sprite.texture = texture;
      sprite.anchor.set(
        settings.anchorX == null ? 0.5 : Number(settings.anchorX),
        settings.anchorY == null ? 0.5 : Number(settings.anchorY)
      );
      sprite.position.set(Number(x || 0), Number(y || 0));
      sprite.rotation = Number(settings.rotation || 0);
      sprite.alpha = clamp(settings.alpha == null ? 1 : Number(settings.alpha), 0, 1);
      sprite.tint = settings.tint == null ? FALLBACK_COLOR : colorToNumber(settings.tint, FALLBACK_COLOR);
      sprite.blendMode = settings.blendMode || 'normal';
      const texW = Math.max(1, Number(texture.width || 1));
      const texH = Math.max(1, Number(texture.height || 1));
      const drawW = Math.max(1, Number(width || texW));
      const drawH = Math.max(1, Number(height || texH));
      sprite.scale.set((settings.flipX ? -1 : 1) * drawW / texW, drawH / texH);
      return true;
    }

    getInfo() {
      return {
        backend: 'PixiJS WebGL',
        ready: !!this.ready,
        failed: !!this.failed,
        active: !!this.active,
        width: this.width,
        height: this.height,
        resolution: this.resolution,
        textureCount: this.textures.size,
        runtimeTextureCount: this.runtimeTextures.size,
        environmentTextureCount: this.environmentTextures.size,
        compositeTextureCount: this.compositeTextures.size,
        trimmedTextureCount: this.trimmedTextures.size,
        rigTextureCount: this.rigTextures.size,
        pendingTextures: this.loadingTextures.size,
        visualQuality: this.lastVisualQuality && this.lastVisualQuality.level || 'normal',
        actorFallbacks: this.frameStats && this.frameStats.actorFallbacks || 0,
        rigDraws: this.frameStats && this.frameStats.rigDraws || 0,
        spritePools: Object.entries(this.spritePools).reduce((result, entry) => {
          result[entry[0]] = { size: entry[1].items.length, active: entry[1].active };
          return result;
        }, {}),
        textPools: Object.entries(this.textPools).reduce((result, entry) => {
          result[entry[0]] = { size: entry[1].items.length, active: entry[1].active };
          return result;
        }, {})
      };
    }

    destroy() {
      this.setActive(false);
      if (this.app && typeof this.app.destroy === 'function') {
        this.app.destroy(true);
      }
      this.clearOwnedCache(this.compositeTextures);
      this.clearOwnedCache(this.trimmedTextures);
      this.clearOwnedCache(this.frameTextures);
      this.clearOwnedCache(this.rigTextures);
      this.clearOwnedCache(this.environmentTextures);
      this.clearOwnedCache(this.runtimeTextures);
      Array.from(this.textures.keys()).forEach((path) => this.deleteBaseTexture(path));
      this.idlePrewarmQueue = [];
      this.idlePrewarmKeys.clear();
      this.idlePrewarmScheduled = false;
      if (this.mapSceneryPlacementCache) this.mapSceneryPlacementCache.clear();
      this.loadingTextures.clear();
      this.app = null;
      this.ready = false;
    }
  }

  function markTiming(timings, name, startedAt) {
    const now = nowMs();
    timings[name] = now - startedAt;
    return now;
  }

  function nowMs() {
    if (global.performance && typeof global.performance.now === 'function') return global.performance.now();
    return Date.now();
  }

  function clamp(value, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number)) return min;
    return Math.min(max, Math.max(min, number));
  }

  function shouldRenderProjectileFxHorizontally(projectile) {
    return projectile &&
      projectile.owner === 'player' &&
      HORIZONTAL_PLAYER_PROJECTILE_TYPES.includes(String(projectile.type || '').toLowerCase());
  }

  function colorToNumber(value, fallback) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    const text = String(value || '').trim();
    if (!text) return fallback == null ? FALLBACK_COLOR : fallback;
    if (text[0] === '#') {
      const hex = text.length === 4
        ? text.slice(1).split('').map((char) => char + char).join('')
        : text.slice(1, 7);
      const parsed = parseInt(hex, 16);
      return Number.isFinite(parsed) ? parsed : fallback;
    }
    const rgb = text.match(/rgba?\(([^)]+)\)/i);
    if (rgb) {
      const parts = rgb[1].split(',').map((part) => Math.max(0, Math.min(255, Number(part.trim()) || 0)));
      return ((parts[0] || 0) << 16) + ((parts[1] || 0) << 8) + (parts[2] || 0);
    }
    return fallback == null ? FALLBACK_COLOR : fallback;
  }

  global.ProjectStarfallPixiRenderer = { createRenderer };
  if (typeof module !== 'undefined') module.exports = global.ProjectStarfallPixiRenderer;
})(typeof window !== 'undefined' ? window : globalThis);
