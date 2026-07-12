(function initProjectStarfallDataMapGeometry(global) {
  'use strict';

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function getPlatformDefX(platform) {
    return Array.isArray(platform) ? Number(platform[0] || 0) : Number(platform && platform.x || 0);
  }

  function getPlatformDefY(platform) {
    return Array.isArray(platform) ? Number(platform[1] || 0) : Number(platform && platform.y || 0);
  }

  function getPlatformDefW(platform) {
    return Array.isArray(platform) ? Number(platform[2] || 0) : Number(platform && platform.w || 0);
  }

  function getPlatformDefH(platform) {
    return Array.isArray(platform) ? Number(platform[3] || 0) : Number(platform && platform.h || 0);
  }

  function getPlatformDefY2(platform) {
    if (Array.isArray(platform)) return getPlatformDefY(platform);
    const value = Number(platform && platform.y2);
    return Number.isFinite(value) ? value : getPlatformDefY(platform);
  }

  function getPlatformDefShape(platform) {
    return Array.isArray(platform) ? 'flat' : String(platform && platform.shape || 'flat');
  }

  function getPlatformDefVisualKind(platform) {
    if (getPlatformDefShape(platform) === 'slope') return 'slope';
    return Array.isArray(platform)
      ? 'flat'
      : String(platform && platform.terrainVisual && platform.terrainVisual.kind || 'flat');
  }

  function getPlatformDefRight(platform) {
    return getPlatformDefX(platform) + getPlatformDefW(platform);
  }

  function getPlatformDefSurfaceY(platform, x) {
    const y = getPlatformDefY(platform);
    const y2 = getPlatformDefY2(platform);
    if (getPlatformDefShape(platform) !== 'slope' || Math.abs(y2 - y) < 1) return y;
    const startX = getPlatformDefX(platform);
    const width = Math.max(1, getPlatformDefW(platform));
    return y + (y2 - y) * clamp((Number(x || 0) - startX) / width, 0, 1);
  }

  function makePlatformDef(x, y, w, h, visual) {
    const platform = {
      x: Math.round(Number(x || 0)),
      y: Math.round(Number(y || 0)),
      w: Math.round(Number(w || 0)),
      h: Math.round(Number(h || 22))
    };
    if (visual) platform.terrainVisual = Object.assign({ segments: Object.freeze([]) }, visual);
    return platform;
  }

  function makeSlopePlatformDef(x, y, y2, w, h, visual) {
    const platform = makePlatformDef(x, y, w, h, Object.assign({ kind: 'slope' }, visual || {}));
    platform.shape = 'slope';
    platform.y2 = Math.round(Number(y2 || y));
    return platform;
  }

  function normalizePlatformIdToken(value, fallback) {
    const token = String(value || '')
      .trim()
      .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
      .replace(/[^a-zA-Z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .toLowerCase();
    return token || fallback || 'platform';
  }

  function assignStablePlatformIds(prefix, platforms) {
    const mapToken = normalizePlatformIdToken(prefix, 'map');
    const counts = Object.create(null);
    return Object.freeze((platforms || []).map((platform, index) => {
      const normalized = Array.isArray(platform)
        ? makePlatformDef(platform[0], platform[1], platform[2], platform[3], index === 0 ? { kind: 'ground' } : { kind: 'solidLane' })
        : Object.assign({}, platform || {});
      const visualKind = index === 0
        ? 'ground'
        : normalizePlatformIdToken(getPlatformDefVisualKind(normalized), 'lane');
      counts[visualKind] = Number(counts[visualKind] || 0) + 1;
      const suffix = visualKind === 'ground' ? 'ground' : `${visualKind}_${String(counts[visualKind]).padStart(2, '0')}`;
      normalized.id = String(normalized.id || `${mapToken}_${suffix}`);
      return Object.freeze(normalized);
    }));
  }

  function isSlopePlatformDef(platform) {
    return getPlatformDefShape(platform) === 'slope' && Math.abs(getPlatformDefY2(platform) - getPlatformDefY(platform)) >= 1;
  }

  function findRampEndpointPlatformDef(platforms, slopeIndex, x, y) {
    return (platforms || [])
      .map((platform, index) => ({ platform, index }))
      .filter((entry) => entry.index !== slopeIndex && !isSlopePlatformDef(entry.platform))
      .map((entry) => {
        const left = getPlatformDefX(entry.platform);
        const right = getPlatformDefRight(entry.platform);
        const horizontalGap = x < left ? left - x : x > right ? x - right : 0;
        const surfaceY = getPlatformDefSurfaceY(entry.platform, clamp(x, left, right));
        return Object.assign({}, entry, {
          horizontalGap,
          verticalGap: Math.abs(surfaceY - y),
          surfaceY
        });
      })
      .filter((entry) => entry.horizontalGap <= 36 && entry.verticalGap <= 24)
      .sort((a, b) =>
        a.verticalGap - b.verticalGap ||
        a.horizontalGap - b.horizontalGap ||
        getPlatformDefW(b.platform) - getPlatformDefW(a.platform)
      )[0] || null;
  }

  function makeRampConnections(prefix, platforms) {
    const connections = [];
    (platforms || []).forEach((platform, index) => {
      if (!isSlopePlatformDef(platform)) return;
      const left = {
        side: 'left',
        x: getPlatformDefX(platform),
        y: getPlatformDefY(platform)
      };
      const right = {
        side: 'right',
        x: getPlatformDefRight(platform),
        y: getPlatformDefY2(platform)
      };
      left.match = findRampEndpointPlatformDef(platforms, index, left.x, left.y);
      right.match = findRampEndpointPlatformDef(platforms, index, right.x, right.y);
      if (!left.match || !right.match) return;
      const lower = left.y >= right.y ? left : right;
      const upper = lower === left ? right : left;
      connections.push(Object.freeze({
        id: `${prefix}_ramp_${connections.length + 1}`,
        rampPlatformIndex: index,
        rampPlatformId: String(platform && platform.id || ''),
        lowerPlatformIndex: lower.match.index,
        lowerPlatformId: String(lower.match.platform && lower.match.platform.id || ''),
        upperPlatformIndex: upper.match.index,
        upperPlatformId: String(upper.match.platform && upper.match.platform.id || ''),
        lowerX: Math.round(lower.x),
        upperX: Math.round(upper.x),
        lowerSide: lower.side,
        upperSide: upper.side
      }));
    });
    return Object.freeze(connections);
  }

  const api = Object.freeze({
    getPlatformDefX,
    getPlatformDefY,
    getPlatformDefW,
    getPlatformDefH,
    getPlatformDefY2,
    getPlatformDefShape,
    getPlatformDefVisualKind,
    getPlatformDefRight,
    getPlatformDefSurfaceY,
    makePlatformDef,
    makeSlopePlatformDef,
    assignStablePlatformIds,
    isSlopePlatformDef,
    findRampEndpointPlatformDef,
    makeRampConnections
  });

  const modules = global.ProjectStarfallDataModules || {};
  modules.mapGeometry = Object.assign({}, modules.mapGeometry || {}, api);
  global.ProjectStarfallDataModules = modules;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof globalThis !== 'undefined' ? globalThis : window);
