(function initProjectStarfallSceneryPlacement(global) {
  'use strict';

  const Geometry = (typeof require === 'function' ? require('../core/geometry.js') : null) || global.ProjectStarfallCore;
  const MathCore = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore;

  function getDecorationBlockers(runtime) {
    const source = runtime || {};
    const blockers = [];
    (source.climbables || []).forEach(item => blockers.push({ x: item.x - 36, y: item.y - 16, w: item.w + 72, h: item.h + 32 }));
    (source.stations || []).forEach(item => blockers.push({ x: item.x - 52, y: item.y - 90, w: item.w + 104, h: item.h + 112 }));
    (source.portals || []).forEach(item => blockers.push({ x: item.x - 56, y: item.y - 40, w: item.w + 112, h: item.h + 74 }));
    (source.questNpcs || []).forEach(item => blockers.push({ x: item.x - 44, y: item.y - 54, w: item.w + 88, h: item.h + 78 }));
    (source.spawnPoints || []).forEach(point => {
      const platform = (source.platforms || [])[point.platformIndex || 0];
      if (!platform) return;
      const x = Number(point.x || 0);
      const leftY = Geometry.getPlatformSurfaceY(platform, x - 34);
      const rightY = Geometry.getPlatformSurfaceY(platform, x + 34);
      blockers.push({ x: x - 34, y: Math.min(leftY, rightY) - 72, w: 68, h: 94 + Math.abs(rightY - leftY) });
    });
    return blockers;
  }

  function isPlacementSafe(platform, x, y, w, h, blockers, visibility) {
    if (!platform || x < platform.x + 28 || x + w > platform.x + platform.w - 28) return false;
    const clearance = Number(visibility && visibility.combatClearancePx || 72);
    const rect = { x: x - clearance * 0.25, y: y - clearance * 0.2, w: w + clearance * 0.5, h: h + clearance * 0.35 };
    return !(blockers || []).some(blocker => MathCore.rectsOverlap(rect, blocker));
  }

  function getPropFooting(platform, x, w, h, kind) {
    if (!platform || w <= 0 || h <= 0) return null;
    const centerX = x + w / 2;
    const surfaceY = Geometry.getPlatformSurfaceY(platform, centerX);
    // Flexible plants have small roots; rigid props need their whole base supported.
    const flexible = ['grass', 'flower', 'vine', 'glow', 'tree', 'tall'].includes(kind);
    const footprint = w * (flexible ? 0.22 : 0.72);
    const left = centerX - footprint / 2;
    const right = centerX + footprint / 2;
    if (left < platform.x + 8 || right > platform.x + platform.w - 8) return null;
    const rise = Math.abs(Geometry.getPlatformSurfaceY(platform, right) - Geometry.getPlatformSurfaceY(platform, left));
    if (rise > (flexible ? 8 : 5)) return null;
    return { surfaceY, x, y: surfaceY - h, w, h, anchorX: centerX };
  }

  function buildPlacements(runtime, map, profile, visibility, layer, densityScale, getKinds, getSize) {
    const blockers = getDecorationBlockers(runtime);
    const density = Number(profile.density ?? 0.5) * densityScale;
    if (!Number.isFinite(density) || density <= 0) return [];
    const spacing = layer === 'rear' ? 430 : 340;
    const placements = [];
    (runtime.platforms || []).forEach((platform, platformIndex) => {
      if (!platform || platform.w < 120) return;
      const kinds = getKinds(profile, layer, platformIndex);
      if (!kinds.length) return;
      const rawCount = Math.max(1, Math.round(platform.w / spacing * density));
      const count = layer === 'rear' ? Math.min(8, rawCount) : Math.min(platformIndex === 0 ? 6 : 2, rawCount);
      for (let index = 0; index < count; index += 1) {
        const seed = `${map.id}:${layer}:${platformIndex}:${index}`;
        const kind = MathCore.seededPick(kinds, seed, 'kind') || 'grass';
        const size = getSize(kind, layer, platformIndex, visibility);
        const usableW = Math.max(1, platform.w - 96);
        const x = platform.x + 48 + Math.floor(usableW * ((index + 0.35 + MathCore.seededUnit(seed, 'x') * 0.3) / Math.max(1, count)));
        const footing = getPropFooting(platform, x, size.w, size.h, kind);
        if (!footing || !isPlacementSafe(platform, x, footing.y, size.w, size.h, blockers, visibility)) continue;
        placements.push({ kind, x, y: footing.y, w: size.w, h: size.h, seed, platformIndex, surfaceY: footing.surfaceY });
      }
    });
    return placements;
  }

  function getTerrainSurfaceTop(platform, style, topHeight, isGround) {
    // Contact-normalized stone atlases start at their walkable edge. Older
    // atlases retain their authored grass/lip allowance above the foot plane.
    return Number(platform.y) - (style && style.contactAligned ? 0 : topHeight - (isGround ? 0 : 2));
  }

  function createRampSurface(image, asset, cellIndex, width, rise, bodyDepth) {
    if (!global.document || !image || !asset) return null;
    const cellSize = Number(asset.cellSize || 128);
    const scan = global.document.createElement('canvas');
    scan.width = cellSize;
    scan.height = cellSize;
    const scanCtx = scan.getContext('2d', { willReadFrequently: true });
    const sourceX = cellIndex % Number(asset.columns || 4) * cellSize;
    const sourceY = Math.floor(cellIndex / Number(asset.columns || 4)) * cellSize;
    try {
      scanCtx.drawImage(image, sourceX, sourceY, cellSize, cellSize, 0, 0, cellSize, cellSize);
      const pixels = scanCtx.getImageData(0, 0, cellSize, cellSize).data;
      const columns = [];
      for (let x = 0; x < cellSize; x += 1) {
        let top = -1;
        let bottom = -1;
        for (let y = 0; y < cellSize; y += 1) {
          if (pixels[(y * cellSize + x) * 4 + 3] < 64) continue;
          if (top < 0) top = y;
          bottom = y;
        }
        // Thin pointed tips cannot provide a readable supporting ledge.
        if (top >= 0 && bottom - top >= 8) columns.push({ x, top, bottom });
      }
      if (columns.length < 8) return null;
      const canvas = global.document.createElement('canvas');
      canvas.width = Math.max(1, Math.ceil(width));
      canvas.height = Math.ceil(Math.abs(rise) + bodyDepth + 4);
      const ctx = canvas.getContext('2d');
      // A one-pixel source strip must not blend with its transparent neighbour.
      ctx.imageSmoothingEnabled = false;
      const first = columns[0].x;
      const last = columns[columns.length - 1].x;
      // Map the painted contact edge to the collision surface. Atlas padding and
      // source ramp angle must never move the walkable edge or create a wedge reset.
      for (let x = 0; x < canvas.width; x += 1) {
        const ratio = x / Math.max(1, canvas.width - 1);
        const sourceColumn = Math.round(first + (last - first) * ratio);
        const sample = columns.reduce((best, column) => Math.abs(column.x - sourceColumn) < Math.abs(best.x - sourceColumn) ? column : best, columns[0]);
        const surface = 2 + (rise < 0 ? Math.abs(rise) : 0) + rise * ratio;
        ctx.drawImage(scan, sample.x, sample.top, 1, Math.min(28, sample.bottom - sample.top + 1), x, surface, 1, bodyDepth);
      }
      return { canvas, topOffset: Math.min(0, rise) - 2, width: canvas.width, height: canvas.height };
    } catch {
      return null;
    }
  }

  const api = { getDecorationBlockers, isPlacementSafe, getPropFooting, buildPlacements, getTerrainSurfaceTop, createRampSurface };
  const modules = global.ProjectStarfallEngineModules || {};
  modules.sceneryPlacement = api;
  global.ProjectStarfallEngineModules = modules;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
