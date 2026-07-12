(function initProjectStarfallEngineMapRuntime(global) {
  'use strict';

  const CoreIds = (typeof require === 'function' ? require('../core/ids.js') : null) || global.ProjectStarfallCore || {};
  const CoreMath = (typeof require === 'function' ? require('../core/math.js') : null) || global.ProjectStarfallCore || {};
  const CoreGeometry = (typeof require === 'function' ? require('../core/geometry.js') : null) || global.ProjectStarfallCore || {};
  const EngineModules = global.ProjectStarfallEngineModules || {};
  const EngineViewport = (typeof require === 'function' ? require('./viewport.js') : null) || EngineModules.viewport || {};
  const EnginePortals = EngineModules.portals || {};
  const EngineQuestNpcs = EngineModules.questNpcs || {};
  const DEFAULT_PLATFORM_LINK_MAX_GAP = 220;
  const DEFAULT_PLATFORM_LINK_MAX_JUMP = 128;
  const DEFAULT_PLATFORM_LINK_MAX_DROP = 300;
  const DEFAULT_AUTHORED_GROUND_Y = 520;
  const getById = CoreIds.getById || function getByIdFallback(items, id) {
    return (items || []).find((item) => item && item.id === id) || null;
  };
  const normalizeId = CoreIds.normalizeId || function normalizeIdFallback(value) {
    return String(value || '').trim();
  };
  const clamp = CoreMath.clamp || function clampFallback(value, min, max) {
    return Math.max(min, Math.min(max, value));
  };
  const isSlopePlatform = CoreGeometry.isSlopePlatform || function isSlopePlatformFallback(platform) {
    return !!(platform && platform.shape === 'slope' && Number.isFinite(Number(platform.y2)));
  };
  const platformContainsX = CoreGeometry.platformContainsX || function platformContainsXFallback(platform, x, padding) {
    const pad = Number(padding) || 0;
    return !!platform && x >= platform.x - pad && x <= platform.x + platform.w + pad;
  };
  const getPlatformSurfaceY = CoreGeometry.getPlatformSurfaceY || function getPlatformSurfaceYFallback(platform, x) {
    if (!platform) return 0;
    const y = Number(platform.y || 0);
    if (!isSlopePlatform(platform)) return y;
    const width = Math.max(1, Number(platform.w || 0));
    const ratio = clamp((Number(x || 0) - Number(platform.x || 0)) / width, 0, 1);
    return y + (Number(platform.y2 || y) - y) * ratio;
  };
  const getPlatformTopY = CoreGeometry.getPlatformTopY || function getPlatformTopYFallback(platform) {
    if (!platform) return 0;
    return isSlopePlatform(platform) ? Math.min(Number(platform.y || 0), Number(platform.y2 || platform.y || 0)) : Number(platform.y || 0);
  };
  const getPlatformBottomY = CoreGeometry.getPlatformBottomY || function getPlatformBottomYFallback(platform) {
    if (!platform) return 0;
    const surfaceBottom = isSlopePlatform(platform) ? Math.max(Number(platform.y || 0), Number(platform.y2 || platform.y || 0)) : Number(platform.y || 0);
    return surfaceBottom + Math.max(1, Number(platform.h || 0));
  };
  const createViewportMetrics = EngineViewport.createViewportMetrics || function createViewportMetricsFallback() {
    const playfieldHeight = Number(EngineViewport.PLAYFIELD_HEIGHT || 640);
    const solidPlatformHeight = Number(EngineViewport.SOLID_PLATFORM_HEIGHT || 48);
    const statusHudHeight = Number(EngineViewport.CANVAS_STATUS_HUD_HEIGHT || 84);
    return {
      width: Number(EngineViewport.PLAYFIELD_WIDTH || 1280),
      height: playfieldHeight + solidPlatformHeight + statusHudHeight,
      playfieldHeight,
      solidPlatformHeight,
      statusHudHeight,
      hudTop: playfieldHeight + solidPlatformHeight,
      worldHeight: playfieldHeight + solidPlatformHeight
    };
  };

  function clonePlain(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createRuntimeTerrainVisual(visual, platform) {
    if (!visual) return null;
    return clonePlain(visual);
  }

  function getAuthoredPlatformX(platform) {
    return Array.isArray(platform) ? Number(platform[0] || 0) : Number(platform && platform.x || 0);
  }

  function getAuthoredPlatformY(platform) {
    return Array.isArray(platform) ? Number(platform[1] || 0) : Number(platform && platform.y || 0);
  }

  function getAuthoredPlatformW(platform) {
    return Array.isArray(platform) ? Number(platform[2] || 0) : Number(platform && platform.w || 0);
  }

  function getAuthoredPlatformH(platform) {
    return Array.isArray(platform) ? Number(platform[3] || 0) : Number(platform && platform.h || 0);
  }

  function getAuthoredPlatformY2(platform) {
    if (Array.isArray(platform)) return getAuthoredPlatformY(platform);
    const y2 = Number(platform && platform.y2);
    return Number.isFinite(y2) ? y2 : getAuthoredPlatformY(platform);
  }

  function getAuthoredPlatformShape(platform) {
    return Array.isArray(platform) ? 'flat' : normalizeId(platform && platform.shape) || 'flat';
  }

  function getAuthoredPlatformVisual(platform) {
    return !Array.isArray(platform) && platform && platform.terrainVisual ? platform.terrainVisual : null;
  }

  function createRuntimePlatformLayout(map, metrics, options) {
    const sourceMap = map || {};
    const sourceMetrics = metrics || {};
    const settings = options || {};
    const width = Number(sourceMetrics.width || EngineViewport.PLAYFIELD_WIDTH || 1280);
    const playfieldHeight = Number(sourceMetrics.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640);
    const solidPlatformHeight = Number(sourceMetrics.solidPlatformHeight || EngineViewport.SOLID_PLATFORM_HEIGHT || 48);
    const authoredWorldHeight = Math.max(0, Number(sourceMap.worldHeight || 0));
    const authoredGroundY = Math.max(0, Number(sourceMap.authoredGroundY || settings.authoredGroundY || DEFAULT_AUTHORED_GROUND_Y));
    const geometryYOffset = authoredWorldHeight > 0 ? 0 : playfieldHeight - authoredGroundY;
    const platformDefinitions = Array.isArray(sourceMap.platforms) && sourceMap.platforms.length ? sourceMap.platforms : [[0, authoredGroundY, 2200, 80]];
    const authoredPlatformRightEdge = platformDefinitions.reduce((rightEdge, platform) =>
      Math.max(rightEdge, getAuthoredPlatformX(platform) + getAuthoredPlatformW(platform)), 0);
    const worldWidth = sourceMap.shopInterior
      ? Math.max(width, authoredPlatformRightEdge)
      : Math.max(2200, authoredPlatformRightEdge) + 160;
    const terrainVisuals = Array.isArray(sourceMap.terrainVisuals) ? sourceMap.terrainVisuals : [];
    const platforms = platformDefinitions.map((platform, index) => {
      const x = getAuthoredPlatformX(platform);
      const y = getAuthoredPlatformY(platform) + geometryYOffset;
      const shape = getAuthoredPlatformShape(platform);
      const y2 = getAuthoredPlatformY2(platform) + geometryYOffset;
      const w = index === 0 ? Math.max(getAuthoredPlatformW(platform), worldWidth - x) : getAuthoredPlatformW(platform);
      const authoredVisual = getAuthoredPlatformVisual(platform);
      const runtimePlatform = {
        id: normalizeId(!Array.isArray(platform) && platform && platform.id) || `${sourceMap.id}_platform_${index}`,
        index,
        x,
        y,
        w,
        h: index === 0 ? solidPlatformHeight : getAuthoredPlatformH(platform),
        dropThrough: index > 0,
        terrainVisual: terrainVisuals[index]
          ? createRuntimeTerrainVisual(terrainVisuals[index], platform)
          : authoredVisual ? createRuntimeTerrainVisual(authoredVisual, platform) : null
      };
      if (shape === 'slope' && Math.abs(y2 - y) >= 1) {
        runtimePlatform.shape = 'slope';
        runtimePlatform.y2 = y2;
      }
      return runtimePlatform;
    });
    return {
      authoredWorldHeight,
      authoredGroundY,
      geometryYOffset,
      platformDefinitions,
      authoredPlatformRightEdge,
      worldWidth,
      platforms
    };
  }

  function platformHorizontalGap(a, b) {
    if (!a || !b) return Infinity;
    if (a.x <= b.x + b.w && b.x <= a.x + a.w) return 0;
    return a.x < b.x ? b.x - (a.x + a.w) : a.x - (b.x + b.w);
  }

  function platformTransferX(fromPlatform, toPlatform) {
    const fromLeft = fromPlatform.x + 28;
    const fromRight = fromPlatform.x + fromPlatform.w - 28;
    const overlapLeft = Math.max(fromLeft, toPlatform.x + 24);
    const overlapRight = Math.min(fromRight, toPlatform.x + toPlatform.w - 24);
    if (overlapLeft <= overlapRight) return (overlapLeft + overlapRight) / 2;
    return toPlatform.x > fromPlatform.x ? fromRight : fromLeft;
  }

  function findSurfacePlatform(platforms, x, y, tolerance, xTolerance) {
    const allowed = Number(tolerance) || 24;
    const allowedX = Number(xTolerance) || 24;
    return (platforms || [])
      .filter((platform) => platformContainsX(platform, x, allowedX) && Math.abs(getPlatformSurfaceY(platform, x) - y) <= allowed)
      .sort((a, b) => Math.abs(getPlatformSurfaceY(a, x) - y) - Math.abs(getPlatformSurfaceY(b, x) - y) || Math.abs((a.x + a.w / 2) - x) - Math.abs((b.x + b.w / 2) - x))[0] || null;
  }

  function alignClimbable(rawClimbable, index, mapId, platforms) {
    const climbable = rawClimbable || {};
    const x = Number(climbable.x) || 0;
    const w = Number(climbable.w) || 28;
    const rawTop = Number(climbable.y) || 0;
    const rawBottom = rawTop + (Number(climbable.h) || 120);
    const centerX = x + w / 2;
    const topPlatform = findSurfacePlatform(platforms, centerX, rawTop, 56, 96);
    const bottomPlatform = findSurfacePlatform(platforms, centerX, rawBottom, 56, 96);
    const topY = topPlatform ? getPlatformSurfaceY(topPlatform, centerX) : rawTop;
    const bottomSurfaceY = bottomPlatform ? getPlatformSurfaceY(bottomPlatform, centerX) : rawBottom;
    const bottomY = bottomPlatform && bottomSurfaceY > topY ? bottomSurfaceY : rawBottom;
    return {
      id: climbable.id || `${mapId}_climbable_${index}`,
      x,
      y: Math.round(topY),
      w,
      h: Math.max(1, Math.round(bottomY - topY)),
      topPlatformId: topPlatform ? topPlatform.id : '',
      topPlatformIndex: topPlatform ? topPlatform.index : -1,
      bottomPlatformId: bottomPlatform ? bottomPlatform.id : '',
      bottomPlatformIndex: bottomPlatform ? bottomPlatform.index : -1
    };
  }

  function createRuntimeRampConnections(map, platforms) {
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    return (Array.isArray(map && map.rampConnections) ? map.rampConnections : [])
      .map((connection, index) => {
        const rampPlatformIndex = Number(connection && connection.rampPlatformIndex);
        const lowerPlatformIndex = Number(connection && connection.lowerPlatformIndex);
        const upperPlatformIndex = Number(connection && connection.upperPlatformIndex);
        const rampPlatform = sourcePlatforms[rampPlatformIndex];
        const lowerPlatform = sourcePlatforms[lowerPlatformIndex];
        const upperPlatform = sourcePlatforms[upperPlatformIndex];
        if (!rampPlatform || !lowerPlatform || !upperPlatform || !isSlopePlatform(rampPlatform)) return null;
        const rawLowerX = Number(connection.lowerX);
        const rawUpperX = Number(connection.upperX);
        if (!Number.isFinite(rawLowerX) || !Number.isFinite(rawUpperX)) return null;
        const lowerX = clamp(rawLowerX, lowerPlatform.x + 2, lowerPlatform.x + Math.max(2, lowerPlatform.w - 2));
        const upperX = clamp(rawUpperX, upperPlatform.x + 2, upperPlatform.x + Math.max(2, upperPlatform.w - 2));
        const rampLowerX = clamp(rawLowerX, rampPlatform.x + 2, rampPlatform.x + Math.max(2, rampPlatform.w - 2));
        const rampUpperX = clamp(rawUpperX, rampPlatform.x + 2, rampPlatform.x + Math.max(2, rampPlatform.w - 2));
        return {
          id: normalizeId(connection.id) || `${map.id}_ramp_${index + 1}`,
          rampPlatformIndex: rampPlatform.index,
          rampPlatformId: rampPlatform.id,
          lowerPlatformIndex: lowerPlatform.index,
          lowerPlatformId: lowerPlatform.id,
          upperPlatformIndex: upperPlatform.index,
          upperPlatformId: upperPlatform.id,
          lowerX,
          upperX,
          rampLowerX,
          rampUpperX,
          lowerY: getPlatformSurfaceY(lowerPlatform, lowerX),
          upperY: getPlatformSurfaceY(upperPlatform, upperX)
        };
      })
      .filter(Boolean);
  }

  function addPlatformLink(graph, fromIndex, toIndex, type, exitX, options) {
    if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex || !graph[fromIndex]) return;
    const settings = typeof options === 'string' ? { climbableId: options } : options || {};
    const entryX = Number.isFinite(Number(settings.entryX)) ? Number(settings.entryX) : exitX;
    const duplicate = graph[fromIndex].some((link) =>
      link.to === toIndex &&
      link.type === type &&
      Math.abs(link.exitX - exitX) < 4 &&
      Math.abs((Number(link.entryX) || 0) - entryX) < 4
    );
    if (duplicate) return;
    graph[fromIndex].push({
      from: fromIndex,
      to: toIndex,
      type,
      exitX,
      entryX,
      climbableId: settings.climbableId || '',
      rampConnectionId: settings.rampConnectionId || ''
    });
  }

  function createPlatformGraph(platforms, climbables, rampConnections, options) {
    const settings = options || {};
    const maxGap = Number(settings.maxGap || settings.platformLinkMaxGap || DEFAULT_PLATFORM_LINK_MAX_GAP);
    const maxJump = Number(settings.maxJump || settings.platformLinkMaxJump || DEFAULT_PLATFORM_LINK_MAX_JUMP);
    const maxDrop = Number(settings.maxDrop || settings.platformLinkMaxDrop || DEFAULT_PLATFORM_LINK_MAX_DROP);
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const graph = sourcePlatforms.map(() => []);
    for (let i = 0; i < sourcePlatforms.length; i += 1) {
      for (let j = i + 1; j < sourcePlatforms.length; j += 1) {
        const a = sourcePlatforms[i];
        const b = sourcePlatforms[j];
        const gap = platformHorizontalGap(a, b);
        if (gap > maxGap) continue;
        const transferAB = platformTransferX(a, b);
        const transferBA = platformTransferX(b, a);
        const vertical = getPlatformSurfaceY(b, transferAB) - getPlatformSurfaceY(a, transferAB);
        if (Math.abs(vertical) <= 24) {
          const sameTierType = gap <= 28 ? 'walk' : 'jump';
          addPlatformLink(graph, a.index, b.index, sameTierType, transferAB);
          addPlatformLink(graph, b.index, a.index, sameTierType, transferBA);
        } else if (vertical > 24) {
          if (vertical <= maxDrop) addPlatformLink(graph, a.index, b.index, 'drop', transferAB);
          if (vertical <= maxJump) addPlatformLink(graph, b.index, a.index, 'jump', transferBA);
        } else if (-vertical > 24) {
          if (-vertical <= maxJump) addPlatformLink(graph, a.index, b.index, 'jump', transferAB);
          if (-vertical <= maxDrop) addPlatformLink(graph, b.index, a.index, 'drop', transferBA);
        }
      }
    }
    (climbables || []).forEach((climbable) => {
      if (climbable.topPlatformIndex < 0 || climbable.bottomPlatformIndex < 0) return;
      const centerX = climbable.x + climbable.w / 2;
      addPlatformLink(graph, climbable.topPlatformIndex, climbable.bottomPlatformIndex, 'ladder-down', centerX, climbable.id);
      addPlatformLink(graph, climbable.bottomPlatformIndex, climbable.topPlatformIndex, 'ladder-up', centerX, climbable.id);
    });
    (rampConnections || []).forEach((connection) => {
      const rampIndex = Number(connection && connection.rampPlatformIndex);
      const lowerIndex = Number(connection && connection.lowerPlatformIndex);
      const upperIndex = Number(connection && connection.upperPlatformIndex);
      const lowerX = Number(connection && connection.lowerX);
      const upperX = Number(connection && connection.upperX);
      const rampLowerX = Number.isFinite(Number(connection && connection.rampLowerX)) ? Number(connection.rampLowerX) : lowerX;
      const rampUpperX = Number.isFinite(Number(connection && connection.rampUpperX)) ? Number(connection.rampUpperX) : upperX;
      if (!Number.isInteger(rampIndex) || !Number.isInteger(lowerIndex) || !Number.isInteger(upperIndex)) return;
      if (!graph[rampIndex] || !graph[lowerIndex] || !graph[upperIndex]) return;
      if (!Number.isFinite(lowerX) || !Number.isFinite(upperX)) return;
      const linkOptions = { rampConnectionId: connection.id || '' };
      addPlatformLink(graph, lowerIndex, rampIndex, 'ramp-up', lowerX, Object.assign({ entryX: rampLowerX }, linkOptions));
      addPlatformLink(graph, rampIndex, upperIndex, 'ramp-up', rampUpperX, Object.assign({ entryX: upperX }, linkOptions));
      addPlatformLink(graph, upperIndex, rampIndex, 'ramp-down', upperX, Object.assign({ entryX: rampUpperX }, linkOptions));
      addPlatformLink(graph, rampIndex, lowerIndex, 'ramp-down', rampLowerX, Object.assign({ entryX: lowerX }, linkOptions));
    });
    graph.forEach((links) => {
      links.sort((a, b) => {
        const typeScore = { walk: 0, 'ramp-up': 0, 'ramp-down': 0, 'ladder-up': 1, 'ladder-down': 1, jump: 2, drop: 3 };
        const scoreA = Object.prototype.hasOwnProperty.call(typeScore, a.type) ? typeScore[a.type] : 9;
        const scoreB = Object.prototype.hasOwnProperty.call(typeScore, b.type) ? typeScore[b.type] : 9;
        return scoreA - scoreB || Math.abs(a.exitX) - Math.abs(b.exitX);
      });
    });
    return graph;
  }

  function findPlatformRouteLink(graph, fromIndex, toIndex, options) {
    if (!Array.isArray(graph) || fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return null;
    const excludedTypes = new Set(Array.isArray(options && options.excludeTypes) ? options.excludeTypes : []);
    const typeCost = { walk: 1, 'ramp-up': 1, 'ramp-down': 1, 'ladder-up': 2, 'ladder-down': 2, jump: 6, drop: 7 };
    const queue = [{ index: fromIndex, first: null, cost: 0 }];
    const bestCost = new Map([[fromIndex, 0]]);
    while (queue.length) {
      queue.sort((a, b) => a.cost - b.cost);
      const current = queue.shift();
      if (current.index === toIndex) return current.first;
      const knownCurrentCost = bestCost.has(current.index) ? bestCost.get(current.index) : Infinity;
      if (current.cost > knownCurrentCost) continue;
      const links = graph[current.index] || [];
      for (const link of links) {
        if (excludedTypes.has(link.type)) continue;
        const linkCost = Object.prototype.hasOwnProperty.call(typeCost, link.type) ? typeCost[link.type] : 9;
        const nextCost = current.cost + linkCost;
        const knownNextCost = bestCost.has(link.to) ? bestCost.get(link.to) : Infinity;
        if (nextCost >= knownNextCost) continue;
        const first = current.first || link;
        bestCost.set(link.to, nextCost);
        queue.push({ index: link.to, first, cost: nextCost });
      }
    }
    return null;
  }

  function isRampRouteLink(link) {
    return !!(link && (link.type === 'ramp-up' || link.type === 'ramp-down'));
  }

  function getPlatformSurfaceLength(platform) {
    if (!platform) return 0;
    const width = Math.max(0, Number(platform.w || 0));
    if (!isSlopePlatform(platform)) return width;
    const rise = Number(platform.y2 || platform.y || 0) - Number(platform.y || 0);
    return Math.round(Math.sqrt(width * width + rise * rise));
  }

  function getPlatformTierId(platform, groundY) {
    if (!platform) return 'unknown';
    const topY = getPlatformTopY(platform);
    const verticalOffset = Number(groundY || 0) - topY;
    if (verticalOffset < 96) return 'ground';
    if (verticalOffset < 260) return 'low';
    if (verticalOffset < 440) return 'mid';
    if (verticalOffset < 640) return 'high';
    return 'sky';
  }

  function getPlatformVisualKind(platform) {
    const visual = platform && platform.terrainVisual || {};
    if (platform && isSlopePlatform(platform)) return 'slope';
    return normalizeId(visual.kind) || 'platform';
  }

  function createRuntimeFootholds(map, platforms, platformGraph, options) {
    const settings = options || {};
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const groundY = sourcePlatforms[0] ? getPlatformSurfaceY(sourcePlatforms[0], sourcePlatforms[0].x) : Number(settings.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640);
    return Object.freeze(sourcePlatforms.map((platform, index) => {
      const x1 = Number(platform.x || 0);
      const x2 = x1 + Number(platform.w || 0);
      const y1 = getPlatformSurfaceY(platform, x1);
      const y2 = getPlatformSurfaceY(platform, x2);
      const centerX = x1 + Number(platform.w || 0) / 2;
      const id = `${map && map.id || 'map'}_foothold_${index}`;
      const links = (platformGraph && platformGraph[index] || [])
        .map((link) => {
          const target = sourcePlatforms[link.to];
          if (!target) return null;
          return {
            toFootholdId: `${map && map.id || 'map'}_foothold_${link.to}`,
            toPlatformId: target.id || '',
            toPlatformIndex: target.index,
            type: link.type,
            exitX: Math.round(Number(link.exitX || 0)),
            entryX: Math.round(Number(link.entryX || 0)),
            climbableId: link.climbableId || '',
            rampConnectionId: link.rampConnectionId || ''
          };
        })
        .filter(Boolean);
      const lateralLinks = links
        .map((link) => Object.assign({}, link, {
          target: sourcePlatforms[link.toPlatformIndex],
          targetCenterX: sourcePlatforms[link.toPlatformIndex] ? sourcePlatforms[link.toPlatformIndex].x + sourcePlatforms[link.toPlatformIndex].w / 2 : centerX
        }))
        .filter((link) => link.target && ['walk', 'jump', 'drop', 'ramp-up', 'ramp-down'].includes(link.type));
      const prev = lateralLinks
        .filter((link) => link.targetCenterX < centerX)
        .sort((a, b) => centerX - b.targetCenterX - (centerX - a.targetCenterX))[0] || null;
      const next = lateralLinks
        .filter((link) => link.targetCenterX >= centerX)
        .sort((a, b) => a.targetCenterX - centerX - (b.targetCenterX - centerX))[0] || null;
      return Object.freeze({
        id,
        platformId: platform.id || '',
        platformIndex: index,
        layerId: getPlatformTierId(platform, groundY),
        groupId: `${map && map.id || 'map'}_${getPlatformTierId(platform, groundY)}`,
        kind: index === 0 ? 'ground' : getPlatformVisualKind(platform),
        x1: Math.round(x1),
        y1: Math.round(y1),
        x2: Math.round(x2),
        y2: Math.round(y2),
        length: getPlatformSurfaceLength(platform),
        dropThrough: !!platform.dropThrough,
        prev: prev ? prev.toFootholdId : '',
        next: next ? next.toFootholdId : '',
        links: Object.freeze(links.map((link) => Object.freeze(link)))
      });
    }));
  }

  function getReachablePlatformIndices(platformGraph, startIndex, allowedTypes) {
    const start = Math.floor(Number(startIndex));
    if (!Array.isArray(platformGraph) || !Number.isInteger(start) || start < 0 || !platformGraph[start]) return new Set();
    const allowed = Array.isArray(allowedTypes) && allowedTypes.length ? new Set(allowedTypes) : null;
    const seen = new Set([start]);
    const queue = [start];
    while (queue.length) {
      const current = queue.shift();
      (platformGraph[current] || []).forEach((link) => {
        if (!link || allowed && !allowed.has(link.type)) return;
        const next = Math.floor(Number(link.to));
        if (!Number.isInteger(next) || seen.has(next) || !platformGraph[next]) return;
        seen.add(next);
        queue.push(next);
      });
    }
    return seen;
  }

  function summarizePlatformGraphLinks(platformGraph) {
    const counts = {};
    (platformGraph || []).forEach((links) => {
      (links || []).forEach((link) => {
        const type = normalizeId(link && link.type) || 'unknown';
        counts[type] = (counts[type] || 0) + 1;
      });
    });
    return Object.freeze(counts);
  }

  function countGraphLinks(counts, types) {
    return (types || []).reduce((total, type) => total + Number(counts && counts[type] || 0), 0);
  }

  function createTrainingRouteContract(map, platforms, platformGraph, spawnPoints, climbables, rampConnections) {
    const combatMap = !!(map && !map.safeZone && !map.shopInterior && !map.adminOnly);
    const broadPlatforms = (platforms || []).filter((platform) =>
      platform &&
      platform.index > 0 &&
      Number(platform.w || 0) >= 640 &&
      !isSlopePlatform(platform) &&
      getPlatformVisualKind(platform) !== 'connector'
    );
    const spawnPlatformIndices = Array.from(new Set((spawnPoints || [])
      .map((point) => Math.floor(Number(point && point.platformIndex)))
      .filter((index) => Number.isInteger(index) && index >= 0)));
    const trainingPlatformIndices = broadPlatforms.map((platform) => platform.index);
    const routeStartIndex = spawnPlatformIndices[0] != null
      ? spawnPlatformIndices[0]
      : trainingPlatformIndices[0] != null ? trainingPlatformIndices[0] : 0;
    const reachable = getReachablePlatformIndices(platformGraph, routeStartIndex);
    const stronglyConnected = spawnPlatformIndices.length > 0 && spawnPlatformIndices.every((index) =>
      reachable.has(index) && getReachablePlatformIndices(platformGraph, index).has(routeStartIndex)
    );
    const unreachableTrainingPlatforms = trainingPlatformIndices.filter((index) => !reachable.has(index));
    const deadEndPlatforms = broadPlatforms.filter((platform) =>
      (platformGraph[platform.index] || []).filter((link) => link && link.to !== platform.index).length < 2
    );
    const linkCounts = summarizePlatformGraphLinks(platformGraph);
    const movementLinkCount = Math.max(1, countGraphLinks(linkCounts, ['walk', 'jump', 'drop', 'ramp-up', 'ramp-down', 'ladder-up', 'ladder-down']));
    const ladderLinkCount = countGraphLinks(linkCounts, ['ladder-up', 'ladder-down']);
    const rampLinkCount = countGraphLinks(linkCounts, ['ramp-up', 'ramp-down']);
    const verticalLinkCount = ladderLinkCount + rampLinkCount;
    const reachableBroadTiers = new Set(broadPlatforms
      .filter((platform) => reachable.has(platform.index))
      .map((platform) => Math.round(Number(platform.y || 0) / 24) * 24));
    const authoredBroadTiers = new Set(broadPlatforms
      .map((platform) => Math.round(Number(platform.y || 0) / 24) * 24));
    const requiredReachableTierCount = Math.min(map && map.isDungeon ? 2 : 3, authoredBroadTiers.size);
    const minCombatLaneWidth = broadPlatforms.length
      ? Math.min(...broadPlatforms.map((platform) => Number(platform.w || 0)))
      : 0;
    const spawnCoverage = broadPlatforms.length
      ? spawnPlatformIndices.filter((index) => trainingPlatformIndices.includes(index)).length / broadPlatforms.length
      : 0;
    const enemyDensity = broadPlatforms.length ? Number(map && map.waveMax || 0) / broadPlatforms.length : 0;
    const spawnDensityPer1000px = Number(platforms && platforms[0] && platforms[0].w || 0)
      ? (spawnPoints || []).length / (Number(platforms[0].w || 0) / 1000)
      : 0;
    const checks = Object.freeze({
      goodEnemyDensity: combatMap && enemyDensity >= (map && map.isDungeon ? 1.2 : 2),
      sensibleSpawnPlacement: combatMap && spawnCoverage >= 0.75 && (spawnPoints || []).every((point) => trainingPlatformIndices.includes(point.platformIndex)),
      loopableMovement: combatMap && stronglyConnected,
      reasonableVerticalTravel: combatMap &&
        verticalLinkCount >= (map && map.isDungeon ? 4 : 6) &&
        reachableBroadTiers.size >= requiredReachableTierCount,
      noUnreachablePlatforms: combatMap && unreachableTrainingPlatforms.length === 0,
      noAwkwardDeadEnds: combatMap && deadEndPlatforms.length === 0,
      noCrampedCombatLanes: combatMap && minCombatLaneWidth >= 640,
      limitedClimbRampDependence: combatMap && ladderLinkCount / movementLinkCount <= 0.35 && rampLinkCount / movementLinkCount <= 0.35,
      lowDowntimeSpawns: combatMap && Number(map && map.waveDelay || 0) <= 8 && spawnDensityPer1000px >= 1
    });
    const viable = combatMap && Object.values(checks).every(Boolean);
    return Object.freeze({
      id: `${map && map.id || 'map'}_training_route`,
      kind: combatMap ? map && map.isDungeon ? 'dungeon-training-loop' : 'field-training-loop' : 'service-hub',
      viable,
      loopable: !!stronglyConnected,
      routePlatformIds: Object.freeze(trainingPlatformIndices.map((index) => platforms[index] && platforms[index].id || '').filter(Boolean)),
      spawnPlatformIds: Object.freeze(spawnPlatformIndices.map((index) => platforms[index] && platforms[index].id || '').filter(Boolean)),
      platformCoverage: Number(spawnCoverage.toFixed(3)),
      enemyDensity: Number(enemyDensity.toFixed(3)),
      spawnDensityPer1000px: Number(spawnDensityPer1000px.toFixed(3)),
      minCombatLaneWidth: Math.round(minCombatLaneWidth),
      reachableTierCount: reachableBroadTiers.size,
      requiredReachableTierCount,
      traversalMix: Object.freeze({
        links: linkCounts,
        ladderDependence: Number((ladderLinkCount / movementLinkCount).toFixed(3)),
        rampDependence: Number((rampLinkCount / movementLinkCount).toFixed(3))
      }),
      checks,
      issues: Object.freeze(Object.entries(checks).filter((entry) => !entry[1]).map((entry) => entry[0]))
    });
  }

  function alignSpawnPoint(rawPoint, index, mapId, platforms, options) {
    const settings = options || {};
    const point = rawPoint || {};
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const platformIndex = clamp(Number(point.platformIndex || 0), 0, Math.max(0, sourcePlatforms.length - 1));
    const platform = sourcePlatforms[platformIndex] || sourcePlatforms[0] || {
      x: 0,
      y: Number(settings.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640),
      w: 2000,
      h: 80,
      id: `${mapId}_platform_0`
    };
    const x = clamp(Number(point.x || platform.x + platform.w / 2), platform.x + 20, platform.x + platform.w - 20);
    return {
      id: point.id || `${mapId}_spawn_${index}`,
      x,
      y: getPlatformSurfaceY(platform, x),
      platformIndex,
      platformId: platform.id,
      sectionId: normalizeId(point.sectionId),
      sectionLabel: String(point.sectionLabel || ''),
      weight: Math.max(1, Number(point.weight || 1))
    };
  }

  function createRuntimeSpawnGroups(map, platforms, spawnPoints) {
    const sourceGroups = Array.isArray(map && map.spawnGroups) ? map.spawnGroups : [];
    const runtimePlatforms = Array.isArray(platforms) ? platforms : [];
    const runtimePoints = Array.isArray(spawnPoints) ? spawnPoints : [];
    const platformById = new Map(runtimePlatforms.map((platform) => [platform.id, platform]));
    const seenIds = new Set();
    return Object.freeze(sourceGroups.map((rawGroup, index) => {
      const source = rawGroup && typeof rawGroup === 'object' ? rawGroup : {};
      let id = normalizeId(source.id) || `${map && map.id || 'map'}_spawn_group_${index + 1}`;
      if (seenIds.has(id)) id = `${id}_${index + 1}`;
      seenIds.add(id);
      const platformIds = Array.from(new Set((source.platformIds || [])
        .map(normalizeId)
        .filter((platformId) => platformId && platformById.has(platformId))));
      const platformIndices = platformIds
        .map((platformId) => platformById.get(platformId))
        .filter(Boolean)
        .map((platform) => platform.index);
      const platformIndexSet = new Set(platformIndices);
      const spawnPointIds = runtimePoints
        .filter((point) => point && platformIndexSet.has(point.platformIndex))
        .map((point) => point.id);
      const enemyWeights = Object.freeze((source.enemyWeights || source.enemies || [])
        .map((entry) => Object.freeze({
          enemyId: normalizeId(entry && typeof entry === 'object' ? entry.enemyId || entry.id : entry),
          weight: Math.max(0, Number(entry && typeof entry === 'object' ? entry.weight : 1) || 0)
        }))
        .filter((entry) => entry.enemyId && entry.weight > 0));
      if (!platformIds.length || !enemyWeights.length) return null;
      const population = Math.max(1, Math.floor(Number(source.population || 0)) || 1);
      const traversal = source.actorTraversal && typeof source.actorTraversal === 'object' ? source.actorTraversal : {};
      return Object.freeze({
        id,
        label: String(source.label || `Spawn Group ${index + 1}`),
        sectionId: normalizeId(source.sectionId),
        platformIds: Object.freeze(platformIds),
        platformIndices: Object.freeze(platformIndices),
        spawnPointIds: Object.freeze(spawnPointIds),
        enemyWeights,
        population,
        respawnSeconds: Math.max(1, Math.min(60, Number(source.respawnSeconds || map.waveDelay || 5) || 5)),
        leash: Math.max(90, Math.min(2400, Number(source.leash || 480) || 480)),
        partyScaling: normalizeId(source.partyScaling) || 'none',
        maxPopulation: Math.max(population, Math.floor(Number(source.maxPopulation || 0)) || Math.ceil(population * 1.5)),
        partyBonusPerMember: Math.max(0, Math.min(4, Number(source.partyBonusPerMember == null ? 1 : source.partyBonusPerMember) || 0)),
        actorTraversal: Object.freeze({
          mode: normalizeId(traversal.mode) || 'ground',
          allowLadders: !!traversal.allowLadders,
          allowRamps: traversal.allowRamps !== false,
          stayInTerritory: traversal.stayInTerritory !== false
        })
      });
    }).filter(Boolean));
  }

  function alignStation(rawStation, index, mapId, platforms, options) {
    const settings = options || {};
    const station = rawStation || {};
    const sourcePlatforms = Array.isArray(platforms) ? platforms : [];
    const platformIndex = clamp(Number(station.platformIndex || 0), 0, Math.max(0, sourcePlatforms.length - 1));
    const stationPlatform = sourcePlatforms[platformIndex] || sourcePlatforms[0] || { y: Number(settings.playfieldHeight || EngineViewport.PLAYFIELD_HEIGHT || 640) };
    const x = clamp(Number(station.x || stationPlatform.x + stationPlatform.w / 2), stationPlatform.x + 12, stationPlatform.x + stationPlatform.w - 100);
    const y = getPlatformSurfaceY(stationPlatform, x + 44) - 44;
    return {
      id: station.id,
      name: station.name,
      asset: station.asset || '',
      x,
      y,
      w: 88,
      h: 56,
      platformIndex,
      platformId: stationPlatform.id || '',
      serviceTier: normalizeId(station.serviceTier),
      serviceRole: normalizeId(station.serviceRole),
      serviceSummary: String(station.serviceSummary || '')
    };
  }

  function createMapRuntime(mapId, viewport, options) {
    const settings = options || {};
    const maps = Array.isArray(settings.maps) ? settings.maps : [];
    const map = mapId && typeof mapId === 'object' ? mapId : getById(maps, mapId) || maps[0] || {};
    const metrics = viewport && typeof viewport === 'object' ? viewport : createViewportMetrics();
    const platformLayout = createRuntimePlatformLayout(map, metrics, {
      authoredGroundY: settings.authoredGroundY || DEFAULT_AUTHORED_GROUND_Y
    });
    const authoredWorldHeight = platformLayout.authoredWorldHeight;
    const geometryYOffset = platformLayout.geometryYOffset;
    const worldWidth = platformLayout.worldWidth;
    const platforms = platformLayout.platforms;
    const rampConnections = createRuntimeRampConnections(map, platforms);
    const lowestPlatformBottom = platforms.reduce((bottom, platform) => Math.max(bottom, getPlatformBottomY(platform)), 0);
    const worldHeight = Math.max(metrics.worldHeight, authoredWorldHeight, lowestPlatformBottom + metrics.solidPlatformHeight);
    const climbables = (map.climbables || []).map((climbable, index) => alignClimbable(Object.assign({}, climbable, {
      y: Number(climbable.y || 0) + geometryYOffset
    }), index, map.id, platforms));
    const alignPortal = typeof settings.alignPortal === 'function'
      ? settings.alignPortal
      : EnginePortals.alignPortal || function alignPortalFallback(portal, index) {
          return Object.assign({}, portal || {}, { id: portal && portal.id || `${map.id}_portal_${index}`, mapId: map.id });
        };
    const distributeOverlappingPortals = typeof settings.distributeOverlappingPortals === 'function'
      ? settings.distributeOverlappingPortals
      : EnginePortals.distributeOverlappingPortals || function distributeOverlappingPortalsFallback(portals) {
          return portals || [];
        };
    const portals = distributeOverlappingPortals((map.portals || []).map((portal, index) => alignPortal(portal, index, map.id, platforms)), platforms, settings.portalOptions);
    const createMapHuntNpcDefinition = typeof settings.createMapHuntNpcDefinition === 'function'
      ? settings.createMapHuntNpcDefinition
      : EngineQuestNpcs.createMapHuntNpcDefinition || function createMapHuntNpcDefinitionFallback(sourceMap) {
          const palette = Array.isArray(sourceMap && sourceMap.palette) ? sourceMap.palette : [];
          return {
            id: `${sourceMap && sourceMap.id || 'map'}_hunt_warden`,
            name: `${sourceMap && sourceMap.name || 'Map'} Warden`,
            x: 320,
            platformIndex: 0,
            questIds: [],
            color: palette[0] || '#5e7d9f',
            accent: palette[2] || palette[1] || '#ffd166'
          };
        };
    const alignQuestNpc = typeof settings.alignQuestNpc === 'function'
      ? settings.alignQuestNpc
      : EngineQuestNpcs.alignQuestNpc || function alignQuestNpcFallback(npc, index) {
          return Object.assign({}, npc || {}, { id: npc && npc.id || `${map.id}_quest_npc_${index}`, mapId: map.id });
        };
    const questNpcDefinitions = (map.questNpcs || []).slice();
    if (!map.safeZone && !questNpcDefinitions.length) questNpcDefinitions.push(createMapHuntNpcDefinition(map));
    const questNpcs = questNpcDefinitions.map((npc, index) => alignQuestNpc(npc, index, map.id, platforms));
    const platformGraph = createPlatformGraph(platforms, climbables, rampConnections, {
      maxGap: settings.maxGap || settings.platformLinkMaxGap || DEFAULT_PLATFORM_LINK_MAX_GAP,
      maxJump: settings.maxJump || settings.platformLinkMaxJump || DEFAULT_PLATFORM_LINK_MAX_JUMP,
      maxDrop: settings.maxDrop || settings.platformLinkMaxDrop || DEFAULT_PLATFORM_LINK_MAX_DROP
    });
    const spawnPoints = (map.spawnPoints || []).map((point, index) => alignSpawnPoint(point, index, map.id, platforms, {
      playfieldHeight: metrics.playfieldHeight
    }));
    const spawnGroups = createRuntimeSpawnGroups(map, platforms, spawnPoints);
    const footholds = createRuntimeFootholds(map, platforms, platformGraph, {
      playfieldHeight: metrics.playfieldHeight
    });
    const trainingRoute = createTrainingRouteContract(map, platforms, platformGraph, spawnPoints, climbables, rampConnections);
    const stations = (map.stations || []).map((station, index) => alignStation(station, index, map.id, platforms, {
      playfieldHeight: metrics.playfieldHeight
    }));
    return {
      id: map.id,
      name: map.name || '',
      asset: map.asset || '',
      palette: map.palette || [],
      environment: map.environment || null,
      townScene: map.townScene || null,
      fieldComposition: map.fieldComposition || null,
      designIntent: map.designIntent || null,
      spawnSections: map.spawnSections || [],
      arenaSkeleton: normalizeId(map.arenaSkeleton),
      arenaMechanic: String(map.arenaMechanic || ''),
      townServicePlan: map.townServicePlan || null,
      safeZone: !!map.safeZone,
      isTrialInstance: !!map.isTrialInstance,
      trialId: normalizeId(map.trialId),
      worldWidth,
      worldHeight,
      playfieldWidth: metrics.width,
      playfieldHeight: metrics.playfieldHeight,
      solidPlatformHeight: metrics.solidPlatformHeight,
      statusHudHeight: metrics.statusHudHeight,
      hudTop: metrics.hudTop,
      platforms,
      climbables,
      rampConnections,
      portals,
      questNpcs,
      platformGraph,
      footholds,
      trainingRoute,
      spawnPoints,
      spawnGroups,
      stations
    };
  }

  const api = {
    createRuntimeTerrainVisual,
    getAuthoredPlatformX,
    getAuthoredPlatformY,
    getAuthoredPlatformW,
    getAuthoredPlatformH,
    getAuthoredPlatformY2,
    getAuthoredPlatformShape,
    getAuthoredPlatformVisual,
    createRuntimePlatformLayout,
    platformHorizontalGap,
    platformTransferX,
    findSurfacePlatform,
    alignClimbable,
    createRuntimeRampConnections,
    addPlatformLink,
    createPlatformGraph,
    findPlatformRouteLink,
    isRampRouteLink,
    getPlatformSurfaceLength,
    getPlatformTierId,
    getPlatformVisualKind,
    createRuntimeFootholds,
    getReachablePlatformIndices,
    summarizePlatformGraphLinks,
    countGraphLinks,
    createTrainingRouteContract,
    alignSpawnPoint,
    createRuntimeSpawnGroups,
    alignStation,
    createMapRuntime
  };

  const modules = global.ProjectStarfallEngineModules || {};
  modules.mapRuntime = Object.assign({}, modules.mapRuntime || {}, api);
  global.ProjectStarfallEngineModules = modules;

  if (typeof module === 'object' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
