const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

function platformShape(platform) {
  return Array.isArray(platform) ? 'flat' : String(platform && platform.shape || 'flat');
}

function platformX(platform) {
  return Array.isArray(platform) ? Number(platform[0] || 0) : Number(platform && platform.x || 0);
}

function platformY(platform) {
  return Array.isArray(platform) ? Number(platform[1] || 0) : Number(platform && platform.y || 0);
}

function platformY2(platform) {
  if (Array.isArray(platform)) return platformY(platform);
  const value = Number(platform && platform.y2);
  return Number.isFinite(value) ? value : platformY(platform);
}

function platformW(platform) {
  return Array.isArray(platform) ? Number(platform[2] || 0) : Number(platform && platform.w || 0);
}

function platformRight(platform) {
  return platformX(platform) + platformW(platform);
}

function platformSurfaceY(platform, x) {
  const y = platformY(platform);
  const y2 = platformY2(platform);
  if (platformShape(platform) !== 'slope' || Math.abs(y2 - y) < 1) return y;
  const ratio = Math.max(0, Math.min(1, (Number(x || 0) - platformX(platform)) / Math.max(1, platformW(platform))));
  return y + (y2 - y) * ratio;
}

function platformVisualKind(platform) {
  if (platformShape(platform) === 'slope') return 'slope';
  return String(platform && !Array.isArray(platform) && platform.terrainVisual && platform.terrainVisual.kind || 'flat');
}

function getMapSlopeBudget(map) {
  const role = String(map && map.layoutRole || '');
  if (map && map.shopInterior) return { maxSlopes: 0, maxGrade: 0, maxSlopesPerWindow: 0 };
  if (map && map.safeZone || role === 'town') return { maxSlopes: 4, maxGrade: 0.7, maxSlopesPerWindow: 3 };
  if (map && map.bossRoom || role === 'bossArena') return { maxSlopes: 4, maxGrade: 0.72, maxSlopesPerWindow: 3 };
  if (map && map.isDungeon || role === 'dungeon') return { maxSlopes: 4, maxGrade: 0.72, maxSlopesPerWindow: 3 };
  if (role === 'starterField') return { maxSlopes: 6, maxGrade: 0.72, maxSlopesPerWindow: 3 };
  if (role === 'endlessField') return { maxSlopes: 8, maxGrade: 0.72, maxSlopesPerWindow: 3 };
  if (role === 'deepField') return { maxSlopes: 8, maxGrade: 0.72, maxSlopesPerWindow: 3 };
  return { maxSlopes: 8, maxGrade: 0.72, maxSlopesPerWindow: 3 };
}

function countSlopesInWindow(slopes, windowWidth) {
  let maxCount = 0;
  const sorted = slopes.slice().sort((a, b) => platformX(a.platform) - platformX(b.platform));
  sorted.forEach((entry, startIndex) => {
    const startX = platformX(entry.platform);
    let count = 0;
    for (let index = startIndex; index < sorted.length; index += 1) {
      if (platformX(sorted[index].platform) - startX > windowWidth) break;
      count += 1;
    }
    maxCount = Math.max(maxCount, count);
  });
  return maxCount;
}

function validateMap(map) {
  const issues = [];
  const warnings = [];
  const platforms = Array.isArray(map && map.platforms) ? map.platforms : [];
  const slopes = platforms
    .map((platform, index) => ({ platform, index }))
    .filter((entry) => platformShape(entry.platform) === 'slope');
  const broadFlats = platforms.filter((platform) => platformShape(platform) !== 'slope' && platformW(platform) >= 640);
  const budget = getMapSlopeBudget(map);
  const rampConnections = Array.isArray(map && map.rampConnections) ? map.rampConnections : [];
  const rampedSlopeIndexes = new Set(rampConnections.map((connection) => Number(connection.rampPlatformIndex)));
  const maxGrade = slopes.reduce((max, entry) => {
    const rise = Math.abs(platformY2(entry.platform) - platformY(entry.platform));
    return Math.max(max, rise / Math.max(1, platformW(entry.platform)));
  }, 0);
  const maxSlopesInWindow = countSlopesInWindow(slopes, 1200);
  const worldWidth = platforms.length ? platformW(platforms[0]) : 0;
  const fieldComposition = map && map.fieldComposition || {};
  const routeSections = Array.isArray(fieldComposition.routeSections) ? fieldComposition.routeSections : [];
  const landmarkBands = Array.isArray(fieldComposition.landmarkBands) ? fieldComposition.landmarkBands : [];
  const runtimeBoundsValidated = !!(map && map.designIntent && map.designIntent.runtimeBoundsValidated);
  const severeOverlaps = [];
  for (let leftIndex = 1; leftIndex < platforms.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < platforms.length; rightIndex += 1) {
      const leftPlatform = platforms[leftIndex];
      const rightPlatform = platforms[rightIndex];
      if (platformShape(leftPlatform) !== 'slope' && platformShape(rightPlatform) !== 'slope') continue;
      const slope = platformShape(leftPlatform) === 'slope' ? leftPlatform : rightPlatform;
      const flat = platformShape(leftPlatform) === 'slope' ? rightPlatform : leftPlatform;
      if (['connector', 'hop'].includes(platformVisualKind(flat))) continue;
      const overlapLeft = Math.max(platformX(flat), platformX(slope));
      const overlapRight = Math.min(platformRight(flat), platformRight(slope));
      if (overlapRight - overlapLeft < 260) continue;
      const middle = (overlapLeft + overlapRight) / 2;
      if (Math.abs(platformY(flat) - platformSurfaceY(slope, middle)) <= 40) {
        severeOverlaps.push(Math.round(middle));
      }
    }
  }
  const platformIds = platforms.map((platform) => String(platform && !Array.isArray(platform) && platform.id || ''));

  if (!map || !map.id) issues.push('Map is missing an id.');
  if (!platforms.length) issues.push(`${map.id} has no platforms.`);
  if (!map.shopInterior && !map.layoutRole) issues.push(`${map.id} is missing layoutRole metadata.`);
  if (!['authored', 'generated'].includes(String(map && map.geometryMode || ''))) {
    issues.push(`${map.id} is missing an explicit authored/generated geometryMode.`);
  }
  if (platformIds.some((id) => !id) || new Set(platformIds).size !== platformIds.length) {
    issues.push(`${map.id} platform ids must be present and unique.`);
  }
  routeSections.forEach((section) => {
    const left = Number(section && section.x || 0);
    const width = Number(section && section.w || 0);
    if (left < 0 || width <= 0 || left + width > worldWidth) {
      issues.push(`${map.id} route section ${section && section.label || '(unnamed)'} exceeds the ${worldWidth}px runtime width.`);
    }
  });
  landmarkBands.forEach((landmark) => {
    const left = Number(landmark && landmark.x || 0);
    const width = Number(landmark && landmark.w || 0);
    if (left < 0 || width <= 0 || left + width > worldWidth) {
      issues.push(`${map.id} landmark ${landmark && landmark.label || '(unnamed)'} exceeds the ${worldWidth}px runtime width.`);
    }
  });
  if (runtimeBoundsValidated) {
    if (!routeSections.length) {
      issues.push(`${map.id} opts into runtime-bound validation without authored route sections.`);
    } else {
      let expectedX = 0;
      routeSections.slice().sort((left, right) => Number(left.x || 0) - Number(right.x || 0)).forEach((section) => {
        const sectionX = Number(section && section.x || 0);
        if (sectionX !== expectedX) {
          issues.push(`${map.id} route section ${section && section.label || '(unnamed)'} begins at ${sectionX}; expected ${expectedX}.`);
        }
        expectedX = sectionX + Number(section && section.w || 0);
      });
      if (expectedX !== worldWidth) {
        issues.push(`${map.id} route sections end at ${expectedX}; runtime width is ${worldWidth}.`);
      }
    }
    platforms.forEach((platform, index) => {
      if (platformX(platform) < 0 || platformW(platform) <= 0 || platformRight(platform) > worldWidth) {
        issues.push(`${map.id} platform ${index} exceeds the ${worldWidth}px runtime width.`);
      }
    });
    landmarkBands.forEach((landmark) => {
      const left = Number(landmark && landmark.x || 0);
      const right = left + Number(landmark && landmark.w || 0);
      if (!routeSections.some((section) => left >= Number(section.x || 0) && right <= Number(section.x || 0) + Number(section.w || 0))) {
        issues.push(`${map.id} landmark ${landmark && landmark.label || '(unnamed)'} crosses an authored route-section boundary.`);
      }
    });
    (map.portals || []).forEach((portal) => {
      const x = Number(portal && portal.x || 0);
      const platform = platforms[Number(portal && portal.platformIndex)];
      if (x < 0 || x > worldWidth) {
        issues.push(`${map.id} portal ${portal && portal.id || '(unnamed)'} is outside the ${worldWidth}px runtime width.`);
      }
      if (!platform) {
        issues.push(`${map.id} portal ${portal && portal.id || '(unnamed)'} references missing platform ${portal && portal.platformIndex}.`);
      } else if (x < platformX(platform) || x > platformRight(platform)) {
        issues.push(`${map.id} portal ${portal && portal.id || '(unnamed)'} is not positioned on platform ${portal.platformIndex}.`);
      }
    });
    (map.climbables || []).forEach((climbable) => {
      const left = Number(climbable && climbable.x || 0);
      const right = left + Number(climbable && climbable.w || 0);
      if (left < 0 || right > worldWidth) {
        issues.push(`${map.id} climbable ${climbable && climbable.id || '(unnamed)'} exceeds the ${worldWidth}px runtime width.`);
      }
    });
  }
  if (slopes.length > budget.maxSlopes) {
    issues.push(`${map.id} uses ${slopes.length} slopes; budget is ${budget.maxSlopes}.`);
  }
  if (maxGrade > budget.maxGrade) {
    issues.push(`${map.id} has slope grade ${maxGrade.toFixed(2)}; maximum is ${budget.maxGrade.toFixed(2)}.`);
  }
  if (maxSlopesInWindow > budget.maxSlopesPerWindow) {
    issues.push(`${map.id} has ${maxSlopesInWindow} slopes inside one 1200px section; maximum is ${budget.maxSlopesPerWindow}.`);
  }
  if (slopes.length && rampConnections.length !== slopes.length) {
    issues.push(`${map.id} has ${slopes.length} slope platforms but ${rampConnections.length} ramp connections.`);
  }
  slopes.forEach((entry) => {
    if (!rampedSlopeIndexes.has(entry.index)) {
      issues.push(`${map.id} slope platform ${entry.index} is missing a ramp connection.`);
    }
  });
  if (!map.shopInterior && !map.adminOnly && broadFlats.length < Math.max(2, slopes.length)) {
    issues.push(`${map.id} has ${broadFlats.length} broad flat lanes for ${slopes.length} slopes; add more flat combat/rest space.`);
  }
  severeOverlaps.forEach((x) => issues.push(`${map.id} has a severe platform/slope overlap near x=${x}.`));
  (map.spawnPoints || []).forEach((spawn) => {
    const platform = platforms[Number(spawn.platformIndex)];
    if (!platform) {
      issues.push(`${map.id} spawn at x=${spawn.x} references missing platform ${spawn.platformIndex}.`);
    } else if (platformShape(platform) === 'slope') {
      issues.push(`${map.id} spawn at x=${spawn.x} is placed on slope platform ${spawn.platformIndex}.`);
    } else if (String(spawn.platformId || '') !== String(platform.id || '')) {
      issues.push(`${map.id} spawn at x=${spawn.x} has stale platformId ${spawn.platformId || '(missing)'}.`);
    } else if (runtimeBoundsValidated && (Number(spawn.x || 0) < platformX(platform) || Number(spawn.x || 0) > platformRight(platform))) {
      issues.push(`${map.id} spawn at x=${spawn.x} is not positioned on platform ${spawn.platformIndex}.`);
    }
  });
  if (!map.shopInterior && !map.safeZone && !map.spawnPoints.length) {
    issues.push(`${map.id} has no spawn points.`);
  }
  if (slopes.length > 0 && broadFlats.length < slopes.length * 1.5) {
    warnings.push(`${map.id} has a low broad-flat-to-slope ratio (${broadFlats.length}:${slopes.length}).`);
  }

  return {
    mapId: map && map.id || '',
    name: map && map.name || '',
    role: map && map.layoutRole || '',
    style: map && map.layoutStyle || '',
    slopeCount: slopes.length,
    rampConnectionCount: rampConnections.length,
    broadFlatCount: broadFlats.length,
    maxGrade,
    maxSlopesInWindow,
    issues,
    warnings
  };
}

function validateProjectStarfallMaps(data, options = {}) {
  const maps = Array.isArray(data && data.MAPS) ? data.MAPS : [];
  const summaries = maps.map(validateMap);
  const issues = summaries.flatMap((summary) => summary.issues);
  const warnings = summaries.flatMap((summary) => summary.warnings);
  const guidePath = path.join(ROOT, 'MAP_AND_LEVEL_DESIGN_GUIDE.md');
  if (!fs.existsSync(guidePath)) {
    issues.push('MAP_AND_LEVEL_DESIGN_GUIDE.md is missing.');
  } else {
    const guide = fs.readFileSync(guidePath, 'utf8');
    [
      'Slope Budget Targets',
      'Tilemap And Collision Guidance',
      'Before-And-After Improvement Workflow',
      'Final Map Polish Checklist'
    ].forEach((heading) => {
      if (!guide.includes(heading)) issues.push(`MAP_AND_LEVEL_DESIGN_GUIDE.md is missing guide section: ${heading}.`);
    });
  }
  [
    'img/project-starfall/environment/terrain',
    'img/project-starfall/environment/ramps',
    'img/project-starfall/environment/props',
    'js/games/project-starfall/data/map-builders.js',
    'js/games/project-starfall/data/map-geometry.js',
    'js/games/project-starfall/engine/movement.js',
    'js/games/project-starfall/engine/viewport.js'
  ].forEach((relPath) => {
    if (!fs.existsSync(path.join(ROOT, relPath))) issues.push(`Required map-design path is missing: ${relPath}.`);
  });
  return {
    ok: issues.length === 0,
    summaries,
    issues,
    warnings: options.includeWarnings === false ? [] : warnings
  };
}

function printReport(result) {
  console.log('Project Starfall map design validation');
  result.summaries
    .filter((summary) => summary.slopeCount > 0 || summary.role)
    .forEach((summary) => {
      console.log([
        `- ${summary.mapId}`,
        `role=${summary.role || 'n/a'}`,
        `style=${summary.style || 'n/a'}`,
        `slopes=${summary.slopeCount}`,
        `broadFlats=${summary.broadFlatCount}`,
        `ramps=${summary.rampConnectionCount}`,
        `maxGrade=${summary.maxGrade.toFixed(2)}`
      ].join(' '));
    });
  if (result.warnings.length) {
    console.log('\nWarnings:');
    result.warnings.forEach((warning) => console.log(`- ${warning}`));
  }
  if (result.issues.length) {
    console.error('\nFailures:');
    result.issues.forEach((issue) => console.error(`- ${issue}`));
  }
}

if (require.main === module) {
  const data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
  const result = validateProjectStarfallMaps(data);
  printReport(result);
  process.exit(result.ok ? 0 : 1);
}

module.exports = {
  getMapSlopeBudget,
  validateMap,
  validateProjectStarfallMaps
};
