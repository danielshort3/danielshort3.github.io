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

  if (!map || !map.id) issues.push('Map is missing an id.');
  if (!platforms.length) issues.push(`${map.id} has no platforms.`);
  if (!map.shopInterior && !map.layoutRole) issues.push(`${map.id} is missing layoutRole metadata.`);
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
  if (!map.shopInterior && broadFlats.length < Math.max(2, slopes.length)) {
    issues.push(`${map.id} has ${broadFlats.length} broad flat lanes for ${slopes.length} slopes; add more flat combat/rest space.`);
  }
  (map.spawnPoints || []).forEach((spawn) => {
    const platform = platforms[Number(spawn.platformIndex)];
    if (!platform) {
      issues.push(`${map.id} spawn at x=${spawn.x} references missing platform ${spawn.platformIndex}.`);
    } else if (platformShape(platform) === 'slope') {
      issues.push(`${map.id} spawn at x=${spawn.x} is placed on slope platform ${spawn.platformIndex}.`);
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
