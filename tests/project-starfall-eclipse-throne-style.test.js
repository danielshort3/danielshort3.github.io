'use strict';

const assert = require('assert');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Environment = require('../js/games/project-starfall/data/environment.js');
const EngineAssets = require('../js/games/project-starfall/engine/assets.js');
const PixiRenderer = require('../js/games/project-starfall/project-starfall-renderer-pixi.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');
const { validateMap } = require('../build/validate-project-starfall-maps.js');

function createCanvasRecorder() {
  const operations = [];
  const target = {};
  ['save', 'restore', 'beginPath', 'closePath', 'moveTo', 'lineTo', 'fill', 'stroke', 'fillRect', 'arc']
    .forEach((method) => {
      target[method] = (...args) => operations.push([method].concat(args));
    });
  return {
    context: new Proxy(target, {
      set(object, property, value) {
        operations.push(['set', String(property), value]);
        object[property] = value;
        return true;
      }
    }),
    operations
  };
}

function createPixiGraphicsRecorder() {
  const operations = [];
  const graphics = { operations };
  ['moveTo', 'lineTo', 'closePath', 'fill', 'stroke', 'circle', 'rect'].forEach((method) => {
    graphics[method] = (...args) => {
      operations.push([method].concat(args));
      return graphics;
    };
  });
  return graphics;
}

function platformFingerprint(platform) {
  return [
    platform.id,
    platform.x,
    platform.y,
    platform.y2 == null ? '' : platform.y2,
    platform.w,
    platform.h,
    platform.shape || 'flat'
  ].join(':');
}

function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'eclipseThrone');
  assert(map, 'Eclipse Throne should remain in the published map catalog');
  assert.strictEqual(Environment.ECLIPSE_OBSERVATORY_DECK_TREATMENT_ID, 'totality-observatory');
  assert.strictEqual(Data.ECLIPSE_OBSERVATORY_DECK_TREATMENT_ID, 'totality-observatory');
  assert.strictEqual(map.environment.platformTreatment, 'totality-observatory');
  assert.strictEqual(map.environment.terrain, 'eclipse-throne');
  assert.strictEqual(map.environment.props, 'eclipse-throne');
  assert.strictEqual(map.environment.ramps, 'eclipse-throne');
  assert(map.environment.terrainStyle.bodyAlpha <= 0.68);
  assert(map.environment.terrainStyle.platformBodyDepth <= 24);

  assert.strictEqual(
    map.platforms.map(platformFingerprint).join('|'),
    'eclipse_throne_ground:0:520::4600:80:flat|' +
      'eclipse_throne_slope_01:120:520:460:260:24:slope|' +
      'eclipse_throne_solid_lane_01:280:460::960:22:flat|' +
      'eclipse_throne_slope_02:340:460:300:260:24:slope|' +
      'eclipse_throne_solid_lane_02:600:300::800:22:flat|' +
      'eclipse_throne_solid_lane_03:940:172::700:22:flat|' +
      'eclipse_throne_slope_03:2220:520:460:280:24:slope|' +
      'eclipse_throne_solid_lane_04:2500:460::960:22:flat|' +
      'eclipse_throne_slope_04:2500:460:300:280:24:slope|' +
      'eclipse_throne_solid_lane_05:2680:300::800:22:flat|' +
      'eclipse_throne_solid_lane_06:2960:172::700:22:flat|' +
      'eclipse_throne_connector_01:1880:387::260:22:flat|' +
      'eclipse_throne_connector_02:1740:244::260:22:flat|' +
      'eclipse_throne_connector_03:2240:248::260:22:flat|' +
      'eclipse_throne_hop_01:2040:110::280:20:flat',
    'the renderer-only treatment must not change encounter collision geometry'
  );
  assert.deepStrictEqual(
    map.rampConnections.map((connection) => connection.rampPlatformId),
    ['eclipse_throne_slope_01', 'eclipse_throne_slope_02', 'eclipse_throne_slope_03', 'eclipse_throne_slope_04']
  );
  assert.deepStrictEqual(validateMap(map).issues, []);

  const criticalPaths = [];
  EngineAssets.collectMapEnvironmentAssetPaths(map, Data, criticalPaths);
  assert(criticalPaths.includes('img/project-starfall/environment/terrain/eclipse-throne.png'));
  assert(criticalPaths.includes('img/project-starfall/environment/props/eclipse-throne.png'));
  assert(criticalPaths.includes('img/project-starfall/environment/ramps/eclipse-throne.png'),
    'critical preloading must include ramps so Canvas never sticks on the brown fallback');

  const engine = createProjectStarfallEngine(null, Data);
  engine.runtime = {
    id: map.id,
    platforms: map.platforms,
    climbables: map.climbables || [],
    questNpcs: map.questNpcs || [],
    stations: map.stations || []
  };
  const slope = map.platforms.find((platform) => platform.shape === 'slope');
  const flat = map.platforms.find((platform) => platform.terrainVisual && platform.terrainVisual.kind === 'solidLane');
  assert(slope && flat);

  const canvasRamp = createCanvasRecorder();
  assert.strictEqual(engine.drawRampPlatformTerrain(canvasRamp.context, map, slope, 1, {}, 'test'), true);
  assert(canvasRamp.operations.some((operation) => operation[0] === 'arc'));
  assert(canvasRamp.operations.filter((operation) => operation[0] === 'stroke').length >= 3);
  const canvasFlat = createCanvasRecorder();
  assert.strictEqual(engine.drawEclipseObservatoryDeckTreatment(canvasFlat.context, map, flat, 2), true);
  assert(canvasFlat.operations.some((operation) => operation[0] === 'fillRect'));

  const ordinaryMap = Data.MAPS.find((candidate) => candidate.id !== map.id && candidate.environment);
  assert.strictEqual(
    engine.drawEclipseObservatoryDeckTreatment(createCanvasRecorder().context, ordinaryMap, ordinaryMap.platforms[0], 0),
    false
  );

  const pixi = PixiRenderer.createRenderer({ data: Data });
  pixi.mapGraphics = createPixiGraphicsRecorder();
  assert.strictEqual(pixi.drawRampPlatformTerrain({}, map, slope, 1, map.environment, {}, 'test'), true);
  assert(pixi.mapGraphics.operations.some((operation) => operation[0] === 'circle'));
  assert(pixi.mapGraphics.operations.filter((operation) => operation[0] === 'stroke').length >= 3);

  const cacheKeyA = engine.getMapStaticGeometryCacheKey(map, 'platforms', 4600, 640);
  const restyledMap = Object.assign({}, map, {
    environment: Object.assign({}, map.environment, { platformTreatment: 'cache-probe' })
  });
  const cacheKeyB = engine.getMapStaticGeometryCacheKey(restyledMap, 'platforms', 4600, 640);
  assert.notStrictEqual(cacheKeyA, cacheKeyB);
  engine.runtime.platforms = map.platforms.map((platform) => platform === slope
    ? Object.assign({}, platform, { y2: Number(platform.y2) + 1 })
    : platform);
  const cacheKeyC = engine.getMapStaticGeometryCacheKey(map, 'platforms', 4600, 640);
  assert.notStrictEqual(cacheKeyA, cacheKeyC);

  process.stdout.write('Project Starfall Eclipse Throne style tests passed.\n');
}

main();
