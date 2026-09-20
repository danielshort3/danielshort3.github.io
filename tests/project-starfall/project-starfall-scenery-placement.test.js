'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js');
const placement = require('../../js/games/project-starfall/engine/scenery-placement.js');
const { getPlatformSurfaceY } = require('../../js/games/project-starfall/core/geometry.js');

const engine = createProjectStarfallEngine(null, data);
const renderer = createRenderer({ PIXI: {}, data });
engine.state.player.classId = 'fighter';
engine.playAudioCue = () => true;
engine.getEnvironmentImage = () => ({});
let count = 0;
let slopes = 0;
for (const map of data.MAPS) {
  engine.changeMap(map.id, { silent: true });
  const profile = engine.getEnvironmentProfile(map);
  if (!profile || !engine.getEnvironmentAsset('props', profile)) continue;
  const visibility = engine.getEnvironmentVisibility(profile);
  assert.deepStrictEqual(engine.getMapDecorationBlockers(), renderer.getMapDecorationBlockers(engine.runtime));
  for (const layer of ['rear', 'front']) {
    const densityScale = layer === 'rear' ? Number(visibility.rearDensityScale ?? .34) : Number(visibility.frontDensityScale ?? 0);
    if (densityScale <= 0) continue;
    const pixi = renderer.buildMapSceneryPlacements({}, engine.runtime, map, profile, visibility, layer, densityScale);
    const canvas = [];
    engine.drawMapProp = (ctx, asset, image, kind, x, y, w, h, seed) => canvas.push({ kind, x, y, w, h, seed });
    engine.drawMapScenery({}, map, layer);
    assert.deepStrictEqual(canvas, pixi.map(({ kind, x, y, w, h, seed }) => ({ kind, x, y, w, h, seed })), `${map.id}/${layer}: renderer parity`);
    for (const prop of pixi) {
      const platform = engine.runtime.platforms[prop.platformIndex];
      assert(Math.abs(prop.y + prop.h - getPlatformSurfaceY(platform, prop.x + prop.w / 2)) < 1e-7, `${map.id}: feet touch surface`);
      assert(placement.isPlacementSafe(platform, prop.x, prop.y, prop.w, prop.h, engine.getMapDecorationBlockers(), visibility));
      count += 1;
      if (platform.shape === 'slope') slopes += 1;
    }
  }
}
assert.deepStrictEqual(placement.buildPlacements({ platforms: [{ x: 0, y: 100, w: 2000, h: 20 }] }, { id: 'empty' }, { density: 0 }, {}, 'rear', 1, () => ['grass'], () => ({ w: 20, h: 20 })), [], 'zero density must suppress scenery');
const ramp = { x: 0, y: 100, y2: 200, w: 200, h: 20, shape: 'slope' };
assert.strictEqual(placement.getPropFooting(ramp, 50, 60, 30, 'crate'), null, 'broad rigid props cannot bridge steep ground');
assert(placement.getPropFooting(ramp, 50, 20, 22, 'flower'), 'small rooted props can sit on slopes');
const blocker = placement.getDecorationBlockers({ platforms: [ramp], spawnPoints: [{ x: 100, platformIndex: 0 }] })[0];
assert(blocker.y <= getPlatformSurfaceY(ramp, 66) - 72 && blocker.y + blocker.h >= getPlatformSurfaceY(ramp, 134) + 22, 'spawn clearance covers both sides of slope');
assert(count > 100 && slopes > 0, 'exercise real authored placements including slopes');
console.log(`Starfall scenery: ${data.MAPS.length} maps, ${count} props (${slopes} on slopes), matching Canvas/Pixi placement and clearance.`);
