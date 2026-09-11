'use strict';

const assert = require('assert');
const path = require('path');
const sharp = require('sharp');
const Data = require('../../js/games/project-starfall/project-starfall-data.js');
const Environment = require('../../js/games/project-starfall/data/environment.js');

const ROOT = path.resolve(__dirname, '..', '..');

async function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'starfallCrossing');
  assert(map && map.safeZone, 'Starfall Crossing should remain the public safe-zone hub');
  assert.notStrictEqual(map.backgroundMode, 'panorama');
  assert.deepStrictEqual(map.palette, ['#f7d28a', '#7ec8d8', '#f8f0dc']);

  assert.strictEqual(map.platforms.length, 17,
    'the Crossing identity pass should preserve the authoritative 17-platform collision graph');
  assert.deepStrictEqual(
    map.platforms.map((platform) => [platform.x, platform.y, platform.w, platform.shape || 'flat']),
    [
      [0, 780, 3800, 'flat'],
      [220, 668, 740, 'flat'],
      [1120, 668, 741, 'flat'],
      [2248, 668, 770, 'flat'],
      [632, 540, 879, 'flat'],
      [1776, 540, 760, 'flat'],
      [2836, 540, 530, 'flat'],
      [412, 414, 682, 'flat'],
      [1416, 414, 788, 'flat'],
      [2458, 414, 634, 'flat'],
      [1100, 302, 619, 'flat'],
      [900, 772, 300, 'slope'],
      [1616, 668, 300, 'slope'],
      [1094, 414, 240, 'slope'],
      [3092, 414, 240, 'slope'],
      [1908, 348, 260, 'flat'],
      [3208, 472, 240, 'flat']
    ]
  );
  assert.deepStrictEqual(
    map.stations.map((station) => station.id),
    ['storage', 'shop', 'slots', 'upgrade', 'class', 'plinko'],
    'the Crossing identity pass should preserve every service station'
  );

  const scene = map.townScene;
  const sceneCells = scene.rearStructures.concat(scene.stationFacades).map((entry) => entry.cell);
  const landmarkLabels = new Set(scene.rearStructures.map((entry) => entry.label));
  assert(['Fracture Survey Array', 'Repair Gantry', 'Greenroot Frontier Gate']
    .every((label) => landmarkLabels.has(label)),
  'Crossing should read as a fractured observatory, repair gantry, and frontier gate');
  assert.deepStrictEqual(scene.rearStructures.map((entry) => entry.cell),
    ['fracturedObservatoryCore', 'expeditionDepot', 'frontierGate', 'lensWorkshop'],
    'Crossing should compose its four purpose-built fractured-frontier structure cells');
  assert(!sceneCells.some((cell) => ['starfallGuildHall', 'marketAwning', 'lanternArch'].includes(cell)),
    'Crossing should not reuse the legacy guild hall, market awning, or lantern arch composition');
  assert(!scene.streetProps.concat(scene.foregroundTrim)
    .some((entry) => ['grass', 'bush', 'flower'].includes(entry.kind)),
  'Crossing should not scatter floral village dressing over the frontier hub');
  const propKinds = new Set(scene.streetProps.map((entry) => entry.kind));
  assert(['sign', 'crate', 'crystal', 'glow'].every((kind) => propKinds.has(kind)),
    'Crossing should use survey markers, expedition crates, star crystal, and utility glow props');

  assert.strictEqual(map.environment.terrain, 'starfall-crossing');
  assert.strictEqual(map.environment.props, 'starfall-crossing');
  assert.strictEqual(map.environment.ramps || map.environment.terrain, 'starfall-crossing');
  assert(map.environment.propKinds.includes('flower'), 'the Crossing should reuse its original village prop palette');
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.terrain['starfall-crossing'].path,
    'img/project-starfall/environment/terrain/starfall-crossing.png'
  );
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.props['starfall-crossing'].path,
    'img/project-starfall/environment/props/starfall-crossing.png'
  );
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.ramps['starfall-crossing'].path,
    'img/project-starfall/environment/ramps/starfall-crossing.png'
  );
  assert.deepStrictEqual({
    fracturedObservatoryCore: Environment.ENVIRONMENT_STRUCTURE_CELLS.fracturedObservatoryCore,
    expeditionDepot: Environment.ENVIRONMENT_STRUCTURE_CELLS.expeditionDepot,
    lensWorkshop: Environment.ENVIRONMENT_STRUCTURE_CELLS.lensWorkshop,
    frontierGate: Environment.ENVIRONMENT_STRUCTURE_CELLS.frontierGate
  }, {
    fracturedObservatoryCore: 0,
    expeditionDepot: 6,
    lensWorkshop: 1,
    frontierGate: 7
  });

  const shopDoors = map.portals.filter((portal) => portal.shopDoor);
  assert.deepStrictEqual(shopDoors.map((portal) => ({
    id: portal.id,
    destinationMapId: portal.destinationMapId,
    x: portal.x,
    platformIndex: portal.platformIndex,
    facadeCell: portal.facadeCell
  })), [
    { id: 'starfallCrossing_weapon_shop_door', destinationMapId: 'starfallCrossingWeaponShop', x: 360, platformIndex: 1, facadeCell: 'cinderForge' },
    { id: 'starfallCrossing_armor_shop_door', destinationMapId: 'starfallCrossingArmorShop', x: 700, platformIndex: 1, facadeCell: 'rustcoilWorkshop' },
    { id: 'starfallCrossing_supply_shop_door', destinationMapId: 'starfallCrossingSupplyShop', x: 1260, platformIndex: 2, facadeCell: 'marketAwning' },
    { id: 'starfallCrossing_special_shop_door', destinationMapId: 'starfallCrossingSpecialShop', x: 1600, platformIndex: 2, facadeCell: 'astralObservatory' }
  ], 'Crossing shop doors should retain stable routes while occupying two readable service shelves');
  shopDoors.forEach((portal) => {
    const platform = map.platforms[portal.platformIndex];
    assert(portal.x >= platform.x && portal.x + 94 <= platform.x + platform.w,
      `${portal.id} should fit on its authored service platform`);
  });

  const frontierPortal = map.portals.find((portal) => portal.id === 'crossing_greenroot');
  const frontierGate = scene.rearStructures.find((entry) => entry.label === 'Greenroot Frontier Gate');
  assert(frontierPortal && frontierPortal.x === 2040 && frontierPortal.destinationMapId === 'greenrootMeadow');
  assert(frontierGate && Math.abs(frontierGate.x + frontierGate.w / 2 - (frontierPortal.x + 29)) <= 36,
    'the decorative frontier gate should frame the real Greenroot portal coordinate');

  const metadata = await sharp(path.join(ROOT, map.asset)).metadata();
  assert.strictEqual(metadata.width, 1280);
  assert.strictEqual(metadata.height, 640);
  const structures = Environment.ENVIRONMENT_STRUCTURE_ASSETS.townLandmarks;
  const structureMetadata = await sharp(path.join(ROOT, structures.path)).metadata();
  assert.strictEqual(structureMetadata.width, 1024);
  assert.strictEqual(structureMetadata.height, 512);
  assert(structureMetadata.hasAlpha, 'the restored town landmark atlas should retain transparency');
  const cellCount = structureMetadata.width / structures.cellSize * structureMetadata.height / structures.cellSize;
  sceneCells.concat(shopDoors.map((portal) => portal.facadeCell)).forEach((cell) => {
    const index = Environment.ENVIRONMENT_STRUCTURE_CELLS[cell];
    assert(Number.isInteger(index) && index >= 0 && index < cellCount,
      cell + ' should resolve inside the restored eight-cell atlas');
  });

  process.stdout.write('Project Starfall Crossing identity tests passed.\n');
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exit(1);
});
