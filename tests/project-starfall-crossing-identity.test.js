'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const Environment = require('../js/games/project-starfall/data/environment.js');
const MapBackgrounds = require('../build/process-project-starfall-map-backgrounds.js');
const CrossingStructures = require('../build/process-project-starfall-crossing-structures.js');

const ROOT = path.resolve(__dirname, '..');

async function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'starfallCrossing');
  assert(map && map.safeZone, 'Starfall Crossing should remain the public safe-zone hub');
  assert.strictEqual(map.backgroundMode, 'panorama');
  assert.deepStrictEqual(map.palette, ['#101827', '#3d6575', '#d38b4c']);

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
  assert.strictEqual(map.environment.ramps, 'starfall-crossing');
  assert(!map.environment.propKinds.some((kind) => ['grass', 'bush', 'flower'].includes(kind)));
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.terrain['starfall-crossing'].path,
    'img/project-starfall/environment/terrain/astral-observatory.png'
  );
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.props['starfall-crossing'].path,
    'img/project-starfall/environment/props/rustcoil-outpost.png'
  );
  assert.strictEqual(
    Environment.ENVIRONMENT_ASSETS.ramps['starfall-crossing'].path,
    'img/project-starfall/environment/ramps/astral-observatory.png'
  );
  assert.deepStrictEqual({
    fracturedObservatoryCore: Environment.ENVIRONMENT_STRUCTURE_CELLS.fracturedObservatoryCore,
    expeditionDepot: Environment.ENVIRONMENT_STRUCTURE_CELLS.expeditionDepot,
    lensWorkshop: Environment.ENVIRONMENT_STRUCTURE_CELLS.lensWorkshop,
    frontierGate: Environment.ENVIRONMENT_STRUCTURE_CELLS.frontierGate
  }, {
    fracturedObservatoryCore: 8,
    expeditionDepot: 9,
    lensWorkshop: 10,
    frontierGate: 11
  });

  const shopDoors = map.portals.filter((portal) => portal.shopDoor);
  assert.deepStrictEqual(shopDoors.map((portal) => ({
    id: portal.id,
    destinationMapId: portal.destinationMapId,
    x: portal.x,
    platformIndex: portal.platformIndex,
    facadeCell: portal.facadeCell
  })), [
    { id: 'starfallCrossing_weapon_shop_door', destinationMapId: 'starfallCrossingWeaponShop', x: 360, platformIndex: 1, facadeCell: 'lensWorkshop' },
    { id: 'starfallCrossing_armor_shop_door', destinationMapId: 'starfallCrossingArmorShop', x: 700, platformIndex: 1, facadeCell: 'expeditionDepot' },
    { id: 'starfallCrossing_supply_shop_door', destinationMapId: 'starfallCrossingSupplyShop', x: 1260, platformIndex: 2, facadeCell: 'frontierGate' },
    { id: 'starfallCrossing_special_shop_door', destinationMapId: 'starfallCrossingSpecialShop', x: 1600, platformIndex: 2, facadeCell: 'fracturedObservatoryCore' }
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

  assert(MapBackgrounds.PANORAMA_OUTPUTS.has('starfall-crossing.webp'),
    'the Crossing processor output should be registered as a panorama');
  assert(MapBackgrounds.SOURCE_BACKED_MAPS.some(([outputName, sourceName]) =>
    outputName === 'starfall-crossing.webp' && sourceName === 'starfall-crossing-fractured-observatory-v1.png'));
  const backgroundPath = path.join(ROOT, map.asset);
  const metadata = await sharp(backgroundPath).metadata();
  assert.strictEqual(metadata.width, 2560);
  assert.strictEqual(metadata.height, 640);
  assert.deepStrictEqual(await CrossingStructures.validateAtlas(), {
    width: 1024,
    height: 768,
    cells: 12
  });

  const promptLedger = fs.readFileSync(path.join(ROOT, 'img/project-starfall/asset-prompts.md'), 'utf8');
  assert(promptLedger.includes('Fractured Starfront Sci-Fantasy'));
  assert(promptLedger.includes('starfall-crossing-fractured-observatory-v1.png'));
  assert(!/maple\s*story/i.test(promptLedger),
    'the forward asset ledger should describe Project Starfall directly instead of another game');

  process.stdout.write('Project Starfall Crossing identity tests passed.\n');
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exit(1);
});
