'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');

const ROOT = path.resolve(__dirname, '..');

async function main() {
  const map = Data.MAPS.find((candidate) => candidate.id === 'greenrootMeadow');
  assert(map, 'Starfall Verge should remain addressable by the stable greenrootMeadow id');
  assert.strictEqual(map.name, 'Starfall Verge');
  assert.deepStrictEqual(Array.from(new Set(map.enemies)).sort(), ['faultSkitter', 'glassback', 'riftLantern']);
  assert.deepStrictEqual(
    map.spawnSections.map((section) => section.label),
    ['Arrival Shelf', 'Glass Basin', 'Fractured Bridge', 'Beacon Approach']
  );
  assert.strictEqual(map.designIntent.implementationStatus, 'fractured-frontier-v1');
  assert.strictEqual(map.designIntent.priorityRedesign, false);
  assert.strictEqual(map.environment.terrain, 'greenroot-meadow');
  assert.strictEqual(map.environment.props, 'greenroot-meadow');
  assert.strictEqual(map.environment.ramps, 'greenroot-meadow');
  assert.strictEqual(map.environment.tint, '#66788b');

  const ramps = map.platforms.filter((platform) => platform.shape === 'slope');
  const solidWidths = map.platforms
    .filter((platform, index) => index > 0 && platform.shape !== 'slope')
    .map((platform) => platform.w);
  assert.strictEqual(ramps.length, 3, 'the opening route should use exactly three authored transitions');
  assert(new Set(solidWidths).size >= 6, 'the opening route should not read as repeated equal-width lane clusters');
  assert(map.climbables.length >= 5, 'the reward pockets should retain deliberate vertical access');

  ['glassback', 'riftLantern', 'faultSkitter'].forEach((enemyId) => {
    const enemy = Data.ENEMIES.find((candidate) => candidate.id === enemyId);
    assert(enemy, `${enemyId} should exist as a real enemy definition`);
    assert(enemy.levelRange[0] <= 2 && enemy.levelRange[1] >= 6,
      `${enemyId} should cover the opening level range`);
    assert(enemy.dropPool && enemy.dropPool.materials.length,
      `${enemyId} should award a real material pool`);
    assert(enemy.dropPool.cards.length && enemy.dropPool.cards.every((entry) => entry.groupId === 'forest'),
      `${enemyId} should use the established early-game card pool`);
    assert(!enemy.dropPool.materials.some((entry) => ['blueStarCard', 'purpleStarCard', 'orangeStarCard'].includes(entry.materialId)),
      `${enemyId} should not inject mid- or late-game Star Cards into the starter economy`);
  });
  assert.strictEqual(Data.ENEMIES.find((enemy) => enemy.id === 'glassback').behavior, 'charger');
  assert.strictEqual(Data.ENEMIES.find((enemy) => enemy.id === 'riftLantern').behavior, 'flyer');
  assert.strictEqual(Data.ENEMIES.find((enemy) => enemy.id === 'faultSkitter').behavior, 'melee');
  ['starGlassChip', 'lanternCore'].forEach((itemId) => {
    assert(Data.MATERIAL_ITEMS.find((item) => item.id === itemId), `${itemId} should be a registered inventory item`);
  });

  const starterQuest = Data.QUESTS.find((quest) => quest.id === 'first_steps');
  const sampleQuest = Data.QUESTS.find((quest) => quest.id === 'greenroot_samples');
  assert(starterQuest.objectives.some((objective) => objective.enemyId === 'glassback'));
  assert(!starterQuest.objectives.some((objective) => objective.enemyId === 'slimelet'));
  assert(!/read an enemy tell|break/i.test(starterQuest.summary),
    'First Expedition copy should describe only tracked travel, defeat, and loot progress');
  assert(sampleQuest.objectives.some((objective) => objective.materialId === 'starGlassChip'));
  assert.strictEqual(
    sampleQuest.objectives.find((objective) => objective.id === 'defeat_glassbacks').label,
    'Defeat 4 Glassbacks'
  );

  const rendererCode = fs.readFileSync(path.join(ROOT,
    'js/games/project-starfall/project-starfall-renderer-pixi.js'), 'utf8');
  assert(rendererCode.includes('renderFracturedFrontierEnemy'));
  assert(rendererCode.includes('renderStarstonePortal'));

  const backgroundPath = path.join(ROOT, 'img/project-starfall/maps/greenroot-meadow.webp');
  const metadata = await sharp(backgroundPath).metadata();
  assert.strictEqual(metadata.width, 1280);
  assert.strictEqual(metadata.height, 640);

  process.stdout.write('Project Starfall fractured-frontier tests passed.\n');
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exit(1);
});
