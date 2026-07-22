'use strict';

const assert = require('assert');
const data = require('../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../js/games/project-starfall/project-starfall-engine.js');

const equipmentCatalog = [].concat(
  data.SHOP_ITEMS || [],
  data.RANDOM_EQUIPMENT_ITEMS || [],
  data.BOSS_EQUIPMENT_ITEMS || []
);
const equipmentById = new Map(equipmentCatalog.map((item) => [item.id, item]));
const expectedStarterWeaponByClass = {
  fighter: 'training_sword',
  mage: 'training_wand',
  archer: 'training_bow'
};
const expectedTrainingDropWeaponByClass = {
  fighter: 'adventurer_cutlass',
  mage: 'birch_wand',
  archer: 'simple_bow'
};

const cutlass = equipmentById.get('adventurer_cutlass');
const balancedFocus = equipmentById.get('balanced_focus');
assert(cutlass && cutlass.classId === 'fighter', 'Adventurer Cutlass should be a fighter weapon');
assert(balancedFocus && balancedFocus.classId === 'mage', 'Balanced Focus should be a mage weapon');
assert.strictEqual(data.DROP_ECONOMY.equipmentLevelLeeway, 8,
  'specific monster equipment should use the shared eight-level progression window');

function createEnemyFixture(enemyId, level) {
  const enemyData = data.ENEMIES.find((enemy) => enemy.id === enemyId);
  assert(enemyData, `${enemyId} should exist for early loot gating tests`);
  return {
    id: enemyData.id,
    name: enemyData.name,
    data: enemyData,
    level
  };
}

function getEquipmentRows(engine, enemy) {
  const info = engine.getMonsterGuideDropInfo(enemy);
  const table = info.tables.find((candidate) => candidate.id === 'equipment');
  return table ? table.entries : [];
}

Object.entries(expectedStarterWeaponByClass).forEach(([classId, starterWeaponId]) => {
  const engine = createProjectStarfallEngine(null, data);
  assert.strictEqual(engine.chooseClass(classId), true, `${classId} should be selectable`);
  assert.strictEqual(engine.state.equipment.weapon.id, starterWeaponId,
    `${classId} should auto-equip its own starter weapon`);

  const trainingRows = getEquipmentRows(engine, createEnemyFixture('slimelet', 6));
  const trainingWeapons = trainingRows
    .map((row) => equipmentById.get(row.itemId))
    .filter((item) => item && item.slot === 'weapon')
    .map((item) => item.id);
  assert.deepStrictEqual(trainingWeapons, [expectedTrainingDropWeaponByClass[classId]],
    `${classId} starter monsters should offer one class-compatible weapon`);

  [2, 9].forEach((level) => {
    const riftRows = getEquipmentRows(engine, createEnemyFixture('riftLantern', level));
    riftRows.forEach((row) => {
      const item = equipmentById.get(row.itemId);
      assert(item, `Rift Lantern drop ${row.itemId} should resolve to an equipment definition`);
      assert(item.level <= level + data.DROP_ECONOMY.equipmentLevelLeeway,
        `level ${level} Rift Lantern should not offer level ${item.level} ${item.name}`);
      if (item.slot === 'weapon') {
        assert.strictEqual(item.classId, classId,
          `level ${level} Rift Lantern should not offer an off-class weapon to ${classId}`);
      }
    });
  });

  if (classId !== 'fighter') {
    const equippedBefore = engine.state.equipment.weapon.uid;
    engine.state.inventory.push(Object.assign({}, cutlass, {
      uid: `test_cutlass_${classId}`,
      stats: Object.assign({}, cutlass.stats),
      baseStats: Object.assign({}, cutlass.stats)
    }));
    assert.strictEqual(engine.equipItem(`test_cutlass_${classId}`, { noEmit: true }), false,
      `${classId} should reject an explicitly granted Adventurer Cutlass`);
    assert.strictEqual(engine.state.equipment.weapon.uid, equippedBefore,
      `${classId} should keep its ranged starter weapon after rejecting the cutlass`);
  }
});

Object.keys(expectedStarterWeaponByClass).forEach((classId) => {
  const engine = createProjectStarfallEngine(null, data);
  engine.chooseClass(classId);
  data.ENEMIES.filter((enemy) => enemy.dropPool && enemy.dropPool.equipment.length).forEach((enemyData) => {
    const levels = Array.from(new Set(enemyData.levelRange || [1]));
    levels.forEach((level) => {
      const rows = getEquipmentRows(engine, createEnemyFixture(enemyData.id, level));
      rows.forEach((row) => {
        const item = equipmentById.get(row.itemId);
        assert(item.level <= level + data.DROP_ECONOMY.equipmentLevelLeeway,
          `${enemyData.name} level ${level} should not offer level ${item.level} ${item.name}`);
        if (item.slot === 'weapon') {
          assert.strictEqual(item.classId, classId,
            `${enemyData.name} should not offer ${item.classId} weapon ${item.name} to ${classId}`);
        }
      });
    });
  });
});

const guideEngine = createProjectStarfallEngine(null, data);
guideEngine.chooseClass('mage');
const maxLevelRiftRows = getEquipmentRows(guideEngine, createEnemyFixture('riftLantern', 9));
assert.deepStrictEqual(maxLevelRiftRows.map((row) => row.itemId), ['wanderer_charm'],
  'the complete Rift Lantern level band should expose only its level-appropriate focus accessory');
assert(!maxLevelRiftRows.some((row) => ['balanced_focus', 'starglass_staff', 'channeler_gloves'].includes(row.itemId)),
  'Rift Lanterns should never leak level 18-28 focus gear into the opening field');

console.log('Project Starfall early loot gating tests passed.');
