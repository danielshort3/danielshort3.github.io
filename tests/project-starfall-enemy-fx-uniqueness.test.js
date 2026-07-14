'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  buildEnemyFxIdentityMap,
  validateEnemyCombatFxUniqueness
} = require('../build/process-project-starfall-ai-visual-sweep.js');

const PREVIOUSLY_COLLIDING_PAIRS = Object.freeze([
  Object.freeze(['cloudcallAcolyte', 'stormboundArcher']),
  Object.freeze(['dewSlime', 'galeHarrier']),
  Object.freeze(['dustImp', 'glowcapHealer']),
  Object.freeze(['indexScribe', 'shardling'])
]);

function digestFor(enemyId) {
  const animation = Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS[enemyId];
  assert(animation && animation.sheet, `missing enemy combat FX asset for ${enemyId}`);
  return crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(ROOT, animation.sheet)))
    .digest('hex');
}

async function main() {
  const enemyIds = Object.keys(Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {});
  const identities = buildEnemyFxIdentityMap(enemyIds);
  const identityKeys = Array.from(identities.values()).map((identity) => [
    identity.hue,
    identity.saturation.toFixed(3),
    identity.brightness.toFixed(3)
  ].join('|'));

  assert.strictEqual(identities.size, enemyIds.length, 'every enemy should receive a combat FX identity');
  assert.strictEqual(new Set(identityKeys).size, enemyIds.length, 'enemy combat FX identities should be unique');
  assert.strictEqual(new Set(Array.from(identities.values()).map((identity) => identity.hue)).size, enemyIds.length,
    'enemy combat FX hues should remain unique after deterministic collision probing');
  const sortedHues = Array.from(identities.values()).map((identity) => identity.hue).sort((left, right) => left - right);
  const hueGaps = sortedHues.map((hue, index) => (sortedHues[(index + 1) % sortedHues.length] - hue + 360) % 360);
  assert(Math.min(...hueGaps) >= 6, 'enemy combat FX hues should occupy visibly separated six-degree palette slots');

  const validation = await validateEnemyCombatFxUniqueness();
  assert.strictEqual(validation.identities, enemyIds.length, 'validator should cover every enemy identity');
  assert.strictEqual(validation.outputs, enemyIds.length, 'validator should find one unique output per enemy');

  for (const [left, right] of PREVIOUSLY_COLLIDING_PAIRS) {
    assert.notStrictEqual(digestFor(left), digestFor(right), `${left} and ${right} should no longer share a combat FX sheet`);
  }

  console.log(`Validated ${enemyIds.length} unique Project Starfall enemy combat FX identities and sheets.`);
}

main().catch((error) => {
  console.error(error && error.stack || error);
  process.exitCode = 1;
});
