'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const Module = require('module');
const vm = require('vm');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const enginePath = require.resolve('../../js/games/project-starfall/project-starfall-engine.js');
const runtimePath = require.resolve('../../js/games/project-starfall/engine/map-runtime.js');
const affected = ['brambleDepths', 'gearworksVault', 'emberjawLair', 'banditAnimationLab', 'rimewardenSanctum'];

// Also exercise the standalone engine's embedded quest-NPC/runtime fallbacks.
const fallback = new Module(enginePath, module);
fallback.filename = enginePath;
fallback.paths = Module._nodeModulePaths(path.dirname(enginePath));
const localRequire = Module.createRequire(enginePath);
fallback.require = (id) => ['./engine/quest-npcs.js', './engine/map-runtime.js'].includes(id) ? {} : localRequire(id);
fallback._compile(fs.readFileSync(enginePath, 'utf8'), enginePath);

let checks = 0;
for (const createEngine of [require(enginePath).createProjectStarfallEngine, fallback.exports.createProjectStarfallEngine]) {
  const engine = createEngine(null, data);
  for (const mapId of affected) {
    engine.changeMap(mapId, { silent: true });
    const npc = engine.runtime.questNpcs[0];
    assert.equal(npc.asset, data.GENERIC_PLAYER_ASSET);
    assert(fs.existsSync(path.resolve(__dirname, '../..', npc.asset)), 'generated Warden artwork exists');
    assert.equal(npc.id, `${mapId}_hunt_warden`);
    assert.equal(npc.name, `${data.MAPS.find((map) => map.id === mapId).name} Warden`);
    assert.deepEqual(npc.questIds, []);
    assert.equal(npc.platformIndex, 0);
    assert.equal(npc.w, 38);
    assert.equal(npc.h, 68);
    checks += 8;
  }
}

// Isolate map-runtime's own last-resort NPC factory from the installed helper.
const context = { module: { exports: {} }, require: Module.createRequire(runtimePath) };
vm.runInNewContext(fs.readFileSync(runtimePath, 'utf8'), context, { filename: runtimePath });
for (const mapId of affected) {
  const map = data.MAPS.find((entry) => entry.id === mapId);
  const runtime = context.module.exports.createMapRuntime(map, null, { maps: data.MAPS });
  assert.equal(runtime.questNpcs[0].asset, data.GENERIC_PLAYER_ASSET);
  checks += 1;
}
console.log(`Project Starfall generated NPC art: ${checks} checks passed.`);
