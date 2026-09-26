'use strict';

const assert = require('assert');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

const engine = createProjectStarfallEngine(null, data);
engine.chooseClass('fighter');
engine.ensureRuntimeState();
const progress = engine.state.progress;
const accomplishments = engine.state.accomplishments;
const modifiers = engine.state.skillModifiers;
engine.updateRiftOperation();
engine.updatePetAssist(1 / 60);
engine.ensureRuntimeState();
assert.strictEqual(engine.state.progress, progress, 'frame-local rift/pet replacements preserve normalized quest collections');
assert.strictEqual(engine.state.accomplishments, accomplishments, 'unchanged accomplishment collections are not rebuilt each frame');
assert.strictEqual(engine.state.skillModifiers, modifiers, 'unchanged skill modifier collections are not rebuilt each frame');

engine.state.audio = { enabled: true, volume: 3 };
engine.state.player.mp = 1e9;
engine.state.player.mobility = { remaining: 0 };
engine.state.player.activeSkillObjects = [{ uid: 'expired', charges: 0, expiresAt: 0 }];
engine.state.party.members = [{ id: 'normalization-archer', classId: 'archer', name: 'Archer', level: 10, airRouteTargetX: 123 }];
engine.ensureRuntimeState();
assert.strictEqual(engine.state.audio.volume, 1, 'replaced domain values still pass their normalizer');
assert(engine.state.player.mp <= engine.getStats().maxMp, 'player vitals keep their temporal normalization');
assert.strictEqual(engine.state.player.mobility, null, 'exhausted mobility is still cleared');
assert.deepStrictEqual(engine.state.player.activeSkillObjects, [], 'expired skill objects are still removed');
assert.strictEqual(engine.state.party.members[0].airRouteTargetX, undefined, 'party runtime field normalization keeps existing movement semantics');
assert.strictEqual(engine.state.progress, progress);

const previousState = engine.state;
engine.state = JSON.parse(JSON.stringify(previousState));
const loadedProgress = engine.state.progress;
engine.ensureRuntimeState();
assert.notStrictEqual(engine.state.progress, loadedProgress, 'loading another state invalidates every domain cache');
assert(engine.runtimeStateShapeCacheMatches());

const inventoryEngine = createProjectStarfallEngine(null, data);
inventoryEngine.getInventoryCapacity = () => 4;
let ids = ['a', 'b'];
inventoryEngine.getInventorySlotIds = () => ids;
const arranged = ['b', '', 'a', ''];
inventoryEngine.state.inventorySlotOrder.equipment = arranged;
assert.strictEqual(inventoryEngine.reconcileInventorySlotOrder('equipment'), arranged, 'valid user slot arrangement is reused');
ids = ['a', 'b', 'c'];
assert.deepStrictEqual(inventoryEngine.reconcileInventorySlotOrder('equipment'), ['b', 'c', 'a', ''], 'new items fill the first free slot');
inventoryEngine.state.inventorySlotOrder.equipment = ['a', 'a', 'missing', ''];
assert.deepStrictEqual(inventoryEngine.reconcileInventorySlotOrder('equipment'), ['a', 'b', 'c', ''], 'duplicates and stale ids still reconcile');
inventoryEngine.getInventoryCapacity = () => 2;
ids = ['a', 'b', 'c', 'd'];
inventoryEngine.state.inventorySlotOrder.equipment = ['a', 'b', 'd', 'c'];
assert.deepStrictEqual(inventoryEngine.reconcileInventorySlotOrder('equipment'), ['a', 'b', 'c', 'd'], 'overflow entries retain the original inventory-order repair');

process.stdout.write('Starfall runtime normalization tests passed: domain reuse, save replacement, temporal cleanup, inventory mutations and overflow.\n');
