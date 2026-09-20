'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const Module = require('module');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const party = require('../../js/games/project-starfall/engine/party.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { prepareTrainingPlayer } = require('./project-starfall-training-harness.js');

// Exercise the standalone engine's embedded party-state fallback as well.
const enginePath = require.resolve('../../js/games/project-starfall/project-starfall-engine.js');
const fallbackModule = new Module(enginePath, module);
fallbackModule.filename = enginePath;
fallbackModule.paths = Module._nodeModulePaths(require('path').dirname(enginePath));
const localRequire = Module.createRequire(enginePath);
fallbackModule.require = (id) => id === './engine/party.js' ? {} : localRequire(id);
fallbackModule._compile(fs.readFileSync(enginePath, 'utf8'), enginePath);

let cases = 0;
const originalNow = Date.now;
let clock = 1700000000000;
Date.now = () => clock;
try {
  for (const classId of ['archer', 'mage']) {
    for (const level of [3, 6]) {
      const fresh = party.createPartyMemberState({ id: 'new-' + classId, classId, level }, 0, { data });
      assert(fresh.maxHp > 1 && fresh.hp === fresh.maxHp, 'new companions begin healthy at their class and level');
      for (const hp of [0, 1, 37]) {
        const copy = party.createPartyMemberState({ ...fresh, hp, mode: hp ? 'follow' : 'down', defeatedUntil: 1700000008 }, 0, { data });
        assert.equal(copy.hp, hp, 'party state restoration preserves explicit HP including defeated zero');
        assert.equal(copy.defeatedUntil, 1700000008, 'state restoration preserves recovery deadline');
      }
      assert.equal(party.createPartyMemberState({ ...fresh, hp: 55, maxHp: 75 }, 0, { data }).maxHp, 75, 'explicit stored maximum is preserved until runtime class normalization');
      for (const hp of [undefined, null, NaN, Infinity]) {
        const copy = party.createPartyMemberState({ ...fresh, hp }, 0, { data });
        assert.equal(copy.hp, copy.maxHp, 'missing or nonfinite HP initializes fully');
      }
      cases += 1;
    }
  }
  for (const createEngine of [createProjectStarfallEngine, fallbackModule.exports.createProjectStarfallEngine]) {
    for (const fps of [30, 60, 120]) {
      for (const level of [3, 6]) {
        clock = 1700000000000;
        const engine = createEngine(null, data);
        prepareTrainingPlayer(data, engine, 'fighter', level);
        engine.state.party.members = ['archer', 'mage'].map((classId, slot) => ({ id: 'training-' + classId, classId, level, slot }));
        engine.changeMap('greenrootMeadow', { silent: true });
        engine.enemies = [];
        const members = engine.getActivePrototypePartyMembers();
        assert(members.every((member) => member.level === level && member.hp === member.maxHp), 'real training setup preserves levels and starts companions healthy');
        assert.equal(engine.getSpawnGroupPartySize(), 3);
        const group = { population: 4, partyScaling: 'section-count', partyBonusPerMember: 1, maxPopulation: 6 };
        assert.equal(engine.getSpawnGroupPopulationTarget(group), 6);
        const member = members[0];
        engine.damagePartyMember(member, member.hp + 5, 'recovery-regression');
        const deadline = member.defeatedUntil;
        assert.equal(member.hp, 0, 'real damage defeats companion');
        assert(deadline > clock / 1000, 'defeat establishes future recovery time');
        assert.equal(engine.getSpawnGroupPartySize(), 2, 'dead member does not count toward living population');
        assert.equal(engine.getSpawnGroupPopulationTarget(group), 5);
        // A corpse below the world must also remain down until recovery.
        member.y = engine.runtime.worldHeight + 200;
        member.grounded = false;
        member.groundedPlatformId = '';
        member.groundedPlatformIndex = -1;
        const x = member.x;
        const y = member.y;
        const start = clock;
        for (let frame = 1; start + frame * 1000 / fps < deadline * 1000 - 0.001; frame += 1) {
          clock = start + frame * 1000 / fps;
          engine.normalizePartyMemberRuntime(member, 0);
          engine.updatePartyMemberAi(member, 0, 1 / fps, clock / 1000, { liveEnemies: [] });
          assert.equal(member.hp, 0, 'AI must not revive before deadline');
          assert.equal(member.mode, 'down');
          assert.equal(member.defeatedUntil, deadline);
          assert.equal(member.x, x, 'defeated companion cannot move');
          assert.equal(member.y, y, 'defeated companion cannot move');
        }
        clock = deadline * 1000;
        engine.updatePartyMemberAi(member, 0, 1 / fps, clock / 1000, { liveEnemies: [] });
        assert.equal(member.hp, member.maxHp, 'recovery restores HP at deadline');
        assert.notEqual(member.mode, 'down');
        assert.equal(member.defeatedUntil, 0, 'recovery consumes deadline');
        assert.equal(engine.getSpawnGroupPartySize(), 3);
        assert.equal(engine.getSpawnGroupPopulationTarget(group), 6);
        engine.damagePartyMember(member, 7, 'after-recovery');
        const woundedHp = member.hp;
        clock += 1000 / fps;
        engine.updatePartyMemberAi(member, 0, 1 / fps, clock / 1000, { liveEnemies: [] });
        assert.equal(member.hp, woundedHp, 'next AI update cannot repeat the recovery heal');
        cases += 1;
      }
    }
  }
} finally {
  Date.now = originalNow;
}
console.log('Starfall party recovery: ' + cases + ' creation and real knockout/recovery cases pass across 30/60/120 FPS.');
