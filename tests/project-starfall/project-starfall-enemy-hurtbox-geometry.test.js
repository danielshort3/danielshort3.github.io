'use strict';

const assert = require('assert');
const { encodeRows } = require('../../build/generate-project-starfall-enemy-hurtboxes.js');
const data = require('../../js/games/project-starfall/data/enemy-hurtboxes.js');
const hurtboxes = require('../../js/games/project-starfall/engine/enemy-hurtboxes.js');
const visuals = require('../../js/games/project-starfall/engine/visuals.js');
const gameData = require('../../js/games/project-starfall/project-starfall-data.js');

assert.strictEqual(data.alphaThreshold, 64);
assert.strictEqual(Object.keys(data.sheets).length, 48, 'all 48 production enemy sheets need masks');
let coveredFrames = 0;
Object.values(data.sheets).forEach(sheet => {
  assert(/^[a-f0-9]{64}$/.test(sheet.sha256), 'each mask must identify its source pixels');
  assert.strictEqual(sheet.frameWidth, 160);
  assert.strictEqual(sheet.frameHeight, 160);
  assert.strictEqual(data.masks[sheet.maskIndex].length, sheet.rows * sheet.columns);
  coveredFrames += sheet.rows * sheet.columns;
});
assert.strictEqual(coveredFrames, 2304, 'all eight rows and six frames of every sheet need masks');
let mappedEnemyFrames = 0;
for (const enemy of gameData.ENEMIES) {
  const animation = enemy.animation;
  const renderBox = visuals.createEnemySpriteRenderBox({ id: enemy.id, data: enemy, x: 100, y: 100, w: 60, h: 60 });
  for (const state of Object.values(animation.states)) {
    for (let frameIndex = 0; frameIndex < state.frames; frameIndex += 1) {
      const frame = { ...state, frameIndex, frameWidth: animation.frameWidth, frameHeight: animation.frameHeight };
      for (const facing of [1, -1]) {
        const hurtbox = hurtboxes.createEnemyHurtbox(animation, frame, renderBox, facing);
        assert(hurtbox, `${enemy.id}: every runtime frame/facing must map to a mask`);
        const aim = hurtboxes.getAimPoint(hurtbox);
        assert(hurtboxes.intersectsRect(hurtbox, { x: aim.x - 0.001, y: aim.y - 0.001, w: 0.002, h: 0.002 }), `${enemy.id}: aim point must be visible`);
      }
      mappedEnemyFrames += 1;
    }
  }
}
assert.strictEqual(mappedEnemyFrames, gameData.ENEMIES.length * 48);

// Deliberately hollow/disconnected anatomy catches the invisible corner and
// large interior gaps that an alpha bounding box would incorrectly fill.
const width = 160;
const rows = Array.from({ length: 160 }, () => []);
for (let y = 25; y < 105; y += 1) {
  const lean = Math.floor((y - 25) / 9);
  rows[y] = [24 + lean, 42 + lean, 105 - lean, 123 - lean];
}
for (let y = 50; y < 54; y += 1) rows[y] = [];
for (let y = 110; y < 115; y += 1) {
  // Extended run counts and absolute-coordinate escapes are part of the codec.
  rows[y] = Array.from({ length: 40 }, (_, index) => 10 + index * 3);
}
const expected = new Uint8Array(160 * 160);
rows.forEach((runs, y) => {
  for (let index = 0; index < runs.length; index += 2) {
    for (let x = runs[index]; x < runs[index + 1]; x += 1) expected[y * width + x] = 1;
  }
});
const sheetPath = 'test/hollow-enemy.png';
const maskIndex = data.masks.length;
data.masks.push([encodeRows(rows)]);
data.sheets[sheetPath] = { frameWidth: 160, frameHeight: 160, columns: 1, rows: 1, maskIndex };
const animation = { sheet: sheetPath, frameWidth: 160, frameHeight: 160, registration: { originX: 80, groundY: 150, authoredBodyHeight: 140 } };
const frame = { row: 0, frameIndex: 0, frameWidth: 160, frameHeight: 160 };
const boxes = [
  { x: 202, y: 110, w: 86, h: 86 },
  { x: 197.75, y: 113.2, w: 91.5, h: 82.8 }
];
let probes = 0;
let firstMask = null;
for (const renderBox of boxes) {
  for (const facing of [1, -1]) {
    const hurtbox = hurtboxes.createEnemyHurtbox(animation, frame, renderBox, facing);
    assert(hurtbox);
    if (!firstMask) firstMask = hurtbox.mask;
    else assert.strictEqual(hurtbox.mask, firstMask, 'one source frame decodes only once across transforms');
    const draw = visuals.createAnimationFrameDrawState(frame, renderBox.x, renderBox.y, renderBox.w, renderBox.h, facing, { registration: animation.registration });
    const scale = draw.drawWidth / 160;
    const transform = (x, y) => ({
      x: draw.translateX + (draw.drawX + x * scale) * draw.scaleX,
      y: draw.translateY + (draw.drawY + y * scale) * draw.scaleY
    });
    for (let y = 0; y < 160; y += 1) {
      for (let x = 0; x < 160; x += 1) {
        const point = transform(x + 0.5, y + 0.5);
        const hit = hurtboxes.intersectsRect(hurtbox, { x: point.x - scale * 0.2, y: point.y - scale * 0.2, w: scale * 0.4, h: scale * 0.4 });
        assert.strictEqual(hit, !!expected[y * width + x], `${facing}: source pixel ${x},${y}`);
        probes += 1;
      }
    }
    const gap = transform(80, 80);
    assert(!hurtboxes.intersectsCircle(hurtbox, gap.x, gap.y, 10 * scale), 'a large hollow gap is not a circular hit');
    const body = transform(34, 30);
    assert(hurtboxes.intersectsCircle(hurtbox, body.x, body.y, 0), 'solid pixels accept an exact point contact');
    const aim = hurtboxes.getAimPoint(hurtbox);
    assert(hurtboxes.intersectsRect(hurtbox, { x: aim.x - 0.001, y: aim.y - 0.001, w: 0.002, h: 0.002 }), 'the aim point must be on visible anatomy');
    const bounds = hurtboxes.getBounds(hurtbox);
    let area = 0;
    hurtboxes.forEachRect(hurtbox, rect => {
      assert(rect.x >= bounds.x - 1e-8 && rect.y >= bounds.y - 1e-8);
      assert(rect.x + rect.w <= bounds.x + bounds.w + 1e-8 && rect.y + rect.h <= bounds.y + bounds.h + 1e-8);
      area += rect.w * rect.h;
    });
    assert(Math.abs(area - expected.reduce((sum, value) => sum + value, 0) * scale * scale) < 1e-7, 'merged rectangles preserve occupied area without filling gaps');
    assert(!hurtboxes.intersectsRect(hurtbox, { x: bounds.x - 21, y: bounds.y, w: 20, h: bounds.h }), 'a twenty-pixel empty strip cannot hit');
  }
}
assert.strictEqual(hurtboxes.createEnemyHurtbox({ ...animation, sheet: 'missing' }, frame, boxes[0], 1), null);
assert.strictEqual(hurtboxes.createEnemyHurtbox(animation, { ...frame, frameIndex: 1 }, boxes[0], 1), null);
assert.strictEqual(hurtboxes.createEnemyHurtbox(animation, { ...frame, frameWidth: 128 }, boxes[0], 1), null);
assert.strictEqual(hurtboxes.intersectsRect(null, boxes[0]), false);
assert.strictEqual(hurtboxes.intersectsCircle(null, 0, 0, 1), false);
delete data.sheets[sheetPath];
data.masks.pop();

// Loaded enemy artwork must reach Pixi before any legacy procedural substitute:
// those substitutes do not share the source pixels used by combat geometry.
const renderer = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js').createRenderer({});
for (const id of ['glassback', 'riftLantern', 'faultSkitter']) {
  for (const loaded of [true, false]) {
    const calls = [];
    const fixture = {
      frameStats: { actorFallbacks: 0 },
      renderActorSprite() { calls.push('sprite'); return loaded; },
      renderFracturedFrontierEnemy() { calls.push('procedural'); return true; },
      renderQuestMarker() {}
    };
    renderer.renderEnemies.call(fixture, {
      enemies: [{ id, hp: 10, maxHp: 10, renderBox: { x: 0, y: 0, w: 86, h: 86 } }],
      bounds: { left: -100, top: -100, right: 1000, bottom: 1000 }
    });
    assert.deepStrictEqual(calls, loaded ? ['sprite'] : ['sprite', 'procedural'], `${id}: sprite and mask artwork must agree`);
  }
}
console.log(`Project Starfall enemy hurtbox geometry passed: ${coveredFrames} production frames, ${mappedEnemyFrames} enemy mappings, ${probes} exact solid/empty probes, both facings and recoil-sized transforms.`);
