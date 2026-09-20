'use strict';

// Independent comparison to original PNG alpha, never the generator's output logic.
const fs = require('fs');
const path = require('path');
const assert = require('assert');
const crypto = require('crypto');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '../..');
const Data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const Masks = require(path.join(ROOT, 'js/games/project-starfall/data/enemy-hurtboxes.js'));
const Hurtboxes = require(path.join(ROOT, 'js/games/project-starfall/engine/enemy-hurtboxes.js'));
const Visuals = require(path.join(ROOT, 'js/games/project-starfall/engine/visuals.js'));
const Feedback = require(path.join(ROOT, 'js/games/project-starfall/engine/combat-feedback.js'));

function bodyFor(enemy) {
  return enemy.behavior === 'boss' ? enemy.id === 'stormbreakRoc' ? { w: 124, h: 96 } : enemy.id === 'astralArchivist' ? { w: 92, h: 112 } : { w: 110, h: 124 } : enemy.id === 'crackedMimic' ? { w: 64, h: 58 } : enemy.behavior === 'flyer' ? { w: 42, h: 42 } : { w: 46, h: 46 };
}

function transformFor(enemy, state, column, facing, reaction = false) {
  const actor = { id: enemy.id, data: enemy, ...bodyFor(enemy), x: 0, y: 0, facing };
  let box = Visuals.createEnemySpriteRenderBox(actor);
  if (reaction) box = Feedback.applyEnemyHitReactionToBox(box, Feedback.getEnemyHitReactionState(Feedback.createEnemyHitReaction({ startedAtMs: 1000, direction: facing, critical: true }), 1000));
  const animation = enemy.animation;
  const frame = { frameIndex: column, row: animation.states[state].row, frameWidth: animation.frameWidth, frameHeight: animation.frameHeight };
  // Derive source-to-world positions independently from mask origin / scale.
  // This is the Canvas/Pixi sprite draw state, not the collision helper output.
  const draw = Visuals.createAnimationFrameDrawState(frame, box.x, box.y, box.w, box.h, facing, { registration: animation.registration });
  return { actor, draw, box, scale: draw.drawWidth / frame.frameWidth };
}

function worldPoint(draw, scale, x, y) {
  return { x: draw.translateX + draw.scaleX * (draw.drawX + x * scale), y: draw.translateY + draw.drawY + y * scale };
}

async function main() {
  let pixelsCompared = 0, transformCases = 0, positiveQueries = 0, negativeQueries = 0, rejectedOldWitnesses = 0;
  // Actual historical dead-space regressions at game scale. These points are
  // inside the old terrain/combat rectangle but >20px from the visible artwork.
  const oldWitnesses = [
    { enemyId: 'lavaTick', action: 'telegraph', column: 0, facing: 1, point: { x: 45.5, y: 45.5 } },
    { enemyId: 'slimelet', action: 'move', column: 3, facing: -1, point: { x: 0.5, y: 45.5 } },
    { enemyId: 'clockworkTitan', action: 'hit', column: 0, facing: 1, point: { x: 0.5, y: 123.5 } }
  ];
  const sourceCache = new Map();
  for (const enemy of Data.ENEMIES) {
    const animation = enemy.animation;
    if (!animation?.sheet) continue;
    if (!sourceCache.has(animation.sheet)) {
      const input = fs.readFileSync(path.join(ROOT, animation.sheet));
      assert.strictEqual(crypto.createHash('sha256').update(input).digest('hex'), Masks.sheets[animation.sheet].sha256);
      sourceCache.set(animation.sheet, await sharp(input).ensureAlpha().raw().toBuffer({ resolveWithObject: true }));
    }
    const source = sourceCache.get(animation.sheet);
    for (const [action, definition] of Object.entries(animation.states)) {
      for (let column = 0; column < definition.frames; column++) {
        const frame = { row: definition.row, frameIndex: column, frameWidth: animation.frameWidth, frameHeight: animation.frameHeight };
        const neutral = transformFor(enemy, action, column, 1);
        const reference = Hurtboxes.createEnemyHurtbox(animation, frame, neutral.box, 1);
        assert(reference, `${enemy.id}/${action}/${column}: missing mask`);
        const actual = new Uint8Array(animation.frameWidth * animation.frameHeight);
        for (const rect of reference.mask.rectangles) {
          assert([rect.x, rect.y, rect.w, rect.h].every(Number.isInteger));
          assert(rect.w > 0 && rect.h > 0 && rect.x >= 0 && rect.y >= 0 && rect.x + rect.w <= animation.frameWidth && rect.y + rect.h <= animation.frameHeight);
          for (let y = rect.y; y < rect.y + rect.h; y++) for (let x = rect.x; x < rect.x + rect.w; x++) {
            assert(!actual[y * animation.frameWidth + x], `${enemy.id}: mask rectangles overlap`);
            actual[y * animation.frameWidth + x] = 1;
          }
        }
        const solid = [], empty = [];
        for (let y = 0; y < animation.frameHeight; y++) for (let x = 0; x < animation.frameWidth; x++) {
          const alpha = source.data[((definition.row * animation.frameHeight + y) * source.info.width + column * animation.frameWidth + x) * 4 + 3];
          const expected = alpha >= Masks.alphaThreshold ? 1 : 0;
          assert.strictEqual(actual[y * animation.frameWidth + x], expected, `${enemy.id}/${action}/${column} alpha mismatch ${x},${y}`);
          (expected ? solid : empty).push([x, y]);
          pixelsCompared++;
        }
        const samples = [];
        for (const [points, expected] of [[solid, true], [empty, false]]) for (let i = 0; i < 20; i++) samples.push({ point: points[Math.min(points.length - 1, Math.floor(points.length * (i + 0.5) / 20))], expected });
        for (const facing of [-1, 1]) for (const reaction of action === 'hit' ? [false, true] : [false]) {
          const transform = transformFor(enemy, action, column, facing, reaction);
          const hurtbox = Hurtboxes.createEnemyHurtbox(animation, frame, transform.box, facing);
          for (const { point: [x, y], expected } of samples) {
            const world = worldPoint(transform.draw, transform.scale, x + 0.5, y + 0.5);
            const pad = transform.scale * 0.1;
            assert.strictEqual(Hurtboxes.intersectsRect(hurtbox, { x: world.x - pad, y: world.y - pad, w: pad * 2, h: pad * 2 }), expected, `${enemy.id}/${action}/${column}: rectangle mismatch facing ${facing} reaction ${reaction} source ${x},${y}`);
            assert.strictEqual(Hurtboxes.intersectsCircle(hurtbox, world.x, world.y, pad), expected, `${enemy.id}/${action}/${column}: circle mismatch`);
            if (expected) positiveQueries += 2; else negativeQueries += 2;
          }
          const aim = Hurtboxes.getAimPoint(hurtbox);
          assert(Hurtboxes.intersectsCircle(hurtbox, aim.x, aim.y, 0), `${enemy.id}: auto aim must hit visible pixel`);
          const old = !reaction && oldWitnesses.find(value => value.enemyId === enemy.id && value.action === action && value.column === column && value.facing === facing);
          if (old) {
            assert.strictEqual(Hurtboxes.intersectsCircle(hurtbox, old.point.x, old.point.y, 1), false, `${enemy.id}: old empty-space witness still collides`);
            rejectedOldWitnesses++;
          }
          transformCases++;
        }
      }
    }
  }
  assert.strictEqual(rejectedOldWitnesses, oldWitnesses.length, 'All historical dead-space regressions must run');
  const report = { generatedAt: new Date().toISOString(), enemyIds: Data.ENEMIES.length, uniqueSheets: sourceCache.size, pixelsCompared, transformCases, positiveQueries, negativeQueries, rejectedOldWitnesses, alphaThreshold: Masks.alphaThreshold, result: 'pass', scope: 'Every sprite pixel in every registered enemy pose independently compared with decoded mask occupancy. World rectangle and circle narrow-phase queries checked at 20 visible and 20 transparent pixels per frame/facing/reaction, plus historical empty-space witnesses; actual render transform independently maps source sample positions.' };
  if (process.env.STARFALL_HITBOX_REPORT) fs.writeFileSync(path.resolve(process.env.STARFALL_HITBOX_REPORT), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}
if (require.main === module) main().catch(error => { console.error(error); process.exit(1); });
module.exports = { main };
