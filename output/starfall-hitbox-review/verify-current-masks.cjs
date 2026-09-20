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
const { transformFor, worldPoint } = require('./audit-enemy-alpha.cjs');

async function main() {
  let pixelsCompared = 0, transformCases = 0, positiveQueries = 0, negativeQueries = 0, rejectedOldWitnesses = 0;
  const before = JSON.parse(fs.readFileSync(path.join(__dirname, 'enemy-alpha-audit.json'), 'utf8'));
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
          const old = before.frames.find(value => value.enemyId === enemy.id && value.action === action && value.column === column && value.facing === facing && value.reaction === reaction);
          if (old.maxEmptyRadiusLowerBoundPx >= 2) {
            assert.strictEqual(Hurtboxes.intersectsCircle(hurtbox, old.emptyWitness.x, old.emptyWitness.y, Math.min(1, old.maxEmptyRadiusLowerBoundPx / 2)), false, `${enemy.id}: old empty-space witness still collides`);
            rejectedOldWitnesses++;
          }
          transformCases++;
        }
      }
    }
  }
  const report = { generatedAt: new Date().toISOString(), enemyIds: Data.ENEMIES.length, uniqueSheets: sourceCache.size, pixelsCompared, transformCases, positiveQueries, negativeQueries, rejectedOldWitnesses, alphaThreshold: Masks.alphaThreshold, result: 'pass', scope: 'Every sprite pixel in every registered enemy pose independently compared with decoded mask occupancy. World rectangle and circle narrow-phase queries checked at 20 visible and 20 transparent pixels per frame/facing/reaction, plus every applicable old empty-space witness; actual render transform independently maps source sample positions.' };
  fs.writeFileSync(path.join(__dirname, 'current-mask-verification.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}
main().catch(error => { console.error(error); process.exit(1); });
