'use strict';

const assert = require('assert');
const path = require('path');
const sharp = require('sharp');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const visuals = require('../../js/games/project-starfall/engine/visuals.js');
const feedback = require('../../js/games/project-starfall/engine/combat-feedback.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');

const ROOT = path.resolve(__dirname, '../..');
const imageCache = new Map();

function createFixture(id, state = 'idle', facing = 1) {
  const engine = createProjectStarfallEngine(null, data);
  Object.assign(engine.state.player, {
    classId: 'fighter', advancedClassId: '', x: 240, y: 472, w: 40, h: 74,
    hp: 100000, maxHp: 100000, invulnerableUntil: 0, grounded: true,
    facing: 1, vx: 0, vy: 0, attackTimer: 0, combatLockUntil: 0,
    movementLockUntil: 0, climbing: false
  });
  engine.resetPlayerFeelRuntime('hurtbox-test');
  const definition = data.ENEMIES.find(entry => entry.id === id);
  assert(definition, `${id}: production enemy definition required`);
  const enemy = engine.createEnemy(definition, engine.runtime.spawnPoints[0]);
  Object.assign(enemy, {
    x: 500, y: 500, facing, vx: 0, vy: 0, hp: 100000, maxHp: 100000,
    defense: 0, level: 1, elite: false, eliteAffixIds: [], attackCd: 10,
    pendingAttack: null, attackRecovery: 0, telegraph: 0, state: 'idle',
    animationState: state, animationPhaseOffset: 0,
    animationStartedAt: Date.now() / 1000 + 3600, actionLockUntil: Infinity,
    animationDuration: 1, animationLoop: true
  });
  engine.enemies = [enemy];
  engine.projectiles = [];
  engine.effects = [];
  engine.playAudioCue = () => true;
  return { engine, enemy };
}

// Decode the actual shipped artwork independently of the hurtbox data/decoder.
// Its world transform is the renderer's public draw contract, including facing
// and hit recoil. Queries below use original opaque pixels, never mask bounds
// as an oracle for whether a hit should occur.
async function readRenderedPixels(engine, enemy) {
  const snapshot = engine.getEnemyRenderSnapshot(enemy, null);
  const frame = snapshot.animationFrame;
  const key = `${frame.sheet}:${frame.row}:${frame.frameIndex}`;
  if (!imageCache.has(key)) {
    imageCache.set(key, await sharp(path.join(ROOT, frame.sheet)).extract({
      left: frame.frameIndex * frame.frameWidth,
      top: frame.row * frame.frameHeight,
      width: frame.frameWidth,
      height: frame.frameHeight
    }).ensureAlpha().raw().toBuffer());
  }
  const pixels = imageCache.get(key);
  const box = snapshot.renderBox;
  const draw = visuals.createAnimationFrameDrawState(frame, box.x, box.y, box.w, box.h,
    snapshot.facing, { registration: snapshot.registration });
  const scaleX = draw.drawWidth / frame.frameWidth * draw.scaleX;
  const scaleY = draw.drawHeight / frame.frameHeight * draw.scaleY;
  const originX = draw.translateX + draw.drawX * draw.scaleX;
  const originY = draw.translateY + draw.drawY * draw.scaleY;
  const rectangles = [];
  for (let y = 0; y < frame.frameHeight; y += 1) {
    for (let x = 0; x < frame.frameWidth; x += 1) {
      if (pixels[(y * frame.frameWidth + x) * 4 + 3] < 64) continue;
      rectangles.push({
        x: Math.min(originX + x * scaleX, originX + (x + 1) * scaleX),
        y: Math.min(originY + y * scaleY, originY + (y + 1) * scaleY),
        w: Math.abs(scaleX), h: Math.abs(scaleY)
      });
    }
  }
  assert(rectangles.length > 100, `${enemy.id}: visible production pose required`);
  return { snapshot, rectangles };
}

function rectanglesOverlap(a, b) {
  return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
}

function pixelsIntersectRect(pixels, rect) {
  return pixels.rectangles.some(pixel => rectanglesOverlap(pixel, rect));
}

function pixelsIntersectCircle(pixels, x, y, radius) {
  return pixels.rectangles.some(pixel => {
    const dx = x - Math.max(pixel.x, Math.min(x, pixel.x + pixel.w));
    const dy = y - Math.max(pixel.y, Math.min(y, pixel.y + pixel.h));
    return dx * dx + dy * dy <= radius * radius;
  });
}

function assertVisiblePoint(pixels, point, message) {
  assert(pixelsIntersectRect(pixels, { x: point.x - 0.01, y: point.y - 0.01, w: 0.02, h: 0.02 }), message);
}

function projectileAt(point) {
  return {
    owner: 'player', type: 'arrow', x: point.x - 1, y: point.y - 1,
    w: 2, h: 2, vx: 0, vy: 0, ttl: 1, damage: 25, pierce: 0,
    homing: false, basicFxId: 'archer'
  };
}

async function testEmptySpaceAndActualCombat() {
  const { engine, enemy } = createFixture('lavaTick', 'telegraph');
  const pixels = await readRenderedPixels(engine, enemy);
  assert.strictEqual(pixels.snapshot.animationFrame.row, 2);
  assert.strictEqual(pixels.snapshot.animationFrame.frameIndex, 0);
  const empty = { x: enemy.x + 45.5, y: enemy.y + 45.5 };
  assert(rectanglesOverlap(enemy, projectileAt(empty)), 'regression probe must overlap the old generic body');
  assert(!pixelsIntersectCircle(pixels, empty.x, empty.y, 20),
    'Lava Tick fixture must have a proven 20px transparent gap in original production pixels');
  assert.strictEqual(engine.enemyIntersectsCircle(enemy, empty.x, empty.y, 20), false,
    'a 20px empty-space circle must not hit the enemy');
  assert.strictEqual(engine.getContextEnemyAtWorldPoint(empty), null,
    'click targeting must not select the old invisible corner');
  assert.deepStrictEqual(engine.findEnemiesNear(empty.x, empty.y, 20), [],
    'AoE target acquisition must reject a circle entirely inside transparent space');
  const hp = enemy.hp;
  engine.projectiles = [projectileAt(empty)];
  engine.updateProjectiles(0);
  assert.strictEqual(enemy.hp, hp, 'real projectile update must pass through transparent padding');
  assert.strictEqual(engine.projectiles.length, 1, 'missing the sprite must not consume a projectile');
  assert.strictEqual(engine.combatFeedback.hitstopRemainingMs, 0, 'a transparent-space miss must not trigger hitstop');
  engine.areaHit(empty.x, empty.y, 20, 20, { id: 'hurtbox_test', owner: 'fighter' });
  assert.strictEqual(enemy.hp, hp, 'actual area damage must not apply inside transparent padding');
  assert.strictEqual(engine.roleAreaHit(empty.x, empty.y, 20, 20, { id: 'hurtbox_test', owner: 'fighter' }), 0,
    'role AoE must use the same silhouette contact predicate');
  assert.strictEqual(enemy.hp, hp);

  const visible = engine.enemyCenter(enemy);
  assertVisiblePoint(pixels, visible, 'homing and target effects must aim at visible artwork');
  assert.strictEqual(engine.getContextEnemyAtWorldPoint(visible), enemy,
    'click targeting must select the visible body');
  assert.deepStrictEqual(engine.findEnemiesNear(visible.x, visible.y, 1), [enemy],
    'AoE acquisition must reach visible pixels above the old physics body');
  engine.projectiles = [projectileAt(visible)];
  engine.updateProjectiles(0);
  assert(enemy.hp < hp, 'actual projectile update must damage visible body pixels');
  assert.strictEqual(engine.projectiles.length, 0, 'visible contact must consume a non-piercing projectile');
  assert(engine.combatFeedback.hitstopRemainingMs > 0 && enemy.basicHitReaction,
    'visible projectile contact must retain authored hit feedback');

  const areaFixture = createFixture('lavaTick', 'telegraph');
  const areaPoint = areaFixture.engine.enemyCenter(areaFixture.enemy);
  const areaHp = areaFixture.enemy.hp;
  areaFixture.engine.areaHit(areaPoint.x, areaPoint.y, 1, 20, { id: 'hurtbox_test', owner: 'fighter' });
  assert(areaFixture.enemy.hp < areaHp, 'actual AoE must damage visible pixels outside the old physics body');
}

async function testMeleeRelease() {
  for (const shouldHit of [false, true]) {
    const { engine, enemy } = createFixture('stormbreakRoc');
    const range = engine.getStats().range;
    const hitbox = { x: enemy.x + (shouldHit ? 42 : 2) - range, y: enemy.y + enemy.h - 24, w: range, h: 48 };
    Object.assign(engine.state.player, { x: hitbox.x - 36, y: hitbox.y - 12, facing: 1 });
    const pixels = await readRenderedPixels(engine, enemy);
    assert(rectanglesOverlap(hitbox, enemy), 'both melee probes must overlap the old generic body');
    assert.strictEqual(pixelsIntersectRect(pixels, hitbox), shouldHit,
      'melee fixture must distinguish a transparent corner from a visible foot');
    const hp = enemy.hp;
    assert.strictEqual(engine.basicAttack(), true);
    assert.strictEqual(enemy.hp, hp, 'melee must still wait for the authored contact moment');
    assert.strictEqual(engine.advancePendingBasicAttack(engine.pendingPlayerAttack.releaseAt), true);
    assert.strictEqual(enemy.hp < hp, shouldHit,
      'released melee damage must match the rendered silhouette, including an old-body corner miss');
    assert.strictEqual(engine.combatFeedback.hitstopRemainingMs > 0, shouldHit,
      'melee contact feedback must follow actual visible contact');
  }
}

async function testSkillContactOrigins() {
  const { engine, enemy } = createFixture('lavaTick', 'telegraph');
  Object.assign(engine.state.player, { classId: 'mage', advancedClassId: 'runeMage', level: 25 });
  engine.state.skills = { rune_mage_rune_mark: 5, rune_mage_ground_glyph: 5 };
  const initialPixels = await readRenderedPixels(engine, enemy);
  const point = engine.enemyCenter(enemy);
  const impacts = [];
  const explosions = [];
  const pushImpact = engine.pushSkillImpactEffect.bind(engine);
  const pushArea = engine.pushSkillAreaEffect.bind(engine);
  engine.pushSkillImpactEffect = (x, y, ...rest) => {
    impacts.push({ x, y });
    return pushImpact(x, y, ...rest);
  };
  engine.pushSkillAreaEffect = (skill, x, y, ...rest) => {
    explosions.push({ x, y });
    return pushArea(skill, x, y, ...rest);
  };
  engine.effects = [{
    type: 'field', x: engine.state.player.x + 20, y: engine.state.player.y + 74,
    r: 230, ttl: 6, duration: 8, baseDuration: 8, maxDuration: 18,
    runeField: true, runeFieldProfileId: 'groundGlyph', skillId: 'rune_mage_ground_glyph'
  }];
  engine.projectiles = [{
    ...projectileAt(point), basicFxId: '', type: 'rune',
    sourceSkillId: 'rune_mage_rune_mark', skillOwner: 'runeMage', skillRank: 5,
    lineCount: 1, applyMark: true, markDuration: 9, hitEnemies: [], hitEnemyUids: {}
  }];
  engine.updateProjectiles(0);
  assert(impacts.length > 0 && explosions.length > 0, 'real Rune Mark contact must produce impact and field explosion');
  assertVisiblePoint(initialPixels, impacts[0], 'skill impact must originate on the contacted visible pose');
  enemy.animationStartedAt = Date.now() / 1000 + 3600;
  const hitPixels = await readRenderedPixels(engine, enemy);
  assertVisiblePoint(hitPixels, explosions[0], 'post-hit Rune Mark explosion must originate on the current visible pose');
}

async function testChargeContact() {
  for (const shouldHit of [false, true]) {
    const { engine, enemy } = createFixture('bristleBoar', 'attack');
    const pixels = await readRenderedPixels(engine, enemy);
    const right = Math.max(...pixels.rectangles.map(pixel => pixel.x + pixel.w));
    Object.assign(engine.state.player, {
      x: right + (shouldHit ? -5 : 20), y: enemy.y + enemy.h - 74, invulnerableUntil: 0
    });
    assert.strictEqual(pixelsIntersectRect(pixels, engine.playerHitbox()), shouldHit,
      'charge fixture must either contact the visible boar or leave 20px clearance');
    engine.getPassiveOffscreenEnemyUpdateStride = () => 1;
    engine.shouldDeferPassiveOffscreenEnemyUpdate = () => false;
    engine.updateEnemyClimbing = () => false;
    engine.updateEnemyPlatformJump = () => false;
    engine.recoverFallenBodyThroughTop = () => false;
    engine.updateEnemyStuckState = () => false;
    engine.setEnemyAggro(enemy, engine.getCombatCharacterByTarget('player', 'player'), 'test');
    assert.strictEqual(engine.beginEnemyCharge(enemy), true);
    Object.assign(enemy, { telegraph: 0, animationState: 'attack', animationStartedAt: Date.now() / 1000 + 3600, actionLockUntil: Infinity });
    const hp = engine.state.player.hp;
    engine.updateEnemies(0);
    assert.strictEqual(engine.state.player.hp < hp, shouldHit,
      'the active charge must damage only where the rendered creature touches the player');
    assert.strictEqual(enemy.state === 'charging', !shouldHit,
      'charge must stop on visible contact, without stopping in empty space');
  }
}

async function testFacingRecoilAndWorldCoordinates() {
  for (const facing of [-1, 1]) {
    const { engine, enemy } = createFixture('lavaTick', 'attack', facing);
    Object.assign(enemy, { x: 1687.25, y: 493.75 });
    enemy.basicHitReaction = feedback.createEnemyHitReaction({
      startedAtMs: performance.now() + 3600000, direction: facing, critical: true
    });
    const pixels = await readRenderedPixels(engine, enemy);
    assert(pixels.snapshot.hitReaction.active, 'the fixture must exercise the recoil/squash transform');
    const aim = engine.enemyCenter(enemy);
    assertVisiblePoint(pixels, aim, 'mirrored, recoiling enemies must retain a visible aim point at their world location');
    const probe = projectileAt(aim);
    assert.strictEqual(engine.enemyIntersectsRect(enemy, probe), true);
    assert.strictEqual(engine.enemyIntersectsCircle(enemy, aim.x, aim.y, 0.1), true);
    assert.strictEqual(engine.enemyIntersectsRect(enemy, { ...probe, x: probe.x - 1187.25 }), false,
      'collision must not accidentally stay at the original spawn or screen coordinate');
    const projectile = { ...projectileAt({ x: aim.x - 80, y: aim.y - 25 }), vx: 200, vy: 0 };
    engine.steerProjectileTowardTarget(projectile, enemy, 0.05,
      { range: 300, verticalTolerance: 100, turnRate: 4 });
    assert(projectile.vy > 0, 'homing must steer toward the visible aim point');
    const speed = Math.hypot(projectile.vx, projectile.vy);
    const dx = aim.x - (projectile.x + projectile.w / 2);
    const dy = aim.y - (projectile.y + projectile.h / 2);
    const blend = 0.2;
    const desiredVx = 200 * (1 - blend) + dx / Math.hypot(dx, dy) * 200 * blend;
    const desiredVy = dy / Math.hypot(dx, dy) * 200 * blend;
    assert(Math.abs(projectile.vy / speed - desiredVy / Math.hypot(desiredVx, desiredVy)) < 1e-10,
      'homing direction must derive from visible anatomy rather than the old physics-box center');
  }

  // Every runtime enemy, including aliases, must provide a usable solid aim
  // point. This is runtime integration coverage; exhaustive per-pixel decoding
  // and all-frame geometry validation live in the dedicated mask test.
  for (const definition of data.ENEMIES) {
    const { engine, enemy } = createFixture(definition.id);
    assert(engine.getEnemyHurtbox(enemy), `${definition.id}: production mask must be connected to the engine`);
    const pixels = await readRenderedPixels(engine, enemy);
    assertVisiblePoint(pixels, engine.enemyCenter(enemy), `${definition.id}: target aim must land on original visible artwork`);
  }
}

async function main() {
  const random = Math.random;
  try {
    Math.random = () => 0.99;
    await testEmptySpaceAndActualCombat();
    await testMeleeRelease();
    await testSkillContactOrigins();
    await testChargeContact();
    await testFacingRecoilAndWorldCoordinates();
  } finally {
    Math.random = random;
  }
  console.log('Project Starfall enemy silhouette combat tests passed (projectiles, melee, AoE, charge, targeting, facing and recoil).');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
