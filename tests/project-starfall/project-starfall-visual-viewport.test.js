'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const viewport = require(path.join(ROOT, 'js/games/project-starfall/engine/viewport.js'));
const visuals = require(path.join(ROOT, 'js/games/project-starfall/engine/visuals.js'));
const animations = require(path.join(ROOT, 'js/games/project-starfall/data/animations.js'));
const shopVendors = require(path.join(ROOT, 'js/games/project-starfall/data/shop-vendors.js'));
const mapPublication = require(path.join(ROOT, 'js/games/project-starfall/data/map-publication.js'));
const animationData = animations.createAnimationData();

const largeViewport = viewport.createViewportMetrics({
  video: { viewportPreset: 'large', width: 1600, height: 930, hudScale: 1.1 }
});

assert.strictEqual(largeViewport.width, 1280, 'render width should remain at the fixed logical width');
assert.strictEqual(largeViewport.height, 806, 'render height should remain at the fixed logical height');
assert.strictEqual(largeViewport.displayWidth, 1600, 'display preference should remain available to presentation code');
assert.strictEqual(largeViewport.displayHeight, 930, 'display preference should remain available to presentation code');
assert.strictEqual(largeViewport.playfieldHeight, 674, 'logical playfield geometry should not change with a display preset');
assert.strictEqual(largeViewport.statusHudHeight, 84, 'HUD scaling should not resize world geometry');
assert.strictEqual(largeViewport.hudScale, 1.1, 'HUD content scale should remain available to presentation code');

const shopDoors = shopVendors.createTownShopDoorPortals('starfallCrossing');
assert.deepStrictEqual(
  shopDoors.map((portal) => portal.facadeCell),
  ['cinderForge', 'rustcoilWorkshop', 'marketAwning', 'astralObservatory'],
  'each Starfall Crossing shop should publish its restored structure-atlas facade'
);

const publication = mapPublication.createMapPublicationData({
  MAP_ASSETS: {},
  STATION_ASSETS: {},
  MAP_ENVIRONMENT_PROFILES: {},
  WORLD_AREAS: [],
  WORLD_MAP_NODES: [],
  MAP_PORTALS: {},
  createSpawnSections: () => [],
  attachSpawnSectionsToPoints: () => [],
  getTownServicePlan: () => null,
  createDefaultFieldComposition: () => null,
  createDesignIntent: (value) => value
});
const publishedMap = publication.attachMapAssets({
  id: 'greenrootMeadow',
  name: 'Greenroot Meadow',
  safeZone: false,
  questNpcs: [{ id: 'greenroot_guide', name: 'Greenroot Guide', questIds: ['first_steps'] }],
  platforms: [],
  spawnPoints: [],
  enemies: []
});
assert.strictEqual(
  publishedMap.questNpcs[0].asset,
  'img/project-starfall/characters/generic-player.png',
  'published quest NPCs should use polished existing character art by default'
);

const stageCss = fs.readFileSync(path.join(ROOT, 'css/games/project-starfall/stage.css'), 'utf8');
assert(stageCss.includes('--starfall-logical-stage-ratio: 1280 / 806'), 'stage CSS should retain the logical aspect ratio');
assert(stageCss.includes('.project-starfall-stage-panel:fullscreen'), 'stage CSS should support native fullscreen layout');
assert(stageCss.includes('[data-starfall-focus-mode="true"]'), 'stage CSS should support a fullscreen fallback focus mode');

const uiCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-ui.js'), 'utf8');
assert(uiCode.includes("host.setAttribute('data-starfall-focus-mode', 'true')"), 'blocked fullscreen requests should enable focus mode');
assert(uiCode.includes("this.setFocusMode(false, 'Focus mode closed.')"), 'Escape should close fallback focus mode');

const rendererCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-renderer-pixi.js'), 'utf8');
assert(rendererCode.includes("portal.facadeCell || 'marketAwning'"), 'shop portals should try atlas-backed facades first');
assert(rendererCode.includes('const npcTexture = this.getTexture(npc.asset)'), 'quest NPCs should try character art first');
assert(rendererCode.includes('graphics.rect(npc.x + 6'), 'quest NPC procedural fallback should remain available');
assert(rendererCode.includes('this.backgroundSprites, this.worldBaseBandGraphics, this.worldLayer'),
  'Pixi should fade above the authored background and behind gameplay');

const standardEnemy = { id: 'slimelet', x: 100, y: 200, w: 46, h: 46, data: { behavior: 'melee' } };
const standardEnemyBox = visuals.createEnemySpriteRenderBox(standardEnemy);
assert.strictEqual(standardEnemyBox.w, 86, 'standard enemy art should retain its established visual height');
assert.strictEqual(standardEnemyBox.w, standardEnemyBox.h, 'square enemy frames should render without aspect distortion');
assert.strictEqual(standardEnemyBox.y + standardEnemyBox.h, standardEnemy.y + standardEnemy.h - 2,
  'grounded enemy art should share the authored foot baseline');

const rocBox = visuals.createEnemySpriteRenderBox({
  id: 'stormbreakRoc', x: 0, y: 0, w: 124, h: 96, data: { behavior: 'boss' }
});
const archivistBox = visuals.createEnemySpriteRenderBox({
  id: 'astralArchivist', x: 0, y: 0, w: 92, h: 112, data: { behavior: 'boss' }
});
assert.deepStrictEqual([rocBox.w, rocBox.h], [178, 178], 'Stormbreak Roc should keep its large scale without being vertically squashed');
assert.deepStrictEqual([archivistBox.w, archivistBox.h], [154, 154], 'Astral Archivist should keep its tall scale without being horizontally squashed');

const flyer = { id: 'galeHarrier', x: 20, y: 40, w: 42, h: 42, data: { behavior: 'flyer' } };
const flyerBox = visuals.createEnemySpriteRenderBox(flyer);
assert.strictEqual(flyerBox.y + flyerBox.h / 2, flyer.y + flyer.h / 2, 'flyer art should remain centered around its body');

const enemyDrawState = visuals.createAnimationFrameDrawState(
  { frameWidth: 160, frameHeight: 160, row: 0, frameIndex: 0 },
  standardEnemyBox.x,
  standardEnemyBox.y,
  standardEnemyBox.w,
  standardEnemyBox.h,
  1,
  { registration: animationData.ENEMY_ANIMATION_ASSETS.slimelet.registration }
);
assert.strictEqual(enemyDrawState.drawWidth, enemyDrawState.drawHeight,
  'authored enemy registration should preserve square frame proportions');
const groundRegistration = animationData.ENEMY_ANIMATION_ASSETS.slimelet.registration;
assert(Math.abs(enemyDrawState.translateY + enemyDrawState.drawY + groundRegistration.groundY * enemyDrawState.drawHeight / 160 -
  (standardEnemyBox.y + standardEnemyBox.h)) < 1e-9,
'the imported foot landmark should map exactly to the grounded render baseline');
assert(Math.abs(enemyDrawState.translateX + enemyDrawState.drawX + groundRegistration.originX * enemyDrawState.drawWidth / 160 -
  (standardEnemyBox.x + standardEnemyBox.w / 2)) < 1e-9,
'the imported horizontal origin should remain centered without per-frame cropping');
const flyerRegistration = animationData.ENEMY_ANIMATION_ASSETS.galeHarrier.registration;
assert.strictEqual(flyerRegistration.centered, true, 'the authored flyer should declare a hover-center anchor');
const flyerDrawState = visuals.createAnimationFrameDrawState(
  { frameWidth: 160, frameHeight: 160, row: 0, frameIndex: 0 },
  flyerBox.x, flyerBox.y, flyerBox.w, flyerBox.h, -1, { registration: flyerRegistration }
);
assert(Math.abs(flyerDrawState.translateY + flyerDrawState.drawY + flyerRegistration.groundY * flyerDrawState.drawHeight / 160 -
  (flyer.y + flyer.h / 2)) < 1e-9,
'the imported hover landmark should map exactly to the flyer center');
assert.strictEqual(flyerDrawState.scaleX, -1, 'facing should mirror around the declared origin');
assert.strictEqual(
  visuals.getActorAnimationElapsed({ frames: 3, fps: 3, loop: true }, { animationStartedAt: 4, animationPhaseOffset: 0.25 }, 5, () => 2),
  1.5,
  'looping enemies should receive deterministic fractional phase offsets'
);
assert.strictEqual(
  visuals.getActorAnimationElapsed({ frames: 3, fps: 3, loop: false }, { animationStartedAt: 4, animationPhaseOffset: 0.75 }, 5, () => 2),
  1,
  'one-shot enemy actions should not be phase shifted'
);

Object.values(animationData.ENEMY_ANIMATION_ASSETS).forEach((animation) => {
  assert.strictEqual(animation.frameWidth, 160, 'illustrated enemy cells should be 160px wide');
  assert.strictEqual(animation.frameHeight, 160, 'illustrated enemy cells should be 160px high');
  assert(animation.registration && animation.registration.authoredBodyHeight > 0,
    'every enemy animation should carry measured identity registration');
  Object.entries(animation.states).forEach(([stateId, state]) => {
    assert.strictEqual(state.frames, 6, `${stateId} should expose all six authored poses`);
    assert.strictEqual(state.holds.length, 6, `${stateId} hold weights should cover all six poses`);
  });
});
assert.strictEqual(animationData.ENEMY_ANIMATION_ASSETS.briarStag.states.attack.fps, 13,
  'illustrated enemy assets should retain per-monster timing overrides');
assert.deepStrictEqual(animationData.ENEMY_ANIMATION_ASSETS.briarStag.states.attack.holds, [3, 1, 1, 2, 2, 4],
  'per-monster attack weights should preserve all authored anticipation, action, and recovery poses');

const engineCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-engine.js'), 'utf8');
assert(engineCode.includes('this.drawBackground(ctx, width, solidBandBottom, palette, map)'),
  'Canvas should continue the authored background through the reserved world-to-HUD band');
assert((engineCode.match(/createEnemySpriteRenderBox\(enemy\)/g) || []).length >= 3,
  'renderer snapshots and Canvas fallback should share the centralized enemy render box');
assert(engineCode.includes('animation && animation.registration ? { registration: animation.registration }'),
  'Canvas enemy animation drawing should use the imported per-identity registration');
assert(engineCode.includes('registration: animation && animation.registration || ENEMY_SPRITE_REGISTRATION'),
  'renderer snapshots should carry the same imported registration');
assert(rendererCode.includes('actor.registration || ENEMY_SPRITE_REGISTRATION'),
  'Pixi enemy animation drawing should consume the snapshot registration');

const Data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
const { createProjectStarfallEngine } = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-engine.js'));
const engine = createProjectStarfallEngine(null, Data);
for (const [id, windup, commitment] of [['slimelet', 0.42, 0.2], ['briarStag', 0.75, 0.2], ['brambleking', 1, 0.3]]) {
  const enemyData = Data.ENEMIES.find((enemy) => enemy.id === id);
  const actor = { data: enemyData, pendingAttack: { windup }, telegraph: windup, animationStartedAt: Date.now() / 1000, animationDuration: windup };
  assert.strictEqual(engine.getAnimationFrame(enemyData.animation, 'telegraph', actor).frameIndex, 0,
    `${id} should begin at its first warning pose`);
  actor.telegraph = commitment + 0.001;
  assert(engine.getAnimationFrame(enemyData.animation, 'telegraph', actor).frameIndex < 5,
    `${id} should not settle into its final pose before commitment`);
  actor.telegraph = commitment;
  assert.strictEqual(engine.getAnimationFrame(enemyData.animation, 'telegraph', actor).frameIndex, 5,
    `${id} should visibly commit at the promised warning boundary`);
  actor.telegraph = 0.001;
  assert.strictEqual(engine.getAnimationFrame(enemyData.animation, 'telegraph', actor).frameIndex, 5,
    `${id} should hold its committed pose through the final preparation millisecond`);
}
const originalNow = Date.now;
try {
  let nowMs = 1000000;
  Date.now = () => nowMs;
  const melee = Data.ENEMIES.find((enemy) => enemy.id === 'slimelet');
  const recovering = { data: melee, animationStartedAt: nowMs / 1000, animationDuration: 0.12 };
  assert.strictEqual(engine.getAnimationFrame(melee.animation, 'attack', recovering).frameIndex, 0,
    'a short attack should start with its authored contact pose');
  nowMs += 119;
  assert.strictEqual(engine.getAnimationFrame(melee.animation, 'attack', recovering).frameIndex, 5,
    'a short attack should reach its final authored recovery pose before its gameplay timer expires');
  const oracle = Data.ENEMIES.find((enemy) => enemy.id === 'icebloomOracle');
  const healing = { data: oracle, animationStartedAt: nowMs / 1000, animationDuration: 0.65 };
  nowMs += 349;
  assert.strictEqual(engine.getAnimationFrame(oracle.animation, 'buff', healing).frameIndex, 3,
    'the healer should remain in its gathering poses before the 350ms contact event');
  nowMs += 2;
  assert.strictEqual(engine.getAnimationFrame(oracle.animation, 'buff', healing).frameIndex, 4,
    'the healer should show its release pose with the 350ms restorative pulse');
} finally {
  Date.now = originalNow;
}

require('./project-starfall-recovery-layering.test');
require('./project-starfall-pixi-viewport-clip.test');
require('./project-starfall-pixi-retention.test');
console.log('Project Starfall visual viewport tests passed.');
