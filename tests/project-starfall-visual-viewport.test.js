'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const viewport = require(path.join(ROOT, 'js/games/project-starfall/engine/viewport.js'));
const visuals = require(path.join(ROOT, 'js/games/project-starfall/engine/visuals.js'));
const animations = require(path.join(ROOT, 'js/games/project-starfall/data/animations.js'));
const shopVendors = require(path.join(ROOT, 'js/games/project-starfall/data/shop-vendors.js'));
const mapPublication = require(path.join(ROOT, 'js/games/project-starfall/data/map-publication.js'));

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
  'each shop category should publish an existing structure-atlas facade'
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
  { frameWidth: 128, frameHeight: 128, row: 0, frameIndex: 0 },
  standardEnemyBox.x,
  standardEnemyBox.y,
  standardEnemyBox.w,
  standardEnemyBox.h,
  1,
  { registration: visuals.ENEMY_SPRITE_REGISTRATION }
);
assert.strictEqual(enemyDrawState.drawWidth, enemyDrawState.drawHeight,
  'authored enemy registration should preserve square frame proportions');
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

const animationData = animations.createAnimationData();
Object.values(animationData.ENEMY_ANIMATION_ASSETS).forEach((animation) => {
  assert.strictEqual(animation.states.hit.frames, 3, 'compact enemy hit rows should expose all three authored frames');
  assert.strictEqual(animation.states.hit.holds.length, 3, 'compact enemy hit timing should cover all three authored frames');
});
assert.strictEqual(animationData.ENEMY_ANIMATION_ASSETS.briarStag.states.attack.fps, 13,
  'compact enemy assets should retain per-monster timing overrides');
assert.deepStrictEqual(animationData.ENEMY_ANIMATION_ASSETS.briarStag.states.attack.holds, [3, 2, 4],
  'six-frame timing accents should map onto compact anticipation, action, and recovery frames');

const engineCode = fs.readFileSync(path.join(ROOT, 'js/games/project-starfall/project-starfall-engine.js'), 'utf8');
assert((engineCode.match(/createEnemySpriteRenderBox\(enemy\)/g) || []).length >= 3,
  'renderer snapshots and Canvas fallback should share the centralized enemy render box');
assert(engineCode.includes('ENEMY_SPRITE_DRAW_OPTIONS'), 'Canvas enemy animation drawing should use authored registration');
assert(rendererCode.includes('? ENEMY_SPRITE_REGISTRATION'), 'Pixi enemy animation drawing should use the same authored registration');

console.log('Project Starfall visual viewport tests passed.');
