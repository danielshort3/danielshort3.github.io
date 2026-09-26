'use strict';

const assert = require('assert');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js');

let mutations = 0;
function point() {
  return {
    x: 0,
    y: 0,
    set(x, y = x) {
      if (this.x !== x || this.y !== y) mutations += 1;
      this.x = x;
      this.y = y;
    }
  };
}

class Sprite {
  constructor(texture) {
    this.position = point();
    this.scale = point();
    this.anchor = point();
    for (const property of ['texture', 'rotation', 'alpha', 'tint', 'blendMode', 'visible']) {
      Object.defineProperty(this, property, {
        get() { return this[`_${property}`]; },
        set(value) {
          if (this[`_${property}`] !== value) mutations += 1;
          this[`_${property}`] = value;
        }
      });
    }
    this.texture = texture;
  }
}

const white = { width: 1, height: 1 };
const renderer = createRenderer({ PIXI: { Sprite, Texture: { WHITE: white } } });
renderer.createPool('map', { addChild() {} });
const texture = { width: 64, height: 128 };
const settings = { anchorX: 0, anchorY: 1, flipX: true, alpha: 0.7, tint: 0x123456, rotation: 0.2, blendMode: 'add' };
renderer.beginPools();
renderer.drawTexture('map', texture, 160, 280, 96, 192, settings);
renderer.hideUnusedSprites();
const first = renderer.spritePools.map.items[0];
assert.strictEqual(first.texture, texture);
assert.deepStrictEqual([first.position.x, first.position.y, first.scale.x, first.scale.y], [160, 280, -1.5, 1.5]);
assert.deepStrictEqual([first.anchor.x, first.anchor.y, first.alpha, first.tint, first.rotation, first.blendMode], [0, 1, 0.7, 0x123456, 0.2, 'add']);

mutations = 0;
for (let frame = 0; frame < 60; frame += 1) {
  renderer.beginPools();
  renderer.drawTexture('map', texture, 160, 280, 96, 192, settings);
  renderer.hideUnusedSprites();
}
assert.strictEqual(mutations, 0, 'steady sprites must not dirty their retained texture or transform on every frame');
assert.strictEqual(renderer.spritePools.map.items.length, 1, 'the same sprite is reused');

renderer.beginPools();
renderer.drawTexture('map', white, 0, 0, 4, 8);
renderer.hideUnusedSprites();
assert.deepStrictEqual([first.anchor.x, first.anchor.y, first.scale.x, first.scale.y], [0.5, 0.5, 4, 8]);
assert.deepStrictEqual([first.alpha, first.tint, first.rotation, first.blendMode], [1, 0xffffff, 0, 'normal'], 'default drawing must overwrite every previous optional property');
renderer.beginPools();
renderer.hideUnusedSprites();
assert.strictEqual(first.visible, false, 'unused retained sprites are hidden');
renderer.beginPools();
renderer.drawTexture('map', texture, 160, 280, 96, 192, settings);
renderer.hideUnusedSprites();
assert.strictEqual(first.visible, true, 'a hidden retained sprite becomes visible when reused');

function graphicsRecorder() {
  const graphics = { builds: 0, commands: [] };
  graphics.clear = () => { graphics.builds += 1; graphics.commands.length = 0; return graphics; };
  for (const method of ['rect', 'fill', 'ellipse', 'moveTo', 'lineTo', 'closePath', 'stroke', 'circle']) {
    graphics[method] = (...args) => { graphics.commands.push([method, ...args]); return graphics; };
  }
  return graphics;
}

renderer.backgroundGraphics = graphicsRecorder();
renderer.worldBaseBandGraphics = graphicsRecorder();
let backgroundTexture = { width: 1280, height: 640 };
renderer.getTexture = () => backgroundTexture;
const backgroundDraws = [];
renderer.drawTexture = (...args) => { backgroundDraws.push(args); return true; };
const snapshot = {
  width: 1280, height: 720, playfieldHeight: 640, solidPlatformHeight: 48,
  map: { id: 'greenrootMeadow', asset: 'background.webp', palette: ['#77bf65', '#91dbe8'] },
  runtime: { worldWidth: 5000 }, camera: { x: 0, y: 0, w: 1280, h: 640 },
  bounds: { left: 0, right: 1280, top: 0, bottom: 640 }
};
renderer.renderBackground(snapshot);
const backgroundCommands = JSON.stringify(renderer.backgroundGraphics.commands);
snapshot.camera.x = 80;
renderer.renderBackground(snapshot);
assert.strictEqual(renderer.backgroundGraphics.builds, 1, 'camera panning retains unchanged background geometry');
assert.strictEqual(JSON.stringify(renderer.backgroundGraphics.commands), backgroundCommands);
assert(backgroundDraws.some((draw) => draw[2] < -2), 'background sprite parallax still updates independently');
snapshot.map.palette[1] = '#123456';
renderer.renderBackground(snapshot);
assert.strictEqual(renderer.backgroundGraphics.builds, 2, 'palette edits invalidate retained graphics');
backgroundTexture = null;
renderer.renderBackground(snapshot);
renderer.renderBackground(snapshot);
assert.strictEqual(renderer.backgroundGraphics.builds, 4, 'procedural backgrounds continue to animate with the camera');
backgroundTexture = { width: 1280, height: 640 };
snapshot.map.backgroundMode = 'panorama';
renderer.renderBackground(snapshot);
assert.strictEqual(renderer.backgroundGraphics.builds, 5, 'a newly loaded panorama replaces the procedural fallback');

const mapGraphics = graphicsRecorder();
const visible = { x: 300, y: 250, w: 60, h: 140 };
const distant = { x: 10000, y: 250, w: 60, h: 140 };
const runtime = { climbables: [visible, distant], portals: [visible, distant], stations: [visible, distant], questNpcs: [visible, distant] };
renderer.renderClimbables(mapGraphics, runtime, {}, snapshot.bounds);
const visibleLadderCommands = mapGraphics.commands.length;
mapGraphics.commands.length = 0;
renderer.renderClimbables(mapGraphics, { climbables: [visible] }, {}, snapshot.bounds);
assert.strictEqual(mapGraphics.commands.length, visibleLadderCommands, 'offscreen climbables add no graphics commands');
const portalCalls = [];
renderer.renderPortalLabel = (graphics, portal) => portalCalls.push(portal);
renderer.renderPortals(mapGraphics, runtime, snapshot);
assert.deepStrictEqual(portalCalls, [visible], 'distant portals and their text do not enter the render batch');
const elevated = { ...visible, y: -1200 };
renderer.renderPortals(mapGraphics, { portals: [elevated] }, snapshot);
assert.strictEqual(portalCalls[portalCalls.length - 1], elevated, 'edge-clamped portal labels remain visible above the camera');
const assetReads = [];
renderer.getTexture = (asset) => { assetReads.push(asset); return null; };
renderer.renderStations(mapGraphics, runtime, snapshot.bounds);
renderer.renderQuestNpcs(mapGraphics, runtime, snapshot.bounds);
assert.strictEqual(assetReads.length, 2, 'only visible stations and NPCs load and draw their artwork');

const retainedRenderer = createRenderer({ PIXI: { Sprite, Texture: { WHITE: white } } });
retainedRenderer.createPool('map', { addChild() {} });
for (const property of ['mapGraphics', 'mapRearGraphics', 'mapFrontGraphics', 'mapClimbableGraphics', 'mapInteractionGraphics']) {
  retainedRenderer[property] = graphicsRecorder();
}
let staticBuilds = 0;
let liveFrames = 0;
const sceneryTexture = { width: 64, height: 64 };
retainedRenderer.setBaseTextureValue('terrain.png', sceneryTexture);
let retainedTextureDestroyed = false;
retainedRenderer.setCacheValue(retainedRenderer.trimmedTextures, 'retained-prop', { texture: { destroy() { retainedTextureDestroyed = true; } }, canvas: {} });
retainedRenderer.setCacheValue(retainedRenderer.environmentTextures, 'retained-tile', sceneryTexture);
retainedRenderer.renderStaticMap = (state) => {
  staticBuilds += 1;
  retainedRenderer.drawTexture('map', retainedRenderer.getTexture('terrain.png'), 300, 280, 64, 64);
  retainedRenderer.getCacheValue(retainedRenderer.trimmedTextures, 'retained-prop');
  retainedRenderer.getCacheValue(retainedRenderer.environmentTextures, 'retained-tile');
  retainedRenderer.mapGraphics.rect(300, 280, 64, 64).fill({ color: 0xffffff });
};
retainedRenderer.renderClimbables = () => {};
retainedRenderer.renderQuestNpcs = () => {};
retainedRenderer.renderStations = () => {};
retainedRenderer.renderPortals = () => {
  liveFrames += 1;
  retainedRenderer.drawTexture('map', white, 500, 300, 20, 20);
};
function frame(overrides = {}) {
  const state = { ...snapshot, ...overrides };
  retainedRenderer.beginPools();
  retainedRenderer.renderMap(state);
  retainedRenderer.hideUnusedSprites();
}
frame();
frame();
assert.strictEqual(staticBuilds, 1, 'static terrain is built once across repeated frames');
assert.strictEqual(liveFrames, 2, 'animated portal rendering still runs on every frame');
assert.strictEqual(retainedRenderer.spritePools.map.active, 2, 'dynamic sprites follow the retained static prefix');
assert(retainedRenderer.spritePools.map.items.every((sprite) => sprite.visible), 'retained sprites are not hidden by pool cleanup');
frame({ bounds: { left: 50, right: 1230, top: 10, bottom: 630 } });
assert.strictEqual(staticBuilds, 1, 'camera motion within retained overscan keeps geometry');
frame({ bounds: { left: 280, right: 1560, top: 10, bottom: 630 } });
assert.strictEqual(staticBuilds, 2, 'crossing the retained area refreshes scenery before it enters view');
frame({ runtime: { ...snapshot.runtime, platforms: [{ id: 'ground', x: 0.5, y: 500, w: 1000, h: 40 }] } });
assert.strictEqual(staticBuilds, 3, 'changed map geometry invalidates scenery');
frame({ runtime: { ...snapshot.runtime, platforms: [{ id: 'ground', x: 0.75, y: 500, w: 1000, h: 40 }] } });
assert.strictEqual(staticBuilds, 4, 'subpixel geometry edits are not lost to rounded cache keys');
retainedRenderer.setBaseTextureValue('newly-loaded.png', { width: 32, height: 32 });
frame();
assert.strictEqual(staticBuilds, 5, 'loaded assets invalidate fallback scenery');
retainedRenderer.setBaseTextureValue('unused.png', { width: 32, height: 32 });
frame();
frame();
assert.strictEqual(Array.from(retainedRenderer.textures.keys()).pop(), 'terrain.png', 'retained scenery protects its active texture from LRU eviction');
retainedRenderer.setCacheValue(retainedRenderer.trimmedTextures, 'other-prop', {}, 8);
retainedRenderer.setCacheValue(retainedRenderer.environmentTextures, 'other-tile', {}, 8);
frame();
assert.strictEqual(Array.from(retainedRenderer.trimmedTextures.keys()).pop(), 'retained-prop', 'retained trimmed textures stay recent during animated actor draws');
assert.strictEqual(Array.from(retainedRenderer.environmentTextures.keys()).pop(), 'retained-tile', 'retained atlas cells stay recent too');
retainedRenderer.deferredMapTextureReleases = [];
retainedRenderer.deleteCacheValue(retainedRenderer.trimmedTextures, 'retained-prop');
assert.strictEqual(retainedTextureDestroyed, false, 'an eviction cannot destroy scenery already queued for the current presentation');
const releases = retainedRenderer.deferredMapTextureReleases;
retainedRenderer.deferredMapTextureReleases = null;
releases.forEach(({ cache, value }) => retainedRenderer.releaseCacheValue(cache, value));
assert.strictEqual(retainedTextureDestroyed, true, 'evicted resources can be released after the queued frame is presented');

const edited = { ...snapshot, runtime: { platforms: [{ id: 'ground', x: 0, y: 500, w: 1000, h: 40 }], climbables: [{ ...visible, id: 'rope' }], stations: [{ ...visible, id: 'storage' }] } };
let editKey = retainedRenderer.getStaticMapRenderState(edited).key;
edited.runtime.platforms[0].x = 10;
assert.notStrictEqual(retainedRenderer.getStaticMapRenderState(edited).key, editKey, 'reusing the same snapshot does not conceal an in-place geometry edit');
editKey = retainedRenderer.getStaticMapRenderState(edited).key;
edited.runtime.climbables[0].id = 'ladder';
assert.notStrictEqual(retainedRenderer.getStaticMapRenderState(edited).key, editKey, 'climbable identity changes invalidate their material style');
editKey = retainedRenderer.getStaticMapRenderState(edited).key;
edited.runtime.stations[0].id = 'upgrade';
assert.notStrictEqual(retainedRenderer.getStaticMapRenderState(edited).key, editKey, 'station identity changes invalidate facade bindings');

console.log('Project Starfall Pixi retained sprite tests passed.');
