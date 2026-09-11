'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const assets = require('../../js/games/project-starfall/core/assets.js');
const rendering = require('../../js/games/project-starfall/ui/asset-rendering.js');
const preview = require('../../js/games/project-starfall/ui/asset-preview.js');
const Data = require('../../js/games/project-starfall/project-starfall-data.js');
const { ProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js');

const revision = '?v=classic-eb8de4e1';
const portrait = Data.GENERIC_PLAYER_ASSET;
const sheet = Data.GENERIC_PLAYER_ANIMATION_ASSET.sheet;

async function main() {
  const restored = [
    ...Object.values(Data.CLASS_ASSETS),
    ...Object.values(Data.PLAYER_ANIMATION_ASSETS).map((animation) => animation.sheet),
    ...Object.values(Data.EQUIPMENT_VISUALS).map((item) => item.atlas && item.atlas.sheet).filter(Boolean),
    Data.MAP_ASSETS.starfallCrossing,
    Data.MAP_ASSETS.greenrootMeadow,
    Data.MAP_ASSETS.eclipseThrone,
    Data.ENVIRONMENT_STRUCTURE_ASSETS.townLandmarks.path,
    'img/project-starfall/ui/start-screen.png',
    'img/project-starfall/ui/start-screen.webp',
    'img/project-starfall/ui/start-screen.avif'
  ];
  for (const assetPath of restored) {
    assert(!assetPath.includes('?'), 'canonical data paths must remain usable as cache keys and filesystem paths');
    assert.strictEqual(assets.getAssetRequestUrl(assetPath), assetPath + revision);
    assert.strictEqual(assets.getAssetRequestUrl(assetPath + revision), assetPath + revision,
      'request revision should be idempotent');
  }
  for (const untouched of [
    '', 'img/games/icons/project-starfall.webp',
    'img/project-starfall/enemies/slimelet.png',
    'img/project-starfall/maps/thornpath-thicket.webp',
    'img/project-starfall/characters/fighter-v5.png',
    'img/project-starfall/equipment-atlases/training-sword-atlas-v2.png',
    'https://example.com/img/project-starfall/characters/generic-player.png'
  ]) {
    assert.strictEqual(assets.getAssetRequestUrl(untouched), untouched,
      'other assets and remote URLs should retain their request identity');
  }
  const framePath = `${sheet}#frame=160,320,160,160,960,1600`;
  const before = assets.parseAssetFrame(framePath);
  const revisedFrame = assets.parseAssetFrame(assets.getAssetRequestUrl(framePath));
  assert.deepStrictEqual(revisedFrame, Object.assign({}, before, { path: sheet + revision }),
    'cache revision must preserve sprite crop coordinates and sheet dimensions');
  assert.strictEqual(assets.getAssetRequestUrl(`${sheet}?preview=1&v=old#frame=0,0,160,160`),
    `${sheet}?preview=1&v=classic-eb8de4e1#frame=0,0,160,160`);
  assert(rendering.renderAssetImage(portrait, 'Player', 'portrait').includes(`src="${portrait}${revision}"`));
  assert(rendering.renderAssetImage(framePath, 'Player', 'portrait').includes(`${sheet}${revision}`));
  const animationStyle = preview.getAssetPreviewAnimationStyle({
    path: sheet,
    animation: { frameWidth: 160, frameHeight: 160, states: [{ id: 'idle', frames: 6, fps: 5, row: 0 }] }
  });
  assert(animationStyle.includes(sheet + revision));

  const originalImage = global.Image;
  try {
    global.Image = class TestImage {
      constructor() { this.complete = true; this.naturalWidth = 160; }
    };
    const engine = Object.create(ProjectStarfallEngine.prototype);
    Object.assign(engine, {
      state: {}, assets: {}, failedAssets: {},
      getMapCriticalAssetSet: () => [portrait],
      emitAssetLoadProgress() {}, scheduleAssetRefresh() {}, recordAssetLoadResult() {},
      decodeAssetImage: () => Promise.resolve()
    });
    engine.loadAssets();
    assert.strictEqual(engine.assets[portrait].src, portrait + revision);
    const result = await engine.ensureAssetReady(framePath);
    assert.strictEqual(result.path, sheet, 'load results must retain the canonical sheet path');
    assert.strictEqual(engine.assets[sheet].src, sheet + revision);
  } finally {
    if (originalImage === undefined) delete global.Image;
    else global.Image = originalImage;
  }

  const requests = [];
  const texture = {};
  const renderer = createRenderer({ PIXI: { Assets: {
    load: async (url) => { requests.push(['load', url]); return texture; },
    unload: (url) => { requests.push(['unload', url]); }
  } } });
  assert.strictEqual(await renderer.loadTexture(sheet), texture);
  assert.strictEqual(await renderer.loadTexture(sheet), texture);
  assert(renderer.textures.has(sheet), 'Pixi local cache keys must remain canonical');
  renderer.releaseBaseTexture(sheet, texture);
  assert.deepStrictEqual(requests, [['load', sheet + revision], ['unload', sheet + revision]],
    'Pixi load/unload must use the same revised URL without duplicate network loads');
  const fallbackRenderer = createRenderer({ PIXI: { Texture: {
    from: (url) => { assert.strictEqual(url, portrait + revision); return texture; }
  } } });
  assert.strictEqual(await fallbackRenderer.loadTexture(portrait), texture);

  const css = fs.readFileSync(path.join(__dirname, '../../css/games/project-starfall/loading.css'), 'utf8');
  for (const extension of ['png', 'webp', 'avif']) {
    assert(css.includes(`start-screen.${extension}${revision}`), 'title artwork must use the same restored-art revision');
  }
  console.log('Project Starfall restored asset request URL checks passed.');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
