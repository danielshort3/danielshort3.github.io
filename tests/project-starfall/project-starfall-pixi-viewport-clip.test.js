'use strict';

const assert = require('assert');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js');
const anchorCanvas = { style: {} };
const worldCanvas = { style: {} };
let resolution = 1;
const sizes = [];
const renderer = createRenderer({ PIXI: {}, anchorCanvas, getRenderScale: () => resolution });
renderer.app = { canvas: worldCanvas, renderer: { resize: (...args) => sizes.push(args) }, render() {} };
renderer.ready = true;
renderer.active = true;
[
  'beginPools', 'clearGraphics', 'renderBackground', 'updateWorldTransform', 'renderMap', 'renderWorldEffects',
  'renderProjectiles', 'renderLoot', 'renderEnemies', 'renderParty', 'renderPet', 'renderPlayer',
  'renderDamageSplats', 'hideUnusedSprites'
].forEach((method) => { renderer[method] = () => {}; });

function assertVisibleBottom(height, expectedBottom) {
  const match = worldCanvas.style.clipPath.match(/^inset\(0 0 ([\d.]+)% 0\)$/);
  assert(match, 'the world canvas must have a responsive bottom clip');
  const actualBottom = height * (1 - Number(match[1]) / 100);
  assert(Math.abs(actualBottom - expectedBottom) < 0.00001,
    `world drawing must stop at ${expectedBottom}, received ${actualBottom}`);
  assert.deepStrictEqual(anchorCanvas.style, {}, 'the HUD canvas must remain unclipped');
}

renderer.renderFrame({ width: 1280, height: 806, playfieldHeight: 674, solidPlatformHeight: 48 });
assertVisibleBottom(806, 722);
resolution = 2;
renderer.renderFrame({ width: 900, height: 600, playfieldHeight: 470, solidPlatformHeight: 46 });
assertVisibleBottom(600, 516);
assert.deepStrictEqual(sizes[sizes.length - 1], [900, 600, 2], 'clip geometry must survive viewport and resolution changes');
renderer.renderFrame({ width: 900, height: 600, playfieldHeight: 590, solidPlatformHeight: 46 });
assertVisibleBottom(600, 600);

const bandRects = [];
renderer.worldBaseBandGraphics = { rect(x, y, w, h) { bandRects.push({ x, y, w, h }); return this; }, fill(style) { Object.assign(bandRects[bandRects.length - 1], style); return this; } };
renderer.renderWorldBaseBand({ height: 806, solidPlatformHeight: 48 }, 1280, 674, {});
assert.strictEqual(bandRects[0].y, 672, 'the fade overlaps the scenery by two pixels');
assert.strictEqual(bandRects[bandRects.length - 1].y + 1, 722, 'the fade ends at the reserved boundary above the HUD');
assert.strictEqual(bandRects[0].alpha, 0, 'the boundary starts without a visible stripe');
assert.strictEqual(bandRects[bandRects.length - 1].alpha, 0.78, 'the boundary settles into the dark HUD');
assert(bandRects.every((rect, index) => rect.w === 1280 && rect.h === 1 && (!index || rect.alpha >= bandRects[index - 1].alpha)), 'the boundary darkens smoothly without gaps');
console.log('Project Starfall Pixi world viewport clipping and boundary fade tests passed.');
