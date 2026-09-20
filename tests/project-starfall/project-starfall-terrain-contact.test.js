'use strict';

const assert = require('assert');
const path = require('path');
const sharp = require('sharp');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const { createProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
const { createRenderer } = require('../../js/games/project-starfall/project-starfall-renderer-pixi.js');

async function main() {
  const engine = createProjectStarfallEngine(null, data);
  const renderer = createRenderer({ PIXI: {}, data });
  engine.state.player.classId = 'fighter';
  engine.playAudioCue = () => true;
  engine.getEnvironmentImage = () => ({});
  engine.drawPlatformThemeTrim = () => {};
  renderer.drawPlatformThemeTrim = () => {};
  renderer.getTexture = () => ({});
  renderer.getEnvironmentCellTexture = (kind, profile, cell) => ({ cell });
  let canvasCalls = [];
  let pixiCalls = [];
  const ctx = {
    save() {},
    restore() {},
    drawImage(image, sx, sy, sw, sh, x, y, w, h) {
      canvasCalls.push({ cell: sy / sh * 8 + sx / sw, x, y, w, h });
    }
  };
  renderer.drawTexture = (layer, texture, x, y, w, h, options) => {
    assert.strictEqual(options.anchorX, 0);
    assert.strictEqual(options.anchorY, 0);
    pixiCalls.push({ cell: texture.cell, x, y, w, h });
    return true;
  };
  const atlas = data.ENVIRONMENT_ASSETS.terrain['ashglass-pass'];
  const pixels = await sharp(path.resolve(__dirname, '../..', atlas.path)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.strictEqual(atlas.cellSize, 64);
  assert.strictEqual(atlas.columns, 8);
  const firstOpaqueRows = Array.from({ length: 8 }, (_, cell) => Array.from({ length: 64 }, (_, x) => {
    for (let y = 0; y < 64; y += 1) {
      const sx = cell * 64 + x;
      if (pixels.data[(y * pixels.info.width + sx) * 4 + 3] >= 64) return y;
    }
    return -1;
  }));
  // Repeating ground and ledge art starts at the same source row. End caps are
  // tapered silhouettes: only their inner plateau belongs to this flat check.
  for (const cell of [1, 2, 5, 6]) {
    assert(firstOpaqueRows[cell].every(y => y === 0), `cell ${cell}: repeating support must start at row zero`);
  }
  for (const cell of [0, 4]) assert(firstOpaqueRows[cell].slice(36).every(y => y === 0), `cell ${cell}: left cap inner plateau`);
  for (const cell of [3, 7]) assert(firstOpaqueRows[cell].slice(0, 28).every(y => y === 0), `cell ${cell}: right cap inner plateau`);

  let platforms = 0;
  let renderedColumns = 0;
  const checkedCells = new Set();
  for (const mapId of ['ashglassPass', 'cinderHollow']) {
    const map = data.MAPS.find(entry => entry.id === mapId);
    engine.changeMap(mapId, { silent: true });
    const snapshot = { runtime: engine.runtime, bounds: { left: -100, right: 10000, top: -100, bottom: 10000 } };
    const geometryBefore = JSON.stringify(engine.runtime.platforms);
    const flatPlatforms = engine.runtime.platforms.map((platform, index) => ({ platform, index })).filter(({ platform }) => platform.shape !== 'slope');
    // Also cover the non-field flat dispatch, which has a different top height.
    flatPlatforms.push({ platform: Object.assign({}, flatPlatforms[1].platform, { terrainVisual: undefined }), index: 1 });
    for (const { platform, index } of flatPlatforms) {
      canvasCalls = [];
      pixiCalls = [];
      engine.drawTiledPlatformTerrain(ctx, map, platform, index);
      renderer.drawTiledPlatformTerrain(snapshot, map, platform, index);
      assert(canvasCalls.length > 0, `${mapId}/${platform.id}: must paint terrain`);
      assert.deepStrictEqual(pixiCalls, canvasCalls, `${mapId}/${platform.id}: actual Canvas drawImage / Pixi texture parity`);
      const legacyOffset = index === 0 ? 24 : platform.terrainVisual ? 12 : 16;
      const expectedY = platform.y - (mapId === 'ashglassPass' ? 0 : legacyOffset);
      for (const call of canvasCalls) {
        assert.strictEqual(call.y, expectedY, `${mapId}/${platform.id}: artwork contact registration`);
        assert(call.h > 0 && call.w > 0);
        if (mapId !== 'ashglassPass') continue;
        checkedCells.add(call.cell);
        const columns = [1, 2, 5, 6].includes(call.cell) ? [0, 64] : [0, 4].includes(call.cell) ? [36, 64] : [0, 28];
        for (let x = columns[0]; x < columns[1]; x += 1) {
          const actualPaintedY = call.y + firstOpaqueRows[call.cell][x] * call.h / 64;
          assert(Math.abs(actualPaintedY - platform.y) <= 1, `${platform.id}/cell ${call.cell}/x${x}: painted plateau meets collision within 1px`);
          renderedColumns += 1;
        }
      }
      platforms += 1;
    }
    assert.strictEqual(JSON.stringify(engine.runtime.platforms), geometryBefore, `${mapId}: drawing cannot change collision geometry`);
  }
  assert.deepStrictEqual([...checkedCells].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7], 'exercise both variants and all ground/ledge endcaps');
  console.log(`Starfall terrain contact: ${platforms} flat dispatches match Canvas/Pixi; ${renderedColumns} painted plateau columns meet Ashglass collision within 1px; Cinder lip registration preserved.`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
