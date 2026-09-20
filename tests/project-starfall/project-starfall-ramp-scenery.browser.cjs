'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const data = require('../../js/games/project-starfall/project-starfall-data.js');
const root = path.resolve(__dirname, '../..');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    for (const file of ['core/math.js', 'core/geometry.js', 'engine/scenery-placement.js']) {
      await page.addScriptTag({ path: path.join(root, 'js/games/project-starfall', file) });
    }
    let cases = 0;
    let columns = 0;
    for (const asset of Object.values(data.ENVIRONMENT_ASSETS.ramps)) {
      const url = 'data:image/png;base64,' + fs.readFileSync(path.join(root, asset.path)).toString('base64');
      const result = await page.evaluate(async ({ url, asset }) => {
        const image = new Image();
        image.src = url;
        await image.decode();
        const failures = [];
        let columns = 0;
        for (let cell = 0; cell < 4; cell += 1) {
          const rise = cell % 2 ? 160 : -160;
          const surface = ProjectStarfallEngineModules.sceneryPlacement.createRampSurface(image, asset, cell, 300, rise, 28);
          if (!surface) { failures.push(`cell ${cell}: missing surface`); continue; }
          const canvas = surface.canvas;
          const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
          for (let x = 1; x < canvas.width - 1; x += 1) {
            let top = -1;
            for (let y = 0; y < canvas.height; y += 1) {
              if (pixels[(y * canvas.width + x) * 4 + 3] >= 64) { top = y; break; }
            }
            const actualWorldY = top + surface.topOffset;
            const collisionY = rise * x / 300;
            if (top < 0 || Math.abs(actualWorldY - collisionY) > 2) failures.push(`cell ${cell} x${x}: visible edge ${actualWorldY}, collision ${collisionY}`);
            columns += 1;
          }
        }
        return { failures: failures.slice(0, 5), columns };
      }, { url, asset });
      assert.deepStrictEqual(result.failures, [], `${asset.path}: painted contact follows collision`);
      cases += 4;
      columns += result.columns;
    }
    console.log(`Starfall ramp art: ${cases} source cells, ${columns} independently sampled pixel columns within 2px of collision surface.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
