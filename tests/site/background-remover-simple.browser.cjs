'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function runBackgroundRemoverSimpleChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const fixture = await sharp(Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="240" height="180"><rect width="240" height="180" fill="white"/><rect x="60" y="40" width="120" height="100" fill="#087f8c"/></svg>')).png().toBuffer();
  for (const width of [1440, 390, 320]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
      await page.goto(base + '/tools/background-remover', { waitUntil: 'networkidle' });
      const essential = page.getByRole('button', { name: 'Essential only', exact: true });
      if (await essential.isVisible()) await essential.click();
      await page.locator('#bgtool-method').selectOption('colorkey');
      assert(await page.locator('.bgtool-detail-options').isHidden(), 'Color removal does not show empty AI options.');
      await page.locator('#bgtool-file').setInputFiles({ name: 'test-shape.png', mimeType: 'image/png', buffer: fixture });
      await page.waitForFunction(() => !document.querySelector('#bgtool-download-selected').disabled);
      assert(await page.locator('#bgtool-format').isVisible(), 'Download format stays beside the preview.');
      assert.equal(await page.locator('#bgtool-format').evaluate(el => el.form.id), 'bgtool-form');
      assert.equal(await page.locator('.bgtool-refine-controls').getAttribute('open'), null);

      let downloading = page.waitForEvent('download');
      await page.locator('#bgtool-download-selected').click();
      const png = await downloading;
      const { data, info } = await sharp(await png.path()).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      assert.equal(info.width, 240);
      assert.equal(info.height, 180);
      assert.equal(data[3], 0, 'The white background is transparent in the actual PNG download.');
      assert.equal(data[(90 * 240 + 120) * 4 + 3], 255, 'The subject stays opaque.');

      await page.locator('#bgtool-format').selectOption('image/jpeg');
      assert(await page.locator('#bgtool-bg').isVisible(), 'Solid formats reveal the background color.');
      downloading = page.waitForEvent('download');
      await page.locator('#bgtool-download-selected').click();
      const jpeg = await downloading;
      const metadata = await sharp(await jpeg.path()).metadata();
      assert.equal(metadata.format, 'jpeg');
      assert.equal(metadata.width, 240);

      await page.locator('.bgtool-refine-controls > summary').click();
      assert(await page.locator('#bgtool-feather').isVisible(), 'Edges are available in the same refinement group as the brush.');
      assert(await page.locator('#bgtool-refine-enabled').isVisible());
      await page.locator('#bgtool-feather').fill('2');
      await page.locator('#bgtool-refine-enabled').check();
      assert(await page.locator('#bgtool-brush-mode').isEnabled());
      await page.locator('#bgtool-refine-enabled').uncheck();

      await page.locator('#bgtool-tolerance').fill('35');
      assert(await page.locator('#bgtool-download-selected').isDisabled(), 'Changed processing settings must disable stale downloads.');
      await page.locator('#bgtool-reprocess').click();
      await page.waitForFunction(() => !document.querySelector('#bgtool-download-selected').disabled);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No horizontal overflow, including open refinements.');
      await page.locator('.bgtool-refine-controls > summary').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(artifactDir, `background-remover-${width}-refinements.png`) });
      await page.locator('#bgtool-reset').click();
      assert(await page.locator('#bgtool-download-selected').isDisabled());
      assert(await page.locator('#bgtool-panel-export').isHidden());
      assert.deepEqual(errors, []);
      console.log(`Background Remover passed: ${width}px, real transparent PNG/JPEG, refinement controls, stale processing, clear.`);
    } finally {
      await context.close();
    }
  }
}

module.exports = runBackgroundRemoverSimpleChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'background-remover-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runBackgroundRemoverSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-background-remover') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
