'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function runImageOptimizerChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
      await page.goto(base + '/tools/image-optimizer', { waitUntil: 'networkidle' });
      await page.locator('#imgopt-sample').waitFor();
      const layout = await page.evaluate(() => {
        const images = document.querySelector('.imgopt-drop-card');
        const settings = document.querySelector('.imgopt-controls-card');
        const add = document.querySelector('[data-imgopt-pick]').getBoundingClientRect();
        const imageBox = images.getBoundingClientRect();
        const settingsBox = settings.getBoundingClientRect();
        return {
          imagesFirst: Boolean(images.compareDocumentPosition(settings) & Node.DOCUMENT_POSITION_FOLLOWING),
          imagesLeft: imageBox.left, imagesBottom: imageBox.bottom,
          settingsLeft: settingsBox.left, settingsTop: settingsBox.top,
          addTop: add.top, addBottom: add.bottom,
          scrollWidth: document.documentElement.scrollWidth,
          formId: document.querySelector('#imgopt-process').form?.id
        };
      });
      assert(layout.imagesFirst, 'Upload precedes settings in the document and keyboard order.');
      assert.equal(layout.formId, 'imgopt-form', 'The primary action submits the real settings form.');
      assert(layout.scrollWidth <= viewport.width + 1, 'The tool does not overflow horizontally.');
      if (viewport.width < 600) {
        assert(layout.imagesBottom <= layout.settingsTop, 'Upload and results appear before settings on mobile.');
        assert(layout.addTop >= 0 && layout.addBottom < viewport.height, 'Add images is visible on the first mobile screen.');
      } else {
        assert(layout.imagesLeft < layout.settingsLeft, 'Desktop visual order matches the upload-first workflow.');
      }
      await page.screenshot({ path: path.join(artifactDir, `image-optimizer-${viewport.width}-empty.png`) });
      await page.locator('#imgopt-format').selectOption('image/webp');
      await page.locator('#imgopt-tab-resize').click();
      await page.locator('#imgopt-resize-mode').selectOption('maxWidth');
      await page.locator('#imgopt-width').fill('480');
      await page.locator('#imgopt-sample').click();
      await page.waitForFunction(() => document.querySelector('#imgopt-status').textContent.includes('image is ready'));
      assert.equal(await page.locator('#imgopt-format').inputValue(), 'image/webp', 'Trying the sample preserves the chosen format.');
      assert.equal(await page.locator('#imgopt-width').inputValue(), '480', 'Trying the sample preserves the resize settings.');
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 1);
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-results a[download]').first().waitFor();
      const downloading = page.waitForEvent('download');
      await page.locator('#imgopt-results a[download]').first().click();
      const download = await downloading;
      const output = await sharp(await download.path()).metadata();
      assert.equal(output.format, 'webp', 'The sample produces a real WebP download.');
      assert.equal(output.width, 480);
      assert.equal(output.height, 320, 'Resizing keeps the sample aspect ratio.');
      assert.match(download.suggestedFilename(), /^sample-landscape.*\.webp$/);
      await page.screenshot({ path: path.join(artifactDir, `image-optimizer-${viewport.width}-result.png`) });
      await page.locator('#imgopt-sample').click();
      await page.waitForFunction(() => document.querySelectorAll('#imgopt-filelist .imgopt-file').length === 2 && document.querySelector('#imgopt-status').textContent.includes('image is ready'));
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 2, 'Adding a sample does not replace selected images.');
      await page.locator('#imgopt-clear').click();
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 0);
      assert.equal(await page.locator('#imgopt-results a[download]').count(), 0);
      assert(await page.locator('#imgopt-process').isDisabled(), 'Clear returns the tool to its empty state.');
      assert.deepEqual(errors, [], 'The sample and optimization flow have no page exceptions.');
      console.log(`Image Optimizer onboarding passed: ${viewport.width}px`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `image-optimizer-${viewport.width}-failure.png`) }).catch(() => {});
      throw error;
    } finally {
      await context.close();
    }
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'image-optimizer-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runImageOptimizerChecks({ browser, base: `http://127.0.0.1:${server.address().port}`,
      artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-image-optimizer') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runImageOptimizerChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
