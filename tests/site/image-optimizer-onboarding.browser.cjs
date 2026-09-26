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
      assert(await page.locator('#imgopt-resize-mode').isVisible(), 'Format and resize controls stay visible together.');
      await page.locator('#imgopt-resize-mode').selectOption('maxWidth');
      await page.locator('#imgopt-width').fill('480');
      await page.locator('#imgopt-sample').click();
      await page.waitForFunction(() => document.querySelector('#imgopt-status').textContent.includes('image is ready'));
      assert.equal(await page.locator('#imgopt-format').inputValue(), 'image/webp', 'Trying the sample preserves the chosen format.');
      assert.equal(await page.locator('#imgopt-width').inputValue(), '480', 'Trying the sample preserves the resize settings.');
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 1);
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-results a[download]').first().waitFor();
      assert.equal(await page.locator('#imgopt-status').getAttribute('data-tone'), 'success', 'A completed optimization has the shared success state.');
      assert.match(await page.locator('#imgopt-status').innerText(), /^Done\./, 'Success remains understandable without color.');
      const downloading = page.waitForEvent('download');
      await page.locator('#imgopt-results a[download]').first().click();
      const download = await downloading;
      const output = await sharp(await download.path()).metadata();
      assert.equal(output.format, 'webp', 'The sample produces a real WebP download.');
      assert.equal(output.width, 480);
      assert.equal(output.height, 320, 'Resizing keeps the sample aspect ratio.');
      assert.match(download.suggestedFilename(), /^sample-landscape.*\.webp$/);
      await page.screenshot({ path: path.join(artifactDir, `image-optimizer-${viewport.width}-result.png`) });
      await page.locator('#imgopt-quality').fill('70');
      assert.equal(await page.locator('#imgopt-results a[download]').count(), 0, 'Setting changes invalidate previous downloads.');
      assert(await page.locator('#imgopt-download-all').isDisabled());
      assert.equal(await page.locator('#imgopt-status').getAttribute('data-tone'), 'info', 'Changing settings clears the old success state.');
      await page.locator('#imgopt-sample').click();
      await page.waitForFunction(() => document.querySelectorAll('#imgopt-filelist .imgopt-file').length === 2 && document.querySelector('#imgopt-status').textContent.includes('image is ready'));
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 2, 'Adding a sample does not replace selected images.');
      // Exercise the real processing failure path without malformed uploads or external services.
      await page.evaluate(() => {
        window.__imgoptOriginalToBlob = HTMLCanvasElement.prototype.toBlob;
        HTMLCanvasElement.prototype.toBlob = function (callback) { callback(null); };
      });
      try {
        await page.locator('#imgopt-process').click();
        await page.waitForFunction(() => document.querySelector('#imgopt-status').dataset.tone === 'error');
        assert.match(await page.locator('#imgopt-status').innerText(), /Unable to encode image/, 'A failed optimization retains its textual error explanation.');
        assert.equal(await page.locator('#imgopt-results a[download]').count(), 0, 'Failed processing cannot leave successful downloads behind.');
      } finally {
        await page.evaluate(() => {
          HTMLCanvasElement.prototype.toBlob = window.__imgoptOriginalToBlob;
          delete window.__imgoptOriginalToBlob;
        });
      }
      await page.locator('#imgopt-clear').click();
      assert.equal(await page.locator('#imgopt-filelist .imgopt-file').count(), 0);
      assert.equal(await page.locator('#imgopt-results a[download]').count(), 0);
      assert(await page.locator('#imgopt-process').isDisabled(), 'Clear returns the tool to its empty state.');
      assert.equal(await page.locator('#imgopt-status').getAttribute('data-tone'), 'info', 'Clear removes the previous error state.');

      await page.locator('#imgopt-format').selectOption('keep');
      await page.locator('#imgopt-resize-mode').selectOption('none');
      await page.locator('#imgopt-sample').click();
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-size-advice').waitFor({ state: 'visible' });
      assert.match(await page.locator('#imgopt-size-advice').innerText(), /No size reduction[\s\S]*metadata removed/);
      assert(await page.locator('#imgopt-status').isHidden(), 'The concise size result replaces the generic completion line.');
      assert.equal(await page.locator('#imgopt-results a[download]').count(), 1, 'The PNG remains downloadable before opting into WebP.');
      await page.locator('#imgopt-try-webp').click();
      await page.waitForFunction(() => document.querySelectorAll('#imgopt-results a[download]').length === 2);
      assert.match(await page.locator('#imgopt-status').innerText(), /Original downloads remain available/);
      const names = await page.locator('#imgopt-results a[download]').evaluateAll(nodes => nodes.map(node => node.download));
      assert(names.some(name => name.endsWith('.png')) && names.some(name => name.endsWith('.webp')), 'Opting in adds a WebP alternative alongside the PNG.');
      const webpDownload = page.waitForEvent('download');
      await page.locator('#imgopt-results a[download$=".webp"]').click();
      assert.equal((await sharp(await (await webpDownload).path()).metadata()).format, 'webp', 'The WebP alternative is genuinely encoded as WebP.');

      await page.locator('#imgopt-sample').click();
      await page.waitForFunction(() => document.querySelectorAll('#imgopt-filelist .imgopt-file').length === 2);
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-size-advice').waitFor({ state: 'visible' });
      assert.match(await page.locator('#imgopt-size-advice-title').innerText(), /No batch size reduction/);
      await page.locator('#imgopt-try-webp').click();
      await page.waitForFunction(() => document.querySelectorAll('#imgopt-results a[download]').length === 4);
      assert.equal(await page.locator('#imgopt-results a[download$=".png"]').count(), 2, 'Batch WebP generation keeps both PNG downloads.');
      assert.equal(await page.locator('#imgopt-results a[download$=".webp"]').count(), 2, 'Batch WebP generation adds one alternative per image.');
      const captured = await page.evaluate(() => {
        const payload = {};
        document.dispatchEvent(new CustomEvent('tools:session-capture', { detail: { toolId: 'image-optimizer', payload } }));
        return payload;
      });
      assert.match(captured.outputSummary, /4 outputs/, 'Session preview counts originals and alternatives.');
      assert.match(captured.output.text, /\.png[\s\S]*\.webp|\.webp[\s\S]*\.png/, 'Session preview names both output formats.');

      await page.locator('#imgopt-clear').click();
      await page.locator('#imgopt-sample').click();
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-size-advice').waitFor({ state: 'visible' });
      await page.evaluate(() => {
        window.__imgoptOriginalToBlob = HTMLCanvasElement.prototype.toBlob;
        HTMLCanvasElement.prototype.toBlob = function (callback, mime, quality) {
          return window.__imgoptOriginalToBlob.call(this, blob => setTimeout(() => callback(blob), 250), mime, quality);
        };
      });
      try {
        await page.locator('#imgopt-try-webp').click();
        await page.waitForFunction(() => document.querySelector('#imgopt-clear').textContent === 'Cancel');
        await page.locator('#imgopt-clear').click();
        await page.waitForFunction(() => document.querySelector('#imgopt-status').textContent.includes('Optimization cancelled'));
        assert.equal(await page.locator('#imgopt-results a[download]').count(), 1, 'Cancelling an alternative keeps the PNG download.');
      } finally {
        await page.evaluate(() => {
          HTMLCanvasElement.prototype.toBlob = window.__imgoptOriginalToBlob;
          delete window.__imgoptOriginalToBlob;
        });
      }

      await page.locator('#imgopt-clear').click();
      await page.locator('#imgopt-sample').click();
      await page.locator('#imgopt-process').click();
      await page.locator('#imgopt-size-advice').waitFor({ state: 'visible' });
      await page.evaluate(() => {
        window.__imgoptOriginalToBlob = HTMLCanvasElement.prototype.toBlob;
        HTMLCanvasElement.prototype.toBlob = function (callback) { callback(new Blob(['PNG fallback'], { type: 'image/png' })); };
      });
      try {
        await page.locator('#imgopt-try-webp').click();
        await page.waitForFunction(() => document.querySelector('#imgopt-status').dataset.tone === 'error');
        assert.match(await page.locator('#imgopt-status').innerText(), /WebP encoding is unavailable[\s\S]*Original downloads remain available/);
        assert.equal(await page.locator('#imgopt-results a[download]').count(), 1, 'An encoding fallback retains the existing PNG download.');
        const pngDownload = page.waitForEvent('download');
        await page.locator('#imgopt-results a[download$=".png"]').click();
        assert.equal((await sharp(await (await pngDownload).path()).metadata()).format, 'png', 'The retained PNG is still valid after a WebP failure.');
      } finally {
        await page.evaluate(() => {
          HTMLCanvasElement.prototype.toBlob = window.__imgoptOriginalToBlob;
          delete window.__imgoptOriginalToBlob;
        });
      }
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
    let browserDidNotClose = false;
    if (browser) {
      // Chromium can disconnect after downloads while Playwright's close promise remains pending.
      let closeTimer;
      try {
        await Promise.race([
          browser.close(),
          new Promise(resolve => { closeTimer = setTimeout(resolve, 5000); })
        ]);
      } finally {
        clearTimeout(closeTimer);
      }
      browserDidNotClose = browser.isConnected();
    }
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
    if (browserDidNotClose) throw new Error('Image Optimizer browser did not close.');
  }
}

module.exports = runImageOptimizerChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
