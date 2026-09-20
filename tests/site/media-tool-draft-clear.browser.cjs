'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');

async function runMediaDraftClearChecks(browser, base) {
  const context = await browser.newContext({ reducedMotion: 'reduce', serviceWorkers: 'block' });
  await context.route('https://**', route => route.abort());
  await context.route('**/api/**', route => route.fulfill({ status: 200, json: { ok: true, sessions: [], activity: [] } }));
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const ready = async () => {
    await page.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
  };
  try {
    for (const tool of [
      { id: 'image-optimizer', field: '#imgopt-format', clear: '#imgopt-clear', value: 'image/webp', original: 'keep' },
      { id: 'background-remover', field: '#bgtool-method', clear: '#bgtool-reset', value: 'colorkey', original: 'ai-best' }
    ]) {
      const key = `ds:session-draft:v1:tools:${tool.id}`;
      const read = () => page.evaluate(key => sessionStorage.getItem(key), key);
      await page.goto(`${base}/tools/${tool.id}`);
      await ready();
      await page.locator(tool.field).selectOption(tool.value);
      await expect.poll(read).toContain(tool.value);
      await page.reload();
      await ready();
      await expect(page.locator(tool.field)).toHaveValue(tool.value);
      await expect(page.locator('.draft-recovery-notice')).toBeVisible();
      await page.evaluate(() => {
        window.__mediaClears = 0;
        document.addEventListener('tools:session-cleared', () => { window.__mediaClears += 1; });
      });

      if (tool.id === 'image-optimizer') {
        // Exercise the same physical button in its Cancel state using the local
        // sample and a gated browser decode. No external files or APIs are used.
        await page.locator('#imgopt-sample').click();
        await expect(page.locator('#imgopt-filelist')).toContainText('1200 × 800');
        await page.evaluate(() => {
          const decode = window.createImageBitmap.bind(window);
          window.createImageBitmap = (...args) => new Promise(resolve => {
            window.__releaseImageDecode = () => { window.createImageBitmap = decode; resolve(decode(...args)); };
          });
        });
        await page.locator('#imgopt-process').click();
        await page.waitForFunction(() => typeof window.__releaseImageDecode === 'function');
        await expect(page.locator(tool.clear)).toHaveText('Cancel');
        await page.locator(tool.clear).click();
        assert.equal(await page.evaluate(() => window.__mediaClears), 0, 'Cancel is not a whole-tool clear');
        await page.evaluate(() => window.__releaseImageDecode());
        await expect(page.locator(tool.clear)).toHaveText('Clear');
        await expect.poll(read).toContain(tool.value);
      } else {
        // Use the browser's color-removal path; do not load an AI model or send
        // a photo to any service. Clear photos is exposed only after an upload.
        const png = await page.evaluate(() => {
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = 32;
          const context = canvas.getContext('2d');
          context.fillStyle = 'white';
          context.fillRect(0, 0, 32, 32);
          context.fillStyle = 'black';
          context.fillRect(8, 8, 16, 16);
          return canvas.toDataURL('image/png').split(',')[1];
        });
        await page.locator('#bgtool-file').setInputFiles({ name: 'local-clear-fixture.png', mimeType: 'image/png', buffer: Buffer.from(png, 'base64') });
        await page.locator('body[data-tools-state="results"]').waitFor();
        await page.locator('.bgtool-refine-controls > summary').click();
        await page.locator('#bgtool-clear-edits').click();
        assert.equal(await page.evaluate(() => window.__mediaClears), 0, 'Clear edits keeps the rest of the tool recoverable');
        await expect.poll(read).toContain(tool.value);
      }

      await page.locator(tool.clear).click();
      assert.equal(await page.evaluate(() => window.__mediaClears), 1, `${tool.id}: successful whole-tool clear signals once`);
      assert.equal(await read(), null, `${tool.id}: Clear removes the stored settings draft`);
      await expect(page.locator('.draft-recovery-notice')).toHaveCount(0);
      await page.goto(`${base}/tools/word-frequency`);
      await page.goto(`${base}/tools/${tool.id}`);
      await ready();
      await expect(page.locator(tool.field)).toHaveValue(tool.original);
      await expect(page.locator('.draft-recovery-notice')).toHaveCount(0);
      assert.equal(await read(), null, `${tool.id}: navigation cleanup cannot resurrect the cleared draft`);
      console.log(`Media tool draft clear passed: ${tool.id}.`);
    }
    assert.deepEqual(errors, [], 'Media clear flows have no uncaught browser errors');
  } finally { await context.close(); }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'media-clear-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    browser = await chromium.launch();
    await runMediaDraftClearChecks(browser, `http://127.0.0.1:${server.address().port}`);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runMediaDraftClearChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
