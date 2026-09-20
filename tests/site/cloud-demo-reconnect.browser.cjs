'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const pixel = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a1d8AAAAASUVORK5CYII=';

async function runCloudReconnectChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const demo of [
    { route: 'shape-demo', endpoint: 'shape', action: '#classify' },
    { route: 'handwriting-rating-demo', endpoint: 'handwriting', action: '#rate' },
    { route: 'digit-generator-demo', endpoint: 'digit-generator', action: '#refresh-seed-btn' }
  ]) {
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    let unavailable = true;
    let generations = 0;
    let frameLoads = 0;
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('framenavigated', frame => { if (frame.url().includes(`/demos/${demo.route}.html`)) frameLoads += 1; });
    await context.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.origin !== base) return route.abort();
      if (!url.pathname.startsWith(`/api/demos/${demo.endpoint}/`)) return route.continue();
      if (unavailable) return route.fulfill({ status: 400, json: { error: 'Controlled unavailable endpoint.' } });
      const operation = url.pathname.split('/').at(-1);
      if (operation === 'generate') {
        generations += 1;
        const payload = route.request().postDataJSON();
        return route.fulfill({ json: { latent_dim: 20, images: Array.from({ length: payload.rows }, () => Array(payload.cols).fill(pixel)) } });
      }
      return route.fulfill({ json: { status: 'ok', model_loaded: true } });
    });
    try {
      await page.goto(`${base}/${demo.route}`, { waitUntil: 'networkidle' });
      if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
      const frame = await (await page.locator('.project-demo-wrapper-iframe').elementHandle()).contentFrame();
      await frame.locator('#health-pill[data-state="err"]').waitFor();
      const reconnect = frame.getByRole('button', { name: /AWS Reconnect/ });
      assert(await reconnect.isEnabled(), `${demo.route}: failed startup exposes an enabled reconnect button.`);
      assert(await frame.locator(demo.action).isDisabled(), `${demo.route}: unavailable service disables submission.`);
      assert.doesNotMatch(await frame.locator('body').innerText(), /reload (?:the )?demo/i, `${demo.route}: recovery does not ask for a reload.`);
      let drawing;
      let settings;
      if (demo.endpoint !== 'digit-generator') {
        await frame.locator('#pad').scrollIntoViewIfNeeded();
        const pad = await frame.locator('#pad').boundingBox();
        await page.mouse.move(pad.x + pad.width * .25, pad.y + pad.height * .25);
        await page.mouse.down();
        await page.mouse.move(pad.x + pad.width * .75, pad.y + pad.height * .7, { steps: 14 });
        await page.mouse.up();
        drawing = await frame.locator('#pad').evaluate(canvas => canvas.toDataURL());
        assert(await frame.locator('#drawing-prompt').isHidden(), `${demo.route}: offline drawing remains possible.`);
      } else {
        settings = await frame.locator('#seed-input, #cluster-select, #grid-select, #dim-select, #value-slider').evaluateAll(inputs => inputs.map(input => input.value));
      }
      await page.screenshot({ path: path.join(artifactDir, `${demo.route}-unavailable.png`) });
      // The wrapper may navigate its iframe while initially attaching it to the
      // site frame. Measure the retry itself, after startup and input settle.
      const loadsBeforeReconnect = frameLoads;
      const documentTimeOrigin = await frame.evaluate(() => performance.timeOrigin);
      unavailable = false;
      await reconnect.click();
      await frame.locator('#health-pill[data-state="ok"]').waitFor();
      if (drawing) {
        assert.equal(await frame.locator('#pad').evaluate(canvas => canvas.toDataURL()), drawing, `${demo.route}: reconnect preserves the exact drawing bitmap.`);
        assert(await frame.locator(demo.action).isEnabled(), `${demo.route}: preserved drawing is ready to submit.`);
      } else {
        await frame.locator('.digit-cell').first().waitFor();
        assert.equal(generations, 1, 'Digit reconnect generates the first collection once.');
        assert.deepEqual(await frame.locator('#seed-input, #cluster-select, #grid-select, #dim-select, #value-slider').evaluateAll(inputs => inputs.map(input => input.value)), settings, 'Digit reconnect preserves all generation settings.');
        const images = await frame.locator('.digit-cell img').evaluateAll(nodes => nodes.map(image => image.src));
        unavailable = true;
        await frame.locator('#health-pill').click();
        await frame.locator('#health-pill[data-state="err"]').waitFor();
        assert.deepEqual(await frame.locator('.digit-cell img').evaluateAll(nodes => nodes.map(image => image.src)), images, 'Failed reconnect preserves the last collection.');
        unavailable = false;
        await frame.locator('#health-pill').click();
        await frame.locator('#health-pill[data-state="ok"]').waitFor();
        assert.equal(generations, 1, 'Reconnecting an existing collection does not regenerate or replace it.');
        assert.deepEqual(await frame.locator('.digit-cell img').evaluateAll(nodes => nodes.map(image => image.src)), images, 'Successful reconnect preserves the last collection.');
      }
      assert.equal(frameLoads, loadsBeforeReconnect, `${demo.route}: retry never reloads the demo document.`);
      assert.equal(await frame.evaluate(() => performance.timeOrigin), documentTimeOrigin, `${demo.route}: retry retains the same document lifetime.`);
      assert.deepEqual(errors, [], `${demo.route}: no uncaught app errors.`);
      await page.screenshot({ path: path.join(artifactDir, `${demo.route}-recovered.png`) });
      console.log(`Cloud reconnect passed: ${demo.route}`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `${demo.route}-failure.png`) }).catch(() => {});
      throw error;
    } finally { await context.close(); }
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cloud-demo-reconnect-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runCloudReconnectChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-cloud-demo-reconnect') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runCloudReconnectChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
