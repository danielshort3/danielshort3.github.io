'use strict';

const { chromium } = require('playwright');
const assert = require('assert');
const fs = require('fs');
const path = require('path');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const results = [];
  for (const width of [736, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1050 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('http://127.0.0.1:4187/', { waitUntil: 'networkidle' });
    const frame = page.frames().find((entry) => entry.parentFrame());
    await frame.waitForSelector('[data-ready="true"]');
    const root = frame.locator('#starfall-sample-review');
    for (const sample of ['oracle', 'fox']) {
      await root.locator('[data-sample]').selectOption(sample);
      if (await root.locator('[data-play]').innerText() === 'Pause') await root.locator('[data-play]').click();
      const expectedFrames = sample === 'oracle' ? [0, 0, 1, 2, 2] : [0, 1, 2, 4, 5];
      const values = [0, 200, 450, 750, 999];
      const frames = [];
      for (const value of values) {
        await root.locator('[data-scrub]').evaluate((input, next) => { input.value = String(next); input.dispatchEvent(new Event('input', { bubbles: true })); }, value);
        const indices = await root.locator('canvas').evaluateAll((elements) => elements.map((canvas) => Number(canvas.dataset.frame)));
        assert.strictEqual(indices[0], indices[1], 'Before/after frames must be synchronized');
        frames.push(indices[0]);
      }
      assert.deepStrictEqual(frames, expectedFrames, `${sample} must use current game hold timing`);
      const paused = await root.locator('[data-before]').getAttribute('data-frame');
      await page.waitForTimeout(230);
      assert.strictEqual(await root.locator('[data-before]').getAttribute('data-frame'), paused, 'Pause must keep current frame');
      await root.locator('[data-step]').click();
      assert.strictEqual(await root.locator('[data-before]').getAttribute('data-frame'), '0', 'Next frame wraps to the first frame');
      await root.locator('[data-facing]').check();
      await root.locator('[data-zoom]').selectOption('1');
      await root.locator('[data-speed]').selectOption('0.5');
      await root.locator('[data-guides]').uncheck();
      await root.locator('[data-play]').click();
      const start = Number(await root.locator('[data-scrub]').inputValue());
      await page.waitForTimeout(280);
      assert.notStrictEqual(Number(await root.locator('[data-scrub]').inputValue()), start, 'Playing must advance loop position');
      await root.locator('[data-play]').click();
      await root.locator('[data-facing]').uncheck();
      await root.locator('[data-zoom]').selectOption('2');
      await root.locator('[data-speed]').selectOption('1');
      await root.locator('[data-guides]').check();
      await root.locator('[data-scrub]').evaluate((input) => { input.value = '850'; input.dispatchEvent(new Event('input', { bubbles: true })); });
      await page.screenshot({ path: path.join(__dirname, `${sample}-${width}.png`), fullPage: true });
      results.push({ viewport: width, sample, frameSampling: frames, controls: 'passed' });
    }
    assert.deepStrictEqual(errors, [], 'Preview must not throw page errors');
    const overflow = await frame.evaluate(() => ({ width: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
    assert(overflow.scrollWidth <= overflow.width, 'Preview must not overflow horizontally');
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(__dirname, 'review-verification.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
})().catch((error) => { console.error(error); process.exitCode = 1; });
