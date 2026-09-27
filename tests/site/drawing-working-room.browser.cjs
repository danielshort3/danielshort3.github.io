'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const demos = [
  { id: 'handwritingRating', route: '/handwriting-rating-demo', action: '#rate' },
  { id: 'shapeClassifier', route: '/shape-demo', action: '#classify' }
];

async function drawingPixels(frame) {
  return frame.locator('#pad').evaluate(canvas => {
    const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let white = 0;
    for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 240 && pixels[i + 3] > 240) white += 1;
    return white / (pixels.length / 4);
  });
}

async function runDrawingWorkingRoomChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const demo of demos) {
    for (const [width, height] of [[1440, 900], [1366, 768], [390, 844], [320, 740]]) {
      const label = `${demo.id}-${width}x${height}`;
      const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce', serviceWorkers: 'block' });
      const page = await context.newPage();
      page.setDefaultTimeout(15000);
      const errors = [];
      const operations = [];
      page.on('pageerror', error => errors.push(error.message));
      await context.route('**/*', async route => {
        const url = new URL(route.request().url());
        if (url.origin !== base) return route.abort();
        if (url.pathname.startsWith('/api/demos/')) {
          operations.push(url.pathname);
          if (url.pathname.endsWith('/score')) return route.fulfill({ json: { scores: [.01, .01, .01, .01, .91, .01, .01, .01, .01, .01] } });
          if (url.pathname.endsWith('/predict')) return route.fulfill({ json: { shape: 'circle', confidence: .91, shape_scores: { circle: .91, triangle: .03, square: .03, hexagon: .02, octagon: .01 } } });
          return route.fulfill({ json: { status: 'ok', model_loaded: true } });
        }
        return route.continue();
      });
      try {
        await page.goto(`${base}/portfolio/${demo.id}`, { waitUntil: 'networkidle' });
        if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
        await page.evaluate(() => document.fonts.ready);
        assert.equal(await page.locator('.project-demo-header').count(), 0, `${label}: no duplicate drawing introduction.`);
        assert.equal(await page.locator('.project-intro-action--demo').count(), 1, `${label}: one launch action.`);
        assert.equal(await page.locator('.project-demo-open, .project-demo-mobile-launch .btn-primary').count(), 0, `${label}: no repeated launch controls.`);
        assert(await page.locator('[data-page-masthead-actions] .project-demo-help-trigger').isVisible(), `${label}: instructions remain available.`);
        await page.screenshot({ path: path.join(artifactDir, `${label}-project.png`) });
        if (width < 769) {
          assert.equal(operations.length, 0, `${label}: mobile preview does not load the drawing service.`);
          await page.locator('.project-intro-action--demo').click();
          await page.waitForURL(`**${demo.route}`);
        }
        const iframe = page.locator(width < 769 ? '.project-demo-wrapper-iframe' : '.project-embed-frame');
        const frame = await (await iframe.elementHandle()).contentFrame();
        await frame.locator('#health-pill[data-state="ok"]').waitFor();
        await frame.evaluate(() => document.fonts.ready);
        await page.waitForTimeout(200);
        const canvas = await frame.locator('#pad').boundingBox();
        const action = await frame.locator(demo.action).boundingBox();
        assert.equal(await frame.locator('.drawing-workspace > .drawing-status, .drawing-workspace > .drawing-input').count(), 2, `${label}: both drawing demos use the same status and input structure.`);
        assert.equal(await frame.locator('.drawing-input-header, .drawing-primary-action, .drawing-canvas-frame').count(), 3, `${label}: both drawing demos share the heading, canvas, and action pattern.`);
        assert(Math.abs(canvas.width - canvas.height) <= 1, `${label}: canvas is square.`);
        assert(Math.abs((canvas.x + canvas.width / 2) - (action.x + action.width / 2)) <= 2, `${label}: the primary action is centered on the drawing canvas.`);
        const input = await frame.locator('.shape-input, .handwriting-input').first().boundingBox();
        assert(input && Math.abs((canvas.x + canvas.width / 2) - (input.x + input.width / 2)) <= 2, `${label}: the drawing canvas is centered within its input column.`);
        assert(action.height >= 44 && action.width >= 44, `${label}: primary action remains a touch target.`);
        if (width > 768) {
          assert(Math.abs(canvas.width - (height <= 800 ? 258 : 298)) <= 2, `${label}: host viewport selects 260/300px canvas frame.`);
          const viewport = await page.locator('.site-frame__viewport').boundingBox();
          const dock = await page.locator('.project-question-dock').boundingBox();
          assert(action.y + action.height <= viewport.y + viewport.height - 8, `${label}: initial view includes the entire primary action.`);
          assert(dock.height >= 56 && dock.height <= 62, `${label}: question dock is compact without shrinking its button.`);
          assert(dock.y >= viewport.y + viewport.height - 1, `${label}: question dock does not cover project content.`);
          const status = await frame.locator('.drawing-status').boundingBox();
          const inputHeader = await frame.locator('.drawing-input-header, .handwriting-input-header').boundingBox();
          assert(Math.abs(status.y - inputHeader.y) <= 1, `${label}: status and input heading align.`);
          if (demo.id === 'handwritingRating') {
            const samples = await frame.locator('.handwriting-sample-buttons').boundingBox();
            assert(samples.y + samples.height <= viewport.y + viewport.height - 8, `${label}: all ten sample choices fit above the question dock at initial scroll.`);
          }
        } else if (width === 320) {
          const dock = await page.locator('.mobile-section-nav:not([hidden])').boundingBox();
          assert(action.y + action.height <= dock.y - 8, `${label}: the primary action clears the bottom dock at initial scroll.`);
        }
        for (const surface of [page, frame]) assert(await surface.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${label}: no horizontal overflow.`);
        if (demo.id === 'handwritingRating') assert.equal(await frame.locator('[data-sample-digit]:visible').count(), 10, `${label}: all digit options stay visible.`);
        await page.screenshot({ path: path.join(artifactDir, `${label}-workspace.png`) });
        await frame.locator('#pad').scrollIntoViewIfNeeded();
        const pad = await frame.locator('#pad').boundingBox();
        await page.mouse.move(pad.x + pad.width * .3, pad.y + pad.height * .25);
        await page.mouse.down();
        await page.mouse.move(pad.x + pad.width * .7, pad.y + pad.height * .75, { steps: 16 });
        await page.mouse.up();
        const before = await drawingPixels(frame);
        assert(before > .005, `${label}: actual pointer input draws visible strokes.`);
        await page.setViewportSize(width > 768 ? { width, height: height > 800 ? 768 : 900 } : { width: width === 390 ? 320 : 390, height });
        await page.waitForTimeout(250);
        const after = await drawingPixels(frame);
        assert(after > before * .7 && after < before * 1.3, `${label}: responsive canvas resizing preserves the drawing.`);
        assert(await frame.locator('#drawing-prompt').isHidden(), `${label}: resizing does not reset drawing state.`);
        assert(await frame.locator(demo.action).isEnabled(), `${label}: resizing keeps the primary action enabled.`);
        assert(operations.every(operation => /\/(health|warmup)$/.test(operation)), `${label}: layout and drawing do not submit inference.`);
        if (demo.id === 'handwritingRating') {
          await frame.locator('#rate').click();
          await frame.locator('#prediction-output:visible').waitFor();
          const meter = await frame.locator('.drawing-confidence-meter').boundingBox();
          const fill = await frame.locator('.drawing-confidence-meter > span').boundingBox();
          assert(meter.width >= 100 && fill.width >= meter.width * .75, `${label}: primary confidence meter reflects the result.`);
          await page.waitForFunction(selector => {
            const embed = document.querySelector(selector);
            return embed?.contentDocument?.querySelector('#main')?.scrollHeight <= embed?.clientHeight + 1;
          }, width < 769 ? '.project-demo-wrapper-iframe' : '.project-embed-frame');
          assert.equal(await frame.locator('#confidence-list > li').count(), 10, `${label}: inference retains all ten score rows.`);
          const scores = await frame.locator('.handwriting-score-details').boundingBox();
          const samples = await frame.locator('.handwriting-samples').boundingBox();
          assert(samples.y >= scores.y + scores.height, `${label}: expanded results do not overlap the sample controls.`);
          await page.screenshot({ path: path.join(artifactDir, `${label}-result.png`) });
          await frame.locator('[data-sample-digit="9"]').click();
          await frame.locator('[data-sample-digit="9"][aria-pressed="true"]').waitFor();
          assert(await frame.locator('#prediction-output').isHidden(), `${label}: a sample starts a new prediction without stale results.`);
          await page.screenshot({ path: path.join(artifactDir, `${label}-sample-after-result.png`) });
        } else {
          await frame.locator('#classify').click();
          await frame.locator('#prediction-output:visible').waitFor();
          const meter = await frame.locator('.drawing-confidence-meter').boundingBox();
          const fill = await frame.locator('.drawing-confidence-meter > span').boundingBox();
          assert(meter.width >= 100 && fill.width >= meter.width * .75, `${label}: primary confidence meter reflects the result.`);
          assert.equal(await frame.locator('#result-shape').innerText(), 'Circle', `${label}: shape inference still renders the selected class.`);
          assert.equal(await frame.locator('#shape-score-list > li').count(), 5, `${label}: all five shape scores remain visible.`);
          await page.screenshot({ path: path.join(artifactDir, `${label}-result.png`) });
        }
        assert.deepEqual(errors, [], `${label}: no uncaught browser errors.`);
        console.log(`Drawing working room passed: ${label}`);
      } catch (error) {
        await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
        throw error;
      } finally { await context.close(); }
    }
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'drawing-working-room-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runDrawingWorkingRoomChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-drawing-working-room') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runDrawingWorkingRoomChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
