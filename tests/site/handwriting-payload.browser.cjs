/**
 * Run after npm run build: node tests/site/handwriting-payload.browser.cjs
 * Exercises the built wrapper and demo with only handwriting API responses stubbed.
 * Supports PLAYWRIGHT_MODULE and BROWSER_EXECUTABLE_PATH like the browser smoke gate.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const { _internal: proxy } = require('../../api/_lib/demo-proxy');

const root = path.resolve(__dirname, '../..');
const scoreLimit = proxy.resolveDemoRoute(['handwriting', 'score'], 'POST', []).route.maxBodyBytes;
const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

async function assertScoringImage(frame, request, label) {
  assert(request.bytes <= scoreLimit, `${label} exceeds the proxy's ${scoreLimit}-byte body budget (${request.bytes}).`);
  const payload = JSON.parse(request.body);
  const imageFields = ['b64', 'image', 'img', 'data'].filter(key => typeof payload[key] === 'string');
  assert.equal(imageFields.length, 1, `${label} sends exactly one supported image field.`);
  const b64 = payload[imageFields[0]];
  const png = Buffer.from(b64.replace(/^data:[^,]+,/, ''), 'base64');
  assert(png.subarray(0, 8).equals(pngSignature), `${label} sends a PNG image.`);
  // Decoding checks the actual encoded image, not just its file signature.
  const dimensions = await frame.evaluate(async (encoded) => {
    const raw = atob(encoded.replace(/^data:[^,]+,/, ''));
    const bytes = Uint8Array.from(raw, value => value.charCodeAt(0));
    const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const result = { width: bitmap.width, height: bitmap.height };
    bitmap.close();
    return result;
  }, b64);
  assert(dimensions.width > 0 && dimensions.height > 0, `${label} decodes to a nonempty image.`);
  assert(dimensions.width <= 256 && dimensions.height <= 256, `${label} keeps the inference image within 256 pixels.`);
  return b64;
}

async function assertWorkspaceLayout(page, frame, label) {
  const pad = await frame.locator('#pad').boundingBox();
  const rate = await frame.locator('#rate').boundingBox();
  assert(pad && rate, `${label} shows the drawing pad and Rate digit button.`);
  assert(Math.abs((pad.x + pad.width / 2) - (rate.x + rate.width / 2)) <= 1, `${label} centers Rate digit below the drawing pad.`);
  assert(rate.y >= pad.y + pad.height, `${label} places Rate digit below the drawing pad.`);
  for (const [name, surface] of [['wrapper', page], ['demo', frame]]) {
    const overflow = await surface.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    assert(overflow <= 1, `${label} ${name} has no horizontal overflow (${overflow}px).`);
  }
}

async function assertRankedScores(frame, expectedScores, label) {
  const rows = await frame.locator('#confidence-list .confidence-row').evaluateAll(elements => elements.map(row => ({
    digit: Number(row.dataset.digit),
    score: Number(row.dataset.score),
    digitLabel: row.querySelector('.confidence-digit')?.textContent.trim(),
    percentage: row.querySelector('.confidence-pct')?.textContent.trim(),
    barWidth: parseFloat(row.querySelector('.confidence')?.style.width)
  })));
  const expected = Object.entries(expectedScores)
    .map(([digit, score]) => ({ digit: Number(digit), score: score * 100 }))
    .sort((left, right) => right.score - left.score || left.digit - right.digit);
  assert.equal(rows.length, 10, `${label} renders all ten digit scores, including zeroes.`);
  assert.deepEqual(rows.map(row => row.digit), expected.map(row => row.digit), `${label} ranks guesses by score with digit order breaking ties.`);
  rows.forEach((row, index) => {
    const { digit, score } = expected[index];
    assert.equal(row.digitLabel, String(digit), `${label} labels digit ${digit}.`);
    assert(Math.abs(row.score - score) < 0.0001, `${label} preserves digit ${digit}'s numerical score.`);
    assert(Math.abs(row.barWidth - score) < 0.0001, `${label} draws digit ${digit}'s bar on the same 0–100 scale.`);
    const percentage = score > 0 && score < 0.1 ? '<0.1%' : `${score.toFixed(1)}%`;
    assert.equal(row.percentage, percentage, `${label} distinguishes tiny positive scores from zero.`);
  });
}

async function drawDigit(page, frame) {
  const pad = frame.locator('#pad');
  await pad.scrollIntoViewIfNeeded();
  const box = await pad.boundingBox();
  assert(box && box.width > 0 && box.height > 0, 'The drawing pad is visible.');
  const points = [[0.3, 0.3], [0.5, 0.23], [0.7, 0.35], [0.65, 0.47], [0.32, 0.73], [0.72, 0.73]];
  await page.mouse.move(box.x + box.width * points[0][0], box.y + box.height * points[0][1]);
  await page.mouse.down();
  for (const [x, y] of points.slice(1)) {
    await page.mouse.move(box.x + box.width * x, box.y + box.height * y, { steps: 5 });
  }
  await page.mouse.up();
}

async function runViewport(browser, base, settings) {
  const context = await browser.newContext({ ...settings.context, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const requests = [];
  const errors = [];
  let activeLabel = 'load';
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.url().startsWith(base) && response.status() >= 400) {
      errors.push(`${response.status()} ${response.url()}`);
    }
  });
  await context.route(`${base}/api/demos/handwriting/**`, async (route) => {
    const request = route.request();
    const action = new URL(request.url()).pathname.split('/').filter(Boolean).at(-1);
    let body;
    if (action === 'health' || action === 'warmup') {
      body = { status: action === 'health' ? 'ok' : 'ready', model_loaded: true };
    } else if (action === 'score' && request.method() === 'POST') {
      const raw = request.postDataBuffer();
      const prediction = requests.length % 10;
      const probabilities = [0.91, 0.04, 0.04, 0.0094, 0.0005, 0.0001, 0, 0, 0, 0];
      const scores = Object.fromEntries(probabilities.map((probability, index) => [(prediction + index) % 10, probability]));
      requests.push({ body: raw.toString('utf8'), bytes: raw.length, prediction, scores, label: activeLabel });
      body = { digit_confidences: scores };
    } else {
      errors.push(`Unexpected handwriting request: ${request.method()} ${request.url()}`);
      await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: 'Unexpected test request.' }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });

  try {
    const response = await page.goto(`${base}/handwriting-rating-demo`, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, 'The handwriting wrapper route loads.');
    assert.match(await page.title(), /Handwriting.*Demo/);
    const iframe = page.locator('iframe.project-demo-wrapper-iframe');
    await iframe.waitFor({ state: 'visible' });
    const frame = await (await iframe.elementHandle()).contentFrame();
    await frame.locator('#health-pill[data-state="ok"]').waitFor({ state: 'visible' });
    assert(await frame.locator('#rate').isDisabled(), 'Rate digit waits for drawing or sample input.');
    assert(await frame.locator('#prediction-output').isHidden(), 'Scores stay hidden until a prediction is available.');
    assert.match(frame.url(), /\/demos\/handwriting-rating-demo\.html/);
    assert.equal(await frame.title(), 'Handwriting Rating Demo');
    const sampleButtons = frame.locator('button[data-sample-digit]');
    assert.deepEqual(await sampleButtons.evaluateAll(buttons => buttons.map(button => button.dataset.sampleDigit)), Array.from({ length: 10 }, (_, digit) => String(digit)));
    assert.equal(await frame.locator('select').count(), 0, 'All sample choices are visible without a dropdown.');
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await frame.evaluate(() => document.fonts.ready);
    await assertWorkspaceLayout(page, frame, `${settings.name} initial`);

    const score = async (label, actualDigit) => {
      activeLabel = label;
      const before = requests.length;
      const scored = page.waitForResponse(result => result.url() === `${base}/api/demos/handwriting/score` && result.request().method() === 'POST');
      await frame.locator('#rate').click();
      assert.equal((await scored).status(), 200, `${label} receives its scoring response.`);
      await frame.locator('#rate:not([disabled])').waitFor();
      assert.equal(requests.length, before + 1, `${label} submits once.`);
      const request = requests.at(-1);
      await frame.waitForFunction(prediction => document.getElementById('result-digit')?.textContent === String(prediction), request.prediction);
      assert(await frame.locator('#prediction-output').isVisible(), `${label} reveals the prediction.`);
      await assertRankedScores(frame, request.scores, label);
      assert.equal(await frame.locator('#confidence-text').textContent(), '91.0% confidence');
      if (actualDigit === null) {
        assert(await frame.locator('#result-actual').isHidden(), `${label} clears the sample's actual digit.`);
      } else {
        assert.equal(await frame.locator('#result-actual').textContent(), `Sample ${actualDigit}`);
        assert(await frame.locator('#result-actual').isVisible());
      }
      return assertScoringImage(frame, request, `${settings.name} ${label}`);
    };

    const sampleImages = [];
    for (let digit = 0; digit < 10; digit += 1) {
      await frame.locator(`button[data-sample-digit="${digit}"]`).click();
      await frame.waitForFunction(value => document.getElementById('sample-status')?.textContent.startsWith(`Sample ${value} loaded.`), digit);
      assert.equal(await frame.locator('button[data-sample-digit][aria-pressed="true"]').count(), 1, `Sample ${digit} is the only selected sample.`);
      assert.equal(await frame.locator(`button[data-sample-digit="${digit}"]`).getAttribute('aria-pressed'), 'true');
      sampleImages.push(await score(`sample-${digit}`, digit));
    }
    assert.equal(new Set(sampleImages).size, 10, 'Every selected sample sends its own distinct image.');
    await assertWorkspaceLayout(page, frame, `${settings.name} scored`);

    await frame.locator('#erase').click();
    assert(await frame.locator('#prediction-output').isHidden(), 'Clear hides a scored sample result.');
    assert(await frame.locator('#rate').isDisabled(), 'Clear disables Rate digit until new input.');
    assert.equal(await frame.locator('button[data-sample-digit][aria-pressed="true"]').count(), 0, 'Clear deselects the active sample button.');
    await frame.locator('button[data-sample-digit="9"]').click();
    await frame.waitForFunction(() => document.getElementById('sample-status')?.textContent.startsWith('Sample 9 loaded.'));
    await drawDigit(page, frame);
    assert.equal(await frame.locator('button[data-sample-digit][aria-pressed="true"]').count(), 0, 'Drawing replaces the selected sample.');
    const drawing = await score('draw-after-sample', null);
    assert(!sampleImages.includes(drawing), 'Drawing submits newly drawn content.');

    await frame.locator('#erase').click();
    assert(await frame.locator('#prediction-output').isHidden(), 'Clear hides the previous result.');
    assert(await frame.locator('#rate').isDisabled(), 'Clear disables Rate digit until new input.');
    assert.equal(await frame.locator('button[data-sample-digit][aria-pressed="true"]').count(), 0, 'Clear deselects sample buttons.');
    await drawDigit(page, frame);
    await score('draw-after-clear', null);
    assert.deepEqual(errors, [], 'The wrapper and handwriting flow produce no page or local HTTP errors.');
    const maximum = Math.max(...requests.map(request => request.bytes));
    console.log(`${settings.name}: ${requests.length} scoring requests passed; largest ${maximum}/${scoreLimit} bytes.`);
  } catch (error) {
    const screenshot = path.join(os.tmpdir(), `handwriting-payload-${settings.name}-failure.png`);
    await page.screenshot({ path: screenshot, fullPage: false }).catch(() => {});
    error.message = `${settings.name} ${activeLabel}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function main() {
  assert(fs.existsSync(path.join(root, 'public/pages/demos/handwriting-rating-demo.html')), 'Run npm run build before this browser regression.');
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'handwriting-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const settings of [
      { name: 'desktop-dpr1', context: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 } },
      { name: 'mobile-dpr3', context: { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true } }
    ]) await runViewport(browser, base, settings);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
