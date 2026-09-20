'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const demos = [
  { name: 'shape', route: '/shape-demo', project: 'shapeClassifier', action: '#classify', badge: '#health-pill', endpoint: 'shape', operation: 'predict', result: '#prediction-output', success: { class: 'circle', confidence: .94, shape_confidences: { circle: .94, triangle: .01, square: .03, hexagon: .005, octagon: .015 } } },
  { name: 'handwriting', route: '/handwriting-rating-demo', project: 'handwritingRating', action: '#rate', badge: '#health-pill', endpoint: 'handwriting', operation: 'score', result: '#prediction-output', success: { digit_confidences: [.01, .01, .01, .01, .91, .01, .01, .01, .01, .01] } },
  { name: 'sentence', route: '/sentence-demo', project: 'smartSentence', action: '[name="submitBtn"]', badge: '#health', endpoint: 'smart-sentence', operation: 'rank', result: '#results', success: { top: [{ sentence: 'Alice was beginning to get very tired of sitting by her sister on the bank.', score: .93 }, { sentence: 'The Rabbit took a watch out of its waistcoat-pocket.', score: .78 }] } }
];

async function assertLayout(frame, demo, label) {
  const layout = await frame.evaluate(drawing => {
    const rect = selector => {
      const box = document.querySelector(selector).getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width, height: box.height };
    };
    return {
      panel: rect('.demo-surface'), badge: rect('.aws-status-badge'),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      canvas: drawing ? rect('#pad') : null,
      status: drawing ? rect('.drawing-status') : null,
      inputHeader: drawing ? rect('.drawing-input-header, .handwriting-input-header') : null,
      action: rect(drawing ? '#classify, #rate' : '[name="submitBtn"]'),
      drawingTitle: drawing ? Object.fromEntries(['fontSize', 'fontWeight', 'lineHeight'].map(property => [property, getComputedStyle(document.querySelector('#drawing-title'))[property]])) : null,
      advanced: drawing ? null : rect('.sentence-advanced'), results: drawing ? null : rect('#results')
    };
  }, demo.name !== 'sentence');
  assert(layout.overflow <= 1, `${label}: demo has no horizontal overflow.`);
  assert(layout.panel.width <= (demo.name === 'sentence' ? 601 : 961), `${label}: bounded workspace.`);
  const statusRight = layout.status ? layout.status.right : layout.panel.right - 20;
  assert(Math.abs(layout.badge.right - statusRight) < 22, `${label}: AWS status aligns with its workspace gutter.`);
  assert(layout.badge.bottom < layout.action.top, `${label}: status sits above the controls.`);
  if (layout.canvas) {
    assert(layout.canvas.width <= 301, `${label}: drawing canvas stays compact.`);
    if (layout.status.left > layout.inputHeader.right) assert(Math.abs(layout.status.top - layout.inputHeader.top) <= 1, `${label}: AWS status aligns with the input heading.`);
    assert(Math.abs(layout.canvas.width - layout.canvas.height) <= 1, `${label}: drawing canvas remains square.`);
    assert(layout.action.top >= layout.canvas.bottom, `${label}: primary action is below drawing canvas.`);
    assert(Math.abs((layout.action.left + layout.action.right) / 2 - (layout.canvas.left + layout.canvas.right) / 2) < 2, `${label}: primary action is centered under canvas.`);
  } else {
    assert(layout.results.bottom <= layout.advanced.top, `${label}: Advanced settings follow results.`);
  }
  assert(await frame.locator('.header-copy').isHidden(), `${label}: outer page owns the title.`);
  return layout;
}

async function assertCanvasPixels(frame, label, drawn = false) {
  const pixels = await frame.locator('#pad').evaluate(canvas => {
    const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
    let black = 0;
    let white = 0;
    let opaque = 0;
    for (let index = 0; index < data.length; index += 4) {
      if (data[index + 3] >= 250) {
        opaque += 1;
        if (Math.max(data[index], data[index + 1], data[index + 2]) <= 5) black += 1;
        if (Math.min(data[index], data[index + 1], data[index + 2]) >= 250) white += 1;
      }
    }
    return { total: data.length / 4, black, white, opaque };
  });
  assert(pixels.opaque / pixels.total >= .99, `${label}: drawing pixels have an opaque background.`);
  assert(pixels.black / pixels.total >= (drawn ? .7 : .99), `${label}: ${drawn ? 'drawing preserves' : 'starts with'} a black canvas.`);
  if (drawn) assert(pixels.white / pixels.total >= .005, `${label}: pointer input creates clearly visible white strokes.`);
  else assert.equal(pixels.white, 0, `${label}: blank canvas has no prefilled strokes.`);
}

async function drawOnCanvas(page, frame, demo) {
  const pad = frame.locator('#pad');
  await pad.scrollIntoViewIfNeeded();
  const box = await pad.boundingBox();
  const strokes = demo.name === 'shape'
    ? [Array.from({ length: 49 }, (_, index) => {
      const angle = index / 48 * Math.PI * 2;
      return [.5 + Math.cos(angle) * .3, .5 + Math.sin(angle) * .3];
    })]
    : [[[.3, .2], [.3, .52], [.72, .52]], [[.63, .18], [.63, .8]]];
  for (const stroke of strokes) {
    await page.mouse.move(box.x + box.width * stroke[0][0], box.y + box.height * stroke[0][1]);
    await page.mouse.down();
    for (const [x, y] of stroke.slice(1)) {
      await page.mouse.move(box.x + box.width * x, box.y + box.height * y, { steps: demo.name === 'shape' ? 1 : 8 });
    }
    await page.mouse.up();
  }
  await page.mouse.move(0, 0);
}

async function assertDrawingBlankState(frame, demo, label) {
  assert(await frame.locator('#drawing-prompt').isVisible(), `${label}: the blank canvas shows Draw here.`);
  assert.equal((await frame.locator('#drawing-prompt').innerText()).trim(), 'Draw here');
  assert(await frame.locator('#result-empty').isVisible(), `${label}: the blank result has a quiet placeholder.`);
  assert.equal((await frame.locator('#result-empty').innerText()).trim(), 'Your result appears here.');
  assert(await frame.locator('#prediction-output').isHidden(), `${label}: result content stays hidden before a drawing is submitted.`);
  assert(await frame.locator(demo.action).isDisabled(), `${label}: the primary action is unavailable until a drawing exists.`);
}

async function assertShapeResult(frame, label) {
  const output = frame.locator('#prediction-output');
  assert(await output.isVisible(), `${label}: the prediction appears after classification.`);
  assert.equal(await frame.locator('#result-shape').innerText(), 'Circle');
  const modelReads = output.getByText('Model reads', { exact: true });
  assert(await modelReads.isVisible(), `${label}: the result uses the same Model reads hierarchy as handwriting.`);
  const icon = await frame.locator('#result-icon').boundingBox();
  const labelBox = await modelReads.boundingBox();
  assert(icon.width >= 40 && icon.height >= 40, `${label}: the predicted shape is a prominent visual.`);
  assert(icon.x + icon.width <= labelBox.x + 2, `${label}: the predicted shape sits beside the model result text.`);
  const scores = frame.locator('#shape-score-list');
  assert.equal(await scores.locator('.shape-row:visible').count(), 5, `${label}: every shape score is visible without a disclosure.`);
  assert.equal(await scores.evaluate(node => Boolean(node.closest('details'))), false, `${label}: shape scores are permanently expanded.`);
  assert.deepEqual(await scores.locator('.shape-name').allTextContents(), ['Circle', 'Square', 'Octagon', 'Triangle', 'Hexagon'], `${label}: scores are ranked from highest to lowest.`);
  const barWidths = await scores.locator('.shape-fill').evaluateAll(nodes => nodes.map(node => parseFloat(node.style.width)));
  assert.deepEqual(barWidths, [94, 3, 1.5, 1, .5], `${label}: all score bars use the same percentage scale.`);
  assert.equal(await frame.locator('.demo-surface details:visible').count(), 0, `${label}: processing and score disclosures do not crowd the result.`);
  const meterHeight = await frame.locator('#confidence-bar').evaluate(node => node.parentElement.getBoundingClientRect().height);
  assert(meterHeight <= 8, `${label}: primary confidence stays a thin bar.`);
}

async function runCase({ browser, base, artifactDir, demo, width, project }) {
  const label = `${demo.name}-${project ? 'project' : 'standalone'}-${width}`;
  const context = await browser.newContext({ viewport: { width, height: 1100 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(18000);
  const errors = [];
  const requests = [];
  let hold = false;
  let fail = false;
  let release;
  page.on('pageerror', error => errors.push(error.message));
  await context.route(`${base}/api/demos/${demo.endpoint}/**`, async route => {
    const operation = new URL(route.request().url()).pathname.split('/').filter(Boolean).at(-1);
    if (operation === 'health' || operation === 'warmup') {
      await route.fulfill({ json: { status: 'ok', model_loaded: true } });
      return;
    }
    if (operation !== demo.operation) {
      errors.push(`Unexpected ${demo.endpoint} operation: ${operation}`);
      await route.fulfill({ status: 400, json: { error: 'Unexpected request' } });
      return;
    }
    requests.push(route.request().postDataJSON());
    if (hold) await new Promise(resolve => { release = resolve; });
    await route.fulfill(fail ? { status: 400, json: { error: 'Controlled failure' } } : { json: demo.success });
  });
  try {
    const response = await page.goto(base + (project ? `/portfolio/${demo.project}` : demo.route), { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label}: route loads.`);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    const iframe = page.locator(project ? 'iframe.project-embed-frame' : 'iframe.project-demo-wrapper-iframe');
    await iframe.scrollIntoViewIfNeeded();
    const frame = await (await iframe.elementHandle()).contentFrame();
    await frame.locator(`${demo.badge}[data-state="ok"]`).waitFor();
    await frame.evaluate(() => document.fonts.ready);
    const initialLayout = await assertLayout(frame, demo, label);
    if (demo.name !== 'sentence') {
      await assertCanvasPixels(frame, label);
      await assertDrawingBlankState(frame, demo, label);
      await frame.locator('.demo-surface').screenshot({ path: path.join(artifactDir, `${label}-initial.png`) });
      assert.equal(requests.length, 0, `${label}: opening the demo does not submit a drawing.`);
      if (demo.name === 'shape') {
        assert.equal(await frame.locator('[data-shape-example], .drawing-samples').count(), 0, `${label}: shape samples and autofill controls are removed.`);
        assert.doesNotMatch(await frame.locator('.demo-surface').innerText(), /Load a sample/i);
        assert.equal((await frame.locator('#classify').innerText()).trim(), 'Classify shape');
        assert.equal(await frame.locator('#classify :is(i, svg)').count(), 0, `${label}: the action matches handwriting's plain button.`);
      }
      await drawOnCanvas(page, frame, demo);
      await assertCanvasPixels(frame, label, true);
      assert(await frame.locator('#drawing-prompt').isHidden(), `${label}: drawing dismisses the canvas prompt.`);
      assert(await frame.locator(demo.action).isEnabled(), `${label}: drawing enables the primary action.`);
      assert.equal(requests.length, 0, `${label}: drawing waits for the primary action before submitting.`);
    } else {
      await frame.locator('[data-query-example]').first().click();
      assert.equal(await frame.locator('#query').inputValue(), 'She wonders about things.');
      assert.equal(await frame.locator('.sentence-advanced').getAttribute('open'), null);
    }
    await frame.locator(demo.action).click();
    await frame.locator(`${demo.badge}[data-state="ok"]`).waitFor();
    assert.equal(requests.length, 1, `${label}: one request per primary action.`);
    if (demo.name === 'shape') await assertShapeResult(frame, label);
    if (demo.name === 'handwriting') {
      assert.equal(await frame.locator('#result-digit').innerText(), '4');
      assert.equal(await frame.locator('#confidence-list .confidence-row:visible').count(), 10, `${label}: all ten digit scores are visible immediately after rating.`);
      assert.equal(await frame.locator('#confidence-list').evaluate(list => Boolean(list.closest('details'))), false, `${label}: digit scores have no collapsible parent.`);
      assert.equal(await frame.locator('.handwriting-score-details summary').count(), 0, `${label}: digit scores have no disclosure control.`);
    }
    if (demo.name === 'sentence') {
      assert.equal(requests[0].top, 5);
      assert.equal(await frame.locator('#results .item').count(), 2);
    }
    const previousResult = await frame.locator(demo.result).innerText();
    hold = true;
    await frame.locator(demo.action).click();
    await frame.locator(`${demo.badge}[data-state="loading"]`).waitFor();
    assert.equal(await frame.locator(demo.result).innerText(), previousResult, `${label}: last result remains during loading.`);
    if (demo.name !== 'sentence') {
      await assertCanvasPixels(frame, `${label} pending`, true);
      await frame.locator('.demo-surface').screenshot({ path: path.join(artifactDir, `${label}-loading.png`) });
    }
    const requestDeadline = Date.now() + 5000;
    while (!release && Date.now() < requestDeadline) await new Promise(resolve => setTimeout(resolve, 20));
    assert(release, `${label}: pending request reaches the API.`);
    fail = true;
    hold = false;
    release();
    release = null;
    await frame.locator(`${demo.badge}[data-state="err"]`).waitFor();
    assert.equal(await frame.locator(demo.result).innerText(), previousResult, `${label}: failure preserves last result.`);
    if (demo.name !== 'sentence') {
      assert(await frame.locator(demo.action).isEnabled(), `${label}: a failed replacement can be retried.`);
      await assertCanvasPixels(frame, `${label} error`, true);
      await frame.locator('.demo-surface').screenshot({ path: path.join(artifactDir, `${label}-error.png`) });
    }
    fail = false;
    await frame.locator(demo.action).click();
    await frame.locator(`${demo.badge}[data-state="ok"]`).waitFor();
    await assertLayout(frame, demo, `${label} after recovery`);
    await frame.locator('.demo-surface').screenshot({ path: path.join(artifactDir, `${label}.png`) });
    if (demo.name === 'sentence') {
      await frame.locator('.sentence-advanced summary').click();
      await frame.locator('#top').fill('8');
      await frame.locator(demo.action).click();
      await frame.locator(`${demo.badge}[data-state="ok"]`).waitFor();
      assert.equal(requests.at(-1).top, 8, `${label}: advanced setting stays connected to the search form.`);
    } else {
      await frame.locator('#erase').click();
      await assertCanvasPixels(frame, `${label} after Clear`);
      await assertDrawingBlankState(frame, demo, `${label} after Clear`);
      await frame.locator('.demo-surface').screenshot({ path: path.join(artifactDir, `${label}-cleared.png`) });
    }
    assert.deepEqual(errors, [], `${label}: no uncaught app errors.`);
    console.log(`Compact project flow passed: ${label}`);
    return initialLayout;
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
    error.message = `${label}: ${error.message}`;
    throw error;
  } finally {
    release?.();
    await context.close();
  }
}

async function runCompactProjectLayoutChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const drawingDimensions = new Map();
  const dimensionMismatches = [];
  for (const demo of demos) {
    for (const [width, project] of [[1440, false], [390, false], [320, false], [1440, true]]) {
      const layout = await runCase({ browser, base, artifactDir, demo, width, project });
      const key = `${width}-${project ? 'project' : 'standalone'}`;
      if (demo.name === 'shape') drawingDimensions.set(key, layout);
      if (demo.name === 'handwriting') {
        const shapeLayout = drawingDimensions.get(key);
        const shape = shapeLayout.canvas;
        const canvas = layout.canvas;
        if (Math.abs(shape.width - canvas.width) > 1 || Math.abs(shape.height - canvas.height) > 1) {
          dimensionMismatches.push(`${key}: shape ${shape.width}×${shape.height}px; handwriting ${canvas.width}×${canvas.height}px`);
        }
        assert(Math.abs(shapeLayout.action.width - layout.action.width) <= 2 && Math.abs(shapeLayout.action.height - layout.action.height) <= 2, `${key}: both drawing actions share the same button size.`);
        assert(Math.abs((shapeLayout.action.top - shape.bottom) - (layout.action.top - canvas.bottom)) <= 2, `${key}: both drawing actions share the same spacing beneath the canvas.`);
        assert(Math.abs(shapeLayout.badge.height - layout.badge.height) <= 2, `${key}: both drawing headers use the same status badge height.`);
        assert(Math.abs((shape.top - shapeLayout.panel.top) - (canvas.top - layout.panel.top)) <= 2, `${key}: both drawing canvases begin at the same offset within the workspace.`);
        assert.deepEqual(shapeLayout.drawingTitle, layout.drawingTitle, `${key}: drawing labels use the same typography.`);
      }
    }
  }
  assert.deepEqual(dimensionMismatches, [], 'Shape and handwriting use matching drawing canvas dimensions.');
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'compact-project-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runCompactProjectLayoutChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-compact-projects') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runCompactProjectLayoutChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
