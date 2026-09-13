/**
 * Exercises the built digit generator with only its AWS API responses stubbed.
 * Run after npm run build: node tests/site/digit-generator-simple.browser.cjs
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function digitFixtures() {
  return Promise.all(Array.from({ length: 6 }, async (_, index) => {
    const png = await sharp(Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><rect width="28" height="28" fill="black"/><g transform="rotate(${index * 2 - 5} 14 14)"><path d="M ${9 + index % 2} 5 L 6 16 L 21 16 M 17 6 L 16 24" fill="none" stroke="white" stroke-width="${1.8 + index * .13}" stroke-linecap="round" stroke-linejoin="round"/></g></svg>`)).png().toBuffer();
    return png.toString('base64');
  }));
}

async function assertLayout(page, frame, expectedSize, label) {
  await frame.evaluate(() => document.fonts.ready);
  const layout = await frame.evaluate(() => {
    const box = selector => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width };
    };
    const cells = Array.from(document.querySelectorAll('.digit-cell'), cell => {
      const rect = cell.getBoundingClientRect();
      return { x: Math.round(rect.x), y: Math.round(rect.y), width: rect.width, height: rect.height };
    });
    const toolbar = document.querySelector('.generation-toolbar');
    let surface = toolbar;
    let background = getComputedStyle(surface).backgroundColor;
    while (background === 'rgba(0, 0, 0, 0)' && surface.parentElement) {
      surface = surface.parentElement;
      background = getComputedStyle(surface).backgroundColor;
    }
    return {
      controls: [box('#cluster-select'), box('#refresh-seed-btn')],
      panel: box('.generation-panel'), badge: box('#health-pill'),
      advanced: box('.generation-settings'), grid: box('#grid'),
      background, cells,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      labels: Array.from(toolbar.querySelectorAll('label'), label => label.textContent.trim())
    };
  });
  assert.equal(layout.cells.length, expectedSize * expectedSize, `${label} displays the selected number of images.`);
  assert.equal(new Set(layout.cells.map(cell => cell.x)).size, expectedSize, `${label} keeps exactly ${expectedSize} columns.`);
  assert.equal(new Set(layout.cells.map(cell => cell.y)).size, expectedSize, `${label} keeps exactly ${expectedSize} rows.`);
  assert(layout.cells.every(cell => Math.abs(cell.width - cell.height) <= 1), `${label} keeps square image tiles.`);
  const controlsLeft = Math.min(...layout.controls.map(box => box.left));
  const controlsRight = Math.max(...layout.controls.map(box => box.right));
  const gridCenter = (layout.grid.left + layout.grid.right) / 2;
  assert(Math.abs((controlsLeft + controlsRight) / 2 - gridCenter) <= 2, `${label} centers the primary controls over the grid.`);
  assert(layout.panel.width <= 601, `${label} keeps the workspace within its 600px maximum width.`);
  assert(layout.badge.bottom <= Math.min(...layout.controls.map(box => box.top)), `${label} reserves a status row above the primary controls.`);
  assert(layout.badge.right <= layout.panel.right && layout.badge.right >= layout.panel.right - 32, `${label} aligns AWS status at the top right of the workspace.`);
  assert(Math.max(...layout.controls.map(box => box.bottom)) <= layout.grid.top + 1, `${label} places the grid directly below the primary controls.`);
  assert(layout.grid.bottom <= layout.advanced.top + 1, `${label} places advanced settings below the generated grid.`);
  assert.equal(layout.background, 'rgb(255, 255, 255)', `${label} uses a white primary control area.`);
  assert.equal(layout.labels.length, 1, `${label} exposes only Digit as a primary setting.`);
  assert(layout.overflow <= 1, `${label} demo has no horizontal overflow (${layout.overflow}px).`);
  const wrapperOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert(wrapperOverflow <= 1, `${label} page has no horizontal overflow (${wrapperOverflow}px).`);
}

async function assertAwsBadge(frame, expected, label) {
  const badge = frame.locator('#health-pill');
  assert(await badge.isVisible(), `${label} shows the AWS connectivity badge.`);
  assert.equal((await badge.innerText()).replace(/\s+/g, ' ').trim(), `AWS · ${expected}`, `${label} reports the current AWS state.`);
  const states = { Connecting: 'warming', Connected: 'ok', Generating: 'loading', 'Request failed': 'err', Unavailable: 'err' };
  assert.equal(await badge.getAttribute('data-state'), states[expected], `${label} applies the matching connectivity appearance.`);
  assert.equal(await frame.locator('.generation-panel #health-pill').count(), 1, `${label} keeps one connectivity badge inside the workspace.`);
}

async function stableWorkspaceLayout(page, frame) {
  // Isolate status layout changes from the site's existing button hover lift.
  await page.mouse.move(0, 0);
  await frame.locator('#refresh-seed-btn').evaluate(button => Promise.allSettled(button.getAnimations().map(animation => animation.finished)));
  return frame.evaluate(() => {
    const panel = document.querySelector('.generation-panel').getBoundingClientRect();
    const boxes = Object.fromEntries(['.generation-toolbar', '#cluster-select', '#refresh-seed-btn', '#grid', '.generation-settings'].map(selector => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return [selector, { left: rect.left - panel.left, top: rect.top - panel.top, width: rect.width, height: rect.height }];
    }));
    const badge = document.querySelector('#health-pill').getBoundingClientRect();
    return { boxes, badge: { right: panel.right - badge.right, top: badge.top - panel.top, height: badge.height }, images: Array.from(document.querySelectorAll('.digit-cell img'), image => image.src) };
  });
}

async function assertStableWorkspace(page, frame, before, label, { includeOutput = true, preserveImages = true } = {}) {
  const after = await stableWorkspaceLayout(page, frame);
  for (const [selector, box] of Object.entries(before.boxes)) {
    if (!includeOutput && (selector === '#grid' || selector === '.generation-settings')) continue;
    for (const [dimension, value] of Object.entries(box)) {
      assert(Math.abs(after.boxes[selector][dimension] - value) <= 1, `${label} keeps ${selector} ${dimension} steady as AWS status changes (${value} to ${after.boxes[selector][dimension]}).`);
    }
  }
  for (const [dimension, value] of Object.entries(before.badge)) {
    assert(Math.abs(after.badge[dimension] - value) <= 1, `${label} keeps the badge ${dimension} steady as its text changes.`);
  }
  if (preserveImages) assert.deepEqual(after.images, before.images, `${label} preserves the existing generated images while awaiting a usable response.`);
}

async function runCase({ browser, base, artifactDir, fixtures, width, project }) {
  const label = `${project ? 'project' : 'standalone'}-${width}`;
  const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1000 }, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const errors = [];
  const requests = [];
  const apiRequests = [];
  let downloads = 0;
  let failGeneration = false;
  let releaseHealth;
  const healthGate = !project && width === 1440 ? new Promise(resolve => { releaseHealth = resolve; }) : Promise.resolve();
  let generationGate = null;
  let releaseGeneration;
  page.on('pageerror', error => errors.push(error.message));
  page.on('download', () => { downloads += 1; });
  await context.route(`${base}/api/demos/digit-generator/**`, async route => {
    const request = route.request();
    const action = new URL(request.url()).pathname.split('/').filter(Boolean).at(-1);
    apiRequests.push(action);
    let body;
    if (action === 'health' || action === 'warmup') {
      if (action === 'health') await healthGate;
      body = { status: action === 'health' ? 'ok' : 'ready', model_loaded: true };
    } else if (action === 'generate' && request.method() === 'POST') {
      const payload = request.postDataJSON();
      requests.push(payload);
      if (generationGate) await generationGate;
      if (failGeneration) {
        await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Fixture temporarily unavailable.' }) });
        return;
      }
      body = { ...payload, latent_dim: 20, images: Array.from({ length: payload.rows }, (_, row) => Array.from({ length: payload.cols }, (_, col) => fixtures[(row + col) % fixtures.length])) };
    } else {
      errors.push(`Unexpected digit request: ${request.method()} ${request.url()}`);
      await route.fulfill({ status: 404, contentType: 'application/json', body: '{}' });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  try {
    const response = await page.goto(base + (project ? '/portfolio/digitGenerator' : '/digit-generator-demo'), { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label} route loads.`);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    if (project) {
      const heading = page.locator('.project-demo-header');
      await heading.scrollIntoViewIfNeeded();
      assert.equal(await heading.locator('.project-demo-title').innerText(), 'Digit Generator');
      assert.equal(await heading.locator('.project-demo-description').innerText(), 'Choose a digit and generate a collection of handwritten variations.');
      if (width >= 600) assert(await heading.locator('.project-demo-open').isVisible(), `${label} preserves the full-demo link beside the unified heading.`);
      const headingBox = await heading.locator('.project-demo-heading').boundingBox();
      const actionsBox = await heading.locator('.project-demo-header-actions').boundingBox();
      assert(actionsBox.y < headingBox.y + headingBox.height, `${label} keeps heading actions beside the copy instead of creating an empty row.`);
      await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-project-heading.png`) });
    }
    if (project && width < 600) {
      const launch = page.locator('.project-demo-mobile-launch .btn-primary');
      await launch.scrollIntoViewIfNeeded();
      assert(await launch.isVisible(), 'The compact project view offers the full demo.');
      await launch.click();
      await page.waitForURL('**/digit-generator-demo');
    }
    const iframe = page.locator(project && width >= 600 ? 'iframe.project-embed-frame' : 'iframe.project-demo-wrapper-iframe');
    await iframe.scrollIntoViewIfNeeded();
    const frame = await (await iframe.elementHandle()).contentFrame();
    let connectingLayout;
    if (releaseHealth) {
      await frame.locator('#health-pill').waitFor({ state: 'visible' });
      await assertAwsBadge(frame, 'Connecting', `${label} startup`);
      assert(await frame.locator('#refresh-seed-btn').isDisabled(), 'Connecting waits for the existing health and warmup requests.');
      assert.equal(requests.length, 0, 'No grid is generated before AWS is ready.');
      await frame.evaluate(() => document.fonts.ready);
      connectingLayout = await stableWorkspaceLayout(page, frame);
      await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-connecting.png`) });
      releaseHealth();
    }
    await frame.waitForFunction(() => document.querySelectorAll('#grid .digit-cell img').length === 36 && document.querySelector('#status')?.textContent === 'Grid updated.');
    await assertAwsBadge(frame, 'Connected', `${label} initial`);
    if (connectingLayout) await assertStableWorkspace(page, frame, connectingLayout, `${label} connection ready`, { includeOutput: false, preserveImages: false });
    const statusBox = await frame.locator('#status').boundingBox();
    assert(statusBox.width <= 1 && statusBox.height <= 1, `${label} keeps detailed live feedback visually unobtrusive.`);
    assert(!await frame.locator('header.card-header').isVisible(), `${label} does not repeat the heading inside the embedded generator.`);
    assert.equal(requests[0].rows, 6, `${label} requests six rows by default.`);
    assert.equal(requests[0].cols, 6, `${label} requests six columns by default.`);
    assert.equal(requests[0].cluster_digit, 4, `${label} preserves the current default digit.`);
    const advanced = frame.locator('.generation-settings');
    assert.equal(await advanced.getAttribute('open'), null, `${label} keeps advanced settings closed initially.`);
    assert.equal(await advanced.locator('summary').innerText(), 'Advanced settings', `${label} uses the concise disclosure label.`);
    assert.equal(await advanced.locator('#grid-select').count(), 1, `${label} places Grid size inside Advanced settings.`);
    assert.equal(await frame.getByRole('heading', { name: 'Generated digits', exact: true }).count(), 0, `${label} omits the redundant grid heading.`);
    assert.equal(await frame.locator('#grid-select').inputValue(), '6');
    assert.equal(await frame.locator('.generation-toolbar #grid-select').count(), 0);
    assert.equal(await frame.locator('#grid-download-hint').count(), 0, `${label} removes the download instruction.`);
    assert(!/select any digit image to download/i.test(await frame.locator('body').innerText()));
    const interactiveTiles = frame.locator('#grid [role="button"], #grid [tabindex], #grid [title], #grid a, #grid button, #grid [download]');
    assert.equal(await interactiveTiles.count(), 0, `${label} has no image interaction or download affordances.`);
    assert.notEqual(await frame.locator('.digit-cell').first().evaluate(cell => getComputedStyle(cell).cursor), 'pointer');
    await frame.locator('.digit-cell img').first().click();
    await page.waitForTimeout(200);
    assert.equal(downloads, 0, `${label} clicking a generated image causes no download.`);
    await assertLayout(page, frame, 6, `${label} default`);
    await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-default.png`) });
    await advanced.locator('summary').click();
    await assertLayout(page, frame, 6, `${label} advanced open`);
    await advanced.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-advanced.png`) });

    const chooseSize = async size => {
      const before = requests.length;
      const heldSizeChange = size === 8;
      const beforeSizeChange = heldSizeChange ? await stableWorkspaceLayout(page, frame) : null;
      if (heldSizeChange) generationGate = new Promise(resolve => { releaseGeneration = resolve; });
      await frame.locator('#grid-select').selectOption(String(size));
      if (heldSizeChange) {
        await assertAwsBadge(frame, 'Generating', `${label} pending size change`);
        await assertStableWorkspace(page, frame, beforeSizeChange, `${label} pending size change`);
        releaseGeneration();
        generationGate = null;
      }
      await frame.waitForFunction(count => document.querySelectorAll('.digit-cell').length === count && document.getElementById('status').textContent === 'Grid updated.', size * size);
      assert.equal(requests.length, before + 1, `${label} changing grid size requests exactly one new grid.`);
      assert.equal(requests.at(-1).rows, size);
      assert.equal(requests.at(-1).cols, size);
      await assertLayout(page, frame, size, `${label} ${size} by ${size}`);
    };
    await chooseSize(8);
    await chooseSize(4);
    await chooseSize(6);
    await advanced.locator('summary').click();
    const beforeGenerate = requests.length;
    const previousSeed = requests.at(-1).seed;
    const stableBeforeGenerate = await stableWorkspaceLayout(page, frame);
    generationGate = new Promise(resolve => { releaseGeneration = resolve; });
    await frame.locator('#refresh-seed-btn').click();
    if (releaseGeneration) {
      await assertAwsBadge(frame, 'Generating', `${label} pending request`);
      assert.equal(await frame.locator('.digit-cell').count(), 36, 'Generating preserves the previous visible grid.');
      await assertStableWorkspace(page, frame, stableBeforeGenerate, `${label} pending request`);
      releaseGeneration();
      generationGate = null;
    }
    await frame.waitForFunction(() => document.getElementById('status').textContent === 'Grid updated.');
    assert.equal(requests.length, beforeGenerate + 1, `${label} Generate requests one refreshed grid.`);
    assert.notEqual(requests.at(-1).seed, previousSeed, `${label} Generate chooses a fresh seed.`);
    assert.equal(requests.at(-1).rows, 6);
    await assertAwsBadge(frame, 'Connected', `${label} refreshed`);
    const beforeResize = requests.length;
    const apiBeforeResize = apiRequests.length;
    await page.setViewportSize({ width: width + 20, height: 900 });
    await page.waitForTimeout(250);
    await page.setViewportSize({ width, height: width < 600 ? 844 : 1000 });
    await page.waitForTimeout(250);
    assert.equal(requests.length, beforeResize, `${label} resizing changes layout without generating another grid.`);
    assert.equal(apiRequests.length, apiBeforeResize, `${label} resizing adds no AWS connectivity polling.`);
    await assertLayout(page, frame, 6, `${label} resized`);

    {
      const stableBeforeFailure = await stableWorkspaceLayout(page, frame);
      failGeneration = true;
      await frame.locator('#refresh-seed-btn').click();
      await frame.waitForFunction(() => document.getElementById('status').textContent === 'Failed to generate digits.');
      await assertAwsBadge(frame, 'Request failed', `${label} failed generation`);
      await assertStableWorkspace(page, frame, stableBeforeFailure, `${label} failed generation`);
      assert.equal(await frame.locator('#status').getAttribute('role'), 'status', 'Generation failures are announced without replacing the layout.');
      assert.equal(await frame.locator('#status').getAttribute('aria-live'), 'polite');
      assert(await frame.locator('#refresh-seed-btn').isEnabled(), 'Generate remains available to retry a transient failure.');
      assert.equal(await frame.locator('.digit-cell').count(), 36, 'A failed generation preserves the last usable grid.');
      await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-request-failed.png`) });
      failGeneration = false;
      await frame.locator('#refresh-seed-btn').click();
      await frame.waitForFunction(() => document.getElementById('status').textContent === 'Grid updated.');
      await assertAwsBadge(frame, 'Connected', `${label} recovered`);
      await assertStableWorkspace(page, frame, stableBeforeFailure, `${label} recovered`, { preserveImages: false });
      assert.equal(await frame.locator('.digit-cell').count(), 36, 'Generate recovers after the API is available again.');
    }
    assert.deepEqual(errors, [], `${label} flow produces no page exceptions or unexpected API calls.`);
    console.log(`Digit Generator simplified flow passed: ${label}`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `digit-generator-${label}-failure.png`) }).catch(() => {});
    error.message = `${label}: ${error.message}`;
    throw error;
  } finally {
    releaseHealth?.();
    releaseGeneration?.();
    await context.close();
  }
}

async function runDigitGeneratorSimpleChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const fixtures = await digitFixtures();
  for (const [project, width] of [[false, 1440], [false, 390], [false, 320], [true, 1440], [true, 390]]) {
    await runCase({ browser, base, artifactDir, fixtures, project, width });
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'digit-generator-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runDigitGeneratorSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-digit-generator') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runDigitGeneratorSimpleChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
