'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const DEMOS = ['baby-names', 'covid-outbreak', 'retail-loss-sales', 'target-empty-package', 'pizza-tips'];

async function assertRepeatedResizePaint(page, frame, artifactDir, requests) {
  await frame.locator('[data-sales-metric="online"]').click();
  await frame.evaluate(() => { window.__dashboardResizeIdentity = 'preserved'; });
  const dataRequests = () => requests.filter(url => url.includes('/demos/data/') || /\/demos\/[^/]+-demo\.html/.test(url)).length;
  const initialRequests = dataRequests();
  let initialHeadingPixels;
  for (const [index, width] of [1440, 390, 1440, 390].entries()) {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 1000 });
    await assertLayout(page, frame, `retail repeated resize ${index}`);
    assert.equal(await frame.evaluate(() => window.__dashboardResizeIdentity), 'preserved', 'Breakpoint changes preserve the mounted iframe.');
    assert.equal(await frame.locator('[data-sales-metric="online"]').getAttribute('aria-pressed'), 'true', 'Breakpoint changes preserve the selected chart metric.');
    assert.equal(dataRequests(), initialRequests, 'Breakpoint changes do not reload the iframe or its datasets.');
    if (width !== 390) continue;
    const rail = await page.locator('.site-frame__tab.is-active').boundingBox();
    const heading = await page.locator('h1').filter({ visible: true }).boundingBox();
    const buffer = await page.screenshot({ path: path.join(artifactDir, `dashboard-retail-resize-${index}-390.png`) });
    const { data, info } = await sharp(buffer).raw().toBuffer({ resolveWithObject: true });
    const offset = (Math.floor(rail.y + 8) * info.width + Math.floor(rail.x + 10)) * info.channels;
    const [red, green, blue] = data.subarray(offset, offset + 3);
    assert(blue > 180 && red < 80 && green < 150, 'The full-width mobile Projects rail is painted after repeated resizes.');
    let headingPixels = 0;
    for (let y = Math.max(0, Math.ceil(heading.y)); y < Math.min(info.height, heading.y + heading.height); y += 1) {
      for (let x = Math.max(0, Math.ceil(heading.x)); x < Math.min(info.width, heading.x + heading.width); x += 1) {
        const position = (y * info.width + x) * info.channels;
        if (data[position] < 60 && data[position + 1] < 90 && data[position + 2] < 150) headingPixels += 1;
      }
    }
    assert(headingPixels > 100, 'The page title remains painted, not just present in the layout tree.');
    if (initialHeadingPixels !== undefined) assert(headingPixels >= initialHeadingPixels * .8, 'Repeated breakpoint changes preserve the visible title without clipping it away.');
    initialHeadingPixels = headingPixels;
  }
}

async function assertLayout(page, frame, label) {
  await frame.evaluate(() => document.fonts.ready);
  await frame.waitForFunction(() => Array.from(document.querySelectorAll('canvas')).every(canvas => {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const chart = window.Chart?.getChart(canvas);
    const settled = !chart || !window.Chart.animator.running(chart);
    return settled && Math.abs(canvas.width - rect.width * ratio) <= 1 && Math.abs(canvas.height - rect.height * ratio) <= 1;
  }));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const layout = await frame.evaluate(() => {
    const surface = document.querySelector('.demo-size-dashboard');
    const rect = surface.getBoundingClientRect();
    const header = surface.querySelector('.demo-surface-header').getBoundingClientRect();
    const badge = surface.querySelector('.aws-status-badge').getBoundingClientRect();
    const content = surface.querySelector('.content, .section').getBoundingClientRect();
    const controls = Array.from(surface.querySelectorAll('button, input:not([type="hidden"]), select')).filter(element => element.checkVisibility()).map(element => {
      const box = element.getBoundingClientRect();
      return { id: element.id || element.textContent.trim(), left: box.left, right: box.right, width: box.width, height: box.height };
    }).filter(box => box.width && box.height);
    return {
      width: rect.width, right: rect.right, left: rect.left, headerBottom: header.bottom,
      contentTop: content.top, badgeRight: badge.right, badgeTop: badge.top, badgeBottom: badge.bottom,
      headerTop: header.top, white: getComputedStyle(surface).backgroundColor,
      radius: parseFloat(getComputedStyle(surface).borderTopLeftRadius),
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth, controls
    };
  });
  assert(layout.width <= 1201, `${label}: dashboard fits the shared width preset.`);
  assert.equal(layout.white, 'rgb(255, 255, 255)', `${label}: workspace is white.`);
  assert(layout.radius >= 10, `${label}: workspace retains rounded corners.`);
  assert(layout.badgeRight <= layout.right && layout.badgeRight >= layout.right - 33, `${label}: status stays at the upper right.`);
  assert(layout.badgeTop >= layout.headerTop && layout.badgeBottom <= layout.headerBottom + 1, `${label}: status remains inside the header.`);
  assert(layout.headerBottom <= layout.contentTop + 1, `${label}: dashboard follows its control header.`);
  assert(layout.overflow <= 1, `${label}: iframe has no horizontal overflow.`);
  const overflowing = layout.controls.filter(control => control.left < layout.left - 1 || control.right > layout.right + 1);
  assert.deepEqual(overflowing, [], `${label}: visible controls fit the surface.`);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${label}: page has no horizontal overflow.`);
  assert.equal(await frame.locator('.demo-surface-header .header-copy').isVisible(), false, `${label}: embedded content has no duplicate title.`);
}

async function exercise(frame, demo, label, page, requests) {
  if (demo === 'baby-names') {
    await frame.locator('#status-pill[data-state="ok"]').waitFor();
    const stat = await frame.locator('#stat-rated').innerText();
    await frame.locator('[data-sex="M"]').click();
    assert.equal(await frame.locator('[data-sex="M"]').getAttribute('aria-pressed'), 'true', `${label}: category updates.`);
    assert.notEqual(await frame.locator('#stat-rated').innerText(), stat, `${label}: category recomputes the summary.`);
    await frame.locator('#name-search').fill('zzzz-no-matching-name');
    await frame.locator('#name-list .empty-state').waitFor();
    await frame.locator('#name-search').fill('');
    assert(await frame.locator('.name-item').count() > 0, `${label}: clearing search restores results.`);
    await frame.locator('[data-sex="F"]').click();
    assert.equal(await frame.locator('#ratings-panel').getAttribute('open'), null, `${label}: method is initially collapsed.`);
  } else if (demo === 'covid-outbreak') {
    await frame.locator('#connection-pill[data-state="ok"]').waitFor();
    await frame.waitForFunction(() => document.querySelector('#prob-value').textContent !== '--');
    await frame.locator('#state-select').selectOption('CO');
    await frame.waitForFunction(() => document.querySelector('#state-title').textContent.includes('Colorado'));
    const previous = await frame.locator('#prob-value').innerText();
    const date = await frame.locator('#date-value').innerText();
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    let delayed = true;
    await page.route('**/demos/data/covid-outbreak/by-date/*.json', async route => {
      if (!delayed) return route.continue();
      delayed = false;
      await gate;
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
    });
    await frame.locator('#date-slider').evaluate(element => {
      element.value = element.value === '0' ? element.max : '0';
      element.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await frame.locator('#connection-pill[data-state="loading"]').waitFor();
    assert.equal(await frame.locator('#prob-value').innerText(), previous, `${label}: prior metric stays visible during loading.`);
    release();
    await frame.locator('#connection-pill[data-state="err"]').waitFor();
    assert.equal(await frame.locator('#date-value').innerText(), date, `${label}: a failed date retains the displayed date.`);
    assert.equal(await frame.locator('#prob-value').innerText(), previous, `${label}: a failed date retains prior metrics.`);
    assert(await frame.locator('.dashboard-feedback').isVisible(), `${label}: errors expose useful context.`);
    await frame.locator('#retry-demo').click();
    await frame.locator('#connection-pill[data-state="ok"]').waitFor();
    assert.notEqual(await frame.locator('#date-value').innerText(), date, `${label}: Retry applies the requested date.`);
    await frame.locator('#date-slider').evaluate(element => {
      element.value = element.max;
      element.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await frame.locator('#connection-pill[data-state="ok"]').waitFor();
  } else if (demo === 'retail-loss-sales') {
    await frame.locator('#status-pill[data-state="ok"]').waitFor();
    assert.notEqual(await frame.locator('#kpi-sales').innerText(), '--', `${label}: summary is populated.`);
    await frame.locator('[data-sales-metric="online"]').click();
    assert.equal(await frame.locator('[data-sales-metric="online"]').getAttribute('aria-pressed'), 'true', `${label}: chart metric changes.`);
    const region = await frame.locator('#incident-region option').nth(1).getAttribute('value');
    await frame.locator('#incident-region').selectOption(region);
    await frame.locator('#reset-filters').click();
    assert.equal(await frame.locator('#incident-region').inputValue(), 'all', `${label}: Reset restores all regions.`);
    assert.equal(await frame.locator('[data-sales-metric="sales"]').getAttribute('aria-pressed'), 'true', `${label}: Reset restores the default sales metric.`);
    assert.match(await frame.locator('.dashboard-method').textContent(), /anonymized/, `${label}: method retains data context.`);
  } else if (demo === 'target-empty-package') {
    await frame.locator('#status-pill[data-state="ok"]').waitFor();
    const total = await frame.locator('#kpi-total').innerText();
    const location = await frame.locator('#filter-location option').nth(1).getAttribute('value');
    await frame.locator('#filter-location').selectOption(location);
    assert.notEqual(await frame.locator('#kpi-total').innerText(), total, `${label}: scope updates every summary.`);
    await frame.locator('[data-metric="count"]').click();
    assert.equal(await frame.locator('[data-metric="count"]').getAttribute('aria-pressed'), 'true', `${label}: trend metric changes.`);
    await frame.locator('#reset-filters').click();
    assert.equal(await frame.locator('#filter-location').inputValue(), 'all', `${label}: Reset restores all locations.`);
    assert.equal(await frame.locator('#kpi-total').innerText(), total, `${label}: Reset restores the original totals.`);
    assert(await frame.locator('.filters-panel').evaluate(element => element.getBoundingClientRect().bottom <= document.querySelector('#executive-panel').getBoundingClientRect().top), `${label}: record filters precede the affected summary.`);
    assert(await frame.locator('#kpi-total').evaluate(element => {
      const range = document.createRange();
      range.selectNodeContents(element);
      return range.getClientRects().length === 1;
    }), `${label}: the primary currency value stays on one line.`);
  } else if (demo === 'pizza-tips') {
    await frame.locator('#estimate-summary[data-state="current"]').waitFor();
    assert.equal(requests.filter(url => url.includes('tile.openstreetmap.org')).length, 0, `${label}: closed map loads no tiles.`);
    const previous = await frame.locator('#tip-amount').innerText();
    await frame.evaluate(() => {
      const original = window.PizzaTipsRuntime.predict;
      window.PizzaTipsRuntime.predict = async payload => {
        await new Promise(resolve => { window.__releaseEstimate = resolve; });
        return original(payload);
      };
    });
    await frame.locator('#cost').fill('55');
    await frame.locator('#predict').click();
    await frame.locator('#estimate-summary[data-state="loading"]').waitFor();
    assert.equal(await frame.locator('#api-status').innerText(), 'Updating estimate', `${label}: the compact badge conveys progress.`);
    assert.equal(await frame.locator('#tip-amount').innerText(), previous, `${label}: prior estimate stays visible while updating.`);
    await frame.evaluate(() => window.__releaseEstimate());
    await frame.locator('#estimate-summary[data-state="current"]').waitFor();
    assert.notEqual(await frame.locator('#tip-amount').innerText(), previous, `${label}: estimate reflects new input.`);
    assert(await frame.locator('#scenario-form > details').evaluate(element => element.getBoundingClientRect().top >= document.querySelector('#scenario-form > .actions').getBoundingClientRect().bottom), `${label}: advanced estimate settings follow the primary action.`);
    assert.equal(await frame.locator('#location-details').getAttribute('open'), null, `${label}: map starts collapsed.`);
    await frame.locator('#location-details > summary').click();
    await frame.locator('#map .leaflet-tile').first().waitFor({ state: 'attached' });
    assert(requests.some(url => url.includes('tile.openstreetmap.org')), `${label}: opening the map loads tiles.`);
    await frame.locator('#location-details > summary').click();
  }
}

async function runCase({ browser, base, artifactDir, width, demo }) {
  const label = `${demo}-${width}`;
  const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1100 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(18000);
  const errors = [];
  const requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));
  // Avoid external tile traffic; verify the real lazy-loader creates its tile requests.
  await context.route('**/*.tile.openstreetmap.org/**', route => route.fulfill({ status: 200, contentType: 'image/png', body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j5GQAAAAASUVORK5CYII=', 'base64') }));
  try {
    const response = await page.goto(`${base}/${demo}-demo`, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label}: wrapper loads.`);
    const iframe = page.locator('iframe').first();
    await iframe.waitFor();
    const frame = await (await iframe.elementHandle()).contentFrame();
    await frame.locator('.demo-size-dashboard').waitFor();
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await exercise(frame, demo, label, page, requests);
    await page.evaluate(() => window.scrollTo(0, 0));
    await assertLayout(page, frame, label);
    await page.screenshot({ path: path.join(artifactDir, `dashboard-${label}.png`), fullPage: true });
    const networkBefore = requests.filter(url => url.includes('/demos/data/')).length;
    await page.setViewportSize({ width: width === 1440 ? 1000 : width + 10, height: 900 });
    await assertLayout(page, frame, `${label} resized`);
    assert.equal(requests.filter(url => url.includes('/demos/data/')).length, networkBefore, `${label}: resizing triggers no data reload.`);
    if (demo === 'retail-loss-sales' && width === 1440) await assertRepeatedResizePaint(page, frame, artifactDir, requests);
    assert.equal(requests.filter(url => url.includes('/api/demos/')).length, 0, `${label}: local dashboards use no AWS demo API.`);
    assert.deepEqual(errors, [], `${label}: interactions raise no page exceptions.`);
    console.log(`Dashboard layout passed: ${label}`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `dashboard-${label}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    await context.close();
  }
}

async function runDashboardProjectLayoutChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const failures = [];
  for (const demo of DEMOS) {
    for (const width of [1440, 390, 320]) {
      try {
        await runCase({ browser, base, artifactDir, width, demo });
      } catch (error) {
        failures.push(error);
        console.error(`Dashboard layout failed: ${demo}-${width}: ${error.message}`);
      }
    }
  }
  if (failures.length) throw new AggregateError(failures, 'Dashboard layout checks failed.');
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dashboard-layout-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runDashboardProjectLayoutChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-dashboard-layout') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runDashboardProjectLayoutChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
