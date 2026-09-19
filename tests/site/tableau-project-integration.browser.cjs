'use strict';

// Verify website integration against controlled cross-origin Tableau documents.
// Native Tableau chart/filter behavior is reviewed separately against Tableau Public.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const DASHBOARDS = [
  { id: 'pizzaDashboard', title: 'Pizza Delivery Dashboard', base: 'https://public.tableau.com/views/Pizza_Delivery/PizzaDeliveryDashboard', height: 880 },
  { id: 'ufoDashboard', title: 'UFO Sightings Dashboard', base: 'https://public.tableau.com/views/UFO_Sightings_16769494135040/UFOSightingDashboard-2013', height: 870 }
];

async function settle(page) {
  await page.waitForFunction(() => window.SiteRoutes?.current()?.root?.isConnected
    && !window.SiteNavigation?.isNavigating?.()
    && !document.querySelector('.site-frame--held, .site-frame--moving, [data-site-route-error]'));
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

function assertDashboardUrl(value, dashboard, device = null) {
  const url = new URL(value);
  assert.equal(`${url.origin}${url.pathname}`, dashboard.base, 'Use the final published native workbook and Overview identities.');
  assert.equal(url.searchParams.get(':embed'), 'y');
  assert.equal(url.searchParams.get(':showVizHome'), 'no');
  assert.equal(url.searchParams.get(':tabs'), 'no', 'Hide legacy worksheet tabs while retaining native dashboard navigation.');
  assert.equal(url.searchParams.get(':device'), device, 'Use the viewport-appropriate device; external launch permits native device selection.');
}

async function assertDashboard(page, dashboard, device, label) {
  await page.locator('.project-demo-shell').scrollIntoViewIfNeeded();
  await settle(page);
  assert.equal(await page.locator('iframe.project-embed-frame').count(), 1, `${label}: exactly one dashboard iframe.`);
  const frame = page.locator('iframe.project-embed-frame');
  const shell = page.locator('.project-demo-shell');
  assert.equal(await shell.locator('.project-demo-title').innerText(), dashboard.title);
  assertDashboardUrl(await frame.getAttribute('data-dashboard-default-src'), dashboard, 'desktop');
  const launch = page.locator('.project-demo-mobile-launch .btn-primary');
  assertDashboardUrl(await launch.getAttribute('href'), dashboard);
  assert.equal(await launch.getAttribute('target'), '_blank');
  assert.match(await launch.getAttribute('rel'), /noopener/);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${label}: no page overflow.`);

  await page.waitForFunction(expected => {
    const iframe = document.querySelector('iframe.project-embed-frame');
    return iframe?.getAttribute('data-dashboard-device') === expected && iframe.getAttribute('src');
  }, device);
  await page.waitForFunction(() => !document.querySelector('iframe.project-embed-frame')?._projectDashboardResizeTimer);
  assertDashboardUrl(await frame.getAttribute('src'), dashboard, device);
  assert(await frame.isVisible(), `${label}: native dashboard is shown.`);
  const view = page.frameLocator('iframe.project-embed-frame');
  await view.getByRole('heading', { name: dashboard.title, exact: true }).waitFor();
  assert.equal(await launch.isVisible(), false, `${label}: avoid duplicate dashboard preview.`);
  assert(await page.locator('.project-demo-open').isVisible(), `${label}: keep native full-view navigation available.`);
  assert.equal(await page.locator('[data-dashboard-reset]').count(), 0, `${label}: no redundant website reset control.`);
  assert.equal(await frame.getAttribute('scrolling'), 'auto', `${label}: the native dashboard can scroll vertically.`);
  const geometry = await frame.evaluate(node => {
    const box = node.getBoundingClientRect();
    const parent = node.parentElement.getBoundingClientRect();
    const parentStyle = getComputedStyle(node.parentElement);
    return {
      width: box.width, height: box.height, left: box.left - parent.left, right: parent.right - box.right,
      available: node.parentElement.clientWidth - parseFloat(parentStyle.paddingLeft) - parseFloat(parentStyle.paddingRight),
      shellWidth: node.closest('.project-demo-shell').getBoundingClientRect().width,
      viewportHeight: innerHeight, viewportWidth: innerWidth,
      nativeWidth: node.clientWidth, nativeHeight: node.clientHeight, parentHeight: parent.height
    };
  });
  assert.equal(device, geometry.viewportWidth > 768 ? 'desktop' : 'phone', `${label}: laptop and desktop viewports retain the desktop layout even inside narrower audience panels.`);
  assert(Math.abs(geometry.left - geometry.right) <= 1, `${label}: equal side gutters.`);
  if (device === 'desktop') {
    const scale = Math.min(1, geometry.available / 1200);
    assert.equal(geometry.nativeWidth, 1200, `${label}: retain the authored desktop viewport so Tableau does not clip the dashboard.`);
    assert.equal(geometry.nativeHeight, dashboard.height, `${label}: include the complete native toolbar.`);
    assert(Math.abs(geometry.width - 1200 * scale) <= 1, `${label}: fit the full desktop canvas proportionally inside the project.`);
    assert(Math.abs(geometry.height - dashboard.height * scale) <= 1, `${label}: scale the dashboard height proportionally.`);
    assert(Math.abs(geometry.parentHeight - geometry.height) <= 2, `${label}: the wrapper follows the displayed height without clipping or blank space.`);
  } else {
    assert(Math.abs(geometry.width - Math.min(720, geometry.available)) <= 1, `${label}: fit the native phone canvas within a bounded readable width.`);
    assert(Math.abs(geometry.height - Math.min(900, Math.max(600, geometry.viewportHeight * .85))) <= 1, `${label}: keep the phone workspace height stable and usable.`);
    const native = await view.locator('html').evaluate(node => ({ width: node.scrollWidth, clientWidth: node.clientWidth, height: node.scrollHeight, clientHeight: node.clientHeight }));
    assert(native.width <= native.clientWidth + 1, `${label}: phone fixture has no horizontal scrolling.`);
    assert(native.height > native.clientHeight, `${label}: the long native phone dashboard remains scrollable.`);
    if (page.viewportSize().width === 320) assert(geometry.width >= 310, `${label}: reclaim outer gutters so native UFO navigation has a readable minimum canvas.`);
  }
}

async function runCase({ browser, base, artifactDir, dashboard, audience, width }) {
  const device = width > 768 ? 'desktop' : 'phone';
  const label = `${dashboard.id}-${audience || 'personal'}-${width}`;
  const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1100 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const requests = [];
  const requestGeometry = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route('https://public.tableau.com/**', async route => {
    requests.push(route.request().url());
    requestGeometry.push(await page.evaluate(url => {
      const frame = document.querySelector('iframe.project-embed-frame');
      return { url, width: frame?.getBoundingClientRect().width, recordedWidth: frame?._projectDashboardWidth, device: frame?.getAttribute('data-dashboard-device') };
    }, route.request().url()).catch(() => ({ url: route.request().url(), detached: true })));
    const phone = new URL(route.request().url()).searchParams.get(':device') === 'phone';
    await route.fulfill({ contentType: 'text/html', body: `<!doctype html><html lang="en"><head><title>Controlled Tableau view</title><style>html,body{margin:0;padding:0}main{box-sizing:border-box;width:${phone ? '100%' : '1200px'};height:${phone ? 2100 : dashboard.height - 30}px;padding:24px;background:#fff;color:#091f3b}footer{height:27px;background:#f4f7fc}</style></head><body><main><h1>${dashboard.title}</h1><p>Controlled native dashboard fixture</p></main><footer>Tableau toolbar</footer></body></html>` });
  });
  const routePath = `/portfolio/${dashboard.id}${audience ? `?audience=${audience}` : ''}`;
  try {
    const response = await page.goto(`${base}${routePath}`, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label}: route resolves.`);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await assertDashboard(page, dashboard, device, label);
    assert.equal(requests.length, 1, `${label}: initial view loads once.`);
    assertDashboardUrl(requests[0], dashboard, device);
    await page.screenshot({ path: path.join(artifactDir, `${label}.png`) });

    if (!audience && width === 1440) {
      await page.evaluate(() => { window.__tableauRouteIdentity = 'preserved'; });
      await page.locator('.project-parent-link').click();
      await page.waitForURL(`${base}/portfolio`);
      await settle(page);
      assert.equal(await page.evaluate(() => window.__tableauRouteIdentity), 'preserved', `${label}: library navigation remains within the persistent website shell.`);
      const count = requests.length;
      await page.locator(`a.home-library__card[href="/portfolio/${dashboard.id}"]`).click();
      await page.waitForURL(`${base}/portfolio/${dashboard.id}`);
      await assertDashboard(page, dashboard, 'desktop', `${label} soft re-entry`);
      assert.equal(await page.evaluate(() => window.__tableauRouteIdentity), 'preserved', `${label}: dashboard re-entry is a soft navigation.`);
      assert.equal(requests.length, count + 1, `${label}: re-entry creates one current dashboard.`);
      const desktopCount = requests.length;
      await page.setViewportSize({ width: 390, height: 844 });
      await assertDashboard(page, dashboard, 'phone', `${label} shrink`);
      assert.equal(requests.length, desktopCount + 1, `${label}: switching to Phone reloads exactly once.`);
      const narrowCount = requests.length;
      const phoneSource = await page.locator('iframe.project-embed-frame').getAttribute('src');
      await page.setViewportSize({ width: 390, height: 760 });
      await assertDashboard(page, dashboard, 'phone', `${label} height-only resize`);
      assert.equal(requests.length, narrowCount, `${label}: height-only resize preserves the native view.`);
      assert.equal(await page.locator('iframe.project-embed-frame').getAttribute('src'), phoneSource, `${label}: height-only resize preserves the native view URL.`);
      const resized = page.waitForResponse(result => result.url().startsWith(dashboard.base));
      for (const phoneWidth of [410, 420, 430]) await page.setViewportSize({ width: phoneWidth, height: 760 });
      await resized;
      await assertDashboard(page, dashboard, 'phone', `${label} settled phone resize`);
      assert.equal(requests.length, narrowCount + 1, `${label}: rapid phone width changes coalesce into one refresh.`);
      const narrowed = page.waitForResponse(result => result.url().startsWith(dashboard.base));
      await page.setViewportSize({ width: 320, height: 760 });
      await narrowed;
      await assertDashboard(page, dashboard, 'phone', `${label} smallest phone resize`);
      assert.equal(requests.length, narrowCount + 2, `${label}: shrinking within Phone refreshes stale native geometry once.`);
      await page.setViewportSize({ width: 1440, height: 1100 });
      await assertDashboard(page, dashboard, 'desktop', `${label} expand`);
      assert.equal(requests.length, narrowCount + 3, `${label}: resizing back loads one visible dashboard.`);
    }
    if (!audience && width === 1024) {
      const cappedCount = requests.length;
      const cappedSource = await page.locator('iframe.project-embed-frame').getAttribute('src');
      await page.setViewportSize({ width: 1100, height: 1100 });
      await assertDashboard(page, dashboard, 'desktop', `${label} desktop resize`);
      assert.equal(await page.locator('iframe.project-embed-frame').evaluate(node => node.clientWidth), 1200);
      assert.equal(requests.length, cappedCount, `${label}: resizing the displayed desktop canvas preserves its interactive state.`);
      assert.equal(await page.locator('iframe.project-embed-frame').getAttribute('src'), cappedSource);
    }
    if (!audience && width === 390) {
      const beforeDeparture = requests.length;
      const outgoingFrame = await page.locator('iframe.project-embed-frame').elementHandle();
      await page.setViewportSize({ width: 400, height: 844 });
      await page.waitForFunction(() => Boolean(document.querySelector('iframe.project-embed-frame')?._projectDashboardResizeTimer));
      await page.locator('.project-parent-link').click();
      await page.waitForURL(`${base}/portfolio`);
      await settle(page);
      assert.equal(await outgoingFrame.evaluate(node => Boolean(node._projectDashboardResizeTimer)), false, `${label}: soft departure cancels a queued native resize.`);
      assert.equal(requests.length, beforeDeparture, `${label}: a departing dashboard does not start a queued reload.`);
    }
    assert.deepEqual(errors, [], `${label}: no website runtime exceptions.`);
    console.log(`Tableau project integration passed: ${label}`);
  } catch (error) {
    console.error(JSON.stringify({ label, requestGeometry }));
    await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
    error.message = `${label}: ${error.message}`;
    throw error;
  } finally { await context.close(); }
}

async function runNoScriptCase({ browser, base, dashboard, audience }) {
  const label = `${dashboard.id}-${audience || 'personal'}-no-js`;
  const context = await browser.newContext({ viewport: { width: audience ? 1920 : 1440, height: 1100 }, javaScriptEnabled: false, serviceWorkers: 'block' });
  const page = await context.newPage();
  const requests = [];
  await context.route('https://public.tableau.com/**', async route => { requests.push(route.request().url()); await route.abort(); });
  try {
    const response = await page.goto(`${base}/portfolio/${dashboard.id}${audience ? `?audience=${audience}` : ''}`, { waitUntil: 'load' });
    assert.equal(response.status(), 200, `${label}: route resolves without script.`);
    const frame = page.locator('iframe.project-embed-frame');
    assert.equal(await frame.isVisible(), false, `${label}: hide the unloaded dashboard.`);
    assert.equal(await frame.getAttribute('src'), null, `${label}: deferred iframe remains unloaded.`);
    assert(await page.locator('.project-demo-launch-image').isVisible(), `${label}: show the complete current preview.`);
    const launch = page.locator('.project-demo-mobile-launch .btn-primary');
    assert(await launch.isVisible(), `${label}: retain the native dashboard launch without site JavaScript.`);
    assertDashboardUrl(await launch.getAttribute('href'), dashboard);
    assert.equal(requests.length, 0, `${label}: no hidden external load.`);
    console.log(`Tableau project integration passed: ${label}`);
  } finally { await context.close(); }
}

async function runTableauProjectIntegrationChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const dashboard of DASHBOARDS) {
    for (const [audience, width] of [[null, 1440], [null, 1366], [null, 320], [null, 390], [null, 768], [null, 1024], ['analytics', 1920], ['data-science', 1920], ['tourism', 1920], ['analytics', 1440], ['analytics', 1024], ['analytics', 390]]) {
      await runCase({ browser, base, artifactDir, dashboard, audience, width });
    }
  }
  for (const dashboard of DASHBOARDS) for (const audience of [null, 'analytics']) await runNoScriptCase({ browser, base, dashboard, audience });
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tableau-project-integration-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runTableauProjectIntegrationChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-tableau-integration') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runTableauProjectIntegrationChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
