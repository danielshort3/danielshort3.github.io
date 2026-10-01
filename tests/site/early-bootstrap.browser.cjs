/**
 * Run after the site build: node tests/site/early-bootstrap.browser.cjs
 * Holds stylesheet responses to prove the root bootstrap runs before CSS, then
 * verifies that the document bootstrap never replays during real soft routes.
 * Response-only counters and API fixtures do not modify authored or public files.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const counter = 'window.__earlyBootstrapRuns = (window.__earlyBootstrapRuns || 0) + 1;\n';

async function runCase(browser, origin, viewport) {
  const context = await browser.newContext({ viewport, serviceWorkers: 'block', reducedMotion: 'reduce' });
  const page = await context.newPage();
  const errors = [];
  const documents = [];
  let releaseStyles;
  let firstStyle;
  let heldStyles = 0;
  let holdingStyles = true;
  const stylesReady = new Promise((resolve) => { releaseStyles = resolve; });
  const stylesheetStarted = new Promise((resolve) => { firstStyle = resolve; });
  page.on('pageerror', (error) => errors.push(error.message));
  await context.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) return route.abort('blockedbyclient');
    if (url.pathname.startsWith('/api/')) {
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{"ok":true,"authenticated":false,"sessions":[],"recentSessions":[],"tools":[]}' });
    }
    if (holdingStyles && url.pathname.endsWith('.css')) {
      heldStyles += 1;
      firstStyle();
      await stylesReady;
    }
    if (url.pathname === '/js/common/no-js.js') {
      const response = await route.fetch();
      return route.fulfill({ response, body: counter + await response.text() });
    }
    if (request.isNavigationRequest() || request.headers()['x-site-route']) {
      const response = await route.fetch();
      if ((response.headers()['content-type'] || '').includes('text/html')) {
        documents.push(url.pathname);
        const body = (await response.text()).replace(/(<script\b[^>]*data-site-bootstrap="early"[^>]*>)/, `$1${counter}`);
        return route.fulfill({ response, body });
      }
      return route.fulfill({ response });
    }
    return route.continue();
  });

  try {
    await page.goto(`${origin}/tools`, { waitUntil: 'commit' });
    await page.waitForFunction(() => window.__earlyBootstrapRuns === 1 &&
      document.documentElement.classList.contains('js') && !document.documentElement.classList.contains('no-js'), null, { timeout: 10000 });
    let stylesheetTimeout;
    try {
      await Promise.race([stylesheetStarted, new Promise((_, reject) => {
        stylesheetTimeout = setTimeout(() => reject(new Error('No stylesheet request reached the timing gate.')), 10000);
      })]);
    } finally {
      clearTimeout(stylesheetTimeout);
    }
    const early = await page.evaluate(() => ({
      count: window.__earlyBootstrapRuns,
      readyState: document.readyState,
      stylesLoaded: [...document.querySelectorAll('link[rel="stylesheet"]')].some((link) => Boolean(link.sheet)),
      shellReady: Boolean(window.SiteFrame?.root())
    }));
    assert.equal(early.count, 1);
    assert.equal(early.stylesLoaded, false, 'The root bootstrap must run before any held stylesheet loads.');
    assert.equal(early.shellReady, false, 'The root bootstrap must run before the deferred shell waits for CSS.');
    assert(heldStyles > 0, 'The timing regression must actually hold render-blocking stylesheet requests.');
    holdingStyles = false;
    releaseStyles();
    await page.waitForLoadState('domcontentloaded');
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await page.waitForFunction(() => window.SiteFrame?.root() && window.SiteNavigation?.version === 1);
    await page.evaluate(() => {
      window.__bootstrapInitialFrame = SiteFrame.root();
      window.__bootstrapInitialTimeOrigin = performance.timeOrigin;
    });

    const navigations = [];
    for (const pathname of ['/tools/text-compare', '/privacy', '/tools']) {
      const result = await page.evaluate(async (target) => {
        await SiteNavigation.navigate(target);
        return {
          pathname: location.pathname,
          count: window.__earlyBootstrapRuns,
          sameFrame: window.__bootstrapInitialFrame === SiteFrame.root(),
          sameDocument: window.__bootstrapInitialTimeOrigin === performance.timeOrigin
        };
      }, pathname);
      assert.equal(result.pathname, pathname, 'The regression must complete the requested real navigation.');
      assert.equal(result.count, 1, `The bootstrap must execute only once after navigating to ${pathname}.`);
      assert(result.sameFrame && result.sameDocument, 'Soft routes must retain the original frame and document.');
      navigations.push(result);
    }
    assert(documents.includes('/tools/text-compare') && documents.includes('/privacy'),
      'The test must fetch destination documents containing the instrumented bootstrap.');
    assert.deepEqual(errors, [], 'The bootstrap flow must have no uncaught browser errors.');
    return { viewport, bootstrapBeforeCss: true, heldStyles, early, navigations, errors };
  } catch (error) {
    const output = process.env.EARLY_BOOTSTRAP_OUTPUT_DIR || path.join(os.tmpdir(), 'site-early-bootstrap');
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, `failure-${viewport.width}.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    holdingStyles = false;
    releaseStyles();
    await context.close();
  }
}

(async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'site-bootstrap-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true });
    const cases = [];
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      cases.push(await runCase(browser, origin, viewport));
    }
    console.log(JSON.stringify({ cases }, null, 2));
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
