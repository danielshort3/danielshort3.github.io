'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function setup(browser, width) {
  const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
  const requests = [];
  await context.route('**/api/**', async route => {
    const request = route.request();
    requests.push({ url: request.url(), method: request.method(), body: request.postDataJSON() });
    if (request.url().includes('/api/ga4/report')) {
      return route.fulfill({ json: { ok: true, rows: [{ pageLocation: 'https://example.test/?utm_source=newsletter&utm_medium=email&utm_campaign=autumn', sessions: 12, totalUsers: 10, eventCount: 20 }], returnedRows: 1, rowCount: 1 } });
    }
    return route.fulfill({ json: { ok: true, authenticated: false, links: [], sets: [], recentSessions: [], tools: [] } });
  });
  await context.addInitScript(() => {
    const auth = {
      getAuth: () => null, getUser: () => null, authIsValid: () => false, isAdmin: () => false,
      getConfig: () => ({}), handleRedirect: async () => ({ handled: false }), ensureFreshAuth: async () => null,
      signIn: async () => { window.privateSignInRequested = true; }, signOut: async () => {},
    };
    Object.defineProperty(window, 'ToolsAuth', { configurable: true, get: () => auth, set: () => {} });
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  return { context, page, requests, errors };
}
async function dismissCookies(page) {
  const button = page.getByRole('button', { name: 'Essential only', exact: true });
  if (await button.isVisible()) await button.click();
}
async function assertLayout(page, width) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `No horizontal page overflow at ${width}px`);
}
async function ga4(options, width) {
  const { context, page, requests, errors } = await setup(options.browser, width);
  try {
    await page.goto(`${options.base}/tools/ga4-utm-performance`);
    await dismissCookies(page);
    assert(await page.locator('[data-ga4-panel="utm"]').isVisible(), 'Report is the initial task');
    assert.deepEqual(await page.locator('[data-ga4-tab]:visible').allTextContents(), ['Report', 'Explore']);
    await page.locator('#ga4-property-id').fill('123456');
    await page.locator('#ga4-start').fill('2026-08-01');
    await page.locator('#ga4-end').fill('2026-08-31');
    await page.locator('#ga4-admin-token').fill('mock-local-token');
    await page.locator('[data-ga4="auth"] button[type="submit"]').click();
    assert(!await page.locator('[data-ga4="access-details"]').evaluate(el => el.open), 'Stored connection collapses');
    await page.locator('[data-ga4="run"]').click();
    await page.waitForFunction(() => document.querySelector('[data-ga4="utm-status"]').textContent.startsWith('Done.'));
    assert(await page.locator('[data-ga4="download-grouped"]').isEnabled());
    const report = requests.find(request => request.url.includes('/api/ga4/report'));
    assert.equal(report.body.propertyId, '123456');
    assert.equal(report.body.startDate, '2026-08-01');
    const downloadPromise = page.waitForEvent('download');
    await page.locator('[data-ga4="download-grouped"]').click();
    assert.equal((await downloadPromise).suggestedFilename(), 'ga4-utm-grouped_2026-08-01_2026-08-31.csv');
    await page.locator('[data-ga4="sort-dir"]').selectOption('asc');
    assert(await page.locator('[data-ga4="download-grouped"]').isEnabled(), 'Local sorting does not invalidate the remote report');
    await page.locator('#ga4-start').fill('2026-08-02');
    assert(await page.locator('[data-ga4="download-grouped"]').isDisabled(), 'Changing remote scope disables stale exports');
    let releaseReport;
    const delayedReport = async route => new Promise(resolve => {
      releaseReport = async () => {
        await route.fulfill({ json: { ok: true, rows: [{ pageLocation: 'https://example.test/?utm_source=outdated', sessions: 99 }], returnedRows: 1 } });
        resolve();
      };
    });
    await page.route('**/api/ga4/report', delayedReport);
    await page.locator('[data-ga4="run"]').click();
    for (let attempt = 0; !releaseReport && attempt < 30; attempt += 1) await page.waitForTimeout(20);
    assert(releaseReport, 'Delayed report request started');
    await page.locator('#ga4-end').fill('2026-09-11');
    await releaseReport();
    await page.waitForTimeout(60);
    assert(await page.locator('[data-ga4="download-grouped"]').isDisabled(), 'Late response cannot restore obsolete results');
    assert.equal(await page.locator('[data-ga4="utm-output"]').innerText(), '');
    await page.unroute('**/api/ga4/report', delayedReport);
    await page.locator('[data-ga4-tab="explore"]').click();
    assert(await page.locator('#ga4-property-id').isVisible() && await page.locator('#ga4-start').isVisible(), 'Explore retains shared scope');
    assert.equal(await page.locator('#ga4-property-id').inputValue(), '123456');
    await page.locator('[data-ga4-tab="utm"]').click();
    await page.locator('[data-ga4="access-details"] > summary').click();
    await page.locator('.ga4-connection-options > summary').click();
    await page.locator('#ga4-profile-label').fill('Test property');
    await page.locator('[data-ga4="save-profile"]').click();
    await page.locator('#ga4-property-id').fill('999');
    await page.locator('#ga4-profile-select').selectOption('123456');
    assert.equal(await page.locator('#ga4-property-id').inputValue(), '123456', 'Saved properties still populate report');
    await page.locator('[data-ga4="run"]').click();
    await page.waitForFunction(() => document.querySelector('[data-ga4="utm-status"]').textContent.startsWith('Done.'));
    await assertLayout(page, width);
    assert(await page.locator('[data-ga4="run"]').evaluate(el => el.getBoundingClientRect().right <= innerWidth), 'Report result table does not push primary controls outside the frame');
    await page.locator('[data-ga4="access-details"] > summary').click();
    await page.screenshot({ path: path.join(options.artifactDir, `ga4-simple-${width}.png`), fullPage: true });
    assert.deepEqual(errors, []);
    console.log(`Private tool passed: GA4 at ${width}px; stored connection, report, shared scope, saved property.`);
  } finally { await context.close(); }
}
async function shortLinks(options, width) {
  const { context, page, requests, errors } = await setup(options.browser, width);
  try {
    await page.goto(`${options.base}/tools/short-links`);
    await dismissCookies(page);
    await page.locator('[data-shortlinks="signed-out"]').waitFor();
    assert(!await page.locator('[data-shortlinks="workspace"]').isVisible());
    assert(!await page.locator('[data-shortlinks="new-link-from-list"]').isVisible());
    assert(!await page.locator('[data-shortlinks="filter"]').isVisible());
    await page.locator('[data-shortlinks="sign-in"]').click();
    assert(await page.evaluate(() => window.privateSignInRequested), 'Sign in delegates to the existing auth service');
    await assertLayout(page, width);
    await page.screenshot({ path: path.join(options.artifactDir, `shortlinks-signed-out-${width}.png`) });
    await page.locator('[data-shortlinks="connect-access"]').click();
    await page.locator('[data-shortlinks="token"]').fill('mock-workspace-key');
    await page.locator('[data-shortlinks="auth"] button[type="submit"]').click();
    await page.locator('[data-shortlinks="workspace"]').waitFor();
    assert(!await page.locator('[data-shortlinks="signed-out"]').isVisible());
    assert(await page.locator('[data-shortlinks="new-link-from-list"]').isVisible());
    await page.locator('.shortlinks-account-menu > summary').click();
    for (const mode of ['qr', 'analytics', 'links']) {
      await page.locator(`[data-shortlinks-mode="${mode}"]`).click();
      assert(await page.locator(`#shortlinks-mode-${mode}`).isVisible());
    }
    await assertLayout(page, width);
    await page.evaluate(() => window.ShortLinksClient.setToken(''));
    await page.locator('[data-shortlinks="signed-out"]').waitFor();
    assert(!await page.locator('[data-shortlinks="workspace"]').isVisible(), 'Disconnect hides management again');
    assert(requests.filter(request => request.url.includes('/api/short-links')).every(request => request.method === 'GET'), 'Auth review never creates or edits server links');
    assert.deepEqual(errors, []);
    console.log(`Private tool passed: Short Links at ${width}px; signed-out gate, key access, task views, disconnect.`);
  } finally { await context.close(); }
}
async function creative(options, width) {
  const { context, page, errors } = await setup(options.browser, width);
  try {
    await page.goto(`${options.base}/tools/campaign-creative-tracker`);
    await dismissCookies(page);
    await page.getByRole('heading', { name: 'Start a campaign', exact: true }).waitFor();
    await page.locator('[data-ctc-action="new-campaign"]').click();
    await page.locator('[name="campaignName"]').fill('Autumn launch');
    await page.locator('[data-ctc-form="campaign"] button[type="submit"]').click();
    assert(await page.locator('.ctc-app-title').innerText().then(text => text.includes('Autumn launch')));
    assert.equal(await page.locator('[data-ctc-action="new-family"]').count(), 1);
    assert(await page.locator('[data-ctc-action="import"]').isDisabled());
    await page.locator('[data-ctc-action="new-family"]').click();
    await page.locator('[name="familyName"]').fill('Newsletter');
    await page.locator('[data-ctc-form="family"] [name="familyId"]').fill('NEWSLETTER');
    await page.locator('[data-ctc-form="family"] button[type="submit"]').click();
    assert(await page.locator('[data-ctc-action="import"]').isEnabled());
    await page.reload();
    await dismissCookies(page);
    assert(await page.locator('.ctc-app-title').innerText().then(text => text.includes('Autumn launch')), 'Authored campaign persists, never reseeds');
    await page.locator('.ctc-campaign-menu > summary').click();
    page.once('dialog', dialog => dialog.accept());
    await page.locator('[data-ctc-action="load-example"]').click();
    assert(await page.locator('.ctc-app-title').innerText().then(text => text.includes('Example campaign')));
    assert.equal(await page.locator('[data-ctc-action="new-family"]').count(), 1);
    assert.equal(await page.locator('[data-ctc-action="import"]').count(), 1);
    if (width <= 1000) {
      await page.locator('[data-ctc-action="select-family"]').first().click();
      await page.locator('[data-ctc-action="select-rendition"]').first().click();
    }
    const settings = page.locator('[data-ctc-rendition-settings]');
    assert(!await settings.evaluate(el => el.open), 'Rendition precision controls start collapsed');
    await settings.locator('summary').click();
    await page.locator('[data-ctc-rendition-meta="width"]').fill('970');
    await page.locator('[data-ctc-rendition-meta="width"]').press('Tab');
    assert(await settings.evaluate(el => el.open), 'Editing refinement does not close its disclosure');
    await page.waitForFunction(() => document.activeElement?.matches('[data-ctc-rendition-meta="height"]'));
    await settings.locator('summary').click();
    await assertLayout(page, width);
    await page.screenshot({ path: path.join(options.artifactDir, `creative-simple-${width}.png`), fullPage: true });
    await page.locator('.ctc-app-header').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(options.artifactDir, `creative-header-${width}.png`) });
    await page.locator('[data-view="dashboard"]').click();
    assert(!(await page.locator('.ctc-view').innerText()).includes('Approval history'), 'No fabricated example approvals');
    assert.deepEqual(errors, []);
    console.log(`Private tool passed: Creative Tracker at ${width}px; blank/example choices, authored persistence, one action location, optional refinement.`);
  } finally { await context.close(); }
}
async function runPrivateToolsSimpleChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const width of [1440, 390, 320]) for (const run of [ga4, shortLinks, creative]) await run(options, width);
}
module.exports = runPrivateToolsSimpleChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'private-tools-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runPrivateToolsSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-private-tools') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
