'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { createLocalServer } = require('../../build/dev');

async function runSearchChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const width of [320, 390, 1440]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    try {
      await page.goto(`${base}/search?q=image`, { waitUntil: 'networkidle' });
      if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
      await page.locator('#search-results a').first().waitFor();
      assert.equal(await page.title(), 'Search | Daniel Short');
      assert.equal(await page.locator('#search-results a').first().getAttribute('href'), '/tools/image-optimizer');
      assert(!new URL(page.url()).searchParams.has('q'), 'raw query stays out of the address/history');
      await page.evaluate(() => document.fonts.ready);
      const geometry = await page.evaluate(() => {
        const panel = document.querySelector('.search-card').getBoundingClientRect();
        return [...document.querySelectorAll('.search-result, .search-input, .search-filters button')].map((node) => {
          const box = node.getBoundingClientRect();
          return { left: box.left, right: box.right, width: box.width, panelLeft: panel.left, panelRight: panel.right, client: node.clientWidth, scroll: node.scrollWidth };
        });
      });
      for (const box of geometry) {
        assert(box.left >= box.panelLeft && box.right <= box.panelRight, `${width}: complete results and controls fit within their card`);
        assert(box.scroll <= box.client + 1, `${width}: result text is not clipped inside the card`);
      }
      await page.screenshot({ path: path.join(artifactDir, `search-${width}.png`) });
      const tools = page.locator('[data-search-category="Tools"]');
      await tools.focus();
      await page.keyboard.press('Enter');
      assert.equal(await tools.getAttribute('aria-pressed'), 'true');
      const count = Number((await tools.getAttribute('aria-label')).match(/\d+/)[0]);
      assert.equal(await page.locator('#search-results a').count(), count);
      assert((await page.locator('.search-badge').allTextContents()).every((value) => value === 'Tools'));
      await page.locator('#search-results a').first().click();
      await page.waitForURL('**/tools/image-optimizer');
      await page.locator('#imgopt-process').waitFor();
      await page.waitForFunction(() => !history.state?.siteRouteProvisional && history.state?.siteRoute?.url?.includes('/tools/image-optimizer'));
      await page.goBack();
      await page.locator('[data-search-category="Tools"][aria-pressed="true"]').waitFor();
      assert.equal(await page.locator('#search-page-q').inputValue(), 'image', 'query and selected category survive returning from a result');
      await page.locator('#search-page-q').fill('synthetic digit generator');
      await page.getByRole('button', { name: 'Show all results', exact: true }).waitFor();
      await page.getByRole('button', { name: 'Show all results', exact: true }).click();
      assert.equal(await page.locator('#search-results a').first().getAttribute('href'), '/portfolio/digitGenerator');
      assert.equal(await page.locator('[data-search-category="All"]').evaluate((node) => node === document.activeElement), true);
      await page.locator('#search-page-q').fill('unfindablezzq');
      await page.waitForFunction(() => document.querySelector('#search-status').textContent.includes('No results'));
      assert.equal(await page.locator('#search-filters').isVisible(), false);
      await page.locator('#search-page-q').fill('');
      await page.waitForFunction(() => document.querySelector('#search-status').textContent.startsWith('Try a tool name'));
      assert.equal(await page.locator('#search-results a').count(), 0);
      assert.deepEqual(errors, []);
      console.log(`Search responsive flow passed: ${width}px`);
    } finally { await context.close(); }
  }

  const professional = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
  const professionalPage = await professional.newPage();
  try {
    await professionalPage.goto(`${base}/search?audience=analytics&q=synthetic+digit+generator`, { waitUntil: 'networkidle' });
    if (await professionalPage.locator('#pcz-reject').isVisible()) await professionalPage.locator('#pcz-reject').click();
    await professionalPage.locator('[data-search-category="Projects"]').click();
    const project = professionalPage.locator('#search-results a').first();
    await professionalPage.waitForFunction(() => document.querySelector('#search-results a')?.href.includes('audience=analytics'));
    const destination = new URL(await project.getAttribute('href'), base);
    assert.equal(destination.pathname, '/portfolio/digitGenerator');
    assert.equal(destination.searchParams.get('audience'), 'analytics');
    assert.equal(new URL(professionalPage.url()).searchParams.get('audience'), 'analytics');
    assert(!new URL(professionalPage.url()).searchParams.has('q'));
    console.log('Professional search preserves audience context.');
  } finally { await professional.close(); }

  // Categories are counted/filtered before the display cap, including matches
  // ranked after the first 50; malformed result text is always escaped.
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
  const fixtures = [
    ...Array.from({ length: 55 }, (_, index) => ({ title: `Signal ${index}`, category: 'Tools', url: `/tools/signal-${index}`, description: 'A signal tool.' })),
    { title: 'Zeta project', category: 'Portfolio', url: '/portfolio/zeta', description: 'A signal project.' },
    { title: 'Signal <img src=x onerror=alert(1)>', category: 'Pages', url: '/signal', description: 'signal' }
  ];
  await context.route('**/dist/search-index.json', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ pages: fixtures }) }));
  const page = await context.newPage();
  try {
    await page.goto(`${base}/search?q=signal`, { waitUntil: 'networkidle' });
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    assert.equal(await page.locator('#search-results a').count(), 50);
    assert.match(await page.locator('#search-status').innerText(), /50 of 57/);
    assert.equal(await page.locator('#search-results img').count(), 0);
    await page.locator('[data-search-category="Projects"]').click();
    assert.equal(await page.locator('#search-results a').count(), 1);
    assert.equal(await page.locator('#search-results a').first().getAttribute('href'), '/portfolio/zeta');
  } finally { await context.close(); }

  // A failed index load must remain an unavailable state, rather than a false
  // zero-match result; explicit retry fetches again and recovers.
  const recovery = await browser.newContext({ serviceWorkers: 'block', reducedMotion: 'reduce' });
  let unavailable = true;
  await recovery.route('**/dist/search-index.json', (route) => route.fulfill({ status: unavailable ? 503 : 200, contentType: 'application/json', body: unavailable ? '{}' : JSON.stringify({ pages: fixtures }) }));
  const recoveryPage = await recovery.newPage();
  try {
    await recoveryPage.goto(`${base}/search?q=signal`, { waitUntil: 'networkidle' });
    if (await recoveryPage.locator('#pcz-reject').isVisible()) await recoveryPage.locator('#pcz-reject').click();
    await recoveryPage.getByRole('button', { name: 'Try again', exact: true }).waitFor();
    assert.match(await recoveryPage.locator('#search-status').innerText(), /unavailable/);
    unavailable = false;
    await recoveryPage.getByRole('button', { name: 'Try again', exact: true }).click();
    await recoveryPage.locator('#search-results a').first().waitFor();
    assert.equal(await recoveryPage.locator('#search-results a').count(), 50);
  } finally { await recovery.close(); }
  console.log('Search ranking, category counts, cap, escaping and retry checks passed.');
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'search-review-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true });
    await runSearchChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: path.join(os.tmpdir(), 'search-implemented-2026-09-26') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runSearchChecks;
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
