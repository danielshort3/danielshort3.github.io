/** Android project download reads the approved public feed without fetching the APK. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const root = path.resolve(__dirname, '../..');
const staged = JSON.parse(fs.readFileSync(path.join(root, 'mobile/android/releases/review/latest.json'), 'utf8'));

async function run() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'android-project-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 320, height: 740 }]) {
      const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'block' });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      try {
        await page.goto(`${base}/portfolio/androidApp`);
        const link = page.getByRole('link', { name: 'Download Android review APK' });
        await link.waitFor();
        await page.waitForFunction((url) => document.querySelector('[data-android-review-download]')?.href === url,
          staged.latest.apk.url);
        assert.equal(await link.getAttribute('href'), staged.latest.apk.url);
        assert.equal(await link.getAttribute('target'), null, 'APK follows normal browser download navigation');
        const releaseStatus = await page.locator('[data-android-release-status]').innerText();
        assert(releaseStatus.startsWith(`Review build ${staged.latest.versionName.replace(/-debug$/i, '')} ·`));
        assert(releaseStatus.includes('Android 8+'));
        await page.locator('.project-hero h1:visible').waitFor();
        assert(await page.locator('.project-preview-shell img[src*="androidApp"]:visible').count());
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), 0,
          `${viewport.width}px viewport must not overflow`);
        assert.deepEqual(errors, []);
      } finally {
        await context.close();
      }
    }

    const navigationContext = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
    try {
      const page = await navigationContext.newPage();
      await page.goto(`${base}/portfolio`);
      await page.locator('a[href="/portfolio/androidApp"]').first().click();
      await page.waitForURL('**/portfolio/androidApp');
      await page.locator('[data-android-review-download]').waitFor();
      assert.equal(await page.locator('[data-android-review-download]').getAttribute('href'), staged.latest.apk.url,
        'library navigation keeps the APK action available');
    } finally {
      await navigationContext.close();
    }

    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
    const page = await context.newPage();
    const future = structuredClone(staged);
    const [major, minor, patch] = staged.latest.versionName.replace(/-debug$/i, '').split('.').map(Number);
    const nextVersion = `${major}.${minor}.${patch + 1}`;
    future.latest.versionCode += 1;
    future.latest.versionName = `${nextVersion}-debug`;
    future.latest.apk.url = `https://github.com/danielshort3/danielshort3.github.io/releases/download/android-v${nextVersion}-review/Daniel-Short-review-v${future.latest.versionCode}-fixture.apk`;
    future.latest.apk.sha256 = 'b'.repeat(64);
    future.releases.push({ versionCode: future.latest.versionCode, sha256: future.latest.apk.sha256,
      size: future.latest.apk.size, signerSha256: future.latest.signerSha256 });
    await context.route('**/app-updates/review/latest.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(future) }));
    try {
      await page.goto(`${base}/portfolio/androidApp`);
      const link = page.getByRole('link', { name: 'Download Android review APK' });
      await page.waitForFunction((url) => document.querySelector('[data-android-review-download]')?.href === url,
        future.latest.apk.url);
      assert.equal(await link.getAttribute('href'), future.latest.apk.url,
        'newer approved release replaces the generated fallback without another site build');
      assert((await page.locator('[data-android-release-status]').innerText()).includes(nextVersion));

      const hostile = structuredClone(future);
      hostile.latest.apk.url = 'https://example.test/not-an-apk';
      await context.unroute('**/app-updates/review/latest.json');
      await context.route('**/app-updates/review/latest.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(hostile) }));
      await page.reload();
      await page.locator('[data-android-release-retry]:visible').waitFor();
      assert.equal(await link.getAttribute('href'), staged.latest.apk.url,
        'untrusted feed must never replace the approved generated download');
      assert.match(await page.locator('[data-android-release-status]').innerText(), /check unavailable/);
      await context.unroute('**/app-updates/review/latest.json');
      await context.route('**/app-updates/review/latest.json', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(future) }));
      await page.getByRole('button', { name: 'Retry check' }).click();
      await page.waitForFunction((url) => document.querySelector('[data-android-review-download]')?.href === url,
        future.latest.apk.url);
    } finally {
      await context.close();
    }
    console.log('Android project download: desktop/mobile layout, current link, newer feed, rejected URL, retry passed.');
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(envDir, { recursive: true, force: true });
  }
}

run().catch(error => { console.error(error); process.exitCode = 1; });
