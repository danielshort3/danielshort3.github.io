/** Compact tool masthead and QR tabs at narrow mobile widths. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const within = (outer, inner) => inner.left >= outer.left - 1 && inner.right <= outer.right + 1;
const sameRow = (a, b) => Math.abs(a.top - b.top) <= 1;

async function geometry(page) {
  return page.evaluate(() => {
    const rect = (selector) => {
      const node = document.querySelector(selector);
      const box = node.getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width, height: box.height };
    };
    return {
      viewport: innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      masthead: rect('[data-page-masthead]'),
      back: rect('[data-page-masthead-parent]'),
      copy: rect('[data-page-masthead-copy]'),
      account: rect('[data-tools-action="open-account"]'),
      signIn: rect('[data-tools-action="sign-in"]'),
      signOut: rect('[data-tools-action="sign-out"]'),
      tabs: [...document.querySelectorAll('[data-qrtool-tab]')].map((tab) => {
        const box = tab.getBoundingClientRect();
        return { top: box.top, bottom: box.bottom };
      })
    };
  });
}

async function runCase(base, browser, width) {
  const context = await browser.newContext({ viewport: { width, height: 844 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await context.route('**/api/tools/**', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: '{"authenticated":false}' }));
  try {
    await page.goto(`${base}/tools/qr-code-generator`, { waitUntil: 'domcontentloaded' });
    assert.match(await page.title(), /QR Code Generator/);
    await page.locator('[data-tools-action="sign-in"]').waitFor({ state: 'visible' });
    await page.evaluate(() => document.fonts.ready);
    const signedOut = await geometry(page);
    assert(signedOut.scrollWidth <= width + 1, `${width}px: signed-out page has no horizontal overflow.`);
    assert(within(signedOut.masthead, signedOut.back) && within(signedOut.masthead, signedOut.signIn), `${width}px: Back and Sign in fit the masthead.`);
    assert(sameRow(signedOut.back, signedOut.signIn), `${width}px: Back and Sign in share a row.`);
    assert(signedOut.copy.top >= signedOut.back.bottom, `${width}px: title follows the Back/Sign in row.`);
    assert(signedOut.back.height >= 44 && signedOut.signIn.height >= 44, `${width}px: header controls retain 44px targets.`);
    assert(signedOut.tabs.every((tab) => sameRow(tab, signedOut.tabs[0])), `${width}px: QR tabs stay on one row.`);
    await page.getByRole('tab', { name: 'Download', exact: true }).click();
    assert(await page.locator('#qrtool-panel-export').isVisible(), `${width}px: Download tab opens its panel.`);

    // Exercise the populated account layout without a real login or account write.
    await page.evaluate(() => {
      const header = document.querySelector('[data-page-masthead]');
      header.querySelector('[data-tools-action="sign-in"]').hidden = true;
      header.querySelector('[data-tools-account="signed-in-actions"]').hidden = false;
    });
    const signedIn = await geometry(page);
    assert(signedIn.scrollWidth <= width + 1, `${width}px: signed-in account controls do not overflow.`);
    assert(signedIn.account.top >= signedIn.copy.bottom, `${width}px: Account and Sign out keep a full row below the title.`);
    assert(sameRow(signedIn.account, signedIn.signOut), `${width}px: Account and Sign out share a row.`);
    assert(within(signedIn.masthead, signedIn.account) && within(signedIn.masthead, signedIn.signOut), `${width}px: signed-in controls fit the masthead.`);
    assert(signedIn.account.height >= 44 && signedIn.signOut.height >= 44, `${width}px: signed-in controls retain 44px targets.`);
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    const enlarged = await geometry(page);
    assert(enlarged.scrollWidth <= width + 1, `${width}px: enlarged text does not cause page overflow.`);
    assert(within(enlarged.masthead, enlarged.account) && within(enlarged.masthead, enlarged.signOut), `${width}px: enlarged account controls wrap inside the masthead.`);
    assert.deepEqual(errors, [], `${width}px: no application errors.`);
    console.log(`Compact tool masthead passed at ${width}px, signed out and signed in mock.`);
  } finally {
    await context.close();
  }
}

async function run() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tool-masthead-compact-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const width of [320, 390]) await runCase(base, browser, width);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

if (require.main === module) run().catch((error) => { console.error(error); process.exitCode = 1; });

module.exports = run;
