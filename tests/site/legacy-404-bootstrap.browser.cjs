'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { normalizeEarlyBootstrap } = require('../../build/inject-head-metadata');

const root = path.resolve(__dirname, '../..');

async function runCase(browser, requested, expectedStatus) {
  const context = await browser.newContext({ serviceWorkers: 'block' });
  const page = await context.newPage();
  const navigations = [];
  const errors = [];
  // Exercise the real published document, including its inline parser bootstrap.
  const errorHtml = fs.readFileSync(path.join(root, 'public/404.html'), 'utf8');
  assert.equal(normalizeEarlyBootstrap(errorHtml), errorHtml,
    'Build the website first: the published error bootstrap must match its authoritative source.');
  const expected = `https://www.danielshort.me${requested.replace(/\.html(?=[?#]|$)/, '')}`;
  page.on('pageerror', (error) => errors.push(error.message));
  await context.route('**/*', (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (!request.isNavigationRequest()) return route.abort('blockedbyclient');
    navigations.push(url.href);
    if (url.hostname === 'danielshort3.github.io') {
      return route.fulfill({ status: 404, contentType: 'text/html', body: errorHtml });
    }
    assert.equal(url.origin, 'https://www.danielshort.me');
    assert.equal(`${url.pathname}${url.search}`, new URL(expected).pathname + new URL(expected).search,
      'The early parser navigation must retain the requested route, including raw encoded/repeated input.');
    return route.fulfill({ status: expectedStatus, contentType: 'text/html', body: '<!doctype html><title>Destination</title><main>Destination</main>' });
  });
  try {
    await page.goto(`https://danielshort3.github.io${requested}`, { waitUntil: 'commit' });
    await page.waitForURL(expected, { waitUntil: 'load' });
    assert.equal(page.url(), expected, 'The fragment must survive the document replacement.');
    assert.equal(navigations.filter((url) => new URL(url).hostname === 'www.danielshort.me').length, 1,
      'A deferred second resolver must not mask an earlier wrong navigation.');
    assert(!navigations.some((url) => /^\/404(?:\.html)?$/.test(new URL(url).pathname)));
    assert.deepEqual(errors, []);
    return { requested, expected, expectedStatus, navigations };
  } finally {
    await context.close();
  }
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const results = [];
    for (const [requested, status] of [
      ['/tools/text-compare?input=a%20b%2Bc%26d&tag=one&tag=two#main', 200],
      ['/pages/text-compare.html?input=a%2Bb&tag=one&tag=two#main', 200],
      ['/portfolio?project=website&tag=a%20b#main', 200],
      ['/genuinely-missing-route?x=a%2Bb#missing', 404]
    ]) results.push(await runCase(browser, requested, status));
    console.log(JSON.stringify({ cases: results }, null, 2));
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
