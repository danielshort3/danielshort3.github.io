'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium, firefox, webkit } = require('playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests } = require('../release/fixtures.cjs');

async function runCase(browser, origin, width, fallback) {
  const context = await browser.newContext({
    viewport: { width, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce'
  });
  await isolateRequests(context, origin);
  if (fallback) await context.addInitScript(() => {
    Object.defineProperty(Element.prototype, 'moveBefore', { configurable: true, value: undefined });
  });
  let releaseShell;
  const shellGate = new Promise((resolve) => { releaseShell = resolve; });
  await context.route('**/dist/site-shell*.js', async (route) => {
    await shellGate;
    return route.continue();
  });
  const page = await context.newPage();
  const errors = [];
  let demoLoads = 0;
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (request.isNavigationRequest() && new URL(request.url()).pathname === '/demos/minesweeper-demo.html') demoLoads += 1;
  });
  try {
    await page.goto(`${origin}/minesweeper-demo`, { waitUntil: 'commit' });
    // Deliberately let the embedded document finish before adopting its parent.
    // This detects a lost browsing context rather than only counting DOM nodes.
    await page.waitForFunction(() => {
      const frame = document.querySelector('.project-demo-wrapper-iframe');
      return frame?.contentDocument?.readyState === 'complete' && frame.contentDocument.querySelector('#demo-box');
    });
    const supportsMove = await page.evaluate(() => {
      const frame = document.querySelector('.project-demo-wrapper-iframe');
      window.__initialDemoNode = frame;
      window.__initialDemoWindow = frame.contentWindow;
      frame.contentWindow.__initialDemoMarker = 'loaded-before-shell';
      const input = frame.contentDocument.createElement('input');
      input.id = 'adoption-state-fixture';
      input.value = 'unsaved draft';
      frame.contentDocument.body.append(input);
      return typeof Element.prototype.moveBefore === 'function';
    });
    assert.equal(demoLoads, 1, 'The embedded document loads once before shell adoption.');
    releaseShell();
    await page.waitForFunction(() => window.SiteFrame?.root()?.isConnected &&
      document.querySelector('.site-frame__slot-content .project-demo-wrapper-iframe'));
    await page.waitForLoadState('load');
    const state = await page.evaluate(() => {
      const frame = document.querySelector('.project-demo-wrapper-iframe');
      return {
        sameNode: frame === window.__initialDemoNode,
        sameWindow: frame.contentWindow === window.__initialDemoWindow,
        marker: frame.contentWindow.__initialDemoMarker,
        draft: frame.contentDocument.querySelector('#adoption-state-fixture')?.value,
        workspace: Boolean(frame.contentDocument.querySelector('#demo-box')),
        adopted: Boolean(SiteFrame.outlet()?.contains(frame))
      };
    });
    assert(state.sameNode && state.workspace && state.adopted, 'Adoption retains a usable embedded workspace.');
    if (supportsMove) {
      assert(state.sameWindow && state.marker === 'loaded-before-shell' && state.draft === 'unsaved draft',
        'Adoption preserves the loaded iframe context and its unsaved input.');
      assert.equal(demoLoads, 1, 'Adoption never reloads the embedded document when native movement is supported.');
    }
    assert.deepEqual(errors, [], 'Native and fallback adoption produce no script errors.');
    console.log(`${browser.browserType().name()} ${width}px ${fallback ? 'fallback' : 'native'}: workspace ready, ${demoLoads} document load(s), preserved=${supportsMove}`);
  } finally {
    releaseShell();
    await context.close();
  }
}

(async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'initial-frame-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    for (const browserType of [chromium, firefox, webkit]) {
      const browser = await browserType.launch();
      try {
        for (const width of [1440, 390]) {
          for (const fallback of [false, true]) await runCase(browser, origin, width, fallback);
        }
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
