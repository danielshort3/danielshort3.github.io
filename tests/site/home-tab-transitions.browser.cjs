/**
 * Run after npm run build: node tests/site/home-tab-transitions.browser.cjs
 * Maps are local response fixtures; repeated transitions never load Google Maps.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const root = path.resolve(__dirname, '../..');
const mapColor = [64, 160, 96];
const mapMarkup = '<!doctype html><html style="height:100%;background:rgb(64,160,96)"><title>Local transition map</title><body style="margin:0;height:100%;background:rgb(64,160,96)">Local transition map</body></html>';

async function settle(page, category) {
  await page.waitForFunction(category => {
    const frame = window.SiteFrame?.root();
    const viewport = window.SiteFrame?.viewport();
    return frame?.dataset.frameCategory === category && frame.dataset.frameView === 'overview' &&
      !frame.classList.contains('site-frame--moving') && !frame.classList.contains('site-frame--held') &&
      !viewport.inert && !viewport.getAnimations().some(animation => animation.pending || animation.playState === 'running') &&
      (location.hash === `#${category}` || category === 'about' && !location.hash) &&
      !window.SiteNavigation?.isNavigating?.();
  }, category);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function beginTransition(page, category) {
  await page.evaluate(category => {
    window.homeTabSamples = [];
    const run = window.homeTabSampleRun = (window.homeTabSampleRun || 0) + 1;
    const start = performance.now();
    const sample = () => {
      if (window.homeTabSampleRun !== run) return;
      const frame = SiteFrame.root();
      const viewport = SiteFrame.viewport();
      const map = document.querySelector('[data-persistent-contact-map]');
      const iframe = map?.querySelector('iframe');
      const stage = document.querySelector('.site-frame__stage');
      const panel = document.querySelector('.site-frame__panel');
      const stageStyle = getComputedStyle(stage);
      const panelStyle = getComputedStyle(panel, '::before');
      const clip = getComputedStyle(viewport).clipPath;
      const insets = (clip.match(/[\d.]+%/g) || []).map(parseFloat);
      const top = insets[0] || 0;
      const right = insets[1] ?? top;
      const bottom = insets[2] ?? top;
      const left = insets[3] ?? right;
      window.homeTabSamples.push({
        elapsed: performance.now() - start,
        category: frame.dataset.frameCategory,
        moving: frame.classList.contains('site-frame--moving'),
        inert: viewport.inert,
        revealed: top + bottom < 99 && left + right < 99,
        scrollTop: viewport.scrollTop,
        documentTop: window.scrollY,
        mapVisible: Boolean(map && !map.hidden && getComputedStyle(map).display !== 'none' && getComputedStyle(map).visibility === 'visible'),
        mapWidth: iframe?.clientWidth || 0,
        mapHeight: iframe?.clientHeight || 0,
        stageRadius: parseFloat(stageStyle.borderTopRightRadius),
        stageBorder: parseFloat(stageStyle.borderTopWidth),
        panelTopRadius: parseFloat(panelStyle.borderTopRightRadius),
        panelBottomRadius: parseFloat(panelStyle.borderBottomRightRadius)
      });
      requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
    document.querySelector(`[data-site-tab="${category}"]`).click();
  }, category);
}

async function endTransition(page, category) {
  await settle(page, category);
  return page.evaluate(() => {
    window.homeTabSampleRun += 1;
    return window.homeTabSamples;
  });
}

async function scrollCurrentToEnd(page) {
  return page.evaluate(() => {
    const viewport = SiteFrame.viewport();
    viewport.scrollTop = viewport.scrollHeight;
    if (SiteFrame.root().dataset.frameCompact === 'true') {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' });
    }
    return Math.max(viewport.scrollTop, window.scrollY);
  });
}

function assertIncomingAtTop(samples, category, label) {
  const incoming = samples.filter(sample => sample.category === category && !sample.inert);
  assert(incoming.length >= 3, `${label} samples the incoming reveal and its settled frame.`);
  assert(incoming.every(sample => sample.scrollTop <= 1),
    `${label} starts at the top throughout the reveal, without a delayed scroll restoration: ${JSON.stringify(incoming.filter(sample => sample.scrollTop > 1))}`);
}

async function countMapPixels(buffer) {
  const { data, info } = await sharp(buffer).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  let count = 0;
  for (let index = 0; index < data.length; index += info.channels) {
    if (mapColor.every((value, channel) => Math.abs(data[index + channel] - value) <= 2)) count += 1;
  }
  return count;
}

async function runViewport({ browser, base, artifactDir }, settings) {
  const context = await browser.newContext({ viewport: settings.viewport, reducedMotion: 'no-preference', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  let mapRequests = 0;
  let phase = 'initial';
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route(/^https:\/\/(?:[a-z0-9-]+\.)*(?:google\.com|googleapis\.com)\/maps(?:\/|\?)/, async route => {
    mapRequests += 1;
    await route.fulfill({ status: 200, contentType: 'text/html', body: mapMarkup });
  });

  try {
    await page.goto(`${base}/`, { waitUntil: 'domcontentloaded' });
    await settle(page, 'about');
    await page.locator('#pcz-reject').click();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(mapRequests, 0, `${settings.name} does not preload the unopened Contact map.`);
    await page.evaluate(() => {
      SiteFrame.root().style.setProperty('--site-frame-geometry-duration', '900ms');
      SiteFrame.root().style.setProperty('--site-frame-wipe-duration', '450ms');
    });
    await page.locator('[data-site-tab="contact"]').click();
    await settle(page, 'contact');
    const iframe = page.locator('[data-persistent-contact-map] iframe');
    await page.locator('[data-contact-map-slot]').scrollIntoViewIfNeeded();
    await page.frameLocator('[data-persistent-contact-map] iframe').getByText('Local transition map').waitFor();
    await iframe.evaluate(iframe => {
      window.homeTabMap = iframe;
      window.homeTabMapWindow = iframe.contentWindow;
      window.homeTabMapLoads = 0;
      window.homeTabMapSourceChanges = 0;
      iframe.addEventListener('load', () => { window.homeTabMapLoads += 1; });
      new MutationObserver(records => { window.homeTabMapSourceChanges += records.length; })
        .observe(iframe, { attributes: true, attributeFilter: ['src'] });
    });

    phase = 'contact-departure';
    await page.locator('[data-contact-map-slot]').scrollIntoViewIfNeeded();
    await page.mouse.move(0, 0);
    const initialMap = await iframe.boundingBox();
    assert(initialMap?.height > 100 && initialMap.width > 100, `${settings.name} starts with a rendered map.`);
    const beforePixels = await countMapPixels(await page.screenshot());
    assert(beforePixels > 1000, `${settings.name} paints the loaded map fixture before switching tabs.`);
    await beginTransition(page, 'about');
    await page.waitForFunction(() => window.homeTabSamples.some(sample => sample.category === 'contact' && sample.inert));
    const departure = await page.screenshot({ path: path.join(artifactDir, `${settings.name}-contact-departure.png`) });
    assert(await countMapPixels(departure) > 1000,
      `${settings.name} keeps the loaded map painted during the departing wipe.`);
    const leaving = await endTransition(page, 'about');
    const departingMap = leaving.filter(sample => sample.category === 'contact' && sample.inert);
    assert(departingMap.length > 0 && departingMap.every(sample => sample.mapVisible && sample.mapWidth > 100 && sample.mapHeight > 100),
      `${settings.name} preserves map visibility and layout until the departing Contact content is replaced.`);
    assertIncomingAtTop(leaving, 'about', `${settings.name} Contact to About`);
    if (settings.name === 'desktop') {
      const moving = leaving.filter(sample => sample.category === 'about' && sample.moving);
      assert(moving.length > 3, 'Desktop corner checks sample the moving tab rails.');
      assert(moving.every(sample => sample.panelTopRadius >= sample.stageRadius - sample.stageBorder - 0.2 &&
        sample.panelBottomRadius >= sample.stageRadius - sample.stageBorder - 0.2),
      'The exposed panel perimeter follows both rounded stage corners throughout Contact to About.');
    }

    phase = 'contact-return';
    await beginTransition(page, 'contact');
    const returning = await endTransition(page, 'contact');
    assertIncomingAtTop(returning, 'contact', `${settings.name} returning to Contact`);
    const incomingMap = returning.filter(sample => sample.category === 'contact' && sample.moving && !sample.inert && sample.revealed);
    assert(incomingMap.length > 2 && incomingMap.every(sample => sample.mapVisible && sample.mapWidth > 100 && sample.mapHeight > 100),
      `${settings.name} reveals the retained map with the incoming Contact geometry.`);

    phase = 'scrolled-tab-return';
    assert(await scrollCurrentToEnd(page) > 100, `${settings.name} exercises a scrolled Contact view.`);
    await beginTransition(page, 'tools');
    assertIncomingAtTop(await endTransition(page, 'tools'), 'tools', `${settings.name} scrolled Contact to Tools`);
    await scrollCurrentToEnd(page);
    await beginTransition(page, 'contact');
    assertIncomingAtTop(await endTransition(page, 'contact'), 'contact', `${settings.name} Tools to previously scrolled Contact`);
    if (settings.name === 'mobile') {
      const alignment = await page.evaluate(() => {
        const tab = document.querySelector('[data-site-tab="contact"]').getBoundingClientRect();
        const headers = [...document.querySelectorAll('[data-mobile-site-masthead], [data-site-shell-header] .nav')];
        const headerBottom = Math.max(0, ...headers.filter(node => node.getClientRects().length && getComputedStyle(node).visibility !== 'hidden')
          .map(node => node.getBoundingClientRect().bottom));
        return { top: tab.top, headerBottom };
      });
      assert(Math.abs(alignment.top - alignment.headerBottom) <= 2,
        `Mobile returns to the selected tab below its header, rather than its previous document scroll: ${JSON.stringify(alignment)}`);
    }
    const identity = await iframe.evaluate(iframe => ({
      sameNode: iframe === window.homeTabMap,
      sameWindow: iframe.contentWindow === window.homeTabMapWindow,
      connected: iframe.isConnected,
      loads: window.homeTabMapLoads,
      sourceChanges: window.homeTabMapSourceChanges
    }));
    assert.deepEqual(identity, { sameNode: true, sameWindow: true, connected: true, loads: 0, sourceChanges: 0 },
      `${settings.name} preserves the loaded iframe and browsing context across repeated transitions.`);
    assert.equal(mapRequests, 1, `${settings.name} requests exactly one map document.`);
    assert.deepEqual(errors, [], `${settings.name} has no page exceptions.`);
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-contact-return.png`) });
    console.log(`Homepage tab transitions passed: ${settings.name}`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-tab-${phase}-failure.png`) }).catch(() => {});
    const samples = await page.evaluate(() => window.homeTabSamples || []).catch(() => []);
    fs.writeFileSync(path.join(artifactDir, `${settings.name}-tab-${phase}-samples.json`), JSON.stringify(samples, null, 2));
    throw error;
  } finally {
    await context.close();
  }
}

async function runHomeTabTransitionChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const settings of [
    { name: 'desktop', viewport: { width: 1440, height: 900 } },
    { name: 'mobile', viewport: { width: 390, height: 844 } }
  ]) await runViewport(options, settings);
}

async function main() {
  assert(fs.existsSync(path.join(root, 'public/index.html')), 'Run npm run build before this browser regression.');
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'home-tabs-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runHomeTabTransitionChecks({ browser, base: `http://127.0.0.1:${server.address().port}`,
      artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-home-tabs') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runHomeTabTransitionChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
