/** Run after npm run build. Exercises the real service worker with stale project CSS; never sends a message. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const root = path.resolve(__dirname, '../..');
const stylesheetPath = '/css/components/project-page.css';
const workerSource = fs.readFileSync(path.join(root, 'sw.js'), 'utf8');
const cacheVersion = workerSource.match(/const VERSION = ['"]([^'"]+)['"]/)[1];
const cacheName = `${cacheVersion}-core`;
// Preserve the rest of the project layout while emulating a pre-dock stylesheet.
const staleStyles = fs.readFileSync(path.join(root, stylesheetPath), 'utf8')
  .replace(/[^{}]*\.project-question[^{}]*\{[^{}]*\}/g, '');
assert(!staleStyles.includes('.project-question'), 'The stale stylesheet fixture must not style the question action');

async function settle(page) {
  await page.waitForFunction(() => {
    const frame = window.SiteFrame?.root();
    return document.readyState !== 'loading' && !window.SiteNavigation?.isNavigating?.()
      && (!frame || !frame.matches('.site-frame--moving, .site-frame--held'));
  });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function isolateRequests(context, base, state, omitProjectStyles = false) {
  await context.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === '/api/contact') {
      state.submissions += 1;
      return route.abort();
    }
    if (url.origin !== base || url.pathname.startsWith('/api/demos/')) return route.abort();
    if (omitProjectStyles && url.pathname === stylesheetPath) {
      return route.fulfill({ status: 200, contentType: 'text/css', body: '/* Project CSS unavailable: intrinsic SVG sizing must remain safe. */' });
    }
    if (request.resourceType() === 'document' && !request.serviceWorker() && request.frame().parentFrame()) {
      return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>Local demo fixture</title><p>Demo fixture</p>' });
    }
    // Local HTML is deliberately no-store. The explicit production-worker test
    // needs cacheable public shell responses for installation; retain real bytes.
    if (request.serviceWorker() && ['/', '/index.html', '/portfolio', '/tools', '/games'].includes(url.pathname)) {
      const response = await route.fetch();
      return route.fulfill({ response, headers: { ...response.headers(), 'cache-control': 'public, max-age=0' } });
    }
    return route.fallback();
  });
}

async function dismissConsent(page) {
  const essential = page.getByRole('button', { name: 'Essential only', exact: true });
  if (await essential.isVisible()) await essential.click();
}

async function assertQuestionLayout(page, label) {
  const question = page.locator('.project-question-link');
  await question.scrollIntoViewIfNeeded();
  await settle(page);
  const geometry = await question.evaluate(link => {
    const dock = link.closest('.project-question-dock');
    const svg = link.querySelector('svg');
    const rect = element => {
      const box = element.getBoundingClientRect();
      return { top: box.top, bottom: box.bottom, width: box.width, height: box.height, center: box.left + box.width / 2 };
    };
    return {
      link: rect(link), dock: rect(dock), icon: rect(svg),
      display: getComputedStyle(link).display,
      position: getComputedStyle(dock).position,
      radius: parseFloat(getComputedStyle(link).borderRadius),
      intrinsic: [svg.getAttribute('width'), svg.getAttribute('height')],
      viewport: rect(document.querySelector('.site-frame__viewport')),
      compact: document.querySelector('.site-frame').dataset.frameCompact === 'true',
      separateDock: dock.parentElement.classList.contains('site-frame__slot-content'),
      inProjectFlow: dock.parentElement.matches('.project-main') && dock.previousElementSibling?.matches('.project-body'),
      mobileQuestion: matchMedia('(max-width: 768px), (max-width: 959px) and (max-height: 619px)').matches,
      overflow: document.documentElement.scrollWidth - innerWidth
    };
  });
  assert.deepEqual(geometry.intrinsic, ['22', '22'], `${label}: icon has explicit intrinsic dimensions`);
  assert.equal(geometry.icon.width, 22, `${label}: icon stays 22px wide`);
  assert.equal(geometry.icon.height, 22, `${label}: icon stays 22px high`);
  assert(['flex', 'inline-flex'].includes(geometry.display), `${label}: the styled button is active`);
  assert.equal(geometry.inProjectFlow, geometry.mobileQuestion, `${label}: the mobile action follows project content`);
  assert.equal(geometry.separateDock, !geometry.mobileQuestion, `${label}: only desktop uses the reserved footer`);
  assert.equal(geometry.position, geometry.mobileQuestion ? 'static' : geometry.compact ? 'sticky' : 'relative',
    `${label}: the action uses the appropriate layout`);
  if (!geometry.compact && !geometry.mobileQuestion) {
    assert(geometry.viewport.bottom <= geometry.dock.top + 1, `${label}: the footer cannot cover project content`);
  }
  assert(geometry.radius >= 8, `${label}: the button has its rounded theme`);
  assert(geometry.link.width <= geometry.dock.width && geometry.link.height >= 44 && geometry.link.height <= 100,
    `${label}: the action remains compact and comfortably tappable: ${JSON.stringify(geometry.link)}`);
  assert(Math.abs(geometry.link.center - geometry.dock.center) <= 1, `${label}: the action is centered within its dock`);
  assert(geometry.overflow <= 1, `${label}: the action does not cause horizontal overflow`);
  return geometry;
}

async function checkCachedStyles({ browser, base, artifactDir }, viewport) {
  const label = `${viewport.width}x${viewport.height}`;
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'allow', isMobile: viewport.width < 600, hasTouch: viewport.width < 1000 });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const state = { submissions: 0 };
  const errors = [];
  const stylesheetResponses = [];
  const evidence = { viewport, cacheName, stylesheetResponses };
  let stage = 'register service worker';
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    const url = new URL(response.url());
    if (url.pathname === stylesheetPath) stylesheetResponses.push({ url: response.url(), worker: response.fromServiceWorker(), status: response.status() });
  });
  await isolateRequests(context, base, state);
  try {
    await page.goto(`${base}/portfolio/website`);
    await dismissConsent(page);
    await settle(page);
    // Local development intentionally skips registration; explicitly enable the production worker here.
    await page.evaluate(async () => {
      await navigator.serviceWorker.register('/sw.js');
      await navigator.serviceWorker.ready;
      if (!navigator.serviceWorker.controller) {
        await new Promise(resolve => navigator.serviceWorker.addEventListener('controllerchange', resolve, { once: true }));
      }
    });
    assert(await page.evaluate(() => Boolean(navigator.serviceWorker.controller)), `${label}: the real service worker controls this page`);

    stage = 'seed stale stylesheet';
    await page.evaluate(async ({ cacheName, cacheVersion, stylesheetPath, staleStyles }) => {
      for (const name of await caches.keys()) {
        const cache = await caches.open(name);
        for (const request of await cache.keys()) {
          if (new URL(request.url).pathname === stylesheetPath) await cache.delete(request);
        }
      }
      const cache = await caches.open(cacheName);
      await cache.put(stylesheetPath, new Response(staleStyles, { headers: { 'Content-Type': 'text/css', 'X-Regression-Fixture': 'stale-project-css' } }));
      // A legitimate cache entry includes its size and age record. Keep this
      // stale fixture eligible so the test proves URL versioning, not eviction.
      const metadata = await caches.open(`${cacheVersion}-metadata`);
      const key = `${location.origin}/__site-cache-metadata__/core`;
      const prior = await metadata.match(key);
      const entries = prior ? await prior.json() : [];
      const url = new URL(stylesheetPath, location.origin).href;
      const current = entries.filter(entry => new URL(entry.url).pathname !== stylesheetPath);
      current.push({ url, bytes: new TextEncoder().encode(staleStyles).length, storedAt: Date.now(), usedAt: Date.now() });
      await metadata.put(key, new Response(JSON.stringify(current), { headers: { 'Content-Type': 'application/json' } }));
    }, { cacheName, cacheVersion, stylesheetPath, staleStyles });
    stylesheetResponses.length = 0;

    stage = 'load current page over stale asset cache';
    await page.reload();
    await settle(page);
    const assetReferences = await page.evaluate(stylesheetPath => {
      const links = [...document.querySelectorAll('link[rel="stylesheet"]')]
        .map(link => link.href).filter(href => new URL(href).pathname === stylesheetPath);
      const manifest = JSON.parse(document.querySelector('[data-site-route-manifest]').textContent);
      return { links, manifest: manifest.styles.filter(href => new URL(href, location.href).pathname === stylesheetPath) };
    }, stylesheetPath);
    assert.equal(assetReferences.links.length, 1, `${label}: one project stylesheet is loaded`);
    assert.match(new URL(assetReferences.links[0]).search, /^\?v=[a-f0-9]{8,64}$/, `${label}: component CSS uses a content version`);
    assert.deepEqual(assetReferences.manifest.map(href => new URL(href, base).href), assetReferences.links,
      `${label}: soft navigation requests the same versioned project stylesheet`);
    assert(stylesheetResponses.some(response => response.worker && new URL(response.url).search),
      `${label}: the versioned stylesheet passes through the real service worker`);
    const staleCacheSurvived = await page.evaluate(async ({ cacheName, stylesheetPath }) => {
      const response = await (await caches.open(cacheName)).match(stylesheetPath);
      return response?.headers.get('X-Regression-Fixture') === 'stale-project-css';
    }, { cacheName, stylesheetPath });
    assert(staleCacheSurvived, `${label}: the old bare asset still exists, proving the versioned URL bypassed it`);
    evidence.assetReferences = assetReferences;
    evidence.initial = await assertQuestionLayout(page, `${label} initial`);
    await page.screenshot({ path: path.join(artifactDir, `question-cache-${label}.png`) });
    if (viewport.width === 390) {
      stage = 'responsive question placement';
      await page.setViewportSize({ width: 980, height: 1000 });
      await settle(page);
      evidence.resizedDesktop = await assertQuestionLayout(page, `${label} resized desktop`);
      await page.setViewportSize(viewport);
      await settle(page);
      evidence.resizedMobile = await assertQuestionLayout(page, `${label} resized mobile`);
    }

    stage = 'open and close contact dialog';
    await page.locator('.project-question-link').click();
    const modal = page.locator('#contact-modal.active');
    await modal.waitFor();
    assert.match(await modal.locator('#contact-message').inputValue(), /^Hi Daniel, I have a question about danielshort\.me:/);
    await page.screenshot({ path: path.join(artifactDir, `question-cache-${label}-modal.png`) });
    await modal.getByRole('button', { name: 'Close dialog', exact: true }).click();
    await modal.waitFor({ state: 'hidden' });
    await page.waitForFunction(() => getComputedStyle(document.querySelector('.project-question-dock')).opacity === '1');
    evidence.closed = await assertQuestionLayout(page, `${label} after dismissal`);

    stage = 'navigate to the next project';
    const next = page.locator('.project-next-link');
    const target = new URL(await next.getAttribute('href'), base).href;
    await page.evaluate(() => { window.__questionCacheSameDocument = true; });
    await next.click();
    await page.waitForURL(target);
    await settle(page);
    assert(await page.evaluate(() => window.__questionCacheSameDocument), `${label}: next-project navigation remains within the current document`);
    evidence.next = await assertQuestionLayout(page, `${label} next project`);
    await page.screenshot({ path: path.join(artifactDir, `question-cache-${label}-next.png`) });

    stage = 'restore the previous project';
    await page.goBack();
    await page.waitForURL(`${base}/portfolio/website`);
    await settle(page);
    assert.equal(await page.locator('.project-question-dock').count(), 1, `${label}: history restores one question footer`);
    assert.equal(await page.locator('.project-question-title').innerText(), 'danielshort.me', `${label}: history restores the correct project question`);
    evidence.restored = await assertQuestionLayout(page, `${label} restored project`);
    assert.equal(state.submissions, 0, `${label}: no contact message was submitted`);
    assert.deepEqual(errors, [], `${label}: no page errors occurred`);
    fs.writeFileSync(path.join(artifactDir, `question-cache-${label}.json`), JSON.stringify(evidence, null, 2));
    console.log(`Project question cache passed: ${label}, stale worker asset bypass, compact icon/button, modal, soft navigation.`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `question-cache-${label}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${label} ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function checkIntrinsicFallback({ browser, base, artifactDir }) {
  const context = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  const state = { submissions: 0 };
  await isolateRequests(context, base, state, true);
  try {
    await page.goto(`${base}/portfolio/website`);
    await dismissConsent(page);
    await settle(page);
    const icon = page.locator('.project-question-link svg');
    await icon.scrollIntoViewIfNeeded();
    const box = await icon.boundingBox();
    assert.equal(box.width, 22, 'Without project CSS, the SVG still occupies only 22px of width');
    assert.equal(box.height, 22, 'Without project CSS, the SVG still occupies only 22px of height');
    assert.equal(await page.locator('.project-question-link').evaluate(link => getComputedStyle(link).display), 'inline',
      'The fallback case really omits the project button styles');
    await page.screenshot({ path: path.join(artifactDir, 'question-intrinsic-fallback.png') });
    assert.equal(state.submissions, 0);
    console.log('Project question intrinsic fallback passed: 22px icon with project CSS unavailable.');
  } finally {
    await context.close();
  }
}

async function runProjectQuestionCacheChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const viewport of [{ width: 390, height: 844 }, { width: 980, height: 1740 }, { width: 1440, height: 1000 }]) {
    await checkCachedStyles(options, viewport);
  }
  await checkIntrinsicFallback(options);
}

module.exports = runProjectQuestionCacheChecks;

if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'project-question-cache-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-project-question-cache');
    await runProjectQuestionCacheChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir });
    console.log(`Evidence: ${artifactDir}`);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
