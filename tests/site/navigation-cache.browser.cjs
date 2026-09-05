/**
 * Optional browser regression; excluded from npm test. Requires a built local
 * server and an externally installed Playwright, with no added dependencies.
 * It never builds, starts a server, submits forms, or writes account data.
 *
 * PowerShell:
 *   $env:PLAYWRIGHT_MODULE = '<absolute path to installed playwright>'
 *   $env:NAVIGATION_CACHE_URL = 'http://127.0.0.1:4173'
 *   $env:NAVIGATION_CACHE_ARTIFACT_DIR = '<temporary output directory>'
 *   node tests/site/navigation-cache.browser.cjs
 *
 * Optional NAVIGATION_CACHE_ENGINES=chromium,firefox,webkit and
 * NAVIGATION_CACHE_LABEL=local. Output defaults to the system temp directory.
 * All fixture responses are restricted to a loopback server. The cases cover
 * obsolete HTML asset references, temporary script/CSS outages, real Retry
 * clicks, changed shell hashes and repeated shell execution. Failures exit nonzero.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = new URL(process.env.NAVIGATION_CACHE_URL || 'http://127.0.0.1:4173');
assert(['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname), 'Fixtures require a loopback server.');
const engines = (process.env.NAVIGATION_CACHE_ENGINES || 'chromium,firefox,webkit').split(',').map((name) => name.trim());
const label = (process.env.NAVIGATION_CACHE_LABEL || 'local').replace(/[^a-zA-Z0-9_.-]/g, '-');
const artifactDir = process.env.NAVIGATION_CACHE_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-navigation-cache');
fs.mkdirSync(artifactDir, { recursive: true });
const output = path.join(artifactDir, `navigation-cache-${label}`);
const results = [];
const manifestPattern = /(<script\b[^>]*\bid="site-route-manifest"[^>]*>)([^<]*)(<\/script>)/;

function rewriteManifest(html, update) {
  assert(manifestPattern.test(html), 'The fixture destination must contain a route manifest.');
  return html.replace(manifestPattern, (_, open, json, close) => {
    const manifest = JSON.parse(json);
    update(manifest);
    return open + JSON.stringify(manifest) + close;
  });
}

function check(record, name, value) {
  record.checks.push({ name, pass: Boolean(value) });
  assert(value, name);
}

async function ready(page) {
  await page.goto(base.href, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.SiteFrame?.root()?.isConnected && window.SiteRoutes?.current()?.root?.isConnected);
  if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
  await page.evaluate(() => {
    window.cacheQA = {
      frame: SiteFrame.root(), api: SiteFrame, navigation: SiteNavigation,
      panel: SiteFrame.root().querySelector('[data-site-route-panel]'),
      viewport: SiteFrame.viewport(), timeOrigin: performance.timeOrigin,
      errors: [], starts: 0
    };
    document.addEventListener('site:route-navigation-error', (event) => cacheQA.errors.push(event.detail.error?.message || 'Unknown route error'));
    document.addEventListener('site:navigation-start', () => { cacheQA.starts += 1; });
  });
}

async function identity(page) {
  return page.evaluate(() => {
    const q = cacheQA;
    return {
      frame: q.frame === SiteFrame.root() && q.frame.isConnected,
      api: q.api === SiteFrame,
      navigation: q.navigation === SiteNavigation,
      panel: q.panel === SiteFrame.root()?.querySelector('[data-site-route-panel]'),
      viewport: q.viewport === SiteFrame.viewport(),
      document: q.timeOrigin === performance.timeOrigin,
      singleFrame: document.querySelectorAll('.site-frame').length === 1,
      errors: q.errors,
      starts: q.starts,
      path: location.pathname,
      errorVisible: Boolean(document.querySelector('[data-site-route-error]')),
      held: SiteFrame.root()?.classList.contains('site-frame--held'),
      moving: SiteFrame.root()?.classList.contains('site-frame--moving')
    };
  });
}

function checkIdentity(record, state) {
  check(record, 'The document, frame, panel, viewport and public APIs retain identity',
    ['frame', 'api', 'navigation', 'panel', 'viewport', 'document', 'singleFrame'].every((key) => state[key]));
  check(record, 'The frame settles after navigation', !state.held && !state.moving);
}

async function runCase(browser, engine, name, action) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const record = { engine, name, checks: [], pageErrors: [] };
  page.on('pageerror', (error) => record.pageErrors.push(error.message));
  try {
    await ready(page);
    await action(page, record);
    check(record, 'No uncaught browser errors', record.pageErrors.length === 0);
    record.pass = true;
  } catch (error) {
    record.pass = false;
    record.failure = error.stack;
    await page.screenshot({ path: `${output}-${engine}-${name}.png`, fullPage: false }).catch(() => {});
  } finally {
    results.push(record);
    fs.writeFileSync(`${output}.json`, JSON.stringify({ base: base.href, results }, null, 2));
    process.stdout.write(`${JSON.stringify(record)}\n`);
    await context.close();
  }
}

function targetUrl(name) {
  return new URL(`/portfolio/retailStore?navigation-cache=${name}`, base).href;
}

async function staleAsset(page, record, kind) {
  const url = targetUrl(`stale-${kind}`);
  const missing = new URL(`/dist/navigation-cache-obsolete.${kind === 'style' ? 'css' : 'js'}`, base).href;
  let documents = 0;
  let misses = 0;
  await page.route(missing, async (route) => {
    misses += 1;
    await route.fulfill({ status: 404, contentType: 'text/plain', body: 'Asset removed by a newer build.' });
  });
  await page.route(url, async (route) => {
    documents += 1;
    const response = await route.fetch();
    let html = await response.text();
    if (documents === 1) html = rewriteManifest(html, (manifest) => {
      manifest[kind === 'style' ? 'styles' : 'scripts'].push(missing);
    });
    await route.fulfill({ response, body: html });
  });
  await page.evaluate((href) => SiteNavigation.prefetch(href), url);
  const navigated = await page.evaluate((href) => SiteNavigation.navigate(href), url);
  record.documents = documents;
  record.assetMisses = misses;
  record.state = await identity(page);
  check(record, 'The obsolete asset reference was exercised', misses >= 1);
  check(record, 'Navigation recovers from refreshed destination HTML', navigated && documents >= 2 && documents <= 3);
  check(record, 'Successful recovery leaves no Retry banner', !record.state.errorVisible);
  check(record, 'The requested destination is active', record.state.path === '/portfolio/retailStore');
  checkIdentity(record, record.state);
}

async function temporaryOutage(page, record, kind) {
  const url = targetUrl(`outage-${kind}`);
  const asset = new URL(`/dist/navigation-cache-required.${kind === 'style' ? 'css' : 'js'}`, base).href;
  let healthy = false;
  let documents = 0;
  let failedAssets = 0;
  let recoveredAssets = 0;
  await page.route(asset, async (route) => {
    if (!healthy) {
      failedAssets += 1;
      await route.fulfill({ status: 503, contentType: 'text/plain', body: 'Temporary fixture outage.' });
      return;
    }
    recoveredAssets += 1;
    await route.fulfill({
      status: 200,
      contentType: kind === 'style' ? 'text/css' : 'application/javascript',
      body: kind === 'style' ? ':root { --navigation-cache-fixture: ready; }' : 'window.navigationCacheFixture = true;'
    });
  });
  await page.route(url, async (route) => {
    documents += 1;
    const response = await route.fetch();
    const html = rewriteManifest(await response.text(), (manifest) => {
      manifest[kind === 'style' ? 'styles' : 'scripts'].push(asset);
    });
    await route.fulfill({ response, body: html });
  });
  const failed = await page.evaluate((href) => SiteNavigation.navigate(href), url);
  record.failedState = await identity(page);
  record.documentsDuringOutage = documents;
  record.failedAssets = failedAssets;
  check(record, 'A continuing outage fails after bounded attempts', failed === false && documents <= 3 && failedAssets <= 3);
  check(record, 'The original page remains usable with Retry', record.failedState.path === '/' && record.failedState.errorVisible);
  checkIdentity(record, record.failedState);
  healthy = true;
  await page.locator('[data-site-route-error]').getByRole('button', { name: 'Retry', exact: true }).click();
  await page.waitForFunction(() => !SiteNavigation.isNavigating() && location.pathname === '/portfolio/retailStore');
  record.state = await identity(page);
  record.documents = documents;
  record.recoveredAssets = recoveredAssets;
  check(record, 'A real Retry succeeds once the resource is healthy', recoveredAssets >= 1 && !record.state.errorVisible);
  if (kind === 'style') {
    check(record, 'Recovered stylesheet rules are active', await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--navigation-cache-fixture').trim() === 'ready'));
  }
  checkIdentity(record, record.state);
}

async function changedShell(page, record) {
  const shell = await page.evaluate(() => [...document.scripts].find((script) => /\/site-shell\.[^/]+\.js/.test(script.src))?.src);
  assert(shell, 'The page must load a hashed shell bundle.');
  const changed = new URL('/dist/site-shell.0badcafe.js', base).href;
  const duplicate = new URL('/dist/site-shell.navigation-cache-duplicate.js', base).href;
  const response = await page.request.get(shell);
  assert(response.ok(), 'The current shell bundle must be available.');
  const code = await response.text();
  let injected = 0;
  await page.route(changed, async (route) => {
    injected += 1;
    await route.fulfill({ status: 200, contentType: 'application/javascript', body: code });
  });
  await page.route(duplicate, (route) => route.fulfill({ status: 200, contentType: 'application/javascript', body: code }));
  const url = targetUrl('changed-shell');
  await page.route(url, async (route) => {
    const destination = await route.fetch();
    const html = rewriteManifest(await destination.text(), (manifest) => {
      manifest.scripts = manifest.scripts.map((src) => /\/site-shell\.[^/]+\.js/.test(src) ? changed : src);
    });
    await route.fulfill({ response: destination, body: html });
  });
  check(record, 'A destination with a different shell generation stays outside the current frame',
    await page.evaluate((href) => SiteNavigation.navigate(href), url) === false);
  record.changedState = await identity(page);
  record.injectedBundles = injected;
  check(record, 'Changing the shell hash does not reinject the persistent bundle', injected === 0);
  check(record, 'A different deployed generation offers explicit Reload', await page.getByRole('button', { name: 'Reload', exact: true }).isVisible());
  checkIdentity(record, record.changedState);
  await page.addScriptTag({ url: duplicate });
  record.duplicateState = await identity(page);
  checkIdentity(record, record.duplicateState);
  const starts = record.duplicateState.starts;
  await page.evaluate(() => {
    const link = document.createElement('a');
    link.href = '/contact';
    link.id = 'navigation-cache-contact';
    link.textContent = 'Open contact';
    SiteFrame.viewport().prepend(link);
  });
  await page.locator('#navigation-cache-contact').click();
  await page.waitForFunction(() => !SiteNavigation.isNavigating() && location.pathname === '/contact');
  record.state = await identity(page);
  check(record, 'Repeated shell execution does not duplicate navigation handlers', record.state.starts === starts + 1);
  check(record, 'Navigation after repeated shell execution has no Retry banner', !record.state.errorVisible);
  checkIdentity(record, record.state);
}

async function staleGeneration(page, record, { unavailable = false, changed = false } = {}) {
  const baseStyle = await page.evaluate(() => JSON.parse(document.querySelector('[data-site-route-manifest]').textContent).styles.find((href) => /\/styles\.[a-f0-9]+\.css$/.test(href)));
  const oldStyle = new URL('/dist/styles.b81be451.css', base).href;
  const oldShell = new URL('/dist/site-shell.ecbf0338.js', base).href;
  let oldCssLoads = 0;
  let documents = 0;
  let incompatibleScriptLoads = 0;
  await page.route(oldStyle, async (route) => {
    oldCssLoads += 1;
    await route.fulfill({ status: 200, contentType: 'text/css', body: '.legacy-header { color: navy; }' });
  });
  await page.route(oldShell, async (route) => {
    incompatibleScriptLoads += 1;
    await route.fulfill({ status: 200, contentType: 'application/javascript', body: 'window.legacyShellFixture = true;' });
  });
  // This old stylesheet is already present and successfully loaded, exactly as
  // in the reported stale tab. No missing-resource/404 fallback can detect it.
  await page.evaluate((href) => new Promise((resolve, reject) => {
    const link = document.createElement('link');
    link.rel = 'stylesheet'; link.href = href; link.media = 'not all';
    link.onload = resolve; link.onerror = reject; document.head.append(link);
  }), oldStyle);
  if (unavailable || changed) {
    await page.evaluate(() => SiteNavigation.navigate('/tools/text-compare'));
    await page.locator('#textcompare-original').fill('Keep this unsaved comparison.');
    await page.evaluate(() => { cacheQA.draftBody = SiteFrame.outlet(); });
  }
  const pathname = unavailable || changed ? '/portfolio/retailStore' : '/tools/text-compare';
  const url = new URL(`${pathname}?navigation-cache=generation-${unavailable ? 'offline' : changed ? 'changed' : 'stale'}`, base).href;
  await page.route(url, async (route) => {
    documents += 1;
    if (unavailable && documents > 1) { await route.abort('failed'); return; }
    const response = await route.fetch();
    let html = await response.text();
    if (documents === 1 || changed) {
      html = html.replaceAll(baseStyle, oldStyle);
      html = rewriteManifest(html, (manifest) => {
        manifest.styles = manifest.styles.map((href) => /\/styles\.[a-f0-9]+\.css$/.test(href) ? oldStyle : href);
        manifest.scripts = manifest.scripts.map((href) => /\/site-shell\.[a-f0-9]+\.js$/.test(href) ? oldShell : href);
      });
    }
    await route.fulfill({ response, body: html });
  });
  await page.evaluate((href) => {
    cacheQA.styleFrames = [];
    cacheQA.sampling = true;
    const sample = () => {
      if (!cacheQA.sampling) return;
      const link = [...document.querySelectorAll('link[rel=stylesheet]')].find((node) => node.href === new URL(href, location.href).href);
      const stage = SiteFrame.root().querySelector('[data-site-frame-stage]');
      const crumbs = document.querySelector('[data-header-breadcrumb-list]');
      cacheQA.styleFrames.push(Boolean(link && !link.disabled && !link.media && getComputedStyle(stage).display === 'grid' && getComputedStyle(crumbs).listStyleType === 'none'));
      requestAnimationFrame(sample);
    };
    requestAnimationFrame(sample);
  }, baseStyle);
  const navigated = await page.evaluate((href) => SiteNavigation.navigate(href), url);
  await page.waitForTimeout(100);
  record.generation = { navigated, documents, oldCssLoads, incompatibleScriptLoads };
  check(record, 'The old stylesheet was successfully cached without a404', oldCssLoads === 1);
  check(record, 'An old shell script is never prepared or executed', incompatibleScriptLoads === 0);
  check(record, 'A generation mismatch revalidates HTML once', documents === 2);
  if (unavailable || changed) {
    check(record, 'A failed generation refresh preserves the current tool and draft', !navigated && await page.evaluate(() =>
      location.pathname === '/tools/text-compare' && SiteFrame.outlet() === cacheQA.draftBody && document.querySelector('#textcompare-original').value === 'Keep this unsaved comparison.'));
    check(record, changed ? 'The newer generation offers Reload' : 'A failed refresh offers Retry',
      await page.getByRole('button', { name: changed ? 'Reload' : 'Retry', exact: true }).isVisible());
    if (changed) {
      await page.evaluate(() => {
        cacheQA.reloadGuards = 0;
        document.addEventListener('site:route-before-leave', (event) => {
          if (event.detail.navigationType === 'reload') { cacheQA.reloadGuards += 1; event.preventDefault(); }
        });
      });
      await page.getByRole('button', { name: 'Reload', exact: true }).click();
      await page.waitForTimeout(100);
      check(record, 'An explicit Reload honors the current leave veto and keeps the draft', await page.evaluate(() =>
        cacheQA.reloadGuards === 1 && document.querySelector('#textcompare-original').value === 'Keep this unsaved comparison.'));
    }
  } else {
    check(record, 'Revalidated Text Compare opens in the same frame', navigated && new URL(page.url()).pathname === '/tools/text-compare');
    await page.goBack();
    await page.waitForFunction(() => !SiteNavigation.isNavigating() && location.pathname === '/');
    check(record, 'Warm navigation after Back reuses the compatible generation', await page.evaluate((href) => SiteNavigation.navigate(href), url));
  }
  record.styleFrames = await page.evaluate(() => { cacheQA.sampling = false; return cacheQA.styleFrames; });
  check(record, 'Every painted frame retains current shell and breadcrumb styles', record.styleFrames.length > 0 && record.styleFrames.every(Boolean));
  record.state = await identity(page);
  checkIdentity(record, record.state);
}

(async () => {
  for (const engine of engines) {
    assert(playwright[engine]?.launch, `Unknown browser engine: ${engine}`);
    const browser = await playwright[engine].launch({ headless: true });
    try {
      for (const kind of ['script', 'style']) {
        await runCase(browser, engine, `stale-${kind}`, (page, record) => staleAsset(page, record, kind));
        await runCase(browser, engine, `outage-${kind}`, (page, record) => temporaryOutage(page, record, kind));
      }
      await runCase(browser, engine, 'changed-shell', changedShell);
      await runCase(browser, engine, 'cached-old-generation', (page, record) => staleGeneration(page, record));
      await runCase(browser, engine, 'generation-refresh-offline', (page, record) => staleGeneration(page, record, { unavailable: true }));
      await runCase(browser, engine, 'generation-reload-guard', (page, record) => staleGeneration(page, record, { changed: true }));
    } finally {
      await browser.close();
    }
  }
  if (results.some((record) => !record.pass)) process.exitCode = 1;
})().catch((error) => {
  process.stderr.write(`${error.stack}\n`);
  process.exitCode = 1;
});
