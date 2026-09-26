/**
 * Run after npm run build: node tests/site/home-closed.browser.cjs
 * Also included in the required browser smoke gate. Google Maps is stubbed so
 * checking iframe preservation never requests a billable external map load.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const root = path.resolve(__dirname, '../..');
const categories = ['about', 'projects', 'tools', 'games', 'contact'];

async function settle(page, view, category = '') {
  await page.waitForFunction(({ view, category }) => {
    const frame = window.SiteFrame?.root();
    const viewport = window.SiteFrame?.viewport();
    const path = view === 'library' ? { projects: '/portfolio', tools: '/tools', games: '/games' }[category] : '/';
    const hash = view === 'closed' ? '#closed' : view === 'overview' ? `#${category}` : '';
    const bareClosed = view === 'closed' && !location.hash;
    // Geometry ends before the viewport reveal. select() commits its URL only
    // after both finish, so the next action must wait for that complete state.
    const routeCommitted = location.pathname === path && (location.hash === hash || bareClosed);
    const revealing = viewport?.getAnimations().some(animation => animation.pending || animation.playState === 'running');
    return frame?.dataset.frameView === view && frame.dataset.frameCategory === category
      && !frame.classList.contains('site-frame--held') && !frame.classList.contains('site-frame--moving')
      && routeCommitted && !revealing && !window.SiteNavigation?.isNavigating?.()
      && SiteFrame.outlet()?.getAttribute('aria-busy') !== 'true';
  }, { view, category });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function geometry(page) {
  return page.evaluate(() => {
    const bounds = (node) => {
      const box = node.getBoundingClientRect();
      return { x: box.x, y: box.y, right: box.right, bottom: box.bottom, width: box.width, height: box.height };
    };
    return {
      width: innerWidth,
      height: innerHeight,
      documentWidth: document.documentElement.scrollWidth,
      compact: SiteFrame.root().dataset.frameCompact === 'true',
      tabs: [...document.querySelectorAll('[data-site-tab]')].filter(node => !node.hidden).map(node => ({
        id: node.dataset.siteTab, active: node.classList.contains('is-active'),
        expanded: node.getAttribute('aria-expanded'), current: node.getAttribute('aria-current'),
        inert: node.inert, tabIndex: node.tabIndex, ...bounds(node)
      }))
    };
  });
}

async function assertClosed(page, label, expectedHash = '') {
  await settle(page, 'closed');
  assert.equal(new URL(page.url()).pathname, '/', `${label} remains on the homepage.`);
  const actualHash = new URL(page.url()).hash;
  assert(Array.isArray(expectedHash) ? expectedHash.includes(actualHash) : actualHash === expectedHash,
    `${label} has the expected closed-state URL (received ${actualHash || '/'}).`);
  assert(await page.locator('.site-frame__welcome').isVisible(), `${label} displays the welcome message.`);
  assert((await page.locator('.site-frame__welcome').innerText()).trim().length >= 20,
    `${label} presents an intentional resting state with meaningful copy.`);
  assert.equal(await page.locator('.site-frame__welcome').count(), 1,
    `${label} reuses one visible welcome after the frame mounts.`);
  const browseLink = page.locator('.site-frame__welcome-browse:visible');
  assert.equal(await browseLink.count(), 1,
    `${label} exposes one project-library link beside the category bars.`);
  assert.equal(await browseLink.getAttribute('href'), '/portfolio',
    `${label} points that link to the canonical project library.`);
  assert.equal((await browseLink.innerText()).replace(/\s+/g, ' ').trim(), 'Browse all projects →',
    `${label} labels the single link clearly.`);
  assert.equal(await page.locator('[data-site-home-welcome]').count(), 0,
    `${label} removes the raw source after mounting its authored welcome copy.`);
  assert.equal(await page.locator('[data-home-accordion-item]:visible').count(), 0,
    `${label} exposes no category content.`);
  const metrics = await geometry(page);
  assert(metrics.documentWidth <= metrics.width + 1, `${label} has no horizontal page overflow.`);
  assert.deepEqual(metrics.tabs.map(tab => tab.id), categories, `${label} keeps all five categories available.`);
  for (const tab of metrics.tabs) {
    assert(!tab.active && tab.expanded === 'false' && (!tab.current || tab.current === 'false'),
      `${label} ${tab.id} has no active or expanded indication.`);
    assert(!tab.inert && tab.tabIndex === 0 && tab.width >= 44 && tab.height >= 44,
      `${label} ${tab.id} remains a usable keyboard and pointer target.`);
    assert(tab.x >= -1 && tab.right <= metrics.width + 1, `${label} ${tab.id} stays on screen.`);
    if (metrics.compact) {
      assert(Math.abs(tab.x) <= 1 && Math.abs(tab.right - metrics.width) <= 1,
        `${label} ${tab.id} is flush with both mobile edges: ${JSON.stringify(tab)}.`);
      assert(tab.height <= 100, `${label} ${tab.id} remains a compact row.`);
    } else {
      assert(tab.height > tab.width * 2, `${label} ${tab.id} retains the desktop vertical-tab treatment.`);
    }
  }
  if (metrics.compact) {
    assert(metrics.tabs.every((tab, index) => index === 0 || Math.abs(tab.y - metrics.tabs[index - 1].bottom) <= 1),
      `${label} mobile category rows form one seamless stack.`);
    const title = await page.locator('.site-frame__welcome-title').boundingBox();
    assert(title && title.x >= 16 && title.x + title.width <= metrics.width - 16,
      `${label} keeps welcome text comfortably inset while navigation meets the screen edges.`);
  }
  return metrics;
}

async function openCategory(page, category) {
  const rail = page.locator(`[data-site-tab="${category}"]`);
  if (await rail.isVisible()) await rail.click();
  else {
    await page.keyboard.press('Tab');
    await page.locator(`[data-mobile-section="${category}"]`).click();
  }
  await settle(page, 'overview', category);
  assert.equal(new URL(page.url()).hash, `#${category}`);
  assert(await page.locator(`[data-home-accordion-item="${category}"]`).isVisible(), `${category} opens its content.`);
  assert(await page.locator('.site-frame__welcome').isHidden(), `${category} replaces the resting message.`);
  assert.equal(await page.locator(`[data-site-tab="${category}"]`).getAttribute('aria-expanded'), 'true');
  const metrics = await geometry(page);
  assert(metrics.documentWidth <= metrics.width + 1, `${category} has no horizontal overflow.`);
  if (metrics.compact) {
    assert(metrics.tabs.every(tab => Math.abs(tab.x) <= 1 && Math.abs(tab.right - metrics.width) <= 1),
      `${category} keeps the mobile tabs flush with both edges while content is open.`);
  }
}

async function captureFrameAnimations(page) {
  await page.evaluate(() => {
    if (!window.closedTestAnimationCapture) {
      const animate = Element.prototype.animate;
      Element.prototype.animate = function (...args) {
        if (this.classList?.contains('site-frame') || this.closest?.('.site-frame')) {
          const options = args[1];
          window.closedTestAnimationCalls?.push({
            tab: this.classList?.contains('site-frame__tab') || false,
            duration: Number(typeof options === 'number' ? options : options?.duration) || 0
          });
        }
        return animate.apply(this, args);
      };
      window.closedTestAnimationCapture = true;
    }
    window.closedTestAnimationCalls = [];
    window.closedTestReturnDocument = document;
    window.closedTestReturnFrame = window.SiteFrame.root();
  });
}

async function assertAnimatedHomeReturn(page, label, reducedMotion) {
  await assertClosed(page, label, '');
  const result = await page.evaluate(() => ({
    documentPreserved: document === window.closedTestReturnDocument,
    framePreserved: window.SiteFrame.root() === window.closedTestReturnFrame,
    animations: window.closedTestAnimationCalls || []
  }));
  assert(result.documentPreserved && result.framePreserved,
    `${label} keeps the document and frame mounted instead of reloading the homepage.`);
  const animatedTabs = result.animations.filter(animation => animation.tab && animation.duration > 0);
  if (reducedMotion === 'reduce') {
    assert.equal(animatedTabs.length, 0, `${label} respects reduced motion.`);
  } else {
    assert(animatedTabs.length > 0, `${label} animates the category tabs into their condensed positions.`);
  }
}

async function runViewport({ browser, base, artifactDir }, settings) {
  const context = await browser.newContext({ viewport: settings.viewport, reducedMotion: settings.reducedMotion, serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const errors = [];
  let mapRequests = 0;
  let stage = 'initial-closed';
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.url().startsWith(base) && response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  await context.route(/^https:\/\/(?:[a-z0-9-]+\.)*(?:google\.com|googleapis\.com)\/maps(?:\/|\?)/, async route => {
    mapRequests += 1;
    await route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>Test map</title><p>Local test map</p>' });
  });

  try {
    const staticContext = await browser.newContext({ viewport: settings.viewport, javaScriptEnabled: false, serviceWorkers: 'block' });
    try {
      const staticPage = await staticContext.newPage();
      const staticResponse = await staticPage.goto(`${base}/`, { waitUntil: 'domcontentloaded' });
      assert.equal(staticResponse.status(), 200, `${settings.name} raw homepage responds successfully.`);
      assert(await staticPage.locator('.home-accordion__welcome h1').isVisible(),
        `${settings.name} raw HTML displays the authored introduction without JavaScript.`);
      assert.equal(await staticPage.locator('.home-accordion__welcome .site-frame__welcome-browse:visible').count(), 1,
        `${settings.name} raw HTML exposes the project-library link without JavaScript.`);
      assert.equal(await staticPage.locator('.home-accordion__noscript a:visible').count(), 3,
        `${settings.name} raw HTML keeps the remaining library destinations available without JavaScript.`);
      assert.equal(await staticPage.locator('[data-home-accordion-panel]:visible').count(), 0,
        `${settings.name} raw HTML keeps collapsed panel content hidden.`);
      assert(await staticPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
        `${settings.name} raw HTML has no horizontal overflow.`);
      await staticPage.screenshot({ path: path.join(artifactDir, `${settings.name}-static.png`), fullPage: true });
    } finally {
      await staticContext.close();
    }
    const response = await page.goto(`${base}/`, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200);
    await assertClosed(page, 'Bare homepage', '');
    await page.locator('#pcz-reject').waitFor({ state: 'visible' });
    await page.locator('#pcz-reject').click();
    await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => { window.closedTestDocument = performance.timeOrigin; window.closedTestFrame = SiteFrame.root(); });
    assert.equal(mapRequests, 0, 'The initial closed homepage does not request a map.');

    stage = 'header-home';
    await openCategory(page, 'about');
    await captureFrameAnimations(page);
    const logo = page.locator(settings.viewport.width < 960
      ? '[data-mobile-site-masthead] .mobile-site-masthead__brand'
      : '[data-site-shell-header] .brand');
    assert.equal(await logo.getAttribute('href'), '/', 'The header logo points to the clean homepage URL.');
    await logo.click();
    await assertAnimatedHomeReturn(page, `${settings.name} header logo`, settings.reducedMotion);

    stage = 'close-about';
    await openCategory(page, 'about');
    await page.locator('[data-site-tab="about"]').click();
    await assertClosed(page, settings.name);
    assert(await page.locator('[data-site-tab="about"]').evaluate(node => node === document.activeElement),
      'Closing About leaves keyboard focus on its tab.');
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-closed.png`), fullPage: true });

    stage = 'history';
    await openCategory(page, 'tools');
    await page.goBack();
    await assertClosed(page, 'Back to the closed state');
    await page.goForward();
    await settle(page, 'overview', 'tools');
    await page.locator('[data-site-tab="tools"]').click();
    await assertClosed(page, 'Close Tools');
    assert(await page.evaluate(() => closedTestDocument === performance.timeOrigin && closedTestFrame === SiteFrame.root()),
      'Closing, reopening and browser history preserve the site document and frame.');

    stage = 'reload-closed';
    await page.reload({ waitUntil: 'domcontentloaded' });
    await assertClosed(page, 'Reload the closed state');
    assert.equal(mapRequests, 0, 'Closing, history and reloading closed never request the Contact map.');

    stage = 'all-categories';
    for (const category of categories.filter(category => category !== 'contact')) {
      await openCategory(page, category);
      await page.locator(`[data-site-tab="${category}"]`).click();
      await assertClosed(page, `Close ${category}`);
    }

    stage = 'keyboard';
    await page.locator('[data-site-tab="about"]').focus();
    await page.keyboard.press('ArrowRight');
    assert(await page.locator('[data-site-tab="projects"]').evaluate(node => node === document.activeElement),
      'Arrow keys move between closed categories.');
    await page.keyboard.press('Space');
    await settle(page, 'overview', 'projects');
    await page.keyboard.press('Space');
    await assertClosed(page, 'Keyboard close');

    stage = 'rapid-toggle';
    await openCategory(page, 'about');
    await page.locator('[data-site-tab="about"]').focus();
    for (let index = 0; index < 6; index += 1) await page.keyboard.press('Enter');
    await settle(page, 'overview', 'about');
    assert(await page.locator('[data-home-accordion-item="about"]').isVisible(), 'Rapid reopening leaves its content visible.');
    await page.keyboard.press('Enter');
    await assertClosed(page, 'Rapid toggles followed by close');

    stage = 'contact-map';
    await openCategory(page, 'contact');
    const map = page.locator('[data-persistent-contact-map] iframe');
    await map.waitFor({ state: 'visible' });
    await map.scrollIntoViewIfNeeded();
    await page.frameLocator('[data-persistent-contact-map] iframe').getByText('Local test map').waitFor();
    await (await (await map.elementHandle()).contentFrame()).waitForLoadState('load');
    assert.equal(mapRequests, 1, 'First Contact activation requests its map once.');
    await map.evaluate(iframe => {
      window.closedTestMap = iframe;
      window.closedTestMapWindow = iframe.contentWindow;
      window.closedTestMapLoads = 0;
      window.closedTestMapSrcChanges = 0;
      iframe.addEventListener('load', () => { window.closedTestMapLoads += 1; });
      new MutationObserver(records => { window.closedTestMapSrcChanges += records.length; })
        .observe(iframe, { attributes: true, attributeFilter: ['src'] });
    });
    for (let index = 0; index < 2; index += 1) {
      await page.locator('[data-site-tab="contact"]').click();
      await assertClosed(page, `Close Contact ${index + 1}`);
      assert(await map.isHidden(), 'The persistent map is hidden with the Contact panel.');
      assert(await map.evaluate(iframe => iframe.isConnected && iframe === closedTestMap && iframe.contentWindow === closedTestMapWindow),
        'Closing Contact keeps the same connected iframe and browsing context.');
      await openCategory(page, 'contact');
      await map.scrollIntoViewIfNeeded();
      await page.frameLocator('[data-persistent-contact-map] iframe').getByText('Local test map').waitFor();
    }
    assert.equal(mapRequests, 1, 'Repeated Contact visits do not request another map.');
    assert.deepEqual(await page.evaluate(() => ({ loads: closedTestMapLoads, srcChanges: closedTestMapSrcChanges })),
      { loads: 0, srcChanges: 0 }, 'Reopening neither changes the map URL nor reloads its iframe.');

    stage = 'library-boundary';
    await openCategory(page, 'projects');
    const projectCard = page.locator('[data-home-accordion-item="projects"] .home-accordion__card').first();
    await page.mouse.move(0, 0);
    await projectCard.scrollIntoViewIfNeeded();
    const cardBeforeHover = await projectCard.boundingBox();
    await projectCard.hover();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const cardAfterHover = await projectCard.boundingBox();
    const cardDimensions = {
      before: cardBeforeHover && { width: cardBeforeHover.width, height: cardBeforeHover.height },
      after: cardAfterHover && { width: cardAfterHover.width, height: cardAfterHover.height }
    };
    assert(cardBeforeHover && cardAfterHover && Math.abs(cardBeforeHover.width - cardAfterHover.width) <= 1 &&
      Math.abs(cardBeforeHover.height - cardAfterHover.height) <= 1,
    `Hovering a project card keeps its dimensions stable so mobile scroll anchoring cannot move the tabs: ${JSON.stringify(cardDimensions)}`);
    await page.mouse.move(0, 0);
    await page.locator('[data-home-library-open="projects"]').click();
    await settle(page, 'library', 'projects');
    await page.locator('[data-site-tab="projects"]').click();
    await settle(page, 'overview', 'projects');
    assert(await page.locator('[data-home-accordion-item="projects"]').isVisible(), 'The library rail still returns to its category overview.');
    await page.locator('[data-site-tab="projects"]').click();
    await assertClosed(page, 'Close the returned overview');
    assert.equal(mapRequests, 1, 'Library navigation retains the map without reloading it.');
    stage = 'direct-closed-entry';
    const direct = await context.newPage();
    direct.setDefaultTimeout(12000);
    await direct.goto(`${base}/#closed`, { waitUntil: 'domcontentloaded' });
    await assertClosed(direct, 'Fresh direct closed URL', ['', '#closed']);
    await openCategory(direct, 'about');
    await direct.goBack();
    await assertClosed(direct, 'Back to a fresh closed entry', ['', '#closed']);
    await direct.close();

    const directAbout = await context.newPage();
    directAbout.setDefaultTimeout(12000);
    await directAbout.goto(`${base}/#about`, { waitUntil: 'domcontentloaded' });
    await settle(directAbout, 'overview', 'about');
    assert(await directAbout.locator('[data-home-accordion-item="about"]').isVisible(),
      'A direct About URL still opens the About content.');
    await directAbout.close();

    if (settings.name === 'desktop') {
      for (const target of [
        { label: 'Project Home breadcrumb', selector: '[data-header-breadcrumb-list] a' },
        { label: 'Project header logo', selector: '[data-site-shell-header] .brand' }
      ]) {
        stage = target.label;
        const project = await context.newPage();
        project.setDefaultTimeout(12000);
        await project.goto(`${base}/portfolio/digitGenerator`, { waitUntil: 'domcontentloaded' });
        const homeLink = project.locator(target.selector).first();
        await homeLink.waitFor({ state: 'visible' });
        assert.equal(await homeLink.getAttribute('href'), '/', `${target.label} points to the clean homepage URL.`);
        await captureFrameAnimations(project);
        await homeLink.click();
        await assertAnimatedHomeReturn(project, target.label, settings.reducedMotion);
        await project.close();
      }
    }

    assert.deepEqual(errors, [], 'The closed-state flow has no runtime or local HTTP errors.');
    console.log(`Closed homepage passed: ${settings.name} (${settings.reducedMotion}; one mocked map load).`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `${settings.name}-closed-failure.png`);
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
    error.message = `${settings.name} ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runClosedHomeChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const settings of [
    { name: 'desktop', viewport: { width: 1440, height: 900 }, reducedMotion: 'no-preference' },
    { name: 'mobile', viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' },
    { name: 'small-mobile', viewport: { width: 320, height: 720 }, reducedMotion: 'no-preference' }
  ]) await runViewport(options, settings);
}

async function main() {
  assert(fs.existsSync(path.join(root, 'public/index.html')), 'Run npm run build before this browser regression.');
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'closed-home-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runClosedHomeChecks({ browser, base: `http://127.0.0.1:${server.address().port}`,
      artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-closed') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runClosedHomeChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
