/**
 * Required Chromium smoke gate for the built site. Uses the repository server's
 * real rewrites, headers and public output; does not build or start a watcher.
 * Run after npm run build: npm run test:browser
 * Install the matching browser once: npx playwright install chromium
 * Optional PLAYWRIGHT_MODULE / BROWSER_EXECUTABLE_PATH reuse installed runtimes.
 * Screenshots and logs default to the system temporary directory.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const runResponsiveSpacingChecks = require('./responsive-spacing.browser.cjs');
const runClosedHomeChecks = require('./home-closed.browser.cjs');

const root = path.resolve(__dirname, '../..');
const personalContent = require('../../content/audiences/personal.json');
const aboutTimeline = personalContent.page.sections.find(section => section.type === 'home-accordion')
  .props.categories.find(category => category.id === 'about').timeline;
const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-smoke');
const cases = [];
fs.mkdirSync(artifactDir, { recursive: true });

async function settle(page) {
  await page.waitForFunction(() => {
    const frame = window.SiteFrame?.root();
    return frame?.isConnected && window.SiteRoutes?.current()?.root?.isConnected
      && !window.SiteNavigation?.isNavigating?.()
      && !frame.classList.contains('site-frame--held')
      && !frame.classList.contains('site-frame--moving')
      && SiteFrame.outlet()?.getAttribute('aria-busy') !== 'true'
      && !document.querySelector('[data-site-route-error]')
      && !document.querySelector('[data-site-frame-loading]:not([hidden])');
  });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function assertLayout(page) {
  const layout = await page.evaluate(() => {
    const tabs = ['about', 'projects', 'tools', 'games', 'resume', 'contact'].map(category => {
      const node = document.querySelector(`[data-site-tab="${category}"]`);
      if (!node) return null;
      const box = node?.getBoundingClientRect();
      return {
        category, hidden: node?.hidden, inert: node?.inert, tabIndex: node?.tabIndex,
        active: node?.classList.contains('is-active'),
        box: box ? { left: box.left, top: box.top, right: box.right, bottom: box.bottom, width: box.width, height: box.height } : null
      };
    }).filter(Boolean);
    const frame = SiteFrame.root();
    const viewportBox = SiteFrame.viewport()?.getBoundingClientRect();
    const mainBox = document.querySelector('#main')?.getBoundingClientRect();
    return {
      width: innerWidth,
      height: innerHeight,
      pageWidth: document.documentElement.scrollWidth,
      mainText: document.querySelector('#main')?.innerText.trim().length || 0,
      demo: document.body.classList.contains('project-demo-wrapper-page'),
      embeddedDemo: Boolean(document.querySelector('#main iframe[title][src]')),
      mainBox: mainBox ? { top: mainBox.top, bottom: mainBox.bottom } : null,
      frames: document.querySelectorAll('.site-frame').length,
      category: frame?.dataset.frameCategory,
      audience: frame?.dataset.frameAudience,
      overview: frame?.dataset.frameHome === 'true' && frame?.dataset.frameView === 'overview',
      compact: frame?.dataset.frameCompact === 'true',
      viewportBox: viewportBox ? { top: viewportBox.top, bottom: viewportBox.bottom, left: viewportBox.left, right: viewportBox.right } : null,
      tabs
    };
  });
  assert(layout.demo ? layout.embeddedDemo : layout.mainText > 80,
    'The route must contain meaningful content or its named, loaded demo iframe.');
  assert(layout.mainBox && layout.mainBox.top < layout.height - 120 && layout.mainBox.bottom > 80,
    `Page content remains visible below the compact navigation: ${JSON.stringify(layout)}`);
  assert.equal(layout.frames, 1, 'Exactly one shared frame is mounted.');
  assert(layout.pageWidth <= layout.width + 1, `Document overflows horizontally: ${JSON.stringify(layout)}`);
  const visibleTabs = layout.tabs.filter(tab => !tab.hidden);
  assert.deepEqual(visibleTabs.map(tab => tab.category), layout.audience !== 'personal'
    ? ['about', 'projects', 'resume', 'contact'] : layout.overview
      ? ['about', 'projects', 'tools', 'games', 'contact'] : [layout.category],
  'Navigation exposes the correct categories for the audience and route.');
  assert.equal(layout.tabs.filter(tab => tab.active).length, 1, 'Exactly one category is active.');
  for (const tab of layout.tabs.filter(tab => tab.hidden)) {
    assert(tab.inert && tab.tabIndex === -1 && tab.box?.width === 0 && tab.box?.height === 0,
      `${tab.category} is removed from layout and keyboard navigation outside the overview.`);
  }
  for (const { category, box, inert, tabIndex } of visibleTabs) {
    assert(box && box.width >= 40 && box.height >= 40 && !inert && tabIndex === 0,
      `${category} has a usable, keyboard-accessible target.`);
    assert(box.left >= -1 && box.right <= layout.width + 1, `${category} remains within the screen.`);
  }
  if (!layout.compact) {
    assert(visibleTabs.every(tab => tab.box.width <= 96 && tab.box.height > tab.box.width * 3),
      'Desktop navigation retains tall, narrow vertical rails.');
    assert(Math.max(...visibleTabs.map(tab => tab.box.top)) - Math.min(...visibleTabs.map(tab => tab.box.top)) <= 2,
      'Desktop rails align along the top of the frame.');
    assert(visibleTabs.every((tab, index) => index === 0 || tab.box.left > visibleTabs[index - 1].box.left),
      'Desktop rails keep the original category order around the expanded panel.');
  } else {
    if (layout.audience === 'personal') {
      assert(visibleTabs.every(tab => tab.box.height <= 100 && tab.box.width >= layout.width - 12),
        'Personal mobile tabs remain full-width compact rows.');
    } else {
      assert(visibleTabs.every(tab => tab.box.height <= 100 && tab.box.width >= layout.width / visibleTabs.length - 12),
        'Professional mobile tabs retain usable compact columns.');
      assert(Math.max(...visibleTabs.map(tab => tab.box.top)) - Math.min(...visibleTabs.map(tab => tab.box.top)) <= 2,
        'Professional mobile categories align in their navigation row.');
    }
    if (layout.overview) {
      assert(visibleTabs.every((tab, index) => index === 0 || tab.box.top >= visibleTabs[index - 1].box.bottom - 2),
        'The mobile overview stacks all five category tabs vertically.');
      const activeIndex = visibleTabs.findIndex(tab => tab.active);
      assert(layout.viewportBox.top >= visibleTabs[activeIndex].box.bottom - 2,
        'Mobile overview content expands directly beneath the active category.');
      const nextTab = visibleTabs[activeIndex + 1];
      if (nextTab) assert(layout.viewportBox.bottom <= nextTab.box.top + 2,
        'The next mobile category remains below the expanded content.');
    }
  }
  return layout;
}

async function scrollState(page) {
  return page.evaluate(() => ({
    documentTop: scrollY,
    documentHeight: document.scrollingElement.scrollHeight,
    documentRange: Math.max(0, document.scrollingElement.scrollHeight - innerHeight),
    frameTop: SiteFrame.viewport().scrollTop,
    frameHeight: SiteFrame.viewport().scrollHeight,
    frameClientHeight: SiteFrame.viewport().clientHeight,
    toolbarHeight: SiteFrame.root().querySelector('.site-frame__toolbar').getBoundingClientRect().height,
    frameRange: Math.max(0, SiteFrame.viewport().scrollHeight - SiteFrame.viewport().clientHeight),
    frameOverflow: getComputedStyle(SiteFrame.viewport()).overflowY,
    frameBox: (() => {
      const box = SiteFrame.viewport().getBoundingClientRect();
      return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
    })(),
    stageBox: (() => {
      const box = SiteFrame.root().querySelector('[data-site-frame-stage]').getBoundingClientRect();
      return { left: box.left, top: box.top, width: box.width, height: box.height };
    })(),
    compact: SiteFrame.root().dataset.frameCompact === 'true',
    fit: SiteFrame.root().dataset.frameFit,
    demo: document.body.classList.contains('project-demo-wrapper-page'),
    footerHeight: document.querySelector('[data-site-shell-footer]')?.getBoundingClientRect().height || 0,
    width: innerWidth,
    height: innerHeight
  }));
}

async function resetScroll(page) {
  await page.evaluate(() => {
    scrollTo({ top: 0, behavior: 'instant' });
    SiteFrame.viewport().scrollTo({ top: 0, behavior: 'instant' });
  });
  await page.waitForFunction(() => scrollY === 0 && SiteFrame.viewport().scrollTop === 0);
}

async function wheelToBottom(page, label, owner = 'document') {
  const before = await scrollState(page);
  const frameOwnsScroll = owner === 'frame';
  const topKey = frameOwnsScroll ? 'frameTop' : 'documentTop';
  const rangeKey = frameOwnsScroll ? 'frameRange' : 'documentRange';
  if (frameOwnsScroll) {
    assert.equal(before.fit, 'viewport', `${label} retains the desktop viewport frame.`);
    assert(/^(auto|scroll)$/.test(before.frameOverflow), `${label} scrolls in the frame viewport.`);
  } else {
    assert(before.fit === 'document' || before.compact, `${label} uses document scrolling.`);
  }
  assert(before[rangeKey] > 40, `${label} must have real content to scroll.`);
  // The frame's content gutter keeps the wheel on the parent scrollport when
  // a project contains an independently interactive iframe.
  const pointerX = frameOwnsScroll ? before.frameBox.left + 12 : before.width / 2;
  const pointerY = frameOwnsScroll
    ? (Math.max(80, before.frameBox.top) + Math.min(before.height - 20, before.frameBox.bottom)) / 2
    : before.height * .65;
  await page.mouse.move(pointerX, pointerY);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const current = await scrollState(page);
    if (current[rangeKey] - current[topKey] <= 3) break;
    await page.mouse.wheel(0, Math.max(800, before.height * 1.5));
    await page.waitForFunction(({ previous, frameOwned }) => {
      const node = frameOwned ? SiteFrame.viewport() : document.scrollingElement;
      const top = frameOwned ? node.scrollTop : scrollY;
      const range = node.scrollHeight - (frameOwned ? node.clientHeight : innerHeight);
      return top > previous + 2 || range - top <= 3;
    }, { previous: current[topKey], frameOwned: frameOwnsScroll }, { timeout: 2500 });
    const otherTopKey = frameOwnsScroll ? 'documentTop' : 'frameTop';
    assert(Math.abs((await scrollState(page))[otherTopKey] - before[otherTopKey]) <= 1,
      `${label} must scroll its ${owner} owner without moving the other scroll container.`);
  }
  const after = await scrollState(page);
  assert(after[topKey] > before[topKey] + 20, `${label} responds to native wheel input.`);
  assert(after[rangeKey] - after[topKey] <= 3, `${label} can reach the ${owner} bottom.`);
  return { owner, before, after };
}

async function assertSharedStage(page, homeStage, label) {
  await assertLayout(page);
  const state = await scrollState(page);
  assert.equal(state.fit, 'viewport', `${label} uses the shared viewport-fit frame.`);
  if (!state.compact) {
    for (const property of ['left', 'top', 'width', 'height']) {
      assert(Math.abs(state.stageBox[property] - homeStage.stageBox[property]) <= 2,
        `${label} preserves the homepage stage ${property}: ${JSON.stringify({ home: homeStage.stageBox, project: state.stageBox })}`);
    }
    assert(Math.abs(state.frameClientHeight + state.toolbarHeight - homeStage.frameClientHeight - homeStage.toolbarHeight) <= 2,
      `${label} keeps the viewport and breadcrumb toolbar inside the homepage's fixed content height: ${JSON.stringify({ home: homeStage.frameClientHeight, viewport: state.frameClientHeight, toolbar: state.toolbarHeight })}`);
    assert(state.documentRange <= 1 && state.documentTop === 0,
      `${label} keeps long content inside the frame without document overflow.`);
  } else {
    assert.equal(state.frameOverflow, 'visible', `${label} expands naturally on mobile.`);
    assert(state.frameRange <= 1, `${label} has no competing mobile frame scrollbar.`);
  }
  const contentBoxes = await page.locator('#main .home-library__card:visible, #main .project-star-value:visible')
    .evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, overflow: node.scrollWidth - node.clientWidth })));
  assert(contentBoxes.every(box => box.width >= 200 && box.overflow <= 1),
    `${label} keeps cards and project summary text readable: ${JSON.stringify(contentBoxes)}`);
  return state;
}

async function checkSharedScroll(page, homeStage, label, screenshotName, { requireScroll = true } = {}) {
  await resetScroll(page);
  const geometry = await assertSharedStage(page, homeStage, label);
  const range = geometry.compact ? geometry.documentRange : geometry.frameRange;
  let scroll = null;
  if (requireScroll || range > 40) {
    scroll = await wheelToBottom(page, label, geometry.compact ? 'document' : 'frame');
    await assertSharedStage(page, homeStage, `${label} after scrolling`);
  } else if (!geometry.compact) {
    const cards = await page.locator('#main .home-library__card:visible').evaluateAll(nodes => nodes.map(node => {
      const box = node.getBoundingClientRect();
      return { top: box.top, bottom: box.bottom };
    }));
    assert(cards.every(box => box.top >= geometry.frameBox.top - 1 && box.bottom <= geometry.frameBox.bottom + 1),
      `${label} shows every library card when scrolling is unnecessary.`);
  }
  await page.screenshot({ path: path.join(artifactDir, screenshotName), fullPage: false });
  await resetScroll(page);
  return { geometry, scroll };
}

async function checkDemo(page, homeStage, settings) {
  const iframe = page.locator('#main .project-demo-wrapper-iframe');
  const demo = page.frameLocator('#main .project-demo-wrapper-iframe');
  await demo.locator('#status-pill[data-state="ok"]').waitFor();
  assert.equal(await demo.locator('h1').innerText(), 'Baby names', 'The demo loads its local data and interactive content.');
  const dimensions = await iframe.evaluate(node => {
    const box = node.getBoundingClientRect();
    return { width: box.width, height: box.height };
  });
  assert(dimensions.width >= 280 && dimensions.height >= 400, 'The embedded demo has a readable workspace.');
  await demo.locator('#ratings-panel summary').click();
  assert(await demo.locator('#ratings-panel').evaluate(node => node.open), 'The embedded demo disclosure opens.');
  await resetScroll(page);
  await demo.locator('body').evaluate(() => scrollTo({ top: 0, behavior: 'instant' }));
  const geometry = await assertSharedStage(page, homeStage, 'Expanded demo wrapper');
  if (geometry.compact) {
    await page.waitForFunction(() => document.body.dataset.projectDemoAutosize === 'true');
    const result = await checkSharedScroll(page, homeStage, 'Mobile demo document', `${settings.name}-demo-bottom.png`);
    assert((await iframe.boundingBox()).height > dimensions.height + 40,
      'Opening demo details grows the mobile iframe with its document.');
    return result;
  }
  const embeddedState = () => demo.locator('body').evaluate(() => ({
    top: scrollY, range: document.scrollingElement.scrollHeight - innerHeight,
    width: innerWidth, pageWidth: document.documentElement.scrollWidth
  }));
  const before = await embeddedState();
  assert(before.range > 40 && before.pageWidth <= before.width + 1, 'The desktop demo has vertical content without horizontal overflow.');
  const box = await iframe.boundingBox();
  await page.mouse.move(box.x + 12, box.y + box.height * .6);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const current = await embeddedState();
    if (current.range - current.top <= 3) break;
    await page.mouse.wheel(0, 1000);
    await demo.locator('body').evaluate((node, previous) => new Promise((resolve, reject) => {
      const deadline = performance.now() + 2000;
      const check = () => {
        if (scrollY > previous + 2 || document.scrollingElement.scrollHeight - innerHeight - scrollY <= 3) resolve();
        else if (performance.now() > deadline) reject(new Error('Embedded demo did not respond to wheel input.'));
        else requestAnimationFrame(check);
      };
      requestAnimationFrame(check);
    }), current.top);
  }
  const after = await embeddedState();
  assert(after.top > before.top + 20 && after.range - after.top <= 3, 'Native wheel reaches the embedded demo bottom.');
  await assertSharedStage(page, homeStage, 'Demo after embedded scrolling');
  assert.equal(await page.evaluate(() => SiteFrame.viewport().scrollTop), 0, 'Embedded scrolling leaves the outer frame fixed.');
  await page.screenshot({ path: path.join(artifactDir, `${settings.name}-demo-bottom.png`), fullPage: false });
  return { geometry, scroll: { owner: 'iframe', before, after } };
}

async function runViewport(browser, base, settings) {
  const context = await browser.newContext({ viewport: settings.viewport, reducedMotion: settings.reducedMotion });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const errors = [];
  const unavailableServices = [];
  const isExpectedUnavailable = (url, status) => {
    // The anonymous CI server intentionally has no AWS configuration or login.
    // Only this observed read-only config failure is an expected limitation.
    try { return status === 401 && new URL(url).pathname === '/api/tools/transcribe/config'; }
    catch (_) { return false; }
  };
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() !== 'error') return;
    if (isExpectedUnavailable(message.location().url, 401) && /Failed to load resource.*401/.test(message.text())) {
      unavailableServices.push({ url: message.location().url, message: message.text() });
    } else errors.push(message.text());
  });
  page.on('response', response => {
    if (!response.url().startsWith(base) || response.status() < 400) return;
    if (isExpectedUnavailable(response.url(), response.status())) unavailableServices.push({ url: response.url(), status: response.status() });
    else errors.push(`${response.status()} ${response.url()}`);
  });
  let stage = 'home';
  try {
    const response = await page.goto(base + '/', { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200);
    await settle(page);
    assert.match(await page.title(), /Daniel Short/);
    assert.equal(new URL(page.url()).pathname, '/');
    await page.locator('#pcz-reject').waitFor({ state: 'visible' });
    await page.locator('#pcz-reject').click();
    await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
    await page.evaluate(() => document.fonts.ready);
    await resetScroll(page);
    await page.evaluate(() => { window.smokeFrame = SiteFrame.root(); window.smokeTimeOrigin = performance.timeOrigin; });
    const homeLayout = await assertLayout(page);
    const homeStage = await scrollState(page);
    const portrait = page.locator('.home-about__portrait');
    await portrait.evaluate(image => image.decode());
    assert.equal(await portrait.getAttribute('alt'), 'Daniel Short', 'The About portrait has a meaningful alternative.');
    assert((await portrait.boundingBox()).width >= 80, 'The About portrait remains recognizable.');
    assert.match(await page.locator('.home-about__intro h2').innerText(), /Hi, I’m Daniel/);
    const stories = page.locator('[data-home-about-connection]');
    assert.equal(await stories.count(), 3, 'About shows three real personal-interest connections.');
    const expectedStories = [
      { id: 'ai', href: '/tools', text: /AI & machine learning/ },
      { id: 'family', href: '/portfolio/babynames', text: /Family/ },
      { id: 'french-horn', href: '/portfolio/sheetMusicUpscale', text: /French horn[\s\S]*20 years/ }
    ];
    for (const { id, href, text } of expectedStories) {
      const story = page.locator(`[data-home-about-connection="${id}"]`);
      assert.match(await story.locator('.home-about__interest').innerText(), text);
      const projectLink = story.locator('a.home-about__project');
      assert.equal(await projectLink.getAttribute('href'), href, `${id} points directly to the work it inspired.`);
      assert(await projectLink.isVisible() && (await projectLink.innerText()).length > 20,
        `${id} presents a readable project title and explanation.`);
    }
    const aboutColumns = await page.locator('.home-about').evaluate(node => {
      const personal = node.querySelector('.home-about__personal').getBoundingClientRect();
      const timeline = node.querySelector('.home-timeline').getBoundingClientRect();
      return { personal: { right: personal.right, bottom: personal.bottom }, timeline: { left: timeline.left, top: timeline.top } };
    });
    assert(homeLayout.compact ? aboutColumns.timeline.top >= aboutColumns.personal.bottom - 1
      : aboutColumns.timeline.left >= aboutColumns.personal.right - 1,
    'Personal stories and the timeline use two desktop columns and a natural mobile stack.');
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-home.png`), fullPage: false });

    stage = 'timeline-content';
    const timeline = page.locator('.home-about .home-timeline');
    assert(await timeline.isVisible(), 'Experience and learning stays mounted beside the personal stories.');
    assert.equal(await timeline.locator(':scope > h3').innerText(), 'Experience & learning');
    assert.equal(await timeline.getAttribute('data-home-timeline-layout'), 'resume');
    assert.equal(await timeline.locator('details, summary, [data-home-timeline-scroller], [data-home-timeline-year]').count(), 0,
      'The resume has no disclosure, nested scroll region, or prominent year chapters.');
    assert.equal(await timeline.locator('[data-home-timeline-item]:visible').count(), 10,
      'All ten milestones stay rendered without opening a dropdown.');
    assert.equal(await timeline.locator('img:visible').count(), 0, 'Organization logos do not crowd the resume.');
    assert.deepEqual(await timeline.locator('[data-home-background-section] > h4').allTextContents(),
      ['Experience', 'Education', 'Credentials'], 'The resume exposes all three labelled sections.');
    const expectedSectionItems = {
      experience: ['visit-grand-junction', 'randall-reilly', 'target'],
      education: ['eastern-ms-data-science', 'purdue-bs-data-analytics'],
      credentials: ['google-advanced-data-analytics', 'google-data-analytics', 'google-analytics', 'ibm-machine-learning', 'ibm-data-analyst']
    };
    for (const [section, ids] of Object.entries(expectedSectionItems)) {
      assert.deepEqual(await timeline.locator(`[data-home-background-section="${section}"] [data-home-timeline-item]`)
        .evaluateAll(nodes => nodes.map(node => node.dataset.homeTimelineItem)), ids,
      `${section} retains its complete milestones in the approved order.`);
    }
    assert.deepEqual(await timeline.locator('[data-home-credential-issuer] > h5').allTextContents(), ['Google', 'IBM'],
      'Credentials are grouped visibly by issuer.');
    const roleStyles = await timeline.locator('[data-home-background-section="experience"] [data-home-timeline-item]')
      .evaluateAll(nodes => nodes.map(node => {
        const style = getComputedStyle(node);
        const titleStyle = getComputedStyle(node.querySelector('.home-background__title'));
        const icon = node.querySelector('.home-background__icon').getBoundingClientRect();
        return { className: node.className, columns: style.gridTemplateColumns.split(' ').length,
          background: style.backgroundColor, paddingTop: style.paddingTop,
          titleFontSize: titleStyle.fontSize, titleFontWeight: titleStyle.fontWeight,
          iconWidth: icon.width, iconHeight: icon.height };
      }));
    assert(roleStyles.length === 3 && roleStyles.every(style => JSON.stringify(style) === JSON.stringify(roleStyles[0])),
      'Current and past roles share the same row treatment.');
    assert.match(await timeline.locator('[data-home-timeline-item="visit-grand-junction"] .home-background__date').innerText(), /Present/,
      'Current work retains its ongoing date in the same secondary date field as past work.');
    const linkedMilestones = aboutTimeline.items.filter(item => item.href);
    assert.equal(await timeline.locator('a[href]').count(), linkedMilestones.length,
      'Every authored degree and certificate link remains available.');
    for (const item of aboutTimeline.items) {
      const milestone = timeline.locator(`[data-home-timeline-item="${item.id}"]`);
      assert.equal(await milestone.count(), 1, `${item.id} appears exactly once.`);
      assert.deepEqual(await milestone.locator('time').evaluateAll(nodes => nodes.map(node => node.dateTime)),
        [item.date, ...(item.endDate ? [item.endDate] : [])], `${item.id} preserves exact semantic dates.`);
      const entry = milestone.locator('.home-background__entry, .home-background__credential-link');
      const descriptionId = await entry.getAttribute('aria-describedby');
      assert(descriptionId && await milestone.locator(`[id="${descriptionId}"]`).count() === 1,
        `${item.id} retains its accessible date description.`);
      if (item.href) {
        assert.equal(await entry.getAttribute('href'), item.href, `${item.id} retains its destination.`);
        assert.equal(await entry.getAttribute('target'), '_blank', `${item.id} opens externally.`);
        assert.match(await entry.getAttribute('rel'), /\bnoopener\b/);
        assert.match(await entry.getAttribute('rel'), /\bnoreferrer\b/);
      }
      if (item.type === 'certification') {
        assert(await milestone.getByRole('link', { name: item.title, exact: true }).isVisible(),
          `${item.id} retains its full accessible credential name.`);
        assert.equal((await milestone.locator('.home-background__title-compact').innerText()).trim(), item.credentialLabel);
        assert.equal(await milestone.locator('.home-background__credential-date.visually-hidden').count(), 1,
          `${item.id} keeps the earned date accessible without emphasizing it visually.`);
        assert.match(await entry.getAttribute('title'), /^Earned /, `${item.id} exposes its earned date on hover.`);
      }
    }
    await assertSharedStage(page, homeStage, 'About with the resume mounted');
    const timelineEntries = await timeline.locator('[data-home-timeline-item]').evaluateAll(nodes => nodes.map(node => {
      const box = node.getBoundingClientRect();
      const title = node.querySelector('.home-background__title, .home-background__credential-label');
      return { top: box.top, left: box.left, right: box.right, bottom: box.bottom, width: box.width, height: box.height,
        credential: node.classList.contains('home-background__credential'),
        titleHeight: title.getBoundingClientRect().height, fontSize: parseFloat(getComputedStyle(title).fontSize) };
    }));
    // Credential groups may share one row; full-width career and education rows
    // retain the original 160px readability minimum.
    assert(timelineEntries.length === 10 && timelineEntries.every((entry, index) => entry.width >= (entry.credential ? 100 : 160) && entry.fontSize >= 12
      && entry.height >= entry.titleHeight - 1
      && timelineEntries.slice(0, index).every(previous => entry.top >= previous.bottom - 1
        || entry.bottom <= previous.top + 1 || entry.left >= previous.right - 1 || entry.right <= previous.left + 1)),
    'All ten resume milestones stay readable without overlap, including side-by-side credential groups.');
    const timelineScroll = await checkSharedScroll(page, homeStage, 'Mounted resume', `${settings.name}-timeline-bottom.png`, { requireScroll: false });
    await assertLayout(page);

    stage = 'personal-story-links';
    for (const { id, href } of expectedStories) {
      await page.locator(`[data-home-about-connection="${id}"] .home-about__project`).click();
      await page.waitForURL(url => url.pathname === href);
      await settle(page);
      await resetScroll(page);
      await assertSharedStage(page, homeStage, `${id} story destination`);
      await page.goBack();
      await page.waitForURL(url => url.pathname === '/');
      await settle(page);
      await resetScroll(page);
      assert.equal(await page.locator('.home-about .home-timeline [data-home-timeline-item]:visible').count(), 10,
        'Back to About retains all ten mounted resume milestones.');
      assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
        `${id} story navigation preserves the frame and document.`);
    }

    stage = 'all-five-categories';
    const categoryLayouts = {};
    for (const category of ['projects', 'tools', 'games', 'contact', 'about']) {
      await page.locator(`[data-site-tab="${category}"]`).focus();
      await page.keyboard.press('Enter');
      await settle(page);
      await page.waitForFunction(expected => SiteFrame.root()?.dataset.frameCategory === expected, category);
      assert.equal(new URL(page.url()).hash, `#${category}`, 'Every category keeps its canonical overview URL.');
      assert(await page.locator(`[data-home-accordion-item="${category}"]`).isVisible(),
        `${category} content appears after keyboard activation.`);
      categoryLayouts[category] = await assertLayout(page);
      assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
        `${category} activation preserves the frame and document.`);
    }

    stage = 'project-library-and-detail';
    await page.locator('[data-site-tab="projects"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#projects');
    await settle(page);
    await resetScroll(page);
    const projectOverview = await assertSharedStage(page, homeStage, 'Projects overview');
    await page.locator('[data-home-library-open="projects"]').click();
    await page.waitForURL(url => url.pathname === '/portfolio');
    await settle(page);
    assert.equal(await page.evaluate(() => SiteFrame.root().dataset.frameHome), 'true',
      'Opening the project library uses the existing homepage frame.');
    assert.deepEqual(await page.locator('#main .home-library__group:visible > h3').allTextContents(),
      ['Start here', 'Machine learning', 'Data stories', 'Practical applications'],
      'The project library keeps its intentional groups.');
    const projectLibrary = await checkSharedScroll(page, homeStage, 'Inline project library', `${settings.name}-project-library-bottom.png`);
    await page.locator('#main a[href="/portfolio/babynames"]:visible').first().click();
    await page.waitForURL(url => url.pathname === '/portfolio/babynames');
    await settle(page);
    assert.equal(await page.locator('#main h1').innerText(), 'Baby Name Predictor');
    assert(await page.evaluate(() => Boolean(document.activeElement?.closest('#main, [data-site-route-content]'))),
      'Opening a project moves keyboard focus into its content.');
    const projectDetail = await checkSharedScroll(page, homeStage, 'Baby Names project detail', `${settings.name}-project-detail-bottom.png`);
    await page.goBack();
    await page.waitForURL(url => url.pathname === '/portfolio');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Project library restored by Back');
    await page.goBack();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#projects');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Projects overview restored by Back');
    await page.goForward();
    await page.waitForURL(url => url.pathname === '/portfolio');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Project library restored by Forward');
    await page.goForward();
    await page.waitForURL(url => url.pathname === '/portfolio/babynames');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Project detail restored by Forward');
    assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
      'Project routes and their history preserve the shared frame and document.');
    await page.locator('[data-site-tab="projects"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#projects');
    await settle(page);

    stage = 'games-library-and-detail';
    await page.locator('[data-site-tab="games"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#games');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Games overview');
    await page.locator('[data-home-library-open="games"]').click();
    await page.waitForURL(url => url.pathname === '/games');
    await settle(page);
    const gamesLibrary = await checkSharedScroll(page, homeStage, 'Inline games library', `${settings.name}-games-library-bottom.png`, { requireScroll: false });
    await page.locator('#main a[href="/games/roulette"]:visible').click();
    await page.waitForURL(url => url.pathname === '/games/roulette');
    await settle(page);
    await page.locator('#roulette-number-grid button').first().click();
    assert.equal(await page.locator('#roulette-total-bet').innerText(), '$1',
      'The game initializes and responds to a local chip placement.');
    await page.locator('#roulette-clear').click();
    assert.equal(await page.locator('#roulette-total-bet').innerText(), '$0', 'Clear bets updates the game output.');
    const gameRules = page.locator('.roulette00-rules-disclosure');
    await gameRules.locator('summary').click();
    assert(await gameRules.evaluate(node => node.open), 'Game rules expand on demand.');
    const gameDetail = await checkSharedScroll(page, homeStage, 'Expanded game rules', `${settings.name}-game-rules-bottom.png`);
    await page.goBack();
    await page.waitForURL(url => url.pathname === '/games');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Games library restored by Back');
    await page.goForward();
    await page.waitForURL(url => url.pathname === '/games/roulette');
    await settle(page);
    await resetScroll(page);
    await assertSharedStage(page, homeStage, 'Game detail restored by Forward');
    assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
      'Game routes and history preserve the shared frame and document.');
    await page.locator('[data-site-tab="games"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#games');
    await settle(page);

    stage = 'keyboard-navigation';
    await page.locator('[data-site-tab="tools"]').focus();
    await page.keyboard.press('Enter');
    await settle(page);
    await page.waitForFunction(() => SiteFrame.root()?.dataset.frameCategory === 'tools');
    assert(await page.locator('[data-site-tab="tools"]').evaluate(node => node.classList.contains('is-active')),
      'Keyboard activation selects Tools.');
    assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
      'Category navigation preserves the frame and document.');

    stage = 'library-navigation';
    await page.locator('[data-home-library-open="tools"]').click();
    await page.waitForURL(url => url.pathname === '/tools');
    await settle(page);
    await assertSharedStage(page, homeStage, 'Inline tools library');
    assert.deepEqual(await page.locator('#main .home-library__group:visible > h3').allTextContents(),
      ['Start here', 'Text', 'Images', 'Links'], 'The grouped public tool library remains intact.');
    assert.deepEqual(await page.locator('#main .home-library__group:visible').first().locator('a.home-library__card')
      .evaluateAll(nodes => nodes.map(node => node.getAttribute('href'))),
    ['/tools/text-compare', '/tools/qr-code-generator', '/tools/screen-recorder'],
    'The Start here group retains the three approved featured tools, including Screen Recorder.');
    await resetScroll(page);
    const libraryScroll = await wheelToBottom(page, 'Tool library', homeLayout.compact ? 'document' : 'frame');
    await assertSharedStage(page, homeStage, 'Inline tools library after scrolling');
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-library-bottom.png`), fullPage: false });
    await resetScroll(page);

    stage = 'contact-navigation';
    await page.locator('[data-site-tab="tools"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#tools');
    await settle(page);
    await assertLayout(page);
    await page.locator('[data-site-tab="contact"]').click();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#contact');
    await settle(page);
    assert(await page.locator('a[href^="mailto:"]').first().isVisible(), 'Contact exposes a working email link.');
    await assertLayout(page);
    await page.goBack();
    await settle(page);
    await page.waitForFunction(() => SiteFrame.root()?.dataset.frameCategory === 'tools');
    assert.equal(new URL(page.url()).hash, '#tools', 'Back restores the Tools overview.');
    await page.goBack();
    await page.waitForURL(url => url.pathname === '/tools');
    await settle(page);
    await assertLayout(page);
    assert.equal(await page.evaluate(() => SiteFrame.root()?.dataset.frameView), 'library',
      'Back restores the active-only library navigation.');
    await page.goForward();
    await page.waitForURL(url => url.pathname === '/' && url.hash === '#tools');
    await settle(page);
    await page.goForward();
    await settle(page);
    assert.equal(new URL(page.url()).hash, '#contact');
    assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
      'Back and Forward preserve the shared frame and page.');

    stage = 'text-compare';
    await page.locator('[data-site-tab="tools"]').click();
    await settle(page);
    await page.locator('[data-home-library-open="tools"]').click();
    await page.waitForURL(url => url.pathname === '/tools');
    await settle(page);
    await page.locator('#main a[href="/tools/text-compare"]:visible').first().click();
    await page.waitForURL(url => url.pathname === '/tools/text-compare');
    await settle(page);
    assert(await page.evaluate(() => Boolean(document.activeElement?.closest('#main, [data-site-route-content]'))),
      'Route completion moves keyboard focus into the destination.');
    assert(await page.evaluate(() => smokeFrame === SiteFrame.root() && smokeTimeOrigin === performance.timeOrigin),
      'Opening a utility preserves the frame and document.');
    await page.locator('#textcompare-original').fill('The quick brown fox.');
    await page.locator('#textcompare-revised').fill('The quick blue fox.');
    await page.locator('#textcompare-form button[type="submit"]').click();
    await page.waitForFunction(() => document.querySelector('#textcompare-output')?.textContent.includes('blue'));
    const comparison = page.locator('#textcompare-output');
    assert(await comparison.isVisible(), 'The generated comparison is visible.');
    assert((await comparison.innerText()).includes('brown'), 'The comparison preserves the removed word.');
    assert((await comparison.innerText()).includes('blue'), 'The comparison includes the added word.');
    await assertSharedStage(page, homeStage, 'Text Compare output');
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-comparison.png`), fullPage: false });

    const directRoutes = {};
    for (const { route, heading, requireScroll = false, audience = 'personal', navigation = 'soft', demo = false } of [
      { route: '/portfolio', heading: /Project library/i, requireScroll: true },
      { route: '/portfolio/babynames', heading: /Baby Name Predictor/, requireScroll: true },
      { route: '/tools', heading: /Tool library/, requireScroll: true },
      { route: '/games', heading: /Game library/ },
      { route: '/tools/text-compare', heading: /Text Compare/ },
      { route: '/games/roulette', heading: /Double-Zero Roulette/, requireScroll: true },
      { route: '/games/stellar-dogfight', heading: /Stellar Dogfight/ },
      { route: '/contact', heading: /Let's Connect/ },
      { route: '/privacy', heading: /Privacy.*Your Data/, requireScroll: true },
      { route: '/search?q=data', heading: /^Search$/, requireScroll: true },
      { route: '/404', heading: /404.*Page Not Found/ },
      { route: '/tools/background-remover', heading: /Background Remover/, navigation: 'hard' },
      { route: '/tools/transcribe', heading: /File Transcriber/, navigation: 'hard' },
      { route: '/baby-names-demo', demo: true },
      { route: '/analytics', heading: /Data Analyst/, audience: 'analytics', requireScroll: true },
      { route: '/portfolio?audience=analytics', heading: /Analytics Portfolio/, audience: 'analytics' },
      { route: '/portfolio/babynames?audience=analytics', heading: /Baby Name Predictor/, audience: 'analytics', requireScroll: true },
      { route: '/contact?audience=analytics', heading: /Let's Talk Analytics Roles/, audience: 'analytics' },
      { route: '/resume-analytics', heading: /Daniel Short/, audience: 'analytics', requireScroll: true }
    ]) {
      stage = `direct-route:${route}`;
      const directResponse = await page.goto(base + route, { waitUntil: 'domcontentloaded' });
      assert.equal(directResponse.status(), 200, `${route} loads directly.`);
      await settle(page);
      await page.evaluate(() => document.fonts.ready);
      assert.equal(await page.evaluate(() => SiteFrame.root().dataset.frameAudience), audience, `${route} uses its intended audience.`);
      assert.equal(await page.evaluate(() => document.body.dataset.siteRouteNavigation), navigation, `${route} retains its route lifecycle.`);
      if (heading) assert.match(await page.locator('.site-frame h1:visible').innerText(), heading);
      if (route.startsWith('/search')) {
        await page.locator('#search-results a').first().waitFor();
        assert((await page.locator('#search-results a').count()) > 1, 'Search renders real matching results before geometry is checked.');
      }
      const artifactName = route.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '');
      directRoutes[route] = demo ? await checkDemo(page, homeStage, settings)
        : await checkSharedScroll(page, homeStage, `Direct ${route}`, `${settings.name}-direct-${artifactName}-bottom.png`, { requireScroll });
      if (route === '/games/stellar-dogfight') {
        await page.locator('[data-overlay-action="launch"]').click();
        await page.waitForFunction(() => document.body.classList.contains('is-playing'));
        await settle(page);
        await resetScroll(page);
        await assertLayout(page);
        const playing = await scrollState(page);
        const canvas = await page.locator('[data-role="battlefield"]').boundingBox();
        assert(canvas && canvas.width >= 280 && canvas.height >= 200, 'Launching the campaign produces a usable combat canvas.');
        if (!playing.compact) {
          for (const property of ['left', 'top', 'width']) {
            assert(Math.abs(playing.stageBox[property] - homeStage.stageBox[property]) <= 2,
              `Active play preserves the shared frame ${property}.`);
          }
          assert(playing.stageBox.height >= homeStage.stageBox.height - 2
            && playing.stageBox.height <= homeStage.stageBox.height + homeStage.footerHeight + 2,
          'Active play can reclaim the hidden footer space without enlarging the frame beyond the screen.');
          const bottomInset = homeStage.height - homeStage.stageBox.top - homeStage.stageBox.height - homeStage.footerHeight;
          assert(playing.stageBox.top + playing.stageBox.height <= playing.height - bottomInset + 2
            && playing.documentRange <= 1, 'Active play preserves the outer inset and has no document overflow.');
          assert(canvas.x >= playing.frameBox.left - 1 && canvas.x + canvas.width <= playing.frameBox.right + 1
            && canvas.y >= playing.frameBox.top - 1 && canvas.y + canvas.height <= playing.frameBox.bottom + 1,
          'The desktop combat canvas stays wholly inside the shared content viewport.');
          assert(canvas.y + canvas.height <= playing.height, 'The desktop combat canvas is not cut off below the screen.');
        }
        directRoutes[route].playing = { geometry: playing, canvas };
        await page.screenshot({ path: path.join(artifactDir, `${settings.name}-stellar-playing.png`), fullPage: false });
      }
    }
    assert.deepEqual(errors, [], 'No runtime, console, or same-origin request errors.');
    cases.push({ name: settings.name, pass: true, viewport: settings.viewport, reducedMotion: settings.reducedMotion,
      homeLayout, aboutColumns, categoryLayouts, timelineScroll, libraryScroll,
      homeStage, projectOverview, projectLibrary, projectDetail, gamesLibrary, gameDetail, directRoutes, unavailableServices });
    console.log(`Browser smoke passed: ${settings.name}`);
  } catch (error) {
    cases.push({ name: settings.name, pass: false, stage, url: page.url(), error: error.message, errors, unavailableServices });
    await page.screenshot({ path: path.join(artifactDir, `${settings.name}-failure.png`), fullPage: false }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync(path.join(artifactDir, 'results.json'), JSON.stringify({ cases }, null, 2));
    await context.close();
  }
}

async function main() {
  assert(fs.existsSync(path.join(root, 'public/index.html')), 'Run npm run build before the browser gate.');
  // An empty env directory keeps the smoke server independent of personal AWS
  // credentials. This gate never calls signed-in or mutating backend actions.
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'site-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    const scripts = JSON.parse(fs.readFileSync(path.join(root, 'public/dist/scripts-manifest.json'), 'utf8'));
    for (const [asset, expected] of [
      [`/dist/${scripts.shell}`, 'public, max-age=31536000, immutable'],
      ['/dist/site-shell.js', 'public, max-age=0, must-revalidate']
    ]) {
      const response = await fetch(base + asset, { method: 'HEAD', signal: AbortSignal.timeout(10000) });
      assert.equal(response.status, 200, asset);
      assert.equal(response.headers.get('cache-control'), expected, asset);
    }
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    for (const settings of [
      { name: 'desktop', viewport: { width: 1440, height: 900 }, reducedMotion: 'no-preference' },
      { name: 'mobile', viewport: { width: 390, height: 844 }, reducedMotion: 'reduce' }
    ]) await runViewport(browser, base, settings);
    await runResponsiveSpacingChecks({ browser, base, settle, assertLayout, artifactDir });
    await runClosedHomeChecks({ browser, base, artifactDir });
    console.log(`Browser artifacts: ${artifactDir}`);
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
