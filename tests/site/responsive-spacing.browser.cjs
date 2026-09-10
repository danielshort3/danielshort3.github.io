'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function measure(page) {
  return page.evaluate(() => {
    const box = (selector) => {
      const node = document.querySelector(selector);
      if (!node) return null;
      const rect = node.getBoundingClientRect();
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height, right: rect.right, bottom: rect.bottom };
    };
    return {
      viewport: { width: innerWidth, height: innerHeight },
      document: { width: document.documentElement.scrollWidth, height: document.documentElement.scrollHeight },
      header: box('[data-site-shell-header] .nav'),
      headerInner: box('[data-site-shell-header] .nav .wrapper'),
      stage: box('[data-site-frame-stage]'),
      contentWidth: parseFloat(getComputedStyle(SiteFrame.viewport()).width),
      libraryLists: [...document.querySelectorAll('.home-library__list')].filter((list) => list.getBoundingClientRect().width > 0).map((list) => {
        const cards = [...list.querySelectorAll('.home-library__card')].map((card) => card.getBoundingClientRect()).filter((rect) => rect.width > 0);
        return {
          columns: getComputedStyle(list).gridTemplateColumns.split(' ').length,
          cards: cards.map((rect) => ({ x: rect.x, y: rect.y, width: rect.width, right: rect.right }))
        };
      }),
      footer: box('[data-site-shell-footer]'),
      footerInner: box('[data-site-shell-footer] .footer-inner'),
      footerVisible: (() => {
        const footer = document.querySelector('[data-site-shell-footer]');
        if (!footer) return false;
        const style = getComputedStyle(footer);
        return style.display !== 'none' && style.visibility === 'visible' && Number(style.opacity) > 0;
      })(),
      about: box('.home-about'),
      profile: box('.home-about__profile'),
      story: box('.home-about__story'),
      timeline: box('.home-about .home-timeline'),
      timelineMounted: Boolean(document.querySelector('.home-about .home-timeline > h3')) &&
        !document.querySelector('.home-about .home-timeline details, .home-about .home-timeline summary') &&
        document.querySelectorAll('.home-about .home-background__entry, .home-about .home-background__credential-link').length === 10 &&
        [...document.querySelectorAll('.home-about .home-background__entry, .home-about .home-background__credential-link')].every((entry) => {
          const rect = entry.getBoundingClientRect();
          return rect.width > 0 && rect.height > 0 && getComputedStyle(entry).visibility === 'visible';
        }),
      timelineItems: document.querySelectorAll('.home-about [data-home-timeline-item]').length
    };
  });
}

function assertDesktopFrame(metrics, label) {
  const { viewport, document, header, stage, footer, footerVisible } = metrics;
  assert.equal(stage.width, Math.min(1500, viewport.width - 28), `${label} caps the shared frame at a readable desktop width.`);
  assert(Math.abs(stage.x - (viewport.width - stage.width) / 2) <= 1, `${label} centers the frame horizontally.`);
  for (const region of [metrics.headerInner, metrics.footerInner]) {
    assert(region && Math.abs(region.x - stage.x) <= 1 && Math.abs(region.width - stage.width) <= 1,
      `${label} aligns the masthead, frame, and footer to one desktop width.`);
  }
  assert(header && Math.abs(stage.y - header.bottom - 14) <= 1, `${label} retains its 14px gap below navigation.`);
  assert(footerVisible && footer && footer.height >= 40 && Math.abs(footer.bottom - viewport.height) <= 1,
    `${label} keeps the footer visible at the bottom of the screen.`);
  assert(Math.abs(footer.y - stage.bottom - 14) <= 1, `${label} fills the height down to its 14px footer gap.`);
  assert(document.width <= viewport.width + 1 && document.height <= viewport.height + 1,
    `${label} keeps scrolling inside the desktop frame.`);
}

function assertLibraryColumns(metrics, route) {
  const threshold = route === '/tools' ? 1180 : 1500;
  const expected = metrics.contentWidth >= threshold ? 3 : metrics.contentWidth >= 760 ? 2 : 1;
  assert(metrics.libraryLists.length > 0, `${route} displays its library at ${metrics.viewport.width}px.`);
  for (const list of metrics.libraryLists) {
    assert.equal(list.columns, expected, `${route} uses ${expected} columns for ${metrics.contentWidth}px of content space.`);
    assert(list.cards.every((card) => card.x >= -1 && card.right <= metrics.viewport.width + 1),
      `${route} keeps every card within the screen.`);
    const firstRow = list.cards.filter((card) => Math.abs(card.y - list.cards[0].y) <= 1);
    assert.equal(firstRow.length, Math.min(expected, list.cards.length), `${route} lays out the expected number of cards side by side.`);
  }
}

async function runResponsiveSpacingChecks({ browser, base, settle, assertLayout, artifactDir }) {
  const context = await browser.newContext({ viewport: { width: 2560, height: 1440 }, reducedMotion: 'reduce' });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const errors = [];
  const results = { wideRoutes: [], librarySizes: [], aboutSizes: [], errors };
  let activeCase = 'wide-home';
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('response', (response) => {
    if (response.url().startsWith(base) && response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });

  const open = async (route) => {
    const response = await page.goto(base + route, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${route} loads normally.`);
    await settle(page);
    await page.evaluate(() => document.fonts.ready);
    if (await page.locator('#pcz-reject').isVisible()) {
      await page.locator('#pcz-reject').click();
      await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
    }
    await assertLayout(page);
    await page.evaluate(() => {
      scrollTo({ top: 0, behavior: 'instant' });
      SiteFrame.viewport().scrollTo({ top: 0, behavior: 'instant' });
    });
  };

  try {
    let baseline;
    for (const route of ['/', '/portfolio', '/tools', '/games', '/contact', '/portfolio/babynames', '/tools/text-compare', '/privacy', '/games/stellar-dogfight']) {
      activeCase = `wide-${route.replace(/^\//, '').replaceAll('/', '-') || 'home'}`;
      await open(route);
      let embeddedDemo;
      if (route === '/portfolio/babynames') {
        const demo = page.frameLocator('.project-demo-shell iframe.project-embed-frame');
        await demo.locator('#status-pill[data-state="ok"]').waitFor({ state: 'visible', timeout: 12000 });
        const ratingEntries = Number((await demo.locator('#stat-rated').innerText()).replaceAll(',', ''));
        assert(ratingEntries > 0, 'The embedded Baby Names demo is populated before its project screenshot.');
        embeddedDemo = { status: 'ready', ratingEntries };
      }
      const metrics = await measure(page);
      baseline ||= metrics.stage;
      assertDesktopFrame(metrics, route);
      assert(metrics.stage.height > 960, `${route} uses the extra height available on a tall desktop screen.`);
      for (const key of ['x', 'y', 'width', 'height']) {
        assert(Math.abs(metrics.stage[key] - baseline[key]) <= 1, `${route} shares the home frame's ${key}.`);
      }
      if (['/portfolio', '/tools', '/games'].includes(route)) assertLibraryColumns(metrics, route);
      if (route === '/games/stellar-dogfight') {
        await page.getByRole('button', { name: 'Open command menu', exact: true }).click();
        await page.waitForFunction(() => document.getElementById('command-menu')?.getAttribute('aria-hidden') === 'false');
        await page.locator('#command-menu button[data-action="command-menu-close"]').click();
        await page.waitForFunction(() => document.getElementById('command-menu')?.getAttribute('aria-hidden') === 'true');
      }
      assert.deepEqual(errors, [], `${route} produces no new browser or local request errors.`);
      results.wideRoutes.push({ route, ...metrics, ...(embeddedDemo ? { embeddedDemo } : {}) });
      await page.screenshot({ path: path.join(artifactDir, `spacing-${activeCase}.png`), fullPage: false });
    }

    for (const viewport of [{ width: 1920, height: 1080 }, { width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      for (const route of ['/portfolio', '/tools', '/games']) {
        activeCase = `library-${route.slice(1)}-${viewport.width}`;
        await open(route);
        const metrics = await measure(page);
        assertLibraryColumns(metrics, route);
        if (viewport.width >= 960) assertDesktopFrame(metrics, activeCase);
        assert(metrics.document.width <= viewport.width + 1, `${activeCase} avoids horizontal document scrolling.`);
        results.librarySizes.push({ route, ...metrics });
      }
    }

    for (const viewport of [{ width: 1200, height: 900 }, { width: 1199, height: 900 }, { width: 768, height: 1024 }, { width: 320, height: 740 }]) {
      activeCase = `about-${viewport.width}`;
      await page.setViewportSize(viewport);
      await open('/');
      await page.locator('.home-about__portrait').evaluate((image) => image.decode());
      const metrics = await measure(page);
      assert(metrics.about && metrics.profile && metrics.story && metrics.timeline,
        `About has its complete profile, personal connections, and timeline at ${viewport.width}px.`);
      assert(metrics.document.width <= viewport.width + 1, `About fits ${viewport.width}px without horizontal scrolling.`);
      assert(metrics.timelineMounted && metrics.timelineItems === 10, 'The complete journey stays mounted without a dropdown.');
      for (const [name, rect] of [['profile', metrics.profile], ['story', metrics.story], ['timeline', metrics.timeline]]) {
        assert(rect.x >= -1 && rect.right <= viewport.width + 1, `${name} stays within ${viewport.width}px.`);
      }
      if (viewport.width === 768) {
        assert(metrics.profile.bottom <= Math.min(metrics.story.y, metrics.timeline.y) + 1,
          'The medium layout places the profile above both content columns.');
        assert(metrics.story.right <= metrics.timeline.x - 12 && metrics.story.width >= 240 && metrics.timeline.width >= 240,
          'The medium layout retains readable, separate story and journey columns.');
      }
      if (viewport.width === 320) {
        assert(metrics.profile.bottom <= metrics.story.y + 1 && metrics.story.bottom <= metrics.timeline.y + 1,
          'The narrow layout stacks profile, connections, and journey in reading order.');
        assert.equal(await page.locator('.home-about__connection').count(), 3, 'All three personal connections remain present.');
        assert.equal(await page.locator('.home-about [data-home-timeline-item]:visible').count(), 10, 'All ten resume entries remain available.');
      }
      results.aboutSizes.push(metrics);
      await page.screenshot({ path: path.join(artifactDir, `spacing-${activeCase}.png`), fullPage: viewport.width <= 768 });
    }
    const [above, below] = results.aboutSizes;
    assert(Math.abs(above.about.height - below.about.height) <= 160,
      `Crossing 1200px does not produce a large About height jump (${above.about.height} vs ${below.about.height}).`);
    assert.deepEqual(errors, [], 'Responsive layout checks produce no new browser or local request errors.');
    results.pass = true;
    console.log('Responsive spacing passed: 9 wide routes, 9 responsive libraries, and 4 About breakpoints');
  } catch (error) {
    results.pass = false;
    results.failure = { case: activeCase, url: page.url(), message: error.message };
    await page.screenshot({ path: path.join(artifactDir, 'spacing-failure.png'), fullPage: false }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync(path.join(artifactDir, 'responsive-spacing.json'), JSON.stringify(results, null, 2));
    await context.close();
  }
}

module.exports = runResponsiveSpacingChecks;
