'use strict';
const { test, expect, ready, settle, compare } = require('./fixtures.cjs');
test.skip(process.platform !== 'linux', 'Reviewed baselines use the pinned Linux Chromium build; run in WSL or CI.');
for (const width of [1440, 390]) {
  for (const state of ['closed', 'about', 'tools', 'website', 'digit', 'comparison', 'contact']) {
    test(`${state}-${width}`, async ({ page }) => {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 900 });
      const routes = { closed: '/#closed', about: '/#about', tools: '/#tools', website: '/portfolio/website', digit: '/portfolio/digitGenerator', comparison: '/tools/text-compare', contact: '/contact' };
      await ready(page, routes[state]);
      if (state === 'comparison') await compare(page);
      if (state === 'contact') {
        await expect(page.locator('[data-contact-inline] #contact-form')).toBeVisible();
      }
      if (state === 'digit') {
        if (width === 390) {
          await page.locator('.project-intro-action--demo').click();
          await settle(page);
        }
        const frame = page.frameLocator('iframe.project-embed-frame, iframe.project-demo-wrapper-iframe');
        await expect(frame.locator('.digit-cell')).toHaveCount(36);
        await frame.locator('body').evaluate(() => document.fonts.ready);
      }
      await settle(page);
      if (state === 'comparison') {
        await page.locator('#textcompare-form').evaluate((form) => {
          for (let node = form; node; node = node.parentElement) node.scrollTop = 0;
          window.scrollTo(0, 0);
        });
        await settle(page);
      }
      await page.locator('img:visible').evaluateAll((images) => Promise.all(images.filter((image) => {
        const box = image.getBoundingClientRect();
        return box.bottom > 0 && box.top < innerHeight;
      }).map((image) => image.decode?.().catch(() => {}))));
      // Dismissing consent can leave the pointer above a newly revealed link.
      // Capture the resting design, independently of that incidental hover.
      await page.mouse.move(0, 0);
      await expect(page).toHaveScreenshot(`${state}-${width}.png`, { fullPage: state === 'comparison' && width === 390 });
    });
  }
}
