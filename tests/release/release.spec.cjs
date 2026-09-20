'use strict';
const AxeBuilder = require('@axe-core/playwright').default;
const { test, expect, ready, settle, compare, mockAccount } = require('./fixtures.cjs');
const runContactRecoveryChecks = require('../site/contact-recovery.browser.cjs');
async function audit(page, testInfo, name) {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  await testInfo.attach(`axe-${name}`, { body: JSON.stringify(results, null, 2), contentType: 'application/json' });
  expect(results.violations, `${name}: ${results.violations.map((v) => v.id + ': ' + v.nodes.map((node) => node.target.join(' ')).join(', ')).join('\n')}`).toEqual([]);
}
async function reflow(page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
}
for (const width of [1440, 320]) {
  test(`homepage panels and libraries: WCAG A/AA and ${width}px reflow`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    await ready(page, '/#closed');
    await audit(page, info, 'closed-home');
    for (const category of ['about', 'projects', 'tools', 'games', 'contact']) {
      const rail = page.locator(`[data-site-tab="${category}"]`);
      if (await rail.isVisible()) await rail.click();
      else await page.locator(`[data-mobile-section="${category}"]`).click();
      await settle(page);
      await reflow(page);
      await audit(page, info, category);
    }
    for (const route of ['/portfolio', '/tools', '/games']) {
      await ready(page, route);
      await reflow(page);
      await audit(page, info, route.slice(1));
    }
  });
}
test('project, result, expanded search and dialogs: accessibility and keyboard focus', async ({ page }, info) => {
  await ready(page, '/portfolio/website');
  await audit(page, info, 'project');
  const question = page.locator('.project-question-link');
  await question.focus();
  await page.keyboard.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Send a Message' });
  await expect(dialog).toBeVisible();
  await page.locator('#contact-form [type="submit"]').click();
  await expect(page.locator('#contact-name')).toBeFocused();
  await page.locator('#contact-modal .modal-close').focus();
  await page.keyboard.press('Shift+Tab');
  await expect(page.locator('#contact-form [type="submit"]')).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(page.locator('#contact-modal .modal-close')).toBeFocused();
  await audit(page, info, 'contact-validation');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(question).toBeFocused();
  await ready(page, '/tools/text-compare');
  await compare(page);
  await audit(page, info, 'tool-result');
  const search = page.locator('.nav-search-button:visible');
  await search.focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#nav-search-q')).toBeFocused();
  await audit(page, info, 'expanded-search');
  await page.locator('#nav-search-q').fill('digit');
  await page.locator('#nav-search-q').press('Enter');
  await expect(page.locator('#search-results a').first()).toBeVisible();
  await audit(page, info, 'search-results');
});
test('account dialog: WCAG A/AA and keyboard focus restoration', async ({ page }, info) => {
  await ready(page, '/tools/text-compare');
  await mockAccount(page);
  await audit(page, info, 'account');
  await page.keyboard.press('Escape');
  await expect(page.locator('[data-tools-action="open-account"]')).toBeFocused();
});
test('keyboard targets, forced colors, enlarged text and reduced motion', async ({ page, browserName }, info) => {
  await ready(page, '/#closed');
  const first = page.locator('[data-site-tab="about"]');
  await page.keyboard.press('Tab');
  await first.focus();
  await page.keyboard.press('Enter');
  await settle(page);
  await expect(first).toHaveAttribute('aria-expanded', 'true');
  const moving = await page.evaluate(() => document.getAnimations().filter((animation) => {
    const timing = animation.effect?.getComputedTiming();
    return animation.playState === 'running' && timing?.iterations !== Infinity && timing?.endTime > 1;
  }).length);
  expect(moving, 'Reduced motion must settle the actual interface, not only the media preference').toBe(0);
  const outline = await first.evaluate((node) => { const style = getComputedStyle(node); return { width: parseFloat(style.outlineWidth), style: style.outlineStyle }; });
  expect(outline.width).toBeGreaterThanOrEqual(2);
  expect(outline.style).not.toBe('none');
  await page.keyboard.press('Enter');
  await settle(page);
  await expect(first).toHaveAttribute('aria-expanded', 'false');
  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true);
  if (browserName !== 'webkit') {
    await page.emulateMedia({ forcedColors: 'active' });
    await first.focus();
    await expect(first).toBeVisible();
    await audit(page, info, 'forced-colors');
  }
  await page.setViewportSize({ width: 320, height: 900 });
  await ready(page, '/tools/text-compare');
  await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
  await reflow(page);
  await expect(page.locator('#textcompare-original')).toBeVisible();
  // A 320 CSS-pixel viewport models 400% zoom from a 1280px-wide desktop.
  await audit(page, info, 'enlarged-text');
});
test('contact request and response-body recovery across modal lifecycles', async ({ browser, baseURL }, info) => {
  await runContactRecoveryChecks({ browser, base: baseURL, artifactDir: info.outputPath('contact-recovery') });
});
