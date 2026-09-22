'use strict';
const { test, expect, ready, settle } = require('./fixtures.cjs');

async function expectUndecoratedCards(cards, label) {
  const decorated = await cards.evaluateAll((links) => links.flatMap((link) => {
    // An underline on the anchor propagates through its inline descendants, so
    // checking only the title/description's own computed style misses the bug.
    const nodes = [link, ...link.querySelectorAll('.home-library__copy, .home-library__copy *')];
    return nodes.filter((node) => getComputedStyle(node).textDecorationLine !== 'none')
      .map((node) => ({ href: link.getAttribute('href'), element: node.tagName,
        className: node.className, decoration: getComputedStyle(node).textDecorationLine }));
  }));
  expect(decorated, `${label}: card text must not inherit prose-link underlines`).toEqual([]);
}

for (const width of [1440, 390, 320]) {
  test(`library cards retain clean text and keyboard navigation at ${width}px`, async ({ page }, info) => {
    await page.setViewportSize({ width, height: 900 });
    for (const route of ['/portfolio', '/tools', '/games']) {
      await ready(page, route);
      expect(new URL(page.url()).pathname).toBe(route);
      const library = page.locator('main .home-library:visible');
      await expect(library.locator('h1')).toBeVisible();
      const cards = library.locator('a.home-library__card:visible');
      const first = cards.first();
      await expect(first).toBeVisible();
      expect(await cards.count()).toBeGreaterThan(0);
      await page.mouse.move(0, 0);
      await expectUndecoratedCards(cards, `${route} default`);
      await first.hover();
      await expectUndecoratedCards(cards, `${route} hover`);
      await page.mouse.move(0, 0);
      // Establish keyboard modality before programmatically selecting the card.
      await page.keyboard.press('Tab');
      await first.focus();
      await expect(first).toBeFocused();
      expect(await first.evaluate((node) => node.matches(':focus-visible'))).toBe(true);
      await expectUndecoratedCards(cards, `${route} keyboard focus`);
      const outline = await first.evaluate((node) => {
        const style = getComputedStyle(node);
        return { width: parseFloat(style.outlineWidth), style: style.outlineStyle };
      });
      expect(outline.width, `${route}: preserve the visible focus indicator`).toBeGreaterThanOrEqual(2);
      expect(outline.style).not.toBe('none');
      await info.attach(`library-${route.slice(1)}-${width}-keyboard-focus`, {
        body: await page.screenshot({ fullPage: false }), contentType: 'image/png'
      });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${route}: no horizontal overflow`).toBeLessThanOrEqual(1);

      // Probes use the real page styles; the fix must not remove the normal
      // underline affordance from prose links outside the card component.
      await page.evaluate(() => {
        const prose = document.createElement('section');
        prose.id = 'library-prose-link-probes';
        prose.innerHTML = '<p><a data-prose-link="paragraph" href="#paragraph">Paragraph link</a></p>' +
          '<ul><li><a data-prose-link="list" href="#list">List link</a></li></ul>';
        document.body.append(prose);
      });
      for (const kind of ['paragraph', 'list']) {
        await expect(page.locator(`#library-prose-link-probes [data-prose-link="${kind}"]`))
          .toHaveCSS('text-decoration-line', 'underline');
      }
      await page.locator('#library-prose-link-probes').evaluate((node) => node.remove());

      if (route === '/portfolio') {
        const destination = new URL(await first.getAttribute('href'), page.url());
        expect(destination.origin).toBe(new URL(page.url()).origin);
        await first.focus();
        await page.keyboard.press('Enter');
        await expect.poll(() => new URL(page.url()).pathname).toBe(destination.pathname);
        await settle(page);
        await expect(page.locator('main')).toBeVisible();
      }
    }
  });
}
