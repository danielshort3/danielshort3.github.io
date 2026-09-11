'use strict';

const assert = require('node:assert/strict');

module.exports = async function runCatalogIconChecks({ browser, base }) {
  for (const width of [390, 1440]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
    const page = await context.newPage();
    try {
      for (const route of ['/#projects', '/tools', '/games']) {
        const requests = [];
        const record = (request) => requests.push(request.url());
        page.on('request', record);
        await page.goto(base + route, { waitUntil: 'load' });
        await page.waitForFunction(() => [...document.querySelectorAll('.catalog-icon img')]
          .some((image) => image.getBoundingClientRect().width > 0 && image.complete && image.naturalWidth > 0));
        const samples = await page.evaluate(() => [...document.querySelectorAll('.catalog-icon img')]
          .filter((image) => image.getBoundingClientRect().width > 0 && image.complete && image.naturalWidth > 0)
          .slice(0, 3).map((image) => {
            const picture = image.parentElement;
            const holder = picture.parentElement;
            const original = image.getAttribute('src');
            const current = image.currentSrc;
            const before = image.getBoundingClientRect();
            const sourceDisplay = getComputedStyle(picture.querySelector('source')).display;
            // Compare actual geometry with the pre-optimization DOM structure.
            // Keep the loaded WebP URL during the comparison to avoid a PNG fetch.
            image.src = current;
            holder.insertBefore(image, picture);
            const native = image.getBoundingClientRect();
            picture.append(image);
            image.setAttribute('src', original);
            return { original, current, sourceDisplay,
              before: { width: before.width, height: before.height },
              native: { width: native.width, height: native.height } };
          }));
        assert(samples.length > 0, `${route} ${width}: visible artwork is available`);
        for (const sample of samples) {
          assert(/\.png\?v=/.test(sample.original), `${route}: preserves the fingerprinted PNG fallback`);
          assert(/\.webp\?v=/.test(sample.current), `${route}: the browser selects WebP`);
          assert.equal(sample.sourceDisplay, 'none', 'source elements must not introduce an extra grid track');
          assert(Math.abs(sample.before.width - sample.native.width) < 1, `${route} ${width}: width matches the original icon`);
          assert(Math.abs(sample.before.height - sample.native.height) < 1, `${route} ${width}: height matches the original icon`);
        }
        assert(!requests.some((url) => /\/img\/(?:projects|tools|games)\/icons\/[^?]+\.png(?:\?|$)/.test(url)),
          `${route} ${width}: optimized browsers do not also download PNG catalog fallbacks`);
        page.off('request', record);
      }
    } finally {
      await context.close();
    }
  }
  console.log('Catalog browser checks: WebP selection, no duplicate PNG downloads, and original desktop/mobile geometry passed.');
};
