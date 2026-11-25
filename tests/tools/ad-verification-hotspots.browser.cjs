'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

const stops = { downtown: 272, horizon: 224, mesa: 120, colonias: 104, lincoln: 80 };
const media = '/demos/data/ad-verification-hotspots-03c20afc.mp4';

async function seek(page, fraction) {
  const slider = page.getByRole('slider', { name: 'Campaign playback position' });
  const box = await slider.boundingBox();
  await slider.click({ position: { x: 8 + fraction * (box.width - 16), y: box.height / 2 } });
  await page.waitForFunction((value) => {
    const movie = document.getElementById('movie');
    return !movie.seeking && Math.abs(movie.currentTime / movie.duration - value) < .025;
  }, fraction);
}

async function areaDots(dialog) {
  return dialog.locator('#detail-map').evaluate((map) => {
    const circles = [...map.querySelectorAll('.district-marks circle')].filter((c) => getComputedStyle(c).display !== 'none');
    const matrix = map.getScreenCTM();
    const bounds = map.getBoundingClientRect();
    const positions = circles.map((c) => new DOMPoint(c.cx.baseVal.value, c.cy.baseVal.value).matrixTransform(matrix));
    const pixels = new Set(positions.map((p) => Math.round(p.x) + ',' + Math.round(p.y)));
    return {
      count: circles.length,
      distinct: pixels.size,
      inView: positions.every((p) => p.x >= bounds.left && p.x <= bounds.right && p.y >= bounds.top && p.y <= bounds.bottom),
      streets: Boolean(map.querySelector('.district-street, .district-major, .district-named'))
    };
  });
}

async function checkLabels(page, mobile, output) {
  const map = page.locator(mobile ? '#mobile-areas' : '#desktop-areas');
  const dialog = page.getByRole('dialog', { name: 'Visit area' });
  assert.equal(await map.getByRole('button').count(), 5);
  for (const [id, count] of Object.entries(stops)) {
    const label = map.locator('button[data-area="' + id + '"]');
    assert.match(await label.getAttribute('aria-label'), /Zoom in on /i);
    if (mobile) await label.tap();
    else await label.click();
    await dialog.waitFor();
    assert.equal(await dialog.getByRole('combobox').inputValue(), id, 'Touching each map label must open its own area');
    assert.equal(await dialog.locator('#area-count').textContent(), String(count));
    assert.deepEqual(await areaDots(dialog), { count, distinct: count, inView: true, streets: true });
    assert.match(await dialog.locator('#area-source').getAttribute('href'), /^https:\/\//);
    if (id === 'downtown') {
      await dialog.screenshot({ path: path.join(output, 'hotspot-detail-' + (mobile ? 'phone' : 'desktop') + '.png') });
    }
    if (mobile) await dialog.getByRole('button', { name: 'Back to city map' }).tap();
    else await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'hidden' });
    assert.equal(await label.evaluate((button) => button === document.activeElement), true);
  }
  await page.getByRole('button', { name: 'Explore visit areas' }).click();
  await dialog.getByRole('combobox').selectOption('horizon');
  const slider = dialog.getByRole('slider', { name: 'Campaign playback position' });
  await slider.press('Home');
  await page.waitForFunction(() => document.getElementById('area-count').textContent === '0');
  assert.equal((await areaDots(dialog)).count, 0);
  await seek(page, .56);
  await page.waitForFunction(() => {
    const movie = document.getElementById('movie');
    const data = JSON.parse(document.getElementById('campaign-data').textContent);
    const frame = Math.floor(movie.currentTime * data.fps + .0001);
    const shown = data.areas.find((s) => s.id === 'horizon').points.filter((p) => p[2] <= frame).length;
    return shown > 0 && shown < 224 && String(shown) === document.getElementById('area-count').textContent;
  });
  const partial = Number(await dialog.locator('#area-count').textContent());
  assert.equal((await areaDots(dialog)).count, partial);
  await dialog.getByRole('button', { name: 'Play campaign' }).click();
  await page.waitForFunction((previous) => Number(document.getElementById('area-count').textContent) > previous, partial);
  await dialog.getByRole('button', { name: 'Pause campaign' }).click();
  await dialog.getByRole('button', { name: 'Back to city map' }).click();
  await dialog.waitFor({ state: 'hidden' });
}

module.exports = async function runHotspotChecks({ browser, url, output }) {
  for (const mobile of [false, true]) {
    const viewport = mobile ? { width: 390, height: 844 } : { width: 1440, height: 900 };
    const origin = new URL(url).origin;
    const context = await browser.newContext({ viewport, isMobile: mobile, hasTouch: mobile, deviceScaleFactor: 1 });
    const errors = [];
    try {
      await context.route('**/*', (route) => {
        const source = new URL(route.request().url());
        if (source.origin === origin) return route.continue();
        errors.push('External request ' + source.origin);
        return route.abort();
      });
      const page = await context.newPage();
      page.on('pageerror', (error) => errors.push(error.message));
      page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
      page.on('response', (response) => { if (response.status() >= 400) errors.push(response.status() + ' ' + response.url()); });

      const response = await page.goto(url);
      assert.equal(response.status(), 200);
      assert.equal(page.url(), url);
      assert.equal(await page.title(), 'From ad to visit | Daniel Short');
      assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
      assert.match(response.headers()['x-robots-tag'], /noindex/i);
      const movie = page.locator('#movie');
      assert.equal(await movie.evaluate((node) => new URL(node.currentSrc || node.src).pathname), media);
      if (!mobile) {
        const bytes = await context.request.get(origin + media, { headers: { Range: 'bytes=0-31' } });
        assert.equal(bytes.status(), 206);
        assert.match(bytes.headers()['content-type'], /^video\/mp4/);
        assert.match(bytes.headers()['content-range'], /^bytes 0-31\/\d+$/);
        assert.match(bytes.headers()['x-robots-tag'], /noindex/i);
        assert.equal((await bytes.body()).subarray(4, 8).toString('ascii'), 'ftyp');
        await bytes.dispose();
      }
      await page.waitForFunction(() => document.body.dataset.playback === 'playing' && document.getElementById('movie').currentTime > .1);
      const privacyChoice = page.locator('#pcz-reject');
      if (await privacyChoice.isVisible()) await privacyChoice.click();
      await page.getByRole('button', { name: 'Pause campaign' }).click();
      await page.waitForFunction(() => document.body.dataset.playback === 'paused');
      assert.equal(await movie.evaluate((video) => video.paused && video.videoWidth === 1600 && video.videoHeight === 960), true);
      assert.equal(await page.locator('#error').isVisible(), false);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      if (mobile) {
        const national = await page.locator('#national').boundingBox();
        const city = await page.locator('#city').boundingBox();
        assert.ok(national.width > 300 && city.width > 300 && city.y > national.y + national.height);
      }
      await seek(page, .52);
      await page.screenshot({ path: path.join(output, 'hotspot-campaign-' + (mobile ? 'phone' : 'desktop') + '.png'), fullPage: mobile });
      await page.getByRole('button', { name: 'Play campaign' }).click();
      await page.getByRole('slider').press('End');
      await page.waitForFunction(() => document.body.dataset.playback === 'ended');
      assert.equal(await page.locator('#reading').textContent(), '10,000 phones were shown an ad. 800 visited Grand Junction.');
      await checkLabels(page, mobile, output);
      await page.getByRole('button', { name: 'Play campaign' }).click();
      await page.getByRole('slider').press('End');
      await page.waitForFunction(() => document.body.dataset.playback === 'ended');
      await page.getByRole('button', { name: 'Replay campaign' }).click();
      await page.waitForFunction(() => document.body.dataset.playback === 'playing' && document.getElementById('movie').currentTime < 2);
      await page.reload();
      await page.waitForFunction(() => document.body.dataset.playback === 'playing' && document.getElementById('movie').currentTime < 2);
      assert.deepEqual(errors, []);
    } finally {
      await context.close();
    }
  }
  console.log('PASS: default Grand Junction film, all five map labels on desktop and touch, 800 distinct visit markers, seeking, close, replay and refresh.');
};
