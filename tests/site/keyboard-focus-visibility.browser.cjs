/** Native keyboard checks for unobscured skip links and short-height contact fields. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests, ready } = require('../release/fixtures.cjs');

async function tabTo(page, selector, key = 'Tab') {
  for (let step = 0; step < 32; step += 1) {
    await page.keyboard.press(key);
    if (await page.locator(selector).evaluate(element => document.activeElement === element)) return;
  }
  assert.fail(`Native ${key} must reach ${selector}.`);
}

async function runKeyboardFocusVisibilityChecks({ browser, base, artifactDir, browserName = 'browser' }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await isolateRequests(context, base);
    const page = await context.newPage();
    const label = `skip-${browserName}-${width}`;
    try {
      await ready(page, base + '/tools/text-compare');
      if (browserName === 'webkit') {
        // This headless WebKit host excludes links from sequential Tab traversal.
        // Still exercise rendered keyboard focus and Enter; Chromium/Firefox
        // independently prove native traversal without selecting the target.
        await page.keyboard.press('Tab');
        await page.locator('.skip-link').focus();
      } else {
        await tabTo(page, '.skip-link', 'Shift+Tab');
      }
      const geometry = await page.locator('.skip-link').evaluate(link => {
        const rect = link.getBoundingClientRect();
        const hits = [0.1, 0.5, 0.9].flatMap(x => [0.1, 0.5, 0.9].map(y => {
          const hit = document.elementFromPoint(rect.left + rect.width * x, rect.top + rect.height * y);
          return hit === link || link.contains(hit);
        }));
        return { rect: rect.toJSON(), hits, position: getComputedStyle(link).position, width: innerWidth, height: innerHeight };
      });
      assert(geometry.hits.every(Boolean), `${label}: all nine points of the focused skip link are unobscured: ${JSON.stringify(geometry)}`);
      assert(geometry.rect.left >= 0 && geometry.rect.right <= geometry.width && geometry.rect.top >= 0 && geometry.rect.bottom <= geometry.height,
        `${label}: the focused skip link stays inside the viewport.`);
      await page.screenshot({ path: path.join(artifactDir, `${label}.png`) });
      await page.keyboard.press('Enter');
      assert.equal(new URL(page.url()).hash, '#main', `${label}: Enter activates the native main-content target.`);
      results.push({ label, geometry });
      console.log(`Keyboard visibility passed: ${label}`);
    } finally { await context.close(); }
  }

  for (const [width, height, touch] of [[1440, 900, false], [390, 844, true], [320, 480, true], [360, 225, false], [390, 260, true], [390, 225, true]]) {
    const label = `contact-${browserName}-${width}x${height}-${touch ? 'touch' : 'mouse'}`;
    const context = await browser.newContext({
      viewport: { width, height }, hasTouch: touch,
      ...(browserName === 'chromium' || browserName === 'webkit' ? { isMobile: touch } : {}),
      serviceWorkers: 'block', reducedMotion: 'reduce'
    });
    await isolateRequests(context, base);
    const page = await context.newPage();
    const errors = [];
    let submissions = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (new URL(request.url()).pathname === '/api/contact') submissions += 1; });
    try {
      await ready(page, base + '/contact');
      await page.locator('#contact-form-toggle').click();
      await page.locator('#contact-modal.active').waitFor();
      const fields = [];
      for (const [id, text] of [['contact-name', 'Keyboard focus check'], ['contact-email', 'focus@example.test'], ['contact-message', 'Message visibility check']]) {
        await tabTo(page, `#${id}`);
        await page.keyboard.type(text);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const geometry = await page.locator(`#${id}`).evaluate(field => {
          const rect = field.getBoundingClientRect();
          const body = field.closest('.modal-body').getBoundingClientRect();
          const style = getComputedStyle(field);
          const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
          const textY = field.tagName === 'TEXTAREA'
            ? rect.top + parseFloat(style.borderTopWidth) + parseFloat(style.paddingTop) + lineHeight / 2 - field.scrollTop
            : rect.top + rect.height / 2;
          const points = [rect.left + parseFloat(style.paddingLeft) + 12, rect.left + rect.width / 2].map(x => {
            const hit = document.elementFromPoint(x, textY);
            return hit === field || field.contains(hit);
          });
          return {
            focused: document.activeElement === field, value: field.value,
            rect: rect.toJSON(), body: body.toJSON(), textY, points,
            coarsePointer: matchMedia('(pointer: coarse)').matches,
            pageOverflow: document.documentElement.scrollWidth - innerWidth
          };
        });
        assert(geometry.focused && geometry.value === text, `${label}: native Tab and typing edit ${id}.`);
        assert(geometry.textY >= geometry.body.top && geometry.textY <= geometry.body.bottom && geometry.points.every(Boolean),
          `${label}: ${id}'s typed line is visible and unobscured: ${JSON.stringify(geometry)}`);
        // WebKit scrolls the editable line into view and can clip the far input
        // border in a tiny scrollport. The same typed-line/hit-test gate above
        // applies in every engine; Chromium/Firefox also reveal the full input.
        if (id !== 'contact-message' && browserName !== 'webkit') {
          assert(geometry.rect.top >= geometry.body.top - 1 && geometry.rect.bottom <= geometry.body.bottom + 1,
            `${label}: ${id}'s entire input is inside the form scrollport: ${JSON.stringify(geometry)}`);
        }
        assert(geometry.pageOverflow <= 1, `${label}: no horizontal page overflow.`);
        fields.push({ id, geometry });
        await page.screenshot({ path: path.join(artifactDir, `${label}-${id}.png`) });
      }
      if (browserName === 'webkit') {
        await page.keyboard.press('Tab');
        await page.locator('#contact-form [type="submit"]').focus();
      } else {
        await tabTo(page, '#contact-form [type="submit"]');
      }
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const action = await page.locator('#contact-form [type="submit"]').evaluate(button => {
        const rect = button.getBoundingClientRect();
        const scroller = button.closest('.modal-body');
        const body = scroller.getBoundingClientRect();
        const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
        return { rect: rect.toJSON(), body: body.toJSON(), scrollTop: scroller.scrollTop, scrollHeight: scroller.scrollHeight,
          viewportHeight: innerHeight, focused: document.activeElement === button,
          ownHit: hit === button || button.contains(hit), hit: hit?.outerHTML.slice(0, 160) };
      });
      assert(action.focused && action.rect.top >= 0 && action.rect.bottom <= action.viewportHeight && action.ownHit,
        `${label}: keyboard focus reaches an unobscured Send Message action: ${JSON.stringify(action)}`);
      await page.keyboard.press('Escape');
      await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
      assert.equal(submissions, 0, `${label}: no message is submitted.`);
      assert.deepEqual(errors, [], `${label}: no browser exceptions.`);
      results.push({ label, fields, action });
      console.log(`Keyboard visibility passed: ${label}`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
      throw error;
    } finally { await context.close(); }
  }
  fs.writeFileSync(path.join(artifactDir, `keyboard-visibility-${browserName}.json`), JSON.stringify(results, null, 2));
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'keyboard-visibility-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runKeyboardFocusVisibilityChecks({ browser, browserName, base,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-keyboard-visibility') });
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runKeyboardFocusVisibilityChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
