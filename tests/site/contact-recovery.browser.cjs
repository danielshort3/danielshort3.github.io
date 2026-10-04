/** Inline and retained project-dialog recovery. Requests are intercepted; no messages are sent. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');
const UNKNOWN = 'We couldn’t confirm delivery. Your message may have been sent. Your draft is still here.';

async function fixture(browser, base, viewport = { width: 1440, height: 900 }, inline = false) {
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'block' });
  await context.route('**/api/contact', (route) => route.abort('blockedbyclient'));
  await context.route('**/*', (route) => new URL(route.request().url()).origin === base ? route.fallback() : route.abort('blockedbyclient'));
  await context.addInitScript(() => {
    const nativeFetch = window.fetch.bind(window);
    window.__contactFixture = { mode: 'success', calls: 0, pending: [] };
    window.fetch = (input, options) => {
      if (new URL(typeof input === 'string' ? input : input.url, location.href).pathname !== '/api/contact') return nativeFetch(input, options);
      const fixture = window.__contactFixture;
      fixture.calls += 1;
      const json = (status, data) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
      if (fixture.mode === 'success') return Promise.resolve(json(200, { ok: true }));
      if (fixture.mode === 'rejection') return Promise.resolve(json(400, { ok: false, code: 'CONTACT_REJECTED' }));
      if (fixture.mode === 'invalid') return Promise.resolve(json(200, {}));
      if (fixture.mode === 'body') {
        // Headers arrive, but the body deliberately never completes until the
        // test releases it. This catches header-only timeout implementations.
        return Promise.resolve(new Response(new ReadableStream({ start(controller) { fixture.pending.push(() => { controller.enqueue(new TextEncoder().encode('{"ok":true}')); controller.close(); }); } }), { status: 200 }));
      }
      return new Promise((resolve) => fixture.pending.push(() => resolve(json(200, { ok: true }))));
    };
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(base + (inline ? '/contact' : '/portfolio/website'), { waitUntil: 'domcontentloaded' });
  if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
  if (!inline) {
    await page.locator('.project-question-link').click();
    await page.locator('#contact-modal.active').waitFor();
  } else {
    await page.waitForFunction(() => window.__contactModalReady && document.querySelector('[data-contact-inline]'));
    await page.locator('#contact-form [type="submit"]').click();
    await page.locator('#contact-errors').waitFor({ state: 'visible' });
    assert.equal(await page.evaluate(() => document.activeElement.id), 'contact-errors', 'Inline validation focuses the visible summary.');
    assert.equal(await page.locator('#contact-errors a').count(), 3);
    await page.locator('#contact-errors a').first().click();
    assert.equal(await page.evaluate(() => document.activeElement.id), 'contact-name', 'Summary links focus the editable field.');
    assert.equal(await page.evaluate(() => window.__contactFixture.calls), 0, 'Invalid inline input never sends.');
  }
  await page.locator('#contact-name').fill('Contact Test');
  await page.locator('#contact-email').fill('contact-test@example.com');
  await page.locator('#contact-message').fill('A local-only draft. No email is sent by this test.');
  return { context, page, errors };
}

async function runSurfaceRecoveryChecks({ browser, base, artifactDir }, inline) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const recovery = await fixture(browser, base, { width: 390, height: 844 }, inline);
  try {
    const { page } = recovery;
    await page.reload({ waitUntil: 'domcontentloaded' });
    if (!inline) await page.locator('.project-question-link').click();
    else await page.waitForFunction(() => window.__contactModalReady);
    await expect(page.locator('#contact-message')).toHaveValue('A local-only draft. No email is sent by this test.');
    await page.getByText('Draft restored', { exact: false }).waitFor();
    await page.screenshot({ path: path.join(artifactDir, 'contact-draft-restored-mobile.png') });
    assert.equal(await page.locator('[data-contact-reset]').count(), 0);
    const notice = page.locator('.draft-recovery-notice');
    await notice.getByRole('button', { name: /discard/i }).click();
    assert.equal(await page.locator('#contact-message').inputValue(), '');
    await page.reload({ waitUntil: 'domcontentloaded' });
    if (!inline) await page.locator('.project-question-link').click();
    else await page.waitForFunction(() => window.__contactModalReady);
    const expectedMessage = inline ? '' : await page.locator('.project-question-link').getAttribute('data-contact-message');
    await expect(page.locator('#contact-message'), 'Discarded text stays removed; a new project opener may supply its own prefill.').toHaveValue(expectedMessage);
    assert.equal(await page.locator('#contact-name').inputValue(), '');
    assert.equal(await page.locator('#contact-email').inputValue(), '');
    assert.deepEqual(recovery.errors, []);
  } finally { await recovery.context.close(); }

  for (const mode of ['success', 'rejection', 'invalid', 'pending', 'body']) {
    const { context, page, errors } = await fixture(browser, base, mode === 'body' ? { width: 320, height: 844 } : undefined, inline);
    try {
      if (mode === 'pending' || mode === 'body') await page.clock.install();
      await page.evaluate((mode) => { window.__contactFixture.mode = mode; }, mode);
      await page.locator('#contact-form [type="submit"]').click();
      if (mode === 'success') {
        await page.locator('#contact-success').waitFor({ state: 'visible' });
        assert.equal(await page.locator('#contact-message').inputValue(), '');
        assert.equal(await page.evaluate(() => window.SiteSessionDrafts.read('contact:personal')), null);
      } else {
        if (mode === 'pending' || mode === 'body') {
          if (!inline) {
            await page.locator('#contact-modal .modal-close').click();
            await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
            await page.locator('.project-question-link').click();
          }
          assert.equal(await page.locator('#contact-form [type="submit"]').isDisabled(), true);
          await page.locator('#contact-form').evaluate((form) => { form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })); });
          assert.equal(await page.evaluate(() => window.__contactFixture.calls), 1, 'close/reopen and repeated submit do not duplicate delivery');
          await page.clock.fastForward(25001);
        }
        await page.getByRole('button', { name: 'Retry sending', exact: true }).waitFor();
        if (mode !== 'rejection') assert.equal(await page.locator('#contact-status').innerText(), UNKNOWN);
        assert.equal(await page.locator('#contact-alt a').isVisible(), true);
        assert.equal(await page.locator('#contact-form').getAttribute('aria-busy'), 'false');
        assert.equal(await page.evaluate(() => window.SiteContact.canLeave()), true);
        assert.equal(await page.evaluate(() => window.__contactFixture.calls), 1, 'no automatic retry');
        assert.equal(await page.locator('#contact-message').inputValue(), 'A local-only draft. No email is sent by this test.');
        await page.screenshot({ path: path.join(artifactDir, `contact-${mode}.png`) });
        await page.evaluate(() => { window.__contactFixture.pending.splice(0).forEach((complete) => complete()); });
        assert.equal(await page.locator('#contact-message').inputValue(), 'A local-only draft. No email is sent by this test.', 'late success does not clear a timed-out draft');
        await page.evaluate(() => { window.__contactFixture.mode = 'success'; });
        await page.getByRole('button', { name: 'Retry sending', exact: true }).click();
        await page.locator('#contact-success').waitFor({ state: 'visible' });
        assert.equal(await page.evaluate(() => window.__contactFixture.calls), 2);
      }
      assert.deepEqual(errors, [], `${mode}: no browser exceptions`);
    } finally { await context.close(); }
  }
  console.log(`Contact ${inline ? 'inline' : 'dialog'} browser recovery passed: validation, refresh, discard, confirmation, rejection, malformed response, request/body deadlines, duplicate prevention, late responses and explicit retry.`);
}

async function runContactRecoveryChecks(options) {
  for (const inline of [false, true]) {
    await runSurfaceRecoveryChecks({ ...options, artifactDir: path.join(options.artifactDir, inline ? 'inline' : 'dialog') }, inline);
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'contact-recovery-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true });
    await runContactRecoveryChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-contact-recovery') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}
module.exports = runContactRecoveryChecks;
if (require.main === module) main().catch((error) => { console.error(error); process.exitCode = 1; });
