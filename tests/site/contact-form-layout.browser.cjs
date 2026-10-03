/** Checks inline Contact and the shared project dialog without submitting a message. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function assertCenteredFields(page, label) {
  const geometry = await page.locator('[data-contact-inline], #contact-modal').evaluate(modal => {
    const rect = node => {
      const box = node.getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width, height: box.height };
    };
    const inline = modal.hasAttribute('data-contact-inline');
    const content = inline ? modal : modal.querySelector('.modal-content');
    const body = inline ? modal.querySelector('[data-contact-form-body]') : modal.querySelector('.modal-body');
    const form = modal.querySelector('.contact-form');
    return {
      inline, dialog: rect(content), body: rect(body), form: rect(form),
      fields: ['#contact-name', '#contact-email', '#contact-message'].map(selector => rect(modal.querySelector(selector))),
      overflow: body.scrollWidth - body.clientWidth,
      pageOverflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      viewport: { width: window.innerWidth, height: window.innerHeight }
    };
  });
  const center = geometry.inline ? (geometry.form.left + geometry.form.right) / 2 : (geometry.dialog.left + geometry.dialog.right) / 2;
  if (!geometry.inline) assert(Math.abs(center - geometry.viewport.width / 2) < 2, `${label}: dialog remains centered horizontally.`);
  for (const field of geometry.fields) {
    assert(Math.abs((field.left + field.right) / 2 - center) < 2, `${label}: form fields have equal left and right space.`);
    assert(Math.abs(field.width - geometry.fields[0].width) < 1, `${label}: field widths match.`);
    if (!geometry.inline) assert(field.left >= geometry.body.left + 4 && field.right <= geometry.body.right - 4, `${label}: focus indicators have space inside the scroll container.`);
    else assert(field.left >= geometry.body.left - 1 && field.right <= geometry.body.right + 1, `${label}: inline fields fit their document column.`);
  }
  if (!geometry.inline) assert(geometry.dialog.top >= 0 && geometry.dialog.bottom <= geometry.viewport.height + 1, `${label}: dialog stays within the viewport.`);
  assert(geometry.overflow <= 1 && geometry.pageOverflow <= 1, `${label}: no horizontal overflow.`);
}

async function assertMobileActionVisible(page, label) {
  const geometry = await page.locator('[data-contact-inline], #contact-modal').evaluate(modal => {
    const inline = modal.hasAttribute('data-contact-inline');
    const body = (inline ? modal : modal.querySelector('.modal-body')).getBoundingClientRect();
    const action = modal.querySelector('#contact-form [type="submit"]').getBoundingClientRect();
    const masthead = document.querySelector('[data-mobile-site-masthead]')?.getBoundingClientRect();
    return { inline, bodyTop: body.top, bodyBottom: body.bottom, actionTop: action.top, actionBottom: action.bottom, viewportHeight: window.innerHeight, mastheadBottom: Math.max(0, masthead?.bottom || 0) };
  });
  assert(geometry.actionTop >= geometry.bodyTop - 1, `${label}: Send Message is below the form header.`);
  if (geometry.inline) assert(geometry.actionTop >= geometry.mastheadBottom - 1, `${label}: inline Send Message clears the mobile masthead.`);
  assert(geometry.actionBottom <= Math.min(geometry.bodyBottom, geometry.viewportHeight) + 1, `${label}: Send Message stays visible without scrolling.`);
}

async function runCase({ browser, base, artifactDir, width, height, route = '/portfolio/digitGenerator', withDraft = false }) {
  const label = `${route === '/contact' ? 'contact' : 'project'}-${width}x${height}${withDraft ? '-restored' : ''}`;
  const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce', serviceWorkers: 'block', isMobile: width < 600, hasTouch: width < 600 });
  if (withDraft) await context.addInitScript(() => {
    sessionStorage.setItem('ds:session-draft:v1:contact:personal', JSON.stringify({
      updated: Date.now(), data: { name: '', email: '', message: 'Hi Daniel, I have a question about this project.' }
    }));
  });
  const page = await context.newPage();
  page.setDefaultTimeout(12000);
  const errors = [];
  let submissions = 0;
  page.on('pageerror', error => errors.push(error.message));
  await context.route(`${base}/api/contact`, async request => {
    submissions += 1;
    await request.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"No contact submission permitted in this layout test."}' });
  });
  await context.route(`${base}/api/demos/digit-generator/**`, async request => {
    const action = new URL(request.request().url()).pathname.split('/').at(-1);
    const body = action === 'generate'
      ? { rows: 6, cols: 6, latent_dim: 20, images: Array.from({ length: 6 }, () => Array.from({ length: 6 }, () => 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')) }
      : { status: 'ready', model_loaded: true };
    await request.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  try {
    const response = await page.goto(base + route, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label}: page loads.`);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    const inline = route === '/contact';
    const launcher = inline ? null : page.locator('.project-question-link');
    const dialog = inline ? page.locator('[data-contact-inline]') : page.getByRole('dialog', { name: 'Send a Message' });
    if (!inline) {
      await launcher.click();
      await dialog.waitFor({ state: 'visible' });
      await page.waitForFunction(() => document.querySelector('#contact-modal').classList.contains('active') && document.querySelector('#contact-modal .modal-content').contains(document.activeElement));
    } else {
      await dialog.waitFor({ state: 'visible' });
      assert.equal(await page.locator('#contact-modal, #contact-form-toggle').count(), 0, `${label}: Contact exposes its form directly.`);
      assert.equal(await page.getByRole('dialog', { name: 'Send a Message' }).count(), 0, `${label}: inline Contact has no dialog role.`);
      assert(await dialog.evaluate(element => !element.closest('[inert]') && !document.body.classList.contains('modal-open')), `${label}: inline Contact does not isolate or lock the document.`);
    }
    await page.evaluate(() => document.fonts.ready);
    await assertCenteredFields(page, label);
    if (withDraft) {
      assert(await dialog.getByText('Draft restored').isVisible(), `${label}: draft recovery notice is shown.`);
      assert.match(await page.locator('#contact-message').inputValue(), /question about this project/, `${label}: draft text is restored.`);
    }
    if (width < 600 && !inline) await assertMobileActionVisible(page, label);
    for (const [selector, text] of [['#contact-name', 'Layout Review'], ['#contact-email', 'layout@example.com'], ['#contact-message', 'Hi Daniel, I have a question about Synthetic Digit Generator.']]) {
      const field = page.locator(selector);
      await field.fill(text);
      await field.press('ControlOrMeta+A');
      const focus = await field.evaluate(element => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        let scroller = element.closest('.modal-body');
        if (!scroller) {
          for (let ancestor = element.parentElement; ancestor; ancestor = ancestor.parentElement) {
            if (/(hidden|auto|scroll|clip)/.test(getComputedStyle(ancestor).overflowX)) { scroller = ancestor; break; }
          }
        }
        const boundary = scroller?.getBoundingClientRect() || { left: 0, right: innerWidth };
        return {
          focused: document.activeElement === element,
          selected: element.selectionStart === 0 && element.selectionEnd === element.value.length,
          outline: parseFloat(style.outlineWidth), style: style.outlineStyle, shadow: style.boxShadow,
          left: rect.left - parseFloat(style.outlineOffset) - parseFloat(style.outlineWidth),
          right: rect.right + parseFloat(style.outlineOffset) + parseFloat(style.outlineWidth),
          scrollerLeft: boundary.left, scrollerRight: boundary.right
        };
      });
      assert(focus.focused && (selector === '#contact-email' || focus.selected), `${label}: text remains selectable in ${selector}.`);
      assert(focus.outline >= 2 && focus.style === 'solid', `${label}: ${selector} retains visible keyboard focus.`);
      assert.equal(focus.shadow, 'none', `${label}: ${selector} has one focus outline without a competing glow.`);
      assert(focus.left >= focus.scrollerLeft - 1 && focus.right <= focus.scrollerRight + 1, `${label}: ${selector}'s focus outline is not clipped.`);
    }
    await page.screenshot({ path: path.join(artifactDir, `contact-form-${label}-selected.png`) });
    const message = 'A longer message with project details and a URL: https://example.com/' + 'long-path-'.repeat(120) + '\nAdditional details.\n'.repeat(80);
    await page.locator('#contact-message').fill(message.slice(0, 3990));
    await assertCenteredFields(page, `${label} long message`);
    // Exercise the supported textarea resize affordance inside the scrolling dialog.
    await page.locator('#contact-message').evaluate(element => { element.style.height = '520px'; });
    const submit = page.locator('#contact-form [type="submit"]');
    await submit.scrollIntoViewIfNeeded();
    assert(await submit.isVisible(), `${label}: actions remain reachable after resizing the message.`);
    if (width < 600) await assertMobileActionVisible(page, `${label} resized message`);
    await submit.focus();
    await page.screenshot({ path: path.join(artifactDir, `contact-form-${label}-scrolled.png`) });
    await assertCenteredFields(page, `${label} resized message`);
    assert.equal(await page.locator('[data-contact-reset]').count(), 0, `${label}: no destructive Clear form action.`);
    assert.equal(await page.locator('#contact-message').inputValue(), message.slice(0, 3990), `${label}: resizing preserves the draft.`);
    await page.keyboard.press('Escape');
    if (inline) {
      assert(await dialog.isVisible(), `${label}: Escape does not hide the inline form.`);
      assert(await dialog.evaluate(element => !element.closest('[inert]') && !document.body.classList.contains('modal-open')), `${label}: the inline document remains unlocked.`);
      await page.keyboard.press('Tab');
      assert(await dialog.evaluate(element => !element.contains(document.activeElement)), `${label}: Tab can leave the inline form instead of being trapped.`);
    } else {
      await dialog.waitFor({ state: 'hidden' });
      assert.equal(await launcher.evaluate(element => document.activeElement === element), true, `${label}: closing restores focus to the launcher.`);
    }
    assert.equal(submissions, 0, `${label}: no messages were submitted.`);
    assert.deepEqual(errors, [], `${label}: flow has no page exceptions.`);
    console.log(`Contact form layout passed: ${label}`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `contact-form-${label}-failure.png`) }).catch(() => {});
    throw error;
  } finally {
    await context.close();
  }
}

async function runContactFormLayoutChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const [width, height] of [[1440, 1000], [390, 844], [320, 844], [320, 480]]) {
    await runCase({ browser, base, artifactDir, width, height });
  }
  for (const [width, height] of [[1440, 1000], [390, 844], [320, 480]]) {
    await runCase({ browser, base, artifactDir, width, height, route: '/contact' });
  }
  await runCase({ browser, base, artifactDir, width: 320, height: 640, route: '/contact', withDraft: true });
  await runCase({ browser, base, artifactDir, width: 320, height: 640, withDraft: true });
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'contact-layout-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runContactFormLayoutChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-contact-layout') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runContactFormLayoutChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
