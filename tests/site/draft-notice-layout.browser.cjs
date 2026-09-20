'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require('playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');

async function geometry(page) {
  return page.evaluate(() => {
    const rect = node => {
      const r = node.getBoundingClientRect();
      return { x: r.x, y: r.y, width: r.width, height: r.height, right: r.right, bottom: r.bottom };
    };
    const notice = document.querySelector('.draft-recovery-notice');
    const foreground = notice?.closest('#contact-modal') || document;
    const overlap = (a, b) => Math.min(a.right, b.right) - Math.max(a.x, b.x) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y) > 1;
    const noticeBox = notice ? rect(notice) : null;
    const collisions = notice ? [...foreground.querySelectorAll('input, textarea, select, button, a[href]')]
      .filter(node => !notice.contains(node) && node.checkVisibility() && overlap(noticeBox, rect(node)))
      .map(node => node.id || node.textContent.trim().slice(0, 60)) : [];
    return {
      notice: noticeBox, discard: notice ? rect(notice.querySelector('button')) : null, collisions,
      original: document.querySelector('#textcompare-original') ? rect(document.querySelector('#textcompare-original')) : null,
      after: document.querySelector('#textcompare-revised') ? rect(document.querySelector('#textcompare-revised')) : null,
      name: document.querySelector('#contact-modal.active #contact-name') ? rect(document.querySelector('#contact-name')) : null,
      dialog: document.querySelector('#contact-modal.active .modal-content') ? rect(document.querySelector('#contact-modal .modal-content')) : null,
      overflow: document.documentElement.scrollWidth - innerWidth
    };
  });
}

async function assertReadableChrome(page, label, { contact = false } = {}) {
  const issues = await page.evaluate(({ contact }) => {
    const failures = [];
    const bounds = node => node.getBoundingClientRect();
    const textBounds = node => {
      const range = document.createRange();
      range.selectNodeContents(node);
      return range.getBoundingClientRect();
    };
    const contained = (inner, outer) => inner.left >= outer.left - 1 && inner.right <= outer.right + 1 && inner.top >= outer.top - 1 && inner.bottom <= outer.bottom + 1;
    if (!contact) {
      const masthead = document.querySelector('.mobile-site-masthead');
      if (masthead?.checkVisibility()) {
        const header = bounds(masthead);
        for (const selector of ['.mobile-site-masthead__brand', '.mobile-site-masthead__explore-button', '.mobile-site-masthead__search-button']) {
          const node = masthead.querySelector(selector);
          if (!contained(bounds(node), header)) failures.push(`${selector} escapes the header`);
          if (bounds(node).right > innerWidth + 1) failures.push(`${selector} escapes the screen`);
          if (bounds(node).height < 44 || bounds(node).width < 44) failures.push(`${selector} is below touch-target size`);
        }
        const exploreLabel = masthead.querySelector('.mobile-site-masthead__explore-button span');
        if (!contained(textBounds(exploreLabel), bounds(exploreLabel.closest('button')))) failures.push('Explore label clips');
        const reserved = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-height'));
        if (Math.abs(header.height - reserved) > 1) failures.push('Page clearance does not follow the wrapped header');
      }
      const signIn = document.querySelector('[data-personal-tool-account-bar] [data-tools-action="sign-in"]');
      if (signIn?.checkVisibility()) {
        if (!contained(textBounds(signIn), bounds(signIn))) failures.push('Sign in label clips');
        if (signIn.scrollWidth > signIn.clientWidth + 1) failures.push('Sign in label overflows');
        if (getComputedStyle(signIn).whiteSpace !== 'nowrap') failures.push('Sign in can split across lines');
      }
    } else {
      const strip = document.querySelector('#contact-modal.active .modal-title-strip');
      const title = strip.querySelector('.modal-title');
      if (!contained(textBounds(title), bounds(strip))) failures.push('Contact heading crosses its title strip/divider');
      if (strip.scrollHeight > strip.clientHeight + 1) failures.push('Contact heading needs its own vertical scrolling');
      const close = bounds(document.querySelector('#contact-modal .modal-close'));
      const heading = textBounds(title);
      if (Math.min(close.right, heading.right) > Math.max(close.left, heading.left) + 1 && Math.min(close.bottom, heading.bottom) > Math.max(close.top, heading.top) + 1) failures.push('Contact heading overlaps Close');
    }
    return failures;
  }, { contact });
  assert.deepEqual(issues, [], `${label}: enlarged text and shared controls remain fully readable`);
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'draft-layout-env-'));
  const server = createLocalServer({ envDir });
  const artifacts = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-draft-notice');
  fs.mkdirSync(artifacts, { recursive: true });
  let browser;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch();
    for (const [width, height, fontPercent] of [[1440, 900, 100], [390, 844, 100], [320, 740, 100], [320, 900, 200]]) {
      const label = `${width}x${height}-${fontPercent}`;
      const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce', serviceWorkers: 'block' });
      await context.route('https://**', route => route.abort());
      await context.route('**/api/**', route => route.fulfill({ status: 200, json: { ok: true, sessions: [] } }));
      const page = await context.newPage();
      const ready = async () => {
        await page.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
        if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
        await page.evaluate(async fontPercent => {
          document.documentElement.style.fontSize = `${fontPercent}%`;
          await document.fonts.ready;
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        }, fontPercent);
      };
      try {
        await page.goto(`${base}/tools/text-compare`);
        await ready();
        const before = await geometry(page);
        await page.evaluate(() => sessionStorage.setItem('ds:session-draft:v1:tools:text-compare', JSON.stringify({ updated: Date.now(), data: { 'textcompare-original': { kind: 'value', value: 'A recovered before draft' }, 'textcompare-revised': { kind: 'value', value: 'A recovered after draft' } } })));
        await page.reload();
        await ready();
        await expect(page.locator('.draft-recovery-notice')).toBeVisible();
        const restored = await geometry(page);
        assert.deepEqual(restored.collisions, [], `${label}: notice cannot cover a control`);
        assert(restored.discard.width >= 44 && restored.discard.height >= 44);
        assert(restored.overflow <= 1, `${label}: no page overflow`);
        await assertReadableChrome(page, label);
        for (const field of ['original', 'after']) assert(Math.abs(restored[field].y - before[field].y) <= 1, `${label}: restoring a draft does not move ${field}`);
        await page.screenshot({ path: path.join(artifacts, `tools-${label}.png`) });
        await page.locator('.draft-recovery-notice button').click();
        const discarded = await geometry(page);
        for (const field of ['original', 'after']) assert(Math.abs(discarded[field].y - before[field].y) <= 1, `${label}: dismissing notice does not move ${field}`);
        await page.evaluate(() => sessionStorage.setItem('ds:session-draft:v1:contact:personal', JSON.stringify({ updated: Date.now(), data: { name: 'Example Person', email: 'example@example.com', message: 'A message restored inside its dialog.' } })));
        await page.goto(`${base}/contact`);
        if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
        await page.evaluate(fontPercent => { document.documentElement.style.fontSize = `${fontPercent}%`; }, fontPercent);
        await page.locator('#contact-form-toggle').click();
        await expect(page.locator('#contact-modal.active .draft-recovery-notice')).toBeVisible();
        const contact = await geometry(page);
        assert.deepEqual(contact.collisions, [], `${label}: contact notice cannot cover a control`);
        assert(contact.notice.x >= contact.dialog.x && contact.notice.right <= contact.dialog.right);
        assert(contact.notice.y >= contact.dialog.y && contact.notice.bottom <= contact.dialog.bottom);
        await assertReadableChrome(page, label, { contact: true });
        await page.screenshot({ path: path.join(artifacts, `contact-${label}.png`) });
        await page.locator('#contact-modal .draft-recovery-notice button').click();
        const cleared = await geometry(page);
        assert(Math.abs(cleared.name.y - contact.name.y) <= 1, `${label}: Discard preserves dialog field placement (${JSON.stringify({ before: contact.name, after: cleared.name })})`);
        await page.locator('#contact-modal .modal-close').click();
        await expect(page.locator('#contact-modal.active')).toHaveCount(0);
        await page.locator('#contact-form-toggle').click();
        assert.equal(await page.locator('#contact-modal .draft-recovery-slot').count(), 0, `${label}: closed notice releases its reserved space`);
        console.log(`Draft notice placement passed: ${label}, no covered controls or tool/editor/dialog shifts.`);
      } finally { await context.close(); }
    }
    console.log(`Draft notice evidence: ${artifacts}`);
  } finally {
    await browser?.close(); server.closeAllConnections();
    await new Promise(resolve => server.close(resolve)); fs.rmdirSync(envDir);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
