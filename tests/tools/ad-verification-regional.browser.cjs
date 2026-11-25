'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const regional = require('../../js/demos/ad-verification-regional');

const root = '[data-region-root]';
const completed = (count) => root + '[data-completed-batches="' + count + '"]';

async function totals(page) {
  return Promise.all(['views', 'observations', 'matched'].map(async (key) =>
    Number(await page.locator('[data-region-total="' + key + '"]').innerText())));
}

async function dismissConsent(page) {
  const essential = page.locator('#pcz-reject');
  if (await essential.isVisible()) await essential.click();
}

async function ready(page) {
  await page.locator(root + '[data-phase="ready"]').waitFor();
  await dismissConsent(page);
}

async function assertFits(page, label) {
  const dimensions = await page.evaluate(() => ({
    display: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
    dialogContent: document.querySelector('dialog[open]')?.scrollWidth,
    dialogDisplay: document.querySelector('dialog[open]')?.clientWidth
  }));
  assert.ok(dimensions.content <= dimensions.display + 1, label + ' overflows the page');
  if (dimensions.dialogDisplay) {
    assert.ok(dimensions.dialogContent <= dimensions.dialogDisplay + 1, label + ' overflows the dialog');
  }
}

async function runCampaign(page) {
  await page.locator('[data-region-run]').click();
  await page.locator(root + '[data-phase="complete"]').waitFor();
  assert.deepEqual(await totals(page), [6, 8, 2]);
}

async function downloadProof(page) {
  const downloading = page.waitForEvent('download');
  await page.locator('[data-region-export]').click();
  const file = await downloading;
  assert.equal(file.suggestedFilename(), 'regional-campaign-proof.json');
  return JSON.parse(fs.readFileSync(await file.path(), 'utf8'));
}

module.exports = async function runRegionalChecks({ browser, url, output }) {
  const origin = new URL(url).origin;
  const errors = [];
  const failed = [];
  // A browser request for an actual ad, location feed or tracking pixel is unexpected on this local demo.
  const external = [];
  const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await desktop.route('**/*', (route) => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    external.push(route.request().url());
    return route.abort();
  });
  try {
    const page = await desktop.newPage();
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => { if (response.status() >= 400) failed.push(response.url()); });
    await page.goto(url);
    await ready(page);
    assert.equal(new URL(page.url()).searchParams.get('lab'), 'regional');
    assert.equal(await page.locator('h1:visible').count(), 1);
    assert.match(await page.locator('h1:visible').innerText(), /ad view.*regional visit/i);
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    assert.equal(await page.locator('[data-region-claim-submit]').isDisabled(), true);
    assert.deepEqual(await totals(page), [0, 0, 0]);
    await page.screenshot({ path: path.join(output, 'regional-ready-desktop.png'), fullPage: true });

    const next = page.locator('[data-region-next]');
    await next.dblclick();
    await page.locator(completed(1) + '[data-phase="ready"]').waitFor();
    assert.deepEqual(await totals(page), [6, 0, 0]);
    await next.click();
    await page.locator(completed(2) + '[data-phase="ready"]').waitFor();
    assert.deepEqual(await totals(page), [6, 5, 1]);
    await next.click();
    await page.locator(completed(3) + '[data-phase="complete"]').waitFor();
    assert.deepEqual(await totals(page), [6, 8, 2]);
    assert.equal(await page.locator('[data-region-journey]').count(), 6);
    assert.equal(await page.locator('[data-region-batch]').count(), 3);
    assert.equal(await next.isDisabled(), true);

    const dialog = page.locator('[data-region-inspector]');
    const cases = [
      ['Three-stop traveler', [/Field Guide/i, /Weekend Reader/i, /Downtown/i, /trailhead/i, /lodging/i, /once/i]],
      ['Day-30 arrival', [/30 days/i, /matched/i]],
      ['Missing app receipt', [/missing signed view/i]],
      ['Transit passenger', [/transit only/i]],
      ['Late arrival', [/outside window/i]],
      ['Unlinked pass', [/\bunlinked\b/i, /no authorized campaign token/i]]
    ];
    for (const [name, expected] of cases) {
      const row = page.locator('[data-region-journey]').filter({ hasText: name });
      await row.click();
      await dialog.waitFor();
      const evidence = await dialog.innerText();
      for (const value of expected) assert.match(evidence, value, name);
      await page.keyboard.press('Escape');
      assert.equal(await dialog.isVisible(), false);
    }

    await page.locator('[data-region-batch]').last().click();
    await dialog.waitFor();
    assert.match(await dialog.innerText(), /fingerprint/i);
    await dialog.locator('[data-region-close]').click();
    await page.locator('[data-region-verify]').click();
    await page.locator('[data-region-verify-result][data-verdict="verified"]').waitFor();
    const proof = await downloadProof(page);
    assert.equal((await regional.verifyProof(proof)).valid, true);
    assert.equal(proof.batches.length, 3);
    assert.deepEqual(proof.batches[2].totals, { views: 6, observations: 8, matched: 2 });
    assert.doesNotMatch(JSON.stringify(proof), /token|credential|traveler|three-stops|field-guide|weekend-reader|downtown|trailhead|lodging|visiblePercent|dwellMinutes|privateKey|viewReceipts/i);
    const changed = structuredClone(proof);
    changed.batches[2].totals.matched += 1;
    assert.equal((await regional.verifyProof(changed)).valid, false);

    const input = page.locator('[data-region-claim-value]');
    const check = page.locator('[data-region-claim-submit]');
    const feedback = page.locator('[data-region-claim-result]');
    for (const bad of ['', '-1', '2.7', 'three']) {
      await input.fill(bad);
      await check.click();
      assert.equal(await feedback.getAttribute('data-verdict'), 'invalid');
      assert.match(await feedback.innerText(), /enter|whole|integer/i);
    }
    await input.fill('99');
    await check.dblclick();
    await page.locator('[data-region-claim-result][data-verdict="unsupported"]').waitFor();
    assert.match(await feedback.innerText(), /claimed 99.*found 2/i);
    await input.fill('2');
    await input.press('Enter');
    await page.locator('[data-region-claim-result][data-verdict="supported"]').waitFor();
    assert.match(await feedback.innerText(), /found 2/i);
    await page.locator('[data-region-example]').click();
    await page.locator('[data-region-claim-result][data-verdict="unsupported"]').waitFor();
    assert.match(await feedback.innerText(), /claimed 5.*found 2/i);
    assert.deepEqual(await downloadProof(page), proof, 'Checking a claim must never change the signed ledger');
    await page.locator('[data-region-root]').evaluate((node) => node.scrollIntoView());
    await page.screenshot({ path: path.join(output, 'regional-complete-desktop.png'), fullPage: true });
    await assertFits(page, '1440px desktop');
    await page.reload();
    await ready(page);
    assert.deepEqual(await totals(page), [0, 0, 0]);
    await page.locator('[data-region-run]').click();
    await page.locator(completed(1) + '[data-phase="running"]').waitFor();
    assert.equal(await page.locator('[data-region-reset]').isEnabled(), true);
    await page.locator('[data-region-reset]').click();
    await ready(page);
    // The retired playback would have sealed more batches after its pacing timer fires.
    await page.waitForTimeout(1100);
    assert.deepEqual(await totals(page), [0, 0, 0]);
    await runCampaign(page);
    const fresh = await downloadProof(page);
    assert.notEqual(fresh.campaign, proof.campaign);
    assert.notEqual(fresh.batches[0].fingerprint, proof.batches[0].fingerprint);
    const lab = page.getByRole('link', { name: /original receipt/i });
    await lab.click();
    await page.locator('[data-receipt-root]:visible').waitFor();
    assert.equal(new URL(page.url()).searchParams.get('lab'), 'receipts');
    await page.getByRole('link', { name: /regional campaign lab/i }).click();
    await ready(page);
    assert.equal(new URL(page.url()).searchParams.get('lab'), 'regional');
    assert.deepEqual(await totals(page), [0, 0, 0]);
    assert.deepEqual(errors, []);
    assert.deepEqual(failed, []);
    assert.deepEqual(external, []);
  } finally {
    await desktop.close();
  }

  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, reducedMotion: 'reduce' });
  try {
    const page = await mobile.newPage();
    await page.goto(url);
    await ready(page);
    await runCampaign(page);
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await assertFits(page, width + 'px campaign');
      await page.locator('[data-region-journey]').first().tap();
      await page.locator('[data-region-inspector]').waitFor();
      await assertFits(page, width + 'px private evidence');
      await page.locator('[data-region-close]').tap();
      await page.locator('[data-region-batch]').last().tap();
      await page.locator('[data-region-inspector]').waitFor();
      await assertFits(page, width + 'px signed hashes');
      await page.locator('[data-region-close]').tap();
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(output, 'regional-complete-' + width + '.png'), fullPage: true });
    }
  } finally {
    await mobile.close();
  }

  const noJs = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await noJs.newPage();
    await page.goto(url);
    assert.match(await page.locator('noscript').innerText(), /JavaScript/i);
    assert.equal(await page.locator('[data-region-run]').isDisabled(), true);
  } finally {
    await noJs.close();
  }
  const missing = await browser.newContext();
  try {
    const page = await missing.newPage();
    await page.route('**/js/demos/ad-verification-regional.js*', (route) => route.abort());
    await page.goto(url);
    await page.locator(root + '[data-phase="error"]').waitFor();
    assert.equal(await page.locator('[data-region-run]').isDisabled(), true);
    assert.match(await page.locator('[data-region-progress]').innerText(), /unavailable|could not|needs|failed|load/i);
  } finally {
    await missing.close();
  }
  console.log('PASS: regional app views, local stops, two unique matches, claim challenge, aggregate export, refresh, reset, phone controls and offline verification.');
};
