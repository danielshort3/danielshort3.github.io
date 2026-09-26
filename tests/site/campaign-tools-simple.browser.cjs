/** Remaining writing/campaign tools: simple controls and current results. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function checkLayout(page, width) {
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `No horizontal page overflow at ${width}px`);
}

async function checkOxford(page) {
  const input = page.locator('#oxford-text');
  const output = page.locator('[data-oxford-output]');
  assert.equal(await page.locator('#oxford-paste').count(), 0);
  await page.locator('#oxford-example').click();
  await page.waitForFunction(() => document.querySelector('[data-oxford-output] .oxford-mark'));
  assert.equal(await page.evaluate(() => document.activeElement.id), 'oxford-example');
  await input.fill('Newer text with no lists.');
  assert.equal(await output.locator('.oxford-mark').count(), 0, 'Editing clears old highlights');
  const payload = await page.evaluate(() => {
    const detail = { toolId: 'oxford-comma-checker', payload: {} };
    document.dispatchEvent(new CustomEvent('tools:session-capture', { detail }));
    return detail.payload;
  });
  assert.equal(payload.output, null, 'Unchecked text must not save old results');
  await page.locator('[data-oxford-form] button[type="submit"]').click();
  assert.match(await page.locator('[data-oxford-summary]').innerText(), /No list candidates/);
  for (const action of ['clear', 'typing', 'restore', 'example']) {
    await page.evaluate(() => {
      const native = File.prototype.text;
      File.prototype.text = function () {
        if (this.name !== 'slow-import.txt') return native.call(this);
        return new Promise(resolve => { window.finishOxfordImport = () => resolve('Late apples, pears and grapes.'); });
      };
    });
    await page.locator('#oxford-file').setInputFiles({ name: 'slow-import.txt', mimeType: 'text/plain', buffer: Buffer.from('Late import') });
    await page.waitForFunction(() => typeof window.finishOxfordImport === 'function');
    if (action === 'clear') await page.locator('[data-oxford-clear]').click();
    else if (action === 'typing') await input.fill('Newer edited text');
    else if (action === 'example') await page.locator('#oxford-example').click();
    else await page.evaluate(() => {
      document.querySelector('#oxford-text').value = 'Restored apples, pears, and grapes.';
      document.dispatchEvent(new CustomEvent('tools:session-applied', { detail: { toolId: 'oxford-comma-checker' } }));
    });
    const expected = await input.inputValue();
    await page.evaluate(() => { window.finishOxfordImport(); delete window.finishOxfordImport; });
    await page.waitForTimeout(50);
    assert.equal(await input.inputValue(), expected, `Late import preserves ${action}`);
    assert(await page.locator('#oxford-import').isEnabled());
    assert.equal(await page.locator('#oxford-file').inputValue(), '', 'Canceled import can be selected again');
    assert(!/Importing/.test(await page.locator('#oxford-input-status').innerText()));
  }
}

async function checkUtm(page) {
  await page.locator('.utmtool-layout').waitFor();
  const landing = page.getByPlaceholder('https://example.com/landing');
  const source = page.getByPlaceholder('google', { exact: true });
  const medium = page.getByPlaceholder('cpc', { exact: true });
  const campaign = page.getByPlaceholder('spring_sale', { exact: true });
  assert.equal(await page.locator('#main [role="tab"]').count(), 0, 'Core inputs do not require tab switching');
  for (const field of [landing, source, medium, campaign]) assert(await field.isVisible(), 'All required campaign fields remain available');
  await landing.fill('https://example.com/landing');
  await source.fill('newsletter');
  await medium.fill('email');
  await campaign.fill('spring_sale');
  await page.getByRole('button', { name: 'Generate links', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.utmtool-workspace-status').textContent.includes('Complete'));
  assert(await page.getByRole('button', { name: 'Copy all', exact: true }).isEnabled());
  const flow = await page.evaluate(() => {
    const output = document.querySelector('.utmtool-output').getBoundingClientRect();
    const actions = document.querySelector('.utmtool-output .utmtool-card-head').getBoundingClientRect();
    const generate = [...document.querySelectorAll('.utmtool-run-actions button')].find(node => node.textContent.includes('Generate links')).getBoundingClientRect();
    return { width: innerWidth, height: innerHeight, outputTop: output.top, actionsBottom: actions.bottom, generateBottom: generate.bottom };
  });
  if (flow.width >= 1051) {
    assert(flow.outputTop >= 50 && flow.actionsBottom < flow.height - 40, 'Desktop result heading and export actions stay visible after Generate');
  } else {
    assert(flow.outputTop - flow.generateBottom < 120, 'Mobile results follow Generate without an advanced-settings stack');
  }
  await page.getByRole('button', { name: 'Copy all', exact: true }).click();
  assert.match(await page.evaluate(() => window.campaignClipboard.at(-1)), /utm_campaign=spring_sale/);
  await campaign.fill('summer_sale');
  assert(await page.getByRole('button', { name: 'Copy all', exact: true }).isDisabled(), 'Edited inputs disable stale copy');
  assert(await page.getByRole('button', { name: 'Export CSV', exact: true }).isDisabled(), 'Edited inputs disable stale export');
  assert.equal(await page.locator('.utmtool-tr').count(), 0, 'Old rows are removed');
  const payload = await page.evaluate(() => {
    const detail = { toolId: 'utm-batch-builder', payload: {} };
    document.dispatchEvent(new CustomEvent('tools:session-capture', { detail }));
    return detail.payload;
  });
  assert.equal(payload.output.meta.results.rows.length, 0, 'Saved session omits old rows');
  await page.getByRole('button', { name: 'Generate links', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.utmtool-workspace-status').textContent.includes('Complete'));
  await page.getByRole('button', { name: 'Copy all', exact: true }).click();
  assert.match(await page.evaluate(() => window.campaignClipboard.at(-1)), /utm_campaign=summer_sale/);
  await page.getByText('More options', { exact: true }).click();
  await page.getByText('Combination rules & formatting', { exact: true }).click();
  assert.equal(await page.locator('#utmtool-panel-rules input[type="radio"]').count(), 4, 'All combination modes remain available');
  await page.getByText('Combination rules & formatting', { exact: true }).click();
  await page.getByText('Import CSV', { exact: true }).first().click();
  await page.evaluate(() => {
    const native = File.prototype.text;
    File.prototype.text = function () {
      if (this.name !== 'slow.csv') return native.call(this);
      return new Promise(resolve => { window.finishUtmImport = () => resolve('page,source\nhttps://example.com/old,old'); });
    };
  });
  await page.locator('#utmtool-csv-file').setInputFiles({ name: 'slow.csv', mimeType: 'text/csv', buffer: Buffer.from('old') });
  await page.waitForFunction(() => typeof window.finishUtmImport === 'function');
  await source.fill('updated');
  await page.evaluate(() => window.finishUtmImport());
  await page.waitForTimeout(60);
  assert.equal(await page.locator('.utmtool-csv-chip').count(), 0, 'A late CSV import cannot override newer inputs');
  await page.locator('#utmtool-csv-file').setInputFiles({ name: 'slow.csv', mimeType: 'text/csv', buffer: Buffer.from('old') });
  await page.getByRole('button', { name: 'Generate links', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.utmtool-workspace-status').textContent.includes('Complete'));
  await page.evaluate(() => window.finishUtmImport());
  await page.waitForTimeout(60);
  assert.equal(await page.locator('.utmtool-csv-chip').count(), 0, 'An explicit generation cancels a pending CSV import');
  assert(await page.getByRole('button', { name: 'Copy all', exact: true }).isEnabled(), 'The new generated output survives an obsolete import');
}

async function checkQr(page, artifactDir, width) {
  const stage = page.locator('#qrtool-stage');
  const meta = page.locator('#qrtool-meta');
  const emptyDescription = page.locator('[data-qrtool-empty-description]');
  await page.waitForFunction(() => document.querySelector('#qrtool-stage')?.dataset.previewState === 'empty');
  assert(await meta.isHidden(), 'The empty preview has one set of instructions');
  assert.equal(await page.locator('#qrtool-canvas').getAttribute('aria-hidden'), 'true', 'An empty canvas is not announced as a QR code');
  assert.match(await emptyDescription.innerText(), /destination URL/);
  const emptyBox = await stage.boundingBox();
  await page.locator('.qrtool-preview-card').screenshot({ path: path.join(artifactDir, `qr-empty-${width}.png`) });
  await page.locator('#qrtool-payload-mode').selectOption('text');
  await page.waitForFunction(() => document.querySelector('[data-qrtool-empty-description]')?.textContent === 'Enter text to see a preview.');
  await page.locator('#qrtool-payload-mode').selectOption('url');
  await page.locator('#qrtool-link-mode').selectOption('managed');
  await page.waitForFunction(() => document.querySelector('[data-qrtool-empty-description]')?.textContent.includes('Create or select a link'));
  await page.locator('#qrtool-link-mode').selectOption('direct');
  await page.locator('#qrtool-data').fill('https://example.com/professional');
  const download = page.getByRole('button', { name: 'Download PNG', exact: true });
  await download.waitFor();
  await page.waitForFunction(() => !document.querySelector('#qrtool-download-png')?.disabled);
  assert.equal(await stage.getAttribute('data-preview-state'), 'ready');
  assert(await meta.isVisible(), 'Generated QR details stay visible');
  assert.equal(await page.locator('#qrtool-canvas').getAttribute('aria-hidden'), 'false');
  const readyBox = await stage.boundingBox();
  assert(Math.abs(readyBox.width - emptyBox.width) < 1 && Math.abs(readyBox.height - emptyBox.height) < 1, 'Empty and generated QR previews reserve the same area');
  assert.equal(await page.locator('#qrtool-export-preset').count(), 0, 'One size selector');
  assert.equal(await page.getByText('Saved presets', { exact: true }).count(), 1, 'One saved presets entry');
  const tabs = await page.locator('[data-qrtool-tab]').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().top));
  assert(Math.max(...tabs) - Math.min(...tabs) <= 1, 'QR settings tabs remain in one row');
  await page.getByRole('tab', { name: 'Download', exact: true }).click();
  await page.locator('#qrtool-image-size').selectOption('512');
  const done = page.waitForEvent('download');
  await download.click();
  const file = await done;
  const saved = path.join(artifactDir, `qr-download-${width}.png`);
  await file.saveAs(saved);
  const png = fs.readFileSync(saved);
  assert.equal(png.readUInt32BE(16), 512, 'Download uses the single chosen size');
  assert.equal(png.readUInt32BE(20), 512);
  await page.getByRole('tab', { name: 'Content', exact: true }).click();
  assert(await download.isVisible(), 'Download remains available in Content');
  await page.getByRole('tab', { name: 'Style', exact: true }).click();
  assert(await download.isVisible(), 'Download remains available in Style');
  await page.getByRole('tab', { name: 'Content', exact: true }).click();
  await page.locator('#qrtool-data').fill('x'.repeat(5000));
  await page.waitForFunction(() => document.querySelector('#qrtool-stage')?.dataset.previewState === 'error');
  assert(await meta.isVisible(), 'An encoder error is never hidden with the empty-state instructions');
  assert.match(await meta.innerText(), /too long/i);
  assert.equal(await page.locator('[data-qrtool-empty-title]').innerText(), 'Preview unavailable');
  assert(await download.isDisabled());
  await page.locator('#qrtool-data').fill('https://example.com/recovered');
  await page.waitForFunction(() => document.querySelector('#qrtool-stage')?.dataset.previewState === 'ready');
  assert(await download.isEnabled(), 'A valid payload recovers from the error');
  await page.locator('#qrtool-clear').click();
  await page.waitForFunction(() => document.querySelector('#qrtool-stage')?.dataset.previewState === 'empty');
  assert(await meta.isHidden());
  assert.match(await emptyDescription.innerText(), /destination URL/);
  if (width === 1440) {
    for (const viewport of [{ width: 1440, height: 350 }, { width: 1024, height: 320 }]) {
      await page.setViewportSize(viewport);
      const fits = await page.locator('#qrtool-empty').evaluate((overlay) => {
        const box = overlay.getBoundingClientRect();
        const textNodes = [...overlay.querySelectorAll('[data-qrtool-empty-title], [data-qrtool-empty-description]')];
        return overlay.scrollHeight <= overlay.clientHeight + 1 && textNodes.every((node) => {
          const textBox = node.getBoundingClientRect();
          return textBox.top >= box.top && textBox.bottom <= box.bottom + 1;
        });
      });
      assert(fits, `QR empty-state instructions fit a ${viewport.width}×${viewport.height} desktop window`);
    }
    await page.setViewportSize({ width, height: 1000 });
  }
}

async function runCampaignToolsSimpleChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const slug of ['oxford-comma-checker', 'utm-batch-builder', 'qr-code-generator']) {
    for (const width of [1440, 390, 320]) {
      const context = await options.browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1000 }, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
      const page = await context.newPage();
      page.setDefaultTimeout(15000);
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await context.route('**/api/tools/**', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{"authenticated":false}' }));
      await context.addInitScript(() => {
        window.campaignClipboard = [];
        Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => window.campaignClipboard.push(text) } });
      });
      try {
        await page.goto(`${options.base}/tools/${slug}`, { waitUntil: 'domcontentloaded' });
        const essential = page.getByRole('button', { name: 'Essential only', exact: true });
        if (await essential.isVisible()) await essential.click();
        await page.evaluate(() => document.fonts.ready);
        if (slug === 'oxford-comma-checker') await checkOxford(page);
        else if (slug === 'utm-batch-builder') await checkUtm(page);
        else await checkQr(page, options.artifactDir, width);
        await checkLayout(page, width);
        await page.locator('#main').scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(options.artifactDir, `${slug}-${width}-simple.png`) });
        assert.deepEqual(errors, []);
        console.log(`Campaign tool passed: ${slug} at ${width}px.`);
      } catch (error) {
        await page.screenshot({ path: path.join(options.artifactDir, `${slug}-${width}-failure.png`) }).catch(() => {});
        error.message = `${slug} ${width}px: ${error.message}`;
        throw error;
      } finally { await context.close(); }
    }
  }
}
module.exports = runCampaignToolsSimpleChecks;

if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'campaign-tools-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runCampaignToolsSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-campaign-tools') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
