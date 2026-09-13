/** Built Text Compare workflow checks. Clipboard and account requests are stubbed. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function assertLayout(page, width, label) {
  const state = await page.evaluate(() => {
    const original = document.querySelector('#textcompare-original');
    const revised = document.querySelector('#textcompare-revised');
    const comparison = document.querySelector('#textcompare-view-comparison');
    const boxes = [original, revised, comparison].map(node => {
      const box = node.getBoundingClientRect();
      return { x: box.x, right: box.right, y: box.y, bottom: box.bottom, width: box.width, height: box.height, hidden: Boolean(node.closest('[hidden]')) };
    });
    return { boxes, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth };
  });
  assert(state.overflow <= 1, `${label}: page has no horizontal overflow.`);
  state.boxes.forEach(box => assert(!box.hidden && box.width > 0 && box.height > 0 && box.x >= -1 && box.right <= width + 1, `${label}: both editors and comparison fit without being hidden.`));
  assert(state.boxes[2].y >= Math.max(state.boxes[0].bottom, state.boxes[1].bottom), `${label}: the result follows both drafts.`);
}

async function runCase({ browser, base, artifactDir }, width) {
  const context = await browser.newContext({ viewport: { width, height: width < 500 ? 844 : 1000 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  const accountRequests = [];
  let stage = 'initial';
  page.on('pageerror', error => errors.push(error.message));
  await context.route('**/api/tools/**', async route => {
    if (new URL(route.request().url()).pathname !== '/api/tools/auth/session') accountRequests.push(route.request().url());
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{"authenticated":false}' });
  });
  await context.addInitScript(() => {
    window.textCompareClipboard = [];
    window.ClipboardItem = class { constructor(parts) { this.parts = parts; } };
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      write: async items => {
        const value = {};
        for (const [type, blob] of Object.entries(items[0].parts)) value[type] = await blob.text();
        window.textCompareClipboard.push(value);
      },
      writeText: async text => { window.textCompareClipboard.push({ 'text/plain': text }); }
    } });
  });
  try {
    await page.goto(`${base}/tools/text-compare`, { waitUntil: 'domcontentloaded' });
    await page.locator('#textcompare-original').waitFor();
    await page.locator('#pcz-reject').click();
    await page.locator('#pcz-banner').waitFor({ state: 'hidden' });
    await page.evaluate(() => document.fonts.ready);
    const original = page.locator('#textcompare-original');
    const revised = page.locator('#textcompare-revised');
    const output = page.locator('#textcompare-output');
    const copy = page.locator('#textcompare-copy');
    const options = page.locator('.textcompare-settings');
    const compare = page.locator('#textcompare-form button[type="submit"]');
    const waitResult = () => page.waitForFunction(() => !document.querySelector('#textcompare-copy').disabled);
    const copyAndCheck = async expected => {
      const count = await page.evaluate(() => textCompareClipboard.length);
      await copy.click();
      await page.waitForFunction(previous => textCompareClipboard.length > previous, count);
      const result = await page.evaluate(() => textCompareClipboard.at(-1));
      assert.equal(result['text/plain'], expected, `${stage}: copying exports the latest revised draft.`);
      assert(result['text/html'] && result['text/rtf'], `${stage}: formatted HTML and RTF remain available.`);
      return result;
    };
    assert.equal(await page.locator('#main [role="tab"]').count(), 0, 'Text Compare has no view tabs.');
    assert.equal(await page.locator('#main button').filter({ hasText: /^Paste$/ }).count(), 0, 'The redundant Paste buttons are gone.');
    assert.equal(await options.getAttribute('open'), null, 'Advanced controls start collapsed.');
    assert(await copy.isDisabled(), 'Empty comparison cannot be copied.');
    await assertLayout(page, width, stage);
    await page.screenshot({ path: path.join(artifactDir, `text-compare-simple-${width}-initial.png`), fullPage: width < 500 });

    stage = 'compare-and-edit';
    await original.fill('Publish the ORIGINALMARKER draft on Monday.');
    await revised.fill('Publish the FIRSTREVISION draft on Tuesday.');
    assert(await copy.isDisabled(), 'Typing both initial drafts still waits for the first explicit Compare.');
    await compare.click();
    await waitResult();
    assert((await output.innerText()).includes('FIRSTREVISION'), 'Compare generates the first real result.');
    await revised.fill('Publish the CURRENTREVISION draft on Friday.');
    const afterEdit = await page.evaluate(() => ({
      disabled: document.querySelector('#textcompare-copy').disabled,
      text: document.querySelector('#textcompare-output').textContent,
      focus: document.activeElement.id,
      y: window.scrollY,
      panel: document.querySelector('[data-site-frame-viewport]').scrollTop,
      selection: document.querySelector('#textcompare-revised').selectionStart
    }));
    assert(afterEdit.disabled && !afterEdit.text.includes('FIRSTREVISION'), 'An edit immediately invalidates stale output and Copy.');
    await waitResult();
    const afterRefresh = await page.evaluate(() => ({ focus: document.activeElement.id, y: window.scrollY, panel: document.querySelector('[data-site-frame-viewport]').scrollTop, selection: document.querySelector('#textcompare-revised').selectionStart }));
    assert.equal(afterRefresh.focus, 'textcompare-revised', 'Debounced refresh keeps focus in the revised editor.');
    assert.equal(afterRefresh.y, afterEdit.y, 'Debounced refresh does not scroll the page.');
    assert.equal(afterRefresh.panel, afterEdit.panel, 'Debounced refresh does not scroll the content panel.');
    assert.equal(afterRefresh.selection, afterEdit.selection, 'Debounced refresh preserves the editor caret.');
    const currentCopy = await copyAndCheck('Publish the CURRENTREVISION draft on Friday.');
    assert(currentCopy['text/html'].includes('CURRENTREVISION') && !currentCopy['text/html'].includes('FIRSTREVISION'), 'Formatted output also uses the current comparison.');
    await assertLayout(page, width, stage);

    stage = 'import-mode-swap-clear';
    await options.locator('summary').first().click();
    await page.locator('#textcompare-clear').click();
    for (const field of ['original', 'revised']) {
      const chooser = page.waitForEvent('filechooser');
      await page.locator(`#textcompare-${field}-import`).click();
      await (await chooser).setFiles({ name: `${field}.csv`, mimeType: 'text/csv', buffer: Buffer.from(`name,count\nalpha,${field === 'original' ? 2 : 7}`) });
      await page.waitForFunction(id => document.getElementById(`textcompare-${id}`).value.includes('alpha,'), field);
    }
    await page.locator('#textcompare-mode-structured').check();
    await compare.click();
    await waitResult();
    await copyAndCheck('name,count\nalpha,7');
    await page.locator('#textcompare-swap').click();
    await waitResult();
    assert.equal(await original.inputValue(), 'name,count\nalpha,7');
    assert.equal(await revised.inputValue(), 'name,count\nalpha,2');
    await copyAndCheck('name,count\nalpha,2');
    await revised.fill('This pending edit will be cleared.');
    await page.locator('#textcompare-clear').click();
    assert.equal(await original.inputValue(), '');
    assert.equal(await revised.inputValue(), '');
    assert(await copy.isDisabled(), 'Clear cancels pending edits and leaves Copy unavailable.');
    await assertLayout(page, width, stage);

    stage = 'example';
    await options.locator('summary').first().click();
    await page.locator('#textcompare-example').click();
    await waitResult();
    const exampleDraft = await revised.inputValue();
    assert(exampleDraft.length > 100 && (await output.locator('ins,del').count()) > 0, 'The example runs immediately and shows actual changes.');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'textcompare-example', 'Loading the example keeps focus on its button.');
    assert.equal(await original.evaluate(node => node.scrollTop), 0, 'The original example opens at its beginning.');
    assert.equal(await revised.evaluate(node => node.scrollTop), 0, 'The revised example opens at its beginning.');
    await copyAndCheck(exampleDraft);
    await page.screenshot({ path: path.join(artifactDir, `text-compare-simple-${width}-compared.png`), fullPage: width < 500 });

    stage = 'legacy-restore';
    for (const view of ['drafts', 'comparison']) {
      const immediate = await page.evaluate(savedView => {
        // The account service restores fields before sending tools:session-applied.
        document.querySelector('#textcompare-original').value = 'Restored original draft.';
        const revisedInput = document.querySelector('#textcompare-revised');
        revisedInput.value = `Restored ${savedView} revised draft.`;
        revisedInput.focus();
        document.dispatchEvent(new CustomEvent('tools:session-applied', { detail: { toolId: 'text-compare', snapshot: { inputs: { view: savedView }, output: { kind: 'html', html: '<p>OLDPREVIEW</p>', summary: 'Saved comparison' } } } }));
        return document.querySelector('#textcompare-copy').disabled;
      }, view);
      assert(immediate, 'Legacy previews wait for restored copyable runs.');
      await waitResult();
      assert.equal(await page.evaluate(() => document.activeElement.id), 'textcompare-revised', 'Restoring an old view does not steal editor focus.');
      await copyAndCheck(`Restored ${view} revised draft.`);
      await assertLayout(page, width, `legacy ${view}`);
    }
    assert.deepEqual(accountRequests, [], 'No real account or save request is required for these workflows.');
    assert.deepEqual(errors, [], 'The simplified workspace produces no runtime exceptions.');
    console.log(`Text Compare simple workspace passed at ${width}px: compare/edit/copy, import/mode/swap/clear, example, legacy restore, and layout.`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `text-compare-simple-${width}-failure.png`), fullPage: true }).catch(() => {});
    error.message = `${width}px ${stage}: ${error.message}`;
    throw error;
  } finally { await context.close(); }
}

async function runTextCompareSimpleChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const width of [1440, 390, 320]) await runCase(options, width);
}

module.exports = runTextCompareSimpleChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'text-compare-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runTextCompareSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-text-compare-simple') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
