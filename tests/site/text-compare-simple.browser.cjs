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
    const NativeWorker = window.Worker;
    window.textCompareWorkerStats = { requests: 0, terminated: 0, held: 0 };
    window.textCompareHoldWorker = false;
    window.Worker = class extends NativeWorker {
      constructor(url, options) { super(url, options); this.isTextCompare = String(url).includes('text-compare-worker.js'); }
      postMessage(payload) {
        if (this.isTextCompare) {
          window.textCompareWorkerStats.requests += 1;
          if (window.textCompareHoldWorker) { window.textCompareWorkerStats.held += 1; return; }
        }
        super.postMessage(payload);
      }
      terminate() {
        if (this.isTextCompare) window.textCompareWorkerStats.terminated += 1;
        super.terminate();
      }
    };
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
      assert((await page.locator('#textcompare-copy-status').innerText()).startsWith('Copied with formatting'), `${stage}: copy confirmation is visible.`);
      return result;
    };
    assert.equal(await page.locator('#main [role="tab"]').count(), 0, 'Text Compare has no view tabs.');
    assert.equal(await page.locator('#main button').filter({ hasText: /^Paste$/ }).count(), 0, 'The redundant Paste buttons are gone.');
    assert.equal(await page.locator('#main input[type="file"]').count(), 0, 'Import file controls are removed.');
    assert.equal(await page.locator('#main button').filter({ hasText: /^(Import|Try an example)$/ }).count(), 0, 'Import and explicit example buttons are removed.');
    assert.equal(await page.locator('label[for="textcompare-original"]').innerText(), 'Original');
    assert.equal(await page.locator('label[for="textcompare-revised"]').innerText(), 'Revised');
    assert.equal(await page.locator('#textcompare-result-title').innerText(), 'Comparison result');
    assert.equal(await copy.innerText(), 'Copy result');
    assert.equal(await compare.innerText(), 'Compare');
    assert.equal(await original.inputValue(), '');
    assert.equal(await revised.inputValue(), '');
    assert((await original.getAttribute('placeholder')).includes('Product analytics') &&
      (await revised.getAttribute('placeholder')).includes('Product analytics'), 'The default example is visible without becoming user input.');
    assert.equal(await options.getAttribute('open'), null, 'Advanced controls start collapsed.');
    assert(await page.locator('#textcompare-clear').isVisible(), 'Clear remains available without opening advanced options.');
    assert((await page.locator('.textcompare-saving-note').innerText()).includes('No account is needed'), 'Optional saving is distinguished from immediate local use.');
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
    const comparedLayout = await page.evaluate(() => {
      const heading = document.querySelector('#textcompare-result-title').getBoundingClientRect();
      const copyButton = document.querySelector('#textcompare-copy').getBoundingClientRect();
      const outputBox = document.querySelector('#textcompare-output').getBoundingClientRect();
      const masthead = document.querySelector('[data-mobile-site-masthead]')?.getBoundingClientRect();
      return { headingY: heading.y, copyY: copyButton.y, copyRight: copyButton.right, copyBottom: copyButton.bottom, outputRight: outputBox.right, outputTop: outputBox.top, mastheadBottom: Math.max(0, masthead?.bottom || 0),
        viewportBottom: innerHeight, mobileDockCount: document.querySelectorAll('[data-mobile-section-nav]').length };
    });
    assert(comparedLayout.copyY >= comparedLayout.headingY - 14 && comparedLayout.copyBottom <= comparedLayout.outputTop,
      `${stage}: Copy result remains in the heading area, wrapping cleanly on narrow screens.`);
    if (width >= 500) assert(Math.abs(comparedLayout.copyRight - comparedLayout.outputRight) <= 1,
      `${stage}: Copy result sits at the right edge of the desktop result area.`);
    if (width < 500) {
      assert.equal(comparedLayout.mobileDockCount, 0, `${stage}: the editor has no bottom section dock.`);
      assert(comparedLayout.outputTop < comparedLayout.viewportBottom - 44,
        `${stage}: explicit Compare brings the result into the mobile viewport: ${JSON.stringify(comparedLayout)}`);
      assert(comparedLayout.headingY >= comparedLayout.mastheadBottom - 1, `${stage}: the result heading is not hidden behind the mobile masthead.`);
    }
    if (width < 500) await page.screenshot({ path: path.join(artifactDir, `text-compare-simple-${width}-results-viewport.png`) });
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
    await page.locator('#textcompare-return').focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'textcompare-original', 'Keyboard activation of Return to inputs focuses Original.');
    assert.equal(await original.inputValue(), 'Publish the ORIGINALMARKER draft on Monday.', 'Returning preserves the original draft.');
    assert.equal(await revised.inputValue(), 'Publish the CURRENTREVISION draft on Friday.', 'Returning preserves the revised draft.');
    assert((await output.innerText()).includes('CURRENTREVISION'), 'Returning preserves the current result.');

    stage = 'structured-mode-swap-clear';
    await options.locator('summary').first().click();
    await page.locator('#textcompare-clear').click();
    await original.fill('name,count\nalpha,2');
    await revised.fill('name,count\nalpha,7');
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

    stage = 'clear-in-flight-worker';
    await original.fill('An original draft that will be cancelled.');
    await revised.fill('A revised draft that will be cancelled.');
    await page.evaluate(() => { window.textCompareHoldWorker = true; });
    const workerBefore = await page.evaluate(() => ({ ...textCompareWorkerStats }));
    await compare.click();
    await page.waitForFunction(previous => textCompareWorkerStats.held > previous, workerBefore.held);
    assert.equal(await page.locator('#textcompare-view-comparison').getAttribute('aria-busy'), 'true', 'Pending work has an accessible busy state.');
    await page.locator('#textcompare-clear').click();
    const workerAfter = await page.evaluate(() => ({ ...textCompareWorkerStats }));
    assert.equal(workerAfter.terminated, workerBefore.terminated + 1, 'Clear terminates the pending worker.');
    assert.equal(await page.locator('#textcompare-view-comparison').getAttribute('aria-busy'), 'false', 'Clear removes the pending state.');
    assert(await copy.isDisabled() && (await output.innerText()).includes('Compare the example'), 'Cancelled work cannot leave a stale result available.');
    await page.evaluate(() => { window.textCompareHoldWorker = false; });

    stage = 'default-example';
    await page.locator('#textcompare-mode-auto').check();
    await options.locator('summary').first().click();
    await compare.click();
    await waitResult();
    const exampleDraft = await revised.getAttribute('placeholder');
    assert(exampleDraft.includes('Friday') && (await output.locator('ins,del').count()) > 0, 'Compare runs the default example and shows actual changes.');
    assert.equal(await original.inputValue(), '', 'The default comparison does not populate Before.');
    assert.equal(await revised.inputValue(), '', 'The default comparison does not populate After.');
    assert(await compare.evaluate(node => document.activeElement === node), 'Running the example keeps focus on Compare.');
    await copyAndCheck(exampleDraft);
    await page.screenshot({ path: path.join(artifactDir, 'text-compare-simple-' + width + '-compared.png'), fullPage: width < 500 });

    stage = 'one-sided-comparisons';
    await original.fill('ONLY_BEFORE_TEXT');
    assert(!(await original.getAttribute('placeholder')).includes('Product analytics') &&
      !(await revised.getAttribute('placeholder')).includes('Product analytics'), 'Typing on either side immediately removes both example previews.');
    assert(!(await output.innerText()).includes('Product analytics'), 'Typing immediately removes the old example result.');
    await waitResult();
    assert.equal(await output.locator('del').innerText(), 'ONLY_BEFORE_TEXT', 'Before-only text is shown as deleted.');
    assert.equal(await output.locator('ins').count(), 0, 'An empty After side receives no example fallback.');
    await copyAndCheck('');
    await original.fill('');
    await revised.fill('ONLY_AFTER_TEXT');
    await waitResult();
    assert.equal(await output.locator('ins').innerText(), 'ONLY_AFTER_TEXT', 'After-only text is shown as inserted.');
    assert.equal(await output.locator('del').count(), 0, 'An empty Before side receives no example fallback.');
    await copyAndCheck('ONLY_AFTER_TEXT');

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
    stage = 'long-worker-diff';
    const longOriginal = Array.from({ length: 1000 }, (_, index) => `Line ${index}: this paragraph stays available for editing.`).join('\n');
    const longRevised = longOriginal.replace('Line 500:', 'Updated line 500:');
    const longRequestCount = await page.evaluate(() => textCompareWorkerStats.requests);
    await page.locator('#textcompare-clear').click();
    await original.fill(longOriginal);
    await revised.fill(longRevised);
    await compare.click();
    await waitResult();
    assert((await output.innerText()).includes('Updated line 500:'), 'Long text still produces the actual change.');
    assert(await page.evaluate(previous => textCompareWorkerStats.requests > previous, longRequestCount), 'Long text is compared through the background worker.');
    await copyAndCheck(longRevised);
    await assertLayout(page, width, stage);
    if (width < 500) {
      stage = 'content-sized-mobile-editors';
      const editorState = locator => locator.evaluate(node => ({
        height: node.getBoundingClientRect().height,
        scrollHeight: node.scrollHeight,
        clientHeight: node.clientHeight,
        focused: node === document.activeElement,
        selection: node.selectionStart
      }));
      await original.fill('A short draft.');
      await revised.fill('An updated short draft.');
      const compact = await editorState(original);
      assert(compact.height >= 90 && compact.height <= 120, 'Short mobile drafts use about three readable lines, not a tall fixed editor.');
      const longDraft = Array.from({ length: 80 }, (_, index) => `Line ${index}: keep this long draft editable.`).join('\n');
      await original.fill(longDraft);
      const expanded = await editorState(original);
      assert(expanded.height > compact.height + 80 && expanded.height <= 361, 'Long drafts expand within a bounded mobile editor.');
      assert(expanded.scrollHeight > expanded.clientHeight, 'Long drafts remain scrollable rather than stretching the whole page.');
      await original.fill('A short replacement.');
      const activeShort = await editorState(original);
      assert.equal(activeShort.height, expanded.height, 'Deleting content does not collapse the editor under an active caret.');
      assert(activeShort.focused && activeShort.selection === 'A short replacement.'.length, 'Resizing preserves focus and the caret.');
      await revised.focus();
      assert((await editorState(original)).height <= compact.height + 1, 'Leaving a shortened draft reclaims its unused editor space.');
      await page.evaluate(value => {
        document.querySelector('#textcompare-original').value = value;
        document.dispatchEvent(new CustomEvent('tools:session-applied', { detail: { toolId: 'text-compare', snapshot: {} } }));
      }, longDraft);
      assert((await editorState(original)).height > compact.height + 80, 'Restored account, shared, and guest inputs are sized without requiring typing.');
      await options.locator('summary').first().click();
      await page.locator('#textcompare-swap').click();
      assert((await editorState(original)).height <= compact.height + 1 && (await editorState(revised)).height > compact.height + 80, 'Swapping drafts also swaps the space each editor needs.');
      await page.locator('#textcompare-clear').click();
      assert((await editorState(original)).height < expanded.height && (await editorState(revised)).height < expanded.height, 'Clear restores compact example previews on both sides.');
      assert.equal(await original.inputValue(), '');
      assert.equal(await revised.inputValue(), '');
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.waitForFunction(() => ['original', 'revised'].every(id => !document.querySelector(`#textcompare-${id}`).style.height));
      await assertLayout(page, 1440, 'mobile-to-desktop resize');
      assert((await editorState(original)).height >= 140, 'Desktop keeps its existing spacious editor layout.');
      await page.setViewportSize({ width, height: 844 });
      await page.waitForFunction(() => document.querySelector('#textcompare-original').style.height);
      await assertLayout(page, width, 'desktop-to-mobile resize');
    }

    stage = 'guest-draft-recovery';
    await page.locator('#textcompare-clear').click();
    await original.fill('GUEST_ORIGINAL: this unfinished draft stays local.');
    await revised.fill('GUEST_REVISED: this unfinished draft stays local.');
    await page.waitForFunction(() => JSON.stringify(window.SiteSessionDrafts?.read('tools:text-compare') || {}).includes('GUEST_REVISED'));
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#textcompare-revised')?.value.includes('GUEST_REVISED'));
    assert.equal(await original.inputValue(), 'GUEST_ORIGINAL: this unfinished draft stays local.', 'Guest recovery restores the editable original.');
    assert.equal(await revised.inputValue(), 'GUEST_REVISED: this unfinished draft stays local.', 'Guest recovery restores the editable revision.');
    assert(await copy.isDisabled(), 'Guest field recovery does not invent an old comparison result.');
    await compare.click();
    await waitResult();
    await copyAndCheck('GUEST_REVISED: this unfinished draft stays local.');
    await page.locator('#textcompare-clear').click();
    await page.waitForFunction(() => !window.SiteSessionDrafts?.read('tools:text-compare'));
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'guest');
    assert.equal(await original.inputValue(), '', 'Clear does not resurrect a guest original on reload.');
    assert.equal(await revised.inputValue(), '', 'Clear does not resurrect a guest revision on reload.');
    assert.deepEqual(accountRequests, [], 'No real account or save request is required for these workflows.');
    assert.deepEqual(errors, [], 'The simplified workspace produces no runtime exceptions.');
    console.log(`Text Compare workspace passed at ${width}px: compare/edit/copy, result navigation, structured mode/swap/clear, worker cancellation/restart, examples, one-sided comparisons, legacy restore, long worker diff, and layout.`);
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
