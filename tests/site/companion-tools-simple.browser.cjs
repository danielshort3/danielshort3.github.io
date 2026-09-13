/** Companion text tools: persistent results, optional controls, and safe imports. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const tools = [
  { slug: 'word-frequency', prefix: 'wordfreq', input: 'wordfreq-text', output: 'wordfreq-results-terms' },
  { slug: 'point-of-view-checker', prefix: 'povcheck', input: 'povcheck-text', output: 'povcheck-results-highlights' },
  { slug: 'nbsp-cleaner', prefix: 'nbsp', input: 'nbsp-input', output: 'nbsp-output' }
];

async function assertLayout(page, tool, width) {
  const layout = await page.evaluate(({ input, output }) => {
    const boxes = [input, output].map(id => {
      const node = document.getElementById(id);
      const box = node.getBoundingClientRect();
      return { width: box.width, height: box.height, left: box.left, right: box.right, hidden: !!node.closest('[hidden]') };
    });
    return { boxes, overflow: document.documentElement.scrollWidth - innerWidth };
  }, tool);
  assert(layout.overflow <= 1, `${tool.slug}: no horizontal page overflow at ${width}px`);
  for (const box of layout.boxes) assert(!box.hidden && box.width > 0 && box.height > 0 && box.left >= -1 && box.right <= width + 1, `${tool.slug}: core input and output remain laid out at ${width}px`);
  if (tool.prefix === 'nbsp' && width <= 760) {
    const actions = await page.evaluate(() => {
      const primary = document.querySelector('#nbsp-form button[type="submit"]');
      const share = document.querySelector('#nbsp-form [data-tool-share-link]');
      const text = document.createRange();
      text.selectNodeContents(primary);
      return {
        textHeight: text.getBoundingClientRect().height,
        lineHeight: parseFloat(getComputedStyle(primary).lineHeight),
        primaryBottom: primary.getBoundingClientRect().bottom,
        shareTop: share.getBoundingClientRect().top
      };
    });
    assert(actions.textHeight <= actions.lineHeight * 1.1, 'Detect & fix remains on one line on mobile');
    assert(actions.shareTop >= actions.primaryBottom, 'Sharing has its own row below the primary mobile action');
  }
}

async function delayedImportCheck(page, tool, action) {
  await page.evaluate(() => {
    const original = File.prototype.text;
    File.prototype.text = function () {
      if (this.name !== 'slow-import.txt') return original.call(this);
      return new Promise(resolve => { window.finishCompanionImport = () => resolve('Late imported text'); });
    };
  });
  await page.locator(`#${tool.prefix}-file`).setInputFiles({ name: 'slow-import.txt', mimeType: 'text/plain', buffer: Buffer.from('Late imported text') });
  await page.waitForFunction(() => typeof window.finishCompanionImport === 'function');
  const input = page.locator(`#${tool.input}`);
  if (action === 'clear') await page.locator(`#${tool.prefix}-clear`).click();
  else if (action === 'restore') await page.evaluate(({ input, slug }) => {
    document.getElementById(input).value = 'Newer restored text';
    document.dispatchEvent(new CustomEvent('tools:session-applied', { detail: { toolId: slug } }));
  }, tool);
  else await input.fill('Newer user text');
  await page.evaluate(() => window.finishCompanionImport());
  await page.waitForTimeout(80);
  const expected = action === 'clear' ? '' : action === 'restore' ? 'Newer restored text' : 'Newer user text';
  assert.equal(await input.inputValue(), expected, `${tool.slug}: a late import must not overwrite ${action}`);
  assert(await page.locator(`#${tool.prefix}-import`).isEnabled(), `${tool.slug}: canceled import releases its controls`);
  assert(!/Importing/i.test(await page.locator(`#${tool.prefix}-input-status`).innerText()), `${tool.slug}: canceled import clears its loading message`);
  await page.evaluate(() => { delete window.finishCompanionImport; });
}

async function runCase({ browser, base, artifactDir }, tool, width) {
  const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1000 }, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route('**/api/tools/**', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{"authenticated":false}' }));
  await context.addInitScript(() => {
    window.companionClipboard = [];
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async text => { window.companionClipboard.push(text); },
      readText: async () => { throw new Error('Tests must not read the clipboard.'); }
    } });
  });
  let stage = 'initial';
  try {
    await page.goto(`${base}/tools/${tool.slug}`, { waitUntil: 'domcontentloaded' });
    const input = page.locator(`#${tool.input}`);
    await input.waitFor();
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await page.locator('#main [role="tab"]').count(), 0, 'Input and output no longer require tabs');
    assert.equal(await page.locator(`#${tool.prefix}-paste`).count(), 0, 'No dedicated clipboard-read control');
    assert.equal(await page.locator(`#${tool.prefix}-file`).getAttribute('tabindex'), '-1', 'Hidden file input is not an extra keyboard stop');
    await assertLayout(page, tool, width);
    await page.screenshot({ path: path.join(artifactDir, `${tool.slug}-${width}-initial.png`) });
    const submit = page.locator(`#${tool.prefix}-form button[type="submit"]`);
    if (tool.prefix !== 'nbsp') {
      await page.locator(`#${tool.prefix}-example`).click();
      const example = await input.evaluate(node => ({ text: node.value, scroll: node.scrollTop, caret: node.selectionStart, active: document.activeElement.id }));
      assert(example.text.length > 40, 'Example supplies useful text');
      assert.equal(example.scroll, 0, 'Example begins at the top');
      assert.equal(example.caret, 0, 'Example caret starts at the beginning');
      assert.equal(example.active, `${tool.prefix}-example`, 'Example leaves keyboard focus on its action');
    }

    stage = 'analysis and optional details';
    if (tool.prefix === 'wordfreq') {
      await input.fill('Handoff handoff handoff. Support support.');
      await submit.click();
      const terms = page.locator('.wordfreq-term-btn');
      await terms.first().click();
      assert.equal(await page.locator('#wordfreq-results-occurrences').getAttribute('open'), '');
      assert(await page.locator('#wordfreq-results-terms').isVisible(), 'Inspecting keeps the table visible');
      assert.match(await page.locator('#wordfreq-occurrence-summary').innerText(), /3 matches/);
      await terms.nth(1).click();
      assert.match(await page.locator('#wordfreq-occurrence-summary').innerText(), /2 matches/);
      assert(await terms.first().isVisible(), 'A second term can be selected without returning to another view');
      await page.locator('#wordfreq-options > summary').click();
      await page.locator('#wordfreq-score').selectOption('share');
      assert(await input.isVisible(), 'Changing options does not hide source text');
      await page.locator('#wordfreq-options > summary').click();
      await input.fill('Review review review.');
      assert(await page.locator('#wordfreq-copy').isDisabled(), 'Edited input cannot copy old results');
      assert(await page.locator('#wordfreq-export-csv').isDisabled(), 'Edited input cannot export old results');
      await submit.click();
      await page.locator('#wordfreq-copy').click();
      assert.match(await page.evaluate(() => window.companionClipboard.at(-1)), /review/i);
      await page.locator('.wordfreq-export-menu > summary').click();
      const exportBox = await page.locator('.wordfreq-export-menu > .tool-workspace-actions').boundingBox();
      assert(exportBox.x >= 0 && exportBox.x + exportBox.width <= width, 'Export choices fit inside the viewport');
      const downloadPromise = page.waitForEvent('download');
      await page.locator('#wordfreq-export-csv').click();
      const download = await downloadPromise;
      assert.match(download.suggestedFilename(), /\.csv$/i);
      assert(fs.statSync(await download.path()).size > 10, 'CSV export retains results');
      if (await page.locator('.wordfreq-export-menu').getAttribute('open') !== null) await page.locator('.wordfreq-export-menu > summary').click();
    } else if (tool.prefix === 'povcheck') {
      await input.fill('I reviewed the draft. You can share it. They will review the final version.');
      await submit.click();
      assert.equal(await page.locator('#povcheck-first-count').innerText(), '1');
      assert.equal(await page.locator('#povcheck-second-count').innerText(), '1');
      await page.locator('#povcheck-results-drift > summary').click();
      await page.locator('#povcheck-drift-list button').first().click();
      assert(await page.locator('#povcheck-output').isVisible(), 'Sentence inspection keeps highlighted output visible');
      await page.locator('#povcheck-results-tokens > summary').click();
      await page.locator('#povcheck-first-list button').first().click();
      assert(await page.locator('#povcheck-output').isVisible(), 'Token inspection keeps highlighted output visible');
      await page.locator('#povcheck-options > summary').click();
      await page.locator('label[for="povcheck-mode-advanced"]').click();
      assert(await page.locator('#povcheck-mode-advanced').isChecked());
      await page.locator('#povcheck-include-it').uncheck();
      assert(await input.isVisible(), 'Custom rules keep source text visible');
      await page.locator('#povcheck-options > summary').click();
      await input.fill('We reviewed the draft.');
      assert(await page.locator('#povcheck-copy-results').isDisabled());
      assert.match(await page.locator('#povcheck-summary').innerText(), /Text changed/);
      await submit.click();
      await page.locator('.povcheck-export-menu > summary').click();
      await page.locator('#povcheck-copy-results').click();
      assert.match(await page.evaluate(() => window.companionClipboard.at(-1)), /first person|first-person/i);
      if (await page.locator('.povcheck-export-menu').getAttribute('open') !== null) await page.locator('.povcheck-export-menu > summary').click();
      await page.locator('#povcheck-results-drift > summary').click();
      await page.locator('#povcheck-results-tokens > summary').click();
    } else {
      await input.fill('A\u00a0B café');
      await submit.click();
      assert.equal(await page.locator('#nbsp-output').inputValue(), 'A B café');
      assert.match(await page.locator('#nbsp-summary').innerText(), /1 hard space replaced/);
      assert(!/1 hard spaces/.test(await page.locator('#nbsp-summary').innerText()), 'A single hard space uses singular wording');
      await page.locator('.nbsp-character-details > summary').click();
      assert(await page.locator('#nbsp-preview').isVisible());
      assert(await page.locator('#nbsp-output').isVisible(), 'Markers do not replace cleaned text');
      await page.locator('#nbsp-copy').click();
      assert.equal(await page.evaluate(() => window.companionClipboard.at(-1)), 'A B café');
      await page.locator('#nbsp-strip-nonascii').check();
      assert.equal(await page.locator('#nbsp-output').inputValue(), 'A B caf');
      await input.fill('New text');
      assert(await page.locator('#nbsp-copy').isDisabled());
      assert.equal(await page.locator('#nbsp-output').inputValue(), '');
      await submit.click();
      await page.locator('.nbsp-character-details > summary').click();
    }

    stage = 'rendered result';
    await assertLayout(page, tool, width);
    await input.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifactDir, `${tool.slug}-${width}-result.png`) });
    stage = 'successful import';
    const imported = 'First imported line.\n' + 'More text for a long import.\n'.repeat(50);
    await page.locator(`#${tool.prefix}-file`).setInputFiles({ name: 'normal-import.txt', mimeType: 'text/plain', buffer: Buffer.from(imported) });
    await page.waitForFunction(id => document.getElementById(id).value.startsWith('First imported line.'), tool.input);
    const importedPosition = await input.evaluate(node => ({ scroll: node.scrollTop, caret: node.selectionStart }));
    assert.equal(importedPosition.scroll, 0, 'Imported text starts at the top');
    assert.equal(importedPosition.caret, 0, 'Imported text caret starts at the beginning');
    stage = 'delayed import cancellation';
    await delayedImportCheck(page, tool, 'clear');
    await delayedImportCheck(page, tool, 'typing');
    await delayedImportCheck(page, tool, 'restore');
    assert.deepEqual(errors, [], `${tool.slug} has no browser errors`);
    console.log(`Companion tool passed: ${tool.slug} at ${width}px; inspect, options, current output, safe imports.`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `${tool.slug}-${width}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${tool.slug} ${width}px ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runCompanionToolsSimpleChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const tool of tools) for (const width of [1440, 390, 320]) await runCase(options, tool, width);
}

module.exports = runCompanionToolsSimpleChecks;

if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'companion-tools-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runCompanionToolsSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-companion-tools') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
