/** Offline checks of the real Sentence Retriever wrapper and its embedded UI. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { expect } = require('@playwright/test');
const { createLocalServer } = require('../../build/dev');
const { isolateRequests } = require('../release/fixtures.cjs');

async function runSentenceRetrieverChecks({ browser, base, artifactDir, browserName = 'browser' }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  for (const width of [1440, 390]) {
    const label = `sentence-${browserName}-${width}`;
    const context = await browser.newContext({ viewport: { width, height: 900 }, serviceWorkers: 'block', reducedMotion: 'reduce' });
    await context.addInitScript(() => localStorage.setItem('pcz_consent_v1', JSON.stringify({ version: 1, timestamp: Date.now(),
      categories: { necessary: true, analytics: false, advertising: false, functional: false } })));
    await isolateRequests(context, base);
    let mode = 'success';
    let healthCalls = 0;
    const ranks = [];
    const held = [];
    await context.route(`${base}/api/demos/smart-sentence/**`, async route => {
      const operation = new URL(route.request().url()).pathname.split('/').at(-1);
      if (operation === 'health') {
        healthCalls += 1;
        return route.fulfill({ json: { configured: true, status: 'ok', ready: false, query_max_chars: 512, query_max_tokens: 128 } });
      }
      assert.equal(operation, 'rank', 'The UI uses only mocked health/rank operations.');
      ranks.push({ payload: route.request().postDataJSON(), requestId: route.request().headers()['x-request-id'] });
      if (mode === 'query-limit') return route.fulfill({ status: 400, json: { error: 'Query must contain 1 to 512 characters.' } });
      if (mode === 'token-limit') return route.fulfill({ status: 422, json: { detail: 'Query exceeds 128 model tokens. Please shorten it.' } });
      if (mode === 'timeout') return route.fulfill({ status: 504, json: { code: 'DEMO_TIMEOUT', error: 'The search took too long.' } });
      if (mode === 'hold') await new Promise(resolve => held.push(resolve));
      return route.fulfill({ json: { top: [
        { sentence: 'The White Rabbit hurried down the passage.', score: 0.825 },
        { sentence: 'Alice wondered what might happen next.', score: 0.61 }
      ] } }).catch(() => {});
    });
    const page = await context.newPage();
    await page.clock.install();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    try {
      await page.goto(base + '/sentence-demo', { waitUntil: 'domcontentloaded' });
      const frame = await (await page.locator('.project-demo-wrapper-iframe').elementHandle()).contentFrame();
      const query = frame.locator('#query');
      const search = frame.locator('[name="submitBtn"]');
      const cancel = frame.locator('#cancel-search');
      const feedback = frame.locator('#search-feedback');
      const output = frame.locator('#results');
      await expect(search).toBeEnabled();
      await expect(frame.locator('#health-text')).toContainText('first search loads the model');
      assert(healthCalls >= 1 && ranks.length === 0, `${label}: cheap health enables Search without performing inference.`);
      await expect(query).toHaveAttribute('maxlength', '512');
      await expect(frame.locator('#query-limit')).toHaveText('Use up to 512 characters. Ideas longer than 128 model tokens need to be shortened.');
      // A malformed restored/programmatic value can bypass the native maxlength.
      const oversizedDraft = 'x'.repeat(513);
      await query.evaluate((input, value) => { input.value = value; }, oversizedDraft);
      await search.click();
      await expect(feedback).toContainText('Use 512 characters or fewer.');
      await expect(feedback).toContainText('shorten your idea');
      assert.equal(ranks.length, 0, `${label}: over-limit validation performs no rank request.`);
      await expect(query).toHaveValue(oversizedDraft);
      await expect(query).toBeFocused();
      await expect(search).toBeEnabled();
      await expect(cancel).toBeHidden();
      await expect(output).not.toHaveAttribute('aria-busy', 'true');

      const boundaryDraft = 'a'.repeat(512);
      await query.fill(boundaryDraft);
      await query.press('End');
      await query.press('a');
      await expect(query).toHaveValue(boundaryDraft);
      await search.click();
      await expect(output.locator('.sentence').first()).toHaveText('The White Rabbit hurried down the passage.');
      assert.equal(ranks.length, 1, `${label}: the exact 512-character boundary makes one request.`);
      assert.deepEqual(ranks[0].payload, { query: boundaryDraft, top: 5 });
      await expect(query).toHaveValue(boundaryDraft);
      await expect(feedback).toBeEmpty();

      for (const validation of [
        { mode: 'query-limit', draft: 'Keep the idea after a server validation error.', message: 'Query must contain 1 to 512 characters.' },
        { mode: 'token-limit', draft: Array(128).fill('z').join(' '), message: 'Query exceeds 128 model tokens. Please shorten it.' }
      ]) {
        mode = validation.mode;
        const previousRequests = ranks.length;
        await query.fill(validation.draft);
        await search.click();
        const visibleValidation = feedback.locator('.item > .small');
        await expect(visibleValidation).toHaveText(validation.message);
        await expect(visibleValidation).toBeVisible();
        await expect(feedback.getByRole('button', { name: 'Reconnect demo', exact: true })).toHaveCount(0);
        await expect(frame.locator('#health')).toHaveText('AWS · Connected');
        await expect(query).toHaveValue(validation.draft);
        await expect(query).toBeFocused();
        await expect(output.locator('.sentence').first()).toHaveText('The White Rabbit hurried down the passage.');
        await expect(search).toBeEnabled();
        await expect(cancel).toBeHidden();
        await expect(output).toHaveAttribute('aria-busy', 'false');
        await page.clock.runFor(10000);
        assert.equal(ranks.length, previousRequests + 1, `${label}: ${mode} rejection preserves the draft without automatic replay.`);
        await page.screenshot({ path: path.join(artifactDir, `${label}-${mode}.png`) });
      }

      mode = 'success';
      await frame.locator('[data-query-example="A rabbit is in a hurry."]').click();
      await frame.getByText('Advanced settings', { exact: true }).click();
      await frame.locator('#top').focus();
      await frame.locator('#top').press('ArrowRight');
      await frame.locator('#top').press('ArrowRight');
      await expect(frame.locator('#top-value')).toHaveText('7');
      await search.click();
      await expect(output.locator('.sentence').first()).toHaveText('The White Rabbit hurried down the passage.');
      await expect(output.locator('.percent').first()).toHaveText('Similarity 82.5%');
      assert.equal(ranks.length, 4, `${label}: one successful example click makes one new rank request.`);
      assert.deepEqual(ranks[3].payload, { query: 'A rabbit is in a hurry.', top: 7 });
      assert.match(ranks[3].requestId, /^[a-f0-9-]{36}$/i, `${label}: inference receives an opaque request ID.`);
      await expect(cancel).toBeHidden();
      await expect(output).toHaveAttribute('aria-busy', 'false');
      await page.screenshot({ path: path.join(artifactDir, `${label}-success.png`) });

      mode = 'timeout';
      await search.click();
      await expect(feedback).toContainText('search took too long');
      await expect(search).toBeEnabled();
      await expect(query).toHaveValue('A rabbit is in a hurry.');
      await expect(output.locator('.sentence').first()).toHaveText('The White Rabbit hurried down the passage.');
      await page.clock.runFor(10000);
      assert.equal(ranks.length, 5, `${label}: HTTP 504 never automatically repeats inference.`);
      await page.screenshot({ path: path.join(artifactDir, `${label}-http504.png`) });

      mode = 'hold';
      const draft = 'An unfinished idea that must survive cancellation.';
      await query.fill(draft);
      await search.click();
      await expect.poll(() => ranks.length).toBe(6);
      await expect(cancel).toBeVisible();
      await expect(search).toBeDisabled();
      await expect(output).toHaveAttribute('aria-busy', 'true');
      await cancel.click();
      await expect(feedback).toContainText('Search cancelled');
      await expect(query).toHaveValue(draft);
      await expect(search).toBeEnabled();
      await expect(cancel).toBeHidden();
      await expect(output).toHaveAttribute('aria-busy', 'false');
      held.splice(0).forEach(resolve => resolve());
      await page.screenshot({ path: path.join(artifactDir, `${label}-cancelled.png`) });

      // Reset the idea through its editable control; no destructive reset button
      // is required. Blank input intentionally uses the displayed example.
      mode = 'success';
      await query.fill('');
      await search.click();
      await expect(output.locator('.sentence').first()).toHaveText('The White Rabbit hurried down the passage.');
      await expect(feedback).toBeEmpty();
      await expect(search).toBeEnabled();
      assert.equal(ranks.length, 7, `${label}: reset and manual rerun make exactly one new request.`);
      assert.equal(ranks.at(-1).payload.query, 'She wonders about things.');

      mode = 'hold';
      await query.fill('Keep this text after the client deadline.');
      await search.click();
      await expect.poll(() => ranks.length).toBe(8);
      await page.clock.fastForward(130001);
      await expect(feedback).toContainText('search took too long');
      await expect(query).toHaveValue('Keep this text after the client deadline.');
      await expect(search).toBeEnabled();
      await expect(cancel).toBeHidden();
      await expect(output).toHaveAttribute('aria-busy', 'false');
      assert.equal(ranks.length, 8, `${label}: the client deadline cancels rather than replaying rank.`);
      held.splice(0).forEach(resolve => resolve());
      mode = 'success';
      await search.click();
      await expect(feedback).toBeEmpty();
      await expect(search).toBeEnabled();
      assert.equal(ranks.length, 9, `${label}: a manual retry works after the client deadline.`);
      assert.deepEqual(errors, [], `${label}: no uncaught errors.`);
      results.push({ label, healthCalls, ranks });
      console.log(`Sentence Retriever offline flow passed: ${label}`);
    } catch (error) {
      await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
      throw error;
    } finally {
      held.splice(0).forEach(resolve => resolve());
      await context.close();
    }
  }
  fs.writeFileSync(path.join(artifactDir, `sentence-${browserName}.json`), JSON.stringify(results, null, 2));
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sentence-browser-env-'));
  const server = createLocalServer({ envDir });
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runSentenceRetrieverChecks({ browser, browserName, base,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-sentence-browser') });
      } finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runSentenceRetrieverChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
