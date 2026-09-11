/** Project evidence, contextual navigation, and contact drafts; never sends a message. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function settle(page) {
  await page.waitForFunction(() => {
    const frame = window.SiteFrame?.root();
    return document.readyState !== 'loading' && !window.SiteNavigation?.isNavigating?.()
      && (!frame || !frame.matches('.site-frame--moving, .site-frame--held'));
  });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function checkViewport({ browser, base, artifactDir }, viewport) {
  const name = viewport.width < 600 ? 'mobile' : 'desktop';
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  let submissions = 0;
  let stage = 'project load';
  page.on('pageerror', error => errors.push(error.message));
  await context.route('**/api/contact', async route => {
    submissions += 1;
    await route.abort();
  });
  // These checks exercise project content, not paid inference or embedded demos.
  await context.route(/\/(?:handwriting-rating-demo|shape-demo)(?:\.html)?(?:\?.*)?$/, route => route.fulfill({
    status: 200, contentType: 'text/html', body: '<!doctype html><title>Local demo fixture</title><p>Interactive demo fixture</p>'
  }));
  try {
    await page.goto(`${base}/portfolio/handwritingRating`);
    await page.locator('.project-evidence-details > summary').waitFor();
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click();
    await settle(page);

    stage = 'closed evidence';
    const evidence = page.locator('.project-evidence');
    const details = page.locator('.project-evidence-details');
    const summary = details.locator('summary');
    assert.equal(await details.getAttribute('open'), null, 'Evidence must start closed');
    assert.equal(await page.locator('.project-evidence-content').isVisible(), false, 'Supporting evidence must not consume space until opened');
    await summary.scrollIntoViewIfNeeded();
    assert.match(await evidence.locator('.project-evidence-note').innerText(), /not a full handwriting benchmark/);
    const summaryBox = await summary.boundingBox();
    assert(summaryBox.height >= 44, 'Evidence disclosure must provide a 44px target');
    await page.screenshot({ path: path.join(artifactDir, `project-evidence-${name}-closed.png`) });

    stage = 'keyboard evidence disclosure';
    await summary.focus();
    await summary.press('Enter');
    await page.locator('.project-evidence-content').waitFor({ state: 'visible' });
    assert.equal(await details.getAttribute('open'), '');
    assert.match(await evidence.innerText(), /98\.952%/);
    assert.match(await evidence.innerText(), /75\.56%/);
    assert.match(await evidence.innerText(), /small and its size is not documented/);
    const typefaces = await page.evaluate(() => ({
      context: getComputedStyle(document.querySelector('.project-evidence-context dd')).fontFamily,
      metricContext: getComputedStyle(document.querySelector('.project-evidence-metrics dd > span')).fontFamily
    }));
    assert.equal(typefaces.metricContext, typefaces.context, 'Metric explanations should use the normal reading typeface');
    await summary.evaluate(node => node.scrollIntoView({ block: 'start' }));
    await settle(page);
    await page.screenshot({ path: path.join(artifactDir, `project-evidence-${name}-open.png`) });
    const overflow = await page.evaluate(() => ({
      document: document.documentElement.scrollWidth - innerWidth,
      content: document.querySelector('.project-evidence').scrollWidth - document.querySelector('.project-evidence').clientWidth
    }));
    assert(overflow.document <= 1 && overflow.content <= 1, `${name} evidence must wrap without horizontal overflow`);
    if (name === 'mobile') {
      await page.locator('.project-evidence-source').scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(artifactDir, `project-evidence-${name}-limitations.png`) });
    }
    await summary.focus();
    await summary.press('Space');
    assert.equal(await details.getAttribute('open'), null, 'Space must close the native disclosure');

    stage = 'project contact prefill';
    const question = page.locator('.project-question-link');
    assert.equal(await question.getAttribute('href'), '/contact', 'Contact must retain a native fallback route');
    await question.click();
    const modal = page.locator('#contact-modal.active');
    await modal.waitFor();
    const message = modal.locator('#contact-message');
    assert.match(await message.inputValue(), /^Hi Daniel, I have a question about Handwriting Legibility Scoring:/);
    const draft = 'Browser verification draft: keep this text. Do not send.';
    await message.fill(draft);
    await modal.getByRole('button', { name: 'Close dialog', exact: true }).click();
    await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });
    await question.click();
    await modal.waitFor();
    assert.equal(await message.inputValue(), draft, 'Reopening a project contact link must preserve the visitor draft');
    await page.screenshot({ path: path.join(artifactDir, `project-contact-${name}.png`) });
    await modal.getByRole('button', { name: 'Close dialog', exact: true }).click();
    await page.locator('#contact-modal.active').waitFor({ state: 'hidden' });

    stage = 'curated next project';
    const next = page.locator('.project-next-link');
    assert.equal(await next.count(), 1);
    assert.equal(await next.getAttribute('href'), '/portfolio/shapeClassifier');
    await next.click();
    await page.waitForURL(`${base}/portfolio/shapeClassifier`);
    await settle(page);
    assert.equal(await page.locator('h1:visible').innerText(), 'Shape Classifier Demo');

    if (name === 'desktop') {
      stage = 'audience-scoped next links';
      await page.goto(`${base}/portfolio/handwritingRating?audience=data-science`);
      await page.locator('.project-next-link').waitFor();
      await settle(page);
      assert.equal(await page.locator('.project-next-link').getAttribute('href'), '/portfolio/shapeClassifier?audience=data-science');
      assert.equal(await page.locator('.project-question-link').getAttribute('href'), '/contact?audience=data-science');
      await page.locator('.project-next-link').click();
      await page.waitForURL(`${base}/portfolio/shapeClassifier?audience=data-science`);
      await settle(page);
      assert.equal(await page.locator('body').getAttribute('data-audience'), 'data-science');
      assert.equal(await page.locator('h1:visible').innerText(), 'Shape Classifier Demo');
    }
    assert.equal(submissions, 0, 'No contact request may be sent during verification');
    assert.deepEqual(errors, [], 'Project content interactions must not produce page errors');
    console.log(`Project content passed: ${name}, native evidence disclosure, contact prefill/draft preservation, related navigation${name === 'desktop' ? ', audience scope' : ''}.`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `project-content-${name}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${name} ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runProjectContentChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) await checkViewport(options, viewport);
}

module.exports = runProjectContentChecks;

if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'project-content-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runProjectContentChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-project-content') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
