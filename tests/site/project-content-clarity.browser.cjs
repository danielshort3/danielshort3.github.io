/** Project content visibility, contextual navigation, and contact drafts; never sends a message. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const { loadProjects, isPublishedProject } = require('../../build/generate-project-pages');

async function settle(page) {
  await page.waitForFunction(() => {
    const frame = window.SiteFrame?.root();
    return document.readyState !== 'loading' && !window.SiteNavigation?.isNavigating?.()
      && (!frame || !frame.matches('.site-frame--moving, .site-frame--held'));
  });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function checkEvidenceAbsent(page) {
  assert.equal(await page.locator('.project-evidence, .project-evidence-details').count(), 0,
    'Disabled evidence sections and disclosures must be absent, not merely collapsed');
  assert.equal(await page.getByText('Evidence & limitations', { exact: true }).count(), 0,
    'The evidence disclosure label must not remain visible');
  assert(await page.locator('.project-star').isVisible(), 'The STAR summary must remain available');
  const layout = await page.evaluate(() => {
    const star = document.querySelector('.project-star');
    const demo = document.querySelector('.project-demo-shell');
    return {
      starBeforeDemo: Boolean(star.compareDocumentPosition(demo) & Node.DOCUMENT_POSITION_FOLLOWING),
      overflow: document.documentElement.scrollWidth - innerWidth
    };
  });
  assert(layout.starBeforeDemo, 'The STAR summary must remain before the demo');
  assert(layout.overflow <= 1, 'Project content must not create horizontal page overflow');
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
  await context.route('**/*', route => {
    const request = route.request();
    if (request.resourceType() === 'document' && request.frame().parentFrame()) {
      return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>Local demo fixture</title><p>Interactive demo fixture</p>' });
    }
    return route.fallback();
  });
  try {
    await page.goto(`${base}/portfolio/handwritingRating`);
    await page.locator('.project-star').waitFor();
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click();
    await settle(page);

    stage = 'disabled evidence';
    await checkEvidenceAbsent(page);
    await page.locator('.project-demo-header').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifactDir, `project-content-${name}.png`) });

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

    stage = 'next project in collection';
    const next = page.locator('.project-next-link');
    assert.equal(await next.count(), 1);
    assert.equal(await next.getAttribute('href'), '/portfolio/digitGenerator');
    await next.click();
    await page.waitForURL(`${base}/portfolio/digitGenerator`);
    await settle(page);
    assert.equal(await page.locator('h1:visible').innerText(), 'Synthetic Digit Generator');
    await checkEvidenceAbsent(page);

    for (const audience of ['analytics', 'data-science', 'tourism']) {
      stage = `${audience} project content`;
      await page.goto(`${base}/portfolio/handwritingRating?audience=${audience}`);
      await page.locator('.project-star').waitFor();
      await settle(page);
      assert.equal(await page.locator('body').getAttribute('data-audience'), audience);
      await checkEvidenceAbsent(page);
    }

    if (name === 'desktop') {
      stage = 'audience-scoped next links';
      await page.goto(`${base}/portfolio/handwritingRating?audience=data-science`);
      await page.locator('.project-next-link').waitFor();
      await settle(page);
      assert.equal(await page.locator('.project-next-link').getAttribute('href'), '/portfolio/digitGenerator?audience=data-science');
      assert.equal(await page.locator('.project-question-link').getAttribute('href'), '/contact?audience=data-science');
      await page.locator('.project-next-link').click();
      await page.waitForURL(`${base}/portfolio/digitGenerator?audience=data-science`);
      await settle(page);
      assert.equal(await page.locator('body').getAttribute('data-audience'), 'data-science');
      assert.equal(await page.locator('h1:visible').innerText(), 'Synthetic Digit Generator');
      await checkEvidenceAbsent(page);

      stage = 'complete published project cycle';
      const projects = loadProjects().filter(isPublishedProject);
      await page.goto(`${base}/portfolio/${projects[0].id}`);
      await settle(page);
      const visited = new Set();
      for (const [index, project] of projects.entries()) {
        assert.equal(new URL(page.url()).pathname, `/portfolio/${project.id}`);
        assert.equal(await page.locator('h1:visible').innerText(), project.title);
        visited.add(project.id);
        const destination = projects[(index + 1) % projects.length];
        await page.locator('.project-next-link').click();
        await page.waitForURL(`${base}/portfolio/${destination.id}`);
        await settle(page);
      }
      assert.equal(visited.size, projects.length, 'Explore next visits every project before repeating');
      assert.equal(new URL(page.url()).pathname, `/portfolio/${projects[0].id}`, 'The last project wraps back to the first');
    }
    assert.equal(submissions, 0, 'No contact request may be sent during verification');
    assert.deepEqual(errors, [], 'Project content interactions must not produce page errors');
    console.log(`Project content passed: ${name}, evidence absent across all audiences, contact prefill/draft preservation, related navigation${name === 'desktop' ? ', audience scope' : ''}.`);
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
