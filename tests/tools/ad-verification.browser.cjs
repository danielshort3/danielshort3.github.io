'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const os = require('node:os');
const { chromium } = require('playwright');
const core = require('../../js/demos/ad-verification-core');
const repository = path.resolve(__dirname, '../..');
const root = fs.existsSync(path.join(repository, 'public/demos/ad-verification.html')) ? path.join(repository, 'public') : repository;
const output = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'ad-verification-browser');
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.woff2': 'font/woff2', '.webp': 'image/webp', '.json': 'application/json' };
const server = http.createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    let file = path.resolve(root, '.' + pathname);
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    if (!path.extname(file)) file += '.html';
    const bytes = fs.readFileSync(file);
    response.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
    response.end(bytes);
  } catch (_) { response.writeHead(404).end('Not found'); }
});

// Runs independently of the application on every rendered animation frame.
function installVisualAudit() {
  window.__avAudit = { frames: 0, activeFrames: 0, commits: [], failures: [], lastCount: 0 };
  function sample() {
    const audit = window.__avAudit;
    const fail = (message) => { if (audit.failures.length < 30) audit.failures.push(message); };
    const main = document.querySelector('#main[data-block-count]');
    if (main) {
      audit.frames += 1;
      const count = Number(main.dataset.blockCount);
      const blocks = [...document.querySelectorAll('.av-block')];
      if (count !== blocks.length) fail(`Count mismatch: ${count} versus ${blocks.length}`);
      if (count < audit.lastCount) audit.lastCount = count; // An explicit campaign reset.
      if (count > audit.lastCount) {
        if (count !== audit.lastCount + 1) fail('More than one block appeared in a frame.');
        const block = blocks.at(-1);
        const id = block.dataset.eventId;
        const traveler = block.dataset.travelerId;
        const type = block.dataset.type;
        const row = traveler && document.querySelector(`[data-traveler="${traveler}"]`);
        const target = traveler ? row?.querySelector(type === 'summary' ? '[data-summary]' : `[data-step="${type}"]`) : document.querySelector(`[data-campaign-step="${type}"]`);
        if (target && (target.dataset.eventId !== id || target.dataset.phase !== 'committed' || target.dataset.state !== 'verified')) fail(`Traveler and block ${id} did not commit together.`);
        if (main.dataset.running === 'true' && !target) fail(`The active traveler for ${id} was not rendered.`);
        audit.commits.push({ id, traveler, type, count });
        audit.lastCount = count;
      }
      if (main.dataset.running === 'true' && ['recording', 'verifying'].includes(main.dataset.phase)) {
        audit.activeFrames += 1;
        const id = main.dataset.activeEvent;
        const pending = document.querySelector('[data-pending]');
        const target = [...document.querySelectorAll(`[data-event-id="${id}"][data-state="active"]`)].find((node) => node !== pending);
        const progress = Number(main.dataset.progress);
        if (!pending || !target) fail(`Missing paired view for ${id}.`);
        else {
          for (const node of [pending, target]) {
            if (node.dataset.eventId !== id || node.dataset.phase !== main.dataset.phase || node.dataset.progress !== main.dataset.progress) fail(`Phase/progress mismatch for ${id}.`);
            if (Math.abs(parseFloat(getComputedStyle(node).getPropertyValue('--progress')) - progress) > .00001) fail(`CSS clock mismatch for ${id}.`);
          }
          const bar = pending.querySelector('.av-pending-bar');
          const width = pending.getBoundingClientRect().width - 2;
          if (width > 0 && Math.abs(bar.getBoundingClientRect().width / width - progress) > .025) fail(`Block progress animation drift for ${id}.`);
          const row = target.closest('[data-traveler]');
          const runner = row?.querySelector('.av-travel-runner');
          if (runner && row.dataset.moving === 'true' && document.body.dataset.reducedMotion !== 'true') {
            const expected = parseFloat(row.style.getPropertyValue('--runner-x')) / 100 * row.querySelector('.av-path').getBoundingClientRect().width;
            if (Math.abs(parseFloat(getComputedStyle(runner).left) - expected) > 1) fail(`Traveler motion drift for ${id}.`);
          }
        }
      }
    }
    requestAnimationFrame(sample);
  }
  requestAnimationFrame(sample);
}

(async () => {
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    const url = origin + '/demos/ad-verification';
    fs.mkdirSync(output, { recursive: true });
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1312, height: 1199 }, reducedMotion: 'no-preference', acceptDownloads: true });
    await context.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await context.addInitScript(installVisualAudit);
    const page = await context.newPage();
    const errors = [];
    const failedResources = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => { if (response.status() >= 400) failedResources.push(response.url()); });
    const ready = () => page.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    const countIs = (count) => page.waitForFunction((n) => Number(document.querySelector('#main').dataset.blockCount) === n, count, { timeout: 90000 });
    const completed = (count) => page.waitForFunction((n) => Number(document.querySelector('#main').dataset.blockCount) === n && document.querySelector('#main').dataset.running === 'false', count, { timeout: 90000 });
    async function exportProof() {
      const promise = page.waitForEvent('download');
      await page.locator('[data-export]').click();
      const download = await promise;
      return JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    }
    await page.goto(url);
    await ready();
    assert.equal(await page.title(), 'Live Campaign Blockchain | Daniel Short');
    assert.equal(await page.evaluate(() => isSecureContext && !!crypto.subtle), true);
    assert.equal(await page.locator('.av-block').count(), 0);
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    await page.screenshot({ path: path.join(output, '01-ready-desktop.png'), fullPage: true });
    await page.locator('[data-continuous]').uncheck();

    // Wall-clock, native-browser runs at every offered speed—not just a clock mock.
    const runs = [];
    for (const speed of [1, 2, 4]) {
      if (speed !== 1) { await page.locator('[data-reset]').click(); await ready(); }
      await page.locator(`[data-speed="${speed}"]`).click();
      const began = Date.now();
      await page.locator('[data-play]').click();
      await page.waitForFunction(() => document.querySelector('[data-pending]')?.dataset.phase === 'verifying');
      if (speed === 1) {
        await page.locator('[data-play-inline]').click();
        const paused = await page.evaluate(() => ({ progress: document.querySelector('#main').dataset.progress, count: document.querySelector('#main').dataset.blockCount,
          css: document.querySelector('[data-pending]').style.getPropertyValue('--progress') }));
        await page.waitForTimeout(450);
        assert.deepEqual(await page.evaluate(() => ({ progress: document.querySelector('#main').dataset.progress, count: document.querySelector('#main').dataset.blockCount,
          css: document.querySelector('[data-pending]').style.getPropertyValue('--progress') })), paused);
        await page.locator('[data-play-inline]').click();
        await page.waitForFunction(() => document.querySelector('#main').dataset.activeEvent === 'event-7' && document.querySelector('#main').dataset.phase === 'verifying', null, { timeout: 90000 });
        await page.screenshot({ path: path.join(output, '02-synchronized-desktop.png'), fullPage: true });
      }
      await completed(14);
      const proof = await exportProof();
      assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
      assert.equal(proof.blocks.length, 14);
      assert.ok(Object.values(proof.trust.publicKeys).every((key) => !('d' in key)));
      runs.push({ speed, blocks: proof.blocks.length, milliseconds: Date.now() - began });
    }
    await page.screenshot({ path: path.join(output, '03-completed-desktop.png'), fullPage: true });

    const expected = { none: 10, website: 14, destination: 14, both: 18 };
    for (const [scenario, count] of Object.entries(expected)) {
      await page.locator('[data-reset]').click(); await ready();
      await page.locator('[data-scenario]').selectOption(scenario);
      await page.locator('[data-play]').click(); await completed(count);
      const proof = await exportProof();
      assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
      const events = proof.blocks.map((block) => block.transactions[0].event);
      if (scenario === 'none' || scenario === 'website') assert.equal(events.filter((event) => event.type === 'destination').length, 0);
      if (scenario === 'none' || scenario === 'destination') assert.equal(events.filter((event) => event.type === 'website').length, 0);
    }

    // Mixed first, then a different group without clearing or forking the ledger.
    await page.locator('[data-reset]').click(); await ready();
    await page.locator('[data-scenario]').selectOption('mixed');
    await page.locator('[data-continuous]').check();
    await page.locator('[data-play]').click(); await countIs(4);
    await page.locator('[data-scenario]').selectOption('none');
    await page.waitForFunction(() => document.querySelectorAll('[data-group] option').length === 2, null, { timeout: 90000 });
    await page.locator('[data-continuous]').uncheck(); await completed(22);
    const continuousProof = await exportProof();
    assert.equal((await core.verifyChain(continuousProof.blocks, continuousProof.trust)).valid, true);
    assert.equal(continuousProof.blocks.filter((block) => block.transactions[0].event.type === 'campaign').length, 1);
    assert.equal(continuousProof.blocks[14].header.previousHash, continuousProof.blocks[13].hash);
    assert.equal(continuousProof.blocks[14].transactions[0].event.travelerId, 'T005');
    await page.locator('[data-group]').selectOption('1');
    await page.locator('[data-highlight-person="T001"]').click();
    assert.equal(await page.locator('.av-block[data-highlight="match"]').count(), 4);
    assert.equal(await page.locator('.av-block').count(), 22);

    // Select ANY block, edit its data, and validate original signatures are retained.
    await page.locator('[data-inspect="2"]').click();
    await page.locator('dialog[open]').waitFor();
    await page.locator('[data-edit-value]').fill('9');
    await page.getByRole('button', { name: 'Apply edit & verify', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-dialog-verdict]').dataset.valid === 'false');
    assert.equal(await page.locator('.av-block[data-state="changed"]').count(), 1);
    assert.equal(await page.locator('.av-block[data-state="dependent"]').count(), 19);
    assert.equal(await page.locator('[data-play]').isDisabled(), true);
    await page.screenshot({ path: path.join(output, '04-tamper-detected.png'), fullPage: true });
    await page.locator('dialog [data-restore]').click();
    await page.waitForFunction(() => document.querySelector('[data-dialog-verdict]').dataset.valid === 'true');
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('dialog[open]').count(), 0);
    assert.deepEqual((await exportProof()).blocks, continuousProof.blocks);
    await page.locator('[data-clear-highlight]').click();

    // A reset while a candidate is visible must not append any stale work later.
    await page.locator('[data-play]').click();
    await page.waitForFunction(() => !!document.querySelector('[data-pending]'));
    await page.locator('[data-reset]').click(); await ready(); await page.waitForTimeout(500);
    assert.equal(await page.locator('.av-block').count(), 0);

    for (const width of [1024, 820, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await page.locator('[data-scenario]').selectOption('mixed');
      await page.locator('[data-play]').click();
      await page.waitForFunction(() => document.querySelector('#main').dataset.activeEvent === 'event-7' && document.querySelector('#main').dataset.phase === 'verifying', null, { timeout: 90000 });
      await page.locator('[data-play-inline]').click();
      await page.screenshot({ path: path.join(output, `05-synchronized-${width}.png`), fullPage: false });
      const layout = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        overflowing: [...document.querySelectorAll('body *')].filter((node) => {
          const bounds = node.getBoundingClientRect();
          return bounds.width && bounds.right > innerWidth + 1;
        }).map((node) => ({ tag: node.tagName, className: String(node.className),
          text: (node.innerText || '').slice(0, 90), right: node.getBoundingClientRect().right })) }));
      fs.writeFileSync(path.join(output, `layout-${width}.json`), JSON.stringify(layout, null, 2));
      fs.writeFileSync(path.join(output, 'animation-audit-latest.json'), JSON.stringify(await page.evaluate(() => window.__avAudit), null, 2));
      assert.ok(layout.documentWidth <= width, `Overflow at ${width}px: ${JSON.stringify(layout.overflowing)}`);
      await page.locator('[data-reset]').click(); await ready();
    }
    const audit = await page.evaluate(() => window.__avAudit);
    assert.ok(audit.activeFrames > 100, 'Too few real animation frames checked.');
    assert.deepEqual(audit.failures, []);
    assert.deepEqual(errors, []);
    assert.deepEqual(failedResources, []);
    fs.writeFileSync(path.join(output, 'animation-sync-evidence.json'), JSON.stringify({ method: 'Independent requestAnimationFrame DOM and computed-CSS audit with native browser Web Crypto', runs, frames: audit.frames, activeFrames: audit.activeFrames, commitsChecked: audit.commits.length, failures: audit.failures, viewports: [1312, 1024, 820, 768, 390, 320] }, null, 2));

    // Real Web Crypto with deliberately delayed signing: no early green checks.
    const slowContext = await browser.newContext();
    await slowContext.addInitScript(() => {
      const original = crypto.subtle.sign.bind(crypto.subtle);
      crypto.subtle.sign = async (...args) => {
        await new Promise((resolve) => setTimeout(resolve, 800));
        return original(...args);
      };
    });
    const slowPage = await slowContext.newPage();
    await slowPage.goto(url);
    await slowPage.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    await slowPage.locator('[data-speed="4"]').click();
    await slowPage.locator('[data-play]').click();
    await slowPage.waitForFunction(() => document.querySelector('#main').dataset.progress === '0.78000');
    assert.equal(await slowPage.locator('.av-block').count(), 0);
    await slowPage.locator('[data-play-inline]').click();
    await slowPage.waitForTimeout(1900);
    assert.equal(await slowPage.locator('.av-block').count(), 0);
    assert.equal(await slowPage.locator('#main').getAttribute('data-progress'), '0.78000');
    await slowPage.locator('[data-play-inline]').click();
    await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.blockCount) === 1);
    await slowPage.waitForFunction(() => !!document.querySelector('[data-pending]'));
    await slowPage.locator('[data-reset]').click();
    await slowPage.waitForTimeout(1900);
    assert.equal(await slowPage.locator('.av-block').count(), 0);
    await slowContext.close();

    const reducedContext = await browser.newContext({ reducedMotion: 'reduce', viewport: { width: 390, height: 900 } });
    const reducedPage = await reducedContext.newPage(); await reducedPage.goto(url);
    await reducedPage.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    await reducedPage.locator('[data-play]').click();
    await reducedPage.waitForFunction(() => !!document.querySelector('[data-pending]'));
    assert.equal(await reducedPage.locator('body').getAttribute('data-reduced-motion'), 'true');
    assert.equal(await reducedPage.locator('.av-travel-runner').first().evaluate((node) => getComputedStyle(node).display), 'none');
    await reducedContext.close();
    const noJs = await browser.newContext({ javaScriptEnabled: false });
    const noJsPage = await noJs.newPage(); await noJsPage.goto(url);
    assert.match(await noJsPage.locator('noscript').innerText(), /JavaScript/);
    assert.equal(await noJsPage.locator('[data-play]').isDisabled(), true);
    await noJs.close();
    const unavailable = await browser.newContext();
    const unavailablePage = await unavailable.newPage();
    await unavailablePage.route('**/js/demos/ad-verification-core.js*', (route) => route.abort());
    await unavailablePage.goto(url);
    await unavailablePage.waitForFunction(() => document.querySelector('#main').dataset.phase === 'error');
    assert.equal(await unavailablePage.locator('.av-block').count(), 0);
    assert.equal(await unavailablePage.locator('[data-play]').isDisabled(), true);
    await unavailable.close();
    console.log(`Ad verification passed: ${audit.activeFrames} synchronized animation frames, ${audit.commits.length} atomic commits, five scenarios, continuous paths, native Web Crypto, tamper/restore, pause/resume, six viewport widths.`);
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
