'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const core = require('../../js/demos/ad-verification-core');
const repository = path.resolve(__dirname, '../..');
const root = fs.existsSync(path.join(repository, 'public/demos/ad-verification.html')) ? path.join(repository, 'public') : repository;
const output = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'campaign-audit-browser');
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.webp': 'image/webp', '.woff2': 'font/woff2', '.json': 'application/json' };
const server = http.createServer((request, response) => {
  try {
    let file = path.resolve(root, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    if (!path.extname(file)) file += '.html';
    const bytes = fs.readFileSync(file);
    response.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' }); response.end(bytes);
  } catch (_) { response.writeHead(404).end('Not found'); }
});
// Independent rendered-frame audit: many receipts can commit in one block.
function installAudit() {
  const audit = window.__audit = { frames: 0, overlapping: 0, mixedStages: 0, peak: 0, blocks: 0, receipts: 0, multiReceiptBlocks: 0, replacements: 0, layoutFrames: 0, maxHeightDelta: 0, maxResultsDelta: 0, failures: [] };
  let lastBlock = 0;
  let oldRows = [];
  let known = new Map();
  let counts = { exposures: 0, websites: 0 };
  let credit = new Map();
  let layout = null;
  window.__watchLayout = false;
  window.__resetLayout = () => { layout = null; window.__watchLayout = true; };
  const fail = (message) => { if (audit.failures.length < 30) audit.failures.push(message); };
  function tick() {
    const main = document.querySelector('#main[data-blocks]');
    const rows = [...document.querySelectorAll('[data-lane]')];
    if (main && rows.length) {
      audit.frames += 1;
      const n = Number(main.dataset.blocks);
      if (n < lastBlock) { lastBlock = n; oldRows = []; known = new Map(); credit = new Map(); counts = { exposures: 0, websites: 0 }; }
      const observing = rows.filter((row) => row.dataset.observing);
      audit.peak = Math.max(audit.peak, observing.length);
      if (observing.length > 1) audit.overlapping += 1;
      if (new Set(observing.map((row) => row.dataset.observing)).size > 1) audit.mixedStages += 1;
      if (n > lastBlock) {
        if (n !== lastBlock + 1) fail('More than one block became visible in a frame.');
        const block = document.querySelector(`[data-chain] [data-height="${n}"]`);
        const ids = (main.dataset.lastIds || '').split(',').filter(Boolean);
        if (!block || !ids.length) fail('New block or its receipts not visible.');
        if (ids.length > 1) audit.multiReceiptBlocks += 1;
        for (const id of ids) {
          const receipt = block?.querySelector(`[data-receipt="${id}"]`);
          if (!receipt || known.has(id)) { fail('Missing or duplicate receipt.'); continue; }
          known.set(id, n); audit.receipts += 1;
          const type = receipt.dataset.type;
          if (type === 'ad') counts.exposures += 1;
          if (type === 'website') counts.websites += 1;
          if (type === 'attribution') credit.set(id, receipt.dataset.credited === 'true');
          if (type === 'correction') credit.set(receipt.dataset.target, false);
          const row = rows.find((item) => item.dataset.person === receipt.dataset.person);
          if (receipt.dataset.person && !row) fail('Traveler disappeared before its receipt was committed.');
          if (row) {
            if (type === 'attribution' && row.dataset.attribution !== id) fail('Attribution milestone and block did not commit together.');
            if (type === 'end' && row.dataset.ended !== id) fail('End milestone and block did not commit together.');
            if (['ad', 'website', 'visit'].includes(type)) {
              const stage = row.querySelector(`[data-stage="${type}"]`);
              if (stage.dataset.state !== 'recorded' || stage.dataset.receiptId !== id || Number(stage.dataset.block) !== n) fail('Traveler check does not match its batched receipt.');
            }
          }
        }
        audit.blocks += 1; lastBlock = n;
      }
      for (const [key, value] of Object.entries({ ...counts, attributed: [...credit.values()].filter(Boolean).length })) {
        if (Number(document.querySelector(`[data-total="${key}"]`).textContent) !== value) fail('Reported totals do not match accepted receipts: ' + key);
      }
      const host = document.querySelector('[data-travelers]'); const bounds = host.getBoundingClientRect();
      rows.forEach((row, i) => {
        const old = oldRows[i]; const r = row.getBoundingClientRect();
        if (r.height < 20 || r.top < bounds.top - 1 || r.bottom > bounds.bottom + 1 || host.scrollHeight > host.clientHeight + 1) fail('Active traveler is clipped or internally scrolled.');
        if (old?.id && row.dataset.person && old.id !== row.dataset.person) {
          if (old.phase !== 'done' || !old.ended) fail('Individual replacement happened before its receipts were recorded.');
          audit.replacements += 1;
        }
        for (const stage of row.querySelectorAll('[data-stage][data-state="recorded"]')) if (known.get(stage.dataset.receiptId) !== Number(stage.dataset.block)) fail('Recorded traveler check has no committed receipt.');
      });
      oldRows = rows.map((row) => ({ id: row.dataset.person, phase: row.dataset.phase, ended: row.dataset.ended }));
      const writer = document.querySelector('[data-writer]');
      const width = writer.clientWidth; const progress = Number(writer.dataset.progress);
      if (width && Math.abs(writer.querySelector('i').getBoundingClientRect().width / width - progress) > .025) fail('Block verification progress differs from the shared clock.');
      if (window.__watchLayout) {
        const rect = (selector) => document.querySelector(selector).getBoundingClientRect();
        const ledger = rect('.av-ledger'); const people = rect('.av-people'); const results = rect('.av-results');
        const current = { width: innerWidth, ledger: ledger.height, people: people.height, results: results.top + scrollY };
        if (!layout || layout.width !== current.width) layout = current;
        const heightDelta = Math.max(Math.abs(current.ledger - layout.ledger), Math.abs(current.people - layout.people));
        const resultsDelta = Math.abs(current.results - layout.results);
        audit.layoutFrames += 1;
        audit.maxHeightDelta = Math.max(audit.maxHeightDelta, heightDelta);
        audit.maxResultsDelta = Math.max(audit.maxResultsDelta, resultsDelta);
        if (heightDelta > 1) fail('Panel height changed as records or labels changed.');
        if (resultsDelta > 1) fail('Changing ledger content moved the results band.');
      }
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
(async () => {
  let browser;
  try {
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const origin = `http://127.0.0.1:${server.address().port}`;
    const url = origin + '/demos/ad-verification';
    fs.mkdirSync(output, { recursive: true });
    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1500, height: 1050 }, reducedMotion: 'no-preference' });
    await context.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await context.addInitScript(installAudit);
    const page = await context.newPage(); const errors = []; const failed = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => { if (response.status() >= 400) failed.push(response.url()); });
    const waitState = (predicate) => page.waitForFunction(predicate, null, { timeout: 120000 });
    const ready = () => waitState(() => !document.querySelector('[data-play]').disabled);
    const pause = async () => { if (await page.locator('#main').getAttribute('data-running') === 'true') await page.locator('[data-play]').click(); };
    const capture = (name) => page.screenshot({ path: path.join(output, name + '.png'), fullPage: true });
    const details = (open) => page.locator('.av-about').evaluate((node, value) => { node.open = value; }, open);
    async function reset(speed = 4, continuous = false) {
      if (await page.locator('dialog[open]').count()) await page.keyboard.press('Escape');
      await page.locator('[data-reset]').click(); await ready();
      await page.locator('[data-speed]').selectOption(String(speed));
      await details(true); await page.locator('[data-continuous]').setChecked(continuous); await details(false);
      await page.evaluate(async () => { await document.fonts.ready; window.__resetLayout(); });
    }
    async function exportProof() {
      await pause(); await details(true);
      const promise = page.waitForEvent('download'); await page.locator('[data-export]').click();
      const download = await promise;
      assert.equal(download.suggestedFilename(), 'campaign-audit-proof.json');
      const proof = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
      await details(false); return proof;
    }
    async function assertControls() {
      assert.equal(await page.locator('[data-scenario]').count(), 0);
      assert.equal(await page.locator('.av-controls select').count(), 1);
      const controls = await page.evaluate(() => {
        const start = document.querySelector('[data-play]').getBoundingClientRect();
        const speed = document.querySelector('[data-speed]').getBoundingClientRect();
        return { gap: speed.left - start.right, centerDelta: Math.abs((start.top + start.height / 2) - (speed.top + speed.height / 2)) };
      });
      assert.ok(controls.gap > 0 && controls.gap < 100, 'Speed must be immediately beside Start/Resume.');
      assert.ok(controls.centerDelta < 2, 'Playback controls must remain on the same line.');
    }
    await page.goto(url); await ready();
    assert.equal(await page.title(), 'Campaign Attribution Lab | Daniel Short');
    assert.equal(await page.locator('h1').textContent(), 'Campaign Attribution Lab');
    assert.match(await page.locator('.av-intro').textContent(), /advertiser.*agency.*measurement partner/i);
    assert.match(await page.locator('.av-explainer').textContent(), /original report.*correction/i);
    assert.equal(await page.locator('[data-continuous]').isChecked(), false);
    assert.equal(await page.locator('[data-speed]').inputValue(), '2');
    assert.equal(await page.locator('.av-header img').count(), 0);
    assert.doesNotMatch(await page.locator('body').textContent(), /Cedar Valley|Visit Grand Junction/);
    assert.equal(await page.evaluate(() => isSecureContext && !!crypto.subtle), true);
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click();
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    await assertControls(); await capture('01-project-ready');
    // Mixed is the only user-facing mode. Individual scenarios remain covered in Node tests.
    for (const speed of [1, 2, 4]) {
      await reset(speed); await page.locator('[data-play]').click();
      await waitState(() => [...document.querySelectorAll('[data-lane]')].filter((row) => row.dataset.observing).length >= 3);
      if (speed === 1) {
        await pause(); const frozen = await page.locator('[data-travelers]').innerHTML(); const n = await page.locator('#main').getAttribute('data-blocks');
        await page.waitForTimeout(500); assert.equal(await page.locator('[data-travelers]').innerHTML(), frozen); assert.equal(await page.locator('#main').getAttribute('data-blocks'), n);
        await assertControls(); await page.locator('[data-play]').click();
      }
      await waitState(() => Number(document.querySelector('#main').dataset.completed) === 5 && document.querySelector('#main').dataset.running === 'false');
      const proof = await exportProof(); const checked = await core.verifyProof(proof);
      assert.equal(checked.valid, true); assert.equal(checked.totals.attributed, 2); assert.ok(proof.blocks.length < 21);
      assert.equal(proof.trust.campaign, 'EXAMPLE-CAMPAIGN');
      assert.doesNotMatch(JSON.stringify(proof), /Cedar Valley|CV-DEMO|"traveler"|T00\d|hiking-guides|"salt"|"privateKey"/);
    }
    await capture('02-stable-desktop');
    await page.locator('[data-history]').click();
    // No named destination even in the optional private example evidence.
    await page.locator('[data-dialog-body] [data-receipt][data-type="visit"]').first().click();
    await waitState(() => document.querySelector('[data-evidence-check]')?.dataset.evidenceCheck === 'checked');
    assert.match(await page.locator('[data-dialog-body]').textContent(), /Example destination/);
    assert.doesNotMatch(await page.locator('[data-dialog-body]').textContent(), /Cedar Valley/);
    await page.keyboard.press('Escape');
    await page.locator('[data-history]').click();
    const decision = page.locator('[data-dialog-body] [data-receipt][data-type="attribution"][data-credited="true"]').first();
    await decision.click();
    await waitState(() => document.querySelector('[data-evidence-check]')?.dataset.evidenceCheck === 'checked');
    assert.equal(await page.locator('[data-record-check]').getAttribute('data-record-check'), 'true');
    await page.locator('[data-withhold]').click();
    await waitState(() => document.querySelector('[data-evidence-check]')?.dataset.evidenceCheck === 'unavailable');
    assert.equal(await page.locator('[data-record-check]').getAttribute('data-record-check'), 'true');
    await page.locator('[data-withhold]').click();
    await waitState(() => document.querySelector('[data-evidence-check]')?.dataset.evidenceCheck === 'checked');
    await page.keyboard.press('Escape');
    const beforeReport = await exportProof();
    await page.locator('[data-report]').click(); await page.locator('[data-test-report]').waitFor();
    await page.locator('[data-report-value]').fill('14'); await page.locator('[data-test-report]').click();
    await waitState(() => document.querySelector('[data-test-verdict]')?.dataset.valid === 'false');
    assert.equal(await page.locator('[data-total="attributed"]').textContent(), '2');
    await page.locator('[data-correct]').click();
    await waitState(() => document.querySelector('[data-current-total]')?.textContent === '1');
    assert.match(await page.locator('[data-test-verdict]').textContent(), /original report stays in the history/);
    const oldReportId = await page.locator('[data-test-report]').getAttribute('data-test-report');
    await page.locator('[data-new-report]').click();
    await page.waitForFunction((old) => document.querySelector('[data-test-report]')?.dataset.testReport !== old, oldReportId);
    await page.keyboard.press('Escape');
    const after = await exportProof();
    assert.deepEqual(after.blocks.slice(0, beforeReport.blocks.length), beforeReport.blocks);
    assert.equal((await core.verifyProof(after)).totals.attributed, 1);
    const reports = core.receipts(after.blocks).filter((r) => r.type === 'report');
    assert.equal(reports[0].data.totals.attributed, 2); assert.equal(reports.at(-1).data.totals.attributed, 1);
    await page.locator('[data-copies]').click();
    await page.locator('[data-copy-action="alter"][data-copy-index="1"]').click();
    await waitState(() => document.querySelector('[data-copy-status="1"]')?.textContent.startsWith('Mismatch'));
    assert.match(await page.locator('[data-copy-status="0"]').textContent(), /Up to date/);
    await page.locator('[data-copy-action="restore"][data-copy-index="1"]').click();
    await waitState(() => document.querySelector('[data-copy-status="1"]')?.textContent.startsWith('Up to date'));
    await page.locator('[data-copy-action="pause"][data-copy-index="1"]').click();
    await waitState(() => document.querySelector('[data-copy-status="1"]')?.textContent.includes('updates paused'));
    await page.keyboard.press('Escape'); await page.locator('[data-report]').click(); await page.locator('[data-test-report]').waitFor(); await page.keyboard.press('Escape');
    assert.equal(await page.locator('[data-copy="1"]').getAttribute('data-status'), 'Behind');
    await page.locator('[data-copies]').click(); await page.locator('[data-copy-action="restore"][data-copy-index="1"]').click();
    await waitState(() => document.querySelector('[data-copy-status="1"]')?.textContent.startsWith('Up to date')); await page.keyboard.press('Escape');
    await reset(4, true); await page.locator('[data-play]').click();
    await waitState(() => Number(document.querySelector('#main').dataset.admitted) >= 9); await pause();
    assert.equal((await core.verifyProof(await exportProof())).valid, true);
    // Playback, changing batch sizes, receipt expansion and deliberately long labels.
    // The outer panels and results must not move; content remains scrollable, not hidden.
    for (const width of [1280, 1024, 901, 900, 768, 390, 320]) {
      await page.evaluate(() => { window.__watchLayout = false; });
      await page.setViewportSize({ width, height: 1000 }); await reset(); await assertControls();
      await page.locator('[data-play]').click();
      await waitState(() => Number(document.querySelector('#main').dataset.blocks) >= 4); await pause();
      await assertControls();
      for (let i = 0; i < 3; i += 1) {
        await page.locator('[data-chain] [data-block]').nth(i).click();
        await page.waitForTimeout(40);
      }
      await page.locator('.av-workspace').evaluate((node) => node.scrollIntoView({ block: 'start' }));
      const layout = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        nested: document.querySelector('[data-travelers]').scrollHeight > document.querySelector('[data-travelers]').clientHeight + 1,
        ledgerHeight: document.querySelector('.av-ledger').getBoundingClientRect().height,
        lanes: [...document.querySelectorAll('[data-lane]')].map((node) => ({ top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom })) }));
      fs.writeFileSync(path.join(output, `layout-${width}.json`), JSON.stringify(layout, null, 2));
      await capture(`03-stable-layout-${width}`);
      assert.ok(layout.documentWidth <= width, `Overflow at ${width}`); assert.equal(layout.nested, false);
      assert.ok(layout.lanes.every((row) => row.top >= -1 && row.bottom <= 1001));
      // Stress only DOM text, not signed data. Reload/reset rebuilds authentic content afterward.
      await page.evaluate(() => {
        document.querySelectorAll('[data-chain] .av-receipt small').forEach((node) => { node.textContent = 'Long provider report title '.repeat(18); });
      });
      await page.waitForTimeout(100);
      const viewport = page.locator('[data-chain-viewport]');
      const height = await page.locator('.av-ledger').evaluate((node) => node.getBoundingClientRect().height);
      assert.ok(Math.abs(height - layout.ledgerHeight) < 1, 'Long receipts must not grow the panel.');
      await viewport.evaluate((node) => { node.scrollTop = 0; });
      await viewport.focus(); await page.keyboard.press('PageDown'); await page.waitForTimeout(150);
      assert.ok(await viewport.evaluate((node) => node.scrollTop > 0), 'Keyboard scrolling must expose the overflow.');
      assert.ok(await viewport.evaluate((node) => node.scrollWidth <= node.clientWidth + 1), 'Long text must wrap inside the ledger.');
    }
    await reset(); await page.locator('[data-play]').click();
    await waitState(() => Number(document.querySelector('[data-writer]').dataset.progress) > 0);
    await reset(); await page.waitForTimeout(500);
    assert.equal(await page.locator('#main').getAttribute('data-blocks'), '0');
    const audit = await page.evaluate(() => window.__audit);
    fs.writeFileSync(path.join(output, 'animation-sync-evidence.json'), JSON.stringify(audit, null, 2));
    assert.ok(audit.overlapping > 100); assert.ok(audit.mixedStages > 0); assert.ok(audit.multiReceiptBlocks > 10); assert.ok(audit.replacements >= 4);
    assert.ok(audit.layoutFrames > 100); assert.ok(audit.maxHeightDelta <= 1); assert.ok(audit.maxResultsDelta <= 1);
    assert.deepEqual(audit.failures, []); assert.deepEqual(errors, []); assert.deepEqual(failed, []);
    // Hold native verification, not source observation. People must still progress.
    const slow = await browser.newContext();
    await slow.addInitScript(() => {
      window.__hold = false; window.__release = [];
      const verify = crypto.subtle.verify.bind(crypto.subtle);
      crypto.subtle.verify = async (...args) => { if (window.__hold) await new Promise((resolve) => window.__release.push(resolve)); return verify(...args); };
    });
    const slowPage = await slow.newPage(); await slowPage.goto(url); await slowPage.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.blocks) === 1);
    await slowPage.evaluate(() => { window.__hold = true; });
    await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.queue) >= 8, null, { timeout: 60000 });
    assert.equal(await slowPage.locator('#main').getAttribute('data-blocks'), '1');
    assert.equal(await slowPage.locator('[data-writer]').getAttribute('data-progress'), '0.80000');
    assert.ok(await slowPage.locator('[data-stage="website"][data-state="reported"]').count());
    await slowPage.locator('[data-play]').click();
    await slowPage.evaluate(() => { window.__hold = false; window.__release.splice(0).forEach((resolve) => resolve()); });
    await slowPage.waitForTimeout(500); assert.equal(await slowPage.locator('#main').getAttribute('data-blocks'), '1');
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.blocks) > 1);
    await slowPage.locator('[data-reset]').click(); await slowPage.waitForTimeout(500);
    assert.equal(await slowPage.locator('#main').getAttribute('data-blocks'), '0'); await slow.close();
    const reduced = await browser.newContext({ reducedMotion: 'reduce' }); const rp = await reduced.newPage(); await rp.goto(url);
    assert.equal(await rp.locator('.av-motion').first().evaluate((node) => getComputedStyle(node).display), 'none'); await reduced.close();
    const noJs = await browser.newContext({ javaScriptEnabled: false }); const np = await noJs.newPage(); await np.goto(url);
    assert.match(await np.locator('noscript').innerText(), /JavaScript/); assert.equal(await np.locator('[data-play]').isDisabled(), true); await noJs.close();
    const missing = await browser.newContext(); const mp = await missing.newPage(); await mp.route('**/js/demos/ad-verification-core.js*', (route) => route.abort()); await mp.goto(url);
    await mp.waitForFunction(() => document.querySelector('#main').dataset.error === 'true'); assert.equal(await mp.locator('[data-play]').isDisabled(), true); await missing.close();
    console.log(`PASS: ${audit.layoutFrames} geometry samples; max panel shift ${audit.maxHeightDelta}px; max results shift ${audit.maxResultsDelta}px; ${audit.blocks} blocks / ${audit.receipts} receipts checked. Default mixed, adjacent speed, generic project identity, keyboard overflow and audit flows passed.`);
  } finally { if (browser) await browser.close(); await new Promise((resolve) => server.close(resolve)); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
