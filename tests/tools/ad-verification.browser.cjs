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
const output = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'ad-verification-browser');
const mime = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.webp': 'image/webp', '.woff2': 'font/woff2', '.json': 'application/json' };
const server = http.createServer((request, response) => {
  try {
    let file = path.resolve(root, '.' + decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    if (!file.startsWith(root + path.sep)) { response.writeHead(403).end(); return; }
    if (!path.extname(file)) file += '.html';
    response.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' }); response.end(fs.readFileSync(file));
  } catch (_) { response.writeHead(404).end('Not found'); }
});
function installAudit() {
  const audit = window.__campaignAudit = { frames: 0, overlappingFrames: 0, mixedStageFrames: 0, maxParallel: 0, commits: 0, replacements: 0, failures: [] };
  let lastCount = 0;
  let prior = [];
  let known = new Map();
  const fail = (text) => { if (audit.failures.length < 25) audit.failures.push(text); };
  function tick() {
    const main = document.querySelector('#main[data-count]');
    const rows = [...document.querySelectorAll('[data-slot]')];
    if (main && rows.length) {
      audit.frames += 1;
      const count = Number(main.dataset.count);
      if (count < lastCount) { lastCount = count; known = new Map(); prior = []; }
      const active = rows.filter((row) => row.dataset.phase === 'recording');
      audit.maxParallel = Math.max(audit.maxParallel, active.length);
      if (active.length > 1) audit.overlappingFrames += 1;
      if (new Set(active.map((row) => row.dataset.key.split('/')[1])).size > 1) audit.mixedStageFrames += 1;
      if (rows.length !== 5) fail('The five-lane pool changed size.');
      const host = document.querySelector('[data-travelers]');
      const box = host.getBoundingClientRect();
      for (const row of rows) {
        const r = row.getBoundingClientRect();
        if (r.height < 20 || r.top < box.top - 1 || r.bottom > box.bottom + 1 || host.scrollHeight > host.clientHeight + 1) fail('A traveler was hidden behind scrolling or clipping.');
      }
      if (count > lastCount) {
        if (count !== lastCount + 1) fail('Non-atomic append.');
        const block = document.querySelector(`[data-blocks] [data-height="${count}"]`);
        if (!block) fail('Latest block missing from the chain.');
        else {
          known.set(block.dataset.key, count);
          const id = block.dataset.travelerId;
          const row = rows.find((item) => item.dataset.travelerId === id);
          if (id && !row) fail('A traveler disappeared before their final record.');
          if (row && id) {
            if (block.dataset.type === 'summary') {
              if (row.dataset.phase !== 'done' || Number(row.dataset.summaryHeight) !== count || row.dataset.summaryKey !== block.dataset.key) fail('Summary and lane completion differ.');
            } else {
              const step = row.querySelector(`[data-step="${block.dataset.type}"]`);
              if (step.dataset.key !== block.dataset.key || step.dataset.phase !== 'committed' || step.dataset.state !== 'verified' || Number(step.dataset.height) !== count) fail('Recorded milestone did not match its block in the same frame.');
            }
          }
        }
        audit.commits += 1; lastCount = count;
      }
      rows.forEach((row, index) => {
        const old = prior[index];
        if (old?.id && row.dataset.travelerId && old.id !== row.dataset.travelerId) {
          if (old.phase !== 'done' || !old.summary) fail('Traveler replaced before measurement ended.');
          audit.replacements += 1;
        }
        for (const step of row.querySelectorAll('[data-step][data-phase="committed"]')) if (known.get(step.dataset.key) !== Number(step.dataset.height)) fail('A recorded check has no matching block.');
      });
      prior = rows.map((row) => ({ id: row.dataset.travelerId, phase: row.dataset.phase, summary: row.dataset.summaryHeight }));
      const writer = document.querySelector('[data-writer]');
      if (writer?.dataset.phase === 'verifying') {
        const row = rows.find((item) => item.dataset.key === writer.dataset.key);
        if (writer.dataset.key.includes('/')) {
          if (!row || row.dataset.phase !== 'verifying' || row.dataset.progress !== writer.dataset.progress) fail('Writer and matching lane are out of sync.');
          else {
            const value = Number(writer.dataset.progress);
            for (const node of [writer, row]) if (Math.abs(parseFloat(getComputedStyle(node).getPropertyValue('--progress')) - value) > .00002) fail('CSS progress differs from shared clock.');
          }
        }
        const bar = writer.querySelector('.av-writer-bar');
        const width = writer.clientWidth;
        if (width > 0 && Math.abs(bar.getBoundingClientRect().width / width - Number(writer.dataset.progress)) > .025) fail('Block progress width drifted.');
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
    const context = await browser.newContext({ viewport: { width: 1280, height: 1000 }, reducedMotion: 'no-preference' });
    await context.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await context.addInitScript(installAudit);
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const ready = () => page.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    const waitState = (predicate) => page.waitForFunction(predicate, null, { timeout: 120000 });
    const pause = async () => { if (await page.locator('#main').getAttribute('data-running') === 'true') await page.locator('[data-play]').click(); };
    const screenshot = (name) => page.screenshot({ path: path.join(output, name + '.png'), fullPage: true });
    async function exportProof() {
      await pause();
      await page.locator('.av-about').evaluate((node) => { node.open = true; });
      const promise = page.waitForEvent('download'); await page.locator('[data-export]').click();
      const download = await promise; const value = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
      await page.locator('.av-about').evaluate((node) => { node.open = false; });
      return value;
    }
    async function reset(scenario = 'mixed') {
      await page.locator('[data-reset]').click(); await ready();
      await page.locator('[data-scenario]').selectOption(scenario);
    }
    await page.goto(url); await ready();
    assert.equal(await page.title(), 'Live Campaign Measurement | Daniel Short');
    assert.equal(await page.evaluate(() => isSecureContext && !!crypto.subtle), true);
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    assert.equal(await page.locator('[data-slot]').count(), 5);
    assert.equal(await page.locator('[data-group]').count(), 0);
    await screenshot('01-ready');
    // Exercise every offered speed with real browser time and real cryptography.
    for (const speed of [1, 2, 4]) {
      await reset(); await page.locator('[data-speed]').selectOption(String(speed));
      await page.locator('.av-about').evaluate((node) => { node.open = true; });
      await page.locator('[data-continuous]').uncheck();
      await page.locator('.av-about').evaluate((node) => { node.open = false; });
      await page.locator('[data-play]').click();
      await waitState(() => Number(document.querySelector('#main').dataset.measuring) >= 3);
      if (speed === 1) {
        await screenshot('02-overlapping-events'); await pause();
        const before = await page.locator('[data-travelers]').innerHTML();
        const count = await page.locator('#main').getAttribute('data-count');
        await page.waitForTimeout(500);
        assert.equal(await page.locator('[data-travelers]').innerHTML(), before);
        assert.equal(await page.locator('#main').getAttribute('data-count'), count);
        await page.locator('[data-play]').click();
      }
      await waitState(() => document.querySelector('#main').dataset.running === 'false' && Number(document.querySelector('#main').dataset.completed) === 5);
      const proof = await exportProof();
      assert.equal(proof.blocks.length, 17);
      assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
    }
    for (const [scenario, expected] of Object.entries({ none: 12, website: 17, destination: 17, both: 22 })) {
      await reset(scenario); await page.locator('[data-play]').click();
      await waitState(() => document.querySelector('#main').dataset.running === 'false' && Number(document.querySelector('#main').dataset.completed) === 5);
      const proof = await exportProof(); assert.equal(proof.blocks.length, expected); assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
      const events = proof.blocks.map((block) => block.transactions[0].event);
      if (['none', 'website'].includes(scenario)) assert.equal(events.filter((event) => event.type === 'destination').length, 0);
      if (['none', 'destination'].includes(scenario)) assert.equal(events.filter((event) => event.type === 'website').length, 0);
    }
    await reset(); await page.locator('.av-about').evaluate((node) => { node.open = true; }); await page.locator('[data-continuous]').check(); await page.locator('.av-about').evaluate((node) => { node.open = false; });
    await page.locator('[data-play]').click();
    await waitState(() => Number(document.querySelector('#main').dataset.admitted) === 5);
    await page.locator('[data-scenario]').selectOption('none');
    await waitState(() => Number(document.querySelector('#main').dataset.admitted) >= 9);
    await pause(); await screenshot('03-individual-replacements');
    const proof = await exportProof();
    assert.equal(proof.blocks.filter((block) => block.transactions[0].event.type === 'campaign').length, 1);
    assert.equal((await core.verifyChain(proof.blocks, proof.trust)).valid, true);
    for (const block of proof.blocks) if (Number(block.transactions[0].event.travelerId?.slice(1)) > 5) assert.ok(!['website', 'destination'].includes(block.transactions[0].event.type));
    await page.locator('[data-history]').click();
    assert.equal(await page.locator('[data-archive] [data-inspect]').count(), proof.blocks.length);
    await page.locator('[data-archive] [data-inspect="2"]').click(); await page.locator('[data-dialog][open]').waitFor();
    await page.locator('[data-edit-value]').fill('9'); await page.getByRole('button', { name: 'Apply edit & verify', exact: true }).click();
    await waitState(() => document.querySelector('[data-verdict]').dataset.valid === 'false');
    assert.equal(await page.locator('[data-play]').isDisabled(), true); await screenshot('04-edit-detected');
    await page.keyboard.press('Escape');
    const altered = await exportProof(); assert.equal((await core.verifyChain(altered.blocks, altered.trust)).valid, false);
    assert.equal(altered.blocks[2].transactions[0].signature, proof.blocks[2].transactions[0].signature);
    await page.locator('[data-warning] [data-restore]').click(); await ready();
    assert.deepEqual((await exportProof()).blocks, proof.blocks);
    // Native screenshots and bounds: ALL active travelers, with no nested scrolling.
    for (const width of [1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 }); await reset(); await page.locator('[data-play]').click();
      await waitState(() => Number(document.querySelector('#main').dataset.measuring) >= 3); await pause();
      await page.locator('.av-workspace').evaluate((node) => node.scrollIntoView({ block: 'start' }));
      const layout = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        lanes: [...document.querySelectorAll('[data-slot]')].map((node) => ({ top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom })),
        innerScroll: document.querySelector('[data-travelers]').scrollHeight > document.querySelector('[data-travelers]').clientHeight + 1 }));
      fs.writeFileSync(path.join(output, `layout-${width}.json`), JSON.stringify(layout, null, 2));
      await page.screenshot({ path: path.join(output, `05-visible-lanes-${width}.png`), fullPage: false });
      assert.ok(layout.documentWidth <= width, `Horizontal overflow at ${width}`);
      assert.equal(layout.innerScroll, false);
      assert.ok(layout.lanes.every((lane) => lane.top >= -1 && lane.bottom <= 901), `A traveler is offscreen at ${width}`);
    }
    await reset(); await page.locator('[data-play]').click();
    await waitState(() => document.querySelector('[data-writer]').dataset.phase === 'verifying');
    await reset(); await page.waitForTimeout(500); assert.equal(await page.locator('#main').getAttribute('data-count'), '0');
    const audit = await page.evaluate(() => window.__campaignAudit);
    fs.writeFileSync(path.join(output, 'animation-sync-evidence.json'), JSON.stringify(audit, null, 2));
    assert.ok(audit.overlappingFrames > 100); assert.ok(audit.mixedStageFrames > 0); assert.ok(audit.maxParallel >= 3); assert.ok(audit.replacements >= 4);
    assert.deepEqual(audit.failures, []); assert.deepEqual(errors, []);
    // Slow native signatures: other travelers continue, but no unverified block commits.
    const slow = await browser.newContext();
    await slow.addInitScript(() => {
      window.__slow = false; window.__waiting = [];
      const sign = crypto.subtle.sign.bind(crypto.subtle);
      crypto.subtle.sign = async (...args) => { if (window.__slow) await new Promise((resolve) => window.__waiting.push(resolve)); return sign(...args); };
    });
    const slowPage = await slow.newPage(); await slowPage.goto(url); await slowPage.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.count) === 2);
    await slowPage.evaluate(() => { window.__slow = true; });
    await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.queued) >= 4);
    assert.equal(await slowPage.locator('#main').getAttribute('data-count'), '2');
    assert.equal(await slowPage.locator('[data-writer]').getAttribute('data-progress'), '0.80000');
    await slowPage.locator('[data-play]').click();
    await slowPage.evaluate(() => { window.__slow = false; window.__waiting.splice(0).forEach((resolve) => resolve()); });
    await slowPage.waitForTimeout(500); assert.equal(await slowPage.locator('#main').getAttribute('data-count'), '2');
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.count) >= 3);
    await slow.close();
    const reduced = await browser.newContext({ reducedMotion: 'reduce' }); const reducedPage = await reduced.newPage(); await reducedPage.goto(url);
    assert.equal(await reducedPage.locator('.av-dot').first().evaluate((node) => getComputedStyle(node).display), 'none'); await reduced.close();
    const noJs = await browser.newContext({ javaScriptEnabled: false }); const noJsPage = await noJs.newPage(); await noJsPage.goto(url);
    assert.match(await noJsPage.locator('noscript').innerText(), /JavaScript/); assert.equal(await noJsPage.locator('[data-play]').isDisabled(), true); await noJs.close();
    const missing = await browser.newContext(); const missingPage = await missing.newPage(); await missingPage.route('**/js/demos/ad-verification-core.js*', (route) => route.abort()); await missingPage.goto(url);
    await missingPage.waitForFunction(() => document.querySelector('#main').dataset.error === 'true'); assert.equal(await missingPage.locator('[data-play]').isDisabled(), true); await missing.close();
    console.log(`PASS: ${audit.overlappingFrames} overlapping frames, ${audit.mixedStageFrames} mixed-stage frames, ${audit.commits} atomic commits, ${audit.replacements} individual replacements; zero mismatches.`);
  } finally { if (browser) await browser.close(); await new Promise((resolve) => server.close(resolve)); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
