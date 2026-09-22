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
    const bytes = fs.readFileSync(file);
    response.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' }); response.end(bytes);
  } catch (_) { response.writeHead(404).end('Not found'); }
});
function installAudit() {
  const audit = window.__campaignAudit = { frames: 0, overlappingFrames: 0, mixedStageFrames: 0, maxParallel: 0, commits: 0, replacements: 0, failures: [] };
  let lastCount = 0;
  let prior = [];
  let known = new Map();
  let totals = { exposures: 0, websiteVisits: 0, attributedVisits: 0 };
  const fail = (text) => { if (audit.failures.length < 30) audit.failures.push(text); };
  function tick() {
    const main = document.querySelector('#main[data-count]');
    const rows = [...document.querySelectorAll('[data-slot]')];
    if (main && rows.length) {
      audit.frames += 1;
      const count = Number(main.dataset.count);
      if (count < lastCount) { lastCount = count; known = new Map(); prior = []; totals = { exposures: 0, websiteVisits: 0, attributedVisits: 0 }; }
      const active = rows.filter((row) => row.dataset.phase === 'recording');
      audit.maxParallel = Math.max(audit.maxParallel, active.length);
      if (active.length > 1) audit.overlappingFrames += 1;
      if (new Set(active.map((row) => row.dataset.key.split('/')[1])).size > 1) audit.mixedStageFrames += 1;
      if (rows.length !== 5) fail('The five-lane pool changed size.');
      const host = document.querySelector('[data-travelers]'); const box = host.getBoundingClientRect();
      for (const row of rows) {
        const r = row.getBoundingClientRect();
        if (r.height < 20 || r.top < box.top - 1 || r.bottom > box.bottom + 1 || host.scrollHeight > host.clientHeight + 1) fail('A traveler was hidden behind list scrolling or clipping.');
      }
      if (count > lastCount) {
        if (count !== lastCount + 1) fail('More than one commit appeared per frame.');
        const block = document.querySelector(`[data-blocks] [data-height="${count}"]`);
        if (!block) fail('Latest appended block is not visible.');
        else {
          const type = block.dataset.type;
          known.set(block.dataset.key, count);
          if (type === 'ad') totals.exposures += 1;
          if (type === 'website') totals.websiteVisits += 1;
          if (type === 'attribution' && block.dataset.credited === 'true') totals.attributedVisits += 1;
          const id = block.dataset.travelerId; const row = rows.find((item) => item.dataset.travelerId === id);
          if (id && !row) fail('A traveler left before its record was committed.');
          if (row && id) {
            if (type === 'summary' || type === 'attribution') {
              const name = type === 'summary' ? 'summary' : 'attribution';
              if (Number(row.dataset[name + 'Height']) !== count || row.dataset[name + 'Key'] !== block.dataset.key) fail(`${name} and traveler were not committed in the same frame.`);
            } else {
              const step = row.querySelector(`[data-step="${type}"]`);
              if (step.dataset.key !== block.dataset.key || step.dataset.phase !== 'committed' || step.dataset.state !== 'verified' || Number(step.dataset.height) !== count) fail('Traveler milestone does not match the new block.');
            }
          }
        }
        audit.commits += 1; lastCount = count;
      }
      for (const [key, value] of Object.entries(totals)) if (Number(document.querySelector(`[data-metric="${key}"]`).textContent) !== value) fail('Displayed total differs from accepted records: ' + key);
      rows.forEach((row, index) => {
        const old = prior[index];
        if (old?.id && row.dataset.travelerId && old.id !== row.dataset.travelerId) {
          if (old.phase !== 'done' || !old.summary) fail('Replacement happened before measurement ended.');
          audit.replacements += 1;
        }
        for (const step of row.querySelectorAll('[data-step][data-phase="committed"]')) if (known.get(step.dataset.key) !== Number(step.dataset.height)) fail('Recorded milestone has no accepted block.');
      });
      prior = rows.map((row) => ({ id: row.dataset.travelerId, phase: row.dataset.phase, summary: row.dataset.summaryHeight }));
      const writer = document.querySelector('[data-writer]');
      if (writer?.dataset.phase === 'verifying') {
        const row = rows.find((item) => item.dataset.key === writer.dataset.key);
        if (writer.dataset.key.includes('/')) {
          if (!row || row.dataset.phase !== 'verifying' || row.dataset.progress !== writer.dataset.progress) fail('Writer and traveler verification progress differ.');
          else for (const node of [writer, row]) if (Math.abs(parseFloat(getComputedStyle(node).getPropertyValue('--progress')) - Number(writer.dataset.progress)) > .00002) fail('CSS clock differs from the active event.');
        }
        const width = writer.clientWidth; const bar = writer.querySelector('.av-writer-bar');
        if (width > 0 && Math.abs(bar.getBoundingClientRect().width / width - Number(writer.dataset.progress)) > .025) fail('Rendered block progress drifted.');
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
    const context = await browser.newContext({ viewport: { width: 1672, height: 941 }, reducedMotion: 'no-preference' });
    await context.route('**/*', (route) => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await context.addInitScript(installAudit);
    const page = await context.newPage(); const errors = []; const failed = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('response', (response) => { if (response.status() >= 400) failed.push(response.url()); });
    const ready = () => page.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    const waitState = (predicate) => page.waitForFunction(predicate, null, { timeout: 120000 });
    const pause = async () => { if (await page.locator('#main').getAttribute('data-running') === 'true') await page.locator('[data-play]').click(); };
    const capture = (name) => page.screenshot({ path: path.join(output, name + '.png'), fullPage: true });
    const details = (open) => page.locator('.av-about').evaluate((node, value) => { node.open = value; }, open);
    async function exportProof() {
      await pause(); await details(true);
      const promise = page.waitForEvent('download'); await page.locator('[data-export]').click();
      const download = await promise; const value = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
      await details(false); return value;
    }
    async function reset(scenario = 'mixed') {
      await page.locator('[data-reset]').click(); await ready(); await page.locator('[data-scenario]').selectOption(scenario);
    }
    await page.goto(url); await ready();
    assert.equal(await page.title(), 'From an Ad to a Visit | Daniel Short');
    assert.equal(await page.evaluate(() => isSecureContext && !!crypto.subtle), true);
    assert.match(await page.locator('meta[name="robots"]').getAttribute('content'), /noindex/);
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click(); // Local test choice, not a bypass of site privacy controls.
    await capture('01-ready');
    await details(true); await page.locator('[data-continuous]').uncheck(); await details(false);
    for (const speed of [1, 2, 4]) {
      await reset(); await page.locator('[data-speed]').selectOption(String(speed)); await page.locator('[data-play]').click();
      await waitState(() => Number(document.querySelector('#main').dataset.measuring) >= 3);
      if (speed === 1) {
        await pause(); const frozen = await page.locator('[data-travelers]').innerHTML(); const count = await page.locator('#main').getAttribute('data-count');
        await page.waitForTimeout(450); assert.equal(await page.locator('[data-travelers]').innerHTML(), frozen); assert.equal(await page.locator('#main').getAttribute('data-count'), count);
        await page.locator('[data-play]').click();
        await waitState(() => !!document.querySelector('[data-evidence-for][data-credited="true"]')); await pause(); await capture('02-approved-layout-attribution');
        await page.locator('[data-support]').click();
        assert.equal(await page.locator('[data-archive] .av-block').count(), 3);
        assert.match(await page.locator('[data-archive-caption]').textContent(), /Website activity is not required/);
        await page.keyboard.press('Escape'); await page.locator('[data-play]').click();
      }
      await waitState(() => document.querySelector('#main').dataset.running === 'false' && Number(document.querySelector('#main').dataset.completed) === 5);
      const proof = await exportProof(); const verified = await core.verifyResults(proof.blocks, proof.trust);
      assert.equal(proof.blocks.length, 21); assert.equal(verified.report.valid, true); assert.equal(verified.totals.attributedVisits, 2);
      assert.equal(Number(await page.locator('[data-metric="attributedVisits"]').textContent()), 2);
      assert.equal(await page.locator('[data-metric="active"]').textContent(), '0');
    }
    for (const [scenario, expected] of Object.entries({ none: [12, 0], website: [17, 0], destination: [22, 5], both: [27, 5], late: [27, 0] })) {
      await reset(scenario); await page.locator('[data-play]').click();
      await waitState(() => document.querySelector('#main').dataset.running === 'false' && Number(document.querySelector('#main').dataset.completed) === 5);
      const proof = await exportProof(); const verified = await core.verifyResults(proof.blocks, proof.trust);
      assert.equal(proof.blocks.length, expected[0]); assert.equal(verified.report.valid, true); assert.equal(verified.totals.attributedVisits, expected[1]);
    }
    // All late visits were reported; no credited total. Try falsely claiming credit.
    const original = await exportProof(); const originalMetrics = await page.locator('.av-results').innerText();
    assert.match(await page.locator('[data-evidence-for]').textContent(), /Why this visit was not counted/);
    await capture('03-not-attributed');
    await page.locator('[data-blocks] [data-inspect]').click(); await page.locator('[data-dialog][open]').waitFor();
    assert.equal(await page.locator('[data-edit-field]').inputValue(), 'credited');
    await page.locator('[data-edit-boolean]').selectOption('true');
    await page.getByRole('button', { name: 'Apply edit & verify', exact: true }).click();
    await waitState(() => document.querySelector('[data-verdict]').dataset.valid === 'false');
    assert.equal(await page.locator('[data-play]').isDisabled(), true);
    assert.equal(await page.locator('[data-metric="attributedVisits"]').textContent(), '0');
    await capture('04-edited-copy-rejected'); await page.keyboard.press('Escape');
    const edited = await exportProof(); assert.equal((await core.verifyResults(edited.blocks, edited.trust)).totals, null);
    await page.locator('[data-warning] [data-restore]').click(); await ready();
    const restored = await exportProof(); assert.deepEqual(restored.blocks, original.blocks); assert.equal(await page.locator('.av-results').innerText(), originalMetrics);
    await page.locator('[data-history]').click(); assert.equal(await page.locator('[data-archive] .av-block').count(), original.blocks.length); await page.keyboard.press('Escape');
    await reset(); await details(true); await page.locator('[data-continuous]').check(); await details(false);
    await page.locator('[data-play]').click(); await waitState(() => Number(document.querySelector('#main').dataset.admitted) === 5);
    await page.locator('[data-scenario]').selectOption('none'); await waitState(() => Number(document.querySelector('#main').dataset.admitted) >= 9);
    await pause(); const continuous = await exportProof(); assert.equal((await core.verifyChain(continuous.blocks, continuous.trust)).valid, true);
    assert.equal(continuous.blocks.filter((block) => block.transactions[0].event.type === 'campaign').length, 1);
    for (const block of continuous.blocks) if (Number(block.transactions[0].event.travelerId?.slice(1)) > 5) assert.ok(!['website', 'destination', 'attribution'].includes(block.transactions[0].event.type));
    for (const width of [1280, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 941 }); await reset(); await page.locator('[data-play]').click();
      await waitState(() => !!document.querySelector('[data-evidence-for][data-credited="true"]')); await pause();
      await page.locator('.av-workspace').evaluate((node) => node.scrollIntoView({ block: 'start' }));
      const layout = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        lanes: [...document.querySelectorAll('[data-slot]')].map((node) => ({ top: node.getBoundingClientRect().top, bottom: node.getBoundingClientRect().bottom })),
        nestedScroll: document.querySelector('[data-travelers]').scrollHeight > document.querySelector('[data-travelers]').clientHeight + 1,
        overflowing: [...document.querySelectorAll('body *')].filter((node) => { const b = node.getBoundingClientRect(); return b.width && b.right > innerWidth + 1; }).map((node) => ({ className: String(node.className), right: node.getBoundingClientRect().right })) }));
      fs.writeFileSync(path.join(output, `layout-${width}.json`), JSON.stringify(layout, null, 2)); await capture(`05-layout-${width}`);
      assert.ok(layout.documentWidth <= width, `Overflow at ${width}: ${JSON.stringify(layout.overflowing)}`); assert.equal(layout.nestedScroll, false);
      assert.ok(layout.lanes.every((lane) => lane.top >= -1 && lane.bottom <= 942), 'All five lanes must be visible together.');
    }
    await reset(); await page.locator('[data-play]').click(); await waitState(() => document.querySelector('[data-writer]').dataset.phase === 'verifying');
    await reset(); await page.waitForTimeout(500); assert.equal(await page.locator('#main').getAttribute('data-count'), '0');
    const audit = await page.evaluate(() => window.__campaignAudit);
    fs.writeFileSync(path.join(output, 'animation-sync-evidence.json'), JSON.stringify(audit, null, 2));
    assert.ok(audit.overlappingFrames > 100); assert.ok(audit.mixedStageFrames > 0); assert.ok(audit.replacements >= 4);
    assert.deepEqual(audit.failures, []); assert.deepEqual(errors, []); assert.deepEqual(failed, []);
    const slow = await browser.newContext();
    await slow.addInitScript(() => {
      window.__slow = false; window.__waiting = [];
      const sign = crypto.subtle.sign.bind(crypto.subtle);
      crypto.subtle.sign = async (...args) => { if (window.__slow) await new Promise((resolve) => window.__waiting.push(resolve)); return sign(...args); };
    });
    const slowPage = await slow.newPage(); await slowPage.goto(url); await slowPage.waitForFunction(() => !document.querySelector('[data-play]').disabled);
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.count) === 2);
    await slowPage.evaluate(() => { window.__slow = true; }); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.queued) >= 4);
    assert.equal(await slowPage.locator('#main').getAttribute('data-count'), '2'); assert.equal(await slowPage.locator('[data-writer]').getAttribute('data-progress'), '0.80000');
    await slowPage.locator('[data-play]').click(); await slowPage.evaluate(() => { window.__slow = false; window.__waiting.splice(0).forEach((resolve) => resolve()); });
    await slowPage.waitForTimeout(500); assert.equal(await slowPage.locator('#main').getAttribute('data-count'), '2');
    await slowPage.locator('[data-play]').click(); await slowPage.waitForFunction(() => Number(document.querySelector('#main').dataset.count) >= 3); await slow.close();
    const reduced = await browser.newContext({ reducedMotion: 'reduce' }); const reducedPage = await reduced.newPage(); await reducedPage.goto(url);
    assert.equal(await reducedPage.locator('.av-dot').first().evaluate((node) => getComputedStyle(node).display), 'none'); await reduced.close();
    const noJs = await browser.newContext({ javaScriptEnabled: false }); const noJsPage = await noJs.newPage(); await noJsPage.goto(url);
    assert.match(await noJsPage.locator('noscript').innerText(), /JavaScript/); assert.equal(await noJsPage.locator('[data-play]').isDisabled(), true); await noJs.close();
    const missing = await browser.newContext(); const missingPage = await missing.newPage(); await missingPage.route('**/js/demos/ad-verification-core.js*', (route) => route.abort()); await missingPage.goto(url);
    await missingPage.waitForFunction(() => document.querySelector('#main').dataset.error === 'true'); assert.equal(await missingPage.locator('[data-play]').isDisabled(), true); await missing.close();
    console.log(`PASS: ${audit.overlappingFrames} overlapping frames, ${audit.mixedStageFrames} mixed-stage frames, ${audit.commits} atomic commits, ${audit.replacements} replacements; attribution evidence and displayed totals match; zero mismatches.`);
  } finally { if (browser) await browser.close(); await new Promise((resolve) => server.close(resolve)); }
})().catch((error) => { console.error(error); process.exitCode = 1; });
