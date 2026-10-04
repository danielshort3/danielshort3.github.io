'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core.js');
const Storage = require('../../js/games/wayfarers-guild/persistence.js');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-compact-guide-')));
const report = { cases: [], errors: [], textScale: '130% text-only CSS emulation; native WebView textZoom remains a separate device gate' };
const scaleText = '.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}.wx-guide .wx-guide-quote{font-size:15.6px!important}.wx-guide-meta{font-size:14.3px!important;line-height:20.8px!important}.wx-game .wx-skill-tile .wx-upgrade-info strong{font-size:16.9px!important;line-height:22.1px!important}.wx-game .wx-skill-tile small{font-size:14.3px!important}.wx-game .wx-price{font-size:15.6px!important}';

async function run() {
  fs.mkdirSync(output, { recursive: true });
  const directory = path.join(output, 'bundle'); bundle(directory);
  const server = http.createServer((request, response) => {
    const relative = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/assets\//, '/');
    const file = path.resolve(directory, '.' + relative);
    if (!file.startsWith(directory + path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file, (error, bytes) => {
      if (error) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type', ({ '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.webp': 'image/webp' })[path.extname(file)] || 'application/json');
      response.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({ headless: true });
  async function geometry(page, label) {
    await page.clock.runFor(200);
    const measure = () => {
      const box = node => { const b = node?.getBoundingClientRect(); return b && { x: b.x, y: b.y, width: b.width, height: b.height, right: b.right, bottom: b.bottom }; };
      const guide = document.querySelector('.wx-guide[open]'), target = document.querySelector('[data-guide-target]'), card = guide?.querySelector('.wx-guide-card'), rect = box(target);
      const hit = rect && document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
      return { width: innerWidth, height: innerHeight, step: guide?.dataset.step, missing: guide?.dataset.missing, card: box(card), target: rect, hit: !!hit && (target === hit || target.contains(hit)), ring: !!guide && !guide.querySelector('.wx-guide-ring').hidden, leave: !!guide && !guide.querySelector('[data-guide-leave]').hidden, retry: !!guide && !guide.querySelector('[data-guide-next]').hidden, dock: box(document.querySelector('.wx-dock')), scrollTop: document.querySelector('.wx-dock')?.scrollTop, overflow: document.documentElement.scrollWidth - innerWidth };
    };
    // ResizeObserver and the browser's real rendering frame settle after the
    // mocked game clock. Wait for geometry, never bypass pointer hit testing.
    await page.waitForFunction(() => {
      const g = document.querySelector('.wx-guide[open]'), t = document.querySelector('[data-guide-target]');
      if (!g || g.dataset.missing !== 'false' || !t) return false;
      const b = t.getBoundingClientRect(), hit = document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2);
      return hit && (hit === t || t.contains(hit));
    }, null, { timeout: 5000 }).catch(async error => { report.cases.push({ label, failed: await page.evaluate(measure) }); throw error; });
    const g = await page.evaluate(measure); report.cases.push({ label, ...g });
    assert.equal(g.missing, 'false'); assert(g.hit && g.ring, label + ': real target receives pointer input');
    assert(!g.leave && !g.retry, label + ': mandatory action remains required');
    assert(g.target.width >= 48 && g.target.height >= 48, label + ': original touch control remains full size');
    assert(g.card.x >= 0 && g.card.y >= 0 && g.card.right <= g.width + 1 && g.card.bottom <= g.height + 1, label + ': coach fits');
    assert(g.overflow <= 1, label + ': no page overflow');
    await page.screenshot({ path: path.join(output, label + '.png') });
    return g;
  }
  try {
    for (const [width, height] of [[670, 268], [640, 256]]) {
      const seed = Core.createState(1000); assert(Core.act(seed, { type: 'onboarding-visit', id: 'greenway' }).ok);
      while (Core.getView(seed).onboarding.active.mode === 'currency') assert(Core.act(seed, Core.getView(seed).onboarding.active.ackAction).ok);
      const record = Storage.createStore({ storage: null, now: () => 1000 }).export(seed); assert(record.ok);
      const context = await browser.newContext({ viewport: { width: 320, height: 710 }, hasTouch: true, reducedMotion: 'reduce' });
      await context.addInitScript(({ key, value }) => { if (!sessionStorage.seeded) { localStorage.setItem(key, value); sessionStorage.seeded = '1'; } }, { key: Storage.SAVE_KEY, value: record.text });
      const page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.message));
      await page.clock.install({ time: new Date(1000) });
      await page.goto('http://127.0.0.1:' + server.address().port + '/assets/wayfarers/index.html'); await page.locator('.wx-game').waitFor(); await page.clock.runFor(1000);
      await page.addStyleTag({ content: scaleText }); await page.setViewportSize({ width, height });
      let g = await geometry(page, 'inspect-' + width); assert.equal(g.step, 'inspect');
      await page.mouse.click(g.target.x + g.target.width / 2, g.target.y + g.target.height / 2); await page.clock.runFor(500);
      // The current world intentionally hides purchases. Inspect follows the
      // actual drawer, skill detail and close controls before the rank lesson.
      for (let n = 0; n < 6 && await page.locator('.wx-guide[open]').getAttribute('data-step') === 'inspect'; n++) {
        g = await geometry(page, 'inspect-route-' + width + '-' + n);
        await page.mouse.click(g.target.x + g.target.width / 2, g.target.y + g.target.height / 2); await page.clock.runFor(500);
      }
      g = await geometry(page, 'buy-' + width); assert.equal(g.step, 'upgrade');
      await page.setViewportSize({ width: 320, height: 710 }); await geometry(page, 'portrait-resume-' + width);
      await page.setViewportSize({ width, height }); g = await geometry(page, 'rotated-buy-' + width);
      await page.mouse.click(g.target.x + g.target.width / 2, g.target.y + g.target.height / 2); await page.clock.runFor(500);
      g = await geometry(page, 'operation-' + width); assert.equal(g.step, 'operate');
      await page.mouse.click(g.target.x + g.target.width / 2, g.target.y + g.target.height / 2); await page.clock.runFor(1000);
      for (let n = 0; n < 6 && await page.locator('.wx-guide[open]').count(); n++) {
        g = await geometry(page, 'objective-route-' + width + '-' + n);
        await page.mouse.click(g.target.x + g.target.width / 2, g.target.y + g.target.height / 2); await page.clock.runFor(500);
      }
      const saved = await page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; });
      assert.equal(saved.onboarding.practice.progress.greenway, 3); assert.equal(saved.stations.ranks['station:greenway:path:pathfinding'], 1);
      assert.equal(saved.onboarding.practice.supplies.filter(id => id === 'greenway:upgrade').length, 1);
      await context.close();
    }
    assert.deepEqual(report.errors, []);
  } finally {
    await browser.close(); await new Promise(resolve => server.close(resolve));
    fs.writeFileSync(path.join(output, 'compact-guide-browser.json'), JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify(report, null, 2));
}
run().catch(error => { console.error(error); process.exitCode = 1; });
