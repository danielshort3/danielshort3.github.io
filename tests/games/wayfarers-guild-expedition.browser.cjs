'use strict';

// Real controls and the exact offline APK bundle, with isolated disposable saves.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(), 'guild-expedition-')));
const files = path.join(output, 'expedition-bundle');
const evidence = { opening: [], sessions: [], retained: [], charters: [], errors: [] };
const anchors = ['.wx-header', '.wx-objective', '.wx-world', '.wx-tray', '.wx-nav'];

function command(type, id) { return '[data-wx-do=' + JSON.stringify(JSON.stringify(id === undefined ? { type } : { type, id })) + ']'; }
async function geometry(page) {
  return page.evaluate(selectors => Object.fromEntries(selectors.map(selector => {
    const box = document.querySelector(selector).getBoundingClientRect();
    return [selector, { x: box.x, y: box.y, width: box.width, height: box.height }];
  })), anchors);
}
async function state(page) {
  return page.evaluate(() => {
    document.dispatchEvent(new Event('freeze'));
    return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;
  });
}
async function layout(page, label) {
  const result = await page.evaluate(() => {
    const sheet = document.querySelector('.wx-sheet[open]');
    const targets = [...document.querySelectorAll('.wx-game button, .wx-sheet[open] button')].filter(button => button.getClientRects().length && (!sheet || sheet.contains(button))).map(button => {
      const box = button.getBoundingClientRect();
      return { name: button.getAttribute('aria-label') || button.textContent.trim().slice(0, 50), width: box.width, height: box.height };
    });
    return { width: document.documentElement.scrollWidth - innerWidth, height: document.body.scrollHeight - innerHeight, sheetWidth: sheet ? sheet.scrollWidth - sheet.clientWidth : 0, targets };
  });
  assert(result.width <= 1 && result.height <= 1 && result.sheetWidth <= 1, label + ': viewport overflow ' + JSON.stringify(result));
  assert.deepEqual(result.targets.filter(target => target.width < 47.5 || target.height < 47.5), [], label + ': touch target below 48 px');
}
async function dismissFind(page) {
  const keep = page.locator('[data-dismiss-find]');
  if (await keep.isVisible()) await keep.click();
}
async function closeSheet(page) {
  if (await page.locator('.wx-sheet[open]').count()) await page.locator('[data-wx-close]').click();
}
async function areaPicker(page) {
  await closeSheet(page);
  await page.locator('[data-wx-nav="expedition"]').click();
  await page.locator('[data-wx-objective]').click();
}
async function selectArea(page,id) {
  await areaPicker(page);
  await page.locator('[data-wx-area="' + id + '"]').click();
}
async function filterArea(page, id) {
  const picker = page.locator('[data-wx-do="upgrade-areas"]');
  if (await picker.isVisible()) { await picker.click(); await page.locator('.wx-sheet [data-wx-do="filter:area:' + id + '"]').click(); }
  else await page.locator('[data-wx-do="filter:area:' + id + '"]').click();
}
async function choose(page, id) {
  await page.locator('[data-wx-do="world-choice"]').click();
  await clickAction(page, 'expedition-choice', id);
  await closeSheet(page);
}
async function screen(page, filename) { await page.screenshot({ path: path.join(output, filename + '.png') }); }
async function paint(page) {
  // Visibility observers settle on the browser's compositor before the next
  // clock-driven canvas frame after returning from a full-screen menu.
  await page.waitForTimeout(35); await page.clock.runFor(50);
  await page.waitForTimeout(35); await page.clock.runFor(50);
}
async function clickAction(page, type, id) {
  const key = await page.locator('.wx-sheet[open] [data-wx-do]').evaluateAll((buttons, wanted) => {
    for (const button of buttons) {
      try { const action = JSON.parse(button.dataset.wxDo); if (action.type === wanted.type && action.id === wanted.id && !button.disabled) return button.dataset.wxDo; } catch (error) {}
    }
    return null;
  }, { type, id });
  assert(key, 'rendered action available: ' + type + ':' + id);
  await page.locator('.wx-sheet[open] [data-wx-do=' + JSON.stringify(key) + ']').click();
}

async function run() {
  fs.mkdirSync(output, { recursive: true });
  bundle(files);
  const server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname).replace(/^\/assets\//, '/');
    const file = path.resolve(files, '.' + pathname);
    if (!file.startsWith(files + path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file, (error, bytes) => {
      if (error) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.webp': 'image/webp' }[path.extname(file)] || 'application/json');
      response.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = 'http://127.0.0.1:' + server.address().port;
  const browser = await chromium.launch({ headless: true });
  async function open(width, height, savedState) {
    const context = await browser.newContext({ viewport: { width, height }, reducedMotion: 'reduce' });
    // Preserve the released economy's natural flow until an explicit reset.
    // New six-area progression has its own rendered-input suite.
    if (!savedState) savedState = Core.migrateState(require('./fixtures/wayfarers-v4-fresh.json'));
    if (savedState) {
      const saved = JSON.parse(JSON.stringify(savedState));
      saved.lastUpdate = 1000;
      Core.act(saved, { type: 'introduction-seen', ids: Core.getPresentation(saved).introductions.map(item => item.id) });
      Core.act(saved, { type: 'discovery-seen', seq: saved.luck.ledger.seq });
      const envelope = Storage.createStore({ storage: null, now: () => 1000 }).export(saved);
      assert(envelope.ok, 'fixture can be exported');
      await context.addInitScript(({ key, value }) => { if (!sessionStorage.getItem('guild-fixture-loaded')) { localStorage.setItem(key, value); sessionStorage.setItem('guild-fixture-loaded', 'true'); } }, { key: Storage.SAVE_KEY, value: envelope.text });
    }
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400 && !response.url().endsWith('favicon.ico')) errors.push(response.status() + ' ' + response.url()); });
    await page.clock.install({ time: new Date(1000) });
    await page.goto(base + '/assets/wayfarers/index.html');
    await page.locator('.wx-game').waitFor();
    await page.clock.runFor(50);
    await page.locator('[data-wx-canvas][data-scene-status="ready"]').waitFor();
    assert.equal(await page.title(), 'Wayfarers’ Guild');
    return { context, page, errors };
  }
  try {
    for (const [width, height] of [[320, 740], [390, 844], [800, 480], [915, 390]]) {
      const { context, page, errors } = await open(width, height);
      assert.equal(await page.locator('[data-wx-buy]').count(), 1, 'one opening upgrade');
      assert.equal(await page.locator('.wx-nav button:visible').count(), 1, 'only the Trail is introduced');
      const before = await geometry(page);
      await layout(page, 'opening ' + width);
      await screen(page, 'network-opening-' + width);
      await page.clock.runFor(8000);
      await page.locator('[data-wx-buy="boots"]').click();
      assert.deepEqual(await geometry(page), before, 'first upgrade preserves the complete playfield');
      await page.locator('[data-wx-do="local:boots"]').click();
      assert.match(await page.locator('.wx-sheet').innerText(), /→|travel|milestone/i);
      await layout(page, 'opening details ' + width);
      await closeSheet(page);
      for (let i = 0; i < 6; i += 1) {
        await page.clock.runFor(5000);
        const enabled = page.locator('[data-wx-buy]:enabled').first();
        if (await enabled.count()) await enabled.click();
      }
      assert((await page.locator('[data-wx-buy]').count()) >= 2, 'new tracks arrive through normal play');
      assert.deepEqual(await geometry(page), before, 'new tracks do not push the scene');
      const textFits = await page.locator('.wx-upgrade-info').evaluateAll(nodes => nodes.every(node => node.scrollHeight <= node.clientHeight + 1 && [...node.querySelectorAll('small,strong')].every(label => label.getBoundingClientRect().bottom <= node.getBoundingClientRect().bottom + 1)));
      assert(textFits, 'upgrade label and effect stay above purchase price at ' + width);
      await layout(page, 'earned upgrades ' + width);
      await screen(page, 'network-upgrades-' + width);
      const saved = await state(page);
      await page.reload(); await page.locator('.wx-game').waitFor(); await page.clock.runFor(50);
      const restored = await state(page);
      assert.equal(restored.createdAt, saved.createdAt);
      assert.deepEqual(restored.expedition.areas.greenway.ranks, saved.expedition.areas.greenway.ranks);
      assert.deepEqual(errors, []);
      evidence.opening.push({ width, height, geometry:before, ranks:saved.expedition.areas.greenway.ranks });
      await context.close();
    }

    const { context, page, errors } = await open(390, 844);
    let stage = 0, purchases = 0, lastPurchase = 0, longestGap = 0;
    const started = [0], selected = new Set(), stages = [], choices = [];
    const ids = ['greenway','quarry','watchtower'];
    for (let seconds = 2; seconds <= 900 && stage < 3; seconds += 2) {
      await page.clock.runFor(2000); await dismissFind(page);
      const saved = await state(page);
      if (saved.expedition.completed && saved.expedition.index === stage) {
        const prior = JSON.parse(JSON.stringify(saved.expedition.areas[ids[stage]].ranks));
        stages.push({ stage, seconds, purchases:saved.expedition.purchases, ranks:prior });
        await layout(page, 'established ' + stage); await screen(page, 'network-established-' + stage);
        if (stage === 1) {
          await page.locator('[data-wx-do="unlock-project:development:trail-caravans"]').click();
          assert.equal((await state(page)).expedition.selectedArea,'greenway','Quarry discovery returns to the earlier Trail');
          assert.match(await page.locator('.wx-sheet').innerText(),/Caravan routes/);
          await layout(page,'earned cross-area discovery'); await screen(page,'network-earned-caravan-discovery');
          await clickAction(page,'expedition-development','trail-caravans'); await closeSheet(page); await paint(page);
          assert((await state(page)).expedition.developments.includes('trail-caravans'),'earlier-area capability bought with naturally earned supplies');
          await screen(page,'network-earned-trail-transformation');
        }
        if (stage < 2) {
          await page.locator('body:has(.wx-sheet[open]) .wx-sheet[open] ' + command('expedition-next') + ', body:not(:has(.wx-sheet[open])) .wx-world-actions ' + command('expedition-next')).click();
          await page.clock.runFor(50);
          const next = await state(page);
          assert.deepEqual(next.expedition.areas[ids[stage]].ranks, prior, 'unlocking another area retains all previous ranks');
          assert(next.expedition.areas[ids[stage]].established);
          await screen(page, 'network-opened-' + (stage + 1));
        }
        stage += 1; started[stage] = seconds; continue;
      }
      await closeSheet(page);
      if (await page.locator('[data-wx-do="world-choice"]').count()) {
        let id = !selected.has(stage) ? ['short','throughput','repair'][stage] : null;
        if (stage === 2 && seconds - started[stage] > 30 && !selected.has('guard')) { id = 'protect'; selected.add('guard'); }
        if (id) { await choose(page,id); selected.add(stage); choices.push({ stage,id,seconds }); }
      }
      const available = await page.locator('[data-wx-buy]:enabled').evaluateAll(buttons => buttons.map(button => ({ id:button.dataset.wxBuy, level:Number(button.closest('article').querySelector('[data-wx-rank]').textContent.split('/')[0].trim()) })).sort((a,b) => a.level - b.level));
      if (available.length) {
        await page.locator('[data-wx-buy="' + available[0].id + '"]').click();
        purchases += 1; longestGap = Math.max(longestGap, seconds - lastPurchase); lastPurchase = seconds;
      }
      if (seconds % 120 === 0) { await layout(page,'natural ' + seconds); await screen(page,'network-session-' + seconds); }
    }
    assert.equal(stage,3,'three areas establish through naturally earned visible purchases');
    assert(purchases >= 20,'many useful upgrades in the opening'); assert(longestGap < 90,'no unexplained 90-second purchase drought');
    await closeSheet(page);
    const built = await state(page);
    assert.equal(await page.locator('.wx-nav button:visible').count(),3,'Areas, Upgrades and Guild remain full-size destinations');
    for (const id of ids) {
      await selectArea(page,id);
      await page.clock.runFor(50);
      assert.equal(await page.locator('[data-wx-canvas]').getAttribute('data-scene-kind'),id);
      assert.equal(await page.locator('[data-wx-canvas]').getAttribute('data-scene-established'),'true');
      assert(await page.locator('[data-wx-do="world-choice"]').isVisible(),'completed frontier must leave the selected area working plans directly accessible');
      assert(await page.locator('.wx-world-actions ' + command('expedition-next')).isVisible(),'expansion remains available beside the working plan');
      const workingBefore = await state(page);
      await page.locator('[data-wx-do="world-choice"]').click();
      assert.equal(await page.locator('.wx-sheet[open]').getAttribute('data-kind'),'choice');
      const plan = {greenway:'freight',quarry:'quality',watchtower:'balanced'}[id];
      await clickAction(page,'expedition-choice',plan); await closeSheet(page);
      const workingAfter = await state(page);
      assert.equal(workingAfter.expedition.areas[id].choices[{greenway:'dispatch',quarry:'smelting',watchtower:'allocation'}[id]],plan,'completed-area working plan is applied and saved');
      assert.equal(workingAfter.expedition.index,workingBefore.expedition.index,'working plan does not expand or restart the frontier');
      assert.equal(workingAfter.expedition.cleared,workingBefore.expedition.cleared);
      assert.equal(workingAfter.expedition.completed,true);
      for (const other of ids.filter(key => key !== id)) assert.deepEqual(workingAfter.expedition.areas[other].choices,workingBefore.expedition.areas[other].choices,'working plan only changes the selected area');
      const switched = await state(page);
      for (const kept of ids) assert.deepEqual(switched.expedition.areas[kept].ranks,built.expedition.areas[kept].ranks,'area visits retain ' + kept);
      await layout(page,'persistent ' + id); await screen(page,'network-persistent-' + id);
    }
    for (const [width,height] of [[320,740],[915,390]]) {
      await page.setViewportSize({width,height}); await paint(page);
      const frame = await geometry(page);
      for (const id of ids) {
        await selectArea(page,id); await paint(page);
        assert.deepEqual(await geometry(page),frame,'completed-area navigation keeps the scene anchored at ' + width);
        assert(await page.locator('[data-wx-do="world-choice"]').isVisible());
        assert.equal(await page.locator('.wx-world-actions ' + command('expedition-next')).getAttribute('aria-label'),'Expand Greenway');
        assert(await page.locator('.wx-world-actions button').evaluateAll(nodes=>nodes.every(node=>node.scrollWidth<=node.clientWidth+1)),'Plans and Expand fit the existing action row');
        await layout(page,'completed plans ' + id + ' ' + width); await screen(page,'network-completed-plans-' + id + '-' + width);
      }
    }
    const beforeIdle = await state(page);
    await page.locator('[data-wx-nav="upgrades"]').click(); await page.clock.runFor(20000);
    const afterIdle = await state(page);
    for (const id of ids) assert(afterIdle.expedition.areas[id].elapsed > beforeIdle.expedition.areas[id].elapsed,'hidden area keeps advancing: ' + id);
    for (const resource of ['coins','ore','knowledge']) assert(Core.Numbers.cmp(afterIdle.resources[resource],beforeIdle.resources[resource]) > 0,resource + ' produced while global menu is open');
    await layout(page,'earned global upgrades'); await screen(page,'network-earned-global');
    await page.reload(); await page.locator('.wx-game').waitFor(); await page.clock.runFor(50);
    const reload = await state(page);
    assert.deepEqual(reload.expedition.areas.greenway.ranks,built.expedition.areas.greenway.ranks);
    assert.deepEqual(reload.expedition.areas.quarry.ranks,built.expedition.areas.quarry.ranks);
    assert.deepEqual(reload.expedition.areas.watchtower.ranks,built.expedition.areas.watchtower.ranks);
    assert.deepEqual(errors,[]); evidence.sessions.push({ stages,purchases,longestGap,choices,hiddenProduction:true });
    await context.close();

    const legacy = Core.migrateState(require('./fixtures/wayfarers-v3-state.json'));
    assert(legacy && Core.validateState(legacy).valid,'old guild migrates into persistent areas');
    for (const resource of ['coins','ore','knowledge','maps','herbs','provisions']) legacy.resources[resource] = Core.Numbers.from(1e12);
    // Mature catalog/causality fixture is isolated and funded. The preceding
    // three-area session never grants resources or edits state during play.
    for (const [width,height] of [[320,740],[390,844],[800,480],[915,390]]) {
      const { context:ctx,page:p,errors:errs } = await open(width,height,legacy);
      await closeSheet(p); await p.locator('[data-wx-nav="upgrades"]').click();
      await layout(p,'global catalog ' + width); await screen(p,'network-catalog-' + width);
      assert.equal(await p.locator('.wx-destination select').count(),0,'catalog uses visible buttons');
      await filterArea(p, 'greenway');
      const targets = await p.locator('[data-wx-upgrade]').evaluateAll(nodes => nodes.map(node => node.dataset.wxUpgrade));
      assert(targets.includes('area:greenway:boots'),'area filter includes its exact persistent tracks');
      await filterArea(p, 'all');
      await p.locator('[data-wx-do="upgrade-filters"]').click();
      await layout(p,'effect filter sheet ' + width); await screen(p,'network-effect-filters-' + width);
      await p.locator('[data-wx-do="filter:effect:all"]').click();
      await p.locator('[data-wx-search]').fill('Caravan routes');
      const caravan = p.locator('[data-wx-upgrade="development:trail-caravans"]');
      assert.equal(await caravan.count(),1); await caravan.getByRole('button',{name:/Details/}).click();
      assert.match(await p.locator('.wx-sheet').innerText(),/Quarry|quarry/);
      assert.match(await p.locator('.wx-sheet').innerText(),/Locked.*Available/s);
      await layout(p,'development details ' + width); await screen(p,'network-development-details-' + width);
      await clickAction(p,'expedition-development','trail-caravans'); await closeSheet(p);
      assert((await state(p)).expedition.developments.includes('trail-caravans'),'canonical development purchased');
      await areaPicker(p);
      assert(await p.locator('[data-wx-area="greenway"] .wx-new').isVisible(),'new earlier-area development is marked');
      await p.locator('[data-wx-area="greenway"]').click(); await paint(p);
      await areaPicker(p);
      assert(!(await p.locator('[data-wx-area="greenway"] .wx-new').isVisible()),'visiting an area acknowledges its new feature');
      assert(await p.locator('[data-wx-area="quarry"] .wx-new').isVisible(),'visiting Trail does not erase Quarry attention');
      await closeSheet(p);
      assert((await p.locator('[data-wx-canvas]').getAttribute('data-scene-developments')).includes('trail-caravans'));
      await choose(p,'freight');
      assert.equal((await state(p)).expedition.areas.greenway.choices.dispatch,'freight');
      await screen(p,'network-trail-caravans-' + width);
      await p.locator('[data-wx-do="world-choice"]').click(); await layout(p,'freight choices ' + width); await screen(p,'network-freight-choices-' + width); await closeSheet(p);
      await p.locator('[data-wx-nav="upgrades"]').click(); await p.locator('[data-wx-search]').fill('Precision smelting');
      const precision = p.locator('[data-wx-upgrade="development:quarry-precision"]');
      await precision.locator('.wx-price').click();
      await selectArea(p,'quarry'); await paint(p);
      assert((await p.locator('[data-wx-canvas]').getAttribute('data-scene-developments')).includes('quarry-precision'));
      assert.match(await p.locator('[data-wx-local-count]').innerText(),/Ore \/s/,'established Quarry displays ongoing production');
      await choose(p,'precision');
      assert.equal((await state(p)).expedition.areas.quarry.choices.smelting,'precision');
      await layout(p,'transformed quarry ' + width); await screen(p,'network-quarry-precision-' + width);
      await p.locator('[data-wx-do="world-choice"]').click(); await layout(p,'precision choices ' + width); await screen(p,'network-precision-choices-' + width);
      assert((await p.locator('.wx-choice-impact').evaluateAll(nodes => nodes.every(node => node.children.length <= 3))),'choice previews keep only three tradeoff rows');
      await p.locator('[data-wx-do="choice-metrics:smelting"]').click();
      assert.match(await p.locator('.wx-sheet').innerText(),/Sustainable raw processing/);
      await layout(p,'full processing comparison ' + width); await screen(p,'network-full-comparison-' + width); await closeSheet(p);
      await p.locator('[data-wx-nav="upgrades"]').click(); await p.locator('[data-wx-search]').fill('Boots');
      const local = p.locator('[data-wx-upgrade="area:greenway:boots"]');
      const oldBoots = (await state(p)).expedition.areas.greenway.ranks.boots;
      await local.locator('.wx-price').click();
      assert.equal((await state(p)).expedition.areas.greenway.ranks.boots,oldBoots + 1,'global menu targets Trail even while Quarry selected');
      await p.locator('[data-wx-search]').fill('Tools');
      const equipment = p.locator('[data-wx-upgrade="guild:buy:gear-tools"]');
      assert.equal(await equipment.count(),1,'canonical equipment row is present');
      const before = (await state(p)).upgrades['gear-tools'];
      await equipment.locator('.wx-price').click(); await equipment.locator('.wx-price').click();
      assert.equal((await state(p)).upgrades['gear-tools'],before + 2,'owned equipment remains repeatable');
      await p.locator('[data-wx-search]').fill('regional atlas');
      assert.equal(await p.locator('[data-wx-upgrade="guild:project:chapter-survey"]').count(),1,'regional projects remain reachable in the comprehensive catalog');
      await p.locator('[data-wx-search]').fill('Relic lore');
      assert.equal(await p.locator('[data-wx-upgrade="guild:luck-research:relic-lore"]').count(),1,'relic research remains reachable in the comprehensive catalog');
      await p.locator('[data-wx-nav="guild"]').click();
      await p.getByRole('button',{name:/Crew.*Explorers/}).click(); await layout(p,'mature crew ' + width); await screen(p,'network-crew-' + width); await closeSheet(p);
      await p.getByRole('button',{name:/Automation/}).click(); await layout(p,'mature automation ' + width); await screen(p,'network-automation-' + width); await closeSheet(p);
      await p.locator('[data-wx-do="guild-atlas"]').click(); await layout(p,'mature atlas ' + width); await screen(p,'network-atlas-' + width);
      const saved = await state(p); await p.reload(); await p.locator('.wx-game').waitFor(); await p.clock.runFor(50); const restored = await state(p);
      assert.equal(restored.createdAt,legacy.createdAt); assert.deepEqual(restored.expedition.developments,saved.expedition.developments);
      for (const id of ids) { assert.deepEqual(restored.expedition.areas[id].ranks,saved.expedition.areas[id].ranks); assert.deepEqual(restored.expedition.areas[id].choices,saved.expedition.areas[id].choices); }
      assert.deepEqual(errs,[]); evidence.retained.push({ width,height,createdAt:saved.createdAt,developments:saved.expedition.developments }); await ctx.close();
    }
    if (process.env.WAYFARERS_QA_CHARTER_FIXTURE) {
      const preCharter = JSON.parse(fs.readFileSync(process.env.WAYFARERS_QA_CHARTER_FIXTURE,'utf8'));
      assert(Core.validateState(preCharter).valid,'naturally played Charter review fixture is valid');
      const {context:ctx,page:p,errors:errs} = await open(320,740,preCharter);
      await closeSheet(p); await p.locator('[data-wx-nav="upgrades"]').click();
      const gates = ['trail-prospectors','tower-control-room','survey-exchange'];
      for (const [index,id] of gates.entries()) {
        const item = Core.getView(preCharter).globalUpgrades.find(entry => entry.id === 'development:' + id);
        await p.locator('[data-wx-search]').fill(item.name || item.label);
        await p.locator('[data-wx-upgrade="development:' + id + '"] .wx-research-info').click();
        assert.match(await p.locator('.wx-sheet').innerText(),new RegExp('Earn ' + (index + 1) + ' Guild Charter'));
        assert.equal(await p.locator('.wx-sheet .wx-detail-hero strong').innerText(),'Locked','locked Charter capability must not advertise availability');
        assert(await p.locator('.wx-sheet .wx-confirm').isDisabled(),'Charter-only capability cannot be bought early');
        await layout(p,'Charter requirement ' + id); await screen(p,'network-charter-gate-' + (index + 1)); await closeSheet(p);
      }
      await p.locator('[data-wx-nav="guild"]').click(); await p.locator('[data-wx-do="guild-atlas"]').click();
      await p.getByRole('button',{name:/Renewals.*Field notes/}).click();
      await p.getByRole('button',{name:/Guild Charter.*guild crests/}).click();
      await p.getByRole('button',{name:'Review reset',exact:true}).click();
      assert.match(await p.locator('[data-dialog-body]').innerText(),/Every area, its upgrade ranks, queues/);
      const capability = p.locator('[data-dialog-body] li').filter({hasText:'Prospecting network'});
      await capability.scrollIntoViewIfNeeded(); await screen(p,'network-charter-preview');
      assert.deepEqual(errs,[]); evidence.charters.push({width:320,gates,preview:'Prospecting network',createdAt:preCharter.createdAt}); await ctx.close();
    }
    fs.writeFileSync(path.join(output,'expedition-browser-evidence.json'),JSON.stringify(evidence,null,2));
    console.log('Persistent guild browser QA passed: four viewport openings, naturally established areas, retained builds, hidden production, global catalog purchases and mature menus. ' + output);
  } finally {
    fs.writeFileSync(path.join(output, 'expedition-browser-partial.json'), JSON.stringify(evidence, null, 2));
    await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
run().catch(error => { console.error(error); process.exitCode = 1; });
