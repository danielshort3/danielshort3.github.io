'use strict';

// Exact offline bundle. Funded fixtures isolate interaction correctness from pacing.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-tiers-')));
const evidence = { viewports:[], flows:[], errors:[], browser:'Browser plugin not available; existing Playwright workflow used.' };
const clone = value => JSON.parse(JSON.stringify(value));
const key = (page,id) => page.locator('[data-wx-do=' + JSON.stringify(id) + ']:visible');
const sheet = page => page.locator('.wx-sheet[open]');
async function close(page) { if (await sheet(page).count()) await page.locator('[data-wx-close]').click(); }
async function save(page) { return page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; }); }
async function shot(page,name) { await page.clock.runFor(50); await page.waitForTimeout(50); await page.screenshot({path:path.join(output,name+'.png')}); }
async function bounds(page) { return Promise.all(['.wx-header','.wx-world','.wx-dock','.wx-nav'].map(selector => page.locator(selector).boundingBox())); }
async function geometry(page) {
  const result = await page.evaluate(() => {
    const dialog = document.querySelector('.wx-sheet[open]');
    const controls = [...(dialog || document.querySelector('.wx-game')).querySelectorAll('button')].filter(node => node.getClientRects().length).map(node => ({label:node.getAttribute('aria-label') || node.textContent.trim(),width:node.getBoundingClientRect().width,height:node.getBoundingClientRect().height}));
    const footer = dialog?.querySelector('.wx-purchase-footer:not([hidden])')?.getBoundingClientRect();
    return { overflow:document.documentElement.scrollWidth-innerWidth, height:document.body.scrollHeight-innerHeight, controls, footer:footer ? {top:footer.top,bottom:footer.bottom} : null, viewport:innerHeight };
  });
  assert(result.overflow <= 1 && result.height <= 1,'Viewport fits ' + JSON.stringify(result));
  assert.deepEqual(result.controls.filter(item => item.width < 47.5 || item.height < 47.5),[],'48px targets');
  if (result.footer) assert(result.footer.top >= 0 && result.footer.bottom <= result.viewport+1,'Footer is visible');
  return result;
}
function ready(state) { return H.Core.getView(state).expedition.tiers.ready; }
function claimAll(state) {
  for (let count = 0; count < 100 && ready(state).length; count += 1) for (const tier of ready(state)) assert(H.Core.act(state,tier.unlockAction).ok);
}
function firstReady() {
  const state = H.Core.createState(1000); H.fund(state);
  for (let i = 0; i < 2; i += 1) {
    const card = H.Core.getView(state).expedition.cards.find(item => item.visible !== false);
    assert(H.Core.act(state,card.action).ok);
  }
  assert(ready(state).length,'First earned tier is ready');
  return state;
}
async function run() {
  fs.mkdirSync(output,{recursive:true});
  const files = path.join(output,'tier-bundle'); bundle(files);
  const server = http.createServer((request,response) => {
    const pathname = decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/');
    const file = path.resolve(files,'.'+pathname);
    if (!file.startsWith(files+path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file,(error,bytes) => {
      if (error) { response.writeHead(404).end(); return; }
      response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'})[path.extname(file)] || 'application/json'); response.end(bytes);
    });
  });
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const browser = await chromium.launch({headless:true});
  async function open(width,height,state,offlineSeconds=0) {
    const context = await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    state = clone(state); state.lastUpdate = 1000;
    H.Core.act(state,{type:'introduction-seen',ids:H.Core.getPresentation(state).introductions.map(item => item.id)});
    H.Core.act(state,{type:'discovery-seen',seq:state.luck.ledger.seq});
    const record = Storage.createStore({storage:null,now:()=>1000}).export(state); assert(record.ok,record.message);
    await context.addInitScript(({key,value}) => { if (!sessionStorage.getItem('seeded')) { localStorage.setItem(key,value); sessionStorage.setItem('seeded','1'); } },{key:Storage.SAVE_KEY,value:record.text});
    const page = await context.newPage(); page.setDefaultTimeout(10000); page.on('pageerror',error => evidence.errors.push(error.message));
    await page.clock.install({time:new Date(1000+offlineSeconds*1000)});
    await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html'); await page.locator('.wx-game').waitFor(); await page.clock.runFor(50);
    return {context,page};
  }
  try {
    for (const [width,height] of [[320,740],[390,844],[915,390]]) {
      const fresh = H.Core.createState(1000); H.fund(fresh);
      const {context,page} = await open(width,height,fresh);
      assert.equal(await page.locator('.wx-upgrade').count(),1);
      assert.equal(await page.locator('[data-wx-tier-ready]:visible').count(),0);
      const anchors = await bounds(page); await shot(page,'fresh-'+width);
      await page.locator('[data-wx-buy="boots"]').click(); await page.locator('[data-wx-buy="boots"]').click();
      assert.equal(await page.locator('.wx-upgrade').count(),1,'Readiness does not reveal the unclaimed row');
      assert.deepEqual(await bounds(page),anchors,'Readiness keeps world and dock fixed');
      await page.clock.runFor(1500);
      assert.equal(await sheet(page).getAttribute('data-kind'),'tier');
      await geometry(page); await shot(page,'tier-ready-'+width);
      await key(page,'tier-later').click(); await page.clock.runFor(3000);
      assert.equal(await sheet(page).count(),0,'Later does not immediately reopen');
      assert.equal(await page.locator('[data-wx-tier-ready]:visible').count(),1);
      await page.reload(); await page.locator('.wx-game').waitFor(); await page.clock.runFor(3000);
      assert.equal(await sheet(page).count(),0,'Durable prompted state survives reload');
      await page.locator('[data-wx-tier-ready]').click();
      await page.locator('[data-wx-do^="tier-unlock:"]').click();
      assert.equal(await sheet(page).count(),0);
      assert.equal(await page.locator('.wx-upgrade').count(),2);
      assert.deepEqual(await bounds(page),anchors,'Explicit tier unlock keeps world and dock fixed');
      const state = await save(page); assert.equal(ready(state).length,0);
      await page.locator('[data-wx-buy="porters"]').click();
      assert.equal((await save(page)).expedition.areas.greenway.ranks.porters,1);
      await geometry(page); await shot(page,'tier-unlocked-'+width); evidence.viewports.push({width,height,geometry:await geometry(page)});
      await page.locator('[data-wx-buy="porters"]').click(); await page.clock.runFor(1500);
      assert.equal(await sheet(page).getAttribute('data-kind'),'tier','Real new player investment re-arms the next earned tier in the same session');
      assert.match(await sheet(page).innerText(),/Scouting/); await shot(page,'new-investment-tier-'+width); await key(page,'tier-later').click();
      await context.close();
    }
    evidence.flows.push('actual first-tier readiness, Later, reload, explicit unlock and new purchase preserve world geometry; later real investment re-arms a new notice in the same session');
    const waiting = firstReady();
    const {context,page} = await open(390,844,waiting);
    await page.locator('[data-wx-options]').click(); await page.clock.runFor(4000);
    assert.equal(await sheet(page).getAttribute('data-kind'),'options','Readiness never interrupts an existing sheet');
    await close(page); await page.clock.runFor(2000); assert.equal(await sheet(page).getAttribute('data-kind'),'tier');
    await page.keyboard.press('Escape'); await page.clock.runFor(3000); assert.equal(await sheet(page).count(),0,'Android/keyboard Back behaves like Later');
    await context.close(); evidence.flows.push('open sheet and Back protection');
    for (const [width,height] of [[320,740],[915,390]]) {
      const offline = await open(width,height,firstReady(),3600);
      const p = offline.page;
      await p.clock.runFor(1500);
      assert.equal(await sheet(p).getAttribute('data-kind'),'return','Offline return takes priority');
      await key(p,'return-close').click(); await p.clock.runFor(2000);
      assert.equal(await sheet(p).getAttribute('data-kind'),'tiers');
      const state = await save(p);
      assert(ready(state).length > 1,'Real offline work creates a multi-tier backlog');
      assert.equal(await p.locator('.wx-tier-ready-list .wx-menu').count(),ready(state).length);
      await geometry(p); await shot(p,'offline-ready-summary-'+width);
      await key(p,'tier-later').click(); await p.clock.runFor(5000); assert.equal(await sheet(p).count(),0,'No popup storm after the coalesced summary');
      await p.reload(); await p.locator('.wx-game').waitFor(); await p.clock.runFor(2000);
      assert.equal(await sheet(p).count(),0,'Backlog acknowledgement is durable');
      await p.locator('[data-wx-nav="upgrades"]').click();
      assert.equal(await key(p,'ready-tiers').count(),1,'Global catalog keeps a direct readiness route');
      assert.equal(await key(p,'filter:area:harbor').count(),0,'Future area filter is absent');
      await p.locator('[data-wx-search]').fill('Railways');
      assert.equal(await p.locator('[data-wx-upgrade]').count(),0,'Future track cannot be found through search');
      await shot(p,'hidden-future-search-'+width);
      await p.locator('[data-wx-search]').fill(''); await key(p,'ready-tiers').click();
      const quarry = ready(await save(p)).find(item => item.id === 'area:quarry:carts'); assert(quarry);
      await key(p,'tier-review:'+quarry.id).click(); await geometry(p); await shot(p,'quarry-tier-'+width);
      await key(p,'tier-unlock:'+quarry.id).click();
      assert(!ready(await save(p)).some(item => item.id === quarry.id));
      await p.locator('[data-wx-nav="expedition"]').click();
      await p.clock.runFor(4000);
      assert.equal(await sheet(p).count(),0,'A newly eligible successor after claim remains on the badge, without chaining another popup');
      const currentArea = (await save(p)).expedition.selectedArea;
      if (currentArea !== 'quarry') { await p.locator('[data-wx-objective]').click(); await p.locator('[data-wx-area="quarry"]').click(); }
      assert.equal(await p.locator('[data-wx-buy="carts"]').count(),1);
      await p.locator('[data-wx-buy="carts"]').click(); assert.equal((await save(p)).expedition.areas.quarry.ranks.carts,1);
      await geometry(p); await shot(p,'quarry-unlocked-'+width); await offline.context.close();
      const old = require('./fixtures/wayfarers-v5-retained.json');
      const legacy = H.Core.normalizeState(old.state,1000); assert.equal(legacy.expedition.version,2); assert.equal(ready(legacy).length,0);
      const retained = await open(width,height,legacy); await retained.page.clock.runFor(2000); assert.equal(await sheet(retained.page).count(),0,'Migration creates no false readiness popup');
      assert.equal(await retained.page.locator('.wx-upgrade').count(),3,'Existing learned legacy tracks remain visible');
      assert.deepEqual((await save(retained.page)).expedition.areas[legacy.expedition.selectedArea].ranks,legacy.expedition.areas[legacy.expedition.selectedArea].ranks);
      await geometry(retained.page); await shot(retained.page,'legacy-retained-'+width); await retained.context.close();
    }
    evidence.flows.push('offline backlog coalescing; deferred direct route; later Quarry tier; invisible future search/filter; legacy expedition2 migration');
    const failure = await open(390,844,firstReady()); const p = failure.page;
    await p.clock.runFor(1600); assert.equal(await sheet(p).getAttribute('data-kind'),'tier');
    const before = await save(p);
    await p.evaluate(() => { window.tierSetItem=Storage.prototype.setItem; Storage.prototype.setItem=function(key,value) { if (key.startsWith('wayfarers-guild-save')) throw new DOMException('Quota full','QuotaExceededError'); return window.tierSetItem.call(this,key,value); }; });
    await p.locator('[data-wx-do^="tier-unlock:"]').click();
    assert.equal(await p.locator('.wx-upgrade').count(),1,'Failed durable claim does not leak newly buyable rows');
    assert(await key(p,'tier-retry').isVisible()); assert(await p.locator('[data-wx-buy="boots"]').isDisabled());
    await shot(p,'failed-tier-save');
    await p.evaluate(() => { Storage.prototype.setItem=window.tierSetItem; }); await key(p,'tier-retry').click();
    assert.equal(await p.locator('.wx-upgrade').count(),1,'Retry keeps the rolled-back claim unpurchased');
    await p.locator('[data-wx-do^="tier-unlock:"]').click();
    assert.equal(await p.locator('.wx-upgrade').count(),2);
    const after = await save(p); assert.equal(after.upgradeTiers.claimed.length,before.upgradeTiers.claimed.length+1);
    await p.reload(); await p.locator('.wx-game').waitFor(); assert.equal(await p.locator('.wx-upgrade').count(),2); await failure.context.close();
    evidence.flows.push('failed tier save rolls back visibility, blocks purchases, allows durable retry and single explicit claim');
    const noticeState = firstReady(); H.advance(noticeState,3600); assert(H.Core.act(noticeState,{type:'collection-unlock',kind:'cards'}).ok);
    assert(noticeState.collection.recent.length,'Older collection feedback exists in this fixture');
    const noticeFailure = await open(390,844,noticeState); const np = noticeFailure.page;
    await np.evaluate(() => { window.tierSetItem=Storage.prototype.setItem; Storage.prototype.setItem=function(key,value) { if (key.startsWith('wayfarers-guild-save')) throw new DOMException('Quota full','QuotaExceededError'); return window.tierSetItem.call(this,key,value); }; });
    await np.clock.runFor(1500);
    assert.equal(await sheet(np).count(),0,'A notice with failed durable acknowledgement is not shown');
    assert(await np.locator('[data-wx-save-alert]').isVisible());
    const unacknowledged = await np.evaluate(() => JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state.upgradeTiers);
    assert.equal(unacknowledged.prompted.length,0);
    await np.evaluate(() => { Storage.prototype.setItem=window.tierSetItem; }); await np.locator('[data-wx-save-alert]').click();
    assert.equal(await np.locator('.wx-sheet[open][data-kind="collection-result"]').count(),0,'Tier retry never opens unrelated older collection feedback');
    await np.clock.runFor(1500); assert.equal(await sheet(np).getAttribute('data-kind'),'tiers'); await geometry(np); await shot(np,'notice-ack-recovered'); await noticeFailure.context.close();
    evidence.flows.push('failed automatic notice acknowledgement stays unseen; retry preserves tiers and does not open unrelated collection feedback');
    assert.deepEqual(evidence.errors,[]);
  } finally {
    fs.writeFileSync(path.join(output,'tier-browser.json'),JSON.stringify(evidence,null,2));
    await browser.close(); await new Promise(resolve => server.close(resolve));
  }
  console.log(JSON.stringify({passed:true,output,viewports:evidence.viewports.length,flows:evidence.flows},null,2));
}
run().catch(error => { console.error(error); process.exitCode=1; });
