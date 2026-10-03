'use strict';

// The granted inventory is an explicit UI fixture, never acquisition or pacing evidence.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Onboarding = require('./helpers/wayfarers-onboarding.cjs');
const K = require('../../js/games/wayfarers-guild/collections');
const D = require('../../js/games/wayfarers-guild/collection-content');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-collections-')));
const evidence = {sourceHashes:Object.fromEntries([
  'js/games/wayfarers-guild/expedition-ui.js',
  'js/games/wayfarers-guild/onboarding-ui.js',
  'js/games/wayfarers-guild/practice-lessons.js',
  'js/games/wayfarers-guild/core.js',
  'css/games/wayfarers-guild.css'
].map(file=>[file,require('node:crypto').createHash('sha256').update(fs.readFileSync(path.resolve(__dirname,'../..',file))).digest('hex')])),viewports:[],flows:[],errors:[]};
const clone = value => JSON.parse(JSON.stringify(value));
const inventory = value => { const result=clone(value); for(const key of ['cardEligibleMs','cardRemainingMs','scrollEligibleMs','scrollRemainingMs']) delete result[key]; return result; };
const key = (page,id) => page.locator('[data-wx-do=' + JSON.stringify(id) + ']:visible');
const sheet = page => page.locator('.wx-sheet[open]');
async function openCollection(page, kind) {
  await page.locator('[data-wx-nav="guild"]').click();
  await key(page, 'guild-page:' + kind).click();
}
async function close(page) { if(await sheet(page).count()) await page.locator('[data-wx-close]').click(); }
async function save(page) { return page.evaluate(()=>{document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;}); }
async function shot(page,name) { await page.clock.runFor(50); await page.waitForTimeout(50); await page.screenshot({path:path.join(output,name+'.png')}); }
async function footer(page) { await page.locator('.wx-sheet[open] .wx-purchase-footer .wx-confirm').click(); }
async function geometry(page) {
  const result=await page.evaluate(()=>{
    const dialog=document.querySelector('.wx-sheet[open]');
    const controls=[...(dialog || document.querySelector('.wx-game')).querySelectorAll('button')].filter(node=>node.getClientRects().length).map(node=>({label:node.getAttribute('aria-label')||node.textContent.trim(),w:node.getBoundingClientRect().width,h:node.getBoundingClientRect().height}));
    const footer=dialog?.querySelector('.wx-purchase-footer:not([hidden])')?.getBoundingClientRect();
    return {overflow:document.documentElement.scrollWidth-innerWidth,height:document.body.scrollHeight-innerHeight,controls,footer:footer?{bottom:footer.bottom,top:footer.top}:null,viewport:innerHeight};
  });
  assert(result.overflow<=1 && result.height<=1,'Viewport fits '+JSON.stringify(result));
  assert.deepEqual(result.controls.filter(item=>item.w<47.5 || item.h<47.5),[],'48px target contract');
  if(result.footer) assert(result.footer.bottom<=result.viewport+1 && result.footer.top>=0,'Action footer remains visible');
  return result;
}
function collected(all=false) {
  const state=H.mature();
  for(const kind of ['cards','equipment']) assert(H.Core.act(state,{type:'collection-unlock',kind}).ok);
  state.collection.cards['trail-courier'].copies=12;state.collection.ink=20;state.collection.scrollRng=39;
  if(all) {
    for(const card of D.CARDS) state.collection.cards[card.id]={rank:1,copies:2};
    for(const gear of D.GEAR) state.collection.gear[gear.id]={successes:{steady:0,bold:0,brilliant:0},failed:0};
  }
  assert(H.Core.validateState(state).valid); return state;
}
function claimReadyTiers(state) {
  for (const tier of H.Core.getView(state).expedition.tiers?.ready || []) assert(H.Core.act(state,tier.unlockAction).ok);
}
function retainLearnedControls(state) {
  // Restore a controlled inventory after teaching on a copy, using the same
  // receipt merge as an older-backup import. Paid-operation fixtures therefore
  // keep their authored copies, points, rates and RNG instead of tutorial gains.
  const learned=clone(state);Onboarding.completeAreaGuides(learned);
  if(learned.collection.equipmentUnlocked && !Object.values(learned.collection.gear).some(item=>item.failed>0)) {
    learned.collection.gear['trail-boots'].failed=1;
  }
  H.Core.act(learned,{type:'collection-ack',sequence:learned.collection.sequence});
  for(const id of ['card-archive','card-craft','gear-craft','gear-repair','gear-reforge']) {
    const guide=H.Core.getView(learned).onboarding.guides.find(row=>row.id===id);
    if(!guide||guide.complete||!guide.available)continue;
    assert(H.Core.act(learned,guide.visitAction).ok);
    for(let i=0;i<20;i+=1){const active=H.Core.getView(learned).onboarding.active;if(!active)break;const result=H.Core.act(learned,active.ackAction||active.practiceAction||active.inspectAction);assert(result.ok,result.message);}
    assert(H.Core.getView(learned).onboarding.guides.find(row=>row.id===id).complete,'Paid-operation fixture already learned '+id);
  }
  H.Core.mergePracticeReceipts(state,learned);assert(H.Core.validateState(state).valid);
  return state;
}
async function run() {
  fs.mkdirSync(output,{recursive:true});const files=path.join(output,'collection-bundle');bundle(files);
  const server=http.createServer((request,response)=>{
    const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/');
    const file=path.resolve(files,'.'+pathname);
    if(!file.startsWith(files+path.sep)){response.writeHead(403).end();return;}
    fs.readFile(file,(error,bytes)=>{if(error){response.writeHead(404).end();return;}response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'})[path.extname(file)]||'application/json');response.end(bytes);});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
  async function open(width,height,state) {
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    state=clone(state);state.lastUpdate=1000;
    Onboarding.announceDiscoveries(retainLearnedControls(state));
    H.Core.act(state,{type:'introduction-seen',ids:H.Core.getPresentation(state).introductions.map(item=>item.id)});
    H.Core.act(state,{type:'discovery-seen',seq:state.luck.ledger.seq});
    const record=Storage.createStore({storage:null,now:()=>1000}).export(state);assert(record.ok,record.message);
    await context.addInitScript(({key,value})=>{if(!sessionStorage.getItem('seeded')){localStorage.setItem(key,value);sessionStorage.setItem('seeded','1');}},{key:Storage.SAVE_KEY,value:record.text});
    const page=await context.newPage();page.setDefaultTimeout(10000);page.on('pageerror',error=>evidence.errors.push(error.message));
    await page.clock.install({time:new Date(1000)});await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.locator('.wx-game').waitFor();await page.clock.runFor(50);return {context,page};
  }
  try {
    const two=H.Core.createState(0);H.fund(two);
    while(!two.expedition.completed) { claimReadyTiers(two); for(const [areaId,area] of Object.entries(two.expedition.areas)) for(const id of area.learned) if(area.ranks[id]<15) H.Core.act(two,{type:'expedition-buy',areaId,id});H.advance(two,20); }
    H.Core.act(two,{type:'expedition-next'});
    while(H.P.progress(two.expedition)<.96) { claimReadyTiers(two); for(const [areaId,area] of Object.entries(two.expedition.areas)) for(const id of area.learned) if(area.ranks[id]<15) H.Core.act(two,{type:'expedition-buy',areaId,id});H.advance(two,1); }
    claimReadyTiers(two);
    H.Core.act(two,{type:'collection-unlock',kind:'cards'});
    for(const [width,height] of [[320,740],[915,390]]) {
      const {context,page}=await open(width,height,two);await openCollection(page,'cards');assert.equal(await page.locator('.wx-deck-slot').count(),2);await shot(page,'two-slots-'+width);
      await page.locator('[data-wx-nav="expedition"]').click();const anchors=await Promise.all(['.wx-header','.wx-world','.wx-dock','.wx-nav'].map(selector=>page.locator(selector).boundingBox()));
      await page.clock.runFor(10000);await close(page);
      assert.deepEqual(await Promise.all(['.wx-header','.wx-world','.wx-dock','.wx-nav'].map(selector=>page.locator(selector).boundingBox())),anchors,'Third slot and earned Equipment tab do not shift the area');
      await openCollection(page,'cards');assert.equal(await page.locator('.wx-deck-slot').count(),3);await geometry(page);await shot(page,'three-slots-'+width);await context.close();
      const three=H.mature({untilProject:'deposit-maps'});H.Core.act(three,{type:'collection-unlock',kind:'cards'});H.fund(three);assert(H.Core.act(three,{type:'expedition-development',id:'workshop-foundation'}).ok);
      const project=H.P.Content.PROJECTS.find(item=>item.id==='workshop-foundation');H.P.tick(three,project.work/H.P.rawRates(three).researchRate-3);
      const fourth=await open(width,height,three);await openCollection(fourth.page,'cards');assert.equal(await fourth.page.locator('.wx-deck-slot').count(),3);const bounds=await fourth.page.locator('.wx-deck-slots').boundingBox();await fourth.page.clock.runFor(4000);assert.equal(await fourth.page.locator('.wx-deck-slot').count(),4);assert.deepEqual(await fourth.page.locator('.wx-deck-slots').boundingBox(),bounds,'Fourth slot retains deck geometry');await geometry(fourth.page);await shot(fourth.page,'fourth-slot-unlock-'+width);await fourth.context.close();
    }
    evidence.flows.push('actual second-to-third and third-to-fourth slot unlocks preserve fixed world/header/deck bounds at320 and915');
    for(const [width,height] of [[320,740],[915,390]]) {
      const fixture=collected();fixture.collection.gear['trail-boots']={successes:{steady:2,bold:1,brilliant:0},failed:1};fixture.collection.scrolls.restoration=3;H.Core.act(fixture,{type:'gear-equip',slot:'boots',id:'trail-boots'});
      const {context,page}=await open(width,height,fixture);await openCollection(page,'equipment');await key(page,'gear:trail-boots').click();await key(page,'menu:'+JSON.stringify({kind:'collection-transaction',entity:'gear',id:'trail-boots',operation:'reforge'})).click();
      assert.match(await sheet(page).innerText(),/Lose all 4 enhancement points/);assert.match(await sheet(page).innerText(),/3 Restoration Scrolls/);assert.match(await sheet(page).innerText(),/3,000 coins/);await geometry(page);await shot(page,'reforge-preview-'+width);
      const before=await save(page);await footer(page);const after=await save(page);assert.equal(K.points(after.collection.gear['trail-boots']),0);assert.equal(K.used(after.collection.gear['trail-boots']),0);assert.equal(after.collection.equipped.boots,'trail-boots');assert.equal(after.collection.scrolls.restoration,0);assert.equal(after.collection.scrollRng,before.collection.scrollRng);assert.equal(after.collection.recent.at(-1).outcome,'reforged');await shot(page,'reforge-result-'+width);await context.close();
    }
    evidence.flows.push('explicit Reforge clears enhancements and failed attempts; exact cost; base/equipped item kept; RNG unchanged');
    for(const [width,height] of [[320,740],[390,844],[915,390]]) {
      const fresh=await open(width,height,H.Core.createState(1000));
      assert.equal(await fresh.page.locator('[data-wx-collection]').count(),0,'Collection shortcuts do not occupy the currency header');assert.equal(await fresh.page.locator('[data-wx-do^="guild-page:"]:visible').count(),0,'Locked collections have no early tabs');await geometry(fresh.page);await fresh.context.close();
      const {context,page}=await open(width,height,H.mature());
      const original=await page.locator('.wx-world').boundingBox();
      await openCollection(page,'cards');await key(page,'collection-intro:cards').click();
      await Onboarding.finishCurrentGuide(page);
      let state=await save(page);assert.equal(state.collection.cardsUnlocked,true);assert.equal(state.collection.cards['trail-courier'].rank,2);assert.equal(state.collection.decks.filter(deck=>deck.slots.includes('trail-courier')).length,2,'Hands-on tutorial builds two real saved decks');
      await page.locator('[data-wx-nav="expedition"]').click();assert.deepEqual(await page.locator('.wx-world').boundingBox(),original,'Claim causes no world shift');
      await openCollection(page,'cards');assert.equal(await page.locator('.wx-deck-slot').count(),4);await geometry(page);await shot(page,'starter-cards-'+width);
      await key(page,'card:trail-courier').click();await geometry(page);state=await save(page);assert.equal(state.collection.decks[0].slots[0],'trail-courier');await close(page);
      await openCollection(page,'equipment');await key(page,'collection-intro:equipment').click();await Onboarding.finishCurrentGuide(page);await geometry(page);await shot(page,'starter-equipment-'+width);await close(page);
      await key(page,'gear:trail-boots').click();state=await save(page);assert.equal(state.collection.equipped.boots,'trail-boots');assert.equal(K.points(state.collection.gear['trail-boots']),1,'Hands-on tutorial makes a real enhancement');await geometry(page);await shot(page,'equipment-detail-'+width);await close(page);await context.close();
      const mature=await open(width,height,collected(true));await openCollection(mature.page,'cards');await geometry(mature.page);await shot(mature.page,'all-cards-'+width);
      await key(mature.page,'card-help').click();assert.match(await sheet(mature.page).innerText(),/next|common|rarity/i);await geometry(mature.page);await shot(mature.page,'card-odds-'+width);await close(mature.page);
      await key(mature.page,'card:trail-courier').click();await key(mature.page,'menu:'+JSON.stringify({kind:'collection-transaction',entity:'card',id:'trail-courier',operation:'fusion'})).click();await geometry(mature.page);await shot(mature.page,'fusion-unequipped-'+width);await close(mature.page);
      await openCollection(mature.page,'equipment');await geometry(mature.page);await shot(mature.page,'all-equipment-'+width);
      await key(mature.page,'gear:trail-boots').click();await key(mature.page,'scroll:bold').click();assert.match(await sheet(mature.page).innerText(),/8%.*→.*10/i);await geometry(mature.page);await shot(mature.page,'scroll-unequipped-'+width);await mature.context.close();evidence.viewports.push({width,height,stableClaim:true});
    }
    const paidFixture=collected();paidFixture.resources.coins=H.Core.Numbers.from(1000000);paidFixture.resources.ore=H.Core.Numbers.from(1000000);
    const {context,page}=await open(390,844,paidFixture);
    await openCollection(page,'cards');
    await key(page,'deck-name:deck-1').click();await page.locator('[data-wx-deck-name]').fill('My travelling guild');await page.clock.runFor(2000);assert.equal(await page.locator('[data-wx-deck-name]').inputValue(),'My travelling guild');await footer(page);assert.equal((await save(page)).collection.decks[0].name,'My travelling guild');
    for(const deck of ['deck-1','deck-2','deck-3']) {
      await key(page,'deck:'+deck).click();await key(page,'card:trail-courier').click();await key(page,'equip-card-slot:0').click();await footer(page);
    }
    await key(page,'card:trail-courier').click();await key(page,'menu:'+JSON.stringify({kind:'collection-transaction',entity:'card',id:'trail-courier',operation:'fusion'})).click();
    assert.match(await sheet(page).innerText(),/Consume 2 duplicates/);await geometry(page);await shot(page,'fusion-preview');await footer(page);
    let state=await save(page);assert.equal(state.collection.cards['trail-courier'].rank,2);assert.equal(state.collection.cards['trail-courier'].copies,10);assert(state.collection.decks.every(deck=>deck.slots[0]==='trail-courier'));await shot(page,'fusion-result');await footer(page);
    await key(page,'card:trail-courier').click();await key(page,'menu:'+JSON.stringify({kind:'card-ink',id:'trail-courier'})).click();await key(page,'menu:'+JSON.stringify({kind:'collection-transaction',entity:'card',id:'trail-courier',operation:'recycle'})).click();await footer(page);assert.equal((await save(page)).collection.ink,21);await footer(page);
    await key(page,'card:trail-courier').click();await key(page,'menu:'+JSON.stringify({kind:'card-ink',id:'trail-courier'})).click();await key(page,'menu:'+JSON.stringify({kind:'collection-transaction',entity:'card',id:'trail-courier',operation:'craft'})).click();await footer(page);assert.equal((await save(page)).collection.ink,11);await footer(page);
    await openCollection(page,'equipment');await key(page,'gear:quarry-pick').click();
    const beforeForge=await save(page),forgeCost=H.Core.getView(beforeForge).collection.equipment.items.find(item=>item.id==='quarry-pick').forge.cost;
    assert.equal(beforeForge.collection.gear['quarry-pick'],undefined);await footer(page);const forged=await save(page);
    const accruedForge=clone(beforeForge);H.Core.advanceTo(accruedForge,forged.lastUpdate);
    for(const [resource,cost] of Object.entries(forgeCost)) {
      const debit=H.Core.Numbers.toNumber(accruedForge.resources[resource])-H.Core.Numbers.toNumber(forged.resources[resource]);
      assert(Math.abs(debit-H.Core.Numbers.toNumber(cost))<1e-6,'Ordinary forging spends its quoted '+resource+' cost after normal elapsed production');
    }
    assert.equal(forged.collection.recent.at(-1).itemId,'quarry-pick');assert.equal(forged.collection.equipped.tool,beforeForge.collection.equipped.tool,'Forging never auto-equips');assert.equal(forged.collection.scrollRng,beforeForge.collection.scrollRng);await footer(page);
    await key(page,'gear:quarry-pick').click();await footer(page);assert.equal((await save(page)).collection.equipped.tool,'quarry-pick');await close(page);
    await key(page,'gear:trail-boots').click();await footer(page);
    for(const id of ['steady','bold','restoration']) {
      if(!(await sheet(page).count())) await key(page,'gear:trail-boots').click();
      await key(page,'scroll:'+id).click();await geometry(page);await shot(page,'scroll-'+id+'-preview');await footer(page);state=await save(page);
      assert.equal(K.points(state.collection.gear['trail-boots']),1);assert.equal(state.collection.gear['trail-boots'].failed,id==='bold'?1:0);
      await shot(page,'scroll-'+id+'-result');await footer(page);
    }
    const checkpoint=await save(page);await page.reload();await page.locator('.wx-game').waitFor();assert.deepEqual(inventory((await save(page)).collection),inventory(checkpoint.collection));evidence.flows.push('three saved decks; naming; exact fusion; recycling; crafting; equip; success, failure, recovery; reload');await context.close();
    const failureFixture=collected();failureFixture.collection.scrollRng=123456789;
    const failed=await open(390,844,failureFixture);const p=failed.page;
    await openCollection(p,'equipment');await key(p,'gear:trail-boots').click();await key(p,'scroll:bold').click();const before=await save(p);
    await p.evaluate(()=>{window.collectionSetItem=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');return window.collectionSetItem.call(this,key,value);};});
    await footer(p);assert.match(await sheet(p).innerText(),/Save recovery required/i);assert.equal(await p.locator('.wx-sheet[data-kind="collection-result"][open]').count(),0);
    const persisted=await p.evaluate(()=>JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state);assert.deepEqual(persisted.collection,before.collection);
    await p.evaluate(()=>{Storage.prototype.setItem=window.collectionSetItem;});await footer(p);
    assert.equal(await p.locator('.wx-sheet[data-kind="collection-result"][open]').count(),0,'Retry of a rolled-back action does not display a historical result');
    assert.equal(await p.locator('.wx-sheet[data-kind="collection-transaction"][open]').count(),1,'The actual scroll review remains available after Retry');
    assert.deepEqual(inventory((await save(p)).collection),inventory(before.collection),'Retry saves the rollback without consuming a scroll or drawing RNG');
    await footer(p);assert.equal(await p.locator('.wx-sheet[data-kind="collection-result"][open]').count(),1,'A new explicit confirmation commits the roll');
    const recovered=await save(p);assert.equal(recovered.collection.scrolls.bold,before.collection.scrolls.bold-1);assert.equal(recovered.collection.gear['trail-boots'].failed,1);assert(recovered.collection.sequence>recovered.collection.seen);
    await p.evaluate(()=>{Storage.prototype.setItem=function(key,value){if(key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');return window.collectionSetItem.call(this,key,value);};});await footer(p);
    assert.equal((await p.evaluate(()=>JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state)).collection.seen,recovered.collection.seen,'Failed acknowledgement never discards pending result');
    await p.evaluate(()=>{Storage.prototype.setItem=window.collectionSetItem;});await footer(p);assert.equal((await save(p)).collection.seen,recovered.collection.seen,'Retry saves the unseen result, not the failed acknowledgement');
    await p.reload();await p.locator('.wx-game').waitFor();assert.deepEqual(inventory((await save(p)).collection),inventory(recovered.collection));await openCollection(p,'equipment');assert(await key(p,'collection-inbox').count());await shot(p,'save-retry-restored');evidence.flows.push('failed durable write rolls back inventory and RNG; retry requires a fresh explicit roll; failed acknowledgement and reload retain its unseen result');await failed.context.close();
    const committed=await open(390,844,failureFixture),f=committed.page;
    await openCollection(f,'equipment');await key(f,'gear:trail-boots').click();await key(f,'scroll:bold').click();const beforeCommit=await save(f);
    await f.evaluate(()=>{
      window.collectionSetItem=Storage.prototype.setItem;window.collectionGetItem=Storage.prototype.getItem;window.collectionWritten=false;
      Storage.prototype.setItem=function(key,value){const result=window.collectionSetItem.call(this,key,value);if(key===WayfarersStorage.SAVE_KEY)window.collectionWritten=true;return result;};
      Storage.prototype.getItem=function(key){if(key===WayfarersStorage.RESET_KEY && window.collectionWritten)throw new DOMException('Fence unavailable','SecurityError');return window.collectionGetItem.call(this,key);};
    });
    await footer(f);assert.equal(await f.locator('.wx-sheet[data-kind="collection-result"][open]').count(),0,'A committed but unchecked result waits for the save fence');
    const durable=await f.evaluate(()=>JSON.parse(window.collectionGetItem.call(localStorage,WayfarersStorage.SAVE_KEY)).state);
    assert.equal(durable.collection.scrolls.bold,beforeCommit.collection.scrolls.bold-1);assert.equal(durable.collection.gear['trail-boots'].failed,1);assert.equal(durable.collection.sequence,beforeCommit.collection.sequence+1);
    assert.notEqual(durable.collection.scrollRng,beforeCommit.collection.scrollRng);assert(durable.collection.sequence>durable.collection.seen);
    await f.evaluate(()=>{Storage.prototype.setItem=window.collectionSetItem;Storage.prototype.getItem=window.collectionGetItem;});await footer(f);
    assert.equal(await f.locator('.wx-sheet[data-kind="collection-result"][open]').count(),1);assert.match(await sheet(f).innerText(),/Scroll did not take/);
    assert.deepEqual(inventory((await save(f)).collection),inventory(durable.collection),'Retry displays the exact committed outcome without spending or rerolling');
    await shot(f,'committed-fence-retry');await footer(f);const acknowledged=await save(f);assert.equal(acknowledged.collection.seen,durable.collection.sequence);
    await f.reload();await f.locator('.wx-game').waitFor();assert.deepEqual(inventory((await save(f)).collection),inventory(acknowledged.collection));
    evidence.flows.push('committed-fence failure keeps one durable roll; Retry shows that exact new result; acknowledgement and reload do not reroll');await committed.context.close();
    assert.deepEqual(evidence.errors,[]);fs.writeFileSync(path.join(output,'collection-browser.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify({ok:true,output,viewports:evidence.viewports.length,flows:evidence.flows.length}));
  } catch(error) {
    evidence.failure=error.stack;evidence.failurePages=[];
    for(const page of browser.contexts().flatMap(context=>context.pages())) {
      const name='failure-'+evidence.failurePages.length;
      try {
        await page.screenshot({path:path.join(output,name+'.png')});
        evidence.failurePages.push(await page.evaluate(()=>({
          guide:Array.from(document.querySelectorAll('.wx-guide[open]')).map(node=>({guide:node.dataset.guide,step:node.dataset.step,text:node.textContent})),
          sheet:document.querySelector('.wx-sheet[open]')?.dataset.kind,
          currencies:Array.from(document.querySelectorAll('[data-wx-currency]')).map(node=>node.dataset.wxCurrency)
        })));
      } catch(captureError) { evidence.failurePages.push({captureError:captureError.message}); }
    }
    fs.writeFileSync(path.join(output,'collection-browser.json'),JSON.stringify(evidence,null,2));throw error;
  } finally { await browser.close();await new Promise(resolve=>server.close(resolve)); }
}
run().catch(error=>{fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'collection-browser-error.txt'),error.stack);console.error(error);process.exitCode=1;});
