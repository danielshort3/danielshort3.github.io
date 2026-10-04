'use strict';

// Exact APK assets, native pointer input, and disposable engine-validated saves.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const Core = require('../../js/games/wayfarers-guild/core');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-progression-')));
const fixtureFile = process.env.WAYFARERS_PROGRESSION_FIXTURE;
const evidence = { opening:[], unlocks:[], areas:[], gestures:[], purchases:[], errors:[] };
const copy = value => JSON.parse(JSON.stringify(value));
async function saved(page) { return page.evaluate(() => { document.dispatchEvent(new Event('freeze')); return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state; }); }
async function close(page) { if (await page.locator('.wx-sheet[open]').count()) await page.locator('[data-wx-close]').click(); }
async function paint(page) { await page.clock.runFor(100); await page.waitForTimeout(40); await page.clock.runFor(100); }
async function shot(page,name) { await paint(page); await page.screenshot({path:path.join(output,name + '.png')}); }
async function anchors(page) { return Promise.all(['.wx-world','.wx-dock','.wx-nav'].map(selector=>page.locator(selector).boundingBox())); }
async function area(page,id) {
  await close(page); await page.locator('[data-wx-nav="expedition"]').click();
  await page.locator('[data-wx-objective]').click(); await page.locator('[data-wx-area="' + id + '"]').click();
  assert.equal(await page.locator('.wx-game').getAttribute('data-area'),id);
}
async function quantity(page,count) {
  const inSheet = page.locator('.wx-sheet[open] [data-wx-do="batch"]');
  const trigger = await inSheet.count() ? inSheet : page.locator('[data-wx-batch]:visible,.wx-destination .wx-batch:visible').first();
  await trigger.click(); await page.locator('[data-wx-do="batch:' + count + '"]').click();
}
async function action(page,type,id) {
  const key = await page.locator('.wx-sheet[open] [data-wx-do]').evaluateAll((nodes,target) => nodes.find(node => { try { const a=JSON.parse(node.dataset.wxDo); return a.type===target.type && a.id===target.id && !node.disabled; } catch { return false; } })?.dataset.wxDo,{type,id});
  assert(key,'An enabled ' + type + ':' + id + ' action is rendered');
  await page.locator('.wx-sheet[open] [data-wx-do=' + JSON.stringify(key) + ']').click();
}
async function geometry(page) {
  const result = await page.evaluate(() => {
    const sheet=document.querySelector('.wx-sheet[open]');
    const targets=[...document.querySelectorAll('.wx-game button,.wx-sheet[open] button')].filter(n=>n.getClientRects().length && (!sheet || sheet.contains(n))).map(n=>({name:n.getAttribute('aria-label')||n.textContent.trim(),width:n.getBoundingClientRect().width,height:n.getBoundingClientRect().height}));
    const clippedCosts=[...document.querySelectorAll('.wx-dock .wx-price>span')].filter(n=>n.getClientRects().length && n.getBoundingClientRect().bottom>n.parentElement.getBoundingClientRect().bottom+1).map(n=>n.textContent);
    return {overflow:document.documentElement.scrollWidth-innerWidth,vertical:document.body.scrollHeight-innerHeight,sheet:sheet ? sheet.scrollWidth-sheet.clientWidth : 0,targets,clippedCosts};
  });
  assert(result.overflow<=1 && result.vertical<=1 && result.sheet<=1,'No viewport overflow: ' + JSON.stringify(result));
  assert.deepEqual(result.targets.filter(t=>t.width<47.5 || t.height<47.5),[],'Every visible control has a 48px target');
  assert.deepEqual(result.clippedCosts,[],'Multi-resource prices remain inside purchase buttons');
  return result;
}
async function swipe(page,points,cancel=false) {
  const client=await page.context().newCDPSession(page);
  await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:points[0][0],y:points[0][1],id:1}]});
  for(const point of points.slice(1)) await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:point[0],y:point[1],id:1}]});
  await client.send('Input.dispatchTouchEvent',{type:cancel?'touchCancel':'touchEnd',touchPoints:[]});
  await client.detach();
}

async function run() {
  fs.mkdirSync(output,{recursive:true}); const files=path.join(output,'progression-bundle'); bundle(files);
  const server=http.createServer((request,response)=>{
    const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/');
    const file=path.resolve(files,'.'+pathname);
    if(!file.startsWith(files+path.sep)) { response.writeHead(403).end(); return; }
    fs.readFile(file,(error,bytes)=>{ if(error) { response.writeHead(404).end(); return; } response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp'})[path.extname(file)]||'application/json'); response.end(bytes); });
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  async function open(width,height,fixture) {
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    const state=copy(fixture||Core.createState(1000)); state.lastUpdate=1000;
    Core.act(state,{type:'introduction-seen',ids:Core.getPresentation(state).introductions.map(i=>i.id)});
    Core.act(state,{type:'discovery-seen',seq:state.luck.ledger.seq});
    const record=Storage.createStore({storage:null,now:()=>1000}).export(state); assert(record.ok,'Valid isolated fixture');
    await context.addInitScript(({key,value})=>{if(!sessionStorage.getItem('seeded')) {localStorage.setItem(key,value);sessionStorage.setItem('seeded','true');}},{key:Storage.SAVE_KEY,value:record.text});
    const page=await context.newPage(); page.setDefaultTimeout(10000); page.on('pageerror',error=>evidence.errors.push(error.message));
    await page.clock.install({time:new Date(1000)}); await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');
    await page.locator('.wx-game').waitFor(); await paint(page); await page.locator('[data-scene-status="ready"]').waitFor();
    return {context,page};
  }
  try {
    for(const [width,height] of [[320,740],[390,844],[915,390]]) {
      const {context,page}=await open(width,height);
      assert.equal(await page.locator('[data-wx-buy]').count(),1); assert.equal(await page.locator('.wx-nav button:visible').count(),1);
      assert.equal(await page.locator('[data-wx-prev]:visible,[data-wx-next]:visible,[data-wx-batch]:visible,[data-wx-focus]:visible').count(),0);
      const anchor=await page.locator('.wx-world').boundingBox();
      await geometry(page); await shot(page,'fresh-'+width);
      await page.clock.runFor(8000); await page.locator('[data-wx-buy="boots"]').click();
      assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,1);
      assert.deepEqual(await page.locator('.wx-world').boundingBox(),anchor,'First purchase keeps world stable');
      const checkpoint=await saved(page); await page.reload(); await page.locator('.wx-game').waitFor(); await paint(page);
      assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,1); assert.equal((await saved(page)).createdAt,checkpoint.createdAt);
      if(width!==390) {
        const beforeUnlock=await anchors(page);
        await page.clock.runFor(10000); await page.locator('[data-wx-buy="boots"]').click();
        assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,2);
        assert.equal(await page.locator('[data-wx-buy]').count(),2);
        assert.deepEqual(await anchors(page),beforeUnlock,'Second learned row keeps world/dock/nav fixed');
        await page.locator('[data-wx-do="local:porters"]').click();assert.match(await page.locator('.wx-sheet h2').innerText(),/Porters/);await close(page);
        await page.clock.runFor(20000);await page.locator('[data-wx-buy="porters"]').click();
        assert.equal((await saved(page)).expedition.areas.greenway.ranks.porters,1);
        assert.deepEqual(await anchors(page),beforeUnlock);await geometry(page);await shot(page,'second-track-unlock-'+width);
        evidence.unlocks.push({width,track:'porters',rankBefore:1,rankAfter:2,geometryStable:true});
      }
      evidence.opening.push({width,height,rank:1}); await context.close();
    }
    const helper=require('./helpers/wayfarers-progression.cjs');
    const fourth=Core.createState(0);helper.fund(fourth);
    for(let index=0;index<2;index++) {
      while(!fourth.expedition.completed) {
        for(const [areaId,a] of Object.entries(fourth.expedition.areas))for(const id of a.learned)if(a.ranks[id]<12)Core.act(fourth,{type:'expedition-buy',areaId,id});
        helper.advance(fourth,10);
      }
      if(index===0)Core.act(fourth,{type:'expedition-next'});
    }
    Core.act(fourth,{type:'expedition-select',areaId:'greenway'});helper.fund(fourth);
    assert(Core.act(fourth,{type:'expedition-development',id:'wheelworks'}).ok);
    const project=helper.P.Content.PROJECTS.find(p=>p.id==='wheelworks');
    helper.P.tick(fourth,project.work/helper.P.rawRates(fourth).researchRate-4);
    assert.equal(fourth.expedition.areas.greenway.learned.length,3);
    for(const [width,height] of [[320,740],[915,390]]) {
      const {context,page}=await open(width,height,fourth);const beforeUnlock=await anchors(page);
      assert.equal(await page.locator('[data-wx-buy]').count(),3);
      await page.clock.runFor(5000);await close(page);
      assert.equal(await page.locator('[data-wx-buy]').count(),4);
      assert.deepEqual(await anchors(page),beforeUnlock,'Fourth learned row keeps world/dock/nav fixed');
      const fourthButton=page.locator('[data-wx-buy="caravans"]');await fourthButton.scrollIntoViewIfNeeded();await fourthButton.click();
      assert.equal((await saved(page)).expedition.areas.greenway.ranks.caravans,1);
      await page.locator('[data-wx-do="local:caravans"]').click();assert.match(await page.locator('.wx-sheet h2').innerText(),/Caravans/);await close(page);
      await geometry(page);await shot(page,'fourth-track-unlock-'+width);
      evidence.unlocks.push({width,track:'caravans',learnedBefore:3,learnedAfter:4,geometryStable:true});await context.close();
    }
    const mature=fixtureFile ? JSON.parse(fs.readFileSync(fixtureFile,'utf8').replace(/^\uFEFF/,'')) : require('./helpers/wayfarers-progression.cjs').mature();
    assert(Core.validateState(mature).valid,'Mature fixture is canonical engine state');
    for(const [width,height] of [[320,740],[390,844],[915,390]]) {
      const {context,page}=await open(width,height,mature);
      assert.equal(await page.locator('.wx-nav button:visible').count(),3);
      const original=await saved(page), ranks=Object.fromEntries(Object.entries(original.expedition.areas).map(([id,a])=>[id,a.ranks]));
      for(const id of Object.keys(mature.expedition.areas)) {
        await area(page,id); await paint(page); assert.equal(await page.locator('[data-wx-buy]').count(),6);
        assert.equal(await page.locator('[data-wx-canvas]').getAttribute('data-scene-kind'),id);
        assert.equal(await page.locator('[data-wx-do="world-choice"]').count(),1,'Earned area plans stay available');
        await geometry(page); await shot(page,id+'-'+width);
        await page.locator('[data-wx-do="world-choice"]').click(); await geometry(page); await shot(page,id+'-plans-'+width); await close(page);
      }
      const after=await saved(page); assert.deepEqual(Object.fromEntries(Object.entries(after.expedition.areas).map(([id,a])=>[id,a.ranks])),ranks,'Area navigation never resets ranks');
      await area(page,'greenway'); const tray=page.locator('[data-wx-tray]'); await tray.evaluate(n=>n.scrollTop=n.scrollHeight);
      const scroll=await tray.evaluate(n=>n.scrollTop); assert(scroll>0,'Six tracks live in bounded scroller');
      await area(page,'quarry'); await area(page,'greenway'); assert.equal(await tray.evaluate(n=>n.scrollTop),scroll,'Per-area scroll is retained');
      await tray.evaluate(n=>n.scrollTop=0); await quantity(page,5);
      const before=await saved(page); const quote=Core.getView(before).expedition.cards.find(i=>i.trackId==='boots');
      assert.equal(quote.quantity,5); assert.equal(quote.action.count,5);
      await page.locator('[data-wx-buy="boots"]').click();
      assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,before.expedition.areas.greenway.ranks.boots+5,'Local click buys exactly selected quantity');
      await page.locator('[data-wx-do="local:boots"]').click(); assert.match(await page.locator('.wx-sheet').innerText(),/Exactly 5 ranks/);
      const purchaseBox=await page.locator('.wx-purchase-footer .wx-confirm').boundingBox(); assert(purchaseBox.y>=0&&purchaseBox.y+purchaseBox.height<=height,'Details keep the purchase visible without scrolling');
      await quantity(page,10); assert.match(await page.locator('.wx-sheet').innerText(),/Exactly 10 ranks/,'Details update the shared exact quantity');
      await quantity(page,5);
      await shot(page,'batch-details-'+width); await close(page);
      await page.locator('[data-wx-nav="upgrades"]').click(); await geometry(page); await shot(page,'catalog-'+width);
      await page.locator('[data-wx-upgrade="area:greenway:boots"] .wx-research-info').click();
      assert.match(await page.locator('.wx-sheet').innerText(),/Exactly 5 ranks/); await close(page);
      if(width===390) {
        const oldGuild=(await saved(page)).upgrades.boots;
        await page.locator('[data-wx-upgrade="guild:buy:boots"] .wx-research-info').click();
        assert.match(await page.locator('.wx-sheet').innerText(),/Exactly 5 ranks/,'Guild repeatable shares batch mode'); await page.locator('.wx-sheet .wx-confirm').click();
        assert.equal((await saved(page)).upgrades.boots,oldGuild+5,'Guild repeatable buys the promised exact quantity'); await close(page);
      }
      await area(page,'watchtower'); await page.locator('[data-wx-do="world-choice"]').click();
      const config=page.locator('[data-wx-do*="configuration"]').first(); assert(await config.count(),'Advanced choices are reachable'); await config.click();
      if(await page.locator('[data-wx-do="config-slot:1"]').count()) {
        await page.locator('[data-wx-do="config-slot:1"]').click(); await action(page,'expedition-config','industry');
        assert.equal((await saved(page)).expedition.areas.watchtower.plans.assignments[1],'industry','Second-slot assignment updates its own canonical slot');
      }
      await geometry(page); await shot(page,'tower-assignment-'+width); await close(page);
      await page.locator('[data-wx-focus]').click(); assert.match(await page.locator('.wx-sheet').innerText(),/Shared across every area/); assert(await page.locator('.wx-sheet .wx-impact').count(),'Focus shows actual rate changes');
      await geometry(page); await shot(page,'focus-'+width); await close(page);
      if(width===390) {
        await area(page,'greenway'); const world=await page.locator('.wx-world').boundingBox(); const y=world.y+world.height*.56;
        await swipe(page,[[world.x+world.width*.74,y],[world.x+world.width*.5,y],[world.x+world.width*.24,y]]);
        assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'quarry','Horizontal scene swipe navigates once');
        assert.equal(await page.locator('.wx-sheet[open]').count(),0,'Swipe cannot also inspect');
        const unchanged=async(label,points,cancel)=>{ await swipe(page,points,cancel); assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'quarry',label); assert.equal(await page.locator('.wx-sheet[open]').count(),0,label+' cannot inspect'); evidence.gestures.push(label); };
        await unchanged('Vertical intent',[[world.x+world.width*.7,y],[world.x+world.width*.65,y+25],[world.x+world.width*.3,y+100]]);
        await unchanged('Diagonal intent',[[world.x+world.width*.7,y],[world.x+world.width*.3,y+100]]);
        await unchanged('Cancelled gesture',[[world.x+world.width*.7,y],[world.x+world.width*.3,y]],true);
        await unchanged('Android edge gesture',[[5,y],[120,y]]);
        const dock=await tray.boundingBox(); await unchanged('Dock gesture',[[300,dock.y+70],[90,dock.y+70]]);
        const client=await context.newCDPSession(page);
        await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:260,y,id:1},{x:280,y:y+20,id:2}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:130,y,id:1},{x:150,y:y+20,id:2}]});
        await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]}); await client.detach();
        assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'quarry','Multitouch cancelled'); assert.equal(await page.locator('.wx-sheet[open]').count(),0);
        await page.locator('[data-wx-next]').click(); assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'watchtower');
        await page.locator('[data-wx-prev]').click(); assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'quarry');
        evidence.gestures.push('Horizontal navigation','Multitouch cancellation','48px buttons');
        for(const [id,group,choice,field] of [['workshop','Manufacturing templates','tools','templates'],['ruins','Selected discovery','metallic','discovery'],['harbor','Next voyage port','ocean','port']]) {
          await area(page,id); await page.locator('[data-wx-do="world-choice"]').click(); await page.getByRole('button',{name:new RegExp('^'+group)}).click();
          await action(page,'expedition-config',choice); const value=(await saved(page)).expedition.areas[id].plans[field];
          assert(Array.isArray(value)?value.includes(choice):value===choice,'Actual '+group+' choice is persisted'); await close(page);
        }
        await area(page,'greenway'); await page.locator('[data-wx-do="world-choice"]').click(); await page.getByRole('button',{name:/^Specialization/}).click(); await action(page,'expedition-specialize','boots'); await close(page);
        assert.equal((await saved(page)).expedition.areas.greenway.specialization,'boots');
        await page.locator('[data-wx-focus]').click(); const charges=(await saved(page)).expedition.focus.charges; await page.locator('[data-wx-do="focus:priority"]').click();
        assert.equal((await saved(page)).expedition.focus.charges,charges-1); await area(page,'quarry'); assert.match(await page.locator('[data-wx-focus]').getAttribute('aria-label'),/Trail active/,'Focus state is shared across areas');
        const beforeReset=await saved(page);
        await page.locator('[data-wx-nav="guild"]').click(); await page.locator('[data-wx-do="guild-atlas"]').click(); await page.getByRole('button',{name:/^Renewals/}).click(); await page.getByRole('button',{name:/^Expedition Refit/}).click(); await page.locator('[data-wx-do="review:refit"]').click();
        const resetText=await page.locator('[data-dialog-body]').innerText();
        assert.match(resetText,/learned track|learned branch/); assert.match(resetText,/ordinary local/); assert.match(resetText,/Focus charges/);
        await shot(page,'refit-preview'); await page.locator('[data-confirm-reset]').click(); await paint(page);
        const reset=await saved(page); assert.equal(Object.keys(reset.expedition.areas).length,6);
        for(const [id,a] of Object.entries(reset.expedition.areas)) { assert(Object.values(a.ranks).every(rank=>rank===0)); assert.deepEqual(a.learned,beforeReset.expedition.areas[id].learned); assert.equal(a.cap,beforeReset.expedition.areas[id].cap); }
        assert.equal(reset.expedition.focus.charges,beforeReset.expedition.focus.charges);
        await area(page,'greenway'); await shot(page,'refit-retained-six-tracks'); await geometry(page);
        await page.evaluate(() => { const act=WayfarersCore.act; window.qaActions=[]; WayfarersCore.act=function(state,action) {const result=act(state,action);window.qaActions.push({action,result});return result;}; });
        await page.locator('[data-wx-buy="boots"]').click();
        if((await saved(page)).expedition.areas.greenway.ranks.boots===0) {
          fs.writeFileSync(path.join(output,'post-refit-rejected-purchase.json'),JSON.stringify({state:await saved(page),button:await page.locator('[data-wx-buy="boots"]').getAttribute('aria-label'),toast:await page.locator('[data-wx-toast]').innerText(),actions:await page.evaluate(()=>window.qaActions)},null,2));
          assert.match(await page.locator('[data-wx-toast]').innerText(),/quote changed/i,'A changed frontier rejects the stale quote explicitly');
          await close(page); await page.locator('[data-wx-buy="boots"]').click();
          evidence.purchases.push({staleQuoteRejected:true,refreshedPurchase:true});
        }
        assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,5,'Retained starter money buys the selected full batch');
      }
      evidence.areas.push({width,height,areas:Object.keys(mature.expedition.areas),batch:5}); await context.close();
    }
    const nearCap=copy(mature); nearCap.expedition.selectedArea='greenway'; nearCap.expedition.batch=100;
    nearCap.expedition.areas.greenway.ranks.boots=950; nearCap.expedition.areas.greenway.highRanks.boots=950;
    for(const id of ['coins','ore','provisions','herbs','knowledge','maps']) nearCap.resources[id]=Core.Numbers.from('1e100');
    {
      const {context,page}=await open(320,740,nearCap); const before=await saved(page);
      assert(await page.locator('[data-wx-buy="boots"]').isDisabled()); assert.match(await page.locator('[data-wx-buy="boots"]').innerText(),/Unavailable/);
      await page.locator('[data-wx-buy="boots"]').evaluate(node=>node.click());
      assert.deepEqual((await saved(page)).expedition.areas.greenway.ranks,before.expedition.areas.greenway.ranks); assert.equal((await saved(page)).expedition.batch,100);
      await page.locator('[data-wx-do="local:boots"]').click(); assert.match(await page.locator('.wx-sheet').innerText(),/Only 50 ranks remain/); await shot(page,'exact-batch-cap-limit');
      assert.doesNotMatch(await page.locator('.wx-purchase-footer').innerText(),/1050/,'Impossible rank endpoint is not offered');
      await quantity(page,25); await page.locator('.wx-sheet .wx-confirm').click(); await page.locator('.wx-sheet .wx-confirm').click();
      assert.equal((await saved(page)).expedition.areas.greenway.ranks.boots,1000); assert.equal((await saved(page)).expedition.batch,25); await close(page); await geometry(page); await shot(page,'exact-batch-cap-complete');
      evidence.purchases.push({cap:1000,disabledCount:100,remaining:50,explicitCount:25,finalRank:1000}); await context.close();
    }
    const developing=require('./helpers/wayfarers-progression.cjs').mature({untilProject:'rail-network'});
    developing.expedition.selectedArea='greenway';
    {
      const {context,page}=await open(390,844,developing);
      assert.match(await page.locator('[data-upgrade="boots"] [data-wx-rank]').innerText(),/\/ 250/);
      await shot(page,'learned-rank-cap250');
      await page.locator('[data-wx-do="local:railways"]').click(); await shot(page,'learned-railway-detail'); await close(page);
      await page.locator('[data-wx-nav="upgrades"]').click(); await page.locator('[data-wx-upgrade="development:industrial-supports"] .wx-research-info').click();
      await page.locator('.wx-sheet .wx-confirm').click(); assert.match(await page.locator('.wx-detail-hero').innerText(),/Researching/);
      assert.match(await page.locator('.wx-purchase-footer').innerText(),/Funded:/);
      assert.doesNotMatch(await page.locator('.wx-sheet').innerText(),/Save for this/);
      await shot(page,'funded-research-progress');
      await context.close();
    }
    assert.deepEqual(evidence.errors,[]); fs.writeFileSync(path.join(output,'progression-browser-evidence.json'),JSON.stringify(evidence,null,2));
    console.log(JSON.stringify({passed:true,output,opening:evidence.opening.length,mature:evidence.areas.length,gestures:evidence.gestures}));
  } finally { await browser.close(); await new Promise(resolve=>server.close(resolve)); }
}
run().catch(error=>{fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'failure.json'),JSON.stringify({error:error.stack,evidence},null,2));console.error(error);process.exitCode=1;});
