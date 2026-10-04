'use strict';

// Real source/Android content, isolated test saves. Art and fixed-camera checks
// are separate from the engine's natural first-session pacing simulation.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const http=require('node:http');
const {chromium}=require('playwright');
const {bundle}=require('../../build/bundle-wayfarers-android.cjs');
const H=require('./helpers/wayfarers-progression.cjs');
const F=require('./helpers/wayfarers-onboarding.cjs');
const Storage=require('../../js/games/wayfarers-guild/persistence');
const Stations=require('../../js/games/wayfarers-guild/stations');
const StationFixtures=require('./helpers/wayfarers-stations.cjs');
const output=path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-stations-')));
const report={browser:'Browser plugin not available; repository Playwright workflow used.',flows:[],viewports:[],errors:[]};
const text130='.wx-game .wx-inline-info>strong{font-size:14.3px!important;line-height:18.2px!important}.wx-game .wx-inline-rank{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-buy{font-size:15.6px!important;line-height:20.8px!important}.wx-game .wx-inline-price>span{font-size:15.6px!important;line-height:19.5px!important}.wx-game .wx-inline-buy>small{font-size:13px!important;line-height:16.9px!important}';
function fresh() {const state=H.Core.createState(1000);H.fund(state);F.completeAreaGuides(state);F.announceDiscoveries(state);return state;}
function mature() {const state=StationFixtures.mature();state.stations.encounter.remaining=0;F.announceDiscoveries(state);return state;}
function sortingOnly() {
  const state=StationFixtures.expansionReady();StationFixtures.act(state,{type:'expedition-next'});F.completeAreaGuides(state);StationFixtures.fund(state);
  const mine=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='mine'),hauling=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='hauling'),sorting=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='sorting');
  while(state.stations.ranks[mine.skillIds[0]]<60)StationFixtures.act(state,{type:'station-skill-buy',id:mine.skillIds[0],count:1});H.Core.advance(state,800);
  for(const station of [hauling,sorting]){StationFixtures.act(state,{type:'station-build',id:station.id});StationFixtures.act(state,{type:'station-select',id:station.id});}
  while((state.stations.ranks[sorting.skillIds[0]] || 0)<25)StationFixtures.act(state,{type:'station-skill-buy',id:sorting.skillIds[0],count:1});
  const skill=Stations.Content.SKILLS.find(skill=>skill.stationId===sorting.id&&skill.alias==='ore-sorting');StationFixtures.act(state,{type:'station-skill-unlock',id:skill.id});StationFixtures.act(state,{type:'station-skill-buy',id:skill.id,count:1});
  StationFixtures.act(state,{type:'expedition-select',areaId:'quarry'});assert.equal(H.Core.getView(state).expedition.configurations.length,0,'Early sorting has no later configurations');StationFixtures.act(state,{type:'onboarding-visit',id:'plans'});assert(H.Core.validateState(state).valid);return state;
}
async function run() {
  fs.mkdirSync(output,{recursive:true});const files=path.join(output,'bundle');bundle(files);
  const server=http.createServer((req,res)=>{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\/assets\//,'/');const file=path.resolve(files,'.'+pathname);if(!file.startsWith(files+path.sep)){res.writeHead(403).end();return;}fs.readFile(file,(error,bytes)=>{if(error){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp'})[path.extname(file)] || 'application/json');res.end(bytes);});});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  async function open(width,height,seed) {
    const state=H.clone(seed);state.lastUpdate=1000;const record=Storage.createStore({storage:null,now:()=>1000}).export(state);assert(record.ok,record.message);
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    await context.addInitScript(({key,value})=>{if(!sessionStorage.seeded){localStorage.setItem(key,value);sessionStorage.seeded='1';}},{key:Storage.SAVE_KEY,value:record.text});
    const page=await context.newPage();page.setDefaultTimeout(12000);page.on('pageerror',error=>report.errors.push(error.message));await page.clock.install({time:new Date(1000)});
    await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.clock.runFor(2000);
    try{await page.locator('.wx-station-world').waitFor();}catch(error){await page.screenshot({path:path.join(output,'startup-failed.png')});process.stderr.write(JSON.stringify({output,errors:report.errors,body:await page.locator('body').innerText()},null,2)+'\n');throw error;}
    await page.waitForFunction(()=>document.querySelector('.wx-station-segment canvas')?.dataset.sceneStatus==='ready');
    return {context,page};
  }
  async function shot(page,name) {await page.waitForTimeout(50);await page.screenshot({path:path.join(output,name+'.png')});}
  async function dismissNotices(page) {for(let i=0;i<40 && await page.locator('.wx-sheet[open][data-kind="onboarding-notice"]').count();i++){await page.locator('[data-wx-close]').click();await page.clock.runFor(500);}}
  async function ready(page) {await page.waitForFunction(()=>{const world=document.querySelector('.wx-station-world')?.getBoundingClientRect();const nodes=[...document.querySelectorAll('.wx-station-segment canvas')].filter(node=>{const box=node.getBoundingClientRect();return box.bottom>world.top && box.top<world.bottom;});return nodes.length && nodes.every(node=>node.dataset.sceneStatus==='ready');});}
  async function geometry(page) {
    return page.evaluate(()=>{const rect=selector=>{const r=document.querySelector(selector).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};};return {hud:rect('.wx-header'),dock:rect('.wx-nav'),world:rect('.wx-station-world'),scroll:document.querySelector('.wx-station-world').scrollTop,horizontal:document.documentElement.scrollWidth-innerWidth,vertical:document.body.scrollHeight-innerHeight};});
  }
  async function saved(page) {return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});}
  async function inlineGeometry(page) {
    const rows=await page.evaluate(()=>[...document.querySelectorAll('.wx-station-segment')].map((segment,index)=>{
      const box=node=>{const rect=node.getBoundingClientRect();return {x:rect.x,y:rect.y,width:rect.width,height:rect.height,bottom:rect.bottom};};
      const art=segment.querySelector(':scope>.wx-station-illustration'),strip=segment.querySelector(':scope>.wx-station-controls'),canvas=art.querySelector('canvas');
      return {id:segment.dataset.wxStation,index,children:[...segment.children].map(node=>node.className),segment:box(segment),art:box(art),strip:box(strip),canvas:box(canvas),scene:canvas.getAttribute('aria-label'),cells:[...strip.children].map(cell=>({id:cell.dataset.wxInlineSkill,box:box(cell),info:box(cell.querySelector('.wx-inline-info')),buy:box(cell.querySelector('.wx-inline-buy'))}))};
    }));
    for(const row of rows) {
      assert.deepEqual(row.children,['wx-station-illustration','wx-station-controls'],'Art precedes the controls in the same station');
      assert(Math.abs(row.art.height-row.art.width*(row.index===0 ? 320 : 208)/384)<1,'Station art preserves its source aspect ratio');
      assert(Math.abs(row.strip.height-104)<.1,'Controls reserve exactly 104px in every state');
      assert(Math.abs(row.strip.y-row.art.bottom)<.1,'Controls sit directly below their station art');
      assert(row.scene?.endsWith(' working'),'Each station retains its illustrated scene');
      assert.equal(row.cells.length,3,'A canonical station has three starter controls');
      for(const cell of row.cells) {
        assert(Math.abs(cell.info.height-48)<.1 && Math.abs(cell.buy.height-48)<.1,'Info and purchase controls remain full touch targets');
        assert(cell.box.x>=row.strip.x && cell.box.x+cell.box.width<=row.strip.x+row.strip.width+.1,'Starter controls stay inside the strip');
      }
    }
    return rows;
  }
  async function rememberInlineNodes(page) {
    await page.evaluate(()=>{window.__inlineNodes=[...document.querySelectorAll('.wx-station-segment')].map(segment=>({segment,canvas:segment.querySelector('canvas'),cells:[...segment.querySelectorAll('.wx-inline-upgrade')].map(cell=>({cell,info:cell.querySelector('.wx-inline-info'),buy:cell.querySelector('.wx-inline-buy')}))}));});
  }
  async function sameInlineNodes(page) {
    assert(await page.evaluate(()=>window.__inlineNodes.every(({segment,canvas,cells})=>segment.isConnected && segment.querySelector('canvas')===canvas && cells.every(({cell,info,buy})=>cell.isConnected && cell.querySelector('.wx-inline-info')===info && cell.querySelector('.wx-inline-buy')===buy))),'Purchases, unlocks and quantity updates retain the actual art and button nodes');
  }
  async function readableInlinePrices(page) {
    const violations=await page.evaluate(()=>[...document.querySelectorAll('.wx-inline-buy')].flatMap(button=>{
      const box=button.getBoundingClientRect();
      return [...button.querySelectorAll('.wx-inline-price>span,.wx-inline-buy>b')].flatMap(node=>{const rect=node.getBoundingClientRect();return node.scrollWidth>node.clientWidth+1 || rect.x<box.x || rect.right>box.right || rect.y<box.y || rect.bottom>box.bottom ? [{skill:button.closest('.wx-inline-upgrade').dataset.wxInlineSkill,text:node.textContent,width:rect.width,scrollWidth:node.scrollWidth,clientWidth:node.clientWidth}] : [];});
    }));
    assert.deepEqual(violations,[],'Every currency price and purchase marker remains fully visible inside its button');
  }
  try {
    for(const [width,height] of [[320,740],[390,844],[640,256]]) {
      const {page,context}=await open(width,height,H.Core.createState(1000));
      const trace=[];
      async function saved() {return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});}
      for(let n=0;n<18 && (await saved()).onboarding.practice.progress.greenway<3;n++) {
        await page.clock.runFor(300);
        const coach=page.locator('.wx-guide[open]');if(!await coach.count())await page.clock.runFor(1500);if(!await coach.count()){await shot(page,'first-guide-missing-'+width);process.stderr.write(JSON.stringify({output,trace,body:await page.locator('body').innerText(),errors:report.errors},null,2)+'\n');}assert.equal(await coach.count(),1,'First area teaches actual controls');
        if(n===0){assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.equal(await coach.count(),1,'Back cannot abandon the mandatory first lesson');}
        while(/^currency:/.test(await coach.getAttribute('data-step'))){await page.locator('[data-guide-next]').click();await page.clock.runFor(100);}
        assert.equal(await coach.getAttribute('data-missing'),'false','Required control exists');
        assert.equal(await page.locator('[data-guide-leave]:visible').count(),0,'Mandatory first lesson cannot be skipped');
        const target=page.locator('[data-guide-target]');assert.equal(await target.count(),1);
        trace.push({step:await coach.getAttribute('data-step'),target:await target.getAttribute('data-wx-do') || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close')});
        await shot(page,'first-guide-'+width+'-'+n);await target.click();await page.clock.runFor(250);
      }
      const result=await saved();assert.equal(result.onboarding.practice.progress.greenway,3,JSON.stringify(trace));assert.equal(result.stations.ranks['station:greenway:path:pathfinding'],1);
      report.flows.push(width+'px first visit: '+JSON.stringify(trace));await context.close();
    }
    const opening=fresh();
    for(const [width,height,largeText] of [[320,740],[390,844],[430,932],[915,390],[320,740,true]]) {
      const {page,context}=await open(width,height,opening);
      if(largeText)await page.addStyleTag({content:text130});
      assert.equal(await page.locator('.wx-dock:visible').count(),0,'No permanent upgrade dock');assert.equal(await page.locator('.wx-inline-upgrade:visible').count(),3,'Three starters are part of the main world');assert.equal(await page.locator('.wx-inline-upgrade[data-state="locked"]').count(),2);
      assert.equal(await page.locator('.wx-inline-quantity:visible').count(),0,'Bulk selection waits for its earned unlock');
      await page.locator('.wx-inline-buy').first().scrollIntoViewIfNeeded();const before=await geometry(page);assert(before.horizontal<=1 && before.vertical<=1);const artBefore=await inlineGeometry(page);await rememberInlineNodes(page);
      const first=page.locator('.wx-inline-upgrade').first(),skill=await first.getAttribute('data-wx-inline-skill'),rankBefore=(await saved(page)).stations.ranks[skill];
      await first.locator('.wx-inline-buy').click();assert.equal((await saved(page)).stations.ranks[skill],rankBefore+1,'Main-screen purchase changes the actual rank');assert.deepEqual(await geometry(page),before,'Inline purchase never moves the camera');assert.deepEqual(await inlineGeometry(page),artBefore,'Inline purchase preserves art and strip geometry');await sameInlineNodes(page);assert.equal(await first.locator('[data-wx-unseen]').count(),0,'Interacted purchase loses its new indicator');
      await first.locator('.wx-inline-info').click();assert.equal(await page.locator('.wx-sheet[open][data-kind="station-detail"]').count(),1,'Inline info explains the real upgrade');assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.equal(await page.locator('[data-wx-station-drawer]:visible').count(),0,'Back from inline details returns to the world');assert.deepEqual(await geometry(page),before);
      await page.locator('.wx-inline-upgrade[data-state="locked"]').first().locator('.wx-inline-buy').click();assert.equal(await page.locator('.wx-sheet[open][data-kind="station-detail"]').count(),1,'Locked inline control exposes requirements');await page.locator('[data-wx-close]').click();
      await shot(page,'opening-world-'+width+(largeText ? '-text130' : ''));
      await page.locator('[data-wx-nav="upgrades"]').click();assert.equal(await page.locator('.wx-station-row').count(),0,'Drawer contains no duplicate starter purchases');assert.equal(await page.locator('[data-wx-do="station-core:greenway:path"]').count(),1,'Drawer points back to the station controls');
      assert.deepEqual(await geometry(page),before,'Drawer preserves HUD, dock, world and scroll');await shot(page,'opening-drawer-'+width);
      await page.locator('[data-wx-do="station-core:greenway:path"]').click();assert.equal(await page.locator('[data-wx-station-drawer]:visible').count(),0);await sameInlineNodes(page);await page.locator('[data-wx-wallet]').click();assert.equal(await page.locator('.wx-wallet-list .wx-menu').count(),H.Core.getView(opening).onboarding.currencies.length);await page.locator('[data-wx-close]').click();
      report.viewports.push({width,height,largeText:!!largeText,...before});await context.close();
    }
    report.flows.push('320/390/430/915 and 320 text130: three inline starters, two initial locks, real main-screen purchase, stable 104px strip and art, retained button nodes; drawer has no duplicate starters and returns to core controls; wallet exposes currencies');
    for(const largeText of [false,true]) {
      const seed=StationFixtures.buildReady(),firstReady=H.Core.getView(seed).stations.currentStation.skills.find(row=>row.state==='ready');StationFixtures.act(seed,{type:'onboarding-visit',id:'tiers',intendedAction:firstReady.unlockAction});for(let n=0;n<40 && H.Core.getView(seed).onboarding.active;n++){const active=H.Core.getView(seed).onboarding.active;StationFixtures.act(seed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(seed).onboarding.active,null);F.announceDiscoveries(seed);
      const {page,context}=await open(320,740,seed);await dismissNotices(page);if(largeText)await page.addStyleTag({content:text130});
      const before=await geometry(page),artBefore=await inlineGeometry(page);await readableInlinePrices(page);await rememberInlineNodes(page);
      const id=await page.locator('.wx-inline-upgrade[data-ready="true"]').first().getAttribute('data-wx-inline-skill');assert(id,'An actually earned core upgrade becomes ready in its existing cell');const unlock=page.locator('[data-wx-inline-skill="'+id+'"]');
      assert.match(await unlock.locator('.wx-inline-buy').getAttribute('data-wx-do'),/^inline-unlock:/);await unlock.locator('.wx-inline-buy').click();assert((await saved(page)).stations.unlocked.includes(id));assert.equal(await unlock.getAttribute('data-state'),'learned');assert.equal(await unlock.getAttribute('data-ready'),'false');
      await unlock.locator('.wx-inline-buy').click();assert.equal((await saved(page)).stations.ranks[id],1,'Unlocked core control buys the real first rank');assert.deepEqual(await geometry(page),before);assert.deepEqual(await inlineGeometry(page),artBefore,'Unlock and subsequent buy never alter the reserved art or strip');await sameInlineNodes(page);
      await shot(page,'inline-unlock-320'+(largeText ? '-text130' : ''));await context.close();
    }
    const bulkSeed=StationFixtures.lesson('bulk');
    F.completeAreaGuides(bulkSeed);for(let n=0;n<40 && H.Core.getView(bulkSeed).onboarding.active;n++){const active=H.Core.getView(bulkSeed).onboarding.active;StationFixtures.act(bulkSeed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(bulkSeed).onboarding.active,null);StationFixtures.act(bulkSeed,{type:'onboarding-visit',id:'techniques'});for(let n=0;n<40 && H.Core.getView(bulkSeed).onboarding.active;n++){const active=H.Core.getView(bulkSeed).onboarding.active;StationFixtures.act(bulkSeed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(bulkSeed).onboarding.active,null);F.announceDiscoveries(bulkSeed);StationFixtures.act(bulkSeed,{type:'expedition-batch',count:1});bulkSeed.stations.encounter.remaining=0;
    for(const largeText of [false,true]) {
      const {page,context}=await open(320,740,bulkSeed);await dismissNotices(page);if(largeText)await page.addStyleTag({content:text130});
      await page.evaluate(()=>{const world=document.querySelector('.wx-station-world'),strip=world.querySelectorAll('.wx-station-controls')[1];world.scrollTop+=strip.getBoundingClientRect().bottom-world.getBoundingClientRect().bottom;});await page.waitForTimeout(100);await page.clock.runFor(200);
      const collision=await page.evaluate(()=>{const world=document.querySelector('.wx-station-world'),strip=world.querySelectorAll('.wx-station-controls')[1];return {bottom:strip.getBoundingClientRect().bottom,worldBottom:world.getBoundingClientRect().bottom,buttons:[...strip.querySelectorAll('button')].map(button=>{const rect=button.getBoundingClientRect(),hit=document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2);return {command:button.dataset.wxDo,covered:!hit || hit!==button && !button.contains(hit),hit:hit?.className};})};});
      assert(Math.abs(collision.bottom-collision.worldBottom)<1,'Lower station strip can meet the world viewport bottom');assert.deepEqual(collision.buttons.filter(button=>button.covered),[],'Focus and optional cache never cover the inline controls at the viewport bottom');
      const before=await geometry(page),artBefore=await inlineGeometry(page);await rememberInlineNodes(page);
      const station=page.locator('.wx-station-segment').nth(1);assert(await station.locator('.wx-inline-quantity').isVisible(),'An earned batch selector appears beside the station title');await station.locator('.wx-inline-quantity').click();assert.equal(await page.locator('.wx-sheet[open][data-kind="batch"]').count(),1,'Inline quantity uses the shared batch dialog');
      await page.locator('[data-wx-do="batch:5"]').click();if(await page.locator('.wx-sheet[open]').count())await page.locator('[data-wx-close]').click();assert.equal((await saved(page)).expedition.batch,5);assert.match(await page.locator('.wx-inline-quantity').first().innerText(),/×5/);
      const first=station.locator('.wx-inline-upgrade').first(),id=await first.getAttribute('data-wx-inline-skill'),rank=(await saved(page)).stations.ranks[id];assert.equal(await first.locator('.wx-inline-count').innerText(),'×5');await readableInlinePrices(page);await first.locator('.wx-inline-buy').click();assert.equal((await saved(page)).stations.ranks[id],rank+5,'Inline batch button buys exactly the earned five ranks');assert.deepEqual(await geometry(page),before);assert.deepEqual(await inlineGeometry(page),artBefore);await readableInlinePrices(page);await sameInlineNodes(page);
      await shot(page,'inline-batch-320'+(largeText ? '-text130' : ''));await context.close();
    }
    report.flows.push('320 and text130: actual ready unlock and five-rank batch purchase retain each canvas, cell, info and buy node, reserved 104px strip and scene aspect; earned title selector uses the shared batch dialog');
    const established=mature();assert(H.Core.act(established,{type:'expedition-select',areaId:'quarry'}).ok);F.announceDiscoveries(established);
    const {page,context}=await open(390,844,established);
    await dismissNotices(page);
    assert.equal(await page.locator('.wx-station-segment').count(),5);await inlineGeometry(page);await shot(page,'quarry-stacked-world-390');
    await page.locator('.wx-station-world').evaluate(node=>node.scrollTop=380);const scrolled=await geometry(page);await page.locator('[data-wx-nav="upgrades"]').click();await shot(page,'quarry-station-drawer-390');assert.deepEqual(await geometry(page),scrolled);
    await page.locator('[data-wx-do="upgrade-scope:area"]').click();assert.equal(await page.locator('.wx-station-row').count(),3);await shot(page,'quarry-area-drawer-390');assert.deepEqual(await geometry(page),scrolled);
    await page.locator('[data-wx-do="upgrade-scope:global"]').click();assert.equal(await page.locator('.wx-station-row').count(),0);assert(await page.locator('.wx-research-row').count()>0);await page.locator('[data-wx-drawer-close]').click();
    await page.locator('[data-wx-station-cache]').click();await shot(page,'quarry-boost-390');assert(await page.locator('[data-wx-station-boost]').isVisible());assert.deepEqual(await geometry(page),scrolled);
    assert.equal(await page.locator('.wx-station-segment canvas').first().getAttribute('data-scene-reduced-motion'),'true');assert.equal(await page.locator('[data-wx-station-cache]').evaluate(node=>getComputedStyle(node).animationName),'none');
    await page.locator('[data-wx-nav="upgrades"]').click();await page.locator('.wx-station-row .wx-upgrade-info').first().click();assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert(await page.locator('[data-wx-station-drawer]').isVisible());assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.deepEqual(await geometry(page),scrolled);
    const swipe=async(from,to)=>{await page.locator('.wx-station-world').dispatchEvent('pointerdown',{pointerId:9,pointerType:'touch',button:0,buttons:1,clientX:from,clientY:370});await page.locator('.wx-station-world').dispatchEvent('pointermove',{pointerId:9,pointerType:'touch',buttons:1,clientX:to,clientY:370});await page.locator('.wx-station-world').dispatchEvent('pointerup',{pointerId:9,pointerType:'touch',button:0,clientX:to,clientY:370});await page.clock.runFor(500);};
    await swipe(320,70);assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'watchtower');await swipe(70,320);assert.equal(await page.locator('.wx-game').getAttribute('data-area'),'quarry');assert.equal((await geometry(page)).scroll,scrolled.scroll);
    await page.clock.runFor(200);await page.reload();await page.clock.runFor(2000);await ready(page);await dismissNotices(page);assert.equal((await geometry(page)).scroll,scrolled.scroll,'World scroll survives app patch reload');
    await page.locator('[data-wx-nav="upgrades"]').click();await page.evaluate(()=>{const nodes=[...document.querySelectorAll('.wx-game button,.wx-game strong,.wx-game small,.wx-game .wx-station-effect')];for(const node of nodes){const style=getComputedStyle(node);node.style.fontSize=parseFloat(style.fontSize)*1.25+'px';node.style.lineHeight=parseFloat(style.lineHeight)*1.25+'px';}});assert((await geometry(page)).horizontal<=1 && (await geometry(page)).vertical<=1);await shot(page,'quarry-text125-390');await page.locator('[data-wx-drawer-close]').click();
    report.flows.push('Reduced motion stays static; native Back closes nested detail then drawer, mandatory first guide stays; horizontal swipes restore each area scroll; app patch reload restores scroll;125% text remains within viewport');
    report.flows.push('Five independently illustrated stations; vertical scroll survives Station/Area/Global drawer scopes and boost; cache grants canonical reward and displays area boost');
    for(const area of Stations.Content.AREAS) {await page.locator('[data-wx-objective]').click();await page.locator('[data-wx-area="'+area.id+'"]').click();await page.clock.runFor(1000);await ready(page);await inlineGeometry(page);await shot(page,'area-'+area.id+'-world-390');}
    await context.close();assert.deepEqual(report.errors,[]);report.flows.push('Every area renders its own source art with no runtime errors');
    const lessons=['quarry','watchtower','workshop','ruins','harbor','cards','equipment','tiers','techniques','guild-upgrades','projects','plans','configuration','refit','shop','bulk','focus','automation','reserves','planner'].map(id=>({id,label:id,seed:()=>StationFixtures.lesson(id)}));
    lessons.push({id:'tiers',label:'tiers-build',seed:()=>{const state=StationFixtures.buildReady();StationFixtures.act(state,{type:'onboarding-visit',id:'tiers',intendedAction:{type:'station-build',id:Stations.Content.STATIONS[1].id}});return state;}});
    lessons.push({id:'tiers',label:'tiers-area',seed:()=>{const state=StationFixtures.buildReady(),station=Stations.Content.STATIONS[1];StationFixtures.act(state,{type:'station-build',id:station.id});StationFixtures.act(state,{type:'station-select',id:station.id});while((state.stations.ranks[station.skillIds[0]] || 0)<5)StationFixtures.act(state,{type:'station-skill-buy',id:station.skillIds[0],count:1});const row=Stations.view(state).currentArea.areaUpgrades.find(row=>row.ready);assert(row,'Second station and actual20ranks earn Area improvement');StationFixtures.act(state,{type:'onboarding-visit',id:'tiers',intendedAction:row.unlockAction});return state;}});
    lessons.push({id:'expansion',label:'expansion',seed:()=>{const state=StationFixtures.expansionReady();StationFixtures.act(state,{type:'onboarding-visit',id:'expansion'});return state;}});
    lessons.push({id:'plans',label:'plans-early',seed:sortingOnly});
    for(const lesson of lessons) {
      const {id,label}=lesson,seed=lesson.seed();F.announceDiscoveries(seed);
      const {page,context}=await open(390,844,seed);const expected=H.Core.getView(seed).onboarding.guides.find(row=>row.id===id).steps.length,trace=[];
      async function saved() {return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});}
      try {
        for(let n=0;n<30 && (await saved()).onboarding.practice.progress[id]<expected;n++) {
          await page.clock.runFor(250);const coach=page.locator('.wx-guide[open]');if(!await coach.count() && await page.locator('.wx-sheet[open][data-kind="collection-result"]').count()){trace.push({step:'result',target:'Done'});await page.locator('.wx-sheet .wx-confirm').click();}if(!await coach.count())await page.clock.runFor(1500);assert.equal(await coach.count(),1,id+' lesson remains active');
          while(/^currency:/.test(await coach.getAttribute('data-step'))){await page.locator('[data-guide-next]').click();await page.clock.runFor(100);}
          assert.equal(await coach.getAttribute('data-missing'),'false',id+' real destination is visible');const target=page.locator('[data-guide-target]');assert.equal(await target.count(),1);
          trace.push({step:await coach.getAttribute('data-step'),target:await target.getAttribute('data-wx-do') || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close')});
          if(label==='plans-early' && await page.locator('.wx-sheet[open][data-kind="choice"]').count()){const actual=H.Core.getView(await saved()).stations.currentArea.stations.filter(station=>station.status==='built');assert.deepEqual(await page.locator('[data-wx-operation-station]').evaluateAll(nodes=>nodes.map(node=>({id:node.dataset.wxOperationStation,rate:node.querySelector('small').textContent}))),actual.map(station=>({id:station.id,rate:station.outputText})),'Processing shows canonical new-station rates');}
          await shot(page,'lesson-'+label+'-'+n);
          if(await target.getAttribute('data-wx-reserve')!==null)await target.fill(H.Core.getView(seed).onboarding.active?.requiredAction?.amount || '10');else await target.click();
          await page.clock.runFor(300);
        }
        const result=await saved();assert.equal(result.onboarding.practice.progress[id],expected,JSON.stringify({id,trace}));assert(H.Core.validateState(result).valid,id+' stays valid');
        if(label==='tiers-area')assert(result.stations.areaUnlocked.includes('area:greenway:training'),'Area lesson claims the actual Area improvement');
        if(label==='tiers-build')assert(result.stations.built.includes('greenway:porter-camp'),'Build lesson constructs actual second station');
        if(label==='expansion')assert(result.expedition.areas.quarry,'Expansion opens the actual next area');
        if(label==='plans-early'){assert.equal(result.expedition.areas.quarry.choice,'rich');const station=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='sorting'),before=Stations.stationEconomy(seed,station),after=Stations.stationEconomy(result,station),knowledge=economy=>economy.byproducts.filter(row=>row.resource==='knowledge').reduce((sum,row)=>sum+row.rate,0);assert(after.primary>before.primary && knowledge(after)<knowledge(before),'Early Ore priority changes actual ore versus knowledge production');}
        report.flows.push('Earned '+label+' lesson completes through actual controls: '+JSON.stringify(trace));
      }catch(error){await shot(page,'lesson-'+label+'-FAILED');process.stderr.write(JSON.stringify({output,id,label,trace,body:await page.locator('body').innerText(),errors:report.errors},null,2)+'\n');throw error;}finally{await context.close();}
    }
    assert.deepEqual(report.errors,[]);
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));process.stdout.write(JSON.stringify({ok:true,output,flows:report.flows,errors:report.errors},null,2)+'\n');
  }catch(error){for(const context of browser.contexts())for(const page of context.pages())if(!page.isClosed()){await page.screenshot({path:path.join(output,'FAILED-'+browser.contexts().indexOf(context)+'-'+context.pages().indexOf(page)+'.png')});report.flows.push('FAILED: '+error.message+'; visible screen: '+await page.locator('body').innerText());}throw error;}finally{fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
