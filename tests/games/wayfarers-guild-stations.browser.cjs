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
const report={browser:'In-app browser unavailable: Browser is not available: iab; repository Playwright workflow used.',flows:[],viewports:[],errors:[]};
function fresh() {const state=H.Core.createState(1000);H.fund(state);F.completeAreaGuides(state);F.announceDiscoveries(state);return state;}
function mature() {const state=StationFixtures.mature();state.stations.encounter.remaining=0;F.announceDiscoveries(state);return state;}
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
    for(const [width,height] of [[320,740],[390,844],[430,932],[915,390]]) {
      const {page,context}=await open(width,height,opening);const before=await geometry(page);assert(before.horizontal<=1 && before.vertical<=1);
      assert.equal(await page.locator('.wx-dock:visible').count(),0,'No permanent upgrade dock');assert.equal(await page.locator('.wx-station-row:visible').count(),0,'Main world hides purchases');
      await shot(page,'opening-world-'+width);
      await page.locator('[data-wx-nav="upgrades"]').click();assert.equal(await page.locator('.wx-station-row:visible').count(),3,'Three starter rows in drawer');assert.equal(await page.locator('.wx-station-row[data-state="locked"]:visible').count(),2);
      assert.deepEqual(await geometry(page),before,'Drawer preserves HUD, dock, world and scroll');await shot(page,'opening-drawer-'+width);
      const first=page.locator('.wx-station-row').first();const rankBefore=await first.innerText();await first.locator('.wx-station-buy').click();assert.notEqual(await first.innerText(),rankBefore,'Purchase updates actual row');assert.deepEqual(await geometry(page),before,'Purchase never moves camera');assert.equal(await first.locator('[data-wx-unseen]').count(),0,'Interacted purchase loses its new indicator');
      await page.locator('[data-wx-drawer-close]').click();await page.locator('[data-wx-wallet]').click();assert.equal(await page.locator('.wx-wallet-list .wx-menu').count(),H.Core.getView(opening).onboarding.currencies.length);await page.locator('[data-wx-close]').click();
      report.viewports.push({width,height,...before});await context.close();
    }
    report.flows.push('320/390/430/915: animated main has no upgrade rows, drawer shows three unique starters and two locks, purchase changes real rank with fixed camera; wallet exposes currencies');
    const established=mature();assert(H.Core.act(established,{type:'expedition-select',areaId:'quarry'}).ok);F.announceDiscoveries(established);
    const {page,context}=await open(390,844,established);
    await dismissNotices(page);
    assert.equal(await page.locator('.wx-station-segment').count(),5);await shot(page,'quarry-stacked-world-390');
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
    for(const area of Stations.Content.AREAS) {await page.locator('[data-wx-objective]').click();await page.locator('[data-wx-area="'+area.id+'"]').click();await page.clock.runFor(1000);await ready(page);await shot(page,'area-'+area.id+'-world-390');}
    await context.close();assert.deepEqual(report.errors,[]);report.flows.push('Every area renders its own source art with no runtime errors');
    const lessons=['quarry','watchtower','workshop','ruins','harbor','cards','equipment','tiers','techniques','guild-upgrades','projects','plans','configuration','refit','shop','bulk','focus','automation','reserves','planner'].map(id=>({id,label:id,seed:()=>StationFixtures.lesson(id)}));
    lessons.push({id:'tiers',label:'tiers-build',seed:()=>{const state=StationFixtures.buildReady();state.onboarding.practice.intentions.tiers={type:'station-build',id:Stations.Content.STATIONS[1].id};StationFixtures.act(state,{type:'onboarding-visit',id:'tiers'});return state;}});
    lessons.push({id:'tiers',label:'tiers-area',seed:()=>{const state=StationFixtures.buildReady(),station=Stations.Content.STATIONS[1];StationFixtures.act(state,{type:'station-build',id:station.id});StationFixtures.act(state,{type:'station-select',id:station.id});while((state.stations.ranks[station.skillIds[0]] || 0)<5)StationFixtures.act(state,{type:'station-skill-buy',id:station.skillIds[0],count:1});const row=Stations.view(state).currentArea.areaUpgrades.find(row=>row.ready);assert(row,'Second station and actual20ranks earn Area improvement');state.onboarding.practice.intentions.tiers=row.unlockAction;StationFixtures.act(state,{type:'onboarding-visit',id:'tiers'});return state;}});
    lessons.push({id:'expansion',label:'expansion',seed:()=>{const state=StationFixtures.expansionReady();StationFixtures.act(state,{type:'onboarding-visit',id:'expansion'});return state;}});
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
          await shot(page,'lesson-'+label+'-'+n);
          if(await target.getAttribute('data-wx-reserve')!==null)await target.fill(H.Core.getView(seed).onboarding.active?.requiredAction?.amount || '10');else await target.click();
          await page.clock.runFor(300);
        }
        const result=await saved();assert.equal(result.onboarding.practice.progress[id],expected,JSON.stringify({id,trace}));assert(H.Core.validateState(result).valid,id+' stays valid');
        report.flows.push('Earned '+label+' lesson completes through actual controls: '+JSON.stringify(trace));
      }catch(error){await shot(page,'lesson-'+label+'-FAILED');process.stderr.write(JSON.stringify({output,id,label,trace,body:await page.locator('body').innerText(),errors:report.errors},null,2)+'\n');throw error;}finally{await context.close();}
    }
    assert.deepEqual(report.errors,[]);
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));process.stdout.write(JSON.stringify({ok:true,output,flows:report.flows,errors:report.errors},null,2)+'\n');
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
}
run().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
