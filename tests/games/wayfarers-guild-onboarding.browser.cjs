'use strict';

// Exact offline bundle; funded fixtures test UI behavior, not natural pacing.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const {chromium} = require('playwright');
const {bundle} = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Onboarding = require('../../js/games/wayfarers-guild/onboarding');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-onboarding-')));
const evidence = {viewports:[],flows:[],errors:[],browser:'Browser plugin not available; repository Playwright workflow used.'};
const clone = state => JSON.parse(JSON.stringify(state));
const guide = page => page.locator('.wx-guide[open]');
const sheet = page => page.locator('.wx-sheet[open]');
const command = (page,id) => page.locator('[data-wx-do=' + JSON.stringify(id) + ']:visible');
async function state(page) { return page.evaluate(() => {document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;}); }
async function shot(page,name) {await page.clock.runFor(50);await page.waitForTimeout(70);await page.screenshot({path:path.join(output,name+'.png')});}
async function anchors(page) {return Promise.all(['.wx-header','.wx-world','.wx-dock','.wx-nav'].map(selector => page.locator(selector).boundingBox()));}
async function geometry(page) {
  const result=await page.evaluate(() => {
    const box=node => {const r=node.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right};};
    const g=document.querySelector('.wx-guide[open]');
    const card=g?.querySelector('.wx-guide-card');
    return {width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth-innerWidth,vertical:document.body.scrollHeight-innerHeight,card:card && box(card),ring:g && box(g.querySelector('.wx-guide-ring')),buttons:g ? [...g.querySelectorAll('button')].map(box) : [],missing:g?.dataset.missing,focusInside:!g || g.contains(document.activeElement)};
  });
  assert(result.overflow<=1 && result.vertical<=1,'Game viewport remains fixed: '+JSON.stringify(result));
  if(result.card) {
    assert(result.card.x>=0 && result.card.y>=0 && result.card.right<=result.width+1 && result.card.bottom<=result.height+1,'Coach fits viewport: '+JSON.stringify(result));
    assert(result.buttons.every(b=>b.width>=47.5 && b.height>=47.5 && b.bottom<=result.height+1),'48px guide controls fit');
    assert.equal(result.missing,'false','Real earned target is visible');
    assert(result.focusInside,'Guide owns keyboard focus');
  }
  return result;
}
function complete(state,id) {
  const definition=H.Core.getView(state).onboarding.guides.find(item=>item.id===id);
  if(definition.complete)return;
  assert(H.Core.act(state,definition.visitAction).ok);
  for(let i=0;i<3-definition.progress;i++)assert(H.Core.act(state,H.Core.getView(state).onboarding.active.action).ok);
}
async function run() {
  fs.mkdirSync(output,{recursive:true});
  const files=path.join(output,'onboarding-bundle');bundle(files);
  const server=http.createServer((request,response)=>{
    const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/');
    const file=path.resolve(files,'.'+pathname);
    if(!file.startsWith(files+path.sep)){response.writeHead(403).end();return;}
    fs.readFile(file,(error,bytes)=>{if(error){response.writeHead(404).end();return;}response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'})[path.extname(file)]||'application/json');response.end(bytes);});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  async function open(width,height,seed,offline=0) {
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    seed=clone(seed);seed.lastUpdate=1000;
    const record=Storage.createStore({storage:null,now:()=>1000}).export(seed);assert(record.ok,record.message);
    await context.addInitScript(({key,value})=>{if(!sessionStorage.getItem('seeded')){localStorage.setItem(key,value);sessionStorage.setItem('seeded','1');}},{key:Storage.SAVE_KEY,value:record.text});
    const page=await context.newPage();page.setDefaultTimeout(8000);page.on('pageerror',error=>evidence.errors.push(error.message));
    await page.clock.install({time:new Date(1000+offline*1000)});
    await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.locator('.wx-game').waitFor();await page.clock.runFor(50);
    return {context,page};
  }
  try {
    for(const [width,height] of [[320,740],[390,844],[915,390]]) {
      const {context,page}=await open(width,height,H.Core.createState(1000));
      const before=await anchors(page);
      await page.clock.runFor(900);
      assert.equal(await guide(page).getAttribute('data-step'),'purpose');
      const purposeGeometry=await geometry(page);await shot(page,'trail-purpose-'+width);
      assert.equal((await state(page)).expedition.areas.greenway.ranks.boots,0,'Guide needs no purchase');
      await page.locator('[data-guide-next]').click();
      assert.equal(await guide(page).getAttribute('data-step'),'operation');
      assert(await page.locator('[data-guide-next]').evaluate(node=>node===document.activeElement),'Advanced step focuses its re-enabled Next button');
      assert.match(await page.locator('[data-guide-announcement]').textContent(),/Step 2 of 3/);
      await page.evaluate(()=>{window.guideAnnouncements=0;new MutationObserver(records=>{window.guideAnnouncements+=records.length;}).observe(document.querySelector('[data-guide-announcement]'),{childList:true,characterData:true,subtree:true});});
      await page.clock.runFor(1200);assert.equal(await page.evaluate(()=>window.guideAnnouncements),0,'Unchanged ticks never repeat the screen-reader step announcement');
      assert.equal((await state(page)).onboarding.progress.greenway,1);
      const target=await page.locator('[data-wx-buy=boots]').boundingBox();
      await page.mouse.click(target.x+target.width/2,target.y+target.height/2);
      assert.equal((await state(page)).expedition.areas.greenway.ranks.boots,0,'Scrim/spotlight never buys');
      await shot(page,'trail-operation-'+width);
      await page.reload();await page.locator('.wx-game').waitFor();await page.clock.runFor(900);
      assert.equal(await guide(page).getAttribute('data-step'),'operation','Reload resumes exact acknowledged step');
      await page.keyboard.press('Escape');
      assert.equal(await guide(page).count(),0);assert.equal(await sheet(page).getAttribute('data-kind'),'options','First area Back opens game options');
      await page.clock.runFor(2000);assert.equal(await guide(page).count(),0,'Options outrank guide');
      await page.locator('[data-wx-close]').click();await page.clock.runFor(900);
      assert.equal(await guide(page).getAttribute('data-step'),'operation');
      await page.locator('[data-guide-next]').click();assert(await page.locator('[data-guide-next]').evaluate(node=>node===document.activeElement),'Final step focuses its re-enabled Start working button');await geometry(page);await shot(page,'trail-next-step-'+width);
      await page.locator('[data-guide-next]').click();assert.equal(await guide(page).count(),0);
      const finished=await state(page);assert.equal(finished.onboarding.progress.greenway,3);assert(finished.onboarding.rewardClaims.includes('greenway'));
      assert.deepEqual(await anchors(page),before,'Tour never moves world/dock/navigation');
      await page.locator('[data-wx-options]').click();await command(page,'guide-replay:greenway').click();
      for(let i=0;i<3;i++)await page.locator('[data-guide-next]').click();
      assert.equal((await state(page)).onboarding.rewardClaims.filter(id=>id==='greenway').length,1,'Replay never repeats reward');
      evidence.viewports.push({width,height,openGuideGeometry:purposeGeometry,geometry:await geometry(page)});
      await context.close();
    }
    evidence.flows.push('fresh zero-spend guide, real highlighted control, inert scrim, saved-step reload, Android-style Back/Options/resume, single reward and replay, stable geometry at three viewports');
    const callback=await open(390,844,H.Core.createState(1000));const kb=callback.page;
    await kb.clock.runFor(900);
    for(const stepId of ['operation','next-step']) {
      const focus=await kb.evaluate(()=>{document.activeElement?.blur();document.querySelector('[data-guide-next]').click();const dialog=document.querySelector('.wx-guide[open]');return {step:dialog?.dataset.step,inside:dialog?.contains(document.activeElement),next:document.activeElement?.hasAttribute('data-guide-next')};});
      assert.equal(focus.step,stepId);assert(focus.inside && focus.next,'Synchronous native-style click restores the enabled next-step action');
    }
    await kb.evaluate(()=>document.querySelector('[data-guide-leave]').click());assert.equal(await guide(kb).count(),0);
    await kb.clock.runFor(100);assert(await kb.evaluate(()=>document.querySelector('.wx-sheet[open]')?.contains(document.activeElement)),'Completed callback does not steal focus from Game options');
    await callback.context.close();
    evidence.flows.push('pointer and synchronous native-style step callbacks focus only the current re-enabled guide action, never steal focus after leaving');
    for(const [width,height] of [[320,740],[915,390]]) {
      const aged=await open(width,height,H.Core.createState(1000));const gp=aged.page;
      await gp.clock.runFor(900);await gp.locator('[data-guide-next]').click();
      await gp.clock.runFor(180000);assert.equal((await state(gp)).expedition.completed,true,'The landmark finishes while the guide remains on operation');
      await gp.reload();await gp.locator('.wx-game').waitFor();await gp.clock.runFor(900);
      assert.equal(await guide(gp).getAttribute('data-step'),'operation');
      await gp.locator('[data-guide-next]').click();await gp.clock.runFor(100);
      assert.match(await gp.locator('#wx-guide-body').textContent(),/Expand Quarry starts your next landmark/);
      const target=await gp.locator('[data-wx-expand]').boundingBox(),ring=await gp.locator('.wx-guide-ring').boundingBox(),status=await gp.locator('[data-wx-world-label]').boundingBox();
      assert(target && ring,'Available expansion has a real anchor');
      assert(ring.x<=target.x && ring.y<=target.y && ring.x+ring.width>=target.x+target.width && ring.y+ring.height>=target.y+target.height,'Spotlight surrounds the actual expansion control');
      assert(ring.y>=status.y+status.height || ring.y+ring.height<=status.y || ring.x>=status.x+status.width || ring.x+ring.width<=status.x,'Producing-status label is not the goal anchor');
      const bounds=await geometry(gp);await shot(gp,'trail-aged-expansion-'+width);
      await gp.locator('[data-guide-next]').click();assert.equal((await state(gp)).expedition.completed,true,'Explanation never starts the expansion');
      assert(await gp.locator('[data-wx-expand]').isEnabled(),'Explained expansion remains ready to use');
      evidence.viewports.push({width,height,scenario:'aged-resumed-expansion',openGuideGeometry:bounds});
      await aged.context.close();
    }
    evidence.flows.push('paused Trail guide ages through landmark completion, reload resumes exact step, final spotlight encloses Expand Quarry and excludes production status at320/915 without starting expansion');
    const seeded=H.Core.createState(1000);H.fund(seeded);complete(seeded,'greenway');
    const {context,page}=await open(390,844,seeded);
    await page.locator('[data-wx-buy=boots]').click();await page.locator('[data-wx-buy=boots]').click();await page.clock.runFor(1000);
    assert.equal(await sheet(page).getAttribute('data-kind'),'onboarding-notice');
    assert.match(await sheet(page).innerText(),/Porters available/);
    await shot(page,'porters-available-390');
    await command(page,'onboarding-open:ready:area:greenway:porters').click();
    assert.equal(await page.locator('[data-wx-buy=porters]').count(),1);
    await page.clock.runFor(2000);assert.equal(await sheet(page).count(),0,'Unlock & go does not chain success modal');
    await shot(page,'porters-go-to-390');await context.close();
    evidence.flows.push('single Porters available → atomic Unlock & go → actual new row, no duplicate unlocked popup');
    for(const [width,height] of [[320,740],[915,390]]) {
      const mature=H.mature();mature.onboarding=Onboarding.migrate(mature);
      const {context,page}=await open(width,height,mature);
      for(const id of ['watchtower','greenway','quarry','workshop','ruins','harbor']) {
        if((await state(page)).expedition.selectedArea!==id){await page.locator('[data-wx-objective]').click();await page.locator('[data-wx-area='+id+']').click();}
        await page.clock.runFor(1000);
        assert.equal(await guide(page).getAttribute('data-guide'),id);
        for(let step=0;step<3;step++){
          if (step === 2 && ['quarry','watchtower'].includes(id)) {
            const target=await page.locator(id==='quarry' ? '[data-wx-objective]' : '[data-wx-nav="upgrades"]').boundingBox(),ring=await page.locator('.wx-guide-ring').boundingBox();
            assert(ring.x<=target.x+4 && ring.y<=target.y+4 && ring.x+ring.width>=target.x+target.width-4 && ring.y+ring.height>=target.y+target.height-4,'Final '+id+' step highlights the actual '+(id==='quarry' ? 'area picker' : 'Upgrades destination')+' with only viewport-edge ring clipping: '+JSON.stringify({ring,target}));
          }
          await geometry(page);await shot(page,id+'-step'+step+'-'+width);await page.locator('[data-guide-next]').click();
          if(step<2)assert(await page.locator('[data-guide-next]').evaluate(node=>node===document.activeElement),'Every new area step restores focus after its durable transaction');
        }
        assert.equal((await state(page)).onboarding.progress[id],3);
      }
      await context.close();
    }
    evidence.flows.push('all six returning-player area guides, actual Plans/upgrade/goal anchors and no retroactive reward wave');
    const failing=await open(390,844,H.Core.createState(1000));const fp=failing.page;
    await fp.clock.runFor(900);
    await fp.evaluate(()=>{window.savedSetItem=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');return window.savedSetItem.call(this,key,value);};});
    await fp.locator('[data-guide-next]').click();
    assert.equal(await guide(fp).getAttribute('data-step'),'purpose','Failed save never acknowledges a step');
    assert(await fp.locator('[data-guide-next]').evaluate(node=>node===document.activeElement),'Failed save restores focus to Retry after re-enabling it');
    assert.equal(await fp.locator('[data-guide-next]').innerText(),'Retry');
    assert.equal((await state(fp)).onboarding.progress.greenway,0);
    await shot(fp,'guide-save-failure-390');
    await fp.locator('[data-guide-leave]').click();assert.equal(await sheet(fp).getAttribute('data-kind'),'options','Failed save still permits safe UI-only exit');
    await command(fp,'settings').click();await fp.locator('[data-export=text]').click();
    assert.equal(JSON.parse(await fp.locator('#wg-save-text').inputValue()).state.onboarding.progress.greenway,0,'Export recovery remains reachable without acknowledging the step');
    await fp.locator('.wg-dialog[open] [data-close-dialog]').first().click();
    await fp.evaluate(()=>{Storage.prototype.setItem=window.savedSetItem;});await fp.locator('[data-wx-save-alert]').click();await fp.clock.runFor(900);
    assert.equal(await guide(fp).getAttribute('data-step'),'purpose','Retry saves without silently acknowledging');
    await fp.locator('[data-guide-next]').click();assert.equal((await state(fp)).onboarding.progress.greenway,1);
    await fp.setViewportSize({width:915,height:390});await fp.waitForTimeout(100);await fp.clock.runFor(500);await geometry(fp);await shot(fp,'guide-rotated-915');
    for(let i=0;i<6;i++){await fp.keyboard.press('Tab');assert(await fp.evaluate(()=>document.querySelector('.wx-guide').contains(document.activeElement)),'Tab remains in guide');}
    await fp.evaluate(()=>{document.querySelector('.wx-upgrade-info').style.display='none';document.querySelector('[data-wx-canvas]').style.display='none';dispatchEvent(new Event('resize'));});
    await fp.clock.runFor(100);assert.equal(await guide(fp).getAttribute('data-missing'),'true');
    await fp.locator('[data-guide-next]').click();assert.equal((await state(fp)).onboarding.progress.greenway,1,'Missing actual target cannot acknowledge');
    await shot(fp,'guide-missing-target-915');
    await fp.evaluate(()=>{document.querySelector('.wx-upgrade-info').style.display='';document.querySelector('[data-wx-canvas]').style.display='';dispatchEvent(new Event('resize'));});
    await fp.clock.runFor(100);await geometry(fp);
    await fp.reload();await fp.locator('.wx-game').waitFor();await fp.clock.runFor(900);assert.equal(await guide(fp).getAttribute('data-step'),'operation');
    await failing.context.close();evidence.flows.push('failed durable step save stays visible with Retry, target loss does not complete, rotation and keyboard focus remain safe, reload preserves last acknowledged step');
    const backlog=H.Core.createState(1000);H.fund(backlog);complete(backlog,'greenway');
    for(let i=0;i<2;i++)assert(H.Core.act(backlog,H.Core.getView(backlog).expedition.cards[0].action).ok);
    H.advance(backlog,3600);
    const offline=await open(320,740,backlog,3600);const op=offline.page;
    await op.clock.runFor(1000);assert.equal(await sheet(op).getAttribute('data-kind'),'return','Return summary wins over discovery queue');
    await command(op,'return-close').click();await op.clock.runFor(1000);assert.equal(await sheet(op).getAttribute('data-kind'),'onboarding-notice');
    assert(await op.locator('.wx-inbox-list .wx-menu').count()>1);await shot(op,'offline-discovery-wave-320');
    await command(op,'onboarding-open:ready:area:greenway:porters').click();
    assert(H.Core.getView(await state(op)).onboarding.notice.items.some(item=>item.id==='ready:area:greenway:scouts'),'Unlocking an offline tier really creates an eligible successor');
    await op.clock.runFor(5000);assert.equal(await sheet(op).count(),0,'Successor does not chain a new popup from the same offline wave');
    await op.locator('[data-wx-options]').click();await command(op,'onboarding-inbox').click();
    assert(await command(op,'onboarding-open:ready:area:greenway:scouts').count(),'Suppressed earned successor remains directly reachable');
    await shot(op,'offline-inbox-after-claim-320');await offline.context.close();
    evidence.flows.push('offline summary priority, one multi-discovery wave, real claim-created successor suppressed without hiding its direct inbox route');
    const collections=H.mature();assert(H.Core.act(collections,{type:'collection-unlock',kind:'cards'}).ok);assert(H.Core.act(collections,{type:'collection-unlock',kind:'equipment'}).ok);
    collections.onboarding=Onboarding.migrate(collections);complete(collections,collections.expedition.selectedArea);complete(collections,'cards');complete(collections,'equipment');
    const optional=await open(390,844,collections);const cp=optional.page;
    const beforeOptional=await state(cp);
    for(const kind of ['cards','equipment']){
      await cp.locator('[data-wx-collection='+kind+']').click();await command(cp,kind==='cards'?'card-help':'gear-help').click();await command(cp,'guide-replay:'+kind).click();
      for(let step=0;step<3;step++){await geometry(cp);await shot(cp,kind+'-replay'+step+'-390');await cp.locator('[data-guide-next]').click();}
    }
    const afterOptional=await state(cp);for(const key of ['cards','gear','decks','equipped','scrolls','ink','revision'])assert.deepEqual(afterOptional.collection[key],beforeOptional.collection[key],'Optional guides leave '+key+' unchanged');assert.deepEqual(afterOptional.onboarding.rewardClaims,beforeOptional.onboarding.rewardClaims);
    await optional.context.close();evidence.flows.push('completed Cards and Equipment guides can be replayed over real collection targets without spending or rewards');
    const firstCollections=H.mature();firstCollections.onboarding=Onboarding.migrate(firstCollections);complete(firstCollections,firstCollections.expedition.selectedArea);
    const learning=await open(320,740,firstCollections);const lp=learning.page;
    for(const kind of ['cards','equipment']) {
      await lp.locator('[data-wx-collection='+kind+']').click();await lp.clock.runFor(900);assert.equal(await guide(lp).count(),0,'Starter introduction remains an explicit free claim');
      await command(lp,'collection-intro:'+kind).click();await lp.clock.runFor(1000);assert.equal(await guide(lp).getAttribute('data-guide'),kind,'First claimed feature starts its saved guide');
      const inventory=clone((await state(lp)).collection);
      await lp.evaluate(()=>{window.featureSetItem=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');return window.featureSetItem.call(this,key,value);};});
      await lp.locator('[data-guide-next]').click();assert.equal(await guide(lp).getAttribute('data-step'),'purpose');assert.equal(await lp.locator('[data-guide-next]').innerText(),'Retry');assert.equal((await state(lp)).onboarding.progress[kind],0);
      await lp.evaluate(()=>{Storage.prototype.setItem=window.featureSetItem;});await lp.locator('[data-guide-next]').click();assert.equal(await guide(lp).getAttribute('data-step'),'purpose');
      await lp.locator('[data-guide-next]').click();await lp.keyboard.press('Escape');assert.equal(await guide(lp).count(),0);
      await lp.locator('[data-wx-collection='+kind+']').click();await lp.clock.runFor(1000);assert.equal(await guide(lp).getAttribute('data-step'),'operation');
      await shot(lp,kind+'-first-visit-320');
      await lp.locator('[data-guide-next]').click();await lp.clock.runFor(300);
      const libraryBounds=await lp.locator('[data-guide-next]').boundingBox();
      for(let frame=0;frame<8;frame++){await lp.clock.runFor(100);assert.deepEqual(await lp.locator('[data-guide-next]').boundingBox(),libraryBounds,'Tall collection grid keeps the guide action stable across resize/scroll frames');}
      await geometry(lp);await shot(lp,kind+'-library-stable-320');await lp.locator('[data-guide-next]').click();
      const learned=await state(lp);assert.equal(learned.onboarding.progress[kind],3);
      for(const key of ['cards','gear','decks','equipped','scrolls','ink','revision'])assert.deepEqual(learned.collection[key],inventory[key],'Feature walkthrough never mutates '+key);
    }
    await learning.context.close();evidence.flows.push('explicit free Cards/Gear starters immediately start mandatory saved first-use guides, pause to Guild overview and resume without spending/equipping');
    const committed=await open(390,844,H.Core.createState(1000));const qp=committed.page;
    await qp.clock.runFor(900);await qp.locator('[data-guide-next]').click();await qp.locator('[data-guide-next]').click();
    await qp.evaluate(()=>{
      window.onboardSet=Storage.prototype.setItem;window.onboardGet=Storage.prototype.getItem;window.onboardWritten=false;
      Storage.prototype.setItem=function(key,value){const result=window.onboardSet.call(this,key,value);if(key===WayfarersStorage.SAVE_KEY)window.onboardWritten=true;return result;};
      Storage.prototype.getItem=function(key){if(key===WayfarersStorage.RESET_KEY && window.onboardWritten)throw new DOMException('Reset record unavailable','SecurityError');return window.onboardGet.call(this,key);};
    });
    await qp.locator('[data-guide-next]').click();
    assert.equal(await guide(qp).count(),0,'Committed completion closes guide even when final fence verification fails');
    assert(await qp.locator('[data-wx-save-alert]').isVisible());
    const written=await qp.evaluate(()=>JSON.parse(window.onboardGet.call(localStorage,WayfarersStorage.SAVE_KEY)).state);
    assert.equal(written.onboarding.progress.greenway,3);assert.equal(written.onboarding.rewardClaims.filter(id=>id==='greenway').length,1);
    await shot(qp,'committed-completion-save-alert-390');
    await qp.evaluate(()=>{Storage.prototype.setItem=window.onboardSet;Storage.prototype.getItem=window.onboardGet;});await qp.locator('[data-wx-save-alert]').click();
    const retried=await state(qp);assert(H.Core.Numbers.toNumber(H.Core.Numbers.sub(retried.resources.coins,written.resources.coins))<1,'Retry only includes sub-second idle income, never a repeated 12-coin reward');assert.equal(retried.onboarding.rewardClaims.filter(id=>id==='greenway').length,1);
    await qp.reload();await qp.locator('.wx-game').waitFor();await qp.clock.runFor(1000);assert.equal(await guide(qp).count(),0);
    await committed.context.close();evidence.flows.push('post-write fence failure retains committed final-step reward, exposes save recovery and never re-awards on retry/reload');
    const importing=await open(390,844,H.Core.createState(1000));const ip=importing.page;
    await ip.locator('[data-wx-options]').click();await command(ip,'settings').click();await ip.clock.runFor(2000);assert.equal(await guide(ip).count(),0,'Settings holds scheduled guide');
    const replacement=H.Core.createState(2000);complete(replacement,'greenway');
    const imported=Storage.createStore({storage:null,now:()=>2000}).export(replacement);assert(imported.ok);
    await ip.locator('#wg-save-text').fill(imported.text);await ip.locator('[data-review-import]').click();await ip.clock.runFor(2000);assert.equal(await guide(ip).count(),0,'Import review outranks guide');
    await ip.locator('[data-confirm-import]').click();await ip.clock.runFor(2500);assert.equal(await guide(ip).count(),0,'Old queued first-step cannot reappear after import');
    assert.equal((await state(ip)).createdAt,2000);assert.equal((await state(ip)).onboarding.progress.greenway,3);await shot(ip,'import-invalidates-guide-390');
    await importing.context.close();evidence.flows.push('ordinary reviewed import invalidates queued old-guild guide and preserves imported completed progress');
    const newArea=H.mature();newArea.onboarding=Onboarding.migrate(newArea);complete(newArea,newArea.expedition.selectedArea);
    for(const key of ['entries','announced','read'])newArea.onboarding[key]=newArea.onboarding[key].filter(id=>id!=='area:harbor');
    H.Core.advance(newArea,1);
    const arriving=await open(320,740,newArea);const ap=arriving.page;
    await ap.clock.runFor(900);assert.equal(await sheet(ap).getAttribute('data-kind'),'onboarding-notice');assert.match(await sheet(ap).innerText(),/Harbor unlocked/);await shot(ap,'harbor-unlocked-320');
    await command(ap,'onboarding-open:area:harbor').click();await ap.clock.runFor(1000);assert.equal(await guide(ap).getAttribute('data-guide'),'harbor');
    assert.equal((await state(ap)).expedition.selectedArea,'harbor');
    await ap.addStyleTag({content:'.wx-guide p {font-size:18.2px!important}.wx-guide h2 {font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}'});
    await ap.clock.runFor(100);await geometry(ap);await shot(ap,'harbor-guide-large-text-320');
    await ap.setViewportSize({width:915,height:390});await ap.waitForTimeout(100);await ap.clock.runFor(500);await geometry(ap);await shot(ap,'harbor-guide-large-text-915');
    await arriving.context.close();evidence.flows.push('Harbor unlocked → Go to actual Harbor → mandatory guide; enlarged browser text + live rotation fit320/915 (native font scaling remains a separate APK check)');
    assert.deepEqual(evidence.errors,[],'No browser runtime errors');
  } finally {
    fs.writeFileSync(path.join(output,'onboarding-browser.json'),JSON.stringify(evidence,null,2));
    await browser.close();await new Promise(resolve=>server.close(resolve));
  }
  console.log(JSON.stringify({result:'PASS',viewports:evidence.viewports.length,flows:evidence.flows.length,output}));
}
run().catch(error=>{console.error(error.stack);process.exitCode=1;});
