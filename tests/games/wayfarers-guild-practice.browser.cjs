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
const PracticeFixtures=require('./helpers/wayfarers-practice.cjs');
const Fixtures = require('./helpers/wayfarers-onboarding.cjs');
const Onboarding = require('../../js/games/wayfarers-guild/onboarding');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-onboarding-')));
const evidence = {sourceHashes:Object.fromEntries(['expedition-ui.js','onboarding-ui.js','expedition-scene.js','practice-lessons.js','core.js'].map(file=>[file,require('node:crypto').createHash('sha256').update(fs.readFileSync(path.resolve(__dirname,'../../js/games/wayfarers-guild',file))).digest('hex')])),lessonFilter:process.env.WAYFARERS_LESSON||null,viewports:[],flows:[],errors:[],browser:'Browser plugin not available; repository Playwright workflow used.'};
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
    return {width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth-innerWidth,vertical:document.body.scrollHeight-innerHeight,card:card && box(card),ring:g && box(g.querySelector('.wx-guide-ring')),buttons:g ? [...g.querySelectorAll('button')].filter(n=>!n.hidden).map(box) : [],headingFits:!g || g.querySelector('h2').scrollWidth<=g.querySelector('h2').clientWidth+1,metaFits:!g || g.querySelector('[data-guide-count]').getBoundingClientRect().right<=card.getBoundingClientRect().right-12,missing:g?.dataset.missing,focusInside:!g || g.contains(document.activeElement) || !!document.activeElement?.closest('[data-guide-target],[data-wx-close],[data-wx-back],[data-wx-options]')};
  });
  assert(result.overflow<=1 && result.vertical<=1,'Game viewport remains fixed: '+JSON.stringify(result));
  if(result.card) {
    assert(result.card.x>=0 && result.card.y>=0 && result.card.right<=result.width+1 && result.card.bottom<=result.height+1,'Coach fits viewport: '+JSON.stringify(result));
    assert(result.buttons.every(b=>b.width>=47.5 && b.height>=47.5 && b.bottom<=result.height+1),'48px guide controls fit');
    assert.equal(result.missing,'false','Real earned target is visible');
    assert(result.focusInside,'Focus stays in the actual target or coach');assert(result.headingFits && result.metaFits,'Coach heading and step count remain within padded bounds');
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
  const files=path.join(output,'practice-bundle');bundle(files);
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
      await page.clock.runFor(900);
      assert.equal(await guide(page).getAttribute('data-step'),'inspect');
      evidence.viewports.push({width,height,step:'inspect',geometry:await geometry(page)});await shot(page,'trail-inspect-'+width);
      await page.locator('[data-upgrade="boots"] .wx-upgrade-info').click();
      await page.clock.runFor(100);
      assert.equal(await guide(page).getAttribute('data-step'),'upgrade');
      evidence.viewports.push({width,height,step:'upgrade',geometry:await geometry(page)});await shot(page,'trail-buy-'+width);
      assert.equal((await state(page)).expedition.areas.greenway.ranks.boots,0);
      await page.locator('[data-wx-practice]').click();await page.clock.runFor(100);
      assert.equal((await state(page)).expedition.areas.greenway.ranks.boots,1);
      assert.equal(await guide(page).getAttribute('data-step'),'operate');
      if(await sheet(page).count()) await page.locator('[data-wx-close]').click();
      await page.locator('[data-wx-world-label]').click();await page.clock.runFor(100);
      assert.equal(await guide(page).count(),0);
      assert.equal((await state(page)).onboarding.practice.progress.greenway,3);
      evidence.viewports.push({width,height});await context.close();
    }
    evidence.flows.push('real upgrade inspection, supplied first rank, actual objective inspector at three viewports');

    for(const [width,height] of [[320,740],[915,390]]) {
      const seed=H.mature();Fixtures.completeAreaGuides(seed);Fixtures.announceDiscoveries(seed);
      const {context,page}=await open(width,height,seed);
      for(const kind of ['cards','equipment']) {
        await page.locator('[data-wx-collection="'+kind+'"]').click();
        await page.locator('[data-wx-do="collection-intro:'+kind+'"]').click();
        await page.clock.runFor(1000);
        for(let action=0;action<25 && await guide(page).count();action++) {
          const step=await guide(page).getAttribute('data-step');
          await geometry(page);await shot(page,kind+'-'+step+'-'+action+'-'+width);
          if(width===320 && kind==='equipment' && step==='scroll' && await page.locator('[data-wx-practice]').count()){await page.addStyleTag({content:'.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-guide button{font-size:16.9px!important}'});await page.clock.runFor(100);await page.waitForFunction(()=>document.querySelector('.wx-guide-card').getBoundingClientRect().bottom<=innerHeight);await geometry(page);await shot(page,'equipment-scroll-large-text-320');await page.setViewportSize({width:915,height:390});await page.waitForTimeout(100);await page.clock.runFor(100);await geometry(page);await shot(page,'equipment-scroll-large-text-915');await page.setViewportSize({width,height});await page.clock.runFor(100);}
          const target=page.locator('[data-guide-target]');
          assert.equal(await target.count(),1,'One actual control is highlighted: '+kind+' '+step);
          await target.click();await page.clock.runFor(1000);
          if(await page.locator('.wx-sheet[data-kind="collection-result"][open]').count()) {
            await page.locator('.wx-sheet[data-kind="collection-result"] .wx-confirm').click();await page.clock.runFor(1000);
          }
        }
        const final=await state(page);
        assert.equal(final.onboarding.practice.progress[kind],kind==='cards'?5:3,kind+' actual lesson completes');
      }
      evidence.flows.push('actual Cards equip/fuse/decks and Equipment equip/Steady/result '+width);
      await context.close();
    }


    for(const [width,height] of [[320,740],[915,390]]) {
      const seed=H.mature();assert(H.Core.act(seed,{type:'refit'}).ok);assert(H.Core.act(seed,{type:'expedition-batch',count:100}).ok);assert(H.Core.act(seed,{type:'expedition-select',areaId:'workshop'}).ok);Fixtures.announceDiscoveries(seed);
      const {context,page}=await open(width,height,seed);await page.clock.runFor(1000);await page.locator('[data-guide-target]').click();await page.clock.runFor(100);
      assert.equal(await guide(page).getAttribute('data-step'),'upgrade');
      const summary=await page.locator('.wx-purchase-summary').innerText();assert.match(summary,/Exactly 1 rank.*0.*1/s);assert.match(summary,/Free/);assert(!summary.includes('100 ranks'));
      const model=H.Core.getView(await state(page)).onboarding.active;assert.equal(model.practicePreview.quantity,1);assert.equal(model.practicePreview.rankAfter,1);
      assert.equal(await page.locator('.wx-sheet [data-wx-do="batch"]').count(),0,'Shared bulk selector does not contradict supplied one-rank purchase');
      await shot(page,'inherited-bulk-exact-practice-'+width);await page.locator('[data-wx-practice]').click();await page.clock.runFor(1000);assert.equal((await state(page)).expedition.areas.workshop.ranks.assembly,1);assert.equal((await state(page)).expedition.batch,100);
      for(let n=0;n<12 && await guide(page).count();n++){await shot(page,'bulk-resume-'+width+'-'+n);await page.locator('[data-guide-target]').click();await page.clock.runFor(1000);}
      if(await sheet(page).count())await page.locator('[data-wx-close]').click();await page.locator('[data-upgrade="assembly"] .wx-upgrade-info').click();assert.match(await page.locator('.wx-purchase-summary').innerText(),/Exactly 100 ranks/);await shot(page,'inherited-bulk-restored-'+width);
      await context.close();
    }
    evidence.flows.push('Inherited ×100 new-area lesson previews exact supplied ×1 effects/cost, buys one rank and restores unchanged shared ×100 afterward');
    for(const id of ['expansion','bulk','focus','plans','configuration','specialization','crew','companions','guild-upgrades','automation','reserves','planner','refit','charter','shop','projects','relics','meals','kits','playbooks','card-archive','card-craft','gear-craft','gear-repair','gear-reforge','supply','caravan'].filter(id=>!process.env.WAYFARERS_LESSON || id===process.env.WAYFARERS_LESSON)) {
      const extra=['projects','relics','meals','kits','playbooks','card-archive','card-craft','gear-craft','gear-repair','gear-reforge','supply','caravan'].includes(id);
      const seed=extra?PracticeFixtures.advanced():H.mature();Fixtures.completeAreaGuides(seed);Fixtures.announceDiscoveries(seed);
      const lesson=H.Core.getView(seed).onboarding.guides.find(g=>g.id===id);
      assert(lesson?.available,'Validated fixture exposes earned '+id);
      assert(H.Core.act(seed,lesson.visitAction).ok);
      const {context,page}=await open(390,844,seed);await page.clock.runFor(1000);
      if(await page.locator('[data-dismiss-find]:visible').count()){assert.equal(await guide(page).count(),0,'Verified find outranks the practice coach');await page.locator('[data-dismiss-find]').click();await page.clock.runFor(1000);}
      const trace=[];if(!await guide(page).count()){await shot(page,'missing-guide-'+id);console.log('NO GUIDE',id,await page.locator('dialog[open]').evaluateAll(ns=>ns.map(n=>({class:n.className,kind:n.dataset.kind,text:n.innerText.slice(0,500)}))));}
      for(let n=0;n<30;n++) {
        const saved=await state(page);if(saved.onboarding.practice.active!==id)break;
        const target=page.locator('[data-guide-target]');
        trace.push({step:await guide(page).getAttribute('data-step'),target:await target.count()?await target.getAttribute('data-wx-do'):null,sheet:await sheet(page).count()?await sheet(page).getAttribute('data-kind'):null});
        await shot(page,'optional-'+id+'-'+n);
        assert.equal(await target.count(),1,id+' has a real target '+JSON.stringify(trace));
        if(id==='reserves' && await page.locator('[data-wx-reserve]:visible').count())await page.locator('[data-wx-reserve]').fill('10');
        const prior=await state(page);await target.click();await page.clock.runFor(1000);
        if(id==='expansion' && trace[trace.length-1].step==='inspect')assert.equal((await state(page)).expedition.index,prior.expedition.index,'Inspect opens a review without performing expansion');
      }
      assert.equal((await state(page)).onboarding.practice.active,null,id+' completes '+JSON.stringify(trace));
      evidence.flows.push('Actual optional controls '+id);await context.close();
    }

    for(const committed of [false,true]) {
      const {context,page}=await open(390,844,H.Core.createState(1000));await page.clock.runFor(900);await page.locator('[data-upgrade="boots"] .wx-upgrade-info').click();await page.clock.runFor(100);
      await page.evaluate(committed=>{
        window.practiceSet=Storage.prototype.setItem;window.practiceGet=Storage.prototype.getItem;window.practiceWritten=false;
        Storage.prototype.setItem=function(key,value){if(!committed && key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');const result=window.practiceSet.call(this,key,value);if(key===WayfarersStorage.SAVE_KEY)window.practiceWritten=true;return result;};
        if(committed)Storage.prototype.getItem=function(key){if(key===WayfarersStorage.RESET_KEY && window.practiceWritten)throw new DOMException('Fence unavailable','SecurityError');return window.practiceGet.call(this,key);};
      },committed);
      await page.locator('[data-wx-practice]').click();await page.clock.runFor(100);
      const stored=await page.evaluate(()=>JSON.parse(window.practiceGet.call(localStorage,WayfarersStorage.SAVE_KEY)).state);
      assert.equal(stored.expedition.areas.greenway.ranks.boots,committed?1:0);
      assert.equal(stored.onboarding.practice.progress.greenway,committed?2:1);
      await shot(page,'practice-save-failure-'+committed);
      await page.locator('[data-guide-leave]').click();assert.equal(await sheet(page).getAttribute('data-kind'),'options');
      assert(await command(page,'settings').isEnabled(),'Settings/export remains reachable during pending save');
      await page.evaluate(()=>{Storage.prototype.setItem=window.practiceSet;Storage.prototype.getItem=window.practiceGet;});
      await page.locator('[data-wx-close]').click();await page.clock.runFor(1000);
      await page.locator('[data-wx-save-alert]').click();await page.clock.runFor(1000);
      if(!committed) {await page.locator('[data-wx-practice]').click();await page.clock.runFor(100);}
      const after=await state(page);assert.equal(after.expedition.areas.greenway.ranks.boots,1);assert.equal(after.onboarding.practice.supplies.filter(id=>id==='greenway:upgrade').length,1);
      await page.reload();await page.locator('.wx-game').waitFor();await page.clock.runFor(1000);assert.equal(await guide(page).getAttribute('data-step'),'operate');
      evidence.flows.push((committed?'Committed-fence':'Uncommitted')+' failure retains one rank/proof/supply with recovery and reload');await context.close();
    }
    {
      const seed=Fixtures.completeAreaGuides(H.mature());Fixtures.announceDiscoveries(seed);const {context,page}=await open(390,844,seed);
      await page.locator('[data-wx-batch]').click();const choice=page.locator('[data-wx-do]').filter({hasText:'×5'}).first();await choice.click();await page.clock.runFor(1000);
      assert.equal((await state(page)).onboarding.practice.active,'bulk','First ordinary batch interaction teaches its actual selected action');
      for(let n=0;n<8 && (await state(page)).onboarding.practice.active==='bulk';n++){await page.locator('[data-guide-target]').click();await page.clock.runFor(1000);}
      assert.equal((await state(page)).expedition.batch,5);assert(!(await state(page)).onboarding.practice.helpRewards.includes('bulk'),'Automatic first-use never collects optional help-open reward');
      if(await sheet(page).count())await page.locator('[data-wx-close]').click();
      await page.locator('[data-wx-options]').click();await command(page,'menu:'+JSON.stringify({kind:'lessons'})).click();await command(page,'menu:'+JSON.stringify({kind:'lesson-help',id:'bulk'})).click();await page.clock.runFor(100);assert((await state(page)).onboarding.practice.helpRewards.includes('bulk'),'Completed practice still earns its separate first Help-open reward');await page.locator('[data-wx-back]').click();await command(page,'menu:'+JSON.stringify({kind:'lesson-help',id:'reserves'})).click();await page.clock.runFor(200);
      assert((await state(page)).onboarding.practice.helpRewards.includes('reserves'),'Deliberately opening actual Help earns once');
      const before=(await state(page)).onboarding.practice;await page.locator('[data-wx-close]').click();await page.locator('[data-wx-options]').click();await command(page,'guide-replay:watchtower').click();await page.clock.runFor(1000);
      assert.equal(await sheet(page).getAttribute('data-kind'),'lesson-help');assert.deepEqual((await state(page)).onboarding.practice,before,'Read-only replay cannot spend or regrant any proof/reward/supply');
      await shot(page,'read-only-replay-390');await context.close();evidence.flows.push('First-use intended batch, optional Help-open once reward, mutation-free descriptive replay');
    }

    for(const id of ['quarry','watchtower','workshop','ruins','harbor']) {
      const seed=H.mature();Fixtures.announceDiscoveries(seed);assert(H.Core.act(seed,{type:'expedition-select',areaId:id}).ok);
      const {context,page}=await open(320,740,seed);await page.clock.runFor(1000);
      for(let n=0;n<15 && await guide(page).count();n++) {await geometry(page);await page.locator('[data-guide-target]').click();await page.clock.runFor(1000);}
      assert.equal((await state(page)).onboarding.practice.progress[id],3,id+' returned guild teaches actual existing upgrade and plan');
      await context.close();
    }
    evidence.flows.push('All six area lessons use real existing comparisons and plans without forcing an extra owned rank');
    {
      const {context,page}=await open(390,844,H.Core.createState(1000));
      await page.locator('[data-wx-options]').click();await command(page,'settings').click();await page.clock.runFor(2000);assert.equal(await guide(page).count(),0);
      const replacement=Fixtures.completeAreaGuides(H.Core.createState(2000));const imported=Storage.createStore({storage:null,now:()=>2000}).export(replacement);assert(imported.ok);
      await page.locator('#wg-save-text').fill(imported.text);await page.locator('[data-review-import]').click();await page.clock.runFor(1500);assert.equal(await guide(page).count(),0);
      await page.locator('[data-confirm-import]').click();await page.clock.runFor(2500);assert.equal(await guide(page).count(),0);assert.equal((await state(page)).createdAt,2000);assert.equal((await state(page)).onboarding.practice.progress.greenway,3);
      await shot(page,'import-invalidates-practice-390');await context.close();
      evidence.flows.push('Reviewed import outranks and invalidates stale practice epoch, preserving incoming completed lessons');
    }

    {
      const seed=Fixtures.completeAreaGuides(H.Core.createState(1000));H.fund(seed);assert(H.Core.act(seed,H.Core.getView(seed).expedition.cards[0].action).ok);
      const {context,page}=await open(320,740,seed);await page.clock.runFor(1000);
      assert.match(await sheet(page).innerText(),/Porters available/);
      await command(page,'onboarding-open:ready:area:greenway:porters').click();await page.clock.runFor(1000);assert.equal((await state(page)).onboarding.practice.active,'tiers');
      await page.locator('[data-wx-practice]').click();await page.clock.runFor(2400);
      assert.equal((await state(page)).onboarding.practice.progress.tiers,2);assert.equal(await sheet(page).count(),0,'Atomic tier lesson has no duplicate unlocked popup');assert(await page.locator('[data-wx-buy=porters]').isVisible());
      await shot(page,'tier-atomic-unlock-and-go-320');await context.close();evidence.flows.push('Ready tier first-use lesson uses atomic Unlock & go, reveals actual row without a popup chain');
    }
    {
      const seed=Fixtures.completeAreaGuides(H.Core.createState(1000));H.advance(seed,180);Fixtures.announceDiscoveries(seed);
      const {context,page}=await open(390,844,seed);await page.clock.runFor(100);
      const snapshot=async()=>page.evaluate(()=>({state:JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state,phase:document.querySelector('canvas[data-wx-canvas]').dataset.deliveryPhase,progress:document.querySelector('canvas[data-wx-canvas]').dataset.deliveryProgress,label:document.querySelector('[data-wx-world-label]').textContent}));
      let sawTravel=false,sawArrival=false;
      for(let t=0;t<450;t++){await page.clock.runFor(200);const x=await snapshot();if(x.phase==='travel')sawTravel=true;if(x.phase==='arrived'){sawArrival=true;assert.equal(Number(x.progress),1);assert.match(x.label,/Outpost.*coins/);await shot(page,'real-trail-arrival-390');break;}}
      assert(sawTravel && sawArrival,'Renderer follows both real trip and actual endpoint award');
      await page.reload();await page.locator('.wx-game').waitFor();await page.clock.runFor(100);assert(!(await page.locator('[data-wx-toast]').textContent()).includes('outpost reached'),'Reload does not replay old arrival celebration');
      await context.close();evidence.flows.push('Quiet-mode Trail traveler follows saved delivery progress and exact rewarded endpoint, reload baselines old sequence');
    }
    assert.deepEqual(evidence.errors,[]);
  } finally {
    fs.writeFileSync(path.join(output,'practice-browser.json'),JSON.stringify(evidence,null,2));
    await browser.close();await new Promise(resolve=>server.close(resolve));
  }
  console.log(JSON.stringify(evidence,null,2));
}
run().catch(error=>{console.error(error);process.exitCode=1;});
