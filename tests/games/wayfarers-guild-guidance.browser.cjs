'use strict';

// Render the exact offline package. Funded fixtures isolate interaction behavior,
// not acquisition or balance; fresh practice is exercised through actual controls.
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
const out=path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-guidance-')));
const report={viewports:[],flows:[],errors:[],sourceHashes:Object.fromEntries(['expedition-ui.js','onboarding-ui.js','onboarding.js','practice-lessons.js','area-motion.js'].map(file=>[file,require('node:crypto').createHash('sha256').update(fs.readFileSync(path.resolve(__dirname,'../../js/games/wayfarers-guild',file))).digest('hex')]))};
const guide=page=>page.locator('.wx-guide[open]');
const stored=page=>page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});
async function shot(page,name){await page.clock.runFor(60);await page.waitForTimeout(70);await page.screenshot({path:path.join(out,name+'.png')});}
async function nextCurrency(page){for(let i=0;i<12 && /^currency:/.test(await guide(page).getAttribute('data-step').catch(()=>''));i++){await page.locator('[data-guide-next]').click();await page.clock.runFor(100);}}
async function fit(page){const r=await page.evaluate(()=>{const b=n=>{const r=n.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};const g=document.querySelector('.wx-guide[open]');return {width:innerWidth,height:innerHeight,overflow:document.documentElement.scrollWidth-innerWidth,header:b(document.querySelector('.wx-header')),coach:g&&b(g.querySelector('.wx-guide-card')),buttons:g?[...g.querySelectorAll('button')].filter(n=>!n.hidden&&n.getClientRects().length).map(b):[],missing:g?.dataset.missing};});assert(r.overflow<=1,JSON.stringify(r));if(r.coach){assert(r.coach.x>=0&&r.coach.y>=0&&r.coach.right<=r.width+1&&r.coach.bottom<=r.height+1,JSON.stringify(r));assert.equal(r.missing,'false');assert(r.buttons.every(b=>b.width>=47.5&&b.height>=47.5));}return r;}
async function run(){
  fs.mkdirSync(out,{recursive:true});const files=path.join(out,'bundle');bundle(files);
  const server=http.createServer((req,res)=>{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\/assets\//,'/');const file=path.resolve(files,'.'+pathname);if(!file.startsWith(files+path.sep)){res.writeHead(403).end();return;}fs.readFile(file,(error,bytes)=>{if(error){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'})[path.extname(file)]||'application/json');res.end(bytes);});});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
  async function open(width,height,seed,reducedMotion='reduce'){
    seed=JSON.parse(JSON.stringify(seed));seed.lastUpdate=1000;const record=Storage.createStore({storage:null,now:()=>1000}).export(seed);assert(record.ok,record.message);
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion});await context.addInitScript(({key,value})=>{if(!sessionStorage.seeded){localStorage.setItem(key,value);sessionStorage.seeded='1';}},{key:Storage.SAVE_KEY,value:record.text});const page=await context.newPage();page.setDefaultTimeout(8000);page.on('pageerror',e=>report.errors.push(e.message));await page.clock.install({time:new Date(1000)});await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.locator('.wx-game').waitFor();await page.clock.runFor(950);return {context,page};
  }
  try {
    for(const [width,height] of [[320,740],[390,844],[915,390]]){
      const {context,page}=await open(width,height,H.Core.createState(1000));
      assert.equal(await guide(page).getAttribute('data-step'),'currency:coins');assert.equal(await page.locator('[data-wx-currency]').count(),1);assert.equal(await page.locator('[data-guide-leave]:visible').count(),0);assert.equal(await page.locator('[data-guide-settings]:visible').count(),1);assert.equal(await page.locator('.wx-header [data-wx-collection]').count(),0);
      const before=await stored(page),bounds=await fit(page);report.viewports.push(bounds);await shot(page,'coins-introduction-'+width);
      if(width===320){const style=await page.addStyleTag({content:'.wx-guide p{font-size:18.2px!important}.wx-guide h2{font-size:23.4px!important}.wx-currency strong{font-size:18.2px!important}.wx-currency small{font-size:11.7px!important;line-height:15.6px!important}'});await page.clock.runFor(100);await fit(page);await shot(page,'coins-font130-320');await page.setViewportSize({width:915,height:390});await page.waitForTimeout(100);await page.clock.runFor(150);await fit(page);await shot(page,'coins-font130-915');await style.evaluate(n=>n.remove());await page.setViewportSize({width,height});await page.waitForTimeout(100);await page.clock.runFor(150);}
      await page.keyboard.press('Escape');await page.clock.runFor(100);assert.equal(await guide(page).getAttribute('data-step'),'currency:coins');
      await page.locator('[data-guide-settings]').click();assert.equal(await page.locator('.wx-sheet[open]').getAttribute('data-kind'),'options');assert.equal(await page.locator('[data-wx-do="settings"]').count(),1);await page.locator('[data-wx-close]').click();await page.clock.runFor(800);assert.equal(await guide(page).getAttribute('data-step'),'currency:coins');
      await nextCurrency(page);const ack=await stored(page);assert.equal(ack.onboarding.practice.progress.greenway,0);assert.deepEqual(ack.onboarding.practice.supplies,before.onboarding.practice.supplies);assert.equal(ack.expedition.areas.greenway.ranks.boots,0);assert(ack.onboarding.practice.currencyRead.includes('coins'));
      await page.locator('[data-upgrade="boots"] .wx-upgrade-info').click();await page.clock.runFor(100);assert.equal(await guide(page).getAttribute('data-step'),'upgrade');assert(!/\bFree\b/.test(await page.locator('.wx-sheet-content,.wx-purchase-footer').allTextContents().then(a=>a.join(' '))));assert.match(await page.locator('.wx-purchase-summary').innerText(),/Cost:.*coins/);assert.match(await page.locator('.wx-purchase-summary').innerText(),/Guild supplies/);assert.match(await page.locator('[data-wx-practice]').innerText(),/Buy/);assert.match(await page.locator('[data-guide-quote]').innerText(),/×1.*6 coins[\s\S]*Guild supplies/);assert.match(await page.locator('[data-wx-practice]').getAttribute('aria-describedby'),/wx-guide-quote/);assert(await page.locator('[data-guide-quote]').evaluate(n=>{const r=n.getBoundingClientRect(),p=n.closest('.wx-guide-card').getBoundingClientRect();return r.top>=p.top&&r.bottom<=p.bottom&&r.bottom<=innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===n;}),'Canonical confirmation quote is visible and unobscured');await fit(page);await shot(page,'normal-funded-quote-'+width);
      const supplyBefore=await stored(page);await page.locator('[data-wx-practice]').click();await page.clock.runFor(100);const bought=await stored(page);assert.equal(bought.expedition.areas.greenway.ranks.boots,1);assert(H.Core.Numbers.cmp(bought.resources.coins,supplyBefore.resources.coins)>=0,'Practice does not debit the wallet while automatic production continues');assert(bought.onboarding.practice.supplies.includes('greenway:upgrade'));
      await F.finishCurrentGuide(page);const end=await stored(page);assert.equal(end.onboarding.practice.progress.greenway,3);assert.equal(await page.locator('.wx-header').boundingBox().then(r=>r.height),bounds.header.height);await context.close();
    }
    report.flows.push('Fresh currency explanation precedes quote, Next spends nothing, mandatory Escape cannot skip, Settings resumes, normal funded purchase preserves wallet at320/390/915');

    const mature=H.mature();F.completeAreaGuides(mature);F.announceDiscoveries(mature);assert(H.Core.act(mature,{type:'collection-unlock',kind:'cards'}).ok);assert(H.Core.act(mature,{type:'collection-unlock',kind:'equipment'}).ok);F.completeAreaGuides(mature);F.announceDiscoveries(mature);
    for(const [width,height] of [[320,740],[390,844],[915,390]]){
      const seed=JSON.parse(JSON.stringify(mature));for(const id of ['ore','herbs','provisions','knowledge','maps','notes','crests','starshards'])seed.resources[id]=H.Core.Numbers.from(0);seed.collection.ink=0;
      const {context,page}=await open(width,height,seed);assert.equal(await guide(page).count(),0);assert.equal(await page.locator('[data-wx-currency]').count(),10);assert.equal(await page.locator('[data-wx-currency-more]:visible').count(),1);await fit(page);await shot(page,'earned-rail-'+width);
      await page.locator('[data-wx-currency="coins"]').focus();await page.evaluate(()=>window.focusedCurrency=document.activeElement);await page.clock.runFor(1200);assert(await page.evaluate(()=>document.activeElement===window.focusedCurrency&&window.focusedCurrency.isConnected),'Currency balance ticks retain the same focused button');const selected=(await stored(page)).expedition.selectedArea;await page.locator('[data-wx-currency-more]').click();await page.clock.runFor(100);assert(await page.locator('[data-wx-currencies]').evaluate(n=>n.scrollLeft)>0);assert.equal((await stored(page)).expedition.selectedArea,selected);
      await page.locator('[data-wx-currency="ink"]').scrollIntoViewIfNeeded();await page.locator('[data-wx-currency="ink"]').click();assert.match(await page.locator('.wx-sheet-content').innerText(),/Craft extra copies/);await page.locator('[data-wx-close]').click();
      await page.locator('[data-wx-nav="guild"]').click();await page.locator('[data-wx-do="guild-page:cards"]').click();assert.equal(await page.locator('.wx-card-library').count(),1);await shot(page,'cards-via-guild-'+width);await page.locator('[data-wx-do="guild-page:equipment"]').click();assert.equal(await page.locator('.wx-gear-grid').count(),1);
      await page.locator('[data-wx-options]').click();await page.locator('[data-wx-do="guide-replay:'+selected+'"]').click();assert.equal(await page.locator('.wx-sheet[open]').getAttribute('data-kind'),'lesson-help');await page.keyboard.press('Escape');assert.equal(await page.locator('.wx-sheet[data-kind="lesson-help"][open]').count(),0);await context.close();
    }
    report.flows.push('Ten earned currencies persist at zero; fixed rail scroll and currency inspector; Cards/Equipment each one Guild tab away; read-only replay closes');
    {
      const seed=JSON.parse(JSON.stringify(mature));const lesson=H.Core.getView(seed).onboarding.guides.find(item=>item.id==='bulk');assert(H.Core.act(seed,lesson.visitAction).ok);const {context,page}=await open(390,844,seed);assert.equal(await page.locator('[data-guide-leave]:visible').innerText(),'Close');const before=await stored(page);await page.locator('[data-guide-leave]').click();const after=await stored(page);assert.equal(after.onboarding.practice.active,null);assert.equal(after.onboarding.practice.progress.bulk,before.onboarding.practice.progress.bulk);assert.deepEqual(after.onboarding.practice.supplies,before.onboarding.practice.supplies);await context.close();
    }
    report.flows.push('Optional Help-origin practice remains cancellable with Close without proof, supply or lesson completion');
    const unseen=JSON.parse(JSON.stringify(mature));unseen.onboarding.attention.seen=[];assert(H.Core.act(unseen,{type:'expedition-select',areaId:'greenway'}).ok);
    const {context,page}=await open(390,844,unseen);const boots=page.locator('[data-upgrade="boots"] .wx-upgrade-info');assert(await boots.getAttribute('data-wx-unseen'));await page.reload();await page.clock.runFor(950);assert(await boots.getAttribute('data-wx-unseen'));await boots.click();assert.equal(await boots.getAttribute('data-wx-unseen'),null);const afterInspect=await stored(page);assert(afterInspect.onboarding.attention.seen.includes('upgrade:area:greenway:boots'));await page.locator('[data-wx-close]').click();await page.locator('[data-wx-nav="guild"]').click();await page.locator('[data-wx-do="guild-page:cards"]').click();const tile=page.locator('.wx-card-tile[data-wx-unseen]').first();const id=await tile.getAttribute('data-wx-unseen');await tile.click();assert((await stored(page)).onboarding.attention.seen.includes(id));await shot(page,'inspected-card');await context.close();
    report.flows.push('Unseen upgrade/card markers survive reload and clear only their saved deliberate inspector receipts');
    for(const receipt of ['currency','inspection']) {
      const seed=receipt==='currency' ? H.Core.createState(1000) : unseen;
      const {context,page}=await open(390,844,seed);
      const before=await stored(page);
      await page.evaluate(()=>{window.originalSetItem=Storage.prototype.setItem;Storage.prototype.setItem=function(key,value){if(key.startsWith('wayfarers-guild-save'))throw new DOMException('Quota full','QuotaExceededError');return window.originalSetItem.call(this,key,value);};});
      if(receipt==='currency')await page.locator('[data-guide-next]').click();
      else await page.locator('[data-upgrade="boots"] .wx-upgrade-info').click();
      await page.clock.runFor(100);const failed=await stored(page);
      assert.deepEqual(failed.onboarding.practice.currencyRead,before.onboarding.practice.currencyRead);assert.deepEqual(failed.onboarding.attention.seen,before.onboarding.attention.seen);
      if(receipt==='currency'){assert.equal(await guide(page).getAttribute('data-step'),'currency:coins');await page.locator('[data-guide-settings]').click();assert.equal(await page.locator('[data-wx-do="settings"]').count(),1);await page.locator('[data-wx-close]').click();}
      await page.evaluate(()=>{Storage.prototype.setItem=window.originalSetItem;});
      if(await page.locator('[data-guide-next]:visible').count())await page.locator('[data-guide-next]').click();else {if(await page.locator('.wx-sheet[open]').count())await page.locator('[data-wx-close]').click();await page.locator('[data-wx-save-alert]').click();}await page.clock.runFor(100);
      if(receipt==='currency'){assert.equal(await guide(page).getAttribute('data-step'),'currency:coins');await page.locator('[data-guide-next]').click();await page.clock.runFor(100);assert((await stored(page)).onboarding.practice.currencyRead.includes('coins'));assert.equal((await stored(page)).onboarding.practice.supplies.length,0);}
      else {if(await page.locator('.wx-sheet[open]').count())await page.locator('[data-wx-close]').click();assert(await page.locator('[data-upgrade="boots"] .wx-upgrade-info').getAttribute('data-wx-unseen'));await page.locator('[data-upgrade="boots"] .wx-upgrade-info').click();assert((await stored(page)).onboarding.attention.seen.includes('upgrade:area:greenway:boots'));}
      await context.close();
    }
    report.flows.push('Failed currency Next/inspection receipts retain explanation and cue; Settings recovery stays accessible; Retry requires deliberate acknowledgment and grants no supply');
    {
      const seed=JSON.parse(JSON.stringify(mature));assert(H.Core.act(seed,{type:'expedition-select',areaId:'greenway'}).ok);
      const {context,page}=await open(390,844,seed,'no-preference');
      async function drag(dx){const p=await page.evaluate(()=>{const n=document.querySelector('[data-wx-canvas]'),r=n.getBoundingClientRect();for(let y=r.top+80;y<r.bottom-100;y+=30){const x=innerWidth/2;if(document.elementFromPoint(x,y)===n)return {x,y};}return null;});assert(p,'Canvas has an unobstructed gesture region');await page.mouse.move(p.x,p.y);await page.mouse.down();await page.mouse.move(p.x+dx,p.y,{steps:5});}
      await drag(-90);assert.equal(await page.locator('.wx-play').getAttribute('data-area-motion'),'drag');const beforeRelease=await page.locator('.wx-play').evaluate(n=>n.style.transform);assert.notEqual(beforeRelease,'');await page.mouse.up();assert.equal((await stored(page)).expedition.selectedArea,'quarry');assert.equal(await page.locator('.wx-play:not([aria-hidden])').getAttribute('data-area-motion'),'transition');assert.equal(await page.locator('.wx-play[aria-hidden="true"]').count(),1);await shot(page,'area-slide');await page.clock.runFor(400);assert.equal(await page.locator('.wx-play[aria-hidden="true"]').count(),0);
      await page.locator('[data-wx-prev]').click();assert.equal((await stored(page)).expedition.selectedArea,'greenway');await page.clock.runFor(400);await drag(90);await page.mouse.up();await page.clock.runFor(400);assert.equal((await stored(page)).expedition.selectedArea,'greenway');assert.equal(await page.locator('.wx-play').evaluate(n=>n.style.transform),'');
      await page.locator('[data-wx-next]').click();await page.locator('[data-wx-options]').click();assert.equal(await page.locator('.wx-play[aria-hidden="true"]').count(),0);assert.equal(await page.locator('.wx-play').evaluate(n=>n.style.transform),'');await page.locator('[data-wx-close]').click();
      const current=(await stored(page)).expedition.selectedArea;await page.locator('[data-wx-currency-more]').click();await page.clock.runFor(400);assert.equal((await stored(page)).expedition.selectedArea,current);
      await context.close();
    }
    report.flows.push('Real canvas drag preserves live offset into slide, arrows animate, first-area overswipe returns, Settings interrupts without ghost, currency scrolling never selects areas');
    assert.deepEqual(report.errors,[]);
  }finally{fs.writeFileSync(path.join(out,'guidance-browser.json'),JSON.stringify(report,null,2));await browser.close();await new Promise(resolve=>server.close(resolve));}
  console.log(JSON.stringify({output:out,flows:report.flows.length,viewports:report.viewports.length,errors:report.errors}));
}
run().catch(error=>{console.error(error);process.exitCode=1;});
