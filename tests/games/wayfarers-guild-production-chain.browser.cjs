'use strict';

// Exact offline assets. Funded fixtures below isolate presentation and actions;
// the engine pacing suite separately measures natural opening acquisition.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const crypto = require('node:crypto');
const { chromium } = require('playwright');
const { bundle } = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const F = require('./helpers/wayfarers-onboarding.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const Skills = require('../../js/games/wayfarers-guild/area-skills');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-production-chain-')));
const report = { browser:'Browser plugin not available; repository Playwright workflow used.',viewports:[],flows:[],errors:[],sourceHashes:Object.fromEntries(['expedition-ui.js','expedition-scene.js','icons.js','core.js','area-skills.js'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(path.resolve(__dirname,'../../js/games/wayfarers-guild',file))).digest('hex')])) };
report.sourceHashes['wayfarers-guild.css']=crypto.createHash('sha256').update(fs.readFileSync(path.resolve(__dirname,'../../css/games/wayfarers-guild.css'))).digest('hex');
const clone = value => JSON.parse(JSON.stringify(value));
async function shot(page,name) { await page.clock.runFor(2700);await page.waitForTimeout(100);await page.screenshot({path:path.join(output,name+'.png')}); }
async function saved(page) { return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;}); }
function richlySkilled() {
  const state=H.mature();H.fund(state);H.Core.act(state,{type:'expedition-batch',count:1});
  // Explicit diagnostic output grants expose the full visual catalogue. Every
  // permission and rank still uses its canonical action; this is not pacing evidence.
  Object.keys(state.areaSkills.output).forEach(id=>state.areaSkills.output[id]=1e6);
  for(const definition of Skills.Content.SKILLS) {
    let item=H.Core.getView(state).areaSkills.items.find(row=>row.skillId===definition.id);
    if(!item.owned)assert(H.Core.act(state,item.unlockAction).ok,'Unlock diagnostic '+definition.id);
    for(let n=0;n<2;n++){item=H.Core.getView(state).areaSkills.items.find(row=>row.skillId===definition.id);const result=H.Core.act(state,item.action);assert(result.ok,result.message);}
  }
  F.completeAreaGuides(state);F.announceDiscoveries(state);assert(H.Core.validateState(state).valid);return state;
}
async function geometry(page) {
  const value = await page.evaluate(()=>{
    const rect=n=>{const r=n.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
    const rows=[...document.querySelectorAll('.wx-tray>.wx-upgrade')];
    return {width:innerWidth,height:innerHeight,horizontal:document.documentElement.scrollWidth-innerWidth,vertical:document.body.scrollHeight-innerHeight,tiles:rows.map(rect),controls:rows.flatMap(row=>[...row.querySelectorAll('button')].map(rect)),world:rect(document.querySelector('.wx-world')),dock:rect(document.querySelector('.wx-dock')),nav:rect(document.querySelector('.wx-nav'))};
  });
  assert(value.horizontal<=1 && value.vertical<=1,'Fixed viewport without overflow: '+JSON.stringify(value));
  assert.equal(value.tiles.length,3,'Three foundation slots stay present');
  assert(value.controls.every(control=>control.width>=47.5 && control.height>=47.5),'Foundation controls retain 48px targets: '+JSON.stringify(value));
  return value;
}
async function run() {
  fs.mkdirSync(output,{recursive:true});const files=path.join(output,'bundle');bundle(files);
  const server=http.createServer((req,res)=>{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\/assets\//,'/');const file=path.resolve(files,'.'+pathname);if(!file.startsWith(files+path.sep)){res.writeHead(403).end();return;}fs.readFile(file,(error,bytes)=>{if(error){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp'})[path.extname(file)]||'application/json');res.end(bytes);});});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const browser=await chromium.launch({headless:true});
  async function open(width,height,seed) {
    const state=clone(seed);state.lastUpdate=1000;const record=Storage.createStore({storage:null,now:()=>1000}).export(state);assert(record.ok,record.message);
    const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
    await context.addInitScript(({key,value})=>{if(!sessionStorage.seeded){localStorage.setItem(key,value);sessionStorage.seeded='1';}},{key:Storage.SAVE_KEY,value:record.text});
    const page=await context.newPage();page.setDefaultTimeout(10000);page.on('pageerror',error=>report.errors.push(error.message));await page.clock.install({time:new Date(1000)});await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.locator('.wx-game').waitFor();await page.clock.runFor(950);return {context,page};
  }
  try {
    const fresh=H.Core.createState(1000);F.completeAreaGuides(fresh);F.announceDiscoveries(fresh);
    for(const [width,height] of [[320,740],[390,844],[915,390]]) {
      const {context,page}=await open(width,height,fresh);
      const before=await geometry(page);assert.equal(await page.locator('.wx-tray [data-state="locked"]').count(),2);
      await shot(page,'first-foundations-'+width);
      await page.locator('[data-upgrade="porters"] .wx-upgrade-info').click();assert.match(await page.locator('.wx-sheet-content').innerText(),/Pathfinding|deliveries/);await shot(page,'foundation-requirements-'+width);await page.locator('[data-wx-close]').click();
      await page.locator('[data-wx-plans]').click();assert.equal(await page.locator('.wx-sheet[open]').getAttribute('data-kind'),'choice');assert.equal(await page.locator('.wx-operation-chain article').count(),3);await shot(page,'initial-operations-'+width);await page.locator('[data-wx-close]').click();
      const after=await geometry(page);assert.deepEqual(after.world,before.world);assert.deepEqual(after.dock,before.dock);report.viewports.push(after);await context.close();
    }
    report.flows.push('Three persistent foundations, canonical locked prerequisites and real Operations inspector at320/390/915 without geometry shifts');
    if(process.env.WAYFARERS_CHAIN_OPENING_ONLY!=='1') {
      const ready=clone(fresh);H.fund(ready,100000);
      while(ready.expedition.areas.greenway.ranks.boots<4)assert(H.Core.act(ready,H.Core.getView(ready).expedition.cards[0].action).ok);
      for(let n=0;n<120 && !H.Core.getView(ready).expedition.foundations.find(row=>row.trackId==='porters').unlockAction;n++)H.advance(ready,5);
      F.announceDiscoveries(ready);
      {
        const {context,page}=await open(320,740,ready);const before=await geometry(page);
        assert.equal(await page.locator('[data-upgrade="porters"]').getAttribute('data-state'),'ready');await shot(page,'porters-ready-320');
        await page.locator('[data-upgrade="porters"] .wx-upgrade-info').click();await page.locator('[data-wx-close]').click();
        assert.equal(await page.locator('[data-upgrade="porters"]').getAttribute('data-state'),'ready','Inspection never consumes readiness');
        await page.reload();await page.locator('.wx-game').waitFor();await page.clock.runFor(950);
        assert.equal(await page.locator('[data-upgrade="porters"]').getAttribute('data-state'),'ready','Ready survives reload');
        await page.locator('[data-upgrade="porters"] .wx-skill-unlock').click();await F.finishCurrentGuide(page);
        if(await page.locator('[data-wx-close]:visible').count())await page.locator('[data-wx-close]').click();
        assert.equal(await page.locator('[data-upgrade="porters"]').getAttribute('data-state'),'learned');assert((await saved(page)).upgradeTiers.claimed.includes('area:greenway:porters'));
        const after=await geometry(page);assert.deepEqual(after.world,before.world);assert.deepEqual(after.dock,before.dock);await shot(page,'porters-claimed-320');await context.close();
      }
      report.flows.push('Ready foundation remains through inspection/reload, then explicit taught claim persists with unchanged scene/dock geometry');
      const mature=H.mature();F.completeAreaGuides(mature);F.announceDiscoveries(mature);
      for(const [width,height] of [[320,740],[390,844],[915,390]]) {
        const seed=clone(mature);assert(H.Core.act(seed,{type:'expedition-select',areaId:'quarry'}).ok);const {context,page}=await open(width,height,seed);
        await geometry(page);assert(await page.locator('[data-wx-currency="coins"] strong').evaluate(node=>node.scrollWidth<=node.clientWidth+1),'The complete100T amount fits its rail chip');await shot(page,'quarry-production-'+width);await page.locator('[data-wx-nav="upgrades"]').click();assert.equal(await page.locator('[data-wx-do="upgrade-scope:current"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.wx-role-column').count(),3);await shot(page,'quarry-skills-'+width);
        await page.locator('[data-wx-do="upgrade-scope:global"]').click();assert.equal(await page.locator('[data-wx-do="upgrade-scope:global"]').getAttribute('aria-pressed'),'true');assert.equal(await page.locator('[data-wx-upgrade^="area:"]').count(),0);await shot(page,'global-upgrades-'+width);await context.close();
      }
      report.flows.push('Current-area functional columns and Global scope show canonical separate catalog ownership at320/390/915');
      const rich=richlySkilled();
      {
        const {context,page}=await open(390,844,rich);
        for(const areaId of Object.keys(rich.expedition.areas)) {
          await page.locator('[data-wx-objective]').click();await page.locator('[data-wx-area="'+areaId+'"]').click();await geometry(page);await shot(page,'area-'+areaId+'-390');
          await page.locator('[data-wx-nav="upgrades"]').click();assert.equal(await page.locator('[data-wx-upgrade^="area:'+areaId+':"]').count()+await page.locator('[data-wx-upgrade^="skill:"]').count(),15,'15 authored skills for '+areaId);await shot(page,'skills-'+areaId+'-390');await page.locator('[data-wx-nav="expedition"]').click();
        }
        await context.close();
      }
      report.flows.push('All six distinct dynamic scenes and 15 canonical skills per area, grouped by actual functional role');
      {
        const seed=clone(rich);assert(H.Core.act(seed,{type:'expedition-select',areaId:'greenway'}).ok);assert(H.Core.act(seed,{type:'expedition-batch',count:100}).ok);F.announceDiscoveries(seed);
        const {context,page}=await open(320,740,seed);await page.locator('[data-wx-nav="upgrades"]').click();await page.locator('[data-wx-upgrade="skill:express-routes"] .wx-upgrade-info').click();
        assert.match(await page.locator('.wx-purchase-footer').innerText(),/100/);assert(await page.locator('.wx-purchase-footer .wx-confirm').isDisabled());await shot(page,'skill-exact-quantity-320');
        await page.locator('[data-wx-do="skill-quantity:skill:express-routes"]').click();await F.finishCurrentGuide(page);assert.equal((await saved(page)).expedition.batch,5);assert.equal((await saved(page)).areaSkills.ranks['express-routes'],2,'Changing quantity does not buy');
        if(await page.locator('[data-wx-close]:visible').count())await page.locator('[data-wx-close]').click();await page.locator('[data-wx-nav="upgrades"]').click();await page.locator('[data-wx-upgrade="skill:express-routes"] .wx-upgrade-info').click();await page.locator('.wx-purchase-footer .wx-confirm').click();assert.equal((await saved(page)).areaSkills.ranks['express-routes'],7);await shot(page,'skill-bought-five-320');await context.close();
      }
      report.flows.push('Bounded skill preserves exact100 mode disabled, explicit Use×5 only changes quantity, following purchase buys exactly5 ranks');
      {
        const seed=clone(rich);assert(H.Core.act(seed,{type:'area-skill-config',id:'express-routes',value:'express'}).ok);assert(H.Core.act(seed,{type:'refit',confirm:true}).ok);F.announceDiscoveries(seed);
        for(const [width,height] of [[320,740],[915,390]]) {
          const {context,page}=await open(width,height,seed);await geometry(page);assert.equal(await page.locator('.wx-tray [data-state="learned"]').count(),3,'Retained foundations do not need reclaiming');assert.equal(await page.locator('.wx-tray .wx-skill-unlock').count(),0);
          await page.locator('[data-wx-nav="upgrades"]').click();await page.locator('[data-wx-upgrade="skill:express-routes"] .wx-upgrade-info').click();assert.match(await page.locator('.wx-sheet-content').innerText(),/Rebuild this skill/);assert(await page.locator('[data-wx-do="skill-config:express-routes:express"]').isDisabled());assert.equal((await saved(page)).areaSkills.configs['express-routes'],'express');await shot(page,'retained-skill-rebuild-'+width);await context.close();
        }
      }
      report.flows.push('Real Refit retains foundation permissions and saved technique choice, shows rank0 rebuilding and prevents dormant configuration use at320/915');
    }
    assert.deepEqual(report.errors,[]);console.log(JSON.stringify(report,null,2));
  } finally { fs.writeFileSync(path.join(output,'production-chain-browser.json'),JSON.stringify(report,null,2));await browser.close();await new Promise(resolve=>server.close(resolve)); }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
