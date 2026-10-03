'use strict';

// Uses a released E2 fixture by default; an exported retained guild can be
// supplied externally without committing personal save data.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const http = require('node:http');
const {chromium} = require('playwright');
const {bundle} = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-retained-practice-')));
const seedPath = process.env.WAYFARERS_RETAINED_SAVE || path.join(__dirname,'fixtures/wayfarers-v5-retained.json');
const source = JSON.parse(fs.readFileSync(seedPath,'utf8'));
const evidence = {fixture:seedPath,cases:[],errors:[],browser:'Browser plugin not available; repository Playwright workflow used.'};
const clone = value => JSON.parse(JSON.stringify(value));
async function saved(page) {return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});}
async function measure(page) {
  return page.evaluate(()=>{
    const box=node=>{if(!node)return null;const r=node.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
    const guide=document.querySelector('.wx-guide[open]'),target=document.querySelector('[data-guide-target]'),sheet=document.querySelector('.wx-sheet[open]');
    const r=target?.getBoundingClientRect(),hit=r&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
    return {step:guide?.dataset.step,guide:box(guide),coach:box(guide?.querySelector('.wx-guide-card')),sheet:box(sheet),target:box(target),targetText:target?.textContent,targetHit:!!target&&(hit===target||target.contains(hit)),popover:guide?.hasAttribute('popover'),popoverOpen:guide?.hasAttribute('popover')&&guide.matches(':popover-open'),sheetTransform:sheet&&getComputedStyle(sheet).transform,guideParent:guide?.parentElement.className,missing:guide?.dataset.missing,body:guide?.querySelector('#wx-guide-body').textContent};
  });
}
(async()=>{
  fs.mkdirSync(output,{recursive:true});
  const files=path.join(output,'bundle');bundle(files);
  const server=http.createServer((request,response)=>{
    const file=path.resolve(files,'.'+decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/'));
    if(!file.startsWith(files+path.sep)){response.writeHead(403).end();return;}
    fs.readFile(file,(error,bytes)=>{if(error){response.writeHead(404).end();return;}response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'})[path.extname(file)]||'application/json');response.end(bytes);});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  try {
    for(const ruleset of ['retained-E2','fresh-P3','rebuilt-P3']) for(const [width,height] of [[320,740],[390,844],[915,390]]) for(const fallback of [false,true]) {
      const seed=ruleset==='retained-E2' ? H.Core.normalizeState(clone(source.state || source)) : ruleset==='fresh-P3' ? H.Core.createState(1000) : H.mature();
      if(ruleset==='rebuilt-P3')assert(H.Core.act(seed,{type:'refit'}).ok);
      if(seed.expedition.selectedArea!=='greenway')assert(H.Core.act(seed,{type:'expedition-select',areaId:'greenway'}).ok);
      seed.lastUpdate=1000;
      const beforeRank=seed.expedition.areas.greenway.ranks.boots;
      const operation=H.Core.getView(seed).onboarding.guides.find(item=>item.id==='greenway').steps.find(step=>step.id==='operate').requiredAction;
      const record=Storage.createStore({storage:null,now:()=>1000}).export(seed);assert(record.ok,record.message);
      const context=await browser.newContext({viewport:{width,height},hasTouch:true,reducedMotion:'reduce'});
      await context.addInitScript(({key,value,fallback})=>{if(fallback)HTMLDialogElement.prototype.showPopover=undefined;if(!sessionStorage.getItem('seeded')){localStorage.setItem(key,value);sessionStorage.setItem('seeded','1');}},{key:Storage.SAVE_KEY,value:record.text,fallback});
      const page=await context.newPage();page.setDefaultTimeout(7000);page.on('pageerror',error=>evidence.errors.push(error.message));await page.clock.install({time:new Date(1000)});
      await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');await page.locator('.wx-game').waitFor();await page.clock.runFor(1000);
      if(await page.locator('.wx-sheet[data-kind="onboarding-notice"][open]').count()) {
        await page.locator('[data-wx-do="onboarding-later"]').tap();await page.clock.runFor(1000);
      }
      for(let wait=0;wait<12&&!await page.locator('.wx-guide[open]').count();wait++)await page.clock.runFor(250);
      const observations=[];
      evidence.cases.push({ruleset,width,height,fallback,observations});
      for(let index=0;index<12;index++) {
        const state=await saved(page);if(state.onboarding.practice.progress.greenway===3)break;
        await page.clock.runFor(300);await page.waitForTimeout(40);
        for(let wait=0;wait<12&&!await page.locator('.wx-guide[open]').count()&&(await saved(page)).onboarding.practice.progress.greenway!==3;wait++)await page.clock.runFor(250);
        if((await saved(page)).onboarding.practice.progress.greenway===3)break;
        const observation=await measure(page);observations.push(observation);
        await page.screenshot({path:path.join(output,`${ruleset}-${width}-${fallback?'fallback':'popover'}-${index}.png`)});
        assert(observation.target,'Actual required control exists at '+ruleset+' step '+index+': '+JSON.stringify(observation));
        if(!observation.step.startsWith('currency:'))assert(observation.targetHit,'Actual required control is pointer-hit-testable: '+JSON.stringify(observation));
        assert(observation.coach.x>=0&&observation.coach.y>=0&&observation.coach.right<=width+1&&observation.coach.bottom<=height+1,'Coach fits viewport: '+JSON.stringify(observation));
        if(observation.step==='operate' && await page.locator('[data-guide-target][data-wx-close]').count())assert.match(observation.body,/Close this panel/,'Intermediate Close has a truthful instruction');
        await page.locator(observation.step.startsWith('currency:') ? '[data-guide-next]' : '[data-guide-target]').tap();await page.clock.runFor(200);
      }
      const final=await saved(page);assert.equal(final.onboarding.practice.progress.greenway,3,'Real retained Trail practice completes');
      assert.equal(await page.locator('.wx-guide-surface').count(),0,'Temporary full-screen sheet framing is restored after completion');
      const area=final.expedition.areas.greenway;assert.equal(area.ranks.boots,Math.max(1,beforeRank),'Exactly one supplied first rank; existing investment is retained');
      if(operation?.type==='expedition-choice')assert(H.Core.getView(final).expedition.choices.flatMap(choice=>choice.options || [choice]).some(option=>option.selected&&option.id===operation.id),'Required working plan actually applied');
      const checkpoint=clone(final.onboarding.practice);await page.reload();await page.locator('.wx-game').waitFor();await page.clock.runFor(1000);assert.deepEqual((await saved(page)).onboarding.practice,checkpoint,'Lesson proof and supply persist without replay');
      await context.close();
    }
    assert.deepEqual(evidence.errors,[]);console.log(JSON.stringify({ok:true,cases:evidence.cases.length,output}));
  } finally {fs.writeFileSync(path.join(output,'retained-practice-browser.json'),JSON.stringify(evidence,null,2));await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
