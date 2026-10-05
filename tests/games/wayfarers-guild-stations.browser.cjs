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
const text130='.wx-game .wx-inline-rank{font-size:13px!important;line-height:18.2px!important}.wx-game .wx-inline-buy{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-price>span{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-price>span[data-wide="true"]{font-size:13px!important}.wx-game .wx-inline-action>small{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-requirement[data-lifetime="true"]{font-size:11.7px!important;line-height:16.9px!important}.wx-game .wx-inline-lock>b{font-size:13px!important;line-height:16.9px!important}.wx-game .wx-inline-count{font-size:11.7px!important;line-height:15.6px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero strong{font-size:22.1px!important;line-height:28.6px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero span{font-size:14.3px!important;line-height:19.5px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-hero small{font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-help-selector>span:not(.wg-icon){font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-station-help-preview strong{font-size:16.9px!important;line-height:23.4px!important}.wx-sheet[data-kind="station-help"] .wx-help-cost{font-size:15.6px!important;line-height:22.1px!important}.wx-sheet[data-kind="station-help"] .wx-help-gate strong{font-size:14.3px!important;line-height:19.5px!important}.wx-sheet[data-kind="station-help"] .wx-help-gate-progress,.wx-sheet[data-kind="station-help"] .wx-help-missing{font-size:13px!important;line-height:18.2px!important}.wx-sheet[data-kind="station-help"] .wx-confirm{font-size:16.9px!important;line-height:23.4px!important}.wx-game .wx-inline-upgrade[data-ready="true"] .wx-inline-action>b{font-size:14.3px!important;line-height:19.5px!important}.wx-game .wx-inline-lock>b[data-stacked="true"]{line-height:14.3px!important}';
const statusText130='.wx-game .wx-station-boost{font-size:15.6px!important;line-height:22.1px!important}.wx-game .wx-station-reward{font-size:28.6px!important;line-height:36.4px!important}.wx-game .wx-station-cache{font-size:14.3px!important;line-height:19.5px!important}';
const wideInlineFont='.wx-game .wx-inline-buy,.wx-game .wx-inline-action>small,.wx-game .wx-inline-action>b,.wx-game .wx-inline-lock>b,.wx-game .wx-inline-price>span,.wx-game .wx-inline-rank{font-family:Verdana,"DejaVu Sans",system-ui,sans-serif!important}';
function progression(state) {return {ranks:state.stations.ranks,unlocked:state.stations.unlocked,built:state.stations.built,areaRanks:state.stations.areaRanks,areaUnlocked:state.stations.areaUnlocked};}
function fresh() {const state=H.Core.createState(1000);H.fund(state);F.completeAreaGuides(state);F.announceDiscoveries(state);return state;}
function mature() {const state=StationFixtures.mature();state.stations.encounter.remaining=0;F.announceDiscoveries(state);return state;}
function underfundedQuarry() {
  const state=StationFixtures.expansionReady();StationFixtures.act(state,{type:'expedition-next'});F.completeAreaGuides(state);StationFixtures.fund(state);
  const mine=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='mine');
  while(state.stations.ranks[mine.skillIds[0]]<60)StationFixtures.act(state,{type:'station-skill-buy',id:mine.skillIds[0],count:1});
  state.resources.ore=H.N.zero();state.stations.encounter.remaining=600;F.announceDiscoveries(state);assert(H.Core.validateState(state).valid);return state;
}
function widestStarterGate() {
  const state=StationFixtures.expansionReady();StationFixtures.act(state,{type:'expedition-next'});F.completeAreaGuides(state);StationFixtures.fund(state);
  const mine=Stations.Content.STATIONS.find(station=>station.areaId==='quarry'&&station.localId==='mine');
  while(state.stations.ranks[mine.skillIds[0]]<10)StationFixtures.act(state,{type:'station-skill-buy',id:mine.skillIds[0],count:1});
  state.stations.output.quarry=H.N.from(1050);state.stations.encounter.remaining=600;F.announceDiscoveries(state);assert(H.Core.validateState(state).valid);return state;
}
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
  async function clickCardRegion(cell,selector) {
    const buy=cell.locator('.wx-inline-buy'),card=await buy.boundingBox(),region=await cell.locator(selector).boundingBox();
    assert(region && region.width>0 && region.height>0,'The requested icon or price region is visible');
    await buy.click({position:{x:region.x-card.x+region.width/2,y:region.y-card.y+region.height/2}});
  }
  async function inlineGeometry(page) {
    const rows=await page.evaluate(()=>[...document.querySelectorAll('.wx-station-segment')].map((segment,index)=>{
      const box=node=>{const rect=node.getBoundingClientRect();return {x:rect.x,y:rect.y,width:rect.width,height:rect.height,bottom:rect.bottom};};
      const art=segment.querySelector('.wx-station-illustration'),strip=segment.querySelector('.wx-station-controls'),canvas=art.querySelector('canvas');
      return {id:segment.dataset.wxStation,index,segment:box(segment),art:box(art),strip:box(strip),canvas:box(canvas),scene:canvas.getAttribute('aria-label'),help:box(segment.querySelector('[data-wx-station-help]')),cells:[...strip.children].map(cell=>({id:cell.dataset.wxInlineSkill,box:box(cell),info:box(cell.querySelector('.wx-inline-info')),buy:box(cell.querySelector('.wx-inline-buy'))}))};
    }));
    for(const row of rows) {
      assert(Math.abs(row.art.height-row.art.width*(row.index===0 ? 320 : 208)/384)<1,'Station art preserves its source aspect ratio');
      assert(Math.abs(row.segment.height-row.art.height)<.1,'Scene-contained controls never add a permanent strip below the art');
      assert(row.strip.height<=80,'Starter controls remain compact within their scene');
      assert(row.strip.x>=row.art.x && row.strip.x+row.strip.width<=row.art.x+row.art.width+.1 && row.strip.y>=row.art.y && row.strip.bottom<=row.art.bottom+.1,'Starter controls belong inside their illustrated station');
      assert(row.help.width>=48 && row.help.height>=48,'Each station has a reachable question-mark help control');
      assert(row.scene?.endsWith(' working'),'Each station retains its illustrated scene');
      assert.equal(row.cells.length,3,'A canonical station has three starter controls');
      for(const cell of row.cells) {
        assert(cell.buy.width>=48 && cell.buy.height>=48,'Compact purchase controls remain full touch targets');
        assert(cell.box.x>=row.strip.x && cell.box.x+cell.box.width<=row.strip.x+row.strip.width+.1,'Starter controls stay inside the strip');
      }
    }
    return rows;
  }
  async function rememberInlineNodes(page) {
    await page.evaluate(()=>{window.__inlineNodes=[...document.querySelectorAll('.wx-station-segment')].map(segment=>({segment,canvas:segment.querySelector('canvas'),cells:[...segment.querySelectorAll('.wx-inline-upgrade')].map(cell=>({cell,info:cell.querySelector('.wx-inline-info'),buy:cell.querySelector('.wx-inline-buy')}))}));});
  }
  async function sameInlineNodes(page) {
    assert(await page.evaluate(()=>window.__inlineNodes.every(({segment,canvas,cells})=>segment.isConnected && segment.querySelector('canvas')===canvas && cells.every(({cell,buy})=>cell.isConnected && cell.querySelector('.wx-inline-buy')===buy))),'Purchases, unlocks and quantity updates retain the actual art and interactive button nodes');
  }
  async function readableInlinePrices(page) {
    const priceMetadata=await page.locator('.wx-inline-price>span').evaluateAll(nodes=>nodes.map(node=>({text:[...node.childNodes].filter(child=>child.nodeType===Node.TEXT_NODE).map(child=>child.textContent).join(''),wide:node.dataset.wide==='true'})));
    for(const price of priceMetadata)assert.equal(price.wide,price.text.length>3,'Both ordinary and supplied tutorial prices retain the layout treatment for their actual displayed length');
    const violations=await page.evaluate(()=>[...document.querySelectorAll('.wx-inline-buy')].flatMap(button=>{
      const box=button.getBoundingClientRect();
      const marker=button.querySelector('.wx-inline-count'),overlaps=[];
      if(marker?.getClientRects().length){const count=marker.getBoundingClientRect();for(const line of button.querySelectorAll('.wx-inline-price>span'))for(const child of line.childNodes){let rect;if(child.nodeType===Node.TEXT_NODE){const range=document.createRange();range.selectNodeContents(child);rect=range.getBoundingClientRect();}else rect=child.getBoundingClientRect();if(rect.width && Math.min(count.right,rect.right)-Math.max(count.x,rect.x)>1 && Math.min(count.bottom,rect.bottom)-Math.max(count.y,rect.y)>1)overlaps.push({skill:button.closest('.wx-inline-upgrade').dataset.wxInlineSkill,text:marker.textContent,overlap:child.textContent || 'currency icon'});}}
      if(overlaps.length)return overlaps;
      return [...button.querySelectorAll('.wx-inline-price>span,.wx-inline-action>small,.wx-inline-action>b,.wx-inline-lock>b,.wx-inline-count,.wx-inline-rank')].flatMap(node=>{if(!node.getClientRects().length)return [];const rect=node.getBoundingClientRect();return node.scrollWidth>node.clientWidth+1 || node.scrollHeight>node.clientHeight+1 || rect.x<box.x-1 || rect.right>box.right+1 || rect.y<box.y-1 || rect.bottom>box.bottom+1 ? [{skill:button.closest('.wx-inline-upgrade').dataset.wxInlineSkill,text:node.textContent,width:rect.width,scrollWidth:node.scrollWidth,clientWidth:node.clientWidth,height:rect.height,scrollHeight:node.scrollHeight,clientHeight:node.clientHeight}] : [];});
    }));
    assert.deepEqual(violations,[],'Every currency price, rank, batch marker and missing-gate caption remains fully visible inside its button');
    const badges=await page.locator('.wx-inline-resource-badge').evaluateAll(nodes=>nodes.map(node=>{const box=item=>{const rect=item.getBoundingClientRect();return {x:rect.x,y:rect.y,right:rect.right,bottom:rect.bottom};};const info=node.closest('.wx-inline-info'),buy=info.closest('.wx-inline-buy'),dot=getComputedStyle(buy,'::after'),button=buy.getBoundingClientRect();return {badge:box(node),info:box(info),rank:box(info.querySelector('.wx-inline-rank')),dot:buy.hasAttribute('data-wx-unseen') ? {x:button.x+parseFloat(dot.left),y:button.y+parseFloat(dot.top),right:button.x+parseFloat(dot.left)+parseFloat(dot.width),bottom:button.y+parseFloat(dot.top)+parseFloat(dot.height)} : null};}));
    const intersects=(a,b)=>Math.min(a.right,b.right)-Math.max(a.x,b.x)>1 && Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y)>1;
    for(const {badge,info,rank,dot} of badges){assert(badge.x>=info.x && badge.right<=info.right && badge.y>=info.y && badge.bottom<=info.bottom,'The lifetime currency badge stays in its existing illustration row');assert(!intersects(badge,rank) && (!dot || !intersects(badge,dot)),'The resource badge stays distinct from rank and unseen indicators');}
  }
  async function helpBounds(page) {
    const bounds=await page.evaluate(()=>{const sheet=document.querySelector('.wx-sheet[open][data-kind="station-help"]'),nav=document.querySelector('.wx-nav'),rect=sheet.getBoundingClientRect(),dock=nav.getBoundingClientRect();return {x:rect.x,right:rect.right,y:rect.y,bottom:rect.bottom,navTop:dock.top,width:innerWidth,height:innerHeight,scrollWidth:sheet.scrollWidth,clientWidth:sheet.clientWidth};});
    assert(bounds.x>=-1 && bounds.right<=bounds.width+1 && bounds.y>=-1 && bounds.bottom<=bounds.navTop+1,'Temporary station help stays within the viewport above navigation');
    assert(bounds.scrollWidth<=bounds.clientWidth+1,'Station help has no concealed horizontal overflow');
    const clipped=await page.locator('.wx-sheet[open][data-kind="station-help"] .wx-help-have,.wx-sheet[open][data-kind="station-help"] .wx-help-need').evaluateAll(nodes=>nodes.filter(node=>node.scrollWidth>node.clientWidth+1).map(node=>node.textContent));
    assert.deepEqual(clipped,[],'Have and Need amounts remain fully readable');
    return bounds;
  }
  async function clearWorldStatus(page,label) {
    const status=await page.evaluate(()=>{
      const world=document.querySelector('.wx-station-world').getBoundingClientRect();
      const box=node=>{const rect=node.getBoundingClientRect();return {x:rect.x,y:rect.y,right:rect.right,bottom:rect.bottom,width:rect.width,height:rect.height};};
      const protectedNodes=[...document.querySelectorAll('.wx-station-controls,.wx-station-heading,.wx-station-rate')].filter(node=>node.getClientRects().length).map(node=>({kind:node.className,...box(node)})).filter(rect=>rect.bottom>world.top && rect.y<world.bottom);
      const overlays=['cache','boost','reward'].map(kind=>{const node=document.querySelector('[data-wx-station-'+kind+']'),rect=box(node),style=getComputedStyle(node),visible=!node.hidden && style.visibility!=='hidden' && rect.width>0 && rect.height>0;const baseline=world.top+parseFloat(node.style.top || 0);return {kind,text:node.textContent,hidden:node.hidden,visible,visibility:style.visibility,...rect,scrollWidth:node.scrollWidth,clientWidth:node.clientWidth,envelope:kind==='reward' && visible ? {x:rect.x-rect.width*.04,right:rect.right+rect.width*.04,y:baseline-38,bottom:baseline+node.offsetHeight+10} : rect};});
      const blockers=protectedNodes.map(rect=>({top:Math.max(world.top,rect.y-6),bottom:Math.min(world.bottom,rect.bottom+6)})).sort((a,b)=>a.top-b.top);let cursor=world.top,maxFree=0;for(const rect of blockers){maxFree=Math.max(maxFree,rect.top-cursor);cursor=Math.max(cursor,rect.bottom);}maxFree=Math.max(maxFree,world.bottom-cursor);
      return {world:{x:world.x,y:world.y,right:world.right,bottom:world.bottom},protectedNodes,overlays,maxFree};
    });
    report.status ||= [];report.status.push({label,...status});
    const intersects=(a,b)=>Math.min(a.right,b.right)-Math.max(a.x,b.x)>1 && Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y)>1;
    for(const overlay of status.overlays.filter(item=>item.visible)) {
      const rect=overlay.envelope;
      assert(rect.x>=status.world.x-1 && rect.right<=status.world.right+1 && rect.y>=status.world.y-1 && rect.bottom<=status.world.bottom+1,label+' keeps '+overlay.kind+' and its motion inside the world');
      assert(overlay.scrollWidth<=overlay.clientWidth+1,label+' displays the full '+overlay.kind+' text');
      assert.deepEqual(status.protectedNodes.filter(target=>intersects(rect,target)),[],label+' keeps '+overlay.kind+' clear of real upgrade triplets, station headings and rate labels');
    }
    const visible=status.overlays.filter(item=>item.visible);
    for(let index=0;index<visible.length;index++)for(const next of visible.slice(index+1))assert(!intersects(visible[index].envelope,next.envelope),label+' reserves separate artwork for '+visible[index].kind+' and '+next.kind);
    return status;
  }
  async function helpModel(page,stationId,skillId) {
    // Read one actual foreground-rendered frame while the mocked clock is
    // paused; a later freeze/save otherwise advances lifetime gate progress.
    await page.clock.pauseAt(new Date(await page.evaluate(()=>Date.now())+1000));
    try {
      await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
      const state=await saved(page),station=H.Core.getView(state).stations.currentArea.stations.find(row=>row.id===stationId),item=station.skills.find(row=>row.id===skillId),sheet=page.locator('.wx-sheet[open][data-kind="station-help"]');
      assert(item,'Help points to an actual engine-owned station upgrade');
      assert.equal(await sheet.locator('[data-wx-help-skill="'+skillId+'"]').getAttribute('aria-pressed'),'true');
      assert.equal(await sheet.locator('.wx-station-help-hero strong').innerText(),item.name || item.label || item.id);
      assert.equal(await sheet.locator('.wx-station-help-hero span').innerText(),item.effectText || item.description || '');
      if(item.comparison)assert.equal(await sheet.locator('.wx-station-help-preview strong').innerText(),item.comparison.text || item.comparison,'Effect preview uses the actual engine comparison');
      assert.deepEqual(await sheet.locator('[data-wx-help-cost]').evaluateAll(nodes=>nodes.map(node=>node.dataset.wxHelpCost)),item.cost.map(cost=>cost.resource),'Help lists each currency in the actual quote');
      for(const cost of item.cost) {
        const node=sheet.locator('[data-wx-help-cost="'+cost.resource+'"]'),have=state.resources[cost.resource],met=H.N.cmp(have,cost.amount)>=0;
        assert.equal(await node.locator('.wx-help-have').innerText(),H.Core.format(have),'Have reflects the canonical wallet');
        assert.equal(await node.locator('.wx-help-need').innerText(),H.Core.format(cost.amount),'Need reflects the canonical selected quantity cost');
        assert.equal(await node.getAttribute('data-met'),String(met));
        if(!met)assert.equal(await node.locator('.wx-help-missing').innerText(),'Need '+H.Core.format(H.N.sub(cost.amount,have))+' more '+cost.resource,'Currency shortfall states the exact missing amount');
      }
      const locked=item.state==='locked' || item.state==='ready';
      assert.equal(await sheet.locator('[data-wx-help-gate]').count(),locked ? item.requirements.length : 0);
      if(locked)for(const [index,requirement] of item.requirements.entries()) {
        const node=sheet.locator('[data-wx-help-gate="'+index+'"]'),target=requirement.required ?? requirement.target;
        assert.equal(await node.locator('strong').innerText(),requirement.label);
        assert.equal(await node.getAttribute('data-met'),String(requirement.met));
        if(target!=null && requirement.current!=null) {
          assert.equal(await node.locator('.wx-help-gate-progress').innerText(),'Have '+H.Core.format(requirement.current)+' / Need '+H.Core.format(target));
          if(!requirement.met)assert.equal(await node.locator('.wx-help-missing').innerText(),'Need '+H.Core.format(H.N.sub(target,requirement.current))+' more','The missing unlock gate is exact and engine owned');
        }
      }
      await helpBounds(page);
      return item;
    } finally {await page.clock.resume();}
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
        const question=await target.getAttribute('data-wx-station-help'),command=await target.getAttribute('data-wx-do');
        trace.push({step:await coach.getAttribute('data-step'),target:command || question || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close')});
        const ranksBefore=(await saved()).stations.ranks;
        await shot(page,'first-guide-'+width+'-'+n);await target.click();await page.clock.runFor(50);
        if(question)assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(),1,'One intended question-mark press opens the first lesson help immediately');
        if(command?.startsWith('inline-buy:'))assert.equal((await saved()).stations.ranks[command.slice('inline-buy:'.length)],(ranksBefore[command.slice('inline-buy:'.length)] || 0)+1,'One intended highlighted upgrade press applies exactly its taught rank');
        await page.clock.runFor(200);
      }
      const result=await saved();assert.equal(result.onboarding.practice.progress.greenway,3,JSON.stringify(trace));assert.equal(result.stations.ranks['station:greenway:path:pathfinding'],1);
      report.flows.push(width+'px first visit: '+JSON.stringify(trace));await context.close();
    }
    const opening=fresh();
    for(const [width,height,largeText,wideFont] of [[320,740],[390,844],[430,932],[915,390],[640,256],[320,740,true],[640,256,true],[320,740,true,true]]) {
      const {page,context}=await open(width,height,opening);
      if(largeText)await page.addStyleTag({content:text130});
      if(wideFont)await page.addStyleTag({content:wideInlineFont});const caseLabel=width+(largeText?'-text130':'')+(wideFont?'-wide-font':'');
      assert.equal(await page.locator('.wx-dock:visible').count(),0,'No permanent upgrade dock');assert.equal(await page.locator('.wx-inline-upgrade:visible').count(),3,'Three starters are part of the main world');assert.equal(await page.locator('.wx-inline-upgrade[data-state="locked"]').count(),2);
      assert.equal(await page.locator('.wx-inline-quantity:visible').count(),0,'Bulk selection waits for its earned unlock');
      await page.locator('.wx-inline-buy').first().scrollIntoViewIfNeeded();const before=await geometry(page);assert(before.horizontal<=1 && before.vertical<=1);const artBefore=await inlineGeometry(page);await rememberInlineNodes(page);
      const first=page.locator('.wx-inline-upgrade').first(),skill=await first.getAttribute('data-wx-inline-skill'),rankBefore=(await saved(page)).stations.ranks[skill];
      await clickCardRegion(first,width===430 ? '.wx-inline-rank' : width===390 || width===915 ? '.wx-inline-price' : '.wx-inline-info');assert.equal((await saved(page)).stations.ranks[skill],rankBefore+1,'Tapping the icon, rank or price buys exactly one actual rank');assert.deepEqual(await geometry(page),before,'Inline purchase never moves the camera');assert.deepEqual(await inlineGeometry(page),artBefore,'Inline purchase preserves art and strip geometry');await sameInlineNodes(page);assert.equal(await first.locator('[data-wx-unseen]').count(),0,'Interacted purchase loses its new indicator');
      const inspectionBefore=progression(await saved(page));
      await page.locator('[data-wx-station-help="greenway:path"]').click();assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(),1,'Station question mark opens temporary help');await page.locator('[data-wx-help-skill="'+skill+'"]').click();await helpModel(page,'greenway:path',skill);
      const selectors=await page.locator('[data-wx-help-skill]').evaluateAll(nodes=>nodes.map(node=>node.dataset.wxHelpSkill));assert.equal(selectors.length,3,'Station help selects each of the three actual starters');
      for(const id of selectors){await page.locator('[data-wx-help-skill="'+id+'"]').click();await page.locator('[data-wx-help-skill="'+id+'"]').click();await helpModel(page,'greenway:path',id);assert.deepEqual(progression(await saved(page)),inspectionBefore,'Repeated help selection never buys or unlocks an upgrade');}
      await shot(page,'opening-help-'+caseLabel);assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.equal(await page.locator('[data-wx-station-drawer]:visible').count(),0,'Back from station help returns to the world');assert.deepEqual(await geometry(page),before);assert.deepEqual(progression(await saved(page)),inspectionBefore);
      await page.locator('[data-wx-station-help="greenway:path"]').click();await helpModel(page,'greenway:path',skill);assert.deepEqual(progression(await saved(page)),inspectionBefore,'The station question mark only opens help');
      if(largeText && width===320){const rank=(await saved(page)).stations.ranks[skill];await page.locator('[data-wx-do="station-help-buy:'+skill+'"]').click();assert.equal((await saved(page)).stations.ranks[skill],rank+1,'Only the explicit help purchase buys its canonical rank');await sameInlineNodes(page);assert.deepEqual(await geometry(page),before);}
      await page.locator('[data-wx-close]').click();
      await page.locator('.wx-inline-upgrade[data-state="locked"]').first().locator('.wx-inline-buy').click();assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(),1,'Locked inline control exposes requirements');await page.locator('[data-wx-close]').click();
      await readableInlinePrices(page);
      await shot(page,'opening-world-'+caseLabel);
      await page.locator('[data-wx-nav="upgrades"]').click();assert.equal(await page.locator('.wx-station-row').count(),0,'Drawer contains no duplicate starter purchases');assert.equal(await page.locator('[data-wx-do="station-core:greenway:path"]').count(),1,'Drawer points back to the station controls');
      assert.deepEqual(await geometry(page),before,'Drawer preserves HUD, dock, world and scroll');await shot(page,'opening-drawer-'+caseLabel);
      await page.locator('[data-wx-do="station-core:greenway:path"]').click();assert.equal(await page.locator('[data-wx-station-drawer]:visible').count(),0);await sameInlineNodes(page);await page.locator('[data-wx-wallet]').click();assert.equal(await page.locator('.wx-wallet-list .wx-menu').count(),H.Core.getView(opening).onboarding.currencies.length);await page.locator('[data-wx-close]').click();
      report.viewports.push({width,height,largeText:!!largeText,wideFont:!!wideFont,...before});await context.close();
    }
    {
      const {page,context}=await open(320,740,widestStarterGate());await dismissNotices(page);await page.addStyleTag({content:text130+wideInlineFont});
      const cell=page.locator('[data-wx-inline-skill="station:quarry:mine:rich-veins"]'),camera=await geometry(page),progressBefore=progression(await saved(page));
      assert.equal(await cell.getAttribute('data-state'),'locked');assert.equal(await cell.locator('.wx-inline-resource-badge [data-icon]').getAttribute('data-icon'),'ore');assert.equal(await cell.locator('.wx-inline-lock>b').innerText(),'1.1K\n/1.2K','Both figures of the largest actual starter lifetime gate remain legible');
      await readableInlinePrices(page);await shot(page,'largest-starter-gate-320-text130-wide-font');await page.locator('[data-wx-station-help="quarry:mine"]').click();await page.locator('[data-wx-help-skill="station:quarry:mine:rich-veins"]').click();await helpModel(page,'quarry:mine','station:quarry:mine:rich-veins');await shot(page,'largest-starter-gate-help-320-text130-wide-font');assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.deepEqual(await geometry(page),camera);assert.deepEqual(progression(await saved(page)),progressBefore,'Wider-font gate inspection never unlocks or buys');await context.close();
    }
    report.flows.push('320/390/430/915/short landscape and text130: three scene-contained starters, two initial locks, real main-screen purchase, no permanent strip, stable art and retained button nodes; station question mark and three selectors show canonical effect, Have/Need and missing gates without purchases; temporary help uses native Back; drawer has no duplicate starters and returns to core controls; wallet exposes currencies');
    for(const gesture of ['drag','cancel','multitouch']) {
      const {page,context}=await open(320,740,opening),buy=page.locator('.wx-inline-buy').first(),id=await buy.locator('..').getAttribute('data-wx-inline-skill');
      const rank=(await saved(page)).stations.ranks[id],box=await buy.boundingBox(),pointer={pointerId:71,pointerType:'touch',button:0,buttons:1,clientX:box.x+box.width/2,clientY:box.y+box.height/2};
      await page.clock.pauseAt(new Date(await page.evaluate(()=>Date.now())+1000));
      try {
        await buy.dispatchEvent('pointerdown',pointer);
        if(gesture==='multitouch'){await buy.dispatchEvent('pointerdown',{...pointer,pointerId:72});await buy.dispatchEvent('pointerup',{...pointer,pointerId:72,buttons:0});}
        if(gesture==='drag')await buy.dispatchEvent('pointermove',{...pointer,clientY:pointer.clientY+25});
        await buy.dispatchEvent(gesture==='cancel' ? 'pointercancel' : 'pointerup',{...pointer,buttons:0,clientY:pointer.clientY+(gesture==='drag' ? 25 : 0)});
        await page.mouse.click(pointer.clientX,pointer.clientY);
        assert.equal((await saved(page)).stations.ranks[id],rank,'A '+gesture+' gesture cannot synthesize a scene purchase');
      } finally {await page.clock.resume();}
      await page.clock.runFor(1100);await buy.click();assert.equal((await saved(page)).stations.ranks[id],rank+1,'An ordinary intentional press works after the gesture guard expires');
      await context.close();
    }
    report.flows.push('Actual control pointer drag, cancellation and multitouch suppress accidental purchases; a later ordinary press purchases once');
    for(const [width,height,largeText] of [[320,740],[390,844],[430,932],[640,256,true]]) {
      const seed=underfundedQuarry();
      const {page,context}=await open(width,height,seed);await dismissNotices(page);if(largeText)await page.addStyleTag({content:text130});
      const station=H.Core.getView(await saved(page)).stations.currentArea.stations[0],id=station.skills[0].id;
      await page.locator('[data-wx-station-help="'+station.id+'"]').scrollIntoViewIfNeeded();
      const before=await geometry(page),progressBefore=progression(await saved(page));await rememberInlineNodes(page);
      await page.locator('[data-wx-station-help="'+station.id+'"]').click();const item=await helpModel(page,station.id,id);assert(item.cost.length>=2,'Mature help retains its actual two-currency quote');assert(await page.locator('[data-wx-help-cost="ore"] .wx-help-missing').count(),'Missing ore is explicitly named');
      assert.equal(await page.locator('[data-wx-do="station-help-buy:'+id+'"]').isDisabled(),true,'An underfunded help purchase cannot execute');
      assert.deepEqual(progression(await saved(page)),progressBefore);assert.deepEqual(await geometry(page),before,'Opening help keeps the world camera fixed');await sameInlineNodes(page);
      await shot(page,'missing-ore-help-'+width+(largeText ? '-text130' : ''));
      if(largeText){for(const target of await page.locator('[data-wx-help-skill]').all()){await target.click();await helpModel(page,station.id,await target.getAttribute('data-wx-help-skill'));assert.deepEqual(await geometry(page),before,'Internal help selection never scrolls the underlying short landscape world');}await page.locator('[data-wx-help-skill="'+id+'"]').click();await helpModel(page,station.id,id);const shortage=page.locator('[data-wx-help-cost="ore"]');await shortage.scrollIntoViewIfNeeded();assert(await shortage.isVisible(),'The actual two-currency shortage can be reached in short landscape');assert.deepEqual(await geometry(page),before,'Scrolling Have/Need inside help never moves the short landscape camera');const confirm=page.locator('[data-wx-do="station-help-buy:'+id+'"]'),box=await confirm.boundingBox(),nav=await page.locator('.wx-nav').boundingBox();assert(box.height>=48 && box.y+box.height<=nav.y+1,'The fixed help action remains reachable while the content scrolls');await shot(page,'short-landscape-shortage-scrolled-text130');}
      if(largeText){await page.setViewportSize({width:320,height:740});await page.clock.runFor(1000);await helpModel(page,station.id,id);assert((await geometry(page)).horizontal<=1 && (await geometry(page)).vertical<=1);await shot(page,'help-rotated-text130-320');await readableInlinePrices(page);}
      const help=await helpBounds(page);assert(help.y>0,'The backdrop exposes a safe dismissal region');await page.mouse.click(4,help.y-1);assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(),0,'Backdrop dismissal closes temporary help without clicking the scene');assert.deepEqual(progression(await saved(page)),progressBefore,'Dismissing an underfunded quote never buys or unlocks');await sameInlineNodes(page);if(!largeText)assert.deepEqual(await geometry(page),before);
      if(width===320){await page.reload();await page.clock.runFor(2000);await ready(page);await dismissNotices(page);assert.deepEqual(progression(await saved(page)),progressBefore,'Refreshing after help preserves progression');assert.equal(await page.locator('.wx-sheet[open][data-kind="station-help"]').count(),0,'A temporary help sheet does not reopen after refresh');}
      await context.close();
    }
    report.flows.push('320/390/430 and short landscape text130: canonical mature two-currency Have/Need, exact ore shortfall, disabled purchase, no help selection/dismissal transactions, native scene identity and camera retained; rotation keeps temporary help above navigation');
    for(const [largeText,wideFont] of [[false],[true],[true,true]]) {
      const seed=StationFixtures.buildReady(),firstReady=H.Core.getView(seed).stations.currentStation.skills.find(row=>row.state==='ready');StationFixtures.act(seed,{type:'onboarding-visit',id:'tiers',intendedAction:firstReady.unlockAction});for(let n=0;n<40 && H.Core.getView(seed).onboarding.active;n++){const active=H.Core.getView(seed).onboarding.active;StationFixtures.act(seed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(seed).onboarding.active,null);F.announceDiscoveries(seed);
      const {page,context}=await open(320,740,seed);await dismissNotices(page);if(largeText)await page.addStyleTag({content:text130});if(wideFont)await page.addStyleTag({content:wideInlineFont});
      const before=await geometry(page),artBefore=await inlineGeometry(page);await readableInlinePrices(page);await rememberInlineNodes(page);
      const id=await page.locator('.wx-inline-upgrade[data-ready="true"]').first().getAttribute('data-wx-inline-skill');assert(id,'An actually earned core upgrade becomes ready in its existing cell');const unlock=page.locator('[data-wx-inline-skill="'+id+'"]');
      assert.match(await unlock.locator('.wx-inline-buy').getAttribute('data-wx-do'),/^inline-unlock:/);
      await shot(page,'inline-ready-320'+(largeText?'-text130':'')+(wideFont?'-wide-font':''));
      if(largeText){const progressBefore=progression(await saved(page)),owner=id.split(':').slice(1,3).join(':');await page.locator('[data-wx-station-help="'+owner+'"]').click();await page.locator('[data-wx-help-skill="'+id+'"]').click();await helpModel(page,owner,id);assert.deepEqual(progression(await saved(page)),progressBefore,'Inspecting an earned unlock never claims it');await shot(page,'ready-unlock-help-320-text130'+(wideFont?'-wide-font':''));const confirm=page.locator('[data-wx-do="station-help-buy:'+id+'"]');assert.equal(await confirm.isDisabled(),false);await confirm.click();assert(await page.evaluate(()=>WayfarersUI.handleBack()));}
      else await unlock.locator('.wx-inline-buy').click();
      assert((await saved(page)).stations.unlocked.includes(id));assert.equal(await unlock.getAttribute('data-state'),'learned');assert.equal(await unlock.getAttribute('data-ready'),'false');
      await unlock.locator('.wx-inline-buy').click();assert.equal((await saved(page)).stations.ranks[id],1,'Unlocked core control buys the real first rank');assert.deepEqual(await geometry(page),before);assert.deepEqual(await inlineGeometry(page),artBefore,'Unlock and subsequent buy never alter the reserved art or strip');await sameInlineNodes(page);
      await shot(page,'inline-unlock-320'+(largeText ? '-text130' : '')+(wideFont?'-wide-font':''));await context.close();
    }
    const bulkSeed=StationFixtures.lesson('bulk');
    F.completeAreaGuides(bulkSeed);for(let n=0;n<40 && H.Core.getView(bulkSeed).onboarding.active;n++){const active=H.Core.getView(bulkSeed).onboarding.active;StationFixtures.act(bulkSeed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(bulkSeed).onboarding.active,null);StationFixtures.act(bulkSeed,{type:'onboarding-visit',id:'techniques'});for(let n=0;n<40 && H.Core.getView(bulkSeed).onboarding.active;n++){const active=H.Core.getView(bulkSeed).onboarding.active;StationFixtures.act(bulkSeed,active.practiceAction || active.inspectAction || active.action);}assert.equal(H.Core.getView(bulkSeed).onboarding.active,null);F.announceDiscoveries(bulkSeed);StationFixtures.act(bulkSeed,{type:'expedition-batch',count:1});bulkSeed.stations.encounter.remaining=0;
    for(const [largeText,wideFont] of [[false],[true],[true,true]]) {
      const {page,context}=await open(320,740,bulkSeed);await dismissNotices(page);if(largeText)await page.addStyleTag({content:text130});if(wideFont)await page.addStyleTag({content:wideInlineFont});
      const owner=await page.evaluate(()=>{const world=document.querySelector('.wx-station-world'),strip=[...world.querySelectorAll('.wx-station-controls')].slice(1).find(node=>node.getBoundingClientRect().bottom>=world.getBoundingClientRect().bottom);if(!strip)throw new Error('The mature fixture has no lower station below the viewport');world.scrollTop+=strip.getBoundingClientRect().bottom-world.getBoundingClientRect().bottom;return strip.closest('.wx-station-segment').dataset.wxStation;});await page.waitForTimeout(100);await page.clock.runFor(200);
      const collision=await page.evaluate(owner=>{const world=document.querySelector('.wx-station-world'),strip=world.querySelector('[data-wx-station="'+owner+'"] .wx-station-controls');return {bottom:strip.getBoundingClientRect().bottom,worldBottom:world.getBoundingClientRect().bottom,buttons:[...strip.querySelectorAll('button')].map(button=>{const rect=button.getBoundingClientRect(),hit=document.elementFromPoint(rect.x+rect.width/2,rect.y+rect.height/2);return {command:button.dataset.wxDo,covered:!hit || hit!==button && !button.contains(hit),hit:hit?.className};})};},owner);
      assert(Math.abs(collision.bottom-collision.worldBottom)<1,'Lower station strip can meet the world viewport bottom');assert.deepEqual(collision.buttons.filter(button=>button.covered),[],'Focus and optional cache never cover the inline controls at the viewport bottom');
      // The separate48px quantity target extends below a flush76px card.
      // Reveal it before the transaction baseline; native actionability scroll
      // must not be mistaken for a camera change caused by the purchase.
      const station=page.locator('[data-wx-station="'+owner+'"]'),quantity=station.locator('.wx-inline-quantity');await quantity.evaluate(node=>node.scrollIntoView({block:'center',inline:'nearest',behavior:'instant'}));await page.waitForTimeout(100);await page.clock.runFor(200);
      const before=await geometry(page),artBefore=await inlineGeometry(page);await rememberInlineNodes(page);
      const quantityBox=await quantity.boundingBox();assert(quantityBox.y>=before.world.y && quantityBox.y+quantityBox.height<=before.world.y+before.world.height,'The complete48px batch selector is inside the world before the transaction baseline');assert(await quantity.isVisible(),'An earned batch selector appears beside the station title');await quantity.click();assert.equal(await page.locator('.wx-sheet[open][data-kind="batch"]').count(),1,'Inline quantity uses the shared batch dialog');
      await page.locator('[data-wx-do="batch:5"]').click();if(await page.locator('.wx-sheet[open]').count())await page.locator('[data-wx-close]').click();assert.equal((await saved(page)).expedition.batch,5);assert.match(await page.locator('.wx-inline-quantity').first().innerText(),/×5/);
      const inspectedId=await station.locator('.wx-inline-upgrade').first().getAttribute('data-wx-inline-skill'),inspectedBefore=progression(await saved(page));await station.locator('[data-wx-station-help]').click();await helpModel(page,owner,inspectedId);assert.equal(await page.locator('.wx-station-help-preview>span').innerText(),'Next ×5');const bulkConfirm=page.locator('[data-wx-do="station-help-buy:'+inspectedId+'"]');assert.equal(await bulkConfirm.innerText(),'Upgrade ×5');assert.match(await bulkConfirm.getAttribute('aria-label'),/^Buy exactly 5 ranks of /);assert.deepEqual(progression(await saved(page)),inspectedBefore,'Five-rank preview never buys the batch');await shot(page,'inline-batch-help-320'+(largeText?'-text130':'')+(wideFont?'-wide-font':''));assert(await page.evaluate(()=>WayfarersUI.handleBack()));assert.deepEqual(await geometry(page),before);
      const first=station.locator('.wx-inline-upgrade').first(),id=await first.getAttribute('data-wx-inline-skill'),rank=(await saved(page)).stations.ranks[id];assert.match(await station.locator('.wx-inline-quantity').innerText(),/×5/,'The actual station selector clearly displays the shared five-rank quantity');assert.equal(await page.locator('.wx-inline-count').count(),0,'Compact cards reserve their action area for the real currency prices');assert.match(await first.locator('.wx-inline-buy').getAttribute('aria-label'),/^Buy 5 ranks of /,'The purchase control names its exact five-rank transaction');await readableInlinePrices(page);await first.locator('.wx-inline-buy').click();assert.equal((await saved(page)).stations.ranks[id],rank+5,'Inline batch button buys exactly the earned five ranks');assert.deepEqual(await geometry(page),before);assert.deepEqual(await inlineGeometry(page),artBefore);await readableInlinePrices(page);await sameInlineNodes(page);
      await shot(page,'inline-batch-320'+(largeText ? '-text130' : '')+(wideFont?'-wide-font':''));await context.close();
    }
    report.flows.push('320 and text130: actual ready unlock and five-rank batch purchase retain each canvas, cell and buy button, compact scene-contained controls and scene aspect; earned title selector uses the shared batch dialog');
    for(const largeText of [false,true]) {
      const statusSeed=mature();assert(H.Core.act(statusSeed,{type:'expedition-select',areaId:'quarry'}).ok);F.announceDiscoveries(statusSeed);
      const {page,context}=await open(390,844,statusSeed);await dismissNotices(page);if(largeText){await page.addStyleTag({content:text130+statusText130});await page.clock.runFor(300);}
      await page.locator('.wx-station-world').evaluate(node=>node.scrollTop=0);await page.waitForTimeout(100);await page.clock.runFor(150);
      const cache=page.locator('[data-wx-station-cache]');assert(await cache.isVisible(),'The actual ready cache is reachable in the first scene');
      const cacheHeight=(await cache.boundingBox()).height,camera=await geometry(page),before=H.Core.getView(await saved(page)).stations.encounter;assert(before.ready);
      await cache.click();await page.clock.runFor(50);const encounter=H.Core.getView(await saved(page)).stations.encounter;assert.equal(encounter.sequence,before.sequence+1,'The visible cache claims its actual reward once');assert(encounter.boost.remaining>0,'The actual claim earns an area boost');assert.deepEqual(await geometry(page),camera,'Claiming the real cache preserves the camera');
      const label='390'+(largeText?'-text130':''),claim=await clearWorldStatus(page,label+' actual claim');assert(claim.overlays.find(item=>item.kind==='boost').visible && claim.overlays.find(item=>item.kind==='reward').visible,'Actual reward and boost each have separate visible artwork after the claim');await shot(page,'status-claim-'+label);
      for(const scroll of [328,380]) {
        await page.locator('.wx-station-world').evaluate((node,value)=>node.scrollTop=value,scroll);await page.waitForTimeout(100);await page.clock.runFor(150);
        const result=await clearWorldStatus(page,label+' scroll'+scroll),boost=result.overlays.find(item=>item.kind==='boost'),reward=result.overlays.find(item=>item.kind==='reward');assert(boost.visible,'The smaller boost moves into a safe interval after the cache disappears');assert.equal(reward.hidden,false,'The reward is still active while its own placement is checked');assert(result.maxFree<cacheHeight+12,'This lower-art interval cannot accommodate the larger cache control');assert.equal((await geometry(page)).scroll,scroll);await shot(page,'status-scroll-'+scroll+'-'+label);
      }
      await page.clock.runFor(1700);const expired=await clearWorldStatus(page,label+' reward expired');assert(expired.overlays.find(item=>item.kind==='reward').hidden);assert(expired.overlays.find(item=>item.kind==='boost').visible,'Boost placement still updates when Focus/cache/reward are absent');await context.close();
    }
    report.flows.push('Normal and text130: actual cache reward, boost and reward animation envelope occupy independent clear artwork; scroll328/380 never covers starter triplets, station headings or rates; smaller boost remains visible where the larger cache cannot fit');
    const established=mature();assert(H.Core.act(established,{type:'expedition-select',areaId:'quarry'}).ok);F.announceDiscoveries(established);
    const {page,context}=await open(390,844,established);
    await dismissNotices(page);
    assert.equal(await page.locator('.wx-station-segment').count(),5);await inlineGeometry(page);await shot(page,'quarry-stacked-world-390');
    await page.locator('.wx-station-world').evaluate(node=>node.scrollTop=380);const scrolled=await geometry(page);await page.locator('[data-wx-nav="upgrades"]').click();await shot(page,'quarry-station-drawer-390');assert.deepEqual(await geometry(page),scrolled);
    await page.locator('[data-wx-do="upgrade-scope:area"]').click();assert.equal(await page.locator('.wx-station-row').count(),3);await shot(page,'quarry-area-drawer-390');assert.deepEqual(await geometry(page),scrolled);
    await page.locator('[data-wx-do="upgrade-scope:global"]').click();assert.equal(await page.locator('.wx-station-row').count(),0);assert(await page.locator('.wx-research-row').count()>0);await page.locator('[data-wx-drawer-close]').click();
    await page.locator('.wx-station-world').evaluate(node=>node.scrollTop=0);await page.clock.runFor(200);const cacheCamera=await geometry(page);await page.locator('[data-wx-station-cache]').click();await shot(page,'quarry-boost-390');assert(await page.locator('[data-wx-station-boost]').isVisible());assert.deepEqual(await geometry(page),cacheCamera,'Tapping an actually visible optional cache never moves the camera');await page.locator('.wx-station-world').evaluate((node,scroll)=>node.scrollTop=scroll,scrolled.scroll);await page.clock.runFor(200);assert.deepEqual(await geometry(page),scrolled);
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
      const {page,context}=await open(label==='bulk'?320:390,label==='bulk'?740:844,seed);if(label==='bulk')await page.addStyleTag({content:text130+wideInlineFont});const expected=H.Core.getView(seed).onboarding.guides.find(row=>row.id===id).steps.length,trace=[];
      async function saved() {return page.evaluate(()=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(WayfarersStorage.SAVE_KEY)).state;});}
      try {
        for(let n=0;n<30 && (await saved()).onboarding.practice.progress[id]<expected;n++) {
          await page.clock.runFor(250);const coach=page.locator('.wx-guide[open]');if(!await coach.count() && await page.locator('.wx-sheet[open][data-kind="collection-result"]').count()){trace.push({step:'result',target:'Done'});await page.locator('.wx-sheet .wx-confirm').click();}if(!await coach.count())await page.clock.runFor(1500);assert.equal(await coach.count(),1,id+' lesson remains active');
          while(/^currency:/.test(await coach.getAttribute('data-step'))){await page.locator('[data-guide-next]').click();await page.clock.runFor(100);}
          assert.equal(await coach.getAttribute('data-missing'),'false',id+' real destination is visible');const target=page.locator('[data-guide-target]');assert.equal(await target.count(),1);
          if(label==='tiers' && await target.getAttribute('data-wx-close')!==null)assert((await coach.innerText()).includes('Close these details, then press the highlighted Unlock control in the scene.'),'The action coach explains the actual highlighted close before the scene unlock');
          trace.push({step:await coach.getAttribute('data-step'),target:await target.getAttribute('data-wx-do') || await target.getAttribute('data-wx-nav') || await target.getAttribute('data-wx-close')});
          if(label==='plans-early' && await page.locator('.wx-sheet[open][data-kind="choice"]').count()){const actual=H.Core.getView(await saved()).stations.currentArea.stations.filter(station=>station.status==='built');assert.deepEqual(await page.locator('[data-wx-operation-station]').evaluateAll(nodes=>nodes.map(node=>({id:node.dataset.wxOperationStation,rate:node.querySelector('small').textContent}))),actual.map(station=>({id:station.id,rate:station.outputText})),'Processing shows canonical new-station rates');}
          if(label==='bulk'){await readableInlinePrices(page);if((await target.getAttribute('data-wx-do') || '').startsWith('inline-buy:'))assert(await target.locator('.wx-inline-price>span[data-wide="true"]').count(),'The actual funded bulk lesson exercises a long supplied tutorial quote');}
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
