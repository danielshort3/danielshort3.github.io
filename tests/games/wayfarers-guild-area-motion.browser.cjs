'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {chromium} = require('playwright');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-area-motion-')));

(async () => {
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true});
  const evidence = {cases:[],errors:[]};
  try {
    for (const [width,height] of [[320,740],[390,844],[915,390]]) {
      const page = await browser.newPage({viewport:{width,height}});
      page.on('pageerror',error=>evidence.errors.push(error.message));
      await page.setContent('<main class="wx-game"><header class="wx-header">Currencies</header><div>Area selector</div><section class="wx-play" id="play"><div class="wx-world"><canvas id="scene" data-wx-scene width="320" height="300"></canvas></div><div class="wx-tray" data-count="3"><button id="upgrade" data-wx-do="upgrade">Upgrade</button></div></section><footer>Guild</footer></main>');
      await page.addStyleTag({path:path.resolve('css/games/wayfarers-guild.css')});
      await page.addStyleTag({content:'body{margin:0}.wx-game{width:100vw;height:100vh}.wx-header{height:60px}.wx-play{background:#152b39}.wx-tray{display:grid;grid-template-columns:repeat(3,1fr)}'});
      await page.addScriptTag({path:path.resolve('js/games/wayfarers-guild/area-motion.js')});
      await page.evaluate(()=>{
        window.commits=0; window.quiet=false;
        const ctx=document.querySelector('#scene').getContext('2d');
        ctx.fillStyle='#ff0000';ctx.fillRect(0,0,320,300);
        window.motion=WayfarersAreaMotion.create({element:document.querySelector('#play'),quiet:()=>window.quiet});
        window.commit=()=>{window.commits++;ctx.fillStyle='#0000ff';ctx.fillRect(0,0,320,300);return {ok:true};};
      });
      const original = await page.locator('#play').boundingBox();
      const header = await page.locator('header').boundingBox();
      const outgoing = await page.evaluate(()=>{
        motion.begin();motion.drag(-50,false);
        const result=motion.transition(1,commit);
        const ghost=document.querySelector('.wx-area-ghost');
        return {
          result,commits,phase:document.querySelector('#play').dataset.areaMotion,
          inert:ghost.inert,hidden:ghost.getAttribute('aria-hidden'),
          duplicateHooks:ghost.querySelectorAll('[id],[data-wx-do],[data-wx-scene]').length,
          disabled:ghost.querySelector('button').disabled,
          count:ghost.querySelector('[data-count]').dataset.count,
          pixel:Array.from(ghost.querySelector('canvas').getContext('2d').getImageData(1,1,1,1).data),
          livePixel:Array.from(document.querySelector('#scene').getContext('2d').getImageData(1,1,1,1).data)
        };
      });
      assert.deepEqual(outgoing,{result:{ok:true},commits:1,phase:'transition',inert:true,hidden:'true',duplicateHooks:0,disabled:false,count:'3',pixel:[255,0,0,255],livePixel:[0,0,255,255]});
      assert.deepEqual(await page.locator('header').boundingBox(),header,'Currency/header geometry stays fixed during area transition');
      await page.screenshot({path:path.join(output,'area-slide-'+width+'.png')});
      await page.waitForFunction(()=>!motion.isAnimating());
      assert.deepEqual(await page.locator('#play').boundingBox(),original);
      assert.equal(await page.locator('.wx-area-ghost').count(),0);
      assert.equal(await page.evaluate(()=>commits),1,'Animation completion never dispatches another action');

      await page.evaluate(()=>{motion.begin();motion.drag(100,true);});
      const edgeOffset=await page.locator('#play').evaluate(node=>new DOMMatrix(getComputedStyle(node).transform).m41);
      assert(edgeOffset<=width*.08 && edgeOffset>0,'First/last area has restrained edge resistance');
      await page.evaluate(()=>motion.cancel());
      await page.waitForFunction(()=>!motion.isAnimating());
      assert.equal(await page.evaluate(()=>commits),1,'Cancelled gestures never select an area');

      assert.deepEqual(await page.evaluate(()=>motion.transition(1,()=>({ok:false,error:'save-failed'}))),{ok:false,error:'save-failed'});
      assert.equal(await page.locator('.wx-area-ghost').count(),0,'Failed selection leaves no phantom outgoing frame');
      assert.equal(await page.evaluate(()=>{try{motion.transition(1,()=>{throw Error('save-failed');});}catch(error){return error.message;}}),'save-failed');
      assert.equal(await page.evaluate(()=>motion.isAnimating()),false);

      await page.evaluate(()=>{motion.transition(1,commit);motion.transition(-1,commit);window.dispatchEvent(new Event('resize'));});
      assert.equal(await page.locator('.wx-area-ghost').count(),0,'Resize cancels every superseded transition');
      assert.equal(await page.evaluate(()=>commits),3);
      await page.waitForTimeout(380);
      assert.deepEqual(await page.locator('#play').boundingBox(),original,'Stale animation callbacks cannot change the new layout');

      await page.evaluate(()=>{window.quiet=true;motion.transition(1,commit);motion.drag(-90,false);});
      assert.equal(await page.locator('.wx-area-ghost').count(),0,'Quiet animations switches immediately');
      assert.equal(await page.evaluate(()=>motion.isAnimating()),false);
      await page.evaluate(()=>{window.quiet=false;});
      await page.emulateMedia({reducedMotion:'reduce'});
      await page.evaluate(()=>motion.transition(1,commit));
      assert.equal(await page.evaluate(()=>motion.isAnimating()),false,'System reduced motion switches immediately');
      await page.emulateMedia({reducedMotion:'no-preference'});
      await page.evaluate(()=>{motion.transition(1,commit);motion.dispose();});
      assert.equal(await page.locator('.wx-area-ghost').count(),0);
      assert.equal(await page.locator('#play').evaluate(node=>node.style.pointerEvents),'','Disposal restores controls');
      assert.equal(await page.locator('.wx-game').evaluate(node=>node.style.overflow),'','Disposal restores overflow');
      evidence.cases.push({width,height,original,header,edgeOffset});
      await page.close();
    }
    assert.deepEqual(evidence.errors,[]);
    console.log(JSON.stringify({result:'PASS',cases:evidence.cases.length,output}));
  } finally {
    fs.writeFileSync(path.join(output,'area-motion-browser.json'),JSON.stringify(evidence,null,2));
    await browser.close();
  }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
