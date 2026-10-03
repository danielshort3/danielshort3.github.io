'use strict';

// Isolated real-control mechanics. The game-flow suite separately verifies
// authoritative action proof, practice inventory and durable save behavior.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {chromium} = require('playwright');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-practice-ui-')));
const evidence = {cases:[],errors:[]};

(async () => {
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true});
  try {
    for (const [width,height] of [[320,740],[390,844],[915,390]]) {
      for (const fallback of [false,true]) {
        const page = await browser.newPage({viewport:{width,height},hasTouch:true});
        page.on('pageerror',error=>evidence.errors.push(error.message));
        await page.setContent('<main><button id="other">Unrelated purchase</button><button id="outside">Options</button></main><dialog class="wx-sheet" data-kind="choice" id="sheet"><header><h2>Plans</h2><button id="close">Close</button></header><div class="wx-sheet-content"><button id="required" style="width:100%;min-height:52px">Choose this plan</button><button id="unrelated" style="min-height:48px">Other plan</button></div></dialog>');
        await page.addStyleTag({path:path.resolve('css/games/wayfarers-guild.css')});
        await page.addScriptTag({path:path.resolve('js/games/wayfarers-guild/onboarding-ui.js')});
        await page.evaluate(({fallback})=>{
          if (fallback) HTMLDialogElement.prototype.showPopover=undefined;
          window.actions=0; window.otherActions=0; window.step=0;
          document.querySelector('#required').addEventListener('click',()=>{window.actions++;});
          document.querySelector('#unrelated').addEventListener('click',()=>{window.otherActions++;});
          document.querySelector('#sheet').showModal();
          window.coach=WayfarersOnboardingUI.create({parent:document.body,resolveTarget:()=>({element:document.querySelector('#required'),allowed:[document.querySelector('#close')]}),onLeave:()=>coach.hide(),onBack:()=>coach.hide(),onNext:()=>window.step++});
          coach.show({guideId:'practice',stepId:'choose',mode:'action',index:0,total:1,title:'Practice',heading:'Choose a plan',body:'Use the highlighted control. The result is saved before this lesson continues.',target:'required'});
        },{fallback});
        assert.equal(await page.locator('[data-guide-next]').isVisible(),false,'Action lesson has no acknowledgment shortcut');
        assert(await page.locator('#required').evaluate(node=>node===document.activeElement),'Actual required control receives focus');
        assert.match(await page.locator('#required').getAttribute('aria-describedby'),/wx-guide-body/);
        await page.locator('#required').click();
        assert.equal(await page.evaluate(()=>window.actions),1,'Real target keeps its original event handler');
        assert.equal(await page.evaluate(()=>window.step),0,'Raw clicks do not acknowledge a lesson');
        await page.locator('#unrelated').evaluate(node=>node.click());
        assert.equal(await page.evaluate(()=>window.otherActions),0,'Captured-input guard blocks unrelated actions');
        assert(await page.locator('#unrelated').evaluate(node=>node.inert),'Unrelated controls leave the accessible action set');
        for (let i=0;i<6;i++) {
          await page.keyboard.press('Tab');
          assert(await page.evaluate(()=>document.activeElement.matches('#required,#close,[data-guide-leave]')),'Focus cycles only through real practice and recovery controls');
        }
        const geometry = await page.locator('.wx-guide-card').boundingBox();
        assert(geometry.x>=0 && geometry.y>=0 && geometry.x+geometry.width<=width+1 && geometry.y+geometry.height<=height+1,'Coach fits viewport');
        assert(await page.locator('[data-guide-leave]').evaluate(node=>{const r=node.getBoundingClientRect();return r.height>=48 && document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===node;}),'Recovery control is visible and hit-testable');
        assert(await page.locator('#close').evaluate(node=>{const r=node.getBoundingClientRect();return document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)===node;}),'Necessary sheet Close remains touch-accessible through the scrim');
        await page.screenshot({path:path.join(output,'practice-helper-'+width+(fallback?'-fallback':'')+'.png')});
        await page.keyboard.press('Escape');
        assert.equal(await page.locator('.wx-guide[open]').count(),0,'Back leaves without completing');
        assert.equal(await page.evaluate(()=>window.step),0);
        assert.equal(await page.locator('#unrelated').evaluate(node=>node.inert),false,'Temporary isolation is restored');
        assert.equal(await page.locator('#sheet').evaluate(node=>node.classList.contains('wx-guide-surface')),false,'Fallback sheet framing is restored');
        assert.equal(await page.locator('#required').getAttribute('aria-describedby'),null,'Temporary description is restored');
        evidence.cases.push({width,height,fallback,geometry});
        await page.close();
      }
    }
    assert.deepEqual(evidence.errors,[]);
    console.log(JSON.stringify({result:'PASS',cases:evidence.cases.length,output}));
  } finally {
    fs.writeFileSync(path.join(output,'practice-helper-browser.json'),JSON.stringify(evidence,null,2));
    await browser.close();
  }
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
