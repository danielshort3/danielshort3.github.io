'use strict';

// Billing is a contract double in isolated profiles; this never makes purchases.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const {chromium} = require('playwright');
const {bundle} = require('../../build/bundle-wayfarers-android.cjs');
const H = require('./helpers/wayfarers-progression.cjs');
const Fixtures = require('./helpers/wayfarers-onboarding.cjs');
const Storage = require('../../js/games/wayfarers-guild/persistence');
const output = path.resolve(process.env.WAYFARERS_QA_DIR || fs.mkdtempSync(path.join(os.tmpdir(),'guild-wallet-rail-')));

(async () => {
  fs.mkdirSync(output,{recursive:true});
  const files=path.join(output,'bundle');bundle(files);
  const server=http.createServer((request,response)=>{
    const pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname).replace(/^\/assets\//,'/');
    const file=path.resolve(files,'.'+pathname);
    if(!file.startsWith(files+path.sep)){response.writeHead(403).end();return;}
    fs.readFile(file,(error,data)=>{if(error){response.writeHead(404).end();return;}response.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.webp':'image/webp'})[path.extname(file)]||'application/json');response.end(data);});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true});
  const errors=[];
  try {
    const seed=Fixtures.announceDiscoveries(Fixtures.completeAreaGuides(H.mature()));
    seed.resources.starshards=H.N.from(3);seed.lastUpdate=1000;
    const saved=Storage.createStore({storage:null,now:()=>1000}).export(seed);assert(saved.ok,saved.message);
    for(const native of [false,true]) {
      const context=await browser.newContext({viewport:{width:320,height:740},reducedMotion:'reduce'});
      await context.addInitScript(({key,text,native})=>{
        localStorage.setItem(key,text);
        if(!native)return;
        window.walletRequests=[];
        window.walletReply=(revision,balance)=>window.dispatchEvent(new CustomEvent('wayfarers:billing',{detail:{id:'event',ok:true,data:{wallet:{walletId:'a'.repeat(64),revision,balance,debt:0,owned:[],catalogOwned:[]}}}}));
        window.WayfarersPlayBilling={postMessage(text){
          const request=JSON.parse(text);window.walletRequests.push(request.method);
          setTimeout(()=>window.dispatchEvent(new CustomEvent('wayfarers:billing',{detail:{id:request.id,ok:true,data:{available:false,configured:true,freePlaySafe:true,message:'Test wallet is still restoring.'}}})),0);
        }};
      },{key:Storage.SAVE_KEY,text:saved.text,native});
      const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
      await page.clock.install({time:new Date(1000)});
      await page.goto('http://127.0.0.1:'+server.address().port+'/assets/wayfarers/index.html');
      await page.clock.runFor(1000);
      const shard=page.locator('[data-wx-currency="starshards"]');
      await shard.scrollIntoViewIfNeeded();
      assert.equal(await shard.locator('strong').textContent(),'3','Unknown paid balance is not counted as owned currency');
      await shard.click();
      const sheet=page.locator('.wx-sheet[open]');
      const unknownText=await sheet.innerText();
      if(native)assert.match(unknownText,/restor|pending|loading|unverified|not yet/i,'Unverified paid wallet has an explicit status');
      else assert.match(unknownText,/unavailable|not available|earned/i,'Standalone earned balance is described truthfully');
      await page.locator('[data-wx-close]').click();
      if(native) {
        await page.evaluate(()=>walletReply(2,7));await page.clock.runFor(50);
        assert.equal(await shard.locator('strong').textContent(),'10','Verified paid and earned Starshards appear together');
        await shard.click();
        assert.match(await sheet.innerText(),/3[\s\S]*7|7[\s\S]*3/,'Detail preserves separate earned and purchased balances');
        await page.locator('[data-wx-close]').click();
        await page.evaluate(()=>{walletReply(1,90);walletReply(3,-5);});await page.clock.runFor(50);
        assert.equal(await shard.locator('strong').textContent(),'10','Stale or invalid wallet cannot change the currency rail');
        assert(!(await page.evaluate(()=>walletRequests)).some(method=>['purchase','spend'].includes(method)));
      }
      const persisted=await page.evaluate(key=>{document.dispatchEvent(new Event('freeze'));return JSON.parse(localStorage.getItem(key)).state;},Storage.SAVE_KEY);
      assert.equal(H.N.toNumber(persisted.resources.starshards),3,'Paid balance never enters an exported game wallet');
      await page.screenshot({path:path.join(output,native?'verified-wallet.png':'earned-wallet.png')});
      await context.close();
    }
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({result:'PASS',profiles:2,output}));
  } finally {await browser.close();server.close();}
})().catch(error=>{console.error(error.stack);process.exitCode=1;});
