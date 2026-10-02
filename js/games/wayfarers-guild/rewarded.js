(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.WayfarersRewarded = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const WEB_MESSAGE = 'Rewarded ads are available in the configured Android app. Your caravan will wait, or you can skip it.';
  const RESOURCES = ['coins','ore','herbs','provisions','knowledge','maps'];
  const RELICS = ['golden-pickaxe','surveyors-lens','living-crucible'];
  const STATES = ['ready','showing','verification_pending','cancelled','rewarded','unavailable'];
  const copy = value => JSON.parse(JSON.stringify(value));
  const exact = (value, fields) => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === fields.length && fields.every(key => Object.prototype.hasOwnProperty.call(value,key));
  const identity = value => typeof value === 'string' && /^[A-Za-z0-9_.:-]{1,160}$/.test(value);
  function validQuote(quote) {
    if (!exact(quote,['version','offerId','kind','golden','issuedAt','runId','charters','reward']) || quote.version !== 1 || !identity(quote.offerId) || !['shipment','surge','relic'].includes(quote.kind) || typeof quote.golden !== 'boolean' || ![quote.issuedAt,quote.runId,quote.charters].every(value=>Number.isSafeInteger(value)&&value>=0)) return false;
    if (!exact(quote.reward,quote.kind === 'shipment' ? ['resources'] : ['resources',quote.kind])) return false;
    const resources=quote.reward.resources;
    if (!resources || typeof resources !== 'object' || Array.isArray(resources) || Object.keys(resources).some(id=>!RESOURCES.includes(id))) return false;
    if (!Object.values(resources).every(value=>exact(value,['m','e']) && Number.isFinite(value.m) && Number.isSafeInteger(value.e) && Math.abs(value.e)<=1e12 && (value.m===0&&value.e===0 || value.m>=1&&value.m<10))) return false;
    if (quote.kind==='shipment' && !Object.values(resources).some(value=>value.m>0)) return false;
    if (quote.kind==='surge') {
      const surge=quote.reward.surge;
      if (!exact(surge,['resource','multiplier','seconds']) || !RESOURCES.includes(surge.resource) || surge.multiplier!==3 || surge.seconds!==(quote.golden?5400:2700)) return false;
    }
    if (quote.kind==='relic') {
      const relic=quote.reward.relic;
      if (!exact(relic,['id','progress']) || !RELICS.includes(relic.id) || relic.progress!==(quote.golden?40:20)) return false;
    }
    return true;
  }
  const validReceipt = receipt => exact(receipt,['receiptId','offerId','quote','completedAt']) && identity(receipt.receiptId) && identity(receipt.offerId) && validQuote(receipt.quote) && receipt.quote.offerId===receipt.offerId && Number.isSafeInteger(receipt.completedAt) && receipt.completedAt>=receipt.quote.issuedAt;
  function createClient(options) {
    const settings=options||{};
    const host=settings.root||globalThis;
    const listeners=new Set();
    const requests=new Map();
    const receipts=new Map();
    const acknowledged=new Set();
    let sequence=0;
    let revision=0;
    let destroyed=false;
    const native=()=>!destroyed&&host.WayfarersRewardedAds&&typeof host.WayfarersRewardedAds.postMessage==='function';
    let snapshot={native:!!native(),available:false,configured:false,pending:false,receipt:null,receipts:[],state:'unavailable',message:WEB_MESSAGE,privacyOptionsRequired:false};
    function publish(data, stale) {
      if (!data || typeof data!=='object' || destroyed) return;
      const next=Object.assign({},snapshot,{native:!!native()});
      const deliveries=Array.isArray(data.receipts)?data.receipts.slice(0,128):data.receipt?[data.receipt]:[];
      deliveries.forEach(receipt=>{if(validReceipt(receipt)&&!acknowledged.has(receipt.receiptId)&&receipts.size<128) receipts.set(receipt.receiptId,copy(receipt));});
      if (!stale) {
        for(const key of ['available','configured','pending','privacyOptionsRequired']) if(typeof data[key]==='boolean')next[key]=data[key];
        if(STATES.includes(data.state))next.state=data.state;
        if(typeof data.message==='string')next.message=data.message.slice(0,500);
        if(Number.isSafeInteger(data.remaining)&&data.remaining>=0&&data.remaining<=3)next.remaining=data.remaining;
      }
      next.receipts=Array.from(receipts.values()).map(copy);
      next.receipt=next.receipts[0]||null;
      snapshot=next;
      listeners.forEach(listener=>listener(copy(snapshot)));
    }
    function call(method,args) {
      if(!native())return Promise.reject(new Error(WEB_MESSAGE));
      const id='wg-ad-'+Date.now().toString(36)+'-'+(++sequence);
      return new Promise((resolve,reject)=>{
        const timeout=host.setTimeout(()=>{requests.delete(id);reject(Object.assign(new Error('The caravan service has not responded. Refresh its status before trying again.'),{uncertain:true}));},settings.timeoutMs||45000);
        requests.set(id,{resolve,reject,timeout,revision});
        try{host.WayfarersRewardedAds.postMessage(JSON.stringify({id,method,args:args||{}}));}
        catch(error){host.clearTimeout(timeout);requests.delete(id);reject(Object.assign(new Error('The ad could not open. Refresh its status before trying again.'),{uncertain:true}));}
      });
    }
    function receive(event) {
      const detail=event&&event.detail;
      if(!detail||typeof detail!=='object'||typeof detail.id!=='string')return;
      const request=requests.get(detail.id);
      if(!request&&detail.id!=='event')return;
      if(detail.ok===true){publish(detail.data,request&&request.revision<revision);revision+=1;}
      else if (request && detail.data && typeof detail.data === 'object') {
        // Failed replies may update readiness, but can never deliver rewards.
        const flags = Object.fromEntries(['available','configured','pending','state','message','privacyOptionsRequired','remaining'].filter(key => Object.prototype.hasOwnProperty.call(detail.data,key)).map(key => [key,detail.data[key]]));
        publish(flags,request.revision<revision);
        revision += 1;
      }
      if(!request)return;
      host.clearTimeout(request.timeout);requests.delete(detail.id);
      if(detail.ok===true)request.resolve(detail.data||{});
      else request.reject(Object.assign(new Error(typeof detail.error==='string'?detail.error.slice(0,500):'The ad did not complete. Your caravan reward has not changed.'),{uncertain:!(detail.data&&detail.data.pending===false&&['ready','unavailable','cancelled'].includes(detail.data.state))}));
    }
    host.addEventListener('wayfarers:rewarded',receive);
    return {
      snapshot:()=>copy(snapshot),
      subscribe(listener){listeners.add(listener);return()=>listeners.delete(listener);},
      async refresh(){if(native())await call('status');return copy(snapshot);},
      async watch(quote){if(!validQuote(quote))throw new Error('The caravan reward is invalid.');if(!snapshot.available||!snapshot.configured||snapshot.pending)throw new Error('An ad is not ready yet. Your caravan will wait.');return call('watch',{quote:copy(quote)});},
      async acknowledge(receiptId){if(!identity(receiptId)||!receipts.has(receiptId))throw new Error('Unknown caravan delivery.');await call('acknowledge',{receiptId});receipts.delete(receiptId);acknowledged.add(receiptId);if(acknowledged.size>256)acknowledged.delete(acknowledged.values().next().value);publish({});},
      async feedback(type){if(!['rare','epic','legendary'].includes(type)||!native())return;return call('feedback',{type});},
      privacyOptions:()=>call('privacyOptions'),
      destroy(){destroyed=true;host.removeEventListener('wayfarers:rewarded',receive);requests.forEach(request=>{host.clearTimeout(request.timeout);request.reject(Object.assign(new Error('The caravan window was closed. Its saved delivery will be checked when you return.'),{uncertain:true}));});requests.clear();listeners.clear();}
    };
  }
  return {WEB_MESSAGE,validQuote,validReceipt,createClient};
});
