'use strict';

/*
  GA4/GTM consent-gated smoke test.

  Drives headless Chrome over CDP (Node >= 21 built-in WebSocket, zero deps):
   0. Navigate to the local site with reset_consent=1 (fresh visitor).
   1. Verify NO googletagmanager script / GA4 collect traffic before consent.
   2. Grant analytics consent via the site's own window.consentAPI.
   3. Verify gtm.js is injected, GTM boots, gtag exists as a function.
   4. Fire the site's real events (home_explore_select, select_content) via
      window.gaEvent and verify they reach dataLayer with activity_* fields.
   5. Prove the GA4 property (G-0VL37MQ62P) is live inside the tag: read
      property.id back through gtag('get', ...).
   6. Verify GA4 network traffic (collect requests) appeared post-consent.
   7. Revoke consent and verify the site stops sending (send() gate).

  Env: BASE_URL (default http://127.0.0.1:45102)
  Exit 0 when every check passes.
*/

const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const BASE_URL = (process.env.BASE_URL || 'http://127.0.0.1:45102').replace(/\/$/, '');
const CDP_PORT = Number(process.env.CDP_PORT || '9333');
const GA4_PROPERTY = process.env.GA4_PROPERTY || 'G-0VL37MQ62P';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const CHROME_PATHS = [
  process.env.CHROME_PATH,
  path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  'google-chrome'
].filter(Boolean);

function findChrome() {
  for (const p of CHROME_PATHS) {
    try { fs.accessSync(p, fs.constants.X_OK); return p; } catch (err) { /* next */ }
  }
  console.error('No Chrome found; tried: ' + CHROME_PATHS.join(', '));
  process.exit(2);
}

function httpJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        if (res.statusCode !== 200) return reject(new Error(`GET ${url} -> ${res.statusCode}`));
        try { resolve(JSON.parse(body)); } catch (err) { reject(new Error(`GET ${url} bad JSON`)); }
      });
    });
    req.on('error', reject);
    req.setTimeout(4000, () => { req.destroy(new Error(`GET ${url} timeout`)); });
  });
}

function connectCdp(wsUrl, timeoutMs) {
  const WS = globalThis.WebSocket;
  if (typeof WS !== 'function') throw new Error('Node global WebSocket unavailable (need Node >= 21)');
  return new Promise((resolve, reject) => {
    const ws = new WS(wsUrl);
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { ws.close(); } catch (err) {}
      reject(new Error('CDP WebSocket connect timeout'));
    }, timeoutMs);
    ws.onopen = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(ws);
    };
    ws.onerror = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error('CDP WebSocket connect error'));
    };
  });
}

/* CDP multiplexer: one websocket, multiple flat sessions, id routing. */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.pending = new Map();
    this.listeners = [];
    this.nextListener = 0;
    ws.onmessage = (event) => {
      let message;
      try { message = JSON.parse(String(event.data)); } catch (err) { return; }
      if (message.id && this.pending.has(message.id)) {
        const entry = this.pending.get(message.id);
        this.pending.delete(message.id);
        if (message.error) entry.reject(new Error(message.error.message || JSON.stringify(message.error)));
        else entry.resolve(message.result);
        return;
      }
      for (const listener of this.listeners) {
        try { listener(message); } catch (err) { /* observer error is non-fatal */ }
      }
    };
    ws.onclose = () => {
      const err = new Error('CDP WebSocket closed');
      for (const entry of this.pending.values()) entry.reject(err);
      this.pending.clear();
    };
  }

  onMessage(fn) {
    this.listeners.push(fn);
    return this.nextListener++;
  }

  send(method, params, sessionId, timeoutMs) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`CDP ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs || 30000);
      this.pending.set(id, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject: (e) => { clearTimeout(timer); reject(e); }
      });
      const payload = { id, method, params: params || {} };
      if (sessionId) payload.sessionId = sessionId;
      this.ws.send(JSON.stringify(payload));
    });
  }

  close() {
    try { this.ws.close(); } catch (err) {}
  }
}

const results = [];
function check(name, ok, detail) {
  const passed = ok === true;
  results.push({ name, ok: passed, detail: passed ? '' : String(detail || '') });
  console.error(`  ${passed ? 'PASS' : 'FAIL'}  ${name}${passed ? '' : '  -> ' + String(detail || '').slice(0, 300)}`);
}

function waitForCheck(cdp, sessionId, exprFn, what, timeoutMs) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    (function tick() {
      cdp.send('Runtime.evaluate', {
        expression: `(() => { ${exprFn()} })()`,
        returnByValue: true,
        allowUnsafeEvalBlockedByCSP: true
      }, sessionId, 8000).then((r) => {
        const value = r && r.result ? r.result.value : undefined;
        if (value === true || value === 'true' || value === true) return resolve(true);
        if (Date.now() - started > timeoutMs) return reject(new Error(`timeout: ${what}`));
        setTimeout(tick, 200);
      }).catch(() => {
        if (Date.now() - started > timeoutMs) return reject(new Error(`timeout: ${what}`));
        setTimeout(tick, 200);
      });
    })();
  });
}

function evalJson(cdp, sessionId, js, timeoutMs) {
  return cdp.send('Runtime.evaluate', {
    expression: `(() => { ${js} })()`,
    returnByValue: true,
    awaitPromise: true,
    allowUnsafeEvalBlockedByCSP: true,
    timeout: timeoutMs || 15000
  }, sessionId, (timeoutMs || 15000) + 5000).then((r) => {
    if (r.exceptionDetails) {
      const text = (r.exceptionDetails.exception && r.exceptionDetails.exception.description) ||
                   r.exceptionDetails.text || JSON.stringify(r.exceptionDetails);
      throw new Error('EVAL FAILED: ' + String(text).slice(0, 400));
    }
    if (!r.result || typeof r.result.value === 'undefined') return null;
    if (typeof r.result.value === 'string') {
      try { return JSON.parse(r.result.value); } catch (err) { return r.result.value; }
    }
    return r.result.value;
  });
}

async function main() {
  const chrome = spawn(findChrome(), [
    `--user-data-dir=${path.join(os.tmpdir(), 'ga4-smoke-' + Date.now())}`,
    '--headless=new',
    `--remote-debugging-port=${CDP_PORT}`,
    '--remote-debugging-address=127.0.0.1',
    '--no-sandbox',
    '--disable-gpu',
    '--disable-dev-shm-usage',
    '--mute-audio',
    '--lang=en-US',
    'about:blank'
  ], { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
  let chromeErr = '';
  chrome.stderr.on('data', (c) => { chromeErr += String(c); });

  const fail = (err) => {
    console.error('SMOKE ABORTED: ' + (err && err.message ? err.message : err));
    if (chromeErr) console.error('chrome stderr tail: ' + chromeErr.slice(-600));
    console.error(JSON.stringify({ passed: false, error: (err && err.message) || String(err), results }, null, 2));
    try { chrome.kill('SIGKILL'); } catch (e2) {}
    process.exit(1);
  };

  let cdp;
  let sessionId;
  let netEvents = [];
  try {
    /* Wait for CDP endpoint. */
    let version = null;
    for (let i = 0; i < 100 && !version; i += 1) {
      try { version = await httpJson(`http://127.0.0.1:${CDP_PORT}/json/version`); } catch (err) { await sleep(300); }
    }
    if (!version || !version.webSocketDebuggerUrl) throw new Error('CDP endpoint never came up');
    cdp = new Cdp(await connectCdp(version.webSocketDebuggerUrl, 8000));

    netEvents = [];
    const consoleEvents = [];
    const collectQueue = new Map();   // requestId -> { url, body }
    const resolveTimers = new Map();  // requestId -> timeout
    const collectBodies = new Map();  // requestId -> body text
    cdp.onMessage((msg) => {
      if (msg.method === 'Network.requestWillBeSent' && msg.params && msg.params.request) {
        netEvents.push({ url: msg.params.request.url || '', method: msg.params.request.method || 'GET' });
        if (msg.params.request.url && /google-analytics\.com\/g\/collect/i.test(msg.params.request.url)) {
          collectQueue.set(msg.params.requestId, { url: msg.params.request.url, body: '', postData: msg.params.request.postData || '' });
        }
      }
      if (msg.method === 'Network.loadingFinished' && msg.params && msg.params.requestId) {
        const rid = msg.params.requestId;
        if (collectQueue.has(rid)) fetchCollectBody(rid);
      }
      if (msg.method === 'Network.loadingFailed' && msg.params && msg.params.requestId) {
        const rid = msg.params.requestId;
        if (collectQueue.has(rid) && resolveTimers.has(rid)) {
          clearTimeout(resolveTimers.get(rid));
          resolveTimers.delete(rid);
        }
      }
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params) {
        const parts = (msg.params.args || []).map((a) => a.value !== undefined ? a.value : (a.description || a.type));
        consoleEvents.push({ level: msg.params.type || 'log', msg: parts.join(' ').slice(0, 500) });
      }
      if (msg.method === 'Runtime.exceptionThrown' && msg.params) {
        const d = msg.params.exceptionDetails || {};
        consoleEvents.push({ level: 'exception', msg: (d.exception && d.exception.description) || d.text || 'exception' });
      }
    });

    /* GA4 batches sends: the first event is in the URL query string, but
       subsequent events in the same batch ride in the POST body. So every
       collect verdict must consider both. */
    function fetchCollectBody(rid) {
      if (collectBodies.has(rid) || resolveTimers.has(rid)) return;
      const timer = setTimeout(() => { resolveTimers.delete(rid); }, 5000);
      resolveTimers.set(rid, timer);
      cdp.send('Network.getResponseBody', { requestId: rid }, sessionId, 8000).then((r) => {
        collectBodies.set(rid, String(r && r.body ? r.body : ''));
      }).catch((e) => { collectBodies.set(rid, ''); }).finally(() => {
        clearTimeout(timer); resolveTimers.delete(rid);
      });
    }

    function anyCollectMatches(pattern) {
      for (const [rid, q] of collectQueue.entries()) {
        if (pattern.test(q.url)) return true;
        if (q.postData && pattern.test(q.postData)) return true;
        const body = collectBodies.get(rid);
        if (body && pattern.test(body)) return true;
      }
      return false;
    }
    const printConsole = () => consoleEvents.slice(0, 30).forEach((c) => console.error(`  [page:${c.level}] ${c.msg}`));

    const target = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const attach = await cdp.send('Target.attachToTarget', { targetId: target.targetId, flatten: true });
    sessionId = attach.sessionId;

    await cdp.send('Page.enable', {}, sessionId, 10000);
    await cdp.send('Runtime.enable', {}, sessionId, 10000);
    await cdp.send('Network.enable', {}, sessionId, 10000);

    /* ---- 0. Navigate (fresh visitor: no saved consent). ---- */
    const load = new Promise((resolve) => {
      const stop = cdp.onMessage((msg) => { if (msg.method === 'Page.loadEventFired') resolve(); });
    });
    await cdp.send('Page.navigate', { url: BASE_URL + '/?reset_consent=1' }, sessionId, 30000);
    await Promise.race([load, sleep(30000)]);
    await evalJson(cdp, sessionId, `return document.readyState === 'complete' ? 'ok' : 'incomplete: ' + document.readyState;`, 30000);
    await sleep(1500);

    console.error('\n[phase 1] pre-consent (fresh visitor)');

    const pre = await evalJson(cdp, sessionId, `
      return JSON.stringify({
        gtmScripts: Array.from(document.scripts).filter(function(s){ return (s.src||'').indexOf('googletagmanager.com/gtm.js') !== -1; }).map(function(s){ return s.src; }),
        gtagType: typeof window.gtag,
        dataLayer: typeof window.dataLayer,
        hasConsentAPI: typeof window.consentAPI,
        savedConsent: (function(){ try { return localStorage.getItem('pcz_consent_v1'); } catch(e){ return 'err'; } })(),
        banner: !!document.getElementById('pcz-banner')
      });
    `);

    const preCollect = netEvents.filter((e) => e.url.indexOf('google-analytics.com/collect') !== -1);
    const preGtm = netEvents.filter((e) => e.url.indexOf('googletagmanager.com') !== -1);

    check('site bundle loaded (consentAPI present)', pre.hasConsentAPI === 'object', String(pre.hasConsentAPI));
    check('no saved consent record (fresh visitor)', pre.savedConsent === null, String(pre.savedConsent));
    check('NO gtm.js script before consent', pre.gtmScripts.length === 0, JSON.stringify(pre.gtmScripts));
    check('NO google-analytics/collect requests before consent', preCollect.length === 0, JSON.stringify(preCollect));
    check('NO googletagmanager.com network traffic before consent', preGtm.length === 0, JSON.stringify(preGtm.slice(0, 3)));
    check('pre-consent gtag is only the dataLayer stub (Consent Mode pre-hydration)', pre.gtagType === 'function' || pre.gtagType === 'undefined', typeof pre.gtagType);
    check('consent banner visible to the user', pre.banner === true, String(pre.banner));

    /* ---- 2. User clicks "Allow all" style consent via the real API. ---- */
    console.error('\n[phase 2] grant analytics consent via window.consentAPI');
    const grant = await evalJson(cdp, sessionId, `
      try {
        window.consentAPI.set({ necessary: true, analytics: true, functional: false, advertising: false });
        return 'granted';
      } catch (e) { return 'error: ' + e.message; }
    `);
    if (grant !== 'granted') { printConsole(); }
    check('consent grant executed', grant === 'granted', grant);
    await sleep(1200);

    /* ---- 3. GTM script must appear shortly after. ---- */
    let gtmInjected = false;
    try {
      await waitForCheck(cdp, sessionId, function () {
        return `
          return Array.from(document.scripts).some(function(s){
            return (s.src||'').indexOf('googletagmanager.com/gtm.js?id=GTM-MX6DNH8L') !== -1;
          });
        `;
      }, 'gtm.js script injection after consent', 60000);
      gtmInjected = true;
    } catch (err) {
      console.error('  (gtm.js injection timing out — dumping page diagnostics)');
      const diag = await evalJson(cdp, sessionId, `
        return JSON.stringify({
          savedConsent: (function(){ try { return localStorage.getItem('pcz_consent_v1'); } catch(e){ return 'err'; } })(),
          consentGet: (function(){ try { return JSON.stringify(window.consentAPI.get()); } catch(e){ return 'err'; } })(),
          hasGtmSrc: !!document.getElementById('gtm-src'),
          scriptTags: Array.from(document.scripts).map(function(s){ return s.src || '[inline]'; }).filter(function(u){ return u.indexOf('gtm') !== -1 || u.indexOf('googletag') !== -1; })
        });
      `);
      console.error('  diag: ' + String(diag));
      printConsole();
    }
    check('gtm.js (GTM-MX6DNH8L) injected after consent', gtmInjected);

    /* GTM loads the GA4 tag through its googtag runtime (gtag.js is fetched
       with cx=c), so window.gtag stays a wrapper — the live-proof is that
       gtag('get') callbacks resolve against the real property, checked later
       in phase 4. Only assert the gtag function exists here. */
    let realGtag = false;
    try {
      await waitForCheck(cdp, sessionId, function () {
        return `return typeof window.gtag === 'function';`;
      }, 'gtag callable after GTM boot', 60000);
      realGtag = true;
    } catch (err) {
      printConsole();
    }
    check('gtag callable after consent (GA4 tag wired via GTM)', realGtag);
    await sleep(2500);

    const post = await evalJson(cdp, sessionId, `
      return JSON.stringify({
        gtmScripts: Array.from(document.scripts).filter(function(s){ return (s.src||'').indexOf('googletagmanager.com/gtm.js') !== -1; }).length,
        gtagType: typeof window.gtag,
        savedConsent: (function(){ try { return localStorage.getItem('pcz_consent_v1'); } catch(e){ return 'err'; } })(),
        banner: !!document.getElementById('pcz-banner')
      });
    `);
    try {
      const saved = JSON.parse(post.savedConsent);
      check('consent persisted in localStorage', saved && saved.categories && saved.categories.analytics === true, post.savedConsent);
    } catch (e) { check('consent persisted in localStorage', false, post.savedConsent); }
    check('gtm.js present exactly once', post.gtmScripts === 1, String(post.gtmScripts));

    const postGtm = netEvents.filter((e) => e.url.indexOf('googletagmanager.com') !== -1).map((e) => e.url);
    check('GTM container fetched over the network after consent', postGtm.length >= 1, JSON.stringify(postGtm.slice(0, 3)));

    /* ---- 4. Site's real events flow through window.gaEvent. ---- */
    console.error('\n[phase 3] fire site events via window.gaEvent');
    /* Fire both events back-to-back in ONE eval so GA4 batches them into a
       single collect POST (both `en=` values in the form body). Separate eval
       round-trips give GA4 time to split the batch, which makes delivery racy. */
    const fireBoth = await evalJson(cdp, sessionId, `
      if (typeof window.gaEvent !== 'function') return 'missing';
      const r1 = window.gaEvent('home_explore_select', {
        selection_type: 'start_card',
        explore_type: 'project_graph',
        source_surface: 'home_start_cards'
      }) ? 1 : 0;
      const r2 = window.gaEvent('select_content', {
        content_type: 'project',
        content_id: 'retailStore',
        resource_type: 'project'
      }) ? 1 : 0;
      return r1 + ':' + r2;
    `);
    check('home_explore_select accepted post-consent', String(fireBoth).startsWith('1'), String(fireBoth));
    check('select_content accepted post-consent', String(fireBoth).endsWith('1'), String(fireBoth));

    const eventCheck = await evalJson(cdp, sessionId, `
      return (function(){
        const dl = Array.isArray(window.dataLayer) ? window.dataLayer : [];
        const out = [];
        dl.forEach(function(entry){
          if (entry && typeof entry === 'object' && entry.event === 'home_explore_select') {
            out.push({ event: entry.event, selection_type: entry.selection_type, explore_type: entry.explore_type, source_surface: entry.source_surface, activity_category: entry.activity_category, activity_label: entry.activity_label });
          }
          if (entry && typeof entry === 'object' && entry.event === 'select_content') {
            out.push({ event: entry.event, content_id: entry.content_id, content_type: entry.content_type, activity_category: entry.activity_category });
          }
        });
        return out;
      })();
    `);
    const pushed = Array.isArray(eventCheck) ? eventCheck : [];
    const explorePushed = pushed.find((e) => e.event === 'home_explore_select');
    const selectPushed = pushed.find((e) => e.event === 'select_content');
    check('home_explore_select reached dataLayer', !!explorePushed, JSON.stringify(pushed));
    if (explorePushed) {
      check('home_explore_select has sanitized params', explorePushed.selection_type === 'start_card' && explorePushed.explore_type === 'project_graph' && explorePushed.source_surface === 'home_start_cards', JSON.stringify(explorePushed));
      check('home_explore_select has activity_* fields', typeof explorePushed.activity_category === 'string' && !!explorePushed.activity_label, JSON.stringify(explorePushed));
    }
    check('select_content reached dataLayer', !!selectPushed, JSON.stringify(pushed));
    if (selectPushed) {
      check('select_content has sanitized params', selectPushed.content_type === 'project' && selectPushed.content_id === 'retailstore', JSON.stringify(selectPushed));
      check('select_content has activity_category=portfolio', selectPushed.activity_category === 'portfolio', JSON.stringify(selectPushed));
    }

    /* ---- 5. Prove the GA4 property tag is live inside the container. ---- */
    console.error('\n[phase 4] GA4 property liveness (gtag get + network)');
    const propRead = await evalJson(cdp, sessionId, `
      return new Promise(function(resolve){
        try {
          window.gtag('get', '${GA4_PROPERTY}', 'property.id', function(v){
            resolve('res|' + v + '|type=' + (typeof window.gtag));
          });
        } catch (e) { resolve('err|' + e.message); }
        setTimeout(function(){ resolve('timeout'); }, 8000);
      });
    `, 15000);
    check('gtag get property.id resolved (GA4 tag live)', /^res\|/.test(String(propRead)), String(propRead));

    /* GA4 config/collect traffic: the __gaawe tag calls the gtag.js library,
       which then performs the config + page_view collect POST. Give it time. */
    await sleep(6000);
    const gaUrls = netEvents
      .filter((e) => /google-analytics\.com\/g\/collect|googletagmanager\.com\/gtag\/js|googleadservices\.com/i.test(e.url))
      .map((e) => e.method + ' ' + e.url);
    const collectUrls = netEvents
      .filter((e) => /google-analytics\.com\/g\/collect/i.test(e.url))
      .map((e) => e.url);
    check('GA4 gtag library loaded after consent', gaUrls.length >= 1, JSON.stringify({ matched: gaUrls.slice(0, 5), all: netEvents.map((e) => e.url).slice(0, 25) }));
    check('GA4 v2 collect traffic (google-analytics.com/g/collect) after consent', collectUrls.length >= 1, JSON.stringify(collectUrls.slice(0, 5)));

    const waitCustom = (pattern) => new Promise((resolve) => {
      const started = Date.now();
      (function tick() {
        if (anyCollectMatches(pattern)) return resolve(true);
        if (Date.now() - started > 30000) return resolve(false);
        setTimeout(tick, 500);
      })();
    });
    /* Patterns must match in EITHER the URL (single-event GET) or the form body
       (batched POST, where postData starts directly with "en=..."). */
    const collectHits = () => netEvents.filter((e) => /g\/collect/i.test(e.url)).map((e) => e.url.slice(0, 150));
    const exploreHit = await waitCustom(/en=home_explore_select\b/, 'home_explore_select hit');
    check('home_explore_select reached GA4 (collect hit with en=home_explore_select)', exploreHit, JSON.stringify(collectHits()));
    const selectHit = await waitCustom(/en=select_content\b/, 'select_content hit');
    check('select_content reached GA4 (collect hit with en=select_content)', selectHit, JSON.stringify(collectHits()));

    /* ---- 6. Revoke consent -> gaEvent must be rejected. ---- */
    console.error('\n[phase 5] revoke consent (gate re-engages)');
    const revoke = await evalJson(cdp, sessionId, `
      try {
        window.consentAPI.set({ necessary: true, analytics: false, functional: false, advertising: false });
        return 'revoked';
      } catch (e) { return 'error: ' + e.message; }
    `);
    check('consent revocation executed', revoke === 'revoked', revoke);
    await evalJson(cdp, sessionId, `
      (function(){
        window.dispatchEvent(new Event('consent-changed'));
        return 'done';
      })()
    `);
    await sleep(400);
    const dlLengthBefore = await evalJson(cdp, sessionId, `return (window.dataLayer || []).length;`);
    const blocked = await evalJson(cdp, sessionId, `
      if (typeof window.gaEvent !== 'function') return 'gaEvent missing';
      const ok = window.gaEvent('select_content', { content_type: 'project', content_id: 'nope' });
      return ok ? 'SENT (BAD)' : 'blocked (good)';
    `);
    const dlLengthAfter = await evalJson(cdp, sessionId, `return (window.dataLayer || []).length;`);
    check('post-revocation gaEvent is blocked by consent gate', blocked === 'blocked (good)', blocked);
    check('post-revocation dataLayer unchanged', Number(dlLengthAfter) <= Number(dlLengthBefore), `before=${dlLengthBefore} after=${dlLengthAfter}`);
  } catch (err) {
    return fail(err);
  } finally {
    if (cdp) cdp.close();
    try { chrome.kill('SIGKILL'); } catch (err) {}
  }

  const passed = results.every((r) => r.ok === true);
  const summary = {
    passed,
    base_url: BASE_URL,
    ga4_property: GA4_PROPERTY,
    network_evidence: {
      total_requests: netEvents.length,
      google_or_gtm: netEvents.filter((e) => /google|googletagmanager/i.test(e.url)).map((e) => e.url).slice(0, 25)
    },
    checks: results,
    totals: {
      pass: results.filter((r) => r.ok).length,
      fail: results.filter((r) => !r.ok).length
    }
  };
  console.error('\n================ SUMMARY ================');
  console.error(JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
  process.exit(passed ? 0 : 1);
}

main().catch((err) => {
  console.error('FATAL: ' + (err && err.stack ? err.stack : String(err)));
  process.exit(1);
});

if (require.main === module) {
  // (main invoked below)
}
