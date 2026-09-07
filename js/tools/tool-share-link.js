/* ===================================================================
   File: js/tools/tool-share-link.js
   Purpose: Shareable tool state — encode form inputs into ?s= and
            restore them after tool scripts create their controls.
   =================================================================== */
(() => {
  'use strict';

  const STORAGE_KEY = 'pcz_tool_s';
  const MAX_STATE_BYTES = 18 * 1024;

  function base64UrlEncode(str) {
    const bytes = new Uint8Array(new TextEncoder().encode(str));
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function base64UrlDecode(s) {
    s = String(s || '').replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) s += '=';
    const bin = atob(s);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }

  function collectState(scope) {
    const state = { v: 1, t: Date.now() };
    const controls = scope.querySelectorAll(
      'input:not([type="file"]):not([type="password"]):not([type="hidden"]):not([type="submit"]):not([type="button"])' +
      ', select, textarea'
    );
    controls.forEach((el) => {
      const id = el.id || el.name;
      if (!id) return;
      const type = String(el.type || '').toLowerCase();
      if (type === 'radio') {
        const group = el.name || id;
        if (el.checked) state[group] = el.value;
      } else if (type === 'checkbox') {
        const key = el.name || el.id;
        const group = Array.isArray(state[key]) ? state[key] : [];
        if (el.checked) group.push(el.value || el.id);
        state[key] = group;
      } else {
        state[id] = el.value;
      }
    });
    const keys = Object.keys(state);
    let total = 2;
    keys.forEach((k) => { total += k.length + String(state[k]).length; });
    return { state, tooBig: total > MAX_STATE_BYTES };
  }

  function buildShareUrl(scope, state) {
    const payload = base64UrlEncode(JSON.stringify(state));
    const url = new URL(window.location.href);
    url.search = '';
    url.hash = '';
    url.searchParams.set('s', payload);
    return url.toString();
  }

  function restore(scope, params) {
    const token = params.get('s');
    if (!token) return false;
    let state;
    try { state = JSON.parse(base64UrlDecode(token)); }
    catch (e) { return false; }
    if (!state || typeof state !== 'object' || state.v !== 1) return false;

    const getStateKey = (el) => {
      const grouped = el.type === 'checkbox' || el.type === 'radio';
      return grouped && el.name && el.name in state ? el.name : el.id || el.name;
    };
    scope.querySelectorAll('input, select, textarea').forEach((el) => {
      const key = getStateKey(el);
      if (!key || !(key in state)) return;
      const val = state[key];
      if (val == null) return;
      const type = String(el.type || '').toLowerCase();
      if (type === 'radio') {
        el.checked = (el.value === val) || (el.name === val);
      } else if (type === 'checkbox') {
        const list = Array.isArray(val) ? val : String(val).split(',').map((s) => s.trim()).filter(Boolean);
        const byValue = el.value && list.indexOf(el.value) !== -1;
        const byId = el.id && list.indexOf(el.id) !== -1;
        el.checked = byValue || byId;
      } else if (el.tagName === 'SELECT') {
        el.value = String(val);
        if (el.value !== String(val)) {
          Array.from(el.options).forEach((opt) => {
            if (opt.value === String(val) || opt.text === String(val)) el.value = opt.value;
          });
        }
      } else {
        el.value = String(val);
      }
    });

    scope.querySelectorAll('input, select, textarea').forEach((el) => {
      const key = getStateKey(el);
      if (key && state[key] != null) {
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    try { history.replaceState({}, '', window.location.pathname + window.location.search.replace(/[?&]s=[^&]*/, '').replace(/[?&]$/, '') + window.location.hash); }
    catch (e) {}
    return true;
  }

  function showCopyButton(scope) {
    const existing = scope.querySelector('[data-tool-share-link], [data-tools-share="copy"]');
    if (existing?.dataset.toolsShareBound === 'true') return;
    const hasInputs = scope.querySelector(
      'input:not([type="file"]):not([type="password"]):not([type="hidden"]):not([type="submit"]):not([type="button"]), select, textarea'
    );
    if (!hasInputs) return;
    const primaryBtn = existing ? null : scope.querySelector('button[type="submit"], .btn-primary, .btn-secondary');
    const anchor = primaryBtn || scope;
    if (!existing && (!anchor || !anchor.parentNode)) return;
    const btn = existing || document.createElement('button');
    if (!existing) {
      btn.type = 'button';
      btn.className = 'btn-ghost tools-share-copy';
      btn.textContent = 'Copy link to my inputs';
      btn.setAttribute('aria-label', 'Copy a link that restores these inputs');
    }
    btn.dataset.toolsShare = 'copy';
    btn.dataset.toolsShareBound = 'true';
    const idleLabel = btn.textContent;
    window.SiteRoutes?.addCleanup?.(() => { delete btn.dataset.toolsShareBound; });
    btn.addEventListener('click', () => {
      const { state, tooBig } = collectState(scope);
      if (tooBig) {
        btn.textContent = 'Input too large to share';
        btn.disabled = true;
        return;
      }
      const url = buildShareUrl(scope, state);
      const flash = () => {
        btn.textContent = 'Link copied \u2014 ready to paste';
        setTimeout(() => {
          btn.textContent = idleLabel;
          btn.disabled = false;
        }, 2500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(flash).catch(() => fallbackCopy(url, flash));
      } else {
        fallbackCopy(url, flash);
      }
    });
    if (!existing) {
      anchor.insertAdjacentElement ? anchor.insertAdjacentElement('afterend', btn) : anchor.insertAdjacentAfterElement(1, btn);
    }
  }

  function fallbackCopy(url, done) {
    const ta = document.createElement('textarea');
    ta.value = url;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    if (ok) done();
    else {
      // last resort: open in a new tab
      window.open(url, '_blank', 'noopener');
    }
  }

  function main() {
    // Only run on tool pages (have a tools-account-dock).
    if (!document.querySelector('[data-tools-account="dock"]')) return;
    const scope = document.querySelector('main') || document;
    const params = new URLSearchParams(window.location.search);
    restore(scope, params);
    showCopyButton(scope);
  }

  if (document.currentScript?.dataset?.siteRouteOwnedScript) {
    // Soft routes load scripts in sequence; format inputs may not exist yet.
    document.addEventListener('site:route-mounted', main, { once: true });
  } else if (document.readyState !== 'complete') {
    // The route runtime replays document readiness early. The native event
    // bubbles to window only after all deferred tool scripts have finished.
    window.addEventListener('DOMContentLoaded', main, { once: true });
  } else {
    main();
  }
})();
