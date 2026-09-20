/* Session-only recovery shared by tools and contact; never an account store. */
(() => {
  'use strict';
  if (window.SiteSessionDrafts) return;
  const PREFIX = 'ds:session-draft:v1:';
  const TTL = 2 * 60 * 60 * 1000;
  const MAX_ENTRY = 1024 * 1024;
  const MAX_TOTAL = 4 * MAX_ENTRY;
  const bytes = (text) => new TextEncoder().encode(text).byteLength;
  const remove = (key) => {
    try { window.sessionStorage.removeItem(PREFIX + key); } catch {}
  };
  const entries = () => {
    const storage = window.sessionStorage;
    const result = [];
    for (const key of Object.keys(storage)) {
      if (!key.startsWith(PREFIX)) continue;
      try {
        const text = storage.getItem(key);
        const record = JSON.parse(text);
        if (!Number.isFinite(record.updated) || Date.now() - record.updated >= TTL || record.updated > Date.now() || bytes(text) > MAX_ENTRY) {
          storage.removeItem(key);
        } else result.push({ key, updated: record.updated, bytes: bytes(text), data: record.data });
      } catch { try { storage.removeItem(key); } catch {} }
    }
    return result.sort((a, b) => a.updated - b.updated);
  };
  const read = (key) => {
    try { return entries().find((entry) => entry.key === PREFIX + key)?.data ?? null; } catch { return null; }
  };
  const write = (key, data) => {
    try {
      const storage = window.sessionStorage;
      const text = JSON.stringify({ updated: Date.now(), data });
      const size = bytes(text);
      // Never leave an older draft behind when the current input is too large.
      if (size > MAX_ENTRY) { remove(key); return false; }
      const all = entries().filter((entry) => entry.key !== PREFIX + key);
      let total = all.reduce((sum, entry) => sum + entry.bytes, 0) + size;
      while (total > MAX_TOTAL && all.length) {
        const oldest = all.shift();
        storage.removeItem(oldest.key);
        total -= oldest.bytes;
      }
      try { storage.setItem(PREFIX + key, text); } catch {
        // A smaller browser quota may apply. Evict only our drafts and retry.
        while (all.length) {
          storage.removeItem(all.shift().key);
          try { storage.setItem(PREFIX + key, text); return true; } catch {}
        }
        remove(key);
        return false;
      }
      return true;
    } catch { remove(key); return false; }
  };
  const removePrefix = (keyPrefix) => {
    try {
      for (const key of Object.keys(window.sessionStorage)) {
        if (key.startsWith(PREFIX + keyPrefix)) window.sessionStorage.removeItem(key);
      }
    } catch {}
  };
  const notice = ({ container = document.body, replace = null, onDiscard, duration = 12000, reserve = false } = {}) => {
    const slot = document.createElement('div');
    slot.className = 'draft-recovery-slot';
    const priorHidden = replace?.getAttribute('aria-hidden');
    if (replace?.parentElement) {
      slot.classList.add('draft-recovery-slot--replacement');
      replace.before(slot);
      slot.append(replace);
      replace.setAttribute('aria-hidden', 'true');
    } else container.prepend(slot);
    const element = document.createElement('div');
    element.className = 'draft-recovery-notice';
    const status = document.createElement('span');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    status.textContent = 'Draft restored';
    const discard = document.createElement('button');
    discard.type = 'button';
    discard.textContent = 'Discard';
    discard.addEventListener('click', () => { if (onDiscard?.() !== false) dismiss(); });
    element.append(status, discard);
    slot.append(element);
    let dismissed = false;
    // Do not remove a focused action from keyboard or screen-reader users.
    const timer = duration > 0 ? window.setTimeout(() => {
      if (!element.contains(document.activeElement)) dismiss();
      else element.addEventListener('focusout', dismiss, { once: true });
    }, duration) : 0;
    function release() {
      window.clearTimeout(timer);
      if (replace && slot.contains(replace)) {
        if (priorHidden === null) replace.removeAttribute('aria-hidden');
        else replace.setAttribute('aria-hidden', priorHidden);
        slot.before(replace);
      }
      slot.remove();
    }
    function dismiss() {
      if (dismissed) return;
      dismissed = true;
      window.clearTimeout(timer);
      if (reserve && !replace) {
        slot.style.minBlockSize = `${slot.getBoundingClientRect().height}px`;
        element.remove();
      } else release();
    }
    dismiss.release = release;
    return dismiss;
  };
  window.SiteSessionDrafts = Object.freeze({ read, write, remove, removePrefix, notice });
})();
