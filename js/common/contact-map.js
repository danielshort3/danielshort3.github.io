/* Keep the Contact map's browsing context alive for the lifetime of the site shell. */
(() => {
  'use strict';
  if (window.ContactMap) return;

  const maps = new Map();
  let scheduled = 0;
  let observedViewport;
  let observedSlot;
  const resizeObserver = new ResizeObserver(() => refresh());
  const frameObserver = new MutationObserver(() => refresh());

  function hide(except) {
    maps.forEach(({ host }) => {
      if (host === except) return;
      delete host.dataset.mapActive;
      host.inert = true;
    });
  }

  function sync() {
    scheduled = 0;
    const site = window.SiteFrame;
    const state = site?.current();
    const viewport = site?.viewport();
    if (viewport && viewport !== observedViewport) {
      frameObserver.disconnect();
      frameObserver.observe(viewport, { childList: true, attributes: true, attributeFilter: ['inert', 'style'] });
      frameObserver.observe(site.root(), { attributes: true, attributeFilter: ['class', 'data-frame-category', 'data-frame-view'] });
      observedViewport = viewport;
    }
    const slot = state?.body?.isConnected && state.category === 'contact' &&
      (!state.home || state.view === 'overview')
      ? state.body.querySelector('[data-contact-map-slot]') : null;
    // The viewport becomes inert during its closing wipe. Keep the loaded map
    // painted under that same clip, and throughout the incoming geometry.
    if (!slot || slot.closest('[hidden], [aria-hidden="true"]')) {
      hide();
      return;
    }

    const pending = slot.querySelector('iframe[data-home-contact-map-src]');
    const src = pending?.getAttribute('data-home-contact-map-src') || slot.dataset.contactMapSource;
    if (!src) {
      hide();
      return;
    }
    let map = maps.get(src);
    if (!map) {
      if (!pending) return;
      const host = document.createElement('div');
      host.className = 'contact-map-persistent';
      host.setAttribute('data-persistent-contact-map', '');
      host.inert = true;
      pending.removeAttribute('data-home-contact-map-src');
      // Move the still-blank iframe once, before assigning its URL. Moving a
      // loaded iframe (even between connected parents) would reload the map.
      host.append(pending);
      viewport.append(host);
      map = { host, iframe: pending, loaded: false };
      maps.set(src, map);
    } else {
      pending?.remove();
    }
    slot.dataset.contactMapSource = src;
    if (observedSlot !== slot) {
      resizeObserver.disconnect();
      resizeObserver.observe(slot);
      resizeObserver.observe(viewport);
      observedSlot = slot;
    }

    const bounds = slot.getBoundingClientRect();
    const owner = viewport.getBoundingClientRect();
    hide(map.host);
    Object.assign(map.host.style, {
      left: `${bounds.left - owner.left + viewport.scrollLeft + slot.clientLeft}px`,
      top: `${bounds.top - owner.top + viewport.scrollTop + slot.clientTop}px`,
      width: `${slot.clientWidth}px`,
      height: `${slot.clientHeight}px`
    });
    map.host.dataset.mapActive = 'true';
    map.host.inert = viewport.inert;
    if (!map.loaded) {
      map.loaded = true;
      map.iframe.setAttribute('src', src);
    }
  }

  function refresh() {
    if (!scheduled) scheduled = window.requestAnimationFrame(sync);
  }

  window.ContactMap = Object.freeze({ refresh, hide });
  ['home:category-change', 'site:route-mounted', 'site:route-change', 'navheightchange'].forEach((event) => {
    document.addEventListener(event, refresh);
  });
  document.addEventListener('site:route-unmounted', () => hide());
  window.addEventListener('resize', refresh, { passive: true });
  window.visualViewport?.addEventListener('resize', refresh, { passive: true });
  document.fonts?.ready.then(refresh);
  refresh();
})();
