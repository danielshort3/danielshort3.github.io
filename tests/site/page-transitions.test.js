'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

module.exports = function runPageTransitionTests({ assert }) {
  const router = read('js/navigation/page-transitions.js');
  const transitionCss = read('css/components/page-transitions.css');
  const noJs = read('js/common/no-js.js');
  const serviceWorker = read('sw.js');
  const homeAccordion = read('js/home/category-accordion.js');
  const routeRuntime = read('js/navigation/site-route-runtime.js');
  const screenRecorder = read('js/tools/screen-recorder.js');

  assert(router.includes("const CONTENT_SELECTOR = '[data-site-route-content], [data-personal-detail-content]';") &&
    router.includes("const MANIFEST_SELECTOR = 'script#site-route-manifest[data-site-route-manifest]';") &&
    router.includes('manifest.version !== 1') &&
    router.includes("manifest.navigation !== 'soft'") &&
    router.includes('manifest.path !== normalizePathname(url.pathname)'),
  'persistent navigation should require a versioned soft-route manifest that matches the requested path');

  assert(router.includes("'/tools/background-remover'") &&
    router.includes("'/tools/transcribe'") &&
    router.includes("'/tools/job-application-tracker'") &&
    router.includes("path === '/job-application-copilot' || path === '/job-application-copilot/privacy'") &&
    router.includes("link.dataset.navigation === 'hard'") &&
    router.includes("/^\\/(?:professional|analytics|data-science|tourism)"),
  'security-bound tools, professional routes, and explicit hard links should remain document navigations');

  assert(router.includes("headers: { Accept: 'text/html', [REQUEST_HEADER]: '1' }") &&
    router.includes("new DOMParser().parseFromString(payload.html, 'text/html')") &&
    router.includes('responseUrl.origin !== window.location.origin') &&
    router.includes("!type.includes('text/html')"),
  'route requests should be same-origin HTML requests that are parsed and validated before commit');

  assert(router.includes('const ROUTE_CACHE_LIMIT = 12;') &&
    router.includes('while (routeCache.size > ROUTE_CACHE_LIMIT)') &&
    router.includes("document.addEventListener('pointerover', prefetchFromEvent") &&
    router.includes("document.addEventListener('focusin', prefetchFromEvent)") &&
    router.includes("document.addEventListener('pointerdown'"),
  'navigation should use a bounded LRU and prefetch from pointer, focus, and press intent');

  assert(router.includes('const PROGRESS_DELAY_MS = 350;') &&
    router.includes("outlet?.setAttribute('aria-busy', 'true')") &&
    router.includes("progress.dataset.siteRouteProgress = 'true'") &&
    router.includes("progress.hidden = false"),
  'slow route requests should expose busy state and a delayed in-panel progress line');

  const beforeLeaveIndex = router.indexOf('runtime.beforeLeave({');
  const unmountIndex = router.indexOf('runtime.unmount({', beforeLeaveIndex);
  const replaceIndex = router.indexOf('replaceRouteContent(route, metadataRoute.document)', unmountIndex);
  const mountIndex = router.indexOf('runtime.mount(routeKey', replaceIndex);
  const historyIndex = router.indexOf('pushRouteHistory(url, route.manifest, targetState);', mountIndex);
  assert(beforeLeaveIndex >= 0 && unmountIndex > beforeLeaveIndex && replaceIndex > unmountIndex &&
    mountIndex > replaceIndex && historyIndex > mountIndex,
  'navigation should veto first, unmount, replace the route scene, mount it, and only then commit history');

  assert(router.includes('provisionalHistory = beginProvisionalHistory(url)') &&
    router.includes('const targetState = window.history.state;') &&
    router.includes('restoreProvisionalHistory(provisionalHistory);') &&
    router.includes('pushRouteHistory(url, route.manifest, targetState);'),
  'legacy mounts should temporarily see the destination URL while final pushState remains post-mount');

  assert(router.includes('currentOutlet.replaceWith(nextOutlet);') &&
    !router.includes('currentOutlet.replaceChildren(') &&
    router.includes('const nextOutlet = document.importNode(route.outlet, true);') &&
    router.includes('syncSkipLink(route.document);'),
  'the router should replace the complete route scene so homepage and detail markup cannot nest incompatible shells');

  assert(router.includes("nextOutlet.classList.add('site-route-content--preparing')") &&
    router.includes("outlet.classList.remove('site-route-content--preparing')") &&
    router.includes("outlet.classList.add('site-route-content--leaving')") &&
    transitionCss.includes('[data-site-route-content].site-route-content--preparing') &&
    transitionCss.includes('[data-site-route-content].site-route-content--leaving'),
  'route content should remain in a defined hidden state while mounting and keep the outgoing fade through teardown');

  assert(router.includes('runtime.ensureLegacyRoute(route.manifest.id, { scripts });') &&
    router.includes('preloadScriptBytes(scripts, signal)') &&
    !router.includes('loadScriptsInOrder(scripts, route.document') &&
    router.includes('registeredRouteKey(route.manifest)'),
  'legacy scripts should preload without binding the old DOM and execute later through a scoped lifecycle adapter');

  assert(router.includes('site-(?:shell|consent|tools-account)') &&
    !router.includes('site-(?:shell|consent|tools-account|tools-landing)') &&
    router.includes('loadScriptsInOrder(persistentScripts, route.document, route.url, signal)') &&
    router.includes("Array.from(document.scripts || []).some((script) =>") &&
    router.includes('if (alreadyPresent) loadedScripts.add(normalized);') &&
    router.includes('loadedScripts.has(normalized)'),
  'persistent shared bundles should load once while route-specific scripts remain lifecycle-owned');

  assert(router.includes('prepareRouteStyles(route, controller.signal)') &&
    router.includes("link.media = 'not all'") &&
    router.includes('activatePreparedStyles(addedStyles)') &&
    router.includes('retireOldStyles(previousManifest, route.manifest)') &&
    router.includes("'link[rel=\"canonical\"]'") &&
    router.includes("meta[property^=\"og:\"]") &&
    router.includes('script[type="application/ld+json"]'),
  'soft navigation should prepare target styles and synchronize canonical, social, and structured metadata');

  assert(router.includes("window.addEventListener('popstate'") &&
    router.includes("currentManifest?.id === 'home' && getHomepageHistoryIntent(event.state)") &&
    router.includes("const documentUrl = homepageHistoryIntent ? resolveUrl('/', url.href) : url;") &&
    router.includes('const metadataPromise = homepageHistoryIntent') &&
    router.includes('syncHead(metadataDocument, route.document)') &&
    router.includes("String(state.siteRoute?.id || '').trim() !== 'home'") &&
    router.includes('delete capturedState.homePanel;') &&
    router.includes('delete capturedState.homeView;') &&
    router.includes('fetchRouteDocument(documentUrl, { signal: controller.signal })') &&
    router.includes('restoreVetoedPop(options.historyState)') &&
    router.includes('window.history.go(delta)') &&
    router.includes("window.history.scrollRestoration = 'manual'") &&
    router.includes('captureScroll()') &&
    router.includes('restoreScroll(') &&
    router.includes("heading.setAttribute('tabindex', '-1')"),
  'history traversal should preserve route-owned scroll, keep homepage state out of detail entries, and restore accessible focus');

  assert(router.includes('const targetKey = `${normalizeRouteUrl(url)}${url.hash}`;') &&
    router.includes("if (navigationType === 'pop') restoreVetoedPop(options.historyState);") &&
    router.includes("if (manifest.id !== 'home')"),
  'rapid hash navigation and failed offline history traversal should preserve the latest URL/content transaction');

  assert(router.includes('[data-home-timeline-scroller]') &&
    router.includes('style?.overflowX') &&
    router.includes("'data-performance-tier'") &&
    router.includes("'data-qrtool-ui-mode'") &&
    router.includes("window.matchMedia?.('(max-width: 959px), (max-height: 619px)')?.matches ? 4 : 6"),
  'route reconciliation should restore both-axis timeline scroll, clear runtime body state, and cap compact motion at four pixels');

  assert(routeRuntime.includes('currentRecord?.id === id && currentRecord.lifecycle?.[LEGACY_LIFECYCLE]') &&
    routeRuntime.includes('const registeredScopes = new WeakSet();') &&
    routeRuntime.includes('scope.fetchControllers.add(controller);') &&
    routeRuntime.includes('result === false || event.defaultPrevented') &&
    screenRecorder.includes("window.addEventListener('beforeunload'") &&
    screenRecorder.includes('window.SiteRoutes?.addCleanup?.(() => {'),
  'route lifecycles should promote explicit modules, bound timer/fetch bookkeeping, and protect and tear down active recording');

  assert(router.includes("dispatch(CONTENT_EVENT, detail)") &&
    router.includes("dispatch(ROUTE_EVENT, detail)") &&
    router.includes("event: 'virtual_page_view'") &&
    router.includes("if (!consent?.analytics) return;"),
  'each successful commit should emit canonical content/route events and one consent-aware virtual pageview');

  assert(router.includes("document.querySelector('[data-site-route-announcer], [data-site-route-status]')") &&
    router.includes("status.dataset.siteRouteAnnouncer = 'true'"),
  'navigation announcements should reuse the universal shell live region');

  assert(router.includes('activeNavigation.controller.abort(makeAbortError())') &&
    router.includes('sequence !== navigationSequence') &&
    router.includes('options.retryAbortedPending !== false') &&
    router.includes('retryAbortedPending: false') &&
    router.includes('navigator.onLine !== false') &&
    router.includes("retry.textContent = 'Retry'"),
  'new navigation should supersede stale work and offline failures should retain the current route with Retry');

  assert(!router.includes('sitePageTransitionFocus') &&
    !router.includes('storePendingNavigation') &&
    !router.includes('supportsCrossDocumentViewTransitions') &&
    !router.includes('startViewTransition') &&
    !router.includes('site-page-transition-out'),
  'the router should not retain delayed assignment, cross-document snapshots, or stored veil handoffs');

  assert(!transitionCss.includes('@view-transition') &&
    !transitionCss.includes('::view-transition') &&
    !transitionCss.includes('personal-veil') &&
    !transitionCss.includes('.personal-accordion__panel::after') &&
    transitionCss.includes('--site-route-enter-duration: 160ms;') &&
    transitionCss.includes('--site-route-motion: 4px;'),
  'transition CSS should animate live content only, with no screen veil or browser snapshot layer');

  assert(transitionCss.includes('[data-site-route-progress]') &&
    transitionCss.includes('block-size: 2px;') &&
    transitionCss.includes('@media (prefers-reduced-motion: reduce)') &&
    transitionCss.includes('animation: none !important;'),
  'the route progress line should not shift layout and reduced motion should disable all route animation');

  assert(!homeAccordion.includes('document.startViewTransition') &&
    homeAccordion.includes("root.classList.add('is-view-leaving')") &&
    homeAccordion.includes("root.classList.add('is-view-entering')"),
  'homepage expansion should use its live content phases instead of a full-page snapshot transition');

  assert(noJs.includes("root.classList.add('js')") &&
    noJs.includes("window.sessionStorage.removeItem('sitePageTransition')") &&
    !noJs.includes('__sitePageRevealState') &&
    !noJs.includes("addEventListener?.('pagereveal'"),
  'the parser bootstrap should clear obsolete veil state without arming an arrival cover');

  assert(serviceWorker.includes("const VERSION = 'ds-v2';") &&
    serviceWorker.includes("request.headers.get('X-Site-Route') === '1'") &&
    serviceWorker.includes('event.respondWith(networkFirstDocument(request));') &&
    serviceWorker.includes('const exact = await caches.match(request);') &&
    serviceWorker.includes("normalizePathname(url.pathname) === '/'") &&
    serviceWorker.includes('HARD_DOCUMENT_PATHS.has(normalizePathname(path))'),
  'the service worker should use exact network-first route responses and bypass security-bound documents');
};
