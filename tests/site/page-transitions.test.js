'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const ROOT = path.resolve(__dirname, '..', '..');
const read = (name) => fs.readFileSync(path.join(ROOT, name), 'utf8');

module.exports = function runPageTransitionTests({ assert }) {
  const router = read('js/navigation/page-transitions.js');
  const frame = read('js/navigation/site-frame.js');
  const css = read('css/components/site-frame.css');
  const transitionCss = read('css/components/page-transitions.css');
  const home = read('js/home/category-accordion.js');
  const section = (name, next) => router.slice(router.indexOf('  function ' + name + '('), router.indexOf('  function ' + next + '('));
  const sandbox = {
    URL,
    window: { location: { origin: 'https://example.test' } },
    HARD_BOUNDARY_PATHS: new Set(['/tools/background-remover', '/tools/transcribe', '/tools/job-application-tracker']),
    PROFESSIONAL_AUDIENCES: new Set(['analytics', 'data-science', 'tourism']),
    ROUTE_CATEGORIES: new Set(['about', 'projects', 'tools', 'games', 'resume', 'contact'])
  };
  vm.createContext(sandbox);
  vm.runInContext([
    section('normalizePathname', 'normalizeRouteUrl'),
    section('isHardBoundary', 'isProfessionalAudienceUrl'),
    section('isProfessionalAudienceUrl', 'getPersonalRouteIntent'),
    section('getPersonalRouteIntent', 'getHomepageHistoryIntent'),
    section('isDocumentLikeUrl', 'hasBlockingInteractionLayer')
  ].join('\n'), sandbox);
  const intent = (value) => sandbox.getPersonalRouteIntent(new URL(value, 'https://example.test'));
  for (const name of ['background-remover', 'transcribe', 'job-application-tracker']) {
    assert(intent('/tools/' + name) === null && intent('/tools/' + name + '.html') === null,
      name + ' must preserve its document security boundary');
  }
  for (const url of ['/analytics', '/data-science', '/tourism', '/professional/analytics/contact', '/contact?audience=analytics', '/resume-analytics']) {
    assert(intent(url) !== null, url + ' should be eligible for validated shared-frame navigation');
  }
  assert(intent('/#games').category === 'games' && intent('/#unknown') === null,
    'homepage categories must remain explicit and unknown hashes must stay native');
  assert(intent('/tools').view === 'library' && intent('/tools/text-compare').view === 'detail',
    'library and detail links should carry their geometry intent before resources arrive');
  assert(!sandbox.isDocumentLikeUrl(new URL('https://external.test/tools')) &&
    !sandbox.isDocumentLikeUrl(new URL('https://example.test/documents/resume.pdf')),
    'external destinations and downloads must remain browser navigations');
  assert(router.includes('manifest.version !== 1') && router.includes("manifest.navigation !== 'soft'") &&
    router.includes('manifest.path !== normalizePathname(url.pathname)'),
    'a fetched page must validate its version, opt-in and exact route before changing content');
  assert(router.includes("new DOMParser().parseFromString(payload.html, 'text/html')") &&
    router.includes('responseUrl.origin !== window.location.origin'),
    'route loading must validate and parse same-origin HTML');
  assert(router.includes('while (preparedRoutes.size > ROUTE_CACHE_LIMIT)') &&
    router.includes('while (routeCache.size > ROUTE_CACHE_LIMIT)') && router.includes('scriptBytes.size > 64'),
    'HTML, prepared routes and script-byte preparation must use bounded caches');
  assert(router.includes('return prepareRoute(url, { prefetch: true })') &&
    router.includes('Promise.all([prepareRouteStyles(route), prepareRouteScripts(route, undefined, options.cache)])'),
    'intent prefetch must prepare route assets as well as the document');
  assert(router.includes('runtime.beforeLeave({') && router.includes('await frame.wipe(false') &&
    router.indexOf('runtime.beforeLeave({') < router.indexOf('await frame.wipe(false'),
    'leave guards must run before the departing content is wiped');
  assert(router.includes('if (!ready) frame.setLoading(true)') && router.includes('await frame.wipe(true'),
    'a prepared destination should reveal immediately while delayed destinations receive loading feedback');
  assert(router.includes('return window.SiteFrame.commit(') && !router.includes('currentOutlet.replaceWith('),
    'route navigation may replace the inner body but never the shared frame');
  for (const hook of ['data-site-persistent-shell', 'data-site-frame-stage', 'data-site-route-panel',
    'data-site-route-toolbar', 'data-site-frame-slot', 'data-site-frame-viewport', 'data-site-frame-loading']) {
    assert(frame.includes(hook), 'the persistent frame must expose ' + hook);
  }
  assert(frame.includes('tabs.get(category)') && frame.includes('tabs.set(category, link)') &&
    frame.includes('viewport.replaceChildren(body)'),
    'category nodes must be keyed and reused while the body is the replacement boundary');
  assert(!frame.includes('opacity:') && !frame.includes('scale(') &&
    frame.includes('before.frame.radius') && frame.includes('before.frame.width'),
    'frame movement must preserve opacity and text proportions while animating the outer border geometry');
  assert(frame.includes("position: 'absolute', gridArea: 'auto'") && frame.includes("stage.style.marginInline = '0'") &&
    frame.includes('before.children?.get(id)'),
    'animated boxes must use stage coordinates without grid offsets or recentering, and tab labels must travel with them');
  assert(css.includes('--site-frame-geometry-duration: 320ms') && css.includes('--site-frame-wipe-duration: 160ms') &&
    frame.includes('clipPath: last') && !transitionCss.includes('opacity: 0'),
    'geometry and content wipes must use the approved timing without whole-scene fades');
  assert(frame.includes('geometry?.finish(false)') && frame.includes('getComputedStyle(viewport).clipPath') &&
    frame.includes('sequence !== localSequence'),
    'retargeting must use the current geometry and wipe and ignore obsolete local completions');
  assert(router.includes('preceding?.controller.abort') && router.includes('sequence !== navigationSequence') &&
    router.includes('frame.restore(savedFrame,') && router.includes("retry.textContent = 'Retry'"),
    'stale navigations must cancel and recoverable failures must restore the previous frame with Retry');
  assert(router.includes('const baseline = committedSnapshot') && router.includes('savedFrame = baseline?.frame') &&
    router.includes('active.root !== savedFrame.body || active.signal?.aborted') &&
    router.includes('rememberCommittedRoute()') && !router.includes('savedDocument = document.cloneNode(true)'),
    'a failed navigation after an interrupted mount must restore the last committed body and controller, without cloning page output');
  assert(frame.includes('(max-width: 959px), (max-height: 619px)') &&
    css.includes('overflow-wrap: anywhere') && css.includes('@media (prefers-reduced-motion: reduce)'),
    'the shared frame must handle narrow, short and enlarged-text layouts and reduced motion');
  assert(router.includes('registeredRouteKey(route.manifest)') && router.includes('runtime.ensureLegacyRoute') &&
    router.includes('preloadScriptBytes(scripts, signal, cache)'),
    'existing module registration and scoped legacy script ownership must be retained');
  assert(router.includes('meta[name="referrer"]') && router.includes('syncHead(metadataDocument, route.document)'),
    'route commits must synchronize referrer policy and the existing document metadata');
  assert(router.includes('pushRouteHistory(displayUrl') && router.includes('requestUrl: documentUrl.href') &&
    router.includes('sendVirtualPageview(displayUrl)'),
    'search mounts may sanitize the URL without the router publishing its query again');
  assert(router.includes('captureScroll()') && router.includes('restoreVetoedPop(options.historyState)') &&
    router.includes("heading.setAttribute('tabindex', '-1')"),
    'history, leave-veto rollback, scrolling and accessible focus remain route-owned');
  assert(home.includes('frame.showHome(category, view') && home.includes('window.SiteRoutes?.addCleanup'),
    'homepage interaction must share the persistent frame and clean up its controller');
};
