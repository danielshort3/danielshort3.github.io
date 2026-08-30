const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));
const count = (value, pattern) => (String(value || '').match(pattern) || []).length;

module.exports = function runHomeCategoryAccordionTests({ assert }) {
  const personal = readJson('content/audiences/personal.json');
  const section = personal.page.sections.find((entry) => entry.type === 'home-accordion');
  const categories = section?.props?.categories || [];
  const ids = categories.map((category) => category.id);
  const html = read('index.html');
  const css = read('css/components/home-category-accordion.css');
  const js = read('js/home/category-accordion.js');
  const homeEntry = read('build/entries/site-home.entry.js');
  const homeStyles = read('css/styles-home.css');
  const navigation = read('js/navigation/navigation.js');
  const activityEvents = read('js/analytics/activity-events.js');

  assert(section && section.variant === 'shallow-wedge',
    'personal homepage should use the selected shallow-wedge accordion variant');
  assert(JSON.stringify(ids) === JSON.stringify(['about', 'projects', 'tools', 'games', 'contact']),
    'homepage categories should stay in the approved About, Projects, Tools, Games, Contact order');
  assert(section.props.defaultPanel === 'tools',
    'homepage should open the Tools panel shown in the accepted concept');
  assert(
    JSON.stringify(categories.map((category) => category.color)) === JSON.stringify([
      '#091f3b', '#155dfc', '#087f8c', '#c94b0a', '#334155'
    ]),
    'homepage rails should use the approved site-native category colors'
  );

  assert(count(html, /data-home-accordion-item=/g) === 5 &&
    count(html, /data-home-accordion-trigger=/g) === 5 &&
    count(html, /data-home-accordion-panel=/g) === 5,
  'generated homepage should render one static item, trigger, and attached panel per category');
  assert(count(html, /aria-expanded="true"/g) === 1 &&
    count(html, /aria-expanded="false"/g) >= 4 &&
    count(html, /data-home-accordion-panel="[^"]+" hidden inert/g) === 4,
  'homepage should author exactly one expanded panel and remove inactive panel content from interaction');
  assert(count(html, /<h1\b/g) === 1 && html.includes('id="home-accordion-title"'),
    'homepage should expose one accessible H1');
  assert(html.includes('<ul class="home-accordion__cards">') &&
    html.includes('<li class="home-accordion__card-item">') &&
    /<a class="home-accordion__card" href="\/tools\/text-compare"/.test(html) &&
    !/<a class="home-accordion__card" role="listitem"/.test(html),
  'homepage cards should use semantic list wrappers without masking native link roles');

  ids.forEach((id) => {
    const pairPattern = new RegExp(
      `<button[^>]+id="home-accordion-trigger-${id}"[^>]+type="button"[^>]+aria-controls="home-accordion-panel-${id}"[\\s\\S]*?<\\/button>\\s*<section[^>]+id="home-accordion-panel-${id}"[^>]+role="region"[^>]+aria-labelledby="home-accordion-trigger-${id}"`
    );
    assert(pairPattern.test(html), `${id} rail should be a native button immediately followed by its labeled region`);
  });

  const requiredRoutes = [
    '/portfolio/handwritingRating',
    '/portfolio/babynames',
    '/portfolio/sheetMusicUpscale',
    '/portfolio/ufoDashboard',
    '/tools/text-compare',
    '/tools/image-optimizer',
    '/tools/qr-code-generator',
    '/tools/screen-recorder',
    '/tools/word-frequency',
    '/games/project-starfall',
    '/games/stormbreak',
    '/games/stellar-dogfight',
    '/games/probability-engine',
    '/games/roulette',
    '/games/ocean-wave-simulation',
    '/contact#contact-modal',
    'mailto:daniel@danielshort.me'
  ];
  requiredRoutes.forEach((route) => {
    assert(html.includes(`href="${route}"`), `generated homepage missing approved route ${route}`);
  });
  ['short-links', 'ga4-utm-performance', 'job-application-tracker', 'transcribe'].forEach((toolId) => {
    assert(!JSON.stringify(categories).includes(`\"id\":\"${toolId}\"`),
      `homepage should not expose hidden tool ${toolId}`);
  });
  assert(!/recruiter|available for hire|resume|case study|kpi|professional analytics profile/i.test(JSON.stringify(categories)),
    'personal homepage copy should avoid job-seeking and dashboard framing');

  assert(!html.includes('data-home-graph') &&
    !html.includes('data-graph-') &&
    !html.includes('home-graph__inspector') &&
    !html.includes('home-graph__mobile-deck'),
  'generated homepage should not retain the detached graph, inspector, or duplicate mobile deck');
  assert(!fs.existsSync(path.join(ROOT, 'js/home/project-graph.js')) &&
    !fs.existsSync(path.join(ROOT, 'css/components/home-project-graph.css')),
  'retired graph source files should be removed');

  assert(homeEntry.includes("import '../../js/home/category-accordion.js';") &&
    homeStyles.includes('@import url("components/home-category-accordion.css");'),
  'home-only bundles should load the category accordion sources');
  assert(personal.page.bottomScripts.some((script) => script.src === 'dist/site-home.js') &&
    !JSON.stringify(personal.page.bottomScripts).includes('project-graph'),
  'managed homepage source should use the stable home bundle without raw graph scripts');

  assert(css.includes('--home-rail-width: 76px;') &&
    css.includes('--home-active-rail-width: 82px;') &&
    css.includes('gap: 0;') &&
    css.includes('border: 3px solid var(--panel-color);') &&
    css.includes('border-left: 0;') &&
    css.includes('right: -15px;') &&
    css.includes('clip-path: polygon(0 0, 100% 50%, 0 100%);'),
  'desktop accordion should use thin touching rails, a subtly wider active rail, wider color frame, and right-facing word notch');
  assert(css.includes('background: #ffffff;') &&
    css.includes('border-radius: 0 0 12px 0;') &&
    css.includes('overflow-y: scroll;') &&
    css.includes('overscroll-behavior: contain;') &&
    css.includes('position: sticky;') &&
    css.includes('scrollbar-color:'),
  'selected panel should stay white with square flush top corners, a sticky heading, and visible contained desktop scroll');
  assert(css.includes('@media (max-width: 768px)') &&
    css.includes('display: block;') &&
    css.includes('max-height: none;') &&
    css.includes('overflow: visible;') &&
    css.includes('clip-path: polygon(0 0, 100% 0, 50% 100%);'),
  'mobile accordion should stack full-width bars, use document scrolling, and point the active notch down');
  assert(css.includes('@media (pointer: coarse)') &&
    css.includes('min-height: 44px;') &&
    css.includes('@media (prefers-reduced-motion: reduce)') &&
    css.includes('.home-accordion__rail:focus-visible'),
  'accordion should preserve touch targets, reduced motion, and visible keyboard focus');
  assert(!/prefers-color-scheme\s*:\s*dark/i.test(css) &&
    !/color-scheme\s*:\s*dark/i.test(css) &&
    !/data-theme/i.test(css),
  'homepage accordion should remain intentionally light-only');

  assert(js.includes("panel.hidden = !selected;") &&
    js.includes("panel.setAttribute('inert', '')") &&
    js.includes("panel.removeAttribute('inert')") &&
    js.includes('if (!ids.includes(id) || id === activeId) return false;'),
  'accordion controller should enforce one expanded panel and keep a repeated active click open');
  assert(js.includes("event.key === 'ArrowDown'") &&
    js.includes("event.key === 'ArrowUp'") &&
    js.includes("event.key === 'Home'") &&
    js.includes("event.key === 'End'") &&
    js.includes("event.key === 'Enter'") &&
    js.includes("event.key === ' '"),
  'accordion rails should support directional movement plus Enter and Space activation');
  assert(js.includes('scrollPositions') &&
    js.includes('scroller.scrollTop') &&
    js.includes("window.matchMedia('(min-width: 769px)')") &&
    js.includes('scrollIntoView'),
  'accordion should preserve desktop panel scroll positions and reveal newly opened mobile sections');

  assert(navigation.includes("if (document.body.dataset.page === 'home') return;") &&
    css.includes('.mobile-site-dock {') &&
    css.includes('display: none !important;'),
  'selected homepage should prevent the shared bottom dock and retain a CSS safety fallback');
  assert(activityEvents.includes("target.closest('[data-home-accordion]')") &&
    activityEvents.includes('category.dataset.homeAccordionTrigger'),
  'homepage category changes should remain analytics-visible');
};
