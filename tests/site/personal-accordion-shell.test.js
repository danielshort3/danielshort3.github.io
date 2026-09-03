'use strict';

const fs = require('fs');
const path = require('path');
const {
  PERSONAL_CONTENT_END,
  PERSONAL_SHELL_END,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
} = require('../../build/lib/personal-accordion-shell');
const {
  GAME_PAGE_PATHS,
  INTERNAL_TOOL_PAGE_IDS,
  TOOL_PAGE_IDS,
  UTILITY_PAGE_CONFIGS,
  buildLibraryPage
} = require('../../build/generate-personal-accordion-pages');

const ROOT = path.resolve(__dirname, '..', '..');

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

function readJson(relativePath) {
  return JSON.parse(read(relativePath));
}

function count(source, pattern) {
  return (String(source || '').match(pattern) || []).length;
}

function getTagAttribute(source, tagPattern, attribute) {
  const tag = tagPattern.exec(source)?.[0] || '';
  return new RegExp(`\\s${attribute}="([^"]*)"`, 'i').exec(tag)?.[1] || '';
}

function getSkipLinkHref(source) {
  const tag = (String(source || '').match(/<a\b[^>]*>/gi) || [])
    .find((candidate) => /\sclass="[^"]*\bskip-link\b[^"]*"/i.test(candidate));
  return tag ? getTagAttribute(tag, /<a\b[^>]*>/i, 'href') : '';
}

function walkHtml(relativeDir) {
  const start = path.join(ROOT, relativeDir);
  if (!fs.existsSync(start)) return [];
  const files = [];
  const stack = [start];
  while (stack.length) {
    const current = stack.pop();
    fs.readdirSync(current, { withFileTypes: true }).forEach((entry) => {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(full);
      if (entry.isFile() && entry.name.endsWith('.html')) files.push(full);
    });
  }
  return files.sort();
}

function runPersonalAccordionShellTests({ assert }) {
  const sample = [
    '<!doctype html>',
    '<html><head><base href="/"><link rel="canonical" href="https://www.danielshort.me/contact"></head>',
    '<body class="contact-page" data-page="contact">',
    '<a href="#main" class="skip-link">Skip to main content</a>',
    '<header>Header</header>',
    '<main id="main"><h1>Contact</h1></main>',
    '<script src="first.js"></script><script src="second.js"></script>',
    '</body></html>'
  ].join('');
  const wrapped = wrapPersonalAccordionHtml(sample, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail',
    backHref: '/#contact',
    backLabel: 'Back to categories',
    backCompactLabel: 'Categories',
    backAriaLabel: 'Back to categories'
  });
  const wrappedAgain = wrapPersonalAccordionHtml(wrapped, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail',
    backHref: '/#contact',
    backLabel: 'Back to categories',
    backCompactLabel: 'Categories',
    backAriaLabel: 'Back to categories'
  });
  assert(wrappedAgain === wrapped, 'Personal shell wrapping should be idempotent');
  assert(getSkipLinkHref(wrapped) === '/contact#main',
  'Personal shell should keep skip links on the current canonical page when a root base URL is present');
  assert(count(wrapped, /<main\b/gi) === 1, 'Personal shell should preserve exactly one main element');
  assert(/<body[^>]*data-page="contact"/i.test(wrapped), 'Personal shell should preserve the original data-page');
  assert(/<body[^>]*data-audience="personal"/i.test(wrapped), 'Personal shell should stamp the personal audience');
  assert(count(wrapped, /data-personal-rail-active="true"/g) === 1,
    'Personal shell should identify one active category marker');
  assert(count(wrapped, /class="personal-accordion__rail(?:\s|")/g) === 1 &&
    !/<a\b[^>]*class="[^"]*\bpersonal-accordion__rail\b/i.test(wrapped),
  'Personal shell should render one static, non-link category marker');
  assert(/class="personal-accordion__rails"[^>]*aria-hidden="true"/i.test(wrapped),
    'Desktop category marker should be decorative because the page heading carries identity');
  assert(wrapped.includes('href="/#contact" aria-label="Back to categories"') &&
    wrapped.includes('personal-accordion__back-label--mobile" aria-hidden="true">Categories</span>'),
    'Contact detail should return to the homepage categories');
  assert(count(wrapped, /class="personal-accordion__toolbar"/g) === 1,
    'Personal shell should render one shared desktop toolbar and mobile context bar');
  assert(wrapped.indexOf('first.js') < wrapped.indexOf('second.js'), 'Personal shell should preserve script order');
  assert(/personal-accordion-shell:end -->\r?\n<script src="first\.js">/.test(wrapped),
    'The shell boundary should leave following scripts at the start of a new line');
  const unwrapped = unwrapPersonalAccordionHtml(wrapped);
  assert(unwrapped.includes('<main id="main"><h1>Contact</h1></main>') &&
    unwrapped.indexOf('first.js') < unwrapped.indexOf('second.js'),
  'Unwrapping should restore the original main fragment and script order');
  const sampleWithManagedStyle = sample.replace(
    '</head>',
    '<link rel="stylesheet" href="dist/styles-personal-accordion.1234abcd.css"></head>'
  );
  const wrappedWithManagedStyle = wrapPersonalAccordionHtml(sampleWithManagedStyle, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail'
  });
  assert(count(wrappedWithManagedStyle, /styles-personal-accordion\.1234abcd\.css/g) === 1,
    'Ordinary unwrap and rewrap should preserve one existing hashed personal shell stylesheet');
  assert(count(wrapPersonalAccordionHtml(wrappedWithManagedStyle, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail'
  }), /styles-personal-accordion\.1234abcd\.css/g) === 1,
  'Repeated wrapping should not remove or duplicate the managed personal shell stylesheet');

  const projectLibrary = buildLibraryPage(sample, 'projects', {
    projects: {
      items: [{ id: 'sample', title: 'Sample project', href: '/portfolio/sample' }]
    }
  });
  assert(projectLibrary.includes('data-personal-accordion-view="library"') &&
    projectLibrary.includes('href="/#projects"') &&
    projectLibrary.includes('aria-label="Back to categories"') &&
    projectLibrary.includes('personal-accordion__back-label--mobile" aria-hidden="true">Categories</span>'),
  'Project library should include a categories back control targeting the Projects overview');
  assert(count(projectLibrary, /<h1\b/gi) === 1 &&
    projectLibrary.includes('<p class="personal-library__meta">1 project</p>') &&
    !projectLibrary.includes('home-library__page-link') &&
    !/Open the dedicated [^<]+ page/i.test(projectLibrary),
  'Canonical library should keep one heading, quiet item count, and no redundant dedicated-page control');

  const gameSample = sample.replace(
    '<main id="main"><h1>Contact</h1></main>',
    '<main id="main"><h1>Game</h1></main>\n<div class="mobile-controls">Controls</div>\n<footer>Footer</footer>'
  );
  const wrappedGameSample = wrapPersonalAccordionHtml(gameSample, {
    category: 'games',
    itemId: 'sample-game',
    view: 'detail',
    includeUntilScripts: true
  });
  assert(wrappedGameSample.indexOf('<div class="mobile-controls">') < wrappedGameSample.indexOf(PERSONAL_CONTENT_END) &&
    wrappedGameSample.indexOf(PERSONAL_SHELL_END) < wrappedGameSample.indexOf('<footer>Footer</footer>'),
  'Game wrapping should include adjacent controls but leave a pre-existing footer outside the accordion shell');

  const manifest = readJson('dist/styles-manifest.json');
  assert(typeof manifest.personalAccordionFile === 'string' && manifest.personalAccordionFile.length > 0,
    'CSS manifest should expose the hashed personal accordion bundle');
  const managedStylesheet = `dist/${manifest.personalAccordionFile}`;
  const projectPages = walkHtml('pages/portfolio').map((filePath) => (
    path.relative(ROOT, filePath).replace(/\\/g, '/')
  ));
  const managedPages = [
    ['pages/portfolio.html', 'projects'],
    ['pages/tools.html', 'tools'],
    ['pages/games.html', 'games'],
    ['pages/contact.html', 'contact'],
    ...projectPages.map((relativePath) => [relativePath, 'projects']),
    ...[...TOOL_PAGE_IDS, ...INTERNAL_TOOL_PAGE_IDS].map((itemId) => [`pages/${itemId}.html`, 'tools']),
    ...Object.values(GAME_PAGE_PATHS).map((relativePath) => [relativePath.replace(/\\/g, '/'), 'games']),
    ...UTILITY_PAGE_CONFIGS.map((config) => [config.relPath.replace(/\\/g, '/'), config.category])
  ];
  const uniqueManagedPages = Array.from(new Map(managedPages.map((entry) => [entry[0], entry])).values());
  const projectDetailPages = new Set(projectPages);
  const toolDetailPages = new Set(
    [...TOOL_PAGE_IDS, ...INTERNAL_TOOL_PAGE_IDS].map((itemId) => `pages/${itemId}.html`)
  );
  const gameDetailPages = new Set(Object.values(GAME_PAGE_PATHS).map((relativePath) => (
    relativePath.replace(/\\/g, '/')
  )));
  const utilityPages = new Map(UTILITY_PAGE_CONFIGS.map((config) => [
    config.relPath.replace(/\\/g, '/'),
    config
  ]));
  assert(uniqueManagedPages.length === 49,
    'The personal shell route sweep should cover four category roots, six utility/fallback pages, 16 projects, 18 tools, and five games');
  assert(INTERNAL_TOOL_PAGE_IDS.length === 8 &&
    INTERNAL_TOOL_PAGE_IDS.every((itemId) => !TOOL_PAGE_IDS.includes(itemId)),
  'Account-reachable tools should remain a distinct internal shell list instead of joining the public catalog');
  assert(!Object.prototype.hasOwnProperty.call(GAME_PAGE_PATHS, 'project-starfall'),
    'Project Starfall should stay out of generated personal game routing');
  uniqueManagedPages.forEach(([relativePath, category]) => {
    const html = read(relativePath);
    assert(html.includes('data-personal-accordion-shell'), `${relativePath} should use the personal shell`);
    assert(html.includes(`data-personal-category="${category}"`), `${relativePath} should activate ${category}`);
    assert(count(html, /data-personal-rail-active="true"/g) === 1,
      `${relativePath} should identify exactly one active category marker`);
    assert(count(html, /class="personal-accordion__rail(?:\s|")/g) === 1 &&
      !/<a\b[^>]*class="[^"]*\bpersonal-accordion__rail\b/i.test(html),
    `${relativePath} should expose one static category marker and no cross-category rail links`);
    assert(/class="personal-accordion__rails"[^>]*aria-hidden="true"/i.test(html),
      `${relativePath} should keep its desktop marker out of the accessibility tree`);
    assert(count(html, /class="personal-accordion__toolbar"/g) === 1,
      `${relativePath} should render one shared toolbar/context bar`);
    assert(count(html, /<main\b/gi) === 1, `${relativePath} should retain one main element`);
    assert(count(html, /<footer\b[^>]*\bfooter--personal-compact\b/gi) === 1,
      `${relativePath} should include one compact personal footer`);
    const isImmersiveGame = relativePath === 'pages/games/stellar-dogfight.html';
    if (isImmersiveGame) {
      assert(/<body[^>]*data-personal-fit="immersive"/i.test(html) &&
        /<body[^>]*data-personal-chrome="compact"/i.test(html),
      `${relativePath} should retain immersive geometry while using compact personal chrome`);
    } else {
      assert(/<body[^>]*data-personal-fit="viewport"/i.test(html) &&
        /<body[^>]*data-personal-chrome="compact"/i.test(html),
      `${relativePath} should use the viewport-fit shell and compact personal chrome`);
    }
    if (projectDetailPages.has(relativePath)) {
      assert(html.includes('href="/portfolio" aria-label="Back to project library"') &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Library</span>'),
      `${relativePath} should return to the canonical project library with a compact mobile label`);
      assert(!html.includes('project-pager'), `${relativePath} should omit Previous and Next project navigation`);
    }
    const libraryBackTargets = {
      'pages/portfolio.html': '/#projects',
      'pages/tools.html': '/#tools',
      'pages/games.html': '/#games',
      'pages/contact.html': '/#contact'
    };
    if (libraryBackTargets[relativePath]) {
      assert(html.includes(`href="${libraryBackTargets[relativePath]}" aria-label="Back to categories"`) &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Categories</span>'),
        `${relativePath} should return to its homepage category selector`);
    }
    if (utilityPages.has(relativePath)) {
      const utility = utilityPages.get(relativePath);
      assert(html.includes(`href="${utility.backHref}" aria-label="Back to categories"`) &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Categories</span>'),
      `${relativePath} should return to its matching homepage category context`);
    }
    if (['pages/portfolio.html', 'pages/tools.html', 'pages/games.html'].includes(relativePath)) {
      assert(count(html, /<h1\b/gi) === 1 &&
        /<p class="personal-library__meta">\d+ (?:project|tool|game)s?<\/p>/i.test(html) &&
        !html.includes('home-library__page-link') &&
        !/Open the dedicated [^<]+ page/i.test(html),
      `${relativePath} should use one scrollable title, concise lead/count metadata, and no redundant page control`);
    }
    if (toolDetailPages.has(relativePath)) {
      assert(html.includes('href="/tools" aria-label="Back to tool library"') &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Library</span>'),
        `${relativePath} should return to the tool library before the homepage categories`);
    }
    if (gameDetailPages.has(relativePath)) {
      assert(html.includes('href="/games" aria-label="Back to game library"') &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Library</span>'),
        `${relativePath} should return to the game library before the homepage categories`);
    }
    const canonical = getTagAttribute(html, /<link\b[^>]*\brel="canonical"[^>]*>/i, 'href');
    const canonicalPath = canonical ? new URL(canonical, 'https://www.danielshort.me').pathname : '';
    const cleanCanonicalPath = canonicalPath.replace(/\.html$/i, '');
    assert(cleanCanonicalPath && getSkipLinkHref(html) === `${cleanCanonicalPath}#main`,
      `${relativePath} should keep its skip link on the current canonical page`);
    assert(/personal-accordion-shell:end -->\r?\n/i.test(html),
      `${relativePath} should keep following document markup newline-separated from the shell`);
    assert(count(html, new RegExp(`href="${managedStylesheet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g')) === 1,
      `${relativePath} should reference the hashed personal shell stylesheet exactly once`);
    const publicHtml = read(path.join('public', relativePath));
    assert(count(publicHtml, new RegExp(`href="${managedStylesheet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g')) === 1,
      `public/${relativePath} should reference the hashed personal shell stylesheet exactly once`);
  });

  const oceanWaveHtml = read('pages/ocean-wave-simulation.html');
  assert(oceanWaveHtml.indexOf('personal-accordion-content:start') < oceanWaveHtml.indexOf('ocean-wave-hero') &&
    oceanWaveHtml.indexOf('ocean-wave-hero') < oceanWaveHtml.indexOf('<main id="main">'),
  'Ocean Wave Simulation should keep its page hero inside the selected Games panel');

  const professionalFiles = walkHtml('pages/professional');
  assert(professionalFiles.length > 0, 'Build should generate internal professional portfolio/contact copies');
  professionalFiles.forEach((filePath) => {
    const html = fs.readFileSync(filePath, 'utf8');
    const relativePath = path.relative(ROOT, filePath).replace(/\\/g, '/');
    const audience = relativePath.split('/')[2];
    const canonical = getTagAttribute(html, /<link\b[^>]*\brel="canonical"[^>]*>/i, 'href');
    const ogUrl = getTagAttribute(html, /<meta\b[^>]*\bproperty="og:url"[^>]*>/i, 'content');
    assert(!html.includes('data-personal-accordion-shell'), `${relativePath} should stay unwrapped`);
    assert(new RegExp(`<body[^>]*data-audience="${audience}"`, 'i').test(html), `${relativePath} should retain its audience`);
    assert(count(html, /<meta\b[^>]*\bname="robots"[^>]*\bcontent="noindex, nofollow"[^>]*>/gi) === 1,
      `${relativePath} should contain one noindex directive`);
    assert(canonical.includes(`?audience=${audience}`), `${relativePath} should canonicalize to its visible audience URL`);
    assert(canonical === ogUrl, `${relativePath} canonical and og:url should match`);
    assert(!canonical.includes('/pages/professional/'), `${relativePath} should not expose its internal storage URL`);
    assert(html.includes('data-footer-realm="professional"') && !html.includes('data-footer-realm="personal"'),
      `${relativePath} should retain professional footer chrome`);
    assert(html.includes(`href="${audience}" class="nav-link" data-professional-home-link="true"`),
      `${relativePath} should retain its audience-specific professional header`);
  });

  const vercel = readJson('vercel.json');
  const hasRewrite = (source, key, value, destination) => vercel.rewrites.some((rule) => (
    rule.source === source &&
    rule.destination === destination &&
    Array.isArray(rule.has) &&
    rule.has.some((condition) => condition.type === 'query' && condition.key === key && condition.value === value)
  ));
  assert(hasRewrite('/portfolio/:project', 'audience', 'data-science', '/pages/professional/data-science/portfolio/:project'),
    'Project audience deep links should route to an unwrapped professional copy');
  assert(hasRewrite('/portfolio', 'audience', 'tourism', '/pages/professional/tourism/portfolio'),
    'Portfolio audience links should route to an unwrapped professional copy');
  assert(hasRewrite('/contact', 'audience', 'analytics', '/pages/professional/analytics/contact'),
    'Contact audience links should route to an unwrapped professional copy');
  assert(hasRewrite('/portfolio/:project', 'mode', '(professional|work|career|analytics)', '/pages/professional/analytics/portfolio/:project'),
    'Legacy professional mode deep links should retain the professional project layout');

  const searchIndex = read('build/generate-search-index.js');
  const chatbotKnowledge = read('build/generate-chatbot-knowledge.js');
  const aiDigests = read('build/generate-ai-digests.js');
  [searchIndex, chatbotKnowledge, aiDigests].forEach((source) => {
    assert(source.includes("startsWith('pages/professional/')"),
      'Recursive public discovery should exclude internal professional copies by source path');
  });

  const probabilityApp = read('js/games/probability-engine/app.js');
  assert(probabilityApp.includes('dom.modalBackground = Array.from(new Set([') &&
    probabilityApp.includes('dom.appShell') &&
    probabilityApp.includes('.personal-accordion__rails'),
  'Probability Engine modal background should include its nested app shell and shell navigation');
  assert(probabilityApp.includes('element.setAttribute("inert", "")') &&
    probabilityApp.includes('dom.offlineModal.setAttribute("aria-hidden", "false")') &&
    probabilityApp.includes('window.requestAnimationFrame(() => dom.claimOfflineButton.focus())'),
  'Probability Engine modal open should inert the background, expose the dialog, and move focus');
  assert(probabilityApp.includes('element.removeAttribute("inert")') &&
    probabilityApp.includes('dom.offlineModal.setAttribute("aria-hidden", "true")') &&
    probabilityApp.includes('lastFocusedBeforeOffline.focus()'),
  'Probability Engine modal close should restore interactivity, hide the dialog, and restore focus');

  const shellCss = read('css/components/personal-accordion-shell.css');
  assert(shellCss.includes('.personal-accordion--contact .personal-accordion__panel') &&
    shellCss.includes('border-radius: 0 12px 12px 0;'),
  'Only the Contact panel should receive the two outer right corner radii');
  assert(shellCss.includes('html:has(body.personal-accordion-page)') &&
    shellCss.includes('--personal-scrollbar-color'),
  'Document scrollbars should inherit the active category theme');
  const railRule = /\.personal-accordion__rail\s*\{([\s\S]*?)\n\s*\}/.exec(shellCss)?.[1] || '';
  assert(shellCss.includes('--personal-rail-size: 68px;') &&
    shellCss.includes('grid-template-rows: minmax(0, 1fr);') &&
    railRule.includes('background: var(--rail-color);') &&
    !railRule.includes('transition:'),
  'Desktop category marker should be a full-height static 68px solid-color rail');
  assert(shellCss.includes('--personal-toolbar-size: 48px;') &&
    shellCss.includes('border: 4px solid var(--panel-color);') &&
    shellCss.includes('min-height: 44px;'),
  'Desktop shell should use a four-pixel frame and compact 48px toolbar with a full touch target');
  assert(shellCss.includes('--personal-toolbar-size: 60px;') &&
    shellCss.includes('.personal-accordion__rails {\n      display: none;') &&
    shellCss.includes('.personal-accordion__context {') &&
    shellCss.includes('background: var(--panel-color);'),
  'Mobile shell should merge navigation into one 60px category-colored context bar');
  assert(shellCss.includes('.personal-library .home-library__header {') &&
    shellCss.includes('position: static;') &&
    shellCss.includes('.personal-library__meta {'),
  'Library title, lead, and quiet count should remain in ordinary scrollable content');
  assert(shellCss.includes('body.personal-accordion-page[data-personal-chrome="compact"] :is(') &&
    shellCss.includes('.speed-dial,') && shellCss.includes('.mobile-site-dock'),
  'Compact personal chrome should consistently suppress duplicate floating navigation');
  assert(shellCss.includes('[data-personal-item="probability-engine"] .personal-accordion__content') &&
    shellCss.includes('[data-personal-item="roulette"] .personal-accordion__content'),
  'Dark game adapters should preserve their authored backgrounds inside the white shell system');
  assert(!shellCss.includes('data-personal-item="project-starfall"'),
    'Removed Project Starfall routing should not retain dead personal-shell adapters');
}

module.exports = runPersonalAccordionShellTests;

if (require.main === module) {
  runPersonalAccordionShellTests({
    assert(condition, message) {
      if (!condition) throw new Error(message);
    }
  });
  process.stdout.write('Personal accordion shell tests passed.\n');
}
