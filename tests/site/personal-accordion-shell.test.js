'use strict';

const fs = require('fs');
const path = require('path');
const {
  CATEGORY_CONFIG,
  HARD_NAVIGATION_PATHS,
  PERSONAL_CONTENT_END,
  PERSONAL_SHELL_END,
  SITE_ROUTE_MANIFEST_VERSION,
  finalizePersonalRouteDocument,
  preparePersonalToolDetailHtml,
  renderPersonalLibraryMain,
  unwrapPersonalAccordionHtml,
  validatePersonalRouteDocument,
  wrapPersonalAccordionHtml
} = require('../../build/lib/personal-accordion-shell');
const {
  GAME_PAGE_PATHS,
  HARD_TOOL_PAGE_IDS,
  INTERNAL_TOOL_PAGE_IDS,
  TOOL_PAGE_IDS,
  UTILITY_PAGE_CONFIGS,
  buildLibraryPage,
  getToolDetailMetadata
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

function getRouteManifest(source) {
  const match = /<script\b[^>]*\bid="site-route-manifest"[^>]*>([\s\S]*?)<\/script>/i.exec(String(source || ''));
  return match ? JSON.parse(match[1]) : null;
}

function assertTransitionBootstrap(assert, relativePath, html) {
  const source = String(html || '');
  const headEnd = source.search(/<\/head>/i);
  const bodyStart = source.search(/<body\b/i);
  const scriptTags = source.match(/<script\b[^>]*><\/script>/gi) || [];
  const noJsTags = scriptTags.filter((tag) => /\bsrc="(?:\/)?js\/common\/no-js\.js"/i.test(tag));
  const shellTags = scriptTags.filter((tag) => /\bsrc="(?:\/)?dist\/site-shell\.[^"/]+\.js"/i.test(tag));
  const noJsIndex = noJsTags.length === 1 ? source.indexOf(noJsTags[0]) : -1;
  const shellIndex = shellTags.length === 1 ? source.indexOf(shellTags[0]) : -1;

  assert(headEnd >= 0 && bodyStart > headEnd,
    `${relativePath} should retain a complete head before its body`);
  assert(noJsTags.length === 1 && noJsIndex >= 0 && noJsIndex < headEnd &&
    !/\b(?:async|defer|type="module")\b/i.test(noJsTags[0]),
  `${relativePath} should run one synchronous transition bootstrap in the document head`);
  assert(shellTags.length === 1 && shellIndex >= 0 && shellIndex < headEnd &&
    /\bdefer\b/i.test(shellTags[0]),
  `${relativePath} should load one deferred site shell from the document head`);
  const stylesheetIndex = source.search(/<link\b[^>]*\brel="stylesheet"[^>]*>/i);
  assert(stylesheetIndex >= 0 && stylesheetIndex < noJsIndex,
    `${relativePath} should declare its transition styles before the synchronous preload bootstrap`);
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
  assert(count(wrapped, /class="personal-accordion__rail(?:\s|")/g) === 5 &&
    /<a\b[^>]*class="[^"]*\bpersonal-accordion__rail\b[^>]*href="\/#contact"[^>]*aria-label="Return to the Contact section on the homepage"/i.test(wrapped) &&
    wrapped.includes('data-personal-transition="collapse"') &&
    count(wrapped, /<a\b[^>]*\bdata-site-tab="[^"]+"[^>]*\bhidden\b[^>]*\binert\b/gi) === 4,
  'Personal shell should preserve five category rail slots while exposing only the active return rail');
  assert(/class="personal-accordion__rails"[^>]*data-personal-category-marker="contact"/i.test(wrapped) &&
    !/class="personal-accordion__rails"[^>]*aria-hidden=/i.test(wrapped),
  'Clickable category rail should remain exposed to assistive technology');
  assert(wrapped.includes('href="/#contact" aria-label="Back to categories"') &&
    wrapped.includes('personal-accordion__back-label--mobile" aria-hidden="true">Categories</span>'),
    'Contact detail should return to the homepage categories');
  assert(count(wrapped, /class="personal-accordion__toolbar"/g) === 1,
    'Personal shell should render one shared desktop toolbar and mobile context bar');
  assert(count(wrapped, /data-site-route-content/g) === 1 &&
    /<section\b[^>]*data-personal-accordion-shell[^>]*data-site-route-content/i.test(wrapped) &&
    !/<div\b[^>]*data-personal-detail-content[^>]*data-site-route-content/i.test(wrapped) &&
    count(wrapped, /data-site-route-toolbar/g) === 1 &&
    count(wrapped, /data-site-route-progress/g) === 1 &&
    count(wrapped, /data-site-route-announcer/g) === 1,
  'Personal shell should expose its outer scene as the only atomic route outlet plus one toolbar, progress line, and live announcer');
  const wrappedManifest = getRouteManifest(wrapped);
  assert(wrappedManifest && wrappedManifest.version === SITE_ROUTE_MANIFEST_VERSION &&
    wrappedManifest.id === 'contact:contact' &&
    wrappedManifest.path === '/contact' &&
    wrappedManifest.category === 'contact' &&
    wrappedManifest.view === 'detail' &&
    wrappedManifest.navigation === 'soft' &&
    wrappedManifest.module === wrappedManifest.id &&
    wrappedManifest.scripts.join(',') === '/first.js,/second.js' &&
    /<body[^>]*data-site-route-id="contact:contact"[^>]*data-site-route-category="contact"[^>]*data-site-route-view="detail"[^>]*data-site-route-navigation="soft"/i.test(wrapped),
  'Personal shell should publish matching body metadata and a versioned per-document route manifest');
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
  assert(count(wrappedWithManagedStyle, /<link\b[^>]*styles-personal-accordion\.1234abcd\.css[^>]*>/g) === 1,
    'Ordinary unwrap and rewrap should preserve one existing hashed personal shell stylesheet');
  assert(count(wrapPersonalAccordionHtml(wrappedWithManagedStyle, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail'
  }), /<link\b[^>]*styles-personal-accordion\.1234abcd\.css[^>]*>/g) === 1,
  'Repeated wrapping should not remove or duplicate the managed personal shell stylesheet');

  const projectLibrary = buildLibraryPage(sample, 'projects', {
    projects: {
      items: [{ id: 'sample', title: 'Sample project', href: '/portfolio/sample' }]
    }
  });
  assert(projectLibrary.includes('data-personal-accordion-view="library"') &&
    projectLibrary.includes('href="/#projects"') &&
    projectLibrary.includes('aria-label="Back to homepage"') &&
    projectLibrary.includes('personal-accordion__back-label--mobile" aria-hidden="true">Home</span>'),
  'Project library should include a homepage back control targeting the Projects overview');
  assert(count(projectLibrary, /<h1\b/gi) === 1 &&
    projectLibrary.includes('<p class="personal-library__meta">1 project</p>') &&
    !projectLibrary.includes('home-library__page-link') &&
    !/Open the dedicated [^<]+ page/i.test(projectLibrary),
  'Canonical library should keep one heading, quiet item count, and no redundant dedicated-page control');

  const toolsLibrary = renderPersonalLibraryMain({
    category: 'tools',
    items: [{ id: 'sample-tool', title: 'Sample tool', summary: 'Sample summary.', href: '/tools/sample-tool' }]
  });
  assert(toolsLibrary.includes('<h1 id="personal-library-title-tools">Tool library</h1>') &&
    toolsLibrary.includes('The complete collection of small utilities for text, links, media, and recurring workflows.') &&
    toolsLibrary.includes('<p class="personal-library__meta">1 tool</p>') &&
    !toolsLibrary.includes('personal-library--tools tools-hero') &&
    toolsLibrary.includes('data-personal-tool-account="true"') &&
    !toolsLibrary.includes('>All tools<'),
  'Tool library renderer should share homepage copy, avoid legacy hero semantics, and expose one compact account slot without duplicate navigation');

  const toolSample = sample.replace(
    '<main id="main"><h1>Contact</h1></main>',
    '<section class="hero hero--tools tools-hero"><div class="wrapper"><p class="hero-eyebrow">Old tool hero</p><h1 class="visually-hidden">Sample tool</h1></div></section><div class="tools-account-dock" data-tools-account="dock"><div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner"><div class="tools-account-bar" data-tools-account="bar"></div></div></div><main id="main"><section><h2>Tool interface</h2></section></main>'
  );
  const toolMetadata = { itemId: 'sample-tool', title: 'Sample tool', summary: 'A focused sample tool.', includeAccount: true };
  const toolOptions = {
    category: 'tools',
    itemId: 'sample-tool',
    view: 'detail',
    backHref: '/tools',
    backLabel: 'Back to tool library',
    backCompactLabel: 'Library',
    backAriaLabel: 'Back to tool library',
    includePersonalToolHeader: true
  };
  const preparedTool = preparePersonalToolDetailHtml(toolSample, toolMetadata);
  const wrappedTool = wrapPersonalAccordionHtml(preparedTool, toolOptions);
  const wrappedToolAgain = wrapPersonalAccordionHtml(
    preparePersonalToolDetailHtml(wrappedTool, toolMetadata),
    toolOptions
  );
  assert(wrappedTool === wrappedToolAgain &&
    count(wrappedTool, /data-personal-tool-header=/g) === 1 &&
    count(wrappedTool, /<h1\b/gi) === 1 &&
    !wrappedTool.includes('class="hero hero--tools tools-hero"') &&
    wrappedTool.includes('data-personal-tool-account="true"') &&
    wrappedTool.includes('data-personal-tool-account-bar="true"') &&
    !wrappedTool.includes('>All tools<') &&
    count(wrappedTool, /<main\b/gi) === 1,
  'Tool detail preparation should be idempotent and replace legacy chrome with one native title and stable account slot');

  const copilotMetadata = getToolDetailMetadata('job-application-copilot', read('pages/job-application-copilot.html'));
  assert(copilotMetadata.title === 'Job Application Copilot' &&
    copilotMetadata.summary.includes('local-first Chrome extension') &&
    copilotMetadata.includeAccount === false,
  'Non-catalog tool pages should use explicit metadata and opt out of an unrelated account slot');

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
  assert(uniqueManagedPages.length === 50,
    'The personal shell route sweep should cover four category roots, seven utility/fallback pages, 16 projects, 18 tools, and five games');
  assert(INTERNAL_TOOL_PAGE_IDS.length === 8 &&
    INTERNAL_TOOL_PAGE_IDS.every((itemId) => !TOOL_PAGE_IDS.includes(itemId)),
  'Account-reachable tools should remain a distinct internal shell list instead of joining the public catalog');
  assert(!Object.prototype.hasOwnProperty.call(GAME_PAGE_PATHS, 'project-starfall'),
    'Project Starfall should stay out of generated personal game routing');
  assertTransitionBootstrap(assert, 'index.html', read('index.html'));
  assertTransitionBootstrap(assert, 'public/index.html', read('public/index.html'));
  uniqueManagedPages.forEach(([relativePath, category]) => {
    const html = read(relativePath);
    assertTransitionBootstrap(assert, relativePath, html);
    assert(html.includes('data-personal-accordion-shell'), `${relativePath} should use the personal shell`);
    assert(html.includes(`data-personal-category="${category}"`), `${relativePath} should activate ${category}`);
    assert(count(html, /data-personal-rail-active="true"/g) === 1,
      `${relativePath} should identify exactly one active category marker`);
    assert(count(html, /class="personal-accordion__rail(?:\s|")/g) === 5 &&
      new RegExp(`<a\\b[^>]*class="[^"]*\\bpersonal-accordion__rail\\b[^>]*href="\\/#${category}"`).test(html) &&
      html.includes(`aria-label="Return to the ${CATEGORY_CONFIG[category].label} section on the homepage"`) &&
      html.includes('data-personal-transition="collapse"') &&
      count(html, /<a\b[^>]*\bdata-site-tab="[^"]+"[^>]*\bhidden\b[^>]*\binert\b/gi) === 4,
    `${relativePath} should retain five stable rail slots while exposing only its active category`);
    assert(!/class="personal-accordion__rails"[^>]*aria-hidden=/i.test(html),
      `${relativePath} should expose its return rail to assistive technology`);
    assert(count(html, /class="personal-accordion__toolbar"/g) === 1,
      `${relativePath} should render one shared toolbar/context bar`);
    assert(count(html, /data-site-route-content/g) === 1 &&
      count(html, /data-site-route-toolbar/g) === 1 &&
      count(html, /data-site-route-progress/g) === 1 &&
      count(html, /data-site-route-announcer/g) === 1 &&
      count(html, /data-site-shell-header/g) === 1 &&
      count(html, /data-site-shell-footer/g) === 1,
    `${relativePath} should expose one persistent shell and one atomic route surface`);
    const routeManifest = getRouteManifest(html);
    const validatedRouteManifest = validatePersonalRouteDocument(html);
    const expectedItem = getTagAttribute(html, /<body\b[^>]*>/i, 'data-personal-item');
    assert(routeManifest && routeManifest.version === SITE_ROUTE_MANIFEST_VERSION &&
      routeManifest.id === `${category}:${expectedItem}` &&
      routeManifest.category === category &&
      routeManifest.module === ({ project: 'page:content', search: 'search:search', contact: 'contact:contact' }[getTagAttribute(html, /<body\b[^>]*>/i, 'data-page')] || routeManifest.id) &&
      validatedRouteManifest.id === routeManifest.id &&
      Array.isArray(routeManifest.styles) && routeManifest.styles.length >= 2 &&
      Array.isArray(routeManifest.scripts) && routeManifest.scripts.length >= 2,
    `${relativePath} should include complete versioned route metadata and ordered resources`);
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
      'pages/games.html': '/#games'
    };
    if (libraryBackTargets[relativePath]) {
      assert(html.includes(`href="${libraryBackTargets[relativePath]}" aria-label="Back to homepage"`) &&
        html.includes('personal-accordion__back-label--mobile" aria-hidden="true">Home</span>'),
        `${relativePath} should return to its homepage category`);
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
      assert(count(html, /data-personal-tool-header=/g) === 1 &&
        count(html, /<h1\b/gi) === 1 &&
        !html.includes('class="hero hero--tools tools-hero"') &&
        !html.includes('>All tools<'),
      `${relativePath} should use one shell-native tool title without legacy hero or duplicate account navigation`);
      const toolId = expectedItem;
      const isHardBoundary = HARD_TOOL_PAGE_IDS.includes(toolId);
      assert(routeManifest.navigation === (isHardBoundary ? 'hard' : 'soft') &&
        getTagAttribute(html, /<body\b[^>]*>/i, 'data-site-route-navigation') === (isHardBoundary ? 'hard' : 'soft'),
      `${relativePath} should classify its security boundary consistently in body and manifest metadata`);
      if (isHardBoundary) {
        assert(/<a\b[^>]*href="\/tools"[^>]*data-navigation="hard"/i.test(html),
          `${relativePath} should force native navigation when leaving its security-header boundary`);
      }
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
    assertTransitionBootstrap(assert, `public/${relativePath}`, publicHtml);
    assert(count(publicHtml, new RegExp(`href="${managedStylesheet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g')) === 1,
      `public/${relativePath} should reference the hashed personal shell stylesheet exactly once`);
  });

  assert(HARD_TOOL_PAGE_IDS.length === 3 &&
    HARD_TOOL_PAGE_IDS.every((itemId) => HARD_NAVIGATION_PATHS.includes(`/tools/${itemId}`)),
  'The generator and route manifest should share the three immutable security-header boundaries');

  const homeHtml = read('index.html');
  const homeManifest = getRouteManifest(homeHtml);
  assert(homeManifest && homeManifest.id === 'home' && homeManifest.path === '/' &&
    homeManifest.category === 'about' && homeManifest.view === 'overview' &&
    homeManifest.navigation === 'soft' && homeManifest.module === 'home' &&
    count(homeHtml, /data-site-tab="(?:about|projects|tools|games|contact)"/g) === 5 &&
    count(homeHtml, /data-site-tab-active="true"/g) === 1 &&
    count(homeHtml, /data-site-route-content/g) === 1 &&
    count(homeHtml, /data-site-shell-header/g) === 1 &&
    count(homeHtml, /data-site-shell-footer/g) === 1,
  'Homepage should expose its five tabs, default About state, persistent chrome, and route manifest');
  assert(read('build/templates/header.partial.html').includes('data-site-shell-header') &&
    read('build/templates/footer.partial.html').includes('data-site-shell-footer') &&
    read('build/lib/cms-renderers.js').includes("'<header id=\"combined-header-nav\" data-site-shell-header>'") &&
    read('build/lib/cms-renderers.js').includes("'<footer class=\"footer footer-classic footer--personal-compact\" data-site-shell-footer>'"),
  'CMS rendering and generated personal templates should preserve both persistent chrome hooks');

  const finalizedHome = finalizePersonalRouteDocument(homeHtml, { home: true });
  assert(count(finalizedHome, /id="site-route-manifest"/g) === 1 &&
    getRouteManifest(finalizedHome).scripts.some((src) => /^\/dist\/site-home(?:\.[0-9a-f]{8})?\.js$/i.test(src)),
  'Route-manifest finalization should remain idempotent while refreshing homepage resources');
  assert(validatePersonalRouteDocument(homeHtml).id === 'home',
    'The built homepage should pass fail-closed route lifecycle validation');
  let rejectedInlineScript = false;
  let rejectedExternalScript = false;
  let rejectedMissingModule = false;
  try {
    validatePersonalRouteDocument(homeHtml.replace('</body>', '<script>window.unclassified = true;</script></body>'));
  } catch (_) {
    rejectedInlineScript = true;
  }
  try {
    validatePersonalRouteDocument(homeHtml.replace('</body>', '<script src="/js/unclassified.js"></script></body>'));
  } catch (_) {
    rejectedExternalScript = true;
  }
  try {
    validatePersonalRouteDocument(homeHtml.replace('"module":"home"', '"module":""'));
  } catch (_) {
    rejectedMissingModule = true;
  }
  assert(rejectedInlineScript && rejectedExternalScript && rejectedMissingModule,
    'Soft-route validation should reject unclassified executable scripts and missing lifecycle modules');

  const oceanWaveHtml = read('pages/ocean-wave-simulation.html');
  assert(oceanWaveHtml.indexOf('personal-accordion-content:start') < oceanWaveHtml.indexOf('ocean-wave-hero') &&
    oceanWaveHtml.indexOf('ocean-wave-hero') < oceanWaveHtml.indexOf('<main id="main">'),
  'Ocean Wave Simulation should keep its page hero inside the selected Games panel');

  const professionalFiles = walkHtml('pages/professional');
  assert(professionalFiles.length > 0, 'Build should generate internal professional portfolio/contact copies');
  professionalFiles.forEach((filePath) => {
    const html = fs.readFileSync(filePath, 'utf8');
    const relativePath = path.relative(ROOT, filePath).replace(/\\/g, '/');
    assertTransitionBootstrap(assert, relativePath, html);
    assertTransitionBootstrap(assert, `public/${relativePath}`, read(path.join('public', relativePath)));
    const audience = relativePath.split('/')[2];
    const canonical = getTagAttribute(html, /<link\b[^>]*\brel="canonical"[^>]*>/i, 'href');
    const ogUrl = getTagAttribute(html, /<meta\b[^>]*\bproperty="og:url"[^>]*>/i, 'content');
    assert(html.includes('data-personal-accordion-shell'), `${relativePath} should use the shared tab shell`);
    assert(new RegExp(`<body[^>]*data-audience="${audience}"`, 'i').test(html), `${relativePath} should retain its audience`);
    assert(count(html, /<meta\b[^>]*\bname="robots"[^>]*\bcontent="noindex, nofollow"[^>]*>/gi) === 1,
      `${relativePath} should contain one noindex directive`);
    assert(canonical.includes(`?audience=${audience}`), `${relativePath} should canonicalize to its visible audience URL`);
    assert(canonical === ogUrl, `${relativePath} canonical and og:url should match`);
    assert(!canonical.includes('/pages/professional/'), `${relativePath} should not expose its internal storage URL`);
    assert(html.includes('footer--personal-compact'), relativePath + ' should use the shared compact footer');
    assert(html.includes('data-site-tab-rail-mode="navigation"') && !html.includes('id="primary-menu"'), relativePath + ' should use audience tabs without header menus');
  });

  const vercel = readJson('vercel.json');
  const hasRewrite = (source, key, value, destination) => vercel.rewrites.some((rule) => (
    rule.source === source &&
    rule.destination === destination &&
    Array.isArray(rule.has) &&
    rule.has.some((condition) => condition.type === 'query' && condition.key === key && condition.value === value)
  ));
  assert(hasRewrite('/portfolio/:project', 'audience', 'data-science', '/pages/professional/data-science/portfolio/:project'),
    'Project audience deep links should route to a professional copy with the shared tab shell');
  assert(hasRewrite('/portfolio', 'audience', 'tourism', '/pages/professional/tourism/portfolio'),
    'Portfolio audience links should route to a professional copy with the shared tab shell');
  assert(hasRewrite('/contact', 'audience', 'analytics', '/pages/professional/analytics/contact'),
    'Contact audience links should route to a professional copy with the shared tab shell');
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
  assert(probabilityApp.includes('window.createModalAccessibility(dom.offlineModal)') &&
    probabilityApp.includes('dom.offlineModal.before(offlineModalPlaceholder)') &&
    probabilityApp.includes('document.body.appendChild(dom.offlineModal)'),
  'Probability Engine offline dialog should use shared accessibility and portal outside the clipped route content');
  assert(probabilityApp.includes('offlineModalAccessibility.show()') &&
    probabilityApp.includes('offlineModalAccessibility.isolateBackground()') &&
    probabilityApp.includes('dom.claimOfflineButton.focus({ preventScroll: true })'),
  'Probability Engine modal open should expose the dialog, focus Claim, and isolate background controls');
  assert(probabilityApp.includes('offlineModalAccessibility.hide({') &&
    probabilityApp.includes('lastFocusedBeforeOffline.focus({ preventScroll: true })') &&
    probabilityApp.includes('offlineModalAccessibility.dispose()') &&
    probabilityApp.includes('window.SiteRoutes?.addCleanup(disposeOfflineModal)') &&
    probabilityApp.includes('offlineModalPlaceholder.replaceWith(dom.offlineModal)'),
  'Probability Engine modal should restore its owner and interactivity after close or route disposal');

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
    railRule.includes('cursor: pointer;') &&
    railRule.includes('transition: filter .18s ease, box-shadow .18s ease;') &&
    shellCss.includes('.personal-accordion__rail:is(:hover, :focus-visible)') &&
    shellCss.includes('.personal-accordion__rail:focus-visible') &&
    shellCss.includes('@media (prefers-reduced-motion: reduce)') &&
    shellCss.includes('.personal-accordion__rail,'),
  'Desktop category rail should be a full-height actionable 68px control with restrained motion and visible focus');
  assert(shellCss.includes('--personal-toolbar-size: 60px;') &&
    shellCss.includes('border: 4px solid var(--panel-color);') &&
    shellCss.includes('min-height: 44px;') &&
    shellCss.includes('padding-block: 8px;') &&
    shellCss.includes('--personal-content-width: 1068px;'),
  'Desktop shell should use a four-pixel frame and a spaced 60px toolbar aligned to the shared content measure');
  assert(shellCss.includes('--personal-toolbar-size: 60px;') &&
    shellCss.includes('--personal-mobile-rail-size: 48px;') &&
    shellCss.includes('grid-template-rows: var(--personal-mobile-rail-size) minmax(0, 1fr);') &&
    shellCss.includes('.personal-accordion__rail[hidden] {\n    display: none !important;') &&
    shellCss.includes('.personal-accordion__rail-label {\n      font-size: .76rem;') &&
    shellCss.includes('.personal-accordion__context {') &&
    shellCss.includes('color: var(--panel-color);'),
  'Mobile shell should expose one horizontal active rail above a restrained 60px route toolbar');
  assert(shellCss.includes('.personal-library .home-library__header {') &&
    shellCss.includes('position: static;') &&
    shellCss.includes('.personal-library__meta {'),
  'Library title, lead, and quiet count should remain in ordinary scrollable content');
  assert(shellCss.includes('[data-personal-item="privacy"] .personal-accordion__content > #main') &&
    shellCss.includes('[data-personal-item="solutions"] .personal-accordion__content > #main > .hero.hero--default') &&
    shellCss.includes('overflow-x: clip;'),
  'Wrapped utility pages should shed legacy viewport offsets and keep horizontal overflow inside owned components');
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
