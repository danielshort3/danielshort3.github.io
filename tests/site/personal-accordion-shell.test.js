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
  TOOL_PAGE_IDS
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
    '<html><head><link rel="canonical" href="https://www.danielshort.me/contact"></head>',
    '<body class="contact-page" data-page="contact">',
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
    backLabel: 'Back to Contact overview'
  });
  const wrappedAgain = wrapPersonalAccordionHtml(wrapped, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail',
    backHref: '/#contact',
    backLabel: 'Back to Contact overview'
  });
  assert(wrappedAgain === wrapped, 'Personal shell wrapping should be idempotent');
  assert(count(wrapped, /<main\b/gi) === 1, 'Personal shell should preserve exactly one main element');
  assert(/<body[^>]*data-page="contact"/i.test(wrapped), 'Personal shell should preserve the original data-page');
  assert(/<body[^>]*data-audience="personal"/i.test(wrapped), 'Personal shell should stamp the personal audience');
  assert(count(wrapped, /aria-current="page"/g) === 1, 'Personal shell should identify one active rail');
  assert(wrapped.includes('href="/#contact"') && wrapped.includes('Back to Contact overview'),
    'Contact detail should return to the homepage Contact overview');
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
    ...TOOL_PAGE_IDS.map((itemId) => [`pages/${itemId}.html`, 'tools']),
    ...Object.values(GAME_PAGE_PATHS).map((relativePath) => [relativePath.replace(/\\/g, '/'), 'games'])
  ];
  const uniqueManagedPages = Array.from(new Map(managedPages.map((entry) => [entry[0], entry])).values());
  assert(uniqueManagedPages.length === 36,
    'The personal shell route sweep should cover four indexes, 16 projects, 10 tools, and six games');
  uniqueManagedPages.forEach(([relativePath, category]) => {
    const html = read(relativePath);
    assert(html.includes('data-personal-accordion-shell'), `${relativePath} should use the personal shell`);
    assert(html.includes(`data-personal-category="${category}"`), `${relativePath} should activate ${category}`);
    assert(count(html, /aria-current="page"/g) === 1, `${relativePath} should identify exactly one active rail`);
    assert(count(html, /<main\b/gi) === 1, `${relativePath} should retain one main element`);
    assert(count(html, /<footer\b[^>]*\bfooter--personal-compact\b/gi) === 1,
      `${relativePath} should include one compact personal footer`);
    assert(/personal-accordion-shell:end -->\r?\n(?:\s*<footer|\s*<script)/i.test(html),
      `${relativePath} should keep footer or script markup newline-separated from the shell`);
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
  assert(shellCss.includes('[data-personal-item="probability-engine"] .personal-accordion__content') &&
    shellCss.includes('[data-personal-item="roulette"] .personal-accordion__content'),
  'Dark game adapters should preserve their authored backgrounds inside the white shell system');
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
