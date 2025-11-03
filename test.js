// Lightweight test runner used by `npm test`
const fs = require('fs');
const vm = require('vm');

// Assert helper
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// Verify an HTML file contains a required snippet
function checkFileContains(file, text) {
  assert(fs.existsSync(file), `${file} does not exist`);
  const content = fs.readFileSync(file, 'utf8');
  assert(content.includes(text), `${file} missing expected text: ${text}`);
}

function createEnv() {
  const env = {
    window: {},
    document: {
      addEventListener: () => {},
      removeEventListener: () => {},
      querySelector: () => null,
      querySelectorAll: () => [],
      getElementById: () => null,
      createElement: () => ({ style: {}, classList: { add() {}, remove() {}, toggle() {} } }),
      body: { dataset: {}, appendChild() {} },
      documentElement: { style: { setProperty() {} } },
      head: { appendChild() {} },
      readyState: 'complete'
    },
    history: { pushState() {}, replaceState() {}, back() {} },
    location: { pathname: '', search: '', hash: '' },
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    performance: { now: () => Date.now() },
    requestAnimationFrame: cb => cb(Date.now()),
    setTimeout,
    clearTimeout,
    console,
  };
  env.dataLayer = [];
  env.window.dataLayer = env.dataLayer;
  env.window.matchMedia = env.matchMedia;
  env.window.addEventListener = () => {};
  env.window.requestAnimationFrame = env.requestAnimationFrame;
  env.window.performance = env.performance;
  env.window.document = env.document;
  return env;
}

// Evaluate a script in a minimal browser-like context
function evalScript(file, env) {
  const code = fs.readFileSync(file, 'utf8');
  const context = env || createEnv();
  vm.runInNewContext(code, context, { filename: file });
  return context;
}

try {
  // HTML checks across pages (moved files live under pages/)
  checkFileContains('index.html', 'made actionable');
checkFileContains('pages/contact.html', '<title>Contact │ Daniel Short');
['index.html','pages/contact.html','pages/portfolio.html','pages/contributions.html'].forEach(f => {
  checkFileContains(f, 'js/common/common.js');
  checkFileContains(f, 'class="skip-link"');
  checkFileContains(f, '<main id="main">');
});
['index.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','404.html'].forEach(f => {
  checkFileContains(f, 'og:image');
});
assert(fs.existsSync('robots.txt'), 'robots.txt missing');
assert(fs.existsSync('sitemap.xml'), 'sitemap.xml missing');

  const homeHtml = fs.readFileSync('index.html', 'utf8');
  assert(!homeHtml.includes('js/portfolio/modal-helpers.js'), 'index.html should lazy load portfolio modal helpers');
  assert(!homeHtml.includes('js/portfolio/projects-data.js'), 'index.html should lazy load portfolio data');
  const portfolioHtml = fs.readFileSync('pages/portfolio.html', 'utf8');
  assert(!portfolioHtml.includes('js/portfolio/modal-helpers.js'), 'pages/portfolio.html should defer portfolio modal helpers');
  assert(!portfolioHtml.includes('js/portfolio/portfolio.js'), 'pages/portfolio.html should rely on lazy loader');
  const commonCode = fs.readFileSync('js/common/common.js', 'utf8');
  assert(commonCode.includes('js/portfolio/projects-data.js'), 'common.js missing portfolio lazy loader');

  const htmlFiles = ['index.html','contact.html','resume.html','privacy.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/privacy.html'];
  htmlFiles.forEach(file => {
    const content = fs.readFileSync(file, 'utf8');
    assert(!content.includes('js/analytics/ga4-events.js'), `${file} should load analytics helpers on demand`);
  });
  const consentCode = fs.readFileSync('js/privacy/consent_manager.js', 'utf8');
  assert(consentCode.includes('ga4-helper'), 'consent manager should inject analytics helper script');

  // Data files expose arrays
  let env = evalScript('js/portfolio/projects-data.js');
  assert(Array.isArray(env.window.PROJECTS) && env.window.PROJECTS.length > 0,
         'projects-data.js failed to define PROJECTS');

  env = evalScript('js/contributions/contributions-data.js');
  assert(Array.isArray(env.window.contributions) && env.window.contributions.length > 0,
         'contributions-data.js failed to define contributions');

  // Analytics helpers
  env = evalScript('js/analytics/ga4-events.js');
  assert(typeof env.window.gaEvent === 'function', 'ga4-events.js missing gaEvent');
  assert(typeof env.window.trackProjectView === 'function', 'ga4-events.js missing trackProjectView');
  assert(typeof env.window.trackModalClose === 'function', 'ga4-events.js missing trackModalClose');
  // multi-project view event after three views
  {
    const startLen = (env.dataLayer || []).length;
    env.window.trackProjectView('alpha');
    env.window.trackProjectView('beta');
    env.window.trackProjectView('gamma');
    const evts = (env.dataLayer || []).slice(startLen).filter(x => x && x[0] === 'event');
    const hasMulti = evts.some(x => x[1] === 'multi_project_view');
    assert(hasMulti, 'multi_project_view event not emitted on third view');
  }
  checkFileContains('js/navigation/navigation.js', 'div id="primary-menu" class="nav-row"');

  // Header/brand structure and styling (ensure we don't regress)
  const navJs = fs.readFileSync('js/navigation/navigation.js', 'utf8');
  assert(navJs.includes('class="brand-logo"'), 'nav markup missing brand-logo');
  assert(navJs.includes('class="brand-name"'), 'nav markup missing brand-name');
  assert(navJs.includes('class="brand-line name"'), 'nav markup missing brand-line name');
  assert(navJs.includes('class="brand-line divider"'), 'nav markup missing vertical divider');
  assert(navJs.includes('class="brand-line tagline"'), 'nav markup missing tagline');

  // CSS: brand colors and stacked divider line
  const navCss = fs.readFileSync('css/layout/nav.css', 'utf8');
  assert(navCss.includes('.brand .divider{color:var(--primary)}'), 'nav.css divider not teal');
  assert(navCss.includes(".brand-name{\n    font-family:'Poppins'"), 'nav.css brand-name block missing');
  assert(navCss.includes('color:var(--text-light)'), 'nav.css brand-name not white');

  const utilCss = fs.readFileSync('css/utilities/layout.css', 'utf8');
  assert(utilCss.includes('.brand-line.tagline'), 'utilities/layout.css missing tagline selector');
  assert(utilCss.includes('color: var(--text-light);'), 'tagline not set to white');
  assert(utilCss.includes('.brand-line.name::after'), 'stacked horizontal rule missing');
  assert(utilCss.includes('background:var(--primary);'), 'stacked horizontal rule not teal');
  assert(/\.brand-logo\s*\{[^}]*height:56px;/.test(utilCss), 'mobile logo size not increased to 56px');

  // CSS variables: secondary should resolve to primary (teal)
  const varsCss = fs.readFileSync('css/variables.css', 'utf8');
  assert(/--secondary\s*:\s*var\(--primary\)\s*;/.test(varsCss), 'variables.css --secondary not mapped to --primary');

  // Components should not rely on var(--secondary) anymore
  const heroCss = fs.readFileSync('css/components/hero.css', 'utf8');
  assert(!heroCss.includes('var(--secondary)'), 'hero.css still references --secondary');
  const modalCss = fs.readFileSync('css/components/modal.css', 'utf8');
  assert(!modalCss.includes('var(--secondary)'), 'modal.css still references --secondary');

  // Core scripts should load without throwing
  [
    'js/common/common.js',
    'js/navigation/navigation.js',
    'js/animations/animations.js',
    'js/forms/contact.js',
    'js/portfolio/modal-helpers.js',
    'js/contributions/contributions.js',
    'js/contributions/carousel.js'
  ].forEach(file => evalScript(file));

  // Basic structure across key pages
  const cssManifestPath = 'dist/styles-manifest.json';
  assert(fs.existsSync(cssManifestPath), 'dist/styles-manifest.json missing');
  const cssManifest = JSON.parse(fs.readFileSync(cssManifestPath, 'utf8'));
  assert(cssManifest.file && /^styles\.[0-9a-f]{8}\.css$/.test(cssManifest.file), 'CSS manifest entry invalid');
  const hashedCss = cssManifest.file;
  assert(fs.existsSync(`dist/${hashedCss}`), `dist/${hashedCss} missing`);
  assert(fs.existsSync('dist/styles.css'), 'dist/styles.css missing');

['index.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','404.html','pages/privacy.html'].forEach(f => {
  checkFileContains(f, '<header id="combined-header-nav">');
  checkFileContains(f, '<main id="main"');
  checkFileContains(f, 'class="skip-link"');
  checkFileContains(f, 'name="viewport"');
  checkFileContains(f, 'name="theme-color"');
  const html = fs.readFileSync(f,'utf8');
  assert(
    html.includes(`dist/${hashedCss}`) || html.includes('dist/styles.css'),
    `${f} missing stylesheet reference`
  );
});

  // Fonts are preloaded on index
  checkFileContains('index.html', 'fonts.googleapis.com');
  checkFileContains('index.html', 'family=Poppins');

  // Navigation behavior: toggle body .menu-open and aria-controls
  const navCode = fs.readFileSync('js/navigation/navigation.js', 'utf8');
  assert(navCode.includes("classList.toggle('menu-open'"), 'burger toggle missing body.menu-open');
  assert(navCode.includes('aria-controls="primary-menu"'), 'burger missing aria-controls');
  assert(navCode.includes('aria-expanded'), 'burger missing aria-expanded');
  assert(navCode.includes('aria-current'), 'active nav link missing aria-current');
  assert(navCode.includes('getBoundingClientRect'), 'setNavHeight missing measurement');

  // CSS layer order present in styles.css
  const stylesCss = fs.readFileSync('css/styles.css', 'utf8');
  assert(stylesCss.includes('@layer tokens, base, layout, components, utilities, overrides;'), 'styles.css layer order missing');

  // Utilities contain mobile stacking rules and scroll offset
  assert(utilCss.includes('flex-direction: column;'), 'brand mobile stack rule missing');
  assert(utilCss.includes('display:none;') && utilCss.includes('.brand-line.divider'), 'mobile divider hide missing');
  assert(utilCss.includes('padding-top:var(--nav-height'), 'page offset for fixed header missing');
  assert(utilCss.includes('clip-path') && utilCss.includes('.nav-row.open'), 'mobile drawer clip-path reveal missing');

  // GA helpers: gtag shim and event helpers
  const ga = evalScript('js/analytics/ga4-events.js');
  assert(typeof ga.window.gtag === 'function', 'gtag shim not defined');
  assert(typeof ga.window.gaEvent === 'function', 'gaEvent not exposed');

  // Portfolio helpers surface functions without DOM
  let portfolioEnv = evalScript('js/portfolio/modal-helpers.js');
  portfolioEnv = evalScript('js/portfolio/portfolio.js', portfolioEnv);
  assert(typeof portfolioEnv.window.openModal === 'function', 'openModal not defined');
  assert(typeof portfolioEnv.window.closeModal === 'function', 'closeModal not defined');
  assert(typeof portfolioEnv.window.generateProjectModal === 'function', 'generateProjectModal not defined');

  // Data contracts: PROJECTS and contributions have unique IDs/titles
  const pdata = evalScript('js/portfolio/projects-data.js');
  const ids = new Set();
  pdata.window.PROJECTS.forEach(p => {
    assert(p.id && typeof p.id === 'string', 'project missing id');
    assert(!ids.has(p.id), 'duplicate project id: ' + p.id);
    ids.add(p.id);
    assert(p.title && p.title.length > 0, 'project missing title: ' + p.id);
  });

  const cdata = evalScript('js/contributions/contributions-data.js');
  assert(Array.isArray(cdata.window.contributions), 'contributions not an array');
  cdata.window.contributions.forEach(section => {
    assert(section.heading && section.items && Array.isArray(section.items), 'bad contributions section');
    section.items.forEach(it => assert(it.title && it.link, 'bad contribution item'));
  });

  // Robots and sitemap basic validation
  const robots = fs.readFileSync('robots.txt','utf8');
  assert(/User-agent:\s*\*/.test(robots), 'robots.txt missing user-agent');
  assert(/Sitemap:\s*https?:\/\//.test(robots), 'robots.txt missing sitemap URL');
  const sitemap = fs.readFileSync('sitemap.xml','utf8');
  assert(/<urlset/.test(sitemap) && /<loc>https:\/\/.+<\/loc>/.test(sitemap), 'sitemap.xml structure invalid');

  // Build scripts exist and include expected directories
  assert(fs.existsSync('build/build-css.js'), 'build-css.js missing');
  const copyJs = fs.readFileSync('build/copy-to-public.js','utf8');
  assert(copyJs.includes('const dirs') &&
         copyJs.includes("'img'") && copyJs.includes("'js'") && copyJs.includes("'css'") &&
         copyJs.includes("'documents'") && copyJs.includes("'dist'") &&
         copyJs.includes("'pages'") && copyJs.includes("'demos'"),
         'copy-to-public.js not copying all asset dirs');

  // Deployment config includes CSP and security headers
  const vercel = fs.readFileSync('vercel.json','utf8');
  assert(vercel.includes('Content-Security-Policy'), 'vercel.json missing CSP');
  assert(vercel.includes('Strict-Transport-Security'), 'vercel.json missing HSTS');
  // Also ensure /img cache header is present
  assert(vercel.includes('"source": "/img/(.*)"') || vercel.includes('"source": "/img/(.*)"'.replace(/\//g,'/')), 'vercel.json missing /img cache rule');
  // Rewrites must be extensionless destinations (avoid cleanUrls redirect loops)
  let vercelObj;
  try { vercelObj = JSON.parse(vercel); } catch {}
  const rewrites = (vercelObj && vercelObj.rewrites) || [];
  assert(rewrites.length > 0, 'vercel.json missing rewrites');
  const badDest = rewrites.filter(r => /\.html$/.test((r.destination||'')));
  assert(badDest.length === 0, 'rewrite destinations must be extensionless to avoid loops');
  // Ensure key rewrites exist
  const hasPortfolio = rewrites.some(r => r.source === '/portfolio' && r.destination === '/pages/portfolio');
  const hasPortfolioHtml = rewrites.some(r => r.source === '/portfolio.html' && r.destination === '/pages/portfolio');
  assert(hasPortfolio && hasPortfolioHtml, 'portfolio rewrites missing');

  // Chatbot demo should tolerate backend startup delays up to ten minutes
  const chatbotHtml = fs.readFileSync('demos/chatbot-demo.html', 'utf8');
  const startConst = chatbotHtml.match(/const START_TIMEOUT_SEC = (\d+);/);
  const warmSec = startConst ? parseInt(startConst[1], 10) : 0;
  assert(warmSec >= 600, 'chatbot-demo start timeout < 10 minutes');

  // Extract and execute the starting timer helpers to simulate long startups
  const startSection = chatbotHtml.match(/\/\/ Starting timer helpers[\s\S]*?\/\/ End starting timer helpers/);
  assert(startSection, 'chatbot-demo starting timer section missing');
  const warmEnv = {
    startingTimer: null,
    startingDeadline: 0,
    START_TIMEOUT_SEC: warmSec,
    STATE: {
      OFFLINE_DETECTED: 'OFFLINE_DETECTED',
      STARTING: 'STARTING',
      ONLINE: 'ONLINE',
      ONLINE_GRACE: 'ONLINE_GRACE',
      SHUTDOWN: 'SHUTDOWN'
    },
    Date: { now: () => 0 },
    Math,
    svcText: { textContent: '' },
    svcETA: { textContent: '' },
    setDot: () => {},
    startGraceCycle: () => {},
    fmtClock: () => '',
    setInterval: () => 1,
    clearInterval: () => {}
  };
  vm.runInNewContext(startSection[0], warmEnv);
  warmEnv.startStartingTimer();
  warmEnv.Date.now = () => 599 * 1000; // 9m59s later
  assert(warmEnv.currentStartingRemaining() > 0, 'starting countdown ended too early');
  warmEnv.Date.now = () => 601 * 1000; // just past 10m
  assert(warmEnv.currentStartingRemaining() === 0, 'starting countdown did not finish after ten minutes');

  // 404 rewrite should redirect clean portfolio slugs
  checkFileContains('404.html', 'portfolio.html?project=');

  // Portfolio page core regions present
  checkFileContains('pages/portfolio.html', 'id="portfolio-carousel"');
  checkFileContains('pages/portfolio.html', 'id="filter-menu"');
  checkFileContains('pages/portfolio.html', 'id="projects"');
  checkFileContains('pages/portfolio.html', 'id="modals"');
  checkFileContains('pages/portfolio.html', 'id="see-more"');

  // Moved pages/demos should include a base href for asset resolution
  ['pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html',
   'demos/chatbot-demo.html','demos/shape-demo.html','demos/sentence-demo.html']
    .forEach(f => checkFileContains(f, '<base href="/">'));

  // Sanity: Avoid accidental client-side redirect loops on pages (exclude 404)
  ;['pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html']
    .forEach(f => {
      const html = fs.readFileSync(f,'utf8');
      if (/http-equiv\s*=\s*"refresh"/i.test(html)) throw new Error(f+': should not use meta refresh');
      if (/location\.replace\(/.test(html)) throw new Error(f+': should not call location.replace');
    });

  // Modal CSS: sentence demo should not allow horizontal scroll
  const distCss = fs.readFileSync('dist/styles.css','utf8');
  assert(distCss.includes('#smartSentence-modal .modal-body{overflow-x:hidden}'), 'sentence modal missing overflow-x hidden');

  // Contact modal and resume embed present
  checkFileContains('pages/contact.html', 'id="contact-modal"');
  checkFileContains('resume.html', 'documents/Resume.pdf');
  // Contact: embed is a Google Form
  checkFileContains('contact.html', 'docs.google.com/forms');
  const contactCss = fs.readFileSync('css/components/modal.css','utf8');
  assert(/#contact-modal\s+iframe[\s\S]*height:100vh;/.test(contactCss), 'contact modal iframe not set to 100vh');

  // Privacy page includes CMP scripts and GA4 vendor id exists
  checkFileContains('privacy.html', 'js/privacy/config.js');
  checkFileContains('privacy.html', 'js/privacy/consent_manager.js');
  const pcfg = evalScript('js/privacy/config.js');
  assert(pcfg.window.PrivacyConfig && pcfg.window.PrivacyConfig.vendors && pcfg.window.PrivacyConfig.vendors.ga4 && pcfg.window.PrivacyConfig.vendors.ga4.id,
         'PrivacyConfig missing GA4 vendor id');

  // Portfolio URL parsing should support both ?project= and #hash formats
  let pEnv = evalScript('js/portfolio/modal-helpers.js');
  pEnv = evalScript('js/portfolio/portfolio.js', pEnv);
  // 1) query param format
  pEnv.location.search = '?project=chatbotLora';
  pEnv.location.hash = '';
  assert(typeof pEnv.window.__portfolio_getIdFromURL === 'function', 'portfolio test hook missing');
  assert(pEnv.window.__portfolio_getIdFromURL() === 'chatbotLora', 'portfolio ?project parsing failed');
  // 2) legacy hash format
  pEnv.location.search = '';
  pEnv.location.hash = '#shapeClassifier';
  assert(pEnv.window.__portfolio_getIdFromURL() === 'shapeClassifier', 'portfolio #hash parsing failed');

  // 3) modal template generation (image-only project)
  const modalHtml = pEnv.window.generateProjectModal({
    id:'t1', title:'T', subtitle:'S', problem:'P',
    image:'img/x.png', tools:[], resources:[], actions:[], results:[]
  });
  assert(/modal-image/.test(modalHtml), 'modal image block missing');
  assert(/<picture>/.test(modalHtml), 'PNG should render with <picture> WebP fallback');

  // 4) modal template generation (tableau embed uses data-base and wide layout)
  const tabHtml = pEnv.window.generateProjectModal({
    id:'t2', title:'Tab', subtitle:'Sub', problem:'Prob',
    tools:[], resources:[], actions:[], results:[],
    embed:{ type:'tableau', base:'https://public.tableau.com/views/Example/Sheet' }
  });
  assert(/class=\"modal-embed tableau-fit\"/.test(tabHtml), 'tableau modal should use wide layout');
  assert(/<iframe[\s\S]*data-base=/.test(tabHtml), 'tableau iframe should use data-base attribute');

  console.log('All tests passed.');
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
