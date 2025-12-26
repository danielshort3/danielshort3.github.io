// Lightweight test runner used by `npm test`
const fs = require('fs');
const vm = require('vm');
const runSlotDemoTests = require('./tests/slot-machine-demo.test.js');

// Assert helper
let assertCount = 0;
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
  assertCount++;
}

// Verify an HTML file contains a required snippet
function checkFileContains(file, text) {
  assert(fs.existsSync(file), `${file} does not exist`);
  const content = fs.readFileSync(file, 'utf8');
  assert(content.includes(text), `${file} missing expected text: ${text}`);
}

function section(name, fn) {
  console.log(`\n• ${name}`);
  const before = assertCount;
  try {
    const result = fn();
    const ran = assertCount - before;
    console.log(`  ✔ ${name}${ran ? ` (${ran} checks)` : ''}`);
    return result;
  } catch (err) {
    console.error(`  ✖ ${name}`);
    throw err;
  }
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
  console.log('Running site contract checks...');

  let hashedCss;

  section('Page shells and required meta', () => {
    checkFileContains('index.html', 'made actionable');
    checkFileContains('pages/contact.html', '<title>Daniel Short - Contact');
    checkFileContains('pages/tools.html', '<title>Tools | Daniel Short');
    checkFileContains('pages/point-of-view-checker.html', '<title>Point of View Checker | Daniel Short');
    checkFileContains('pages/oxford-comma-checker.html', '<title>Oxford Comma Checker | Daniel Short');
    checkFileContains('pages/ocean-wave-simulation.html', '<title>Ocean Wave Simulation | Daniel Short');
    checkFileContains('pages/qr-code-generator.html', '<title>QR Code Generator | Daniel Short');
    checkFileContains('pages/image-optimizer.html', '<title>Image Optimizer | Daniel Short');
    ['index.html','pages/contact.html','pages/portfolio.html','pages/contributions.html'].forEach(f => {
      checkFileContains(f, 'js/common/common.js');
      checkFileContains(f, 'class="skip-link"');
      checkFileContains(f, '<main id="main">');
    });
    ['pages/tools.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/screen-recorder.html','pages/job-application-tracker.html'].forEach(f => {
      checkFileContains(f, 'js/common/common.js');
      checkFileContains(f, 'class="skip-link"');
      checkFileContains(f, '<main id="main">');
      checkFileContains(f, 'noindex, nofollow');
    });
    ['index.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','pages/tools.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','404.html'].forEach(f => {
      checkFileContains(f, 'og:image');
    });
    assert(fs.existsSync('robots.txt'), 'robots.txt missing');
    assert(fs.existsSync('sitemap.xml'), 'sitemap.xml missing');
    const robots = fs.readFileSync('robots.txt','utf8');
    assert(/User-agent:\s*\*/.test(robots), 'robots.txt missing user-agent');
    assert(/Sitemap:\s*https?:\/\//.test(robots), 'robots.txt missing sitemap URL');
    const sitemap = fs.readFileSync('sitemap.xml','utf8');
    assert(/<urlset/.test(sitemap) && /<loc>https:\/\/.+<\/loc>/.test(sitemap), 'sitemap.xml structure invalid');
  });

  section('Lazy loading and analytics defers', () => {
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
  });

  section('Job tracker UI additions', () => {
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-import-file"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="import-submit"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-resume"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-cover"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-posting-date"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-posting-unknown"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="recent-status"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack-tab="applications"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack-tab="prospects"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="prospect-form"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-prospect-url"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-prospect-posting-date"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-prospect-posting-unknown"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-prospect-capture-date"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="prospect-list"');
  });

  section('Data contracts', () => {
    let env = evalScript('js/portfolio/projects-data.js');
    assert(Array.isArray(env.window.PROJECTS) && env.window.PROJECTS.length > 0,
           'projects-data.js failed to define PROJECTS');

    env = evalScript('js/contributions/contributions-data.js');
    assert(Array.isArray(env.window.contributions) && env.window.contributions.length > 0,
           'contributions-data.js failed to define contributions');

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
  });

  section('Project pages and sitemap entries', () => {
    const pdata = evalScript('js/portfolio/projects-data.js');
    const ids = pdata.window.PROJECTS.map(p => p.id);
    assert(ids.length > 0, 'no project ids found');

    const sitemap = fs.readFileSync('sitemap.xml', 'utf8');
    ids.forEach(id => {
      const file = `pages/portfolio/${id}.html`;
      assert(fs.existsSync(file), `${file} missing`);
      const html = fs.readFileSync(file, 'utf8');
      checkFileContains(file, '<base href="/">');
      checkFileContains(file, 'data-page="project"');
      checkFileContains(file, '<meta property="og:type" content="article">');
      checkFileContains(file, `href="portfolio.html?project=${encodeURIComponent(id)}`);
      checkFileContains(file, `<link rel="canonical" href="https://danielshort.me/portfolio/${id}">`);
      checkFileContains(file, `<meta property="og:url" content="https://danielshort.me/portfolio/${id}">`);
      assert(sitemap.includes(`https://danielshort.me/portfolio/${id}`), `sitemap.xml missing project url: ${id}`);
    });
  });

  section('Analytics helpers and events', () => {
    const env = evalScript('js/analytics/ga4-events.js');
    assert(typeof env.window.gaEvent === 'function', 'ga4-events.js missing gaEvent');
    assert(typeof env.window.trackProjectView === 'function', 'ga4-events.js missing trackProjectView');
    assert(typeof env.window.trackModalClose === 'function', 'ga4-events.js missing trackModalClose');
    const startLen = (env.dataLayer || []).length;
    env.window.trackProjectView('alpha');
    env.window.trackProjectView('beta');
    env.window.trackProjectView('gamma');
    const evts = (env.dataLayer || []).slice(startLen).filter(x => x && x[0] === 'event');
    const hasMulti = evts.some(x => x[1] === 'multi_project_view');
    assert(hasMulti, 'multi_project_view event not emitted on third view');
    assert(typeof env.window.gtag === 'function', 'gtag shim not defined');
  });

  section('Portfolio modal analytics hook', () => {
    const modalEnv = createEnv();
    modalEnv.__tracked = null;
    modalEnv.window.trackProjectView = id => { modalEnv.__tracked = id; };
    modalEnv.document.body.classList = { add() {}, remove() {} };
    const contentNode = { focus() {}, querySelectorAll: () => [], addEventListener() {} };
    const modalNode = {
      id: 'proj1-modal',
      classList: { add() {}, remove() {}, contains() { return true; } },
      dataset: {},
      addEventListener() {},
      querySelector(selector) {
        if (selector === '.modal-content') return contentNode;
        if (selector === '.modal-embed iframe') return null;
        if (selector === '.gif-video') return null;
        return null;
      },
      querySelectorAll() { return []; },
      focus() {}
    };
    modalEnv.document.getElementById = (id) => id === 'proj1-modal' ? modalNode : null;
    modalEnv.document.querySelector = () => null;
    modalEnv.history.replaceState = () => {};
    modalEnv.location = { hash: '' };
    const mCtx = evalScript('js/portfolio/modal-helpers.js', modalEnv);
    mCtx.window.openModal('proj1');
    assert(mCtx.__tracked === 'proj1', 'openModal should call trackProjectView');
  });

  section('Navigation markup and branding', () => {
    checkFileContains('js/navigation/navigation.js', 'div id="primary-menu" class="nav-row"');
    const navJs = fs.readFileSync('js/navigation/navigation.js', 'utf8');
    assert(navJs.includes('class="brand-logo"'), 'nav markup missing brand-logo');
    assert(navJs.includes('class="brand-name"'), 'nav markup missing brand-name');
    assert(navJs.includes('class="brand-title"'), 'nav markup missing brand-title');
    assert(navJs.includes('class="brand-divider"'), 'nav markup missing brand-divider');
    assert(navJs.includes('class="brand-tagline"'), 'nav markup missing brand-tagline');
    assert(navJs.includes('class="brand-tagline-chunk">Data Science'), 'nav markup missing tagline chunk for Data Science');
  });

  section('Navigation CSS and mobile layout', () => {
    const navCss = fs.readFileSync('css/layout/nav.css', 'utf8');
    assert(navCss.includes('--brand-logo-size'), 'nav.css missing brand logo scale variable');
    assert(navCss.includes('.brand-divider'), 'nav.css missing brand-divider rules');
    assert(navCss.includes('.brand-tagline'), 'nav.css missing brand-tagline rules');
    assert(navCss.includes('flex-wrap:wrap;'), 'nav.css tagline should wrap whole chunks');

    const utilCss = fs.readFileSync('css/utilities/layout.css', 'utf8');
    assert(utilCss.includes('--brand-title-size'), 'utilities/layout.css missing mobile brand sizing overrides');
    assert(utilCss.includes('.brand-divider'), 'utilities/layout.css missing mobile divider override');
    assert(/flex-direction\s*:\s*column;/.test(utilCss), 'brand mobile stack rule missing');
    assert(utilCss.includes('background:linear-gradient(90deg'), 'mobile divider gradient missing');
    assert(utilCss.includes('padding-top:var(--nav-height'), 'page offset for fixed header missing');
    assert(utilCss.includes('clip-path') && utilCss.includes('.nav-row.open'), 'mobile drawer clip-path reveal missing');
  });

  section('CSS tokens, layers, and components', () => {
    const varsCss = fs.readFileSync('css/variables.css', 'utf8');
    assert(/--secondary\s*:\s*var\(--primary\)\s*;/.test(varsCss), 'variables.css --secondary not mapped to --primary');

    const heroCss = fs.readFileSync('css/components/hero.css', 'utf8');
    assert(!heroCss.includes('var(--secondary)'), 'hero.css still references --secondary');
    const modalCss = fs.readFileSync('css/components/modal.css', 'utf8');
    assert(!modalCss.includes('var(--secondary)'), 'modal.css still references --secondary');
    assert(modalCss.includes('.contact-form input') && modalCss.includes('.contact-form-status'), 'contact modal form styles missing');

    const stylesCss = fs.readFileSync('css/styles.css', 'utf8');
    assert(stylesCss.includes('@layer tokens, base, layout, components, utilities, overrides;'), 'styles.css layer order missing');
  });

  section('Core scripts load without DOM', () => {
    [
      'js/common/common.js',
      'js/navigation/navigation.js',
      'js/animations/animations.js',
      'js/forms/contact.js',
      'js/portfolio/modal-helpers.js',
      'js/contributions/contributions.js',
      'js/contributions/carousel.js'
    ].forEach(file => evalScript(file));
  });

  section('CSS bundle manifest and page references', () => {
    const cssManifestPath = 'dist/styles-manifest.json';
    assert(fs.existsSync(cssManifestPath), 'dist/styles-manifest.json missing');
    const cssManifest = JSON.parse(fs.readFileSync(cssManifestPath, 'utf8'));
    assert(cssManifest.file && /^styles\.[0-9a-f]{8}\.css$/.test(cssManifest.file), 'CSS manifest entry invalid');
    hashedCss = cssManifest.file;
    assert(fs.existsSync(`dist/${hashedCss}`), `dist/${hashedCss} missing`);
    assert(fs.existsSync('dist/styles.css'), 'dist/styles.css missing');

    const projectIds = evalScript('js/portfolio/projects-data.js').window.PROJECTS.map(p => p.id);
    const projectPages = projectIds.map(id => `pages/portfolio/${id}.html`);
    const toolPages = ['pages/tools.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html'];
    ['index.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/privacy.html','404.html', ...toolPages, ...projectPages].forEach(f => {
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
  });

  section('Fonts and navigation behavior', () => {
    checkFileContains('index.html', 'fonts.googleapis.com');
    checkFileContains('index.html', 'family=Poppins');

    const navCode = fs.readFileSync('js/navigation/navigation.js', 'utf8');
    assert(navCode.includes("classList.toggle('menu-open'"), 'burger toggle missing body.menu-open');
    assert(navCode.includes('aria-controls="primary-menu"'), 'burger missing aria-controls');
    assert(navCode.includes('aria-expanded'), 'burger missing aria-expanded');
    assert(navCode.includes('aria-current'), 'active nav link missing aria-current');
    assert(navCode.includes('getBoundingClientRect'), 'setNavHeight missing measurement');
  });

  section('Contributions interactions emit analytics', () => {
    const root = {
      _handlers: {},
      dataset: {},
      classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
      addEventListener(type, fn) {
        this._handlers[type] = this._handlers[type] || [];
        this._handlers[type].push(fn);
      },
      removeEventListener() {},
      appendChild() {},
      innerHTML: '',
    };
    const contribEnv = createEnv();
    contribEnv.__events = [];
    contribEnv.window.gaEvent = (name, params) => contribEnv.__events.push({ name, params });
    evalScript('js/contributions/contributions-data.js', contribEnv);
    contribEnv.document.getElementById = (id) => id === 'contrib-root' ? root : null;
    contribEnv.document.createElement = (tag) => ({
      tagName: String(tag || '').toUpperCase(),
      children: [],
      dataset: {},
      className: '',
      classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
      setAttribute() {},
      appendChild(child) { this.children.push(child); child.parent = this; },
      insertAdjacentHTML() {},
      querySelector() { return null; },
      querySelectorAll() { return []; },
      addEventListener() {},
      removeEventListener() {},
      focus() {}
    });
    contribEnv.document.querySelector = () => null;
    contribEnv.document.body.classList = { add() {}, remove() {} };
    const ctx = evalScript('js/contributions/contributions.js', contribEnv);
    if (typeof ctx.initContributions === 'function') {
      ctx.initContributions();
    }
    assert(root._handlers.click && root._handlers.click.length, 'contributions click handler not bound');
    assert(root._handlers.toggle && root._handlers.toggle.length, 'contributions toggle handler not bound');
    const sectionNode = { dataset: { heading: 'Public Reports' }, classList: { contains() { return false; } }, closest: () => sectionNode };
    const link = {
      dataset: { section: 'Public Reports', title: 'Budget', kind: 'pdf' },
      getAttribute() { return ''; },
      closest(sel) {
        if (sel === '.doc-links a') return this;
        if (sel === '.contrib-section') return sectionNode;
        return null;
      }
    };
    root._handlers.click[0]({ target: link });
    const docEvt = contribEnv.__events.find(e => e.name === 'contrib_doc_click');
    assert(docEvt, 'contrib_doc_click not emitted');
    assert(docEvt.params.section === 'Public Reports' && docEvt.params.title === 'Budget', 'contrib_doc_click params incorrect');
    const details = {
      tagName: 'DETAILS',
      dataset: { year: '2024' },
      open: true,
      classList: { contains(cls) { return cls === 'timeline-year'; } },
      closest() { return sectionNode; }
    };
    root._handlers.toggle[0]({ target: details });
    const toggleEvt = contribEnv.__events.find(e => e.name === 'contrib_timeline_toggle');
    assert(toggleEvt && toggleEvt.params.year === '2024' && toggleEvt.params.expanded === true,
      'contrib_timeline_toggle params incorrect');
  });

  section('Build and deployment configuration', () => {
    assert(fs.existsSync('build/build-css.js'), 'build-css.js missing');
    assert(fs.existsSync('build/generate-project-pages.js'), 'generate-project-pages.js missing');
    const copyJs = fs.readFileSync('build/copy-to-public.js','utf8');
    assert(copyJs.includes('const dirs') &&
           copyJs.includes("'img'") && copyJs.includes("'js'") && copyJs.includes("'css'") &&
           copyJs.includes("'documents'") && copyJs.includes("'dist'") &&
           copyJs.includes("'pages'") && copyJs.includes("'demos'"),
           'copy-to-public.js not copying all asset dirs');

    const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    assert(pkg.scripts && pkg.scripts['build:projects'], 'package.json missing build:projects script');
    assert(pkg.scripts.build && pkg.scripts.build.includes('build:projects'), 'package.json build script should run build:projects');

    const vercel = fs.readFileSync('vercel.json','utf8');
    assert(vercel.includes('Content-Security-Policy'), 'vercel.json missing CSP');
    assert(vercel.includes('Strict-Transport-Security'), 'vercel.json missing HSTS');
    assert(vercel.includes('"source": "/img/(.*)"') || vercel.includes('"source": "/img/(.*)"'.replace(/\//g,'/')), 'vercel.json missing /img cache rule');
    let vercelObj;
    try { vercelObj = JSON.parse(vercel); } catch {}
    const rewrites = (vercelObj && vercelObj.rewrites) || [];
    assert(rewrites.length > 0, 'vercel.json missing rewrites');
    const badDest = rewrites.filter(r => /\.html$/.test((r.destination||'')));
    assert(badDest.length === 0, 'rewrite destinations must be extensionless to avoid loops');
    const hasPortfolio = rewrites.some(r => r.source === '/portfolio' && r.destination === '/pages/portfolio');
    const hasPortfolioHtml = rewrites.some(r => r.source === '/portfolio.html' && r.destination === '/pages/portfolio');
    const hasProjectRewrite = rewrites.some(r => r.source === '/portfolio/:project' && r.destination === '/pages/portfolio/:project');
    assert(hasPortfolio && hasPortfolioHtml, 'portfolio rewrites missing');
    assert(hasProjectRewrite, 'project rewrite missing (/portfolio/:project)');
  });

  section('Chatbot demo startup timer', () => {
    const chatbotHtml = fs.readFileSync('demos/chatbot-demo.html', 'utf8');
    const startConst = chatbotHtml.match(/const START_TIMEOUT_SEC = (\d+);/);
    const warmSec = startConst ? parseInt(startConst[1], 10) : 0;
    assert(warmSec >= 600, 'chatbot-demo start timeout < 10 minutes');

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
    warmEnv.Date.now = () => 599 * 1000;
    assert(warmEnv.currentStartingRemaining() > 0, 'starting countdown ended too early');
    warmEnv.Date.now = () => 601 * 1000;
    assert(warmEnv.currentStartingRemaining() === 0, 'starting countdown did not finish after ten minutes');
  });

  section('404 rewrites and portfolio page sections', () => {
    checkFileContains('404.html', 'portfolio.html?project=');
    checkFileContains('pages/portfolio.html', 'id="portfolio-carousel"');
    checkFileContains('pages/portfolio.html', 'id="filter-menu"');
    checkFileContains('pages/portfolio.html', 'id="projects"');
    checkFileContains('pages/portfolio.html', 'id="modals"');
    checkFileContains('pages/portfolio.html', 'id="see-more"');
  });

  section('Base hrefs and redirect sanity', () => {
    ['pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html',
     'pages/tools.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html',
     'demos/chatbot-demo.html','demos/shape-demo.html','demos/sentence-demo.html']
      .forEach(f => checkFileContains(f, '<base href="/">'));

    ['pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html']
      .forEach(f => {
        const html = fs.readFileSync(f,'utf8');
        if (/http-equiv\s*=\s*"refresh"/i.test(html)) throw new Error(f+': should not use meta refresh');
        if (/location\.replace\(/.test(html)) throw new Error(f+': should not call location.replace');
      });
  });

  section('Modals, contact, and resume assets', () => {
    const distCss = fs.readFileSync('dist/styles.css','utf8');
    assert(distCss.includes('#smartSentence-modal .modal-body{overflow-x:hidden}'), 'sentence modal missing overflow-x hidden');

    checkFileContains('pages/contact.html', 'id="contact-modal"');
    checkFileContains('resume.html', 'documents/Resume.pdf');
    checkFileContains('contact.html', 'id="contact-form"');
  });

  section('Privacy CMP', () => {
    checkFileContains('privacy.html', 'js/privacy/config.js');
    checkFileContains('privacy.html', 'js/privacy/consent_manager.js');
    const pcfg = evalScript('js/privacy/config.js');
    assert(pcfg.window.PrivacyConfig && pcfg.window.PrivacyConfig.vendors && pcfg.window.PrivacyConfig.vendors.ga4 && pcfg.window.PrivacyConfig.vendors.ga4.id,
           'PrivacyConfig missing GA4 vendor id');
  });

  section('Portfolio helpers, URL parsing, and templates', () => {
    let pEnv = evalScript('js/portfolio/modal-helpers.js');
    pEnv = evalScript('js/portfolio/portfolio.js', pEnv);
    assert(typeof pEnv.window.openModal === 'function', 'openModal not defined');
    assert(typeof pEnv.window.closeModal === 'function', 'closeModal not defined');
    assert(typeof pEnv.window.generateProjectModal === 'function', 'generateProjectModal not defined');
    assert(typeof pEnv.window.__portfolio_getIdFromURL === 'function', 'portfolio test hook missing');

    pEnv.location.search = '?project=chatbotLora';
    pEnv.location.hash = '';
    assert(pEnv.window.__portfolio_getIdFromURL() === 'chatbotLora', 'portfolio ?project parsing failed');
    pEnv.location.search = '';
    pEnv.location.hash = '#shapeClassifier';
    assert(pEnv.window.__portfolio_getIdFromURL() === 'shapeClassifier', 'portfolio #hash parsing failed');

    const modalHtml = pEnv.window.generateProjectModal({
      id:'t1', title:'T', subtitle:'S', problem:'P',
      image:'img/x.png', tools:[], resources:[], actions:[], results:[]
    });
    assert(/modal-image/.test(modalHtml), 'modal image block missing');
    assert(/<picture>/.test(modalHtml), 'PNG should render with <picture> WebP fallback');

    const tabHtml = pEnv.window.generateProjectModal({
      id:'t2', title:'Tab', subtitle:'Sub', problem:'Prob',
      tools:[], resources:[], actions:[], results:[],
      embed:{ type:'tableau', base:'https://public.tableau.com/views/Example/Sheet' }
    });
    assert(/class=\"modal-embed tableau-fit\"/.test(tabHtml), 'tableau modal should use wide layout');
    assert(/<iframe[\s\S]*data-base=/.test(tabHtml), 'tableau iframe should use data-base attribute');

    const modalHelpersCode = fs.readFileSync('js/portfolio/modal-helpers.js', 'utf8');
    assert(/`\$\{origin\}\/portfolio\/\$\{encodeURIComponent\(id\)\}`/.test(modalHelpersCode),
      'modal copy-link should prefer /portfolio/<id> canonical URLs');
  });

  section('Slot machine demo', () => {
    runSlotDemoTests({ assert, checkFileContains });
  });

  console.log(`\nAll tests passed. Total checks: ${assertCount}`);
  process.exit(0);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
