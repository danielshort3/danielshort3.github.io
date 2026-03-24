// Lightweight test runner used by `npm test`
const fs = require('fs');
const vm = require('vm');
const runSlotDemoTests = require('./tests/slot-machine-demo.test.js');
const runUtmBatchBuilderTests = require('./tests/utm-batch-builder.test.js');
const runQrCodeGeneratorUtilsTests = require('./tests/qr-code-generator-utils.test.js');

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

function readFile(file) {
  assert(fs.existsSync(file), `${file} does not exist`);
  return fs.readFileSync(file, 'utf8');
}

function extractTitle(file, html) {
  const titleMatch = html.match(/<title>([^<]+)<\/title>/i);
  assert(titleMatch, `${file} missing <title> tag`);
  return String(titleMatch[1] || '').trim();
}

function assertBodyShellContract(file, html) {
  const bodyMatch = html.match(/<body\b([^>]*)>/i);
  assert(bodyMatch, `${file} missing <body> tag`);
  const bodyAttrs = String(bodyMatch[1] || '');
  assert(/\bdata-page="[^"]+"/.test(bodyAttrs), `${file} body missing data-page attribute`);
  assert(/\bclass="[^"]*\bsite-page\b[^"]*"/.test(bodyAttrs), `${file} body missing site-page class`);
  checkFileContains(file, 'class="skip-link"');
  assert(/<main[^>]*\bid="main"[^>]*>/i.test(html), `${file} missing <main id="main">`);
}

function assertHeroVariantClasses(file, html) {
  const heroSections = Array.from(html.matchAll(/<section[^>]*class="([^"]*\bhero\b[^"]*)"/gi));
  heroSections.forEach((match) => {
    const classNames = String(match[1] || '').trim();
    assert(
      /\bhero--(?:default|tools|games)\b/.test(classNames),
      `${file} hero section missing variant class: ${classNames}`
    );
  });
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
    checkFileContains('index.html', 'Made Actionable');

    const expectedTitles = {
      'index.html': 'Daniel Short | Tourism &amp; Destination Analytics Made Actionable',
      'pages/destination-analytics.html': 'Destination Analytics | Daniel Short',
      'pages/contact.html': 'Contact | Daniel Short',
      'pages/tools.html': 'Tools | Daniel Short',
      'pages/tools-dashboard.html': 'Tools Dashboard | Daniel Short',
      'pages/games.html': 'Games | Daniel Short',
      'pages/sitemap.html': 'Sitemap | Daniel Short',
      'pages/point-of-view-checker.html': 'Point of View Checker | Daniel Short',
      'pages/oxford-comma-checker.html': 'Oxford Comma Checker | Daniel Short',
      'pages/ocean-wave-simulation.html': 'Ocean Wave Simulation | Daniel Short',
      'pages/qr-code-generator.html': 'QR Code Generator | Daniel Short',
      'pages/image-optimizer.html': 'Image Optimizer | Daniel Short',
      'pages/utm-batch-builder.html': 'UTM Batch Builder | Daniel Short',
      'pages/whisper-transcribe-monitor.html': 'Whisper Capacity Monitor | Daniel Short',
      'pages/ga4-utm-performance.html': 'GA4 UTM Performance | Daniel Short',
      'probability-engine.html': 'Probability Engine | Daniel Short'
    };
    Object.entries(expectedTitles).forEach(([file, title]) => {
      checkFileContains(file, `<title>${title}</title>`);
    });

    const titleConventionPages = [
      ...Object.keys(expectedTitles),
      '404.html',
      'dshort.html',
      'pages/portfolio.html',
      'pages/contributions.html',
      'pages/privacy.html'
    ];
    titleConventionPages.forEach((file) => {
      const html = readFile(file);
      const title = extractTitle(file, html);
      assert(title.includes(' | '), `${file} title should use pipe separator`);
      assert(!title.includes(' - '), `${file} title should not use dash separator`);
      if (file === 'index.html') {
        assert(/^Daniel Short \| .+/.test(title), 'index.html title should start with Daniel Short |');
      } else {
        assert(/\| Daniel Short$/.test(title), `${file} title should end with | Daniel Short`);
      }
    });

    checkFileContains('pages/games.html', 'href="games/roulette"');
    checkFileContains('probability-engine.html', '<link rel="canonical" href="https://www.danielshort.me/games/probability-engine">');
    checkFileContains('probability-engine.html', '<meta name="description"');

    const toolPages = [
      'pages/tools.html',
      'pages/tools-dashboard.html',
      'pages/word-frequency.html',
      'pages/text-compare.html',
      'pages/point-of-view-checker.html',
      'pages/oxford-comma-checker.html',
      'pages/background-remover.html',
      'pages/nbsp-cleaner.html',
      'pages/qr-code-generator.html',
      'pages/image-optimizer.html',
      'pages/screen-recorder.html',
      'pages/job-application-tracker.html',
      'pages/short-links.html',
      'pages/utm-batch-builder.html',
      'pages/ga4-utm-performance.html',
      'pages/whisper-transcribe-monitor.html'
    ];
    const privateToolPages = [
      'pages/tools-dashboard.html',
      'pages/short-links.html',
      'pages/ga4-utm-performance.html',
      'pages/whisper-transcribe-monitor.html'
    ];

    const shellPages = [
      'index.html',
      'pages/destination-analytics.html',
      'pages/contact.html',
      'pages/portfolio.html',
      'pages/contributions.html',
      'pages/sitemap.html',
      'pages/games.html',
      'pages/ocean-wave-simulation.html',
      'pages/privacy.html',
      '404.html',
      'dshort.html',
      'probability-engine.html',
      ...toolPages
    ];
    shellPages.forEach((file) => {
      const html = readFile(file);
      assertBodyShellContract(file, html);
      assertHeroVariantClasses(file, html);
    });

    ['index.html','pages/destination-analytics.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','pages/sitemap.html'].forEach((f) => {
      checkFileContains(f, 'js/common/common.js');
    });
    ['pages/games.html','pages/ocean-wave-simulation.html', ...toolPages].forEach((f) => {
      checkFileContains(f, 'js/common/common.js');
    });

    ['pages/games.html','pages/ocean-wave-simulation.html','404.html','dshort.html', ...privateToolPages].forEach((f) => {
      checkFileContains(f, 'noindex, nofollow');
    });

    ['404.html', 'pages/privacy.html'].forEach((f) => {
      const html = readFile(f);
      const skipIndex = html.indexOf('class="skip-link"');
      const headerIndex = html.indexOf('<header id="combined-header-nav">');
      assert(skipIndex >= 0 && headerIndex >= 0, `${f} missing skip-link or shared header`);
      assert(skipIndex < headerIndex, `${f} skip-link should appear before shared header`);
    });

    toolPages.filter((f) => !privateToolPages.includes(f)).forEach((f) => {
      const content = readFile(f);
      assert(!content.includes('noindex, nofollow'), `${f} should be indexable`);
    });

    assert(!readFile('pages/tools-dashboard.html').includes('id="tool-jsonld"'),
      'tools dashboard should not include WebApplication JSON-LD');

    ['index.html','pages/destination-analytics.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','pages/tools.html','pages/games.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/utm-batch-builder.html','404.html'].forEach((f) => {
      checkFileContains(f, 'og:image');
    });

    assert(fs.existsSync('build/route-component-styles.json'), 'route component styles manifest missing');
    const injectHeadMetadataCode = readFile('build/inject-head-metadata.js');
    assert(injectHeadMetadataCode.includes('route-component-styles.json'),
      'inject-head-metadata.js should load the route component styles manifest');

    assert(fs.existsSync('robots.txt'), 'robots.txt missing');
    assert(fs.existsSync('sitemap.xml'), 'sitemap.xml missing');
    assert(fs.existsSync('sitemap.xsl'), 'sitemap.xsl missing');
    const robots = readFile('robots.txt');
    assert(/User-agent:\s*\*/.test(robots), 'robots.txt missing user-agent');
    assert(/Sitemap:\s*https?:\/\//.test(robots), 'robots.txt missing sitemap URL');
    const sitemap = readFile('sitemap.xml');
    assert(!/xml-stylesheet/i.test(sitemap), 'sitemap.xml should not include xml-stylesheet');
    assert(/<urlset/.test(sitemap) && /<loc>https:\/\/.+<\/loc>/.test(sitemap), 'sitemap.xml structure invalid');
    assert(!/ds:hash=/.test(sitemap), 'sitemap.xml should not contain build metadata');
  });

  section('Lazy loading and analytics defers', () => {
    const homeHtml = fs.readFileSync('index.html', 'utf8');
    assert(!homeHtml.includes('js/portfolio/modal-helpers.js'), 'index.html should lazy load portfolio modal helpers');
    assert(!homeHtml.includes('js/portfolio/projects-data.js'), 'index.html should lazy load portfolio data');
    assert(!homeHtml.includes('js/forms/contact.js'), 'index.html should lazy load contact form script');
    const portfolioHtml = fs.readFileSync('pages/portfolio.html', 'utf8');
    assert(!portfolioHtml.includes('js/portfolio/modal-helpers.js'), 'pages/portfolio.html should defer portfolio modal helpers');
    assert(!portfolioHtml.includes('js/portfolio/portfolio.js'), 'pages/portfolio.html should rely on lazy loader');
    const commonCode = fs.readFileSync('js/common/common.js', 'utf8');
    assert(commonCode.includes('js/portfolio/projects-data.js'), 'common.js missing portfolio lazy loader');

    const htmlFiles = ['index.html','contact.html','resume.html','resume-pdf.html','privacy.html','pages/destination-analytics.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/resume-pdf.html','pages/privacy.html','pages/short-links.html','pages/utm-batch-builder.html'];
    htmlFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      assert(!content.includes('js/analytics/ga4-events.js'), `${file} should load analytics helpers on demand`);
    });
    const consentCode = fs.readFileSync('js/privacy/consent_manager.js', 'utf8');
    assert(consentCode.includes('ga4-helper'), 'consent manager should inject analytics helper script');
  });

  section('Root/pages drift guard', () => {
    [
      ['contact.html', 'pages/contact.html'],
      ['resume.html', 'pages/resume.html'],
      ['resume-pdf.html', 'pages/resume-pdf.html'],
      ['privacy.html', 'pages/privacy.html'],
      ['sitemap.html', 'pages/sitemap.html']
    ].forEach(([rootFile, pageFile]) => {
      assert(fs.existsSync(rootFile), `${rootFile} missing`);
      assert(fs.existsSync(pageFile), `${pageFile} missing`);
      const rootHtml = fs.readFileSync(rootFile, 'utf8');
      const pageHtml = fs.readFileSync(pageFile, 'utf8');
      assert(rootHtml === pageHtml, `${rootFile} differs from ${pageFile}; run node build/sync-root-pages.js`);
    });
  });

  section('Portfolio featured no-JS cards stay in sync', () => {
    const pdata = evalScript('js/portfolio/projects-data.js');
    const featured = Array.isArray(pdata.window.FEATURED_IDS) ? pdata.window.FEATURED_IDS : [];
    assert(featured.length, 'FEATURED_IDS missing or empty');
    const html = fs.readFileSync('pages/portfolio.html', 'utf8');
    featured.forEach((id) => {
      assert(html.includes(`href="portfolio/${id}"`), `pages/portfolio.html missing featured href for ${id}`);
    });
  });

  section('Job tracker UI additions', () => {
    const trackerHtml = fs.readFileSync('pages/job-application-tracker.html', 'utf8');
    const trackerJs = fs.readFileSync('js/tools/job-application-tracker.js', 'utf8');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack-tab="account"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack-tab="entries"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-form"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-type"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-list"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-filter-query"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-select-all"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-bulk-delete"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-selected-count"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="export-submit"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="map"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="map-remote"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="kpi-found-to-applied"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-import-file"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="import-submit"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-prospect-import-file"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="prospect-import-submit"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="prospect-import-status"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-resume"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-cover"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-posting-date"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-posting-unknown"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-job-url"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-location"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-source"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-follow-up-date"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-follow-up-note"');
    checkFileContains('pages/job-application-tracker.html', 'id="jobtrack-tags"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="custom-field-list"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="saved-view-select"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="followup-list"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="entry-filter-tags"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="funnel-list"');
    checkFileContains('pages/job-application-tracker.html', 'data-jobtrack="time-in-stage-list"');
    assert(!trackerHtml.includes('js/vendor/chartjs/chart.umd.min.js'), 'job tracker should lazy-load Chart.js');
    assert(!trackerHtml.includes('js/vendor/fflate/fflate.min.js'), 'job tracker should lazy-load fflate');
    assert(trackerJs.includes('const CHART_JS_SRC = \'/js/vendor/chartjs/chart.umd.min.js\';'),
      'job tracker missing lazy Chart.js source constant');
    assert(trackerJs.includes('const FFLATE_SRC = \'/js/vendor/fflate/fflate.min.js\';'),
      'job tracker missing lazy fflate source constant');
    assert(trackerJs.includes('await ensureFflate();'), 'job tracker should load fflate on demand');
    assert(trackerJs.includes('ensureChartJs()'), 'job tracker should load Chart.js on demand');
  });

  section('QR generator enhanced workflow contracts', () => {
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-mode-basic"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-mode-advanced"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-payload-mode"');
    checkFileContains('pages/qr-code-generator.html', 'data-qrtool-payload-pane="wifi"');
    checkFileContains('pages/qr-code-generator.html', 'data-qrtool-payload-pane="vcard"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-warning-list"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-autofix"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-verify"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-copy-png"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-copy-svg"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-download-all"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-config-save"');
    checkFileContains('pages/qr-code-generator.html', 'id="qrtool-config-share"');
    checkFileContains('pages/qr-code-generator.html', 'js/tools/qr-code-generator-utils.js');
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

  section('QR generator utility tests', () => {
    runQrCodeGeneratorUtilsTests({ assert });
  });

  section('Project pages and sitemap entries', () => {
    const pdata = evalScript('js/portfolio/projects-data.js');
    const ids = pdata.window.PROJECTS
      .filter(p => p && p.published !== false)
      .map(p => p.id);
    assert(ids.length > 0, 'no project ids found');

    const sitemap = fs.readFileSync('sitemap.xml', 'utf8');
    ids.forEach(id => {
      const file = `pages/portfolio/${id}.html`;
      assert(fs.existsSync(file), `${file} missing`);
      const html = fs.readFileSync(file, 'utf8');
      checkFileContains(file, '<base href="/">');
      checkFileContains(file, 'data-page="project"');
      checkFileContains(file, '<meta property="og:type" content="article">');
      checkFileContains(file, 'href="portfolio">Back to Portfolio');
      checkFileContains(file, `<link rel="canonical" href="https://www.danielshort.me/portfolio/${id}">`);
      checkFileContains(file, `<meta property="og:url" content="https://www.danielshort.me/portfolio/${id}">`);
      assert(sitemap.includes(`https://www.danielshort.me/portfolio/${id}`), `sitemap.xml missing project url: ${id}`);
    });

    const toolsHtml = fs.readFileSync('pages/tools.html', 'utf8');
    const adminToolPaths = toolsHtml
      .split('<article class="tool-card"')
      .slice(1)
      .filter((card) => /data-tools-visibility="admin"/i.test(card))
      .map((card) => {
        const match = /<a\s+[^>]*href="tools\/([^"#?]+)"/i.exec(card);
        return match ? `/tools/${String(match[1] || '').trim()}` : '';
      })
      .filter(Boolean);

    adminToolPaths.forEach((toolPath) => {
      assert(!sitemap.includes(`https://www.danielshort.me${toolPath}`), `sitemap.xml should exclude admin tool URL: ${toolPath}`);
    });

    const normalizePath = (raw) => {
      const source = String(raw || '').trim();
      if (!source) return '';
      let next = source.split('#')[0].split('?')[0];
      if (!next.startsWith('/')) next = `/${next}`;
      next = next.replace(/\/+$/, '') || '/';
      if (next !== '/' && next.endsWith('.html')) next = next.slice(0, -5) || '/';
      return next;
    };

    const vercel = JSON.parse(fs.readFileSync('vercel.json', 'utf8'));
    const noindexPathSet = new Set();
    (vercel.headers || []).forEach((rule) => {
      const source = String(rule && rule.source ? rule.source : '').trim();
      if (!source || /[:*()]/.test(source)) return;
      const hasNoindex = Array.isArray(rule.headers) && rule.headers.some((entry) => {
        const key = String(entry && entry.key ? entry.key : '').toLowerCase();
        const value = String(entry && entry.value ? entry.value : '').toLowerCase();
        return key === 'x-robots-tag' && value.includes('noindex');
      });
      if (!hasNoindex) return;
      const normalized = normalizePath(source);
      if (normalized) noindexPathSet.add(normalized);
    });

    noindexPathSet.forEach((pathName) => {
      assert(!sitemap.includes(`https://www.danielshort.me${pathName}`), `sitemap.xml should exclude noindex URL: ${pathName}`);
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
    checkFileContains('build/templates/header.partial.html', 'div id="primary-menu" class="nav-row"');
    const headerTemplate = fs.readFileSync('build/templates/header.partial.html', 'utf8');
    assert(headerTemplate.includes('class="brand-logo"'), 'nav markup missing brand-logo');
    assert(headerTemplate.includes('class="brand-name"'), 'nav markup missing brand-name');
    assert(headerTemplate.includes('class="brand-title"'), 'nav markup missing brand-title');
    assert(headerTemplate.includes('class="brand-divider"'), 'nav markup missing brand-divider');
    assert(headerTemplate.includes('class="brand-tagline"'), 'nav markup missing brand-tagline');
    assert(headerTemplate.includes('class="brand-tagline-chunk">Destination Analytics'), 'nav markup missing tagline chunk for Destination Analytics');
    assert(headerTemplate.includes('class="brand-tagline-chunk">&amp; Data Science'), 'nav markup missing tagline chunk for Data Science');
    assert(headerTemplate.includes('class="nav-search"'), 'nav markup missing header search');
    assert(headerTemplate.includes('action="search"'), 'header search missing action="search"');
    assert(headerTemplate.includes('name="q"'), 'header search missing query param name="q"');
    assert(!headerTemplate.includes('role="button"'), 'header nav links should not be forced to role="button"');
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
    assert(cssManifest.toolsFile && /^styles-tools\.[0-9a-f]{8}\.css$/.test(cssManifest.toolsFile), 'Tools CSS manifest entry invalid');
    hashedCss = cssManifest.file;
    const hashedToolsCss = cssManifest.toolsFile;
    assert(fs.existsSync(`dist/${hashedCss}`), `dist/${hashedCss} missing`);
    assert(fs.existsSync(`dist/${hashedToolsCss}`), `dist/${hashedToolsCss} missing`);
    assert(fs.existsSync('dist/styles.css'), 'dist/styles.css missing');
    assert(fs.existsSync('dist/styles-tools.css'), 'dist/styles-tools.css missing');

    const projectIds = evalScript('js/portfolio/projects-data.js').window.PROJECTS
      .filter(p => p && p.published !== false)
      .map(p => p.id);
    const projectPages = projectIds.map(id => `pages/portfolio/${id}.html`);
    const toolPages = ['pages/tools.html','pages/tools-dashboard.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html','pages/whisper-transcribe-monitor.html','pages/ga4-utm-performance.html'];
    ['index.html','pages/destination-analytics.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/privacy.html','pages/search.html','404.html', ...toolPages, ...projectPages].forEach(f => {
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
    toolPages.forEach((f) => {
      const html = fs.readFileSync(f, 'utf8');
      assert(
        html.includes(`dist/${hashedToolsCss}`) || html.includes('dist/styles-tools.css'),
        `${f} missing tools stylesheet reference`
      );
    });
  });

  section('Fonts and navigation behavior', () => {
    checkFileContains('index.html', 'fonts.googleapis.com');
    checkFileContains('index.html', 'family=Poppins');

    const navCode = fs.readFileSync('js/navigation/navigation.js', 'utf8');
    const headerTemplate = fs.readFileSync('build/templates/header.partial.html', 'utf8');
    assert(headerTemplate.includes('aria-controls="primary-menu"'), 'header missing aria-controls="primary-menu"');
    assert(headerTemplate.includes('id="primary-menu"'), 'header missing primary-menu');
    assert(navCode.includes("classList.toggle('menu-open'"), 'burger toggle missing body.menu-open');
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
    assert(pkg.scripts && pkg.scripts.build, 'package.json missing build script');

    const buildScript = String(pkg.scripts.build || '');
    const usesLegacyChain = buildScript.includes('build:projects');
    const usesBuildRunner = /build\/build-site\.js/.test(buildScript);
    assert(usesLegacyChain || usesBuildRunner,
      'package.json build script should run build:projects or use build/build-site.js');

    if (usesBuildRunner) {
      assert(fs.existsSync('build/build-site.js'), 'build-site.js missing');
      const buildRunner = fs.readFileSync('build/build-site.js', 'utf8');
      assert(buildRunner.includes('generate-project-pages.js'), 'build-site.js should generate project pages');
      assert(buildRunner.includes('copy-to-public.js'), 'build-site.js should prepare public/ output');
    }

    assert(!buildScript.includes('build:models') &&
           !buildScript.includes('build:pizza-tips-model') &&
           !buildScript.includes('tips_data_geocoded'),
           'package.json build script should not rebuild pizza tips model');

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
    const hasGames = rewrites.some(r => r.source === '/games' && r.destination === '/pages/games');
    const hasGameSlot = rewrites.some(r => r.source === '/games/slot-machine' && r.destination === '/demos/slot-machine-demo');
    const hasGameDogfight = rewrites.some(r => r.source === '/games/stellar-dogfight' && r.destination === '/demos/stellar-dogfight-demo');
    const hasGameDogfightHtml = rewrites.some(r => r.source === '/games/stellar-dogfight.html' && r.destination === '/demos/stellar-dogfight-demo');
    const hasGameRoulette = rewrites.some(r => r.source === '/games/roulette' && r.destination === '/demos/roulette-double-zero-demo');
    const hasGameRouletteHtml = rewrites.some(r => r.source === '/games/roulette.html' && r.destination === '/demos/roulette-double-zero-demo');
    assert(hasGames, 'games landing rewrite missing');
    assert(hasGameSlot, 'slot machine games rewrite missing');
    assert(hasGameDogfight, 'stellar dogfight games rewrite missing');
    assert(hasGameDogfightHtml, 'stellar dogfight html rewrite missing');
    assert(hasGameRoulette, 'roulette games rewrite missing');
    assert(hasGameRouletteHtml, 'roulette html rewrite missing');
    const hasToolsDashboard = rewrites.some(r => r.source === '/tools/dashboard' && r.destination === '/pages/tools-dashboard');
    const hasToolsDashboardHtml = rewrites.some(r => r.source === '/tools/dashboard.html' && r.destination === '/pages/tools-dashboard');
    assert(hasToolsDashboard && hasToolsDashboardHtml, 'tools dashboard rewrites missing');
    const hasGa4UtmTool = rewrites.some(r => r.source === '/tools/ga4-utm-performance' && r.destination === '/pages/ga4-utm-performance');
    const hasGa4UtmToolHtml = rewrites.some(r => r.source === '/tools/ga4-utm-performance.html' && r.destination === '/pages/ga4-utm-performance');
    assert(hasGa4UtmTool && hasGa4UtmToolHtml, 'GA4 UTM tool rewrites missing');
    const hasSearch = rewrites.some(r => r.source === '/search' && r.destination === '/pages/search');
    const hasSearchHtml = rewrites.some(r => r.source === '/search.html' && r.destination === '/pages/search');
    assert(hasSearch && hasSearchHtml, 'search rewrites missing');
    const hasSitemap = rewrites.some(r => r.source === '/sitemap' && r.destination === '/pages/sitemap');
    const hasSitemapHtml = rewrites.some(r => r.source === '/sitemap.html' && r.destination === '/pages/sitemap');
    assert(hasSitemap && hasSitemapHtml, 'sitemap rewrites missing');

    const hasDshortTwoSeg = rewrites.some(r =>
      r.source === '/:first/:rest' &&
      r.destination === '/api/go/:first%2F:rest' &&
      Array.isArray(r.has) &&
      r.has.some(h => h && h.type === 'host' && h.value === 'dshort.me')
    );
    assert(hasDshortTwoSeg, 'dshort.me missing 2-segment shortlink rewrite');

    const hasGoTwoSeg = rewrites.some(r =>
      r.source === '/go/:first/:rest' &&
      r.destination === '/api/go/:first%2F:rest'
    );
    assert(hasGoTwoSeg, 'missing 2-segment /go shortlink rewrite');

    const headers = (vercelObj && vercelObj.headers) || [];
    const hasNoindexShortLinks = headers.some(h =>
      h && h.source === '/short-links' &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexToolsDashboard = headers.some(h =>
      h && h.source === '/tools/dashboard' &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexGa4Tool = headers.some(h =>
      h && h.source === '/tools/ga4-utm-performance' &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexWhisperTool = headers.some(h =>
      h && h.source === '/tools/whisper-transcribe-monitor' &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    assert(hasNoindexShortLinks, 'short-links noindex header missing');
    assert(hasNoindexToolsDashboard, 'tools dashboard noindex header missing');
    assert(hasNoindexGa4Tool, 'GA4 tool noindex header missing');
    assert(hasNoindexWhisperTool, 'Whisper tool noindex header missing');
  });

  section('Search index', () => {
    assert(fs.existsSync('dist/search-index.json'), 'dist/search-index.json missing');
    const raw = fs.readFileSync('dist/search-index.json', 'utf8');
    let parsed;
    try { parsed = JSON.parse(raw); } catch {}
    assert(parsed && Array.isArray(parsed.pages), 'search index should contain pages array');
    assert(parsed.pages.length >= 10, 'search index has too few entries');
    const urls = new Set(parsed.pages.map((entry) => String(entry && entry.url || '').trim()));
    assert(!urls.has('/tools/ga4-utm-performance'), 'search index should exclude noindex GA4 tool');
    assert(!urls.has('/tools/whisper-transcribe-monitor'), 'search index should exclude noindex Whisper tool');
  });

  section('GA4 report race guards', () => {
    const ga4Tool = fs.readFileSync('js/tools/ga4-utm-performance.js', 'utf8');
    assert(ga4Tool.includes('let utmRequestSeq = 0;'), 'GA4 tool missing UTM request sequencing guard');
    assert(ga4Tool.includes('let exploreRequestSeq = 0;'), 'GA4 tool missing Explore request sequencing guard');
    assert(ga4Tool.includes('let accessRequestSeq = 0;'), 'GA4 tool missing access request sequencing guard');
    assert(ga4Tool.includes('if (utmBusy) return;'), 'GA4 tool should guard overlapping UTM requests');
    assert(ga4Tool.includes('if (exploreBusy) return;'), 'GA4 tool should guard overlapping Explore requests');
    assert(ga4Tool.includes('if (accessBusy) return;'), 'GA4 tool should guard overlapping access checks');
    assert(ga4Tool.includes('requestId !== utmRequestSeq'), 'GA4 tool should ignore stale UTM responses');
    assert(ga4Tool.includes('requestId !== exploreRequestSeq'), 'GA4 tool should ignore stale Explore responses');
    assert(ga4Tool.includes('requestId !== accessRequestSeq'), 'GA4 tool should ignore stale access responses');
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
    checkFileContains('404.html', 'js/common/404-redirect.js');
    checkFileContains('js/common/404-redirect.js', '/portfolio/${encodeURIComponent(project)}');
    checkFileContains('pages/portfolio.html', 'id="portfolio-carousel"');
    checkFileContains('pages/portfolio.html', 'id="filter-menu"');
    checkFileContains('pages/portfolio.html', 'id="projects"');
    checkFileContains('pages/portfolio.html', 'id="modals"');
    checkFileContains('pages/portfolio.html', 'id="see-more"');
  });

  section('Base hrefs and redirect sanity', () => {
    ['pages/destination-analytics.html','pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html','pages/resume-pdf.html',
     'pages/tools.html','pages/tools-dashboard.html','pages/search.html','pages/sitemap.html','pages/games.html','pages/short-links.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html','pages/ga4-utm-performance.html',
     'demos/chatbot-demo.html','demos/shape-demo.html','demos/sentence-demo.html','demos/slot-machine-demo.html','demos/stellar-dogfight-demo.html']
      .forEach(f => checkFileContains(f, '<base href="/">'));

    ['pages/destination-analytics.html','pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html','pages/resume-pdf.html']
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
    checkFileContains('resume-pdf.html', 'documents/Resume.pdf');
    checkFileContains('pages/resume-pdf.html', 'documents/Resume.pdf');
    checkFileContains('contact.html', 'id="contact-form"');
    checkFileContains('pages/contact.html', 'action="/api/contact"');
    checkFileContains('index.html', 'action="/api/contact"');
    assert(fs.existsSync('api/contact.js'), 'api/contact.js missing');
  });

  section('Search page form contract', () => {
    const html = fs.readFileSync('pages/search.html', 'utf8');
    assert(html.includes('id="search-page-form"'), 'pages/search.html missing in-page search form');
    assert(html.includes('id="search-page-q"'), 'pages/search.html missing in-page search input');
    assert(html.includes('id="search-results"'), 'pages/search.html missing search results container');
    assert(html.includes('id="search-status"'), 'pages/search.html missing search status region');
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

  section('Stellar dogfight demo', () => {
    const demoHtml = fs.readFileSync('demos/stellar-dogfight-demo.html', 'utf8');
    assert(demoHtml.includes('<link rel="stylesheet" href="css/components/stellar-dogfight.css" />'),
      'stellar dogfight demo should load component CSS');
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-data.js"></script>'),
      'stellar dogfight demo should load data script');
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-audio.js"></script>'),
      'stellar dogfight demo should load audio script');
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-demo.js"></script>'),
      'stellar dogfight demo should load runtime script');
    assert(demoHtml.includes('data-setting="audio"'),
      'stellar dogfight demo should expose audio settings');
    assert(demoHtml.includes('data-setting="hud-layout"'),
      'stellar dogfight demo should expose HUD layout settings');
    assert(demoHtml.includes('data-setting="hud-scale"'),
      'stellar dogfight demo should expose HUD scale settings');
    assert(demoHtml.includes('data-setting="target-assist"'),
      'stellar dogfight demo should expose target assist settings');
    assert(demoHtml.includes('data-setting="camera-mode"'),
      'stellar dogfight demo should expose camera mode settings');
    assert(demoHtml.includes('class="mission-pill-row"') && demoHtml.includes('class="action-cluster"'),
      'stellar dogfight demo should group highlights and actions for cleaner hierarchy');
    assert(demoHtml.includes('class="panel-subsection panel-disclosure"'),
      'stellar dogfight demo should collapse advanced sidebar sections to reduce clutter');
    assert(demoHtml.includes('class="hud-pill-group hud-pill-group-primary"') && demoHtml.includes('class="hud-pill-group hud-pill-group-meta"'),
      'stellar dogfight demo should separate primary combat HUD data from secondary meta data');
    assert(demoHtml.includes('data-role="command-overview"') && demoHtml.includes('data-role="command-roadmap"'),
      'stellar dogfight demo should expose command overview and roadmap surfaces');
    assert(demoHtml.includes('data-role="progress-overview"') && demoHtml.includes('data-role="settings-guidance"'),
      'stellar dogfight demo should expose progress and settings guidance surfaces');
    assert(demoHtml.includes('data-panel-feature="premium"') && demoHtml.includes('data-settings-tier="2"'),
      'stellar dogfight demo should mark advanced panels and settings for progressive unlocking');
    assert(demoHtml.includes('data-action="help"'),
      'stellar dogfight demo should expose help action');
    assert(demoHtml.includes('data-action="preset-save"') && demoHtml.includes('data-action="preset-load"'),
      'stellar dogfight demo should expose loadout preset actions');
    assert(demoHtml.includes('data-action="tutorial"') && demoHtml.includes('data-action="glossary"'),
      'stellar dogfight demo should expose tutorial and glossary actions');
    assert(demoHtml.includes('data-action="replay-last-loadout"'),
      'stellar dogfight demo should expose replay-last-loadout action');
    assert(demoHtml.includes('data-tab-target="premium"') && demoHtml.includes('data-tab-panel="premium"'),
      'stellar dogfight demo should expose premium tab panel');
    assert(demoHtml.includes('data-role="premium-shop"'),
      'stellar dogfight demo should expose premium shop container');
    assert(demoHtml.includes('data-role="run-analytics"'),
      'stellar dogfight demo should expose run analytics container');
    assert(demoHtml.includes('data-stat="premium-currency"') && demoHtml.includes('data-stat="premium-currency-total"'),
      'stellar dogfight demo should expose premium currency stats');
    assert(demoHtml.includes('data-stat="threat-tier"') && demoHtml.includes('data-stat="objective"'),
      'stellar dogfight demo should expose threat/objective stats');
    assert(demoHtml.includes('data-stat="dps"') && demoHtml.includes('data-stat="ehp"') && demoHtml.includes('data-stat="energy-sustain"'),
      'stellar dogfight demo should expose derived combat stats');
    assert(demoHtml.includes('data-stat="weekly-mutator"') && demoHtml.includes('data-stat="challenge-seed"'),
      'stellar dogfight demo should expose weekly mutator and challenge seed stats');
    assert(demoHtml.includes('data-setting="palette"'),
      'stellar dogfight demo should expose palette settings');
    assert(demoHtml.includes('data-option="controller"'),
      'stellar dogfight demo should expose controller input mode');
    assert(!demoHtml.includes('const STORAGE_KEY = "stellarDogfightProgress";'),
      'stellar dogfight runtime should not be inline in HTML');

    const demoCss = fs.readFileSync('css/components/stellar-dogfight.css', 'utf8');
    assert(demoCss.includes('.mission-shell') && demoCss.includes('.arena-frame'),
      'stellar dogfight CSS component appears incomplete');
    assert(demoCss.includes('body.is-hud-compact') && demoCss.includes('body.is-hud-scale-lg'),
      'stellar dogfight CSS should include HUD presentation variants');
    assert(demoCss.includes('.premium-card'),
      'stellar dogfight CSS should include premium card presentation');
    assert(demoCss.includes('.mission-pill-row') && demoCss.includes('.action-cluster'),
      'stellar dogfight CSS should style the streamlined header layout');
    assert(demoCss.includes('.panel-disclosure'),
      'stellar dogfight CSS should style disclosure sections for progressive disclosure');
    assert(demoCss.includes('.hud-pill-group-meta') && demoCss.includes('body.is-hud-focused'),
      'stellar dogfight CSS should support contextual combat HUD decluttering');
    assert(demoCss.includes('.tab-badge') && demoCss.includes('.debrief-grid'),
      'stellar dogfight CSS should style new-tab badges and post-run debrief cards');
    assert(demoCss.includes('.progress-step[data-state="new"]'),
      'stellar dogfight CSS should style newly unlocked roadmap states');

    const runtimeJs = fs.readFileSync('js/demos/stellar-dogfight-demo.js', 'utf8');
    assert(runtimeJs.includes('const DEFERRED_UI_FLUSH_MS = 220;'),
      'stellar dogfight runtime missing deferred UI flush constant');
    assert(runtimeJs.includes('const DEFERRED_SAVE_FLUSH_MS = 180;'),
      'stellar dogfight runtime missing deferred save flush constant');
    assert(runtimeJs.includes('function queueProgressSave() {'),
      'stellar dogfight runtime missing queued save helper');
    assert(runtimeJs.includes('function flushDeferredState(force = false) {'),
      'stellar dogfight runtime missing deferred state flush helper');
    assert(/function updateHud\(\)\s*{\s*flushDeferredState\(\);/.test(runtimeJs),
      'stellar dogfight HUD should flush deferred state before HUD updates');
    assert(runtimeJs.includes('const dpr = Math.max(1, state.renderScale || window.devicePixelRatio || 1);'),
      'stellar dogfight minimap should use renderScale-aware DPR');
    assert(runtimeJs.includes('function getKeybindConflicts(targetAction, key) {'),
      'stellar dogfight runtime missing keybind conflict guard');
    assert(runtimeJs.includes('function saveLoadoutPreset(slot) {') && runtimeJs.includes('function loadLoadoutPreset(slot) {'),
      'stellar dogfight runtime should support loadout presets');
    assert(runtimeJs.includes('function syncHudPresentation() {'),
      'stellar dogfight runtime should support HUD presentation settings');
    assert(runtimeJs.includes('document.body.classList.toggle("is-hud-compact"'),
      'stellar dogfight runtime should apply HUD compact class');
    assert(runtimeJs.includes('enabled: (progress.settings.audio || "on") !== "off"'),
      'stellar dogfight runtime should initialize audio from settings');
    assert(runtimeJs.includes('state.runEndedByAbort = true;'),
      'stellar dogfight runtime missing aborted run flag');
    assert(runtimeJs.includes('function drawThreatIndicators() {'),
      'stellar dogfight runtime missing off-screen threat indicators');
    assert(runtimeJs.includes('function drawEnemyThreatHalo(enemy) {'),
      'stellar dogfight runtime missing elite/boss visual telegraphs');
    assert(runtimeJs.includes('function getPredictedInterceptPoint(origin, target, projectileSpeed) {'),
      'stellar dogfight runtime should predict intercept points for lead assistance');
    assert(runtimeJs.includes('function updateTargetAssist() {') && runtimeJs.includes('function getPlayerShotAngle() {'),
      'stellar dogfight runtime should include target assist and aim magnetism helpers');
    assert(runtimeJs.includes('function drawEnemyAttackTelegraph(enemy) {'),
      'stellar dogfight runtime should telegraph dangerous enemy fire windows');
    assert(runtimeJs.includes('progress.settings.cameraMode || "dynamic"'),
      'stellar dogfight runtime should support dynamic camera settings');
    assert(runtimeJs.includes('function isHudCombatFocusActive(now = performance.now()) {') && runtimeJs.includes('function syncContextualHudState(now = performance.now()) {'),
      'stellar dogfight runtime should drive contextual HUD focus states during combat');
    assert(runtimeJs.includes('const WAVE_ROLE_PROFILES = [') && runtimeJs.includes('function getWaveRoleProfile(globalWave, objectiveId, isHardWave, isBossWave) {'),
      'stellar dogfight runtime should build waves around role-based composition profiles');
    assert(runtimeJs.includes('const BOSS_PHASES = [') && runtimeJs.includes('function applyBossPhase(enemy, phaseIndex, options = {}) {') && runtimeJs.includes('function updateBossBehavior(enemy, delta) {'),
      'stellar dogfight runtime should support multi-phase boss behavior');
    assert(runtimeJs.includes('const braking = isActionActive("brake") || input.padBrake;') && runtimeJs.includes('state.brakeTurnBoost') && runtimeJs.includes('state.brakeSpeedClamp'),
      'stellar dogfight runtime should apply air brake handling to player flight');
    assert(runtimeJs.includes('const BALANCE_TUNING = {'),
      'stellar dogfight runtime missing balance tuning constants');
    assert(runtimeJs.includes('const DROP_PITY_THRESHOLDS = {'),
      'stellar dogfight runtime missing drop pity thresholds');
    assert(runtimeJs.includes('const PREMIUM_CURRENCY_LABEL = "Astralite";'),
      'stellar dogfight runtime missing premium currency label');
    assert(runtimeJs.includes('const PREMIUM_DROP_RUN_CAP = 4;'),
      'stellar dogfight runtime missing premium drop cap');
    assert(runtimeJs.includes('function renderPremiumShop() {') && runtimeJs.includes('function buyPremiumItem(itemId) {'),
      'stellar dogfight runtime missing premium shop handlers');
    assert(runtimeJs.includes('state.runPremiumDrops < PREMIUM_DROP_RUN_CAP'),
      'stellar dogfight runtime missing capped premium drop logic');
    assert(runtimeJs.includes('progress.premiumCurrency += premiumAmount;'),
      'stellar dogfight runtime missing premium currency drops');
    assert(runtimeJs.includes('function updateRunRecords(summary) {'),
      'stellar dogfight runtime missing run-record tracking');
    assert(runtimeJs.includes('function startTutorial() {') && runtimeJs.includes('function openGlossary() {'),
      'stellar dogfight runtime should include tutorial/glossary handlers');
    assert(runtimeJs.includes('function replayLastLoadout() {'),
      'stellar dogfight runtime should include replay-last-loadout handler');
    assert(runtimeJs.includes('function renderRunAnalytics() {'),
      'stellar dogfight runtime should include run analytics renderer');
    assert(runtimeJs.includes('function captureRunUnlockBaseline() {') && runtimeJs.includes('function buildRunDebrief(summary, telemetry, reason) {'),
      'stellar dogfight runtime should capture unlock baselines and build post-run debriefs');
    assert(runtimeJs.includes('function updateTabBadges() {') && runtimeJs.includes('function markPanelSeen(target) {'),
      'stellar dogfight runtime should track first-visit UI badges');
    assert(runtimeJs.includes('function renderStatusIcons() {'),
      'stellar dogfight runtime should include status icon rendering');
    assert(runtimeJs.includes('setOverlay("choice-event");'),
      'stellar dogfight runtime should include milestone choice overlay flow');
    assert(runtimeJs.includes('data-overlay-action="restart"'),
      'stellar dogfight runtime missing gameover restart action');

    const trainingStart = runtimeJs.indexOf('function startTraining() {');
    const trainingEnd = runtimeJs.indexOf('function resetMission() {');
    assert(trainingStart >= 0 && trainingEnd > trainingStart,
      'stellar dogfight startTraining function not found');
    const trainingBlock = runtimeJs.slice(trainingStart, trainingEnd);
    const trainingPlayerCreate = trainingBlock.indexOf('player = createPlayer();');
    const trainingLoadoutAssign = trainingBlock.indexOf('state.runLoadout = { ship: player.ship?.name || "Unknown", weapon: player.weapon?.name || "Unknown" };');
    assert(trainingPlayerCreate >= 0 && trainingLoadoutAssign > trainingPlayerCreate,
      'stellar dogfight training loadout should be assigned after player creation');

    const abortStart = runtimeJs.indexOf('function triggerAbortRewards() {');
    const abortEnd = runtimeJs.indexOf('function renderLootBursts() {');
    assert(abortStart >= 0 && abortEnd > abortStart,
      'stellar dogfight triggerAbortRewards function not found');
    const abortBlock = runtimeJs.slice(abortStart, abortEnd);
    assert(abortBlock.includes('endRun("abort");'),
      'stellar dogfight abort flow should end run as abort');
    assert(abortBlock.includes('state.lossRewards = null;'),
      'stellar dogfight abort flow should not grant loss rewards');
    assert(!abortBlock.includes('applyLossRewards'),
      'stellar dogfight abort flow should never apply loss rewards');
    assert(!abortBlock.includes('buildLossRewards'),
      'stellar dogfight abort flow should never build loss rewards');

    const audioJs = fs.readFileSync('js/demos/stellar-dogfight-audio.js', 'utf8');
    assert(audioJs.includes('function createAudioController(options = {}) {'),
      'stellar dogfight audio controller missing createAudioController');
    assert(audioJs.includes('window.STELLAR_DOGFIGHT_AUDIO = {'),
      'stellar dogfight audio module missing global export');

    const dataEnv = evalScript('js/demos/stellar-dogfight-data.js');
    const db = dataEnv.window.STELLAR_DOGFIGHT_DB || {};
    const premiumItems = db.PREMIUM_SHOP_ITEMS;
    assert(Array.isArray(premiumItems) && premiumItems.length >= 8,
      'stellar dogfight data should export premium shop inventory');
    assert(premiumItems.some((item) => item && item.kind === 'one-time'),
      'stellar dogfight premium shop should include one-time unlocks');
    assert(premiumItems.some((item) => item && item.kind === 'scalable' && item.maxLevel > 1),
      'stellar dogfight premium shop should include scalable upgrades');
    assert(premiumItems.every((item) => item && item.id && typeof item.apply === 'function'),
      'stellar dogfight premium shop items must define id and apply');
  });

  section('Slot machine demo', () => {
    runSlotDemoTests({ assert, checkFileContains });
  });

  section('Roulette demo', () => {
    checkFileContains('demos/roulette-double-zero-demo.html', '<title>Double-Zero Roulette | Daniel Short</title>');
    checkFileContains('demos/roulette-double-zero-demo.html', 'id="roulette-spin"');
    checkFileContains('demos/roulette-double-zero-demo.html', 'id="roulette-number-grid"');
    checkFileContains('demos/roulette-double-zero-demo.html', 'id="roulette-hot-list"');
    checkFileContains('demos/roulette-double-zero-demo.html', 'js/demos/roulette-double-zero-demo.js');
    const rouletteJs = fs.readFileSync('js/demos/roulette-double-zero-demo.js', 'utf8');
    assert(rouletteJs.includes('const WHEEL_ORDER = ['), 'roulette demo missing wheel order constant');
    assert(rouletteJs.includes('const MAX_HISTORY = 200;'), 'roulette demo should track 200-spin history');
    assert(rouletteJs.includes('const STORAGE_KEY = "roulette-double-zero-session-v1";'), 'roulette demo missing local storage key');
    assert(rouletteJs.includes('window.localStorage.setItem(STORAGE_KEY'), 'roulette demo should save session to local storage');
    assert(rouletteJs.includes('window.localStorage.getItem(STORAGE_KEY)'), 'roulette demo should load session from local storage');
    assert(rouletteJs.includes('const CHIP_VALUES = [1, 5, 25, 100];'), 'roulette demo should expose chip values');
    assert(rouletteJs.includes('function computeSpinTargets(winningPocket) {'), 'roulette demo missing spin target resolver');
    assert(rouletteJs.includes('finalBallMod = normalizeDeg(pocketLocalAngle + finalWheelMod);'),
      'roulette demo ball should land based on wheel + pocket angle');
    assert(rouletteJs.includes('id: "basket-first-five"'), 'roulette demo missing first five basket bet');
  });

  section('UTM Batch Builder core', () => {
    runUtmBatchBuilderTests({ assert });
  });

  console.log(`\nAll tests passed. Total checks: ${assertCount}`);
  process.exit(0);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
