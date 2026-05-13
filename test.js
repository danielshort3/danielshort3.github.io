// Lightweight test runner used by `npm test`
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const runSlotDemoTests = require('./tests/slot-machine-demo.test.js');
const runUtmBatchBuilderTests = require('./tests/utm-batch-builder.test.js');
const runQrCodeGeneratorUtilsTests = require('./tests/qr-code-generator-utils.test.js');
const runTextCompareCoreTests = require('./tests/text-compare-core.test.js');

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

function checkFileContainsOneOf(file, texts, message) {
  assert(fs.existsSync(file), `${file} does not exist`);
  const content = fs.readFileSync(file, 'utf8');
  assert(texts.some((text) => content.includes(text)), message || `${file} missing expected text`);
}

function readFile(file) {
  assert(fs.existsSync(file), `${file} does not exist`);
  return fs.readFileSync(file, 'utf8');
}

function htmlHasManagedBundle(html, baseName) {
  const escaped = String(baseName || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`dist\\/${escaped}(?:\\.[0-9a-f]{8})?\\.js`, 'i').test(String(html || ''));
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
    checkFileContains('index.html', 'Taking you to the default analyst site.');

    const expectedTitles = {
      'index.html': 'Daniel Short | Redirecting to Analytics',
      'pages/analytics.html': 'Data Analytics | Daniel Short',
      'pages/data-science.html': 'Data Science | Daniel Short',
      'pages/destination-analytics.html': 'Destination Analytics | Daniel Short',
      'pages/contact.html': 'Contact | Daniel Short',
      'pages/resume.html': 'Resume Versions | Daniel Short',
      'pages/resume-pdf.html': 'Resume PDF Versions | Daniel Short',
      'pages/resume-analytics.html': 'Analytics Resume | Daniel Short',
      'pages/resume-data-science.html': 'Data Science Resume | Daniel Short',
      'pages/resume-tourism.html': 'Tourism Resume | Daniel Short',
      'pages/resume-analytics-pdf.html': 'Analytics Resume PDF | Daniel Short',
      'pages/resume-data-science-pdf.html': 'Data Science Resume PDF | Daniel Short',
      'pages/resume-tourism-pdf.html': 'Tourism Resume PDF | Daniel Short',
      'pages/tourism.html': 'Tourism Analytics | Daniel Short',
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
      'pages/analytics.html',
      'pages/data-science.html',
      'pages/destination-analytics.html',
      'pages/tourism.html',
      'pages/contact.html',
      'pages/portfolio.html',
      'pages/contributions.html',
      'pages/resume.html',
      'pages/resume-pdf.html',
      'pages/resume-analytics.html',
      'pages/resume-data-science.html',
      'pages/resume-tourism.html',
      'pages/resume-analytics-pdf.html',
      'pages/resume-data-science-pdf.html',
      'pages/resume-tourism-pdf.html',
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

    ['pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','pages/resume.html','pages/resume-pdf.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html','pages/sitemap.html'].forEach((f) => {
      checkFileContainsOneOf(f, ['js/common/common.js', 'dist/site-shell.'], `${f} missing shared shell script reference`);
    });
    ['pages/games.html','pages/ocean-wave-simulation.html', ...toolPages].forEach((f) => {
      checkFileContainsOneOf(f, ['js/common/common.js', 'dist/site-shell.'], `${f} missing shared shell script reference`);
    });

    ['pages/resume.html','pages/resume-pdf.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html','pages/games.html','pages/ocean-wave-simulation.html','404.html','dshort.html', ...privateToolPages].forEach((f) => {
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

    ['pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/contact.html','pages/portfolio.html','pages/contributions.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/tools.html','pages/games.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/utm-batch-builder.html','404.html'].forEach((f) => {
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
    assert(!sitemap.includes('<loc>https://www.danielshort.me/</loc>'), 'sitemap.xml should not include redirecting root URL');
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
    assert(!commonCode.includes('--jump-link-progress') && !commonCode.includes('updateLinkProgress'),
      'common.js should not add scroll-progress styling to the homepage jump rail');

    const htmlFiles = ['index.html','contact.html','resume.html','resume-pdf.html','privacy.html','pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/resume-pdf.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html','pages/privacy.html','pages/short-links.html','pages/utm-batch-builder.html'];
    htmlFiles.forEach(file => {
      const content = fs.readFileSync(file, 'utf8');
      assert(!content.includes('js/analytics/ga4-events.js'), `${file} should load analytics helpers on demand`);
    });
    const consentCode = fs.readFileSync('js/privacy/consent_manager.js', 'utf8');
    assert(consentCode.includes('ga4-helper'), 'consent manager should inject analytics helper script');
  });

  section('Tools directory contracts', () => {
    assert(!fs.existsSync('content/tools/.json'), 'content/tools/.json placeholder should not exist');

    const toolFiles = fs.readdirSync('content/tools')
      .filter((name) => name.endsWith('.json'))
      .sort();
    assert(toolFiles.length >= 10, 'tools content catalog unexpectedly small');
    assert(toolFiles.every((name) => !name.startsWith('.')), 'tools content catalog should ignore dotfiles');

    const vercelConfig = JSON.parse(readFile('vercel.json'));
    const rewrites = Array.isArray(vercelConfig.rewrites) ? vercelConfig.rewrites : [];
    const catalogJs = readFile('js/accounts/tools-account-ui.js');
    const toolsHtml = readFile('pages/tools.html');

    assert(toolsHtml.includes('<h1>Tools</h1>'), 'tools page should expose a visible h1');
    assert(toolsHtml.includes('data-tools-filter-input'), 'tools page missing directory search input');
    assert(toolsHtml.includes('data-tools-filter-status'), 'tools page missing filter status');
    assert(toolsHtml.includes('class="tools-nav"'), 'tools page missing category shortcuts');
    assert(!toolsHtml.includes('More tools soon'), 'tools page should not render placeholder cards');
    assert(!toolsHtml.includes('href="tools/"'), 'tools page should not render empty tool links');
    assert(!toolsHtml.includes('id="tools-experiments"'), 'empty tool categories should not render');

    toolFiles.forEach((fileName) => {
      const tool = JSON.parse(readFile(path.join('content', 'tools', fileName)));
      const slug = String(tool.slug || '').trim();
      const href = String(tool.href || '').trim();
      assert(/^[-a-z0-9]+$/.test(slug), `${fileName} has invalid or empty slug`);
      assert(href === `tools/${slug}`, `${fileName} href should match tools/${slug}`);
      assert(toolsHtml.includes(`href="${href}"`), `${fileName} missing from tools page`);
      assert(rewrites.some((rule) => rule.source === `/tools/${slug}` && rule.destination === `/pages/${slug}`),
        `${fileName} missing clean URL rewrite`);
      assert(catalogJs.includes(`'${slug}':`), `${fileName} missing from TOOL_CATALOG`);
    });

    ['build/templates/header.partial.html', 'build/templates/footer.partial.html', 'pages/tools.html', 'pages/games.html'].forEach((file) => {
      assert(!/\bhref=""/.test(readFile(file)), `${file} should not contain empty href attributes`);
    });
  });

  section('Website chatbot contracts', () => {
    const pkg = JSON.parse(readFile('package.json'));
    assert(pkg.dependencies && pkg.dependencies['@aws-sdk/client-bedrock-runtime'], 'package.json missing Bedrock Runtime dependency');
    assert(pkg.dependencies && pkg.dependencies['@aws-sdk/client-dynamodb'], 'package.json missing DynamoDB dependency');
    assert(pkg.dependencies && pkg.dependencies['@aws-sdk/lib-dynamodb'], 'package.json missing DynamoDB document client dependency');

    const buildRunner = readFile('build/build-site.js');
    const devServer = readFile('build/dev.js');
    const copyPublic = readFile('build/copy-to-public.js');
    const generator = readFile('build/generate-chatbot-knowledge.js');
    assert(buildRunner.includes('generate-chatbot-knowledge.js'), 'build runner should generate chatbot knowledge');
    assert(devServer.includes("pathname === '/api/chatbot'") && devServer.includes("pathname === '/api/chatbot/logs'") && devServer.includes('loadChatbotLogsApi'),
      'local dev server should route the chatbot APIs');
    assert(copyPublic.includes("'chatbot-knowledge.json'"), 'public copy should include chatbot knowledge JSON');
    assert(generator.includes('excludedPathPatterns') && generator.includes('extractMainText') && generator.includes('loadProjectMetadata') && generator.includes('chatbot-knowledge.json'),
      'chatbot knowledge generator should extract scoped page content and project metadata');
    assert(generator.includes('InvokeModelCommand') &&
      generator.includes("DEFAULT_EMBED_MODEL_ID = 'amazon.titan-embed-text-v2:0'") &&
      generator.includes('applyEmbeddings') &&
      generator.includes('CHATBOT_EMBEDDINGS_REQUIRED') &&
      generator.includes('loadExistingEmbeddingCache'),
      'chatbot knowledge generator should support reusable build-time Bedrock embeddings');

    const knowledge = JSON.parse(readFile('dist/chatbot-knowledge.json'));
    assert(knowledge.version === 1, 'chatbot knowledge should declare version 1');
    assert(knowledge.origin === 'https://www.danielshort.me', 'chatbot knowledge should use the public site origin');
    assert(Array.isArray(knowledge.pages) && knowledge.pages.length >= 8, 'chatbot knowledge should include public site pages');
    assert(Array.isArray(knowledge.chunks) && knowledge.chunks.length >= knowledge.pages.length, 'chatbot knowledge should include page chunks');
    assert(knowledge.embeddings &&
      knowledge.embeddings.modelId === 'amazon.titan-embed-text-v2:0' &&
      Number(knowledge.embeddings.dimensions) === 512 &&
      ['ready', 'partial', 'skipped'].includes(knowledge.embeddings.status),
      'chatbot knowledge should record Bedrock embedding metadata even when vectors are unavailable');
    const knowledgeUrls = new Set(knowledge.pages.map((page) => page && page.url));
    ['/analytics', '/data-science', '/tourism', '/contact', '/portfolio', '/resume-analytics'].forEach((url) => {
      assert(knowledgeUrls.has(url), `chatbot knowledge missing ${url}`);
    });
    ['/tools', '/games', '/privacy', '/destination-analytics', '/contributions', '/resume-analytics-pdf', '/chatbot-demo'].forEach((url) => {
      assert(!knowledgeUrls.has(url), `chatbot knowledge should exclude ${url}`);
    });
    assert(knowledge.chunks.every((chunk) => chunk && chunk.id && chunk.url && chunk.title && chunk.text),
      'chatbot knowledge chunks should be citeable');
    const { retrieveKnowledge } = require('./api/_lib/chatbot-knowledge');
    const chatbotProjectRetrieval = retrieveKnowledge('Tell me about the Visit Grand Junction chatbot project', { url: '/analytics', title: 'Analytics' });
    assert(chatbotProjectRetrieval.chunks[0] && chatbotProjectRetrieval.chunks[0].url === '/portfolio/chatbotLora',
      'chatbot retrieval should rank the Visit Grand Junction chatbot project first for matching project questions');
    const contactRetrieval = retrieveKnowledge('How do I contact Daniel?', { url: '/analytics', title: 'Analytics' });
    assert(contactRetrieval.chunks[0] && contactRetrieval.chunks[0].url === '/contact',
      'chatbot retrieval should rank the contact page first for contact questions');
    const resumeRetrieval = retrieveKnowledge('Where is the best resume?', { url: '/analytics', title: 'Analytics' });
    assert(resumeRetrieval.chunks[0] && resumeRetrieval.chunks[0].url === '/resume-analytics',
      'chatbot retrieval should rank the analytics resume first for generic resume questions');
    const dataScienceResumeRetrieval = retrieveKnowledge('Where is the best resume?', { url: '/data-science', title: 'Data Science', audience: 'data-science' });
    assert(dataScienceResumeRetrieval.chunks[0] && dataScienceResumeRetrieval.chunks[0].url === '/resume-data-science',
      'chatbot retrieval should keep generic resume questions scoped to the data science audience lens');
    const tourismResumeRetrieval = retrieveKnowledge('Where is the best resume?', { url: '/tourism', title: 'Tourism', audience: 'tourism' });
    assert(tourismResumeRetrieval.chunks[0] && tourismResumeRetrieval.chunks[0].url === '/resume-tourism',
      'chatbot retrieval should keep generic resume questions scoped to the tourism audience lens');
    const projectRetrieval = retrieveKnowledge('What projects show SQL and Tableau experience?', { url: '/analytics', title: 'Analytics' });
    assert(projectRetrieval.chunks[0] && String(projectRetrieval.chunks[0].url || '').startsWith('/portfolio/'),
      'chatbot retrieval should keep project questions focused on portfolio projects');
    const chatbotApiPrivate = require('./api/chatbot.js')._private;
    const dataScienceLinks = chatbotApiPrivate.suggestedLinksFromRetrieval(
      'Where is the best resume?',
      { chunks: [], audience: 'data-science' },
      { includeDefaults: true, pageContext: { audience: 'data-science' } }
    );
    assert(dataScienceLinks.some((link) => link.url === '/resume-data-science') &&
      !dataScienceLinks.some((link) => link.url === '/resume-analytics'),
      'chatbot suggested links should respect the active data science audience lens');
    const tourismNavigation = chatbotApiPrivate.navigationAnswer(
      'Which resume fits this role?',
      { chunks: [], audience: 'tourism', bestScore: 0 },
      { audience: 'tourism' }
    );
    assert(tourismNavigation && tourismNavigation.answer.includes('Tourism Resume') &&
      tourismNavigation.answer.includes('[Tourism Resume](/resume-tourism)') &&
      tourismNavigation.suggestedLinks.some((link) => link.url === '/resume-tourism'),
      'chatbot navigation answers should advocate within the active tourism audience lens with inline links');
    const analyticsProofNavigation = chatbotApiPrivate.navigationAnswer(
      'Show analytics portfolio proof for this resume',
      { chunks: [], audience: 'analytics', bestScore: 0 },
      { audience: 'analytics' }
    );
    assert(analyticsProofNavigation && analyticsProofNavigation.answer.includes('Analytics Portfolio') &&
      analyticsProofNavigation.answer.includes('/portfolio?audience=analytics') &&
      !analyticsProofNavigation.answer.includes('Use the [Analytics Resume]'),
      'chatbot navigation answers should route proof follow-ups to the portfolio instead of repeating resume guidance');
    assert(chatbotApiPrivate.ensureInlineLinks('Daniel has strong analytics proof.', [{ title: 'Analytics Resume', url: '/resume-analytics' }]) ===
      'Daniel has strong analytics proof.\n\nSee: [Analytics Resume](/resume-analytics).',
      'chatbot API should add a concise inline link when a model answer omits one');
    assert(chatbotApiPrivate.ensureInlineLinks(
      'Review [Store-Level Loss & Sales ETL](/analytics).',
      [{ title: 'Data Analytics', url: '/analytics' }, { title: 'Analytics Portfolio', url: '/portfolio?audience=analytics' }]
    ) === 'Review Store-Level Loss & Sales ETL.\n\nSee: [Data Analytics](/analytics) and [Analytics Portfolio](/portfolio?audience=analytics).',
      'chatbot API should strip mismatched inline links and append accurate website links');
    assert(chatbotApiPrivate.retrievalOnlyAnswer(
      'Show analytics proof',
      { chunks: [{ title: 'Analytics Resume', url: '/resume-analytics', category: 'Resume', text: 'Daniel built dashboards and automated reporting.' }], audience: 'analytics', bestScore: 8 },
      { audience: 'analytics' }
    ).answer.includes('[Analytics Resume](/resume-analytics)'),
      'chatbot retrieval fallback should include inline website links');
    assert(!chatbotApiPrivate.buildContext([{ title: 'Store-Level Loss & Sales ETL', url: '/portfolio/retailStore', category: 'Project', text: 'Project Overview' }]).includes('[1]') &&
      chatbotApiPrivate.buildContext([{ title: 'Store-Level Loss & Sales ETL', url: '/portfolio/retailStore', category: 'Project', text: 'Project Overview' }]).includes('Source: Store-Level Loss & Sales ETL'),
      'chatbot model context should not number source titles in a way that leaks citation markers into answers');
    assert(chatbotApiPrivate.stripSourceCitations('Store-Level Loss & Sales ETL [3]\n- Project Overview') === 'Store-Level Loss & Sales ETL\n- Project Overview',
      'chatbot API should remove bracketed numeric source citations from visible answers');
    const nextFollowups = chatbotApiPrivate.makeFollowups(
      "Show project proof of Daniel's analytics impact",
      'Daniel has analytics proof in SQL ETL, dashboarding, and forecasting projects.',
      { chunks: [], audience: 'analytics', queryTerms: ['analytics', 'portfolio', 'projects'], retrievalMode: 'lexical' },
      { audience: 'analytics', title: 'Analytics Portfolio' },
      [{ role: 'user', text: 'Why is Daniel a strong analytics candidate?' }],
      [{ title: 'Analytics Portfolio', url: '/portfolio?audience=analytics' }],
      {
        source: 'recommended_followup',
        prompt: "Show project proof of Daniel's analytics impact",
        previousQuestion: 'Why is Daniel a strong analytics candidate?'
      }
    );
    const blockedFollowupPhrases = [
      'Which resume matches this work?',
      'Show similar projects',
      'Summarize the strongest project',
      'What should I look at next?',
      'Which skills should a recruiter notice here?',
      'What should recruiters notice'
    ];
    assert(!nextFollowups.includes("Show project proof of Daniel's analytics impact") &&
      !nextFollowups.includes('Why is Daniel a strong analytics candidate?') &&
      nextFollowups.some((item) => /skills|team|fit|candidate|impact|project evidence/i.test(item)) &&
      blockedFollowupPhrases.every((phrase) => !nextFollowups.includes(phrase)),
      'chatbot follow-up chips should not repeat clicked prompts and should advance readers toward Daniel skill evidence');
    const parsedModelFollowups = chatbotApiPrivate.parseFollowupJson('```json\n{"followups":["Which project proves Daniel analytics impact best?","How would Daniel SQL work help an analytics team?","What should I ask Daniel before contacting him?"]}\n```');
    assert(parsedModelFollowups.length === 3 &&
      parsedModelFollowups[0].includes('project proves'),
      'chatbot API should parse JSON follow-up chips from model responses');
    const validatedModelFollowups = chatbotApiPrivate.validateModelFollowups([
      "Which project proves Daniel's analytics impact best?",
      "How would Daniel's SQL dashboards help an analytics team?",
      'What should I ask Daniel before scheduling an interview?',
      'How did Daniel improve revenue by 999%?',
      'Which tourism project should I review next?',
      'Why is Daniel a strong analytics candidate?'
    ], {
      message: 'Why is Daniel a strong analytics candidate?',
      answer: 'Daniel has SQL and Tableau proof, including 99% faster reporting turnaround and 200+ hours saved annually.',
      retrieval: { chunks: [], audience: 'analytics', queryTerms: ['analytics'] },
      pageContext: { audience: 'analytics', title: 'Analytics' },
      history: [{ role: 'user', text: 'What analytics skills does Daniel demonstrate?' }]
    });
    assert(validatedModelFollowups.length === 3 &&
      validatedModelFollowups.every((item) => /\?$/.test(item)) &&
      !validatedModelFollowups.some((item) => /999|tourism|strong analytics candidate/i.test(item)),
      'chatbot API should validate model-written follow-up chips for novelty, audience fit, and supported claims');
    const modelFollowupPrompt = chatbotApiPrivate.buildModelFollowupPrompt(
      'Summarize Daniel analytics fit',
      'Daniel connects SQL reporting, dashboards, and business outcomes.',
      { chunks: [{ title: 'Analytics Resume', url: '/resume-analytics', category: 'Resume', text: 'SQL, Tableau, reporting automation.' }], audience: 'analytics' },
      { audience: 'analytics', title: 'Analytics' },
      [],
      [{ title: 'Analytics Resume', url: '/resume-analytics' }],
      null
    );
    assert(modelFollowupPrompt.includes('"activeAudience": "Analytics"') &&
      modelFollowupPrompt.includes('99% faster reporting turnaround') &&
      modelFollowupPrompt.includes('Return exactly 3 follow-up questions'),
      'chatbot API should give the model audience-specific evidence and strict follow-up rules');

    const api = readFile('api/chatbot.js');
    const knowledgeLib = readFile('api/_lib/chatbot-knowledge.js');
    const rateLimit = readFile('api/_lib/chatbot-rate-limit.js');
    const logStore = readFile('api/_lib/chatbot-logs.js');
    const logApi = `${readFile('api/_lib/chatbot-logs-api.js')}\n${readFile('api/chatbot/logs.js')}`;
    const envExample = readFile('.env.example');
    assert(api.includes("DEFAULT_MODEL_ID = 'us.amazon.nova-lite-v1:0'"), 'chatbot API should default to the Nova Lite inference profile');
    assert(api.includes('ConverseCommand') && api.includes('ConverseStreamCommand') && api.includes('InvokeModelCommand') &&
      api.includes('retrieveKnowledge') && api.includes('checkChatbotRateLimit'),
      'chatbot API should use Bedrock chat, streaming, embeddings, retrieval, and rate limiting');
    assert(api.includes('verifyTurnstile') && api.includes('CHATBOT_TURNSTILE_SECRET_KEY') && api.includes('CHATBOT_ENABLED'),
      'chatbot API should gate requests and verify Turnstile challenges');
    assert(api.includes('recordChatbotLog') && api.includes('suggestedLinksFromRetrieval') && api.includes('navigationAnswer') && api.includes('isLogsRoute') && api.includes('logId'),
      'chatbot API should record logs, serve the admin logs route, and return deterministic navigation suggestions');
    assert(api.includes('stripSourceCitations') &&
      api.includes('Do not include bracketed numeric citations') &&
      api.includes('Source: ${chunk.title}') &&
      !api.includes('Mention source numbers like [1]'),
      'chatbot API should avoid source-number citations in model prompts and visible answers');
    assert(api.includes('Be concise by default') &&
      api.includes('one short paragraph or at most two bullets') &&
      api.includes('include one or two inline markdown links') &&
      api.includes('Number(process.env.CHATBOT_MAX_OUTPUT_TOKENS) || 180') &&
      api.includes('ensureInlineLinks') &&
      api.includes('sanitizeInlineLinks') &&
      !api.includes('moderately detailed') &&
      !api.includes('two to four short paragraphs'),
      'chatbot API should keep default answers concise and inline-linked');
    assert(api.includes("boolEnv('CHATBOT_ENABLED', true)") && api.includes('retrievalOnlyAnswer') && api.includes("status: 'model_fallback'"),
      'chatbot API should be enabled by default and fall back to retrieval-only answers when Bedrock is unavailable');
    assert(api.includes('streamModelAnswer') &&
      api.includes("writeStreamEvent(res, 'token'") &&
      api.includes('normalizeHistory') &&
      api.includes('followupContext') &&
      api.includes('wantsFreshFollowupAnswer') &&
      api.includes("const navigation = freshFollowupAnswer ? null : navigationAnswer") &&
      api.includes('Do not repeat the previous answer') &&
      api.includes('makeFollowups') &&
      api.includes('generateModelFollowups') &&
      api.includes('validateModelFollowups') &&
      api.includes('CHATBOT_MODEL_FOLLOWUPS_ENABLED') &&
      api.includes('retrievalMode'),
      'chatbot API should stream answers and return model-written, conversation-aware follow-up metadata');
    blockedFollowupPhrases.forEach((phrase) => {
      assert(!api.includes(phrase), `chatbot API should not generate awkward follow-up phrase: ${phrase}`);
    });
    assert(api.includes('What skills does Daniel demonstrate in this project?') &&
      api.includes("How would Daniel's work help a team?") &&
      api.includes("Why is Daniel a strong ${roleLabel} candidate?"),
      'chatbot API follow-up chips should use natural visitor wording and advocate for Daniel');
    assert(api.includes('embedQuery') &&
      api.includes('CHATBOT_BEDROCK_EMBED_MODEL_ID') &&
      api.includes('retrievalMode') &&
      api.includes('CHATBOT_EMBEDDINGS_REQUIRED'),
      'chatbot API should use query embeddings with lexical fallback');
    assert(knowledgeLib.includes('loadKnowledge') && knowledgeLib.includes('scoreChunk') && knowledgeLib.includes('detectQueryIntent') &&
      knowledgeLib.includes('dotProduct') && knowledgeLib.includes('queryEmbedding') && knowledgeLib.includes("retrievalMode: canUseEmbedding ? 'embedding' : 'lexical'") &&
      knowledgeLib.includes('publicSources') && knowledgeLib.includes('scoreAudienceBoost'),
      'chatbot knowledge helper should load, score, rank hybrid embedding results, expose sources, and support audience boosts');
    ['CHATBOT_DDB_TABLE', 'CHATBOT_HASH_SALT', 'CHATBOT_WINDOW_LIMIT', 'CHATBOT_DAILY_LIMIT', 'CHATBOT_GLOBAL_DAILY_LIMIT'].forEach((name) => {
      assert(rateLimit.includes(name), `chatbot rate limiter missing ${name}`);
      assert(envExample.includes(name), `.env.example missing ${name}`);
    });
    ['CHATBOT_LOG_TTL_DAYS', 'CHATBOT_ADMIN_TOKEN'].forEach((name) => {
      assert(logStore.includes(name) || logApi.includes(name), `chatbot logging missing ${name}`);
      assert(envExample.includes(name), `.env.example missing ${name}`);
    });
    ['CHATBOT_EMBEDDINGS_ENABLED', 'CHATBOT_EMBEDDINGS_REQUIRED', 'CHATBOT_BEDROCK_EMBED_MODEL_ID', 'CHATBOT_BEDROCK_EMBED_DIMENSIONS'].forEach((name) => {
      assert(envExample.includes(name), `.env.example missing ${name}`);
    });
    assert(rateLimit.includes('isProductionRuntime') && rateLimit.includes('memoryStore'),
      'chatbot rate limiter should have production guards and local fallback');
    assert(rateLimit.includes("boolEnv('CHATBOT_REQUIRE_DDB', false)") && rateLimit.includes('requiresDdbRateLimit'),
      'chatbot rate limiter should only require DynamoDB when explicitly configured');
    assert(logStore.includes('recordChatbotLog') && logStore.includes('listChatbotLogs') && logStore.includes('getChatbotLog'),
      'chatbot log store should record and read investigation logs');
    assert(logStore.includes('CHATBOT#LOGS') && logStore.includes('ttl') && logStore.includes('actorHash'),
      'chatbot logs should be indexed with TTL and hashed actor metadata');
    assert(!logStore.includes('x-forwarded-for') && !logStore.includes('remoteAddress'),
      'chatbot logs should not store raw IP address headers');
    assert(logApi.includes('CHATBOT_ADMIN_TOKEN') && logApi.includes('isAdminRequest') && logApi.includes('listChatbotLogs'),
      'chatbot log API should be admin-token protected');

    const widget = readFile('js/chatbot/site-chatbot.js');
    const entry = readFile('build/entries/site-shell.entry.js');
    const css = readFile('css/components/site-chatbot.css');
    const cssImports = readFile('css/styles.css');
    const footerTemplate = readFile('build/templates/footer.partial.html');
    assert(entry.includes('../../js/chatbot/site-chatbot.js'), 'site shell should load chatbot widget');
    assert(cssImports.includes('components/site-chatbot.css'), 'main stylesheet should import chatbot styles');
    assert(widget.includes("const API_PATH = '/api/chatbot'") && widget.includes('conversationId') && widget.includes('turnstile'),
      'chatbot widget should call the API, keep a conversation id, and handle challenges');
    assert(widget.includes('data-chatbot-prompt') &&
      widget.includes('suggestedLinks') &&
      widget.includes('sourceLinks = normalizeLinks(sources, 5)') &&
      !widget.includes("navList.className = 'site-chatbot__nav-links'") &&
      !widget.includes('const navLinks = normalizeLinks(suggestedLinks'),
      'chatbot widget should offer quick prompts and keep website links inline instead of rendering suggested-link chips');
    assert(widget.includes("UI_VERSION = 'fresh-followups-2026-05-12'") &&
      widget.includes('root.dataset.chatbotVersion = UI_VERSION') &&
      widget.includes('scheduleInitialNudge();') &&
      widget.includes('if (!state.ready) loadConfig();'),
      'chatbot widget should expose a current UI marker and preflight availability so nudge and prompts are not stale');
    assert(widget.includes("body.dataset.siteChatbotActive = 'true'"),
      'chatbot widget should mark chatbot pages so the bottom-right chatbot replaces the speed dial');
    assert(widget.includes('site-chatbot__header-expand') &&
      widget.includes("headerExpand.addEventListener('click', () => requestExpanded('header'))") &&
      widget.includes("headerToggle.addEventListener('click', () => toggleExpanded('toggle'))") &&
      widget.includes('function requestCollapsed') &&
      widget.includes('if (!state.expanded) return;'),
      'chatbot header should expand only while the chevron explicitly toggles expansion');
    assert(widget.includes("'project'") && widget.includes("'resume-tourism'") && !widget.includes("'tools'") && !widget.includes("'games'"),
      'chatbot widget should be scoped to public content pages');
    assert(widget.includes('window.visualViewport') && widget.includes("root.dataset.keyboard") && widget.includes('--site-chatbot-keyboard-offset'),
      'chatbot widget should track mobile keyboard viewport changes');
    assert(widget.includes('NUDGE_STORAGE_KEY') &&
      widget.includes('NUDGE_DESKTOP_DELAY_MS = 6000') &&
      widget.includes('NUDGE_MOBILE_DELAY_MS = 10000') &&
      widget.includes('NUDGE_AUTO_DISMISS_MS = 6000') &&
      widget.includes('NUDGE_MOBILE_SCROLL_RATIO = 0.35') &&
      widget.includes('localStorage.getItem(NUDGE_STORAGE_KEY)') &&
      widget.includes('localStorage.setItem(NUDGE_STORAGE_KEY') &&
      widget.includes('root.dataset.nudge') &&
      widget.includes("root.dataset.enabled !== 'true'") &&
      widget.includes('site-chatbot__nudge-action') &&
      widget.includes('isConsentBannerOpen') &&
      widget.includes('hasScrolledEnoughForNudge') &&
      widget.includes('scheduleInitialNudge') &&
      widget.includes('dismissNudge'), 'chatbot widget should show a targeted once-per-visitor launcher nudge');
    assert(widget.includes('chatbot_nudge_shown') &&
      widget.includes('chatbot_nudge_dismissed') &&
      widget.includes('chatbot_nudge_opened') &&
      widget.includes('chatbot_launcher_opened'), 'chatbot widget should track launcher and nudge events');
    assert(widget.includes('getQuickPrompts') &&
      widget.includes('SQL proof') &&
      widget.includes("What analytics projects prove Daniel's SQL and reporting strength?") &&
      widget.includes('Model proof') &&
      widget.includes("Which projects show Daniel's machine learning and Python skills?") &&
      widget.includes('DMO fit') &&
      widget.includes("What ${config.roleLabel} skills does Daniel demonstrate in this project?") &&
      widget.includes("How would Daniel's work help a team?") &&
      !widget.includes('recruiter notice') &&
      !widget.includes('Show similar projects') &&
      !widget.includes('opener:') &&
      !widget.includes('config.opener') &&
      !widget.includes('Ask me about Daniel Short as'),
      'chatbot widget should render three role-aware default questions without an intro paragraph');
    blockedFollowupPhrases.forEach((phrase) => {
      assert(!widget.includes(phrase), `chatbot widget should not generate awkward follow-up phrase: ${phrase}`);
    });
    assert(widget.includes('SESSION_STATE_KEY') &&
      widget.includes('handleChatbotLinkClick') &&
      widget.includes('restoreOpenAfterChatbotNavigation') &&
      widget.includes('audience: state.audience ||') &&
      widget.includes('hideStarterPrompts') &&
      widget.includes('clearFollowups') &&
      widget.includes('starterPromptsHidden') &&
      widget.includes('site-chatbot__reset') &&
      widget.includes('resetChat') &&
      widget.includes('resetConversationId') &&
      widget.includes('handleOutsidePointerDown') &&
      widget.includes("document.addEventListener('pointerdown', handleOutsidePointerDown, true)") &&
      widget.includes('DISPLAY_TRANSCRIPT_MAX_CHARS = 6000') &&
      widget.includes('HISTORY_TRANSCRIPT_MAX_CHARS = 700') &&
      widget.includes('normalizeTranscriptMarkdown') &&
      widget.includes(".replace(/\\r\\n?/g, '\\n')") &&
      widget.includes('apiTranscript'),
      'chatbot widget should persist role context, stay open across internal chatbot navigation, hide all chips after a chip is chosen, reset conversations, close on outside clicks, and keep restored markdown transcripts readable');
    assert(widget.includes('new AbortController()') &&
      widget.includes("accept: 'application/x-ndjson, application/json'") &&
      widget.includes('submitStreamingRequest') &&
      widget.includes("event.type === 'token'") &&
      widget.includes('abortActiveRequest') &&
      widget.includes("sendButton.setAttribute('aria-label', state.sending ? 'Stop response' : 'Send question')"),
      'chatbot widget should stream answers and let the send button stop an active response');
    assert(widget.includes('state.transcript') &&
      widget.includes('rememberTurn') &&
      widget.includes('normalizeStoredLinks') &&
      widget.includes('previousQuestion') &&
      widget.includes("options.previousQuestion || '', text") &&
      widget.includes('followupContext') &&
      widget.includes("source: 'recommended_followup'") &&
      widget.includes('site-chatbot__followups'),
      'chatbot widget should keep recent history and restore conversation-aware follow-up chips');
    assert(widget.includes('renderMarkdown') &&
      widget.includes('appendInlineMarkdown') &&
      widget.includes('autoLinkSourcePhrases') &&
      widget.includes('normalizeLinks(sources, 8)') &&
      widget.includes('Sources (${sourceLinks.length})'),
      'chatbot widget should render safe markdown and richer source links');
    assert(widget.includes('[data-site-chatbot-open]') &&
      footerTemplate.includes('data-site-chatbot-open hidden') &&
      footerTemplate.includes('Ask the site assistant'), 'footer should expose a JS-enabled chatbot help entry');
    assert(css.includes('.site-chatbot__panel') && css.includes('@media (max-width: 640px)'),
      'chatbot CSS should include panel and mobile behavior');
    assert(css.includes('.site-chatbot__input-shell') &&
      css.includes('.site-chatbot__input-shell:focus-within') &&
      css.includes('outline: 3px solid rgba(21, 101, 192, 0.16);') &&
      css.includes('gap: 0;') &&
      css.includes('align-items: stretch;') &&
      css.includes('--site-chatbot-input-height: 64px;') &&
      css.includes('height: var(--site-chatbot-input-height);') &&
      css.includes('max-height: var(--site-chatbot-input-height);') &&
      css.includes('resize: none;') &&
      css.includes('border-radius: 11px 0 0 11px;') &&
      css.includes('border-radius: 0 10px 10px 0;') &&
      css.includes('.site-chatbot__input:focus{\n    outline: none;'),
      'chatbot input and send button should be a fixed two-line flush control with one shared focus ring');
    assert(css.includes('.site-chatbot__followups') &&
      css.includes('.site-chatbot__followups button{\n    flex: 1 1 100%;') &&
      css.includes('.site-chatbot__bubble ul') &&
      css.includes('.site-chatbot__bubble a') &&
      css.includes('overflow-wrap: anywhere;') &&
      css.includes('text-overflow: clip;') &&
      css.includes('overflow-x: hidden;') &&
      css.includes('.site-chatbot__quick-prompts{\n    order: 1;\n    display: grid;') &&
      css.includes('.site-chatbot__quick-prompts[hidden]') &&
      css.includes('display: none !important;') &&
      css.includes('grid-template-columns: minmax(0, 1fr);') &&
      css.includes('text-align: left;') &&
      css.includes('white-space: normal;'),
      'chatbot CSS should style markdown answers, list default questions, wrap long chatbot text, and avoid horizontal prompt scrolling');
    assert(!css.includes('.site-chatbot[data-enabled="false"] .site-chatbot__quick-prompts,'),
      'chatbot quick prompt chips should remain visible even when the assistant falls back');
    assert(css.includes('.site-chatbot[data-nudge="true"][data-state="closed"] .site-chatbot__nudge') &&
      css.includes('.site-chatbot__nudge-close') &&
      css.includes('pointer-events: auto;') &&
      css.includes('@keyframes site-chatbot-nudge-pulse') &&
      css.includes('width: max-content;') &&
      css.includes('bottom: calc(100% + 10px);') &&
      css.includes('animation: none;'), 'chatbot CSS should style a responsive launcher nudge and respect reduced motion');
    assert(css.includes('body[data-site-chatbot-active="true"] .speed-dial') && css.includes('transform: translateY(18px) scale(0.96)'),
      'chatbot CSS should replace the speed dial and animate the expanded panel');
    assert(css.includes('--site-chatbot-panel-max-height: min(520px') &&
      css.includes('--site-chatbot-center-height: clamp(180px') &&
      css.includes('--site-chatbot-expanded-height: calc(') &&
      css.includes('--site-chatbot-panel-max-height: var(--site-chatbot-expanded-height);') &&
      css.includes('--site-chatbot-center-height: max(220px') &&
      css.includes('.site-chatbot[data-expanded="true"][data-chips="hidden"] .site-chatbot__panel') &&
      css.includes('grid-template-rows: auto minmax(0, 1fr) auto auto auto;') &&
      css.includes('.site-chatbot__status:empty') &&
      css.includes('.site-chatbot__messages') &&
      css.includes('height: var(--site-chatbot-center-height);') &&
      css.includes('transition: height 280ms cubic-bezier(0.2, 0.8, 0.2, 1);'),
      'chatbot expansion should use the available viewport height while keeping header and input chrome fixed-size without a bottom gap');
    assert(css.includes('--site-chatbot-viewport-height') &&
      css.includes('.site-chatbot[data-keyboard="true"][data-state="open"] .site-chatbot__panel') &&
      css.includes('var(--site-chatbot-consent-offset) - var(--site-chatbot-keyboard-offset)') &&
      css.includes('var(--site-chatbot-keyboard-offset)') &&
      css.includes('--site-chatbot-panel-max-height: max(260px') &&
      css.includes('-webkit-text-fill-color: #fff;'),
      'chatbot CSS should keep user text white and keep the mobile panel above the keyboard');
    assert(css.includes('.site-chatbot[data-expanded="false"] .site-chatbot__header') &&
      css.includes('.site-chatbot[data-expanded="true"] .site-chatbot__header-expand') &&
      css.includes('.site-chatbot__header-toggle{') &&
      css.includes('.site-chatbot__reset'),
      'chatbot CSS should distinguish expand-only header, reset, and explicit chevron controls');
    assert(css.includes('.site-chatbot__expand-icon svg') &&
      css.includes('transform: rotate(180deg);') &&
      css.includes('.site-chatbot[data-expanded="true"] .site-chatbot__expand-icon svg') &&
      css.includes('transform: rotate(0deg);'),
      'chatbot expand icon should point up before expanding and down before retracting');

    const vercel = readFile('vercel.json');
    assert(vercel.includes('https://challenges.cloudflare.com'), 'CSP should allow Cloudflare Turnstile');
  });

  section('AI digest contracts', () => {
    const buildRunner = readFile('build/build-site.js');
    const copyPublic = readFile('build/copy-to-public.js');
    const devServer = readFile('build/dev.js');
    const generator = readFile('build/generate-ai-digests.js');
    const userAgents = readFile('build/lib/ai-bot-user-agents.js');
    const robots = readFile('robots.txt');
    const vercelConfig = JSON.parse(readFile('vercel.json'));
    const rewrites = Array.isArray(vercelConfig.rewrites) ? vercelConfig.rewrites : [];
    const headers = Array.isArray(vercelConfig.headers) ? vercelConfig.headers : [];

    assert(buildRunner.includes('generate-ai-digests.js'), 'build runner should generate AI digests');
    assert(copyPublic.includes("'ai-digest-manifest.json'") && copyPublic.includes("'dist', 'ai-pages'"),
      'public copy should include AI digest artifacts');
    assert(devServer.includes('hasConditionsMatch') && !devServer.includes('aiPageApiPath') && !devServer.includes('dispatchAiPageApi'),
      'local dev server should test header-based static digest rewrites without an AI digest API');
    assert(generator.includes('data-ai-digest') && generator.includes('ai-digest-manifest.json') && generator.includes('aiDigest'),
      'AI digest generator should emit marked deterministic digests and support overrides');
    assert(userAgents.includes('GPTBot') && userAgents.includes('ChatGPT-User') && userAgents.includes('ClaudeBot') && userAgents.includes('PerplexityBot'),
      'AI user-agent matcher should include major AI retrieval bots');
    assert(!fs.existsSync('api/ai-page/[...path].js') && !fs.existsSync('api/ai-page'),
      'AI digest delivery should not add a Vercel serverless function');
    assert(robots.includes('User-agent: GPTBot') && robots.includes('User-agent: ClaudeBot') && robots.includes('User-agent: PerplexityBot'),
      'robots.txt should explicitly allow major AI bots');
    assert(robots.includes('Disallow: /dist/ai-pages/') && robots.includes('Disallow: /ai/'),
      'robots.txt should hide duplicate AI digest implementation paths');
    assert(headers.some((rule) => rule.source === '/ai/(.*)' && JSON.stringify(rule.headers).includes('X-Robots-Tag')),
      'debug AI digest paths should be noindex');
    assert(headers.some((rule) => rule.source === '/dist/ai-pages/(.*)' && JSON.stringify(rule.headers).includes('X-Robots-Tag')),
      'static AI digest implementation paths should be noindex');
    assert(!JSON.stringify(rewrites).includes('/api/ai-page'), 'AI digest rewrites should not target serverless functions');

    assert(rewrites.some((rule) => rule.source === '/ai/:path*' && rule.destination === '/dist/ai-pages/:path*'),
      'vercel rewrites should expose /ai debug digests');
    assert(rewrites.some((rule) => rule.source === '/analytics' && /user-agent/i.test(JSON.stringify(rule)) && rule.destination === '/dist/ai-pages/analytics'),
      'vercel rewrites should serve same-URL analytics digest to AI agents');
    assert(rewrites.some((rule) => rule.source === '/portfolio/:project' && rule.destination === '/dist/ai-pages/portfolio/:project'),
      'vercel rewrites should serve portfolio project digests to AI agents');
    assert(rewrites.some((rule) => rule.source === '/tools/:tool' && rule.destination === '/dist/ai-pages/tools/:tool'),
      'vercel rewrites should serve public tool digests to AI agents');

    assert(fs.existsSync('dist/ai-digest-manifest.json'), 'AI digest manifest missing');
    const manifest = JSON.parse(readFile('dist/ai-digest-manifest.json'));
    assert(manifest.version === 1, 'AI digest manifest should declare version 1');
    assert(manifest.origin === 'https://www.danielshort.me', 'AI digest manifest should use the public site origin');
    assert(Array.isArray(manifest.pages) && manifest.pages.length >= 10, 'AI digest manifest should include public pages');
    const urls = new Set(manifest.pages.map((page) => String(page && page.url || '').trim()));
    ['/analytics', '/data-science', '/tourism', '/portfolio', '/resume-analytics', '/tools/text-compare'].forEach((url) => {
      assert(urls.has(url), `AI digest manifest missing ${url}`);
    });
    ['/search', '/tools/dashboard', '/tools/short-links', '/resume-analytics-pdf'].forEach((url) => {
      assert(!urls.has(url), `AI digest manifest should exclude ${url}`);
    });

    const analyticsDigest = readFile('dist/ai-pages/analytics.html');
    assert(analyticsDigest.includes('data-ai-digest="true"'), 'analytics AI digest should be marked');
    assert(analyticsDigest.includes('<link rel="canonical" href="https://www.danielshort.me/analytics">'),
      'analytics AI digest should preserve canonical URL');
    assert(!/<script\b/i.test(analyticsDigest), 'AI digest pages should not include scripts');
    assert(!/<nav\b/i.test(analyticsDigest), 'AI digest pages should not include nav chrome');
    assert(analyticsDigest.includes('Key Facts') && analyticsDigest.includes('Source Metadata'),
      'AI digest pages should include key facts and source metadata');
  });

  section('Local CMS contracts', () => {
    const pkg = JSON.parse(readFile('package.json'));
    assert(!pkg.devDependencies || !pkg.devDependencies.vercel, 'vercel should not be a devDependency');
    assert(!pkg.dependencies || !pkg.dependencies.vercel, 'vercel should not be a dependency');
    assert(!pkg.scripts['cms:seed'] && !pkg.scripts['cms:export'] && !pkg.scripts['cms:diff'],
      'database CMS migration scripts should not be exposed');

    const cmsModel = require('./api/_lib/cms-content-model');
    assert(Array.isArray(cmsModel.CMS_COLLECTIONS) && cmsModel.CMS_COLLECTIONS.length === 6, 'CMS collections should define all managed content groups');
    ['site', 'pages', 'audiences', 'resumes', 'projects', 'tools'].forEach((collection) => {
      assert(cmsModel.CMS_COLLECTION_NAMES.includes(collection), `CMS collection missing ${collection}`);
    });

    const records = cmsModel.listFileContentRecords(process.cwd());
    assert(records.length >= 40, 'CMS file record catalog unexpectedly small');
    assert(records.some((record) => record.collection === 'site' && record.id === 'settings'), 'CMS records missing site/settings');
    assert(records.some((record) => record.collection === 'tools' && record.id === 'word-frequency'), 'CMS records missing word-frequency tool');

    const fileContent = cmsModel.loadFileSiteContent(process.cwd());
    assert(fileContent.site.settings && fileContent.site.settings.siteName, 'file-backed CMS content missing site settings');
    assert(fileContent.audiencesByKey.analytics, 'file-backed CMS content missing analytics audience');
    assert(fileContent.projectsById.website, 'file-backed CMS content missing website project');

    const loader = require('./build/lib/content-loader');
    assert(loader.normalizeContentSource('files') === 'files', 'content loader should default to files source');
    assert(loader.normalizeContentSource('db') === 'files', 'content loader should not support DB source');

    const adminHtml = readFile('admin/index.html');
    const adminJs = readFile('admin/cms-admin.js');
    const cmsApi = readFile('api/cms/[...slug].js');
    const cmsStore = readFile('api/_lib/cms-file-store.js');
    const cmsLibraryStore = readFile('api/_lib/cms-library-store.js');
    const cmsSnapshotStore = readFile('api/_lib/cms-snapshot-store.js');
    const cmsWidgets = readFile('api/_lib/cms-widgets.js');
    const devJs = readFile('build/dev.js');
    const localDevWsl = readFile('start-local-dev-wsl.bat');
    const copyJs = readFile('build/copy-to-public.js');
    const envExample = readFile('.env.example');

    assert(adminHtml.includes('/admin/cms-admin.js'), 'admin should load custom CMS admin script');
    assert(adminHtml.includes('cms-builder-layout') && adminHtml.includes('data-cms="preview"'),
      'admin should default to the visual builder with live preview');
    assert(adminHtml.includes('data-cms-view-target="dashboard"') && adminHtml.includes('data-cms-view="library"') && adminHtml.includes('data-cms="global-header"'),
      'admin should expose dashboard, library, and globals CMS views');
    assert(adminHtml.includes('<select data-cms="ollama-model"') && adminHtml.includes('data-cms="ollama-refresh"') && adminHtml.includes('Preview AI Edit'),
      'admin should expose installed Ollama models and before/after AI review controls');
    assert(adminHtml.includes('data-cms-preview-device="desktop"') && adminHtml.includes('data-cms="preview-open"'),
      'admin should expose responsive live preview controls');
    assert(adminHtml.includes('data-cms="dashboard-health"') &&
           adminHtml.includes('data-cms="dashboard-snapshots"') &&
           adminHtml.includes('data-cms="preview-audience"') &&
           adminHtml.includes('data-cms-ai-action="metadata"'),
      'admin should expose CMS health, local snapshots, audience preview, and field-aware AI shortcuts');
    assert(!adminHtml.includes('/js/accounts/tools-auth.js'), 'local CMS admin should not load Cognito tools auth');
    assert(!adminHtml.toLowerCase().includes('decap'), 'admin should not load Decap');
    assert(!adminHtml.includes('unpkg.com'), 'admin should not depend on a CMS CDN');
    assert(!adminHtml.includes('data-cms="deploy"'), 'admin should not expose deploy hook controls');
    assert(!fs.existsSync('admin/config.yml'), 'Decap config should not be published under admin/');
    assert(!fs.existsSync('api/decap/auth.js') && !fs.existsSync('api/decap/callback.js'), 'Decap OAuth endpoints should be removed');
    assert(!fs.existsSync('api/_lib/cms-auth.js'), 'CMS Cognito auth helper should not exist');
    assert(!fs.existsSync('api/_lib/cms-store-ddb.js'), 'CMS DynamoDB store should not exist');
    assert(!fs.existsSync('build/cms-seed.js') && !fs.existsSync('build/cms-export.js') && !fs.existsSync('build/cms-diff.js'),
      'CMS database scripts should be removed');

    assert(adminJs.includes("API_BASE = '/api/cms'") && adminJs.includes("apiFetch('/content'"), 'admin script missing content API path');
    assert(adminJs.includes("apiFetch('/widgets'") && adminJs.includes("apiFetch('/preview'") && adminJs.includes("apiFetch('/media'"),
      'admin script should load widgets, media, and render live previews');
    assert(adminJs.includes("'/preview-tool'") &&
           adminJs.includes("'/preview-project'") &&
           adminJs.includes('normalizePreviewHtml'),
      'admin script should render project/tool previews and normalize iframe preview assets');
    assert(adminJs.includes("apiFetch('/library'") && adminJs.includes('local-cms-autosave-v2'),
      'admin script should use local library files and autosave recovery');
    assert(adminJs.includes("apiFetch('/health'") &&
           adminJs.includes("apiFetch('/snapshots") &&
           adminJs.includes('renderDashboardHealth') &&
           adminJs.includes('loadSnapshot'),
      'admin script should use local health checks and save-history snapshots');
    assert(adminJs.includes('projectResourceField') &&
           adminJs.includes('projectCaseField') &&
           adminJs.includes('previewAudience') &&
           adminJs.includes('sectionLockSummary'),
      'admin script should expose structured project editing, audience preview, and library section lock metadata');
    assert(adminJs.includes("apiFetch('/ollama-models'") &&
           adminJs.includes('pendingOllamaEdit') &&
           adminJs.includes('applyOllamaEditsToSnapshot') &&
           adminJs.includes('cms-assistant-review-frame') &&
           adminJs.includes('data-ollama-preview-device="desktop"') &&
           adminJs.includes('card.remove()') &&
           adminJs.includes('Accept Changes'),
      'admin script should load Ollama models and show pending visual AI edit reviews');
    assert(adminJs.includes('contenteditable') && adminJs.includes('data-cms-inline-editing'),
      'admin preview should support inline element editing');
    assert(adminJs.includes('data-section-setting') && adminJs.includes('renderProjectInspector') && adminJs.includes('renderToolInspector'),
      'admin should expose structured section, project, and tool inspectors');
    assert(adminJs.includes('Advanced JSON'), 'admin should keep JSON as an advanced fallback');
    assert(adminJs.includes('Saved to content/'), 'admin should explain local file saves');
    assert(!adminJs.includes('/deploy') && !adminJs.includes('/rollback') && !adminJs.includes('/revisions'),
      'admin script should not call deploy, rollback, or revisions endpoints');
    assert(!adminJs.includes('ToolsAuth'), 'local CMS admin should not use Cognito tools auth');

    assert(cmsApi.includes('isLocalRequest'), 'CMS API should enforce localhost access');
    assert(cmsApi.includes('saveCurrentDocument'), 'CMS API should save current documents');
    assert(cmsApi.includes('handlePreview') && cmsApi.includes('handleWidgets'),
      'CMS API should expose local preview and widget schema endpoints');
    assert(cmsApi.includes('handleProjectPreview') && cmsApi.includes('renderProjectPage'),
      'CMS API should render generated portfolio project previews');
    assert(cmsApi.includes('handleToolPreview') && cmsApi.includes('renderToolsDirectoryBody') && cmsApi.includes('handleMedia'),
      'CMS API should render tool previews and expose local media assets');
    assert(cmsApi.includes('handleLibrary') && cmsApi.includes('saveLibraryItem'),
      'CMS API should expose local CMS library endpoints');
    assert(cmsApi.includes('handleOllamaModels') && cmsApi.includes('/api/tags'),
      'CMS API should expose installed Ollama model discovery');
    assert(cmsApi.includes('handleHealth') &&
           cmsApi.includes('buildCmsHealthReport') &&
           cmsApi.includes('handleSnapshots') &&
           cmsApi.includes('usageCount') &&
           cmsApi.includes('readImageDimensions'),
      'CMS API should expose health checks, local snapshots, and enriched media metadata');
    assert(!cmsApi.includes('requireCmsAdmin'), 'CMS API should not require Cognito admin auth');
    assert(!cmsApi.includes('CMS_VERCEL_DEPLOY_HOOK_URL'), 'CMS API should not trigger deploy hooks');
    assert(!cmsApi.includes('rollbackDocument'), 'CMS API should not expose rollback support');
    assert(cmsStore.includes('writeFileSync'), 'CMS file store should write local JSON files');
    assert(cmsStore.includes('createDocumentSnapshot'), 'CMS file store should create local save snapshots before overwriting documents');
    assert(cmsStore.includes('content'), 'CMS file store should constrain writes to content/');
    assert(cmsLibraryStore.includes('content/cms-library') && cmsLibraryStore.includes('templates') && cmsLibraryStore.includes('drafts'),
      'CMS library store should constrain reusable CMS data to content/cms-library/');
    assert(cmsSnapshotStore.includes('content/cms-library') &&
           cmsSnapshotStore.includes('snapshots') &&
           cmsSnapshotStore.includes('createJsonDiffSummary'),
      'CMS snapshot store should keep local save history under content/cms-library/snapshots/');
    assert(cmsWidgets.includes("type: 'kpi-band'") &&
           cmsWidgets.includes("type: 'proof-block'") &&
           cmsWidgets.includes("type: 'media-showcase'"),
      'CMS widgets should include portfolio-focused reusable components');
    assert(fs.existsSync('content/cms-library/templates/basic-page.json'), 'CMS starter basic page template missing');
    assert(fs.existsSync('content/cms-library/sections/contact-cta.json'), 'CMS starter reusable section missing');

    assert(!envExample.includes('CMS_DDB_TABLE') &&
           !envExample.includes('CMS_VERCEL_DEPLOY_HOOK_URL') &&
           !envExample.includes('CMS_ADMIN_GROUPS'),
           '.env.example should not include CMS database/deploy env vars');

    assert(devJs.includes("const http = require('http')"), 'dev server should use Node http');
    assert(devJs.includes("loadCmsApi()(req, res)") && devJs.includes('delete require.cache[modulePath]'),
      'dev server should route local CMS API and reload CMS modules');
    assert(devJs.includes('generate-project-pages.js'), 'dev server should reload project preview renderer changes');
    assert(devJs.includes('Local CMS available'), 'dev server should advertise local CMS URL');
    assert(devJs.includes('MAX_PORT_SEARCH_ATTEMPTS') &&
           devJs.includes("err.code === 'EADDRINUSE'") &&
           devJs.includes('Port ${previousPort} is in use; trying ${candidatePort}...'),
      'dev server should automatically try the next port when the requested port is in use');
    assert(localDevWsl.includes('No available local port found') &&
           localDevWsl.includes('Port %PORT% is already in use; using %LOCAL_DEV_PORT% instead.') &&
           localDevWsl.includes('LOCAL_DEV_ENDPOINT') &&
           localDevWsl.includes("Write-Output ($hostName + '|' + $p)") &&
           localDevWsl.includes('npm run dev -- --host 0.0.0.0 --port %PORT%'),
      'WSL local dev launcher should choose and open an available port for npm run dev');
    assert(!devJs.includes('vercel dev') && !devJs.includes('npx --yes vercel'), 'dev server should not launch Vercel CLI');
    assert(copyJs.includes("path.join(outDir, 'admin')") && !/const dirs = \[[^\]]*'admin'/s.test(copyJs),
      'copy-to-public.js should keep admin local-only');
    assert(!fs.existsSync('public/admin'), 'public/admin should not be generated');
  });

  section('Short links dashboard hooks', () => {
    const html = readFile('pages/short-links.html');
    [
      'data-shortlinks="admin-tools"',
      'data-shortlinks="access-card"',
      'data-shortlinks="admin-access-summary"',
      'data-shortlinks="admin-project-summary"',
      'data-shortlinks="admin-export-summary"',
      'data-shortlinks="auth"',
      'data-shortlinks="summary"',
      'data-shortlinks="mode-tab"',
      'data-shortlinks-mode-panel="single"',
      'data-shortlinks="editor"',
      'data-shortlinks="audience-field"',
      'data-shortlinks="audience"',
      'data-shortlinks="slug-mode"',
      'data-shortlinks="expiration-mode"',
      'data-shortlinks="expiration-duration-fields"',
      'data-shortlinks="expiration-duration-value"',
      'data-shortlinks="expiration-duration-unit"',
      'data-shortlinks="random-length"',
      'data-shortlinks="create-link"',
      'data-shortlinks="editor-meta"',
      'data-shortlinks="projects-list"',
      'data-shortlinks="sets-list"',
      'data-shortlinks="set-editor"',
      'data-shortlinks="set-rows"',
      'data-shortlinks="set-generate"',
      'data-shortlinks="batch-results"',
      'data-shortlinks="export-mode"',
      'data-shortlinks="export-click-limit"',
      'data-shortlinks="export"',
      'data-shortlinks="list"'
    ].forEach((snippet) => {
      assert(html.includes(snippet), `pages/short-links.html missing expected short-links hook: ${snippet}`);
    });
    assert(!html.includes('data-shortlinks="view"'), 'pages/short-links.html should not expose the old view switch');
    assert((html.match(/data-cookie-settings="true"/g) || []).length === 1, 'pages/short-links.html should include one cookie settings widget');
    assert((html.match(/data-speed-dial="true"/g) || []).length === 1, 'pages/short-links.html should include one speed dial');
  });

  section('Short links helper logic', () => {
    const helpers = require('./api/_lib/short-links.js');
    const setHelpers = require('./api/_lib/short-links-sets.js');

    assert(helpers.normalizeSlug('Ab12Cd') === 'Ab12Cd', 'normalizeSlug should preserve case for valid slugs');
    assert(helpers.normalizeSlug('a/b-C_9') === 'a/b-C_9', 'normalizeSlug should allow nested mixed-case slugs');
    assert(helpers.normalizeSlug('bad slug') === '', 'normalizeSlug should reject spaces');
    assert(helpers.normalizeSlugLower('Ab12Cd') === 'ab12cd', 'normalizeSlugLower should lowercase normalized slugs');

    const randomSlug = helpers.generateRandomSlug(6);
    assert(/^[A-Za-z0-9]{6}$/.test(randomSlug), 'generateRandomSlug should create mixed-case alphanumeric codes');
    assert(helpers.normalizeRandomLength(3, 6) === 4, 'normalizeRandomLength should clamp to min');
    assert(helpers.normalizeRandomLength(40, 6) === 12, 'normalizeRandomLength should clamp to max');

    const template = setHelpers.buildSetTemplateRecord({
      title: ' Data Analyst Resume ',
      defaultRandomLength: 6,
      defaultExpirationMode: 'temporary',
      defaultDurationValue: 14,
      defaultDurationUnit: 'days',
      entries: [
        { label: 'Analytics site', destination: '/analytics', enabled: true },
        { label: 'LinkedIn', destination: 'https://www.linkedin.com/in/danielshort3/', enabled: true }
      ]
    }, null);

    assert(template.title === 'Data Analyst Resume', 'buildSetTemplateRecord should trim titles');
    assert(Array.isArray(template.entries) && template.entries.length === 2, 'buildSetTemplateRecord should keep valid entries');
    assert(template.entries[0].destination === '/analytics', 'buildSetTemplateRecord should preserve internal paths for template rows');

    const timing = setHelpers.resolveBatchTiming({
      expirationMode: 'temporary',
      durationValue: 2,
      durationUnit: 'weeks'
    }, template);
    assert(timing.ok === true, 'resolveBatchTiming should accept temporary durations');
    assert(timing.permanent === false, 'resolveBatchTiming should mark temporary batches as non-permanent');
    assert(timing.durationUnit === 'weeks', 'resolveBatchTiming should preserve supported duration units');
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
    assert(trackerJs.includes('buildShortlinksSetSection'), 'job tracker should build the short-link set section');
    assert(trackerJs.includes('SHORTLINKS_SETS_API_PATH'), 'job tracker should include short-links set API integration');
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

  section('Text compare core', () => {
    runTextCompareCoreTests({ assert });
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
    assert(!ids.includes('destinationReporting'), 'destinationReporting should not be a published project');
    assert(!fs.existsSync('pages/portfolio/destinationReporting.html'), 'destinationReporting page should be removed');
    assert(!sitemap.includes('https://www.danielshort.me/portfolio/destinationReporting'), 'sitemap.xml should not include destinationReporting');
    assert(!readFile('pages/portfolio.html').includes('portfolio/destinationReporting'), 'portfolio.html should not link to destinationReporting');
    assert(!readFile('pages/resume-analytics.html').includes('portfolio/destinationReporting'), 'resume-analytics should not link to destinationReporting');
    assert(!readFile('pages/resume-tourism.html').includes('portfolio/destinationReporting'), 'resume-tourism should not link to destinationReporting');

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
    const footerTemplate = fs.readFileSync('build/templates/footer.partial.html', 'utf8');
    assert(headerTemplate.includes('class="brand-logo"'), 'nav markup missing brand-logo');
    assert(headerTemplate.includes('img/brand/00-ds-logo-master-full-color.svg'), 'nav should use approved DS brand logo asset');
    assert(!headerTemplate.includes('ds-decision-path-logo'), 'nav should not use the retired decision-path logo');
    assert(headerTemplate.includes('class="brand-name"'), 'nav markup missing brand-name');
    assert(headerTemplate.includes('class="brand-title"'), 'nav markup missing brand-title');
    assert(headerTemplate.includes('class="brand-divider"'), 'nav markup missing brand-divider');
    assert(headerTemplate.includes('class="brand-tagline"'), 'nav markup missing brand-tagline');
    assert(headerTemplate.includes('data-brand-tagline-primary="true"'), 'nav markup missing primary brand tagline hook');
    assert(!headerTemplate.includes('nav-item-audience'), 'header should not expose audience switcher');
    const entryHomeMatches = headerTemplate.match(/data-entry-home-link="true"/g) || [];
    assert(entryHomeMatches.length === 2, 'header should mark exactly two entry-home links');
    assert(!footerTemplate.includes('data-entry-home-link'), 'footer should not include entry-home markers');
    assert(!footerTemplate.includes('data-audience-crosslinks'), 'footer should not include audience cross-links');
    assert(headerTemplate.includes('class="nav-search"'), 'nav markup missing header search');
    assert(headerTemplate.includes('data-nav-search="collapsed"'), 'header search should expose collapsed state for compact expansion');
    assert(headerTemplate.includes('action="search"'), 'header search missing action="search"');
    assert(headerTemplate.includes('name="q"'), 'header search missing query param name="q"');
    assert(headerTemplate.includes('aria-controls="nav-search-q" aria-expanded="false"'), 'header search button should control the expandable input');
    assert(!headerTemplate.includes('role="button"'), 'header nav links should not be forced to role="button"');
    assert(headerTemplate.includes('class="nav-dropdown nav-dropdown-simple" id="nav-dropdown-resume"'), 'resume dropdown should use the shared simple dropdown shell');
    assert(headerTemplate.includes('class="nav-dropdown nav-dropdown-simple nav-dropdown-contact" id="nav-dropdown-contact"'), 'contact dropdown should use the shared simple dropdown shell');
    assert((headerTemplate.match(/nav-dropdown-inner nav-dropdown-inner-simple/g) || []).length === 2, 'resume and contact dropdowns should share the same inner structure');
  });

  section('Navigation CSS and mobile layout', () => {
    const navCss = fs.readFileSync('css/layout/nav.css', 'utf8');
    assert(navCss.includes('--brand-logo-size'), 'nav.css missing brand logo scale variable');
    assert(navCss.includes('.brand-divider'), 'nav.css missing brand-divider rules');
    assert(navCss.includes('.brand-tagline'), 'nav.css missing brand-tagline rules');
    assert(navCss.includes('flex-wrap:wrap;'), 'nav.css tagline should wrap whole chunks');
    assert(navCss.includes('grid-template-columns:repeat(4, minmax(0, 1fr)) auto;'), 'desktop nav should reserve only compact space for search by default');
    assert(navCss.includes('.nav-search.nav-search-is-enhanced.is-expanded') && navCss.includes('width:clamp(120px, 9vw, 132px);'), 'header search should expand to a compact desktop input width');
    assert(navCss.includes('.nav-search.nav-search-is-enhanced:not(.is-expanded) .nav-search-input') && navCss.includes('pointer-events:none;'), 'collapsed desktop search input should not intercept pointer events');
    assert(navCss.includes('grid-template-columns:30px 82px minmax(0, 1fr);'), 'portfolio dropdown project rows should use a compact rank/media/text grid');
    assert(navCss.includes('.nav-project-card:nth-child(1) .nav-project-rank::before{content:"01";}'), 'portfolio dropdown rank labels should render as quiet numeric labels');
    assert(navCss.includes('.nav-project-rank') && navCss.includes('background:transparent;'), 'portfolio dropdown rank labels should not use filled badges');
    assert(navCss.includes('.nav-dropdown-inner.nav-dropdown-inner-simple'), 'simple dropdowns should have a shared one-column inner shell');

    const utilCss = fs.readFileSync('css/utilities/layout.css', 'utf8');
    assert(utilCss.includes('--brand-title-size'), 'utilities/layout.css missing mobile brand sizing overrides');
    assert(utilCss.includes('.brand-divider'), 'utilities/layout.css missing mobile divider override');
    assert(/flex-direction\s*:\s*column;/.test(utilCss), 'brand mobile stack rule missing');
    assert(utilCss.includes('background:linear-gradient(90deg'), 'mobile divider gradient missing');
    assert(utilCss.includes('padding-top:var(--nav-height'), 'page offset for fixed header missing');
    assert(utilCss.includes('clip-path') && utilCss.includes('.nav-row.open'), 'mobile drawer clip-path reveal missing');
    assert(utilCss.includes('padding-inline:var(--mobile-page-gutter);'), 'mobile wrapper should use compact page gutters');
    assert(utilCss.includes('.nav-search.nav-search-is-enhanced.is-expanded') && utilCss.includes('width:100%;'), 'mobile drawer search should stay full width even with enhanced search classes');
    assert(utilCss.includes('body .surface-band,\n  body .interest-pad'), 'mobile global sections should share compact vertical spacing');
    assert(utilCss.includes('body:not([data-page="home"]):not(.home-pattern-page) .hero.hero--default'), 'mobile hero compacting should live in the utilities layer');
    assert(utilCss.includes('#cta #cta-link'), 'mobile CTA shell should be compacted');
    const helperCss = fs.readFileSync('css/utilities/helpers.css', 'utf8');
    assert(helperCss.includes('.contact-big .contact-card') && helperCss.includes('padding:22px 16px;'), 'mobile contact cards should override utility padding');
  });

  section('CSS tokens, layers, and components', () => {
    const varsCss = fs.readFileSync('css/variables.css', 'utf8');
    assert(/--secondary\s*:\s*var\(--primary\)\s*;/.test(varsCss), 'variables.css --secondary not mapped to --primary');
    assert(varsCss.includes('--brand-midnight:#091F3B;'), 'variables.css missing Logo Midnight token');
    assert(varsCss.includes('--brand-signal-blue:#005FED;'), 'variables.css missing Signal Blue token');
    assert(varsCss.includes('--brand-deep-blue:#0145C8;'), 'variables.css missing Deep Blue token');
    assert(varsCss.includes('--brand-canvas:#F9F9FA;'), 'variables.css missing Canvas token');
    assert(varsCss.includes('--brand-action-copper:#D97706;'), 'variables.css missing Action Copper token');
    assert(varsCss.includes('--font-sans:"Inter"'), 'variables.css should define Inter as the primary font');
    assert(varsCss.includes('--font-mono:"IBM Plex Mono"'), 'variables.css should define IBM Plex Mono for code/tool labels');
    assert(!varsCss.includes('#06B6D4'), 'variables.css should not keep the old cyan palette value');
    assert(varsCss.includes('--mobile-page-gutter:clamp(12px,4vw,18px);'), 'variables.css missing mobile page gutter token');
    assert(varsCss.includes('--mobile-section-y:clamp(1.55rem,5vw,2.25rem);'), 'variables.css missing compact mobile section token');

    const heroCss = fs.readFileSync('css/components/hero.css', 'utf8');
    assert(!heroCss.includes('var(--secondary)'), 'hero.css still references --secondary');
    assert(heroCss.includes('body:not([data-page="home"]):not(.home-pattern-page) .hero.hero--default'), 'hero.css should flatten non-home default heroes on mobile');
    const modalCss = fs.readFileSync('css/components/modal.css', 'utf8');
    assert(!modalCss.includes('var(--secondary)'), 'modal.css still references --secondary');
    assert(modalCss.includes('.contact-form input') && modalCss.includes('.contact-form-status'), 'contact modal form styles missing');
    assert(modalCss.includes('#contact-modal .modal-content') && modalCss.includes('width:calc(100vw - 12px);'), 'contact modal should use compact mobile viewport width');

    const projectCss = fs.readFileSync('css/components/project-page.css', 'utf8');
    const brandOverrideCss = fs.readFileSync('css/utilities/design-system-overrides.css', 'utf8');
    const toolsWorkspaceCss = fs.readFileSync('css/components/tools-workspace.css', 'utf8');
    const toolsAccountCss = fs.readFileSync('css/components/tools-account.css', 'utf8');
    const siteChatbotCss = fs.readFileSync('css/components/site-chatbot.css', 'utf8');
    const cookieSettingsCss = fs.readFileSync('css/components/cookie-settings.css', 'utf8');
    const privacyCss = fs.readFileSync('css/privacy.css', 'utf8');
    const toolThemeCss = fs.readFileSync('css/components/tool-theme.css', 'utf8');
    const resumeCss = fs.readFileSync('css/components/resume.css', 'utf8');
    const contactCardCss = fs.readFileSync('css/components/contact-card.css', 'utf8');
    const homeProofCss = fs.readFileSync('css/components/home-proof.css', 'utf8');
    const homeScrollCss = fs.readFileSync('css/components/home-scroll.css', 'utf8');
    const jumpPanelCss = fs.readFileSync('css/components/jump-panel.css', 'utf8');
    assert(projectCss.includes('--project-mobile-edge:calc(var(--mobile-page-gutter, 14px) * -1);'), 'project pages should flatten demo shells to mobile edges');
    assert(projectCss.includes('margin-inline:var(--project-mobile-edge);'), 'project demo shell should consume redundant mobile wrapper gutters');
    assert(projectCss.includes('.project-case-study') && projectCss.includes('.project-decision-flow'), 'project pages should include branded decision memo case-study styles');
    assert(homeProofCss.includes('grid-auto-rows: 1fr;') &&
      homeProofCss.includes('.home-proof-card') &&
      homeProofCss.includes('box-sizing: border-box;') &&
      homeProofCss.includes('width: 100%;'), 'home proof KPI cards should fill equal-height grid tracks without overlap');
    assert(homeScrollCss.includes('@supports (animation-timeline: view())') &&
      homeScrollCss.includes('animation:home-section-enter both ease-out;') &&
      homeScrollCss.includes('@media (min-width:769px) and (prefers-reduced-motion:no-preference)'),
      'audience home scroll polish should be progressive, subtle, and desktop-only');
    assert(!jumpPanelCss.includes('--jump-link-progress') &&
      !jumpPanelCss.includes('.jump-panel-link::before') &&
      jumpPanelCss.includes('color-mix(in srgb,var(--brand-midnight, #091f3b) 13%, transparent)'),
      'jump panel should avoid scroll-progress affordances and dark drop shadows');
    assert(brandOverrideCss.includes('--hero-art-layer: url("../img/brand/23-hero-general-light.png");'), 'brand overrides should define the general alternate light hero raster');
    assert(brandOverrideCss.includes('body[data-audience="analytics"]') &&
      brandOverrideCss.includes('--hero-art-layer: url("../img/brand/24-hero-analytics-light.png");'), 'brand overrides should keep the analytics audience light hero raster available');
    assert(brandOverrideCss.includes('body[data-page="analytics"]') &&
      brandOverrideCss.includes('--hero-art-layer: url("../img/brand/07-website-hero-light-version.png");') &&
      brandOverrideCss.includes('--hero-mobile-art-layer: url("../img/brand/27-hero-mobile-light.png");'), 'analytics homepage should use the main website hero raster on desktop and the portrait raster on mobile');
    assert(brandOverrideCss.includes('--hero-art-layer: url("../img/brand/25-hero-data-science-light.png");'), 'brand overrides should define the data science alternate light hero raster');
    assert(brandOverrideCss.includes('--hero-art-layer: url("../img/brand/26-hero-tourism-light.png");'), 'brand overrides should define the tourism alternate light hero raster');
    assert(brandOverrideCss.includes('--hero-mobile-art-layer: url("../img/brand/27-hero-mobile-light.png");'), 'brand overrides should define the portrait mobile hero raster');
    assert(brandOverrideCss.includes('var(--hero-art-layer, url("../img/brand/23-hero-general-light.png")) right bottom / auto 100% no-repeat'), 'desktop audience hero should anchor the selected light hero raster to the bottom-right and fill the hero height');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .hero.hero--default') &&
      brandOverrideCss.includes('box-sizing: border-box;') &&
      brandOverrideCss.includes('min-height: calc(100svh - var(--nav-height, 72px));'),
      'audience homepage heroes should fill the viewport without adding padding beyond it');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .chevron-hint') &&
      brandOverrideCss.includes('display: inline-flex;'),
      'audience homepage scroll indicators should remain visible across viewport sizes');
    assert(brandOverrideCss.includes('var(--hero-mobile-art-layer, var(--hero-art-layer, url("../img/brand/23-hero-general-light.png"))) center bottom / cover no-repeat'), 'mobile hero should use the portrait hero raster anchored to the bottom of the hero');
    assert(brandOverrideCss.includes('body[data-page="portfolio"] .hero.hero--default') &&
      brandOverrideCss.includes('var(--hero-art-layer, url("../img/brand/23-hero-general-light.png")) right bottom / auto 100% no-repeat'), 'desktop portfolio hero should anchor the selected light hero raster to the bottom-right and fill the hero height');
    [
      'analytics-project-examples-bg.png',
      'analytics-business-results-bg.png',
      'analytics-work-experience-bg.png',
      'analytics-skills-practice-bg.png',
      'analytics-certifications-bg.png',
      'analytics-contact-cta-bg.png'
    ].forEach((file) => {
      assert(brandOverrideCss.includes(`url("../img/brand/${file}")`), `analytics homepage section CSS should reference ${file}`);
    });
    [
      'img1.png',
      'img2.png',
      'img3.png',
      'img4.png',
      'img5.png',
      'img6.png',
      'home-project-examples-bg.png',
      'home-selected-outcomes-bg.png',
      'home-work-experience-bg.png',
      'home-about-me-bg.png',
      'home-certifications-bg.png',
      'home-analytics-cta-bg.png'
    ].forEach((file) => {
      assert(!brandOverrideCss.includes(`url("../img/brand/${file}")`), `analytics homepage section CSS should no longer reference ${file}`);
    });
    assert(brandOverrideCss.includes('background-image: var(--home-section-art);') &&
      brandOverrideCss.includes('background-size: cover;'), 'analytics homepage section art should render unwashed at natural section heights');
    assert(brandOverrideCss.includes('body[data-page="analytics"].home-pattern-page #cta #cta-link') &&
      brandOverrideCss.includes('width: min(100%, 560px);') &&
      brandOverrideCss.includes('background: #ffffff;') &&
      brandOverrideCss.includes('body[data-page="analytics"].home-pattern-page #cta #cta-link p') &&
      brandOverrideCss.includes('color: var(--brand-slate);'), 'analytics mobile CTA should keep readable foreground card on dark background art');
    assert(!brandOverrideCss.includes('--home-section-frame-height') &&
      !brandOverrideCss.includes('height: var(--home-section-frame-height);') &&
      !brandOverrideCss.includes('min-height: var(--home-section-frame-height);') &&
      !brandOverrideCss.includes('scroll-snap-type: x proximity;'), 'analytics homepage sections should not force equal-height frames');
    assert(!brandOverrideCss.includes('linear-gradient(90deg, rgba(255, 255, 255, .9), rgba(255, 255, 255, .68))') &&
      !brandOverrideCss.includes('linear-gradient(180deg, rgba(255, 255, 255, .92), rgba(255, 255, 255, .72))'), 'analytics homepage section art should not be lightened by white overlay gradients');
    assert(brandOverrideCss.includes('var(--brand-action-copper)'), 'brand overrides should use restrained Action Copper accents');
    assert(brandOverrideCss.includes('body[data-page="analytics"] #project-examples,\n    body[data-page="analytics"] #work-experience,\n    body[data-page="analytics"] #certifications') &&
      brandOverrideCss.includes('body[data-page="analytics"] #selected-outcomes,\n    body[data-page="analytics"] #about-me,\n    body[data-page="analytics"] #cta'), 'analytics homepage sections should have explicit alternating desktop backgrounds');
    assert(!brandOverrideCss.includes('color-mix(in srgb, var(--brand-navy) 94%, transparent)'), 'mobile homepage section bands should not use white-to-dark-blue gradients');
    assert(brandOverrideCss.includes('body .project-examples-band,\n    body .home-proof-band,\n    body .cert-band,\n    body .portfolio-library-section') &&
      brandOverrideCss.includes('background: var(--brand-mist);'), 'mobile homepage section bands should use a solid mist background');
    assert(brandOverrideCss.includes('--home-section-pad: clamp(3.2rem, 5vw, 4.4rem);'), 'analytics homepage desktop sections should share a consistent vertical padding token');
    assert(brandOverrideCss.includes('body[data-page="analytics"] .section-title::before,\n    body[data-page="analytics"] .project-examples-head h2::before') &&
      brandOverrideCss.includes('margin-inline: auto;'), 'analytics homepage section accents should align consistently');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .hero .cta-group .btn-ghost') &&
      brandOverrideCss.includes('background: #ffffff;') &&
      brandOverrideCss.includes('color: var(--brand-ink);'),
      'audience hero contact button should be opaque with black text');
    assert(brandOverrideCss.includes('body[data-page="analytics"] .cert img,\n    body[data-page="analytics"] .cert-card-logo') &&
      brandOverrideCss.includes('background: #ffffff;') &&
      brandOverrideCss.includes('img[src*="purdue_global"]') &&
      brandOverrideCss.includes('background: var(--brand-midnight);'), 'certification logos should sit on neutral readable tiles with a dark tile for light artwork');
    assert(brandOverrideCss.includes('.skill-subtitle') && fs.readFileSync('css/components/core.css', 'utf8').includes('#about-me .skill-subtitle'), 'skills section subtitle should have matching brand styling');
    assert(brandOverrideCss.includes('font-family: var(--font-mono)'), 'brand overrides should reserve mono typography for labels and metrics');
    assert(!/\\.nav-project-rank,\\s*\\n\\s*\\.nav-dropdown-badge/.test(brandOverrideCss), 'portfolio dropdown ranks should not inherit copper badge styling');
    assert(brandOverrideCss.includes('.nav-dropdown-inner-simple') && brandOverrideCss.includes('grid-template-columns: minmax(0, 1fr);'), 'dropdown overrides should normalize simple dropdown inner structure');
    assert(brandOverrideCss.includes('.nav-dropdown-list .nav-dropdown-link + .nav-dropdown-link') && brandOverrideCss.includes('margin-top: 0;'), 'dropdown list rows should use shared gap spacing instead of stacked margins');
    assert(brandOverrideCss.includes('#nav-dropdown-resume,\n    #nav-dropdown-contact') && brandOverrideCss.includes('width: min(360px, 64vw);'), 'resume and contact dropdowns should share matching desktop width');
    assert(brandOverrideCss.includes('#nav-dropdown-portfolio .nav-project-rank') && brandOverrideCss.includes('color: color-mix(in srgb, var(--brand-slate) 82%, var(--brand-midnight) 18%);'), 'portfolio dropdown ranks should use restrained brand-neutral colors');
    assert(brandOverrideCss.includes('#nav-dropdown-portfolio .nav-dropdown-footer-inline') && brandOverrideCss.includes('grid-template-columns: 1fr;'), 'portfolio dropdown footer should use a single full-width portfolio link');
    assert(brandOverrideCss.includes('#nav-dropdown-portfolio .nav-dropdown-footer-inline::before') && brandOverrideCss.includes('background: var(--brand-action-copper);'), 'portfolio dropdown footer should use a restrained copper divider accent');
    assert(brandOverrideCss.includes('#nav-dropdown-portfolio .nav-dropdown-all .nav-dropdown-subtitle') && brandOverrideCss.includes('display: block;'), 'portfolio dropdown footer link should show matching subtitle text');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .hero-avatar') &&
      brandOverrideCss.includes('width: clamp(96px, 9vw, 132px);') &&
      brandOverrideCss.includes('width: clamp(84px, 24vw, 112px);') &&
      brandOverrideCss.includes('opacity: 1;'),
      'brand overrides should reveal larger restrained audience hero headshots');
    assert(!/body:is\(\[data-page="data-science"\], \[data-page="tourism"\]\) \.hero-avatar\s*\{[^}]*display:\s*none/.test(brandOverrideCss), 'brand overrides should not hide data-science or tourism hero headshots');
    assert(brandOverrideCss.includes('@media (max-width: 430px)') &&
      brandOverrideCss.includes('width: min(100%, 310px);') &&
      brandOverrideCss.includes('width: min(100%, 300px);'),
      'audience mobile hero should keep headline and action clusters inside narrow phone viewports');
    assert(brandOverrideCss.includes('body[data-page="portfolio"] #projects.portfolio-library-grid'), 'brand overrides should contain mobile portfolio cards to the viewport');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .project-examples-card') && brandOverrideCss.includes('grid-template-rows: auto minmax(104px, auto);'), 'audience project cards should use the portfolio page media plus white text panel layout');
    assert(brandOverrideCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]) .project-examples-card .overlay') && brandOverrideCss.includes('display: none;'), 'audience project cards should not depend on dark image overlays for readability');
    assert(toolsWorkspaceCss.includes('--tools-shell-width:100%;'), 'tool workspaces should use full mobile shell width');
    assert(toolsAccountCss.includes('.tools-account-bar:empty'), 'empty tools account bars should not render as orphan containers');
    assert(siteChatbotCss.includes('right: calc(100% + 8px);'), 'chatbot launcher tooltip should sit inward with a button gap');
    assert(cookieSettingsCss.includes('left:calc(100% + 8px);'), 'cookie tooltip should sit inward with a button gap');
    assert(privacyCss.includes('--pcz-surface: #ffffff;'), 'cookie consent popup should use a light readable surface');
    assert(privacyCss.includes('--pcz-fg: var(--brand-midnight, #091f3b);'), 'cookie consent popup should use dark readable foreground text');
    assert(privacyCss.includes('--pcz-link: var(--brand-signal-blue, #005fed);'), 'cookie consent links should use brand blue on light surfaces');
    assert(privacyCss.includes('#pcz-modal .pref-toggle[aria-pressed="true"] .pref-state'), 'cookie consent active preference state should keep readable text on blue buttons');
    assert(toolThemeCss.includes('body[data-page="text-compare"]') && toolThemeCss.includes('padding:var(--mobile-card-pad);'), 'tool pages should compact mobile cards');
    assert(resumeCss.includes('.resume-paper') && resumeCss.includes('border-radius:12px;'), 'resume paper should use compact mobile card radius');
    assert(contactCardCss.includes('.contact-card') && contactCardCss.includes('padding:22px 16px;'), 'contact cards should use compact mobile padding');

    const stylesCss = fs.readFileSync('css/styles.css', 'utf8');
    assert(stylesCss.includes('@layer tokens, base, layout, components, utilities, overrides;'), 'styles.css layer order missing');
    assert(stylesCss.includes('@import url("components/home-scroll.css");'), 'styles.css should include shared audience scroll polish');
    ['css/base/base.css','css/components/buttons.css','css/layout/nav.css','css/utilities/design-system-overrides.css'].forEach((file) => {
      assert(!fs.readFileSync(file, 'utf8').includes('Poppins'), `${file} should not reference Poppins`);
    });
    assert(!fs.readFileSync('pages/portfolio.html', 'utf8').includes('Poppins'), 'portfolio page should not preload legacy Poppins fonts');
    ['components/home-proof.css','components/jump-panel.css','components/work-experience.css','components/contact-card.css','components/search.css','components/project-page.css'].forEach((snippet) => {
      assert(!stylesCss.includes(snippet), `styles.css should not eagerly import ${snippet}`);
    });

    const routeStyles = JSON.parse(fs.readFileSync('build/route-component-styles.json', 'utf8'));
    assert(Array.isArray(routeStyles['/']), 'route styles manifest missing home entry');
    assert(Array.isArray(routeStyles['/analytics']), 'route styles manifest missing analytics entry');
    assert(Array.isArray(routeStyles['/data-science']), 'route styles manifest missing data-science entry');
    assert(Array.isArray(routeStyles['/tourism']), 'route styles manifest missing tourism entry');
    ['css/components/home-proof.css','css/components/jump-panel.css','css/components/certification.css','css/components/work-experience.css','css/components/destination-analytics.css'].forEach((stylePath) => {
      assert(routeStyles['/analytics'].includes(stylePath), `/analytics route styles missing ${stylePath}`);
      assert(routeStyles['/data-science'].includes(stylePath), `/data-science route styles missing ${stylePath}`);
      assert(routeStyles['/tourism'].includes(stylePath), `/tourism route styles missing ${stylePath}`);
    });
    const certCss = fs.readFileSync('css/components/certification.css', 'utf8');
    assert(certCss.includes('background:#ffffff;') &&
      certCss.includes('height:76px;') &&
      certCss.includes('box-shadow:var(--shadow-sm);') &&
      certCss.includes('.cert img[src*="purdue_global"]') &&
      certCss.includes('background:var(--brand-midnight);') &&
      certCss.includes('object-fit:contain;') &&
      certCss.includes('box-sizing:content-box;') &&
      !certCss.includes('transition:border-color .2s ease, box-shadow .2s ease, transform'),
      'certification ticker logos should use stable neutral tiles without easing the JS-driven ticker transform');
    const animationsJs = fs.readFileSync('js/animations/animations.js', 'utf8');
    assert(animationsJs.includes('BASE=90') &&
      !animationsJs.includes("track.addEventListener('focusin'") &&
      !animationsJs.includes("track.addEventListener('focusout'"),
      'certification ticker should keep the original scroll speed and hover pause behavior');
    assert(brandOverrideCss.includes('body .cert-track') &&
      brandOverrideCss.includes('height: 64px;'),
      'mobile certification ticker should leave vertical room for readable logo tiles');
    assert(Array.isArray(routeStyles['/resume-analytics']), 'route styles manifest missing resume analytics entry');
    assert(Array.isArray(routeStyles['/search']), 'route styles manifest missing search entry');
    assert(Array.isArray(routeStyles['/portfolio/*']), 'route styles manifest missing portfolio wildcard entry');
  });

  section('Core scripts load without DOM', () => {
    [
      'js/common/common.js',
      'js/navigation/navigation.js',
      'js/animations/animations.js',
      'js/accounts/tools-page-loader.js',
      'js/forms/contact.js',
      'js/portfolio/modal-helpers.js',
      'js/contributions/contributions.js',
      'js/contributions/carousel.js'
    ].forEach(file => evalScript(file));

    const lateLoadEnv = createEnv();
    let handleRedirectCalls = 0;
    lateLoadEnv.document.body.dataset = { page: 'tools' };
    lateLoadEnv.document.title = 'Tools | Daniel Short';
    lateLoadEnv.window.ToolsAuth = {
      handleRedirect: async () => {
        handleRedirectCalls++;
        return { redirected: false };
      }
    };
    evalScript('js/accounts/tools-account-ui.js', lateLoadEnv);
    assert(
      handleRedirectCalls === 1,
      'tools-account-ui should self-initialize when loaded after DOMContentLoaded'
    );
  });

  section('CSS bundle manifest and page references', () => {
    const cssManifestPath = 'dist/styles-manifest.json';
    const scriptsManifestPath = 'dist/scripts-manifest.json';
    assert(fs.existsSync(cssManifestPath), 'dist/styles-manifest.json missing');
    assert(fs.existsSync(scriptsManifestPath), 'dist/scripts-manifest.json missing');
    const cssManifest = JSON.parse(fs.readFileSync(cssManifestPath, 'utf8'));
    const scriptsManifest = JSON.parse(fs.readFileSync(scriptsManifestPath, 'utf8'));
    assert(cssManifest.file && /^styles\.[0-9a-f]{8}\.css$/.test(cssManifest.file), 'CSS manifest entry invalid');
    assert(cssManifest.toolsFile && /^styles-tools\.[0-9a-f]{8}\.css$/.test(cssManifest.toolsFile), 'Tools CSS manifest entry invalid');
    ['shell','consent','home','contact','search','contributions','sitemap','privacy','toolsAccount','toolsLanding'].forEach((key) => {
      assert(typeof scriptsManifest[key] === 'string' && /^site-[a-z-]+\.[0-9a-f]{8}\.js$/.test(scriptsManifest[key]), `Scripts manifest entry invalid for ${key}`);
      assert(fs.existsSync(`dist/${scriptsManifest[key]}`), `dist/${scriptsManifest[key]} missing`);
    });
    hashedCss = cssManifest.file;
    const hashedToolsCss = cssManifest.toolsFile;
    assert(fs.existsSync(`dist/${hashedCss}`), `dist/${hashedCss} missing`);
    assert(fs.existsSync(`dist/${hashedToolsCss}`), `dist/${hashedToolsCss} missing`);
    assert(fs.existsSync('dist/styles.css'), 'dist/styles.css missing');
    assert(fs.existsSync('dist/styles-tools.css'), 'dist/styles-tools.css missing');
    assert(fs.existsSync('dist/site-shell.js'), 'dist/site-shell.js missing');
    assert(fs.existsSync('dist/site-tools-account.js'), 'dist/site-tools-account.js missing');

    const projectIds = evalScript('js/portfolio/projects-data.js').window.PROJECTS
      .filter(p => p && p.published !== false)
      .map(p => p.id);
    const projectPages = projectIds.map(id => `pages/portfolio/${id}.html`);
    const toolPages = ['pages/tools.html','pages/tools-dashboard.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html','pages/whisper-transcribe-monitor.html','pages/ga4-utm-performance.html'];
    ['pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/portfolio.html','pages/contributions.html','pages/contact.html','pages/resume.html','pages/resume-pdf.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html','pages/privacy.html','pages/search.html','404.html', ...toolPages, ...projectPages].forEach(f => {
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
      assert(
        htmlHasManagedBundle(html, 'site-shell') || html.includes('js/common/common.js'),
        `${f} missing shared JS bundle reference`
      );
      assert(
        htmlHasManagedBundle(html, 'site-consent') || html.includes('js/privacy/config.js'),
        `${f} missing consent JS bundle reference`
      );
    });
    toolPages.forEach((f) => {
      const html = fs.readFileSync(f, 'utf8');
      assert(
        html.includes(`dist/${hashedToolsCss}`) || html.includes('dist/styles-tools.css'),
        `${f} missing tools stylesheet reference`
      );
    });

    assert(htmlHasManagedBundle(readFile('pages/contact.html'), 'site-contact'), 'pages/contact.html missing contact bundle reference');
    assert(htmlHasManagedBundle(readFile('pages/search.html'), 'site-search'), 'pages/search.html missing search bundle reference');
    assert(htmlHasManagedBundle(readFile('pages/contributions.html'), 'site-contributions'), 'pages/contributions.html missing contributions bundle reference');
    assert(htmlHasManagedBundle(readFile('pages/sitemap-pretty.html'), 'site-sitemap'), 'pages/sitemap-pretty.html missing sitemap bundle reference');
    assert(htmlHasManagedBundle(readFile('pages/privacy.html'), 'site-privacy'), 'pages/privacy.html missing privacy bundle reference');

    const toolsIndexHtml = readFile('pages/tools.html');
    assert(htmlHasManagedBundle(toolsIndexHtml, 'site-tools-landing'), 'pages/tools.html missing tools landing loader bundle');
    assert(/data-tools-account-src="dist\/site-tools-account(?:\.[0-9a-f]{8})?\.js"/.test(toolsIndexHtml), 'pages/tools.html missing managed tools-account source');
    ['pages/word-frequency.html','pages/text-compare.html','pages/job-application-tracker.html','pages/tools-dashboard.html'].forEach((file) => {
      assert(htmlHasManagedBundle(readFile(file), 'site-tools-account'), `${file} missing tools-account bundle`);
    });

    const toolPageEntryScripts = {
      'pages/background-remover.html': ['js/tools/background-remover.js'],
      'pages/ga4-utm-performance.html': ['js/tools/ga4-utm-performance.js'],
      'pages/image-optimizer.html': ['js/tools/image-optimizer.js'],
      'pages/job-application-tracker.html': ['js/tools/job-application-tracker.js'],
      'pages/nbsp-cleaner.html': ['js/tools/nbsp-cleaner.js'],
      'pages/ocean-wave-simulation.html': ['js/tools/ocean-wave-simulation.js'],
      'pages/oxford-comma-checker.html': ['js/tools/oxford-comma-checker.js'],
      'pages/point-of-view-checker.html': ['js/tools/point-of-view-checker.js'],
      'pages/qr-code-generator.html': ['js/tools/qr-code-generator-utils.js', 'js/tools/qr-code-generator.js'],
      'pages/screen-recorder.html': ['js/tools/screen-recorder.js'],
      'pages/text-compare.html': ['js/tools/text-compare-core.js', 'js/tools/text-compare.js'],
      'pages/whisper-transcribe-monitor.html': ['js/tools/whisper-transcribe-monitor.js'],
      'pages/word-frequency.html': ['js/tools/word-frequency.js']
    };
    Object.entries(toolPageEntryScripts).forEach(([file, scripts]) => {
      scripts.forEach((scriptPath) => {
        checkFileContains(file, scriptPath);
      });
    });

    const textCompareHtml = readFile('pages/text-compare.html');
    assert(textCompareHtml.includes('id="textcompare-mode-auto"'), 'pages/text-compare.html missing auto mode control');
    assert(textCompareHtml.includes('id="textcompare-mode-document"'), 'pages/text-compare.html missing document mode control');
    assert(textCompareHtml.includes('id="textcompare-mode-structured"'), 'pages/text-compare.html missing structured mode control');
    assert(textCompareHtml.includes('id="textcompare-warning"'), 'pages/text-compare.html missing compare warning status');
    assert(fs.existsSync('js/tools/text-compare-worker.js'), 'js/tools/text-compare-worker.js missing');

    checkFileContains('index.html', 'http-equiv="refresh"');
    checkFileContains('pages/search.html', 'css/components/search.css');
    checkFileContains('pages/contact.html', 'css/components/contact-card.css');
    checkFileContains('pages/resume.html', 'css/components/resume.css');
    checkFileContains('pages/analytics.html', 'css/components/destination-analytics.css');
    checkFileContains('pages/data-science.html', 'css/components/destination-analytics.css');
    ['pages/analytics.html', 'pages/data-science.html', 'pages/tourism.html'].forEach((file) => {
      checkFileContains(file, 'css/components/home-proof.css');
      checkFileContains(file, 'css/components/jump-panel.css');
      checkFileContains(file, 'css/components/certification.css');
      checkFileContains(file, 'css/components/work-experience.css');
    });
    checkFileContains('pages/resume-analytics.html', 'css/components/resume.css');
    checkFileContains('pages/portfolio/nonogram.html', 'css/components/project-page.css');
  });

  section('Audience landing page structure', () => {
    ['pages/analytics.html', 'pages/data-science.html', 'pages/tourism.html'].forEach((file) => {
      const html = readFile(file);
      checkFileContains(file, 'home-pattern-page');
      checkFileContains(file, 'id="selected-outcomes"');
      checkFileContains(file, 'class="jump-panel"');
      checkFileContains(file, 'href="#selected-outcomes"');
      checkFileContains(file, 'id="project-examples"');
      checkFileContains(file, 'id="work-experience"');
      checkFileContains(file, 'id="about-me"');
      checkFileContains(file, 'class="skill-subtitle"');
      checkFileContains(file, 'Examples of how the technical toolkit shows up in real projects');
      checkFileContains(file, 'id="certifications"');
      checkFileContains(file, 'id="cta"');
      checkFileContains(file, 'id="contact-modal"');
      checkFileContains(file, 'data-cert-modal-open');
      checkFileContains(file, 'class="chevron-hint scroll-indicator" href="#selected-outcomes"');
      checkFileContains(file, 'Scroll for results');
      assert(htmlHasManagedBundle(html, 'site-home'), `${file} missing home bundle for certifications modal`);
      assert(!html.includes('audience-gateway-hero'), `${file} should not use audience gateway hero`);
      assert(!html.includes('class="hero-proof-row"'), `${file} hero should not include the old metric strip`);
      assert(!html.includes('this version'), `${file} should not mention "this version"`);
    });
    assert(!readFile('pages/data-science.html').includes('hero-bullet-list'), 'data-science page should not use hero bullets');
    assert(!readFile('pages/tourism.html').includes('hero-bullet-list'), 'tourism page should not use hero bullets');
    checkFileContains('pages/data-science.html', 'Applied Modeling With Measured Impact');
    checkFileContains('pages/data-science.html', 'home-proof-value">95%<');
    checkFileContains('pages/data-science.html', 'home-proof-value">10x<');
    checkFileContains('pages/data-science.html', 'home-proof-value">98%<');
    checkFileContains('pages/data-science.html', 'home-proof-value">+14.13%<');
    const analyticsHtml = readFile('pages/analytics.html');
    const projectExamplesIndex = analyticsHtml.indexOf('id="project-examples"');
    const selectedOutcomesIndex = analyticsHtml.indexOf('id="selected-outcomes"');
    assert(projectExamplesIndex >= 0 && selectedOutcomesIndex >= 0 && projectExamplesIndex < selectedOutcomesIndex,
      'analytics page should show project examples before the results snapshot');
    checkFileContains('pages/analytics.html', '<h2 id="project-examples-title">Project Examples</h2>');
    checkFileContains('pages/analytics.html', '<h2 id="selected-outcomes-title" class="section-title">Business-Facing Results</h2>');
    checkFileContains('pages/analytics.html', 'data-brand-tagline-primary="true">Data Analytics<');
    ['pages/analytics.html', 'pages/data-science.html', 'pages/tourism.html'].forEach((file) => {
      checkFileContains(file, 'class="hero-identity"');
      checkFileContains(file, 'class="hero-avatar"');
      checkFileContains(file, 'img/hero/head-avatar-192.jpg');
      checkFileContains(file, 'sizes="(max-width: 768px) 112px, 132px"');
      checkFileContains(file, 'contact#contact-modal');
      checkFileContains(file, 'Daniel Short ·');
      const html = readFile(file);
      const heroHeadlineIndex = html.indexOf('<h1>');
      const heroIdentityIndex = html.indexOf('class="hero-identity"');
      assert(heroHeadlineIndex >= 0 && heroIdentityIndex > heroHeadlineIndex,
        `${file} hero identity should sit below the headline`);
    });
    checkFileContains('pages/data-science.html', 'data-brand-tagline-primary="true">Data Science<');
    checkFileContains('pages/tourism.html', 'data-brand-tagline-primary="true">Tourism Analytics<');
    checkFileContains('pages/resume-analytics.html', 'data-brand-tagline-primary="true">Data Analytics<');
    checkFileContains('pages/resume-data-science.html', 'data-brand-tagline-primary="true">Data Science<');
    checkFileContains('pages/resume-tourism.html', 'data-brand-tagline-primary="true">Tourism Analytics<');
    checkFileContains('pages/data-science.html', 'id="transferability"');
    checkFileContains('pages/tourism.html', 'id="transferability"');
    checkFileContains('pages/data-science.html', 'href="#transferability"');
    checkFileContains('pages/tourism.html', 'href="#transferability"');
    const homeAccentCss = readFile('css/utilities/design-system-overrides.css');
    ['--home-section-accent', '--home-section-accent-soft', '--home-section-accent-border', '--home-section-accent-strong'].forEach((token) => {
      assert(homeAccentCss.includes(token), `audience home accent CSS missing ${token}`);
    });
    ['#project-examples', '#selected-outcomes', '#transferability', '#work-experience', '#about-me', '#certifications', '#cta'].forEach((selector) => {
      assert(homeAccentCss.includes(selector), `audience home accent CSS missing ${selector}`);
    });
    ['.project-examples-card .project-text::before', '.home-proof-card::after', '.work-card::after', '.icon-info::before', '.cert-card-label', '.transfer-map-head .transfer-map-cell', '#cta #cta-link'].forEach((selector) => {
      assert(homeAccentCss.includes(selector), `audience home visible accent CSS missing ${selector}`);
    });
    ['var(--brand-signal-blue)', 'var(--brand-deep-blue)', 'var(--brand-midnight)', 'var(--brand-slate)', 'var(--brand-action-copper)'].forEach((color) => {
      assert(homeAccentCss.includes(color), `audience home accent CSS missing brand color ${color}`);
    });
    ['#2f9d58', '#4f73d9', '#0f766e', '#5a78a8', '#5a7edb'].forEach((color) => {
      assert(!homeAccentCss.includes(color), `audience home accent CSS should not use off-brand color ${color}`);
    });
    assert(homeAccentCss.includes('section:not(.hero) *:is(.section-title'),
      'audience home title accents should use minifier-safe descendant selectors');
    assert(homeAccentCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]).home-pattern-page'),
      'audience home accents should cover analytics, data science, and tourism homes');
    ['pages/analytics.html', 'pages/data-science.html', 'pages/tourism.html'].forEach((file) => {
      const html = readFile(file);
      const targetIndex = html.indexOf('<h3 class="work-company">Target</h3>');
      const randallIndex = html.indexOf('<h3 class="work-company">Randall Reilly</h3>');
      const visitIndex = html.indexOf('<h3 class="work-company">Visit Grand Junction</h3>');
      assert(targetIndex >= 0, `${file} missing Target work card`);
      assert(randallIndex >= 0, `${file} missing Randall Reilly work card`);
      assert(visitIndex >= 0, `${file} missing Visit Grand Junction work card`);
      assert(targetIndex < randallIndex && randallIndex < visitIndex,
        `${file} work cards should be in ascending chronological order`);
    });
    assert(!readFile('pages/analytics.html').includes('destinationReporting'), 'analytics page should not feature destinationReporting');
  });

  section('Project-first public copy', () => {
    const publicHtmlFiles = [
      'index.html',
      '404.html',
      'contact.html',
      'privacy.html',
      'resume.html',
      'resume-pdf.html',
      ...fs.readdirSync('pages')
        .filter((name) => name.endsWith('.html'))
        .map((name) => `pages/${name}`)
    ];
    const forbiddenPhrases = [
      'Featured Analytics Proof',
      'Featured Analytics Case Studies',
      'Proof first',
      'Decision-Ready Analytics Proof',
      'business-facing analytics proof',
      'Business proof,'
    ];
    publicHtmlFiles.forEach((file) => {
      const html = readFile(file);
      forbiddenPhrases.forEach((phrase) => {
        assert(!html.includes(phrase), `${file} should not include proof-framed public copy: ${phrase}`);
      });
    });
    const headerHtml = readFile('build/templates/header.partial.html');
    const portfolioStart = headerHtml.indexOf('id="nav-dropdown-portfolio"');
    const portfolioEnd = headerHtml.indexOf('<div class="nav-item nav-item-resume"');
    const portfolioDropdown = portfolioStart >= 0 && portfolioEnd > portfolioStart
      ? headerHtml.slice(portfolioStart, portfolioEnd)
      : '';
    assert(portfolioDropdown.includes('Featured Projects'), 'header portfolio dropdown should use project-first wording');
    assert(!portfolioDropdown.includes('Technical depth'), 'portfolio dropdown should not include the technical depth shortcut');
    assert(!portfolioDropdown.includes('Tools'), 'portfolio dropdown should not include tools shortcuts');
    assert((portfolioDropdown.match(/class="nav-project-card"/g) || []).length === 5, 'portfolio dropdown should keep five featured project links');
    assert((portfolioDropdown.match(/class="nav-dropdown-link nav-dropdown-all"/g) || []).length === 1, 'portfolio dropdown should have one footer portfolio link');
    assert(portfolioDropdown.includes('View full portfolio'), 'portfolio dropdown should link to the full portfolio');
    assert(portfolioDropdown.includes('Browse the complete project library'), 'portfolio dropdown footer should include descriptive subtext');
  });

  section('Internal site links stay in the same tab', () => {
    const projectIds = evalScript('js/portfolio/projects-data.js').window.PROJECTS
      .filter(p => p && p.published !== false)
      .map(p => p.id);
    const htmlFiles = [
      'index.html',
      ...fs.readdirSync('pages').filter((name) => name.endsWith('.html')).map((name) => `pages/${name}`),
      ...projectIds.map((id) => `pages/portfolio/${id}.html`)
    ];

    const targetRe = /<a\b[^>]*\bhref="([^"]+)"[^>]*\btarget="_blank"[^>]*>/gi;
    htmlFiles.forEach((file) => {
      const html = readFile(file);
      let match;
      while ((match = targetRe.exec(html))) {
        const href = String(match[1] || '').trim();
        if (!href) continue;
        if (/^(mailto:|tel:|https?:\/\/(?!www\.danielshort\.me))/i.test(href)) continue;
        if (/^(?:https:\/\/www\.danielshort\.me\/|\/)?documents\//i.test(href)) continue;
        if (/^(?:https:\/\/www\.danielshort\.me\/|\/)?(?:demos\/|[^/]*-demo(?:\.html)?)(?:$|[?#])/i.test(href)) continue;
        if (/^(?:https:\/\/www\.danielshort\.me\/|\/)?games\/(?:stellar-dogfight|roulette|probability-engine)(?:\.html)?(?:$|[?#])/i.test(href)) continue;
        assert(false, `${file} internal site link should not open in a new tab: ${href}`);
      }
    });
  });

  section('Fonts and navigation behavior', () => {
    const navCode = fs.readFileSync('js/navigation/navigation.js', 'utf8');
    const headerTemplate = fs.readFileSync('build/templates/header.partial.html', 'utf8');
    assert(navCode.includes('ENTRY_HOME_KEY'), 'navigation missing entry-home storage key');
    assert(navCode.includes('[data-entry-home-link="true"]'), 'navigation missing entry-home link selector');
    assert(navCode.includes('detectAudienceFromPath'), 'navigation missing audience path detection');
    assert(navCode.includes('[data-portfolio-home-link="true"]'), 'navigation missing portfolio home selector');
    assert(navCode.includes('[data-resume-home-link="true"]'), 'navigation missing resume home selector');
    assert(navCode.includes('[data-brand-tagline-primary="true"]'), 'navigation missing primary brand tagline selector');
    assert(navCode.includes('setupHeaderSearch(host)'), 'navigation should initialize expandable header search');
    assert(navCode.includes("form.classList.toggle('nav-search-is-enhanced'") && navCode.includes("form.classList.toggle('is-expanded'"), 'navigation should toggle compact and expanded search states');
    assert(navCode.includes("form.dataset.navSearch = enhanced ? (nextExpanded ? 'expanded' : 'collapsed') : 'full';"), 'navigation should distinguish full mobile search from collapsed desktop search');
    assert(navCode.includes("button.setAttribute('aria-label', enhanced && !nextExpanded ? 'Open search' : 'Search site');"), 'navigation should announce the visible mobile search button correctly');
    assert(navCode.includes('input.tabIndex = enhanced && !nextExpanded ? -1 : 0;'), 'collapsed desktop search input should leave the tab order');
    assert(headerTemplate.includes('aria-controls="primary-menu"'), 'header missing aria-controls="primary-menu"');
    assert(headerTemplate.includes('id="primary-menu"'), 'header missing primary-menu');
    assert(navCode.includes("classList.toggle('menu-open'"), 'burger toggle missing body.menu-open');
    assert(navCode.includes('aria-expanded'), 'burger missing aria-expanded');
    assert(navCode.includes('aria-current'), 'active nav link missing aria-current');
    assert(navCode.includes('getBoundingClientRect'), 'setNavHeight missing measurement');
  });

  section('Brand assets and case-study rendering', () => {
    [
      'img/brand/00-ds-logo-master-full-color.svg',
      'img/brand/04-ds-logo-reversed-white-blue.svg',
      'img/brand/05-ds-favicon-small-icon.svg',
      'img/brand/07-website-hero-light-version.png',
      'img/brand/07-website-hero-light-version.svg',
      'img/brand/23-hero-general-light.png',
      'img/brand/24-hero-analytics-light.png',
      'img/brand/25-hero-data-science-light.png',
      'img/brand/26-hero-tourism-light.png',
      'img/brand/27-hero-mobile-light.png',
      'img/brand/analytics-project-examples-bg.png',
      'img/brand/analytics-business-results-bg.png',
      'img/brand/analytics-work-experience-bg.png',
      'img/brand/analytics-skills-practice-bg.png',
      'img/brand/analytics-certifications-bg.png',
      'img/brand/analytics-contact-cta-bg.png',
      'img/brand/13-case-study-diagram-question-to-decision.svg',
      'img/brand/20-data-visualization-style-sample.svg'
    ].forEach((file) => assert(fs.existsSync(file), `${file} missing`));
    [
      'img/brand/img1.png',
      'img/brand/img2.png',
      'img/brand/img3.png',
      'img/brand/img4.png',
      'img/brand/img5.png',
      'img/brand/img6.png',
      'img/brand/home-project-examples-bg.png',
      'img/brand/home-selected-outcomes-bg.png',
      'img/brand/home-work-experience-bg.png',
      'img/brand/home-about-me-bg.png',
      'img/brand/home-certifications-bg.png',
      'img/brand/home-analytics-cta-bg.png'
    ].forEach((file) => assert(!fs.existsSync(file), `${file} should be removed after section background rename`));

    const generator = fs.readFileSync('build/generate-project-pages.js', 'utf8');
    assert(generator.includes('project-case-study'), 'project generator should render case-study content');
    assert(generator.includes('Question to Decision'), 'project generator should label the decision memo section');
    assert(generator.includes('13-case-study-diagram-question-to-decision.svg'), 'project generator should use approved case-study diagram asset');
    assert(generator.includes('img/brand/05-ds-favicon-small-icon.svg'), 'project generator should include approved SVG favicon');

    const portfolioJs = fs.readFileSync('js/portfolio/portfolio.js', 'utf8');
    assert(!portfolioJs.includes('project-card-proof'), 'portfolio cards should not render proof/result lines');
    assert(!portfolioJs.includes('card.target = "_blank"'), 'portfolio cards should keep internal project links in the same tab');

    [
      'css/components/projects.css',
      'css/utilities/design-system-overrides.css',
      'pages/portfolio.html',
      'pages/analytics.html',
      'pages/data-science.html',
      'pages/tourism.html'
    ].forEach((file) => {
      assert(!fs.readFileSync(file, 'utf8').includes('project-card-proof'), `${file} should not include project-card-proof`);
    });
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
    assert(copyJs.includes('shouldSkipPublicCopy') &&
           copyJs.includes("rel === 'img/slot'") &&
           copyJs.includes("rel === 'slot-config'") &&
           copyJs.includes("rel === 'demos/slot-machine-demo.html'"),
           'copy-to-public.js should keep slot assets local-only');

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
    const vercelIgnore = fs.readFileSync('.vercelignore', 'utf8');
    ['/api/cms/', '/api/chatbot/logs.js', '/api/short-domain.js', '/api/short-links/test/'].forEach((entry) => {
      assert(vercelIgnore.includes(entry), `.vercelignore should exclude ${entry} from Hobby deployments`);
    });
    let vercelObj;
    try { vercelObj = JSON.parse(vercel); } catch {}
    assert(vercelObj && vercelObj.env && vercelObj.env.CHATBOT_ENABLED === 'true' &&
           vercelObj.env.CHATBOT_REQUIRE_DDB === 'true' &&
           vercelObj.env.CHATBOT_BEDROCK_MODEL_ID === 'us.amazon.nova-lite-v1:0' &&
           vercelObj.env.CHATBOT_EMBEDDINGS_ENABLED === 'true' &&
           vercelObj.env.CHATBOT_BEDROCK_EMBED_MODEL_ID === 'amazon.titan-embed-text-v2:0' &&
           vercelObj.env.CHATBOT_BEDROCK_EMBED_DIMENSIONS === '512',
      'vercel.json should enable the chatbot with durable DynamoDB protection, Nova Lite, and Bedrock embeddings');
    const rewrites = (vercelObj && vercelObj.rewrites) || [];
    const redirects = (vercelObj && vercelObj.redirects) || [];
    assert(rewrites.length > 0, 'vercel.json missing rewrites');
    assert(rewrites.some(r => r.source === '/api/chatbot/logs' && r.destination === '/api/chatbot?__route=logs'),
      'chatbot logs should route through the main chatbot function to stay under the Hobby function limit');
    const badDest = rewrites.filter(r => /\.html$/.test((r.destination||'')));
    assert(badDest.length === 0, 'rewrite destinations must be extensionless to avoid loops');
    const hasPortfolio = rewrites.some(r => r.source === '/portfolio' && r.destination === '/pages/portfolio');
    const hasPortfolioHtml = rewrites.some(r => r.source === '/portfolio.html' && r.destination === '/pages/portfolio');
    const hasProjectRewrite = rewrites.some(r => r.source === '/portfolio/:project' && r.destination === '/pages/portfolio/:project');
    const hasRootRedirect = redirects.some(r => r.source === '/' && r.destination === '/analytics');
    const hasIndexRedirect = redirects.some(r => r.source === '/index.html' && r.destination === '/analytics');
    const hasDshortRootRedirect = redirects.some(r =>
      r.source === '/' &&
      Array.isArray(r.has) &&
      r.has.some(entry => entry && entry.type === 'host' && entry.value === 'dshort.me') &&
      r.destination === 'https://www.danielshort.me/analytics'
    );
    const hasAnalytics = rewrites.some(r => r.source === '/analytics' && r.destination === '/pages/analytics');
    const hasAnalyticsHtml = rewrites.some(r => r.source === '/analytics.html' && r.destination === '/pages/analytics');
    const hasDataScience = rewrites.some(r => r.source === '/data-science' && r.destination === '/pages/data-science');
    const hasDataScienceHtml = rewrites.some(r => r.source === '/data-science.html' && r.destination === '/pages/data-science');
    const hasTourism = rewrites.some(r => r.source === '/tourism' && r.destination === '/pages/tourism');
    const hasTourismHtml = rewrites.some(r => r.source === '/tourism.html' && r.destination === '/pages/tourism');
    const hasResumeAnalytics = rewrites.some(r => r.source === '/resume-analytics' && r.destination === '/pages/resume-analytics');
    const hasResumeDataScience = rewrites.some(r => r.source === '/resume-data-science' && r.destination === '/pages/resume-data-science');
    const hasResumeTourism = rewrites.some(r => r.source === '/resume-tourism' && r.destination === '/pages/resume-tourism');
    const hasResumeAnalyticsPdf = rewrites.some(r => r.source === '/resume-analytics-pdf' && r.destination === '/pages/resume-analytics-pdf');
    assert(hasPortfolio && hasPortfolioHtml, 'portfolio rewrites missing');
    assert(hasProjectRewrite, 'project rewrite missing (/portfolio/:project)');
    assert(hasRootRedirect, 'root redirect to /analytics missing');
    assert(hasIndexRedirect, '/index.html redirect to /analytics missing');
    assert(hasDshortRootRedirect, 'dshort.me root redirect to analytics missing');
    assert(hasAnalytics && hasAnalyticsHtml, 'analytics rewrites missing');
    assert(hasDataScience && hasDataScienceHtml, 'data-science rewrites missing');
    assert(hasTourism && hasTourismHtml, 'tourism rewrites missing');
    assert(hasResumeAnalytics && hasResumeDataScience && hasResumeTourism, 'resume rewrites missing');
    assert(hasResumeAnalyticsPdf, 'resume analytics PDF rewrite missing');
    const hasGames = rewrites.some(r => r.source === '/games' && r.destination === '/pages/games');
    const hasGameSlot = rewrites.some(r => /slot-machine/i.test(`${r.source || ''} ${r.destination || ''}`));
    const hasGameDogfight = rewrites.some(r => r.source === '/games/stellar-dogfight' && r.destination === '/demos/stellar-dogfight-demo');
    const hasGameDogfightHtml = rewrites.some(r => r.source === '/games/stellar-dogfight.html' && r.destination === '/demos/stellar-dogfight-demo');
    const hasGameRoulette = rewrites.some(r => r.source === '/games/roulette' && r.destination === '/demos/roulette-double-zero-demo');
    const hasGameRouletteHtml = rewrites.some(r => r.source === '/games/roulette.html' && r.destination === '/demos/roulette-double-zero-demo');
    assert(hasGames, 'games landing rewrite missing');
    assert(!hasGameSlot, 'slot machine should not be publicly rewritten');
    assert(!readFile('pages/games.html').includes('games/slot-machine'), 'games page should not link to local-only slot machine');
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
    const hasNoindexResumeAnalyticsPdf = headers.some(h =>
      h && (h.source === '/resume-analytics-pdf' || h.source === '/resume-analytics-pdf.html') &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexResumeDataSciencePdf = headers.some(h =>
      h && (h.source === '/resume-data-science-pdf' || h.source === '/resume-data-science-pdf.html') &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexResumeTourismPdf = headers.some(h =>
      h && (h.source === '/resume-tourism-pdf' || h.source === '/resume-tourism-pdf.html') &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    const hasNoindexDestinationAnalytics = headers.some(h =>
      h && (h.source === '/destination-analytics' || h.source === '/destination-analytics.html') &&
      Array.isArray(h.headers) &&
      h.headers.some(x => x && x.key === 'X-Robots-Tag' && /noindex/i.test(String(x.value || '')))
    );
    assert(hasNoindexShortLinks, 'short-links noindex header missing');
    assert(hasNoindexToolsDashboard, 'tools dashboard noindex header missing');
    assert(hasNoindexGa4Tool, 'GA4 tool noindex header missing');
    assert(hasNoindexWhisperTool, 'Whisper tool noindex header missing');
    assert(hasNoindexResumeAnalyticsPdf, 'resume analytics PDF noindex header missing');
    assert(hasNoindexResumeDataSciencePdf, 'resume data science PDF noindex header missing');
    assert(hasNoindexResumeTourismPdf, 'resume tourism PDF noindex header missing');
    assert(hasNoindexDestinationAnalytics, 'destination analytics noindex header missing');
  });

  section('Search index', () => {
    assert(fs.existsSync('dist/search-index.json'), 'dist/search-index.json missing');
    const raw = fs.readFileSync('dist/search-index.json', 'utf8');
    let parsed;
    try { parsed = JSON.parse(raw); } catch {}
    assert(parsed && Array.isArray(parsed.pages), 'search index should contain pages array');
    assert(parsed.pages.length >= 10, 'search index has too few entries');
    const urls = new Set(parsed.pages.map((entry) => String(entry && entry.url || '').trim()));
    assert(urls.has('/analytics'), 'search index should include analytics page');
    assert(urls.has('/data-science'), 'search index should include data-science page');
    assert(urls.has('/tourism'), 'search index should include tourism page');
    assert(!urls.has('/'), 'search index should exclude redirecting root URL');
    assert(!urls.has('/resume-analytics-pdf'), 'search index should exclude analytics PDF preview');
    assert(!urls.has('/resume-data-science-pdf'), 'search index should exclude data science PDF preview');
    assert(!urls.has('/resume-tourism-pdf'), 'search index should exclude tourism PDF preview');
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

  section('AWS demo health and warmup contracts', () => {
    const awsClient = fs.readFileSync('js/demos/aws-client.js', 'utf8');
    const digitDemo = fs.readFileSync('demos/digit-generator-demo.html', 'utf8');
    const handwritingDemo = fs.readFileSync('demos/handwriting-rating-demo.html', 'utf8');
    const shapeDemo = fs.readFileSync('demos/shape-demo.html', 'utf8');
    const nonogramDemo = fs.readFileSync('demos/nonogram-demo.html', 'utf8');
    const minesweeperDemo = fs.readFileSync('demos/minesweeper-demo.html', 'utf8');
    const pizzaDemo = fs.readFileSync('demos/pizza-tips-demo.html', 'utf8');
    const whisperMonitor = fs.readFileSync('js/tools/whisper-transcribe-monitor.js', 'utf8');
    const digitLambda = fs.readFileSync('aws/digit-generator/app.py', 'utf8');
    const handwritingLambda = fs.readFileSync('aws/handwriting-rating/app.py', 'utf8');
    const nonogramLambda = fs.readFileSync('aws/nonogram-solver/app.py', 'utf8');
    const nonogramEntrypoint = fs.readFileSync('aws/nonogram-solver/entrypoint.py', 'utf8');
    const nonogramDockerfile = fs.readFileSync('aws/nonogram-solver/Dockerfile', 'utf8');
    const minesweeperLambda = fs.readFileSync('aws/minesweeper-solver/app.py', 'utf8');
    const minesweeperEntrypoint = fs.readFileSync('aws/minesweeper-solver/entrypoint.py', 'utf8');
    const minesweeperDockerfile = fs.readFileSync('aws/minesweeper-solver/Dockerfile', 'utf8');
    const pizzaLambda = fs.readFileSync('aws/pizza-tips-predict/index.js', 'utf8');
    const whisperLambda = fs.readFileSync('aws/whisper-transcribe/app.py', 'utf8');
    const whisperEntrypoint = fs.readFileSync('aws/whisper-transcribe/entrypoint.py', 'utf8');
    const whisperDockerfile = fs.readFileSync('aws/whisper-transcribe/Dockerfile', 'utf8');

    assert(awsClient.includes('const healthJson = (base, options = {})') &&
           awsClient.includes("joinUrl(normalizeBase(base), 'health')") &&
           awsClient.includes('const warmupJson = (base, payload = {}, options = {})') &&
           awsClient.includes("joinUrl(normalizeBase(base), 'warmup')"),
      'shared AWS demo client should expose /health and /warmup helpers');

    const digitWarmupStart = digitDemo.indexOf('async function warmUpServer()');
    const digitWarmupEnd = digitDemo.indexOf('async function fetchGrid()', digitWarmupStart);
    const digitWarmup = digitDemo.slice(digitWarmupStart, digitWarmupEnd);
    assert(digitDemo.includes('healthJson,') && digitDemo.includes('warmupJson,'), 'digit demo should import health and warmup helpers');
    assert(digitWarmup.includes('await retryRequest(() => healthJson(base') &&
           digitWarmup.includes('await retryRequest(() => warmupJson(base, warmPayload') &&
           digitWarmup.indexOf('warmupJson(base') < digitWarmup.indexOf('postToEndpoint(base, warmPayload)'),
      'digit demo should warm via /health and /warmup before legacy inference fallback');

    const handwritingWarmupStart = handwritingDemo.indexOf('async function warmUpServer({ manual = false } = {})');
    const handwritingWarmupEnd = handwritingDemo.indexOf('async function scoreDigit()', handwritingWarmupStart);
    const handwritingWarmup = handwritingDemo.slice(handwritingWarmupStart, handwritingWarmupEnd);
    assert(handwritingDemo.includes('healthJson,') && handwritingDemo.includes('warmupJson,'), 'handwriting demo should import health and warmup helpers');
    assert(handwritingWarmup.includes('await retryRequest(() => healthJson(base') &&
           handwritingWarmup.includes('await retryRequest(() => warmupJson(base, warmPayload') &&
           handwritingWarmup.indexOf('warmupJson(base') < handwritingWarmup.indexOf('postToEndpoint(base, warmPayload)'),
      'handwriting demo should warm via /health and /warmup before legacy score fallback');

    [
      ['nonogram', nonogramDemo, 'async function warmUpServer()', 'async function loadPuzzle()', 'warmupJson(base, {}', 'postToEndpoint(base, {})'],
      ['minesweeper', minesweeperDemo, 'async function warmUpServer()', 'async function loadPuzzle()', 'warmupJson(base, warmPayload', 'postToEndpoint(base, warmPayload)'],
      ['pizza tips', pizzaDemo, 'async function warmUpServer()', 'function applyResponse', 'warmupJson(base, payload', 'postToEndpoint(base, payload)']
    ].forEach(([name, source, startNeedle, endNeedle, warmupNeedle, fallbackNeedle]) => {
      const start = source.indexOf(startNeedle);
      const end = source.indexOf(endNeedle, start);
      const sectionText = source.slice(start, end);
      assert(source.includes('healthJson,') && source.includes('warmupJson,'), `${name} demo should import health and warmup helpers`);
      assert(sectionText.includes('await retryRequest(() => healthJson(base') &&
             sectionText.includes(warmupNeedle) &&
             sectionText.indexOf('warmupJson(base') < sectionText.indexOf(fallbackNeedle),
        `${name} demo should warm via /health and /warmup before legacy inference fallback`);
    });
    const shapeWarmupStart = shapeDemo.indexOf('async function warmUpServer()');
    const shapeWarmupEnd = shapeDemo.indexOf('async function classify()', shapeWarmupStart);
    const shapeWarmup = shapeDemo.slice(shapeWarmupStart, shapeWarmupEnd);
    assert(shapeDemo.includes('healthJson,') && shapeDemo.includes('warmupJson,'), 'shape demo should import health and warmup helpers');
    assert(shapeWarmup.includes('await retryRequest(() => healthJson(base') &&
           shapeWarmup.includes('warmupJson(base, warmPayload') &&
           shapeWarmup.indexOf('warmupJson(base') < shapeWarmup.indexOf('postToEndpoint(base, warmPayload)'),
      'shape demo should warm via /health and /warmup before legacy inference fallback');

    assert(whisperMonitor.includes("requestJson(joinUrl(normalized, '/warmup')") &&
           whisperMonitor.includes("fetch(joinUrl(normalized, '/transcribe')") &&
           whisperMonitor.indexOf("joinUrl(normalized, '/warmup')") < whisperMonitor.indexOf("joinUrl(normalized, '/transcribe')"),
      'Whisper monitor should use /warmup before falling back to silent transcription');

    assert(digitLambda.includes('def health_response():') &&
           digitLambda.includes('def warmup_response(start):') &&
           digitLambda.includes('path in ("/", "/health")') &&
           digitLambda.includes('path == "/warmup"') &&
           digitLambda.includes('load_latent_stats()') &&
           digitLambda.includes('model.decode(torch.zeros(1, LATENT_DIM'),
      'digit Lambda should expose cheap health and real warmup routes');
    assert(handwritingLambda.includes('def health_response():') &&
           handwritingLambda.includes('def warmup_response(start):') &&
           handwritingLambda.includes('path in ("/", "/health")') &&
           handwritingLambda.includes('path == "/warmup"') &&
           handwritingLambda.includes('model(torch.zeros(1, 1, 28, 28'),
      'handwriting Lambda should expose cheap health and real warmup routes');
    assert(nonogramLambda.includes('def health_response():') &&
           nonogramLambda.includes('def warmup_response(start):') &&
           nonogramLambda.includes('path in ("/", "/health")') &&
           nonogramLambda.includes('path == "/warmup"') &&
           nonogramLambda.includes('torch.full((1, GRID_SIZE, GRID_SIZE)'),
      'nonogram Lambda should expose cheap health and model warmup routes');
    assert(minesweeperLambda.includes('def health_response():') &&
           minesweeperLambda.includes('def warmup_response(start):') &&
           minesweeperLambda.includes('path in ("/", "/health")') &&
           minesweeperLambda.includes('path == "/warmup"') &&
           minesweeperLambda.includes('solve_game(seed=1, max_steps=1'),
      'minesweeper Lambda should expose cheap health and model warmup routes');
    assert(pizzaLambda.includes('const healthPayload = () =>') &&
           pizzaLambda.includes('const warmupPayload = (startedAt) =>') &&
           pizzaLambda.includes("path === '/warmup'") &&
           pizzaLambda.includes("path === '/' || path === '/health'"),
      'pizza tips Lambda should expose health and warmup routes');
    assert(whisperLambda.includes('def health_payload():') &&
           whisperLambda.includes('def warmup_response(start):') &&
           whisperLambda.includes('path == "/warmup"') &&
           whisperLambda.includes('load_whisper()'),
      'Whisper Lambda should expose health and explicit model warmup routes');
    [
      ['nonogram', nonogramEntrypoint, nonogramDockerfile],
      ['minesweeper', minesweeperEntrypoint, minesweeperDockerfile],
      ['whisper', whisperEntrypoint, whisperDockerfile]
    ].forEach(([name, entrypoint, dockerfile]) => {
      assert(dockerfile.includes('COPY entrypoint.py') && dockerfile.includes('CMD ["entrypoint.handler"]'),
        `${name} image should route through the lightweight health entrypoint`);
      assert(entrypoint.includes('def load_app():') &&
             entrypoint.includes('importlib.import_module("app")') &&
             entrypoint.includes('if method == "GET" and path not in ("/warmup",):') &&
             entrypoint.includes('return health_response()'),
        `${name} entrypoint should answer health without importing heavy app code`);
    });
  });

  section('Chatbot demo manual warmup and views', () => {
    const chatbotHtml = fs.readFileSync('demos/chatbot-demo.html', 'utf8');
    const projectCss = fs.readFileSync('css/components/project-page.css', 'utf8');
    const commonJs = fs.readFileSync('js/common/common.js', 'utf8');
    const projectGenerator = fs.readFileSync('build/generate-project-pages.js', 'utf8');
    const chatbotProject = JSON.parse(fs.readFileSync('content/projects/chatbotLora.json', 'utf8'));
    const vercel = fs.readFileSync('vercel.json', 'utf8');
    assert(chatbotHtml.includes("const DEFAULT_QWEN_API_URL = 'https://k8bys9gicf.execute-api.us-east-2.amazonaws.com/prod';"), 'chatbot-demo missing Qwen API URL');
    assert(chatbotHtml.includes("const DEFAULT_BEDROCK_API_URL = 'https://k8bys9gicf.execute-api.us-east-2.amazonaws.com/prod/bedrock';"), 'chatbot-demo missing Bedrock API URL');
    assert(chatbotHtml.includes("const DEFAULT_BEDROCK_STREAM_URL = 'https://6i6akxbdxx5qexaajudxuayoey0iopga.lambda-url.us-east-2.on.aws/';"), 'chatbot-demo missing Bedrock stream URL');
    assert(chatbotHtml.includes("const DEFAULT_BACKEND_ID = 'bedrock';"), 'chatbot-demo should default to Bedrock');
    assert(chatbotHtml.includes("const BACKEND_DEFAULT_VERSION = 'bedrock-default-2026-05-06';"), 'chatbot-demo should version the Bedrock default migration');
    assert(chatbotHtml.includes('let selectedBackendId = storedDefaultVersion === BACKEND_DEFAULT_VERSION'), 'chatbot-demo should use Bedrock until the default migration has run');
    assert(chatbotHtml.includes('localStorage.setItem(BACKEND_DEFAULT_VERSION_KEY, BACKEND_DEFAULT_VERSION);'), 'chatbot-demo should record the Bedrock default migration');
    assert(chatbotHtml.includes('if (!BACKENDS[selectedBackendId]) selectedBackendId = DEFAULT_BACKEND_ID;'), 'chatbot-demo should fall back to Bedrock for invalid stored backend');
    assert(!chatbotHtml.includes('ovodkr9oad'), 'chatbot-demo still references old API');
    assert(chatbotHtml.includes("postJson(backendUrl('/warmup'), {})"), 'chatbot-demo missing warm-up API call');
    assert(chatbotHtml.includes('id="backend-select"'), 'chatbot-demo missing backend selector');
    assert(chatbotHtml.includes('<option value="bedrock">Bedrock</option>'), 'chatbot-demo missing Bedrock backend option');
    assert(chatbotHtml.includes('<body data-chatbot-template="portfolio" data-mobile-layout="single-surface">'), 'chatbot-demo should identify the portfolio-aligned single-surface template');
    assert(chatbotHtml.includes('color-scheme: dark;'), 'chatbot-demo should use the dark website-aligned color scheme');
    assert(chatbotHtml.includes('--panel-raised: #1a2230;'), 'chatbot-demo missing raised panel token for website-aligned surfaces');
    assert(chatbotHtml.includes('--chat-height: 100%;'), 'chatbot-demo mobile layout should let the grid own the remaining viewport height');
    assert(chatbotHtml.includes('body[data-chatbot-template="portfolio"] {\n      padding-top: 0;') &&
           chatbotHtml.includes('html[data-embedded="true"] body {\n      height: 100dvh;\n      overflow: hidden;') &&
           chatbotHtml.includes('html[data-embedded="true"] .views,\n    html[data-embedded="true"] #regular-view'),
      'embedded chatbot should reset site body offset and fill the iframe without document clipping');
    assert(chatbotHtml.includes('.view-tabs {\n        display: grid;'), 'chatbot-demo should keep view tabs usable on mobile');
    assert(chatbotHtml.includes('.view-panel.active {\n        display: grid !important;'), 'chatbot-demo mobile layout should render the active chat surface');
    assert(chatbotHtml.includes('html[data-embedded="true"] .chat-stage {\n        height: 100%;'), 'embedded chatbot should size the chat stage to the iframe content area');
    assert(projectGenerator.includes('project-embed-${embedProjectId.toLowerCase()}'), 'project generator should add project-specific embed classes');
    assert(projectGenerator.includes('data-project-embed="${escapeHtml(embedProjectId)}"'), 'project generator should mark embeds with project ids');
    assert(projectCss.includes('.project-demo-panel .project-embed-chatbotlora'), 'project CSS should target the chatbot embed without affecting all demos');
    assert(projectCss.includes('.project-demo-shell:has(.project-embed-chatbotlora)') &&
           projectCss.includes('height:clamp(500px, 64svh, 620px);'),
      'chatbot embed should use a bounded mobile iframe height');
    assert(commonJs.includes('shouldAutoResizeProjectEmbed') &&
           commonJs.includes("!ifr.closest('.project-embed-chatbotlora')") &&
           commonJs.includes("ifr.setAttribute('scrolling', 'auto');"),
      'project iframe resizing should not auto-grow or disable scrolling on the chatbot embed');
    assert(chatbotProject.demoInstructions.bullets.some((item) => item.includes('Bedrock is the default live backend')), 'chatbot project instructions should describe the current Bedrock default');
    assert(chatbotHtml.includes('id="qwen-startup-notice"'), 'chatbot-demo should show Qwen cold-start notice when Qwen is selected');
    assert(chatbotHtml.includes('function showQwenStartupNotice()'), 'chatbot-demo missing Qwen startup notice handler');
    assert(chatbotHtml.includes("if (selectedBackendId === 'qwen-sagemaker') showQwenStartupNotice();"), 'chatbot-demo should show the startup notice when switching to Qwen');
    assert(chatbotHtml.includes('Press Start server and allow up to about five minutes'), 'chatbot-demo Qwen notice should explain cold-start time');
    assert(chatbotHtml.includes('alwaysOn: true'), 'chatbot-demo should treat Bedrock as always live');
    assert(chatbotHtml.includes('streamUrl: (RUNTIME_CONFIG.bedrockStreamUrl || DEFAULT_BEDROCK_STREAM_URL).replace(/\\/$/, \'\')'), 'chatbot-demo should configure a Bedrock stream URL');
    assert(chatbotHtml.includes('function submitBedrockStream(ctx, requestId, body, backend, liveNode = null)'), 'chatbot-demo missing Bedrock streaming submit helper');
    assert(chatbotHtml.includes("console.log('[chatbot-demo] bedrock:token', text);"), 'chatbot-demo should log Bedrock stream tokens to the console');
    assert(chatbotHtml.includes('renderAssistantAnswer(liveNode, streamedAnswer, liveLinks, {'), 'chatbot-demo should format streamed Bedrock tokens through the markdown renderer immediately');
    assert(chatbotHtml.includes('liveLinks = sourceItems(event);'), 'chatbot-demo should use stream metadata while rendering live source links');
    assert(!chatbotHtml.includes('window.setTimeout(renderLiveAnswer, 80)'), 'chatbot-demo should not batch visible Bedrock tokens behind a timer');
    assert(chatbotHtml.includes('markAlwaysLiveBackend(`${activeBackend().label} is ready. No startup or shutdown timer is needed.`);'), 'chatbot-demo should mark Bedrock as always live without timers');
    assert(chatbotHtml.includes('function normalizeAnswerLinks(container)'), 'chatbot-demo should normalize rendered answer links');
    assert(chatbotHtml.includes("link.target = '_blank';"), 'chatbot-demo answer links should open outside the embedded frame');
    assert(chatbotHtml.includes("link.rel = 'noopener noreferrer';"), 'chatbot-demo answer links should use safe external-link rel attributes');
    assert(chatbotHtml.includes("localStorage.setItem(BACKEND_STORAGE_KEY, selectedBackendId);"), 'chatbot-demo should persist selected backend');
    assert(chatbotHtml.includes('let serverReady = false;'), 'chatbot-demo should gate sending on warm server state');
    assert(chatbotHtml.includes('function isStartupInProgressPayload(payload)'), 'chatbot-demo missing startup-in-progress status helper');
    assert(chatbotHtml.includes("createChatContext('regular')"), 'chatbot-demo missing regular chat context');
    assert(chatbotHtml.includes("createChatContext('popup')"), 'chatbot-demo missing pop-up chat context');
    assert(chatbotHtml.includes('id="regular-view"'), 'chatbot-demo missing regular view');
    assert(chatbotHtml.includes('id="popup-view"'), 'chatbot-demo missing pop-up view');
    assert(chatbotHtml.includes('id="popup-launcher"'), 'chatbot-demo missing pop-up launcher');
    assert(chatbotHtml.includes('class="server-control"'), 'chatbot-demo server control should live in toolbar');
    assert(!chatbotHtml.includes('id="server-panel"'), 'chatbot-demo should not render a standalone server panel');
    assert(chatbotHtml.includes('<details class="timer-details" id="timer-details">'), 'chatbot-demo timer should reveal details on tap');
    assert(chatbotHtml.includes('id="server-timer-label"'), 'chatbot-demo radial timer missing label');
    assert(chatbotHtml.includes('class="timer-ring"'), 'chatbot-demo radial timer missing ring');
    assert(chatbotHtml.includes('id="server-timer-value"'), 'chatbot-demo radial timer missing counter');
    assert(chatbotHtml.includes('id="server-detail-endpoint"'), 'chatbot-demo timer details missing endpoint status');
    assert(chatbotHtml.includes('id="server-detail-instances"'), 'chatbot-demo timer details missing instance counts');
    assert(chatbotHtml.includes('id="server-detail-warm"'), 'chatbot-demo timer details missing warm ETA');
    assert(chatbotHtml.includes('id="server-detail-shutoff"'), 'chatbot-demo timer details missing shutoff ETA');
    assert(chatbotHtml.includes('id="server-detail-basis"'), 'chatbot-demo timer details missing timer basis');
    assert(chatbotHtml.includes('function updateTimerDetails(info, basis'), 'chatbot-demo should update AWS timer details from status payloads');
    assert(chatbotHtml.includes('--timer-progress'), 'chatbot-demo radial timer missing progress CSS variable');
    assert(chatbotHtml.includes('conic-gradient(var(--timer-color) var(--timer-progress)'), 'chatbot-demo radial timer missing conic progress ring');
    assert(chatbotHtml.includes("serverTimer.style.setProperty('--timer-progress'"), 'chatbot-demo radial timer should update progress dynamically');
    assert(chatbotHtml.includes("label: 'Starts in'"), 'chatbot-demo startup timer should label time until start');
    assert(chatbotHtml.includes("label: 'Shuts off in'"), 'chatbot-demo ready timer should label time until shutoff');
    assert(chatbotHtml.includes('function readyTimerColor(remaining)'), 'chatbot-demo missing ready timer color helper');
    assert(chatbotHtml.includes("if (remaining <= 60) return 'var(--danger)';"), 'chatbot-demo ready timer should turn red in the final minute');
    assert(chatbotHtml.includes('syncStartupEta(submitInfo.warmSeconds'), 'chatbot-demo should sync startup timer to AWS submit ETA');
    assert(chatbotHtml.includes('syncStartupEta(resultInfo.warmSeconds'), 'chatbot-demo should sync startup timer to AWS poll ETA');
    assert(chatbotHtml.includes('function beginSharedStartup(status, reason'), 'chatbot-demo should resume shared AWS warmup from status');
    assert(chatbotHtml.includes('function pollSharedStartupStatus()'), 'chatbot-demo should poll shared startup status');
    assert(chatbotHtml.includes('sharedStatusTimer = window.setInterval(pollSharedStartupStatus, 3000);'), 'chatbot-demo should keep polling shared startup status');
    assert(chatbotHtml.includes('ctx.send.disabled = active ? false : !serverReady;'), 'chatbot-demo send buttons should be gated by shared serverReady state');
    assert(chatbotHtml.includes('suggestions: []'), 'chatbot-demo should track shortcut buttons by context');
    assert(chatbotHtml.includes('followups: []'), 'chatbot-demo should track follow-up buttons by context');
    assert(chatbotHtml.includes('pendingFollowupContext: null'), 'chatbot-demo should store pending follow-up context');
    assert(chatbotHtml.includes('[...ctx.suggestions, ...ctx.followups].forEach(button =>'), 'chatbot-demo readiness updates should include shortcut and follow-up buttons');
    assert(chatbotHtml.includes('button.disabled = !serverReady;'), 'chatbot-demo shortcut buttons should be disabled until the server is ready');
    assert(chatbotHtml.includes('if (!serverReady) return;'), 'chatbot-demo shortcut click handlers should guard against cold server state');
    assert(chatbotHtml.includes('submitSuggestedPrompt(ctx, text);'), 'chatbot-demo shortcut buttons should auto-submit through the guarded prompt flow');
    assert(chatbotHtml.includes('ctx.suggestions.push(button);'), 'chatbot-demo should register generated shortcut buttons');
    assert(chatbotHtml.includes('(data?.resource_suggestions || [])'), 'chatbot-demo should include API resource suggestions in source links');
    assert(chatbotHtml.includes('function renderAssistantAnswer(container, text, links, options = {})'), 'chatbot-demo should render formatted assistant answers through one source-aware path');
    assert(chatbotHtml.includes('function renderFormattedAnswer(container, text, state)'), 'chatbot-demo should preserve markdown formatting before source linking');
    assert(chatbotHtml.includes('function appendInlineMarkdown(container, text, state'), 'chatbot-demo should preserve bold, italic, markdown links, and bare URL fallback formatting');
    assert(chatbotHtml.includes('function autoLinkSourcePhrases(container, links, state)'), 'chatbot-demo should link source phrases inside answers');
    assert(chatbotHtml.includes('function appendSourceDropdown(container, links)'), 'chatbot-demo should render sources in an optional dropdown');
    assert(chatbotHtml.includes("details.className = 'source-menu'"), 'chatbot-demo source dropdown missing details element');
    assert(chatbotHtml.includes('Other Relevant Sources'), 'chatbot-demo source dropdown missing label');
    assert(chatbotHtml.includes("link.className = 'source-pill'"), 'chatbot-demo source dropdown missing source link pills');
    assert(chatbotHtml.includes('function addFollowups(container, ctx, data, links, previousQuestion)'), 'chatbot-demo should render post-answer follow-up chips');
    assert(chatbotHtml.includes('grid-template-columns: minmax(0, 1fr);'), 'chatbot-demo follow-up questions should stack vertically');
    assert(chatbotHtml.includes('width: 100%;'), 'chatbot-demo follow-up question buttons should span the response width');
    assert(chatbotHtml.includes("source: 'recommended_followup'"), 'chatbot-demo follow-up context should identify recommended follow-ups');
    assert(chatbotHtml.includes('body.followup_context = followupContext'), 'chatbot-demo should submit follow-up context to the API');
    assert(chatbotHtml.includes('submitSuggestedPrompt(ctx, text, followupContext'), 'chatbot-demo follow-up chips should submit through the guarded prompt flow');
    assert(chatbotHtml.includes('Plan a Grand Junction first day with Colorado National Monument and downtown'), 'chatbot-demo default prompts should target indexed Colorado National Monument and downtown content from a Grand Junction base');
    assert(chatbotHtml.includes('Build a Grand Junction weekend with Riverfront Trail, downtown, and local food'), 'chatbot-demo default prompts should target indexed Riverfront Trail, downtown, and restaurant content');
    assert(chatbotHtml.includes('Create a Grand Junction base-camp day trip to Grand Mesa lakes'), 'chatbot-demo default prompts should target indexed Grand Mesa content from a Grand Junction base');
    assert(!chatbotHtml.includes('Plan a red-rock first day in Grand Junction'), 'chatbot-demo should not use older broad default prompts');
    assert(!chatbotHtml.includes('Plan a scenic first day'), 'chatbot-demo should not use generic default prompts');
    assert(!chatbotHtml.includes('Find outdoor ideas'), 'chatbot-demo should not use generic default prompts');
    assert(chatbotHtml.includes('white-space: normal;'), 'chatbot-demo assistant markdown should not preserve extra source newlines as visual gaps');
    assert(!chatbotHtml.includes('appendSources(answer'), 'chatbot-demo should not render always-visible source chips after answers');
    assert(chatbotHtml.includes('status.shutdownSeconds ?? DEFAULT_WARM_HOLD_SEC'), 'chatbot-demo should preserve accurate zero-second shutdown estimates');
    assert(chatbotHtml.includes('statusInfoFromPayload(result).shutdownSeconds'), 'chatbot-demo should use API shutdown estimates from completed jobs');
    assert(chatbotHtml.includes('if (duration <= 0) return;'), 'chatbot-demo ready timer should not loop when AWS reports zero seconds remaining');
    assert(!chatbotHtml.includes('status.shutdownSeconds || DEFAULT_WARM_HOLD_SEC'), 'chatbot-demo should not replace an accurate zero-second shutdown estimate');
    assert(!chatbotHtml.includes('id="start-notice"'), 'chatbot-demo should not show the old startup modal on load');
    assert(!chatbotHtml.includes('chatbot-start-deadline'), 'chatbot-demo should not persist startup countdown deadlines');
    assert(vercel.includes('k8bys9gicf.execute-api.us-east-2.amazonaws.com'), 'vercel.json missing new chatbot API host');
    assert(vercel.includes('6i6akxbdxx5qexaajudxuayoey0iopga.lambda-url.us-east-2.on.aws'), 'vercel.json missing Bedrock stream Function URL');
    assert(!vercel.includes('ovodkr9oad.execute-api.us-east-2.amazonaws.com'), 'vercel.json still allows old chatbot API host');
    const startConst = chatbotHtml.match(/const START_TIMEOUT_SEC = (\d+);/);
    const warmSec = startConst ? parseInt(startConst[1], 10) : 0;
    assert(warmSec >= 270, 'chatbot-demo start timeout < measured cold-start estimate');

    const passiveStart = chatbotHtml.indexOf('async function syncPassiveStatus()');
    const passiveEnd = chatbotHtml.indexOf('function stageMessage', passiveStart);
    const passiveSection = chatbotHtml.slice(passiveStart, passiveEnd);
    assert(passiveSection.includes('fetchStatusInfo()'), 'passive status should only fetch status');
    assert(passiveSection.includes("beginSharedStartup(status, 'passive-status')"), 'passive status should resume a warmup started elsewhere');
    assert(!passiveSection.includes('userStartedWarmup'), 'passive shared warmup pickup should not depend on this instance starting it');
    assert(!passiveSection.includes("postJson(backendUrl('/warmup')"), 'passive status must not initiate warmup');
    assert(!passiveSection.includes('startWarmupTimer('), 'passive status must not start countdown timer');

    const sharedStart = chatbotHtml.indexOf('async function pollSharedStartupStatus()');
    const sharedEnd = chatbotHtml.indexOf('async function syncPassiveStatus()', sharedStart);
    const sharedSection = chatbotHtml.slice(sharedStart, sharedEnd);
    assert(sharedSection.includes('fetchStatusInfo()'), 'shared startup polling should use status endpoint');
    assert(!sharedSection.includes("postJson(backendUrl('/warmup')"), 'shared startup polling must not submit a warmup job');

    const warmupStart = chatbotHtml.indexOf('async function warmupServer()');
    const warmupEnd = chatbotHtml.indexOf('async function pollWarmup', warmupStart);
    const warmupSection = chatbotHtml.slice(warmupStart, warmupEnd);
    assert(warmupSection.includes('if (isAlwaysLiveBackend())'), 'Bedrock warmup button should run a status check only');
    assert(warmupSection.includes('await checkAlwaysLiveBackend();'), 'Bedrock warmup button should check status without starting a timer');
    assert(warmupSection.includes('userStartedWarmup = true;'), 'warmup should be explicitly user initiated');
    assert(warmupSection.includes('startWarmupTimer({ durationSec: START_TIMEOUT_SEC, reason: \'manual-warmup\' });'), 'manual warmup should start countdown timer');
    assert(warmupSection.includes("postJson(backendUrl('/warmup'), {})"), 'manual warmup should call AWS warmup endpoint');

    const startupSection = chatbotHtml.match(/\/\/ Startup status helpers[\s\S]*?\/\/ End startup status helpers/);
    assert(startupSection, 'chatbot-demo startup-in-progress helper section missing');
    const startupEnv = {};
    vm.runInNewContext(`${startupSection[0]}
      checks = [
        isStartupInProgressPayload({
          status: 'Off',
          current: 0,
          desired: 0,
          eta: { to_warm_seconds: 270 },
          stage: { name: 'accepted', endpoint: { current: 0, desired: 0 }, metrics: { approx_backlog_per_instance: 0, has_backlog_without_capacity: 0 } }
        }),
        isStartupInProgressPayload({ status: 'Starting', current: 0, desired: 1 }),
        isStartupInProgressPayload({ stage: { name: 'booting', endpoint: { current: 0, desired: 1 } } }),
        isStartupInProgressPayload({ queue_estimate: 1 }),
        isStartupInProgressPayload({ stage: { name: 'accepted', metrics: { approx_backlog_per_instance: 1 } } })
      ];`, startupEnv);
    assert(startupEnv.checks[0] === false, 'cold off status with ETA should not start timer');
    assert(startupEnv.checks.slice(1).every(Boolean), 'active startup statuses should start timer');

    const timerStart = chatbotHtml.indexOf('function startWarmupTimer');
    const timerEnd = chatbotHtml.indexOf('function startReadyTimer', timerStart);
    const startSection = chatbotHtml.slice(timerStart, timerEnd);
    assert(startSection.includes('function currentStartingRemaining()'), 'chatbot-demo starting countdown helper missing');
    const warmEnv = {
      startingTimer: null,
      startupTimer: null,
      startupDeadline: 0,
      startupDuration: 0,
      START_TIMEOUT_SEC: warmSec,
      Date: { now: () => 0 },
      Math,
      serverTimer: { textContent: '' },
      recordLog: () => {},
      setServerVisual: () => {},
      setTimerDisplay: () => {},
      formatClock: () => '',
      window: {
        setInterval: () => 1,
        clearInterval: () => {}
      }
    };
    vm.runInNewContext(startSection, warmEnv);
    warmEnv.startWarmupTimer({ durationSec: warmSec });
    warmEnv.Date.now = () => (warmSec - 1) * 1000;
    assert(warmEnv.currentStartingRemaining() > 0, 'starting countdown ended too early');
    warmEnv.Date.now = () => (warmSec + 1) * 1000;
    assert(warmEnv.currentStartingRemaining() === 0, 'starting countdown did not finish after configured startup budget');
  });

  section('404 rewrites and portfolio page sections', () => {
    checkFileContains('404.html', 'js/common/404-redirect.js');
    checkFileContains('js/common/404-redirect.js', '/portfolio/${encodeURIComponent(project)}');
    checkFileContains('pages/portfolio.html', 'id="portfolio-carousel"');
    checkFileContains('pages/portfolio.html', 'id="projects"');
    checkFileContains('pages/portfolio.html', 'id="modals"');
    checkFileContains('pages/portfolio.html', 'id="filters"');
    checkFileContains('pages/portfolio.html', 'portfolio-library-section');
    const portfolioHtml = readFile('pages/portfolio.html');
    assert(!portfolioHtml.includes('id="filter-menu"'), 'portfolio page should not include filter menu');
    assert(!portfolioHtml.includes('id="see-more"'), 'portfolio page should not include see-more toggle');
    assert(!portfolioHtml.includes('Which hiring track?'), 'portfolio page should not include audience filter copy');
    assert(!portfolioHtml.includes('use the filters'), 'portfolio hero copy should not reference filters');

    const portfolioScript = readFile('js/portfolio/portfolio.js');
    assert(!portfolioScript.includes('function initSeeMore'), 'portfolio script should not include see-more initializer');
    assert(!portfolioScript.includes('filter-menu'), 'portfolio script should not depend on filter menu');

    const commonScript = readFile('js/common/common.js');
    assert(!commonScript.includes('run(window.initSeeMore);'), 'common portfolio init should not run see-more');

    checkFileContains('js/common/audience-config.js', "portfolioAllPath: '/portfolio?audience=analytics'");
    checkFileContains('js/common/audience-config.js', "portfolioAllPath: '/portfolio?audience=data-science'");
    checkFileContains('js/common/audience-config.js', "portfolioAllPath: '/portfolio?audience=tourism'");
    checkFileContains('build/templates/header.partial.html', 'href="portfolio"');
    checkFileContains('pages/analytics.html', 'href="portfolio?audience=analytics"');
    checkFileContains('pages/data-science.html', 'href="portfolio?audience=data-science"');
    checkFileContains('pages/tourism.html', 'href="portfolio?audience=tourism"');
  });

  section('Base hrefs and redirect sanity', () => {
    checkFileContains('index.html', 'content="0; url=/analytics"');
    checkFileContains('index.html', "window.location.replace(target)");
    checkFileContains('index.html', 'rel="canonical" href="https://www.danielshort.me/analytics"');
    checkFileContains('index.html', 'name="robots" content="noindex, follow"');

    ['pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html','pages/resume-pdf.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html',
     'pages/tools.html','pages/tools-dashboard.html','pages/search.html','pages/sitemap.html','pages/games.html','pages/short-links.html','pages/word-frequency.html','pages/text-compare.html','pages/point-of-view-checker.html','pages/oxford-comma-checker.html','pages/background-remover.html','pages/nbsp-cleaner.html','pages/ocean-wave-simulation.html','pages/qr-code-generator.html','pages/image-optimizer.html','pages/job-application-tracker.html','pages/ga4-utm-performance.html',
     'probability-engine.html','demos/chatbot-demo.html','demos/shape-demo.html','demos/sentence-demo.html','demos/slot-machine-demo.html','demos/stellar-dogfight-demo.html']
      .forEach(f => checkFileContains(f, '<base href="/">'));

    ['pages/analytics.html','pages/data-science.html','pages/destination-analytics.html','pages/tourism.html','pages/portfolio.html','pages/contact.html','pages/contributions.html','pages/privacy.html','pages/resume.html','pages/resume-pdf.html','pages/resume-analytics.html','pages/resume-data-science.html','pages/resume-tourism.html','pages/resume-analytics-pdf.html','pages/resume-data-science-pdf.html','pages/resume-tourism-pdf.html']
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
    checkFileContains('pages/resume-analytics-pdf.html', 'documents/Resume-Analytics.pdf');
    checkFileContains('pages/resume-data-science-pdf.html', 'documents/Resume-Data-Science.pdf');
    checkFileContains('pages/resume-tourism-pdf.html', 'documents/Resume-Tourism.pdf');
    checkFileContains('resume-pdf.html', 'resume-analytics-pdf');
    checkFileContains('pages/resume-pdf.html', 'resume-analytics-pdf');
    checkFileContains('contact.html', 'id="contact-form"');
    checkFileContains('pages/contact.html', 'action="/api/contact"');
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
    checkFileContainsOneOf('privacy.html', ['js/privacy/config.js', 'dist/site-consent.'], 'privacy.html missing consent loader reference');
    checkFileContainsOneOf('privacy.html', ['js/privacy/consent_manager.js', 'dist/site-consent.'], 'privacy.html missing consent manager reference');
    const pcfg = evalScript('js/privacy/config.js');
    const consentCode = readFile('js/privacy/consent_manager.js');
    const privacyCss = readFile('css/privacy.css');
    assert(pcfg.window.PrivacyConfig && pcfg.window.PrivacyConfig.vendors && pcfg.window.PrivacyConfig.vendors.ga4 && pcfg.window.PrivacyConfig.vendors.ga4.id,
           'PrivacyConfig missing GA4 vendor id');
    assert(consentCode.includes('pref-status-row'),
      'consent manager should render a locked necessary status row');
    assert(consentCode.includes('Required for site operation'),
      'consent manager should clarify that necessary cookies are required');
    assert(consentCode.includes('pref-disclosure'),
      'consent manager should render inline preference disclosures');
    assert(consentCode.includes("row.classList.toggle('is-expanded'"),
      'consent manager should toggle expanded preference rows');
    assert(consentCode.includes('pcz-consent-critical-styles') && consentCode.includes('#pcz-banner .pcz-primary'),
      'consent manager should inject critical readable banner styles before privacy.css loads');
    assert(privacyCss.includes('.pref-option-head'),
      'privacy.css missing aligned preference row layout');
    assert(privacyCss.includes('.pref-disclosure'),
      'privacy.css missing animated preference disclosure styling');
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
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-unlocks.js"></script>'),
      'stellar dogfight demo should load unlock pacing script');
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-art.js"></script>'),
      'stellar dogfight demo should load art manifest script');
    assert(!demoHtml.includes('<script src="js/vendor/pixi.min.js"></script>') &&
           !demoHtml.includes('<script src="js/demos/stellar-dogfight-renderer-pixi.js"></script>'),
      'stellar dogfight demo should not load the experimental Pixi renderer by default');
    assert(demoHtml.includes('<script src="js/demos/stellar-dogfight-demo.js"></script>'),
      'stellar dogfight demo should load runtime script');
    assert(demoHtml.indexOf('js/demos/stellar-dogfight-unlocks.js') < demoHtml.indexOf('js/demos/stellar-dogfight-demo.js'),
      'stellar dogfight unlock pacing script should load before the runtime');
    assert(demoHtml.indexOf('js/demos/stellar-dogfight-art.js') < demoHtml.indexOf('js/demos/stellar-dogfight-demo.js'),
      'stellar dogfight art manifest should load before the runtime');
    assert(demoHtml.includes('Space+Grotesk') && demoHtml.includes('Inter:wght'),
      'stellar dogfight demo should load professional cockpit UI fonts');
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
    assert(demoHtml.includes('class="mission-header mission-topbar"') &&
           demoHtml.includes('data-role="topbar-summary"') &&
           demoHtml.includes('data-action="command-menu"'),
      'stellar dogfight demo should use a compact play-first top bar');
    assert(demoHtml.includes('class="mission-sidebar command-drawer"') &&
           demoHtml.includes('data-role="command-menu"') &&
           demoHtml.includes('data-action="command-menu-close"'),
      'stellar dogfight demo should move management panels behind a command drawer');
    assert(!demoHtml.includes('mission-brief-grid') && !demoHtml.includes('mission-pill-row'),
      'stellar dogfight first load should not include long explanatory header clutter');
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
    assert(demoHtml.includes('data-mode-feature="scoreMode"') && demoHtml.includes('data-mode-feature="frontierMode"'),
      'stellar dogfight demo should gate advanced run modes behind progressive unlocks');
    assert(demoHtml.includes('data-feature-reveal="scoreMode"') && demoHtml.includes('data-feature-reveal="premium"'),
      'stellar dogfight demo should gate advanced details behind feature reveal attributes');
    assert(demoHtml.includes('data-action="help"'),
      'stellar dogfight demo should expose help action');
    assert(demoHtml.includes('data-action="preset-save"') && demoHtml.includes('data-action="preset-load"'),
      'stellar dogfight demo should expose loadout preset actions');
    assert(demoHtml.includes('data-action="tutorial"') && demoHtml.includes('data-action="glossary"'),
      'stellar dogfight demo should expose tutorial and glossary actions');
    assert(demoHtml.includes('data-action="replay-last-loadout"'),
      'stellar dogfight demo should expose replay-last-loadout action');
    assert(demoHtml.includes('data-ui-icon="play"') &&
           demoHtml.includes('data-ui-icon="menu"') &&
           demoHtml.includes('data-tab-icon="ship"') &&
           demoHtml.includes('data-icon-only="true"'),
      'stellar dogfight demo should use compact SVG icon hooks for menus and shortcuts');
    assert(demoHtml.includes('data-role="touch-controls"') &&
           demoHtml.includes('data-touch-stick="move"') &&
           demoHtml.includes('data-touch-stick="aim"') &&
           demoHtml.includes('data-touch-action="boost"') &&
           demoHtml.includes('data-touch-action="brake"') &&
           demoHtml.includes('data-touch-action="pause"'),
      'stellar dogfight demo should expose mobile twin-stick touch controls');
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
    assert(demoHtml.includes('<span class="panel-disclosure-title">Performance</span>') &&
           demoHtml.includes('<span class="panel-disclosure-meta">Debug</span>') &&
           demoHtml.includes('data-option="0.75"'),
      'stellar dogfight demo should expose ungated performance diagnostics and sub-1 render scale');
    assert(demoHtml.includes('data-option="controller"'),
      'stellar dogfight demo should expose controller input mode');
    assert(demoHtml.includes('Astralite Forge') && demoHtml.includes('data-option="score"') && demoHtml.includes('data-option="daily"') && demoHtml.includes('data-option="boss"'),
      'stellar dogfight demo should expose Forge wording and quick-run modes');
    assert(!demoHtml.includes('const STORAGE_KEY = "stellarDogfightProgress";'),
      'stellar dogfight runtime should not be inline in HTML');

    const demoCss = fs.readFileSync('css/components/stellar-dogfight.css', 'utf8');
    assert(demoCss.includes('.mission-shell') && demoCss.includes('.arena-frame'),
      'stellar dogfight CSS component appears incomplete');
    assert(demoCss.includes('body.is-hud-compact') && demoCss.includes('body.is-hud-scale-lg'),
      'stellar dogfight CSS should include HUD presentation variants');
    assert(demoCss.includes('.premium-card'),
      'stellar dogfight CSS should include premium card presentation');
    assert(demoCss.includes('.mission-title-compact') && demoCss.includes('.command-drawer') && demoCss.includes('body.is-command-menu-open'),
      'stellar dogfight CSS should style the compact top bar and command drawer');
    assert(demoCss.includes('.ui-icon') && demoCss.includes('.btn-icon-only') && demoCss.includes('.command-drawer .tab-btn-label'),
      'stellar dogfight CSS should style concise SVG icon buttons and compact drawer tabs');
    assert(demoCss.includes('.touch-controls') &&
           demoCss.includes('.touch-stick') &&
           demoCss.includes('.touch-action-btn') &&
           demoCss.includes('body.is-touch-controls-active') &&
           demoCss.includes('touch-action: none') &&
           demoCss.includes('env(safe-area-inset-bottom)'),
      'stellar dogfight CSS should style safe mobile tap controls');
    assert(demoCss.includes('body.is-hangar') &&
           demoCss.includes('height: var(--viewport-height);') &&
           demoCss.includes('overflow: hidden;') &&
           demoCss.includes('body.is-hangar .arena-frame'),
      'stellar dogfight hangar should fit the initial game screen without document scrolling');
    assert(demoCss.includes('body.is-playing .arena-frame') &&
           demoCss.includes('aspect-ratio: auto;') &&
           demoCss.includes('width: 100%;'),
      'stellar dogfight playing arena should fit narrow mobile viewports without horizontal overflow');
    assert(demoCss.includes('body.is-playing.is-performance-mode .arena-frame') &&
           demoCss.includes('body.is-playing.is-performance-mode .perf-overlay') &&
           demoCss.includes('backdrop-filter: none;'),
      'stellar dogfight CSS should strip costly fullscreen chrome in performance mode');
    assert(demoCss.includes('.pixi-stage') && demoCss.includes('body.is-pixi-renderer .arena-frame > canvas[data-role=') &&
           demoCss.includes('pointer-events: none;'),
      'stellar dogfight CSS should isolate the experimental WebGL renderer without stealing pointer input');
    assert(demoCss.includes('.panel-disclosure'),
      'stellar dogfight CSS should style disclosure sections for progressive disclosure');
    assert(demoCss.includes('.hud-pill-group-meta') && demoCss.includes('body.is-hud-focused'),
      'stellar dogfight CSS should support contextual combat HUD decluttering');
    assert(demoCss.includes('.tab-badge') && demoCss.includes('.debrief-grid'),
      'stellar dogfight CSS should style new-tab badges and post-run debrief cards');
    assert(demoCss.includes('.result-rewards') && demoCss.includes('.result-details') && demoCss.includes('.overlay-card.is-result-card'),
      'stellar dogfight CSS should style simplified result rewards and collapsible run details');
    assert(demoCss.includes('.progress-step[data-state="new"]'),
      'stellar dogfight CSS should style newly unlocked roadmap states');
    assert(demoCss.includes('body.is-hangar .mission-main') && demoCss.includes('body.is-hangar .arena-frame'),
      'stellar dogfight CSS should keep the playable arena visible in the hangar');
    assert(demoCss.includes('.arena-frame::after') && demoCss.includes('rgba(151, 214, 255, 0.18)'),
      'stellar dogfight CSS should include polished cockpit frame and HUD styling');

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
    assert(runtimeJs.includes('const dpr = Math.max(AUTO_RENDER_MIN_SCALE, state.renderScale || window.devicePixelRatio || 1);'),
      'stellar dogfight minimap should support sub-1 renderScale-aware DPR');
    assert(runtimeJs.includes('function getKeybindConflicts(targetAction, key) {'),
      'stellar dogfight runtime missing keybind conflict guard');
    assert(runtimeJs.includes('function saveLoadoutPreset(slot) {') && runtimeJs.includes('function loadLoadoutPreset(slot) {'),
      'stellar dogfight runtime should support loadout presets');
    assert(runtimeJs.includes('function syncHudPresentation() {'),
      'stellar dogfight runtime should support HUD presentation settings');
    assert(runtimeJs.includes('const UI_ICON_PATHS = {') &&
           runtimeJs.includes('function renderUiIcon(name) {') &&
           runtimeJs.includes('function hydrateIconButtons(root = document) {') &&
           runtimeJs.includes('function getTabIconName(target) {'),
      'stellar dogfight runtime should render reusable SVG icons for menu shortcuts and options');
    assert(runtimeJs.includes('function supportsTouchControls() {') &&
           runtimeJs.includes('function setupTouchControls() {') &&
           runtimeJs.includes('function syncTouchControls() {') &&
           runtimeJs.includes('function handleTouchStickPointerDown(event) {') &&
           runtimeJs.includes('input.touch.firing') &&
           runtimeJs.includes('input.touch.moveX'),
      'stellar dogfight runtime should map mobile touch controls into gameplay input');
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
    assert(runtimeJs.includes('const braking = isActionActive("brake") || input.padBrake || input.touch.brake;') && runtimeJs.includes('state.brakeTurnBoost') && runtimeJs.includes('state.brakeSpeedClamp'),
      'stellar dogfight runtime should apply air brake handling to player flight');
    assert(runtimeJs.includes('const BALANCE_TUNING = {'),
      'stellar dogfight runtime missing balance tuning constants');
    assert(runtimeJs.includes('const DROP_PITY_THRESHOLDS = {'),
      'stellar dogfight runtime missing drop pity thresholds');
    assert(runtimeJs.includes('const PREMIUM_CURRENCY_LABEL = "Astralite";'),
      'stellar dogfight runtime missing premium currency label');
    assert(runtimeJs.includes('const PREMIUM_DROP_RUN_CAP = 4;'),
      'stellar dogfight runtime missing premium drop cap');
    assert(runtimeJs.includes('const UNLOCK_FEATURES = Array.isArray(UNLOCK_CONFIG.features)'),
      'stellar dogfight runtime should consume data-driven unlock feature config');
    assert(runtimeJs.includes('function shouldUnlockFeature(feature) {'),
      'stellar dogfight runtime should evaluate configured unlock predicates');
    assert(runtimeJs.includes('function syncModeOptions() {') && runtimeJs.includes('data-feature-reveal'),
      'stellar dogfight runtime should sync progressive mode and feature visibility');
    assert(runtimeJs.includes('function renderPremiumShop() {') && runtimeJs.includes('function buyPremiumItem(itemId) {'),
      'stellar dogfight runtime missing premium shop handlers');
    assert(runtimeJs.includes('state.runPremiumDrops < PREMIUM_DROP_RUN_CAP'),
      'stellar dogfight runtime missing capped premium drop logic');
    assert(runtimeJs.includes('progress.premiumCurrency += premiumAmount;'),
      'stellar dogfight runtime missing premium currency drops');
    assert(runtimeJs.includes('function updateRunRecords(summary) {'),
      'stellar dogfight runtime missing run-record tracking');
    assert(runtimeJs.includes('function isScoreAttackMode() {') && runtimeJs.includes('function finishQuickRun(reason) {') && runtimeJs.includes('const BOSS_RUSH_WAVES = 5;'),
      'stellar dogfight runtime should include timed and boss quick-run modes');
    assert(runtimeJs.includes('function unlockAchievement(id) {') && runtimeJs.includes('function noteTutorialGoal(goal) {'),
      'stellar dogfight runtime should include achievements and live tutorial goals');
    assert(runtimeJs.includes('function drawEnemyRoleLabel(enemy) {') && runtimeJs.includes('function drawEnemyShip(enemy) {'),
      'stellar dogfight runtime should include enemy role readability helpers');
    assert(runtimeJs.includes('function getUpgradeChoiceDescription(upgrade, nextLevel) {') && runtimeJs.includes('class="upgrade-synergy"'),
      'stellar dogfight runtime should add readable upgrade-card descriptions and synergy hints');
    assert(runtimeJs.includes('function startTutorial() {') && runtimeJs.includes('function openGlossary() {'),
      'stellar dogfight runtime should include tutorial/glossary handlers');
    assert(runtimeJs.includes('function replayLastLoadout() {'),
      'stellar dogfight runtime should include replay-last-loadout handler');
    assert(runtimeJs.includes('function renderRunAnalytics() {'),
      'stellar dogfight runtime should include run analytics renderer');
    assert(runtimeJs.includes('function captureRunUnlockBaseline() {') && runtimeJs.includes('function buildRunDebrief(summary, telemetry, reason) {'),
      'stellar dogfight runtime should capture unlock baselines and build post-run debriefs');
    assert(runtimeJs.includes('function openCommandMenu() {') &&
           runtimeJs.includes('function closeCommandMenu(options = {}) {') &&
           runtimeJs.includes('function syncTopbarSummary() {'),
      'stellar dogfight runtime should manage the simplified command drawer and top bar');
    assert(runtimeJs.includes('function buildRewardSummaryItems(rewards) {') &&
           runtimeJs.includes('function getRunResultModel(mode) {') &&
           runtimeJs.includes('function renderRunResultOverlay(mode) {'),
      'stellar dogfight runtime should render simplified end-run result overlays');
    assert(runtimeJs.includes('class="result-rewards"') &&
           runtimeJs.includes('class="result-summary-strip"') &&
           runtimeJs.includes('<details class="result-details">'),
      'stellar dogfight result overlay should emphasize rewards and collapse detailed debriefs');
    assert(runtimeJs.includes('function updateTabBadges() {') && runtimeJs.includes('function markPanelSeen(target) {'),
      'stellar dogfight runtime should track first-visit UI badges');
    assert(runtimeJs.includes('function renderStatusIcons() {'),
      'stellar dogfight runtime should include status icon rendering');
    assert(runtimeJs.includes('setOverlay("choice-event");'),
      'stellar dogfight runtime should include milestone choice overlay flow');
    assert(runtimeJs.includes('action: "restart"'),
      'stellar dogfight runtime missing result replay action');
    assert(runtimeJs.includes('const SPATIAL_GRID_CELL_SIZE = 240;') &&
           runtimeJs.includes('enemyGrid: new Map()') &&
           runtimeJs.includes('obstacleGrid: new Map()'),
      'stellar dogfight runtime should maintain spatial grids for hot collision queries');
    assert(runtimeJs.includes('function rebuildEnemySpatialGrid() {') &&
           runtimeJs.includes('function forEachEnemyNear(origin, radius, callback) {') &&
           runtimeJs.includes('forEachEnemyNear(bullet, bullet.radius + MAX_COLLISION_RADIUS'),
      'stellar dogfight runtime should use spatial enemy queries for projectile collisions');
    assert(runtimeJs.includes('function ensureObstacleSpatialGrid() {') &&
           runtimeJs.includes('function forEachObstacleNear(origin, radius, callback) {') &&
           runtimeJs.includes('forEachObstacleNear(point, buffer + MAX_COLLISION_RADIUS') &&
           runtimeJs.includes('forEachObstacleNear(viewOrigin, viewRadius'),
      'stellar dogfight runtime should use spatial obstacle queries for movement and shots');
    assert(runtimeJs.includes('function removeArrayIndex(array, index) {') &&
           runtimeJs.includes('removeArrayIndex(enemies, index);'),
      'stellar dogfight runtime should use swap-remove helpers on hot entity arrays');
    assert(runtimeJs.includes('const MINIMAP_RENDER_INTERVAL_MS = 125;') &&
           runtimeJs.includes('now - state.minimapLastRenderAt < interval') &&
           runtimeJs.includes('function queueMinimapRefresh() {'),
      'stellar dogfight minimap should throttle redraws and support forced refreshes');
    assert(runtimeJs.includes('const AUTO_RENDER_PIXEL_TIERS = [') &&
           runtimeJs.includes('{ id: "emergency", label: "Emergency", pixels: 220000 }') &&
           runtimeJs.includes('const WEBGL_RENDER_PIXEL_TIERS = [') &&
           runtimeJs.includes('{ id: "emergency", label: "Emergency", pixels: 450000 }') &&
           runtimeJs.includes('function getAutoRenderScaleInfo(baseDpr') &&
           runtimeJs.includes('function getPerformanceQualityLevel() {') &&
           runtimeJs.includes('AUTO_RENDER_MIN_SCALE') &&
           runtimeJs.includes('state.renderScaleAutoCap'),
      'stellar dogfight runtime should cap auto render scale through backend-aware performance-first backing-pixel tiers');
    assert(runtimeJs.includes('function acquireEffect(pool) {') &&
           runtimeJs.includes('function releaseEffect(pool, item) {') &&
           runtimeJs.includes('releaseEffect(effectPools.particles, removeArrayIndex(particles, i))'),
      'stellar dogfight runtime should pool short-lived visual effects');
    assert(runtimeJs.includes('function updateAdaptiveQuality(rawDeltaMs) {') &&
           runtimeJs.includes('function getAdaptiveRenderScaleCap() {') &&
           runtimeJs.includes('function shouldDrawEnemyTelegraph(enemy) {'),
      'stellar dogfight runtime should adapt render cost under frame pressure');
    assert(runtimeJs.includes('function parsePerfQueryOverrides() {') &&
           runtimeJs.includes('function markPerfBucket(kind, key, start) {') &&
           runtimeJs.includes('Top Bottleneck') &&
           runtimeJs.includes('Performance Mode') &&
           runtimeJs.includes('RAF Wait') &&
           runtimeJs.includes('update_ambient_ms') &&
           runtimeJs.includes('render_background_ms'),
      'stellar dogfight runtime should expose detailed optional performance diagnostics');
    assert(runtimeJs.includes('function setCachedText(element, value) {') &&
           runtimeJs.includes('function setCachedWidth(element, value) {') &&
           runtimeJs.includes('setCachedWidth(meters.hull, hullPct);'),
      'stellar dogfight HUD should avoid redundant DOM text and style writes');
    assert(runtimeJs.includes('const ART_CONFIG = window.STELLAR_DOGFIGHT_ART || {};') &&
           runtimeJs.includes('function loadGameArt() {') &&
           runtimeJs.includes('function drawSprite(id, x, y') &&
           runtimeJs.includes('function rasterizeArtEntry(entry) {') &&
           runtimeJs.includes('function getCachedBackgroundSurface() {'),
      'stellar dogfight runtime should load and cache manifest-driven art assets');
    assert(runtimeJs.includes('function setupPixiRenderer() {') &&
           runtimeJs.includes('function renderPixiFrame() {') &&
           runtimeJs.includes('function buildPixiFrameSnapshot() {') &&
           runtimeJs.includes('getRendererSetting() !== "pixi"') &&
           runtimeJs.includes('function loadPixiRendererScripts() {') &&
           runtimeJs.includes('state.renderBackend = "pixi";'),
      'stellar dogfight runtime should keep Pixi/WebGL behind an explicit experimental renderer flag');
    assert(runtimeJs.includes('image.src = definition.src;') &&
           runtimeJs.includes('render_backend,webgl_px_w,webgl_px_h'),
      'stellar dogfight runtime should keep Canvas on SVG art and log renderer backend diagnostics');
    assert(runtimeJs.includes('function renderBackground() {') &&
           runtimeJs.includes('function renderParticles() {') &&
           runtimeJs.includes('function renderBullets() {') &&
           runtimeJs.includes('function shouldUsePremiumVfx() {') &&
           runtimeJs.includes('function isCircleInBounds(item') &&
           runtimeJs.includes('function shouldDrawSecondaryCombatDetail() {'),
      'stellar dogfight runtime should include polished adaptive VFX renderers');
    assert(runtimeJs.includes('drawSprite("asteroidRock"') &&
           runtimeJs.includes('drawSprite("pickupCore"') &&
           runtimeJs.includes('drawShip(player.x, player.y, player.angle, player.radius, "#44d2c2", player.hitFlash > 0, player.thrusting, getPlayerArtId(player));'),
      'stellar dogfight runtime should use art sprites for battlefield entities with procedural fallbacks');
    assert(!runtimeJs.includes('forEachObstacleNear(entity, (entity.radius || 0) + MAX_COLLISION_RADIUS'),
      'stellar dogfight obstacle renderer should not reference an undefined entity');

    const unlockJs = fs.readFileSync('js/demos/stellar-dogfight-unlocks.js', 'utf8');
    assert(unlockJs.includes('window.STELLAR_DOGFIGHT_UNLOCKS') && unlockJs.includes('id: "frontierMode"'),
      'stellar dogfight unlock config should export late-game feature gates');
    assert(unlockJs.includes('id: "score"') && unlockJs.includes('feature: "scoreMode"'),
      'stellar dogfight unlock config should map run modes to feature gates');

    const artJs = fs.readFileSync('js/demos/stellar-dogfight-art.js', 'utf8');
    assert(artJs.includes('window.STELLAR_DOGFIGHT_ART') && artJs.includes('playerScout') && artJs.includes('enemyCommand'),
      'stellar dogfight art manifest should export player and enemy sprite definitions');
    assert(artJs.includes('rasterSrc') && artJs.includes('img/stellar-dogfight/raster/'),
      'stellar dogfight art manifest should include pre-rendered PNG asset paths');
    assert(fs.existsSync(path.join('js', 'vendor', 'pixi.min.js')),
      'stellar dogfight Pixi vendor script should be vendored for static deployment');
    assert(fs.existsSync(path.join('js', 'demos', 'stellar-dogfight-renderer-pixi.js')),
      'stellar dogfight WebGL renderer module missing');
    assert(fs.existsSync(path.join('build', 'build-stellar-dogfight-assets.js')),
      'stellar dogfight raster art build script missing');
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    assert(packageJson.scripts && packageJson.scripts['build:stellar-dogfight-assets'],
      'stellar dogfight raster art build script should be exposed through npm scripts');
    assert(!packageJson.dependencies || !packageJson.dependencies['pixi.js'],
      'experimental Pixi renderer should not be part of normal package dependencies');
    const buildSiteJs = fs.readFileSync('build/build-site.js', 'utf8');
    assert(!buildSiteJs.includes('dogfight-assets'),
      'normal site build should not generate experimental Stellar Dogfight raster assets');
    [
      'ship-player-scout.svg',
      'ship-player-heavy.svg',
      'enemy-screen.svg',
      'enemy-line.svg',
      'enemy-interceptor.svg',
      'enemy-support.svg',
      'enemy-artillery.svg',
      'enemy-siege.svg',
      'enemy-brawler.svg',
      'enemy-command.svg',
      'helper-drone.svg',
      'asteroid-rock.svg',
      'obstacle-plate.svg',
      'pickup-core.svg',
      'mine-core.svg',
      'vfx-burst.svg',
      'vfx-shield-ripple.svg',
      'background-nebula.svg'
    ].forEach((asset) => {
      assert(fs.existsSync(path.join('img', 'stellar-dogfight', asset)), `stellar dogfight art asset missing: ${asset}`);
    });

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
