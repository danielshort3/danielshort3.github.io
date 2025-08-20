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

// Evaluate a script in a minimal browser-like context
function evalScript(file) {
  const code = fs.readFileSync(file, 'utf8');
  const env = {
    window: {},
    document: {
      addEventListener: () => {},
      removeEventListener: () => {},
      querySelector: () => null,
      querySelectorAll: () => [],
      getElementById: () => null,
      createElement: () => ({ style: {}, classList: { add() {}, remove() {}, toggle() {} } }),
      body: {},
      documentElement: { style: { setProperty() {} } },
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
  vm.runInNewContext(code, env, { filename: file });
  return env;
}

try {
  // HTML checks across pages
  checkFileContains('index.html', 'Turning data into actionable insights');
checkFileContains('contact.html', '<title>Contact │ Daniel Short');
['index.html','contact.html','portfolio.html','contributions.html'].forEach(f => {
  checkFileContains(f, 'js/common/common.js');
  checkFileContains(f, 'class="skip-link"');
  checkFileContains(f, '<main id="main">');
});
['index.html','contact.html','portfolio.html','contributions.html','404.html'].forEach(f => {
  checkFileContains(f, 'og:image');
});
assert(fs.existsSync('robots.txt'), 'robots.txt missing');
assert(fs.existsSync('sitemap.xml'), 'sitemap.xml missing');

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
  checkFileContains('js/navigation/navigation.js', 'div id="primary-menu" class="nav-row"');

  // Core scripts should load without throwing
  [
    'js/common/common.js',
    'js/navigation/navigation.js',
    'js/animations/animations.js',
    'js/forms/contact.js',
    'js/portfolio/portfolio.js',
    'js/contributions/contributions.js',
    'js/contributions/carousel.js'
  ].forEach(evalScript);

  // Chatbot demo should tolerate backend startup delays up to ten minutes
  const chatbotHtml = fs.readFileSync('chatbot-demo.html', 'utf8');
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

  console.log('All tests passed.');
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
