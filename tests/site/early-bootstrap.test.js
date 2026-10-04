'use strict';

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { normalizeEarlyBootstrap } = require('../../build/inject-head-metadata');
const { finalizePersonalRouteDocument, validatePersonalRouteDocument } = require('../../build/lib/personal-accordion-shell');

const root = path.resolve(__dirname, '../..');
const bootstrapPath = path.join(root, 'js/common/no-js.js');
const inlinePattern = /<script\b[^>]*data-site-bootstrap="early"[^>]*>([\s\S]*?)<\/script>/;

function fixture(pathname = '/tools/text-compare', lineBreak = '\n') {
  return [
    '<!doctype html><html class="existing-class no-js"><head>',
    `<link rel="canonical" href="https://www.danielshort.me${pathname}">`,
    '<link rel="stylesheet" href="/fixture.css">',
    '<script src="/js/common/no-js.js?old=1" defer></script>',
    '<script data-site-bootstrap="early">window.staleBootstrap = true;</script>',
    '</head><body data-site-route-id="fixture" data-site-route-category="tools" data-site-route-view="detail">',
    '<header data-site-shell-header>Header</header>',
    '<main id="main" data-site-route-content><h1>Fixture content</h1></main>',
    '<span data-site-route-progress></span><span data-site-route-announcer></span>',
    '<footer data-site-shell-footer>Footer</footer>',
    '<script src="js/common/no-js.js"></script>',
    '</body></html>'
  ].join(lineBreak);
}

function executeBootstrap(source, url, canonical) {
  const replacements = [];
  const classes = new Set(['no-js']);
  const rootElement = {
    dataset: {},
    setAttribute() {},
    classList: {
      add: (value) => classes.add(value),
      remove: (value) => classes.delete(value)
    }
  };
  const location = new URL(url);
  location.replace = (target) => replacements.push(target);
  const window = { location, localStorage: { getItem: () => 'saved-consent' }, sessionStorage: { removeItem() {} } };
  window.top = window.self = window;
  const canonicalElement = { href: canonical, getAttribute: (name) => name === 'href' ? canonical : null };
  const document = {
    documentElement: rootElement,
    querySelector: (selector) => selector.includes('canonical') ? canonicalElement : null
  };
  vm.runInNewContext(source, { window, document, URL, URLSearchParams });
  return { replacements, classes };
}

function runEarlyBootstrapTests({ assert }) {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };
  const reject = (html, message) => {
    let error;
    try { validatePersonalRouteDocument(html); } catch (caught) { error = caught; }
    check(error && /modified early bootstrap|unclassified inline executable script/.test(error.message), message);
  };

  for (const lineBreak of ['\n', '\r\n']) {
    for (const pathname of ['/tools/text-compare', '/tools/job-application-tracker']) {
      const normalized = normalizeEarlyBootstrap(fixture(pathname, lineBreak));
      check(normalizeEarlyBootstrap(normalized) === normalized,
        `${pathname} bootstrap normalization must be byte-idempotent for ${JSON.stringify(lineBreak)} input.`);
      check(normalizeEarlyBootstrap(normalizeEarlyBootstrap(normalized)) === normalized,
        `${pathname} repeated builds must not accumulate whitespace or bootstrap tags.`);
      const tags = normalized.match(/<script\b[^>]*>[\s\S]*?<\/script>/g) || [];
      check(tags.length === 1 && normalized.indexOf(tags[0]) < normalized.indexOf('<link rel="stylesheet"'),
        `${pathname} must retain one synchronous bootstrap before its first stylesheet.`);
      check(normalized.includes('<main id="main" data-site-route-content><h1>Fixture content</h1></main>') &&
        normalized.includes('class="existing-class no-js"'), 'Bootstrap normalization must preserve route content and existing root classes.');
      const strict = pathname.endsWith('job-application-tracker');
      check(strict ? tags[0] === '<script src="js/common/no-js.js"></script>' : inlinePattern.test(tags[0]),
        `${pathname} must select the CSP-compatible bootstrap transport.`);
      check(!normalized.includes('staleBootstrap'), 'Generated bootstrap content must replace stale marked inline code.');
    }
  }

  const withoutStyles = fixture().replace('<link rel="stylesheet" href="/fixture.css">', '');
  const normalizedWithoutStyles = normalizeEarlyBootstrap(withoutStyles);
  check(normalizeEarlyBootstrap(normalizedWithoutStyles) === normalizedWithoutStyles,
    'Documents without stylesheets must also normalize without repeated growth.');

  const normalized = normalizeEarlyBootstrap(fixture());
  const finalized = finalizePersonalRouteDocument(normalized, { module: 'fixture' });
  const manifest = validatePersonalRouteDocument(finalized);
  check(manifest.module === 'fixture' && manifest.scripts.length === 0,
    'The document bootstrap must validate while remaining outside the soft-route execution plan.');
  const content = inlinePattern.exec(finalized)?.[1];
  check(Boolean(content), 'The normal route fixture must contain the authoritative inline bootstrap.');
  reject(finalized.replace(content, `${content}\nwindow.injectedBootstrap = true;`),
    'A bootstrap marker must not allow additional executable code.');
  reject(finalized.replace('data-site-bootstrap="early"', 'data-site-bootstrap="other"'),
    'Changing the bootstrap marker must not permit unclassified executable code.');
  reject(finalized.replace('data-site-bootstrap="early"', ''),
    'Removing the bootstrap marker must fail closed.');

  const originalRead = fs.readFileSync;
  const originalSource = originalRead(bootstrapPath, 'utf8');
  const errorDocument = normalizeEarlyBootstrap(originalRead(path.join(root, '404.html'), 'utf8'));
  const errorCanonical = /<link\b[^>]*rel="canonical"[^>]*href="([^"]+)"/i.exec(errorDocument)?.[1];
  const errorBootstrap = inlinePattern.exec(errorDocument)?.[1];
  check(errorCanonical === 'https://www.danielshort.me/404.html' && Boolean(errorBootstrap),
    'The regression must use the actual custom 404 document and its fixed error canonical.');
  for (const requested of [
    '/tools/text-compare?input=a%20b%2Bc%26d&tag=one&tag=two#main',
    '/pages/text-compare.html?input=left%2Bright&tag=one&tag=two#main',
    '/portfolio?project=website&tag=a%20b#main',
    '/genuinely-missing-route?input=a%2Bb#missing'
  ]) {
    const result = executeBootstrap(errorBootstrap, `https://danielshort3.github.io${requested}`, errorCanonical);
    check(result.replacements.length === 1 && result.replacements[0] ===
      `https://www.danielshort.me${requested.replace(/\.html(?=[?#]|$)/, '')}`,
    'GitHub custom 404 startup must forward the requested route and raw query/hash rather than its error canonical.');
  }
  check(executeBootstrap(errorBootstrap, 'https://www.danielshort.me/genuinely-missing-route?x=1#missing', errorCanonical).replacements.length === 0,
    'The custom 404 on the canonical host must preserve its real missing-route location and HTTP error response.');
  for (const url of [
    'https://www.danielshort.me/?audience=analytics&mode=professional#projects',
    'https://www.danielshort.me/contact?audience=tourism&mode=work#contact-modal',
    'https://www.danielshort.me/portfolio/digitGenerator?audience=data-science#demo',
    'https://preview.vercel.app/?mode=professional',
    'https://danielshort3.github.io.example.test/?mode=professional'
  ]) {
    const result = executeBootstrap(originalSource, url, 'https://www.danielshort.me/');
    check(result.replacements.length === 0 && result.classes.has('js') && !result.classes.has('no-js'),
      'Canonical and preview hosts must initialize root state without audience/mode navigation.');
  }
  const legacyCases = [
    { path: '/index.html', canonical: '/', query: '?audience=analytics&mode=professional', hash: '#projects' },
    { path: '/tools/text-compare.html', canonical: '/tools/text-compare', query: '?input=a%2Bb%26c&tag=one&tag=two', hash: '#main' },
    { path: '/pages/text-compare.html', canonical: '/tools/text-compare', query: '?input=a%2Bb%26c&tag=one&tag=two', hash: '#main' },
    { path: '/pages/ocean-wave-simulation.html', canonical: '/games/ocean-wave-simulation', query: '?quality=low', hash: '#controls' },
    { path: '/pages/demos/digit-generator-demo.html', canonical: '/digit-generator-demo', query: '?digit=7&seed=42', hash: '#demo' }
  ];
  for (const { path: pathname, canonical, query, hash } of legacyCases) {
    const result = executeBootstrap(originalSource, `https://danielshort3.github.io${pathname}${query}${hash}`, `https://www.danielshort.me${canonical}`);
    check(result.replacements.length === 1 && result.replacements[0] === `https://www.danielshort.me${canonical}${query}${hash}`,
      `${pathname} on the legacy host must reach its authoritative canonical route with encoded queries, repeated values and fragments intact.`);
  }
  for (const canonical of [undefined, 'https://foreign.example/tools/text-compare']) {
    const result = executeBootstrap(originalSource,
      'https://danielshort3.github.io/pages/text-compare.html?input=a%2Bb#main', canonical);
    check(result.replacements[0] === 'https://www.danielshort.me/pages/text-compare?input=a%2Bb#main',
      'Missing or foreign canonical metadata must preserve the legacy alias for Vercel without changing the destination host.');
  }
  const professionalLegacyCases = [
    {
      incoming: '#contact-modal',
      canonical: '/contact?audience=analytics',
      expected: '/contact?audience=analytics#contact-modal',
      message: 'Legacy professional documents must retain their canonical audience when the request supplies no query.'
    },
    {
      incoming: '?audience=tourism&input=hello%20world&tag=one&tag=two#contact-modal',
      canonical: '/contact?audience=analytics',
      expected: '/contact?audience=tourism&input=hello%20world&tag=one&tag=two#contact-modal',
      message: 'Incoming audience selections must override canonical defaults without rewriting the existing query.'
    },
    {
      incoming: '?input=a%2Bb%26c&tag=one&tag=two&space=hello%20world#contact-modal',
      canonical: '/contact?audience=data-science',
      expected: '/contact?input=a%2Bb%26c&tag=one&tag=two&space=hello%20world&audience=data-science#contact-modal',
      message: 'Appending a missing canonical audience must preserve encoded separators, spaces, repeated values and the fragment.'
    }
  ];
  for (const { incoming, canonical, expected, message } of professionalLegacyCases) {
    const result = executeBootstrap(originalSource,
      `https://danielshort3.github.io/pages/professional/analytics/contact.html${incoming}`, `https://www.danielshort.me${canonical}`);
    check(result.replacements.length === 1 && result.replacements[0] === `https://www.danielshort.me${expected}`, message);
  }
  const refreshedSource = `${originalSource.trim()}\n// authoritative refresh fixture </script>\n`;
  try {
    // Inject a source revision without modifying the shared checkout.
    fs.readFileSync = function readRefreshed(file, ...options) {
      return path.resolve(String(file)) === bootstrapPath ? refreshedSource : originalRead.call(this, file, ...options);
    };
    const refreshed = normalizeEarlyBootstrap(finalized);
    check(refreshed.includes('authoritative refresh fixture <\\/script>') &&
      (refreshed.match(/data-site-bootstrap="early"/g) || []).length === 1,
    'A later build must reread the current source and escape script terminators without adding another tag.');
    check(validatePersonalRouteDocument(refreshed).id === manifest.id,
      'Refreshed authoritative bootstrap content must pass route validation.');
    reject(finalized, 'Stale inline bootstrap content must not validate against a changed authoritative source.');
    check(normalizeEarlyBootstrap(refreshed) === refreshed,
      'Refreshed source content must remain idempotent across subsequent builds.');
  } finally {
    fs.readFileSync = originalRead;
  }
  check(validatePersonalRouteDocument(finalized).id === manifest.id,
    'The source-revision test must restore the real filesystem reader.');

  const vercel = JSON.parse(originalRead(path.join(root, 'vercel.json'), 'utf8'));
  const trackerCsp = vercel.headers.find((rule) => rule.source === '/tools/job-application-tracker')
    ?.headers.find((header) => header.key.toLowerCase() === 'content-security-policy')?.value || '';
  check(trackerCsp.includes("script-src 'self'") && !/script-src[^;]*'unsafe-inline'/.test(trackerCsp),
    'The Tracker document must retain the strict policy that requires its external bootstrap.');
  return checks;
}

module.exports = runEarlyBootstrapTests;
if (require.main === module) {
  const checks = runEarlyBootstrapTests({ assert: require('node:assert/strict') });
  console.log(`Early document bootstrap: ${checks} checks passed.`);
}
