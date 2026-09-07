'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { processHtml: injectHead } = require('../../build/inject-head-metadata');
const { processHtml: injectScripts } = require('../../build/inject-script-bundles');
const { validatePersonalRouteDocument } = require('../../build/lib/personal-accordion-shell');

const root = path.resolve(__dirname, '../..');
const registrationPath = '/js/common/service-worker-register.js';
const registrationSource = fs.readFileSync(path.join(root, registrationPath), 'utf8');
const vercel = JSON.parse(fs.readFileSync(path.join(root, 'vercel.json'), 'utf8'));
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const attribute = (tag, name) => new RegExp(`\\b${name}=["']([^"']*)["']`, 'i').exec(tag)?.[1] || '';
const scriptTags = (html) => [...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)];
const executable = (tag) => {
  const type = attribute(tag[1], 'type').trim().toLowerCase();
  return !type || type === 'module' || /^(?:text|application)\/(?:java|ecma)script$/.test(type);
};
const registrations = (html) => scriptTags(html).filter((tag) => attribute(tag[1], 'id') === 'ds-sw-register');
const directive = (policy, name) => policy.split(';').map((part) => part.trim()).find((part) => part.startsWith(`${name} `))?.slice(name.length + 1) || '';

function assertRegistration(html, label) {
  const tags = registrations(html);
  assert.equal(tags.length, 1, `${label} must have exactly one service worker registration`);
  assert.equal(attribute(tags[0][1], 'src'), registrationPath, `${label} must load registration from its own origin`);
  assert.match(tags[0][1], /\bdefer\b/, `${label} must not block HTML parsing`);
  assert.equal(tags[0][2].trim(), '', `${label} must not embed executable registration code`);
}

function testMetadataGeneration() {
  const fresh = '<!DOCTYPE html><html><head><title>Fixture</title><link rel="manifest" href="/manifest.json"></head><body></body></html>';
  const legacy = '<script id="ds-sw-register">navigator.serviceWorker.register("/sw.js");</script>';
  const oldExternal = '<script src="/old-register.js" id="ds-sw-register"></script>';
  for (const fixture of [fresh, fresh.replace('</head>', `${legacy}</head>`), fresh.replace('</head>', `${legacy}${oldExternal}</head>`)]) {
    const result = injectHead(fixture, 'pages/fixture.html').html;
    assertRegistration(result, 'Metadata output');
    assertRegistration(injectHead(result, 'pages/fixture.html').html, 'Repeated metadata output');
    assert(!result.includes('/old-register.js'));
  }

  for (const tool of ['text-compare', 'job-application-tracker']) {
    const file = `pages/${tool}.html`;
    const legacyPage = read(file).replace(/<script\b[^>]*\bid="ds-sw-register"[^>]*>[\s\S]*?<\/script>/i, legacy);
    const output = injectScripts(injectHead(legacyPage, file).html, file).html;
    assertRegistration(output, `${tool} final build output`);
    const manifest = validatePersonalRouteDocument(output);
    assert.equal(manifest.scripts.filter((src) => src === registrationPath).length, 1, 'Route manifests must classify the shared external script once');
  }

  const softPage = injectScripts(injectHead(read('pages/text-compare.html'), 'pages/text-compare.html').html, 'pages/text-compare.html').html;
  assert.throws(() => validatePersonalRouteDocument(softPage.replace('</head>', `${legacy}</head>`)), /inline executable script/, 'A legacy registration id must not bypass inline-script validation');
}

function testToolPolicies() {
  const toolPages = [...new Set(vercel.rewrites
    .filter((rule) => /^\/tools\//.test(rule.source) && !rule.source.includes(':'))
    .map((rule) => /^\/pages\/([^/?]+)/.exec(rule.destination)?.[1]?.replace(/\.html$/, ''))
    .filter(Boolean))];
  assert(toolPages.length >= 16, 'The policy audit must cover public and internal tool pages');
  for (const name of toolPages) {
    const file = `pages/${name}.html`;
    const html = read(file);
    assertRegistration(html, file);
    for (const tag of scriptTags(html).filter(executable)) {
      assert(attribute(tag[1], 'src'), `${file} must keep executable scripts external`);
      assert(!/^https?:\/\//i.test(attribute(tag[1], 'src')), `${file} must keep initial tool scripts on the same origin`);
    }
    assert.doesNotMatch(html, /<style\b/i, `${file} must not depend on an inline style block`);
    assert.doesNotMatch(html, /<[^>]+\son[a-z]+\s*=/i, `${file} must not use event-handler attributes blocked by script-src-attr`);
    if (process.argv.includes('--public')) {
      const output = read(`public/${file}`);
      assertRegistration(output, `public/${file}`);
      for (const tag of scriptTags(output).filter(executable)) {
        assert(attribute(tag[1], 'src'), `public/${file} must not regain inline executable scripts during the build`);
      }
    }
  }
  if (process.argv.includes('--public')) {
    assert.equal(read(`public${registrationPath}`), registrationSource, 'The deployed registration asset must match its source');
  }

  const trackerRoutes = ['/tools/job-application-tracker', '/tools/job-application-tracker.html', '/pages/job-application-tracker', '/pages/job-application-tracker.html'];
  for (const route of trackerRoutes) {
    const policy = vercel.headers.find((rule) => rule.source === route)?.headers.find((header) => header.key === 'Content-Security-Policy')?.value || '';
    assert.equal(directive(policy, 'script-src'), "'self'", `${route} must retain strict script-src`);
    assert.equal(directive(policy, 'script-src-elem'), "'self'", `${route} must retain strict script-src-elem`);
    assert.equal(directive(policy, 'script-src-attr'), "'none'", `${route} must reject inline event handlers`);
    assert.equal(directive(policy, 'style-src-elem'), "'self'", `${route} must retain strict stylesheet sources`);
  }

  const authConfig = read('js/accounts/tools-config.js');
  const cognitoOrigin = 'https://job-tracker-auth-886623862678.auth.us-east-2.amazoncognito.com';
  assert(authConfig.includes(new URL(cognitoOrigin).hostname), 'The policy origin must match the shared Tools authentication configuration');
  const backgroundHeaders = vercel.headers.find((rule) => rule.source === '/tools/background-remover')?.headers;
  for (const route of ['/tools/background-remover', '/tools/background-remover.html', '/pages/background-remover', '/pages/background-remover.html']) {
    const headers = vercel.headers.find((rule) => rule.source === route)?.headers;
    assert.deepEqual(headers, backgroundHeaders, `${route} must preserve the same capture and security policy as every Background Remover alias`);
    const policy = headers?.find((header) => header.key === 'Content-Security-Policy')?.value || '';
    const connect = directive(policy, 'connect-src').split(/\s+/);
    assert(connect.includes(cognitoOrigin), `${route} must allow the shared sign-in token exchange`);
    assert(connect.includes("'self'"), `${route} must preserve same-origin API access`);
    assert(!connect.includes('*'), `${route} must not broaden network access to all hosts`);
  }
  return toolPages.length;
}

async function testRegistrationRuntime() {
  function run(options = {}) {
    const registrations = [];
    const listeners = [];
    const navigator = options.supported === false ? {} : {
      serviceWorker: {
        register(url) {
          registrations.push(url);
          return options.reject ? Promise.reject(new Error('Registration unavailable')) : Promise.resolve({});
        }
      }
    };
    vm.runInNewContext(registrationSource, {
      URLSearchParams,
      navigator,
      document: { readyState: options.readyState || 'loading' },
      window: {
        isSecureContext: options.secure !== false,
        location: { hostname: options.host ?? 'www.danielshort.me', search: options.search || '' },
        addEventListener: (type, callback, listenerOptions) => listeners.push({ type, callback, options: listenerOptions })
      }
    });
    return { registrations, listeners };
  }

  const pending = run();
  assert.deepEqual(pending.registrations, []);
  assert.equal(pending.listeners.length, 1);
  assert.equal(pending.listeners[0].type, 'load');
  assert.equal(pending.listeners[0].options.once, true);
  pending.listeners[0].callback();
  assert.deepEqual(pending.registrations, ['/sw.js']);
  assert.deepEqual(run({ readyState: 'complete' }).registrations, ['/sw.js'], 'A late-loaded external script must still register');

  for (const options of [
    { secure: false }, { supported: false }, { search: '?no_sw' }, { search: '?no_sw=0' },
    ...['', 'localhost', 'app.localhost', '127.0.0.1', '192.168.1.10', '[::1]', '::1'].map((host) => ({ host }))
  ]) {
    const skipped = run(options);
    assert.equal(skipped.registrations.length, 0, 'Development, disabled and unsupported contexts must not register');
    assert.equal(skipped.listeners.length, 0, 'Skipped contexts must not schedule registration');
  }
  run({ readyState: 'complete', reject: true });
  await new Promise((resolve) => setImmediate(resolve));
}

(async () => {
  testMetadataGeneration();
  const count = testToolPolicies();
  await testRegistrationRuntime();
  console.log(`Service worker registration: external metadata, route manifests, strict tracker policy, shared sign-in access, ${count} tool pages and runtime guards passed${process.argv.includes('--public') ? ' (including public output)' : ''}.`);
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
