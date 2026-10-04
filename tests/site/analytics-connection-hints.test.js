'use strict';

const assert = require('node:assert/strict');
const { processHtml } = require('../../build/inject-head-metadata');

const hints = [
  '<link rel="preconnect" href="https://www.googletagmanager.com" crossorigin>',
  '<link href="//www.google-analytics.com/" rel="dns-prefetch">',
  "<link rel='dns-prefetch' href='https://googletagmanager.com'>",
  '<link REL="preconnect" HREF="https://google-analytics.com">'
];
for (const separator of ['\n', '\r\n', '']) {
  const html = '<!doctype html><html><head>' + hints.join(separator) +
    '<link rel="preconnect" href="https://fonts.example.test">' +
    '<link rel="dns-prefetch" href="https://www.google-analytics.com.example.test">' +
    '<link rel="preload" href="/css/fonts/Inter-Latin.woff2" as="font">' +
    '</head><body><main id="main">Fixture</main></body></html>';
  const generated = processHtml(html, 'hint-fixture.html').html;
  assert(!/<link\b[^>]*(?:googletagmanager\.com["']|google-analytics\.com\/?["'])/i.test(generated),
    'Initial HTML removes old Google connection hints regardless of attribute order or line breaks.');
  assert(generated.includes('href="https://fonts.example.test"'), 'Unrelated connection hints are preserved.');
  assert(generated.includes('href="https://www.google-analytics.com.example.test"'), 'Similar external hostnames are preserved.');
  assert(generated.includes('rel="preload" href="/css/fonts/Inter-Latin.woff2"'), 'First-party font preload is preserved.');
  assert.equal(processHtml(generated, 'hint-fixture.html').html, generated, 'Rebuilding does not reintroduce hints or accumulate changes.');
}
const fresh = processHtml('<html><head>\n</head><body>Fixture</body></html>', 'hint-fixture.html').html;
assert(!fresh.includes('googletagmanager.com') && !fresh.includes('google-analytics.com'),
  'New generated documents do not introduce pre-consent analytics connections.');
console.log('Analytics connection hint generation checks passed (16 assertions).');
