'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const config = require('../../vercel.json');
const { compileRoutes, applyResponseHeaders } = require('../../build/dev');
const { processHtml } = require('../../build/inject-head-metadata');
const routeStyles = require('../../build/route-component-styles.json');

const rules = compileRoutes(config.headers);
function cacheControl(pathname) {
  const headers = new Map();
  applyResponseHeaders(pathname, rules, { headers: {} }, {
    setHeader: (name, value) => headers.set(name.toLowerCase(), value)
  }, new URL(pathname, 'http://127.0.0.1'));
  return headers.get('cache-control') || '';
}

for (const asset of [
  '/dist/styles.0123abcd.css',
  '/dist/styles-personal-accordion.0123abcd.css',
  '/dist/site-shell.0123abcd.js',
  '/dist/site-tools-account.0123abcd.js',
  '/dist/project-starfall.0123abcd.js'
]) {
  assert.equal(cacheControl(asset), 'public, max-age=31536000, immutable', asset);
}

for (const asset of [
  '/dist/styles.css',
  '/dist/styles-home.css',
  '/dist/site-shell.js',
  '/dist/site-home.js',
  '/dist/project-starfall.js',
  '/js/tools/utm-batch-builder.js',
  '/js/tools/utm-batch-builder.worker.js'
]) {
  assert.equal(cacheControl(asset), 'public, max-age=0, must-revalidate', asset);
}

for (const asset of [
  '/dist/site-shell.0123abc.js',
  '/dist/site-shell.notahash.js',
  '/dist/site-shell.0123abcdXjs',
  '/dist/nested/site-shell.0123abcd.js',
  '/js/tools/text-compare.js',
  '/css/fonts/Inter-Latin.woff2'
]) {
  assert(!cacheControl(asset).includes('immutable'), `${asset} is not a content-hashed bundle`);
}

assert.equal(cacheControl('/css/privacy.css'), 'public, max-age=300, must-revalidate');
assert.equal(cacheControl('/dist/scripts-manifest.json'), 'public, max-age=300, stale-while-revalidate=60');

// Browser Cache Storage ignores HTTP revalidation directives. Route CSS must
// therefore get a new URL when its bytes change, including on repeated builds.
for (const [route, styles] of Object.entries(routeStyles)) {
  for (const stylesheet of styles) {
    const pathname = route.replace('*', 'website');
    const hash = crypto.createHash('sha256')
      .update(fs.readFileSync(path.join(__dirname, '../..', stylesheet))).digest('hex').slice(0, 12);
    const expectedHref = `${stylesheet}?v=${hash}`;
    const fixture = `<!doctype html><html><head><title>Cache fixture</title>
      <link rel="canonical" href="https://www.danielshort.me${pathname}">
      <link rel="stylesheet" href="${stylesheet}">
      <link rel="stylesheet" href="/${stylesheet}?v=outdated">
      </head><body></body></html>`;
    const first = processHtml(fixture, 'pages/cache-fixture.html').html;
    const second = processHtml(first, 'pages/cache-fixture.html').html;
    const links = (html) => [...html.matchAll(/<link\b[^>]*\bhref="([^"]+)"[^>]*>/g)]
      .map((match) => match[1]).filter((href) => href.replace(/^\//, '').split('?')[0] === stylesheet);
    assert.deepEqual(links(first), [expectedHref], `${route} must replace stale/bare CSS URLs and avoid duplicates`);
    assert.deepEqual(links(second), [expectedHref], `${route} must preserve the same content version on rebuild`);
  }
}
const projectHtml = fs.readFileSync(path.join(__dirname, '../../pages/portfolio/website.html'), 'utf8');
const manifest = JSON.parse(projectHtml.match(/<script\b[^>]*id="site-route-manifest"[^>]*>([\s\S]*?)<\/script>/)[1]);
const projectStyle = projectHtml.match(/<link\b[^>]*href="(css\/components\/project-page\.css\?v=[a-f0-9]{12})"/)[1];
assert(manifest.styles.includes(`/${projectStyle}`), 'Soft navigation must request the same versioned project CSS as a full page load');

// Browser favicon caches can outlive a deployment. Fresh and stale icon links
// must both resolve to the current artwork without accumulating query strings.
for (const icon of ['favicon.ico', 'img/brand/05-ds-favicon-small-icon.svg', 'img/ui/logo-16.png', 'img/ui/logo-180.png']) {
  const hash = crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(__dirname, '../..', icon))).digest('hex').slice(0, 12);
  const rel = icon.endsWith('180.png') ? 'apple-touch-icon' : 'icon';
  for (const href of [icon, `/${icon}?v=old`]) {
    const fixture = `<html><head><title>Icon</title><link rel="${rel}" href="${href}"></head><body></body></html>`;
    const first = processHtml(fixture, 'pages/icon-fixture.html').html;
    const second = processHtml(first, 'pages/icon-fixture.html').html;
    assert(first.includes(`rel="${rel}" href="${icon}?v=${hash}"`), `${icon} must reflect current artwork`);
    assert.equal(second, first, `${icon} must remain stable on a repeated build`);
  }
}
console.log('Asset cache policy tests passed.');
