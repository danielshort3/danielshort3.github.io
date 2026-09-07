'use strict';

const assert = require('node:assert/strict');
const config = require('../../vercel.json');
const { compileRoutes, applyResponseHeaders } = require('../../build/dev');

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
console.log('Asset cache policy tests passed.');
