'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { declaredAssets, readAsset, createReport, toMarkdown } = require('../../build/report-performance');

const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'site-performance-'));
try {
  fs.mkdirSync(path.join(fixture, 'assets'));
  fs.mkdirSync(path.join(fixture, 'icons'));
  fs.writeFileSync(path.join(fixture, 'assets/site.css'), 'body { color: navy; }');
  fs.writeFileSync(path.join(fixture, 'assets/site.js'), 'window.siteReady = true;');
  fs.writeFileSync(path.join(fixture, 'icons/example.png'), Buffer.alloc(100));
  fs.writeFileSync(path.join(fixture, 'icons/example.webp'), Buffer.alloc(20));
  const html = `<base href="/"><link href="assets/site.css" rel="stylesheet">
    <link rel="preload" href="assets/later.css"><script data-tools-account-src="assets/later.js"></script>
    <script defer src="assets/site.js?v=1#fragment"></script><script src="assets/site.js?v=1"></script>
    <script src="https://external.example/library.js"></script><!-- <script src="missing.js"></script> -->`;
  fs.writeFileSync(path.join(fixture, 'index.html'), html);
  const references = declaredAssets(html, '/nested/route');
  assert.deepEqual(references.assets.map((asset) => asset.url), ['/assets/site.css', '/assets/site.js?v=1']);
  assert.deepEqual(references.external, ['https://external.example/library.js']);
  assert.deepEqual(declaredAssets("<script src='local.js'></script>", '/nested/page').assets,
    [{ url: '/nested/local.js', file: 'nested/local.js' }], 'relative resources honor the page URL when no base is present');
  const budgets = {
    routes: [{ route: '/', file: 'index.html', category: 'home', maxGzipBytes: 10000 }],
    catalogs: [{ directory: 'icons', maxWebpBytes: 25 }]
  };
  const report = createReport(fixture, budgets);
  assert.equal(report.passed, true);
  assert.equal(report.routes[0].assets.length, 2, 'lazy data-src and preload hints do not become fictitious requests');
  assert.equal(report.routes[0].bytes, Buffer.byteLength(html) + 21 + 24, 'document and each unique declared asset count once');
  assert.equal(report.catalogs[0].savingsPercent, 80);
  assert(toMarkdown(report).includes('not a browser waterfall or Core Web Vitals measurement'));
  const limited = createReport(fixture, { ...budgets, routes: [{ ...budgets.routes[0], maxGzipBytes: 1 }] });
  assert.equal(limited.passed, false, 'real growth above a route limit fails the gate');
  assert.equal(createReport(fixture, { ...budgets, catalogs: [{ directory: 'icons', maxWebpBytes: 1 }] }).passed, false);
  fs.unlinkSync(path.join(fixture, 'icons/example.webp'));
  assert.throws(() => createReport(fixture, budgets), /ENOENT/, 'missing published optimized artwork is a build failure');
  assert.throws(() => readAsset(fixture, '../outside.js'), /Unsafe/, 'budget scans stay within the selected output directory');
  assert.throws(() => readAsset(fixture, 'assets/missing.js'), /ENOENT/, 'missing declared first-party resources fail instead of reducing the total');
} finally {
  const resolved = path.resolve(fixture);
  assert.equal(path.dirname(resolved), path.resolve(os.tmpdir()));
  assert(path.basename(resolved).startsWith('site-performance-'));
  fs.rmSync(resolved, { recursive: true, force: true });
}
console.log('Performance budgets: resource discovery, exclusions, missing assets, and oversize failures passed.');
