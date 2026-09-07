'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { once } = require('events');
const { createLocalServer, resolveCurrentStylesheetFile } = require('../../build/dev');

async function run() {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'site-stale-stylesheets-'));
  let server;
  try {
    const deploy = path.join(temporary, 'deploy');
    const source = path.join(temporary, 'source');
    [deploy, source].forEach((directory) => fs.mkdirSync(path.join(directory, 'dist'), { recursive: true }));
    const writeBuild = (directory, filename, contents) => {
      fs.writeFileSync(path.join(directory, 'dist', filename), contents);
      fs.writeFileSync(path.join(directory, 'dist', 'styles-manifest.json'), JSON.stringify({ file: filename }));
    };
    writeBuild(deploy, 'styles.11111111.css', 'body { color: navy; }');
    writeBuild(source, 'styles.22222222.css', 'body { color: blue; }');
    const stale = '/dist/styles.6ec3f3dd.css';
    assert.strictEqual(resolveCurrentStylesheetFile(stale, [deploy, source]), path.join(deploy, 'dist', 'styles.11111111.css'));
    writeBuild(deploy, 'styles.33333333.css', 'body { color: teal; }');
    assert.strictEqual(resolveCurrentStylesheetFile(stale, [deploy, source]), path.join(deploy, 'dist', 'styles.33333333.css'),
      'An already running resolver must follow the next build without caching its manifest');
    fs.writeFileSync(path.join(deploy, 'dist', 'styles-manifest.json'), '{');
    assert.strictEqual(resolveCurrentStylesheetFile(stale, [deploy, source]), path.join(source, 'dist', 'styles.22222222.css'),
      'A deploy copy in progress should fall back to the ready source bundle');
    for (const request of ['/not-a-real-page', '/dist/unknown.12345678.css', '/dist/styles.bad.css', '/dist/styles.12345678.js', '/dist/../styles.12345678.css']) {
      assert.strictEqual(resolveCurrentStylesheetFile(request, [source]), null, 'Only recognized CSS bundle families may fall back');
    }
    fs.writeFileSync(path.join(deploy, 'dist', 'styles-manifest.json'), JSON.stringify({ file: '../styles.33333333.css' }));
    assert.strictEqual(resolveCurrentStylesheetFile(stale, [deploy]), null, 'Manifest paths cannot escape the dist directory');

    server = createLocalServer({ envDir: temporary });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const origin = `http://127.0.0.1:${server.address().port}`;
    const repository = path.resolve(__dirname, '../..');
    const manifest = JSON.parse(fs.readFileSync(path.join(repository, 'public/dist/styles-manifest.json'), 'utf8'));
    for (const [request, file] of [['/dist/styles.6ec3f3dd.css', manifest.file], ['/dist/styles-home.7f761444.css', manifest.homeFile]]) {
      const response = await fetch(origin + request);
      assert.strictEqual(response.status, 200, 'The attached stale URL should serve its current stylesheet');
      assert.match(response.headers.get('content-type'), /^text\/css;/);
      assert.strictEqual(response.headers.get('cache-control'), 'no-store');
      assert.strictEqual(await response.text(), fs.readFileSync(path.join(repository, 'public/dist', file), 'utf8'));
      const head = await fetch(origin + request, { method: 'HEAD' });
      assert.strictEqual(head.status, 200);
      assert.match(head.headers.get('content-type'), /^text\/css;/);
      assert.strictEqual(await head.text(), '');
    }
    const missing = await fetch(origin + '/not-a-real-page-stale-css-test');
    assert.strictEqual(missing.status, 404, 'A missing page must never turn into an index fallback');
    const unknown = await fetch(origin + '/dist/unknown.12345678.css');
    assert.strictEqual(unknown.status, 404, 'Unknown stylesheet names must remain missing');
    const sheet = await fetch(origin + '/portfolio/sheetMusicUpscale');
    assert.strictEqual(sheet.status, 200);
    assert.match(sheet.headers.get('content-type'), /^text\/html;/);
    assert.strictEqual(sheet.headers.get('cache-control'), 'no-store');
    assert.match(await sheet.text(), /data-project-image-comparison/);
    console.log('Local stale stylesheet tests passed: CSS MIME, current bytes, build refresh, fallback, HEAD, and real route.');
  } finally {
    if (server) await new Promise((resolve) => server.close(resolve));
    const temporaryRoot = path.resolve(os.tmpdir()) + path.sep;
    assert(path.resolve(temporary).startsWith(temporaryRoot) && path.basename(temporary).startsWith('site-stale-stylesheets-'));
    fs.rmSync(temporary, { recursive: true, force: true });
  }
}

run().catch((error) => { console.error(error); process.exitCode = 1; });
