'use strict';

const assert = require('assert');
const childProcess = require('child_process');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
require('./project-starfall-pixi-csp.test.js');
require('./project-starfall-pixi-viewport-clip.test.js');
require('./project-starfall-start-data.test.js');

const root = path.resolve(__dirname, '..', '..');
const read = (relativePath) => fs.readFileSync(path.join(root, relativePath), 'utf8');

const entry = read('build/entries/project-starfall.entry.js');
const buildScript = read('build/build-js.js');
const injector = read('build/inject-script-bundles.js');
const page = read('pages/games/project-starfall.html');
const scriptPaths = [...page.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*>/g)]
  .map((match) => new URL(match[1], 'https://example.test/').pathname);

[
  'project-starfall-data.js',
  'project-starfall-rig.js',
  'project-starfall-renderer-pixi.js',
  'project-starfall-engine.js',
  'project-starfall-ui.js',
  'project-starfall-main.js'
].forEach((fileName) => {
  assert(entry.includes(fileName), `Starfall bundle entry should include ${fileName}`);
});

assert(buildScript.includes("baseName: 'project-starfall'") && buildScript.includes("manifestKey: 'projectStarfall'"),
  'shared JS build should publish the Project Starfall bundle');
assert(injector.includes("projectStarfall: resolveHref('project-starfall.js', manifest.projectStarfall)"),
  'script injector should resolve the hashed Project Starfall bundle');
assert(scriptPaths.filter((scriptPath) => scriptPath === '/js/vendor/pixi.min.js').length === 1,
  'Project Starfall page should load Pixi exactly once');
assert(scriptPaths.filter((scriptPath) => /^\/dist\/project-starfall(?:\.[0-9a-f]{8})?\.js$/.test(scriptPath)).length === 1,
  'Project Starfall page should load one production game bundle');
assert(!scriptPaths.some((scriptPath) => scriptPath.startsWith('/js/games/project-starfall/')),
  'Project Starfall page should not ship the former per-module script waterfall');
assert(!entry.includes("import '../../js/games/project-starfall/data/enemy-hurtboxes.js'"),
  'exact collision data belongs to the Start-triggered chunk');
assert(!scriptPaths.some((scriptPath) => /project-starfall-hurtboxes/.test(scriptPath)),
  'collision data must not be fetched before Start through a script tag');

childProcess.execFileSync(process.execPath, ['build/build-js.js'], {
  cwd: root,
  stdio: 'pipe',
  maxBuffer: 4 * 1024 * 1024
});

const manifest = JSON.parse(read('dist/scripts-manifest.json'));
assert(manifest.projectStarfall, 'scripts manifest should include the Project Starfall bundle');
const bundlePath = path.join(root, 'dist', manifest.projectStarfall);
assert(fs.existsSync(bundlePath), 'hashed Project Starfall bundle should exist after the JS build');
const bundle = fs.readFileSync(bundlePath);
const gzipBytes = zlib.gzipSync(bundle, { level: 9 }).length;
assert(manifest.projectStarfallHurtboxes, 'the scripts manifest must publish the collision chunk');
assert(page.includes(`data-starfall-hurtboxes-src="dist/${manifest.projectStarfallHurtboxes}"`),
  'the game must reference the current content-hashed collision chunk');
const collisionBundle = fs.readFileSync(path.join(root, 'dist', manifest.projectStarfallHurtboxes));
const collisionGzipBytes = zlib.gzipSync(collisionBundle, { level: 9 }).length;
assert(!bundle.includes(Buffer.from('starfall-enemy-hurtboxes-v1')) && collisionBundle.includes(Buffer.from('starfall-enemy-hurtboxes-v1')),
  'the exact table must exist only in the deferred chunk');
assert(bundle.length + collisionBundle.length < 5 * 1024 * 1024,
  `Combined Starfall scripts should stay below 5 MiB raw (received ${bundle.length + collisionBundle.length} bytes)`);
assert(gzipBytes < 1.2 * 1024 * 1024, `Initial Starfall bundle should stay below 1.2 MiB gzip (received ${gzipBytes} bytes)`);
// Preserve the previous complete-game transfer ceiling across both chunks.
assert(gzipBytes + collisionGzipBytes < 1.5 * 1024 * 1024,
  `Combined Starfall scripts should stay below 1.5 MiB gzip (received ${gzipBytes + collisionGzipBytes} bytes)`);

console.log(`Project Starfall bundle tests passed (${gzipBytes} initial gzip bytes, ${collisionGzipBytes} Start-triggered gzip bytes).`);
