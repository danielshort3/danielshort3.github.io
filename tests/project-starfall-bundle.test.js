'use strict';

const assert = require('assert');
const childProcess = require('child_process');
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const root = path.resolve(__dirname, '..');
const read = (relativePath) => fs.readFileSync(path.join(root, relativePath), 'utf8');

const entry = read('build/entries/project-starfall.entry.js');
const buildScript = read('build/build-js.js');
const injector = read('build/inject-script-bundles.js');
const page = read('pages/games/project-starfall.html');

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
assert((page.match(/js\/vendor\/pixi\.min\.js/g) || []).length === 1,
  'Project Starfall page should load Pixi exactly once');
assert((page.match(/dist\/project-starfall(?:\.[0-9a-f]{8})?\.js/g) || []).length === 1,
  'Project Starfall page should load one production game bundle');
assert(!/src="js\/games\/project-starfall\//.test(page),
  'Project Starfall page should not ship the former per-module script waterfall');

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
assert(bundle.length < 5 * 1024 * 1024, `Starfall bundle should stay below 5 MiB raw (received ${bundle.length} bytes)`);
assert(gzipBytes < 1.25 * 1024 * 1024, `Starfall bundle should stay below 1.25 MiB gzip (received ${gzipBytes} bytes)`);

const runtimeProbe = childProcess.execFileSync(process.execPath, ['-e', `
  global.window = global;
  global.document = { readyState: 'loading', addEventListener() {} };
  require(${JSON.stringify(bundlePath)});
  process.stdout.write(JSON.stringify({
    mapCount: global.ProjectStarfallData && global.ProjectStarfallData.MAPS && global.ProjectStarfallData.MAPS.length,
    mapIds: (global.ProjectStarfallData && global.ProjectStarfallData.MAPS || []).map((map) => map.id),
    hasEngineFactory: typeof global.createProjectStarfallEngine === 'function'
  }));
`], {
  cwd: root,
  stdio: ['ignore', 'pipe', 'pipe'],
  maxBuffer: 4 * 1024 * 1024
}).toString();
const runtime = JSON.parse(runtimeProbe);
assert(runtime.mapCount >= 56 &&
  runtime.mapIds.includes('greenrootMeadow') &&
  runtime.mapIds.includes('ashglassPass'),
  'production bundle should assemble the complete playable map catalog before boot');
assert(runtime.hasEngineFactory,
  'production bundle should expose the engine factory before boot');

console.log(`Project Starfall bundle tests passed (${bundle.length} raw bytes, ${gzipBytes} gzip bytes).`);
