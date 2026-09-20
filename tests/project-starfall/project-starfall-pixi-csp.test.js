'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '../..');
const read = (relativePath) => fs.readFileSync(path.join(root, relativePath), 'utf8');
const context = vm.createContext({ console, setTimeout, clearTimeout, navigator: { userAgent: 'node' }, performance }, {
  codeGeneration: { strings: false, wasm: false }
});

vm.runInContext(read('js/vendor/pixi.min.js'), context);
assert.strictEqual(context.PIXI.VERSION, '8.18.1', 'the CSP interpreter must match the vendored Pixi version');
assert.throws(() => context.PIXI.AbstractRenderer.prototype._unsafeEvalCheck(), /does not allow unsafe-eval/,
  'the fixture must reproduce the original strict-CSP startup failure');
vm.runInContext(read('js/vendor/pixi-unsafe-eval.min.js'), context);
assert.doesNotThrow(() => context.PIXI.AbstractRenderer.prototype._unsafeEvalCheck());

const group = new context.PIXI.UniformGroup({ uAlpha: { value: 0.25, type: 'f32' } });
const uniforms = { uAlpha: { location: 'alpha-location', value: 0 } };
const uploads = [];
const synchronize = context.PIXI.GlUniformGroupSystem.prototype._generateUniformsSync(group, uniforms);
synchronize(uniforms, { uAlpha: 0.25 }, { gl: { uniform1f: (...args) => uploads.push(args) } });
assert.deepStrictEqual(uploads, [['alpha-location', 0.25]],
  'shader uniforms must upload through the interpreter with string code generation disabled');

const rendererModule = require(path.join(root, 'js/games/project-starfall/project-starfall-renderer-pixi.js'));
const renderer = rendererModule.createRenderer({ PIXI: {} });
renderer.app = {
  get canvas() { throw new Error('uninitialized Application.canvas getter'); },
  destroy() { throw new Error('uninitialized application must not be destroyed through Pixi'); }
};
assert.doesNotThrow(() => renderer.setActive(false), 'fallback must preserve the original init error');
assert.doesNotThrow(() => renderer.syncCanvasStyle());
assert.doesNotThrow(() => renderer.destroy());
assert.strictEqual(renderer.active, false);

const injector = require(path.join(root, 'build/inject-script-bundles.js'));
const minimalPage = '<html><head></head><body>\n<script defer src="js/vendor/pixi.min.js?v=20260619"></script>\n</body></html>';
const first = injector.processHtml(minimalPage, 'pages/games/project-starfall.html').html;
const second = injector.processHtml(first, 'pages/games/project-starfall.html').html;
const scripts = [...second.matchAll(/<script\b[^>]*src="([^"]+)"/g)].map((match) => match[1]);
const pixiIndex = scripts.findIndex((src) => src.startsWith('js/vendor/pixi.min.js'));
assert.strictEqual(scripts[pixiIndex + 1], 'js/vendor/pixi-unsafe-eval.min.js?v=8.18.1');
assert.match(scripts[pixiIndex + 2], /^dist\/project-starfall/);
assert.strictEqual(scripts.filter((src) => src.includes('pixi-unsafe-eval.min.js')).length, 1,
  'repeated builds must keep exactly one interpreter before game startup');

console.log('Project Starfall Pixi strict-CSP and failed-init recovery tests passed.');
