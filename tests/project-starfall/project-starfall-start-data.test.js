'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const rootDir = path.resolve(__dirname, '../..');
const bootSource = fs.readFileSync(path.join(rootDir, 'js/games/project-starfall/project-starfall-main.js'), 'utf8');

function fixture(options = {}) {
  const scripts = [];
  const timers = new Map();
  let nextTimer = 0;
  const button = { disabled: false };
  const start = { setAttribute(name, value) { this[name] = value; }, querySelectorAll: () => [button] };
  const loader = { setAttribute(name, value) { this[name] = value; } };
  const stageClasses = new Set();
  const stage = { classList: { add: name => stageClasses.add(name) } };
  const ui = {
    elements: { stage, loader }, opens: 0, messages: [],
    openCharacterSelect() { this.opens += 1; return true; },
    init() {}, completeInitialLoad() {},
    setLoaderProgress(progress, status) { this.status = status; },
    updateLoadProgress() { stageClasses.delete('is-loading'); },
    showToast(message) { this.messages.push(message); }
  };
  const root = {
    isConnected: true,
    getAttribute: () => options.raw ? null : '/dist/project-starfall-hurtboxes.12345678.js',
    querySelector: () => start
  };
  const context = vm.createContext({
    URL, console,
    setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    document: {
      baseURI: 'https://example.test/', readyState: 'complete',
      querySelector: () => root, getElementById: () => ({}),
      createElement: () => ({ remove() { this.removed = true; } }),
      head: { append: script => scripts.push(script) }
    },
    createProjectStarfallEngine: () => ({}), createProjectStarfallUi: () => ui
  });
  vm.runInContext(bootSource, context);
  return { context, ui, root, scripts, timers, button, start, loader, stageClasses };
}

async function main() {
  const f = fixture();
  assert.equal(f.scripts.length, 0, 'the start screen must not request combat data');
  const first = f.ui.openCharacterSelect();
  assert.strictEqual(f.ui.openCharacterSelect(), first, 'repeated Start actions share one pending request');
  assert.equal(f.scripts.length, 1);
  assert.equal(f.ui.opens, 0, 'character selection cannot open before combat data is ready');
  assert.equal(f.button.disabled, true);
  assert.equal(f.start['aria-busy'], 'true');
  assert.equal(f.loader['aria-hidden'], 'false');
  assert.equal(f.ui.status, 'Preparing game');
  assert.equal(f.scripts[0].src, 'https://example.test/dist/project-starfall-hurtboxes.12345678.js');
  f.scripts[0].onerror();
  assert.equal(await first, false);
  assert.equal(f.ui.opens, 0);
  assert.equal(f.button.disabled, false);
  assert.match(f.ui.messages[0], /Start to retry/);
  assert.equal(f.timers.size, 0);

  const retry = f.ui.openCharacterSelect();
  assert.equal(f.scripts.length, 2, 'a failed request is retried only after another Start');
  f.context.ProjectStarfallEnemyHurtboxesData = { sheets: {}, masks: [] };
  f.scripts[1].onload();
  assert.equal(await retry, true);
  assert.equal(f.ui.opens, 1);
  assert.equal(f.button.disabled, false);
  assert.equal(f.start['aria-busy'], 'false');
  assert.equal(f.loader['aria-hidden'], 'true');
  assert.equal(f.ui.openCharacterSelect(), true);
  assert.equal(f.scripts.length, 2, 'reopening selection reuses the loaded table');

  const timeout = fixture();
  const timed = timeout.ui.openCharacterSelect();
  [...timeout.timers.values()][0]();
  assert.equal(await timed, false);
  assert.equal(timeout.ui.opens, 0);
  assert.equal(timeout.button.disabled, false);
  const missing = timeout.ui.openCharacterSelect();
  timeout.scripts[1].onload();
  assert.equal(await missing, false, 'a successful script response without data cannot start combat');

  const detached = fixture();
  const leaving = detached.ui.openCharacterSelect();
  detached.root.isConnected = false;
  detached.context.ProjectStarfallEnemyHurtboxesData = {};
  detached.scripts[0].onload();
  assert.equal(await leaving, false, 'late completion must not reopen a departed game');
  assert.equal(detached.ui.opens, 0);

  const raw = fixture({ raw: true });
  const direct = raw.ui.openCharacterSelect();
  assert.equal(raw.scripts[0].src, 'https://example.test/js/games/project-starfall/data/enemy-hurtboxes.js');
  raw.context.ProjectStarfallEnemyHurtboxesData = {};
  raw.scripts[0].onload();
  assert.equal(await direct, true, 'unbundled source pages retain a working Start flow');
  console.log('Starfall Start data gate passed: deferred request, single flight, retry, timeout, accessibility and navigation cleanup.');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
