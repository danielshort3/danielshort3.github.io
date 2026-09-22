'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ROOT = path.resolve(__dirname, '../..');
const ICON = 'img/project-starfall/items/icons/admin-worldwright-console.png';

function loadHelpers() {
  const sandbox = vm.createContext({ console, Image: class TestImage {} });
  for (const file of ['core/assets.js', 'engine/assets.js', 'ui/item-assets.js']) {
    const filename = path.join(ROOT, 'js/games/project-starfall', file);
    vm.runInContext(fs.readFileSync(filename, 'utf8'), sandbox, { filename });
  }
  return sandbox;
}

function createRuntime(data) {
  return {
    data, assets: {}, failedAssets: {}, requests: [], pending: new Map(), refreshes: 0,
    scheduleAssetRefresh() { this.refreshes += 1; },
    ensureAssetReady(assetPath) {
      this.requests.push(assetPath);
      this.assets[assetPath] = { complete: false, naturalWidth: 0 };
      return new Promise(resolve => this.pending.set(assetPath, resolve));
    },
    settle(assetPath, loaded = true) {
      const image = this.assets[assetPath];
      image.complete = true;
      image.naturalWidth = loaded ? 64 : 0;
      if (!loaded) this.failedAssets[assetPath] = true;
      this.scheduleAssetRefresh();
      this.pending.get(assetPath)({ path: assetPath, loaded, failed: !loaded });
      this.pending.delete(assetPath);
    }
  };
}

async function unitTests() {
  const sandbox = loadHelpers();
  const core = sandbox.ProjectStarfallCore;
  const helper = sandbox.ProjectStarfallEngineModules.assets;
  const itemUi = sandbox.ProjectStarfallUiModules.itemAssets;
  const data = {
    ITEM_ASSETS: Object.freeze({ admin_worldwright_console: ICON }),
    CARD_ASSETS: Object.freeze({ card: 'img/project-starfall/cards/icons/test-card.png' }),
    MENU_ICON_ASSETS: Object.freeze({ inventory: 'img/project-starfall/ui/menu-icons/inventory.png' }),
    SKILLS: [{ id: 'skill', iconAsset: 'img/project-starfall/skills/base/test-skill.png' }]
  };
  const runtime = createRuntime(data);
  const savedItem = Object.freeze({ id: 'admin_worldwright_console', kind: 'consumable', quantity: 1 });
  const before = JSON.stringify(savedItem);
  const draws = [];
  const ctx = { drawImage(...args) { draws.push(args); } };
  const options = {
    data, frame: false, showAura: false,
    getAsset: assetPath => helper.getResolvedAssetImage(runtime, assetPath),
    drawAssetFrame: core.drawAssetFrame
  };
  for (let index = 0; index < 10; index += 1) {
    assert.equal(itemUi.drawCanvasItemIcon(ctx, savedItem, 0, 0, 64, options), false);
  }
  assert.deepEqual(runtime.requests, [ICON], 'cold inventory/tooltip icons must request the assigned image once');
  assert.equal(draws.length, 0, 'an incomplete image must not be passed to drawImage');
  runtime.settle(ICON);
  assert.equal(itemUi.drawCanvasItemIcon(ctx, savedItem, 0, 0, 64, options), true);
  assert.equal(draws[0][0], runtime.assets[ICON]);
  assert.deepEqual(draws[0].slice(1), [0, 0, 64, 64]);
  assert.equal(JSON.stringify(savedItem), before, 'loading artwork must not alter an inventory item');
  assert.equal(runtime.requests.length, 1);

  for (const iconPath of [data.CARD_ASSETS.card, data.MENU_ICON_ASSETS.inventory, data.SKILLS[0].iconAsset]) {
    assert.equal(helper.getResolvedAssetImage(runtime, iconPath), null);
    assert.equal(helper.getResolvedAssetImage(runtime, iconPath), null);
    runtime.settle(iconPath);
    assert.equal(helper.getResolvedAssetImage(runtime, iconPath), runtime.assets[iconPath]);
  }
  assert.equal(runtime.requests.length, 4, 'cards, menus and skill icons use the same demand-loading path');
  for (const assetPath of ['', 'img/project-starfall/maps/unloaded.webp', 'img/project-starfall/characters/unloaded.png', 'img/project-starfall/items/source/old.png']) {
    assert.equal(helper.getResolvedAssetImage(runtime, assetPath), null);
  }
  assert.equal(runtime.requests.length, 4, 'do not turn a cache lookup into an eager download of large non-icon artwork');

  const failed = createRuntime(data);
  helper.getResolvedAssetImage(failed, ICON);
  failed.settle(ICON, false);
  for (let index = 0; index < 50; index += 1) helper.getResolvedAssetImage(failed, ICON);
  assert.equal(failed.requests.length, 1, 'failed icons must not retry on every frame');

  const backup = 'img/project-starfall/items/sheets/test-backup.png';
  const fallback = createRuntime(Object.assign({}, data, { ASSET_BACKUP_PATHS: { [ICON]: backup } }));
  helper.getResolvedAssetImage(fallback, ICON);
  fallback.settle(ICON, false);
  helper.getResolvedAssetImage(fallback, ICON);
  helper.getResolvedAssetImage(fallback, ICON);
  assert.deepEqual(fallback.requests, [ICON, backup]);
  fallback.settle(backup);
  assert.equal(helper.getResolvedAssetImage(fallback, ICON), fallback.assets[backup]);

  const rejected = createRuntime(data);
  rejected.ensureAssetReady = function rejectRequest(assetPath) {
    this.requests.push(assetPath);
    return Promise.reject(new Error('test request rejection'));
  };
  helper.getResolvedAssetImage(rejected, ICON);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(rejected.failedAssets[ICON], true);
  helper.getResolvedAssetImage(rejected, ICON);
  assert.equal(rejected.requests.length, 1);
  const thrown = createRuntime(data);
  thrown.ensureAssetReady = () => { throw new Error('test synchronous failure'); };
  assert.doesNotThrow(() => helper.getResolvedAssetImage(thrown, ICON));
  assert.equal(thrown.failedAssets[ICON], true);

  const frameRuntime = createRuntime({ ITEM_ASSETS: { sprite: `${ICON}#frame=0,0,32,32,64,64` } });
  helper.getResolvedAssetImage(frameRuntime, `${ICON}#frame=0,0,32,32,64,64`);
  helper.getResolvedAssetImage(frameRuntime, ICON);
  assert.deepEqual(frameRuntime.requests, [ICON], 'frame fragments must not create distinct network/cache identities');
  frameRuntime.settle(ICON);

  const nextIcon = 'img/project-starfall/items/icons/next-icon.png';
  runtime.data.ITEM_ASSETS = Object.freeze({ next: nextIcon });
  helper.getResolvedAssetImage(runtime, nextIcon);
  assert.equal(runtime.requests.at(-1), nextIcon, 'replacing a catalog must invalidate the icon-path cache');
  runtime.settle(nextIcon);
  const withoutData = createRuntime(undefined);
  helper.getResolvedAssetImage(withoutData, nextIcon, runtime.data);
  assert.deepEqual(withoutData.requests, [nextIcon], 'engine fallback data must be supported');
  withoutData.settle(nextIcon);
  delete sandbox.Image;
  const headless = createRuntime(runtime.data);
  assert.equal(helper.getResolvedAssetImage(headless, nextIcon), null);
  assert.equal(headless.requests.length, 0, 'Node-only consumers must not start fake image loads');
  console.log('Starfall icon-loading unit checks passed: cold draw, deduplication, redraw, all icon families, failures, backups, frame keys, catalog refresh, headless mode and unchanged item data.');
}

async function repositoryTests() {
  const Data = require('../../js/games/project-starfall/project-starfall-data.js');
  const core = require('../../js/games/project-starfall/core/assets.js');
  const itemUi = require('../../js/games/project-starfall/ui/item-assets.js');
  const { ProjectStarfallEngine } = require('../../js/games/project-starfall/project-starfall-engine.js');
  const sharp = require('sharp');
  const items = Object.entries(Data.ITEM_ASSETS);
  assert(items.length > 0);
  assert.equal(new Set(items.map(([, assetPath]) => assetPath)).size, items.length, 'individual item IDs must not collide on an output filename');
  for (const [id, assetPath] of items) {
    assert.equal(itemUi.getItemAsset({ id }, { data: Data }), assetPath);
    assert.equal(assetPath, `img/project-starfall/items/icons/${id.replace(/_/g, '-').toLowerCase()}.png`);
  }
  assert.equal(Data.ITEM_ASSETS.admin_worldwright_console, ICON);
  const assignments = [
    ...items.map(([id, assetPath]) => ({ id, assetPath, family: 'item' })),
    ...Object.entries(Data.CARD_ASSETS || {}).map(([id, assetPath]) => ({ id, assetPath, family: 'card' })),
    ...Object.entries(Data.MENU_ICON_ASSETS || {}).map(([id, assetPath]) => ({ id, assetPath, family: 'menu' })),
    ...(Data.SKILLS || []).filter(skill => skill.iconAsset).map(skill => ({ id: skill.id, assetPath: skill.iconAsset, family: 'skill' }))
  ];
  const files = new Map();
  for (const entry of assignments) files.set(core.getAssetSourcePath(entry.assetPath), entry.family);
  for (const [assetPath, family] of files) {
    const filename = path.join(ROOT, assetPath);
    assert(fs.existsSync(filename), `missing assigned ${family} icon: ${assetPath}`);
    const { data: pixels, info } = await sharp(filename).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    if (family === 'item') {
      assert.equal(info.width, 64, `item width: ${assetPath}`);
      assert.equal(info.height, 64, `item height: ${assetPath}`);
    }
    let visible = 0;
    for (let index = info.channels - 1; index < pixels.length; index += info.channels) {
      if (pixels[index] >= 128) visible += 1;
    }
    assert(visible >= 32, `assigned icon has no usable visible artwork: ${assetPath}`);
  }
  for (const item of [...(Data.SHOP_ITEMS || []), ...(Data.RANDOM_EQUIPMENT_ITEMS || []), ...(Data.BOSS_EQUIPMENT_ITEMS || []), ...(Data.MATERIAL_ITEMS || [])]) {
    const assetPath = itemUi.getItemAsset(item, { data: Data });
    assert(files.has(core.getAssetSourcePath(assetPath)), `catalog item is not assigned to a known icon: ${item.id}`);
  }

  const oldImage = global.Image;
  const requests = [];
  try {
    global.Image = class TestImage {
      constructor() { this.complete = false; this.naturalWidth = 0; this.listeners = {}; }
      addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
      set src(value) {
        this.url = value;
        requests.push(value);
        queueMicrotask(() => {
          this.complete = true;
          this.naturalWidth = 64;
          for (const listener of this.listeners.load || []) listener();
        });
      }
      get src() { return this.url; }
      decode() { return Promise.resolve(); }
    };
    const engine = Object.create(ProjectStarfallEngine.prototype);
    Object.assign(engine, {
      data: Data, assets: {}, failedAssets: {},
      recordAssetLoadResult() {}, scheduleAssetRefresh() {}
    });
    for (const assetPath of files.keys()) {
      assert.equal(engine.getAsset(assetPath), null);
      assert.equal(engine.getAsset(assetPath), null);
    }
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(requests.length, files.size, 'the real engine must request each icon once without a map preload');
    for (const assetPath of files.keys()) {
      assert.equal(engine.getAsset(assetPath), engine.assets[assetPath]);
      assert.equal(engine.assets[assetPath].src, core.getAssetRequestUrl(assetPath));
    }
    const draws = [];
    assert.equal(itemUi.drawCanvasItemIcon({ drawImage: (...args) => draws.push(args) },
      { id: 'admin_worldwright_console', kind: 'consumable' }, 0, 0, 64,
      { data: Data, frame: false, showAura: false, getAsset: assetPath => engine.getAsset(assetPath), drawAssetFrame: core.drawAssetFrame }), true);
    assert.equal(draws[0][0], engine.assets[ICON]);
  } finally {
    if (oldImage === undefined) delete global.Image;
    else global.Image = oldImage;
  }
  console.log(`Starfall assignment and engine checks passed: ${items.length} item mappings; ${assignments.length} icon assignments; ${files.size} decoded, visibly nonempty icon files; real-engine lazy loading and canvas draw.`);
}

(async () => {
  await unitTests();
  if (!process.argv.includes('--unit')) await repositoryTests();
})().catch(error => { console.error(error); process.exitCode = 1; });
