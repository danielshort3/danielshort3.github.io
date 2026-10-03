'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const sharp = require('sharp');
const D = require('../../js/games/wayfarers-guild/collection-content.js');
const Icons = require('../../js/games/wayfarers-guild/icons.js');
const { bundle, MODULES } = require('../../build/bundle-wayfarers-android.cjs');
const root = path.resolve(__dirname, '../..');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

test('every authored card and gear item has its own transparent, shipped, verified artwork', async () => {
  const definitions = [...D.CARDS, ...D.GEAR];
  const manifest = require('../../img/wayfarers-guild/collection-art.json');
  assert.equal(manifest.assets.length, definitions.length);
  assert.equal(new Set(definitions.map(def => def.artId)).size, definitions.length);
  assert.deepEqual(new Set(Icons.COLLECTION_ART), new Set(definitions.map(def => def.artId)));
  const output = fs.mkdtempSync(path.join(os.tmpdir(), 'guild-collection-art-'));
  try {
    const records = bundle(output);
    const hashes = new Set();
    for (const def of definitions) {
      const record = manifest.assets.find(item => item.id === def.id);
      assert.equal(record.kind, def.slot ? 'equipment' : 'card');
      assert.equal(record.file, def.artId + '.png');
      const bytes = fs.readFileSync(path.join(root, 'img/wayfarers-guild', record.file));
      assert.equal(hash(bytes), record.sha256);
      assert.ok(!hashes.has(record.sha256), 'No item may reuse another item image');
      hashes.add(record.sha256);
      const { data, info } = await sharp(bytes).raw().toBuffer({ resolveWithObject: true });
      assert.equal(info.width, 96); assert.equal(info.height, 96); assert.equal(info.channels, 4);
      const alpha = [...data].filter((value, index) => index % 4 === 3);
      assert.ok(alpha.some(value => value === 0) && alpha.some(value => value > 200), def.id + ' retains visible art and transparency');
      const markup = Icons.markup(def.artId, { label: def.name });
      assert.ok(markup.includes(record.file) && markup.includes('wg-collection-art'), def.id + ' has no generic icon fallback');
      assert.ok(markup.includes('aria-label='));
      const shipped = records.find(item => item.path === 'img/wayfarers-guild/' + record.file);
      assert.equal(shipped.sha256, record.sha256);
      assert.equal(hash(fs.readFileSync(path.join(output, shipped.path))), record.sha256);
    }
    assert.ok(MODULES.indexOf('collection-content') < MODULES.indexOf('collections'));
    assert.ok(MODULES.indexOf('collections') < MODULES.indexOf('expeditions'));
    assert.ok(MODULES.indexOf('collections') < MODULES.indexOf('progression'));
  } finally {
    assert.equal(path.dirname(path.resolve(output)), path.resolve(os.tmpdir()));
    assert.ok(path.basename(output).startsWith('guild-collection-art-'));
    fs.rmSync(output, { recursive: true, force: true });
  }
});
