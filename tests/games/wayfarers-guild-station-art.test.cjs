'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const content = require('../../js/games/wayfarers-guild/station-content');
const art = require('../../js/games/wayfarers-guild/station-art');
const root = path.resolve(__dirname, '../..');
const manifest = require('../../img/wayfarers-guild/station-art.json');
const sources = require('../../asset-sources/wayfarers-guild/stations/sources.json');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

test('all 180 local illustrations have stable semantic IDs and distinct export bytes', () => {
  assert.equal(content.SKILLS.length, 180);
  assert.equal(new Set(content.SKILLS.map(skill => skill.icon)).size, 180);
  assert.equal(Object.keys(art.icons).length, 180);
  const hashes = new Set();
  for (const skill of content.SKILLS) {
    const record = manifest.records.find(row => row.file === art.icons[skill.icon]);
    assert(record, skill.id);
    assert.equal(record.width, 128);
    assert.equal(record.height, 128);
    assert(sources.icons[skill.areaId].names.includes(skill.name));
    hashes.add(record.sha256);
  }
  assert.equal(hashes.size, 180, 'Each upgrade has a different illustration');
});

test('six area kits provide separately animated backgrounds, machines, additions and workers', async () => {
  assert.equal(Object.keys(art.areas).length, 6);
  for (const area of content.AREAS) {
    const kit = art.areas[area.id];
    assert(kit.sky && kit.underground);
    const stations = content.STATIONS.filter(row => row.areaId === area.id);
    assert.equal(stations.length, 5);
    for (const station of stations) {
      const picture = kit.stations[station.id];
      assert(picture && picture.background && picture.machine && picture.addition, station.id);
      for (const kind of ['machine', 'addition']) {
        const meta = await sharp(path.join(root, 'img/wayfarers-guild', picture[kind])).metadata();
        const stats = await sharp(path.join(root, 'img/wayfarers-guild', picture[kind])).stats();
        assert(meta.hasAlpha, station.id + ' ' + kind);
        assert.equal(stats.channels.at(-1).min, 0, 'Transparent surroundings');
      }
    }
  }
  const worker = await sharp(path.join(root, 'img/wayfarers-guild', art.workers.file)).metadata();
  assert.equal(worker.width, 4 * 64);
  assert.equal(worker.height, 6 * 96);
  assert(worker.hasAlpha);
});

test('art hashes and flat published paths satisfy the signed snapshot limits', () => {
  const names = new Set();
  let total = 0;
  for (const record of manifest.records) {
    assert.match(record.file, /^[a-z0-9][a-z0-9-]*\.webp$/);
    assert(!names.has(record.file));
    names.add(record.file);
    const bytes = fs.readFileSync(path.join(root, 'img/wayfarers-guild', record.file));
    assert.equal(bytes.length, record.bytes);
    assert.equal(hash(bytes), record.sha256);
    assert(bytes.length < 8 * 1024 * 1024);
    total += bytes.length;
  }
  assert(total < 16 * 1024 * 1024, 'Station art leaves room for the full snapshot');
  assert.equal(art.logicalWidth, 384);
  assert.equal(art.stationHeight, 208);
  assert.equal(art.surfaceHeight, 320);
});
