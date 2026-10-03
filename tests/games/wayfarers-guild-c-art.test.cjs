'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const Content = require('../../js/games/wayfarers-guild/progression-content.js');
const Skills = require('../../js/games/wayfarers-guild/area-skills-content.js');
const root = path.resolve(__dirname, '../../img/wayfarers-guild');

test('each of the 90 skills has distinct transparent art and each area has a production scene', async () => {
  const manifest = require('../../img/wayfarers-guild/c-art.json');
  const expected = Content.AREAS.flatMap(area => area.tracks.map(track => 'track-' + area.id + '-' + track.id))
    .concat(Skills.SKILLS.map(skill => 'skill-' + skill.id), Content.AREAS.map(area => 'scene-' + area.id));
  assert.deepEqual(manifest.entries.map(entry => entry.id).sort(), expected.sort());
  assert.equal(new Set(manifest.entries.map(entry => entry.sha256)).size, 96);
  for (const entry of manifest.entries) {
    const bytes = fs.readFileSync(path.join(root, entry.file));
    assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), entry.sha256);
    if (entry.id.startsWith('scene-')) continue;
    const metadata = await sharp(bytes).metadata(), stats = await sharp(bytes).stats();
    assert.equal(metadata.width, 128);
    assert.equal(metadata.height, 128);
    assert.equal(metadata.hasAlpha, true);
    assert.equal(stats.isOpaque, false);
  }
});
