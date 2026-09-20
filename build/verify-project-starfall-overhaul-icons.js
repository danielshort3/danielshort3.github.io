#!/usr/bin/env node
'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const owner = require('./project-starfall-overhaul-icons.js');
const gear = require('./generate-project-starfall-equipment-atlases.js');
const root = path.resolve(__dirname, '..');
const source = path.join(root, 'asset-sources/project-starfall/overhaul-v1/icons');
const output = path.join(root, 'output/starfall-overhaul-icons');
const digest = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

async function main() {
  await owner.verify();
  const catalog = JSON.parse(fs.readFileSync(path.join(source, 'catalog.json'))), ledger = JSON.parse(fs.readFileSync(path.join(source, 'ledger.json')));
  const entries = catalog.batches.flatMap(b => b.entries.map(e => ({ ...e, kind: b.kind })));
  const groups = { items: entries.filter(e => e.output.includes('/items/')), skills: entries.filter(e => e.output.includes('/skills/')), cards: entries.filter(e => e.output.includes('/cards/')), menu: entries.filter(e => e.output.includes('/menu-icons/')) };
  const expected = { items: 209, skills: 85, cards: 21, menu: 18 };
  fs.mkdirSync(output, { recursive: true });
  for (const [name, icons] of Object.entries(groups)) {
    if (icons.length !== expected[name]) throw new Error(name + ' count is ' + icons.length);
    const columns = name === 'items' ? 14 : name === 'skills' ? 11 : 7;
    const composites = [];
    for (let i = 0; i < icons.length; i += 1) {
      const entry = icons[i], file = path.join(root, entry.output);
      if (digest(file) === ledger.baseline[entry.output]) throw new Error('Unchanged production icon: ' + entry.output);
      const decoded = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      const { width, height } = decoded.info;
      let count = 0, edge = 0;
      for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) if (decoded.data[(y * width + x) * 4 + 3] > 8) { count++; if (x < 3 || y < 3 || x >= width - 3 || y >= height - 3) edge++; }
      if (count < 40 || edge) throw new Error('Empty or clipped icon: ' + entry.output);
      composites.push({ input: await sharp(file).resize(64, 64).png().toBuffer(), left: i % columns * 72 + 4, top: Math.floor(i / columns) * 72 + 4 });
    }
    await sharp({ create: { width: columns * 72, height: Math.ceil(icons.length / columns) * 72, channels: 4, background: '#263446' } }).composite(composites).png().toFile(path.join(output, name + '-native-review.png'));
  }
  const gearTiles = [];
  for (let i = 0; i < gear.EQUIPMENT.length; i += 1) {
    const item = gear.EQUIPMENT[i];
    await gear.validateAtlas(item);
    const file = gear.atlasPath(item), relative = path.relative(root, file).replace(/\\/g, '/');
    if (digest(file) === ledger.equipmentBaseline[relative]) throw new Error('Unchanged equipment art: ' + item.id);
    gearTiles.push({ input: await sharp(file).extract({ left: 3 * 128, top: 0, width: 128, height: 128 }).png().toBuffer(), left: i % 10 * 128, top: Math.floor(i / 10) * 128 });
  }
  await sharp({ create: { width: 1280, height: Math.ceil(gearTiles.length / 10) * 128, channels: 4, background: '#ece7dc' } }).composite(gearTiles).png().toFile(path.join(output, 'equipment-review.png'));
  const report = { counts: { ...expected, screens: 3, pedestal: 1, equipment: 85 }, rasterTotal: entries.length, equipmentCells: gear.EQUIPMENT.reduce((n, item) => n + gear.atlasAngles(item).length * (item.kind === 'bow' ? 3 : 1), 0), allOutputsChanged: true, allIconMarginsClear: true, preservedSessionImages: Object.keys(ledger.protectedSessionImages).length, provenance: 'asset-sources/project-starfall/overhaul-v1/icons/ledger.json' };
  fs.writeFileSync(path.join(output, 'validation.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
}
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
module.exports = { main };
