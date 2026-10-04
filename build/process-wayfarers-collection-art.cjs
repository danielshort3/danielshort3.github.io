'use strict';

// Pack independently generated portraits at their shipped resolution. The art
// stays untouched apart from nearest-neighbor sizing; retain source alpha.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const root = path.resolve(__dirname, '..');
const content = require('../js/games/wayfarers-guild/collection-content.js');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function processArt(sourceManifest) {
  const sources = JSON.parse(fs.readFileSync(sourceManifest, 'utf8'));
  const definitions = [...content.CARDS, ...content.GEAR];
  const records = [];
  for (const definition of definitions) {
    const source = sources.find(item => item.id === definition.artId);
    if (!source || !fs.existsSync(source.path)) throw new Error('Missing generated art: ' + definition.id);
    const input = fs.readFileSync(source.path);
    const metadata = await sharp(input).metadata();
    if (!metadata.hasAlpha) throw new Error('Transparent source required: ' + definition.id);
    const file = definition.artId + '.png';
    const bytes = await sharp(input).resize(96, 96, { fit: 'contain', kernel: 'nearest', background: '#00000000' }).png({ compressionLevel: 9 }).toBuffer();
    fs.writeFileSync(path.join(root, 'img/wayfarers-guild', file), bytes);
    records.push({ id: definition.id, name: definition.name, kind: definition.slot ? 'equipment' : 'card', file, width: 96, height: 96, sha256: hash(bytes), sourceSha256: hash(input), sourceFile: path.basename(source.path), prompt: source.prompt });
  }
  const manifest = { version: 1, generator: 'OpenAI built-in ImageGen', processing: 'Nearest-neighbor 96px sizing; generated alpha preserved. Card frames and labels are semantic HTML.', assets: records };
  fs.writeFileSync(path.join(root, 'img/wayfarers-guild/collection-art.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log('Saved ' + records.length + ' unique collection assets (' + records.reduce((sum, item) => sum + fs.statSync(path.join(root, 'img/wayfarers-guild', item.file)).size, 0) + ' bytes).');
}

if (require.main === module) processArt(process.argv[2]).catch(error => { console.error(error); process.exitCode = 1; });
module.exports = { processArt };
