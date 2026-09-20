#!/usr/bin/env node
'use strict';

// Run generate-project-starfall-overhaul-review.js first to refresh runtime clips.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'output/starfall-alignment-review');
const read = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const clone = (value) => JSON.parse(JSON.stringify(value));
const review = read(path.join(ROOT, 'output/starfall-overhaul-review/review-data.json'));
const ids = ['lava-tick', 'clockbug', 'cinder-spitter', 'briar-stag', 'cracked-mimic', 'bandit-cutter', 'dew-slime', 'index-scribe', 'brambleking', 'clockwork-titan', 'rimewarden', 'eclipse-sovereign'];
fs.mkdirSync(path.join(OUT, 'before'), { recursive: true });
const baselinePath = path.join(OUT, 'before-baseline.json');
const baseline = fs.existsSync(baselinePath) ? read(baselinePath) : {};

const actors = ids.map((id) => {
  const actor = review.actors.find((entry) => entry.after.image.split('?')[0].endsWith(`/${id}-sheet.png`) && entry.name !== 'Fault Skitter');
  if (!actor) throw new Error(`Missing runtime review clip: ${id}`);
  const backupDirectory = id === 'lava-tick' ? 'output/lava-tick-alignment' : `output/enemy-idle-alignment/${id}`;
  const backup = path.join(ROOT, backupDirectory, 'before-sheet.png');
  const beforeHash = hash(backup);
  if (baseline[id] && baseline[id] !== beforeHash) throw new Error(`Previous preview backup changed: ${id}`);
  baseline[id] = beforeHash;
  fs.copyFileSync(backup, path.join(OUT, 'before', `${id}-sheet.png`));
  const result = clone(actor);
  if (id !== 'lava-tick') result.after.actions = { idle: result.after.actions.idle };
  result.before = clone(result.after);
  result.before.image = `before/${id}-sheet.png?v=${beforeHash.slice(0, 12)}`;
  result.reviewScope = id === 'lava-tick' ? 'All eight rows: horizontal registration' : 'Idle only: horizontal registration';
  result.beforeSheetSha256 = beforeHash;
  const livePath = path.resolve(OUT, result.after.image.split('?')[0]);
  const afterHash = hash(livePath);
  if (!result.after.image.endsWith(afterHash.slice(0, 12))) throw new Error(`Refresh the general review data before generating ${id}`);
  result.afterSheetSha256 = afterHash;
  return result;
});

const data = {
  actors,
  backgrounds: review.backgrounds,
  counts: { playerActions: 0, enemyIds: actors.length, scenery: 0, icons: 0 },
  scenerySheets: [], iconSheets: [],
  storageKey: 'starfall.alignment.review.v1', defaultAction: 'idle', showAnchor: true
};
let template = fs.readFileSync(path.join(ROOT, 'build/templates/starfall-overhaul-review.template.html'), 'utf8');
template = template.replaceAll('Starfall art review', 'Starfall alignment review')
  .replace('Compare the original drawings with the updated game assets.', 'The previous preview beside the corrected body alignment. Same drawings, scale, and timing.')
  .replace('<nav aria-label="Review sections"><a href="#animation">Animation</a><a href="#scenery">Scenery</a><a href="#icons">Icons</a></nav>', '<nav aria-label="Related review"><a href="../starfall-overhaul-review/index.html">Full art collection</a></nav>')
  .replace('<h2>Before</h2>', '<h2>Previous preview</h2>')
  .replace('<h2>After</h2>', '<h2>Alignment corrected</h2>')
  .replace('<section class="inventory">', '<section class="inventory" hidden>')
  .replace('Original animation images are verified against the saved asset baseline. This page uses the current game animation definitions and can be rebuilt as the assets change.', 'Lava Tick: all eight action rows. The other eleven creatures: idle only. Guides stay fixed; vertical motion and limb articulation are preserved. This pass corrects horizontal registration; remaining drawing changes and other action rows still need review.')
  .replace('Before uses original timing.', 'Both sides use identical timing.')
  .replaceAll('Actions repeat at their configured timing. Frame controls follow the updated animation. Character art is shown without equipment or combat effects.', 'Both sides use identical timing and a fixed origin. Frame-step to inspect the body and loop seam. Character art is shown without combat effects.');
fs.writeFileSync(baselinePath, JSON.stringify(baseline, null, 2) + '\n');
fs.writeFileSync(path.join(OUT, 'review-data.json'), JSON.stringify(data, null, 2) + '\n');
fs.writeFileSync(path.join(OUT, 'index.html'), template.replace('__REVIEW_DATA__', JSON.stringify(data).replace(/</g, '\\u003c')));
console.log(`Generated alignment comparison for ${actors.length} identities: Lava Tick all rows, eleven idle loops. Previous illustrated sheets preserved and hash checked.`);
