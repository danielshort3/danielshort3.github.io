'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const source = fs.readFileSync(path.join(__dirname, '../../js/tools/image-optimizer.js'), 'utf8');
const block = (start, end) => {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert(from >= 0 && to > from, `Missing image optimizer behavior: ${start}`);
  return source.slice(from, to);
};
const context = vm.createContext({});
vm.runInContext(`${block('  const formatBytes =', '  const describeSelectionLimits =')}
${block('  const describeSizeChange =', '  const renderOutputs =')}
globalThis.api = { describeSizeChange, chooseOutput };`, context);
const { describeSizeChange, chooseOutput } = context.api;

const original = { size: 100, type: 'image/png' };
const encoded = { size: 250, type: 'image/png' };
const unchanged = { original, encoded, mime: 'image/png', sourceWidth: 100, sourceHeight: 80, width: 100, height: 80 };

assert.equal(chooseOutput(unchanged).blob, encoded, 'Default must re-encode, honoring metadata removal.');
assert.equal(chooseOutput({ ...unchanged, keepSmaller: true }).blob, original, 'Explicit opt-in may keep a smaller original.');
assert.equal(chooseOutput({ ...unchanged, keepSmaller: true }).originalKept, true);
for (const changes of [
  { width: 50 },
  { height: 40 },
  { mime: 'image/webp', encoded: { size: 250, type: 'image/webp' } },
  { changesBackground: true },
  { encoded: { size: 90, type: 'image/png' } },
  { encoded: { size: 100, type: 'image/png' } }
]) {
  const result = chooseOutput({ ...unchanged, keepSmaller: true, ...changes });
  assert.equal(result.originalKept, false, `Must honor requested conversion or smaller encoding: ${JSON.stringify(changes)}`);
  assert.equal(result.blob, changes.encoded || encoded);
}

assert.match(describeSizeChange(100, 250), /150 B larger \(150%\) than original/);
assert.match(describeSizeChange(100, 75), /25\.0 B smaller \(25%\) than original/);
assert.equal(describeSizeChange(100, 100), 'Same file size as original');
assert.match(describeSizeChange(100000, 100001), /larger \(<0\.1%\)/, 'Small increases must not be rounded to zero.');
assert.match(describeSizeChange(100, 300, 'the originals'), /larger \(200%\) than the originals/);

console.log('Image optimizer output tests passed: metadata opt-in, dimensions, format, background, and honest size changes.');
