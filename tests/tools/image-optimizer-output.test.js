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
${block('  const isWebpBlob =', '  const buildOutputName =')}
${block('  const describeSizeChange =', '  const renderOutputs =')}
globalThis.api = { describeSizeChange, getSizeFeedback, chooseOutput, isWebpBlob };`, context);
const { describeSizeChange, getSizeFeedback, chooseOutput, isWebpBlob } = context.api;

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

const items = [{ id: 1, file: { size: 100 } }];
const outputs = [{ inputId: 1, blob: { size: 100 }, mime: 'image/png', originalKept: false }];
const feedbackArgs = {
  items,
  outputs,
  responsiveEnabled: false,
  supportsWebp: true,
  requestedMime: 'keep'
};
assert.equal(getSizeFeedback(feedbackArgs).canTryWebp, true);
assert.equal(getSizeFeedback(feedbackArgs).metadataKept, 0);
assert.equal(getSizeFeedback({ ...feedbackArgs, outputs: [{ ...outputs[0], blob: { size: 150 } }] }).outputBytes, 150);
assert.equal(getSizeFeedback({ ...feedbackArgs, outputs: [{ ...outputs[0], blob: { size: 75 } }] }), null);
assert.equal(getSizeFeedback({ ...feedbackArgs, supportsWebp: false }).canTryWebp, false);
assert.equal(getSizeFeedback({ ...feedbackArgs, requestedMime: 'image/webp' }).canTryWebp, false);
assert.equal(getSizeFeedback({ ...feedbackArgs, outputs: [{ ...outputs[0], mime: 'image/webp' }] }).canTryWebp, false);
assert.equal(getSizeFeedback({ ...feedbackArgs, outputs: [{ ...outputs[0], originalKept: true }] }).metadataKept, 1);
assert.equal(getSizeFeedback({ ...feedbackArgs, responsiveEnabled: true }), null);
assert.equal(getSizeFeedback({ ...feedbackArgs, outputs: [{ ...outputs[0], inputId: 2 }] }), null);
const batch = {
  ...feedbackArgs,
  items: [{ id: 1, file: { size: 100 } }, { id: 2, file: { size: 200 } }],
  outputs: [outputs[0], { inputId: 2, blob: { size: 200 }, mime: 'image/png', originalKept: true }]
};
assert.equal(getSizeFeedback(batch).batch, true);
assert.equal(getSizeFeedback(batch).inputBytes, 300);
assert.equal(getSizeFeedback(batch).metadataKept, 1);
assert.equal(getSizeFeedback({ ...batch, outputs: [outputs[0], { ...outputs[0] }] }), null);

(async () => {
  const riffWebp = Uint8Array.from([82, 73, 70, 70, 0, 0, 0, 0, 87, 69, 66, 80]);
  assert.equal(await isWebpBlob(new Blob([riffWebp], { type: 'image/webp' })), true);
  assert.equal(await isWebpBlob(new Blob([riffWebp], { type: 'image/png' })), false);
  assert.equal(await isWebpBlob(new Blob(['not WebP'], { type: 'image/webp' })), false);
  console.log('Image optimizer output tests passed: metadata opt-in, size feedback, batch eligibility, and actual WebP encoding.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
