'use strict';

const assert = require('assert');
const {
  clearOutputChromaSpill,
  detectChromaKeyMode,
  removeChromaAndGuides
} = require('../build/process-project-starfall-compact-bandits.js');

function makeRaw(width, height, color) {
  const raw = Buffer.alloc(width * height * 4);
  for (let offset = 0; offset < raw.length; offset += 4) {
    raw[offset] = color[0];
    raw[offset + 1] = color[1];
    raw[offset + 2] = color[2];
    raw[offset + 3] = 255;
  }
  return raw;
}

function setPixel(raw, width, x, y, color) {
  const offset = (y * width + x) * 4;
  raw[offset] = color[0];
  raw[offset + 1] = color[1];
  raw[offset + 2] = color[2];
  raw[offset + 3] = color.length > 3 ? color[3] : 255;
}

function getPixel(raw, width, x, y) {
  const offset = (y * width + x) * 4;
  return Array.from(raw.subarray(offset, offset + 4));
}

function testConnectedKeyRemoval(mode, key, artColor, preservedColor, haloColor) {
  const width = 13;
  const height = 13;
  const raw = makeRaw(width, height, key);
  for (let y = 3; y <= 9; y += 1) {
    for (let x = 3; x <= 9; x += 1) setPixel(raw, width, x, y, artColor);
  }
  setPixel(raw, width, 5, 5, preservedColor);
  setPixel(raw, width, 7, 7, key);
  setPixel(raw, width, 0, 6, haloColor);

  assert.strictEqual(detectChromaKeyMode(raw, width, height, {}), mode,
    `${mode} background should be detected from source pixels`);
  removeChromaAndGuides(raw, width, height, {}, { chromaMode: mode });

  assert.strictEqual(getPixel(raw, width, 1, 1)[3], 0,
    `${mode} border-connected background should be transparent`);
  assert.strictEqual(getPixel(raw, width, 7, 7)[3], 0,
    `${mode} exact-key holes enclosed by art should still be transparent`);
  assert.deepStrictEqual(getPixel(raw, width, 5, 5), preservedColor.concat(255),
    `${mode} art color enclosed by a non-key outline should be preserved`);
}

testConnectedKeyRemoval('green', [0, 255, 0], [72, 42, 106], [58, 174, 88], [34, 218, 48]);
testConnectedKeyRemoval('magenta', [255, 0, 255], [62, 96, 52], [174, 68, 184], [222, 48, 218]);

const greenOutput = Buffer.from([
  58, 174, 88, 255,
  0, 255, 0, 255,
  34, 218, 48, 255
]);
clearOutputChromaSpill(greenOutput, 'green');
assert.deepStrictEqual(Array.from(greenOutput.subarray(0, 4)), [58, 174, 88, 255],
  'moderate internal green art should survive output spill cleanup');
assert.strictEqual(greenOutput[7], 0, 'exact green output key should be removed');
assert.strictEqual(greenOutput[11], 0, 'strong green antialias halo should be removed');

const magentaOutput = Buffer.from([
  174, 68, 184, 255,
  255, 0, 255, 255,
  222, 48, 218, 255
]);
clearOutputChromaSpill(magentaOutput, 'magenta');
assert.deepStrictEqual(Array.from(magentaOutput.subarray(0, 4)), [174, 68, 184, 255],
  'moderate internal magenta art should survive output spill cleanup');
assert.strictEqual(magentaOutput[7], 0, 'exact magenta output key should be removed');
assert.strictEqual(magentaOutput[11], 0, 'strong magenta antialias halo should be removed');

process.stdout.write('Project Starfall monster sheet processor tests passed.\n');
