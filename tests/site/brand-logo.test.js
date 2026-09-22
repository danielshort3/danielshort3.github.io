'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const faviconPath = path.join(root, 'img/brand/05-ds-favicon-small-icon.svg');
const generatorPath = path.join(root, 'build/resize_logo.js');

const favicon = fs.readFileSync(faviconPath, 'utf8');
const generator = fs.readFileSync(generatorPath, 'utf8');

// The bars are part of the approved DS identity and must survive compact exports.
for (const barStart of ['M165.0,111.0', 'M105.0,150.0', 'M70.0,191.0']) {
  assert.ok(favicon.includes(barStart), `favicon must preserve DS chart bar ${barStart}`);
}

assert.ok(
  !generator.includes('omit the three tiny chart bars') &&
  !generator.includes('generateFaviconSource'),
  'icon generation must not replace the approved favicon with a bar-less silhouette'
);

for (const asset of ['logo-16.png', 'logo-32.png', 'logo-64.png', 'logo-180.png', 'logo-192.png']) {
  assert.ok(fs.existsSync(path.join(root, 'img/ui', asset)), `missing generated icon ${asset}`);
}
assert.ok(fs.existsSync(path.join(root, 'favicon.ico')), 'missing favicon.ico');

console.log('Brand logo tests passed: compact favicon retains all three DS chart bars.');
