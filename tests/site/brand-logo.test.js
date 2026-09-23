'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '../..');
const faviconPath = path.join(root, 'img/brand/05-ds-favicon-small-icon.svg');
const generatorPath = path.join(root, 'build/resize_logo.js');

const favicon = fs.readFileSync(faviconPath, 'utf8');
const generator = fs.readFileSync(generatorPath, 'utf8');
const navCss = fs.readFileSync(path.join(root, 'css/layout/nav.css'), 'utf8');
const utilityLayoutCss = fs.readFileSync(path.join(root, 'css/utilities/layout.css'), 'utf8');

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

assert.ok(
  !navCss.includes('.nav .brand-title::after'),
  'tablet header branding must not add a responsive underline beneath the name'
);
assert.ok(
  !utilityLayoutCss.includes('.brand-title::after'),
  'mobile/legacy header branding must not add an underline beneath the name'
);

async function verifyPwaIcon() {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
  const maskableIcon = manifest.icons.find((icon) => icon.src === 'img/ui/logo-512-maskable.png');
  assert.ok(maskableIcon, 'PWA manifest must reference the current 512px icon');
  assert.equal(maskableIcon.sizes, '512x512');
  assert.equal(maskableIcon.type, 'image/png');
  assert.deepEqual(maskableIcon.purpose.split(/\s+/).sort(), ['any', 'maskable']);
  assert.ok(!manifest.icons.some((icon) => icon.src === 'img/ui/logo-512.png'), 'PWA manifest must not use the old cropped icon');

  const iconPath = path.join(root, maskableIcon.src);
  const { data, info } = await sharp(iconPath).raw().toBuffer({ resolveWithObject: true });
  assert.equal(info.width, 512);
  assert.equal(info.height, 512);
  assert.equal(info.channels, 3, 'maskable icon must have an opaque background');

  let markPixels = 0;
  const safeRadiusSquared = (info.width / 3) ** 2;
  for (let y = 0; y < info.height; y += 1) {
    for (let x = 0; x < info.width; x += 1) {
      const index = (y * info.width + x) * info.channels;
      const red = data[index];
      const green = data[index + 1];
      const blue = data[index + 2];
      if (red >= 245 && green >= 245 && blue >= 245) continue;
      markPixels += 1;
      assert.ok((x - 255.5) ** 2 + (y - 255.5) ** 2 <= safeRadiusSquared,
        `PWA mark leaves the maskable safe circle at (${x}, ${y})`);
      assert.ok(red > 3 || green > 3 || blue > 3,
        `PWA mark contains a black resize artifact at (${x}, ${y})`);
    }
  }
  assert.ok(markPixels > 10000, 'PWA icon must contain the DS mark');
}

verifyPwaIcon()
  .then(() => console.log('Brand logo tests passed: favicon and maskable PWA icon are intact.'))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
