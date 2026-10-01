'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');
const { getImageJob, renderVariant } = require('../../../build/lib/image-variant-recipes');
const { versionedImageUrl } = require('../../../build/lib/versioned-image-url');
const icons = require('../../../js/common/catalog-icons');

const digest = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

async function imageFixture(root) {
  const relative = 'img/games/icons/cache-regression.png';
  const filename = path.join(root, relative);
  const pixels = Buffer.alloc(256 * 256 * 4);
  for (let y = 0; y < 256; y += 1) {
    for (let x = 0; x < 256; x += 1) {
      const offset = (y * 256 + x) * 4;
      pixels[offset] = x;
      pixels[offset + 1] = y;
      pixels[offset + 2] = (x * 7 + y * 11) % 256;
      pixels[offset + 3] = 255;
    }
  }
  const png = await sharp(pixels, { raw: { width: 256, height: 256, channels: 4 } }).png().toBuffer();
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  fs.writeFileSync(filename, png);
  const current = getImageJob(relative);
  const previous = JSON.parse(JSON.stringify(current));
  previous.outputs[0].width = 256;
  previous.outputs[0].options.quality = 90;
  const [oldWebp, newWebp] = await Promise.all([
    renderVariant(png, previous.outputs[0]), renderVariant(png, current.outputs[0])
  ]);
  const oldGenerationUrl = icons.webpSource(versionedImageUrl(`/${relative}`, { root, resolveJob: () => previous }));
  const newPngUrl = versionedImageUrl(`/${relative}`, { root });
  return {
    relative, filename, png, previous, current, oldWebp, newWebp, oldGenerationUrl, newPngUrl,
    legacyUrl: icons.webpSource(`/${relative}?v=${digest(png).slice(0, 12)}`),
    newUrl: icons.webpSource(newPngUrl),
    pngHash: digest(png), oldHash: digest(oldWebp), newHash: digest(newWebp)
  };
}

module.exports = { imageFixture, digest };
