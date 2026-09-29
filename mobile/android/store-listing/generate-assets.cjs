#!/usr/bin/env node
'use strict';

const path = require('node:path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '../../..');
const logo = path.join(root, 'img/brand/05-ds-favicon-small-icon.svg');
const socialCard = path.join(root, 'img/brand/personal-social-card.png');

async function main() {
  const iconMark = await sharp(logo).resize(340, 340, { fit: 'contain' }).png().toBuffer();
  await sharp({
    create: { width: 512, height: 512, channels: 4, background: '#ffffff' },
  }).composite([{ input: iconMark, gravity: 'centre' }]).png().toFile(path.join(__dirname, 'play-icon.png'));

  await sharp(socialCard)
    .resize(1024, 500, { fit: 'cover', position: 'centre' })
    .flatten({ background: '#ffffff' })
    .removeAlpha()
    .png()
    .toFile(path.join(__dirname, 'feature-graphic.png'));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
