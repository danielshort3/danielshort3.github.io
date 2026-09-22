#!/usr/bin/env node
/* Resize approved brand logo assets for favicons and compact UI surfaces. */

const fs = require('node:fs/promises');
const path = require('node:path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const sourceFavicon = path.join(root, 'img', 'brand', '05-ds-favicon-small-icon.svg');
const uiDir = path.join(root, 'img', 'ui');
const faviconBackground = { r: 255, g: 255, b: 255, alpha: 1 };

const logoSizes = [
  ['logo-16.png', 16],
  ['logo-32.png', 32],
  ['logo-64.png', 64],
  ['logo-180.png', 180],
  ['logo-192.png', 192],
];

const faviconSizes = [16, 32, 48, 64];

function rel(filePath) {
  return path.relative(root, filePath).replaceAll(path.sep, '/');
}

async function pngBuffer(size) {
  return faviconImage(size)
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

function faviconImage(size) {
  return sharp(sourceFavicon)
    .resize(size, size, {
      fit: 'contain',
      background: faviconBackground,
    })
    .flatten({ background: faviconBackground })
    .removeAlpha();
}

function createIco(images) {
  const headerSize = 6;
  const entrySize = 16;
  const directorySize = headerSize + images.length * entrySize;
  const header = Buffer.alloc(directorySize);

  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(images.length, 4);

  let offset = directorySize;
  images.forEach(({ size, buffer }, index) => {
    const entryOffset = headerSize + index * entrySize;
    header.writeUInt8(size === 256 ? 0 : size, entryOffset);
    header.writeUInt8(size === 256 ? 0 : size, entryOffset + 1);
    header.writeUInt8(0, entryOffset + 2);
    header.writeUInt8(0, entryOffset + 3);
    header.writeUInt16LE(1, entryOffset + 4);
    header.writeUInt16LE(32, entryOffset + 6);
    header.writeUInt32LE(buffer.length, entryOffset + 8);
    header.writeUInt32LE(offset, entryOffset + 12);
    offset += buffer.length;
  });

  return Buffer.concat([header, ...images.map((image) => image.buffer)]);
}

async function generateMainLogo() {
  await fs.access(sourceFavicon);
  await fs.mkdir(uiDir, { recursive: true });

  await Promise.all(
    logoSizes.map(async ([name, size]) => {
      const output = path.join(uiDir, name);
      await faviconImage(size)
        .png({ compressionLevel: 9, adaptiveFiltering: true })
        .toFile(output);
      console.log(`[resize_logo] Wrote ${rel(output)}`);
    })
  );

  const faviconImages = await Promise.all(
    faviconSizes.map(async (size) => ({ size, buffer: await pngBuffer(size) }))
  );
  const ico = createIco(faviconImages);
  const favicon = path.join(root, 'favicon.ico');
  await fs.writeFile(favicon, ico);
  console.log(`[resize_logo] Wrote ${rel(favicon)}`);
}

async function generateCertLogos() {
  const certDir = path.join(root, 'img', 'cert_logos');
  const entries = await fs.readdir(certDir, { withFileTypes: true }).catch(() => []);
  const sourceFiles = entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.png'))
    .filter((entry) => !/-(?:24|48)\.png$/i.test(entry.name))
    .map((entry) => path.join(certDir, entry.name));

  await Promise.all(
    sourceFiles.flatMap((filePath) =>
      [24, 48].map(async (targetHeight) => {
        const parsed = path.parse(filePath);
        const output = path.join(parsed.dir, `${parsed.name}-${targetHeight}.png`);
        await sharp(filePath)
          .resize({ height: targetHeight, fit: 'inside', withoutEnlargement: false })
          .png({ compressionLevel: 9, adaptiveFiltering: true })
          .toFile(output);
        console.log(`[resize_logo] Wrote ${rel(output)}`);
      })
    )
  );
}

async function main() {
  try {
    await generateMainLogo();
    await generateCertLogos();
  } catch (error) {
    console.error(`[resize_logo] ${error.message}`);
    process.exitCode = 1;
  }
}

main();
