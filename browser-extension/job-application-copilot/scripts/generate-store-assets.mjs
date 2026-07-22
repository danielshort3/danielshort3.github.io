import { mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const extensionDirectory = path.resolve(scriptDirectory, '..');
const assetDirectory = path.join(extensionDirectory, 'store', 'assets');
const requireFromRoot = createRequire(new URL('../../../package.json', import.meta.url));
const sharp = requireFromRoot('sharp');

await mkdir(assetDirectory, { recursive: true });

const outputs = [
  {
    source: 'icon-source.svg',
    target: 'icon-128.png',
    width: 128,
    height: 128
  },
  {
    source: 'small-promo-source.svg',
    target: 'small-promo-440x280.png',
    width: 440,
    height: 280
  }
];

for (const output of outputs) {
  const sourcePath = path.join(assetDirectory, output.source);
  const targetPath = path.join(assetDirectory, output.target);
  await sharp(sourcePath, { density: 192 })
    .resize(output.width, output.height, { fit: 'fill' })
    .png({ compressionLevel: 9, adaptiveFiltering: true, palette: false })
    .toFile(targetPath);

  const metadata = await sharp(targetPath).metadata();
  if (metadata.width !== output.width || metadata.height !== output.height || metadata.format !== 'png') {
    throw new Error(`Unexpected generated asset metadata for ${output.target}`);
  }
  process.stdout.write(`Generated ${path.relative(extensionDirectory, targetPath)} (${metadata.width}x${metadata.height})\n`);
}
