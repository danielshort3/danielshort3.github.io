import { mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourcePath = path.join(packageDir, 'store', 'assets', 'icon-source.svg');
const outputDir = path.join(packageDir, 'src', 'assets', 'icons');
const requireFromRepository = createRequire(new URL('../../../package.json', import.meta.url));
const sharp = requireFromRepository('sharp');

await mkdir(outputDir, { recursive: true });

for (const size of [16, 32, 48, 128]) {
  const outputPath = path.join(outputDir, `icon-${size}.png`);
  await sharp(sourcePath, { density: 384 })
    .resize(size, size, { fit: 'fill' })
    .png({ compressionLevel: 9, adaptiveFiltering: true, palette: false })
    .toFile(outputPath);
  const metadata = await sharp(outputPath).metadata();
  if (metadata.format !== 'png' || metadata.width !== size || metadata.height !== size) {
    throw new Error(`Unexpected generated runtime icon metadata for ${outputPath}`);
  }
  process.stdout.write(`Generated ${path.relative(packageDir, outputPath)} (${size}x${size})\n`);
}
