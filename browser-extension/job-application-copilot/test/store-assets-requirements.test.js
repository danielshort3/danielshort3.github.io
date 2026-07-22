import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const packageDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const assetPath = name => path.join(packageDirectory, 'store', 'assets', name);

const pngDimensions = async (filePath) => {
  const bytes = await readFile(filePath);
  assert.deepEqual(
    [...bytes.subarray(0, 8)],
    [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    `${path.basename(filePath)} must be a PNG`
  );
  assert.equal(bytes.subarray(12, 16).toString('ascii'), 'IHDR');
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20)
  };
};

test('Chrome Web Store PNG assets use the required dimensions', async () => {
  assert.deepEqual(await pngDimensions(assetPath('icon-128.png')), { width: 128, height: 128 });
  assert.deepEqual(await pngDimensions(assetPath('small-promo-440x280.png')), { width: 440, height: 280 });
  assert.deepEqual(await pngDimensions(assetPath('screenshots/01-reviewed-answers.png')), { width: 1280, height: 800 });
});

test('square icon artwork maps to a 96px safe area with 16px padding', async () => {
  const source = await readFile(assetPath('icon-source.svg'), 'utf8');
  const transform = source.match(/id="icon-artwork" transform="translate\(([\d.]+) ([\d.]+)\) scale\(([\d.]+)\)"/u);
  assert.ok(transform, 'icon-source.svg must retain the named safe-area transform');

  const [, translateXText, translateYText, scaleText] = transform;
  const translateX = Number(translateXText);
  const translateY = Number(translateYText);
  const scale = Number(scaleText);
  const sourceToPng = 128 / 512;
  const originalSquare = { x: 16, y: 16, size: 480 };
  const mapped = {
    x: (translateX + originalSquare.x * scale) * sourceToPng,
    y: (translateY + originalSquare.y * scale) * sourceToPng,
    size: originalSquare.size * scale * sourceToPng
  };

  assert.deepEqual(mapped, { x: 16, y: 16, size: 96 });
});
