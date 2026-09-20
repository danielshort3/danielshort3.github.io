'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const out = __dirname;
const root = path.resolve(out, '../..');
const registrations = JSON.parse(fs.readFileSync(path.join(out, 'registration.json'), 'utf8'));

async function build() {
  const images = {};
  const samples = {};
  const integrity = [];
  for (const registration of registrations.samples) {
    const id = registration.id.startsWith('icebloom') ? 'oracle' : 'fox';
    const source = fs.readFileSync(path.join(root, registration.source));
    const hash = crypto.createHash('sha256').update(source).digest('hex');
    if (hash !== registration.sourceSha256) throw new Error(`Source changed: ${registration.source}`);
    const { frameWidth: w, frameHeight: h, frameCount: count } = registration;
    // Export the original atlas row losslessly for the standalone review.
    // Corrections are drawing metadata; the encoded pixels stay unchanged.
    const row = await sharp(source).extract({ left: 0, top: registration.row * h, width: w * count, height: h }).png({ compressionLevel: 9 }).toBuffer();
    const originalPixels = await sharp(source).extract({ left: 0, top: registration.row * h, width: w * count, height: h }).ensureAlpha().raw().toBuffer();
    const reviewPixels = await sharp(row).ensureAlpha().raw().toBuffer();
    if (!originalPixels.equals(reviewPixels)) throw new Error(`Lossless review conversion failed: ${id}`);
    images[id] = `data:image/png;base64,${row.toString('base64')}`;
    const rootX = id === 'oracle' ? 64 : 80;
    const rootY = id === 'oracle' ? 118 : 80 / 0.55;
    const before = {
      image: id,
      scale: id === 'oracle' ? 86 / 102 : 88 / 160,
      duration: registration.playback.cycleSeconds,
      holds: registration.playback.holds,
      frames: Array.from({ length: count }, (_, frame) => ({ x: frame * w, y: 0, w, h, rootX, rootY }))
    };
    const after = {
      ...before,
      frames: before.frames.map((frame, i) => ({ ...frame, rootX: rootX - registration.offsets[i].x, rootY: rootY - registration.offsets[i].y }))
    };
    samples[id] = {
      before,
      after,
      note: id === 'oracle' ? 'Original poses and 2-second timing · steady ground alignment' : 'Original poses and 1-second timing · steady paw alignment'
    };
    integrity.push({ id, source: registration.source, sourceSha256: hash, encodedRowBytes: row.length, originalPixelsPreserved: true });
  }
  const template = fs.readFileSync(path.join(out, 'review-template.html'), 'utf8');
  const fragment = template.replace('__REVIEW_DATA__', JSON.stringify({ images, samples }));
  if (Buffer.byteLength(fragment) >= 1000000) throw new Error('Review exceeds the inline size limit');
  fs.writeFileSync(path.join(out, 'starfall-before-after.html'), fragment);
  fs.writeFileSync(path.join(out, 'review-integrity.json'), JSON.stringify({ fragmentBytes: Buffer.byteLength(fragment), samples: integrity }, null, 2));
  console.log(JSON.stringify({ fragmentBytes: Buffer.byteLength(fragment), samples: integrity }));
}

build().catch((error) => { console.error(error); process.exitCode = 1; });
