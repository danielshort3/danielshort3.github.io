'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const out = __dirname;
const repo = path.resolve(out, '../..');
const visualDir = 'C:/Users/clopt/.codex/visualizations/2026/09/19/01a0bb26-35be-7a93-9773-ce24aa252f30';

function bounds(rgba, width, height, x0, y0, w, h) {
  let left = w;
  let top = h;
  let right = -1;
  let bottom = -1;
  let pixels = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (rgba[((y0 + y) * width + x0 + x) * 4 + 3] < 16) continue;
      pixels += 1;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }
  return { left, top, right, bottom, width: right - left + 1, height: bottom - top + 1, pixels };
}

async function main() {
  const beforePng = await sharp(path.join(repo, 'img/project-starfall/animations/enemies/glowcap-healer-compact-sheet.png'))
    .extract({ left: 0, top: 128, width: 384, height: 128 }).png().toBuffer();
  const candidatePath = path.join(out, 'glowcap-hop-study.png');
  const meta = await sharp(candidatePath).metadata();
  if (!meta.hasAlpha) throw new Error('Candidate does not contain transparency');
  if (meta.width % 4 || meta.height % 4) throw new Error('Candidate does not fit a 4x4 grid');
  const raw = await sharp(candidatePath).ensureAlpha().raw().toBuffer();
  const fw = meta.width / 4;
  const fh = meta.height / 4;
  const boxes = Array.from({ length: 16 }, (_, i) => bounds(raw, meta.width, meta.height, i % 4 * fw, Math.floor(i / 4) * fh, fw, fh));
  if (boxes.some((box) => !box.pixels)) throw new Error('Candidate contains empty cells');
  if (boxes.some((box) => box.left <= 0 || box.top <= 0 || box.right >= fw - 1 || box.bottom >= fh - 1)) throw new Error('Candidate touches a cell boundary');
  let encoded = await sharp(candidatePath).webp({ lossless: true, effort: 6 }).toBuffer();
  if (encoded.length > 590000) encoded = await sharp(candidatePath).webp({ quality: 96, alphaQuality: 100, effort: 6 }).toBuffer();
  const original = {
    image: 'before', duration: 4 / 9, holds: [1, 1, 2], scale: 86 / 102,
    frames: Array.from({ length: 3 }, (_, i) => ({ x: i * 128, y: 0, w: 128, h: 128, rootX: 64, rootY: 118 }))
  };
  const neutral = boxes[0];
  const rootX = (neutral.left + neutral.right) / 2;
  const rootY = neutral.bottom;
  const after = {
    image: 'after', duration: 1, holds: Array(16).fill(1), scale: (94 * 86 / 102) / neutral.height,
    frames: Array.from({ length: 16 }, (_, i) => ({ x: i % 4 * fw, y: Math.floor(i / 4) * fh, w: fw, h: fh, rootX, rootY }))
  };
  const data = {
    images: { before: `data:image/png;base64,${beforePng.toString('base64')}`, after: `data:image/webp;base64,${encoded.toString('base64')}` },
    before: original,
    after,
    keyPoses: [{ frame: 2, label: 'Crouch' }, { frame: 3, label: 'Push-off' }, { frame: 7, label: 'Apex' }, { frame: 12, label: 'Landing' }],
    phaseNames: ['Stand', 'Anticipation', 'Crouch', 'Push-off', 'Ascent', 'Ascent', 'Near apex', 'Apex', 'Early descent', 'Descent', 'Pre-landing', 'Contact', 'Landing', 'Rebound', 'Settle', 'Recovery']
  };
  const template = fs.readFileSync(path.join(out, 'glowcap-review-template.html'), 'utf8');
  const fragment = template.replace('__REVIEW_DATA__', JSON.stringify(data));
  if (Buffer.byteLength(fragment) >= 1000000) throw new Error('Fragment too large');
  fs.mkdirSync(visualDir, { recursive: true });
  fs.writeFileSync(path.join(visualDir, 'glowcap-hop-study.html'), fragment);
  fs.writeFileSync(path.join(out, 'glowcap-review-meta.json'), JSON.stringify({ candidate: candidatePath, dimensions: [meta.width, meta.height], boxes, rootX, rootY, uniformScale: after.scale, afterCycleSeconds: after.duration, beforeCycleSeconds: original.duration, fragmentBytes: Buffer.byteLength(fragment) }, null, 2));
  console.log(JSON.stringify({ fragment: path.join(visualDir, 'glowcap-hop-study.html'), bytes: Buffer.byteLength(fragment), candidateDimensions: [meta.width, meta.height], boxes, rootX, rootY }));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
