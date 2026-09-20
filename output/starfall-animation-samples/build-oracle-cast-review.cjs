'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const out = __dirname;
const repo = path.resolve(out, '../..');
const visualDir = 'C:/Users/clopt/.codex/visualizations/2026/09/19/01a0bb26-35be-7a93-9773-ce24aa252f30';

// Measurements only. No source pixels are changed or removed.
function measure(rgba, imageWidth, rect) {
  const { left: x0, top: y0, width: w, height: h } = rect;
  const seen = new Uint8Array(w * h);
  let largest = [];
  const alpha = (i) => rgba[((y0 + Math.floor(i / w)) * imageWidth + x0 + i % w) * 4 + 3];
  for (let start = 0; start < seen.length; start += 1) {
    if (seen[start] || alpha(start) < 128) continue;
    const pixels = [start];
    seen[start] = 1;
    for (let at = 0; at < pixels.length; at += 1) {
      const i = pixels[at];
      const x = i % w;
      const adjacent = [i - w, i + w];
      if (x > 0) adjacent.push(i - 1);
      if (x < w - 1) adjacent.push(i + 1);
      adjacent.forEach((next) => {
        if (next < 0 || next >= seen.length || seen[next] || alpha(next) < 128) return;
        seen[next] = 1;
        pixels.push(next);
      });
    }
    if (pixels.length > largest.length) largest = pixels;
  }
  if (!largest.length) throw new Error('Empty frame');
  const box = largest.reduce((b, i) => ({
    left: Math.min(b.left, i % w), top: Math.min(b.top, Math.floor(i / w)),
    right: Math.max(b.right, i % w), bottom: Math.max(b.bottom, Math.floor(i / w))
  }), { left: w, top: h, right: -1, bottom: -1 });
  const footBand = Math.max(2, Math.round((box.bottom - box.top + 1) * 0.07));
  const footXs = largest.filter((i) => Math.floor(i / w) >= box.bottom - footBand).map((i) => i % w);
  return { visible: box, rootX: (Math.min(...footXs) + Math.max(...footXs)) / 2, rootY: box.bottom + 1, pixels: largest.length };
}

async function main() {
  const originalPath = path.join(repo, 'img/project-starfall/animations/enemies/icebloom-oracle-compact-sheet.png');
  const idleRow = await sharp(originalPath).extract({ left: 0, top: 0, width: 384, height: 128 }).png().toBuffer();
  const buffRow = await sharp(originalPath).extract({ left: 0, top: 640, width: 384, height: 128 }).png().toBuffer();
  const beforePng = await sharp({ create: { width: 384, height: 256, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } })
    .composite([{ input: idleRow, top: 0, left: 0 }, { input: buffRow, top: 128, left: 0 }]).png().toBuffer();
  const beforeRaw = await sharp(beforePng).ensureAlpha().raw().toBuffer();
  const candidatePath = path.join(out, 'oracle-cast-study.png');
  const meta = await sharp(candidatePath).metadata();
  if (!meta.hasAlpha) throw new Error('Candidate does not contain transparency');
  const raw = await sharp(candidatePath).ensureAlpha().raw().toBuffer();
  const customPath = path.join(out, 'oracle-cast-registration.json');
  const custom = fs.existsSync(customPath) ? JSON.parse(fs.readFileSync(customPath, 'utf8')) : {};
  const sourceRects = custom.rects || Array.from({ length: 8 }, (_, i) => {
    const left = Math.round(i % 4 * meta.width / 4);
    const top = Math.round(Math.floor(i / 4) * meta.height / 2);
    return { left, top, width: Math.round((i % 4 + 1) * meta.width / 4) - left, height: Math.round((Math.floor(i / 4) + 1) * meta.height / 2) - top };
  });
  const measured = sourceRects.map((rect) => measure(raw, meta.width, rect));
  let encoding = 'lossless WebP';
  let encoded = await sharp(candidatePath).webp({ lossless: true, effort: 6 }).toBuffer();
  if (encoded.length > 620000) {
    encoded = await sharp(candidatePath).webp({ quality: 96, alphaQuality: 100, effort: 6 }).toBuffer();
    encoding = 'WebP quality 96, alpha quality 100; unmodified PNG retained beside review';
  }
  const originalRects = [{ left: 0, top: 0 }, { left: 0, top: 128 }, { left: 128, top: 128 }, { left: 256, top: 128 }, { left: 0, top: 0 }];
  const original = {
    image: 'before', duration: 1.2, holds: [210, 50, 50, 248, 162], scale: 86 / 102,
    frames: originalRects.map((rect) => ({
      x: rect.left, y: rect.top, w: 128, h: 128, rootX: 64, rootY: 118,
      visible: measure(beforeRaw, 384, { ...rect, width: 128, height: 128 }).visible
    }))
  };
  const neutral = measured[0].visible;
  const after = {
    image: 'after', duration: custom.duration || 1.2, holds: custom.holds || [210, 48, 48, 48, 76, 64, 64, 162],
    scale: (103 * 86 / 102) / (neutral.bottom - neutral.top + 1),
    frames: sourceRects.map((rect, i) => ({
      x: rect.left, y: rect.top, w: rect.width, h: rect.height,
      rootX: custom.anchors?.[i]?.x ?? measured[i].rootX,
      rootY: custom.anchors?.[i]?.y ?? measured[i].rootY,
      visible: measured[i].visible
    }))
  };
  const data = {
    title: 'Icebloom Oracle · healing cast', beforeLabel: 'Current healing cast', afterLabel: 'New bloom cast',
    images: { before: `data:image/png;base64,${beforePng.toString('base64')}`, after: `data:image/webp;base64,${encoded.toString('base64')}` },
    before: original, after,
    keyPoses: [{ frame: 0, label: 'Ready' }, { frame: 2, label: 'Gather' }, { frame: 4, label: 'Bloom' }, { frame: 6, label: 'Recover' }],
    phaseNames: ['Ready', 'Draw inward', 'Gather', 'Unfurl', 'Bloom', 'Follow-through', 'Recover', 'Ready again']
  };
  const template = fs.readFileSync(path.join(out, 'oracle-review-template.html'), 'utf8');
  const fragment = template.replace('__REVIEW_DATA__', JSON.stringify(data));
  if (Buffer.byteLength(fragment) >= 1000000) throw new Error(`Fragment too large: ${Buffer.byteLength(fragment)} bytes`);
  fs.mkdirSync(visualDir, { recursive: true });
  const fragmentPath = path.join(visualDir, 'oracle-cast-study.html');
  fs.writeFileSync(fragmentPath, fragment);
  const record = {
    candidate: candidatePath,
    sourceSha256: crypto.createHash('sha256').update(fs.readFileSync(candidatePath)).digest('hex'),
    dimensions: [meta.width, meta.height], measured, sourceRects, frames: after.frames, uniformScale: after.scale,
    method: (custom.method || 'Largest opaque component bounds; bottom 7 percent foot-band midpoint and lowest contact pixel.') + ' One fixed scale for all frames; per-frame translation only. No generated in-between frames or morphing.',
    encoding,
    originalSource: originalPath,
    originalSourceSha256: crypto.createHash('sha256').update(fs.readFileSync(originalPath)).digest('hex'),
    currentRuntimeCastSeconds: 0.58,
    staging: 'Both sides use 0.35 seconds ready, 0.58 seconds casting, and 0.27 seconds recovery/idle. The 1.2-second replay is for review; native healing cooldown is 2.8 seconds. Current buff row 5 has pose durations 1/12, 1/12, and 0.58-2/12 seconds. Separate runtime world effects are not shown.',
    afterCycleSeconds: after.duration, holds: after.holds, beforeCycleSeconds: original.duration,
    fragment: fragmentPath, fragmentBytes: Buffer.byteLength(fragment),
    limitations: 'This is a visual study for feedback, not a production-ready atlas or a guarantee of seamless animation.'
  };
  fs.writeFileSync(path.join(out, 'oracle-review-meta.json'), JSON.stringify(record, null, 2));
  console.log(JSON.stringify({ fragment: fragmentPath, bytes: record.fragmentBytes, scale: after.scale, frames: after.frames.length, timing: record.staging }));
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
