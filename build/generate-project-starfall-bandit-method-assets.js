#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const {
  GUIDE_LINE_HEX,
  clearGuidePixels,
  detectGuideGrid,
  getGridCellRect
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = path.join(ROOT, 'asset-sources/project-starfall/enemies/compact');
const FRAME = 128;
const COLUMNS = 3;
const ROWS = 8;
const SOURCE_CELL = 180;
const GUIDE = 6;
const KEY_BACKGROUND = '#00ff00';
const TRANSPARENT = Object.freeze({ r: 0, g: 0, b: 0, alpha: 0 });

const HYBRID_SOURCE = path.join(SOURCE_DIR, 'bandit-cutter-hybrid-keyposes-source.png');
const PUPPET_SOURCE = path.join(SOURCE_DIR, 'bandit-cutter-puppet-source.png');

function isRemovedPixel(raw, offset) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return true;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  if (r < 112 && g > 132 && b > 132 && Math.abs(g - b) < 112) return true;
  return g > 120 && g > r * 1.45 && g > b * 1.45;
}

function removeChroma(raw, width, height) {
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (!isRemovedPixel(raw, offset)) continue;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
  }
}

function getAlphaBounds(raw, width, rect, threshold = 8) {
  let minX = rect.x + rect.w;
  let minY = rect.y + rect.h;
  let maxX = rect.x - 1;
  let maxY = rect.y - 1;
  for (let y = rect.y; y < rect.y + rect.h; y += 1) {
    for (let x = rect.x; x < rect.x + rect.w; x += 1) {
      if (raw[(y * width + x) * 4 + 3] <= threshold) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }
  if (maxX < minX || maxY < minY) return null;
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function isBodyAnchorPixel(raw, offset) {
  if (raw[offset + 3] <= 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  if (r > 176 && g > 176 && b > 156 && max - min < 96) return false;
  if (g > 170 && b > 145 && r < 150) return false;
  if (r > 210 && g > 180 && b < 120) return false;
  return true;
}

function expandBounds(bounds, rect, pad) {
  const x = Math.max(rect.x, bounds.x - pad);
  const y = Math.max(rect.y, bounds.y - pad);
  const right = Math.min(rect.x + rect.w, bounds.x + bounds.w + pad);
  const bottom = Math.min(rect.y + rect.h, bounds.y + bounds.h + pad);
  return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
}

function getBodyAnchorBounds(raw, width, rect) {
  let minX = rect.x + rect.w;
  let minY = rect.y + rect.h;
  let maxX = rect.x - 1;
  let maxY = rect.y - 1;
  let area = 0;
  for (let y = rect.y; y < rect.y + rect.h; y += 1) {
    for (let x = rect.x; x < rect.x + rect.w; x += 1) {
      const offset = (y * width + x) * 4;
      if (!isBodyAnchorPixel(raw, offset)) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      area += 1;
    }
  }
  if (area < 180 || maxX < minX || maxY < minY) return null;
  const body = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
  return expandBounds(body, rect, 18);
}

function cropRaw(raw, width, bounds) {
  const cropped = Buffer.alloc(bounds.w * bounds.h * 4);
  for (let y = 0; y < bounds.h; y += 1) {
    const sourceStart = ((bounds.y + y) * width + bounds.x) * 4;
    raw.copy(cropped, y * bounds.w * 4, sourceStart, sourceStart + bounds.w * 4);
  }
  return cropped;
}

async function normalizeCell(raw, width, rect) {
  const alphaBounds = getAlphaBounds(raw, width, rect);
  if (!alphaBounds) throw new Error('Generated key-pose cell is missing visible art');
  const bounds = getBodyAnchorBounds(raw, width, rect) || alphaBounds;
  const cropped = cropRaw(raw, width, bounds);
  const scale = Math.min(110 / Math.max(1, bounds.w), 106 / Math.max(1, bounds.h));
  const outW = Math.max(1, Math.round(bounds.w * scale));
  const outH = Math.max(1, Math.round(bounds.h * scale));
  const resized = await sharp(cropped, {
    raw: { width: bounds.w, height: bounds.h, channels: 4 }
  })
    .resize(outW, outH, { fit: 'fill', background: TRANSPARENT, kernel: 'nearest' })
    .png()
    .toBuffer();
  const left = Math.round((FRAME - outW) / 2);
  const top = Math.round(118 - outH);
  return sharp({
    create: { width: FRAME, height: FRAME, channels: 4, background: TRANSPARENT }
  })
    .composite([{ input: resized, left, top: Math.max(4, top) }])
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

async function extractPoseFrames(sourcePath, label) {
  if (!fs.existsSync(sourcePath)) throw new Error(`Missing ${label} source: ${path.relative(ROOT, sourcePath)}`);
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(data, info.width, info.height, {
    columns: 3,
    rows: 1,
    label,
    guideColor: GUIDE_LINE_HEX
  });
  clearGuidePixels(data, info.width, info.height, GUIDE_LINE_HEX);
  removeChroma(data, info.width, info.height);
  const frames = [];
  for (let col = 0; col < 3; col += 1) {
    frames.push(await normalizeCell(data, info.width, getGridCellRect(grid, 0, col, 10)));
  }
  return frames;
}

async function translateFrame(frame, dx, dy) {
  const { data, info } = await sharp(frame)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const out = Buffer.alloc(info.width * info.height * 4);
  for (let y = 0; y < info.height; y += 1) {
    const targetY = y + dy;
    if (targetY < 0 || targetY >= info.height) continue;
    for (let x = 0; x < info.width; x += 1) {
      const targetX = x + dx;
      if (targetX < 0 || targetX >= info.width) continue;
      const sourceOffset = (y * info.width + x) * 4;
      if (data[sourceOffset + 3] <= 0) continue;
      data.copy(out, (targetY * info.width + targetX) * 4, sourceOffset, sourceOffset + 4);
    }
  }
  return sharp(out, { raw: { width: info.width, height: info.height, channels: 4 } })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

function svgBuffer(content) {
  return Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${FRAME}" height="${FRAME}" viewBox="0 0 ${FRAME} ${FRAME}">${content}</svg>`);
}

function slashOverlay(stage) {
  const paths = [
    '<path d="M58 74 C78 58 98 58 118 70" fill="none" stroke="#efffff" stroke-width="8" stroke-linecap="round" opacity=".9"/><path d="M60 78 C80 64 99 64 116 74" fill="none" stroke="#79f7c8" stroke-width="3" stroke-linecap="round" opacity=".75"/>',
    '<path d="M42 70 C70 45 105 46 124 72" fill="none" stroke="#f7ffff" stroke-width="9" stroke-linecap="round" opacity=".95"/><path d="M44 76 C72 54 105 56 122 78" fill="none" stroke="#7af4ce" stroke-width="3" stroke-linecap="round" opacity=".8"/>',
    '<path d="M52 84 C82 76 104 62 124 42" fill="none" stroke="#efffff" stroke-width="8" stroke-linecap="round" opacity=".86"/><path d="M58 88 C86 80 107 66 122 48" fill="none" stroke="#9af6d8" stroke-width="3" stroke-linecap="round" opacity=".72"/>'
  ];
  return svgBuffer(paths[Math.max(0, Math.min(paths.length - 1, stage))]);
}

function daggerOverlay(stage) {
  const x = [74, 88, 102][Math.max(0, Math.min(2, stage))];
  return svgBuffer(`<g transform="translate(${x} 66)"><path d="M2 8 L34 0 L28 8 L34 16 Z" fill="#dfe8ec" stroke="#25323a" stroke-width="2"/><rect x="-3" y="5" width="10" height="6" rx="1" fill="#7b4a25" stroke="#2a1a13" stroke-width="1"/></g>`);
}

function auraOverlay(stage) {
  const opacity = [0.45, 0.7, 0.55][Math.max(0, Math.min(2, stage))];
  return svgBuffer(`<g opacity="${opacity}"><ellipse cx="64" cy="112" rx="${30 + stage * 4}" ry="8" fill="none" stroke="#f7d95d" stroke-width="3"/><path d="M32 70 L36 78 L44 82 L36 86 L32 94 L28 86 L20 82 L28 78 Z" fill="#fff171"/><path d="M92 30 L96 38 L104 42 L96 46 L92 54 L88 46 L80 42 L88 38 Z" fill="#fff171"/><path d="M104 86 L107 92 L113 95 L107 98 L104 104 L101 98 L95 95 L101 92 Z" fill="#fff171"/></g>`);
}

function idleOverlay(stage) {
  if (stage === 1) {
    return svgBuffer('<g opacity=".72"><ellipse cx="62" cy="51" rx="5" ry="3" fill="#fff1bd"/><ellipse cx="83" cy="51" rx="5" ry="3" fill="#fff1bd"/><path d="M46 69 C58 75 76 76 91 70" fill="none" stroke="#d96128" stroke-width="5" stroke-linecap="round"/></g>');
  }
  return svgBuffer('<g opacity=".6"><ellipse cx="62" cy="50" rx="4" ry="4" fill="#ffe6aa"/><ellipse cx="83" cy="50" rx="4" ry="4" fill="#ffe6aa"/><path d="M92 62 C99 63 105 65 110 69" fill="none" stroke="#c04f24" stroke-width="4" stroke-linecap="round"/></g>');
}

function hitOverlay(stage) {
  const x = [40, 36, 46][Math.max(0, Math.min(2, stage))];
  return svgBuffer(`<path d="M${x} 34 L54 45 L74 36 L64 54 L82 66 L58 64 L48 84 L44 62 L22 58 L42 50 Z" fill="#ffe36b" stroke="#4b2b24" stroke-width="2" opacity=".9"/>`);
}

function defeatOverlay(stage) {
  if (stage === 1) {
    return svgBuffer('<g opacity=".85"><path d="M78 62 C84 56 92 58 91 65 C91 72 80 72 84 80 C88 86 100 84 104 78" fill="none" stroke="#e8eef0" stroke-width="4" stroke-linecap="round"/><circle cx="54" cy="86" r="4" fill="#fff071"/><circle cx="110" cy="92" r="3" fill="#fff071"/></g>');
  }
  return svgBuffer('<g opacity=".8"><path d="M70 68 L74 76 L82 80 L74 84 L70 92 L66 84 L58 80 L66 76 Z" fill="#fff071"/><path d="M98 72 L102 80 L110 84 L102 88 L98 96 L94 88 L86 84 L94 80 Z" fill="#fff071"/><path d="M48 102 C62 96 78 96 92 102" fill="none" stroke="#d5e9ef" stroke-width="3" stroke-linecap="round"/></g>');
}

async function composeFrame(baseFrame, overlays) {
  if (!overlays.length) return baseFrame;
  return sharp({
    create: { width: FRAME, height: FRAME, channels: 4, background: TRANSPARENT }
  })
    .composite([{ input: baseFrame, left: 0, top: 0 }].concat(overlays.map((overlay) => ({ input: overlay, left: 0, top: 0 }))))
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
}

async function makeHybridFrames(hybridPoses, puppetPoses) {
  const [neutral, step, attack] = hybridPoses;
  const attackBody = step || attack;
  const defeat = puppetPoses[2];
  return [
    [
      await translateFrame(neutral, 0, 0),
      await composeFrame(await translateFrame(neutral, 0, -2), [idleOverlay(1)]),
      await composeFrame(await translateFrame(neutral, 0, 0), [idleOverlay(2)])
    ],
    [await translateFrame(step, -4, 0), await translateFrame(neutral, 0, 0), await translateFrame(step, 4, 0)],
    [await translateFrame(neutral, -2, 0), await translateFrame(attackBody, 0, -1), await translateFrame(attackBody, 2, 0)],
    [
      await composeFrame(await translateFrame(attackBody, -4, 0), [slashOverlay(0)]),
      await composeFrame(await translateFrame(attackBody, 0, 0), [slashOverlay(1)]),
      await composeFrame(await translateFrame(attackBody, 4, 0), [slashOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(neutral, -2, 0), [daggerOverlay(0)]),
      await composeFrame(await translateFrame(neutral, 0, 0), [daggerOverlay(1)]),
      await composeFrame(await translateFrame(neutral, 2, 0), [daggerOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(neutral, 0, 0), [auraOverlay(0)]),
      await composeFrame(await translateFrame(neutral, 0, -2), [auraOverlay(1)]),
      await composeFrame(await translateFrame(attackBody, 0, 0), [auraOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(neutral, -3, 1), [hitOverlay(0)]),
      await composeFrame(await translateFrame(neutral, 1, 2), [hitOverlay(1)]),
      await composeFrame(await translateFrame(neutral, 3, 1), [hitOverlay(2)])
    ],
    [
      await translateFrame(defeat, -3, 0),
      await composeFrame(await translateFrame(defeat, 0, 0), [defeatOverlay(1)]),
      await composeFrame(await translateFrame(defeat, 3, 0), [defeatOverlay(2)])
    ]
  ];
}

async function makePuppetFrames(puppetPoses) {
  const [neutral, guard, defeat] = puppetPoses;
  return [
    [
      await translateFrame(neutral, 0, 0),
      await composeFrame(await translateFrame(neutral, 0, -2), [idleOverlay(1)]),
      await composeFrame(await translateFrame(neutral, 0, 0), [idleOverlay(2)])
    ],
    [await translateFrame(neutral, -5, 0), await translateFrame(guard, 0, 0), await translateFrame(neutral, 5, 0)],
    [await translateFrame(neutral, -2, 0), await translateFrame(guard, 0, 0), await translateFrame(guard, 2, 0)],
    [
      await composeFrame(await translateFrame(guard, -4, 0), [slashOverlay(0)]),
      await composeFrame(await translateFrame(guard, 0, 0), [slashOverlay(1)]),
      await composeFrame(await translateFrame(guard, 4, 0), [slashOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(neutral, -2, 0), [daggerOverlay(0)]),
      await composeFrame(await translateFrame(neutral, 0, 0), [daggerOverlay(1)]),
      await composeFrame(await translateFrame(neutral, 2, 0), [daggerOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(neutral, 0, 0), [auraOverlay(0)]),
      await composeFrame(await translateFrame(neutral, 0, -1), [auraOverlay(1)]),
      await composeFrame(await translateFrame(guard, 0, 0), [auraOverlay(2)])
    ],
    [
      await composeFrame(await translateFrame(guard, -3, 1), [hitOverlay(0)]),
      await composeFrame(await translateFrame(guard, 1, 2), [hitOverlay(1)]),
      await composeFrame(await translateFrame(guard, 3, 1), [hitOverlay(2)])
    ],
    [
      await translateFrame(defeat, -3, 0),
      await composeFrame(await translateFrame(defeat, 0, 0), [defeatOverlay(1)]),
      await composeFrame(await translateFrame(defeat, 3, 0), [defeatOverlay(2)])
    ]
  ];
}

async function writeCompactSource(frames, targetPath) {
  const width = GUIDE + COLUMNS * (SOURCE_CELL + GUIDE);
  const height = GUIDE + ROWS * (SOURCE_CELL + GUIDE);
  const rects = [];
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLUMNS; col += 1) {
      const x = GUIDE + col * (SOURCE_CELL + GUIDE);
      const y = GUIDE + row * (SOURCE_CELL + GUIDE);
      rects.push(`<rect x="${x}" y="${y}" width="${SOURCE_CELL}" height="${SOURCE_CELL}" fill="${KEY_BACKGROUND}"/>`);
    }
  }
  const background = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><rect width="${width}" height="${height}" fill="${GUIDE_LINE_HEX}"/>${rects.join('')}</svg>`);
  const composites = [];
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLUMNS; col += 1) {
      composites.push({
        input: frames[row][col],
        left: GUIDE + col * (SOURCE_CELL + GUIDE) + Math.round((SOURCE_CELL - FRAME) / 2),
        top: GUIDE + row * (SOURCE_CELL + GUIDE) + Math.round((SOURCE_CELL - FRAME) / 2)
      });
    }
  }
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  await sharp(background)
    .composite(composites)
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(targetPath);
  process.stdout.write(`Wrote ${path.relative(ROOT, targetPath).replace(/\\/g, '/')}\n`);
}

async function main() {
  const hybridPoses = await extractPoseFrames(HYBRID_SOURCE, 'Bandit Cutter hybrid key poses');
  const puppetPoses = await extractPoseFrames(PUPPET_SOURCE, 'Bandit Cutter puppet poses');
  await writeCompactSource(await makeHybridFrames(hybridPoses, puppetPoses), path.join(SOURCE_DIR, 'bandit-cutter-hybrid-compact-source.png'));
  await writeCompactSource(await makePuppetFrames(puppetPoses), path.join(SOURCE_DIR, 'bandit-cutter-puppet-compact-source.png'));
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.message ? error.message : error);
    process.exit(1);
  });
}
