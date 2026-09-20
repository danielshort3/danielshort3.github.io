'use strict';
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '../..');
const beforePath = path.join(__dirname, 'before-sheet.png');
const actions = ['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat'];

async function load(file) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width };
}

function shellTemplate(image, row = 0, column = 0, rect = { left: 42, top: 75, width: 48, height: 34 }) {
  const result = [];
  for (let y = rect.top; y < rect.top + rect.height; y += 2) {
    for (let x = rect.left; x < rect.left + rect.width; x += 2) {
      const offset = ((row * 160 + y) * image.width + column * 160 + x) * 4;
      if (image.data[offset + 3] <= 128) continue;
      result.push([x, y, ...image.data.subarray(offset, offset + 4)]);
    }
  }
  return result;
}

function match(image, template, row, column, range = { x: 30, y: 40 }) {
  let best = { error: Infinity };
  for (let dy = -range.y; dy <= range.y; dy += 1) {
    for (let dx = -range.x; dx <= range.x; dx += 1) {
      let error = 0;
      for (const p of template) {
        const offset = ((row * 160 + p[1] + dy) * image.width + column * 160 + p[0] + dx) * 4;
        error += Math.abs(p[2] - image.data[offset]) + Math.abs(p[3] - image.data[offset + 1]) + Math.abs(p[4] - image.data[offset + 2]) + Math.abs(p[5] - image.data[offset + 3]);
      }
      error /= template.length;
      if (error < best.error) best = { dx, dy, error };
    }
  }
  return { ...best, error: Number(best.error.toFixed(2)) };
}

function contour(image, row, column) {
  let top = 160;
  const alpha = (x, y) => image.data[((row * 160 + y) * image.width + column * 160 + x) * 4 + 3];
  for (let y = 0; y < 160; y += 1) {
    let count = 0;
    for (let x = 0; x < 160; x += 1) if (alpha(x, y) > 128) count += 1;
    if (count >= 6) { top = y; break; }
  }
  let sx = 0, count = 0, minX = 160, maxX = 0;
  for (let y = top; y < Math.min(160, top + 16); y += 1) {
    for (let x = 10; x < 135; x += 1) {
      if (alpha(x, y) <= 128) continue;
      sx += x; count += 1; minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    }
  }
  return { top, dorsalCenter: Number((sx / count).toFixed(2)), dorsalWidth: maxX - minX };
}

// Lava Tick-specific landmark: the upper dome belongs to the rigid carapace.
// The restricted x window excludes the raised head horns, face, feet and jaw.
function crown(image, row, column) {
  const alpha = (x, y) => image.data[((row * 160 + y) * image.width + column * 160 + x) * 4 + 3];
  let top = 0;
  for (; top < 160; top += 1) {
    let count = 0;
    for (let x = 25; x < 108; x += 1) if (alpha(x, top) > 128) count += 1;
    if (count >= 6) break;
  }
  let sx = 0, count = 0;
  for (let y = top; y < Math.min(top + 5, 160); y += 1) {
    for (let x = 25; x < 108; x += 1) {
      if (alpha(x, y) <= 128) continue;
      sx += x; count += 1;
    }
  }
  return { x: Number((sx / count).toFixed(4)), y: top };
}

async function main() {
  const before = await load(beforePath);
  const targetPath = process.argv[2] ? path.resolve(ROOT, process.argv[2]) : beforePath;
  const image = targetPath === beforePath ? before : await load(targetPath);
  const template = shellTemplate(before);
  const rows = [];
  for (let row = 0; row < 8; row += 1) {
    const frames = [];
    for (let column = 0; column < 6; column += 1) frames.push({ column, ...contour(image, row, column), crown: crown(image, row, column), match: match(image, template, row, column) });
    rows.push({ row, action: actions[row], frames });
  }
  const output = path.join(__dirname, targetPath === beforePath ? 'before-measurements.json' : 'after-measurements.json');
  fs.writeFileSync(output, JSON.stringify({ template: 'Before idle frame 0, dorsal carapace rectangle x42..89 y75..108, every second pixel with alpha >128; excludes head, jaw and feet. The independently measured dorsal contour includes all solid pixels in the upper16px, not full actor bounds.', rows }, null, 2) + '\n');
  console.log(JSON.stringify(rows));
}
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
module.exports = { load, shellTemplate, match, contour, crown };
