'use strict';

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
async function readActors(file, columns, rows) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const visited = new Uint8Array(info.width * info.height);
  const queue = new Int32Array(visited.length);
  const components = [];
  for (let start = 0; start < visited.length; start += 1) {
    if (visited[start] || data[start * 4 + 3] < 32) continue;
    let count = 0, end = 1, x0 = info.width, y0 = info.height, x1 = 0, y1 = 0;
    queue[0] = start; visited[start] = 1;
    for (let offset = 0; offset < end; offset += 1) {
      const pixel = queue[offset], x = pixel % info.width, y = Math.floor(pixel / info.width);
      count += 1; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
      for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        const nx = x + dx, ny = y + dy, next = ny * info.width + nx;
        if (nx < 0 || nx >= info.width || ny < 0 || ny >= info.height || visited[next] || data[next * 4 + 3] < 32) continue;
        visited[next] = 1; queue[end++] = next;
      }
    }
    if (count >= 500) components.push({ left: x0, top: y0, width: x1 - x0 + 1, height: y1 - y0 + 1, count, componentPixels: queue.slice(0, end) });
  }
  const expected = columns * rows;
  if (components.length !== expected) throw new Error(`Expected ${expected} disconnected actors in ${file}; found ${components.length}. Review components before export.`);
  components.sort((a, b) => a.top + a.height / 2 - b.top - b.height / 2);
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    const group = components.slice(row * columns, (row + 1) * columns).sort((a, b) => a.left - b.left);
    group.forEach(({ componentPixels, ...bounds }, column) => cells.push({ row, column, bounds, componentPixels, visible: bounds.count }));
  }
  return { data, info, cells, file };
}
async function readCells(file, columns, rows, rowCuts) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const result = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const left = Math.round(column * info.width / columns);
      const top = rowCuts ? rowCuts[row] : Math.round(row * info.height / rows);
      const right = Math.round((column + 1) * info.width / columns);
      const bottom = rowCuts ? rowCuts[row + 1] : Math.round((row + 1) * info.height / rows);
      let x0 = right, y0 = bottom, x1 = left, y1 = top, visible = 0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) {
          if (data[(y * info.width + x) * 4 + 3] < 32) continue;
          x0 = Math.min(x0, x); y0 = Math.min(y0, y);
          x1 = Math.max(x1, x); y1 = Math.max(y1, y); visible += 1;
        }
      }
      if (!visible) throw new Error(`Empty sprite cell: ${file} ${row},${column}`);
      const bounds = { left: x0, top: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
      result.push({ row, column, bounds, cell: { left, top, width: right - left, height: bottom - top }, visible });
    }
  }
  return { data, info, cells: result, file };
}

// Scale is authored once per source, never computed from an individual pose.
// Cropping and translation preserve genuine changes in pose silhouette.
async function packCell(source, cell, options) {
  const size = options.size || 160;
  const scale = options.scale;
  let input = sharp(source.file).extract({ left: cell.bounds.left, top: cell.bounds.top, width: cell.bounds.width, height: cell.bounds.height });
  if (cell.componentPixels) {
    const isolated = Buffer.alloc(cell.bounds.width * cell.bounds.height * 4);
    for (const pixel of cell.componentPixels) {
      const x = pixel % source.info.width, y = Math.floor(pixel / source.info.width);
      for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        const lx = x + dx - cell.bounds.left, ly = y + dy - cell.bounds.top;
        if (lx < 0 || ly < 0 || lx >= cell.bounds.width || ly >= cell.bounds.height) continue;
        const offset = ((y + dy) * source.info.width + x + dx) * 4;
        if ((dx || dy) && source.data[offset + 3] >= 32) continue;
        source.data.copy(isolated, (ly * cell.bounds.width + lx) * 4, offset, offset + 4);
      }
    }
    input = sharp(isolated, { raw: { width: cell.bounds.width, height: cell.bounds.height, channels: 4 } });
  }
  const buffer = await input
    .resize(Math.round(cell.bounds.width * scale), Math.round(cell.bounds.height * scale), { kernel: 'lanczos3' })
    .ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  for (let offset = 0; offset < buffer.data.length; offset += 4) {
    if (buffer.data[offset + 3] < 8) buffer.data.fill(0, offset, offset + 4);
  }
  const rootX = options.rootX == null ? (cell.bounds.left + cell.bounds.width / 2) : options.rootX;
  const groundY = options.groundY == null ? cell.bounds.top + cell.bounds.height : options.groundY;
  const left = Math.round((options.originX == null ? 80 : options.originX) - (rootX - cell.bounds.left) * scale);
  const top = Math.round((options.baseline == null ? 154 : options.baseline) - (groundY - cell.bounds.top) * scale);
  if (left < 1 || top < 1 || left + buffer.info.width > size - 1 || top + buffer.info.height > size - 1) {
    throw new Error(`Clipped sprite ${source.file} cell ${cell.row},${cell.column}: ${JSON.stringify({ left, top, width: buffer.info.width, height: buffer.info.height })}`);
  }
  const png = await sharp({ create: { width: size, height: size, channels: 4, background: '#00000000' } })
    .composite([{ input: buffer.data, raw: { width: buffer.info.width, height: buffer.info.height, channels: 4 }, left, top }]).png().toBuffer();
  return { png, transform: { scale, left, top, sourceLeft: cell.bounds.left, sourceTop: cell.bounds.top }, rootX, groundY };
}

async function writeAtlas(frames, output, columns, rows, size = 160) {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  await sharp({ create: { width: size * columns, height: size * rows, channels: 4, background: '#00000000' } })
    .composite(frames.map((frame, index) => ({ input: frame.png || frame, left: (index % columns) * size, top: Math.floor(index / columns) * size })))
    .png({ compressionLevel: 9 }).toFile(output);
}

module.exports = { sha256, readActors, readCells, packCell, writeAtlas };
