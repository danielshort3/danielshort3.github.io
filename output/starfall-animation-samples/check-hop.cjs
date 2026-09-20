'use strict';

// Read-only pixel inspection. Writes only a separate JSON report, never the PNG.
// Usage: node output/starfall-animation-samples/check-hop.cjs [input.png] [report.json] [layout.json]
// layout.json may contain frames[].suggestedSourceRect/sourceRect or sourceRects[].
// With no rectangles, its nominalGrid/columns/rows can override the default 4x4.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const sharp = require('sharp');

const inputPath = path.resolve(process.argv[2] || path.join(__dirname, 'glowcap-hop-study.png'));
const reportPath = path.resolve(process.argv[3] || path.join(__dirname, 'glowcap-hop-check.json'));
const layoutPath = process.argv[4] ? path.resolve(process.argv[4]) : null;
const GRID_COLUMNS = 4;
const GRID_ROWS = 4;
const ALPHA_THRESHOLD = 16;
const GUTTER_PIXELS = 4;

function connectedComponents(rgba, width, height, threshold = ALPHA_THRESHOLD) {
  const visited = new Uint8Array(width * height);
  const queue = new Int32Array(width * height);
  const components = [];
  for (let start = 0; start < visited.length; start += 1) {
    if (visited[start] || rgba[start * 4 + 3] < threshold) continue;
    let read = 0;
    let write = 1;
    queue[0] = start;
    visited[start] = 1;
    let left = width;
    let top = height;
    let right = -1;
    let bottom = -1;
    let maxAlpha = 0;
    let alphaMass = 0;
    while (read < write) {
      const index = queue[read++];
      const x = index % width;
      const y = Math.floor(index / width);
      maxAlpha = Math.max(maxAlpha, rgba[index * 4 + 3]);
      alphaMass += rgba[index * 4 + 3] / 255;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if ((!dx && !dy) || nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (visited[next] || rgba[next * 4 + 3] < threshold) continue;
          visited[next] = 1;
          queue[write++] = next;
        }
      }
    }
    components.push({ pixels: write, maxAlpha, alphaMass, bboxExclusive: [left, top, right + 1, bottom + 1] });
  }
  components.sort((a, b) => b.pixels - a.pixels);
  return {
    connectivity: 8,
    alphaThreshold: threshold,
    count: components.length,
    largest: components[0] || null,
    detachedVisiblePixels: components.slice(1).reduce((sum, item) => sum + item.pixels, 0),
    detachedComponentsAtLeast4Pixels: components.slice(1).filter((item) => item.pixels >= 4),
    caution: 'Disconnected pixels may be intended decoration or loose artifacts; this report does not remove them.'
  };
}

function hash(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function inspectCell(raw, sourceWidth, bounds, frame) {
  const { left, top, width, height } = bounds;
  const rgba = Buffer.alloc(width * height * 4);
  const visibleRgba = Buffer.alloc(rgba.length);
  const edges = { top: 0, bottom: 0, left: 0, right: 0 };
  const gutter = { top: 0, bottom: 0, left: 0, right: 0 };
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let visiblePixels = 0;
  let alphaSum = 0;
  let weightedX = 0;
  let weightedY = 0;
  let transparentPixels = 0;
  let partialAlphaPixels = 0;
  for (let y = 0; y < height; y += 1) {
    const sourceStart = ((top + y) * sourceWidth + left) * 4;
    raw.copy(rgba, y * width * 4, sourceStart, sourceStart + width * 4);
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const alpha = rgba[index + 3];
      alphaSum += alpha;
      weightedX += x * alpha;
      weightedY += y * alpha;
      if (alpha === 0) transparentPixels += 1;
      if (alpha > 0 && alpha < 255) partialAlphaPixels += 1;
      // Ignore invisible RGB when checking exact visible duplicates.
      if (alpha > 0) rgba.copy(visibleRgba, index, index, index + 4);
      if (alpha < ALPHA_THRESHOLD) continue;
      visiblePixels += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      if (y === 0) edges.top += 1;
      if (y === height - 1) edges.bottom += 1;
      if (x === 0) edges.left += 1;
      if (x === width - 1) edges.right += 1;
      if (y < GUTTER_PIXELS) gutter.top += 1;
      if (y >= height - GUTTER_PIXELS) gutter.bottom += 1;
      if (x < GUTTER_PIXELS) gutter.left += 1;
      if (x >= width - GUTTER_PIXELS) gutter.right += 1;
    }
  }
  return {
    frame,
    row: Math.floor(frame / GRID_COLUMNS),
    column: frame % GRID_COLUMNS,
    sourceRect: bounds,
    bboxExclusive: visiblePixels ? [minX, minY, maxX + 1, maxY + 1] : null,
    visibleWidth: visiblePixels ? maxX - minX + 1 : 0,
    visibleHeight: visiblePixels ? maxY - minY + 1 : 0,
    visiblePixels,
    visibleFraction: visiblePixels / (width * height),
    alphaMassPixels: alphaSum / 255,
    alphaWeightedCentroid: alphaSum ? [weightedX / alphaSum, weightedY / alphaSum] : null,
    transparentPixels,
    partialAlphaPixels,
    components: connectedComponents(rgba, width, height),
    faintAlphaComponents: connectedComponents(rgba, width, height, 1),
    edgePixelsBySide: edges,
    edgeContactSides: Object.keys(edges).filter((key) => edges[key] > 0),
    gutterPixelsBySide: gutter,
    gutterContactSides: Object.keys(gutter).filter((key) => gutter[key] > 0),
    exactRgbaSha256: hash(rgba),
    visibleRgbaSha256: hash(visibleRgba)
  };
}

async function main() {
  if (inputPath === reportPath) throw new Error('Report path must differ from input image path.');
  if (layoutPath === reportPath) throw new Error('Report path must differ from layout metadata path.');
  if (!fs.existsSync(inputPath)) throw new Error(`Candidate has not arrived: ${inputPath}`);
  const bytes = fs.readFileSync(inputPath);
  const inputHash = hash(bytes);
  const metadata = await sharp(bytes).metadata();
  const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  if (info.channels !== 4) throw new Error(`Expected RGBA data, found ${info.channels} channels.`);
  const layout = layoutPath ? JSON.parse(fs.readFileSync(layoutPath, 'utf8').replace(/^\uFEFF/, '')) : {};
  const columns = Number(layout.nominalGrid?.columns || layout.grid?.columns || layout.columns || GRID_COLUMNS);
  const rows = Number(layout.nominalGrid?.rows || layout.grid?.rows || layout.rows || GRID_ROWS);
  if (![columns, rows].every((value) => Number.isInteger(value) && value > 0)) throw new Error('Grid columns/rows must be positive integers.');
  const manualFrames = Array.isArray(layout.sourceRects) ? layout.sourceRects : Array.isArray(layout.frames) && layout.frames.some((frame) => frame.suggestedSourceRect || frame.sourceRect || frame.rect) ? layout.frames : null;
  const gridDividesExactly = info.width % columns === 0 && info.height % rows === 0;
  const cells = [];
  const frames = manualFrames || Array.from({ length: rows * columns }, (_, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;
    const left = Math.floor(column * info.width / columns);
    const top = Math.floor(row * info.height / rows);
    const right = Math.floor((column + 1) * info.width / columns);
    const bottom = Math.floor((row + 1) * info.height / rows);
    return { row, column, sourceRect: { left, top, width: right - left, height: bottom - top } };
  });
  const coverage = new Uint8Array(info.width * info.height);
  for (let index = 0; index < frames.length; index += 1) {
    const frame = frames[index];
    const bounds = frame.suggestedSourceRect || frame.sourceRect || frame.rect || frame;
    if (![bounds.left, bounds.top, bounds.width, bounds.height].every(Number.isInteger) || bounds.left < 0 || bounds.top < 0 || bounds.width < 1 || bounds.height < 1 || bounds.left + bounds.width > info.width || bounds.top + bounds.height > info.height) throw new Error(`Invalid source rectangle at frame ${index}: ${JSON.stringify(bounds)}`);
    const cell = inspectCell(data, info.width, bounds, frame.index ?? frame.frame ?? index);
    cell.row = frame.row ?? Math.floor(index / columns);
    cell.column = frame.column ?? frame.col ?? index % columns;
    cells.push(cell);
    for (let y = bounds.top; y < bounds.top + bounds.height; y += 1) {
      for (let x = bounds.left; x < bounds.left + bounds.width; x += 1) coverage[y * info.width + x] = Math.min(255, coverage[y * info.width + x] + 1);
    }
  }
  const coverageReport = { overlappingPixels: 0, overlappingVisiblePixels: 0, uncoveredPixels: 0, uncoveredVisiblePixels: 0 };
  for (let index = 0; index < coverage.length; index += 1) {
    const visible = data[index * 4 + 3] >= ALPHA_THRESHOLD;
    if (coverage[index] > 1) {
      coverageReport.overlappingPixels += 1;
      if (visible) coverageReport.overlappingVisiblePixels += 1;
    } else if (coverage[index] === 0) {
      coverageReport.uncoveredPixels += 1;
      if (visible) coverageReport.uncoveredVisiblePixels += 1;
    }
  }
  const duplicateGroups = [];
  const byHash = new Map();
  for (const cell of cells) {
    const frames = byHash.get(cell.visibleRgbaSha256) || [];
    frames.push(cell.frame);
    byHash.set(cell.visibleRgbaSha256, frames);
  }
  for (const frames of byHash.values()) if (frames.length > 1) duplicateGroups.push(frames);
  const warnings = [];
  if (!gridDividesExactly) warnings.push(`Image dimensions do not divide evenly into the nominal ${columns}x${rows} grid; ${manualFrames ? 'explicit measured rectangles were used' : 'inspect unequal cell rectangles'}.`);
  if (!metadata.hasAlpha) warnings.push('Source has no alpha channel. ensureAlpha added opaque alpha for measurement, not transparency.');
  if (!cells.some((cell) => cell.transparentPixels > 0)) warnings.push('No fully transparent pixels: a painted/checker/solid background may be baked into the image.');
  if (cells.some((cell) => !cell.visiblePixels)) warnings.push('At least one nominal cell is empty.');
  if (cells.some((cell) => cell.edgeContactSides.length)) warnings.push('At least one pose touches a nominal cell edge; inspect for clipping or background contamination.');
  if (duplicateGroups.length) warnings.push('At least two nominal cells contain byte-identical visible pixels.');
  if (coverageReport.overlappingVisiblePixels) warnings.push('Explicit rectangles overlap visible source pixels.');
  if (coverageReport.uncoveredVisiblePixels) warnings.push('Visible source pixels are outside all selected rectangles.');
  if (cells.some((cell) => cell.components.detachedComponentsAtLeast4Pixels.length)) warnings.push('Some poses have detached alpha components of at least four pixels; inspect for particles/strays.');
  const report = {
    inputPath,
    inputSha256: inputHash,
    generatedAt: new Date().toISOString(),
    source: { width: info.width, height: info.height, format: metadata.format, hasAlphaChannel: metadata.hasAlpha, channels: metadata.channels, bytes: bytes.length },
    nominalGrid: { columns, rows, frameOrder: 'row-major', dividesExactly: gridDividesExactly, alphaThreshold: ALPHA_THRESHOLD, gutterPixels: GUTTER_PIXELS },
    layout: { mode: manualFrames ? 'explicit-source-rectangles' : 'uniform-grid', metadataPath: layoutPath, measuredFrames: cells.length, coverage: coverageReport },
    warnings,
    cells,
    sequences: {
      visibleWidth: cells.map((cell) => cell.visibleWidth),
      visibleHeight: cells.map((cell) => cell.visibleHeight),
      visibleArea: cells.map((cell) => cell.visiblePixels),
      alphaMass: cells.map((cell) => cell.alphaMassPixels)
    },
    exactVisibleDuplicateGroups: duplicateGroups,
    reviewLimits: [
      'Dimension and pixel checks cannot establish coherent limb counts, cap-spot identity, deliberate poses, or loop quality; inspect every frame visually.',
      'Widths, heights and area should change during a deliberately authored hop. Do not normalize or rescale individual cells from these metrics.',
      'The checker does not crop, rewrite, rescale, clean alpha, or otherwise modify the candidate.'
    ]
  };
  if (hash(fs.readFileSync(inputPath)) !== inputHash) throw new Error('Candidate changed while it was inspected; rerun against a stable file.');
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ reportPath, source: report.source, warnings, sequences: report.sequences, exactVisibleDuplicateGroups: duplicateGroups, edgeContactFrames: cells.filter((cell) => cell.edgeContactSides.length).map((cell) => cell.frame), gutterContactFrames: cells.filter((cell) => cell.gutterContactSides.length).map((cell) => cell.frame) }, null, 2));
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
