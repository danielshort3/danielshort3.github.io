#!/usr/bin/env node
'use strict';

const GUIDE_LINE_HEX = '#00ffff';

function parseHexColor(hex) {
  const value = String(hex || GUIDE_LINE_HEX).trim().replace(/^#/, '');
  const normalized = value.length === 3
    ? value.split('').map((char) => `${char}${char}`).join('')
    : value.padEnd(6, '0').slice(0, 6);
  return {
    r: parseInt(normalized.slice(0, 2), 16),
    g: parseInt(normalized.slice(2, 4), 16),
    b: parseInt(normalized.slice(4, 6), 16)
  };
}

function isGuidePixelRgba(raw, offset, guideColor) {
  const color = typeof guideColor === 'string' ? parseHexColor(guideColor) : (guideColor || parseHexColor(GUIDE_LINE_HEX));
  const alpha = raw[offset + 3] == null ? 255 : raw[offset + 3];
  if (alpha < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const dr = Math.abs(r - color.r);
  const dg = Math.abs(g - color.g);
  const db = Math.abs(b - color.b);
  if (dr + dg + db <= 120 && Math.max(dr, dg, db) <= 86) return true;
  const cyanGuide = color.g > 170 && color.b > 170 && color.r < 90;
  return cyanGuide && r < 112 && g > 132 && b > 132 && Math.abs(g - b) < 112;
}

function scanGuideRuns(raw, width, height, axis, guideColor, minCoverage) {
  const length = axis === 'x' ? width : height;
  const span = axis === 'x' ? height : width;
  const coverageFloor = Math.max(0.05, Math.min(0.98, Number(minCoverage || 0.42)));
  const candidate = new Uint8Array(length);
  const scores = new Float32Array(length);
  for (let line = 0; line < length; line += 1) {
    let matches = 0;
    for (let along = 0; along < span; along += 1) {
      const x = axis === 'x' ? line : along;
      const y = axis === 'x' ? along : line;
      const offset = (y * width + x) * 4;
      if (isGuidePixelRgba(raw, offset, guideColor)) matches += 1;
    }
    const score = matches / Math.max(1, span);
    scores[line] = score;
    if (score >= coverageFloor) candidate[line] = 1;
  }

  const runs = [];
  let start = -1;
  for (let line = 0; line <= length; line += 1) {
    if (line < length && candidate[line]) {
      if (start < 0) start = line;
      continue;
    }
    if (start < 0) continue;
    const end = line - 1;
    let scoreTotal = 0;
    for (let index = start; index <= end; index += 1) scoreTotal += scores[index];
    runs.push({
      start,
      end,
      center: (start + end) / 2,
      thickness: end - start + 1,
      score: scoreTotal / Math.max(1, end - start + 1)
    });
    start = -1;
  }
  return runs;
}

function formatRuns(runs) {
  return runs.map((run) => `${Math.round(run.center)}(${run.start}-${run.end})`).join(', ');
}

function chooseGuideRuns(runs, expectedCount, total, label, axis) {
  if (runs.length === expectedCount) return runs;
  const expectedSpacing = total / Math.max(1, expectedCount - 1);
  const tolerance = Math.max(8, expectedSpacing * 0.22);
  if (runs.length > expectedCount) {
    const selected = [];
    const used = new Set();
    for (let index = 0; index < expectedCount; index += 1) {
      const target = index * (total - 1) / Math.max(1, expectedCount - 1);
      let best = null;
      let bestRunIndex = -1;
      let bestDistance = Infinity;
      runs.forEach((run, runIndex) => {
        if (used.has(runIndex)) return;
        const distance = Math.abs(run.center - target);
        if (distance > tolerance || distance >= bestDistance) return;
        best = run;
        bestRunIndex = runIndex;
        bestDistance = distance;
      });
      if (!best) break;
      used.add(bestRunIndex);
      selected.push(best);
    }
    if (selected.length === expectedCount) return selected.sort((a, b) => a.center - b.center);
  }
  const hint = runs.length ? ` Detected ${axis} guide centers: ${formatRuns(runs)}.` : '';
  throw new Error(`${label} must contain ${expectedCount} ${axis} guide lines, including outer borders; detected ${runs.length}.${hint}`);
}

function validateGuideSpacing(lines, expectedCells, total, label, axis) {
  if (lines.length !== expectedCells + 1) {
    throw new Error(`${label} ${axis} guide line count mismatch: expected ${expectedCells + 1}, received ${lines.length}`);
  }
  const spans = [];
  for (let index = 0; index < expectedCells; index += 1) {
    const start = lines[index];
    const end = lines[index + 1];
    const span = end.start - start.end - 1;
    if (span <= 1) {
      throw new Error(`${label} ${axis} cell ${index + 1} has no usable interior between guide lines`);
    }
    spans.push(span);
  }
  const sorted = spans.slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || total / Math.max(1, expectedCells);
  spans.forEach((span, index) => {
    if (span < median * 0.48 || span > median * 1.68) {
      throw new Error(`${label} ${axis} cell ${index + 1} has implausible guide spacing (${span}px vs median ${Math.round(median)}px)`);
    }
  });
}

function detectGuideGrid(raw, width, height, options) {
  const settings = options || {};
  const columns = Math.max(1, Number(settings.columns || 0));
  const rows = Math.max(1, Number(settings.rows || 0));
  if (!raw || !width || !height || !columns || !rows) {
    throw new Error('detectGuideGrid requires raw RGBA data, dimensions, columns, and rows');
  }
  const label = settings.label || 'sprite sheet';
  const guideColor = settings.guideColor || GUIDE_LINE_HEX;
  const minCoverage = settings.minCoverage || 0.42;
  const verticalRuns = scanGuideRuns(raw, width, height, 'x', guideColor, minCoverage);
  const horizontalRuns = scanGuideRuns(raw, width, height, 'y', guideColor, minCoverage);
  const verticalLines = chooseGuideRuns(verticalRuns, columns + 1, width, label, 'vertical');
  const horizontalLines = chooseGuideRuns(horizontalRuns, rows + 1, height, label, 'horizontal');
  validateGuideSpacing(verticalLines, columns, width, label, 'vertical');
  validateGuideSpacing(horizontalLines, rows, height, label, 'horizontal');
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    const cellRow = [];
    for (let col = 0; col < columns; col += 1) {
      cellRow.push(getGridCellRect({ verticalLines, horizontalLines }, row, col));
    }
    cells.push(cellRow);
  }
  return { columns, rows, width, height, verticalLines, horizontalLines, cells };
}

function getGridCellRect(grid, row, col, inset) {
  if (!grid || !grid.verticalLines || !grid.horizontalLines) {
    throw new Error('getGridCellRect requires a detected guide grid');
  }
  const pad = Math.max(0, Number(inset || 0));
  const leftLine = grid.verticalLines[col];
  const rightLine = grid.verticalLines[col + 1];
  const topLine = grid.horizontalLines[row];
  const bottomLine = grid.horizontalLines[row + 1];
  if (!leftLine || !rightLine || !topLine || !bottomLine) {
    throw new Error(`Guide grid is missing cell ${row + 1}:${col + 1}`);
  }
  const x = Math.round(leftLine.end + 1 + pad);
  const y = Math.round(topLine.end + 1 + pad);
  const right = Math.round(rightLine.start - pad);
  const bottom = Math.round(bottomLine.start - pad);
  return {
    x,
    y,
    w: Math.max(1, right - x),
    h: Math.max(1, bottom - y)
  };
}

function clearGuidePixels(raw, width, height, guideColor) {
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (!isGuidePixelRgba(raw, offset, guideColor || GUIDE_LINE_HEX)) continue;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
  }
  return raw;
}

function frameTouchesRectEdge(raw, width, x0, y0, frameWidth, frameHeight, threshold) {
  const minAlpha = threshold == null ? 8 : Number(threshold);
  for (let x = x0; x < x0 + frameWidth; x += 1) {
    if (raw[(y0 * width + x) * 4 + 3] > minAlpha) return true;
    if (raw[((y0 + frameHeight - 1) * width + x) * 4 + 3] > minAlpha) return true;
  }
  for (let y = y0; y < y0 + frameHeight; y += 1) {
    if (raw[(y * width + x0) * 4 + 3] > minAlpha) return true;
    if (raw[(y * width + x0 + frameWidth - 1) * 4 + 3] > minAlpha) return true;
  }
  return false;
}

module.exports = {
  GUIDE_LINE_HEX,
  parseHexColor,
  isGuidePixelRgba,
  detectGuideGrid,
  getGridCellRect,
  clearGuidePixels,
  frameTouchesRectEdge
};
