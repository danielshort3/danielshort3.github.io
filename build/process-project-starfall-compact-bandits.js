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
const Data = require('../js/games/project-starfall/project-starfall-data.js');

const ROOT = path.resolve(__dirname, '..');
const RAW_DIR = path.join(ROOT, 'asset-sources/project-starfall/enemies/compact');
const ENEMY_OUT_DIR = path.join(ROOT, 'img/project-starfall/animations/enemies');
const PORTRAIT_OUT_DIR = path.join(ROOT, 'img/project-starfall/enemies');
const PROJECTILE_OUT_DIR = path.join(ROOT, 'img/project-starfall/animations/enemy-projectiles');
const ENEMY_FRAME = 128;
const PROJECTILE_FRAME = 64;
const ENEMY_COLUMNS = 3;
const ENEMY_ROWS = 8;
const PROJECTILE_COLUMNS = 3;
const TRANSPARENT = Object.freeze({ r: 0, g: 0, b: 0, alpha: 0 });
const ALPHA_THRESHOLD = 8;
const BODY_TARGET_HEIGHT = 102;
const BODY_MAX_WIDTH = 110;
const BODY_BASELINE = 118;
const DEFEAT_BASELINE = 116;
const SAFE_FRAME_GUTTER = 4;
const COMPONENT_NEAR_BODY_PX = 32;
const SOURCE_CELL_INSET = 8;
const PROJECTILE_SOURCE_CELL_INSET = 4;
const STABLE_SCALE_ROWS = Object.freeze([0, 1, 2, 3, 6]);
const STABLE_QUALITY_ROWS = Object.freeze([0, 1, 6]);

function getFileIdFromAsset(assetPath) {
  return path.basename(String(assetPath || ''), '.png');
}

function getEnemySources() {
  return Object.freeze((Data.ENEMIES || []).map((enemy) => {
    const fileId = getFileIdFromAsset(enemy.asset);
    return Object.freeze({
      id: enemy.id,
      name: enemy.name,
      family: enemy.family,
      role: enemy.role,
      behavior: enemy.behavior,
      mechanic: enemy.mechanic,
      fileId,
      source: `${fileId}-compact-source.png`,
      standardSheet: `${fileId}-sheet.png`,
      sheet: `${fileId}-compact-sheet.png`,
      portrait: `${fileId}.png`,
      portraitColumn: 0
    });
  }));
}

function getProjectileSources() {
  return Object.freeze(Object.entries(Data.ENEMY_PROJECTILE_ANIMATION_ASSETS || {}).map(([enemyId, animation]) => {
    const fileId = path.basename(String(animation.sheet || ''), '-sheet.png');
    return Object.freeze({
      id: enemyId,
      fileId,
      source: `${fileId}-source.png`,
      sheet: `${fileId}-sheet.png`
    });
  }).filter((source) => source.fileId));
}

function shouldUseMagentaKey(source) {
  const text = `${source && source.name || ''} ${source && source.family || ''} ${source && source.fileId || ''}`.toLowerCase();
  return text.includes('ooze') ||
    text.includes('slime') ||
    text.includes('plant') ||
    text.includes('moss') ||
    text.includes('thorn') ||
    text.includes('vine') ||
    text.includes('briar') ||
    text.includes('bloom') ||
    text.includes('cap');
}

function getChromaKeyHex(source) {
  return shouldUseMagentaKey(source) ? '#ff00ff' : '#00ff00';
}

function detectChromaKeyMode(raw, width, height, source) {
  let green = 0;
  let magenta = 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (raw[offset + 3] < 8) continue;
    const r = raw[offset];
    const g = raw[offset + 1];
    const b = raw[offset + 2];
    if (g > 180 && g > r * 1.45 && g > b * 1.45) green += 1;
    if (r > 175 && b > 175 && g < 120 && r > g * 1.45 && b > g * 1.45) magenta += 1;
  }
  if (green === magenta) return shouldUseMagentaKey(source) ? 'magenta' : 'green';
  return magenta > green ? 'magenta' : 'green';
}

function isRemovedColor(raw, offset, source) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return true;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  if (shouldUseMagentaKey(source)) {
    if (r > 145 && b > 145 && g < 130 && Math.abs(r - b) < 120) return true;
    if (r > 45 && b > 45 && g < 105 && r > g * 1.35 && b > g * 1.35 && Math.abs(r - b) < 100) return true;
    if (r > 35 && b > 30 && g < 112 && r > g * 1.25 && b > g * 1.2 && Math.abs(r - b) < 170) return true;
    if (r > 120 && b > 60 && g < 80) return true;
  } else if (g > 120 && g > r * 1.45 && g > b * 1.45) {
    return true;
  }
  if (r < 90 && g > 145 && b > 145 && Math.abs(g - b) < 80) return true;
  return false;
}

function isConnectedBackgroundCandidate(raw, offset, chromaMode) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  if (chromaMode === 'magenta') {
    return r > 118 && b > 118 && g < 132 &&
      r > g * 1.35 && b > g * 1.35 && Math.abs(r - b) < 126;
  }
  return g > 105 && g - r > 46 && g - b > 46 &&
    g > r * 1.32 && g > b * 1.32;
}

function isExactChromaKeyPixel(raw, offset, chromaMode) {
  if (raw[offset + 3] < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  return chromaMode === 'magenta'
    ? r > 235 && b > 235 && g < 28
    : g > 235 && r < 28 && b < 28;
}

function isGuideArtifactPixel(raw, offset) {
  const alpha = raw[offset + 3];
  if (alpha < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  return r < 105 && g > 138 && b > 138 && Math.abs(g - b) < 88;
}

function removeChromaAndGuides(raw, width, height, source, sourceInfo) {
  const pixelCount = width * height;
  const candidates = new Uint8Array(pixelCount);
  const queued = new Uint8Array(pixelCount);
  const queue = [];
  const exactSeeds = [];
  const chromaMode = sourceInfo && sourceInfo.chromaMode || detectChromaKeyMode(raw, width, height, source);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const offset = pixel * 4;
    if (isConnectedBackgroundCandidate(raw, offset, chromaMode) || isGuideArtifactPixel(raw, offset)) {
      candidates[pixel] = 1;
    }
    if (isExactChromaKeyPixel(raw, offset, chromaMode)) exactSeeds.push(pixel);
  }
  const enqueue = (pixel) => {
    if (pixel < 0 || pixel >= pixelCount || queued[pixel] || !candidates[pixel]) return;
    queued[pixel] = 1;
    queue.push(pixel);
  };
  exactSeeds.forEach(enqueue);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const pixel = y * width + x;
      if (!candidates[pixel]) continue;
      const edge = x === 0 || x === width - 1 || y === 0 || y === height - 1;
      const besideTransparentGuide = (!edge && (
        raw[(pixel - 1) * 4 + 3] < 8 ||
        raw[(pixel + 1) * 4 + 3] < 8 ||
        raw[(pixel - width) * 4 + 3] < 8 ||
        raw[(pixel + width) * 4 + 3] < 8
      ));
      if (edge || besideTransparentGuide) enqueue(pixel);
    }
  }
  if (sourceInfo && sourceInfo.grid && Array.isArray(sourceInfo.grid.cells)) {
    sourceInfo.grid.cells.forEach((row) => row.forEach((rect) => {
      for (let x = rect.x; x < rect.x + rect.w; x += 1) {
        enqueue(rect.y * width + x);
        enqueue((rect.y + rect.h - 1) * width + x);
      }
      for (let y = rect.y; y < rect.y + rect.h; y += 1) {
        enqueue(y * width + rect.x);
        enqueue(y * width + rect.x + rect.w - 1);
      }
    }));
  }
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const pixel = queue[cursor];
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const offset = pixel * 4;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
    if (x > 0) enqueue(pixel - 1);
    if (x + 1 < width) enqueue(pixel + 1);
    if (y > 0) enqueue(pixel - width);
    if (y + 1 < height) enqueue(pixel + width);
  }
}

function getAlphaBounds(raw, width, x0, y0, w, h, threshold) {
  let minX = x0 + w;
  let minY = y0 + h;
  let maxX = x0 - 1;
  let maxY = y0 - 1;
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
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

function cropRaw(raw, width, bounds) {
  const cropped = Buffer.alloc(bounds.w * bounds.h * 4);
  for (let y = 0; y < bounds.h; y += 1) {
    const sourceStart = ((bounds.y + y) * width + bounds.x) * 4;
    raw.copy(cropped, y * bounds.w * 4, sourceStart, sourceStart + bounds.w * 4);
  }
  return cropped;
}

function pasteRaw(target, targetWidth, targetHeight, source, sourceWidth, sourceHeight, x0, y0) {
  for (let y = 0; y < sourceHeight; y += 1) {
    const targetY = y0 + y;
    if (targetY < 0 || targetY >= targetHeight) continue;
    for (let x = 0; x < sourceWidth; x += 1) {
      const targetX = x0 + x;
      if (targetX < 0 || targetX >= targetWidth) continue;
      const sourceOffset = (y * sourceWidth + x) * 4;
      const alpha = source[sourceOffset + 3];
      if (alpha <= 0) continue;
      const targetOffset = (targetY * targetWidth + targetX) * 4;
      source.copy(target, targetOffset, sourceOffset, sourceOffset + 4);
    }
  }
}

function clearOuterBorder(raw, width, height, frameWidth, frameHeight, columns, rows) {
  const clearPixel = (pixelOffset) => {
    raw[pixelOffset] = 0;
    raw[pixelOffset + 1] = 0;
    raw[pixelOffset + 2] = 0;
    raw[pixelOffset + 3] = 0;
  };
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const x0 = col * frameWidth;
      const y0 = row * frameHeight;
      for (let x = x0; x < x0 + frameWidth; x += 1) {
        clearPixel((y0 * width + x) * 4);
        clearPixel(((y0 + frameHeight - 1) * width + x) * 4);
      }
      for (let y = y0; y < y0 + frameHeight; y += 1) {
        clearPixel((y * width + x0) * 4);
        clearPixel((y * width + x0 + frameWidth - 1) * 4);
      }
    }
  }
}

function clearTransparentColorData(raw, width, height) {
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (raw[offset + 3] > ALPHA_THRESHOLD) continue;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
  }
}

function clearOutputChromaSpill(raw, chromaMode) {
  for (let offset = 0; offset < raw.length; offset += 4) {
    if (raw[offset + 3] <= ALPHA_THRESHOLD) continue;
    const r = raw[offset];
    const g = raw[offset + 1];
    const b = raw[offset + 2];
    const spill = chromaMode === 'magenta'
      ? Math.min(r, b) > 190 && g < 85 && Math.min(r, b) - g > 120
      : g > 190 && g - r > 105 && g - b > 105;
    if (!spill) continue;
    raw[offset] = 0;
    raw[offset + 1] = 0;
    raw[offset + 2] = 0;
    raw[offset + 3] = 0;
  }
}

function clearRawRect(raw, width, rect) {
  for (let y = rect.y; y < rect.y + rect.h; y += 1) {
    raw.fill(0, (y * width + rect.x) * 4, (y * width + rect.x + rect.w) * 4);
  }
}

function clearFrameGutter(raw, frameSize, gutter) {
  for (let y = 0; y < frameSize; y += 1) {
    for (let x = 0; x < frameSize; x += 1) {
      if (x >= gutter && x < frameSize - gutter && y >= gutter && y < frameSize - gutter) continue;
      const offset = (y * frameSize + x) * 4;
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    }
  }
}

function getSourceCellRect(sourceInfo, row, col, inset = SOURCE_CELL_INSET) {
  if (!sourceInfo || !sourceInfo.grid) {
    throw new Error('Compact enemy source must be sliced from a detected guide grid');
  }
  return getGridCellRect(sourceInfo.grid, row, col, inset);
}

function findAlphaComponents(raw, width, rect, threshold) {
  const visited = new Uint8Array(rect.w * rect.h);
  const components = [];
  const stack = [];
  for (let startY = 0; startY < rect.h; startY += 1) {
    for (let startX = 0; startX < rect.w; startX += 1) {
      const startIndex = startY * rect.w + startX;
      if (visited[startIndex]) continue;
      const startOffset = ((rect.y + startY) * width + rect.x + startX) * 4;
      if (raw[startOffset + 3] <= threshold) {
        visited[startIndex] = 1;
        continue;
      }
      visited[startIndex] = 1;
      stack.length = 0;
      stack.push(startIndex);
      let minX = startX;
      let minY = startY;
      let maxX = startX;
      let maxY = startY;
      let area = 0;
      while (stack.length) {
        const index = stack.pop();
        const x = index % rect.w;
        const y = Math.floor(index / rect.w);
        area += 1;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        [
          [x - 1, y],
          [x + 1, y],
          [x, y - 1],
          [x, y + 1]
        ].forEach(([nextX, nextY]) => {
          if (nextX < 0 || nextX >= rect.w || nextY < 0 || nextY >= rect.h) return;
          const nextIndex = nextY * rect.w + nextX;
          if (visited[nextIndex]) return;
          visited[nextIndex] = 1;
          const offset = ((rect.y + nextY) * width + rect.x + nextX) * 4;
          if (raw[offset + 3] > threshold) stack.push(nextIndex);
        });
      }
      components.push({
        x: rect.x + minX,
        y: rect.y + minY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        area
      });
    }
  }
  return components;
}

function rectDistance(a, b) {
  const dx = Math.max(0, Math.max(a.x - (b.x + b.w), b.x - (a.x + a.w)));
  const dy = Math.max(0, Math.max(a.y - (b.y + b.h), b.y - (a.y + a.h)));
  return Math.hypot(dx, dy);
}

function unionBounds(boundsList) {
  const bounds = boundsList.filter(Boolean);
  if (!bounds.length) return null;
  const minX = Math.min(...bounds.map((entry) => entry.x));
  const minY = Math.min(...bounds.map((entry) => entry.y));
  const maxX = Math.max(...bounds.map((entry) => entry.x + entry.w - 1));
  const maxY = Math.max(...bounds.map((entry) => entry.y + entry.h - 1));
  return { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };
}

function expandBounds(bounds, rect, pad) {
  const x = Math.max(rect.x, bounds.x - pad);
  const y = Math.max(rect.y, bounds.y - pad);
  const right = Math.min(rect.x + rect.w, bounds.x + bounds.w + pad);
  const bottom = Math.min(rect.y + rect.h, bounds.y + bounds.h + pad);
  return { x, y, w: Math.max(1, right - x), h: Math.max(1, bottom - y) };
}

function chooseBodyComponent(components, rect) {
  const candidates = components.filter((component) =>
    component.area >= 80 &&
    component.h >= rect.h * 0.22 &&
    component.w >= rect.w * 0.04
  );
  const pool = candidates.length ? candidates : components;
  return pool.slice().sort((a, b) => {
    const scoreA = a.area + a.h * 90 + a.w * 12 - (a.w > rect.w * 0.76 && a.h < rect.h * 0.36 ? 4000 : 0);
    const scoreB = b.area + b.h * 90 + b.w * 12 - (b.w > rect.w * 0.76 && b.h < rect.h * 0.36 ? 4000 : 0);
    return scoreB - scoreA;
  })[0] || null;
}

function isThinLineArtifact(component) {
  return component.h <= 4 && component.w >= 8 && component.w / Math.max(1, component.h) >= 3;
}

function isCellEdgeBleedArtifact(component, rect) {
  const nearEdge = component.x <= rect.x + 2 ||
    component.y <= rect.y + 2 ||
    component.x + component.w >= rect.x + rect.w - 2 ||
    component.y + component.h >= rect.y + rect.h - 2;
  const thin = component.h <= 8 || component.w <= 8;
  return nearEdge &&
    thin &&
    component.area <= 1200;
}

function clearComponentBounds(raw, width, component) {
  for (let y = component.y; y < component.y + component.h; y += 1) {
    for (let x = component.x; x < component.x + component.w; x += 1) {
      const offset = (y * width + x) * 4;
      raw[offset] = 0;
      raw[offset + 1] = 0;
      raw[offset + 2] = 0;
      raw[offset + 3] = 0;
    }
  }
}

function clearDetachedLineArtifacts(raw, width, frameWidth, frameHeight, columns, rows) {
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const rect = {
        x: col * frameWidth,
        y: row * frameHeight,
        w: frameWidth,
        h: frameHeight
      };
      const components = findAlphaComponents(raw, width, rect, ALPHA_THRESHOLD)
        .filter((component) => component.area >= 8);
      if (components.length < 2) continue;
      const primary = components.slice().sort((a, b) => b.area - a.area)[0];
      for (const component of components) {
        if (component === primary) continue;
        const horizontalLine = component.h <= 2 && component.w >= Math.max(24, frameWidth * 0.24);
        const verticalLine = component.w <= 2 && component.h >= Math.max(24, frameHeight * 0.24);
        if (!horizontalLine && !verticalLine) continue;
        const nearFrameEdge = component.x <= rect.x + 3 ||
          component.y <= rect.y + 3 ||
          component.x + component.w >= rect.x + rect.w - 3 ||
          component.y + component.h >= rect.y + rect.h - 3;
        const smallRelative = component.area <= primary.area * 0.22;
        if (nearFrameEdge || smallRelative) clearComponentBounds(raw, width, component);
      }
    }
  }
}

function isBodyAnchorPixel(raw, offset) {
  if (raw[offset + 3] <= ALPHA_THRESHOLD) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  if (r > 166 && g > 166 && b > 146 && max - min < 86) return false;
  if (r > 170 && g > 130 && b < 96) return false;
  if (r < 120 && g > 145 && b > 145) return false;
  return true;
}

function getBodyAnchorBounds(raw, width, rect) {
  const columnCounts = new Uint16Array(rect.w);
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
      columnCounts[x - rect.x] += 1;
    }
  }
  if (maxX < minX || maxY < minY) return null;
  let denseStart = -1;
  let denseEnd = -1;
  let runStart = -1;
  let runEnd = -1;
  let runScore = 0;
  let bestScore = 0;
  const maxColumnCount = Math.max(...columnCounts);
  const denseThreshold = Math.max(5, Math.floor(maxColumnCount * 0.18));
  for (let index = 0; index < rect.w; index += 1) {
    const dense = columnCounts[index] >= denseThreshold;
    if (dense) {
      if (runStart < 0) {
        runStart = index;
        runScore = 0;
      }
      runEnd = index;
      runScore += columnCounts[index];
      continue;
    }
    if (runStart >= 0 && runScore > bestScore) {
      bestScore = runScore;
      denseStart = runStart;
      denseEnd = runEnd;
    }
    runStart = -1;
    runEnd = -1;
    runScore = 0;
  }
  if (runStart >= 0 && runScore > bestScore) {
    denseStart = runStart;
    denseEnd = runEnd;
  }
  if (denseStart >= 0 && denseEnd >= denseStart) {
    const denseMinX = rect.x + denseStart;
    const denseMaxX = rect.x + denseEnd;
    minX = rect.x + rect.w;
    minY = rect.y + rect.h;
    maxX = rect.x - 1;
    maxY = rect.y - 1;
    area = 0;
    for (let y = rect.y; y < rect.y + rect.h; y += 1) {
      for (let x = denseMinX; x <= denseMaxX; x += 1) {
        const offset = (y * width + x) * 4;
        if (!isBodyAnchorPixel(raw, offset)) continue;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        area += 1;
      }
    }
  }
  if (maxX < minX || maxY < minY) return null;
  const bounds = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area };
  if (bounds.area < 200 || bounds.h < rect.h * 0.42) return null;
  return bounds;
}

function analyzeEnemyCell(sourceRaw, sourceInfo, row, col) {
  const rect = getSourceCellRect(sourceInfo, row, col);
  const components = findAlphaComponents(sourceRaw, sourceInfo.width, rect, ALPHA_THRESHOLD)
    .filter((component) => component.area >= 12 && !isCellEdgeBleedArtifact(component, rect));
  const bodyComponent = chooseBodyComponent(components, rect);
  if (!bodyComponent) return null;
  const bodyAnchor = getBodyAnchorBounds(sourceRaw, sourceInfo.width, rect);
  const body = bodyAnchor && rectDistance(bodyAnchor, bodyComponent) <= COMPONENT_NEAR_BODY_PX
    ? bodyAnchor
    : bodyComponent;
  const renderable = components.filter((component) =>
    !isThinLineArtifact(component) &&
    (component === bodyComponent ||
      rectDistance(component, bodyComponent) <= COMPONENT_NEAR_BODY_PX ||
      component.area >= bodyComponent.area * 0.18)
  );
  const renderBounds = expandBounds(unionBounds(renderable) || body, rect, 2);
  const bodyBottom = Math.max(body.y + body.h, bodyComponent.y + bodyComponent.h);
  return { row, col, rect, body, bodyBottom, renderBounds };
}

function getMedian(values) {
  const ordered = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
  if (!ordered.length) return 0;
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}

function getBoundsFitScale(analysis, bounds, baseline) {
  const anchorX = analysis.body.x + analysis.body.w / 2;
  const anchorY = analysis.bodyBottom;
  const left = Math.max(1, anchorX - bounds.x);
  const right = Math.max(1, bounds.x + bounds.w - anchorX);
  const top = Math.max(1, anchorY - bounds.y);
  const bottom = Math.max(0, bounds.y + bounds.h - anchorY);
  const horizontalSpace = ENEMY_FRAME / 2 - SAFE_FRAME_GUTTER;
  const scales = [
    horizontalSpace / left,
    horizontalSpace / right,
    (baseline - SAFE_FRAME_GUTTER) / top
  ];
  if (bottom > 0) scales.push((ENEMY_FRAME - SAFE_FRAME_GUTTER - baseline) / bottom);
  return Math.max(0.01, Math.min(...scales));
}

function getSharedEnemyScale(analyses) {
  const stable = analyses.filter((analysis) =>
    STABLE_SCALE_ROWS.includes(analysis.row)
  );
  const reference = stable.length ? stable : analyses.filter((analysis) => analysis.row !== ENEMY_ROWS - 1);
  const referenceHeight = Math.max(1, getMedian(reference.map((analysis) => analysis.body.h)));
  const referenceWidth = Math.max(1, getMedian(reference.map((analysis) => analysis.body.w)));
  const targetScale = Math.min(
    BODY_TARGET_HEIGHT / referenceHeight,
    BODY_MAX_WIDTH / referenceWidth
  );
  const bodyFitScale = Math.min(...analyses
    .filter((analysis) => analysis.row !== ENEMY_ROWS - 1)
    .map((analysis) => getBoundsFitScale(analysis, analysis.body, BODY_BASELINE)));
  return Math.max(0.01, Math.min(targetScale, bodyFitScale));
}

function constrainEnemyRenderBounds(analysis, scale, baseline) {
  const anchorX = analysis.body.x + analysis.body.w / 2;
  const anchorY = analysis.bodyBottom;
  const horizontalSpace = ENEMY_FRAME / 2 - SAFE_FRAME_GUTTER;
  const allowedLeft = anchorX - horizontalSpace / scale;
  const allowedRight = anchorX + horizontalSpace / scale;
  const allowedTop = anchorY - (baseline - SAFE_FRAME_GUTTER) / scale;
  const allowedBottom = anchorY + (ENEMY_FRAME - SAFE_FRAME_GUTTER - baseline) / scale;
  const renderRight = analysis.renderBounds.x + analysis.renderBounds.w;
  const renderBottom = analysis.renderBounds.y + analysis.renderBounds.h;
  const x = Math.max(analysis.renderBounds.x, Math.ceil(allowedLeft));
  const y = Math.max(analysis.renderBounds.y, Math.ceil(allowedTop));
  const right = Math.min(renderRight, Math.floor(allowedRight));
  const bottom = Math.min(renderBottom, Math.floor(allowedBottom));
  if (right <= x || bottom <= y) {
    throw new Error(`Unable to constrain ${analysis.row + 1}:${analysis.col + 1} to the safe frame gutter`);
  }
  return { x, y, w: right - x, h: bottom - y };
}

function featherClippedRenderEdges(raw, width, height, clippedEdges) {
  const feather = 3;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let opacity = 1;
      if (clippedEdges.left && x < feather) opacity = Math.min(opacity, x / feather);
      if (clippedEdges.right && width - 1 - x < feather) opacity = Math.min(opacity, (width - 1 - x) / feather);
      if (clippedEdges.top && y < feather) opacity = Math.min(opacity, y / feather);
      if (clippedEdges.bottom && height - 1 - y < feather) opacity = Math.min(opacity, (height - 1 - y) / feather);
      if (opacity >= 1) continue;
      const offset = (y * width + x) * 4;
      raw[offset + 3] = Math.round(raw[offset + 3] * Math.max(0, opacity));
    }
  }
}

async function drawAnalyzedCell(sourceRaw, sourceWidth, target, settings) {
  const analysis = settings.analysis;
  const scale = settings.scale;
  const renderBounds = settings.renderBounds || analysis.renderBounds;
  const cropped = cropRaw(sourceRaw, sourceWidth, renderBounds);
  const outW = Math.max(1, Math.round(renderBounds.w * scale));
  const outH = Math.max(1, Math.round(renderBounds.h * scale));
  const resized = await sharp(cropped, {
    raw: { width: renderBounds.w, height: renderBounds.h, channels: 4 }
  })
    .resize(outW, outH, { fit: 'fill', background: TRANSPARENT, kernel: 'lanczos3' })
    .raw()
    .toBuffer();
  featherClippedRenderEdges(resized, outW, outH, {
    left: renderBounds.x > analysis.renderBounds.x,
    right: renderBounds.x + renderBounds.w < analysis.renderBounds.x + analysis.renderBounds.w,
    top: renderBounds.y > analysis.renderBounds.y,
    bottom: renderBounds.y + renderBounds.h < analysis.renderBounds.y + analysis.renderBounds.h
  });
  const bodyCenterX = (analysis.body.x - renderBounds.x + analysis.body.w / 2) * scale;
  const bodyBottom = (analysis.bodyBottom - renderBounds.y) * scale;
  const frameX = settings.targetCol * settings.frameSize;
  const frameY = settings.targetRow * settings.frameSize;
  const minX = frameX + SAFE_FRAME_GUTTER;
  const maxX = frameX + settings.frameSize - SAFE_FRAME_GUTTER - outW;
  const minY = frameY + SAFE_FRAME_GUTTER;
  const maxY = frameY + settings.frameSize - SAFE_FRAME_GUTTER - outH;
  const targetX = Math.max(minX, Math.min(maxX, frameX + Math.round(settings.frameSize / 2 - bodyCenterX)));
  const targetY = Math.max(minY, Math.min(maxY, frameY + Math.round(settings.baseline - bodyBottom)));
  pasteRaw(target, settings.targetWidth, settings.targetHeight, resized, outW, outH, targetX, targetY);
  return true;
}

async function drawProjectileCell(sourceRaw, sourceInfo, target, col, settings) {
  const rect = getSourceCellRect(sourceInfo, 0, col, PROJECTILE_SOURCE_CELL_INSET);
  const components = findAlphaComponents(sourceRaw, sourceInfo.width, rect, ALPHA_THRESHOLD)
    .filter((component) => component.area >= 8);
  const primary = components.slice().sort((a, b) => b.area - a.area)[0];
  if (!primary) return false;
  const renderable = components.filter((component) => component === primary || rectDistance(component, primary) <= 22);
  const renderBounds = expandBounds(unionBounds(renderable) || primary, rect, 1);
  const cropped = cropRaw(sourceRaw, sourceInfo.width, renderBounds);
  const scale = Math.min(settings.maxWidth / Math.max(1, renderBounds.w), settings.maxHeight / Math.max(1, renderBounds.h));
  const outW = Math.max(1, Math.round(renderBounds.w * scale));
  const outH = Math.max(1, Math.round(renderBounds.h * scale));
  const resized = await sharp(cropped, {
    raw: { width: renderBounds.w, height: renderBounds.h, channels: 4 }
  })
    .resize(outW, outH, { fit: 'fill', background: TRANSPARENT, kernel: 'lanczos3' })
    .raw()
    .toBuffer();
  const targetX = col * PROJECTILE_FRAME + Math.round((PROJECTILE_FRAME - outW) / 2);
  const targetY = Math.round((PROJECTILE_FRAME - outH) / 2);
  pasteRaw(target, settings.targetWidth, settings.targetHeight, resized, outW, outH, targetX, targetY);
  return true;
}

async function writePng(raw, width, height, targetPath) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = path.join(path.dirname(targetPath), `.${path.basename(targetPath)}.${process.pid}.${Date.now()}.tmp.png`);
  try {
    await sharp(raw, {
      raw: { width, height, channels: 4 }
    })
      .png({ compressionLevel: 9, adaptiveFiltering: true })
      .toFile(tempPath);
    fs.renameSync(tempPath, targetPath);
  } finally {
    if (fs.existsSync(tempPath)) fs.rmSync(tempPath, { force: true });
  }
}

async function writePortraitFromSheet(sheetRaw, sheetWidth, source) {
  const portraitBounds = getAlphaBounds(sheetRaw, sheetWidth, source.portraitColumn * ENEMY_FRAME, 0, ENEMY_FRAME, ENEMY_FRAME, 8);
  if (!portraitBounds) return false;
  const cropped = cropRaw(sheetRaw, sheetWidth, portraitBounds);
  const portraitPath = path.join(PORTRAIT_OUT_DIR, source.portrait);
  const portrait = await sharp(cropped, {
    raw: { width: portraitBounds.w, height: portraitBounds.h, channels: 4 }
  })
    .resize(280, 280, { fit: 'contain', background: TRANSPARENT, kernel: 'lanczos3' })
    .extend({ top: 20, bottom: 20, left: 20, right: 20, background: TRANSPARENT })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toBuffer();
  fs.mkdirSync(path.dirname(portraitPath), { recursive: true });
  fs.writeFileSync(portraitPath, portrait);
  return true;
}

function getFrameBufferDiff(a, b, threshold) {
  let changed = 0;
  const length = Math.min(a.length, b.length);
  for (let offset = 0; offset < length; offset += 4) {
    const delta = Math.abs(a[offset] - b[offset]) +
      Math.abs(a[offset + 1] - b[offset + 1]) +
      Math.abs(a[offset + 2] - b[offset + 2]) +
      Math.abs(a[offset + 3] - b[offset + 3]);
    if (delta >= threshold) changed += 1;
  }
  return changed;
}

function shiftFrame(frame, frameSize, dx, dy) {
  const shifted = Buffer.alloc(frame.length);
  for (let y = 0; y < frameSize; y += 1) {
    for (let x = 0; x < frameSize; x += 1) {
      const sourceX = x - dx;
      const sourceY = y - dy;
      if (sourceX < 0 || sourceX >= frameSize || sourceY < 0 || sourceY >= frameSize) continue;
      const sourceOffset = (sourceY * frameSize + sourceX) * 4;
      const targetOffset = (y * frameSize + x) * 4;
      frame.copy(shifted, targetOffset, sourceOffset, sourceOffset + 4);
    }
  }
  return shifted;
}

function chooseDistinctFrames(frames, row) {
  const preferred = [0, 2, 4].map((index) => Math.min(frames.length - 1, index));
  const selected = preferred.map((index) => frames[index]);
  const minimumDiff = row === 0 ? 192 : 576;
  if (getFrameBufferDiff(selected[0], selected[1], 32) < minimumDiff) {
    let bestIndex = 1;
    let bestDiff = -1;
    frames.forEach((frame, index) => {
      if (index === 0) return;
      const diff = getFrameBufferDiff(selected[0], frame, 32);
      if (diff > bestDiff) {
        bestIndex = index;
        bestDiff = diff;
      }
    });
    selected[1] = bestDiff >= minimumDiff ? frames[bestIndex] : shiftFrame(selected[0], ENEMY_FRAME, row === ENEMY_ROWS - 1 ? 4 : 2, row === ENEMY_ROWS - 1 ? 1 : 0);
  }
  let bestThirdIndex = preferred[2];
  let bestThirdScore = -1;
  frames.forEach((frame, index) => {
    const score = Math.min(
      getFrameBufferDiff(selected[0], frame, 32),
      getFrameBufferDiff(selected[1], frame, 32)
    );
    if (score > bestThirdScore) {
      bestThirdIndex = index;
      bestThirdScore = score;
    }
  });
  selected[2] = bestThirdScore >= Math.round(minimumDiff * 0.5)
    ? frames[bestThirdIndex]
    : shiftFrame(selected[1], ENEMY_FRAME, row === ENEMY_ROWS - 1 ? 3 : -2, 0);
  return selected;
}

function ensureDistinctEnemyFrames(raw, width) {
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    let previous = cropRaw(raw, width, {
      x: 0,
      y: row * ENEMY_FRAME,
      w: ENEMY_FRAME,
      h: ENEMY_FRAME
    });
    for (let col = 1; col < ENEMY_COLUMNS; col += 1) {
      const rect = {
        x: col * ENEMY_FRAME,
        y: row * ENEMY_FRAME,
        w: ENEMY_FRAME,
        h: ENEMY_FRAME
      };
      let current = cropRaw(raw, width, rect);
      const area = getAlphaArea(current, ENEMY_FRAME, 0, 0, ENEMY_FRAME, ENEMY_FRAME, ALPHA_THRESHOLD);
      const duplicateLimit = Math.max(12, Math.round(area * 0.002));
      if (getFrameBufferDiff(previous, current, 32) <= duplicateLimit) {
        const dx = col === 1 ? 2 : -2;
        const dy = row === 0 ? 0 : (col === 1 ? -1 : 1);
        current = shiftFrame(current, ENEMY_FRAME, dx, dy);
        clearFrameGutter(current, ENEMY_FRAME, SAFE_FRAME_GUTTER);
        clearRawRect(raw, width, rect);
        pasteRaw(raw, width, ENEMY_FRAME * ENEMY_ROWS, current, ENEMY_FRAME, ENEMY_FRAME, rect.x, rect.y);
      }
      previous = current;
    }
  }
}

async function processEnemyFromCompactSource(source, sourcePath) {
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(data, info.width, info.height, {
    columns: ENEMY_COLUMNS,
    rows: ENEMY_ROWS,
    label: `${source.id} compact enemy sheet ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}`,
    guideColor: GUIDE_LINE_HEX
  });
  const sourceInfo = Object.assign({}, info, { grid });
  sourceInfo.chromaMode = detectChromaKeyMode(data, info.width, info.height, source);
  clearGuidePixels(data, info.width, info.height, GUIDE_LINE_HEX);
  removeChromaAndGuides(data, info.width, info.height, source, sourceInfo);
  const targetWidth = ENEMY_FRAME * ENEMY_COLUMNS;
  const targetHeight = ENEMY_FRAME * ENEMY_ROWS;
  const out = Buffer.alloc(targetWidth * targetHeight * 4);
  const analyses = [];
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    for (let col = 0; col < ENEMY_COLUMNS; col += 1) {
      const analysis = analyzeEnemyCell(data, sourceInfo, row, col);
      if (!analysis) {
        throw new Error(`${source.id} row ${row + 1} frame ${col + 1} is missing a readable body`);
      }
      analyses.push(analysis);
    }
  }
  const sharedScale = getSharedEnemyScale(analyses);
  for (const analysis of analyses) {
    const baseline = analysis.row === ENEMY_ROWS - 1 ? DEFEAT_BASELINE : BODY_BASELINE;
    await drawAnalyzedCell(data, info.width, out, {
      analysis,
      renderBounds: constrainEnemyRenderBounds(analysis, sharedScale, baseline),
      scale: sharedScale,
      targetRow: analysis.row,
      targetCol: analysis.col,
      targetWidth,
      targetHeight,
      frameSize: ENEMY_FRAME,
      baseline
    });
  }
  ensureDistinctEnemyFrames(out, targetWidth);
  clearOutputChromaSpill(out, sourceInfo.chromaMode);
  clearOuterBorder(out, targetWidth, targetHeight, ENEMY_FRAME, ENEMY_FRAME, ENEMY_COLUMNS, ENEMY_ROWS);
  clearDetachedLineArtifacts(out, targetWidth, ENEMY_FRAME, ENEMY_FRAME, ENEMY_COLUMNS, ENEMY_ROWS);
  clearTransparentColorData(out, targetWidth, targetHeight);
  const sheetPath = path.join(ENEMY_OUT_DIR, source.sheet);
  await writePng(out, targetWidth, targetHeight, sheetPath);
  await writePortraitFromSheet(out, targetWidth, source);
  process.stdout.write(`Processed ${path.relative(ROOT, sheetPath).replace(/\\/g, '/')} from compact source\n`);
}

async function processEnemyFromStandardSheet(source) {
  const sourcePath = path.join(ENEMY_OUT_DIR, source.standardSheet);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing compact source and standard fallback sheet for ${source.id}: ${source.source}`);
  }
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const sourceFrame = Math.floor(info.height / ENEMY_ROWS);
  const sourceColumns = Math.max(1, Math.floor(info.width / Math.max(1, sourceFrame)));
  if (sourceFrame <= 0 || info.height < ENEMY_ROWS * sourceFrame || sourceColumns < 1) {
    throw new Error(`${source.id} standard fallback sheet has an unsupported layout: ${info.width}x${info.height}`);
  }
  const targetWidth = ENEMY_FRAME * ENEMY_COLUMNS;
  const targetHeight = ENEMY_FRAME * ENEMY_ROWS;
  const out = Buffer.alloc(targetWidth * targetHeight * 4);
  const resizedRows = [];
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    const sourceFrames = [];
    for (let sourceCol = 0; sourceCol < sourceColumns; sourceCol += 1) {
      const cell = cropRaw(data, info.width, {
        x: sourceCol * sourceFrame,
        y: row * sourceFrame,
        w: sourceFrame,
        h: sourceFrame
      });
      const resized = await sharp(cell, {
        raw: { width: sourceFrame, height: sourceFrame, channels: 4 }
      })
        .resize(ENEMY_FRAME, ENEMY_FRAME, { fit: 'fill', background: TRANSPARENT, kernel: 'lanczos3' })
        .raw()
        .toBuffer();
      sourceFrames.push(resized);
    }
    resizedRows.push(chooseDistinctFrames(sourceFrames, row));
  }
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    for (let col = 0; col < ENEMY_COLUMNS; col += 1) {
      pasteRaw(out, targetWidth, targetHeight, resizedRows[row][col], ENEMY_FRAME, ENEMY_FRAME, col * ENEMY_FRAME, row * ENEMY_FRAME);
    }
  }
  ensureDistinctEnemyFrames(out, targetWidth);
  clearOuterBorder(out, targetWidth, targetHeight, ENEMY_FRAME, ENEMY_FRAME, ENEMY_COLUMNS, ENEMY_ROWS);
  clearDetachedLineArtifacts(out, targetWidth, ENEMY_FRAME, ENEMY_FRAME, ENEMY_COLUMNS, ENEMY_ROWS);
  clearTransparentColorData(out, targetWidth, targetHeight);
  const sheetPath = path.join(ENEMY_OUT_DIR, source.sheet);
  await writePng(out, targetWidth, targetHeight, sheetPath);
  await writePortraitFromSheet(out, targetWidth, source);
  process.stdout.write(`Processed ${path.relative(ROOT, sheetPath).replace(/\\/g, '/')} from ${path.basename(sourcePath)}\n`);
}

async function processEnemy(source, settings = {}) {
  const sourcePath = settings.sourcePath || path.join(RAW_DIR, source.source);
  if (fs.existsSync(sourcePath)) return processEnemyFromCompactSource(source, sourcePath);
  return processEnemyFromStandardSheet(source);
}

async function processProjectile(source) {
  const sourcePath = path.join(RAW_DIR, source.source);
  if (!fs.existsSync(sourcePath)) {
    process.stdout.write(`Skipped missing projectile source ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}\n`);
    return;
  }
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const grid = detectGuideGrid(data, info.width, info.height, {
    columns: PROJECTILE_COLUMNS,
    rows: 1,
    label: `${source.fileId} projectile sheet ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}`,
    guideColor: GUIDE_LINE_HEX
  });
  const sourceInfo = Object.assign({}, info, { grid });
  sourceInfo.chromaMode = detectChromaKeyMode(data, info.width, info.height, source);
  clearGuidePixels(data, info.width, info.height, GUIDE_LINE_HEX);
  removeChromaAndGuides(data, info.width, info.height, source, sourceInfo);
  const targetWidth = PROJECTILE_FRAME * PROJECTILE_COLUMNS;
  const targetHeight = PROJECTILE_FRAME;
  const out = Buffer.alloc(targetWidth * targetHeight * 4);
  for (let col = 0; col < PROJECTILE_COLUMNS; col += 1) {
    await drawProjectileCell(data, sourceInfo, out, col, {
      targetWidth,
      targetHeight,
      maxWidth: 52,
      maxHeight: 26
    });
  }
  clearOutputChromaSpill(out, sourceInfo.chromaMode);
  clearOuterBorder(out, targetWidth, targetHeight, PROJECTILE_FRAME, PROJECTILE_FRAME, PROJECTILE_COLUMNS, 1);
  clearDetachedLineArtifacts(out, targetWidth, PROJECTILE_FRAME, PROJECTILE_FRAME, PROJECTILE_COLUMNS, 1);
  clearTransparentColorData(out, targetWidth, targetHeight);
  const sheetPath = path.join(PROJECTILE_OUT_DIR, source.sheet);
  await writePng(out, targetWidth, targetHeight, sheetPath);
  process.stdout.write(`Processed ${path.relative(ROOT, sheetPath).replace(/\\/g, '/')}\n`);
}

function buildEnemyPrompt(source) {
  const identity = [
    source.name,
    source.family,
    source.role,
    source.behavior,
    source.mechanic
  ].filter(Boolean).join('; ');
  return [
    `Create one original charming 2D side-scroller MMO enemy sprite sheet for ${identity}.`,
    `The sheet must be exactly ${ENEMY_COLUMNS} columns by ${ENEMY_ROWS} rows with ${ENEMY_FRAME}px square cells, using visible ${GUIDE_LINE_HEX} borders between every cell.`,
    'Rows, in order: idle, walk, wind-up, attack, projectile or ranged action, buff or special action, hit, defeat.',
    'Use one right-facing enemy only, consistent body proportions, same head size, same outfit or markings, feet on one shared baseline, no cropped limbs, no duplicate characters, no text, no UI, no watermark.',
    `Use a perfectly flat ${getChromaKeyHex(source)} chroma-key background, with no shadows or floor plane. Do not use the chroma-key color inside the enemy art.`
  ].join(' ');
}

function writePromptManifest(target, enemySources, projectileSources) {
  const manifest = {
    sheet: {
      columns: ENEMY_COLUMNS,
      rows: ENEMY_ROWS,
      frameSize: ENEMY_FRAME,
      width: ENEMY_COLUMNS * ENEMY_FRAME,
      height: ENEMY_ROWS * ENEMY_FRAME,
      guideLine: GUIDE_LINE_HEX
    },
    enemies: enemySources.map((source) => ({
      id: source.id,
      name: source.name,
      rawSource: path.relative(ROOT, path.join(RAW_DIR, source.source)).replace(/\\/g, '/'),
      fallbackSource: path.relative(ROOT, path.join(ENEMY_OUT_DIR, source.standardSheet)).replace(/\\/g, '/'),
      outputSheet: `img/project-starfall/animations/enemies/${source.sheet}`,
      outputPortrait: `img/project-starfall/enemies/${source.portrait}`,
      prompt: buildEnemyPrompt(source)
    })),
    projectiles: projectileSources.map((source) => ({
      id: source.id,
      rawSource: path.relative(ROOT, path.join(RAW_DIR, source.source)).replace(/\\/g, '/'),
      outputSheet: `img/project-starfall/animations/enemy-projectiles/${source.sheet}`,
      prompt: `Create a ${PROJECTILE_COLUMNS} by 1 projectile sprite sheet with ${PROJECTILE_FRAME}px square cells, visible ${GUIDE_LINE_HEX} borders, flat #00ff00 chroma-key background, no text, and one consistent projectile moving right.`
    }))
  };
  const json = `${JSON.stringify(manifest, null, 2)}\n`;
  if (!target || target === '-') {
    process.stdout.write(json);
    return;
  }
  fs.mkdirSync(path.dirname(path.resolve(ROOT, target)), { recursive: true });
  fs.writeFileSync(path.resolve(ROOT, target), json);
}

function getAlphaArea(raw, width, x0, y0, w, h, threshold) {
  let area = 0;
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
      if (raw[(y * width + x) * 4 + 3] > threshold) area += 1;
    }
  }
  return area;
}

function frameTouchesEdge(raw, width, x0, y0, frameSize, threshold) {
  for (let x = x0; x < x0 + frameSize; x += 1) {
    if (raw[(y0 * width + x) * 4 + 3] > threshold) return true;
    if (raw[((y0 + frameSize - 1) * width + x) * 4 + 3] > threshold) return true;
  }
  for (let y = y0; y < y0 + frameSize; y += 1) {
    if (raw[(y * width + x0) * 4 + 3] > threshold) return true;
    if (raw[(y * width + x0 + frameSize - 1) * 4 + 3] > threshold) return true;
  }
  return false;
}

function countUnsafeGutterPixels(raw, width, x0, y0, frameSize, gutter, threshold) {
  let count = 0;
  for (let y = 0; y < frameSize; y += 1) {
    for (let x = 0; x < frameSize; x += 1) {
      if (x >= gutter && x < frameSize - gutter && y >= gutter && y < frameSize - gutter) continue;
      if (raw[((y0 + y) * width + x0 + x) * 4 + 3] > threshold) count += 1;
    }
  }
  return count;
}

function getFrameQualityMetrics(raw, width, row, col) {
  const x = col * ENEMY_FRAME;
  const y = row * ENEMY_FRAME;
  const rect = { x, y, w: ENEMY_FRAME, h: ENEMY_FRAME };
  const alphaBounds = getAlphaBounds(raw, width, x, y, ENEMY_FRAME, ENEMY_FRAME, ALPHA_THRESHOLD);
  const bodyBounds = getBodyAnchorBounds(raw, width, rect) || alphaBounds;
  const area = getAlphaArea(raw, width, x, y, ENEMY_FRAME, ENEMY_FRAME, ALPHA_THRESHOLD);
  return {
    row,
    col,
    area,
    unsafeGutterPixels: countUnsafeGutterPixels(
      raw,
      width,
      x,
      y,
      ENEMY_FRAME,
      SAFE_FRAME_GUTTER,
      ALPHA_THRESHOLD
    ),
    body: bodyBounds ? {
      w: bodyBounds.w,
      h: bodyBounds.h,
      centerX: bodyBounds.x - x + bodyBounds.w / 2,
      bottom: bodyBounds.y - y + bodyBounds.h - 1
    } : null,
    silhouette: alphaBounds ? {
      w: alphaBounds.w,
      h: alphaBounds.h,
      centerX: alphaBounds.x - x + alphaBounds.w / 2,
      bottom: alphaBounds.y - y + alphaBounds.h - 1
    } : null,
    buffer: cropRaw(raw, width, rect)
  };
}

function getEnemySheetQuality(raw, width) {
  const frames = [];
  const innerEdgeFrames = [];
  const duplicatePairs = [];
  const stableRowDrift = [];
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    const frameCount = ENEMY_COLUMNS;
    const rowFrames = [];
    for (let col = 0; col < frameCount; col += 1) {
      const metrics = getFrameQualityMetrics(raw, width, row, col);
      frames.push(metrics);
      rowFrames.push(metrics);
      const unsafeLimit = Math.max(12, Math.round(metrics.area * 0.004));
      if (metrics.unsafeGutterPixels >= unsafeLimit) {
        innerEdgeFrames.push({ row, col, pixels: metrics.unsafeGutterPixels, limit: unsafeLimit });
      }
    }
    for (let col = 1; col < rowFrames.length; col += 1) {
      const previous = rowFrames[col - 1];
      const current = rowFrames[col];
      const changedPixels = getFrameBufferDiff(previous.buffer, current.buffer, 32);
      const duplicateLimit = Math.max(12, Math.round(Math.min(previous.area, current.area) * 0.002));
      if (changedPixels <= duplicateLimit) {
        duplicatePairs.push({ row, from: col - 1, to: col, changedPixels, limit: duplicateLimit });
      }
    }
    if (STABLE_QUALITY_ROWS.includes(row) && rowFrames.every((frame) => frame.body && frame.silhouette)) {
      const getRatio = (key, kind) => {
        const values = rowFrames.map((frame) => frame[kind][key]);
        return Math.max(...values) / Math.max(1, Math.min(...values));
      };
      const getSpread = (key, kind) => {
        const values = rowFrames.map((frame) => frame[kind][key]);
        return Math.max(...values) - Math.min(...values);
      };
      const heightRatio = Math.min(getRatio('h', 'body'), getRatio('h', 'silhouette'));
      const centerSpread = Math.min(getSpread('centerX', 'body'), getSpread('centerX', 'silhouette'));
      const baselineSpread = Math.min(getSpread('bottom', 'body'), getSpread('bottom', 'silhouette'));
      const heightLimit = row === 6 ? 1.3 : 1.18;
      if (heightRatio > heightLimit || centerSpread > 10 || baselineSpread > 4) {
        stableRowDrift.push({
          row,
          heightRatio: Number(heightRatio.toFixed(3)),
          centerSpread: Number(centerSpread.toFixed(1)),
          baselineSpread
        });
      }
    }
  }
  return {
    innerEdgeFrames,
    duplicatePairs,
    stableRowDrift,
    frames: frames.map(({ buffer, ...frame }) => frame)
  };
}

function countVisibleChromaPixels(raw, chromaMode) {
  let count = 0;
  const magenta = chromaMode === 'magenta';
  for (let offset = 0; offset < raw.length; offset += 4) {
    if (raw[offset + 3] <= ALPHA_THRESHOLD) continue;
    const r = raw[offset];
    const g = raw[offset + 1];
    const b = raw[offset + 2];
    if (magenta ? (r > 235 && b > 235 && g < 28) : (g > 235 && r < 28 && b < 28)) count += 1;
  }
  return count;
}

async function getSourceChromaMode(source) {
  const sourcePath = path.join(RAW_DIR, source.source);
  if (!fs.existsSync(sourcePath)) return shouldUseMagentaKey(source) ? 'magenta' : 'green';
  const { data, info } = await readPngRaw(sourcePath);
  return detectChromaKeyMode(data, info.width, info.height, source);
}

async function readPngRaw(filePath) {
  return sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
}

async function validateEnemy(source) {
  const errors = [];
  const warnings = [];
  const sheetPath = path.join(ENEMY_OUT_DIR, source.sheet);
  const portraitPath = path.join(PORTRAIT_OUT_DIR, source.portrait);
  if (!fs.existsSync(sheetPath)) {
    errors.push(`Missing compact sheet ${path.relative(ROOT, sheetPath).replace(/\\/g, '/')}`);
    return { id: source.id, errors, warnings };
  }
  const { data, info } = await readPngRaw(sheetPath);
  const expectedWidth = ENEMY_COLUMNS * ENEMY_FRAME;
  const expectedHeight = ENEMY_ROWS * ENEMY_FRAME;
  if (info.width !== expectedWidth || info.height !== expectedHeight) {
    errors.push(`Expected ${expectedWidth}x${expectedHeight}, found ${info.width}x${info.height}`);
  }
  for (let row = 0; row < ENEMY_ROWS; row += 1) {
    const frameCount = ENEMY_COLUMNS;
    for (let col = 0; col < frameCount; col += 1) {
      const x = col * ENEMY_FRAME;
      const y = row * ENEMY_FRAME;
      if (getAlphaArea(data, info.width, x, y, ENEMY_FRAME, ENEMY_FRAME, 8) < 80) {
        errors.push(`Row ${row + 1} frame ${col + 1} has too little visible art`);
      }
      if (frameTouchesEdge(data, info.width, x, y, ENEMY_FRAME, 8)) {
        warnings.push(`Row ${row + 1} frame ${col + 1} touches the cell edge`);
      }
    }
  }
  const quality = info.width === expectedWidth && info.height === expectedHeight
    ? getEnemySheetQuality(data, info.width)
    : { innerEdgeFrames: [], duplicatePairs: [], stableRowDrift: [], frames: [] };
  quality.chromaMode = await getSourceChromaMode(source);
  quality.visibleChromaPixels = countVisibleChromaPixels(data, quality.chromaMode);
  if (quality.visibleChromaPixels > 4) {
    warnings.push(`Sheet retains ${quality.visibleChromaPixels} visible chroma-key pixels`);
  }
  quality.innerEdgeFrames.forEach((frame) => {
    warnings.push(`Row ${frame.row + 1} frame ${frame.col + 1} crowds the ${SAFE_FRAME_GUTTER}px safe gutter (${frame.pixels} pixels)`);
  });
  quality.duplicatePairs.forEach((pair) => {
    warnings.push(`Row ${pair.row + 1} frames ${pair.from + 1}-${pair.to + 1} are adjacent near-duplicates`);
  });
  quality.stableRowDrift.forEach((row) => {
    warnings.push(`Row ${row.row + 1} has unstable body framing (height ${row.heightRatio}x, center ${row.centerSpread}px, baseline ${row.baselineSpread}px)`);
  });
  if (!fs.existsSync(portraitPath)) {
    errors.push(`Missing portrait ${path.relative(ROOT, portraitPath).replace(/\\/g, '/')}`);
  } else {
    const portraitMeta = await sharp(portraitPath).metadata();
    if (portraitMeta.width !== 320 || portraitMeta.height !== 320) {
      errors.push(`Expected 320x320 portrait, found ${portraitMeta.width}x${portraitMeta.height}`);
    }
  }
  return { id: source.id, errors, warnings, quality };
}

async function validateProjectile(source) {
  const errors = [];
  const warnings = [];
  const sheetPath = path.join(PROJECTILE_OUT_DIR, source.sheet);
  if (!fs.existsSync(sheetPath)) {
    errors.push(`Missing projectile sheet ${path.relative(ROOT, sheetPath).replace(/\\/g, '/')}`);
    return { id: source.id, errors, warnings };
  }
  const { data, info } = await readPngRaw(sheetPath);
  const expectedWidth = PROJECTILE_COLUMNS * PROJECTILE_FRAME;
  if (info.width !== expectedWidth || info.height !== PROJECTILE_FRAME) {
    errors.push(`Expected ${expectedWidth}x${PROJECTILE_FRAME}, found ${info.width}x${info.height}`);
  }
  for (let col = 0; col < PROJECTILE_COLUMNS; col += 1) {
    if (getAlphaArea(data, info.width, col * PROJECTILE_FRAME, 0, PROJECTILE_FRAME, PROJECTILE_FRAME, 8) < 12) {
      errors.push(`Projectile frame ${col + 1} has too little visible art`);
    }
  }
  return { id: source.id, errors, warnings };
}

async function runAudit(enemySources, projectileSources, strict) {
  const enemies = [];
  const projectiles = [];
  for (const source of enemySources) enemies.push(await validateEnemy(source));
  for (const source of projectileSources) projectiles.push(await validateProjectile(source));
  const totals = enemies.concat(projectiles).reduce((summary, result) => {
    summary.errors += result.errors.length;
    summary.warnings += result.warnings.length;
    return summary;
  }, { errors: 0, warnings: 0 });
  return {
    checked: {
      enemies: enemies.length,
      projectiles: projectiles.length
    },
    strict: Boolean(strict),
    totals,
    enemies,
    projectiles
  };
}

function parseArgs(argv) {
  const options = {
    enemyIds: [],
    writePrompts: null,
    audit: false,
    validate: false,
    json: false,
    strict: false,
    sourceOverride: null
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--enemy') {
      options.enemyIds.push(argv[index + 1]);
      index += 1;
    } else if (arg === '--source') {
      options.sourceOverride = argv[index + 1];
      index += 1;
    } else if (arg === '--write-prompts') {
      options.writePrompts = argv[index + 1] && !argv[index + 1].startsWith('--') ? argv[index + 1] : '-';
      if (options.writePrompts !== '-') index += 1;
    } else if (arg === '--audit') {
      options.audit = true;
    } else if (arg === '--validate') {
      options.validate = true;
    } else if (arg === '--json') {
      options.json = true;
    } else if (arg === '--strict') {
      options.strict = true;
    }
  }
  return options;
}

async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  let enemySources = getEnemySources();
  const projectileSources = getProjectileSources();
  if (options.enemyIds.length) {
    const wanted = new Set(options.enemyIds.filter(Boolean));
    enemySources = enemySources.filter((source) =>
      wanted.has(source.id) || wanted.has(source.fileId) || wanted.has(source.name)
    );
  }
  let sourceOverridePath = null;
  if (options.sourceOverride) {
    if (enemySources.length !== 1 || options.enemyIds.length !== 1) {
      throw new Error('--source requires exactly one --enemy selection');
    }
    sourceOverridePath = path.isAbsolute(options.sourceOverride)
      ? options.sourceOverride
      : path.resolve(ROOT, options.sourceOverride);
    if (!fs.existsSync(sourceOverridePath)) {
      throw new Error(`Source override does not exist: ${options.sourceOverride}`);
    }
  }
  if (options.writePrompts !== null) {
    writePromptManifest(options.writePrompts, enemySources, projectileSources);
    return;
  }
  if (options.audit || options.validate) {
    const report = await runAudit(enemySources, projectileSources, options.strict);
    if (options.json || options.audit) {
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    } else {
      process.stdout.write(`Checked ${report.checked.enemies} enemies and ${report.checked.projectiles} projectile sheets. Errors: ${report.totals.errors}. Warnings: ${report.totals.warnings}.\n`);
    }
    if (report.totals.errors || (options.strict && report.totals.warnings)) process.exitCode = 1;
    return;
  }
  for (const source of enemySources) await processEnemy(source, { sourcePath: sourceOverridePath });
  if (!options.enemyIds.length) {
    for (const source of projectileSources) await processProjectile(source);
  }
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.message ? error.message : error);
    process.exit(1);
  });
}

module.exports = {
  buildEnemyPrompt,
  getEnemySources,
  getEnemySheetQuality,
  getProjectileSources,
  getSharedEnemyScale,
  main,
  processEnemy,
  runAudit
};
