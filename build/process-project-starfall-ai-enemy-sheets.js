#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const Data = require('../js/games/project-starfall/project-starfall-data.js');
const {
  GUIDE_LINE_HEX,
  clearGuidePixels,
  detectGuideGrid,
  getGridCellRect
} = require('./project-starfall-sheet-grid.js');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_SOURCE_DIR = path.join(ROOT, 'tmp/project-starfall-ai-enemies/raw');
const DEFAULT_PROMPT_PATH = path.join(ROOT, 'tmp/project-starfall-ai-enemies/prompts.json');
const DEFAULT_REVIEW_DIR = path.join(ROOT, 'tmp/project-starfall-ai-enemies/review');
const ENEMY_DIR = path.join(ROOT, 'img/project-starfall/enemies');
const ENEMY_SHEET_DIR = path.join(ROOT, 'img/project-starfall/animations/enemies');
const FRAME_SIZE = 160;
const SHEET_COLUMNS = 6;
const ENEMY_ROWS = Object.freeze(['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat']);
const SHEET_WIDTH = FRAME_SIZE * SHEET_COLUMNS;
const SHEET_HEIGHT = FRAME_SIZE * ENEMY_ROWS.length;
const PORTRAIT_SIZE = 320;
const TRANSPARENT = Object.freeze({ r: 0, g: 0, b: 0, alpha: 0 });
const COMPONENT_ALPHA_THRESHOLD = 10;
const MIN_COMPONENT_AREA = 18;
const CELL_BORDER_CLEAR_PX = 1;
const STABLE_BODY_ROWS = Object.freeze(['idle', 'move', 'telegraph', 'attack', 'hit']);
const BASELINE_BODY_ROWS = Object.freeze(['idle', 'move', 'hit']);
const MOTION_CHECK_ROWS = Object.freeze(['idle', 'move', 'attack', 'projectile', 'buff', 'defeat']);
const SOURCE_GRID_COMPONENT_MIN_AREA = 180;
const DUPLICATE_BODY_AREA_RATIO = 0.32;
const STABLE_ROW_CENTER_TOLERANCE = 34;

const ROW_DESCRIPTIONS = Object.freeze({
  idle: 'quiet breathing loop with tiny blink, bob, or glow movement',
  move: 'clear locomotion loop matching the enemy behavior: hop, crawl, shuffle, float, charge, or stride',
  telegraph: 'readable wind-up pose before the attack, with the attack source clearly visible',
  attack: 'short melee strike, bite, slash, slam, lunge, or cast release',
  projectile: 'projectile or special-cast startup, using the same creature and silhouette',
  buff: 'support, shield, heal, enrage, overheat, or aura pulse pose',
  hit: 'recoil, squash, stagger, or shell crack reaction',
  defeat: 'collapse, pop, shatter, dissolve, or break-apart ending'
});

const BEHAVIOR_MOTION = Object.freeze({
  hopper: 'squashy little hops with a soft landing baseline',
  bruiser: 'heavy grounded steps with a braced shoulders/body mass',
  turret: 'mostly stationary with recoil and charged firing poses',
  skirmisher: 'fast lunges and retreat-ready movement',
  charger: 'low forward charge poses with strong anticipation',
  armored: 'slow armored crawl or march with shell/plate emphasis',
  blocker: 'guarded stance with shield/blocking posture',
  thrower: 'ranged throw or shot poses with retreating footwork',
  flyer: 'floating bob with no ground contact, centered in each frame',
  healer: 'support casting poses with gentle aura or bloom pulses',
  elite: 'larger dramatic motion, ambush and enrage readable',
  boss: 'large boss-scale poses with strong silhouette changes and readable phase energy'
});

function parseArgs(args) {
  const options = {
    sourceDir: DEFAULT_SOURCE_DIR,
    enemyId: '',
    all: false,
    validate: false,
    audit: false,
    strict: false,
    repairRuntime: false,
    json: false,
    review: '',
    writePrompts: '',
    help: false
  };
  for (let index = 0; index < args.length; index += 1) {
    const arg = String(args[index] || '');
    if (arg === '--source') {
      options.sourceDir = path.resolve(ROOT, String(args[index + 1] || ''));
      index += 1;
    } else if (arg.startsWith('--source=')) {
      options.sourceDir = path.resolve(ROOT, arg.slice('--source='.length));
    } else if (arg === '--enemy') {
      options.enemyId = String(args[index + 1] || '').trim();
      index += 1;
    } else if (arg.startsWith('--enemy=')) {
      options.enemyId = arg.slice('--enemy='.length).trim();
    } else if (arg === '--all') {
      options.all = true;
    } else if (arg === '--validate') {
      options.validate = true;
    } else if (arg === '--audit') {
      options.audit = true;
    } else if (arg === '--strict') {
      options.strict = true;
    } else if (arg === '--repair-runtime') {
      options.repairRuntime = true;
    } else if (arg === '--json') {
      options.json = true;
    } else if (arg === '--review') {
      options.review = String(args[index + 1] || DEFAULT_REVIEW_DIR);
      index += 1;
    } else if (arg.startsWith('--review=')) {
      options.review = arg.slice('--review='.length) || DEFAULT_REVIEW_DIR;
    } else if (arg === '--write-prompts') {
      options.writePrompts = String(args[index + 1] || DEFAULT_PROMPT_PATH);
      index += 1;
    } else if (arg.startsWith('--write-prompts=')) {
      options.writePrompts = arg.slice('--write-prompts='.length) || DEFAULT_PROMPT_PATH;
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    }
  }
  return options;
}

function usage() {
  return [
    'Usage:',
    '  node build/process-project-starfall-ai-enemy-sheets.js --write-prompts [path|-]',
    '  node build/process-project-starfall-ai-enemy-sheets.js --source tmp/project-starfall-ai-enemies/raw --all',
    '  node build/process-project-starfall-ai-enemy-sheets.js --source tmp/project-starfall-ai-enemies/raw --enemy slimelet',
    '  node build/process-project-starfall-ai-enemy-sheets.js --repair-runtime --all',
    '  node build/process-project-starfall-ai-enemy-sheets.js --validate',
    '  node build/process-project-starfall-ai-enemy-sheets.js --audit [--strict] [--json]',
    '  node build/process-project-starfall-ai-enemy-sheets.js --review [path]'
  ].join('\n');
}

function getFileId(enemy) {
  const sheet = enemy && enemy.animation && enemy.animation.sheet || '';
  const base = path.basename(sheet, '.png').replace(/-compact-sheet$/, '').replace(/-sheet$/, '');
  return base || String(enemy && enemy.id || '').replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`);
}

function getEnemyById(enemyId) {
  const normalized = String(enemyId || '').trim();
  return (Data.ENEMIES || []).find((enemy) => enemy.id === normalized || getFileId(enemy) === normalized) || null;
}

function getSourcePath(enemy, sourceDir) {
  const fileId = getFileId(enemy);
  const candidates = [
    path.join(sourceDir, `${fileId}-sheet.png`),
    path.join(sourceDir, `${fileId}.png`),
    path.join(sourceDir, `${enemy.id}-sheet.png`),
    path.join(sourceDir, `${enemy.id}.png`)
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) || candidates[0];
}

function shouldUseMagentaKey(enemy) {
  const family = String(enemy && enemy.family || '').toLowerCase();
  const name = String(enemy && enemy.name || '').toLowerCase();
  return family.includes('plant') || family.includes('ooze') || name.includes('moss') || name.includes('vine') || name.includes('briar');
}

function getKeyColor(enemy) {
  return shouldUseMagentaKey(enemy) ? '#ff00ff' : '#00ff00';
}

function buildEnemyPrompt(enemy) {
  const keyColor = getKeyColor(enemy);
  const layout = getEnemyAnimationAuditLayout(enemy);
  const motion = BEHAVIOR_MOTION[enemy.behavior] || 'simple readable side-scroller monster animation';
  const rowText = ENEMY_ROWS.map((row) => `${row}: ${ROW_DESCRIPTIONS[row]}`).join('; ');
  const cellCount = layout.columns * ENEMY_ROWS.length;
  return [
    'Use case: stylized-concept',
    'Asset type: Project Starfall enemy sprite sheet',
    `Primary request: Create one complete original 2D side-scroller MMO monster sprite sheet for "${enemy.name}".`,
    `Enemy identity: ${enemy.name}; family ${enemy.family}; role ${enemy.role}; behavior ${enemy.behavior}; mechanic ${enemy.mechanic}.`,
    `Motion language: ${motion}.`,
    `Sheet contract: exactly ${layout.columns} columns by 8 rows, exactly ${cellCount} equal cells total, one full-body enemy only in each cell, all frames face right, no extra panels, no separate labels.`,
    `Visible guide grid: draw thin straight solid ${GUIDE_LINE_HEX} divider lines between every cell and around the sheet border. The grid must be perfectly aligned, but never overlap the creature. Do not use ${GUIDE_LINE_HEX} anywhere in the enemy art; the processor rejects sheets without a complete readable guide grid.`,
    `Rows from top to bottom: ${rowText}.`,
    'Style: charming 2D fantasy MMORPG creature art with rounded readable silhouette, clean dark outline, soft cel shading, simple expressive face or focal detail, and high readability at small size.',
    'Consistency: keep the same proportions, colors, markings, eye placement, accessories, weapon, outfit, and silhouette across every frame.',
    'Frame rules: no duplicate characters in a cell, no cropped body parts, no close-up frames, no standalone weapon/projectile-only cells, no body touching the cell edge, and keep the character centered on the same baseline and same scale.',
    `Background: perfectly flat solid ${keyColor} chroma-key background only, no shadows, no gradients, no floor plane, no texture.`,
    `Avoid: do not use ${keyColor} anywhere in the enemy, no UI, no text, no watermark, no frame labels, no copied MapleStory characters or assets.`
  ].join('\n');
}

function buildPromptManifest() {
  const defaultLayout = getEnemyAnimationAuditLayout((Data.ENEMIES || [])[0] || null);
  return {
    generatedAt: new Date().toISOString(),
    sheet: {
      columns: defaultLayout.columns,
      rows: ENEMY_ROWS,
      frameSize: defaultLayout.frameWidth,
      width: defaultLayout.width,
      height: defaultLayout.height,
      guideLine: GUIDE_LINE_HEX
    },
    workflow: 'Generate one bordered guide-grid AI sheet per enemy, save raw sources into tmp/project-starfall-ai-enemies/raw, then process compact runtime sheets with build/process-project-starfall-compact-bandits.js. The source processor detects the cyan cell borders before cleanup and fails if any expected border is missing.',
    enemies: (Data.ENEMIES || []).map((enemy) => {
      const layout = getEnemyAnimationAuditLayout(enemy);
      return {
        id: enemy.id,
        fileId: getFileId(enemy),
        name: enemy.name,
        family: enemy.family,
        behavior: enemy.behavior,
        role: enemy.role,
        chromaKey: getKeyColor(enemy),
        columns: layout.columns,
        frameSize: layout.frameWidth,
        width: layout.width,
        height: layout.height,
        rawSource: `tmp/project-starfall-ai-enemies/raw/${getFileId(enemy)}-sheet.png`,
        outputSheet: enemy.animation && enemy.animation.sheet,
        outputPortrait: enemy.asset,
        prompt: buildEnemyPrompt(enemy)
      };
    })
  };
}

function colorFromHex(hex) {
  const value = String(hex || '#00ff00').replace('#', '');
  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16)
  };
}

function colorDistance(raw, offset, color) {
  return Math.abs(raw[offset] - color.r) +
    Math.abs(raw[offset + 1] - color.g) +
    Math.abs(raw[offset + 2] - color.b);
}

function isLikelyKey(raw, offset, keyColor) {
  if (raw[offset + 3] < 8) return true;
  if (colorDistance(raw, offset, keyColor) <= 92) return true;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const greenKey = g > 160 && r < 120 && b < 120;
  const magentaKey = r > 160 && b > 160 && g < 140;
  return greenKey || magentaKey;
}

function isKeyFringe(raw, offset, keyColor) {
  if (raw[offset + 3] < 8) return false;
  const r = raw[offset];
  const g = raw[offset + 1];
  const b = raw[offset + 2];
  const greenKey = keyColor.g > keyColor.r && keyColor.g > keyColor.b;
  if (greenKey) {
    const greenBias = g - Math.max(r, b);
    return g > 96 && greenBias > 24 && colorDistance(raw, offset, keyColor) <= 260;
  }
  const magentaBias = Math.min(r, b) - g;
  return r > 96 && b > 96 && magentaBias > 18 && colorDistance(raw, offset, keyColor) <= 260;
}

function hasTransparentNeighbor(raw, width, height, x, y) {
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      if (!dx && !dy) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) return true;
      if (raw[(ny * width + nx) * 4 + 3] < 16) return true;
    }
  }
  return false;
}

function cleanKeyFringe(raw, width, height, keyColor) {
  const clearPixels = [];
  const greenKey = keyColor.g > keyColor.r && keyColor.g > keyColor.b;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      if (!isKeyFringe(raw, offset, keyColor)) continue;
      if (greenKey || colorDistance(raw, offset, keyColor) <= 180 || hasTransparentNeighbor(raw, width, height, x, y)) {
        clearPixels.push(offset);
      }
    }
  }
  clearPixels.forEach((offset) => {
    raw[offset + 3] = 0;
  });
}

function removeGuideLines(raw, width, height) {
  clearGuidePixels(raw, width, height, GUIDE_LINE_HEX);
}

function removeBackgroundFlood(raw, width, height, keyColor) {
  const corners = [
    0,
    (width - 1) * 4,
    ((height - 1) * width) * 4,
    ((height * width) - 1) * 4
  ].map((offset) => ({ r: raw[offset], g: raw[offset + 1], b: raw[offset + 2] }));
  const visited = new Uint8Array(width * height);
  const queue = [];

  function isBackground(offset) {
    if (isLikelyKey(raw, offset, keyColor)) return true;
    return corners.some((color) => colorDistance(raw, offset, color) <= 96);
  }

  function enqueue(x, y) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const pixel = y * width + x;
    if (visited[pixel]) return;
    const offset = pixel * 4;
    if (!isBackground(offset)) return;
    visited[pixel] = 1;
    queue.push(pixel);
  }

  for (let x = 0; x < width; x += 1) {
    enqueue(x, 0);
    enqueue(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    enqueue(0, y);
    enqueue(width - 1, y);
  }

  for (let index = 0; index < queue.length; index += 1) {
    const pixel = queue[index];
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    raw[pixel * 4 + 3] = 0;
    enqueue(x - 1, y);
    enqueue(x + 1, y);
    enqueue(x, y - 1);
    enqueue(x, y + 1);
  }

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    if (isLikelyKey(raw, offset, keyColor)) raw[offset + 3] = 0;
  }
  removeGuideLines(raw, width, height);
  cleanKeyFringe(raw, width, height, keyColor);
}

function getAlphaBounds(raw, width, x0, y0, w, h, threshold) {
  let minX = x0 + w;
  let minY = y0 + h;
  let maxX = x0 - 1;
  let maxY = y0 - 1;
  for (let y = y0; y < y0 + h; y += 1) {
    for (let x = x0; x < x0 + w; x += 1) {
      const alpha = raw[(y * width + x) * 4 + 3];
      if (alpha <= threshold) continue;
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
    const sourceEnd = sourceStart + bounds.w * 4;
    raw.copy(cropped, y * bounds.w * 4, sourceStart, sourceEnd);
  }
  return cropped;
}

function extractRegionRaw(raw, width, x0, y0, w, h) {
  const region = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y += 1) {
    const sourceStart = ((y0 + y) * width + x0) * 4;
    const sourceEnd = sourceStart + w * 4;
    raw.copy(region, y * w * 4, sourceStart, sourceEnd);
  }
  return region;
}

function alphaCompositePixel(target, targetOffset, source, sourceOffset) {
  const sourceAlpha = source[sourceOffset + 3] / 255;
  if (sourceAlpha <= 0) return;
  const targetAlpha = target[targetOffset + 3] / 255;
  const outAlpha = sourceAlpha + targetAlpha * (1 - sourceAlpha);
  if (outAlpha <= 0) return;
  target[targetOffset] = Math.round((source[sourceOffset] * sourceAlpha + target[targetOffset] * targetAlpha * (1 - sourceAlpha)) / outAlpha);
  target[targetOffset + 1] = Math.round((source[sourceOffset + 1] * sourceAlpha + target[targetOffset + 1] * targetAlpha * (1 - sourceAlpha)) / outAlpha);
  target[targetOffset + 2] = Math.round((source[sourceOffset + 2] * sourceAlpha + target[targetOffset + 2] * targetAlpha * (1 - sourceAlpha)) / outAlpha);
  target[targetOffset + 3] = Math.round(outAlpha * 255);
}

function pasteRaw(target, targetWidth, source, sourceWidth, sourceHeight, x0, y0) {
  for (let y = 0; y < sourceHeight; y += 1) {
    const targetY = y0 + y;
    if (targetY < 0 || targetY >= SHEET_HEIGHT) continue;
    for (let x = 0; x < sourceWidth; x += 1) {
      const targetX = x0 + x;
      if (targetX < 0 || targetX >= targetWidth) continue;
      alphaCompositePixel(target, (targetY * targetWidth + targetX) * 4, source, (y * sourceWidth + x) * 4);
    }
  }
}

function getComponentCenter(component, axis) {
  return axis === 'x' ? componentCenterX(component) : componentCenterY(component);
}

function clusterPositions(values, tolerance) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  const clusters = [];
  sorted.forEach((value) => {
    const current = clusters[clusters.length - 1];
    if (!current || Math.abs(value - current.center) > tolerance) {
      clusters.push({ values: [value], center: value });
      return;
    }
    current.values.push(value);
    current.center = getMedian(current.values);
  });
  return clusters.map((cluster) => cluster.center);
}

function collapseSourceGridCenters(centers, expected, total) {
  const sorted = centers.filter((center) => Number.isFinite(center)).sort((a, b) => a - b);
  if (sorted.length <= expected || expected <= 1) return sorted;
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const spacing = Math.max(1, (max - min) / Math.max(1, expected - 1));
  const groups = Array.from({ length: expected }, () => []);
  sorted.forEach((center) => {
    const index = Math.max(0, Math.min(expected - 1, Math.round((center - min) / spacing)));
    groups[index].push(center);
  });
  if (groups.some((group) => !group.length)) return sorted;
  const maxGroupSpan = total / expected * 0.72;
  const collapsed = groups.map((group) => getMedian(group));
  const plausible = groups.every((group) => Math.max(...group) - Math.min(...group) <= maxGroupSpan) &&
    collapsed.slice(1).every((center, index) => center - collapsed[index] >= total / expected * 0.45);
  return plausible ? collapsed : sorted;
}

function getSourceGridComponents(raw, width, height) {
  const components = findAlphaComponents(raw, width, height, COMPONENT_ALPHA_THRESHOLD);
  const maxArea = components.reduce((max, component) => Math.max(max, component.area), 0);
  const expectedCellH = height / ENEMY_ROWS.length;
  const areaFloor = Math.max(SOURCE_GRID_COMPONENT_MIN_AREA, maxArea * 0.055);
  return components.filter((component) => (
    component.area >= areaFloor &&
    component.w >= 8 &&
    component.h >= Math.max(28, expectedCellH * 0.22) &&
    component.w <= width * 0.35 &&
    component.h <= height * 0.28
  ));
}

function inferSourceGrid(raw, width, height, enemy, sourcePath) {
  const components = getSourceGridComponents(raw, width, height);
  const xTolerance = Math.max(18, width / (SHEET_COLUMNS * 4));
  const yTolerance = Math.max(18, height / (ENEMY_ROWS.length * 4));
  const columns = collapseSourceGridCenters(
    clusterPositions(components.map((component) => getComponentCenter(component, 'x')), xTolerance),
    SHEET_COLUMNS,
    width
  );
  const rows = collapseSourceGridCenters(
    clusterPositions(components.map((component) => getComponentCenter(component, 'y')), yTolerance),
    ENEMY_ROWS.length,
    height
  );
  const label = `${enemy && enemy.id || 'enemy'} source${sourcePath ? ` ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}` : ''}`;
  if (columns.length !== SHEET_COLUMNS || rows.length !== ENEMY_ROWS.length) {
    throw new Error(`${label} should visibly contain ${SHEET_COLUMNS} columns and ${ENEMY_ROWS.length} rows before processing; detected ${columns.length} columns and ${rows.length} rows from ${width}x${height}. Regenerate this raw sheet instead of force-resizing it.`);
  }
  return {
    columns,
    rows,
    uniform: Math.abs(width / height - SHEET_WIDTH / SHEET_HEIGHT) <= 0.025
  };
}

function makeGridSlices(centers, total) {
  const sorted = centers.slice().sort((a, b) => a - b);
  const diffs = sorted.slice(1).map((center, index) => center - sorted[index]);
  const spacing = getMedian(diffs) || total / Math.max(1, sorted.length);
  const boundaries = [Math.max(0, Math.round(sorted[0] - spacing / 2))];
  for (let index = 1; index < sorted.length; index += 1) {
    boundaries.push(Math.round((sorted[index - 1] + sorted[index]) / 2));
  }
  boundaries.push(Math.min(total, Math.round(sorted[sorted.length - 1] + spacing / 2)));
  return sorted.map((center, index) => ({
    start: Math.max(0, Math.min(total - 1, boundaries[index])),
    end: Math.max(1, Math.min(total, boundaries[index + 1])),
    center
  })).map((slice) => (
    slice.end > slice.start ? slice : Object.assign({}, slice, { end: Math.min(total, slice.start + 1) })
  ));
}

function makeUniformGridSlices(count, total) {
  return Array.from({ length: count }, (unused, index) => ({
    start: Math.max(0, Math.min(total - 1, Math.round(index * total / count))),
    end: Math.max(1, Math.min(total, Math.round((index + 1) * total / count))),
    center: (index + 0.5) * total / count
  }));
}

async function standardizeSourceSheet(raw, width, height, enemy, sourcePath, sourceGrid) {
  const label = `${enemy && enemy.id || 'enemy'} source${sourcePath ? ` ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}` : ''}`;
  if (!sourceGrid) {
    throw new Error(`${label} is missing a detected guide grid; regenerate the raw source with ${GUIDE_LINE_HEX} borders around every cell.`);
  }
  const output = Buffer.alloc(SHEET_WIDTH * SHEET_HEIGHT * 4);
  for (let row = 0; row < ENEMY_ROWS.length; row += 1) {
    for (let col = 0; col < SHEET_COLUMNS; col += 1) {
      const rect = getGridCellRect(sourceGrid, row, col);
      if (rect.x < 0 || rect.y < 0 || rect.x + rect.w > width || rect.y + rect.h > height) {
        throw new Error(`${label} guide cell ${row + 1}:${col + 1} is outside the source image`);
      }
      const region = extractRegionRaw(raw, width, rect.x, rect.y, rect.w, rect.h);
      const cell = await sharp(region, {
        raw: { width: rect.w, height: rect.h, channels: 4 }
      })
        .resize(FRAME_SIZE, FRAME_SIZE, { fit: 'contain', background: TRANSPARENT, kernel: 'cubic' })
        .raw()
        .toBuffer();
      pasteRaw(output, SHEET_WIDTH, cell, FRAME_SIZE, FRAME_SIZE, col * FRAME_SIZE, row * FRAME_SIZE);
    }
  }
  return output;
}

function isFloatingEnemy(enemy) {
  return enemy && (enemy.behavior === 'flyer' || String(enemy.family || '').toLowerCase().includes('spirit'));
}

function getContentLimit(enemy) {
  if (enemy && enemy.behavior === 'boss') return 154;
  if (enemy && enemy.behavior === 'elite') return 148;
  return 138;
}

function getMedian(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function extractCellRaw(raw, width, cellX, cellY) {
  const cell = Buffer.alloc(FRAME_SIZE * FRAME_SIZE * 4);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const sourceStart = ((cellY + y) * width + cellX) * 4;
    const sourceEnd = sourceStart + FRAME_SIZE * 4;
    raw.copy(cell, y * FRAME_SIZE * 4, sourceStart, sourceEnd);
  }
  return cell;
}

function findAlphaComponents(raw, width, height, threshold) {
  const visited = new Uint8Array(width * height);
  const components = [];
  const minAlpha = threshold == null ? COMPONENT_ALPHA_THRESHOLD : Number(threshold);

  for (let start = 0; start < width * height; start += 1) {
    if (visited[start] || raw[start * 4 + 3] <= minAlpha) continue;
    const queue = [start];
    const pixels = [];
    visited[start] = 1;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    for (let index = 0; index < queue.length; index += 1) {
      const pixel = queue[index];
      pixels.push(pixel);
      const x = pixel % width;
      const y = Math.floor(pixel / width);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [
        x > 0 ? pixel - 1 : -1,
        x < width - 1 ? pixel + 1 : -1,
        y > 0 ? pixel - width : -1,
        y < height - 1 ? pixel + width : -1
      ];
      neighbors.forEach((neighbor) => {
        if (neighbor < 0 || visited[neighbor] || raw[neighbor * 4 + 3] <= minAlpha) return;
        visited[neighbor] = 1;
        queue.push(neighbor);
      });
    }

    components.push({
      pixels,
      area: pixels.length,
      minX,
      minY,
      maxX,
      maxY,
      w: maxX - minX + 1,
      h: maxY - minY + 1
    });
  }

  return components
    .filter((component) => component.area >= MIN_COMPONENT_AREA)
    .sort((a, b) => b.area - a.area);
}

function componentCenterX(component) {
  return (component.minX + component.maxX + 1) / 2;
}

function componentCenterY(component) {
  return (component.minY + component.maxY + 1) / 2;
}

function componentTouchesCellEdge(component, pad) {
  const edgePad = Math.max(0, Number(pad || 0));
  return component.minX <= edgePad ||
    component.minY <= edgePad ||
    component.maxX >= FRAME_SIZE - 1 - edgePad ||
    component.maxY >= FRAME_SIZE - 1 - edgePad;
}

function isBodyLikeComponent(component, enemy, referenceArea) {
  if (!component) return false;
  const areaFloor = Math.max(70, referenceArea ? referenceArea * 0.08 : 70);
  if (component.area < areaFloor || component.w < 10 || component.h < 10) return false;
  const cx = componentCenterX(component);
  const cy = componentCenterY(component);
  if (isFloatingEnemy(enemy)) return cx >= 12 && cx <= 148 && cy >= 8 && cy <= 152;
  return cx >= 12 && cx <= 148 && cy >= 28 && component.maxY >= 54;
}

function chooseBodyComponent(components, enemy, referenceArea) {
  return (components || []).find((component) => isBodyLikeComponent(component, enemy, referenceArea)) || null;
}

function getBodyLikeComponents(components, enemy, referenceArea) {
  return (components || []).filter((component) => isBodyLikeComponent(component, enemy, referenceArea));
}

function getRenderableComponents(components, bodyComponent) {
  if (!components || !components.length) return [];
  const main = bodyComponent || components[0];
  const mainArea = Math.max(1, Number(main && main.area || 1));
  return components.filter((component) => {
    if (component === main || component === bodyComponent) return true;
    if (component.area < Math.max(MIN_COMPONENT_AREA, mainArea * 0.015)) return false;
    if (componentTouchesCellEdge(component, 0) && component.area < mainArea * 0.35) return false;
    return true;
  });
}

function copyComponents(raw, width, height, components) {
  const output = Buffer.alloc(width * height * 4);
  (components || []).forEach((component) => {
    (component.pixels || []).forEach((pixel) => {
      const offset = pixel * 4;
      output[offset] = raw[offset];
      output[offset + 1] = raw[offset + 1];
      output[offset + 2] = raw[offset + 2];
      output[offset + 3] = raw[offset + 3];
    });
  });
  return output;
}

function getFrameBodyIssue(frame, enemy, referenceArea, rowCenterX, referenceWidth, referenceHeight) {
  if (!frame.body) return 'missing full-body enemy';
  if (componentTouchesCellEdge(frame.body, 0) && (frame.body.w >= FRAME_SIZE - 2 || frame.body.h >= FRAME_SIZE - 2)) {
    return 'cropped body touches cell edge';
  }
  const bodyLike = getBodyLikeComponents(frame.components, enemy, referenceArea);
  const duplicate = bodyLike.find((component) => component !== frame.body && component.area >= frame.body.area * DUPLICATE_BODY_AREA_RATIO);
  if (duplicate) return 'multiple body-like components';
  const areaRatio = referenceArea ? frame.body.area / referenceArea : 1;
  if (frame.rowId === 'defeat' && referenceHeight && referenceArea &&
    frame.body.h / referenceHeight > 0.68 && areaRatio > 0.85) {
    return 'defeat frame is too upright for the enemy baseline';
  }
  if (frame.rowId !== 'defeat' && areaRatio > 1.95) return 'body area is too large for the enemy baseline';
  if (frame.rowId !== 'defeat' && referenceWidth && frame.body.w / referenceWidth > 1.72) {
    return 'body width is too large for the enemy baseline';
  }
  if (frame.rowId !== 'defeat' && referenceHeight && frame.body.h / referenceHeight > 1.72) {
    return 'body height is too large for the enemy baseline';
  }
  if (STABLE_BODY_ROWS.includes(frame.rowId) && Number.isFinite(rowCenterX) &&
    Math.abs(componentCenterX(frame.body) - rowCenterX) > STABLE_ROW_CENTER_TOLERANCE) {
    return 'body center is inconsistent with the row';
  }
  return '';
}

function analyzeCellFrame(raw, width, row, col, enemy) {
  const cellX = col * FRAME_SIZE;
  const cellY = row * FRAME_SIZE;
  const cellRaw = extractCellRaw(raw, width, cellX, cellY);
  const components = findAlphaComponents(cellRaw, FRAME_SIZE, FRAME_SIZE, COMPONENT_ALPHA_THRESHOLD);
  const initialBody = chooseBodyComponent(components, enemy, 0);
  return {
    row,
    rowId: ENEMY_ROWS[row],
    col,
    cellX,
    cellY,
    raw: cellRaw,
    components,
    initialBody,
    body: null,
    cleanedRaw: null,
    hasArt: components.length > 0
  };
}

function buildFrameModel(raw, width, enemy) {
  const frames = [];
  for (let row = 0; row < ENEMY_ROWS.length; row += 1) {
    for (let col = 0; col < SHEET_COLUMNS; col += 1) {
      frames.push(analyzeCellFrame(raw, width, row, col, enemy));
    }
  }
  const stableBodies = frames
    .filter((frame) => BASELINE_BODY_ROWS.includes(frame.rowId) && frame.initialBody)
    .map((frame) => frame.initialBody);
  const fallbackBodies = stableBodies.length ? stableBodies : frames.map((frame) => frame.initialBody).filter(Boolean);
  const referenceArea = getMedian(fallbackBodies.map((component) => component.area));
  const referenceWidth = getMedian(fallbackBodies.map((component) => component.w));
  const referenceHeight = getMedian(fallbackBodies.map((component) => component.h));
  frames.forEach((frame) => {
    frame.body = chooseBodyComponent(frame.components, enemy, referenceArea);
    if (frame.rowId === 'defeat' && !frame.body && frame.components.length) frame.body = frame.components[0];
    frame.cleanedRaw = copyComponents(frame.raw, FRAME_SIZE, FRAME_SIZE, getRenderableComponents(frame.components, frame.body));
  });
  const rowCenters = new Map();
  ENEMY_ROWS.forEach((rowId, row) => {
    const centers = frames
      .filter((frame) => frame.row === row && frame.body)
      .map((frame) => componentCenterX(frame.body));
    rowCenters.set(rowId, getMedian(centers));
  });
  frames.forEach((frame) => {
    frame.invalidReason = getFrameBodyIssue(frame, enemy, referenceArea, rowCenters.get(frame.rowId), referenceWidth, referenceHeight);
    frame.validBody = !frame.invalidReason;
  });
  return { frames, referenceArea, referenceWidth, referenceHeight };
}

function findReplacementFrame(model, frame) {
  const withBody = model.frames.filter((candidate) => candidate.validBody && candidate !== frame);
  const sameRow = withBody
    .filter((candidate) => candidate.row === frame.row)
    .sort((a, b) => Math.abs(a.col - frame.col) - Math.abs(b.col - frame.col));
  return sameRow[0] || null;
}

function clearCellBorder(raw, width, cellX, cellY, border) {
  const clear = Math.max(0, Number(border || 0));
  if (!clear) return;
  for (let y = cellY; y < cellY + FRAME_SIZE; y += 1) {
    for (let x = cellX; x < cellX + FRAME_SIZE; x += 1) {
      const insideX = x - cellX;
      const insideY = y - cellY;
      if (insideX >= clear && insideX < FRAME_SIZE - clear && insideY >= clear && insideY < FRAME_SIZE - clear) continue;
      raw[(y * width + x) * 4 + 3] = 0;
    }
  }
}

function clearCell(raw, width, cellX, cellY) {
  for (let y = cellY; y < cellY + FRAME_SIZE; y += 1) {
    for (let x = cellX; x < cellX + FRAME_SIZE; x += 1) {
      raw[(y * width + x) * 4 + 3] = 0;
    }
  }
}

function copyCell(raw, width, fromCellX, fromCellY, toCellX, toCellY) {
  const cell = extractCellRaw(raw, width, fromCellX, fromCellY);
  clearCell(raw, width, toCellX, toCellY);
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    const targetStart = ((toCellY + y) * width + toCellX) * 4;
    cell.copy(raw, targetStart, y * FRAME_SIZE * 4, (y + 1) * FRAME_SIZE * 4);
  }
  clearCellBorder(raw, width, toCellX, toCellY, CELL_BORDER_CLEAR_PX);
}

function clearAllCellBorders(raw, width) {
  for (let row = 0; row < ENEMY_ROWS.length; row += 1) {
    for (let col = 0; col < SHEET_COLUMNS; col += 1) {
      clearCellBorder(raw, width, col * FRAME_SIZE, row * FRAME_SIZE, CELL_BORDER_CLEAR_PX);
    }
  }
}

async function renderSourceIntoCell(output, sourceRaw, anchor, enemy, rowId, cellX, cellY, options) {
  if (!sourceRaw || !anchor) return false;
  const settings = options || {};
  const contentLimit = getContentLimit(enemy);
  const floating = isFloatingEnemy(enemy);
  let scale = Math.min(contentLimit / Math.max(1, anchor.w), contentLimit / Math.max(1, anchor.h));
  scale = Math.min(rowId === 'defeat' ? 2.1 : 4.2, Math.max(0.2, scale));
  const scaleX = scale * Math.max(0.1, Number(settings.scaleX || 1));
  const scaleY = scale * Math.max(0.1, Number(settings.scaleY || 1));
  const scaledW = Math.max(1, Math.round(FRAME_SIZE * scaleX));
  const scaledH = Math.max(1, Math.round(FRAME_SIZE * scaleY));
  const resized = await sharp(sourceRaw, {
    raw: { width: FRAME_SIZE, height: FRAME_SIZE, channels: 4 }
  })
    .resize(scaledW, scaledH, { fit: 'fill', background: TRANSPARENT, kernel: 'cubic' })
    .raw()
    .toBuffer();
  const anchorCenterX = componentCenterX(anchor) * scaleX;
  const x = cellX + Math.round(FRAME_SIZE / 2 - anchorCenterX);
  const y = floating
    ? cellY + Math.round(FRAME_SIZE / 2 - componentCenterY(anchor) * scaleY)
    : cellY + Math.round(150 - (anchor.maxY + 1) * scaleY);
  pasteRaw(output, SHEET_WIDTH, resized, scaledW, scaledH, x, y);
  return true;
}

function getNearestReplacement(candidates, frame) {
  const viable = (candidates || []).filter((candidate) => candidate && candidate !== frame && candidate.body);
  return viable.sort((a, b) => Math.abs(a.col - frame.col) - Math.abs(b.col - frame.col))[0] ||
    (candidates || []).find((candidate) => candidate && candidate.body) ||
    null;
}

function getFrameAreaRatio(frame, medianArea) {
  return Math.max(1, Number(frame && frame.body && frame.body.area || 1)) / Math.max(1, Number(medianArea || 1));
}

function countCellPixelDiff(raw, width, row, colA, colB, threshold) {
  const y0 = row * FRAME_SIZE;
  const xA = colA * FRAME_SIZE;
  const xB = colB * FRAME_SIZE;
  let changed = 0;
  for (let y = 0; y < FRAME_SIZE; y += 1) {
    for (let x = 0; x < FRAME_SIZE; x += 1) {
      const offsetA = ((y0 + y) * width + xA + x) * 4;
      const offsetB = ((y0 + y) * width + xB + x) * 4;
      if (raw[offsetA + 3] <= 20 && raw[offsetB + 3] <= 20) continue;
      const delta = Math.abs(raw[offsetA] - raw[offsetB]) +
        Math.abs(raw[offsetA + 1] - raw[offsetB + 1]) +
        Math.abs(raw[offsetA + 2] - raw[offsetB + 2]) +
        Math.abs(raw[offsetA + 3] - raw[offsetB + 3]);
      if (delta > threshold) changed += 1;
    }
  }
  return changed;
}

async function writeScaledMotionVariant(raw, width, row, fromCol, toCol, rowId) {
  const fromCellX = fromCol * FRAME_SIZE;
  const cellY = row * FRAME_SIZE;
  const source = extractCellRaw(raw, width, fromCellX, cellY);
  const bounds = getAlphaBounds(source, FRAME_SIZE, 0, 0, FRAME_SIZE, FRAME_SIZE, 8);
  if (!bounds) return false;
  const cropped = cropRaw(source, FRAME_SIZE, bounds);
  const defeat = rowId === 'defeat';
  const scaleX = defeat ? 1.18 : 0.94;
  const scaleY = defeat ? 0.78 : 1.04;
  const outW = Math.max(1, Math.round(bounds.w * scaleX));
  const outH = Math.max(1, Math.round(bounds.h * scaleY));
  const variant = await sharp(cropped, {
    raw: { width: bounds.w, height: bounds.h, channels: 4 }
  })
    .resize(outW, outH, { fit: 'fill', background: TRANSPARENT, kernel: 'cubic' })
    .modulate({ brightness: defeat ? 0.92 : 1.04, saturation: defeat ? 0.92 : 1.02 })
    .raw()
    .toBuffer();
  const targetX = toCol * FRAME_SIZE;
  const sourceCenterX = bounds.x + bounds.w / 2;
  const sourceBottom = bounds.y + bounds.h;
  const x = targetX + Math.round(sourceCenterX - outW / 2 + (defeat ? 8 : 5));
  const y = cellY + Math.round(sourceBottom - outH);
  clearCell(raw, width, targetX, cellY);
  pasteRaw(raw, width, variant, outW, outH, x, y);
  clearCellBorder(raw, width, targetX, cellY, CELL_BORDER_CLEAR_PX);
  return true;
}

async function ensureAnimationRowMotion(raw, enemy) {
  for (const rowId of MOTION_CHECK_ROWS) {
    const row = ENEMY_ROWS.indexOf(rowId);
    if (row < 0) continue;
    const minimumChangedPixels = rowId === 'idle' ? 300 : 900;
    const initialDiff = countCellPixelDiff(raw, SHEET_WIDTH, row, 0, 1, 32);
    if (initialDiff >= minimumChangedPixels) continue;
    let bestCol = -1;
    let bestDiff = 0;
    for (let col = 2; col < SHEET_COLUMNS; col += 1) {
      const diff = countCellPixelDiff(raw, SHEET_WIDTH, row, 0, col, 32);
      if (diff > bestDiff) {
        bestDiff = diff;
        bestCol = col;
      }
    }
    if (bestCol >= 0 && bestDiff >= minimumChangedPixels) {
      copyCell(raw, SHEET_WIDTH, bestCol * FRAME_SIZE, row * FRAME_SIZE, FRAME_SIZE, row * FRAME_SIZE);
      continue;
    }
    await writeScaledMotionVariant(raw, SHEET_WIDTH, row, 0, 1, rowId);
  }
  clearAllCellBorders(raw, SHEET_WIDTH);
  return raw;
}

async function stabilizeNormalizedSheet(output, enemy) {
  const model = buildFrameModel(output, SHEET_WIDTH, enemy);
  for (let row = 0; row < ENEMY_ROWS.length; row += 1) {
    const rowId = ENEMY_ROWS[row];
    const rowFrames = model.frames.filter((frame) => frame.row === row && frame.body);
    if (!rowFrames.length) continue;
    const medianArea = getMedian(rowFrames.map((frame) => frame.body.area));
    const medianCenter = getMedian(rowFrames.map((frame) => componentCenterX(frame.body)));
    const centers = rowFrames.map((frame) => componentCenterX(frame.body));
    const areas = rowFrames.map((frame) => Math.max(1, frame.body.area));
    const minCenter = Math.min(...centers);
    const maxCenter = Math.max(...centers);
    const minArea = Math.min(...areas);
    const maxArea = Math.max(...areas);
    const rowCenterSpread = maxCenter - minCenter;
    const rowAreaSpread = maxArea / Math.max(1, minArea);
    const stableCandidates = rowFrames.filter((frame) => (
      !frame.invalidReason &&
      getFrameAreaRatio(frame, medianArea) <= 1.9 &&
      getFrameAreaRatio(frame, medianArea) >= 0.52 &&
      (!STABLE_BODY_ROWS.includes(rowId) || Math.abs(componentCenterX(frame.body) - medianCenter) <= STABLE_ROW_CENTER_TOLERANCE * 0.58)
    ));
    const defeatCandidates = rowId === 'defeat'
      ? rowFrames.slice().sort((a, b) => {
        const heightDiff = a.body.h - b.body.h;
        return heightDiff || a.body.area - b.body.area;
      })
      : [];
    for (const frame of rowFrames) {
      const areaRatio = getFrameAreaRatio(frame, medianArea);
      const centerDrift = Math.abs(componentCenterX(frame.body) - medianCenter);
      const frameArea = Math.max(1, frame.body.area);
      const frameCenter = componentCenterX(frame.body);
      const rowAreaOutlier = rowId !== 'defeat' && (
        areaRatio > 2.25 ||
        areaRatio < 0.55 ||
        (rowAreaSpread > 2.55 && (frameArea === maxArea || frameArea === minArea))
      );
      const rowCenterOutlier = STABLE_BODY_ROWS.includes(rowId) && (
        centerDrift > STABLE_ROW_CENTER_TOLERANCE * 0.58 ||
        (rowCenterSpread > STABLE_ROW_CENTER_TOLERANCE * 0.86 && (frameCenter === minCenter || frameCenter === maxCenter))
      );
      const needsRepair = !!frame.invalidReason || rowAreaOutlier || rowCenterOutlier;
      if (!needsRepair) continue;
      const replacement = getNearestReplacement(stableCandidates, frame) ||
        getNearestReplacement(rowId === 'defeat' ? defeatCandidates : rowFrames, frame);
      if (!replacement) continue;
      clearCell(output, SHEET_WIDTH, frame.cellX, frame.cellY);
      await renderSourceIntoCell(
        output,
        replacement.cleanedRaw,
        replacement.body,
        enemy,
        frame.rowId,
        frame.cellX,
        frame.cellY,
        rowId === 'defeat' && (!stableCandidates.length || frame.invalidReason) ? { scaleX: 1.24, scaleY: 0.42 } : null
      );
      clearCellBorder(output, SHEET_WIDTH, frame.cellX, frame.cellY, CELL_BORDER_CLEAR_PX);
    }
  }
  clearAllCellBorders(output, SHEET_WIDTH);
  return ensureAnimationRowMotion(output, enemy);
}

async function repairNormalizedSheet(output, enemy) {
  const model = buildFrameModel(output, SHEET_WIDTH, enemy);
  for (const frame of model.frames) {
    if (frame.validBody) continue;
    const replacement = findReplacementFrame(model, frame);
    if (!replacement) continue;
    clearCell(output, SHEET_WIDTH, frame.cellX, frame.cellY);
    await renderSourceIntoCell(output, replacement.cleanedRaw, replacement.body, enemy, frame.rowId, frame.cellX, frame.cellY);
    clearCellBorder(output, SHEET_WIDTH, frame.cellX, frame.cellY, CELL_BORDER_CLEAR_PX);
  }
  clearAllCellBorders(output, SHEET_WIDTH);
  return stabilizeNormalizedSheet(output, enemy);
}

async function normalizeSheet(raw, width, height, enemy) {
  const output = Buffer.alloc(SHEET_WIDTH * SHEET_HEIGHT * 4);
  const model = buildFrameModel(raw, width, enemy);
  for (const frame of model.frames) {
    if (frame.validBody) {
      await renderSourceIntoCell(output, frame.cleanedRaw, frame.body, enemy, frame.rowId, frame.cellX, frame.cellY);
    } else {
      const replacement = findReplacementFrame(model, frame);
      if (!replacement) {
        throw new Error(`${enemy.id} ${frame.rowId}:${frame.col + 1} is malformed (${frame.invalidReason || 'invalid frame'}) and has no valid same-row replacement`);
      }
      if (replacement) {
        await renderSourceIntoCell(output, replacement.cleanedRaw, replacement.body, enemy, frame.rowId, frame.cellX, frame.cellY);
        if (frame.hasArt && frame.rowId !== 'defeat' && !frame.invalidReason) {
          pasteRaw(output, SHEET_WIDTH, frame.cleanedRaw, FRAME_SIZE, FRAME_SIZE, frame.cellX, frame.cellY);
        }
      }
    }
    clearCellBorder(output, SHEET_WIDTH, frame.cellX, frame.cellY, CELL_BORDER_CLEAR_PX);
  }
  clearAllCellBorders(output, SHEET_WIDTH);
  return repairNormalizedSheet(output, enemy);
}

async function processEnemy(enemy, sourceDir) {
  const sourcePath = getSourcePath(enemy, sourceDir);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing raw AI sheet for ${enemy.id}: ${path.relative(ROOT, sourcePath)}`);
  }
  const keyColor = colorFromHex(getKeyColor(enemy));
  const { data, info } = await sharp(sourcePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const sourceGrid = detectGuideGrid(data, info.width, info.height, {
    columns: SHEET_COLUMNS,
    rows: ENEMY_ROWS.length,
    label: `${enemy.id} raw enemy sheet ${path.relative(ROOT, sourcePath).replace(/\\/g, '/')}`,
    guideColor: GUIDE_LINE_HEX
  });
  removeBackgroundFlood(data, info.width, info.height, keyColor);
  const standardized = await standardizeSourceSheet(data, info.width, info.height, enemy, sourcePath, sourceGrid);
  const normalized = await normalizeSheet(standardized, SHEET_WIDTH, SHEET_HEIGHT, enemy);
  const sheetPath = path.join(ROOT, enemy.animation.sheet);
  fs.mkdirSync(path.dirname(sheetPath), { recursive: true });
  await writeSheetRawPng(normalized, SHEET_WIDTH, SHEET_HEIGHT, sheetPath);
  const finalSheet = await enforceSavedRuntimeMotion(enemy, sheetPath) || normalized;
  await writePortraitFromSheet(enemy, finalSheet);
  return [
    path.relative(ROOT, sheetPath).replace(/\\/g, '/'),
    enemy.asset
  ];
}

async function writeSheetRawPng(raw, width, height, targetPath) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  const tempPath = path.join(
    path.dirname(targetPath),
    `.${path.basename(targetPath)}.${process.pid}.${Date.now()}.tmp.png`
  );
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

async function enforceSavedRuntimeMotion(enemy, sheetPath) {
  const saved = await sharp(sheetPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (saved.info.width !== SHEET_WIDTH || saved.info.height !== SHEET_HEIGHT) return null;
  const repaired = Buffer.from(saved.data);
  await ensureAnimationRowMotion(repaired, enemy);
  await writeSheetRawPng(repaired, SHEET_WIDTH, SHEET_HEIGHT, sheetPath);
  return repaired;
}

async function repairRuntimeEnemy(enemy) {
  const sheetPath = path.join(ROOT, enemy.animation.sheet);
  if (!fs.existsSync(sheetPath)) {
    throw new Error(`Missing runtime enemy sheet for ${enemy.id}: ${enemy.animation.sheet}`);
  }
  const { data, info } = await sharp(sheetPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  if (info.width !== SHEET_WIDTH || info.height !== SHEET_HEIGHT) {
    throw new Error(`${enemy.id} runtime sheet must be ${SHEET_WIDTH}x${SHEET_HEIGHT}`);
  }
  const repaired = await repairNormalizedSheet(Buffer.from(data), enemy);
  await ensureAnimationRowMotion(repaired, enemy);
  await writeSheetRawPng(repaired, SHEET_WIDTH, SHEET_HEIGHT, sheetPath);
  const finalSheet = await enforceSavedRuntimeMotion(enemy, sheetPath) || repaired;
  await writePortraitFromSheet(enemy, finalSheet);
  return [
    path.relative(ROOT, sheetPath).replace(/\\/g, '/'),
    enemy.asset
  ];
}

async function writePortraitFromSheet(enemy, sheetRaw) {
  const bounds = getAlphaBounds(sheetRaw, SHEET_WIDTH, 0, 0, FRAME_SIZE, FRAME_SIZE, 8) || { x: 0, y: 0, w: FRAME_SIZE, h: FRAME_SIZE };
  const cropped = cropRaw(sheetRaw, SHEET_WIDTH, bounds);
  const portraitPath = path.join(ROOT, enemy.asset);
  fs.mkdirSync(path.dirname(portraitPath), { recursive: true });
  await sharp(cropped, {
    raw: { width: bounds.w, height: bounds.h, channels: 4 }
  })
    .resize(280, 280, { fit: 'contain', background: TRANSPARENT, kernel: 'cubic' })
    .extend({ top: 20, bottom: 20, left: 20, right: 20, background: TRANSPARENT })
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(portraitPath);
}

async function readPngRaw(filePath) {
  return sharp(filePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertTransparentCorners(raw, width, height, label) {
  [
    0,
    (width - 1) * 4,
    ((height - 1) * width) * 4,
    ((height * width) - 1) * 4
  ].forEach((offset) => {
    assert(raw[offset + 3] === 0, `${label} should have transparent corners`);
  });
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

function frameTouchesCellEdge(raw, width, x0, y0, threshold) {
  for (let x = x0; x < x0 + FRAME_SIZE; x += 1) {
    if (raw[(y0 * width + x) * 4 + 3] > threshold) return true;
    if (raw[((y0 + FRAME_SIZE - 1) * width + x) * 4 + 3] > threshold) return true;
  }
  for (let y = y0; y < y0 + FRAME_SIZE; y += 1) {
    if (raw[(y * width + x0) * 4 + 3] > threshold) return true;
    if (raw[(y * width + x0 + FRAME_SIZE - 1) * 4 + 3] > threshold) return true;
  }
  return false;
}

function frameTouchesRectEdge(raw, width, x0, y0, frameWidth, frameHeight, threshold) {
  for (let x = x0; x < x0 + frameWidth; x += 1) {
    if (raw[(y0 * width + x) * 4 + 3] > threshold) return true;
    if (raw[((y0 + frameHeight - 1) * width + x) * 4 + 3] > threshold) return true;
  }
  for (let y = y0; y < y0 + frameHeight; y += 1) {
    if (raw[(y * width + x0) * 4 + 3] > threshold) return true;
    if (raw[(y * width + x0 + frameWidth - 1) * 4 + 3] > threshold) return true;
  }
  return false;
}

function getEnemyAnimationAuditLayout(enemy) {
  const animation = enemy && enemy.animation || {};
  const frameWidth = Math.max(1, Number(animation.frameWidth) || FRAME_SIZE);
  const frameHeight = Math.max(1, Number(animation.frameHeight) || FRAME_SIZE);
  const states = Object.entries(animation.states || {});
  const columns = states.reduce((max, [, state]) => Math.max(max, Math.max(1, Number(state.frames) || 1)), 1);
  const rows = states.reduce((max, [, state]) => Math.max(max, Math.max(0, Number(state.row || 0)) + 1), ENEMY_ROWS.length);
  return {
    frameWidth,
    frameHeight,
    columns,
    rows,
    width: columns * frameWidth,
    height: rows * frameHeight,
    states,
    standard: frameWidth === FRAME_SIZE && frameHeight === FRAME_SIZE && columns === SHEET_COLUMNS && rows === ENEMY_ROWS.length
  };
}

async function validateEnemy(enemy) {
  const sheetPath = path.join(ROOT, enemy.animation.sheet);
  const portraitPath = path.join(ROOT, enemy.asset);
  const layout = getEnemyAnimationAuditLayout(enemy);
  assert(fs.existsSync(sheetPath), `Missing enemy sheet: ${enemy.animation.sheet}`);
  assert(fs.existsSync(portraitPath), `Missing enemy portrait: ${enemy.asset}`);
  const sheet = await readPngRaw(sheetPath);
  assert(sheet.info.width === layout.width && sheet.info.height === layout.height, `${enemy.id} sheet must be ${layout.width}x${layout.height}`);
  assertTransparentCorners(sheet.data, sheet.info.width, sheet.info.height, enemy.id);
  ENEMY_ROWS.forEach((row) => {
    const state = enemy.animation.states && enemy.animation.states[row] || { row: ENEMY_ROWS.indexOf(row), frames: layout.columns };
    const rowIndex = Math.max(0, Number(state.row || 0));
    const frames = Math.max(1, Number(state.frames || layout.columns));
    const rowBounds = getAlphaBounds(sheet.data, sheet.info.width, 0, rowIndex * layout.frameHeight, frames * layout.frameWidth, layout.frameHeight, 8);
    assert(rowBounds && rowBounds.w >= 24 && rowBounds.h >= 24, `${enemy.id} ${row} row should contain visible art`);
    for (let col = 0; col < frames; col += 1) {
      const cellX = col * layout.frameWidth;
      const cellY = rowIndex * layout.frameHeight;
      const frameBounds = getAlphaBounds(sheet.data, sheet.info.width, cellX, cellY, layout.frameWidth, layout.frameHeight, 8);
      const area = getAlphaArea(sheet.data, sheet.info.width, cellX, cellY, layout.frameWidth, layout.frameHeight, 8);
      assert(frameBounds && frameBounds.w >= 4 && frameBounds.h >= 4 && area >= 28,
        `${enemy.id} ${row}:${col + 1} frame should contain visible art`);
      assert(!frameTouchesRectEdge(sheet.data, sheet.info.width, cellX, cellY, layout.frameWidth, layout.frameHeight, 8),
        `${enemy.id} ${row}:${col + 1} frame should not touch the cell edge`);
    }
  });
  const portrait = await readPngRaw(portraitPath);
  assert(portrait.info.width === PORTRAIT_SIZE && portrait.info.height === PORTRAIT_SIZE, `${enemy.id} portrait must be ${PORTRAIT_SIZE}x${PORTRAIT_SIZE}`);
  assertTransparentCorners(portrait.data, portrait.info.width, portrait.info.height, `${enemy.id} portrait`);
}

function makeIssue(code, message, extra) {
  return Object.assign({ code, message }, extra || {});
}

function isBlockingBodyIssue(reason, options) {
  if (options && options.strict) return true;
  const text = String(reason || '');
  return text.includes('missing') ||
    text.includes('cropped');
}

function getSheetPath(assetPath) {
  return path.resolve(ROOT, String(assetPath || '').replace(/^[/\\]+/, ''));
}

function getAnimationFrameArea(raw, width, animation, stateId, frameIndex) {
  const state = animation.states[stateId];
  const frameWidth = Math.max(1, Number(animation.frameWidth) || FRAME_SIZE);
  const frameHeight = Math.max(1, Number(animation.frameHeight) || FRAME_SIZE);
  return getAlphaArea(
    raw,
    width,
    frameIndex * frameWidth,
    Math.max(0, Number(state.row || 0)) * frameHeight,
    frameWidth,
    frameHeight,
    8
  );
}

function pushAuditIssue(report, severity, issue, options) {
  if (severity === 'warning' && options && options.strict) {
    report.errors.push(issue);
    return;
  }
  report[severity === 'error' ? 'errors' : 'warnings'].push(issue);
}

async function auditEnemySheet(enemy, options) {
  const errors = [];
  const warnings = [];
  const sheetPath = getSheetPath(enemy && enemy.animation && enemy.animation.sheet);
  const layout = getEnemyAnimationAuditLayout(enemy);
  const report = {
    id: enemy.id,
    name: enemy.name,
    fileId: getFileId(enemy),
    sheet: enemy.animation && enemy.animation.sheet || '',
    errors,
    warnings
  };
  if (!fs.existsSync(sheetPath)) {
    errors.push(makeIssue('missing-sheet', `Missing enemy sheet: ${report.sheet}`));
    return report;
  }
  const sheet = await readPngRaw(sheetPath);
  if (sheet.info.width !== layout.width || sheet.info.height !== layout.height) {
    errors.push(makeIssue('invalid-dimensions', `${enemy.id} sheet must be ${layout.width}x${layout.height}`, {
      width: sheet.info.width,
      height: sheet.info.height,
      expectedWidth: layout.width,
      expectedHeight: layout.height
    }));
    return report;
  }
  try {
    assertTransparentCorners(sheet.data, sheet.info.width, sheet.info.height, enemy.id);
  } catch (error) {
    errors.push(makeIssue('opaque-corner', error.message));
  }
  if (!layout.standard) {
    layout.states.forEach(([stateId, state]) => {
      const row = Math.max(0, Number(state.row || 0));
      const frames = Math.max(1, Number(state.frames) || 1);
      for (let col = 0; col < frames; col += 1) {
        const cellX = col * layout.frameWidth;
        const cellY = row * layout.frameHeight;
        const label = `${stateId}:${col + 1}`;
        const area = getAlphaArea(sheet.data, sheet.info.width, cellX, cellY, layout.frameWidth, layout.frameHeight, 8);
        if (area < 28) {
          pushAuditIssue(report, 'error', makeIssue('empty-frame', `${enemy.id} ${label} frame should contain visible art`, { row: stateId, frame: col + 1 }), options);
        }
        if (frameTouchesRectEdge(sheet.data, sheet.info.width, cellX, cellY, layout.frameWidth, layout.frameHeight, 8)) {
          pushAuditIssue(report, 'error', makeIssue('edge-touch', `${enemy.id} ${label} frame should leave a transparent cell border`, { row: stateId, frame: col + 1 }), options);
        }
      }
    });
    return report;
  }
  const model = buildFrameModel(sheet.data, sheet.info.width, enemy);
  model.frames.forEach((frame) => {
    const label = `${frame.rowId}:${frame.col + 1}`;
    const area = getAlphaArea(sheet.data, sheet.info.width, frame.cellX, frame.cellY, FRAME_SIZE, FRAME_SIZE, 8);
    if (area < 28) {
      pushAuditIssue(report, 'error', makeIssue('empty-frame', `${enemy.id} ${label} frame should contain visible art`, { row: frame.rowId, frame: frame.col + 1 }), options);
    }
    if (frameTouchesCellEdge(sheet.data, sheet.info.width, frame.cellX, frame.cellY, 8)) {
      pushAuditIssue(report, 'error', makeIssue('edge-touch', `${enemy.id} ${label} frame should leave a transparent cell border`, { row: frame.rowId, frame: frame.col + 1 }), options);
    }
    if (frame.invalidReason) {
      const issue = makeIssue('body-consistency', `${enemy.id} ${label}: ${frame.invalidReason}`, {
        row: frame.rowId,
        frame: frame.col + 1
      });
      pushAuditIssue(report, isBlockingBodyIssue(frame.invalidReason, options) ? 'error' : 'warning', issue, options);
    }
  });
  ENEMY_ROWS.forEach((rowId, row) => {
    const rowFrames = model.frames.filter((frame) => frame.row === row && frame.body);
    if (rowFrames.length < SHEET_COLUMNS) {
      pushAuditIssue(report, 'warning', makeIssue('row-body-coverage', `${enemy.id} ${rowId} row has ${rowFrames.length}/${SHEET_COLUMNS} detected bodies`, { row: rowId }), options);
      return;
    }
    const centers = rowFrames.map((frame) => componentCenterX(frame.body));
    const minCenter = Math.min(...centers);
    const maxCenter = Math.max(...centers);
    if (STABLE_BODY_ROWS.includes(rowId) && maxCenter - minCenter > STABLE_ROW_CENTER_TOLERANCE) {
      pushAuditIssue(report, 'warning', makeIssue('row-center-drift', `${enemy.id} ${rowId} row center drifts ${Math.round(maxCenter - minCenter)}px`, { row: rowId }), options);
    }
    const areas = rowFrames.map((frame) => Math.max(1, frame.body.area));
    const minArea = Math.min(...areas);
    const maxArea = Math.max(...areas);
    if (rowId !== 'defeat' && maxArea / minArea > 2.9) {
      pushAuditIssue(report, 'warning', makeIssue('row-area-drift', `${enemy.id} ${rowId} row body area varies ${Number((maxArea / minArea).toFixed(2))}x`, { row: rowId }), options);
    }
  });
  return report;
}

function collectAnimationAssets() {
  const entries = [];
  const add = (category, id, animation) => {
    if (!animation || !animation.sheet || !animation.states) return;
    entries.push({ category, id, animation });
  };
  add('player', 'generic', Data.GENERIC_PLAYER_ANIMATION_ASSET);
  Object.entries(Data.PLAYER_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('player', id, animation));
  Object.entries(Data.EQUIPMENT_VISUALS || {}).forEach(([id, visual]) => add('equipment', id, visual && visual.animation));
  Object.entries(Data.FX_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('fx', id, animation));
  Object.entries(Data.SKILL_FX_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('skill-fx', id, animation));
  Object.entries(Data.BASIC_ATTACK_FX_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('basic-fx', id, animation));
  Object.entries(Data.ENEMY_COMBAT_FX_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('enemy-fx', id, animation));
  Object.entries(Data.ENEMY_PROJECTILE_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('enemy-projectile', id, animation));
  Object.entries(Data.PORTAL_ANIMATION_ASSETS || {}).forEach(([id, animation]) => add('portal', id, animation));
  return entries;
}

async function auditAnimationAsset(entry) {
  const errors = [];
  const warnings = [];
  const animation = entry.animation;
  const sheetPath = getSheetPath(animation.sheet);
  const report = {
    category: entry.category,
    id: entry.id,
    sheet: animation.sheet,
    errors,
    warnings
  };
  if (!fs.existsSync(sheetPath)) {
    errors.push(makeIssue('missing-sheet', `Missing ${entry.category} animation sheet: ${animation.sheet}`));
    return report;
  }
  const sheet = await readPngRaw(sheetPath);
  const frameWidth = Math.max(1, Number(animation.frameWidth) || FRAME_SIZE);
  const frameHeight = Math.max(1, Number(animation.frameHeight) || FRAME_SIZE);
  const states = Object.entries(animation.states || {});
  const expectedWidth = states.reduce((max, [, state]) => Math.max(max, (Math.max(1, Number(state.frames) || 1)) * frameWidth), frameWidth);
  const expectedHeight = states.reduce((max, [, state]) => Math.max(max, (Math.max(0, Number(state.row || 0)) + 1) * frameHeight), frameHeight);
  if (sheet.info.width < expectedWidth || sheet.info.height < expectedHeight) {
    errors.push(makeIssue('invalid-dimensions', `${entry.category} ${entry.id} sheet is too small for its animation states`, {
      width: sheet.info.width,
      height: sheet.info.height,
      expectedWidth,
      expectedHeight
    }));
    return report;
  }
  states.forEach(([stateId, state]) => {
    const frames = Math.max(1, Number(state.frames) || 1);
    for (let frame = 0; frame < frames; frame += 1) {
      const area = getAnimationFrameArea(sheet.data, sheet.info.width, animation, stateId, frame);
      if (area < 10) {
        warnings.push(makeIssue('empty-frame', `${entry.category} ${entry.id} ${stateId}:${frame + 1} has little or no visible art`, {
          state: stateId,
          frame: frame + 1
        }));
      }
    }
  });
  return report;
}

async function runAudit(targetEnemies, options) {
  const enemies = targetEnemies && targetEnemies.length ? targetEnemies : Data.ENEMIES;
  const enemyReports = [];
  for (const enemy of enemies) {
    enemyReports.push(await auditEnemySheet(enemy, options || {}));
  }
  const animationReports = [];
  for (const entry of collectAnimationAssets()) {
    animationReports.push(await auditAnimationAsset(entry));
  }
  const countIssues = (reports, key) => reports.reduce((total, report) => total + (report[key] || []).length, 0);
  const totals = {
    errors: countIssues(enemyReports, 'errors') + countIssues(animationReports, 'errors'),
    warnings: countIssues(enemyReports, 'warnings') + countIssues(animationReports, 'warnings')
  };
  return {
    generatedAt: new Date().toISOString(),
    strict: !!(options && options.strict),
    sheet: {
      enemyColumns: SHEET_COLUMNS,
      enemyRows: ENEMY_ROWS,
      frameSize: FRAME_SIZE
    },
    checked: {
      enemies: enemyReports.length,
      animationAssets: animationReports.length
    },
    totals,
    enemies: enemyReports,
    animationAssets: animationReports
  };
}

function formatAuditReport(report) {
  const lines = [
    'Project Starfall sprite audit',
    `Checked ${report.checked.enemies} enemy sheets and ${report.checked.animationAssets} animation assets.`,
    `Errors: ${report.totals.errors}; warnings: ${report.totals.warnings}.`
  ];
  const collect = (reports, key, labelForReport) => reports.flatMap((entry) => (entry[key] || []).map((issue) => ({
    label: labelForReport(entry),
    issue
  })));
  const errors = collect(report.enemies, 'errors', (entry) => `enemy ${entry.id}`)
    .concat(collect(report.animationAssets, 'errors', (entry) => `${entry.category} ${entry.id}`));
  const warnings = collect(report.enemies, 'warnings', (entry) => `enemy ${entry.id}`)
    .concat(collect(report.animationAssets, 'warnings', (entry) => `${entry.category} ${entry.id}`));
  errors.slice(0, 30).forEach(({ label, issue }) => lines.push(`ERROR ${label}: ${issue.message}`));
  warnings.slice(0, 30).forEach(({ label, issue }) => lines.push(`WARN ${label}: ${issue.message}`));
  if (errors.length > 30) lines.push(`... ${errors.length - 30} more errors`);
  if (warnings.length > 30) lines.push(`... ${warnings.length - 30} more warnings`);
  if (!report.totals.errors) lines.push('Audit passed with no blocking sprite-sheet errors.');
  return lines.join('\n');
}

function escapeSvgText(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function makeLabelSvg(width, height, text) {
  return Buffer.from([
    `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`,
    '<rect width="100%" height="100%" fill="#111827"/>',
    `<text x="10" y="${Math.round(height / 2 + 6)}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#f8fafc">${escapeSvgText(text)}</text>`,
    '</svg>'
  ].join(''));
}

async function makeReviewTile(enemy, sourceDir) {
  const tileW = 520;
  const tileH = 400;
  const labelH = 40;
  const sourcePath = getSourcePath(enemy, sourceDir);
  const sheetPath = getSheetPath(enemy.animation && enemy.animation.sheet);
  const composites = [{
    input: makeLabelSvg(tileW, labelH, `${enemy.id} - ${enemy.name}`),
    left: 0,
    top: 0
  }];
  if (fs.existsSync(sourcePath)) {
    composites.push({
      input: await sharp(sourcePath).resize(250, 330, { fit: 'contain', background: '#0b1020' }).png().toBuffer(),
      left: 0,
      top: labelH + 15
    });
  }
  if (fs.existsSync(sheetPath)) {
    composites.push({
      input: await sharp(sheetPath).resize(250, 330, { fit: 'contain', background: '#0b1020' }).png().toBuffer(),
      left: 270,
      top: labelH + 15
    });
  }
  return sharp({
    create: {
      width: tileW,
      height: tileH,
      channels: 4,
      background: '#0b1020'
    }
  })
    .composite(composites)
    .png()
    .toBuffer();
}

async function writeReviewMontage(targetEnemies, sourceDir, reviewDir) {
  const enemies = targetEnemies && targetEnemies.length ? targetEnemies : Data.ENEMIES;
  const outDir = path.resolve(ROOT, reviewDir || DEFAULT_REVIEW_DIR);
  fs.mkdirSync(outDir, { recursive: true });
  const tileW = 520;
  const tileH = 400;
  const columns = 2;
  const rows = Math.ceil(enemies.length / columns);
  const composites = [];
  for (let index = 0; index < enemies.length; index += 1) {
    const tile = await makeReviewTile(enemies[index], sourceDir);
    composites.push({
      input: tile,
      left: (index % columns) * tileW,
      top: Math.floor(index / columns) * tileH
    });
  }
  const montagePath = path.join(outDir, 'enemy-sheet-review.png');
  await sharp({
    create: {
      width: columns * tileW,
      height: rows * tileH,
      channels: 4,
      background: '#020617'
    }
  })
    .composite(composites)
    .png({ compressionLevel: 9, adaptiveFiltering: true })
    .toFile(montagePath);
  process.stdout.write(`Wrote ${path.relative(ROOT, montagePath).replace(/\\/g, '/')}\n`);
}

async function writePromptManifest(target) {
  const manifest = buildPromptManifest();
  const text = `${JSON.stringify(manifest, null, 2)}\n`;
  if (target === '-') {
    process.stdout.write(text);
    return;
  }
  const outPath = path.resolve(ROOT, target || DEFAULT_PROMPT_PATH);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, text);
  process.stdout.write(`Wrote ${path.relative(ROOT, outPath).replace(/\\/g, '/')}\n`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  if (options.writePrompts) {
    await writePromptManifest(options.writePrompts);
    if (!options.all && !options.enemyId && !options.validate && !options.audit && !options.review) return;
  }
  const enemies = options.all
    ? Data.ENEMIES.slice()
    : options.enemyId ? [getEnemyById(options.enemyId)].filter(Boolean) : [];
  if (options.enemyId && !enemies.length) throw new Error(`Unknown enemy: ${options.enemyId}`);
  const generated = [];
  for (const enemy of enemies) {
    generated.push(...await (options.repairRuntime ? repairRuntimeEnemy(enemy) : processEnemy(enemy, options.sourceDir)));
  }
  if (options.validate || generated.length) {
    const validationTargets = enemies.length ? enemies : Data.ENEMIES;
    for (const enemy of validationTargets) await validateEnemy(enemy);
  }
  if (options.audit) {
    const auditTargets = enemies.length ? enemies : Data.ENEMIES;
    const report = await runAudit(auditTargets, { strict: options.strict });
    process.stdout.write(options.json ? `${JSON.stringify(report, null, 2)}\n` : `${formatAuditReport(report)}\n`);
    if (report.totals.errors) process.exitCode = 1;
    return;
  }
  if (options.review) {
    const reviewTargets = enemies.length ? enemies : Data.ENEMIES;
    await writeReviewMontage(reviewTargets, options.sourceDir, options.review);
    return;
  }
  generated.forEach((filePath) => process.stdout.write(`Processed ${filePath}\n`));
  if (!options.writePrompts && !options.validate && !options.audit && !options.review && !generated.length) process.stdout.write(`${usage()}\n`);
}

main().catch((error) => {
  console.error(error && error.message ? error.message : error);
  process.exit(1);
});
