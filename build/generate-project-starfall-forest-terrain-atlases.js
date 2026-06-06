#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = path.join(ROOT, 'img/project-starfall/environment/source');
const TERRAIN_DIR = path.join(ROOT, 'img/project-starfall/environment/terrain');
const CELL = 64;
const COLUMNS = 8;
const ROWS = 4;
const ATLAS_WIDTH = CELL * COLUMNS;
const ATLAS_HEIGHT = CELL * ROWS;
const REPEAT_CELLS = new Set([1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31]);
const NATURAL_CAP_CELLS = new Set([0, 3, 4, 7]);
const LEFT_CAP_CELLS = new Set([0, 4]);
const RIGHT_CAP_CELLS = new Set([3, 7]);

const THEMES = Object.freeze({
  'greenroot-meadow': {
    source: 'greenroot-meadow-terrain-imagegen-source.png',
    keyed: 'greenroot-meadow-terrain-imagegen-keyed.png',
    output: 'greenroot-meadow.png',
    row4LongUndersideColumns: [4, 5],
    shadow: { r: 38, g: 31, b: 24, a: 96 },
    palette: {
      grass: '#65bd47',
      grassLight: '#b4ea66',
      grassDark: '#2f7b38',
      moss: '#2f934a',
      soil: '#7b5733',
      soilMid: '#68472b',
      soilDark: '#382b21',
      root: '#8a663f',
      stone: '#817b64',
      flower: '#f1db72',
      accent: '#f0f4aa'
    }
  },
  'thornpath-thicket': {
    source: 'thornpath-thicket-terrain-imagegen-source.png',
    keyed: 'thornpath-thicket-terrain-imagegen-keyed.png',
    output: 'thornpath-thicket.png',
    row4LongUndersideColumns: [4, 5],
    shadow: { r: 24, g: 24, b: 20, a: 108 },
    palette: {
      grass: '#4c9352',
      grassLight: '#83bd56',
      grassDark: '#1f4e35',
      moss: '#2c6f42',
      soil: '#4e392b',
      soilMid: '#3d2d24',
      soilDark: '#211d19',
      root: '#6f4f32',
      stone: '#626458',
      flower: '#995066',
      accent: '#b98e50'
    }
  },
  'bramble-depths': {
    source: 'thornpath-thicket-terrain-imagegen-source.png',
    keyed: 'bramble-depths-terrain-imagegen-keyed.png',
    output: 'bramble-depths.png',
    row4LongUndersideColumns: [4, 5],
    modulate: { brightness: 0.76, saturation: 1.08 },
    shadow: { r: 18, g: 18, b: 18, a: 116 },
    palette: {
      grass: '#355f3f',
      grassLight: '#6aa86b',
      grassDark: '#1f2f28',
      moss: '#2f6f42',
      soil: '#513325',
      soilMid: '#3d2b24',
      soilDark: '#1f1c18',
      root: '#6f4f32',
      stone: '#626458',
      flower: '#e05b75',
      accent: '#ff7c9b'
    }
  },
  'brambleking-court': {
    source: 'thornpath-thicket-terrain-imagegen-source.png',
    keyed: 'brambleking-court-terrain-imagegen-keyed.png',
    output: 'brambleking-court.png',
    row4LongUndersideColumns: [4, 5],
    modulate: { brightness: 0.9, saturation: 1.12 },
    shadow: { r: 24, g: 22, b: 20, a: 112 },
    palette: {
      grass: '#416f45',
      grassLight: '#9bc776',
      grassDark: '#213728',
      moss: '#345f3f',
      soil: '#513325',
      soilMid: '#3d2b24',
      soilDark: '#1f1c18',
      root: '#6f4f32',
      stone: '#626458',
      flower: '#e05b75',
      accent: '#ff7c9b'
    }
  }
});

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function isMagentaKey(r, g, b, a) {
  if (a <= 2) return true;
  return r >= 160 && b >= 150 && g <= 120 && (r + b - g) >= 260;
}

function isSourceKeyPixel(r, g, b, a) {
  if (isMagentaKey(r, g, b, a)) return true;
  return r > 72 && b > 72 && g < 58 && Math.min(r, b) > g * 1.6 && Math.abs(r - b) < 95;
}

function removeConnectedKeyBackground(raw, info) {
  const { width, height, channels } = info;
  const data = Buffer.from(raw);
  const background = new Uint8Array(width * height);
  const queue = [];

  function add(x, y) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const pixel = y * width + x;
    if (background[pixel]) return;
    const offset = pixel * channels;
    if (!isSourceKeyPixel(data[offset], data[offset + 1], data[offset + 2], data[offset + 3])) return;
    background[pixel] = 1;
    queue.push(pixel);
  }

  for (let x = 0; x < width; x += 1) {
    add(x, 0);
    add(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    add(0, y);
    add(width - 1, y);
  }

  for (let index = 0; index < queue.length; index += 1) {
    const pixel = queue[index];
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    add(x + 1, y);
    add(x - 1, y);
    add(x, y + 1);
    add(x, y - 1);
    add(x + 1, y + 1);
    add(x - 1, y - 1);
    add(x + 1, y - 1);
    add(x - 1, y + 1);
  }

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    if (!background[pixel]) continue;
    const offset = pixel * channels;
    data[offset] = 0;
    data[offset + 1] = 0;
    data[offset + 2] = 0;
    data[offset + 3] = 0;
  }

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const pixel = y * width + x;
      const offset = pixel * channels;
      if (data[offset + 3] === 0) continue;
      let nearTransparent = false;
      for (let yy = -2; yy <= 2 && !nearTransparent; yy += 1) {
        for (let xx = -2; xx <= 2; xx += 1) {
          const nx = x + xx;
          const ny = y + yy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          if (data[(ny * width + nx) * channels + 3] === 0) {
            nearTransparent = true;
            break;
          }
        }
      }
      if (!nearTransparent) continue;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const magentaDominant = Math.min(r, b) > Math.max(55, g * 1.35) && Math.abs(r - b) < 90;
      if (!magentaDominant) continue;
      if (r > 95 && b > 95 && g < 95) {
        data[offset + 3] = Math.round(data[offset + 3] * 0.16);
      } else {
        const target = Math.max(g + 18, Math.min(r, b) * 0.48);
        data[offset] = Math.min(r, target);
        data[offset + 2] = Math.min(b, target);
      }
    }
  }

  return data;
}

async function applySourceModulation(raw, info, config) {
  const modulate = config.modulate || null;
  if (!modulate) return raw;
  const transformed = await sharp(raw, { raw: info })
    .modulate({
      brightness: Number(modulate.brightness || 1),
      saturation: Number(modulate.saturation || 1),
      hue: Number(modulate.hue || 0)
    })
    .ensureAlpha()
    .raw()
    .toBuffer();
  return Buffer.from(transformed);
}

async function createKeyedSource(theme, config) {
  const sourcePath = path.join(SOURCE_DIR, config.source);
  const keyedPath = path.join(SOURCE_DIR, config.keyed);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(`Missing forest terrain source: ${path.relative(ROOT, sourcePath)}`);
  }

  const image = await sharp(sourcePath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  let data = removeConnectedKeyBackground(image.data, image.info);
  data = await applySourceModulation(data, image.info, config);
  for (let index = 0; index < data.length; index += image.info.channels) {
    if (data[index + 3] <= 2) {
      data[index] = 0;
      data[index + 1] = 0;
      data[index + 2] = 0;
      data[index + 3] = 0;
    }
  }

  await sharp(data, { raw: image.info }).png().toFile(keyedPath);
  return { theme, path: keyedPath, data, info: image.info };
}

function clusterProjection(counts, options = {}) {
  const threshold = Number(options.threshold || 1);
  const maxGap = Number(options.maxGap || 0);
  const minSpan = Number(options.minSpan || 1);
  const clusters = [];
  let start = -1;
  let last = -1;
  let gap = 0;

  counts.forEach((count, index) => {
    if (count > threshold) {
      if (start < 0) start = index;
      last = index;
      gap = 0;
      return;
    }
    if (start < 0) return;
    gap += 1;
    if (gap <= maxGap) return;
    if (last - start + 1 >= minSpan) clusters.push({ start, end: last });
    start = -1;
    last = -1;
    gap = 0;
  });

  if (start >= 0 && last - start + 1 >= minSpan) clusters.push({ start, end: last });
  return clusters;
}

function tightenBounds(source, xRange, yRange, pad = 3) {
  const { width, height, channels } = source.info;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  const x0 = clamp(xRange.start, 0, width - 1);
  const x1 = clamp(xRange.end, 0, width - 1);
  const y0 = clamp(yRange.start, 0, height - 1);
  const y1 = clamp(yRange.end, 0, height - 1);

  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const offset = (y * width + x) * channels;
      if (source.data[offset + 3] <= 18) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return {
    x: clamp(minX - pad, 0, width - 1),
    y: clamp(minY - pad, 0, height - 1),
    w: clamp(maxX - minX + 1 + pad * 2, 1, width - minX + pad),
    h: clamp(maxY - minY + 1 + pad * 2, 1, height - minY + pad)
  };
}

function detectSourceRows(source) {
  const { width, height, channels } = source.info;
  const yCounts = new Array(height).fill(0);
  for (let y = 0; y < height; y += 1) {
    let count = 0;
    for (let x = 0; x < width; x += 1) {
      if (source.data[(y * width + x) * channels + 3] > 18) count += 1;
    }
    yCounts[y] = count;
  }

  const rows = clusterProjection(yCounts, {
    threshold: Math.max(24, Math.floor(width * 0.015)),
    maxGap: 24,
    minSpan: 42
  }).slice(0, ROWS);

  if (rows.length < ROWS) {
    throw new Error(`${source.theme} terrain source should expose four detectable sprite rows`);
  }

  return rows.map((row) => {
    const xCounts = new Array(width).fill(0);
    for (let x = 0; x < width; x += 1) {
      let count = 0;
      for (let y = row.start; y <= row.end; y += 1) {
        if (source.data[(y * width + x) * channels + 3] > 18) count += 1;
      }
      xCounts[x] = count;
    }
    const xClusters = clusterProjection(xCounts, {
      threshold: 6,
      maxGap: 20,
      minSpan: 24
    });
    return xClusters.map((cluster) => tightenBounds(source, cluster, row, 4)).filter(Boolean);
  });
}

function getDetectedBox(rows, row, column) {
  const rowBoxes = rows[row] || [];
  if (!rowBoxes.length) return null;
  return rowBoxes[clamp(column, 0, rowBoxes.length - 1)];
}

function getCellSourceBox(rows, cell, config) {
  const row = Math.floor(cell / COLUMNS);
  const column = cell % COLUMNS;
  if (cell === 31) return null;
  if (cell >= 24 && cell <= 26) return getDetectedBox(rows, 3, column - 24);
  if (cell >= 27 && cell <= 30) {
    const longColumns = config.row4LongUndersideColumns || [4, 5];
    return getDetectedBox(rows, 3, longColumns[(cell - 27) % longColumns.length]);
  }
  return getDetectedBox(rows, row, column);
}

function insetBox(box, options = {}) {
  if (!box) return null;
  const insetX = Number(options.insetX || 0);
  const insetY = Number(options.insetY || 0);
  const x = Math.round(box.x + box.w * insetX);
  const y = Math.round(box.y + box.h * insetY);
  const w = Math.max(1, Math.round(box.w * (1 - insetX * 2)));
  const h = Math.max(1, Math.round(box.h * (1 - insetY * 2)));
  return { left: x, top: y, width: w, height: h };
}

function scrubCellMagenta(raw) {
  const data = Buffer.from(raw);
  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const a = data[index + 3];
    if (isMagentaKey(r, g, b, a)) {
      data[index] = 0;
      data[index + 1] = 0;
      data[index + 2] = 0;
      data[index + 3] = 0;
    }
  }
  return data;
}

function makeHorizontalSeamless(raw) {
  const data = Buffer.from(raw);
  const blend = 8;
  for (let y = 0; y < CELL; y += 1) {
    for (let x = 0; x < blend; x += 1) {
      const leftX = x;
      const rightX = CELL - 1 - x;
      const weight = (blend - x) / blend;
      for (let channel = 0; channel < 4; channel += 1) {
        const left = (y * CELL + leftX) * 4 + channel;
        const right = (y * CELL + rightX) * 4 + channel;
        const average = Math.round((data[left] + data[right]) / 2);
        data[left] = Math.round(data[left] * (1 - weight) + average * weight);
        data[right] = Math.round(data[right] * (1 - weight) + average * weight);
      }
    }
  }
  return data;
}

async function renderInsideCell(keyedPath, crop, options = {}) {
  const margin = Number(options.margin || 5);
  const resized = await sharp(keyedPath)
    .extract(crop)
    .resize({
      width: CELL - margin * 2,
      height: CELL - margin * 2,
      fit: 'inside',
      kernel: 'nearest',
      withoutEnlargement: false
    })
    .png()
    .toBuffer({ resolveWithObject: true });
  const left = Math.floor((CELL - resized.info.width) / 2);
  const top = options.anchor === 'bottom'
    ? CELL - resized.info.height - margin
    : Math.floor((CELL - resized.info.height) / 2);
  return sharp({
    create: {
      width: CELL,
      height: CELL,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    }
  })
    .composite([{ input: resized.data, left, top: clamp(top, 0, CELL - resized.info.height) }])
    .ensureAlpha()
    .raw()
    .toBuffer();
}

async function renderCoverCell(keyedPath, crop, options = {}) {
  return sharp(keyedPath)
    .extract(crop)
    .resize({
      width: CELL,
      height: CELL,
      fit: 'cover',
      position: options.position || 'centre',
      kernel: 'nearest'
    })
    .ensureAlpha()
    .raw()
    .toBuffer();
}

async function renderNaturalCapCell(keyedPath, crop, cell) {
  return sharp(keyedPath)
    .extract(crop)
    .resize({
      width: CELL,
      height: CELL,
      fit: 'cover',
      position: 'centre',
      kernel: 'nearest',
      background: { r: 0, g: 0, b: 0, alpha: 0 }
    })
    .ensureAlpha()
    .raw()
    .toBuffer();
}

function softenCapOuterEdge(raw, cell) {
  const data = Buffer.from(raw);
  const fadeWidth = 10;
  for (let y = 0; y < CELL; y += 1) {
    for (let x = 0; x < fadeWidth; x += 1) {
      const edgeX = LEFT_CAP_CELLS.has(cell) ? x : CELL - 1 - x;
      const alphaOffset = (y * CELL + edgeX) * 4 + 3;
      const t = x / fadeWidth;
      const eased = t * t * (3 - 2 * t);
      data[alphaOffset] = Math.round(data[alphaOffset] * eased);
    }
  }
  return data;
}

function makeShadowCell(config) {
  const data = Buffer.alloc(CELL * CELL * 4);
  const color = config.shadow || { r: 32, g: 28, b: 24, a: 96 };
  for (let y = 0; y < CELL; y += 1) {
    const distance = Math.abs(y - 38);
    const fade = clamp(1 - distance / 18, 0, 1);
    for (let x = 0; x < CELL; x += 1) {
      const wave = 0.88 + Math.sin((x / CELL) * Math.PI * 2) * 0.08;
      const alpha = Math.round(color.a * fade * wave);
      const offset = (y * CELL + x) * 4;
      data[offset] = color.r;
      data[offset + 1] = color.g;
      data[offset + 2] = color.b;
      data[offset + 3] = alpha;
    }
  }
  return makeHorizontalSeamless(data);
}

function waveY(base, seed, x, amplitude = 3) {
  return base + Math.round(Math.sin((x + seed * 17) * 0.29) * amplitude + Math.sin((x + seed * 9) * 0.11) * (amplitude * 0.55));
}

function topGrassPath(baseY, seed) {
  const points = [];
  for (let x = 0; x <= CELL; x += 8) {
    points.push(`${x} ${clamp(waveY(baseY, seed, x, 2), 0, 18)}`);
  }
  return `M0 ${baseY + 12} L ${points.join(' L ')} L64 ${baseY + 12} Z`;
}

function bottomEdgePoints(baseY, seed, fullDepth) {
  const points = [];
  for (let x = CELL; x >= 0; x -= 8) {
    const y = fullDepth ? CELL : clamp(waveY(baseY, seed, x, 5), 42, 63);
    points.push(`${x} ${y}`);
  }
  return points.join(' L ');
}

function soilDecorations(palette, seed, options = {}) {
  const rows = [];
  const yBase = Number(options.y || 28);
  const count = Number(options.count || 5);
  for (let index = 0; index < count; index += 1) {
    const x = 10 + ((seed * 19 + index * 17) % 44);
    const y = yBase + ((seed * 13 + index * 11) % 26);
    const w = 4 + ((seed + index * 3) % 7);
    const h = 3 + ((seed + index * 5) % 6);
    const fill = index % 3 === 0 ? palette.stone : (index % 2 ? palette.soilMid : palette.soil);
    rows.push(`<ellipse cx="${x}" cy="${y}" rx="${w}" ry="${h}" fill="${fill}" opacity="${index % 3 === 0 ? '0.42' : '0.28'}"/>`);
  }
  for (let index = 0; index < 3; index += 1) {
    const x = 12 + ((seed * 23 + index * 15) % 40);
    const y = 22 + ((seed * 7 + index * 13) % 28);
    const x2 = clamp(x + (index % 2 ? 10 : -8), 4, 60);
    const y2 = clamp(y + 16 + ((seed + index) % 8), 20, 62);
    rows.push(`<path d="M${x} ${y} C${x + 4} ${y + 8} ${x2 - 4} ${y2 - 8} ${x2} ${y2}" fill="none" stroke="${palette.root}" stroke-width="2" stroke-linecap="round" opacity="0.65"/>`);
  }
  return rows.join('');
}

function vineDecorations(palette, seed, y, length) {
  const rows = [];
  for (let index = 0; index < 4; index += 1) {
    const x = 10 + ((seed * 11 + index * 14) % 44);
    const h = length - ((seed + index * 5) % 16);
    rows.push(`<path d="M${x} ${y} C${x - 4} ${y + 8} ${x + 5} ${y + 18} ${x} ${clamp(y + h, y + 8, 63)}" fill="none" stroke="${palette.moss}" stroke-width="2" stroke-linecap="round" opacity="0.78"/>`);
  }
  return rows.join('');
}

function surfaceCellSvg(cell, config) {
  const palette = config.palette;
  const column = cell % COLUMNS;
  const platform = cell >= 4;
  const fullDepth = !platform;
  const seed = cell + (config.output === 'thornpath-thicket.png' ? 19 : 3);
  const leftCap = column === 0 || column === 4;
  const rightCap = column === 3 || column === 7;
  if (platform) {
    const bodyLeft = leftCap ? 8 : 0;
    const bodyRight = rightCap ? 56 : 64;
    const bodyPath = [
      `M${bodyLeft} 12`,
      `H${bodyRight}`,
      rightCap ? 'C62 15 64 22 64 30 C64 40 57 50 46 51' : 'H64 V50',
      leftCap ? 'H18 C7 50 0 40 0 30 C0 22 2 15 8 12' : 'H0 V12',
      'Z'
    ].join(' ');
    const highlightPath = [
      `M${bodyLeft} 13`,
      `H${bodyRight}`,
      rightCap ? 'C60 16 61 22 61 29 C61 36 56 43 47 45' : 'H64 V35',
      leftCap ? 'H17 C8 43 3 36 3 29 C3 22 4 16 8 13' : 'H0 V13',
      'Z'
    ].join(' ');
    const grassLeft = leftCap ? 3 : 0;
    const grassRight = rightCap ? 61 : 64;
    const pebbleA = 15 + (seed % 12);
    const pebbleB = 39 + (seed % 10);
    return `
      <path d="${bodyPath}" fill="${palette.soilMid}"/>
      <path d="${highlightPath}" fill="${palette.soil}" opacity="0.72"/>
      <path d="M${grassLeft} 9 H${grassRight} V19 H${grassLeft} Z" fill="${palette.grassDark}"/>
      <path d="${topGrassPath(3, seed)}" fill="${palette.grass}"/>
      <path d="${topGrassPath(0, seed + 7)}" fill="${palette.grassLight}" opacity="0.76"/>
      <path d="M4 45 C18 50 44 50 60 45 V51 H4 Z" fill="${palette.soilDark}" opacity="0.3"/>
      <rect x="0" y="18" width="64" height="3" fill="${palette.moss}" opacity="0.48"/>
      <ellipse cx="${pebbleA}" cy="32" rx="5" ry="3" fill="${palette.stone}" opacity="0.25"/>
      <ellipse cx="${pebbleB}" cy="37" rx="7" ry="4" fill="${palette.soilDark}" opacity="0.18"/>
      <path d="M22 24 C30 29 38 29 46 24" fill="none" stroke="${palette.root}" stroke-width="2" stroke-linecap="round" opacity="0.22"/>
    `;
  }
  const leftX = leftCap ? 5 : 0;
  const rightX = rightCap ? 59 : 64;
  const bottom = bottomEdgePoints(platform ? 55 : 64, seed, fullDepth);
  const capStart = leftCap ? `M${leftX} 14 Q0 24 0 42 L0 ${platform ? 55 : 64} L` : `M0 14 L`;
  const capEnd = rightCap ? ` L64 ${platform ? 55 : 64} L64 42 Q64 24 ${rightX} 14 Z` : ' Z';
  const soilPath = `${capStart}${bottom} L${rightX} 14${capEnd}`;
  const topLeft = leftCap ? 2 : 0;
  const topRight = rightCap ? 62 : 64;
  return `
    <path d="${soilPath}" fill="${palette.soilMid}"/>
    <path d="M${topLeft} 18 H${topRight} V${platform ? 34 : 64} H${topLeft} Z" fill="${palette.soil}" opacity="0.62"/>
    ${soilDecorations(palette, seed, { y: 24, count: platform ? 4 : 6 })}
    <path d="M${topLeft} 10 H${topRight} V22 H${topLeft} Z" fill="${palette.grassDark}"/>
    <path d="${topGrassPath(4, seed)}" fill="${palette.grass}"/>
    <path d="${topGrassPath(1, seed + 7)}" fill="${palette.grassLight}" opacity="0.78"/>
    ${vineDecorations(palette, seed, platform ? 37 : 48, platform ? 22 : 13)}
    <rect x="0" y="13" width="64" height="3" fill="${palette.moss}" opacity="0.62"/>
  `;
}

function bodyCellSvg(cell, config) {
  const palette = config.palette;
  const seed = cell + (config.output === 'thornpath-thicket.png' ? 31 : 5);
  const bottom = bottomEdgePoints(58, seed, false);
  return `
    <path d="M0 0 H64 L64 58 L${bottom} L0 0 Z" fill="${palette.soilDark}"/>
    <path d="M0 0 H64 L64 56 L${bottom} L0 0 Z" fill="${palette.soilMid}" opacity="0.78"/>
    <path d="M0 0 H64 V48 H0 Z" fill="${palette.soil}" opacity="${cell >= 12 ? '0.28' : '0.38'}"/>
    ${soilDecorations(palette, seed, { y: 8, count: cell >= 12 ? 5 : 4 })}
  `;
}

function undersideCellSvg(cell, config, long = false) {
  const palette = config.palette;
  const seed = cell + (config.output === 'thornpath-thicket.png' ? 43 : 11);
  const baseY = long ? 42 : 30;
  const bottom = bottomEdgePoints(baseY, seed, false);
  return `
    <path d="M0 0 H64 L64 ${baseY - 4} L${bottom} L0 0 Z" fill="${palette.soilMid}"/>
    <path d="M0 0 H64 V11 H0 Z" fill="${palette.grassDark}"/>
    <path d="${topGrassPath(0, seed)}" fill="${palette.grass}"/>
    <rect x="0" y="11" width="64" height="4" fill="${palette.moss}" opacity="0.72"/>
    ${soilDecorations(palette, seed, { y: 14, count: 3 })}
    ${vineDecorations(palette, seed, long ? 32 : 24, long ? 30 : 22)}
  `;
}

function sideCellSvg(cell, config) {
  const palette = config.palette;
  const left = cell === 20;
  const seed = cell + (config.output === 'thornpath-thicket.png' ? 53 : 17);
  const pathData = left
    ? 'M18 4 H64 V64 H10 C2 52 4 17 18 4 Z'
    : 'M0 4 H46 C60 17 62 52 54 64 H0 Z';
  const grassPath = left
    ? 'M16 0 H64 V14 H15 Q8 8 16 0 Z'
    : 'M0 0 H48 Q56 8 49 14 H0 Z';
  return `
    <path d="${pathData}" fill="${palette.soilMid}"/>
    <path d="${pathData}" fill="${palette.soil}" opacity="0.48"/>
    ${soilDecorations(palette, seed, { y: 14, count: 4 })}
    <path d="${grassPath}" fill="${palette.grass}"/>
    <path d="${grassPath}" fill="${palette.grassLight}" opacity="0.5"/>
  `;
}

function detailCellSvg(cell, config) {
  const palette = config.palette;
  const variant = cell % 4;
  if (variant === 0) {
    return `
      <ellipse cx="32" cy="49" rx="22" ry="9" fill="${palette.soilMid}"/>
      <path d="M12 48 C18 37 27 36 32 47 C38 35 49 37 54 48" fill="none" stroke="${palette.root}" stroke-width="4" stroke-linecap="round"/>
      <ellipse cx="30" cy="48" rx="18" ry="7" fill="${palette.moss}" opacity="0.7"/>
    `;
  }
  if (variant === 1) {
    return `
      <ellipse cx="30" cy="50" rx="24" ry="8" fill="${palette.moss}"/>
      <path d="M18 49 C20 35 24 28 30 47 M34 49 C37 35 42 31 48 47" fill="none" stroke="${palette.grassDark}" stroke-width="3" stroke-linecap="round"/>
      <circle cx="20" cy="36" r="3" fill="${palette.flower}"/>
      <circle cx="47" cy="38" r="3" fill="${palette.accent}"/>
    `;
  }
  if (variant === 2) {
    return `
      <ellipse cx="34" cy="49" rx="23" ry="9" fill="${palette.soilDark}" opacity="0.75"/>
      <ellipse cx="26" cy="43" rx="9" ry="7" fill="${palette.stone}"/>
      <ellipse cx="42" cy="45" rx="12" ry="8" fill="${palette.soilMid}"/>
      <path d="M15 40 C28 30 37 31 50 41" fill="none" stroke="${palette.root}" stroke-width="3" stroke-linecap="round"/>
    `;
  }
  return `
    <ellipse cx="31" cy="51" rx="25" ry="7" fill="${palette.grassDark}"/>
    <path d="M13 51 C18 37 25 34 31 50 C37 35 45 37 51 50" fill="none" stroke="${palette.moss}" stroke-width="4" stroke-linecap="round"/>
    <circle cx="30" cy="38" r="3" fill="${palette.flower}"/>
  `;
}

function cleanTerrainCellSvg(cell, config) {
  let body = '';
  if (cell < 8) body = surfaceCellSvg(cell, config);
  else if (cell < 16) body = bodyCellSvg(cell, config);
  else if (cell < 20) body = undersideCellSvg(cell, config);
  else if (cell < 22) body = sideCellSvg(cell, config);
  else if (cell < 27) body = detailCellSvg(cell, config);
  else if (cell < 31) body = undersideCellSvg(cell, config, true);
  else return null;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${CELL}" height="${CELL}" viewBox="0 0 ${CELL} ${CELL}">${body}</svg>`;
}

async function buildCleanCell(cell, config) {
  if (cell === 31) return makeShadowCell(config);
  const svg = cleanTerrainCellSvg(cell, config);
  let raw = await sharp(Buffer.from(svg)).ensureAlpha().raw().toBuffer();
  raw = scrubCellMagenta(raw);
  if (REPEAT_CELLS.has(cell)) raw = makeHorizontalSeamless(raw);
  return scrubCellMagenta(raw);
}

async function buildCell(source, rows, cell, config) {
  if (cell === 31) return makeShadowCell(config);

  const box = getCellSourceBox(rows, cell, config);
  if (!box) throw new Error(`${source.theme} source did not expose a box for terrain cell ${cell}`);

  const isRepeat = REPEAT_CELLS.has(cell);
  const isNaturalCap = NATURAL_CAP_CELLS.has(cell);
  const isDetail = cell >= 22 && cell <= 26;
  const isSide = cell === 20 || cell === 21;
  const crop = insetBox(box, {
    insetX: isRepeat ? 0.16 : 0,
    insetY: isRepeat && cell < 20 ? 0.02 : 0
  });

  let raw;
  if (isNaturalCap) {
    raw = await renderNaturalCapCell(source.path, crop, cell);
    raw = softenCapOuterEdge(raw, cell);
  } else if (isDetail || isSide) {
    raw = await renderInsideCell(source.path, crop, {
      margin: isSide ? 2 : 7,
      anchor: isSide ? 'bottom' : 'center'
    });
  } else {
    raw = await renderCoverCell(source.path, crop, {
      position: cell >= 16 && cell <= 19 ? 'north' : 'centre'
    });
  }

  raw = scrubCellMagenta(raw);
  if (isRepeat) raw = makeHorizontalSeamless(raw);
  return scrubCellMagenta(raw);
}

function copyCell(atlas, cell, raw) {
  const cellX = (cell % COLUMNS) * CELL;
  const cellY = Math.floor(cell / COLUMNS) * CELL;
  for (let y = 0; y < CELL; y += 1) {
    const sourceStart = y * CELL * 4;
    const targetStart = ((cellY + y) * ATLAS_WIDTH + cellX) * 4;
    raw.copy(atlas, targetStart, sourceStart, sourceStart + CELL * 4);
  }
}

function seamScore(raw, cell) {
  const cellX = (cell % COLUMNS) * CELL;
  const cellY = Math.floor(cell / COLUMNS) * CELL;
  let total = 0;
  let samples = 0;
  for (let y = 0; y < CELL; y += 1) {
    const left = ((cellY + y) * ATLAS_WIDTH + cellX) * 4;
    const right = ((cellY + y) * ATLAS_WIDTH + cellX + CELL - 1) * 4;
    if (Math.max(raw[left + 3], raw[right + 3]) < 8) continue;
    for (let channel = 0; channel < 4; channel += 1) {
      total += Math.abs(raw[left + channel] - raw[right + channel]);
      samples += 1;
    }
  }
  return samples ? total / samples : 0;
}

function countVisibleMagenta(raw) {
  let count = 0;
  for (let index = 0; index < raw.length; index += 4) {
    if (raw[index + 3] > 12 && isMagentaKey(raw[index], raw[index + 1], raw[index + 2], raw[index + 3])) count += 1;
  }
  return count;
}

function countCapEdgeAlpha(raw, cell) {
  const cellX = (cell % COLUMNS) * CELL;
  const cellY = Math.floor(cell / COLUMNS) * CELL;
  const edgeX = LEFT_CAP_CELLS.has(cell) ? cellX : cellX + CELL - 1;
  let count = 0;
  for (let y = 0; y < CELL; y += 1) {
    const offset = ((cellY + y) * ATLAS_WIDTH + edgeX) * 4;
    if (raw[offset + 3] > 18) count += 1;
  }
  return count;
}

function validateAtlas(theme, raw) {
  const failed = Array.from(REPEAT_CELLS)
    .map((cell) => ({ cell, score: seamScore(raw, cell) }))
    .filter((entry) => entry.score > 14);
  if (failed.length) {
    throw new Error(`${theme} forest terrain has visible seam scores: ${failed.map((entry) => `${entry.cell}:${entry.score.toFixed(1)}`).join(', ')}`);
  }
  const magenta = countVisibleMagenta(raw);
  if (magenta > 0) throw new Error(`${theme} forest terrain contains ${magenta} visible magenta pixels`);
  const hardCaps = Array.from(NATURAL_CAP_CELLS)
    .map((cell) => ({ cell, pixels: countCapEdgeAlpha(raw, cell) }))
    .filter((entry) => entry.pixels > 4);
  if (hardCaps.length) {
    throw new Error(`${theme} forest terrain has hard cap edge pixels: ${hardCaps.map((entry) => `${entry.cell}:${entry.pixels}`).join(', ')}`);
  }
}

async function generateTheme(theme, config, options = {}) {
  const source = await createKeyedSource(theme, config);
  const rows = detectSourceRows(source);
  const atlas = Buffer.alloc(ATLAS_WIDTH * ATLAS_HEIGHT * 4);
  for (let cell = 0; cell < COLUMNS * ROWS; cell += 1) {
    copyCell(atlas, cell, await buildCell(source, rows, cell, config));
  }
  validateAtlas(theme, atlas);
  if (!options.validateOnly) {
    const outputPath = path.join(TERRAIN_DIR, config.output);
    await sharp(atlas, {
      raw: { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, channels: 4 }
    }).png().toFile(outputPath);
  }
  return { theme, rowCounts: [] };
}

async function main() {
  fs.mkdirSync(SOURCE_DIR, { recursive: true });
  fs.mkdirSync(TERRAIN_DIR, { recursive: true });
  const validateOnly = process.argv.includes('--validate');
  const results = [];
  for (const [theme, config] of Object.entries(THEMES)) {
    if (validateOnly) {
      const outputPath = path.join(TERRAIN_DIR, config.output);
      if (!fs.existsSync(outputPath)) throw new Error(`Missing forest terrain atlas: ${path.relative(ROOT, outputPath)}`);
      const image = await sharp(outputPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      if (image.info.width !== ATLAS_WIDTH || image.info.height !== ATLAS_HEIGHT) {
        throw new Error(`${path.relative(ROOT, outputPath)} should be ${ATLAS_WIDTH}x${ATLAS_HEIGHT}`);
      }
      validateAtlas(theme, image.data);
      results.push({ theme, rowCounts: [] });
    } else {
      results.push(await generateTheme(theme, config));
    }
  }
  const action = validateOnly ? 'Validated' : 'Generated';
  console.log(`${action} ${results.length} forest terrain atlases: ${results.map((result) => result.theme).join(', ')}`);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exit(1);
  });
}
