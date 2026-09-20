#!/usr/bin/env node
'use strict';

// Import authored actor cells with one scale per identity. This deliberately
// avoids the legacy processors' per-pose resizing and manufactured variants.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const { readActors, packCell } = require('./lib/starfall-overhaul-images.js');

const ROOT = path.resolve(__dirname, '..');
const SOURCES = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies');
const ROWS = ['idle', 'move', 'telegraph', 'attack', 'projectile', 'buff', 'hit', 'defeat'];
const CLEAR = { r: 0, g: 0, b: 0, alpha: 0 };
const ALPHA = 32;

function hash(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function bounds(data, width, rect) {
  let minX = rect.width;
  let minY = rect.height;
  let maxX = -1;
  let maxY = -1;
  let pixels = 0;
  let edgePixels = 0;
  for (let y = 0; y < rect.height; y += 1) {
    for (let x = 0; x < rect.width; x += 1) {
      if (data[((y + rect.top) * width + x + rect.left) * 4 + 3] <= ALPHA) continue;
      pixels += 1;
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      if (!x || !y || x === rect.width - 1 || y === rect.height - 1) edgePixels += 1;
    }
  }
  return { left: minX, top: minY, right: maxX, bottom: maxY, width: maxX - minX + 1, height: maxY - minY + 1, pixels, edgePixels };
}

function detectRows(data, width, height, expectedRows = 8) {
  const bands = [];
  let start = -1;
  let end = -1;
  for (let y = 0; y < height; y += 1) {
    let count = 0;
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4 + 3] > ALPHA) count += 1;
    }
    if (count >= 10) {
      if (start < 0) start = y;
      end = y;
    } else if (start >= 0 && y - end > 10) {
      if (end - start > 16) bands.push({ start, end });
      start = -1;
    }
  }
  if (start >= 0 && end - start > 16) bands.push({ start, end });
  if (bands.length !== expectedRows) throw new Error(`Expected ${expectedRows} separated actor rows; measured ${bands.length}. Supply reviewed rowBoundaries.`);
  return [0, ...bands.slice(1).map((band, index) => Math.round((bands[index].end + band.start) / 2)), height];
}

function extractRaw(data, width, rect) {
  const result = Buffer.alloc(rect.width * rect.height * 4);
  for (let y = 0; y < rect.height; y += 1) {
    const from = ((rect.top + y) * width + rect.left) * 4;
    data.copy(result, y * rect.width * 4, from, from + rect.width * 4);
  }
  return result;
}

function clearInvisibleRgb(raw) {
  for (let i = 0; i < raw.length; i += 4) {
    if (raw[i + 3] < 8) raw.fill(0, i, i + 4);
  }
  return raw;
}

async function processEnemy(fileId, importAssets) {
  if (!/^[a-z0-9-]+$/.test(fileId)) throw new Error('Invalid enemy file ID.');
  const folder = path.join(SOURCES, fileId);
  const configPath = path.join(folder, 'source.json');
  const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const sourcePath = path.join(folder, config.source);
  const bytes = fs.readFileSync(sourcePath);
  if (hash(bytes) !== config.sourceSha256) throw new Error('Raw source checksum changed.');
  const metadata = await sharp(bytes).metadata();
  if (!metadata.hasAlpha) throw new Error('Source has no alpha channel.');
  const { data, info } = await sharp(bytes).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  let transparent = 0;
  for (let i = 3; i < data.length; i += 4) if (!data[i]) transparent += 1;
  if (transparent < info.width * info.height * 0.1) throw new Error('Source does not have meaningful transparent gutters.');
  // Count coherent actors first; unequal generated gutters must not cut a body.
  const sourceRows = config.sourceRows || 8;
  let objects;
  let rowBoundaries = config.rowBoundaries;
  if (!config.supplementalRows) {
    objects = await readActors(sourcePath, 6, sourceRows);
    rowBoundaries ||= [0, ...Array.from({ length: sourceRows - 1 }, (_, row) => {
      const previous = objects.cells.filter(cell => cell.row === row);
      const next = objects.cells.filter(cell => cell.row === row + 1);
      return Math.round((Math.max(...previous.map(cell => cell.bounds.top + cell.bounds.height)) + Math.min(...next.map(cell => cell.bounds.top))) / 2);
    }), info.height];
  } else {
    rowBoundaries ||= detectRows(data, info.width, info.height, sourceRows);
  }
  const rowSources = [];
  const supplementalCache = new Map();
  for (let row = 0; row < 8; row += 1) {
    const patch = config.supplementalRows && config.supplementalRows[ROWS[row]];
    if (patch) {
      const patchPath = patch.repoPath ? path.join(ROOT, patch.repoPath) : path.join(folder, patch.file);
      if (patch.sha256 && hash(fs.readFileSync(patchPath)) !== patch.sha256) throw new Error('Supplemental source checksum changed.');
      if (!supplementalCache.has(patchPath)) supplementalCache.set(patchPath, await readActors(patchPath, patch.columns, patch.rows));
      const context = supplementalCache.get(patchPath);
      const selected = patch.indices || [0, 1, 2, 3, 4, 5];
      if (selected.length !== 6) throw new Error('A supplemental row must select exactly six authored poses or intentional held poses.');
      rowSources.push({ context, cells: selected.map(index => context.cells[index]), patch, referenceHeight: context.cells[patch.neutralIndex || 0].bounds.height });
    } else {
      const sourceRow = config.sourceRowMap ? config.sourceRowMap[row] : row;
      if (sourceRow == null) throw new Error(`Missing authored source row for ${ROWS[row]}.`);
      if (objects) {
        rowSources.push({ context: objects, cells: objects.cells.filter(cell => cell.row === sourceRow), rowTop: 0 });
      } else {
        const cropFile = path.join(folder, 'derived', `source-row-${sourceRow}.png`);
        fs.mkdirSync(path.dirname(cropFile), { recursive: true });
        await sharp(sourcePath).extract({ left: 0, top: rowBoundaries[sourceRow], width: info.width, height: rowBoundaries[sourceRow + 1] - rowBoundaries[sourceRow] }).png().toFile(cropFile);
        const groups = config.componentGroups && config.componentGroups[ROWS[row]];
        const context = await readActors(cropFile, groups ? groups.flat().length : 6, 1);
        if (groups) {
          const parts = context.cells;
          context.cells = groups.map((indices, column) => {
            const members = indices.map(index => parts[index]);
            const left = Math.min(...members.map(cell => cell.bounds.left));
            const top = Math.min(...members.map(cell => cell.bounds.top));
            const right = Math.max(...members.map(cell => cell.bounds.left + cell.bounds.width));
            const bottom = Math.max(...members.map(cell => cell.bounds.top + cell.bounds.height));
            return { row: 0, column, bounds: { left, top, width: right - left, height: bottom - top }, componentPixels: Int32Array.from(members.flatMap(cell => Array.from(cell.componentPixels))), visible: members.reduce((sum, cell) => sum + cell.visible, 0) };
          });
        }
        rowSources.push({ context, cells: context.cells, rowTop: rowBoundaries[sourceRow] });
      }
    }
  }
  const baseIdleHeight = config.baseReferenceHeight || rowSources[0].cells[0].bounds.height;
  const frames = [];
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 6; column += 1) {
      const index = row * 6 + column;
      const sourceIndex = config.sourcePoseOverrides && config.sourcePoseOverrides[index];
      const rowSource = rowSources[typeof sourceIndex === 'object' ? sourceIndex.row : row];
      const actor = rowSource.cells[typeof sourceIndex === 'object' ? sourceIndex.column : sourceIndex == null ? column : sourceIndex];
      const rect = { left: actor.bounds.left, top: actor.bounds.top, width: actor.bounds.width, height: actor.bounds.height };
      const content = { left: 0, top: 0, right: rect.width - 1, bottom: rect.height - 1, width: rect.width, height: rect.height, pixels: actor.visible };
      if (content.pixels < 80) throw new Error(`Empty or incomplete cell ${row}/${column}.`);
      const scaleFactor = rowSource.patch ? rowSource.patch.scaleFactor || baseIdleHeight / rowSource.referenceHeight : 1;
      let overrideAnchor = null;
      if (typeof sourceIndex === 'object') {
        overrideAnchor = rowSource.patch ? rowSource.patch.anchors && rowSource.patch.anchors[sourceIndex.column] || {
          x: actor.bounds.left + actor.bounds.width / 2,
          y: actor.bounds.top + actor.bounds.height * (config.floating ? 0.5 : 1)
        } : {
          x: (sourceIndex.column + 0.5) * info.width / 6,
          y: config.floating ? median(rowSource.cells.map(cell => cell.bounds.top + cell.bounds.height / 2)) : median(rowSource.cells.map(cell => cell.bounds.top + cell.bounds.height))
        };
      }
      frames.push({ index, row, column, action: ROWS[row], sourceRect: rect, content, actor, context: rowSource.context, sourcePath: path.relative(ROOT, rowSource.context.file).replaceAll('\\', '/'), scaleFactor, overrideAnchor });
    }
  }
  const rowAnchors = ROWS.map((_, row) => {
    const rowFrames = frames.filter(frame => frame.row === row);
    return config.rowAnchors && config.rowAnchors[row] || {
      x: info.width / 12,
      y: config.floating ? median(rowFrames.map(frame => frame.sourceRect.top + frame.content.height / 2)) : median(rowFrames.map(frame => frame.sourceRect.top + frame.content.height))
    };
  });
  for (const frame of frames) {
    const overrides = config.frames && config.frames[frame.index] || {};
    const rowSource = rowSources[frame.row];
    const patchAnchor = rowSource.patch && rowSource.patch.anchors && rowSource.patch.anchors[frame.column];
    frame.anchor = overrides.anchor || patchAnchor || frame.overrideAnchor || { x: rowSource.patch ? frame.actor.bounds.left + frame.actor.bounds.width / 2 : (frame.column + 0.5) * info.width / 6, y: rowSource.patch ? (config.floating ? frame.actor.bounds.top + frame.actor.bounds.height / 2 : frame.actor.bounds.top + frame.actor.bounds.height) : rowAnchors[frame.row].y };
    frame.anchorSource = overrides.anchor ? 'explicit-pose-landmark' : patchAnchor ? 'explicit-supplement-landmark' : frame.overrideAnchor ? 'reused-pose-anchor' : rowSource.patch ? 'supplement-bounds-fallback' : 'nominal-source-grid';
  }
  // Fit every pose around the same root using ONE identity-wide scale. A wide
  // extension must not be squeezed separately from a quiet idle pose.
  const baseline = config.floating ? 80 : 150;
  const limits = frames.flatMap(frame => {
    const rect = frame.sourceRect;
    return [76 / Math.max(1, (frame.anchor.x - rect.left) * frame.scaleFactor), 76 / Math.max(1, (rect.left + rect.width - frame.anchor.x) * frame.scaleFactor), (baseline - 4) / Math.max(1, (frame.anchor.y - rect.top) * frame.scaleFactor), (156 - baseline) / Math.max(1, (rect.top + rect.height - frame.anchor.y) * frame.scaleFactor)];
  });
  const scale = config.sharedScale || Math.min(142 / Math.max(...frames.map(frame => frame.content.width * frame.scaleFactor)), 136 / Math.max(...frames.map(frame => frame.content.height * frame.scaleFactor)), ...limits) * 0.98;
  const canvas = [];
  const hashes = new Map();
  const warnings = [];
  for (const frame of frames) {
    const anchor = frame.anchor;
    const packed = await packCell(frame.context, frame.actor, { size: 160, scale: scale * frame.scaleFactor, rootX: anchor.x, groundY: anchor.y, originX: 80, baseline });
    const tile = await sharp(packed.png).ensureAlpha().raw().toBuffer();
    clearInvisibleRgb(tile);
    const outputBounds = bounds(tile, 160, { left: 0, top: 0, width: 160, height: 160 });
    if (outputBounds.edgePixels) throw new Error(`Cell ${frame.row}/${frame.column} touches the output edge.`);
    const frameHash = hash(tile);
    if (hashes.has(frameHash)) warnings.push(`Exact repeated authored drawing ${frame.index} and ${hashes.get(frameHash)}; review intentional holds.`);
    hashes.set(frameHash, frame.index);
    const input = await sharp(tile, { raw: { width: 160, height: 160, channels: 4 } }).png().toBuffer();
    const rowOrder = config.rowPoseOrder && config.rowPoseOrder[ROWS[frame.row]] || [0, 1, 2, 3, 4, 5];
    if (rowOrder.length !== 6 || new Set(rowOrder).size !== 6 || rowOrder.some(index => index < 0 || index > 5)) throw new Error('Row pose mapping must contain each source column exactly once.');
    frame.outputColumn = rowOrder.indexOf(frame.column);
    canvas.push({ input, left: frame.outputColumn * 160, top: frame.row * 160 });
    frame.anchor = anchor;
    frame.outputBounds = outputBounds;
    frame.sha256 = frameHash;
    frame.tile = input;
    delete frame.actor;
    delete frame.context;
  }
  const sheet = await sharp({ create: { width: 960, height: 1280, channels: 4, background: CLEAR } }).composite(canvas).png().toBuffer();
  const registration = { originX: 80, groundY: baseline, authoredBodyHeight: frames[0].outputBounds.height, centered: !!config.floating, frameWidth: 160, frameHeight: 160 };
  const report = { fileId, sourceSha256: hash(bytes), sourceDimensions: { width: info.width, height: info.height }, substantialActorCount: frames.length, rowBoundaries, sharedScale: scale, floating: !!config.floating, registration, output: config.import.output, dimensions: { width: 960, height: 1280 }, transparentSourcePixels: transparent, alphaCleanup: 'Component-isolated actor plus adjacent sub32alpha fringe; alpha below8 discarded. No color/chroma keys. Disconnected minor pixels excluded.', noManufacturedMotion: true, noPerFrameScaling: true, warnings, frames: frames.map(({ tile, ...frame }) => frame) };
  report.intentionalHolds = config.intentionalHolds || {};
  report.sourcePoseOverrides = config.sourcePoseOverrides || {};
  report.rowPoseOrder = config.rowPoseOrder || {};
  report.registrationReview = config.registrationReview || null;
  report.nominalGridAnchorCount = frames.filter(frame => frame.anchorSource === 'nominal-source-grid').length;
  fs.writeFileSync(path.join(folder, 'import-report.json'), `${JSON.stringify(report, null, 2)}\n`);
  fs.writeFileSync(path.join(folder, 'review-sheet.png'), sheet);
  if (importAssets) {
    if (config.visualReview.status !== 'approved-for-import') throw new Error('Review the generated atlas and imported review-sheet; mark approved-for-import before replacing assets.');
    fs.writeFileSync(path.join(ROOT, config.import.output), sheet);
    const portraitFrame = frames[config.import.portraitFrame || 0];
    await sharp(portraitFrame.tile).resize(320, 320, { kernel: 'lanczos3' }).png().toFile(path.join(ROOT, config.import.portrait));
    const inventoryPath = path.join(SOURCES, 'inventory.json');
    const inventory = JSON.parse(fs.readFileSync(inventoryPath, 'utf8'));
    const item = inventory.items.find(entry => entry.fileId === fileId);
    item.status = 'imported-expanded-pending-runtime-integration';
    item.replacement = { oldSheet: item.animations[0].sheet, sheet: config.import.output, portrait: config.import.portrait, sheetSha256: hash(sheet), sourceSha256: hash(bytes), registration, report: path.relative(ROOT, path.join(folder, 'import-report.json')).replaceAll('\\', '/') };
    for (const aliasId of config.import.aliases || []) {
      const alias = inventory.items.find(entry => entry.fileId === aliasId);
      if (!alias) throw new Error(`Unknown historical enemy alias ${aliasId}.`);
      const aliasSheet = `img/project-starfall/animations/enemies/${aliasId}-sheet.png`;
      fs.writeFileSync(path.join(ROOT, aliasSheet), sheet);
      fs.copyFileSync(path.join(ROOT, config.import.portrait), path.join(ROOT, alias.portrait));
      alias.aliasOf = fileId;
      alias.status = 'imported-expanded-pending-runtime-integration';
      alias.replacement = { ...item.replacement, oldSheet: alias.animations[0].sheet, sheet: aliasSheet, portrait: alias.portrait, aliasOf: fileId };
    }
    fs.writeFileSync(inventoryPath, `${JSON.stringify(inventory, null, 2)}\n`);
    const registrations = Object.fromEntries(inventory.items.filter(entry => entry.replacement && entry.replacement.registration).map(entry => [entry.fileId, entry.replacement.registration]));
    fs.writeFileSync(path.join(SOURCES, 'enemy-registration.json'), `${JSON.stringify(registrations, null, 2)}\n`);
  }
  console.log(JSON.stringify({ fileId, imported: importAssets, source: report.sourceDimensions, output: report.dimensions, sharedScale: scale, rowBoundaries, warnings }));
}

const args = process.argv.slice(2);
const enemyIndex = args.indexOf('--enemy');
if (enemyIndex < 0 || !args[enemyIndex + 1]) {
  console.error('Usage: node build/process-project-starfall-overhaul-enemies.js --enemy <file-id> [--import]');
  process.exitCode = 1;
} else {
  processEnemy(args[enemyIndex + 1], args.includes('--import')).catch(error => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
