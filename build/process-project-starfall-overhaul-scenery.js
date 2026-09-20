#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_ROOT = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/scenery');
const LEDGER_PATH = path.join(SOURCE_ROOT, 'ledger.json');
const SCENES = JSON.parse(fs.readFileSync(path.join(SOURCE_ROOT, 'scenes.json'), 'utf8'));
const KITS = JSON.parse(fs.readFileSync(path.join(SOURCE_ROOT, 'kits.json'), 'utf8'));
const hash = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const relative = (file) => path.relative(ROOT, file).replace(/\\/g, '/');
const absolute = (file) => path.resolve(ROOT, file);

function readLedger() {
  if (fs.existsSync(LEDGER_PATH)) return JSON.parse(fs.readFileSync(LEDGER_PATH, 'utf8'));
  return { schemaVersion: 1, domain: 'scenery', generationTool: 'built-in image_gen', enabledGroups: [], assets: [] };
}

function writeLedger(ledger) {
  fs.mkdirSync(SOURCE_ROOT, { recursive: true });
  fs.writeFileSync(LEDGER_PATH, `${JSON.stringify(ledger, null, 2)}\n`);
}

function isEnabled(group) {
  return readLedger().enabledGroups.includes(group);
}

function putAsset(record) {
  const ledger = readLedger();
  const index = ledger.assets.findIndex((item) => item.originalPath === record.originalPath);
  if (index >= 0) {
    const previous = { ...ledger.assets[index] };
    // Source partitions are measurements of one exact master. Reusing them
    // after accepting a new generated master can clip neighboring components.
    if (record.sourceSha256 && previous.sourceSha256 !== record.sourceSha256) {
      delete previous.sourceRects;
      delete previous.terrainContactTop;
      delete previous.outputSha256;
    }
    ledger.assets[index] = { ...previous, ...record };
  }
  else ledger.assets.push(record);
  writeLedger(ledger);
}

async function registerBackground(id, generationPath) {
  if (!Object.hasOwn(SCENES, id)) throw new Error(`Unknown scenery scene: ${id}`);
  const sourcePath = path.join(SOURCE_ROOT, 'backgrounds', `${id}.png`);
  fs.mkdirSync(path.dirname(sourcePath), { recursive: true });
  if (path.resolve(generationPath) !== sourcePath) fs.copyFileSync(generationPath, sourcePath);
  const metadata = await sharp(sourcePath).metadata();
  if (metadata.width < 1000 || metadata.height < 500) throw new Error(`Background source too small: ${id}`);
  const target = `img/project-starfall/maps/${id.endsWith('-trial') ? 'trials/' : ''}${id}.webp`;
  putAsset({ id, group: 'backgrounds', originalPath: target, outputPath: target, sourcePath: relative(sourcePath), promptPath: `asset-sources/project-starfall/overhaul-v1/scenery/prompts/${id}.txt`, generatedSourcePath: generationPath, sourceSha256: hash(sourcePath), sourceDimensions: [metadata.width, metadata.height], outputDimensions: [1280, 640], treatment: 'uniform resize to runtime panorama; no recoloring or synthetic scene variants', status: 'generated' });
}

async function buildAsset(record) {
  const source = absolute(record.sourcePath);
  if (!fs.existsSync(source) || hash(source) !== record.sourceSha256) throw new Error(`Missing or changed approved scenery source: ${record.id}`);
  const output = absolute(record.outputPath);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  let pipeline = sharp(source);
  if (record.kitLayout) {
    await buildKitAtlas(record, output);
    finishAsset(record, output);
    return;
  }
  if (record.sourceRect) pipeline = pipeline.extract(record.sourceRect);
  if (record.alphaPacking) {
    const margin = record.alphaPacking.margin || 8;
    pipeline = pipeline.trim({ threshold: 8 }).resize(record.outputDimensions[0] - margin * 2, record.outputDimensions[1] - margin * 2, { fit: 'contain', position: 'bottom', background: { r: 0, g: 0, b: 0, alpha: 0 } }).extend({ top: margin, bottom: margin, left: margin, right: margin, background: { r: 0, g: 0, b: 0, alpha: 0 } });
  }
  else if (record.outputDimensions) pipeline = pipeline.resize(record.outputDimensions[0], record.outputDimensions[1], { fit: 'fill', kernel: sharp.kernel.lanczos3 });
  if (output.endsWith('.webp')) await pipeline.webp({ quality: 91, effort: 4 }).toFile(output);
  else await pipeline.png().toFile(output);
  finishAsset(record, output);
}

function finishAsset(record, output) {
  record.outputSha256 = hash(output);
  record.status = 'built';
  if (record.backupOutputPath) {
    const backup = absolute(record.backupOutputPath);
    fs.mkdirSync(path.dirname(backup), { recursive: true });
    fs.copyFileSync(output, backup);
    record.backupSha256 = hash(backup);
  }
}

async function registerStatic(id, group, generationPath, options) {
  const sourcePath = path.join(SOURCE_ROOT, group, `${id}.png`);
  fs.mkdirSync(path.dirname(sourcePath), { recursive: true });
  if (path.resolve(generationPath) !== sourcePath) fs.copyFileSync(generationPath, sourcePath);
  const metadata = await sharp(sourcePath).metadata();
  if (group !== 'world-map' && !metadata.hasAlpha) throw new Error(`Transparent scenery source required: ${id}`);
  putAsset({ id, group, originalPath: options.outputPath, sourcePath: relative(sourcePath), promptPath: `asset-sources/project-starfall/overhaul-v1/scenery/prompts/${id}.txt`, generatedSourcePath: generationPath, sourceSha256: hash(sourcePath), sourceDimensions: [metadata.width, metadata.height], treatment: 'new built-in image generation; deterministic runtime packing with preserved source alpha', status: 'generated', ...options });
}

async function registerKit(id, group, generationPath) {
  if (!KITS[id] || !['terrain', 'props', 'ramps'].includes(group) || (KITS[id].groups && !KITS[id].groups.includes(group))) throw new Error('Unknown modular scenery kit or unsupported group.');
  const sourcePath = path.join(SOURCE_ROOT, 'modular', `${id}-${group}.png`);
  if (path.resolve(generationPath) !== sourcePath) fs.copyFileSync(generationPath, sourcePath);
  const metadata = await sharp(sourcePath).metadata();
  if (!metadata.hasAlpha) throw new Error(`Kit source must have genuine transparency: ${id}-${group}`);
  const grids = { terrain: [4, 2, 8, 4, 64], props: [6, 2, 6, 2, 64], ramps: [2, 2, 4, 1, 128] };
  const grid = grids[group];
  for (const theme of KITS[id].themes) {
    const outputPath = `img/project-starfall/environment/${group}/${theme}.png`;
    putAsset({ id: `${theme}-${group}`, kit: id, group, originalPath: outputPath, outputPath, sourcePath: relative(sourcePath), promptPath: `asset-sources/project-starfall/overhaul-v1/scenery/prompts/${id}-${group}.txt`, generatedSourcePath: generationPath, sourceSha256: hash(sourcePath), sourceDimensions: [metadata.width, metadata.height], outputDimensions: [grid[2] * grid[4], grid[3] * grid[4]], kitLayout: { sourceColumns: grid[0], sourceRows: grid[1], columns: grid[2], rows: grid[3], cell: grid[4] }, treatment: 'authored biome-kit components, deterministic alpha-aware crop and runtime-cell packing; no hue variants', status: 'generated' });
  }
}

async function buildKitAtlas(record, output) {
  const source = absolute(record.sourcePath);
  const { width, height } = await sharp(source).metadata();
  const { sourceColumns, sourceRows, columns, rows, cell } = record.kitLayout;
  const terrainMap = [0, 1, 2, 3, 0, 1, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 0, 3, 1, 7, 7, 7, 7, 6, 6, 6, 6, 5];
  const sourceCells = [];
  for (let index = 0; index < sourceColumns * sourceRows; index += 1) {
    const left = Math.round(index % sourceColumns * width / sourceColumns);
    const top = Math.round(Math.floor(index / sourceColumns) * height / sourceRows);
    const rect = record.sourceRects?.[index] || { left, top, width: Math.round((index % sourceColumns + 1) * width / sourceColumns) - left, height: Math.round((Math.floor(index / sourceColumns) + 1) * height / sourceRows) - top };
    const raw = await sharp(source).extract(rect).ensureAlpha().raw().toBuffer();
    let minX = rect.width; let minY = rect.height; let maxX = -1; let maxY = -1; let visible = 0;
    for (let y = 0; y < rect.height; y += 1) for (let x = 0; x < rect.width; x += 1) {
      if (raw[(y * rect.width + x) * 4 + 3] >= 16) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); visible += 1; }
    }
    if (visible < 20) throw new Error(`Empty modular source cell ${record.kit} ${record.group} ${index}`);
    const crop = { left: rect.left + minX, top: rect.top + minY, width: maxX - minX + 1, height: maxY - minY + 1 };
    if (record.group === 'terrain' && index < sourceColumns && Number.isInteger(record.terrainContactTop)) {
      const contactTop = record.terrainContactTop;
      if (contactTop < crop.top || contactTop >= crop.top + crop.height) throw new Error(`Terrain contact crop is outside source cell ${record.id} ${index}`);
      crop.height -= contactTop - crop.top;
      crop.top = contactTop;
    }
    const transparent = { r: 0, g: 0, b: 0, alpha: 0 };
    const margin = record.group === 'props' ? 4 : record.group === 'structures' ? 8 : record.group === 'ramps' ? 2 : 0;
    sourceCells.push(await sharp(source).extract(crop).resize(cell - margin * 2, cell - margin * 2, { fit: record.group === 'terrain' ? 'fill' : 'contain', position: 'bottom', background: transparent }).extend({ top: margin, bottom: margin, left: margin, right: margin, background: transparent }).png().toBuffer());
  }
  if (record.group === 'terrain') {
    // Match repeatable source edges before packing. This is a static tile export
    // contract; character frames never pass through this processor.
    for (const indices of [[1, 2], [4], [5], [6]]) {
      const rawCells = await Promise.all(indices.map((i) => sharp(sourceCells[i]).ensureAlpha().raw().toBuffer()));
      for (let y = 0; y < cell; y += 1) {
        const seam = [0, 0, 0, 0];
        for (const raw of rawCells) for (const x of [0, cell - 1]) {
          const p = (y * cell + x) * 4;
          const alpha = raw[p + 3] / 255;
          for (let c = 0; c < 3; c += 1) seam[c] += raw[p + c] * alpha;
          seam[3] += raw[p + 3];
        }
        for (let c = 0; c < 4; c += 1) seam[c] /= rawCells.length * 2;
        for (const raw of rawCells) for (let offset = 0; offset < 8; offset += 1) {
          const weight = (8 - offset) / 8;
          for (const x of [offset, cell - 1 - offset]) {
            const p = (y * cell + x) * 4;
            const oldAlpha = raw[p + 3] / 255;
            const alpha = raw[p + 3] * (1 - weight) + seam[3] * weight;
            for (let c = 0; c < 3; c += 1) raw[p + c] = alpha > 0 ? Math.round((raw[p + c] * oldAlpha * (1 - weight) + seam[c] * weight) / (alpha / 255)) : 0;
            raw[p + 3] = Math.round(alpha);
          }
        }
      }
      for (let i = 0; i < indices.length; i += 1) sourceCells[indices[i]] = await sharp(rawCells[i], { raw: { width: cell, height: cell, channels: 4 } }).png().toBuffer();
    }
    record.treatment = 'authored biome-kit components; measured alpha-aware crop; static repeat tiles use an eight-pixel premultiplied-alpha shared-edge transition; no hue variants';
    if (Number.isInteger(record.terrainContactTop)) record.treatment += '; top-row components share the measured opaque contact row before runtime packing';
  }
  const composite = Array.from({ length: columns * rows }, (_, index) => ({ input: sourceCells[record.group === 'terrain' ? terrainMap[index] : index], left: index % columns * cell, top: Math.floor(index / columns) * cell }));
  await sharp({ create: { width: columns * cell, height: rows * cell, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } }).composite(composite).png().toFile(output);
}

async function processGroup(group, options = {}) {
  const ledger = readLedger();
  const records = ledger.assets.filter((record) => (!group || record.group === group) && (!options.only || record.id === options.only));
  for (const record of records) {
    if (options.validate) {
      const output = absolute(record.outputPath);
      if (!fs.existsSync(output) || hash(output) !== record.outputSha256) throw new Error(`Scenery output differs from approved source build: ${record.outputPath}`);
      const metadata = await sharp(output).metadata();
      if (record.outputDimensions && (metadata.width !== record.outputDimensions[0] || metadata.height !== record.outputDimensions[1])) throw new Error(`Scenery dimensions mismatch: ${record.outputPath}`);
      if (['terrain', 'props', 'ramps', 'structures', 'stations'].includes(record.group) && !metadata.hasAlpha) throw new Error(`Missing source transparency: ${record.outputPath}`);
      if (record.backupOutputPath && (!fs.existsSync(absolute(record.backupOutputPath)) || hash(absolute(record.backupOutputPath)) !== record.outputSha256 || record.backupSha256 !== record.outputSha256)) throw new Error(`Scenery fallback differs from its accepted source build: ${record.backupOutputPath}`);
    } else await buildAsset(record);
  }
  if (!options.validate) writeLedger(ledger);
  console.log(`[Starfall overhaul scenery] ${options.validate ? 'Validated' : 'Built'} ${records.length} ${group || 'scenery'} assets.`);
  return records;
}

async function main() {
  const args = process.argv.slice(2);
  const value = (flag) => { const index = args.indexOf(flag); return index >= 0 ? args[index + 1] : undefined; };
  if (value('--register-background')) {
    await registerBackground(value('--register-background'), value('--source'));
    return;
  }
  if (value('--register-kit')) {
    await registerKit(value('--register-kit'), value('--group'), value('--source'));
    return;
  }
  await processGroup(value('--group'), { only: value('--only'), validate: args.includes('--validate') });
}

if (require.main === module) main().catch((error) => { console.error(error); process.exit(1); });
module.exports = { readLedger, writeLedger, putAsset, isEnabled, processGroup, registerBackground, registerKit, registerStatic };
