/* Read-only screening: translation fits flag candidates, never certify animation quality. */
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const root = path.resolve(__dirname, '../..');
const sourceRoot = path.join(root, 'asset-sources/project-starfall/overhaul-v1/enemies');
const inventory = JSON.parse(fs.readFileSync(path.join(sourceRoot, 'inventory.json')));
const size = 160;
const round = value => Math.round(value * 100) / 100;

function geometry(data) {
  let minX = 160, minY = 160, maxX = -1, maxY = -1, count = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const alpha = data[(y * size + x) * 4 + 3];
    if (alpha < 128) continue;
    minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    count++; sumX += x; sumY += y;
  }
  return { minX, minY, maxX, maxY, count, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2, centroidX: sumX / count, centroidY: sumY / count };
}

function values(data, index) {
  const alpha = data[index + 3] / 255;
  return [data[index] * alpha / 255, data[index + 1] * alpha / 255, data[index + 2] * alpha / 255, alpha];
}

function points(data, box, mode) {
  const result = [];
  const width = box.maxX - box.minX + 1, height = box.maxY - box.minY + 1;
  const left = mode === 'core' ? box.minX + width * .17 : box.minX - 5;
  const right = mode === 'core' ? box.maxX - width * .17 : box.maxX + 5;
  const top = mode === 'core' ? box.minY + height * .12 : box.minY - 5;
  const bottom = mode === 'core' ? box.maxY - height * .26 : box.maxY + 5;
  for (let y = Math.max(0, Math.round(top)); y <= Math.min(159, bottom); y += 2) {
    for (let x = Math.max(0, Math.round(left)); x <= Math.min(159, right); x += 2) {
      const rgba = values(data, (y * size + x) * 4);
      result.push({ x, y, rgba });
    }
  }
  return result;
}

function error(samples, target, dx, dy) {
  let sum = 0;
  for (const point of samples) {
    const x = point.x + dx, y = point.y + dy;
    const pixel = x >= 0 && y >= 0 && x < 160 && y < 160 ? (y * size + x) * 4 : -1;
    const alpha = pixel >= 0 ? target[pixel + 3] / 255 : 0;
    sum += .5 * (alpha - point.rgba[3]) ** 2;
    for (let c = 0; c < 3; c++) {
      const value = pixel >= 0 ? target[pixel + c] / 255 * alpha : 0;
      sum += (value - point.rgba[c]) ** 2;
    }
  }
  return sum / samples.length / 3.5;
}

function fit(samples, target) {
  const unshiftedError = error(samples, target, 0, 0);
  let best = { dx: 0, dy: 0, error: unshiftedError };
  for (let dy = -18; dy <= 18; dy += 2) for (let dx = -40; dx <= 40; dx += 2) {
    const score = error(samples, target, dx, dy);
    if (score < best.error) best = { dx, dy, error: score };
  }
  const coarse = { ...best };
  for (let dy = coarse.dy - 1; dy <= coarse.dy + 1; dy++) for (let dx = coarse.dx - 1; dx <= coarse.dx + 1; dx++) {
    const score = error(samples, target, dx, dy);
    if (score < best.error) best = { dx, dy, error: score };
  }
  return { ...best, error: round(best.error), unshiftedError: round(unshiftedError), improvement: round(1 - best.error / (unshiftedError || 1)) };
}

async function main() {
  const rows = [];
  for (const item of inventory.items) {
    if (item.aliasOf || item.replacement?.aliasOf) continue;
    const source = JSON.parse(fs.readFileSync(path.join(sourceRoot, item.fileId, 'source.json')));
    if (source.aliasOf) continue;
    const file = path.join(root, item.replacement.sheet);
    const frames = [];
    for (let column = 0; column < 6; column++) frames.push(await sharp(file).extract({ left: column * size, top: 0, width: size, height: size }).ensureAlpha().raw().toBuffer());
    const boxes = frames.map(geometry);
    const core = points(frames[0], boxes[0], 'core');
    const full = points(frames[0], boxes[0], 'full');
    const fits = frames.map((frame, index) => ({ frame: index, core: fit(core, frame), full: fit(full, frame) }));
    const coreX = fits.map(row => row.core.dx);
    const fullX = fits.map(row => row.full.dx);
    const span = values => Math.max(...values) - Math.min(...values);
    rows.push({ fileId: item.fileId, sheet: item.replacement.sheet, canonical: true, boundsCenterSpanX: round(span(boxes.map(box => box.centerX))), alphaCentroidSpanX: round(span(boxes.map(box => box.centroidX))), coreTranslationSpanX: span(coreX), fullTranslationSpanX: span(fullX), maxCoreFullDisagreementX: Math.max(...fits.map(row => Math.abs(row.core.dx - row.full.dx))), loopClosureCoreTranslationX: fits[5].core.dx, fits, boxes: boxes.map(box => Object.fromEntries(Object.entries(box).map(([key, value]) => [key, round(value)]))) });
  }
  rows.sort((a, b) => b.coreTranslationSpanX - a.coreTranslationSpanX);
  const report = { generatedAt: new Date().toISOString(), method: 'Six-frame idle row only. Fit premultiplied RGB and alpha under translation versus frame 0. Core excludes the outer 17% horizontal width, top 12%, and bottom 26% to reduce limbs and ground contacts; full silhouette fit is an independent cross-check. Search +/-40 px X and +/-18 px Y. All figures are runtime sheet pixels. Translation includes both intentional animation and unwanted packing drift, and changes in anatomy can bias fits. These are visual-review flags, not automatic corrections, acceptance, or proof of registration quality.', canonicalCount: rows.length, rows };
  fs.mkdirSync(__dirname, { recursive: true });
  fs.writeFileSync(path.join(__dirname, 'enemy-idle-drift-screen.json'), JSON.stringify(report, null, 2) + '\n');
  const columns = '| Enemy | Core X span | Whole silhouette X span | Bounds-center X span | Loop endpoint X | Core/full disagreement |';
  const lines = ['# Enemy idle translation screening', '', report.method, '', columns, '|---|---:|---:|---:|---:|---:|', ...rows.map(row => `| ${row.fileId} | ${row.coreTranslationSpanX} | ${row.fullTranslationSpanX} | ${row.boundsCenterSpanX} | ${row.loopClosureCoreTranslationX} | ${row.maxCoreFullDisagreementX} |`), ''];
  fs.writeFileSync(path.join(__dirname, 'enemy-idle-drift-screen.md'), lines.join('\n'));
  for (let group = 0; group < Math.ceil(rows.length / 11); group++) {
    const selected = rows.slice(group * 11, (group + 1) * 11);
    const layers = [];
    for (const [index, row] of selected.entries()) {
      const strip = await sharp(path.join(root, row.sheet)).extract({ left: 0, top: 0, width: 960, height: 160 }).png().toBuffer();
      layers.push({ input: strip, left: 240, top: index * 176 + 16 });
      const label = Buffer.from(`<svg width="240" height="176"><text x="12" y="66" font-family="Arial" font-size="17" fill="#172030">${row.fileId}</text><text x="12" y="92" font-family="Arial" font-size="14" fill="#475569">Core X span ${row.coreTranslationSpanX}px</text><text x="12" y="114" font-family="Arial" font-size="14" fill="#475569">Full X span ${row.fullTranslationSpanX}px</text></svg>`);
      layers.push({ input: label, left: 0, top: index * 176 });
    }
    const guides = Buffer.from(`<svg width="1200" height="${selected.length * 176}">${selected.map((row, i) => Array.from({length:6}, (_, j) => `<path d="M${320 + 160 * j} ${i * 176 + 16}v160" stroke="#6b8591" stroke-width="1" stroke-dasharray="3 4"/>`).join('')).join('')}</svg>`);
    layers.push({ input: guides, left: 0, top: 0 });
    await sharp({ create: { width: 1200, height: selected.length * 176, channels: 4, background: '#eef2ef' } }).composite(layers).png().toFile(path.join(__dirname, `enemy-idle-drift-contact-${group + 1}.png`));
  }
  console.log(lines.join('\n'));
}

module.exports = { geometry, points, fit };
if (require.main === module) main().catch(error => { console.error(error); process.exit(1); });
