const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '../../../..');
const hash = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

async function main() {
  const ledgerPath = path.join(__dirname, 'ledger.json');
  const ledger = JSON.parse(fs.readFileSync(ledgerPath, 'utf8'));
  const baseline = new Map(JSON.parse(fs.readFileSync(path.join(__dirname, '../baseline.json'), 'utf8')).assets.map((r) => [r.path, r.sha256]));
  const layouts = JSON.parse(fs.readFileSync(path.join(__dirname, 'layout-measurements.json'), 'utf8'));
  const Data = require(path.join(ROOT, 'js/games/project-starfall/project-starfall-data.js'));
  const activeStructures = new Set(Object.values(Data.ENVIRONMENT_STRUCTURE_ASSETS || {}).map((asset) => asset.path));
  const activeTerrain = new Set(Object.values(Data.ENVIRONMENT_ASSETS.terrain).map((asset) => asset.path));
  const activeInteriors = new Set(Data.MAPS.filter((map) => map.shopInterior).map((map) => map.asset));
  const failures = [];
  const outputs = [];
  const counts = {};
  for (const r of ledger.assets) {
    counts[r.group] = (counts[r.group] || 0) + 1;
    r.originalSha256 = baseline.get(r.originalPath) || null;
    r.promptSha256 = hash(path.join(ROOT, r.promptPath));
    if (r.referencePath) r.referenceSha256 = hash(path.join(ROOT, r.referencePath));
    const p = path.join(ROOT, r.outputPath);
    const { data, info } = await sharp(p).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    if (hash(p) !== r.outputSha256) failures.push(`${r.id}: output hash mismatch`);
    if (r.backupOutputPath && (!fs.existsSync(path.join(ROOT, r.backupOutputPath)) || hash(path.join(ROOT, r.backupOutputPath)) !== r.outputSha256 || r.backupSha256 !== r.outputSha256)) failures.push(`${r.id}: accepted-art fallback missing or changed`);
    if (hash(path.join(ROOT, r.sourcePath)) !== r.sourceSha256) failures.push(`${r.id}: source hash mismatch`);
    const registeredAddition = r.additionReason && ((r.group === 'structures' && activeStructures.has(r.outputPath)) || (r.group === 'backgrounds' && activeInteriors.has(r.outputPath)) || (r.group === 'terrain' && activeTerrain.has(r.outputPath)));
    if (!r.originalSha256 && !registeredAddition) failures.push(`${r.id}: absent from active baseline without a registered addition`);
    if (r.outputSha256 === r.originalSha256) failures.push(`${r.id}: unchanged baseline output`);
    const cell = r.kitLayout?.cell || Math.min(info.width, info.height);
    const cellCount = r.kitLayout ? r.kitLayout.columns * r.kitLayout.rows : 1;
    const cellMetrics = [];
    if (r.group === 'terrain') {
      for (const index of [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 27, 28, 29, 30, 31]) {
        let total = 0; let samples = 0;
        for (let y = 0; y < cell; y++) {
          const left = ((Math.floor(index / 8) * cell + y) * info.width + index % 8 * cell) * 4;
          const right = left + (cell - 1) * 4;
          if (Math.max(data[left + 3], data[right + 3]) < 8) continue;
          for (let c = 0; c < 4; c++) { total += Math.abs(data[left + c] - data[right + c]); samples++; }
        }
        const horizontalSeamScore = samples ? total / samples : 0;
        cellMetrics.push({ frame: index, horizontalSeamScore });
        if (horizontalSeamScore > 14) failures.push(`${r.id}: terrain cell ${index} seam score ${horizontalSeamScore}`);
      }
    }
    if (['props', 'ramps', 'structures', 'stations'].includes(r.group)) {
      for (let i = 0; i < cellCount; i++) {
        const left = r.kitLayout ? i % r.kitLayout.columns * cell : 0;
        const top = r.kitLayout ? Math.floor(i / r.kitLayout.columns) * cell : 0;
        let visible = 0, edgePixels = 0, minX = cell, minY = cell, maxX = -1, maxY = -1;
        for (let y = 0; y < cell; y++) for (let x = 0; x < cell; x++) if (data[((top + y) * info.width + left + x) * 4 + 3] >= 16) {
          visible++; minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
          if (x === 0 || y === 0 || x === cell - 1 || y === cell - 1) edgePixels++;
        }
        cellMetrics.push({ frame: i, visible, edgePixels, bbox: [minX, minY, maxX, maxY] });
        if (visible < 30 || edgePixels > 0) failures.push(`${r.id}: cell ${i} empty or touches border`);
      }
    }
    outputs.push({ id: r.id, dimensions: [info.width, info.height], changed: r.outputSha256 !== r.originalSha256, cellMetrics });
  }
  for (const r of layouts) for (const [i, m] of r.metrics.entries()) if (Object.values(m.contacts).some(Boolean)) failures.push(`${r.source}: source cell ${i} touches extraction border`);
  const expected = { backgrounds: 40 + activeInteriors.size, terrain: activeTerrain.size, props: 40, ramps: 40, structures: activeStructures.size, stations: 5, 'world-map': 1 };
  for (const [group, n] of Object.entries(expected)) if (counts[group] !== n) failures.push(`${group}: expected ${n}, found ${counts[group]}`);
  fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
  const report = { counts, total: ledger.assets.length, uniqueSourceMasters: new Set(ledger.assets.map((r) => r.sourcePath)).size, failures, outputs };
  fs.writeFileSync(path.join(__dirname, 'validation.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ counts, total: report.total, uniqueSourceMasters: report.uniqueSourceMasters, failures }, null, 2));
  if (failures.length) process.exitCode = 1;
}
main().catch((e) => { console.error(e); process.exit(1); });
