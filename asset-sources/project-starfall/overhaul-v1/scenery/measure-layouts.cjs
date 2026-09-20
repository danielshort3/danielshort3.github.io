const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

async function main() {
  const reports = [];
  const ledgerPath = path.join(__dirname, 'ledger.json');
  const ledger = JSON.parse(fs.readFileSync(ledgerPath, 'utf8'));
  const seen = new Set();
  for (const record of ledger.assets.filter((r) => r.kitLayout)) {
    if (seen.has(record.sourcePath)) continue;
    seen.add(record.sourcePath);
    const source = path.resolve(__dirname, '../../../..', record.sourcePath);
    const { data, info } = await sharp(source).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    const cols = record.kitLayout.sourceColumns;
    const rows = record.kitLayout.sourceRows;
    const count = (left, top, width, height) => {
      let n = 0;
      for (let y = top; y < top + height; y++) for (let x = left; x < left + width; x++) if (data[(y * info.width + x) * 4 + 3] >= 16) n++;
      return n;
    };
    function valley(nominal, radius, max, fn) {
      let best = Math.round(nominal); let score = Infinity;
      for (let v = Math.max(1, Math.round(nominal - radius)); v < Math.min(max - 1, nominal + radius); v++) {
        const s = fn(v) * 10000 + Math.abs(v - nominal);
        if (s < score) { score = s; best = v; }
      }
      return best;
    }
    const xs = [0];
    for (let c = 1; c < cols; c++) xs.push(valley(c * info.width / cols, info.width / cols * .2, info.width, (x) => count(x - 2, 0, 5, info.height)));
    xs.push(info.width);
    const rects = [];
    const metrics = [];
    for (let c = 0; c < cols; c++) {
      const ys = [0];
      for (let row = 1; row < rows; row++) ys.push(valley(row * info.height / rows, info.height / rows * .2, info.height, (y) => count(xs[c], y - 2, xs[c + 1] - xs[c], 5)));
      ys.push(info.height);
      for (let r = 0; r < rows; r++) {
        const rect = { left: xs[c], top: ys[r], width: xs[c + 1] - xs[c], height: ys[r + 1] - ys[r] };
        let x0 = info.width, y0 = info.height, x1 = -1, y1 = -1, pixels = 0;
        for (let y = rect.top; y < rect.top + rect.height; y++) for (let x = rect.left; x < rect.left + rect.width; x++) if (data[(y * info.width + x) * 4 + 3] >= 16) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); pixels++; }
        const contacts = { left: x0 === rect.left, right: x1 === rect.left + rect.width - 1, top: y0 === rect.top, bottom: y1 === rect.top + rect.height - 1 };
        rects[r * cols + c] = rect;
        metrics[r * cols + c] = { bbox: [x0, y0, x1, y1], pixels, contacts };
      }
    }
    for (const r of ledger.assets.filter((r) => r.sourcePath === record.sourcePath)) r.sourceRects = rects;
    reports.push({ source: record.sourcePath, dimensions: [info.width, info.height], rects, metrics });
  }
  fs.writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + '\n');
  fs.writeFileSync(path.join(__dirname, 'layout-measurements.json'), JSON.stringify(reports, null, 2) + '\n');
  console.log(JSON.stringify(reports.map((r) => ({ source: path.basename(r.source), contacts: r.metrics.map((m, i) => Object.values(m.contacts).some(Boolean) ? { i, ...m.contacts } : null).filter(Boolean) })), null, 2));
}
main().catch((e) => { console.error(e); process.exit(1); });
