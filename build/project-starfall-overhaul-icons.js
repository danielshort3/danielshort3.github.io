#!/usr/bin/env node
'use strict';

// Source-owned icon/UI import pipeline. Raw generations and session references are immutable.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const vm = require('vm');
const { createRequire } = require('module');
const sharp = require('sharp');
const ROOT = path.resolve(__dirname, '..');
const HOME = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/icons');
const CATALOG = path.join(HOME, 'catalog.json');
const LEDGER = path.join(HOME, 'ledger.json');
const Data = require('../js/games/project-starfall/data/index.js');
const refs = [
  'output/starfall-animation-samples/glowcap-spring-study.png',
  'output/starfall-animation-samples/oracle-cast-study.png',
  'output/starfall-animation-samples/review-v2/player-run-study.png'
];
const rel = p => path.relative(ROOT, p).replace(/\\/g, '/');
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const human = s => s.replace(/[_-]+/g, ' ');
const save = (p, v) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, typeof v === 'string' ? v : JSON.stringify(v, null, 2) + '\n'); };

function legacyConstants(file, expressions) {
  const full = path.join(ROOT, file);
  let code = fs.readFileSync(full, 'utf8');
  code = code.slice(0, code.lastIndexOf('main().catch'));
  const context = { require: createRequire(full), __dirname: path.dirname(full), Buffer, console, process: { argv: [] } };
  code += `\nglobalThis.result = {${expressions}};`;
  vm.runInNewContext(code, context);
  return context.result;
}

function makePrompt(batch) {
  const style = 'Project Starfall: clean illustrated fantasy game art matching the supplied appearance references, crisp dark contours, controlled shading, clear forms, restrained texture, warm adventure with credible danger. References are appearance/style only: do not reproduce their characters. Upper-left/front lighting. No strict pixel grid, no tiny detail, no text, numbers, captions, labels, watermark, contact-sheet headings, borders or UI frames.';
  if (batch.kind === 'screen') return `${style}\nCreate a polished widescreen 16:9 game illustration, opaque background. ${batch.description}\nNo baked title or buttons; the real UI supplies them. Keep the center and lower third quiet enough for overlaid controls. Crisp foreground silhouettes, softly atmospheric distant scenery. Original fantasy world, no copied IP. Output a full bleed landscape image at high resolution.`;
  if (batch.kind === 'pedestal') return `${style}\nCreate a standalone wide, low fantasy character-selection pedestal on a genuinely transparent background. Single carved blue-gray stone oval platform with gold trim and a subtle cyan inset star sigil, three-quarter front view, warm highlights, entirely inside the frame with generous transparent padding. No character, no floating particle field, no scenery, no lettering. Wide 16:5 composition. This is one isolated platform, not a UI mockup.`;
  const slots = batch.columns * batch.rows;
  const cells = batch.entries.map((entry, i) => `Row ${Math.floor(i / batch.columns) + 1}, column ${i % batch.columns + 1}: ${entry.description || human(entry.id)}.`);
  for (let i = batch.entries.length; i < slots; i += 1) cells.push(`Row ${Math.floor(i / batch.columns) + 1}, column ${i % batch.columns + 1}: completely empty transparent cell.`);
  return `${style}\nCreate ONE production ${batch.kind} atlas, exactly ${batch.columns} equally wide columns by ${batch.rows} equally high rows, regular rectangular grid covering the entire image, read left-to-right then top-to-bottom. This is one coherent atlas asset, not separate images. Each icon is centered in its assigned cell, entirely isolated, no overlap, all artwork within the central 72% of each cell, generous gutters. No visible grid lines. Genuine alpha transparency behind every object; do not paint a checkerboard or colored background. Clear silhouettes readable after export at ${batch.size}px. Item materials are physical leather/cloth/metal/crystal, not flat UI pictograms. ${batch.kind === 'skill icons' ? 'Skill effects use meaning-first colors: coral #F06A60 sharp damage bursts, mint #62D995 healing plus marks, gold #F2C45E buff chevrons, cyan #63D7E8 protective shells, violet #B88AF3 impairment symbols, blue #668FFF resource droplets; neutral pearl for pure movement. Class/element hues are secondary.' : 'No large outer glow or background aura; rarity decoration is drawn separately by the game.'}\n${cells.join('\n')}\nKeep exact ordering and the full equal-cell grid. Render at high resolution with clean alpha edges.`;
}

function prepare() {
  if (fs.existsSync(CATALOG)) throw new Error('Catalog already exists; edit it explicitly rather than resetting provenance.');
  const item = legacyConstants('build/process-project-starfall-ai-item-icons.js', 'sheets:SHEETS, external:EXTERNAL_ITEM_SHEETS');
  const skill = legacyConstants('build/process-project-starfall-skill-icons.js', 'sheets:SHEETS, mastery:MASTERY_ICONS');
  const batches = [];
  for (const sheet of item.sheets) {
    const entries = sheet.items.map(v => typeof v === 'string' ? v : v.id).map(id => ({ id, output: Data.ITEM_ASSETS[id] }));
    batches.push({ id: sheet.source.replace(/^ai-/, '').replace('.png', ''), kind: 'item icons', columns: sheet.cols, rows: Math.ceil(entries.length / sheet.cols), size: 64, entries });
  }
  for (const sheet of item.external) batches.push({ id: 'items-star-cards', kind: 'item icons', columns: 3, rows: 2, size: 64, entries: sheet.ids.map(id => ({ id, output: Data.ITEM_ASSETS[id], description: human(id) + ', a small ornate collectible card with one large inset star in its named color' })) });
  const mapped = new Set(batches.flatMap(b => b.entries.map(e => e.id)));
  const missing = Object.keys(Data.ITEM_ASSETS).filter(id => !mapped.has(id));
  // All 30 previously unmapped regional/specialization items are first-class source entries.
  for (let offset = 0; offset < missing.length; offset += 15) batches.push({ id: 'items-regional-equipment-' + (offset / 15 + 1), kind: 'item icons', columns: 5, rows: 3, size: 64, entries: missing.slice(offset, offset + 15).map(id => ({ id, output: Data.ITEM_ASSETS[id] })) });
  for (const sheet of skill.sheets) batches.push({ id: 'skills-' + sheet.source.replace('-sheet.png', ''), kind: 'skill icons', columns: sheet.columns, rows: sheet.rows, size: 256, entries: sheet.skills.map(id => ({ id, output: rel(path.join(sheet.outputDir, id + '.png')) })) });
  batches.push({ id: 'skills-mastery', kind: 'skill icons', columns: 4, rows: 3, size: 256, entries: skill.mastery.map(e => ({ id: e.file, output: rel(path.join(e.outputDir, e.file + '.png')), description: human(e.file) + ', class-specific weapon or magic emblem with three rising gold enhancement chevrons; distinguish each class silhouette' })) });
  batches.push({ id: 'monster-cards', kind: 'monster card icons', columns: 7, rows: 3, size: 64, entries: Data.CARD_DEFINITIONS.map(e => ({ id: e.id, output: Data.CARD_ASSETS[e.id], description: e.name + ', collectible talisman card whose central emblem depicts ' + human(e.id) + ', ' + e.tags.join(' and ') + ' motif; no text' })) });
  const seen = new Set();
  batches.push({ id: 'menu-icons', kind: 'menu icons', columns: 6, rows: 3, size: 64, entries: Object.entries(Data.MENU_ICON_ASSETS).filter(([id, p]) => { if (seen.has(p)) return false; seen.add(p); return true; }).map(([id, output]) => ({ id, output, description: ({character:'compact adventurer bust',equipment:'steel helmet and leather shoulder guard',partyPanel:'three adventurer silhouettes',inventory:'leather satchel',skills:'open grimoire with star spark',quests:'parchment scroll and wax seal',worldmap:'folded parchment map and compass',monsters:'friendly green monster head silhouette',shop:'wooden merchant stall and coin',upgrade:'blacksmith hammer and anvil',daily:'calendar page with a gold star, no numerals',beta:'gold star medallion with a small ribbon',cashShop:'gemstone in a gold coffer',guide:'open guidebook and compass',log:'rolled journal and quill',settings:'brass gear',keybinds:'three dark keycaps without letters',admin:'ornate steward key',logout:'open wooden door with a simple outward arrow'})[id] })) });
  const screenBriefs = {
    splashScreen: 'A sweeping vista of a welcoming frontier guild town below a luminous fallen-star crystal. Forest paths lead toward distant mysterious mountains; a tiny compact brown-haired adventurer in a cream shirt stands at the left foreground looking outward. Adventure and discovery, balanced rich color, no combat. Foreground foliage and stone are clean illustrated forms.',
    startScreen: 'A welcoming blue-and-gold frontier guild lodge overlooking a peaceful fantasy crossroads town at golden hour. Warm lanterns, crafted timber, hanging banners with simple star motifs, a distant cyan fallen-star crystal and mysterious mountain silhouettes. No people in the center; an inviting path leads into the scene. Detailed but quiet behind the central login/start controls.',
    characterSelectScreen: 'An elegant open-air frontier guild courtyard at dawn with warm stone paving, wooden arcades, blue-and-gold banners and small cyan star crystals. Leave a broad open central paved stage across the lower half for character slots that will be rendered by the game. No characters, no individual pedestals, no interface elements. Soft atmospheric mountains beyond.'
  };
  for (const [id, output] of Object.entries(Data.UI_ASSETS)) batches.push({ id: 'ui-' + id, kind: 'screen', width: 1672, height: 941, description: screenBriefs[id], entries: [{ id, output }] });
  batches.push({ id: 'ui-pedestal', kind: 'pedestal', width: 512, height: 160, entries: [{ id: 'pedestal', output: Data.CHARACTER_SLOT_PEDESTAL_ASSET }] });
  const outputs = batches.flatMap(b => b.entries.map(e => e.output));
  if (outputs.length !== 337 || new Set(outputs).size !== 337) throw new Error('Expected 337 unique raster icon/UI outputs.');
  const baseline = Object.fromEntries(outputs.map(p => [p, hash(path.join(ROOT, p))]));
  const protect = {};
  function collect(dir) { for (const entry of fs.readdirSync(dir, { withFileTypes: true })) { const p = path.join(dir, entry.name); if (entry.isDirectory()) collect(p); else if (/\.png$/i.test(entry.name)) protect[rel(p)] = hash(p); } }
  collect(path.join(ROOT, 'output/starfall-animation-samples'));
  for (const batch of batches) { batch.promptPath = `asset-sources/project-starfall/overhaul-v1/icons/prompts/${batch.id}.md`; save(path.join(ROOT, batch.promptPath), makePrompt(batch) + '\n'); }
  save(CATALOG, { version: 1, owner: 'build/project-starfall-overhaul-icons.js', references: refs, batches });
  save(LEDGER, { version: 1, expectedRasterOutputs: 337, expectedEquipmentOutputs: 85, baseline, protectedSessionImages: protect, imports: {}, outputs: {} });
  console.log(JSON.stringify({ batches: batches.length, rasterOutputs: outputs.length, protectedSessionImages: Object.keys(protect).length, catalog: rel(CATALOG) }));
}

async function importBatch(id, sourcePath) {
  const catalog = JSON.parse(fs.readFileSync(CATALOG));
  const ledger = JSON.parse(fs.readFileSync(LEDGER));
  const batch = catalog.batches.find(b => b.id === id);
  if (!batch) throw new Error('Unknown batch: ' + id);
  const source = path.resolve(sourcePath);
  const sourceHash = hash(source);
  const raw = path.join(HOME, 'raw', id + '-' + sourceHash.slice(0, 12) + '.png');
  fs.mkdirSync(path.dirname(raw), { recursive: true });
  if (!fs.existsSync(raw)) fs.copyFileSync(source, raw);
  if (hash(raw) !== sourceHash) throw new Error('Raw source hash mismatch');
  const meta = await sharp(raw).metadata();
  // Locate transparent gutters close to the requested regular grid; image generators can
  // vary the outer margin without changing icon order. Never cut through visible artwork.
  const rgba = await sharp(raw).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  if (batch.chromaKey === '#00ff00') {
    // Flood the chroma background, preserving enclosed physical emerald facets.
    // Exact key pixels also clear enclosed gaps; mint healing is outside this key.
    const queued = new Uint8Array(meta.width * meta.height), queue = new Int32Array(queued.length);
    let head = 0, tail = 0;
    const enqueue = pixel => {
      if (queued[pixel]) return;
      const i = pixel * 4, r = rgba.data[i], g = rgba.data[i + 1], b = rgba.data[i + 2];
      if (!(g > 150 && g > r * 1.7 && g > b * 1.7)) return;
      queued[pixel] = 1; queue[tail++] = pixel;
    };
    for (let x = 0; x < meta.width; x += 1) { enqueue(x); enqueue((meta.height - 1) * meta.width + x); }
    for (let y = 0; y < meta.height; y += 1) { enqueue(y * meta.width); enqueue(y * meta.width + meta.width - 1); }
    while (head < tail) {
      const pixel = queue[head++], x = pixel % meta.width, y = Math.floor(pixel / meta.width);
      rgba.data[pixel * 4 + 3] = 0;
      if (x) enqueue(pixel - 1); if (x < meta.width - 1) enqueue(pixel + 1);
      if (y) enqueue(pixel - meta.width); if (y < meta.height - 1) enqueue(pixel + meta.width);
    }
    for (let i = 0; i < rgba.data.length; i += 4) {
      const r = rgba.data[i], g = rgba.data[i + 1], b = rgba.data[i + 2];
      if (r < 25 && g > 220 && b < 25 || batch.kind === 'skill icons' && g > 150 && g > r * 1.7 && g > b * 1.7) rgba.data[i + 3] = 0;
    }
  }
  if (batch.extractionPolygons) {
    const inside = (x, y, polygon) => {
      let result = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const a = polygon[i], b = polygon[j];
        if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) result = !result;
      }
      return result;
    };
    for (let y = 0; y < meta.height; y += 1) for (let x = 0; x < meta.width; x += 1) {
      if (!batch.extractionPolygons.some(polygon => inside(x, y, polygon))) rgba.data[(y * meta.width + x) * 4 + 3] = 0;
    }
  }
  const cuts = (count, axis, orthogonalStart = 0, orthogonalEnd) => {
    const length = axis === 'x' ? meta.width : meta.height;
    const orthogonal = axis === 'x' ? meta.height : meta.width;
    const result = [0];
    for (let i = 1; i < count; i += 1) {
      const ideal = i * length / count, radius = .22 * length / count;
      let best = -1, bestScore = Infinity;
      for (let p = Math.floor(ideal - radius); p <= Math.ceil(ideal + radius); p += 1) {
        let visible = 0;
        for (let q = orthogonalStart; q < (orthogonalEnd || orthogonal); q += 1) for (let gutter = -3; gutter <= 3; gutter += 1) {
          const x = axis === 'x' ? p + gutter : q, y = axis === 'y' ? p + gutter : q;
          if (rgba.data[(y * meta.width + x) * 4 + 3] > 8) visible++;
        }
        const score = visible + Math.abs(p - ideal) / length;
        if (score < bestScore) { best = p; bestScore = score; }
      }
      result.push(best);
    }
    result.push(length);
    return result;
  };
  const rowCuts = cuts(batch.rows || 1, 'y');
  const grid = { xByRow: rowCuts.slice(0, -1).map((y, i) => cuts(batch.columns || 1, 'x', y, rowCuts[i + 1])), y: rowCuts };
  // Component ownership preserves long diagonal weapons when neighboring rows have
  // overlapping bounding boxes. Transparent gutters need not form a straight line.
  const components = [], labels = new Int32Array(meta.width * meta.height);
  if (batch.columns) {
    const queue = new Int32Array(labels.length);
    for (let seed = 0; seed < labels.length; seed += 1) {
      if (labels[seed] || rgba.data[seed * 4 + 3] <= 8) continue;
      const label = components.length + 1;
      let head = 0, tail = 1, sumX = 0, sumY = 0, left = meta.width, top = meta.height, right = 0, bottom = 0;
      queue[0] = seed; labels[seed] = label;
      while (head < tail) {
        const pixel = queue[head++], x = pixel % meta.width, y = Math.floor(pixel / meta.width);
        sumX += x; sumY += y; left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x); bottom = Math.max(bottom, y);
        for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx, ny = y + dy, next = ny * meta.width + nx;
          if (nx < 0 || ny < 0 || nx >= meta.width || ny >= meta.height || labels[next] || rgba.data[next * 4 + 3] <= 8) continue;
          labels[next] = label; queue[tail++] = next;
        }
      }
      const cx = sumX / tail, cy = sumY / tail;
      const row = Math.min(batch.rows - 1, Math.floor(cy / meta.height * batch.rows));
      const column = Math.min(batch.columns - 1, Math.floor(cx / meta.width * batch.columns));
      components.push({ label, area: tail, left, top, right, bottom, cx, cy, owner: row * batch.columns + column });
    }
    const major = components.filter(c => c.area >= 80);
    for (const c of components) if (c.area < 80 && major.length) {
      const closest = major.reduce((best, candidate) => {
        const distance = Math.hypot(Math.max(candidate.left - c.cx, 0, c.cx - candidate.right), Math.max(candidate.top - c.cy, 0, c.cy - candidate.bottom));
        return !best || distance < best.distance ? { owner: candidate.owner, distance } : best;
      }, null);
      c.owner = closest.owner;
    }
    grid.segmentation = 'connected alpha components assigned to requested cell centers; small edge fragments follow nearest major component';
  }
  const prepared = [];
  for (let i = 0; i < batch.entries.length; i += 1) {
    const entry = batch.entries[i];
    const sourceIndex = entry.sourceIndex == null ? i : entry.sourceIndex;
    let bytes;
    if (batch.kind === 'screen') bytes = await sharp(raw).resize(batch.width, batch.height, { fit: 'cover', position: 'centre' }).removeAlpha().png().toBuffer();
    else {
      const columns = batch.columns || 1, rows = batch.rows || 1;
      const columnCuts = grid.xByRow[Math.floor(sourceIndex / columns)];
      const x0 = columnCuts[sourceIndex % columns], x1 = columnCuts[sourceIndex % columns + 1];
      const y0 = grid.y[Math.floor(sourceIndex / columns)], y1 = grid.y[Math.floor(sourceIndex / columns) + 1];
      let cell;
      if (batch.columns) {
        const members = components.filter(c => c.owner === sourceIndex && c.area >= 2);
        if (!members.length) throw new Error(`${id}/${entry.id}: empty component group`);
        const left = Math.min(...members.map(c => c.left)), top = Math.min(...members.map(c => c.top));
        const right = Math.max(...members.map(c => c.right)), bottom = Math.max(...members.map(c => c.bottom));
        if (left < 2 || top < 2 || right > meta.width - 3 || bottom > meta.height - 3) throw new Error(`${id}/${entry.id}: art touches outer source boundary`);
        const memberLabels = new Set(members.map(c => c.label)), width = right - left + 5, height = bottom - top + 5;
        const data = Buffer.alloc(width * height * 4);
        for (let y = top; y <= bottom; y += 1) for (let x = left; x <= right; x += 1) {
          const sourcePixel = y * meta.width + x;
          if (!memberLabels.has(labels[sourcePixel])) continue;
          rgba.data.copy(data, ((y - top + 2) * width + x - left + 2) * 4, sourcePixel * 4, sourcePixel * 4 + 4);
        }
        cell = { data, info: { width, height, channels: 4 } };
      } else cell = await sharp(raw).extract({ left: x0, top: y0, width: x1 - x0, height: y1 - y0 }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
      let left = cell.info.width, top = cell.info.height, right = -1, bottom = -1, opaque = 0;
      for (let y = 0; y < cell.info.height; y += 1) for (let x = 0; x < cell.info.width; x += 1) { if (cell.data[(y * cell.info.width + x) * 4 + 3] > 8) { opaque++; left = Math.min(left, x); top = Math.min(top, y); right = Math.max(right, x); bottom = Math.max(bottom, y); } }
      if (opaque < 80 || opaque > cell.info.width * cell.info.height * .92) throw new Error(`${id}/${entry.id}: empty cell or missing genuine transparency`);
      if (left < 2 || top < 2 || right > cell.info.width - 3 || bottom > cell.info.height - 3) throw new Error(`${id}/${entry.id}: artwork touches source cell edge ${JSON.stringify({ left, top, right, bottom, width: cell.info.width, height: cell.info.height, grid })}`);
      const width = entry.size || batch.width || batch.size, height = entry.size || batch.height || batch.size;
      const pad = batch.kind === 'pedestal' ? 8 : width === 64 ? 6 : 16;
      const art = await sharp(cell.data, { raw: cell.info }).extract({ left, top, width: right - left + 1, height: bottom - top + 1 }).resize(width - 2 * pad, height - 2 * pad, { fit: 'inside', kernel: 'lanczos3' }).png().toBuffer();
      const am = await sharp(art).metadata();
      bytes = await sharp({ create: { width, height, channels: 4, background: { r: 0, g: 0, b: 0, alpha: 0 } } }).composite([{ input: art, left: Math.floor((width - am.width) / 2), top: Math.floor((height - am.height) / 2) }]).png().toBuffer();
    }
    prepared.push({ entry, bytes });
  }
  // Every crop is validated before replacing any member of the batch.
  for (const { entry, bytes } of prepared) { const target = path.join(ROOT, entry.output); fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, bytes); ledger.outputs[entry.output] = { batch: id, sha256: hash(target), sourceHash }; }
  ledger.imports[id] = { ...ledger.imports[id], raw: rel(raw), sha256: sourceHash, sourcePath: source, prompt: batch.promptPath, references: batch.references || catalog.references, promptSha256: hash(path.join(ROOT, batch.promptPath)), extraction: batch.extractionNotes || (batch.chromaKey ? 'Flood-connected chroma removal plus exact-key holes' : 'Native alpha component extraction'), tool: 'builtin image_gen', grid, importedAt: new Date().toISOString(), count: prepared.length };
  save(LEDGER, ledger);
  console.log(JSON.stringify({ imported: id, count: prepared.length, completed: Object.keys(ledger.outputs).length, expected: ledger.expectedRasterOutputs, raw: rel(raw) }));
}

async function verify() {
  const catalog = JSON.parse(fs.readFileSync(CATALOG)), ledger = JSON.parse(fs.readFileSync(LEDGER));
  const pending = [];
  for (const batch of catalog.batches) for (const e of batch.entries) { const record = ledger.outputs[e.output]; if (!record) { pending.push(e.output); continue; } if (hash(path.join(ROOT, e.output)) !== record.sha256) throw new Error('Output changed outside owner: ' + e.output); const m = await sharp(path.join(ROOT, e.output)).metadata(); if (m.width !== (e.size || batch.width || batch.size) || m.height !== (e.size || batch.height || batch.size)) throw new Error('Invalid output dimensions: ' + e.output); }
  for (const [p, h] of Object.entries(ledger.protectedSessionImages)) if (hash(path.join(ROOT, p)) !== h) throw new Error('Session reference changed: ' + p);
  for (const entry of Object.values(ledger.imports)) {
    if (hash(path.join(ROOT, entry.raw)) !== entry.sha256) throw new Error('Raw generation changed: ' + entry.raw);
    if (hash(path.join(ROOT, entry.prompt)) !== entry.promptSha256) throw new Error('Generation prompt changed: ' + entry.prompt);
  }
  for (const [p, e] of Object.entries(ledger.equipmentOutputs || {})) {
    if (hash(path.join(ROOT, p)) !== e.sha256 || hash(path.join(ROOT, e.source)) !== e.sourceHash) throw new Error('Equipment output or editable master changed outside owner: ' + p);
  }
  const outputPaths = catalog.batches.flatMap(b => b.entries.map(e => e.output));
  if (outputPaths.length !== 337 || new Set(outputPaths).size !== 337) throw new Error('Catalog must own exactly337 unique raster outputs.');
  console.log(JSON.stringify({ completed: Object.keys(ledger.outputs).length, expected: 337, pending: pending.length, pendingBatches: catalog.batches.filter(b => !ledger.imports[b.id]).map(b => b.id), referencesPreserved: Object.keys(ledger.protectedSessionImages).length, equipment: Object.keys(ledger.equipmentOutputs || {}).length }));
}

function owns() { return fs.existsSync(CATALOG); }

async function rebuild(family = 'all') {
  const catalog = JSON.parse(fs.readFileSync(CATALOG)), ledger = JSON.parse(fs.readFileSync(LEDGER));
  const matches = e => family === 'all' || (family === 'items' && e.output.includes('/items/')) || (family === 'skills' && e.output.includes('/skills/')) || (family === 'cards' && e.output.includes('/cards/')) || (family === 'menu' && e.output.includes('/ui/menu-icons/')) || (family === 'coupons' && /coupon/.test(e.id));
  const batches = catalog.batches.filter(b => b.entries.some(matches));
  if (!batches.length) throw new Error('Unknown icon family: ' + family);
  for (const b of batches) if (!ledger.imports[b.id]) throw new Error('Source-owned family is incomplete; cannot fall back to legacy art: ' + b.id);
  for (const b of batches) await importBatch(b.id, path.join(ROOT, ledger.imports[b.id].raw));
  return batches.flatMap(b => b.entries.map(e => e.output));
}

async function main() { const [mode, id, source] = process.argv.slice(2); if (mode === 'prepare') return prepare(); if (mode === 'import') return importBatch(id, source); if (mode === 'verify') return verify(); if (mode === 'rebuild') return rebuild(id); throw new Error('Use prepare | import <batch> <raw.png> | verify | rebuild [family]'); }
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1; });
module.exports = { prepare, importBatch, verify, makePrompt, owns, rebuild };
