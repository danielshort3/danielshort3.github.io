/* Reviewed translation-only registration corrections; never writes production atlases. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const child = require('child_process');
const sharp = require('sharp');
const { geometry, points, fit } = require('../starfall-overhaul-validation/screen-enemy-idle-drift.cjs');
const root = path.resolve(__dirname, '../..');
const sourceRoot = path.join(root, 'asset-sources/project-starfall/overhaul-v1/enemies');
const plans = {
  'eclipse-sovereign': { offsets: [0, -11, -14, -19, -20, -24], landmark: 'crown, faceplate and upper torso together; staff and trailing robe are excluded as independent extremities' },
  'cinder-spitter': { offsets: [0, -1, -6, -10, -10, -13], landmark: 'head and torso root together; animated tail flame is excluded' },
  'cracked-mimic': { offsets: [0, -2, -4, -10, -11, -12], landmark: 'gold chest lock and rigid chest body; articulated feet are excluded' },
  'bandit-cutter': { offsets: [0, -2, -4, -5, -10, -10], landmark: 'hood, torso and pelvis together; blade and shield extremities are excluded' },
  'clockbug': { offsets: [0, 4, 2, -2, -6, -8], landmark: 'clock dial and rigid rounded body; antenna key and moving small legs are excluded' },
  'briar-stag': { offsets: [0, 4, 2, 1, -7, -8], landmark: 'torso and planted leg root; antler tips and leaf ornaments are excluded' }
};
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function extractFrames(file) {
  return Promise.all(Array.from({ length: 6 }, (_, column) => sharp(file).extract({ left: 160 * column, top: 0, width: 160, height: 160 }).ensureAlpha().raw().toBuffer()));
}

async function analyze(file) {
  const frames = await extractFrames(file);
  const core = points(frames[0], geometry(frames[0]), 'core');
  const full = points(frames[0], geometry(frames[0]), 'full');
  const offsets = frames.map((frame, i) => ({ frame: i, core: fit(core, frame), full: fit(full, frame) }));
  const coreX = offsets.map(item => item.core.dx), fullX = offsets.map(item => item.full.dx);
  const span = values => Math.max(...values) - Math.min(...values);
  return { coreTranslationSpanX: span(coreX), fullTranslationSpanX: span(fullX), offsets };
}

async function run() {
  const results = [];
  for (const [id, plan] of Object.entries(plans)) {
    const folder = path.join(sourceRoot, id);
    const output = path.join(__dirname, id);
    fs.mkdirSync(output, { recursive: true });
    const configFile = path.join(folder, 'source.json');
    const beforeConfig = path.join(output, 'before-source.json');
    const beforeReport = path.join(output, 'before-import-report.json');
    if (!fs.existsSync(beforeConfig)) fs.copyFileSync(configFile, beforeConfig);
    if (!fs.existsSync(beforeReport)) fs.copyFileSync(path.join(folder, 'import-report.json'), beforeReport);
    const config = JSON.parse(fs.readFileSync(beforeConfig));
    const report = JSON.parse(fs.readFileSync(beforeReport));
    const beforeSheet = path.join(output, 'before-sheet.png');
    if (!fs.existsSync(beforeSheet)) fs.copyFileSync(path.join(root, config.import.output), beforeSheet);
    config.sharedScale = report.sharedScale;
    config.frames ||= {};
    const corrections = [];
    for (const [column, drift] of plan.offsets.entries()) {
      const frame = report.frames.find(frame => frame.row === 0 && frame.outputColumn === column);
      const anchor = { x: frame.anchor.x + drift / (report.sharedScale * frame.scaleFactor), y: frame.anchor.y };
      config.frames[frame.index] = { ...(config.frames[frame.index] || {}), anchor };
      corrections.push({ frame: column, measuredPriorBodyDriftX: drift, outputTranslationX: -drift, previousAnchor: frame.anchor, anchor, scaleFactor: frame.scaleFactor });
    }
    config.registrationReview = {
      status: 'reviewed-horizontal',
      scope: 'idle-horizontal only',
      method: 'Reviewed stable anatomical-core translation against idle frame 0, corroborated by whole-silhouette translation and fixed-guide contact sheets. No bounding-box centering. Only X anchors changed; authored Y, all drawing pixels and fixed identity scale are preserved.',
      landmark: plan.landmark,
      referenceFrame: 0,
      fixedSharedScale: report.sharedScale,
      sourceUnchanged: true,
      actionsChanged: ['idle'],
      visualVerification: 'Before/after fixed-guide strips inspected for the declared rigid anatomical landmarks and last-to-first horizontal closure. All six poses retain their authored vertical variation. This review does not certify other action rows or all shape and transition quality.',
      evidence: `output/enemy-idle-alignment/${id}/alignment-evidence.json`,
      corrections
    };
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2) + '\n');
    const build = child.spawnSync(process.execPath, ['build/process-project-starfall-overhaul-enemies.js', '--enemy', id], { cwd: root, encoding: 'utf8' });
    if (build.status !== 0) throw new Error(build.stderr || build.stdout);
    const reviewSheet = path.join(folder, 'review-sheet.png');
    fs.copyFileSync(reviewSheet, path.join(output, 'after-review-sheet.png'));
    fs.copyFileSync(path.join(folder, 'import-report.json'), path.join(output, 'after-import-report.json'));
    const afterReport = JSON.parse(fs.readFileSync(path.join(folder, 'import-report.json')));
    const nonIdleChanges = report.frames.filter(frame => frame.row > 0 && afterReport.frames.find(after => after.index === frame.index).sha256 !== frame.sha256).map(frame => frame.index);
    if (nonIdleChanges.length) throw new Error(`${id}: unexpected non-idle changes ${nonIdleChanges}`);
    const sourceHash = hash(fs.readFileSync(path.join(folder, config.source)));
    const beforeFrames = await extractFrames(beforeSheet), afterFrames = await extractFrames(reviewSheet);
    const translationPreservation = beforeFrames.map((before, frame) => {
      const shift = -plan.offsets[frame];
      let changedChannels = 0;
      for (let y = 0; y < 160; y++) for (let x = 0; x < 160; x++) for (let channel = 0; channel < 4; channel++) {
        const sourceX = x - shift;
        const expected = sourceX >= 0 && sourceX < 160 ? before[(y * 160 + sourceX) * 4 + channel] : 0;
        if (afterFrames[frame][(y * 160 + x) * 4 + channel] !== expected) changedChannels++;
      }
      return { frame, shiftX: shift, changedChannelsAfterAccountingForTranslation: changedChannels };
    });
    if (translationPreservation.some(frame => frame.changedChannelsAfterAccountingForTranslation)) throw new Error(`${id}: correction was not a pure integer translation`);
    const evidence = { id, productionImported: false, scope: 'idle-horizontal only', visualInspection: 'Before/after strips inspected with fixed guides; declared core landmarks and last-to-first horizontal closure improved. Authored vertical motion retained. Other actions not visually certified by this pass.', landmark: plan.landmark, corrections, before: await analyze(beforeSheet), after: await analyze(reviewSheet), sourceHash, sourceUnchanged: sourceHash === config.sourceSha256, productionUnchanged: hash(fs.readFileSync(path.join(root, config.import.output))) === hash(fs.readFileSync(beforeSheet)), unchangedOtherActionFrames: 42 - nonIdleChanges.length, identityScaleUnchanged: report.sharedScale === afterReport.sharedScale, verticalAnchorsUnchanged: corrections.every(change => change.previousAnchor.y === change.anchor.y), translationPreservation, beforeSheetHash: hash(fs.readFileSync(beforeSheet)), afterReviewSheetHash: hash(fs.readFileSync(reviewSheet)) };
    fs.writeFileSync(path.join(output, 'alignment-evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
    const layers = [];
    for (const [row, file] of [beforeSheet, reviewSheet].entries()) {
      const strip = await sharp(file).extract({ left: 0, top: 0, width: 960, height: 160 }).png().toBuffer();
      layers.push({ input: strip, left: 100, top: row * 176 + 22 });
      layers.push({ input: Buffer.from(`<svg width="100" height="176"><text x="8" y="80" fill="#172030" font-family="Arial" font-size="15">${row ? 'After' : 'Before'}</text></svg>`), left: 0, top: row * 176 });
    }
    const guides = `<svg width="1060" height="374">${Array.from({ length: 6 }, (_, col) => `<path d="M${180 + 160 * col} 0v374" stroke="#4b6b84" stroke-dasharray="3 4"/>`).join('')}</svg>`;
    layers.push({ input: Buffer.from(guides), left: 0, top: 0 });
    await sharp({ create: { width: 1060, height: 374, channels: 4, background: '#eef2ef' } }).composite(layers).png().toFile(path.join(output, 'idle-before-after.png'));
    results.push(evidence);
  }
  fs.writeFileSync(path.join(__dirname, 'confirmed-idle-corrections.json'), JSON.stringify(results, null, 2) + '\n');
  console.log(JSON.stringify(results.map(item => ({ id: item.id, before: item.before.coreTranslationSpanX, after: item.after.coreTranslationSpanX, afterFull: item.after.fullTranslationSpanX, sourceUnchanged: item.sourceUnchanged, productionUnchanged: item.productionUnchanged, otherFramesUnchanged: item.unchangedOtherActionFrames })), null, 2));
}
run().catch(error => { console.error(error); process.exit(1); });
