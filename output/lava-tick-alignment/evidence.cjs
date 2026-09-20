'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const sharp = require('sharp');
const assert = require('assert/strict');
const { load, shellTemplate, match, crown } = require('./measure.cjs');
const ROOT = path.resolve(__dirname, '../..');
const SOURCE = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies/lava-tick');
const AFTER = path.join(ROOT, 'img/project-starfall/animations/enemies/lava-tick-sheet.png');
const BEFORE = path.join(__dirname, 'before-sheet.png');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const span = values => Math.max(...values) - Math.min(...values);
const actions = ['Idle', 'Move', 'Telegraph', 'Attack', 'Projectile', 'Buff', 'Hit', 'Defeat'];

async function compareSheets(before, after, beforeReport, afterReport, corrections) {
  const checks = [];
  for (const change of corrections) {
    const frame = beforeReport.frames[change.index];
    const next = afterReport.frames.find(entry => entry.index === change.index);
    const dx = change.translationX;
    assert.equal(frame.anchor.y, next.anchor.y, 'Vertical source anchor changed.');
    assert.equal(frame.outputBounds.top, next.outputBounds.top, 'Vertical output position changed.');
    assert.equal(frame.outputBounds.bottom, next.outputBounds.bottom, 'Vertical output extent changed.');
    assert.equal(frame.outputBounds.width, next.outputBounds.width, 'Pose width changed.');
    assert.equal(frame.outputBounds.height, next.outputBounds.height, 'Pose height changed.');
    let differentPixels = 0;
    for (let y = 0; y < 160; y += 1) {
      for (let x = 0; x < 160; x += 1) {
        const fromX = x - dx;
        const a = ((frame.row * 160 + y) * before.width + frame.outputColumn * 160 + fromX) * 4;
        const b = ((frame.row * 160 + y) * after.width + frame.outputColumn * 160 + x) * 4;
        let differs = false;
        for (let channel = 0; channel < 4; channel += 1) {
          if ((fromX < 0 || fromX >= 160 ? 0 : before.data[a + channel]) !== after.data[b + channel]) differs = true;
        }
        if (differs) differentPixels += 1;
      }
    }
    assert.equal(differentPixels, 0, `Pose ${frame.index} changed beyond declared translation.`);
    checks.push({ index: frame.index, action: frame.action, column: frame.column, translationX: dx, changedPixelsAfterUndoingTranslation: differentPixels, verticalPositionUnchanged: true, scaleUnchanged: true });
  }
  return checks;
}

async function contacts() {
  const composites = [];
  const width = 2052, header = 60, rowHeight = 184, labelWidth = 100;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${header + rowHeight * 8}"><style>text{font-family:Arial,sans-serif;fill:#293944}.heading{font-size:20px;font-weight:700}.label{font-size:15px}.note{font-size:12px;fill:#5a676d}</style><rect width="100%" height="100%" fill="#edeae3"/><text class="heading" x="100" y="26">Before: uniform grid anchors</text><text class="heading" x="1080" y="26">After: reviewed carapace anchors</text><text class="note" x="100" y="47">Same 48 authored poses, scale and vertical motion. Dashed lines mark the fixed body axis and ground.</text>`;
  for (let row = 0; row < 8; row += 1) {
    const top = header + row * rowHeight;
    svg += `<text class="label" x="8" y="${top + 85}">${actions[row]}</text>`;
    for (let side = 0; side < 2; side += 1) {
      const left = labelWidth + side * 980;
      svg += `<rect x="${left}" y="${top}" width="960" height="160" rx="4" fill="${row % 2 ? '#dfe7e8' : '#f7f4ee'}"/>`;
      for (let column = 0; column < 6; column += 1) {
        const x = left + column * 160;
        svg += `<path d="M${x + 80} ${top}v160 M${x} ${top + 150}h160" stroke="#829297" opacity=".4" stroke-dasharray="3 4"/><text class="note" x="${x + 74}" y="${top + 176}">${column + 1}</text>`;
      }
      composites.push({ input: await sharp(side ? AFTER : BEFORE).extract({ left: 0, top: row * 160, width: 960, height: 160 }).png().toBuffer(), left, top });
    }
  }
  svg += '</svg>';
  await sharp(Buffer.from(svg)).composite(composites).png().toFile(path.join(__dirname, 'comparison-all-actions.png'));
  const idle = [];
  const idleSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="408"><rect width="960" height="408" fill="#edeae3"/><g font-family="Arial" font-size="18" fill="#293944"><text x="12" y="26">Before — visible sideways drift through the six idle poses</text><text x="12" y="228">After — carapace stays registered; leg and facial motion preserved</text></g>${[40,242].map(top => Array.from({length:6},(_,c)=>`<path d="M${c*160+80} ${top}v160" stroke="#829297" opacity=".5" stroke-dasharray="3 4"/>`).join('')).join('')}</svg>`;
  for (let side = 0; side < 2; side += 1) idle.push({ input: await sharp(side ? AFTER : BEFORE).extract({ left: 0, top: 0, width: 960, height: 160 }).png().toBuffer(), left: 0, top: side ? 242 : 40 });
  await sharp(Buffer.from(idleSvg)).composite(idle).png().toFile(path.join(__dirname, 'comparison-idle.png'));
}

async function main() {
  const before = await load(BEFORE), after = await load(AFTER);
  const beforeReport = JSON.parse(fs.readFileSync(path.join(__dirname, 'before-import-report.json'), 'utf8'));
  const afterReport = JSON.parse(fs.readFileSync(path.join(SOURCE, 'import-report.json'), 'utf8'));
  const { corrections } = JSON.parse(fs.readFileSync(path.join(__dirname, 'corrections.json'), 'utf8'));
  assert.equal(beforeReport.sharedScale, afterReport.sharedScale);
  const template = shellTemplate(before);
  const idleMatches = (image) => Array.from({ length: 6 }, (_, column) => match(image, template, 0, column));
  const oldMatches = idleMatches(before), newMatches = idleMatches(after);
  const oldCrown = Array.from({ length: 6 }, (_, column) => crown(before, 0, column).x);
  const newCrown = Array.from({ length: 6 }, (_, column) => crown(after, 0, column).x);
  assert.ok(span(newMatches.map(frame => frame.dx)) <= 2, 'Rigid shell registration still drifts horizontally.');
  assert.ok(Math.abs(newMatches[5].dx - newMatches[0].dx) <= 1, 'Idle loop retains a horizontal shell jump.');
  const evidence = {
    schema: 'starfall-lava-tick-horizontal-registration-review-v1',
    reviewedAt: new Date().toISOString(),
    sourceSha256: hash(path.join(SOURCE, 'source.png')),
    supplementalSourceSha256: hash(path.join(SOURCE, 'hit-supplement.png')),
    beforeSheetSha256: hash(BEFORE), afterSheetSha256: hash(AFTER),
    scope: '48 horizontal source anchors, all eight Lava Tick rows. No image generation, pose redraw, rescaling, or vertical correction.',
    sharedScale: beforeReport.sharedScale,
    idle: {
      beforeCrownX: oldCrown, afterCrownX: newCrown,
      beforeCrownRangePx: span(oldCrown), afterCrownRangePx: span(newCrown),
      independentCheck: 'RGB/alpha correspondence using an upper carapace interior patch x42..89,y75..108 from the original idle frame 0. This is separate from the contour landmark used to set anchors and excludes feet, face and jaw.',
      beforeMatches: oldMatches, afterMatches: newMatches,
      beforeShellTranslationRangePx: span(oldMatches.map(frame => frame.dx)),
      afterShellTranslationRangePx: span(newMatches.map(frame => frame.dx)),
      beforeLoopHorizontalJumpPx: Math.abs(oldMatches[5].dx - oldMatches[0].dx),
      afterLoopHorizontalJumpPx: Math.abs(newMatches[5].dx - newMatches[0].dx)
    },
    posePreservation: await compareSheets(before, after, beforeReport, afterReport, corrections),
    reviewLimitations: [
      'The independent rigid patch check establishes improved idle body registration; it is not a claim of identical shell artwork across frames.',
      'Existing 1-2px vertical idle differences, fissure/highlight changes and shell/head shape variations remain in the original generated drawings.',
      'Attack, support, hit and defeat poses deliberately change posture. Their explicit carapace anchors were reviewed as action poses; all vertical movement and the original pixel artwork are preserved.'
    ]
  };
  await contacts();
  fs.writeFileSync(path.join(__dirname, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
  console.log(JSON.stringify({ poses: evidence.posePreservation.length, idleBeforePx: evidence.idle.beforeShellTranslationRangePx, idleAfterPx: evidence.idle.afterShellTranslationRangePx, seamBeforePx: evidence.idle.beforeLoopHorizontalJumpPx, seamAfterPx: evidence.idle.afterLoopHorizontalJumpPx, pixelArtworkUnchangedAfterUndoingTranslation: true, sharedScale: evidence.sharedScale }));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
