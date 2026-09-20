'use strict';
const fs = require('fs');
const path = require('path');
const { load, crown } = require('./measure.cjs');
const ROOT = path.resolve(__dirname, '../..');
const sourcePath = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies/lava-tick/source.json');

async function main() {
  const report = JSON.parse(fs.readFileSync(path.join(__dirname, 'before-import-report.json'), 'utf8'));
  const config = JSON.parse(fs.readFileSync(path.join(__dirname, 'before-source.json'), 'utf8'));
  const image = await load(path.join(__dirname, 'before-sheet.png'));
  const targetX = 80;
  const corrections = [];
  config.sharedScale = report.sharedScale;
  config.frames ||= {};
  for (const frame of report.frames) {
    const landmark = crown(image, frame.row, frame.outputColumn);
    const translateX = Math.round(targetX - landmark.x);
    const scale = report.sharedScale * frame.scaleFactor;
    const anchor = { x: frame.anchor.x - translateX / scale, y: frame.anchor.y };
    if (frame.outputBounds.left + translateX < 4 || frame.outputBounds.right + translateX > 155) throw new Error(`Correction would compromise padding at ${frame.action}/${frame.column}`);
    config.frames[frame.index] = { ...(config.frames[frame.index] || {}), anchor };
    corrections.push({ index: frame.index, action: frame.action, column: frame.column, sourcePath: frame.sourcePath, beforeCrown: landmark, targetCrownX: targetX, translationX: translateX, beforeAnchor: frame.anchor, sourceAnchor: anchor, projectedCrownX: landmark.x + translateX });
  }
  config.registrationReview = {
    status: 'reviewed-horizontal',
    reviewedAt: '2026-09-20',
    scope: 'All 48 Lava Tick poses in idle, move, telegraph, attack, projectile, buff, hit and defeat. Horizontal registration only; no new artwork.',
    landmark: 'Rigid dorsal carapace crown, measured from the upper five solid contour rows in the reviewed body region. Excludes head horns, jaw and feet; not the complete silhouette or alpha bounding box.',
    sourceCoordinates: 'Each frames[index].anchor is absolute in that frame sourcePath coordinate space. Main-source rows use the existing derived row image coordinate space; hit poses use the supplemental image coordinate space.',
    corrections: 'Explicit per-pose horizontal anchors remove erroneous source-grid placement. All vertical anchors, authored crouch/rise/recoil shapes, jaw/leg motion and the original shared scale remain unchanged. Pixel artwork is only translated within each existing 160px cell.',
    evidencePath: 'output/lava-tick-alignment/evidence.json',
    comparisonPath: 'output/lava-tick-alignment/comparison-all-actions.png',
    limitations: 'This resolves horizontal carapace registration. Generated fissure texture, shell shape and pose-to-pose anatomical differences remain visible; it does not certify all poses or all game animations as perfectly seamless.'
  };
  config.visualReview.registrationFollowup = 'Horizontal source anchors reviewed against the dorsal shell crown and independent shell-region correspondence. Original generated pose/art and existing vertical motion retained.';
  fs.writeFileSync(sourcePath, JSON.stringify(config, null, 2) + '\n');
  fs.writeFileSync(path.join(__dirname, 'corrections.json'), JSON.stringify({ targetX, sharedScale: report.sharedScale, corrections }, null, 2) + '\n');
  console.log(JSON.stringify({ count: corrections.length, sharedScale: config.sharedScale, perRowTranslations: Array.from({ length: 8 }, (_, row) => corrections.slice(row * 6, row * 6 + 6).map(frame => frame.translationX)) }));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
