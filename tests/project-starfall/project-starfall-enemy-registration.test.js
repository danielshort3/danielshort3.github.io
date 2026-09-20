'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const ROOT = path.resolve(__dirname, '../..');
const SOURCE_ROOT = path.join(ROOT, 'asset-sources/project-starfall/overhaul-v1/enemies');
const SIZE = 160;
const MAX_HORIZONTAL_DRIFT = 2;

// Reviewed image regions, not registration metadata or silhouette centers.
// Match stable anatomy; independently animated feet, weapons and tail tips are
// excluded. Vertical translation is a free fit so breathing and squash survive.
const CASES = Object.freeze({
  'eclipse-sovereign': { region: [52, 47, 62, 55], landmark: 'crown, face and upper torso', stride: 2 },
  'cinder-spitter': { region: [45, 77, 81, 50], landmark: 'head and torso', stride: 2 },
  'cracked-mimic': { region: [46, 73, 75, 53], landmark: 'lock and rigid chest body', stride: 2 },
  'bandit-cutter': { region: [64, 88, 44, 43], landmark: 'hood and torso', stride: 2 },
  'clockbug': { region: [51, 50, 77, 70], landmark: 'clock dial and body', stride: 2 },
  'briar-stag': { region: [66, 59, 58, 64], landmark: 'neck and torso', stride: 2 },
  'clockwork-titan': { region: [95, 83, 32, 37], landmark: 'chest clock', stride: 1 },
  'rimewarden': { region: [73, 77, 31, 29], landmark: 'face and upper chest', stride: 1 },
  'dew-slime': { region: [86, 106, 47, 29], landmark: 'eyes and face', stride: 1 },
  'brambleking': { region: [80, 90, 30, 31], landmark: 'face and trunk chest', stride: 1 },
  'index-scribe': { region: [85, 79, 38, 32], landmark: 'face and glasses', stride: 1 },
  'lava-tick': { region: [42, 75, 48, 34], landmark: 'dorsal carapace interior', stride: 1 }
});

const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const span = values => Math.max(...values) - Math.min(...values);

async function readIdleFrames(file) {
  const metadata = await sharp(file).metadata();
  assert.strictEqual(metadata.width, SIZE * 6, `${file}: six 160px columns required`);
  assert.strictEqual(metadata.height, SIZE * 8, `${file}: eight 160px rows required`);
  assert(metadata.hasAlpha, `${file}: alpha required`);
  return Promise.all(Array.from({ length: 6 }, (_, column) => sharp(file)
    .extract({ left: column * SIZE, top: 0, width: SIZE, height: SIZE })
    .ensureAlpha().raw().toBuffer()));
}

function regionSamples(reference, specification) {
  const [left, top, width, height] = specification.region;
  const result = [];
  for (let y = top; y < top + height; y += specification.stride) {
    for (let x = left; x < left + width; x += specification.stride) {
      const index = (y * SIZE + x) * 4;
      const alpha = reference[index + 3] / 255;
      result.push({ x, y, alpha, colors: [0, 1, 2].map(channel => reference[index + channel] / 255 * alpha) });
    }
  }
  assert(result.filter(sample => sample.alpha > 0.5).length > 100, 'Landmark region must contain a substantial visible body patch');
  return result;
}

function pixelError(samples, target, dx, dy) {
  let error = 0;
  for (const sample of samples) {
    const x = sample.x + dx;
    const y = sample.y + dy;
    const index = x >= 0 && y >= 0 && x < SIZE && y < SIZE ? (y * SIZE + x) * 4 : -1;
    const alpha = index >= 0 ? target[index + 3] / 255 : 0;
    error += 0.5 * (alpha - sample.alpha) ** 2;
    for (let channel = 0; channel < 3; channel += 1) {
      const color = index >= 0 ? target[index + channel] / 255 * alpha : 0;
      error += (color - sample.colors[channel]) ** 2;
    }
  }
  return error / samples.length / 3.5;
}

function fitTranslation(samples, target) {
  let best = { dx: 0, dy: 0, error: pixelError(samples, target, 0, 0) };
  // Broad enough to detect the original Sovereign 24px error and negative controls.
  for (let dy = -18; dy <= 18; dy += 2) {
    for (let dx = -40; dx <= 40; dx += 2) {
      const error = pixelError(samples, target, dx, dy);
      if (error < best.error) best = { dx, dy, error };
    }
  }
  const coarse = { ...best };
  for (let dy = coarse.dy - 1; dy <= coarse.dy + 1; dy += 1) {
    for (let dx = coarse.dx - 1; dx <= coarse.dx + 1; dx += 1) {
      const error = pixelError(samples, target, dx, dy);
      if (error < best.error) best = { dx, dy, error };
    }
  }
  return best;
}

function measureRegistration(frames, specification) {
  const samples = regionSamples(frames[0], specification);
  const fits = frames.map(frame => fitTranslation(samples, frame));
  return { fits, horizontalSpan: span(fits.map(fit => fit.dx)), horizontalLoopDelta: Math.abs(fits[5].dx) };
}

function assertRegistration(measurement, label) {
  assert(measurement.horizontalSpan <= MAX_HORIZONTAL_DRIFT,
    `${label}: stable body drifts ${measurement.horizontalSpan}px horizontally (limit ${MAX_HORIZONTAL_DRIFT}px), offsets [${measurement.fits.map(fit => fit.dx)}]`);
  assert(measurement.horizontalLoopDelta <= MAX_HORIZONTAL_DRIFT,
    `${label}: last-to-first body jump ${measurement.horizontalLoopDelta}px exceeds ${MAX_HORIZONTAL_DRIFT}px`);
}

function translateFrame(frame, dx, dy = 0) {
  const output = Buffer.alloc(frame.length);
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const tx = x + dx;
      const ty = y + dy;
      if (tx < 0 || ty < 0 || tx >= SIZE || ty >= SIZE) continue;
      frame.copy(output, (ty * SIZE + tx) * 4, (y * SIZE + x) * 4, (y * SIZE + x) * 4 + 4);
    }
  }
  return output;
}

function lavaCrownCenter(frame) {
  let top = -1;
  for (let y = 0; y < SIZE; y += 1) {
    let count = 0;
    for (let x = 25; x <= 107; x += 1) if (frame[(y * SIZE + x) * 4 + 3] > 128) count += 1;
    if (count >= 6) { top = y; break; }
  }
  assert(top >= 0 && top + 4 < SIZE, 'Lava Tick dorsal crown must be visible');
  let totalX = 0;
  let count = 0;
  for (let y = top; y <= top + 4; y += 1) {
    for (let x = 25; x <= 107; x += 1) {
      if (frame[(y * SIZE + x) * 4 + 3] <= 128) continue;
      totalX += x;
      count += 1;
    }
  }
  return totalX / count;
}

function assertLavaCrown(frames) {
  const centers = frames.map(lavaCrownCenter);
  assert(span(centers) <= MAX_HORIZONTAL_DRIFT, `Lava Tick crown drifts ${span(centers).toFixed(2)}px`);
  assert(Math.abs(centers[5] - centers[0]) <= MAX_HORIZONTAL_DRIFT, 'Lava Tick crown jumps at loop closure');
}

async function main() {
  const inventory = JSON.parse(fs.readFileSync(path.join(SOURCE_ROOT, 'inventory.json'), 'utf8'));
  for (const [id, specification] of Object.entries(CASES)) {
    const folder = path.join(SOURCE_ROOT, id);
    const config = JSON.parse(fs.readFileSync(path.join(folder, 'source.json'), 'utf8'));
    const report = JSON.parse(fs.readFileSync(path.join(folder, 'import-report.json'), 'utf8'));
    const frames = await readIdleFrames(path.join(ROOT, config.import.output));
    const published = inventory.items.find(item => item.fileId === id);
    assert.strictEqual(sha256(fs.readFileSync(path.join(ROOT, config.import.output))), published.replacement.sheetSha256,
      `${id}: production atlas disagrees with the imported inventory`);
    assert.strictEqual(config.registrationReview.status, 'reviewed-horizontal', `${id}: reviewed anchor provenance required`);
    assert.deepStrictEqual(report.registrationReview, config.registrationReview, `${id}: import report review is stale`);
    assert.strictEqual(report.sharedScale, config.sharedScale, `${id}: reviewed shared identity scale must remain fixed`);
    assert.strictEqual(sha256(fs.readFileSync(path.join(folder, config.source))), config.sourceSha256, `${id}: authored source checksum changed`);
    for (let column = 0; column < 6; column += 1) {
      const frame = report.frames.find(frame => frame.row === 0 && frame.outputColumn === column);
      assert.strictEqual(frame.anchorSource, 'explicit-pose-landmark', `${id}/${column}: nominal source-grid anchor returned`);
      assert.deepStrictEqual(frame.anchor, config.frames[frame.index].anchor, `${id}/${column}: importer did not use reviewed anchor`);
    }
    const measurement = measureRegistration(frames, specification);
    assertRegistration(measurement, `${id} ${specification.landmark}`);

    // A moved final pose must fail both steady registration and loop closure.
    // Controls use the actual image data, without old atlas/output dependencies.
    const control = Array.from({ length: 6 }, () => frames[0]);
    control[5] = translateFrame(frames[0], 8);
    const drift = measureRegistration(control, specification);
    assert(drift.horizontalSpan >= 7 && drift.horizontalLoopDelta >= 7, `${id}: +8px drift negative control was not detected`);
    assert.throws(() => assertRegistration(drift, `${id} negative control`), /drifts/);

    // A vertical breathing shift is not a horizontal registration regression.
    const verticalControl = Array.from({ length: 6 }, () => frames[0]);
    verticalControl[3] = translateFrame(frames[0], 0, -4);
    assertRegistration(measureRegistration(verticalControl, specification), `${id} vertical-motion control`);
    if (id === 'lava-tick') {
      assertLavaCrown(frames);
      assert.throws(() => assertLavaCrown(control), /drifts/);
    }
    console.log(`${id}: stable ${specification.landmark}, X span ${measurement.horizontalSpan}px, loop ${measurement.horizontalLoopDelta}px; drift control detected`);
  }
  console.log('Project Starfall enemy idle registration: 12 reviewed identities passed image-based checks. Other rows and unreviewed identities are outside this test.');
}

module.exports = { CASES, readIdleFrames, measureRegistration, assertRegistration, lavaCrownCenter, assertLavaCrown };
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
