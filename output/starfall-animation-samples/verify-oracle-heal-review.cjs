'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const reviewUrl = process.argv[2] || process.env.STARFALL_REVIEW_URL || 'http://127.0.0.1:4187/';
const outputDirectory = __dirname;
const widths = [736, 390, 320];

function expectedFrame(clip, seconds) {
  const holds = clip.frames.map((frame, index) => Number(clip.holds && clip.holds[index]) || 1);
  const total = holds.reduce((sum, hold) => sum + hold, 0);
  const phase = ((seconds % clip.duration) + clip.duration) % clip.duration;
  const value = phase / clip.duration * total;
  let end = 0;
  for (let index = 0; index < holds.length; index += 1) {
    end += holds[index];
    if (value < end - 1e-8) return index;
  }
  return holds.length - 1;
}

async function pause(root) {
  if (await root.locator('[data-play]').innerText() === 'Pause') await root.locator('[data-play]').click();
}

async function setScrub(root, value) {
  await root.locator('[data-scrub]').evaluate((input, next) => {
    input.value = String(next);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function inspectFrames(root) {
  return root.evaluate((element) => ({
    elapsed: Number(element.dataset.elapsed),
    indices: ['before', 'after'].map((name) => Number(element.querySelector(`[data-${name}]`).dataset.frame)),
    bounds: ['before', 'after'].map((name) => JSON.parse(element.querySelector(`[data-${name}]`).dataset.bounds)),
    fxFrames: ['before', 'after'].map((name) => JSON.parse(element.querySelector(`[data-${name}]`).dataset.fxFrames)),
    fxBounds: ['before', 'after'].map((name) => JSON.parse(element.querySelector(`[data-${name}]`).dataset.fxBounds)),
    fxOpacity: ['before', 'after'].map((name) => JSON.parse(element.querySelector(`[data-${name}]`).dataset.fxOpacity))
  }));
}

function frameMidpoints(clip) {
  const holds = clip.frames.map((frame, index) => Math.max(1, Number(clip.holds && clip.holds[index]) || 1));
  const total = holds.reduce((sum, hold) => sum + hold, 0);
  let cursor = 0;
  return holds.map((hold) => {
    const midpoint = (cursor + hold / 2) / total * clip.duration;
    cursor += hold;
    return midpoint;
  });
}

function expectedEffects(clip, seconds) {
  const phase = ((seconds % clip.duration) + clip.duration) % clip.duration;
  return (clip.effects || []).map((effect) => {
    const localTime = phase - (Number(effect.start) || 0);
    if (localTime < 0 || localTime >= effect.duration) return { frame: null, opacity: 0 };
    const fadeIn = Number(effect.fadeIn) || 0;
    const fadeOut = Number(effect.fadeOut) || 0;
    return {
      frame: expectedFrame(effect, localTime),
      opacity: (effect.opacity === undefined ? 1 : Number(effect.opacity))
        * (fadeIn ? Math.min(1, localTime / fadeIn) : 1)
        * (fadeOut ? Math.min(1, (effect.duration - localTime) / fadeOut) : 1)
    };
  });
}

async function setElapsed(frame, elapsed, zoom = 2, playing = false) {
  await frame.evaluate((value) => {
    window.dispatchEvent(new CustomEvent('openai:set_globals', {
      detail: { globals: { widgetState: {
        modelContent: { speed: 1 },
        privateContent: { zoom: value.zoom, playing: value.playing, elapsedSeconds: value.elapsed }
      } } }
    }));
  }, { elapsed, zoom, playing });
}

function checkEffects(data, sample, label, scene) {
  [data.before, data.after].forEach((clip, clipIndex) => {
    const expected = expectedEffects(clip, sample.elapsed);
    assert.deepStrictEqual(sample.fxFrames[clipIndex], expected.map((entry) => entry.frame), `${label}: independently sampled FX frames`);
    expected.forEach((entry, effectIndex) => {
      assert(Math.abs(sample.fxOpacity[clipIndex][effectIndex] - entry.opacity) < 1e-7, `${label}: correct fade opacity`);
      const bounds = sample.fxBounds[clipIndex][effectIndex];
      if (entry.frame === null) assert.strictEqual(bounds, null, `${label}: inactive effect has no bounds`);
      else assertVisibleBounds(bounds, scene.width, scene.height, `${label}: effect ${effectIndex} frame ${entry.frame}`);
    });
  });
}

function assertVisibleBounds(bounds, width, height, label) {
  for (const [name, value] of Object.entries(bounds)) assert(Number.isFinite(value), `${label}: finite ${name}`);
  assert(bounds.left >= -0.5, `${label}: visible left ${bounds.left} must fit canvas`);
  assert(bounds.top >= -0.5, `${label}: visible top ${bounds.top} must fit canvas`);
  assert(bounds.right <= width + 0.5, `${label}: visible right ${bounds.right} must fit canvas`);
  assert(bounds.bottom <= height + 0.5, `${label}: visible bottom ${bounds.bottom} must fit canvas`);
  assert(bounds.right > bounds.left && bounds.bottom > bounds.top, `${label}: visible sprite must have positive area`);
}

async function checkLayout(frame, root, width) {
  const geometry = await root.evaluate((element) => {
    const rect = (node) => {
      const value = node.getBoundingClientRect();
      return { left: value.left, top: value.top, right: value.right, bottom: value.bottom, width: value.width, height: value.height };
    };
    const controls = ['speed', 'zoom', 'play', 'step', 'scrub'].map((name) => ({ name, ...rect(element.querySelector(`[data-${name}]`)) }));
    return { root: rect(element), controls, panels: [...element.querySelectorAll('.hop-view')].map(rect) };
  });
  const overflow = await frame.evaluate(() => ({ viewport: innerWidth, scrollWidth: document.documentElement.scrollWidth }));
  assert(overflow.scrollWidth <= overflow.viewport + 1, `${width}: no horizontal overflow`);
  for (const control of geometry.controls) {
    assert(control.width > 0 && control.height > 0, `${width}: ${control.name} is visible`);
    assert(control.left >= geometry.root.left - 1 && control.right <= geometry.root.right + 1, `${width}: ${control.name} fits inline width`);
  }
  for (let a = 0; a < geometry.controls.length; a += 1) {
    for (let b = a + 1; b < geometry.controls.length; b += 1) {
      const first = geometry.controls[a];
      const second = geometry.controls[b];
      const overlapX = Math.min(first.right, second.right) - Math.max(first.left, second.left);
      const overlapY = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
      assert(overlapX <= 1 || overlapY <= 1, `${width}: ${first.name} and ${second.name} do not overlap`);
    }
  }
  const [beforePanel, afterPanel] = geometry.panels;
  if (width <= 390) assert(afterPanel.top >= beforePanel.bottom - 1, `${width}: comparison panels stack`);
  else assert(afterPanel.left >= beforePanel.right - 1, `${width}: comparison panels remain side by side`);
  return { viewport: overflow.viewport, scrollWidth: overflow.scrollWidth, controls: geometry.controls.map((control) => control.name) };
}

async function run() {
  const browser = await chromium.launch({ headless: true });
  const results = [];
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1600 }, deviceScaleFactor: 1 });
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      page.on('console', (message) => {
        if (message.type() === 'error') errors.push(message.text());
      });
      await page.goto(reviewUrl, { waitUntil: 'networkidle' });
      await page.frameLocator('iframe').locator('#oracle-heal-review[data-ready="true"]').waitFor({ timeout: 15000 });
      let frame = null;
      for (const candidate of page.frames()) {
        if (candidate.parentFrame() && await candidate.locator('#oracle-heal-review').count()) frame = candidate;
      }
      assert(frame, 'The rendered visualization must exist inside its iframe');
      const root = frame.locator('#oracle-heal-review');
      const data = await root.locator('[data-review-data]').evaluate((element) => JSON.parse(element.textContent));
      const scene = { width: 320, height: 280, ...data.scene };
      assert(data.before.frames.length > 0 && data.after.frames.length > 0, 'Both clips must contain real frames');
      assert.strictEqual(await root.locator('[data-before-label]').innerText(), data.beforeLabel || data.before.label || 'Current cast');
      assert.strictEqual(await root.locator('[data-after-label]').innerText(), data.afterLabel || data.after.label || 'Healing cast study');
      assert.strictEqual(await root.locator('[data-title]').textContent(), typeof data.title === 'string' ? data.title : 'Icebloom Oracle · healing study');
      assert.strictEqual(await root.locator('[data-key-poses] canvas').count(), Math.min(4, data.keyPoses.length), 'Requested key poses are visible without playing');
      await pause(root);
      await setScrub(root, 0);
      const sampleResults = [];
      for (const value of [0, 100, 333, 500, 750, 999]) {
        const previous = await inspectFrames(root);
        const requestedElapsed = (Math.floor(previous.elapsed / data.after.duration) + value / 1000) * data.after.duration;
        await setScrub(root, value);
        const sample = await inspectFrames(root);
        assert(Math.abs(sample.elapsed - requestedElapsed) < 1e-5, `Scrub ${value} preserves its exact requested position instead of snapping to a frame start`);
        assert.strictEqual(sample.indices[0], expectedFrame(data.before, sample.elapsed), `Current clip uses its own duration at scrub ${value}`);
        assert.strictEqual(sample.indices[1], expectedFrame(data.after, sample.elapsed), `New clip uses its own duration at scrub ${value}`);
        checkEffects(data, sample, `${width}: scrub ${value}`, scene);
        sampleResults.push({ scrub: value, elapsed: sample.elapsed, frames: sample.indices });
      }
      await setScrub(root, 333);
      const beforeEcho = await inspectFrames(root);
      const savedZoom = Number(await root.locator('[data-zoom]').inputValue());
      const savedSpeed = Number(await root.locator('[data-speed]').inputValue());
      await frame.evaluate((snapshot) => {
        window.dispatchEvent(new CustomEvent('openai:set_globals', {
          detail: { globals: { widgetState: {
            modelContent: { speed: snapshot.speed, selectedFrame: snapshot.selectedFrame },
            privateContent: { zoom: snapshot.zoom, playing: false, elapsedSeconds: snapshot.elapsed }
          } } }
        }));
      }, { speed: savedSpeed, zoom: savedZoom, selectedFrame: beforeEcho.indices[1], elapsed: beforeEcho.elapsed });
      const afterEcho = await inspectFrames(root);
      assert(Math.abs(afterEcho.elapsed - beforeEcho.elapsed) < 1e-8, 'A host persistence echo preserves the exact elapsed time within a pose');
      assert.deepStrictEqual(afterEcho.indices, beforeEcho.indices, 'A host persistence echo preserves both independently timed poses');
      const paused = await inspectFrames(root);
      await page.waitForTimeout(180);
      assert.deepStrictEqual(await inspectFrames(root), paused, 'Pause preserves time and both rendered frames');

      const visitedBefore = new Set();
      const visitedAfter = new Set();
      const stepCount = data.after.frames.length * (Math.ceil(data.before.duration / data.after.duration) + 1);
      for (const zoom of [1, 2]) {
        await root.locator('[data-zoom]').selectOption(String(zoom));
        await setScrub(root, 0);
        for (let step = 0; step < stepCount; step += 1) {
          const beforeStep = await inspectFrames(root);
          visitedBefore.add(beforeStep.indices[0]);
          visitedAfter.add(beforeStep.indices[1]);
          beforeStep.bounds.forEach((bounds, index) => assertVisibleBounds(bounds, scene.width, scene.height, `${width} / ${zoom}× / ${index === 0 ? 'before' : 'after'} / frame ${beforeStep.indices[index]}`));
          checkEffects(data, beforeStep, `${width}: step ${step} / ${zoom}×`, scene);
          await root.locator('[data-step]').click();
          const afterStep = await inspectFrames(root);
          assert.strictEqual(afterStep.indices[1], (beforeStep.indices[1] + 1) % data.after.frames.length, 'Next frame advances exactly one new-study frame');
          assert.strictEqual(afterStep.indices[0], expectedFrame(data.before, afterStep.elapsed), 'Stepping keeps current clip on its own timeline');
        }
      }
      assert.strictEqual(visitedAfter.size, data.after.frames.length, 'Every new pose was inspected');
      assert.strictEqual(visitedBefore.size, data.before.frames.length, 'Every current pose was inspected');
      const thumbnailBounds = await root.locator('[data-key-poses] canvas').evaluateAll((elements) => elements.map((canvas) => JSON.parse(canvas.dataset.bounds)));
      thumbnailBounds.forEach((bounds, index) => assertVisibleBounds(bounds, 160, 160, `${width} key pose ${index}`));
      const thumbnails = await root.locator('[data-key-poses] canvas').evaluateAll((elements) => elements.map((canvas) => ({
        elapsed: Number(canvas.dataset.sampleTime), frame: Number(canvas.dataset.frame),
        fxFrames: JSON.parse(canvas.dataset.fxFrames), fxBounds: JSON.parse(canvas.dataset.fxBounds)
      })));
      const midpoints = frameMidpoints(data.after);
      thumbnails.forEach((thumbnail, index) => {
        assert(Math.abs(thumbnail.elapsed - midpoints[thumbnail.frame]) < 1e-8, `${width}: key pose ${index} uses its own midpoint time`);
        assert.deepStrictEqual(thumbnail.fxFrames, expectedEffects(data.after, thumbnail.elapsed).map((entry) => entry.frame), `${width}: key pose ${index} has corresponding FX`);
        thumbnail.fxBounds.filter(Boolean).forEach((bounds) => assertVisibleBounds(bounds, 160, 160, `${width}: key pose ${index} FX`));
      });

      const effectResults = [];
      for (const [effectIndex, effect] of (data.after.effects || []).entries()) {
        const times = frameMidpoints(effect).map((time) => Number(effect.start || 0) + time);
        times.push(Math.max(0, Number(effect.start || 0) - 0.001), Number(effect.start || 0) + effect.duration + 0.001);
        if (effect.fadeIn) times.push(Number(effect.start || 0) + effect.fadeIn / 2);
        if (effect.fadeOut) times.push(Number(effect.start || 0) + effect.duration - effect.fadeOut / 2);
        const visited = new Set();
        for (const zoom of [1, 2]) {
          for (const seconds of times) {
            await setElapsed(frame, seconds, zoom);
            const sample = await inspectFrames(root);
            checkEffects(data, sample, `${width}: effect ${effectIndex} at ${seconds} / ${zoom}×`, scene);
            if (sample.fxFrames[1][effectIndex] !== null) visited.add(sample.fxFrames[1][effectIndex]);
          }
        }
        assert.strictEqual(visited.size, effect.frames.length, `${width}: every effect ${effectIndex} frame inspected`);

        // Find a FX boundary inside one held character pose, then let playback cross it.
        let sum = 0;
        const weights = effect.frames.map((entry, index) => Number(effect.holds && effect.holds[index]) || 1);
        const weightTotal = weights.reduce((total, value) => total + value, 0);
        let heldBoundary = null;
        for (let index = 0; index < weights.length - 1; index += 1) {
          sum += weights[index];
          const boundary = Number(effect.start || 0) + sum / weightTotal * effect.duration;
          if (expectedFrame(data.after, boundary - 0.04) === expectedFrame(data.after, boundary + 0.04)) {
            heldBoundary = boundary;
            break;
          }
        }
        assert.notStrictEqual(heldBoundary, null, 'Study supplies an effect transition during a held character pose');
        await setElapsed(frame, heldBoundary - 0.04, 2);
        const heldStart = await inspectFrames(root);
        const initialPixels = await root.locator('[data-after]').evaluate((canvas) => canvas.toDataURL());
        await root.locator('[data-after]').evaluate((canvas, values) => new Promise((resolve, reject) => {
          const play = canvas.closest('#oracle-heal-review').querySelector('[data-play]');
          play.click();
          const started = performance.now();
          function check() {
            if (JSON.parse(canvas.dataset.fxFrames)[values.index] !== values.oldFrame) {
              play.click();
              return resolve();
            }
            if (performance.now() - started > 1000) return reject(new Error('FX failed to advance while body pose was held'));
            requestAnimationFrame(check);
          }
          check();
        }), { index: effectIndex, oldFrame: heldStart.fxFrames[1][effectIndex] });
        const heldEnd = await inspectFrames(root);
        assert.strictEqual(heldEnd.indices[1], heldStart.indices[1], 'FX transition occurs while the character keeps the same pose');
        assert.notStrictEqual(heldEnd.fxFrames[1][effectIndex], heldStart.fxFrames[1][effectIndex], 'FX animation advances independently of the body pose cache');
        assert.notStrictEqual(await root.locator('[data-after]').evaluate((canvas) => canvas.toDataURL()), initialPixels, 'Independent FX animation repaints actual pixels');
        effectResults.push({ effect: effectIndex, framesInspected: visited.size, heldPoseAnimation: 'passed', opacityAndBounds: 'passed' });
      }

      const rates = [];
      for (const speed of [0.5, 1]) {
        await root.locator('[data-speed]').selectOption(String(speed));
        const start = await inspectFrames(root);
        await root.locator('[data-play]').click();
        await page.waitForTimeout(380);
        await root.locator('[data-play]').click();
        const end = await inspectFrames(root);
        const advanced = end.elapsed - start.elapsed;
        assert(advanced > 0.08, `${width}: resume advances both clocks at speed ${speed}`);
        assert.strictEqual(end.indices[0], expectedFrame(data.before, end.elapsed));
        assert.strictEqual(end.indices[1], expectedFrame(data.after, end.elapsed));
        rates.push(advanced);
      }
      assert(rates[1] / rates[0] > 1.35 && rates[1] / rates[0] < 2.8, 'Normal playback runs approximately twice as fast as half speed');

      const selectedFrame = data.after.frames.length - 1;
      await frame.evaluate((selected) => {
        window.dispatchEvent(new CustomEvent('openai:set_globals', {
          detail: { globals: { widgetState: {
            modelContent: { speed: 0.5, selectedFrame: selected },
            privateContent: { zoom: 1, playing: false }
          } } }
        }));
      }, selectedFrame);
      assert.strictEqual(await root.locator('[data-speed]').inputValue(), '0.5', 'Saved speed restores');
      assert.strictEqual(await root.locator('[data-zoom]').inputValue(), '1', 'Saved zoom restores');
      assert.strictEqual(await root.locator('[data-play]').innerText(), 'Play', 'Saved paused state restores');
      assert.strictEqual((await inspectFrames(root)).indices[1], selectedFrame, 'Saved selected frame restores');
      await root.locator('[data-zoom]').selectOption('2');
      await root.locator('[data-speed]').selectOption('1');
      await setScrub(root, 0);
      const screenshotPose = Math.max(0, Math.min(data.after.frames.length - 1, Number(data.keyPoses[2] && data.keyPoses[2].frame) || 0));
      await setElapsed(frame, midpoints[screenshotPose], 2);
      const layout = await checkLayout(frame, root, width);
      assert.deepStrictEqual(errors, [], `${width}: preview has no browser errors`);
      await page.screenshot({ path: path.join(outputDirectory, `oracle-heal-review-${width}.png`), fullPage: true });
      results.push({ width, sampleResults, persistenceEcho: { before: beforeEcho.elapsed, after: afterEcho.elapsed, result: 'passed' }, inspectedFrames: { before: visitedBefore.size, after: visitedAfter.size }, effects: effectResults, visibleBounds: 'passed', pauseResumeStepSpeedRestore: 'passed', layout, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const report = { url: reviewUrl, viewports: results };
  fs.writeFileSync(path.join(outputDirectory, 'oracle-heal-review-verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
