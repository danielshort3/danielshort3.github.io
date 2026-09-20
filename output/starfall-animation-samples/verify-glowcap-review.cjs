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
    bounds: ['before', 'after'].map((name) => JSON.parse(element.querySelector(`[data-${name}]`).dataset.bounds))
  }));
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
      await page.frameLocator('iframe').locator('#glowcap-hop-review[data-ready="true"]').waitFor({ timeout: 15000 });
      let frame = null;
      for (const candidate of page.frames()) {
        if (candidate.parentFrame() && await candidate.locator('#glowcap-hop-review').count()) frame = candidate;
      }
      assert(frame, 'The rendered visualization must exist inside its iframe');
      const root = frame.locator('#glowcap-hop-review');
      const data = await root.locator('[data-review-data]').evaluate((element) => JSON.parse(element.textContent));
      assert(data.before.frames.length > 0 && data.after.frames.length > 0, 'Both clips must contain real frames');
      assert.strictEqual(await root.locator('[data-before-label]').innerText(), data.beforeLabel || data.before.label || 'Current idle');
      assert.strictEqual(await root.locator('[data-after-label]').innerText(), data.afterLabel || data.after.label || 'Grounded spring study');
      assert.strictEqual(await root.locator('[data-title]').textContent(), typeof data.title === 'string' ? data.title : 'Glowcap · pose study');
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
          beforeStep.bounds.forEach((bounds, index) => assertVisibleBounds(bounds, 320, 280, `${width} / ${zoom}× / ${index === 0 ? 'before' : 'after'} / frame ${beforeStep.indices[index]}`));
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
      const screenshotPose = Math.max(0, Math.min(data.after.frames.length - 1, Number(data.keyPoses[1] && data.keyPoses[1].frame) || 0));
      for (let index = 0; index < screenshotPose; index += 1) await root.locator('[data-step]').click();
      const layout = await checkLayout(frame, root, width);
      assert.deepStrictEqual(errors, [], `${width}: preview has no browser errors`);
      await page.screenshot({ path: path.join(outputDirectory, `glowcap-review-${width}.png`), fullPage: true });
      results.push({ width, sampleResults, persistenceEcho: { before: beforeEcho.elapsed, after: afterEcho.elapsed, result: 'passed' }, inspectedFrames: { before: visitedBefore.size, after: visitedAfter.size }, visibleBounds: 'passed', pauseResumeStepSpeedRestore: 'passed', layout, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const report = { url: reviewUrl, viewports: results };
  fs.writeFileSync(path.join(outputDirectory, 'glowcap-review-verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
