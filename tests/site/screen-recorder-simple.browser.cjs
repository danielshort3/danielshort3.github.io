/** Recorder workflow with synthetic canvas capture and the real browser encoder. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

function installSyntheticCapture() {
  const test = window.__screenRecorderTest = { cancelNext: true, captureCalls: 0, microphoneCalls: 0, frames: 0, bytes: 0, streams: [], recorders: [], calls: {} };
  Object.defineProperty(navigator.mediaDevices, 'getDisplayMedia', { configurable: true, value: async () => {
    test.captureCalls += 1;
    if (test.cancelNext) {
      test.cancelNext = false;
      throw new DOMException('Synthetic capture permission canceled.', 'NotAllowedError');
    }
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    const stream = canvas.captureStream(20);
    test.streams.push(stream);
    const paint = () => {
      if (!stream.active) return;
      test.frames += 1;
      ctx.fillStyle = '#091f3b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#087f8c';
      ctx.fillRect((test.frames * 6) % 560, 220, 80, 80);
      ctx.fillStyle = '#ffffff';
      ctx.font = '28px sans-serif';
      ctx.fillText('Synthetic recording test', 36, 90);
      ctx.font = '18px sans-serif';
      ctx.fillText('No screen or microphone was captured.', 36, 130);
      requestAnimationFrame(paint);
    };
    paint();
    return stream;
  } });
  Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { configurable: true, value: async () => {
    test.microphoneCalls += 1;
    throw new Error('This test must never request a real microphone.');
  } });
  Object.defineProperty(navigator.mediaDevices, 'enumerateDevices', { configurable: true, value: async () => [] });
  for (const method of ['start', 'pause', 'resume', 'stop']) {
    const original = MediaRecorder.prototype[method];
    MediaRecorder.prototype[method] = function (...args) {
      test.calls[method] = (test.calls[method] || 0) + 1;
      if (method === 'start' && !test.recorders.includes(this)) {
        test.recorders.push(this);
        this.addEventListener('dataavailable', event => { test.bytes += event.data?.size || 0; });
      }
      return original.apply(this, args);
    };
  }
}

async function checkStatusLayout(page, viewport, stage) {
  const layout = await page.evaluate(() => {
    const status = document.querySelector('[data-screenrec="status"]');
    const panel = document.querySelector('[data-screenrec="preview-panel"]');
    const videoStage = document.querySelector('[data-screenrec="stage"]');
    const box = status.getBoundingClientRect();
    const panelBox = panel.getBoundingClientRect();
    return {
      text: status.textContent.trim(), width: box.width, height: box.height,
      left: box.left, right: box.right, bottom: box.bottom,
      panelLeft: panelBox.left, panelRight: panelBox.right,
      videoTop: videoStage.getBoundingClientRect().top,
      overflow: document.documentElement.scrollWidth - innerWidth
    };
  });
  assert(layout.text.length > 0 && layout.text.length <= 70, `${stage}: status above preview should be short`);
  assert(!/Auto-stops|Limit:|Stop capture when you are done/i.test(layout.text), `${stage}: lengthy recording guidance should not occupy the status badge`);
  assert(layout.height <= 60, `${stage}: status should stay compact at ${viewport.width}px`);
  assert(layout.left >= layout.panelLeft - 1 && layout.right <= layout.panelRight + 1, `${stage}: status should stay inside its preview panel`);
  assert(layout.bottom <= layout.videoTop + 1, `${stage}: status should sit above the preview`);
  assert(layout.overflow <= 1, `${stage}: recorder should not overflow horizontally`);
}

async function checkViewport({ browser, base, artifactDir }, viewport) {
  const name = viewport.width <= 320 ? 'mobile-narrow' : viewport.width < 600 ? 'mobile' : 'desktop';
  const context = await browser.newContext({ viewport, reducedMotion: 'reduce', serviceWorkers: 'block', acceptDownloads: true });
  await context.addInitScript(installSyntheticCapture);
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  let stage = 'initial';
  const control = key => page.locator(`[data-screenrec="${key}"]`);
  try {
    await page.goto(base + '/tools/screen-recorder', { waitUntil: 'networkidle' });
    await control('start-capture').waitFor();
    const essential = page.getByRole('button', { name: 'Essential only', exact: true });
    if (await essential.isVisible()) await essential.click();
    assert.equal(await page.locator('[data-screenrec="test-capture"], [data-screenrec="delay-record"], [data-screenrec="countdown"]').count(), 0, 'Removed test/countdown controls should not remain in the page');
    assert(await control('start-record').isDisabled(), 'Recording requires an active capture');
    assert(await control('download-all').isDisabled(), 'Download requires a completed clip');
    assert.equal(await page.evaluate(() => window.__screenRecorderTest.captureCalls), 0, 'Loading the tool must not start screen capture');
    await control('start-capture').scrollIntoViewIfNeeded();
    await checkStatusLayout(page, viewport, stage);

    stage = 'permission cancellation';
    await control('start-capture').click();
    await page.waitForFunction(() => window.__screenRecorderTest.captureCalls === 1 && !document.querySelector('[data-screenrec="start-capture"]').disabled);
    assert.match(await control('status').innerText(), /cancel|blocked|denied|permission/i);
    assert(await control('start-record').isDisabled(), 'Permission cancellation must not enable recording');
    assert(await control('stop-capture').isDisabled(), 'Permission cancellation must not leave an active capture');
    await checkStatusLayout(page, viewport, stage);

    stage = 'capture';
    await control('start-capture').click();
    await page.waitForFunction(() => {
      const video = document.querySelector('[data-screenrec="video"]');
      return video.srcObject instanceof MediaStream && video.videoWidth > 0 && !document.querySelector('[data-screenrec="start-record"]').disabled;
    });
    assert.equal(await page.evaluate(() => window.__screenRecorderTest.captureCalls), 2);
    await checkStatusLayout(page, viewport, stage);
    await control('status').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifactDir, `screen-recorder-${name}-capture.png`) });

    stage = 'recording';
    await control('start-record').click();
    await page.waitForFunction(() => window.__screenRecorderTest.recorders.some(recorder => recorder.state === 'recording'));
    await page.waitForFunction(() => window.__screenRecorderTest.bytes > 0);
    assert(await control('start-record').isDisabled(), 'A second recording cannot start while recording');
    await checkStatusLayout(page, viewport, stage);

    stage = 'pause and resume';
    await control('pause-record').click();
    await page.waitForFunction(() => window.__screenRecorderTest.recorders.every(recorder => recorder.state === 'paused'));
    assert.match(await control('pause-record').innerText(), /Resume/i);
    const pausedTimer = await control('timer').innerText();
    await page.waitForTimeout(350);
    assert.equal(await control('timer').innerText(), pausedTimer, 'Pause freezes the recording timer');
    await checkStatusLayout(page, viewport, 'paused');
    await control('pause-record').click();
    await page.waitForFunction(() => window.__screenRecorderTest.recorders.every(recorder => recorder.state === 'recording'));
    assert.match(await control('pause-record').innerText(), /Pause/i);
    await checkStatusLayout(page, viewport, 'resumed');

    stage = 'stop and download';
    await control('stop-record').click();
    await page.waitForFunction(() => !document.querySelector('[data-screenrec="download-all"]').disabled);
    assert(await control('pause-record').isDisabled(), 'Pause is disabled after finalizing');
    assert(await control('stop-record').isDisabled(), 'Stop recording is disabled after finalizing');
    assert((await control('download-items').innerText()).trim().length > 0, 'Completed download shows its format and size');
    await checkStatusLayout(page, viewport, 'clip ready');
    const downloadEvent = page.waitForEvent('download');
    await control('download-all').click();
    const download = await downloadEvent;
    assert.match(download.suggestedFilename(), /\.(?:webm|mp4|mkv|zip)$/i, 'Download has a recording format filename');
    assert.equal(await download.failure(), null);
    const downloadedFile = await download.path();
    assert(fs.statSync(downloadedFile).size > 100, 'The real encoder must produce a nonempty downloadable recording');
    await control('download-panel').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifactDir, `screen-recorder-${name}-download.png`) });
    await control('stop-capture').click();
    await page.waitForFunction(() => window.__screenRecorderTest.streams.every(stream => stream.getTracks().every(track => track.readyState === 'ended')));
    const capture = await page.evaluate(() => ({ microphoneCalls: window.__screenRecorderTest.microphoneCalls, calls: window.__screenRecorderTest.calls }));
    assert.equal(capture.microphoneCalls, 0, 'No microphone was requested');
    for (const method of ['start', 'pause', 'resume', 'stop']) assert(capture.calls[method] >= 1, `The real MediaRecorder.${method} path was exercised`);
    assert.deepEqual(errors, [], 'Recorder interactions should not produce page errors');
    console.log(`Simple recorder passed: ${name}, canceled permission, synthetic capture, real recording/pause/resume/stop/download, track cleanup.`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `screen-recorder-${name}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${name} ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runScreenRecorderSimpleChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }, { width: 320, height: 740 }]) await checkViewport(options, viewport);
}

module.exports = runScreenRecorderSimpleChecks;

if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'screen-recorder-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runScreenRecorderSimpleChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-screen-recorder') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
