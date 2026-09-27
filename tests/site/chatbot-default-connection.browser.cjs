/**
 * Browser check for passive Bedrock connection, failure recovery, and the
 * one-time default migration. No live AWS calls or chat submissions.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chatbot-default-connection-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    const base = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const context = await browser.newContext({ serviceWorkers: 'block' });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    let releaseFirstStatus;
    const firstStatusGate = new Promise(resolve => { releaseFirstStatus = resolve; });
    let delayFirstStatus = true;
    let statusMode = 'ready';
    let statusCalls = 0;
    let mutationCalls = 0;

    await context.route(`${base}/api/chatbot-demo/**`, async route => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      if (request.method() !== 'GET' || pathname !== '/api/chatbot-demo/bedrock/status') {
        mutationCalls += 1;
        await route.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"Unexpected demo mutation"}' });
        return;
      }
      statusCalls += 1;
      if (delayFirstStatus) {
        delayFirstStatus = false;
        await firstStatusGate;
      }
      await route.fulfill({
        status: statusMode === 'ready' ? 200 : 503,
        contentType: 'application/json',
        body: statusMode === 'ready'
          ? '{"status":"READY","online":true,"stage":{"message":"Bedrock ready."}}'
          : '{"error":"Fixture unavailable"}'
      });
    });

    await page.goto(`${base}/chatbot-demo`, { waitUntil: 'domcontentloaded' });
    const frameHandle = await page.locator('iframe.project-demo-wrapper-iframe').elementHandle();
    const frame = await frameHandle.contentFrame();
    const pill = frame.locator('#chat-connection-pill');
    const prompt = frame.locator('#regular-prompt');
    await pill.waitFor({ state: 'visible' });
    assert.match(await pill.innerText(), /Bedrock.*Connecting/i, 'Initial state waits for the actual Bedrock status response.');
    assert.equal(await prompt.isEnabled(), false, 'Chat stays disabled while the connection is unconfirmed.');
    releaseFirstStatus();
    await frame.waitForFunction(() => document.querySelector('#chat-connection-pill')?.dataset.state === 'ok');
    assert.equal(await prompt.isEnabled(), true, 'A successful status check enables chat automatically.');
    assert.equal(await frame.locator('#backend-select').inputValue(), 'bedrock', 'Bedrock is selected by default.');
    assert.equal(await frame.locator('#warmup-button').isVisible(), false, 'A successful automatic check needs no extra connection action.');

    await prompt.fill('Keep this unsent draft.');
    statusMode = 'error';
    await frame.evaluate(() => refreshSelectedBackendStatus('fixture-failure'));
    await frame.waitForFunction(() => document.querySelector('#chat-connection-pill')?.dataset.state === 'err');
    assert.match(await pill.innerText(), /Bedrock.*Unavailable/i, 'Failed checks do not claim the backend is ready.');
    assert.equal(await prompt.inputValue(), 'Keep this unsent draft.', 'Connection failure preserves the draft.');
    assert.equal(await frame.locator('#warmup-button').isVisible(), true, 'Reconnect appears only when the status check fails.');
    statusMode = 'ready';
    await frame.locator('#warmup-button').click();
    await frame.waitForFunction(() => document.querySelector('#chat-connection-pill')?.dataset.state === 'ok');
    assert.equal(await prompt.inputValue(), 'Keep this unsent draft.', 'Explicit reconnect preserves the draft.');

    await page.evaluate(() => {
      localStorage.setItem('demo.endpoint.chatbot.backend', 'qwen-sagemaker');
      localStorage.setItem('demo.endpoint.chatbot.backend.defaultVersion', 'bedrock-default-2026-05-06');
    });
    await page.reload({ waitUntil: 'domcontentloaded' });
    const reloadedFrameHandle = await page.locator('iframe.project-demo-wrapper-iframe').elementHandle();
    const reloadedFrame = await reloadedFrameHandle.contentFrame();
    await reloadedFrame.waitForFunction(() => document.querySelector('#chat-connection-pill')?.dataset.state === 'ok');
    assert.equal(await reloadedFrame.locator('#backend-select').inputValue(), 'bedrock', 'The updated default resets older saved Qwen selections.');
    assert(statusCalls >= 4, 'The page checks Bedrock on initial load, retry, and reload.');
    assert.equal(mutationCalls, 0, 'Loading and checking connectivity never sends a chat or starts SageMaker.');
    await context.close();
    console.log('chatbot-default-connection: automatic status, failure, retry, migration passed');
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
