/**
 * Exercises the inline-greeting chatbot as a visitor would. AWS replies are
 * intercepted so the check sends no live messages.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

async function checkViewport(browser, base, width, height, artifactDir) {
  const context = await browser.newContext({
    viewport: { width, height },
    reducedMotion: 'reduce',
    serviceWorkers: 'block',
    isMobile: width < 600,
    hasTouch: width < 600
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const prompts = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.route(`${base}/api/chatbot-demo/**`, async route => {
    if (new URL(route.request().url()).pathname !== '/api/chatbot-demo/bedrock/status') {
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"Unexpected demo request"}' });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: '{"status":"READY","online":true,"stage":{"message":"Fixture ready."}}'
    });
  });
  await context.route(`${base}/api/chatbot-stream`, async route => {
    prompts.push(JSON.parse(route.request().postData() || '{}').prompt);
    await route.fulfill({
      status: 200,
      contentType: 'application/x-ndjson',
      body: [
        JSON.stringify({ type: 'token', text: 'Visit downtown and the riverfront.' }),
        JSON.stringify({ type: 'done', data: { answer: 'Visit downtown and the riverfront.', source_details: [] } })
      ].join('\n') + '\n'
    });
  });

  try {
    const response = await page.goto(`${base}/chatbot-demo`, { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    const frame = await (await page.locator('iframe.project-demo-wrapper-iframe').elementHandle()).contentFrame();
    await frame.waitForFunction(() => document.querySelector('#chat-connection-pill')?.dataset.state === 'ok');
    assert.equal(await frame.locator('.conversation-header #chat-connection-pill').count(), 1, 'Connection state appears once in the conversation header.');
    assert.equal(await frame.locator('#warmup-button').isVisible(), false, 'Reconnect stays hidden when Bedrock is ready.');
    assert.equal(await frame.locator('.guided-start').count(), 0, 'There is no separate starter pane.');
    assert.equal(await frame.locator('#regular-messages .chat-greeting').isVisible(), true, 'The opening greeting is inside the conversation.');
    assert.equal(await frame.locator('#regular-messages [data-suggestion-prompt]').count(), 3, 'Three inline suggestions are available.');
    assert.equal(await frame.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), true, 'Chatbot fits the narrow viewport.');
    await page.screenshot({ path: path.join(artifactDir, `chatbot-inline-${width}-ready.png`) });
    if (width < 600) {
      const dockLocator = page.locator('.mobile-site-dock, .personal-accordion__rails').first();
      for (const control of [frame.locator('#regular-messages [data-suggestion-prompt]').last(), frame.locator('#regular-send')]) {
        await control.scrollIntoViewIfNeeded();
        const box = await control.boundingBox();
        const dock = await dockLocator.isVisible() ? await dockLocator.boundingBox() : null;
        assert(box && box.y + box.height <= (dock?.y ?? height) + 1, 'All greeting choices and the composer remain reachable above the mobile dock.');
      }
    }

    await frame.locator('#regular-messages [data-suggestion-prompt]').first().click();
    await frame.locator('#regular-messages .message.assistant').getByText('Visit downtown and the riverfront.').waitFor();
    assert.deepEqual(prompts, ['Plan my first day in Grand Junction'], 'The inline suggestion sends the intended prompt once.');
    assert.match(await frame.locator('#regular-messages .message.user').innerText(), /Plan my first day in Grand Junction/);
    assert.equal(await frame.locator('.chat-greeting').count(), 0, 'The greeting and its choices go away after selection in both views.');
    if (width < 600) {
      await frame.locator('#regular-send').scrollIntoViewIfNeeded();
      const send = await frame.locator('#regular-send').boundingBox();
      const dockLocator = page.locator('.mobile-site-dock, .personal-accordion__rails').first();
      const dock = await dockLocator.isVisible() ? await dockLocator.boundingBox() : null;
      assert(send && send.y + send.height <= (dock?.y ?? height) + 1, 'The composer can be scrolled above the mobile dock.');
    }

    await frame.locator('#regular-prompt').fill('Keep a draft across views.');
    await frame.locator('#chat-settings > summary').click();
    await frame.locator('#popup-view-button').click();
    await frame.locator('#popup-launcher').click();
    assert.equal(await frame.locator('#popup-prompt').inputValue(), 'Keep a draft across views.');
    assert.match(await frame.locator('#popup-messages').innerText(), /Visit downtown and the riverfront/);
    await frame.locator('#popup-close').click();
    await frame.locator('#chat-settings > summary').click();
    await frame.locator('#regular-view-button').click();
    assert.equal(await frame.locator('#regular-prompt').inputValue(), 'Keep a draft across views.');
    await frame.locator('#chat-settings > summary').click();
    await frame.locator('#regular-clear').click();
    assert.equal(await frame.locator('#regular-messages .chat-greeting').isVisible(), true, 'Clearing restores the inline greeting.');
    assert.equal(await frame.locator('#popup-messages [data-suggestion-prompt]').count(), 3, 'The restored choices also appear in the pop-up.');
    await frame.locator('#regular-prompt').fill('What is a good riverfront walk?');
    await frame.locator('#regular-send').click();
    await frame.locator('#regular-messages .message.assistant').getByText('Visit downtown and the riverfront.').waitFor();
    assert.deepEqual(prompts, ['Plan my first day in Grand Junction', 'What is a good riverfront walk?'], 'A typed first question sends once.');
    assert.equal(await frame.locator('.chat-greeting').count(), 0, 'Typing a first question also removes the greeting and choices.');
    await page.screenshot({ path: path.join(artifactDir, `chatbot-inline-${width}-conversation.png`) });
    assert.deepEqual(errors, [], 'The inline conversation has no page errors.');
    console.log(`chatbot-inline-greeting: ${width}×${height} passed`);
  } finally {
    await context.close();
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chatbot-guided-start-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    const artifactDir = process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-chatbot-guided-start');
    fs.mkdirSync(artifactDir, { recursive: true });
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const [width, height] of [[1440, 900], [390, 844], [320, 740]]) {
      await checkViewport(browser, base, width, height, artifactDir);
    }
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(envDir, { recursive: true, force: true });
  }
}

if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
