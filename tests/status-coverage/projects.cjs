'use strict';

const assert = require('node:assert/strict');
const { expect } = require('@playwright/test');
const { local, unknown, visible, text, status, visit } = require('./tools.cjs');

function demoPath(item) {
  const slug = String(item.embed?.url || '').replace(/^\//, '').replace(/\.html$/, '');
  assert(/^[a-z0-9-]+-demo$/.test(slug), 'Unexpected local demo route');
  return '/' + slug;
}

async function demo(page, base, item) {
  await visit(page, base, demoPath(item));
  const frame = page.frameLocator('iframe.project-demo-wrapper-iframe');
  await expect(frame.locator('#demo-box, main, .demo-workspace').first()).toBeVisible();
  return frame;
}

function bindDemo(context, base, name, body) {
  return context.route(base + '/api/demos/' + name + '/**', (route) => {
    const action = new URL(route.request().url()).pathname.split('/').at(-1);
    return route.fulfill({ json: ['health', 'warmup', 'status'].includes(action)
      ? { status: 'ok', model_loaded: true, ready: true } : body });
  });
}

async function android({ context, page, base, item }) {
  let feedRequests = 0;
  await context.route(base + '/app-updates/review/latest.json', (route) => {
    feedRequests++;
    return route.fallback();
  });
  await visit(page, base, item.route);
  await expect.poll(() => feedRequests).toBe(1);
  const response = await page.request.get(base + '/app-updates/review/latest.json', { maxRedirects: 0 });
  assert.equal(response.status(), 200);
  const release = (await response.json()).latest;
  assert(/^[a-f0-9]{64}$/.test(release.apk.sha256) && release.apk.size > 0);
  const link = page.locator('[data-android-review-download]');
  await expect(link).toHaveAttribute('href', release.apk.url);
  await expect(link).toHaveAttribute('data-release-version-code', String(release.versionCode));
  await status(page, '[data-android-release-status]', release.versionName.replace(/-debug$/, ''));
  return local('The project loaded its local review release feed and matched the approved APK metadata.');
}

async function names({ page, base, item }) {
  const frame = await demo(page, base, item);
  const list = frame.locator('#name-list');
  await expect(list.locator('li').first()).toBeVisible();
  const before = await list.innerText();
  await frame.locator('[data-sex="M"]').click();
  await status(frame, '#summary-scope', 'boy');
  await expect(list.locator('li').first()).toBeVisible();
  assert.notEqual(await list.innerText(), before);
  return local('Selecting a different name group changed recommendations from the bundled dataset.');
}

async function chatbot({ context, page, base, item }) {
  let requests = 0;
  const answer = 'A fixture answer about Grand Junction.';
  await context.route(base + '/api/chatbot-demo/bedrock/status', (route) =>
    route.fulfill({ json: { online: true, status: 'READY' } }));
  await context.route(base + '/api/chatbot-stream', (route) => {
    if (route.request().method() !== 'POST') return route.fulfill({ json: { ok: true, status: 'ready' } });
    requests++;
    return route.fulfill({ contentType: 'application/x-ndjson', body: [
      { type: 'token', text: answer }, { type: 'done', data: { answer } }
    ].map(JSON.stringify).join('\n') + '\n' });
  });
  const frame = await demo(page, base, item);
  await expect(frame.locator('#backend-select')).toHaveValue('bedrock');
  await frame.locator('#regular-prompt').fill('What can I see in Grand Junction?');
  await frame.locator('#regular-send').click();
  await status(frame, '#regular-messages', answer);
  assert.equal(requests, 1);
  return local('The browser rendered the complete bounded Bedrock response fixture.');
}

async function covid({ page, base, item }) {
  const frame = await demo(page, base, item);
  const states = frame.locator('#state-select');
  await expect(states).toBeEnabled();
  const choice = await states.locator('option').all().then(async (options) => {
    for (const option of options) {
      if ((await option.textContent()).trim() === 'Colorado') return option.getAttribute('value');
    }
    return null;
  });
  assert(choice);
  await states.selectOption(choice);
  await expect(frame.locator('#risk-pill')).not.toHaveAttribute('data-risk', 'unknown');
  await expect(frame.locator('#trend-chart path').first()).toBeVisible();
  return local('Colorado displayed a historical risk and chart from the bundled dashboard data.');
}

function archiveNames(bytes) {
  const index = bytes.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  assert(index >= 0 && index + 22 <= bytes.length);
  const count = bytes.readUInt16LE(index + 10);
  assert(count > 0 && count <= 512);
  const names = [];
  let cursor = bytes.readUInt32LE(index + 16);
  for (let i = 0; i < count; i++) {
    assert(cursor + 46 < index && bytes.readUInt32LE(cursor) === 0x02014b50);
    const size = bytes.readUInt16LE(cursor + 28);
    const end = cursor + 46 + size;
    assert(end <= index);
    names.push(bytes.toString('utf8', cursor + 46, end));
    cursor = end + bytes.readUInt16LE(cursor + 30) + bytes.readUInt16LE(cursor + 32);
  }
  return names;
}

async function linkedFile(page, base, file, maxBytes) {
  const link = page.locator('a[href="' + file + '"]').first();
  await expect(link).toHaveAttribute('href', file);
  const response = await page.request.get(base + file, { timeout: 8000, maxRedirects: 0 });
  assert.equal(response.status(), 200);
  const size = Number(response.headers()['content-length']);
  assert(size > 0 && size <= maxBytes);
  const bytes = await response.body();
  assert.equal(bytes.length, size);
  return bytes;
}

async function delivery({ page, base, item }) {
  await visit(page, base, item.route);
  const pdf = await linkedFile(page, base, '/documents/Project_11.pdf', 5_000_000);
  const workbook = await linkedFile(page, base, '/documents/Project_11.xlsx', 8_000_000);
  assert.equal(pdf.toString('ascii', 0, 5), '%PDF-');
  const names = archiveNames(workbook);
  assert(names.includes('xl/workbook.xml') && names.some((name) => /^xl\/worksheets\/sheet\d+\.xml$/.test(name)));
  return local('Both linked local artifacts opened as the expected PDF and Excel workbook formats.');
}

async function digit({ browser, base, artifactDir }) {
  await require('../site/digit-generator-simple.browser.cjs')({ browser, base, artifactDir });
  return local('The existing browser suite generated and decoded digit images from a model-response fixture.');
}

async function handwriting({ context, page, base, item }) {
  await bindDemo(context, base, 'handwriting', { digit_confidences: [0, 0, 0, 0.97, 0, 0, 0.03, 0, 0, 0] });
  const frame = await demo(page, base, item);
  await visible(frame, '#health-pill[data-state="ok"]');
  await expect(frame.locator('#rate')).toBeDisabled();
  await frame.locator('[data-sample-digit="3"]').click();
  await frame.locator('#rate').click();
  await text(frame, '#result-digit', '3');
  return local('A built-in digit sample returned and displayed the expected fixture classification.');
}

async function nonogram({ context, page, base, item }) {
  const solution = [[1, 0, 1], [0, 1, 0], [1, 0, 1]];
  const steps = solution.flatMap((row, r) => row.map((actual, c) => ({ row: r, col: c, actual, correct: true })));
  const clues = [[1, 1], [1], [1, 1]];
  await bindDemo(context, base, 'nonogram', { grid: 3, row_clues: clues, col_clues: clues, solution, steps });
  const frame = await demo(page, base, item);
  await expect(frame.locator('#action-btn')).toHaveText('Solve With AI');
  await frame.locator('#action-btn').click();
  await status(frame, '#status', 'Puzzle solved.');
  await expect(frame.locator('.cell--filled')).toHaveCount(5);
  const rendered = await frame.locator('[data-row][data-col]').evaluateAll((cells) =>
    cells.map((cell) => Number(cell.classList.contains('cell--filled'))));
  assert.deepEqual(rendered, solution.flat());
  return local('The rendered fixture solution matched the expected rows and columns.');
}

async function pizza({ page, base, item }) {
  const frame = await demo(page, base, item);
  await expect(frame.locator('#estimate-summary')).toHaveAttribute('data-state', 'current');
  const before = await frame.locator('#tip-amount').textContent();
  await frame.locator('#cost').fill('85');
  await frame.locator('#predict').click();
  await expect(frame.locator('#estimate-summary')).toHaveAttribute('data-state', 'current');
  await expect(frame.locator('#tip-amount')).not.toHaveText(before);
  assert.match(await frame.locator('#tip-amount').textContent(), /\d/);
  return local('Changing the order cost produced a different estimate from the bundled model.');
}

async function tableau({ context, page, base, item }) {
  const provider = new URL(item.embed.base);
  assert.equal(provider.origin, 'https://public.tableau.com');
  await context.route((url) => url.origin === provider.origin && url.pathname === provider.pathname, (route) =>
    route.fulfill({ contentType: 'text/html', body: '<main><label>Fixture filter <select><option>All</option>' +
      '<option>Sample</option></select></label><p>Tableau fixture</p></main>' }));
  await visit(page, base, item.route);
  const iframe = page.locator('iframe.project-embed-frame');
  const load = page.getByRole('button', { name: /load dashboard|load tableau|open dashboard/i }).first();
  if (await load.isVisible()) await load.click();
  await iframe.scrollIntoViewIfNeeded();
  const frame = page.frameLocator('iframe.project-embed-frame');
  await visible(frame, 'select');
  await frame.locator('select').selectOption('Sample');
  await expect(frame.locator('select')).toHaveValue('Sample');
  return local('The website loaded the exact external frame and allowed interaction with the provider fixture.');
}

async function retail({ page, base, item }) {
  const frame = await demo(page, base, item);
  await expect(frame.locator('#status-pill')).toHaveAttribute('data-state', /ready|ok/);
  const read = () => frame.locator('#sales-chart').evaluate((node) => {
    const chart = window.Chart?.getChart(node);
    return chart?.data.datasets[0].data || null;
  });
  const before = await read();
  assert(Array.isArray(before) && before.length > 0);
  await frame.locator('[data-sales-metric="online"]').click();
  await expect.poll(read).not.toEqual(before);
  return local('Changing the sales metric changed the Chart.js values from the bundled dataset.');
}

async function shape({ context, page, base, item }) {
  await bindDemo(context, base, 'shape', { class: 'triangle', confidence: 0.96 });
  const frame = await demo(page, base, item);
  await visible(frame, '#health-pill[data-state="ok"]');
  await expect(frame.locator('#classify')).toBeDisabled();
  const bounds = await frame.locator('#pad').boundingBox();
  assert(bounds && bounds.width > 50);
  await page.mouse.move(bounds.x + bounds.width * 0.25, bounds.y + bounds.height * 0.65);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width * 0.60, bounds.y + bounds.height * 0.25, { steps: 10 });
  await page.mouse.up();
  await frame.locator('#classify').click();
  await text(frame, '#result-shape', 'Triangle');
  return local('Drawing on the canvas produced the expected synthetic classification result.');
}

async function sheetMusic({ page, base, item }) {
  await visit(page, base, item.route);
  const zip = await linkedFile(page, base, '/documents/Project_10_pdf.zip', 3_000_000);
  assert.deepEqual(archiveNames(zip).sort(), ['Project_10_GUI.pdf', 'Project_10_UNet.pdf', 'Project_10_VDSR.pdf']);
  return local('The linked local report archive contained all three expected PDF names.');
}

async function sentence({ browser, base, artifactDir }) {
  await require('../site/sentence-retriever.browser.cjs')({ browser, base, artifactDir, browserName: 'chromium' });
  return local('The existing browser suite verified ranked fixture results, failure recovery and cancellation.');
}

async function target({ page, base, item }) {
  const frame = await demo(page, base, item);
  const filter = frame.locator('#filter-location');
  await expect(filter).toBeEnabled();
  await expect(frame.locator('#kpi-incidents')).not.toHaveText('--');
  const before = await frame.locator('#kpi-incidents').textContent();
  const value = await filter.locator('option').nth(1).getAttribute('value');
  assert(value);
  await filter.selectOption(value);
  await expect(frame.locator('#kpi-incidents')).not.toHaveText(before);
  return local('The location filter updated the displayed incident total from the bundled dataset.');
}

async function website({ page, base }) {
  await visit(page, base, '/');
  await expect(page.locator('#main')).toBeVisible();
  await visit(page, base, '/tools');
  await visible(page, 'main a[href="/tools/text-compare"]');
  await visit(page, base, '/search?q=image');
  await expect(page.locator('#search-results a[href="/tools/image-optimizer"]').first()).toBeVisible();
  return local('Home and Tools loaded, and the built search returned the expected tool.');
}

module.exports = {
  androidApp: android, babynames: names, chatbotLora: chatbot, covidAnalysis: covid, deliveryTip: delivery, digitGenerator: digit,
  handwritingRating: handwriting, nonogram, pizza, pizzaDashboard: tableau, retailStore: retail,
  shapeClassifier: shape, sheetMusicUpscale: sheetMusic, smartSentence: sentence, targetEmptyPackage: target,
  ufoDashboard: tableau, website,
  fallback: () => unknown('This project has no deterministic local feature fixture yet.')
};
