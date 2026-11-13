'use strict';

const assert = require('node:assert/strict');
const sharp = require('sharp');
const { expect } = require('@playwright/test');

const local = (reason) => ({ status: 'pass', reason });
const unknown = (reason) => ({ status: 'unknown', reason });
const visible = (page, selector) => expect(page.locator(selector)).toBeVisible();
const text = (page, selector, value) => expect(page.locator(selector)).toHaveText(value);
const status = (page, selector, value) => expect(page.locator(selector)).toContainText(value);

async function dismissConsent(page) {
  const reject = page.locator('#pcz-reject');
  if (await reject.isVisible()) await reject.evaluate((button) => button.click());
}

async function visit(page, base, route) {
  const response = await page.goto(base + route, { waitUntil: 'domcontentloaded' });
  assert.equal(response?.status(), 200);
  await dismissConsent(page);
}

async function download(page, selector) {
  const pending = page.waitForEvent('download');
  await page.locator(selector).first().click();
  return (await pending).path();
}

async function draw(page, selector) {
  const canvas = page.locator(selector);
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  assert(box && box.width > 50 && box.height > 50);
  const point = (x, y) => [box.x + x * box.width, box.y + y * box.height];
  await page.mouse.move(...point(0.25, 0.7));
  await page.mouse.down();
  await page.mouse.move(...point(0.5, 0.25), { steps: 8 });
  await page.mouse.move(...point(0.75, 0.7), { steps: 8 });
  await page.mouse.move(...point(0.25, 0.7), { steps: 8 });
  await page.mouse.up();
}

function fixtureAuth(context, mode = 'admin') {
  return context.addInitScript((selected) => {
    const getMode = () => window.fixtureRole || selected;
    const auth = { expiresAt: 2_000_000_000_000, idToken: 'fixture-token' };
    const api = {
      getAuth: () => getMode() === 'guest' ? null : auth,
      authIsValid: (value) => Boolean(value),
      getUser: () => ({ sub: 'fixture-account', name: 'Fixture Reviewer', email: 'fixture@example.test' }),
      isAdmin: () => getMode() === 'admin',
      handleRedirect: async () => ({ handled: false }),
      ensureFreshAuth: async () => getMode() === 'guest' ? null : auth,
      getConfig: () => ({}),
      signIn: async () => {},
      signOut: async () => {},
      fetchWithAuth: (url, options) => fetch(url, options)
    };
    Object.defineProperty(window, 'ToolsAuth', { configurable: true, get: () => api, set() {} });
  }, mode);
}

async function statusTool({ context, page, base, item }) {
  await fixtureAuth(context, 'ordinary');
  await page.clock.install();
  let queries = 0;
  const checkedAt = new Date().toISOString();
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Denver', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
  const snapshot = {
    ok: true, region: 'us-east-2', checkedAt, days: 90,
    connections: [{ label: 'Fixture connection', days: [{ date, state: 'operational', from: checkedAt, to: checkedAt, issue: '' }] }],
    catalog: [
      { label: 'Fixture tool', group: 'Tools', state: 'unknown', lastChecked: null, issue: 'No fixture reading' },
      { label: 'Browser-checked tool', group: 'Tools', state: 'operational', lastChecked: checkedAt, issue: '', source: 'browser' }
    ]
  };
  await context.route(base + '/api/tools/aws-status', (route) => {
    queries++;
    return route.fulfill({ json: snapshot });
  });
  await visit(page, base, item.route);
  const dashboard = page.locator('[data-aws-status-part="dashboard"]');
  await expect(dashboard).toBeHidden();
  assert.equal(queries, 0);
  const tabs = page.locator('#aws-status-directory-tab');
  if (await tabs.count() === 0) return unknown('The selected build predates the two-tab status page.');
  await page.evaluate(() => {
    window.fixtureRole = 'admin';
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
  await expect(dashboard).toBeVisible();
  await expect(page.locator('[data-aws-status-part="history-grid"]')).toContainText('Fixture connection');
  await expect(page.locator('.aws-status__tick')).toHaveCount(90);
  await page.getByRole('tab', { name: 'Projects & Tools' }).click();
  const directory = page.locator('#aws-status-directory');
  await expect(directory).toContainText('Fixture tool');
  await expect(directory).toContainText('Unknown');
  await expect(directory.getByText('Browser check passed', { exact: true })).toBeVisible();
  assert.equal(queries, 1);
  snapshot.catalog[1].state = 'degraded';
  await page.clock.fastForward(5 * 60 * 1000 + 1000);
  await expect(directory.getByText('Browser check had issues', { exact: true })).toBeVisible();
  await expect(tabs).toHaveAttribute('aria-selected', 'true');
  assert.equal(queries, 2);
  return local('The fixture verified the ordinary-account gate, history, and a timed refresh that retained the catalog tab.');
}

async function background({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#bgtool-method').selectOption('colorkey');
  const rgba = Buffer.alloc(120 * 80 * 4);
  for (let i = 0; i < rgba.length; i += 4) {
    const x = (i / 4) % 120;
    const y = Math.floor(i / 480);
    rgba.set(x > 40 && x < 80 && y > 20 && y < 60 ? [220, 0, 0, 255] : [255, 255, 255, 255], i);
  }
  const png = await sharp(rgba, { raw: { width: 120, height: 80, channels: 4 } }).png().toBuffer();
  await page.locator('#bgtool-file').setInputFiles({ name: 'fixture.png', mimeType: 'image/png', buffer: png });
  await expect(page.locator('#bgtool-download-selected')).toBeEnabled();
  const output = await download(page, '#bgtool-download-selected');
  const { data, info } = await sharp(output).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.equal(info.width, 120);
  assert.equal(data[3], 0);
  assert(data[(40 * 120 + 60) * 4 + 3] > 200);
  return local('The local color-removal path exported an image with transparent background and opaque subject.');
}

async function creative({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('[data-ctc-action="new-campaign"]').click();
  await page.locator('[name="campaignName"]').fill('Fixture launch');
  await page.locator('[data-ctc-form="campaign"] button[type="submit"]').click();
  await status(page, '.ctc-app-title', 'Fixture launch');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await status(page, '.ctc-app-title', 'Fixture launch');
  return local('The created campaign survived a browser reload in isolated storage.');
}

async function ga4({ context, page, base, item }) {
  let reports = 0;
  await context.route(base + '/api/ga4/report', (route) => {
    reports++;
    return route.fulfill({ json: { ok: true, rows: [{
      pageLocation: 'https://example.test/?utm_source=fixture&utm_medium=email&utm_campaign=launch',
      sessions: 12, totalUsers: 10, eventCount: 20
    }], returnedRows: 1, rowCount: 1 } });
  });
  await visit(page, base, item.route);
  await page.locator('#ga4-property-id').fill('123456');
  await page.locator('#ga4-start').fill('2026-08-01');
  await page.locator('#ga4-end').fill('2026-08-31');
  await page.locator('#ga4-admin-token').fill('fixture-access');
  await page.locator('[data-ga4="auth"] button[type="submit"]').click();
  await page.locator('[data-ga4="run"]').click();
  await status(page, '[data-ga4="utm-output"]', 'launch');
  await expect(page.locator('[data-ga4="download-grouped"]')).toBeEnabled();
  assert.equal(reports, 1);
  return local('The fixture report rendered the expected campaign and enabled export.');
}

async function optimize({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#imgopt-format').selectOption('image/png');
  await page.locator('#imgopt-resize-mode').selectOption('maxWidth');
  await page.locator('#imgopt-width').fill('480');
  await page.locator('#imgopt-sample').click();
  await status(page, '#imgopt-status', 'image is ready');
  await page.locator('#imgopt-process').click();
  const result = await sharp(await download(page, '#imgopt-results a[download]')).metadata();
  assert.equal(result.format, 'png');
  assert.equal(result.width, 480);
  assert.equal(result.height, 320);
  return local('The built-in sample exported as a valid 480 × 320 PNG.');
}

async function tracker({ context, page, base, item }) {
  await fixtureAuth(context, 'ordinary');
  let saved = null;
  let submissions = 0;
  const calendarDate = '2026-11-11';
  await context.route(base + '/api/job-tracker/**', async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname.slice('/api/job-tracker'.length);
    let payload = {};
    if (pathname === '/api/applications' && request.method() === 'POST') {
      const draft = request.postDataJSON();
      submissions++;
      saved = { ...draft, applicationId: 'fixture-app', status: 'Applied', entryType: 'application', attachments: [],
        createdAt: calendarDate + 'T12:00:00Z', updatedAt: calendarDate + 'T12:00:00Z' };
      payload = saved;
    } else if (pathname === '/api/applications') {
      payload = { items: saved ? [saved] : [] };
    } else if (pathname.startsWith('/api/analytics/')) {
      payload = { items: [], series: [], statusBreakdown: {}, applications: saved ? [saved] : [] };
    } else {
      payload = { items: [] };
    }
    return route.fulfill({ json: payload });
  });
  await visit(page, base, item.route);
  await page.locator('[data-jobtrack-tab="entry"]').click();
  await page.locator('#jobtrack-company').fill('Fixture Company');
  await page.locator('#jobtrack-title').fill('Fixture Analyst');
  await page.locator('#jobtrack-date').fill(calendarDate);
  await page.locator('[data-jobtrack="entry-submit"]').click();
  await expect.poll(() => submissions).toBe(1);
  assert.equal(saved.appliedDate, calendarDate);
  await page.reload({ waitUntil: 'domcontentloaded' });
  const applications = page.getByRole('button', { name: /^Applications/ }).first();
  if (await applications.isVisible()) await applications.click();
  await expect(page.locator('[data-jobtrack="entry-list"]')).toContainText('Fixture Company');
  return local('The browser saved the exact calendar date and restored the synthetic application from the API fixture.');
}

async function cleaner({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#nbsp-input').fill('A\u00a0B café');
  await page.locator('#nbsp-form button[type="submit"]').click();
  await expect(page.locator('#nbsp-output')).toHaveValue('A B café');
  await status(page, '#nbsp-summary', '1 hard space replaced');
  return local('The browser produced the exact cleaned text and replacement count.');
}

async function oxford({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#oxford-example').click();
  assert((await page.locator('#oxford-text').inputValue()).length > 10);
  if ((await page.locator('[data-oxford-output] .oxford-mark').count()) === 0) {
    await page.locator('[data-oxford-form] button[type="submit"]').click();
  }
  await expect(page.locator('[data-oxford-output] .oxford-mark').first()).toBeVisible();
  return local('The built-in writing example produced a highlighted list suggestion.');
}

async function pov({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#povcheck-text').fill('I reviewed the draft. You can share it. They will review the final version.');
  await page.locator('#povcheck-form button[type="submit"]').click();
  await text(page, '#povcheck-first-count', '1');
  await text(page, '#povcheck-second-count', '1');
  return local('The fixed passage displayed one first-person and one second-person pronoun.');
}

async function qr({ page, base, item }) {
  const payload = 'https://example.test/fixture';
  await visit(page, base, item.route);
  await page.locator('#qrtool-data').fill(payload);
  await expect(page.locator('#qrtool-stage')).toHaveAttribute('data-preview-state', 'ready');
  const image = await sharp(await download(page, '#qrtool-download-png')).metadata();
  assert.equal(image.format, 'png');
  assert(image.width >= 128 && image.width === image.height);
  return local('The fixture URL generated a square PNG. Exact independent payload decoding was outside this run.');
}

async function shortLinks({ context, page, base, item }) {
  await context.route(base + '/api/short-links/**', (route) => {
    assert.equal(route.request().method(), 'GET');
    return route.fulfill({ json: { ok: true, links: [], sets: [], recentSessions: [], tools: [] } });
  });
  await visit(page, base, item.route);
  await visible(page, '[data-shortlinks="signed-out"]');
  await page.locator('[data-shortlinks="connect-access"]').click();
  await page.locator('[data-shortlinks="token"]').fill('fixture-workspace-key');
  await page.locator('[data-shortlinks="auth"] button[type="submit"]').click();
  await visible(page, '[data-shortlinks="workspace"]');
  await page.locator('.shortlinks-account-menu > summary').click();
  await page.locator('[data-shortlinks-mode="qr"]').click();
  await visible(page, '#shortlinks-mode-qr');
  return local('A read-only workspace fixture verified sign-in gating and the QR view.');
}

async function compare({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#textcompare-original').fill('The draft is ready.');
  await page.locator('#textcompare-revised').fill('The final is ready.');
  await page.locator('#textcompare-form button[type="submit"]').click();
  await status(page, '#textcompare-output del', 'draft');
  await status(page, '#textcompare-output ins', 'final');
  return local('The rendered comparison identified the expected removed and added words.');
}

async function utm({ context, page, base, item }) {
  await context.addInitScript(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async (value) => { window.fixtureClipboard = value; }
    } });
  });
  await visit(page, base, item.route);
  await page.getByPlaceholder('https://example.com/landing').fill('https://example.test/landing');
  await page.getByPlaceholder('google', { exact: true }).fill('newsletter');
  await page.getByPlaceholder('cpc', { exact: true }).fill('email');
  await page.getByPlaceholder('spring_sale', { exact: true }).fill('fixture');
  await page.getByRole('button', { name: 'Generate links', exact: true }).click();
  await status(page, '.utmtool-workspace-status', 'Complete');
  await page.getByRole('button', { name: 'Copy all', exact: true }).click();
  const copied = new URL(await page.evaluate(() => window.fixtureClipboard));
  assert.equal(copied.searchParams.get('utm_source'), 'newsletter');
  assert.equal(copied.searchParams.get('utm_campaign'), 'fixture');
  await expect(page.getByRole('button', { name: 'Export CSV', exact: true })).toBeEnabled();
  return local('The generated URL included the expected UTM fields and enabled CSV export.');
}

async function words({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#wordfreq-text').fill('Handoff handoff handoff. Support support.');
  await page.locator('#wordfreq-form button[type="submit"]').click();
  await page.locator('.wordfreq-term-btn').first().click();
  await status(page, '#wordfreq-occurrence-summary', '3 matches');
  return local('The repeated-word fixture reported exactly three matches for the selected term.');
}

function audio() {
  const bytes = Buffer.alloc(44 + 16_000);
  bytes.write('RIFF');
  bytes.writeUInt32LE(bytes.length - 8, 4);
  bytes.write('WAVEfmt ', 8);
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(8000, 24);
  bytes.writeUInt32LE(16000, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write('data', 36);
  bytes.writeUInt32LE(bytes.length - 44, 40);
  return bytes;
}

async function transcribe({ page, base, item }) {
  await visit(page, base, item.route);
  await page.locator('#transcribe-files').setInputFiles({
    name: 'fixture-audio.wav', mimeType: 'audio/wav', buffer: audio()
  });
  await status(page, '#transcribe-file-rows', 'fixture-audio.wav');
  return local('The synthetic audio appeared in the upload queue. AWS transcription was not run.');
}

function priorSuite(relative, reason) {
  return async ({ browser, base, artifactDir }) => {
    await require('../site/' + relative)({ browser, base, artifactDir });
    return local(reason);
  };
}

const tools = {
  'aws-status': statusTool,
  'background-remover': background,
  'campaign-creative-tracker': creative,
  'ga4-utm-performance': ga4,
  'image-optimizer': optimize,
  'job-application-tracker': tracker,
  'nbsp-cleaner': cleaner,
  'oxford-comma-checker': oxford,
  'point-of-view-checker': pov,
  'qr-code-generator': qr,
  'screen-recorder': priorSuite('screen-recorder-simple.browser.cjs', 'The existing suite recorded and downloaded a synthetic browser capture.'),
  'short-links': shortLinks,
  'text-compare': compare,
  transcribe,
  'utm-batch-builder': utm,
  'word-frequency': words
};

module.exports = { tools, local, unknown, visible, text, status, visit, draw };
