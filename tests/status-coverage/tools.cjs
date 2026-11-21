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

async function showStatusAsAdmin(page) {
  await page.evaluate(() => {
    window.fixtureRole = 'admin';
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
}

function statusGroup(page, name) {
  const group = page.locator('.aws-status__group[data-group="' + name + '"]');
  return {
    toggle: group.getByRole('button', { name, exact: true }),
    rows: group.locator('.aws-status__history-row')
  };
}

function statusRow(page, id) {
  const row = page.locator('.aws-status__history-row[data-item-id="' + id + '"]');
  return { row, dots: row.locator('.aws-status__tick'), panel: row.locator('.aws-status__period-detail'),
    checks: row.locator('.aws-status__check-list li') };
}

async function expectMatrix(page, starts) {
  const grid = page.locator('[data-aws-status-part="history-grid"]');
  await expect(grid).toHaveAttribute('data-granularity', 'six-hour');
  const headers = grid.locator('.aws-status__axis-times .aws-status__axis-time');
  await expect(headers).toHaveCount(6);
  await expect(headers.last().locator('.aws-status__axis-latest')).toHaveText('Latest');
  await expect.poll(() => headers.locator('time').evaluateAll((times) =>
    times.map((time) => time.dateTime))).toEqual(starts);
  const positions = await grid.evaluate((matrix) => {
    const headers = [...matrix.querySelectorAll('.aws-status__axis-time')];
    const rows = [...matrix.querySelectorAll('.aws-status__ticks')];
    const center = (node) => {
      const bounds = node.getBoundingClientRect();
      return bounds.x + bounds.width / 2;
    };
    return rows.map((row) => {
      const buttons = [...row.querySelectorAll('button')];
      return {
        starts: buttons.map((button) => button.dataset.period),
        aligned: buttons.every((button, index) => Math.abs(center(button) - center(headers[index])) < 2),
        latest: buttons.flatMap((button, index) => button.hasAttribute('data-latest') ? [index] : [])
      };
    });
  });
  expect(positions.length).toBeGreaterThan(0);
  for (const position of positions) expect(position).toEqual({ starts, aligned: true, latest: [5] });
}

async function expectGroupMotion(toggle, opening, measure = 'axis') {
  const heights = await toggle.evaluate(async (button, measure) => {
    const grid = button.closest('[data-aws-status-part="history-grid"]');
    const summary = grid.querySelector('.aws-status__group-summary');
    const body = document.getElementById(button.getAttribute('aria-controls'));
    const offset = () => measure === 'body' ? body.getBoundingClientRect().height :
      summary.getBoundingClientRect().top - grid.getBoundingClientRect().top;
    const before = offset();
    button.click();
    const immediate = offset();
    const animations = grid.getAnimations({ subtree: true });
    for (const animation of animations) {
      animation.pause();
      animation.currentTime = animation.effect.getTiming().duration / 2;
    }
    const midway = offset();
    for (const animation of animations) animation.play();
    await Promise.all(animations.map(animation => animation.finished.catch(() => {})));
    return { before, immediate, midway, after: offset() };
  }, measure);
  const direction = opening ? 1 : -1;
  expect(Math.abs(heights.immediate - heights.before)).toBeLessThan(1);
  expect(direction * (heights.midway - heights.before)).toBeGreaterThan(1);
  expect(direction * (heights.after - heights.midway)).toBeGreaterThan(1);
}

async function openStatusGroup(page, name) {
  const group = statusGroup(page, name);
  if (await group.toggle.getAttribute('aria-expanded') !== 'true') await group.toggle.click();
  await expect(group.rows.first()).toBeVisible();
  return group;
}

async function verifyStatusLinks(page, base, item, snapshot, catalog) {
  const services = [
    ['cognito', 'Cognito sign-in', '/tools'],
    ['tracker', 'Tracker database', '/tools/job-application-tracker'],
    ['short-links', 'Short Links', '/tools/short-links'],
    ['contact', 'Contact email', '/contact#contact-modal'],
    ['transcribe', 'File Transcriber (AWS)', '/tools/transcribe'],
    ['chatbot', 'Website chatbot', '/chatbot-demo'],
    ['markdown', 'SageMaker Markdown', 'https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/endpoints/VGJ-Markdown-Async']
  ];
  const groups = { tool: 'Tools', project: 'Projects', game: 'Games' };
  snapshot.connections = services.map(([id, label]) => ({ id, label, days: [], hours: [] }));
  snapshot.homeGpu = { id: 'home-gpu', label: 'Home GPU', days: [], hours: [] };
  snapshot.catalog = catalog.map(({ id, title, category }) => ({
    id, label: title, group: groups[category], state: 'unknown', lastChecked: null, issue: ''
  }));
  await visit(page, base, item.route);
  for (const group of [...Object.values(groups), 'Site & shared services']) {
    await openStatusGroup(page, group);
  }
  const expected = [
    ...services.filter(([id]) => ['cognito', 'contact', 'markdown'].includes(id))
      .map(([, label, route]) => ['Site & shared services', label, route]),
    ...catalog.map(({ category, title, route }) => [groups[category], title, route])
  ];
  const headingLinks = page.locator('.aws-status__history-row h4 a[href]');
  await expect(headingLinks).toHaveCount(expected.length);
  const destinations = await headingLinks.evaluateAll((links) => links.map((link) => [
    link.closest('[data-group]').dataset.group, link.textContent.trim(), link.getAttribute('href')
  ]));
  expect(destinations).toEqual(expect.arrayContaining(expected));

  const project = catalog.find(({ id }) => id === 'project:handwritingRating');
  const row = statusGroup(page, 'Projects').rows.filter({ has: page.getByRole('heading', { name: project.title, exact: true }) });
  const link = row.getByRole('link', { name: project.title, exact: true });
  const hour = row.locator('.aws-status__tick').last();
  await hour.press('Enter');
  await expect(row.locator('.aws-status__period-detail')).toContainText('No report');
  await expect(page).toHaveURL(base + item.route);
  await hour.press('Escape');
  await hour.press('Shift+Tab');
  await expect(link).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(base + project.route);
  await expect(page.getByRole('heading', { level: 1, name: project.title, exact: true })).toBeVisible();

  await visit(page, base, item.route);
  const tools = await openStatusGroup(page, 'Tools');
  await tools.rows.getByRole('link', { name: 'Text Compare', exact: true }).click();
  await expect(page).toHaveURL(base + '/tools/text-compare');
  await expect(page.getByRole('heading', { level: 1, name: 'Text Compare', exact: true })).toBeVisible();
}

function matrixFixture(snapshot) {
  const hour = 3_600_000;
  const first = Date.parse('2026-11-19T18:00:00Z');
  const starts = Array.from({ length: 7 }, (_, index) => new Date(first + index * 6 * hour).toISOString());
  const reading = (start, state, detail = {}) => ({
    start, from: start, to: start, state, issue: '', complete: true,
    observedAt: new Date(Date.parse(start) + 10 * 60_000).toISOString(), ...detail
  });
  const olderHours = Array.from({ length: 6 }, (_, i) => new Date(first + i * hour).toISOString());
  const idle = { completed: 0, failed: 0, inProgress: 0 };
  const transcript = (index, state, canary, issue) => reading(starts[index], state, {
    canary, issue, previousSixHourActivity: idle
  });
  const gpu = (index, state, issue) => reading(starts[index], state, { issue });
  const aws = (index) => reading(starts[index], 'operational', { issue: 'The scheduled check passed.' });
  Object.assign(snapshot, {
    checkedAt: '2026-11-21T06:47:00.000Z', cadenceHours: 6, cadenceStart: starts[1],
    hourlyReadyThrough: starts[6],
    connections: [
      { id: 'cognito', label: 'Cognito sign-in', days: [], hours: [
        ...olderHours.slice(1).map((time) => reading(time, 'operational')), ...[1, 2, 3, 5, 6].map(aws)
      ] },
      ...[['short-links', 'Short Links'], ['tracker', 'Tracker database'], ['chatbot', 'Website chatbot']]
        .map(([id, label]) => ({ id, label, days: [], hours: [
          reading(starts[5], 'operational', { issue: 'Recorded service probe for ' + id + '.' })
        ] })),
      { id: 'transcribe', label: 'File Transcriber (AWS)', days: [], hours: [
        ...olderHours.map((time, index) => reading(time, 'unknown', { transcription: {
          ...idle, inProgress: index < 5 ? 1 : 0
        } })),
        transcript(1, 'degraded', 'timeout', 'The AWS test timed out.'),
        transcript(2, 'degraded', 'failed', 'The AWS test failed.'),
        transcript(3, 'degraded', 'mismatch', 'The transcript differed from the expected phrase.'),
        transcript(5, 'operational', 'passed', 'AWS returned the expected transcript.'),
        transcript(6, 'degraded', 'mismatch', 'The newest transcript did not match.')
      ] }
    ],
    homeGpu: { id: 'home-gpu', label: 'Home GPU', days: [], hours: [
      gpu(3, 'degraded', 'The worker was unavailable.'), gpu(5, 'operational', 'The worker was ready.'),
      gpu(6, 'outage', 'No report arrived from the PC.')
    ] },
    catalog: [
      ['tool:background-remover', 'Background Remover', 'Tools'],
      ['tool:campaign-creative-tracker', 'Campaign Creative Tracker', 'Tools'],
      ['tool:transcribe', 'File Transcriber', 'Tools'], ['tool:short-links', 'Short Links', 'Tools'],
      ['tool:text-compare', 'Text Compare', 'Tools'], ['project:handwritingRating', 'Fixture project', 'Projects'],
      ['tool:job-application-tracker', 'Job Application Tracker', 'Tools'],
      ['project:chatbotLora', 'Fixture chatbot', 'Projects'], ['game:ocean-wave-simulation', 'Fixture game', 'Games']
    ].map(([id, label, group]) => ({ id, label, group, state: 'operational',
      lastChecked: '2026-11-21T06:47:00.000Z', source: 'browser', issue: '' }))
  });

  // Catalog history uses one character per UTC hour even after collection switches to six hours.
  const hourStart = first - 32 * hour;
  const offset = (time) => (Date.parse(time) - hourStart) / hour;
  const observedAt = Array(72).fill(null);
  const results = {};
  for (const { id } of snapshot.catalog) {
    const codes = Array(72).fill('-');
    const reasons = Array(72).fill('-');
    for (const time of [...olderHours, ...starts.slice(1)]) {
      codes[offset(time)] = 'o';
      observedAt[offset(time)] = new Date(Date.parse(time) + 47 * 60_000).toISOString();
    }
    if (id === 'tool:transcribe') {
      codes[offset(starts[3])] = 'd';
      reasons[offset(starts[3])] = 't';
      codes[offset(starts[4])] = '-';
      codes[offset(starts[6])] = 'x';
    }
    if (id === 'project:handwritingRating') codes[offset(olderHours[2])] = 'x';
    results[id] = { hours: codes.join(''), days: '-'.repeat(90),
      details: { hours: reasons.join(''), days: '-'.repeat(90) } };
  }
  snapshot.catalogHistory = { hourStart: new Date(hourStart).toISOString(),
    dayStart: '2026-08-24', observedAt, results };
  return starts;
}

async function unifiedStatusTool({ browser, context, page, base, item, catalog, snapshot, queryCount }) {
  const starts = matrixFixture(snapshot);
  const now = new Date('2026-11-21T11:30:00Z');
  await page.clock.setSystemTime(now);
  const dashboard = page.locator('[data-aws-status-part="dashboard"]');
  const transcriber = statusRow(page, 'tool:transcribe');
  const textCompare = statusRow(page, 'tool:text-compare');
  const openPanels = dashboard.locator('.aws-status__period-detail:visible');
  const check = (name) => transcriber.checks.filter({ has: page.locator('.aws-status__check-name', { hasText: name }) });
  await page.evaluate(() => {
    window.fixtureRole = 'guest';
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
  await expect(page.locator('[data-aws-status-part="access"]')).toHaveText(
    'Sign in with an administrator account to view ' + item.title + '.', { ignoreCase: true });
  await expect(dashboard).toBeHidden();
  assert.equal(queryCount(), 0);
  await showStatusAsAdmin(page);
  await expect(dashboard).toBeVisible();
  await expect(dashboard.getByRole('combobox')).toHaveCount(0);
  await expect(dashboard.getByText('Last 36 hours', { exact: false }).first()).toBeVisible();
  await expect(dashboard.getByText('UTC', { exact: true }).first()).toBeVisible();

  const groupNames = ['Tools', 'Projects', 'Games', 'Site & shared services'];
  await expect(dashboard.locator('.aws-status__group')).toHaveCount(groupNames.length);
  const tools = statusGroup(page, 'Tools');
  const projects = statusGroup(page, 'Projects');
  await expect(tools.toggle).toHaveAttribute('aria-expanded', 'true');
  await expectMatrix(page, starts.slice(0, 6));
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await expectGroupMotion(tools.toggle, false);
  await expect(dashboard.locator('.aws-status__axis')).toBeHidden();
  await expectGroupMotion(tools.toggle, true);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const name of groupNames) await openStatusGroup(page, name);
  await expectMatrix(page, starts.slice(0, 6));
  await expect(openPanels).toHaveCount(0);
  await expect(dashboard.locator('.aws-status__summary-status')).toHaveCount(0);
  await expect(transcriber.dots.last()).toHaveAttribute('data-state', 'operational');
  await expect(transcriber.dots.nth(4)).toHaveAttribute('data-state', 'unknown');
  await expect(statusRow(page, 'project:handwritingRating').dots.first()).toHaveAttribute('data-state', 'outage');
  await expect(statusRow(page, 'cognito').dots.first()).toHaveAttribute('data-state', 'unknown');

  // Activating a dot reveals only that period's measured checks, including legacy job activity.
  await transcriber.dots.first().press('Enter');
  await expect(transcriber.checks).toHaveCount(3);
  await expect(check('AWS transcription activity')).toContainText('0 queued or running at the end of this period.');
  await transcriber.dots.first().press('Escape');
  await expect(openPanels).toHaveCount(0);
  await expect(transcriber.dots.first()).toBeFocused();
  await transcriber.dots.nth(1).click();
  await expect(check('AWS transcription')).toContainText('The AWS test timed out.');
  await transcriber.dots.nth(2).click();
  await expect(check('AWS transcription')).toContainText('The AWS test failed.');
  await transcriber.dots.nth(3).press('Space');
  await expect(transcriber.dots.nth(3)).toHaveAttribute('aria-expanded', 'true');
  await expect(check('Browser file queue')).toContainText(/timed out/i);
  await expect(check('AWS transcription')).toContainText(/transcript differed/i);
  await expect(check('Home GPU readiness')).toContainText('Worker unavailable');
  await transcriber.dots.nth(4).click();
  await expect(transcriber.dots.nth(3)).toHaveAttribute('aria-expanded', 'false');
  await expect(transcriber.checks.locator('.aws-status__check-result')).toHaveText(['No report', 'No report', 'No report']);
  await transcriber.dots.last().click();
  await expect(transcriber.checks.locator('.aws-status__check-name')).toHaveText(
    ['Browser file queue', 'AWS transcription', 'Home GPU readiness']);
  await expect(transcriber.panel.locator('.aws-status__period-heading')).toContainText(/Nov 21.*00:00.*06:00 UTC/);
  await expect(transcriber.dots.last()).toHaveAttribute('aria-controls', await transcriber.panel.getAttribute('id'));
  await expect(check('Browser file queue')).toContainText('Passed');
  await expect(check('AWS transcription')).toContainText('AWS returned the expected transcript.');
  await expect(check('Home GPU readiness')).toContainText(/online|ready/i);
  await expect(openPanels).toHaveCount(1);
  await transcriber.dots.last().click();
  await expect(openPanels).toHaveCount(0);
  await transcriber.dots.last().click();
  await expect(transcriber.panel).toBeVisible();
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await expectGroupMotion(tools.toggle, false, 'body');
  await expect(openPanels).toHaveCount(0);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openStatusGroup(page, 'Tools');

  for (const [id, service] of [
    ['tool:short-links', 'short-links'], ['tool:job-application-tracker', 'tracker'], ['project:chatbotLora', 'chatbot']
  ]) {
    const feature = statusRow(page, id);
    await feature.dots.last().click();
    await expect(feature.checks).toHaveCount(2);
    await expect(feature.panel).toContainText('Recorded service probe for ' + service + '.');
    await expect(openPanels).toHaveCount(1);
  }
  const chatbot = statusRow(page, 'project:chatbotLora');
  await expect(chatbot.panel.getByRole('link')).toHaveAttribute('href', '/chatbot-demo');
  await expect(chatbot.row.locator('h4 a')).toHaveAttribute('href', '/portfolio/chatbotLora');
  await textCompare.dots.last().click();
  await expect(textCompare.checks).toHaveCount(1);
  await expect(textCompare.checks.first()).toContainText('Passed');
  await textCompare.dots.last().press('ArrowLeft');
  await expect(textCompare.dots.nth(4)).toBeFocused();
  await textCompare.dots.nth(4).press('Home');
  await expect(textCompare.dots.first()).toBeFocused();
  await textCompare.dots.first().press('End');
  await expect(textCompare.dots.last()).toBeFocused();
  await textCompare.dots.last().press('Escape');
  await expect(openPanels).toHaveCount(0);

  const phoneContext = await browser.newContext({ viewport: { width: 320, height: 430 }, isMobile: true, hasTouch: true });
  try {
    await fixtureAuth(phoneContext);
    await phoneContext.route(base + '/api/tools/aws-status', (route) => route.fulfill({ json: snapshot }));
    const phone = await phoneContext.newPage();
    await phone.clock.install({ time: now });
    await visit(phone, base, item.route);
    await openStatusGroup(phone, 'Tools');
    const phoneTool = statusRow(phone, 'tool:transcribe');
    const belowFold = statusRow(phone, 'tool:text-compare');
    await phone.getByRole('heading', { level: 1, name: item.title }).scrollIntoViewIfNeeded();
    await belowFold.dots.last().evaluate((dot) => dot.focus({ preventScroll: true }));
    assert(await belowFold.dots.last().evaluate((dot) => dot.getBoundingClientRect().top >= innerHeight));
    await phone.keyboard.press('Enter');
    await belowFold.panel.scrollIntoViewIfNeeded();
    await expect(belowFold.checks.first()).toContainText('Passed');
    await phone.keyboard.press('Escape');
    await expect(belowFold.panel).toBeHidden();

    // Put a fully tappable dot at the lower edge, then use real touch with no implicit Playwright scrolling.
    await phone.setViewportSize({ width: 390, height: 844 });
    const tapPoint = await phoneTool.dots.last().evaluate((dot) => {
      dot.scrollIntoView({ block: 'end', inline: 'nearest', behavior: 'instant' });
      const box = dot.getBoundingClientRect();
      let bottom = innerHeight;
      for (let ancestor = dot.parentElement; ancestor; ancestor = ancestor.parentElement) {
        if (/auto|scroll|hidden|clip/.test(getComputedStyle(ancestor).overflowY)) {
          bottom = Math.min(bottom, ancestor.getBoundingClientRect().bottom);
        }
      }
      const x = box.left + box.width / 2;
      const y = box.top + box.height / 2;
      return { x, y, gap: bottom - box.bottom, hittable: document.elementFromPoint(x, y) === dot };
    });
    expect(tapPoint.hittable).toBe(true);
    expect(tapPoint.gap).toBeGreaterThanOrEqual(-1);
    expect(tapPoint.gap).toBeLessThan(14);
    await phone.touchscreen.tap(tapPoint.x, tapPoint.y);
    await expect(phoneTool.dots.last()).toHaveAttribute('aria-expanded', 'true');
    await expect.poll(() => phoneTool.panel.evaluate((panel) => {
      let top = 0;
      let bottom = innerHeight;
      for (let ancestor = panel.parentElement; ancestor; ancestor = ancestor.parentElement) {
        if (/auto|scroll|hidden|clip/.test(getComputedStyle(ancestor).overflowY)) {
          const bounds = ancestor.getBoundingClientRect();
          top = Math.max(top, bounds.top);
          bottom = Math.min(bottom, bounds.bottom);
        }
      }
      const selected = panel.closest('.aws-status__history-row').querySelector('.aws-status__tick[aria-expanded="true"]');
      const visibleItems = [selected, panel.querySelector('.aws-status__period-heading'),
        panel.querySelector('.aws-status__check-name'), panel.querySelector('.aws-status__check-result')];
      return visibleItems.every((item) => {
        const bounds = item.getBoundingClientRect();
        return bounds.top >= top - 1 && bounds.bottom <= bottom + 1;
      });
    })).toBe(true);
    await phoneTool.dots.last().tap();
    await expect(phoneTool.panel).toBeHidden();

    for (const width of [320, 390]) {
      await phone.setViewportSize({ width, height: 780 });
      await expectMatrix(phone, starts.slice(0, 6));
      const usable = await phoneTool.dots.evaluateAll((buttons) => {
        const bounds = buttons.map((button) => button.getBoundingClientRect());
        return document.documentElement.scrollWidth <= innerWidth + 1 && bounds.every((rect, index) =>
          rect.width >= 24 && rect.height >= 44 && rect.left >= 0 && rect.right <= innerWidth + 1 &&
          (!index || rect.left >= bounds[index - 1].right - 1));
      });
      assert(usable, 'All six individual touch targets should fit the phone viewport');
      await phoneTool.dots.nth(4).tap();
      await expect(phoneTool.checks.locator('.aws-status__check-result')).toHaveText(['No report', 'No report', 'No report']);
      await phoneTool.dots.last().tap();
      await expect(phoneTool.panel).toContainText('AWS returned the expected transcript.');
      expect(await phoneTool.panel.evaluate((node) => {
        const rect = node.getBoundingClientRect();
        return rect.left >= 0 && rect.right <= innerWidth + 1 && node.scrollWidth <= node.clientWidth + 1;
      })).toBe(true);
      await expect(phone).toHaveURL(base + item.route);
    }
    await phoneTool.row.getByRole('link', { name: 'File Transcriber' }).tap();
    await expect(phone).toHaveURL(base + '/tools/transcribe');
  } finally {
    await phoneContext.close();
  }

  // At noon the oldest window leaves the grid and the 06:00 report becomes eligible for display.
  await transcriber.dots.first().click();
  await page.clock.pauseAt(new Date('2026-11-21T11:45:00Z'));
  await page.clock.fastForward(15 * 60_000 - 1);
  await expectMatrix(page, starts.slice(0, 6));
  await page.clock.runFor(1);
  await expectMatrix(page, starts.slice(1));
  await expect(openPanels).toHaveCount(0);
  await expect(transcriber.dots.last()).toHaveAttribute('data-state', 'outage');
  await projects.toggle.click();
  await transcriber.dots.last().click();
  await expect(check('AWS transcription')).toContainText('The newest transcript did not match.');

  const refresh = dashboard.getByRole('button', { name: 'Refresh', exact: true });
  async function refreshSnapshot() {
    await Promise.all([page.waitForResponse((response) => response.url() === base + '/api/tools/aws-status'), refresh.click()]);
    await expect(refresh).toBeEnabled();
  }
  snapshot.checkedAt = '2026-11-21T11:58:00.000Z';
  await refreshSnapshot();
  await expect(dashboard.locator('[data-aws-status-part="freshness"] time')).toHaveAttribute('datetime', snapshot.checkedAt);
  await expect(projects.toggle).toHaveAttribute('aria-expanded', 'false');
  await expect(transcriber.dots.last()).toHaveAttribute('aria-expanded', 'true');
  await expectMatrix(page, starts.slice(1));

  // An unpublished window can already contain raw readings in the response.
  snapshot.hourlyReadyThrough = starts[5];
  await refreshSnapshot();
  await expect(dashboard.locator('.aws-status__tick[data-latest][data-state="unknown"]')).toHaveCount(
    await dashboard.locator('.aws-status__history-row').count());
  await expect(transcriber.checks.locator('.aws-status__check-result')).toHaveText(
    ['Awaiting publication', 'Awaiting publication', 'Awaiting publication']);
  await expect(transcriber.panel).not.toContainText('The newest transcript did not match.');
  snapshot.hourlyReadyThrough = starts[6];
  delete snapshot.catalogHistory;
  await refreshSnapshot();
  await expect(check('Browser file queue')).toContainText('No report');
  await expect(check('AWS transcription')).toContainText('The newest transcript did not match.');
  await expect(check('Home GPU readiness')).toContainText('No report arrived from the PC.');
  delete snapshot.homeGpu;
  await refreshSnapshot();
  await expect(transcriber.checks.locator('.aws-status__check-name')).toHaveText(['Browser file queue', 'AWS transcription']);
  await page.evaluate(() => {
    window.fixtureRole = 'guest';
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
  await expect(dashboard).toBeHidden();
  await expect(openPanels).toHaveCount(0);
  await expect(dashboard.locator('.aws-status__tick')).toHaveCount(0);
  await context.addInitScript(() => { window.fixtureRole = 'admin'; });
  await verifyStatusLinks(page, base, item, snapshot, catalog);
  return local('Verified per-check period details, six closed UTC windows, rollover, refresh, mobile and keyboard input, admin access, and links.');
}

async function statusTool({ browser, context, page, base, item, catalog }) {
  await fixtureAuth(context, 'ordinary');
  // After Mountain clocks fall back, the last two buckets both start at 1 a.m.
  const now = new Date('2026-11-01T08:35:00.000Z');
  const checkedAt = now.toISOString();
  await page.clock.install({ time: now });
  let queries = 0;
  const observation = (state, from, to, issue) => ({ state, from, to, issue });
  const dayStart = '2026-08-04';
  const catalogDays = '-'.repeat(90).split('');
  for (const [date, code] of [['2026-10-20', 'x'], ['2026-10-21', 'd'], ['2026-11-01', 'u']]) {
    catalogDays[(Date.parse(date) - Date.parse(dayStart)) / 86_400_000] = code;
  }
  const snapshot = {
    ok: true, region: 'us-east-2', checkedAt, days: 90,
    connections: [{
      label: 'Fixture connection',
      days: [
        { date: '2026-10-20', ...observation('outage', '2026-10-20T18:03:00Z', '2026-10-20T18:20:00Z', 'Known outage') },
        { date: '2026-10-21', ...observation('degraded', '2026-10-21T16:11:00Z', '2026-10-21T16:20:00Z', 'Known degradation') },
        { date: '2026-11-01', ...observation('operational', checkedAt, checkedAt, '') }
      ],
      hours: [
        { start: '2026-11-01T07:00:00.000Z', ...observation('degraded', '2026-11-01T07:05:00Z', '2026-11-01T07:15:00Z', 'First repeated hour incident') },
        { start: '2026-11-01T08:00:00.000Z', ...observation('operational', '2026-11-01T08:00:00Z', '2026-11-01T09:00:00Z', 'Second repeated hour check'), observedAt: checkedAt }
      ]
    }],
    catalog: [
      { id: 'tool:fixture', label: 'Fixture tool', group: 'Tools', state: 'unknown', lastChecked: null, issue: 'No fixture reading' },
      { id: 'tool:browser-fixture', label: 'Browser-checked tool', group: 'Tools', state: 'operational', lastChecked: checkedAt, issue: '', source: 'browser' },
      { id: 'project:fixture', label: 'Fixture project', group: 'Projects', state: 'unknown', lastChecked: null, issue: '' },
      { id: 'game:fixture', label: 'Fixture game', group: 'Games', state: 'unknown', lastChecked: null, issue: '' }
    ],
    catalogHistory: {
      hourStart: '2026-10-29T09:00:00.000Z',
      dayStart,
      observedAt: [...Array(70).fill(null), '2026-11-01T07:40:00.000Z', checkedAt],
      results: {
        'tool:browser-fixture': { hours: '-'.repeat(70) + 'do', days: catalogDays.join('') }
      }
    }
  };
  await context.route(base + '/api/tools/aws-status', (route) => {
    queries++;
    return route.fulfill({ json: snapshot });
  });
  await visit(page, base, item.route);
  const dashboard = page.locator('[data-aws-status-part="dashboard"]');
  await expect(page.getByRole('heading', { level: 1, name: item.title, exact: true })).toBeVisible();
  await expect(dashboard).toBeHidden();
  assert.equal(queries, 0);
  if (await dashboard.locator('#aws-status-heading').count()) {
    await expect(page.locator('[data-aws-status-part="access"]')).toHaveText(
      'This account does not have access to ' + item.title + '.', { ignoreCase: true });
    return unifiedStatusTool({ browser, context, page, base, item, catalog, snapshot, queryCount: () => queries });
  }
  if (await page.locator('#aws-status-directory-tab').count() === 0) {
    return unknown('The selected build predates the two-tab status page.');
  }
  const directoryTab = page.getByRole('tab', { name: 'Projects & Tools' });
  await showStatusAsAdmin(page);
  await expect(dashboard).toBeVisible();
  const connections = page.locator('#aws-status-connections');
  await expect(connections.locator('[data-aws-status-part="history-grid"]')).toContainText('Fixture connection');
  const granularity = connections.getByRole('combobox', { name: 'Group by' });
  const ticks = connections.locator('.aws-status__tick');
  if (await granularity.count() === 0) {
    await expect(ticks).toHaveCount(90);
    return unknown('The selected build exposes daily history only.');
  }
  const tooltip = page.getByRole('tooltip');
  await expect(granularity).toHaveValue('hourly');
  await expect(ticks).toHaveCount(72);
  await expect(ticks.nth(69)).toHaveAttribute('data-state', 'unknown');
  await ticks.nth(71).hover();
  await expect(tooltip).toContainText(/1:00 AM MST/);
  await expect(tooltip).toContainText('Second repeated hour check');
  await expect(tooltip).toContainText('Observed');
  await ticks.nth(71).press('ArrowLeft');
  await expect(ticks.nth(70)).toBeFocused();
  await expect(tooltip).toContainText(/1:00 AM MDT/);
  await expect(tooltip).toContainText('First repeated hour incident');
  await ticks.nth(70).press('Escape');
  await expect(tooltip).toBeHidden();

  await granularity.selectOption('daily');
  await expect(ticks).toHaveCount(90);
  await expect(page.getByText('Last 90 days', { exact: true })).toBeVisible();
  await expect(ticks.last()).toHaveAttribute('data-state', 'operational');
  await granularity.selectOption('weekly');
  await expect(ticks).toHaveCount(13);
  await expect(ticks.last()).toHaveAttribute('data-state', 'unknown');
  await expect(ticks.nth(11)).toHaveAttribute('data-state', 'outage');
  await ticks.nth(11).hover();
  await expect(tooltip).toContainText('Known outage');
  await expect(tooltip).toContainText(/Oct 19.*Oct 25/);

  await directoryTab.click();
  await expect(granularity).toBeHidden();
  const directory = page.locator('#aws-status-directory');
  const catalogDropdown = directory.getByRole('combobox', { name: 'Group by' });
  const hasCatalogHistory = await directory.locator('#aws-status-catalog-granularity').count() > 0;
  await expect(directory).toContainText('Fixture tool');
  await expect(directory).toContainText('Unknown');
  const currentBadge = directory.getByText('Browser check passed', { exact: true });
  await expect(currentBadge).toBeVisible();
  if (hasCatalogHistory) {
    await expect(currentBadge).toHaveAttribute('title', /Browser check.*Last checked/i);
    for (const heading of ['Tools', 'Projects', 'Games']) {
      await expect(directory.getByRole('heading', { name: heading, exact: true })).toBeVisible();
    }
    await expect(catalogDropdown).toHaveValue('hourly');
    await expect(ticks).toHaveCount(0);
    const row = directory.locator('.aws-status__history-row').filter({
      has: page.getByRole('heading', { name: 'Browser-checked tool', exact: true })
    });
    const catalogTicks = row.locator('.aws-status__tick');
    await expect(catalogTicks).toHaveCount(72);
    await expect(catalogTicks.first()).toHaveAttribute('data-state', 'unknown');
    await catalogTicks.last().hover();
    await expect(tooltip).toContainText(/Observed.*1:35 AM MST/);
    await expect(tooltip).toContainText('The recorded check passed.');
    await catalogTicks.last().press('ArrowLeft');
    await expect(catalogTicks.nth(70)).toBeFocused();
    await expect(tooltip).toContainText(/Observed.*1:40 AM MDT/);
    await expect(tooltip).toContainText('The recorded check had issues.');
    await catalogTicks.nth(70).press('Escape');
    await expect(tooltip).toBeHidden();

    await catalogDropdown.selectOption('daily');
    await expect(catalogTicks).toHaveCount(90);
    await catalogTicks.last().hover();
    await expect(tooltip).toContainText('Coverage was incomplete or inconclusive.');
    await catalogDropdown.selectOption('weekly');
    await expect(catalogTicks).toHaveCount(13);
    await expect(catalogTicks.nth(11)).toHaveAttribute('data-state', 'outage');
    await catalogTicks.nth(11).hover();
    await expect(tooltip).toContainText(/Oct 19.*Oct 25/);
    await expect(tooltip).toContainText(/Issue on Oct 20.*The recorded check failed/);
  }
  assert.equal(queries, 1);
  snapshot.catalog[1].state = 'degraded';
  await page.clock.fastForward(5 * 60 * 1000 + 1000);
  await expect(directory.getByText('Browser check had issues', { exact: true })).toBeVisible();
  await expect(directoryTab).toHaveAttribute('aria-selected', 'true');
  if (hasCatalogHistory) await expect(catalogDropdown).toHaveValue('weekly');
  assert.equal(queries, 2);
  await page.getByRole('tab', { name: 'AWS Connections' }).click();
  await expect(granularity).toHaveValue('weekly');
  if (hasCatalogHistory) {
    await expect(directory.locator('.aws-status__tick')).toHaveCount(0);
    await granularity.selectOption('daily');
    await directoryTab.click();
    await expect(catalogDropdown).toHaveValue('weekly');
    await page.getByRole('tab', { name: 'AWS Connections' }).click();
    await expect(granularity).toHaveValue('daily');
  }

  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  try {
    await fixtureAuth(mobile);
    await mobile.route(base + '/api/tools/aws-status', (route) => route.fulfill({ json: snapshot }));
    const phone = await mobile.newPage();
    await phone.clock.install({ time: now });
    await visit(phone, base, item.route);
    const mobileConnections = phone.locator('#aws-status-connections');
    const mobileDropdown = mobileConnections.getByRole('combobox', { name: 'Group by' });
    await expect(mobileDropdown).toHaveValue('hourly');
    const mobileTicks = mobileConnections.locator('.aws-status__tick');
    await mobileTicks.last().tap();
    await expect(phone.getByRole('tooltip')).toContainText('Second repeated hour check');
    await mobileDropdown.selectOption('weekly');
    await expect(phone.getByRole('tooltip')).toBeHidden();
    await expect(mobileTicks).toHaveCount(13);
    await expect(mobileTicks.last()).toBeInViewport();
    assert(await phone.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    if (hasCatalogHistory) {
      await phone.getByRole('tab', { name: 'Projects & Tools' }).click();
      const mobileCatalog = phone.locator('#aws-status-directory');
      const mobileCatalogDropdown = mobileCatalog.getByRole('combobox', { name: 'Group by' });
      await expect(mobileCatalogDropdown).toHaveValue('hourly');
      const mobileCatalogTicks = mobileCatalog.locator('.aws-status__history-row').filter({
        has: phone.getByRole('heading', { name: 'Browser-checked tool', exact: true })
      }).locator('.aws-status__tick');
      await mobileCatalogTicks.last().tap();
      await expect(phone.getByRole('tooltip')).toContainText(/Observed.*1:35 AM MST/);
      await mobileCatalogDropdown.selectOption('weekly');
      await expect(phone.getByRole('tooltip')).toBeHidden();
      await expect(mobileCatalogTicks).toHaveCount(13);
      await expect(mobileCatalogTicks.last()).toBeInViewport();
      assert(await phone.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    }
  } finally {
    await mobile.close();
  }

  delete snapshot.connections[0].hours;
  delete snapshot.catalogHistory;
  await context.addInitScript(() => { window.fixtureRole = 'admin'; });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(granularity).toHaveValue('hourly');
  await expect(ticks).toHaveCount(72);
  await expect(ticks.last()).toHaveAttribute('data-state', 'unknown');
  await granularity.selectOption('daily');
  await expect(ticks.last()).toHaveAttribute('data-state', 'operational');
  if (hasCatalogHistory) {
    await directoryTab.click();
    await expect(catalogDropdown).toHaveValue('hourly');
    await expect(directory.locator('.aws-status__tick')).toHaveCount(4 * 72);
    await expect(directory.locator('.aws-status__tick[data-state="unknown"]')).toHaveCount(4 * 72);
    await expect(directory.getByText('Browser check had issues', { exact: true })).toBeVisible();
  }
  return local('The fixture verified access, history views, repeated clock hours, touch details and automatic refresh.');
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

async function watermark({ context, page, base, item }) {
  await fixtureAuth(context, 'guest');
  const ticket = 'status-fixture.ticket';
  const launchUrl = base + '/home-tools/watermark/session';
  const apiMethods = [];
  const workerPosts = [];
  const launchResponse = Promise.withResolvers();
  await context.route(base + '/api/tools/watermark-remover', async (route) => {
    apiMethods.push(route.request().method());
    await launchResponse.promise;
    return route.fulfill({ json: { ok: true, launchUrl, ticket } });
  });
  await context.route(launchUrl, (route) => {
    const request = route.request();
    workerPosts.push({
      method: request.method(),
      navigation: request.isNavigationRequest(),
      fields: [...new URLSearchParams(request.postData() || '')]
    });
    return route.fulfill({ contentType: 'text/html', body: '<h1>Fixture worker reached</h1>' });
  });

  await visit(page, base, item.route);
  const open = page.locator('[data-watermark-open]');
  const statusSelector = '[data-watermark-status]';
  await text(page, statusSelector, 'Sign in with an administrator account to open Watermark Remover.');
  await expect(open).toBeHidden();
  await expect(open).toBeDisabled();

  await page.evaluate(() => {
    window.fixtureRole = 'member';
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
  await text(page, statusSelector, 'This account does not have administrator access.');
  await expect(open).toBeHidden();
  await expect(open).toBeDisabled();
  assert.deepEqual(apiMethods, []);
  assert.deepEqual(workerPosts, []);

  await showStatusAsAdmin(page);
  await expect(open).toBeVisible();
  await expect(open).toBeEnabled();
  try {
    await open.dblclick();
    await expect(open).toBeDisabled();
  } finally {
    launchResponse.resolve();
  }
  await page.waitForURL(launchUrl);
  await expect(page.getByRole('heading', { name: 'Fixture worker reached' })).toBeVisible();
  assert.deepEqual(apiMethods, ['POST']);
  assert.deepEqual(workerPosts, [{ method: 'POST', navigation: true, fields: [['ticket', ticket]] }]);
  return local('The portal withheld access from guests and members and posted one admin launch ticket to the fixture worker.');
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
  'watermark-remover': watermark,
  'word-frequency': words
};

module.exports = { tools, local, unknown, visible, text, status, visit, draw };
