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
    summary: group.locator('.aws-status__group-summary .aws-status__tick'),
    rows: group.locator('.aws-status__history-row')
  };
}

async function expectStatusAxis(dashboard, expectedBands) {
  const axis = dashboard.locator('.aws-status__axis');
  await expect(axis).toHaveCount(1);
  const actual = await axis.evaluate((element) => {
    const grid = element.closest('[data-aws-status-part="history-grid"]');
    const rows = [...grid.querySelectorAll('.aws-status__ticks')];
    const bands = [...element.querySelectorAll('.aws-status__axis-band')];
    let start = 0;
    const aligned = bands.every((band) => {
      const count = Number(band.dataset.periodCount);
      const rect = band.getBoundingClientRect();
      const matches = rows.every((row) => {
        const first = row.children[start];
        const last = row.children[start + count - 1];
        return first && last &&
          Math.abs(rect.left - first.getBoundingClientRect().left) < 1 &&
          Math.abs(rect.right - last.getBoundingClientRect().right) < 1;
      });
      start += count;
      return matches && Boolean(band.querySelector('.aws-status__axis-label')?.textContent.trim());
    });
    return {
      bands: bands.map((band) => [band.dataset.start, Number(band.dataset.periodCount)]),
      aligned,
      coversEveryPeriod: rows.length > 0 && rows.every((row) => row.children.length === start)
    };
  });
  expect(actual).toEqual({ bands: expectedBands, aligned: true, coversEveryPeriod: true });
}

async function unifiedStatusTool({ browser, context, page, base, item, snapshot, queryCount }) {
  const period = page.getByRole('combobox', { name: 'Group by' });
  const tooltip = page.getByRole('tooltip');
  const groups = ['AWS Connections', 'Local services', 'Tools', 'Projects', 'Games'].map((name) => statusGroup(page, name));
  const [aws, localServices, tools, projects, games] = groups;
  const now = new Date('2026-11-01T09:05:00.000Z');
  await page.clock.setSystemTime(now);
  const latestShown = '2026-11-01T07:00:00.000Z';
  const nextHour = '2026-11-01T08:00:00.000Z';
  snapshot.hourlyReadyThrough = latestShown;
  snapshot.connections.push({
    id: 'transcribe',
    label: 'File Transcriber (AWS)',
    days: [],
    hours: [latestShown, nextHour].map((start) => ({
      start, from: start, to: start, state: 'unknown', issue: 'No AWS transcription activity recorded.'
    }))
  });
  snapshot.homeGpu = {
    id: 'home-gpu',
    label: 'Home GPU',
    days: [
      ['2026-10-30', 'outage', 'No report arrived from the PC.'],
      ['2026-10-31', 'operational', 'The transcription worker is ready.']
    ].map(([date, state, issue]) => ({ date, from: date + 'T12:00:00Z', to: date + 'T12:00:00Z', state, issue })),
    hours: [
      ['2026-11-01T06:00:00.000Z', 'degraded', 'The PC reported that the transcription worker is unavailable.'],
      [latestShown, 'outage', 'No report arrived from the PC.'],
      [nextHour, 'operational', 'The transcription worker is ready.']
    ].map(([start, state, issue]) => ({ start, from: start, to: start, state, issue, complete: true,
      observedAt: start === nextHour ? undefined : new Date(Date.parse(start) + 10 * 60 * 1000).toISOString() }))
  };
  for (const [id, hours] of [['tool:fixture', 'uu'], ['project:fixture', 'od'], ['game:fixture', 'oo']]) {
    snapshot.catalogHistory.results[id] = { hours: '-'.repeat(70) + hours, days: '-'.repeat(90) };
  }
  const newestCatalogHour = Object.values(snapshot.catalogHistory.results).map((result) => {
    const latest = result.hours.at(-1);
    result.hours = result.hours.slice(0, -1) + '-';
    return { result, latest };
  });

  await showStatusAsAdmin(page);
  const dashboard = page.locator('[data-aws-status-part="dashboard"]');
  await expect(dashboard.getByRole('heading', { name: 'System status', exact: true })).toBeVisible();
  await expect(dashboard.getByRole('tab')).toHaveCount(0);
  await expect(period).toHaveCount(1);
  await expect(period.locator('option')).toHaveText(['Hourly', 'Daily', 'Weekly']);
  await expect(period).toHaveValue('hourly');
  for (const group of groups) {
    await expect(group.toggle).toHaveAttribute('aria-expanded', 'false');
    await expect(group.rows).toHaveCount(0);
    await expect(group.summary).toHaveCount(70);
    await expect(group.summary.last()).toBeVisible();
    await group.toggle.click();
    await expect(group.toggle).toHaveAttribute('aria-expanded', 'true');
  }

  const ticks = (group, name) => group.rows.filter({ hasText: name }).locator('.aws-status__tick');
  const awsTicks = ticks(aws, 'Fixture connection');
  const transcribeTicks = ticks(aws, 'File Transcriber (AWS)');
  const homeGpuTicks = ticks(localServices, 'Home GPU');
  const homeGpuLastCheck = dashboard.locator('[data-aws-status-part="home-gpu-last-check"]');
  const browserTicks = ticks(tools, 'Browser-checked tool');
  const projectTicks = ticks(projects, 'Fixture project');
  const gameTicks = ticks(games, 'Fixture game');
  const rowLengths = () => dashboard.locator('.aws-status__ticks').evaluateAll((rows) =>
    rows.map((row) => row.querySelectorAll('.aws-status__tick').length));
  await expect.poll(rowLengths).toEqual(Array(12).fill(70));
  await tools.toggle.click();
  await expect(tools.rows).toHaveCount(0);
  await expect(tools.summary.last()).toBeVisible();
  await tools.toggle.press('Enter');
  await expect(tools.rows).toHaveCount(2);
  await expect(dashboard).not.toContainText(/Browser check (passed|had issues|failed)/i);

  await awsTicks.last().hover();
  await expect(tooltip.locator('strong')).toContainText('1:00 AM MDT');
  await expect(tooltip.locator('p')).toContainText('First repeated hour incident');
  await expect(tooltip.locator(':scope > *')).toHaveCount(2);
  await expect(tooltip).not.toContainText('Observed');
  await transcribeTicks.last().hover();
  await expect(transcribeTicks.and(page.locator('[data-state="unknown"]'))).toHaveCount(70);
  await expect(transcribeTicks.last()).toHaveAttribute('data-state', 'unknown');
  await expect(tooltip.locator('p')).toContainText('No AWS transcription activity recorded.');
  await expect(aws.rows.filter({ hasText: 'Home GPU' })).toHaveCount(0);
  await expect(localServices.rows.filter({ hasText: 'File Transcriber (AWS)' })).toHaveCount(0);
  await expect(dashboard).toContainText('Offline means the PC stopped reporting.');
  await expect(homeGpuLastCheck).toContainText(/Home GPU last check: Offline.*Stale/);
  await expect(homeGpuLastCheck.locator('time')).toHaveAttribute('datetime', '2026-11-01T07:10:00.000Z');
  await expect(homeGpuTicks.nth(-3)).toHaveAccessibleName(/Unknown$/);
  await expect(homeGpuTicks.nth(-2)).toHaveAccessibleName(/Worker unavailable$/);
  await homeGpuTicks.nth(-2).hover();
  await expect(tooltip.locator('p')).toHaveText('Worker unavailable · The PC reported that the transcription worker is unavailable.');
  await expect(homeGpuTicks.last()).toHaveAccessibleName(/Offline$/);
  await homeGpuTicks.last().hover();
  await expect(tooltip.locator('p')).toHaveText('Offline · No report arrived from the PC.');
  await expect(localServices.summary.last()).toHaveAccessibleName(/Offline$/);

  // AWS starts the hour before the catalog reports arrive. A refresh with both
  // types of data still holds the hour until the API marks it ready.
  const freshness = dashboard.locator('[data-aws-status-part="freshness"] time');
  const refresh = async (checkedAt) => {
    snapshot.checkedAt = checkedAt;
    await page.clock.fastForward(5 * 60 * 1000 + 1000);
    await expect(freshness).toHaveAttribute('datetime', checkedAt);
  };
  assert.equal(queryCount(), 1);
  for (const { result, latest } of newestCatalogHour) result.hours = result.hours.slice(0, -1) + latest;
  snapshot.homeGpu.hours.push({ start: '2026-11-01T09:00:00.000Z', from: '2026-11-01T09:00:00.000Z',
    to: '2026-11-01T09:00:00.000Z', state: 'operational', issue: '', complete: true, observedAt: '2026-11-01T09:09:00.000Z' });
  await refresh('2026-11-01T09:09:00.000Z');
  assert.equal(queryCount(), 2);
  await expect.poll(rowLengths).toEqual(Array(12).fill(70));
  await expect(homeGpuTicks.last()).toHaveAccessibleName(/Offline$/);
  await expect(homeGpuLastCheck).toContainText('Home GPU last check: Online');
  await expect(homeGpuLastCheck.locator('time')).toHaveAttribute('datetime', '2026-11-01T09:09:00.000Z');
  await expect(homeGpuLastCheck).not.toContainText('Stale');
  await expect(aws.summary.last()).toHaveAttribute('data-state', 'degraded');
  await expect(awsTicks.last()).toHaveAttribute('data-state', 'degraded');
  await expect(browserTicks.last()).toHaveAttribute('data-state', 'degraded');
  await expect(projectTicks.last()).toHaveAttribute('data-state', 'operational');

  await dashboard.evaluate((element) => {
    window.fixtureMixedColumns = false;
    new MutationObserver(() => {
      const rows = [...element.querySelectorAll('.aws-status__ticks')];
      const lengths = new Set(rows.map((row) => row.querySelectorAll('.aws-status__tick').length));
      if (lengths.size > 1) window.fixtureMixedColumns = true;
    }).observe(element, { childList: true, subtree: true });
  });
  snapshot.hourlyReadyThrough = nextHour;
  await refresh('2026-11-01T09:14:00.000Z');
  assert.equal(queryCount(), 3);
  await expect.poll(rowLengths).toEqual(Array(12).fill(71));
  assert.equal(await page.evaluate(() => window.fixtureMixedColumns), false);
  await expect(aws.summary.last()).toHaveAttribute('data-state', 'unknown');
  await expect(awsTicks.last()).toHaveAttribute('data-state', 'operational');
  await expect(transcribeTicks.last()).toHaveAccessibleName(/Unknown$/);
  await expect(homeGpuTicks.last()).toHaveAccessibleName(/Online$/);
  await homeGpuTicks.last().hover();
  await expect(tooltip.locator('p')).toHaveText('Online');
  await expect(browserTicks.last()).toHaveAttribute('data-state', 'operational');
  await expect(projectTicks.last()).toHaveAttribute('data-state', 'degraded');
  await expect(gameTicks.last()).toHaveAttribute('data-state', 'operational');
  // Both versions of 1 AM belong to the same three-period Mountain date band.
  await expectStatusAxis(dashboard, [
    ['2026-10-29', 20], ['2026-10-30', 24], ['2026-10-31', 24], ['2026-11-01', 3]
  ]);
  const marks = (ticks) => ticks.evaluateAll((elements) => elements.slice(-4).map((tick) => {
    const { height, backgroundColor, backgroundImage, borderRadius } = getComputedStyle(tick, '::before');
    return { state: tick.dataset.state, height, backgroundColor, backgroundImage, borderRadius };
  }));
  const summaryMarks = await marks(localServices.summary);
  expect(summaryMarks.map((mark) => mark.state)).toEqual(['unknown', 'degraded', 'outage', 'operational']);
  expect(await marks(homeGpuTicks)).toEqual(summaryMarks);
  await awsTicks.last().hover();
  await expect(tooltip.locator('strong')).toContainText('1:00 AM MST');
  await awsTicks.last().press('ArrowLeft');
  await expect(awsTicks.nth(-2)).toBeFocused();
  await expect(tooltip.locator('strong')).toContainText('1:00 AM MDT');
  await awsTicks.nth(-2).press('Escape');
  await expect(tooltip).toBeHidden();

  await period.selectOption('daily');
  await expect.poll(rowLengths).toEqual(Array(12).fill(89));
  await expectStatusAxis(dashboard, [
    ['2026-08-04', 6], ['2026-08-10', 7], ['2026-08-17', 7], ['2026-08-24', 7],
    ['2026-08-31', 7], ['2026-09-07', 7], ['2026-09-14', 7], ['2026-09-21', 7],
    ['2026-09-28', 7], ['2026-10-05', 7], ['2026-10-12', 7], ['2026-10-19', 7], ['2026-10-26', 6]
  ]);
  await expect(homeGpuTicks.nth(-2)).toHaveAccessibleName(/Offline$/);
  await expect(homeGpuTicks.last()).toHaveAccessibleName(/Online$/);
  await period.selectOption('weekly');
  await expect.poll(rowLengths).toEqual(Array(12).fill(13));
  await expectStatusAxis(dashboard, [['2026-08-04', 5], ['2026-09-07', 4], ['2026-10-05', 4]]);
  await expect(homeGpuTicks.last()).toHaveAccessibleName(/Offline$/);
  await expect(awsTicks.nth(11)).toHaveAttribute('data-state', 'outage');
  await awsTicks.nth(11).hover();
  await expect(tooltip.locator('strong')).toContainText(/Oct 19.*Oct 25/);
  await expect(tooltip.locator('p')).toContainText('Known outage');

  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  try {
    await fixtureAuth(mobile);
    await mobile.route(base + '/api/tools/aws-status', (route) => route.fulfill({ json: snapshot }));
    const phone = await mobile.newPage();
    await phone.clock.install({ time: now });
    await visit(phone, base, item.route);
    const mobilePeriod = phone.getByRole('combobox', { name: 'Group by' });
    const mobileTools = statusGroup(phone, 'Tools');
    const mobileLocal = statusGroup(phone, 'Local services');
    const mobileTicks = ticks(mobileTools, 'Browser-checked tool');
    const mobileHomeGpuTicks = ticks(mobileLocal, 'Home GPU');
    const expectSeparatedTouchTargets = async (row) => {
      const { height, gap } = await row.evaluateAll((ticks) => {
        const boxes = ticks.map((tick) => tick.getBoundingClientRect());
        return {
          height: Math.min(...boxes.map((box) => box.height)),
          gap: Math.min(...boxes.slice(1).map((box, index) => box.left - boxes[index].right))
        };
      });
      expect(gap).toBeGreaterThan(0);
      expect(height).toBeGreaterThanOrEqual(44);
    };
    await expect(mobilePeriod).toHaveValue('hourly');
    await mobileTools.toggle.tap();
    await expect(mobileTicks).toHaveCount(71);
    await expectSeparatedTouchTargets(mobileTools.summary);
    await expectSeparatedTouchTargets(mobileTicks);
    await mobileTicks.nth(-2).tap();
    await expect(phone.getByRole('tooltip').locator('strong')).toContainText('1:00 AM MDT');
    await expect(phone.getByRole('tooltip').locator('p')).toContainText('Degraded');
    await mobileTicks.last().tap();
    await expect(phone.getByRole('tooltip').locator('strong')).toContainText('1:00 AM MST');
    await expect(phone.getByRole('tooltip').locator('p')).toContainText('Operational');
    await mobilePeriod.selectOption('weekly');
    await expect(phone.getByRole('tooltip')).toBeHidden();
    await expect(mobileTicks).toHaveCount(13);
    await expectSeparatedTouchTargets(mobileTools.summary);
    await expectSeparatedTouchTargets(mobileTicks);
    await expect(mobileTicks.last()).toBeInViewport();
    await mobilePeriod.selectOption('hourly');
    await mobileLocal.toggle.tap();
    await expectSeparatedTouchTargets(mobileHomeGpuTicks);
    await mobileHomeGpuTicks.nth(-2).tap();
    await expect(phone.getByRole('tooltip').locator('p')).toContainText('Offline');
    await mobileHomeGpuTicks.last().tap();
    await expect(phone.getByRole('tooltip').locator('p')).toHaveText('Online');
    const phoneHomeGpuLastCheck = phone.locator('[data-aws-status-part="home-gpu-last-check"]');
    const { left, right, instantWidth } = await phoneHomeGpuLastCheck.evaluate((node) => {
      const { left, right } = node.getBoundingClientRect();
      return { left, right, instantWidth: window.innerWidth };
    });
    expect(left).toBeGreaterThanOrEqual(0);
    expect(right).toBeLessThanOrEqual(instantWidth);
    assert(await phone.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  } finally {
    await mobile.close();
  }

  delete snapshot.hourlyReadyThrough;
  delete snapshot.connections[0].hours;
  delete snapshot.catalogHistory;
  delete snapshot.homeGpu;
  await context.addInitScript(() => { window.fixtureRole = 'admin'; });
  await page.reload({ waitUntil: 'domcontentloaded' });
  await expect(period).toHaveValue('hourly');
  await expect(localServices.toggle).toHaveCount(0);
  await expect(aws.toggle).toHaveAttribute('aria-expanded', 'false');
  await aws.toggle.click();
  await expect(awsTicks).toHaveCount(72);
  await expect(awsTicks.last()).toHaveAttribute('data-state', 'unknown');
  await period.selectOption('daily');
  await expect(awsTicks.last()).toHaveAttribute('data-state', 'operational');
  return local('The fixture verified access, separate Home GPU history, grouped refresh, concise details and touch controls.');
}

async function statusTool({ browser, context, page, base, item }) {
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
  await expect(dashboard).toBeHidden();
  assert.equal(queries, 0);
  if (await dashboard.locator('#aws-status-heading').count()) {
    await expect(page.locator('[data-aws-status-part="access"]')).toHaveText('This account does not have access to AWS status.');
    return unifiedStatusTool({ browser, context, page, base, item, snapshot, queryCount: () => queries });
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
