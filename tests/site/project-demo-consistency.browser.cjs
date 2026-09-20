/**
 * Shared project-demo layout and puzzle/chat workflows against built output.
 * Puzzle data and passive chat status are fixtures; no AWS startup or chat sends.
 */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');

const DEMOS = [
  { id: 'nonogram', project: 'nonogram', size: 5, maxWidth: 600 },
  { id: 'minesweeper', project: 'minesweeper', published: false, size: 9, maxWidth: 600 },
  { id: 'chatbot', project: 'chatbotLora', maxWidth: 960 }
];

function puzzleFixture(id, revision) {
  if (id === 'nonogram') {
    const solution = Array.from({ length: 5 }, (_, row) => Array.from({ length: 5 }, (_, col) => Number(row === 2 || col === 2 || (revision % 2 && row === col))));
    const clues = cells => {
      const result = [];
      let run = 0;
      for (const cell of cells.concat(0)) {
        if (cell) run += 1;
        else if (run) { result.push(run); run = 0; }
      }
      return result.length ? result : [0];
    };
    const steps = solution.flatMap((row, rowIndex) => row.map((actual, col) => ({ row: rowIndex, col, actual, predicted: actual, correct: true })));
    return { grid: 5, solution, row_clues: solution.map(clues), col_clues: solution.map((_, col) => clues(solution.map(row => row[col]))), steps, step_count: 25, correct_count: 25, correct_rate: 1 };
  }
  const mines = new Set(['0,0', '0,4', '0,8', '2,2', '2,6', '4,0', '4,4', '4,8', '6,2', '8,6']);
  const solution = Array.from({ length: 9 }, (_, row) => Array.from({ length: 9 }, (_, col) => {
    if (mines.has(`${row},${col}`)) return -1;
    let adjacent = 0;
    for (let dr = -1; dr <= 1; dr += 1) for (let dc = -1; dc <= 1; dc += 1) if (mines.has(`${row + dr},${col + dc}`)) adjacent += 1;
    return adjacent;
  }));
  const opened = solution.flatMap((row, rowIndex) => row.flatMap((value, col) => value < 0 ? [] : [[col, rowIndex]]));
  return {
    grid: 9, solution, success: true, hit_mine: false, step_count: 1,
    steps: [{ row: 8, col: 8, newly_opened: opened, decision: 'logic', message: 'Revealed safe cells.' }],
    model: { dqn_variant: 'DoubleDQN', cnn_variant: 'CNN', replay_type: 'prioritized', success_rate: .72 }
  };
}

async function assertSurface(page, frame, demo, label) {
  await frame.evaluate(() => document.fonts.ready);
  assert.equal(await frame.locator('.demo-surface').count(), 1, `${label} has one shared interaction surface.`);
  assert.equal(await frame.locator('h1:visible').count(), 0, `${label} does not repeat the outer page heading inside the iframe.`);
  assert.equal(await frame.locator('.demo-surface-header').count(), 1, `${label} has one compact status header.`);
  const geometry = await frame.evaluate(() => {
    const surface = document.querySelector('.demo-surface');
    const rect = surface.getBoundingClientRect();
    return { width: rect.width, left: rect.left, right: rect.right, viewport: innerWidth, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth, background: getComputedStyle(surface).backgroundColor };
  });
  assert(geometry.width <= demo.maxWidth + 1, `${label} respects the ${demo.maxWidth}px interaction preset.`);
  assert(Math.abs(geometry.left - (geometry.viewport - geometry.right)) <= 2, `${label} centers the workspace.`);
  assert.equal(geometry.background, 'rgb(255, 255, 255)', `${label} keeps the workspace white.`);
  assert(geometry.overflow <= 1, `${label} iframe has no horizontal overflow.`);
  const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert(pageOverflow <= 1, `${label} outer page has no horizontal overflow.`);
}

async function snapshotCells(cells) {
  return cells.evaluateAll(nodes => nodes.map(node => ({
    classes: Array.from(node.classList).filter(name => !['newly-opened', 'ai-step'].includes(name)).sort(),
    text: node.textContent
  })));
}

async function runPuzzle(page, frame, demo, state, artifactDir, label) {
  const cells = frame.locator('#grid .cell');
  await frame.waitForFunction(size => document.querySelectorAll('#grid .cell').length === size * size && document.querySelector('#action-btn')?.dataset.mode === 'solve' && !document.querySelector('#action-btn').disabled, demo.size);
  assert.equal(await cells.count(), demo.size * demo.size, `${label} renders its ${demo.size} by ${demo.size} board.`);
  if (demo.id === 'minesweeper') {
    assert.equal(await frame.locator('#stat-moves').innerText(), '0', `${label} starts with no played moves.`);
    assert.equal(await frame.locator('#stat-outcome').innerText(), 'Not run', `${label} waits for playback before presenting its outcome.`);
  }
  assert.match(await frame.locator('#health-pill').innerText(), /AWS.*Connected/i, `${label} retains visible AWS connectivity.`);
  assert.equal(await frame.locator('.demo-status-actions #health-pill').count(), 1, `${label} puts connectivity in the shared status area.`);
  const disclosure = frame.locator('.puzzle-details');
  assert.equal(await disclosure.getAttribute('open'), null, `${label} keeps solver details collapsed initially.`);
  const actions = await frame.locator('.board-actions').boundingBox();
  const board = await frame.locator('.board-shell').boundingBox();
  assert(actions.y + actions.height <= board.y + 1, `${label} keeps primary actions above the board.`);
  assert(Math.abs(actions.x + actions.width / 2 - board.x - board.width / 2) <= 2, `${label} centers actions on the board.`);
  const grid = await frame.locator('#grid').boundingBox();
  assert(grid.x >= board.x - 1 && grid.x + grid.width <= board.x + board.width + 1, `${label} shows every board column without horizontal scrolling.`);
  if (demo.id === 'minesweeper') {
    assert(grid.width >= board.width - 40, `${label} fills its board surface with a readable grid rather than shrinking to intrinsic cell content.`);
    assert(Math.abs(grid.width - grid.height) <= 2, `${label} keeps the board square.`);
  }
  await assertSurface(page, frame, demo, label);
  await page.screenshot({ path: path.join(artifactDir, `${label}-ready.png`) });

  const solution = frame.locator('#solution-btn');
  const blank = await snapshotCells(cells);
  await solution.click();
  await page.mouse.move(0, 0);
  assert.equal(await solution.getAttribute('aria-pressed'), 'true', `${label} supports a latched touch/click solution preview.`);
  const preview = await snapshotCells(cells);
  assert.notDeepEqual(preview, blank, `${label} displays the solution without making an API request.`);
  await solution.press('Enter');
  await frame.locator('#action-btn').focus();
  assert.equal(await solution.getAttribute('aria-pressed'), 'false', `${label} keyboard activation dismisses the solution.`);
  assert.deepEqual(await snapshotCells(cells), blank, `${label} restores the prior board after solution preview.`);
  const requestsBeforeSolve = state.solves.length;
  await frame.locator('#action-btn').click();
  await frame.waitForFunction(() => document.querySelector('#action-btn')?.dataset.mode === 'new' && !document.querySelector('#action-btn').disabled);
  assert.equal(state.solves.length, requestsBeforeSolve, `${label} animates the already-loaded solver trace without another API request.`);
  if (demo.id === 'nonogram') assert.match(await frame.locator('#stat-accuracy').innerText(), /100%.*25\/25/);
  else {
    assert.equal(await frame.locator('#stat-outcome').innerText(), 'Solved');
    assert.match(await frame.locator('#stat-moves').innerText(), /^1(?: moves?)?$/);
    assert.doesNotMatch(await frame.locator('#status').innerText(), /Puzzle ready/i, `${label} updates its live status when the trace completes.`);
  }
  const completed = await snapshotCells(cells);
  assert(await solution.isEnabled(), `${label} keeps the solution available after solving.`);
  await solution.click();
  assert.equal(await solution.getAttribute('aria-pressed'), 'true');
  await solution.press('Enter');
  assert.equal(await solution.getAttribute('aria-pressed'), 'false');
  assert.deepEqual(await snapshotCells(cells), completed, `${label} restores the actual completed board after solution preview.`);
  await frame.locator('.board-actions').scrollIntoViewIfNeeded();
  await page.mouse.move(0, 0);
  await page.screenshot({ path: path.join(artifactDir, `${label}-solved.png`) });
  await disclosure.locator('summary').click();
  await disclosure.scrollIntoViewIfNeeded();
  assert.equal(await frame.locator('#log .log-entry').count(), demo.id === 'nonogram' ? 25 : 1, `${label} retains solver history below the board.`);
  await assertSurface(page, frame, demo, `${label} details`);
  await page.screenshot({ path: path.join(artifactDir, `${label}-details.png`) });
  await disclosure.locator('summary').click();
  state.failSolve = true;
  await frame.locator('#action-btn').click();
  await frame.waitForFunction(() => document.querySelector('#health-pill')?.dataset.state === 'err' && !document.querySelector('#action-btn').disabled);
  assert.match(await frame.locator('#status').innerText(), /failed|could not|try again/i, `${label} explains a failed puzzle request.`);
  assert(await frame.locator('#action-btn').isEnabled(), `${label} keeps retry available after a failed puzzle request.`);
  assert.deepEqual(await snapshotCells(cells), completed, `${label} preserves the completed board when a replacement request fails.`);
  assert(await solution.isEnabled(), `${label} keeps the prior solution available after a failed replacement.`);
  state.failSolve = false;
  await frame.locator('#action-btn').click();
  await frame.waitForFunction(() => document.querySelector('#action-btn')?.dataset.mode === 'solve' && !document.querySelector('#action-btn').disabled);
  assert.equal(await cells.count(), demo.size * demo.size, `${label} recovers to a usable new puzzle.`);
  await assertSurface(page, frame, demo, `${label} recovered`);
}

async function runChat(page, frame, demo, state, artifactDir, label) {
  await frame.locator('#regular-prompt').waitFor({ state: 'visible' });
  const connection = frame.locator('.demo-status-actions .aws-status-badge');
  assert(await connection.isVisible(), `${label} keeps the chat service state visible in its shared status area.`);
  assert.match(await connection.innerText(), /AWS.*Ready/i, `${label} identifies the default chat service as ready.`);
  const settings = frame.locator('#chat-settings');
  const openSettings = async () => {
    if (!await settings.evaluate(node => node.open)) await settings.locator(':scope > summary').click();
  };
  const selectBackend = async (id) => {
    const select = frame.locator('#backend-select');
    // Exercise the visible native control. A programmatic selectOption can
    // open its dialog while the parent panel has scrolled the iframe offscreen.
    await select.click();
    await select.press(id === 'qwen-sagemaker' ? 'Home' : 'End');
    await select.press('Enter');
    assert.equal(await select.inputValue(), id, `${label} changes the backend through its native control.`);
  };
  assert.equal(await settings.getAttribute('open'), null, `${label} keeps advanced chat controls closed initially.`);
  assert.match(await settings.locator(':scope > summary').innerText(), /Advanced settings/i);
  const composer = await frame.locator('#regular-view .chat-composer').boundingBox();
  const messages = await frame.locator('#regular-messages').boundingBox();
  const shell = await frame.locator('#regular-view .chat-shell').boundingBox();
  assert(composer.y >= messages.y + messages.height - 1, `${label} retains the conversation composer at the bottom.`);
  assert(Math.abs(composer.y + composer.height - shell.y - shell.height) <= 2, `${label} keeps the empty-state composer at the bottom of the chat shell.`);
  await assertSurface(page, frame, demo, label);
  await page.screenshot({ path: path.join(artifactDir, `${label}-ready.png`) });
  const draft = 'A local draft for layout review. Do not submit.';
  await frame.locator('#regular-prompt').fill(draft);
  await openSettings();
  await frame.locator('#popup-view-button').click();
  await frame.locator('#popup-launcher').click();
  assert.equal(await frame.locator('#popup-prompt').inputValue(), draft, `${label} keeps an unsent draft when changing chat presentation.`);
  assert.equal(await frame.locator('#popup-launcher').getAttribute('aria-expanded'), 'true');
  const popupMessages = await frame.locator('#popup-messages').boundingBox();
  const popupGreeting = await frame.locator('#popup-messages .empty-title').boundingBox();
  assert(popupGreeting.y >= popupMessages.y - 1, `${label} keeps the popup greeting below its header without clipping.`);
  await assertSurface(page, frame, demo, `${label} popup`);
  await page.screenshot({ path: path.join(artifactDir, `${label}-popup.png`) });
  await frame.locator('#popup-close').click();
  await openSettings();
  await frame.locator('#regular-view-button').click();
  assert.equal(await frame.locator('#regular-prompt').inputValue(), draft);
  await openSettings();
  await selectBackend('qwen-sagemaker');
  await frame.locator('#qwen-startup-notice').waitFor({ state: 'visible' });
  await frame.locator('#qwen-startup-close').click();
  await frame.waitForFunction(() => document.querySelector('#warmup-button')?.textContent === 'Prepare demo');
  assert(await frame.locator('#warmup-button').isEnabled(), `${label} preserves explicit on-demand startup.`);
  assert.equal(state.mutations.length, 0, `${label} never warms a backend or sends a message while reviewing layout/settings.`);
  await openSettings();
  await selectBackend('bedrock');
  await page.keyboard.press('Escape');
  await assertSurface(page, frame, demo, `${label} restored`);
}

async function runCase({ browser, base, artifactDir, demo, width, project }) {
  const label = `${demo.id}-${project ? 'project' : 'standalone'}-${width}`;
  const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1000 }, reducedMotion: 'reduce', serviceWorkers: 'block', isMobile: width < 600, hasTouch: width < 600 });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  const state = { solves: [], failSolve: false, mutations: [] };
  page.on('pageerror', error => errors.push(error.message));
  await context.route(/https:\/\/[^/]*(?:amazonaws\.com|\.on\.aws)\//, async route => {
    errors.push(`Unexpected live AWS request: ${route.request().url()}`);
    await route.abort();
  });
  await context.route(`${base}/api/**`, async route => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    const puzzle = pathname.match(/^\/api\/demos\/(nonogram|minesweeper)\/(health|warmup|solve)$/);
    if (puzzle) {
      if (puzzle[2] === 'solve') {
        state.solves.push(request.postDataJSON());
        await route.fulfill({ status: state.failSolve ? 503 : 200, contentType: 'application/json', body: JSON.stringify(state.failSolve ? { error: 'Fixture temporarily unavailable.' } : puzzleFixture(puzzle[1], state.solves.length)) });
      } else await route.fulfill({ status: 200, contentType: 'application/json', body: '{"status":"ready","model_loaded":true}' });
      return;
    }
    if (pathname.startsWith('/api/chatbot')) {
      if (request.method() === 'GET' && pathname.endsWith('/status')) {
        const online = pathname.includes('/bedrock/');
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: online ? 'READY' : 'OFF', online, stage: { message: online ? 'Fixture ready.' : 'Fixture stopped.' } }) });
      } else {
        state.mutations.push(`${request.method()} ${pathname}`);
        await route.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"Startup and conversation requests are not permitted in layout review."}' });
      }
      return;
    }
    if (pathname === '/api/contact') {
      state.mutations.push(`${request.method()} ${pathname}`);
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{}' });
      return;
    }
    await route.continue();
  });
  try {
    const response = await page.goto(base + (project ? `/portfolio/${demo.project}` : `/${demo.id}-demo`), { waitUntil: 'domcontentloaded' });
    assert.equal(response.status(), 200, `${label} route loads.`);
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    let embedded = project;
    if (project) {
      const heading = page.locator('.project-demo-header');
      await heading.scrollIntoViewIfNeeded();
      assert.notEqual((await heading.locator('.project-demo-title').innerText()).trim(), 'Demo', `${label} supplies one descriptive outer heading.`);
      assert((await heading.locator('.project-demo-description').innerText()).trim().length > 15, `${label} supplies one concise outer description.`);
      if (width < 600 && await page.locator('.project-intro-action--demo').isVisible()) {
        await page.locator('.project-intro-action--demo').click();
        await page.waitForURL(`**/${demo.id}-demo`);
        embedded = false;
      }
    }
    const iframe = page.locator(embedded ? 'iframe.project-embed-frame' : 'iframe.project-demo-wrapper-iframe');
    await iframe.scrollIntoViewIfNeeded();
    const frame = await (await iframe.elementHandle()).contentFrame();
    if (demo.id === 'chatbot') await runChat(page, frame, demo, state, artifactDir, label);
    else await runPuzzle(page, frame, demo, state, artifactDir, label);
    assert.deepEqual(errors, [], `${label} has no runtime exceptions or direct AWS requests.`);
    assert.equal(state.mutations.length, 0, `${label} submits no conversation, startup, or contact requests.`);
    console.log(`Shared project demo consistency passed: ${label}`);
  } catch (error) {
    await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
    error.message = `${label}: ${error.message}`;
    throw error;
  } finally { await context.close(); }
}

async function runProjectDemoConsistencyChecks({ browser, base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  for (const demo of DEMOS) for (const [project, width] of [[false, 1440], [false, 390], [false, 320], [true, 1440], [true, 390]]) {
    if (project && demo.published === false) continue;
    await runCase({ browser, base, artifactDir, demo, width, project });
  }
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'project-demo-consistency-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runProjectDemoConsistencyChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-demo-consistency') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}

module.exports = runProjectDemoConsistencyChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
