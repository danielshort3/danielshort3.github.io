/** Account autosave UI checks with simulated auth/storage. No real account writes. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { createLocalServer } = require('../../build/dev');
const accountSelector = '[data-tools-action="open-account"]';
const signOutSelector = '[data-tools-action="sign-out"]';

async function metrics(page) {
  return page.evaluate(() => {
    const box = selector => {
      const node = document.querySelector(selector);
      const rect = node.getBoundingClientRect();
      const transform = getComputedStyle(node).transform;
      const offset = transform === 'none' ? { e: 0, f: 0 } : new DOMMatrixReadOnly(transform);
      return { x: rect.x - offset.e, y: rect.y - offset.f, width: rect.width, height: rect.height, right: rect.right - offset.e };
    };
    return { header: box('[data-page-masthead]'), account: box('[data-tools-action="open-account"]'), signOut: box('[data-tools-action="sign-out"]') };
  });
}

function sameLayout(actual, expected, state) {
  for (const element of ['header', 'account', 'signOut']) {
    for (const property of ['x', 'y', 'width', 'height']) {
      assert(Math.abs(actual[element][property] - expected[element][property]) <= 1,
        `${state} must not move or resize ${element}.${property}: ${JSON.stringify({ actual, expected })}`);
    }
  }
}

async function noManualSave(page, state) {
  assert.equal(await page.locator('[data-tools-action="save-session"]').count(), 0, `${state} must not render a Save or Retry save control.`);
  assert.equal(await page.locator('[data-tools-account="save-privacy"]').count(), 0, `${state} must not retain a manual-save tooltip.`);
}

async function runCase({ browser, base, artifactDir }, { width, route }) {
  const context = await browser.newContext({ viewport: { width, height: width === 320 ? 740 : 900 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const unexpectedRequests = [];
  const errors = [];
  const dialogs = [];
  let stage = 'initial';
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
  await context.route('**/api/tools/**', async request => {
    const req = request.request();
    const isInitialSessionProbe = stage === 'initial' && req.method() === 'GET' && new URL(req.url()).pathname === '/api/tools/auth/session';
    if (!isInitialSessionProbe) unexpectedRequests.push(req.url());
    await request.fulfill({ status: 200, contentType: 'application/json', body: '{"ok":true,"sessions":[],"activity":[]}' });
  });
  await context.route(/https:\/\/[^/]*(?:amazoncognito|cognito-idp)[^/]*\//, async request => {
    unexpectedRequests.push(request.request().url());
    await request.abort();
  });
  try {
    await page.goto(`${base}/tools/${route}`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__toolsAccountUiController && document.querySelector('[data-tools-account-route-id]'));
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => {
      window.accountTestSignedIn = true;
      window.accountTestRequests = [];
      window.accountTestPending = new Map();
      window.accountTestSignOutCalls = 0;
      window.accountTestResolve = (number, version) => accountTestPending.get(number).resolve({ session: { sessionId: 'local-ui-test', version } });
      window.accountTestReject = (number) => accountTestPending.get(number).reject(new Error('Offline. Retrying automatically.'));
      window.ToolsAuth = {
        ...window.ToolsAuth,
        getAuth: () => accountTestSignedIn ? { test: true } : null,
        authIsValid: () => accountTestSignedIn,
        getUser: () => accountTestSignedIn ? { sub: 'local-account-ui-test', email: 'local-ui@example.test' } : {},
        ensureFreshAuth: async () => null,
        signIn: async () => { throw new Error('Real sign-in is not part of this test.'); },
        signOut: async () => { window.accountTestSignOutCalls += 1; window.accountTestSignedIn = false; }
      };
      window.ToolsState = {
        ...window.ToolsState,
        logActivity: async () => ({}),
        saveSession: (request) => {
          accountTestRequests.push(request);
          const number = accountTestRequests.length;
          return new Promise((resolve, reject) => accountTestPending.set(number, { resolve, reject }));
        },
        listSessions: async () => ({ sessions: [] }),
        getSession: async () => ({ session: { sessionId: 'local-ui-test', version: 1, snapshot: {} } }),
        getDashboard: async () => ({ recentSessions: [], tools: [] })
      };
      document.dispatchEvent(new CustomEvent('tools:auth-changed'));
    });
    const account = page.locator(accountSelector);
    const signOut = page.locator(signOutSelector);
    const status = page.locator('[data-tools-account="bar"] [data-tools-account="status"]');
    await account.waitFor({ state: 'visible' });
    await signOut.waitFor({ state: 'visible' });
    await page.waitForFunction(() => !SiteFrame.root().matches('.site-frame--moving, .site-frame--held'));
    const clean = await metrics(page);
    assert(clean.account.right + 4 <= clean.signOut.x && Math.abs(clean.account.y - clean.signOut.y) <= 1,
      'Sign out stays directly to the right of Account.');
    assert(clean.account.x >= 0 && clean.signOut.right <= width, 'The account controls fit the viewport.');
    await noManualSave(page, 'Clean work');
    assert.equal(await page.locator('[data-tools-account="actions"] button:visible').count(), 2);
    await account.focus();
    await page.keyboard.press('Tab');
    assert(await signOut.evaluate(node => node === document.activeElement), 'Keyboard order follows Account then Sign out.');

    stage = 'automatic-save-and-retry';
    const dirty = () => page.evaluate(toolId => document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId } })), route);
    await dirty();
    await page.waitForFunction(() => accountTestRequests.length === 1);
    assert((await status.innerText()).includes('Saving'), 'Autosave gives quiet saving feedback.');
    sameLayout(await metrics(page), clean, 'Saving');
    await noManualSave(page, 'Saving work');
    await page.evaluate(() => accountTestReject(1));
    await page.waitForFunction(() => /retrying/i.test(document.querySelector('[data-tools-account="bar"] [data-tools-account="status"]').textContent));
    sameLayout(await metrics(page), clean, 'Automatic retry feedback');
    await noManualSave(page, 'Failed save');
    await page.evaluate(() => window.dispatchEvent(new Event('online')));
    await page.waitForFunction(() => accountTestRequests.length === 2);
    await page.evaluate(() => accountTestResolve(2, 1));
    await page.waitForFunction(() => document.querySelector('[data-tools-account="bar"] [data-tools-account="status"]').textContent.includes('Saved'));
    sameLayout(await metrics(page), clean, 'Saved');

    stage = 'signout-flushes-newest-work';
    await dirty();
    await page.waitForFunction(() => accountTestRequests.length === 3);
    await signOut.click();
    await dirty();
    assert.equal(await page.evaluate(() => accountTestSignOutCalls), 0, 'Sign out waits for an in-flight autosave.');
    await page.evaluate(() => accountTestResolve(3, 2));
    await page.waitForFunction(() => accountTestRequests.length === 4);
    assert.equal(await page.evaluate(() => accountTestSignOutCalls), 0, 'Sign out also waits for edits made during the first save.');
    const lastRequest = await page.evaluate(() => ({ sessionId: accountTestRequests[3].sessionId, expectedVersion: accountTestRequests[3].expectedVersion }));
    assert.deepEqual(lastRequest, { sessionId: 'local-ui-test', expectedVersion: 2 }, 'Follow-up saves stay in the existing session and use the latest version.');
    await noManualSave(page, 'Pending sign out');
    await page.evaluate(() => accountTestResolve(4, 3));
    await page.waitForFunction(() => accountTestSignOutCalls === 1);
    await page.locator('[data-tools-action="sign-in"]').waitFor({ state: 'visible' });
    await noManualSave(page, 'Signed out');
    assert.deepEqual(dialogs, [], 'Automatic saving never asks to discard ordinary unsaved edits on sign-out.');
    assert.deepEqual(unexpectedRequests, [], 'All auth and storage activity stayed inside simulated services.');
    assert.deepEqual(errors, [], 'The account controls produce no runtime errors.');
    await page.screenshot({ path: path.join(artifactDir, `account-autosave-${route}-${width}.png`) });
    console.log(`Account autosave passed: ${route} ${width}px (no Save button, stable feedback, automatic retry, latest work flushed before sign-out).`);
  } catch (error) {
    const screenshot = path.join(artifactDir, `account-autosave-${route}-${width}-failure.png`);
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `${route} ${width}px ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

function installAccountSwitchServices() {
  const ownerKey = 'account-switch-owner';
  const traceKey = 'account-switch-trace';
  const documentKey = 'account-switch-document';
  if (!sessionStorage.getItem(ownerKey)) sessionStorage.setItem(ownerKey, 'person-a');
  const documentNumber = Number(sessionStorage.getItem(documentKey) || 0) + 1;
  sessionStorage.setItem(documentKey, String(documentNumber));
  const owner = () => sessionStorage.getItem(ownerKey);
  const trace = (entry) => {
    const entries = JSON.parse(sessionStorage.getItem(traceKey) || '[]');
    entries.push({ owner: owner(), documentNumber, ...entry });
    sessionStorage.setItem(traceKey, JSON.stringify(entries));
  };
  const snapshot = {
    fields: {
      'textcompare-original': { kind: 'value', value: 'BRAVO_ONLY original draft' },
      'textcompare-revised': { kind: 'value', value: 'BRAVO_ONLY revised draft' }
    },
    inputs: { view: 'comparison' },
    output: { kind: 'html', html: '<p>BRAVO_ONLY saved comparison output</p>', summary: 'BRAVO_ONLY saved comparison' }
  };
  const auth = {
    getAuth: () => ({ test: true }),
    getUser: () => ({ sub: owner(), email: `${owner()}@example.test` }),
    authIsValid: () => true,
    isAdmin: () => false,
    getConfig: () => ({}),
    handleRedirect: async () => ({ handled: false }),
    ensureFreshAuth: async () => ({ test: true }),
    signIn: async () => { throw new Error('This test must not start real sign-in.'); },
    signOut: async () => { throw new Error('This test must not start real sign-out.'); },
    fetchWithAuth: async () => { throw new Error('This test must not access real account storage.'); }
  };
  const storage = {
    logActivity: async () => ({}),
    getDashboard: async () => ({ recentSessions: [], tools: [] }),
    listSessions: async () => ({ sessions: owner() === 'person-b' ? [{ toolId: 'text-compare', sessionId: 'person-b-work' }] : [] }),
    getSession: async (request) => {
      trace({ type: 'get', request });
      return { session: { sessionId: 'person-b-work', version: 3, snapshot } };
    },
    saveSession: async (request) => {
      trace({ type: 'save', request });
      return { session: { sessionId: `${owner()}-work`, version: (request.expectedVersion || 0) + 1 } };
    }
  };
  // Install before application bootstrap on both documents. The real scripts
  // still load, while every account operation stays in these simulated APIs.
  Object.defineProperty(window, 'ToolsAuth', { configurable: true, get: () => auth, set: () => {} });
  Object.defineProperty(window, 'ToolsState', { configurable: true, get: () => storage, set: () => {} });
  document.addEventListener('tools:session-capture', (event) => {
    const capturedOwner = owner();
    queueMicrotask(() => trace({ type: 'capture', owner: capturedOwner, snapshot: event.detail.snapshot, payload: event.detail.payload }));
  });
}

async function runAccountSwitchCase({ browser, base, artifactDir }) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce', serviceWorkers: 'block' });
  await context.addInitScript(installAccountSwitchServices);
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  const unexpectedRequests = [];
  let stage = 'person-a-work';
  page.on('pageerror', error => errors.push(error.message));
  await context.route('**/api/tools/**', async route => {
    unexpectedRequests.push(route.request().url());
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await context.route(/https:\/\/[^/]*(?:amazoncognito|cognito-idp)[^/]*\//, async route => {
    unexpectedRequests.push(route.request().url());
    await route.abort();
  });
  try {
    await page.goto(`${base}/tools/text-compare`, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#main')?.dataset.toolsDraftOwner === 'person-a');
    if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
    await page.locator('#textcompare-original').fill('ALPHA_PRIVATE original draft');
    await page.locator('#textcompare-revised').fill('ALPHA_PRIVATE revised draft');
    await page.getByRole('button', { name: 'Compare drafts', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('#textcompare-output')?.textContent.includes('ALPHA_PRIVATE'));
    await page.waitForFunction(() => JSON.parse(sessionStorage.getItem('account-switch-trace') || '[]').some(entry =>
      entry.type === 'save' && entry.request.snapshot.output?.html?.includes('ALPHA_PRIVATE')));
    const initialDocument = await page.evaluate(() => Number(sessionStorage.getItem('account-switch-document')));
    assert.match(page.url(), /session=person-a-work/, 'The initial document has A’s active session and real rendered comparison.');

    stage = 'account-switch-reload';
    await page.evaluate(() => {
      const oldField = document.querySelector('#textcompare-original');
      sessionStorage.setItem('account-switch-owner', 'person-b');
      document.dispatchEvent(new CustomEvent('tools:auth-changed'));
      // Exercise late tool events after the account remount has started but
      // before the browser finishes replacing this document.
      queueMicrotask(() => {
        oldField.value = 'BRAVO_ONLY edited first field';
        oldField.dispatchEvent(new Event('input', { bubbles: true }));
        document.dispatchEvent(new CustomEvent('tool:tab-change', { detail: { panelId: 'textcompare-view-drafts' } }));
        document.dispatchEvent(new CustomEvent('tools:save-session', { detail: { toolId: 'text-compare' } }));
      });
    });
    await page.waitForFunction(previous => Number(sessionStorage.getItem('account-switch-document')) > previous &&
      document.querySelector('#main')?.dataset.toolsDraftOwner === 'person-b' &&
      document.querySelector('#textcompare-output')?.textContent.includes('BRAVO_ONLY saved comparison output'), initialDocument);
    assert.equal(await page.locator('#textcompare-original').inputValue(), 'BRAVO_ONLY original draft');
    assert.equal(await page.locator('#textcompare-revised').inputValue(), 'BRAVO_ONLY revised draft');
    assert.match(page.url(), /session=person-b-work/, 'The fresh document restores B’s own session selection.');
    assert(!(await page.locator('#main').innerText()).includes('ALPHA_PRIVATE'), 'The new account cannot see either old draft or its rendered output.');

    stage = 'person-b-save';
    await page.locator('#textcompare-view-tab-drafts').click();
    await page.locator('#textcompare-original').fill('BRAVO_ONLY newer original draft');
    await page.waitForFunction(() => JSON.parse(sessionStorage.getItem('account-switch-trace') || '[]').some(entry =>
      entry.type === 'save' && entry.owner === 'person-b' && entry.request.snapshot.fields['textcompare-original']?.value === 'BRAVO_ONLY newer original draft'));
    const trace = await page.evaluate(() => JSON.parse(sessionStorage.getItem('account-switch-trace') || '[]'));
    const bEntries = trace.filter(entry => entry.owner === 'person-b');
    assert(bEntries.some(entry => entry.type === 'get' && entry.request.sessionId === 'person-b-work'), 'B’s continuation loads its own saved state.');
    assert(bEntries.every(entry => entry.documentNumber > initialDocument), 'No old-document capture or save may run under B.');
    assert(!JSON.stringify(bEntries).includes('ALPHA_PRIVATE'), 'Neither A input nor A output may be captured or saved under B.');
    const saved = bEntries.filter(entry => entry.type === 'save').at(-1).request;
    assert.equal(saved.snapshot.fields['textcompare-revised'].value, 'BRAVO_ONLY revised draft');
    assert(saved.snapshot.output.html.includes('BRAVO_ONLY saved comparison output'), 'B’s later edits retain only B’s output.');
    assert.deepEqual(unexpectedRequests, [], 'Account switching uses simulated auth/storage across the reload.');
    assert.deepEqual(errors, [], 'The real account-switch reload causes no JavaScript errors.');
    await page.screenshot({ path: path.join(artifactDir, 'account-switch-text-compare.png') });
    console.log('Account switch passed: two private drafts and rendered output isolated across reload; new-account continuation and autosave restored.');
  } catch (error) {
    const screenshot = path.join(artifactDir, 'account-switch-text-compare-failure.png');
    await page.screenshot({ path: screenshot }).catch(() => {});
    error.message = `Text Compare ${stage}: ${error.message} (screenshot: ${screenshot})`;
    throw error;
  } finally {
    await context.close();
  }
}

async function runAccountSaveChecks(options) {
  fs.mkdirSync(options.artifactDir, { recursive: true });
  for (const route of ['screen-recorder', 'image-optimizer', 'text-compare']) {
    for (const width of [1440, 320]) await runCase(options, { route, width });
  }
  await runAccountSwitchCase(options);
}

module.exports = runAccountSaveChecks;
if (require.main === module) (async () => {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'account-save-browser-env-'));
  const server = createLocalServer({ envDir });
  let browser;
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_EXECUTABLE_PATH ? { executablePath: process.env.BROWSER_EXECUTABLE_PATH } : {}) });
    await runAccountSaveChecks({ browser, base: `http://127.0.0.1:${server.address().port}`, artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-browser-account-save') });
  } finally {
    await browser?.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
