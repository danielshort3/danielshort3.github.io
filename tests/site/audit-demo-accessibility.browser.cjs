/** Narrow audit regressions. Cloud requests use fixtures; no message is sent. */
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const engines = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const { createLocalServer } = require('../../build/dev');
const { settle } = require('../release/fixtures.cjs');

async function audit(page, selector, artifactDir, label, frameSelector = null) {
  const result = await new AxeBuilder({ page }).include(frameSelector ? [frameSelector, selector] : selector)
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa']).analyze();
  fs.writeFileSync(path.join(artifactDir, `${label}-axe.json`), JSON.stringify(result, null, 2));
  assert.deepEqual(result.violations.map(item => ({ id: item.id, targets: item.nodes.map(node => node.target) })), [], `${label}: scoped WCAG A/AA checks`);
}

async function checkDescription(page, inputId, descriptionId) {
  assert.equal(await page.locator(`#${inputId}`).getAttribute('aria-describedby'), descriptionId);
  const description = page.locator(`#${descriptionId}`);
  assert(await description.isVisible(), `${inputId}: privacy notice is visible beside the composer`);
  assert.match(await description.innerText(), /AWS.*Avoid private information/);
  assert.equal(await description.locator('a').getAttribute('href'), '/privacy#assistant-and-links');
}

async function runAuditDemoAccessibilityChecks({ browser, browserName = 'chromium', base, artifactDir }) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const results = [];
  const failures = [];
  for (const width of [1440, 390, 320]) {
    for (const surface of ['background-remover', 'covid', 'target', 'travel-assistant', 'site-assistant']) {
      const label = `${browserName}-${surface}-${width}`;
      const context = await browser.newContext({ viewport: { width, height: width < 600 ? 844 : 1000 },
        reducedMotion: 'reduce', serviceWorkers: 'block' });
      const page = await context.newPage();
      page.setDefaultTimeout(15000);
      const errors = [];
      const writes = [];
      page.on('pageerror', error => errors.push(error.message));
      await context.route('**/*', async route => {
        const request = route.request();
        const url = new URL(request.url());
        if (url.origin !== base) return route.abort('blockedbyclient');
        if (url.pathname === '/api/chatbot' && request.method() === 'GET') {
          return route.fulfill({ status: 200, contentType: 'application/json', body: '{"enabled":true,"turnstileSiteKey":""}' });
        }
        if (url.pathname === '/api/chatbot-demo/bedrock/status') {
          return route.fulfill({ status: 200, contentType: 'application/json',
            body: '{"status":"READY","online":true,"stage":{"message":"Fixture ready."}}' });
        }
        if (url.pathname.startsWith('/api/')) {
          if (request.method() !== 'GET') writes.push({ method: request.method(), path: url.pathname });
          return route.fulfill({ status: url.pathname.startsWith('/api/tools/') ? 200 : 503,
            contentType: 'application/json', body: '{"ok":true,"authenticated":false,"sessions":[],"error":"Blocked by audit fixture"}' });
        }
        return route.continue();
      });
      try {
        const route = {
          'background-remover': '/tools/background-remover', covid: '/covid-outbreak-demo',
          target: '/target-empty-package-demo', 'travel-assistant': '/chatbot-demo',
          'site-assistant': '/#closed'
        }[surface];
        const response = await page.goto(`${base}${route}`, { waitUntil: 'domcontentloaded' });
        assert.equal(response.status(), 200, `${label}: page identity`);
        assert((await page.title()).trim().length > 0);
        if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
        await settle(page);
        const frameSelector = ['covid', 'target', 'travel-assistant'].includes(surface) ? 'iframe.project-demo-wrapper-iframe' : null;
        const product = frameSelector ? await (await page.locator(frameSelector).elementHandle()).contentFrame() : page;
        assert(product, `${label}: the rendered demo frame is available`);
        await product.evaluate(() => document.fonts.ready);
        const auditProduct = (selector, name = label) => audit(page, selector, artifactDir, name, frameSelector);
        let scope = '#main';
        if (surface === 'background-remover') {
          assert.equal(await page.locator('#bgtool-file').getAttribute('aria-label'), 'Add photos for background removal');
          assert.equal(await page.locator('#bgtool-dropzone').getAttribute('role'), 'group');
          assert.equal(await page.locator('#bgtool-dropzone').getAttribute('tabindex'), null);
          const pick = page.locator('#bgtool-dropzone button[data-bgtool="pick"]');
          assert.equal(await pick.evaluate(node => node.tagName), 'BUTTON');
          await page.evaluate(() => {
            window.__auditPhotoPickerClicks = 0;
            document.querySelector('#bgtool-file').addEventListener('click', event => {
              event.preventDefault();
              window.__auditPhotoPickerClicks += 1;
            });
          });
          await pick.click();
          assert.equal(await page.evaluate(() => window.__auditPhotoPickerClicks), 1, 'Pointer click activates the native picker once');
          await pick.focus();
          // Firefox retains pointer modality on a programmatically refocused
          // button. Navigate away and back to prove genuine keyboard focus.
          await page.keyboard.press('Tab');
          await page.keyboard.press('Shift+Tab');
          assert(await pick.evaluate(node => document.activeElement === node), 'Native Tab navigation reaches Add photos');
          await page.keyboard.press('Enter');
          assert.equal(await page.evaluate(() => window.__auditPhotoPickerClicks), 2, 'Enter activates the native picker once');
          await page.keyboard.press('Space');
          assert.equal(await page.evaluate(() => window.__auditPhotoPickerClicks), 3, 'Space activates the native picker once');
          const focus = await pick.evaluate(node => ({ active: document.activeElement === node,
            width: parseFloat(getComputedStyle(node).outlineWidth), style: getComputedStyle(node).outlineStyle }));
          assert(focus.active && focus.width >= 2 && focus.style !== 'none', 'Native activation retains a visible keyboard focus outline');
          await auditProduct(scope);
        } else if (surface === 'covid') {
          await product.locator('#state-select:not([disabled])').waitFor();
          assert.equal(await product.locator('#map-svg').getAttribute('role'), 'group');
          const state = product.locator('.state-shape[data-state="CO"]');
          await state.focus();
          await product.locator('#map-tooltip:not([hidden])').waitFor();
          assert.equal(await state.getAttribute('aria-describedby'), 'map-tooltip');
          assert.match(await product.locator('#map-tooltip').innerText(), /Colorado/);
          await page.keyboard.press('Escape');
          assert(await product.locator('#map-tooltip').isHidden(), 'Escape dismisses a focused map tooltip');
          assert.equal(await state.getAttribute('aria-describedby'), null);
          await page.keyboard.press('Enter');
          await product.waitForFunction(() => document.querySelector('#state-select').value === 'CO');
          assert.equal(await state.evaluate(node => document.activeElement === node), true, 'Selecting a state keeps keyboard focus');
          for (const risk of ['low', 'elevated', 'high']) {
            const option = await product.locator('.state-shape').evaluateAll((nodes, value) => nodes.find(node => node.dataset.risk === value)?.dataset.state, risk);
            assert(option, `${label}: historical dataset supplies ${risk} state`);
            await product.locator('#state-select').selectOption(option);
            await product.waitForFunction(value => document.querySelector('#risk-pill').dataset.risk === value, risk);
            await auditProduct('#risk-pill', `${label}-${risk}`);
          }
          await auditProduct(scope);
        } else if (surface === 'target') {
          await product.locator('#filter-location:not([disabled])').waitFor();
          const value = await product.locator('#filter-location option').evaluateAll(nodes => nodes.find(node => node.value !== 'all')?.value);
          assert(value, 'Published target dataset has a location filter');
          await product.locator('#filter-location').selectOption(value);
          await product.locator('#reset-filters').click();
          await product.locator('#filter-location').selectOption(value);
          await product.locator('#reset-filters').focus();
          await page.keyboard.press('Enter');
          assert.equal(await product.locator('#filter-location').inputValue(), 'all', 'Reset clears the actual filter');
          assert(await product.locator('#reset-filters').evaluate(node => document.activeElement === node), 'Reset keeps keyboard focus');
          await auditProduct(scope);
        } else if (surface === 'travel-assistant') {
          await product.locator('#chat-connection-pill[data-state="ok"]').waitFor();
          await checkDescription(product, 'regular-prompt', 'regular-chat-privacy');
          await product.locator('#regular-prompt').fill('Safe local draft only.');
          await auditProduct('.chat-shell--regular .chat-composer', `${label}-regular`);
          await page.screenshot({ path: path.join(artifactDir, `${label}-regular.png`) });
          await product.locator('#chat-settings > summary').click();
          await product.locator('#popup-view-button').click();
          await product.locator('#popup-launcher').click();
          await checkDescription(product, 'popup-prompt', 'popup-chat-privacy');
          assert.equal(await product.locator('#popup-prompt').inputValue(), 'Safe local draft only.', 'Changing views preserves the existing draft');
          await auditProduct('#popup-window .chat-composer', `${label}-popup`);
          scope = '#popup-window';
        } else {
          // The retired launcher is not in the normal shell. Mount its existing
          // source explicitly to verify its optional, non-production surface.
          await page.addStyleTag({ url: `${base}/css/components/site-chatbot.css` });
          await page.addScriptTag({ url: `${base}/js/chatbot/site-chatbot.js` });
          await page.locator('[data-site-chatbot][data-enabled="true"]').waitFor();
          await page.locator('.site-chatbot__launcher').click();
          await checkDescription(page, 'site-chatbot-message', 'site-chatbot-privacy');
          await page.locator('#site-chatbot-message').fill('Safe local draft only.');
          await audit(page, '.site-chatbot__form', artifactDir, label);
          const states = [];
          const checkGeometry = async name => {
            await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
            const geometry = await page.evaluate(() => {
              const rect = selector => {
                const node = document.querySelector(selector);
                if (!node || !node.getBoundingClientRect().height) return null;
                const box = node.getBoundingClientRect();
                return { top: box.top, bottom: box.bottom };
              };
              return { panel: rect('.site-chatbot__panel'), input: rect('#site-chatbot-message'),
                masthead: rect('.mobile-site-masthead'), footer: rect('[data-site-shell-footer]'),
                keyboard: document.querySelector('[data-site-chatbot]').dataset.keyboard, viewport: innerHeight };
            });
            assert(geometry.panel.top >= Math.max(0, (geometry.masthead?.bottom || 0) + (geometry.masthead ? 8 : 0)), `${label}/${name}: optional assistant clears the current masthead`);
            assert(geometry.panel.bottom <= geometry.viewport, `${label}/${name}: optional assistant stays in the viewport`);
            assert(geometry.input.top >= geometry.panel.top && geometry.input.bottom <= geometry.panel.bottom, `${label}/${name}: input is contained and visible`);
            await checkDescription(page, 'site-chatbot-message', 'site-chatbot-privacy');
            states.push({ name, geometry });
            await page.screenshot({ path: path.join(artifactDir, `${label}-${name}.png`) });
          };
          if (width < 600) await page.waitForFunction(() => document.querySelector('[data-site-chatbot]').dataset.keyboard === 'true');
          await checkGeometry('input-focused-collapsed');
          await page.locator('.site-chatbot__header-expand').click();
          await page.waitForFunction(() => document.querySelector('[data-site-chatbot]').dataset.expanded === 'true');
          if (width < 600) await page.waitForFunction(() => document.querySelector('[data-site-chatbot]').dataset.keyboard === 'false');
          await checkGeometry('expanded');
          await page.locator('#site-chatbot-message').focus();
          if (width < 600) await page.waitForFunction(() => document.querySelector('[data-site-chatbot]').dataset.keyboard === 'true');
          await checkGeometry('expanded-input-focused');
          await page.locator('.site-chatbot__header-toggle').click();
          if (width < 600) await page.waitForFunction(() => document.querySelector('[data-site-chatbot]').dataset.keyboard === 'false');
          await checkGeometry('collapsed');
          fs.writeFileSync(path.join(artifactDir, `${label}-geometry.json`), JSON.stringify({ states }, null, 2));
          assert.equal(await page.locator('#site-chatbot-message').inputValue(), 'Safe local draft only.', 'Expanding and collapsing preserves the draft');
          scope = '.site-chatbot__panel';
        }
        assert((await product.locator(scope).innerText()).trim().length > 0, `${label}: meaningful product content`);
        assert(await product.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${label}: no demo horizontal overflow`);
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1), `${label}: no horizontal overflow`);
        await page.screenshot({ path: path.join(artifactDir, `${label}.png`) });
        assert.deepEqual(errors, [], `${label}: no page errors`);
        assert.deepEqual(writes, [], `${label}: no cloud writes or inference`);
        results.push({ label, url: page.url(), fixture: surface.includes('assistant'), errors, writes });
        console.log(`Audit demo accessibility passed: ${label}`);
      } catch (error) {
        await page.screenshot({ path: path.join(artifactDir, `${label}-failure.png`) }).catch(() => {});
        fs.writeFileSync(path.join(artifactDir, `${label}-failure.json`), JSON.stringify({ error: error.message, errors, writes }, null, 2));
        failures.push(`${label}: ${error.message}`);
        results.push({ label, url: page.url(), error: error.message, errors, writes });
        console.error(`Audit demo accessibility failed: ${label}: ${error.message}`);
      } finally { await context.close(); }
    }
  }
  fs.writeFileSync(path.join(artifactDir, `audit-demo-accessibility-${browserName}.json`), JSON.stringify(results, null, 2));
  assert.deepEqual(failures, [], `${browserName}: all affected flows pass`);
  return results;
}

async function main() {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'audit-demo-accessibility-env-'));
  const server = createLocalServer({ envDir });
  const failures = [];
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
    for (const browserName of ['chromium', 'firefox', 'webkit']) {
      const browser = await engines[browserName].launch({ headless: true });
      try {
        await runAuditDemoAccessibilityChecks({ browser, browserName, base: `http://127.0.0.1:${server.address().port}`,
          artifactDir: process.env.BROWSER_ARTIFACT_DIR || path.join(os.tmpdir(), 'site-audit-demo-accessibility') });
      } catch (error) { failures.push(error.message); }
      finally { await browser.close(); }
    }
  } finally {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
  assert.deepEqual(failures, [], 'All engines pass the audit flows');
}

module.exports = runAuditDemoAccessibilityChecks;
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
