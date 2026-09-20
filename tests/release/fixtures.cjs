'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { test: base, expect } = require('@playwright/test');
const pixel = fs.readFileSync(path.join(__dirname, 'digit-fixture.png')).toString('base64');
async function isolateRequests(context, baseURL) {
  await context.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== baseURL) return route.abort('blockedbyclient');
    if (url.pathname === '/api/contact') return route.fulfill({ status: 400, contentType: 'application/json', body: '{"ok":false,"code":"CONTACT_REJECTED"}' });
    if (url.pathname.startsWith('/api/tools/')) return route.fulfill({ status: 200, contentType: 'application/json', body: '{"authenticated":false,"ok":true,"sessions":[],"recentSessions":[],"tools":[]}' });
    if (url.pathname.startsWith('/api/demos/')) {
      const body = url.pathname.endsWith('/generate')
        ? { rows: 6, cols: 6, latent_dim: 20, images: Array.from({ length: 6 }, () => Array(6).fill(pixel)) }
        : { status: 'ready', model_loaded: true };
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    }
    if (url.pathname.startsWith('/api/')) return route.fulfill({ status: 503, contentType: 'application/json', body: '{"ok":false,"error":"Unavailable in the release fixture"}' });
    return route.continue();
  });
}
const test = base.extend({
  page: async ({ page, context, baseURL }, use) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await isolateRequests(context, baseURL);
    await use(page);
    expect(errors, 'No uncaught first-party browser errors').toEqual([]);
  }
});
async function ready(page, route) {
  await page.goto(route, { waitUntil: 'domcontentloaded' });
  if (await page.locator('#pcz-reject').isVisible()) await page.locator('#pcz-reject').click();
  await settle(page);
}
async function settle(page) {
  await page.waitForFunction(() => document.readyState !== 'loading' && !window.SiteNavigation?.isNavigating?.() && !window.SiteFrame?.root()?.matches('.site-frame--held, .site-frame--moving'));
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}
async function compare(page) {
  await page.locator('#textcompare-original').fill('Publish the draft on Monday.');
  await page.locator('#textcompare-revised').fill('Publish the final draft on Tuesday.');
  await page.locator('#textcompare-form button[type="submit"]').click();
  await expect(page.locator('#textcompare-copy')).toBeEnabled();
}
async function mockAccount(page) {
  await page.waitForFunction(() => window.__toolsAccountUiController);
  await page.evaluate(() => {
    window.ToolsAuth = { ...window.ToolsAuth, getAuth: () => ({ test: true }), authIsValid: () => true, getUser: () => ({ sub: 'release-fixture', name: 'Local reviewer', email: 'review@example.test' }), ensureFreshAuth: async () => null };
    window.ToolsState = { ...window.ToolsState, getDashboard: async () => ({ recentSessions: [], tools: [] }), listSessions: async () => ({ sessions: [] }), logActivity: async () => ({}) };
    document.dispatchEvent(new CustomEvent('tools:auth-changed'));
  });
  await page.locator('[data-tools-action="open-account"]').click();
  await expect(page.getByRole('dialog', { name: 'Account', exact: true })).toBeVisible();
}
module.exports = { test, expect, ready, settle, compare, mockAccount, isolateRequests };
