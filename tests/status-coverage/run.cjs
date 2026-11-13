'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const { parseArgs } = require('node:util');
const { chromium } = require('playwright');
const { readCatalog } = require('./catalog.cjs');
const { startServer } = require('./static-server.cjs');
const { tools, unknown } = require('./tools.cjs');
const projects = require('./projects.cjs');
const games = require('./games.cjs');

const implementations = { tool: tools, project: projects, game: games };
const checkTimeout = 75_000;
const reportLimit = 70;

async function isolateContext(context, base) {
  await context.route('**/*', (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== base) return route.abort('blockedbyclient');
    if (/^\/api\/tools\/(?:auth|activity|dashboard|state|me)(?:\/|$)/.test(url.pathname)) {
      return route.fulfill({ json: { ok: true, authenticated: false, recentSessions: [], tools: [], sessions: [] } });
    }
    if (/^\/api(?:\/|$)/.test(url.pathname)) {
      return route.fulfill({ status: 503, json: { ok: false, error: 'No configured fixture' } });
    }
    if (!['GET', 'HEAD'].includes(request.method())) return route.abort('blockedbyclient');
    return route.continue();
  });
}

function isolatedBrowser(browser, base) {
  const contexts = new Set();
  return {
    async newContext(options) {
      const context = await browser.newContext({
        viewport: { width: 1360, height: 900 }, reducedMotion: 'reduce', ...options,
        serviceWorkers: 'block', acceptDownloads: true
      });
      await isolateContext(context, base);
      contexts.add(context);
      context.once('close', () => contexts.delete(context));
      return context;
    },
    async closeContexts() {
      await Promise.allSettled([...contexts].map((context) => context.close()));
    }
  };
}

function failureCategory(error) {
  if (error?.name === 'TimeoutError') return 'The expected browser result did not appear within the check deadline.';
  if (error?.code === 'ERR_ASSERTION' || error?.matcherResult) return 'The local feature output did not match the expected fixture.';
  return 'The local feature check could not complete.';
}

async function runItem({ browser, base, item, artifactDir }) {
  const started = Date.now();
  const isolated = isolatedBrowser(browser, base);
  let routeStatus = 'unknown';
  let result;
  let deadline;
  try {
    const context = await isolated.newContext();
    const page = await context.newPage();
    page.setDefaultTimeout(8_000);
    const probe = await page.request.get(base + item.route, { timeout: 6_000, maxRedirects: 0 });
    routeStatus = probe.status() === 200 && /text\/html/.test(probe.headers()['content-type'] || '') ? 'pass' : 'fail';
    await probe.dispose();
    const category = implementations[item.category];
    const check = category[item.slug] || category.fallback;
    if (routeStatus === 'fail') {
      result = { status: 'fail', reason: 'The built catalog route did not return its local page.' };
    } else if (!check) {
      result = unknown('This catalog item has no deterministic feature fixture yet.');
    } else {
      result = await Promise.race([
        check({ browser: isolated, context, page, base, item, artifactDir }),
        new Promise((_, reject) => {
          deadline = setTimeout(() => reject(Object.assign(new Error('Check deadline'), { name: 'TimeoutError' })), checkTimeout);
        })
      ]);
    }
  } catch (error) {
    result = { status: 'fail', reason: failureCategory(error) };
    if (process.env.STATUS_COVERAGE_DEBUG === '1') {
      process.stderr.write(item.id + ' ' + String(error?.stack || error) + '\n');
    }
  } finally {
    clearTimeout(deadline);
    await isolated.closeContexts();
  }
  assert(['pass', 'fail', 'unknown'].includes(result?.status) &&
    typeof result.reason === 'string' && result.reason.length <= 160);
  return {
    id: item.id, category: item.category, route: item.route, action: item.action,
    status: result.status, method: 'browser', routeStatus, durationMs: Date.now() - started,
    checkedAt: new Date().toISOString(), reason: result.reason
  };
}

async function run({ siteRoot, only = [] }) {
  const all = readCatalog(siteRoot);
  assert(all.length > 0 && all.length <= reportLimit, 'Unexpected catalog size');
  const requested = new Set(only);
  const items = requested.size ? all.filter(({ id }) => requested.has(id)) : all;
  assert(!requested.size || requested.size === items.length, 'Unknown requested coverage check');
  const artifactDir = fs.mkdtempSync(path.join(os.tmpdir(), 'website-coverage-'));
  let server;
  let browser;
  try {
    server = await startServer(siteRoot);
    browser = await chromium.launch({ args: ['--no-proxy-server'] });
    const checks = [];
    for (const item of items) {
      checks.push(await runItem({ browser, base: server.base, item, artifactDir }));
    }
    const summary = Object.fromEntries(['pass', 'fail', 'unknown'].map((status) =>
      [status, checks.filter((check) => check.status === status).length]));
    return { schemaVersion: 1, environment: 'local', checkedAt: new Date().toISOString(), checks, summary };
  } finally {
    await browser?.close();
    await server?.close();
    for (const name of fs.readdirSync(artifactDir)) {
      const file = path.join(artifactDir, name);
      if (fs.lstatSync(file).isFile()) fs.unlinkSync(file);
    }
    fs.rmdirSync(artifactDir);
  }
}

module.exports = { run };

if (require.main === module) {
  (async () => {
    const { values } = parseArgs({ options: {
      'site-root': { type: 'string' },
      report: { type: 'string' },
      only: { type: 'string', multiple: true }
    } });
    const siteRoot = path.resolve(values['site-root'] || path.join(__dirname, '../..'));
    assert(!values.report || (path.isAbsolute(values.report) && !fs.existsSync(values.report)),
      'Use a new absolute report path');
    const report = await run({ siteRoot, only: values.only });
    const json = JSON.stringify(report, null, 2) + '\n';
    if (values.report) fs.writeFileSync(values.report, json, { flag: 'wx' });
    if (!values.report) process.stdout.write(json);
    else process.stdout.write(JSON.stringify(report.summary) + '\n');
    if (report.summary.fail) process.exitCode = 1;
  })().catch((error) => {
    process.stderr.write('Local coverage could not run. Check the built site, browser, catalog, and output path.\n');
    if (process.env.STATUS_COVERAGE_DEBUG === '1') process.stderr.write(String(error?.stack || error) + '\n');
    process.exitCode = 2;
  });
}
