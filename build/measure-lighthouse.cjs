'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { gzipSync } = require('node:zlib');
const { chromium } = require('playwright');
const { createLocalServer } = require('./dev');
const { ABSOLUTE, assertBudgets } = require('./lighthouse-budgets.cjs');

async function main() {
  const { default: lighthouse } = await import('lighthouse');
  const { launch } = await import('chrome-launcher');
  const output = path.resolve('tmp/quality/lighthouse');
  fs.mkdirSync(output, { recursive: true });
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ds-lab-env-'));
  const server = createLocalServer({ envDir });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  // Match the CDN's compressed text delivery, without changing dev-server
  // semantics or using production services during repeatable local audits.
  const delivery = http.createServer((req, res) => {
    if (req.url.startsWith('/api/')) { res.writeHead(503); res.end('{}'); return; }
    const upstream = http.request({ hostname: '127.0.0.1', port: server.address().port, path: req.url, method: req.method }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(chunk));
      response.on('end', () => {
        let body = Buffer.concat(chunks);
        const headers = { ...response.headers };
        if (/text\/|javascript|json|svg/.test(headers['content-type'] || '') && body.length > 1024) {
          body = gzipSync(body);
          headers['content-encoding'] = 'gzip';
          headers.vary = 'Accept-Encoding';
        }
        headers['content-length'] = body.length;
        res.writeHead(response.statusCode, headers); res.end(body);
      });
    });
    upstream.on('error', () => { res.writeHead(502); res.end(); }); upstream.end();
  });
  await new Promise(resolve => delivery.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${delivery.address().port}`;
  const chrome = await launch({ chromePath: chromium.executablePath(), chromeFlags: ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage'] });
  const results = [];
  const median = values => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
  try {
    for (const route of ['/', '/tools/text-compare', '/portfolio/website', '/minesweeper-demo']) {
      const runs = [];
      for (let i = 0; i < 3; i++) {
        const { lhr } = await lighthouse(base + route, { port: chrome.port, logLevel: 'error', output: 'json', onlyCategories: ['performance'] }, {
          extends: 'lighthouse:default', settings: { formFactor: 'mobile', blockedUrlPatterns: ['https://*'], maxWaitForLoad: 30000 }
        });
        if (lhr.runtimeError) throw Error(`${route}: ${lhr.runtimeError.message}`);
        const audits = lhr.audits;
        const requests = audits['network-requests'].details.items;
        const run = {
          lcp: audits['largest-contentful-paint'].numericValue,
          cls: audits['cumulative-layout-shift'].numericValue,
          tbt: audits['total-blocking-time'].numericValue,
          requests: requests.length,
          bytes: requests.reduce((sum, request) => sum + (request.transferSize || 0), 0)
        };
        runs.push(run);
        const slug = route.replace(/\W/g, '-') || 'home';
        fs.writeFileSync(path.join(output, `${slug}-${i + 1}.json`), JSON.stringify(lhr));
        console.log(route, `run ${i + 1}`, JSON.stringify(run));
      }
      results.push({ route, runs, median: Object.fromEntries(Object.keys(runs[0]).map(key => [key, median(runs.map(run => run[key]))])) });
    }
    const report = { absoluteBudgets: ABSOLUTE, generatedAt: new Date().toISOString(), lighthouse: require('lighthouse/package.json').version, browser: await (await fetch(`http://localhost:${chrome.port}/json/version`)).json(), note: 'Three mobile lab runs; TBT is not real-user INP. External services blocked; no cloud inference or contact submissions.', results };
    fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(report, null, 2));
    const baselinePath = path.resolve('tests/site/baselines/lighthouse.json');
    if (process.argv.includes('--update-baseline')) {
      fs.mkdirSync(path.dirname(baselinePath), { recursive: true });
      fs.writeFileSync(baselinePath, JSON.stringify({ lighthouse: report.lighthouse, results: results.map(({ route, median }) => ({ route, median })) }, null, 2));
    } else {
      const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
      for (const result of results) {
        const before = baseline.results.find(item => item.route === result.route)?.median;
        assertBudgets(result, before);
      }
    }
  } finally {
    try { await chrome.kill(); } catch (error) { console.warn('Chrome cleanup:', error.code || error.message); }
    delivery.closeAllConnections();
    await new Promise(resolve => delivery.close(resolve));
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmdirSync(envDir);
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
