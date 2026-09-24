'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { execFileSync } = require('node:child_process');
const { gzipSync } = require('node:zlib');
const { chromium } = require('playwright');
const { createLocalServer } = require('./dev');
const { median, compareLighthouseRoute } = require('./lighthouse-regression.cjs');

const ROUTES = ['/', '/tools/text-compare', '/portfolio/website', '/minesweeper-demo'];
const RUNS_PER_ROUTE = 3;
const METRICS = ['lcp', 'cls', 'tbt', 'requests', 'bytes', 'benchmarkIndex'];

function getReferenceDirectory(args) {
  const index = args.indexOf('--reference-dir');
  if (index < 0) return null;
  if (!args[index + 1] || args[index + 1].startsWith('--')) {
    throw Error('--reference-dir needs a built reference checkout');
  }
  return path.resolve(args[index + 1]);
}

async function startDelivery(createServer) {
  const envDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ds-lab-env-'));
  const server = createServer({ envDir });
  let delivery;
  try {
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    // Match the CDN's compressed text delivery without using production services.
    delivery = http.createServer((req, res) => {
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
    return {
      base: `http://127.0.0.1:${delivery.address().port}`,
      async close() {
        delivery.closeAllConnections();
        await new Promise(resolve => delivery.close(resolve));
        server.closeAllConnections();
        await new Promise(resolve => server.close(resolve));
        fs.rmdirSync(envDir);
      }
    };
  } catch (error) {
    if (delivery?.listening) {
      delivery.closeAllConnections();
      await new Promise(resolve => delivery.close(resolve));
    }
    if (server.listening) {
      server.closeAllConnections();
      await new Promise(resolve => server.close(resolve));
    }
    fs.rmdirSync(envDir);
    throw error;
  }
}

function medianRun(runs) {
  return Object.fromEntries(METRICS.map(key => [key, median(runs.map(run => run[key]))]));
}

async function main() {
  const args = process.argv.slice(2);
  const updatingBaseline = args.includes('--update-baseline');
  const referenceDir = getReferenceDirectory(args);
  if (updatingBaseline && referenceDir) throw Error('Update the reviewed baseline without a paired reference');
  if (process.env.CI === 'true' && !updatingBaseline && !referenceDir) {
    throw Error('CI Lighthouse checks require --reference-dir');
  }
  const baselinePath = path.resolve('tests/site/baselines/lighthouse.json');
  const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8'));
  if (referenceDir) {
    if (!/^[a-f0-9]{40}$/.test(baseline.referenceCommit || '')) throw Error('Reviewed baseline needs a pinned referenceCommit');
    const actualCommit = execFileSync('git', ['-C', referenceDir, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
    if (actualCommit !== baseline.referenceCommit) throw Error(`Reference checkout is ${actualCommit}, expected ${baseline.referenceCommit}`);
    if (!fs.existsSync(path.join(referenceDir, 'public/index.html'))) throw Error('Reference checkout needs a completed website build');
  }

  const { default: lighthouse } = await import('lighthouse');
  const { launch } = await import('chrome-launcher');
  const output = path.resolve('tmp/quality/lighthouse');
  fs.mkdirSync(output, { recursive: true });
  const hosts = [];
  let chrome;
  try {
    const candidate = await startDelivery(createLocalServer);
    hosts.push(candidate);
    let reference;
    if (referenceDir) {
      const referenceServer = require(path.join(referenceDir, 'build/dev.js')).createLocalServer;
      reference = await startDelivery(referenceServer);
      hosts.push(reference);
    }
    chrome = await launch({ chromePath: chromium.executablePath(), chromeFlags: ['--headless=new', '--no-sandbox', '--disable-dev-shm-usage'] });
    const results = [];
    async function measure(base, route, label, index) {
      const { lhr } = await lighthouse(base + route, { port: chrome.port, logLevel: 'error', output: 'json', onlyCategories: ['performance'] }, {
          extends: 'lighthouse:default', settings: { formFactor: 'mobile', blockedUrlPatterns: ['https://*'], maxWaitForLoad: 30000 }
      });
      if (lhr.runtimeError) throw Error(`${route} ${label}: ${lhr.runtimeError.message}`);
      const audits = lhr.audits;
      const requests = audits['network-requests'].details.items;
      const run = {
        lcp: audits['largest-contentful-paint'].numericValue,
        cls: audits['cumulative-layout-shift'].numericValue,
        tbt: audits['total-blocking-time'].numericValue,
        requests: requests.length,
        bytes: requests.reduce((sum, request) => sum + (request.transferSize || 0), 0),
        benchmarkIndex: lhr.environment.benchmarkIndex
      };
      const slug = route.replace(/\W/g, '-') || 'home';
      const suffix = label === 'candidate' ? '' : '-reference';
      fs.writeFileSync(path.join(output, `${slug}${suffix}-${index + 1}.json`), JSON.stringify(lhr));
      console.log(route, `${label} run ${index + 1}`, JSON.stringify(run));
      return run;
    }

    for (const route of ROUTES) {
      const runs = [];
      const referenceRuns = [];
      for (let i = 0; i < RUNS_PER_ROUTE; i++) {
        if (reference && i % 2 === 0) referenceRuns.push(await measure(reference.base, route, 'reference', i));
        runs.push(await measure(candidate.base, route, 'candidate', i));
        if (reference && i % 2 === 1) referenceRuns.push(await measure(reference.base, route, 'reference', i));
      }
      const result = { route, runs, median: medianRun(runs) };
      if (reference) {
        result.referenceRuns = referenceRuns;
        result.referenceMedian = medianRun(referenceRuns);
      }
      results.push(result);
    }
    const report = {
      generatedAt: new Date().toISOString(),
      lighthouse: require('lighthouse/package.json').version,
      browser: await (await fetch(`http://localhost:${chrome.port}/json/version`)).json(),
      referenceCommit: reference ? baseline.referenceCommit : null,
      note: reference
        ? 'Three interleaved mobile lab pairs per route. TBT is compared with a pinned same-run reference; raw TBT is not real-user INP. External services and APIs are blocked.'
        : 'Three mobile lab runs; TBT is not real-user INP. External services and APIs are blocked.',
      results
    };
    if (updatingBaseline) {
      const referenceCommit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
      fs.writeFileSync(baselinePath, JSON.stringify({
        lighthouse: report.lighthouse,
        referenceCommit,
        results: results.map(({ route, median: values }) => ({
          route,
          median: Object.fromEntries(METRICS.filter(key => key !== 'benchmarkIndex').map(key => [key, values[key]]))
        }))
      }, null, 2));
    } else {
      const failures = [];
      for (const result of results) {
        const before = baseline.results.find(item => item.route === result.route)?.median;
        result.comparison = compareLighthouseRoute(before, result);
        failures.push(...result.comparison.failures);
      }
      fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(report, null, 2));
      if (failures.length) throw Error(failures.join('\n'));
    }
    if (updatingBaseline) fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(report, null, 2));
  } finally {
    if (chrome) {
      try { await chrome.kill(); } catch (error) { console.warn('Chrome cleanup:', error.code || error.message); }
    }
    for (const host of hosts.reverse()) await host.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
