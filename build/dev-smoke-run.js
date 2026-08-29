'use strict';

/*
  Orchestrator:
   1. Starts the local dev server (build/dev.js --no-watch) on a fixed port.
   2. Waits until it serves the site.
   3. Runs the GA4 smoke test (build/ga4-smoke.js).
   4. Tears the dev server down and mirrors the smoke test exit code.
*/

const { spawn, execFileSync } = require('child_process');
const http = require('http');
const path = require('path');

const PORT = Number(process.env.SMOKE_PORT || '45102');
const BASE = `http://127.0.0.1:${PORT}`;

function log(msg) {
  process.stdout.write(`[smoke-run] ${msg}\n`);
}

function pollServer(url, timeoutMs) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    function attempt() {
      const req = http.get(url, (res) => {
        res.resume();
        res.on('end', () => resolve(true));
      });
      req.on('error', () => {
        if (Date.now() - start > timeoutMs) reject(new Error('dev server did not come up within ' + timeoutMs + 'ms'));
        else setTimeout(attempt, 500);
      });
      req.end();
    }
    attempt();
  });
}

function killChild(child, sig) {
  try { if (child && child.exitCode === null) child.kill(sig); } catch (err) {}
}

/* CI / headless machines often have no browser. If Chrome is unavailable,
   report SKIP and exit 0 so "npm test" keeps passing in that environment. */
function findChrome() {
  const { spawnSync } = require('child_process');
  const candidates = [
    process.env.CHROME_PATH,
    path.join(process.env.ProgramFiles || 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(process.env.LOCALAPPDATA || '', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    '/usr/bin/google-chrome',
    '/usr/bin/chromium-browser',
    'google-chrome'
  ].filter(Boolean).filter((candidate) => {
    try { return spawnSync(candidate, ['--version'], { timeout: 5000, stdio: 'ignore' }).status === 0; }
    catch (err) { return false; }
  });
  return candidates[0] || null;
}

function main() {
  const chrome = findChrome();
  if (!chrome) {
    log('SKIP: Chromium is not available in this environment; analytics smoke test skipped.');
    process.exit(0);
  }

  const dev = spawn(process.execPath, [path.join(__dirname, 'dev.js'), '--no-watch', '--port', String(PORT)], {
    cwd: path.resolve(__dirname, '..'),
    stdio: ['ignore', 'pipe', 'pipe']
  });

  const onChunk = (chunk) => process.stdout.write(chunk);
  dev.stdout.on('data', onChunk);
  dev.stderr.on('data', onChunk);

  const spawnError = new Promise((_, reject) => {
    dev.on('error', (err) => reject(new Error('failed to spawn dev server: ' + err.message)));
  });

  const exitEarly = new Promise((resolve) => {
    dev.on('close', (code) => resolve(code));
  });

  return Promise.race([
    (async () => {
      await pollServer(BASE + '/index.html', 180000);
      log('dev server ready at ' + BASE);
      const smoke = path.join(__dirname, 'ga4-smoke.js');
      execFileSync(process.execPath, [smoke], {
        stdio: 'inherit',
        env: { ...process.env, BASE_URL: BASE, CDP_PORT: process.env.CDP_PORT || '9333' }
      });
      process.exit(0);
    })(),
    spawnError,
    exitEarly.then((code) => {
      if (code !== null && code !== 0) throw new Error(`dev server exited early with code ${code}`);
      throw new Error('dev server exited before becoming ready');
    })
  ]);
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error('[smoke-run] FAILED: ' + (err && err.message ? err.message : err));
    process.exit(1);
  }
);
