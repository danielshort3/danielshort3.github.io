#!/usr/bin/env node
'use strict';

/*
  Unified site build runner.
  Keeps `npm run build` output consistent and avoids piecemeal npm-script chaining.
*/

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');

function log(line) {
  process.stdout.write(`[build] ${line}\n`);
}

function logError(line) {
  process.stderr.write(`[build] ${line}\n`);
}

function formatDuration(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n) || n < 0) return '';
  if (n < 1000) return `${Math.round(n)}ms`;
  if (n < 60_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}s`;
  const minutes = Math.floor(n / 60_000);
  const seconds = Math.round((n - minutes * 60_000) / 1000);
  return `${minutes}m${String(seconds).padStart(2, '0')}s`;
}

function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n < 0) return '';
  if (n < 1024) return `${n}B`;
  const kb = n / 1024;
  if (kb < 1024) return `${kb.toFixed(kb < 10 ? 1 : 0)}KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(mb < 10 ? 1 : 0)}MB`;
  const gb = mb / 1024;
  return `${gb.toFixed(1)}GB`;
}

function readJson(filePath) {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function safeStat(filePath) {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

function countFilesRecursive(dirPath) {
  let count = 0;
  const stack = [dirPath];
  while (stack.length) {
    const current = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch {
      continue;
    }
    entries.forEach((entry) => {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        return;
      }
      if (entry.isFile()) count++;
    });
  }
  return count;
}

function runNodeScript(scriptRelPath, options = {}) {
  const scriptPath = path.join(root, scriptRelPath);
  const started = Date.now();
  const result = spawnSync(process.execPath, [scriptPath], {
    cwd: root,
    env: { ...process.env },
    encoding: 'utf8',
    stdio: options.verbose ? 'inherit' : ['ignore', 'pipe', 'pipe']
  });

  const durationMs = Date.now() - started;

  if (result.error) {
    const err = new Error(`Failed to run ${scriptRelPath}: ${result.error.message}`);
    err.cause = result.error;
    throw err;
  }

  if (result.status !== 0) {
    const err = new Error(`Build step failed: ${scriptRelPath} (exit ${result.status})`);
    err.exitCode = result.status;
    err.stdout = String(result.stdout || '');
    err.stderr = String(result.stderr || '');
    throw err;
  }

  return {
    durationMs,
    stdout: String(result.stdout || ''),
    stderr: String(result.stderr || '')
  };
}

function logStep(label, durationMs, detail) {
  const padded = String(label).padEnd(22, ' ');
  const time = formatDuration(durationMs).padStart(6, ' ');
  const suffix = detail ? `  ${detail}` : '';
  log(`${padded} OK  ${time}${suffix}`);
}

function main() {
  const args = new Set(process.argv.slice(2));
  const verbose = args.has('--verbose') || args.has('-v');
  const started = Date.now();

  log('Starting site build');

  try {
    // 1) CSS bundle (css/ -> dist/)
    const cssStep = runNodeScript(path.join('build', 'build-css.js'), { verbose });
    const cssManifestPath = path.join(root, 'dist', 'styles-manifest.json');
    const manifest = readJson(cssManifestPath);
    const cssFile = manifest && typeof manifest.file === 'string' ? path.join('dist', manifest.file) : 'dist/styles.[hash].css';
    const cssStats = safeStat(path.join(root, cssFile));
    const cssDetail = cssStats ? `${cssFile} (${formatBytes(cssStats.size)})` : cssFile;
    logStep('css', cssStep.durationMs, cssDetail);

    // 2) UTM Batch Builder bundle (src/ -> dist/)
    const utmStep = runNodeScript(path.join('build', 'build-utm-batch-builder.js'), { verbose });
    const utmMain = path.join(root, 'dist', 'utm-batch-builder.js');
    const utmWorker = path.join(root, 'dist', 'utm-batch-builder.worker.js');
    const utmMainStat = safeStat(utmMain);
    const utmWorkerStat = safeStat(utmWorker);
    const utmPieces = [];
    utmPieces.push(`dist/utm-batch-builder.js${utmMainStat ? ` (${formatBytes(utmMainStat.size)})` : ''}`);
    utmPieces.push(`dist/utm-batch-builder.worker.js${utmWorkerStat ? ` (${formatBytes(utmWorkerStat.size)})` : ''}`);
    logStep('utm-batch-builder', utmStep.durationMs, utmPieces.join(', '));

    // 3) Project pages (js/portfolio -> pages/portfolio + sitemap.xml)
    const projectsStep = runNodeScript(path.join('build', 'generate-project-pages.js'), { verbose });
    const portfolioDir = path.join(root, 'pages', 'portfolio');
    const projectPages = fs.existsSync(portfolioDir)
      ? fs.readdirSync(portfolioDir).filter((name) => name.endsWith('.html')).length
      : 0;
    logStep('projects', projectsStep.durationMs, `pages/portfolio (${projectPages} pages), sitemap.xml`);

    // 4) Search index (sitemap.xml + pages -> dist/)
    const searchIndexStep = runNodeScript(path.join('build', 'generate-search-index.js'), { verbose });
    const searchIndexPath = path.join(root, 'dist', 'search-index.json');
    const searchIndexStat = safeStat(searchIndexPath);
    const searchIndexDetail = searchIndexStat ? `dist/search-index.json (${formatBytes(searchIndexStat.size)})` : 'dist/search-index.json';
    logStep('search-index', searchIndexStep.durationMs, searchIndexDetail);

    // 5) Shortlinks destinations manifest (site HTML + vercel.json -> dist/)
    const shortlinksStep = runNodeScript(path.join('build', 'generate-shortlinks-destinations.js'), { verbose });
    const destinationsPath = path.join(root, 'dist', 'shortlinks-destinations.json');
    const destinations = readJson(destinationsPath);
    const destinationsCount = destinations && Array.isArray(destinations.pages) ? destinations.pages.length : null;
    const destinationsStat = safeStat(destinationsPath);
    const destinationsDetailParts = ['dist/shortlinks-destinations.json'];
    if (destinationsCount != null) destinationsDetailParts.push(`(${destinationsCount} destinations)`);
    if (destinationsStat) destinationsDetailParts.push(`${formatBytes(destinationsStat.size)}`);
    logStep('shortlinks', shortlinksStep.durationMs, destinationsDetailParts.join(' '));

    // 6) Shared header/nav (build-time injected)
    const headerStep = runNodeScript(path.join('build', 'inject-header.js'), { verbose });
    logStep('header', headerStep.durationMs);

    // 7) Shared footer (build-time injected)
    const footerStep = runNodeScript(path.join('build', 'inject-footer.js'), { verbose });
    logStep('footer', footerStep.durationMs);

    // 8) Shared head metadata (build-time injected)
    const metaStep = runNodeScript(path.join('build', 'inject-head-metadata.js'), { verbose });
    logStep('head-metadata', metaStep.durationMs);

    // 9) Keep root HTML copies in sync with /pages
    const syncStep = runNodeScript(path.join('build', 'sync-root-pages.js'), { verbose });
    logStep('sync-root-pages', syncStep.durationMs);

    // 10) Public output (deployable mirror)
    const publicStep = runNodeScript(path.join('build', 'copy-to-public.js'), { verbose });
    const publicDir = path.join(root, 'public');
    const publicFiles = countFilesRecursive(publicDir);
    logStep('public', publicStep.durationMs, `public/ (${publicFiles} files)`);

    log(`Done in ${formatDuration(Date.now() - started)}`);
  } catch (err) {
    logError(String(err && err.message ? err.message : err));
    if (err && err.stdout && String(err.stdout).trim()) {
      process.stderr.write(String(err.stdout).trimEnd() + '\n');
    }
    if (err && err.stderr && String(err.stderr).trim()) {
      process.stderr.write(String(err.stderr).trimEnd() + '\n');
    }
    process.exitCode = err && err.exitCode ? err.exitCode : 1;
  }
}

main();
