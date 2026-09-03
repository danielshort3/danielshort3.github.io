#!/usr/bin/env node
'use strict';

/*
  Sync select root-level HTML pages from their `/pages/` equivalents.

  This repo serves clean URLs via Vercel rewrites to `/pages/*`, but also
  keeps root-level copies (e.g. for direct/static hosting). This script
  prevents those pairs from drifting over time.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const TRANSIENT_WRITE_ERROR_CODES = new Set(['EACCES', 'EBUSY', 'EPERM', 'UNKNOWN']);
const WRITE_RETRY_DELAYS_MS = Object.freeze([25, 50, 100, 200, 400, 800]);
const writeRetrySignal = new Int32Array(new SharedArrayBuffer(4));

const PAIRS = [
  { from: 'pages/contact.html', to: 'contact.html' },
  { from: 'pages/privacy.html', to: 'privacy.html' },
  { from: 'pages/sitemap.html', to: 'sitemap.html' }
];

function read(filePath) {
  return fs.readFileSync(path.join(root, filePath), 'utf8');
}

function write(filePath, contents) {
  const absolutePath = path.join(root, filePath);
  for (let attempt = 0; ; attempt += 1) {
    try {
      fs.writeFileSync(absolutePath, contents, 'utf8');
      return;
    } catch (error) {
      const canRetry = TRANSIENT_WRITE_ERROR_CODES.has(error && error.code) &&
        attempt < WRITE_RETRY_DELAYS_MS.length;
      if (!canRetry) throw error;
      Atomics.wait(writeRetrySignal, 0, 0, WRITE_RETRY_DELAYS_MS[attempt]);
    }
  }
}

function exists(filePath) {
  return fs.existsSync(path.join(root, filePath));
}

function main() {
  let changed = 0;
  PAIRS.forEach(({ from, to }) => {
    if (!exists(from)) {
      throw new Error(`[sync-root-pages] Missing source file: ${from}`);
    }
    const src = read(from);
    const destPath = path.join(root, to);
    const prev = fs.existsSync(destPath) ? fs.readFileSync(destPath, 'utf8') : null;
    if (prev !== src) {
      write(to, src);
      changed++;
      process.stdout.write(`[sync-root-pages] Updated ${to} from ${from}\n`);
    } else {
      process.stdout.write(`[sync-root-pages] OK ${to}\n`);
    }
  });
  process.stdout.write(`[sync-root-pages] Done (${changed} updated)\n`);
}

main();
