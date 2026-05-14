#!/usr/bin/env node
'use strict';

/*
  Local developer workflow:
  1) Run a full site build once.
  2) Watch source files and rebuild automatically.
  3) Serve the static site plus the local-only CMS file API.
*/

const fs = require('fs');
const http = require('http');
const path = require('path');
const { spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const publicDir = path.join(root, 'public');
const adminDir = path.join(root, 'admin');
const cmsApiPath = path.join(root, 'api', 'cms', '[...slug].js');
const chatbotApiPath = path.join(root, 'api', 'chatbot.js');
const chatbotLogsApiPath = path.join(root, 'api', 'chatbot', 'logs.js');
const sentenceDemoApiPath = path.join(root, 'api', 'sentence-demo', '[...slug].js');
const MAX_PORT_SEARCH_ATTEMPTS = 50;
const WATCH_POLL_INTERVAL_MS = 1000;

const WATCH_ROOTS = [
  'api',
  'admin',
  'build',
  'css',
  'content',
  'demos',
  'documents',
  'img',
  'js',
  'pages',
  'src',
  'vercel.json'
];

const IGNORED_PREFIXES = [
  '.git/',
  '.vercel/',
  'build/cache/',
  'build/templates/',
  'dist/',
  'js/portfolio/projects-data.js',
  'node_modules/',
  'pages/',
  'pages/portfolio/',
  'public/',
  'tmp/'
];

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.avif': 'image/avif',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.mp4': 'video/mp4',
  '.pdf': 'application/pdf',
  '.png': 'image/png',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.webm': 'video/webm',
  '.webp': 'image/webp',
  '.xml': 'application/xml; charset=utf-8'
};

function log(line) {
  process.stdout.write(`[dev] ${line}\n`);
}

function hasFlag(flag) {
  return process.argv.slice(2).includes(flag);
}

function parsePort() {
  const args = process.argv.slice(2);
  const explicitPortIndex = args.indexOf('--port');
  if (explicitPortIndex >= 0 && args[explicitPortIndex + 1]) {
    return String(args[explicitPortIndex + 1]).trim();
  }
  if (process.env.PORT) return String(process.env.PORT).trim();
  return '3000';
}

function normalizePort(value) {
  const port = Number(String(value || '').trim());
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    log(`Invalid port: ${value}. Use a whole number between 1 and 65535.`);
    process.exit(1);
  }
  return port;
}

function parseHost() {
  const args = process.argv.slice(2);
  const explicitHostIndex = args.indexOf('--host');
  if (explicitHostIndex >= 0 && args[explicitHostIndex + 1]) {
    return String(args[explicitHostIndex + 1]).trim();
  }
  if (process.env.HOST) return String(process.env.HOST).trim();
  return '127.0.0.1';
}

function getDisplayHost(host) {
  return host === '0.0.0.0' || host === '::' ? 'localhost' : host;
}

function listenWithPortFallback(server, { host, startPort, onReady, onFatal }) {
  let candidatePort = startPort;
  let attemptCount = 0;

  function tryListen() {
    function handleListening() {
      server.removeListener('error', handleError);
      onReady(candidatePort);
    }

    function handleError(err) {
      server.removeListener('listening', handleListening);

      if (err && err.code === 'EADDRINUSE' && candidatePort < 65535 && attemptCount < MAX_PORT_SEARCH_ATTEMPTS) {
        const previousPort = candidatePort;
        candidatePort += 1;
        attemptCount += 1;
        log(`Port ${previousPort} is in use; trying ${candidatePort}...`);
        setTimeout(tryListen, 0);
        return;
      }

      if (err && err.code === 'EADDRINUSE') {
        const exhausted = new Error(`No available dev port found from ${startPort} through ${candidatePort}.`);
        exhausted.code = err.code;
        onFatal(exhausted);
        return;
      }

      onFatal(err);
    }

    server.once('listening', handleListening);
    server.once('error', handleError);
    server.listen(candidatePort, host);
  }

  tryListen();
}

function runSiteBuild({ exitOnFail }) {
  const result = spawnSync(process.execPath, [path.join('build', 'build-site.js')], {
    cwd: root,
    env: { ...process.env },
    stdio: 'inherit'
  });

  if (result.error) {
    const message = `Build failed: ${result.error.message}`;
    if (exitOnFail) {
      log(message);
      process.exit(1);
    }
    log(message);
    return false;
  }

  if (result.status !== 0) {
    if (exitOnFail) process.exit(result.status || 1);
    log(`Build exited with status ${result.status}.`);
    return false;
  }

  return true;
}

function getWatchRoots() {
  return WATCH_ROOTS.filter((target) => fs.existsSync(path.join(root, target)));
}

function normalizeRelative(filePath) {
  return filePath.split(path.sep).join('/');
}

function isIgnoredRelative(relativePath) {
  const normalized = normalizeRelative(relativePath);
  return IGNORED_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}

async function collectSnapshotAsync(watchRoots) {
  const snapshot = new Map();

  for (const target of watchRoots) {
    const absoluteTarget = path.join(root, target);
    try {
      await fs.promises.stat(absoluteTarget);
    } catch {
      continue;
    }

    const stack = [absoluteTarget];
    while (stack.length) {
      const current = stack.pop();
      const relCurrent = path.relative(root, current);
      if (relCurrent && isIgnoredRelative(relCurrent)) continue;

      let stat;
      try {
        stat = await fs.promises.stat(current);
      } catch {
        continue;
      }

      if (stat.isDirectory()) {
        let entries;
        try {
          entries = await fs.promises.readdir(current, { withFileTypes: true });
        } catch {
          continue;
        }
        entries.forEach((entry) => stack.push(path.join(current, entry.name)));
        continue;
      }

      if (!stat.isFile()) continue;
      const relFile = path.relative(root, current);
      if (!relFile || isIgnoredRelative(relFile)) continue;
      snapshot.set(normalizeRelative(relFile), `${stat.mtimeMs}:${stat.size}`);
    }
  }

  return snapshot;
}

function snapshotsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [key, value] of a.entries()) {
    if (b.get(key) !== value) return false;
  }
  return true;
}

function getSnapshotDelta(previousSnapshot, nextSnapshot) {
  for (const [key, value] of nextSnapshot.entries()) {
    if (!previousSnapshot.has(key)) return `added ${key}`;
    if (previousSnapshot.get(key) !== value) return `changed ${key}`;
  }

  for (const key of previousSnapshot.keys()) {
    if (!nextSnapshot.has(key)) return `removed ${key}`;
  }

  return 'unknown change';
}

function startBuildWatcher({ watchRoots, onChange }) {
  log(`Watching ${watchRoots.length} source target(s) for changes...`);

  let snapshot = null;
  let building = false;
  let queued = false;
  let scanning = true;
  let closed = false;

  async function setBaselineSnapshot() {
    snapshot = await collectSnapshotAsync(watchRoots);
  }

  setBaselineSnapshot()
    .catch((err) => {
      log(`Initial watch scan failed: ${err && err.message ? err.message : err}`);
    })
    .finally(() => {
      scanning = false;
    });

  async function rebuild() {
    if (building) {
      queued = true;
      return;
    }

    building = true;
    onChange();
    try {
      snapshot = await collectSnapshotAsync(watchRoots);
    } catch (err) {
      log(`Watch scan failed after rebuild: ${err && err.message ? err.message : err}`);
    }
    building = false;

    if (queued && !closed) {
      queued = false;
      await rebuild();
    }
  }

  const timer = setInterval(() => {
    if (closed || building || scanning) return;

    scanning = true;
    collectSnapshotAsync(watchRoots)
      .then((nextSnapshot) => {
        if (!snapshot) {
          snapshot = nextSnapshot;
          return null;
        }

        if (snapshotsEqual(snapshot, nextSnapshot)) return null;
        const delta = getSnapshotDelta(snapshot, nextSnapshot);
        snapshot = nextSnapshot;
        log(`Source change detected (${delta}). Rebuilding...`);
        return rebuild();
      })
      .catch((err) => {
        log(`Watch scan failed: ${err && err.message ? err.message : err}`);
      })
      .finally(() => {
        scanning = false;
      });
  }, WATCH_POLL_INTERVAL_MS);

  return {
    close() {
      closed = true;
      clearInterval(timer);
    }
  };
}

function safeDecodePathname(value) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function readVercelConfig() {
  try {
    return JSON.parse(fs.readFileSync(path.join(root, 'vercel.json'), 'utf8'));
  } catch {
    return {};
  }
}

function compileRoute(source) {
  const names = [];
  const pattern = String(source || '').replace(/\/\(\.\*\)$/g, '/:path*');
  let regexSource = '^';

  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index];
    if (char === ':') {
      let end = index + 1;
      while (end < pattern.length && /[A-Za-z0-9_]/.test(pattern[end])) end += 1;
      const name = pattern.slice(index + 1, end);
      const wildcard = pattern[end] === '*';
      if (name) {
        names.push(name);
        regexSource += wildcard ? '(.*)' : '([^/]+)';
        index = wildcard ? end : end - 1;
        continue;
      }
    }
    regexSource += char.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  regexSource += '$';

  return {
    source,
    names,
    regex: new RegExp(regexSource)
  };
}

function compileRoutes(rules) {
  return (Array.isArray(rules) ? rules : [])
    .filter((rule) => rule && rule.source)
    .map((rule) => ({
      ...rule,
      compiled: compileRoute(rule.source)
    }));
}

function hasRegexMeta(value) {
  return /[.*+?^${}()|[\]\\]/.test(String(value || ''));
}

function matchConditionValue(actual, expected) {
  const value = String(actual || '');
  if (typeof expected === 'undefined' || expected === null || expected === '') return !!value;
  const pattern = String(expected);
  if (!hasRegexMeta(pattern)) return value === pattern;
  try {
    return new RegExp(pattern, 'i').test(value);
  } catch {
    return value === pattern;
  }
}

function getHeaderValue(req, key) {
  if (!req || !req.headers || !key) return '';
  const header = req.headers[String(key).toLowerCase()];
  if (Array.isArray(header)) return header.join(', ');
  return typeof header === 'string' ? header : '';
}

function hasConditionsMatch(rule, req, url) {
  const conditions = Array.isArray(rule && rule.has) ? rule.has : [];
  if (!conditions.length) return true;
  return conditions.every((condition) => {
    const type = String(condition && condition.type || '').toLowerCase();
    const key = String(condition && condition.key || '').toLowerCase();
    const expected = condition && condition.value;
    if (type === 'host') {
      const host = getHeaderValue(req, 'host').split(':')[0].trim().toLowerCase();
      return matchConditionValue(host, expected);
    }
    if (type === 'header') {
      return matchConditionValue(getHeaderValue(req, key), expected);
    }
    if (type === 'query') {
      const params = url && url.searchParams ? url.searchParams : new URLSearchParams();
      if (!params.has(key)) return false;
      return matchConditionValue(params.get(key), expected);
    }
    return false;
  });
}

function matchRule(pathname, rules, req, url) {
  for (const rule of rules) {
    const match = rule.compiled.regex.exec(pathname);
    if (!match) continue;
    if (!hasConditionsMatch(rule, req, url)) continue;
    const params = {};
    rule.compiled.names.forEach((name, index) => {
      params[name] = match[index + 1] || '';
    });
    return { rule, params };
  }
  return null;
}

function applyParams(value, params) {
  let next = String(value || '');
  Object.entries(params || {}).forEach(([key, raw]) => {
    const encoded = encodeURIComponent(raw);
    next = next.replace(new RegExp(`:${key}\\*`, 'g'), raw);
    next = next.replace(new RegExp(`:${key}`, 'g'), encoded);
  });
  return next;
}

function isInside(baseDir, filePath) {
  const rel = path.relative(baseDir, filePath);
  return !!rel && !rel.startsWith('..') && !path.isAbsolute(rel);
}

function resolveStaticFile(baseDir, pathname) {
  const decoded = safeDecodePathname(pathname).replace(/\\/g, '/');
  const clean = decoded.split('?')[0].split('#')[0];
  const withoutLeading = clean.replace(/^\/+/, '');
  const candidates = [];

  if (!withoutLeading) {
    candidates.push(path.join(baseDir, 'index.html'));
  } else {
    candidates.push(path.join(baseDir, withoutLeading));
    if (!path.extname(withoutLeading)) {
      candidates.push(path.join(baseDir, `${withoutLeading}.html`));
      candidates.push(path.join(baseDir, withoutLeading, 'index.html'));
    }
  }

  for (const candidate of candidates) {
    if (!isInside(baseDir, candidate)) continue;
    try {
      const stat = fs.statSync(candidate);
      if (stat.isFile()) return candidate;
    } catch {}
  }

  return null;
}

function parseByteRange(value, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(String(value || '').trim());
  if (!match || !Number.isFinite(size) || size < 1) return null;
  let start = match[1] === '' ? null : Number(match[1]);
  let end = match[2] === '' ? null : Number(match[2]);

  if (start === null && end === null) return null;
  if ((start !== null && !Number.isSafeInteger(start)) || (end !== null && !Number.isSafeInteger(end))) return null;

  if (start === null) {
    const suffixLength = end;
    if (!suffixLength || suffixLength < 1) return null;
    start = Math.max(size - suffixLength, 0);
    end = size - 1;
  } else if (end === null || end >= size) {
    end = size - 1;
  }

  if (start < 0 || start >= size || end < start) return null;
  return { start, end };
}

function sendRangeNotSatisfiable(res, size) {
  res.statusCode = 416;
  res.setHeader('Content-Range', `bytes */${size}`);
  res.setHeader('Content-Length', '0');
  res.end();
}

function sendFile(req, res, filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const stat = fs.statSync(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  const rangeHeader = req && req.headers ? req.headers.range : '';
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', contentType);

  if (rangeHeader) {
    const range = parseByteRange(rangeHeader, stat.size);
    if (!range) {
      sendRangeNotSatisfiable(res, stat.size);
      return;
    }
    const contentLength = range.end - range.start + 1;
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${range.start}-${range.end}/${stat.size}`);
    res.setHeader('Content-Length', String(contentLength));
    if (req && req.method === 'HEAD') {
      res.end();
      return;
    }
    fs.createReadStream(filePath, range).pipe(res);
    return;
  }

  res.statusCode = 200;
  res.setHeader('Content-Length', String(stat.size));
  if (req && req.method === 'HEAD') {
    res.end();
    return;
  }
  fs.createReadStream(filePath).pipe(res);
}

function sendNotFound(res) {
  const notFound = path.join(publicDir, '404.html');
  if (fs.existsSync(notFound)) {
    res.statusCode = 404;
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    fs.createReadStream(notFound).pipe(res);
    return;
  }
  res.statusCode = 404;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end('Not found');
}

function clearCmsApiCache() {
  const prefixes = [
    path.join(root, 'api', 'cms'),
    path.join(root, 'api', '_lib', 'cms'),
    path.join(root, 'build', 'generate-project-pages.js'),
    path.join(root, 'build', 'lib', 'cms-renderers')
  ];
  Object.keys(require.cache).forEach((modulePath) => {
    if (prefixes.some((prefix) => modulePath.startsWith(prefix))) {
      delete require.cache[modulePath];
    }
  });
}

function loadCmsApi() {
  clearCmsApiCache();
  return require(cmsApiPath);
}

function clearChatbotApiCache() {
  const prefixes = [
    chatbotApiPath,
    path.join(root, 'api', 'chatbot'),
    path.join(root, 'api', '_lib', 'chatbot')
  ];
  Object.keys(require.cache).forEach((modulePath) => {
    if (prefixes.some((prefix) => modulePath.startsWith(prefix))) {
      delete require.cache[modulePath];
    }
  });
}

function loadChatbotApi() {
  clearChatbotApiCache();
  return require(chatbotApiPath);
}

function loadChatbotLogsApi() {
  clearChatbotApiCache();
  return require(chatbotLogsApiPath);
}

function clearSentenceDemoApiCache() {
  const prefixes = [
    sentenceDemoApiPath,
    path.join(root, 'api', 'sentence-demo')
  ];
  Object.keys(require.cache).forEach((modulePath) => {
    if (prefixes.some((prefix) => modulePath.startsWith(prefix))) {
      delete require.cache[modulePath];
    }
  });
}

function loadSentenceDemoApi() {
  clearSentenceDemoApiCache();
  return require(sentenceDemoApiPath);
}

function createLocalServer() {
  const vercelConfig = readVercelConfig();
  const redirects = compileRoutes(vercelConfig.redirects);
  const rewrites = compileRoutes(vercelConfig.rewrites);

  return http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const pathname = url.pathname.replace(/\/+$/, '') || '/';

    if (pathname === '/api/chatbot' || pathname === '/api/chatbot/logs') {
      req.query = Object.fromEntries(url.searchParams.entries());
      try {
        const handler = pathname === '/api/chatbot/logs' ? loadChatbotLogsApi() : loadChatbotApi();
        handler(req, res);
      } catch (err) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({
          ok: false,
          error: err && err.message ? err.message : 'Local chatbot API failed to load'
        }));
      }
      return;
    }

    if (pathname.startsWith('/api/cms/')) {
      req.query = Object.fromEntries(url.searchParams.entries());
      req.query.slug = pathname.slice('/api/cms/'.length).split('/')[0] || '';
      try {
        loadCmsApi()(req, res);
      } catch (err) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({
          ok: false,
          error: err && err.message ? err.message : 'Local CMS API failed to load'
        }));
      }
      return;
    }

    if (pathname === '/api/sentence-demo' || pathname.startsWith('/api/sentence-demo/')) {
      req.query = Object.fromEntries(url.searchParams.entries());
      const rawSlug = pathname.startsWith('/api/sentence-demo/')
        ? pathname.slice('/api/sentence-demo/'.length)
        : '';
      req.query.slug = rawSlug ? rawSlug.split('/') : '';
      try {
        loadSentenceDemoApi()(req, res);
      } catch (err) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({
          ok: false,
          error: err && err.message ? err.message : 'Local sentence demo API failed to load'
        }));
      }
      return;
    }

    if (pathname === '/admin' || pathname.startsWith('/admin/')) {
      const adminPath = pathname === '/admin' ? '/index.html' : pathname.slice('/admin'.length);
      const filePath = resolveStaticFile(adminDir, adminPath);
      if (filePath) {
        sendFile(req, res, filePath);
        return;
      }
      sendNotFound(res);
      return;
    }

    const redirectMatch = matchRule(pathname, redirects, req, url);
    if (redirectMatch) {
      const destination = applyParams(redirectMatch.rule.destination, redirectMatch.params);
      res.statusCode = redirectMatch.rule.permanent ? 308 : 307;
      res.setHeader('Location', destination);
      res.end();
      return;
    }

    const rewriteMatch = matchRule(pathname, rewrites, req, url);
    const rewrittenPath = rewriteMatch
      ? applyParams(rewriteMatch.rule.destination, rewriteMatch.params)
      : pathname;

    const rewrittenUrl = new URL(rewrittenPath, 'http://localhost');
    const rewrittenPathname = rewrittenUrl.pathname.replace(/\/+$/, '') || '/';

    const filePath = resolveStaticFile(publicDir, rewrittenPathname);
    if (filePath) {
      sendFile(req, res, filePath);
      return;
    }

    sendNotFound(res);
  });
}

function main() {
  const port = normalizePort(parsePort());
  const host = parseHost();
  const displayHost = getDisplayHost(host);
  const noWatch = hasFlag('--no-watch');

  log('Running initial build...');
  runSiteBuild({ exitOnFail: true });

  const watcher = noWatch
    ? null
    : startBuildWatcher({
      watchRoots: getWatchRoots(),
      onChange: () => runSiteBuild({ exitOnFail: false })
    });

  if (noWatch) log('Build watcher disabled (--no-watch).');

  const server = createLocalServer();
  listenWithPortFallback(server, {
    host,
    startPort: port,
    onReady: (selectedPort) => {
      log(`Serving local site on http://${displayHost}:${selectedPort}`);
      if (displayHost !== host) log(`Bound to ${host} for WSL/host access.`);
      log(`Local CMS available at http://${displayHost}:${selectedPort}/admin`);
    },
    onFatal: (err) => {
      log(`Local server failed: ${err && err.message ? err.message : err}`);
      if (watcher) watcher.close();
      process.exit(1);
    }
  });

  function shutdown(exitCode) {
    if (watcher) watcher.close();
    server.close(() => process.exit(exitCode));
    setTimeout(() => process.exit(exitCode), 500).unref();
  }

  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));
}

main();
