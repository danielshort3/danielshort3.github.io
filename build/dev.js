#!/usr/bin/env node
'use strict';

/*
  Local developer workflow:
  1) Run a full site build once.
  2) Watch source files and rebuild automatically.
  3) Serve routes/API locally using Vercel dev routing.
*/

const fs = require('fs');
const path = require('path');
const { spawn, spawnSync } = require('child_process');

const root = path.resolve(__dirname, '..');
const isWindows = process.platform === 'win32';
const localVercelConfigPath = path.join(root, '.vercel', 'dev.local.json');

const WATCH_ROOTS = [
  'api',
  'build',
  'css',
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
  'dist/',
  'node_modules/',
  'pages/portfolio/',
  'public/'
];

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
    if (exitOnFail) {
      process.exit(result.status || 1);
    }
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

function collectSnapshot(watchRoots) {
  const snapshot = new Map();

  watchRoots.forEach((target) => {
    const absoluteTarget = path.join(root, target);
    try {
      fs.statSync(absoluteTarget);
    } catch {
      return;
    }

    const stack = [absoluteTarget];

    while (stack.length) {
      const current = stack.pop();
      const relCurrent = path.relative(root, current);
      if (relCurrent && isIgnoredRelative(relCurrent)) continue;

      let stat;
      try {
        stat = fs.statSync(current);
      } catch {
        continue;
      }

      if (stat.isDirectory()) {
        let entries;
        try {
          entries = fs.readdirSync(current, { withFileTypes: true });
        } catch {
          continue;
        }

        entries.forEach((entry) => {
          stack.push(path.join(current, entry.name));
        });
        continue;
      }

      if (!stat.isFile()) continue;
      const relFile = path.relative(root, current);
      if (!relFile || isIgnoredRelative(relFile)) continue;
      snapshot.set(normalizeRelative(relFile), `${stat.mtimeMs}:${stat.size}`);
    }
  });

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

  let snapshot = collectSnapshot(watchRoots);
  let building = false;
  let queued = false;

  function rebuild() {
    if (building) {
      queued = true;
      return;
    }

    building = true;
    onChange();
    snapshot = collectSnapshot(watchRoots);
    building = false;

    if (queued) {
      queued = false;
      rebuild();
    }
  }

  const timer = setInterval(() => {
    if (building) return;
    const nextSnapshot = collectSnapshot(watchRoots);
    if (snapshotsEqual(snapshot, nextSnapshot)) return;
    const delta = getSnapshotDelta(snapshot, nextSnapshot);
    snapshot = nextSnapshot;
    log(`Source change detected (${delta}). Rebuilding...`);
    rebuild();
  }, 1000);

  return {
    close() {
      clearInterval(timer);
    }
  };
}

function quoteForCmd(value) {
  return `"${String(value).replace(/"/g, '""')}"`;
}

function ensureLocalVercelConfig() {
  const sourcePath = path.join(root, 'vercel.json');
  let parsed;

  try {
    parsed = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
  } catch (err) {
    throw new Error(`Could not parse vercel.json: ${err.message}`);
  }

  const localConfig = JSON.parse(JSON.stringify(parsed || {}));
  delete localConfig.framework;
  delete localConfig.installCommand;

  // Vercel dev ignores `has` matching, so host/query-specific rules can hijack
  // local asset requests (e.g. /dist/styles.css). Remove those rules locally.
  if (Array.isArray(localConfig.rewrites)) {
    localConfig.rewrites = localConfig.rewrites
      .filter((rule) => {
        return !(rule && Object.prototype.hasOwnProperty.call(rule, 'has'));
      })
      .map((rule) => {
        if (!rule || typeof rule !== 'object') return rule;
        const next = { ...rule };

        // Avoid noisy path-to-regexp warnings in local Vercel dev output.
        if (typeof next.destination === 'string') {
          next.destination = next.destination.replace(':first%2F:rest', ':first/:rest');
        }

        return next;
      });
  }
  if (Array.isArray(localConfig.redirects)) {
    localConfig.redirects = localConfig.redirects.filter((rule) => {
      return !(rule && Object.prototype.hasOwnProperty.call(rule, 'has'));
    });
  }

  localConfig.builds = [
    { src: 'api/**/*.js', use: '@vercel/node' },
    { src: '**/*', use: '@vercel/static' }
  ];

  fs.mkdirSync(path.dirname(localVercelConfigPath), { recursive: true });
  fs.writeFileSync(localVercelConfigPath, `${JSON.stringify(localConfig, null, 2)}\n`, 'utf8');
}

function buildDevEnv(port) {
  const env = { ...process.env, PORT: port };
  const key = Object.prototype.hasOwnProperty.call(env, 'Path') ? 'Path' : 'PATH';
  const delim = isWindows ? ';' : ':';
  const existing = env[key] || '';
  env[key] = existing ? `${root}${delim}${existing}` : root;
  return env;
}

function resolveVercelCommand(port) {
  const localVercel = path.join(root, 'node_modules', '.bin', isWindows ? 'vercel.cmd' : 'vercel');

  if (isWindows) {
    if (fs.existsSync(localVercel)) {
      return {
        command: 'cmd.exe',
        args: ['/d', '/s', '/c', `${quoteForCmd(localVercel)} dev --listen ${port} --local-config ${quoteForCmd(localVercelConfigPath)}`],
        viaNpx: false
      };
    }

    return {
      command: 'cmd.exe',
      args: ['/d', '/s', '/c', `npx --yes vercel dev --listen ${port} --local-config ${quoteForCmd(localVercelConfigPath)}`],
      viaNpx: true
    };
  }

  if (fs.existsSync(localVercel)) {
    return {
      command: localVercel,
      args: ['dev', '--listen', port, '--local-config', localVercelConfigPath],
      viaNpx: false
    };
  }

  return {
    command: 'npx',
    args: ['--yes', 'vercel', 'dev', '--listen', port, '--local-config', localVercelConfigPath],
    viaNpx: true
  };
}

function startVercelDev(port) {
  ensureLocalVercelConfig();
  const vercel = resolveVercelCommand(port);
  if (vercel.viaNpx) {
    log('Using npx to run Vercel CLI (first run may download the package).');
  }
  log(`Starting local server on http://localhost:${port} ...`);
  return spawn(vercel.command, vercel.args, {
    cwd: root,
    env: buildDevEnv(port),
    stdio: 'inherit'
  });
}

function main() {
  const port = parsePort();
  const noWatch = hasFlag('--no-watch');

  log('Running initial build...');
  runSiteBuild({ exitOnFail: true });

  const watcher = noWatch
    ? null
    : startBuildWatcher({
      watchRoots: getWatchRoots(),
      onChange: () => runSiteBuild({ exitOnFail: false })
    });

  if (noWatch) {
    log('Build watcher disabled (--no-watch).');
  }

  const server = startVercelDev(port);
  let shuttingDown = false;

  function shutdown(exitCode) {
    if (shuttingDown) return;
    shuttingDown = true;

    if (watcher) watcher.close();

    if (server && server.exitCode == null) {
      server.kill('SIGTERM');
    }

    setTimeout(() => {
      process.exit(exitCode);
    }, 200);
  }

  server.on('exit', (code, signal) => {
    if (shuttingDown) return;
    log(`Local server exited (${signal || code || 'unknown'}).`);
    shutdown(typeof code === 'number' ? code : 1);
  });

  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));
}

main();


