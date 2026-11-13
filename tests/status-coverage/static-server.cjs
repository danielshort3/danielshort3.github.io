'use strict';

const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const mime = new Map(Object.entries({
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.avif': 'image/avif', '.woff': 'font/woff', '.woff2': 'font/woff2', '.ico': 'image/x-icon',
  '.wasm': 'application/wasm', '.pdf': 'application/pdf', '.zip': 'application/zip',
  '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.ogg': 'audio/ogg', '.wav': 'audio/wav'
}));

function routeRules(siteRoot) {
  const config = JSON.parse(fs.readFileSync(path.join(siteRoot, 'vercel.json'), 'utf8'));
  return (config.rewrites || []).filter(({ has, missing, source, destination }) =>
    !has?.length && !missing?.length && /^\/(?!\/)/.test(destination) && /^\/[A-Za-z0-9/.:_*-]+$/.test(source))
    .map(({ source, destination }) => {
      const names = [];
      const escaped = source.replace(/\./g, '\\.').replace(/:([A-Za-z]\w*)(\*)?/g, (_, name, wildcard) => {
        names.push(name);
        return wildcard ? '(.*)' : '([^/]+)';
      });
      return { regex: new RegExp('^' + escaped + '$'), names, destination };
    });
}

function findFile(publicRoot, urlPath) {
  const filename = path.resolve(publicRoot, '.' + urlPath);
  const candidates = [filename];
  if (!path.extname(filename)) candidates.push(filename + '.html', path.join(filename, 'index.html'));
  for (const candidate of candidates) {
    try {
      const actual = fs.realpathSync(candidate);
      const relative = path.relative(publicRoot, actual);
      if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) continue;
      const stat = fs.statSync(actual);
      if (stat.isFile()) return { actual, stat };
    } catch {}
  }
  return null;
}

async function startServer(siteRoot) {
  const publicRoot = fs.realpathSync(path.join(siteRoot, 'public'));
  const rewrites = routeRules(siteRoot);
  const server = http.createServer((req, res) => {
    try {
      const parsed = new URL(req.url, 'http://127.0.0.1');
      const pathname = decodeURIComponent(parsed.pathname).replace(/\/+$/, '') || '/';
      if (!['GET', 'HEAD'].includes(req.method) || /[\\\x00-\x1f]/.test(pathname) ||
        /(?:^|\/)\.[^/]/.test(pathname) || /^\/api(?:\/|$)/.test(pathname)) {
        res.writeHead(404).end();
        return;
      }
      const rule = rewrites.find((entry) => entry.regex.test(pathname));
      let rewritten = pathname;
      if (rule) {
        const match = rule.regex.exec(pathname);
        for (const [index, name] of rule.names.entries()) {
          if (index === 0) rewritten = rule.destination;
          rewritten = rewritten.replace(new RegExp(':' + name + '\\*?(?![\\w])', 'g'), () => match[index + 1]);
        }
        if (rule.names.length === 0) rewritten = rule.destination;
      }
      const found = findFile(publicRoot, rewritten === '/' ? '/index.html' : rewritten);
      if (!found) {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, {
        'Content-Type': mime.get(path.extname(found.actual).toLowerCase()) || 'application/octet-stream',
        'Content-Length': found.stat.size,
        'Cache-Control': 'no-store'
      });
      if (req.method === 'HEAD') res.end();
      else fs.createReadStream(found.actual).on('error', () => res.destroy()).pipe(res);
    } catch {
      res.writeHead(400).end();
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return {
    base: 'http://127.0.0.1:' + server.address().port,
    close: () => new Promise((resolve) => {
      server.close(resolve);
      server.closeAllConnections();
    })
  };
}

module.exports = { startServer };
