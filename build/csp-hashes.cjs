'use strict';
// Only reviewed, build-owned demo scripts receive hashes. Never hash user input.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '..');
const ADMIN_SOURCE = '/admin/:path*';
function inlineScripts(html) {
  const scripts = [];
  for (const match of html.replace(/\r\n?/g, '\n').matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script\s*>/gi)) {
    const attrs = match[1];
    const type = (attrs.match(/\btype\s*=\s*["']([^"']*)["']/i) || [])[1];
    if (/\bsrc\s*=/i.test(attrs) || !match[2].trim()) continue;
    if (type && !/^(?:module|(?:text|application)\/(?:java|ecma)script)$/i.test(type)) continue;
    scripts.push(match[2]);
  }
  return scripts;
}
function hashesForBuild(root = ROOT) {
  const publicDir = path.join(root, 'public');
  if (!fs.existsSync(path.join(publicDir, 'index.html'))) throw Error('Build the site before checking CSP hashes.');
  const hashes = new Set();
  let pages = 0, count = 0;
  function walk(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) { walk(file); continue; }
      if (!entry.isFile() || !file.endsWith('.html')) continue;
      pages++;
      const html = fs.readFileSync(file, 'utf8');
      const scripts = inlineScripts(html);
      if (/<[a-z][^>]*\s+on[a-z]+\s*=/i.test(html)) throw Error(`Inline event handler: ${file}`);
      if (scripts.length && !file.startsWith(path.join(publicDir, 'demos') + path.sep)) {
        throw Error(`Unreviewed executable inline script outside demos: ${file}`);
      }
      for (const script of scripts) {
        hashes.add(`'sha256-${crypto.createHash('sha256').update(script).digest('base64')}'`);
        count++;
      }
    }
  }
  walk(publicDir);
  if (pages < 30) throw Error('Incomplete build; refusing to replace CSP hashes.');
  return { hashes: [...hashes].sort(), pages, count };
}
function replaceScriptHashes(value, hashes) {
  return value.split(';').map(part => {
    const tokens = part.trim().split(/\s+/);
    if (!['script-src', 'script-src-elem'].includes(tokens[0])) return part.trim();
    return [...tokens.filter(token => token !== "'unsafe-inline'" && !/^'sha(?:256|384|512)-/.test(token)), ...hashes].join(' ');
  }).filter(Boolean).join('; ');
}
function updateConfig(config, hashes) {
  const next = JSON.parse(JSON.stringify(config));
  for (const rule of next.headers) for (const header of rule.headers) {
    if (header.key.toLowerCase() !== 'content-security-policy' || rule.source === ADMIN_SOURCE) continue;
    // Hashes are confined to raw demos; normal pages and sensitive tools
    // retain smaller and narrower policies with no demo scripts authorized.
    header.value = replaceScriptHashes(header.value, rule.source === '/demos/:path*' ? hashes : []);
  }
  return next;
}
function main() {
  const file = path.join(ROOT, 'vercel.json');
  const config = JSON.parse(fs.readFileSync(file, 'utf8'));
  const report = hashesForBuild();
  const next = updateConfig(config, report.hashes);
  if (process.argv.includes('--write')) fs.writeFileSync(file, JSON.stringify(next, null, 2) + '\n');
  else if (JSON.stringify(next) !== JSON.stringify(config)) throw Error('CSP hashes are stale. Review changed inline demo scripts, then run node build/csp-hashes.cjs --write and commit the policy.');
  console.log(`CSP: ${report.pages} built pages, ${report.count} inline demo scripts, ${report.hashes.length} reviewed hashes.`);
}
module.exports = { inlineScripts, hashesForBuild, replaceScriptHashes, updateConfig, ADMIN_SOURCE };
if (require.main === module) { try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; } }
