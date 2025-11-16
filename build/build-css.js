#!/usr/bin/env node
/*
  Minimal CSS bundler: resolves @import url("...") lines in css/styles.css
  and writes a single minified bundle with a content hash to ./dist.
  No external deps.
*/
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const root = path.resolve(__dirname, '..');
const cssDir = path.join(root, 'css');
const entry = path.join(cssDir, 'styles.css');
const outDir = path.join(root, 'dist');
const legacyFile = path.join(outDir, 'styles.css');

function inline(file, seen = new Set()){
  const css = fs.readFileSync(file, 'utf8');
  const dir = path.dirname(file);
  const lines = css.split(/\r?\n/);
  let out = '';
  for (const line of lines){
    const m = line.match(/^\s*@import\s+url\(["'](.+?)["']\)\s*;\s*$/);
    if (m){
      const rel = m[1].trim();
      const target = path.resolve(dir, rel);
      if (!fs.existsSync(target)) continue;
      if (seen.has(target)) continue;
      seen.add(target);
      out += inline(target, seen);
    } else {
      out += line + '\n';
    }
  }
  return out;
}

function minify(css){
  return css
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\s*([{}:;,])\s*/g, '$1')
    .replace(/\s+/g, ' ')
    .replace(/;}/g, '}')
    .replace(/,\s+/g, ',')
    .trim();
}

function writeManifest(manifestPath, fileName){
  const data = { file: fileName };
  fs.writeFileSync(manifestPath, JSON.stringify(data, null, 2), 'utf8');
}

fs.mkdirSync(outDir, { recursive: true });
const hashedCssPattern = /^styles\.[0-9a-f]{8}\.css$/i;
fs.readdirSync(outDir).forEach(file => {
  if (hashedCssPattern.test(file)) {
    fs.rmSync(path.join(outDir, file), { force: true });
  }
});
const bundled = inline(entry);
const minified = minify(bundled);
const hash = crypto.createHash('sha256').update(minified).digest('hex').slice(0, 8);
const hashedName = `styles.${hash}.css`;
const hashedPath = path.join(outDir, hashedName);
const manifest = path.join(outDir, 'styles-manifest.json');

fs.writeFileSync(hashedPath, minified, 'utf8');
fs.writeFileSync(legacyFile, minified, 'utf8');
writeManifest(manifest, hashedName);

console.log(`Bundled CSS written to dist/${hashedName}`);
