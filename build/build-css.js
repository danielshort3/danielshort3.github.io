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
const outDir = path.join(root, 'dist');
const entries = [
  { entry: path.join(cssDir, 'styles.css'), baseName: 'styles', manifestKey: 'file' },
  { entry: path.join(cssDir, 'styles-tools.css'), baseName: 'styles-tools', manifestKey: 'toolsFile' }
];

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
  const data = { ...fileName };
  fs.writeFileSync(manifestPath, JSON.stringify(data, null, 2), 'utf8');
}

function buildBundle({ entry: entryPath, baseName }){
  const legacyPath = path.join(outDir, `${baseName}.css`);
  const bundled = inline(entryPath);
  const minified = minify(bundled);
  const hash = crypto.createHash('sha256').update(minified).digest('hex').slice(0, 8);
  const hashedName = `${baseName}.${hash}.css`;
  const hashedPath = path.join(outDir, hashedName);

  fs.writeFileSync(hashedPath, minified, 'utf8');
  fs.writeFileSync(legacyPath, minified, 'utf8');

  return hashedName;
}

fs.mkdirSync(outDir, { recursive: true });
fs.readdirSync(outDir).forEach(file => {
  const shouldRemove = entries.some(({ baseName }) => {
    const pattern = new RegExp(`^${baseName.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\.[0-9a-f]{8}\\.css$`, 'i');
    return pattern.test(file);
  });
  if (shouldRemove) {
    fs.rmSync(path.join(outDir, file), { force: true });
  }
});
const manifest = path.join(outDir, 'styles-manifest.json');
const outputs = {};

entries.forEach((entryConfig) => {
  const hashedName = buildBundle(entryConfig);
  outputs[entryConfig.manifestKey] = hashedName;
  console.log(`Bundled CSS written to dist/${hashedName}`);
});

writeManifest(manifest, outputs);
