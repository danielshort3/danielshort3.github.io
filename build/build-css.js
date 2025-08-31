#!/usr/bin/env node
/*
  Minimal CSS bundler: resolves @import url("...") lines in css/styles.css
  and writes a single bundle to dist/styles.css. No external deps.
*/
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const cssDir = path.join(root, 'css');
const entry = path.join(cssDir, 'styles.css');
const outDir = path.join(root, 'dist');
const outFile = path.join(outDir, 'styles.css');

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
      if (!fs.existsSync(target)) {
        out += `/* Skipping missing import: ${rel} */\n`;
        continue;
      }
      if (seen.has(target)) {
        out += `/* Skipping duplicate import: ${rel} */\n`;
        continue;
      }
      seen.add(target);
      out += `\n/* === Begin import: ${path.relative(root, target)} === */\n`;
      out += inline(target, seen);
      out += `\n/* === End import: ${path.relative(root, target)} === */\n`;
    } else {
      out += line + '\n';
    }
  }
  return out;
}

fs.mkdirSync(outDir, { recursive: true });
const bundled = inline(entry);
fs.writeFileSync(outFile, bundled, 'utf8');
console.log(`Bundled CSS written to ${path.relative(root, outFile)}`);

