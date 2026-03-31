#!/usr/bin/env node
'use strict';

/*
  Bundle shared and route-specific site JavaScript into ./dist with content hashes.
  The source HTML can continue to use stable fallback paths; build injectors upgrade
  them to hashed filenames for production caching.
*/

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const esbuild = require('esbuild');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'dist');
const manifestPath = path.join(outDir, 'scripts-manifest.json');

const entries = [
  { entry: path.join(root, 'build', 'entries', 'site-shell.entry.js'), baseName: 'site-shell', manifestKey: 'shell' },
  { entry: path.join(root, 'build', 'entries', 'site-consent.entry.js'), baseName: 'site-consent', manifestKey: 'consent' },
  { entry: path.join(root, 'build', 'entries', 'site-home.entry.js'), baseName: 'site-home', manifestKey: 'home' },
  { entry: path.join(root, 'build', 'entries', 'site-contact.entry.js'), baseName: 'site-contact', manifestKey: 'contact' },
  { entry: path.join(root, 'build', 'entries', 'site-search.entry.js'), baseName: 'site-search', manifestKey: 'search' },
  { entry: path.join(root, 'build', 'entries', 'site-contributions.entry.js'), baseName: 'site-contributions', manifestKey: 'contributions' },
  { entry: path.join(root, 'build', 'entries', 'site-sitemap.entry.js'), baseName: 'site-sitemap', manifestKey: 'sitemap' },
  { entry: path.join(root, 'build', 'entries', 'site-privacy.entry.js'), baseName: 'site-privacy', manifestKey: 'privacy' },
  { entry: path.join(root, 'build', 'entries', 'site-tools-account.entry.js'), baseName: 'site-tools-account', manifestKey: 'toolsAccount' },
  { entry: path.join(root, 'build', 'entries', 'site-tools-landing.entry.js'), baseName: 'site-tools-landing', manifestKey: 'toolsLanding' }
];

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function cleanupOldBundles(baseName) {
  const pattern = new RegExp(`^${baseName.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\.[0-9a-f]{8}\\.js$`, 'i');
  fs.readdirSync(outDir).forEach((fileName) => {
    if (!pattern.test(fileName)) return;
    fs.rmSync(path.join(outDir, fileName), { force: true });
  });
}

async function buildBundle(entryConfig) {
  const result = await esbuild.build({
    entryPoints: [entryConfig.entry],
    bundle: true,
    minify: true,
    sourcemap: false,
    platform: 'browser',
    target: ['es2019'],
    format: 'iife',
    legalComments: 'none',
    write: false,
    logLevel: 'silent',
    define: {
      'process.env.NODE_ENV': '"production"'
    }
  });

  const output = result.outputFiles.find((file) => file.path.endsWith('.js')) || result.outputFiles[0];
  if (!output) {
    throw new Error(`Missing JS output for ${entryConfig.baseName}`);
  }

  const contents = Buffer.from(output.contents).toString('utf8');
  const hash = crypto.createHash('sha256').update(contents).digest('hex').slice(0, 8);
  const hashedName = `${entryConfig.baseName}.${hash}.js`;
  const hashedPath = path.join(outDir, hashedName);
  const legacyPath = path.join(outDir, `${entryConfig.baseName}.js`);

  fs.writeFileSync(hashedPath, contents, 'utf8');
  fs.writeFileSync(legacyPath, contents, 'utf8');

  return hashedName;
}

async function main() {
  ensureDir(outDir);
  entries.forEach(({ baseName }) => cleanupOldBundles(baseName));

  const manifest = {};
  for (const entryConfig of entries) {
    const hashedName = await buildBundle(entryConfig);
    manifest[entryConfig.manifestKey] = hashedName;
    process.stdout.write(`Bundled JS written to dist/${hashedName}\n`);
  }

  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf8');
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
