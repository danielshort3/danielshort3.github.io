#!/usr/bin/env node
/*
  Bundle the UTM Batch Builder tool (React + worker) into ./dist.
  - No runtime external deps (React is bundled)
*/
'use strict';

const path = require('path');

async function run() {
  const esbuild = require('esbuild');
  const root = path.resolve(__dirname, '..');
  const outdir = path.join(root, 'dist');

  await esbuild.build({
    entryPoints: {
      'utm-batch-builder': path.join(root, 'src/utm-batch-builder/app.tsx'),
      'utm-batch-builder.worker': path.join(root, 'src/utm-batch-builder/worker.ts'),
    },
    outdir,
    bundle: true,
    minify: true,
    sourcemap: false,
    platform: 'browser',
    target: ['es2019'],
    format: 'iife',
    define: {
      'process.env.NODE_ENV': '"production"',
    },
    logLevel: 'info',
  });
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

