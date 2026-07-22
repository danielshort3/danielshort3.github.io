#!/usr/bin/env node
'use strict';

/*
  Generate deterministic next-generation variants for the site's highest-impact
  raster images. Original PNGs remain as compatibility fallbacks.
*/

const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const root = path.resolve(__dirname, '..');
const jobs = [
  {
    source: 'img/brand/27-hero-mobile-light.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 70, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 90, effort: 6, smartSubsample: true } }
    ]
  },
  {
    source: 'img/brand/24-hero-analytics-light.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 70, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 88, effort: 6, smartSubsample: true } },
      { suffix: '-960', extension: '.avif', format: 'avif', width: 960, options: { quality: 68, effort: 6, chromaSubsampling: '4:4:4' } },
      { suffix: '-960', extension: '.webp', format: 'webp', width: 960, options: { quality: 86, effort: 6, smartSubsample: true } }
    ]
  },
  {
    source: 'img/project-starfall/ui/start-screen.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 70, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 90, effort: 6, smartSubsample: true } }
    ]
  }
];

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value}B`;
  return `${(value / 1024).toFixed(value < 10 * 1024 ? 1 : 0)}KB`;
}

function outputPathFor(sourcePath, output) {
  const suffix = String(output.suffix || '');
  return sourcePath.replace(/\.[^.]+$/, `${suffix}${output.extension}`);
}

function writeIfChanged(filePath, contents) {
  let previous = null;
  try {
    previous = fs.readFileSync(filePath);
  } catch {}
  if (previous && previous.equals(contents)) return false;
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, contents);
  return true;
}

async function renderVariant(sourcePath, output) {
  let pipeline = sharp(sourcePath, { failOn: 'error', sequentialRead: true }).rotate();
  if (Number.isFinite(Number(output.width)) && Number(output.width) > 0) {
    pipeline = pipeline.resize({ width: Number(output.width), withoutEnlargement: true });
  }
  const encoded = output.format === 'avif'
    ? pipeline.avif(output.options)
    : pipeline.webp(output.options);
  return encoded.toBuffer();
}

async function main() {
  let generated = 0;
  let unchanged = 0;

  for (const job of jobs) {
    const sourcePath = path.join(root, job.source);
    if (!fs.existsSync(sourcePath)) {
      throw new Error(`Missing image source: ${job.source}`);
    }

    for (const output of job.outputs) {
      const outputRelPath = outputPathFor(job.source, output);
      const outputPath = path.join(root, outputRelPath);
      const contents = await renderVariant(sourcePath, output);
      const changed = writeIfChanged(outputPath, contents);
      if (changed) generated += 1;
      else unchanged += 1;
      process.stdout.write(`[images] ${outputRelPath} (${formatBytes(contents.length)})${changed ? '' : ' unchanged'}\n`);
    }
  }

  process.stdout.write(`[images] Generated ${generated} variant(s); ${unchanged} unchanged.\n`);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
