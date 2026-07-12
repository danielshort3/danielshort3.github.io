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

function outputPathFor(sourcePath, extension) {
  return sourcePath.replace(/\.[^.]+$/, extension);
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
  const pipeline = sharp(sourcePath, { failOn: 'error', sequentialRead: true }).rotate();
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
      const outputRelPath = outputPathFor(job.source, output.extension);
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
