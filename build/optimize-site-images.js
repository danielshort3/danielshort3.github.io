#!/usr/bin/env node
'use strict';

/*
  Generate deterministic next-generation variants for the site's highest-impact
  raster images. Original PNGs remain as compatibility fallbacks.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const { staticJobs: jobs, getCatalogJobs, outputPathFor, renderVariant } = require('./lib/image-variant-recipes');
const catalogJobs = getCatalogJobs(root);

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value}B`;
  return `${(value / 1024).toFixed(value < 10 * 1024 ? 1 : 0)}KB`;
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

async function main() {
  let generated = 0;
  let unchanged = 0;

  const selectedJobs = process.argv.includes('--catalog-only') ? catalogJobs
    : process.argv.includes('--link-icons-only') ? jobs.filter(job => job.source.startsWith('img/icons/'))
      : [...jobs, ...catalogJobs];
  for (const job of selectedJobs) {
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

if (require.main === module) main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

module.exports = { main };
