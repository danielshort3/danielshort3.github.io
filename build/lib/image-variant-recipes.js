'use strict';

// Shared by encoding and cache keys: changing the recipe changes the URL before
// encoding runs, without fingerprinting a stale variant from an earlier build.
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

const pipelineOptions = { input: { failOn: 'error', sequentialRead: true }, rotate: true, resize: { withoutEnlargement: true } };
const encoderVersions = sharp.versions;
const staticJobs = [
  ...['about-ai-network-v1', 'about-family-frame-v1', 'about-french-horn-sheet-music-v1'].map(name => ({
    source: `img/hero/${name}.webp`,
    outputs: [192, 384].map(width => ({ suffix: `-${width}`, extension: '.webp', format: 'webp', width, options: { quality: 88, effort: 6, smartSubsample: true } }))
  })),
  {
    source: 'img/brand/27-hero-mobile-light.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 56, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 90, effort: 6, smartSubsample: true } }
    ]
  },
  {
    source: 'img/brand/23-hero-general-light.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 56, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 90, effort: 6, smartSubsample: true } }
    ]
  },
  {
    source: 'img/project-starfall/ui/start-screen.png',
    outputs: [
      { extension: '.avif', format: 'avif', options: { quality: 66, effort: 6, chromaSubsampling: '4:4:4' } },
      { extension: '.webp', format: 'webp', options: { quality: 90, effort: 6, smartSubsample: true } }
    ]
  },
  {
    source: 'img/icons/website-icon.png',
    outputs: [{ suffix: '-64', extension: '.webp', format: 'webp', width: 64,
      options: { quality: 90, effort: 6, smartSubsample: true } }]
  },
  {
    source: 'img/icons/github-icon.png',
    outputs: [{ suffix: '-64', extension: '.webp', format: 'webp', width: 64,
      options: { quality: 90, effort: 6, smartSubsample: true } }]
  },
  {
    source: 'img/projects/website.png',
    outputs: [
      ...[null, 640, 960].flatMap((width) => [
        { suffix: width ? `-${width}` : '', extension: '.avif', format: 'avif', width, options: { quality: 68, effort: 6, chromaSubsampling: '4:4:4' } },
        { suffix: width ? `-${width}` : '', extension: '.webp', format: 'webp', width, options: { quality: 92, effort: 6, smartSubsample: true } }
      ]),
      { suffix: '-preview', extension: '.webp', format: 'webp', width: 1280, options: { quality: 94, effort: 6, smartSubsample: true } }
    ]
  }
];

const catalogDirectories = ['img/projects/icons', 'img/tools/icons', 'img/games/icons', 'img/ui/site-icons'];

function outputPathFor(source, output) {
  return source.replace(/\.[^.]+$/, `${String(output.suffix || '')}${output.extension}`);
}

function catalogJob(source) {
  const directory = path.posix.dirname(source);
  if (!catalogDirectories.includes(directory) || !/^[a-z0-9_-]+\.png$/i.test(path.posix.basename(source))) return null;
  const compact = directory === 'img/games/icons' || directory === 'img/ui/site-icons';
  return { source, outputs: [{ extension: '.webp', format: 'webp', width: compact ? 128 : null,
    options: { quality: compact ? 80 : 90, effort: 6, smartSubsample: true } }] };
}

function getCatalogJobs(root) {
  return catalogDirectories.flatMap((directory) => fs.readdirSync(path.join(root, directory))
    .filter((file) => /\.png$/i.test(file)).sort()
    .map((file) => catalogJob(`${directory}/${file}`)));
}

function getImageJob(relative) {
  const candidate = catalogJob(relative.replace(/\.(?:webp|avif)$/i, '.png'));
  if (candidate && (candidate.source === relative || candidate.outputs.some((output) => outputPathFor(candidate.source, output) === relative))) return candidate;
  return staticJobs.find((job) => job.source === relative || job.outputs.some((output) => outputPathFor(job.source, output) === relative)) || null;
}

async function renderVariant(source, output) {
  let pipeline = sharp(source, pipelineOptions.input);
  if (pipelineOptions.rotate) pipeline = pipeline.rotate();
  if (Number.isFinite(Number(output.width)) && Number(output.width) > 0) {
    pipeline = pipeline.resize({ ...pipelineOptions.resize, width: Number(output.width) });
  }
  return (output.format === 'avif' ? pipeline.avif(output.options) : pipeline.webp(output.options)).toBuffer();
}

module.exports = { staticJobs, catalogDirectories, getCatalogJobs, getImageJob, pipelineOptions, encoderVersions, outputPathFor, renderVariant };
