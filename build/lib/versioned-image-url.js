'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { getImageJob, pipelineOptions, encoderVersions } = require('./image-variant-recipes');
const repositoryRoot = path.resolve(__dirname, '../..');
const toolIcon = /^img\/tools\/icons\/[a-z0-9_-]+\.(?:png|webp|avif)$/i;
const projectIcon = /^img\/projects\/icons\/[a-z0-9_-]+\.png$/i;
const gameIcon = /^img\/games\/icons\/[a-z0-9_-]+\.png$/i;
const libraryIcon = /^img\/home-icons\/[a-z0-9_-]+\.svg$/i;
const sheetPoster = /^img\/projects\/sheetMusicUpscale(?:-(?:640|960))?\.(?:png|webp|avif)$/;
const sheetStage = /^img\/projects\/sheetMusicUpscale-(?:original|watermark-removed|upscaled)-(?:full|comparison)\.webp$/;
const projectPreview = /^img\/projects\/[a-z0-9-]+-preview\.webp$/i;
const websitePoster = /^img\/projects\/website(?:-(?:640|960))?\.(?:png|webp|avif)$/;

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).filter((key) => value[key] !== undefined).sort()
      .map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function imageGenerationFingerprint(bytes, job, { pipeline = pipelineOptions, versions = encoderVersions } = {}) {
  return crypto.createHash('sha256').update('ds-image-generation-v2\0').update(bytes)
    .update(stableJson({ source: job.source, outputs: job.outputs, pipeline, versions })).digest('hex').slice(0, 12);
}

function versionedImageUrl(value, { root = repositoryRoot, resolveJob = getImageJob } = {}) {
  if (typeof value !== 'string') return value;
  const pathname = value.replace(/[?#].*$/, '');
  const relative = pathname.replace(/^\//, '');
  const job = resolveJob(relative);
  if (!job && !toolIcon.test(relative) && !projectIcon.test(relative) && !gameIcon.test(relative) && !libraryIcon.test(relative) && !sheetPoster.test(relative) && !sheetStage.test(relative) && !projectPreview.test(relative) && !websitePoster.test(relative)) return value;
  // Generated families share source pixels, the complete encoder recipe and
  // encoder versions. Changing dimensions/quality invalidates cached variants
  // before encoding, even when the original PNG has not changed.
  const source = job ? job.source : sheetPoster.test(relative) ? 'img/projects/sheetMusicUpscale.png'
    : websitePoster.test(relative) ? 'img/projects/website.png' : relative;
  const bytes = fs.readFileSync(path.join(root, source));
  let hash = job ? imageGenerationFingerprint(bytes, job) : crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 12);
  if (!job && sheetPoster.test(relative)) {
    // This family is checked in rather than encoded by the site build. Include
    // its actual variants so independently replaced WebP/AVIF files also expire.
    const family = crypto.createHash('sha256').update('ds-checked-in-image-family-v2\0').update(bytes);
    for (const suffix of ['.avif', '.webp', '-640.avif', '-640.webp', '-960.avif', '-960.webp']) {
      const variant = `img/projects/sheetMusicUpscale${suffix}`;
      if (fs.existsSync(path.join(root, variant))) family.update(variant).update(fs.readFileSync(path.join(root, variant)));
    }
    hash = family.digest('hex').slice(0, 12);
  }
  const hashIndex = value.indexOf('#');
  const fragment = hashIndex >= 0 ? value.slice(hashIndex) : '';
  const queryIndex = value.indexOf('?');
  const query = queryIndex >= 0 ? value.slice(queryIndex + 1, hashIndex >= 0 ? hashIndex : undefined) : '';
  const parameters = new URLSearchParams(query);
  parameters.set('v', hash);
  return `${pathname}?${parameters}${fragment}`;
}

function versionImageContent(value, options = {}, seen = new WeakMap()) {
  if (typeof value === 'string') return versionedImageUrl(value, options);
  if (!value || typeof value !== 'object') return value;
  if (seen.has(value)) return seen.get(value);
  const copy = Array.isArray(value) ? [] : {};
  seen.set(value, copy);
  Object.entries(value).forEach(([key, entry]) => { copy[key] = versionImageContent(entry, options, seen); });
  return copy;
}

module.exports = { versionedImageUrl, versionImageContent, imageGenerationFingerprint };
