'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const repositoryRoot = path.resolve(__dirname, '../..');
const toolIcon = /^img\/tools\/icons\/[a-z0-9_-]+\.(?:png|webp|avif)$/i;
const sheetPoster = /^img\/projects\/sheetMusicUpscale(?:-(?:640|960))?\.(?:png|webp|avif)$/;
const sheetStage = /^img\/projects\/sheetMusicUpscale-(?:original|watermark-removed|upscaled)-(?:full|comparison)\.webp$/;

function versionedImageUrl(value, { root = repositoryRoot } = {}) {
  if (typeof value !== 'string') return value;
  const pathname = value.replace(/[?#].*$/, '');
  const relative = pathname.replace(/^\//, '');
  if (!toolIcon.test(relative) && !sheetPoster.test(relative) && !sheetStage.test(relative)) return value;
  // Responsive encoders run after CMS generation. Their source PNG is the
  // stable generation key, so build order cannot fingerprint stale variants.
  const source = sheetPoster.test(relative) ? 'img/projects/sheetMusicUpscale.png' : relative;
  const hash = crypto.createHash('sha256').update(fs.readFileSync(path.join(root, source))).digest('hex').slice(0, 12);
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

module.exports = { versionedImageUrl, versionImageContent };
