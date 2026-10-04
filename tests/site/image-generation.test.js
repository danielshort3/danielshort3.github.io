'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sharp = require('sharp');
const { imageFixture, digest } = require('./helpers/image-generation-fixture.cjs');
const { imageGenerationFingerprint, versionedImageUrl } = require('../../build/lib/versioned-image-url');
const { pipelineOptions, encoderVersions, getImageJob, getCatalogJobs, renderVariant } = require('../../build/lib/image-variant-recipes');

async function run() {
  const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'image-generation-'));
  try {
    const fixture = await imageFixture(fixtureRoot);
    const { png, current } = fixture;
    assert.equal(digest(fs.readFileSync(fixture.filename)), fixture.pngHash, 'Both encoded generations must use an unchanged PNG.');
    assert.notEqual(fixture.oldHash, fixture.newHash, 'The regression must encode actually different WebP bytes.');
    assert.equal((await sharp(fixture.oldWebp).metadata()).width, 256);
    assert.equal((await sharp(fixture.newWebp).metadata()).width, 128);
    assert.notEqual(fixture.legacyUrl, fixture.newUrl, 'The first recipe-aware release must invalidate legacy PNG-only versions.');
    assert.notEqual(fixture.oldGenerationUrl, fixture.newUrl, 'Changing dimensions/quality without changing PNG pixels must invalidate the WebP URL.');
    assert.equal(versionedImageUrl(fixture.newPngUrl, { root: fixtureRoot }), fixture.newPngUrl, 'Unchanged source and recipe versions remain idempotent.');
    assert.equal(versionedImageUrl(fixture.newUrl, { root: fixtureRoot }).split('?')[1], fixture.newPngUrl.split('?')[1],
      'The generated PNG and WebP share the complete generation key before encoding.');

    const qualityOnly = JSON.parse(JSON.stringify(current));
    qualityOnly.outputs[0].options.quality = 55;
    const qualityWebp = await renderVariant(png, qualityOnly.outputs[0]);
    assert.notEqual(digest(qualityWebp), fixture.newHash, 'Changing quality alone must actually alter the encoded bytes.');
    assert.notEqual(imageGenerationFingerprint(png, current), imageGenerationFingerprint(png, qualityOnly), 'Quality changes invalidate generation keys.');
    assert.notEqual(imageGenerationFingerprint(png, current), imageGenerationFingerprint(png, current, {
      pipeline: { ...pipelineOptions, rotate: false }
    }), 'Pixel pipeline changes invalidate generation keys.');
    assert.notEqual(imageGenerationFingerprint(png, current), imageGenerationFingerprint(png, current, {
      versions: { ...encoderVersions, webp: 'future-encoder-fixture' }
    }), 'Encoder upgrades invalidate generation keys.');
    const reordered = { outputs: current.outputs.map((output) => ({ ...output, options: Object.fromEntries(Object.entries(output.options).reverse()) })), source: current.source };
    assert.equal(imageGenerationFingerprint(png, current), imageGenerationFingerprint(png, reordered), 'Equivalent recipe key ordering does not churn cache URLs.');

    const repositoryRoot = path.resolve(__dirname, '../..');
    for (const job of getCatalogJobs(repositoryRoot)) assert.deepEqual(getImageJob(job.source), job,
      'Encoding and fingerprinting must resolve the same catalog recipe.');
    const roulette = getImageJob('img/games/icons/roulette.png');
    const encodedRoulette = await renderVariant(path.join(repositoryRoot, roulette.source), roulette.outputs[0]);
    assert(encodedRoulette.equals(fs.readFileSync(path.join(repositoryRoot, 'img/games/icons/roulette.webp'))),
      'Moving recipe ownership must preserve the already approved actual icon bytes.');

    // Checked-in variants also require invalidation independently of their PNG.
    const posterDirectory = path.join(fixtureRoot, 'img/projects');
    fs.mkdirSync(posterDirectory, { recursive: true });
    fs.writeFileSync(path.join(posterDirectory, 'sheetMusicUpscale.png'), png);
    const posterVariant = path.join(posterDirectory, 'sheetMusicUpscale-640.webp');
    fs.writeFileSync(posterVariant, fixture.oldWebp);
    const posterBefore = versionedImageUrl('/img/projects/sheetMusicUpscale.png', { root: fixtureRoot });
    fs.writeFileSync(posterVariant, fixture.newWebp);
    const posterAfter = versionedImageUrl('/img/projects/sheetMusicUpscale.png', { root: fixtureRoot });
    assert.notEqual(posterBefore, posterAfter, 'Replacing a checked-in variant must invalidate its family even with an unchanged PNG.');
    assert.equal(versionedImageUrl('/img/projects/sheetMusicUpscale-640.webp', { root: fixtureRoot }).split('?')[1], posterAfter.split('?')[1]);
    console.log(`Image generations: unchanged PNG; ${fixture.oldWebp.length} -> ${fixture.newWebp.length} WebP bytes; recipe, encoder, checked-in family and stable-key checks passed.`);
  } finally {
    fs.rmSync(fixtureRoot, { recursive: true, force: true });
  }
}

if (require.main === module) run().catch((error) => { console.error(error); process.exitCode = 1; });
module.exports = run;
