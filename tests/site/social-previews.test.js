'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const sharp = require('sharp');
const { loadSocialPreviewRecords, renderSocialPreview, socialPreviewMetadata } = require('../../build/lib/social-previews');
const { formatResourceLabel } = require('../../build/generate-project-pages');

async function main() {
  const records = loadSocialPreviewRecords();
  assert(records.length >= 15, 'Public tools and games have social cards.');
  assert.strictEqual(new Set(records.map(record => record.file)).size, records.length, 'Each public route has a distinct card.');
  assert(!records.some(record => record.pathname.includes('short-links')), 'Administrator tools are not included in public cards.');
  let first;
  for (const record of records) {
    const image = await renderSocialPreview(record);
    const metadata = await sharp(image).metadata();
    assert.strictEqual(metadata.width, 1200, record.id);
    assert.strictEqual(metadata.height, 630, record.id);
    assert.strictEqual(metadata.format, 'png', record.id);
    if (!first) first = image;
  }
  assert(first.equals(await renderSocialPreview(records[0])), 'Identical source content renders identical PNG bytes.');

  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'danielshort-social-test-'));
  try {
    const record = records[0];
    const destination = path.join(temporaryRoot, record.file);
    const options = { root: temporaryRoot, siteOrigin: 'https://example.com' };
    assert.strictEqual(socialPreviewMetadata(record, options), null, 'Missing cards retain fallback metadata.');
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.writeFileSync(destination, first);
    const metadata = socialPreviewMetadata(record, options);
    assert(metadata.url.startsWith(`https://example.com/${record.file}?v=`));
    assert.strictEqual(metadata.alt, record.alt);
    assert.strictEqual(metadata.type, 'image/png');
    fs.writeFileSync(destination, Buffer.concat([first, Buffer.from('changed')]));
    assert.notStrictEqual(socialPreviewMetadata(record, options).url, metadata.url, 'Changed artwork changes the social image URL.');
  } finally {
    assert.strictEqual(path.dirname(temporaryRoot), path.resolve(os.tmpdir()));
    assert(path.basename(temporaryRoot).startsWith('danielshort-social-test-'));
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }

  assert.strictEqual(formatResourceLabel({ label: 'PDFs', url: 'https://example.com/report.pdf' }), 'Project report · PDF');
  assert.strictEqual(formatResourceLabel({ label: 'PDFs', url: 'https://example.com/reports.zip' }), 'Project reports · ZIP');
  assert.strictEqual(formatResourceLabel({ label: 'Notebook', url: 'https://example.com/notebooks.zip' }), 'Notebooks · ZIP');
  assert.strictEqual(formatResourceLabel({ label: 'GitHub', url: 'https://github.com/danielshort3' }), 'GitHub');
  process.stdout.write(`Social previews passed: ${records.length} images, deterministic rendering, versioned metadata, and file labels.\n`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
