'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');

const trackerSource = fs.readFileSync('js/tools/job-application-tracker.js', 'utf8');
const helperStart = trackerSource.indexOf('const filterAttachmentsForRetry =');
const helperEnd = trackerSource.indexOf('\n\n  const uploadAttachment =', helperStart);
assert.ok(helperStart >= 0 && helperEnd > helperStart, 'attachment retry helper must remain independently testable');

const filterAttachmentsForRetry = Function(
  `'use strict';\n${trackerSource.slice(helperStart, helperEnd)}\nreturn filterAttachmentsForRetry;`
)();

const fileAttachment = (kind, name, size, sha256 = '') => ({
  kind,
  file: { name, size, ...(sha256 ? { sha256 } : {}) }
});

const savedAttachment = (kind, filename, size, sha256 = '') => ({
  kind,
  filename,
  size,
  ...(sha256 ? { sha256 } : {})
});

test('partial retry skips an already-saved approved attachment and uploads the remainder', () => {
  const resume = fileAttachment('resume', 'Daniel-Resume.pdf', 1200);
  const cover = fileAttachment('cover-letter', 'Daniel-Cover.pdf', 800);
  const result = filterAttachmentsForRetry(
    [resume, cover],
    [savedAttachment('resume', 'Daniel-Resume.pdf', 1200)]
  );

  assert.deepEqual(result.pending, [cover]);
  assert.equal(result.skippedCount, 1);
});

test('retry matching is one-for-one and does not suppress unapproved attachment kinds', () => {
  const first = fileAttachment('resume', 'Resume.pdf', 900);
  const second = fileAttachment('resume', 'Resume.pdf', 900);
  const supplemental = fileAttachment('attachment', 'Resume.pdf', 900);
  const result = filterAttachmentsForRetry(
    [first, second, supplemental],
    [savedAttachment('resume', 'Resume.pdf', 900)]
  );

  assert.deepEqual(result.pending, [second, supplemental]);
  assert.equal(result.skippedCount, 1);
});

test('retry requires matching SHA-256 when both records safely provide one', () => {
  const matchingHash = 'a'.repeat(64);
  const differentHash = 'b'.repeat(64);
  const candidate = fileAttachment('cover-letter', 'Cover.pdf', 700, matchingHash);

  assert.equal(
    filterAttachmentsForRetry(
      [candidate],
      [savedAttachment('cover-letter', 'Cover.pdf', 700, matchingHash)]
    ).pending.length,
    0
  );
  assert.deepEqual(
    filterAttachmentsForRetry(
      [candidate],
      [savedAttachment('cover-letter', 'Cover.pdf', 700, differentHash)]
    ).pending,
    [candidate]
  );
});

test('retry does not infer a duplicate without exact filename and safe byte size', () => {
  const candidate = fileAttachment('resume', 'Resume.pdf', 900);
  const result = filterAttachmentsForRetry(candidate ? [candidate] : [], [
    savedAttachment('resume', 'resume.pdf', 900),
    { kind: 'resume', filename: 'Resume.pdf' }
  ]);

  assert.deepEqual(result.pending, [candidate]);
  assert.equal(result.skippedCount, 0);
});
