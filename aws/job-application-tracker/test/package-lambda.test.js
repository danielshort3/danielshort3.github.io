'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  createDeterministicArchive,
  hashFile,
  isExcludedDependencyPath,
  readZipEntries,
  validateArchiveEntries
} = require('../scripts/package-lambda');

test('Lambda archive validation allows only the handler and production dependency tree', () => {
  assert.deepEqual(
    validateArchiveEntries([
      'node_modules/example/index.js',
      'index.js',
      'node_modules/example/package.json'
    ]),
    ['index.js', 'node_modules/example/index.js', 'node_modules/example/package.json']
  );

  for (const blockedEntry of [
    'README.md',
    'package.json',
    'test.js',
    'template.yaml',
    'job-application-tracker.zip',
    'scripts/package-lambda.js',
    '../index.js',
    'node_modules\\example\\other.js',
    'node_modules/example/tests/example.test.js',
    'node_modules/example/docs/guide.html',
    'node_modules/example/archive.zip'
  ]) {
    assert.throws(
      () => validateArchiveEntries(['index.js', 'node_modules/example/index.js', blockedEntry]),
      /allowlist|non-runtime|Unsafe/,
      `${blockedEntry} must be rejected`
    );
  }
});

test('dependency pruning rules reject documentation, tests, maps, examples, and nested ZIPs', () => {
  for (const blockedPath of [
    'example/README.md',
    'example/CHANGELOG',
    'example/dist/index.js.map',
    'example/docs/guide.html',
    'example/examples/demo.js',
    'example/test.js',
    'example/lib/value.spec.js',
    'example/fixture.zip'
  ]) {
    assert.equal(isExcludedDependencyPath(blockedPath), true, `${blockedPath} should be excluded`);
  }
  for (const runtimePath of [
    '@scope/example/package.json',
    '@scope/example/dist/index.js',
    'example/LICENSE',
    'example/lib/template.js'
  ]) {
    assert.equal(isExcludedDependencyPath(runtimePath), false, `${runtimePath} should remain`);
  }
});

test('archive creation is deterministic and its central directory is inspectable', async () => {
  const temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'lambda-package-test-'));
  try {
    fs.mkdirSync(path.join(temporaryDirectory, 'node_modules', 'example'), { recursive: true });
    fs.writeFileSync(path.join(temporaryDirectory, 'index.js'), "module.exports.handler = async () => ({ statusCode: 200 });\n");
    fs.writeFileSync(path.join(temporaryDirectory, 'node_modules', 'example', 'index.js'), "module.exports = 'ok';\n");
    fs.writeFileSync(path.join(temporaryDirectory, 'node_modules', 'example', 'package.json'), '{"name":"example"}\n');

    const entries = validateArchiveEntries([
      'index.js',
      'node_modules/example/index.js',
      'node_modules/example/package.json'
    ]);
    const firstZip = path.join(temporaryDirectory, 'first.zip');
    const secondZip = path.join(temporaryDirectory, 'second.zip');
    await createDeterministicArchive(temporaryDirectory, entries, firstZip);
    await createDeterministicArchive(temporaryDirectory, entries, secondZip);

    assert.equal(hashFile(firstZip), hashFile(secondZip));
    assert.deepEqual(validateArchiveEntries(readZipEntries(firstZip)), entries);
  } finally {
    fs.rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});
