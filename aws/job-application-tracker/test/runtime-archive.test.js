'use strict';

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { EventEmitter, once } = require('node:events');
const path = require('node:path');
const { PassThrough, Writable } = require('node:stream');
const test = require('node:test');

process.env.APPLICATIONS_TABLE = process.env.APPLICATIONS_TABLE || 'job-application-tracker-test';

const {
  createZipArchive,
  finishArchiveUpload,
  monitorArchiveCompletion
} = require('../index').__test;

test('CommonJS runtime loads and lazily imports Archiver when synchronous ESM require is disabled', () => {
  const runtimePath = path.resolve(__dirname, '..', 'index.js');
  const script = [
    "process.env.APPLICATIONS_TABLE='job-application-tracker-child-test';",
    '(async () => {',
    `  const runtime = require(${JSON.stringify(runtimePath)});`,
    '  const archive = await runtime.__test.createZipArchive({ zlib: { level: 1 } });',
    "  if (archive.constructor.name !== 'ZipArchive') throw new Error('Unexpected archive constructor.');",
    '  archive.abort();',
    '})().catch(error => {',
    '  console.error(error);',
    '  process.exitCode = 1;',
    '});'
  ].join('\n');
  const result = spawnSync(
    process.execPath,
    ['--no-experimental-require-module', '-e', script],
    { encoding: 'utf8' }
  );

  assert.equal(result.status, 0, result.stderr || result.stdout);
});

test('runtime export creates a ZIP with the Archiver 8 ZipArchive constructor API', async () => {
  const chunks = [];
  const output = new Writable({
    write(chunk, encoding, callback) {
      chunks.push(Buffer.from(chunk));
      callback();
    }
  });
  const archive = await createZipArchive({ zlib: { level: 6 } });
  const finished = once(output, 'finish');

  archive.pipe(output);
  archive.append('job application export', { name: 'applications.txt' });
  await archive.finalize();
  await finished;

  const zip = Buffer.concat(chunks);
  assert.equal(zip.readUInt32LE(0), 0x04034b50, 'archive should start with a ZIP local-file header');
  assert.notEqual(zip.indexOf(Buffer.from('applications.txt')), -1, 'archive should contain the requested entry');
  assert.notEqual(zip.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06])), -1,
    'archive should contain a ZIP end-of-central-directory record');
});

test('runtime export completion preserves the first archive error', async () => {
  const archive = new EventEmitter();
  const output = new PassThrough();
  const completion = monitorArchiveCompletion(archive, output);
  const failure = new Error('archive stream failed');

  archive.emit('error', failure);
  output.emit('finish');

  const result = await completion;
  assert.equal(result.error, failure);
});

test('runtime export finalization propagates archive and upload failures', async (t) => {
  for (const failureSource of ['archive', 'upload']) {
    await t.test(failureSource, async () => {
      const failure = new Error(`${failureSource} failed`);
      const context = {
        archive: { finalize: async () => {} },
        archiveResult: Promise.resolve({ error: failureSource === 'archive' ? failure : null }),
        uploadResult: Promise.resolve({ error: failureSource === 'upload' ? failure : null })
      };

      await assert.rejects(finishArchiveUpload(context), error => error === failure);
    });
  }
});
