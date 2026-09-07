'use strict';

const assert = require('node:assert/strict');
const { normalizeYouTubeUrl, updateLocalSourceMetadata, isRecoverableItem, captureUrlSources } = require('../../js/tools/whisper-transcribe-monitor');

const canonical = 'https://www.youtube.com/watch?v=BaW_jenozKc';
for (const source of [
  canonical,
  ' https://youtube.com/watch?v=BaW_jenozKc&t=45#section ',
  'https://m.youtube.com/watch?v=BaW_jenozKc',
  'https://youtu.be/BaW_jenozKc?si=tracking',
  'https://www.youtube.com/shorts/BaW_jenozKc/',
  'https://youtube.com/live/BaW_jenozKc',
  'https://youtube.com/embed/BaW_jenozKc'
]) {
  assert.equal(normalizeYouTubeUrl(source), canonical);
}

for (const source of [
  null,
  { url: canonical },
  'http://www.youtube.com/watch?v=BaW_jenozKc',
  'https://music.youtube.com/watch?v=BaW_jenozKc',
  'https://youtube.com.evil.example/watch?v=BaW_jenozKc',
  'https://user:password@youtube.com/watch?v=BaW_jenozKc',
  'https://youtube.com:8443/watch?v=BaW_jenozKc',
  'https://127.0.0.1/watch?v=BaW_jenozKc',
  'https://youtube.com/watch?v=BaW_jenozKc&list=PLtest',
  'https://youtu.be/BaW_jenozKc?list=',
  'https://youtube.com/playlist?list=PLtest',
  'https://youtube.com/watch?v=BaW_jenozKc&v=BaW_jenozKc',
  'https://youtube.com/watch?v=short',
  'https://youtu.be/BaW_jenozKc/extra',
  'https://youtube.com\\@127.0.0.1/watch?v=BaW_jenozKc',
  `${canonical}\n`,
  `${canonical}&tracking=${'a'.repeat(2048)}`
]) {
  assert.throws(() => normalizeYouTubeUrl(source), /single public YouTube/);
}

const urlItem = { sourceKind: 'url', sourceUrl: canonical, name: 'YouTube video BaW_jenozKc', durationSeconds: null, bytes: null };
updateLocalSourceMetadata(urlItem, { stage: 'resolving', maxDurationSeconds: 28800 });
assert.equal(urlItem.durationSeconds, null, 'the allowed duration must not become the source duration');
updateLocalSourceMetadata(urlItem, { title: 'A useful video', filename: 'A useful video.webm', durationSeconds: 60.5, bytes: 12345 });
assert.equal(urlItem.name, 'A useful video');
assert.equal(urlItem.downloadName, 'A useful video.webm');
assert.equal(urlItem.durationSeconds, 60.5);
assert.equal(urlItem.bytes, 12345);
updateLocalSourceMetadata(urlItem, { durationSeconds: -1, bytes: 'unknown' });
assert.equal(urlItem.durationSeconds, 60.5);
assert.equal(urlItem.bytes, 12345);
const fileItem = { sourceKind: 'file', name: 'original.wav', durationSeconds: 42 };
updateLocalSourceMetadata(fileItem, { title: 'replacement', durationSeconds: 100 });
assert.deepEqual(fileItem, { sourceKind: 'file', name: 'original.wav', durationSeconds: 42 });
assert.equal(isRecoverableItem({ provider: 'local', sourceKind: 'url', status: 'failed', runErrorType: 'network', runToken: 'never-use-aws' }), false);

assert.deepEqual(captureUrlSources([
  fileItem,
  { ...urlItem, sourceUrl: `${canonical}&si=tracking`, status: 'complete', transcript: 'private transcript', localTicket: 'private ticket', runToken: 'private token' }
]), [{ sourceKind: 'url', sourceUrl: canonical, title: 'A useful video', durationSeconds: 60.5, status: 'complete' }],
'session metadata includes only canonical sources, never transcript bodies or job credentials');

console.log('transcribe-url-input tests passed');
