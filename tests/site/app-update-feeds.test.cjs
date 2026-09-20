'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { copyAppUpdateFeeds } = require('../../build/lib/app-update-feeds.cjs');

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'app-update-feed-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(root, 'mobile/android/releases/review');
  const output = path.join(root, 'public');
  fs.mkdirSync(source, { recursive: true });
  const manifest = {
    schemaVersion: 1, channel: 'review', packageName: 'me.danielshort.app.debug',
    latest: { versionCode: 3, versionName: '0.3.0-debug', minSdk: 26, signerSha256: 'a'.repeat(64), apk: {
      url: 'https://github.com/danielshort3/danielshort3.github.io/releases/download/android-v0.3.0-review/app.apk',
      sha256: 'b'.repeat(64), size: 100
    } },
    releases: [{ versionCode: 3, sha256: 'b'.repeat(64), size: 100, signerSha256: 'a'.repeat(64) }], patches: []
  };
  const write = () => fs.writeFileSync(path.join(source, 'latest.json'), JSON.stringify(manifest));
  return { root, source, output, manifest, write };
}

test('copies only the explicitly staged channel manifest, never Android build files', t => {
  const f = fixture(t);
  f.write();
  fs.writeFileSync(path.join(f.source, 'do-not-publish.apk'), 'private');
  fs.writeFileSync(path.join(f.source, 'signing.keystore'), 'private');
  assert.equal(copyAppUpdateFeeds(f.root, f.output), 1);
  const published = path.join(f.output, 'app-updates/review');
  assert.deepEqual(fs.readdirSync(published), ['latest.json']);
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(published, 'latest.json'))), f.manifest);
});

test('an absent source feed stays unpublished and removes a stale deployment manifest', t => {
  const f = fixture(t);
  assert.equal(copyAppUpdateFeeds(f.root, f.output), 0);
  f.write();
  copyAppUpdateFeeds(f.root, f.output);
  fs.unlinkSync(path.join(f.source, 'latest.json'));
  assert.equal(copyAppUpdateFeeds(f.root, f.output), 0);
  assert.equal(fs.existsSync(path.join(f.output, 'app-updates/review/latest.json')), false);
});

test('invalid release metadata fails the build instead of being published', t => {
  const f = fixture(t);
  f.manifest.latest.apk.sha256 = 'invalid';
  f.write();
  assert.throws(() => copyAppUpdateFeeds(f.root, f.output));
  assert.equal(fs.existsSync(path.join(f.output, 'app-updates/review/latest.json')), false);
});

test('a manifest cannot be published to the wrong channel', t => {
  const f = fixture(t);
  f.manifest.channel = 'stable';
  f.write();
  assert.throws(() => copyAppUpdateFeeds(f.root, f.output));
});
