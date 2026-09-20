'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { validateManifest } = require('../../mobile/android/scripts/prepare-app-update.cjs');

// Only a deliberately staged release manifest can become public. Never copy APKs,
// keystores, Gradle output, or the rest of the Android source tree into the site.
function copyAppUpdateFeeds(root, output) {
  let copied = 0;
  for (const channel of ['review', 'stable']) {
    const source = path.join(root, 'mobile/android/releases', channel, 'latest.json');
    const destination = path.join(output, 'app-updates', channel, 'latest.json');
    if (!fs.existsSync(source)) {
      // Withdrawing a staged feed must also remove a stale deployment copy.
      fs.rmSync(destination, { force: true });
      continue;
    }
    if (fs.statSync(source).size > 512 * 1024) throw new Error('App update manifest is too large');
    const manifest = JSON.parse(fs.readFileSync(source, 'utf8'));
    validateManifest(manifest);
    if (manifest.channel !== channel) throw new Error('App update channel does not match its directory');
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(source, destination);
    copied += 1;
  }
  return copied;
}

module.exports = { copyAppUpdateFeeds };
