'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '..');
const MODULES = ['numbers', 'content', 'collection-content', 'collections', 'expeditions', 'progression-content', 'area-skills-content', 'area-skills', 'station-content', 'progression-modifiers', 'progression', 'progression-purchases', 'stations', 'upgrade-tiers', 'practice-lessons', 'onboarding', 'trail-deliveries', 'core', 'persistence', 'debug-updates', 'billing', 'rewarded', 'icons', 'station-art', 'station-scene', 'scene', 'expedition-scene', 'onboarding-ui', 'area-motion', 'expedition-ui', 'app'];
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function bundle(output) {
  const destination = path.resolve(output);
  if (destination === ROOT || ROOT.startsWith(destination + path.sep)) throw new Error('Output must be a dedicated assets directory.');
  fs.mkdirSync(path.join(destination, 'wayfarers'), { recursive: true });
  const records = [];
  function write(relative, bytes, source) {
    const target = path.join(destination, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, bytes);
    records.push({ path: relative.replaceAll('\\', '/'), sha256: hash(bytes), source });
  }
  const authored = fs.readFileSync(path.join(ROOT, 'pages/games/wayfarers-guild.html'), 'utf8');
  const main = authored.match(/<main id="main" class="wg-app"[\s\S]*?<\/main>/)?.[0];
  if (!main || !main.includes('data-wayfarers-guild')) throw new Error('Canonical Guild markup is missing.');
  const game = main.replace(/<header class="personal-game-header[\s\S]*?<\/header>/, '');
  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<base href="/assets/"><meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'self'">
<title>Wayfarers’ Guild</title><link rel="stylesheet" href="wayfarers/game.css"><link rel="stylesheet" href="wayfarers/android.css">
${MODULES.map(name => `<script defer src="wayfarers/${name}.js"></script>${name === 'persistence' ? '\n<script defer src="wayfarers/native-checkpoint.js"></script>\n<script defer src="wayfarers/checkpoint.js"></script>' : ''}`).join('\n')}
<script defer src="wayfarers/android.js"></script></head><body class="wayfarers-guild-page">${game}</body></html>\n`;
  write('wayfarers/index.html', Buffer.from(html), 'pages/games/wayfarers-guild.html');
  // Native intercepts this same-origin script with its verified private checkpoint.
  // The empty fallback keeps exact bundled-content previews independently runnable.
  write('wayfarers/native-checkpoint.js', Buffer.from('window.WayfarersNativeCheckpoint=null;\n'), 'generated');
  write('wayfarers/game.css', fs.readFileSync(path.join(ROOT, 'css/games/wayfarers-guild.css')), 'css/games/wayfarers-guild.css');
  for (const name of MODULES) {
    const source = `js/games/wayfarers-guild/${name}.js`;
    write(`wayfarers/${name}.js`, fs.readFileSync(path.join(ROOT, source)), source);
  }
  for (const name of ['android.js', 'android.css', 'checkpoint.js']) {
    const source = `mobile/android/wayfarers/web/${name}`;
    write(`wayfarers/${name}`, fs.readFileSync(path.join(ROOT, source)), source);
  }
  for (const name of fs.readdirSync(path.join(ROOT, 'img/wayfarers-guild')).filter(name => /\.(png|webp)$/.test(name))) {
    const source = `img/wayfarers-guild/${name}`;
    write(source, fs.readFileSync(path.join(ROOT, source)), source);
  }
  write('wayfarers/bundle-manifest.json', Buffer.from(JSON.stringify({ schemaVersion: 1, origin: 'https://appassets.androidplatform.net', records }, null, 2) + '\n'), 'generated');
  return records;
}

if (require.main === module) {
  if (process.argv[2] !== '--output' || !process.argv[3] || process.argv.length !== 4) throw new Error('Usage: node build/bundle-wayfarers-android.cjs --output DIR');
  const records = bundle(process.argv[3]);
  process.stdout.write(`Bundled ${records.length} Guild files from canonical sources.\n`);
}
module.exports = { bundle, MODULES };
