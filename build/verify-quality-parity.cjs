'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const files = [
  'sw.js', 'js/common/session-drafts.js', 'js/forms/contact.js',
  'js/accounts/tools-account-ui.js', 'js/analytics/ga4-events.js',
  'js/analytics/web-vitals.js', 'js/common/common.js',
  'js/tools/image-optimizer.js', 'js/tools/background-remover.js',
  'css/components/draft-recovery.css', 'css/components/project-page.css',
  'css/components/project-demo-compact-layout.css', 'css/components/site-modal-theme.css',
  'css/components/mobile-site-dock.css', 'css/components/tools-account.css',
  'demos/handwriting-rating-demo.html', 'demos/shape-demo.html', 'demos/digit-generator-demo.html',
  'pages/text-compare.html', 'pages/privacy.html', 'index.html'
];
for (const directory of ['pages/portfolio', 'pages/demos']) {
  for (const file of fs.readdirSync(path.join(root, directory))) {
    if (file.endsWith('.html')) files.push(`${directory}/${file}`);
  }
}
const styles = JSON.parse(fs.readFileSync(path.join(root, 'dist/styles-manifest.json')));
for (const manifest of ['styles-manifest.json', 'scripts-manifest.json']) {
  const data = JSON.parse(fs.readFileSync(path.join(root, 'dist', manifest)));
  // copy-to-public intentionally removes the retired Contributions feature.
  if (manifest === 'scripts-manifest.json' && data.contributions) {
    assert(!fs.existsSync(path.join(root, 'public/dist', data.contributions)), 'Retired Contributions bundle must not be published');
    delete data.contributions;
  }
  files.push(`dist/${manifest}`);
  for (const value of Object.values(data)) if (typeof value === 'string' && /\.(css|js)$/.test(value)) files.push(`dist/${value}`);
}
for (const file of files) {
  let expected = fs.readFileSync(path.join(root, file));
  if (file === 'dist/scripts-manifest.json') {
    const manifest = JSON.parse(expected.toString('utf8'));
    delete manifest.contributions;
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(root, 'public', file), 'utf8')), manifest, `${file}: public manifest differs from the publishing policy`);
    continue;
  }
  if (file.endsWith('.html')) {
    // copy-to-public resolves authored legacy CSS names to the current bundle.
    let html = expected.toString('utf8');
    for (const value of Object.values(styles)) {
      const legacy = value.replace(/\.[a-f0-9]{8}\.css$/, '.css');
      html = html.replaceAll(`href="dist/${legacy}"`, `href="dist/${value}"`);
    }
    expected = Buffer.from(html);
  }
  assert(expected.equals(fs.readFileSync(path.join(root, 'public', file))), `${file}: public does not match authoritative build output`);
}
console.log(`Source/public parity passed for ${files.length} changed sources, pages and bundle assets.`);
