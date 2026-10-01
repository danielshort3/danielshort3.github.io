'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const routes = JSON.parse(read('build/route-component-styles.json'));
const shared = read('css/styles.css');
for (const name of ['contact-layout', 'project-image-viewer']) {
  assert(!shared.includes(`components/${name}.css`));
}
assert(routes['/contact'].includes('css/components/contact-layout.css'));
assert(routes['/portfolio/deliveryTip'].includes('css/components/project-image-viewer.css'));

if (process.argv.includes('--published')) {
  const html = read('public/index.html');
  assert(!html.includes('css/components/contact-layout.css'));
  assert(!html.includes('css/components/project-image-viewer.css'));
  assert(read('public/pages/contact.html').includes('css/components/contact-layout.css?v='));
  assert(read('public/pages/portfolio/deliveryTip.html').includes('css/components/project-image-viewer.css?v='));
}
console.log('Route loading checks passed.');
