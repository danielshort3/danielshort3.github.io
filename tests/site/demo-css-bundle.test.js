'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { entries, inline, minify } = require('../../build/build-css');
const { collectDistArtifacts, rewriteCssLinksInHtml } = require('../../build/copy-to-public');
const { PROJECT_DEMO_IDS } = require('../../build/lib/project-demo-routes');

const root = path.resolve(__dirname, '../..');
const read = (relative) => fs.readFileSync(path.join(root, relative), 'utf8');
const demoImports = [
  'components/project-demo-theme.css',
  'components/handwriting-demo.css',
  'components/project-demo-layout.css',
  'components/project-demo-compact-layout.css',
  'components/project-demo-dashboard-layout.css'
];

function runDemoCssBundleTests({ assert, published = false }) {
  let checks = 0;
  const check = (condition, message) => { assert(condition, message); checks += 1; };
  const baseEntry = read('css/styles.css');
  const demoEntry = read('css/styles-demo.css');
  const imports = [...demoEntry.matchAll(/@import\s+url\("([^"]+)"\);/g)].map((match) => match[1]);
  check(JSON.stringify(imports) === JSON.stringify(demoImports), 'The demo entry must preserve all five unlayered imports in their original order.');
  check(demoImports.every((file) => !baseEntry.includes(file)), 'The shared entry must not eagerly import raw demo rules.');
  check(!/@layer\b/.test(demoEntry), 'The demo entry must retain its existing unlayered cascade priority.');
  const demoCss = minify(inline(path.join(root, 'css/styles-demo.css')));
  const expectedCss = minify(demoImports.map((file) => inline(path.join(root, 'css', file))).join('\n'));
  check(demoCss === expectedCss, 'The composed demo bundle must preserve every existing rule in order without wrapping it in a layer.');
  check(entries.some((entry) => entry.manifestKey === 'demoFile' && entry.baseName === 'styles-demo' &&
    entry.entry === path.join(root, 'css/styles-demo.css')), 'The CSS builder must publish a distinct hashed demo entry.');

  const fonts = read('css/base/fonts.css');
  check(/font-family:\s*'Inter';[\s\S]*?font-style:\s*normal;[\s\S]*?font-weight:\s*400 800;/.test(fonts) &&
    fs.statSync(path.join(root, 'css/fonts/Inter-Latin.woff2')).size > 0,
  'Shared local Inter must cover the normal 400–800 weights used by these demos.');
  const themedFiles = fs.readdirSync(path.join(root, 'demos')).filter((file) => file.endsWith('.html') &&
    read(`demos/${file}`).includes('data-project-demo-theme="brand"'));
  check(themedFiles.length === PROJECT_DEMO_IDS.length && PROJECT_DEMO_IDS.every((id) => themedFiles.includes(`${id}.html`)),
    'Every themed raw project demo must participate in the bundle split.');
  for (const file of themedFiles) {
    const html = read(`demos/${file}`);
    const baseIndex = html.indexOf('href="dist/styles.css"');
    const demoIndex = html.indexOf('href="dist/styles-demo.css"');
    const inlineStyleIndex = html.search(/<style\b/i);
    check(baseIndex >= 0 && demoIndex > baseIndex && (html.match(/href="dist\/styles-demo\.css"/g) || []).length === 1 &&
      (inlineStyleIndex < 0 || demoIndex < inlineStyleIndex), `${file} must load exactly one demo bundle after shared CSS and before its local styles.`);
    const redundantInter = [...html.matchAll(/<link\b[^>]*href="(https:\/\/fonts\.googleapis\.com\/[^"\s]+)"[^>]*>/g)]
      .some((match) => {
        const families = new URL(match[1].replace(/&amp;/g, '&')).searchParams.getAll('family');
        return families.length && families.every((family) => {
          const weights = /^Inter:wght@([\d;]+)$/.exec(family)?.[1];
          return weights && weights.split(';').every((weight) => Number(weight) >= 400 && Number(weight) <= 800);
        });
      });
    check(!redundantInter, `${file} must use the available local Inter face rather than a redundant font stylesheet.`);
  }
  check(read('demos/shape-demo.html').includes('cdnjs.cloudflare.com/ajax/libs/font-awesome/') &&
    read('demos/pizza-tips-demo.html').includes('css/vendor/leaflet/leaflet.css'), 'The split must preserve unrelated icon and map stylesheets.');

  const gameFiles = fs.readdirSync(path.join(root, 'pages/games')).filter((file) => file.endsWith('.html'))
    .map((file) => `pages/games/${file}`).concat('pages/ocean-wave-simulation.html');
  const demoOnlyHooks = /data-project-demo-theme|\b(?:drawing-demo|handwriting-rating-workspace|demo-size-dashboard|drawing-input|drawing-workspace)\b/;
  for (const file of gameFiles) {
    const html = read(file);
    check(!demoOnlyHooks.test(html) && !html.includes('styles-demo'),
      `${file} remains separate from the project demo theme and bundle.`);
  }

  const cssHrefs = { base: 'styles.12345678.css', demo: 'styles-demo.abcdef12.css' };
  const cssFixture = '<link href="dist/styles.css"><link href=\'dist/styles-demo.css\'><style>.fixture { color: red; }</style>';
  const rewritten = rewriteCssLinksInHtml(cssFixture, cssHrefs);
  check(rewritten === '<link href="dist/styles.12345678.css"><link href="dist/styles-demo.abcdef12.css"><style>.fixture { color: red; }</style>',
    'Public CSS rewriting must hash both entries while preserving their order and local styles.');
  check(rewriteCssLinksInHtml(rewritten, cssHrefs) === rewritten, 'Public CSS rewriting must be idempotent.');
  check(collectDistArtifacts({ file: cssHrefs.base, demoFile: cssHrefs.demo }, {}).includes(cssHrefs.demo) &&
    collectDistArtifacts({}, {}).includes('styles-demo.css'), 'Publication must include both hashed and legacy demo bundle names.');

  if (published) {
    const manifest = JSON.parse(read('dist/styles-manifest.json'));
    check(/^styles-demo\.[0-9a-f]{8}\.css$/.test(manifest.demoFile), 'The built CSS manifest must name a hashed demo bundle.');
    const built = read(`dist/${manifest.demoFile}`);
    check(built === demoCss && read(`public/dist/${manifest.demoFile}`) === built && read('public/dist/styles-demo.css') === built,
      'The authored, hashed, mirrored and legacy demo bundle contents must agree.');
    check(manifest.demoFile === `styles-demo.${crypto.createHash('sha256').update(built).digest('hex').slice(0, 8)}.css`,
      'The deployed demo filename must change when its contents change.');
    for (const id of PROJECT_DEMO_IDS) {
      const raw = read(`public/demos/${id}.html`);
      const baseIndex = raw.indexOf(`href="dist/${manifest.file}"`);
      const demoIndex = raw.indexOf(`href="dist/${manifest.demoFile}"`);
      const inlineStyleIndex = raw.search(/<style\b/i);
      check(baseIndex >= 0 && demoIndex > baseIndex && (inlineStyleIndex < 0 || demoIndex < inlineStyleIndex),
        `${id} must publish both hashed stylesheets before its local styles.`);
      check(!read(`public/pages/demos/${id}.html`).includes('styles-demo'), `${id} outer wrapper must retain only its own shell styles.`);
    }
    check(!read('public/pages/portfolio/digitGenerator.html').includes('styles-demo'),
      'Project detail pages must leave raw demo rules inside the iframe document.');
    for (const file of gameFiles) {
      check(!read(`public/${file}`).includes('styles-demo'), `${file} must not download the project demo bundle.`);
    }
  }
  return checks;
}

module.exports = runDemoCssBundleTests;
if (require.main === module) {
  const checks = runDemoCssBundleTests({ assert: require('node:assert/strict'), published: process.argv.includes('--published') });
  console.log(`Demo CSS isolation: ${checks} checks passed.`);
}
