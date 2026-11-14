'use strict';

const fs = require('node:fs');
const path = require('node:path');

module.exports = function runPublicPageCleanupTests({ assert }) {
  const root = path.resolve(__dirname, '../..');
  const config = JSON.parse(fs.readFileSync(path.join(root, 'vercel.json'), 'utf8'));
  const exclusions = require('../../build/public-deployment-exclusions.json');
  const { shouldSkipPublicCopy } = require('../../build/copy-to-public.js');
  const { collectAssetPaths, createAssetPreviewCatalog } = require('../../js/games/project-starfall/engine/assets.js');
  const starfallData = require('../../js/games/project-starfall/project-starfall-data.js');
  const enemyInventory = require('../../asset-sources/project-starfall/overhaul-v1/enemies/inventory.json');
  const activeStarfallPaths = new Set([
    ...collectAssetPaths(starfallData),
    ...createAssetPreviewCatalog(starfallData).map(entry => entry.path)
  ]);
  const excludedPaths = new Set(exclusions);
  assert(exclusions.length === 111 && excludedPaths.size === 111,
    'Deployment exclusions must contain only the 111 reviewed file paths');
  for (const file of exclusions) {
    assert(typeof file === 'string' && !path.isAbsolute(file) && !/[\\*?]/.test(file)
      && !file.split('/').some(part => !part || part === '.' || part === '..'),
    `${file} must be one exact relative file path, without wildcards or traversal`);
    assert(fs.statSync(path.join(root, file)).isFile(), `${file} source must be retained`);
    assert(!fs.existsSync(path.join(root, 'public', file)), `${file} deployment copy must be absent`);
    assert(shouldSkipPublicCopy(path.join(root, file)), `${file} must be omitted by the copier`);
    assert(!shouldSkipPublicCopy(path.join(root, `${file}.retained`)),
      `${file} exclusion must not match neighboring filenames`);
    assert(!activeStarfallPaths.has(file), `${file} must not be in the active Starfall asset registry`);
  }
  for (const { replacement } of enemyInventory.items) {
    if (!replacement) continue;
    assert(excludedPaths.has(replacement.oldSheet), `${replacement.oldSheet} must stay retired`);
  }
  for (const file of activeStarfallPaths) {
    if (!/^img\/project-starfall\/animations\/(enemies|players)\//.test(file)) continue;
    assert(fs.existsSync(path.join(root, 'public', file)), `${file} active animation must remain published`);
  }
  for (const source of ['/sitemap', '/sitemap.html']) {
    assert(config.rewrites.some(rule => rule.source === source && rule.destination === '/pages/sitemap'),
      `${source} must retain its canonical Sitemap rewrite`);
  }
  for (const file of [
    'pages/sitemap.html', 'sitemap.xml', 'sitemap.xsl',
    'img/brand/analytics-work-experience-bg.png',
    'img/brand/analytics-certifications-bg.png',
    'img/project-starfall/items/item-visual-manifest.json',
    'img/hero/about-french-horn-isolated.source.txt',
    'img/games/ocean/kloppenheim_06_puresky_4k.hdr'
  ]) {
    assert(!shouldSkipPublicCopy(path.join(root, file)) && fs.existsSync(path.join(root, 'public', file)),
      `${file} must remain published for current routes, runtime, attribution or shared packaging`);
  }
  for (const file of ['privacy.html', 'pages/sitemap-pretty.html']) {
    assert(!fs.existsSync(path.join(root, file)), `${file} must stay retired`);
    assert(!fs.existsSync(path.join(root, 'public', file)), `${file} must not be published`);
  }
  for (const source of ['/sitemap-pretty', '/sitemap-pretty.html', '/pages/sitemap-pretty', '/pages/sitemap-pretty.html']) {
    const rules = config.redirects.filter((rule) => rule.source === source);
    assert(rules.length === 1 && rules[0].destination === '/sitemap' && rules[0].permanent === true,
      `${source} must permanently resolve to the standard Sitemap`);
    assert(!config.rewrites.some((rule) => rule.source === source), `${source} must not target a deleted page`);
  }
  for (const source of ['/privacy', '/privacy.html']) {
    assert(config.rewrites.some((rule) => rule.source === source && rule.destination === '/pages/privacy'),
      `${source} must retain the canonical Privacy page`);
  }
  for (const file of ['pages/privacy.html', 'pages/sitemap.html']) {
    assert(fs.existsSync(path.join(root, file)) && fs.existsSync(path.join(root, 'public', file)),
      `${file} must remain available`);
  }
  for (const audience of ['analytics', 'data-science', 'tourism']) {
    for (const prefix of ['/', '/pages/']) {
      for (const suffix of ['', '.html', '-pdf', '-pdf.html']) {
        const source = `${prefix}resume-${audience}${suffix}`;
        assert(config.redirects.some((rule) => rule.source === source && rule.destination === '/' && rule.permanent === true),
          `${source} must retain the existing public-site destination`);
      }
    }
    assert(!fs.existsSync(path.join(root, 'public', 'pages', `${audience}.html`)),
      'Cleanup must not publish private professional homes on the main website');
  }
};

if (require.main === module) {
  let checks = 0;
  module.exports({ assert(condition, message) {
    require('node:assert/strict').ok(condition, message);
    checks += 1;
  } });
  console.log(`Public page cleanup: ${checks} checks passed.`);
}
