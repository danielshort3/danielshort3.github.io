'use strict';

const fs = require('node:fs');
const path = require('node:path');

module.exports = function runPublicPageCleanupTests({ assert }) {
  const root = path.resolve(__dirname, '../..');
  const config = JSON.parse(fs.readFileSync(path.join(root, 'vercel.json'), 'utf8'));
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
