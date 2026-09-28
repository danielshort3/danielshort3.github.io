'use strict';

const fs = require('fs');
const path = require('path');
const {
  extractMainHtml,
  finalizePersonalRouteDocument,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
} = require('../../build/lib/personal-accordion-shell');

const ROOT = path.resolve(__dirname, '../..');
const count = (html, pattern) => (html.match(pattern) || []).length;
const manifest = (html) => JSON.parse(/<script\b[^>]*id="site-route-manifest"[^>]*>([\s\S]*?)<\/script>/i.exec(html)?.[1] || '{}');
const rails = (html) => [...html.matchAll(/<a\b[^>]*data-site-tab-category="([^"]+)"[^>]*>/g)]
  .map((match) => ({ category: match[1], tag: match[0] }));

function sampleDocument(page = 'project', route = '/portfolio/example') {
  return `<!DOCTYPE html><html class="no-js"><head><base href="/">
<link rel="canonical" href="https://www.danielshort.me${route}">
<meta name="referrer" content="strict-origin-when-cross-origin">
<link rel="stylesheet" href="css/components/project-page.css">
<script defer src="js/common/certifications-modal.js"></script>
</head><body data-page="${page}"><a class="skip-link" href="#main">Skip to content</a>
<header id="combined-header-nav"><nav>Navigation</nav></header>
<main id="main"><h1>Original content</h1><p>Supporting evidence.</p>
<form action="/api/contact"><label for="message">Message</label><textarea id="message" name="message">Draft</textarea></form>
<a href="/portfolio/website">Case study</a></main>
<footer><button id="privacy-settings-link-footer">Cookie settings</button></footer></body></html>`;
}

module.exports = function runSharedThemeShellTests({ assert, verifyGenerated = true }) {
  const sample = sampleDocument();
  const options = { category: 'projects', itemId: 'example', navigation: 'soft', backHref: '/portfolio' };
  const output = wrapPersonalAccordionHtml(sample, options);
  const railLinks = rails(output);
  assert(count(output, /data-personal-accordion-shell(?:\s|>)/g) === 1 && count(output, /<main\b/gi) === 1,
    'Canonical projects must contain one shared shell and one main landmark.');
  assert(railLinks.length === 5 && railLinks.map((rail) => rail.category).join(',') === 'about,projects,tools,games,contact',
    'Canonical projects retain all five site sections.');
  assert(railLinks.filter((rail) => rail.tag.includes('aria-current="page"')).length === 1 &&
    railLinks.find((rail) => rail.category === 'projects').tag.includes('data-personal-transition="collapse"'),
  'The active rail is the single visible, collapsible section.');
  assert(railLinks.filter((rail) => !rail.tag.includes('hidden')).length === 1 &&
    railLinks.filter((rail) => rail.tag.includes('hidden inert')).length === 4,
  'Detail pages retain their existing one-rail layout.');
  assert(!railLinks.some((rail) => /audience=|resume|tourism|data-science/.test(rail.tag)) &&
    !output.includes('data-site-tab-rail-mode="navigation"'),
  'The shared shell does not expose retired audience navigation.');
  assert(manifest(output).id === 'projects:example' && manifest(output).module === 'page:content' &&
    manifest(output).navigation === 'soft' && output.includes('data-audience="personal"'),
  'The project keeps canonical route identity and content lifecycle.');
  assert(output.includes('href="/portfolio" aria-label="Back to Projects"') &&
    output.includes('name="referrer" content="strict-origin-when-cross-origin"') &&
    output.includes('href="css/components/project-page.css"') &&
    output.includes('src="js/common/certifications-modal.js"'),
  'Wrapping retains the library return, policy, and dependencies.');
  assert(extractMainHtml(output) === extractMainHtml(sample) &&
    extractMainHtml(unwrapPersonalAccordionHtml(output)) === extractMainHtml(sample),
  'Wrapping and unwrapping preserve page content and form fields.');
  assert(wrapPersonalAccordionHtml(output, options) === output,
    'Repeated wrapping does not duplicate the shell.');
  assert(manifest(finalizePersonalRouteDocument(output, { home: false })).module === 'page:content',
    'Refreshing bundle metadata preserves the route module.');

  for (const tool of ['background-remover', 'transcribe', 'job-application-tracker']) {
    const toolOutput = wrapPersonalAccordionHtml(sampleDocument(tool, '/tools/' + tool), {
      category: 'tools', itemId: tool
    });
    assert(manifest(toolOutput).navigation === 'hard' && manifest(toolOutput).module === 'tools:' + tool,
      `${tool} retains its document security and runtime boundary.`);
  }
  const workbench = wrapPersonalAccordionHtml(sampleDocument('portfolio', '/portfolio')
    .replace('<main id="main">', '<main id="main" data-portfolio-workbench>'), {
    category: 'projects', itemId: 'portfolio'
  });
  assert(manifest(workbench).module === 'portfolio:workbench',
    'The canonical project library resolves its reusable controller.');
  const search = wrapPersonalAccordionHtml(sampleDocument('search', '/search'), {
    category: 'about', itemId: 'search'
  });
  assert(manifest(search).module === 'search:search',
    'The canonical search page resolves its shared controller.');

  if (!verifyGenerated) return;
  for (const route of ['analytics', 'data-science', 'tourism', 'resume', 'resume-pdf']) {
    assert(!fs.existsSync(path.join(ROOT, 'pages', `${route}.html`)) &&
      !fs.existsSync(path.join(ROOT, 'public', 'pages', `${route}.html`)),
    `Retired ${route} page is not published.`);
  }
  assert(!fs.existsSync(path.join(ROOT, 'pages/professional')) &&
    !fs.existsSync(path.join(ROOT, 'public/pages/professional')),
  'Retired professional project copies are not published.');
  const fallback = fs.readFileSync(path.join(ROOT, 'dshort.html'), 'utf8');
  assert(fallback.includes('data-personal-category="about"') &&
    count(fallback, /data-personal-accordion-shell(?:\s|>)/g) === 1,
  'The short-link failure page keeps the canonical About shell.');
};

if (require.main === module) {
  let total = 0;
  module.exports({
    verifyGenerated: !process.argv.includes('--generator-only'),
    assert(condition, message) { total += 1; require('assert').ok(condition, message); }
  });
  process.stdout.write(`Shared theme shell tests passed (${total} assertions).\n`);
}
