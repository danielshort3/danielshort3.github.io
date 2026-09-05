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
const AUDIENCES = ['analytics', 'data-science', 'tourism'];
const read = (file) => fs.readFileSync(path.join(ROOT, file), 'utf8');
const count = (html, pattern) => (html.match(pattern) || []).length;
const decode = (value) => String(value || '').replace(/&amp;/g, '&');
function attributes(tag) {
  const result = {};
  for (const match of tag.matchAll(/([\w-]+)(?:="([^"]*)")?/g)) result[match[1]] = decode(match[2] || '');
  return result;
}
function tags(html, name) {
  return (html.match(new RegExp(`<${name}\\b[^>]*>`, 'gi')) || []).map(attributes);
}
function manifest(html) {
  return JSON.parse(/<script\b[^>]*id="site-route-manifest"[^>]*>([\s\S]*?)<\/script>/i.exec(html)?.[1] || '{}');
}
function railLinks(html) {
  return tags(html, 'a').filter((tag) => Object.hasOwn(tag, 'data-site-tab-category'));
}
function contentHtml(html) {
  return extractMainHtml(html).replace(/\sdata-navigation="hard"/g, '');
}
function validateProfessionalShell(assert, html, audience, label) {
  const body = tags(html, 'body')[0] || {};
  const rails = railLinks(html);
  assert(count(html, /data-personal-accordion-shell(?:\s|>)/g) === 1 && count(html, /<main\b/gi) === 1,
    `${label} must contain one shared shell and one main landmark`);
  assert(body['data-audience'] === audience && body['data-site-route-navigation'] === 'soft' && manifest(html).navigation === 'soft',
    `${label} must preserve professional audience while using the persistent frame`);
  assert(html.includes('data-site-tab-rail-mode="navigation"') && rails.length === 4,
    `${label} must use the four professional navigation rails`);
  assert(rails.every((rail) => !Object.hasOwn(rail, 'hidden') && !Object.hasOwn(rail, 'inert') && rail['aria-hidden'] !== 'true' && rail.tabindex !== '-1'),
    `${label} navigation must work without JavaScript`);
  const expected = {
    about: '/' + audience,
    projects: '/portfolio?audience=' + audience,
    resume: '/resume-' + audience,
    contact: '/contact?audience=' + audience
  };
  assert(rails.every((rail) => rail.href === expected[rail['data-site-tab-category']]),
    `${label} rails must remain in the selected professional audience`);
  assert(rails.filter((rail) => rail['aria-current'] === 'page').length === 1,
    `${label} navigation must identify one current section`);
  const robots = tags(html, 'meta').find((tag) => tag.name === 'robots')?.content || '';
  assert(/\bnoindex\b/.test(robots) && /\bnofollow\b/.test(robots),
    `${label} must retain professional search-exclusion metadata`);
  const referrers = tags(html, 'meta').filter((tag) => tag.name === 'referrer');
  assert(referrers.length === 1 && referrers[0].content === 'no-referrer',
    `${label} must declare its professional referrer policy for same-document audience changes`);
}

function railsHaveNoHardNavigation(html) { return railLinks(html).every((rail) => rail['data-navigation'] !== 'hard'); }

function sampleDocument(audience, category) {
  const canonicalPath = category === 'resume' ? '/resume-' + audience : '/portfolio?audience=' + audience;
  return `<!DOCTYPE html>
<html class="no-js"><head><base href="/">
<link rel="canonical" href="https://www.danielshort.me${canonicalPath}">
<meta property="og:url" content="https://www.danielshort.me${canonicalPath}">
<meta name="robots" content="noindex, nofollow, nosnippet, noimageindex">
<meta name="referrer" content="no-referrer">
<link rel="stylesheet" href="css/components/resume.css">
<script defer src="js/common/certifications-modal.js"></script>
</head><body data-page="${category === 'resume' ? 'resume' : category === 'contact' ? 'contact' : category === 'projects' ? 'project' : audience}" data-audience="${audience}" data-internal-professional-copy="true">
<a class="skip-link" href="#main">Skip to content</a>
<header id="combined-header-nav"><nav>Legacy navigation</nav></header>
<main id="main"><h1>Original professional content</h1><p>Role description &amp; supporting evidence.</p>
<form action="/api/contact"><label for="message">Message</label><textarea id="message" name="message">Draft</textarea></form>
<a href="/portfolio/evidence?audience=${audience}">Case study</a>
<a href="https://example.test/resume.pdf" download>Download PDF</a>
<a href="/resume-${audience}-pdf"><img src="/img/resume-previews/${audience}.png" alt="Resume preview"></a>
<div id="certifications-modal" class="modal"><div class="modal-content">Credential evidence</div></div></main>
<footer><button id="privacy-settings-link-footer">Cookie settings</button></footer></body></html>`;
}

module.exports = function runSharedThemeShellTests({ assert, verifyGenerated = true }) {
  for (const audience of AUDIENCES) {
    for (const category of ['about', 'projects', 'resume', 'contact']) {
      const sample = sampleDocument(audience, category);
      const options = { audience, category, itemId: 'test-' + category, navigation: 'soft', chrome: 'compact', fit: 'document', backHref: '/' + audience };
      const output = wrapPersonalAccordionHtml(sample, options);
      const missingPolicy = wrapPersonalAccordionHtml(sample.replace('<meta name="referrer" content="no-referrer">', ''), options);
      assert(tags(missingPolicy, 'meta').find((tag) => tag.name === 'referrer')?.content === 'no-referrer',
        `${audience} ${category} must add a referrer policy when the shared personal source has none`);
      const label = audience + ' ' + category + ' generator';
      validateProfessionalShell(assert, output, audience, label);
      const expectedModule = category === 'contact' ? 'contact:contact' : 'page:content';
      assert(manifest(output).module === expectedModule && manifest(output).id !== expectedModule,
        `${label} must share a lifecycle module while keeping audience-specific route identity`);
      assert(railsHaveNoHardNavigation(output), `${label} must not leave generated hard navigation on normal section links`);
      const bundled = finalizePersonalRouteDocument(output, { home: false });
      assert(manifest(bundled).navigation === 'soft' && tags(bundled, 'body')[0]['data-site-route-navigation'] === 'soft',
        `${label} must retain soft navigation when the bundle-injection build pass refreshes the manifest`);
      assert(manifest(bundled).module === expectedModule, `${label} must preserve module identity during bundle injection`);
      assert(wrapPersonalAccordionHtml(output, options) === output, `${label} must produce identical output on repeated wrapping`);
      assert(contentHtml(output) === contentHtml(sample), `${label} must retain all original content, form state, credential markup, and document links`);
      assert(contentHtml(unwrapPersonalAccordionHtml(output)) === contentHtml(sample), `${label} must recover its original content for the next build`);
      assert(tags(output, 'meta').find((tag) => tag.name === 'robots')?.content === 'noindex, nofollow, nosnippet, noimageindex' &&
        tags(output, 'meta').find((tag) => tag.name === 'referrer')?.content === 'no-referrer',
      `${label} must preserve every supplied privacy directive`);
      assert(tags(output, 'link').find((tag) => tag.rel === 'canonical')?.href === tags(sample, 'link').find((tag) => tag.rel === 'canonical')?.href &&
        tags(output, 'meta').find((tag) => tag.property === 'og:url')?.content === tags(sample, 'meta').find((tag) => tag.property === 'og:url')?.content,
      `${label} must preserve canonical and social URL identity`);
      const skip = tags(output, 'a').find((tag) => (tag.class || '').split(' ').includes('skip-link'));
      const canonical = new URL(tags(sample, 'link').find((tag) => tag.rel === 'canonical').href);
      assert(skip?.href === canonical.pathname + canonical.search + '#main', `${label} skip link must stay on the audience route under the document base URL`);
      assert(tags(output, 'link').filter((tag) => tag.href === 'css/components/resume.css').length === 1 &&
        tags(output, 'script').filter((tag) => tag.src === 'js/common/certifications-modal.js').length === 1,
      `${label} must preserve one copy of its styles and behavior dependencies`);
    }
  }
  const personalSample = sampleDocument('personal', 'projects').replace('https://www.danielshort.me/portfolio?audience=personal', 'https://www.danielshort.me/portfolio');
  const oldProfessional = wrapPersonalAccordionHtml(sampleDocument('analytics', 'projects'), {
    audience: 'analytics', category: 'projects', itemId: 'example', navigation: 'hard'
  });
  const upgraded = wrapPersonalAccordionHtml(oldProfessional, {
    audience: 'analytics', category: 'projects', itemId: 'example', navigation: 'soft'
  });
  assert(manifest(upgraded).navigation === 'soft' && railsHaveNoHardNavigation(upgraded),
    'regenerating an existing professional document must remove its obsolete forced navigation boundary');
  const workbench = wrapPersonalAccordionHtml(sampleDocument('analytics', 'projects')
    .replace('data-page="project"', 'data-page="portfolio"')
    .replace('<main id="main">', '<main id="main" data-portfolio-workbench>'), {
    audience: 'analytics', category: 'projects', itemId: 'portfolio'
  });
  assert(manifest(workbench).module === 'portfolio:workbench', 'professional workbenches must resolve their reusable route controller');
  const search = wrapPersonalAccordionHtml(sampleDocument('tourism', 'about').replace('data-page="tourism"', 'data-page="search"'), {
    audience: 'tourism', category: 'about', itemId: 'search'
  });
  assert(manifest(search).module === 'search:search', 'audience search copies must resolve the shared search controller');
  for (const tool of ['background-remover', 'transcribe', 'job-application-tracker']) {
    const toolSample = personalSample.replaceAll('https://www.danielshort.me/portfolio', 'https://www.danielshort.me/tools/' + tool)
      .replace('data-page="project"', 'data-page="' + tool + '"');
    const toolOutput = wrapPersonalAccordionHtml(toolSample, { category: 'tools', itemId: tool });
    assert(manifest(toolOutput).navigation === 'hard' && manifest(toolOutput).module === 'tools:' + tool,
      `${tool} must retain its document security and runtime boundary`);
  }
  const personal = wrapPersonalAccordionHtml(personalSample, { category: 'projects', itemId: 'test' });
  const personalPolicy = wrapPersonalAccordionHtml(personalSample.replace('name="referrer" content="no-referrer"', 'name="referrer" content="strict-origin-when-cross-origin"'), { category: 'projects', itemId: 'test' });
  assert(tags(personalPolicy, 'meta').find((tag) => tag.name === 'referrer')?.content === 'strict-origin-when-cross-origin',
    'personal routes must preserve their own declared referrer policy');
  assert(tags(personal, 'body')[0]['data-audience'] === 'personal' && railLinks(personal).length === 5 &&
    railLinks(personal).filter((rail) => !Object.hasOwn(rail, 'hidden')).length === 1,
  'the default personal shell must retain its five categories and one expanded return rail');
  assert(!railLinks(personal).some((rail) => /audience=|resume|analytics|tourism|data-science/.test(rail.href)),
    'personal rails must not introduce professional discovery links');
  if (!verifyGenerated) return;

  assert(!fs.existsSync(path.join(ROOT, 'public/contact.html')) && fs.existsSync(path.join(ROOT, 'public/pages/contact.html')),
    'Contact must be published behind rewrites without a root clean-URL file shadowing audience variants');

  const projects = fs.readdirSync(path.join(ROOT, 'pages/portfolio')).filter((name) => name.endsWith('.html'));
  const records = [];
  AUDIENCES.forEach((audience) => {
    records.push({ file: `pages/${audience}.html`, audience });
    for (const page of ['portfolio', 'contact', 'search']) records.push({ file: `pages/professional/${audience}/${page}.html`, audience, internal: true });
    for (const project of projects) records.push({ file: `pages/professional/${audience}/portfolio/${project}`, audience, internal: true });
    for (const suffix of ['', '-pdf']) records.push({ file: `pages/resume-${audience}${suffix}.html`, audience, resume: true });
  });
  for (const suffix of ['', '-pdf']) records.push({ file: `pages/resume${suffix}.html`, audience: 'analytics', resume: true });
  for (const record of records) {
    const html = read(record.file);
    validateProfessionalShell(assert, html, record.audience, record.file);
    if (record.internal) {
      assert(tags(html, 'body')[0]['data-internal-professional-copy'] === 'true', `${record.file} must retain its internal source-copy marker`);
      assert(new URL(tags(html, 'link').find((tag) => tag.rel === 'canonical').href).searchParams.get('audience') === record.audience,
        `${record.file} canonical must distinguish its audience query`);
    }
    if (record.resume) {
      assert(tags(html, 'a').some((tag) => Object.hasOwn(tag, 'download') && /\.pdf(?:$|[?#])/i.test(tag.href)),
        `${record.file} must retain a direct downloadable PDF link`);
      assert(tags(html, 'meta').find((tag) => tag.name === 'referrer')?.content === 'no-referrer', `${record.file} must retain its referrer policy`);
    }
  }
  const fallback = read('dshort.html');
  assert(tags(fallback, 'body')[0]['data-personal-category'] === 'about' && tags(fallback, 'body')[0]['data-audience'] === 'personal' &&
    count(fallback, /data-personal-accordion-shell(?:\s|>)/g) === 1,
  'the short-link failure page must use the personal About tab shell');
  assert(/noindex/.test(tags(fallback, 'meta').find((tag) => tag.name === 'robots')?.content || ''),
    'the short-link failure page must remain excluded from search');
};

if (require.main === module) {
  let count = 0;
  module.exports({ verifyGenerated: !process.argv.includes('--generator-only'), assert(condition, message) { count += 1; require('assert').ok(condition, message); } });
  process.stdout.write(`Shared theme shell tests passed (${count} assertions).\n`);
}
