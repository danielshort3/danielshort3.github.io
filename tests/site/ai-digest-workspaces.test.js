'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { createRequire } = require('module');

const root = path.resolve(__dirname, '../..');
const generatorPath = path.join(root, 'build/generate-ai-digests.js');
const source = fs.readFileSync(generatorPath, 'utf8');
const context = {
  require: createRequire(generatorPath),
  __dirname: path.dirname(generatorPath),
  module: { exports: {} },
  URL,
};
vm.runInNewContext(source.replace(/\nmain\(\);\s*$/, '\nmodule.exports = { buildDigestPage, buildToolStructuredPage, buildProjectStructuredPage, buildPortfolioStructuredPage, buildToolsDirectoryStructuredPage, buildPersonalHomeStructuredPage, applyStructuredPage, extractCleanMainText, shouldExcludeUrl };'), context);
const api = context.module.exports;
assert.equal(typeof api.buildDigestPage, 'function');

const tool = JSON.parse(fs.readFileSync(path.join(root, 'content/tools/text-compare.json'), 'utf8'));
const structuredPage = api.buildToolStructuredPage({ data: tool, relPath: 'content/tools/text-compare.json' }, new Map());
const options = {
  html: '<title>Text Compare</title><main><form><textarea>PRIVATE_DRAFT</textarea><input value="PRIVATE_SETTING"></form></main>',
  relPath: 'pages/text-compare.html',
  urlPath: '/tools/text-compare',
  generatedAt: '2026-01-01T00:00:00.000Z',
};
assert.equal(api.extractCleanMainText(options.html), '', 'Interactive input content should remain excluded.');
assert.equal(api.buildDigestPage(options), null, 'A page with no prose or authored metadata should still be rejected.');

const digest = api.buildDigestPage({ ...options, structuredPage });
assert(digest, 'A form-based public tool should remain discoverable from its authored metadata.');
assert.equal(digest.url, '/tools/text-compare');
const finalDigest = api.applyStructuredPage(digest, structuredPage);
assert.equal(finalDigest.summary, tool.summary);
assert.equal(finalDigest.sourcePath, 'content/tools/text-compare.json');
assert(finalDigest.sections.some((section) => section.title === 'What It Does'));
assert(!JSON.stringify(finalDigest).includes('PRIVATE_DRAFT'));
assert(!JSON.stringify(finalDigest).includes('PRIVATE_SETTING'));

const realHtml = fs.readFileSync(path.join(root, 'pages/text-compare.html'), 'utf8');
assert(api.buildDigestPage({ ...options, html: realHtml, structuredPage }), 'The shipped Text Compare workspace should produce a digest.');
assert(api.shouldExcludeUrl('/tools/text-compare', '<meta name="robots" content="noindex">', new Set()));
assert(api.shouldExcludeUrl('/tools/text-compare', '', new Set(['/tools/text-compare'])));
assert(api.shouldExcludeUrl('/tools/text-compare', '', new Set(), { exclude: true }));
assert.equal(api.buildToolStructuredPage({ data: { ...tool, hidden: true }, relPath: 'content/tools/text-compare.json' }, new Map()), null);
assert.equal(api.buildToolStructuredPage({ data: { ...tool, noindex: true }, relPath: 'content/tools/text-compare.json' }, new Map()), null);
assert.equal(api.buildToolStructuredPage({ data: { ...tool, visibility: 'admin' }, relPath: 'content/tools/text-compare.json' }, new Map()), null);

const hiddenStatusHtml = '<main><p>Visible guide with useful instructions for the public visitor.</p>' +
  '<section hidden><h2>Message sent</h2><p>This should only appear after submitting the form.</p></section>' +
  '<section inert><p>INERT_PRIVATE_STATE</p></section>' +
  '<section aria-hidden="true"><p>ARIA_HIDDEN_PRIVATE_STATE</p></section>' +
  '<div role="status" aria-live="polite">Tracking 0 of 200 spins.</div></main>';
const cleanHtml = api.extractCleanMainText(hiddenStatusHtml);
assert(cleanHtml.includes('Visible guide'), 'The public prose should survive HTML extraction.');
for (const hiddenText of ['Message sent', 'INERT_PRIVATE_STATE', 'ARIA_HIDDEN_PRIVATE_STATE', 'Tracking 0 of 200 spins.']) {
  assert(!cleanHtml.includes(hiddenText), `Hidden or live-only text should not be presented as a page fact: ${hiddenText}`);
}

const project = JSON.parse(fs.readFileSync(path.join(root, 'content/projects/retailStore.json'), 'utf8'));
const projectRecord = { data: project, relPath: 'content/projects/retailStore.json' };
const structuredProject = api.buildProjectStructuredPage(projectRecord);
assert(structuredProject, 'A published project should have a structured digest.');
const star = structuredProject.sections.find((section) => section.title === 'STAR Summary');
assert(star, 'The authored project should have a STAR section.');
for (const [label, values] of [
  ['Situation', [project.problem]],
  ['Task', [project.task]],
  ['Action', project.actions],
  ['Result', project.results]
]) {
  for (const value of values.filter(Boolean)) {
    assert(star.items.includes(`${label}: ${value}`), `The STAR section should retain the authored ${label}: ${value}`);
  }
}
for (const state of [{ published: false }, { hidden: true }, { noindex: true }]) {
  assert.equal(api.buildProjectStructuredPage({ ...projectRecord, data: { ...project, ...state } }), null,
    `A ${Object.keys(state)[0]} project should not have a structured digest.`);
}
const longProject = { ...project, id: 'longFixture', actions: Array.from({ length: 18 }, (_, i) => `Action ${i + 1}.`),
  results: Array.from({ length: 18 }, (_, i) => `Result ${i + 1}.`) };
const longStar = api.buildProjectStructuredPage({ ...projectRecord, data: longProject }).sections
  .find((section) => section.title === 'STAR Summary');
assert(longStar.items.includes('Action: Action 18.') && longStar.items.includes('Result: Result 18.'),
  'Long project histories should keep all authored actions and results.');

const portfolio = api.buildPortfolioStructuredPage([
  projectRecord,
  { ...projectRecord, data: { ...project, id: 'draftFixture', published: false, title: 'Draft Project' } },
  { ...projectRecord, data: { ...project, id: 'hiddenFixture', hidden: true, title: 'Hidden Project' } },
  { ...projectRecord, data: { ...project, id: 'noindexFixture', noindex: true, title: 'Noindex Project' } }
]);
const portfolioLinks = portfolio.sections.flatMap((section) => section.links || []);
assert(portfolioLinks.some((link) => link.url.endsWith('/portfolio/retailStore')));
assert(!portfolioLinks.some((link) => /(?:draftFixture|hiddenFixture|noindexFixture)/.test(link.url)),
  'Portfolio listings should exclude unpublished, hidden, and noindex projects.');

const toolsDirectory = api.buildToolsDirectoryStructuredPage({
  data: { canonicalPath: '/tools', title: 'Tools', description: 'Public tools.',
    categories: [{ id: tool.categoryId, title: 'Writing', description: 'Writing tools.' }] },
  relPath: 'content/pages/tools.json'
}, [
  { data: tool },
  { data: { ...tool, slug: 'adminFixture', href: 'tools/adminFixture', visibility: 'admin', title: 'Admin Tool' } },
  { data: { ...tool, slug: 'hiddenFixture', href: 'tools/hiddenFixture', hidden: true, title: 'Hidden Tool' } },
  { data: { ...tool, slug: 'noindexFixture', href: 'tools/noindexFixture', noindex: true, title: 'Noindex Tool' } }
]);
const directoryLinks = toolsDirectory.sections.flatMap((section) => section.links || []);
assert(directoryLinks.some((link) => link.url.endsWith('/tools/text-compare')));
assert(!directoryLinks.some((link) => /(?:adminFixture|hiddenFixture|noindexFixture)/.test(link.url)),
  'Tools listings should exclude admin, hidden, and noindex tools.');

const personal = JSON.parse(fs.readFileSync(path.join(root, 'content/audiences/personal.json'), 'utf8'));
const about = personal.page.sections.find(section => section.type === 'home-accordion').props.categories.find(category => category.id === 'about');
const homePage = api.buildPersonalHomeStructuredPage({ data: personal, relPath: 'content/audiences/personal.json' }, [], [], { data: { games: [] } });
assert(homePage.introParagraphs.includes(about.lead) && homePage.introParagraphs.includes(about.context),
  'The personal introduction remains part of the actual rendered digest content.');
const stories = homePage.sections.find(section => section.title === about.aboutStory.title);
assert(stories, 'The home digest includes the visible personal-story section.');
const storyText = JSON.stringify(stories);
for (const connection of about.aboutStory.connections) {
  for (const text of [connection.title, connection.description, connection.project.title, connection.project.summary].filter(Boolean)) {
    assert(storyText.includes(text), `The ${connection.id} connection keeps its substantive interest and project explanation.`);
  }
  assert(stories.links.some(link => new URL(link.url, 'https://www.danielshort.me').pathname === connection.project.href),
    `The ${connection.id} connection links to its actual project or directory.`);
}

console.log('AI digest workspace metadata checks passed.');
