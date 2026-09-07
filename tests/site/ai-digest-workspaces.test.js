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
vm.runInNewContext(source.replace(/\nmain\(\);\s*$/, '\nmodule.exports = { buildDigestPage, buildToolStructuredPage, buildPersonalHomeStructuredPage, applyStructuredPage, extractCleanMainText, shouldExcludeUrl };'), context);
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
