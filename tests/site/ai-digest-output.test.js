'use strict';

const assert = require('assert/strict');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '../..');
const origin = 'https://www.danielshort.me';
const read = (relPath) => fs.readFileSync(path.join(root, relPath), 'utf8');
const readJson = (relPath) => JSON.parse(read(relPath));
const fileHash = (relPath) => crypto.createHash('sha256').update(read(relPath)).digest('hex');
const manifest = readJson('dist/ai-digest-manifest.json');
const sitemapRoutes = [...read('sitemap.xml').matchAll(/<loc>([^<]+)<\/loc>/g)]
  .map((match) => new URL(match[1]).pathname);
const digestByRoute = new Map(manifest.pages.map((page) => [page.url, page]));

function sorted(values) {
  return [...values].sort();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function htmlFiles(dirPath) {
  return fs.readdirSync(path.join(root, dirPath), { withFileTypes: true })
    .flatMap((entry) => {
      const child = `${dirPath}/${entry.name}`;
      return entry.isDirectory() ? htmlFiles(child) : entry.name.endsWith('.html') ? [child] : [];
    });
}

function digest(route) {
  const page = digestByRoute.get(route);
  assert(page, `No AI digest for ${route}`);
  return read(page.outputPath);
}

function contentHref(href) {
  const parsed = new URL(href, `${origin}/`);
  return parsed.origin === origin ? `${parsed.pathname}${parsed.search}${parsed.hash}` : parsed.href;
}

function assertLink(html, href, context) {
  const expected = `${origin}${contentHref(href)}`;
  assert(html.includes(`href="${escapeHtml(expected)}"`), `${context} should link to ${expected}`);
}

function sectionHtml(html, title) {
  const section = [...html.matchAll(/<section\b[^>]*>[\s\S]*?<\/section>/g)]
    .map((match) => match[0])
    .find((value) => new RegExp(`<h[2-6]\\b[^>]*>${escapeHtml(title)}<\\/h[2-6]>`, 'i').test(value));
  assert(section, `Missing ${title} section`);
  return section;
}

assert.equal(manifest.origin, origin);
assert.equal(digestByRoute.size, manifest.pages.length, 'AI digest routes must be unique');
assert.deepEqual(sorted(digestByRoute.keys()), sorted(sitemapRoutes),
  'AI digest routes should equal the indexable canonical HTML routes in the sitemap');
assert.deepEqual(sorted(htmlFiles('dist/ai-pages')),
  sorted(manifest.pages.map((page) => page.outputPath)),
  'The digest output directory should contain exactly the manifest pages');

for (const page of manifest.pages) {
  const html = digest(page.url);
  const canonical = `${origin}${page.url}`;
  const expectedAiUrl = `${origin}/ai/${page.url === '/' ? 'index' : page.url.slice(1)}`;
  assert.equal(page.canonicalUrl, canonical, `${page.url} canonical URL`);
  assert.equal(page.aiUrl, expectedAiUrl, `${page.url} AI URL`);
  assert(!/[.!?]:/.test(page.summary || ''), `${page.url} should not join a complete sentence to another thought with a colon`);
  assert(page.outputPath.startsWith('dist/ai-pages/') && !page.outputPath.includes('..'),
    `${page.url} should have a safe output path`);
  assert(html.includes('<html lang="en" data-ai-digest="true">'), `${page.url} should be marked as an AI digest`);
  assert(html.includes(`<link rel="canonical" href="${escapeHtml(canonical)}">`),
    `${page.url} should canonicalize to its public page`);
  assert(html.includes(`<a href="${escapeHtml(canonical)}">View the full page</a>`),
    `${page.url} should visibly link to its full public page`);
  assert(html.includes(`<main id="main" data-ai-digest="true" data-canonical-url="${escapeHtml(canonical)}">`),
    `${page.url} should identify its canonical public page`);
  assert(/<meta name="robots" content="[^"]*noindex[^"]*">/i.test(html), `${page.url} should be noindex`);
  assert.equal((html.match(/<h1\b/gi) || []).length, 1, `${page.url} should have one H1`);
  assert(!/<(?:script|form|input|textarea|button|nav)\b/i.test(html),
    `${page.url} should contain only readable static content`);
  assert(html.includes(`<meta name="source-path" content="${escapeHtml(page.sourcePath)}">`),
    `${page.url} should disclose its source path`);
  assert(html.includes(`<meta name="source-hash" content="${page.sourceHash}">`),
    `${page.url} should disclose the matching source hash`);
  assert(/^[a-f0-9]{16}$/.test(page.sourceHash), `${page.url} should have a source hash`);

  for (const [, href] of html.matchAll(/<a\b[^>]*\bhref="([^"]+)"/g)) {
    const parsed = new URL(href.replace(/&amp;/g, '&'), origin);
    assert(['https:', 'mailto:', 'tel:'].includes(parsed.protocol), `${page.url} has unsafe link ${href}`);
    if (parsed.origin === origin) {
      assert(!/^\/(?:api|pages|ai|dist\/ai-pages)(?:\/|$)/i.test(parsed.pathname),
        `${page.url} exposes an implementation or API path: ${href}`);
    }
  }
}

const tools = fs.readdirSync(path.join(root, 'content/tools'))
  .filter((name) => name.endsWith('.json'))
  .map((name) => ({ data: readJson(`content/tools/${name}`), relPath: `content/tools/${name}` }));
const publicTools = tools.filter(({ data }) => !data.hidden && !data.noindex &&
  (!data.visibility || data.visibility === 'public'));
const restrictedTools = tools.filter(({ data }) => !publicTools.some(({ data: item }) => item.slug === data.slug));
const toolsDirectory = digest('/tools');
const home = digest('/');

for (const { data: tool, relPath } of publicTools) {
  const route = `/tools/${tool.slug}`;
  const html = digest(route);
  assert.equal(digestByRoute.get(route).sourcePath, relPath, `${route} should use authored tool metadata`);
  const sourceHash = crypto.createHash('sha256').update(JSON.stringify(tool)).digest('hex').slice(0, 16);
  assert.equal(digestByRoute.get(route).sourceHash, sourceHash, `${route} should be fresh against tool JSON`);
  assertLink(toolsDirectory, tool.href || route, 'Tools directory');
  assertLink(home, tool.href || route, 'Homepage');
  for (const [title, values] of [
    ['Inputs', tool.inputs],
    ['Outputs', tool.outputs],
    ['Privacy and runtime', [tool.privacy]]
  ]) {
    const section = sectionHtml(html, title);
    for (const value of values) {
      assert(section.includes(escapeHtml(value)), `${route} ${title} should preserve authored text: ${value}`);
    }
  }
}

for (const { data: tool } of restrictedTools) {
  const route = `/tools/${tool.slug}`;
  assert(!digestByRoute.has(route), `${route} should not have a digest`);
  for (const [directoryName, html] of [['Tools directory', toolsDirectory], ['Homepage', home]]) {
    assert(!html.includes(`href="${origin}${route}"`), `${directoryName} should not link to ${route}`);
  }
}

const projects = fs.readdirSync(path.join(root, 'content/projects'))
  .filter((name) => name.endsWith('.json'))
  .map((name) => ({ data: readJson(`content/projects/${name}`), relPath: `content/projects/${name}` }));
const publicProjects = projects.filter(({ data }) => data.id && data.published !== false &&
  !data.hidden && !data.noindex);
const restrictedProjects = projects.filter(({ data }) => !publicProjects.some(({ data: item }) => item.id === data.id));
const portfolioDirectory = digest('/portfolio');

for (const { data: project, relPath } of publicProjects) {
  const route = `/portfolio/${project.id}`;
  const html = digest(route);
  assert.equal(digestByRoute.get(route).sourcePath, relPath, `${route} should use authored project metadata`);
  const sourceHash = crypto.createHash('sha256').update(JSON.stringify(project)).digest('hex').slice(0, 16);
  assert.equal(digestByRoute.get(route).sourceHash, sourceHash, `${route} should be fresh against project JSON`);
  assertLink(portfolioDirectory, route, 'Portfolio directory');
  assertLink(home, route, 'Homepage');
  const star = sectionHtml(html, 'STAR Summary');
  for (const [label, values] of [
    ['Situation', [project.problem]],
    ['Task', [project.task]],
    ['Action', project.actions || []],
    ['Result', project.results || []]
  ]) {
    for (const value of values.filter(Boolean)) {
      assert(star.includes(escapeHtml(`${label}: ${value}`)),
        `${route} STAR summary should preserve ${label.toLowerCase()}: ${value}`);
    }
  }
}

for (const { data: project } of restrictedProjects) {
  const route = `/portfolio/${project.id}`;
  assert(!digestByRoute.has(route), `${route} should not have a digest`);
  for (const [directoryName, html] of [['Portfolio directory', portfolioDirectory], ['Homepage', home]]) {
    assert(!html.includes(`href="${origin}${route}"`), `${directoryName} should not link to ${route}`);
  }
}

assert(!/\bMessage sent\b/i.test(digest('/contact')),
  'Contact digest should not present hidden form success as an already completed action');
assert(!/Tracking 0 of 200 spins\./i.test(digest('/games/roulette')),
  'Roulette digest should omit the unplayed session counter');
assert(!/Showing 0 of 0 pages/i.test(digest('/sitemap')),
  'Sitemap digest should not report an uninitialized client-side page count');
for (const title of ['UFO Sightings Dashboard', 'QR Code Generator', 'UTM Batch Builder']) {
  assert(digest('/sitemap').includes(`>${escapeHtml(title)}</a>`),
    `Sitemap digest should use the authored ${title} title`);
}
const privacyDigest = digest('/privacy');
assert.equal((privacyDigest.match(/With your permission, Google Analytics 4/g) || []).length, 1,
  'Privacy digest should not repeat a section paragraph as its introduction');
assert(!/(?:How Google uses this data|Cookie settings)\s+\./i.test(privacyDigest),
  'Privacy digest should not leave punctuation detached from inline links');
const solutionsDigest = digest('/solutions');
for (const route of ['/tools/utm-batch-builder', '/tools/text-compare', '/tools/word-frequency',
  '/portfolio/retailStore', '/portfolio/pizzaDashboard', '/portfolio/digitGenerator',
  '/tools', '/contact']) {
  assertLink(solutionsDigest, route, 'Solutions digest');
}
assert(!solutionsDigest.includes(`href="${origin}/tools/ga4-utm-performance"`),
  'Solutions digest should not link to a noindex account tool');
assert(!/Browse the tool directory Start a conversation/i.test(solutionsDigest),
  'Solutions digest should keep adjacent call-to-action links distinct');

const llms = read('llms.txt');
for (const page of manifest.pages) {
  assert(llms.includes(`](${page.canonicalUrl})`), `llms.txt should link to canonical ${page.url}`);
  assert(llms.includes(`](${page.aiUrl})`), `llms.txt should explicitly link to AI summary ${page.url}`);
}
assert(!/\]\((?:https:\/\/www\.danielshort\.me)?\/(?:api|pages|dist\/ai-pages)\//i.test(llms),
  'llms.txt should expose only public routes');

assert.equal(fileHash('llms.txt'), fileHash('public/llms.txt'),
  'The published llms.txt should match its source');
assert.equal(fileHash('dist/ai-digest-manifest.json'), fileHash('public/dist/ai-digest-manifest.json'),
  'The published manifest should match its source');
for (const page of manifest.pages) {
  assert.equal(fileHash(page.outputPath), fileHash(`public/${page.outputPath}`),
    `${page.url} published HTML should match its source`);
}

console.log(`AI digest output checks passed for ${manifest.pages.length} indexable routes.`);
