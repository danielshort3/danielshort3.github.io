'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { loadSiteContent } = require('../../build/lib/content-loader');
const { createMobileContent, generateMobileContent, plainText, publicHttpsUrl } = require('../../build/generate-mobile-content');

const root = path.resolve(__dirname, '../..');
const content = loadSiteContent(root);
const catalog = createMobileContent(content);

assert.deepStrictEqual(Object.keys(catalog), ['schemaVersion', 'revision', 'site', 'about', 'projects', 'tools', 'games']);
assert.strictEqual(catalog.schemaVersion, 1);
assert.match(catalog.revision, /^[a-f0-9]{64}$/);
assert.strictEqual(catalog.site.name, 'Daniel Short');
assert.strictEqual(catalog.site.url, 'https://www.danielshort.me/');
assert.strictEqual(catalog.about.greeting, 'Hi, I’m Daniel.');
assert.strictEqual(catalog.about.interests.length, 3);
assert(catalog.about.interests.find((interest) => interest.title === 'French horn').body.includes('20 years'));
assert.strictEqual(catalog.about.experience[0].organization, 'Visit Grand Junction');
assert.strictEqual(catalog.about.experience[0].date, 'Feb 2024–present');
assert(catalog.about.education.find((entry) => entry.title === 'B.S. Data Analytics').url.startsWith('https://www.credential.net/'));
assert(catalog.about.credentials.every((entry) => entry.url.startsWith('https://')));
assert.strictEqual(catalog.projects.length, content.projects.filter((project) => project.published !== false).length);
assert(!catalog.projects.some((project) => project.id === 'minesweeper'), 'Unpublished projects stay out of the app');
assert(catalog.projects.every((project) => project.url === `${catalog.site.url}portfolio/${project.id}`), 'Only personal project routes are exported');
assert.strictEqual(catalog.tools.length, content.tools.filter((tool) => tool.visibility === 'public' && !tool.hidden && !tool.noindex).length);
assert(!catalog.tools.some((tool) => ['transcribe', 'job-application-tracker', 'short-links'].includes(tool.id)), 'Admin and account tools are not published');
assert.strictEqual(catalog.games.length, content.pagesById.games.games.length);
for (const item of [...catalog.projects, ...catalog.tools, ...catalog.games]) {
  assert(item.id && item.title && item.summary && item.url);
  assert.match(item.iconUrl, /^https:\/\/www\.danielshort\.me\/img\/.*\?v=[a-f0-9]{12}$/);
}
assert(catalog.projects.every((project) => /^https:\/\/www\.danielshort\.me\/img\/.*\?v=[a-f0-9]{12}$/.test(project.imageUrl)), 'Every project preview is versioned');
assert.strictEqual(catalog.revision, createMobileContent(content).revision, 'No wall clock timestamps change the revision');
const reordered = Object.fromEntries(Object.entries(content).reverse());
reordered.projects = [...content.projects].reverse();
reordered.tools = [...content.tools].reverse();
assert.deepStrictEqual(createMobileContent(reordered), catalog, 'Input object and catalog collection order do not change output');

assert.strictEqual(plainText('<strong>Hello</strong> &amp; <em>goodbye</em><script>private()</script>'), 'Hello & goodbye');
assert.strictEqual(plainText('&lt;script&gt;secret()&lt;/script&gt;Public&#32;text'), 'Public text');
assert.strictEqual(plainText('&amp;lt;script&amp;gt;secret()&amp;lt;/script&amp;gt;Public'), 'Public');
assert.strictEqual(plainText('A<br>B<!-- hidden -->\nC'), 'A B C');
assert.strictEqual(plainText('&#x110000; &#55296; &#0; safe'), 'safe');
for (const unsafe of [
  'javascript:alert(1)', 'data:text/html,test', 'file:///secret', 'intent://settings',
  'http://www.danielshort.me/', '//evil.example/file', 'https://user:password@example.com/',
  'https://localhost/', 'https://127.0.0.1/', 'https://10.0.0.1/', 'https://[::1]/',
  'https://service.internal/', 'https://example.com:8443/', '/api/tools/state', '/admin',
  '/professional/analytics/portfolio/website', '/analytics/portfolio/website', '/resume-analytics',
  '/%61pi/tools/state', '/tools/../../api/private', 'https://example.com/?api_key=private',
  'https://bucket.example.com/file?X-Amz-Signature=secret', 'https://example.com/?access_token=secret',
  'https://www.danielshort.me\\@evil.example/path', 'https://www.danielshort.me/\nadmin'
]) assert.strictEqual(publicHttpsUrl(unsafe), '', `Reject unsafe URL: ${unsafe}`);
assert.strictEqual(publicHttpsUrl('/tools/text-compare'), 'https://www.danielshort.me/tools/text-compare');
assert.strictEqual(publicHttpsUrl('/analytics-demo'), 'https://www.danielshort.me/analytics-demo');
assert.strictEqual(publicHttpsUrl('https://github.com/danielshort3'), 'https://github.com/danielshort3');

const modified = JSON.parse(JSON.stringify(content));
modified.site.settings.secret = 'DO_NOT_EXPORT';
modified.site.settings.siteOrigin = 'https://unapproved.example.com';
modified.site.settings.profileImage = 'https://unapproved.example.com/avatar.png';
modified.resumes = [{ text: 'DO_NOT_EXPORT' }];
modified.audiences.find((entry) => entry.key !== 'personal').page.description = 'DO_NOT_EXPORT';
modified.projects.push(...[
  { published: false }, { enabled: false }, { hidden: true }, { noindex: true }, { private: true }, { internal: true },
  { visibility: 'admin' }, { visibility: 'authed' }, { status: 'draft' }, { audience: 'professional' }
].map((flags, index) => ({ id: `private-${index}`, title: 'DO_NOT_EXPORT', ...flags })));
modified.tools.push({ slug: 'hidden', title: 'DO_NOT_EXPORT', href: '/tools/hidden', visibility: 'admin' });
modified.pages.find((entry) => entry.id === 'games').games.push({ id: 'secret-game', title: 'DO_NOT_EXPORT', href: '/games/secret-game', hidden: true });
modified.projects[0].internalNotes = 'DO_NOT_EXPORT';
modified.projects[0].resources.push({ label: 'DO_NOT_EXPORT', url: '/api/private' });
modified.projects[0].resources.push({ label: 'DO_NOT_EXPORT', url: 'https://example.com/?token=private' });
const clean = createMobileContent(modified);
assert(!JSON.stringify(clean).includes('DO_NOT_EXPORT'), 'Whitelist prevents private, unpublished, professional or arbitrary CMS fields from leaking');
assert.strictEqual(clean.site.url, catalog.site.url, 'Only approved brand origin is exported');
assert.strictEqual(clean.projects.length, catalog.projects.length);
assert.strictEqual(clean.tools.length, catalog.tools.length);
assert.strictEqual(clean.games.length, catalog.games.length);
assert.strictEqual(clean.revision, catalog.revision, 'Private content changes do not trigger a public revision');
modified.projects[0].title = 'A changed public title';
assert.notStrictEqual(createMobileContent(modified).revision, catalog.revision, 'Published content changes trigger an app refresh');

async function testImageRevisionsAndOutput() {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'native-content-test-'));
  try {
    const imageDir = path.join(tempRoot, 'img', 'projects');
    fs.mkdirSync(imageDir, { recursive: true });
    fs.writeFileSync(path.join(imageDir, 'example.png'), 'original image bytes');
    fs.writeFileSync(path.join(imageDir, 'example-640.webp'), 'responsive image bytes');
    const source = {
      site: { settings: { siteName: 'Example' } }, audiences: [], pages: [], tools: [],
      projects: [{ id: 'example', title: 'Example', image: 'img/projects/example.png', iconImage: 'https://unapproved.example.com/icon.png' }]
    };
    const first = createMobileContent(source, { root: tempRoot });
    assert(first.projects[0].imageUrl.includes('/example-640.webp?v='), 'Prefer lightweight responsive preview when available');
    assert.strictEqual(first.projects[0].iconUrl, '', 'Unapproved image hosts are rejected');
    fs.writeFileSync(path.join(imageDir, 'example-640.webp'), 'updated responsive image bytes');
    const second = createMobileContent(source, { root: tempRoot });
    assert.notStrictEqual(first.projects[0].imageUrl, second.projects[0].imageUrl, 'Changed image bytes invalidate native image caches');
    assert.notStrictEqual(first.revision, second.revision, 'Changed image bytes invalidate the catalog');
    const { outputPath } = await generateMobileContent({ root: tempRoot, content: source });
    assert.deepStrictEqual(JSON.parse(fs.readFileSync(outputPath, 'utf8')), second);
    assert(outputPath.endsWith(path.join('dist', 'app-content', 'v1', 'catalog.json')));
  } finally {
    const resolvedTemporaryRoot = fs.realpathSync(tempRoot);
    assert.strictEqual(path.dirname(resolvedTemporaryRoot), fs.realpathSync(os.tmpdir()));
    assert(path.basename(resolvedTemporaryRoot).startsWith('native-content-test-'));
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
}

testImageRevisionsAndOutput().then(() => {
  process.stdout.write(`Native mobile content passed: ${catalog.projects.length} projects, ${catalog.tools.length} public tools, ${catalog.games.length} games; schema, privacy, safe links, stable revisions and image updates.\n`);
}).catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
