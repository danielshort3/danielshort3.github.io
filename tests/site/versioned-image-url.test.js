'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { versionedImageUrl, versionImageContent } = require('../../build/lib/versioned-image-url');
const { loadSiteContent } = require('../../build/lib/content-loader');
const { buildHomeLibraryData } = require('../../build/generate-cms-artifacts');
const { renderToolsDirectoryBody, renderProjectsDataJs } = require('../../build/lib/cms-renderers');
const { renderVisualPageBody } = require('../../api/_lib/cms-widgets');
const { renderProjectPage } = require('../../build/generate-project-pages');

const root = path.resolve(__dirname, '../..');
const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'site-image-version-'));
try {
  fs.mkdirSync(path.join(temporaryRoot, 'img/tools/icons'), { recursive: true });
  fs.mkdirSync(path.join(temporaryRoot, 'img/projects'), { recursive: true });
  const iconPath = path.join(temporaryRoot, 'img/tools/icons/text-compare.png');
  const posterPath = path.join(temporaryRoot, 'img/projects/sheetMusicUpscale.png');
  fs.writeFileSync(iconPath, 'icon first version');
  fs.writeFileSync(posterPath, 'poster first version');
  const options = { root: temporaryRoot };
  const first = versionedImageUrl('/img/tools/icons/text-compare.png?existing=yes#preview', options);
  assert(/^\/img\/tools\/icons\/text-compare\.png\?existing=yes&v=[a-f0-9]{12}#preview$/.test(first));
  assert.strictEqual(versionedImageUrl(first, options), first, 'versioning is idempotent');
  fs.writeFileSync(iconPath, 'icon second version');
  assert.notStrictEqual(versionedImageUrl(first, options), first, 'changed pixels invalidate long-lived cached URLs');
  const poster = versionedImageUrl('/img/projects/sheetMusicUpscale.png', options);
  for (const variant of ['.webp', '.avif', '-640.webp', '-640.avif', '-960.webp', '-960.avif']) {
    const url = versionedImageUrl(`/img/projects/sheetMusicUpscale${variant}`, options);
    assert.strictEqual(url.split('?')[1], poster.split('?')[1], 'responsive variants share the PNG generation even before optimization');
  }
  fs.writeFileSync(posterPath, 'poster second version');
  assert.notStrictEqual(versionedImageUrl('/img/projects/sheetMusicUpscale-640.webp', options).split('?')[1], poster.split('?')[1]);
  for (const untouched of ['img/projects/babynames.png', '/img/games/game.png', 'https://external.test/img/tools/icons/text-compare.png', '/api/tools/auth/session', '/img/projects/sheetMusicUpscale.mp4']) {
    assert.strictEqual(versionedImageUrl(untouched, options), untouched, 'unrelated resources retain their existing URLs');
  }
} finally {
  const resolvedTemp = path.resolve(temporaryRoot);
  const allowedTemp = path.resolve(os.tmpdir());
  assert(path.dirname(resolvedTemp) === allowedTemp && path.basename(resolvedTemp).startsWith('site-image-version-'), 'cleanup stays inside the dedicated temporary test directory');
  fs.rmSync(resolvedTemp, { recursive: true, force: true });
}

const canonical = loadSiteContent(root);
const before = JSON.stringify(canonical);
const content = versionImageContent(canonical);
assert.strictEqual(JSON.stringify(canonical), before, 'canonical content and disk paths remain unchanged');
const library = buildHomeLibraryData(content);
assert(library.tools.items.every(item => /\/img\/tools\/icons\/[^?]+\.png\?v=[a-f0-9]{12}$/.test(item.image)), 'every tool library image is versioned');
assert(/\?v=[a-f0-9]{12}$/.test(library.projects.items.find(item => item.id === 'sheetMusicUpscale').image));
const targets = /(?:\/)?img\/(?:tools\/icons\/[a-z0-9_-]+\.(?:png|webp|avif)|projects\/sheetMusicUpscale(?:-[a-z0-9-]+)?\.(?:png|webp|avif))(?:\?[^\s"'<>]+)?/gi;
function checkRendered(html, label) {
  const references = html.match(targets) || [];
  assert(references.length > 0, `${label} contains target images`);
  assert(references.every(reference => /\?v=[a-f0-9]{12}$/.test(reference)), `${label} versions every target reference`);
}
checkRendered(renderVisualPageBody(content.audiencesByKey.personal.page), 'homepage');
checkRendered(renderToolsDirectoryBody(content.pagesById.tools, content.tools), 'direct tools library');
checkRendered(renderProjectsDataJs(content.projects), 'generated project data');
const project = canonical.projects.find(item => item.id === 'sheetMusicUpscale');
checkRendered(renderProjectPage(project), 'direct sheet comparison');
const posterHtml = renderProjectPage({ ...project, previewComparison: null });
checkRendered(posterHtml, 'responsive project poster');
for (const extension of ['webp', 'avif']) {
  assert(posterHtml.includes(versionedImageUrl(`img/projects/sheetMusicUpscale-640.${extension}`)), 'responsive sources retain their optimized formats');
}
for (const [file, startMarker, endMarker] of [
  ['js/portfolio/portfolio.js', 'const buildResponsiveSrcset =', 'const projectMedia ='],
  ['js/portfolio/modal-helpers.js', '  function buildResponsiveSrcset(', '  function computeTableauSrc(']
]) {
  const source = fs.readFileSync(path.join(root, file), 'utf8');
  const snippet = source.slice(source.indexOf(startMarker), source.indexOf(endMarker));
  const context = {};
  vm.runInNewContext(`${snippet}\nthis.picture = buildResponsivePicture;`, context);
  const rendered = context.picture(versionedImageUrl('img/projects/sheetMusicUpscale.png'), 'Example', { width: 1600, height: 900 });
  for (const extension of ['webp', 'avif']) {
    assert(rendered.includes(versionedImageUrl(`img/projects/sheetMusicUpscale-640.${extension}`)), `${file} preserves optimized versioned sources`);
  }
  assert(context.picture('img/projects/babynames.png', 'Other', { width: 1600 }).includes('img/projects/babynames-640.webp 640w'), `${file} preserves unrelated source URLs`);
}
process.stdout.write('Image cache versions: content freshness, responsive generation, and rendered references passed.\n');
