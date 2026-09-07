'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { versionedImageUrl, versionImageContent } = require('../../build/lib/versioned-image-url');
const { loadSiteContent } = require('../../build/lib/content-loader');
const { buildHomeLibraryData } = require('../../build/generate-cms-artifacts');
const { renderToolsDirectoryBody, renderGamesDirectoryBody, renderProjectsDataJs } = require('../../build/lib/cms-renderers');
const { renderVisualPageBody } = require('../../api/_lib/cms-widgets');
const { renderProjectPage, renderPortfolioStaticResults } = require('../../build/generate-project-pages');
const { renderPersonalLibraryMain } = require('../../build/lib/personal-accordion-shell');

const root = path.resolve(__dirname, '../..');
const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'site-image-version-'));
try {
  fs.mkdirSync(path.join(temporaryRoot, 'img/tools/icons'), { recursive: true });
  fs.mkdirSync(path.join(temporaryRoot, 'img/projects'), { recursive: true });
  fs.mkdirSync(path.join(temporaryRoot, 'img/projects/icons'), { recursive: true });
  fs.mkdirSync(path.join(temporaryRoot, 'img/games/icons'), { recursive: true });
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
  const projectIconPath = path.join(temporaryRoot, 'img/projects/icons/smartSentence.png');
  fs.writeFileSync(projectIconPath, 'first project icon');
  const projectIcon = versionedImageUrl('/img/projects/icons/smartSentence.png', options);
  assert(/\?v=[a-f0-9]{12}$/.test(projectIcon), 'library-only project icons use content fingerprints');
  fs.writeFileSync(projectIconPath, 'replacement project icon');
  assert.notStrictEqual(versionedImageUrl(projectIcon, options), projectIcon, 'replacing a project icon invalidates its cached URL');
  const gameIconPath = path.join(temporaryRoot, 'img/games/icons/stellar-dogfight.png');
  fs.writeFileSync(gameIconPath, 'first game icon');
  const gameIcon = versionedImageUrl('/img/games/icons/stellar-dogfight.png', options);
  assert(/\?v=[a-f0-9]{12}$/.test(gameIcon), 'game library icons use content fingerprints');
  fs.writeFileSync(gameIconPath, 'replacement game icon');
  assert.notStrictEqual(versionedImageUrl(gameIcon, options), gameIcon, 'replacing a game icon invalidates its cached URL');
  const poster = versionedImageUrl('/img/projects/sheetMusicUpscale.png', options);
  for (const variant of ['.webp', '.avif', '-640.webp', '-640.avif', '-960.webp', '-960.avif']) {
    const url = versionedImageUrl(`/img/projects/sheetMusicUpscale${variant}`, options);
    assert.strictEqual(url.split('?')[1], poster.split('?')[1], 'responsive variants share the PNG generation even before optimization');
  }
  fs.writeFileSync(posterPath, 'poster second version');
  assert.notStrictEqual(versionedImageUrl('/img/projects/sheetMusicUpscale-640.webp', options).split('?')[1], poster.split('?')[1]);
  fs.writeFileSync(path.join(temporaryRoot, 'img/projects/website.png'), 'current website');
  const website = versionedImageUrl('/img/projects/website.png', options);
  assert.strictEqual(versionedImageUrl('/img/projects/website-640.webp', options).split('?')[1], website.split('?')[1]);
  const previewPath = path.join(temporaryRoot, 'img/projects/shapeClassifier-preview.webp');
  fs.writeFileSync(previewPath, 'current demo');
  const preview = versionedImageUrl('/img/projects/shapeClassifier-preview.webp', options);
  fs.writeFileSync(previewPath, 'updated demo');
  assert.notStrictEqual(versionedImageUrl(preview, options), preview, 'refreshed demo previews invalidate cached images');
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
const iconProjects = canonical.projects.filter(item => item.published !== false && item.iconImage);
assert.strictEqual(iconProjects.length, 16, 'all sixteen published projects receive library icons');
const homeProjects = content.audiencesByKey.personal.page.sections
  .find(section => section.type === 'home-accordion').props.categories
  .find(category => category.id === 'projects').items;
const homeHtml = renderVisualPageBody(content.audiencesByKey.personal.page);
for (const featured of homeProjects) {
  const project = content.projects.find(item => item.id === featured.contentId);
  const libraryItem = library.projects.items.find(item => item.id === featured.contentId);
  assert.strictEqual(featured.iconImage, project.iconImage, 'homepage featured cards use the canonical project icon');
  assert.strictEqual(`/${featured.iconImage}`, libraryItem.iconImage, 'homepage and full library share the same versioned artwork');
  assert(homeHtml.includes(`home-accordion__card-media--icon" aria-hidden="true"><img src="${featured.iconImage}" alt="" loading="lazy" decoding="async" width="256" height="256">`), 'homepage renders a decorative, intrinsically sized icon tile');
  assert(!homeHtml.includes(`img/projects/${project.id}-640.webp`), 'homepage featured cards do not fall back to old screenshots');
}
for (const original of iconProjects) {
  const item = library.projects.items.find(entry => entry.id === original.id);
  assert.strictEqual(original.image, `img/projects/${original.id}.png`, 'canonical project screenshot remains independent from its library icon');
  assert.strictEqual(original.iconImage, `img/projects/icons/${original.id}.png`);
  assert.strictEqual(item.iconImage, versionedImageUrl(`/${original.iconImage}`));
  const rendered = renderPersonalLibraryMain({ category: 'projects', items: [item] });
  assert(rendered.includes(`src="${item.iconImage}"`) && rendered.includes('home-library__card--icon'), 'the standalone library renders the icon tile');
  assert(!rendered.includes(`src="${item.image}"`), 'the library does not load the larger screenshot behind its icon');
  const staticResults = renderPortfolioStaticResults([original]);
  assert(staticResults.includes(`src="${versionedImageUrl(original.iconImage)}"`) && !staticResults.includes(`src="${versionedImageUrl(original.image)}"`), 'professional initial HTML uses the same library icon before JavaScript starts');
}
for (const id of ['smartSentence', 'sheetMusicUpscale', 'deliveryTip']) {
  const original = canonical.projects.find(item => item.id === id);
  const projectPage = renderProjectPage(original);
  assert(!projectPage.includes('img/projects/icons/'), `${id} detail and sharing imagery remain separate from its library icon`);
}
const publishedGames = canonical.pagesById.games.games.filter(game => !game.hidden && !game.noindex);
assert.strictEqual(library.games.items.length, publishedGames.length, 'game icons retain the published catalog');
for (const game of publishedGames) {
  const item = library.games.items.find(entry => entry.id === game.id);
  assert.strictEqual(game.iconImage, `img/games/icons/${game.id}.png`);
  assert.strictEqual(item.iconImage, versionedImageUrl(`/${game.iconImage}`));
  assert.strictEqual(item.image, `/img/home-previews/games/${game.id}.webp`, 'game preview artwork remains available independently of library icons');
  const rendered = renderPersonalLibraryMain({ category: 'games', items: [item] });
  assert(rendered.includes(`src="${item.iconImage}"`) && !rendered.includes(`src="${item.image}"`), 'standalone game library loads the icon instead of the preview');
}
const homeGames = content.audiencesByKey.personal.page.sections
  .find(section => section.type === 'home-accordion').props.categories
  .find(category => category.id === 'games').items;
for (const featured of homeGames) {
  const item = library.games.items.find(entry => entry.id === featured.contentId);
  assert.strictEqual(`/${featured.iconImage}`, item.iconImage, 'featured and full game libraries share the same artwork');
  assert(homeHtml.includes(`src="${featured.iconImage}" alt=""`), 'featured game icons are decorative images');
}
const gamesHtml = renderGamesDirectoryBody(content.pagesById.games);
for (const game of content.pagesById.games.games) {
  assert(gamesHtml.includes(`src="${game.iconImage}"`), 'the direct CMS game directory renders the library icon');
  if (game.image) assert(!gamesHtml.includes(`src="${game.image}"`), 'game directory icons do not load in-game artwork');
}
const fallbackGames = { ...content.pagesById.games, games: content.pagesById.games.games.map(game => ({ ...game, iconImage: '' })) };
assert(renderGamesDirectoryBody(fallbackGames).includes(`src="${fallbackGames.games[0].image}"`), 'games without optional icons retain their original preview fallback');
const fallbackItem = { ...library.projects.items[0], iconImage: '' };
const fallbackRendered = renderPersonalLibraryMain({ category: 'projects', items: [fallbackItem] });
assert(fallbackRendered.includes(`src="${fallbackItem.image}"`) && !fallbackRendered.includes('home-library__card--icon'), 'a project without optional icon artwork retains its original preview treatment');
const workbenchSource = fs.readFileSync(path.join(root, 'js/portfolio/portfolio.js'), 'utf8');
const workbenchMediaStart = workbenchSource.indexOf('  const renderWorkbenchMedia =');
const workbenchMediaSource = workbenchSource.slice(workbenchMediaStart, workbenchSource.indexOf('  const renderResults =', workbenchMediaStart));
const workbenchContext = { escapeHtml: value => String(value), buildResponsivePicture: source => `<picture data-source="${source}"></picture>` };
vm.runInNewContext(`${workbenchMediaSource}\nthis.renderMedia = renderWorkbenchMedia;`, workbenchContext);
const projectWithIcon = content.projects.find(item => item.id === 'smartSentence');
const iconMarkup = workbenchContext.renderMedia(projectWithIcon);
assert(iconMarkup.includes(projectWithIcon.iconImage) && !iconMarkup.includes(projectWithIcon.image), 'professional workbench results prefer icons even when a canonical screenshot exists');
const deliveryProject = content.projects.find(item => item.id === 'deliveryTip');
assert(workbenchContext.renderMedia(deliveryProject).includes(deliveryProject.iconImage), 'professional workbench uses the new Delivery Tip library icon');
assert(workbenchContext.renderMedia({ ...deliveryProject, iconImage: '' }).includes(deliveryProject.image), 'professional workbench retains canonical preview fallback when no icon exists');
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
