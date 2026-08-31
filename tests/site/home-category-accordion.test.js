const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {
  getHomeAccordionIconDefinitions,
  getWidgetDefinitions,
  renderVisualPageBody,
  resolveHomeAccordionIconId
} = require('../../api/_lib/cms-widgets');
const {
  HOME_LIBRARY_VISUALS
} = require('../../build/validate-home-library-visuals');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));
const count = (value, pattern) => (String(value || '').match(pattern) || []).length;
const extractBlock = (source, marker) => {
  const markerIndex = source.indexOf(marker);
  if (markerIndex < 0) return '';
  const openIndex = source.indexOf('{', markerIndex);
  if (openIndex < 0) return '';
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(markerIndex, index + 1);
  }
  return '';
};
const extractFunctionBlock = (source, marker) => {
  const markerIndex = source.indexOf(marker);
  if (markerIndex < 0) return '';
  const signatureEnd = source.indexOf(')', markerIndex + marker.length);
  if (signatureEnd < 0) return '';
  const openIndex = source.indexOf('{', signatureEnd + 1);
  if (openIndex < 0) return '';
  let depth = 0;
  for (let index = openIndex; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(markerIndex, index + 1);
  }
  return '';
};
const readWebpDimensions = (relativePath) => {
  const diskPath = path.join(ROOT, String(relativePath || '').replace(/^[/\\]+/, ''));
  const buffer = fs.readFileSync(diskPath);
  if (buffer.length < 20 ||
    buffer.toString('ascii', 0, 4) !== 'RIFF' ||
    buffer.toString('ascii', 8, 12) !== 'WEBP') {
    throw new Error(`${relativePath} is not a valid WebP file`);
  }

  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const chunkType = buffer.toString('ascii', offset, offset + 4);
    const chunkLength = buffer.readUInt32LE(offset + 4);
    const dataOffset = offset + 8;
    if (dataOffset + chunkLength > buffer.length) break;

    if (chunkType === 'VP8X' && chunkLength >= 10) {
      return {
        width: buffer.readUIntLE(dataOffset + 4, 3) + 1,
        height: buffer.readUIntLE(dataOffset + 7, 3) + 1
      };
    }
    if (chunkType === 'VP8 ' && chunkLength >= 10 &&
      buffer[dataOffset + 3] === 0x9d &&
      buffer[dataOffset + 4] === 0x01 &&
      buffer[dataOffset + 5] === 0x2a) {
      return {
        width: buffer.readUInt16LE(dataOffset + 6) & 0x3fff,
        height: buffer.readUInt16LE(dataOffset + 8) & 0x3fff
      };
    }
    if (chunkType === 'VP8L' && chunkLength >= 5 && buffer[dataOffset] === 0x2f) {
      const sizeBits = buffer.readUInt32LE(dataOffset + 1);
      return {
        width: (sizeBits & 0x3fff) + 1,
        height: ((sizeBits >>> 14) & 0x3fff) + 1
      };
    }

    offset = dataOffset + chunkLength + (chunkLength % 2);
  }
  throw new Error(`${relativePath} does not contain supported WebP dimensions`);
};

module.exports = function runHomeCategoryAccordionTests({ assert }) {
  const personal = readJson('content/audiences/personal.json');
  const section = personal.page.sections.find((entry) => entry.type === 'home-accordion');
  const categories = section?.props?.categories || [];
  const ids = categories.map((category) => category.id);
  const html = renderVisualPageBody(personal.page);
  const css = read('css/components/home-category-accordion.css');
  const libraryCss = read('css/components/home-library.css');
  const timelineCss = read('css/components/home-timeline.css');
  const js = read('js/home/category-accordion.js');
  const homeEntry = read('build/entries/site-home.entry.js');
  const homeStyles = read('css/styles-home.css');
  const sharedStyles = read('css/styles.css');
  const generator = read('build/generate-cms-artifacts.js');
  const visualValidator = read('build/validate-home-library-visuals.js');
  const buildSite = read('build/build-site.js');
  const copyToPublic = read('build/copy-to-public.js');
  const packageJson = readJson('package.json');
  const navigation = read('js/navigation/navigation.js');
  const activityEvents = read('js/analytics/activity-events.js');
  const homeLibraryData = require('../../js/home/home-library-data.js');
  const publishedProjects = fs.readdirSync(path.join(ROOT, 'content', 'projects'))
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => readJson(path.join('content', 'projects', fileName)))
    .filter((project) => project && project.id && project.published !== false);
  const toolsDirectoryContext = { window: {} };
  vm.runInNewContext(read('js/portfolio/tools-directory-data.js'), toolsDirectoryContext);
  const toolsDirectoryItems = toolsDirectoryContext.window.DIRECTORY_WORKBENCH?.items || [];
  const homeAccordionWidget = getWidgetDefinitions()
    .find((widget) => widget.type === 'home-accordion');
  const iconDefinitions = getHomeAccordionIconDefinitions();

  assert(section && section.variant === 'shallow-wedge',
    'personal homepage should use the selected shallow-wedge accordion variant');
  assert(JSON.stringify(ids) === JSON.stringify(['about', 'projects', 'tools', 'games', 'contact']),
    'homepage categories should stay in the approved About, Projects, Tools, Games, Contact order');
  assert(section.props.defaultPanel === 'about',
    'homepage should open the About panel by default');
  assert(homeAccordionWidget?.defaultProps?.defaultPanel === 'about',
    'new CMS homepage accordion widgets should also default to About');
  assert(
    JSON.stringify(categories.map((category) => category.color)) === JSON.stringify([
      '#091f3b', '#155dfc', '#087f8c', '#c94b0a', '#334155'
    ]),
    'homepage rails should use the approved site-native category colors'
  );
  const categoryIconIds = ['about', 'projects', 'tools', 'games', 'contact'];
  const uniqueCardIconIds = [
    'stormbreak', 'stellar-dogfight', 'probability', 'message', 'email', 'github'
  ];
  const specificIconIds = [...categoryIconIds, ...uniqueCardIconIds];
  assert(specificIconIds.every((id) => iconDefinitions[id]) &&
    new Set(specificIconIds.map((id) => iconDefinitions[id])).size === specificIconIds.length,
  'category and glyph-backed card icons should resolve to distinct on-brand SVG definitions');
  const authoredIconIds = categories.flatMap((category) => (category.items || [])
    .map((item) => item.icon)
    .filter(Boolean));
  assert(authoredIconIds.every((id) => iconDefinitions[id]) &&
    resolveHomeAccordionIconId('unknown-home-icon') === 'spark',
  'every authored homepage icon should resolve explicitly and unknown keys should use a neutral fallback');
  const expectedItemIcons = {
    stormbreak: 'stormbreak',
    'stellar-dogfight': 'stellar-dogfight',
    'probability-engine': 'probability',
    'contact-form': 'message',
    email: 'email',
    github: 'github'
  };
  const actualItemIcons = Object.fromEntries(categories
    .flatMap((category) => category.items || [])
    .filter((item) => expectedItemIcons[item.id])
    .map((item) => [item.id, item.icon]));
  assert(JSON.stringify(actualItemIcons) === JSON.stringify(expectedItemIcons),
    'icon-backed homepage cards should use specific semantic icon assignments');
  const expectedItemIds = {
    about: [],
    projects: ['babynames', 'handwritingRating', 'sheetMusicUpscale', 'ufoDashboard'],
    tools: ['text-compare', 'image-optimizer', 'qr-code-generator', 'screen-recorder'],
    games: ['project-starfall', 'stormbreak', 'stellar-dogfight', 'probability-engine'],
    contact: ['contact-form', 'email', 'github']
  };
  categories.forEach((category) => {
    assert(JSON.stringify((category.items || []).map((item) => item.id)) === JSON.stringify(expectedItemIds[category.id]),
      `${category.id} should remain a concise, curated homepage preview`);
  });
  const about = categories.find((category) => category.id === 'about');
  const games = categories.find((category) => category.id === 'games');
  const contact = categories.find((category) => category.id === 'contact');
  assert(!about.meta && !about.cta && /Based in Grand Junction/.test(about.context || '') &&
    about.profile?.image === 'img/hero/head-avatar-384.jpg' &&
    about.profile?.imageAlt === 'Daniel Short' &&
    about.profile?.imageWidth === 384 &&
    about.profile?.imageHeight === 384,
  'About should use personal prose and the deliberately sized personal portrait instead of cards or a directory CTA');
  const timelineItems = about.timeline?.items || [];
  const expectedTimelineIds = [
    'target',
    'google-data-analytics',
    'ibm-data-analyst',
    'ibm-machine-learning',
    'purdue-bs-data-analytics',
    'google-advanced-data-analytics',
    'randall-reilly',
    'visit-grand-junction',
    'google-analytics',
    'eastern-ms-data-science',
    'project-starfall'
  ];
  assert(about.timeline?.title === 'My path so far' &&
    JSON.stringify(timelineItems.map((item) => item.id)) === JSON.stringify(expectedTimelineIds),
  'About should carry the approved 11-event personal timeline in chronological narrative order');
  const expectedCertificateDates = {
    'google-data-analytics': '2023-01-03',
    'ibm-data-analyst': '2023-01-11',
    'ibm-machine-learning': '2023-02-12',
    'google-advanced-data-analytics': '2023-05-20',
    'google-analytics': '2024-04-18'
  };
  const certificateDates = Object.fromEntries(timelineItems
    .filter((item) => item.type === 'certification')
    .map((item) => [item.id, item.date]));
  assert(JSON.stringify(certificateDates) === JSON.stringify(expectedCertificateDates),
    'About certifications should retain their exact issue dates');
  assert(timelineItems.every((item) => !Object.keys(item)
    .some((key) => /expir|validUntil/i.test(key))),
  'homepage timeline events should not invent certificate expiration or validity dates');
  assert(games.items[0].presentation === 'featured' && games.items[0].image &&
    games.items.slice(1).every((item) => item.icon && !item.image),
  'Games should feature Starfall artwork and use consistent glyph tiles for the remaining previews');
  assert(!contact.meta && !contact.cta && contact.items.at(-1).presentation === 'tertiary',
    'Contact should avoid repeated guidance and keep GitHub as a quieter tertiary option');

  assert(count(html, /data-home-accordion-item=/g) === 5 &&
    count(html, /data-home-accordion-trigger=/g) === 5 &&
    count(html, /data-home-accordion-panel=/g) === 5,
  'generated homepage should render one static item, trigger, and attached panel per category');
  assert(count(html, /aria-expanded="true"/g) === 1 &&
    count(html, /aria-expanded="false"/g) >= 4 &&
    count(html, /aria-disabled="true"/g) === 1 &&
    count(html, /data-home-accordion-panel="[^"]+" hidden inert/g) === 4 &&
    !html.includes('aria-current="page"'),
  'homepage should author only About as expanded and noncollapsible, without treating same-page accordion buttons as page links');
  const getItemHtml = (id) => {
    const marker = html.indexOf(`data-home-accordion-item="${id}"`);
    const itemMarker = '<article class="home-accordion__item';
    const start = marker >= 0 ? html.lastIndexOf(itemMarker, marker) : -1;
    const next = start >= 0 ? html.indexOf(itemMarker, marker + 1) : -1;
    return start >= 0 ? html.slice(start, next >= 0 ? next : html.length) : '';
  };
  ids.forEach((id) => {
    const itemHtml = getItemHtml(id);
    const panelTag = itemHtml.match(new RegExp(`<section[^>]+data-home-accordion-panel="${id}"[^>]*>`))?.[0] || '';
    if (id === 'about') {
      assert(itemHtml.includes('home-accordion__item--about is-active') &&
        itemHtml.includes('aria-expanded="true"') &&
        itemHtml.includes('aria-disabled="true"') &&
        panelTag && !/\bhidden\b/.test(panelTag) && !/\binert\b/.test(panelTag),
      'About should be the specific expanded and interactive panel in authored homepage markup');
    } else {
      assert(!itemHtml.includes(`home-accordion__item--${id} is-active`) &&
        itemHtml.includes('aria-expanded="false"') &&
        /\bhidden\b/.test(panelTag) && /\binert\b/.test(panelTag),
      `${id} should be specifically authored as collapsed and non-interactive`);
    }
    assert(count(itemHtml, new RegExp(`data-home-icon="${id}"`, 'g')) === 2,
      `${id} rail and panel title should share one deliberate category icon identity`);
    assert(itemHtml.indexOf('home-accordion__rail-icon') >= 0 &&
      itemHtml.indexOf('home-accordion__rail-icon') < itemHtml.indexOf('home-accordion__rail-label'),
    `${id} rail should keep its icon before its label for the desktop column layout`);
    assert(itemHtml.includes(`aria-labelledby="home-accordion-trigger-${id}"`) &&
      new RegExp(`<h2 class="home-accordion__heading">[\\s\\S]*?<button[^>]+id="home-accordion-trigger-${id}"`).test(itemHtml),
    `${id} item should be labelled by a semantic accordion heading button`);
  });
  const aboutHtml = getItemHtml('about');
  assert(aboutHtml.includes('home-accordion__panel-head--profile') &&
    /<img src="img\/hero\/head-avatar-384\.jpg" alt="Daniel Short"[^>]+width="384" height="384">/.test(aboutHtml),
  'rendered About content should pair its heading with a meaningful, intrinsically sized profile image');
  assert(aboutHtml.includes('<section class="home-timeline" data-home-timeline aria-labelledby="home-timeline-about-title">') &&
    aboutHtml.includes('<h4 id="home-timeline-about-title">My path so far</h4>') &&
    aboutHtml.includes('<ol class="home-timeline__list">') &&
    count(aboutHtml, /<li class="home-timeline__item[^>]+data-home-timeline-item=/g) === 11 &&
    !aboutHtml.includes('role="list"') &&
    !aboutHtml.includes('role="listitem"'),
  'rendered About timeline should be a labelled section with a native ordered list of 11 semantic events');
  Object.entries(expectedCertificateDates).forEach(([id, issueDate]) => {
    const marker = `data-home-timeline-item="${id}"`;
    const markerIndex = aboutHtml.indexOf(marker);
    const itemMarker = '<li class="home-timeline__item';
    const start = markerIndex >= 0 ? aboutHtml.lastIndexOf(itemMarker, markerIndex) : -1;
    const next = start >= 0 ? aboutHtml.indexOf(itemMarker, markerIndex + marker.length) : -1;
    const itemHtml = start >= 0 ? aboutHtml.slice(start, next >= 0 ? next : aboutHtml.length) : '';
    assert(itemHtml.includes('home-timeline__item--certification') &&
      itemHtml.includes(`<time datetime="${issueDate}">`) &&
      /<a class="home-timeline__entry"[^>]+target="_blank" rel="noopener noreferrer"/.test(itemHtml),
    `${id} should render its exact issue date in a semantic time element and expose a safe external credential link`);
  });
  uniqueCardIconIds.forEach((id) => {
    assert(count(html, new RegExp(`data-home-icon="${id}"`, 'g')) === 1,
      `${id} should render exactly once as a unique homepage card glyph`);
  });
  assert(count(getItemHtml('contact'), /data-home-icon="external-arrow"/g) === 1,
    'the external GitHub card should use one dedicated external-link arrow');
  assert(count(html, /<h1\b/g) === 1 && html.includes('id="home-accordion-title"'),
    'homepage should expose one accessible H1');
  [
    ['projects', '/portfolio', 'View all projects', 'Open the dedicated project page'],
    ['tools', '/tools', 'View all tools', 'Open the dedicated tools page'],
    ['games', '/games', 'View all games', 'Open the dedicated games page']
  ].forEach(([id, href, label, pageLabel]) => {
    const itemHtml = getItemHtml(id);
    const cta = `<button class="home-accordion__panel-cta home-accordion__panel-cta--primary" type="button" aria-controls="home-library-view-${id}" aria-expanded="false" data-home-library-open="${id}">${label}`;
    const library = `<section class="home-library" id="home-library-view-${id}" data-home-library-view="${id}" aria-labelledby="home-library-view-${id}-title" hidden inert>`;
    assert(itemHtml.includes(cta) &&
      itemHtml.indexOf(cta) > itemHtml.indexOf('<ul class="home-accordion__cards">') &&
      itemHtml.includes(library) &&
      itemHtml.includes(`data-home-library-close="${id}"`) &&
      itemHtml.includes(`<h3 id="home-library-view-${id}-title" data-home-library-heading tabindex="-1">`) &&
      itemHtml.includes(`<a class="home-library__page-link" href="${href}">${pageLabel}`) &&
      itemHtml.includes(`data-home-library-list aria-label="All ${id}"`),
    `${id} CTA should open its own initially hidden inline library with back, focus target, list, and dedicated-page link`);
  });
  assert(count(html, /data-home-library-open=/g) === 3 &&
    count(html, /data-home-library-view=/g) === 3 &&
    count(html, /data-home-library-view="[^"]+"[^>]+hidden inert/g) === 3 &&
    !getItemHtml('about').includes('data-home-library-view=') &&
    !getItemHtml('contact').includes('data-home-library-view='),
  'only Projects, Tools, and Games should author inline library disclosure controls and all library views should start inert');
  assert(html.includes('<ul class="home-accordion__cards">') &&
    html.includes('<li class="home-accordion__card-item">') &&
    /<a class="home-accordion__card" href="\/tools\/text-compare"/.test(html) &&
    !/<a class="home-accordion__card" role="listitem"/.test(html),
  'homepage cards should use semantic list wrappers without masking native link roles');

  ids.forEach((id) => {
    const pairPattern = new RegExp(
      `<h2[^>]+home-accordion__heading[^>]*>\\s*<button[^>]+id="home-accordion-trigger-${id}"[^>]+type="button"[^>]+aria-controls="home-accordion-panel-${id}"[\\s\\S]*?<\\/button>\\s*<\\/h2>\\s*<section[^>]+id="home-accordion-panel-${id}"[^>]+role="region"[^>]+aria-labelledby="home-accordion-trigger-${id}"`
    );
    assert(pairPattern.test(html), `${id} rail should be a heading-wrapped native button followed by its labeled region`);
  });

  const requiredRoutes = [
    '/portfolio/handwritingRating',
    '/portfolio/babynames',
    '/portfolio/sheetMusicUpscale',
    '/portfolio/ufoDashboard',
    '/tools/text-compare',
    '/tools/image-optimizer',
    '/tools/qr-code-generator',
    '/tools/screen-recorder',
    '/games/project-starfall',
    '/games/stormbreak',
    '/games/stellar-dogfight',
    '/games/probability-engine',
    '/contact#contact-modal',
    'mailto:daniel@danielshort.me'
  ];
  requiredRoutes.forEach((route) => {
    assert(html.includes(`href="${route}"`), `generated homepage missing approved route ${route}`);
  });
  ['/tools/word-frequency', '/games/roulette', '/games/ocean-wave-simulation'].forEach((route) => {
    assert(!html.includes(`href="${route}"`), `homepage preview should leave ${route} to its full directory`);
  });
  ['campaign-creative-tracker', 'short-links', 'ga4-utm-performance', 'job-application-tracker', 'transcribe'].forEach((toolId) => {
    assert(!JSON.stringify(categories).includes(`\"id\":\"${toolId}\"`),
      `homepage should not expose hidden tool ${toolId}`);
  });
  const expectedLibraryCounts = {
    projects: 16,
    tools: 10,
    games: 6
  };
  assert(JSON.stringify(Object.fromEntries(Object.entries(homeLibraryData)
    .map(([id, library]) => [id, library.items?.length || 0]))) === JSON.stringify(expectedLibraryCounts),
  'generated HOME_LIBRARY_DATA should expose all 16 projects, 10 public tools, and 6 games');
  const publishedProjectIds = new Set(publishedProjects.map((project) => String(project.id)));
  assert(publishedProjects.length === expectedLibraryCounts.projects &&
    publishedProjects.every((project) => homeLibraryData.projects.items
      .some((item) => item.id === String(project.id))) &&
    homeLibraryData.projects.items.every((item) => publishedProjectIds.has(item.id)),
  'homepage project library should remain a complete projection of published project content');

  const expectedProjectMotifs = {
    smartSentence: 'semantic-retrieval',
    chatbotLora: 'grounded-chat',
    shapeClassifier: 'shape-classification',
    ufoDashboard: 'sighting-report',
    covidAnalysis: 'hospital-decision-tree',
    targetEmptyPackage: 'package-anomaly',
    handwritingRating: 'digit-legibility',
    digitGenerator: 'synthetic-digit-generation',
    sheetMusicUpscale: 'music-restoration',
    deliveryTip: 'delivery-tip-inputs',
    retailStore: 'retail-etl',
    pizza: 'pizza-regression-inputs',
    babynames: 'name-preference-learning',
    pizzaDashboard: 'delivery-operations-inputs',
    nonogram: 'nonogram-model',
    website: 'site-accordion'
  };
  const projectVisualEntries = Object.entries(HOME_LIBRARY_VISUALS.projects);
  const manifestMotifs = projectVisualEntries.map(([, motif]) => motif);
  assert(projectVisualEntries.length === expectedLibraryCounts.projects &&
    Object.entries(expectedProjectMotifs).every(([id, motif]) =>
      HOME_LIBRARY_VISUALS.projects[id] === motif) &&
    projectVisualEntries.every(([id]) => publishedProjectIds.has(id)),
  'project preview manifest should explicitly map all 16 published projects to their approved truth-safe concepts');
  const allManifestMotifs = Object.values(HOME_LIBRARY_VISUALS).flatMap((visuals) => Object.values(visuals));
  assert(new Set(manifestMotifs).size === expectedLibraryCounts.projects &&
    allManifestMotifs.length === 32 &&
    new Set(allManifestMotifs).size === 32 &&
    Object.entries(HOME_LIBRARY_VISUALS).every(([category, visuals]) =>
      JSON.stringify(Object.keys(visuals).sort()) ===
      JSON.stringify(homeLibraryData[category].items.map((item) => item.id).sort())),
  'every generated homepage preview should retain one unique semantic concept');
  assert(!fs.existsSync(path.join(ROOT, 'build', 'generate-home-library-visuals.js')) &&
    !fs.existsSync(path.join(ROOT, 'img', 'home-previews', 'sources')),
  'generated preview WebPs should be authoritative static assets without an old screenshot or code-art regeneration path');

  const previewPaths = [];
  const previewRoot = path.join(ROOT, 'img', 'home-previews');
  const actualPreviewCategories = fs.readdirSync(previewRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert(JSON.stringify(actualPreviewCategories) ===
    JSON.stringify(Object.keys(expectedLibraryCounts).sort()),
  'homepage preview root should contain only the exact lowercase projects, tools, and games directories');
  Object.entries(expectedLibraryCounts).forEach(([category, expectedCount]) => {
    const items = homeLibraryData[category]?.items || [];
    const expectedFileNames = items.map((item) => `${item.id}.webp`).sort();
    const categoryDir = path.join(previewRoot, category);
    const actualEntries = fs.readdirSync(categoryDir, { withFileTypes: true });
    const actualFileNames = actualEntries
      .map((entry) => entry.name)
      .sort();
    items.forEach((item) => previewPaths.push(item.image));
    assert(items.length === expectedCount &&
      actualEntries.length === expectedCount &&
      actualEntries.every((entry) => entry.isFile() && entry.name.endsWith('.webp')) &&
      items.every((item) => item.image === `/img/home-previews/${category}/${item.id}.webp` &&
        item.imageAlt === '') &&
      JSON.stringify(actualFileNames) === JSON.stringify(expectedFileNames),
    `${category} should map every item to one exact-case, decorative generated WebP asset`);
  });
  assert(previewPaths.length === 32 && new Set(previewPaths).size === 32,
    'all 32 homepage library preview paths should be unique');
  assert(previewPaths.every((previewPath) => {
    const dimensions = readWebpDimensions(previewPath);
    return dimensions.width === 640 && dimensions.height === 360;
  }),
  'every homepage library preview should carry exact 640 by 360 WebP metadata');
  const previewHashes = previewPaths.map((previewPath) => crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(ROOT, previewPath.replace(/^\/+/, ''))))
    .digest('hex'));
  assert(new Set(previewHashes).size === 32,
    'all 32 generated homepage preview files should have unique visual content');

  const cmsPreviewMappings = [
    "image: homeLibraryPreviewAsset('projects', project.id)",
    "image: homeLibraryPreviewAsset('tools', tool.id)",
    "image: homeLibraryPreviewAsset('games', game.id)"
  ];
  const visualBuildIndex = buildSite.indexOf(
    "runNodeScript(path.join('build', 'validate-home-library-visuals.js')"
  );
  const publicCopyIndex = buildSite.indexOf("runNodeScript(path.join('build', 'copy-to-public.js')");
  const publicVisualBuildIndex = buildSite.indexOf("{ verbose, args: ['--public'] }");
  assert(generator.includes('function homeLibraryPreviewAsset(category, id)') &&
    cmsPreviewMappings.every((mapping) => generator.includes(mapping)) &&
    count(generator, /imageAlt: '',/g) >= 3 &&
    visualValidator.includes("const sharp = require('sharp');") &&
    visualValidator.includes('const HOME_LIBRARY_VISUALS = {') &&
    visualValidator.includes('function validateCatalogMappings()') &&
    visualValidator.includes('function listPreviewTree(baseDir)') &&
    visualValidator.includes("validatePreviewTree(publicPreviewRoot, 'public/img/home-previews')") &&
    visualValidator.includes('validateMatchingHashes(sourceHashes, deployedHashes)') &&
    visualValidator.includes("metadata.width !== 640 || metadata.height !== 360") &&
    visualValidator.includes("new Set(hashes.values()).size !== hashes.size") &&
    visualValidator.includes('Validated ${sourceHashes.size} generated previews') &&
    packageJson.scripts?.['validate:home-library-visuals'] === 'node build/validate-home-library-visuals.js' &&
    Boolean(packageJson.devDependencies?.sharp) &&
    buildSite.includes('const scriptArgs = Array.isArray(options.args) ? options.args : [];') &&
    visualBuildIndex >= 0 && publicCopyIndex > visualBuildIndex && publicVisualBuildIndex > publicCopyIndex &&
    /const dirs = \[[^\]]*'img'/.test(copyToPublic),
  'CMS generation, full-tree validation, package scripts, main build, and hash-identical public copy should stay integrated');

  const createLibraryMediaJs = extractFunctionBlock(js, 'function createLibraryMedia');
  assert(createLibraryMediaJs.includes("image.alt = String(item.imageAlt || '')") &&
    createLibraryMediaJs.includes('image.width = 640;') &&
    createLibraryMediaJs.includes('image.height = 360;') &&
    createLibraryMediaJs.includes("image.loading = 'lazy';") &&
    createLibraryMediaJs.includes("image.decoding = 'async';"),
  'library renderer should emit decorative images with intrinsic 640 by 360 dimensions and lazy asynchronous decoding');
  const libraryToolIds = new Set(homeLibraryData.tools.items.map((tool) => tool.id));
  const excludedTools = toolsDirectoryItems.filter((tool) =>
    String(tool.visibility || 'public').toLowerCase() !== 'public' || tool.hidden || tool.noindex);
  const publicTools = toolsDirectoryItems.filter((tool) =>
    String(tool.visibility || 'public').toLowerCase() === 'public' && !tool.hidden && !tool.noindex);
  assert(excludedTools.length === 5 &&
    excludedTools.every((tool) => !libraryToolIds.has(tool.id)) &&
    publicTools.every((tool) => libraryToolIds.has(tool.id)) &&
    homeLibraryData.tools.items.every((tool) => publicTools.some((source) => source.id === tool.id)),
  'homepage tool library should include every public tool while excluding every hidden, admin, authenticated, or noindex tool');
  assert(generator.includes("String(tool.visibility || 'public').trim().toLowerCase() === 'public'") &&
    generator.includes('!tool.hidden && !tool.noindex'),
  'the HOME_LIBRARY_DATA generator should keep its explicit public, visible, and indexable tool filter');
  assert(!/recruiter|available for hire|resume|case study|kpi|professional analytics profile/i.test(JSON.stringify(categories)),
    'personal homepage copy should avoid job-seeking and dashboard framing');

  assert(!html.includes('data-home-graph') &&
    !html.includes('data-graph-') &&
    !html.includes('home-graph__inspector') &&
    !html.includes('home-graph__mobile-deck'),
  'generated homepage should not retain the detached graph, inspector, or duplicate mobile deck');
  assert(!fs.existsSync(path.join(ROOT, 'js/home/project-graph.js')) &&
    !fs.existsSync(path.join(ROOT, 'css/components/home-project-graph.css')),
  'retired graph source files should be removed');

  const libraryDataImportIndex = homeEntry.indexOf("import '../../js/home/home-library-data.js';");
  const accordionImportIndex = homeEntry.indexOf("import '../../js/home/category-accordion.js';");
  assert(libraryDataImportIndex >= 0 && accordionImportIndex > libraryDataImportIndex &&
    homeStyles.includes('@import url("components/home-category-accordion.css");') &&
    homeStyles.includes('@import url("components/home-library.css");') &&
    sharedStyles.includes('@import url("components/home-timeline.css");'),
  'homepage bundles should initialize library data before the controller and load accordion, library, and shared timeline styles');
  assert(personal.page.bottomScripts.some((script) => script.src === 'dist/site-home.js') &&
    !JSON.stringify(personal.page.bottomScripts).includes('project-graph'),
  'managed homepage source should use the stable home bundle without raw graph scripts');

  const overviewPanelCss = extractBlock(css, '.home-accordion__panel {');
  const overviewScrollerCss = extractBlock(css, '.home-accordion__scroller {');
  const desktopLibraryCss = extractBlock(libraryCss, '@media (min-width: 960px) and (min-height: 620px)');
  const desktopLibraryRailCss = extractBlock(
    desktopLibraryCss,
    '.home-accordion.is-library-mode .home-accordion__rail,'
  );
  const desktopLibraryRailIconCss = extractBlock(
    desktopLibraryCss,
    '.home-accordion.is-library-mode .home-accordion__rail-icon {'
  );
  const desktopLibraryRailLabelCss = extractBlock(
    desktopLibraryCss,
    '.home-accordion.is-library-mode .home-accordion__rail-label {'
  );
  const mobileLibraryCss = extractBlock(libraryCss, '@media (max-width: 959px), (max-height: 619px)');
  const phoneLibraryCss = extractBlock(libraryCss, '@media (max-width: 768px)');
  const mobileTimelineCss = extractBlock(timelineCss, '@media (max-width: 959px), (max-height: 619px)');
  assert(css.includes('--home-rail-width: 76px;') &&
    css.includes('--home-active-rail-width: 82px;') &&
    css.includes('--home-panel-motion: 520ms;') &&
    css.includes('flex-basis: calc(100% - var(--home-collapsed-rails-width));') &&
    css.includes('@keyframes homeAccordionContentIn') &&
    css.includes('gap: 0;') &&
    overviewPanelCss.includes('border: 5px solid var(--panel-color);') &&
    overviewPanelCss.includes('border-left: 0;') &&
    css.includes('right: -15px;') &&
    css.includes('clip-path: polygon(0 0, 100% 50%, 0 100%);'),
  'desktop overview should smoothly exchange panel width while keeping touching rails, a subtly wider active rail, a 5px frame, and right-facing word notch');
  assert(overviewPanelCss.includes('background: #ffffff;') &&
    overviewPanelCss.includes('border-radius: 0 0 12px 0;') &&
    overviewScrollerCss.includes('overflow-y: scroll;') &&
    overviewScrollerCss.includes('overscroll-behavior: contain;') &&
    css.includes('position: sticky;') &&
    overviewScrollerCss.includes('scrollbar-color:'),
  'selected panel should stay white with square flush top corners, a sticky heading, and visible contained desktop scroll');
  assert(css.includes('.home-accordion__card-glyph {\n    box-sizing: border-box;'),
    'homepage card glyphs should keep their padding inside the media tile instead of clipping');
  assert(css.includes('@media (max-width: 959px), (max-height: 619px)') &&
    css.includes('display: block;') &&
    css.includes('max-height: none;') &&
    css.includes('overflow: visible;') &&
    css.includes('clip-path: polygon(0 0, 100% 0, 50% 100%);'),
  'narrow or short accordion layouts should stack full-width bars, use document scrolling, and point the active notch down');
  assert(desktopLibraryCss.includes('--home-library-tab-width: 84px;') &&
    desktopLibraryCss.includes('grid-template-columns: var(--home-library-tab-width) minmax(0, 1fr);') &&
    desktopLibraryCss.includes('grid-template-rows: repeat(5, minmax(0, 1fr));') &&
    [1, 2, 3, 4, 5].every((position) => desktopLibraryCss.includes(
      `.home-accordion__item:nth-child(${position}) .home-accordion__heading {\n      grid-row: ${position};`
    )) &&
    desktopLibraryCss.includes('grid-column: 1;') &&
    desktopLibraryCss.includes('width: var(--home-library-tab-width);') &&
    desktopLibraryRailCss.includes('flex-direction: column;') &&
    desktopLibraryRailCss.includes('align-items: center;') &&
    desktopLibraryRailIconCss.includes('position: static;') &&
    desktopLibraryRailIconCss.includes('flex: 0 0 22px;') &&
    desktopLibraryRailLabelCss.includes('writing-mode: vertical-rl;') &&
    desktopLibraryRailLabelCss.includes('transform: rotate(180deg);') &&
    desktopLibraryCss.includes('grid-column: 2;') &&
    desktopLibraryCss.includes('grid-row: 1 / -1;') &&
    desktopLibraryCss.includes('border: 5px solid var(--panel-color);') &&
    desktopLibraryCss.includes('overflow-y: scroll;'),
  'expanded desktop libraries should use five equal, narrow icon-above-vertical-label tabs beside one 5px-framed, internally scrolling content column');
  assert(mobileLibraryCss.includes('grid-template-columns: repeat(5, minmax(0, 1fr));') &&
    mobileLibraryCss.includes('grid-template-rows: 78px auto;') &&
    [1, 2, 3, 4, 5].every((position) => mobileLibraryCss.includes(
      `.home-accordion__item:nth-child(${position}) .home-accordion__heading {\n      grid-column: ${position};`
    )) &&
    mobileLibraryCss.includes('writing-mode: horizontal-tb;') &&
    mobileLibraryCss.includes('grid-column: 1 / -1;') &&
    mobileLibraryCss.includes('grid-row: 2;') &&
    mobileLibraryCss.includes('border: 5px solid var(--panel-color);') &&
    phoneLibraryCss.includes('position: fixed;') &&
    phoneLibraryCss.includes('top: var(--mobile-site-masthead-height, 62px);') &&
    phoneLibraryCss.includes('width: 20vw;') &&
    phoneLibraryCss.includes('left: 80vw;'),
  'expanded mobile libraries should present five equal-width persistent horizontal tabs above one full-width 5px-framed content row');
  assert(mobileTimelineCss.includes('grid-auto-flow: column;') &&
    mobileTimelineCss.includes('grid-auto-columns: minmax(270px, 82vw);') &&
    mobileTimelineCss.includes('overflow-x: auto;') &&
    mobileTimelineCss.includes('overflow-y: hidden;') &&
    mobileTimelineCss.includes('scroll-snap-type: inline mandatory;') &&
    mobileTimelineCss.includes('overscroll-behavior-inline: contain;') &&
    mobileTimelineCss.includes('scroll-snap-align: start;') &&
    mobileTimelineCss.includes('scroll-snap-stop: always;'),
  'mobile timeline should be a contained horizontal card scroller with explicit snap positions');
  assert(css.includes('@media (pointer: coarse)') &&
    css.includes('min-height: 44px;') &&
    css.includes('@media (prefers-reduced-motion: reduce)') &&
    css.includes('.home-accordion__rail:focus-visible') &&
    libraryCss.includes('@media (prefers-reduced-motion: reduce)') &&
    timelineCss.includes('@media (prefers-reduced-motion: reduce)') &&
    timelineCss.includes('scroll-behavior: auto;'),
  'accordion should preserve touch targets, reduced motion, and visible keyboard focus');
  assert(!/prefers-color-scheme\s*:\s*dark/i.test(css) &&
    !/prefers-color-scheme\s*:\s*dark/i.test(libraryCss) &&
    !/prefers-color-scheme\s*:\s*dark/i.test(timelineCss) &&
    !/color-scheme\s*:\s*dark/i.test(css) &&
    !/data-theme/i.test(css),
  'homepage accordion should remain intentionally light-only');

  const canonicalPanelHashJs = extractFunctionBlock(js, 'function canonicalPanelHash');
  const updateLocationJs = extractFunctionBlock(js, 'function updateLocation');
  const updateTriggerStateJs = extractFunctionBlock(js, 'function updateTriggerState');
  const updateLibraryVisibilityJs = extractFunctionBlock(js, 'function updateLibraryViewVisibility');
  const applyLibraryModeJs = extractFunctionBlock(js, 'function applyLibraryMode');
  const resolveTriggerTargetJs = extractFunctionBlock(js, 'function resolveTriggerTarget');
  const activatePanelTriggerJs = extractFunctionBlock(js, 'function activatePanelTrigger');
  const openLibraryJs = extractFunctionBlock(js, 'function openLibrary');
  const closeLibraryJs = extractFunctionBlock(js, 'function closeLibrary');
  const handleLocationChangeJs = extractFunctionBlock(js, 'function handleLocationChange');
  assert(js.includes('const closeTimers = new Map();') &&
    js.includes('const PANEL_TRANSITION_MS = 520;') &&
    js.includes("item.classList.add('is-closing')") &&
    js.includes('finishClosingPanel(itemId)') &&
    js.includes("panel.setAttribute('inert', '')") &&
    js.includes("panel.removeAttribute('inert')") &&
    js.includes('if (!ids.includes(id)) return false;') &&
    js.includes('if (id === activeId) {'),
  'accordion controller should animate the outgoing desktop panel, enforce one interactive panel, and keep its low-level selection operation idempotent');
  const triggerResolutionContext = {};
  vm.runInNewContext(
    `${resolveTriggerTargetJs}\nthis.resolveTriggerTarget = resolveTriggerTarget;`,
    triggerResolutionContext
  );
  const triggerResolutionCases = [
    ['about', 'about', 'about'],
    ['projects', 'projects', 'about'],
    ['tools', 'tools', 'about'],
    ['games', 'games', 'about'],
    ['contact', 'contact', 'about'],
    ['projects', 'tools', 'tools']
  ];
  assert(triggerResolutionCases.every(([currentId, requestedId, expectedId]) =>
    triggerResolutionContext.resolveTriggerTarget(currentId, requestedId, 'about') === expectedId) &&
    activatePanelTriggerJs.includes('if (isLibraryMode) return selectPanel(id);') &&
    activatePanelTriggerJs.includes('const nextId = resolveTriggerTarget(activeId, id, defaultPanel);') &&
    activatePanelTriggerJs.includes('triggerById.get(defaultPanel)?.focus({ preventScroll: true })') &&
    count(js, /activatePanelTrigger\(String\(trigger\.dataset\.homeAccordionTrigger \|\| ''\)\)/g) === 2,
  're-activating any expanded non-About overview tab should return to and focus About through both pointer and keyboard controls, while library mode remains explicit');
  assert(js.includes("event.key === 'ArrowDown'") &&
    js.includes("event.key === 'ArrowRight'") &&
    js.includes("event.key === 'ArrowUp'") &&
    js.includes("event.key === 'ArrowLeft'") &&
    js.includes("event.key === 'Home'") &&
    js.includes("event.key === 'End'") &&
    js.includes("event.key === 'Enter'") &&
    js.includes("event.key === ' '"),
  'accordion rails should support directional movement plus Enter and Space activation');
  assert(js.includes('scrollPositions') &&
    js.includes('scroller.scrollTop') &&
    js.includes("window.matchMedia('(min-width: 960px) and (min-height: 620px)')") &&
    js.includes('scrollIntoView'),
  'accordion should preserve spacious-rail panel scroll positions and reveal newly opened stacked sections');
  assert(js.includes('decodeURIComponent(rawId)') &&
    js.includes('catch (error)') &&
    js.includes("new URL(window.location.href).searchParams.get('view') === 'library'") &&
    handleLocationChangeJs.includes('if (window.location.hash && !hashPanel) return;') &&
    handleLocationChangeJs.includes("const nextPanel = hashPanel || (nextLibraryMode && ids.includes('projects') ? 'projects' : defaultPanel);") &&
    handleLocationChangeJs.includes("else updateLocation(nextPanel, 'replace', nextLibraryMode);") &&
    handleLocationChangeJs.includes('selectPanel(nextPanel, { updateHistory: false, reveal: !nextLibraryMode })') &&
    js.includes("window.addEventListener('hashchange', handleLocationChange)") &&
    js.includes("window.addEventListener('popstate', handleLocationChange)"),
  'accordion deep links should safely decode tab hashes, preserve unrelated anchors, restore history state, and default query-only library links to Projects');
  const canonicalHashContext = {};
  vm.runInNewContext(
    `${canonicalPanelHashJs}\nthis.canonicalPanelHash = canonicalPanelHash;`,
    canonicalHashContext
  );
  const expectedTabHashes = {
    about: '#about',
    projects: '#projects',
    tools: '#tools',
    games: '#games',
    contact: '#contact'
  };
  assert(ids.every((id) => canonicalHashContext.canonicalPanelHash(id) === expectedTabHashes[id]) &&
    updateLocationJs.includes('url.hash = id;'),
  'every homepage tab should have a stable, canonical hash URL');
  assert(updateLocationJs.includes("url.searchParams.set('view', 'library')") &&
    updateLocationJs.includes("url.searchParams.delete('view')") &&
    updateLocationJs.includes("mode === 'replace' ? 'replaceState' : 'pushState'") &&
    updateLocationJs.includes("homeView: libraryMode ? 'library' : 'overview'") &&
    js.includes("updateLocation(id, options.historyMode || 'push')") &&
    js.includes("updateLocation(id, 'replace')"),
  'explicit category and view selections should preserve a canonical hash/query URL and use push versus replace history deliberately');
  assert(updateLibraryVisibilityJs.includes('const visible = isLibraryMode && id === activeId;') &&
    updateLibraryVisibilityJs.includes('view.hidden = false;') &&
    updateLibraryVisibilityJs.includes("view.removeAttribute('inert')") &&
    updateLibraryVisibilityJs.includes('view.hidden = true;') &&
    updateLibraryVisibilityJs.includes("view.setAttribute('inert', '')") &&
    updateLibraryVisibilityJs.includes("button.setAttribute('aria-expanded', String(isLibraryMode && activeId === id))"),
  'library visibility should keep only the active inline view interactive and synchronize its disclosure button');
  assert(openLibraryJs.includes('if (!libraryIds.has(id)) return;') &&
    openLibraryJs.includes('selectPanel(id, { updateHistory: false, reveal: false })') &&
    openLibraryJs.includes('renderLibrary(id);') &&
    openLibraryJs.includes('applyLibraryMode(true);') &&
    openLibraryJs.includes("updateLocation(id, 'push', true)") &&
    openLibraryJs.includes('focusLibraryHeading(id);') &&
    closeLibraryJs.includes('applyLibraryMode(false, { afterApply: restoreFocus });') &&
    closeLibraryJs.includes("updateLocation(closingId, options.historyMode || 'push', false)") &&
    closeLibraryJs.includes('returnTarget?.focus({ preventScroll: true })'),
  'library open and close flows should render inline, push view state, move focus inward, and restore focus on return');
  assert(applyLibraryModeJs.includes("root.classList.toggle('is-library-mode', next)") &&
    applyLibraryModeJs.includes("root.dataset.homeView = next ? 'library' : 'overview'") &&
    applyLibraryModeJs.includes("typeof options.afterApply === 'function'") &&
    applyLibraryModeJs.includes('!reducedMotionQuery.matches') &&
    applyLibraryModeJs.includes("typeof document.startViewTransition === 'function'") &&
    applyLibraryModeJs.includes('document.startViewTransition(apply)') &&
    applyLibraryModeJs.includes('markViewTransition()') &&
    js.includes('const animateIncoming = options.animateIncoming === true && !reducedMotionQuery.matches;'),
  'library and panel transitions should use native or fallback animation only when reduced motion is not requested');
  assert(updateTriggerStateJs.includes("const isNonCollapsible = selected && (triggerId === defaultPanel || isLibraryMode);") &&
    updateTriggerStateJs.includes("trigger.setAttribute('aria-disabled', 'true')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-disabled')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-current')") &&
    applyLibraryModeJs.includes('updateTriggerState(triggerById.get(activeId), true);') &&
    js.includes("scroller.setAttribute('tabindex', '0')") &&
    js.includes("scroller.removeAttribute('tabindex')") &&
    js.includes('scroller.scrollHeight > scroller.clientHeight + 1'),
  'only active About and library-mode rails should expose a noncollapsible state, and only independently scrollable rail panels should add a tab stop');
  assert(js.includes('const initialLibraryMode = locationRequestsLibrary();') &&
    js.includes('const initialHashPanel = panelIdFromHash();') &&
    js.includes("const initialPanel = initialHashPanel || (initialLibraryMode && ids.includes('projects') ? 'projects' : defaultPanel);") &&
    js.includes('applyLibraryMode(initialLibraryMode, { animate: false, force: true })') &&
    js.includes("updateLocation(initialPanel, 'replace', true)") &&
    js.includes("updateLocation(initialPanel, 'replace', false)") &&
    js.includes('revealPanelTrigger(initialHashPanel);'),
  'initial deep links should restore the requested category/library without animation and canonicalize bare overview and query-only library URLs');
  assert(!html.includes('data-home-accordion-scroller tabindex="0"') &&
    html.includes('<h3>Hi, I’m Daniel.</h3>'),
  'authored panels should avoid generic scroller tab stops and keep panel titles beneath accordion headings');

  assert(navigation.includes("const nextExpanded = enhanced && Boolean(expanded);") &&
    navigation.includes("if (!form.classList.contains('is-expanded'))") &&
    !navigation.includes('const isHomeSearch ='),
  'homepage search should use the same compact, explicitly expandable desktop behavior as the rest of the site');

  assert(navigation.includes("if (document.body.dataset.page === 'home') return;") &&
    css.includes('.mobile-site-dock {') &&
    css.includes('display: none !important;'),
  'selected homepage should prevent the shared bottom dock and retain a CSS safety fallback');
  assert(activityEvents.includes("target.closest('[data-home-accordion]')") &&
    activityEvents.includes('category.dataset.homeAccordionTrigger'),
  'homepage category changes should remain analytics-visible');
};
