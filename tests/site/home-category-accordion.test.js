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
  GENERATED_HOME_LIBRARY_VISUALS,
  RETAINED_GAME_PREVIEW_IDS,
  RETAINED_PROJECT_PREVIEW_IDS,
  projectLibraryAsset
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
    games: ['stormbreak', 'stellar-dogfight', 'probability-engine'],
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
    'eastern-ms-data-science'
  ];
  assert(about.timeline?.title === 'My path so far' &&
    JSON.stringify(timelineItems.map((item) => item.id)) === JSON.stringify(expectedTimelineIds),
  'About should carry the approved 10-event personal timeline in chronological narrative order');
  const purdueTimelineItem = timelineItems.find((item) => item.id === 'purdue-bs-data-analytics');
  assert(purdueTimelineItem?.image === 'img/cert_logos/purdue_global.png' &&
    purdueTimelineItem?.imageWidth === 137 &&
    purdueTimelineItem?.imageHeight === 136 &&
    purdueTimelineItem?.imageTone === 'dark',
  'Purdue should use its full-resolution logo and an authored high-contrast plaque treatment');
  assert(count(html, /class="home-timeline__axis"/g) === timelineItems.length &&
    count(html, /class="home-timeline__dot"/g) === timelineItems.length &&
    html.includes('<ol class="home-timeline__list" data-home-timeline-scroller>'),
  'each timeline event should render one shared axis and explicit dot aligned with the dedicated timeline scroll region');
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
  assert(games.items.every((item) => item.icon && !item.image) &&
    !JSON.stringify(games).includes('project-starfall'),
  'Games should use consistent semantic glyph tiles and exclude inactive Project Starfall content');
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
    aboutHtml.includes('<ol class="home-timeline__list" data-home-timeline-scroller>') &&
    count(aboutHtml, /<li class="home-timeline__item[^>]+data-home-timeline-item=/g) === 10 &&
    !aboutHtml.includes('role="list"') &&
    !aboutHtml.includes('role="listitem"'),
  'rendered About timeline should be a labelled section with a native ordered list of 10 semantic events');
  assert(/data-home-timeline-item="purdue-bs-data-analytics"[\s\S]*?data-home-timeline-media-tone="dark"><img src="img\/cert_logos\/purdue_global\.png"[^>]+width="137" height="136"/.test(aboutHtml),
    'rendered Purdue milestone should retain the explicit dark plaque flag and undistorted intrinsic logo dimensions');
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
    ['projects', '/portfolio', 'View all projects'],
    ['tools', '/tools', 'View all tools'],
    ['games', '/games', 'View all games']
  ].forEach(([id, href, label]) => {
    const itemHtml = getItemHtml(id);
    const cta = `<a class="home-accordion__panel-cta" href="${href}">${label}`;
    assert(itemHtml.includes(cta) &&
      itemHtml.indexOf(cta) > itemHtml.indexOf('<ul class="home-accordion__cards">') &&
      !itemHtml.includes('data-home-library') &&
      !itemHtml.includes('home-library'),
    `${id} View all control should be a semantic link to its canonical directory without an authored inline library`);
  });
  assert(!html.includes('data-home-library') &&
    !html.includes('home-library') &&
    !html.includes('data-home-view=') &&
    !html.includes('Back to categories') &&
    !html.includes('Back to overview'),
  'homepage markup should contain only the five-category overview without inline-library state or return controls');
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
    '/games/stormbreak',
    '/games/stellar-dogfight',
    '/games/probability-engine',
    '/contact#contact-modal',
    'mailto:daniel@danielshort.me'
  ];
  requiredRoutes.forEach((route) => {
    assert(html.includes(`href="${route}"`), `generated homepage missing approved route ${route}`);
  });
  assert(/href="\/contact#contact-modal"[^>]*data-contact-modal-link/.test(getItemHtml('contact')),
    'homepage Send a message card should opt into the shared in-page contact modal');
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
    games: 5
  };
  assert(JSON.stringify(Object.fromEntries(Object.entries(homeLibraryData)
    .map(([id, library]) => [id, library.items?.length || 0]))) === JSON.stringify(expectedLibraryCounts),
  'generated HOME_LIBRARY_DATA should expose all 16 projects, 10 public tools, and 5 games');
  const publishedProjectIds = new Set(publishedProjects.map((project) => String(project.id)));
  assert(publishedProjects.length === expectedLibraryCounts.projects &&
    publishedProjects.every((project) => homeLibraryData.projects.items
      .some((item) => item.id === String(project.id))) &&
    homeLibraryData.projects.items.every((item) => publishedProjectIds.has(item.id)),
  'homepage project library should remain a complete projection of published project content');

  const publishedProjectsById = new Map(publishedProjects.map((project) => [String(project.id), project]));
  const projectPreviewPaths = homeLibraryData.projects.items.map((item) => item.image);
  assert(homeLibraryData.projects.items.every((item) => {
    const project = publishedProjectsById.get(item.id);
    return project &&
      item.image === projectLibraryAsset(project.image) &&
      item.image === `/img/projects/${item.id}-640.webp` &&
      item.imageAlt === '';
  }),
  'all 16 project library cards should derive their original optimized preview from the canonical project image');
  const allManifestMotifs = Object.values(GENERATED_HOME_LIBRARY_VISUALS)
    .flatMap((visuals) => Object.values(visuals));
  assert(JSON.stringify(Object.keys(GENERATED_HOME_LIBRARY_VISUALS).sort()) === JSON.stringify(['games', 'tools']) &&
    allManifestMotifs.length === expectedLibraryCounts.tools + expectedLibraryCounts.games &&
    new Set(allManifestMotifs).size === allManifestMotifs.length &&
    Object.entries(GENERATED_HOME_LIBRARY_VISUALS).every(([category, visuals]) =>
      JSON.stringify(Object.keys(visuals).sort()) ===
      JSON.stringify(homeLibraryData[category].items.map((item) => item.id).sort())),
  'every generated tool and game preview should retain one unique semantic concept');
  assert(!fs.existsSync(path.join(ROOT, 'build', 'generate-home-library-visuals.js')) &&
    !fs.existsSync(path.join(ROOT, 'img', 'home-previews', 'sources')),
  'generated preview WebPs should be authoritative static assets without an old screenshot or code-art regeneration path');

  const generatedPreviewPaths = [];
  const previewRoot = path.join(ROOT, 'img', 'home-previews');
  const actualPreviewCategories = fs.readdirSync(previewRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
  assert(JSON.stringify(actualPreviewCategories) ===
    JSON.stringify(Object.keys(expectedLibraryCounts).sort()),
  'homepage preview root should contain only the exact lowercase projects, tools, and games directories');
  const retainedProjectPreviewNames = fs.readdirSync(path.join(previewRoot, 'projects')).sort();
  assert(JSON.stringify(retainedProjectPreviewNames) === JSON.stringify(
    RETAINED_PROJECT_PREVIEW_IDS.map((id) => `${id}.webp`).sort()) &&
    homeLibraryData.projects.items.every((item) =>
      !item.image.startsWith('/img/home-previews/projects/')),
  'legacy AI project previews should remain available but unused by the project library');
  Object.entries(GENERATED_HOME_LIBRARY_VISUALS).forEach(([category, visuals]) => {
    const expectedCount = Object.keys(visuals).length;
    const items = homeLibraryData[category]?.items || [];
    const retainedFileNames = category === 'games'
      ? RETAINED_GAME_PREVIEW_IDS.map((id) => `${id}.webp`)
      : [];
    const expectedFileNames = [
      ...items.map((item) => `${item.id}.webp`),
      ...retainedFileNames
    ].sort();
    const categoryDir = path.join(previewRoot, category);
    const actualEntries = fs.readdirSync(categoryDir, { withFileTypes: true });
    const actualFileNames = actualEntries
      .map((entry) => entry.name)
      .sort();
    items.forEach((item) => generatedPreviewPaths.push(item.image));
    assert(items.length === expectedCount &&
      actualEntries.length === expectedCount + retainedFileNames.length &&
      actualEntries.every((entry) => entry.isFile() && entry.name.endsWith('.webp')) &&
      items.every((item) => item.image === `/img/home-previews/${category}/${item.id}.webp` &&
        item.imageAlt === '') &&
      JSON.stringify(actualFileNames) === JSON.stringify(expectedFileNames),
    `${category} should map every item to one exact-case, decorative generated WebP asset`);
  });
  assert(projectPreviewPaths.length === expectedLibraryCounts.projects &&
    generatedPreviewPaths.length === expectedLibraryCounts.tools + expectedLibraryCounts.games &&
    new Set([...projectPreviewPaths, ...generatedPreviewPaths]).size === 31,
  'all 31 public library preview paths should remain unique');
  assert(projectPreviewPaths.every((previewPath) => {
    const dimensions = readWebpDimensions(previewPath);
    return dimensions.width === 640 && dimensions.height > 0;
  }),
  'every original project library preview should be a valid 640px-wide WebP');
  assert(generatedPreviewPaths.every((previewPath) => {
    const dimensions = readWebpDimensions(previewPath);
    return dimensions.width === 640 && dimensions.height === 360;
  }),
  'every generated tool and game preview should carry exact 640 by 360 WebP metadata');
  const previewHashes = generatedPreviewPaths.map((previewPath) => crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(ROOT, previewPath.replace(/^\/+/, ''))))
    .digest('hex'));
  assert(new Set(previewHashes).size === generatedPreviewPaths.length,
  'all 15 public tool and game preview files should have unique visual content');

  const cmsPreviewMappings = [
    'image: projectLibraryPreviewAsset(project.image)',
    "image: homeLibraryPreviewAsset('tools', tool.id)",
    "image: homeLibraryPreviewAsset('games', game.id)"
  ];
  const visualBuildIndex = buildSite.indexOf(
    "runNodeScript(path.join('build', 'validate-home-library-visuals.js')"
  );
  const publicCopyIndex = buildSite.indexOf("runNodeScript(path.join('build', 'copy-to-public.js')");
  const publicVisualBuildIndex = buildSite.indexOf("{ verbose, args: ['--public'] }");
  assert(generator.includes('function homeLibraryPreviewAsset(category, id)') &&
    generator.includes('function projectLibraryPreviewAsset(image)') &&
    cmsPreviewMappings.every((mapping) => generator.includes(mapping)) &&
    count(generator, /imageAlt: '',/g) >= 3 &&
    visualValidator.includes("const sharp = require('sharp');") &&
    visualValidator.includes('const GENERATED_HOME_LIBRARY_VISUALS = {') &&
    visualValidator.includes("const RETAINED_GAME_PREVIEW_IDS = ['project-starfall'];") &&
    visualValidator.includes('const RETAINED_PROJECT_PREVIEW_IDS = [') &&
    visualValidator.includes('function projectLibraryAsset(image)') &&
    visualValidator.includes('function validateCatalogMappings()') &&
    visualValidator.includes('async function validateProjectAssets(baseDir, projects)') &&
    visualValidator.includes('function listPreviewTree(baseDir)') &&
    visualValidator.includes("validatePreviewTree(publicPreviewRoot, 'public/img/home-previews')") &&
    visualValidator.includes('validateMatchingHashes(sourceHashes, deployedHashes)') &&
    visualValidator.includes("metadata.width !== 640 || metadata.height !== 360") &&
    visualValidator.includes("new Set(hashes.values()).size !== hashes.size") &&
    visualValidator.includes('Validated ${projectSourceHashes.size} original project previews') &&
    packageJson.scripts?.['validate:home-library-visuals'] === 'node build/validate-home-library-visuals.js' &&
    Boolean(packageJson.devDependencies?.sharp) &&
    buildSite.includes('const scriptArgs = Array.isArray(options.args) ? options.args : [];') &&
    visualBuildIndex >= 0 && publicCopyIndex > visualBuildIndex && publicVisualBuildIndex > publicCopyIndex &&
    /const dirs = \[[^\]]*'img'/.test(copyToPublic),
  'CMS generation, project-original validation, generated-preview validation, main build, and hash-identical public copy should stay integrated');

  assert(!js.includes('HOME_LIBRARY_DATA') &&
    !js.includes('createLibraryMedia') &&
    !js.includes('createLibraryCard') &&
    !js.includes('renderLibrary'),
  'homepage controller should not ship the retired inline-library renderer or its data dependency');
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

  const accordionImportIndex = homeEntry.indexOf("import '../../js/home/category-accordion.js';");
  assert(accordionImportIndex >= 0 &&
    homeStyles.includes('@import url("components/home-category-accordion.css");') &&
    sharedStyles.includes('@import url("components/home-timeline.css");'),
  'homepage bundles should initialize the overview controller and load its accordion and shared timeline styles');
  assert(personal.page.bottomScripts.some((script) => script.src === 'dist/site-home.js') &&
    !JSON.stringify(personal.page.bottomScripts).includes('project-graph'),
  'managed homepage source should use the stable home bundle without raw graph scripts');

  const overviewPanelCss = extractBlock(css, '.home-accordion__panel {');
  const overviewScrollerCss = extractBlock(css, '.home-accordion__scroller {');
  const overviewRailCss = extractBlock(css, '.home-accordion__rail {');
  const profileImageCss = extractBlock(css, '.home-accordion__profile-portrait img {');
  const mobileAccordionCss = extractBlock(css, '@media (max-width: 959px), (max-height: 619px)');
  const desktopTimelineCss = extractBlock(timelineCss, '@media (min-width: 960px) and (min-height: 620px)');
  const mobileTimelineCss = extractBlock(timelineCss, '@media (max-width: 959px), (max-height: 619px)');
  const evenTimelineAxisCss = extractBlock(timelineCss, '.home-timeline__item:nth-child(even) .home-timeline__axis::after');
  assert(css.includes('--home-rail-width: 64px;') &&
    css.includes('--home-active-rail-width: 68px;') &&
    css.includes('--home-collapsed-rails-width: 256px;') &&
    css.includes('--home-panel-motion: 520ms;') &&
    css.includes('flex-basis: calc(100% - var(--home-collapsed-rails-width));') &&
    css.includes('@keyframes homeAccordionContentIn') &&
    css.includes('gap: 0;') &&
    overviewRailCss.includes('background: var(--panel-color);') &&
    !overviewRailCss.includes('linear-gradient') &&
    overviewPanelCss.includes('border: 4px solid var(--panel-color);') &&
    overviewPanelCss.includes('border-left: 0;') &&
    css.includes('right: -11px;') &&
    css.includes('width: 12px;') &&
    css.includes('height: 22px;') &&
    css.includes('clip-path: polygon(0 0, 100% 50%, 0 100%);'),
  'desktop overview should use compact solid 64px rails, a 68px active rail, a 4px frame, and a smaller right-facing notch');
  assert(overviewPanelCss.includes('background: #ffffff;') &&
    overviewPanelCss.includes('border-radius: 0;') &&
    overviewScrollerCss.includes('overflow-y: auto;') &&
    overviewScrollerCss.includes('overscroll-behavior: contain;') &&
    css.includes('position: sticky;') &&
    overviewScrollerCss.includes('scrollbar-color: var(--home-scrollbar-thumb) var(--home-scrollbar-track);') &&
    css.includes('.home-accordion__item:last-child.is-active .home-accordion__panel') &&
    css.includes('border-radius: 0 12px 12px 0;'),
  'selected overview panels should stay white and square, use conditional themed scrolling, and round both outer-right corners only for Contact');
  assert(css.includes('.home-accordion__panel-head {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__context {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__meta {\n    box-sizing: border-box;') &&
    css.includes('.home-accordion__cards {\n    box-sizing: border-box;') &&
    css.includes('max-inline-size: 100%;') &&
    profileImageCss.includes('object-fit: contain;') &&
    profileImageCss.includes('object-position: center;'),
  'all padded panel content should remain inside the scroller width and keep the profile portrait fully visible');
  assert(css.includes('.home-accordion__card-glyph {\n    box-sizing: border-box;'),
    'homepage card glyphs should keep their padding inside the media tile instead of clipping');
  assert(mobileAccordionCss.includes('display: block;') &&
    mobileAccordionCss.includes('max-height: none;') &&
    mobileAccordionCss.includes('overflow: visible;') &&
    mobileAccordionCss.includes('height: 48px;') &&
    mobileAccordionCss.includes('min-height: 48px;') &&
    mobileAccordionCss.includes('height: 54px;') &&
    mobileAccordionCss.includes('min-height: 54px;') &&
    mobileAccordionCss.includes('border: 4px solid var(--panel-color);') &&
    mobileAccordionCss.includes('width: 20px;') &&
    mobileAccordionCss.includes('height: 10px;') &&
    mobileAccordionCss.includes('clip-path: polygon(0 0, 100% 0, 50% 100%);'),
  'narrow or short accordion layouts should stack compact 48px and 54px bars, use document scrolling, and retain the 4px framed panel');
  assert(mobileTimelineCss.includes('grid-auto-flow: column;') &&
    mobileTimelineCss.includes('grid-auto-columns: minmax(260px, 82vw);') &&
    mobileTimelineCss.includes('overflow-x: auto;') &&
    mobileTimelineCss.includes('overflow-y: hidden;') &&
    mobileTimelineCss.includes('scroll-snap-type: inline mandatory;') &&
    mobileTimelineCss.includes('overscroll-behavior-inline: contain;') &&
    mobileTimelineCss.includes('scroll-snap-align: start;') &&
    mobileTimelineCss.includes('scroll-snap-stop: always;') &&
    mobileTimelineCss.includes('scrollbar-width: none;') &&
    mobileTimelineCss.includes('display: none;') &&
    mobileTimelineCss.includes('.home-timeline__dot') &&
    mobileTimelineCss.includes('top: 11px;') &&
    mobileTimelineCss.includes('width: calc(100% + 14px);'),
  'mobile timeline should preserve horizontal snapping, hide native scrollbars, show the next card, and align connectors through each dot center');
  assert(timelineCss.includes('.home-timeline__axis {') &&
    timelineCss.includes('grid-column: 2;') &&
    timelineCss.includes('grid-row: 2;') &&
    timelineCss.includes('.home-timeline__item::before') &&
    timelineCss.includes('.home-timeline__item::after') &&
    timelineCss.includes('height: var(--home-timeline-gap);') &&
    timelineCss.includes('.home-timeline__item:first-child .home-timeline__axis::before') &&
    timelineCss.includes('.home-timeline__item:last-child .home-timeline__axis::before') &&
    timelineCss.includes('.home-timeline__dot {') &&
    timelineCss.includes('transform: translate(-50%, -50%);') &&
    evenTimelineAxisCss.includes('right: 0;') &&
    evenTimelineAxisCss.includes('left: 50%;') &&
    desktopTimelineCss.includes('overflow-y: hidden;') &&
    desktopTimelineCss.includes('width: 100%;') &&
    desktopTimelineCss.includes('padding-right: 0;') &&
    desktopTimelineCss.includes('width: min(100%, var(--home-timeline-readable-width));') &&
    desktopTimelineCss.includes('.home-accordion__item--about .home-timeline__head') &&
    desktopTimelineCss.includes('flex: 0 0 auto;') &&
    desktopTimelineCss.includes('.home-accordion__item--about .home-timeline__list') &&
    desktopTimelineCss.includes('overflow-y: auto;'),
  'desktop timeline should connect exact milestone centers across variable rows and gaps while its full-width scrollport keeps readable cards below the fixed profile and divider');
  assert(timelineCss.includes('.home-timeline__media[data-home-timeline-media-tone="dark"]') &&
    timelineCss.includes('object-fit: contain;') &&
    timelineCss.includes('object-position: center;') &&
    timelineCss.includes('background: #091f3b;'),
  'timeline logo plaques should preserve aspect ratios and support an explicit high-contrast treatment');
  const homepageCardCss = extractBlock(css, '.home-accordion__card {');
  const homepageCardInteractiveCss = extractBlock(css, '.home-accordion__card:is(a):is(:hover, :focus-visible)');
  const reducedMotionCss = extractBlock(css, '@media (prefers-reduced-motion: reduce)');
  assert(homepageCardCss.includes('padding-inline-start .18s ease') &&
    homepageCardInteractiveCss.includes('padding-inline-start: 8px;') &&
    reducedMotionCss.includes('.home-accordion__card') &&
    reducedMotionCss.includes('transition: none;'),
  'main-tab preview cards should gain a subtle smooth left inset on hover and keyboard focus without animating for reduced motion');
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
  const legacyLocationJs = extractFunctionBlock(js, 'function normalizeLegacyLibraryLocation');
  const resolveTriggerTargetJs = extractFunctionBlock(js, 'function resolveTriggerTarget');
  const activatePanelTriggerJs = extractFunctionBlock(js, 'function activatePanelTrigger');
  const handleLocationChangeJs = extractFunctionBlock(js, 'function handleLocationChange');
  assert(js.includes('const closeTimers = new Map();') &&
    js.includes('const PANEL_TRANSITION_MS = 520;') &&
    js.includes("item.classList.add('is-closing')") &&
    js.includes('finishClosingPanel(itemId)') &&
    js.includes("panel.setAttribute('inert', '')") &&
    js.includes("panel.removeAttribute('inert')") &&
    js.includes('if (!ids.includes(id)) return false;') &&
    js.includes('if (id === activeId) return false;'),
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
    activatePanelTriggerJs.includes('const nextId = resolveTriggerTarget(activeId, id, defaultPanel);') &&
    activatePanelTriggerJs.includes('triggerById.get(defaultPanel)?.focus({ preventScroll: true })') &&
    count(js, /activatePanelTrigger\(String\(trigger\.dataset\.homeAccordionTrigger \|\| ''\)\)/g) === 2,
  'overview tabs should retain their five-category selection and collapse-to-About behavior');
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
    js.includes('timelineScrollerById') &&
    js.includes("panel?.querySelector('[data-home-timeline-scroller]')") &&
    js.includes("window.matchMedia('(min-width: 960px) and (min-height: 620px)')") &&
    js.includes('scrollIntoView'),
  'accordion should preserve spacious-rail panel scroll positions and reveal newly opened stacked sections');
  assert(js.includes('decodeURIComponent(rawId)') &&
    js.includes('catch (error)') &&
    legacyLocationJs.includes("url.searchParams.get('view') !== 'library'") &&
    js.includes("projects: '/portfolio'") &&
    js.includes("tools: '/tools'") &&
    js.includes("games: '/games'") &&
    legacyLocationJs.includes('window.location.replace(') &&
    legacyLocationJs.includes("url.searchParams.delete('view')") &&
    legacyLocationJs.includes('window.history.replaceState(window.history.state') &&
    handleLocationChangeJs.includes('if (normalizeLegacyLibraryLocation()) return;') &&
    handleLocationChangeJs.includes('if (window.location.hash && !hashPanel) return;') &&
    handleLocationChangeJs.includes('const nextPanel = hashPanel || defaultPanel;') &&
    handleLocationChangeJs.includes("else updateLocation(nextPanel, 'replace');") &&
    handleLocationChangeJs.includes('selectPanel(nextPanel, { updateHistory: false, reveal: true })') &&
    js.includes("window.addEventListener('hashchange', handleLocationChange)") &&
    js.includes("window.addEventListener('popstate', handleLocationChange)"),
  'accordion should preserve hash history, redirect valid legacy library URLs, and strip invalid legacy query state without discarding the hash');
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
  assert(updateLocationJs.includes("url.searchParams.delete('view')") &&
    !updateLocationJs.includes("url.searchParams.set('view', 'library')") &&
    updateLocationJs.includes("mode === 'replace' ? 'replaceState' : 'pushState'") &&
    updateLocationJs.includes('{ homePanel: id }') &&
    js.includes("updateLocation(id, options.historyMode || 'push')") &&
    js.includes("updateLocation(id, 'replace')"),
  'explicit category selections should preserve a clean canonical hash and use push versus replace history deliberately');
  assert(!/LibraryMode|libraryOpen|libraryClose|libraryView|documentScrollPositions|startViewTransition/.test(js) &&
    updateTriggerStateJs.includes('selected && triggerId === defaultPanel') &&
    updateTriggerStateJs.includes("trigger.setAttribute('aria-disabled', 'true')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-disabled')") &&
    updateTriggerStateJs.includes("trigger.removeAttribute('aria-current')") &&
    js.includes("scrollTarget.setAttribute('tabindex', '0')") &&
    js.includes("region.removeAttribute('tabindex')") &&
    js.includes('scrollTarget.scrollHeight > scrollTarget.clientHeight + 1') &&
    js.includes('scrollTarget.scrollWidth > scrollTarget.clientWidth + 1'),
  'only active About should expose a noncollapsible state, and only truly overflowing vertical or horizontal regions should add a tab stop');
  assert(js.includes('const initialHashPanel = panelIdFromHash();') &&
    js.includes('const initialPanel = initialHashPanel || defaultPanel;') &&
    js.includes('if (normalizeLegacyLibraryLocation()) return;') &&
    js.includes("updateLocation(initialPanel, 'replace')") &&
    js.includes('revealPanelTrigger(initialHashPanel);'),
  'initial deep links should restore the requested overview category and canonicalize a bare homepage URL to About');
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
  assert(count(css, /var\(--personal-footer-block-size, 0px\)/g) === 3 &&
    css.includes('min-height: min(540px, calc(100svh') &&
    !css.includes('.footer.footer-classic'),
  'desktop homepage height should reserve the compact personal footer without hiding it');
  assert(activityEvents.includes("target.closest('[data-home-accordion]')") &&
    activityEvents.includes('category.dataset.homeAccordionTrigger'),
  'homepage category changes should remain analytics-visible');
};
