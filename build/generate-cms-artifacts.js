#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { loadSiteContentAsync } = require('./lib/content-loader');
const { versionedImageUrl, versionImageContent } = require('./lib/versioned-image-url');
const {
  buildGamesDirectoryWorkbenchData,
  buildToolsDirectoryWorkbenchData,
  renderAudienceConfigJs,
  renderDirectoryDataJs,
  renderFooter,
  renderFullPage,
  renderGamesDirectoryBody,
  renderHeader,
  renderProjectsDataJs,
  renderToolsDirectoryBody
} = require('./lib/cms-renderers');
const { renderVisualPageBody } = require('../api/_lib/cms-widgets');
const { getPersonalToolGroup } = require('./lib/personal-accordion-shell');

const root = path.resolve(__dirname, '..');

function write(relPath, contents) {
  const absPath = path.join(root, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, contents, 'utf8');
}

function normalizeHomeLibraryHref(value) {
  const normalized = String(value || '').trim();
  if (!normalized) return '';
  if (/^(?:[a-z]+:)?\/\//i.test(normalized)) return '';
  return normalized.startsWith('/') ? normalized : `/${normalized}`;
}

function normalizeHomeLibraryAsset(value) {
  const normalized = String(value || '').trim();
  if (!normalized) return '';
  if (/^(?:[a-z]+:)?\/\//i.test(normalized)) return normalized;
  return normalized.startsWith('/') ? normalized : `/${normalized}`;
}

function homeLibraryPreviewAsset(category, id) {
  const safeCategory = String(category || '').trim().replace(/[^a-z0-9-]/gi, '');
  const safeId = String(id || '').trim().replace(/[^a-z0-9-]/gi, '');
  if (!safeCategory || !safeId) return '';
  return `/img/home-previews/${safeCategory}/${safeId}.webp`;
}

function projectLibraryPreviewAsset(image) {
  const normalized = normalizeHomeLibraryAsset(image).replace(/[?#].*$/, '');
  if (!normalized) return '';
  const extension = path.posix.extname(normalized);
  const basename = extension ? normalized.slice(0, -extension.length) : normalized;
  return versionedImageUrl(`${basename}-640.webp`);
}

function homeLibraryItem({
  id,
  title,
  summary,
  href,
  image,
  imageAlt,
  iconImage,
  iconHtml,
  contentType,
  contentId,
  resourceType,
  group,
  badge
}) {
  return {
    id: String(id || '').trim(),
    title: String(title || 'Explore').trim(),
    summary: String(summary || '').trim(),
    href: normalizeHomeLibraryHref(href),
    image: normalizeHomeLibraryAsset(image),
    imageAlt: String(imageAlt || '').trim(),
    ...(iconImage ? { iconImage: versionedImageUrl(normalizeHomeLibraryAsset(iconImage)) } : {}),
    iconHtml: String(iconHtml || '').trim(),
    external: false,
    contentType: String(contentType || '').trim(),
    contentId: String(contentId || id || '').trim(),
    resourceType: String(resourceType || contentType || '').trim(),
    group: String(group || '').trim(),
    badge: String(badge || '').trim()
  };
}

function buildHomeLibraryData(content) {
  const personal = (content.audiences || []).find((audience) => audience.key === 'personal');
  const startHereIds = Array.isArray(personal?.projectLibrary?.startHereProjectIds)
    ? personal.projectLibrary.startHereProjectIds
    : [];
  const projectGroups = ['Start here', 'Machine learning', 'Data stories', 'Practical applications'];
  const projects = (Array.isArray(content.projects) ? content.projects : [])
    .filter((project) => project && project.id && project.published !== false)
    .map((project) => homeLibraryItem({
      id: String(project.id).trim(),
      title: String(project.title || 'Project').trim(),
      summary: String(project.subtitle || project.personalStory && project.personalStory.why || project.problem || '').trim(),
      href: `/portfolio/${encodeURIComponent(String(project.id).trim())}`,
      image: projectLibraryPreviewAsset(project.image),
      imageAlt: '',
      iconImage: project.iconImage,
      contentType: 'project',
      contentId: project.id,
      resourceType: 'case_study',
      group: startHereIds.includes(project.id) ? 'Start here' :
        (project.concepts || []).includes('Analytics') ? 'Data stories' :
          (project.concepts || []).includes('Machine Learning') ? 'Machine learning' : 'Practical applications',
      badge: ['iframe', 'tableau'].includes(project.embed?.type) ? 'Interactive project' : 'Case study'
    }))
    .sort((a, b) => projectGroups.indexOf(a.group) - projectGroups.indexOf(b.group) ||
      (a.group === 'Start here' ? startHereIds.indexOf(a.id) - startHereIds.indexOf(b.id) : 0));

  const toolsPage = content.pagesById && content.pagesById.tools;
  const startHereToolIds = Array.isArray(personal?.toolLibrary?.startHereToolIds)
    ? personal.toolLibrary.startHereToolIds
    : [];
  const toolGroups = ['Start here', 'Text', 'Images', 'Links', 'Recording'];
  const toolCategoryIds = new Map(content.tools.map((tool) => [tool.slug, tool.categoryId]));
  const toolsDirectory = toolsPage
    ? buildToolsDirectoryWorkbenchData(toolsPage, content.tools)
    : { items: [] };
  const tools = (Array.isArray(toolsDirectory.items) ? toolsDirectory.items : [])
    .filter((tool) => tool && tool.id && tool.href &&
      String(tool.visibility || 'public').trim().toLowerCase() === 'public' &&
      !tool.hidden && !tool.noindex)
    .map((tool) => homeLibraryItem({
      id: String(tool.id).trim(),
      title: String(tool.title || 'Tool').trim(),
      summary: String(tool.summary || '').trim(),
      href: tool.href,
      image: tool.iconImage,
      imageAlt: '',
      iconHtml: tool.iconHtml,
      contentType: 'tool',
      contentId: tool.id,
      resourceType: 'tool',
      group: startHereToolIds.includes(tool.id) ? 'Start here' :
        getPersonalToolGroup({ id: tool.id, categoryId: toolCategoryIds.get(tool.id) })
    }))
    .sort((a, b) => toolGroups.indexOf(a.group) - toolGroups.indexOf(b.group) ||
      (a.group === 'Start here' ? startHereToolIds.indexOf(a.id) - startHereToolIds.indexOf(b.id) : 0));

  const gamesPage = content.pagesById && content.pagesById.games;
  const gamesDirectory = gamesPage
    ? buildGamesDirectoryWorkbenchData(gamesPage)
    : { items: [] };
  const games = (Array.isArray(gamesDirectory.items) ? gamesDirectory.items : [])
    .filter((game) => game && game.id && game.href)
    .map((game) => homeLibraryItem({
      id: String(game.id).trim(),
      title: String(game.title || 'Game').trim(),
      summary: String(game.summary || '').trim(),
      href: game.href,
      image: homeLibraryPreviewAsset('games', game.id),
      imageAlt: '',
      iconImage: game.iconImage,
      iconHtml: game.iconHtml,
      contentType: 'game',
      contentId: game.id,
      resourceType: 'game'
    }));

  return {
    projects: { items: projects },
    tools: { items: tools },
    games: { items: games }
  };
}

function renderHomeLibraryDataJs(data) {
  return [
    '/* Generated by build/generate-cms-artifacts.js. Do not edit directly. */',
    '(function (root, factory) {',
    '  const data = factory();',
    "  if (typeof module === 'object' && module.exports) {",
    '    module.exports = data;',
    '  }',
    "  if (typeof window !== 'undefined') {",
    '    window.HOME_LIBRARY_DATA = data;',
    '  } else if (root) {',
    '    root.HOME_LIBRARY_DATA = data;',
    '  }',
    "})(typeof globalThis !== 'undefined' ? globalThis : this, function () {",
    "  'use strict';",
    '',
    `  return ${JSON.stringify(data, null, 2).replace(/^/gm, '  ').trimStart()};`,
    '});',
    ''
  ].join('\n');
}

function buildManagedPages(content) {
  const personal = content.audiencesByKey.personal;
  return [
    ...content.pages,
    ...(personal && personal.page ? [{ ...personal.page, audienceKey: 'personal' }] : [])
  ];
}

async function main() {
  const content = versionImageContent(await loadSiteContentAsync(root));
  const navigation = content.site.navigation || {};
  const personalAudience = content.audiencesByKey.personal || null;
  const headerHtml = renderHeader({
    settings: content.site.settings,
    navigation,
    projectsById: content.projectsById,
    pagesById: content.pagesById,
    tools: content.tools,
    audience: personalAudience,
    audienceLabel: personalAudience?.brandNavPrimary || 'Projects, Tools, and Games'
  });
  const footerHtml = renderFooter({
    footer: content.site.footer,
    year: new Date().getFullYear(),
    audience: personalAudience
  });

  write(path.join('js', 'portfolio', 'projects-data.js'), renderProjectsDataJs(
    content.projects,
    Array.isArray(navigation.portfolio && navigation.portfolio.featuredProjectIds)
      ? navigation.portfolio.featuredProjectIds
      : []
  ));
  if (content.pagesById && content.pagesById.tools) {
    write(
      path.join('js', 'portfolio', 'tools-directory-data.js'),
      renderDirectoryDataJs(buildToolsDirectoryWorkbenchData(content.pagesById.tools, content.tools))
    );
  }
  if (content.pagesById && content.pagesById.games) {
    write(
      path.join('js', 'portfolio', 'games-directory-data.js'),
      renderDirectoryDataJs(buildGamesDirectoryWorkbenchData(content.pagesById.games))
    );
  }
  write(
    path.join('js', 'home', 'home-library-data.js'),
    renderHomeLibraryDataJs(buildHomeLibraryData(content))
  );
  write(path.join('js', 'common', 'audience-config.js'), renderAudienceConfigJs(content.site.settings, content.audiences));
  write(path.join('build', 'templates', 'header.partial.html'), `${headerHtml}\n`);
  write(path.join('build', 'templates', 'footer.partial.html'), `${footerHtml}\n`);

  const managedPages = buildManagedPages(content);
  managedPages.forEach((page) => {
    let pageDef = page;
    if (pageDef.template === 'tools-directory') {
      pageDef = {
        ...pageDef,
        bodyHtml: renderToolsDirectoryBody(pageDef, content.tools)
      };
    } else if (pageDef.template === 'games-directory') {
      pageDef = {
        ...pageDef,
        bodyHtml: renderGamesDirectoryBody(pageDef)
      };
    } else if (pageDef.template === 'visual-page') {
      pageDef = {
        ...pageDef,
        bodyHtml: renderVisualPageBody(pageDef)
      };
    }

    if (!pageDef.outputPath) {
      throw new Error(`Managed page "${pageDef.id || pageDef.title || 'unknown'}" is missing outputPath`);
    }

    const html = renderFullPage({
      settings: content.site.settings,
      navigation: content.site.navigation,
      footer: content.site.footer,
      projectsById: content.projectsById,
      pagesById: content.pagesById,
      tools: content.tools,
      page: pageDef,
      audience: personalAudience,
      audienceLabel: personalAudience?.brandNavPrimary || 'Projects, Tools, and Games'
    });
    write(pageDef.outputPath, html);
  });

  process.stdout.write(`[cms] Generated ${managedPages.length} managed page(s) and shared content artifacts from content/.\n`);
}

if (require.main === module) {
  main().catch((err) => {
    process.stderr.write(`[cms] ${err && err.message ? err.message : err}\n`);
    process.exitCode = 1;
  });
}

module.exports = { buildHomeLibraryData };
