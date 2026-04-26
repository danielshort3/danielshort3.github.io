#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { loadSiteContentAsync } = require('./lib/content-loader');
const {
  renderAudienceConfigJs,
  renderFooter,
  renderFullPage,
  renderHeader,
  renderProjectsDataJs,
  renderToolsDirectoryBody
} = require('./lib/cms-renderers');
const { renderVisualPageBody } = require('../api/_lib/cms-widgets');

const root = path.resolve(__dirname, '..');

function write(relPath, contents) {
  const absPath = path.join(root, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, contents, 'utf8');
}

function buildManagedPages(content) {
  const pages = [];

  content.pages.forEach((page) => {
    pages.push(page);
  });

  content.audiences.forEach((audience) => {
    if (audience.page) {
      pages.push({
        ...audience.page,
        audienceKey: audience.key,
        id: audience.page.id || `audience-${audience.key}`
      });
    }
  });

  content.resumes.forEach((resume) => {
    if (resume.digitalPage) {
      pages.push({
        ...resume.digitalPage,
        audienceKey: resume.audience || resume.key,
        id: resume.digitalPage.id || `resume-${resume.key}`
      });
    }
    if (resume.pdfPage) {
      pages.push({
        ...resume.pdfPage,
        audienceKey: resume.audience || resume.key,
        id: resume.pdfPage.id || `resume-pdf-${resume.key}`
      });
    }
  });

  return pages;
}

function getAudienceLabel(content, audienceKey) {
  const audience = content.audiencesByKey[audienceKey];
  if (audience && audience.brandNavPrimary) return audience.brandNavPrimary;
  const defaultAudience = content.audiencesByKey[content.site.settings.defaultAudience];
  return defaultAudience && defaultAudience.brandNavPrimary
    ? defaultAudience.brandNavPrimary
    : 'Data Analytics';
}

async function main() {
  const content = await loadSiteContentAsync(root);
  const navigation = content.site.navigation || {};
  const headerHtml = renderHeader({
    settings: content.site.settings,
    navigation,
    projectsById: content.projectsById,
    audienceLabel: getAudienceLabel(content, content.site.settings.defaultAudience)
  });
  const footerHtml = renderFooter({
    footer: content.site.footer,
    year: new Date().getFullYear()
  });

  write(path.join('js', 'portfolio', 'projects-data.js'), renderProjectsDataJs(
    content.projects,
    Array.isArray(navigation.portfolio && navigation.portfolio.featuredProjectIds)
      ? navigation.portfolio.featuredProjectIds
      : []
  ));
  write(path.join('js', 'common', 'audience-config.js'), renderAudienceConfigJs(content.site.settings, content.audiences));
  write(path.join('build', 'templates', 'header.partial.html'), `${headerHtml}\n`);
  write(path.join('build', 'templates', 'footer.partial.html'), `${footerHtml}\n`);

  const managedPages = buildManagedPages(content);
  managedPages.forEach((page) => {
    let pageDef = page;
    if (page.template === 'tools-directory') {
      pageDef = {
        ...page,
        bodyHtml: renderToolsDirectoryBody(page, content.tools)
      };
    } else if (page.template === 'visual-page') {
      pageDef = {
        ...page,
        bodyHtml: renderVisualPageBody(page)
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
      page: pageDef,
      audienceLabel: getAudienceLabel(content, pageDef.audienceKey)
    });
    write(pageDef.outputPath, html);
  });

  process.stdout.write(`[cms] Generated ${managedPages.length} managed page(s) and shared content artifacts from content/.\n`);
}

main().catch((err) => {
  process.stderr.write(`[cms] ${err && err.message ? err.message : err}\n`);
  process.exitCode = 1;
});
