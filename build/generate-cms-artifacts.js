#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { loadSiteContentAsync } = require('./lib/content-loader');
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
    : 'Projects, Tools, and Games';
}

function applyAudienceLinks(page, content) {
  const audienceKey = page.audienceKey
    || (page.bodyAttributes && page.bodyAttributes['data-audience'])
    || '';
  const audience = content.audiencesByKey[audienceKey];
  if (!audience || audience.key === 'personal') return page;

  const resumePath = String(audience.resumePath || '').replace(/^\/+/, '');
  const resumePreviewPath = String(audience.resumePreviewPath || '').replace(/^\/+/, '');
  const resumeDownloadPath = String(audience.resumeDownloadPath || '').replace(/^\/+/, '');
  const portfolioPath = String(audience.portfolioPath || '/portfolio').replace(/^\/+/, '');
  const contactPath = String(audience.contactPath || `/contact?audience=${audience.key}`).replace(/^\/+/, '');
  const updateHtml = (value) => {
    let html = String(value || '')
      .replace(/href="(\/?portfolio\/[^"?#]+)"/g, (match, path) => (
        `href="${String(path).replace(/^\/+/, '')}?audience=${encodeURIComponent(audience.key)}"`
      ))
      .replaceAll('href="/portfolio"', `href="/${portfolioPath}"`)
      .replaceAll('href="portfolio"', `href="${portfolioPath}"`)
      .replaceAll('href="/contact#contact-modal"', `href="/${contactPath}#contact-modal"`)
      .replaceAll('href="contact#contact-modal"', `href="${contactPath}#contact-modal"`);

    if (audience.key === 'analytics') {
      html = html
        .replaceAll('href="/resume-pdf"', `href="/${resumePreviewPath}"`)
        .replaceAll('href="resume-pdf"', `href="${resumePreviewPath}"`)
        .replaceAll('href="/resume"', `href="/${resumePath}"`)
        .replaceAll('href="resume"', `href="${resumePath}"`)
        .replaceAll('documents/Resume.pdf', resumeDownloadPath);
    }
    return html;
  };

  return {
    ...page,
    ...(page.bodyHtml ? { bodyHtml: updateHtml(page.bodyHtml) } : {}),
    sections: Array.isArray(page.sections)
      ? page.sections.map((section) => ({
          ...section,
          props: section && section.props && typeof section.props.html === 'string'
            ? { ...section.props, html: updateHtml(section.props.html) }
            : section.props
        }))
      : page.sections
  };
}

async function main() {
  const content = await loadSiteContentAsync(root);
  const navigation = content.site.navigation || {};
  const headerHtml = renderHeader({
    settings: content.site.settings,
    navigation,
    projectsById: content.projectsById,
    pagesById: content.pagesById,
    tools: content.tools,
    audience: content.audiencesByKey[content.site.settings.defaultAudience]
      || content.audiencesByKey.personal
      || null,
    audienceLabel: getAudienceLabel(content, content.site.settings.defaultAudience)
  });
  const personalAudience = content.audiencesByKey[content.site.settings.defaultAudience]
    || content.audiencesByKey.personal
    || null;
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
  write(path.join('js', 'common', 'audience-config.js'), renderAudienceConfigJs(content.site.settings, content.audiences));
  write(path.join('build', 'templates', 'header.partial.html'), `${headerHtml}\n`);
  content.audiences.forEach((audience) => {
    if (audience.key === content.site.settings.defaultAudience) return;
    write(
      path.join('build', 'templates', `header.${audience.key}.partial.html`),
      `${renderHeader({
        settings: content.site.settings,
        navigation,
        projectsById: content.projectsById,
        pagesById: content.pagesById,
        tools: content.tools,
        audience,
        audienceLabel: getAudienceLabel(content, audience.key)
      })}\n`
    );
  });
  write(path.join('build', 'templates', 'footer.partial.html'), `${footerHtml}\n`);
  content.audiences.forEach((audience) => {
    if (audience.key === content.site.settings.defaultAudience) return;
    write(
      path.join('build', 'templates', `footer.${audience.key}.partial.html`),
      `${renderFooter({ footer: content.site.footer, year: new Date().getFullYear(), audience })}\n`
    );
  });

  const managedPages = buildManagedPages(content);
  managedPages.forEach((page) => {
    let pageDef = applyAudienceLinks(page, content);
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
      audience: content.audiencesByKey[pageDef.audienceKey
        || (pageDef.bodyAttributes && pageDef.bodyAttributes['data-audience'])
        || content.site.settings.defaultAudience],
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
