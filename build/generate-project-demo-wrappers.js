#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { wrapPersonalAccordionHtml } = require('./lib/personal-accordion-shell');
const {
  PROJECT_DEMO_IDS,
  getProjectDemoId,
  toCanonicalProjectDemoUrl,
  toRawProjectDemoUrl
} = require('./lib/project-demo-routes');

const root = path.resolve(__dirname, '..');
const contentDir = path.join(root, 'content', 'projects');
const rawDemoDir = path.join(root, 'demos');
const wrapperDir = path.join(root, 'pages', 'demos');
const headerTemplatePath = path.join(root, 'build', 'templates', 'header.partial.html');
const footerTemplatePath = path.join(root, 'build', 'templates', 'footer.partial.html');
const SITE_ORIGIN = 'https://www.danielshort.me';
const GUARD_START = '<!-- project-demo-wrapper-guard:start -->';
const GUARD_END = '<!-- project-demo-wrapper-guard:end -->';

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeRegExp(value) {
  return String(value || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function loadProjects() {
  return fs.readdirSync(contentDir)
    .filter((name) => name.endsWith('.json'))
    .sort()
    .map((name) => readJson(path.join(contentDir, name)));
}

function loadProjectDemoDefinitions() {
  const projects = loadProjects();
  const byDemoId = new Map();

  projects.forEach((project) => {
    const demoId = getProjectDemoId(project && project.embed && project.embed.url);
    if (!demoId) return;
    if (byDemoId.has(demoId)) {
      throw new Error(`Multiple projects reference ${demoId}.`);
    }
    byDemoId.set(demoId, project);
  });

  return PROJECT_DEMO_IDS.map((demoId) => {
    const project = byDemoId.get(demoId);
    if (!project) throw new Error(`No project content references ${demoId}.`);

    const rawFile = path.join(rawDemoDir, `${demoId}.html`);
    if (!fs.existsSync(rawFile)) throw new Error(`Missing raw project demo: demos/${demoId}.html`);

    const projectId = String(project.id || '').trim();
    const title = String(project.title || projectId || demoId).trim();
    const isPublished = project.published !== false && project.hidden !== true && project.noindex !== true;
    const canonicalPath = toCanonicalProjectDemoUrl(demoId);
    return Object.freeze({
      demoId,
      projectId,
      title,
      subtitle: String(project.subtitle || '').trim(),
      canonicalPath,
      canonicalUrl: `${SITE_ORIGIN}${canonicalPath}`,
      rawPath: toRawProjectDemoUrl(demoId),
      rawFile,
      wrapperFile: path.join(wrapperDir, `${demoId}.html`),
      backHref: isPublished ? `/portfolio/${encodeURIComponent(projectId)}` : '/?view=library#projects',
      backLabel: isPublished ? `Back to ${title}` : 'Back to project library',
      backCompactLabel: isPublished ? 'Project' : 'Library'
    });
  });
}

function renderRawDemoGuard(definition) {
  return [
    GUARD_START,
    '<script>',
    '  (() => {',
    '    if (window.self === window.top) {',
    `      window.location.replace(${JSON.stringify(definition.canonicalPath)} + window.location.search + window.location.hash);`,
    '    }',
    '  })();',
    '</script>',
    GUARD_END
  ].join('\n');
}

function injectRawDemoGuard(html, definition) {
  const source = String(html || '');
  const markerPattern = new RegExp(
    `${escapeRegExp(GUARD_START)}[\\s\\S]*?${escapeRegExp(GUARD_END)}(?:\\r?\\n)?`,
    'g'
  );
  const clean = source.replace(markerPattern, '');
  const headMatch = /<head\b[^>]*>/i.exec(clean);
  if (!headMatch) throw new Error(`Raw demo ${definition.demoId} is missing <head>.`);

  const lineBreak = clean.includes('\r\n') ? '\r\n' : '\n';
  const headEnd = headMatch.index + headMatch[0].length;
  const tail = clean.slice(headEnd).replace(/^\r?\n/, '');
  const guard = renderRawDemoGuard(definition).replace(/\n/g, lineBreak);
  return `${clean.slice(0, headEnd)}${lineBreak}${guard}${lineBreak}${tail}`;
}

function renderDemoWrapperPage(definition) {
  const header = fs.readFileSync(headerTemplatePath, 'utf8').trim();
  const footer = fs.readFileSync(footerTemplatePath, 'utf8')
    .replace(/__YEAR__/g, String(new Date().getFullYear()))
    .trim();
  const description = `${definition.title} interactive demo, presented inside Daniel Short's project library.`;
  const pageTitle = /\bdemo$/i.test(definition.title)
    ? definition.title
    : `${definition.title} Demo`;
  const frameId = `project-demo-frame-${definition.demoId}`;
  const basePage = `<!DOCTYPE html>
<html lang="en" class="no-js">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <base href="/">
  <title>${escapeHtml(pageTitle)} | Daniel Short</title>
  <link rel="canonical" href="${escapeHtml(definition.canonicalUrl)}">
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="robots" content="noindex, nofollow">
  <meta name="theme-color" content="#091F3B">
  <link rel="stylesheet" href="dist/styles.css">
  <link rel="stylesheet" href="dist/styles-personal-accordion.css">
  <link rel="icon" href="favicon.ico" sizes="any">
  <script src="js/common/no-js.js"></script>
</head>
<body class="project-demo-wrapper-page" data-page="project-demo">
  <a href="#main" class="skip-link">Skip to interactive demo</a>
${header}

  <main id="main" class="project-demo-wrapper-main" aria-label="${escapeHtml(definition.title)} interactive demo">
    <div class="project-demo-wrapper-frame">
      <iframe id="${escapeHtml(frameId)}" class="project-demo-wrapper-iframe" src="${escapeHtml(definition.rawPath)}" data-project-demo-src="${escapeHtml(definition.rawPath)}" title="${escapeHtml(definition.title)} interactive demo" loading="eager" allowfullscreen></iframe>
    </div>
  </main>

${footer}
  <script defer src="js/navigation/project-demo-wrapper.js"></script>
  <script defer src="js/common/common.js"></script>
  <script defer src="js/navigation/navigation.js"></script>
  <script defer src="js/animations/animations.js"></script>
  <script src="js/privacy/config.js"></script>
  <script defer src="js/privacy/consent_manager.js"></script>
</body>
</html>
`;

  return wrapPersonalAccordionHtml(basePage, {
    category: 'projects',
    itemId: definition.demoId,
    view: 'detail',
    fit: 'viewport',
    chrome: 'compact',
    backHref: definition.backHref,
    backLabel: definition.backLabel,
    backCompactLabel: definition.backCompactLabel,
    backAriaLabel: definition.backLabel
  });
}

function writeProjectDemoWrappers(definitions = loadProjectDemoDefinitions()) {
  fs.mkdirSync(wrapperDir, { recursive: true });
  let guarded = 0;
  let wrapped = 0;

  definitions.forEach((definition) => {
    const rawSource = fs.readFileSync(definition.rawFile, 'utf8');
    const guardedSource = injectRawDemoGuard(rawSource, definition);
    if (guardedSource !== rawSource) {
      fs.writeFileSync(definition.rawFile, guardedSource, 'utf8');
      guarded += 1;
    }

    const wrapperSource = renderDemoWrapperPage(definition);
    const previousWrapper = fs.existsSync(definition.wrapperFile)
      ? fs.readFileSync(definition.wrapperFile, 'utf8')
      : '';
    if (wrapperSource !== previousWrapper) {
      fs.writeFileSync(definition.wrapperFile, wrapperSource, 'utf8');
      wrapped += 1;
    }
  });

  return { definitions: definitions.length, guarded, wrapped };
}

function main() {
  const result = writeProjectDemoWrappers();
  process.stdout.write(
    `[project-demo-wrappers] Verified ${result.definitions} demo routes; updated ${result.wrapped} wrappers and ${result.guarded} raw guards.\n`
  );
}

if (require.main === module) main();

module.exports = {
  GUARD_END,
  GUARD_START,
  injectRawDemoGuard,
  loadProjectDemoDefinitions,
  renderDemoWrapperPage,
  renderRawDemoGuard,
  writeProjectDemoWrappers
};
