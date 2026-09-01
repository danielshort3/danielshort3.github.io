#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const {
  extractMainHtml,
  markProfessionalInternalHtml,
  renderPersonalLibraryMain,
  replaceMainHtml,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
} = require('./lib/personal-accordion-shell');

const root = path.resolve(__dirname, '..');
const pagesDir = path.join(root, 'pages');
const professionalDir = path.join(pagesDir, 'professional');
const portfolioDir = path.join(pagesDir, 'portfolio');
const homeLibraryDataPath = path.join(root, 'js', 'home', 'home-library-data.js');
const PROFESSIONAL_AUDIENCES = Object.freeze(['analytics', 'data-science', 'tourism']);

const TOOL_PAGE_IDS = Object.freeze([
  'text-compare',
  'nbsp-cleaner',
  'oxford-comma-checker',
  'point-of-view-checker',
  'word-frequency',
  'utm-batch-builder',
  'qr-code-generator',
  'image-optimizer',
  'background-remover',
  'screen-recorder'
]);

const GAME_PAGE_PATHS = Object.freeze({
  'stellar-dogfight': path.join('pages', 'games', 'stellar-dogfight.html'),
  roulette: path.join('pages', 'games', 'roulette.html'),
  'probability-engine': path.join('pages', 'games', 'probability-engine.html'),
  'project-starfall': path.join('pages', 'games', 'project-starfall.html'),
  stormbreak: path.join('pages', 'games', 'stormbreak.html'),
  'ocean-wave-simulation': path.join('pages', 'ocean-wave-simulation.html')
});

const LIBRARY_COPY = Object.freeze({
  projects: Object.freeze({
    title: 'Project library',
    description: 'Projects and experiments organized around the questions, systems, and practical problems behind them.'
  }),
  tools: Object.freeze({
    title: 'Tool library',
    description: 'Focused browser utilities for writing, campaign links, images, and recurring workflow tasks.'
  }),
  games: Object.freeze({
    title: 'Games and simulations',
    description: 'Playable experiments in probability, progression, feedback loops, and interactive systems.'
  })
});

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8');
}

function write(relPath, contents) {
  const absPath = path.join(root, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, contents, 'utf8');
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath));
}

function loadHomeLibraryData() {
  if (!fs.existsSync(homeLibraryDataPath)) {
    throw new Error('Missing js/home/home-library-data.js. Run CMS content generation first.');
  }
  delete require.cache[require.resolve(homeLibraryDataPath)];
  const data = require(homeLibraryDataPath);
  return data && typeof data === 'object' ? data : {};
}

function countMainElements(html) {
  return (String(html || '').match(/<main\b/gi) || []).length;
}

function validateWrappedPage(html, relPath, expectedCategory) {
  if (!html.includes('data-personal-accordion-shell')) {
    throw new Error(`${relPath} is missing the personal accordion shell.`);
  }
  if (!new RegExp(`data-personal-category="${expectedCategory}"`, 'i').test(html)) {
    throw new Error(`${relPath} has the wrong personal accordion category.`);
  }
  if (countMainElements(html) !== 1 || !/<main\b[^>]*\bid="main"/i.test(html)) {
    throw new Error(`${relPath} must contain exactly one <main id="main">.`);
  }
}

function writeWrapped(relPath, options) {
  const source = read(relPath);
  const wrapped = wrapPersonalAccordionHtml(source, options);
  validateWrappedPage(wrapped, relPath, options.category);
  write(relPath, wrapped);
  return wrapped;
}

function writeProfessionalCopy(relPath, html, audience) {
  const marked = markProfessionalInternalHtml(html, audience);
  if (marked.includes('data-personal-accordion-shell')) {
    throw new Error(`${relPath} professional copy must remain unwrapped.`);
  }
  write(relPath, marked);
}

function buildLibraryPage(sourceHtml, category, libraryData) {
  const copy = LIBRARY_COPY[category];
  const items = Array.isArray(libraryData && libraryData[category] && libraryData[category].items)
    ? libraryData[category].items
    : [];
  const mainHtml = renderPersonalLibraryMain({
    category,
    items,
    title: copy.title,
    description: copy.description
  });
  const withLibraryMain = replaceMainHtml(sourceHtml, mainHtml);
  return wrapPersonalAccordionHtml(withLibraryMain, {
    category,
    itemId: `${category}-library`,
    view: 'library',
    ...(category === 'projects' ? { fit: 'viewport', chrome: 'compact' } : {})
  });
}

function buildPortfolioIndex(libraryData) {
  const personalRelPath = path.join('pages', 'portfolio.html');
  const canonicalProfessionalRelPath = path.join('pages', 'professional', 'analytics', 'portfolio.html');
  const personalHtml = read(personalRelPath);
  const source = unwrapPersonalAccordionHtml(personalHtml);
  const sourceIsLibrary = /\bpersonal-library-main\b/i.test(source);
  let professionalMain = '';
  if (sourceIsLibrary) {
    if (!exists(canonicalProfessionalRelPath)) {
      throw new Error('Cannot restore the professional portfolio workbench from an already wrapped library without its generated snapshot. Run CMS content generation first.');
    }
    const snapshot = unwrapPersonalAccordionHtml(read(canonicalProfessionalRelPath));
    if (/\bpersonal-library-main\b/i.test(snapshot)) {
      throw new Error('The generated professional portfolio snapshot contains the personal library instead of the workbench. Run CMS content generation first.');
    }
    professionalMain = extractMainHtml(snapshot);
  }
  PROFESSIONAL_AUDIENCES.forEach((audience) => {
    const professionalRelPath = path.join('pages', 'professional', audience, 'portfolio.html');
    let professionalSource = source;
    if (sourceIsLibrary && exists(professionalRelPath)) {
      professionalSource = unwrapPersonalAccordionHtml(read(professionalRelPath));
    }
    if (professionalMain) professionalSource = replaceMainHtml(professionalSource, professionalMain);
    writeProfessionalCopy(professionalRelPath, professionalSource, audience);
  });
  const personal = buildLibraryPage(source, 'projects', libraryData);
  validateWrappedPage(personal, personalRelPath, 'projects');
  write(personalRelPath, personal);
}

function buildDirectoryIndex(relPath, category, libraryData) {
  const source = unwrapPersonalAccordionHtml(read(relPath));
  const personal = buildLibraryPage(source, category, libraryData);
  validateWrappedPage(personal, relPath, category);
  write(relPath, personal);
}

function buildContactPage() {
  const personalRelPath = path.join('pages', 'contact.html');
  const personalHtml = read(personalRelPath);
  const source = unwrapPersonalAccordionHtml(personalHtml);
  const wasWrapped = personalHtml.includes('data-personal-accordion-shell');
  PROFESSIONAL_AUDIENCES.forEach((audience) => {
    const professionalRelPath = path.join('pages', 'professional', audience, 'contact.html');
    const professionalSource = wasWrapped && exists(professionalRelPath)
      ? unwrapPersonalAccordionHtml(read(professionalRelPath))
      : source;
    writeProfessionalCopy(professionalRelPath, professionalSource, audience);
  });
  const personal = wrapPersonalAccordionHtml(source, {
    category: 'contact',
    itemId: 'contact',
    view: 'detail',
    backHref: '/#contact',
    backLabel: 'Back to Contact overview'
  });
  validateWrappedPage(personal, personalRelPath, 'contact');
  write(personalRelPath, personal);
}

function buildProjectPages() {
  if (!fs.existsSync(portfolioDir)) return 0;
  const projectFiles = fs.readdirSync(portfolioDir)
    .filter((fileName) => fileName.toLowerCase().endsWith('.html'))
    .sort((a, b) => a.localeCompare(b));

  projectFiles.forEach((fileName) => {
    const personalRelPath = path.join('pages', 'portfolio', fileName);
    const itemId = path.basename(fileName, '.html');
    const personalHtml = read(personalRelPath);
    const source = unwrapPersonalAccordionHtml(personalHtml);
    const wasWrapped = personalHtml.includes('data-personal-accordion-shell');
    PROFESSIONAL_AUDIENCES.forEach((audience) => {
      const professionalRelPath = path.join('pages', 'professional', audience, 'portfolio', fileName);
      const professionalSource = wasWrapped && exists(professionalRelPath)
        ? unwrapPersonalAccordionHtml(read(professionalRelPath))
        : source;
      writeProfessionalCopy(professionalRelPath, professionalSource, audience);
    });
    const personal = wrapPersonalAccordionHtml(source, {
      category: 'projects',
      itemId,
      view: 'detail',
      fit: 'viewport',
      chrome: 'compact',
      backHref: '/?view=library#projects',
      backLabel: 'Back to project library'
    });
    validateWrappedPage(personal, personalRelPath, 'projects');
    write(personalRelPath, personal);
  });
  return projectFiles.length;
}

function buildToolPages() {
  TOOL_PAGE_IDS.forEach((itemId) => {
    const relPath = path.join('pages', `${itemId}.html`);
    if (!exists(relPath)) throw new Error(`Missing public tool page: ${relPath}`);
    writeWrapped(relPath, {
      category: 'tools',
      itemId,
      view: 'detail',
      backHref: '/tools',
      backLabel: 'Back to tool library',
      includeToolChrome: true
    });
  });
  return TOOL_PAGE_IDS.length;
}

function buildGamePages() {
  Object.entries(GAME_PAGE_PATHS).forEach(([itemId, relPath]) => {
    if (!exists(relPath)) throw new Error(`Missing game page: ${relPath}`);
    writeWrapped(relPath, {
      category: 'games',
      itemId,
      view: 'detail',
      fit: ['stellar-dogfight', 'project-starfall'].includes(itemId) ? 'immersive' : 'document',
      backHref: '/games',
      backLabel: 'Back to games',
      includePageHero: itemId === 'ocean-wave-simulation',
      includeProbabilityShell: itemId === 'probability-engine',
      includeUntilScripts: ['probability-engine', 'roulette'].includes(itemId)
    });
  });
  return Object.keys(GAME_PAGE_PATHS).length;
}

function main() {
  const libraryData = loadHomeLibraryData();
  fs.mkdirSync(professionalDir, { recursive: true });

  buildPortfolioIndex(libraryData);
  buildDirectoryIndex(path.join('pages', 'tools.html'), 'tools', libraryData);
  buildDirectoryIndex(path.join('pages', 'games.html'), 'games', libraryData);
  buildContactPage();
  const projectCount = buildProjectPages();
  const toolCount = buildToolPages();
  const gameCount = buildGamePages();

  process.stdout.write(
    `[personal-accordion] Wrapped 4 personal indexes, ${projectCount} projects, ${toolCount} tools, and ${gameCount} games; three audience-specific portfolio/contact copies remain unwrapped.\n`
  );
}

if (require.main === module) main();

module.exports = {
  GAME_PAGE_PATHS,
  PROFESSIONAL_AUDIENCES,
  TOOL_PAGE_IDS,
  buildLibraryPage,
  buildPortfolioIndex,
  buildProjectPages,
  buildToolPages,
  buildGamePages,
  main
};
