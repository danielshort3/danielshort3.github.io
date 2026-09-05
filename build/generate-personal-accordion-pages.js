#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const {
  HARD_NAVIGATION_PATHS,
  extractMainHtml,
  getPersonalLibraryPresentation,
  markProfessionalInternalHtml,
  preparePersonalToolDetailHtml,
  renderPersonalLibraryMain,
  replaceMainHtml,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
} = require('./lib/personal-accordion-shell');

const root = path.resolve(__dirname, '..');
const pagesDir = path.join(root, 'pages');
const professionalDir = path.join(pagesDir, 'professional');
const portfolioDir = path.join(pagesDir, 'portfolio');
const toolsContentDir = path.join(root, 'content', 'tools');
const homeLibraryDataPath = path.join(root, 'js', 'home', 'home-library-data.js');
const PROFESSIONAL_AUDIENCES = Object.freeze(['analytics', 'data-science', 'tourism']);
const HARD_TOOL_PAGE_IDS = Object.freeze(HARD_NAVIGATION_PATHS.map((routePath) => (
  String(routePath || '').split('/').filter(Boolean).pop()
)));

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

const INTERNAL_TOOL_PAGE_IDS = Object.freeze([
  'job-application-tracker',
  'short-links',
  'campaign-creative-tracker',
  'ga4-utm-performance',
  'transcribe',
  'tools-dashboard',
  'job-application-copilot',
  'job-application-copilot-privacy'
]);

const ALL_TOOL_PAGE_IDS = Object.freeze([
  ...TOOL_PAGE_IDS,
  ...INTERNAL_TOOL_PAGE_IDS
]);

const UTILITY_PAGE_CONFIGS = Object.freeze([
  Object.freeze({
    relPath: path.join('pages', 'privacy.html'),
    itemId: 'privacy',
    category: 'about',
    backHref: '/#about'
  }),
  Object.freeze({
    relPath: path.join('pages', 'search.html'),
    itemId: 'search',
    category: 'tools',
    backHref: '/#tools'
  }),
  Object.freeze({
    relPath: path.join('pages', 'sitemap.html'),
    itemId: 'sitemap',
    category: 'about',
    backHref: '/#about'
  }),
  Object.freeze({
    relPath: path.join('pages', 'sitemap-pretty.html'),
    itemId: 'sitemap-pretty',
    category: 'about',
    backHref: '/#about'
  }),
  Object.freeze({
    relPath: path.join('pages', 'solutions.html'),
    itemId: 'solutions',
    category: 'projects',
    backHref: '/#projects'
  }),
  Object.freeze({
    relPath: 'dshort.html',
    itemId: 'dshort',
    category: 'about',
    backHref: '/'
  }),
  Object.freeze({
    relPath: '404.html',
    itemId: 'not-found',
    category: 'about',
    backHref: '/'
  })
]);

const GAME_PAGE_PATHS = Object.freeze({
  'stellar-dogfight': path.join('pages', 'games', 'stellar-dogfight.html'),
  roulette: path.join('pages', 'games', 'roulette.html'),
  'probability-engine': path.join('pages', 'games', 'probability-engine.html'),
  stormbreak: path.join('pages', 'games', 'stormbreak.html'),
  'ocean-wave-simulation': path.join('pages', 'ocean-wave-simulation.html')
});

const TOOL_DETAIL_METADATA = Object.freeze({
  'tools-dashboard': Object.freeze({
    title: 'Tools Dashboard',
    summary: 'Sign in once and manage your saved tool sessions across danielshort.me/tools.',
    includeAccount: true
  }),
  'job-application-copilot': Object.freeze({
    title: 'Job Application Copilot',
    summary: 'Set up a local-first Chrome extension for grounded job-application answers, explicit review, and controlled field filling.',
    includeAccount: false
  }),
  'job-application-copilot-privacy': Object.freeze({
    title: 'Job Application Copilot Privacy Policy',
    summary: 'How Job Application Copilot handles local evidence, application-page structure, Ollama requests, encrypted storage, and optional tracker transfers.',
    includeAccount: false
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

function decodeHtmlText(value) {
  return String(value || '')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&#x27;/gi, "'")
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&amp;/gi, '&');
}

function getHtmlAttribute(html, attribute) {
  const escaped = String(attribute || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  return decodeHtmlText(new RegExp(`\\s${escaped}="([^"]*)"`, 'i').exec(String(html || ''))?.[1] || '');
}

function getToolDetailMetadata(itemId, sourceHtml) {
  const id = String(itemId || '').trim();
  const explicit = TOOL_DETAIL_METADATA[id] || {};
  const metadataPath = path.join(toolsContentDir, `${id}.json`);
  let record = {};
  if (fs.existsSync(metadataPath)) {
    record = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
  }
  const bodyTag = /<body\b[^>]*>/i.exec(String(sourceHtml || ''))?.[0] || '';
  const titleTag = /<title>([\s\S]*?)<\/title>/i.exec(String(sourceHtml || ''))?.[1] || '';
  const descriptionTag = (String(sourceHtml || '').match(/<meta\b[^>]*\bname="description"[^>]*>/i) || [])[0] || '';
  const fallbackTitle = decodeHtmlText(titleTag).split('|')[0].trim() || id
    .split('-')
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ');
  const title = String(explicit.title || record.title || getHtmlAttribute(bodyTag, 'data-tools-title') || fallbackTitle).trim();
  const summary = String(
    explicit.summary ||
    record.summary ||
    getHtmlAttribute(descriptionTag, 'content') ||
    getHtmlAttribute(bodyTag, 'data-tools-eyebrow') ||
    `${title} browser tool.`
  ).trim();
  const includeAccount = typeof explicit.includeAccount === 'boolean'
    ? explicit.includeAccount
    : /data-tools-account="(?:dock|bar)"|site-tools-account/i.test(String(sourceHtml || ''));
  return Object.freeze({ itemId: id, title, summary, includeAccount });
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
  const itemId = path.basename(relPath, '.html');
  const category = itemId === 'contact' ? 'contact' : itemId === 'search' ? 'about' : 'projects';
  const backHref = category === 'projects' && itemId !== 'portfolio'
    ? `/portfolio?audience=${audience}`
    : `/${audience}`;
  const wrapped = wrapPersonalAccordionHtml(marked, {
    audience,
    category,
    itemId,
    navigation: 'soft',
    chrome: 'compact',
    fit: 'document',
    backHref,
    backLabel: category === 'projects' && itemId !== 'portfolio' ? 'Back to projects' : 'Back to about',
    backCompactLabel: category === 'projects' && itemId !== 'portfolio' ? 'Projects' : 'About'
  });
  validateWrappedPage(wrapped, relPath, category);
  write(relPath, wrapped);
}

function buildLibraryPage(sourceHtml, category, libraryData) {
  const items = Array.isArray(libraryData && libraryData[category] && libraryData[category].items)
    ? libraryData[category].items
    : [];
  const presentation = getPersonalLibraryPresentation(category, items.length);
  const mainHtml = renderPersonalLibraryMain({
    category,
    items
  });
  const withLibraryMain = replaceMainHtml(sourceHtml, mainHtml);
  return wrapPersonalAccordionHtml(withLibraryMain, {
    category,
    itemId: `${category}-library`,
    view: 'library',
    backHref: presentation.backHref,
    backLabel: presentation.backLabel,
    backCompactLabel: presentation.backCompactLabel,
    backAriaLabel: presentation.backAriaLabel,
    fit: 'viewport',
    chrome: 'compact'
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
    fit: 'viewport',
    chrome: 'compact',
    backHref: '/#contact',
    backLabel: 'Back to categories',
    backCompactLabel: 'Categories',
    backAriaLabel: 'Back to categories'
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
      backHref: '/portfolio',
      backLabel: 'Back to project library',
      backCompactLabel: 'Library',
      backAriaLabel: 'Back to project library'
    });
    validateWrappedPage(personal, personalRelPath, 'projects');
    write(personalRelPath, personal);
  });
  return projectFiles.length;
}

function buildToolPages() {
  ALL_TOOL_PAGE_IDS.forEach((itemId) => {
    const relPath = path.join('pages', `${itemId}.html`);
    if (!exists(relPath)) throw new Error(`Missing personal tool page: ${relPath}`);
    const source = read(relPath);
    const metadata = getToolDetailMetadata(itemId, source);
    const prepared = preparePersonalToolDetailHtml(source, metadata);
    const wrapped = wrapPersonalAccordionHtml(prepared, {
      category: 'tools',
      itemId,
      view: 'detail',
      navigation: HARD_TOOL_PAGE_IDS.includes(itemId) ? 'hard' : 'soft',
      fit: 'viewport',
      chrome: 'compact',
      backHref: '/tools',
      backLabel: 'Back to tool library',
      backCompactLabel: 'Library',
      backAriaLabel: 'Back to tool library',
      includePersonalToolHeader: true
    });
    validateWrappedPage(wrapped, relPath, 'tools');
    write(relPath, wrapped);
  });
  return ALL_TOOL_PAGE_IDS.length;
}

function buildUtilityPages() {
  UTILITY_PAGE_CONFIGS.forEach((config) => {
    if (!exists(config.relPath)) throw new Error(`Missing personal utility page: ${config.relPath}`);
    writeWrapped(config.relPath, {
      category: config.category,
      itemId: config.itemId,
      view: 'detail',
      fit: 'viewport',
      chrome: 'compact',
      backHref: config.backHref,
      backLabel: 'Back to categories',
      backCompactLabel: 'Categories',
      backAriaLabel: 'Back to categories'
    });
  });
  return UTILITY_PAGE_CONFIGS.length;
}

function buildGamePages() {
  Object.entries(GAME_PAGE_PATHS).forEach(([itemId, relPath]) => {
    if (!exists(relPath)) throw new Error(`Missing game page: ${relPath}`);
    const isImmersive = itemId === 'stellar-dogfight';
    writeWrapped(relPath, {
      category: 'games',
      itemId,
      view: 'detail',
      fit: isImmersive ? 'immersive' : 'viewport',
      chrome: 'compact',
      backHref: '/games',
      backLabel: 'Back to game library',
      backCompactLabel: 'Library',
      backAriaLabel: 'Back to game library',
      includePageHero: itemId === 'ocean-wave-simulation',
      includeProbabilityShell: itemId === 'probability-engine',
      includeUntilScripts: ['probability-engine', 'roulette'].includes(itemId)
    });
  });
  return Object.keys(GAME_PAGE_PATHS).length;
}

function buildProfessionalPages() {
  PROFESSIONAL_AUDIENCES.forEach((audience) => {
    const pages = [
      { itemId: audience, category: 'about' },
      { itemId: `resume-${audience}`, category: 'resume' },
      { itemId: `resume-${audience}-pdf`, category: 'resume' }
    ];
    if (audience === 'analytics') {
      pages.push({ itemId: 'resume', category: 'resume' }, { itemId: 'resume-pdf', category: 'resume' });
    }
    pages.forEach(({ itemId, category }) => writeWrapped(path.join('pages', `${itemId}.html`), {
      audience,
      category,
      itemId,
      navigation: 'soft',
      chrome: 'compact',
      fit: 'document',
      backHref: itemId.endsWith('-pdf') ? `/resume-${audience}` : `/${audience}`,
      backLabel: itemId.endsWith('-pdf') ? 'Back to resume' : 'Back to about',
      backCompactLabel: itemId.endsWith('-pdf') ? 'Resume' : 'About'
    }));
    writeProfessionalCopy(path.join('pages', 'professional', audience, 'search.html'), read(path.join('pages', 'search.html')), audience);
  });
}

function main() {
  const libraryData = loadHomeLibraryData();
  fs.mkdirSync(professionalDir, { recursive: true });

  buildPortfolioIndex(libraryData);
  buildDirectoryIndex(path.join('pages', 'tools.html'), 'tools', libraryData);
  buildDirectoryIndex(path.join('pages', 'games.html'), 'games', libraryData);
  buildContactPage();
  const utilityCount = buildUtilityPages();
  const projectCount = buildProjectPages();
  const toolCount = buildToolPages();
  const gameCount = buildGamePages();
  buildProfessionalPages();

  process.stdout.write(
    `[personal-accordion] Wrapped 4 personal category roots, ${utilityCount} utility/fallback pages, ${projectCount} projects, ${toolCount} tools, ${gameCount} games, and all professional landing, project, contact, search, and resume pages.\n`
  );
}

if (require.main === module) main();

module.exports = {
  ALL_TOOL_PAGE_IDS,
  GAME_PAGE_PATHS,
  HARD_TOOL_PAGE_IDS,
  INTERNAL_TOOL_PAGE_IDS,
  PROFESSIONAL_AUDIENCES,
  TOOL_DETAIL_METADATA,
  TOOL_PAGE_IDS,
  UTILITY_PAGE_CONFIGS,
  buildLibraryPage,
  buildPortfolioIndex,
  buildProjectPages,
  buildProfessionalPages,
  buildToolPages,
  buildUtilityPages,
  buildGamePages,
  getToolDetailMetadata,
  main
};
