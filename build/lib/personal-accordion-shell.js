'use strict';

const audienceApi = require('../../js/common/audience-config');

const PERSONAL_SHELL_START = '<!-- personal-accordion-shell:start -->';
const PERSONAL_SHELL_END = '<!-- personal-accordion-shell:end -->';
const PERSONAL_CONTENT_START = '<!-- personal-accordion-content:start -->';
const PERSONAL_CONTENT_END = '<!-- personal-accordion-content:end -->';
const PERSONAL_TOOL_HEADER_START = '<!-- personal-tool-header:start -->';
const PERSONAL_TOOL_HEADER_END = '<!-- personal-tool-header:end -->';
const SITE_ROUTE_MANIFEST_ID = 'site-route-manifest';
const SITE_ROUTE_MANIFEST_VERSION = 1;
const HARD_NAVIGATION_PATHS = Object.freeze([
  '/tools/background-remover',
  '/tools/job-application-tracker',
  '/tools/transcribe'
]);

const CATEGORY_CONFIG = Object.freeze({
  about: Object.freeze({
    label: 'About',
    color: '#091f3b',
    colorEnd: '#032b57',
    href: '/#about',
    icon: '<circle cx="12" cy="7" r="4"></circle><path d="M4.5 21c.7-4.1 3.2-6.2 7.5-6.2s6.8 2.1 7.5 6.2"></path>'
  }),
  projects: Object.freeze({
    label: 'Projects',
    color: '#155dfc',
    colorEnd: '#0b4bd4',
    href: '/#projects',
    libraryHref: '/portfolio',
    icon: '<path d="M3 7.5h7l2-2h9v14H3z"></path><path d="M3 9h18"></path>'
  }),
  tools: Object.freeze({
    label: 'Tools',
    color: '#087f8c',
    colorEnd: '#006973',
    href: '/#tools',
    libraryHref: '/tools',
    icon: '<path d="M14.7 6.1a5 5 0 0 0-6.8 6.8L3 17.8 6.2 21l4.9-4.9a5 5 0 0 0 6.8-6.8l-3.1 3.1-3.2-3.2z"></path>'
  }),
  games: Object.freeze({
    label: 'Games',
    color: '#c94b0a',
    colorEnd: '#e35d00',
    href: '/#games',
    libraryHref: '/games',
    icon: '<path d="M7.5 8h9a5 5 0 0 1 4.7 3.3l1.2 3.6a3.2 3.2 0 0 1-5.3 3.3L15 16H9l-2.1 2.2a3.2 3.2 0 0 1-5.3-3.3l1.2-3.6A5 5 0 0 1 7.5 8z"></path><path d="M7 11v4M5 13h4M16.5 12h.01M19 14h.01"></path>'
  }),
  resume: Object.freeze({
    label: 'Resume',
    color: '#087f8c',
    colorEnd: '#006973',
    href: '/resume-analytics',
    icon: '<path d="M6 3h8l4 4v14H6z"></path><path d="M14 3v5h5M9 12h6M9 16h6"></path>'
  }),
  contact: Object.freeze({
    label: 'Contact',
    color: '#334155',
    colorEnd: '#263648',
    href: '/#contact',
    libraryHref: '/contact',
    icon: '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"></path><path d="M8 9h8M8 13h5"></path>'
  })
});

const CATEGORY_ORDER = Object.freeze(['about', 'projects', 'tools', 'games', 'contact']);
const PROFESSIONAL_CATEGORY_ORDER = Object.freeze(['about', 'projects', 'resume', 'contact']);

function getShellAudience(value) {
  return ['analytics', 'data-science', 'tourism'].includes(value) ? value : 'personal';
}

function getShellCategory(categoryId, audience = 'personal') {
  const category = CATEGORY_CONFIG[categoryId];
  if (getShellAudience(audience) === 'personal') return category;
  const config = audienceApi.getAudience(audience);
  const href = {
    about: config.homePath,
    projects: config.portfolioPath,
    resume: config.resumePath,
    contact: config.contactPath
  }[categoryId] || config.homePath;
  return { ...category, href, libraryHref: href };
}
const LIBRARY_PRESENTATION = Object.freeze({
  projects: Object.freeze({
    title: 'Project library',
    summary: 'A collection of machine learning projects, practical tools, and playful experiments.'
  }),
  tools: Object.freeze({
    title: 'Tool library',
    summary: 'The complete collection of small utilities for text, links, media, and recurring workflows.'
  }),
  games: Object.freeze({
    title: 'Game library',
    summary: 'All of my browser games and simulations, from action RPG systems to probability experiments.'
  })
});
const ARROW_LEFT = '<path d="m15 18-6-6 6-6"></path>';
const ARROW_RIGHT = '<path d="m9 5 7 7-7 7"></path>';

function getTagAttribute(tag, name) {
  const escapedName = String(name || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const match = new RegExp(`\\s${escapedName}="([^"]*)"`, 'i').exec(String(tag || ''));
  return match ? match[1] : '';
}

function normalizeRoutePath(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw, 'https://www.danielshort.me');
    let pathname = url.pathname || '/';
    pathname = pathname.replace(/\/index\.html$/i, '/') || '/';
    pathname = pathname !== '/' ? pathname.replace(/\.html$/i, '').replace(/\/$/, '') : pathname;
    return pathname || '/';
  } catch (_) {
    return '';
  }
}

function isHardNavigationPath(value) {
  return HARD_NAVIGATION_PATHS.includes(normalizeRoutePath(value));
}

function getCanonicalRoutePath(html, fallbackPath = '') {
  const canonicalTag = (String(html || '').match(/<link\b[^>]*>/gi) || [])
    .find((tag) => /\srel="canonical"/i.test(tag));
  return normalizeRoutePath(getTagAttribute(canonicalTag, 'href')) || normalizeRoutePath(fallbackPath) || '/';
}

function normalizeResourceHref(value) {
  const raw = String(value || '').trim();
  if (!raw || raw.startsWith('data:') || raw.startsWith('blob:')) return raw;
  try {
    const url = new URL(raw, 'https://www.danielshort.me/');
    return url.origin === 'https://www.danielshort.me'
      ? `${url.pathname}${url.search}`
      : url.href;
  } catch (_) {
    return raw;
  }
}

function uniqueDocumentResources(html, tagPattern, attribute) {
  const seen = new Set();
  const resources = [];
  (String(html || '').match(tagPattern) || []).forEach((tag) => {
    const href = normalizeResourceHref(getTagAttribute(tag, attribute));
    if (!href || seen.has(href)) return;
    seen.add(href);
    resources.push(href);
  });
  return resources;
}

function extractRouteStyles(html) {
  return uniqueDocumentResources(
    html,
    /<link\b(?=[^>]*\srel="stylesheet")[^>]*>/gi,
    'href'
  );
}

function extractRouteScripts(html) {
  return uniqueDocumentResources(html, /<script\b(?=[^>]*\ssrc=")[^>]*>/gi, 'src');
}

function escapeJsonForHtml(value) {
  return JSON.stringify(value)
    .replace(/&/g, '\\u0026')
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

function removeSiteRouteManifest(html) {
  return String(html || '').replace(
    new RegExp(`[\\t ]*<script\\b[^>]*\\bid="${SITE_ROUTE_MANIFEST_ID}"[^>]*>[\\s\\S]*?<\\/script>\\s*`, 'gi'),
    ''
  );
}

function readBodyRouteMetadata(html) {
  const bodyTag = /<body\b[^>]*>/i.exec(String(html || ''))?.[0] || '';
  return {
    id: getTagAttribute(bodyTag, 'data-site-route-id'),
    category: getTagAttribute(bodyTag, 'data-site-route-category'),
    view: getTagAttribute(bodyTag, 'data-site-route-view'),
    navigation: getTagAttribute(bodyTag, 'data-site-route-navigation'),
    module: getTagAttribute(bodyTag, 'data-site-route-module')
  };
}

function getRouteModule(html, fallbackId) {
  const bodyTag = /<body\b[^>]*>/i.exec(String(html || ''))?.[0] || '';
  const page = getTagAttribute(bodyTag, 'data-page');
  if (page === 'contact') return 'contact:contact';
  if (page === 'search') return 'search:search';
  if (/\bdata-portfolio-workbench(?:[\s=>])/i.test(html)) return 'portfolio:workbench';
  if (page === 'project' || /^resume(?:-|$)/.test(page) ||
      ['analytics', 'data-science', 'tourism'].includes(page)) return 'page:content';
  return fallbackId;
}

function renderSiteRouteManifest(html, options = {}) {
  const bodyMetadata = readBodyRouteMetadata(html);
  const path = getCanonicalRoutePath(html, options.path);
  const id = String(options.id || bodyMetadata.id || '').trim();
  const category = String(options.category || bodyMetadata.category || '').trim();
  const view = String(options.view || bodyMetadata.view || '').trim();
  const navigation = isHardNavigationPath(path) || options.navigation === 'hard' || bodyMetadata.navigation === 'hard'
    ? 'hard'
    : 'soft';
  if (!id || !category || !view) {
    throw new Error('Personal route manifest requires id, category, and view metadata.');
  }
  const manifest = {
    version: SITE_ROUTE_MANIFEST_VERSION,
    id,
    path,
    category,
    view,
    navigation,
    styles: Array.isArray(options.styles) ? options.styles : extractRouteStyles(html),
    scripts: Array.isArray(options.scripts) ? options.scripts : extractRouteScripts(html),
    module: String(options.module || bodyMetadata.module || getRouteModule(html, id)).trim() || id
  };
  return `<script type="application/json" id="${SITE_ROUTE_MANIFEST_ID}" data-site-route-manifest data-version="${SITE_ROUTE_MANIFEST_VERSION}">${escapeJsonForHtml(manifest)}</script>`;
}

function upsertSiteRouteManifest(html, options = {}) {
  const source = removeSiteRouteManifest(html);
  const manifest = renderSiteRouteManifest(source, options);
  if (!/<\/head>/i.test(source)) {
    throw new Error('Personal route document is missing </head>.');
  }
  return source.replace(/<\/head>/i, `  ${manifest}\n</head>`);
}

function validatePersonalRouteDocument(html) {
  const source = String(html || '');
  const manifestTags = source.match(
    new RegExp(`<script\\b[^>]*\\bid="${SITE_ROUTE_MANIFEST_ID}"[^>]*>[\\s\\S]*?<\\/script>`, 'gi')
  ) || [];
  if (manifestTags.length !== 1) {
    throw new Error(`Personal route document must contain exactly one ${SITE_ROUTE_MANIFEST_ID} manifest.`);
  }

  const manifestMatch = new RegExp(
    `<script\\b[^>]*\\bid="${SITE_ROUTE_MANIFEST_ID}"[^>]*>([\\s\\S]*?)<\\/script>`,
    'i'
  ).exec(manifestTags[0]);
  let manifest;
  try {
    manifest = JSON.parse(manifestMatch?.[1] || '{}');
  } catch (error) {
    throw new Error(`Personal route document has invalid ${SITE_ROUTE_MANIFEST_ID} JSON: ${error.message}`);
  }
  if (manifest.navigation !== 'soft') return manifest;
  if (!String(manifest.module || '').trim()) {
    throw new Error(`Soft route ${manifest.path || manifest.id || '(unknown)'} is missing a lifecycle module.`);
  }

  const requiredShellHooks = [
    'data-site-shell-header',
    'data-site-shell-footer',
    'data-site-route-content',
    'data-site-route-progress',
    'data-site-route-announcer'
  ];
  requiredShellHooks.forEach((hook) => {
    if (!source.includes(hook)) {
      throw new Error(`Soft route ${manifest.path || manifest.id} is missing the persistent-shell hook ${hook}.`);
    }
  });

  const listedScripts = new Set((Array.isArray(manifest.scripts) ? manifest.scripts : [])
    .map(normalizeResourceHref));
  const scriptTags = source.match(/<script\b[^>]*>[\s\S]*?<\/script>/gi) || [];
  scriptTags.forEach((tag) => {
    if (new RegExp(`\\bid="${SITE_ROUTE_MANIFEST_ID}"`, 'i').test(tag)) return;
    const src = normalizeResourceHref(getTagAttribute(tag, 'src'));
    if (src) {
      if (!listedScripts.has(src)) {
        throw new Error(`Soft route ${manifest.path || manifest.id} has an unclassified script: ${src}.`);
      }
      return;
    }

    const type = getTagAttribute(tag, 'type').trim().toLowerCase();
    const executable = !type || type === 'module' || /^(?:text|application)\/(?:java|ecma)script$/.test(type);
    if (!executable) return;
    const content = tag.replace(/^<script\b[^>]*>|<\/script>$/gi, '').trim();
    if (!content) return;
    if (getTagAttribute(tag, 'id') === 'ds-sw-register') return;
    throw new Error(`Soft route ${manifest.path || manifest.id} has an unclassified inline executable script.`);
  });
  return manifest;
}

function markHardNavigationLinks(html, forceAll = false, clearPrevious = false) {
  return String(html || '').replace(/<a\b[^>]*>/gi, (tag) => {
    const href = getTagAttribute(tag, 'href');
    return forceAll || isHardNavigationPath(href)
      ? setTagAttribute(tag, 'data-navigation', 'hard')
      : clearPrevious ? removeTagAttribute(tag, 'data-navigation') : tag;
  });
}

function markPersistentShellChrome(html) {
  let output = String(html || '').replace(
    /<header\b[^>]*\bid="combined-header-nav"[^>]*>/i,
    (tag) => setTagAttribute(tag, 'data-site-shell-header', true)
  );
  output = output.replace(
    /<footer\b[^>]*\bfooter--personal-compact\b[^>]*>/i,
    (tag) => setTagAttribute(tag, 'data-site-shell-footer', true)
  );
  return output;
}

function finalizePersonalRouteDocument(html, options = {}) {
  let output = String(html || '');
  const isHomepage = options.home === true;
  const bodyMetadata = readBodyRouteMetadata(output);
  if (!isHomepage && (!bodyMetadata.id || !bodyMetadata.category || !bodyMetadata.view)) return output;

  const routeOptions = isHomepage
    ? { id: 'home', category: 'about', view: 'overview', navigation: 'soft', path: '/', module: 'home' }
    : options;
  const routePath = getCanonicalRoutePath(output, routeOptions.path);
  const navigation = isHardNavigationPath(routePath) || routeOptions.navigation === 'hard' || (!isHomepage && bodyMetadata.navigation === 'hard') ? 'hard' : 'soft';
  output = output.replace(/<body\b[^>]*>/i, (bodyTag) => {
    let next = setTagAttribute(bodyTag, 'data-site-route-id', routeOptions.id || bodyMetadata.id);
    next = setTagAttribute(next, 'data-site-route-category', routeOptions.category || bodyMetadata.category);
    next = setTagAttribute(next, 'data-site-route-view', routeOptions.view || bodyMetadata.view);
    next = setTagAttribute(next, 'data-site-route-navigation', navigation);
    next = setTagAttribute(next, 'data-site-route-module', routeOptions.module || bodyMetadata.module || getRouteModule(output, routeOptions.id || bodyMetadata.id));
    return next;
  });
  output = markPersistentShellChrome(markHardNavigationLinks(output, navigation === 'hard'));
  output = upsertSiteRouteManifest(output, routeOptions);
  return output;
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeCategory(value) {
  const key = String(value || '').trim().toLowerCase();
  if (!CATEGORY_CONFIG[key]) throw new Error(`Unknown personal accordion category: ${value}`);
  return key;
}

function getPersonalLibraryPresentation(categoryValue, itemCount = 0) {
  const categoryId = normalizeCategory(categoryValue);
  const category = CATEGORY_CONFIG[categoryId];
  const library = LIBRARY_PRESENTATION[categoryId] || {};
  const count = Math.max(0, Number.parseInt(itemCount, 10) || 0);
  const singularLabel = categoryId.replace(/s$/, '');
  const countNoun = count === 1 ? singularLabel : categoryId;
  return Object.freeze({
    categoryId,
    title: String(library.title || `${category.label} library`).trim(),
    summary: String(library.summary || '').trim(),
    count,
    countNoun,
    countLabel: `${count} ${countNoun}`,
    backHref: category.href,
    backLabel: 'Back to homepage',
    backCompactLabel: 'Home',
    backAriaLabel: 'Back to homepage'
  });
}

function renderPersonalLibraryHeader(options = {}) {
  const presentation = getPersonalLibraryPresentation(options.category, options.itemCount);
  const containerTag = options.containerTag === 'header' ? 'header' : 'div';
  const headingTag = options.headingTag === 'h3' ? 'h3' : 'h1';
  const headingId = String(options.headingId || `personal-library-title-${presentation.categoryId}`).trim();
  const headingAttributes = [
    `id="${escapeHtml(headingId)}"`,
    options.headingFocusable ? 'data-home-library-heading tabindex="-1"' : ''
  ].filter(Boolean).join(' ');
  const headingClass = ['home-library__heading', options.wrapper ? 'wrapper' : ''].filter(Boolean).join(' ');
  const countMarkup = options.dynamicCount
    ? `<span data-home-library-count>${presentation.count}</span> ${escapeHtml(presentation.countNoun)}`
    : escapeHtml(presentation.countLabel);
  const countClass = [
    'personal-library__meta',
    options.countVisuallyHidden ? 'visually-hidden' : ''
  ].filter(Boolean).join(' ');
  const back = options.includeBack
    ? `  <button class="home-library__back" type="button" data-home-library-close="${escapeHtml(presentation.categoryId)}"><span aria-hidden="true">${renderIcon(ARROW_LEFT)}</span>${escapeHtml(presentation.backLabel)}</button>`
    : '';

  return [
    `<${containerTag} class="home-library__header">`,
    back,
    `  <div class="${headingClass}">`,
    `    <${headingTag} ${headingAttributes}>${escapeHtml(presentation.title)}</${headingTag}>`,
    presentation.summary ? `    <p>${escapeHtml(presentation.summary)}</p>` : '',
    `    <p class="${countClass}">${countMarkup}</p>`,
    '  </div>',
    `</${containerTag}>`
  ].filter(Boolean).join('\n');
}

function renderIcon(paths, className = '') {
  return `<svg${className ? ` class="${escapeHtml(className)}"` : ''} viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>`;
}

function renderToolsAccountBar() {
  return '<div class="tools-account-bar" data-tools-account="bar" data-personal-tool-account-bar="true"></div>';
}

function renderToolsAccountDock(className = '') {
  const classes = ['tools-account-dock', className].filter(Boolean).join(' ');
  return [
    `<div class="${escapeHtml(classes)}" data-tools-account="dock" data-personal-tool-account="true">`,
    '  <div class="tools-account-dock-inner personal-tool-header__account-inner" data-tools-account="dock-inner">',
    renderToolsAccountBar().split('\n').map((line) => `    ${line}`).join('\n'),
    '  </div>',
    '</div>'
  ].join('\n');
}

function findElementRangeByClass(html, className, beforeIndex = String(html || '').length) {
  const source = String(html || '');
  const classPattern = String(className || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const openingPattern = new RegExp(`<([a-z][a-z0-9:-]*)\\b[^>]*\\sclass="[^"]*\\b${classPattern}\\b[^"]*"[^>]*>`, 'gi');
  let opening = null;
  let match;
  while ((match = openingPattern.exec(source))) {
    if (match.index >= beforeIndex) break;
    opening = { index: match.index, end: openingPattern.lastIndex, tagName: match[1] };
  }
  if (!opening) return null;

  const tagPattern = new RegExp(`<\\/?${opening.tagName}\\b[^>]*>`, 'gi');
  tagPattern.lastIndex = opening.index;
  let depth = 0;
  while ((match = tagPattern.exec(source))) {
    if (/^<\//.test(match[0])) {
      depth -= 1;
      if (depth === 0) return { start: opening.index, end: tagPattern.lastIndex };
    } else if (!/\/>$/.test(match[0])) {
      depth += 1;
    }
  }
  return null;
}

function hydrateToolsAccountBar(html) {
  const source = String(html || '');
  if (!source.includes('data-tools-account="bar"')) return source;
  return source.replace(/<div\b[^>]*data-tools-account="bar"[^>]*>/i, (tag) => (
    setTagAttribute(tag, 'data-personal-tool-account-bar', 'true')
  ));
}

function demoteMainH1(html) {
  return String(html || '').replace(/<h1\b([^>]*)>([\s\S]*?)<\/h1>/gi, (match, attributes, contents) => {
    const markedAttributes = /\bdata-personal-tool-content-title\b/i.test(attributes)
      ? attributes
      : `${attributes} data-personal-tool-content-title`;
    return `<h2${markedAttributes}>${contents}</h2>`;
  });
}

function renderPersonalToolHeader(options = {}) {
  const itemId = String(options.itemId || 'tool').trim() || 'tool';
  const title = String(options.title || itemId).trim() || itemId;
  const summary = String(options.summary || '').trim();
  const extraHtml = hydrateToolsAccountBar(options.extraHtml).replace(
    /<div\b[^>]*data-tools-account="dock"[^>]*>/i,
    (tag) => setTagAttribute(tag, 'data-personal-tool-account', 'true')
  );
  const hasEmbeddedAccount = /data-tools-account="dock"/i.test(extraHtml);
  const includeAccount = options.includeAccount !== false;
  const headerClasses = [
    'personal-tool-header',
    'tools-hero',
    extraHtml ? 'personal-tool-header--with-actions' : '',
    extraHtml && /\bshortlinks-command-actions\b/i.test(extraHtml) ? 'shortlinks-command-header' : ''
  ].filter(Boolean).join(' ');

  return [
    PERSONAL_TOOL_HEADER_START,
    `<header class="${headerClasses}" data-personal-tool-header="${escapeHtml(itemId)}">`,
    '  <div class="wrapper personal-tool-header__inner">',
    '    <div class="personal-tool-header__copy">',
    `      <h1 id="personal-tool-title-${escapeHtml(itemId)}">${escapeHtml(title)}</h1>`,
    summary ? `      <p class="personal-tool-header__summary">${escapeHtml(summary)}</p>` : '',
    '    </div>',
    includeAccount && !hasEmbeddedAccount
      ? renderToolsAccountDock('personal-tool-header__account').split('\n').map((line) => `    ${line}`).join('\n')
      : '',
    extraHtml ? `    <div class="personal-tool-header__actions">${extraHtml}</div>` : '',
    '  </div>',
    '</header>',
    PERSONAL_TOOL_HEADER_END
  ].filter(Boolean).join('\n');
}

function preparePersonalToolDetailHtml(html, options = {}) {
  const source = unwrapPersonalAccordionHtml(html);
  const main = findMainRange(source);
  const beforeMain = source.slice(0, main.start);
  let chromeStart = main.start;
  let extraHtml = '';

  const generatedHeaderStart = beforeMain.lastIndexOf(PERSONAL_TOOL_HEADER_START);
  if (generatedHeaderStart !== -1) {
    const generatedHeaderEnd = beforeMain.indexOf(PERSONAL_TOOL_HEADER_END, generatedHeaderStart);
    if (generatedHeaderEnd === -1) throw new Error('Personal tool header markers are incomplete.');
    const generatedHeader = beforeMain.slice(
      generatedHeaderStart,
      generatedHeaderEnd + PERSONAL_TOOL_HEADER_END.length
    );
    const actionsRange = findElementRangeByClass(generatedHeader, 'shortlinks-command-actions');
    if (actionsRange) extraHtml = generatedHeader.slice(actionsRange.start, actionsRange.end);
    chromeStart = generatedHeaderStart;
  } else {
    const legacyHeroRange = findElementRangeByClass(beforeMain, 'tools-hero');
    if (legacyHeroRange) {
      const actionsRange = findElementRangeByClass(beforeMain, 'shortlinks-command-actions', legacyHeroRange.end);
      if (actionsRange && actionsRange.start >= legacyHeroRange.start && actionsRange.end <= legacyHeroRange.end) {
        extraHtml = beforeMain.slice(actionsRange.start, actionsRange.end);
      }
      chromeStart = legacyHeroRange.start;
    }
  }

  const mainHtml = demoteMainH1(source.slice(main.start, main.end));
  const header = renderPersonalToolHeader({ ...options, extraHtml });
  const suffix = source.slice(main.end);
  const boundary = suffix && !/^\r?\n/.test(suffix) ? '\n' : '';
  return `${source.slice(0, chromeStart)}${header}\n${mainHtml}${boundary}${suffix}`;
}

function renderPersonalRails(activeCategory, audience = 'personal') {
  const active = normalizeCategory(activeCategory);
  const isProfessional = getShellAudience(audience) !== 'personal';
  const order = isProfessional ? PROFESSIONAL_CATEGORY_ORDER : CATEGORY_ORDER;
  const rails = order.map((categoryId) => {
    const category = getShellCategory(categoryId, audience);
    const isActive = categoryId === active;
    const stateAttributes = isProfessional
      ? (isActive ? 'aria-current="page" data-personal-rail-active="true" data-site-tab-active="true"' : '')
      : isActive
      ? 'aria-current="page" data-personal-rail-active="true" data-site-tab-active="true" data-personal-transition="collapse"'
      : 'hidden inert aria-hidden="true" tabindex="-1"';
    const label = isProfessional ? category.label : `Return to the ${category.label} section on the homepage`;
    return [
      `  <a class="personal-accordion__rail personal-accordion__rail--${categoryId}${isActive ? ' is-active' : ''}" href="${escapeHtml(category.href)}" aria-label="${escapeHtml(label)}" style="--rail-color: ${category.color}; --rail-color-end: ${category.colorEnd};" data-site-tab="${categoryId}" data-site-tab-category="${categoryId}" ${stateAttributes}>`,
      `    <span class="personal-accordion__rail-icon" aria-hidden="true">${renderIcon(category.icon)}</span>`,
      `    <span class="personal-accordion__rail-label">${category.label}</span>`,
      '    <span class="personal-accordion__rail-notch" aria-hidden="true"></span>',
      '  </a>'
    ].join('\n');
  });
  return [
    `<div class="personal-accordion__rails"${isProfessional ? ' role="navigation" aria-label="Site sections"' : ''} data-personal-category-marker="${active}" data-site-tab-rail data-site-tab-rail-mode="${isProfessional ? 'navigation' : 'expanded'}">`,
    rails.join('\n'),
    '</div>'
  ].join('\n');
}

function setTagAttribute(tag, name, value) {
  const escapedName = String(name || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  const attrPattern = new RegExp(`\\s${escapedName}(?:="[^"]*")?`, 'i');
  const attribute = value === true ? ` ${name}` : ` ${name}="${escapeHtml(value)}"`;
  if (attrPattern.test(tag)) return tag.replace(attrPattern, attribute);
  return tag.replace(/>$/, `${attribute}>`);
}

function removeTagAttribute(tag, name) {
  const escapedName = String(name || '').replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  return tag.replace(new RegExp(`\\s${escapedName}(?:="[^"]*")?`, 'gi'), '');
}

function addBodyClass(html, className) {
  return String(html || '').replace(/<body\b[^>]*>/i, (bodyTag) => {
    const classMatch = /\sclass="([^"]*)"/i.exec(bodyTag);
    const classes = new Set(String(classMatch ? classMatch[1] : '').split(/\s+/).filter(Boolean));
    classes.add(className);
    return setTagAttribute(bodyTag, 'class', [...classes].join(' '));
  });
}

function removeBodyClass(html, className) {
  return String(html || '').replace(/<body\b[^>]*>/i, (bodyTag) => {
    const classMatch = /\sclass="([^"]*)"/i.exec(bodyTag);
    if (!classMatch) return bodyTag;
    const classes = classMatch[1].split(/\s+/).filter((token) => token && token !== className);
    return classes.length
      ? setTagAttribute(bodyTag, 'class', classes.join(' '))
      : removeTagAttribute(bodyTag, 'class');
  });
}

function setBodyAttributes(html, attributes) {
  let output = addBodyClass(html, 'personal-accordion-page');
  return output.replace(/<body\b[^>]*>/i, (bodyTag) => Object.entries(attributes).reduce(
    (next, [name, value]) => setTagAttribute(next, name, value),
    bodyTag
  ));
}

function removePersonalBodyAttributes(html) {
  let output = removeBodyClass(html, 'personal-accordion-page');
  return output.replace(/<body\b[^>]*>/i, (bodyTag) => {
    let next = [
      'data-personal-accordion-view',
      'data-personal-category',
      'data-personal-item',
      'data-personal-fit',
      'data-personal-chrome',
      'data-site-route-id',
      'data-site-route-category',
      'data-site-route-view',
      'data-site-route-navigation',
      'data-site-route-module'
    ].reduce((tag, name) => removeTagAttribute(tag, name), bodyTag);
    if (/\sdata-audience="personal"/i.test(next)) next = removeTagAttribute(next, 'data-audience');
    return next;
  });
}

function stripPersonalStylesheet(html) {
  return String(html || '').replace(/^\s*<link\b[^>]*href="(?:\/?dist\/)?styles-personal-accordion(?:\.[0-9a-f]{8})?\.css"[^>]*>\s*$/gim, '');
}

function stripLegacyProjectPager(html) {
  return String(html || '').replace(
    /<nav\b[^>]*class="[^"]*\bproject-pager\b[^"]*"[^>]*>[\s\S]*?<\/nav>\s*/gi,
    ''
  );
}

function normalizeSkipLinkHrefs(html) {
  const source = String(html || '');
  const canonicalTag = (source.match(/<link\b[^>]*>/gi) || [])
    .find((tag) => /\srel="canonical"/i.test(tag));
  const canonicalHref = canonicalTag && /\shref="([^"]+)"/i.exec(canonicalTag)?.[1];
  if (!canonicalHref) return source;

  let pathname = '';
  try {
    const url = new URL(canonicalHref.replace(/&amp;/g, '&'), 'https://www.danielshort.me');
    pathname = `${url.pathname || '/'}${url.search}`;
  } catch (_) {
    return source;
  }
  const localMainHref = `${pathname}#main`;

  return source.replace(/<a\b[^>]*>/gi, (tag) => {
    if (!/\sclass="[^"]*\bskip-link\b[^"]*"/i.test(tag)) return tag;
    const hrefMatch = /\shref="([^"]*)"/i.exec(tag);
    if (!hrefMatch || !/#main$/i.test(hrefMatch[1])) return tag;
    return setTagAttribute(tag, 'href', localMainHref);
  });
}

function unwrapPersonalAccordionHtml(html) {
  let output = removeSiteRouteManifest(String(html || ''));
  const isProjectWrapper = output.includes(PERSONAL_SHELL_START) &&
    /<body\b[^>]*\bdata-personal-category="projects"/i.test(output);
  const shellStart = output.indexOf(PERSONAL_SHELL_START);
  const shellEnd = output.indexOf(PERSONAL_SHELL_END);
  if (shellStart !== -1 && shellEnd > shellStart) {
    const contentStart = output.indexOf(PERSONAL_CONTENT_START, shellStart);
    const contentEnd = output.indexOf(PERSONAL_CONTENT_END, contentStart + PERSONAL_CONTENT_START.length);
    if (contentStart === -1 || contentEnd === -1 || contentEnd > shellEnd) {
      throw new Error('Personal accordion shell markers are incomplete.');
    }
    const fragment = output.slice(contentStart + PERSONAL_CONTENT_START.length, contentEnd)
      .replace(/^\s*\r?\n/, '')
      .replace(/\r?\n\s*$/, '');
    output = output.slice(0, shellStart) + fragment + output.slice(shellEnd + PERSONAL_SHELL_END.length);
  } else if (shellStart !== -1) {
    // Recover a partially generated shell if a legacy footer replacement
    // removed the closing markers. This keeps the build self-healing while
    // still failing closed for any other incomplete marker shape.
    const contentStart = output.indexOf(PERSONAL_CONTENT_START, shellStart);
    const footerStart = output.indexOf('<footer', contentStart + PERSONAL_CONTENT_START.length);
    const scriptStart = footerStart === -1 ? -1 : output.indexOf('<script', footerStart);
    if (contentStart === -1 || footerStart === -1 || scriptStart === -1) {
      throw new Error('Personal accordion shell markers are incomplete.');
    }
    const fragment = output.slice(contentStart + PERSONAL_CONTENT_START.length, footerStart)
      .replace(/^\s*\r?\n/, '')
      .replace(/\r?\n\s*$/, '');
    output = `${output.slice(0, shellStart)}${fragment}\n${output.slice(scriptStart)}`;
  }
  if (isProjectWrapper) output = stripLegacyProjectPager(output);
  return removePersonalBodyAttributes(output);
}

function findMainRange(html) {
  const openMatch = /<main\b[^>]*\bid="main"[^>]*>/i.exec(html);
  if (!openMatch) throw new Error('Personal accordion target is missing <main id="main">.');
  const closeIndex = html.indexOf('</main>', openMatch.index + openMatch[0].length);
  if (closeIndex === -1) throw new Error('Personal accordion target has an unclosed <main id="main">.');
  return { start: openMatch.index, end: closeIndex + '</main>'.length };
}

function extractMainHtml(html) {
  const cleanHtml = unwrapPersonalAccordionHtml(html);
  const main = findMainRange(cleanHtml);
  return cleanHtml.slice(main.start, main.end);
}

function findFragmentRange(html, options = {}) {
  const main = findMainRange(html);
  let start = main.start;
  let end = main.end;

  if (options.includePersonalToolHeader) {
    const beforeMain = html.slice(0, main.start);
    const headerStart = beforeMain.lastIndexOf(PERSONAL_TOOL_HEADER_START);
    const headerEnd = headerStart === -1
      ? -1
      : beforeMain.indexOf(PERSONAL_TOOL_HEADER_END, headerStart);
    if (headerStart === -1 || headerEnd === -1) {
      throw new Error('Personal tool detail is missing its generated compact header.');
    }
    start = headerStart;
  }

  if (options.includeToolChrome) {
    const beforeMain = html.slice(0, main.start);
    const heroMatches = [...beforeMain.matchAll(/<section\b[^>]*class="[^"]*\btools-hero\b[^"]*"[^>]*>/gi)];
    if (heroMatches.length) start = heroMatches[heroMatches.length - 1].index;
  }

  if (options.includePageHero) {
    const beforeMain = html.slice(0, main.start);
    const heroMatches = [...beforeMain.matchAll(/<section\b[^>]*class="[^"]*\bhero\b[^"]*"[^>]*>/gi)];
    if (heroMatches.length) start = heroMatches[heroMatches.length - 1].index;
  }

  if (options.includeProbabilityShell) {
    const beforeMain = html.slice(0, main.start);
    const appShellMatches = [...beforeMain.matchAll(/<div\b[^>]*class="[^"]*\bapp-shell\b[^"]*"[^>]*>/gi)];
    if (appShellMatches.length) start = appShellMatches[appShellMatches.length - 1].index;
  }

  if (options.includeUntilScripts) {
    const scriptIndex = html.indexOf('<script', main.end);
    const footerIndex = html.indexOf('<footer', main.end);
    const boundaries = [scriptIndex, footerIndex].filter((index) => index !== -1);
    if (boundaries.length) {
      end = Math.min(...boundaries);
      while (end > main.end && /[\r\n\t ]/.test(html[end - 1])) end -= 1;
    }
  }

  while (start > 0 && (html[start - 1] === ' ' || html[start - 1] === '\t')) start -= 1;
  return { start, end };
}

function renderPersonalAccordionShell(fragment, options = {}) {
  const categoryId = normalizeCategory(options.category);
  const category = getShellCategory(categoryId, options.audience);
  const isLibrary = options.view === 'library';
  const itemId = String(options.itemId || categoryId).trim() || categoryId;
  const fit = String(options.fit || 'document').trim() || 'document';
  const backLabel = String(options.backLabel || `Back to ${category.label}`).trim();
  const backCompactLabel = String(options.backCompactLabel || (isLibrary ? 'Categories' : 'Library')).trim();
  const backAriaLabel = String(options.backAriaLabel || backLabel).trim();
  const backHref = String(options.backHref || category.libraryHref || category.href).trim();
  const toolbar = [
    '<div class="personal-accordion__toolbar" data-site-route-toolbar>',
    `  <a class="personal-accordion__back" href="${escapeHtml(backHref)}" aria-label="${escapeHtml(backAriaLabel)}">`,
    `    <span class="personal-accordion__back-icon" aria-hidden="true">${renderIcon(ARROW_LEFT)}</span>`,
    `    <span class="personal-accordion__back-label personal-accordion__back-label--desktop" aria-hidden="true">${escapeHtml(backLabel)}</span>`,
    `    <span class="personal-accordion__back-label personal-accordion__back-label--mobile" aria-hidden="true">${escapeHtml(backCompactLabel)}</span>`,
    '  </a>',
    '  <div class="personal-accordion__context" aria-hidden="true">',
    `    <span class="personal-accordion__context-icon">${renderIcon(category.icon)}</span>`,
    `    <span class="personal-accordion__context-label">${escapeHtml(category.label)}</span>`,
    '  </div>',
    '</div>'
  ].join('\n');

  return [
    PERSONAL_SHELL_START,
    '  <div class="site-route-progress" data-site-route-progress hidden aria-hidden="true"><span class="site-route-progress__bar"></span></div>',
    `<section class="personal-accordion personal-accordion--${escapeHtml(categoryId)} personal-accordion--${isLibrary ? 'library' : 'detail'}" data-personal-accordion-shell data-personal-active-category="${escapeHtml(categoryId)}" data-site-route-content style="--panel-color: ${category.color}; --panel-color-end: ${category.colorEnd};">`,
    '  <div class="personal-accordion__shell">',
    renderPersonalRails(categoryId, options.audience).split('\n').map((line) => `    ${line}`).join('\n'),
    `    <div class="personal-accordion__panel" data-personal-detail-panel="${escapeHtml(categoryId)}">`,
    toolbar ? toolbar.split('\n').map((line) => `      ${line}`).join('\n') : '',
    '      <div class="personal-accordion__content" data-personal-detail-content>',
    `        ${PERSONAL_CONTENT_START}`,
    String(fragment || '').trim(),
    `        ${PERSONAL_CONTENT_END}`,
    '      </div>',
    '    </div>',
    '  </div>',
    '</section>',
    '  <p class="visually-hidden" role="status" aria-live="polite" aria-atomic="true" data-site-route-announcer></p>',
    PERSONAL_SHELL_END
  ].filter(Boolean).join('\n');
}

function wrapPersonalAccordionHtml(html, options = {}) {
  const category = normalizeCategory(options.category);
  const audience = getShellAudience(options.audience);
  const unwrappedHtml = unwrapPersonalAccordionHtml(html);
  const cleanHtml = category === 'projects'
    ? stripLegacyProjectPager(unwrappedHtml)
    : unwrappedHtml;
  const range = findFragmentRange(cleanHtml, options);
  const fragment = cleanHtml.slice(range.start, range.end);
  const shell = renderPersonalAccordionShell(fragment, options);
  const bodyAttributes = {
    'data-audience': audience,
    'data-personal-accordion-view': options.view === 'library' ? 'library' : 'detail',
    'data-personal-category': category,
    'data-personal-item': String(options.itemId || category).trim() || category,
    'data-personal-fit': String(options.fit || 'document').trim() || 'document',
    'data-site-route-id': `${audience === 'personal' ? '' : `${audience}:`}${category}:${String(options.itemId || category).trim() || category}`,
    'data-site-route-category': category,
    'data-site-route-view': options.view === 'library' ? 'library' : 'detail'
  };
  const canonicalPath = getCanonicalRoutePath(cleanHtml);
  bodyAttributes['data-site-route-navigation'] = isHardNavigationPath(canonicalPath) || options.navigation === 'hard'
    ? 'hard'
    : 'soft';
  const chrome = String(options.chrome || '').trim();
  if (chrome) bodyAttributes['data-personal-chrome'] = chrome;
  const suffix = cleanHtml.slice(range.end);
  const boundary = suffix && !/^\r?\n/.test(suffix) ? '\n' : '';
  const output = cleanHtml.slice(0, range.start) + shell + boundary + suffix;
  bodyAttributes['data-site-route-module'] = options.module || getRouteModule(cleanHtml, bodyAttributes['data-site-route-id']);
  let normalized = normalizeSkipLinkHrefs(setBodyAttributes(
    markHardNavigationLinks(output, false, audience !== 'personal' && bodyAttributes['data-site-route-navigation'] === 'soft'),
    bodyAttributes
  ));
  if (audience !== 'personal') {
    const referrerTag = '<meta name="referrer" content="no-referrer">';
    let referrerWritten = false;
    normalized = normalized.replace(/<meta\b[^>]*\bname=["']referrer["'][^>]*>/gi, () => {
      if (referrerWritten) return '';
      referrerWritten = true;
      return referrerTag;
    });
    if (!referrerWritten) normalized = normalized.replace(/<\/head>/i, `  ${referrerTag}\n</head>`);
  }
  return finalizePersonalRouteDocument(normalized, {
    id: bodyAttributes['data-site-route-id'],
    path: canonicalPath,
    category,
    view: bodyAttributes['data-site-route-view'],
    navigation: bodyAttributes['data-site-route-navigation'],
    module: bodyAttributes['data-site-route-module']
  });
}

function replaceMainHtml(html, mainHtml) {
  const cleanHtml = unwrapPersonalAccordionHtml(html);
  const main = findMainRange(cleanHtml);
  const suffix = cleanHtml.slice(main.end);
  const boundary = suffix && !/^\r?\n/.test(suffix) ? '\n' : '';
  return cleanHtml.slice(0, main.start) + String(mainHtml || '').trim() + boundary + suffix;
}

function renderLibraryCard(item, categoryId) {
  const href = String(item && item.href || '').trim();
  const title = String(item && item.title || 'Explore').trim();
  const summary = String(item && item.summary || '').trim();
  const image = String(item && item.image || '').trim();
  const imageAlt = String(item && item.imageAlt || '').trim();
  const iconHtml = String(item && item.iconHtml || '').trim();
  const media = image
    ? `<img src="${escapeHtml(image)}" alt="${escapeHtml(imageAlt)}" loading="lazy" decoding="async">`
    : (iconHtml || `<span class="personal-library__initial" aria-hidden="true">${escapeHtml(title.charAt(0) || '?')}</span>`);
  const mediaType = image ? 'image' : 'glyph';
  const contentType = String(item && item.contentType || categoryId.replace(/s$/, '')).trim();
  const contentId = String(item && item.contentId || item && item.id || '').trim();
  const resourceType = String(item && item.resourceType || contentType).trim();
  return [
    '      <li class="home-library__item">',
    `        <a class="home-library__card" href="${escapeHtml(href)}" data-content-open="true" data-content-id="${escapeHtml(contentId)}" data-content-type="${escapeHtml(contentType)}" data-resource-type="${escapeHtml(resourceType)}" data-source-surface="personal_library_page">`,
    `          <span class="home-library__media home-library__media--${mediaType}" aria-hidden="${imageAlt ? 'false' : 'true'}">${media}</span>`,
    '          <span class="home-library__copy">',
    `            <strong>${escapeHtml(title)}</strong>`,
    summary ? `            <span>${escapeHtml(summary)}</span>` : '',
    '          </span>',
    `          <span class="home-library__arrow" aria-hidden="true">${renderIcon(ARROW_RIGHT)}</span>`,
    '        </a>',
    '      </li>'
  ].filter(Boolean).join('\n');
}

function renderPersonalLibraryMain(options = {}) {
  const categoryId = normalizeCategory(options.category);
  const category = CATEGORY_CONFIG[categoryId];
  const items = Array.isArray(options.items) ? options.items.filter((item) => item && item.href) : [];
  const header = renderPersonalLibraryHeader({
    category: categoryId,
    itemCount: items.length,
    containerTag: 'div',
    headingTag: 'h1',
    wrapper: true
  });
  const toolsDock = categoryId === 'tools'
    ? renderToolsAccountDock('tools-account-dock--directory personal-library__account')
    : '';
  const libraryClasses = [
    'home-library',
    'personal-library',
    `personal-library--${categoryId}`
  ].join(' ');

  return [
    `<main id="main" class="personal-library-main personal-library-main--${categoryId}">`,
    toolsDock ? toolsDock.split('\n').map((line) => `  ${line}`).join('\n') : '',
    `  <section class="${libraryClasses}" aria-labelledby="personal-library-title-${categoryId}">`,
    header.split('\n').map((line) => `    ${line}`).join('\n'),
    `    <ul class="home-library__list wrapper" aria-label="${escapeHtml(category.label)}">`,
    items.map((item) => renderLibraryCard(item, categoryId)).join('\n'),
    '    </ul>',
    '  </section>',
    '</main>'
  ].filter(Boolean).join('\n');
}

function markProfessionalInternalHtml(html, audience = 'analytics') {
  const audienceKey = String(audience || 'analytics').trim() || 'analytics';
  let output = stripPersonalStylesheet(unwrapPersonalAccordionHtml(html));
  output = output.replace(/<body\b[^>]*>/i, (bodyTag) => {
    let next = setTagAttribute(bodyTag, 'data-internal-professional-copy', 'true');
    next = setTagAttribute(next, 'data-audience', audienceKey);
    return next;
  });
  output = output.replace(/<a\b[^>]*>/gi, (tag) => {
    const href = getTagAttribute(tag, 'href');
    if (!href || /^(?:#|mailto:|tel:)/i.test(href)) return tag;
    try {
      const url = new URL(href.replace(/&amp;/g, '&'), 'https://www.danielshort.me/');
      if (url.origin !== 'https://www.danielshort.me' || !/^\/(?:portfolio(?:\/[^/]+)?|contact|search)(?:\.html)?\/?$/i.test(url.pathname)) return tag;
      url.searchParams.set('audience', audienceKey);
      return setTagAttribute(tag, 'href', `${url.pathname}${url.search}${url.hash}`);
    } catch (_) {
      return tag;
    }
  });
  output = output.replace(/<link\b[^>]*\brel="canonical"[^>]*>/i, (tag) => {
    const hrefMatch = /\shref="([^"]+)"/i.exec(tag);
    if (!hrefMatch) return tag;
    try {
      const url = new URL(hrefMatch[1], 'https://www.danielshort.me');
      url.searchParams.set('audience', audienceKey);
      return setTagAttribute(tag, 'href', url.href);
    } catch (_) {
      return tag;
    }
  });
  output = output.replace(/<meta\b[^>]*\bproperty="og:url"[^>]*>/i, (tag) => {
    const contentMatch = /\scontent="([^"]+)"/i.exec(tag);
    if (!contentMatch) return tag;
    try {
      const url = new URL(contentMatch[1], 'https://www.danielshort.me');
      url.searchParams.set('audience', audienceKey);
      return setTagAttribute(tag, 'content', url.href);
    } catch (_) {
      return tag;
    }
  });
  const robotsTag = '<meta name="robots" content="noindex, nofollow">';
  if (/<meta\b[^>]*\bname="robots"[^>]*>/i.test(output)) {
    return output.replace(/<meta\b[^>]*\bname="robots"[^>]*>/i, robotsTag);
  }
  return output.replace(/<\/head>/i, `  ${robotsTag}\n</head>`);
}

module.exports = {
  CATEGORY_CONFIG,
  CATEGORY_ORDER,
  HARD_NAVIGATION_PATHS,
  LIBRARY_PRESENTATION,
  PERSONAL_CONTENT_END,
  PERSONAL_CONTENT_START,
  PERSONAL_SHELL_END,
  PERSONAL_SHELL_START,
  PERSONAL_TOOL_HEADER_END,
  PERSONAL_TOOL_HEADER_START,
  SITE_ROUTE_MANIFEST_ID,
  SITE_ROUTE_MANIFEST_VERSION,
  extractRouteScripts,
  extractRouteStyles,
  extractMainHtml,
  finalizePersonalRouteDocument,
  findFragmentRange,
  findMainRange,
  getPersonalLibraryPresentation,
  markProfessionalInternalHtml,
  markHardNavigationLinks,
  normalizeSkipLinkHrefs,
  preparePersonalToolDetailHtml,
  renderPersonalAccordionShell,
  renderPersonalLibraryHeader,
  renderPersonalLibraryMain,
  renderPersonalRails,
  renderSiteRouteManifest,
  renderPersonalToolHeader,
  renderToolsAccountBar,
  renderToolsAccountDock,
  replaceMainHtml,
  upsertSiteRouteManifest,
  validatePersonalRouteDocument,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
};
