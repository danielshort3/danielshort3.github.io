'use strict';

const PERSONAL_SHELL_START = '<!-- personal-accordion-shell:start -->';
const PERSONAL_SHELL_END = '<!-- personal-accordion-shell:end -->';
const PERSONAL_CONTENT_START = '<!-- personal-accordion-content:start -->';
const PERSONAL_CONTENT_END = '<!-- personal-accordion-content:end -->';

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
const ARROW_LEFT = '<path d="m15 18-6-6 6-6"></path>';
const ARROW_RIGHT = '<path d="m9 5 7 7-7 7"></path>';

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

function renderIcon(paths, className = '') {
  return `<svg${className ? ` class="${escapeHtml(className)}"` : ''} viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>`;
}

function renderPersonalRails(activeCategory) {
  const active = normalizeCategory(activeCategory);
  return [
    '<nav class="personal-accordion__rails" aria-label="Explore Daniel Short">',
    ...CATEGORY_ORDER.map((id) => {
      const category = CATEGORY_CONFIG[id];
      const isActive = id === active;
      return [
        `<a class="personal-accordion__rail personal-accordion__rail--${id}${isActive ? ' is-active' : ''}" href="${category.href}" style="--rail-color: ${category.color}; --rail-color-end: ${category.colorEnd};"${isActive ? ` data-personal-rail-active="true" aria-current="page" aria-label="${category.label}, current category"` : ''}>`,
        `  <span class="personal-accordion__rail-icon" aria-hidden="true">${renderIcon(category.icon)}</span>`,
        `  <span class="personal-accordion__rail-label">${category.label}</span>`,
        isActive ? '  <span class="personal-accordion__rail-notch" aria-hidden="true"></span>' : '',
        '</a>'
      ].filter(Boolean).join('\n');
    }),
    '</nav>'
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
      'data-personal-fit'
    ].reduce((tag, name) => removeTagAttribute(tag, name), bodyTag);
    if (/\sdata-audience="personal"/i.test(next)) next = removeTagAttribute(next, 'data-audience');
    return next;
  });
}

function stripPersonalStylesheet(html) {
  return String(html || '').replace(/^\s*<link\b[^>]*href="(?:\/?dist\/)?styles-personal-accordion(?:\.[0-9a-f]{8})?\.css"[^>]*>\s*$/gim, '');
}

function unwrapPersonalAccordionHtml(html) {
  let output = String(html || '');
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

  if (options.includeToolChrome) {
    const beforeMain = html.slice(0, main.start);
    const heroMatches = [...beforeMain.matchAll(/<section\b[^>]*class="[^"]*\btools-hero\b[^"]*"[^>]*>/gi)];
    if (heroMatches.length) start = heroMatches[heroMatches.length - 1].index;
  }

  if (options.includeProjectPager) {
    const beforeMain = html.slice(0, main.start);
    const pagerMatches = [...beforeMain.matchAll(/<nav\b[^>]*class="[^"]*\bproject-pager\b[^"]*"[^>]*>/gi)];
    if (pagerMatches.length) start = pagerMatches[pagerMatches.length - 1].index;
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
  const category = CATEGORY_CONFIG[categoryId];
  const isLibrary = options.view === 'library';
  const itemId = String(options.itemId || categoryId).trim() || categoryId;
  const fit = String(options.fit || 'document').trim() || 'document';
  const backLabel = String(options.backLabel || `Back to ${category.label}`).trim();
  const backHref = String(options.backHref || category.libraryHref || category.href).trim();
  const toolbar = isLibrary ? '' : [
    '<div class="personal-accordion__toolbar">',
    `  <a class="personal-accordion__back" href="${escapeHtml(backHref)}">`,
    `    <span aria-hidden="true">${renderIcon(ARROW_LEFT)}</span>`,
    `    <span>${escapeHtml(backLabel)}</span>`,
    '  </a>',
    '</div>'
  ].join('\n');

  return [
    PERSONAL_SHELL_START,
    `<section class="personal-accordion personal-accordion--${escapeHtml(categoryId)} personal-accordion--${isLibrary ? 'library' : 'detail'}" data-personal-accordion-shell data-personal-active-category="${escapeHtml(categoryId)}" style="--panel-color: ${category.color}; --panel-color-end: ${category.colorEnd};">`,
    '  <div class="personal-accordion__shell">',
    renderPersonalRails(categoryId).split('\n').map((line) => `    ${line}`).join('\n'),
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
    PERSONAL_SHELL_END
  ].filter(Boolean).join('\n');
}

function wrapPersonalAccordionHtml(html, options = {}) {
  const cleanHtml = unwrapPersonalAccordionHtml(html);
  const range = findFragmentRange(cleanHtml, options);
  const fragment = cleanHtml.slice(range.start, range.end);
  const shell = renderPersonalAccordionShell(fragment, options);
  const category = normalizeCategory(options.category);
  const bodyAttributes = {
    'data-audience': 'personal',
    'data-personal-accordion-view': options.view === 'library' ? 'library' : 'detail',
    'data-personal-category': category,
    'data-personal-item': String(options.itemId || category).trim() || category,
    'data-personal-fit': String(options.fit || 'document').trim() || 'document'
  };
  const suffix = cleanHtml.slice(range.end);
  const boundary = suffix && !/^\r?\n/.test(suffix) ? '\n' : '';
  const output = cleanHtml.slice(0, range.start) + shell + boundary + suffix;
  return setBodyAttributes(output, bodyAttributes);
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
  const title = String(options.title || `${category.label} library`).trim();
  const description = String(options.description || '').trim();
  const toolsDock = categoryId === 'tools' ? [
    '  <div class="tools-account-dock tools-account-dock--directory personal-library__account" data-tools-account="dock">',
    '    <div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner">',
    '      <div class="tools-account-bar" data-tools-account="bar"></div>',
    '    </div>',
    '  </div>'
  ] : [];
  const libraryClasses = [
    'home-library',
    'personal-library',
    `personal-library--${categoryId}`,
    ...(categoryId === 'tools' ? ['tools-hero'] : [])
  ].join(' ');

  return [
    `<main id="main" class="personal-library-main personal-library-main--${categoryId}">`,
    ...toolsDock,
    `  <section class="${libraryClasses}" aria-labelledby="personal-library-title-${categoryId}">`,
    '    <div class="home-library__header">',
    '      <div class="home-library__heading wrapper">',
    `        <h1 id="personal-library-title-${categoryId}">${escapeHtml(title)}</h1>`,
    description ? `        <p>${escapeHtml(description)}</p>` : '',
    '      </div>',
    '    </div>',
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
  PERSONAL_CONTENT_END,
  PERSONAL_CONTENT_START,
  PERSONAL_SHELL_END,
  PERSONAL_SHELL_START,
  extractMainHtml,
  findFragmentRange,
  findMainRange,
  markProfessionalInternalHtml,
  renderPersonalAccordionShell,
  renderPersonalLibraryMain,
  renderPersonalRails,
  replaceMainHtml,
  unwrapPersonalAccordionHtml,
  wrapPersonalAccordionHtml
};
