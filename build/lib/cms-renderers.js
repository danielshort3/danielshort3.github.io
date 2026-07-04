#!/usr/bin/env node
'use strict';

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function ensureLeadingSlash(value) {
  const raw = String(value || '').trim();
  if (!raw) return '/';
  return raw.startsWith('/') ? raw : `/${raw}`;
}

function trimLeadingSlash(value) {
  return String(value || '').trim().replace(/^\/+/, '');
}

function normalizeRelativeHref(value, fallback = '') {
  const raw = String(value || '').trim();
  const selected = raw === '/' ? fallback : (raw || fallback);
  if (selected === '/') return '/';
  return trimLeadingSlash(selected || '');
}

function indentBlock(block, indent = '  ') {
  return String(block || '')
    .split('\n')
    .map((line) => `${indent}${line}`.trimEnd())
    .join('\n');
}

function attrsToString(attrs) {
  const pairs = [];
  Object.entries(attrs || {}).forEach(([key, rawValue]) => {
    if (rawValue === false || rawValue == null) return;
    if (rawValue === true) {
      pairs.push(key);
      return;
    }
    pairs.push(`${key}="${escapeHtml(rawValue)}"`);
  });
  return pairs.length ? ` ${pairs.join(' ')}` : '';
}

function renderToolIconMarkup(tool) {
  const iconImage = String(tool && tool.iconImage ? tool.iconImage : '').trim();
  if (iconImage) {
    return `<img src="${escapeHtml(trimLeadingSlash(iconImage))}" alt="" loading="lazy" decoding="async">`;
  }
  return String(tool && tool.iconHtml ? tool.iconHtml : '').trim();
}

function hasToolIconImage(tool) {
  return Boolean(String(tool && tool.iconImage ? tool.iconImage : '').trim());
}

function renderSvgMarkup(iconType) {
  switch (String(iconType || '').trim()) {
    case 'direct-message':
      return [
        '<svg viewBox="0 0 24 24" aria-hidden="true">',
        '  <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>',
        '  <path d="M7 9h10"></path>',
        '  <path d="M7 13h6"></path>',
        '</svg>'
      ].join('\n');
    case 'email':
      return [
        '<svg viewBox="0 0 24 24" aria-hidden="true">',
        '  <rect x="3" y="5" width="18" height="14" rx="2"></rect>',
        '  <path d="M3 7l9 6 9-6"></path>',
        '</svg>'
      ].join('\n');
    case 'linkedin':
      return [
        '<svg class="brand-fill" viewBox="0 0 24 24" aria-hidden="true">',
        '  <circle cx="4" cy="4" r="2"></circle>',
        '  <rect x="2" y="9" width="4" height="12" rx="1"></rect>',
        '  <path d="M10 9h3.8v2.1h.1C14.8 9.7 16.1 9 17.9 9c3 0 5.1 1.9 5.1 5.9V21h-4v-5.9c0-1.7-.7-2.9-2.6-2.9s-2.7 1.4-2.7 3V21H10z"></path>',
        '</svg>'
      ].join('\n');
    case 'github':
      return '<span class="icon icon-github" aria-hidden="true"></span>';
    case 'speed-dial-toggle':
      return [
        '<svg viewBox="0 0 24 24" aria-hidden="true">',
        '  <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>',
        '  <path d="M12 8v6"></path>',
        '  <path d="M9 11h6"></path>',
        '</svg>'
      ].join('\n');
    case 'search':
      return [
        '<svg viewBox="0 0 24 24" aria-hidden="true">',
        '  <circle cx="11" cy="11" r="7"></circle>',
        '  <path d="M20 20l-3.5-3.5"></path>',
        '</svg>'
      ].join('\n');
    default:
      return '';
  }
}

function renderNavLink(link, extraClasses = '') {
  if (!link) return '';
  const classes = [link.className, extraClasses].filter(Boolean).join(' ').trim();
  const attrs = {
    href: trimLeadingSlash(link.href || ''),
    class: classes || 'nav-link',
    ...(link.target ? { target: link.target } : {}),
    ...(link.rel ? { rel: link.rel } : {}),
    ...(link.ariaLabel ? { 'aria-label': link.ariaLabel } : {}),
    ...link.dataAttributes
  };
  return `<a${attrsToString(attrs)}>${escapeHtml(link.label || '')}</a>`;
}

function renderDropdownLink(link, extraClasses = '') {
  const classes = [link.className, extraClasses].filter(Boolean).join(' ').trim();
  const inferredDataAttributes = { ...(link.dataAttributes || {}) };
  const href = String(link.href || '').trim();
  if (!inferredDataAttributes['data-contact-modal-link'] && href.includes('#contact-modal')) {
    inferredDataAttributes['data-contact-modal-link'] = 'true';
  }
  if (!inferredDataAttributes['data-resume-home-link'] && /^resume(?:$|[#?])/i.test(href)) {
    inferredDataAttributes['data-resume-home-link'] = 'true';
  }
  if (!inferredDataAttributes['data-resume-preview-link'] && /^resume-pdf(?:$|[#?])/i.test(href)) {
    inferredDataAttributes['data-resume-preview-link'] = 'true';
  }
  if (!inferredDataAttributes['data-resume-download-link'] && /documents\/resume/i.test(href)) {
    inferredDataAttributes['data-resume-download-link'] = 'true';
  }
  const attrs = {
    href: trimLeadingSlash(link.href || ''),
    class: classes || 'nav-dropdown-link',
    role: 'listitem',
    ...(link.target ? { target: link.target } : {}),
    ...(link.rel ? { rel: link.rel } : {}),
    ...(link.download ? { download: true } : {}),
    ...inferredDataAttributes
  };
  return [
    `<a${attrsToString(attrs)}>`,
    `  <span class="nav-dropdown-title">${link.badge ? `${escapeHtml(link.title)}<span class="nav-dropdown-badge" aria-hidden="true">${escapeHtml(link.badge)}</span>` : escapeHtml(link.title)}</span>`,
    `  <span class="nav-dropdown-subtitle">${escapeHtml(link.subtitle || '')}</span>`,
    '</a>'
  ].join('\n');
}

function projectThumbPath(project) {
  const raw = String(project && project.image ? project.image : '').trim();
  if (!raw) return '';
  if (project.thumbImage) return ensureLeadingSlash(project.thumbImage);
  return ensureLeadingSlash(raw.replace(/\.(png|jpe?g)$/i, '.webp'));
}

function normalizeProjectSubtitle(project) {
  return String(project && (project.navSubtitle || project.subtitle) ? (project.navSubtitle || project.subtitle) : '').trim();
}

function renderPortfolioProjectCard(project, rank) {
  const thumb = projectThumbPath(project);
  const thumbAttrs = {
    class: 'nav-project-thumb',
    style: thumb ? `background-image:url('${thumb}');` : null,
    'aria-hidden': 'true',
    ...(project.videoWebm ? { 'data-preview-webm': ensureLeadingSlash(project.videoWebm) } : {}),
    ...(project.videoMp4 ? { 'data-preview-mp4': ensureLeadingSlash(project.videoMp4) } : {}),
    ...(thumb ? { 'data-preview-poster': thumb } : {})
  };

  return [
    `<a href="portfolio/${escapeHtml(project.id)}" class="nav-project-card" data-project-id="${escapeHtml(project.id)}" role="listitem">`,
    `  <span class="nav-project-rank">#${rank}</span>`,
    `  <span${attrsToString(thumbAttrs)}></span>`,
    '  <span class="nav-project-meta">',
    `    <span class="nav-dropdown-title">${escapeHtml(project.title || '')}</span>`,
    `    <span class="nav-dropdown-subtitle">${escapeHtml(normalizeProjectSubtitle(project))}</span>`,
    '  </span>',
    '</a>'
  ].join('\n');
}

function sortByOrderThenTitle(items, titleKey = 'title') {
  return [...(Array.isArray(items) ? items : [])].sort((a, b) => {
    const orderA = Number.isFinite(Number(a && a.order)) ? Number(a.order) : Number.MAX_SAFE_INTEGER;
    const orderB = Number.isFinite(Number(b && b.order)) ? Number(b.order) : Number.MAX_SAFE_INTEGER;
    if (orderA !== orderB) return orderA - orderB;
    return String(a && a[titleKey] || '').localeCompare(String(b && b[titleKey] || ''));
  });
}

function isPublicTool(tool) {
  const visibility = String(tool && tool.visibility ? tool.visibility : 'public').trim().toLowerCase();
  return Boolean(tool && !tool.hidden && !tool.noindex && visibility === 'public');
}

function toolHref(tool) {
  const slug = String(tool && tool.slug ? tool.slug : '').trim();
  return trimLeadingSlash(tool && tool.href ? tool.href : (slug ? `tools/${slug}` : ''));
}

function renderDropdownFooterLinks(links, extraClass = '') {
  return (Array.isArray(links) ? links : [])
    .map((link) => [
      `<a href="${escapeHtml(trimLeadingSlash(link.href || ''))}" class="nav-dropdown-link nav-dropdown-all${extraClass ? ` ${escapeHtml(extraClass)}` : ''}"${attrsToString(link.dataAttributes || {})}>`,
      `  <span class="nav-dropdown-title">${escapeHtml(link.title || '')}</span>`,
      `  <span class="nav-dropdown-subtitle">${escapeHtml(link.subtitle || '')}</span>`,
      '</a>'
    ].join('\n'))
    .join('\n');
}

function renderToolsDropdown(toolsNav, toolsPage, tools) {
  if (!toolsNav || toolsNav.enabled === false) return '';
  const categories = sortByOrderThenTitle(Array.isArray(toolsPage && toolsPage.categories) ? toolsPage.categories : []);
  const publicTools = sortByOrderThenTitle((Array.isArray(tools) ? tools : []).filter(isPublicTool));
  const featuredPerCategory = Math.max(1, Math.min(4, Number(toolsNav.featuredPerCategory) || 2));
  const groups = categories
    .map((category) => {
      const items = publicTools
        .filter((tool) => String(tool.categoryId || '') === String(category.id || ''))
        .slice(0, featuredPerCategory);
      if (!items.length) return '';
      const links = items
        .map((tool) => renderDropdownLink({
          title: tool.title,
          subtitle: tool.summary,
          href: toolHref(tool)
        }, 'nav-dropdown-link nav-dropdown-tool-link'))
        .join('\n');
      return [
        '<div class="nav-dropdown-group">',
        `  <div class="nav-dropdown-header" aria-hidden="true">${escapeHtml(category.title || '')}</div>`,
        '  <div class="nav-dropdown-list" role="list">',
        indentBlock(links, '    '),
        '  </div>',
        '</div>'
      ].join('\n');
    })
    .filter(Boolean)
    .join('\n');
  const footerLinks = renderDropdownFooterLinks(toolsNav.links, 'nav-dropdown-tools-all');
  if (!groups && !footerLinks) return '';

  return [
    '        <div class="nav-item nav-item-tools">',
    `          <a href="${escapeHtml(trimLeadingSlash(toolsNav.href || 'tools'))}" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-tools">`,
    `            ${escapeHtml(toolsNav.label || 'Tools')}`,
    '            <span class="nav-link-caret" aria-hidden="true"></span>',
    '          </a>',
    '          <div class="nav-dropdown nav-dropdown-directory nav-dropdown-tools" id="nav-dropdown-tools" aria-label="Featured tools">',
    '            <div class="nav-dropdown-inner nav-dropdown-inner-directory nav-dropdown-inner-tools">',
    indentBlock(groups, '              '),
    footerLinks ? '              <div class="nav-dropdown-footer nav-dropdown-footer-inline">' : '',
    footerLinks ? indentBlock(footerLinks, '                ') : '',
    footerLinks ? '              </div>' : '',
    '            </div>',
    '          </div>',
    '        </div>'
  ].filter(Boolean).join('\n');
}

function gameHref(game) {
  const id = String(game && game.id ? game.id : '').trim();
  return trimLeadingSlash(game && game.href ? game.href : (id ? `games/${id}` : ''));
}

function gameSubtitle(game) {
  const tags = Array.isArray(game && game.tags) ? game.tags.filter(Boolean).join(', ') : '';
  return String(game && (game.navSubtitle || tags || game.summary) || '').trim();
}

function renderGamesDropdown(gamesNav, gamesPage) {
  if (!gamesNav || gamesNav.enabled === false) return '';
  const games = sortByOrderThenTitle(Array.isArray(gamesPage && gamesPage.games) ? gamesPage.games : []);
  const gameLinks = games
    .map((game) => renderDropdownLink({
      title: game.title,
      subtitle: gameSubtitle(game),
      href: gameHref(game)
    }, 'nav-dropdown-link nav-dropdown-game-link'))
    .join('\n');
  const footerLinks = renderDropdownFooterLinks(gamesNav.links, 'nav-dropdown-games-all');
  if (!gameLinks && !footerLinks) return '';

  return [
    '        <div class="nav-item nav-item-games">',
    `          <a href="${escapeHtml(trimLeadingSlash(gamesNav.href || 'games'))}" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-games">`,
    `            ${escapeHtml(gamesNav.label || 'Games')}`,
    '            <span class="nav-link-caret" aria-hidden="true"></span>',
    '          </a>',
    '          <div class="nav-dropdown nav-dropdown-simple nav-dropdown-games" id="nav-dropdown-games" aria-label="Browser games">',
    '            <div class="nav-dropdown-inner nav-dropdown-inner-simple nav-dropdown-inner-games">',
    '              <div class="nav-dropdown-column nav-dropdown-column-list">',
    `                <div class="nav-dropdown-header" aria-hidden="true">${escapeHtml(gamesNav.header || 'Games')}</div>`,
    '                <div class="nav-dropdown-list" role="list">',
    indentBlock(gameLinks, '                  '),
    '                </div>',
    footerLinks ? '                <div class="nav-dropdown-footer nav-dropdown-footer-inline">' : '',
    footerLinks ? indentBlock(footerLinks, '                  ') : '',
    footerLinks ? '                </div>' : '',
    '              </div>',
    '            </div>',
    '          </div>',
    '        </div>'
  ].filter(Boolean).join('\n');
}

function renderGameIconMarkup(iconType) {
  switch (String(iconType || '').trim()) {
    case 'dogfight':
      return [
        '<svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">',
        '  <path d="M12 2l4 6-4 12-4-12 4-6z" class="icon-fill" opacity=".12"></path>',
        '  <path d="M12 2l4 6-4 12-4-12 4-6z"></path>',
        '  <path d="M8 14l-4 2 2-4"></path>',
        '  <path d="M16 14l4 2-2-4"></path>',
        '</svg>'
      ].join('\n');
    case 'roulette':
      return [
        '<svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">',
        '  <circle cx="12" cy="12" r="8" class="icon-fill" opacity=".12"></circle>',
        '  <circle cx="12" cy="12" r="8"></circle>',
        '  <circle cx="12" cy="12" r="2.1"></circle>',
        '  <path d="M12 4v2.3M20 12h-2.3M12 20v-2.3M4 12h2.3"></path>',
        '</svg>'
      ].join('\n');
    case 'probability':
      return [
        '<svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">',
        '  <rect x="4" y="4" width="16" height="16" rx="3" class="icon-fill" opacity=".12"></rect>',
        '  <rect x="4" y="4" width="16" height="16" rx="3"></rect>',
        '  <path d="M7 9h10M7 12h10M7 15h10M9 7v10M12 7v10M15 7v10"></path>',
        '</svg>'
      ].join('\n');
    case 'starfall':
      return [
        '<svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">',
        '  <path d="M12 3l2.2 5.1 5.4.5-4.1 3.5 1.2 5.3L12 14.6 7.3 17.4l1.2-5.3-4.1-3.5 5.4-.5L12 3z" class="icon-fill" opacity=".12"></path>',
        '  <path d="M12 3l2.2 5.1 5.4.5-4.1 3.5 1.2 5.3L12 14.6 7.3 17.4l1.2-5.3-4.1-3.5 5.4-.5L12 3z"></path>',
        '  <path d="M12 7.3v6.1M9.2 10.2h5.6"></path>',
        '</svg>'
      ].join('\n');
    case 'ocean':
      return [
        '<svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">',
        '  <rect x="4" y="4" width="16" height="16" rx="3" class="icon-fill" opacity=".12"></rect>',
        '  <rect x="4" y="4" width="16" height="16" rx="3"></rect>',
        '  <path d="M6 11c1.6-1.6 3.8-1.6 5.4 0s3.8 1.6 5.4 0"></path>',
        '  <path d="M6 15c1.6-1.6 3.8-1.6 5.4 0s3.8 1.6 5.4 0"></path>',
        '</svg>'
      ].join('\n');
    default:
      return '';
  }
}

function renderHeader({ settings, navigation, projectsById, pagesById, tools, audienceLabel }) {
  const brand = navigation.brand || {};
  const portfolio = navigation.portfolio || {};
  const toolsNav = navigation.tools || {};
  const gamesNav = navigation.games || {};
  const resume = navigation.resume || {};
  const contact = navigation.contact || {};
  const search = navigation.search || {};
  const toolsPage = pagesById && pagesById.tools;
  const gamesPage = pagesById && pagesById.games;
  const featuredProjectIds = Array.isArray(portfolio.featuredProjectIds) ? portfolio.featuredProjectIds : [];
  const featuredCards = featuredProjectIds
    .map((id, index) => {
      const project = projectsById && projectsById[id];
      return project ? renderPortfolioProjectCard(project, index + 1) : '';
    })
    .filter(Boolean)
    .join('\n');

  const contactLinks = (Array.isArray(contact.links) ? contact.links : [])
    .map((link) => renderDropdownLink(link, 'nav-dropdown-link'))
    .join('\n');

  const resumeLinks = (Array.isArray(resume.links) ? resume.links : [])
    .map((link) => renderDropdownLink(link, 'nav-dropdown-link'))
    .join('\n');

  const portfolioFooterLinks = renderDropdownFooterLinks(portfolio.links);
  const primaryLinks = (Array.isArray(navigation.primary) ? navigation.primary : [])
    .filter((link) => link && link.href && link.label)
    .map((link) => `<a href="${escapeHtml(trimLeadingSlash(link.href || ''))}" class="nav-link"${attrsToString(link.dataAttributes || {})}>${escapeHtml(link.label || '')}</a>`)
    .join('\n');
  const resumeEnabled = resume.enabled !== false && (resume.label || resumeLinks);
  const toolsDropdown = renderToolsDropdown(toolsNav, toolsPage, tools);
  const gamesDropdown = renderGamesDropdown(gamesNav, gamesPage);

  return [
    '<header id="combined-header-nav">',
    '  <nav class="nav" aria-label="Primary">',
    '    <div class="wrapper nav-wrapper">',
    `      <a href="${escapeHtml(normalizeRelativeHref(brand.homePath || '/', '/'))}" class="brand" aria-label="Home" data-entry-home-link="true">`,
    `        <img src="${escapeHtml(brand.logoSrc || 'img/ui/logo-64.png')}" srcset="${escapeHtml(brand.logoSrcSet || 'img/ui/logo-64.png 1x, img/ui/logo-192.png 3x')}" sizes="${escapeHtml(brand.logoSizes || '64px')}" alt="${escapeHtml(brand.logoAlt || 'DS logo')}" class="brand-logo" decoding="async" loading="eager" width="${escapeHtml(brand.logoWidth || 64)}" height="${escapeHtml(brand.logoHeight || 64)}">`,
    '        <span class="brand-name">',
    `          <span class="brand-title">${escapeHtml(settings.ownerName || 'Daniel Short')}</span>`,
    '        </span>',
    '      </a>',
    '      <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false" aria-controls="primary-menu">',
    '        <span class="bar"></span><span class="bar"></span><span class="bar"></span>',
    '      </button>',
    '      <div id="primary-menu" class="nav-row" data-collapsible role="navigation">',
    '        <div class="nav-item nav-item-portfolio">',
    `          <a href="${escapeHtml(trimLeadingSlash(portfolio.href || 'portfolio'))}" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-portfolio" data-portfolio-home-link="true">`,
    `            ${escapeHtml(portfolio.label || 'Portfolio')}`,
    '            <span class="nav-link-caret" aria-hidden="true"></span>',
    '          </a>',
    '          <div class="nav-dropdown" id="nav-dropdown-portfolio" aria-label="Highlighted projects">',
    '            <div class="nav-dropdown-inner nav-dropdown-inner-portfolio">',
    '              <div class="nav-dropdown-column nav-dropdown-column-list nav-portfolio-stack">',
    `                <div class="nav-dropdown-header" aria-hidden="true">${escapeHtml(portfolio.header || 'Featured Projects')}</div>`,
    '                <div class="nav-project-grid nav-project-stack" role="list">',
    indentBlock(featuredCards, '                  '),
    '                </div>',
    '                <div class="nav-dropdown-footer nav-dropdown-footer-inline">',
    indentBlock(portfolioFooterLinks, '                  '),
    '                </div>',
    '              </div>',
    '            </div>',
    '          </div>',
    '        </div>',
    toolsDropdown,
    gamesDropdown,
    primaryLinks ? indentBlock(primaryLinks, '        ') : '',
    ...(resumeEnabled ? [
      '        <div class="nav-item nav-item-resume">',
      `          <a href="${escapeHtml(trimLeadingSlash(resume.href || 'resume'))}" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-resume" data-resume-home-link="true">`,
      `            ${escapeHtml(resume.label || 'Resume')}`,
      '            <span class="nav-link-caret" aria-hidden="true"></span>',
      '          </a>',
      `          <div class="nav-dropdown nav-dropdown-simple" id="nav-dropdown-resume" aria-label="${escapeHtml(resume.ariaLabel || 'Resume download')}">`,
      '            <div class="nav-dropdown-inner nav-dropdown-inner-simple">',
      '              <div class="nav-dropdown-column nav-dropdown-column-list">',
      `                <div class="nav-dropdown-header" aria-hidden="true">${escapeHtml(resume.header || 'Resume shortcuts')}</div>`,
      '                <div class="nav-dropdown-list" role="list">',
      indentBlock(resumeLinks, '                  '),
      '                </div>',
      '              </div>',
      '            </div>',
      '          </div>',
      '        </div>'
    ] : []),
    '        <div class="nav-item nav-item-contact">',
    `          <a href="${escapeHtml(trimLeadingSlash(contact.href || 'contact'))}" class="nav-link nav-link-cta nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-contact">`,
    `            ${escapeHtml(contact.label || 'Contact')}`,
    '            <span class="nav-link-caret" aria-hidden="true"></span>',
    '          </a>',
    '          <div class="nav-dropdown nav-dropdown-simple nav-dropdown-contact" id="nav-dropdown-contact" aria-label="Contact options">',
    '            <div class="nav-dropdown-inner nav-dropdown-inner-simple nav-dropdown-inner-contact">',
    '              <div class="nav-dropdown-column nav-dropdown-column-list">',
    `                <div class="nav-dropdown-header" aria-hidden="true">${escapeHtml(contact.header || 'Get in touch')}</div>`,
    '                <div class="nav-dropdown-list" role="list">',
    indentBlock(contactLinks, '                  '),
    '                </div>',
    '              </div>',
    '            </div>',
    '          </div>',
    '        </div>',
    `        <form class="nav-search" action="${escapeHtml(trimLeadingSlash(search.action || 'search'))}" method="get" role="search" data-nav-search="collapsed">`,
    `          <label class="visually-hidden" for="nav-search-q">${escapeHtml(search.label || 'Search site')}</label>`,
    '          <div class="nav-search-field">',
    `            <input id="nav-search-q" class="nav-search-input" type="search" name="q" placeholder="${escapeHtml(search.placeholder || 'Search…')}">`,
    '            <button class="nav-search-button" type="submit" aria-controls="nav-search-q" aria-expanded="false">',
    '              <span class="visually-hidden">Search</span>',
    indentBlock(renderSvgMarkup('search'), '              '),
    '            </button>',
    '          </div>',
    '        </form>',
    '      </div>',
    '    </div>',
    '  </nav>',
    '</header>'
  ].join('\n');
}

function renderFooter({ footer, year }) {
  const columns = Array.isArray(footer.columns) ? footer.columns : [];
  const renderedColumns = columns.map((column, index) => {
    const titleId = `footer-${String(column.id || index + 1).trim() || index + 1}`;
    const links = (Array.isArray(column.links) ? column.links : [])
      .map((link) => {
        const inferredDataAttributes = { ...(link.dataAttributes || {}) };
        const href = String(link.href || '').trim();
        const label = String(link.label || '').trim().toLowerCase();
        if (!inferredDataAttributes['data-portfolio-home-link'] && (href === 'portfolio' || href === '/portfolio' || label === 'portfolio')) {
          inferredDataAttributes['data-portfolio-home-link'] = 'true';
        }
        if (!inferredDataAttributes['data-resume-home-link'] && (href === 'resume' || href === '/resume' || label === 'resume')) {
          inferredDataAttributes['data-resume-home-link'] = 'true';
        }
        if (!inferredDataAttributes['data-smooth-scroll'] && href.startsWith('#')) {
          inferredDataAttributes['data-smooth-scroll'] = 'true';
        }
        const attrs = {
          href: normalizeRelativeHref(href, label === 'home' ? '/' : ''),
          class: 'footer-link',
          ...(link.target ? { target: link.target } : {}),
          ...(link.rel ? { rel: link.rel } : {}),
          ...(link.download ? { download: true } : {}),
          ...(link.hidden ? { hidden: true } : {}),
          ...(link.type === 'button'
            ? {
                href: null,
                type: 'button',
                id: link.id || null,
                'aria-haspopup': link.ariaHaspopup || null
              }
            : {}),
          ...inferredDataAttributes
        };

        if (link.type === 'button') {
          const buttonAttrs = {
            type: 'button',
            class: 'footer-link',
            ...(link.id ? { id: link.id } : {}),
            ...(link.ariaHaspopup ? { 'aria-haspopup': link.ariaHaspopup } : {}),
            ...inferredDataAttributes,
            ...(link.hidden ? { hidden: true } : {})
          };
          return `<button${attrsToString(buttonAttrs)}>${escapeHtml(link.label || '')}</button>`;
        }

        return `<a${attrsToString(attrs)}>${escapeHtml(link.label || '')}</a>`;
      })
      .join('\n');

    return [
      `<section class="footer-col" aria-labelledby="${escapeHtml(titleId)}">`,
      `  <h2 class="footer-col-title" id="${escapeHtml(titleId)}">${escapeHtml(column.title || '')}</h2>`,
      indentBlock(links, '  '),
      '</section>'
    ].join('\n');
  }).join('\n');

  const speedDial = footer.speedDial || {};
  const speedDialItems = (Array.isArray(speedDial.items) ? speedDial.items : [])
    .map((item) => {
      const inferredDataAttributes = { ...(item.dataAttributes || {}) };
      if (!inferredDataAttributes['data-contact-modal-link'] && String(item.href || '').includes('#contact-modal')) {
        inferredDataAttributes['data-contact-modal-link'] = 'true';
      }
      const linkAttrs = {
        href: trimLeadingSlash(item.href || ''),
        class: `speed-dial__action btn-icon${item.variant === 'direct' ? ' speed-dial__action--direct' : ''}`,
        'aria-label': item.ariaLabel || item.label,
        role: 'menuitem',
        ...(item.target ? { target: item.target } : {}),
        ...(item.rel ? { rel: item.rel } : {}),
        ...inferredDataAttributes,
        'data-speed-dial-action': true
      };
      return [
        '<div class="speed-dial__item">',
        `  <span class="speed-dial__label" aria-hidden="true">${escapeHtml(item.label || '')}</span>`,
        `  <a${attrsToString(linkAttrs)}>`,
        indentBlock(renderSvgMarkup(item.iconType), '    '),
        '  </a>',
        '</div>'
      ].join('\n');
    })
    .join('\n');

  return [
    '<footer class="footer footer-classic">',
    '  <div class="wrapper footer-inner">',
    '    <nav class="footer-nav" aria-label="Footer">',
    indentBlock(renderedColumns, '      '),
    '    </nav>',
    `    <p class="footer-meta">© ${escapeHtml(year)} ${escapeHtml(footer.copyrightName || 'Daniel Short')}. All rights reserved.</p>`,
    '  </div>',
    '</footer>',
    '<div class="cookie-settings" data-cookie-settings="true">',
    '  <div class="cookie-settings__item">',
    `    <span class="cookie-settings__label" aria-hidden="true">${escapeHtml(footer.cookieSettingsLabel || 'Cookie settings')}</span>`,
    `    <button id="${escapeHtml(footer.cookieSettingsButtonId || 'privacy-settings-link')}" type="button" class="cookie-settings__toggle btn-icon btn-icon-featured" aria-label="${escapeHtml(footer.cookieSettingsLabel || 'Cookie settings')}" aria-haspopup="dialog">`,
    '      <svg viewBox="0 0 24 24" aria-hidden="true">',
    '        <path d="M21 13a4 4 0 0 1-4-4a4 4 0 0 1-4-4A9 9 0 1 0 21 13z"></path>',
    '        <circle cx="10" cy="10" r="1" fill="currentColor" stroke="none"></circle>',
    '        <circle cx="13" cy="13" r="1" fill="currentColor" stroke="none"></circle>',
    '        <circle cx="9" cy="15.5" r="1" fill="currentColor" stroke="none"></circle>',
    '      </svg>',
    '    </button>',
    '  </div>',
    '</div>',
    '<div class="speed-dial" data-speed-dial="true">',
    '  <div class="speed-dial__tray" data-speed-dial-tray>',
    `    <div class="speed-dial__actions" id="${escapeHtml(speedDial.menuId || 'speed-dial-menu')}" role="menu" aria-label="${escapeHtml(speedDial.menuLabel || 'Contact options')}" aria-hidden="true" data-speed-dial-menu>`,
    indentBlock(speedDialItems, '      '),
    '    </div>',
    '  </div>',
    `  <button class="speed-dial__toggle btn-icon btn-icon-featured" type="button" aria-expanded="false" aria-haspopup="menu" aria-controls="${escapeHtml(speedDial.menuId || 'speed-dial-menu')}" aria-label="${escapeHtml(speedDial.toggleLabel || 'Open contact options')}" data-speed-dial-toggle>`,
    indentBlock(renderSvgMarkup('speed-dial-toggle'), '    '),
    '  </button>',
    '</div>'
  ].join('\n');
}

function renderHead({ settings, page }) {
  const siteOrigin = String(settings.siteOrigin || 'https://www.danielshort.me').trim().replace(/\/+$/, '');
  const ogImage = settings.ogImage || {};
  const canonicalUrl = `${siteOrigin}${ensureLeadingSlash(page.canonicalPath || '/')}`;
  const description = String(page.description || '').trim();
  const ogTitle = String(page.ogTitle || page.title || '').trim();
  const ogDescription = String(page.ogDescription || description).trim();
  const siteName = String(page.siteName || settings.siteName || '').trim();
  const twitterTitle = String(page.twitterTitle || ogTitle).trim();
  const twitterDescription = String(page.twitterDescription || ogDescription).trim();
  const twitterSite = String(page.twitterSite || settings.twitterSite || '').trim();
  const scripts = Array.isArray(page.headScripts) ? page.headScripts : [];
  const stylesheetLines = (Array.isArray(page.stylesheets) ? page.stylesheets : [])
    .map((href) => `  <link rel="stylesheet" href="${escapeHtml(href)}">`)
    .join('\n');
  const scriptLines = scripts
    .map((script) => renderScriptTag(script, '  '))
    .join('\n');

  return [
    '<!DOCTYPE html>',
    `<html lang="${escapeHtml(page.lang || 'en')}" class="no-js">`,
    '<head>',
    '  <meta charset="UTF-8">',
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">',
    '  <base href="/">',
    `  <title>${escapeHtml(page.title || '')}</title>`,
    `  <link rel="canonical" href="${escapeHtml(canonicalUrl)}">`,
    description ? `  <meta name="description" content="${escapeHtml(description)}">` : '',
    page.robots ? `  <meta name="robots" content="${escapeHtml(page.robots)}">` : '',
    `  <meta property="og:title" content="${escapeHtml(ogTitle)}">`,
    siteName ? `  <meta property="og:site_name" content="${escapeHtml(siteName)}">` : '',
    `  <meta property="og:description" content="${escapeHtml(ogDescription)}">`,
    `  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">`,
    `  <meta property="og:image" content="${escapeHtml(ogImage.url || '')}">`,
    ogImage.width ? `  <meta property="og:image:width" content="${escapeHtml(ogImage.width)}">` : '',
    ogImage.height ? `  <meta property="og:image:height" content="${escapeHtml(ogImage.height)}">` : '',
    ogImage.alt ? `  <meta property="og:image:alt" content="${escapeHtml(ogImage.alt)}">` : '',
    `  <meta property="og:type" content="${escapeHtml(page.ogType || 'website')}">`,
    '  <meta name="twitter:card" content="summary_large_image">',
    twitterSite ? `  <meta name="twitter:site" content="${escapeHtml(twitterSite)}">` : '',
    `  <meta name="twitter:title" content="${escapeHtml(twitterTitle)}">`,
    `  <meta name="twitter:description" content="${escapeHtml(twitterDescription)}">`,
    ogImage.url ? `  <meta name="twitter:image" content="${escapeHtml(ogImage.url)}">` : '',
    ogImage.alt ? `  <meta name="twitter:image:alt" content="${escapeHtml(ogImage.alt)}">` : '',
    `  <meta name="theme-color" content="${escapeHtml(page.themeColor || settings.themeColor || '#091F3B')}">`,
    stylesheetLines,
    '  <link rel="icon" href="favicon.ico" sizes="any">',
    '  <link rel="icon" type="image/svg+xml" href="img/brand/05-ds-favicon-small-icon.svg">',
    '  <link rel="icon" type="image/png" sizes="16x16" href="img/ui/logo-16.png">',
    '  <link rel="icon" type="image/png" sizes="32x32" href="img/ui/logo-32.png">',
    '  <link rel="icon" type="image/png" sizes="64x64" href="img/ui/logo-64.png">',
    '  <link rel="icon" type="image/png" sizes="192x192" href="img/ui/logo-192.png">',
    '  <link rel="apple-touch-icon" sizes="180x180" href="img/ui/logo-180.png">',
    scriptLines,
    '</head>'
  ].filter(Boolean).join('\n');
}

function renderScriptTag(script, indent = '') {
  if (!script || !script.src) return '';
  const attributes = { ...(script.attributes || {}) };
  if (script.src === 'dist/site-tools-landing.js' && !attributes['data-tools-account-src']) {
    attributes['data-tools-account-src'] = 'dist/site-tools-account.js';
  }
  const attrs = {
    ...(script.defer ? { defer: true } : {}),
    src: script.src,
    ...attributes
  };
  return `${indent}<script${attrsToString(attrs)}></script>`;
}

function renderBodyOpen(page) {
  const attrs = { ...(page.bodyAttributes || {}) };
  return `<body${attrsToString(attrs)}>`;
}

function renderFullPage({ settings, navigation, footer, projectsById, pagesById, tools, page, audienceLabel }) {
  const headerHtml = renderHeader({
    settings,
    navigation,
    projectsById,
    pagesById,
    tools,
    audienceLabel
  });
  const footerHtml = renderFooter({
    footer,
    year: new Date().getFullYear()
  });
  const bottomScripts = (Array.isArray(page.bottomScripts) ? page.bottomScripts : [])
    .map((script) => renderScriptTag(script, '  '))
    .join('\n');

  return [
    renderHead({ settings, page }),
    renderBodyOpen(page),
    '  <a href="#main" class="skip-link">Skip to main content</a>',
    indentBlock(headerHtml, '  '),
    indentBlock(String(page.bodyHtml || '').trim(), '  '),
    indentBlock(footerHtml, '  '),
    bottomScripts,
    '</body>',
    '</html>',
    ''
  ].filter(Boolean).join('\n');
}

function renderToolsDirectoryBody(page, tools) {
  const categories = Array.isArray(page.categories) ? page.categories : [];
  const usableTools = (Array.isArray(tools) ? tools : []).filter((tool) => {
    const slug = String(tool && tool.slug ? tool.slug : '').trim();
    const href = String(tool && tool.href ? tool.href : '').trim();
    return Boolean(slug || href);
  });
  const toolsByCategory = categories.map((category) => {
    const items = usableTools
      .filter((tool) => String(tool.categoryId || '') === String(category.id || ''))
      .sort((a, b) => Number(a.order || 0) - Number(b.order || 0));
    return { category, items };
  }).filter(({ items }) => items.length > 0);

  const renderedCategories = toolsByCategory.map(({ category, items }) => {
    const cards = items.map((tool) => {
      const href = trimLeadingSlash(tool.href || `tools/${tool.slug}`);
      const visibility = String(tool.visibility || 'public').trim().toLowerCase();
      const restricted = visibility && visibility !== 'public';
      const iconClass = hasToolIconImage(tool) ? 'tool-icon tool-icon-image' : 'tool-icon';
      const cardAttrs = {
        class: 'tool-card',
        'data-tools-card': true,
        'data-tools-category-id': category.id || '',
        'data-tools-category-title': category.title || '',
        ...(restricted ? { 'data-tools-visibility': tool.visibility } : {}),
        ...(tool.hidden || restricted ? { hidden: true } : {})
      };
      return [
        `<article${attrsToString(cardAttrs)}>`,
        `  <a class="tool-launch-card" href="${escapeHtml(href)}" aria-describedby="${escapeHtml(tool.slug || href)}-details">`,
        `    <span class="${iconClass}" aria-hidden="true">`,
        indentBlock(renderToolIconMarkup(tool), '        '),
        '    </span>',
        '    <span class="tool-card-main">',
        `      <span class="tool-card-title">${escapeHtml(tool.title || '')}</span>`,
        '    </span>',
        `    <span class="tool-card-details" id="${escapeHtml(tool.slug || href)}-details" role="tooltip">`,
        `      <span class="tool-card-summary">${escapeHtml(tool.summary || '')}</span>`,
        '    </span>',
        '  </a>',
        '</article>'
      ].join('\n');
    }).join('\n\n');

    return [
      `<section class="tools-category" id="${escapeHtml(category.id || '')}" aria-labelledby="${escapeHtml(category.id || '')}-title" data-tools-category data-tools-category-title="${escapeHtml(category.title || '')}">`,
      '  <header class="tools-category-head">',
      `    <h2 class="section-title" id="${escapeHtml(category.id || '')}-title">${escapeHtml(category.title || '')}</h2>`,
      '  </header>',
      '  <div class="tools-grid">',
      indentBlock(cards, '    '),
      '  </div>',
      '</section>'
    ].join('\n');
  }).join('\n\n');
  const categoryRail = toolsByCategory.map(({ category, items }, index) => [
    `<a class="tools-rail-link" href="#${escapeHtml(category.id || '')}">`,
    `  <span class="tools-rail-index">${String(index + 1).padStart(2, '0')}</span>`,
    '  <span class="tools-rail-copy">',
    `    <span class="tools-rail-title">${escapeHtml(category.title || '')}</span>`,
    category.description ? `    <span class="tools-rail-description">${escapeHtml(category.description)}</span>` : '',
    '  </span>',
    `  <span class="tools-rail-count">${items.length}</span>`,
    '</a>'
  ].filter(Boolean).join('\n')).join('\n');
  const totalToolCount = toolsByCategory.reduce((sum, { items }) => sum + items.length, 0);

  const resumePanel = page.resumePanel || {};
  const heroLead = String(page.heroLead || '').trim();
  const directoryKicker = String(page.directoryKicker || '').trim();
  const directoryTitle = String(page.directoryTitle || '').trim();
  const directoryDescription = String(page.directoryDescription || '').trim();
  const directoryControls = (directoryKicker || directoryTitle || directoryDescription)
    ? [
      '      <section class="tools-directory-controls" aria-labelledby="tools-directory-title">',
      '        <div class="tools-directory-head">',
      directoryKicker ? `          <p class="tools-directory-kicker">${escapeHtml(directoryKicker)}</p>` : '',
      directoryTitle ? `          <h2 id="tools-directory-title">${escapeHtml(directoryTitle)}</h2>` : '',
      directoryDescription ? `          <p>${escapeHtml(directoryDescription)}</p>` : '',
      '        </div>',
      '      </section>',
      ''
    ].filter(Boolean)
    : [];

  return [
    '<section class="hero hero--tools tools-hero">',
    '  <div class="wrapper">',
    '    <div class="tools-hero-copy">',
    `      <p class="hero-eyebrow">${escapeHtml(page.heroEyebrow || 'Tools')}</p>`,
    `      <h1>${escapeHtml(page.heroTitle || 'Tools')}</h1>`,
    heroLead ? `      <p class="tools-hero-lead">${escapeHtml(heroLead)}</p>` : '',
    '    </div>',
    '  </div>',
    '</section>',
    '',
    '<div class="tools-account-dock" data-tools-account="dock">',
    '  <div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner">',
    '    <div class="tools-account-bar" data-tools-account="bar"></div>',
    '    <section class="tools-resume-panel" data-tools-resume="panel" data-tools-auth-only aria-labelledby="tools-resume-title" hidden>',
    '      <div class="tools-resume-head">',
    '        <div class="tools-resume-head-copy">',
    `          <p class="tools-resume-kicker">${escapeHtml(resumePanel.kicker || '')}</p>`,
    `          <h2 class="tools-resume-title" id="tools-resume-title">${escapeHtml(resumePanel.title || '')}</h2>`,
    '        </div>',
    '      </div>',
    '      <p class="tools-resume-status" data-tools-resume="status" role="status" aria-live="polite"></p>',
    '      <div class="tools-resume-content" data-tools-resume="content"></div>',
    '    </section>',
    '  </div>',
    '</div>',
    '',
    '<main id="main">',
    '  <section class="surface-band tools-section">',
    '    <div class="wrapper tools-directory-layout">',
    '      <aside class="tools-directory-rail" aria-label="Tool categories">',
    `        <p class="tools-directory-stat">${totalToolCount} tools</p>`,
    indentBlock(categoryRail, '        '),
    '      </aside>',
    '      <div class="tools-directory-main">',
    ...directoryControls,
    '        <div id="tools-directory-results" data-tools-results>',
    indentBlock(renderedCategories, '          '),
    '        </div>',
    '      </div>',
    '      <aside class="tools-directory-note" aria-labelledby="tools-directory-note-title">',
    '        <p class="tools-directory-note-kicker">Why tools?</p>',
    '        <h2 id="tools-directory-note-title">Small utilities should feel finished.</h2>',
    '        <p>I keep these focused on repeatable output, local-first behavior where possible, and clear state when a tool needs account-backed work.</p>',
    '      </aside>',
    '    </div>',
    '  </section>',
    '</main>'
  ].join('\n');
}

function renderGamesDirectoryBody(page) {
  const games = sortByOrderThenTitle((Array.isArray(page.games) ? page.games : [])
    .filter((game) => game && !game.hidden && !game.noindex && (game.href || game.id)));
  const cards = games.map((game) => {
    const id = String(game.id || game.href || '').trim();
    const detailsId = `${id || 'game'}-details`;
    const tags = Array.isArray(game.tags) ? game.tags.filter(Boolean) : [];
    const renderedTags = tags.length
      ? [
        '      <span class="game-card-tags">',
        ...tags.map((tag) => `        <span class="game-pill">${escapeHtml(tag)}</span>`),
        '      </span>'
      ].join('\n')
      : '';
    return [
      '<article class="game-card" data-game-card role="listitem">',
      `  <a class="game-launch-card" href="${escapeHtml(gameHref(game))}" aria-describedby="${escapeHtml(detailsId)}">`,
      `    <span class="game-card-index">${String(Number(game.order || 0) || 0).padStart(2, '0')}</span>`,
      '    <span class="game-icon" aria-hidden="true">',
      indentBlock(renderGameIconMarkup(game.iconType), '      '),
      '    </span>',
      '    <span class="game-card-main">',
      `      <span class="game-card-title">${escapeHtml(game.title || '')}</span>`,
      '    </span>',
      `    <span class="game-card-details" id="${escapeHtml(detailsId)}" role="tooltip">`,
      `      <span class="game-card-summary">${escapeHtml(game.summary || '')}</span>`,
      renderedTags,
      '    </span>',
      '  </a>',
      '</article>'
    ].filter(Boolean).join('\n');
  }).join('\n\n');
  const heroLead = String(page.heroLead || '').trim();

  return [
    '<section class="hero hero--games games-hero">',
    '  <div class="wrapper">',
    '    <div class="games-hero-copy">',
    `      <p class="hero-eyebrow">${escapeHtml(page.heroEyebrow || 'Games')}</p>`,
    `      <h1>${escapeHtml(page.heroTitle || 'Games')}</h1>`,
    heroLead ? `      <p class="games-hero-lead">${escapeHtml(heroLead)}</p>` : '',
    '    </div>',
    '  </div>',
    '</section>',
    '',
    '<main id="main">',
    '  <section class="surface-band games-section">',
    '    <div class="wrapper games-directory-layout">',
    '      <aside class="games-system-panel" aria-labelledby="games-system-title">',
    '        <p class="games-system-kicker">Design lens</p>',
    '        <h2 id="games-system-title">Systems before spectacle.</h2>',
    '        <p>These are places to test loops: probability, enemy behavior, upgrades, pacing, and readable feedback.</p>',
    '        <dl>',
    '          <div><dt>State</dt><dd>What the player tracks</dd></div>',
    '          <div><dt>Balance</dt><dd>How choices compound</dd></div>',
    '          <div><dt>Feedback</dt><dd>What the interface teaches</dd></div>',
    '        </dl>',
    '      </aside>',
    '      <div class="games-grid" role="list">',
    indentBlock(cards, '        '),
    '      </div>',
    '    </div>',
    '  </section>',
    '</main>'
  ].join('\n');
}

function renderProjectsDataJs(projects, featuredIds) {
  return [
    '/* Generated by build/generate-cms-artifacts.js. Do not edit directly. */',
    `window.PROJECTS = ${JSON.stringify(projects, null, 2)};`,
    '',
    `window.FEATURED_IDS = ${JSON.stringify(featuredIds, null, 2)};`,
    ''
  ].join('\n');
}

function serializeJsLiteral(value, depth = 0) {
  const indent = '  '.repeat(depth);
  const nextIndent = '  '.repeat(depth + 1);

  if (Array.isArray(value)) {
    if (!value.length) return '[]';
    return [
      '[',
      value.map((item) => `${nextIndent}${serializeJsLiteral(item, depth + 1)}`).join(',\n'),
      `${indent}]`
    ].join('\n');
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    if (!entries.length) return '{}';
    return [
      '{',
      entries.map(([key, entryValue]) => {
        const safeKey = /^[A-Za-z_$][\w$]*$/.test(key)
          ? key
          : `'${String(key).replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`;
        return `${nextIndent}${safeKey}: ${serializeJsLiteral(entryValue, depth + 1)}`;
      }).join(',\n'),
      `${indent}}`
    ].join('\n');
  }

  if (typeof value === 'string') {
    return `'${value
      .replace(/\\/g, '\\\\')
      .replace(/'/g, "\\'")
      .replace(/\r/g, '\\r')
      .replace(/\n/g, '\\n')}'`;
  }

  if (value == null) return 'null';
  return String(value);
}

function renderAudienceConfigJs(settings, audiences) {
  const order = audiences
    .sort((a, b) => Number(a.order || 0) - Number(b.order || 0))
    .map((audience) => audience.key);
  const audienceMap = audiences.reduce((acc, audience) => {
    acc[audience.key] = {
      key: audience.key,
      label: audience.label,
      shortLabel: audience.shortLabel,
      homePath: audience.homePath,
      portfolioPath: audience.portfolioPath,
      portfolioAllPath: audience.portfolioAllPath,
      resumePath: audience.resumePath,
      resumePreviewPath: audience.resumePreviewPath,
      resumeDownloadPath: audience.resumeDownloadPath,
      featuredProjectIds: audience.featuredProjectIds,
      portfolioTitle: audience.portfolioTitle,
      portfolioDescription: audience.portfolioDescription,
      resumeNavTitle: audience.resumeNavTitle,
      resumeNavSubtitle: audience.resumeNavSubtitle,
      resumePreviewSubtitle: audience.resumePreviewSubtitle,
      resumeDownloadSubtitle: audience.resumeDownloadSubtitle,
      brandNavPrimary: audience.brandNavPrimary
    };
    return acc;
  }, {});
  const defaultAudience = settings.defaultAudience || order[0] || 'personal';

  return [
    '/* Generated by build/generate-cms-artifacts.js. Do not edit directly. */',
    '(function (root, factory) {',
    '  const api = factory();',
    "  if (typeof module === 'object' && module.exports) {",
    '    module.exports = api;',
    '  }',
    '  if (root) {',
    '    root.SITE_AUDIENCE_CONFIG = api;',
    '    root.SITE_AUDIENCES = api.audiences;',
    '    root.SITE_AUDIENCE_ORDER = api.order;',
    '    root.SITE_AUDIENCE_DEFAULT = api.defaultAudience;',
    '    root.getSiteAudienceConfig = api.getAudience;',
    '    root.normalizeSiteAudience = api.normalizeAudience;',
    '    root.detectSiteAudienceFromPath = api.detectAudienceFromPath;',
    '  }',
    "})(typeof globalThis !== 'undefined' ? globalThis : this, function () {",
    "  'use strict';",
    '',
    `  const audiences = ${serializeJsLiteral(audienceMap, 1)};`,
    '',
    `  const order = ${serializeJsLiteral(order, 1)};`,
    `  const defaultAudience = ${serializeJsLiteral(defaultAudience)};`,
    '',
    '  function normalizeAudience(value) {',
    "    const raw = String(value || '').trim().toLowerCase();",
    '    if (!raw) return defaultAudience;',
    "    if (raw === 'datascience' || raw === 'data_science') return 'data-science';",
    "    if (raw === 'tourism-analytics') return 'tourism';",
    '    return audiences[raw] ? raw : defaultAudience;',
    '  }',
    '',
    '  function getAudience(value) {',
    '    return audiences[normalizeAudience(value)] || audiences[defaultAudience];',
    '  }',
    '',
    '  function detectAudienceFromPath(pathname) {',
    "    const path = String(pathname || '').trim().replace(/\\/+$/, '') || '/';",
    '    return order.find((key) => {',
    '      const audience = audiences[key];',
    '      if (!audience) return false;',
    '      return path === audience.homePath',
    '        || path === audience.resumePath',
    '        || path === audience.resumePreviewPath;',
    '    }) || null;',
    '  }',
    '',
    '  return {',
    '    defaultAudience,',
    '    order,',
    '    audiences,',
    '    normalizeAudience,',
    '    getAudience,',
    '    detectAudienceFromPath',
    '  };',
    '});',
    ''
  ].join('\n');
}

module.exports = {
  renderAudienceConfigJs,
  renderFooter,
  renderFullPage,
  renderGamesDirectoryBody,
  renderHeader,
  renderProjectsDataJs,
  renderToolsDirectoryBody
};
