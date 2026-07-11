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
    return `<img src="${escapeHtml(trimLeadingSlash(iconImage))}" alt="" width="256" height="256" loading="lazy" decoding="async">`;
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

function toolVisibility(tool) {
  const visibility = String(tool && tool.visibility ? tool.visibility : 'public').trim().toLowerCase();
  return visibility || 'public';
}

function isDirectoryTool(tool) {
  const visibility = toolVisibility(tool);
  return Boolean(tool && (tool.slug || tool.href) && ['public', 'authed', 'authenticated', 'logged-in', 'admin', 'admins'].includes(visibility));
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

function slugify(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function uniqueLabels(values) {
  const seen = new Set();
  return (Array.isArray(values) ? values : [])
    .map((value) => String(value || '').trim())
    .filter(Boolean)
    .filter((value) => {
      const key = slugify(value);
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function optionList(labels, field) {
  return uniqueLabels(labels)
    .sort((a, b) => a.localeCompare(b))
    .map((label) => ({
      value: slugify(label),
      label,
      field
    }));
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
  const normalizeVariantMap = (value, valueKey) => {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    return Object.entries(value).map(([id, config]) => {
      if (Array.isArray(config)) return { id, [valueKey]: config };
      return { id, ...(config || {}) };
    });
  };

  const inferFooterDataAttributes = (link) => {
    const inferredDataAttributes = { ...(link.dataAttributes || {}) };
    const href = String(link.href || '').trim();
    const label = String(link.label || '').trim().toLowerCase();
    if (!inferredDataAttributes['data-portfolio-home-link'] && (href === 'portfolio' || href === '/portfolio' || href.startsWith('portfolio?') || label === 'portfolio')) {
      inferredDataAttributes['data-portfolio-home-link'] = 'true';
    }
    if (!inferredDataAttributes['data-resume-home-link'] && (href === 'resume' || href === '/resume' || label === 'resume')) {
      inferredDataAttributes['data-resume-home-link'] = 'true';
    }
    if (!inferredDataAttributes['data-contact-modal-link'] && href.includes('#contact-modal')) {
      inferredDataAttributes['data-contact-modal-link'] = 'true';
    }
    if (!inferredDataAttributes['data-smooth-scroll'] && href.startsWith('#')) {
      inferredDataAttributes['data-smooth-scroll'] = 'true';
    }
    return inferredDataAttributes;
  };

  const renderFooterLink = (link, className = 'footer-link') => {
    const inferredDataAttributes = inferFooterDataAttributes(link);
    const href = String(link.href || '').trim();
    const label = String(link.label || '').trim().toLowerCase();
    const hrefFallback = href === '/' || label === 'home' ? '/' : '';

    if (link.type === 'button') {
      const buttonAttrs = {
        type: 'button',
        class: className,
        ...(link.id ? { id: link.id } : {}),
        ...(link.ariaHaspopup ? { 'aria-haspopup': link.ariaHaspopup } : {}),
        ...inferredDataAttributes,
        ...(link.hidden ? { hidden: true } : {})
      };
      return `<button${attrsToString(buttonAttrs)}>${escapeHtml(link.label || '')}</button>`;
    }

    const attrs = {
      href: normalizeRelativeHref(href, hrefFallback),
      class: className,
      ...(link.target ? { target: link.target } : {}),
      ...(link.rel ? { rel: link.rel } : {}),
      ...(link.download ? { download: true } : {}),
      ...(link.hidden ? { hidden: true } : {}),
      ...inferredDataAttributes
    };
    return `<a${attrsToString(attrs)}>${escapeHtml(link.label || '')}</a>`;
  };

  const renderFooterColumn = (column, index, realm = '') => {
    const realmPrefix = realm ? `${realm}-` : '';
    const titleId = `footer-${realmPrefix}${String(column.id || index + 1).trim() || index + 1}`;
    const links = (Array.isArray(column.links) ? column.links : [])
      .map((link) => renderFooterLink(link))
      .join('\n');

    return [
      `<section class="footer-col" aria-labelledby="${escapeHtml(titleId)}">`,
      `  <h2 class="footer-col-title" id="${escapeHtml(titleId)}">${escapeHtml(column.title || '')}</h2>`,
      indentBlock(links, '  '),
      '</section>'
    ].join('\n');
  };

  const identityVariants = normalizeVariantMap(footer.identity && footer.identity.variants, 'identity')
    .filter((variant) => variant && String(variant.id || '').trim());
  const renderedIdentity = (identityVariants.length ? identityVariants : [{
    id: 'personal',
    name: footer.copyrightName || 'Daniel Short',
    summary: 'Projects, tools, games, and experiments.',
    links: []
  }]).map((variant) => {
    const realm = String(variant.id || 'personal').trim();
    const name = variant.name || footer.copyrightName || 'Daniel Short';
    const eyebrow = String(variant.eyebrow || '').trim();
    const identityLinks = (Array.isArray(variant.links) ? variant.links : [])
      .map((link) => renderFooterLink(link, 'footer-link footer-identity-link'))
      .join('\n');
    return [
      `<section class="footer-identity-panel" data-footer-realm="${escapeHtml(realm)}" aria-label="${escapeHtml(`${name} footer summary`)}">`,
      eyebrow ? `  <p class="footer-identity-eyebrow">${escapeHtml(eyebrow)}</p>` : '',
      `  <h2 class="footer-identity-name">${escapeHtml(name)}</h2>`,
      `  <p class="footer-identity-summary">${escapeHtml(variant.summary || '')}</p>`,
      identityLinks ? `  <div class="footer-identity-actions">\n${indentBlock(identityLinks, '    ')}\n  </div>` : '',
      '</section>'
    ].filter(Boolean).join('\n');
  }).join('\n');

  const navVariants = normalizeVariantMap(footer.navVariants, 'columns');
  const renderedNavPanels = (navVariants.length ? navVariants : [{ id: 'personal', columns }])
    .map((variant) => {
      const realm = String(variant.id || 'personal').trim();
      const variantColumns = Array.isArray(variant.columns) ? variant.columns : columns;
      return [
        `<div class="footer-nav-panel" data-footer-realm="${escapeHtml(realm)}">`,
        indentBlock(variantColumns.map((column, index) => renderFooterColumn(column, index, realm)).join('\n'), '  '),
        '</div>'
      ].join('\n');
    }).join('\n');

  const utilityLinks = Array.isArray(footer.utilityLinks)
    ? footer.utilityLinks
    : (columns.find((column) => column && column.id === 'site')?.links || []);
  const renderedUtilityLinks = utilityLinks
    .map((link) => renderFooterLink(link))
    .join('\n');

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
    '    <div class="footer-identity">',
    indentBlock(renderedIdentity, '      '),
    '    </div>',
    '    <nav class="footer-nav" aria-label="Footer">',
    indentBlock(renderedNavPanels, '      '),
    '    </nav>',
    '    <div class="footer-bottom">',
    '      <nav class="footer-utility" aria-label="Footer utility">',
    indentBlock(renderedUtilityLinks, '        '),
    '      </nav>',
    `      <p class="footer-meta">© ${escapeHtml(year)} ${escapeHtml(footer.copyrightName || 'Daniel Short')}. All rights reserved.</p>`,
    '    </div>',
    '  </div>',
    '</footer>',
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
  const twitterCreator = String(page.twitterCreator || settings.twitterCreator || twitterSite).trim();
  const ownerName = String(settings.ownerName || settings.siteName || 'Daniel Short').trim();
  const locale = String(page.locale || settings.locale || 'en_US').trim();
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
    ownerName ? `  <meta name="author" content="${escapeHtml(ownerName)}">` : '',
    page.robots ? `  <meta name="robots" content="${escapeHtml(page.robots)}">` : '',
    `  <meta property="og:title" content="${escapeHtml(ogTitle)}">`,
    siteName ? `  <meta property="og:site_name" content="${escapeHtml(siteName)}">` : '',
    locale ? `  <meta property="og:locale" content="${escapeHtml(locale)}">` : '',
    `  <meta property="og:description" content="${escapeHtml(ogDescription)}">`,
    `  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">`,
    `  <meta property="og:image" content="${escapeHtml(ogImage.url || '')}">`,
    ogImage.width ? `  <meta property="og:image:width" content="${escapeHtml(ogImage.width)}">` : '',
    ogImage.height ? `  <meta property="og:image:height" content="${escapeHtml(ogImage.height)}">` : '',
    ogImage.type ? `  <meta property="og:image:type" content="${escapeHtml(ogImage.type)}">` : '',
    ogImage.alt ? `  <meta property="og:image:alt" content="${escapeHtml(ogImage.alt)}">` : '',
    `  <meta property="og:type" content="${escapeHtml(page.ogType || 'website')}">`,
    '  <meta name="twitter:card" content="summary_large_image">',
    twitterSite ? `  <meta name="twitter:site" content="${escapeHtml(twitterSite)}">` : '',
    twitterCreator ? `  <meta name="twitter:creator" content="${escapeHtml(twitterCreator)}">` : '',
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

function pillLabels(tool) {
  return uniqueLabels((Array.isArray(tool && tool.pills) ? tool.pills : [])
    .map((pill) => (typeof pill === 'string' ? pill : pill && pill.label)));
}

function toolAvailability(tool) {
  const tags = pillLabels(tool).map((tag) => slugify(tag));
  if (tags.includes('local')) return 'Local';
  if (tags.includes('cloud')) return 'Cloud';
  return 'Browser';
}

function toolAccess(tool) {
  const visibility = toolVisibility(tool);
  if (visibility === 'admin' || visibility === 'admins') return 'Admin';
  if (visibility === 'authed' || visibility === 'authenticated' || visibility === 'logged-in') return 'Account';
  return 'Public';
}

function buildToolsDirectoryWorkbenchData(page, tools) {
  const categories = sortByOrderThenTitle(Array.isArray(page && page.categories) ? page.categories : []);
  const directoryTools = (Array.isArray(tools) ? tools : []).filter(isDirectoryTool);
  const usedTools = new Set();
  const orderedTools = categories.flatMap((category) => {
    const items = sortByOrderThenTitle(directoryTools.filter((tool) => String(tool.categoryId || '') === String(category.id || '')));
    items.forEach((tool) => usedTools.add(tool));
    return items;
  }).concat(sortByOrderThenTitle(directoryTools.filter((tool) => !usedTools.has(tool))));
  const categoriesById = new Map(categories.map((category) => [String(category.id || ''), category]));
  const items = orderedTools.map((tool, index) => {
    const category = categoriesById.get(String(tool.categoryId || '')) || {};
    const tags = pillLabels(tool);
    const availability = toolAvailability(tool);
    const access = toolAccess(tool);
    const categoryTitle = category.title || 'Tools';
    return {
      id: String(tool.slug || tool.href || `tool-${index + 1}`).trim(),
      title: tool.title || 'Tool',
      subtitle: categoryTitle,
      summary: tool.summary || '',
      href: toolHref(tool),
      type: `${availability} Tool`,
      availability,
      access,
      category: categoryTitle,
      tags,
      tools: tags,
      concepts: [categoryTitle],
      formats: [`${availability} Tool`],
      results: tool.summary ? [tool.summary] : [],
      actions: [],
      privacy: String(tool && tool.privacy ? tool.privacy : '').trim(),
      inputs: uniqueLabels(Array.isArray(tool && tool.inputs) ? tool.inputs : []),
      outputs: uniqueLabels(Array.isArray(tool && tool.outputs) ? tool.outputs : []),
      iconImage: tool.iconImage ? trimLeadingSlash(tool.iconImage) : '',
      iconHtml: tool.iconImage ? '' : renderToolIconMarkup(tool),
      visibility: toolVisibility(tool),
      hidden: Boolean(tool.hidden),
      noindex: Boolean(tool.noindex),
      order: index + 1
    };
  });

  return {
    kind: 'tools',
    title: page.heroTitle || 'Tools',
    itemSingular: 'tool',
    itemPlural: 'tools',
    queryParam: 'tool',
    ctaLabel: 'Open tool',
    itemSignalPrefix: 'Tool',
    summaryTitle: 'What it does',
    privacyTitle: 'Privacy',
    accessTitle: 'Access',
    inputsOutputsTitle: 'Inputs & outputs',
    stackTitle: 'Tags',
    emptySelectionText: 'Choose a tool to see details.',
    filterGroups: [
      { id: 'category', title: 'Category', options: optionList(items.map((item) => item.category), 'category') },
      { id: 'availability', title: 'Availability', options: optionList(items.map((item) => item.availability), 'availability') },
      { id: 'access', title: 'Access', options: optionList(items.map((item) => item.access), 'access') }
    ].filter((group) => group.options.length),
    items
  };
}

function toolsByCategory(page, tools) {
  const categories = sortByOrderThenTitle(Array.isArray(page && page.categories) ? page.categories : []);
  const directoryTools = (Array.isArray(tools) ? tools : []).filter(isDirectoryTool);
  const usedTools = new Set();
  const groups = categories
    .map((category) => {
      const items = sortByOrderThenTitle(directoryTools.filter((tool) => String(tool.categoryId || '') === String(category.id || '')));
      items.forEach((tool) => usedTools.add(tool));
      return { category, items };
    })
    .filter((group) => group.items.length);

  const uncategorized = sortByOrderThenTitle(directoryTools.filter((tool) => !usedTools.has(tool)));
  if (uncategorized.length) {
    groups.push({
      category: {
        id: 'tools-other',
        title: 'Other tools',
        description: 'Additional focused utilities.'
      },
      items: uncategorized
    });
  }

  return groups;
}

function renderToolsResumePanel(page) {
  const resumePanel = page && page.resumePanel ? page.resumePanel : {};
  const kicker = resumePanel.kicker || 'Saved sessions';
  const title = resumePanel.title || 'Continue where you left off';
  return [
    '<section class="tools-resume-panel" data-tools-resume="panel" data-tools-auth-only aria-labelledby="tools-resume-title" hidden aria-hidden="true">',
    '  <div class="tools-resume-head">',
    '    <div class="tools-resume-head-copy">',
    `      <p class="tools-resume-kicker">${escapeHtml(kicker)}</p>`,
    `      <h2 class="tools-resume-title" id="tools-resume-title">${escapeHtml(title)}</h2>`,
    '    </div>',
    '  </div>',
    '  <p class="tools-resume-status" data-tools-resume="status" role="status" aria-live="polite"></p>',
    '  <div class="tools-resume-content" data-tools-resume="content"></div>',
    '</section>'
  ].join('\n');
}

function renderToolsDirectoryRail(groups) {
  const publicCount = groups.reduce((total, group) => total + group.items.filter(isPublicTool).length, 0);
  const links = groups.map((group, index) => {
    const category = group.category || {};
    const id = String(category.id || `tools-category-${index + 1}`).trim();
    const title = category.title || 'Tools';
    const description = category.description || '';
    const count = group.items.filter(isPublicTool).length;
    const attrs = {
      class: 'tools-rail-link',
      href: `#${id}`,
      'data-tools-category-link': id,
      hidden: count === 0 ? true : null,
      'aria-hidden': count === 0 ? 'true' : null
    };
    return [
      `  <a${attrsToString(attrs)}>`,
      `    <span class="tools-rail-index">${String(index + 1).padStart(2, '0')}</span>`,
      '    <span class="tools-rail-copy">',
      `      <span class="tools-rail-title">${escapeHtml(title)}</span>`,
      description ? `      <span class="tools-rail-description">${escapeHtml(description)}</span>` : '',
      '    </span>',
      `    <span class="tools-rail-count" data-tools-category-count>${escapeHtml(count)} ${count === 1 ? 'tool' : 'tools'}</span>`,
      '  </a>'
    ].filter(Boolean).join('\n');
  });

  return [
    '<nav class="tools-directory-rail" aria-label="Tool categories">',
    `  <p class="tools-directory-stat" data-tools-directory-stat>${escapeHtml(publicCount)} public tools</p>`,
    ...links,
    '</nav>'
  ].join('\n');
}

function renderToolsDirectoryNote(page) {
  const title = page && page.directoryTitle ? page.directoryTitle : 'Choose the workflow surface';
  const kicker = page && page.directoryKicker ? page.directoryKicker : 'Tool catalog';
  const description = page && page.directoryDescription ? page.directoryDescription : 'Grouped by the kind of repeated work each tool is meant to remove.';
  return [
    '<aside class="tools-directory-note" aria-labelledby="tools-directory-note-title">',
    `  <p class="tools-directory-note-kicker">${escapeHtml(kicker)}</p>`,
    `  <h2 id="tools-directory-note-title">${escapeHtml(title)}</h2>`,
    `  <p>${escapeHtml(description)}</p>`,
    '  <p>Public tools run without an account. Signed-in tools reveal private session history and account-backed workflows when available.</p>',
    '</aside>'
  ].join('\n');
}

function renderToolsDirectoryCard(tool) {
  const visibility = toolVisibility(tool);
  const hiddenByDefault = !isPublicTool(tool);
  const slug = String(tool && tool.slug ? tool.slug : '').trim();
  const title = tool && tool.title ? tool.title : 'Tool';
  const detailsId = `${slug || slugify(title)}-details`;
  const iconClass = hasToolIconImage(tool) ? 'tool-icon tool-icon-image' : 'tool-icon';
  const attrs = {
    class: 'tool-card',
    'data-tools-card': true,
    'data-tools-category-id': tool && tool.categoryId ? tool.categoryId : null,
    'data-tools-visibility': visibility === 'public' ? null : visibility,
    hidden: hiddenByDefault ? true : null,
    'aria-hidden': hiddenByDefault ? 'true' : null
  };

  return [
    `  <article${attrsToString(attrs)}>`,
    `    <a class="tool-launch-card" href="${escapeHtml(toolHref(tool))}" aria-describedby="${escapeHtml(detailsId)}">`,
    `      <span class="${escapeHtml(iconClass)}" aria-hidden="true">`,
    indentBlock(renderToolIconMarkup(tool), '        '),
    '      </span>',
    '      <span class="tool-card-main">',
    `        <span class="tool-card-title">${escapeHtml(title)}</span>`,
    '      </span>',
    `      <span class="tool-card-details" id="${escapeHtml(detailsId)}" role="tooltip">`,
    `        <span class="tool-card-summary">${escapeHtml(tool && tool.summary ? tool.summary : '')}</span>`,
    '      </span>',
    '    </a>',
    '  </article>'
  ].join('\n');
}

function renderToolsDirectoryCategory(group) {
  const category = group.category || {};
  const id = String(category.id || '').trim();
  const title = category.title || 'Tools';
  const description = category.description || '';
  const hiddenByDefault = !group.items.some(isPublicTool);
  const attrs = {
    class: 'tools-category',
    id,
    'aria-labelledby': `${id}-title`,
    'data-tools-category': true,
    'data-tools-category-title': title,
    hidden: hiddenByDefault ? true : null
  };

  return [
    `<section${attrsToString(attrs)}>`,
    '  <header class="tools-category-head">',
    `    <h2 class="section-title" id="${escapeHtml(id)}-title">${escapeHtml(title)}</h2>`,
    description ? `    <p>${escapeHtml(description)}</p>` : '',
    '  </header>',
    '  <div class="tools-grid">',
    group.items.map(renderToolsDirectoryCard).join('\n'),
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function buildGamesDirectoryWorkbenchData(page) {
  const games = sortByOrderThenTitle((Array.isArray(page && page.games) ? page.games : [])
    .filter((game) => game && !game.hidden && !game.noindex && (game.href || game.id)));
  const items = games.map((game, index) => {
    const tags = uniqueLabels(Array.isArray(game.tags) ? game.tags : []);
    const isSimulation = tags.some((tag) => /simulation|sandbox|canvas|probability/i.test(tag))
      || /simulation|sandbox/i.test(game.summary || '');
    const type = isSimulation ? 'Simulation' : 'Browser Game';
    return {
      id: String(game.id || game.href || `game-${index + 1}`).trim(),
      title: game.title || 'Game',
      subtitle: type,
      summary: game.summary || '',
      href: gameHref(game),
      type,
      category: type,
      tags,
      tools: tags,
      concepts: tags,
      formats: [type],
      results: game.summary ? [game.summary] : [],
      actions: tags.length ? [`Focus areas: ${tags.join(', ')}`] : [],
      iconHtml: renderGameIconMarkup(game.iconType),
      order: index + 1
    };
  });

  return {
    kind: 'games',
    title: page.heroTitle || 'Games',
    itemSingular: 'game',
    itemPlural: 'games',
    queryParam: 'game',
    ctaLabel: 'Play game',
    itemSignalPrefix: 'Game',
    summaryTitle: 'Overview',
    highlightsTitle: 'Play loop',
    approachTitle: 'System focus',
    stackTitle: 'Tags',
    emptySelectionText: 'Choose a game to see details.',
    filterGroups: [
      { id: 'format', title: 'Format', options: optionList(items.map((item) => item.type), 'type') },
      { id: 'tag', title: 'Tags', options: optionList(items.flatMap((item) => item.tags), 'tags') }
    ].filter((group) => group.options.length),
    items
  };
}

function renderDirectoryDataJs(data) {
  return [
    '/* Generated by build/generate-cms-artifacts.js. Do not edit directly. */',
    `window.DIRECTORY_WORKBENCH = ${JSON.stringify(data, null, 2)};`,
    ''
  ].join('\n');
}

function renderDirectoryWorkbenchStaticResults(items, kind) {
  const safeKind = String(kind || '').trim().toLowerCase();
  const publicItems = (Array.isArray(items) ? items : []).filter((item) => {
    if (!item || item.hidden || item.noindex) return false;
    if (safeKind !== 'tools') return true;
    return String(item.visibility || 'public').trim().toLowerCase() === 'public';
  });

  return publicItems.map((item) => {
    const title = String(item.title || (safeKind === 'games' ? 'Game' : 'Tool')).trim();
    const summary = String(item.summary || '').trim();
    const href = String(item.href || '').trim();
    const media = item.iconImage
      ? `<span class="portfolio-result-card__icon"><img src="${escapeHtml(item.iconImage)}" alt="" width="256" height="256" loading="lazy" decoding="async"></span>`
      : String(item.iconHtml || `<span class="portfolio-result-card__initial">${escapeHtml(title.charAt(0) || '?')}</span>`);
    const cardBody = [
      '<span class="portfolio-result-card__body">',
      `  <span class="portfolio-result-card__title">${escapeHtml(title)}</span>`,
      summary ? `  <span class="portfolio-result-card__summary">${escapeHtml(summary)}</span>` : '',
      '</span>'
    ].filter(Boolean).join('\n');

    if (safeKind === 'tools') {
      return [
        `<article class="portfolio-result-card tools-workbench-result" role="listitem" data-project-id="${escapeHtml(item.id || '')}" data-tools-visibility="public">`,
        `  <a class="tools-workbench-result__select" href="${escapeHtml(href)}" aria-label="Open ${escapeHtml(title)}">`,
        `    <span class="portfolio-result-card__media portfolio-result-card__media--icon" aria-hidden="true">${media}</span>`,
        indentBlock(cardBody, '    '),
        '  </a>',
        `  <a class="tools-workbench-result__open" href="${escapeHtml(href)}" aria-label="Open ${escapeHtml(title)}"><span>Open</span><span aria-hidden="true">&rarr;</span></a>`,
        '</article>'
      ].join('\n');
    }

    return [
      `<a class="portfolio-result-card" role="listitem" href="${escapeHtml(href)}" data-project-id="${escapeHtml(item.id || '')}">`,
      `  <span class="portfolio-result-card__media portfolio-result-card__media--icon" aria-hidden="true">${media}</span>`,
      indentBlock(cardBody, '  '),
      '</a>'
    ].join('\n');
  }).join('\n');
}

function renderToolsWorkbenchHeader(page, publicCount) {
  const title = page && page.heroTitle ? page.heroTitle : 'Tools';
  const lead = page && page.heroLead ? page.heroLead : 'Focused utilities for writing, campaigns, and media.';
  const statusLabel = page && page.statusLabel ? page.statusLabel : 'Local-first';
  return [
    '<header class="portfolio-workbench__header tools-workbench-header">',
    '  <div class="tools-workbench-header__identity" aria-hidden="true">',
    '    <img src="img/brand/00-ds-logo-master-full-color.svg" alt="" width="48" height="48" decoding="async">',
    '  </div>',
    '  <div class="portfolio-workbench__title-block">',
    `    <h1>${escapeHtml(title)}</h1>`,
    `    <p class="tools-workbench-header__description">${escapeHtml(lead)}</p>`,
    '  </div>',
    '  <div class="tools-workbench-header__actions">',
    '    <p class="tools-workbench-header__status">',
    `      <span data-tools-directory-stat>${escapeHtml(publicCount)} public tools</span>`,
    '      <span aria-hidden="true">&middot;</span>',
    `      <span>${escapeHtml(statusLabel)}</span>`,
    '    </p>',
    '    <div class="tools-account-dock tools-account-dock--directory" data-tools-account="dock">',
    '      <div class="tools-account-dock-inner" data-tools-account="dock-inner">',
    '        <div class="tools-account-bar" data-tools-account="bar"></div>',
    '      </div>',
    '    </div>',
    '  </div>',
    '</header>'
  ].join('\n');
}

function renderDirectoryWorkbenchBody(page, options = {}) {
  const kind = String(options.kind || page.id || 'directory').trim();
  const itemSingular = String(options.itemSingular || 'item').trim();
  const itemPlural = String(options.itemPlural || `${itemSingular}s`).trim();
  const title = String(options.title || page.heroTitle || page.title || 'Library').trim();
  const resultsId = `${kind}-results-title`;
  const displaySingular = `${itemSingular.charAt(0).toUpperCase()}${itemSingular.slice(1)}`;
  const headerHtml = options.headerHtml || [
    '<header class="portfolio-workbench__header">',
    '  <div class="portfolio-workbench__title-block">',
    `    <h1>${escapeHtml(title)}</h1>`,
    '  </div>',
    '</header>'
  ].join('\n');
  const supplementalHtml = String(options.supplementalHtml || '').trim();
  const fallbackHtml = String(options.fallbackHtml || '').trim();
  const initialItemsHtml = String(options.initialItemsHtml || '').trim();
  const initialResultsText = String(options.initialResultsText || `Loading ${itemPlural}...`).trim();
  const supplementalLines = supplementalHtml ? [indentBlock(supplementalHtml, '      ')] : [];
  const fallbackLines = fallbackHtml ? [indentBlock(fallbackHtml, '          ')] : [];
  const initialItemLines = initialItemsHtml ? [indentBlock(initialItemsHtml, '            ')] : [];
  const accountDock = options.accountDock
    ? [
      '  <div class="tools-account-dock tools-account-dock--directory" data-tools-account="dock">',
      '    <div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner"></div>',
      '  </div>'
    ]
    : [];

  return [
    '<main id="main" class="portfolio-main directory-workbench-main">',
    ...accountDock,
    `  <section class="portfolio-workbench" id="${escapeHtml(kind)}-workbench" data-portfolio-workbench data-directory-workbench="${escapeHtml(kind)}" aria-label="${escapeHtml(title)}">`,
    '    <div class="portfolio-workbench__shell">',
    indentBlock(headerHtml, '      '),
    ...supplementalLines,
    '',
    '      <div class="portfolio-workbench__layout">',
    `        <aside class="portfolio-workbench__filters" aria-label="${escapeHtml(displaySingular)} filters">`,
    '          <div class="portfolio-filter-head">',
    '            <h2>Filters</h2>',
    '            <button type="button" class="portfolio-filter-clear" data-portfolio-clear-filters>Clear all</button>',
    '          </div>',
    '          <div class="portfolio-filter-groups" data-portfolio-filters></div>',
    '        </aside>',
    '',
    `        <section class="portfolio-workbench__results" aria-labelledby="${escapeHtml(resultsId)}">`,
    '          <div class="portfolio-results-toolbar">',
    '            <div>',
    `              <h2 id="${escapeHtml(resultsId)}" class="visually-hidden">${escapeHtml(title)} results</h2>`,
    `              <p class="portfolio-results-count" data-portfolio-results-count>${escapeHtml(initialResultsText)}</p>`,
    '            </div>',
    '            <label class="portfolio-sort-control">',
    '              <span>Sort by:</span>',
    `              <select data-portfolio-sort aria-label="Sort ${escapeHtml(itemPlural)}">`,
    '                <option value="default">Default</option>',
    '                <option value="title">Alphabetical</option>',
    '              </select>',
    '            </label>',
    '          </div>',
    '          <div class="portfolio-search">',
    `            <label class="visually-hidden" for="${escapeHtml(kind)}-search-input">Search ${escapeHtml(itemPlural)}</label>`,
    `            <input id="${escapeHtml(kind)}-search-input" type="search" placeholder="Search ${escapeHtml(itemPlural)}" autocomplete="off" data-portfolio-search>`,
    '          </div>',
    '          <div class="portfolio-results-list" role="list" data-portfolio-results>',
    ...initialItemLines,
    '          </div>',
    `          <p class="portfolio-empty-state" data-portfolio-empty hidden>No ${escapeHtml(itemPlural)} match those filters.</p>`,
    ...fallbackLines,
    '        </section>',
    '',
    `        <aside class="portfolio-inspector" aria-label="Selected ${escapeHtml(itemSingular)} details" aria-live="polite" data-portfolio-inspector>`,
    `          <div class="portfolio-inspector__loading">Choose a ${escapeHtml(itemSingular)} to see details.</div>`,
    '        </aside>',
    '      </div>',
    '    </div>',
    '  </section>',
    '</main>'
  ].join('\n');
}

function renderToolsDirectoryBody(page, tools) {
  const data = buildToolsDirectoryWorkbenchData(page, tools);
  const publicCount = data.items.filter((item) => item.visibility === 'public').length;
  return renderDirectoryWorkbenchBody(page, {
    kind: 'tools',
    title: page.heroTitle || 'Tools',
    itemSingular: 'tool',
    itemPlural: 'tools',
    headerHtml: renderToolsWorkbenchHeader(page, publicCount),
    supplementalHtml: renderToolsResumePanel(page),
    initialResultsText: `${publicCount} tools`,
    initialItemsHtml: renderDirectoryWorkbenchStaticResults(data.items, 'tools')
  });
}

function renderGamesDirectoryBody(page) {
  const data = buildGamesDirectoryWorkbenchData(page);
  return renderDirectoryWorkbenchBody(page, {
    kind: 'games',
    title: page.heroTitle || 'Games',
    itemSingular: 'game',
    itemPlural: 'games',
    initialResultsText: `${data.items.length} games`,
    initialItemsHtml: renderDirectoryWorkbenchStaticResults(data.items, 'games')
  });
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
  buildGamesDirectoryWorkbenchData,
  buildToolsDirectoryWorkbenchData,
  renderDirectoryDataJs,
  renderAudienceConfigJs,
  renderFooter,
  renderFullPage,
  renderGamesDirectoryBody,
  renderHeader,
  renderProjectsDataJs,
  renderToolsDirectoryBody
};
