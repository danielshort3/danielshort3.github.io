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

function renderHeader({ settings, navigation, projectsById, audienceLabel }) {
  const brand = navigation.brand || {};
  const portfolio = navigation.portfolio || {};
  const resume = navigation.resume || {};
  const contact = navigation.contact || {};
  const search = navigation.search || {};
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

  const portfolioFooterLinks = (Array.isArray(portfolio.links) ? portfolio.links : [])
    .map((link) => [
      `<a href="${escapeHtml(trimLeadingSlash(link.href || ''))}" class="nav-dropdown-link nav-dropdown-all"${attrsToString(link.dataAttributes || {})}>`,
      `  <span class="nav-dropdown-title">${escapeHtml(link.title || '')}</span>`,
      `  <span class="nav-dropdown-subtitle">${escapeHtml(link.subtitle || '')}</span>`,
      '</a>'
    ].join('\n'))
    .join('\n');

  return [
    '<header id="combined-header-nav">',
    '  <nav class="nav" aria-label="Primary">',
    '    <div class="wrapper nav-wrapper">',
    `      <a href="${escapeHtml(normalizeRelativeHref(brand.homePath || '/', '/analytics'))}" class="brand" aria-label="Home" data-entry-home-link="true">`,
    `        <img src="${escapeHtml(brand.logoSrc || 'img/ui/logo-64.png')}" srcset="${escapeHtml(brand.logoSrcSet || 'img/ui/logo-64.png 1x, img/ui/logo-192.png 3x')}" sizes="${escapeHtml(brand.logoSizes || '64px')}" alt="${escapeHtml(brand.logoAlt || 'DS logo')}" class="brand-logo" decoding="async" loading="eager" width="${escapeHtml(brand.logoWidth || 64)}" height="${escapeHtml(brand.logoHeight || 64)}">`,
    '        <span class="brand-name">',
    `          <span class="brand-title">${escapeHtml(settings.ownerName || 'Daniel Short')}</span>`,
    '          <span class="brand-divider" aria-hidden="true"></span>',
    '          <span class="brand-tagline">',
    `            <span class="brand-tagline-chunk" data-brand-tagline-primary="true">${escapeHtml(audienceLabel || brand.defaultTagline || 'Data Analytics')}</span>`,
    '          </span>',
    '        </span>',
    '      </a>',
    '      <button id="nav-toggle" class="burger" aria-label="Toggle navigation" aria-expanded="false" aria-controls="primary-menu">',
    '        <span class="bar"></span><span class="bar"></span><span class="bar"></span>',
    '      </button>',
    '      <div id="primary-menu" class="nav-row" data-collapsible role="navigation">',
    '        <a href="/" class="nav-link" data-entry-home-link="true">Home</a>',
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
    '        </div>',
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
          href: normalizeRelativeHref(href, label === 'home' ? '/analytics' : ''),
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

function renderFullPage({ settings, navigation, footer, projectsById, page, audienceLabel }) {
  const headerHtml = renderHeader({
    settings,
    navigation,
    projectsById,
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

  const toolCount = usableTools.filter((tool) => !tool.hidden && (!tool.visibility || tool.visibility === 'public')).length;

  const renderedCategoryNav = toolsByCategory.map(({ category }) => {
    const id = String(category.id || '').trim();
    if (!id) return '';
    return `<a class="tools-nav-link" href="#${escapeHtml(id)}" data-tools-category-link="${escapeHtml(id)}">${escapeHtml(category.title || '')}</a>`;
  }).filter(Boolean).join('\n');

  const renderedCategories = toolsByCategory.map(({ category, items }) => {
    const cards = items.map((tool) => {
      const href = trimLeadingSlash(tool.href || `tools/${tool.slug}`);
      const cardAttrs = {
        class: 'tool-card',
        'data-tools-card': true,
        'data-tools-category-id': category.id || '',
        'data-tools-category-title': category.title || '',
        ...(tool.visibility && tool.visibility !== 'public' ? { 'data-tools-visibility': tool.visibility } : {}),
        ...(tool.hidden ? { hidden: true } : {})
      };
      const pills = (Array.isArray(tool.pills) ? tool.pills : [])
        .map((pill) => `<span class="tool-pill${pill && pill.variant === 'local' ? ' tool-pill-local' : ''}">${escapeHtml(pill && pill.label ? pill.label : '')}</span>`)
        .join('\n');
      return [
        `<article${attrsToString(cardAttrs)}>`,
        `  <a href="${escapeHtml(href)}">`,
        '    <div class="tool-top">',
        '      <span class="tool-icon" aria-hidden="true">',
        indentBlock(String(tool.iconHtml || '').trim(), '        '),
        '      </span>',
        '      <div>',
        `        <h3>${escapeHtml(tool.title || '')}</h3>`,
        '        <div class="tool-meta">',
        indentBlock(pills, '          '),
        '        </div>',
        '      </div>',
        '    </div>',
        `    <p>${escapeHtml(tool.summary || '')}</p>`,
        '  </a>',
        '</article>'
      ].join('\n');
    }).join('\n\n');

    return [
      `<section class="tools-category" id="${escapeHtml(category.id || '')}" aria-labelledby="${escapeHtml(category.id || '')}-title" data-tools-category data-tools-category-title="${escapeHtml(category.title || '')}">`,
      '  <header class="tools-category-head">',
      `    <h2 class="section-title" id="${escapeHtml(category.id || '')}-title">${escapeHtml(category.title || '')}</h2>`,
      `    <p>${escapeHtml(category.description || '')}</p>`,
      '  </header>',
      '  <div class="tools-grid">',
      indentBlock(cards, '    '),
      '  </div>',
      '</section>'
    ].join('\n');
  }).join('\n\n');

  const resumePanel = page.resumePanel || {};
  const filter = page.filter || {};
  return [
    '<section class="hero hero--tools tools-hero">',
    '  <div class="wrapper">',
    '    <div class="tools-hero-copy">',
    `      <p class="hero-eyebrow">${escapeHtml(page.heroEyebrow || 'Tools')}</p>`,
    `      <h1>${escapeHtml(page.heroTitle || 'Tools')}</h1>`,
    `      <p class="tools-hero-lead">${escapeHtml(page.heroLead || 'Fast utilities for writing, marketing, media, and reporting workflows.')}</p>`,
    '    </div>',
    '  </div>',
    '</section>',
    '',
    '<div class="tools-account-dock" data-tools-account="dock">',
    '  <div class="wrapper tools-account-dock-inner" data-tools-account="dock-inner">',
    '    <div class="tools-account-bar" data-tools-account="bar"></div>',
    '  </div>',
    '</div>',
    '',
    '<main id="main">',
    '  <section class="surface-band tools-section">',
    '    <div class="wrapper">',
    '      <section class="tools-directory-controls" aria-labelledby="tools-directory-title">',
    '        <div class="tools-directory-head">',
    `          <p class="tools-directory-kicker">${escapeHtml(page.directoryKicker || 'Directory')}</p>`,
    `          <h2 id="tools-directory-title">${escapeHtml(page.directoryTitle || 'Find a tool')}</h2>`,
    `          <p>${escapeHtml(page.directoryDescription || 'Search by workflow, format, or category, then jump into the tool you need.')}</p>`,
    '        </div>',
    '        <div class="tools-filter" role="search">',
    '          <label class="visually-hidden" for="tools-filter-query">Search tools</label>',
    `          <input id="tools-filter-query" class="tools-filter-input" type="search" autocomplete="off" spellcheck="false" placeholder="${escapeHtml(filter.placeholder || 'Search tools by name, task, or tag')}" data-tools-filter-input>`,
    `          <button class="btn-secondary tools-filter-clear" type="button" data-tools-filter-clear>${escapeHtml(filter.clearLabel || 'Clear')}</button>`,
    '        </div>',
    `        <p class="tools-filter-status" role="status" aria-live="polite" data-tools-filter-status>Showing ${escapeHtml(toolCount)} tools.</p>`,
    '        <nav class="tools-nav" aria-label="Tool categories">',
    indentBlock(renderedCategoryNav, '          '),
    '        </nav>',
    '      </section>',
    '',
    '      <section class="tools-resume-panel" data-tools-resume="panel" aria-labelledby="tools-resume-title" hidden>',
    '        <div class="tools-resume-head">',
    '          <div class="tools-resume-head-copy">',
    `            <p class="tools-resume-kicker">${escapeHtml(resumePanel.kicker || '')}</p>`,
    `            <h2 class="tools-resume-title" id="tools-resume-title">${escapeHtml(resumePanel.title || '')}</h2>`,
    '          </div>',
    `          <a class="btn-secondary tools-resume-dashboard-link" href="${escapeHtml(trimLeadingSlash(resumePanel.dashboardHref || 'tools-dashboard'))}">${escapeHtml(resumePanel.dashboardLabel || 'Open dashboard')}</a>`,
    '        </div>',
    '        <p class="tools-resume-status" data-tools-resume="status" role="status" aria-live="polite"></p>',
    '        <div class="tools-resume-content" data-tools-resume="content"></div>',
    '      </section>',
    '',
    '      <div id="tools-directory-results" data-tools-results>',
    indentBlock(renderedCategories, '        '),
    '      </div>',
    '      <p class="tools-empty-state" data-tools-empty-state hidden>No tools match that search.</p>',
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
  const defaultAudience = settings.defaultAudience || order[0] || 'analytics';

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
  renderHeader,
  renderProjectsDataJs,
  renderToolsDirectoryBody
};
