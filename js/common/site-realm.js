(() => {
  'use strict';

  const STORAGE_KEY = 'siteRealm';
  const PROFESSIONAL_MODE = 'professional';
  const PERSONAL_MODE = 'personal';
  const LEGACY_AUDIENCE = 'analytics';
  const LEGACY_ANALYTICS_PATHS = new Set([
    '/resume',
    '/resume-pdf'
  ]);

  const audienceApi = window.SITE_AUDIENCE_CONFIG || {};
  const audiences = audienceApi.audiences || {};
  const normalizeAudience = typeof audienceApi.normalizeAudience === 'function'
    ? audienceApi.normalizeAudience
    : (value) => String(value || '').trim().toLowerCase() || PERSONAL_MODE;
  const getAudience = typeof audienceApi.getAudience === 'function'
    ? audienceApi.getAudience
    : (value) => audiences[normalizeAudience(value)] || audiences.personal || { key: PERSONAL_MODE, homePath: '/' };
  const detectAudienceFromPath = typeof audienceApi.detectAudienceFromPath === 'function'
    ? audienceApi.detectAudienceFromPath
    : () => null;

  const clearStoredRealm = () => {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {}
  };

  const normalizeMode = (value) => {
    const raw = String(value || '').trim().toLowerCase();
    if (['professional', 'work', 'career', 'analytics'].includes(raw)) return PROFESSIONAL_MODE;
    if (['personal', 'hobby', 'default'].includes(raw)) return PERSONAL_MODE;
    return '';
  };

  const currentPath = () => {
    try {
      return (window.location.pathname || '/').replace(/\/index\.html$/i, '/') || '/';
    } catch {
      return '/';
    }
  };

  const normalizedPath = () => currentPath().replace(/\.html$/i, '') || '/';

  const readQuery = () => {
    try {
      return new URLSearchParams(window.location.search || '');
    } catch {
      return new URLSearchParams();
    }
  };

  const queryMode = () => normalizeMode(readQuery().get('mode'));

  const queryAudience = () => {
    const raw = String(readQuery().get('audience') || '').trim();
    if (!raw) return '';
    const normalized = normalizeAudience(raw);
    return audiences[normalized] ? normalized : '';
  };

  const pathAudience = () => {
    const path = normalizedPath();
    if (LEGACY_ANALYTICS_PATHS.has(path)) return LEGACY_AUDIENCE;
    return detectAudienceFromPath(path) || '';
  };

  const bodyAudience = () => {
    const raw = String(document.body?.dataset?.audience || '').trim();
    if (!raw) return '';
    const normalized = normalizeAudience(raw);
    return audiences[normalized] ? normalized : '';
  };

  const canonicalizeLegacyMode = () => {
    const mode = queryMode();
    if (!mode || currentPath() === '/') return;

    const url = new URL(window.location.href);
    if (mode === PROFESSIONAL_MODE && !url.searchParams.get('audience')) {
      url.searchParams.set('audience', LEGACY_AUDIENCE);
    }
    url.searchParams.delete('mode');
    const next = `${url.pathname}${url.search}${url.hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== current) {
      window.history.replaceState(window.history.state, '', next);
    }
  };

  const redirectLegacyRoot = () => {
    if (currentPath() !== '/' || queryMode() !== PROFESSIONAL_MODE) return false;
    const analytics = getAudience(LEGACY_AUDIENCE);
    window.location.replace(analytics.homePath || '/analytics');
    return true;
  };

  const detectAudience = () => {
    clearStoredRealm();
    return queryAudience()
      || pathAudience()
      || bodyAudience()
      || (queryMode() === PROFESSIONAL_MODE ? LEGACY_AUDIENCE : PERSONAL_MODE);
  };

  const applyProfessionalRobots = (isProfessional) => {
    const selector = 'meta[name="robots"][data-site-realm-robots="professional"]';
    const dynamicRobots = document.head?.querySelector(selector);
    if (!isProfessional) {
      dynamicRobots?.remove();
      return;
    }

    const staticNoindex = Array.from(document.head?.querySelectorAll('meta[name="robots"]') || [])
      .some((meta) => meta !== dynamicRobots && /(?:^|,)\s*noindex\b/i.test(meta.getAttribute('content') || ''));
    if (staticNoindex) {
      dynamicRobots?.remove();
      return;
    }

    const robots = dynamicRobots || document.createElement('meta');
    robots.setAttribute('name', 'robots');
    robots.setAttribute('content', 'noindex, nofollow');
    robots.dataset.siteRealmRobots = PROFESSIONAL_MODE;
    if (!dynamicRobots) document.head?.appendChild(robots);
  };

  const setDocumentRealm = (audienceKey) => {
    const audience = getAudience(audienceKey);
    const key = normalizeAudience(audience && audience.key);
    const isProfessional = key !== PERSONAL_MODE;
    const mode = isProfessional ? PROFESSIONAL_MODE : PERSONAL_MODE;

    document.documentElement.classList.toggle('site-realm-professional', isProfessional);
    document.documentElement.classList.toggle('site-realm-personal', !isProfessional);
    document.documentElement.classList.remove('site-realm-query-pending');
    document.documentElement.classList.remove('site-realm-professional-home');
    if (document.body) {
      document.body.dataset.siteRealm = mode;
      document.body.dataset.audience = key;
      delete document.body.dataset.siteRealmHome;
      document.body.classList.remove('professional-home-page');
    }

    window.SITE_REALM = mode;
    window.SITE_AUDIENCE = key;
    window.getSiteRealm = () => window.SITE_REALM || PERSONAL_MODE;
    window.getSiteAudience = () => window.SITE_AUDIENCE || PERSONAL_MODE;
    window.isProfessionalRealm = () => window.getSiteRealm() === PROFESSIONAL_MODE;
    applyProfessionalRobots(isProfessional);
    return audience;
  };

  const escapeHtml = (value) => String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

  const trimLeadingSlash = (value) => String(value || '').replace(/^\/+/, '');

  const buildResumeNavHtml = (audience) => {
    const resumePath = trimLeadingSlash(audience.resumePath);
    const previewPath = trimLeadingSlash(audience.resumePreviewPath);
    const downloadPath = trimLeadingSlash(audience.resumeDownloadPath);
    return `
      <div class="nav-item nav-item-resume">
        <a href="${escapeHtml(resumePath)}" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-resume" data-resume-home-link="true">
          ${escapeHtml(audience.resumeNavTitle || 'Resume')}
          <span class="nav-link-caret" aria-hidden="true"></span>
        </a>
        <div class="nav-dropdown nav-dropdown-simple" id="nav-dropdown-resume" aria-label="Resume shortcuts">
          <div class="nav-dropdown-inner nav-dropdown-inner-simple">
            <div class="nav-dropdown-column nav-dropdown-column-list">
              <div class="nav-dropdown-header" aria-hidden="true">Resume shortcuts</div>
              <div class="nav-dropdown-list" role="list">
                <a href="${escapeHtml(resumePath)}" class="nav-dropdown-link" role="listitem" data-resume-home-link="true">
                  <span class="nav-dropdown-title">${escapeHtml(audience.resumeNavTitle || 'Resume')}</span>
                  <span class="nav-dropdown-subtitle">${escapeHtml(audience.resumeNavSubtitle || 'View the digital resume')}</span>
                </a>
                <a href="${escapeHtml(previewPath)}" class="nav-dropdown-link" role="listitem" data-resume-preview-link="true">
                  <span class="nav-dropdown-title">Preview PDF</span>
                  <span class="nav-dropdown-subtitle">${escapeHtml(audience.resumePreviewSubtitle || 'Open the PDF preview')}</span>
                </a>
                <a href="${escapeHtml(downloadPath)}" class="nav-dropdown-link" role="listitem" download data-resume-download-link="true">
                  <span class="nav-dropdown-title">Download Resume</span>
                  <span class="nav-dropdown-subtitle">${escapeHtml(audience.resumeDownloadSubtitle || 'Download the PDF')}</span>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  };

  const setNavLinkLabel = (link, label) => {
    if (!link) return;
    const caret = link.querySelector('.nav-link-caret');
    link.textContent = '';
    link.append(document.createTextNode(label));
    if (caret) link.append(document.createTextNode('\n            '), caret);
  };

  const applyAudienceNavigation = (audience) => {
    if (!audience || normalizeAudience(audience.key) === PERSONAL_MODE) return;
    const header = document.getElementById('combined-header-nav');
    const menu = header?.querySelector('#primary-menu');
    if (!header || !menu) return;

    header.querySelectorAll('[data-entry-home-link="true"], [data-audience-home-link="true"]')
      .forEach((link) => link.setAttribute('href', audience.homePath || '/'));
    menu.querySelectorAll('.nav-item-tools, .nav-item-games').forEach((item) => item.remove());
    menu.querySelector('.nav-search')?.remove();

    let homeLink = menu.querySelector('[data-professional-home-link="true"]');
    if (!homeLink) {
      homeLink = document.createElement('a');
      homeLink.className = 'nav-link';
      homeLink.textContent = 'Home';
      homeLink.dataset.professionalHomeLink = 'true';
      menu.insertBefore(homeLink, menu.firstElementChild);
    }
    homeLink.setAttribute('href', audience.homePath || '/');

    const portfolioItem = menu.querySelector('.nav-item-portfolio');
    const portfolioLink = portfolioItem?.querySelector(':scope > .nav-link');
    setNavLinkLabel(portfolioLink, 'Portfolio');
    portfolioItem?.querySelectorAll('[data-portfolio-home-link="true"], [data-portfolio-default-link="true"]')
      .forEach((link) => link.setAttribute('href', audience.portfolioPath || '/portfolio'));
    const expectedFeaturedIds = new Set(Array.isArray(audience.featuredProjectIds) ? audience.featuredProjectIds : []);
    const renderedFeaturedIds = new Set(Array.from(portfolioItem?.querySelectorAll('[data-project-id]') || [])
      .map((card) => card.dataset.projectId)
      .filter(Boolean));
    const hasAudienceFeaturedSet = expectedFeaturedIds.size > 0
      && [...expectedFeaturedIds].every((id) => renderedFeaturedIds.has(id));
    if (portfolioItem && !hasAudienceFeaturedSet) {
      portfolioItem.querySelector('.nav-dropdown')?.remove();
      portfolioLink?.querySelector('.nav-link-caret')?.remove();
      portfolioLink?.classList.remove('nav-link-has-menu');
      portfolioLink?.removeAttribute('aria-haspopup');
      portfolioLink?.removeAttribute('aria-expanded');
      portfolioLink?.removeAttribute('aria-controls');
    }

    let resumeItem = menu.querySelector('.nav-item-resume');
    if (resumeItem) resumeItem.remove();
    const contactItem = menu.querySelector('.nav-item-contact');
    if (audience.resumePath && contactItem) {
      contactItem.insertAdjacentHTML('beforebegin', buildResumeNavHtml(audience).trim());
      resumeItem = menu.querySelector('.nav-item-resume');
    }

    const contactPath = audience.contactPath || `/contact?audience=${encodeURIComponent(audience.key)}`;
    const contactLink = contactItem?.querySelector(':scope > .nav-link');
    if (contactLink) contactLink.setAttribute('href', contactPath);
    const firstContactLink = contactItem?.querySelector('.nav-dropdown-list .nav-dropdown-link');
    const contactTitle = firstContactLink?.querySelector('.nav-dropdown-title');
    const contactSubtitle = firstContactLink?.querySelector('.nav-dropdown-subtitle');
    if (contactTitle) contactTitle.textContent = 'Message about a role';
    if (contactSubtitle) contactSubtitle.textContent = `Best for ${audience.label || audience.shortLabel || 'professional'} opportunities`;

    header.dataset.siteRealmNav = audience.key;
  };

  const applyAudienceFooter = (audience) => {
    if (!audience || normalizeAudience(audience.key) === PERSONAL_MODE) return;
    const footer = document.querySelector('.footer.footer-classic');
    if (!footer || footer.dataset.audience === audience.key) return;

    const contactPath = audience.contactPath || `/contact?audience=${encodeURIComponent(audience.key)}`;
    const contactModalPath = `${contactPath.replace(/#.*$/, '')}#contact-modal`;
    const identity = footer.querySelector('.footer-identity');
    const nav = footer.querySelector('.footer-nav');
    const externalLinks = Array.from(footer.querySelectorAll('.footer-nav a.footer-link'))
      .filter((link) => /^(?:mailto:|https?:)/i.test(link.getAttribute('href') || ''))
      .map((link) => link.cloneNode(true));

    if (identity) {
      identity.innerHTML = `
        <section class="footer-identity-panel" data-footer-realm="professional" aria-label="Daniel Short footer summary">
          <h2 class="footer-identity-name">Daniel Short</h2>
          <p class="footer-identity-summary">${escapeHtml(audience.label || audience.shortLabel || 'Professional')} portfolio</p>
          <div class="footer-identity-actions">
            <a href="${escapeHtml(trimLeadingSlash(audience.resumePath || ''))}" class="footer-link footer-identity-link" data-resume-home-link="true">${escapeHtml(audience.resumeNavTitle || 'Resume')}</a>
            <a href="${escapeHtml(trimLeadingSlash(contactModalPath))}" class="footer-link footer-identity-link" data-contact-modal-link="true">Contact</a>
          </div>
        </section>
      `.trim();
    }

    if (nav) {
      nav.innerHTML = `
        <div class="footer-nav-panel" data-footer-realm="professional">
          <section class="footer-col" aria-labelledby="footer-professional-work">
            <h2 class="footer-col-title" id="footer-professional-work">Work</h2>
            <a href="${escapeHtml(trimLeadingSlash(audience.homePath || '/'))}" class="footer-link">Home</a>
            <a href="${escapeHtml(trimLeadingSlash(audience.portfolioPath || '/portfolio'))}" class="footer-link" data-portfolio-home-link="true">Portfolio</a>
            <a href="${escapeHtml(trimLeadingSlash(audience.resumePath || ''))}" class="footer-link" data-resume-home-link="true">${escapeHtml(audience.resumeNavTitle || 'Resume')}</a>
          </section>
          <section class="footer-col" aria-labelledby="footer-professional-connect">
            <h2 class="footer-col-title" id="footer-professional-connect">Connect</h2>
            <a href="${escapeHtml(trimLeadingSlash(contactModalPath))}" class="footer-link" data-contact-modal-link="true">Contact</a>
          </section>
        </div>
      `.trim();
      const connect = nav.querySelector('#footer-professional-connect')?.closest('.footer-col');
      externalLinks.forEach((link) => connect?.appendChild(link));
    }

    footer.querySelectorAll('.footer-utility a[href*="sitemap"]').forEach((link) => link.remove());
    document.querySelectorAll('.speed-dial [data-contact-modal-link]').forEach((link) => {
      link.setAttribute('href', trimLeadingSlash(contactModalPath));
    });
    footer.dataset.audience = audience.key;
  };

  const isInternalHttpUrl = (url) => url && /^https?:$/i.test(url.protocol) && url.origin === window.location.origin;

  const withAudienceContext = (href, audienceKey) => {
    const raw = String(href || '').trim();
    if (!raw || raw.startsWith('#') || /^(mailto|tel|sms|javascript):/i.test(raw)) return href;
    let url;
    try {
      url = new URL(raw, document.baseURI || window.location.href);
    } catch {
      return href;
    }
    if (!isInternalHttpUrl(url)) return href;

    const path = (url.pathname || '/').replace(/\/index\.html$/i, '/').replace(/\.html$/i, '');
    const needsAudience = path === '/portfolio'
      || path.startsWith('/portfolio/')
      || path === '/contact';
    if (!needsAudience || url.searchParams.has('audience')) return href;

    url.searchParams.set('audience', audienceKey);
    return `${url.pathname}${url.search}${url.hash}`;
  };

  const preserveAudienceContext = (audienceKey) => {
    if (normalizeAudience(audienceKey) === PERSONAL_MODE) return;
    document.querySelectorAll('a[href]').forEach((link) => {
      if (link.matches('[data-site-realm-switch]')) return;
      const current = link.getAttribute('href');
      const next = withAudienceContext(current, audienceKey);
      if (next && next !== current) link.setAttribute('href', next);
    });
  };

  const updateSwitches = (audience) => {
    const isProfessional = normalizeAudience(audience.key) !== PERSONAL_MODE;
    document.querySelectorAll('[data-site-realm-switch]').forEach((link) => {
      const targetMode = normalizeMode(link.dataset.siteRealmSwitch) || (isProfessional ? PERSONAL_MODE : PROFESSIONAL_MODE);
      if (!link.dataset.siteRealmLabel) link.dataset.siteRealmLabel = String(link.textContent || '').trim();
      if ((targetMode === PROFESSIONAL_MODE) === isProfessional) {
        link.hidden = true;
        link.textContent = '';
        link.removeAttribute('href');
        link.setAttribute('aria-hidden', 'true');
        return;
      }

      link.hidden = false;
      link.removeAttribute('aria-hidden');
      link.textContent = link.dataset.siteRealmLabel || (targetMode === PROFESSIONAL_MODE ? 'Work' : 'Home');
      link.setAttribute('href', targetMode === PROFESSIONAL_MODE ? (getAudience(LEGACY_AUDIENCE).homePath || '/analytics') : '/');
      link.setAttribute('aria-label', targetMode === PROFESSIONAL_MODE ? 'Open work-focused pages' : 'Go to the personal home page');
    });
  };

  let linkObserver = null;
  const observeAudienceLinks = (audienceKey) => {
    if (normalizeAudience(audienceKey) === PERSONAL_MODE || linkObserver || !document.body || !('MutationObserver' in window)) return;
    linkObserver = new MutationObserver((mutations) => {
      const hasNewLinks = mutations.some((mutation) => Array.from(mutation.addedNodes || []).some((node) => (
        node && node.nodeType === 1 && (
          (typeof node.matches === 'function' && node.matches('a[href]'))
          || (typeof node.querySelector === 'function' && node.querySelector('a[href]'))
        )
      )));
      if (hasNewLinks) window.requestAnimationFrame(() => preserveAudienceContext(audienceKey));
    });
    try {
      linkObserver.observe(document.body, { childList: true, subtree: true });
    } catch {
      linkObserver.disconnect();
      linkObserver = null;
    }
  };

  const applyRealm = () => {
    canonicalizeLegacyMode();
    const audience = setDocumentRealm(detectAudience());
    applyAudienceNavigation(audience);
    applyAudienceFooter(audience);
    preserveAudienceContext(audience.key);
    updateSwitches(audience);
    observeAudienceLinks(audience.key);
  };

  if (redirectLegacyRoot()) return;
  canonicalizeLegacyMode();
  setDocumentRealm(detectAudience());
  document.addEventListener('DOMContentLoaded', applyRealm);
})();
