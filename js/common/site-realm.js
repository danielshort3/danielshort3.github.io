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
    const root = document.documentElement;

    root.classList.toggle('site-realm-professional', isProfessional);
    root.classList.toggle('site-realm-personal', !isProfessional);
    root.classList.remove('site-realm-query-pending');
    root.classList.remove('site-realm-professional-home');
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

  const applyAudienceNavigation = (audience) => {
    if (!audience) return;
    const header = document.getElementById('combined-header-nav');
    if (!header) return;
    header.querySelectorAll('[data-entry-home-link="true"]')
      .forEach((link) => link.setAttribute('href', audience.homePath || '/'));
    header.dataset.siteRealmNav = audience.key;
    const form = header.querySelector('.nav-search');
    if (!form) return;
    let contextInput = form.querySelector('[data-search-audience]');
    if (normalizeAudience(audience.key) === PERSONAL_MODE) {
      contextInput?.remove();
      return;
    }
    if (!contextInput) {
      contextInput = document.createElement('input');
      contextInput.type = 'hidden';
      contextInput.name = 'audience';
      contextInput.dataset.searchAudience = '';
      form.appendChild(contextInput);
    }
    contextInput.value = audience.key;
  };

  const applyAudienceFooter = (audience) => {
    const footer = document.querySelector('[data-site-shell-footer]');
    if (footer && audience) footer.dataset.audience = audience.key;
  };

  const applyAudienceContact = (audience) => {
    if (!audience || normalizeAudience(audience.key) === PERSONAL_MODE || document.body?.dataset.page !== 'contact') return;
    const key = normalizeAudience(audience.key);
    if (document.body.dataset.contactAudience === key) return;

    const focusByAudience = {
      analytics: 'data analytics, BI, and reporting automation',
      'data-science': 'data science, applied machine learning, and model evaluation',
      tourism: 'tourism analytics, destination intelligence, and stakeholder reporting'
    };
    const focus = focusByAudience[key] || `${audience.label || audience.shortLabel || 'professional'} work`;
    const shortLabel = audience.shortLabel || audience.label || 'Professional';
    const hero = document.querySelector('.contact-page .hero');
    const heading = hero?.querySelector('h1');
    const tagline = hero?.querySelector('.hero-tagline');
    const ctaGroup = hero?.querySelector('.cta-group');
    if (heading) heading.textContent = `Let's Talk ${shortLabel} Roles`;
    if (tagline) tagline.textContent = `Reach out about ${focus} roles. Include the team, priorities, and timeline; I'll reply with the most relevant work.`;

    if (ctaGroup && !hero.querySelector('[data-professional-contact-links]')) {
      const proofLinks = document.createElement('nav');
      proofLinks.className = 'contact-professional-links';
      proofLinks.dataset.professionalContactLinks = 'true';
      proofLinks.setAttribute('aria-label', 'Professional proof links');
      proofLinks.innerHTML = `
        <a href="${escapeHtml(trimLeadingSlash(audience.portfolioPath || '/portfolio'))}">View matching portfolio</a>
        <a href="${escapeHtml(trimLeadingSlash(audience.resumePath || ''))}">View matching resume</a>
      `.trim();
      ctaGroup.insertAdjacentElement('afterend', proofLinks);
    }

    const optionsHeading = document.querySelector('#contact-options .contact-options-heading .section-title');
    const optionsSubtitle = document.querySelector('#contact-options .contact-options-heading .section-subtitle');
    if (optionsHeading) optionsHeading.textContent = 'Direct Contact';
    if (optionsSubtitle) optionsSubtitle.textContent = 'Email is fastest, and GitHub shows implementation detail.';
    const duplicateMessageCard = document.querySelector('#contact-options .contact-card-recommended');
    if (duplicateMessageCard) duplicateMessageCard.hidden = true;

    const messageInput = document.getElementById('contact-message');
    if (messageInput) messageInput.setAttribute('placeholder', 'Share the role, team, priorities, timeline, and any questions.');

    const location = document.getElementById('grand-junction-location');
    const locationBody = location?.querySelector('.section-subtitle');
    const mapShell = location?.querySelector('.cms-map-shell');
    if (locationBody) locationBody.textContent = 'Based in Grand Junction, Colorado and open to remote, hybrid, and Colorado-based opportunities.';
    if (mapShell) mapShell.hidden = true;

    document.body.classList.add('professional-contact-page');
    document.body.dataset.contactAudience = key;
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
      || path === '/contact'
      || path === '/search';
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
  let audienceLinkRevision = 0;
  const observeAudienceLinks = (audienceKey) => {
    const revision = ++audienceLinkRevision;
    linkObserver?.disconnect();
    linkObserver = null;
    if (normalizeAudience(audienceKey) === PERSONAL_MODE || !document.body || !('MutationObserver' in window)) return;
    linkObserver = new MutationObserver((mutations) => {
      const hasNewLinks = mutations.some((mutation) => Array.from(mutation.addedNodes || []).some((node) => (
        node && node.nodeType === 1 && (
          (typeof node.matches === 'function' && node.matches('a[href]'))
          || (typeof node.querySelector === 'function' && node.querySelector('a[href]'))
        )
      )));
      if (hasNewLinks) window.requestAnimationFrame(() => {
        if (revision === audienceLinkRevision) preserveAudienceContext(audienceKey);
      });
    });
    try {
      linkObserver.observe(document.body, { childList: true, subtree: true });
    } catch {
      linkObserver.disconnect();
      linkObserver = null;
    }
  };

  const applyRealm = (options = {}) => {
    canonicalizeLegacyMode();
    let audienceKey = '';
    if (options.url) {
      try {
        const url = new URL(options.url, window.location.href);
        audienceKey = normalizeAudience(url.searchParams.get('audience') || detectAudienceFromPath(url.pathname) || bodyAudience() || PERSONAL_MODE);
      } catch {}
    }
    const audience = setDocumentRealm(audienceKey || detectAudience());
    applyAudienceNavigation(audience);
    applyAudienceFooter(audience);
    applyAudienceContact(audience);
    preserveAudienceContext(audience.key);
    updateSwitches(audience);
    observeAudienceLinks(audience.key);
    return audience;
  };

  if (redirectLegacyRoot()) return;
  canonicalizeLegacyMode();
  setDocumentRealm(detectAudience());
  document.addEventListener('DOMContentLoaded', applyRealm);
  document.addEventListener('site:route-before-mount', (event) => applyRealm(event.detail || {}));
  document.addEventListener('site:route-mounted', (event) => applyRealm(event.detail || {}));
  window.SiteRealm = Object.freeze({ sync: applyRealm });
})();
