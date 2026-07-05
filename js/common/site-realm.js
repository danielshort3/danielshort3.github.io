(() => {
  'use strict';

  const STORAGE_KEY = 'siteRealm';
  const PROFESSIONAL_MODE = 'professional';
  const PERSONAL_MODE = 'personal';
  const PROFESSIONAL_AUDIENCE = 'analytics';
  const PROFESSIONAL_PATHS = new Set([
    '/analytics',
    '/data-science',
    '/tourism',
    '/resume',
    '/resume-pdf',
    '/resume-analytics',
    '/resume-analytics-pdf',
    '/resume-data-science',
    '/resume-data-science-pdf',
    '/resume-tourism',
    '/resume-tourism-pdf'
  ]);
  const PROFESSIONAL_HOME_SOURCE = '/analytics';
  const PROFESSIONAL_RESUME_HTML = `
    <div class="nav-item nav-item-resume">
      <a href="resume" class="nav-link nav-link-has-menu" aria-haspopup="true" aria-expanded="false" aria-controls="nav-dropdown-resume" data-resume-home-link="true">
        Resume
        <span class="nav-link-caret" aria-hidden="true"></span>
      </a>
      <div class="nav-dropdown nav-dropdown-simple" id="nav-dropdown-resume" aria-label="Resume download">
        <div class="nav-dropdown-inner nav-dropdown-inner-simple">
          <div class="nav-dropdown-column nav-dropdown-column-list">
            <div class="nav-dropdown-header" aria-hidden="true">Resume shortcuts</div>
            <div class="nav-dropdown-list" role="list">
              <a href="resume" class="nav-dropdown-link" role="listitem" data-resume-home-link="true">
                <span class="nav-dropdown-title">Resume</span>
                <span class="nav-dropdown-subtitle">BI, reporting, SQL, Tableau, and automation</span>
              </a>
              <a href="resume-pdf" class="nav-dropdown-link" role="listitem" data-resume-preview-link="true">
                <span class="nav-dropdown-title">Preview PDF</span>
                <span class="nav-dropdown-subtitle">Preview the PDF resume</span>
              </a>
              <a href="documents/Resume.pdf" class="nav-dropdown-link" role="listitem" download data-resume-download-link="true">
                <span class="nav-dropdown-title">Download Resume</span>
                <span class="nav-dropdown-subtitle">Download the PDF copy</span>
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;


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

  const queryMode = () => {
    try {
      return normalizeMode(new URLSearchParams(window.location.search || '').get('mode'));
    } catch {
      return '';
    }
  };

  const pathLooksProfessional = () => {
    const path = currentPath().replace(/\.html$/i, '');
    return PROFESSIONAL_PATHS.has(path);
  };

  const bodyLooksProfessional = () => {
    const audience = document.body?.dataset?.audience || '';
    return Boolean(audience && audience !== 'personal');
  };

  const detectMode = () => {
    const explicit = queryMode();
    if (explicit) return explicit;
    clearStoredRealm();
    if (pathLooksProfessional() || bodyLooksProfessional()) return PROFESSIONAL_MODE;
    return PERSONAL_MODE;
  };

  const applyHiringRobots = (mode) => {
    const explicitMode = queryMode();
    const isHiringView = mode === PROFESSIONAL_MODE && explicitMode === PROFESSIONAL_MODE;
    const selector = 'meta[name="robots"][data-site-realm-robots="hiring"]';
    const existing = document.head?.querySelector(selector);
    if (!isHiringView) {
      if (existing) existing.remove();
      return;
    }

    const robots = existing || document.createElement('meta');
    robots.setAttribute('name', 'robots');
    robots.setAttribute('content', 'noindex, nofollow');
    robots.dataset.siteRealmRobots = 'hiring';
    if (!existing && document.head) {
      document.head.appendChild(robots);
    }
  };

  const setDocumentMode = (mode) => {
    const isProfessional = mode === PROFESSIONAL_MODE;
    const isProfessionalHome = isProfessional && currentPath() === '/';
    document.documentElement.classList.toggle('site-realm-professional', isProfessional);
    document.documentElement.classList.toggle('site-realm-personal', !isProfessional);
    document.documentElement.classList.toggle('site-realm-professional-home', isProfessionalHome);
    if (document.body) {
      document.body.dataset.siteRealm = mode;
      document.body.classList.remove('professional-home-page');
      if (isProfessionalHome) {
        document.body.dataset.siteRealmHome = PROFESSIONAL_MODE;
        document.body.dataset.page = PROFESSIONAL_AUDIENCE;
        document.body.dataset.audience = PROFESSIONAL_AUDIENCE;
      } else {
        delete document.body.dataset.siteRealmHome;
      }
      if (isProfessional && (!document.body.dataset.audience || document.body.dataset.audience === 'personal')) {
        document.body.dataset.audience = PROFESSIONAL_AUDIENCE;
      }
    }
    window.SITE_REALM = mode;
    window.getSiteRealm = () => window.SITE_REALM || PERSONAL_MODE;
    window.isProfessionalRealm = () => window.getSiteRealm() === PROFESSIONAL_MODE;
    applyHiringRobots(mode);
  };

  const isInternalHttpUrl = (url) => {
    if (!url || !/^https?:$/i.test(url.protocol)) return false;
    return url.origin === window.location.origin;
  };

  const isSkippableHref = (href) => {
    const value = String(href || '').trim();
    return !value
      || /^(mailto|tel|sms|javascript):/i.test(value);
  };

  const isHashOnlyHref = (href) => {
    const value = String(href || '').trim();
    return value.startsWith('#') && value.length > 1;
  };

  const shouldSkipModeParam = (url) => (
    /\.(?:pdf|docx?|xlsx?|zip|png|jpe?g|webp|avif|svg|ico)$/i.test(url.pathname || '')
  );

  const currentHashHref = (hash) => {
    let url;
    try {
      url = new URL(window.location.href);
    } catch {
      return hash;
    }
    url.searchParams.set('mode', PROFESSIONAL_MODE);
    url.hash = hash;
    return `${url.pathname}${url.search}${url.hash}`;
  };

  const withProfessionalMode = (href) => {
    if (isHashOnlyHref(href)) return currentHashHref(String(href || '').trim());
    if (isSkippableHref(href)) return href;
    let url;
    try {
      url = new URL(href, window.location.href);
    } catch {
      return href;
    }
    if (!isInternalHttpUrl(url) || shouldSkipModeParam(url)) return href;

    const path = (url.pathname || '/').replace(/\/index\.html$/i, '/');
    if ((path === '/portfolio' || path.startsWith('/portfolio/')) && !url.searchParams.get('audience')) {
      url.searchParams.set('audience', PROFESSIONAL_AUDIENCE);
    }
    url.searchParams.set('mode', PROFESSIONAL_MODE);
    return `${url.pathname}${url.search}${url.hash}`;
  };

  const updateSwitches = (mode) => {
    const isProfessional = mode === PROFESSIONAL_MODE;
    document.querySelectorAll('[data-site-realm-switch]').forEach((link) => {
      if (isProfessional) {
        link.hidden = true;
        link.textContent = '';
        link.removeAttribute('href');
        link.setAttribute('aria-hidden', 'true');
        link.dataset.siteRealmSwitch = PERSONAL_MODE;
        return;
      }
      link.hidden = false;
      link.textContent = 'Work';
      link.setAttribute('href', '/?mode=professional');
      link.setAttribute('aria-label', 'Switch to the professional site view');
      link.removeAttribute('aria-hidden');
      link.dataset.siteRealmSwitch = PROFESSIONAL_MODE;
    });
  };

  const setNavLinkLabel = (link, label) => {
    if (!link) return;
    const caret = link.querySelector('.nav-link-caret');
    link.textContent = '';
    link.append(document.createTextNode(label));
    if (caret) {
      link.append(document.createTextNode('\n            '));
      link.append(caret);
    }
  };

  const ensureProfessionalHomeLink = (menu) => {
    let homeLink = menu.querySelector('[data-professional-home-link="true"]');
    if (homeLink) return homeLink;
    homeLink = document.createElement('a');
    homeLink.href = '/';
    homeLink.className = 'nav-link';
    homeLink.textContent = 'Home';
    homeLink.dataset.entryHomeLink = 'true';
    homeLink.dataset.professionalHomeLink = 'true';
    menu.insertBefore(homeLink, menu.firstElementChild);
    return homeLink;
  };

  const applyProfessionalPortfolioNav = (menu) => {
    const portfolioItem = menu.querySelector('.nav-item-portfolio');
    if (!portfolioItem) return;
    const portfolioLink = portfolioItem.querySelector(':scope > .nav-link');
    setNavLinkLabel(portfolioLink, 'Portfolio');
    const footerTitle = portfolioItem.querySelector('.nav-dropdown-all .nav-dropdown-title');
    const footerSubtitle = portfolioItem.querySelector('.nav-dropdown-all .nav-dropdown-subtitle');
    if (footerTitle) footerTitle.textContent = 'View full portfolio';
    if (footerSubtitle) footerSubtitle.textContent = 'Browse the complete project library';
  };

  const ensureProfessionalResumeNav = (menu) => {
    let resumeItem = menu.querySelector('.nav-item-resume');
    if (!resumeItem) {
      const contactItem = menu.querySelector('.nav-item-contact');
      if (contactItem) {
        contactItem.insertAdjacentHTML('beforebegin', PROFESSIONAL_RESUME_HTML.trim());
      } else {
        menu.insertAdjacentHTML('beforeend', PROFESSIONAL_RESUME_HTML.trim());
      }
      resumeItem = menu.querySelector('.nav-item-resume');
    }
    const resumeLink = resumeItem?.querySelector(':scope > .nav-link');
    if (resumeLink) resumeLink.setAttribute('href', 'resume');
  };

  const applyProfessionalContactNav = (menu) => {
    const firstContactLink = menu.querySelector('.nav-item-contact .nav-dropdown-list .nav-dropdown-link');
    if (!firstContactLink) return;
    const title = firstContactLink.querySelector('.nav-dropdown-title');
    const subtitle = firstContactLink.querySelector('.nav-dropdown-subtitle');
    if (title) {
      title.textContent = 'Message about a role';
      const badge = document.createElement('span');
      badge.className = 'nav-dropdown-badge';
      badge.setAttribute('aria-hidden', 'true');
      badge.textContent = 'Recommended';
      title.appendChild(badge);
    }
    if (subtitle) {
      subtitle.textContent = 'Best for analytics, BI, reporting, and automation work';
    }
  };

  const applyProfessionalNavigation = () => {
    if (window.getSiteRealm() !== PROFESSIONAL_MODE) return;
    const header = document.getElementById('combined-header-nav');
    const menu = header?.querySelector('#primary-menu');
    if (!header || !menu || header.dataset.siteRealmNav === PROFESSIONAL_MODE) return;

    const brandLink = header.querySelector('.brand[data-entry-home-link="true"]');
    if (brandLink) brandLink.setAttribute('href', 'analytics');

    ensureProfessionalHomeLink(menu);
    menu.querySelectorAll('.nav-item-tools, .nav-item-games').forEach((item) => item.remove());
    applyProfessionalPortfolioNav(menu);
    ensureProfessionalResumeNav(menu);
    applyProfessionalContactNav(menu);

    header.dataset.siteRealmNav = PROFESSIONAL_MODE;
  };

  const preserveProfessionalLinks = () => {
    if (window.getSiteRealm() !== PROFESSIONAL_MODE) return;
    document.querySelectorAll('a[href]').forEach((link) => {
      if (link.matches('[data-site-realm-switch]')) return;
      const next = withProfessionalMode(link.getAttribute('href'));
      if (next && next !== link.getAttribute('href')) {
        link.setAttribute('href', next);
      }
    });
  };

  let preserveLinksQueued = false;
  const schedulePreserveProfessionalLinks = () => {
    if (preserveLinksQueued || window.getSiteRealm() !== PROFESSIONAL_MODE) return;
    preserveLinksQueued = true;
    const run = () => {
      preserveLinksQueued = false;
      preserveProfessionalLinks();
    };
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(run);
    } else {
      window.setTimeout(run, 0);
    }
  };

  let linkMutationObserver = null;
  const observeProfessionalLinks = () => {
    if (linkMutationObserver || window.getSiteRealm() !== PROFESSIONAL_MODE || !document.body || !('MutationObserver' in window)) return;
    linkMutationObserver = new MutationObserver((mutations) => {
      const hasNewLinks = mutations.some((mutation) => Array.from(mutation.addedNodes || []).some((node) => (
        node && node.nodeType === 1 && (
          (typeof node.matches === 'function' && node.matches('a[href]')) ||
          (typeof node.querySelector === 'function' && node.querySelector('a[href]'))
        )
      )));
      if (hasNewLinks) schedulePreserveProfessionalLinks();
    });
    linkMutationObserver.observe(document.body, { childList: true, subtree: true });
  };

  const ensureProfessionalHomeStyles = (sourceDoc) => {
    const sourceLinks = Array.from(sourceDoc.querySelectorAll('link[rel~="stylesheet"][href]'));
    const existingHrefs = new Set(Array.from(document.querySelectorAll('link[rel~="stylesheet"][href]')).map((link) => {
      try {
        return new URL(link.getAttribute('href'), window.location.href).href;
      } catch {
        return link.getAttribute('href');
      }
    }));

    const pending = sourceLinks.map((sourceLink) => {
      let absoluteHref;
      try {
        absoluteHref = new URL(sourceLink.getAttribute('href'), window.location.href).href;
      } catch {
        absoluteHref = sourceLink.getAttribute('href');
      }
      if (!absoluteHref || existingHrefs.has(absoluteHref)) return null;

      const link = sourceLink.cloneNode(false);
      existingHrefs.add(absoluteHref);
      return new Promise((resolve) => {
        const finish = () => resolve();
        const timeout = window.setTimeout(finish, 3000);
        link.addEventListener('load', () => {
          window.clearTimeout(timeout);
          finish();
        }, { once: true });
        link.addEventListener('error', () => {
          window.clearTimeout(timeout);
          finish();
        }, { once: true });
        document.head.appendChild(link);
      });
    }).filter(Boolean);

    return Promise.all(pending);
  };

  let professionalHomeRenderPromise = null;
  const renderProfessionalHome = () => {
    if (window.getSiteRealm() !== PROFESSIONAL_MODE || currentPath() !== '/') return;
    const main = document.getElementById('main');
    if (!main || main.dataset.siteRealmRendered === PROFESSIONAL_MODE) return;
    if (professionalHomeRenderPromise) return professionalHomeRenderPromise;

    if (document.body) {
      document.body.dataset.page = PROFESSIONAL_AUDIENCE;
      document.body.dataset.audience = PROFESSIONAL_AUDIENCE;
      document.body.dataset.siteRealmHome = PROFESSIONAL_MODE;
    }

    const sourceUrl = new URL(PROFESSIONAL_HOME_SOURCE, window.location.origin);
    sourceUrl.searchParams.set('mode', PROFESSIONAL_MODE);
    main.dataset.siteRealmRendered = 'loading';

    professionalHomeRenderPromise = window.fetch(sourceUrl.toString(), { credentials: 'same-origin' })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load professional home source: ${response.status}`);
        }
        return response.text();
      })
      .then((html) => {
        const sourceDoc = new DOMParser().parseFromString(html, 'text/html');
        const sourceMain = sourceDoc.getElementById('main');
        if (!sourceMain) {
          throw new Error('Professional home source is missing #main');
        }
        return ensureProfessionalHomeStyles(sourceDoc).then(() => ({ sourceDoc, sourceMain }));
      })
      .then(({ sourceDoc, sourceMain }) => {
        main.innerHTML = sourceMain.innerHTML;
        main.dataset.siteRealmRendered = PROFESSIONAL_MODE;
        document.title = sourceDoc.title || 'Data Analytics | Daniel Short';
        const sourceDescription = sourceDoc.querySelector('meta[name="description"]');
        const description = document.querySelector('meta[name="description"]');
        if (sourceDescription && description) {
          description.setAttribute('content', sourceDescription.getAttribute('content') || '');
        }
        preserveProfessionalLinks();
        document.dispatchEvent(new CustomEvent('site:content-updated', {
          detail: { source: 'professional-home', mode: PROFESSIONAL_MODE }
        }));
      })
      .catch((error) => {
        main.dataset.siteRealmRendered = 'error';
        console.warn(error);
      });

    return professionalHomeRenderPromise;
  };

  const applyMode = () => {
    const mode = detectMode();
    setDocumentMode(mode);
    applyProfessionalNavigation();
    renderProfessionalHome();
    updateSwitches(mode);
    preserveProfessionalLinks();
    observeProfessionalLinks();
    window.setTimeout(preserveProfessionalLinks, 0);
    window.setTimeout(preserveProfessionalLinks, 500);
    window.requestAnimationFrame?.(preserveProfessionalLinks);
  };

  const handleInternalClick = (event) => {
    if (window.getSiteRealm() !== PROFESSIONAL_MODE) return;
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest('a[href]');
    if (!link || link.matches('[data-site-realm-switch]')) return;
    const next = withProfessionalMode(link.getAttribute('href'));
    if (next && next !== link.getAttribute('href')) {
      link.setAttribute('href', next);
    }
  };

  setDocumentMode(detectMode());
  document.addEventListener('DOMContentLoaded', applyMode);
  document.addEventListener('click', handleInternalClick, true);
})();
