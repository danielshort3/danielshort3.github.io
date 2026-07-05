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

  const PROFESSIONAL_HOME_HTML = `
    <section class="hero hero--default">
      <div class="wrapper">
        <h1>Data Analytics,<br>Made Actionable</h1>
        <div class="hero-identity">
          <img class="hero-avatar" src="img/hero/head-avatar-192.jpg" srcset="img/hero/head-avatar-192.jpg 192w, img/hero/head-avatar-384.jpg 384w" sizes="(max-width: 768px) 112px, 132px" alt="Portrait of Daniel Short" width="132" height="132" decoding="async" fetchpriority="high">
          <p class="hero-eyebrow">Daniel Short · Analytics &amp; BI</p>
        </div>
        <p class="hero-tagline">I build SQL, Tableau, Excel, and Python workflows that turn recurring reporting, KPI tracking, and operational questions into outputs leaders can act on quickly.</p>
        <p class="hero-status">SQL · Tableau · Python · Reporting</p>
        <div class="cta-group">
          <a href="resume" class="btn-primary hero-cta">Resume</a>
          <a href="portfolio?audience=analytics" class="btn-secondary hero-cta">Portfolio</a>
          <a href="contact#contact-modal" class="btn-ghost hero-cta" data-contact-modal-link="true">Contact</a>
        </div>
      </div>
      <a class="chevron-hint scroll-indicator" href="#professional-projects" data-smooth-scroll="true">
        <span class="chevron-label">View project examples</span>
        <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 16.5a1 1 0 0 1-.7-.29l-6-6a1 1 0 1 1 1.4-1.42L12 14.08l5.3-5.29a1 1 0 0 1 1.4 1.42l-6 6a1 1 0 0 1-.7.29z"/></svg>
      </a>
    </section>

    <section id="professional-projects" class="project-examples-band surface-band no-reveal" role="region" aria-labelledby="professional-projects-title">
      <div class="wrapper">
        <div class="project-examples-head">
          <h2 id="professional-projects-title">Project Examples</h2>
          <p class="project-examples-subtitle">A focused view of dashboarding, SQL workflows, forecasting, and decision support tied to business questions.</p>
        </div>
        <div class="home-showcase-grid" role="list" aria-label="Professional project examples">
          <article class="home-showcase-card" role="listitem">
            <h3>Store-Level Loss &amp; Sales ETL</h3>
            <p class="home-showcase-summary">SQL ETL and anomaly views for comparing loss signals against store sales context.</p>
            <a class="btn-secondary" href="portfolio/retailStore?audience=analytics">View project</a>
          </article>
          <article class="home-showcase-card" role="listitem">
            <h3>Empty-Package Shrink Dashboard</h3>
            <p class="home-showcase-summary">Excel forecasting and BI workflow for tracking theft trends, hotspots, and prevention follow-up.</p>
            <a class="btn-secondary" href="portfolio/targetEmptyPackage?audience=analytics">View project</a>
          </article>
          <article class="home-showcase-card" role="listitem">
            <h3>Pizza Delivery Dashboard</h3>
            <p class="home-showcase-summary">Tableau KPI tracking and forecasting for planning, regional comparison, and operational review.</p>
            <a class="btn-secondary" href="portfolio/pizzaDashboard?audience=analytics">View project</a>
          </article>
        </div>
        <div class="project-examples-actions">
          <a href="portfolio?audience=analytics" class="btn-secondary project-examples-cta">View portfolio</a>
        </div>
      </div>
    </section>

    <section id="professional-results" class="home-proof-band surface-band no-reveal" role="region" aria-labelledby="professional-results-title">
      <div class="wrapper">
        <div class="home-proof-head">
          <h2 id="professional-results-title" class="section-title">Business-Facing Results</h2>
          <p>A concise snapshot of outcomes behind the professional project examples.</p>
        </div>
        <ul class="home-proof-kpis" aria-label="Business-facing analytics results">
          <li class="home-proof-item"><div class="home-proof-card"><span class="home-proof-value">99%</span><span class="home-proof-label">Faster reporting turnaround on recurring leadership and stakeholder updates.</span></div></li>
          <li class="home-proof-item"><div class="home-proof-card"><span class="home-proof-value">200+</span><span class="home-proof-label">Hours saved annually by automating recurring report prep and delivery.</span></div></li>
          <li class="home-proof-item"><div class="home-proof-card"><span class="home-proof-value">24%</span><span class="home-proof-label">Inventory loss reduction through analytics-driven investigations and workflow improvements.</span></div></li>
          <li class="home-proof-item"><div class="home-proof-card"><span class="home-proof-value">57.6%</span><span class="home-proof-label">Improvement in theft reporting after dashboard and process redesign.</span></div></li>
        </ul>
      </div>
    </section>

    <section id="professional-experience" class="surface-band reveal destination-section" aria-labelledby="professional-experience-title">
      <div class="wrapper">
        <div class="destination-section-head">
          <h2 id="professional-experience-title">Work Snapshot</h2>
          <p>Recent roles across destination reporting, AI data quality, retail operations, and business analysis.</p>
        </div>
        <div class="home-showcase-grid" role="list" aria-label="Professional work experience">
          <article class="home-showcase-card" role="listitem">
            <h3>Visit Grand Junction</h3>
            <p class="home-showcase-summary">Automated stakeholder reporting across lodging, tax, web, campaign, airport, and visitor data.</p>
          </article>
          <article class="home-showcase-card" role="listitem">
            <h3>Randall Reilly</h3>
            <p class="home-showcase-summary">Re-platformed R workflows into Python and built anomaly models for QA review prioritization.</p>
          </article>
          <article class="home-showcase-card" role="listitem">
            <h3>Target</h3>
            <p class="home-showcase-summary">Designed dashboards and investigations that improved theft reporting and reduced inventory loss.</p>
          </article>
        </div>
      </div>
    </section>

    <section id="professional-skills" class="surface-band reveal destination-section" aria-labelledby="professional-skills-title">
      <div class="wrapper">
        <div class="destination-section-head">
          <h2 id="professional-skills-title">Skills in Practice</h2>
          <p>SQL, Tableau, Excel, Python, forecasting, anomaly detection, stakeholder reporting, GA4, and dashboard design.</p>
        </div>
        <div class="home-contact-panel" role="group" aria-label="Professional call to action">
          <div class="home-contact-copy">
            <h2 class="section-title">Open to analytics, BI, and reporting automation roles</h2>
            <p>I’m open to roles where practical data workflows turn recurring business questions into decision-ready outputs.</p>
          </div>
          <div class="home-contact-actions" aria-label="Professional contact options">
            <a href="resume" class="btn-primary">Resume</a>
            <a href="contact#contact-modal" class="btn-secondary" data-contact-modal-link="true">Contact form</a>
          </div>
        </div>
      </div>
    </section>
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
      document.body.classList.toggle('professional-home-page', isProfessionalHome);
      if (isProfessionalHome) {
        document.body.dataset.siteRealmHome = PROFESSIONAL_MODE;
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
      || value.startsWith('#')
      || /^(mailto|tel|sms|javascript):/i.test(value);
  };

  const shouldSkipModeParam = (url) => (
    /\.(?:pdf|docx?|xlsx?|zip|png|jpe?g|webp|avif|svg|ico)$/i.test(url.pathname || '')
  );

  const withProfessionalMode = (href) => {
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

  const renderProfessionalHome = () => {
    if (window.getSiteRealm() !== PROFESSIONAL_MODE || currentPath() !== '/') return;
    const main = document.getElementById('main');
    if (!main || main.dataset.siteRealmRendered === PROFESSIONAL_MODE) return;
    main.innerHTML = PROFESSIONAL_HOME_HTML.trim();
    main.dataset.siteRealmRendered = PROFESSIONAL_MODE;
    document.title = 'Data Analytics | Daniel Short';
    const description = document.querySelector('meta[name="description"]');
    if (description) {
      description.setAttribute('content', 'Professional analytics and BI portfolio for Daniel Short: SQL, Tableau, reporting automation, dashboarding, forecasting, and business decision support.');
    }
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
