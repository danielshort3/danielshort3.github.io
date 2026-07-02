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
          <a href="resume-analytics" class="btn-primary hero-cta">Resume</a>
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
          <a href="portfolio?audience=analytics" class="btn-secondary project-examples-cta">View analytics portfolio</a>
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
            <a href="resume-analytics" class="btn-primary">Resume</a>
            <a href="contact#contact-modal" class="btn-secondary" data-contact-modal-link="true">Contact form</a>
          </div>
        </div>
      </div>
    </section>
  `;

  const readStoredRealm = () => {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch {
      return '';
    }
  };

  const writeStoredRealm = (mode) => {
    try {
      if (mode === PROFESSIONAL_MODE) {
        window.localStorage.setItem(STORAGE_KEY, PROFESSIONAL_MODE);
      } else {
        window.localStorage.removeItem(STORAGE_KEY);
      }
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
    if (explicit) {
      writeStoredRealm(explicit);
      return explicit;
    }
    if (pathLooksProfessional() || bodyLooksProfessional()) return PROFESSIONAL_MODE;
    return normalizeMode(readStoredRealm()) || PERSONAL_MODE;
  };

  const setDocumentMode = (mode) => {
    const isProfessional = mode === PROFESSIONAL_MODE;
    document.documentElement.classList.toggle('site-realm-professional', isProfessional);
    document.documentElement.classList.toggle('site-realm-personal', !isProfessional);
    if (document.body) {
      document.body.dataset.siteRealm = mode;
      if (isProfessional && (!document.body.dataset.audience || document.body.dataset.audience === 'personal')) {
        document.body.dataset.audience = PROFESSIONAL_AUDIENCE;
      }
    }
    window.SITE_REALM = mode;
    window.getSiteRealm = () => window.SITE_REALM || PERSONAL_MODE;
    window.isProfessionalRealm = () => window.getSiteRealm() === PROFESSIONAL_MODE;
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
      link.textContent = isProfessional ? 'Personal' : 'Work';
      link.setAttribute('href', isProfessional ? '/?mode=personal' : '/?mode=professional');
      link.setAttribute('aria-label', isProfessional ? 'Switch to the personal site view' : 'Switch to the professional site view');
      link.dataset.siteRealmSwitch = isProfessional ? PERSONAL_MODE : PROFESSIONAL_MODE;
    });
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
    renderProfessionalHome();
    updateSwitches(mode);
    preserveProfessionalLinks();
    observeProfessionalLinks();
    window.setTimeout(preserveProfessionalLinks, 0);
    window.setTimeout(preserveProfessionalLinks, 500);
    window.requestAnimationFrame?.(preserveProfessionalLinks);
  };

  const handleSwitchClick = (event) => {
    const switchLink = event.target.closest('[data-site-realm-switch]');
    if (!switchLink) return;
    const mode = normalizeMode(switchLink.dataset.siteRealmSwitch);
    if (mode) writeStoredRealm(mode);
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
  document.addEventListener('click', handleSwitchClick, true);
  document.addEventListener('click', handleInternalClick, true);
})();
