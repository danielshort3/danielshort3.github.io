/* ===================================================================
   File: common.js
   Purpose: Site-wide helpers and home page popups
=================================================================== */
(() => {
  'use strict';
  const $  = (s,c=document)=>c.querySelector(s);
  const $$ = (s,c=document)=>[...c.querySelectorAll(s)];
  const on = (n,e,f,o)=>n&&n.addEventListener(e,f,o);
  const run = fn=>typeof fn==='function'&&fn();
  const isPage = (...names)=>names.includes(document.body.dataset.page);
  const CONTACT_API_ENDPOINT = '/api/contact';
  const CONTACT_CONTEXT_KEY = 'contactOrigin';
  const CONTACT_MODAL_ID = 'contact-modal';
  const CONTACT_MODAL_SCRIPT = 'js/forms/contact.js';
  const initClientErrorTelemetry = () => {
    if (typeof window === 'undefined' || window.__clientErrorTelemetryInit) return;
    window.__clientErrorTelemetryInit = true;

    const sendErrorEvent = (kind, details) => {
      try {
        if (typeof window.gaEvent !== 'function') return;
        window.gaEvent('client_error', {
          kind: String(kind || 'unknown').slice(0, 32),
          page_path: String((window.location && window.location.pathname) || '').slice(0, 120),
          message: String(details || '').slice(0, 160)
        });
      } catch {}
    };

    window.addEventListener('error', (event) => {
      const message = event && (event.message || (event.error && event.error.message)) || 'error';
      sendErrorEvent('error', message);
    });

    window.addEventListener('unhandledrejection', (event) => {
      const reason = event && event.reason;
      const message = reason && reason.message ? reason.message : String(reason || 'unhandledrejection');
      sendErrorEvent('unhandledrejection', message);
    });
  };
  const CONTACT_MODAL_MARKUP = `
    <div id="contact-modal" class="modal">
      <div class="modal-content" role="dialog" aria-modal="true" tabindex="0" aria-labelledby="contact-modal-title">
        <button class="modal-close" aria-label="Close dialog">&times;</button>
        <div class="modal-title-strip">
          <h3 class="modal-title" id="contact-modal-title">Send a Message</h3>
        </div>
        <div class="modal-body">
          <form id="contact-form" class="contact-form" method="post" action="${CONTACT_API_ENDPOINT}" data-endpoint="${CONTACT_API_ENDPOINT}" novalidate>
            <div class="form-field">
              <label for="contact-name">Name <span class="field-required" id="contact-name-required" hidden>- Required</span></label>
              <input id="contact-name" name="name" type="text" autocomplete="name" required maxlength="200" placeholder="Jane Doe" aria-describedby="contact-name-required">
            </div>
            <div class="form-field">
              <label for="contact-email">Email <span class="field-required" id="contact-email-required" hidden>- Required</span></label>
              <input id="contact-email" name="email" type="email" autocomplete="email" required placeholder="you@example.com" aria-describedby="contact-email-required">
            </div>
            <div class="form-field">
              <label for="contact-message">How can I help? <span class="field-required" id="contact-message-required" hidden>- Required</span></label>
              <textarea id="contact-message" name="message" rows="5" maxlength="4000" required placeholder="Share a few details about your project, idea, or question." aria-describedby="contact-message-required"></textarea>
            </div>
            <div class="form-field honeypot" aria-hidden="true">
              <label for="contact-company">Company</label>
              <input id="contact-company" name="company" type="text" tabindex="-1" autocomplete="off">
            </div>
            <p id="contact-status" class="contact-form-status" role="status" aria-live="polite" tabindex="-1"></p>
            <div id="contact-alt" class="contact-form-alt" hidden>
              <a href="mailto:daniel@danielshort.me" class="btn-ghost">Email me directly</a>
            </div>
            <div class="form-actions">
              <button type="submit" class="btn-primary">
                <span class="btn-spinner" aria-hidden="true"></span>
                <span class="btn-label">Send Message</span>
              </button>
              <button type="button" class="btn-ghost" data-contact-reset>Clear form</button>
            </div>
          </form>
          <div class="contact-form-success" id="contact-success" hidden tabindex="-1" role="status" aria-live="polite">
            <span class="success-icon" aria-hidden="true"></span>
            <h4>Message sent</h4>
            <p>Thanks for reaching out. I received your note and will reply shortly. If it&rsquo;s urgent, feel free to send a direct email as well.</p>
            <div class="form-actions">
              <button type="button" class="btn-primary" data-contact-new>Start another message</button>
              <a href="mailto:daniel@danielshort.me" class="btn-secondary">Email me directly</a>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;

  const storeContactOrigin = () => {
    try {
      const title = (document.title || '').trim();
      const url = (window.location && window.location.href) ? window.location.href.trim() : '';
      const audience = typeof window.getSiteAudience === 'function'
        ? window.getSiteAudience()
        : String(document.body?.dataset?.audience || 'personal').trim();
      sessionStorage.setItem(CONTACT_CONTEXT_KEY, JSON.stringify({ title, url, audience, ts: Date.now() }));
    } catch {}
  };

  const trackContactOrigin = () => {
    if (!document || !document.addEventListener) return;
    document.addEventListener('click', (event) => {
      const trigger = event.target.closest('[data-contact-modal-link], #contact-form-toggle');
      if (!trigger) return;
      storeContactOrigin();
    });
  };

  let jumpPanelScrollToken = 0;
  let activeJumpPanelScrollToken = 0;
  let jumpPanelScrollTimer = null;
  let activeSmoothScrollCancel = null;
  const beginJumpPanelAutoScroll = (timeoutMs) => {
    jumpPanelScrollToken += 1;
    const token = jumpPanelScrollToken;
    activeJumpPanelScrollToken = token;
    if (jumpPanelScrollTimer) {
      clearTimeout(jumpPanelScrollTimer);
      jumpPanelScrollTimer = null;
    }
    if (typeof timeoutMs === 'number' && timeoutMs > 0) {
      jumpPanelScrollTimer = setTimeout(() => {
        if (activeJumpPanelScrollToken === token) activeJumpPanelScrollToken = 0;
        jumpPanelScrollTimer = null;
      }, timeoutMs);
    }
    return token;
  };
  const endJumpPanelAutoScroll = (token) => {
    if (activeJumpPanelScrollToken === token) activeJumpPanelScrollToken = 0;
    if (jumpPanelScrollTimer) {
      clearTimeout(jumpPanelScrollTimer);
      jumpPanelScrollTimer = null;
    }
  };
  const isJumpPanelAutoScrolling = () => activeJumpPanelScrollToken !== 0;
  const normalizePagePath = (pathname) => {
    let next = String(pathname || '/');
    next = next.replace(/\/index\.html$/i, '/');
    next = next.replace(/\.html$/i, '');
    next = next.replace(/\/+$/, '');
    return next || '/';
  };
  const samePageHashFromHref = (href) => {
    const value = String(href || '').trim();
    if (!value) return '';
    let targetUrl;
    let currentUrl;
    try {
      targetUrl = new URL(value, window.location.href);
      currentUrl = new URL(window.location.href);
    } catch {
      return '';
    }
    if (!targetUrl.hash || targetUrl.hash.length < 2) return '';
    if (targetUrl.origin !== currentUrl.origin) return '';
    if (normalizePagePath(targetUrl.pathname) !== normalizePagePath(currentUrl.pathname)) return '';
    const searchWithoutRealm = (url) => {
      const params = new URLSearchParams(url.search || '');
      params.delete('mode');
      params.sort();
      return params.toString();
    };
    if (searchWithoutRealm(targetUrl) !== searchWithoutRealm(currentUrl)) return '';
    return targetUrl.hash;
  };
  const currentHashUrl = (hash) => {
    try {
      const url = new URL(window.location.href);
      url.hash = hash;
      return `${url.pathname}${url.search}${url.hash}`;
    } catch {
      return hash;
    }
  };

  const workDateRank = (value) => {
    const text = String(value || '').trim();
    if (/present/i.test(text)) return Number.MAX_SAFE_INTEGER;
    const matches = [...text.matchAll(/\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b/gi)];
    const last = matches.length ? matches[matches.length - 1][0] : '';
    const parsed = last ? Date.parse(`1 ${last}`) : Number.NaN;
    return Number.isFinite(parsed) ? parsed : 0;
  };

  const WORK_EXPERIENCE_MONTHS = Object.freeze({
    jan: 0,
    january: 0,
    feb: 1,
    february: 1,
    mar: 2,
    march: 2,
    apr: 3,
    april: 3,
    may: 4,
    jun: 5,
    june: 5,
    jul: 6,
    july: 6,
    aug: 7,
    august: 7,
    sep: 8,
    sept: 8,
    september: 8,
    oct: 9,
    october: 9,
    nov: 10,
    november: 10,
    dec: 11,
    december: 11
  });

  const parseWorkExperienceRange = (value, now = new Date()) => {
    const text = String(value || '').trim();
    const matches = [...text.matchAll(/\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\b/gi)];
    if (!matches.length) return null;
    const monthIndex = (match) => {
      const month = WORK_EXPERIENCE_MONTHS[String(match?.[1] || '').toLowerCase()];
      const year = Number.parseInt(match?.[2], 10);
      return Number.isInteger(month) && Number.isFinite(year) ? (year * 12) + month : Number.NaN;
    };
    const start = monthIndex(matches[0]);
    const currentMonth = (now.getFullYear() * 12) + now.getMonth();
    const parsedEnd = /\bpresent\b/i.test(text)
      ? currentMonth
      : monthIndex(matches[matches.length - 1]);
    const end = Math.min(parsedEnd, currentMonth);
    if (start > currentMonth) return null;
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
    return { start, end };
  };

  const mergeWorkExperienceIntervals = (intervals) => {
    const sorted = intervals
      .filter((interval) => interval && Number.isFinite(interval.start) && Number.isFinite(interval.end))
      .map((interval) => ({ start: interval.start, end: interval.end }))
      .sort((a, b) => (a.start - b.start) || (a.end - b.end));
    return sorted.reduce((merged, interval) => {
      const previous = merged[merged.length - 1];
      if (!previous || interval.start > previous.end + 1) {
        merged.push(interval);
      } else {
        previous.end = Math.max(previous.end, interval.end);
      }
      return merged;
    }, []);
  };

  const formatWorkExperienceDuration = (totalMonths) => {
    const months = Math.max(0, Math.floor(Number(totalMonths) || 0));
    const years = Math.floor(months / 12);
    if (years < 1) return 'Under 1 year';
    return `${years}+ ${years === 1 ? 'year' : 'years'}`;
  };

  const updateWorkExperienceSummaries = (root = document) => {
    $$('.work-grid', root).forEach((grid) => {
      const section = grid.closest('#work-experience, .work-band');
      const head = section?.querySelector('.work-head');
      if (!head) return;
      const timeframes = $$('.work-timeframe', grid);
      const intervals = timeframes.map((node) => parseWorkExperienceRange(node.textContent));
      const mergedIntervals = mergeWorkExperienceIntervals(intervals);
      const totalMonths = mergedIntervals.reduce(
        (total, interval) => total + Math.max(1, interval.end - interval.start + 1),
        0
      );
      let summary = head.querySelector('[data-work-experience-summary]');
      if (!mergedIntervals.length || totalMonths <= 0) {
        summary?.remove();
        return;
      }
      if (!summary) {
        summary = document.createElement('p');
        summary.className = 'work-experience-summary';
        summary.dataset.workExperienceSummary = 'true';
        const value = document.createElement('strong');
        value.className = 'work-experience-summary__value';
        const label = document.createElement('span');
        label.className = 'work-experience-summary__label';
        summary.append(value, label);
        const heading = head.querySelector('h2');
        if (heading) {
          heading.insertAdjacentElement('afterend', summary);
        } else {
          head.appendChild(summary);
        }
      }
      const duration = formatWorkExperienceDuration(totalMonths);
      const roleCount = intervals.filter(Boolean).length;
      const value = summary.querySelector('.work-experience-summary__value');
      const label = summary.querySelector('.work-experience-summary__label');
      if (value) value.textContent = duration;
      if (label) label.textContent = ' of professional analytics experience';
      summary.dataset.totalMonths = String(totalMonths);
      summary.dataset.roleCount = String(roleCount);
      summary.setAttribute(
        'aria-label',
        `${duration} of professional analytics experience, calculated from ${roleCount} listed ${roleCount === 1 ? 'role' : 'roles'}.`
      );
    });
  };

  const sortWorkCardsOldestFirst = (root = document) => {
    $$('.work-grid:not([data-work-order="oldest-first"])', root).forEach((grid) => {
      const cards = $$('.work-card', grid);
      if (cards.length < 2) return;
      cards
        .map((card, index) => ({ card, index, rank: workDateRank($('.work-timeframe', card)?.textContent) }))
        .sort((a, b) => (a.rank - b.rank) || (a.index - b.index))
        .forEach(({ card }) => grid.appendChild(card));
      grid.dataset.workOrder = 'oldest-first';
    });
  };

  const normalizeAudienceSectionOrder = () => {
    if (!document.body?.matches('[data-page="analytics"], [data-page="data-science"], [data-page="tourism"]')) return;
    const main = document.getElementById('main');
    if (!main) return;
    const order = document.body.matches('[data-page="analytics"]')
      ? [
          'selected-outcomes',
          'work-experience',
          'project-examples',
          'about-me',
          'certifications',
          'cta'
        ]
      : [
          'selected-outcomes',
          'transferability',
          'project-examples',
          'work-experience',
          'about-me',
          'certifications',
          'cta'
        ];
    const children = [...main.children];
    const orderedSections = order
      .map((id) => children.find((node) => node.id === id))
      .filter(Boolean);
    if (orderedSections.length < 2) return;
    orderedSections.forEach((section) => main.appendChild(section));
    main.dataset.audienceSectionOrder = document.body.matches('[data-page="analytics"]')
      ? 'proof-experience-projects'
      : 'proof-projects-experience';
  };

  const ANALYTICS_STORY_CHAPTERS = Object.freeze([
    { id: 'selected-outcomes', label: 'Business-Facing Results' },
    { id: 'work-experience', label: 'Work Experience' },
    { id: 'project-examples', label: 'Project Examples' },
    { id: 'about-me', label: 'Skills in Practice' },
    { id: 'certifications', label: 'Education & Credentials' },
    { id: 'cta', label: 'Start a Conversation' }
  ]);

  const ANALYTICS_STORY_CARD_SELECTORS = Object.freeze({
    'selected-outcomes': '.home-proof-kpis > .home-proof-item',
    'work-experience': '.work-grid > .work-card',
    'project-examples': '.project-examples-grid > .project-examples-card',
    'about-me': '.grid-container > .icon-info.skill-link',
    certifications: '.cert-track > .cert',
    cta: '#cta-link'
  });

  const ANALYTICS_CREDENTIAL_GROUPS = Object.freeze(['degree', 'google', 'ibm']);

  const groupAnalyticsWorkCardMeta = (root = document) => {
    $$('.work-card:not([data-work-meta-grouped="true"])', root).forEach((card) => {
      const company = card.querySelector('.work-company');
      const role = card.querySelector('.work-role');
      const timeframe = card.querySelector('.work-timeframe');
      if (!company || !role || !timeframe) return;
      const meta = document.createElement('div');
      meta.className = 'work-card-meta';
      company.before(meta);
      meta.append(company, role, timeframe);
      card.dataset.workMetaGrouped = 'true';
    });
  };

  const getAnalyticsCredentialGroup = (card) => {
    const text = `${card?.querySelector('img')?.alt || ''} ${card?.textContent || ''}`.toLowerCase();
    if (text.includes('google')) return 'google';
    if (text.includes('ibm')) return 'ibm';
    return 'degree';
  };

  const groupAnalyticsCredentialCards = (container, selector) => {
    if (!container || container.dataset.credentialsGrouped === 'true') return;
    const cards = $$(selector, container);
    cards.forEach((card) => {
      card.dataset.credentialGroup = getAnalyticsCredentialGroup(card);
    });
    ANALYTICS_CREDENTIAL_GROUPS.forEach((group) => {
      cards
        .filter((card) => card.dataset.credentialGroup === group)
        .forEach((card) => container.appendChild(card));
    });
    container.dataset.credentialsGrouped = 'true';
  };

  const ANALYTICS_SKILL_STORIES = Object.freeze([
    {
      href: 'portfolio/retailStore',
      label: 'BI & Reporting',
      tools: 'SQL, Tableau, KPI development',
      detail: 'Built 200+ dashboards and recurring KPI reports for decision-makers.'
    },
    {
      href: 'portfolio/retailStore',
      label: 'Automation & Data Quality',
      tools: 'Excel, Power Query, Python, Pandas',
      detail: 'Automated recurring workflows and validation logic, saving 200+ hours annually.'
    },
    {
      href: 'portfolio/targetEmptyPackage',
      label: 'Analysis & Decision Support',
      tools: 'Forecasting, anomaly detection, root-cause analysis',
      detail: 'Turned operational questions into clear, actionable findings.'
    },
    {
      href: 'resume',
      label: 'Web & Marketing Analytics',
      tools: 'GA4, campaign performance, conversion tracking',
      detail: 'Measured campaign impact, traveler behavior, and emerging search trends.'
    }
  ]);

  const ANALYTICS_PROJECT_EVIDENCE = Object.freeze([
    [
      'Question: Where are loss and margin signals changing?',
      'Approach: SQL ETL with anomaly scoring.',
      'Outcome: Faster, targeted investigations.'
    ],
    [
      'Question: How can shrink be forecast and reduced?',
      'Approach: Excel forecasting with live KPIs.',
      'Outcome: Clear action drivers for teams.'
    ],
    [
      'Question: How do delivery demand and staffing connect?',
      'Approach: Tableau trend and forecast views.',
      'Outcome: Faster shift-planning decisions.'
    ]
  ]);

  const prepareAnalyticsStory = () => {
    if (!document.body?.matches('[data-page="analytics"].home-pattern-page')) return;
    const main = document.getElementById('main');
    const panel = document.querySelector('.jump-panel');
    if (!main || !panel) return;

    groupAnalyticsWorkCardMeta(main);

    panel.dataset.storyRail = 'true';
    panel.setAttribute('aria-label', 'Explore Daniel Short\'s analytics story');
    const storyOrigin = main.querySelector('.hero-identity');
    if (storyOrigin && !storyOrigin.matches('.is-story-active, .is-story-complete')) {
      storyOrigin.classList.add('is-story-active');
    }
    const hideButton = panel.querySelector('[data-jump-hide]');
    const linkById = new Map();
    $$('.jump-panel-link', panel).forEach((link) => {
      const hash = samePageHashFromHref(link.getAttribute('href') || '');
      if (hash) linkById.set(hash.slice(1), link);
    });

    ANALYTICS_STORY_CHAPTERS.forEach((chapter, index) => {
      const target = document.getElementById(chapter.id);
      const link = linkById.get(chapter.id);
      if (!target || !link) return;
      const chapterNumber = String(index + 1).padStart(2, '0');
      target.classList.add('story-chapter');
      target.dataset.storyChapter = chapterNumber;
      const frame = target.querySelector(':scope > .wrapper, :scope > .cert-band-inner');
      frame?.classList.add('story-chapter__frame');
      const heading = target.querySelector('h2');
      if (heading) {
        heading.dataset.storyAnchor = 'true';
        heading.dataset.storyIndex = chapterNumber;
        if (chapter.id === 'certifications') heading.textContent = 'Education & Credentials';
      }

      link.dataset.storyIndex = chapterNumber;
      link.dataset.storyTarget = chapter.id;
      const label = link.querySelector('.jump-panel-text');
      if (label) label.textContent = chapter.label;
      panel.insertBefore(link, hideButton || null);
    });

    const projectCards = $$('#project-examples .project-examples-card', main);
    projectCards.forEach((card, index) => {
      const textPanel = card.querySelector('.project-text');
      const evidence = ANALYTICS_PROJECT_EVIDENCE[index];
      if (!textPanel || !evidence || textPanel.querySelector('.story-project-evidence')) return;
      const list = document.createElement('ul');
      list.className = 'story-project-evidence';
      evidence.forEach((line) => {
        const item = document.createElement('li');
        const separator = line.indexOf(':');
        const label = document.createElement('strong');
        label.textContent = separator >= 0 ? line.slice(0, separator + 1) : '';
        const detail = document.createElement('span');
        detail.textContent = separator >= 0 ? line.slice(separator + 1).trim() : line;
        item.append(label, detail);
        list.appendChild(item);
      });
      textPanel.appendChild(list);
    });

    const skillGrid = main.querySelector('#about-me .grid-container');
    if (skillGrid && skillGrid.dataset.storySkills !== 'true') {
      const existingCards = $$('.skill-link', skillGrid);
      const selectedCards = [existingCards[1], existingCards[2], existingCards[3], existingCards[5]].filter(Boolean);
      selectedCards.forEach((card, index) => {
        const story = ANALYTICS_SKILL_STORIES[index];
        if (!story) return;
        card.setAttribute('href', story.href);
        card.setAttribute('aria-label', `View evidence for ${story.label}`);
        const heading = card.querySelector('p');
        if (heading) {
          heading.textContent = story.label;
          const tools = document.createElement('small');
          tools.className = 'skill-link-btn';
          tools.textContent = story.tools;
          heading.appendChild(tools);
        }
        const detail = card.querySelector(':scope > small');
        if (detail) detail.textContent = story.detail;
      });
      skillGrid.replaceChildren(...selectedCards);
      skillGrid.dataset.storySkills = 'true';
    }

    groupAnalyticsCredentialCards(
      document.querySelector('#certifications .cert-track'),
      ':scope > .cert'
    );
    groupAnalyticsCredentialCards(
      document.querySelector('#certifications-modal .cert-modal-grid'),
      ':scope > .cert-card'
    );

    const ctaActions = main.querySelector('#cta-link > div');
    if (ctaActions && ctaActions.dataset.storyCta !== 'true') {
      ctaActions.classList.add('story-cta-actions');
      const resumeLink = document.createElement('a');
      resumeLink.className = 'btn-secondary';
      resumeLink.href = 'documents/Resume.pdf';
      resumeLink.textContent = 'Download resume';
      resumeLink.setAttribute('download', 'Daniel-Short-Resume.pdf');
      const linkedInLink = document.createElement('a');
      linkedInLink.className = 'btn-ghost';
      linkedInLink.href = 'https://www.linkedin.com/in/danielshort3/';
      linkedInLink.target = '_blank';
      linkedInLink.rel = 'noopener noreferrer';
      linkedInLink.textContent = 'View LinkedIn';
      ctaActions.append(resumeLink, linkedInLink);
      ctaActions.dataset.storyCta = 'true';
    }

    Object.entries(ANALYTICS_STORY_CARD_SELECTORS).forEach(([chapterId, selector]) => {
      const chapter = document.getElementById(chapterId);
      if (!chapter) return;
      $$(selector, chapter).forEach((card, index) => {
        card.classList.add('story-cascade-card');
        card.style.setProperty('--story-card-index', String(index));
        card.style.setProperty('--story-card-delay', `${90 + (index * 55)}ms`);
        if (card.classList.contains('project-examples-card')) card.classList.remove('ripple-in');
      });
    });
  };

  const loadedScripts = new Map();
  const portfolioBundles = new Map();
  let modalsPromise = null;
  let modalsHydrated = false;

  const resetScrollLocks = () => {
    const body = document.body;
    if (!body) return;
    const menu = document.getElementById('primary-menu');
    if (!menu || !menu.classList.contains('open')) {
      body.classList.remove('menu-open');
    }
    if (!document.querySelector('.modal.active')) {
      body.classList.remove('modal-open');
    }
    if (!document.querySelector('.media-viewer.active')) {
      body.classList.remove('media-viewer-open');
    }
  };

  window.addEventListener('pageshow', resetScrollLocks);
  document.addEventListener('DOMContentLoaded', () => {
    initClientErrorTelemetry();
    resetScrollLocks();
    normalizeAudienceSectionOrder();
    prepareAnalyticsStory();
    initSmoothScrollLinks();
    sortWorkCardsOldestFirst();
    updateWorkExperienceSummaries();
    if ((window.location && window.location.hash) === `#${CONTACT_MODAL_ID}`) {
      requestContactModal();
    }
    if (isPage('portfolio') || (document.body && document.body.matches('.portfolio-workbench-page') && document.querySelector('[data-portfolio-workbench]'))) {
      ensurePortfolioScripts(document.body.dataset.page || 'portfolio').then(() => {
        run(window.buildPortfolioCarousel);
        run(window.buildPortfolio);
      }).catch(err => console.warn('Failed to initialize portfolio page', err));
    }
    if (isPage('home')) {
      initSkillPopups();
    }
    if (isPage('home') || document.querySelector('.jump-panel')) {
      initJumpPanelSpy();
    }
    if (isPage('project')) {
      initProjectDemoTabs();
    }
  });
  document.addEventListener('site:content-updated', () => {
    normalizeAudienceSectionOrder();
    prepareAnalyticsStory();
    initSmoothScrollLinks();
    sortWorkCardsOldestFirst();
    updateWorkExperienceSummaries();
    if (isPage('home') || document.querySelector('.jump-panel')) {
      initJumpPanelSpy();
    }
    if (isPage('project')) {
      initProjectDemoTabs();
    }
  });
  trackContactOrigin();

  function loadScriptOnce(src){
    if (loadedScripts.has(src)) return loadedScripts.get(src);
    const promise = new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.src = src;
      tag.async = false;
      tag.onload = () => resolve();
      tag.onerror = () => reject(new Error(`Failed to load script: ${src}`));
      document.head.appendChild(tag);
    });
    loadedScripts.set(src, promise);
    return promise;
  }

  const ensureContactModal = () => {
    if (!document || !document.body || typeof document.createElement !== 'function') return null;
    const existing = document.getElementById(CONTACT_MODAL_ID);
    if (existing) return existing;
    const wrapper = document.createElement('div');
    wrapper.innerHTML = CONTACT_MODAL_MARKUP.trim();
    const modal = wrapper.firstElementChild;
    if (!modal) return null;
    document.body.appendChild(modal);
    return modal;
  };

  const ensureContactScript = () => {
    const promise = loadScriptOnce(CONTACT_MODAL_SCRIPT);
    if (promise && typeof promise.catch === 'function') {
      promise.catch(err => console.warn('Failed to load contact form script', err));
    }
    return promise;
  };

  const applyContactPrefill = (payload) => {
    if (!payload) return;
    const nameField = document.getElementById('contact-name');
    const emailField = document.getElementById('contact-email');
    const messageField = document.getElementById('contact-message');
    if (nameField && payload.name) nameField.value = payload.name;
    if (emailField && payload.email) emailField.value = payload.email;
    if (messageField && payload.message) messageField.value = payload.message;
  };

  const requestContactModal = (payload) => {
    storeContactOrigin();
    const ensured = document.getElementById(CONTACT_MODAL_ID) || ensureContactModal();
    if (!ensured) return;
    const open = () => {
      if (typeof window.openContactModal === 'function') {
        window.openContactModal();
        applyContactPrefill(payload);
        return;
      }
      applyContactPrefill(payload);
      try {
        if (location.hash !== `#${CONTACT_MODAL_ID}`) {
          location.hash = `#${CONTACT_MODAL_ID}`;
        }
      } catch {}
    };
    if (window.__contactModalReady) {
      open();
      return;
    }
    const scriptPromise = ensureContactScript();
    if (scriptPromise && typeof scriptPromise.then === 'function') {
      scriptPromise.then(open);
    } else {
      open();
    }
  };

  window.requestContactModal = requestContactModal;

  document.addEventListener('click', (event) => {
    const trigger = event.target.closest('[data-contact-modal-link]');
    if (!trigger) return;
    event.preventDefault();
    event.__contactHandled = true;
    requestContactModal();
  });

  function ensurePortfolioScripts(pageId = 'portfolio'){
    const normalizedPageId = String(pageId || 'portfolio').trim();
    const directoryDataScripts = {
      tools: 'js/portfolio/tools-directory-data.js',
      games: 'js/portfolio/games-directory-data.js'
    };
    const dataScript = directoryDataScripts[normalizedPageId] || 'js/portfolio/projects-data.js';
    if (portfolioBundles.has(normalizedPageId)) return portfolioBundles.get(normalizedPageId);
    const chain = [dataScript,'js/portfolio/modal-helpers.js','js/portfolio/portfolio.js']
      .reduce((p, src) => p.then(() => loadScriptOnce(src)), Promise.resolve());
    const bundle = chain.catch(err => {
      portfolioBundles.delete(normalizedPageId);
      throw err;
    });
    portfolioBundles.set(normalizedPageId, bundle);
    return bundle;
  }

  function hydrateProjectModals(){
    if (modalsHydrated) return;
    if (!Array.isArray(window.PROJECTS)) return;
    const host = $('#modals') || (() => {
      const d = document.createElement('div');
      d.id = 'modals';
      document.body.appendChild(d);
      return d;
    })();
    window.PROJECTS.forEach(p => {
      if (!p || p.published === false) return;
      if ($('#' + p.id + '-modal')) return;
      const modal = document.createElement('div');
      modal.className = 'modal';
      modal.id = `${p.id}-modal`;
      modal.innerHTML = window.generateProjectModal(p);
      host.appendChild(modal);
    });
    modalsHydrated = true;
  }

  function preparePortfolioModals(){
    if (modalsPromise) return modalsPromise;
    modalsPromise = ensurePortfolioScripts()
      .then(() => {
        if (typeof window.generateProjectModal !== 'function' || typeof window.openModal !== 'function') {
          throw new Error('Portfolio modal helpers missing');
        }
        hydrateProjectModals();
      })
      .catch(err => {
        console.warn('Failed to prepare project modals', err);
        modalsHydrated = false;
        modalsPromise = null;
        throw err;
      });
    return modalsPromise;
  }

  function openProjectModal(id){
    if (!id) return;
    const p = preparePortfolioModals();
    if (!p || typeof p.then !== 'function') return;
    return p.then(() => {
      if (typeof window.openModal === 'function') window.openModal(id);
    });
  }

	  function initSkillPopups(){
	    if (!isPage('home')) return;
	    const buttons = $$('[data-project-modal="true"]');
	    if (!buttons.length) return;

    const safePreload = () => {
      const promise = preparePortfolioModals();
      if (promise && typeof promise.then === 'function') promise.catch(() => {});
    };

    buttons.forEach(btn => {
      if (btn.dataset.modalBound === 'yes') return;
      btn.dataset.modalBound = 'yes';
      btn.setAttribute('aria-haspopup', 'dialog');
      on(btn, 'pointerenter', safePreload, { once: true });
      on(btn, 'focus', safePreload, { once: true });
      on(btn, 'touchstart', safePreload, { once: true, passive: true });
      on(btn, 'click', (evt) => {
        if (!btn.dataset.project) return;
        if (evt && (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey)) return;
        if (evt && typeof evt.button === 'number' && evt.button !== 0) return;
        const tag = (btn.tagName || '').toLowerCase();
        if (tag === 'a') evt.preventDefault();
        const promise = openProjectModal(btn.dataset.project);
        if (promise && typeof promise.catch === 'function') promise.catch(() => {});
      });
      on(btn, 'keydown', (evt) => {
        const tag = (btn.tagName || '').toLowerCase();
        if (tag === 'a') return;
        if (evt.key === 'Enter' || evt.key === ' ') {
          evt.preventDefault();
          if (!btn.dataset.project) return;
          const promise = openProjectModal(btn.dataset.project);
          if (promise && typeof promise.catch === 'function') promise.catch(() => {});
        }
      });
    });

    const projectFromLocation = () => {
      try {
        const search = (location.search || '').replace(/^\?/, '');
        if (search) {
          const parts = search.split('&');
          for (const kv of parts) {
            const [k, v] = kv.split('=');
            if (decodeURIComponent(k) === 'project' && v) return decodeURIComponent(v);
          }
        }
        if (location.hash && location.hash.length > 1) {
          const hashId = decodeURIComponent(location.hash.slice(1));
          if (!document.getElementById(hashId)) return hashId;
        }
      } catch {
        if (location.hash && location.hash.length > 1) {
          const hashId = location.hash.slice(1);
          if (!document.getElementById(hashId)) return hashId;
        }
      }
      return null;
    };

    const deepLinkId = projectFromLocation();
    if (deepLinkId) {
      const promise = openProjectModal(deepLinkId);
      if (promise && typeof promise.catch === 'function') promise.catch(() => {});
    }
  }

  function initSmoothScrollLinks(){
    const links = $$('a[data-smooth-scroll="true"]');
    if (!links.length) return;

    const prefersReducedMotion = () => {
      try {
        return Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
      } catch {
        return false;
      }
    };

    const focusScrollTarget = (target) => {
      if (!target || typeof target.focus !== 'function') return;
      const isNaturallyFocusable = target.matches?.('a[href], button, input, select, textarea, summary, [tabindex]');
      const needsTemporaryTabindex = !isNaturallyFocusable;
      if (needsTemporaryTabindex) target.setAttribute('tabindex', '-1');
      try {
        target.focus({ preventScroll: true });
      } catch {
        target.focus();
      }
      if (!needsTemporaryTabindex) return;
      const restoreTabindex = () => {
        if (target.getAttribute('tabindex') === '-1') target.removeAttribute('tabindex');
      };
      target.addEventListener('blur', restoreTabindex, { once: true });
    };

    const smoothScrollToTarget = (target, options = {}) => {
      if (!target) return;

      if (typeof activeSmoothScrollCancel === 'function') {
        activeSmoothScrollCancel();
      }

      const openAncestorDetails = (node) => {
        let current = node;
        while (current && current.closest) {
          const details = current.closest('details');
          if (!details) break;
          if (!details.open) details.open = true;
          current = details.parentElement;
        }
      };
      openAncestorDetails(target);

      const startTop = window.scrollY || window.pageYOffset || 0;
      const navOffset = typeof window.getNavOffset === 'function' ? window.getNavOffset() : 0;
      let marginTop = 0;
      try {
        marginTop = parseFloat(window.getComputedStyle(target).scrollMarginTop || '0') || 0;
      } catch {}
      const targetTop = target.getBoundingClientRect().top + startTop - navOffset - marginTop;
      const maxScroll = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      const clampedTop = Math.min(Math.max(0, targetTop), maxScroll);
      const distance = clampedTop - startTop;

      const jumpPanelToken = options.jumpPanel ? beginJumpPanelAutoScroll() : 0;
      let jumpPanelTailTimer = null;
      let rafId = null;
      let startTime = null;
      let cleaned = false;
      const duration = Math.min(1200, Math.max(260, Math.abs(distance) * 0.6));
      const ease = (t) => (
        t < 0.5
          ? 2 * t * t
          : 1 - Math.pow(-2 * t + 2, 2) / 2
      );

      const releaseJumpPanelAutoScroll = (delayMs = 0) => {
        if (!jumpPanelToken) return;
        if (jumpPanelTailTimer) {
          clearTimeout(jumpPanelTailTimer);
          jumpPanelTailTimer = null;
        }
        if (delayMs > 0) {
          jumpPanelTailTimer = setTimeout(() => {
            endJumpPanelAutoScroll(jumpPanelToken);
          }, delayMs);
          return;
        }
        endJumpPanelAutoScroll(jumpPanelToken);
      };

      const cleanup = ({ cancelled = false } = {}) => {
        if (cleaned) return;
        cleaned = true;
        if (rafId) cancelAnimationFrame(rafId);
        window.removeEventListener('wheel', cancel);
        window.removeEventListener('touchstart', cancel);
        window.removeEventListener('pointerdown', cancel);
        window.removeEventListener('keydown', onKeydown);
        if (activeSmoothScrollCancel === cancel) activeSmoothScrollCancel = null;
        if (jumpPanelToken) {
          if (cancelled) {
            releaseJumpPanelAutoScroll();
          } else {
            releaseJumpPanelAutoScroll(180);
          }
        }
        if (!cancelled) focusScrollTarget(target);
      };

      const cancel = () => {
        cleanup({ cancelled: true });
      };

      const onKeydown = (event) => {
        const key = event?.key;
        if (!key) return;
        if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(key)) {
          cancel();
        }
      };

      const step = (now) => {
        if (cleaned) return;
        if (startTime === null) startTime = now;
        const progress = Math.min(1, (now - startTime) / duration);
        const nextTop = startTop + distance * ease(progress);
        window.scrollTo(0, nextTop);
        if (progress < 1) {
          rafId = requestAnimationFrame(step);
        } else {
          cleanup();
        }
      };

      if (options.reducedMotion || Math.abs(distance) < 1) {
        if (Math.abs(distance) >= 1) {
          try {
            window.scrollTo({ top: clampedTop, left: 0, behavior: 'auto' });
          } catch {
            window.scrollTo(0, clampedTop);
          }
        }
        cleanup();
        return;
      }

      window.addEventListener('wheel', cancel, { passive: true });
      window.addEventListener('touchstart', cancel, { passive: true });
      window.addEventListener('pointerdown', cancel, { passive: true });
      window.addEventListener('keydown', onKeydown);
      activeSmoothScrollCancel = cancel;
      rafId = requestAnimationFrame(step);
    };

    links.forEach((link) => {
      if (link.dataset.smoothBound === 'yes') return;
      link.dataset.smoothBound = 'yes';
      on(link, 'click', (evt) => {
        if (evt && (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey)) return;
        if (evt && typeof evt.button === 'number' && evt.button !== 0) return;

        const href = link.getAttribute('href') || '';
        const hash = samePageHashFromHref(href);
        if (!hash) return;
        const targetId = decodeURIComponent(hash.slice(1));
        const target = document.getElementById(targetId);
        if (!target) return;
        const isJumpPanelLink = Boolean(link.closest('.jump-panel'));
        evt.preventDefault();
        smoothScrollToTarget(target, {
          jumpPanel: isJumpPanelLink,
          reducedMotion: prefersReducedMotion()
        });
        try {
          history.pushState(null, '', currentHashUrl(hash));
        } catch {}
      });
    });
  }

  function initProjectDemoTabs() {
    const shells = $$('[data-demo-tabs="true"]');
    if (!shells.length) return;

    shells.forEach((shell) => {
      const tabs = $$('[role="tab"]', shell);
      const panels = $$('[role="tabpanel"]', shell);
      if (tabs.length < 2 || panels.length < 2) return;

      const panelById = new Map(
        panels
          .map((panel) => [panel.id, panel])
          .filter(([id]) => Boolean(id))
      );

      const getPanelForTab = (tab) => {
        if (!tab) return null;
        const panelId = tab.getAttribute('aria-controls');
        if (!panelId) return null;
        return panelById.get(panelId) || document.getElementById(panelId);
      };

      const loadPanelIframes = (panel) => {
        if (!panel) return;
        $$('iframe[data-src]', panel).forEach((iframe) => {
          if (!iframe || iframe.getAttribute('src')) return;
          const dataSrc = iframe.getAttribute('data-src');
          if (!dataSrc) return;
          iframe.setAttribute('src', dataSrc);
          iframe.removeAttribute('data-src');
        });
      };

      const setActiveTab = (nextTab, { focus = false } = {}) => {
        if (!nextTab) return;
        const nextPanel = getPanelForTab(nextTab);
        if (!nextPanel) return;

        tabs.forEach((tab) => {
          const active = tab === nextTab;
          tab.classList.toggle('is-active', active);
          tab.setAttribute('aria-selected', String(active));
          if (active) {
            tab.removeAttribute('tabindex');
          } else {
            tab.setAttribute('tabindex', '-1');
          }
        });

        panels.forEach((panel) => {
          const active = panel === nextPanel;
          panel.classList.toggle('is-active', active);
          if (active) {
            panel.removeAttribute('hidden');
          } else {
            panel.setAttribute('hidden', '');
          }
        });

        loadPanelIframes(nextPanel);
        if (focus) nextTab.focus();
      };

      tabs.forEach((tab) => {
        on(tab, 'click', () => setActiveTab(tab));
        on(tab, 'keydown', (event) => {
          const key = event?.key;
          if (!key) return;
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key)) return;
          event.preventDefault();
          const currentIndex = tabs.indexOf(tab);
          if (currentIndex === -1) return;
          const lastIndex = tabs.length - 1;
          const nextIndex = (() => {
            if (key === 'Home') return 0;
            if (key === 'End') return lastIndex;
            if (key === 'ArrowLeft') return currentIndex === 0 ? lastIndex : currentIndex - 1;
            return currentIndex === lastIndex ? 0 : currentIndex + 1;
          })();
          const nextTab = tabs[nextIndex];
          setActiveTab(nextTab, { focus: true });
        });
      });

      on(shell, 'click', (event) => {
        const trigger = event.target.closest('[data-demo-tabs-open="demo"]');
        if (!trigger) return;
        const demoTab = tabs.find((tab) => tab.textContent.trim().toLowerCase() === 'demo') || tabs[1];
        setActiveTab(demoTab, { focus: true });
      });
    });
  }

  const setupProfessionalJumpRail = (panel, links) => {
    const isProfessionalPage = document.body?.matches('[data-page="analytics"], [data-page="data-science"], [data-page="tourism"]');
    if (!isProfessionalPage || !panel || !links.length) return null;

    let track = panel.querySelector('[data-jump-rail-track]');
    let previousButton = panel.querySelector('[data-jump-rail-step="previous"]');
    let nextButton = panel.querySelector('[data-jump-rail-step="next"]');

    if (!track) {
      track = document.createElement('div');
      track.className = 'jump-panel-track';
      track.dataset.jumpRailTrack = 'true';
      track.id = `${panel.id || 'jump-panel'}-track`;
      panel.insertBefore(track, links[0]);
      links.forEach((link) => track.appendChild(link));
    }

    const createStepButton = (direction, label, path) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `jump-panel-step jump-panel-step--${direction}`;
      button.dataset.jumpRailStep = direction;
      button.setAttribute('aria-label', label);
      button.setAttribute('aria-controls', track.id);
      button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"></path></svg>`;
      return button;
    };

    if (!previousButton) {
      previousButton = createStepButton('previous', 'Show previous section links', 'm15 18-6-6 6-6');
      panel.insertBefore(previousButton, track);
    }
    if (!nextButton) {
      nextButton = createStepButton('next', 'Show next section links', 'm9 18 6-6-6-6');
      track.insertAdjacentElement('afterend', nextButton);
    }

    const prefersReducedMotion = () => {
      try {
        return Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
      } catch {
        return false;
      }
    };
    const getMaxScroll = () => Math.max(0, track.scrollWidth - track.clientWidth);
    const scrollTrackTo = (left) => {
      const nextLeft = Math.min(getMaxScroll(), Math.max(0, left));
      try {
        track.scrollTo({ left: nextLeft, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
      } catch {
        track.scrollLeft = nextLeft;
      }
    };
    const updateControls = () => {
      const maxScroll = getMaxScroll();
      const hasOverflow = maxScroll > 2;
      const canScrollPrevious = hasOverflow && track.scrollLeft > 2;
      const canScrollNext = hasOverflow && track.scrollLeft < maxScroll - 2;
      previousButton.disabled = !canScrollPrevious;
      nextButton.disabled = !canScrollNext;
      panel.classList.toggle('has-overflow', hasOverflow);
      panel.classList.toggle('can-scroll-prev', canScrollPrevious);
      panel.classList.toggle('can-scroll-next', canScrollNext);
    };
    const centerLink = (link) => {
      if (!link || getMaxScroll() <= 2) {
        updateControls();
        return;
      }
      const left = link.offsetLeft - ((track.clientWidth - link.offsetWidth) / 2);
      scrollTrackTo(left);
      requestAnimationFrame(updateControls);
    };
    const stepTrack = (direction) => {
      const center = track.scrollLeft + (track.clientWidth / 2);
      const centers = links.map((link) => ({
        link,
        center: link.offsetLeft + (link.offsetWidth / 2)
      }));
      const currentIndex = centers.reduce((bestIndex, entry, index) => (
        Math.abs(entry.center - center) < Math.abs(centers[bestIndex].center - center)
          ? index
          : bestIndex
      ), 0);
      const targetIndex = Math.min(
        centers.length - 1,
        Math.max(0, currentIndex + (direction > 0 ? 1 : -1))
      );
      const target = centers[targetIndex];
      if (target) centerLink(target.link);
    };

    if (panel.dataset.jumpRailControlsBound !== 'yes') {
      panel.dataset.jumpRailControlsBound = 'yes';
      previousButton.addEventListener('click', () => stepTrack(-1));
      nextButton.addEventListener('click', () => stepTrack(1));
      track.addEventListener('scroll', updateControls, { passive: true });
    }
    requestAnimationFrame(updateControls);
    return { centerLink, updateControls };
  };

  function initJumpPanelSpy(){
    const panel = document.querySelector('.jump-panel');
    if (!panel) return;
    const isStoryRail = panel.dataset.storyRail === 'true';
    const storyMain = isStoryRail ? (panel.closest('main') || document.getElementById('main')) : null;
    const storyOrigin = storyMain?.querySelector('.hero-identity') || null;
    const links = $$('.jump-panel-link', panel);
    const railControls = setupProfessionalJumpRail(panel, links);
    const hideBtn = panel.querySelector('[data-jump-hide]');
    const showBtn = document.querySelector('[data-jump-show]');
    if (!links.length) return;
    if (panel.dataset.jumpPanelSpyBound === 'yes') return;
    panel.dataset.jumpPanelSpyBound = 'yes';
    const focusables = [...links];
    if (hideBtn) focusables.push(hideBtn);
    const items = links.map((link) => {
      const href = link.getAttribute('href') || '';
      const hash = samePageHashFromHref(href);
      if (!hash) return null;
      let id = hash.slice(1);
      try {
        id = decodeURIComponent(id);
      } catch {}
      const target = document.getElementById(id);
      if (!target) return null;
      return { id, link, target };
    }).filter(Boolean);
    if (!items.length) return;

    let activeId = null;
    let ticking = false;
    let manualOverrideId = null;
    let storyStartY = 0;
    let storyEndY = 0;
    let storyMeasureToken = 0;
    let storyAnimationsReady = false;
    let storyRenderedFill = 0;
    let storyTargetIndex = -1;
    let storyAnimationFrame = 0;
    let storyAnimationToken = 0;
    let storyOriginConfirmTimer = 0;
    let storyLastScrollY = window.scrollY || window.pageYOffset || 0;
    let storyInitialized = false;
    let storyInputIntent = false;
    const STORY_FIRST_ACTIVATE_RATIO = .60;
    const STORY_ACTIVATE_RATIO = .70;
    const STORY_FIRST_RETRACT_RATIO = .64;
    const STORY_RETRACT_RATIO = .78;
    const STORY_STEP_DURATION_MS = 620;
    const STORY_MAX_DURATION_MS = 850;
    const storyMotionQuery = typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;

    const prefersReducedStoryMotion = () => Boolean(storyMotionQuery?.matches);
    const clearStoryOriginConfirmation = () => {
      if (!storyOrigin) return;
      if (storyOriginConfirmTimer) {
        window.clearTimeout(storyOriginConfirmTimer);
        storyOriginConfirmTimer = 0;
      }
      storyOrigin.classList.remove('story-node-confirm');
    };
    const renderStoryOriginState = (isComplete, { confirm = true } = {}) => {
      if (!storyOrigin) return;
      const wasComplete = storyOrigin.classList.contains('is-story-complete');
      storyOrigin.classList.toggle('is-story-active', !isComplete);
      storyOrigin.classList.toggle('is-story-complete', isComplete);
      if (!isComplete) {
        clearStoryOriginConfirmation();
        return;
      }
      if (
        !wasComplete &&
        confirm &&
        storyAnimationsReady &&
        !prefersReducedStoryMotion()
      ) {
        clearStoryOriginConfirmation();
        storyOrigin.classList.add('story-node-confirm');
        storyOriginConfirmTimer = window.setTimeout(() => {
          clearStoryOriginConfirmation();
        }, 420);
      }
    };
    const storyScrollKeys = new Set([
      'ArrowDown',
      'ArrowUp',
      'End',
      'Home',
      'PageDown',
      'PageUp',
      ' ',
      'Spacebar'
    ]);
    const markStoryInputIntent = (event) => {
      if (event?.type === 'keydown' && !storyScrollKeys.has(event.key)) return;
      storyInputIntent = true;
    };

    const getNavOffset = () => {
      if (typeof window.getNavOffset === 'function') {
        return window.getNavOffset();
      }
      return 72;
    };

    const setActive = (id) => {
      const activeIndex = items.findIndex((item) => item.id === id);
      items.forEach((item) => {
        const isActive = item.id === id;
        item.link.classList.toggle('is-active', isActive);
        item.link.classList.toggle('is-complete', activeIndex >= 0 && items.indexOf(item) < activeIndex);
        if (isStoryRail) {
          item.target.classList.toggle('is-current', isActive);
          item.target.classList.toggle('is-complete', activeIndex >= 0 && items.indexOf(item) < activeIndex);
        }
        if (isActive) {
          item.link.setAttribute('aria-current', 'location');
        } else {
          item.link.removeAttribute('aria-current');
        }
      });
      const activeItem = activeIndex >= 0 ? items[activeIndex] : null;
      if (activeItem) railControls?.centerLink(activeItem.link);
      railControls?.updateControls();
    };

    const remapStoryFill = (fill, previousStops, nextStops) => {
      const oldBreakpoints = [0, ...previousStops];
      const newBreakpoints = [0, ...nextStops];
      if (oldBreakpoints.length !== newBreakpoints.length || oldBreakpoints.length < 2) {
        return fill;
      }
      for (let index = 0; index < oldBreakpoints.length - 1; index += 1) {
        const oldStart = oldBreakpoints[index];
        const oldEnd = oldBreakpoints[index + 1];
        if (fill > oldEnd && index < oldBreakpoints.length - 2) continue;
        const span = Math.max(1, oldEnd - oldStart);
        const progress = Math.min(1, Math.max(0, (fill - oldStart) / span));
        const newStart = newBreakpoints[index];
        const newEnd = newBreakpoints[index + 1];
        return newStart + ((newEnd - newStart) * progress);
      }
      return newBreakpoints[newBreakpoints.length - 1];
    };

    const measureStoryRail = () => {
      if (!isStoryRail) return;
      const main = panel.closest('main') || document.getElementById('main');
      if (!main) return;
      const previousStops = items.map((item) => (
        Number.isFinite(item.storyStopY) ? Math.max(0, item.storyStopY - storyStartY) : 0
      ));
      const previousFill = storyRenderedFill;
      const hadStoryGeometry = storyInitialized && storyEndY > storyStartY;
      const scrollTop = window.scrollY || window.pageYOffset || 0;
      const mainTop = main.getBoundingClientRect().top + scrollTop;
      const heroIdentity = main.querySelector('.hero-identity');
      const heroOrigin = heroIdentity || main.querySelector('.hero h1');
      const heroOriginRect = heroOrigin?.getBoundingClientRect();
      const heroWrapper = main.querySelector('.hero.hero--default > .wrapper');
      const panelRect = panel.getBoundingClientRect();
      const panelStyle = window.getComputedStyle(panel);
      const storyRailOffset = Number.parseFloat(panelStyle.getPropertyValue('--story-rail-x')) || 23;
      const storyOriginClearance = Number.parseFloat(panelStyle.getPropertyValue('--story-origin-clearance')) || 0;
      const isMobileStoryRail = window.matchMedia('(max-width: 768px)').matches;
      if (heroIdentity && heroWrapper) {
        const identityRect = heroIdentity.getBoundingClientRect();
        const wrapperRect = heroWrapper.getBoundingClientRect();
        const originY = identityRect.top - wrapperRect.top + (identityRect.height / 2);
        const railX = panelRect.left + storyRailOffset;
        const connectorWidth = Math.max(0, identityRect.left - railX);
        heroWrapper.style.setProperty('--story-hero-origin-y', `${Math.max(0, originY)}px`);
        heroIdentity.style.setProperty('--story-hero-connector-width', `${connectorWidth}px`);
      }
      const stops = items.map((item) => {
        const anchor = item.target.querySelector('[data-story-anchor]') || item.target.querySelector('h2') || item.target;
        const anchorRect = anchor.getBoundingClientRect();
        const targetRect = item.target.getBoundingClientRect();
        const stopY = anchorRect.top + (window.scrollY || window.pageYOffset || 0) - mainTop + Math.min(20, anchorRect.height / 2);
        item.storyStopY = Math.max(0, stopY);
        item.link.style.setProperty('--story-stop-y', `${Math.max(0, stopY)}px`);
        item.target.style.setProperty('--story-anchor-y', `${Math.max(0, anchorRect.top - targetRect.top + Math.min(20, anchorRect.height / 2))}px`);
        const connectorTarget = item.id === 'cta'
          ? item.target.querySelector('#cta-link')
          : null;
        if (isMobileStoryRail || connectorTarget) {
          const branchStyle = window.getComputedStyle(item.target, '::after');
          const branchLeft = Number.parseFloat(branchStyle.left) || 0;
          let connectorLeft = connectorTarget?.getBoundingClientRect().left ?? anchorRect.left;
          if (isMobileStoryRail && !connectorTarget && document.createRange) {
            const textRange = document.createRange();
            textRange.selectNodeContents(anchor);
            const textRect = textRange.getBoundingClientRect();
            if (textRect.width > 0) connectorLeft = textRect.left;
          }
          const branchGap = connectorTarget ? 0 : 4;
          const branchWidth = Math.max(0, connectorLeft - targetRect.left - branchLeft - branchGap);
          item.target.style.setProperty('--story-branch-width', `${branchWidth}px`);
        }
        return stopY;
      });
      if (!stops.length) return;
      storyStartY = heroOriginRect
        ? Math.max(0, heroOriginRect.top + scrollTop - mainTop + (heroOriginRect.height / 2) + storyOriginClearance)
        : Math.max(0, stops[0]);
      storyEndY = Math.max(storyStartY, stops[stops.length - 1]);
      panel.style.setProperty('--story-start-y', `${storyStartY}px`);
      panel.style.setProperty('--story-end-y', `${storyEndY}px`);
      panel.classList.add('is-rail-ready');
      if (hadStoryGeometry) {
        const nextStops = items.map((item) => getStoryStopDistance(items.indexOf(item)));
        const geometryChanged = nextStops.some((stop, index) => Math.abs(stop - previousStops[index]) > .5);
        if (geometryChanged) {
          const remappedFill = remapStoryFill(previousFill, previousStops, nextStops);
          renderStoryFill(remappedFill, { confirm: false });
          retargetStoryMilestone(storyTargetIndex, { force: true, confirm: false });
        }
      }
    };

    const requestStoryMeasure = () => {
      if (!isStoryRail || storyMeasureToken) return;
      storyMeasureToken = requestAnimationFrame(() => {
        storyMeasureToken = 0;
        measureStoryRail();
        requestUpdate();
      });
    };

    const clampStoryFill = (value) => Math.min(
      Math.max(0, storyEndY - storyStartY),
      Math.max(0, Number(value) || 0)
    );

    const storyEaseInOutCubic = (progress) => (
      progress < .5
        ? 4 * progress * progress * progress
        : 1 - (Math.pow((-2 * progress) + 2, 3) / 2)
    );

    const getStoryStopDistance = (index) => {
      if (index < 0) return 0;
      const stopY = Number.isFinite(items[index]?.storyStopY)
        ? items[index].storyStopY
        : storyEndY;
      return clampStoryFill(stopY - storyStartY);
    };

    const clearStoryConfirmation = (item) => {
      if (!item) return;
      if (item.storyConfirmTimer) {
        window.clearTimeout(item.storyConfirmTimer);
        item.storyConfirmTimer = 0;
      }
      item.link.classList.remove('story-node-confirm');
      item.target.classList.remove('is-segment-confirming');
    };

    const renderStoryFill = (nextFill, { confirm = true } = {}) => {
      const previousFill = storyRenderedFill;
      storyRenderedFill = clampStoryFill(nextFill);
      const fillHeadY = storyStartY + storyRenderedFill;
      const movingForward = storyRenderedFill >= previousFill;
      panel.style.setProperty('--story-fill-y', `${storyRenderedFill}px`);
      items.forEach((item) => {
        const stopY = Number.isFinite(item.storyStopY) ? item.storyStopY : storyEndY;
        const cellRamp = 120;
        const rawProgress = Math.min(1, Math.max(0, (fillHeadY - (stopY - cellRamp)) / cellRamp));
        const cellProgress = rawProgress * rawProgress * (3 - (2 * rawProgress));
        const wasComplete = item.target.classList.contains('is-segment-complete');
        const isComplete = wasComplete ? rawProgress >= .92 : rawProgress >= .999;
        item.target.style.setProperty('--story-cell-progress', cellProgress.toFixed(4));
        item.target.style.setProperty('--story-cell-fill', `${(cellProgress * 100).toFixed(2)}%`);
        item.target.style.setProperty('--story-title-opacity', (.68 + (.32 * cellProgress)).toFixed(3));
        item.target.style.setProperty('--story-title-shift', `${((1 - cellProgress) * 6).toFixed(2)}px`);
        item.target.classList.toggle('is-segment-complete', isComplete);
        item.link.classList.toggle('is-segment-complete', isComplete);
        if (isComplete) item.target.classList.add('has-revealed-cards');
        if (!isComplete) clearStoryConfirmation(item);
        if (
          !wasComplete &&
          isComplete &&
          movingForward &&
          confirm &&
          storyAnimationsReady &&
          !prefersReducedStoryMotion()
        ) {
          clearStoryConfirmation(item);
          item.link.classList.add('story-node-confirm');
          item.target.classList.add('is-segment-confirming');
          item.storyConfirmTimer = window.setTimeout(() => {
            clearStoryConfirmation(item);
          }, 420);
        }
      });
    };

    const getReachedStoryIndex = () => {
      let reachedIndex = -1;
      items.forEach((item, index) => {
        if (getStoryStopDistance(index) <= storyRenderedFill + .5) reachedIndex = index;
      });
      return reachedIndex;
    };

    const retargetStoryMilestone = (index, { immediate = false, confirm = true, force = false } = {}) => {
      if (!isStoryRail || storyEndY <= storyStartY) return;
      const numericIndex = Number(index);
      const normalizedIndex = Number.isFinite(numericIndex)
        ? Math.min(items.length - 1, Math.max(-1, numericIndex))
        : -1;
      if (!force && normalizedIndex === storyTargetIndex) return;
      const previousTargetIndex = storyTargetIndex;
      storyTargetIndex = normalizedIndex;
      renderStoryOriginState(normalizedIndex >= 0, { confirm });
      panel.dataset.storyMilestone = normalizedIndex < 0
        ? 'start'
        : String(normalizedIndex + 1).padStart(2, '0');
      const targetFill = getStoryStopDistance(normalizedIndex);
      storyAnimationToken += 1;
      const animationToken = storyAnimationToken;
      if (storyAnimationFrame) {
        cancelAnimationFrame(storyAnimationFrame);
        storyAnimationFrame = 0;
      }
      const delta = targetFill - storyRenderedFill;
      if (immediate || prefersReducedStoryMotion() || Math.abs(delta) < .5) {
        panel.dataset.storyAnimating = 'false';
        renderStoryFill(targetFill, { confirm: false });
        return;
      }
      const reachedIndex = getReachedStoryIndex();
      const crossedMilestones = Math.max(
        1,
        Math.abs(normalizedIndex - (reachedIndex >= 0 ? reachedIndex : previousTargetIndex))
      );
      const duration = Math.min(
        STORY_MAX_DURATION_MS,
        STORY_STEP_DURATION_MS + (Math.max(0, crossedMilestones - 1) * 115)
      );
      const startFill = storyRenderedFill;
      const startTime = performance.now();
      panel.dataset.storyAnimating = 'true';
      const stepStoryAnimation = (time) => {
        if (animationToken !== storyAnimationToken) return;
        const elapsed = Math.min(1, Math.max(0, (time - startTime) / duration));
        renderStoryFill(startFill + (delta * storyEaseInOutCubic(elapsed)), { confirm });
        if (elapsed < 1) {
          storyAnimationFrame = requestAnimationFrame(stepStoryAnimation);
          return;
        }
        storyAnimationFrame = 0;
        panel.dataset.storyAnimating = 'false';
        renderStoryFill(targetFill, { confirm });
      };
      storyAnimationFrame = requestAnimationFrame(stepStoryAnimation);
    };

    const resetStoryToStart = () => {
      storyInputIntent = false;
      manualOverrideId = null;
      storyLastScrollY = window.scrollY || window.pageYOffset || 0;
      items.forEach((item) => {
        clearStoryConfirmation(item);
        item.target.classList.remove('has-revealed-cards', 'is-current', 'is-complete', 'is-segment-complete');
        item.link.classList.remove('is-active', 'is-complete', 'is-segment-complete');
        item.link.removeAttribute('aria-current');
        item.target.style.setProperty('--story-cell-progress', '0');
        item.target.style.setProperty('--story-cell-fill', '0%');
        item.target.style.setProperty('--story-title-opacity', '.68');
        item.target.style.setProperty('--story-title-shift', '6px');
      });
      storyRenderedFill = 0;
      storyTargetIndex = -1;
      panel.dataset.storyMilestone = 'start';
      panel.dataset.storyAnimating = 'false';
      panel.style.setProperty('--story-fill-y', '0px');
      retargetStoryMilestone(-1, { immediate: true, confirm: false, force: true });
      activeId = null;
      setActive(null);
    };

    const getStoryAnchorTop = (item) => {
      const anchor = item?.target.querySelector('[data-story-anchor]') || item?.target;
      return anchor ? anchor.getBoundingClientRect().top : Number.POSITIVE_INFINITY;
    };

    const getStoryActivationLine = (index, viewportHeight) => (
      viewportHeight * (index === 0 ? STORY_FIRST_ACTIVATE_RATIO : STORY_ACTIVATE_RATIO)
    );

    const getStoryRetractionLine = (index, viewportHeight) => (
      viewportHeight * (index === 0 ? STORY_FIRST_RETRACT_RATIO : STORY_RETRACT_RATIO)
    );

    const resolveStoryMilestone = (direction, viewportHeight, atBottom, { initial = false } = {}) => {
      if (atBottom) return items.length - 1;
      let nextIndex = initial ? -1 : storyTargetIndex;
      if (direction > 0 || initial) {
        while (nextIndex + 1 < items.length) {
          const candidateIndex = nextIndex + 1;
          if (
            getStoryAnchorTop(items[candidateIndex]) > getStoryActivationLine(candidateIndex, viewportHeight)
          ) break;
          nextIndex += 1;
        }
      } else if (direction < 0) {
        while (
          nextIndex >= 0 &&
          getStoryAnchorTop(items[nextIndex]) > getStoryRetractionLine(nextIndex, viewportHeight)
        ) {
          nextIndex -= 1;
        }
      }
      return nextIndex;
    };

    const storyIndexFromHash = () => {
      const hash = String(window.location?.hash || '').replace(/^#/, '');
      if (!hash) return -1;
      let id = hash;
      try {
        id = decodeURIComponent(hash);
      } catch {}
      return items.findIndex((item) => item.id === id);
    };

    const initializeStoryMilestone = (scrollTop, viewportHeight, atBottom) => {
      const hashIndex = storyIndexFromHash();
      const initialIndex = hashIndex >= 0
        ? hashIndex
        : -1;
      storyInitialized = true;
      retargetStoryMilestone(initialIndex, { immediate: true, confirm: false, force: true });
      if (initialIndex < 0) {
        activeId = null;
        setActive(null);
      }
      storyLastScrollY = scrollTop;
      storyAnimationsReady = true;
      return initialIndex;
    };

    const setManualActive = (item, event) => {
      if (!item) return;
      if (event && (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)) return;
      if (event && typeof event.button === 'number' && event.button !== 0) return;
      manualOverrideId = item.id;
      if (isStoryRail) {
        const itemIndex = items.indexOf(item);
        retargetStoryMilestone(itemIndex, { immediate: prefersReducedStoryMotion() });
      }
      activeId = item.id;
      setActive(item.id);
    };

    const blurAfterPointer = (link) => {
      requestAnimationFrame(() => {
        if (document.activeElement === link) link.blur();
      });
    };

    const bindPointerBlur = (link) => {
      if (link.dataset.jumpBlur === 'yes') return;
      link.dataset.jumpBlur = 'yes';
      if ('PointerEvent' in window) {
        link.addEventListener('pointerup', (event) => {
          const type = event?.pointerType;
          if (type && !['mouse', 'touch', 'pen'].includes(type)) return;
          blurAfterPointer(link);
        });
      } else {
        link.addEventListener('mouseup', (event) => {
          if (event && event.button && event.button !== 0) return;
          blurAfterPointer(link);
        });
        link.addEventListener('touchend', () => blurAfterPointer(link), { passive: true });
      }
    };

    const shouldCondenseOnScroll = () => {
      try {
        return Boolean(window.matchMedia && window.matchMedia('(hover: none) and (pointer: coarse), (max-width: 768px)').matches);
      } catch {
        return true;
      }
    };

    const setPanelCondensed = (condensed) => {
      if (!shouldCondenseOnScroll()) return;
      panel.classList.toggle('is-condensed', condensed);
    };

    const clearPanelFocusOnScroll = () => {
      if (!shouldCondenseOnScroll()) return;
      const active = document.activeElement;
      if (active && panel.contains(active)) active.blur();
    };

    const setFocusable = (el, hidden) => {
      if (!el) return;
      if (hidden) {
        if (!Object.prototype.hasOwnProperty.call(el.dataset, 'jumpTabindex')) {
          const prev = el.getAttribute('tabindex');
          el.dataset.jumpTabindex = prev === null ? '' : prev;
        }
        el.setAttribute('tabindex', '-1');
        return;
      }
      if (!Object.prototype.hasOwnProperty.call(el.dataset, 'jumpTabindex')) {
        return;
      }
      const prev = el.dataset.jumpTabindex;
      delete el.dataset.jumpTabindex;
      if (prev === '') {
        el.removeAttribute('tabindex');
      } else {
        el.setAttribute('tabindex', prev);
      }
    };

    const setPanelHidden = (hidden, { focus = true } = {}) => {
      panel.classList.toggle('is-hidden', hidden);
      panel.setAttribute('aria-hidden', hidden ? 'true' : 'false');
      focusables.forEach((el) => setFocusable(el, hidden));
      if (showBtn) {
        showBtn.setAttribute('aria-hidden', hidden ? 'false' : 'true');
        showBtn.setAttribute('aria-expanded', hidden ? 'false' : 'true');
        showBtn.setAttribute('tabindex', hidden ? '0' : '-1');
      }
      if (hideBtn) {
        hideBtn.setAttribute('aria-expanded', hidden ? 'false' : 'true');
      }
      if (!focus) return;
      if (hidden) {
        showBtn?.focus();
        return;
      }
      links[0]?.focus();
    };

    items.forEach((item) => {
      bindPointerBlur(item.link);
      if (item.link.dataset.jumpManual === 'yes') return;
      item.link.dataset.jumpManual = 'yes';
      item.link.addEventListener('click', (event) => setManualActive(item, event));
    });

    const clearPanelFocus = (event) => {
      if (event?.pointerType && event.pointerType !== 'mouse') return;
      const active = document.activeElement;
      if (active && panel.contains(active)) active.blur();
    };
    panel.addEventListener('pointerleave', clearPanelFocus);
    panel.addEventListener('mouseleave', clearPanelFocus);
    panel.addEventListener('pointerdown', () => setPanelCondensed(false));
    panel.addEventListener('focusin', () => setPanelCondensed(false));

    if (hideBtn && hideBtn.dataset.jumpHideBound !== 'yes') {
      hideBtn.dataset.jumpHideBound = 'yes';
      hideBtn.addEventListener('click', () => setPanelHidden(true));
    }
    if (showBtn && showBtn.dataset.jumpShowBound !== 'yes') {
      showBtn.dataset.jumpShowBound = 'yes';
      showBtn.addEventListener('click', () => {
        setPanelHidden(false);
        setPanelCondensed(false);
      });
    }
    if (hideBtn || showBtn) {
      setPanelHidden(false, { focus: false });
    }

    const update = () => {
      ticking = false;
      const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
      if (!viewportHeight) {
        setActive(null);
        return;
      }
      const navOffset = getNavOffset();
      const topLimit = Math.min(Math.max(0, navOffset), viewportHeight);
      const bottomLimit = viewportHeight;
      const focusSpan = Math.max(0, bottomLimit - topLimit);
      const doc = document.documentElement;
      const scrollTop = window.scrollY || window.pageYOffset || 0;
      const atBottom = scrollTop + viewportHeight >= (doc.scrollHeight - 2);
      const autoScrolling = isJumpPanelAutoScrolling();
      const manualLocked = Boolean(manualOverrideId && autoScrolling);
      if (manualOverrideId && !manualLocked) manualOverrideId = null;
      let nextId = null;

      if (isStoryRail && storyEndY > storyStartY) {
        if (!storyInitialized) {
          initializeStoryMilestone(scrollTop, viewportHeight, atBottom);
        } else if (!manualLocked) {
          const scrollDirection = scrollTop > storyLastScrollY + 1
            ? 1
            : (scrollTop < storyLastScrollY - 1 ? -1 : 0);
          const atStoryStart = scrollTop <= 2 && storyIndexFromHash() < 0;
          const canResolveStory = storyInputIntent || storyIndexFromHash() >= 0;
          const nextMilestone = atStoryStart || !canResolveStory
            ? -1
            : resolveStoryMilestone(scrollDirection, viewportHeight, atBottom);
          if (nextMilestone !== storyTargetIndex) retargetStoryMilestone(nextMilestone);
        }
        storyLastScrollY = scrollTop;
        nextId = manualLocked
          ? manualOverrideId
          : (storyTargetIndex >= 0 ? items[storyTargetIndex]?.id || null : null);
      } else {
        let best = null;
        let bestRatio = 0;
        let bestDistance = Infinity;

        items.forEach((item) => {
          const rect = item.target.getBoundingClientRect();
          const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
          if (visible <= 0) return;
          const denom = Math.min(rect.height || 0, focusSpan) || 1;
          const ratio = visible / denom;
          const distance = Math.abs(rect.top - topLimit);
          if (ratio > bestRatio || (Math.abs(ratio - bestRatio) < 0.001 && distance < bestDistance)) {
            bestRatio = ratio;
            bestDistance = distance;
            best = item;
          }
        });

        nextId = best ? best.id : null;
        const lastItem = items[items.length - 1];
        if (lastItem && atBottom) {
          const rect = lastItem.target.getBoundingClientRect();
          const visible = Math.max(0, Math.min(rect.bottom, bottomLimit) - Math.max(rect.top, topLimit));
          if (visible > 0) nextId = lastItem.id;
        }
      }

      if (nextId === activeId) return;
      activeId = nextId;
      setActive(nextId);
    };

    const requestUpdate = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    };

    const syncStoryLocation = () => {
      if (!isStoryRail || !storyInitialized) return;
      const hashIndex = storyIndexFromHash();
      if (hashIndex < 0) return;
      manualOverrideId = items[hashIndex].id;
      retargetStoryMilestone(hashIndex, {
        immediate: prefersReducedStoryMotion(),
        confirm: false,
        force: true
      });
      activeId = items[hashIndex].id;
      setActive(activeId);
    };

    const handleStoryMotionChange = () => {
      if (!storyInitialized || !prefersReducedStoryMotion()) return;
      retargetStoryMilestone(storyTargetIndex, {
        immediate: true,
        confirm: false,
        force: true
      });
    };

    measureStoryRail();
    requestUpdate();
    const handleScroll = () => {
      const autoScrolling = isJumpPanelAutoScrolling();
      if (!autoScrolling) {
        clearPanelFocusOnScroll();
        setPanelCondensed(true);
        if (manualOverrideId) manualOverrideId = null;
      }
      requestUpdate();
    };
    const handleResize = () => {
      if (!shouldCondenseOnScroll()) {
        panel.classList.remove('is-condensed');
      }
      railControls?.updateControls();
      requestStoryMeasure();
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    window.addEventListener('wheel', markStoryInputIntent, { passive: true });
    window.addEventListener('touchmove', markStoryInputIntent, { passive: true });
    window.addEventListener('pointerdown', markStoryInputIntent, { passive: true });
    window.addEventListener('keydown', markStoryInputIntent);
    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleResize);
    window.addEventListener('hashchange', syncStoryLocation);
    window.addEventListener('popstate', syncStoryLocation);
    window.addEventListener('pageshow', () => {
      if (storyIndexFromHash() < 0) resetStoryToStart();
      requestStoryMeasure();
      requestUpdate();
    });
    document.addEventListener('navheightchange', requestUpdate);
    if (storyMotionQuery) {
      if (typeof storyMotionQuery.addEventListener === 'function') {
        storyMotionQuery.addEventListener('change', handleStoryMotionChange);
      } else if (typeof storyMotionQuery.addListener === 'function') {
        storyMotionQuery.addListener(handleStoryMotionChange);
      }
    }
    if (isStoryRail) {
      if (document.fonts?.ready && typeof document.fonts.ready.then === 'function') {
        document.fonts.ready.then(requestStoryMeasure).catch(() => {});
      }
      $$('img', document.getElementById('main') || document).forEach((image) => {
        if (!image.complete) image.addEventListener('load', requestStoryMeasure, { once: true });
      });
      if ('ResizeObserver' in window) {
        const storyObserver = new ResizeObserver(requestStoryMeasure);
        items.forEach((item) => storyObserver.observe(item.target));
      }
    }
  }

  function initSpeedDial(){
    if (!document || !document.body || typeof document.createElement !== 'function') return;
    let dial = document.querySelector('[data-speed-dial]');
    if (!dial) {
      dial = document.createElement('div');
      if (!dial || typeof dial.setAttribute !== 'function') return;
      const menuId = 'speed-dial-menu';
      dial.className = 'speed-dial';
      dial.setAttribute('data-speed-dial', 'true');
      dial.innerHTML = `
        <div class="speed-dial__tray" data-speed-dial-tray>
          <div class="speed-dial__actions" id="${menuId}" role="menu" aria-label="Contact options" aria-hidden="true" data-speed-dial-menu>
            <div class="speed-dial__item">
              <span class="speed-dial__label" aria-hidden="true">Direct Message</span>
              <a class="speed-dial__action btn-icon speed-dial__action--direct" href="#contact-modal" data-contact-modal-link="true" aria-label="Send a direct message" role="menuitem" data-speed-dial-action>
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>
                  <path d="M7 9h10"></path>
                  <path d="M7 13h6"></path>
                </svg>
              </a>
            </div>
            <div class="speed-dial__item">
              <span class="speed-dial__label" aria-hidden="true">Send Email</span>
              <a class="speed-dial__action btn-icon" href="mailto:daniel@danielshort.me" aria-label="Send Email" role="menuitem" data-speed-dial-action>
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="3" y="5" width="18" height="14" rx="2"></rect>
                  <path d="M3 7l9 6 9-6"></path>
                </svg>
              </a>
            </div>
            <div class="speed-dial__item">
              <span class="speed-dial__label" aria-hidden="true">View LinkedIn</span>
              <a class="speed-dial__action btn-icon" href="https://www.linkedin.com/in/danielshort3/" target="_blank" rel="noopener noreferrer" aria-label="View LinkedIn" role="menuitem" data-speed-dial-action>
                <svg class="brand-fill" viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="4" cy="4" r="2"></circle>
                  <rect x="2" y="9" width="4" height="12" rx="1"></rect>
                  <path d="M10 9h3.8v2.1h.1C14.8 9.7 16.1 9 17.9 9c3 0 5.1 1.9 5.1 5.9V21h-4v-5.9c0-1.7-.7-2.9-2.6-2.9s-2.7 1.4-2.7 3V21H10z"></path>
                </svg>
              </a>
            </div>
            <div class="speed-dial__item">
              <span class="speed-dial__label" aria-hidden="true">View GitHub</span>
              <a class="speed-dial__action btn-icon" href="https://github.com/danielshort3" target="_blank" rel="noopener noreferrer" aria-label="View GitHub" role="menuitem" data-speed-dial-action>
                <span class="icon icon-github" aria-hidden="true"></span>
              </a>
            </div>
          </div>
        </div>
        <button class="speed-dial__toggle btn-icon btn-icon-featured" type="button" aria-expanded="false" aria-haspopup="menu" aria-controls="${menuId}" aria-label="Open contact options" data-speed-dial-toggle>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 4h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-5.17L9 22.5V17H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"></path>
            <path d="M12 8v6"></path>
            <path d="M9 11h6"></path>
          </svg>
        </button>
      `;
      document.body.appendChild(dial);
    }

    if (dial.dataset.speedDialBound === 'yes') return;
    dial.dataset.speedDialBound = 'yes';

    const toggle = dial.querySelector('[data-speed-dial-toggle]');
    const menu = dial.querySelector('[data-speed-dial-menu]');
    const actions = [...dial.querySelectorAll('[data-speed-dial-action]')];
    if (!toggle || !menu || !actions.length) return;

    let isLocked = false;
    let suppressHover = false;

    const setExpanded = (expanded) => {
      dial.classList.toggle('is-open', expanded);
      toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      toggle.setAttribute('aria-label', expanded ? 'Close contact options' : 'Open contact options');
      menu.setAttribute('aria-hidden', expanded ? 'false' : 'true');
      actions.forEach(action => {
        action.tabIndex = expanded ? 0 : -1;
      });
      if (!expanded && menu.contains(document.activeElement)) {
        toggle.focus({ preventScroll: true });
      }
    };

    const closeMenu = () => {
      isLocked = false;
      suppressHover = false;
      setExpanded(false);
    };

    setExpanded(false);

    toggle.addEventListener('click', (event) => {
      event.preventDefault();
      if (isLocked) {
        isLocked = false;
        try {
          if (dial.matches(':hover')) suppressHover = true;
        } catch {}
        setExpanded(false);
        return;
      }
      isLocked = true;
      suppressHover = false;
      setExpanded(true);
    });

    actions.forEach(action => {
      action.addEventListener('click', closeMenu);
    });

    let canHover = false;
    try {
      canHover = Boolean(window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches);
    } catch {}
    if (canHover) {
      dial.addEventListener('pointerenter', () => {
        if (isLocked || suppressHover) return;
        setExpanded(true);
      });
      dial.addEventListener('pointerleave', () => {
        if (isLocked) return;
        suppressHover = false;
        setExpanded(false);
      });
    }

    document.addEventListener('click', (event) => {
      if (!dial.classList.contains('is-open')) return;
      if (dial.contains(event.target)) return;
      closeMenu();
    });

    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (!dial.classList.contains('is-open')) return;
      event.preventDefault();
      closeMenu();
      toggle.focus({ preventScroll: true });
    });
  }

  function initCookieSettingsButton(){
    if (!document || !document.body || typeof document.createElement !== 'function') return;
    try {
      if (window.self !== window.top) return;
    } catch {
      return;
    }
    $$('[data-cookie-settings]').forEach((host) => {
      host.remove();
    });
  }

  initCookieSettingsButton();
  initSpeedDial();

  // ---- Global modal close handlers (X button and backdrop) ----
  document.addEventListener('click', (e) => {
    // 1) Close when X is clicked
    const closeBtn = e.target.closest('.modal-close');
    if (closeBtn) {
      const modal = closeBtn.closest('.modal');
      if (modal && modal.id === CONTACT_MODAL_ID) return;
      e.preventDefault();
      if (modal) {
        const id = modal.id?.replace(/-modal$/, '') || modal.id || 'modal';
        window.closeModal && window.closeModal(id);
      }
      return;
    }
    // 2) Close when clicking the backdrop (outside modal-content)
    const backdrop = e.target.closest('.modal');
    const insideContent = e.target.closest('.modal-content');
    if (backdrop && !insideContent) {
      if (backdrop.id === CONTACT_MODAL_ID) return;
      const id = backdrop.id?.replace(/-modal$/, '') || backdrop.id || 'modal';
      window.closeModal && window.closeModal(id);
    }
  });

  const PROJECT_EMBED_MIN_HEIGHT_PX = 360;
  const PROJECT_EMBED_MAX_HEIGHT_PX = 5000;
  const PROJECT_EMBED_RESIZE_TYPE_RE = /(?:^portfolio-demo:resize$|-demo-resize$)/;

  const projectEmbedForFrame = (ifr) => (
    ifr ? ifr.closest('.project-embed') : null
  );

  const projectEmbedFit = (ifr) => {
    const embed = projectEmbedForFrame(ifr);
    const fit = String(embed?.dataset?.embedFit || '').trim().toLowerCase();
    if (['content', 'viewport', 'dashboard', 'fixed'].includes(fit)) return fit;
    if (embed?.classList?.contains('project-embed-tableau')) return 'dashboard';
    if (embed?.classList?.contains('project-embed-chatbotlora')) return 'viewport';
    return 'content';
  };

  const readPositiveNumber = (value, fallback) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
  };

  const projectEmbedHeightLimits = (ifr) => {
    const embed = projectEmbedForFrame(ifr);
    const min = readPositiveNumber(embed?.dataset?.embedMinHeight, PROJECT_EMBED_MIN_HEIGHT_PX);
    const max = readPositiveNumber(embed?.dataset?.embedMaxHeight, PROJECT_EMBED_MAX_HEIGHT_PX);
    return {
      min: Math.max(1, Math.floor(min)),
      max: Math.max(Math.floor(min), Math.floor(max))
    };
  };

  const shouldAutoResizeProjectEmbed = (ifr) => (
    !!ifr && projectEmbedFit(ifr) === 'content'
  );

  const setProjectEmbedIframeHeight = (ifr, height) => {
    if (!shouldAutoResizeProjectEmbed(ifr)) return;
    if (!Number.isFinite(height) || height <= 0) return;
    const { min, max } = projectEmbedHeightLimits(ifr);
    const measured = Math.ceil(height);
    const constrained = Math.min(max, Math.max(min, measured));
    const next = `${constrained}px`;
    const embed = projectEmbedForFrame(ifr);
    if (ifr.style.height === next && embed?.style?.getPropertyValue('--project-demo-height') === next) return;
    ifr.style.height = next;
    if (embed) embed.style.setProperty('--project-demo-height', next);
  };

  const measureProjectEmbedDocumentHeight = (doc) => {
    if (!doc) return 0;
    const body = doc.body;
    const docEl = doc.documentElement;
    if (body) {
      let marginY = 0;
      try {
        const style = doc.defaultView?.getComputedStyle?.(body);
        const marginTop = Number.parseFloat(style?.marginTop || '0');
        const marginBottom = Number.parseFloat(style?.marginBottom || '0');
        marginY = (Number.isFinite(marginTop) ? marginTop : 0) + (Number.isFinite(marginBottom) ? marginBottom : 0);
      } catch {}
      let rectHeight = 0;
      try {
        rectHeight = body.getBoundingClientRect().height + marginY;
      } catch {}
      const bodyHeight = Math.max(
        body.scrollHeight || 0,
        body.offsetHeight || 0,
        rectHeight || 0
      );
      if (bodyHeight > 0) return bodyHeight;
    }
    if (!docEl) return 0;
    return Math.max(
      docEl.scrollHeight || 0,
      docEl.offsetHeight || 0
    );
  };

  const resizeProjectEmbedIframe = (ifr) => {
    if (!shouldAutoResizeProjectEmbed(ifr)) return;
    try {
      const doc = ifr.contentDocument || ifr.contentWindow?.document;
      if (!doc) return;
      const height = measureProjectEmbedDocumentHeight(doc);
      setProjectEmbedIframeHeight(ifr, height);
    } catch {}
  };

  const observeProjectEmbedIframe = (ifr) => {
    if (!shouldAutoResizeProjectEmbed(ifr)) return;
    if (typeof ResizeObserver !== 'function') return;

    try {
      if (ifr._projectEmbedResizeObserver) {
        ifr._projectEmbedResizeObserver.disconnect();
      }
    } catch {}
    ifr._projectEmbedResizeObserver = null;

    let doc = null;
    try {
      doc = ifr.contentDocument || ifr.contentWindow?.document;
    } catch {}
    if (!doc) return;

    const body = doc.body;
    const docEl = doc.documentElement;
    if (!body && !docEl) return;

    const scheduleResize = () => {
      if (ifr._projectEmbedResizeScheduled) return;
      ifr._projectEmbedResizeScheduled = true;
      requestAnimationFrame(() => {
        ifr._projectEmbedResizeScheduled = false;
        resizeProjectEmbedIframe(ifr);
      });
    };

    const ro = new ResizeObserver(scheduleResize);
    try { if (docEl) ro.observe(docEl); } catch {}
    try { if (body) ro.observe(body); } catch {}
    ifr._projectEmbedResizeObserver = ro;
    scheduleResize();
  };

  const bindProjectEmbedResize = () => {
    document.querySelectorAll('.project-embed-frame').forEach((ifr) => {
      if (ifr._resizeBound) return;
      ifr._resizeBound = true;
      if (!shouldAutoResizeProjectEmbed(ifr)) {
        ifr.setAttribute('scrolling', 'auto');
        ifr.style.removeProperty('overflow');
        return;
      }
      ifr.setAttribute('scrolling', 'no');
      ifr.style.overflow = 'hidden';
      ifr.addEventListener('load', () => {
        resizeProjectEmbedIframe(ifr);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 50);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 350);
        setTimeout(() => resizeProjectEmbedIframe(ifr), 1000);
        observeProjectEmbedIframe(ifr);
      });
      resizeProjectEmbedIframe(ifr);
      observeProjectEmbedIframe(ifr);
    });
  };

  document.addEventListener('DOMContentLoaded', bindProjectEmbedResize);
  window.addEventListener('load', bindProjectEmbedResize);
  window.addEventListener('message', (event) => {
    if (event.origin && event.origin !== window.location.origin) return;
    const data = event && event.data || {};
    const type = typeof data?.type === 'string' ? data.type : '';
    if (!PROJECT_EMBED_RESIZE_TYPE_RE.test(type)) return;
    const ifrs = document.querySelectorAll('.project-embed-frame');
    for (const ifr of ifrs) {
      if (ifr.contentWindow === event.source) {
        if (!shouldAutoResizeProjectEmbed(ifr)) break;
        const h = typeof data.height === 'number' && isFinite(data.height)
          ? Math.max(0, Number(data.height))
          : null;
        if (h) {
          setProjectEmbedIframeHeight(ifr, h);
        } else {
          resizeProjectEmbedIframe(ifr);
        }
        break;
      }
    }
  });
})();
