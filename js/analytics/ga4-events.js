/* ===================================================================
   File: ga4-events.js
   Purpose: Utility helpers to send Google Analytics 4 events
   =================================================================== */
(() => {
  'use strict';

  // Always prepare a dataLayer and a global gtag shim; load GA only on consent
  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function(){ (window.dataLayer = window.dataLayer || []).push(arguments); };

  // Set conservative Consent Mode defaults as early as possible
  try {
    window.gtag('consent', 'default', {
      'ad_storage': 'denied',
      'analytics_storage': 'denied',
      'ad_user_data': 'denied',
      'ad_personalization': 'denied',
      // Short wait window so GA respects updates promptly
      'wait_for_update': 500
    });
  } catch {}

  // Load GA library only when allowed
  function loadGA4(id) {
    // If GA already present, just ensure config is set
    if (typeof window.gtag === 'function' && document.getElementById('ga4-src')) {
      window.gtag('config', id);
      return;
    }
    if (document.getElementById('ga4-src')) {
      // Script tag exists but gtag may not be ready yet; schedule a config
      const onReady = () => {
        try {
          window.gtag('js', new Date());
          window.gtag('config', id);
        } catch {}
      };
      window.addEventListener('load', onReady, { once: true });
      setTimeout(onReady, 0);
      return;
    }
    const s = document.createElement('script');
    s.id = 'ga4-src';
    s.async = true;
    s.src = `https://www.googletagmanager.com/gtag/js?id=${id}`;
    s.onload = () => {
      try {
        window.gtag('js', new Date());
        window.gtag('config', id);
      } catch {}
    };
    document.head.appendChild(s);
  }

  // Init when analytics consent is granted (now or later)
  function tryInitFromConsent(detail) {
    const state = detail && detail.categories ? detail.categories : detail;
    const ok = !!(state && state.analytics);
    if (ok) loadGA4('G-0VL37MQ62P');
  }

  // If consent manager already ran:
  if (window.consentAPI && typeof window.consentAPI.get === 'function') {
    tryInitFromConsent(window.consentAPI.get());
  }
  // Listen for changes from our CMP
  window.addEventListener('consent-changed', (e) => tryInitFromConsent(e.detail));

  // --- below here: keep your existing helpers unchanged ---
  // Helper to dispatch GA events
  const send = (name, params={}) => {
    if (typeof window.gtag === 'function') {
      window.gtag('event', name, params);
    }
  };

  // expose send so other scripts can trigger custom events
  window.gaEvent = send;

  // Count views to fire a special event after three different projects
  let projectViews = 0;
  window.trackProjectView = id => {
    projectViews++;
    send('project_view', { project_id: id });
    if (projectViews === 3) {
      send('multi_project_view', { view_count: 3 });
    }
  };

  // Track when a modal dialog is dismissed
  window.trackModalClose = id => {
    send('modal_close', { project_id: id });
  };

  // Wire up click tracking once the DOM is ready
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.hero-cta').forEach(btn => {
      btn.addEventListener('click', () => {
        send('hero_cta_click', { cta_label: btn.textContent.trim() });
      });
    });

    document.querySelectorAll('a[href*="Resume.pdf"]').forEach(link => {
      link.addEventListener('click', () => {
        send('resume_download', { file_name: 'Resume.pdf' });
      });
    });

    document.querySelectorAll('a[href^="mailto:"]').forEach(link => {
      link.addEventListener('click', () => send('email_cta_click'));
    });

    document.querySelectorAll('a[target="_blank"]').forEach(link => {
      link.addEventListener('click', () => {
        send('outbound_click', {
          link_url: link.href,
          link_text: link.textContent.trim()
        });
      });
    });

    document.querySelectorAll('.nav-link:not([target="_blank"])').forEach(link => {
      link.addEventListener('click', () => {
        send('nav_link_click', { link_url: link.getAttribute('href') });
      });
    });

    const filterMenu = document.getElementById('filter-menu');
    if (filterMenu) {
      filterMenu.addEventListener('click', e => {
        if (e.target.dataset.filter) {
          send('project_filter_select', { filter_name: e.target.dataset.filter });
        }
      });
    }

    const seeMore = document.getElementById('see-more');
    if (seeMore) {
      seeMore.addEventListener('click', () => {
        const expanded = seeMore.dataset.expanded === 'true';
        send('see_more_toggle', { expanded: !expanded });
      });
    }

    if (document.body.dataset.page === 'contact') {
      send('contact_page_view');
      document.querySelectorAll('.contact-card').forEach(card => {
        card.addEventListener('click', () => {
          const label = card.querySelector('span')?.textContent.trim() || card.href;
          send('contact_card_click', { card_label: label });
        });
      });
    }

    // Fire an event when users stay on the page for 60 seconds
    setTimeout(() => send('engaged_time', { seconds: 60 }), 60000);

    let sent50 = false;
    // Record when the user scrolls halfway down the page
    window.addEventListener('scroll', () => {
      if (sent50) return;
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const pct = (window.scrollY || window.pageYOffset) / scrollable;
      if (pct >= 0.5) {
        sent50 = true;
        send('scroll_depth', { percent: 50 });
      }
    }, { passive: true });
  });
})();
