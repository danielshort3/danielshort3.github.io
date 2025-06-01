(() => {
  'use strict';
  const send = (name, params={}) => {
    if (typeof gtag === 'function') {
      gtag('event', name, params);
    }
  };

  // expose send so other scripts can trigger custom events
  window.gaEvent = send;

  let projectViews = 0;
  window.trackProjectView = id => {
    projectViews++;
    send('project_view', { project_id: id });
    if (projectViews === 3) {
      send('multi_project_view', { view_count: 3 });
    }
  };

  // track when a modal is closed
  window.trackModalClose = id => {
    send('modal_close', { project_id: id });
  };

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

    setTimeout(() => send('engaged_time', { seconds: 60 }), 60000);

    let sent50 = false;
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
