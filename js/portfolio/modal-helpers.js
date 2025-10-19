/* ===================================================================
   File: modal-helpers.js
   Purpose: Shared modal helpers + media utilities for portfolio items
=================================================================== */
(() => {
  'use strict';

  if (window.generateProjectModal && window.openModal && window.closeModal) {
    // Helpers are already initialised (avoid double-binding listeners)
    return;
  }

  let __modalPrevFocus = null;
  let __srStatus = null;

  function srStatus() {
    if (__srStatus) return __srStatus;
    const el = document.createElement('div');
    el.id = 'sr-status';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.setAttribute('aria-atomic', 'true');
    el.style.position = 'absolute';
    el.style.left = '-9999px';
    el.style.width = '1px';
    el.style.height = '1px';
    el.style.overflow = 'hidden';
    document.body.appendChild(el);
    __srStatus = el;
    return __srStatus;
  }

  window.getSrStatusNode = srStatus;

  function trapFocus(modalEl) {
    const focusables = modalEl.querySelectorAll(
      'a,button,input,textarea,select,[tabindex]:not([tabindex="-1"])'
    );
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    modalEl.addEventListener(
      'keydown',
      (modalEl)._trap = (e) => {
        if (e.key !== 'Tab') return;
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    );
  }

  function untrapFocus(modalEl) {
    if (modalEl._trap) modalEl.removeEventListener('keydown', modalEl._trap);
  }

  function activateGifVideo(container) {
    const vid = container && container.querySelector && container.querySelector('video.gif-video');
    if (!vid) return;
    try {
      vid.muted = true;
      vid.autoplay = true;
      vid.playsInline = true;
      vid.setAttribute('muted', '');
      vid.setAttribute('autoplay', '');
      vid.setAttribute('playsinline', '');
    } catch {}

    const showVideo = () => {
      vid.style.display = 'block';
      const next = vid.nextElementSibling;
      if (next && (next.tagName === 'IMG' || next.tagName === 'PICTURE')) next.style.display = 'none';
      try { vid.play && vid.play().catch(() => {}); } catch {}
    };

    if (vid.readyState >= 2) {
      showVideo();
    } else {
      ['loadeddata', 'canplay', 'canplaythrough', 'playing'].forEach(evt => {
        vid.addEventListener(evt, showVideo, { once: true });
      });
    }
  }

  function portfolioBasePath() {
    try {
      const path = location.pathname || '';
      const m = path.match(/^(.*?)(\/(?:pages\/)?portfolio(?:\.html)?)(?:\/[A-Za-z0-9_-]+)?\/?$/);
      if (m) return (m[1] || '') + (m[2] || '');
    } catch {}
    return null;
  }

  function resizeIframeToContent(ifr) {
    if (!ifr) return;
    const doc = ifr.contentDocument || ifr.contentWindow?.document;
    if (!doc) return;
    const b = doc.body;
    const de = doc.documentElement;
    const h = Math.max(
      b ? b.scrollHeight : 0,
      b ? b.offsetHeight : 0,
      de ? de.clientHeight : 0,
      de ? de.scrollHeight : 0,
      de ? de.offsetHeight : 0
    );
    if (h > 0) {
      ifr.style.height = `${h}px`;
    }
  }

  function projectMedia(p) {
  const hasVideo = !!(p.videoWebm || p.videoMp4);
    const size = PROJECT_IMAGE_SIZES[p.id] || { width: 1280, height: 720 };
    const img = (() => {
      const src = p.image || '';
      const lower = src.toLowerCase();
      const webp = lower.endsWith('.png') ? src.replace(/\.png$/i, '.webp')
                 : lower.endsWith('.jpg') ? src.replace(/\.jpg$/i, '.webp')
                 : lower.endsWith('.jpeg') ? src.replace(/\.jpeg$/i, '.webp')
                 : null;
      if (webp) {
        return `<picture>
        <source srcset="${webp}" type="image/webp">
        <img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false" width="${size.width}" height="${size.height}">
      </picture>`;
      }
      return `<img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false" width="${size.width}" height="${size.height}">`;
    })();

    if (!hasVideo) return img;
    const mp4 = p.videoMp4 ? `<source src="${p.videoMp4}" type="video/mp4">` : '';
    const webm = p.videoWebm ? `<source src="${p.videoWebm}" type="video/webm">` : '';
    return `
    <video class="gif-video" muted playsinline loop autoplay preload="metadata" aria-label="${p.title}" draggable="false">
      ${mp4}
      ${webm}
    </video>
    ${img}`;
  }

  const PROJECT_IMAGE_SIZES = {
    smartSentence: { width: 1280, height: 720 },
    chatbotLora: { width: 1280, height: 720 },
    shapeClassifier: { width: 1280, height: 720 },
    ufoDashboard: { width: 2008, height: 1116 },
    covidAnalysis: { width: 792, height: 524 },
    targetEmptyPackage: { width: 896, height: 480 },
    handwritingRating: { width: 600, height: 960 },
    digitGenerator: { width: 400, height: 400 },
    sheetMusicUpscale: { width: 1604, height: 1230 },
    deliveryTip: { width: 960, height: 794 },
    retailStore: { width: 1490, height: 690 },
    pizza: { width: 1726, height: 1054 },
    babynames: { width: 1200, height: 800 },
    pizzaDashboard: { width: 1250, height: 1092 },
    nonogram: { width: 1080, height: 1080 },
    website: { width: 1240, height: 1456 }
  };

  window.activateGifVideo = activateGifVideo;
  window.projectMedia = projectMedia;

  const normalizeList = (value) => {
    if (!value) return [];
    if (Array.isArray(value)) return value.filter(Boolean);
    return [value];
  };

  const getProjectDetails = (project) => {
    const all = window.PROJECT_DETAILS || {};
    const detail = all[project.id] || {};
    const data = normalizeList(detail.data);
    const methods = normalizeList(detail.methods || project.actions);
    const results = normalizeList(detail.results || project.results);
    const impact = normalizeList(detail.impact);
    const keyResults = normalizeList(detail.keyResults || detail.results || project.results);
    const related = Array.isArray(detail.related) ? detail.related.filter(Boolean) : [];
    return {
      overview: detail.overview || project.problem || '',
      goal: detail.goal || detail.objective || '',
      data,
      methods,
      results,
      impact,
      keyResults,
      related
    };
  };

  const renderList = (items, className = '') => {
    if (!items.length) return '';
    const cls = className ? ` class="${className}"` : '';
    return `<ul${cls}>${items.map(item => `<li>${item}</li>`).join('')}</ul>`;
  };

  const svgIcon = {
    github: "<svg viewBox='0 0 24 24' aria-hidden='true'><path fill='currentColor' d='M12 .5C5.4.5 0 6 0 12.7c0 5.4 3.4 10 8.2 11.6.6.1.8-.3.8-.6v-2.4c-3.3.8-4-1.6-4-1.6-.5-1.4-1.3-1.8-1.3-1.8-1.1-.8.1-.7.1-.7 1.2.1 1.8 1.3 1.8 1.3 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.4-1.4-5.4-6.1 0-1.3.5-2.4 1.2-3.3-.1-.3-.5-1.6.1-3.3 0 0 1-.3 3.3 1.2a11.8 11.8 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.7.2 3 .1 3.3.8.9 1.2 2 1.2 3.3 0 4.8-2.8 5.8-5.5 6.1.4.3.8 1 .8 2.1v3.1c0 .3.2.7.8.6C20.6 22.7 24 18.1 24 12.7 24 6 18.6.5 12 .5Z'/></svg>",
    website: "<svg viewBox='0 0 24 24' aria-hidden='true'><path fill='none' stroke='currentColor' stroke-width='1.6' d='M12 21c4.97 0 9-4.03 9-9s-4.03-9-9-9-9 4.03-9 9 4.03 9 9 9Zm0-18c2.5 2.26 4 5.32 4 9s-1.5 6.74-4 9c-2.5-2.26-4-5.32-4-9s1.5-6.74 4-9Zm-9 9h18M3.6 8h16.8M3.6 16h16.8'/></svg>"
  };

  const renderResourceIcon = (resource) => {
    const typeHint = (resource.type || resource.label || '').toLowerCase();
    if (typeHint.includes('github')) {
      return `<span class="resource-icon" aria-hidden="true">${svgIcon.github}</span>`;
    }
    if (typeHint.includes('site') || typeHint.includes('demo') || typeHint.includes('dashboard')) {
      return `<span class="resource-icon" aria-hidden="true">${svgIcon.website}</span>`;
    }
    if (resource.icon) {
      const base = resource.icon.replace(/\.png$/i, '');
      const webp = `${base}.webp`;
      const png = resource.icon;
      return `<picture>
        <source srcset="${webp}" type="image/webp">
        <img src="${png}" alt="" width="30" height="30" loading="lazy" decoding="async">
      </picture>`;
    }
    return '';
  };

  const renderResources = (resources = []) => {
    if (!Array.isArray(resources) || !resources.length) return '';
    return resources.map(r => {
      const iconHtml = renderResourceIcon(r);
      const label = r.label || r.url;
      const rel = /^https?:/i.test(r.url || '') ? ' rel="noopener noreferrer"' : '';
      const target = /^https?:/i.test(r.url || '') ? ' target="_blank"' : '';
      return `<a href="${r.url}"${target}${rel} aria-label="${label}" class="resource-link">${iconHtml}<span class="sr-only">${label}</span></a>`;
    }).join('');
  };

  window.closeModal = function(id) {
    const modal = document.getElementById(`${id}-modal`) || document.getElementById(id);
    if (!modal) return;
    modal.classList.remove('active');
    document.body.classList.remove('modal-open');
    untrapFocus(modal);
    if (__modalPrevFocus) {
      try {
        __modalPrevFocus.focus();
      } catch {}
      __modalPrevFocus = null;
    }
    window.trackModalClose && window.trackModalClose(id);
    try {
      const p = (window.PROJECTS || []).find(x => x.id === id);
      if (p) srStatus().textContent = `Closed: ${p.title}`;
    } catch {}

    try {
      const base = portfolioBasePath();
      if (base && history && history.replaceState) {
        history.replaceState(null, '', base);
      } else if (location.hash) {
        location.hash = '';
      }
    } catch {}
  };

  window.openModal = function(id) {
    const modal = document.getElementById(`${id}-modal`) || document.getElementById(id);
    if (!modal) return;
    __modalPrevFocus = document.activeElement;
    modal.classList.add('active');
    document.body.classList.add('modal-open');
    const content = modal.querySelector('.modal-content') || modal;
    content.focus({ preventScroll: true });
    trapFocus(content);

    const ifr = modal.querySelector('.modal-embed iframe');
    if (ifr) {
      if (ifr.dataset.src && !ifr.src) {
        ifr.src = ifr.dataset.src;
      }
      if (ifr.dataset.base && !ifr.src) {
        const isPhone = window.matchMedia && window.matchMedia('(max-width:768px)').matches;
        const base = ifr.dataset.base;
        const src = `${base}?${[':embed=y', ':showVizHome=no', `:device=${isPhone ? 'phone' : 'desktop'}`].join('&')}`;
        ifr.src = src;
      }
      if (!ifr._resizeBound) {
        ifr._resizeBound = true;
        ifr.addEventListener('load', () => {
          try { resizeIframeToContent(ifr); } catch {}
          setTimeout(() => { try { resizeIframeToContent(ifr); } catch {}; }, 50);
          setTimeout(() => { try { resizeIframeToContent(ifr); } catch {}; }, 350);
          try { ifr.contentWindow?.document?.fonts?.ready?.then(() => resizeIframeToContent(ifr)); } catch {}
        });
      }
    }

    try {
      const p = (window.PROJECTS || []).find(x => x.id === id);
      if (p) srStatus().textContent = `Opened: ${p.title}`;
    } catch {}

    try {
      const base = portfolioBasePath();
      if (base && history && history.replaceState) {
        const url = `${base}?project=${encodeURIComponent(id)}`;
        history.replaceState(null, '', url);
      }
    } catch {}

    modal.querySelectorAll('[data-related-id]').forEach(btn => {
      if (btn.dataset.relatedBound === 'yes') return;
      btn.dataset.relatedBound = 'yes';
      btn.addEventListener('click', () => {
        const next = btn.getAttribute('data-related-id');
        window.closeModal(id);
        if (next) window.openModal(next);
      });
    });

    try {
      if (typeof window.trackProjectView === 'function') {
        window.trackProjectView(id);
      }
    } catch {}

    const copyBtn = modal.querySelector('.modal-copy');
    if (copyBtn && !copyBtn._bound) {
      copyBtn._bound = true;
      copyBtn.addEventListener('click', async () => {
        let url;
        try {
          const origin = location.origin || '';
          const base = portfolioBasePath() || '/portfolio.html';
          const href = `${origin}${base}?project=${encodeURIComponent(id)}`;
          url = new URL(href);
        } catch {
          const href = (location.href || '').split('#')[0] + `?project=${encodeURIComponent(id)}`;
          try { url = new URL(href); } catch { url = { toString: () => href }; }
        }
        let ok = false;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          try { await navigator.clipboard.writeText(url.toString()); ok = true; } catch {}
        }
        if (!ok) {
          const ta = document.createElement('textarea');
          ta.value = url.toString();
          ta.style.position = 'fixed';
          ta.style.left = '-9999px';
          document.body.appendChild(ta);
          ta.focus();
          ta.select();
          try { document.execCommand('copy'); ok = true; } catch {}
          document.body.removeChild(ta);
        }
        const toast = modal.querySelector('.modal-toast') || (() => {
          const t = document.createElement('div');
          t.className = 'modal-toast';
          t.setAttribute('role', 'status');
          t.setAttribute('aria-live', 'polite');
          modal.querySelector('.modal-content').appendChild(t);
          return t;
        })();
        toast.textContent = ok ? 'Link copied' : 'Copy failed';
        toast.classList.add('show');
        srStatus().textContent = ok ? 'Link copied to clipboard' : 'Copy to clipboard failed';
        setTimeout(() => toast.classList.remove('show'), 1400);
      });
    }
  };

  window.generateProjectModal = function(p) {
    const isTableau = p.embed?.type === 'tableau';
    const isIframe = p.embed?.type === 'iframe';
    const details = getProjectDetails(p);

    const visual = (() => {
      if (isIframe) {
        return `
        <div class="modal-embed">
          <iframe data-src="${p.embed.url}" loading="lazy"></iframe>
        </div>`;
      }
      if (!isTableau) {
        return `
        <div class="modal-image">
          ${projectMedia(p)}
        </div>`;
      }
      const base = p.embed.base || p.embed.url;
      return `
      <div class="modal-embed tableau-fit">
        <iframe
          loading="lazy"
          allowfullscreen
          data-base="${base}"></iframe>
      </div>`;
    })();

    const overviewBlock = details.overview ? `<div class="modal-section"><h4>Overview</h4><p>${details.overview}</p></div>` : '';
    const goalBlock = details.goal ? `<div class="modal-section"><h4>Goal</h4><p>${details.goal}</p></div>` : '';
    const dataBlock = details.data.length ? `<div class="modal-section"><h4>Data</h4>${renderList(details.data)}</div>` : '';
    const methodsBlock = details.methods.length ? `<div class="modal-section"><h4>Methods</h4>${renderList(details.methods)}</div>` : '';
    const resultsBlock = details.results.length ? `<div class="modal-section"><h4>Results</h4>${renderList(details.results)}</div>` : '';
    const impactBlock = details.impact.length ? `<div class="modal-section"><h4>Impact</h4>${renderList(details.impact)}</div>` : '';
    const keyResultsBlock = details.keyResults.length ? `<div class="modal-section"><h4>Key Results</h4>${renderList(details.keyResults, 'key-results-list')}</div>` : '';
    const relatedBlock = details.related.length ? `<div class="modal-section related-projects"><h4>Related Work</h4><div class="related-links">${details.related.map(rel => `<button type="button" class="related-link" data-related-id="${rel.id}">${rel.label}</button>`).join('')}</div></div>` : '';

    const resourcesMarkup = renderResources(Array.isArray(p.resources) ? p.resources : []);

    return `
    <div class="modal-content ${(isTableau || isIframe) ? 'modal-wide' : ''}" role="dialog" aria-modal="true" tabindex="0" aria-labelledby="${p.id}-title">
      <button class="modal-copy" type="button" aria-label="Copy link to this project">Copy link</button>
      <button class="modal-close" aria-label="Close dialog">&times;</button>
      <div class="modal-title-strip"><h3 class="modal-title" id="${p.id}-title">${p.title}</h3></div>

      <div class="modal-header-details">
        <div class="modal-half">
          <p class="header-label">Tools</p>
          <div class="tool-badges">
            ${p.tools.map(t => `<span class="badge">${t}</span>`).join('')}
          </div>
        </div>
        <div class="modal-divider" aria-hidden="true"></div>
        <div class="modal-half">
          <p class="header-label">Downloads / Links</p>
          <div class="icon-row">
            ${resourcesMarkup || '<span class="resource-empty">Assets available on request</span>'}
          </div>
        </div>
      </div>

      <div class="modal-body ${isTableau ? 'stacked' : ''}">
        <div class="modal-text">
          <p class="modal-subtitle">${p.subtitle || ''}</p>
          ${overviewBlock}
          ${goalBlock}
          ${dataBlock}
          ${methodsBlock}
          ${resultsBlock}
          ${impactBlock}
          ${keyResultsBlock}
          ${relatedBlock}
        </div>

        ${visual}
      </div>
    </div>`;
  };

  window.addEventListener('message', (e) => {
    const data = e && e.data || {};
    const type = typeof data?.type === 'string' ? data.type : '';
    if (!/(chatbot|shape|sentence)-demo-resize/.test(type)) return;
    try {
      const ifrs = document.querySelectorAll('.modal-embed iframe');
      for (const f of ifrs) {
        if (f.contentWindow === e.source) {
          const h = typeof data.height === 'number' && isFinite(data.height) ? Math.max(0, Math.floor(data.height)) : null;
          if (h) f.style.height = `${h}px`;
          else resizeIframeToContent(f);
          break;
        }
      }
    } catch {}
  });

  document.addEventListener('keydown', (e) => {
    const open = document.querySelector('.modal.active');
    if (e.key === 'Escape') {
      if (open) {
        const id = open.id?.replace('-modal', '') || 'modal';
        window.closeModal(id);
      }
      return;
    }
    if (!open) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const id = open.id?.replace(/-modal$/, '');
      if (!id || !Array.isArray(window.PROJECTS)) return;
      const idx = window.PROJECTS.findIndex(p => p.id === id);
      if (idx < 0) return;
      const nextIdx = e.key === 'ArrowRight'
        ? (idx + 1) % window.PROJECTS.length
        : (idx - 1 + window.PROJECTS.length) % window.PROJECTS.length;
      window.closeModal(id);
      window.openModal(window.PROJECTS[nextIdx].id);
      e.preventDefault();
    }
  });
})();
