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

  let __mediaViewer = null;
  let __mediaPrevFocus = null;

  function ensureMediaViewer() {
    if (__mediaViewer) return __mediaViewer;
    const wrap = document.createElement('div');
    wrap.className = 'media-viewer';
    wrap.innerHTML = `
      <div class="media-viewer-backdrop" aria-hidden="true"></div>
      <div class="media-viewer-content" role="dialog" aria-modal="true" aria-label="Expanded media">
        <button class="media-viewer-close" type="button" aria-label="Close expanded media">&times;</button>
        <div class="media-viewer-frame"></div>
      </div>
    `;
    document.body.appendChild(wrap);

    const closeBtn = wrap.querySelector('.media-viewer-close');
    closeBtn.addEventListener('click', () => closeMediaViewer());
    wrap.addEventListener('click', (e) => {
      if (e.target === wrap || e.target.classList.contains('media-viewer-backdrop')) {
        closeMediaViewer();
      }
    });
    return (__mediaViewer = wrap);
  }

  function datasetNumber(el, key) {
    if (!el || !el.dataset) return null;
    const val = parseInt(el.dataset[key] || '', 10);
    return Number.isFinite(val) && val > 0 ? val : null;
  }

  function getMediaDimensions(sourceEl) {
    if (!sourceEl) return null;
    const w = datasetNumber(sourceEl, 'videoWidth') || datasetNumber(sourceEl, 'imageWidth');
    const h = datasetNumber(sourceEl, 'videoHeight') || datasetNumber(sourceEl, 'imageHeight');
    return (w && h) ? { width: w, height: h } : null;
  }

  function mediaNaturalSize(el) {
    if (!el) return null;
    if (el.videoWidth && el.videoHeight) return { width: el.videoWidth, height: el.videoHeight };
    if (el.naturalWidth && el.naturalHeight) return { width: el.naturalWidth, height: el.naturalHeight };
    return null;
  }

  function mediaAspectFromProject(p = {}) {
    const w = Number(p.videoWidth || p.imageWidth);
    const h = Number(p.videoHeight || p.imageHeight);
    if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
      return `${w} / ${h}`;
    }
    return '';
  }

  function computeTableauSrc(base) {
    if (!base) return '';
    const isPhone = window.matchMedia && window.matchMedia('(max-width:768px)').matches;
    return `${base}?${[':embed=y', ':showVizHome=no', `:device=${isPhone ? 'phone' : 'desktop'}`].join('&')}`;
  }

  function sizeMediaViewer(viewer, frame, sourceEl, mediaEl) {
    if (!viewer || !frame) return;
    const maxW = Math.min(window.innerWidth - 48, 1200);
    const maxH = Math.min(window.innerHeight - 48, 900);
    const dims = mediaNaturalSize(mediaEl) || getMediaDimensions(sourceEl);
    const aspect = (() => {
      if (dims && dims.width && dims.height) {
        const a = dims.width / Math.max(1, dims.height);
        if (Number.isFinite(a) && a > 0) return a;
      }
      return 16 / 9;
    })();
    let targetW = maxW;
    let targetH = targetW / aspect;
    if (targetH > maxH) {
      targetH = maxH;
      targetW = targetH * aspect;
    }
    frame.style.width = `${Math.round(targetW)}px`;
    frame.style.height = `${Math.round(targetH)}px`;
    frame.style.maxWidth = `${maxW}px`;
    frame.style.maxHeight = `${maxH}px`;
    const content = viewer.querySelector('.media-viewer-content');
    if (content) {
      content.style.maxWidth = `${maxW}px`;
      content.style.maxHeight = `${maxH}px`;
    }
  }

  function closeMediaViewer() {
    if (!__mediaViewer || !__mediaViewer.classList.contains('active')) return false;
    try {
      const vid = __mediaViewer.querySelector('video');
      if (vid) vid.pause();
    } catch {}
    if (__mediaViewer._resize) {
      window.removeEventListener('resize', __mediaViewer._resize);
      __mediaViewer._resize = null;
    }
    const content = __mediaViewer.querySelector('.media-viewer-content');
    if (content) untrapFocus(content);
    __mediaViewer.classList.remove('active');
    document.body.classList.remove('media-viewer-open');
    if (__mediaPrevFocus) {
      try { __mediaPrevFocus.focus(); } catch {}
      __mediaPrevFocus = null;
    }
    return true;
  }

  function openMediaViewer(sourceEl, title) {
    if (!sourceEl) return;
    const viewer = ensureMediaViewer();
    const frame = viewer.querySelector('.media-viewer-frame');
    const image = sourceEl.dataset.image || '';
    const videoMp4 = sourceEl.dataset.videoMp4 || '';
    const videoWebm = sourceEl.dataset.videoWebm || '';
    const hasVideo = !!(videoMp4 || videoWebm);
    const label = title || sourceEl.dataset.title || 'Media preview';
    const embedBase = sourceEl.dataset.embedBase || '';
    const embedSrc = sourceEl.dataset.embedSrc || '';
    const hasEmbed = !!(embedBase || embedSrc);

    let html = '';
    if (hasEmbed) {
      const src = embedSrc || computeTableauSrc(embedBase);
      if (!src) return;
      html = `<iframe class="media-viewer-embed" src="${src}" loading="lazy" allowfullscreen aria-label="${label}"></iframe>`;
    } else if (hasVideo) {
      html = `
        <video class="media-viewer-video" controls autoplay playsinline muted loop aria-label="${label}">
          ${videoMp4 ? `<source src="${videoMp4}" type="video/mp4">` : ''}
          ${videoWebm ? `<source src="${videoWebm}" type="video/webm">` : ''}
          ${image ? `<img src="${image}" alt="${label}">` : ''}
        </video>`;
    } else if (image) {
      html = `<img src="${image}" alt="${label}" loading="lazy">`;
    } else {
      return;
    }

    frame.innerHTML = html;
    const mediaEl = frame.querySelector('img, video, iframe');
    const resizeNow = () => sizeMediaViewer(viewer, frame, sourceEl, mediaEl);
    resizeNow();
    if (__mediaViewer._resize) {
      window.removeEventListener('resize', __mediaViewer._resize);
    }
    __mediaViewer._resize = resizeNow;
    window.addEventListener('resize', resizeNow);
    if (mediaEl) {
      const onReady = () => resizeNow();
      if (mediaEl.tagName === 'VIDEO') {
        mediaEl.addEventListener('loadedmetadata', onReady, { once: true });
      } else if (mediaEl.complete) {
        onReady();
      } else {
        mediaEl.addEventListener('load', onReady, { once: true });
      }
    }

    const content = viewer.querySelector('.media-viewer-content');
    content.setAttribute('aria-label', `Expanded view: ${label}`);
    trapFocus(content);
    __mediaPrevFocus = document.activeElement;
    viewer.classList.add('active');
    document.body.classList.add('media-viewer-open');
    const closeBtn = viewer.querySelector('.media-viewer-close');
    try { closeBtn.focus({ preventScroll: true }); } catch {}
  }

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
    const hasImage = !!p.image;
    const img = (() => {
      if (!hasImage) return '';
      const src = p.image || '';
      const lower = src.toLowerCase();
      const webp = lower.endsWith('.png') ? src.replace(/\.png$/i, '.webp')
                 : lower.endsWith('.jpg') ? src.replace(/\.jpg$/i, '.webp')
                 : lower.endsWith('.jpeg') ? src.replace(/\.jpeg$/i, '.webp')
                 : null;
      if (webp) {
        return `<picture>
        <source srcset="${webp}" type="image/webp">
        <img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false">
      </picture>`;
      }
      return `<img src="${src}" alt="${p.title}" loading="lazy" decoding="async" draggable="false">`;
    })();

    if (!hasVideo) return img;
    const mp4 = p.videoMp4 ? `<source src="${p.videoMp4}" type="video/mp4">` : '';
    const webm = p.videoWebm ? `<source src="${p.videoWebm}" type="video/webm">` : '';
    const video = `
    <video class="gif-video" muted playsinline loop autoplay preload="metadata" aria-label="${p.title}" draggable="false">
      ${mp4}
      ${webm}
    </video>`;
    if (p.videoOnly || !hasImage) {
      return video;
    }
    return `
    ${video}
    ${img}`;
  }

  window.activateGifVideo = activateGifVideo;
  window.projectMedia = projectMedia;

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
    if (!modal || !modal.classList || !modal.classList.contains('modal')) return;
    // Count views when a project modal is opened
    window.trackProjectView && window.trackProjectView(id);
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
        const base = ifr.dataset.base;
        ifr.src = computeTableauSrc(base);
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

    const copyBtn = modal.querySelector('.modal-copy');
    if (copyBtn && !copyBtn._bound) {
      copyBtn._bound = true;
      copyBtn.addEventListener('click', async () => {
        let url;
        try {
          const origin = location.origin || '';
          const href = `${origin}/portfolio/${encodeURIComponent(id)}`;
          url = new URL(href);
        } catch {
          const href = `/portfolio/${encodeURIComponent(id)}`;
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

    const mediaToggle = modal.querySelector('.media-zoom-toggle');
    if (mediaToggle && !mediaToggle._bound) {
      mediaToggle._bound = true;
      mediaToggle.addEventListener('click', () => {
        const mediaContainer = mediaToggle.closest('.modal-image, .modal-embed');
        openMediaViewer(mediaContainer);
      });
    }
  };

  window.generateProjectModal = function(p) {
    const isTableau = p.embed?.type === 'tableau';
    const isIframe = p.embed?.type === 'iframe';
    const mediaAspect = mediaAspectFromProject(p);

    const visual = (() => {
      if (isIframe) {
        return `
        <div class="modal-embed">
          <iframe data-src="${p.embed.url}" loading="lazy"></iframe>
        </div>`;
      }
      if (!isTableau) {
        const aspectStyle = mediaAspect ? ` style="--media-aspect:${mediaAspect};"` : '';
        return `
        <div class="modal-image media-zoomable"
             ${aspectStyle}
             data-image="${p.image || ''}"
             data-image-width="${p.imageWidth || ''}"
             data-image-height="${p.imageHeight || ''}"
             data-video-mp4="${p.videoMp4 || ''}"
             data-video-webm="${p.videoWebm || ''}"
             data-video-width="${p.videoWidth || p.imageWidth || ''}"
             data-video-height="${p.videoHeight || p.imageHeight || ''}"
             data-title="${p.title || ''}">
          <button class="media-zoom-toggle" type="button" aria-label="Open larger media" aria-pressed="false">
            <span class="media-zoom-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
                <circle cx="11" cy="11" r="6"></circle>
                <line x1="15" y1="15" x2="21" y2="21"></line>
              </svg>
            </span>
            <span class="sr-only">Open full view</span>
          </button>
          ${projectMedia(p)}
        </div>`;
      }
      const base = p.embed.base || p.embed.url;
      return `
      <div class="modal-embed tableau-fit"
           data-embed-base="${base || ''}"
           data-embed-src=""
           data-title="${p.title || ''}">
        <button class="media-zoom-toggle" type="button" aria-label="Open larger media" aria-pressed="false">
          <span class="media-zoom-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" focusable="false" aria-hidden="true">
              <circle cx="11" cy="11" r="6"></circle>
              <line x1="15" y1="15" x2="21" y2="21"></line>
            </svg>
          </span>
          <span class="sr-only">Open full view</span>
        </button>
        <iframe
          loading="lazy"
          allowfullscreen
          data-base="${base}"></iframe>
      </div>`;
    })();

    const textBlock = `
        <div class="modal-text">
          <p class="modal-subtitle">${p.subtitle}</p>
          <h4>Problem</h4><p>${p.problem}</p>
          <h4>Action</h4><ul>${p.actions.map(a => `<li>${a}</li>`).join('')}</ul>
          <h4>Result</h4><ul>${p.results.map(r => `<li>${r}</li>`).join('')}</ul>
        </div>`;

    const contentBlock = isTableau
      ? `<div class="modal-tableau-layout">${textBlock}${visual}</div>`
      : `${textBlock}${visual}`;

    return `
    <div class="modal-content ${(isTableau || isIframe) ? 'modal-wide' : ''}" role="dialog" aria-modal="true" tabindex="0" aria-labelledby="${p.id}-title">
      <button class="modal-copy" type="button" aria-label="Copy link to this project">Copy link</button>
      <button class="modal-close" aria-label="Close dialog">&times;</button>
      <div class="modal-title-strip"><h3 class="modal-title" id="${p.id}-title">${p.title}</h3></div>

      <div class="modal-body ${isTableau ? 'stacked' : ''}">
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
              ${p.resources.map(r => `
                <a href="${r.url}" target="_blank" rel="noopener" title="${r.label}">
                  <img src="${r.icon}" alt="${r.label}" class="icon" width="30" height="30">
                </a>`).join('')}
            </div>
          </div>
        </div>

        ${contentBlock}
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
      if (closeMediaViewer()) {
        e.preventDefault();
        return;
      }
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
