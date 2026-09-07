/* Full-resolution project images with bounded pan and zoom. */
(() => {
  'use strict';

  if (window.ProjectImageViewer) return;
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
  let nextId = 0;
  let activeClose = null;

  function mount(root = document) {
    const disposers = [];
    root.querySelectorAll('[data-project-image-viewer]').forEach((trigger) => {
      trigger._projectImageViewerCleanup?.();
      const listeners = new AbortController();
      const options = { signal: listeners.signal };
      let dialog;
      let viewport;
      let image;
      let zoomInput;
      let zoomOutput;
      let status;
      let retry;
      let resizeObserver;
      let animation;
      let disposed = false;
      let loaded = false;
      let closing = false;
      let openVersion = 0;
      let source = '';
      let width = 0;
      let height = 0;
      let fit = 1;
      let zoom = 1;
      let x = 0;
      let y = 0;
      const pointers = new Map();

      function paint() {
        if (!loaded || !width || !height) return;
        const scaledWidth = image.naturalWidth * fit * zoom;
        const scaledHeight = image.naturalHeight * fit * zoom;
        x = clamp(x, -Math.max(0, (scaledWidth - width) / 2), Math.max(0, (scaledWidth - width) / 2));
        y = clamp(y, -Math.max(0, (scaledHeight - height) / 2), Math.max(0, (scaledHeight - height) / 2));
        image.style.transform = `translate(${(width - scaledWidth) / 2 + x}px, ${(height - scaledHeight) / 2 + y}px) scale(${fit * zoom})`;
        zoomInput.value = String(zoom);
        zoomOutput.value = `${zoom.toFixed(1)}×`;
        zoomInput.setAttribute('aria-valuetext', `${zoom.toFixed(1)} times fitted size`);
        dialog.querySelector('[data-viewer-out]').disabled = zoom <= 1;
        dialog.querySelector('[data-viewer-in]').disabled = zoom >= 8;
        viewport.dataset.zoomed = String(zoom > 1);
      }

      function measure() {
        if (!dialog?.open || !loaded) return;
        const oldFit = fit;
        width = viewport.clientWidth;
        height = viewport.clientHeight;
        fit = Math.min(1, width / image.naturalWidth, height / image.naturalHeight);
        if (oldFit > 0) {
          x *= fit / oldFit;
          y *= fit / oldFit;
        }
        paint();
      }

      function setZoom(value, anchor = { x: width / 2, y: height / 2 }, movement = { x: 0, y: 0 }) {
        if (!loaded) return;
        const next = clamp(value, 1, 8);
        const ratio = next / zoom;
        x = (x + width / 2 - anchor.x) * ratio + anchor.x - width / 2 + movement.x;
        y = (y + height / 2 - anchor.y) * ratio + anchor.y - height / 2 + movement.y;
        zoom = next;
        paint();
      }

      function reset() {
        zoom = 1;
        x = 0;
        y = 0;
        paint();
      }

      function close({ immediate = false, restoreFocus = true } = {}) {
        if (!dialog?.open || (closing && !immediate)) return;
        closing = true;
        const version = ++openVersion;
        pointers.clear();
        viewport.classList.remove('is-dragging');
        dialog.classList.remove('is-visible');
        animation?.cancel();
        const finish = () => {
          if (version !== openVersion) return;
          dialog.close();
          closing = false;
          document.body.classList.remove('project-image-viewer-open');
          if (activeClose === close) activeClose = null;
          if (restoreFocus && trigger.isConnected && !trigger.closest('[inert]')) trigger.focus({ preventScroll: true });
        };
        if (immediate || window.matchMedia('(prefers-reduced-motion: reduce)').matches || !dialog.animate) {
          finish();
        } else {
          animation = dialog.animate([{ opacity: 1, transform: 'translateY(0)' }, { opacity: 0, transform: 'translateY(8px)' }], { duration: 160, easing: 'ease-out' });
          animation.finished.then(finish, () => {});
        }
      }

      function loadImage() {
        loaded = false;
        image.hidden = true;
        status.textContent = 'Loading map…';
        retry.hidden = true;
        viewport.setAttribute('aria-busy', 'true');
        dialog.querySelectorAll('[data-viewer-control]').forEach((control) => { control.disabled = true; });
        image.src = source;
      }

      function createDialog() {
        const id = `project-image-viewer-${++nextId}`;
        dialog = document.createElement('dialog');
        dialog.className = 'project-image-viewer';
        dialog.setAttribute('aria-labelledby', `${id}-title`);
        dialog.setAttribute('aria-modal', 'true');
        dialog.innerHTML = `<header class="project-image-viewer-header">
          <h2 id="${id}-title"></h2>
          <button class="project-image-viewer-close" type="button" data-viewer-close aria-label="Close enlarged map"><span aria-hidden="true">×</span></button>
        </header>
        <div class="project-image-viewer-toolbar">
          <button type="button" data-viewer-out data-viewer-control aria-label="Zoom out">−</button>
          <label class="project-image-viewer-zoom">Zoom <output data-viewer-output>1.0×</output><input type="range" min="1" max="8" step="0.1" value="1" data-viewer-zoom data-viewer-control></label>
          <button type="button" data-viewer-in data-viewer-control aria-label="Zoom in">+</button>
          <button type="button" data-viewer-reset data-viewer-control>Reset</button>
        </div>
        <div class="project-image-viewer-viewport" tabindex="0" role="region" aria-label="Enlarged map" aria-describedby="${id}-help">
          <img draggable="false" hidden>
          <div class="project-image-viewer-feedback"><p role="status" data-viewer-status></p><button type="button" data-viewer-retry hidden>Retry</button></div>
        </div>
        <p class="project-image-viewer-help" id="${id}-help">Drag to pan. Pinch or use Zoom to enlarge. Arrow keys pan; + and − zoom; 0 resets.</p>`;
        dialog.querySelector('h2').textContent = trigger.dataset.imageViewerTitle || 'Enlarged project image';
        viewport = dialog.querySelector('.project-image-viewer-viewport');
        image = viewport.querySelector('img');
        image.alt = trigger.dataset.imageViewerAlt || trigger.querySelector('img')?.alt || 'Project image';
        zoomInput = dialog.querySelector('[data-viewer-zoom]');
        zoomOutput = dialog.querySelector('[data-viewer-output]');
        status = dialog.querySelector('[data-viewer-status]');
        retry = dialog.querySelector('[data-viewer-retry]');
        document.body.append(dialog);

        image.addEventListener('load', () => {
          if (disposed || !image.naturalWidth) return;
          loaded = true;
          image.width = image.naturalWidth;
          image.height = image.naturalHeight;
          image.hidden = false;
          status.textContent = '';
          viewport.setAttribute('aria-busy', 'false');
          dialog.querySelectorAll('[data-viewer-control]').forEach((control) => { control.disabled = false; });
          measure();
        }, options);
        image.addEventListener('error', () => {
          if (disposed) return;
          status.textContent = 'The map could not be loaded. Please try again.';
          retry.hidden = false;
          viewport.setAttribute('aria-busy', 'false');
        }, options);
        retry.addEventListener('click', () => {
          image.removeAttribute('src');
          loadImage();
        }, options);
        dialog.querySelector('[data-viewer-close]').addEventListener('click', () => close(), options);
        dialog.addEventListener('cancel', (event) => {
          event.preventDefault();
          close();
        }, options);
        dialog.addEventListener('keydown', (event) => {
          if (event.key !== 'Tab') return;
          const focusable = [...dialog.querySelectorAll('button:not(:disabled), input:not(:disabled), [tabindex="0"]')]
            .filter((element) => !element.hidden && element.getClientRects().length);
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          if ((event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
            event.preventDefault();
            (event.shiftKey ? last : first)?.focus({ preventScroll: true });
          }
        }, options);
        let backdropDown = false;
        const outside = (event) => {
          const rect = dialog.getBoundingClientRect();
          return event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom;
        };
        dialog.addEventListener('pointerdown', (event) => { backdropDown = event.target === dialog && outside(event); }, options);
        dialog.addEventListener('click', (event) => {
          if (backdropDown && event.target === dialog && outside(event)) close();
          backdropDown = false;
        }, options);
        zoomInput.addEventListener('input', () => setZoom(Number(zoomInput.value)), options);
        dialog.querySelector('[data-viewer-out]').addEventListener('click', () => setZoom(zoom / 1.25), options);
        dialog.querySelector('[data-viewer-in]').addEventListener('click', () => setZoom(zoom * 1.25), options);
        dialog.querySelector('[data-viewer-reset]').addEventListener('click', reset, options);
        viewport.addEventListener('keydown', (event) => {
          if (event.target !== viewport || !loaded) return;
          const step = event.shiftKey ? 100 : 40;
          if (['+', '='].includes(event.key)) setZoom(zoom * 1.25);
          else if (event.key === '-') setZoom(zoom / 1.25);
          else if (['0', 'Home'].includes(event.key)) reset();
          else if (event.key === 'ArrowLeft') x += step;
          else if (event.key === 'ArrowRight') x -= step;
          else if (event.key === 'ArrowUp') y += step;
          else if (event.key === 'ArrowDown') y -= step;
          else return;
          event.preventDefault();
          paint();
        }, options);
        const point = (event) => {
          const rect = viewport.getBoundingClientRect();
          return { x: event.clientX - rect.left, y: event.clientY - rect.top };
        };
        const pair = () => {
          const [first, second] = [...pointers.values()];
          return { center: { x: (first.x + second.x) / 2, y: (first.y + second.y) / 2 }, distance: Math.hypot(first.x - second.x, first.y - second.y) };
        };
        viewport.addEventListener('pointerdown', (event) => {
          if (!loaded || closing || (event.pointerType === 'mouse' && event.button !== 0)) return;
          event.preventDefault();
          viewport.focus({ preventScroll: true });
          pointers.set(event.pointerId, point(event));
          viewport.setPointerCapture(event.pointerId);
          viewport.classList.add('is-dragging');
        }, options);
        viewport.addEventListener('pointermove', (event) => {
          if (!pointers.has(event.pointerId)) return;
          const before = pointers.get(event.pointerId);
          const previousPair = pointers.size === 2 ? pair() : null;
          const current = point(event);
          pointers.set(event.pointerId, current);
          if (previousPair) {
            const currentPair = pair();
            setZoom(zoom * currentPair.distance / Math.max(1, previousPair.distance), previousPair.center, {
              x: currentPair.center.x - previousPair.center.x,
              y: currentPair.center.y - previousPair.center.y
            });
          } else if (pointers.size === 1) {
            x += current.x - before.x;
            y += current.y - before.y;
            paint();
          }
        }, options);
        const release = (event) => {
          pointers.delete(event.pointerId);
          if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
          if (!pointers.size) viewport.classList.remove('is-dragging');
        };
        ['pointerup', 'pointercancel', 'lostpointercapture'].forEach((type) => viewport.addEventListener(type, release, options));
        viewport.addEventListener('wheel', (event) => {
          if (!loaded) return;
          event.preventDefault();
          setZoom(zoom * Math.exp(-event.deltaY * 0.002), point(event));
        }, { ...options, passive: false });
        resizeObserver = new ResizeObserver(measure);
        resizeObserver.observe(viewport);
        window.visualViewport?.addEventListener('resize', measure, options);
      }

      trigger.addEventListener('click', (event) => {
        if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        if (disposed || typeof HTMLDialogElement === 'undefined' || !HTMLDialogElement.prototype.showModal) return;
        event.preventDefault();
        source = trigger.href || trigger.dataset.imageViewerSrc;
        if (!source) return;
        activeClose?.({ immediate: true, restoreFocus: false });
        if (!dialog) createDialog();
        ++openVersion;
        closing = false;
        activeClose = close;
        dialog.showModal();
        document.body.classList.add('project-image-viewer-open');
        // Establish the backdrop's initial opacity before its entrance transition.
        dialog.getBoundingClientRect();
        dialog.classList.add('is-visible');
        reset();
        if (!loaded) loadImage();
        else measure();
        dialog.querySelector('[data-viewer-close]').focus({ preventScroll: true });
        if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches && dialog.animate) {
          animation = dialog.animate([{ opacity: 0, transform: 'translateY(8px)' }, { opacity: 1, transform: 'translateY(0)' }], { duration: 180, easing: 'ease-out' });
        }
      }, options);

      const dispose = () => {
        if (disposed) return;
        disposed = true;
        close({ immediate: true, restoreFocus: false });
        listeners.abort();
        resizeObserver?.disconnect();
        animation?.cancel();
        image?.removeAttribute('src');
        dialog?.remove();
        if (trigger._projectImageViewerCleanup === dispose) delete trigger._projectImageViewerCleanup;
      };
      trigger._projectImageViewerCleanup = dispose;
      disposers.push(dispose);
    });
    return () => disposers.forEach((dispose) => dispose());
  }

  window.ProjectImageViewer = Object.freeze({ mount });
  const initialize = (root) => {
    const dispose = mount(root);
    window.SiteRoutes?.addCleanup?.(dispose);
  };
  document.addEventListener('DOMContentLoaded', () => initialize(document));
  document.addEventListener('site:content-updated', (event) => initialize(event.detail?.root || document));
  if (document.readyState !== 'loading') initialize(document.querySelector('[data-site-route-content], [data-personal-detail-content]') || document);
})();
