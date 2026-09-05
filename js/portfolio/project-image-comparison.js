/* ===================================================================
   File: project-image-comparison.js
   Purpose: Reveal three aligned project images with two accessible dividers.
=================================================================== */
(() => {
  'use strict';

  const $ = (selector, context = document) => context.querySelector(selector);
  const $$ = (selector, context = document) => [...context.querySelectorAll(selector)];
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
  const readNumber = (value, fallback) => {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  };
  // Retain small view state across soft navigation without keeping detached DOM.
  const savedComparisons = new Map();

  function initRegionSelection(comparison, slides, saved, listenerOptions) {
    const overview = $('[data-comparison-overview]', comparison);
    const selection = $('[data-comparison-selection]', comparison);
    const controls = $('[data-selection-controls]', comparison);
    const zoomInput = $('[data-selection-zoom]', comparison);
    const zoomValue = $('[data-selection-zoom-value]', comparison);
    const reset = $('[data-selection-reset]', comparison);
    const status = $('[data-selection-status]', comparison);
    const retry = $('[data-selection-retry]', comparison);
    const instructions = $('[data-selection-instructions]', comparison);
    if (!overview || !selection || !controls || !zoomInput || !reset) return () => {};

    let initial;
    try { initial = JSON.parse(comparison.dataset.comparisonCrop); } catch (_) { return () => {}; }
    const pageRatio = readNumber(comparison.dataset.comparisonPageRatio, 612 / 792);
    const defaultState = {
      x: (initial.left + initial.width / 2) / 100,
      y: (initial.top + initial.height / 2) / 100,
      zoom: 100 / initial.width
    };
    let state = { ...(saved.region || defaultState) };
    let disposed = false;
    let ready = false;
    let request = 0;
    let pointer = null;
    let frame = 0;
    let pending = null;
    const pageImages = slides.map((slide) => $('img', slide));
    selection.disabled = true;

    const renderRegion = () => {
      state.zoom = clamp(readNumber(state.zoom, defaultState.zoom), 1, 6);
      const width = 1 / state.zoom;
      const height = width * pageRatio / (16 / 9);
      state.x = clamp(readNumber(state.x, defaultState.x), width / 2, 1 - width / 2);
      state.y = clamp(readNumber(state.y, defaultState.y), height / 2, 1 - height / 2);
      const left = state.x - width / 2;
      const top = state.y - height / 2;
      comparison.style.setProperty('--comparison-crop-left', `${left * 100}%`);
      comparison.style.setProperty('--comparison-crop-top', `${top * 100}%`);
      comparison.style.setProperty('--comparison-crop-width', `${width * 100}%`);
      comparison.style.setProperty('--comparison-crop-height', `${height * 100}%`);
      comparison.style.setProperty('--comparison-crop-center-x', `${state.x * 100}%`);
      comparison.style.setProperty('--comparison-crop-center-y', `${state.y * 100}%`);
      comparison.style.setProperty('--comparison-source-width', `${100 / width}%`);
      comparison.style.setProperty('--comparison-source-height', `${100 / height}%`);
      comparison.style.setProperty('--comparison-source-left', `${-left / width * 100}%`);
      comparison.style.setProperty('--comparison-source-top', `${-top / height * 100}%`);
      zoomInput.value = String(state.zoom);
      if (zoomValue) zoomValue.textContent = `${Number(state.zoom.toFixed(2))}×`;
      zoomInput.setAttribute('aria-valuetext', `${Number(state.zoom.toFixed(2))} times magnification`);
      selection.setAttribute('aria-label', `Move selected area, centered ${Math.round(state.x * 100)}% across and ${Math.round(state.y * 100)}% down the sheet`);
      saved.region = { ...state };
    };

    const positionFromPointer = (event) => {
      const rect = overview.getBoundingClientRect();
      return { x: (event.clientX - rect.left) / Math.max(rect.width, 1), y: (event.clientY - rect.top) / Math.max(rect.height, 1) };
    };
    const flush = () => {
      frame = 0;
      if (!pending) return;
      state.x = pending.x;
      state.y = pending.y;
      pending = null;
      renderRegion();
    };
    const queue = (position) => {
      pending = position;
      if (!frame) frame = window.requestAnimationFrame(flush);
    };
    const focusSelection = () => {
      try { selection.focus({ preventScroll: true }); } catch (_) { selection.focus(); }
    };

    overview.addEventListener('click', (event) => {
      if (!ready || event.defaultPrevented || event.button !== 0) return;
      if (event.target.closest?.('[data-comparison-selection]')) return;
      const position = positionFromPointer(event);
      state.x = position.x;
      state.y = position.y;
      renderRegion();
      focusSelection();
    }, listenerOptions);
    selection.addEventListener('pointerdown', (event) => {
      if (!ready || event.isPrimary === false || (event.pointerType === 'mouse' && event.button !== 0)) return;
      const position = positionFromPointer(event);
      pointer = { id: event.pointerId, x: position.x, y: position.y, startX: state.x, startY: state.y };
      selection.setPointerCapture(event.pointerId);
      selection.classList.add('is-dragging');
      focusSelection();
    }, listenerOptions);
    selection.addEventListener('pointermove', (event) => {
      if (!pointer || event.pointerId !== pointer.id) return;
      const position = positionFromPointer(event);
      queue({ x: pointer.startX + position.x - pointer.x, y: pointer.startY + position.y - pointer.y });
    }, listenerOptions);
    const endPointer = (event) => {
      if (!pointer || pointer.id !== event.pointerId) return;
      if (frame) window.cancelAnimationFrame(frame);
      flush();
      const id = pointer.id;
      pointer = null;
      selection.classList.remove('is-dragging');
      if (selection.hasPointerCapture(id)) selection.releasePointerCapture(id);
    };
    selection.addEventListener('pointerup', endPointer, listenerOptions);
    selection.addEventListener('pointercancel', endPointer, listenerOptions);
    selection.addEventListener('lostpointercapture', endPointer, listenerOptions);
    selection.addEventListener('keydown', (event) => {
      const step = event.shiftKey ? 0.05 : 0.01;
      if (event.key === 'ArrowLeft') state.x -= step;
      else if (event.key === 'ArrowRight') state.x += step;
      else if (event.key === 'ArrowUp') state.y -= step;
      else if (event.key === 'ArrowDown') state.y += step;
      else return;
      event.preventDefault();
      renderRegion();
    }, listenerOptions);
    zoomInput.addEventListener('input', () => {
      state.zoom = readNumber(zoomInput.value, defaultState.zoom);
      renderRegion();
    }, listenerOptions);
    reset.addEventListener('click', () => {
      state = { ...defaultState };
      renderRegion();
    }, listenerOptions);

    const loadImages = async () => {
      const version = ++request;
      if (status) status.textContent = 'Preparing the selectable sheet…';
      if (retry) retry.hidden = true;
      try {
        const loaded = await Promise.all(pageImages.map(async (image) => {
          const source = image?.dataset.comparisonSource;
          if (!source) throw new Error('Missing comparison image');
          const preload = new Image();
          preload.src = source;
          await preload.decode();
          if (!preload.naturalWidth || !preload.naturalHeight || Math.abs(preload.naturalWidth / preload.naturalHeight - pageRatio) > 0.0001) {
            throw new Error('Comparison images are not aligned');
          }
          return preload;
        }));
        if (disposed || request !== version) return;
        // Keep the working static crop visible until every full image is decoded.
        loaded.forEach((image, index) => {
          pageImages[index].src = image.src;
          pageImages[index].width = image.naturalWidth;
          pageImages[index].height = image.naturalHeight;
        });
        ready = true;
        comparison.classList.add('is-selectable');
        controls.hidden = false;
        if (instructions) instructions.hidden = false;
        selection.disabled = false;
        if (status) status.textContent = '';
        renderRegion();
      } catch (_) {
        if (disposed || request !== version) return;
        if (status) status.textContent = 'The full sheet could not be loaded. The sample comparison is still available.';
        if (retry) retry.hidden = false;
      }
    };
    retry?.addEventListener('click', loadImages, listenerOptions);
    loadImages();
    return () => {
      disposed = true;
      request += 1;
      if (frame) window.cancelAnimationFrame(frame);
      pending = null;
      if (pointer && selection.hasPointerCapture(pointer.id)) selection.releasePointerCapture(pointer.id);
      pointer = null;
      selection.classList.remove('is-dragging');
      selection.disabled = true;
    };
  }

  function initProjectImageComparisons(root = document) {
    const initialized = [];
    $$('[data-project-image-comparison]', root).forEach((comparison) => {
      if (comparison.dataset.projectImageComparisonReady === 'true') return;
      if (window.CSS && typeof window.CSS.supports === 'function' && !window.CSS.supports('clip-path', 'inset(0)')) return;

      const viewport = $('[data-comparison-viewport]', comparison);
      const slides = $$('[data-stage-slide]', comparison);
      const dividers = $$('[data-comparison-divider]', comparison);
      const controls = $('[data-comparison-controls]', comparison);
      const leftDivider = dividers.find((divider) => divider.dataset.comparisonDivider === 'left');
      const rightDivider = dividers.find((divider) => divider.dataset.comparisonDivider === 'right');
      if (!viewport || slides.length !== 3 || !controls || !leftDivider || !rightDivider) return;

      const configuredGap = clamp(readNumber(comparison.dataset.comparisonMinimumGap, 10), 6, 30);
      const stateKey = comparison.dataset.comparisonId;
      const saved = savedComparisons.get(stateKey) || {};
      if (stateKey) {
        savedComparisons.delete(stateKey);
        savedComparisons.set(stateKey, saved);
        if (savedComparisons.size > 8) savedComparisons.delete(savedComparisons.keys().next().value);
      }
      let left = readNumber(saved.left ?? comparison.dataset.comparisonLeft, 33);
      let right = readNumber(saved.right ?? comparison.dataset.comparisonRight, 67);
      let animationFrame = 0;
      let pendingPointerMove = null;
      const eventController = typeof AbortController === 'function' ? new AbortController() : null;
      const listenerOptions = eventController ? { signal: eventController.signal } : undefined;

      const getBounds = () => {
        const width = Math.max(viewport.getBoundingClientRect().width, 1);
        const edge = Math.min(12, (22 / width) * 100);
        const pointerGap = Math.min(24, (44 / width) * 100);
        return {
          edge,
          gap: Math.max(configuredGap, pointerGap)
        };
      };

      const getDividerRange = (side, bounds = getBounds()) => side === 'left'
        ? {
          minimum: bounds.edge,
          maximum: 100 - bounds.edge - bounds.gap
        }
        : {
          minimum: bounds.edge + bounds.gap,
          maximum: 100 - bounds.edge
        };

      const normalizeState = () => {
        const bounds = getBounds();
        right = clamp(right, bounds.edge + bounds.gap, 100 - bounds.edge);
        left = clamp(left, bounds.edge, right - bounds.gap);
        right = clamp(right, left + bounds.gap, 100 - bounds.edge);
      };

      const updateDividerAria = (divider, side) => {
        const range = getDividerRange(side);
        const value = side === 'left' ? left : right;
        const roundedMinimum = Math.round(range.minimum);
        const roundedMaximum = Math.round(range.maximum);
        const roundedValue = clamp(Math.round(value), roundedMinimum, roundedMaximum);
        const before = divider.dataset.comparisonBefore || 'Previous stage';
        const after = divider.dataset.comparisonAfter || 'Next stage';
        divider.setAttribute('aria-valuemin', String(roundedMinimum));
        divider.setAttribute('aria-valuemax', String(roundedMaximum));
        divider.setAttribute('aria-valuenow', String(roundedValue));
        divider.setAttribute('aria-valuetext', `${before} ends at ${roundedValue}%; ${after} begins at ${roundedValue}%`);
      };

      const render = () => {
        comparison.style.setProperty('--comparison-left', `${left.toFixed(2)}%`);
        comparison.style.setProperty('--comparison-right', `${right.toFixed(2)}%`);
        comparison.dataset.comparisonLeft = left.toFixed(2);
        comparison.dataset.comparisonRight = right.toFixed(2);
        saved.left = left;
        saved.right = right;
        updateDividerAria(leftDivider, 'left');
        updateDividerAria(rightDivider, 'right');
      };

      const moveDivider = (side, requestedValue) => {
        const bounds = getBounds();
        const range = getDividerRange(side, bounds);
        const value = clamp(requestedValue, range.minimum, range.maximum);

        if (side === 'left') {
          left = value;
          right = Math.max(right, left + bounds.gap);
        } else {
          right = value;
          left = Math.min(left, right - bounds.gap);
        }

        render();
      };

      const valueFromPointer = (clientX) => {
        const rect = viewport.getBoundingClientRect();
        if (!rect.width) return 0;
        return ((clientX - rect.left) / rect.width) * 100;
      };

      const focusDivider = (divider) => {
        try {
          divider.focus({ preventScroll: true });
        } catch (_) {
          divider.focus();
        }
      };

      const flushPointerMove = () => {
        if (!pendingPointerMove) return;
        const { side, clientX } = pendingPointerMove;
        pendingPointerMove = null;
        moveDivider(side, valueFromPointer(clientX));
      };

      const queuePointerMove = (side, clientX) => {
        pendingPointerMove = { side, clientX };
        if (animationFrame) return;
        animationFrame = window.requestAnimationFrame(() => {
          animationFrame = 0;
          flushPointerMove();
        });
      };

      const bindDivider = (divider, side) => {
        let activePointerId = null;

        divider.addEventListener('pointerdown', (event) => {
          if (event.isPrimary === false || (event.pointerType === 'mouse' && event.button !== 0)) return;
          activePointerId = event.pointerId;
          divider.classList.add('is-dragging');
          divider.setPointerCapture(event.pointerId);
          focusDivider(divider);
          queuePointerMove(side, event.clientX);
        }, listenerOptions);

        divider.addEventListener('pointermove', (event) => {
          if (event.pointerId !== activePointerId) return;
          queuePointerMove(side, event.clientX);
        }, listenerOptions);

        const endPointer = (event) => {
          if (event.pointerId !== activePointerId) return;
          if (animationFrame) {
            window.cancelAnimationFrame(animationFrame);
            animationFrame = 0;
          }
          flushPointerMove();
          const pointerId = activePointerId;
          activePointerId = null;
          divider.classList.remove('is-dragging');
          if (divider.hasPointerCapture(pointerId)) divider.releasePointerCapture(pointerId);
        };

        divider.addEventListener('pointerup', endPointer, listenerOptions);
        divider.addEventListener('pointercancel', endPointer, listenerOptions);
        divider.addEventListener('lostpointercapture', (event) => {
          if (event.pointerId !== activePointerId) return;
          activePointerId = null;
          divider.classList.remove('is-dragging');
        }, listenerOptions);

        divider.addEventListener('keydown', (event) => {
          const range = getDividerRange(side);
          const current = side === 'left' ? left : right;
          const arrowStep = event.shiftKey ? 10 : 1;
          let nextValue = null;

          if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') nextValue = current - arrowStep;
          if (event.key === 'ArrowRight' || event.key === 'ArrowUp') nextValue = current + arrowStep;
          if (event.key === 'PageDown') nextValue = current - 10;
          if (event.key === 'PageUp') nextValue = current + 10;
          if (event.key === 'Home') nextValue = range.minimum;
          if (event.key === 'End') nextValue = range.maximum;
          if (nextValue === null) return;

          event.preventDefault();
          moveDivider(side, nextValue);
        }, listenerOptions);
      };

      normalizeState();
      comparison.dataset.projectImageComparisonReady = 'true';
      comparison.classList.add('is-enhanced');
      comparison.inert = false;
      controls.hidden = false;
      dividers.forEach((divider) => {
        divider.hidden = false;
      });
      bindDivider(leftDivider, 'left');
      bindDivider(rightDivider, 'right');
      viewport.addEventListener('click', (event) => {
        if (event.defaultPrevented || event.button !== 0 || event.target.closest?.('[data-comparison-divider]')) return;

        const requestedValue = valueFromPointer(event.clientX);
        const side = Math.abs(requestedValue - left) <= Math.abs(requestedValue - right)
          ? 'left'
          : 'right';
        const divider = side === 'left' ? leftDivider : rightDivider;

        moveDivider(side, requestedValue);
        focusDivider(divider);
      }, listenerOptions);
      render();
      const disposeSelection = initRegionSelection(comparison, slides, saved, listenerOptions);

      slides.forEach((slide) => {
        const image = $('img', slide);
        if (!image) return;
        image.loading = 'eager';
        if (typeof image.decode === 'function') image.decode().catch(() => {});
      });

      const handleResize = () => {
        normalizeState();
        render();
      };
      if (typeof window.ResizeObserver === 'function') {
        const observer = new window.ResizeObserver(handleResize);
        observer.observe(viewport);
        comparison._projectImageComparisonObserver = observer;
      } else {
        window.addEventListener('resize', handleResize, eventController
          ? { passive: true, signal: eventController.signal }
          : { passive: true });
      }

      let cleaned = false;
      comparison._projectImageComparisonCleanup = () => {
        if (cleaned) return;
        cleaned = true;
        disposeSelection();
        eventController?.abort();
        if (animationFrame) window.cancelAnimationFrame(animationFrame);
        animationFrame = 0;
        pendingPointerMove = null;
        comparison._projectImageComparisonObserver?.disconnect?.();
        comparison._projectImageComparisonObserver = null;
        comparison._projectImageComparisonCleanup = null;
        // Retain the outgoing layout while the route frame removes its content.
        comparison.inert = true;
        delete comparison.dataset.projectImageComparisonReady;
        dividers.forEach((divider) => {
          divider.classList.remove('is-dragging');
        });
      };
      initialized.push(comparison._projectImageComparisonCleanup);
    });
    return () => initialized.forEach((dispose) => dispose());
  }

  window.ProjectImageComparisons = Object.freeze({ mount: initProjectImageComparisons });
  document.addEventListener('DOMContentLoaded', () => {
    const dispose = initProjectImageComparisons();
    window.SiteRoutes?.addCleanup?.(dispose);
  });
  document.addEventListener('site:content-updated', (event) => {
    const dispose = initProjectImageComparisons(event.detail?.root || document);
    window.SiteRoutes?.addCleanup?.(dispose);
  });
  if (document.readyState !== 'loading') {
    const dispose = initProjectImageComparisons(
      document.querySelector('[data-site-route-content], [data-personal-detail-content]') || document
    );
    window.SiteRoutes?.addCleanup?.(dispose);
  }
})();
