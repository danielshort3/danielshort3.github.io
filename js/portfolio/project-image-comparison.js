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

  function initProjectImageComparisons(root = document) {
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
      let left = readNumber(comparison.dataset.comparisonLeft, 33);
      let right = readNumber(comparison.dataset.comparisonRight, 67);
      let animationFrame = 0;
      let pendingPointerMove = null;

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
        });

        divider.addEventListener('pointermove', (event) => {
          if (event.pointerId !== activePointerId) return;
          queuePointerMove(side, event.clientX);
        });

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

        divider.addEventListener('pointerup', endPointer);
        divider.addEventListener('pointercancel', endPointer);
        divider.addEventListener('lostpointercapture', (event) => {
          if (event.pointerId !== activePointerId) return;
          activePointerId = null;
          divider.classList.remove('is-dragging');
        });

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
        });
      };

      normalizeState();
      comparison.dataset.projectImageComparisonReady = 'true';
      comparison.classList.add('is-enhanced');
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
      });
      render();

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
        window.addEventListener('resize', handleResize, { passive: true });
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => initProjectImageComparisons());
  document.addEventListener('site:content-updated', () => initProjectImageComparisons());
})();
