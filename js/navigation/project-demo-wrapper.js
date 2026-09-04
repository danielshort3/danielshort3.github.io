/* ===================================================================
   File: project-demo-wrapper.js
   Purpose: Route-safe sizing for same-origin project demo wrappers.
=================================================================== */
(() => {
  'use strict';

  const mountProjectDemoWrapper = (context) => {
    const root = context?.root || document;
    const localCleanups = [];
    const cleanup = typeof context?.cleanup === 'function'
      ? context.cleanup
      : (callback) => localCleanups.push(callback);
    const frame = root.querySelector?.('.project-demo-wrapper-iframe');
    if (!frame) return;
    const frameContainer = frame.closest('.project-demo-wrapper-frame');
    const frameMain = frame.closest('.project-demo-wrapper-main');
    const mobileQuery = window.matchMedia('(max-width: 959px)');
    const routeUrl = context?.url instanceof URL
      ? context.url
      : new URL(String(context?.url || window.location.href), window.location.href);
    const suffix = routeUrl.search + routeUrl.hash;
    let resizeObserver = null;
    let resizeFrame = 0;
    let resizeTimers = [];
    let lastHeight = 0;
    let active = true;

    const listen = (target, type, listener, options) => {
      target?.addEventListener?.(type, listener, options);
      cleanup(() => target?.removeEventListener?.(type, listener, options));
    };

    const clearScheduledMeasurements = () => {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = 0;
      resizeTimers.forEach((timer) => window.clearTimeout(timer));
      resizeTimers = [];
    };

    const disconnectObserver = () => {
      resizeObserver?.disconnect?.();
      resizeObserver = null;
    };

    const clearMobileHeight = () => {
      clearScheduledMeasurements();
      [frameMain, frameContainer, frame].forEach((element) => element?.style?.removeProperty('height'));
      document.body.removeAttribute('data-project-demo-autosize');
      lastHeight = 0;
    };

    const measureFrame = () => {
      resizeFrame = 0;
      if (!active || context?.signal?.aborted || !frameContainer || !frameMain || !mobileQuery.matches) return;
      try {
        const frameDocument = frame.contentDocument;
        const frameBody = frameDocument?.body;
        const frameRoot = frameDocument?.documentElement;
        if (!frameBody || !frameRoot) return;
        const nextHeight = Math.max(
          560,
          frameBody.scrollHeight,
          frameBody.offsetHeight,
          frameRoot.scrollHeight,
          frameRoot.offsetHeight
        );
        if (!Number.isFinite(nextHeight) || Math.abs(nextHeight - lastHeight) < 2) return;
        lastHeight = Math.ceil(nextHeight);
        const height = `${lastHeight}px`;
        frameMain.style.height = height;
        frameContainer.style.height = height;
        frame.style.height = height;
        document.body.setAttribute('data-project-demo-autosize', 'true');
      } catch (_) {
        clearMobileHeight();
      }
    };

    const scheduleMeasurement = () => {
      if (!active || !mobileQuery.matches || resizeFrame) return;
      resizeFrame = window.requestAnimationFrame(measureFrame);
    };

    const observeFrame = () => {
      disconnectObserver();
      clearScheduledMeasurements();
      if (!active || !mobileQuery.matches) {
        clearMobileHeight();
        return;
      }
      try {
        const frameDocument = frame.contentDocument;
        const frameBody = frameDocument?.body;
        const frameRoot = frameDocument?.documentElement;
        if (!frameBody || !frameRoot) return;
        const FrameResizeObserver = frame.contentWindow?.ResizeObserver;
        if (FrameResizeObserver) {
          resizeObserver = new FrameResizeObserver(scheduleMeasurement);
          resizeObserver.observe(frameBody);
          resizeObserver.observe(frameRoot);
        }
        scheduleMeasurement();
        [120, 500, 1500].forEach((delay) => {
          resizeTimers.push(window.setTimeout(scheduleMeasurement, delay));
        });
      } catch (_) {
        clearMobileHeight();
      }
    };

    const handleViewportChange = () => {
      if (mobileQuery.matches) observeFrame();
      else {
        disconnectObserver();
        clearMobileHeight();
      }
    };

    listen(frame, 'load', observeFrame);
    if (typeof mobileQuery.addEventListener === 'function') {
      listen(mobileQuery, 'change', handleViewportChange);
    } else if (typeof mobileQuery.addListener === 'function') {
      mobileQuery.addListener(handleViewportChange);
      cleanup(() => mobileQuery.removeListener(handleViewportChange));
    }

    const baseSrc = frame.dataset.projectDemoSrc || frame.getAttribute('src') || '';
    if (suffix && baseSrc) frame.src = `${baseSrc}${suffix}`;
    resizeTimers.push(window.setTimeout(observeFrame, 0));
    cleanup(() => {
      active = false;
      disconnectObserver();
      clearMobileHeight();
    });
    return () => localCleanups.splice(0).reverse().forEach((callback) => callback());
  };

  window.ProjectDemoWrapper = Object.freeze({ mount: mountProjectDemoWrapper });
  const mountDirect = () => {
    const dispose = mountProjectDemoWrapper({
      root: document,
      url: window.location.href,
      navigationType: document.readyState === 'loading' ? 'load' : 'route'
    });
    window.SiteRoutes?.addCleanup?.(dispose);
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountDirect, { once: true });
  } else {
    mountDirect();
  }
})();
