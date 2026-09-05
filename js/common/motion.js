(() => {
  'use strict';

  if (window.SiteMotion) return;

  const pending = new Map();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const milliseconds = (value) => {
    const number = parseFloat(value);
    return Number.isFinite(number) ? number * (String(value).trim().endsWith('ms') ? 1 : 1000) : 0;
  };
  const duration = (element, token, fallback = 220) => {
    if (reducedMotion.matches) return 0;
    if (typeof token === 'number') return token;
    const value = getComputedStyle(element).getPropertyValue(token || '--motion-base').trim();
    return value ? milliseconds(value) : fallback;
  };
  const easing = (element) => getComputedStyle(element).getPropertyValue('--easing-standard').trim() || 'ease-out';

  const begin = (element, cleanup, onFinish) => {
    pending.get(element)?.settle(false);
    let resolve;
    const promise = new Promise((done) => { resolve = done; });
    const record = {
      animation: null,
      timer: 0,
      settle(completed = true) {
        if (pending.get(element) !== record) return;
        pending.delete(element);
        clearTimeout(record.timer);
        record.animation?.cancel();
        cleanup?.(completed);
        if (completed) onFinish?.();
        resolve(completed);
      }
    };
    pending.set(element, record);
    return { record, promise };
  };

  const finish = (element) => pending.get(element)?.settle(true);
  const cancel = (element) => pending.get(element)?.settle(false);

  const transitionTime = (element) => {
    const style = getComputedStyle(element);
    const times = style.transitionDuration.split(',').map(milliseconds);
    const delays = style.transitionDelay.split(',').map(milliseconds);
    return Math.max(0, ...times.map((time, index) => time + delays[index % delays.length]));
  };

  const presence = (element, open, options = {}) => {
    if (!element) return Promise.resolve(false);
    const { className = 'is-open', enter = '--motion-base', exit = '--motion-fast', hidden = true, onFinish } = options;
    const { record, promise } = begin(element, (completed) => {
      if (!completed) return;
      element.dataset.motionState = open ? 'open' : 'closed';
      if (hidden) element.hidden = !open;
    }, onFinish);
    element.style.setProperty('--motion-duration', `${duration(element, open ? enter : exit)}ms`);
    element.dataset.motionState = open ? 'opening' : 'closing';
    if (open && hidden) element.hidden = false;
    // Establish the visible starting style before changing the target class.
    element.getBoundingClientRect();
    if (className) element.classList.toggle(className, open);
    const time = reducedMotion.matches ? 0 : transitionTime(element);
    if (!time || !element.isConnected) {
      record.settle();
      return promise;
    }
    // Only transitions on this surface determine completion, never a bubbling
    // transitionend from a hovered button or a descendant's looping animation.
    const animations = element.getAnimations?.().filter((animation) => animation.transitionProperty) || [];
    if (animations.length) {
      Promise.all(animations.map((animation) => animation.finished)).then(() => record.settle(), () => {});
    }
    record.timer = window.setTimeout(() => record.settle(), time + 50);
    return promise;
  };

  const height = (element, expanded, options = {}) => {
    if (!element) return Promise.resolve(false);
    const startHeight = element.hidden ? 0 : element.getBoundingClientRect().height;
    cancel(element);
    const originalHeight = element.style.height;
    const originalOverflow = element.style.overflow;
    const originalMaxHeight = element.style.maxHeight;
    const originalBoxSizing = element.style.boxSizing;
    element.hidden = false;
    element.style.height = 'auto';
    element.style.maxHeight = 'none';
    const endHeight = expanded ? element.getBoundingClientRect().height : 0;
    const { record, promise } = begin(element, (completed) => {
      element.style.height = originalHeight;
      element.style.overflow = originalOverflow;
      element.style.maxHeight = originalMaxHeight;
      element.style.boxSizing = originalBoxSizing;
      if (completed) {
        element.dataset.motionState = expanded ? 'open' : 'closed';
        if (options.hidden !== false) element.hidden = !expanded;
      }
    }, options.onFinish);
    element.dataset.motionState = expanded ? 'opening' : 'closing';
    element.style.overflow = 'clip';
    element.style.boxSizing = 'border-box';
    const time = duration(element, options.duration || '--motion-slow', 320);
    if (!time || !element.animate || !element.isConnected || Math.abs(startHeight - endHeight) < 1) {
      record.settle();
      return promise;
    }
    record.animation = element.animate([
      { height: `${startHeight}px` },
      { height: `${endHeight}px` }
    ], { duration: time, easing: easing(element), fill: 'both' });
    record.animation.finished.then(() => record.settle(), () => {});
    record.timer = window.setTimeout(() => record.settle(), time + 50);
    return promise;
  };

  const swap = (element, update, options = {}) => {
    if (!element) {
      update();
      return Promise.resolve(true);
    }
    const startHeight = element.getBoundingClientRect().height;
    cancel(element);
    const originalOverflow = element.style.overflow;
    update();
    const endHeight = element.getBoundingClientRect().height;
    const { record, promise } = begin(element, () => {
      element.style.overflow = originalOverflow;
    }, options.onFinish);
    const time = duration(element, options.duration || '--motion-fast', 160);
    if (!time || !element.animate || !startHeight || !element.isConnected) {
      record.settle();
      return promise;
    }
    element.style.overflow = 'clip';
    const box = getComputedStyle(element);
    const edges = box.boxSizing === 'border-box' ? 0 :
      ['paddingTop', 'paddingBottom', 'borderTopWidth', 'borderBottomWidth']
        .reduce((total, property) => total + (parseFloat(box[property]) || 0), 0);
    record.animation = element.animate([
      { height: `${Math.max(0, startHeight - edges)}px`, opacity: 0 },
      { height: `${Math.max(0, endHeight - edges)}px`, opacity: 1 }
    ], { duration: time, easing: easing(element), fill: 'both' });
    record.animation.finished.then(() => record.settle(), () => {});
    record.timer = window.setTimeout(() => record.settle(), time + 50);
    return promise;
  };

  reducedMotion.addEventListener('change', () => {
    if (reducedMotion.matches) [...pending.values()].forEach((record) => record.settle());
  });
  window.addEventListener('resize', () => {
    [...pending.values()].filter((record) => record.animation).forEach((record) => record.settle());
  }, { passive: true });
  new MutationObserver(() => {
    pending.forEach((record, element) => {
      if (!element.isConnected) record.settle(false);
    });
  }).observe(document.documentElement, { childList: true, subtree: true });

  window.SiteMotion = { duration, presence, height, swap, finish, cancel };
})();
