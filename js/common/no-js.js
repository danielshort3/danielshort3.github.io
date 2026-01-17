(() => {
  'use strict';
  try {
    const root = document.documentElement;
    if (!root) return;
    if (root.classList) {
      root.classList.remove('no-js');
      return;
    }
    root.className = (root.className || '').replace(/\bno-js\b/g, '').trim();
  } catch (_) {}
})();

