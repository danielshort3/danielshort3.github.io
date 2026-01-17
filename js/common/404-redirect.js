(() => {
  'use strict';
  try {
    const path = String(location.pathname || '');
    const match = path.match(/(?:^|\/)portfolio\/([A-Za-z0-9_-]+)\/?$/);
    if (match && match[1]) {
      location.replace(`/portfolio?project=${encodeURIComponent(match[1])}`);
      return;
    }
    if (/^\/?portfolio\/?$/.test(path)) {
      location.replace(`/portfolio${location.search || ''}${location.hash || ''}`);
    }
  } catch (_) {}
})();
