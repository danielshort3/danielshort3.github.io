(() => {
  'use strict';
  try {
    const path = String(location.pathname || '');
    const params = new URLSearchParams(location.search || '');
    const project = params.get('project');
    if (project) {
      location.replace(`/portfolio/${encodeURIComponent(project)}`);
      return;
    }
    const match = path.match(/(?:^|\/)portfolio\/([A-Za-z0-9_-]+)\/?$/);
    if (match && match[1]) {
      location.replace(`/pages/portfolio/${encodeURIComponent(match[1])}.html`);
      return;
    }
    if (/^\/?portfolio\/?$/.test(path)) {
      location.replace(`/pages/portfolio.html${location.search || ''}${location.hash || ''}`);
    }
  } catch (_) {}
})();
