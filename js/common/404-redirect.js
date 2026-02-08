(() => {
  'use strict';
  try {
    const path = String(location.pathname || '');
    const params = new URLSearchParams(location.search || '');
    const normalizeProject = (value) => String(value || '').trim().replace(/^\/+|\/+$/g, '');
    const project = normalizeProject(params.get('project'));

    if (project) {
      params.delete('project');
      const rest = params.toString();
      location.replace(`/portfolio/${encodeURIComponent(project)}${rest ? `?${rest}` : ''}${location.hash || ''}`);
      return;
    }

    const projectHtmlMatch = path.match(/^\/portfolio\/([A-Za-z0-9_-]+)\.html\/?$/i);
    if (projectHtmlMatch && projectHtmlMatch[1]) {
      location.replace(`/portfolio/${encodeURIComponent(projectHtmlMatch[1])}${location.search || ''}${location.hash || ''}`);
      return;
    }

    const projectPageMatch = path.match(/^\/pages\/portfolio\/([A-Za-z0-9_-]+)(?:\.html)?\/?$/i);
    if (projectPageMatch && projectPageMatch[1]) {
      location.replace(`/portfolio/${encodeURIComponent(projectPageMatch[1])}${location.search || ''}${location.hash || ''}`);
      return;
    }

    if (/^\/portfolio\.html\/?$/i.test(path) || /^\/pages\/portfolio(?:\.html)?\/?$/i.test(path)) {
      location.replace(`/portfolio${location.search || ''}${location.hash || ''}`);
    }
  } catch (_) {}
})();
