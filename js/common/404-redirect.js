(() => {
  'use strict';
  try {
    if (window.location.hostname === 'danielshort3.github.io') {
      let canonicalPath = String(window.location.pathname || '/');
      canonicalPath = canonicalPath.replace(/^\/pages\//i, '/').replace(/\/index\.html$/i, '/');
      canonicalPath = canonicalPath.replace(/\.html$/i, '') || '/';
      window.location.replace(`https://www.danielshort.me${canonicalPath}${window.location.search || ''}${window.location.hash || ''}`);
      return;
    }

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

  try {
    if (window.dataLayer) {
      window.dataLayer.push({
        event: 'page_404',
        failed_url: location.href,
        referrer: document.referrer || ''
      });
    }
  } catch (_) {}
})();
