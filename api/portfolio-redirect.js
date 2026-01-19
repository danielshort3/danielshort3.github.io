/*
  Redirect legacy portfolio deep links like:
  /portfolio?project=<id> -> /portfolio/<id>
*/
'use strict';

function normalizeProjectId(value) {
  const raw = Array.isArray(value) ? value[0] : value;
  const id = typeof raw === 'string' ? raw.trim() : '';
  if (!id) return '';
  if (!/^[A-Za-z0-9_-]+$/.test(id)) return '';
  return id;
}

module.exports = (req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.statusCode = 405;
    res.setHeader('Allow', 'GET, HEAD');
    res.end('Method Not Allowed');
    return;
  }

  const project = normalizeProjectId(req.query && req.query.project);
  if (!project) {
    res.statusCode = 404;
    res.setHeader('Cache-Control', 'no-store');
    res.end('Not Found');
    return;
  }

  res.statusCode = 308;
  res.setHeader('Location', `/portfolio/${encodeURIComponent(project)}`);
  res.setHeader('Cache-Control', 'public, max-age=600');
  res.end();
};

