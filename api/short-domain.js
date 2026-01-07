/*
  dshort.me is reserved for short link redirects only.
  Anything else should not serve the full website.
*/
'use strict';

module.exports = (req, res) => {
  res.statusCode = 404;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('X-Robots-Tag', 'noindex');
  res.end('Not Found');
};

