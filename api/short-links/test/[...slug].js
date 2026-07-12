/*
  Local compatibility entrypoint for /api/short-links/test/<slug>.
  Vercel deploys this behavior through api/short-links/[...slug].js so the
  endpoint does not consume a separate function.
*/
'use strict';

module.exports = require('../../_lib/short-links-test');
