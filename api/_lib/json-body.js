/* Reuse the tools transport's dependency-free, bounded UTF-8 JSON reader.
   Keep one implementation and the existing BODY_TOO_LARGE / HTTP 413 contract. */
'use strict';

const { readJson } = require('./tools-api');
module.exports = { readJson };
