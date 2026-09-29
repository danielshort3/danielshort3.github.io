'use strict';

const { loadFileSiteContent } = require('./content-model');

function loadSiteContent(root) {
  return loadFileSiteContent(root);
}

async function loadSiteContentAsync(root) {
  return loadSiteContent(root);
}

module.exports = { loadSiteContent, loadSiteContentAsync };
