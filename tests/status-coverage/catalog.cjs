'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const actions = {
  'tool:aws-status': 'Load the 90-day history and catalog fixture as admin, and verify the ordinary-account gate.',
  'tool:background-remover': 'Remove a solid-color background from a synthetic image and verify transparent PNG output.',
  'tool:campaign-creative-tracker': 'Create a synthetic campaign and verify that it survives a reload.',
  'tool:ga4-utm-performance': 'Run a bounded fixture report and verify its UTM campaign and export control.',
  'tool:image-optimizer': 'Resize and download a synthetic PNG, then verify its dimensions.',
  'tool:job-application-tracker': 'Save and reload a synthetic application with the correct calendar date.',
  'tool:nbsp-cleaner': 'Clean a non-breaking space and verify the exact text and replacement count.',
  'tool:oxford-comma-checker': 'Check the built-in list example and verify a reported suggestion.',
  'tool:point-of-view-checker': 'Analyze a fixed passage and verify its first- and second-person counts.',
  'tool:qr-code-generator': 'Generate a QR code for a fixture URL and verify the downloadable image.',
  'tool:screen-recorder': 'Start and stop a synthetic capture, then verify a playable browser export.',
  'tool:short-links': 'Verify the signed-out gate, open a fixture workspace, and check its views.',
  'tool:text-compare': 'Compare two fixed texts and verify the rendered addition and removal.',
  'tool:transcribe': 'Select synthetic audio and verify that it is queued in the browser.',
  'tool:utm-batch-builder': 'Generate one tagged URL and verify its fields and export control.',
  'tool:word-frequency': 'Analyze repeated words and verify the selected word count.',
  'project:androidApp': 'Load the Android review release feed and verify the published download metadata.',
  'project:babynames': 'Load the bundled data and verify name recommendations change with the selected group.',
  'project:chatbotLora': 'Submit a synthetic prompt and verify the completed streamed fixture answer.',
  'project:covidAnalysis': 'Select a state and verify the historical risk display from bundled data.',
  'project:deliveryTip': 'Follow the project resource links and verify the local PDF and workbook packages.',
  'project:digitGenerator': 'Generate a digit from a fixture response and verify a decodable result image.',
  'project:handwritingRating': 'Draw a sample and verify the rendered score from a fixture response.',
  'project:nonogram': 'Solve the built-in puzzle and verify a returned grid satisfies the clues.',
  'project:pizza': 'Change the order cost and verify the bundled model returns an updated estimate.',
  'project:pizzaDashboard': 'Open the Tableau frame and use a filter in a controlled provider fixture.',
  'project:retailStore': 'Change the sales metric and verify the bundled chart data changes.',
  'project:shapeClassifier': 'Draw a sample and verify the rendered fixture prediction.',
  'project:sheetMusicUpscale': 'Follow the report link and verify the local archive contains the expected sample reports.',
  'project:smartSentence': 'Search a known phrase and verify the rendered ranked fixture results.',
  'project:targetEmptyPackage': 'Change a location filter and verify the bundled dashboard totals update.',
  'project:ufoDashboard': 'Open the Tableau frame and use a filter in a controlled provider fixture.',
  'project:website': 'Navigate Home to Tools and verify search returns a known tool.',
  'game:project-starfall': 'Start a fighter run and verify the loaded runtime responds to movement.',
  'game:stellar-dogfight': 'Launch a campaign and verify the active flight responds to controls.',
  'game:roulette': 'Place a synthetic wager, spin, and verify the displayed settlement.',
  'game:probability-engine': 'Spin a seeded scenario and verify the displayed result.',
  'game:stormbreak': 'Earn a synthetic battle reward and verify the saved progress after reload.',
  'game:ocean-wave-simulation': 'Change the simulation location and verify the renderer reports the new scene.',
  'game:chasing-light': 'Fire a pulse in the guided mission and verify the displayed speed measurement.',
  'game:wayfarers-guild': 'Acknowledge the first guild tutorial lesson and verify the next step survives a reload.'
};

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function readDirectory(root, directory) {
  const location = path.join(root, directory);
  return fs.readdirSync(location).filter((name) => name.endsWith('.json')).sort()
    .map((name) => readJson(path.join(location, name)));
}

function readCatalog(root) {
  const tools = readDirectory(root, 'content/tools').map(({ slug, href, title }) => ({ category: 'tool', slug, href, title }));
  const projects = readDirectory(root, 'content/projects')
    .filter((project) => project.published !== false && !project.hidden && project.visibility !== 'unlisted')
    .map(({ id, title, embed }) => ({ category: 'project', slug: id, href: 'portfolio/' + id, title, embed }));
  const games = readJson(path.join(root, 'content/pages/games.json')).games
    .filter((game) => game.published !== false && !game.hidden)
    .map(({ id, href, title }) => ({ category: 'game', slug: id, href, title }));
  const seen = new Set();
  return [...tools, ...projects, ...games].map(({ category, slug, href, title, embed }) => {
    assert(/^[A-Za-z0-9][A-Za-z0-9-]{0,70}$/.test(slug) && typeof title === 'string' && title.length <= 150 &&
      typeof href === 'string' && /^\/?(?:tools|portfolio|games)\/[A-Za-z0-9][A-Za-z0-9-]*$/.test(href), 'Invalid catalog entry');
    const id = category + ':' + slug;
    assert(!seen.has(id), 'Duplicate coverage check ID');
    seen.add(id);
    return {
      id, category, slug, route: '/' + href.replace(/^\//, ''), title, embed,
      action: actions[id] || 'Exercise and verify a representative interaction for this catalog item.'
    };
  });
}

module.exports = { readCatalog };
