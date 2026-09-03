'use strict';

const SITE_ORIGIN = 'https://www.danielshort.me';

const PROJECT_DEMO_IDS = Object.freeze([
  'baby-names-demo',
  'chatbot-demo',
  'covid-outbreak-demo',
  'digit-generator-demo',
  'handwriting-rating-demo',
  'minesweeper-demo',
  'nonogram-demo',
  'pizza-tips-demo',
  'retail-loss-sales-demo',
  'sentence-demo',
  'shape-demo',
  'target-empty-package-demo'
]);

const PROJECT_DEMO_ID_SET = new Set(PROJECT_DEMO_IDS);

function parseProjectDemoUrl(value) {
  const raw = String(value || '').trim();
  if (!raw) return null;

  let parsed;
  try {
    parsed = new URL(raw, `${SITE_ORIGIN}/`);
  } catch (_) {
    return null;
  }
  if (parsed.origin !== SITE_ORIGIN) return null;

  const pathname = parsed.pathname.replace(/\/+$/, '');
  const match = /(?:^|\/)([a-z0-9-]+-demo)(?:\.html)?$/i.exec(pathname);
  if (!match) return null;
  const demoId = match[1].toLowerCase();
  return PROJECT_DEMO_ID_SET.has(demoId) ? { demoId, parsed } : null;
}

function getProjectDemoId(value) {
  return parseProjectDemoUrl(value)?.demoId || '';
}

function toCanonicalProjectDemoUrl(value) {
  const match = parseProjectDemoUrl(value);
  return match
    ? `/${match.demoId}${match.parsed.search}${match.parsed.hash}`
    : String(value || '').trim();
}

function toRawProjectDemoUrl(value) {
  const match = parseProjectDemoUrl(value);
  return match
    ? `/demos/${match.demoId}.html${match.parsed.search}${match.parsed.hash}`
    : String(value || '').trim();
}

module.exports = {
  PROJECT_DEMO_IDS,
  getProjectDemoId,
  toCanonicalProjectDemoUrl,
  toRawProjectDemoUrl
};
