'use strict';

const { unwrapPersonalAccordionHtml } = require('./personal-accordion-shell');

function escapeHtml(value) {
  return String(value || '').replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[character]));
}

function requireMatch(html, pattern, description) {
  const match = pattern.exec(html);
  if (!match) throw new Error(`Cannot prepare personal game header: missing ${description}.`);
  return match;
}

function renderGameHeader({ itemId, copy, actions = '', classes = '', tag = 'header' }) {
  return [
    `<${tag} class="personal-game-header${classes ? ` ${classes}` : ''}" data-page-masthead data-personal-game-header="${escapeHtml(itemId)}">`,
    '  <a href="/games" data-page-masthead-parent aria-label="Back to game library">',
    '    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 12H5m7 7-7-7 7-7"></path></svg>',
    '    <span>Game library</span>',
    '  </a>',
    '  <div data-page-masthead-intro>',
    '    <div data-page-masthead-copy>',
    copy,
    '    </div>',
    actions,
    '  </div>',
    `</${tag}>`
  ].filter(Boolean).join('\n');
}

function prepareStarfall(html, metadata) {
  const header = requireMatch(html, /<section\b[^>]*class="[^"]*\bproject-starfall-masthead\b[^"]*"[^>]*>[\s\S]*?<\/section>/i, 'Project Starfall header')[0];
  const title = requireMatch(header, /<h1\b[^>]*>[\s\S]*?<\/h1>/i, 'Project Starfall title')[0];
  if (!metadata.summary) throw new Error('Project Starfall needs its authored game summary.');
  return html.replace(header, renderGameHeader({
    itemId: metadata.itemId,
    copy: `${title}\n<p>${escapeHtml(metadata.summary)}</p>`
  }));
}

function prepareStellar(html, metadata) {
  const header = requireMatch(html, /<header\b[^>]*class="[^"]*\bmission-header\b[^"]*"[^>]*>[\s\S]*?<\/header>/i, 'Stellar Dogfight header')[0];
  const title = requireMatch(header, /<h1\b[^>]*>[\s\S]*?<\/h1>/i, 'Stellar Dogfight title')[0];
  const actions = requireMatch(header, /<div class="mission-topbar-actions">([\s\S]*?)<\/div>/i, 'Stellar Dogfight actions')[1]
    .replace(/\s*<a\b[^>]*>[\s\S]*?<\/a>/gi, '');
  const status = requireMatch(header, /<p\b[^>]*data-role="topbar-summary"[^>]*>[\s\S]*?<\/p>/i, 'Stellar Dogfight session status')[0];
  const guidance = requireMatch(header, /<p\b[^>]*data-role="launch-guidance"[^>]*>[\s\S]*?<\/p>/i, 'Stellar Dogfight launch guidance')[0];
  if (!metadata.summary) throw new Error('Stellar Dogfight needs its authored game summary.');
  const masthead = renderGameHeader({
    itemId: metadata.itemId,
    classes: 'mission-header',
    copy: `${title}\n<p>${escapeHtml(metadata.summary)}</p>`,
    actions: `<div class="game-header-actions" data-page-masthead-actions aria-label="Game actions">${actions}\n</div>`
  });
  return html.replace(header, `${masthead}\n<div class="mission-session-status" aria-label="Campaign status">\n${status}\n${guidance}\n</div>`);
}

function prepareRoulette(html, metadata) {
  const header = requireMatch(html, /<header class="roulette00-header">([\s\S]*?)<\/header>/i, 'Roulette header');
  const title = requireMatch(header[1], /<h1\b[^>]*>[\s\S]*?<\/h1>/i, 'Roulette title')[0];
  const lead = requireMatch(header[1], /<p class="roulette00-lead">[\s\S]*?<\/p>/i, 'Roulette introduction')[0];
  const disclaimer = requireMatch(header[1], /<p class="roulette00-disclaimer">[\s\S]*?<\/p>/i, 'Roulette virtual-credit explanation')[0];
  const metricsStart = header[1].indexOf('<div class="roulette00-header-metrics"');
  if (metricsStart === -1) throw new Error('Cannot prepare personal game header: missing Roulette metrics.');
  const metrics = header[1].slice(metricsStart).trim();
  const masthead = renderGameHeader({ itemId: metadata.itemId, copy: `${title}\n${lead}` });
  // Bankroll and bets belong beside the table, below the shared introduction.
  return html.replace(header[0], `${masthead}\n<div class="roulette00-header roulette00-session-status">\n${disclaimer}\n${metrics}\n</div>`);
}

function prepareProbability(html, metadata) {
  const header = requireMatch(html, /<header class="topbar">([\s\S]*?)<\/header>/i, 'Probability Engine header');
  const brand = requireMatch(header[1], /<div class="engine-brand">([\s\S]*?)<\/div>/i, 'Probability Engine introduction')[1];
  const actions = requireMatch(header[1], /<div class="save-tools"[^>]*>[\s\S]*?<\/div>/i, 'Probability Engine save controls')[0]
    .replace('<div class="save-tools"', '<div class="save-tools game-header-actions" data-page-masthead-actions');
  const stats = requireMatch(header[1], /<div id="stats" class="stats">[\s\S]*?<\/div>/i, 'Probability Engine statistics')[0];
  const masthead = renderGameHeader({ itemId: metadata.itemId, copy: brand.trim(), actions });
  return html.replace(header[0], `${masthead}\n<div class="topbar probability-session-status" aria-label="Game statistics">\n${stats}\n</div>`);
}

function prepareStormbreak(html, metadata) {
  const titleBlock = requireMatch(html, /<div class="stormbreak-title-block">[\s\S]*?<\/div>/i, 'Stormbreak title block')[0];
  const title = requireMatch(titleBlock, /<h1\b[^>]*>[\s\S]*?<\/h1>/i, 'Stormbreak title')[0];
  if (!metadata.title || !metadata.summary) throw new Error('Stormbreak needs its authored game title and summary.');
  const masthead = renderGameHeader({
    itemId: metadata.itemId,
    copy: `${title.replace(/>[^<]*<\/h1>/i, `>${escapeHtml(metadata.title)}</h1>`)}\n<p>${escapeHtml(metadata.summary)}</p>`
  });
  // Keep every control under the actual game root: its runtime delegates there.
  return html.replace(titleBlock, '').replace(/(<main\b[^>]*\bid="main"[^>]*>)/i, `$1\n${masthead}`);
}

function prepareOcean(html, metadata) {
  const hero = requireMatch(html, /<section\b[^>]*class="[^"]*\bocean-wave-hero\b[^"]*"[^>]*>[\s\S]*?<\/section>/i, 'Ocean Wave Simulation introduction')[0];
  const title = requireMatch(hero, /<h1\b[^>]*>[\s\S]*?<\/h1>/i, 'Ocean Wave Simulation title')[0];
  const lead = requireMatch(hero, /<p class="ocean-wave-lead">[\s\S]*?<\/p>/i, 'Ocean Wave Simulation description')[0];
  return html.replace(hero, renderGameHeader({
    itemId: metadata.itemId,
    tag: 'section',
    // The extraction boundary intentionally continues to include the pre-main hero.
    classes: 'hero hero--games ocean-wave-hero',
    copy: `${title}\n${lead}`
  }));
}

const PREPARERS = Object.freeze({
  'project-starfall': prepareStarfall,
  'stellar-dogfight': prepareStellar,
  roulette: prepareRoulette,
  'probability-engine': prepareProbability,
  stormbreak: prepareStormbreak,
  'ocean-wave-simulation': prepareOcean
});

function preparePersonalGameDetailHtml(html, metadata = {}) {
  const source = unwrapPersonalAccordionHtml(html);
  const itemId = String(metadata.itemId || '').trim();
  const prepare = PREPARERS[itemId];
  if (!prepare) throw new Error(`Unknown personal game header: ${itemId || '(missing id)'}.`);
  if (source.includes(`data-personal-game-header="${itemId}"`)) return source;
  return prepare(source, { ...metadata, itemId });
}

module.exports = { preparePersonalGameDetailHtml };
