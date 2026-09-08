'use strict';

const fs = require('fs');
const path = require('path');
const { preparePersonalGameDetailHtml } = require('../../build/lib/personal-game-header');
const { GAME_PAGE_PATHS } = require('../../build/generate-personal-accordion-pages');
const { unwrapPersonalAccordionHtml, wrapPersonalAccordionHtml } = require('../../build/lib/personal-accordion-shell');

const ROOT = path.resolve(__dirname, '../..');
const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const gameMetadata = JSON.parse(read('content/pages/games.json')).games;
const documentHtml = (content) => `<html><head><base href="/"><title>Game</title></head><body><header id="combined-header-nav">DS Search Breadcrumbs</header>${content}<footer>Footer</footer><script src="game.js"></script></body></html>`;
const fixtures = {
  'project-starfall': documentHtml('<main id="main"><section class="project-starfall-masthead" aria-labelledby="project-starfall-title"><div class="project-starfall-masthead__inner"><h1 id="project-starfall-title">Project Starfall</h1><p class="project-starfall-masthead__mission">Recover the fallen beacon.</p></div></section><section class="project-starfall-shell" data-starfall-root><canvas id="project-starfall-canvas"></canvas><button data-starfall-action="load">Start</button><div data-starfall-loader role="status"></div></section></main>'),
  'stellar-dogfight': documentHtml('<main id="main" class="mission-shell"><header class="mission-header mission-topbar"><div class="mission-title-compact"><p class="eyebrow">Space blaster</p><h1>Stellar Dogfight</h1></div><p class="mission-topbar-summary" data-role="topbar-summary">Campaign · Rank 1</p><div class="mission-topbar-actions"><a class="btn ghost" href="games">All Games</a><button class="btn primary" data-action="launch">Play Campaign</button><button type="button" data-action="command-menu" aria-controls="command-menu">Menu</button></div><p class="mission-guidance" data-role="launch-guidance">Start a campaign run.</p></header><section class="mission-grid"><canvas id="game-canvas"></canvas><aside id="command-menu"></aside></section></main>'),
  roulette: documentHtml('<main id="main" class="roulette00-shell"><header class="roulette00-header"><div><p class="roulette00-eyebrow">Classic American layout</p><h1>Double-Zero Roulette</h1><p class="roulette00-lead">Place chips and spin the wheel.</p><p class="roulette00-disclaimer">Virtual credits with no cash value.</p></div><div class="roulette00-header-metrics" aria-live="polite"><div><span>Bankroll</span><strong id="roulette-bankroll">$2,000</strong></div><div><span>Total</span><strong id="roulette-total-bet">$0</strong></div></div></header><section class="roulette00-grid"><button id="roulette-spin">Spin wheel</button></section></main><div id="roulette-modal"></div>'),
  'probability-engine': documentHtml('<div class="app-shell"><header class="topbar"><div class="engine-brand"><h1>Probability Engine</h1><p>Build reels and bend odds.</p></div><div class="save-tools" aria-label="Save management"><button id="export-save-button">Export Save</button><button id="import-save-button">Import Save</button><button id="reset-save-button" class="danger-button">Reset Progress</button><input id="import-save-input" type="file" hidden><p id="save-status" role="status"></p></div><div id="stats" class="stats"></div></header><main id="main"><nav><button id="tab-machine">Machine</button></nav></main><div id="offline-modal"></div></div>'),
  stormbreak: documentHtml('<main id="main" class="stormbreak-main"><section class="stormbreak-game" data-stormbreak-root aria-labelledby="stormbreak-title"><div class="stormbreak-topbar"><div class="stormbreak-title-block"><h1 id="stormbreak-title">Stormbreak</h1><p>Idle Olympus</p></div><div class="stormbreak-mission"><span data-bind="wave">1</span></div><div class="stormbreak-player"><span data-bind="level">1</span></div></div><canvas id="stormbreak-canvas"></canvas><button data-action="pause">Pause</button><button data-action="reset">Reset</button></section></main>'),
  'ocean-wave-simulation': documentHtml('<section class="hero hero--games ocean-wave-hero"><div class="wrapper"><p class="hero-eyebrow">Ocean conditions</p><h1>Ocean Wave Simulation</h1><p class="ocean-wave-lead">Explore wind and swell.</p></div></section><main id="main"><canvas id="ocean-wave-canvas"></canvas><button id="ocean-wave-toggle">Pause</button></main>')
};

const tags = (html, pattern) => (html.match(pattern) || []).slice().sort();
const optionsFor = (itemId) => ({
  category: 'games', itemId, view: 'detail', chrome: 'compact', backHref: '/games',
  includePageHero: itemId === 'ocean-wave-simulation',
  includeProbabilityShell: itemId === 'probability-engine',
  includeUntilScripts: ['probability-engine', 'roulette'].includes(itemId)
});

function runPersonalGameHeaderTests({ assert }) {
  for (const [itemId, fixture] of Object.entries(fixtures)) {
    const metadata = { ...gameMetadata.find((game) => game.id === itemId), itemId };
    const prepared = preparePersonalGameDetailHtml(fixture, metadata);
    const header = prepared.match(/<(?:header|section) class="personal-game-header[^>]*>[\s\S]*?<\/(?:header|section)>/)[0];
    assert((prepared.match(/\sdata-page-masthead(?=\s|>)/g) || []).length === 1,
      `${itemId} gets exactly one shared game masthead.`);
    assert(header.includes('href="/games" data-page-masthead-parent') && header.includes('<span>Game library</span>')
      && header.includes('data-page-masthead-copy') && header.includes('data-page-masthead-intro'),
    `${itemId} places its library parent and introduction above the shared divider.`);
    assert((prepared.match(/<h1\b/g) || []).length === 1,
      `${itemId} moves its authored title without duplicating the page heading.`);
    assert(prepared.includes('<header id="combined-header-nav">DS Search Breadcrumbs</header>'),
      `${itemId} preserves the separate site logo, search, and breadcrumb header.`);
    assert(JSON.stringify(tags(prepared, /<button\b[^>]*>[\s\S]*?<\/button>/g)) === JSON.stringify(tags(fixture, /<button\b[^>]*>[\s\S]*?<\/button>/g))
      && JSON.stringify(tags(prepared, /\bid="[^"]+"/g)) === JSON.stringify(tags(fixture, /\bid="[^"]+"/g))
      && JSON.stringify(tags(prepared, /<script\b[^>]*>[\s\S]*?<\/script>/g)) === JSON.stringify(tags(fixture, /<script\b[^>]*>[\s\S]*?<\/script>/g)),
    `${itemId} retains every actual button, element ID, and runtime script exactly once.`);
    assert(preparePersonalGameDetailHtml(prepared, metadata) === prepared,
      `${itemId} preparation is idempotent.`);
    const wrapped = wrapPersonalAccordionHtml(prepared, optionsFor(itemId));
    assert(!wrapped.includes('data-site-route-toolbar') && wrapped.includes(`data-personal-game-header="${itemId}"`),
      `${itemId} keeps the new masthead inside the vertical-tab frame and removes duplicate library navigation.`);
    assert(wrapPersonalAccordionHtml(preparePersonalGameDetailHtml(wrapped, metadata), optionsFor(itemId)) === wrapped,
      `${itemId} rewrapping does not duplicate its header, controls, or dialogs.`);

    const liveSource = unwrapPersonalAccordionHtml(read(GAME_PAGE_PATHS[itemId]));
    const livePrepared = preparePersonalGameDetailHtml(liveSource, metadata);
    assert(JSON.stringify(tags(livePrepared, /<button\b[^>]*>[\s\S]*?<\/button>/g)) === JSON.stringify(tags(liveSource, /<button\b[^>]*>[\s\S]*?<\/button>/g))
      && JSON.stringify(tags(livePrepared, /\bid="[^"]+"/g)) === JSON.stringify(tags(liveSource, /\bid="[^"]+"/g)),
    `${itemId} preserves all real runtime controls and IDs, including the full game interior.`);

    if (itemId === 'project-starfall') {
      assert(header.includes('id="project-starfall-title"') && header.includes('work-in-progress')
        && prepared.includes('data-starfall-root') && prepared.includes('data-starfall-action="load"'),
      'Starfall introduces its work-in-progress status while retaining the actual game start controls.');
    } else if (itemId === 'stellar-dogfight') {
      assert(header.includes('data-action="launch"') && header.includes('data-action="command-menu"') && !header.includes('All Games'),
        'Stellar keeps its actual launch and menu buttons above the divider with one canonical game-library link.');
      assert(prepared.indexOf('mission-session-status') > prepared.indexOf('</header>', prepared.indexOf('data-personal-game-header')),
        'Campaign state remains below the common page introduction.');
    } else if (itemId === 'probability-engine') {
      assert(header.includes('data-page-masthead-actions') && header.includes('id="import-save-input"') && header.includes('id="save-status"')
        && !header.includes('id="stats"'),
      'Probability preserves save/import feedback in its action group while game statistics follow the divider.');
    } else if (itemId === 'stormbreak') {
      const gameRoot = prepared.slice(prepared.indexOf('<section class="stormbreak-game"'));
      assert(!gameRoot.includes('stormbreak-title-block') && gameRoot.includes('data-action="pause"') && gameRoot.includes('data-action="reset"')
        && header.includes('id="stormbreak-title"'),
      'Stormbreak avoids a duplicate title while all delegated gameplay controls remain under the actual game root.');
    } else if (itemId === 'roulette') {
      assert(!header.includes('roulette00-header-metrics') && prepared.includes('Virtual credits with no cash value.')
        && prepared.indexOf('roulette00-session-status') < prepared.indexOf('roulette00-grid'),
      'Roulette retains its virtual-credit context and live bankroll beside the game, below the common title.');
    } else {
      assert(prepared.indexOf('data-personal-game-header') < prepared.indexOf('<main'),
        'The Ocean masthead keeps the existing pre-main hero extraction boundary.');
    }
  }
  assert(Object.hasOwn(GAME_PAGE_PATHS, 'project-starfall'),
    'The enabled Project Starfall route participates in game header standardization.');
}

module.exports = runPersonalGameHeaderTests;

if (require.main === module) {
  let checks = 0;
  runPersonalGameHeaderTests({ assert(condition, message) {
    if (!condition) throw new Error(message);
    checks += 1;
  } });
  process.stdout.write(`Personal game header tests passed (${checks} checks).\n`);
}
