const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));
const countMatches = (value, pattern) => (String(value || '').match(pattern) || []).length;

function runResponsiveDensityContractTests({ assert }) {
  const cmsRenderers = require('../../build/lib/cms-renderers.js');
  const projectGenerator = require('../../build/generate-project-pages.js');

  const toolsPage = readJson('content/pages/tools.json');
  const tools = fs.readdirSync(path.join(ROOT, 'content', 'tools'))
    .filter((fileName) => fileName.endsWith('.json'))
    .map((fileName) => readJson(path.join('content', 'tools', fileName)));
  const toolsData = cmsRenderers.buildToolsDirectoryWorkbenchData(toolsPage, tools);
  const toolsBody = cmsRenderers.renderToolsDirectoryBody(toolsPage, tools);
  const publicToolCount = toolsData.items.filter((tool) => tool.visibility === 'public').length;

  assert(
    toolsData.filterGroups.length === 1 && toolsData.filterGroups[0].title === 'Category',
    'Tools should expose Category as its only directory filter',
  );
  assert(
    toolsBody.includes('data-portfolio-search') &&
      !toolsBody.includes('data-portfolio-sort') &&
      !toolsBody.includes('data-portfolio-inspector'),
    'Tools should keep search while omitting the sort control and inspector panel',
  );
  assert(
    countMatches(toolsBody, /<a class="portfolio-result-card tools-workbench-result tools-workbench-result__select" role="listitem"/g) === publicToolCount &&
      !toolsBody.includes('data-tool-details') &&
      !toolsBody.includes('tools-workbench-result__open'),
    'Tools should render each public utility as one full-card launch action',
  );

  const gamesPage = readJson('content/pages/games.json');
  const gamesData = cmsRenderers.buildGamesDirectoryWorkbenchData(gamesPage);
  const gamesBody = cmsRenderers.renderGamesDirectoryBody(gamesPage);
  assert(gamesData.items.length === 6, 'Games should keep the approved six playable entries');
  assert(
    gamesBody.includes('data-games-directory') &&
      gamesBody.includes('class="games-directory__grid" role="list"') &&
      countMatches(gamesBody, /<a class="games-directory-card" role="listitem"/g) === 6,
    'Games should render a simple six-card list with native launch links',
  );
  assert(
    !gamesBody.includes('data-directory-workbench') &&
      !gamesBody.includes('data-portfolio-search') &&
      !gamesBody.includes('data-portfolio-sort') &&
      !gamesBody.includes('data-portfolio-inspector') &&
      !gamesBody.includes('data-content-open'),
    'Games should not retain workbench controls, an inspector, or intercepted launch links',
  );

  const directContactHeader = cmsRenderers.renderHeader({
    settings: {},
    navigation: {
      brand: {},
      portfolio: { links: [] },
      tools: { enabled: false },
      games: { enabled: false },
      resume: { enabled: false },
      contact: { label: 'Contact', href: 'contact' },
      search: {},
    },
    projectsById: {},
    pagesById: {},
    tools: [],
    audience: { key: 'personal' },
  });
  const contactStart = directContactHeader.indexOf('class="nav-item nav-item-contact"');
  const contactEnd = directContactHeader.indexOf('class="nav-search"', contactStart);
  const contactMarkup = directContactHeader.slice(contactStart, contactEnd);
  assert(
    contactMarkup.includes('<a href="contact" class="nav-link nav-link-cta">Contact</a>') &&
      !contactMarkup.includes('nav-link-has-menu') &&
      !contactMarkup.includes('nav-dropdown-contact'),
    'Contact should be one direct navigation link without a dropdown',
  );

  const compactStaticCard = projectGenerator.renderPortfolioStaticResults([{
    id: 'responsive-contract',
    title: 'Responsive contract',
    summary: 'A concise authored outcome.',
    subtitle: 'Analysis',
    concepts: ['Responsive design', 'Accessibility'],
    tools: ['JavaScript'],
  }]);
  const staticTags = compactStaticCard.match(/<span class="portfolio-result-tags">([\s\S]*?)<\/span>\s*<\/span>/)?.[1] || '';
  assert(
    compactStaticCard.includes('class="portfolio-result-card__outcome"') &&
      countMatches(staticTags, /<span>/g) === 2 &&
      !compactStaticCard.includes('Preview summary') &&
      !compactStaticCard.includes('<button'),
    'no-JS portfolio cards should show one outcome, at most two chips, and no unavailable preview action',
  );

  const portfolioJs = read('js/portfolio/portfolio.js');
  const projectCardStart = portfolioJs.indexOf("if (!isDirectoryWorkbench) {");
  const projectCardEnd = portfolioJs.indexOf('return `\n        <button type="button"', projectCardStart);
  const projectCardTemplate = portfolioJs.slice(projectCardStart, projectCardEnd);
  assert(
    projectCardTemplate.includes('aria-label="Quick view for ${escapeHtml(project.title)}">Quick view</button>') &&
      projectCardTemplate.includes(': visibleChips, 2)') &&
      !projectCardTemplate.includes('Preview summary') &&
      !projectCardTemplate.includes('portfolio-result-card__open'),
    'hydrated portfolio cards should use one compact Quick view action and no duplicate case-study action',
  );
  assert(
    portfolioJs.includes('function hydrateSimpleGamesDirectory()') &&
      portfolioJs.includes('if (hydrateSimpleGamesDirectory()) return;'),
    'the shared portfolio controller should hydrate the simple Games directory without rebuilding it as a workbench',
  );

  const workbenchCss = read('css/components/portfolio-workbench.css');
  assert(
    /\.portfolio-result-card\s*\{[^}]*box-sizing:\s*border-box;[^}]*width:\s*100%;[^}]*min-width:\s*0;/s.test(workbenchCss) &&
      /\.portfolio-results-list\s*\{[^}]*min-width:\s*0;/s.test(workbenchCss) &&
      workbenchCss.includes('overflow-x: hidden;'),
    'portfolio results should use border-box sizing, shrinkable grid children, and horizontal overflow containment',
  );
  assert(
    workbenchCss.includes('@media (min-width: 821px) and (min-height: 650px)') &&
      !workbenchCss.includes('@media (min-width: 821px) {\n    body[data-page="portfolio"],'),
    'the fixed desktop workbench should activate only when the viewport is tall enough',
  );
  assert(
    /\.portfolio-sort-control\s*\{[^}]*box-sizing:\s*border-box;[^}]*width:\s*100%;[^}]*min-width:\s*0;[^}]*min-height:\s*44px;/s.test(workbenchCss) &&
      /\.portfolio-sort-control select,\s*\.portfolio-search input\s*\{[^}]*max-width:\s*100%;[^}]*min-width:\s*0;[^}]*min-height:\s*44px;/s.test(workbenchCss),
    'mobile workbench controls should stay inside the viewport and preserve 44px targets',
  );

  const starfallLoadingCss = read('css/games/project-starfall/loading.css');
  assert(
    /\.project-starfall-start-content\s*\{[^}]*box-sizing:\s*border-box;[^}]*width:\s*min\(520px, 100%\);[^}]*max-width:\s*100%;[^}]*min-width:\s*0;/s.test(starfallLoadingCss) &&
      /\.project-starfall-start-actions button\s*\{[^}]*min-height:\s*44px;/s.test(starfallLoadingCss),
    'Project Starfall should contain its start panel and keep its actions touch-sized',
  );

  const stellarCss = read('css/games/stellar-dogfight.css');
  assert(
    stellarCss.includes('@media (max-height: 760px)') &&
      /body\.is-hangar\s*\{[^}]*height:\s*auto;[^}]*overflow-x:\s*hidden;[^}]*overflow-y:\s*auto;/s.test(stellarCss) &&
      /body\.is-hangar \.mission-shell\s*\{[^}]*height:\s*auto;[^}]*overflow:\s*visible;/s.test(stellarCss) &&
      stellarCss.includes('@media (min-width: 769px) and (max-height: 760px)'),
    'Stellar Dogfight should switch its hangar to document scrolling on short viewports while preserving desktop nav clearance',
  );

  const gamesCss = read('css/components/games.css');
  const oceanCss = read('css/components/ocean-wave-simulation.css');
  const rouletteCss = read('css/games/roulette.css');
  const rouletteHtml = read('pages/games/roulette.html');
  assert(
    /\.games-directory-card__action\s*\{[^}]*min-height:\s*44px;/s.test(gamesCss) &&
      /\.ocean-wave-preset\s*\{[^}]*min-height:\s*44px;/s.test(oceanCss) &&
      /\.roulette00-rules-disclosure summary\s*\{[^}]*min-height:\s*44px;/s.test(rouletteCss) &&
      rouletteHtml.includes('<details class="roulette00-rules-disclosure">'),
    'the simplified Games surfaces should retain 44px actions and place secondary roulette rules in a disclosure',
  );

  const smartSentenceProject = readJson('content/projects/smartSentence.json');
  const projectPage = projectGenerator.renderProjectPage(smartSentenceProject, {
    projects: [smartSentenceProject],
    index: 0,
  });
  const projectIframe = projectPage.match(/<iframe\b[^>]*class="project-embed-frame"[^>]*>/)?.[0] || '';
  assert(
    projectPage.includes('class="project-demo-mobile-launch"') &&
      projectPage.includes('>Launch demo</a>') &&
      projectIframe.includes('data-src="sentence-demo.html"') &&
      !/\ssrc="/.test(projectIframe),
    'same-origin content demos should defer iframe loading and provide a mobile launch card',
  );
  assert(
    countMatches(projectPage, /<details\b[^>]*data-project-mobile-disclosure open>/g) >= 4 &&
      projectPage.includes('class="project-disclosure-summary"'),
    'secondary project notes, evaluation, links, and notes should render as responsive disclosures',
  );

  const commonJs = read('js/common/common.js');
  const projectCss = read('css/components/project-page.css');
  assert(
    commonJs.includes("window.matchMedia('(max-width: 768px)')") &&
      commonJs.includes("ifr.removeAttribute('src')") &&
      commonJs.includes("ifr.setAttribute('src', deferredSrc)") &&
      commonJs.includes('observeProjectEmbedIframe(ifr)') &&
      commonJs.includes("details.open = viewport === 'desktop'"),
    'project runtime should unload content iframes on mobile, restore observation on desktop, and sync disclosure state',
  );
  assert(
    /\.project-disclosure-summary\s*\{[^}]*min-height:\s*44px;/s.test(projectCss) &&
      /@media \(max-width: 768px\)[\s\S]*?\.project-disclosure-summary\s*\{[^}]*min-height:\s*48px;/s.test(projectCss) &&
      projectCss.includes('.project-demo-shell[data-demo-fit="content"] .project-demo-mobile-launch') &&
      projectCss.includes('.project-demo-shell[data-demo-fit="content"] .project-embed') &&
      projectCss.includes('max-height:min(960px, calc(100svh - var(--nav-height, 72px) - 48px));'),
    'project CSS should expose touch-sized disclosures, swap content embeds for the launch card on mobile, and cap iframe height',
  );

  const vercelConfig = readJson('vercel.json');
  ['/short-links', '/short-links.html'].forEach((source) => {
    assert(
      vercelConfig.redirects.some((redirect) => redirect.source === source &&
        redirect.destination === '/tools/short-links' && redirect.permanent === true),
      `${source} should permanently redirect to the canonical /tools/short-links route`,
    );
  });
}

module.exports = runResponsiveDensityContractTests;

if (require.main === module) {
  let checks = 0;
  const assert = (condition, message) => {
    if (!condition) throw new Error(message);
    checks += 1;
  };
  runResponsiveDensityContractTests({ assert });
  process.stdout.write(`Responsive density contracts passed (${checks} checks).\n`);
}
