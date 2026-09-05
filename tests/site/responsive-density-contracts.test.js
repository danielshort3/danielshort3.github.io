const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));
const countMatches = (value, pattern) => (String(value || '').match(pattern) || []).length;

function runResponsiveDensityContractTests({ assert }) {
  const cmsRenderers = require('../../build/lib/cms-renderers.js');
  const projectGenerator = require('../../build/generate-project-pages.js');
  const personalPageGenerator = require('../../build/generate-personal-accordion-pages.js');
  const personalShell = require('../../build/lib/personal-accordion-shell.js');

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
  assert(
    personalPageGenerator.INTERNAL_TOOL_PAGE_IDS.every((itemId) => (
      !toolsBody.includes(`data-project-id="${itemId}"`)
    )),
    'Internal and account-reachable tools should receive the shared shell without appearing in the public directory',
  );

  const gamesPage = readJson('content/pages/games.json');
  const gamesData = cmsRenderers.buildGamesDirectoryWorkbenchData(gamesPage);
  const gamesBody = cmsRenderers.renderGamesDirectoryBody(gamesPage);
  assert(
    gamesData.items.length === 5 &&
      !gamesData.items.some((game) => game.id === 'project-starfall') &&
      Object.keys(personalPageGenerator.GAME_PAGE_PATHS).length === 5 &&
      !Object.prototype.hasOwnProperty.call(personalPageGenerator.GAME_PAGE_PATHS, 'project-starfall'),
    'Games should expose five public entries and keep Project Starfall out of generated routing',
  );
  assert(
    gamesBody.includes('data-games-directory') &&
      gamesBody.includes('class="games-directory__grid" role="list"') &&
      countMatches(gamesBody, /<a class="games-directory-card" role="listitem"/g) === 5 &&
      !gamesBody.includes('project-starfall'),
    'Games should render a simple five-card list with native launch links',
  );
  assert(
    !gamesBody.includes('data-directory-workbench') &&
      !gamesBody.includes('data-portfolio-search') &&
      !gamesBody.includes('data-portfolio-sort') &&
      !gamesBody.includes('data-portfolio-inspector') &&
      !gamesBody.includes('data-content-open'),
    'Games should not retain workbench controls, an inspector, or intercepted launch links',
  );

  const shellSample = [
    '<!doctype html>',
    '<html><head></head><body><main id="main"><h1>Games</h1></main></body></html>',
  ].join('');
  const gamesLibraryShell = personalPageGenerator.buildLibraryPage(shellSample, 'games', {
    games: {
      items: gamesData.items.map((game) => ({
        id: game.id,
        title: game.title,
        href: game.href,
      })),
    },
  });
  const gameDetailShell = personalShell.wrapPersonalAccordionHtml(shellSample, {
    category: 'games',
    itemId: 'probability-engine',
    view: 'detail',
    fit: 'viewport',
    chrome: 'compact',
    backHref: '/games',
    backLabel: 'Back to game library',
    backCompactLabel: 'Library',
    backAriaLabel: 'Back to game library',
  });
  [gamesLibraryShell, gameDetailShell].forEach((html) => {
    assert(
      countMatches(html, /class="personal-accordion__rail(?:\s|")/g) === 5 &&
        countMatches(html, /data-site-tab="[^"]+"[^>]*hidden inert aria-hidden="true"/g) === 4 &&
        countMatches(html, /class="personal-accordion__toolbar(?:\s|")/g) === 1 &&
        !/class="personal-accordion__rails"[^>]*aria-hidden=/i.test(html) &&
        /<a\b[^>]*class="[^"]*\bpersonal-accordion__rail\b[^>]*href="\/#games"[^>]*aria-current="page"/i.test(html) &&
        html.includes('data-personal-transition="collapse"'),
      'canonical personal shells should preserve five category rails while exposing only one active rail and one shared context toolbar',
    );
  });
  assert(
    gamesLibraryShell.includes('href="/#games" aria-label="Back to homepage"') &&
      gamesLibraryShell.includes('personal-accordion__back-label--mobile" aria-hidden="true">Home</span>') &&
      gameDetailShell.includes('href="/games" aria-label="Back to game library"') &&
      gameDetailShell.includes('personal-accordion__back-label--mobile" aria-hidden="true">Library</span>'),
    'canonical Games shells should return through the homepage or game library without cross-category navigation',
  );
  const personalGeneratorSource = read('build/generate-personal-accordion-pages.js');
  assert(
    personalGeneratorSource.includes("backHref: '/portfolio'") &&
      personalGeneratorSource.includes("backHref: '/tools'") &&
      personalGeneratorSource.includes("backHref: '/games'") &&
      !personalGeneratorSource.includes("backHref: '/?view=library#projects'"),
    'generated details should use their canonical project, tool, and game libraries',
  );
  const personalShellCss = read('css/components/personal-accordion-shell.css');
  const utilityLayoutCss = read('css/utilities/layout.css');
  assert(
    /@media \(pointer:\s*fine\)\s*\{\s*html:has\(body\.personal-accordion-page\)::-webkit-scrollbar,/s.test(personalShellCss) &&
      /@media \(pointer:\s*fine\)\s*\{\s*\.contact-page::-webkit-scrollbar\s*\{/s.test(utilityLayoutCss),
    'custom root WebKit scrollbars must remain limited to fine pointers so touch viewports retain their full visible width',
  );
  assert(
    personalShellCss.includes('--personal-rail-size: 68px;') &&
      personalShellCss.includes('--personal-mobile-rail-size: 48px;') &&
      personalShellCss.includes('--personal-toolbar-size: 60px;') &&
      !personalShellCss.includes('--personal-toolbar-size: 48px;') &&
      personalShellCss.includes('grid-template-rows: var(--personal-mobile-rail-size) minmax(0, 1fr);') &&
      personalShellCss.includes('.personal-accordion__rail[hidden] {\n    display: none !important;') &&
      personalShellCss.includes('writing-mode: horizontal-tb;'),
    'responsive personal shells should use a 68px desktop marker and one visible 48px horizontal return rail on mobile',
  );
  const homeAccordionCss = read('css/components/home-category-accordion.css');
  assert(
    homeAccordionCss.includes('.home-accordion__item.is-active .home-accordion__rail {\n      width: 100%;\n      height: 54px;'),
    'the active mobile homepage rail should override the desktop rail width and fill the viewport',
  );
  const homeCtaRules = Array.from(homeAccordionCss.matchAll(/\.home-accordion__panel-cta\s*\{([^}]+)\}/g), (match) => match[1]);
  const homeCtaBase = homeCtaRules[0];
  const homeCtaMobile = homeCtaRules.find((rule) => /width:\s*calc\(100% - \d+px\)/.test(rule));
  const ctaWidthDeduction = Number(homeCtaMobile?.match(/width:\s*calc\(100% - (\d+)px\)/)?.[1]);
  const ctaMargin = Number(homeCtaMobile?.match(/margin:\s*\d+px (\d+)px/)?.[1]);
  const ctaPadding = Number(homeCtaBase?.match(/padding:\s*0 (\d+)px/)?.[1]);
  const ctaBorder = Number(homeCtaBase?.match(/border:\s*(\d+)px/)?.[1]);
  const ctaIsBorderBox = /box-sizing:\s*border-box;/.test(homeCtaBase);
  [297, 312, 367, 382].forEach((containerWidth) => {
    const outerWidth = containerWidth - ctaWidthDeduction + ctaMargin * 2 +
      (ctaIsBorderBox ? 0 : (ctaPadding + ctaBorder) * 2);
    assert(
      Number.isFinite(outerWidth) && outerWidth <= containerWidth &&
        /min-height:\s*(?:4[4-9]|[5-9]\d)px;/.test(homeCtaBase),
      `the homepage overview CTA should fit a ${containerWidth}px panel including padding, borders and margins, even when a scrollbar reduces the viewport`,
    );
  });
  const personalMobileCss = personalShellCss.slice(personalShellCss.indexOf('@media (max-width: 959px), (max-height: 619px)'));
  const contextRule = personalMobileCss.match(/\.personal-accordion__context\s*\{([^}]+)\}/)?.[1] || '';
  assert(
    /\.personal-accordion__toolbar\s*\{[^}]*flex-wrap:\s*wrap;[^}]*gap:\s*8px 12px;/s.test(personalMobileCss) &&
      !/position:\s*absolute|translate\(/.test(contextRule) &&
      /max-width:\s*100%;/.test(contextRule) &&
      /\.personal-accordion__back\s*\{[^}]*min-height:\s*44px;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;/s.test(personalShellCss),
    'mobile Back and context must share wrapping layout space so enlarged labels cannot overlap an absolutely centered title',
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
  assert(directContactHeader.includes('class="nav-search"') && !directContactHeader.includes('nav-item-contact') && !directContactHeader.includes('nav-dropdown'),
    'the shared masthead should use brand and search while category tabs provide navigation');

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
  assert(
    portfolioJs.includes("const PORTFOLIO_COMPACT_QUERY = '(max-width: 820px), (max-height: 480px) and (pointer: coarse)'") &&
      (portfolioJs.match(/window\.matchMedia\(PORTFOLIO_COMPACT_QUERY\)/g) || []).length >= 2 &&
      workbenchCss.includes('@media (max-width: 820px), (max-height: 480px) and (pointer: coarse)') &&
      portfolioJs.includes('aria-haspopup="dialog"') &&
      portfolioJs.includes("document.body?.classList.toggle('portfolio-inspector-open', active)") &&
      portfolioJs.includes('createBackgroundIsolation(inspector, [inspectorBackdrop])'),
    'filter and Quick view dialogs should share one compact breakpoint for portrait and short touch landscape and isolate their background',
  );

  const baseCss = read('css/base/base.css');
  const mobileDockCss = read('css/components/mobile-site-dock.css');
  const designSystemCss = read('css/utilities/design-system-overrides.css');
  const recruiterStoryCss = read('css/components/recruiter-story.css');
  const contactCardCss = read('css/components/contact-card.css');
  const shortLinksCss = read('css/components/short-links.css');
  const stormbreakCss = read('css/games/stormbreak.css');
  const toolsWorkspaceCss = read('css/components/tools-workspace.css');
  const toolsAccountCss = read('css/components/tools-account.css');
  const personalAccordionCss = read('css/components/personal-accordion-shell.css');
  const utmBatchBuilderCss = read('css/components/utm-batch-builder.css');
  const toolsMobileCss = toolsWorkspaceCss.slice(
    toolsWorkspaceCss.indexOf('@media (max-width: 959px), (max-height: 619px)'),
  );
  const utmMobileCss = utmBatchBuilderCss.slice(
    utmBatchBuilderCss.indexOf('@media (max-width: 600px)'),
  );
  const accountCompactCss = toolsAccountCss.slice(toolsAccountCss.indexOf('@media (max-width:959px), (max-height:619px)'));
  assert(
    /\.tools-account-disclosure\s*\{[^}]*box-sizing:\s*border-box;/s.test(toolsAccountCss) &&
      /\.tools-account-disclosure\s*\{[^}]*inset-inline-start:\s*auto;[^}]*inset-inline-end:\s*0;/s.test(accountCompactCss) &&
      /:is\(\.personal-tool-header__account,\.personal-library__account\) \.tools-account-disclosure\s*\{[^}]*inset-inline-start:\s*auto;[^}]*inset-inline-end:\s*0;/s.test(toolsAccountCss),
    'right-side account menus should open inward across compact widths, including 621–959px and short landscape viewports',
  );
  assert(
    /\.tools-account-structure\s*\{[^}]*display:\s*flex;[^}]*flex-wrap:\s*wrap;/s.test(accountCompactCss) &&
      /\.tools-account-actions\s*\{[^}]*flex-wrap:\s*wrap;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;/s.test(toolsAccountCss) &&
      /\.tools-account-trigger\s*\{[^}]*min-block-size:\s*44px;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;[^}]*white-space:\s*normal;[^}]*overflow-wrap:\s*anywhere;/s.test(toolsAccountCss) &&
      /\.tools-account-status\s*\{[^}]*min-width:\s*0;[^}]*max-width:\s*100%;[^}]*overflow-wrap:\s*anywhere;/s.test(toolsAccountCss),
    'signed-out labels, account actions and long save statuses should wrap inside their available width while retaining 44px targets',
  );
  assert(
    /\.personal-tool-header__account:has\(\.tools-account-extensions:not\(\[hidden\]\)\)\s*\{[^}]*grid-column:\s*1 \/ -1;[^}]*grid-row:\s*auto;[^}]*width:\s*100%;[^}]*max-width:\s*100%;/s.test(toolsMobileCss) &&
      /\.personal-tool-header__actions > \*\s*\{[^}]*box-sizing:\s*border-box;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;[^}]*min-block-size:\s*44px;[^}]*white-space:\s*normal;/s.test(toolsWorkspaceCss),
    'visible Save/status controls should receive a full compact header row and extra header actions must fit their containing block',
  );
  assert(
    !baseCss.includes('font-size:16px !important;') &&
      !baseCss.includes('input:is(') &&
      mobileDockCss.includes('max(14px, env(safe-area-inset-right, 0px))') &&
      mobileDockCss.includes('max(14px, env(safe-area-inset-left, 0px))') &&
      /\.mobile-site-masthead__search-input\s*\{[^}]*font-size:\s*1rem;/s.test(mobileDockCss),
    'shared mobile styles should preserve component typography while the masthead honors lateral safe areas',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-item="short-links"\]\s*\{[^}]*--shortlinks-shell-width:\s*100%;/s.test(shortLinksCss) &&
      /body\.personal-accordion-page\[data-personal-item="short-links"\]\s+:is\(\s*\.shortlinks-app-shell,\s*\.shortlinks-workspace,\s*\.shortlinks-main-workflow\s*\)\s*\{[^}]*min-width:\s*0;[^}]*max-width:\s*100%;/s.test(shortLinksCss),
    'Short Links should shrink its nested personal shell to the containing panel width',
  );
  assert(
    /body\.stormbreak-page\.personal-accordion-page\[data-personal-item="stormbreak"\]\s*\{[^}]*overflow-x:\s*clip;[^}]*overflow-y:\s*visible;/s.test(stormbreakCss),
    'Stormbreak should preserve the personal shell scroll owner while clipping horizontal overflow',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-category="tools"\]\[data-tools-layout="directory"\] \.personal-library-main--tools > \.tools-account-dock\s*\{[^}]*position:\s*absolute;[^}]*top:\s*16px;[^}]*right:\s*var\(--tools-content-gutter\);[^}]*z-index:\s*9;[^}]*width:\s*auto;[^}]*max-width:\s*min\(42%,\s*390px\);[^}]*padding:\s*0;/s.test(toolsMobileCss) &&
      /body\.personal-accordion-page\[data-personal-category="tools"\]\[data-tools-layout="directory"\] \.personal-library-main--tools > \.tools-account-dock \.tools-account-dock-inner\s*\{[^}]*width:\s*auto;[^}]*max-width:\s*100%;/s.test(toolsMobileCss) &&
      /body\.personal-accordion-page\[data-personal-category="tools"\]\[data-tools-layout="directory"\] \.personal-library__account \+ \.personal-library \.home-library__heading\s*\{[^}]*padding-right:\s*min\(42%,\s*390px\);/s.test(toolsMobileCss),
    'the mobile Tools directory should overlay its compact account dock while reserving heading space',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-item="utm-batch-builder"\] #utmtool-exclude-rules\s*\{[^}]*white-space:\s*pre-wrap;[^}]*overflow-wrap:\s*anywhere;/s.test(utmMobileCss),
    'the mobile UTM exclusion editor should wrap long rules inside the personal shell',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-tools-link,\s*body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-bar :is\(\.btn-primary,\s*\.btn-secondary,\s*\.btn-ghost\),\s*body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-trigger,\s*body\.personal-accordion-page\[data-personal-category="tools"\] :is\(\s*\.textcompare-field-btn,\s*\.wordfreq-field-btn,\s*\.povcheck-field-btn,\s*\.oxford-field-btn,\s*\.nbsp-field-btn,\s*\.qrtool-mode-btn,\s*\.qrtool-tab,\s*\.screenrec-test-button,\s*\.screenrec-crop-presets-toggle,\s*\.screenrec-crop-button,\s*\.screenrec-delay-button,\s*\.jobtrack-tab,\s*\.ga4-tab,\s*\.ctc-button,\s*\.ctc-icon-button\s*\)\s*\{[^}]*min-width:\s*44px;[^}]*min-height:\s*44px;[^}]*min-inline-size:\s*44px;[^}]*min-block-size:\s*44px;/s.test(toolsMobileCss),
    'mobile tool account actions and compact field controls should keep exact 44px targets',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-tools-link,\s*body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-bar :is\(\.btn-primary,\s*\.btn-secondary,\s*\.btn-ghost\),\s*body\.personal-accordion-page\[data-personal-category="tools"\] \.tools-account-trigger\s*\{[^}]*min-block-size:\s*44px;[^}]*padding:\s*8px 11px;/s.test(toolsWorkspaceCss),
    'compact desktop tool account actions should retain a full 44px target',
  );
  assert(
    /body\.personal-accordion-page\[data-personal-category="tools"\] :is\(\s*button,\s*summary\[role="button"\],\s*input:not\(\[type="hidden"\]\):not\(\[type="checkbox"\]\):not\(\[type="radio"\]\):not\(\[type="range"\]\):not\(\.visually-hidden\),\s*select\s*\)\s*\{[^}]*box-sizing:\s*border-box;[^}]*min-inline-size:\s*44px;[^}]*min-block-size:\s*44px;/s.test(personalAccordionCss) &&
      /body\.personal-accordion-page\[data-personal-category="tools"\] \.utmtool-file-input::file-selector-button\s*\{[^}]*box-sizing:\s*border-box;[^}]*min-block-size:\s*44px;/s.test(personalAccordionCss),
    'mobile personal tool pages should normalize native controls, generated share actions, and file pickers to 44px targets',
  );
  assert(
    /body \.hero\.hero--default > \.wrapper,[\s\S]*?body \.tools-hero > \.wrapper \{[^}]*width:\s*calc\(100% - \(var\(--mobile-page-gutter\) \* 2\)\);[^}]*max-width:\s*calc\(100% - \(var\(--mobile-page-gutter\) \* 2\)\);/s.test(designSystemCss) &&
      /body\[data-page="analytics"\]\.home-pattern-page \.hero\.hero--default > \.wrapper \{[^}]*width:\s*calc\(100% - 28px\);[^}]*max-width:\s*calc\(100% - 28px\) !important;/s.test(recruiterStoryCss),
    'mobile hero wrappers should calculate their gutters from the containing block instead of the scrollbar-inclusive viewport',
  );
  assert(
    /#certifications \.cert-band-inner \{[^}]*box-sizing:\s*border-box;[^}]*max-width:\s*100% !important;/s.test(recruiterStoryCss),
    'the analytics certification frame should stay contained while its inner track owns horizontal scrolling',
  );
  assert(
    /\.contact-professional-links a\{[^}]*box-sizing:\s*border-box;[^}]*display:\s*inline-flex;[^}]*min-height:\s*44px;[^}]*padding-inline:\s*10px;/s.test(contactCardCss) &&
      /\.mobile-site-masthead__brand \{[^}]*box-sizing:\s*border-box;[^}]*min-height:\s*44px;/s.test(mobileDockCss),
    'professional proof links and the mobile masthead brand should preserve 44px touch targets',
  );

  const privacyCss = read('css/privacy.css');
  const consentJs = read('js/privacy/consent_manager.js');
  const contactJs = read('js/forms/contact.js');
  const variablesCss = read('css/variables.css');
  const modalCss = read('css/components/modal.css');
  assert(
    workbenchCss.includes('grid-template-columns: min(5.5rem, 28%) minmax(0, 1fr);') &&
      /\.portfolio-result-card__open\s*\{[^}]*box-sizing:\s*border-box;[^}]*min-height:\s*44px;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;[^}]*white-space:\s*normal;[^}]*overflow-wrap:\s*anywhere;/s.test(workbenchCss),
    'enlarged text must not let the compact project thumbnail or Quick view action exceed its card width',
  );
  assert(
    /\.contact-form \.form-actions > \*\s*\{[^}]*box-sizing:\s*border-box;[^}]*min-width:\s*0;[^}]*max-width:\s*100%;[^}]*min-block-size:\s*44px;/s.test(modalCss) &&
      /\.portfolio-inspector__cta\s*\{[^}]*min-height:\s*44px;/s.test(workbenchCss),
    'Clear form and the case-study CTA must retain a full 44px target in every layout',
  );
  const modalMobileCss = modalCss.slice(modalCss.lastIndexOf('@media (max-width: 768px)'));
  const privacyMobileCss = privacyCss.slice(privacyCss.indexOf('@media (max-width: 640px)'));
  assert(
    /--modal-radius\s*:\s*var\(--radius-16\)\s*;/.test(variablesCss) &&
      /--modal-radius-mobile\s*:\s*var\(--radius-12\)\s*;/.test(variablesCss) &&
      /\.modal-content\s*\{[^}]*border-radius\s*:\s*var\(--modal-radius\)\s*;/s.test(modalCss) &&
      /#pcz-modal \.pcz-panel\s*\{[^}]*--pcz-panel-radius\s*:\s*var\(--modal-radius,[^;]+\)\s*;[^}]*border-radius\s*:\s*var\(--pcz-panel-radius\)\s*;/s.test(privacyCss) &&
      /\.modal-content\s*\{[^}]*border-radius\s*:\s*var\(--modal-radius-mobile\)\s*;/s.test(modalMobileCss) &&
      /#pcz-modal \.pcz-panel\s*\{[^}]*--pcz-panel-radius\s*:\s*var\(--modal-radius-mobile,[^;]+\)\s*;[^}]*border-radius\s*:\s*var\(--pcz-panel-radius\)\s*;/s.test(privacyMobileCss),
    'generic and Cookie Settings shells should use 16px desktop and 12px mobile radius tokens',
  );
  assert(
    /\.modal-close\s*\{[^}]*width\s*:\s*44px\s*;[^}]*height\s*:\s*44px\s*;/s.test(modalCss) &&
      /#pcz-modal \.pcz-panel-close\s*\{[^}]*width\s*:\s*44px\s*;[^}]*height\s*:\s*44px\s*;/s.test(privacyCss),
    'generic and Cookie Settings close controls should preserve 44px targets',
  );
  assert(
    modalMobileCss.includes('--modal-mobile-bottom-clearance: calc(8px + env(safe-area-inset-bottom, 0px));') &&
      modalMobileCss.includes('max(8px, env(safe-area-inset-right))') &&
      modalMobileCss.includes('max(8px, env(safe-area-inset-left))') &&
      /\.modal-content\s*\{[^}]*max-height\s*:\s*calc\(100svh - var\(--modal-mobile-top-clearance\) - var\(--modal-mobile-bottom-clearance\)\)/s.test(modalMobileCss) &&
      /\.modal-body\s*\{[^}]*overflow-y\s*:\s*auto\s*;/s.test(modalCss) &&
      privacyMobileCss.includes('max(8px, env(safe-area-inset-right))') &&
      privacyMobileCss.includes('max(8px, env(safe-area-inset-left))') &&
      /#pcz-modal \.pcz-panel\s*\{[^}]*max-height\s*:\s*calc\(100svh - var\(--pcz-mobile-top-clearance\) - 8px\)/s.test(privacyMobileCss) &&
      /#pcz-modal \.pcz-panel\s*\{[^}]*overflow\s*:\s*auto\s*;/s.test(privacyCss),
    'shared modal shells should retain mobile safe-area clearance and internal scrolling',
  );
  assert(
    /body\.consent-blocked:has\(#pcz-modal\.pcz-visible\)::before\s*\{[^}]*opacity\s*:\s*0\s*!important\s*;[^}]*pointer-events\s*:\s*none\s*!important\s*;[^}]*backdrop-filter\s*:\s*none\s*;/s.test(privacyCss) &&
      consentJs.includes("const CSS_VERSION = 'v13';") &&
      consentJs.includes('#pcz-modal{background:var(--modal-backdrop,rgba(9,31,59,.58))') &&
      consentJs.includes('body.consent-blocked:has(#pcz-modal.pcz-visible):before{opacity:0!important;pointer-events:none!important;') &&
      consentJs.includes('#pcz-modal .pcz-panel{--pcz-panel-radius:var(--modal-radius,12px);') &&
      consentJs.includes('@media(max-width:640px){#pcz-modal .pcz-panel{--pcz-panel-radius:var(--modal-radius-mobile,12px);}}') &&
      consentJs.includes('#pcz-modal .pcz-panel-close{width:44px;height:44px;border-radius:12px;'),
    'Cookie Settings critical CSS v13 should match the shared shell without stacking the first-run backdrop',
  );
  assert(
    !privacyCss.includes('@media (prefers-color-scheme: dark)') &&
      !privacyCss.includes('data-theme-scope') &&
      /#pcz-banner \.pcz-btn,\s*#pcz-modal \.pcz-save-preferences\s*\{[^}]*min-height:\s*44px;/s.test(privacyCss) &&
      /#pcz-banner \.pcz-close\s*\{[^}]*width:\s*32px;[^}]*height:\s*32px;/s.test(privacyCss) &&
      /#pcz-modal \.pref-info\s*\{[^}]*width:\s*44px;[^}]*height:\s*44px;/s.test(privacyCss) &&
      consentJs.includes('modal._pczRestoreBackground = isolateModalBackground(modal);') &&
      consentJs.includes('modal._pczReturnFocus = returnFocus;'),
    'consent surfaces should remain compact and light while the modal isolates the page and restores focus',
  );
  assert(
    contactJs.includes('const focusDialog = () => {') &&
      contactJs.includes('window.requestAnimationFrame(focusDialog);'),
    'direct contact-modal hashes should move focus into the dialog after it becomes active',
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
  assert(
    /body\.personal-accordion-page\[data-personal-item="stellar-dogfight"\] :is\([\s\S]*?\.btn,[\s\S]*?\.tab-btn,[\s\S]*?\.option-btn,[\s\S]*?\.keybind-btn,[\s\S]*?\.panel-disclosure-toggle[\s\S]*?\)\s*\{[^}]*min-height:\s*44px;[^}]*min-block-size:\s*44px;/s.test(stellarCss) &&
      /body\.personal-accordion-page\[data-personal-item="stellar-dogfight"\] :is\([\s\S]*?\.btn-icon-only,[\s\S]*?\.command-drawer \.tab-btn[\s\S]*?\)\s*\{[^}]*width:\s*44px;[^}]*min-width:\s*44px;[^}]*min-inline-size:\s*44px;/s.test(stellarCss),
    'mobile Stellar Dogfight menu controls should keep 44px targets without resizing the canvas controls',
  );

  const gamesCss = read('css/components/games.css');
  const oceanCss = read('css/components/ocean-wave-simulation.css');
  const probabilityCss = read('css/games/probability-engine.css');
  const rouletteCss = read('css/games/roulette.css');
  const rouletteHtml = read('pages/games/roulette.html');
  assert(
    /\.games-directory-card__action\s*\{[^}]*min-height:\s*44px;/s.test(gamesCss) &&
      /\.ocean-wave-preset\s*\{[^}]*min-height:\s*44px;/s.test(oceanCss) &&
      /\.roulette00-rules-disclosure summary\s*\{[^}]*min-height:\s*44px;/s.test(rouletteCss) &&
      rouletteHtml.includes('<details class="roulette00-rules-disclosure">'),
    'the simplified Games surfaces should retain 44px actions and place secondary roulette rules in a disclosure',
  );
  assert(
    /\.roulette00-spin-btn,\s*\.roulette00-action,\s*\.roulette00-line-tab,\s*\.roulette00-bet\.is-line\s*\{[^}]*min-height:\s*44px;[^}]*min-block-size:\s*44px;/s.test(rouletteCss) &&
      /body\.personal-accordion-page\[data-personal-item="probability-engine"\] :is\(\s*\.app-shell button,\s*#offline-modal button\s*\)\s*\{[^}]*min-height:\s*44px;[^}]*min-block-size:\s*44px;/s.test(probabilityCss),
    'mobile Roulette and Probability Engine actions should preserve 44px touch targets',
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
      projectIframe.includes('data-src="/demos/sentence-demo.html"') &&
      !/\ssrc="/.test(projectIframe),
    'same-origin content demos should defer iframe loading and provide a mobile launch card',
  );
  assert(
    projectPage.includes('class="project-section project-resources project-resources--flat"') &&
      countMatches(projectPage, /class="project-links"/g) === 1 &&
      !projectPage.includes('data-project-mobile-disclosure') &&
      !projectPage.includes('class="project-disclosure-summary"'),
    'simplified project details should render one flat Links section without legacy disclosures',
  );

  const commonJs = read('js/common/common.js');
  const projectCss = read('css/components/project-page.css');
  assert(
    commonJs.includes("window.matchMedia('(max-width: 768px)')") &&
      commonJs.includes("ifr.removeAttribute('src')") &&
      commonJs.includes("ifr.setAttribute('src', deferredSrc)") &&
      commonJs.includes('observeProjectEmbedIframe(ifr)'),
    'project runtime should unload content iframes on mobile and restore observation on desktop',
  );
  assert(
    commonJs.includes('const resetPersonalProjectDetailScroll = () => {') &&
      commonJs.includes("navigation?.type === 'back_forward'") &&
      commonJs.includes("document.querySelector('[data-personal-detail-content]')") &&
      commonJs.includes('content.scrollTop = 0;'),
    'new project navigations should start at the title while browser history can restore an earlier detail position',
  );
  assert(
    /\.project-main--compact \.project-link\s*\{[^}]*min-height:\s*48px;/s.test(projectCss) &&
      projectCss.includes('.project-demo-shell[data-demo-fit="content"] .project-demo-mobile-launch') &&
      projectCss.includes('.project-demo-shell[data-demo-fit="content"] .project-embed') &&
      projectCss.includes('max-height:min(960px, calc(100svh - var(--nav-height, 72px) - 48px));'),
    'project CSS should keep touch-sized flat links, swap content embeds for the launch card on mobile, and cap iframe height',
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
