'use strict';

const fs = require('fs');
const path = require('path');
const {
  GUARD_END,
  GUARD_START,
  extractDemoDescriptionHtml,
  injectRawDemoGuard,
  loadProjectDemoDefinitions,
  renderDemoWrapperPage
} = require('../../build/generate-project-demo-wrappers');
const {
  PROJECT_DEMO_IDS,
  toCanonicalProjectDemoUrl,
  toRawProjectDemoUrl
} = require('../../build/lib/project-demo-routes');
const { renderProjectPage } = require('../../build/generate-project-pages');
const { PERSONAL_CONTENT_START, PERSONAL_CONTENT_END } = require('../../build/lib/personal-accordion-shell');

const ROOT = path.resolve(__dirname, '..', '..');

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

function count(source, pattern) {
  return (String(source || '').match(pattern) || []).length;
}

function escapeHtml(value) {
  return String(value || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function runProjectDemoWrapperTests({ assert }) {
  const definitions = loadProjectDemoDefinitions();
  const rewrites = JSON.parse(read('vercel.json')).rewrites || [];
  const css = read('css/components/personal-accordion-shell.css');
  const projectCss = read('css/components/project-page.css');
  const buildRunner = read('build/build-site.js');
  const demoWrapperLifecycle = read('js/navigation/project-demo-wrapper.js');
  const frameCss = read('css/components/site-frame.css');

  assert(definitions.length === 12 && PROJECT_DEMO_IDS.length === 12,
    'Project demo continuity should cover all 12 raw demo documents');
  assert(new Set(definitions.map((item) => item.demoId)).size === 12,
    'Project demo wrapper definitions should have unique route ids');
  assert(extractDemoDescriptionHtml('<div class="header-copy"><h1>Demo</h1><p class="subtitle">Read <a href="/privacy">privacy details</a>.</p><p>Keep this disclosure.</p></div>') ===
    '<p>Read <a href="/privacy">privacy details</a>.</p>\n<p>Keep this disclosure.</p>',
  'Standalone demo descriptions should preserve instructional links and every existing disclosure paragraph');
  assert(extractDemoDescriptionHtml('<div class="other"><p>Unrelated copy</p></div>') === '',
    'Unrecognized demo layouts should use the project description fallback');
  const fallbackDefinition = { ...definitions[0], descriptionHtml: '', subtitle: 'Text & examples <inside the browser>' };
  assert(renderDemoWrapperPage(fallbackDefinition).includes('<p>Text &amp; examples &lt;inside the browser&gt;</p>'),
    'Project subtitles should remain safely escaped when a raw demo has no instructional description');
  assert(definitions.find((definition) => definition.demoId === 'chatbot-demo')?.descriptionHtml.includes('Inputs and responses are saved on AWS.'),
    'The standalone chatbot masthead should retain its existing storage disclosure');
  assert(toRawProjectDemoUrl('https://www.danielshort.me/shape-demo') === '/demos/shape-demo.html' &&
    toRawProjectDemoUrl('shape-demo.html') === '/demos/shape-demo.html' &&
    toCanonicalProjectDemoUrl('/demos/shape-demo.html') === '/shape-demo' &&
    toRawProjectDemoUrl('/shape-demo?model=small#draw') === '/demos/shape-demo.html?model=small#draw' &&
    toCanonicalProjectDemoUrl('/demos/shape-demo.html?model=small#draw') === '/shape-demo?model=small#draw' &&
    toRawProjectDemoUrl('https://example.com/shape-demo') === 'https://example.com/shape-demo',
  'Project demo routing should separate canonical same-origin wrappers from raw iframe documents');

  definitions.forEach((definition) => {
    const rawRelativePath = `demos/${definition.demoId}.html`;
    const wrapperRelativePath = `pages/demos/${definition.demoId}.html`;
    const raw = read(rawRelativePath);
    const wrapper = read(wrapperRelativePath);
    const wrapperFromSource = renderDemoWrapperPage(definition);
    const guardedAgain = injectRawDemoGuard(raw, definition);

    assert(wrapperFromSource.includes(`class="project-demo-wrapper-iframe" src="${definition.rawPath}"`) &&
      wrapperFromSource.includes('data-personal-category="projects"'),
    `${wrapperRelativePath} should be reproducible from the authoritative wrapper generator`);
    assert(wrapperFromSource.includes('data-project-demo-src=') &&
      wrapperFromSource.includes('<script defer src="js/navigation/project-demo-wrapper.js"></script>') &&
      demoWrapperLifecycle.includes("window.matchMedia('(max-width: 959px), (max-height: 619px)')") &&
      demoWrapperLifecycle.includes("listen(frame, 'load', observeFrame);") &&
      demoWrapperLifecycle.includes('new FrameResizeObserver(scheduleMeasurement)') &&
      demoWrapperLifecycle.includes("style?.removeProperty('height')") &&
      demoWrapperLifecycle.includes('window.SiteRoutes?.addCleanup?.(dispose);'),
    `${wrapperRelativePath} should auto-size its same-origin demo on mobile and restore fixed desktop sizing`);
    assert(wrapperFromSource.includes(`data-project-demo-fit="${definition.fit}"`) &&
      ['content', 'viewport'].includes(definition.fit),
    `${wrapperRelativePath} should distinguish natural content from bounded chat viewports`);
    assert(wrapperFromSource.includes(`>${definition.backCompactLabel}</span>`),
      `${wrapperRelativePath} should use a concise visible return label with a descriptive accessible name`);
    const fragment = wrapperFromSource.slice(
      wrapperFromSource.indexOf(PERSONAL_CONTENT_START) + PERSONAL_CONTENT_START.length,
      wrapperFromSource.indexOf(PERSONAL_CONTENT_END)
    );
    const masthead = /<header\b[^>]*\sdata-project-demo-masthead=[^>]*>[\s\S]*?<\/header>/i.exec(fragment)?.[0] || '';
    const expectedTitle = /\bdemo$/i.test(definition.title) ? definition.title : `${definition.title} Demo`;
    assert(count(fragment, /\sdata-page-masthead(?=[\s=>])/g) === 1 &&
      masthead.includes(`data-project-demo-masthead="${definition.demoId}"`) &&
      masthead.includes('data-page-masthead-intro') && masthead.includes('data-page-masthead-copy') &&
      masthead.includes(`<h1>${escapeHtml(expectedTitle)}</h1>`) &&
      definition.descriptionHtml && masthead.includes(definition.descriptionHtml),
    `${wrapperRelativePath} should render its project title and existing demo instructions in the shared section masthead`);
    assert(fragment.indexOf(masthead) < fragment.indexOf('<main id="main"') &&
      !/<main\b[^>]*>[\s\S]*data-page-masthead/.test(fragment) &&
      !/\sdata-site-route-toolbar(?=[\s=>])/.test(wrapperFromSource) &&
      !wrapperFromSource.includes('class="personal-accordion__back"'),
    `${wrapperRelativePath} should keep its masthead outside the measured demo main without a duplicate back toolbar`);
    assert(count(raw, new RegExp(GUARD_START, 'g')) === 1 &&
      count(raw, new RegExp(GUARD_END, 'g')) === 1 &&
      raw.includes('if (window.self === window.top)') &&
      raw.includes(`window.location.replace(${JSON.stringify(definition.canonicalPath)} + window.location.search + window.location.hash);`),
    `${rawRelativePath} should redirect only top-level visits to its canonical wrapper`);
    assert(guardedAgain === raw,
      `${rawRelativePath} top-level wrapper guard should be idempotent`);

    assert(wrapper.includes('data-page="project-demo"') &&
      wrapper.includes('data-personal-category="projects"') &&
      wrapper.includes('data-personal-fit="viewport"') &&
      wrapper.includes('data-personal-chrome="compact"') &&
      count(wrapper, /data-personal-rail-active="true"/g) === 1,
    `${wrapperRelativePath} should use one compact Projects shell`);
    assert(wrapper.includes(`href="${escapeHtml(definition.backHref)}" aria-label="${escapeHtml(definition.backLabel)}" data-page-masthead-parent`) &&
      wrapper.includes(`href="${definition.canonicalUrl}"`),
    `${wrapperRelativePath} should retain its project/library return path and canonical URL`);
    assert(count(wrapper, /\sdata-page-masthead(?=[\s=>])/g) === 1 &&
      count(wrapper, /\sdata-page-masthead-parent(?=[\s=>])/g) === 1 &&
      !/\sdata-site-route-toolbar(?=[\s=>])/.test(wrapper),
    `${wrapperRelativePath} should publish the shared masthead without a legacy toolbar`);
    assert(count(wrapper, /<iframe\b/gi) === 1 &&
      wrapper.includes(`class="project-demo-wrapper-iframe" src="${definition.rawPath}"`) &&
      !wrapper.includes(`class="project-demo-wrapper-iframe" src="${definition.canonicalPath}"`),
    `${wrapperRelativePath} should isolate the raw demo in exactly one non-recursive iframe`);
    assert(wrapper.includes(`data-project-demo-src="${definition.rawPath}"`) &&
      wrapper.includes('<script defer src="js/navigation/project-demo-wrapper.js"></script>') &&
      !wrapper.includes('const suffix = window.location.search + window.location.hash;'),
    `${wrapperRelativePath} should forward canonical query and fragment state into the raw demo`);
    assert(!/<title>[^<]*\bDemo Demo\b/i.test(wrapper),
      `${wrapperRelativePath} should not duplicate Demo in the document title`);

    [definition.canonicalPath, `${definition.canonicalPath}.html`].forEach((source) => {
      assert(rewrites.some((rewrite) => rewrite.source === source &&
        rewrite.destination === `/pages/demos/${definition.demoId}`),
      `${source} should rewrite to the themed demo wrapper`);
    });
  });

  const sentenceProject = JSON.parse(read('content/projects/smartSentence.json'));
  const sentencePage = renderProjectPage(sentenceProject);
  assert(/class="project-embed-frame"(?:\s+src|\s+data-src)="\/demos\/sentence-demo\.html"/.test(sentencePage) &&
    sentencePage.includes('href="/sentence-demo"') &&
    !sentencePage.includes('href="https://www.danielshort.me/demos/sentence-demo.html"'),
  'Project details should iframe the raw demo while launch actions target the canonical wrapper');
  assert(!sentencePage.includes('project-link-label">Live Demo</span>'),
    'The resource list should not duplicate the primary demo launch action');

  assert(buildRunner.includes("generate-project-demo-wrappers.js") &&
    buildRunner.indexOf("generate-personal-accordion-pages.js") < buildRunner.indexOf("generate-project-demo-wrappers.js"),
  'The site build should generate themed demo wrappers after the personal project shell');
  assert(css.includes('body.project-demo-wrapper-page .project-demo-wrapper-iframe') &&
    css.includes('body.project-demo-wrapper-page .personal-accordion__content') &&
    css.includes('overflow: hidden !important;'),
  'The personal shell should give wrapper iframes a full, isolated content viewport');
  const desktopDemoBody = /body\.project-demo-wrapper-page \.site-frame\[data-frame-fit="viewport"\]\[data-frame-compact="false"\] \.site-frame__body\s*\{([^}]+)\}/.exec(frameCss)?.[1] || '';
  assert(/\bheight:\s*100%;/.test(desktopDemoBody),
    'The persistent desktop frame must preserve a definite height for demo iframe ancestors');
  require('./project-demo-sizing.test')({ assert });
  assert(css.includes('scrollbar-gutter: stable both-edges;'),
    'Desktop compact panels should reserve symmetric scrollbar space so return and content axes stay aligned');
  assert(projectCss.includes('.project-main--compact .project-demo-header{') &&
    projectCss.includes('.project-main--compact .project-demo-help{') &&
    projectCss.includes('.project-main--compact .project-demo-tooltip{') &&
    projectCss.includes('top:calc(50% + 32px);') &&
    projectCss.includes('left:0;') &&
    projectCss.includes('width:auto;'),
  'Mobile project demo help should anchor to the full demo header without clipping either edge');
}

if (require.main === module) {
  let checks = 0;
  runProjectDemoWrapperTests({
    assert(condition, message) {
      checks += 1;
      if (!condition) throw new Error(message);
    }
  });
  process.stdout.write(`Project demo wrapper tests passed (${checks} checks).\n`);
}

module.exports = runProjectDemoWrapperTests;
