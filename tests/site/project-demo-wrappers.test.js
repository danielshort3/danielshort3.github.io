'use strict';

const fs = require('fs');
const path = require('path');
const {
  GUARD_END,
  GUARD_START,
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

const ROOT = path.resolve(__dirname, '..', '..');

function read(relativePath) {
  return fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
}

function count(source, pattern) {
  return (String(source || '').match(pattern) || []).length;
}

function runProjectDemoWrapperTests({ assert }) {
  const definitions = loadProjectDemoDefinitions();
  const rewrites = JSON.parse(read('vercel.json')).rewrites || [];
  const css = read('css/components/personal-accordion-shell.css');
  const buildRunner = read('build/build-site.js');

  assert(definitions.length === 12 && PROJECT_DEMO_IDS.length === 12,
    'Project demo continuity should cover all 12 raw demo documents');
  assert(new Set(definitions.map((item) => item.demoId)).size === 12,
    'Project demo wrapper definitions should have unique route ids');
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
    assert(wrapper.includes(`href="${definition.backHref}" aria-label="${definition.backLabel.replace(/&/g, '&amp;').replace(/"/g, '&quot;')}"`) &&
      wrapper.includes(`href="${definition.canonicalUrl}"`),
    `${wrapperRelativePath} should retain its project/library return path and canonical URL`);
    assert(count(wrapper, /<iframe\b/gi) === 1 &&
      wrapper.includes(`class="project-demo-wrapper-iframe" src="${definition.rawPath}"`) &&
      !wrapper.includes(`class="project-demo-wrapper-iframe" src="${definition.canonicalPath}"`),
    `${wrapperRelativePath} should isolate the raw demo in exactly one non-recursive iframe`);
    assert(wrapper.includes('const suffix = window.location.search + window.location.hash;') &&
      wrapper.includes(`frame.src = ${JSON.stringify(definition.rawPath)} + suffix;`),
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
    sentencePage.includes('href="https://www.danielshort.me/sentence-demo"') &&
    !sentencePage.includes('href="https://www.danielshort.me/demos/sentence-demo.html"'),
  'Project details should iframe the raw demo while Live Demo continues to target the canonical wrapper');

  assert(buildRunner.includes("generate-project-demo-wrappers.js") &&
    buildRunner.indexOf("generate-personal-accordion-pages.js") < buildRunner.indexOf("generate-project-demo-wrappers.js"),
  'The site build should generate themed demo wrappers after the personal project shell');
  assert(css.includes('body.project-demo-wrapper-page .project-demo-wrapper-iframe') &&
    css.includes('body.project-demo-wrapper-page .personal-accordion__content') &&
    css.includes('overflow: hidden !important;'),
  'The personal shell should give wrapper iframes a full, isolated content viewport');
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
