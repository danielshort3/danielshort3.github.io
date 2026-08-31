const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));

const countMatches = (value, pattern) => (String(value || '').match(pattern) || []).length;

module.exports = function runPortfolioRecommendationTests({ assert }) {
  const personal = readJson('content/audiences/personal.json');
  const startHere = Array.isArray(personal.startHere) ? personal.startHere : [];
  assert(
    JSON.stringify(startHere.map((item) => item.id)) === JSON.stringify([
      'handwritingRating',
      'project-starfall',
      'tools',
    ]),
    'personal Start Here should use the approved project, game, and tools entry points',
  );
  assert(
    JSON.stringify(startHere.map((item) => item.href)) === JSON.stringify([
      '/portfolio/handwritingRating',
      '/games/project-starfall',
      '/tools',
    ]),
    'personal Start Here should use stable clean routes',
  );

  const audienceConfig = read('js/common/audience-config.js');
  const personalSource = read('content/audiences/personal.json');
  const indexHtml = read('index.html');
  const accordionJs = read('js/home/category-accordion.js');
  const accordionCss = read('css/components/home-category-accordion.css');
  startHere.forEach((item) => {
    assert(audienceConfig.includes(`id: '${item.id}'`), `generated audience config missing Start Here item ${item.id}`);
    assert(personalSource.includes(item.href), `personal no-JS source missing Start Here route ${item.href}`);
    assert(indexHtml.includes(item.href), `generated homepage missing Start Here route ${item.href}`);
  });
  assert(!personalSource.includes('href=\\"/analytics') && !indexHtml.includes('href="/analytics"'),
    'personal homepage sources should not advertise an unlisted professional route');
  assert(!personalSource.includes('professional analytics profile') && !personalSource.includes('professional analytics work'),
    'personal Start Here copy should not disclose hidden professional entry points');
  assert(
    indexHtml.includes('data-home-accordion-item="about"') &&
      indexHtml.includes('data-content-id="handwritingRating"') &&
      indexHtml.includes('data-content-id="project-starfall"') &&
      indexHtml.includes('href="/tools"'),
    'home accordion should keep the approved personal starting points in its static panels',
  );
  assert(
    !/animation[^;\n}]*\binfinite\b/.test(accordionCss),
    'home accordion motion should not loop infinitely',
  );
  assert(
    accordionCss.includes('@media (pointer: coarse)') &&
      accordionCss.includes('min-height: 44px') &&
      accordionJs.includes("event.key === 'ArrowDown'"),
    'home accordion should preserve coarse-pointer targets and keyboard rail navigation',
  );

  const storyIds = ['retailStore', 'chatbotLora', 'digitGenerator', 'smartSentence', 'website'];
  const evaluationStatuses = {
    chatbotLora: 'not-benchmarked',
    covidAnalysis: 'measured',
    digitGenerator: 'partial',
    handwritingRating: 'measured',
    nonogram: 'not-benchmarked',
    shapeClassifier: 'not-benchmarked',
    sheetMusicUpscale: 'partial',
    smartSentence: 'not-benchmarked',
  };

  storyIds.forEach((id) => {
    const project = readJson(`content/projects/${id}.json`);
    assert(project.personalStory && typeof project.personalStory === 'object', `${id} missing personalStory`);
    ['why', 'surprise', 'next'].forEach((field) => {
      assert(
        typeof project.personalStory[field] === 'string' && project.personalStory[field].trim(),
        `${id} personalStory.${field} should be a non-empty string`,
      );
    });
  });

  Object.entries(evaluationStatuses).forEach(([id, expectedStatus]) => {
    const project = readJson(`content/projects/${id}.json`);
    const evaluation = project.evaluation;
    assert(evaluation && evaluation.status === expectedStatus, `${id} should use evaluation status ${expectedStatus}`);
    ['goal', 'dataset', 'split', 'baseline', 'decision'].forEach((field) => {
      assert(typeof evaluation[field] === 'string' && evaluation[field].trim(), `${id} evaluation.${field} missing`);
    });
    assert(Array.isArray(evaluation.metrics), `${id} evaluation.metrics should be an array`);
    evaluation.metrics.forEach((metric) => {
      assert(metric && metric.label && metric.value && metric.context, `${id} has an incomplete evaluation metric`);
    });
    assert(
      Array.isArray(evaluation.limitations) && evaluation.limitations.length > 0,
      `${id} should disclose at least one evaluation limitation`,
    );
    assert(
      evaluation.evidence && evaluation.evidence.label && evaluation.evidence.url,
      `${id} should link its evaluation context`,
    );
  });

  const covid = readJson('content/projects/covidAnalysis.json');
  const covidResources = JSON.stringify(covid.resources);
  assert(
    covidResources.includes('https://github.com/danielshort3/Covid-Analysis/blob/main/covid_analysis.ipynb') &&
      !covidResources.includes('documents/Project_6.pdf') &&
      !covidResources.includes('documents/Project_6.ipynb'),
    'COVID resources should point to the current XGBoost notebook instead of stale local artifacts',
  );
  assert(
    covid.evaluation.metrics.some((metric) => metric.label === 'AUROC' && metric.value === '0.606') &&
      covid.evaluation.metrics.some((metric) => metric.label === 'PR-AUC' && metric.value === '0.060') &&
      covid.evaluation.metrics.some((metric) => metric.label === 'Recall at 75% precision' && metric.value === '0.000'),
    'COVID evaluation should publish the weak measured results without hiding the target failure',
  );

  const publicProofText = [
    read('content/audiences/data-science.json'),
    read('content/audiences/tourism.json'),
    read('js/portfolio/portfolio.js'),
    ...Object.keys(evaluationStatuses).map((id) => read(`content/projects/${id}.json`)),
  ].join('\n');
  ['+14.13%', '+23.3%', 'High accuracy', 'strong solve rates'].forEach((claim) => {
    assert(!publicProofText.includes(claim), `unsupported public claim should be removed: ${claim}`);
  });

  const audienceExpectations = {
    analytics: ['99%', '200+', '24%', '57.6%'],
    'data-science': ['95%', '10x', '98%', '+9.4%'],
    tourism: ['99%', '200+', '+9.4%', '10x'],
  };
  Object.entries(audienceExpectations).forEach(([audienceKey, values]) => {
    const source = read(`content/audiences/${audienceKey}.json`);
    const generated = read(`pages/${audienceKey}.html`);
    [source, generated].forEach((html) => {
      assert(html.includes('professional-hero-proof'), `${audienceKey} should render first-viewport proof`);
      values.forEach((value) => assert(html.includes(value), `${audienceKey} proof missing ${value}`));
      assert(countMatches(html, /professional-hero-proof-link/g) >= 4, `${audienceKey} hero proof should expose four links`);
    });
    assert(countMatches(source, /home-proof-source/g) >= 4,
      `${audienceKey} source should retain the authored context links`);
    assert(countMatches(generated, /home-proof-source/g) === 0,
      `${audienceKey} rendered page should not repeat hero proof in a second outcome-card section`);
  });
  const homeProofCss = read('css/components/home-proof.css');
  assert(
    homeProofCss.includes('body:is([data-page="analytics"], [data-page="data-science"], [data-page="tourism"]).home-pattern-page .professional-hero-proof-link strong') &&
      homeProofCss.includes('color: var(--story-blue-strong'),
    'all professional hero proof values should use a readable light-theme color',
  );
  const dataScienceSource = read('content/audiences/data-science.json');
  const tourismSource = read('content/audiences/tourism.json');
  assert(
    dataScienceSource.includes('href=\"#work-experience\"') || dataScienceSource.includes('href=\\\"#work-experience\\\"'),
    'data-science web-growth proof should link to the on-page work context',
  );
  assert(
    countMatches(tourismSource, /#work-experience/g) >= 2,
    'tourism organic and AI-referral proof should link to the on-page work context',
  );

  const projectsData = read('js/portfolio/projects-data.js');
  const projectGenerator = read('build/generate-project-pages.js');
  assert(
    projectGenerator.includes("join(' \\u00b7 ')") && !projectGenerator.includes("join(' ? ')"),
    'generated project stack labels should use a middle dot instead of a question mark',
  );
  storyIds.forEach((id) => {
    const project = readJson(`content/projects/${id}.json`);
    assert(projectsData.includes(project.personalStory.why), `${id} story should survive generated project data`);
  });
  Object.keys(evaluationStatuses).forEach((id) => {
    const page = read(`pages/portfolio/${id}.html`);
    const starIndex = page.indexOf('STAR Summary');
    const evaluationIndex = page.indexOf('Evaluation &amp; tradeoffs');
    const demoIndex = page.indexOf('project-demo-shell');
    assert(
      starIndex >= 0 && evaluationIndex > starIndex && demoIndex > evaluationIndex,
      `${id} should render STAR, then evaluation, then the demo or preview`,
    );
  });
  storyIds.forEach((id) => {
    const page = read(`pages/portfolio/${id}.html`);
    const starIndex = page.indexOf('STAR Summary');
    const storyIndex = page.indexOf('Personal notes');
    const demoIndex = page.indexOf('project-demo-shell');
    assert(
      starIndex >= 0 && storyIndex > starIndex && demoIndex > storyIndex,
      `${id} should render personal notes after STAR and before the demo or preview`,
    );
  });

  const portfolioHtml = read('pages/portfolio.html');
  const professionalPortfolioHtml = read('pages/professional/analytics/portfolio.html');
  const portfolioJs = read('js/portfolio/portfolio.js');
  assert(
    !portfolioHtml.includes('<option value="default">Featured first</option>') &&
      portfolioHtml.includes('data-personal-accordion-shell') &&
      professionalPortfolioHtml.includes('<option value="default">Featured first</option>') &&
      portfolioJs.includes('Most relevant') &&
      portfolioJs.includes('Featured first'),
    'professional workbench sorting should keep audience-aware labels while personal uses the direct project library',
  );
  assert(
    !portfolioHtml.includes('Preview summary') &&
      !portfolioJs.includes('Preview summary') &&
      portfolioJs.includes('aria-label="Quick view for ${escapeHtml(project.title)}">Quick view</button>'),
    'portfolio cards should remove the wordy Preview summary action and use a concise Quick view control only when JavaScript is available',
  );
  assert(
    portfolioJs.includes('personalStoryItems') &&
      portfolioJs.includes('Personal notes') &&
      portfolioJs.includes('!isAudienceScopedView'),
    'personal portfolio search and inspector should expose authored story fields',
  );

  const contentModel = read('api/_lib/cms-content-model.js');
  assert(
    contentModel.includes('validateProjectPersonalStory') &&
      contentModel.includes('validateProjectEvaluation') &&
      contentModel.includes("'measured', 'partial', 'not-benchmarked'"),
    'CMS validation should enforce the optional story and evaluation contracts',
  );
};
