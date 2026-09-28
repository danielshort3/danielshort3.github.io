const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

const read = (relativePath) => fs.readFileSync(path.join(ROOT, relativePath), 'utf8');
const readJson = (relativePath) => JSON.parse(read(relativePath));

const countMatches = (value, pattern) => (String(value || '').match(pattern) || []).length;

module.exports = function runPortfolioRecommendationTests({ assert }) {
  require('./project-content-clarity.test.js').runProjectContentClarityTests({ assert });
  const personal = readJson('content/audiences/personal.json');
  const startHere = Array.isArray(personal.startHere) ? personal.startHere : [];
  assert(
    JSON.stringify(startHere.map((item) => item.id)) === JSON.stringify([
      'handwritingRating',
      'tools',
    ]),
    'personal Start Here should use the approved project and tools entry points',
  );
  assert(
    JSON.stringify(startHere.map((item) => item.href)) === JSON.stringify([
      '/portfolio/handwritingRating',
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
      indexHtml.includes('Work in progress') &&
      indexHtml.includes('href="/tools"'),
    'home accordion should keep the selected starting points and expose the work-in-progress Project Starfall',
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
      `${id} should preserve at least one authored evaluation limitation`,
    );
    assert(
      evaluation.evidence && evaluation.evidence.label && evaluation.evidence.url,
      `${id} should preserve its authored evaluation source`,
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
    read('content/audiences/personal.json'),
    read('js/portfolio/portfolio.js'),
    ...Object.keys(evaluationStatuses).map((id) => read(`content/projects/${id}.json`)),
  ].join('\n');
  ['+14.13%', '+23.3%', 'High accuracy', 'strong solve rates'].forEach((claim) => {
    assert(!publicProofText.includes(claim), `unsupported public claim should be removed: ${claim}`);
  });

  assert(['analytics', 'data-science', 'tourism'].every((audience) =>
    !fs.existsSync(path.join(ROOT, `content/audiences/${audience}.json`)) &&
    !fs.existsSync(path.join(ROOT, `pages/${audience}.html`))),
  'Retired audience homepages and sources should be absent');

  const projectsData = read('js/portfolio/projects-data.js');
  const projectGenerator = read('build/generate-project-pages.js');
  assert(
    !projectGenerator.includes('project-pager') &&
      !projectGenerator.includes('project-personal-notes') &&
      !projectGenerator.includes('project-evidence-details') &&
      projectGenerator.includes('project-next-steps'),
    'generated project details should omit the disabled evidence disclosure and provide a curated next step without a pager or repeated personal-notes section',
  );
  storyIds.forEach((id) => {
    const project = readJson(`content/projects/${id}.json`);
    assert(projectsData.includes(project.personalStory.why), `${id} story should survive generated project data`);
  });
  Object.keys(evaluationStatuses).forEach((id) => {
    const page = read(`pages/portfolio/${id}.html`);
    const starIndex = page.indexOf('STAR Summary');
    const evaluationIndex = page.indexOf('Evidence &amp; limitations');
    const demoIndex = page.indexOf('project-demo-shell');
    assert(
      demoIndex >= 0 && evaluationIndex === -1 && starIndex > demoIndex,
      `${id} should place the demo before STAR without the disabled evidence disclosure`,
    );
  });
  storyIds.forEach((id) => {
    const page = read(`pages/portfolio/${id}.html`);
    const starIndex = page.indexOf('STAR Summary');
    const storyIndex = page.indexOf('Personal notes');
    const demoIndex = page.indexOf('project-demo-shell');
    assert(
      demoIndex >= 0 && storyIndex === -1 && starIndex > demoIndex,
      `${id} should render STAR after the demo or preview without personal notes`,
    );
  });

  const portfolioHtml = read('pages/portfolio.html');
  const portfolioJs = read('js/portfolio/portfolio.js');
  assert(
    !portfolioHtml.includes('<option value="default">Featured first</option>') &&
      portfolioHtml.includes('data-personal-accordion-shell'),
    'The canonical project library should use the direct shared shell',
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
