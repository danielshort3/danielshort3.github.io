'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { renderProjectPage } = require('../../build/generate-project-pages');
const { wrapPersonalAccordionHtml } = require('../../build/lib/personal-accordion-shell');

const ROOT = path.resolve(__dirname, '../..');
const read = (file) => fs.readFileSync(path.join(ROOT, file), 'utf8');
const project = (id) => JSON.parse(read(`content/projects/${id}.json`));
const intro = (html) => html.match(/<section class="project-hero project-hero--compact">([\s\S]*?)<\/section>/)?.[1] || '';

function runProjectPrivacyLayoutTests({ assert }) {
  const babyNames = renderProjectPage(project('babynames'));
  const babyIntro = intro(babyNames);
  assert(babyIntro.includes('class="project-intro-actions" aria-label="Project actions"'),
    'Project introductions expose a named navigation group for existing project actions.');
  assert(/class="project-intro-action project-intro-action--demo" href="\/baby-names-demo"[^>]*>Open demo<\/a>/.test(babyIntro),
    'The early Baby Names demo link opens the canonical shared wrapper, not its raw iframe.');
  assert(/class="project-intro-action project-intro-action--report" href="\/documents\/Project_2.pdf"[^>]*>Report · PDF · \d+ KB<\/a>/.test(babyIntro),
    'The early Baby Names report retains its real PDF destination, format, and file size.');
  assert(!babyIntro.includes('target="_blank"') && babyNames.indexOf('project-intro-actions') < babyNames.indexOf('class="project-star"'),
    'Local actions stay in the current tab and precede the full project narrative.');
  assert((babyIntro.match(/data-content-open="true"/g) || []).length === 2
    && (babyIntro.match(/data-content-type="project_resource"/g) || []).length === 2,
  'Intro actions enter existing content-click measurement as resources rather than counting another project view.');
  assert(babyNames.includes('class="project-demo-shell"') && babyNames.includes('project-link-label">Notebook'),
    'Promoting two actions preserves the embedded demo and supporting resources.');

  const musicIntro = intro(renderProjectPage(project('sheetMusicUpscale')));
  assert(musicIntro.includes('href="/documents/Project_10_pdf.zip"') && /Reports · ZIP · [\d.]+ (?:KB|MB)/.test(musicIntro)
    && !musicIntro.includes('href="/documents/Project_10.zip"'),
  'A report archive is identified as ZIP and is not confused with the notebook archive.');

  const externalIntro = intro(renderProjectPage(project('smartSentence')));
  assert(/project-intro-action--report" href="https:[^"]+\.pdf" target="_blank" rel="noopener noreferrer"/.test(externalIntro),
    'External reports retain explicit safe new-tab behavior.');
  const starContent = {
    problem: 'Readers need to inspect the project deliverable.',
    task: 'Provide the available demo and report entry points.',
    actions: ['Expose the supplied resources in the project introduction.'],
    results: ['Readers can open the available project resources.']
  };
  const dashboardIntro = intro(renderProjectPage({ ...starContent, id: 'dashboard-fixture', title: 'Dashboard', embed: { type: 'tableau', base: 'https://public.tableau.com/views/Example/Main' } }));
  assert(dashboardIntro.includes('href="https://public.tableau.com/views/Example/Main?:showVizHome=no&amp;:embed=y"')
    && dashboardIntro.includes('>Open dashboard</a>'),
  'Tableau projects promote the actual dashboard URL with its required parameters.');
  const previewIntro = intro(renderProjectPage({ ...starContent, id: 'preview-fixture', title: 'Preview', resources: [{ label: 'Notebook', url: '/documents/example.zip' }] }));
  assert(!previewIntro.includes('project-intro-actions'),
    'Projects without an actual demo or report do not get invented or empty actions.');

  const privacy = read('pages/privacy.html');
  const privacyMain = privacy.match(/<main\b[^>]*>([\s\S]*?)<\/main>/)?.[1] || '';
  assert(privacyMain.includes('class="privacy-intro"') && privacyMain.includes('class="policy-card privacy-article"')
    && privacyMain.includes('class="privacy-rail"'),
  'Privacy separates its introduction, readable article, and one preferences/navigation rail.');
  assert(privacyMain.indexOf('class="privacy-preferences-jump"') < privacyMain.indexOf('id="optional-analytics"'),
    'The direct preferences action precedes the policy text on small screens.');
  const idCounts = new Map();
  for (const match of privacyMain.matchAll(/\bid="([^"]+)"/g)) idCounts.set(match[1], (idCounts.get(match[1]) || 0) + 1);
  assert([...idCounts.values()].every((count) => count === 1), 'Moving preferences creates no duplicate IDs.');
  const shortcuts = [...privacyMain.matchAll(/href="(\/privacy#[^"]+)"/g)].map((match) => match[1]);
  assert(shortcuts.length === 8, 'Privacy has an early preferences jump and seven section shortcuts.');
  for (const href of shortcuts) {
    const url = new URL(href, 'https://www.danielshort.me/');
    assert(url.pathname === '/privacy' && idCounts.get(url.hash.slice(1)) === 1,
      `${href} stays on Privacy despite the root base URL and reaches one real section.`);
  }
  for (const id of ['privacy-preferences-form', 'save-privacy-preferences', 'privacy-preferences-status']) {
    assert(idCounts.get(id) === 1, `${id} remains a single consent API control.`);
  }
  for (const pref of ['necessary', 'analytics', 'functional', 'advertising']) {
    assert((privacyMain.match(new RegExp(`class="pref-toggle" data-pref="${pref}"`, 'g')) || []).length === 1,
      `${pref} has exactly one actual preference toggle.`);
    assert(idCounts.get(`pref-desc-${pref}`) === 1 && privacyMain.includes(`aria-controls="pref-desc-${pref}"`),
      `${pref} keeps its existing expandable explanation.`);
  }
  assert(/data-pref="necessary"[^>]*data-locked="true" disabled/.test(privacyMain),
    'Necessary cookies remain locked on.');
  const wrapped = wrapPersonalAccordionHtml(privacy, { category: 'about', view: 'detail', itemId: 'privacy', backHref: '/#about', fit: 'viewport' });
  assert(wrapped.includes('class="privacy-layout"') && wrapped.includes('class="privacy-rail"')
    && (wrapped.match(/id="privacy-preferences-form"/g) || []).length === 1,
  'The authoritative personal-shell pass preserves the new content and one preferences form.');
}

if (require.main === module) {
  let checks = 0;
  runProjectPrivacyLayoutTests({ assert(condition, message) { checks += 1; if (!condition) throw new Error(message); } });
  process.stdout.write(`Project/privacy layout tests passed (${checks} checks).\n`);
}

module.exports = runProjectPrivacyLayoutTests;
