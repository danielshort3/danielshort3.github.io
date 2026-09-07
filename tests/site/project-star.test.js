'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { renderProjectPage } = require('../../build/generate-project-pages');

const ROOT = path.resolve(__dirname, '../..');
const LABELS = ['Situation', 'Task', 'Action', 'Result'];
const normalize = (value) => String(value).replace(/\s+/g, ' ').trim();
const sentence = (value) => /[.!?]$/.test(normalize(value)) ? normalize(value) : `${normalize(value)}.`;
const textContent = (html) => normalize(html.replace(/<[^>]*>/g, '')
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&'));
const starRows = (html) => Array.from(html.matchAll(/<dt class="project-star-label">([^<]+)<\/dt>\s*<dd class="project-star-value">([\s\S]*?)<\/dd>/g),
  (match) => ({ label: match[1], html: match[2] }));

function assertProjectStarSummary({ assert, html, project, file }) {
  const rows = starRows(html);
  assert(JSON.stringify(rows.map((row) => row.label)) === JSON.stringify(LABELS),
    `${file} should render Situation, Task, Action, and Result exactly once in order`);
  for (const [field, label] of [['problem', 'Situation'], ['task', 'Task']]) {
    assert(typeof project[field] === 'string' && project[field].trim().length > 0,
      `${project.id} should supply authored ${label} content`);
    assert(textContent(rows.find((row) => row.label === label).html) === sentence(project[field]),
      `${file} should show the complete authored ${label} text`);
  }
  for (const [field, label] of [['actions', 'Action'], ['results', 'Result']]) {
    assert(Array.isArray(project[field]) && project[field].length > 0 &&
      project[field].every((item) => typeof item === 'string' && item.trim().length > 0),
    `${project.id} should supply populated ${label} bullets`);
    const row = rows.find((entry) => entry.label === label);
    const bullets = Array.from(row.html.matchAll(/<li>([\s\S]*?)<\/li>/g), (match) => textContent(match[1]));
    assert(JSON.stringify(bullets) === JSON.stringify(project[field].map(normalize)),
      `${file} should preserve every authored ${label} bullet in order`);
  }
  assert(!html.includes('Owned the end-to-end build, from implementation through the final deliverable.'),
    `${file} should not replace its Task with a generic ownership sentence`);
}

function runProjectStarRendererTests({ assert }) {
  const fixture = {
    id: 'star-fixture',
    title: 'STAR regression fixture',
    problem: '  Editors need a complete <brief> & "summary"  ',
    task: "  Preserve the user's <scope> & \"goals\"  ",
    role: ['Legacy role text must not replace the explicit objective.'],
    actions: [
      '  Gather the <source> & "notes".  ',
      'Check the authored objective.',
      'Render the project narrative.',
      'Retain the fourth action.',
      'Verify the fifth action.'
    ],
    results: [
      "  The author's <evidence> & \"outcomes\" remain available.  ",
      'Readers can review each step.',
      'The summary includes its objective.',
      'The fourth result remains visible.',
      'The fifth result remains visible.'
    ]
  };
  const html = renderProjectPage(fixture);
  assertProjectStarSummary({ assert, html, project: fixture, file: fixture.id });
  const starHtml = html.match(/<section class="project-star"[^>]*>([\s\S]*?)<\/section>/)?.[1] || '';
  assert(starHtml.includes('&lt;brief&gt; &amp; &quot;summary&quot;') &&
    starHtml.includes('user&#39;s &lt;scope&gt; &amp; &quot;goals&quot;') &&
    starHtml.includes('&lt;source&gt; &amp; &quot;notes&quot;') &&
    starHtml.includes('author&#39;s &lt;evidence&gt; &amp; &quot;outcomes&quot;'),
  'All four STAR fields should escape HTML and preserve literal authored text');
  assert(!starHtml.includes(fixture.role[0]), 'Task should use the explicit objective instead of the legacy role');
  const withoutRole = { ...fixture };
  delete withoutRole.role;
  assertProjectStarSummary({ assert, html: renderProjectPage(withoutRole), project: withoutRole, file: 'project without role' });

  for (const [field, label] of [['problem', 'Situation'], ['task', 'Task'], ['actions', 'Action'], ['results', 'Result']]) {
    const invalidValues = field === 'problem' || field === 'task'
      ? [undefined, null, '', ' \t\n ', [], {}, 0]
      : [undefined, null, [], ['', ' \t\n '], [null, {}, 0], 'A list is required.'];
    for (const invalid of invalidValues) {
      const incomplete = { ...fixture, [field]: invalid };
      if (invalid === undefined) delete incomplete[field];
      let failure;
      try { renderProjectPage(incomplete); } catch (error) { failure = error; }
      assert(failure instanceof Error && failure.message.includes(fixture.id) && failure.message.includes(label),
        `A missing or invalid ${label} should stop rendering and identify the project and STAR field`);
    }
  }

  const mixedLists = {
    ...fixture,
    actions: ['', null, ...fixture.actions, ' \t '],
    results: [' \n ', ...fixture.results, {}, '']
  };
  assertProjectStarSummary({ assert, html: renderProjectPage(mixedLists), project: fixture, file: 'normalized STAR lists' });
}

if (require.main === module) {
  let checks = 0;
  const assert = (condition, message) => { checks += 1; if (!condition) throw new Error(message); };
  runProjectStarRendererTests({ assert });
  if (!process.argv.includes('--renderer-only')) {
    const projects = fs.readdirSync(path.join(ROOT, 'content/projects'))
      .filter((file) => file.endsWith('.json'))
      .map((file) => JSON.parse(fs.readFileSync(path.join(ROOT, 'content/projects', file), 'utf8')))
      .filter((project) => project.published !== false);
    for (const project of projects) {
      for (const directory of ['pages/portfolio', ...['analytics', 'data-science', 'tourism'].map((audience) => `pages/professional/${audience}/portfolio`)]) {
        const file = `${directory}/${project.id}.html`;
        assertProjectStarSummary({ assert, html: fs.readFileSync(path.join(ROOT, file), 'utf8'), project, file });
      }
    }
  }
  process.stdout.write(`Project STAR tests passed (${checks} checks).\n`);
}

module.exports = { assertProjectStarSummary, runProjectStarRendererTests };
