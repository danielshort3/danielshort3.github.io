'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { renderProjectPage } = require('../../build/generate-project-pages');
const { buildHomeLibraryData } = require('../../build/generate-cms-artifacts');

const ROOT = path.resolve(__dirname, '../..');
const readJson = (file) => JSON.parse(fs.readFileSync(path.join(ROOT, file), 'utf8'));
const escape = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
const descriptions = (html) => Array.from(html.matchAll(/<meta (?:name|property)="(?:description|og:description|twitter:description)" content="([^"]*)"/g), (match) => match[1]);

function runProjectContentClarityTests({ assert }) {
  const projects = fs.readdirSync(path.join(ROOT, 'content/projects'))
    .filter((file) => file.endsWith('.json'))
    .map((file) => readJson(`content/projects/${file}`))
    .filter((project) => project.published !== false);
  const byId = new Map(projects.map((project) => [project.id, project]));
  const personal = readJson('content/audiences/personal.json');
  const library = buildHomeLibraryData({ projects, audiences: [personal], pages: [], tools: [] });
  const libraryProjects = library.projects.items;

  for (const project of projects) {
    assert(typeof project.metaDescription === 'string' && project.metaDescription.length >= 80 && project.metaDescription.length <= 160,
      `${project.id} should have a concise authored search description`);
    assert(!project.metaDescription.includes('.:') && !project.metaDescription.endsWith('…'),
      `${project.id} should use complete edited sentences rather than stitched or truncated copy`);
    const relatedProject = byId.get(project.relatedProjectId);
    assert(relatedProject && relatedProject.id !== project.id, `${project.id} should recommend a different published project`);
    const html = renderProjectPage(project, { relatedProject });
    assert(JSON.stringify(descriptions(html)) === JSON.stringify(Array(3).fill(escape(project.metaDescription))),
      `${project.id} should use the same authored description in search, Open Graph, and Twitter metadata`);
    const structured = JSON.parse(html.match(/<script type="application\/ld\+json">\s*([\s\S]*?)\s*<\/script>/)[1]);
    assert(structured['@graph'].find((entry) => entry['@type'] === 'CreativeWork').description === project.metaDescription,
      `${project.id} structured data should describe the same project as its social and search metadata`);
    assert(libraryProjects.find((item) => item.id === project.id).summary === project.subtitle,
      `${project.id} library card should use its benefit-oriented subtitle`);
    const next = html.match(/<nav class="project-next-steps"[^>]*>([\s\S]*?)<\/nav>/)[1];
    assert((next.match(/class="project-next-link"/g) || []).length === 1 &&
      next.includes(`href="/portfolio/${relatedProject.id}"`) && next.includes(escape(relatedProject.title)),
    `${project.id} should provide exactly one correctly named curated next project`);
    assert(next.includes('href="/contact" data-contact-modal-link="true"') &&
      next.includes(`data-contact-message="Hi Daniel, I have a question about ${escape(project.title)}:`),
    `${project.id} should retain a native contact fallback and provide the project name to the existing prefill flow`);

    const evidence = html.match(/<section class="project-evidence"[^>]*>([\s\S]*?)<\/section>/)?.[1];
    if (!project.evaluation) {
      assert(!evidence, `${project.id} should not invent an evaluation section without authored evaluation data`);
      continue;
    }
    assert(evidence && /<details class="project-evidence-details">/.test(evidence),
      `${project.id} should keep supporting evaluation detail closed initially with native disclosure semantics`);
    assert(evidence.includes(escape(project.notes || project.evaluation.limitations[0])),
      `${project.id} should expose relevant context even before the disclosure is opened`);
    for (const field of ['dataset', 'split', 'baseline', 'decision']) {
      assert(evidence.includes(escape(project.evaluation[field])), `${project.id} should retain authored evaluation ${field}`);
    }
    for (const metric of project.evaluation.metrics) {
      assert(evidence.includes(escape(metric.label)) && evidence.includes(escape(metric.value)) && evidence.includes(escape(metric.context)),
        `${project.id} should retain every metric and its interpretation, including weak results`);
    }
    for (const limitation of project.evaluation.limitations) {
      assert(evidence.includes(escape(limitation)), `${project.id} should retain every authored limitation`);
    }
    assert(evidence.includes(`href="${escape(project.evaluation.evidence.url)}"`), `${project.id} should link to its actual evidence`);
  }

  const fixture = { ...byId.get('handwritingRating'), metaDescription: undefined, subtitle: 'One clear sentence.', problem: 'This should remain in the case study.' };
  assert(descriptions(renderProjectPage(fixture))[0] === 'One clear sentence.', 'Metadata fallback should choose one clear source instead of joining complete sentences with a colon');
  const unsafe = { ...fixture, evaluation: { ...fixture.evaluation, evidence: { label: 'Unsafe', url: 'javascript:alert(1)' } } };
  assert(!renderProjectPage(unsafe).includes('href="javascript:'), 'Evidence links should reject executable URLs');
  const literal = { ...fixture, metaDescription: 'A <literal> & "quoted" description.' };
  assert(descriptions(renderProjectPage(literal))[0] === escape(literal.metaDescription), 'Authored metadata should be escaped without changing its meaning');
  const homeProjects = personal.page.sections.find((section) => section.type === 'home-accordion').props.categories.find((category) => category.id === 'projects');
  assert(!homeProjects.items.find((item) => item.id === 'handwritingRating').summary.includes('objective'),
    'The homepage should not promise an objective measure of general handwriting legibility');
}

module.exports = { runProjectContentClarityTests };

if (require.main === module) {
  let count = 0;
  runProjectContentClarityTests({ assert(condition, message) { count += 1; if (!condition) throw new Error(message); } });
  process.stdout.write(`Project content clarity tests passed (${count} checks).\n`);
}
