#!/usr/bin/env node
/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const templatePath = path.join(__dirname, 'templates', 'project-page.html');
const dataPath = path.join(root, 'js/portfolio/projects-data.js');
const outDir = path.join(root, 'projects');

function escapeHtml(value = '') {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeImagePath(p) {
  if (!p || typeof p !== 'string') return 'img/hero/head.jpg';
  return p.replace(/^\/+/, '');
}

function loadProjects() {
  const code = fs.readFileSync(dataPath, 'utf8');
  const context = {
    window: {},
    console
  };
  vm.createContext(context);
  vm.runInContext(code, context, { filename: dataPath });
  const projects = context.window.PROJECTS;
  if (!Array.isArray(projects)) {
    throw new Error('PROJECTS array was not defined');
  }
  return projects;
}

function render(template, project) {
  return template
    .replace(/__PROJECT_ID__/g, project.id)
    .replace(/__PROJECT_TITLE__/g, escapeHtml(project.title || 'Case Study'))
    .replace(/__PROJECT_SUBTITLE__/g, escapeHtml(project.subtitle || project.problem || 'Data science and analytics'))
    .replace(/__PROJECT_IMAGE__/g, normalizeImagePath(project.image));
}

function main() {
  const template = fs.readFileSync(templatePath, 'utf8');
  const projects = loadProjects();
  fs.mkdirSync(outDir, { recursive: true });

  // Clean previous html files to avoid stale slugs
  fs.readdirSync(outDir)
    .filter(name => name.endsWith('.html'))
    .forEach(name => fs.unlinkSync(path.join(outDir, name)));

  projects.forEach(project => {
    const html = render(template, project);
    const outFile = path.join(outDir, `${project.id}.html`);
    fs.writeFileSync(outFile, html, 'utf8');
  });
  console.log(`Generated ${projects.length} project pages in /projects`);
}

main();
