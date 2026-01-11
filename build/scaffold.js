#!/usr/bin/env node
'use strict';

/*
  Minimal scaffolding helper for new pages/tools/projects.
  No external deps; intended for developer convenience.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const templatesDir = path.join(__dirname, 'templates');

function log(line) {
  process.stdout.write(String(line || '') + '\n');
}

function logError(line) {
  process.stderr.write(String(line || '') + '\n');
}

function die(message, code = 1) {
  logError(message);
  process.exit(code);
}

function hasFlag(args, flag) {
  return args.includes(flag);
}

function readFlagValue(args, flag) {
  const idx = args.indexOf(flag);
  if (idx < 0) return '';
  return String(args[idx + 1] || '').trim();
}

function isSafeSlug(value) {
  const slug = String(value || '').trim();
  if (!slug) return false;
  if (slug.length > 80) return false;
  return /^[a-z0-9][a-z0-9-]*$/.test(slug);
}

function readTemplate(name) {
  const abs = path.join(templatesDir, name);
  if (!fs.existsSync(abs)) die(`Missing template: ${path.relative(root, abs)}`);
  return fs.readFileSync(abs, 'utf8');
}

function applyTokens(template, tokens) {
  let out = String(template || '');
  Object.entries(tokens || {}).forEach(([key, value]) => {
    const token = `__${key}__`;
    out = out.split(token).join(String(value ?? ''));
  });
  return out;
}

function writeFile(relPath, content) {
  const abs = path.join(root, relPath);
  if (fs.existsSync(abs)) die(`Refusing to overwrite existing file: ${relPath}`);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, String(content ?? ''), 'utf8');
  log(`Created ${relPath}`);
}

function usage() {
  log(`
Usage:
  node build/scaffold.js tool <tool-id> "<Tool Name>" [--eyebrow "..."] [--description "..."]
  node build/scaffold.js page <slug> "<Page Title>" [--description "..."]
  node build/scaffold.js project <project-id> "<Project Title>"

Examples:
  node build/scaffold.js tool sentiment-checker "Sentiment Checker" --eyebrow "Sentiment Checker - Quick tone scan." --description "Scan text for sentiment. Runs locally in your browser."
  node build/scaffold.js page about "About" --description "Learn more about Daniel Short."
  node build/scaffold.js project demoProject "Demo Project"
`.trim());
}

function scaffoldTool(args) {
  const toolId = String(args[0] || '').trim();
  const toolTitle = String(args[1] || '').trim();
  if (!isSafeSlug(toolId)) die('Invalid tool-id. Expected kebab-case like "text-compare".');
  if (!toolTitle) die('Missing tool name.');

  const eyebrow = readFlagValue(args, '--eyebrow') || `${toolTitle} - One-line summary.`;
  const description = readFlagValue(args, '--description') || `${toolTitle} tool. Runs locally in your browser.`;
  const ogDescription = description.length > 180 ? `${description.slice(0, 177).trimEnd()}…` : description;

  const pageTemplate = readTemplate('tool-page.template.html');
  const scriptTemplate = readTemplate('tool-script.template.js');

  writeFile(
    path.join('pages', `${toolId}.html`),
    applyTokens(pageTemplate, {
      TOOL_ID: toolId,
      TOOL_TITLE: toolTitle,
      TOOL_EYEBROW: eyebrow,
      TOOL_DESCRIPTION: description,
      TOOL_OG_DESCRIPTION: ogDescription
    })
  );

  writeFile(
    path.join('js', 'tools', `${toolId}.js`),
    applyTokens(scriptTemplate, { TOOL_ID: toolId })
  );

  log('');
  log('Next steps (manual):');
  log(`- Add rewrites in \`vercel.json\`:\n  { "source": "/tools/${toolId}", "destination": "/pages/${toolId}" },\n  { "source": "/tools/${toolId}.html", "destination": "/pages/${toolId}" }`);
  log(`- Add a card in \`pages/tools.html\` (so it shows up in the tools index).`);
  log(`- Add a catalog entry in \`js/accounts/tools-account-ui.js\` (TOOL_CATALOG) so sessions show a friendly name.`);
  log(`- Run \`npm test\` and \`npm run build\`.`);
}

function scaffoldPage(args) {
  const slug = String(args[0] || '').trim();
  const title = String(args[1] || '').trim();
  if (!isSafeSlug(slug)) die('Invalid page slug. Expected kebab-case like "contact".');
  if (!title) die('Missing page title.');

  const description = readFlagValue(args, '--description') || `${title}.`;
  const template = readTemplate('page.template.html');

  writeFile(
    path.join('pages', `${slug}.html`),
    applyTokens(template, {
      PAGE_SLUG: slug,
      PAGE_TITLE: title,
      PAGE_DESCRIPTION: description
    })
  );

  log('');
  log('Next steps (manual):');
  log(`- Add rewrites in \`vercel.json\`:\n  { "source": "/${slug}", "destination": "/pages/${slug}" },\n  { "source": "/${slug}.html", "destination": "/pages/${slug}" }`);
  log(`- If the page should be indexed, update \`sitemap.xml\` (or the build script if generated).`);
  log(`- Run \`npm test\` and \`npm run build\`.`);
}

function scaffoldProject(args) {
  const projectId = String(args[0] || '').trim();
  const title = String(args[1] || '').trim();
  if (!projectId) die('Missing project-id.');
  if (!title) die('Missing project title.');

  const template = readTemplate('project-entry.template.js');
  const snippet = applyTokens(template, {
    PROJECT_ID: projectId,
    PROJECT_TITLE: title,
    PROJECT_SUBTITLE: '',
    PROJECT_IMAGE_ALT: `Preview image for ${title}`,
    PROJECT_PROBLEM: ''
  });

  log(snippet.trimEnd());
  log('');
  log('Next steps:');
  log('- Paste the snippet into `js/portfolio/projects-data.js` (window.PROJECTS).');
  log('- Add a preview image at `img/projects/<id>.png` (and optional `.webp`).');
  log('- Run `npm run build` to regenerate `pages/portfolio/<id>.html` + update `sitemap.xml`.');
}

function main() {
  const args = process.argv.slice(2);
  const command = String(args[0] || '').trim();
  const rest = args.slice(1);

  if (!command || hasFlag(rest, '--help') || hasFlag(rest, '-h')) {
    usage();
    process.exit(0);
  }

  if (command === 'tool') {
    scaffoldTool(rest);
    return;
  }
  if (command === 'page') {
    scaffoldPage(rest);
    return;
  }
  if (command === 'project') {
    scaffoldProject(rest);
    return;
  }

  logError(`Unknown command: ${command}`);
  logError('');
  usage();
  process.exit(1);
}

main();
