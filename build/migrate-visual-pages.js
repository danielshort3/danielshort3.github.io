#!/usr/bin/env node
'use strict';

/*
  Converts managed raw-body pages into visual-page documents with preserved
  legacy-html sections. This is intentionally opt-in; run it only when you are
  ready to store page sections structurally instead of as one bodyHtml string.
*/

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const targets = [
  { relDir: path.join('content', 'pages'), paths: [''] },
  { relDir: path.join('content', 'audiences'), paths: ['page'] },
  { relDir: path.join('content', 'resumes'), paths: ['digitalPage', 'pdfPage'] }
];

function readJson(absPath) {
  return JSON.parse(fs.readFileSync(absPath, 'utf8'));
}

function writeJson(absPath, value) {
  fs.writeFileSync(absPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function getAtPath(object, dottedPath) {
  if (!dottedPath) return object;
  return dottedPath.split('.').reduce((acc, key) => acc && acc[key], object);
}

function slugify(value, fallback) {
  const slug = String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug || fallback;
}

function parseAttributes(rawAttrs) {
  const attrs = {};
  String(rawAttrs || '').replace(/([:\w.-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g, (match, name, doubleQuoted, singleQuoted, bare) => {
    attrs[name] = doubleQuoted ?? singleQuoted ?? bare ?? true;
    return match;
  });
  if (!attrs.id) attrs.id = 'main';
  return attrs;
}

function parseMain(bodyHtml) {
  const source = String(bodyHtml || '');
  const mainMatch = source.match(/<main\b([^>]*)>([\s\S]*)<\/main>/i);
  if (!mainMatch) {
    return {
      attrs: { id: 'main' },
      inner: source
    };
  }
  return {
    attrs: parseAttributes(mainMatch[1]),
    inner: mainMatch[2]
  };
}

function splitTopLevelElements(innerHtml) {
  const source = String(innerHtml || '');
  const chunks = [];
  const voidElements = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr']);
  const tagPattern = /<\/?([a-z][\w:-]*)(?:\s[^>]*)?>/ig;
  let depth = 0;
  let start = -1;
  let match;

  while ((match = tagPattern.exec(source))) {
    const rawTag = match[0];
    const tagName = match[1].toLowerCase();
    const isClosing = rawTag.startsWith('</');
    const isSelfClosing = rawTag.endsWith('/>') || voidElements.has(tagName);

    if (isClosing) {
      if (depth > 0) depth -= 1;
      if (depth === 0 && start >= 0) {
        chunks.push(source.slice(start, tagPattern.lastIndex).trim());
        start = -1;
      }
      continue;
    }

    if (depth === 0) start = match.index;
    if (!isSelfClosing) depth += 1;
    if (isSelfClosing && depth === 0 && start >= 0) {
      chunks.push(source.slice(start, tagPattern.lastIndex).trim());
      start = -1;
    }
  }

  return chunks.filter(Boolean);
}

function classifySection(html) {
  const raw = String(html || '');
  const classMatch = raw.match(/\bclass\s*=\s*"([^"]*)"/i);
  const className = classMatch ? classMatch[1] : '';
  if (className.includes('hero')) return 'Hero';
  if (className.includes('cta')) return 'Call to Action';
  if (className.includes('resume')) return 'Resume';
  if (className.includes('cert')) return 'Certifications';
  if (className.includes('work')) return 'Work Experience';
  if (className.includes('project')) return 'Projects';
  if (className.includes('modal')) return 'Modal';
  const tagMatch = raw.match(/^<([a-z][\w:-]*)\b/i);
  if (tagMatch && tagMatch[1].toLowerCase() === 'nav') return 'Navigation';
  return tagMatch ? tagMatch[1].replace(/^\w/, (char) => char.toUpperCase()) : 'Section';
}

function sectionLabel(html, fallback) {
  const raw = String(html || '');
  const headingMatch = raw.match(/<h[1-4]\b[^>]*>([\s\S]*?)<\/h[1-4]>/i);
  if (headingMatch) {
    const label = headingMatch[1].replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
    if (label) return label.slice(0, 80);
  }
  const rootMatch = raw.match(/^<([a-z][\w:-]*)\b([^>]*)>/i);
  const rootAttrs = rootMatch ? parseAttributes(rootMatch[2]) : {};
  if (rootAttrs['aria-label']) return String(rootAttrs['aria-label']).trim().slice(0, 80);
  if (rootAttrs.id) return String(rootAttrs.id).trim().slice(0, 80);
  return fallback;
}

function splitSections(bodyHtml) {
  const main = parseMain(bodyHtml);
  const chunks = splitTopLevelElements(main.inner);

  return {
    mainAttributes: main.attrs,
    chunks: chunks.length ? chunks : [main.inner.trim()].filter(Boolean)
  };
}

function convertPage(page, fileStem) {
  if (!page || typeof page !== 'object') return false;
  if (page.template !== 'raw-body' || typeof page.bodyHtml !== 'string') return false;

  const split = splitSections(page.bodyHtml);
  const sections = split.chunks.map((html, index) => ({
    id: `${slugify(page.id || fileStem, 'page')}-${index + 1}`,
    type: 'legacy-html',
    label: sectionLabel(html, `${classifySection(html)} ${index + 1}`),
    enabled: true,
    variant: 'default',
    props: { html }
  }));

  page.template = 'visual-page';
  page.mainAttributes = split.mainAttributes;
  page.sections = sections;
  delete page.bodyHtml;
  return true;
}

function refreshVisualPageLabels(page, fileStem) {
  if (!page || page.template !== 'visual-page' || !Array.isArray(page.sections)) return false;
  let changed = false;
  page.sections.forEach((section, index) => {
    const html = section && section.props && typeof section.props.html === 'string' ? section.props.html : '';
    if (!html) return;
    const nextLabel = sectionLabel(html, `${classifySection(html)} ${index + 1}`);
    if (nextLabel && section.label !== nextLabel) {
      section.label = nextLabel;
      if (!section.id) section.id = `${slugify(page.id || fileStem, 'page')}-${index + 1}`;
      changed = true;
    }
  });
  return changed;
}

function main() {
  let changed = 0;
  targets.forEach((target) => {
    const absDir = path.join(root, target.relDir);
    if (!fs.existsSync(absDir)) return;

    fs.readdirSync(absDir)
      .filter((name) => name.endsWith('.json'))
      .sort((a, b) => a.localeCompare(b))
      .forEach((fileName) => {
        const absPath = path.join(absDir, fileName);
        const doc = readJson(absPath);
        const fileStem = path.basename(fileName, '.json');
        let fileChanged = false;
        target.paths.forEach((pagePath) => {
          const page = getAtPath(doc, pagePath);
          const converted = convertPage(page, fileStem);
          fileChanged = converted || refreshVisualPageLabels(page, fileStem) || fileChanged;
        });
        if (fileChanged) {
          writeJson(absPath, doc);
          changed += 1;
          process.stdout.write(`[cms] Updated ${path.relative(root, absPath).replace(/\\/g, '/')}\n`);
        }
      });
  });

  process.stdout.write(`[cms] Visual page migration complete. Changed ${changed} file(s).\n`);
}

main();
