'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');
const {
  flipValue,
  generateDarkCore,
  isLightSurfaceColor,
  scopeSelector,
  targetFiles
} = require('../../build/gen-dark-core.cjs');

const root = path.resolve(__dirname, '..', '..');
const outputPath = path.join(root, 'css', 'components', 'dark-core.css');
const corePrefix = /^html\[data-theme-scope="core"\](?=$|[\s.#:>+~]|\[)/i;

function splitTopLevelSelectorList(selector) {
  const branches = [];
  let start = 0;
  let parenDepth = 0;
  let bracketDepth = 0;
  let quote = '';
  let escaped = false;
  let inComment = false;

  for (let index = 0; index < selector.length; index += 1) {
    const char = selector[index];
    if (inComment) {
      if (char === '*' && selector[index + 1] === '/') {
        inComment = false;
        index += 1;
      }
      continue;
    }
    if (escaped) {
      escaped = false;
      continue;
    }
    if (char === '\\') {
      escaped = true;
      continue;
    }
    if (quote) {
      if (char === quote) quote = '';
      continue;
    }
    if (char === '/' && selector[index + 1] === '*') {
      inComment = true;
      index += 1;
      continue;
    }
    if (char === '"' || char === "'") {
      quote = char;
      continue;
    }
    if (char === '(') parenDepth += 1;
    else if (char === ')') parenDepth -= 1;
    else if (char === '[') bracketDepth += 1;
    else if (char === ']') bracketDepth -= 1;
    else if (char === ',' && parenDepth === 0 && bracketDepth === 0) {
      branches.push(selector.slice(start, index).trim());
      start = index + 1;
    }
  }

  branches.push(selector.slice(start).trim());
  return branches.filter(Boolean);
}

function collectRuleSelectors(css) {
  const selectors = [];

  function walk(source) {
    let position = 0;

    function skipWhitespaceAndComments() {
      while (position < source.length) {
        if (/\s/.test(source[position])) {
          position += 1;
          continue;
        }
        if (source.startsWith('/*', position)) {
          const end = source.indexOf('*/', position + 2);
          position = end === -1 ? source.length : end + 2;
          continue;
        }
        break;
      }
    }

    while (position < source.length) {
      skipWhitespaceAndComments();
      const headStart = position;
      let quote = '';
      let escaped = false;
      let parenDepth = 0;
      let bracketDepth = 0;

      while (position < source.length) {
        const char = source[position];
        if (escaped) escaped = false;
        else if (char === '\\') escaped = true;
        else if (quote) {
          if (char === quote) quote = '';
        } else if (char === '"' || char === "'") quote = char;
        else if (char === '(') parenDepth += 1;
        else if (char === ')') parenDepth -= 1;
        else if (char === '[') bracketDepth += 1;
        else if (char === ']') bracketDepth -= 1;
        else if ((char === '{' || char === ';') && parenDepth === 0 && bracketDepth === 0) break;
        position += 1;
      }

      const head = source.slice(headStart, position).trim();
      if (!head || position >= source.length) break;
      if (source[position] === ';') {
        position += 1;
        continue;
      }

      position += 1;
      const bodyStart = position;
      let depth = 1;
      quote = '';
      escaped = false;
      while (position < source.length && depth > 0) {
        const char = source[position];
        if (escaped) escaped = false;
        else if (char === '\\') escaped = true;
        else if (quote) {
          if (char === quote) quote = '';
        } else if (char === '"' || char === "'") quote = char;
        else if (source.startsWith('/*', position)) {
          const end = source.indexOf('*/', position + 2);
          position = end === -1 ? source.length : end + 2;
          continue;
        } else if (char === '{') depth += 1;
        else if (char === '}') depth -= 1;
        position += 1;
      }

      const body = source.slice(bodyStart, position - 1);
      if (/^@(layer|media|supports)\b/i.test(head)) walk(body);
      else if (!head.startsWith('@')) selectors.push(head);
    }
  }

  walk(css);
  return selectors;
}

const check = spawnSync(process.execPath, ['build/gen-dark-core.cjs', '--check'], {
  cwd: root,
  encoding: 'utf8'
});
assert.strictEqual(check.status, 0, `${check.stdout}\n${check.stderr}`.trim());

const generated = generateDarkCore().css;
const committed = fs.readFileSync(outputPath, 'utf8');
assert.strictEqual(committed, generated, 'dark-core.css must match the current generator output');

const selectors = collectRuleSelectors(committed);
assert(selectors.length > 100, 'expected a substantial generated dark-mode stylesheet');
selectors.forEach((selector) => {
  splitTopLevelSelectorList(selector).forEach((branch) => {
    assert(corePrefix.test(branch), `generated dark-mode selector branch is not core-scoped: ${branch}`);
  });
});

const selectorFixture = 'html.site-realm-professional, body:is(.alpha, .beta), a[data-label="one,two"]';
assert.strictEqual(
  scopeSelector(selectorFixture),
  'html[data-theme-scope="core"].site-realm-professional,\n'
    + '    html[data-theme-scope="core"] body:is(.alpha, .beta),\n'
    + '    html[data-theme-scope="core"] a[data-label="one,two"]',
  'selector scoping must merge html attributes and preserve nested commas'
);

assert.strictEqual(
  flipValue('var(--brand-midnight, #091f3b)', 'text'),
  'var(--text-light, #E4EBF5)',
  'brand midnight must become a semantic light text token in text declarations'
);
assert.strictEqual(
  flipValue('var(--brand-deep-blue, #0145c8)', 'text'),
  'var(--link, #6AA8FF)',
  'brand deep blue must become the accessible dark-theme link token in text declarations'
);
assert.strictEqual(
  flipValue('var(--brand-midnight, #091f3b)', 'surface'),
  'var(--brand-midnight, #091f3b)',
  'brand midnight must remain available as a surface primitive'
);
assert.strictEqual(isLightSurfaceColor('#eef2f7'), true, 'near-white neutral surfaces must be detected by luminance');
assert.strictEqual(isLightSurfaceColor('#d97706'), false, 'bright chromatic accents must not be treated as neutral surfaces');

const toolsBundle = fs.readFileSync(path.join(root, 'css', 'styles-tools.css'), 'utf8');
const toolsImports = Array.from(toolsBundle.matchAll(/@import\s+url\(["']([^"']+)["']\)/g))
  .map((match) => path.posix.join('css', match[1].replace(/\\/g, '/')));
toolsImports.forEach((source) => {
  assert(targetFiles.includes(source), `dark-mode generator is missing tools bundle source: ${source}`);
});
assert.strictEqual(new Set(targetFiles).size, targetFiles.length, 'dark-mode source targets must not contain duplicates');
targetFiles.forEach((source) => {
  assert(fs.existsSync(path.join(root, source)), `dark-mode source target does not exist: ${source}`);
});

assert(committed.includes('html[data-theme-scope="core"] .tools-account-bar'), 'tools account styles must be generated');
assert(committed.includes('html[data-theme-scope="core"] .tools-resume-panel'), 'tools workspace styles must be generated');
assert(!/color\s*:[^;]*(?:--brand-midnight|--brand-deep-blue)/i.test(committed), 'dark output must not use dark brand primitives as text');

console.log(`Dark-core generator tests passed: ${selectors.length} emitted rules are fully scoped.`);
