/* Source of truth for css/components/dark-core.css (dark-mode rules for
 * pages that opt in via html[data-theme-scope="core"]).
 * Regenerate with:  node build/gen-dark-core.cjs
 * The output is imported last in css/styles.css and must stay committed.
 *
 * Invariant (avoids the "invert white text on a colored button" bug):
 *   – surface properties  (background, border, box-shadow, and custom
 *     tokens whose names read as surfaces such as -soft / -surface / -bg /
 *     -shadow)
 *       LIGHT hard-coded colors   → dark equivalent
 *       dark / semi-transparent   → left alone (already fine on a dark page)
 *   – text properties       (color, caret-color, and custom tokens whose
 *     names read as text such as -ink / -muted / -label / -text)
 *       DARK-INK hard-coded      → light equivalent
 *       WHITE                    → never inverted (always fine on colored
 *                                  or dark surfaces)
 *
 * The file is emitted inside @layer overrides (the last cascade layer) and
 * is wrapped in @media (prefers-color-scheme: dark).  Every selector is
 * prefixed with html[data-theme-scope="core"] so it only applies to pages
 * that opt in with that attribute.
 */
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');

/* ── Parse helpers ──────────────────────────────────────────────────────── */

// Tokenize a CSS string into a list of flat rules.
// Each rule = { selector: string, decls: [{name,value}], context: string[] }
// context is the list of enclosing @media / @supports conditions (innermost last).
function extractRules(css, context) {
  context = context || [];
  const rules = [];
  let pos = 0;
  const n = css.length;

  function skipWs() {
    while (pos < n) {
      if (/\s/.test(css[pos])) { pos++; }
      else if (css[pos] === '/' && pos + 1 < n && css[pos + 1] === '*') {
        const end = css.indexOf('*/', pos + 2);
        pos = end === -1 ? n : end + 2;
      } else break;
    }
  }

  function readHead() {
    const start = pos;
    while (pos < n && css[pos] !== '{' && css[pos] !== ';') {
      if (css[pos] === '/' && pos + 1 < n && css[pos + 1] === '*') {
        const end = css.indexOf('*/', pos + 2);
        pos = end === -1 ? n : end + 2;
      } else pos++;
    }
    return css.slice(start, pos).trim();
  }

  function readBlock() {
    // pos is right after the opening {
    const start = pos;
    let depth = 1;
    while (pos < n && depth > 0) {
      if (css[pos] === '/' && pos + 1 < n && css[pos + 1] === '*') {
        const end = css.indexOf('*/', pos + 2);
        pos = end === -1 ? n : end + 2;
        continue;
      }
      if (css[pos] === '{') depth++;
      else if (css[pos] === '}') depth--;
      pos++;
    }
    // pos is now right after the final matching }
    return css.slice(start, pos - 1);
  }

  function parseDecls(body) {
    body = body.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\s+/g, ' ');
    const decls = [];
    for (const raw of body.split(';')) {
      const s = raw.trim();
      if (!s) continue;
      const ci = s.indexOf(':');
      if (ci === -1) continue;
      const name = s.slice(0, ci).trim();
      const value = s.slice(ci + 1).trim();
      if (name && value) decls.push({ name, value });
    }
    return decls;
  }

  while (true) {
    skipWs();
    if (pos >= n) break;
    const head = readHead();
    if (css[pos] === ';') { pos++; continue; } // @import, etc.
    if (pos >= n) break;
    pos++; // consume {

    const atMatch = head.startsWith('@') ? (head.match(/^@[\w-]+/) || [''])[0] : '';
    const body = readBlock();

    if (atMatch === '@keyframes' || atMatch === '@font-face' || atMatch === '@property') {
      continue;
    }
    if (atMatch === '@media' || atMatch === '@supports') {
      rules.push(...extractRules(body, [...context, head]));
    } else if (atMatch === '@layer') {
      rules.push(...extractRules(body, context));
    } else if (!head.startsWith('@')) {
      const decls = parseDecls(body);
      if (decls.length > 0) {
        rules.push({ selector: head, decls, context: [...context] });
      }
    }
  }
  return rules;
}

/* ── Color helpers ──────────────────────────────────────────────────────── */

function parseColor(t) {
  t = t.trim();
  if (/^#[0-9a-fA-F]{3,8}$/.test(t)) {
    let h = t.slice(1).toLowerCase();
    if (h.length === 3) h = h.split('').map(c => c + c).join('');
    if (h.length === 8) h = h.slice(0, 6);
    return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
  }
  const m = t.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
  if (m) return { r: +m[1], g: +m[2], b: +m[3] };
  return null;
}

const isLightHex   = (h) => { const c = parseColor(h); return !!(c && c.r >= 240 && c.g >= 240 && c.b >= 240); };
const isDarkInkHex = (h) => { const c = parseColor(h); return !!(c && c.r < 140  && c.g < 140  && c.b < 160); };
const isLightRgba  = (s) => { const m = s.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/); if (!m) return false; return +m[1] >= 240 && +m[2] >= 240 && +m[3] >= 240; };
const isDarkInkRgba= (s) => { const m = s.match(/^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/); if (!m) return false; return +m[1] < 140 && +m[2] < 140 && +m[3] < 160; };

const DARK_SURF  = '#121E30';
const DARK_RGBA  = 'rgba(10,16,28';
const LIGHT_TXT  = '#E4EBF5';
const LIGHT_RGBA = 'rgba(228,235,245';

/* ── Property role detection ────────────────────────────────────────────── */

const SURF_PROP = /^(background|background-color|border|border-color|border-top-color|border-bottom-color|border-left-color|border-right-color|border-top|border-bottom|border-left|border-right|box-shadow|outline|outline-color|text-shadow)$/i;
const TEXT_PROP = /^(color|caret-color|fill)$/i;
const PROP_SURFACE_NAME = /-(soft|surface|bg|tint|panel|chip|field|shadow|line|line-strong)$/i;
const PROP_TEXT_NAME    = /-(ink|muted|text|label|caption|title|heading|body|fg|line-ink|sub|hint|desc)$/i;

function roleOf(propName) {
  if (propName.startsWith('--')) {
    if (PROP_SURFACE_NAME.test(propName)) return 'surface';
    if (PROP_TEXT_NAME.test(propName)) return 'text';
    return null;
  }
  if (SURF_PROP.test(propName)) return 'surface';
  if (TEXT_PROP.test(propName)) return 'text';
  return null;
}

/* ── Flip a single value according to role ─────────────────────────────── */

function flipValue(val, role) {
  if (role === null) return val;

  let v = val;

  // hex colors (3-8 digit)
  v = v.replace(/#[0-9a-fA-F]{3,8}/g, (m) => {
    if (role === 'surface' && isLightHex(m)) return DARK_SURF;
    if (role === 'text'    && isDarkInkHex(m)) return LIGHT_TXT;
    return m;
  });

  // rgba() / rgb()
  v = v.replace(/rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*[\d.]+\s*)?\)/g, (m) => {
    const aMatch = m.match(/,\s*([\d.]+)\s*\)$/);
    const alpha = aMatch ? ',' + aMatch[1] : '';
    if (role === 'surface' && isLightRgba(m)) return DARK_RGBA + alpha + ')';
    if (role === 'text'    && isDarkInkRgba(m)) return LIGHT_RGBA + alpha + ')';
    return m;
  });

  return v;
}

/* ── Determine whether a rule is worth emitting ─────────────────────────── */

function worthProcessing(rule) {
  for (const d of rule.decls) {
    const role = roleOf(d.name);
    if (role === null) continue;
    const flipped = flipValue(d.value, role);
    if (flipped !== d.value) return true;
  }
  return false;
}

/* ── Scope a selector for core pages ───────────────────────────────────── */

function scopeSelector(sel) {
  // If the selector already starts with html, keep it as-is.
  if (/^\s*html\b/.test(sel)) return sel;
  // Add html[data-theme-scope="core"] as a prefix (descendant selector).
  return 'html[data-theme-scope="core"] ' + sel;
}

/* ── Main: process all target files ─────────────────────────────────────── */

const targetFiles = [
  'css/components/home-project-graph.css',
  'css/components/home-proof.css',
  'css/components/projects.css',
  'css/components/tools.css',
  'css/components/search.css',
  'css/components/sitemap-page.css',
  'css/components/contact-card.css',
  'css/components/certification.css',
  'css/layout/footer.css',
  'css/components/buttons.css',
  'css/components/core.css',
  'css/components/mobile-site-dock.css',
  'css/components/work-experience.css',
  'css/components/privacy-page.css',
  'css/components/portfolio-workbench.css',
  'css/components/project-page.css',
  'css/components/project-demo-theme.css',
  'css/utilities/design-system-overrides.css',
  'css/utilities/helpers.css',
  'css/utilities/typography.css',
  'css/components/skills.css',
  'css/components/speed-dial.css',
  'css/components/doc-card.css',
  'css/components/cms-map.css',
  'css/components/audience-gateway.css',
  'css/components/page-transitions.css',
  'css/components/cookie-settings.css',
  'css/components/home-scroll.css',
  'css/components/destination-analytics.css'
];

const allRules = [];
const fileReports = [];

for (const rel of targetFiles) {
  const abs = path.join(root, rel);
  if (!fs.existsSync(abs)) {
    fileReports.push([rel, 0, 'MISSING']);
    continue;
  }
  const css = fs.readFileSync(abs, 'utf8');
  const all = extractRules(css, []);
  const useful = all.filter(worthProcessing);

  for (const rule of useful) {
    const scoped = scopeSelector(rule.selector);
    // Collect only the declarations that would flip
    const flippedDecls = [];
    for (const d of rule.decls) {
      const role = roleOf(d.name);
      if (role === null) continue;
      const newval = flipValue(d.value, role);
      if (newval !== d.value) flippedDecls.push({ name: d.name, value: newval });
    }
    if (flippedDecls.length === 0) continue;
    allRules.push({ selector: scoped, decls: flippedDecls, context: rule.context, file: rel });
  }
  fileReports.push([rel, useful.length, null]);
}

/* ── Manual overrides (hard to auto-flip correctly) ─────────────────────── */

const manualOverrides = [
  {
    // hero.css:247 — color:var(--brand-cloud) on a dark hero; token flips to dark → invisible
    selector: 'html[data-theme-scope="core"] .hero-proof-row strong',
    decls: [{ name: 'color', value: '#F0F4FA' }],
    context: [],
  },
  {
    // home-proof.css:80 — same
    selector: 'html[data-theme-scope="core"] .professional-hero-proof-link strong',
    decls: [{ name: 'color', value: '#F0F4FA' }],
    context: [],
  },
];

/* ── Emit CSS ───────────────────────────────────────────────────────────── */

let css = '';
css += '/* Dark-mode overrides for pages that opt in with data-theme-scope="core" */\n';
css += '/* Generated by build/gen-dark-core.cjs — do not edit manually. */\n';
css += '/* ' + allRules.length + ' auto-flipped rules + ' + manualOverrides.length + ' manual overrides */\n';
css += '\n';

// Group by context to reduce nesting complexity
const emitAll = [];
for (const r of allRules) emitAll.push(r);
for (const r of manualOverrides) emitAll.push(r);

function emitFull(rules) {
  let out = '@layer overrides {\n';
  out += '  @media (prefers-color-scheme: dark) {\n';
  for (const r of rules) {
    const ctxs = r.context || [];
    let body = '    ' + r.selector + ' {\n';
    for (const d of r.decls) body += '      ' + d.name + ': ' + d.value + ';\n';
    body += '    }\n';
    for (let i = ctxs.length - 1; i >= 0; i--) {
      body = '    ' + ctxs[i].trim() + ' {\n' + body + '    }\n';
    }
    out += body + '\n';
  }
  out += '  }\n';
  out += '}\n';
  return out;
}

css += emitFull(emitAll);
css += '\n';

const outPath = path.join(root, 'css/components/dark-core.css');
fs.writeFileSync(outPath, css, 'utf8');

console.log('--- per-file report ---');
for (const [f, n, err] of fileReports) {
  console.log(f.padEnd(52), err || (n + ' rules'));
}
console.log('---');
console.log('Total auto-flipped rules:', allRules.length);
console.log('Manual overrides:', manualOverrides.length);
console.log('Output size:', fs.statSync(outPath).size, 'bytes');
