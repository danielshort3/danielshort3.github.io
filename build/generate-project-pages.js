#!/usr/bin/env node
/*
  Generate SEO-friendly, shareable project pages under /portfolio/<id>.
  - Keeps existing /portfolio?project=<id> deep links working.
  - Outputs static HTML pages in ./pages/portfolio/<id>.html
  - Updates ./sitemap.xml to include project URLs.
  No external deps.
*/
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const childProcess = require('child_process');
const crypto = require('crypto');
const { normalizePathname, loadNoindexPathnamesFromVercel } = require('./lib/seo-routing');
const { unwrapPersonalAccordionHtml } = require('./lib/personal-accordion-shell');

const root = path.resolve(__dirname, '..');
const dataFile = path.join(root, 'js', 'portfolio', 'projects-data.js');
const outDir = path.join(root, 'pages', 'portfolio');
const personalPortfolioIndexPath = path.join(root, 'pages', 'portfolio.html');
const professionalPortfolioIndexPath = path.join(root, 'pages', 'professional', 'analytics', 'portfolio.html');
const sitemapPath = path.join(root, 'sitemap.xml');
const sitemapCachePath = path.join(root, 'build', 'cache', 'sitemap-cache.json');
const SITE_ORIGIN = 'https://www.danielshort.me';
const toolsContentDir = path.join(root, 'content', 'tools');
const gamesContentFile = path.join(root, 'content', 'pages', 'games.json');

const noindexMetaCache = new Map();

function toLocPathname(loc) {
  const raw = String(loc || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw);
    if (url.origin !== SITE_ORIGIN) return '';
    return normalizePathname(url.pathname || '/');
  } catch (_) {
    return '';
  }
}

function hasNoindexMeta(relPath) {
  const safeRelPath = String(relPath || '').trim();
  if (!safeRelPath) return false;
  if (noindexMetaCache.has(safeRelPath)) return noindexMetaCache.get(safeRelPath);

  let hasNoindex = false;
  try {
    const html = fs.readFileSync(path.join(root, safeRelPath), 'utf8');
    hasNoindex = /<meta\s+[^>]*\bname="robots"[^>]*\bcontent="[^"]*noindex[^"]*"[^>]*>/i.test(html);
  } catch (_) {
    hasNoindex = false;
  }
  noindexMetaCache.set(safeRelPath, hasNoindex);
  return hasNoindex;
}

function shouldSkipSitemapEntry(options, noindexPathnames) {
  const locPathname = toLocPathname(options && options.loc);
  if (locPathname && noindexPathnames && noindexPathnames.has(locPathname)) {
    return true;
  }

  const sourceFile = String(options && options.sourceFile ? options.sourceFile : '').trim();
  if (sourceFile && hasNoindexMeta(sourceFile)) return true;
  return false;
}

function computeContentHash(relPath) {
  if (!relPath) return null;
  try {
    const abs = path.isAbsolute(relPath) ? relPath : path.join(root, relPath);
    const buf = fs.readFileSync(abs);
    return crypto.createHash('sha256').update(buf).digest('hex');
  } catch (_) {
    return null;
  }
}

function loadToolUrls() {
  const urls = new Set();
  try {
    if (!fs.existsSync(toolsContentDir)) return [];
    fs.readdirSync(toolsContentDir)
      .filter((name) => name.endsWith('.json') && !name.startsWith('.'))
      .sort()
      .forEach((name) => {
        const tool = JSON.parse(fs.readFileSync(path.join(toolsContentDir, name), 'utf8'));
        const visibility = String(tool && tool.visibility ? tool.visibility : 'public').trim().toLowerCase();
        if (visibility !== 'public' || tool.hidden || tool.noindex) return;

        const href = String(tool && tool.href ? tool.href : '').trim();
        const hrefMatch = /^\/?tools\/([^#?]+)$/i.exec(href);
        const slug = String(tool && tool.slug ? tool.slug : (hrefMatch ? hrefMatch[1] : '')).trim();
        if (!slug) return;
        urls.add(`${SITE_ORIGIN}/tools/${slug}`);
      });
  } catch (_) {}
  return [...urls].sort();
}

function loadGameEntries() {
  const entries = new Map();
  try {
    const page = JSON.parse(fs.readFileSync(gamesContentFile, 'utf8'));
    const games = Array.isArray(page && page.games) ? page.games : [];
    games
      .filter((game) => {
        const visibility = String(game && game.visibility ? game.visibility : 'public').trim().toLowerCase();
        return Boolean(game && !game.hidden && !game.noindex && visibility === 'public');
      })
      .sort((a, b) => {
        const orderA = Number.isFinite(Number(a && a.order)) ? Number(a.order) : Number.MAX_SAFE_INTEGER;
        const orderB = Number.isFinite(Number(b && b.order)) ? Number(b.order) : Number.MAX_SAFE_INTEGER;
        if (orderA !== orderB) return orderA - orderB;
        return String(a && (a.title || a.id) || '').localeCompare(String(b && (b.title || b.id) || ''));
      })
      .forEach((game) => {
        const id = String(game && game.id ? game.id : '').trim();
        const href = String(game && game.href ? game.href : (id ? `games/${id}` : '')).trim();
        const hrefMatch = /^\/?games\/([a-z0-9][a-z0-9-]*)$/i.exec(href);
        if (!hrefMatch) return;

        const slug = hrefMatch[1];
        const pathname = `/games/${slug}`;
        const nestedSource = `pages/games/${slug}.html`;
        const fallbackSource = `pages/${slug}.html`;
        const pageFile = fileExists(nestedSource)
          ? nestedSource
          : (fileExists(fallbackSource) ? fallbackSource : '');
        if (!pageFile) return;

        const loc = toAbsoluteUrl(pathname);
        if (!entries.has(loc)) {
          entries.set(loc, { loc, sourceFile: 'content/pages/games.json' });
        }
      });
  } catch (_) {}
  return [...entries.values()];
}

function formatLastmod(dateLike) {
  if (!dateLike) return null;
  if (typeof dateLike === 'string') {
    const s = dateLike.trim();
    if (/^\d{4}-\d{2}-\d{2}/.test(s)) return s.slice(0, 10);
    const asDate = new Date(s);
    if (!Number.isNaN(asDate.getTime())) return asDate.toISOString().slice(0, 10);
    return null;
  }
  if (dateLike instanceof Date && !Number.isNaN(dateLike.getTime())) {
    return dateLike.toISOString().slice(0, 10);
  }
  return null;
}

function getGitLastmod(relPath) {
  if (!relPath) return null;
  try {
    const iso = childProcess.execFileSync(
      'git',
      ['log', '-1', '--format=%cI', '--', relPath],
      { cwd: root, stdio: ['ignore', 'pipe', 'ignore'] }
    ).toString().trim();
    return formatLastmod(iso);
  } catch (_) {
    return null;
  }
}

function getFsLastmod(relPath) {
  if (!relPath) return null;
  try {
    const abs = path.isAbsolute(relPath) ? relPath : path.join(root, relPath);
    const stat = fs.statSync(abs);
    return formatLastmod(stat.mtime);
  } catch (_) {
    return null;
  }
}

function resolveLastmod(relPath) {
  return getGitLastmod(relPath) || getFsLastmod(relPath) || formatLastmod(new Date());
}

function loadSitemapCache() {
  const entries = new Map();
  try {
    if (!fs.existsSync(sitemapCachePath)) return entries;
    const raw = fs.readFileSync(sitemapCachePath, 'utf8');
    const parsed = JSON.parse(raw);
    const record = parsed && typeof parsed === 'object' ? parsed.entries : null;
    if (!record || typeof record !== 'object') return entries;
    Object.entries(record).forEach(([loc, meta]) => {
      const safeLoc = String(loc || '').trim();
      if (!safeLoc) return;
      const lastmod = formatLastmod(meta && meta.lastmod);
      const hash = meta && typeof meta.hash === 'string' ? String(meta.hash).trim().toLowerCase() : null;
      entries.set(safeLoc, { lastmod, hash });
    });
  } catch (_) {}
  return entries;
}

function writeSitemapCache(entries) {
  try {
    fs.mkdirSync(path.dirname(sitemapCachePath), { recursive: true });
    const record = {};
    [...entries.entries()]
      .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      .forEach(([loc, meta]) => {
        if (!loc) return;
        const lastmod = formatLastmod(meta && meta.lastmod);
        const hash = meta && typeof meta.hash === 'string' ? String(meta.hash).trim().toLowerCase() : null;
        record[loc] = { ...(lastmod ? { lastmod } : {}), ...(hash ? { hash } : {}) };
      });
    fs.writeFileSync(sitemapCachePath, JSON.stringify({ entries: record }, null, 2) + '\n', 'utf8');
  } catch (_) {}
}

function resolveSitemapMeta(options, previousEntries) {
  const loc = String(options?.loc || '').trim();
  const sourceFile = options?.sourceFile;
  const previous = previousEntries && loc ? previousEntries.get(loc) : null;
  const previousLastmod = previous && previous.lastmod ? formatLastmod(previous.lastmod) : null;
  const previousHash = previous && previous.hash ? String(previous.hash).trim().toLowerCase() : null;
  const currentHash = computeContentHash(sourceFile);

  if (currentHash && previousHash && currentHash === previousHash && previousLastmod) {
    return { lastmod: previousLastmod, hash: currentHash };
  }

  const changed = currentHash && previousHash && currentHash !== previousHash;
  if (changed) {
    return { lastmod: formatLastmod(new Date()), hash: currentHash };
  }

  const gitLastmod = getGitLastmod(sourceFile);
  if (gitLastmod) return { lastmod: gitLastmod, hash: currentHash };

  if (!changed && previousLastmod) {
    return { lastmod: previousLastmod, hash: currentHash };
  }

  const fsLastmod = getFsLastmod(sourceFile);
  if (fsLastmod) {
    return { lastmod: fsLastmod, hash: currentHash };
  }

  return { lastmod: previousLastmod, hash: currentHash };
}

function toSitemapUrlEntry(options, previousEntries, nextEntries, noindexPathnames) {
  const loc = String(options?.loc || '').trim();
  if (!loc) return '';
  if (shouldSkipSitemapEntry(options, noindexPathnames)) return '';
  const meta = resolveSitemapMeta({ loc, sourceFile: options?.sourceFile }, previousEntries);
  const lastmod = meta && meta.lastmod ? meta.lastmod : null;
  const hash = meta && meta.hash ? meta.hash : null;

  if (nextEntries && typeof nextEntries.set === 'function') {
    nextEntries.set(loc, { lastmod, hash });
  }

  const lines = [
    '  <url>',
    `    <loc>${loc.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</loc>`
  ];
  if (lastmod) lines.push(`    <lastmod>${lastmod}</lastmod>`);
  lines.push('  </url>');
  return lines.join('\n');
}

function isPublishedProject(project) {
  return project && project.published !== false;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function normalizeWhitespace(value) {
  return String(value ?? '').replace(/\s+/g, ' ').trim();
}

function normalizeTextArray(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.map(normalizeWhitespace).filter(Boolean);
  }
  if (typeof value === 'string') {
    const s = normalizeWhitespace(value);
    return s ? [s] : [];
  }
  return [];
}

function isDataResource(resource) {
  if (!resource || typeof resource !== 'object') return false;
  const explicitType = normalizeWhitespace(resource.type || '').toLowerCase();
  if (explicitType === 'data' || explicitType === 'dataset') return true;
  if (explicitType === 'project' || explicitType === 'general') return false;

  const label = normalizeWhitespace(resource.label || '');
  const url = String(resource.url || '').trim();
  const haystack = `${label} ${url}`.toLowerCase();
  const keywords = ['dataset', 'database', 'data source', 'datasource', 'corpus'];
  return keywords.some((keyword) => haystack.includes(keyword));
}

function toDomIdSafe(value) {
  return String(value ?? '')
    .trim()
    .replace(/[^a-z0-9_-]+/gi, '-')
    .replace(/^-+|-+$/g, '') || 'project';
}

function toMetaDescription(project) {
  const pieces = [
    project.subtitle,
    project.problem
  ]
    .map(normalizeWhitespace)
    .filter(Boolean);
  const combined = pieces.join(': ');
  if (combined.length <= 160) return combined;
  return combined.slice(0, 157).replace(/\s+\S*$/, '') + '…';
}

function toAbsoluteUrl(urlOrPath) {
  const raw = String(urlOrPath ?? '').trim();
  if (!raw) return '';
  if (/^https?:\/\//i.test(raw)) return raw;
  return `${SITE_ORIGIN}/${raw.replace(/^\/+/, '')}`;
}

function fileExists(relPath) {
  if (!relPath) return false;
  return fs.existsSync(path.join(root, relPath));
}

function buildResponsiveSrcset(base, ext, width) {
  const fullW = Number(width);
  if (!Number.isFinite(fullW) || fullW <= 0) {
    const candidate = `${base}.${ext}`;
    return fileExists(candidate) ? candidate : '';
  }
  const parts = [];
  const w640 = `${base}-640.${ext}`;
  if (fullW > 640 && fileExists(w640)) parts.push(`${w640} 640w`);
  const w960 = `${base}-960.${ext}`;
  if (fullW > 960 && fileExists(w960)) parts.push(`${w960} 960w`);
  const full = `${base}.${ext}`;
  if (fileExists(full)) parts.push(`${full} ${fullW}w`);
  return parts.join(', ');
}

function numberDataAttr(name, value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return '';
  return `data-${name}="${escapeHtml(String(Math.round(numeric)))}"`;
}

function isSameOriginEmbedUrl(src) {
  const safeSrc = String(src || '').trim();
  if (!safeSrc) return false;
  try {
    const parsed = new URL(safeSrc, SITE_ORIGIN);
    return parsed.origin === SITE_ORIGIN;
  } catch (_) {
    return !/^[a-z][a-z\d+.-]*:/i.test(safeSrc);
  }
}

function resolveEmbedFit(embed) {
  const requested = String(embed && embed.fit || '').trim().toLowerCase();
  if (['content', 'viewport', 'dashboard', 'fixed'].includes(requested)) return requested;

  const type = String(embed && embed.type || '').trim().toLowerCase();
  if (type === 'tableau') return 'dashboard';
  if (type === 'iframe') {
    return isSameOriginEmbedUrl(embed && embed.url) ? 'content' : 'viewport';
  }
  return 'fixed';
}

function renderEmbedAttrs(embed, id, extraClass = '') {
  const embedProjectId = toDomIdSafe(id);
  const fit = resolveEmbedFit(embed);
  return {
    className: `${extraClass} project-embed-${embedProjectId.toLowerCase()}`.trim(),
    attrs: [
      `data-project-embed="${escapeHtml(embedProjectId)}"`,
      `data-embed-fit="${escapeHtml(fit)}"`,
      numberDataAttr('embed-min-height', embed && embed.minHeight),
      numberDataAttr('embed-max-height', embed && embed.maxHeight)
    ].filter(Boolean).join(' ')
  };
}

function loadProjects() {
  const code = fs.readFileSync(dataFile, 'utf8');
  const context = { window: {} };
  vm.runInNewContext(code, context, { filename: dataFile });
  const projects = context.window.PROJECTS;
  if (!Array.isArray(projects) || projects.length === 0) {
    throw new Error('projects-data.js did not define window.PROJECTS');
  }
  return projects;
}

function renderProjectPager(projects, currentIndex) {
  if (!Array.isArray(projects) || projects.length < 2) return '';
  if (!Number.isInteger(currentIndex) || currentIndex < 0 || currentIndex >= projects.length) return '';

  const total = projects.length;
  const previous = projects[(currentIndex - 1 + total) % total];
  const next = projects[(currentIndex + 1) % total];

  const renderLink = (project, direction) => {
    const id = String(project?.id || '').trim();
    if (!id) return '';
    const title = normalizeWhitespace(project?.title || id);
    const href = `portfolio/${encodeURIComponent(id)}`;
    const label = direction === 'prev' ? 'Previous' : 'Next';
    const ariaLabel = `${label} project: ${title}`;

    if (direction === 'prev') {
      return `<a class="project-pager-link project-pager-prev" href="${escapeHtml(href)}" aria-label="${escapeHtml(ariaLabel)}">
        <span class="project-pager-arrow" aria-hidden="true">←</span>
        <span class="project-pager-text">
          <span class="project-pager-label">${label}</span>
          <span class="project-pager-title">${escapeHtml(title)}</span>
        </span>
      </a>`;
    }

    return `<a class="project-pager-link project-pager-next" href="${escapeHtml(href)}" aria-label="${escapeHtml(ariaLabel)}">
        <span class="project-pager-text">
          <span class="project-pager-label">${label}</span>
          <span class="project-pager-title">${escapeHtml(title)}</span>
        </span>
        <span class="project-pager-arrow" aria-hidden="true">→</span>
      </a>`;
  };

  const prevMarkup = renderLink(previous, 'prev');
  const nextMarkup = renderLink(next, 'next');

  return `<nav class="project-pager" aria-label="Project navigation">
    <div class="wrapper">
      ${prevMarkup}
      ${nextMarkup}
	    </div>
	  </nav>`;
}

function getProjectTagSet(project) {
  const tools = Array.isArray(project?.tools) ? project.tools : [];
  const concepts = Array.isArray(project?.concepts) ? project.concepts : [];
  const audiences = Array.isArray(project?.audiences) ? project.audiences : [];
  const audienceLabels = audiences.map((audience) => {
    const key = normalizeWhitespace(audience).toLowerCase();
    if (key === 'data-science') return 'Data Science';
    if (key === 'analytics') return 'Analytics';
    if (key === 'tourism') return 'Tourism';
    return '';
  }).filter(Boolean);
  const tags = [...tools, ...concepts, ...audiences, ...audienceLabels]
    .map((t) => normalizeWhitespace(t).toLowerCase())
    .filter(Boolean);
  return new Set(tags);
}

function renderRelatedProjectMedia(project) {
  const img = String(project?.image || '').trim();
  if (!img) return '';

  const title = normalizeWhitespace(project?.title || '');
  const alt = normalizeWhitespace(project?.imageAlt || title);
  const width = Number(project?.imageWidth);
  const height = Number(project?.imageHeight);
  const sizeAttr = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
    ? ` width="${width}" height="${height}"`
    : '';

  const match = img.match(/\.(png|jpe?g)$/i);
  if (!match) {
    return `<img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}" loading="lazy" decoding="async"${sizeAttr} sizes="(max-width: 960px) 92vw, 320px">`;
  }

  const base = img.replace(/\.(png|jpe?g)$/i, '');
  const avif = buildResponsiveSrcset(base, 'avif', width);
  const webp = buildResponsiveSrcset(base, 'webp', width);

  if (avif || webp) {
    return `<picture>
      ${avif ? `<source srcset="${escapeHtml(avif)}" type="image/avif">` : ''}
      ${webp ? `<source srcset="${escapeHtml(webp)}" type="image/webp">` : ''}
      <img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}" loading="lazy" decoding="async"${sizeAttr} sizes="(max-width: 960px) 92vw, 320px">
    </picture>`;
  }

  return `<img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}" loading="lazy" decoding="async"${sizeAttr} sizes="(max-width: 960px) 92vw, 320px">`;
}

function selectRelatedProjects(projects, currentIndex, desiredCount) {
  if (!Array.isArray(projects) || projects.length === 0) return [];
  const desired = Number.isFinite(desiredCount) ? Math.max(0, Math.floor(desiredCount)) : 0;
  if (desired <= 0) return [];

  const current = projects[currentIndex];
  const currentId = String(current?.id || '').trim();
  if (!current || !currentId) return [];

  const currentTags = getProjectTagSet(current);
  const scored = projects
    .map((candidate, index) => {
      const id = String(candidate?.id || '').trim();
      if (!candidate || !id) return null;
      if (index === currentIndex) return null;
      const tags = getProjectTagSet(candidate);
      let score = 0;
      currentTags.forEach((tag) => {
        if (tags.has(tag)) score += 1;
      });
      if (score <= 0) return null;
      return { project: candidate, index, score };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score || a.index - b.index);

  const selected = [];
  const usedIds = new Set([currentId]);

  scored.forEach((item) => {
    if (selected.length >= desired) return;
    const id = String(item.project?.id || '').trim();
    if (!id || usedIds.has(id)) return;
    selected.push(item.project);
    usedIds.add(id);
  });

  for (let offset = 1; selected.length < desired; offset++) {
    const before = currentIndex - offset;
    const after = currentIndex + offset;
    const indexes = [before, after].filter((i) => i >= 0 && i < projects.length);
    if (indexes.length === 0) break;
    indexes.forEach((idx) => {
      if (selected.length >= desired) return;
      const candidate = projects[idx];
      const id = String(candidate?.id || '').trim();
      if (!candidate || !id) return;
      if (usedIds.has(id)) return;
      selected.push(candidate);
      usedIds.add(id);
    });
  }

  return selected.slice(0, desired);
}

function renderRelatedProjectsSection(projects, currentIndex) {
  const related = selectRelatedProjects(projects, currentIndex, 3);
  if (!related.length) return '';

  const cards = related
    .map((p) => {
      const id = String(p?.id || '').trim();
      if (!id) return '';
      const title = normalizeWhitespace(p?.title || id);
      const subtitle = normalizeWhitespace(p?.subtitle || '');
      const href = `portfolio/${encodeURIComponent(id)}`;
      const label = `Open project: ${title}`;
      const safeSubtitle = subtitle ? `<div class="project-subtitle">${escapeHtml(subtitle)}</div>` : '';

      return `<a class="project-card project-related-card" role="listitem" href="${escapeHtml(href)}" aria-label="${escapeHtml(label)}">
        <div class="overlay"></div>
        <div class="project-text">
          <div class="project-title">${escapeHtml(title)}</div>
          ${safeSubtitle}
        </div>
        ${renderRelatedProjectMedia(p)}
      </a>`;
    })
    .filter(Boolean)
    .join('\n        ');

  return `<section class="project-section project-related" aria-label="Other projects">
      <h2 class="section-title">Other Projects</h2>
      <div class="project-related-grid" role="list">
        ${cards}
      </div>
    </section>`;
}

function renderProjectPage(project, options = {}) {
  const id = String(project.id || '').trim();
  const title = normalizeWhitespace(project.title || id);
  const subtitle = normalizeWhitespace(project.subtitle || '');
  const description = toMetaDescription(project);
  const canonicalPath = `/portfolio/${encodeURIComponent(id)}`;
  const canonicalUrl = `${SITE_ORIGIN}${canonicalPath}`;
  const ogImage = toAbsoluteUrl(project.image || 'img/hero/head.png');
  const ogImageAlt = normalizeWhitespace(project.imageAlt || `Preview image for ${title}`);

  const tools = Array.isArray(project.tools) ? project.tools : [];
  const concepts = Array.isArray(project.concepts) ? project.concepts : [];
  const actions = Array.isArray(project.actions) ? project.actions : [];
  const results = Array.isArray(project.results) ? project.results : [];
  const resources = Array.isArray(project.resources) ? project.resources : [];
  const comparisonSource = project.previewComparison && typeof project.previewComparison === 'object' && !Array.isArray(project.previewComparison)
    ? project.previewComparison
    : null;
  const comparisonStages = comparisonSource && Array.isArray(comparisonSource.stages)
    ? comparisonSource.stages.map((stage, index) => {
      const normalizedStage = stage && typeof stage === 'object' ? stage : {};
      const width = Number(normalizedStage.width);
      const height = Number(normalizedStage.height);
      const fullWidth = Number(normalizedStage.fullWidth);
      const fullHeight = Number(normalizedStage.fullHeight);
      return {
        label: normalizeWhitespace(normalizedStage.label || `Stage ${index + 1}`),
        shortLabel: normalizeWhitespace(normalizedStage.shortLabel || normalizedStage.label || `Stage ${index + 1}`),
        description: normalizeWhitespace(normalizedStage.description || ''),
        fullImage: String(normalizedStage.fullImage || '').trim(),
        fullAlt: normalizeWhitespace(normalizedStage.fullAlt || `${title} full stage ${index + 1}`),
        fullWidth: Number.isFinite(fullWidth) && fullWidth > 0 ? fullWidth : null,
        fullHeight: Number.isFinite(fullHeight) && fullHeight > 0 ? fullHeight : null,
        image: String(normalizedStage.image || '').trim(),
        alt: normalizeWhitespace(normalizedStage.alt || `${title} stage ${index + 1}`),
        width: Number.isFinite(width) && width > 0 ? width : null,
        height: Number.isFinite(height) && height > 0 ? height : null
      };
    })
    : [];
  const comparisonDimensionsMatch = comparisonStages.length === 3 && comparisonStages.every((stage) => (
    stage.image && stage.width && stage.height &&
    stage.width === comparisonStages[0].width &&
    stage.height === comparisonStages[0].height
  ));
  const comparisonFullDimensionsMatch = comparisonStages.length === 3 && comparisonStages.every((stage) => (
    stage.fullImage && stage.fullWidth && stage.fullHeight &&
    stage.fullWidth === comparisonStages[0].fullWidth &&
    stage.fullHeight === comparisonStages[0].fullHeight
  ));
  const comparisonLabelsAreUnique = new Set(comparisonStages.map((stage) => stage.label)).size === comparisonStages.length;
  const comparisonAltsAreUnique = new Set(comparisonStages.map((stage) => stage.alt)).size === comparisonStages.length;
  const comparisonFullAltsAreUnique = new Set(comparisonStages.map((stage) => stage.fullAlt)).size === comparisonStages.length;
  const cropSource = comparisonSource && comparisonSource.sourceCrop && typeof comparisonSource.sourceCrop === 'object' && !Array.isArray(comparisonSource.sourceCrop)
    ? comparisonSource.sourceCrop
    : null;
  const cropLeft = Number(cropSource && cropSource.left);
  const cropTop = Number(cropSource && cropSource.top);
  const cropWidth = Number(cropSource && cropSource.width);
  const cropHeight = Number(cropSource && cropSource.height);
  const sourceCrop = comparisonFullDimensionsMatch &&
    Number.isFinite(cropLeft) && Number.isFinite(cropTop) && Number.isFinite(cropWidth) && Number.isFinite(cropHeight) &&
    cropLeft >= 0 && cropTop >= 0 && cropWidth > 0 && cropHeight > 0 &&
    cropLeft + cropWidth <= comparisonStages[0].fullWidth && cropTop + cropHeight <= comparisonStages[0].fullHeight
    ? {
      left: cropLeft / comparisonStages[0].fullWidth * 100,
      top: cropTop / comparisonStages[0].fullHeight * 100,
      width: cropWidth / comparisonStages[0].fullWidth * 100,
      height: cropHeight / comparisonStages[0].fullHeight * 100
    }
    : null;
  const requestedDividers = comparisonSource && Array.isArray(comparisonSource.initialDividers)
    ? comparisonSource.initialDividers.map(Number)
    : [];
  const requestedGap = Number(comparisonSource && comparisonSource.minimumGap);
  const comparisonGap = Number.isFinite(requestedGap) ? Math.min(30, Math.max(6, requestedGap)) : 10;
  const comparisonLeft = Number.isFinite(requestedDividers[0]) ? requestedDividers[0] : 33;
  const comparisonRight = Number.isFinite(requestedDividers[1]) ? requestedDividers[1] : 67;
  const previewComparison = comparisonSource && comparisonSource.type === 'three-way' &&
    comparisonDimensionsMatch && comparisonFullDimensionsMatch && comparisonLabelsAreUnique &&
    comparisonAltsAreUnique && comparisonFullAltsAreUnique &&
    comparisonLeft > 0 && comparisonRight < 100 && comparisonLeft + comparisonGap <= comparisonRight
    ? {
      stages: comparisonStages,
      left: comparisonLeft,
      right: comparisonRight,
      minimumGap: comparisonGap,
      width: comparisonStages[0].width,
      height: comparisonStages[0].height,
      fullWidth: comparisonStages[0].fullWidth,
      fullHeight: comparisonStages[0].fullHeight,
      sourceCrop
    }
    : null;
  const role = project.role;
  const notes = normalizeWhitespace(project.notes || '');
  const personalStory = project.personalStory && typeof project.personalStory === 'object' && !Array.isArray(project.personalStory)
    ? project.personalStory
    : null;
  const evaluation = project.evaluation && typeof project.evaluation === 'object' && !Array.isArray(project.evaluation)
    ? project.evaluation
    : null;
  const audiences = Array.isArray(project.audiences) ? project.audiences : [];
  const audienceTags = audiences.map((audience) => {
    const key = normalizeWhitespace(audience).toLowerCase();
    if (key === 'data-science') return 'Data Science';
    if (key === 'analytics') return 'Analytics';
    if (key === 'tourism') return 'Tourism';
    return '';
  }).filter(Boolean);

  const tags = [...new Set([...audienceTags, ...concepts, ...tools])]
    .map((t) => normalizeWhitespace(t))
    .filter(Boolean);

  const embed = project && typeof project.embed === 'object' ? project.embed : null;
  const demoInstructions = project && typeof project.demoInstructions === 'object'
    ? project.demoInstructions
    : null;
  const tableauPreconnect = embed && String(embed.type || '').trim() === 'tableau'
    ? '  <link rel="preconnect" href="https://public.tableau.com" crossorigin>\n'
    : '';
  const comparisonScript = previewComparison
    ? '  <script defer src="js/portfolio/project-image-comparison.js"></script>\n'
    : '';

  const ogImageWidth = Number(project.imageWidth);
  const ogImageHeight = Number(project.imageHeight);
  const ogImageDimensionsMeta = Number.isFinite(ogImageWidth) && Number.isFinite(ogImageHeight) && ogImageWidth > 0 && ogImageHeight > 0
    ? `  <meta property="og:image:width" content="${escapeHtml(ogImageWidth)}">\n  <meta property="og:image:height" content="${escapeHtml(ogImageHeight)}">\n`
    : '';

  const projectLd = {
    '@type': 'CreativeWork',
    '@id': `${canonicalUrl}#project`,
    name: title,
    description,
    url: canonicalUrl,
    image: ogImage,
    mainEntityOfPage: { '@id': `${canonicalUrl}#webpage` },
    creator: {
      '@type': 'Person',
      '@id': `${SITE_ORIGIN}/#person`,
      name: 'Daniel Short',
      url: `${SITE_ORIGIN}/`
    },
    ...(tags.length ? { keywords: tags.join(', ') } : {})
  };
  const breadcrumbsLd = {
    '@type': 'BreadcrumbList',
    '@id': `${canonicalUrl}#breadcrumb`,
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: `${SITE_ORIGIN}/` },
      { '@type': 'ListItem', position: 2, name: 'Portfolio', item: `${SITE_ORIGIN}/portfolio` },
      { '@type': 'ListItem', position: 3, name: title, item: canonicalUrl }
    ]
  };
  const ldJson = JSON.stringify({ '@context': 'https://schema.org', '@graph': [projectLd, breadcrumbsLd] })
    .replace(/</g, '\\u003c');

  const heroTags = tags.slice(0, 4);
  const safeTagPills = heroTags.length
    ? `<div class="project-tags" role="list">
      ${heroTags.map((t) => `<span class="project-tag" role="listitem">${escapeHtml(t)}</span>`).join('\n      ')}
    </div>`
    : '';

  const safeProblem = normalizeWhitespace(project.problem || '');

  const hasResources = resources.length > 0;
  const hasNotes = Boolean(notes);

  const renderResourceCards = (list) => `<div class="project-links" role="list">
        ${list.map((r) => {
          const href = String(r.url || '').trim();
          const label = normalizeWhitespace(r.label || href);
          const icon = String(r.icon || '').trim();
          const isExternal = /^https?:\/\//i.test(href);
          const attrs = isExternal ? ' target="_blank" rel="noopener noreferrer"' : '';
          const iconMarkup = icon
            ? `<img class="project-link-icon" src="${escapeHtml(icon)}" alt="" aria-hidden="true" loading="lazy" decoding="async" width="20" height="20">`
            : '';
          return `<a class="project-link" role="listitem" href="${escapeHtml(href)}"${attrs}>${iconMarkup}<span class="project-link-label">${escapeHtml(label)}</span></a>`;
        }).join('\n        ')}
      </div>`;

  const safeResources = hasResources
    ? (() => {
      const dataResources = resources.filter(isDataResource);
      const projectResources = resources.filter((resource) => !isDataResource(resource));

      const groups = [];
      if (projectResources.length) {
        groups.push(`<div class="project-links-group">
        <h3 class="project-links-group-title">Project Links</h3>
        ${renderResourceCards(projectResources)}
      </div>`);
      }
      if (dataResources.length) {
        groups.push(`<div class="project-links-group">
        <h3 class="project-links-group-title">Data Links</h3>
        ${renderResourceCards(dataResources)}
      </div>`);
      }

      return `<details class="project-section project-disclosure project-resources" id="links" data-project-mobile-disclosure open>
      <summary class="project-disclosure-summary"><span class="section-title">Links</span></summary>
      <div class="project-links-groups">
        ${groups.join('\n        ')}
      </div>
    </details>`;
    })()
    : '';

  const safeNotes = hasNotes
    ? `<details class="project-section project-disclosure project-notes" id="notes" data-project-mobile-disclosure open>
      <summary class="project-disclosure-summary"><span class="section-title">Notes</span></summary>
      <p class="project-lead">${escapeHtml(notes)}</p>
    </details>`
    : '';

  const allProjects = Array.isArray(options.projects) ? options.projects : null;
  const projectIndex = Number.isInteger(options.index) ? options.index : -1;
  const projectPager = allProjects && projectIndex >= 0
    ? renderProjectPager(allProjects, projectIndex)
    : '';
  const relatedProjects = allProjects && projectIndex >= 0
    ? renderRelatedProjectsSection(allProjects, projectIndex)
    : '';

  const ensureSentence = (value) => {
    const s = normalizeWhitespace(value);
    if (!s) return '';
    return /[.!?]$/.test(s) ? s : `${s}.`;
  };
  const starSituation = ensureSentence(safeProblem);
  const starTask = (() => {
    if (Array.isArray(role) && role.length) return ensureSentence(role[0]);
    if (typeof role === 'string') return ensureSentence(role);
    return 'Owned the end-to-end build, from implementation through the final deliverable.';
  })();
  const starActions = actions.slice(0, 3).map((a) => normalizeWhitespace(a)).filter(Boolean);
  const starResults = results.slice(0, 3).map((r) => normalizeWhitespace(r)).filter(Boolean);
  const stackLabel = tools.map((tool) => normalizeWhitespace(tool)).filter(Boolean).join(' \u00b7 ') || 'Project-specific tools and methods';
  const deliveryStatus = normalizeWhitespace(project.deliveryStatus || project.status || '')
    || (embed ? 'Live interactive demo' : 'Completed case study');

  const starSummary = `<section class="project-star" aria-label="STAR summary">
      <h2 class="section-title">STAR Summary</h2>
      <dl class="project-star-grid">
        <div class="project-star-row">
          <dt class="project-star-label">Situation</dt>
          <dd class="project-star-value">${escapeHtml(starSituation || safeProblem)}</dd>
        </div>
        <div class="project-star-row">
          <dt class="project-star-label">Task</dt>
          <dd class="project-star-value">${escapeHtml(starTask)}</dd>
        </div>
        <div class="project-star-row">
          <dt class="project-star-label">Action</dt>
          <dd class="project-star-value">
            <ul class="project-star-list">
              ${starActions.map((item) => `<li>${escapeHtml(item)}</li>`).join('\n              ')}
            </ul>
          </dd>
        </div>
        <div class="project-star-row">
          <dt class="project-star-label">Result</dt>
          <dd class="project-star-value">
            <ul class="project-star-list">
              ${starResults.map((item) => `<li>${escapeHtml(item)}</li>`).join('\n              ')}
            </ul>
          </dd>
        </div>
        <div class="project-star-row">
          <dt class="project-star-label">Stack</dt>
          <dd class="project-star-value">${escapeHtml(stackLabel)}</dd>
        </div>
        <div class="project-star-row">
          <dt class="project-star-label">Status</dt>
          <dd class="project-star-value">${escapeHtml(deliveryStatus)}</dd>
        </div>
      </dl>
    </section>`;

  const renderDefinitionRows = (rows) => rows
    .filter((row) => row && normalizeWhitespace(row.value || ''))
    .map((row) => `<div class="project-star-row">
          <dt class="project-star-label">${escapeHtml(row.label)}</dt>
          <dd class="project-star-value">${escapeHtml(normalizeWhitespace(row.value))}</dd>
        </div>`)
    .join('\n        ');

  const personalStoryRows = personalStory ? [
    { label: 'Why I built it', value: personalStory.why },
    { label: 'What surprised me', value: personalStory.surprise },
    { label: 'What I’d try next', value: personalStory.next }
  ] : [];
  const personalNotes = personalStoryRows.some((row) => normalizeWhitespace(row.value || ''))
    ? `<details class="project-star project-personal-notes project-disclosure" data-project-mobile-disclosure open>
      <summary class="project-disclosure-summary"><span class="section-title" id="${escapeHtml(toDomIdSafe(id))}-personal-notes-title">Personal notes</span></summary>
      <dl class="project-star-grid">
        ${renderDefinitionRows(personalStoryRows)}
      </dl>
    </details>`
    : '';

  const evaluationStatusLabels = {
    measured: 'Measured',
    partial: 'Partial evaluation',
    'not-benchmarked': 'Not benchmarked'
  };
  const evaluationStatus = evaluation
    ? normalizeWhitespace(evaluation.status || '').toLowerCase()
    : '';
  const evaluationMetrics = evaluation && Array.isArray(evaluation.metrics)
    ? evaluation.metrics.filter((metric) => metric && typeof metric === 'object' && !Array.isArray(metric))
    : [];
  const evaluationLimitations = evaluation && Array.isArray(evaluation.limitations)
    ? evaluation.limitations.map(normalizeWhitespace).filter(Boolean)
    : [];
  const evaluationEvidence = evaluation && evaluation.evidence && typeof evaluation.evidence === 'object' && !Array.isArray(evaluation.evidence)
    ? evaluation.evidence
    : null;
  const evaluationEvidenceUrl = evaluationEvidence ? String(evaluationEvidence.url || '').trim() : '';
  const evaluationEvidenceLabel = evaluationEvidence
    ? normalizeWhitespace(evaluationEvidence.label || evaluationEvidenceUrl)
    : '';
  const evaluationEvidenceExternal = /^https?:\/\//i.test(evaluationEvidenceUrl);
  const evaluationRows = evaluation ? [
    { label: 'Status', value: evaluationStatusLabels[evaluationStatus] || '' },
    { label: 'Goal', value: evaluation.goal },
    { label: 'Dataset', value: evaluation.dataset },
    { label: 'Split', value: evaluation.split },
    { label: 'Baseline', value: evaluation.baseline },
    { label: 'Decision', value: evaluation.decision }
  ] : [];
  const evaluationDetails = evaluation && evaluationStatusLabels[evaluationStatus]
    ? `<details class="project-star project-evaluation project-disclosure" data-project-mobile-disclosure open>
      <summary class="project-disclosure-summary"><span class="section-title" id="${escapeHtml(toDomIdSafe(id))}-evaluation-title">Evaluation &amp; tradeoffs</span></summary>
      <dl class="project-star-grid">
        ${renderDefinitionRows(evaluationRows)}
${evaluationMetrics.length ? `<div class="project-star-row">
          <dt class="project-star-label">Metrics</dt>
          <dd class="project-star-value">
            <ul class="project-star-list">
              ${evaluationMetrics.map((metric) => {
                const label = normalizeWhitespace(metric.label || 'Metric');
                const value = normalizeWhitespace(metric.value || '');
                const context = normalizeWhitespace(metric.context || '');
                const metricText = [label && value ? `${label}: ${value}` : (label || value), context].filter(Boolean).join(' — ');
                return `<li>${escapeHtml(metricText)}</li>`;
              }).join('\n              ')}
            </ul>
          </dd>
        </div>` : ''}
${evaluationLimitations.length ? `<div class="project-star-row">
          <dt class="project-star-label">Limitations</dt>
          <dd class="project-star-value">
            <ul class="project-star-list">
              ${evaluationLimitations.map((item) => `<li>${escapeHtml(item)}</li>`).join('\n              ')}
            </ul>
          </dd>
        </div>` : ''}
${evaluationEvidenceUrl && evaluationEvidenceLabel ? `<div class="project-star-row">
          <dt class="project-star-label">Evidence</dt>
          <dd class="project-star-value"><a href="${escapeHtml(evaluationEvidenceUrl)}"${evaluationEvidenceExternal ? ' target="_blank" rel="noopener noreferrer"' : ''}>${escapeHtml(evaluationEvidenceLabel)}</a></dd>
        </div>` : ''}
      </dl>
    </details>`
    : '';

  const renderImageMedia = () => {
    const img = String(project.image || '').trim();
    if (!img) return '';
    const alt = escapeHtml(ogImageAlt);
    const width = Number(project.imageWidth);
    const height = Number(project.imageHeight);
    const sizeAttr = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
      ? ` width="${width}" height="${height}"`
      : '';
    const match = img.match(/\.(png|jpe?g)$/i);
    if (!match) {
      return `<img class="project-media" src="${escapeHtml(img)}" alt="${alt}" loading="eager" decoding="async"${sizeAttr} fetchpriority="high">`;
    }

    const base = img.replace(/\.(png|jpe?g)$/i, '');
    const avif = buildResponsiveSrcset(base, 'avif', width);
    const webp = buildResponsiveSrcset(base, 'webp', width);
    const sizes = ' sizes="(max-width: 960px) 92vw, 840px"';
    if (avif || webp) {
      return `<picture class="project-media">
        ${avif ? `<source srcset="${escapeHtml(avif)}" type="image/avif">` : ''}
        ${webp ? `<source srcset="${escapeHtml(webp)}" type="image/webp">` : ''}
        <img src="${escapeHtml(img)}" alt="${alt}" loading="eager" decoding="async"${sizeAttr}${sizes} fetchpriority="high">
      </picture>`;
    }
    return `<img class="project-media" src="${escapeHtml(img)}" alt="${alt}" loading="eager" decoding="async"${sizeAttr} fetchpriority="high">`;
  };

  const renderVideoMedia = () => {
    const webm = String(project.videoWebm || '').trim();
    const mp4 = String(project.videoMp4 || '').trim();
    if (!webm && !mp4) return '';

    const poster = String(project.image || '').trim();
    const posterAttr = poster ? ` poster="${escapeHtml(poster)}"` : '';
    const sources = [
      webm ? `<source src="${escapeHtml(webm)}" type="video/webm">` : '',
      mp4 ? `<source src="${escapeHtml(mp4)}" type="video/mp4">` : ''
    ].filter(Boolean).join('\n          ');
    const label = escapeHtml(`${title} video`);

    return `<div class="project-media project-video">
      <video class="project-video-frame" controls autoplay muted loop playsinline preload="metadata"${posterAttr} aria-label="${label}">
        ${sources}
      </video>
    </div>`;
  };

  const renderPreviewComparison = () => {
    if (!previewComparison) return '';

    const safeId = toDomIdSafe(id);
    const fullHeadingId = `project-comparison-full-heading-${safeId}`;
    const fullDescriptionId = `project-comparison-full-description-${safeId}`;
    const zoomHeadingId = `project-comparison-zoom-heading-${safeId}`;
    const zoomDescriptionId = `project-comparison-zoom-description-${safeId}`;
    const viewportId = `project-comparison-viewport-${safeId}`;
    const instructionsId = `project-comparison-instructions-${safeId}`;
    const stages = previewComparison.stages;
    const cropCue = previewComparison.sourceCrop
      ? '<span class="project-stage-full-crop" aria-hidden="true"></span>'
      : '';
    const fullStages = stages.map((stage, index) => {
      const description = stage.description
        ? `<span class="project-stage-full-description">${escapeHtml(stage.description)}</span>`
        : '';
      return `<figure class="project-stage-full" data-full-stage>
          <figcaption class="project-stage-full-caption">
            <span class="project-stage-full-index" aria-hidden="true">0${index + 1}</span>
            <span class="project-stage-full-copy"><strong>${escapeHtml(stage.label)}</strong>${description}</span>
          </figcaption>
          <div class="project-stage-full-image-frame">
            <img class="project-stage-full-image" src="${escapeHtml(stage.fullImage)}" alt="${escapeHtml(stage.fullAlt)}" loading="lazy" decoding="async" width="${stage.fullWidth}" height="${stage.fullHeight}">
            ${cropCue}
          </div>
        </figure>`;
    }).join('\n        ');
    const slides = stages.map((stage, index) => {
      const loading = index === 0 ? 'eager' : 'lazy';
      const priority = index === 0 ? ' fetchpriority="high"' : '';
      const description = stage.description
        ? `<span>${escapeHtml(stage.description)}</span>`
        : '';
      return `<figure class="project-stage-slide" data-stage-slide data-stage-label="${escapeHtml(stage.label)}">
          <img class="project-stage-image" src="${escapeHtml(stage.image)}" alt="${escapeHtml(stage.alt)}" loading="${loading}" decoding="async" width="${stage.width}" height="${stage.height}"${priority}>
          <figcaption class="project-stage-slide-caption"><strong>${escapeHtml(stage.label)}</strong>${description}</figcaption>
        </figure>`;
    }).join('\n        ');
    const stageRail = stages.map((stage, index) => `<li title="${escapeHtml(stage.label)}">
            <span class="project-image-comparison-stage-index">0${index + 1}</span>
            <span class="project-image-comparison-stage-name">${escapeHtml(stage.shortLabel)}</span>
          </li>`).join('\n          ');
    const dividerIcon = `<svg viewBox="0 0 24 18" aria-hidden="true" focusable="false">
              <path d="M9 4 4 9l5 5M15 4l5 5-5 5"></path>
            </svg>`;
    const leftValue = Math.round(previewComparison.left);
    const rightValue = Math.round(previewComparison.right);
    const leftMax = Math.round(previewComparison.right - previewComparison.minimumGap);
    const rightMin = Math.round(previewComparison.left + previewComparison.minimumGap);
    const cropStyle = previewComparison.sourceCrop
      ? `;--comparison-crop-left:${Number(previewComparison.sourceCrop.left.toFixed(4))}%;--comparison-crop-top:${Number(previewComparison.sourceCrop.top.toFixed(4))}%;--comparison-crop-width:${Number(previewComparison.sourceCrop.width.toFixed(4))}%;--comparison-crop-height:${Number(previewComparison.sourceCrop.height.toFixed(4))}%`
      : '';
    const fullDescription = previewComparison.sourceCrop
      ? 'Each pipeline stage is shown in full. The blue outlined area is enlarged in the comparison below.'
      : 'Each pipeline stage is shown in full.';

    return `<div class="project-image-comparison" data-project-image-comparison data-comparison-left="${previewComparison.left}" data-comparison-right="${previewComparison.right}" data-comparison-minimum-gap="${previewComparison.minimumGap}" style="--comparison-left:${previewComparison.left}%;--comparison-right:${previewComparison.right}%;--comparison-aspect:${previewComparison.width} / ${previewComparison.height};--comparison-full-aspect:${previewComparison.fullWidth} / ${previewComparison.fullHeight}${cropStyle}">
      <section class="project-image-comparison-section project-image-comparison-full" aria-labelledby="${escapeHtml(fullHeadingId)}" aria-describedby="${escapeHtml(fullDescriptionId)}">
        <div class="project-image-comparison-heading">
          <h3 id="${escapeHtml(fullHeadingId)}">Full images</h3>
          <p id="${escapeHtml(fullDescriptionId)}">${escapeHtml(fullDescription)}</p>
        </div>
        <div class="project-stage-full-grid">
          ${fullStages}
        </div>
      </section>
      <section class="project-image-comparison-section project-image-comparison-zoom" aria-labelledby="${escapeHtml(zoomHeadingId)}" aria-describedby="${escapeHtml(zoomDescriptionId)}">
        <div class="project-image-comparison-heading">
          <h3 id="${escapeHtml(zoomHeadingId)}">Zoomed comparison</h3>
          <p id="${escapeHtml(zoomDescriptionId)}">Inspect the same detail across all three stages.</p>
        </div>
        <div class="project-image-comparison-zoom-card">
          <div class="project-image-comparison-viewport" id="${escapeHtml(viewportId)}" data-comparison-viewport>
            ${slides}
            <div class="project-image-comparison-divider project-image-comparison-divider-left" data-comparison-divider="left" role="slider" tabindex="0" aria-label="${escapeHtml(`${stages[0].label} / ${stages[1].label} boundary`)}" aria-orientation="horizontal" aria-controls="${escapeHtml(viewportId)}" aria-describedby="${escapeHtml(instructionsId)}" aria-valuemin="0" aria-valuemax="${leftMax}" aria-valuenow="${leftValue}" aria-valuetext="${escapeHtml(`${stages[0].label} ends at ${leftValue}%; ${stages[1].label} begins at ${leftValue}%`)}" data-comparison-before="${escapeHtml(stages[0].label)}" data-comparison-after="${escapeHtml(stages[1].label)}" hidden>
              <span class="project-image-comparison-handle">${dividerIcon}</span>
            </div>
            <div class="project-image-comparison-divider project-image-comparison-divider-right" data-comparison-divider="right" role="slider" tabindex="0" aria-label="${escapeHtml(`${stages[1].label} / ${stages[2].label} boundary`)}" aria-orientation="horizontal" aria-controls="${escapeHtml(viewportId)}" aria-describedby="${escapeHtml(instructionsId)}" aria-valuemin="${rightMin}" aria-valuemax="100" aria-valuenow="${rightValue}" aria-valuetext="${escapeHtml(`${stages[1].label} ends at ${rightValue}%; ${stages[2].label} begins at ${rightValue}%`)}" data-comparison-before="${escapeHtml(stages[1].label)}" data-comparison-after="${escapeHtml(stages[2].label)}" hidden>
              <span class="project-image-comparison-handle">${dividerIcon}</span>
            </div>
          </div>
          <div class="project-image-comparison-controls" data-comparison-controls hidden>
            <ol class="project-image-comparison-stage-rail" aria-hidden="true">
              ${stageRail}
            </ol>
            <p class="project-image-comparison-instruction" id="${escapeHtml(instructionsId)}">Click or tap the image to move the nearest divider. Drag either divider; it pushes the other when they meet. Use the arrow keys when a divider is focused.</p>
          </div>
        </div>
      </section>
    </div>`;
  };

  const renderDemoLaunchPreview = () => {
    const img = String(project.image || '').trim();
    if (!img) return '';
    const width = Number(project.imageWidth);
    const height = Number(project.imageHeight);
    const sizeAttr = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
      ? ` width="${width}" height="${height}"`
      : '';
    return `<img class="project-demo-launch-image" src="${escapeHtml(img)}" alt="Preview of ${escapeHtml(title)}" loading="lazy" decoding="async"${sizeAttr}>`;
  };

  const renderEmbeddedMedia = (options = {}) => {
    if (!embed) return '';
    const lazy = options && options.lazy === true;
    const type = String(embed.type || '').trim();
    if (type === 'iframe') {
      const src = String(embed.url || '').trim();
      if (!src) return '';
      const srcAttr = lazy ? ` data-src="${escapeHtml(src)}"` : ` src="${escapeHtml(src)}"`;
      const embedMeta = renderEmbedAttrs(embed, id, 'project-embed-iframe');
      return `<div class="project-media project-embed ${embedMeta.className}" ${embedMeta.attrs}>
        <iframe class="project-embed-frame"${srcAttr} title="${escapeHtml(title)} interactive demo" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    if (type === 'tableau') {
      const base = String(embed.base || '').trim();
      if (!base) return '';
      const joiner = base.includes('?') ? '&' : '?';
      const src = `${base}${joiner}:showVizHome=no&:embed=y`;
      const srcAttr = lazy ? ` data-src="${escapeHtml(src)}"` : ` src="${escapeHtml(src)}"`;
      const embedMeta = renderEmbedAttrs(embed, id, 'project-embed-tableau');
      return `<div class="project-media project-embed ${embedMeta.className}" ${embedMeta.attrs}>
        <iframe class="project-embed-frame"${srcAttr} title="${escapeHtml(title)} interactive dashboard" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    return '';
  };

  const renderDemoShell = () => {
    if (!embed) return '';
    const safeId = toDomIdSafe(id);
    const baseId = `project-demo-${safeId}`;
    const tooltipId = `${baseId}-instructions`;
    const embedFit = resolveEmbedFit(embed);
    const embedType = String(embed.type || '').trim();
    const launchHref = embedType === 'iframe' ? String(embed.url || '').trim() : '';

    const lead = normalizeWhitespace(demoInstructions?.lead || '');
    const bullets = normalizeTextArray(demoInstructions?.bullets);
    const safeLead = lead ? `<p class="project-demo-tooltip-lead">${escapeHtml(lead)}</p>` : '';
    const safeBullets = bullets.length
      ? `<ul class="project-demo-tooltip-list">
        ${bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join('\n        ')}
      </ul>`
      : '';
    const tooltip = lead || bullets.length
      ? `<div class="project-demo-help">
          <button class="project-demo-help-trigger" type="button" aria-label="Demo instructions" aria-describedby="${escapeHtml(tooltipId)}">
            <span aria-hidden="true">?</span>
          </button>
          <div class="project-demo-tooltip" id="${escapeHtml(tooltipId)}" role="tooltip">
            ${safeLead}
            ${safeBullets}
          </div>
        </div>`
      : '';

    const mobileLaunch = embedFit === 'content' && launchHref
      ? `<div class="project-demo-mobile-launch">
          ${renderDemoLaunchPreview()}
          <div class="project-demo-launch-copy">
            <p>Open the standalone demo for the full interactive workspace.</p>
            <a class="btn-primary" href="${escapeHtml(launchHref)}">Launch demo</a>
          </div>
        </div>`
      : '';

    return `<section class="project-demo-shell" data-demo-fit="${escapeHtml(embedFit)}" aria-label="Interactive demo">
      <div class="project-demo-header">
        <h2 class="section-title project-demo-title">Demo</h2>
        ${tooltip}
      </div>

      <div class="project-demo-panels">
        <section class="project-demo-panel is-active" data-demo-panel="demo">
          <div class="project-demo-panel-inner">
${mobileLaunch ? `            ${mobileLaunch}\n` : ''}            ${renderEmbeddedMedia({ lazy: embedFit === 'content' })}
          </div>
        </section>
      </div>
    </section>`;
  };

  const media = (() => {
    if (embed) return '';
    const comparison = renderPreviewComparison();
    if (comparison) return comparison;
    const video = renderVideoMedia();
    if (video) return video;
    return renderImageMedia();
  })();

  const demoTabs = embed ? renderDemoShell() : '';
  const projectPreview = !embed && media
    ? `<section class="project-demo-shell project-preview-shell" data-demo-fit="fixed" aria-label="Project preview">
      <div class="project-demo-header">
        <h2 class="section-title project-demo-title">Project Preview</h2>
      </div>

      <div class="project-demo-panels">
        <section class="project-demo-panel is-active" data-demo-panel="demo">
          <div class="project-demo-panel-inner">
            ${media}
          </div>
        </section>
      </div>
    </section>`
    : '';

  const projectBodySections = [
    starSummary,
    personalNotes,
    evaluationDetails,
    demoTabs || projectPreview,
    safeResources,
    safeNotes,
    relatedProjects
  ].filter(Boolean).join('\n        ');

  return `<!DOCTYPE html>
<html lang="en" class="no-js">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <base href="/">
  <title>${escapeHtml(title)} | Daniel Short</title>
  <link rel="canonical" href="${escapeHtml(canonicalUrl)}">
  <meta name="description" content="${escapeHtml(description)}">

  <meta property="og:title" content="${escapeHtml(title)} | Daniel Short">
  <meta property="og:site_name" content="Daniel Short">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">
  <meta property="og:image" content="${escapeHtml(ogImage)}">
  <meta property="og:image:alt" content="${escapeHtml(ogImageAlt)}">
${ogImageDimensionsMeta}
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@danielshort3">

  <meta name="theme-color" content="#091F3B">
  <link rel="stylesheet" href="dist/styles.css">
  <!-- Legacy source reference retained for tooling: css/components/project-page.css -->
  <link rel="icon" href="favicon.ico" sizes="any">
  <link rel="icon" type="image/svg+xml" href="img/brand/05-ds-favicon-small-icon.svg">
  <link rel="icon" type="image/png" sizes="16x16" href="img/ui/logo-16.png">
  <link rel="icon" type="image/png" sizes="32x32" href="img/ui/logo-32.png">
  <link rel="icon" type="image/png" sizes="64x64" href="img/ui/logo-64.png">
  <link rel="icon" type="image/png" sizes="192x192" href="img/ui/logo-192.png">
  <link rel="apple-touch-icon" sizes="180x180" href="img/ui/logo-180.png">
${tableauPreconnect}

  <!-- Local fonts with legacy reference retained for tooling: https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Inter:wght@500;600&display=swap -->
  <script src="js/common/no-js.js"></script>
  <script type="application/ld+json">
    ${ldJson}
  </script>
</head>
<body data-page="project" class="project-page">
  <a href="#main" class="skip-link">Skip to main content</a>
  <header id="combined-header-nav"></header>
  ${projectPager}

  <main id="main">
    <section class="project-hero">
      <div class="wrapper">
        <p class="hero-eyebrow">Portfolio Project</p>
        <h1>${escapeHtml(title)}</h1>
        ${subtitle ? `<p class="project-subtitle">${escapeHtml(subtitle)}</p>` : ''}
        <div class="cta-group project-cta">
          <a class="btn-primary hero-cta" href="portfolio">Back to Portfolio</a>
        </div>
        ${safeTagPills}
		      </div>
	    </section>

			    <section class="project-body">
			      <div class="wrapper">
			        ${projectBodySections}
			      </div>
			    </section>
			  </main>

  <footer>
	    <nav class="privacy-links" aria-label="Privacy shortcuts">
	      <button id="privacy-settings-link" type="button" class="pcz-link">Privacy settings</button>
	      <a href="privacy#prefs-title" class="pcz-link" data-consent-open="true">Do Not Sell/Share My Personal Information</a>
	      <a href="sitemap-pretty" class="pcz-link">Sitemap</a>
	    </nav>
	  </footer>

  <script defer src="js/common/common.js"></script>
  <script defer src="js/navigation/navigation.js"></script>
  <script defer src="js/animations/animations.js"></script>
${comparisonScript}  <script src="js/privacy/config.js"></script>
  <script defer src="js/privacy/consent_manager.js"></script>
</body>
</html>
`;
}

function writeProjectPages(projects) {
  fs.mkdirSync(outDir, { recursive: true });
  const expected = new Set(
    projects
      .map((project) => String(project?.id || '').trim())
      .filter(Boolean)
      .map((id) => `${id}.html`)
  );
  try {
    fs.readdirSync(outDir).forEach((name) => {
      if (!name.endsWith('.html')) return;
      if (expected.has(name)) return;
      fs.rmSync(path.join(outDir, name), { force: true });
    });
  } catch (_) {}

  projects.forEach((project, index) => {
    const id = String(project.id || '').trim();
    if (!id) throw new Error('Project missing id');
    const outPath = path.join(outDir, `${id}.html`);
    fs.writeFileSync(outPath, renderProjectPage(project, { projects, index }), 'utf8');
  });
}

function renderPortfolioStaticResults(projects) {
  return (Array.isArray(projects) ? projects : []).map((project) => {
    const id = String(project && project.id ? project.id : '').trim();
    if (!id) return '';
    const title = normalizeWhitespace(project.title || id);
    const summary = toMetaDescription(project);
    const image = String(project.image || '').trim();
    const width = Number(project.imageWidth);
    const height = Number(project.imageHeight);
    const sizeAttrs = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0
      ? ` width="${escapeHtml(width)}" height="${escapeHtml(height)}"`
      : '';
    const media = image
      ? `<img src="${escapeHtml(image)}" alt="Preview of ${escapeHtml(title)}"${sizeAttrs} loading="lazy" decoding="async">`
      : `<span class="portfolio-result-card__initial">${escapeHtml(title.charAt(0) || '?')}</span>`;
    const labels = [...new Set([
      normalizeWhitespace(project.subtitle || ''),
      ...(Array.isArray(project.concepts) ? project.concepts : []),
      ...(Array.isArray(project.tools) ? project.tools : [])
    ].map(normalizeWhitespace).filter(Boolean))].slice(0, 2);
    const chips = labels.length
      ? `<span class="portfolio-result-tags">${labels.map((label) => `<span>${escapeHtml(label)}</span>`).join('')}</span>`
      : '';

    return [
      `<article class="portfolio-result-card portfolio-project-result portfolio-project-result--static" role="listitem" data-project-id="${escapeHtml(id)}">`,
      `  <span class="portfolio-result-card__media" aria-hidden="true">${media}</span>`,
      '  <div class="portfolio-result-card__body">',
      `    <h2 class="portfolio-result-card__title">${escapeHtml(title)}</h2>`,
      summary ? `    <p class="portfolio-result-card__outcome"><span>Outcome</span>${escapeHtml(summary)}</p>` : '',
      chips ? `    ${chips}` : '',
      '    <div class="portfolio-result-card__actions">',
      `      <a class="portfolio-result-card__open" href="portfolio/${escapeHtml(encodeURIComponent(id))}" data-content-open="true" data-content-id="${escapeHtml(id)}" data-content-type="project" data-resource-type="case_study" data-source-surface="portfolio_results">View case study <span aria-hidden="true">-&gt;</span></a>`,
      '    </div>',
      '  </div>',
      '</article>'
    ].filter(Boolean).join('\n');
  }).filter(Boolean).join('\n');
}

function assertCompletePortfolioIndex(html) {
  const source = String(html || '');
  const requiredMarkers = ['</main>', '<footer', '</footer>', '</body>', '</html>'];
  const missing = requiredMarkers.filter((marker) => !source.includes(marker));
  const hasPartialTrailingTag = /<[^>]*$/.test(source.trim());
  if (!missing.length && !hasPartialTrailingTag) return;

  const detail = [
    missing.length ? `missing ${missing.join(', ')}` : '',
    hasPartialTrailingTag ? 'ends with a partial HTML tag' : ''
  ].filter(Boolean).join('; ');
  throw new Error(`Portfolio index source appears truncated (${detail}). Stop active build watchers, restore the complete document, and rebuild.`);
}

function syncPortfolioStaticResults(projects) {
  if (!fs.existsSync(personalPortfolioIndexPath)) return;
  const personalHtml = fs.readFileSync(personalPortfolioIndexPath, 'utf8');
  let portfolioIndexPath = personalPortfolioIndexPath;
  let html = unwrapPersonalAccordionHtml(personalHtml);
  assertCompletePortfolioIndex(html);
  const start = '<!-- portfolio-static-results:start -->';
  const end = '<!-- portfolio-static-results:end -->';
  const matcher = new RegExp(`${start}[\\s\\S]*?${end}`);
  if (!matcher.test(html)) {
    const isWrappedLibrary = /data-personal-accordion-shell|\bpersonal-library-main\b/i.test(personalHtml);
    if (!isWrappedLibrary || !fs.existsSync(professionalPortfolioIndexPath)) return;
    portfolioIndexPath = professionalPortfolioIndexPath;
    html = unwrapPersonalAccordionHtml(fs.readFileSync(professionalPortfolioIndexPath, 'utf8'));
    assertCompletePortfolioIndex(html);
    if (!matcher.test(html)) return;
  }
  const results = renderPortfolioStaticResults(projects);
  const indentedResults = results.split('\n').map((line) => `              ${line}`).join('\n');
  const block = `${start}\n${indentedResults}\n              ${end}`;
  const next = html.replace(matcher, block);
  assertCompletePortfolioIndex(next);
  if (next !== html) fs.writeFileSync(portfolioIndexPath, next, 'utf8');
}

function writeSitemap(projects) {
  const previousEntries = loadSitemapCache();
  const nextEntries = new Map();
  const noindexPathnames = loadNoindexPathnamesFromVercel(root);
  const baseEntries = [
    { loc: `${SITE_ORIGIN}/`, sourceFile: 'content/audiences/personal.json' },
    { loc: `${SITE_ORIGIN}/portfolio`, sourceFile: 'js/portfolio/projects-data.js' },
    { loc: `${SITE_ORIGIN}/games`, sourceFile: 'content/pages/games.json' },
    { loc: `${SITE_ORIGIN}/contact`, sourceFile: 'content/pages/contact.json' },
    { loc: `${SITE_ORIGIN}/solutions`, sourceFile: 'pages/solutions.html' },
    { loc: `${SITE_ORIGIN}/tools`, sourceFile: 'js/portfolio/tools-directory-data.js' },
    { loc: `${SITE_ORIGIN}/privacy`, sourceFile: 'pages/privacy.html' },
    { loc: `${SITE_ORIGIN}/sitemap`, sourceFile: 'pages/sitemap.html' }
  ];

  const toolEntries = loadToolUrls().map((loc) => {
    const slug = String(loc).replace(`${SITE_ORIGIN}/tools/`, '').replace(/^\/+/, '');
    return { loc, sourceFile: `content/tools/${slug}.json` };
  });

  const gameEntries = loadGameEntries();

  const projectEntries = projects
    .map((p) => String(p.id || '').trim())
    .filter(Boolean)
    .map((id) => ({
      loc: `${SITE_ORIGIN}/portfolio/${encodeURIComponent(id)}`,
      sourceFile: `content/projects/${id}.json`
    }));

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...baseEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries, noindexPathnames)).filter(Boolean),
    '',
    ...gameEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries, noindexPathnames)).filter(Boolean),
    '',
    ...toolEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries, noindexPathnames)).filter(Boolean),
    '',
    ...projectEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries, noindexPathnames)).filter(Boolean),
    '</urlset>',
    ''
  ].join('\n');
  fs.writeFileSync(sitemapPath, xml, 'utf8');
  writeSitemapCache(nextEntries);
}

function main() {
  const projects = loadProjects().filter(isPublishedProject);
  writeProjectPages(projects);
  syncPortfolioStaticResults(projects);
  writeSitemap(projects);
  process.stdout.write(`Generated ${projects.length} project pages in pages/portfolio/ and updated sitemap.xml\n`);
}

if (require.main === module) {
  main();
}

module.exports = {
  isPublishedProject,
  loadProjects,
  renderPortfolioStaticResults,
  renderProjectPage
};
