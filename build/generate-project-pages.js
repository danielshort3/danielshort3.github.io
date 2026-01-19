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

const root = path.resolve(__dirname, '..');
const dataFile = path.join(root, 'js', 'portfolio', 'projects-data.js');
const outDir = path.join(root, 'pages', 'portfolio');
const sitemapPath = path.join(root, 'sitemap.xml');
const sitemapCachePath = path.join(root, 'sitemap-cache.json');
const SITE_ORIGIN = 'https://danielshort.me';
const toolsIndexPath = path.join(root, 'pages', 'tools.html');

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
    if (!fs.existsSync(toolsIndexPath)) return [];
    const html = fs.readFileSync(toolsIndexPath, 'utf8');
    const re = /href="tools\/([^"#?]+)"/g;
    let match;
    while ((match = re.exec(html))) {
      const slug = String(match[1] || '').trim();
      if (!slug) continue;
      urls.add(`${SITE_ORIGIN}/tools/${slug}`);
    }
  } catch (_) {}
  return [...urls].sort();
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

  const gitLastmod = getGitLastmod(sourceFile);
  if (gitLastmod) return { lastmod: gitLastmod, hash: currentHash };

  if (currentHash && previousHash && currentHash === previousHash && previousLastmod) {
    return { lastmod: previousLastmod, hash: currentHash };
  }

  const changed = currentHash && previousHash && currentHash !== previousHash;

  if (!changed && previousLastmod) {
    return { lastmod: previousLastmod, hash: currentHash };
  }

  const fsLastmod = getFsLastmod(sourceFile);
  if (fsLastmod) {
    return { lastmod: fsLastmod, hash: currentHash };
  }

  return { lastmod: previousLastmod, hash: currentHash };
}

function toSitemapUrlEntry(options, previousEntries, nextEntries) {
  const loc = String(options?.loc || '').trim();
  if (!loc) return '';
  const meta = resolveSitemapMeta({ loc, sourceFile: options?.sourceFile }, previousEntries);
  const lastmod = meta && meta.lastmod ? meta.lastmod : null;
  const hash = meta && meta.hash ? meta.hash : null;
  const priority = Number(options?.priority);

  if (nextEntries && typeof nextEntries.set === 'function') {
    nextEntries.set(loc, { lastmod, hash });
  }

  const lines = [
    '  <url>',
    `    <loc>${loc.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</loc>`
  ];
  if (lastmod) lines.push(`    <lastmod>${lastmod}</lastmod>`);
  if (Number.isFinite(priority) && priority >= 0 && priority <= 1) {
    lines.push(`    <priority>${priority.toFixed(1)}</priority>`);
  }
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

function getProjectId(project) {
  return String(project && project.id ? project.id : '').trim();
}

function getProjectTagSet(project) {
  const tools = Array.isArray(project?.tools) ? project.tools : [];
  const concepts = Array.isArray(project?.concepts) ? project.concepts : [];
  const tags = [...tools, ...concepts]
    .map((t) => normalizeWhitespace(t).toLowerCase())
    .filter(Boolean);
  return new Set(tags);
}

function renderProjectCardMedia(project, options = {}) {
  const img = String(project && project.image ? project.image : '').trim();
  if (!img) return '';

  const title = normalizeWhitespace(project?.title || '');
  const alt = normalizeWhitespace(project?.imageAlt || title);
  const width = Number(project?.imageWidth);
  const height = Number(project?.imageHeight);
  const sizeAttr = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
    ? ` width="${width}" height="${height}"`
    : '';

  const loading = normalizeWhitespace(options.loading || 'lazy');
  const sizes = normalizeWhitespace(options.sizes || '(max-width: 640px) 92vw, 340px');
  const loadingAttr = loading ? ` loading="${escapeHtml(loading)}"` : '';
  const sizesAttr = sizes ? ` sizes="${escapeHtml(sizes)}"` : '';

  const match = img.match(/\.(png|jpe?g)$/i);
  if (!match) {
    return `<img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}"${loadingAttr} decoding="async"${sizeAttr}${sizesAttr}>`;
  }

  const base = img.replace(/\.(png|jpe?g)$/i, '');
  const avif = buildResponsiveSrcset(base, 'avif', width);
  const webp = buildResponsiveSrcset(base, 'webp', width);

  if (avif || webp) {
    return `<picture>
      ${avif ? `<source srcset="${escapeHtml(avif)}" type="image/avif">` : ''}
      ${webp ? `<source srcset="${escapeHtml(webp)}" type="image/webp">` : ''}
      <img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}"${loadingAttr} decoding="async"${sizeAttr}${sizesAttr}>
    </picture>`;
  }

  return `<img src="${escapeHtml(img)}" alt="${escapeHtml(alt)}"${loadingAttr} decoding="async"${sizeAttr}${sizesAttr}>`;
}

function renderProjectRecommendationCard(project, options = {}) {
  const id = getProjectId(project);
  if (!id) return '';

  const title = normalizeWhitespace(project?.title || id);
  const subtitle = normalizeWhitespace(project?.subtitle || '');
  const badge = normalizeWhitespace(options.badge || '');
  const href = `portfolio/${encodeURIComponent(id)}`;

  const label = normalizeWhitespace(
    options.ariaLabel || (badge ? `${badge} project: ${title}` : `View project: ${title}`)
  );

  const badgeMarkup = badge ? `<div class="project-metric">${escapeHtml(badge)}</div>` : '';
  const media = renderProjectCardMedia(project, { loading: 'lazy', sizes: '(max-width: 960px) 92vw, 320px' });
  const safeSubtitle = subtitle ? `<div class="project-subtitle">${escapeHtml(subtitle)}</div>` : '';

  return `<a class="project-card ripple-in project-recommendation-card" role="listitem" href="${escapeHtml(href)}" aria-label="${escapeHtml(label)}">
      ${badgeMarkup}
      <div class="overlay"></div>
      <div class="project-text">
        <div class="project-title">${escapeHtml(title)}</div>
        ${safeSubtitle}
      </div>
      ${media}
    </a>`;
}

function selectRelatedProjects(projects, currentIndex, desiredCount, excludedIds) {
  if (!Array.isArray(projects) || projects.length === 0) return [];
  const desired = Number.isFinite(desiredCount) ? Math.max(0, Math.floor(desiredCount)) : 0;
  if (desired <= 0) return [];

  const current = projects[currentIndex];
  const currentId = getProjectId(current);
  if (!current || !currentId) return [];

  const excluded = excludedIds instanceof Set ? excludedIds : new Set();
  excluded.add(currentId);

  const currentTags = getProjectTagSet(current);
  const scored = projects
    .map((candidate, index) => {
      const candidateId = getProjectId(candidate);
      if (!candidate || !candidateId) return null;
      if (index === currentIndex) return null;
      if (excluded.has(candidateId)) return null;
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
  scored.forEach((item) => {
    if (selected.length >= desired) return;
    const candidateId = getProjectId(item.project);
    if (!candidateId || excluded.has(candidateId)) return;
    selected.push(item.project);
    excluded.add(candidateId);
  });

  for (let offset = 1; selected.length < desired; offset++) {
    const before = currentIndex - offset;
    const after = currentIndex + offset;
    const indexes = [before, after].filter((i) => i >= 0 && i < projects.length);
    if (indexes.length === 0) break;
    indexes.forEach((idx) => {
      if (selected.length >= desired) return;
      const candidate = projects[idx];
      const candidateId = getProjectId(candidate);
      if (!candidate || !candidateId) return;
      if (excluded.has(candidateId)) return;
      selected.push(candidate);
      excluded.add(candidateId);
    });
  }

  return selected.slice(0, desired);
}

function renderProjectRecommendationsSection(projects, currentIndex) {
  if (!Array.isArray(projects) || projects.length < 2) return '';
  const current = projects[currentIndex];
  const currentId = getProjectId(current);
  if (!current || !currentId) return '';

  const excluded = new Set([currentId]);
  const previous = currentIndex > 0 ? projects[currentIndex - 1] : null;
  const next = currentIndex < projects.length - 1 ? projects[currentIndex + 1] : null;

  const navCards = [];
  if (previous) {
    excluded.add(getProjectId(previous));
    navCards.push(renderProjectRecommendationCard(previous, { badge: 'Previous' }));
  }
  if (next) {
    excluded.add(getProjectId(next));
    navCards.push(renderProjectRecommendationCard(next, { badge: 'Next' }));
  }

  const related = selectRelatedProjects(projects, currentIndex, 3, excluded);
  const relatedCards = related.map((p) => renderProjectRecommendationCard(p)).filter(Boolean);

  if (!navCards.length && !relatedCards.length) return '';

  const navMarkup = navCards.length
    ? `<div class="project-recommendations-nav" role="list">
        ${navCards.join('\n        ')}
      </div>`
    : '';

  const relatedMarkup = relatedCards.length
    ? `<h3 class="project-recommendations-subtitle">Related projects</h3>
      <div class="project-recommendations-grid" role="list">
        ${relatedCards.join('\n        ')}
      </div>`
    : '';

  return `<section class="project-section project-recommendations" aria-label="More projects">
      <h2 class="section-title">More Projects</h2>
      ${navMarkup}
      ${relatedMarkup}
    </section>`;
}

function renderProjectPage(project, options = {}) {
  const id = String(project.id || '').trim();
  const title = normalizeWhitespace(project.title || id);
  const subtitle = normalizeWhitespace(project.subtitle || '');
  const description = toMetaDescription(project);
  const canonicalPath = `/portfolio/${encodeURIComponent(id)}`;
  const canonicalUrl = `${SITE_ORIGIN}${canonicalPath}`;
  const ogImage = toAbsoluteUrl(project.image || 'img/hero/head.jpg');
  const ogImageAlt = normalizeWhitespace(project.imageAlt || `Preview image for ${title}`);

  const tools = Array.isArray(project.tools) ? project.tools : [];
  const concepts = Array.isArray(project.concepts) ? project.concepts : [];
  const actions = Array.isArray(project.actions) ? project.actions : [];
  const results = Array.isArray(project.results) ? project.results : [];
  const resources = Array.isArray(project.resources) ? project.resources : [];
  const role = project.role;
  const notes = normalizeWhitespace(project.notes || '');

  const tags = [...new Set([...concepts, ...tools])]
    .map((t) => normalizeWhitespace(t))
    .filter(Boolean);

  const embed = project && typeof project.embed === 'object' ? project.embed : null;
  const demoInstructions = project && typeof project.demoInstructions === 'object'
    ? project.demoInstructions
    : null;
  const tableauPreconnect = embed && String(embed.type || '').trim() === 'tableau'
    ? '  <link rel="preconnect" href="https://public.tableau.com" crossorigin>\n'
    : '';

  const ogImageWidth = Number(project.imageWidth);
  const ogImageHeight = Number(project.imageHeight);
  const ogImageDimensionsMeta = Number.isFinite(ogImageWidth) && Number.isFinite(ogImageHeight) && ogImageWidth > 0 && ogImageHeight > 0
    ? `  <meta property="og:image:width" content="${escapeHtml(ogImageWidth)}">\n  <meta property="og:image:height" content="${escapeHtml(ogImageHeight)}">\n`
    : '';

  const projectLd = {
    '@type': 'CreativeWork',
    name: title,
    description,
    url: canonicalUrl,
    image: ogImage
  };
  const breadcrumbsLd = {
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Home', item: `${SITE_ORIGIN}/` },
      { '@type': 'ListItem', position: 2, name: 'Portfolio', item: `${SITE_ORIGIN}/portfolio` },
      { '@type': 'ListItem', position: 3, name: title, item: canonicalUrl }
    ]
  };
  const ldJson = JSON.stringify({ '@context': 'https://schema.org', '@graph': [projectLd, breadcrumbsLd] })
    .replace(/</g, '\\u003c');

  const safeTagPills = tags.length
    ? `<div class="project-tags" role="list">
      ${tags.map((t) => `<span class="project-tag" role="listitem">${escapeHtml(t)}</span>`).join('\n      ')}
    </div>`
    : '';

  const safeProblem = normalizeWhitespace(project.problem || '');

  const hasResources = resources.length > 0;
  const hasNotes = Boolean(notes);

  const safeResources = hasResources
    ? `<section class="project-section" id="links">
      <h2 class="section-title">Links</h2>
      <div class="project-links" role="list">
        ${resources.map((r) => {
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
      </div>
    </section>`
    : '';

  const safeNotes = hasNotes
    ? `<section class="project-section" id="notes">
      <h2 class="section-title">Notes</h2>
      <p class="project-lead">${escapeHtml(notes)}</p>
    </section>`
    : '';

  const allProjects = Array.isArray(options.projects) ? options.projects : null;
  const projectIndex = Number.isInteger(options.index) ? options.index : -1;
  const recommendations = allProjects && projectIndex >= 0
    ? renderProjectRecommendationsSection(allProjects, projectIndex)
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
      </dl>
    </section>`;

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

  const renderEmbeddedMedia = (options = {}) => {
    if (!embed) return '';
    const lazy = options && options.lazy === true;
    const type = String(embed.type || '').trim();
    if (type === 'iframe') {
      const src = String(embed.url || '').trim();
      if (!src) return '';
      const srcAttr = lazy ? ` data-src="${escapeHtml(src)}"` : ` src="${escapeHtml(src)}"`;
      return `<div class="project-media project-embed project-embed-iframe">
        <iframe class="project-embed-frame"${srcAttr} title="${escapeHtml(title)} interactive demo" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    if (type === 'tableau') {
      const base = String(embed.base || '').trim();
      if (!base) return '';
      const joiner = base.includes('?') ? '&' : '?';
      const src = `${base}${joiner}:showVizHome=no&:embed=y`;
      const srcAttr = lazy ? ` data-src="${escapeHtml(src)}"` : ` src="${escapeHtml(src)}"`;
      return `<div class="project-media project-embed project-embed-tableau">
        <iframe class="project-embed-frame"${srcAttr} title="${escapeHtml(title)} interactive dashboard" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    return '';
  };

  const renderDemoTabs = () => {
    if (!embed) return '';
    const safeId = toDomIdSafe(id);
    const baseId = `project-demo-${safeId}`;
    const tabInstructionsId = `${baseId}-tab-instructions`;
    const tabDemoId = `${baseId}-tab-demo`;
    const panelInstructionsId = `${baseId}-panel-instructions`;
    const panelDemoId = `${baseId}-panel-demo`;

    const lead = normalizeWhitespace(demoInstructions?.lead || '');
    const bullets = normalizeTextArray(demoInstructions?.bullets);
    const safeLead = lead ? `<p class="project-demo-lead">${escapeHtml(lead)}</p>` : '';
    const safeBullets = bullets.length
      ? `<ul class="project-demo-list">
        ${bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join('\n        ')}
      </ul>`
      : '';

    const demoOpenHref = (() => {
      const type = String(embed.type || '').trim();
      if (type === 'iframe') return String(embed.url || '').trim();
      if (type === 'tableau') {
        const base = String(embed.base || '').trim();
        if (!base) return '';
        const joiner = base.includes('?') ? '&' : '?';
        return `${base}${joiner}:showVizHome=no&:embed=y`;
      }
      return '';
    })();
    const openInNewTab = demoOpenHref
      ? `<a class="btn-secondary" href="${escapeHtml(demoOpenHref)}" target="_blank" rel="noopener noreferrer">Open demo in new tab</a>`
      : '';

    return `<section class="project-demo-shell" data-demo-tabs="true" aria-label="Demo">
      <div class="project-demo-tabs" role="tablist" aria-label="Demo tabs">
        <button class="project-demo-tab is-active" type="button" role="tab" id="${escapeHtml(tabInstructionsId)}" aria-controls="${escapeHtml(panelInstructionsId)}" aria-selected="true">How to use</button>
        <button class="project-demo-tab" type="button" role="tab" id="${escapeHtml(tabDemoId)}" aria-controls="${escapeHtml(panelDemoId)}" aria-selected="false" tabindex="-1">Demo</button>
      </div>

      <div class="project-demo-panels">
        <section class="project-demo-panel is-active" data-demo-panel="instructions" role="tabpanel" id="${escapeHtml(panelInstructionsId)}" aria-labelledby="${escapeHtml(tabInstructionsId)}">
          <div class="project-demo-panel-inner">
            ${safeLead}
            ${safeBullets}
            <div class="project-demo-actions">
              <button class="btn-primary" type="button" data-demo-tabs-open="demo">Open demo</button>
              ${openInNewTab}
            </div>
          </div>
        </section>

        <section class="project-demo-panel" data-demo-panel="demo" role="tabpanel" id="${escapeHtml(panelDemoId)}" aria-labelledby="${escapeHtml(tabDemoId)}" hidden>
          <div class="project-demo-panel-inner">
            ${renderEmbeddedMedia({ lazy: true })}
          </div>
        </section>
      </div>
    </section>`;
  };

  const media = (() => {
    if (embed) return '';
    const video = renderVideoMedia();
    if (video) return video;
    return renderImageMedia();
  })();

  const demoTabs = embed ? renderDemoTabs() : '';

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
  <meta property="og:site_name" content="Daniel Short – Data Science &amp; Analytics">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">
  <meta property="og:image" content="${escapeHtml(ogImage)}">
  <meta property="og:image:alt" content="${escapeHtml(ogImageAlt)}">
${ogImageDimensionsMeta}
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:site" content="@danielshort3">

  <meta name="theme-color" content="#0D1117">
  <link rel="stylesheet" href="dist/styles.css">
  <link rel="icon" href="favicon.ico" sizes="any">
  <link rel="icon" type="image/png" sizes="16x16" href="img/ui/logo-16.png">
  <link rel="icon" type="image/png" sizes="32x32" href="img/ui/logo-32.png">
  <link rel="icon" type="image/png" sizes="64x64" href="img/ui/logo-64.png">
  <link rel="icon" type="image/png" sizes="192x192" href="img/ui/logo-192.png">
  <link rel="apple-touch-icon" sizes="180x180" href="img/ui/logo-180.png">
${tableauPreconnect}

  <!-- Local fonts with legacy reference retained for tooling: https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Poppins:wght@500;600&display=swap -->
  <script src="js/common/no-js.js"></script>
  <script type="application/ld+json">
    ${ldJson}
  </script>
</head>
<body data-page="project" class="project-page">
  <a href="#main" class="skip-link">Skip to main content</a>
  <header id="combined-header-nav"></header>

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
		        ${demoTabs || media}
		        ${starSummary}
		        ${safeResources}
		        ${safeNotes}
		        ${recommendations}
		      </div>
		    </section>
		  </main>

	  <footer>
	    <nav class="privacy-links" aria-label="Privacy shortcuts">
	      <button id="privacy-settings-link" type="button" class="pcz-link">Privacy settings</button>
	      <a href="privacy#prefs-title" class="pcz-link" data-consent-open="true">Do Not Sell/Share My Personal Information</a>
	      <a href="sitemap.xml" class="pcz-link">Sitemap</a>
	    </nav>
	  </footer>

  <script defer src="js/common/common.js"></script>
  <script defer src="js/navigation/navigation.js"></script>
  <script defer src="js/animations/animations.js"></script>
  <script src="js/privacy/config.js"></script>
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

function writeSitemap(projects) {
  const previousEntries = loadSitemapCache();
  const nextEntries = new Map();
  const baseEntries = [
    { loc: `${SITE_ORIGIN}/`, sourceFile: 'index.html', priority: 1.0 },
    { loc: `${SITE_ORIGIN}/portfolio`, sourceFile: 'pages/portfolio.html', priority: 0.9 },
    { loc: `${SITE_ORIGIN}/resume`, sourceFile: 'pages/resume.html', priority: 0.9 },
    { loc: `${SITE_ORIGIN}/contact`, sourceFile: 'pages/contact.html', priority: 0.7 },
    { loc: `${SITE_ORIGIN}/tools`, sourceFile: 'pages/tools.html', priority: 0.7 },
    { loc: `${SITE_ORIGIN}/contributions`, sourceFile: 'pages/contributions.html', priority: 0.6 },
    { loc: `${SITE_ORIGIN}/resume-pdf`, sourceFile: 'pages/resume-pdf.html', priority: 0.5 },
    { loc: `${SITE_ORIGIN}/privacy`, sourceFile: 'pages/privacy.html', priority: 0.2 },
    { loc: `${SITE_ORIGIN}/sitemap`, sourceFile: 'pages/sitemap.html', priority: 0.2 },
    { loc: `${SITE_ORIGIN}/tools/dashboard`, sourceFile: 'pages/tools-dashboard.html', priority: 0.3 }
  ];

  const toolEntries = loadToolUrls().map((loc) => {
    const slug = String(loc).replace(`${SITE_ORIGIN}/tools/`, '').replace(/^\/+/, '');
    return { loc, sourceFile: `pages/${slug}.html`, priority: 0.6 };
  });

  const projectEntries = projects
    .map((p) => String(p.id || '').trim())
    .filter(Boolean)
    .map((id) => ({
      loc: `${SITE_ORIGIN}/portfolio/${encodeURIComponent(id)}`,
      sourceFile: `pages/portfolio/${id}.html`,
      priority: 0.6
    }));

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...baseEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries)).filter(Boolean),
    '',
    ...toolEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries)).filter(Boolean),
    '',
    ...projectEntries.map((entry) => toSitemapUrlEntry(entry, previousEntries, nextEntries)).filter(Boolean),
    '</urlset>',
    ''
  ].join('\n');
  fs.writeFileSync(sitemapPath, xml, 'utf8');
  writeSitemapCache(nextEntries);
}

function main() {
  const projects = loadProjects().filter(isPublishedProject);
  writeProjectPages(projects);
  writeSitemap(projects);
  process.stdout.write(`Generated ${projects.length} project pages in pages/portfolio/ and updated sitemap.xml\n`);
}

main();
