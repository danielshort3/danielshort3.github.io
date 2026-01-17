#!/usr/bin/env node
/*
  Generate SEO-friendly, shareable project pages under /portfolio/<id>.
  - Keeps existing /portfolio.html?project=<id> deep links working.
  - Outputs static HTML pages in ./pages/portfolio/<id>.html
  - Updates ./sitemap.xml to include project URLs.
  No external deps.
*/
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
const dataFile = path.join(root, 'js', 'portfolio', 'projects-data.js');
const outDir = path.join(root, 'pages', 'portfolio');
const sitemapPath = path.join(root, 'sitemap.xml');
const SITE_ORIGIN = 'https://danielshort.me';

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

function renderCaseStudySections(sections) {
  if (!Array.isArray(sections) || sections.length === 0) return '';
  const blocks = sections
    .map((sec) => {
      if (!sec || typeof sec !== 'object') return '';
      const title = normalizeWhitespace(sec.title);
      if (!title) return '';

      const lead = normalizeWhitespace(sec.lead || '');
      const paragraphs = normalizeTextArray(sec.paragraphs);
      const bullets = normalizeTextArray(sec.bullets);

      const parts = [];
      if (lead) {
        parts.push(`<p class="project-lead">${escapeHtml(lead)}</p>`);
      }
      paragraphs.forEach((p) => {
        parts.push(`<p class="project-lead">${escapeHtml(p)}</p>`);
      });
      if (bullets.length) {
        parts.push(`<ul class="project-list">
        ${bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join('\n        ')}
      </ul>`);
      }

      const body = parts.length ? `\n      ${parts.join('\n      ')}\n    ` : '';
      return `<section class="project-section">
      <h2 class="section-title">${escapeHtml(title)}</h2>${body}</section>`;
    })
    .filter(Boolean)
    .join('\n');

  return blocks ? `${blocks}\n` : '';
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

function renderProjectPage(project) {
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
  const tableauPreconnect = embed && String(embed.type || '').trim() === 'tableau'
    ? '  <link rel="preconnect" href="https://public.tableau.com" crossorigin>\n'
    : '';

  const projectLd = {
    '@context': 'https://schema.org',
    '@type': 'CreativeWork',
    name: title,
    description,
    url: canonicalUrl,
    image: ogImage
  };
  const ldJson = JSON.stringify(projectLd).replace(/</g, '\\u003c');

  const safeTagPills = tags.length
    ? `<div class="project-tags" role="list">
      ${tags.map((t) => `<span class="project-tag" role="listitem">${escapeHtml(t)}</span>`).join('\n      ')}
    </div>`
    : '';

  const safeResources = resources.length
    ? `<section class="project-section">
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

  const safeRole = (() => {
    if (!role) return '';
    if (Array.isArray(role) && role.length) {
      return `<section class="project-section">
      <h2 class="section-title">Role</h2>
      <ul class="project-list">
        ${role.map((item) => `<li>${escapeHtml(normalizeWhitespace(item))}</li>`).join('\n        ')}
      </ul>
    </section>`;
    }
    const text = normalizeWhitespace(role);
    if (!text) return '';
    return `<section class="project-section">
      <h2 class="section-title">Role</h2>
      <p class="project-lead">${escapeHtml(text)}</p>
    </section>`;
  })();

  const safeActions = actions.length
    ? `<section class="project-section">
      <h2 class="section-title">Approach</h2>
      <ul class="project-list">
        ${actions.map((a) => `<li>${escapeHtml(normalizeWhitespace(a))}</li>`).join('\n        ')}
      </ul>
    </section>`
    : '';

  const safeResults = results.length
    ? `<section class="project-section">
      <h2 class="section-title">Impact</h2>
      <ul class="project-list">
        ${results.map((r) => `<li>${escapeHtml(normalizeWhitespace(r))}</li>`).join('\n        ')}
      </ul>
    </section>`
    : '';

  const safeProblem = normalizeWhitespace(project.problem || '');
  const safeOverview = safeProblem
    ? `<section class="project-section">
      <h2 class="section-title">Context</h2>
      <p class="project-lead">${escapeHtml(safeProblem)}</p>
    </section>`
    : '';

  const safeNotes = notes
    ? `<section class="project-section">
      <h2 class="section-title">Notes</h2>
      <p class="project-lead">${escapeHtml(notes)}</p>
    </section>`
    : '';

  const safeCaseStudy = renderCaseStudySections(project.caseStudy);

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

  const renderEmbeddedMedia = () => {
    if (!embed) return '';
    const type = String(embed.type || '').trim();
    if (type === 'iframe') {
      const src = String(embed.url || '').trim();
      if (!src) return '';
      return `<div class="project-media project-embed project-embed-iframe">
        <iframe class="project-embed-frame" src="${escapeHtml(src)}" title="${escapeHtml(title)} interactive demo" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    if (type === 'tableau') {
      const base = String(embed.base || '').trim();
      if (!base) return '';
      const joiner = base.includes('?') ? '&' : '?';
      const src = `${base}${joiner}:showVizHome=no&:embed=y`;
      return `<div class="project-media project-embed project-embed-tableau">
        <iframe class="project-embed-frame" src="${escapeHtml(src)}" title="${escapeHtml(title)} interactive dashboard" loading="lazy" allowfullscreen></iframe>
      </div>`;
    }
    return '';
  };

  const media = (() => {
    const embedded = renderEmbeddedMedia();
    if (embedded) return embedded;
    return renderImageMedia();
  })();

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
          <a class="btn-primary hero-cta" href="portfolio.html?project=${escapeHtml(encodeURIComponent(id))}">Open interactive view</a>
          <a class="btn-secondary hero-cta" href="portfolio.html">View all projects</a>
        </div>
        ${safeTagPills}
      </div>
    </section>

    <section class="project-body">
      <div class="wrapper">
        ${media}
        ${safeOverview}
        ${safeRole}
        ${safeActions}
        ${safeResults}
        ${safeCaseStudy}
        ${safeResources}
        ${safeNotes}
      </div>
    </section>
  </main>

  <footer>
    <nav class="privacy-links" aria-label="Privacy shortcuts">
      <button id="privacy-settings-link" type="button" class="pcz-link">Privacy settings</button>
      <a href="privacy.html#prefs-title" class="pcz-link" data-consent-open="true">Do Not Sell/Share My Personal Information</a>
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

  projects.forEach((project) => {
    const id = String(project.id || '').trim();
    if (!id) throw new Error('Project missing id');
    const outPath = path.join(outDir, `${id}.html`);
    fs.writeFileSync(outPath, renderProjectPage(project), 'utf8');
  });
}

function writeSitemap(projects) {
  const baseUrls = [
    `${SITE_ORIGIN}/`,
    `${SITE_ORIGIN}/portfolio.html`,
    `${SITE_ORIGIN}/contributions.html`,
    `${SITE_ORIGIN}/contact.html`,
    `${SITE_ORIGIN}/resume.html`,
    `${SITE_ORIGIN}/privacy.html`
  ];

  const projectUrls = projects
    .map((p) => String(p.id || '').trim())
    .filter(Boolean)
    .map((id) => `${SITE_ORIGIN}/portfolio/${encodeURIComponent(id)}`);

  const all = [...baseUrls, ...projectUrls];
  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...all.map((loc) => `  <url><loc>${loc}</loc></url>`),
    '</urlset>',
    ''
  ].join('\n');
  fs.writeFileSync(sitemapPath, xml, 'utf8');
}

function main() {
  const projects = loadProjects().filter(isPublishedProject);
  writeProjectPages(projects);
  writeSitemap(projects);
  process.stdout.write(`Generated ${projects.length} project pages in pages/portfolio/ and updated sitemap.xml\n`);
}

main();
