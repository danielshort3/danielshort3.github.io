/* contributions.js - Build contributions UI components.
   Contribution data now lives in contributions-data.js */

function buildDocLinks(item, opts = {}) {
  const links = [];

  if (item.pdf) {
    links.push(`
      <a href="${item.pdf}" target="_blank" rel="noopener noreferrer" class="doc-link" aria-label="Open PDF" download>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <path d="M14 2v6h6"/>
          <path d="M8 15h8"/>
        </svg>
      </a>
    `);
  }

  if (item.link) {
    links.push(`
      <a href="${item.link}" target="_blank" rel="noopener noreferrer" class="doc-link" aria-label="Open external link">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 3h7v7"/>
          <path d="M10 14L21 3"/>
          <path d="M21 14v7h-7"/>
          <path d="M3 10v11h11"/>
        </svg>
      </a>
    `);
  }

  if (!links.length) return '';

  const classes = ['doc-links'];
  if (opts.compact) classes.push('doc-links-compact');
  return `<div class="${classes.join(' ')}">${links.join('')}</div>`;
}

function buildMetaLine(item) {
  const parts = [];
  if (item.year && item.year !== 'Earlier') parts.push(item.year);
  if (item.quarter) parts.push(item.quarter);
  if (item.focus) parts.push(item.focus);
  return parts.length ? `<p class="doc-meta">${parts.join(' · ')}</p>` : '';
}

function buildFeaturedCard(item) {
  const card = document.createElement('article');
  card.className = 'doc-card featured-doc';
  card.innerHTML = `
    <div class="doc-layout">
      <p class="doc-label">Latest contribution</p>
      <h3 class="doc-title">${item.title}</h3>
      ${buildMetaLine(item)}
      <div class="doc-footer">
        ${item.role ? `<p class="doc-role">${item.role}</p>` : ''}
        ${buildDocLinks(item) || ''}
      </div>
    </div>`;
  return card;
}

function buildImpactCard(section, includeDownloadFallback) {
  if (!section.impact) return null;
  const card = document.createElement('article');
  card.className = 'impact-card';
  const metrics = Array.isArray(section.impact.metrics)
    ? section.impact.metrics.map(metric => `
        <div>
          <p class="impact-metric-value">${metric.value}</p>
          <p class="impact-metric-label">${metric.label}</p>
        </div>
      `).join('')
    : '';

  card.innerHTML = `
    <p class="impact-label">Key impact</p>
    <h3>${section.impact.title}</h3>
    <p class="impact-summary">${section.impact.summary}</p>
    ${metrics ? `<div class="impact-metrics">${metrics}</div>` : ''}
  `;

  if (includeDownloadFallback && section.download) {
    card.appendChild(buildDownloadButton(section.download));
  }

  return card;
}

function buildDownloadButton(download) {
  if (!download || !download.file) return null;
  const wrapper = document.createElement('div');
  wrapper.className = 'download-block';
  const size = download.size ? `<span class="download-size">${download.size}</span>` : '';
  wrapper.innerHTML = `
    <a href="${download.file}" class="download-pill" download>
      <span class="download-label">${download.label}</span>
      ${size}
    </a>
    ${download.description ? `<p class="download-description">${download.description}</p>` : ''}
  `;
  return wrapper;
}

function groupByYear(items = []) {
  const groups = new Map();
  items.forEach(item => {
    const year = item.year || 'Earlier';
    if (!groups.has(year)) groups.set(year, []);
    groups.get(year).push(item);
  });
  return Array.from(groups.entries())
    .sort((a, b) => {
      const aYear = parseInt(a[0], 10);
      const bYear = parseInt(b[0], 10);
      if (Number.isFinite(aYear) && Number.isFinite(bYear)) return bYear - aYear;
      if (Number.isFinite(aYear)) return 1;
      if (Number.isFinite(bYear)) return -1;
      return 0;
    })
    .map(([year, entries]) => ({ year, entries }));
}

function buildTimelineMeta(item) {
  const parts = [];
  if (item.year && item.year !== 'Earlier') parts.push(item.year);
  if (item.quarter) parts.push(item.quarter);
  if (item.focus) parts.push(item.focus);
  return parts.length ? `<span class="timeline-item-meta">${parts.join(' · ')}</span>` : '';
}

function buildTimeline(section, previousItems) {
  if (!previousItems.length) return null;

  const shell = document.createElement('section');
  shell.className = 'timeline-shell';

  const header = document.createElement('div');
  header.className = 'timeline-header';
  header.innerHTML = `
    <div>
      <p class="timeline-label">Historical contributions</p>
      <p class="timeline-subtext">Expand a year to scan supporting work.</p>
    </div>
  `;

  const downloadButton = buildDownloadButton(section.download);
  if (downloadButton) header.appendChild(downloadButton);

  shell.appendChild(header);

  const timeline = document.createElement('div');
  timeline.className = 'contrib-timeline';

  groupByYear(previousItems).forEach(group => {
    const details = document.createElement('details');
    details.className = 'timeline-year';
    details.dataset.year = group.year;

    const summary = document.createElement('summary');
    summary.innerHTML = `
      <span class="timeline-year-pill">${group.year}</span>
      <span class="timeline-year-meta">${group.entries.length} ${group.entries.length === 1 ? 'entry' : 'entries'}</span>
    `;
    details.appendChild(summary);

    const list = document.createElement('ul');
    list.className = 'timeline-list';

    group.entries.forEach(item => {
      const li = document.createElement('li');
      li.className = 'timeline-item';
      li.innerHTML = `
        <div class="timeline-item-text">
          <span class="timeline-item-title">${item.title}</span>
          ${buildTimelineMeta(item)}
          ${item.role ? `<span class="timeline-item-role">${item.role}</span>` : ''}
        </div>
        ${buildDocLinks(item, { compact: true }) || ''}
      `;
      list.appendChild(li);
    });

    details.appendChild(list);
    timeline.appendChild(details);
  });

  shell.appendChild(timeline);
  return shell;
}

function buildHeroOverview(sections) {
  const list = document.getElementById('contrib-hero-list');
  if (!list) return;
  list.innerHTML = '';

  sections.forEach(section => {
    const li = document.createElement('li');
    li.innerHTML = `
      <a href="#${section.id}" data-target="${section.id}">
        <span class="hero-mini-label">${section.shortTitle || section.heading}</span>
        <span class="hero-mini-desc">${section.heroSummary || section.desc || ''}</span>
      </a>
    `;
    list.appendChild(li);
  });
}

function buildNav(sections) {
  const list = document.getElementById('contrib-nav-list');
  if (!list) return;
  list.innerHTML = '';

  sections.forEach(section => {
    const li = document.createElement('li');
    li.innerHTML = `
      <a href="#${section.id}" data-target="${section.id}">
        <span class="nav-label">${section.shortTitle || section.heading}</span>
        <span class="nav-meta">${section.navSummary || ''}</span>
      </a>
    `;
    list.appendChild(li);
  });
}

function initScrollSpy(sections) {
  const navLinks = Array.from(document.querySelectorAll('.contrib-nav a[data-target]'));
  if (!navLinks.length || typeof IntersectionObserver === 'undefined') return;

  const linkMap = new Map();
  navLinks.forEach(link => linkMap.set(link.dataset.target, link));

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const link = linkMap.get(entry.target.id);
      if (!link) return;
      if (entry.isIntersecting) {
        navLinks.forEach(l => l.classList.remove('is-active'));
        link.classList.add('is-active');
      }
    });
  }, { threshold: 0.4 });

  sections.forEach(section => {
    const el = document.getElementById(section.id);
    if (el) observer.observe(el);
  });
}

function buildSections(sections) {
  const root = document.getElementById('contrib-root');
  if (!root) return;
  root.innerHTML = '';

  sections.forEach((section, index) => {
    if (!Array.isArray(section.items) || !section.items.length) return;
    const [latest, ...previousItems] = section.items;

    const container = document.createElement('section');
    container.className = 'surface-band reveal contrib-section';
    container.id = section.id;
    container.dataset.heading = section.heading;

    const wrap = document.createElement('div');
    wrap.className = 'wrapper';
    container.appendChild(wrap);

    wrap.insertAdjacentHTML('beforeend', `
      <div class="section-head">
        <div>
          <h2 class="section-title">${section.heading}</h2>
          <p class="section-desc">${section.desc || ''}</p>
        </div>
      </div>
    `);

    const stack = document.createElement('div');
    stack.className = 'contrib-stack';

    const featureRow = document.createElement('div');
    featureRow.className = 'contrib-feature-row';
    featureRow.appendChild(buildFeaturedCard(latest));

    const timeline = buildTimeline(section, previousItems);
    const impactCard = buildImpactCard(section, !timeline);
    if (impactCard) featureRow.appendChild(impactCard);

    stack.appendChild(featureRow);
    if (timeline) stack.appendChild(timeline);

    wrap.appendChild(stack);
    root.appendChild(container);

    if (index !== sections.length - 1) {
      const gap = document.createElement('section');
      gap.className = 'contrib-gap';
      root.appendChild(gap);
    }
  });
}

function initContributions() {
  const sections = window.contributions;
  if (!Array.isArray(sections)) return;
  buildHeroOverview(sections);
  buildNav(sections);
  buildSections(sections);
  initScrollSpy(sections);
}

document.addEventListener('DOMContentLoaded', initContributions);
