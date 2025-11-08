/* contributions.js - Build contributions UI components.
   Contribution data now lives in contributions-data.js */

const MONTH_PATTERN = '(January|February|March|April|May|June|July|August|September|October|November|December)';
const MONTH_DAY_YEAR_RE = new RegExp(`${MONTH_PATTERN}\\s+\\d{1,2},\\s+\\d{4}`);
const MONTH_YEAR_RE = new RegExp(`${MONTH_PATTERN}\\s+\\d{4}`);
const YEAR_RE = /(20\d{2})/;
const META_LABELS = {
  audience   : 'Audience',
  cadence    : 'Cadence',
  focus      : 'Focus',
  deliverable: 'Signature deliverable'
};

function slugify(text){
  return (text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'section';
}

function pluralize(count, singular){
  const base = singular || 'item';
  return count === 1 ? base : `${base}s`;
}

function formatMonthYear(date){
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  return date.toLocaleString('en-US', { month: 'short', year: 'numeric' });
}

function parseMonthYear(match){
  if (!match) return null;
  const [month, year] = match[0].split(/\s+/);
  return new Date(`${month} 1, ${year} 00:00:00 UTC`);
}

function getItemDate(item){
  if (!item) return null;
  if (item._parsedDate !== undefined) return item._parsedDate;

  if (item.date) {
    const explicit = new Date(item.date);
    if (!Number.isNaN(explicit.getTime())) {
      item._parsedDate = explicit;
      return explicit;
    }
  }

  const haystack = `${item.title ?? ''} ${item.role ?? ''}`;
  const monthDayYear = haystack.match(MONTH_DAY_YEAR_RE);
  if (monthDayYear) {
    const inferred = new Date(`${monthDayYear[0]} 00:00:00 UTC`);
    if (!Number.isNaN(inferred.getTime())) {
      item._parsedDate = inferred;
      return inferred;
    }
  }

  const monthYear = haystack.match(MONTH_YEAR_RE);
  if (monthYear) {
    const inferred = parseMonthYear(monthYear);
    if (!Number.isNaN(inferred.getTime())) {
      item._parsedDate = inferred;
      return inferred;
    }
  }

  const yearOnly = haystack.match(YEAR_RE);
  if (yearOnly) {
    const inferred = new Date(`${yearOnly[1]}-01-01T00:00:00Z`);
    if (!Number.isNaN(inferred.getTime())) {
      item._parsedDate = inferred;
      return inferred;
    }
  }

  item._parsedDate = null;
  return null;
}

function extractYear(item){
  if (item.year) return String(item.year);
  const date = getItemDate(item);
  if (date) return String(date.getUTCFullYear());
  const haystack = `${item.title ?? ''} ${item.role ?? ''}`;
  const match = haystack.match(YEAR_RE);
  return match ? match[1] : 'Earlier';
}

function sortItems(items = []){
  return items
    .map((entry, index) => ({ entry, index }))
    .sort((a, b) => {
      const dateA = getItemDate(a.entry);
      const dateB = getItemDate(b.entry);
      if (dateA && dateB && dateA.getTime() !== dateB.getTime()) {
        return dateB.getTime() - dateA.getTime();
      }
      if (dateA && !dateB) return -1;
      if (!dateA && dateB) return 1;
      return a.index - b.index;
    })
    .map(pair => pair.entry);
}

function buildDocLinks(item, opts = {}){
  const links = [];

  if (item.pdf) {
    links.push(`
      <a href="${item.pdf}" target="_blank" rel="noopener noreferrer" class="doc-link" aria-label="Open PDF" download>
        <span class="doc-link-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <path d="M14 2v6h6"/>
            <path d="M8 15h8"/>
          </svg>
        </span>
        <span class="doc-link-label">PDF</span>
      </a>
    `);
  }

  if (item.link) {
    links.push(`
      <a href="${item.link}" target="_blank" rel="noopener noreferrer" class="doc-link" aria-label="Open external link">
        <span class="doc-link-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 3h7v7"/>
            <path d="M10 14L21 3"/>
            <path d="M21 14v7h-7"/>
            <path d="M3 10v11h11"/>
          </svg>
        </span>
        <span class="doc-link-label">Live</span>
      </a>
    `);
  }

  if (!links.length) return '';

  const classes = ['doc-links'];
  if (opts.compact) classes.push('doc-links-compact');
  return `<div class="${classes.join(' ')}">${links.join('')}</div>`;
}

function buildFeaturedCard(item){
  if (!item) return null;
  const date = getItemDate(item);
  const dateLabel = date ? ` · ${formatMonthYear(date)}` : '';

  const card = document.createElement('article');
  card.className = 'doc-card featured-doc';
  card.innerHTML = `
    <div class="doc-layout">
      <p class="doc-label">Latest contribution${dateLabel}</p>
      <h3 class="doc-title">${item.title}</h3>
      <div class="doc-footer">
        ${item.role ? `<p class="doc-role">${item.role}</p>` : ''}
        ${buildDocLinks(item)}
      </div>
    </div>`;
  return card;
}

function buildAtGlanceCard(section){
  if (!section.meta) return null;
  const entries = Object.keys(META_LABELS)
    .map(key => {
      if (!section.meta[key]) return '';
      return `
        <div class="doc-meta-item">
          <p class="doc-meta-label">${META_LABELS[key]}</p>
          <p class="doc-meta-value">${section.meta[key]}</p>
        </div>`;
    })
    .filter(Boolean)
    .join('');
  if (!entries) return null;

  const card = document.createElement('article');
  card.className = 'doc-card doc-card-meta';
  card.innerHTML = `
    <div class="doc-layout">
      <p class="doc-label">At a glance</p>
      <div class="doc-meta-grid">
        ${entries}
      </div>
    </div>`;
  return card;
}

function groupByYear(items){
  const buckets = [];
  const map = new Map();

  items.forEach(entry => {
    const year = extractYear(entry);
    if (!map.has(year)) {
      const bucket = { year, label: year, entries: [] };
      map.set(year, bucket);
      buckets.push(bucket);
    }
    map.get(year).entries.push(entry);
  });

  return buckets;
}

function groupByQuarter(items){
  const buckets = [];
  const map = new Map();

  items.forEach(entry => {
    const date = getItemDate(entry);
    if (!date) {
      const fallbackKey = 'earlier';
      if (!map.has(fallbackKey)) {
        const bucket = { label: 'Earlier', key: fallbackKey, entries: [] };
        map.set(fallbackKey, bucket);
        buckets.push(bucket);
      }
      map.get(fallbackKey).entries.push(entry);
      return;
    }

    const quarter = Math.floor(date.getUTCMonth() / 3) + 1;
    const year = date.getUTCFullYear();
    const key = `${year}-Q${quarter}`;

    if (!map.has(key)) {
      const bucket = { label: `Q${quarter} ${year}`, key, entries: [] };
      map.set(key, bucket);
      buckets.push(bucket);
    }
    map.get(key).entries.push(entry);
  });

  return buckets;
}

function buildTimeline(previousItems, options = {}){
  if (!previousItems.length) return null;

  const groups = options.groupBy === 'quarter'
    ? groupByQuarter(previousItems)
    : groupByYear(previousItems);

  if (!groups.length) return null;

  const timeline = document.createElement('div');
  timeline.className = 'contrib-timeline';

  groups.forEach((group, index) => {
    const details = document.createElement('details');
    details.className = 'timeline-year';
    details.dataset.group = group.label;
    if (group.year) details.dataset.year = group.year;
    if (!index) details.open = true;
    if (!index) details.dataset.containsLatest = 'true';

    const summary = document.createElement('summary');
    const noun = options.timelineLabel || 'report';
    summary.innerHTML = `
      <span class="timeline-year-pill">${group.label}</span>
      <span class="timeline-year-meta">${group.entries.length} ${pluralize(group.entries.length, noun)}</span>
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
          ${item.role ? `<span class="timeline-item-role">${item.role}</span>` : ''}
        </div>
        ${buildDocLinks(item, { compact: true }) || ''}
      `;
      list.appendChild(li);
    });

    details.appendChild(list);
    timeline.appendChild(details);
  });

  return timeline;
}

function computeOverviewStats(){
  const stats = {
    categories: 0,
    artifacts : 0,
    latestDate: null
  };

  if (!Array.isArray(window.contributions)) return stats;

  stats.categories = window.contributions.length;
  window.contributions.forEach(section => {
    const items = Array.isArray(section.items) ? section.items : [];
    stats.artifacts += items.length;
    items.forEach(item => {
      const date = getItemDate(item);
      if (date && (!stats.latestDate || date.getTime() > stats.latestDate.getTime())) {
        stats.latestDate = date;
      }
    });
  });

  return stats;
}

function buildOverview(){
  const mount = document.getElementById('contrib-meta');
  if (!mount) return;

  const stats = computeOverviewStats();
  const artifactLabel = `public ${pluralize(stats.artifacts, 'artifact')}`;
  const updatedLabel = stats.latestDate ? formatMonthYear(stats.latestDate) : 'Current';

  const navLinks = Array.isArray(window.contributions)
    ? window.contributions
        .map(section => {
          const slug = section.slug || slugify(section.heading);
          return `<a href="#${slug}">${section.heading}</a>`;
        })
        .join('')
    : '';

  mount.innerHTML = `
    <div class="overview-stat-grid">
      <div class="overview-stat">
        <p class="overview-stat-value">${stats.categories}</p>
        <p class="overview-stat-label">publication categories</p>
      </div>
      <div class="overview-stat">
        <p class="overview-stat-value">${stats.artifacts}</p>
        <p class="overview-stat-label">${artifactLabel}</p>
      </div>
      <div class="overview-stat">
        <p class="overview-stat-value">${updatedLabel}</p>
        <p class="overview-stat-label">latest update</p>
      </div>
    </div>
    <div class="overview-anchor-row">
      <span class="overview-anchor-label">Jump to</span>
      <div class="overview-anchor-links">
        ${navLinks}
      </div>
    </div>
  `;
}

function buildContributions(){
  const root = document.getElementById('contrib-root');
  if(!root || !Array.isArray(window.contributions)) return;

  window.contributions.forEach((sec, index) => {
    const items = Array.isArray(sec.items) ? sec.items : [];
    if (!items.length) return;

    const sorted = sortItems(items);
    const [latest, ...previous] = sorted;

    const section = document.createElement('section');
    section.className = 'surface-band reveal contrib-section';
    section.dataset.heading = sec.heading;
    section.id = sec.slug || slugify(sec.heading);

    const wrap   = document.createElement('div');
    wrap.className = 'wrapper';
    section.appendChild(wrap);

    wrap.insertAdjacentHTML('beforeend',
      `<h2 class="section-title">${sec.heading}</h2>
       <p class="section-desc">${sec.desc}</p>`);

    const stack = document.createElement('div');
    stack.className = 'contrib-stack';

    const featured = buildFeaturedCard(latest);
    if (featured) stack.appendChild(featured);

    const glance = buildAtGlanceCard(sec);
    if (glance) stack.appendChild(glance);

    const timeline = buildTimeline(previous, {
      groupBy: sec.groupBy,
      timelineLabel: sec.timelineLabel
    });
    if (timeline) stack.appendChild(timeline);

    wrap.appendChild(stack);
    root.appendChild(section);

    if (index !== window.contributions.length - 1){
      const gap = document.createElement('section');
      gap.className = 'contrib-gap';
      root.appendChild(gap);
    }
  });
}

function initContributions(){
  buildOverview();
  buildContributions();
}

document.addEventListener('DOMContentLoaded', initContributions);
