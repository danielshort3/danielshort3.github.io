/* contributions.js - Build contributions UI components.
   Contribution data now lives in contributions-data.js */

const YEAR_FALLBACK = 'Additional highlights';
const numberFormatter = new Intl.NumberFormat('en-US');

const ICONS = {
  pdf: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <path d="M14 2v6h6"/>
          <path d="M8 15h8"/>
        </svg>`,
  link: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
           <path d="M14 3h7v7"/>
           <path d="M10 14L21 3"/>
           <path d="M21 14v7h-7"/>
           <path d="M3 10v11h11"/>
         </svg>`
};

const slugify = str => str?.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || '';

const extractYear = title => {
  const match = title?.match(/(20\d{2})/);
  return match ? match[1] : YEAR_FALLBACK;
};

function buildHeroMetrics(contributions){
  const host = document.getElementById('contrib-metrics');
  if(!host) return;
  host.innerHTML = '';

  const metrics = computeMetricData(contributions);
  metrics.forEach(metric => {
    const row = document.createElement('div');
    row.className = 'hero-metric';

    const valueEl = document.createElement('p');
    valueEl.className = 'metric-value';
    valueEl.textContent = metric.value;

    const labelEl = document.createElement('p');
    labelEl.className = 'metric-label';
    labelEl.textContent = metric.label;

    row.appendChild(valueEl);
    row.appendChild(labelEl);
    host.appendChild(row);
  });
}

function computeMetricData(contributions){
  if(!Array.isArray(contributions)) return [];

  const byHeading = Object.fromEntries(
    contributions.map(sec => [sec.heading, sec])
  );

  const councilCount    = byHeading['Council Briefings']?.items?.length || 0;
  const newsletterCount = byHeading['Visit Grand Junction eNewsletters']?.items?.length || 0;
  const publicReports   = contributions.reduce((sum, sec) => (
    sum + (sec.items?.filter(item => Boolean(item.pdf)).length || 0)
  ), 0);

  const years = new Set();
  contributions.forEach(sec => {
    sec.items?.forEach(item => {
      const year = extractYear(item.title);
      if(year !== YEAR_FALLBACK) years.add(year);
    });
  });

  const sortedYears = [...years].sort();
  const range = sortedYears.length
    ? (sortedYears[0] === sortedYears.at(-1)
        ? sortedYears[0]
        : `${sortedYears[0]}–${sortedYears.at(-1)}`)
    : '';

  const metrics = [
    { label: 'Council briefings delivered', value: numberFormatter.format(councilCount) },
    { label: 'Stakeholder newsletters',      value: numberFormatter.format(newsletterCount) },
    { label: 'Published reports',            value: numberFormatter.format(publicReports) }
  ];

  if(range) metrics.push({ label: 'Years covered', value: range });
  return metrics;
}

function createSpotlightCard(entry){
  const card = document.createElement('article');
  card.className = 'spotlight-card';

  if(entry.focus){
    const focus = document.createElement('p');
    focus.className = 'spotlight-focus';
    focus.textContent = entry.focus;
    card.appendChild(focus);
  }

  const title = document.createElement('h3');
  title.textContent = entry.title;
  card.appendChild(title);

  if(entry.context){
    const context = document.createElement('p');
    context.className = 'spotlight-context';
    context.textContent = entry.context;
    card.appendChild(context);
  }

  if(entry.contribution){
    const summary = document.createElement('p');
    summary.className = 'spotlight-summary';
    summary.textContent = entry.contribution;
    card.appendChild(summary);
  }

  if(entry.impact){
    const impact = document.createElement('p');
    impact.className = 'spotlight-impact';
    impact.innerHTML = `<strong>Outcome:</strong> ${entry.impact}`;
    card.appendChild(impact);
  }

  if(Array.isArray(entry.skills) && entry.skills.length){
    const list = document.createElement('ul');
    list.className = 'spotlight-tags';
    entry.skills.forEach(skill => {
      const li = document.createElement('li');
      li.textContent = skill;
      list.appendChild(li);
    });
    card.appendChild(list);
  }

  const actions = document.createElement('div');
  actions.className = 'spotlight-actions';

  if(entry.pdf){
    actions.appendChild(createSpotlightLink('Download PDF', entry.pdf, { download: true }));
  }
  if(entry.link){
    actions.appendChild(createSpotlightLink('View online', entry.link));
  }

  if(actions.childElementCount) card.appendChild(actions);
  return card;
}

function createSpotlightLink(label, url, opts = {}){
  const link = document.createElement('a');
  link.href = url;
  link.className = 'btn-secondary spotlight-btn';
  link.textContent = label;
  if(opts.download){
    link.setAttribute('download', '');
  } else {
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  }
  return link;
}

function buildSpotlightSection(entries){
  if(!Array.isArray(entries) || !entries.length) return null;
  const section = document.createElement('section');
  section.className = 'surface-band reveal spotlight-section';

  const wrap = document.createElement('div');
  wrap.className = 'wrapper';
  section.appendChild(wrap);

  wrap.insertAdjacentHTML('beforeend',
    `<p class="eyebrow">Featured case studies</p>
     <h2 class="section-title">Executive-ready highlights</h2>
     <p class="section-desc">A quick sample of the decision support work I deliver for statewide partners and city leadership.</p>`
  );

  const grid = document.createElement('div');
  grid.className = 'spotlight-grid';
  entries.forEach(entry => grid.appendChild(createSpotlightCard(entry)));
  wrap.appendChild(grid);
  return section;
}

function buildFilterSection(contributions){
  if(!Array.isArray(contributions) || !contributions.length) return null;

  const section = document.createElement('section');
  section.className = 'surface-band reveal contrib-filter-band';

  const wrap = document.createElement('div');
  wrap.className = 'wrapper';
  section.appendChild(wrap);

  const bar = document.createElement('div');
  bar.className = 'contrib-filter-bar';
  wrap.appendChild(bar);

  const label = document.createElement('label');
  label.setAttribute('for', 'contrib-focus-filter');
  label.textContent = 'Focus area';
  bar.appendChild(label);

  const select = document.createElement('select');
  select.id = 'contrib-focus-filter';
  select.className = 'contrib-select';
  select.innerHTML = '<option value="all">All focus areas</option>';

  contributions.forEach(sec => {
    const option = document.createElement('option');
    option.value = slugify(sec.heading);
    option.textContent = sec.heading;
    select.appendChild(option);
  });

  bar.appendChild(select);

  const status = document.createElement('p');
  status.id = 'contrib-filter-status';
  status.className = 'filter-status';
  status.setAttribute('aria-live', 'polite');
  status.textContent = 'Showing all focus areas';
  wrap.appendChild(status);

  select.addEventListener('change', e => {
    const choice = e.target;
    const statusEl = document.getElementById('contrib-filter-status');
    const selectedText = choice.selectedOptions[0]?.textContent || 'All focus areas';
    applyFocusFilter(choice.value, selectedText, statusEl);
  });

  return section;
}

function applyFocusFilter(value, label, statusEl){
  const sections = document.querySelectorAll('[data-contrib-category]');
  let visibleCount = 0;

  sections.forEach(section => {
    const match = value === 'all' || section.dataset.contribCategory === value;
    section.hidden = !match;
    if(match) visibleCount += 1;
  });

  if(statusEl){
    statusEl.textContent = value === 'all'
      ? 'Showing all focus areas'
      : `Showing ${visibleCount} section${visibleCount === 1 ? '' : 's'} for “${label}”.`;
  }
}

function createDocIconLink(type, url){
  if(!url) return null;
  const link = document.createElement('a');
  link.href = url;
  link.className = 'doc-link';
  link.innerHTML = ICONS[type];
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  if(type === 'pdf'){
    link.setAttribute('download', '');
  }
  link.setAttribute('aria-label', type === 'pdf' ? 'Open PDF' : 'Open external link');
  return link;
}

function createDocCard(item){
  const card = document.createElement('article');
  card.className = 'doc-card';

  const layout = document.createElement('div');
  layout.className = 'doc-layout';

  const title = document.createElement('h3');
  title.className = 'doc-title';
  title.textContent = item.title;
  layout.appendChild(title);

  const footer = document.createElement('div');
  footer.className = 'doc-footer';

  if(item.role){
    const role = document.createElement('p');
    role.className = 'doc-role';
    role.textContent = item.role;
    footer.appendChild(role);
  } else {
    const placeholder = document.createElement('p');
    placeholder.className = 'doc-role';
    placeholder.textContent = '\u00A0';
    placeholder.setAttribute('aria-hidden', 'true');
    footer.appendChild(placeholder);
  }

  const links = document.createElement('div');
  links.className = 'doc-links';
  const pdfLink = createDocIconLink('pdf', item.pdf);
  const extLink = createDocIconLink('link', item.link);
  if(pdfLink) links.appendChild(pdfLink);
  if(extLink) links.appendChild(extLink);
  footer.appendChild(links);

  layout.appendChild(footer);
  card.appendChild(layout);
  return card;
}

function buildYearAccordions(items){
  const fragment = document.createDocumentFragment();
  const byYear = items.reduce((map, item) => {
    const year = extractYear(item.title);
    if(!map[year]) map[year] = [];
    map[year].push(item);
    return map;
  }, {});

  const sorted = Object.entries(byYear).sort((a, b) => {
    const yearA = /^\d{4}$/.test(a[0]) ? parseInt(a[0], 10) : null;
    const yearB = /^\d{4}$/.test(b[0]) ? parseInt(b[0], 10) : null;
    if(yearA === null && yearB === null) return a[0].localeCompare(b[0]);
    if(yearA === null) return 1;
    if(yearB === null) return -1;
    return yearB - yearA;
  });

  sorted.forEach(([year, docs], idx) => {
    const details = document.createElement('details');
    details.className = 'contrib-accordion';
    details.open = idx === 0;

    const summary = document.createElement('summary');
    summary.innerHTML = `<span>${year}</span><span class="badge">${docs.length}</span>`;
    details.appendChild(summary);

    const grid = document.createElement('div');
    grid.className = 'docs-grid';
    docs.forEach(doc => grid.appendChild(createDocCard(doc)));
    details.appendChild(grid);

    fragment.appendChild(details);
  });

  return fragment;
}

function buildSection(sec){
  if(!sec || !Array.isArray(sec.items) || !sec.items.length) return null;
  const section = document.createElement('section');
  section.className = 'surface-band reveal contrib-section';
  section.dataset.contribCategory = slugify(sec.heading);
  section.dataset.contribLabel = sec.heading;

  const wrap = document.createElement('div');
  wrap.className = 'wrapper';
  section.appendChild(wrap);

  wrap.insertAdjacentHTML('beforeend',
    `<div class="section-header">
       <div>
         <p class="eyebrow">Focus area</p>
         <h2 class="section-title">${sec.heading}</h2>
         <p class="section-desc">${sec.desc}</p>
       </div>
       <p class="section-count">${sec.items.length} docs</p>
     </div>`
  );

  wrap.appendChild(buildYearAccordions(sec.items));
  return section;
}

function buildContributions(){
  const root = document.getElementById('contrib-root');
  if(!root || !window.contributions) return;

  const contributions = window.contributions;
  buildHeroMetrics(contributions);

  root.innerHTML = '';

  const spotlightSection = buildSpotlightSection(window.contributionSpotlight || []);
  if(spotlightSection) root.appendChild(spotlightSection);

  const filterSection = buildFilterSection(contributions);
  if(filterSection) root.appendChild(filterSection);

  contributions.forEach(sec => {
    const sectionEl = buildSection(sec);
    if(sectionEl) root.appendChild(sectionEl);
  });

  const statusEl = document.getElementById('contrib-filter-status');
  applyFocusFilter('all', 'All focus areas', statusEl);
}

document.addEventListener('DOMContentLoaded', buildContributions);
