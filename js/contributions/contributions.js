/* contributions.js - Build contributions UI components.
   Contribution data now lives in contributions-data.js */

function buildDocLinks(item, opts = {}){
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

function buildFeaturedCard(item){
  const card = document.createElement('article');
  card.className = 'doc-card featured-doc';
  card.innerHTML = `
    <div class="doc-layout">
      <p class="doc-label">Latest contribution</p>
      <h3 class="doc-title">${item.title}</h3>
      <div class="doc-footer">
        ${item.role ? `<p class="doc-role">${item.role}</p>` : ''}
        ${buildDocLinks(item)}
      </div>
    </div>`;
  return card;
}

function buildPreviousDropdown(previousItems, heading){
  if (!previousItems.length) return null;

  const details = document.createElement('details');
  details.className = 'contrib-dropdown';
  details.setAttribute('aria-label', `Previous contributions for ${heading}`);

  const summary = document.createElement('summary');
  summary.innerHTML = `Previous contributions <span>(${previousItems.length})</span>`;
  details.appendChild(summary);

  const list = document.createElement('ul');
  list.className = 'doc-list';

  previousItems.forEach(item => {
    const li = document.createElement('li');
    li.className = 'doc-row';
    li.innerHTML = `
      <div class="doc-row-text">
        <span class="doc-row-title">${item.title}</span>
        ${item.role ? `<span class="doc-row-role">${item.role}</span>` : ''}
      </div>
      ${buildDocLinks(item, { compact: true }) || ''}
    `;
    list.appendChild(li);
  });

  details.appendChild(list);
  return details;
}

function buildContributions(){
  const root = document.getElementById('contrib-root');
  if(!root || !Array.isArray(window.contributions)) return;

  window.contributions.forEach((sec, index) => {
    if (!sec.items || !sec.items.length) return;

    const [latest, ...previous] = sec.items;

    const section = document.createElement('section');
    section.className = 'surface-band reveal contrib-section';
    section.dataset.heading = sec.heading;

    const wrap   = document.createElement('div');
    wrap.className = 'wrapper';
    section.appendChild(wrap);

    wrap.insertAdjacentHTML('beforeend',
      `<h2 class="section-title">${sec.heading}</h2>
       <p class="section-desc">${sec.desc}</p>`);

    const stack = document.createElement('div');
    stack.className = 'contrib-stack';
    stack.appendChild(buildFeaturedCard(latest));

    const dropdown = buildPreviousDropdown(previous, sec.heading);
    if (dropdown) stack.appendChild(dropdown);

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
  buildContributions();
}

document.addEventListener('DOMContentLoaded', initContributions);
