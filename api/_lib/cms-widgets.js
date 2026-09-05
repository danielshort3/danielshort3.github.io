'use strict';

const {
  renderPersonalLibraryHeader,
  renderToolsAccountDock
} = require('../../build/lib/personal-accordion-shell');

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function attrsToString(attrs) {
  return Object.entries(attrs || {})
    .filter(([, value]) => value !== false && value != null && value !== '')
    .map(([key, value]) => value === true ? key : `${key}="${escapeHtml(value)}"`)
    .join(' ');
}

function normalizeHref(value, fallback = '#') {
  const raw = String(value || '').trim();
  return raw || fallback;
}

function normalizeMapAddress(value) {
  const raw = String(value || '').trim();
  return raw || 'Grand Junction, CO';
}

function normalizeMapZoom(value) {
  const zoom = Number(value);
  if (!Number.isFinite(zoom)) return 10;
  return Math.max(0, Math.min(21, Math.round(zoom)));
}

function isTruthy(value) {
  if (value === true) return true;
  return /^(1|true|yes)$/i.test(String(value || '').trim());
}

function googleMapsSearchUrl(address) {
  return `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(address)}`;
}

function googleMapsFallbackEmbedUrl(address) {
  return `https://www.google.com/maps?q=${encodeURIComponent(address)}&output=embed`;
}

function sectionAttrs(section, className) {
  const attrs = {
    class: className,
    'data-cms-section-id': section.id || '',
    'data-cms-section-type': section.type || ''
  };
  const rendered = attrsToString(attrs);
  return rendered ? ` ${rendered}` : '';
}

function paragraphLines(value) {
  return String(value || '')
    .split(/\n{2,}/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `<p>${escapeHtml(line)}</p>`)
    .join('\n');
}

function renderHero(section) {
  const props = section.props || {};
  const primaryLabel = String(props.primaryLabel || '').trim();
  const secondaryLabel = String(props.secondaryLabel || '').trim();
  const actions = [
    primaryLabel ? `<a href="${escapeHtml(normalizeHref(props.primaryHref))}" class="btn-primary hero-cta">${escapeHtml(primaryLabel)}</a>` : '',
    secondaryLabel ? `<a href="${escapeHtml(normalizeHref(props.secondaryHref))}" class="btn-secondary hero-cta">${escapeHtml(secondaryLabel)}</a>` : ''
  ].filter(Boolean).join('\n        ');

  return [
    `<section${sectionAttrs(section, `hero hero--default${props.altBand ? ' alt-band' : ''}`)}>`,
    '  <div class="wrapper">',
    props.eyebrow ? `    <p class="hero-eyebrow">${escapeHtml(props.eyebrow)}</p>` : '',
    `    <h1>${escapeHtml(props.title || 'New Page')}</h1>`,
    props.lead ? `    <p class="hero-tagline">${escapeHtml(props.lead)}</p>` : '',
    actions ? `    <div class="cta-group">\n        ${actions}\n    </div>` : '',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderRichText(section) {
  const props = section.props || {};
  const body = paragraphLines(props.body || 'Add body copy.');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.kicker ? `    <p class="section-kicker">${escapeHtml(props.kicker)}</p>` : '',
    props.title ? `    <h2 class="section-title">${escapeHtml(props.title)}</h2>` : '',
    `    <div class="cms-rich-text">\n${body.split('\n').map((line) => `      ${line}`).join('\n')}\n    </div>`,
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderCta(section) {
  const props = section.props || {};
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    '    <div id="cta-link" role="group" aria-label="Contact call to action">',
    `      <h2 class="section-title">${escapeHtml(props.title || 'Call to Action')}</h2>`,
    props.body ? `      <p>${escapeHtml(props.body)}</p>` : '',
    props.label ? `      <div><a href="${escapeHtml(normalizeHref(props.href))}" class="btn-primary">${escapeHtml(props.label)}</a></div>` : '',
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderImageGallery(section) {
  const props = section.props || {};
  const images = Array.isArray(props.images) && props.images.length
    ? props.images
    : [{ src: 'img/hero/head.png', alt: 'Gallery image', caption: 'Gallery image' }];
  const cards = images.map((image) => [
    '      <figure class="project-card">',
    `        <img src="${escapeHtml(image.src || '')}" alt="${escapeHtml(image.alt || '')}" loading="lazy" decoding="async">`,
    image.caption ? `        <figcaption class="project-text"><span class="project-title">${escapeHtml(image.caption)}</span></figcaption>` : '',
    '      </figure>'
  ].filter(Boolean).join('\n')).join('\n');

  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.title ? `    <h2 class="section-title">${escapeHtml(props.title)}</h2>` : '',
    '    <div class="project-examples-grid" role="list">',
    cards,
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderDocumentList(section) {
  const props = section.props || {};
  const documents = Array.isArray(props.documents) && props.documents.length
    ? props.documents
    : [{ label: 'Document', href: 'https://danielshort-public-documents-886623862678-us-east-2.s3.us-east-2.amazonaws.com/documents/Resume.pdf' }];
  const links = documents.map((doc) => {
    return `      <li><a href="${escapeHtml(normalizeHref(doc.href))}">${escapeHtml(doc.label || doc.href || 'Document')}</a></li>`;
  }).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    `    <h2 class="section-title">${escapeHtml(props.title || 'Documents')}</h2>`,
    '    <ul class="cms-document-links">',
    links,
    '    </ul>',
    '  </div>',
    '</section>'
  ].join('\n');
}

function renderMap(section) {
  const props = section.props || {};
  const address = normalizeMapAddress(props.address);
  const zoom = normalizeMapZoom(props.zoom);
  const shouldEmbed = isTruthy(props.embed);
  const mapHref = googleMapsSearchUrl(address);
  const iframeTitle = String(props.iframeTitle || `Map of ${address}`).trim();
  const anchorId = String(props.anchorId || section.id || '').trim();
  const frameAttrs = {
    class: 'cms-map-iframe',
    title: iframeTitle,
    src: googleMapsFallbackEmbedUrl(address),
    loading: 'lazy',
    allowfullscreen: true,
    referrerpolicy: 'strict-origin-when-cross-origin',
    ...(shouldEmbed ? {
      'data-google-maps-iframe': true,
      'data-google-maps-address': address,
      'data-google-maps-zoom': zoom
    } : {})
  };
  return [
    `<section${anchorId ? ` id="${escapeHtml(anchorId)}"` : ''}${sectionAttrs(section, 'surface-band reveal cms-location')}>`,
    '  <div class="wrapper cms-location-inner">',
    '    <div class="cms-location-copy">',
    `      <h2 class="section-title">${escapeHtml(props.title || 'Location')}</h2>`,
    props.body ? `      <p class="section-subtitle">${escapeHtml(props.body)}</p>` : '',
    `      <p><a class="btn-secondary" href="${escapeHtml(mapHref)}" target="_blank" rel="noopener noreferrer">${escapeHtml(props.buttonLabel || 'Open map')}</a></p>`,
    '    </div>',
    '    <div class="cms-map-shell">',
    `      <iframe${attrsToString(frameAttrs) ? ` ${attrsToString(frameAttrs)}` : ''}></iframe>`,
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderEmbed(section) {
  const props = section.props || {};
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.title ? `    <h2 class="section-title">${escapeHtml(props.title)}</h2>` : '',
    '    <div class="video-shell">',
    `      <iframe src="${escapeHtml(normalizeHref(props.src, 'about:blank'))}" title="${escapeHtml(props.title || 'Embedded content')}" loading="lazy"></iframe>`,
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderKpiBand(section) {
  const props = section.props || {};
  const items = Array.isArray(props.items) && props.items.length
    ? props.items
    : [{ value: '99%', label: 'Faster reporting' }, { value: '200+', label: 'Hours saved' }, { value: '$13.1M', label: 'Measured impact' }];
  const cards = items.map((item) => [
    '      <div class="resume-highlight">',
    `        <div class="resume-highlight-value">${escapeHtml(item.value || '')}</div>`,
    `        <div class="resume-highlight-label">${escapeHtml(item.label || '')}</div>`,
    '      </div>'
  ].join('\n')).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.kicker ? `    <p class="section-kicker">${escapeHtml(props.kicker)}</p>` : '',
    props.title ? `    <h2 class="section-title">${escapeHtml(props.title)}</h2>` : '',
    '    <div class="resume-highlights" aria-label="Key metrics">',
    cards,
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderProofBlock(section) {
  const props = section.props || {};
  const bullets = Array.isArray(props.bullets) && props.bullets.length
    ? props.bullets
    : ['Describe the evidence, result, or decision this supports.'];
  const items = bullets.map((item) => `      <li>${escapeHtml(item)}</li>`).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.kicker ? `    <p class="section-kicker">${escapeHtml(props.kicker)}</p>` : '',
    `    <h2 class="section-title">${escapeHtml(props.title || 'Proof point')}</h2>`,
    props.lead ? `    <p class="section-lead">${escapeHtml(props.lead)}</p>` : '',
    '    <ul class="cms-proof-list">',
    items,
    '    </ul>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderProjectGrid(section) {
  const props = section.props || {};
  const projects = Array.isArray(props.projects) && props.projects.length
    ? props.projects
    : [{ title: 'Project title', href: 'portfolio', summary: 'Add the project outcome or audience fit.' }];
  const cards = projects.map((project) => [
    `      <a class="project-card" role="listitem" href="${escapeHtml(normalizeHref(project.href, 'portfolio'))}">`,
    '        <span class="project-text">',
    `          <span class="project-title">${escapeHtml(project.title || 'Project title')}</span>`,
    project.summary ? `          <span>${escapeHtml(project.summary)}</span>` : '',
    '        </span>',
    '      </a>'
  ].filter(Boolean).join('\n')).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.kicker ? `    <p class="section-kicker">${escapeHtml(props.kicker)}</p>` : '',
    `    <h2 class="section-title">${escapeHtml(props.title || 'Selected projects')}</h2>`,
    '    <div class="project-examples-grid" role="list">',
    cards,
    '    </div>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderCertificationStrip(section) {
  const props = section.props || {};
  const certifications = Array.isArray(props.certifications) && props.certifications.length
    ? props.certifications
    : [{ title: 'Certification', issuer: 'Issuer', icon: 'img/cert_logos/google-48.png', href: '#' }];
  const items = certifications.map((certification) => [
    '      <li class="resume-cert">',
    `        <a href="${escapeHtml(normalizeHref(certification.href))}">`,
    certification.icon ? `          <img src="${escapeHtml(certification.icon)}" width="24" height="24" loading="lazy" decoding="async" alt="">` : '',
    `          <span class="resume-cert-title">${escapeHtml(certification.title || 'Certification')}</span>`,
    certification.issuer ? `          <span class="resume-cert-meta">${escapeHtml(certification.issuer)}</span>` : '',
    '        </a>',
    '      </li>'
  ].filter(Boolean).join('\n')).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    `    <h2 class="section-title">${escapeHtml(props.title || 'Certifications')}</h2>`,
    '    <ul class="resume-cert-grid">',
    items,
    '    </ul>',
    '  </div>',
    '</section>'
  ].join('\n');
}

function renderResumeHighlight(section) {
  const props = section.props || {};
  const bullets = Array.isArray(props.bullets) && props.bullets.length
    ? props.bullets
    : ['Add an accomplishment with a metric, audience, and business result.'];
  const items = bullets.map((item) => `      <li>${escapeHtml(item)}</li>`).join('\n');
  return [
    `<section${sectionAttrs(section, 'surface-band resume-section')}>`,
    '  <div class="wrapper">',
    '    <article class="resume-block">',
    `      <h2 class="resume-block-title">${escapeHtml(props.title || 'Resume highlight')}</h2>`,
    props.meta ? `      <p class="resume-education-meta">${escapeHtml(props.meta)}</p>` : '',
    '      <ul class="resume-role-list">',
    items,
    '      </ul>',
    '    </article>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function renderMediaShowcase(section) {
  const props = section.props || {};
  return [
    `<section${sectionAttrs(section, 'surface-band reveal')}>`,
    '  <div class="wrapper">',
    props.kicker ? `    <p class="section-kicker">${escapeHtml(props.kicker)}</p>` : '',
    `    <h2 class="section-title">${escapeHtml(props.title || 'Media showcase')}</h2>`,
    props.lead ? `    <p class="section-lead">${escapeHtml(props.lead)}</p>` : '',
    '    <figure class="project-card">',
    `      <img src="${escapeHtml(props.src || 'img/hero/head.png')}" alt="${escapeHtml(props.alt || '')}" loading="lazy" decoding="async">`,
    props.caption ? `      <figcaption class="project-text"><span>${escapeHtml(props.caption)}</span></figcaption>` : '',
    '    </figure>',
    '  </div>',
    '</section>'
  ].filter(Boolean).join('\n');
}

function workDateRank(value) {
  const text = String(value || '').trim();
  if (/present/i.test(text)) return Number.MAX_SAFE_INTEGER;
  const matches = [...text.matchAll(/\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b/gi)];
  const last = matches.length ? matches[matches.length - 1][0] : '';
  const parsed = last ? Date.parse(`1 ${last}`) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : 0;
}

function sortLegacyWorkCards(html) {
  const source = String(html || '');
  if (!source.includes('id="work-experience"')) return source;

  const cardPattern = /<article\b[^>]*class="[^"]*\bwork-card\b[^"]*"[^>]*>[\s\S]*?<\/article>/gi;
  const cards = [...source.matchAll(cardPattern)].map((match, index) => {
    const cardHtml = match[0];
    const timeframe = /class="[^"]*\bwork-timeframe\b[^"]*"[^>]*>([\s\S]*?)<\//i.exec(cardHtml);
    return {
      html: cardHtml,
      index,
      rank: workDateRank(timeframe ? timeframe[1].replace(/<[^>]+>/g, ' ') : '')
    };
  });
  if (cards.length < 2) return source;

  const sorted = cards
    .slice()
    .sort((a, b) => (a.rank - b.rank) || (a.index - b.index));
  let replacementIndex = 0;
  return source.replace(cardPattern, () => sorted[replacementIndex++].html);
}

function renderLegacyHtml(section) {
  const html = String(section && section.props && section.props.html ? section.props.html : '');
  return sortLegacyWorkCards(html).replace(
    /(<a\b[^>]*class=")btn-secondary("[^>]*\bdownload>Download PDF<\/a>)/gi,
    '$1btn-primary$2'
  );
}

const HOME_ACCORDION_ICONS = {
  about: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="7" r="4"></circle><path d="M4.5 21c.7-4.1 3.2-6.2 7.5-6.2s6.8 2.1 7.5 6.2"></path></svg>',
  projects: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5h7l2-2h9v14H3z"></path><path d="M3 9h18"></path></svg>',
  tools: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.1a5 5 0 0 0-6.8 6.8L3 17.8 6.2 21l4.9-4.9a5 5 0 0 0 6.8-6.8l-3.1 3.1-3.2-3.2z"></path></svg>',
  games: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.5 8h9a5 5 0 0 1 4.7 3.3l1.2 3.6a3.2 3.2 0 0 1-5.3 3.3L15 16H9l-2.1 2.2a3.2 3.2 0 0 1-5.3-3.3l1.2-3.6A5 5 0 0 1 7.5 8z"></path><path d="M7 11v4M5 13h4M16.5 12h.01M19 14h.01"></path></svg>',
  contact: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"></path><path d="M8 9h8M8 13h5"></path></svg>',
  playground: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="7" height="7" rx="1.5"></rect><rect x="14" y="3" width="7" height="7" rx="1.5"></rect><rect x="3" y="14" width="7" height="7" rx="1.5"></rect><path d="M17.5 14v7M14 17.5h7"></path></svg>',
  stormbreak: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13.5 2 5 13h6l-1 9 9-13h-6z"></path><path d="M4 5h5M15 19h5"></path></svg>',
  'stellar-dogfight': '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3z"></path><circle cx="12" cy="12" r="2.2"></circle><path d="M3 5h3M18 19h3"></path></svg>',
  probability: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"></rect><path d="M9 5v14M15 5v14"></path><circle cx="6" cy="12" r="1.4"></circle><circle cx="12" cy="12" r="1.4"></circle><circle cx="18" cy="12" r="1.4"></circle></svg>',
  roulette: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><circle cx="12" cy="12" r="2"></circle><circle cx="18.5" cy="7" r="1"></circle><path d="M12 3v7M12 14v7M3 12h7M14 12h7M5.6 5.6l4.9 4.9M13.5 13.5l4.9 4.9M18.4 5.6l-4.9 4.9M10.5 13.5l-4.9 4.9"></path></svg>',
  wave: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 8c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3M2 16c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3"></path></svg>',
  message: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m22 2-7 20-4-9-9-4z"></path><path d="M22 2 11 13"></path></svg>',
  email: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="2.5" y="5" width="19" height="14" rx="2"></rect><path d="m3.5 7 8.5 6 8.5-6"></path></svg>',
  github: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="5" r="2"></circle><circle cx="18" cy="5" r="2"></circle><circle cx="12" cy="19" r="2"></circle><path d="M6 7v3c0 2 1.6 3 3.5 3H12M18 7v3c0 2-1.6 3-3.5 3H12M12 13v4"></path></svg>',
  spark: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 1.2 3.8L17 8l-3.8 1.2L12 13l-1.2-3.8L7 8l3.8-1.2zM5 15v4M3 17h4M19 14v3M17.5 15.5h3"></path></svg>',
  timeline: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 4v16M5 7h7M5 12h11M5 17h8"></path><circle cx="5" cy="7" r="1.5"></circle><circle cx="5" cy="12" r="1.5"></circle><circle cx="5" cy="17" r="1.5"></circle></svg>',
  arrow: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"></path></svg>',
  'external-arrow': '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3h7v7M10 14 21 3M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"></path></svg>'
};

function resolveHomeAccordionIconId(id) {
  const key = String(id || '').trim();
  return Object.prototype.hasOwnProperty.call(HOME_ACCORDION_ICONS, key) ? key : 'spark';
}

function homeAccordionIcon(id) {
  return HOME_ACCORDION_ICONS[resolveHomeAccordionIconId(id)];
}

function getHomeAccordionIconDefinitions() {
  return { ...HOME_ACCORDION_ICONS };
}

function renderHomeAccordionCard(item, categoryId) {
  const href = String(item && item.href || '').trim();
  const normalizedHref = href ? normalizeHref(href) : '';
  const tag = href ? 'a' : 'article';
  const iconId = resolveHomeAccordionIconId(item && item.icon || categoryId);
  const presentation = ['featured', 'tertiary'].includes(String(item && item.presentation || '').trim())
    ? String(item.presentation).trim()
    : '';
  const mediaType = item && item.image ? 'image' : 'glyph';
  const media = item && item.image
    ? `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.imageAlt || '')}" loading="lazy" decoding="async"${item.imageWidth ? ` width="${escapeHtml(item.imageWidth)}"` : ''}${item.imageHeight ? ` height="${escapeHtml(item.imageHeight)}"` : ''}>`
    : `<span class="home-accordion__card-glyph" data-home-icon="${escapeHtml(iconId)}" aria-hidden="true">${homeAccordionIcon(iconId)}</span>`;
  const contentType = String(item && item.contentType || '').trim();
  const contentId = String(item && item.contentId || item && item.id || '').trim();
  const resourceType = String(item && item.resourceType || contentType || '').trim();
  const analytics = href && contentType && contentId
    ? ` data-content-open="true" data-content-id="${escapeHtml(contentId)}" data-content-type="${escapeHtml(contentType)}" data-resource-type="${escapeHtml(resourceType)}" data-source-surface="home_category_accordion"`
    : '';
  const external = Boolean(item && item.external);
  const contactModal = normalizedHref === '/contact#contact-modal'
    ? ' data-contact-modal-link'
    : '';
  const linkAttrs = href
    ? ` href="${escapeHtml(normalizedHref)}"${external ? ' target="_blank" rel="noopener noreferrer"' : ''}${analytics}${contactModal}`
    : '';

  return [
    '            <li class="home-accordion__card-item">',
    `              <${tag} class="home-accordion__card${presentation ? ` home-accordion__card--${presentation}` : ''}"${linkAttrs}>`,
    `                <span class="home-accordion__card-media home-accordion__card-media--${mediaType}">${media}</span>`,
    '                <span class="home-accordion__card-copy">',
    item && item.badge ? `                  <span class="home-accordion__card-badge">${escapeHtml(item.badge)}</span>` : '',
    `                  <strong>${escapeHtml(item && item.title || 'Explore')}</strong>`,
    item && item.summary ? `                  <span>${escapeHtml(item.summary)}</span>` : '',
    '                </span>',
    href ? `                <span class="home-accordion__card-arrow" data-home-icon="${external ? 'external-arrow' : 'arrow'}" aria-hidden="true">${HOME_ACCORDION_ICONS[external ? 'external-arrow' : 'arrow']}</span>` : '',
    `              </${tag}>`,
    '            </li>'
  ].filter(Boolean).join('\n');
}

const HOME_TIMELINE_MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
];

const HOME_TIMELINE_TYPE_LABELS = {
  certification: 'Certification',
  degree: 'Degree',
  job: 'Work',
  personal: 'Personal',
  project: 'Project'
};

function parseHomeTimelineDate(value) {
  const text = String(value || '').trim();
  const match = /^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$/.exec(text);
  if (!match) return null;

  const year = Number(match[1]);
  const month = match[2] ? Number(match[2]) : null;
  const day = match[3] ? Number(match[3]) : null;
  if (month !== null && (month < 1 || month > 12)) return null;
  if (day !== null) {
    const lastDay = new Date(Date.UTC(year, month, 0)).getUTCDate();
    if (day < 1 || day > lastDay) return null;
  }

  const label = day !== null
    ? `${HOME_TIMELINE_MONTHS[month - 1]} ${day}, ${year}`
    : (month !== null ? `${HOME_TIMELINE_MONTHS[month - 1]} ${year}` : String(year));
  return { label, value: text };
}

function renderHomeTimelineDate(item) {
  if (item && item.current) {
    return '              <span class="home-timeline__current">Now</span>';
  }

  const start = parseHomeTimelineDate(item && item.date);
  if (!start) return '';
  const end = parseHomeTimelineDate(item && item.endDate);
  const startHtml = `<time datetime="${escapeHtml(start.value)}">${escapeHtml(start.label)}</time>`;
  const endHtml = end
    ? `<time datetime="${escapeHtml(end.value)}">${escapeHtml(end.label)}</time>`
    : (item && item.ongoing ? '<span class="home-timeline__current">Present</span>' : '');
  return `              ${startHtml}${endHtml ? ` <span class="home-timeline__date-separator">–</span> ${endHtml}` : ''}`;
}

function renderHomeTimelineItem(item, categoryId) {
  const authoredType = String(item && item.type || '').trim();
  const type = Object.prototype.hasOwnProperty.call(HOME_TIMELINE_TYPE_LABELS, authoredType)
    ? authoredType
    : 'personal';
  const href = String(item && item.href || '').trim();
  const tag = href ? 'a' : 'article';
  const external = Boolean(item && item.external);
  const contentType = String(item && item.contentType || '').trim();
  const contentId = String(item && item.contentId || item && item.id || '').trim();
  const resourceType = String(item && item.resourceType || contentType || '').trim();
  const mediaTone = String(item && item.imageTone || '').trim() === 'dark' ? 'dark' : '';
  const title = String(item && item.title || 'Milestone');
  const compactTitle = type === 'certification'
    ? title.replace(/\s+(?:Professional\s+Certificate|Certification|Certificate)$/i, '')
    : title;
  const titleHtml = compactTitle !== title
    ? `<span class="home-timeline__title-full">${escapeHtml(title)}</span><span class="home-timeline__title-compact" aria-hidden="true">${escapeHtml(compactTitle)}</span>`
    : escapeHtml(title);
  const dateId = `home-timeline-${categoryId}-${String(item && item.id || '').replace(/[^a-z0-9_-]+/gi, '-')}-date`;
  const analytics = href && contentType && contentId
    ? ` data-content-open="true" data-content-id="${escapeHtml(contentId)}" data-content-type="${escapeHtml(contentType)}" data-resource-type="${escapeHtml(resourceType)}" data-source-surface="home_timeline"`
    : '';
  const linkAttrs = href
    ? ` href="${escapeHtml(normalizeHref(href))}"${external ? ' target="_blank" rel="noopener noreferrer"' : ''}${analytics}`
    : '';
  const media = item && item.image
    ? `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.imageAlt || '')}" loading="lazy" decoding="async"${item.imageWidth ? ` width="${escapeHtml(item.imageWidth)}"` : ''}${item.imageHeight ? ` height="${escapeHtml(item.imageHeight)}"` : ''}>`
    : '<span class="home-timeline__marker" aria-hidden="true"></span>';

  return [
    `          <li class="home-timeline__item home-timeline__item--${escapeHtml(type)}" data-home-timeline-item="${escapeHtml(item && item.id || '')}">`,
    `            <div class="home-timeline__date" id="${escapeHtml(dateId)}">`,
    renderHomeTimelineDate(item),
    '            </div>',
    '            <span class="home-timeline__axis" aria-hidden="true"><span class="home-timeline__dot"></span></span>',
    `            <${tag} class="home-timeline__entry" aria-describedby="${escapeHtml(dateId)}"${linkAttrs}>`,
    `              <span class="home-timeline__media"${mediaTone ? ` data-home-timeline-media-tone="${mediaTone}"` : ''}>${media}</span>`,
    '              <span class="home-timeline__copy">',
    `                <span class="home-timeline__type">${HOME_TIMELINE_TYPE_LABELS[type]}</span>`,
    `                <strong class="home-timeline__title">${titleHtml}</strong>`,
    item && item.subtitle ? `                <span class="home-timeline__subtitle">${escapeHtml(item.subtitle)}</span>` : '',
    item && item.summary ? `                <span class="home-timeline__summary">${escapeHtml(item.summary)}</span>` : '',
    '              </span>',
    href ? `              <span class="home-timeline__arrow" data-home-icon="${external ? 'external-arrow' : 'arrow'}" aria-hidden="true">${HOME_ACCORDION_ICONS[external ? 'external-arrow' : 'arrow']}</span>` : '',
    `            </${tag}>`,
    '          </li>'
  ].filter(Boolean).join('\n');
}

function renderHomeTimeline(timeline, categoryId) {
  const items = Array.isArray(timeline && timeline.items) ? timeline.items : [];
  if (!items.length) return '';

  const safeCategoryId = String(categoryId || 'about').trim().replace(/[^a-z0-9_-]+/gi, '-');
  return [
    '          <section class="home-timeline" data-home-timeline aria-label="Timeline">',
    '            <ol class="home-timeline__list" data-home-timeline-scroller>',
    items.map((item) => renderHomeTimelineItem(item, safeCategoryId)).join('\n'),
    '            </ol>',
    '          </section>'
  ].filter(Boolean).join('\n');
}

function renderHomeLibraryView(category, categoryId) {
  const cta = category && category.cta;
  if (!cta || !cta.href || !['projects', 'tools', 'games'].includes(categoryId)) return '';

  const viewId = `home-library-view-${categoryId}`;
  const headingId = `${viewId}-title`;
  const header = renderPersonalLibraryHeader({
    category: categoryId,
    itemCount: 0,
    containerTag: 'header',
    headingTag: 'h3',
    headingId,
    headingFocusable: true,
    includeBack: true,
    dynamicCount: true
  });
  const account = categoryId === 'tools'
    ? renderToolsAccountDock('tools-account-dock--directory personal-library__account')
    : '';
  return [
    `          <section class="home-library" id="${escapeHtml(viewId)}" data-home-library-view="${escapeHtml(categoryId)}" aria-labelledby="${escapeHtml(headingId)}" hidden inert>`,
    header.split('\n').map((line) => `            ${line}`).join('\n'),
    account ? account.split('\n').map((line) => `            ${line}`).join('\n') : '',
    `            <ul class="home-library__list" data-home-library-list aria-label="All ${escapeHtml(String(category.label || categoryId).toLowerCase())}"></ul>`,
    '          </section>'
  ].filter(Boolean).join('\n');
}

function renderHomeAccordion(section) {
  const props = section.props || {};
  const categories = Array.isArray(props.categories) ? props.categories : [];
  const allowedIds = new Set(categories.map((category) => String(category && category.id || '').trim()).filter(Boolean));
  const defaultPanel = allowedIds.has(String(props.defaultPanel || '').trim())
    ? String(props.defaultPanel).trim()
    : (categories[0] && categories[0].id || 'about');

  const panels = categories.map((category) => {
    const id = String(category && category.id || '').trim();
    const label = String(category && category.label || id || 'Section').trim();
    const isActive = id === defaultPanel;
    const items = Array.isArray(category && category.items) ? category.items : [];
    const meta = Array.isArray(category && category.meta) ? category.meta : [];
    const context = String(category && category.context || '').trim();
    const triggerId = `home-accordion-trigger-${id}`;
    const panelId = `home-accordion-panel-${id}`;
    const categoryIconId = resolveHomeAccordionIconId(id);
    const color = String(category && category.color || '#091f3b').trim();
    const colorEnd = String(category && category.colorEnd || color).trim();
    const hasInlineLibrary = ['projects', 'tools', 'games'].includes(id);
    const cards = items.map((item) => renderHomeAccordionCard(item, id)).join('\n');
    const timelineHtml = renderHomeTimeline(category && category.timeline, id);
    const libraryHtml = renderHomeLibraryView(category, id);
    const profile = category && category.profile && category.profile.image
      ? category.profile
      : null;
    const metaHtml = meta.length
      ? `          <ul class="home-accordion__meta" aria-label="${escapeHtml(label)} highlights">${meta.map((entry) => `<li>${escapeHtml(entry)}</li>`).join('')}</ul>`
      : '';
    const contextHtml = context && !profile
      ? `          <p class="home-accordion__context">${escapeHtml(context)}</p>`
      : '';
    const cta = category && category.cta && category.cta.href && category.cta.label
      ? (hasInlineLibrary
          ? `          <a class="home-accordion__panel-cta home-accordion__panel-cta--primary" href="${escapeHtml(normalizeHref(category.cta.href))}" aria-controls="home-library-view-${escapeHtml(id)}" aria-expanded="false" data-home-library-open="${escapeHtml(id)}">${escapeHtml(category.cta.label)} <span aria-hidden="true">${HOME_ACCORDION_ICONS.arrow}</span></a>`
          : `          <a class="home-accordion__panel-cta" href="${escapeHtml(normalizeHref(category.cta.href))}">${escapeHtml(category.cta.label)} <span aria-hidden="true">${HOME_ACCORDION_ICONS.arrow}</span></a>`)
      : '';
    const panelHeader = profile
      ? [
          '          <header class="home-accordion__panel-head home-accordion__panel-head--profile">',
          '            <div class="home-accordion__profile-copy">',
          `              <p class="home-accordion__eyebrow">${escapeHtml(label)}</p>`,
          '              <div class="home-accordion__title-row">',
          `                <span class="home-accordion__title-icon" data-home-icon="${escapeHtml(categoryIconId)}" aria-hidden="true">${homeAccordionIcon(categoryIconId)}</span>`,
          `                <h3>${escapeHtml(category && category.title || label)}</h3>`,
          '              </div>',
          category && category.lead ? `              <p class="home-accordion__lead">${escapeHtml(category.lead)}</p>` : '',
          context ? `              <p class="home-accordion__context home-accordion__context--profile">${escapeHtml(context)}</p>` : '',
          '            </div>',
          '            <figure class="home-accordion__profile-portrait">',
          `              <img src="${escapeHtml(profile.image)}" alt="${escapeHtml(profile.imageAlt || '')}" loading="eager" decoding="async"${profile.imageWidth ? ` width="${escapeHtml(profile.imageWidth)}"` : ''}${profile.imageHeight ? ` height="${escapeHtml(profile.imageHeight)}"` : ''}>`,
          '            </figure>',
          '          </header>'
        ].filter(Boolean).join('\n')
      : [
          '          <header class="home-accordion__panel-head">',
          `            <p class="home-accordion__eyebrow">${escapeHtml(label)}</p>`,
          '            <div class="home-accordion__title-row">',
          `              <span class="home-accordion__title-icon" data-home-icon="${escapeHtml(categoryIconId)}" aria-hidden="true">${homeAccordionIcon(categoryIconId)}</span>`,
          `              <h3>${escapeHtml(category && category.title || label)}</h3>`,
          '            </div>',
          category && category.lead ? `            <p class="home-accordion__lead">${escapeHtml(category.lead)}</p>` : '',
          '          </header>'
        ].filter(Boolean).join('\n');

    return [
      `    <article class="home-accordion__item home-accordion__item--${escapeHtml(id)}${hasInlineLibrary ? ' home-accordion__item--has-library' : ''}${isActive ? ' is-active' : ''}" data-home-accordion-item="${escapeHtml(id)}" aria-labelledby="${escapeHtml(triggerId)}" style="--panel-color: ${escapeHtml(color)}; --panel-color-end: ${escapeHtml(colorEnd)};">`,
      '      <h2 class="home-accordion__heading">',
      `        <button class="home-accordion__rail" id="${escapeHtml(triggerId)}" type="button" aria-expanded="${isActive}" aria-controls="${escapeHtml(panelId)}"${isActive ? ' aria-disabled="true"' : ''} data-home-accordion-trigger="${escapeHtml(id)}" data-site-tab="${escapeHtml(id)}" data-site-tab-category="${escapeHtml(id)}"${isActive ? ' data-site-tab-active="true"' : ''}>`,
      `          <span class="home-accordion__rail-icon" data-home-icon="${escapeHtml(categoryIconId)}" aria-hidden="true">${homeAccordionIcon(categoryIconId)}</span>`,
      `          <span class="home-accordion__rail-label">${escapeHtml(label)}</span>`,
      '          <span class="home-accordion__rail-chevron" aria-hidden="true"></span>',
      '        </button>',
      '      </h2>',
      `      <section class="home-accordion__panel" id="${escapeHtml(panelId)}" role="region" aria-labelledby="${escapeHtml(triggerId)}" data-home-accordion-panel="${escapeHtml(id)}"${isActive ? '' : ' hidden inert'}>`,
      `        <div class="home-accordion__scroller" data-home-accordion-scroller aria-label="${escapeHtml(label)} content">`,
      panelHeader,
      contextHtml,
      metaHtml,
      cards ? `          <ul class="home-accordion__cards">\n${cards}\n          </ul>` : '',
      timelineHtml,
      cta,
      libraryHtml,
      '        </div>',
      '      </section>',
      '    </article>'
    ].filter(Boolean).join('\n');
  }).join('\n');

  const noScriptLinks = categories
    .map((category) => category && category.cta && category.cta.href && category.cta.label
      ? `<a href="${escapeHtml(normalizeHref(category.cta.href))}">${escapeHtml(category.cta.label)}</a>`
      : '')
    .filter(Boolean)
    .join('');

  return [
    `<section${sectionAttrs(section, 'home-accordion')} data-home-accordion data-default-panel="${escapeHtml(defaultPanel)}" data-active-panel="${escapeHtml(defaultPanel)}" data-home-view="overview" aria-labelledby="home-accordion-title">`,
    `  <h1 class="visually-hidden" id="home-accordion-title">${escapeHtml(props.accessibleTitle || 'Explore Daniel Short')}</h1>`,
    '  <div class="home-accordion__shell" data-site-tab-rail data-site-tab-rail-mode="overview">',
    panels,
    '  </div>',
    noScriptLinks ? `  <noscript><nav class="home-accordion__noscript" aria-label="Explore the site">${noScriptLinks}</nav></noscript>` : '',
    '</section>'
  ].filter(Boolean).join('\n');
}

const WIDGETS = [
  {
    type: 'hero',
    label: 'Hero',
    category: 'Core',
    description: 'Top page banner with headline, lead, and action links.',
    defaultProps: {
      eyebrow: 'Page',
      title: 'New page section',
      lead: 'Add a concise supporting message.',
      primaryLabel: 'Primary action',
      primaryHref: '#main',
      secondaryLabel: '',
      secondaryHref: ''
    },
    fields: [
      { name: 'eyebrow', label: 'Eyebrow', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'lead', label: 'Lead', type: 'textarea' },
      { name: 'primaryLabel', label: 'Primary label', type: 'text' },
      { name: 'primaryHref', label: 'Primary link', type: 'text' },
      { name: 'secondaryLabel', label: 'Secondary label', type: 'text' },
      { name: 'secondaryHref', label: 'Secondary link', type: 'text' }
    ],
    render: renderHero
  },
  {
    type: 'rich-text',
    label: 'Rich Text',
    category: 'Content',
    description: 'Heading and formatted copy block.',
    defaultProps: { kicker: '', title: 'Section title', body: 'Add body copy.' },
    fields: [
      { name: 'kicker', label: 'Kicker', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'body', label: 'Body', type: 'textarea' }
    ],
    render: renderRichText
  },
  {
    type: 'cta',
    label: 'Call To Action',
    category: 'Core',
    description: 'Conversion block with a button.',
    defaultProps: { title: 'Ready to connect?', body: 'Add a short call to action.', label: 'Contact', href: 'contact' },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'body', label: 'Body', type: 'textarea' },
      { name: 'label', label: 'Button label', type: 'text' },
      { name: 'href', label: 'Button link', type: 'text' }
    ],
    render: renderCta
  },
  {
    type: 'image-gallery',
    label: 'Image Gallery',
    category: 'Media',
    description: 'Small image gallery section.',
    defaultProps: { title: 'Image gallery', images: [{ src: 'img/hero/head.png', alt: 'Gallery image', caption: 'Gallery image' }] },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'images', label: 'Images', type: 'json' }
    ],
    render: renderImageGallery
  },
  {
    type: 'document-list',
    label: 'Document List',
    category: 'Assets',
    description: 'List of local documents or downloadable files.',
    defaultProps: { title: 'Documents', documents: [{ label: 'Resume', href: 'https://danielshort-public-documents-886623862678-us-east-2.s3.us-east-2.amazonaws.com/documents/Resume.pdf' }] },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'documents', label: 'Documents', type: 'json' }
    ],
    render: renderDocumentList
  },
  {
    type: 'map',
    label: 'Location Map',
    category: 'Utility',
    description: 'Location block with an optional Google Maps embed and map link.',
    defaultProps: { title: 'Location', body: 'Add location context.', address: 'Grand Junction, CO', buttonLabel: 'Open map', embed: false, zoom: 10 },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'body', label: 'Body', type: 'textarea' },
      { name: 'address', label: 'Address', type: 'text' },
      { name: 'buttonLabel', label: 'Button label', type: 'text' },
      { name: 'embed', label: 'Embed map', type: 'checkbox' },
      { name: 'zoom', label: 'Zoom', type: 'number' }
    ],
    render: renderMap
  },
  {
    type: 'embed',
    label: 'Embed',
    category: 'Utility',
    description: 'Iframe embed for demos, maps, or dashboards.',
    defaultProps: { title: 'Embedded content', src: 'about:blank' },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'src', label: 'Embed URL', type: 'text' }
    ],
    render: renderEmbed
  },
  {
    type: 'kpi-band',
    label: 'KPI Band',
    category: 'Portfolio',
    description: 'Metric strip for impact numbers and measurable outcomes.',
    defaultProps: { kicker: 'Impact', title: 'Measured outcomes', items: [{ value: '99%', label: 'Faster reporting' }, { value: '200+', label: 'Hours saved' }, { value: '$13.1M', label: 'Measured impact' }] },
    fields: [
      { name: 'kicker', label: 'Kicker', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'items', label: 'Metrics', type: 'json' }
    ],
    render: renderKpiBand
  },
  {
    type: 'proof-block',
    label: 'Proof Block',
    category: 'Portfolio',
    description: 'Evidence block for decisions, outcomes, or case-study proof.',
    defaultProps: { kicker: 'Proof', title: 'What changed', lead: 'Add the main evidence or result.', bullets: ['Add supporting evidence.'] },
    fields: [
      { name: 'kicker', label: 'Kicker', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'lead', label: 'Lead', type: 'textarea' },
      { name: 'bullets', label: 'Bullets', type: 'json' }
    ],
    render: renderProofBlock
  },
  {
    type: 'project-grid',
    label: 'Project Grid',
    category: 'Portfolio',
    description: 'Curated project links for audience-specific pages.',
    defaultProps: { kicker: 'Work', title: 'Selected projects', projects: [{ title: 'Project title', href: 'portfolio', summary: 'Add the project outcome or audience fit.' }] },
    fields: [
      { name: 'kicker', label: 'Kicker', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'projects', label: 'Projects', type: 'json' }
    ],
    render: renderProjectGrid
  },
  {
    type: 'certification-strip',
    label: 'Certification Strip',
    category: 'Portfolio',
    description: 'Compact certification list for resume and proof sections.',
    defaultProps: { title: 'Certifications', certifications: [{ title: 'Certification', issuer: 'Issuer', icon: 'img/cert_logos/google-48.png', href: '#' }] },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'certifications', label: 'Certifications', type: 'json' }
    ],
    render: renderCertificationStrip
  },
  {
    type: 'resume-highlight',
    label: 'Resume Highlight',
    category: 'Portfolio',
    description: 'Resume-style accomplishment block with bullets.',
    defaultProps: { title: 'Resume highlight', meta: 'Role or context', bullets: ['Add an accomplishment with a metric, audience, and business result.'] },
    fields: [
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'meta', label: 'Meta', type: 'text' },
      { name: 'bullets', label: 'Bullets', type: 'json' }
    ],
    render: renderResumeHighlight
  },
  {
    type: 'media-showcase',
    label: 'Media Showcase',
    category: 'Media',
    description: 'Single visual proof image with caption.',
    defaultProps: { kicker: 'Preview', title: 'Media showcase', lead: '', src: 'img/hero/head.png', alt: 'Showcase image', caption: 'Add a caption.' },
    fields: [
      { name: 'kicker', label: 'Kicker', type: 'text' },
      { name: 'title', label: 'Title', type: 'text' },
      { name: 'lead', label: 'Lead', type: 'textarea' },
      { name: 'src', label: 'Image', type: 'media' },
      { name: 'alt', label: 'Alt text', type: 'text' },
      { name: 'caption', label: 'Caption', type: 'text' }
    ],
    render: renderMediaShowcase
  },
  {
    type: 'home-accordion',
    label: 'Home Category Accordion',
    category: 'Portfolio',
    description: 'Personal homepage with five attached expanding category panels.',
    defaultProps: {
      accessibleTitle: 'Explore Daniel Short',
      defaultPanel: 'about',
      categories: []
    },
    fields: [
      { name: 'accessibleTitle', label: 'Accessible title', type: 'text' },
      { name: 'defaultPanel', label: 'Default panel', type: 'text' },
      { name: 'categories', label: 'Categories', type: 'json' }
    ],
    render: renderHomeAccordion
  },
  {
    type: 'legacy-html',
    label: 'Existing Section',
    category: 'Advanced',
    description: 'Preserved existing markup, editable visually where possible.',
    defaultProps: { html: '<section class="surface-band reveal"><div class="wrapper"><h2 class="section-title">Existing section</h2><p>Edit this section visually or use Advanced HTML.</p></div></section>' },
    fields: [
      { name: 'html', label: 'HTML', type: 'textarea' }
    ],
    render: renderLegacyHtml
  }
];

const WIDGET_MAP = new Map(WIDGETS.map((widget) => [widget.type, widget]));

function getWidgetDefinitions() {
  return WIDGETS.map((widget) => {
    const section = createDefaultSection(widget.type);
    return {
      type: widget.type,
      label: widget.label,
      category: widget.category,
      description: widget.description,
      fields: widget.fields,
      defaultProps: widget.defaultProps,
      defaultSection: {
        ...section,
        html: renderSection(section)
      }
    };
  });
}

function createDefaultSection(type) {
  const widget = WIDGET_MAP.get(type) || WIDGET_MAP.get('rich-text');
  return {
    id: `${widget.type}-${Date.now().toString(36)}`,
    type: widget.type,
    label: widget.label,
    enabled: true,
    variant: 'default',
    props: JSON.parse(JSON.stringify(widget.defaultProps || {}))
  };
}

function renderSection(section) {
  if (!section || section.enabled === false) return '';
  const widget = WIDGET_MAP.get(section.type) || WIDGET_MAP.get('legacy-html');
  return widget.render(section);
}

function renderVisualPageBody(page) {
  const sections = Array.isArray(page && page.sections) ? page.sections : [];
  const requestedOrder = Array.isArray(page && page.sectionOrder) ? page.sectionOrder : [];
  const sectionsById = new Map(sections.map((section) => [String(section && section.id || ''), section]));
  const orderedIds = new Set(requestedOrder.map((id) => String(id || '')));
  const orderedSections = requestedOrder
    .map((id) => sectionsById.get(String(id || '')))
    .filter(Boolean)
    .concat(sections.filter((section) => !orderedIds.has(String(section && section.id || ''))));
  const body = orderedSections
    .map((section) => renderSection(section))
    .filter(Boolean)
    .join('\n\n');
  const mainAttributes = { ...((page && page.mainAttributes) || { id: 'main' }) };
  if (page && page.id === 'home' && page.canonicalPath === '/') {
    mainAttributes['data-site-route-content'] = true;
  }
  const mainAttrs = attrsToString(mainAttributes);
  const routeToolbar = page && page.id === 'home' && page.canonicalPath === '/'
    ? '  <div data-site-route-toolbar hidden aria-hidden="true"></div>\n'
    : '';
  return `<main${mainAttrs ? ` ${mainAttrs}` : ''}>\n${routeToolbar}${body}\n</main>`;
}

module.exports = {
  createDefaultSection,
  getHomeAccordionIconDefinitions,
  getWidgetDefinitions,
  renderSection,
  renderVisualPageBody,
  resolveHomeAccordionIconId
};
