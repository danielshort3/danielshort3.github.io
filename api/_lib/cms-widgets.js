'use strict';

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
    : [{ label: 'Document', href: 'documents/Resume.pdf' }];
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
    .sort((a, b) => (b.rank - a.rank) || (a.index - b.index));
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
    defaultProps: { title: 'Documents', documents: [{ label: 'Resume', href: 'documents/Resume.pdf' }] },
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
  const mainAttrs = attrsToString((page && page.mainAttributes) || { id: 'main' });
  return `<main${mainAttrs ? ` ${mainAttrs}` : ''}>\n${body}\n</main>`;
}

module.exports = {
  createDefaultSection,
  getWidgetDefinitions,
  renderSection,
  renderVisualPageBody
};
