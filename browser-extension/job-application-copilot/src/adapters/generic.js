const MAX_JOB_TEXT_LENGTH = 240;
const MAX_JOB_DESCRIPTION_LENGTH = 20_000;
const MAX_NEARBY_TEXT_LENGTH = 320;

export const GENERIC_CANDIDATE_SELECTOR = [
  'input',
  'textarea',
  'select',
  '[contenteditable="true"]',
  '[contenteditable="plaintext-only"]',
  '[role="textbox"]',
  '[role="combobox"]'
].join(',');

const FIELD_CONTAINER_SELECTOR = [
  'fieldset',
  '.form-group',
  '.form-field',
  '.form-item',
  '.field-group',
  '.field',
  '.input-wrapper',
  '.question',
  '.application-question',
  '[data-field]',
  '[data-question]',
  '[data-testid*="field" i]',
  '[data-qa*="field" i]',
  '[data-automation-id*="field" i]',
  '[role="group"]'
].join(',');

const normalizeText = (value, maxLength = MAX_JOB_TEXT_LENGTH) => String(value || '')
  .replace(/[\u0000-\u001f\u007f]+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim()
  .slice(0, maxLength);

const safeQuery = (root, selector) => {
  try {
    return root?.querySelector?.(selector) || null;
  } catch {
    return null;
  }
};

const safeQueryAll = (root, selector) => {
  try {
    return Array.from(root?.querySelectorAll?.(selector) || []);
  } catch {
    return [];
  }
};

const textFromNode = (node) => normalizeText(node?.textContent || '');

const normalizeMarker = (value, maxLength = 600) => normalizeText(String(value || '')
  .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
  .replace(/[^a-z0-9]+/giu, ' '), maxLength).toLowerCase();

const stripMarkup = (value, maxLength = MAX_JOB_DESCRIPTION_LENGTH) => normalizeText(String(value || '')
  .replace(/<script\b[^>]*>[\s\S]*<\/script>/giu, ' ')
  .replace(/<style\b[^>]*>[\s\S]*<\/style>/giu, ' ')
  .replace(/<[^>]+>/gu, ' ')
  .replace(/&nbsp;|&#160;/giu, ' ')
  .replace(/&amp;/giu, '&')
  .replace(/&quot;|&#34;/giu, '"')
  .replace(/&#39;|&apos;/giu, "'"), maxLength);

const metaContent = (doc, selectors) => {
  for (const selector of selectors) {
    const node = safeQuery(doc, selector);
    const value = normalizeText(node?.getAttribute?.('content') || '');
    if (value) return value;
  }
  return '';
};

const visibleText = (doc, selectors) => {
  for (const selector of selectors) {
    const node = safeQuery(doc, selector);
    const value = textFromNode(node);
    if (value) return value;
  }
  return '';
};

const visibleLongText = (doc, selectors, maxLength = MAX_JOB_DESCRIPTION_LENGTH) => {
  for (const selector of selectors) {
    const node = safeQuery(doc, selector);
    if (!node || node.hidden || node.getAttribute?.('aria-hidden') === 'true') continue;
    const value = normalizeText(node.textContent || '', maxLength);
    if (value) return value;
  }
  return '';
};

const findJobPosting = (value, depth = 0) => {
  if (!value || depth > 5) return null;
  if (Array.isArray(value)) {
    for (const item of value.slice(0, 40)) {
      const match = findJobPosting(item, depth + 1);
      if (match) return match;
    }
    return null;
  }
  if (typeof value !== 'object') return null;
  const types = Array.isArray(value['@type']) ? value['@type'] : [value['@type']];
  if (types.some((type) => String(type || '').toLowerCase() === 'jobposting')) return value;
  if (value['@graph']) return findJobPosting(value['@graph'], depth + 1);
  return null;
};

const readStructuredJobPosting = (doc) => {
  for (const node of safeQueryAll(doc, 'script[type="application/ld+json"]').slice(0, 20)) {
    const raw = String(node?.textContent || '').trim();
    if (!raw || raw.length > 1_000_000) continue;
    try {
      const posting = findJobPosting(JSON.parse(raw));
      if (posting) return posting;
    } catch {}
  }
  return null;
};

const structuredLocation = (posting) => {
  const locations = Array.isArray(posting?.jobLocation) ? posting.jobLocation : [posting?.jobLocation];
  for (const location of locations) {
    const address = location?.address || location;
    const value = normalizeText([
      address?.addressLocality,
      address?.addressRegion,
      address?.addressCountry?.name || address?.addressCountry
    ].filter(Boolean).join(', '));
    if (value) return value;
  }
  return normalizeText(posting?.jobLocationType);
};

const isValidationFeedbackNode = (node) => {
  const role = normalizeText(node?.getAttribute?.('role') || '', 40).toLowerCase();
  const ariaLive = normalizeText(node?.getAttribute?.('aria-live') || '', 40).toLowerCase();
  const marker = [
    node?.getAttribute?.('id'),
    node?.getAttribute?.('class'),
    node?.getAttribute?.('data-error'),
    node?.getAttribute?.('data-validation')
  ].map(value => normalizeText(value || '', 160).toLowerCase()).join(' ');
  return role === 'alert'
    || role === 'status'
    || Boolean(ariaLive)
    || /(?:^|[\s_-])(?:danger|error|feedback|invalid|validation)(?:$|[\s_-])/u.test(marker);
};

const nearbyHelpText = (element) => {
  const container = genericAdapter.getFieldContainer(element);
  const describedIds = normalizeText(element?.getAttribute?.('aria-describedby') || '', 320)
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 6);
  let localRoot = null;
  try {
    localRoot = element?.getRootNode?.() || element?.ownerDocument || null;
  } catch {}
  const describedNodes = describedIds.map((id) => {
    try {
      return localRoot?.getElementById?.(id) || element?.ownerDocument?.getElementById?.(id) || null;
    } catch {
      return null;
    }
  }).filter(Boolean);
  const nodes = container ? safeQueryAll(container, [
    '[data-field-description]',
    '[data-help-text]',
    '[data-testid*="description" i]',
    '[data-automation-id*="description" i]',
    '.field-description',
    '.help-text',
    '.hint',
    '.instruction',
    'small'
  ].join(',')) : [];
  return normalizeText([...describedNodes, ...nodes]
    .filter((node, index, all) => all.indexOf(node) === index)
    .filter(node => node !== element
      && !node.hidden
      && node.getAttribute?.('aria-hidden') !== 'true'
      && !isValidationFeedbackNode(node))
    .map(node => node.textContent || '')
    .join(' '), MAX_NEARBY_TEXT_LENGTH);
};

const humanizeHostname = (hostname) => {
  const firstLabel = String(hostname || '').split('.')[0] || '';
  return normalizeText(firstLabel.replace(/[-_]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase()));
};

const ALLOWED_JOB_QUERY_KEYS = new Set([
  'gh_jid',
  'jid',
  'job',
  'job_id',
  'jobid',
  'position_id',
  'positionid',
  'posting_id',
  'postingid',
  'req_id',
  'reqid',
  'requisition_id',
  'requisitionid'
]);

export const sanitizeJobUrl = (rawUrl) => {
  try {
    const url = new URL(String(rawUrl || ''));
    if (!/^https?:$/.test(url.protocol)) return '';
    url.username = '';
    url.password = '';
    url.hash = '';
    const retained = [];
    url.searchParams.forEach((value, key) => {
      const normalizedKey = String(key || '').toLowerCase();
      if (!ALLOWED_JOB_QUERY_KEYS.has(normalizedKey)) return;
      const boundedValue = normalizeText(value, 160);
      if (boundedValue) retained.push([key, boundedValue]);
    });
    url.search = '';
    retained.slice(0, 2).forEach(([key, value]) => url.searchParams.append(key, value));
    return url.toString().slice(0, 2048);
  } catch {
    return '';
  }
};

const documentTitleMetadata = (doc, hostname) => {
  const documentTitle = normalizeText(doc?.title || '');
  const greenhouseMatch = /^Job Application for (.+) at (.+)$/iu.exec(documentTitle);
  if (greenhouseMatch) {
    return {
      title: normalizeText(greenhouseMatch[1]),
      company: normalizeText(greenhouseMatch[2])
    };
  }
  if (/(^|\.)lever\.co$/iu.test(hostname)) {
    const divider = documentTitle.indexOf(' - ');
    if (divider > 0) {
      return {
        company: normalizeText(documentTitle.slice(0, divider)),
        title: normalizeText(documentTitle.slice(divider + 3))
      };
    }
  }
  return { company: '', title: '' };
};

export const extractGenericJobMetadata = (doc, rawUrl) => {
  let hostname = '';
  try {
    hostname = new URL(String(rawUrl || '')).hostname.toLowerCase();
  } catch {}

  const documentMetadata = documentTitleMetadata(doc, hostname);
  const structured = readStructuredJobPosting(doc);
  const title = visibleText(doc, [
    '[data-job-title]',
    '.job-title',
    '.posting-header h2',
    'main h1',
    'h1'
  ]) || metaContent(doc, [
    'meta[property="og:title"]',
    'meta[name="twitter:title"]'
  ]) || normalizeText(structured?.title) || documentMetadata.title;
  const company = visibleText(doc, [
    '[data-company-name]',
    '.company-name',
    '.job-company'
  ]) || metaContent(doc, [
    'meta[property="og:site_name"]',
    'meta[name="application-name"]'
  ]) || normalizeText(structured?.hiringOrganization?.name) || documentMetadata.company;
  const location = visibleText(doc, [
    '[data-job-location]',
    '.job-location',
    '.job__location',
    '.posting-location',
    '.location'
  ]) || metaContent(doc, [
    'meta[name="job:location"]',
    'meta[property="job:location"]'
  ]) || structuredLocation(structured);
  const description = visibleLongText(doc, [
    '[data-job-description]',
    '.job__description',
    '.job-description',
    '.posting-description',
    '.job-description-container',
    '[itemprop="description"]',
    '[data-testid*="job-description" i]',
    '[data-automation-id*="job-description" i]'
  ]) || stripMarkup(structured?.description);

  return {
    company: normalizeText(company || humanizeHostname(hostname)),
    title: normalizeText(title),
    jobUrl: sanitizeJobUrl(rawUrl),
    location: normalizeText(location),
    source: normalizeText(hostname || 'Web application'),
    ...(description ? { description } : {})
  };
};

const GENERIC_APPLICATION_ROOT_SELECTORS = [
  '[data-application-form]',
  '[data-candidate-application]',
  '[data-testid="application-form"]',
  '[data-testid*="candidate-application" i]',
  '[data-testid*="apply-flow" i]',
  '[data-automation-id="applicationForm"]',
  '[data-automation-id*="candidateApplication" i]',
  '[data-automation-id*="applyFlow" i]',
  '[data-qa="application-form"]',
  '[data-qa*="candidate-application" i]',
  '.application-form',
  '#application-form',
  '[role="form"][aria-label*="application" i]'
];
const GENERIC_APPLICATION_MARKER_RE = /(?:^|[\s_-])(?:application|apply|applicant|candidate|job)(?:$|[\s_-])/iu;
const GENERIC_AUXILIARY_FORM_RE = /(?:^|[\s_-])(?:alert|newsletter|refer|referral|search|subscribe|subscription)(?:$|[\s_-])/iu;
const GENERIC_IDENTITY_RE = /\b(?:first[\s_-]*name|last[\s_-]*name|full[\s_-]*name|candidate[\s_-]*name|email|e[\s_-]*mail|phone|telephone|mobile)\b/iu;
const GENERIC_RESUME_RE = /\b(?:resume|curriculum[\s_-]*vitae|\bcv\b)\b/iu;

const genericRootMarker = (root) => normalizeMarker([
  root?.tagName,
  root?.id,
  root?.name,
  root?.className,
  root?.getAttribute?.('action'),
  root?.getAttribute?.('aria-label'),
  root?.getAttribute?.('data-testid'),
  root?.getAttribute?.('data-automation-id'),
  root?.title,
  root?.location?.href
].filter(Boolean).join(' '), 600);

const genericControlMarker = (control) => normalizeMarker([
  control?.name,
  control?.id,
  control?.getAttribute?.('aria-label'),
  control?.getAttribute?.('placeholder'),
  control?.getAttribute?.('title'),
  control?.getAttribute?.('autocomplete'),
  control?.getAttribute?.('data-field-name'),
  control?.getAttribute?.('data-testid'),
  control?.getAttribute?.('data-automation-id'),
  ...Array.from(control?.labels || []).map((label) => label?.textContent || '')
].join(' '), 480);

const controlsForRoot = (root, doc = null) => {
  const controls = safeQueryAll(root, GENERIC_CANDIDATE_SELECTOR);
  try {
    controls.push(...Array.from(root?.elements || []));
  } catch {}
  const rootId = normalizeText(root?.id || root?.getAttribute?.('id') || '', 160);
  if (doc && rootId) {
    const escapedId = globalThis.CSS?.escape ? globalThis.CSS.escape(rootId) : rootId.replace(/["\\]/g, '\\$&');
    controls.push(...safeQueryAll(doc, `[form="${escapedId}"]`));
  }
  return Array.from(new Set(controls));
};

export const scoreGenericApplicationRoot = (root) => {
  if (!root) return Number.NEGATIVE_INFINITY;
  const controls = controlsForRoot(root, root?.ownerDocument || null);
  const rootMarker = genericRootMarker(root);
  const markers = controls.map(genericControlMarker);
  const hasPassword = controls.some((control) => {
    const type = String(control?.type || control?.getAttribute?.('type') || '').toLowerCase();
    const autocomplete = String(control?.getAttribute?.('autocomplete') || '').toLowerCase();
    return type === 'password' || autocomplete.includes('password');
  });
  const hasResume = controls.some((control, index) => {
    const type = String(control?.type || control?.getAttribute?.('type') || '').toLowerCase();
    return type === 'file' && GENERIC_RESUME_RE.test(markers[index] || '');
  });
  const identityCount = markers.filter((marker) => GENERIC_IDENTITY_RE.test(marker)).length;
  const questionCount = controls.filter((control) => {
    const tagName = String(control?.tagName || '').toLowerCase();
    return tagName === 'textarea' || tagName === 'select';
  }).length;
  let score = 0;
  if (GENERIC_APPLICATION_MARKER_RE.test(rootMarker)) score += 8;
  if (hasResume) score += 8;
  score += Math.min(identityCount, 4) * 2;
  if (identityCount >= 2) score += 4;
  score += Math.min(questionCount, 3);
  if (hasPassword) score -= 20;
  if (GENERIC_AUXILIARY_FORM_RE.test(rootMarker)) score -= 20;
  return score;
};

export const selectGenericApplicationRoot = (doc) => {
  const forms = safeQueryAll(doc, 'form')
    .filter((node) => String(node?.tagName || '').toLowerCase() === 'form');
  const explicitRoots = GENERIC_APPLICATION_ROOT_SELECTORS
    .flatMap((selector) => safeQueryAll(doc, selector));
  const roots = Array.from(new Set([...explicitRoots, ...forms]));
  let best = null;
  let bestScore = 7;
  roots.forEach((root) => {
    const score = scoreGenericApplicationRoot(root);
    if (score > bestScore) {
      best = root;
      bestScore = score;
    }
  });
  if (best) return best;
  if (forms.length) return null;
  return scoreGenericApplicationRoot(doc) > 7 ? doc : null;
};
export const genericAdapter = Object.freeze({
  id: 'generic',
  sourceName: 'Web application',
  matches: () => true,
  getApplicationRoot: selectGenericApplicationRoot,
  getDiscoveryContexts: (doc) => {
    const root = selectGenericApplicationRoot(doc);
    return root ? [{ kind: 'application', root }] : [];
  },
  collectCandidates: (doc) => {
    const root = selectGenericApplicationRoot(doc);
    return root ? controlsForRoot(root, doc) : [];
  },
  getFieldContainer: (element) => {
    try {
      return element?.closest?.(FIELD_CONTAINER_SELECTOR) || element?.parentElement || null;
    } catch {
      return element?.parentElement || null;
    }
  },
  getSupplementalLabel: (element) => {
    const container = genericAdapter.getFieldContainer(element);
    const node = safeQuery(container, [
      ':scope > .field-label',
      ':scope > .question-label',
      ':scope > .application-label',
      ':scope > .control-label',
      ':scope > label',
      ':scope > legend',
      '.application-label'
    ].join(','));
    return textFromNode(node)
      || normalizeText(element?.getAttribute?.('title') || '')
      || normalizeText(element?.getAttribute?.('placeholder') || '')
      || normalizeText(element?.getAttribute?.('data-testid') || '')
      || normalizeText(element?.getAttribute?.('data-automation-id') || '');
  },
  getNearbyText: nearbyHelpText,
  extractJobMetadata: extractGenericJobMetadata
});

export const adapterInternals = Object.freeze({
  controlsForRoot,
  normalizeText,
  normalizeMarker,
  safeQuery,
  safeQueryAll,
  textFromNode,
  visibleLongText
});
