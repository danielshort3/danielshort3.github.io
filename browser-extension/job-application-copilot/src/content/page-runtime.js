import { selectAdapter } from '../adapters/index.js';

export const RUNTIME_CHANNEL = 'job-application-copilot';
export const RUNTIME_VERSION = 1;

export const RISK_CLASSES = Object.freeze({
  F0_EXCLUDED: 'F0_EXCLUDED',
  F1_VERIFIED: 'F1_VERIFIED',
  F2_REVIEW: 'F2_REVIEW',
  F3_SENSITIVE: 'F3_SENSITIVE',
  F4_CONSENT: 'F4_CONSENT'
});

const INBOUND_MESSAGE_TYPES = new Set([
  'PAGE_SCAN_REQUEST',
  'PAGE_PROPOSALS_UPDATE',
  'FIELD_FILL_REQUEST',
  'SANITIZED_EXPORT_REQUEST'
]);
const OUTBOUND_MESSAGE_TYPES = new Set([
  'PAGE_SCAN_RESULT',
  'FIELD_SELECTED',
  'FIELD_FILL_RESULT',
  'SANITIZED_EXPORT_RESULT'
]);
const ACTIONABLE_RISKS = new Set([
  RISK_CLASSES.F1_VERIFIED,
  RISK_CLASSES.F2_REVIEW
]);
const MAX_REQUEST_ID_LENGTH = 128;
const MAX_FIELDS = 50;
const MAX_CANDIDATES = 600;
const MAX_LABEL_LENGTH = 240;
const MAX_NEARBY_LENGTH = 320;
const MAX_JOB_DESCRIPTION_LENGTH = 20_000;
const MAX_OPTIONS = 50;
const MAX_OPTION_LENGTH = 160;
const MAX_PROPOSAL_LENGTH = 12_000;
const MAX_PROPOSALS = 50;
const MAX_ARRAY_ANSWERS = 50;
const MAX_CITATION_COUNT = 12;
const MAX_FRAME_DOCUMENTS = 12;
const MAX_SHADOW_ROOTS = 64;
const MAX_DISCOVERY_CONTEXTS = 8;
const MAX_DISCOVERY_CONTEXT_COUNT = 600;
const MAX_DISCOVERY_ELEMENTS = 5_000;
const MIN_ACTIONABLE_DIMENSION = 2;
const OVERLAY_HOST_ID = '__job_application_copilot_runtime_v1__';
const RUNTIME_INSTANCE_KEY = '__jobApplicationCopilotPageRuntimeV1__';
const STRUCTURAL_FIELD_SELECTOR = [
  'input',
  'textarea',
  'select',
  '[contenteditable="true"]',
  '[contenteditable="plaintext-only"]',
  '[role="textbox"]',
  '[role="combobox"]'
].join(',');
const STRUCTURAL_OPTION_SELECTOR = [
  'button[type="button"]',
  'button:not([type])',
  '[role="radio"]'
].join(',');

const CAPTCHA_SELECTOR = [
  'iframe[src*="recaptcha" i]',
  'iframe[src*="hcaptcha" i]',
  'iframe[src*="challenges.cloudflare.com" i]',
  'iframe[title*="captcha" i]',
  'script[src*="recaptcha" i]',
  'script[src*="hcaptcha" i]',
  'script[src*="challenges.cloudflare.com" i]',
  'script[src*="turnstile" i]',
  '.g-recaptcha',
  '.grecaptcha-badge',
  '.h-captcha',
  '.cf-turnstile',
  '[data-sitekey]',
  '[data-recaptcha]',
  '[data-hcaptcha]',
  '[data-turnstile]',
  'textarea[name="g-recaptcha-response" i]',
  'textarea[name="h-captcha-response" i]',
  'input[name="cf-turnstile-response" i]',
  'textarea[name="cf-turnstile-response" i]',
  '[id^="g-recaptcha-response" i]',
  '[id^="h-captcha-response" i]',
  '[id*="captcha" i]',
  '[class*="captcha" i]',
  'input[name*="captcha" i]'
].join(',');

const CAPTCHA_INTERACTION_SELECTOR = [
  'iframe[src*="recaptcha" i]',
  'iframe[src*="hcaptcha" i]',
  'iframe[src*="challenges.cloudflare.com" i]',
  'iframe[title*="captcha" i]',
  '.g-recaptcha',
  '.h-captcha',
  '.cf-turnstile',
  '[data-recaptcha]',
  '[data-hcaptcha]',
  '[data-turnstile]',
  '[role="dialog"][aria-label*="captcha" i]',
  'input[name*="captcha" i]:not([type="hidden" i])'
].join(',');

const F0_PATTERN = /\b(?:captcha|one[\s_-]*time[\s_-]*(?:code|password)|otp|pin|verification[\s_-]*code|security[\s_-]*(?:code|question|answer)|system[\s_-]*field|csrf|authenticity[\s_-]*token)\b/i;
const BOT_TRAP_PATTERN = /\b(?:honeypot|honey[\s_-]*pot|bot[\s_-]*trap|robots?[\s_-]*only|leave[\s_-]+(?:this[\s_-]+)?(?:field[\s_-]+)?blank|do[\s_-]*not[\s_-]+(?:enter|fill)(?:[\s_-]+(?:this|out))?)\b/i;
const ACCOUNT_CREDENTIAL_PATTERN = /\b(?:user[\s_-]*name|user[\s_-]*id|login|password|passcode|access[\s_-]*code|api[\s_-]*key|secret[\s_-]*key)\b/i;
const F3_PATTERN = /\b(?:race|racial|ethnicity|ethnic|gender|sex(?:ual)?|male|female|non[\s_-]*binary|pronouns?|sexual[\s_-]*orientation|marital[\s_-]*status|religion|political[\s_-]*affiliation|union[\s_-]*(?:member|membership)|age|over[\s_-]*18|hispanic|latino|asian|african[\s_-]*american|native[\s_-]*(?:american|hawaiian)|pacific[\s_-]*islander|indigenous|equal[\s_-]*employment[\s_-]*opportunity|eeo|self[\s_-]*identification|protected[\s_-]*class|disabilit(?:y|ies)|disabled|medical|health[\s_-]*(?:condition|history)|genetic|pregnan(?:t|cy)|workers?[\s_-]*compensation|drug[\s_-]*test|veteran|military[\s_-]*status|felon(?:y|ies)?|criminal|conviction|arrest|background[\s_-]*check|social[\s_-]*security|ssn|passport|driver'?s?[\s_-]*licen[cs]e|national[\s_-]*(?:id|insurance)|government[\s_-]*(?:id|identifier)|tax[\s_-]*(?:id|identifier)|taxpayer[\s_-]*identification|\btin\b|\bsin\b|alien[\s_-]*registration|a[\s_-]*number|biometric|date[\s_-]*of[\s_-]*birth|birth[\s_-]*date|dob)\b/i;
const F4_PATTERN = /\b(?:signature|e[\s_-]*signature|electronic[\s_-]*signature|sign[\s_-]*here|initials?|consent|attest|attestation|i[\s_-]*certify|certify[\s_-]*(?:that|these|the[\s_-]*(?:information|answers))|certification[\s_-]*(?:statement|of[\s_-]*(?:accuracy|truth))|acknowledge|agreement|agree[\s_-]*to|i[\s_-]*agree|accept[\s_-]*(?:the[\s_-]*)?terms|terms[\s_-]*(?:and|of)|privacy[\s_-]*(?:policy|notice)|i[\s_-]*(?:understand|declare|swear|authorize)|authorization[\s_-]*to[\s_-]*(?:release|contact|verify))\b/i;
const THIRD_PARTY_CONTACT_PATTERN = /\b(?:(?:professional|personal|employment|academic)[\s_-]*references?|references?(?:['’]s)?[\s_-]*(?:name|email|e[\s_-]*mail|phone|telephone|mobile|address|contact|website)|referee(?:s)?(?:[\s_-]*(?:name|email|e[\s_-]*mail|phone|telephone|mobile|address|contact|website))?|(?:supervisor|manager|recruiter|hiring[\s_-]*manager)(?:['’]s)?[\s_-]*(?:name|email|e[\s_-]*mail|phone|telephone|mobile|address|contact|website)|(?:emergency|hiring|hiring[\s_-]*team|school|college|university|employer|company)[\s_-]*(?:contact|contact[\s_-]*person|contact[\s_-]*name|contact[\s_-]*email|contact[\s_-]*phone)|(?:school|college|university|employer|company)[\s_-]*(?:email|e[\s_-]*mail|phone|telephone|mobile|address|website)|(?:name|email|e[\s_-]*mail|phone|telephone|mobile|address|website)[\s_-]*(?:of|for)[\s_-]*(?:a[\s_-]*)?(?:reference|referee|supervisor|manager|recruiter|hiring[\s_-]*manager)|contact[\s_-]*person)\b/i;
const F2_PATTERN = /\b(?:cover[\s_-]*letter|additional[\s_-]*(?:information|comments)|tell[\s_-]*us|describe|explain|why|reason[\s_-]*(?:for[\s_-]*)?leaving|years?[\s_-]*(?:of[\s_-]*)?experience|salary|compensation|pay[\s_-]*(?:expectation|requirement)|desired[\s_-]*pay|availability|available[\s_-]*(?:date|to[\s_-]*start)|start[\s_-]*date|interview[\s_-]*date|relocat(?:e|ion)|sponsor(?:ship)?|work[\s_-]*authori[sz]ation|authorized[\s_-]*to[\s_-]*work|citizenship|visa|notice[\s_-]*period|travel[\s_-]*(?:required|percentage)|work[\s_-]*(?:schedule|shift|arrangement)|remote|on[\s_-]*site|hybrid|referr(?:al|ed)|how[\s_-]*(?:did[\s_-]*)?you[\s_-]*(?:hear|learn)|school|college|university|education|degree|major|field[\s_-]*of[\s_-]*study|gpa|graduation[\s_-]*date|employer|company[\s_-]*name|job[\s_-]*title|position[\s_-]*title|current[\s_-]*role|employment[\s_-]*(?:start|end)[\s_-]*date|professional[\s_-]*certifications?|certifications?|credentials?|skills?|languages?)\b/i;
const THIRD_PARTY_PROFILE_PATTERN = /\b(?:reference|referee|supervisor|manager|recruiter|employer|company|school|college|university)\b.{0,60}\bprofile\b|\bprofile\b.{0,60}\b(?:of|for)\b.{0,20}\b(?:reference|referee|supervisor|manager|recruiter|employer|company|school|college|university)\b/i;
const COMMUNICATION_PREFERENCE_PATTERN = /\b(?:preferred[\s_-]*(?:method|mode)[\s_-]*of[\s_-]*communication|communication[\s_-]*(?:method|preference)|preferred[\s_-]*contact[\s_-]*(?:method|channel))\b/i;
const REVIEW_FACT_PATTERN = /\b(?:legally[\s_-]*(?:eligible|permitted)[\s_-]*to[\s_-]*work|eligible[\s_-]*to[\s_-]*work|previously[\s_-]*worked|worked[\s_-]*for|former[\s_-]*employee|current[\s_-]*(?:company|employer|location))\b/i;
const SENSITIVE_IDENTITY_PATTERN = /\btransgender\b/i;
const MANUAL_WORK_IDENTITY_PATTERN = /\b(?:citizenship|nationality|immigration[\s_-]*status|i[\s_-]*9|e[\s_-]*verify|itar|ear|export[\s_-]*control|security[\s_-]*clearance)\b/i;
const MANUAL_IMMIGRATION_DETAIL_PATTERN = /\b(?:lawful[\s_-]*permanent[\s_-]*resident|permanent[\s_-]*resident|green[\s_-]*card|employment[\s_-]*authorization[\s_-]*document|work[\s_-]*permit|h[\s_-]*1b|o[\s_-]*1|l[\s_-]*1|f[\s_-]*1|j[\s_-]*1|tn[\s_-]*status)\b/i;
const VISA_DETAIL_PATTERN = /\bvisa\b/i;
const VISA_STATUS_OR_TYPE_PATTERN = /\b(?:visa\b.{0,40}\b(?:status|type|class|category|kind)|(?:status|type|class|category|kind)\b.{0,40}\bvisa)\b/i;
const VISA_SPONSORSHIP_PATTERN = /\b(?:visa\b.{0,50}\bsponsor|sponsor\w*\b.{0,50}\bvisa)\b/i;
const CLEAR_VISA_SPONSORSHIP_PATTERN = /\b(?:visa\b.{0,50}\bsponsor\w*|sponsor\w*\b.{0,50}\bvisa)\b/i;
const F1_PATTERN = /\b(?:first[\s_-]*name|middle[\s_-]*name|last[\s_-]*name|full[\s_-]*name|preferred[\s_-]*name|email|e[\s_-]*mail|phone|telephone|mobile|home[\s_-]*address|mailing[\s_-]*address|street[\s_-]*address|address[\s_-]*line|city|state|province|postal[\s_-]*code|zip[\s_-]*code|country|linkedin|github|portfolio|website|web[\s_-]*site|personal[\s_-]*(?:website|site|url)|candidate[\s_-]*(?:website|site|url))\b/i;
const CUSTOM_LINK_FIELD_PATTERN = /\b(?:url|website|web[\s_-]*site|link|profile)\b/i;
const UNLABELED_FIELD_PATTERN = /\b(?:unlabeled|unknown)[\s_-]*(?:field|input|control)\b/i;
const OPEN_ENDED_PROSE_PATTERN = /\b(?:(?:what|which|why|how)\b|tell(?:[\s_-]+us)?\b|describe\b|explain\b|elaborate\b|detail\b|outline\b|share\b|discuss\b|summari[sz]e\b|provide\b|highlight\b|give(?:[\s_-]+us)?[\s_-]+an?[\s_-]+example\b|walk[\s_-]+us[\s_-]+through\b|anything[\s_-]+else\b|additional[\s_-]+(?:information|context|comments?)\b|comments?\b|message[\s_-]+to[\s_-]+(?:the[\s_-]+)?hiring[\s_-]+(?:manager|team)\b|most[\s_-]+proud\b)/i;
const NON_APPLICATION_WIDGET_PATTERN = /\b(?:site[\s_-]*search|job[\s_-]*search|search[\s_-]*(?:box|field|input|query)|filter[\s_-]*(?:box|field|input|query)|navigation[\s_-]*(?:box|field|input)|newsletter[\s_-]*(?:signup|email)|chat[\s_-]*(?:box|input|message))\b/i;
const CUSTOM_WIDGET_ROLES = new Set(['combobox', 'listbox', 'textbox']);
const DISCOVERY_CONTEXT_KINDS = new Set([
  'cross_origin_iframe',
  'closed_shadow_root',
  'custom_aria_widget',
  'account_gate'
]);

const F0_AUTOCOMPLETE_TOKENS = new Set([
  'cc-additional-name',
  'cc-csc',
  'cc-exp',
  'cc-exp-month',
  'cc-exp-year',
  'cc-family-name',
  'cc-given-name',
  'cc-name',
  'cc-number',
  'current-password',
  'new-password',
  'one-time-code',
  'transaction-amount',
  'transaction-currency',
  'username'
]);
const F1_AUTOCOMPLETE_TOKENS = new Set([
  'additional-name',
  'address-level1',
  'address-level2',
  'address-level3',
  'address-level4',
  'address-line1',
  'address-line2',
  'address-line3',
  'country',
  'country-name',
  'email',
  'family-name',
  'given-name',
  'honorific-prefix',
  'honorific-suffix',
  'name',
  'nickname',
  'postal-code',
  'street-address',
  'tel',
  'tel-area-code',
  'tel-country-code',
  'tel-extension',
  'tel-local',
  'tel-local-prefix',
  'tel-local-suffix',
  'tel-national',
  'url'
]);
const F2_AUTOCOMPLETE_TOKENS = new Set(['organization', 'organization-title']);
const F3_AUTOCOMPLETE_TOKENS = new Set(['bday', 'bday-day', 'bday-month', 'bday-year', 'sex']);

const TEXT_INPUT_TYPES = new Set([
  'date',
  'datetime-local',
  'email',
  'month',
  'number',
  'search',
  'tel',
  'text',
  'time',
  'url',
  'week'
]);
const F0_INPUT_TYPES = new Set(['file', 'hidden', 'password']);
const F4_INPUT_TYPES = new Set(['button', 'image', 'reset', 'submit']);

const isPlainObject = (value) => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value)
  && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

const boundedText = (value, maxLength) => String(value || '')
  .replace(/[\u0000-\u001f\u007f]+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim()
  .slice(0, maxLength);

export const sanitizeLabelText = (value, maxLength = MAX_LABEL_LENGTH) => boundedText(value, maxLength * 2)
  .replace(/\b(?:https?:\/\/|www\.)[^\s|]+/giu, '[redacted]')
  .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu, '[redacted]')
  .replace(/\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/giu, '[redacted]')
  .replace(/\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b/gu, '[redacted]')
  .replace(/\b(?:\d{1,2}[\/-]){2}\d{2,4}\b/gu, '[redacted]')
  .replace(/\b\+?\d[\d().\s-]{7,}\d\b/gu, '[redacted]')
  .replace(/\b\d{6,}\b/gu, '[redacted]')
  .replace(/\b(?=[A-Z0-9_-]{12,}\b)(?=[A-Z0-9_-]*[A-Z])(?=[A-Z0-9_-]*\d)[A-Z0-9_-]+\b/giu, '[redacted]')
  .replace(/(?:\[redacted\][\s|]*){2,}/gu, '[redacted] ')
  .replace(/[\u0000-\u001f\u007f]+/gu, ' ')
  .replace(/\s+/gu, ' ')
  .trim()
  .slice(0, maxLength);

const normalizeTokenText = (value) => boundedText(value, 2_000).toLowerCase();

const humanizeIdentifier = (value) => boundedText(String(value || '')
  .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
  .replace(/[-_.:[\]]+/g, ' '), MAX_LABEL_LENGTH);

const safeAttribute = (element, name) => boundedText(element?.getAttribute?.(name) || '', MAX_LABEL_LENGTH);

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
const safeMatches = (node, selector) => {
  try {
    return Boolean(node?.matches?.(selector));
  } catch {
    return false;
  }
};

const isOverlayNodeOrDescendant = (node) => {
  let current = node;
  for (let depth = 0; current && depth < 12; depth += 1) {
    if (safeAttribute(current, 'id') === OVERLAY_HOST_ID
      || safeAttribute(current, 'data-job-application-copilot-overlay') === 'true') {
      return true;
    }
    try {
      if (current.closest?.(`#${OVERLAY_HOST_ID}`)) return true;
    } catch {}
    let host = null;
    try {
      host = current.getRootNode?.()?.host || null;
    } catch {}
    if (!host || host === current) break;
    current = host;
  }
  return false;
};

const getOpenShadowRoot = (host) => {
  if (!host || isOverlayNodeOrDescendant(host)) return null;
  try {
    // A missing public root does not prove a component uses closed Shadow DOM.
    // Closed-root diagnostics must come from an adapter's independent structural signal.
    const root = host.shadowRoot || null;
    if (!root || root.mode === 'closed') return null;
    return root;
  } catch {
    return null;
  }
};

const shadowRootDiscoveryPriority = (shadowRoot) => {
  if (safeQuery(shadowRoot, STRUCTURAL_FIELD_SELECTOR)) return 3;
  const host = shadowRoot?.host;
  const marker = boundedText([
    host?.tagName,
    host?.id,
    host?.getAttribute?.('name'),
    host?.getAttribute?.('role'),
    host?.getAttribute?.('data-testid'),
    host?.getAttribute?.('data-automation-id')
  ].filter(Boolean).join(' '), 240);
  return /(?:application|question|form|field|input|textarea|select|phone|autocomplete|combobox|listbox)/iu.test(marker)
    ? 2
    : 1;
};

const collectOpenShadowRoots = (root) => {
  const roots = [];
  const pending = [root];
  const visitedContainers = new Set();
  const visitedRoots = new Set();
  while (pending.length && roots.length < MAX_SHADOW_ROOTS) {
    const container = pending.shift();
    if (!container || visitedContainers.has(container)) continue;
    visitedContainers.add(container);
    const elements = [
      ...(container?.tagName ? [container] : []),
      ...safeQueryAll(container, '*').slice(0, MAX_DISCOVERY_ELEMENTS)
    ];
    const discoveredRoots = [];
    for (const element of elements) {
      if (isOverlayNodeOrDescendant(element)) continue;
      const shadowRoot = getOpenShadowRoot(element);
      if (!shadowRoot || visitedRoots.has(shadowRoot)) continue;
      discoveredRoots.push(shadowRoot);
    }
    discoveredRoots.sort((left, right) => shadowRootDiscoveryPriority(right) - shadowRootDiscoveryPriority(left));
    for (const shadowRoot of discoveredRoots) {
      if (visitedRoots.has(shadowRoot)) continue;
      visitedRoots.add(shadowRoot);
      roots.push(shadowRoot);
      pending.push(shadowRoot);
      if (roots.length >= MAX_SHADOW_ROOTS) break;
    }
  }
  return roots;
};

const collectTreeElements = (root) => {
  const elements = [];
  const seen = new Set();
  const addElements = (container) => {
    const candidates = [
      ...(container?.tagName ? [container] : []),
      ...safeQueryAll(container, '*').slice(0, MAX_DISCOVERY_ELEMENTS)
    ];
    for (const element of candidates) {
      if (!element || seen.has(element) || isOverlayNodeOrDescendant(element)) continue;
      seen.add(element);
      elements.push(element);
      if (elements.length >= MAX_DISCOVERY_ELEMENTS) break;
    }
  };
  addElements(root);
  if (elements.length < MAX_DISCOVERY_ELEMENTS) collectOpenShadowRoots(root).forEach(addElements);
  return elements.slice(0, MAX_DISCOVERY_ELEMENTS);
};

const collectShadowCandidates = (root) => {
  const candidates = [];
  const seen = new Set();
  collectOpenShadowRoots(root).forEach((shadowRoot) => {
    safeQueryAll(shadowRoot, STRUCTURAL_FIELD_SELECTOR).forEach((element) => {
      if (!element || seen.has(element) || isOverlayNodeOrDescendant(element)) return;
      seen.add(element);
      candidates.push(element);
    });
  });
  return candidates;
};

const getInputType = (element) => {
  const tagName = String(element?.tagName || '').toLowerCase();
  const role = safeAttribute(element, 'role').toLowerCase();
  if (tagName === 'textarea') return 'textarea';
  if (tagName === 'select') return element?.multiple ? 'select-multiple' : 'select-one';
  if (element?.isContentEditable || safeAttribute(element, 'contenteditable')) return 'contenteditable';
  if (tagName !== 'input' && ['combobox', 'textbox'].includes(role)) return 'text';
  if (tagName !== 'input') return 'unsupported';
  return String(element?.type || safeAttribute(element, 'type') || 'text').toLowerCase();
};

const labelTextFromNodes = (nodes) => sanitizeLabelText(Array.from(nodes || [])
  .map((node) => sanitizeLabelText(node?.textContent || '', MAX_LABEL_LENGTH))
  .filter(Boolean)
  .join(' | '), MAX_LABEL_LENGTH);

const getAriaLabelledText = (element, doc) => {
  const ids = safeAttribute(element, 'aria-labelledby').split(/\s+/).filter(Boolean).slice(0, 4);
  let localRoot = null;
  try {
    localRoot = element?.getRootNode?.() || null;
  } catch {}
  return labelTextFromNodes(ids.map((id) => {
    try {
      return localRoot?.getElementById?.(id) || doc?.getElementById?.(id) || null;
    } catch {
      return doc?.getElementById?.(id) || null;
    }
  }).filter(Boolean));
};

const getElementLabel = (element, doc, adapter) => {
  const directLabels = labelTextFromNodes(element?.labels || []);
  const ariaLabel = sanitizeLabelText(safeAttribute(element, 'aria-label'));
  const ariaLabelled = getAriaLabelledText(element, doc);
  let supplemental = '';
  try {
    supplemental = sanitizeLabelText(adapter?.getSupplementalLabel?.(element) || '');
  } catch {}
  const identifier = sanitizeLabelText(humanizeIdentifier(
    element?.name || element?.id || safeAttribute(element, 'data-field-name')
  ));
  const boundedDirect = supplemental
    && directLabels.startsWith(supplemental)
    && directLabels.length >= supplemental.length + 24
    ? supplemental
    : directLabels;
  return boundedDirect || ariaLabel || ariaLabelled || supplemental || identifier || 'Unlabeled field';
};

const getOptionLabel = (element, doc, adapter) => {
  const direct = labelTextFromNodes(element?.labels || []);
  const ariaLabel = sanitizeLabelText(safeAttribute(element, 'aria-label'));
  const ariaLabelled = getAriaLabelledText(element, doc);
  const title = sanitizeLabelText(safeAttribute(element, 'title'));
  return direct || ariaLabel || ariaLabelled || title || getElementLabel(element, doc, adapter);
};

const isNodeRendered = (node, doc) => {
  if (!node || node.isConnected === false) return false;
  if (node.hidden || safeAttribute(node, 'aria-hidden').toLowerCase() === 'true') return false;
  const view = node?.ownerDocument?.defaultView || doc?.defaultView;
  try {
    const style = view?.getComputedStyle?.(node);
    if (style && (style.display === 'none'
      || style.visibility === 'hidden'
      || style.visibility === 'collapse'
      || style.contentVisibility === 'hidden'
      || Number.parseFloat(style.opacity) === 0)) {
      return false;
    }
  } catch {}
  const hasActionableArea = (rect) => {
    if (!rect) return false;
    const width = Number.isFinite(Number(rect.width))
      ? Number(rect.width)
      : Math.abs(Number(rect.right) - Number(rect.left));
    const height = Number.isFinite(Number(rect.height))
      ? Number(rect.height)
      : Math.abs(Number(rect.bottom) - Number(rect.top));
    return width >= MIN_ACTIONABLE_DIMENSION && height >= MIN_ACTIONABLE_DIMENSION;
  };
  try {
    const rect = node.getBoundingClientRect?.();
    if (hasActionableArea(rect)) return true;
    const rects = node.getClientRects?.();
    if (Array.from(rects || []).some(hasActionableArea)) return true;
  } catch {}
  return typeof node.getBoundingClientRect !== 'function';
};

const isFieldVisible = (element, doc, type) => {
  if (isNodeRendered(element, doc)) return true;
  if (type !== 'radio' && type !== 'checkbox') return false;
  return Array.from(element?.labels || []).some((label) => isNodeRendered(label, doc));
};

const getStructuralPath = (element) => {
  const parts = [];
  let node = element;
  for (let depth = 0; node && depth < 10; depth += 1) {
    const tagName = String(node.tagName || 'node').toLowerCase();
    const id = safeAttribute(node, 'id');
    const name = safeAttribute(node, 'name');
    let position = 1;
    let parent = node.parentElement;
    if (parent?.children) {
      const siblings = Array.from(parent.children).filter((candidate) => candidate?.tagName === node.tagName);
      const index = siblings.indexOf(node);
      if (index >= 0) position = index + 1;
    }
    parts.push(`${tagName}:${id}:${name}:${position}`);
    if (tagName === 'form' || tagName === 'body') break;
    if (!parent) {
      try {
        parent = node.getRootNode?.()?.host || null;
        if (parent) parts.push('#shadow-root');
      } catch {}
    }
    node = parent;
  }
  return parts.join('>');
};

const getShadowHostChain = (element) => {
  const parts = [];
  let current = element;
  for (let depth = 0; current && depth < 6; depth += 1) {
    let host = null;
    try {
      host = current.getRootNode?.()?.host || null;
    } catch {}
    if (!host || isOverlayNodeOrDescendant(host)) break;
    parts.push([
      String(host.tagName || 'host').toLowerCase(),
      safeAttribute(host, 'id'),
      safeAttribute(host, 'name')
    ].join(':'));
    current = host;
  }
  return parts.join('>');
};

const fnv1a = (value, seed = 0x811c9dc5) => {
  let hash = seed >>> 0;
  const input = String(value || '');
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
};

export const createFieldFingerprint = (parts) => {
  const material = Array.isArray(parts) ? parts.join('|') : String(parts || '');
  return `${fnv1a(material)}${fnv1a(material, 0x9e3779b1)}`;
};

export const classifyFieldMetadata = (metadata = {}) => {
  const type = String(metadata.type || '').toLowerCase();
  const autocompleteTokens = String(metadata.autocomplete || '').toLowerCase().split(/\s+/).filter(Boolean);
  const haystack = normalizeTokenText([
    metadata.label,
    metadata.name,
    metadata.id,
    metadata.role,
    ...(Array.isArray(metadata.options) ? metadata.options : [])
  ].filter(Boolean).join(' '));
  const guardHaystack = normalizeTokenText([
    haystack,
    metadata.placeholder,
    metadata.nearbyText
  ].filter(Boolean).join(' '));
  const widgetIdentity = normalizeTokenText([
    metadata.name,
    metadata.id,
    metadata.role,
    metadata.placeholder
  ].filter(Boolean).join(' '));
  const role = normalizeTokenText(metadata.role);
  const navigationWidget = role === 'searchbox' || NON_APPLICATION_WIDGET_PATTERN.test(widgetIdentity);

  if (metadata.captcha || metadata.hidden || metadata.disabled || F0_INPUT_TYPES.has(type)
    || autocompleteTokens.some((token) => F0_AUTOCOMPLETE_TOKENS.has(token))
    || F0_PATTERN.test(haystack) || BOT_TRAP_PATTERN.test(guardHaystack)
    || ACCOUNT_CREDENTIAL_PATTERN.test(guardHaystack) || navigationWidget) {
    return RISK_CLASSES.F0_EXCLUDED;
  }
  if (F4_INPUT_TYPES.has(type) || F4_PATTERN.test(haystack)) return RISK_CLASSES.F4_CONSENT;
  if (CLEAR_VISA_SPONSORSHIP_PATTERN.test(haystack)
    && !F3_PATTERN.test(haystack) && !SENSITIVE_IDENTITY_PATTERN.test(haystack)
    && !MANUAL_WORK_IDENTITY_PATTERN.test(haystack) && !MANUAL_IMMIGRATION_DETAIL_PATTERN.test(haystack)
    && !VISA_STATUS_OR_TYPE_PATTERN.test(haystack)) return RISK_CLASSES.F2_REVIEW;
  if (autocompleteTokens.some((token) => F3_AUTOCOMPLETE_TOKENS.has(token))
    || F3_PATTERN.test(haystack) || SENSITIVE_IDENTITY_PATTERN.test(haystack)
    || MANUAL_WORK_IDENTITY_PATTERN.test(haystack)
    || MANUAL_IMMIGRATION_DETAIL_PATTERN.test(haystack)
    || VISA_STATUS_OR_TYPE_PATTERN.test(haystack)
    || (VISA_DETAIL_PATTERN.test(haystack) && !VISA_SPONSORSHIP_PATTERN.test(haystack))) {
    return RISK_CLASSES.F3_SENSITIVE;
  }
  if (THIRD_PARTY_CONTACT_PATTERN.test(haystack) || THIRD_PARTY_PROFILE_PATTERN.test(haystack)) return RISK_CLASSES.F0_EXCLUDED;
  if (autocompleteTokens.some((token) => F2_AUTOCOMPLETE_TOKENS.has(token))
    || F2_PATTERN.test(haystack) || COMMUNICATION_PREFERENCE_PATTERN.test(haystack) || REVIEW_FACT_PATTERN.test(haystack)) {
    return RISK_CLASSES.F2_REVIEW;
  }
  if (autocompleteTokens.some((token) => F1_AUTOCOMPLETE_TOKENS.has(token))
    || (['text', 'url'].includes(type) && CUSTOM_LINK_FIELD_PATTERN.test(haystack) && !UNLABELED_FIELD_PATTERN.test(haystack))
    || (type === 'url' && !UNLABELED_FIELD_PATTERN.test(haystack)) || F1_PATTERN.test(haystack)) {
    return RISK_CLASSES.F1_VERIFIED;
  }
  if (['textarea', 'contenteditable'].includes(type)
    && !UNLABELED_FIELD_PATTERN.test(haystack)
    && OPEN_ENDED_PROSE_PATTERN.test(haystack)) {
    return RISK_CLASSES.F2_REVIEW;
  }
  return RISK_CLASSES.F0_EXCLUDED;
};

const isCustomWidgetElement = (element) => {
  const role = safeAttribute(element, 'role').toLowerCase();
  const tagName = String(element?.tagName || '').toLowerCase();
  const textLikeControl = tagName === 'input' || Boolean(element?.isContentEditable)
    || safeAttribute(element, 'contenteditable') === 'true'
    || safeAttribute(element, 'contenteditable') === 'plaintext-only';
  if (CUSTOM_WIDGET_ROLES.has(role)) return true;
  if (textLikeControl && safeAttribute(element, 'aria-autocomplete')) return true;
  if (textLikeControl && ['grid', 'listbox', 'tree'].includes(safeAttribute(element, 'aria-haspopup').toLowerCase())) return true;
  const frameworkAttributes = [
    'data-headlessui-state',
    'data-radix-collection-item',
    'data-reach-combobox-input',
    'data-select2-id'
  ];
  return frameworkAttributes.some((name) => {
    try {
      return Boolean(element?.hasAttribute?.(name));
    } catch {
      return false;
    }
  });
};

const collectSelectOptions = (element) => Array.from(element?.options || [])
  .slice(0, MAX_OPTIONS)
  .map((option) => ({
    label: sanitizeLabelText(option?.label || option?.textContent || '', MAX_OPTION_LENGTH),
    node: option
  }))
  .filter((option) => option.label);

const buttonChoiceKind = (value) => {
  const normalized = normalizeTokenText(value);
  if (/^yes(?:\b|$)/u.test(normalized)) return 'yes';
  if (/^no(?:\b|$)/u.test(normalized)) return 'no';
  return '';
};

const boundedButtonLabel = (button) => sanitizeLabelText(
  safeAttribute(button, 'aria-label') || button?.textContent || '',
  MAX_OPTION_LENGTH
);

const collectBoundedButtonGroups = (root, doc, adapter) => {
  if (!root) return [];
  const grouped = new Map();
  safeQueryAll(root, STRUCTURAL_OPTION_SELECTOR).forEach((button) => {
    const tagName = String(button?.tagName || '').toLowerCase();
    const role = safeAttribute(button, 'role').toLowerCase();
    const type = safeAttribute(button, 'type').toLowerCase();
    if ((tagName !== 'button' && role !== 'radio') || ['submit', 'reset'].includes(type)) return;
    const label = boundedButtonLabel(button);
    if (!buttonChoiceKind(label) || !isFieldVisible(button, doc, 'button-group')) return;
    let container = null;
    try {
      container = adapter?.getFieldContainer?.(button)
        || button?.closest?.('fieldset,[role="radiogroup"],[role="group"],[data-question-id],.application-question,.question')
        || button?.parentElement
        || null;
    } catch {
      container = button?.parentElement || null;
    }
    if (!container || container === root && root === doc) return;
    if (!grouped.has(container)) grouped.set(container, []);
    grouped.get(container).push(button);
  });

  const groups = [];
  grouped.forEach((buttons, container) => {
    const uniqueButtons = Array.from(new Set(buttons)).slice(0, 4);
    const choices = uniqueButtons.map((button) => buttonChoiceKind(boundedButtonLabel(button)));
    if (uniqueButtons.length !== 2 || new Set(choices).size !== 2
      || !choices.includes('yes') || !choices.includes('no')) return;
    let supplemental = '';
    try {
      supplemental = sanitizeLabelText(adapter?.getSupplementalLabel?.(uniqueButtons[0]) || '');
    } catch {}
    const label = supplemental
      || sanitizeLabelText(safeAttribute(container, 'aria-label'))
      || getAriaLabelledText(container, doc)
      || labelTextFromNodes(safeQueryAll(container, ':scope > legend,:scope > label'));
    if (!label || ['yes', 'no', 'yes | no'].includes(normalizeTokenText(label))) return;
    groups.push({
      adapter,
      elements: uniqueButtons,
      label,
      sourceIndex: MAX_CANDIDATES + groups.length,
      type: 'button-group'
    });
  });
  return groups;
};

const buildGroups = (candidates) => {
  const groups = [];
  const keyed = new Map();
  candidates.forEach((element, index) => {
    const type = getInputType(element);
    const name = safeAttribute(element, 'name');
    if ((type === 'radio' || type === 'checkbox') && name) {
      const formKey = safeAttribute(element?.form, 'id') || getStructuralPath(element?.form);
      const key = `${type}|${formKey}|${name}`;
      if (!keyed.has(key)) {
        const group = { elements: [], sourceIndex: index, type };
        keyed.set(key, group);
        groups.push(group);
      }
      keyed.get(key).elements.push(element);
      return;
    }
    groups.push({ elements: [element], sourceIndex: index, type });
  });
  return groups.sort((left, right) => left.sourceIndex - right.sourceIndex);
};

const getGroupLabel = (elements, doc, adapter, type, explicitLabel = '') => {
  const first = elements[0];
  if (explicitLabel) return sanitizeLabelText(explicitLabel);
  if ((type === 'radio' || type === 'checkbox') && elements.length > 1) {
    let supplemental = '';
    try {
      supplemental = sanitizeLabelText(adapter?.getSupplementalLabel?.(first) || '');
    } catch {}
    const fieldset = first?.closest?.('fieldset');
    const legend = sanitizeLabelText(safeQuery(fieldset, ':scope > legend')?.textContent || '');
    const ariaLabelled = getAriaLabelledText(first, doc);
    return legend || ariaLabelled || supplemental || sanitizeLabelText(humanizeIdentifier(first?.name)) || 'Option group';
  }
  return getElementLabel(first, doc, adapter);
};

const getGroupOptions = (elements, doc, adapter, type) => {
  if (type === 'select-one' || type === 'select-multiple') return collectSelectOptions(elements[0]);
  if (type === 'button-group') {
    return elements.slice(0, MAX_OPTIONS).map((element) => ({
      label: boundedButtonLabel(element),
      node: element
    })).filter((option) => option.label);
  }
  if (type !== 'radio' && type !== 'checkbox') return [];
  return elements.slice(0, MAX_OPTIONS).map((element) => ({
    label: sanitizeLabelText(getOptionLabel(element, doc, adapter), MAX_OPTION_LENGTH),
    node: element
  })).filter((option) => option.label);
};

const isSupportedType = (type) => TEXT_INPUT_TYPES.has(type)
  || ['button-group', 'checkbox', 'contenteditable', 'radio', 'select-multiple', 'select-one', 'textarea'].includes(type)
  || F0_INPUT_TYPES.has(type)
  || F4_INPUT_TYPES.has(type);

const rawCaptchaAttribute = (node, name, maxLength = 2_048) => {
  try {
    return String(node?.getAttribute?.(name) || '').slice(0, maxLength);
  } catch {
    return '';
  }
};

const isPassiveCaptchaMarker = (node) => {
  const tagName = String(node?.tagName || '').toLowerCase();
  if (tagName === 'script') return true;
  const identifier = `${rawCaptchaAttribute(node, 'name')} ${rawCaptchaAttribute(node, 'id')}`.toLowerCase();
  if ((tagName === 'input' || tagName === 'textarea')
    && /(?:g-recaptcha-response|h-captcha-response|cf-turnstile-response)/u.test(identifier)) {
    return true;
  }
  try {
    if (node?.closest?.('.grecaptcha-badge')) return true;
  } catch {}
  if (rawCaptchaAttribute(node, 'data-size').toLowerCase() === 'invisible') return true;
  const src = rawCaptchaAttribute(node, 'src');
  if (/[?&#]size=invisible(?:[&#]|$)/iu.test(src)) return true;
  if (tagName !== 'iframe') {
    const appearance = rawCaptchaAttribute(node, 'data-appearance').toLowerCase();
    const execution = rawCaptchaAttribute(node, 'data-execution').toLowerCase();
    if (appearance === 'interaction-only' || execution === 'execute') return true;
  }
  return false;
};

export const detectCaptchaPresence = (doc) => {
  const nodes = [];
  const seen = new Set();
  const documents = [doc, ...collectAccessibleFrameDocuments(doc)
    .filter((record) => record.visible)
    .map((record) => record.doc)];
  documents.forEach((currentDoc) => {
    [currentDoc, ...collectOpenShadowRoots(currentDoc)].forEach((root) => {
      safeQueryAll(root, CAPTCHA_INTERACTION_SELECTOR).forEach((node) => {
        if (!node || seen.has(node) || isOverlayNodeOrDescendant(node)) return;
        seen.add(node);
        nodes.push(node);
      });
    });
  });
  return nodes.some((node) => !isPassiveCaptchaMarker(node) && isNodeRendered(node, doc));
};

const readAdapterDiscoveryContexts = (adapter, doc) => {
  try {
    const contexts = adapter?.getDiscoveryContexts?.(doc);
    return Array.isArray(contexts) ? contexts.slice(0, MAX_DISCOVERY_CONTEXTS * 2) : [];
  } catch {
    return [];
  }
};

const resolveApplicationRoot = (adapter, doc, adapterContexts) => {
  if (typeof adapter?.getApplicationRoot === 'function') {
    try {
      const root = adapter.getApplicationRoot(doc);
      return root?.querySelectorAll ? root : null;
    } catch {
      return null;
    }
  }
  const contextualRoot = adapterContexts.find((context) => context?.kind === 'application' && context?.root?.querySelectorAll)?.root;
  return contextualRoot || doc;
};

const readAccessibleFrameDocument = (frame) => {
  try {
    const frameDoc = frame?.contentDocument || frame?.contentWindow?.document || null;
    return frameDoc?.querySelectorAll ? frameDoc : null;
  } catch {
    return null;
  }
};

const collectAccessibleFrameDocuments = (doc) => {
  const records = [];
  const seenDocuments = new Set([doc]);
  const pending = [{ depth: 0, doc, visible: true }];
  while (pending.length && records.length < MAX_FRAME_DOCUMENTS) {
    const current = pending.shift();
    const roots = [current.doc, ...collectOpenShadowRoots(current.doc)];
    const seenFrames = new Set();
    roots.forEach((root) => {
      safeQueryAll(root, 'iframe').forEach((frame) => {
        if (!frame || seenFrames.has(frame) || isOverlayNodeOrDescendant(frame)) return;
        seenFrames.add(frame);
        const frameDoc = readAccessibleFrameDocument(frame);
        if (!frameDoc || seenDocuments.has(frameDoc)) return;
        seenDocuments.add(frameDoc);
        const record = {
          depth: current.depth + 1,
          doc: frameDoc,
          frame,
          parentDoc: current.doc,
          visible: current.visible && isNodeRendered(frame, current.doc)
        };
        records.push(record);
        if (record.depth < 4 && records.length < MAX_FRAME_DOCUMENTS) pending.push(record);
      });
    });
  }
  return records;
};

const frameDocumentUrl = (record) => {
  try {
    return String(record?.doc?.location?.href || rawCaptchaAttribute(record?.frame, 'src'));
  } catch {
    return rawCaptchaAttribute(record?.frame, 'src');
  }
};

const collectFramedSurfaces = (doc) => collectAccessibleFrameDocuments(doc)
  .filter((record) => record.visible)
  .map((record) => {
    const url = frameDocumentUrl(record);
    const adapter = selectAdapter({ doc: record.doc, url });
    const contexts = readAdapterDiscoveryContexts(adapter, record.doc);
    const accountGate = contexts.some((context) => context?.kind === 'account_gate');
    const root = accountGate ? null : resolveApplicationRoot(adapter, record.doc, contexts);
    let candidates = [];
    if (root) {
      try {
        candidates = Array.from(adapter.collectCandidates(record.doc) || []);
      } catch {}
      candidates.push(...collectShadowCandidates(root));
    }
    return {
      ...record,
      adapter,
      contexts,
      root,
      candidates
    };
  });

const boundedDiscoveryCount = (value, fallback = 0) => {
  const count = Number(value);
  if (!Number.isSafeInteger(count) || count < 0) return Math.min(MAX_DISCOVERY_CONTEXT_COUNT, fallback);
  return Math.min(MAX_DISCOVERY_CONTEXT_COUNT, count);
};

const isCrossOriginFrame = (frame, doc) => {
  const src = rawCaptchaAttribute(frame, 'src');
  if (!src || /^(?:about:blank|javascript:)/iu.test(src)
    || /(?:recaptcha|hcaptcha|challenges\.cloudflare\.com|turnstile)/iu.test(src)) return false;
  try {
    const base = new URL(String(doc?.location?.href || 'https://invalid.local/'));
    const target = new URL(src, base);
    return /^https?:$/u.test(target.protocol) && target.origin !== base.origin;
  } catch {
    return false;
  }
};

const APPLICATION_FRAME_MARKER_RE = /(?:^|\s)(?:application|apply|applicant|candidate)(?:\s|$)/iu;

const frameApplicationMarker = (frame) => [
  rawCaptchaAttribute(frame, 'src'),
  rawCaptchaAttribute(frame, 'title'),
  rawCaptchaAttribute(frame, 'name'),
  rawCaptchaAttribute(frame, 'id'),
  rawCaptchaAttribute(frame, 'class'),
  rawCaptchaAttribute(frame, 'aria-label'),
  rawCaptchaAttribute(frame, 'data-testid'),
  rawCaptchaAttribute(frame, 'data-automation-id')
].join(' ')
  .replace(/([a-z0-9])([A-Z])/gu, '$1 $2')
  .replace(/[^a-z0-9]+/giu, ' ')
  .toLowerCase();

const isLikelyApplicationFrame = (frame) => {
  const marker = frameApplicationMarker(frame);
  return APPLICATION_FRAME_MARKER_RE.test(marker)
    || /(?:^|\s)job\s+(?:app|application|apply)(?:\s|$)/iu.test(marker);
};

const collectDocumentFrames = (doc) => {
  const frames = [];
  const seen = new Set();
  const documents = [doc, ...collectAccessibleFrameDocuments(doc).map((record) => record.doc)];
  documents.forEach((currentDoc) => {
    [currentDoc, ...collectOpenShadowRoots(currentDoc)].forEach((root) => {
      safeQueryAll(root, 'iframe').forEach((frame) => {
        const tagName = String(frame?.tagName || '').toLowerCase();
        if (tagName !== 'iframe' || seen.has(frame) || isOverlayNodeOrDescendant(frame)) return;
        seen.add(frame);
        frames.push(frame);
      });
    });
  });
  return frames.slice(0, MAX_DISCOVERY_CONTEXT_COUNT);
};

const buildDiscoverySummary = ({
  adapter,
  doc,
  applicationRoot,
  applicationRoots = [],
  adapterContexts,
  fields,
  exclusionCounts,
  truncated
}) => {
  const mergedContexts = new Map();
  const addContext = ({ kind, count, status } = {}) => {
    const normalizedKind = boundedText(kind, 40);
    if (!DISCOVERY_CONTEXT_KINDS.has(normalizedKind)) return;
    const normalizedStatus = ['manual', 'unsupported'].includes(status)
      ? status
      : ['cross_origin_iframe', 'closed_shadow_root'].includes(normalizedKind) ? 'unsupported' : 'manual';
    const normalizedCount = boundedDiscoveryCount(count);
    if (!normalizedCount) return;
    const key = `${normalizedKind}:${normalizedStatus}`;
    const previous = mergedContexts.get(key) || 0;
    mergedContexts.set(key, boundedDiscoveryCount(previous + normalizedCount));
  };

  adapterContexts.forEach((context) => {
    if (!context || context.kind === 'application') return;
    addContext({
      kind: context.kind,
      count: context.count ?? (context.root ? 1 : 0),
      status: context.status
    });
  });

  const roots = Array.from(new Set([
    ...(applicationRoot ? [applicationRoot] : []),
    ...applicationRoots
  ])).filter((root) => root?.querySelectorAll);
  const elements = roots.flatMap((root) => collectTreeElements(root));
  const applicationElements = new Set(elements);
  const crossOriginFrames = collectDocumentFrames(doc).filter((frame) => {
    const belongsToApplicationRoot = roots.includes(doc) || applicationElements.has(frame);
    return isCrossOriginFrame(frame, doc)
      && (belongsToApplicationRoot || isLikelyApplicationFrame(frame));
  }).length;
  const customWidgets = Math.max(
    fields.filter((field) => field.fillMode === 'copy_only').length,
    elements.filter((element) => {
      const role = safeAttribute(element, 'role').toLowerCase();
      return role === 'searchbox' || isCustomWidgetElement(element);
    }).length
  );
  const accountGates = elements.filter((element) => {
    const type = getInputType(element);
    const autocomplete = safeAttribute(element, 'autocomplete').toLowerCase().split(/\s+/).filter(Boolean);
    return type === 'password' || autocomplete.some((token) => ['current-password', 'new-password'].includes(token));
  }).length;
  addContext({ kind: 'cross_origin_iframe', count: crossOriginFrames, status: 'unsupported' });
  addContext({ kind: 'custom_aria_widget', count: customWidgets, status: 'manual' });
  addContext({ kind: 'account_gate', count: accountGates, status: 'manual' });

  const contexts = Array.from(mergedContexts.entries())
    .slice(0, MAX_DISCOVERY_CONTEXTS)
    .map(([key, count]) => {
      const separator = key.lastIndexOf(':');
      return Object.freeze({
        kind: key.slice(0, separator),
        count,
        status: key.slice(separator + 1)
      });
    });
  const unsupportedCount = boundedDiscoveryCount(contexts
    .filter((context) => context.status === 'unsupported')
    .reduce((sum, context) => sum + context.count, 0));
  const adapterId = boundedText(adapter?.id || 'generic', 64);
  const mode = !roots.length || unsupportedCount || (adapterId === 'generic' && !fields.length && contexts.length)
    ? 'limited'
    : adapterId === 'generic' ? 'free_format' : 'standard';
  return Object.freeze({
    mode,
    recognizedCount: Math.min(MAX_FIELDS, fields.length),
    unsupportedCount,
    exclusionCounts: Object.freeze({
      [RISK_CLASSES.F0_EXCLUDED]: boundedDiscoveryCount(exclusionCounts[RISK_CLASSES.F0_EXCLUDED]),
      [RISK_CLASSES.F3_SENSITIVE]: boundedDiscoveryCount(exclusionCounts[RISK_CLASSES.F3_SENSITIVE]),
      [RISK_CLASSES.F4_CONSENT]: boundedDiscoveryCount(exclusionCounts[RISK_CLASSES.F4_CONSENT])
    }),
    contexts: Object.freeze(contexts),
    truncated: Boolean(truncated || mergedContexts.size > MAX_DISCOVERY_CONTEXTS)
  });
};

const createEmptyExclusionCounts = () => ({
  [RISK_CLASSES.F0_EXCLUDED]: 0,
  [RISK_CLASSES.F3_SENSITIVE]: 0,
  [RISK_CLASSES.F4_CONSENT]: 0
});

const sanitizeJob = (job = {}) => ({
  company: boundedText(job.company, 240),
  title: boundedText(job.title, 240),
  jobUrl: boundedText(job.jobUrl, 2_048),
  location: boundedText(job.location, 240),
  source: boundedText(job.source, 120),
  description: boundedText(job.description, MAX_JOB_DESCRIPTION_LENGTH)
});

export const scanDocument = ({
  doc,
  url,
  adapter: suppliedAdapter,
  maxFields = MAX_FIELDS,
  domRevision = 0
} = {}) => {
  if (!doc) throw new TypeError('scanDocument requires a document.');
  const resolvedUrl = String(url || doc?.location?.href || '');
  const adapter = suppliedAdapter || selectAdapter({ doc, url: resolvedUrl });
  const captchaPresent = detectCaptchaPresence(doc);
  const mainAdapterContexts = readAdapterDiscoveryContexts(adapter, doc);
  const accountGateDeclared = mainAdapterContexts.some((context) => context?.kind === 'account_gate');
  const applicationRoot = accountGateDeclared ? null : resolveApplicationRoot(adapter, doc, mainAdapterContexts);
  const framedSurfaces = collectFramedSurfaces(doc);
  const framedApplications = framedSurfaces.filter((surface) => surface.root);
  const adapterContexts = [
    ...mainAdapterContexts,
    ...framedSurfaces.flatMap((surface) => surface.contexts.filter((context) => context?.kind !== 'application'))
  ];
  const scanAdapter = applicationRoot ? adapter : framedApplications[0]?.adapter || adapter;
  const applicationRoots = [
    ...(applicationRoot ? [applicationRoot] : []),
    ...framedApplications.map((surface) => surface.root)
  ];
  const candidateAdapters = new WeakMap();
  let adapterCandidates = [];
  if (applicationRoot) {
    try {
      adapterCandidates = Array.from(adapter.collectCandidates(doc) || []);
    } catch {}
    adapterCandidates.forEach((element) => candidateAdapters.set(element, adapter));
  }
  const mainShadowCandidates = collectShadowCandidates(applicationRoot);
  mainShadowCandidates.forEach((element) => candidateAdapters.set(element, adapter));
  const framedCandidates = framedApplications.flatMap((surface) => {
    surface.candidates.forEach((element) => candidateAdapters.set(element, surface.adapter));
    return surface.candidates;
  });
  const combinedCandidates = [
    ...adapterCandidates,
    ...mainShadowCandidates,
    ...framedCandidates
  ];
  const uniqueCandidates = [];
  const seenCandidates = new Set();
  combinedCandidates.forEach((element) => {
    if (!element || seenCandidates.has(element) || isOverlayNodeOrDescendant(element)) return;
    seenCandidates.add(element);
    uniqueCandidates.push(element);
  });
  const rawCandidates = uniqueCandidates.slice(0, MAX_CANDIDATES);
  const buttonGroups = [
    ...(applicationRoot ? collectBoundedButtonGroups(applicationRoot, doc, adapter) : []),
    ...framedApplications.flatMap((surface) => collectBoundedButtonGroups(
      surface.root,
      surface.doc,
      surface.adapter
    ))
  ];
  const groups = [
    ...buildGroups(rawCandidates),
    ...buttonGroups
  ].sort((left, right) => left.sourceIndex - right.sourceIndex);
  const fields = [];
  const records = new Map();
  const exclusionCounts = createEmptyExclusionCounts();
  const collisionCounts = new Map();
  let truncated = uniqueCandidates.length > MAX_CANDIDATES;

  for (const group of groups) {
    const elements = group.elements.filter(Boolean);
    if (!elements.length) continue;
    const first = elements[0];
    const type = group.type;
    const descriptorType = type === 'button-group' ? 'select-one' : type;
    const fieldAdapter = group.adapter || candidateAdapters.get(first) || scanAdapter;
    const fieldDoc = first?.ownerDocument || doc;
    const hidden = !isFieldVisible(first, fieldDoc, type) || first.hidden || safeAttribute(first, 'aria-hidden') === 'true';
    const disabled = elements.every((element) => Boolean(element.disabled));
    const options = getGroupOptions(elements, fieldDoc, fieldAdapter, type);
    const label = getGroupLabel(elements, fieldDoc, fieldAdapter, type, group.label);
    let nearbyText = '';
    try {
      nearbyText = fieldAdapter.getNearbyText?.(first) || '';
    } catch {}
    const metadata = {
      type: isSupportedType(type) ? descriptorType : 'hidden',
      label,
      name: safeAttribute(first, 'name'),
      id: safeAttribute(first, 'id'),
      role: safeAttribute(first, 'role'),
      nearbyText,
      placeholder: safeAttribute(first, 'placeholder'),
      autocomplete: safeAttribute(first, 'autocomplete'),
      options: options.map((option) => option.label),
      hidden,
      disabled,
      captcha: F0_PATTERN.test(normalizeTokenText(`${label} ${safeAttribute(first, 'name')} ${safeAttribute(first, 'id')}`))
    };
    const riskClass = classifyFieldMetadata(metadata);
    if (!ACTIONABLE_RISKS.has(riskClass)) {
      if (Object.hasOwn(exclusionCounts, riskClass)) exclusionCounts[riskClass] += 1;
      continue;
    }
    if (fields.length >= Math.max(1, Math.min(MAX_FIELDS, Number(maxFields) || MAX_FIELDS))) {
      truncated = true;
      break;
    }

    const hasSemanticIdentity = Boolean(metadata.name
      || metadata.id
      || (label && label !== 'Unlabeled field'));
    const fingerprint = createFieldFingerprint([
      fieldAdapter.id,
      type,
      metadata.name,
      metadata.id,
      label,
      getShadowHostChain(first),
      hasSemanticIdentity ? '' : getStructuralPath(first)
    ]);
    const collision = collisionCounts.get(fingerprint) || 0;
    collisionCounts.set(fingerprint, collision + 1);
    const fieldId = `field-${fingerprint}${collision ? `-${collision + 1}` : ''}`;
    const rawMaxLength = Number(first.maxLength);
    const maxLength = Number.isSafeInteger(rawMaxLength) && rawMaxLength >= 1
      ? Math.min(rawMaxLength, MAX_PROPOSAL_LENGTH)
      : undefined;
    const fillMode = type === 'button-group' || elements.some(isCustomWidgetElement)
      ? 'copy_only'
      : undefined;
    const descriptor = Object.freeze({
      fieldId,
      fingerprint,
      label: sanitizeLabelText(label),
      type: descriptorType,
      options: Object.freeze(options.map((option) => option.label)),
      nearbyText: sanitizeLabelText(nearbyText, MAX_NEARBY_LENGTH),
      required: elements.some((element) => Boolean(element.required) || safeAttribute(element, 'aria-required') === 'true'),
      riskClass,
      ...(fillMode ? { fillMode } : {}),
      ...(maxLength ? { maxLength } : {})
    });
    fields.push(descriptor);
    records.set(fieldId, {
      descriptor,
      elements,
      optionRecords: options,
      type,
      fillMode
    });
  }

  let job = {};
  try {
    job = adapter.extractJobMetadata(doc, resolvedUrl) || {};
  } catch {}
  const primaryFramedApplication = framedApplications[0];
  if (primaryFramedApplication) {
    let framedJob = {};
    try {
      framedJob = primaryFramedApplication.adapter.extractJobMetadata(
        primaryFramedApplication.doc,
        frameDocumentUrl(primaryFramedApplication)
      ) || {};
    } catch {}
    job = {
      ...job,
      company: job.company || framedJob.company,
      title: job.title || framedJob.title,
      location: job.location || framedJob.location,
      description: job.description || framedJob.description,
      source: applicationRoot ? job.source : framedJob.source || job.source
    };
  }
  const sanitizedJob = sanitizeJob(job);
  const normalizedPageUrl = sanitizedJob.jobUrl || boundedText(resolvedUrl.split('#')[0].split('?')[0], 2_048);
  const urlHash = createFieldFingerprint(['url', normalizedPageUrl]);
  const discovery = buildDiscoverySummary({
    adapter: scanAdapter,
    doc,
    applicationRoot,
    applicationRoots,
    adapterContexts,
    fields,
    exclusionCounts,
    truncated
  });

  return {
    result: Object.freeze({
      schemaVersion: RUNTIME_VERSION,
      pageId: `page-${urlHash}`,
      urlHash,
      domRevision: Number.isSafeInteger(domRevision) && domRevision >= 0 ? domRevision : 0,
      adapter: boundedText(scanAdapter.id || 'generic', 64),
      job: Object.freeze(sanitizedJob),
      captchaPresent,
      fields: Object.freeze(fields),
      exclusionCounts: Object.freeze(exclusionCounts),
      discovery,
      truncated
    }),
    records,
    adapter: scanAdapter
  };
};

const createSanitizedDiscoveryExport = (scanResult, fields) => {
  const source = scanResult?.discovery || {};
  const contexts = Array.from(source.contexts || [])
    .slice(0, MAX_DISCOVERY_CONTEXTS)
    .filter((context) => DISCOVERY_CONTEXT_KINDS.has(context?.kind)
      && ['manual', 'unsupported'].includes(context?.status))
    .map((context) => ({
      kind: context.kind,
      count: boundedDiscoveryCount(context.count),
      status: context.status
    }))
    .filter((context) => context.count > 0);
  const sourceExclusions = source.exclusionCounts || scanResult?.exclusionCounts || {};
  const mode = ['standard', 'free_format', 'limited'].includes(source.mode)
    ? source.mode
    : boundedText(scanResult?.adapter || 'generic', 64) === 'generic' ? 'free_format' : 'standard';
  return {
    mode,
    recognizedCount: boundedDiscoveryCount(source.recognizedCount, fields.length),
    unsupportedCount: boundedDiscoveryCount(source.unsupportedCount),
    exclusionCounts: {
      [RISK_CLASSES.F0_EXCLUDED]: boundedDiscoveryCount(sourceExclusions[RISK_CLASSES.F0_EXCLUDED]),
      [RISK_CLASSES.F3_SENSITIVE]: boundedDiscoveryCount(sourceExclusions[RISK_CLASSES.F3_SENSITIVE]),
      [RISK_CLASSES.F4_CONSENT]: boundedDiscoveryCount(sourceExclusions[RISK_CLASSES.F4_CONSENT])
    },
    contexts,
    truncated: Boolean(source.truncated ?? scanResult?.truncated)
  };
};

export const createSanitizedExport = (scanResult) => {
  const fields = Array.from(scanResult?.fields || []).slice(0, MAX_FIELDS).map((field) => ({
    fieldId: boundedText(field.fieldId, 80),
    label: boundedText(field.label, MAX_LABEL_LENGTH),
    type: boundedText(field.type, 40),
    options: Array.from(field.options || []).slice(0, MAX_OPTIONS)
      .map((option) => boundedText(option, MAX_OPTION_LENGTH)),
    riskClass: ACTIONABLE_RISKS.has(field.riskClass) ? field.riskClass : RISK_CLASSES.F2_REVIEW,
    ...(field.fillMode === 'copy_only' ? { fillMode: 'copy_only' } : {})
  }));
  return {
    schemaVersion: RUNTIME_VERSION,
    adapter: boundedText(scanResult?.adapter || 'generic', 64),
    captchaPresent: Boolean(scanResult?.captchaPresent),
    discovery: createSanitizedDiscoveryExport(scanResult, fields),
    fields
  };
};

const isValidRequestId = (value) => typeof value === 'string'
  && value.length >= 1
  && value.length <= MAX_REQUEST_ID_LENGTH
  && /^[A-Za-z0-9._:-]+$/.test(value);

const isValidFieldId = (value) => typeof value === 'string'
  && value.length <= 80
  && /^field-[a-f0-9]{16}(?:-[2-9][0-9]*)?$/.test(value);

const sanitizeAnswer = (value) => {
  if (typeof value === 'boolean') return { ok: true, value };
  if (typeof value === 'string' && value.length <= MAX_PROPOSAL_LENGTH) return { ok: true, value };
  if (Array.isArray(value) && value.length <= MAX_ARRAY_ANSWERS
    && value.every((item) => typeof item === 'string' && item.length <= MAX_OPTION_LENGTH)) {
    return { ok: true, value: value.slice() };
  }
  return { ok: false, error: 'invalid_answer' };
};

export const validateRuntimeMessage = (message) => {
  if (!isPlainObject(message)) return { ok: false, error: 'invalid_message' };
  if (message.channel !== RUNTIME_CHANNEL || message.version !== RUNTIME_VERSION) {
    return { ok: false, error: 'invalid_channel' };
  }
  if (!INBOUND_MESSAGE_TYPES.has(message.type)) return { ok: false, error: 'invalid_type' };
  if (!isValidRequestId(message.requestId)) return { ok: false, error: 'invalid_request_id' };
  const payload = message.payload === undefined ? {} : message.payload;
  if (!isPlainObject(payload)) return { ok: false, error: 'invalid_payload' };

  if (message.type === 'PAGE_PROPOSALS_UPDATE') {
    if (!Array.isArray(payload.proposals) || payload.proposals.length > MAX_PROPOSALS) {
      return { ok: false, error: 'invalid_proposals' };
    }
    const proposals = [];
    for (const proposal of payload.proposals) {
      if (!isPlainObject(proposal)) return { ok: false, error: 'invalid_proposal' };
      const fieldId = proposal.fieldId || proposal.field_id;
      if (!isValidFieldId(fieldId)
        || (proposal.fieldId && proposal.field_id && proposal.fieldId !== proposal.field_id)) {
        return { ok: false, error: 'invalid_proposal' };
      }
      if (!['high', 'review', 'needs_input'].includes(proposal.confidence)
        || !ACTIONABLE_RISKS.has(proposal.risk_class)) {
        return { ok: false, error: 'invalid_proposal_metadata' };
      }
      if (proposal.action !== undefined && !['fill', 'skip', 'ask_user'].includes(proposal.action)) {
        return { ok: false, error: 'invalid_proposal_action' };
      }
      let rawAnswer = proposal.value;
      if (proposal.value_type === 'selected_values') rawAnswer = proposal.selected_values;
      else if (proposal.value_type === 'checked') rawAnswer = proposal.checked;
      const answer = sanitizeAnswer(rawAnswer);
      if (!answer.ok) return answer;
      const citationCount = proposal.citationCount ?? (Array.isArray(proposal.citation_ids)
        ? proposal.citation_ids.length
        : 0);
      if (!Number.isSafeInteger(citationCount) || citationCount < 0 || citationCount > MAX_CITATION_COUNT) {
        return { ok: false, error: 'invalid_citation_count' };
      }
      proposals.push({
        fieldId,
        value: answer.value,
        confirmed: proposal.confirmed === true,
        confidence: proposal.confidence,
        risk_class: proposal.risk_class,
        citationCount,
        action: proposal.action || (proposal.confidence === 'needs_input' ? 'ask_user' : 'fill')
      });
    }
    return {
      ok: true,
      value: {
        channel: RUNTIME_CHANNEL,
        version: RUNTIME_VERSION,
        type: message.type,
        requestId: message.requestId,
        payload: { proposals }
      }
    };
  }

  if (message.type === 'FIELD_FILL_REQUEST') {
    if (!isValidFieldId(payload.fieldId)) return { ok: false, error: 'invalid_field_id' };
    if (payload.fingerprint !== undefined
      && (typeof payload.fingerprint !== 'string' || !/^[a-f0-9]{16}$/.test(payload.fingerprint))) {
      return { ok: false, error: 'invalid_fingerprint' };
    }
    if (payload.skipIfPopulated !== undefined && typeof payload.skipIfPopulated !== 'boolean') {
      return { ok: false, error: 'invalid_skip_if_populated' };
    }
    const answer = sanitizeAnswer(payload.value);
    if (!answer.ok) return answer;
    return {
      ok: true,
      value: {
        channel: RUNTIME_CHANNEL,
        version: RUNTIME_VERSION,
        type: message.type,
        requestId: message.requestId,
        payload: {
          fieldId: payload.fieldId,
          fingerprint: payload.fingerprint,
          value: answer.value,
          confirmed: payload.confirmed === true,
          skipIfPopulated: payload.skipIfPopulated === true
        }
      }
    };
  }

  return {
    ok: true,
    value: {
      channel: RUNTIME_CHANNEL,
      version: RUNTIME_VERSION,
      type: message.type,
      requestId: message.requestId,
      payload: {}
    }
  };
};

const createEnvelope = (type, requestId, payload) => {
  if (!OUTBOUND_MESSAGE_TYPES.has(type)) throw new TypeError(`Unsupported outbound message type: ${type}`);
  if (!isValidRequestId(requestId)) throw new TypeError('Invalid outbound request ID.');
  return {
    channel: RUNTIME_CHANNEL,
    version: RUNTIME_VERSION,
    type,
    requestId,
    payload
  };
};

const normalizedOption = (value) => boundedText(value, MAX_OPTION_LENGTH).toLocaleLowerCase();

const findOptionByAnswer = (record, answer) => {
  if (typeof answer !== 'string') return null;
  const expected = normalizedOption(answer);
  return record.optionRecords.find((option) => normalizedOption(option.label) === expected) || null;
};

const setNativeProperty = (element, property, value) => {
  let prototype = Object.getPrototypeOf(element);
  for (let depth = 0; prototype && depth < 8; depth += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(prototype, property);
    if (typeof descriptor?.set === 'function') {
      descriptor.set.call(element, value);
      return true;
    }
    prototype = Object.getPrototypeOf(prototype);
  }
  try {
    element[property] = value;
    return true;
  } catch {
    return false;
  }
};

const stablePageStructure = scan => JSON.stringify({
  adapter: String(scan?.adapter || ''),
  captchaPresent: Boolean(scan?.captchaPresent),
  fields: Array.isArray(scan?.fields) ? scan.fields.map(field => ({
    fieldId: String(field?.fieldId || ''),
    fingerprint: String(field?.fingerprint || ''),
    label: String(field?.label || ''),
    type: String(field?.type || ''),
    options: Array.isArray(field?.options) ? field.options : [],
    required: Boolean(field?.required),
    riskClass: String(field?.riskClass || ''),
    fillMode: field?.fillMode === 'copy_only' ? 'copy_only' : '',
    maxLength: Number.isSafeInteger(field?.maxLength) ? field.maxLength : null
  })) : [],
  discovery: {
    mode: String(scan?.discovery?.mode || ''),
    recognizedCount: Number(scan?.discovery?.recognizedCount || 0),
    unsupportedCount: Number(scan?.discovery?.unsupportedCount || 0),
    contexts: Array.from(scan?.discovery?.contexts || []).map(context => ({
      kind: String(context?.kind || ''),
      count: Number(context?.count || 0),
      status: String(context?.status || '')
    }))
  },
  pageId: String(scan?.pageId || ''),
  urlHash: String(scan?.urlHash || '')
});

const dispatchFieldEvents = (element) => {
  const EventConstructor = element?.ownerDocument?.defaultView?.Event || globalThis.Event;
  if (typeof EventConstructor !== 'function' || typeof element?.dispatchEvent !== 'function') return;
  element.dispatchEvent(new EventConstructor('input', { bubbles: true, composed: true }));
  element.dispatchEvent(new EventConstructor('change', { bubbles: true, composed: true }));
  element.dispatchEvent(new EventConstructor('blur', { bubbles: false, composed: true }));
};

const parseBooleanAnswer = (value) => {
  if (typeof value === 'boolean') return value;
  const normalized = normalizeTokenText(value);
  if (['true', 'yes', '1', 'checked', 'selected'].includes(normalized)) return true;
  if (['false', 'no', '0', 'unchecked', 'not selected'].includes(normalized)) return false;
  return null;
};

const canMutateElement = (element) => Boolean(element)
  && element.isConnected !== false
  && !element.disabled
  && !element.readOnly
  && !element.hidden;

const isPlaceholderOption = (option) => {
  const label = normalizeTokenText(option?.label || option?.textContent || '');
  const value = boundedText(option?.value, MAX_OPTION_LENGTH);
  return (!label && !value)
    || /^(?:--+\s*)?(?:choose|please select|select)\b/u.test(label)
    || (Boolean(option?.disabled) && !value);
};

export const fieldRecordHasValue = (record) => {
  if (!record?.elements?.length) return false;
  const element = record.elements[0];
  try {
    if (TEXT_INPUT_TYPES.has(record.type) || record.type === 'textarea') {
      return boundedText(element.value, MAX_PROPOSAL_LENGTH).length > 0;
    }
    if (record.type === 'contenteditable') {
      return boundedText(element.textContent, MAX_PROPOSAL_LENGTH).length > 0;
    }
    if (record.type === 'select-one') {
      const selected = Array.from(element.options || [])[Number(element.selectedIndex)];
      return Boolean(selected) && !isPlaceholderOption(selected);
    }
    if (record.type === 'select-multiple') {
      return record.optionRecords.some((option) => Boolean(option.node?.selected));
    }
    if (record.type === 'radio' || record.type === 'checkbox') {
      return record.elements.some((candidate) => Boolean(candidate.checked));
    }
  } catch {}
  return false;
};

export const fillFieldRecord = async (record, answer) => {
  if (!record || !ACTIONABLE_RISKS.has(record.descriptor?.riskClass)) {
    return { attempted: false, reason: 'field_not_actionable' };
  }
  if (record.fillMode === 'copy_only' || record.descriptor?.fillMode === 'copy_only') {
    return { attempted: false, reason: 'custom_widget_copy_only' };
  }
  const element = record.elements[0];
  if (!canMutateElement(element)) return { attempted: false, reason: 'field_not_mutable' };

  if (TEXT_INPUT_TYPES.has(record.type) || record.type === 'textarea') {
    if (typeof answer !== 'string') return { attempted: false, reason: 'answer_type_mismatch' };
    const maxLength = Number(element.maxLength);
    if (Number.isFinite(maxLength) && maxLength >= 0 && answer.length > maxLength) {
      return { attempted: false, reason: 'answer_exceeds_maxlength' };
    }
    if (!setNativeProperty(element, 'value', answer)) return { attempted: false, reason: 'native_setter_failed' };
    dispatchFieldEvents(element);
    return { attempted: true };
  }

  if (record.type === 'contenteditable') {
    if (typeof answer !== 'string') return { attempted: false, reason: 'answer_type_mismatch' };
    try {
      element.textContent = answer;
    } catch {
      return { attempted: false, reason: 'native_setter_failed' };
    }
    dispatchFieldEvents(element);
    return { attempted: true };
  }

  if (record.type === 'select-one') {
    const option = findOptionByAnswer(record, answer);
    if (!option) return { attempted: false, reason: 'option_not_found' };
    const index = Array.from(element.options || []).indexOf(option.node);
    if (index < 0 || !setNativeProperty(element, 'selectedIndex', index)) {
      return { attempted: false, reason: 'native_setter_failed' };
    }
    dispatchFieldEvents(element);
    return { attempted: true };
  }

  if (record.type === 'select-multiple') {
    if (!Array.isArray(answer)) return { attempted: false, reason: 'answer_type_mismatch' };
    const expected = new Set(answer.map(normalizedOption));
    const available = new Set(record.optionRecords.map((option) => normalizedOption(option.label)));
    if (Array.from(expected).some((value) => !available.has(value))) {
      return { attempted: false, reason: 'option_not_found' };
    }
    record.optionRecords.forEach((option) => {
      const selected = expected.has(normalizedOption(option.label));
      setNativeProperty(option.node, 'selected', selected);
    });
    dispatchFieldEvents(element);
    return { attempted: true };
  }

  if (record.type === 'radio') {
    const option = findOptionByAnswer(record, answer);
    if (!option || !canMutateElement(option.node)) return { attempted: false, reason: 'option_not_found' };
    if (!setNativeProperty(option.node, 'checked', true)) return { attempted: false, reason: 'native_setter_failed' };
    dispatchFieldEvents(option.node);
    return { attempted: true };
  }

  if (record.type === 'checkbox') {
    if (record.elements.length === 1) {
      const checked = parseBooleanAnswer(answer);
      if (checked === null) return { attempted: false, reason: 'answer_type_mismatch' };
      if (!setNativeProperty(element, 'checked', checked)) return { attempted: false, reason: 'native_setter_failed' };
      dispatchFieldEvents(element);
      return { attempted: true };
    }
    if (!Array.isArray(answer)) return { attempted: false, reason: 'answer_type_mismatch' };
    const expected = new Set(answer.map(normalizedOption));
    const available = new Set(record.optionRecords.map((option) => normalizedOption(option.label)));
    if (Array.from(expected).some((value) => !available.has(value))) {
      return { attempted: false, reason: 'option_not_found' };
    }
    record.optionRecords.forEach((option) => {
      const checked = expected.has(normalizedOption(option.label));
      setNativeProperty(option.node, 'checked', checked);
      dispatchFieldEvents(option.node);
    });
    return { attempted: true };
  }

  return { attempted: false, reason: 'unsupported_field_type' };
};

const verifyFieldRecord = (record, answer) => {
  if (!record) return false;
  const element = record.elements[0];
  try {
    if (TEXT_INPUT_TYPES.has(record.type) || record.type === 'textarea') return element.value === answer;
    if (record.type === 'contenteditable') return element.textContent === answer;
    if (record.type === 'select-one') {
      const selected = Array.from(element.options || [])[Number(element.selectedIndex)];
      return normalizedOption(selected?.label || selected?.textContent || '') === normalizedOption(answer);
    }
    if (record.type === 'select-multiple') {
      const expected = new Set(Array.from(answer || []).map(normalizedOption));
      const actual = new Set(record.optionRecords.filter((option) => option.node.selected)
        .map((option) => normalizedOption(option.label)));
      return expected.size === actual.size && Array.from(expected).every((value) => actual.has(value));
    }
    if (record.type === 'radio') return Boolean(findOptionByAnswer(record, answer)?.node.checked);
    if (record.type === 'checkbox' && record.elements.length === 1) {
      return element.checked === parseBooleanAnswer(answer);
    }
    if (record.type === 'checkbox') {
      const expected = new Set(Array.from(answer || []).map(normalizedOption));
      const actual = new Set(record.optionRecords.filter((option) => option.node.checked)
        .map((option) => normalizedOption(option.label)));
      return expected.size === actual.size && Array.from(expected).every((value) => actual.has(value));
    }
  } catch {}
  return false;
};

export const overlayActionPresentation = ({ field, proposal } = {}) => {
  const label = boundedText(field?.label || 'this field', MAX_LABEL_LENGTH);
  const deterministic = field?.riskClass === RISK_CLASSES.F1_VERIFIED;
  const hasProposal = Boolean(proposal);
  const regenerateDisabled = deterministic || !hasProposal;
  const regenerateTitle = deterministic
    ? 'Saved profile facts are deterministic'
    : hasProposal
      ? 'Regenerate from saved sources'
      : 'Analyze the page before regenerating';
  return Object.freeze({
    review: Object.freeze({
      ariaLabel: `Open the answer for ${label} in Job Application Copilot`,
      title: 'Review in sidebar'
    }),
    regenerate: Object.freeze({
      ariaLabel: `Regenerate the answer for ${label} from saved sources`,
      title: regenerateTitle,
      disabled: regenerateDisabled
    })
  });
};

const createOverlayIcon = (doc, name) => {
  const svg = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 20 20');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  const path = doc.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', name === 'regenerate'
    ? 'M16.2 6.8A7 7 0 1 0 16 13.7M16.2 3.3v3.5h-3.5'
    : 'M3.5 4.5h9M3.5 8h7M3.5 11.5h5M12.5 14.5l3.8-3.8m0 0h-3m3 0v3');
  svg.append(path);
  return svg;
};

const createOverlay = ({ doc, view, onSelect, onRegenerate, onCopy }) => {
  if (!doc?.documentElement || typeof doc.createElement !== 'function') return null;
  if (doc.getElementById?.(OVERLAY_HOST_ID)) return null;
  const host = doc.createElement('div');
  host.id = OVERLAY_HOST_ID;
  host.setAttribute('data-job-application-copilot-overlay', 'true');
  host.style.setProperty('all', 'initial', 'important');
  host.style.setProperty('position', 'fixed', 'important');
  host.style.setProperty('inset', '0', 'important');
  host.style.setProperty('z-index', '2147483647', 'important');
  host.style.setProperty('pointer-events', 'none', 'important');
  host.style.setProperty('width', '100vw', 'important');
  host.style.setProperty('height', '100vh', 'important');
  const shadow = host.attachShadow?.({ mode: 'closed' });
  if (!shadow) return null;
  const style = doc.createElement('style');
  style.textContent = `
    :host { all: initial; }
    #layer { position: fixed; inset: 0; pointer-events: none; font: 600 11px/1.2 system-ui, sans-serif; }
    .anchor { position: fixed; display: flex; gap: 4px; pointer-events: auto; transform: translateY(-2px); }
    button { appearance: none; border: 1px solid #CBD5E1; border-radius: 6px; background: #FFFFFF; color: #091F3B; cursor: pointer; min-width: 28px; height: 28px; padding: 5px 7px; box-shadow: 0 2px 6px rgba(9, 31, 59, .14); font: inherit; }
    button.icon { align-items: center; display: inline-flex; justify-content: center; padding: 5px; width: 28px; }
    button.icon svg { display: block; fill: none; height: 16px; stroke: currentColor; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.7; width: 16px; }
    button:hover, button:focus-visible, button[aria-pressed="true"] { border-color: #005FED; box-shadow: 0 0 0 2px rgba(0, 95, 237, .18); outline: none; }
    button:disabled { background: #F1F5F9; color: #94A3B8; cursor: not-allowed; }
    .copy[hidden] { display: none; }
    .copy { border-color: #005FED; }
  `;
  const layer = doc.createElement('div');
  layer.id = 'layer';
  shadow.append(style, layer);
  doc.documentElement.append(host);
  const controls = new Map();
  const copyOnlyFields = new Set();
  let currentRecords = new Map();
  let currentProposals = new Map();
  let selectedFieldId = '';

  const stopOverlayEvent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const getAnchorElement = (record) => record?.elements?.find((element) => {
    try {
      const rect = element.getBoundingClientRect();
      return rect.width > 0 || rect.height > 0;
    } catch {
      return false;
    }
  }) || record?.elements?.[0];

  const position = () => {
    const viewportWidth = Math.max(320, Number(view?.innerWidth) || 1_024);
    const viewportHeight = Math.max(240, Number(view?.innerHeight) || 768);
    controls.forEach((wrapper, fieldId) => {
      const anchor = getAnchorElement(currentRecords.get(fieldId));
      let rect;
      try {
        rect = anchor?.getBoundingClientRect?.();
      } catch {}
      if (!rect || rect.bottom < 0 || rect.top > viewportHeight || rect.right < 0 || rect.left > viewportWidth) {
        wrapper.hidden = true;
        return;
      }
      wrapper.hidden = false;
      const left = Math.max(4, Math.min(viewportWidth - 132, Number(rect.right) + 6));
      const top = Math.max(4, Math.min(viewportHeight - 30, Number(rect.top)));
      wrapper.style.left = `${left}px`;
      wrapper.style.top = `${top}px`;
    });
  };

  const render = (scan, records, proposals = currentProposals) => {
    currentRecords = records;
    currentProposals = proposals;
    controls.clear();
    layer.replaceChildren();
    scan.fields.forEach((field) => {
      const wrapper = doc.createElement('div');
      wrapper.className = 'anchor';
      const proposal = currentProposals.get(field.fieldId);
      const presentation = overlayActionPresentation({ field, proposal });
      const select = doc.createElement('button');
      select.type = 'button';
      select.className = 'icon review';
      select.dataset.fieldId = field.fieldId;
      select.dataset.risk = field.riskClass;
      select.append(createOverlayIcon(doc, 'review'));
      select.setAttribute('aria-pressed', field.fieldId === selectedFieldId ? 'true' : 'false');
      select.setAttribute('aria-label', presentation.review.ariaLabel);
      select.setAttribute('title', presentation.review.title);
      select.addEventListener('click', (event) => {
        stopOverlayEvent(event);
        selectedFieldId = field.fieldId;
        controls.forEach((control) => control.querySelector?.('button[data-field-id]')?.setAttribute('aria-pressed', 'false'));
        select.setAttribute('aria-pressed', 'true');
        onSelect(field.fieldId);
      });
      const regenerate = doc.createElement('button');
      regenerate.type = 'button';
      regenerate.className = 'icon regenerate';
      regenerate.dataset.regenerateFieldId = field.fieldId;
      regenerate.disabled = presentation.regenerate.disabled;
      regenerate.append(createOverlayIcon(doc, 'regenerate'));
      regenerate.setAttribute('aria-label', presentation.regenerate.ariaLabel);
      regenerate.setAttribute('title', presentation.regenerate.title);
      regenerate.addEventListener('click', (event) => {
        stopOverlayEvent(event);
        if (regenerate.disabled) return;
        selectedFieldId = field.fieldId;
        controls.forEach((control) => control.querySelector?.('button[data-field-id]')?.setAttribute('aria-pressed', 'false'));
        select.setAttribute('aria-pressed', 'true');
        onRegenerate(field.fieldId);
      });
      const copy = doc.createElement('button');
      copy.type = 'button';
      copy.className = 'copy';
      copy.textContent = 'Copy';
      const proposalCanCopy = Boolean(proposal)
        && copyOnlyFields.has(field.fieldId)
        && proposal.action === 'fill'
        && proposal.confidence !== 'needs_input'
        && (field.riskClass !== RISK_CLASSES.F2_REVIEW || proposal.confirmed === true);
      copy.hidden = !proposalCanCopy;
      copy.setAttribute('aria-label', `Open the side panel to copy the reviewed answer for ${field.label}`);
      copy.addEventListener('click', async (event) => {
        stopOverlayEvent(event);
        if (!currentProposals.has(field.fieldId)) return;
        await onCopy(field.fieldId);
      });
      wrapper.append(select, regenerate, copy);
      layer.append(wrapper);
      controls.set(field.fieldId, wrapper);
    });
    position();
  };

  const setCopyOnly = (fieldId) => {
    copyOnlyFields.add(fieldId);
    const wrapper = controls.get(fieldId);
    const copy = wrapper?.querySelector?.('.copy');
    const proposal = currentProposals.get(fieldId);
    const riskClass = currentRecords.get(fieldId)?.descriptor?.riskClass;
    const allowed = Boolean(proposal)
      && proposal.action === 'fill'
      && proposal.confidence !== 'needs_input'
      && (riskClass !== RISK_CLASSES.F2_REVIEW || proposal.confirmed === true);
    if (copy && allowed) copy.hidden = false;
  };

  const onViewportChange = () => {
    if (typeof view?.requestAnimationFrame === 'function') view.requestAnimationFrame(position);
    else position();
  };
  view?.addEventListener?.('scroll', onViewportChange, { capture: true, passive: true });
  view?.addEventListener?.('resize', onViewportChange, { passive: true });

  return {
    render,
    setCopyOnly,
    clearCopyOnly: (fieldId) => {
      copyOnlyFields.delete(fieldId);
      const copy = controls.get(fieldId)?.querySelector?.('.copy');
      if (copy) copy.hidden = true;
    },
    resetCopyOnly: () => {
      copyOnlyFields.clear();
      controls.forEach((control) => {
        const copy = control.querySelector?.('.copy');
        if (copy) copy.hidden = true;
      });
    },
    destroy: () => {
      view?.removeEventListener?.('scroll', onViewportChange, { capture: true });
      view?.removeEventListener?.('resize', onViewportChange);
      host.remove?.();
    }
  };
};

const createLocalRequestId = () => `overlay-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const createSelectedFieldDescriptor = (descriptor) => ({
  fieldId: descriptor.fieldId,
  fingerprint: descriptor.fingerprint,
  label: descriptor.label,
  type: descriptor.type,
  options: descriptor.options,
  nearbyText: descriptor.nearbyText,
  required: descriptor.required,
  riskClass: descriptor.riskClass,
  ...(descriptor.fillMode === 'copy_only' ? { fillMode: 'copy_only' } : {})
});

export const createPageRuntime = ({
  doc = globalThis.document,
  view = globalThis.window,
  runtimeApi = globalThis.chrome?.runtime,
  settle = () => new Promise((resolve) => setTimeout(resolve, 40)),
  overlayFactory = createOverlay
} = {}) => {
  if (!doc || !runtimeApi) throw new TypeError('Page runtime requires document and chrome.runtime APIs.');
  let activeScan = null;
  let destroyed = false;
  let domRevision = 0;
  let mutationTimer = null;
  let mutationObserver = null;
  let observeMutationRoots = () => {};
  const observedMutationRoots = new Set();
  const proposals = new Map();

  const emit = async (envelope) => {
    try {
      await runtimeApi.sendMessage(envelope);
    } catch {}
  };

  const scan = () => {
    activeScan = scanDocument({
      doc,
      url: doc?.location?.href || view?.location?.href || '',
      domRevision
    });
    observeMutationRoots();
    ensureOverlay();
    overlay?.render(activeScan.result, activeScan.records, proposals);
    return activeScan;
  };

  const emitFieldIntent = (fieldId, intent) => {
    const current = scan();
    const record = current.records.get(fieldId);
    if (!record) return;
    const requestId = createLocalRequestId();
    void emit(createEnvelope('FIELD_SELECTED', requestId, {
      field: createSelectedFieldDescriptor(record.descriptor),
      intent
    }));
  };

  const onSelect = (fieldId) => emitFieldIntent(fieldId, 'review');

  const onRegenerate = (fieldId) => emitFieldIntent(fieldId, 'regenerate');

  const onCopy = async (fieldId) => {
    const current = scan();
    const record = current.records.get(fieldId);
    if (!record) return;
    const requestId = createLocalRequestId();
    await emit(createEnvelope('FIELD_SELECTED', requestId, {
      field: createSelectedFieldDescriptor(record.descriptor),
      intent: 'review'
    }));
  };

  let overlay = null;
  const ensureOverlay = () => {
    if (!overlay && doc?.documentElement) overlay = overlayFactory({ doc, view, onSelect, onRegenerate, onCopy });
  };
  ensureOverlay();

  const nodeTouchesFormStructure = (node) => {
    if (!node || isOverlayNodeOrDescendant(node)) return false;
    try {
      if (safeMatches(node, STRUCTURAL_FIELD_SELECTOR)
        || safeMatches(node, STRUCTURAL_OPTION_SELECTOR)
        || safeMatches(node, CAPTCHA_SELECTOR)) return true;
      if (node.querySelector?.(`${STRUCTURAL_FIELD_SELECTOR},${STRUCTURAL_OPTION_SELECTOR},${CAPTCHA_SELECTOR}`)) return true;
    } catch {}
    return Boolean(getOpenShadowRoot(node)
      || collectOpenShadowRoots(node).length
      || collectShadowCandidates(node).length);
  };

  const scheduleMutationScan = () => {
    const previousStructure = stablePageStructure(activeScan?.result);
    if (mutationTimer !== null) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => {
      mutationTimer = null;
      if (destroyed) return;
      const current = scan();
      if (previousStructure && stablePageStructure(current.result) === previousStructure) return;
      domRevision += 1;
      const revised = scan();
      proposals.clear();
      overlay?.resetCopyOnly();
      overlay?.render(revised.result, revised.records, proposals);
      const requestId = `mutation-${domRevision}-${Date.now().toString(36)}`;
      void emit(createEnvelope('PAGE_SCAN_RESULT', requestId, {
        ...revised.result,
        stale: true
      }));
    }, 80);
  };

  const mutationObserverOptions = {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: [
      'aria-autocomplete',
      'aria-controls',
      'aria-expanded',
      'aria-haspopup',
      'aria-hidden',
      'aria-label',
      'aria-labelledby',
      'aria-required',
      'autocomplete',
      'class',
      'contenteditable',
      'data-callback',
      'data-headlessui-state',
      'data-radix-collection-item',
      'data-reach-combobox-input',
      'data-select2-id',
      'data-sitekey',
      'disabled',
      'hidden',
      'id',
      'multiple',
      'name',
      'placeholder',
      'readonly',
      'required',
      'role',
      'src',
      'style',
      'type'
    ]
  };
  const MutationObserverConstructor = view?.MutationObserver || globalThis.MutationObserver;
  mutationObserver = typeof MutationObserverConstructor === 'function'
    ? new MutationObserverConstructor((mutations) => {
      const structuralChange = Array.from(mutations || []).some((mutation) => {
        if (mutation.type === 'attributes') return nodeTouchesFormStructure(mutation.target);
        return [...Array.from(mutation.addedNodes || []), ...Array.from(mutation.removedNodes || [])]
          .some(nodeTouchesFormStructure);
      });
      if (!structuralChange) return;
      scheduleMutationScan();
    })
    : null;
  const observeMutationRoot = (root) => {
    if (!mutationObserver || !root || observedMutationRoots.has(root)
      || observedMutationRoots.size >= MAX_SHADOW_ROOTS + MAX_FRAME_DOCUMENTS + 1) return;
    try {
      mutationObserver.observe(root, mutationObserverOptions);
      observedMutationRoots.add(root);
    } catch {}
  };
  observeMutationRoots = () => {
    observeMutationRoot(doc.documentElement);
    collectOpenShadowRoots(doc).forEach(observeMutationRoot);
    collectAccessibleFrameDocuments(doc)
      .filter((record) => record.visible)
      .forEach((record) => {
        observeMutationRoot(record.doc?.documentElement);
        collectOpenShadowRoots(record.doc).forEach(observeMutationRoot);
      });
  };
  observeMutationRoots();

  const verifyPersistedFieldValue = async (fieldId, value) => {
    let current = null;
    for (let check = 0; check < 2; check += 1) {
      await settle();
      current = scan();
      if (!verifyFieldRecord(current.records.get(fieldId), value)) {
        return { current, verified: false };
      }
    }
    return { current, verified: true };
  };

  const attemptPersistentFieldFill = async ({
    record,
    fieldId,
    fingerprint,
    value
  }) => {
    const firstAttempt = await fillFieldRecord(record, value);
    if (!firstAttempt.attempted) {
      return { reason: firstAttempt.reason || 'controlled_field_rejected_value', verified: false };
    }

    let persistence = await verifyPersistedFieldValue(fieldId, value);
    if (persistence.verified) return { reason: 'verified_after_fill', verified: true };

    const retryRecord = persistence.current?.records?.get(fieldId);
    const retryAllowed = !persistence.current?.result?.captchaPresent
      && retryRecord
      && retryRecord.descriptor.fingerprint === fingerprint
      && retryRecord.descriptor.riskClass === record.descriptor.riskClass
      && !fieldRecordHasValue(retryRecord);
    if (!retryAllowed) return { reason: 'controlled_field_rejected_value', verified: false };

    const retryAttempt = await fillFieldRecord(retryRecord, value);
    if (!retryAttempt.attempted) {
      return { reason: retryAttempt.reason || 'controlled_field_rejected_value', verified: false };
    }
    persistence = await verifyPersistedFieldValue(fieldId, value);
    return {
      reason: persistence.verified ? 'verified_after_retry' : 'controlled_field_rejected_value',
      verified: persistence.verified
    };
  };

  const handleValidatedMessage = async (message) => {
    if (message.type === 'PAGE_SCAN_REQUEST') {
      const current = scan();
      return createEnvelope('PAGE_SCAN_RESULT', message.requestId, current.result);
    }

    if (message.type === 'PAGE_PROPOSALS_UPDATE') {
      const current = scan();
      proposals.clear();
      overlay?.resetCopyOnly();
      message.payload.proposals.forEach((proposal) => {
        const record = current.records.get(proposal.fieldId);
        if (!record || record.descriptor.riskClass !== proposal.risk_class) return;
        proposals.set(proposal.fieldId, proposal);
      });
      overlay?.render(current.result, current.records, proposals);
      return {
        channel: RUNTIME_CHANNEL,
        version: RUNTIME_VERSION,
        type: 'PAGE_PROPOSALS_ACCEPTED',
        requestId: message.requestId,
        payload: { acceptedCount: proposals.size }
      };
    }

    if (message.type === 'SANITIZED_EXPORT_REQUEST') {
      const current = scan();
      const exported = createSanitizedExport(current.result);
      return createEnvelope('SANITIZED_EXPORT_RESULT', message.requestId, {
        json: JSON.stringify(exported)
      });
    }

    const before = scan();
    const record = before.records.get(message.payload.fieldId);
    const descriptor = record?.descriptor;
    const fingerprintMatches = !message.payload.fingerprint
      || descriptor?.fingerprint === message.payload.fingerprint;
    let status = 'copy_only';
    let reason = 'field_missing_after_rescan';
    let verified = false;
    let skipped = false;

    if (before.result.captchaPresent) {
      reason = 'captcha_present';
    } else if (record && fingerprintMatches) {
      if (record.fillMode === 'copy_only' || descriptor.fillMode === 'copy_only') {
        reason = 'custom_widget_copy_only';
      } else if (descriptor.riskClass === RISK_CLASSES.F2_REVIEW && !message.payload.confirmed) {
        reason = 'review_confirmation_required';
      } else if (message.payload.skipIfPopulated && fieldRecordHasValue(record)) {
        status = 'skipped';
        reason = 'field_already_has_value';
        skipped = true;
      } else {
        const attempted = await attemptPersistentFieldFill({
          record,
          fieldId: message.payload.fieldId,
          fingerprint: descriptor.fingerprint,
          value: message.payload.value
        });
        reason = attempted.reason;
        verified = attempted.verified;
        if (verified) {
          status = 'filled';
          proposals.delete(message.payload.fieldId);
          overlay?.clearCopyOnly(message.payload.fieldId);
        }
      }
    } else if (record) {
      reason = 'field_fingerprint_changed';
    }

    if (!verified && !skipped) {
      proposals.set(message.payload.fieldId, {
        fieldId: message.payload.fieldId,
        value: message.payload.value,
        confirmed: message.payload.confirmed,
        confidence: 'review',
        risk_class: descriptor?.riskClass || RISK_CLASSES.F2_REVIEW,
        citationCount: 0,
        action: 'fill'
      });
      overlay?.render(activeScan.result, activeScan.records, proposals);
      overlay?.setCopyOnly(message.payload.fieldId);
    }

    return createEnvelope('FIELD_FILL_RESULT', message.requestId, {
      fieldId: message.payload.fieldId,
      status,
      verified,
      copyOnly: !verified && !skipped,
      reason
    });
  };

  const listener = (rawMessage, sender, sendResponse) => {
    if (destroyed) return false;
    if (sender?.id && runtimeApi.id && sender.id !== runtimeApi.id) return false;
    const validated = validateRuntimeMessage(rawMessage);
    if (!validated.ok) return false;
    void handleValidatedMessage(validated.value).then(async (response) => {
      if (OUTBOUND_MESSAGE_TYPES.has(response.type)) await emit(response);
      sendResponse?.(response);
    }).catch(() => {
      sendResponse?.({
        channel: RUNTIME_CHANNEL,
        version: RUNTIME_VERSION,
        type: 'PAGE_RUNTIME_ERROR',
        requestId: validated.value.requestId,
        payload: { code: 'runtime_failure' }
      });
    });
    return true;
  };

  runtimeApi.onMessage?.addListener?.(listener);

  return {
    scan,
    handleMessage: async (message) => {
      const validated = validateRuntimeMessage(message);
      if (!validated.ok) return validated;
      return handleValidatedMessage(validated.value);
    },
    destroy: () => {
      destroyed = true;
      mutationObserver?.disconnect?.();
      observedMutationRoots.clear();
      if (mutationTimer !== null) clearTimeout(mutationTimer);
      runtimeApi.onMessage?.removeListener?.(listener);
      proposals.clear();
      overlay?.destroy();
    }
  };
};

if (typeof document !== 'undefined' && globalThis.chrome?.runtime) {
  const isolatedWorld = globalThis;
  if (!isolatedWorld[RUNTIME_INSTANCE_KEY]) {
    try {
      isolatedWorld[RUNTIME_INSTANCE_KEY] = createPageRuntime();
    } catch {}
  }
}
