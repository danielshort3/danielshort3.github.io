import {
  GENERIC_CANDIDATE_SELECTOR,
  adapterInternals,
  genericAdapter
} from './generic.js';

const { controlsForRoot, normalizeMarker, normalizeText, safeQuery, safeQueryAll, textFromNode } = adapterInternals;

const EXCLUDED_INPUT_TYPES = new Set([
  'button',
  'file',
  'hidden',
  'image',
  'password',
  'reset',
  'submit'
]);

const APPLICATION_MARKER_RE = /(?:^|[\s_-])(?:application|apply|applicant|candidate|job)(?:$|[\s_-])/iu;
const ACCOUNT_GATE_RE = /(?:^|[\s_-])(?:account|create-account|login|log-in|signin|sign-in|register|registration)(?:$|[\s_-])/iu;
const AUXILIARY_FORM_RE = /(?:^|[\s_-])(?:alert|newsletter|refer|referral|search|subscribe|subscription)(?:$|[\s_-])/iu;
const CONSENT_RE = /\b(?:acknowledge|agree|agreement|attest|certif(?:y|ication)|consent|privacy|signature|terms)\b/iu;
const HONEYPOT_RE = /\b(?:bot[\s_-]*trap|honeypot|leave(?:\s+this)?\s+blank|do\s+not\s+(?:enter|fill)|robots?\s+only)\b/iu;
const CAPTCHA_RE = /\b(?:captcha|recaptcha|hcaptcha|turnstile)\b/iu;
const IDENTITY_RE = /\b(?:first[\s_-]*name|last[\s_-]*name|full[\s_-]*name|candidate[\s_-]*name|email|e[\s_-]*mail|phone|telephone|mobile)\b/iu;
const RESUME_RE = /\b(?:resume|r[eé]sum[eé]|curriculum[\s_-]*vitae|\bcv\b)\b/iu;

export const hostnameFromUrl = (rawUrl) => {
  try {
    return new URL(String(rawUrl || '')).hostname.toLowerCase();
  } catch {
    return '';
  }
};

const nodeMarker = (node) => normalizeText([
  node?.tagName,
  node?.id,
  node?.name,
  node?.className,
  node?.getAttribute?.('action'),
  node?.getAttribute?.('aria-label'),
  node?.getAttribute?.('data-testid'),
  node?.getAttribute?.('data-automation-id'),
  node?.getAttribute?.('data-qa'),
  node?.getAttribute?.('role')
].filter(Boolean).map((value) => String(value).replace(/([a-z0-9])([A-Z])/g, '$1 $2')).join(' '), 600).toLowerCase();

const directLabelText = (element) => normalizeMarker([
  ...Array.from(element?.labels || []).map((label) => label?.textContent || ''),
  element?.getAttribute?.('aria-label') || '',
  element?.getAttribute?.('placeholder') || '',
  element?.getAttribute?.('title') || '',
  element?.getAttribute?.('autocomplete') || '',
  element?.getAttribute?.('data-field-name') || '',
  element?.getAttribute?.('data-testid') || '',
  element?.getAttribute?.('data-automation-id') || '',
  element?.name || '',
  element?.id || ''
].join(' '), 480);

export const isCollectableControl = (element) => {
  const tagName = String(element?.tagName || '').toLowerCase();
  const type = String(element?.type || element?.getAttribute?.('type') || 'text').toLowerCase();
  if (tagName === 'input' && EXCLUDED_INPUT_TYPES.has(type)) return false;
  const label = directLabelText(element);
  if (CAPTCHA_RE.test(label)) return false;
  if (HONEYPOT_RE.test(label)) return false;
  if (CONSENT_RE.test(label)) return false;
  return true;
};

export const collectControls = (root, doc = root?.ownerDocument || null) => controlsForRoot(root, doc)
  .filter(isCollectableControl);

const rootSignals = (root) => {
  const controls = controlsForRoot(root, root?.ownerDocument || null);
  const markers = controls.map((control) => `${nodeMarker(control)} ${directLabelText(control)}`);
  const identityCount = new Set(markers
    .filter((marker) => IDENTITY_RE.test(marker))
    .map((marker) => marker.match(IDENTITY_RE)?.[0]?.toLowerCase() || marker)).size;
  const hasResume = controls.some((control, index) => {
    const type = String(control?.type || control?.getAttribute?.('type') || '').toLowerCase();
    return type === 'file' && RESUME_RE.test(markers[index] || '');
  });
  const hasPassword = controls.some((control) => {
    const type = String(control?.type || control?.getAttribute?.('type') || '').toLowerCase();
    const autocomplete = String(control?.getAttribute?.('autocomplete') || '').toLowerCase();
    return type === 'password' || autocomplete.includes('password');
  });
  const questionCount = controls.filter((control) => {
    const tagName = String(control?.tagName || '').toLowerCase();
    return tagName === 'textarea' || tagName === 'select';
  }).length;
  return {
    controls,
    hasPassword,
    hasResume,
    identityCount,
    marker: nodeMarker(root),
    questionCount
  };
};

export const scoreApplicationRoot = (root) => {
  if (!root) return Number.NEGATIVE_INFINITY;
  const signals = rootSignals(root);
  let score = 0;
  if (APPLICATION_MARKER_RE.test(signals.marker)) score += 8;
  if (signals.hasResume) score += 8;
  score += Math.min(signals.identityCount, 4) * 2;
  if (signals.identityCount >= 2) score += 4;
  if (signals.questionCount) score += Math.min(signals.questionCount, 3);
  if (signals.hasPassword || ACCOUNT_GATE_RE.test(signals.marker)) score -= 18;
  if (AUXILIARY_FORM_RE.test(signals.marker)) score -= 18;
  return score;
};

export const selectApplicationRoot = (doc, selectors = []) => {
  const explicitRoots = selectors.flatMap((selector) => safeQueryAll(doc, selector));
  const forms = safeQueryAll(doc, 'form');
  const roots = Array.from(new Set([...explicitRoots, ...forms]));
  let best = null;
  let bestScore = 7;
  roots.forEach((root) => {
    const score = scoreApplicationRoot(root);
    if (score > bestScore) {
      best = root;
      bestScore = score;
    }
  });
  return best;
};

export const isAccountGate = (root) => {
  if (!root) return false;
  const signals = rootSignals(root);
  return signals.hasPassword || ACCOUNT_GATE_RE.test(signals.marker);
};

export const findAccountGate = (doc, selectors = []) => {
  const roots = [
    ...selectors.flatMap((selector) => safeQueryAll(doc, selector)),
    ...safeQueryAll(doc, 'form')
  ];
  return roots.find(isAccountGate) || null;
};

export const applicationContexts = ({ applicationRoot, accountGate = null } = {}) => {
  if (applicationRoot) return [{ kind: 'application', root: applicationRoot }];
  if (accountGate) return [{ kind: 'account_gate', root: accountGate }];
  return [];
};

export const createFieldHelpers = ({ containerSelector, labelSelectors = [] } = {}) => {
  const getFieldContainer = (element) => {
    try {
      return element?.closest?.(containerSelector) || genericAdapter.getFieldContainer(element);
    } catch {
      return genericAdapter.getFieldContainer(element);
    }
  };
  const getSupplementalLabel = (element) => {
    const container = getFieldContainer(element);
    const label = textFromNode(safeQuery(container, labelSelectors.join(',')));
    return label || genericAdapter.getSupplementalLabel(element);
  };
  return { getFieldContainer, getSupplementalLabel };
};

export const adapterSharedInternals = Object.freeze({
  ACCOUNT_GATE_RE,
  APPLICATION_MARKER_RE,
  AUXILIARY_FORM_RE,
  CAPTCHA_RE,
  CONSENT_RE,
  HONEYPOT_RE,
  directLabelText,
  nodeMarker,
  rootSignals
});
