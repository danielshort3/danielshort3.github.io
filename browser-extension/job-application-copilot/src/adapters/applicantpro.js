import {
  GENERIC_CANDIDATE_SELECTOR,
  adapterInternals,
  extractGenericJobMetadata,
  genericAdapter
} from './generic.js';

const { normalizeText, safeQuery, safeQueryAll, textFromNode } = adapterInternals;

const APPLICANTPRO_HOST_RE = /(^|\.)applicantpro(?:\.com|\.net)$/i;
const APPLICANTPRO_CONTAINER_SELECTOR = [
  '.form-group',
  '.control-group',
  '.application-question',
  '.question',
  '.field',
  'fieldset',
  '[data-question-id]'
].join(',');

const hostnameFromUrl = (rawUrl) => {
  try {
    return new URL(String(rawUrl || '')).hostname;
  } catch {
    return '';
  }
};

export const isApplicantProPage = ({ doc, url } = {}) => {
  if (APPLICANTPRO_HOST_RE.test(hostnameFromUrl(url))) return true;
  const generator = safeQuery(doc, 'meta[name="generator"]');
  if (/applicantpro/i.test(generator?.getAttribute?.('content') || '')) return true;
  return Boolean(safeQuery(doc, [
    'form[action*="applicantpro.com"]',
    'form[action*="applicantpro.net"]',
    '[data-applicantpro]'
  ].join(',')));
};

const getContainer = (element) => {
  try {
    return element?.closest?.(APPLICANTPRO_CONTAINER_SELECTOR)
      || genericAdapter.getFieldContainer(element);
  } catch {
    return genericAdapter.getFieldContainer(element);
  }
};

const getApplicantProLabel = (element) => {
  const container = getContainer(element);
  const labelNode = safeQuery(container, [
    ':scope > .control-label',
    ':scope > .question-label',
    ':scope > .field-label',
    ':scope > label',
    ':scope > legend',
    '.control-label',
    '.question-label'
  ].join(','));
  return textFromNode(labelNode) || genericAdapter.getSupplementalLabel(element);
};

const getApplicationRoot = (doc) => {
  const explicitApplyForm = safeQuery(doc, 'form#apply');
  if (explicitApplyForm) return explicitApplyForm;

  const forms = safeQueryAll(doc, 'form')
    .filter((node) => String(node?.tagName || '').toLowerCase() === 'form');
  const applicationForm = forms.find((form) => {
    const hasListing = Boolean(safeQuery(form, 'input[name="listing_id"]'));
    const hasCandidateIdentity = Boolean(safeQuery(form, [
      'input[name="first_name"]',
      'input[name="last_name"]',
      'input[name="email"]',
      'input[name="contact_number"]'
    ].join(',')));
    return hasListing && hasCandidateIdentity;
  });
  if (applicationForm) return applicationForm;
  if (forms.length) return null;
  const documentControls = safeQueryAll(doc, GENERIC_CANDIDATE_SELECTOR);
  const hasApplicationField = documentControls.some((control) => /^(?:first_name|last_name|email|contact_number|linkedin|resume|cover_letter)$/iu.test(
    String(control?.name || control?.id || '')
  ));
  return hasApplicationField ? doc : null;
};

const extractApplicantProJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-company-name]',
    '.company-name',
    '.job-company',
    '#company-name'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-job-title]',
    '.job-title',
    '.posting-title',
    'main h1',
    'h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-job-location]',
    '.job-location',
    '.posting-location',
    '.location'
  ].join(',')));
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'ApplicantPro'
  };
};

export const applicantProAdapter = Object.freeze({
  id: 'applicantpro',
  sourceName: 'ApplicantPro',
  matches: isApplicantProPage,
  getApplicationRoot,
  getDiscoveryContexts: (doc) => {
    const root = getApplicationRoot(doc);
    return root ? [{ kind: 'application', root }] : [];
  },
  collectCandidates: (doc) => {
    const root = getApplicationRoot(doc);
    return root ? safeQueryAll(root, GENERIC_CANDIDATE_SELECTOR) : [];
  },
  getFieldContainer: getContainer,
  getSupplementalLabel: getApplicantProLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractApplicantProJobMetadata
});
