import {
  adapterInternals,
  extractGenericJobMetadata,
  genericAdapter
} from './generic.js';
import {
  applicationContexts,
  collectControls,
  createFieldHelpers,
  hostnameFromUrl,
  selectApplicationRoot
} from './shared.js';

const { normalizeText, safeQuery, textFromNode, visibleLongText } = adapterInternals;

const GREENHOUSE_HOST_RE = /(^|\.)(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io$/iu;
const GREENHOUSE_APPLICATION_SELECTORS = [
  '#application-form',
  'form#application-form',
  'form.application-form',
  '[data-greenhouse-application-form]',
  '[data-testid="application-form"]',
  'form[action*="greenhouse.io"][action*="application"]'
];
const GREENHOUSE_CUSTOM_DOMAIN_MARKERS = [
  'form[action*="greenhouse.io"][action*="application"]',
  'iframe[src*="greenhouse.io"][src*="application"]',
  'iframe[src*="greenhouse.io"][src*="job_app"]',
  'script[src*="greenhouse.io"][src*="job_board"]',
  '[data-greenhouse-job-board]'
];

const fieldHelpers = createFieldHelpers({
  containerSelector: [
    '.application-question',
    '.application-field',
    '.field-container',
    '.form-field',
    '.field',
    '[data-field]',
    '[data-testid*="field"]',
    'fieldset'
  ].join(','),
  labelSelectors: [
    ':scope > .application-label',
    ':scope > .field-label',
    ':scope > label',
    ':scope > legend',
    '[data-testid*="label"]',
    '.application-label'
  ]
});

export const isGreenhousePage = ({ doc, url } = {}) => GREENHOUSE_HOST_RE.test(hostnameFromUrl(url))
  || GREENHOUSE_CUSTOM_DOMAIN_MARKERS.some((selector) => Boolean(safeQuery(doc, selector)));

const getGreenhouseApplicationRoot = (doc) => selectApplicationRoot(doc, GREENHOUSE_APPLICATION_SELECTORS);

const extractGreenhouseJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-company-name]',
    '.job__company',
    '.company-name'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-job-title]',
    '.job__title',
    'main h1',
    'h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-job-location]',
    '.job__location',
    '.job-location',
    '.location'
  ].join(',')));
  const description = visibleLongText(doc, [
    '[data-job-description]',
    '.job__description',
    '.job-description',
    '[itemprop="description"]'
  ]);
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'Greenhouse',
    ...(description ? { description } : {})
  };
};

export const greenhouseAdapter = Object.freeze({
  id: 'greenhouse',
  sourceName: 'Greenhouse',
  matches: isGreenhousePage,
  getApplicationRoot: getGreenhouseApplicationRoot,
  getDiscoveryContexts: (doc) => applicationContexts({
    applicationRoot: getGreenhouseApplicationRoot(doc)
  }),
  collectCandidates: (doc) => collectControls(getGreenhouseApplicationRoot(doc)),
  getFieldContainer: fieldHelpers.getFieldContainer,
  getSupplementalLabel: fieldHelpers.getSupplementalLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractGreenhouseJobMetadata
});
