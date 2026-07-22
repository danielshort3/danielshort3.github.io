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

const ASHBY_HOST_RE = /(^|\.)jobs\.ashbyhq\.com$/iu;
const ASHBY_APPLICATION_SELECTORS = [
  '[data-ashby-application-form]',
  '[data-testid="application-form"]',
  'form[data-testid="application-form"]',
  'form[action*="/application"]',
  '#form'
];

const fieldHelpers = createFieldHelpers({
  containerSelector: [
    '[data-field-id]',
    '[data-testid="application-form-field"]',
    '[data-testid*="field"]',
    '.application-question',
    '.field',
    'fieldset'
  ].join(','),
  labelSelectors: [
    ':scope > label',
    ':scope > legend',
    '[data-testid="field-label"]',
    '[data-testid*="label"]'
  ]
});

export const isAshbyPage = ({ doc, url } = {}) => ASHBY_HOST_RE.test(hostnameFromUrl(url))
  || Boolean(safeQuery(doc, '[data-ashby-application-form],form[action*="ashbyhq.com"][action*="application"]'));

const getAshbyApplicationRoot = (doc) => selectApplicationRoot(doc, ASHBY_APPLICATION_SELECTORS);

const extractAshbyJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-testid="company-name"]',
    '[data-company-name]',
    '.company-name'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-testid="job-title"]',
    '[data-job-title]',
    'main h1',
    'h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-testid="job-location"]',
    '[data-job-location]',
    '.job-location',
    '.location'
  ].join(',')));
  const description = visibleLongText(doc, [
    '[data-testid="job-description"]',
    '[data-job-description]',
    '.job-description',
    '[itemprop="description"]'
  ]);
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'Ashby',
    ...(description ? { description } : {})
  };
};

export const ashbyAdapter = Object.freeze({
  id: 'ashby',
  sourceName: 'Ashby',
  matches: isAshbyPage,
  getApplicationRoot: getAshbyApplicationRoot,
  getDiscoveryContexts: (doc) => applicationContexts({
    applicationRoot: getAshbyApplicationRoot(doc)
  }),
  collectCandidates: (doc) => collectControls(getAshbyApplicationRoot(doc)),
  getFieldContainer: fieldHelpers.getFieldContainer,
  getSupplementalLabel: fieldHelpers.getSupplementalLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractAshbyJobMetadata
});
