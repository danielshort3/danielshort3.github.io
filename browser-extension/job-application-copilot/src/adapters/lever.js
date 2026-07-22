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

const LEVER_HOST_RE = /(^|\.)jobs(?:\.eu)?\.lever\.co$/iu;
const LEVER_APPLICATION_SELECTORS = [
  'form.application-form',
  'form#application-form',
  '[data-qa="application-form"]',
  'form[action*="/apply"]'
];

const fieldHelpers = createFieldHelpers({
  containerSelector: [
    '.application-question',
    '.application-field',
    '.application-additional',
    '.form-field',
    '.field',
    'fieldset'
  ].join(','),
  labelSelectors: [
    ':scope > .application-label',
    ':scope > .field-label',
    ':scope > label',
    ':scope > legend',
    '.application-label'
  ]
});

export const isLeverPage = ({ doc, url } = {}) => LEVER_HOST_RE.test(hostnameFromUrl(url))
  || Boolean(safeQuery(doc, 'form[action*="lever.co"][action*="apply"]'));

const getLeverApplicationRoot = (doc) => selectApplicationRoot(doc, LEVER_APPLICATION_SELECTORS);

const extractLeverJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-company-name]',
    '.posting-company',
    '.company-name'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-job-title]',
    '.posting-headline h2',
    '.posting-header h2',
    'main h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-job-location]',
    '.posting-categories .location',
    '.posting-location',
    '.location'
  ].join(',')));
  const description = visibleLongText(doc, [
    '[data-job-description]',
    '.posting-description',
    '.posting-page .content',
    '.section-wrapper.page-full-width'
  ]);
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'Lever',
    ...(description ? { description } : {})
  };
};

export const leverAdapter = Object.freeze({
  id: 'lever',
  sourceName: 'Lever',
  matches: isLeverPage,
  getApplicationRoot: getLeverApplicationRoot,
  getDiscoveryContexts: (doc) => applicationContexts({
    applicationRoot: getLeverApplicationRoot(doc)
  }),
  collectCandidates: (doc) => collectControls(getLeverApplicationRoot(doc)),
  getFieldContainer: fieldHelpers.getFieldContainer,
  getSupplementalLabel: fieldHelpers.getSupplementalLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractLeverJobMetadata
});
