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

const SMARTRECRUITERS_HOST_RE = /(^|\.)(?:jobs|careers)\.smartrecruiters\.com$/iu;
const SMARTRECRUITERS_APPLICATION_SELECTORS = [
  'smartrecruiters-application',
  'sr-application',
  '[data-smartrecruiters-application]',
  '[data-testid="application-form"]',
  '[data-automation-id="application-form"]',
  'form[action*="smartrecruiters.com"][action*="apply"]'
];

const fieldHelpers = createFieldHelpers({
  containerSelector: [
    '[data-testid="application-field"]',
    '[data-automation-id="application-field"]',
    'sr-form-field',
    '.application-field',
    '.field',
    'fieldset'
  ].join(','),
  labelSelectors: [
    ':scope > label',
    ':scope > legend',
    '[data-testid="field-label"]',
    '[data-automation-id="field-label"]'
  ]
});

export const isSmartRecruitersPage = ({ doc, url } = {}) => SMARTRECRUITERS_HOST_RE.test(hostnameFromUrl(url))
  || Boolean(safeQuery(doc, 'smartrecruiters-application,[data-smartrecruiters-application]'));

const getSmartRecruitersApplicationRoot = (doc) => {
  const explicitRoot = selectApplicationRoot(doc, SMARTRECRUITERS_APPLICATION_SELECTORS);
  if (explicitRoot) return explicitRoot;
  const main = safeQuery(doc, 'main');
  if (!main) return null;
  return safeQuery(main, [
    'spl-input',
    'spl-phone-field',
    'spl-textarea',
    'spl-autocomplete',
    'spl-form-field'
  ].join(','))
    ? main
    : null;
};

const extractSmartRecruitersJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-testid="company-name"]',
    '[itemprop="hiringOrganization"] [itemprop="name"]',
    '[data-company-name]',
    '.company-name'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-testid="job-title"]',
    '[itemprop="title"]',
    '[data-job-title]',
    'main h1',
    'h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-testid="job-location"]',
    '[itemprop="jobLocation"]',
    '[data-job-location]',
    '.job-location'
  ].join(',')));
  const description = visibleLongText(doc, [
    '[data-testid="job-description"]',
    '[itemprop="description"]',
    '[data-job-description]',
    '.job-description'
  ]);
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'SmartRecruiters',
    ...(description ? { description } : {})
  };
};

export const smartRecruitersAdapter = Object.freeze({
  id: 'smartrecruiters',
  sourceName: 'SmartRecruiters',
  matches: isSmartRecruitersPage,
  getApplicationRoot: getSmartRecruitersApplicationRoot,
  getDiscoveryContexts: (doc) => applicationContexts({
    applicationRoot: getSmartRecruitersApplicationRoot(doc)
  }),
  collectCandidates: (doc) => collectControls(getSmartRecruitersApplicationRoot(doc)),
  getFieldContainer: fieldHelpers.getFieldContainer,
  getSupplementalLabel: fieldHelpers.getSupplementalLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractSmartRecruitersJobMetadata
});
