import {
  adapterInternals,
  extractGenericJobMetadata,
  genericAdapter
} from './generic.js';
import {
  applicationContexts,
  collectControls,
  createFieldHelpers,
  findAccountGate,
  hostnameFromUrl,
  selectApplicationRoot
} from './shared.js';

const { normalizeText, safeQuery, textFromNode, visibleLongText } = adapterInternals;

const WORKDAY_HOST_RE = /(^|\.)myworkdayjobs\.com$/iu;
const WORKDAY_APPLICATION_SELECTORS = [
  '[data-automation-id="applicationForm"]',
  '[data-automation-id="jobApplication"]',
  '[data-automation-id="applyFlow"]',
  '[data-automation-id="applicationPage"]',
  '[data-automation-id="applicationSteps"]',
  'form[data-automation-id*="application"]'
];
const WORKDAY_ACCOUNT_SELECTORS = [
  '[data-automation-id="signInPage"]',
  '[data-automation-id="signInForm"]',
  '[data-automation-id="createAccount"]',
  '[data-automation-id="createAccountForm"]',
  'form[action*="login"]',
  'form[action*="signin"]'
];

const fieldHelpers = createFieldHelpers({
  containerSelector: [
    '[data-automation-id="formField"]',
    '[data-automation-id*="Question"]',
    '[data-automation-id*="question"]',
    '.field',
    'fieldset'
  ].join(','),
  labelSelectors: [
    ':scope > label',
    ':scope > legend',
    '[data-automation-id="formLabel"]',
    '[data-automation-id="prompt"]'
  ]
});

export const isWorkdayPage = ({ doc, url } = {}) => WORKDAY_HOST_RE.test(hostnameFromUrl(url))
  || Boolean(safeQuery(doc, '[data-automation-id="jobPostingPage"],[data-automation-id="applicationForm"]'));

const getWorkdayApplicationRoot = (doc) => selectApplicationRoot(doc, WORKDAY_APPLICATION_SELECTORS);
const getWorkdayAccountGate = (doc) => getWorkdayApplicationRoot(doc)
  ? null
  : findAccountGate(doc, WORKDAY_ACCOUNT_SELECTORS);

const extractWorkdayJobMetadata = (doc, rawUrl) => {
  const fallback = extractGenericJobMetadata(doc, rawUrl);
  const company = textFromNode(safeQuery(doc, [
    '[data-automation-id="company"]',
    '[data-automation-id="companyName"]',
    '[data-company-name]'
  ].join(',')));
  const title = textFromNode(safeQuery(doc, [
    '[data-automation-id="jobPostingHeader"] h1',
    '[data-automation-id="jobPostingHeader"] h2',
    '[data-automation-id="jobTitle"]',
    '[data-job-title]',
    'main h1'
  ].join(',')));
  const location = textFromNode(safeQuery(doc, [
    '[data-automation-id="locations"]',
    '[data-automation-id="jobPostingLocation"]',
    '[data-job-location]'
  ].join(',')));
  const description = visibleLongText(doc, [
    '[data-automation-id="jobPostingDescription"]',
    '[data-job-description]',
    '[itemprop="description"]'
  ]);
  return {
    ...fallback,
    company: normalizeText(company || fallback.company),
    title: normalizeText(title || fallback.title),
    location: normalizeText(location || fallback.location),
    source: 'Workday',
    ...(description ? { description } : {})
  };
};

export const workdayAdapter = Object.freeze({
  id: 'workday',
  sourceName: 'Workday',
  matches: isWorkdayPage,
  getApplicationRoot: getWorkdayApplicationRoot,
  getDiscoveryContexts: (doc) => applicationContexts({
    applicationRoot: getWorkdayApplicationRoot(doc),
    accountGate: getWorkdayAccountGate(doc)
  }),
  collectCandidates: (doc) => collectControls(getWorkdayApplicationRoot(doc)),
  getFieldContainer: fieldHelpers.getFieldContainer,
  getSupplementalLabel: fieldHelpers.getSupplementalLabel,
  getNearbyText: genericAdapter.getNearbyText,
  extractJobMetadata: extractWorkdayJobMetadata
});
