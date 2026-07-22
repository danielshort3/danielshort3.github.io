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

const COMMON_APPLICATION_SELECTORS = [
  '[data-application-form]',
  '[data-candidate-application]',
  '[data-testid="application-form"]',
  '[data-testid*="candidate-application" i]',
  '[data-automation-id="applicationForm"]',
  '[data-automation-id*="candidateApplication" i]',
  '[data-qa="application-form"]',
  '[role="form"][aria-label*="application" i]',
  '.application-form',
  '#application-form'
];

const COMMON_ACCOUNT_SELECTORS = [
  '[data-testid*="sign-in" i]',
  '[data-testid*="login" i]',
  '[data-automation-id*="signIn" i]',
  '[data-automation-id*="createAccount" i]',
  'form[action*="login" i]',
  'form[action*="signin" i]',
  'form[action*="register" i]'
];

const COMMON_CONTAINER_SELECTORS = [
  '[data-field-id]',
  '[data-question-id]',
  '[data-testid*="field" i]',
  '[data-qa*="field" i]',
  '[data-automation-id*="field" i]',
  '.application-question',
  '.application-field',
  '.form-field',
  '.form-group',
  '.field',
  'fieldset',
  '[role="group"]'
];

const COMMON_LABEL_SELECTORS = [
  ':scope > label',
  ':scope > legend',
  ':scope > .field-label',
  ':scope > .question-label',
  ':scope > .control-label',
  '[data-testid*="label" i]',
  '[data-qa*="label" i]',
  '[data-automation-id*="label" i]'
];

const COMMON_TITLE_SELECTORS = [
  '[data-job-title]',
  '[data-testid*="job-title" i]',
  '[data-automation-id*="jobTitle" i]',
  '.job-title',
  'main h1',
  'h1'
];

const COMMON_COMPANY_SELECTORS = [
  '[data-company-name]',
  '[data-testid*="company-name" i]',
  '[data-automation-id*="companyName" i]',
  '.company-name',
  '.job-company'
];

const COMMON_LOCATION_SELECTORS = [
  '[data-job-location]',
  '[data-testid*="job-location" i]',
  '[data-automation-id*="location" i]',
  '.job-location',
  '.location'
];

const COMMON_DESCRIPTION_SELECTORS = [
  '[data-job-description]',
  '[data-testid*="job-description" i]',
  '[data-automation-id*="description" i]',
  '.job-description',
  '[itemprop="description"]'
];

const hostMatchesSuffix = (hostname, suffix) => hostname === suffix
  || hostname.endsWith('.' + suffix);

const firstText = (doc, selectors) => {
  for (const selector of selectors) {
    const value = textFromNode(safeQuery(doc, selector));
    if (value) return value;
  }
  return '';
};

const createProfileAdapter = (profile) => {
  const applicationSelectors = Array.from(new Set([
    ...profile.applicationSelectors,
    ...COMMON_APPLICATION_SELECTORS
  ]));
  const accountSelectors = Array.from(new Set([
    ...(profile.accountSelectors || []),
    ...COMMON_ACCOUNT_SELECTORS
  ]));
  const fieldHelpers = createFieldHelpers({
    containerSelector: [
      ...(profile.containerSelectors || []),
      ...COMMON_CONTAINER_SELECTORS
    ].join(','),
    labelSelectors: [
      ...(profile.labelSelectors || []),
      ...COMMON_LABEL_SELECTORS
    ]
  });

  const getApplicationRoot = (doc) => selectApplicationRoot(doc, applicationSelectors);
  const getAccountGate = (doc) => getApplicationRoot(doc)
    ? null
    : findAccountGate(doc, accountSelectors);

  return Object.freeze({
    id: profile.id,
    sourceName: profile.sourceName,
    matches: ({ doc, url } = {}) => {
      const hostname = hostnameFromUrl(url);
      if (profile.hostSuffixes.some((suffix) => hostMatchesSuffix(hostname, suffix))) return true;
      return profile.markerSelectors.some((selector) => Boolean(safeQuery(doc, selector)));
    },
    getApplicationRoot,
    getDiscoveryContexts: (doc) => applicationContexts({
      applicationRoot: getApplicationRoot(doc),
      accountGate: getAccountGate(doc)
    }),
    collectCandidates: (doc) => collectControls(getApplicationRoot(doc), doc),
    getFieldContainer: fieldHelpers.getFieldContainer,
    getSupplementalLabel: fieldHelpers.getSupplementalLabel,
    getNearbyText: genericAdapter.getNearbyText,
    extractJobMetadata: (doc, rawUrl) => {
      const fallback = extractGenericJobMetadata(doc, rawUrl);
      const title = firstText(doc, [
        ...(profile.titleSelectors || []),
        ...COMMON_TITLE_SELECTORS
      ]);
      const company = firstText(doc, [
        ...(profile.companySelectors || []),
        ...COMMON_COMPANY_SELECTORS
      ]);
      const location = firstText(doc, [
        ...(profile.locationSelectors || []),
        ...COMMON_LOCATION_SELECTORS
      ]);
      const description = visibleLongText(doc, [
        ...(profile.descriptionSelectors || []),
        ...COMMON_DESCRIPTION_SELECTORS
      ]);
      return {
        ...fallback,
        title: normalizeText(title || fallback.title),
        company: normalizeText(company || fallback.company),
        location: normalizeText(location || fallback.location),
        source: profile.sourceName,
        ...(description ? { description } : {})
      };
    }
  });
};

export const icimsAdapter = createProfileAdapter({
  id: 'icims',
  sourceName: 'iCIMS',
  hostSuffixes: ['icims.com', 'icims.eu'],
  markerSelectors: [
    '[data-icims-application]',
    '.iCIMS_Application',
    'form[action*="icims.com" i]',
    'form[action*="icims.eu" i]'
  ],
  applicationSelectors: [
    '[data-icims-application]',
    '.iCIMS_Application',
    '#iCIMS_Application',
    'form[action*="icims" i][action*="apply" i]'
  ],
  containerSelectors: ['.iCIMS_Field', '.iCIMS_Question'],
  labelSelectors: ['.iCIMS_Label']
});

export const oracleRecruitingAdapter = createProfileAdapter({
  id: 'oracle-recruiting',
  sourceName: 'Oracle Recruiting / Taleo',
  hostSuffixes: ['taleo.net', 'oraclecloud.com'],
  markerSelectors: [
    '[data-qa="candidate-application"]',
    '[data-testid="candidate-application"]',
    'form[action*="taleo.net" i]',
    '[data-bind*="candidateApplication" i]'
  ],
  applicationSelectors: [
    '[data-qa="candidate-application"]',
    '[data-testid="candidate-application"]',
    '[id*="candidateApplication" i]',
    'form[action*="taleo.net" i]',
    'form[action*="/CandidateExperience/" i]'
  ],
  containerSelectors: ['[data-oj-context]', '.candidate-application-field'],
  labelSelectors: ['oj-label', '.candidate-application-label']
});

export const bambooHrAdapter = createProfileAdapter({
  id: 'bamboohr',
  sourceName: 'BambooHR',
  hostSuffixes: ['bamboohr.com'],
  markerSelectors: [
    '[data-bamboo-application]',
    'form[action*="bamboohr.com" i][action*="jobs" i]',
    '#applicationForm'
  ],
  applicationSelectors: [
    '[data-bamboo-application]',
    '#applicationForm',
    'form[action*="/jobs/" i][action*="application" i]'
  ],
  containerSelectors: ['.BambooHR-ATS-board__field', '[data-bamboo-field]']
});

export const adpAdapter = createProfileAdapter({
  id: 'adp',
  sourceName: 'ADP Recruiting',
  hostSuffixes: [
    'workforcenow.adp.com',
    'myjobs.adp.com',
    'recruiting.adp.com',
    'careercenter.adp.com'
  ],
  markerSelectors: [
    '[data-adp-application]',
    '[data-automation-id="applicationForm"]',
    'form[action*="adp.com" i][action*="apply" i]'
  ],
  applicationSelectors: [
    '[data-adp-application]',
    '[data-automation-id="applicationForm"]',
    '[data-testid="application-form"]',
    'form[action*="/apply" i]'
  ],
  containerSelectors: ['adp-form-field', '[data-adp-field]']
});

export const ukgAdapter = createProfileAdapter({
  id: 'ukg',
  sourceName: 'UKG Recruiting',
  hostSuffixes: ['ultipro.com', 'ultipro.ca', 'ukg.com'],
  markerSelectors: [
    '[data-ukg-application]',
    '[data-component="application-form"]',
    'form[action*="ultipro" i][action*="apply" i]'
  ],
  applicationSelectors: [
    '[data-ukg-application]',
    '[data-component="application-form"]',
    '[data-testid="application-form"]',
    'form[action*="/apply" i]'
  ],
  containerSelectors: ['[data-ukg-field]', '.application-form-field']
});

export const jobviteAdapter = createProfileAdapter({
  id: 'jobvite',
  sourceName: 'Jobvite',
  hostSuffixes: ['jobvite.com'],
  markerSelectors: [
    '.jv-application',
    'form#jvform',
    'form[action*="jobvite.com" i][action*="apply" i]'
  ],
  applicationSelectors: [
    '.jv-application',
    'form#jvform',
    'form[action*="jobvite.com" i][action*="apply" i]'
  ],
  containerSelectors: ['.jv-form-field', '.jv-application-question']
});

export const jazzHrAdapter = createProfileAdapter({
  id: 'jazzhr',
  sourceName: 'JazzHR',
  hostSuffixes: ['applytojob.com', 'jazz.co'],
  markerSelectors: [
    '[data-jazzhr-application]',
    'form#application_form',
    'form[action*="applytojob.com" i]'
  ],
  applicationSelectors: [
    '[data-jazzhr-application]',
    'form#application_form',
    'form[action*="applytojob.com" i]'
  ],
  containerSelectors: ['.form-group', '.application-question']
});

export const recruiteeAdapter = createProfileAdapter({
  id: 'recruitee',
  sourceName: 'Recruitee',
  hostSuffixes: ['recruitee.com'],
  markerSelectors: [
    '[data-recruitee-application]',
    'form[action*="recruitee.com" i][action*="apply" i]',
    'form[action*="/c/" i][action*="apply" i]'
  ],
  applicationSelectors: [
    '[data-recruitee-application]',
    '[data-testid="application-form"]',
    'form[action*="/c/" i][action*="apply" i]'
  ],
  containerSelectors: ['[data-recruitee-field]', '.application-field']
});

export const pinpointAdapter = createProfileAdapter({
  id: 'pinpoint',
  sourceName: 'Pinpoint',
  hostSuffixes: ['pinpointhq.com'],
  markerSelectors: [
    '[data-pinpoint-application]',
    'form[action*="pinpointhq.com" i][action*="application" i]'
  ],
  applicationSelectors: [
    '[data-pinpoint-application]',
    '[data-testid="application-form"]',
    'form[action*="/postings/" i][action*="applications" i]'
  ],
  containerSelectors: ['[data-pinpoint-field]', '.application-field']
});

export const successFactorsAdapter = createProfileAdapter({
  id: 'successfactors',
  sourceName: 'SAP SuccessFactors',
  hostSuffixes: [
    'successfactors.com',
    'successfactors.eu',
    'successfactors.cn',
    'jobs.sap.com'
  ],
  markerSelectors: [
    '[data-successfactors-application]',
    '[data-automation-id="applicationForm"]',
    'form[action*="successfactors" i][action*="apply" i]'
  ],
  applicationSelectors: [
    '[data-successfactors-application]',
    '[data-automation-id="applicationForm"]',
    '#applicationForm',
    'form[action*="career" i][action*="apply" i]'
  ],
  containerSelectors: ['[data-successfactors-field]', '.application-field']
});

export const PLATFORM_PROFILE_ADAPTERS = Object.freeze([
  icimsAdapter,
  oracleRecruitingAdapter,
  bambooHrAdapter,
  adpAdapter,
  ukgAdapter,
  jobviteAdapter,
  jazzHrAdapter,
  recruiteeAdapter,
  pinpointAdapter,
  successFactorsAdapter
]);
