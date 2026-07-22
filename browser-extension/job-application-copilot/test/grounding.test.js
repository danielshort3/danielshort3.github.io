import assert from 'node:assert/strict';
import test from 'node:test';
import {
  GROUNDING_REASONS,
  GroundingError,
  enforceGroundingAcceptance,
  resolveCitationCards
} from '../src/grounding/postprocessor.js';

const fields = [{
  fieldId: 'name',
  label: 'Full name',
  type: 'text',
  options: [],
  riskClass: 'F1_VERIFIED'
}, {
  fieldId: 'phone',
  label: 'Phone',
  type: 'tel',
  options: [],
  riskClass: 'F1_VERIFIED'
}, {
  fieldId: 'start-date',
  label: 'Available date',
  type: 'date',
  options: [],
  riskClass: 'F1_VERIFIED'
}, {
  fieldId: 'skill',
  label: 'Primary skill',
  type: 'select-one',
  options: ['Python', 'R'],
  riskClass: 'F1_VERIFIED'
}, {
  fieldId: 'authorized',
  label: 'Authorized to work',
  type: 'checkbox',
  options: [],
  riskClass: 'F1_VERIFIED'
}, {
  fieldId: 'summary',
  label: 'Professional summary',
  type: 'textarea',
  options: [],
  riskClass: 'F2_REVIEW'
}];

const request = { pageId: 'page-1', urlHash: 'url-1', domRevision: 1, fields };
const citations = [{
  citationId: 'c1',
  sourceRole: 'candidate_evidence',
  documentId: 'doc:resume',
  documentVersion: 1,
  chunkId: 'chunk:resume:1',
  quoteHash: 'hash-1',
  text: 'Daniel Short\nPhone: 303-555-1212\nAvailable July 18, 2026\nSkills: Python\nAuthorized to work: Yes\nBS degree',
  locator: { pageStart: 1, paragraphStart: 1 }
}, {
  citationId: 'c2',
  sourceRole: 'candidate_evidence',
  documentId: 'doc:resume',
  documentVersion: 1,
  chunkId: 'chunk:resume:2',
  quoteHash: 'hash-2',
  text: 'Delivered 12 analytics dashboards in 2025. Contact analyst@example.com or https://portfolio.example/work. Phone 303-555-1212.',
  locator: { pageStart: 2, paragraphStart: 1 }
}, {
  citationId: 'c3',
  sourceRole: 'job_requirement',
  documentId: 'doc:job',
  documentVersion: 1,
  chunkId: 'chunk:job:1',
  quoteHash: 'hash-3',
  text: 'The team supports 40 offices.',
  locator: { paragraphStart: 3 }
}, {
  citationId: 'c4',
  sourceRole: 'style_example',
  documentId: 'doc:style',
  documentVersion: 1,
  chunkId: 'chunk:style:1',
  quoteHash: 'hash-4',
  text: 'Write in a concise first-person voice.',
  locator: { paragraphStart: 1 }
}, {
  citationId: 'c5',
  sourceRole: 'user_verified',
  documentId: 'doc:profile',
  documentVersion: 4,
  chunkId: 'chunk:profile:1',
  quoteHash: 'hash-5',
  text: 'Preferred name: Daniel Short.',
  locator: { section: 'Profile' }
}];

const evidence = {
  citations,
  byField: {
    name: ['c1'],
    phone: ['c1'],
    'start-date': ['c1'],
    skill: ['c1'],
    authorized: ['c1'],
    summary: ['c2', 'c3', 'c4']
  }
};

const proposal = (overrides = {}) => ({
  field_id: 'name',
  action: 'fill',
  confidence: 'high',
  risk_class: 'F1_VERIFIED',
  value_type: 'text',
  value: 'Daniel Short',
  selected_values: [],
  checked: false,
  citation_ids: ['c1'],
  short_rationale: 'Supported by candidate evidence.',
  abstain_reason: '',
  ...overrides
});

const processOne = (candidate, suppliedEvidence = evidence) => enforceGroundingAcceptance({
  output: { page_id: 'page-1', proposals: [candidate] },
  request,
  evidence: suppliedEvidence
}).proposals[0];

const assertNonFillable = (result, reason) => {
  assert.equal(result.action, 'ask_user');
  assert.equal(result.confidence, 'needs_input');
  assert.equal(result.value_type, 'none');
  assert.equal(result.value, '');
  assert.deepEqual(result.selected_values, []);
  assert.equal(result.checked, false);
  assert.deepEqual(result.citation_ids, []);
  assert.equal(result.abstain_reason, reason);
};

test('citation cards resolve only exact field-scoped stored evidence', () => {
  const cards = resolveCitationCards({ citationIds: ['c1'], fieldId: 'name', evidence });
  assert.deepEqual(cards, [{
    citationId: 'c1',
    sourceRole: 'candidate_evidence',
    documentId: 'doc:resume',
    documentVersion: 1,
    chunkId: 'chunk:resume:1',
    quoteHash: 'hash-1',
    text: citations[0].text,
    locator: { pageStart: 1, paragraphStart: 1 }
  }]);
  cards[0].locator.pageStart = 99;
  assert.equal(citations[0].locator.pageStart, 1);
  assert.throws(() => resolveCitationCards({ citationIds: ['c5'], fieldId: 'name', evidence }), GroundingError);
  assert.throws(() => resolveCitationCards({ citationIds: ['c1', 'c1'], fieldId: 'name', evidence }), /unique exact strings/u);
  assert.throws(() => resolveCitationCards({ citationIds: ['C1'], fieldId: 'name', evidence }), GroundingError);
});

test('valid F1 exact and canonical values are preserved without mutating input', () => {
  const input = proposal();
  const before = structuredClone(input);
  assert.deepEqual(processOne(input), input);
  assert.deepEqual(input, before);

  assert.equal(processOne(proposal({
    field_id: 'phone',
    value: '(303) 555-1212',
    citation_ids: ['c1']
  })).action, 'fill');
  assert.equal(processOne(proposal({
    field_id: 'start-date',
    value: '2026-07-18',
    citation_ids: ['c1']
  })).action, 'fill');
  assert.equal(processOne(proposal({
    field_id: 'skill',
    value_type: 'selected_values',
    value: '',
    selected_values: ['Python'],
    citation_ids: ['c1']
  })).action, 'fill');
  assert.equal(processOne(proposal({
    field_id: 'authorized',
    value_type: 'checked',
    value: '',
    checked: true,
    citation_ids: ['c1']
  })).action, 'fill');
});

test('invalid field citations and ineligible source roles fail closed', () => {
  assertNonFillable(processOne(proposal({ citation_ids: ['c5'] })), GROUNDING_REASONS.INVALID_CITATIONS);
  assertNonFillable(processOne(proposal({ citation_ids: ['c1', 'c1'] })), GROUNDING_REASONS.INVALID_CITATIONS);
  const styleOnlyEvidence = { ...evidence, byField: { ...evidence.byField, name: ['c4'] } };
  assertNonFillable(processOne(proposal({ citation_ids: ['c4'] }), styleOnlyEvidence), GROUNDING_REASONS.MISSING_CANDIDATE_EVIDENCE);
  const noFieldEvidence = { ...evidence, byField: { ...evidence.byField } };
  delete noFieldEvidence.byField.name;
  assertNonFillable(processOne(proposal(), noFieldEvidence), GROUNDING_REASONS.INVALID_CITATIONS);
});

test('fabricated F1 values and incoherent value shapes fail closed without quoting the model', () => {
  const fabricated = processOne(proposal({ value: 'Made Up Person', short_rationale: 'The model says "Made Up Person".' }));
  assertNonFillable(fabricated, GROUNDING_REASONS.UNSUPPORTED_EXACT_VALUE);
  assert.doesNotMatch(fabricated.short_rationale, /Made Up Person/u);
  const incoherent = processOne(proposal({ value_type: 'selected_values', value: 'Python', selected_values: ['Python'] }));
  assertNonFillable(incoherent, GROUNDING_REASONS.INCOHERENT_VALUE);
  const unsupportedOption = processOne(proposal({
    field_id: 'skill',
    value_type: 'selected_values',
    value: '',
    selected_values: ['Rust']
  }));
  assertNonFillable(unsupportedOption, GROUNDING_REASONS.UNSUPPORTED_EXACT_VALUE);
});

test('F2 prose requires candidate evidence and every concrete detail in cited excerpts', () => {
  const grounded = processOne(proposal({
    field_id: 'summary',
    confidence: 'high',
    risk_class: 'F2_REVIEW',
    value: 'I delivered 12 analytics dashboards in 2025. Contact analyst@example.com or https://portfolio.example/work, or call (303) 555-1212.',
    citation_ids: ['c2']
  }));
  assert.equal(grounded.action, 'fill');
  assert.equal(grounded.confidence, 'review');

  const unsupportedNumber = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I delivered 13 analytics dashboards in 2025.',
    citation_ids: ['c2']
  }));
  assertNonFillable(unsupportedNumber, GROUNDING_REASONS.UNSUPPORTED_CONCRETE_DETAIL);

  const mixedSources = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'My analytics background can support the team across 40 offices.',
    citation_ids: ['c2', 'c3']
  }));
  assert.equal(mixedSources.action, 'fill');

  const wrongRoles = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I can support 40 offices.',
    citation_ids: ['c3', 'c4']
  }));
  assertNonFillable(wrongRoles, GROUNDING_REASONS.MISSING_CANDIDATE_EVIDENCE);
});

test('unsupported credentials are treated as concrete candidate claims', () => {
  const result = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I hold a PhD in analytics.',
    citation_ids: ['c1']
  }), { ...evidence, byField: { ...evidence.byField, summary: ['c1'] } });
  assertNonFillable(result, GROUNDING_REASONS.UNSUPPORTED_CONCRETE_DETAIL);
});

test('unrelated candidate evidence cannot ground a fabricated Kubernetes expertise claim', () => {
  const result = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I am an expert in Kubernetes.',
    citation_ids: ['c2']
  }));
  assertNonFillable(result, GROUNDING_REASONS.UNSUPPORTED_SALIENT_CLAIM);
});

test('pronoun-free candidate claim fragments require related candidate evidence', () => {
  const unsupported = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'Experienced in Kubernetes and distributed systems.',
    citation_ids: ['c2']
  }));
  assertNonFillable(unsupported, GROUNDING_REASONS.UNSUPPORTED_SALIENT_CLAIM);

  const mixed = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'Delivered 12 analytics dashboards in 2025. Experienced in Kubernetes and distributed systems.',
    citation_ids: ['c2']
  }));
  assertNonFillable(mixed, GROUNDING_REASONS.UNSUPPORTED_SALIENT_CLAIM);
});

test('grounded pronoun-free candidate prose remains fillable for review', () => {
  const grounded = processOne(proposal({
    field_id: 'summary',
    confidence: 'high',
    risk_class: 'F2_REVIEW',
    value: 'Delivered 12 analytics dashboards in 2025.',
    citation_ids: ['c2']
  }));
  assert.equal(grounded.action, 'fill');
  assert.equal(grounded.confidence, 'review');

  const kubernetesCitation = {
    citationId: 'c6',
    sourceRole: 'candidate_evidence',
    documentId: 'doc:resume',
    documentVersion: 1,
    chunkId: 'chunk:resume:3',
    quoteHash: 'hash-6',
    text: 'Experienced in Kubernetes and distributed systems.',
    locator: { pageStart: 3, paragraphStart: 1 }
  };
  const supportedEvidence = {
    citations: [...citations, kubernetesCitation],
    byField: { ...evidence.byField, summary: ['c6'] }
  };
  const supportedFragment = processOne(proposal({
    field_id: 'summary',
    confidence: 'high',
    risk_class: 'F2_REVIEW',
    value: 'Experienced in Kubernetes and distributed systems.',
    citation_ids: ['c6']
  }), supportedEvidence);
  assert.equal(supportedFragment.action, 'fill');
  assert.equal(supportedFragment.confidence, 'review');

  const tenseCitation = {
    citationId: 'c7',
    sourceRole: 'candidate_evidence',
    documentId: 'doc:resume',
    documentVersion: 1,
    chunkId: 'chunk:resume:4',
    quoteHash: 'hash-7',
    text: 'Built forecasting tools and executive dashboards.',
    locator: { pageStart: 4, paragraphStart: 1 }
  };
  const tenseEvidence = {
    citations: [...citations, tenseCitation],
    byField: { ...evidence.byField, summary: ['c7'] }
  };
  const naturalTense = processOne(proposal({
    field_id: 'summary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I have experience building forecasting tools and executive dashboards.',
    citation_ids: ['c7']
  }), tenseEvidence);
  assert.equal(naturalTense.action, 'fill');
  assert.equal(naturalTense.confidence, 'review');
});
test('salary expectations may use an exact posted range without treating it as candidate history', () => {
  const salaryField = {
    fieldId: 'salary',
    label: 'What are your salary and additional compensation expectations?',
    type: 'textarea',
    options: [],
    riskClass: 'F2_REVIEW'
  };
  const salaryEvidence = {
    citations: [{
      citationId: 'salary-job',
      sourceRole: 'job_requirement',
      documentId: 'live-job:salary',
      documentVersion: 1,
      chunkId: 'chunk:salary',
      quoteHash: 'salary-hash',
      text: 'The posted salary range is $65,000 - $85,000 USD.',
      locator: { section: 'Live job posting' }
    }],
    byField: { salary: ['salary-job'] }
  };
  const salaryDraft = proposal({
    field_id: 'salary',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: 'I am open to discussing compensation within the posted range of $65,000 - $85,000 USD, taking the full compensation package into account.',
    citation_ids: ['salary-job']
  });
  const grounded = enforceGroundingAcceptance({
    output: { page_id: 'salary-page', proposals: [salaryDraft] },
    request: { pageId: 'salary-page', fields: [salaryField] },
    evidence: salaryEvidence
  }).proposals[0];
  assert.equal(grounded.action, 'fill');
  assert.equal(grounded.confidence, 'review');

  const outsideRange = enforceGroundingAcceptance({
    output: {
      page_id: 'salary-page',
      proposals: [proposal({
        field_id: 'salary',
        confidence: 'review',
        risk_class: 'F2_REVIEW',
        value: 'I expect $95,000 USD.',
        citation_ids: ['salary-job']
      })]
    },
    request: { pageId: 'salary-page', fields: [salaryField] },
    evidence: salaryEvidence
  }).proposals[0];
  assert.equal(outsideRange.action, 'ask_user');

  const unsupportedBonus = enforceGroundingAcceptance({
    output: {
      page_id: 'salary-page',
      proposals: [proposal({
        field_id: 'salary',
        confidence: 'review',
        risk_class: 'F2_REVIEW',
        value: 'I am open to $65,000 - $85,000 USD plus an equity bonus.',
        citation_ids: ['salary-job']
      })]
    },
    request: { pageId: 'salary-page', fields: [salaryField] },
    evidence: salaryEvidence
  }).proposals[0];
  assert.equal(unsupportedBonus.action, 'ask_user');
});

test('experience drafts may disclose an evidence gap and cite adjacent experience without inventing tenure', () => {
  const experienceField = {
    fieldId: 'fintech',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    type: 'textarea',
    options: [],
    riskClass: 'F2_REVIEW'
  };
  const experienceEvidence = {
    citations: [{
      citationId: 'resume-adjacent',
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      documentVersion: 1,
      chunkId: 'resume-adjacent-chunk',
      quoteHash: 'resume-adjacent-hash',
      text: 'Analytics dashboards and SQL reporting.',
      locator: { pageStart: 1 }
    }, {
      citationId: 'job-domain',
      sourceRole: 'job_requirement',
      documentId: 'live-job:fintech',
      documentVersion: 1,
      chunkId: 'job-domain-chunk',
      quoteHash: 'job-domain-hash',
      text: 'Fintech and payments experience is requested.',
      locator: { section: 'Live job posting' }
    }],
    byField: { fintech: ['resume-adjacent', 'job-domain'] }
  };
  const gapDraft = proposal({
    field_id: 'fintech',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value: "I don't have a specific number of years of direct fintech or payments experience to report. My relevant experience includes analytics dashboards and SQL reporting.",
    citation_ids: ['resume-adjacent', 'job-domain']
  });
  const grounded = enforceGroundingAcceptance({
    output: { page_id: 'experience-page', proposals: [gapDraft] },
    request: { pageId: 'experience-page', fields: [experienceField] },
    evidence: experienceEvidence
  }).proposals[0];
  assert.equal(grounded.action, 'fill');
  assert.equal(grounded.confidence, 'review');
  assert.doesNotMatch(grounded.value, /\b(?:supplied materials?|documents?|evidence|resume|sources?|citations?)\b/iu);

  const inventedTenure = enforceGroundingAcceptance({
    output: {
      page_id: 'experience-page',
      proposals: [proposal({
        field_id: 'fintech',
        confidence: 'review',
        risk_class: 'F2_REVIEW',
        value: 'I have 5 years of fintech and payments experience.',
        citation_ids: ['resume-adjacent', 'job-domain']
      })]
    },
    request: { pageId: 'experience-page', fields: [experienceField] },
    evidence: experienceEvidence
  }).proposals[0];
  assert.equal(inventedTenure.action, 'ask_user');
});

test('user-verified salary preferences ground expectations but not salary history', () => {
  const salaryField = {
    fieldId: 'salary',
    label: 'What are your salary expectations?',
    type: 'textarea',
    options: [],
    riskClass: 'F2_REVIEW'
  };
  const preferenceCitation = {
    citationId: 'salary-preference',
    sourceRole: 'user_verified',
    documentId: 'fact:salary-preference',
    documentVersion: 1,
    chunkId: 'chunk:salary-preference',
    quoteHash: 'salary-preference-hash',
    text: 'Salary expectation: I am targeting $80,000 - $90,000 USD.',
    locator: { section: 'Custom profile thought' }
  };
  const preferenceEvidence = {
    citations: [preferenceCitation],
    byField: { salary: ['salary-preference'] }
  };
  const preference = enforceGroundingAcceptance({
    output: {
      page_id: 'salary-preference-page',
      proposals: [proposal({
        field_id: 'salary',
        confidence: 'review',
        risk_class: 'F2_REVIEW',
        value: 'I am targeting $80,000 - $90,000 USD.',
        citation_ids: ['salary-preference']
      })]
    },
    request: { pageId: 'salary-preference-page', fields: [salaryField] },
    evidence: preferenceEvidence
  }).proposals[0];
  assert.equal(preference.action, 'fill');
  assert.equal(preference.confidence, 'review');

  for (const unsafeValue of [
    'My prior salary was $70,000.',
    'Salary expectation: I currently earn $70,000 and target $90,000.',
    'My last compensation was $70,000.',
    'I was paid $70,000.'
  ]) {
    const historyEvidence = {
      citations: [{ ...preferenceCitation, text: `Salary expectation: ${unsafeValue}` }],
      byField: { salary: ['salary-preference'] }
    };
    const history = enforceGroundingAcceptance({
      output: {
        page_id: 'salary-preference-page',
        proposals: [proposal({
          field_id: 'salary',
          confidence: 'review',
          risk_class: 'F2_REVIEW',
          value: unsafeValue,
          citation_ids: ['salary-preference']
        })]
      },
      request: { pageId: 'salary-preference-page', fields: [salaryField] },
      evidence: historyEvidence
    }).proposals[0];
    assert.equal(history.action, 'ask_user', unsafeValue);
  }
});

test('valid non-fill actions are preserved but manual fills are not', () => {
  const ask = proposal({
    action: 'ask_user',
    confidence: 'needs_input',
    value_type: 'none',
    value: '',
    citation_ids: [],
    abstain_reason: 'Please confirm.'
  });
  assert.deepEqual(processOne(ask), ask);
  const manualRequest = {
    ...request,
    fields: request.fields.map(field => field.fieldId === 'name' ? { ...field, manual: true } : field)
  };
  const result = enforceGroundingAcceptance({
    output: { page_id: 'page-1', proposals: [proposal()] },
    request: manualRequest,
    evidence
  }).proposals[0];
  assertNonFillable(result, GROUNDING_REASONS.MANUAL_FIELD);
});
