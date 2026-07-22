import assert from 'node:assert/strict';
import test from 'node:test';
import {
  generatePageBatch,
  OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA,
  OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA,
  normalizePageBatchWireProposal,
  pageBatchPredictionLimit,
  regenerateField
} from '../src/ollama/structured-generation.js';
import {
  FIELD_REGENERATION_OUTPUT_SCHEMA,
  PAGE_BATCH_OUTPUT_SCHEMA,
  VALIDATION_LIMITS
} from '../src/shared/schemas.js';
import {
  ValidationError,
  validateFieldRegenerationOutput,
  validatePageBatchOutput
} from '../src/shared/validators.js';
import {
  buildFieldRegenerationMessages,
  buildPageBatchMessages,
  estimatePageBatchInputTokens,
  normalizeFieldRegenerationFeedback
} from '../src/ollama/prompts.js';

const containsKey = (value, targetKey) => {
  if (!value || typeof value !== 'object') return false;
  if (!Array.isArray(value) && Object.hasOwn(value, targetKey)) return true;
  return Object.values(value).some(entry => containsKey(entry, targetKey));
};

const omitMaxLength = (value) => {
  if (Array.isArray(value)) return value.map(omitMaxLength);
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => key !== 'maxLength')
    .map(([key, entry]) => [key, omitMaxLength(entry)]));
};

const request = {
  pageId: 'page-1',
  urlHash: 'url-hash',
  domRevision: 2,
  fields: [{
    fieldId: 'summary',
    label: 'Professional summary',
    type: 'textarea',
    riskClass: 'F1_VERIFIED',
    options: [],
    nearbyText: '',
    maxLength: 500,
    manual: false
  }, {
    fieldId: 'attestation',
    label: 'I certify this application',
    type: 'checkbox',
    riskClass: 'F2_REVIEW',
    options: [],
    nearbyText: '',
    manual: true
  }]
};

const fillProposal = {
  field_id: 'summary',
  action: 'fill',
  confidence: 'high',
  risk_class: 'F1_VERIFIED',
  value_type: 'text',
  value: 'Built evidence-grounded analytics systems.',
  selected_values: [],
  checked: false,
  citation_ids: ['c1'],
  short_rationale: 'Supported by the resume.',
  abstain_reason: ''
};

const askProposal = {
  field_id: 'attestation',
  action: 'ask_user',
  confidence: 'needs_input',
  risk_class: 'F2_REVIEW',
  value_type: 'none',
  value: '',
  selected_values: [],
  checked: false,
  citation_ids: [],
  short_rationale: 'Requires the applicant.',
  abstain_reason: 'This is a personal attestation.'
};

const evidence = {
  citations: [{
    citationId: 'c1',
    sourceRole: 'candidate_evidence',
    documentId: 'doc:resume',
    locator: { pageStart: 1 },
    text: 'Built evidence-grounded analytics systems.'
  }, {
    citationId: 'c2',
    sourceRole: 'company_context',
    documentId: 'doc:research',
    locator: { paragraphStart: 1 },
    text: 'The employer values analytics.'
  }],
  byField: { summary: ['c1'], attestation: [] }
};

test('field feedback normalization treats free text as optional and honors preset-only requests', () => {
  assert.deepEqual(normalizeFieldRegenerationFeedback(), {
    preset: 'none',
    text: '',
    maxChars: null,
    mustInclude: [],
    mustAvoid: []
  });
  assert.deepEqual(normalizeFieldRegenerationFeedback({ text: ' \n\t ' }), {
    preset: 'none',
    text: '',
    maxChars: null,
    mustInclude: [],
    mustAvoid: []
  });
  assert.deepEqual(normalizeFieldRegenerationFeedback({
    preset: 'shorter',
    text: '   ',
    fieldMaxLength: 500
  }), {
    preset: 'shorter',
    text: '',
    maxChars: 500,
    mustInclude: [],
    mustAvoid: []
  });
  assert.deepEqual(normalizeFieldRegenerationFeedback({
    text: 'I have three years of direct payments experience.',
    mustInclude: ['payments']
  }), {
    preset: 'other',
    text: 'I have three years of direct payments experience.',
    maxChars: null,
    mustInclude: ['payments'],
    mustAvoid: []
  });
});

test('blank regeneration feedback preserves the exact draft and includes the request and frozen evidence', () => {
  const field = {
    ...request.fields[0],
    nearbyText: 'Please explain the experience most relevant to this request.'
  };
  const messages = buildFieldRegenerationMessages({
    field,
    priorDraft: fillProposal,
    feedback: { preset: '', text: '   ' },
    evidence
  });
  const payload = JSON.parse(messages[1].content);

  assert.deepEqual(payload.prior_draft, fillProposal);
  assert.equal(payload.field.label, field.label);
  assert.equal(payload.field.nearby_text, field.nearbyText);
  assert.deepEqual(payload.feedback, {
    preset: 'none',
    text: '',
    max_chars: field.maxLength,
    must_include: [],
    must_avoid: []
  });
  assert.deepEqual(payload.evidence, evidence.citations.map(citation => ({
    citation_id: citation.citationId,
    source_role: citation.sourceRole,
    document_id: citation.documentId,
    locator: citation.locator,
    text: citation.text
  })));
  assert.match(messages[0].content, /Feedback is optional/iu);
  assert.match(messages[0].content, /revise the exact current draft using the field label, nearby request, and frozen evidence/iu);
  assert.match(messages[0].content, /Do not ask for more input solely because feedback is blank/iu);
});
test('Ollama wire schemas stay grammar-safe while page proposals use a compact canonicalized contract', () => {
  assert.equal(containsKey(OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA, 'maxLength'), false);
  assert.equal(containsKey(OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA, 'maxLength'), false);
  assert.deepEqual(OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA, omitMaxLength(FIELD_REGENERATION_OUTPUT_SCHEMA));
  const pageItem = OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.items;
  assert.deepEqual(Object.keys(pageItem.properties).sort(), [
    'abstain_reason',
    'action',
    'checked',
    'citation_ids',
    'field_id',
    'selected_values',
    'value',
    'value_type'
  ]);
  assert.deepEqual(pageItem.required, ['field_id', 'action', 'value_type', 'citation_ids']);
  assert.equal(Object.hasOwn(pageItem.properties, 'confidence'), false);
  assert.equal(Object.hasOwn(pageItem.properties, 'risk_class'), false);
  assert.equal(Object.hasOwn(pageItem.properties, 'short_rationale'), false);

  assert.equal(containsKey(PAGE_BATCH_OUTPUT_SCHEMA, 'maxLength'), true);
  assert.equal(containsKey(FIELD_REGENERATION_OUTPUT_SCHEMA, 'maxLength'), true);
  assert.equal(
    PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.items.properties.value.maxLength,
    VALIDATION_LIMITS.maxTextValueLength
  );
  assert.equal(
    FIELD_REGENERATION_OUTPUT_SCHEMA.properties.value.maxLength,
    VALIDATION_LIMITS.maxTextValueLength
  );

  assert.equal(OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA.additionalProperties, false);
  assert.equal(OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA.properties.page_id.minLength, 1);
  assert.equal(
    OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.maxItems,
    VALIDATION_LIMITS.maxFieldsPerPage
  );
  assert.deepEqual(
    OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA.properties.confidence.enum,
    FIELD_REGENERATION_OUTPUT_SCHEMA.properties.confidence.enum
  );
  assert.deepEqual(
    OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA.required,
    FIELD_REGENERATION_OUTPUT_SCHEMA.required
  );
  assert.equal(Object.isFrozen(OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA), true);
  assert.equal(Object.isFrozen(OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.items), true);
  assert.equal(Object.isFrozen(OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA.required), true);

  const oversizedValue = 'x'.repeat(VALIDATION_LIMITS.maxTextValueLength + 1);
  assert.throws(() => validatePageBatchOutput({
    page_id: request.pageId,
    proposals: [{ ...fillProposal, value: oversizedValue }, askProposal]
  }, {
    request,
    allowedCitationIds: ['c1'],
    allowedCitationIdsByField: evidence.byField,
    requireProposalForEveryField: true
  }), error => error instanceof ValidationError
    && error.issues.some(issue => /value must be a bounded string/u.test(issue)));

  assert.throws(() => validateFieldRegenerationOutput({
    field_id: request.fields[0].fieldId,
    confidence: 'review',
    risk_class: request.fields[0].riskClass,
    value_type: 'text',
    value: oversizedValue,
    selected_values: [],
    checked: false,
    citation_ids: ['c1'],
    changes_summary: '',
    abstain_reason: ''
  }, {
    field: { ...request.fields[0], maxLength: undefined },
    allowedCitationIds: ['c1']
  }), error => error instanceof ValidationError
    && error.issues.some(issue => /value must be a bounded string/u.test(issue)));
});

test('page output enforces confidence, risk class, exact keys, and field-scoped citations', () => {
  const valid = { page_id: request.pageId, proposals: [fillProposal, askProposal] };
  assert.equal(validatePageBatchOutput(valid, {
    request,
    allowedCitationIds: ['c1', 'c2'],
    allowedCitationIdsByField: evidence.byField,
    requireProposalForEveryField: true
  }), valid);

  assert.throws(() => validatePageBatchOutput({
    page_id: request.pageId,
    proposals: [{ ...fillProposal, citation_ids: ['c2'] }, askProposal]
  }, { request, allowedCitationIdsByField: evidence.byField }), ValidationError);
  assert.throws(() => validatePageBatchOutput({
    page_id: request.pageId,
    proposals: [fillProposal, { ...askProposal, confidence: 'review' }]
  }, { request, allowedCitationIds: ['c1'] }), error => error instanceof ValidationError
    && error.issues.some(issue => /ask_user requires confidence needs_input/u.test(issue)));
  assert.throws(() => validatePageBatchOutput({
    page_id: request.pageId,
    proposals: [{ ...fillProposal, risk_class: 'F2_REVIEW' }, askProposal]
  }, { request, allowedCitationIds: ['c1'] }), error => error instanceof ValidationError
    && error.issues.some(issue => /risk_class must match/u.test(issue)));
  assert.throws(() => validatePageBatchOutput({
    page_id: request.pageId,
    proposals: [{ ...fillProposal, surprise: true }, askProposal]
  }, { request, allowedCitationIds: ['c1'] }), error => error instanceof ValidationError
    && error.issues.some(issue => /unexpected or missing keys/u.test(issue)));
});

test('structured generation passes a schema to Ollama and validates the result', async () => {
  let call;
  const client = {
    async chatStructured(options) {
      call = options;
      return {
        content: JSON.stringify({ page_id: request.pageId, proposals: [fillProposal, askProposal] }),
        metrics: { evalCount: 42 }
      };
    }
  };
  const result = await generatePageBatch({ client, request, evidence });
  assert.equal(result.output.proposals[0].confidence, 'high');
  assert.equal(result.metrics.evalCount, 42);
  assert.equal(call.model, 'qwen3.5:27b');
  assert.equal(call.format, OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA);
  assert.equal(call.format.additionalProperties, false);
  assert.equal(containsKey(call.format, 'maxLength'), false);
  assert.match(call.messages[0].content, /untrusted data/u);
  assert.match(call.messages[0].content, /natural first person from the applicant's perspective/iu);
  assert.match(call.messages[0].content, /Never mention the model, system, evidence, citations, sources, resume, documents, or supplied materials in a field value/iu);
});

test('field regeneration preserves risk and only cites the frozen field evidence', async () => {
  const field = request.fields[0];
  let call;
  const client = {
    async chatStructured(options) {
      call = options;
      return {
        content: JSON.stringify({
          field_id: field.fieldId,
          confidence: 'review',
          risk_class: field.riskClass,
          value_type: 'text',
          value: 'Built grounded analytics tools.',
          selected_values: [],
          checked: false,
          citation_ids: ['c1'],
          changes_summary: 'Shortened the answer.',
          abstain_reason: ''
        }),
        metrics: {}
      };
    }
  };
  const result = await regenerateField({ client, field, priorDraft: fillProposal, feedback: {}, evidence });
  assert.equal(result.output.risk_class, 'F1_VERIFIED');
  assert.equal(call.format, OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA);
  assert.equal(containsKey(call.format, 'maxLength'), false);
});
test('page prompt removes UI-only citation metadata and exposes a conservative token estimate', () => {
  const messages = buildPageBatchMessages({ request, evidence });
  const payload = JSON.parse(messages[1].content);
  assert.deepEqual(payload.evidence[0], {
    citation_id: 'c1',
    source_role: 'candidate_evidence',
    text: 'Built evidence-grounded analytics systems.'
  });
  assert.equal(Object.hasOwn(payload.evidence[0], 'document_id'), false);
  assert.equal(Object.hasOwn(payload.evidence[0], 'locator'), false);
  assert.ok(estimatePageBatchInputTokens({ request, evidence }) > 100);
});

test('compact page proposals derive canonical risk and confidence locally', () => {
  assert.deepEqual(normalizePageBatchWireProposal({
    field_id: 'summary',
    action: 'fill',
    value_type: 'text',
    value: 'Built evidence-grounded analytics systems.',
    citation_ids: ['c1']
  }, request.fields[0]), {
    field_id: 'summary',
    action: 'fill',
    confidence: 'high',
    risk_class: 'F1_VERIFIED',
    value_type: 'text',
    value: 'Built evidence-grounded analytics systems.',
    selected_values: [],
    checked: false,
    citation_ids: ['c1'],
    short_rationale: 'Supported by retrieved evidence.',
    abstain_reason: ''
  });
  assert.equal(pageBatchPredictionLimit(1), 640);
  assert.equal(pageBatchPredictionLimit(50), 3072);
});

test('partial page validation retains grounded proposals and identifies only rejected fields', async () => {
  const client = {
    async chatStructured() {
      return {
        content: JSON.stringify({
          page_id: request.pageId,
          proposals: [{
            field_id: 'summary',
            action: 'fill',
            value_type: 'text',
            value: 'Built evidence-grounded analytics systems.',
            citation_ids: ['c1']
          }, {
            field_id: 'attestation',
            action: 'fill',
            value_type: 'checked',
            checked: true,
            citation_ids: ['c1']
          }]
        }),
        metrics: {}
      };
    }
  };
  const result = await generatePageBatch({ client, request, evidence, allowPartial: true });
  assert.deepEqual(result.output.proposals.map(proposal => proposal.field_id), ['summary']);
  assert.deepEqual(result.rejectedFields.map(entry => entry.fieldId), ['attestation']);
});
