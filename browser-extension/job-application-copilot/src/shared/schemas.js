const deepFreeze = (value) => {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  Object.freeze(value);
  Object.values(value).forEach(deepFreeze);
  return value;
};

export const SOURCE_ROLES = deepFreeze({
  CANDIDATE_EVIDENCE: 'candidate_evidence',
  JOB_REQUIREMENT: 'job_requirement',
  COMPANY_CONTEXT: 'company_context',
  STYLE_EXAMPLE: 'style_example',
  USER_VERIFIED: 'user_verified'
});

export const FIELD_ACTIONS = deepFreeze({
  FILL: 'fill',
  SKIP: 'skip',
  ASK_USER: 'ask_user'
});

export const PROPOSAL_CONFIDENCE = deepFreeze({
  HIGH: 'high',
  REVIEW: 'review',
  NEEDS_INPUT: 'needs_input'
});

export const FIELD_RISK_CLASSES = deepFreeze({
  VERIFIED: 'F1_VERIFIED',
  REVIEW: 'F2_REVIEW'
});

export const ANSWER_VALUE_TYPES = deepFreeze({
  TEXT: 'text',
  SELECTED_VALUES: 'selected_values',
  CHECKED: 'checked',
  NONE: 'none'
});

export const VALIDATION_LIMITS = deepFreeze({
  maxFieldsPerPage: 50,
  maxOptionsPerField: 100,
  maxLabelLength: 500,
  maxNearbyTextLength: 4000,
  maxTextValueLength: 12000,
  maxCitationsPerProposal: 12,
  maxEvidenceChunksPerPage: 20
});

const proposalProperties = {
  field_id: { type: 'string', minLength: 1, maxLength: 200 },
  action: { type: 'string', enum: Object.values(FIELD_ACTIONS) },
  confidence: { type: 'string', enum: Object.values(PROPOSAL_CONFIDENCE) },
  risk_class: { type: 'string', enum: Object.values(FIELD_RISK_CLASSES) },
  value_type: { type: 'string', enum: Object.values(ANSWER_VALUE_TYPES) },
  value: { type: 'string', maxLength: VALIDATION_LIMITS.maxTextValueLength },
  selected_values: {
    type: 'array',
    items: { type: 'string', maxLength: 1000 },
    maxItems: VALIDATION_LIMITS.maxOptionsPerField
  },
  checked: { type: 'boolean' },
  citation_ids: {
    type: 'array',
    items: { type: 'string', minLength: 1, maxLength: 200 },
    maxItems: VALIDATION_LIMITS.maxCitationsPerProposal
  },
  short_rationale: { type: 'string', maxLength: 1000 },
  abstain_reason: { type: 'string', maxLength: 1000 }
};

export const PAGE_BATCH_OUTPUT_SCHEMA = deepFreeze({
  type: 'object',
  additionalProperties: false,
  properties: {
    page_id: { type: 'string', minLength: 1, maxLength: 200 },
    proposals: {
      type: 'array',
      maxItems: VALIDATION_LIMITS.maxFieldsPerPage,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: proposalProperties,
        required: Object.keys(proposalProperties)
      }
    }
  },
  required: ['page_id', 'proposals']
});

export const FIELD_REGENERATION_OUTPUT_SCHEMA = deepFreeze({
  type: 'object',
  additionalProperties: false,
  properties: {
    field_id: { type: 'string', minLength: 1, maxLength: 200 },
    confidence: { type: 'string', enum: Object.values(PROPOSAL_CONFIDENCE) },
    risk_class: { type: 'string', enum: Object.values(FIELD_RISK_CLASSES) },
    value_type: { type: 'string', enum: Object.values(ANSWER_VALUE_TYPES) },
    value: { type: 'string', maxLength: VALIDATION_LIMITS.maxTextValueLength },
    selected_values: {
      type: 'array',
      items: { type: 'string', maxLength: 1000 },
      maxItems: VALIDATION_LIMITS.maxOptionsPerField
    },
    checked: { type: 'boolean' },
    citation_ids: {
      type: 'array',
      items: { type: 'string', minLength: 1, maxLength: 200 },
      maxItems: VALIDATION_LIMITS.maxCitationsPerProposal
    },
    changes_summary: { type: 'string', maxLength: 1000 },
    abstain_reason: { type: 'string', maxLength: 1000 }
  },
  required: [
    'field_id',
    'confidence',
    'risk_class',
    'value_type',
    'value',
    'selected_values',
    'checked',
    'citation_ids',
    'changes_summary',
    'abstain_reason'
  ]
});

export const DEFAULT_LOCAL_MODEL_CONFIG = deepFreeze({
  generationModel: 'qwen3.5:27b',
  fallbackGenerationModel: 'qwen3:8b',
  embeddingModel: 'nomic-embed-text',
  numContextTokens: 12288,
  maxPageOutputTokens: 3072,
  maxFieldOutputTokens: 768,
  keepAlive: '10m',
  think: false,
  mappingTemperature: 0,
  regenerationTemperature: 0.3
});
