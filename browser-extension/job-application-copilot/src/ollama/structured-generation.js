import {
  ANSWER_VALUE_TYPES,
  DEFAULT_LOCAL_MODEL_CONFIG,
  FIELD_ACTIONS,
  FIELD_RISK_CLASSES,
  FIELD_REGENERATION_OUTPUT_SCHEMA,
  PAGE_BATCH_OUTPUT_SCHEMA,
  PROPOSAL_CONFIDENCE
} from '../shared/schemas.js';
import {
  ValidationError,
  parseJsonObject,
  validateFieldRegenerationOutput,
  validatePageBatchOutput,
  validatePageBatchRequest
} from '../shared/validators.js';
import { buildFieldRegenerationMessages, buildPageBatchMessages } from './prompts.js';
import {
  enforceFieldRegenerationGrounding,
  enforceGroundingAcceptance
} from '../grounding/postprocessor.js';

const OLLAMA_GRAMMAR_UNSAFE_KEYS = new Set(['maxLength']);

export const createOllamaGrammarSafeSchema = (value) => {
  if (Array.isArray(value)) return Object.freeze(value.map(createOllamaGrammarSafeSchema));
  if (!value || typeof value !== 'object') return value;
  return Object.freeze(Object.fromEntries(Object.entries(value)
    .filter(([key]) => !OLLAMA_GRAMMAR_UNSAFE_KEYS.has(key))
    .map(([key, entry]) => [key, createOllamaGrammarSafeSchema(entry)])));
};

const canonicalProposalProperties = PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.items.properties;
const PAGE_BATCH_WIRE_OUTPUT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    page_id: PAGE_BATCH_OUTPUT_SCHEMA.properties.page_id,
    proposals: {
      type: 'array',
      maxItems: PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.maxItems,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          field_id: canonicalProposalProperties.field_id,
          action: { type: 'string', enum: [FIELD_ACTIONS.FILL, FIELD_ACTIONS.ASK_USER] },
          value_type: canonicalProposalProperties.value_type,
          value: canonicalProposalProperties.value,
          selected_values: canonicalProposalProperties.selected_values,
          checked: canonicalProposalProperties.checked,
          citation_ids: canonicalProposalProperties.citation_ids,
          abstain_reason: canonicalProposalProperties.abstain_reason
        },
        required: ['field_id', 'action', 'value_type', 'citation_ids']
      }
    }
  },
  required: ['page_id', 'proposals']
};

export const OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA = createOllamaGrammarSafeSchema(PAGE_BATCH_WIRE_OUTPUT_SCHEMA);
export const OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA = createOllamaGrammarSafeSchema(FIELD_REGENERATION_OUTPUT_SCHEMA);

const allowedCitationIds = (evidence) => {
  if (!evidence || !Array.isArray(evidence.citations) || !evidence.byField) {
    throw new Error('A validated evidence pack is required.');
  }
  return evidence.citations.map(citation => citation.citationId);
};

export const normalizePageBatchWireProposal = (value, field) => {
  const raw = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const fill = raw.action === FIELD_ACTIONS.FILL;
  const valueType = fill ? raw.value_type : ANSWER_VALUE_TYPES.NONE;
  return {
    field_id: raw.field_id,
    action: raw.action,
    confidence: fill
      ? field?.riskClass === FIELD_RISK_CLASSES.VERIFIED
        ? PROPOSAL_CONFIDENCE.HIGH
        : PROPOSAL_CONFIDENCE.REVIEW
      : PROPOSAL_CONFIDENCE.NEEDS_INPUT,
    risk_class: field?.riskClass || raw.risk_class || '',
    value_type: valueType,
    value: fill && valueType === ANSWER_VALUE_TYPES.TEXT ? String(raw.value || '') : '',
    selected_values: fill && valueType === ANSWER_VALUE_TYPES.SELECTED_VALUES
      ? raw.selected_values ?? []
      : [],
    checked: fill && valueType === ANSWER_VALUE_TYPES.CHECKED ? raw.checked ?? false : false,
    citation_ids: raw.citation_ids ?? [],
    short_rationale: fill ? 'Supported by retrieved evidence.' : 'User input is required.',
    abstain_reason: raw.action === FIELD_ACTIONS.ASK_USER
      ? String(raw.abstain_reason || 'More information is required.')
      : ''
  };
};

const normalizePageBatchWireOutput = (value, request) => {
  const raw = parseJsonObject(value, 'page batch output');
  if (!Array.isArray(raw.proposals)) {
    throw new ValidationError('Page batch output is invalid.', ['proposals must be an array.']);
  }
  const fieldsById = new Map(request.fields.map(field => [field.fieldId, field]));
  return {
    page_id: raw.page_id,
    proposals: raw.proposals.map(proposal => normalizePageBatchWireProposal(
      proposal,
      fieldsById.get(proposal?.field_id)
    ))
  };
};

const validateAndGroundPage = ({ output, request, evidence, citations, requireEveryField = true }) => {
  const validationOptions = {
    request,
    allowedCitationIds: citations,
    allowedCitationIdsByField: evidence.byField,
    requireProposalForEveryField: requireEveryField
  };
  const validated = validatePageBatchOutput(output, validationOptions);
  const grounded = enforceGroundingAcceptance({ output: validated, request, evidence });
  return validatePageBatchOutput(grounded, validationOptions);
};

const salvagePageBatchOutput = ({ output, request, evidence, citations }) => {
  const proposals = [];
  const rejectedFields = [];
  for (const field of request.fields) {
    const candidates = output.proposals.filter(proposal => proposal.field_id === field.fieldId);
    if (candidates.length !== 1) {
      rejectedFields.push({
        fieldId: field.fieldId,
        issues: [candidates.length ? 'Duplicate proposal.' : 'Proposal was not returned.']
      });
      continue;
    }
    const fieldRequest = { ...request, fields: [field] };
    try {
      const validated = validateAndGroundPage({
        output: { page_id: output.page_id, proposals: candidates },
        request: fieldRequest,
        evidence,
        citations,
        requireEveryField: true
      });
      proposals.push(validated.proposals[0]);
    } catch (error) {
      rejectedFields.push({
        fieldId: field.fieldId,
        issues: error?.issues?.length ? [...error.issues] : [String(error?.message || error)]
      });
    }
  }
  if (!proposals.length && rejectedFields.length) {
    throw new ValidationError(
      'No page proposals could be validated.',
      rejectedFields.flatMap(entry => entry.issues.map(issue => `${entry.fieldId}: ${issue}`))
    );
  }
  return { output: { page_id: request.pageId, proposals }, rejectedFields };
};

export const pageBatchPredictionLimit = fieldCount => Math.min(
  DEFAULT_LOCAL_MODEL_CONFIG.maxPageOutputTokens,
  Math.max(640, (Math.max(1, Number(fieldCount) || 1) * 384) + 128)
);

export const generatePageBatch = async ({
  client,
  request,
  evidence,
  model = DEFAULT_LOCAL_MODEL_CONFIG.generationModel,
  constrainedRetry = false,
  signal,
  onProgress,
  allowPartial = false
}) => {
  if (!client?.chatStructured) throw new Error('An OllamaClient instance is required.');
  validatePageBatchRequest(request);
  const citations = allowedCitationIds(evidence);
  const response = await client.chatStructured({
    model,
    messages: buildPageBatchMessages({ request, evidence, constrainedRetry }),
    format: OLLAMA_PAGE_BATCH_OUTPUT_SCHEMA,
    think: DEFAULT_LOCAL_MODEL_CONFIG.think,
    keepAlive: DEFAULT_LOCAL_MODEL_CONFIG.keepAlive,
    options: {
      temperature: DEFAULT_LOCAL_MODEL_CONFIG.mappingTemperature,
      num_ctx: DEFAULT_LOCAL_MODEL_CONFIG.numContextTokens,
      num_predict: pageBatchPredictionLimit(request.fields.length)
    },
    signal,
    onProgress
  });
  const normalized = normalizePageBatchWireOutput(response.content, request);
  if (allowPartial) {
    const salvaged = salvagePageBatchOutput({ output: normalized, request, evidence, citations });
    return {
      ...salvaged,
      metrics: response.metrics
    };
  }
  return {
    output: validateAndGroundPage({ output: normalized, request, evidence, citations }),
    rejectedFields: [],
    metrics: response.metrics
  };
};

export const regenerateField = async ({
  client,
  field,
  priorDraft,
  feedback,
  evidence,
  model = DEFAULT_LOCAL_MODEL_CONFIG.generationModel,
  constrainedRetry = false,
  signal,
  onProgress
}) => {
  if (!client?.chatStructured) throw new Error('An OllamaClient instance is required.');
  if (field?.manual) throw new Error('Manual fields cannot be regenerated by the model.');
  const retrievedCitations = allowedCitationIds(evidence);
  const retrievedSet = new Set(retrievedCitations);
  const fieldCitations = Object.hasOwn(evidence.byField, field.fieldId)
    && Array.isArray(evidence.byField[field.fieldId])
    ? evidence.byField[field.fieldId].filter(citationId => retrievedSet.has(citationId))
    : [];
  const response = await client.chatStructured({
    model,
    messages: buildFieldRegenerationMessages({ field, priorDraft, feedback, evidence, constrainedRetry }),
    format: OLLAMA_FIELD_REGENERATION_OUTPUT_SCHEMA,
    think: DEFAULT_LOCAL_MODEL_CONFIG.think,
    keepAlive: DEFAULT_LOCAL_MODEL_CONFIG.keepAlive,
    options: {
      temperature: DEFAULT_LOCAL_MODEL_CONFIG.regenerationTemperature,
      num_ctx: DEFAULT_LOCAL_MODEL_CONFIG.numContextTokens,
      num_predict: DEFAULT_LOCAL_MODEL_CONFIG.maxFieldOutputTokens
    },
    signal,
    onProgress
  });
  const validationOptions = { field, allowedCitationIds: fieldCitations };
  const validated = validateFieldRegenerationOutput(response.content, validationOptions);
  const grounded = enforceFieldRegenerationGrounding({ output: validated, field, evidence });
  return {
    output: validateFieldRegenerationOutput(grounded, validationOptions),
    metrics: response.metrics
  };
};
