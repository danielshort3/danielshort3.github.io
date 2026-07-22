import {
  ANSWER_VALUE_TYPES,
  FIELD_RISK_CLASSES,
  FIELD_REGENERATION_OUTPUT_SCHEMA,
  FIELD_ACTIONS,
  PAGE_BATCH_OUTPUT_SCHEMA,
  PROPOSAL_CONFIDENCE,
  SOURCE_ROLES,
  VALIDATION_LIMITS
} from './schemas.js';

const isPlainObject = (value) => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value);

const boundedString = (value, maxLength, { allowEmpty = false } = {}) => {
  if (typeof value !== 'string') return false;
  if (!allowEmpty && !value.trim()) return false;
  return value.length <= maxLength;
};

const uniqueStrings = (values) => Array.isArray(values)
  && values.every(value => typeof value === 'string')
  && new Set(values).size === values.length;

const hasExactKeys = (value, expectedKeys) => {
  if (!isPlainObject(value)) return false;
  const actual = Object.keys(value).sort();
  const expected = [...expectedKeys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
};

export class ValidationError extends Error {
  constructor(message, issues = []) {
    super(message);
    this.name = 'ValidationError';
    this.issues = issues;
  }
}

export const parseJsonObject = (value, label = 'structured model output') => {
  let parsed = value;
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value);
    } catch (error) {
      throw new ValidationError(`${label} is not valid JSON.`, [error.message]);
    }
  }
  if (!isPlainObject(parsed)) throw new ValidationError(`${label} must be a JSON object.`);
  return parsed;
};

export const validateCitationSubset = (citationIds, allowedCitationIds, label = 'citation_ids') => {
  if (!uniqueStrings(citationIds)) {
    throw new ValidationError(`${label} must contain unique strings.`);
  }
  if (citationIds.length > VALIDATION_LIMITS.maxCitationsPerProposal) {
    throw new ValidationError(`${label} contains too many citations.`);
  }
  const allowed = new Set(allowedCitationIds || []);
  const unknown = citationIds.filter(citationId => !allowed.has(citationId));
  if (unknown.length) {
    throw new ValidationError(`${label} contains citations that were not retrieved.`, unknown);
  }
  return citationIds;
};

export const validatePageBatchRequest = (request) => {
  const issues = [];
  if (!isPlainObject(request)) throw new ValidationError('Page batch request must be an object.');
  if (!boundedString(request.pageId, 200)) issues.push('pageId is required and must be at most 200 characters.');
  if (!boundedString(request.urlHash, 200)) issues.push('urlHash is required and must be at most 200 characters.');
  if (!Number.isSafeInteger(request.domRevision) || request.domRevision < 0) {
    issues.push('domRevision must be a non-negative safe integer.');
  }
  if (!Array.isArray(request.fields) || !request.fields.length) {
    issues.push('fields must contain at least one field descriptor.');
  } else if (request.fields.length > VALIDATION_LIMITS.maxFieldsPerPage) {
    issues.push(`fields cannot exceed ${VALIDATION_LIMITS.maxFieldsPerPage} items.`);
  } else {
    const ids = new Set();
    request.fields.forEach((field, index) => {
      if (!isPlainObject(field)) {
        issues.push(`fields[${index}] must be an object.`);
        return;
      }
      if (!boundedString(field.fieldId, 200)) issues.push(`fields[${index}].fieldId is invalid.`);
      if (ids.has(field.fieldId)) issues.push(`fields[${index}].fieldId is duplicated.`);
      ids.add(field.fieldId);
      if (!boundedString(field.label, VALIDATION_LIMITS.maxLabelLength)) {
        issues.push(`fields[${index}].label is invalid.`);
      }
      if (!boundedString(field.type, 100)) issues.push(`fields[${index}].type is invalid.`);
      if (!Object.values(FIELD_RISK_CLASSES).includes(field.riskClass)) {
        issues.push(`fields[${index}].riskClass must be F1_VERIFIED or F2_REVIEW.`);
      }
      if (field.options !== undefined) {
        if (!uniqueStrings(field.options) || field.options.length > VALIDATION_LIMITS.maxOptionsPerField) {
          issues.push(`fields[${index}].options is invalid.`);
        }
      }
      if (field.nearbyText !== undefined
        && !boundedString(field.nearbyText, VALIDATION_LIMITS.maxNearbyTextLength, { allowEmpty: true })) {
        issues.push(`fields[${index}].nearbyText is invalid.`);
      }
      if (field.maxLength !== undefined
        && (!Number.isSafeInteger(field.maxLength) || field.maxLength < 1 || field.maxLength > VALIDATION_LIMITS.maxTextValueLength)) {
        issues.push(`fields[${index}].maxLength is invalid.`);
      }
      if (field.manual !== undefined && typeof field.manual !== 'boolean') {
        issues.push(`fields[${index}].manual must be boolean.`);
      }
    });
  }
  if (issues.length) throw new ValidationError('Page batch request is invalid.', issues);
  return request;
};

const validateProposalValue = (proposal, field, issues) => {
  const valueTypes = Object.values(ANSWER_VALUE_TYPES);
  if (!valueTypes.includes(proposal.value_type)) issues.push(`${field.fieldId}: invalid value_type.`);
  if (!boundedString(proposal.value, VALIDATION_LIMITS.maxTextValueLength, { allowEmpty: true })) {
    issues.push(`${field.fieldId}: value must be a bounded string.`);
  }
  if (!uniqueStrings(proposal.selected_values)) issues.push(`${field.fieldId}: selected_values must be unique strings.`);
  if (typeof proposal.checked !== 'boolean') issues.push(`${field.fieldId}: checked must be boolean.`);
  if (field.maxLength && typeof proposal.value === 'string' && proposal.value.length > field.maxLength) {
    issues.push(`${field.fieldId}: value exceeds the field maxLength.`);
  }
  if (field.options?.length && proposal.value_type === ANSWER_VALUE_TYPES.SELECTED_VALUES) {
    const allowedOptions = new Set(field.options);
    const invalidOptions = proposal.selected_values.filter(option => !allowedOptions.has(option));
    if (invalidOptions.length) issues.push(`${field.fieldId}: selected_values contains unavailable options.`);
  }
};

export const validatePageBatchOutput = (value, {
  request,
  allowedCitationIds = [],
  allowedCitationIdsByField = null,
  requireProposalForEveryField = false
} = {}) => {
  validatePageBatchRequest(request);
  const output = parseJsonObject(value, 'page batch output');
  const issues = [];
  if (!hasExactKeys(output, Object.keys(PAGE_BATCH_OUTPUT_SCHEMA.properties))) {
    issues.push('page batch output has unexpected or missing keys.');
  }
  if (output.page_id !== request.pageId) issues.push('page_id does not match the analyzed page.');
  if (!Array.isArray(output.proposals)) issues.push('proposals must be an array.');
  if (Array.isArray(output.proposals) && output.proposals.length > VALIDATION_LIMITS.maxFieldsPerPage) {
    issues.push('proposals contains too many items.');
  }
  if (issues.length) throw new ValidationError('Page batch output is invalid.', issues);

  const fieldsById = new Map(request.fields.map(field => [field.fieldId, field]));
  const retrievedCitationIds = new Set(allowedCitationIds);
  const proposalIds = new Set();
  for (const proposal of output.proposals) {
    if (!isPlainObject(proposal)) {
      issues.push('Every proposal must be an object.');
      continue;
    }
    if (!hasExactKeys(proposal, Object.keys(PAGE_BATCH_OUTPUT_SCHEMA.properties.proposals.items.properties))) {
      issues.push(`${String(proposal.field_id || 'proposal')}: unexpected or missing keys.`);
    }
    const field = fieldsById.get(proposal.field_id);
    if (!field) {
      issues.push(`Unknown field_id: ${String(proposal.field_id)}`);
      continue;
    }
    if (proposalIds.has(proposal.field_id)) issues.push(`Duplicate proposal: ${proposal.field_id}`);
    proposalIds.add(proposal.field_id);
    if (!Object.values(FIELD_ACTIONS).includes(proposal.action)) {
      issues.push(`${field.fieldId}: invalid action.`);
    }
    if (field.manual && proposal.action === FIELD_ACTIONS.FILL) {
      issues.push(`${field.fieldId}: manual fields cannot be filled by the model.`);
    }
    validateProposalValue(proposal, field, issues);
    const authoritativeFieldCitationIds = allowedCitationIdsByField
      ? (Object.hasOwn(allowedCitationIdsByField, field.fieldId)
        && Array.isArray(allowedCitationIdsByField[field.fieldId])
        ? allowedCitationIdsByField[field.fieldId].filter(citationId => retrievedCitationIds.has(citationId))
        : [])
      : allowedCitationIds;
    try {
      validateCitationSubset(
        proposal.citation_ids,
        authoritativeFieldCitationIds,
        `${field.fieldId}.citation_ids`
      );
    } catch (error) {
      issues.push(...(error.issues?.length ? error.issues : [error.message]));
    }
    if (proposal.action === FIELD_ACTIONS.FILL && proposal.citation_ids?.length === 0) {
      issues.push(`${field.fieldId}: fill requires at least one field-scoped citation.`);
    }
    if (!boundedString(proposal.short_rationale, 1000, { allowEmpty: true })) {
      issues.push(`${field.fieldId}: short_rationale is invalid.`);
    }
    if (!boundedString(proposal.abstain_reason, 1000, { allowEmpty: true })) {
      issues.push(`${field.fieldId}: abstain_reason is invalid.`);
    }
    if (proposal.action === FIELD_ACTIONS.ASK_USER
      && (typeof proposal.abstain_reason !== 'string' || !proposal.abstain_reason.trim())) {
      issues.push(`${field.fieldId}: ask_user requires an abstain_reason.`);
    }
    if (!Object.values(PROPOSAL_CONFIDENCE).includes(proposal.confidence)) {
      issues.push(`${field.fieldId}: invalid confidence.`);
    }
    if (proposal.risk_class !== field.riskClass) {
      issues.push(`${field.fieldId}: risk_class must match the field descriptor.`);
    }
    if (proposal.action === FIELD_ACTIONS.ASK_USER
      && proposal.confidence !== PROPOSAL_CONFIDENCE.NEEDS_INPUT) {
      issues.push(`${field.fieldId}: ask_user requires confidence needs_input.`);
    }
  }
  if (requireProposalForEveryField && proposalIds.size !== fieldsById.size) {
    issues.push('The model did not return a proposal for every field.');
  }
  if (issues.length) throw new ValidationError('Page batch output is invalid.', issues);
  return output;
};

export const validateFieldRegenerationOutput = (value, {
  field,
  allowedCitationIds = []
} = {}) => {
  if (!isPlainObject(field)
    || !boundedString(field.fieldId, 200)
    || !Object.values(FIELD_RISK_CLASSES).includes(field.riskClass)) {
    throw new ValidationError('A valid field descriptor is required for regeneration.');
  }
  const output = parseJsonObject(value, 'field regeneration output');
  const issues = [];
  if (!hasExactKeys(output, Object.keys(FIELD_REGENERATION_OUTPUT_SCHEMA.properties))) {
    issues.push('field regeneration output has unexpected or missing keys.');
  }
  if (output.field_id !== field.fieldId) issues.push('field_id does not match the requested field.');
  if (!Object.values(PROPOSAL_CONFIDENCE).includes(output.confidence)) {
    issues.push('confidence is invalid.');
  }
  if (output.risk_class !== field.riskClass) {
    issues.push('risk_class must match the field descriptor.');
  }
  validateProposalValue(output, field, issues);
  try {
    validateCitationSubset(output.citation_ids, allowedCitationIds);
  } catch (error) {
    issues.push(...(error.issues?.length ? error.issues : [error.message]));
  }
  const isFillLike = output.value_type !== ANSWER_VALUE_TYPES.NONE
    && output.confidence !== PROPOSAL_CONFIDENCE.NEEDS_INPUT
    && !output.abstain_reason;
  if (isFillLike && output.citation_ids?.length === 0) {
    issues.push('Regenerated fill requires at least one field-scoped citation.');
  }
  if (!boundedString(output.changes_summary, 1000, { allowEmpty: true })) {
    issues.push('changes_summary is invalid.');
  }
  if (!boundedString(output.abstain_reason, 1000, { allowEmpty: true })) {
    issues.push('abstain_reason is invalid.');
  }
  if (issues.length) throw new ValidationError('Field regeneration output is invalid.', issues);
  return output;
};

export const validateSourceRole = (role) => {
  if (!Object.values(SOURCE_ROLES).includes(role)) {
    throw new ValidationError(`Unknown source role: ${String(role)}`);
  }
  return role;
};
