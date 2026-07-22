import {
  ANSWER_VALUE_TYPES,
  FIELD_ACTIONS,
  FIELD_RISK_CLASSES,
  PROPOSAL_CONFIDENCE,
  SOURCE_ROLES
} from '../shared/schemas.js';

const ELIGIBLE_CANDIDATE_ROLES = new Set([
  SOURCE_ROLES.CANDIDATE_EVIDENCE,
  SOURCE_ROLES.USER_VERIFIED
]);

const SALARY_FIELD_PATTERN = /\b(?:salary|compensation|pay)\b/iu;
const SALARY_UNSUPPORTED_PATTERN = /\b(?:current|previous|prior|last)\s+(?:base\s+)?(?:salary|compensation|pay|earnings)\b|\b(?:salary|compensation|pay)\s+history\b|\b(?:currently|previously|formerly)\s+(?:earn|earned|earning|make|made|making|paid)\b|\b(?:i|my)\s+(?:(?:currently|previously)\s+)?(?:earn|earned|earning|make|made|making|am\s+paid|was\s+paid)\b|\bmy\s+(?:salary|compensation|pay)\s+(?:is|was)\b|\b(?:minimum|at least|bonus|equity|stock|commission|signing)\b/iu;
const SALARY_PREFERENCE_EVIDENCE_PATTERN = /\b(?:salary|compensation|pay)\s*(?:expectation|expectations|preference|preferences|target|range|desired)?\s*:/iu;
const SALARY_SAFE_TERMS = new Set([
  'account', 'compensation', 'consider', 'considering', 'discuss', 'discussing', 'discussion', 'expectation', 'expectations',
  'desired', 'full', 'into', 'open', 'package', 'pay', 'posted', 'prefer', 'preferred', 'range', 'role', 'salary',
  'take', 'taking', 'target', 'targeting', 'total', 'within', 'would'
]);

export const isSalaryExpectationField = field => SALARY_FIELD_PATTERN.test(
  String(field?.label || '') + ' ' + String(field?.nearbyText || '')
);

export const GROUNDING_REASONS = Object.freeze({
  INVALID_CITATIONS: 'One or more citations were not part of this field\'s frozen evidence.',
  MISSING_CANDIDATE_EVIDENCE: 'The proposed answer was not supported by cited candidate or user-verified evidence.',
  UNSUPPORTED_EXACT_VALUE: 'The exact answer was not directly present or canonically equivalent in cited candidate evidence.',
  UNSUPPORTED_CONCRETE_DETAIL: 'A concrete detail in the draft was not present in the cited evidence.',
  UNSUPPORTED_SALIENT_CLAIM: 'A meaningful claim term in the draft was not present in cited candidate evidence.',
  INCOHERENT_VALUE: 'The proposed answer shape was not safe to apply to this field.',
  MANUAL_FIELD: 'This field requires direct user input.'
});

export class GroundingError extends Error {
  constructor(message) {
    super(message);
    this.name = 'GroundingError';
  }
}

const isPlainObject = (value) => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value)
  && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

const normalizeDisplay = (value) => String(value || '')
  .normalize('NFC')
  .replaceAll('\r\n', '\n')
  .replaceAll('\r', '\n')
  .replace(/\s+/gu, ' ')
  .trim();

const uniqueStrings = (values) => Array.isArray(values)
  && values.every(value => typeof value === 'string')
  && new Set(values).size === values.length;

const cloneLocator = (locator) => {
  if (!isPlainObject(locator)) return {};
  return Object.fromEntries(Object.entries(locator).map(([key, value]) => [key, value]));
};

const evidenceIndex = (evidence) => {
  if (!isPlainObject(evidence) || !Array.isArray(evidence.citations) || !isPlainObject(evidence.byField)) {
    throw new GroundingError('A frozen evidence pack is required.');
  }
  const index = new Map();
  for (const citation of evidence.citations) {
    if (!isPlainObject(citation)
      || typeof citation.citationId !== 'string'
      || !citation.citationId
      || index.has(citation.citationId)
      || !Object.values(SOURCE_ROLES).includes(citation.sourceRole)
      || typeof citation.text !== 'string'
      || !citation.text.trim()) {
      throw new GroundingError('The frozen evidence pack contains an invalid citation.');
    }
    index.set(citation.citationId, citation);
  }
  return index;
};

export const resolveCitationCards = ({ citationIds, fieldId, evidence }) => {
  if (!uniqueStrings(citationIds)) throw new GroundingError('Citation IDs must be unique exact strings.');
  if (typeof fieldId !== 'string' || !fieldId) throw new GroundingError('fieldId is required.');
  const index = evidenceIndex(evidence);
  const allowedIds = Object.hasOwn(evidence.byField, fieldId) && Array.isArray(evidence.byField[fieldId])
    ? evidence.byField[fieldId]
    : [];
  if (!uniqueStrings(allowedIds)) throw new GroundingError('Field evidence IDs must be unique exact strings.');
  const allowed = new Set(allowedIds);
  return citationIds.map(citationId => {
    const stored = index.get(citationId);
    if (!allowed.has(citationId) || !stored) {
      throw new GroundingError(`Citation ${citationId} is not in the frozen evidence for ${fieldId}.`);
    }
    return {
      citationId: stored.citationId,
      sourceRole: stored.sourceRole,
      documentId: stored.documentId,
      documentVersion: stored.documentVersion,
      chunkId: stored.chunkId,
      quoteHash: stored.quoteHash,
      text: stored.text,
      locator: cloneLocator(stored.locator)
    };
  });
};

const canonicalUrl = (value) => {
  const raw = normalizeDisplay(value).replace(/[),.;!?]+$/u, '');
  if (!/^(?:https?:\/\/|www\.)/iu.test(raw)) return null;
  try {
    const url = new URL(/^www\./iu.test(raw) ? `https://${raw}` : raw);
    url.hash = '';
    if (url.pathname === '/') url.pathname = '';
    return url.toString().replace(/\/$/u, '');
  } catch {
    return null;
  }
};

const phoneDigits = (value) => {
  const normalized = normalizeDisplay(value);
  if (!/^\+?[\d().\s-]+$/u.test(normalized)) return null;
  const digits = normalized.replace(/\D/gu, '');
  return digits.length >= 7 && digits.length <= 15 ? digits : null;
};

const MONTHS = Object.freeze({
  jan: 1,
  january: 1,
  feb: 2,
  february: 2,
  mar: 3,
  march: 3,
  apr: 4,
  april: 4,
  may: 5,
  jun: 6,
  june: 6,
  jul: 7,
  july: 7,
  aug: 8,
  august: 8,
  sep: 9,
  sept: 9,
  september: 9,
  oct: 10,
  october: 10,
  nov: 11,
  november: 11,
  dec: 12,
  december: 12
});

const validIsoDate = (year, month, day) => {
  const date = new Date(Date.UTC(year, month - 1, day));
  if (date.getUTCFullYear() !== year || date.getUTCMonth() + 1 !== month || date.getUTCDate() !== day) return null;
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
};

const canonicalDate = (value) => {
  const normalized = normalizeDisplay(value);
  let match = /^(\d{4})-(\d{2})-(\d{2})$/u.exec(normalized);
  if (match) return validIsoDate(Number(match[1]), Number(match[2]), Number(match[3]));
  match = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/u.exec(normalized);
  if (match) return validIsoDate(Number(match[3]), Number(match[1]), Number(match[2]));
  match = /^([A-Za-z]+)\s+(\d{1,2})(?:,)?\s+(\d{4})$/u.exec(normalized);
  if (match && MONTHS[match[1].toLocaleLowerCase('en-US')]) {
    return validIsoDate(Number(match[3]), MONTHS[match[1].toLocaleLowerCase('en-US')], Number(match[2]));
  }
  return null;
};

const dateCandidates = (text) => {
  const patterns = [
    /\b\d{4}-\d{2}-\d{2}\b/gu,
    /\b\d{1,2}\/\d{1,2}\/\d{4}\b/gu,
    /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,)?\s+\d{4}\b/giu
  ];
  return patterns.flatMap(pattern => text.match(pattern) || []);
};

const phoneCandidates = (text) => (text.match(/(?:\+?\d[\d(). -]{5,}\d)/gu) || [])
  .filter(value => /[()+ -]/u.test(value));

const urlCandidates = text => (text.match(/\b(?:https?:\/\/|www\.)[^\s<>"']+/giu) || []);
const emailCandidates = text => (text.match(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/giu) || []);

const canonicalNumber = (value) => {
  const normalized = normalizeDisplay(value);
  const match = /^([$\u20ac\u00a3]?)(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(%)$/u.exec(normalized)
    || /^([$\u20ac\u00a3]?)(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)()$/u.exec(normalized);
  return match ? `${match[1]}${match[2].replaceAll(',', '')}${match[3]}` : null;
};

const numberCandidates = text => (text.match(/(?:[$\u20ac\u00a3])?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?/gu) || []);

const directOrCanonicalMatch = (value, evidenceText, field) => {
  const expected = normalizeDisplay(value);
  if (!expected) return false;
  const normalizedEvidence = normalizeDisplay(evidenceText);
  if (normalizedEvidence.includes(expected)) return true;

  const expectedPhone = phoneDigits(expected);
  if (expectedPhone && phoneCandidates(normalizedEvidence).some(candidate => phoneDigits(candidate) === expectedPhone)) {
    return true;
  }
  const expectedUrl = canonicalUrl(expected);
  if (expectedUrl && urlCandidates(normalizedEvidence).some(candidate => canonicalUrl(candidate) === expectedUrl)) {
    return true;
  }
  if (field?.type === 'date') {
    const expectedDate = canonicalDate(expected);
    if (expectedDate && dateCandidates(normalizedEvidence).some(candidate => canonicalDate(candidate) === expectedDate)) {
      return true;
    }
  }
  if (field?.type === 'number') {
    const expectedNumber = canonicalNumber(expected);
    if (expectedNumber && numberCandidates(normalizedEvidence).some(candidate => canonicalNumber(candidate) === expectedNumber)) {
      return true;
    }
  }
  return false;
};

const coherentValue = (proposal) => {
  if (typeof proposal.value !== 'string'
    || !uniqueStrings(proposal.selected_values)
    || typeof proposal.checked !== 'boolean') return false;
  if (proposal.action !== FIELD_ACTIONS.FILL) {
    return proposal.value_type === ANSWER_VALUE_TYPES.NONE
      && proposal.value === ''
      && proposal.selected_values.length === 0
      && proposal.checked === false;
  }
  if (proposal.value_type === ANSWER_VALUE_TYPES.TEXT) {
    return Boolean(proposal.value.trim()) && proposal.selected_values.length === 0 && proposal.checked === false;
  }
  if (proposal.value_type === ANSWER_VALUE_TYPES.SELECTED_VALUES) {
    return proposal.value === '' && proposal.selected_values.length > 0 && proposal.checked === false;
  }
  if (proposal.value_type === ANSWER_VALUE_TYPES.CHECKED) {
    return proposal.value === '' && proposal.selected_values.length === 0;
  }
  return false;
};

const exactValuesForProposal = (proposal) => {
  if (proposal.value_type === ANSWER_VALUE_TYPES.TEXT) return [proposal.value];
  if (proposal.value_type === ANSWER_VALUE_TYPES.SELECTED_VALUES) return proposal.selected_values;
  if (proposal.value_type === ANSWER_VALUE_TYPES.CHECKED) return [proposal.checked ? 'yes' : 'no'];
  return [];
};

const f1ValuesGrounded = (proposal, field, candidateCards) => {
  if (proposal.value_type === ANSWER_VALUE_TYPES.CHECKED) {
    const pattern = proposal.checked
      ? /\b(?:yes|true|checked)\b/iu
      : /\b(?:no|false|unchecked)\b/iu;
    return candidateCards.some(card => pattern.test(normalizeDisplay(card.text)));
  }
  const values = exactValuesForProposal(proposal);
  if (!values.length) return false;
  if (proposal.value_type === ANSWER_VALUE_TYPES.SELECTED_VALUES) {
    const options = new Set(field.options || []);
    if (values.some(value => !options.has(value))) return false;
  }
  return values.every(value => candidateCards.some(card => directOrCanonicalMatch(value, card.text, field)));
};

const concreteTokens = (text) => {
  const normalized = normalizeDisplay(text);
  const tokens = [
    ...emailCandidates(normalized).map(raw => ({ kind: 'email', raw, canonical: normalizeDisplay(raw) })),
    ...urlCandidates(normalized).map(raw => ({ kind: 'url', raw, canonical: canonicalUrl(raw) })),
    ...phoneCandidates(normalized).map(raw => ({ kind: 'phone', raw, canonical: phoneDigits(raw) })),
    ...dateCandidates(normalized).map(raw => ({ kind: 'date', raw, canonical: canonicalDate(raw) || normalizeDisplay(raw) })),
    ...numberCandidates(normalized).map(raw => ({ kind: 'number', raw, canonical: canonicalNumber(raw) || normalizeDisplay(raw) })),
    ...(normalized.match(/\b(?:PhD|DPhil|EdD|MD|JD|MBA|MSc|MS|MA|BSc|BS|BA)\b/gu) || [])
      .map(raw => ({ kind: 'credential', raw, canonical: raw }))
  ];
  const seen = new Set();
  return tokens.filter(token => {
    const key = `${token.kind}:${token.canonical}`;
    if (!token.canonical || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const tokenAppears = (token, evidenceText) => {
  const normalized = normalizeDisplay(evidenceText);
  if (normalized.includes(normalizeDisplay(token.raw))) return true;
  if (token.kind === 'url') return urlCandidates(normalized).some(value => canonicalUrl(value) === token.canonical);
  if (token.kind === 'phone') return phoneCandidates(normalized).some(value => phoneDigits(value) === token.canonical);
  if (token.kind === 'date') return dateCandidates(normalized).some(value => canonicalDate(value) === token.canonical);
  if (token.kind === 'number') return numberCandidates(normalized).some(value => canonicalNumber(value) === token.canonical);
  return false;
};

const f2DetailsGrounded = (proposal, cards) => concreteTokens(proposal.value)
  .every(token => cards.some(card => tokenAppears(token, card.text)));

const CLAIM_STOP_WORDS = new Set([
  'about', 'across', 'adjacent', 'after', 'also', 'am', 'an', 'and', 'are', 'as', 'at', 'background', 'be', 'because',
  'area', 'been', 'before', 'being', 'bring', 'but', 'by', 'call', 'can', 'cannot', 'candidate', 'claim', 'confidently', 'could', 'do', 'does', 'experience',
  'direct', 'document', 'documented', 'documents', 'dont', 'establish', 'evidence', 'for', 'from', 'had', 'has', 'have',
  'following', 'having', 'help', 'her', 'here', 'him', 'his', 'how', 'i', 'if', 'in', 'includes',
  'into', 'is', 'it', 'its', 'me', 'my', 'myself', 'of', 'on', 'or', 'our', 'role', 'she', 'skills', 'so',
  'material', 'materials', 'month', 'months', 'not', 'number', 'provided', 'resume', 'show', 'shown', 'specific', 'strong', 'supplied', 'support', 'than',
  'that', 'the', 'their', 'them', 'there', 'these', 'they', 'this', 'those', 'to', 'relevant',
  'report', 'using', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'work',
  'working', 'would', 'year', 'years', 'you', 'your'
]);
const CONTEXT_NOUNS = new Set([
  'company', 'customer', 'customers', 'employer', 'environment', 'industry', 'mission', 'office', 'offices',
  'opportunity', 'organization', 'position', 'product', 'products', 'role', 'team', 'teams'
]);
const CONTEXT_PREPOSITIONS = new Set(['at', 'for', 'their', 'with', 'your']);

const IRREGULAR_CLAIM_STEMS = new Map([
  ['built', 'build'],
  ['drove', 'drive'],
  ['led', 'lead'],
  ['ran', 'run'],
  ['wrote', 'write']
]);

const stemClaimToken = (value) => {
  let token = String(value || '').normalize('NFC').toLocaleLowerCase('en-US')
    .replace(/^[^\p{L}\p{N}+#]+|[^\p{L}\p{N}+#]+$/gu, '');
  token = IRREGULAR_CLAIM_STEMS.get(token) || token;
  if (token.length > 5 && token.endsWith('ies')) token = `${token.slice(0, -3)}y`;
  else if (token.length > 5 && token.endsWith('ing')) token = token.slice(0, -3);
  else if (token.length > 4 && token.endsWith('ed')) token = token.slice(0, -2);
  else if (token.length > 4 && token.endsWith('es')) token = token.slice(0, -2);
  else if (token.length > 3 && token.endsWith('s')) token = token.slice(0, -1);
  return token;
};

const claimTokens = text => normalizeDisplay(text).match(/[\p{L}\p{N}][\p{L}\p{N}+#.'-]*/gu) || [];

const termSet = cards => new Set(cards.flatMap(card => claimTokens(card.text)
  .map(stemClaimToken)
  .filter(Boolean)));

const isEvidenceGapSentence = (sentence) => {
  const normalized = normalizeDisplay(sentence).toLocaleLowerCase('en-US');
  const namesEvidence = /\b(?:resume|materials?|documents?|evidence)\b/iu.test(normalized);
  const negates = /\b(?:do not|does not|did not|not|no)\b/iu.test(normalized);
  const boundsClaim = /\b(?:document|establish|show|confirm|list|state|include)\w*\b/iu.test(normalized);
  const applicantUncertainty = /\bi\s+(?:(?:cannot|can not|can't)\s+(?:confidently\s+)?claim\b[^.!?]{0,120}\bdirect\b[^.!?]{0,80}\bexperience\b|(?:do not|don't)\s+have\b[^.!?]{0,80}\bspecific\s+number\b[^.!?]{0,80}\bdirect\b[^.!?]{0,80}\bexperience\b[^.!?]{0,40}\breport\b)/iu.test(normalized);
  return (namesEvidence && negates && boundsClaim) || applicantUncertainty;
};

const salaryProposalGrounded = (proposal, field, cards) => {
  if (!isSalaryExpectationField(field) || SALARY_UNSUPPORTED_PATTERN.test(proposal.value)) return false;
  const jobCards = cards.filter(card => card.sourceRole === SOURCE_ROLES.JOB_REQUIREMENT);
  const preferenceCards = cards.filter(card => card.sourceRole === SOURCE_ROLES.USER_VERIFIED
    && SALARY_PREFERENCE_EVIDENCE_PATTERN.test(card.text));
  const supportCards = [...jobCards, ...preferenceCards];
  if (!supportCards.length || !f2DetailsGrounded(proposal, supportCards)) return false;
  const supportTerms = termSet(supportCards);
  return claimTokens(proposal.value).every((raw) => {
    const normalized = raw.toLocaleLowerCase('en-US').replace(/[.'-]/gu, '');
    const stem = stemClaimToken(raw);
    return !stem
      || stem.length < 3
      || /^\d/u.test(stem)
      || CLAIM_STOP_WORDS.has(normalized)
      || SALARY_SAFE_TERMS.has(normalized)
      || supportTerms.has(stem);
  });
};

const f2SalientClaimsGrounded = (proposal, candidateCards, allCards, field) => {
  const candidateTerms = termSet(candidateCards);
  const allTerms = termSet(allCards);
  const fieldTerms = new Set(claimTokens(String(field?.label || '') + ' ' + String(field?.nearbyText || ''))
    .map(stemClaimToken)
    .filter(Boolean));
  const sentences = normalizeDisplay(proposal.value).split(/(?<=[.!?])\s+|\n+/u).filter(Boolean);
  for (const sentence of sentences) {
    const gapSentence = isEvidenceGapSentence(sentence);
    const tokens = claimTokens(sentence);
    for (let index = 0; index < tokens.length; index += 1) {
      const raw = tokens[index];
      const normalized = raw.toLocaleLowerCase('en-US').replace(/[.'’-]/gu, '');
      const stem = stemClaimToken(raw);
      if (!stem || stem.length < 3 || /^\d/u.test(stem) || CLAIM_STOP_WORDS.has(normalized)) continue;
      if (candidateTerms.has(stem)) continue;
      if (gapSentence && (allTerms.has(stem) || fieldTerms.has(stem))) continue;
      const previous = String(tokens[index - 1] || '').toLocaleLowerCase('en-US').replace(/[.'’-]/gu, '');
      const contextual = allTerms.has(stem)
        && (CONTEXT_NOUNS.has(normalized) || CONTEXT_PREPOSITIONS.has(previous));
      if (!contextual) return false;
    }
  }
  return true;
};

const nonFillable = (proposal, reason) => ({
  ...proposal,
  action: FIELD_ACTIONS.ASK_USER,
  confidence: PROPOSAL_CONFIDENCE.NEEDS_INPUT,
  value_type: ANSWER_VALUE_TYPES.NONE,
  value: '',
  selected_values: [],
  checked: false,
  citation_ids: [],
  short_rationale: 'Grounding review requires user input.',
  abstain_reason: String(reason || GROUNDING_REASONS.INCOHERENT_VALUE).slice(0, 500)
});

export const enforceGroundingAcceptance = ({ output, request, evidence }) => {
  if (!isPlainObject(output) || !Array.isArray(output.proposals) || !isPlainObject(request) || !Array.isArray(request.fields)) {
    throw new GroundingError('A structured page output and field request are required.');
  }
  const fieldsById = new Map(request.fields.map(field => [field.fieldId, field]));
  const proposals = output.proposals.map(proposal => {
    if (!isPlainObject(proposal) || typeof proposal.field_id !== 'string') {
      throw new GroundingError('Every proposal must have a field_id.');
    }
    const field = fieldsById.get(proposal.field_id);
    if (!field) throw new GroundingError(`Unknown proposal field: ${proposal.field_id}`);
    if (!Object.values(FIELD_RISK_CLASSES).includes(field.riskClass)
      || proposal.risk_class !== field.riskClass) {
      throw new GroundingError(`Proposal risk class does not match ${proposal.field_id}.`);
    }

    let cards;
    try {
      cards = resolveCitationCards({ citationIds: proposal.citation_ids, fieldId: field.fieldId, evidence });
    } catch {
      return nonFillable(proposal, GROUNDING_REASONS.INVALID_CITATIONS);
    }
    if (!Object.values(FIELD_ACTIONS).includes(proposal.action)
      || !Object.values(PROPOSAL_CONFIDENCE).includes(proposal.confidence)
      || !Object.values(ANSWER_VALUE_TYPES).includes(proposal.value_type)) {
      return nonFillable(proposal, GROUNDING_REASONS.INCOHERENT_VALUE);
    }
    if (!coherentValue(proposal)) return nonFillable(proposal, GROUNDING_REASONS.INCOHERENT_VALUE);
    if (proposal.action !== FIELD_ACTIONS.FILL) return { ...proposal };
    if (field.manual) return nonFillable(proposal, GROUNDING_REASONS.MANUAL_FIELD);
    if (proposal.confidence === PROPOSAL_CONFIDENCE.NEEDS_INPUT) {
      return nonFillable(proposal, GROUNDING_REASONS.MISSING_CANDIDATE_EVIDENCE);
    }

    const candidateCards = cards.filter(card => ELIGIBLE_CANDIDATE_ROLES.has(card.sourceRole));
    if (field.riskClass === FIELD_RISK_CLASSES.VERIFIED) {
      if (!candidateCards.length) return nonFillable(proposal, GROUNDING_REASONS.MISSING_CANDIDATE_EVIDENCE);
      if (!f1ValuesGrounded(proposal, field, candidateCards)) {
        return nonFillable(proposal, GROUNDING_REASONS.UNSUPPORTED_EXACT_VALUE);
      }
      return { ...proposal };
    }

    if (isSalaryExpectationField(field)) {
      if (proposal.value_type === ANSWER_VALUE_TYPES.TEXT && salaryProposalGrounded(proposal, field, cards)) {
        return { ...proposal, confidence: PROPOSAL_CONFIDENCE.REVIEW };
      }
      return nonFillable(proposal, GROUNDING_REASONS.UNSUPPORTED_SALIENT_CLAIM);
    }
    if (!candidateCards.length) return nonFillable(proposal, GROUNDING_REASONS.MISSING_CANDIDATE_EVIDENCE);
    if (proposal.value_type !== ANSWER_VALUE_TYPES.TEXT) {
      if (!f1ValuesGrounded(proposal, field, candidateCards)) {
        return nonFillable(proposal, GROUNDING_REASONS.UNSUPPORTED_EXACT_VALUE);
      }
    } else if (!f2DetailsGrounded(proposal, cards)) {
      return nonFillable(proposal, GROUNDING_REASONS.UNSUPPORTED_CONCRETE_DETAIL);
    } else if (!f2SalientClaimsGrounded(proposal, candidateCards, cards, field)) {
      return nonFillable(proposal, GROUNDING_REASONS.UNSUPPORTED_SALIENT_CLAIM);
    }
    return {
      ...proposal,
      confidence: PROPOSAL_CONFIDENCE.REVIEW
    };
  });
  return { ...output, proposals };
};

export const enforceFieldRegenerationGrounding = ({ output, field, evidence }) => {
  if (!isPlainObject(output) || !isPlainObject(field)) {
    throw new GroundingError('A regenerated field output and descriptor are required.');
  }
  const fillLike = output.value_type !== ANSWER_VALUE_TYPES.NONE
    && output.confidence !== PROPOSAL_CONFIDENCE.NEEDS_INPUT
    && !output.abstain_reason;
  const proposal = {
    field_id: output.field_id,
    action: fillLike ? FIELD_ACTIONS.FILL : FIELD_ACTIONS.ASK_USER,
    confidence: output.confidence,
    risk_class: output.risk_class,
    value_type: output.value_type,
    value: output.value,
    selected_values: output.selected_values,
    checked: output.checked,
    citation_ids: output.citation_ids,
    short_rationale: output.changes_summary,
    abstain_reason: output.abstain_reason
  };
  const grounded = enforceGroundingAcceptance({
    output: { page_id: 'field-regeneration', proposals: [proposal] },
    request: { fields: [field] },
    evidence
  }).proposals[0];
  return {
    field_id: grounded.field_id,
    confidence: grounded.confidence,
    risk_class: grounded.risk_class,
    value_type: grounded.value_type,
    value: grounded.value,
    selected_values: grounded.selected_values,
    checked: grounded.checked,
    citation_ids: grounded.citation_ids,
    changes_summary: grounded.action === FIELD_ACTIONS.FILL
      ? output.changes_summary
      : 'Grounding review requires user input.',
    abstain_reason: grounded.abstain_reason
  };
};
