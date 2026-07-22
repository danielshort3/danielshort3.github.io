import { canonicalJson, sha256Base64Url } from '../vault/crypto.js';
import {
  ANSWER_VALUE_TYPES,
  FIELD_ACTIONS,
  FIELD_RISK_CLASSES,
  PROPOSAL_CONFIDENCE
} from '../shared/schemas.js';

export const APPLICATION_PACKET_SCHEMA_VERSION = 1;
export const APPLICATION_PACKET_RECORD_KIND = 'application-packet';

export const APPLICATION_CONTEXT_ORIGINS = Object.freeze({
  LIVE_PAGE: 'live_page',
  MANUAL: 'manual'
});

export const APPLICATION_CONTEXT_MODES = Object.freeze({
  POSTING: 'posting',
  APPLICATION: 'application',
  MANUAL: 'manual'
});

const MAX_CONTEXT_CONTENT_LENGTH = 20_000;
const MAX_CONTEXT_PREVIEW_LENGTH = 240;
const MAX_CONTEXT_SNAPSHOTS = 16;
const MAX_APPROVED_ANSWERS = 100;
const MAX_CLARIFICATIONS = 100;
const TRACKING_QUERY_PATTERN = /^(?:utm_.+|gclid|dclid|fbclid|msclkid|mc_[ce]id)$/iu;
const REQUISITION_QUERY_KEYS = new Set([
  'gh_jid',
  'job',
  'jobid',
  'job_id',
  'jobreqid',
  'postingid',
  'req',
  'reqid',
  'requisition',
  'requisitionid'
]);

const isPlainObject = value => Boolean(value)
  && typeof value === 'object'
  && !Array.isArray(value);

const boundedString = (value, maxLength, label, { allowEmpty = true } = {}) => {
  if (value === undefined || value === null) return '';
  if (typeof value !== 'string') throw new TypeError(`${label} must be a string.`);
  const normalized = value.normalize('NFKC').trim();
  if (!allowEmpty && !normalized) throw new TypeError(`${label} is required.`);
  if (normalized.length > maxLength) throw new TypeError(`${label} exceeds ${maxLength} characters.`);
  return normalized;
};

const canonicalTimestamp = (value, label) => {
  const date = value instanceof Date ? value : new Date(value || Date.now());
  if (Number.isNaN(date.getTime())) throw new TypeError(`${label} must be a valid date.`);
  return date.toISOString();
};

const normalizeContent = value => String(value || '')
  .normalize('NFKC')
  .replace(/\r\n?/gu, '\n')
  .replace(/[\t ]+\n/gu, '\n')
  .replace(/\n{3,}/gu, '\n\n')
  .trim();

const compactText = value => String(value || '').replace(/\s+/gu, ' ').trim();

const normalizeIdentityText = (value, { company = false } = {}) => {
  let normalized = compactText(String(value || '').normalize('NFKC'))
    .toLocaleLowerCase('en-US')
    .replace(/&/gu, ' and ')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/gu, ' ')
    .trim();
  if (company) {
    normalized = normalized
      .replace(/\s+(?:co|company|corp|corporation|inc|incorporated|llc|llp|ltd|limited|plc)$/u, '')
      .trim();
  }
  return normalized;
};

const canonicalHttpUrl = (value, label = 'context URL') => {
  const raw = String(value || '').trim();
  if (!raw) return '';
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    throw new TypeError(`${label} must be a valid URL.`);
  }
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new TypeError(`${label} must be an HTTP(S) URL without credentials.`);
  }
  parsed.hash = '';
  parsed.hostname = parsed.hostname.toLocaleLowerCase('en-US');
  if ((parsed.protocol === 'https:' && parsed.port === '443')
    || (parsed.protocol === 'http:' && parsed.port === '80')) parsed.port = '';
  const retainedParameters = [...parsed.searchParams.entries()]
    .filter(([key]) => !TRACKING_QUERY_PATTERN.test(key))
    .sort(([leftKey, leftValue], [rightKey, rightValue]) => leftKey.localeCompare(rightKey)
      || leftValue.localeCompare(rightValue));
  parsed.search = '';
  retainedParameters.forEach(([key, parameterValue]) => parsed.searchParams.append(key, parameterValue));
  parsed.pathname = parsed.pathname.replace(/\/{2,}/gu, '/').replace(/\/$/u, '') || '/';
  return parsed.toString();
};

const normalizeRequisitionId = value => compactText(value)
  .toLocaleLowerCase('en-US')
  .replace(/[^a-z0-9._-]+/gu, '')
  .slice(0, 80);

const requisitionIdsFromUrl = value => {
  if (!value) return [];
  const parsed = new URL(value);
  const identifiers = [];
  parsed.searchParams.forEach((parameterValue, key) => {
    if (REQUISITION_QUERY_KEYS.has(key.toLocaleLowerCase('en-US'))) {
      const normalized = normalizeRequisitionId(parameterValue);
      if (normalized.length >= 3) identifiers.push(normalized);
    }
  });
  const pathSegments = parsed.pathname.split('/').filter(Boolean);
  const finalSegment = normalizeRequisitionId(pathSegments.at(-1));
  if (finalSegment.length >= 4 && /\d/u.test(finalSegment)) identifiers.push(finalSegment);
  return [...new Set(identifiers)];
};

const contextIdentity = context => {
  const job = context?.job || {};
  const company = normalizeIdentityText(job.company, { company: true });
  const title = normalizeIdentityText(job.title);
  const location = normalizeIdentityText(job.location);
  const url = canonicalHttpUrl(context?.url || job.jobUrl || '', 'job URL');
  const explicitRequisitionId = normalizeRequisitionId(context?.requisitionId || job.requisitionId || '');
  const requisitionIds = [...new Set([
    ...(explicitRequisitionId ? [explicitRequisitionId] : []),
    ...requisitionIdsFromUrl(url)
  ])];
  const roleKey = company && title ? `${company}|${title}` : '';
  const roleLocationKey = roleKey && location ? `${roleKey}|${location}` : '';
  return {
    company,
    title,
    location,
    roleKey,
    roleLocationKey,
    urls: url ? [url] : [],
    requisitionIds
  };
};

const mergeUnique = (...collections) => [...new Set(collections.flat().filter(Boolean))].sort();

const assertOriginAndMode = (origin, mode) => {
  if (!Object.values(APPLICATION_CONTEXT_ORIGINS).includes(origin)) {
    throw new TypeError(`Unsupported application context origin: ${String(origin)}`);
  }
  if (!Object.values(APPLICATION_CONTEXT_MODES).includes(mode)) {
    throw new TypeError(`Unsupported application context mode: ${String(mode)}`);
  }
  if (origin === APPLICATION_CONTEXT_ORIGINS.MANUAL && mode !== APPLICATION_CONTEXT_MODES.MANUAL) {
    throw new TypeError('Manual application context must use manual mode.');
  }
  if (origin === APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE && mode === APPLICATION_CONTEXT_MODES.MANUAL) {
    throw new TypeError('Live-page application context cannot use manual mode.');
  }
};

const normalizePageMetadata = page => {
  if (page === undefined || page === null) return null;
  if (!isPlainObject(page)) throw new TypeError('page metadata must be an object.');
  const domRevision = page.domRevision === undefined ? null : page.domRevision;
  if (domRevision !== null && (!Number.isSafeInteger(domRevision) || domRevision < 0)) {
    throw new TypeError('page.domRevision must be a non-negative safe integer.');
  }
  return {
    adapter: boundedString(page.adapter, 64, 'page.adapter'),
    pageId: boundedString(page.pageId, 200, 'page.pageId'),
    urlHash: boundedString(page.urlHash, 200, 'page.urlHash'),
    domRevision
  };
};

export const createApplicationContextSnapshot = async (input = {}) => {
  if (!isPlainObject(input)) throw new TypeError('Application context must be an object.');
  const origin = input.origin || APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE;
  const mode = input.mode || (origin === APPLICATION_CONTEXT_ORIGINS.MANUAL
    ? APPLICATION_CONTEXT_MODES.MANUAL
    : APPLICATION_CONTEXT_MODES.APPLICATION);
  assertOriginAndMode(origin, mode);
  const jobInput = isPlainObject(input.job) ? input.job : {};
  const jobUrl = canonicalHttpUrl(input.url || jobInput.jobUrl || '', 'job URL');
  const contentSource = input.content ?? jobInput.description ?? '';
  const normalizedSourceContent = normalizeContent(contentSource);
  const content = normalizedSourceContent.slice(0, MAX_CONTEXT_CONTENT_LENGTH);
  const company = boundedString(jobInput.company, 240, 'job.company');
  const title = boundedString(jobInput.title, 240, 'job.title');
  const location = boundedString(jobInput.location, 240, 'job.location');
  if (!company && !title && !jobUrl && !content) {
    throw new TypeError('Application context requires job identity, a URL, or content.');
  }
  const capturedAt = canonicalTimestamp(input.capturedAt, 'capturedAt');
  const identity = contextIdentity({ ...input, url: jobUrl, job: { ...jobInput, company, title, location, jobUrl } });
  const previewText = compactText(content).slice(0, MAX_CONTEXT_PREVIEW_LENGTH);
  const preview = {
    text: previewText,
    sourceLength: normalizedSourceContent.length,
    retainedLength: content.length,
    truncated: normalizedSourceContent.length > MAX_CONTEXT_PREVIEW_LENGTH,
    contentTruncated: normalizedSourceContent.length > content.length,
    sha256: await sha256Base64Url(content)
  };
  const source = {
    origin,
    mode,
    label: boundedString(input.label || jobInput.source || title || 'Application context', 160, 'source label', { allowEmpty: false }),
    url: jobUrl
  };
  const job = {
    company,
    title,
    location,
    source: boundedString(jobInput.source, 120, 'job.source'),
    jobUrl,
    requisitionId: boundedString(input.requisitionId || jobInput.requisitionId, 80, 'job.requisitionId')
  };
  const page = normalizePageMetadata(input.page);
  const contextId = `context:${(await sha256Base64Url(canonicalJson({
    source,
    job,
    page: page ? { adapter: page.adapter, pageId: page.pageId, urlHash: page.urlHash } : null,
    contentSha256: preview.sha256
  }))).slice(0, 32)}`;
  return {
    contextId,
    capturedAt,
    source,
    job,
    page,
    content,
    preview,
    identity
  };
};

const questionIdentity = field => {
  if (!isPlainObject(field)) throw new TypeError('A field descriptor is required.');
  const fieldId = boundedString(field.fieldId, 200, 'field.fieldId', { allowEmpty: false });
  const label = boundedString(field.label, 500, 'field.label', { allowEmpty: false });
  const type = boundedString(field.type, 100, 'field.type', { allowEmpty: false });
  const riskClass = boundedString(field.riskClass, 40, 'field.riskClass', { allowEmpty: false });
  return {
    fieldId,
    label,
    type,
    riskClass,
    normalizedQuestion: `${normalizeIdentityText(label)}|${normalizeIdentityText(type)}`
  };
};

const uniqueBoundedStrings = (values, label, maxItems = 20, maxLength = 200) => {
  if (!Array.isArray(values) || values.length > maxItems) throw new TypeError(`${label} must be a bounded array.`);
  const normalized = values.map((value, index) => boundedString(value, maxLength, `${label}[${index}]`, { allowEmpty: false }));
  if (new Set(normalized).size !== normalized.length) throw new TypeError(`${label} must contain unique values.`);
  return normalized;
};

export const createApprovedAnswerSnapshot = async ({
  applicationId,
  field,
  proposal,
  approved = false,
  approvedAt,
  sourceSignature = '',
  contextId = ''
} = {}) => {
  if (approved !== true) throw new TypeError('Approved answer snapshots require explicit user approval.');
  const normalizedApplicationId = boundedString(applicationId, 200, 'applicationId', { allowEmpty: false });
  const question = questionIdentity(field);
  if (!isPlainObject(proposal) || proposal.action !== FIELD_ACTIONS.FILL || proposal.field_id !== question.fieldId) {
    throw new TypeError('Approved answer snapshot requires a fill proposal for the same field.');
  }
  const valueType = boundedString(proposal.value_type, 40, 'proposal.value_type', { allowEmpty: false });
  if (!Object.values(ANSWER_VALUE_TYPES).includes(valueType)) {
    throw new TypeError('proposal.value_type is unsupported.');
  }
  if (!Object.values(FIELD_RISK_CLASSES).includes(question.riskClass)) {
    throw new TypeError('field.riskClass is unsupported.');
  }
  if (![PROPOSAL_CONFIDENCE.HIGH, PROPOSAL_CONFIDENCE.REVIEW].includes(proposal.confidence)) {
    throw new TypeError('Approved answers require high or review confidence.');
  }
  const value = boundedString(proposal.value, 12_000, 'proposal.value');
  const selectedValues = uniqueBoundedStrings(proposal.selected_values || [], 'proposal.selected_values', 100, 1000);
  if (typeof proposal.checked !== 'boolean') throw new TypeError('proposal.checked must be boolean.');
  const citationIds = uniqueBoundedStrings(proposal.citation_ids || [], 'proposal.citation_ids', 12, 200);
  if (!citationIds.length) throw new TypeError('Approved answers must retain at least one citation.');
  const normalizedApprovedAt = canonicalTimestamp(approvedAt, 'approvedAt');
  const answerKey = `answer:${(await sha256Base64Url(question.normalizedQuestion)).slice(0, 32)}`;
  const answer = { valueType, value, selectedValues, checked: proposal.checked };
  return {
    snapshotId: `approved:${(await sha256Base64Url(canonicalJson({
      applicationId: normalizedApplicationId,
      answerKey,
      answer,
      citationIds,
      approvedAt: normalizedApprovedAt
    }))).slice(0, 32)}`,
    answerKey,
    applicationId: normalizedApplicationId,
    fieldId: question.fieldId,
    fieldFingerprint: boundedString(field.fingerprint, 200, 'field.fingerprint'),
    label: question.label,
    type: question.type,
    riskClass: question.riskClass,
    confidence: boundedString(proposal.confidence, 40, 'proposal.confidence', { allowEmpty: false }),
    answer,
    citationIds,
    sourceSignature: boundedString(sourceSignature, 200, 'sourceSignature'),
    contextId: boundedString(contextId, 200, 'contextId'),
    approvedAt: normalizedApprovedAt,
    approvedBy: 'user'
  };
};

export const createClarificationSnapshot = async ({
  applicationId,
  field,
  text,
  verified = false,
  verifiedAt,
  citationIds = [],
  evidenceRecordId = '',
  contextId = ''
} = {}) => {
  if (verified !== true) throw new TypeError('Clarification snapshots require explicit user verification.');
  const normalizedApplicationId = boundedString(applicationId, 200, 'applicationId', { allowEmpty: false });
  const question = questionIdentity(field);
  const normalizedText = boundedString(text, 4000, 'clarification text', { allowEmpty: false });
  const normalizedCitationIds = uniqueBoundedStrings(citationIds, 'citationIds', 12, 200);
  const normalizedEvidenceRecordId = boundedString(evidenceRecordId, 200, 'evidenceRecordId');
  if (!normalizedCitationIds.length && !normalizedEvidenceRecordId) {
    throw new TypeError('Clarification snapshots require citations or an encrypted evidence record link.');
  }
  const normalizedVerifiedAt = canonicalTimestamp(verifiedAt, 'verifiedAt');
  const clarificationKey = `clarification:${(await sha256Base64Url(question.normalizedQuestion)).slice(0, 32)}`;
  return {
    snapshotId: `clarified:${(await sha256Base64Url(canonicalJson({
      applicationId: normalizedApplicationId,
      clarificationKey,
      text: normalizedText,
      verifiedAt: normalizedVerifiedAt
    }))).slice(0, 32)}`,
    clarificationKey,
    applicationId: normalizedApplicationId,
    fieldId: question.fieldId,
    label: question.label,
    text: normalizedText,
    citationIds: normalizedCitationIds,
    evidenceRecordId: normalizedEvidenceRecordId,
    contextId: boundedString(contextId, 200, 'contextId'),
    verifiedAt: normalizedVerifiedAt,
    verifiedBy: 'user'
  };
};

const isCopyOnlyField = field => field?.fillMode === 'copy_only';

export const summarizeApplicationPreflight = ({
  fields = [],
  proposals = [],
  filledFieldIds = [],
  unsupportedCount = 0
} = {}) => {
  if (!Array.isArray(fields) || !Array.isArray(proposals)) {
    throw new TypeError('Preflight fields and proposals must be arrays.');
  }
  if (!Number.isSafeInteger(unsupportedCount) || unsupportedCount < 0) {
    throw new TypeError('unsupportedCount must be a non-negative safe integer.');
  }
  if (!(filledFieldIds instanceof Set) && !Array.isArray(filledFieldIds)) {
    throw new TypeError('filledFieldIds must be an array or Set.');
  }
  const filled = new Set(filledFieldIds instanceof Set ? [...filledFieldIds] : filledFieldIds);
  const proposalsByField = new Map();
  proposals.forEach(proposal => {
    if (proposal?.field_id && !proposalsByField.has(proposal.field_id)) proposalsByField.set(proposal.field_id, proposal);
  });
  const summary = {
    total: fields.length,
    filled: 0,
    verifiedReady: 0,
    reviewRequired: 0,
    needsInput: 0,
    copyOnly: 0,
    manual: 0,
    skipped: 0,
    unsupported: unsupportedCount,
    fillable: 0,
    requiresUserAction: 0,
    remaining: 0
  };
  fields.forEach(field => {
    const fieldId = String(field?.fieldId || '');
    const proposal = proposalsByField.get(fieldId);
    if (filled.has(fieldId)) {
      summary.filled += 1;
    } else if (field?.manual) {
      summary.manual += 1;
    } else if (isCopyOnlyField(field)) {
      summary.copyOnly += 1;
    } else if (!proposal || proposal.action === FIELD_ACTIONS.ASK_USER
      || proposal.confidence === PROPOSAL_CONFIDENCE.NEEDS_INPUT) {
      summary.needsInput += 1;
    } else if (proposal.action === FIELD_ACTIONS.SKIP) {
      summary.skipped += 1;
    } else if (proposal.action === FIELD_ACTIONS.FILL
      && field?.riskClass === FIELD_RISK_CLASSES.VERIFIED
      && proposal.confidence === PROPOSAL_CONFIDENCE.HIGH) {
      summary.verifiedReady += 1;
    } else if (proposal.action === FIELD_ACTIONS.FILL) {
      summary.reviewRequired += 1;
    } else {
      summary.needsInput += 1;
    }
  });
  summary.fillable = summary.verifiedReady + summary.reviewRequired;
  summary.requiresUserAction = summary.reviewRequired + summary.needsInput + summary.copyOnly + summary.manual;
  summary.remaining = summary.total - summary.filled - summary.skipped;
  return Object.freeze(summary);
};

const isPacketRecord = record => record?.kind === APPLICATION_PACKET_RECORD_KIND
  && record?.value?.schemaVersion === APPLICATION_PACKET_SCHEMA_VERSION
  && typeof record.value.applicationId === 'string'
  && isPlainObject(record.value.identity);

const contextCompatibleCompany = (left, right) => !left || !right || left === right;

const scorePacketMatch = (record, inputIdentity, applicationIdHint) => {
  const packet = record.value;
  if (applicationIdHint && packet.applicationId === applicationIdHint) return { score: 120, reason: 'application_id' };
  const aliases = packet.identity.aliases || {};
  const packetUrls = new Set(aliases.urls || []);
  if (inputIdentity.urls.some(url => packetUrls.has(url))) return { score: 110, reason: 'url' };
  const packetRequisitionIds = new Set(aliases.requisitionIds || []);
  if (contextCompatibleCompany(packet.identity.company, inputIdentity.company)
    && inputIdentity.requisitionIds.some(identifier => packetRequisitionIds.has(identifier))) {
    return { score: 100, reason: 'requisition_id' };
  }
  const packetRoleLocations = new Set(aliases.roleLocationKeys || []);
  if (inputIdentity.roleLocationKey && packetRoleLocations.has(inputIdentity.roleLocationKey)) {
    return { score: 90, reason: 'company_title_location' };
  }
  const packetRoles = new Set(aliases.roleKeys || []);
  if (inputIdentity.roleKey && packetRoles.has(inputIdentity.roleKey)) {
    return { score: 70, reason: 'company_title' };
  }
  return { score: 0, reason: '' };
};

export const matchApplicationPacketRecord = (records = [], input = {}) => {
  if (!Array.isArray(records)) throw new TypeError('Application packet records must be an array.');
  if (!isPlainObject(input)) throw new TypeError('Application context must be an object.');
  const identity = contextIdentity(input);
  const applicationIdHint = boundedString(input.applicationId, 200, 'applicationId');
  const matches = records
    .filter(isPacketRecord)
    .map(record => ({ record, ...scorePacketMatch(record, identity, applicationIdHint) }))
    .filter(match => match.score > 0)
    .sort((left, right) => right.score - left.score || left.record.id.localeCompare(right.record.id));
  if (!matches.length) return null;
  if (matches[1]?.score === matches[0].score) return null;
  return matches[0];
};

const assertPacketSnapshot = (snapshot, applicationId, label, keyName) => {
  if (!isPlainObject(snapshot) || snapshot.applicationId !== applicationId
    || typeof snapshot[keyName] !== 'string' || !snapshot[keyName]) {
    throw new TypeError(`${label} does not belong to this application packet.`);
  }
  return snapshot;
};

const assertPreflightSummary = preflight => {
  if (!isPlainObject(preflight)) throw new TypeError('preflight must be a summary returned by summarizeApplicationPreflight.');
  const countKeys = [
    'total',
    'filled',
    'verifiedReady',
    'reviewRequired',
    'needsInput',
    'copyOnly',
    'manual',
    'skipped',
    'unsupported',
    'fillable',
    'requiresUserAction',
    'remaining'
  ];
  if (Object.keys(preflight).length !== countKeys.length
    || countKeys.some(key => !Number.isSafeInteger(preflight[key]) || preflight[key] < 0)
    || preflight.fillable !== preflight.verifiedReady + preflight.reviewRequired
    || preflight.requiresUserAction !== preflight.reviewRequired + preflight.needsInput
      + preflight.copyOnly + preflight.manual
    || preflight.remaining !== preflight.total - preflight.filled - preflight.skipped
    || preflight.total !== preflight.filled + preflight.verifiedReady + preflight.reviewRequired
      + preflight.needsInput + preflight.copyOnly + preflight.manual + preflight.skipped) {
    throw new TypeError('preflight must be a summary returned by summarizeApplicationPreflight.');
  }
  return preflight;
};

const mergeLatestByKey = (existing, incoming, keyName, timestampName, limit) => {
  const merged = new Map();
  [...existing, ...incoming].forEach(item => {
    const current = merged.get(item[keyName]);
    if (!current || String(item[timestampName]) >= String(current[timestampName])) merged.set(item[keyName], item);
  });
  return [...merged.values()]
    .sort((left, right) => String(right[timestampName]).localeCompare(String(left[timestampName])))
    .slice(0, limit);
};

export const upsertApplicationPacketRecord = async ({
  existingRecord = null,
  context,
  approvedAnswers = [],
  clarifications = [],
  preflight = null,
  now
} = {}) => {
  if (existingRecord && !isPacketRecord(existingRecord)) {
    throw new TypeError('existingRecord is not a supported application packet.');
  }
  if (!Array.isArray(approvedAnswers) || !Array.isArray(clarifications)) {
    throw new TypeError('Application packet snapshots must be arrays.');
  }
  const contextSnapshot = await createApplicationContextSnapshot(context);
  const updatedAt = canonicalTimestamp(now, 'now');
  const identity = contextSnapshot.identity;
  const seed = canonicalJson({
    roleKey: identity.roleKey,
    roleLocationKey: identity.roleLocationKey,
    requisitionIds: identity.requisitionIds,
    urls: identity.urls
  });
  const generatedApplicationId = `local-app:${(await sha256Base64Url(seed || contextSnapshot.contextId)).slice(0, 32)}`;
  const applicationId = existingRecord?.value?.applicationId || generatedApplicationId;
  approvedAnswers.forEach(snapshot => assertPacketSnapshot(snapshot, applicationId, 'Approved answer snapshot', 'answerKey'));
  clarifications.forEach(snapshot => assertPacketSnapshot(snapshot, applicationId, 'Clarification snapshot', 'clarificationKey'));
  if (preflight !== null) assertPreflightSummary(preflight);
  const previous = existingRecord?.value || {};
  const previousIdentity = previous.identity || {};
  const previousAliases = previousIdentity.aliases || {};
  const aliases = {
    urls: mergeUnique(previousAliases.urls || [], identity.urls),
    requisitionIds: mergeUnique(previousAliases.requisitionIds || [], identity.requisitionIds),
    roleKeys: mergeUnique(previousAliases.roleKeys || [], identity.roleKey ? [identity.roleKey] : []),
    roleLocationKeys: mergeUnique(
      previousAliases.roleLocationKeys || [],
      identity.roleLocationKey ? [identity.roleLocationKey] : []
    )
  };
  const { identity: _contextIdentity, ...storedContextSnapshot } = contextSnapshot;
  const contextSnapshots = mergeLatestByKey(
    previous.contextSnapshots || [],
    [storedContextSnapshot],
    'contextId',
    'capturedAt',
    MAX_CONTEXT_SNAPSHOTS
  );
  const value = {
    schemaVersion: APPLICATION_PACKET_SCHEMA_VERSION,
    applicationId,
    createdAt: previous.createdAt || updatedAt,
    updatedAt,
    identity: {
      company: identity.company || previousIdentity.company || '',
      title: identity.title || previousIdentity.title || '',
      location: identity.location || previousIdentity.location || '',
      aliases
    },
    contextSnapshots,
    approvedAnswers: mergeLatestByKey(
      previous.approvedAnswers || [],
      approvedAnswers,
      'answerKey',
      'approvedAt',
      MAX_APPROVED_ANSWERS
    ),
    clarifications: mergeLatestByKey(
      previous.clarifications || [],
      clarifications,
      'clarificationKey',
      'verifiedAt',
      MAX_CLARIFICATIONS
    ),
    preflight: preflight || previous.preflight || null,
    lastContextId: contextSnapshot.contextId
  };
  return {
    id: `application-packet:${applicationId}`,
    kind: APPLICATION_PACKET_RECORD_KIND,
    value
  };
};
