import {
  APPLICATION_CONTEXT_MODES,
  APPLICATION_CONTEXT_ORIGINS,
  APPLICATION_PACKET_RECORD_KIND,
  createApprovedAnswerSnapshot,
  createClarificationSnapshot,
  matchApplicationPacketRecord,
  summarizeApplicationPreflight,
  upsertApplicationPacketRecord
} from '../application/application-packet.js';
import { enforceGroundingAcceptance, isSalaryExpectationField, resolveCitationCards } from '../grounding/postprocessor.js';
import { assertLocalModelName, OllamaClient } from '../ollama/client.js';
import { normalizeFieldRegenerationFeedback } from '../ollama/prompts.js';
import {
  GENERATION_ERROR_CODES,
  GENERATION_STATUS_CODES,
  embedChunksWithLexicalFallback,
  orchestrateFieldRegeneration,
  orchestratePageGeneration
} from '../ollama/generation-orchestrator.js';
import { createDocumentVaultRecord } from '../parsers/import-record.js';
import { chunkDocument } from '../rag/chunker.js';
import { buildEvidencePack, hybridRetrieve } from '../rag/retrieval.js';
import { ManualResearchProvider } from '../research/manual-research-provider.js';
import {
  PRIVACY_CONSENT_KEY,
  PRIVACY_NOTICE_VERSION
} from '../shared/privacy-consent.js';
import { DEFAULT_LOCAL_MODEL_CONFIG, SOURCE_ROLES, VALIDATION_LIMITS } from '../shared/schemas.js';
import { TRACKER_INTERNAL_MESSAGE_TYPES } from '../shared/tracker-protocol.js';
import { base64ToBytes, canonicalJson, sha256Base64Url } from '../vault/crypto.js';
import { EncryptedIndexedDbVault } from '../vault/indexeddb-vault.js';
import {
  carrySourceSelection as carryCatalogueSourceSelection,
  importedSourceApplicationId,
  isReusableSourceRecord as isReusableCatalogueSourceRecord,
  recommendedSourceSelection,
  selectImportedSource,
  sourceCatalogueSummary
} from '../vault/source-catalogue.js';

export { PRIVACY_CONSENT_KEY, PRIVACY_NOTICE_VERSION };
export { normalizeFieldRegenerationFeedback };
export {
  importedSourceApplicationId,
  recommendedSourceSelection,
  selectImportedSource,
  sourceCatalogueSummary
};

const RUNTIME_CHANNEL = 'job-application-copilot';
const RUNTIME_VERSION = 1;
const MODEL_SETTINGS_KEY = 'jobCopilotModelSettingsV1';
const SELECTED_FIELD_KEY = 'jobCopilotSelectedField';
const TRACKER_PENDING_KEY_PREFIX = 'jobCopilotPendingTrackerCapture:';
const VAULT_SESSION_KEY_PREFIX = 'jobCopilotVaultKey:';
const CACHE_SCHEMA_VERSION = 10;
const MAX_TRACKER_FILE_BYTES = 10 * 1024 * 1024;
const MAX_SOURCE_FILE_BYTES = 10 * 1024 * 1024;
const UNSCOPED_SOURCE_SELECTION = 'unscoped';
const PREVIEW_MODE = typeof location !== 'undefined'
  && new URL(location.href).searchParams.get('preview') === '1';

export const FACT_DEFINITIONS = Object.freeze({
  first_name: Object.freeze({ label: 'First name', aliases: ['first name', 'given name', 'legal first name'] }),
  last_name: Object.freeze({ label: 'Last name', aliases: ['last name', 'surname', 'family name', 'legal last name'] }),
  full_name: Object.freeze({ label: 'Full name', aliases: ['full name', 'legal name', 'applicant name', 'candidate name', 'name'] }),
  email: Object.freeze({ label: 'Email address', aliases: ['email', 'email address', 'e mail', 'e mail address', 'applicant email', 'candidate email'] }),
  phone: Object.freeze({ label: 'Phone number', aliases: ['phone', 'phone number', 'mobile', 'mobile phone', 'telephone', 'candidate phone'] }),
  street_address: Object.freeze({ label: 'Street address', aliases: ['street address', 'address line 1', 'home address', 'mailing address'] }),
  city: Object.freeze({ label: 'City', aliases: ['city', 'home city', 'mailing city'] }),
  state_province: Object.freeze({ label: 'State or province', aliases: ['state', 'province', 'state or province', 'state province', 'region'] }),
  postal_code: Object.freeze({ label: 'Postal code', aliases: ['postal code', 'zip code', 'zip', 'postcode'] }),
  country: Object.freeze({ label: 'Country', aliases: ['country', 'country of residence', 'residence country'] }),
  linkedin: Object.freeze({ label: 'LinkedIn URL', aliases: ['linkedin', 'linkedin url', 'linkedin profile', 'linkedin profile url'] }),
  website: Object.freeze({ label: 'Website URL', aliases: ['website', 'website url', 'personal website', 'personal website url', 'personal site', 'personal site url', 'candidate website'] }),
  portfolio: Object.freeze({ label: 'Portfolio URL', aliases: ['portfolio', 'portfolio url', 'portfolio website', 'portfolio site'] }),
  github: Object.freeze({ label: 'GitHub URL', aliases: ['github', 'github url', 'github profile', 'github profile url'] }),
  authorized_to_work_us: Object.freeze({ label: 'Authorized to work in the U.S.', aliases: ['us work authorization'], category: 'eligibility' }),
  sponsorship_now: Object.freeze({ label: 'Requires U.S. employment sponsorship now', aliases: ['us sponsorship now'], category: 'eligibility' }),
  sponsorship_future: Object.freeze({ label: 'Requires U.S. employment sponsorship in the future', aliases: ['us future sponsorship'], category: 'eligibility' })
});

export const ELIGIBILITY_FACT_KEYS = Object.freeze([
  'authorized_to_work_us',
  'sponsorship_now',
  'sponsorship_future'
]);
const ELIGIBILITY_FACT_KEY_SET = new Set(ELIGIBILITY_FACT_KEYS);

const MAX_VERIFIED_FACT_LENGTH = 2000;

export const verifiedProfileValues = (records = []) => {
  const values = Object.fromEntries(Object.keys(FACT_DEFINITIONS).map(key => [key, '']));
  for (const record of records || []) {
    const key = String(record?.value?.key || '');
    if (!Object.hasOwn(FACT_DEFINITIONS, key)) continue;
    values[key] = String(record?.value?.value || '');
  }
  return values;
};

export const verifiedFactChanges = ({
  values = {},
  existingFacts = [],
  verifiedAt = new Date().toISOString()
} = {}) => {
  const existingByKey = new Map();
  for (const record of existingFacts || []) {
    const key = String(record?.value?.key || '');
    if (Object.hasOwn(FACT_DEFINITIONS, key)) existingByKey.set(key, record);
  }

  const upserts = [];
  const deletes = [];
  let savedCount = 0;
  for (const [key, definition] of Object.entries(FACT_DEFINITIONS)) {
    const existing = existingByKey.get(key);
    const existingValue = String(existing?.value?.value || '');
    if (!Object.hasOwn(values, key)) {
      if (existingValue.trim()) savedCount += 1;
      continue;
    }
    const rawNextValue = String(values[key] || '').trim();
    const nextValue = definition.category === 'eligibility' ? rawNextValue.toLocaleLowerCase('en-US') : rawNextValue;
    if (definition.category === 'eligibility' && nextValue && !['yes', 'no'].includes(nextValue)) {
      throw new Error(`${definition.label} must be Yes, No, or left unanswered.`);
    }
    if (nextValue.length > MAX_VERIFIED_FACT_LENGTH) {
      throw new Error(`${definition.label} must be ${MAX_VERIFIED_FACT_LENGTH} characters or fewer.`);
    }
    if (!nextValue) {
      if (existing) deletes.push(existing.id || `fact:${key}`);
      continue;
    }
    savedCount += 1;
    if (existing && nextValue === existingValue) continue;
    upserts.push({
      id: `fact:${key}`,
      kind: 'verified-fact',
      value: {
        key,
        label: definition.label,
        value: nextValue,
        sourceRole: SOURCE_ROLES.USER_VERIFIED,
        verifiedAt
      }
    });
  }
  return { upserts, deletes, savedCount, changedCount: upserts.length + deletes.length };
};

const SOURCE_ROLE_LABELS = Object.freeze({
  [SOURCE_ROLES.CANDIDATE_EVIDENCE]: 'Resume / candidate evidence',
  [SOURCE_ROLES.STYLE_EXAMPLE]: 'Cover letter / style evidence',
  [SOURCE_ROLES.JOB_REQUIREMENT]: 'Position / requirement',
  [SOURCE_ROLES.COMPANY_CONTEXT]: 'Company notes / context',
  [SOURCE_ROLES.USER_VERIFIED]: 'Verified material'
});

const ALLOWED_MODEL_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$/u;
const normalizeLabel = value => String(value || '')
  .normalize('NFKC')
  .toLocaleLowerCase('en-US')
  .replace(/\(required\)|\brequired\b|\*/gu, ' ')
  .replace(/[^\p{L}\p{N}]+/gu, ' ')
  .replace(/\s+/gu, ' ')
  .trim();

export const CUSTOM_PROFILE_CATEGORIES = Object.freeze({
  LINK: 'custom_link',
  THOUGHT: 'custom_thought'
});
const CUSTOM_PROFILE_CATEGORY_SET = new Set(Object.values(CUSTOM_PROFILE_CATEGORIES));
const MAX_CUSTOM_PROFILE_RECORDS = 24;
const MAX_CUSTOM_LABEL_LENGTH = 120;
const MAX_CUSTOM_LINK_LENGTH = 2000;
const MAX_CUSTOM_THOUGHT_LENGTH = 4000;
const CUSTOM_THOUGHT_BLOCKED_LABEL_PATTERN = /\b(?:citizenship|nationality|immigration|visa status|visa type|visa class|employment visa|sponsorship|work authorization|employment eligibility|i 9|e verify|passport|social security|ssn|race|ethnicity|gender|sex|disability|medical information|health condition|veteran status|military status|criminal history|date of birth|religion|attestation|signature)\b/iu;
const CUSTOM_THOUGHT_BLOCKED_VALUE_PATTERN = /\b(?:my|i\s+(?:am|have|identify as|need|require))\b.{0,60}\b(?:(?:u s |united states )?citizen(?:ship)?|nationality|immigration(?: status)?|sponsorship|work authorization|employment eligibility|race|ethnicity|gender identity|sexual orientation|disability|medical condition|health condition|veteran status|military status|criminal history|conviction|date of birth|religion)\b|\b(?:social security(?: number)?|ssn|passport(?: number)?|i 9|e verify|attestation|signature)\b/iu;
const CUSTOM_THOUGHT_BLOCKED_VISA_PATTERN = /\b(?:my\s+visa|visa\s+(?:status|type|class)|(?:i\s+)?(?:need|require)\s+(?:an?\s+)?visa|(?:i\s+)?have\s+(?:an?|my)\s+visa|(?:h\s?1b|o\s?1|l\s?1|f\s?1|j\s?1|tn)\s+visa)\b/iu;

export const normalizeCustomLinkUrl = (value) => {
  const raw = String(value || '').trim();
  if (!raw || raw.length > MAX_CUSTOM_LINK_LENGTH) {
    throw new Error(`Custom link URLs must be between 1 and ${MAX_CUSTOM_LINK_LENGTH} characters.`);
  }
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('Custom links must use a valid HTTP(S) URL.');
  }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    throw new Error('Custom links must use HTTP(S) and cannot contain embedded credentials.');
  }
  return url.toString();
};

const builtInFactLabelCollision = (label) => {
  const normalized = normalizeLabel(label);
  return Object.values(FACT_DEFINITIONS).some(definition => [definition.label, ...definition.aliases]
    .some(candidate => normalizeLabel(candidate) === normalized));
};

export const isFieldClarificationRecord = record => record?.value?.category === CUSTOM_PROFILE_CATEGORIES.THOUGHT
  && Boolean(String(record?.value?.applicationId || '').trim())
  && Boolean(String(record?.value?.fieldId || '').trim());

export const customProfileRecords = (facts = [], category = '') => (facts || [])
  .filter(record => (!category || record?.value?.category === category)
    && CUSTOM_PROFILE_CATEGORY_SET.has(record?.value?.category)
    && !isFieldClarificationRecord(record))
  .slice()
  .sort((left, right) => String(left.value?.label || '').localeCompare(String(right.value?.label || '')));

export const createCustomProfileRecord = ({
  category,
  label,
  value,
  existingFacts = [],
  recordToken = crypto.randomUUID(),
  verifiedAt = new Date().toISOString()
} = {}) => {
  const normalizedCategory = String(category || '').trim();
  if (!CUSTOM_PROFILE_CATEGORY_SET.has(normalizedCategory)) throw new Error('Unsupported custom profile item type.');
  const normalizedLabel = String(label || '').replace(/\s+/gu, ' ').trim();
  if (!normalizedLabel || normalizedLabel.length > MAX_CUSTOM_LABEL_LENGTH) {
    throw new Error(`Custom labels must be between 1 and ${MAX_CUSTOM_LABEL_LENGTH} characters.`);
  }
  if (normalizedCategory === CUSTOM_PROFILE_CATEGORIES.THOUGHT
    && CUSTOM_THOUGHT_BLOCKED_LABEL_PATTERN.test(normalizeLabel(normalizedLabel))) {
    throw new Error('Protected, identity, immigration-status, attestation, and signature topics must remain manual.');
  }
  if (normalizedCategory === CUSTOM_PROFILE_CATEGORIES.LINK && builtInFactLabelCollision(normalizedLabel)) {
    throw new Error('Use the matching built-in profile field instead of duplicating it as a custom link.');
  }
  const normalizedValue = normalizedCategory === CUSTOM_PROFILE_CATEGORIES.LINK
    ? normalizeCustomLinkUrl(value)
    : String(value || '').trim();
  const normalizedThoughtValue = normalizeLabel(normalizedValue);
  if (normalizedCategory === CUSTOM_PROFILE_CATEGORIES.THOUGHT
    && (CUSTOM_THOUGHT_BLOCKED_VALUE_PATTERN.test(normalizedThoughtValue)
      || CUSTOM_THOUGHT_BLOCKED_VISA_PATTERN.test(normalizedThoughtValue))) {
    throw new Error('Protected, identity, immigration-status, attestation, and signature content must remain manual.');
  }
  if (normalizedCategory === CUSTOM_PROFILE_CATEGORIES.THOUGHT
    && (!normalizedValue || normalizedValue.length > MAX_CUSTOM_THOUGHT_LENGTH)) {
    throw new Error(`Custom thoughts must be between 1 and ${MAX_CUSTOM_THOUGHT_LENGTH} characters.`);
  }
  const categoryRecords = customProfileRecords(existingFacts, normalizedCategory);
  if (categoryRecords.length >= MAX_CUSTOM_PROFILE_RECORDS) {
    throw new Error(`The vault supports at most ${MAX_CUSTOM_PROFILE_RECORDS} custom items of this type.`);
  }
  if (categoryRecords.some(record => normalizeLabel(record.value?.label) === normalizeLabel(normalizedLabel))) {
    throw new Error('A custom item with this label already exists. Delete it before adding a replacement.');
  }
  const safeToken = String(recordToken || '').trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$/u.test(safeToken)) throw new Error('Custom profile record identity is invalid.');
  const key = `${normalizedCategory}:${safeToken}`;
  return {
    id: `fact:${key}`,
    kind: 'verified-fact',
    value: {
      key,
      category: normalizedCategory,
      label: normalizedLabel,
      value: normalizedValue,
      sourceRole: SOURCE_ROLES.USER_VERIFIED,
      verifiedAt
    }
  };
};

const VERIFIED_FACT_CONTEXT_DENY_TOKENS = new Set([
  'account', 'alternate', 'assistant', 'beneficiary', 'billing', 'birth', 'business',
  'citizenship', 'client', 'college', 'company', 'coworker', 'customer', 'dependent',
  'emergency', 'employer', 'employment', 'former', 'guardian', 'hiring', 'landlord',
  'maiden', 'manager', 'nationality', 'office', 'organisation', 'organization', 'other',
  'parent', 'partner', 'party', 'previous', 'recruiter', 'reference', 'referee', 'school',
  'secondary', 'shipping', 'spouse', 'supervisor', 'tenant', 'third', 'university',
  'vendor', 'work'
]);

const VERIFIED_FACT_LABEL_FILLERS = new Set([
  'a', 'an', 'applicant', 'are', 'below', 'candidate', 'choose', 'confirm', 'do', 'does',
  'enter', 'for', 'here', 'in', 'is', 'of', 'on', 'optional', 'please', 'provide', 's', 'select',
  'the', 'type', 'was', 'what', 'where', 'which', 'will', 'you', 'your'
]);

const VERIFIED_FACT_SAFE_QUALIFIERS = Object.freeze({
  first_name: Object.freeze(['current', 'given', 'legal', 'preferred']),
  last_name: Object.freeze(['current', 'family', 'legal']),
  full_name: Object.freeze(['complete', 'current', 'legal']),
  email: Object.freeze(['best', 'contact', 'current', 'personal', 'preferred', 'primary']),
  phone: Object.freeze(['best', 'cell', 'contact', 'current', 'daytime', 'mobile', 'personal', 'preferred', 'primary']),
  street_address: Object.freeze(['current', 'home', 'mailing', 'personal', 'physical', 'primary', 'residence', 'resident', 'residential']),
  city: Object.freeze(['current', 'currently', 'home', 'mailing', 'reside', 'residence', 'resident', 'residential', 'residing']),
  state_province: Object.freeze(['current', 'currently', 'home', 'mailing', 'reside', 'residence', 'resident', 'residential', 'residing', 'states', 'united', 'us']),
  postal_code: Object.freeze(['current', 'currently', 'home', 'mailing', 'reside', 'residence', 'resident', 'residential', 'residing']),
  country: Object.freeze(['current', 'currently', 'home', 'mailing', 'reside', 'residence', 'resident', 'residential', 'residing']),
  linkedin: Object.freeze(['current', 'personal', 'profile', 'url']),
  website: Object.freeze(['candidate', 'current', 'personal', 'profile', 'site', 'url', 'website']),
  portfolio: Object.freeze(['current', 'personal', 'portfolio', 'profile', 'site', 'url', 'website']),
  github: Object.freeze(['current', 'personal', 'profile', 'url']),
  custom_link: Object.freeze(['current', 'link', 'profile', 'url', 'website'])
});

const containsContiguousTokens = (tokens, phraseTokens, offset) => phraseTokens
  .every((token, index) => tokens[offset + index] === token);

const hasSafeQualifiedAlias = (tokens, alias, key) => {
  const aliasTokens = normalizeLabel(alias).split(' ').filter(Boolean);
  if (!aliasTokens.length || aliasTokens.length >= tokens.length) return false;
  const safeQualifiers = VERIFIED_FACT_SAFE_QUALIFIERS[key] || [];
  for (let offset = 0; offset <= tokens.length - aliasTokens.length; offset += 1) {
    if (!containsContiguousTokens(tokens, aliasTokens, offset)) continue;
    const contextTokens = tokens.slice(0, offset).concat(tokens.slice(offset + aliasTokens.length));
    if (contextTokens.every(token => VERIFIED_FACT_LABEL_FILLERS.has(token) || safeQualifiers.includes(token))) {
      return true;
    }
  }
  return false;
};

export const matchVerifiedFactFieldLabel = (label, facts = []) => {
  const normalized = normalizeLabel(label);
  if (!normalized) return null;
  const exactMatches = Object.entries(FACT_DEFINITIONS)
    .filter(([key, definition]) => !ELIGIBILITY_FACT_KEY_SET.has(key) && definition.aliases.some(alias => normalizeLabel(alias) === normalized))
    .map(([key]) => key);
  if (exactMatches.length === 1) return { key: exactMatches[0], matchKind: 'exact_alias' };
  if (exactMatches.length > 1) return null;

  const tokens = normalized.split(' ');
  if (tokens.some(token => VERIFIED_FACT_CONTEXT_DENY_TOKENS.has(token))) return null;
  const qualifiedMatches = Object.entries(FACT_DEFINITIONS)
    .filter(([key, definition]) => !ELIGIBILITY_FACT_KEY_SET.has(key) && definition.aliases.some(alias => hasSafeQualifiedAlias(tokens, alias, key)))
    .map(([key]) => key);
  const uniqueMatches = [...new Set(qualifiedMatches)];
  if (uniqueMatches.length === 1) return { key: uniqueMatches[0], matchKind: 'qualified_alias' };
  if (uniqueMatches.length > 1) return null;

  const customMatches = customProfileRecords(facts, CUSTOM_PROFILE_CATEGORIES.LINK)
    .filter(record => normalizeLabel(record.value?.label) === normalized
      || hasSafeQualifiedAlias(tokens, record.value?.label, CUSTOM_PROFILE_CATEGORIES.LINK));
  return customMatches.length === 1
    ? { key: customMatches[0].value.key, matchKind: 'custom_link_alias' }
    : null;
};

export const factKeyForFieldLabel = (label, facts = []) => matchVerifiedFactFieldLabel(label, facts)?.key || null;

const ELIGIBILITY_BLOCKED_PATTERN = /\b(?:citizen|citizenship|nationality|immigration status|visa status|visa type|i 9|e verify|itar|ear|export control|security clearance|passport|social security|ssn|alien registration|a number)\b/iu;
const ELIGIBILITY_BLOCKED_DETAIL_PATTERN = /\b(?:visa class|visa category|visa kind|lawful permanent resident|permanent resident|green card|employment authorization document|work permit|h 1b|o 1|l 1|f 1|j 1|tn status)\b/iu;
const US_ELIGIBILITY_CONTEXT_PATTERN = /\b(?:us|u s(?: a)?|usa|united states|america(?:n)?)\b/iu;
const WORK_AUTHORIZATION_PATTERN = /(?:\b(?:authori[sz](?:ed|ation)|eligible|eligibility)\b.{0,80}\bwork\b|\blegal right to work\b|\bwork\b.{0,80}\b(?:authori[sz](?:ed|ation)|eligible|eligibility)\b)/iu;
const SPONSORSHIP_PATTERN = /\b(?:sponsor|sponsored|sponsoring|sponsorship|employment visa)\b/iu;
export const isWorkEligibilityQuestion = (fieldOrLabel) => {
  const normalized = normalizeLabel(typeof fieldOrLabel === 'object' ? fieldOrLabel?.label : fieldOrLabel);
  return Boolean(normalized && (WORK_AUTHORIZATION_PATTERN.test(normalized) || SPONSORSHIP_PATTERN.test(normalized)));
};


export const eligibilityQuestionKind = (fieldOrLabel) => {
  const normalized = normalizeLabel(typeof fieldOrLabel === 'object' ? fieldOrLabel?.label : fieldOrLabel);
  if (!normalized || ELIGIBILITY_BLOCKED_PATTERN.test(normalized)) return null;
  if (ELIGIBILITY_BLOCKED_DETAIL_PATTERN.test(normalized)) return null;
  if (!US_ELIGIBILITY_CONTEXT_PATTERN.test(normalized) && isWorkEligibilityQuestion(normalized)) return 'manual_scope';
  if (!US_ELIGIBILITY_CONTEXT_PATTERN.test(normalized)) return null;
  const asksAuthorization = WORK_AUTHORIZATION_PATTERN.test(normalized);
  const asksSponsorship = SPONSORSHIP_PATTERN.test(normalized);
  if (!asksAuthorization && !asksSponsorship) return null;
  const asksFuture = /\b(?:future|later|ever|subsequently)\b|\bat any time\b/iu.test(normalized);
  const asksNow = /\b(?:now|current|currently|present)\b|\bat this time\b/iu.test(normalized);
  const asksWithoutSponsorship = /\bwithout\b.{0,50}\bsponsor/iu.test(normalized)
    || /\b(?:do not|does not|not)\b.{0,40}\b(?:need|require)\b.{0,40}\bsponsor/iu.test(normalized);
  if (asksAuthorization && asksSponsorship && asksWithoutSponsorship && asksFuture) {
    return 'authorization_without_sponsorship_now_or_future';
  }
  if (asksAuthorization && asksSponsorship && asksWithoutSponsorship) return 'authorization_without_sponsorship';
  if (asksSponsorship && asksNow && asksFuture) return 'sponsorship_now_or_future';
  if (asksSponsorship && asksFuture) return 'sponsorship_future';
  if (asksSponsorship) return 'sponsorship_now';
  return asksAuthorization ? 'authorization' : null;
};

const eligibilityValue = value => {
  const normalized = String(value || '').trim().toLocaleLowerCase('en-US');
  return ['yes', 'no'].includes(normalized) ? normalized : '';
};

export const deriveEligibilityAnswer = ({ kind, values = {} } = {}) => {
  const authorized = eligibilityValue(values.authorized_to_work_us);
  const sponsorshipNow = eligibilityValue(values.sponsorship_now);
  const sponsorshipFuture = eligibilityValue(values.sponsorship_future);
  if (kind === 'authorization' && authorized) {
    return { value: authorized, factKeys: ['authorized_to_work_us'] };
  }
  if (kind === 'sponsorship_now' && sponsorshipNow) {
    return { value: sponsorshipNow, factKeys: ['sponsorship_now'] };
  }
  if (kind === 'sponsorship_future' && sponsorshipFuture) {
    return { value: sponsorshipFuture, factKeys: ['sponsorship_future'] };
  }
  if (kind === 'sponsorship_now_or_future') {
    if (sponsorshipNow === 'yes') return { value: 'yes', factKeys: ['sponsorship_now'] };
    if (sponsorshipFuture === 'yes') return { value: 'yes', factKeys: ['sponsorship_future'] };
    if (sponsorshipNow === 'no' && sponsorshipFuture === 'no') {
      return { value: 'no', factKeys: ['sponsorship_now', 'sponsorship_future'] };
    }
  }
  if (kind === 'authorization_without_sponsorship') {
    if (authorized === 'yes' && sponsorshipNow === 'no') {
      return { value: 'yes', factKeys: ['authorized_to_work_us', 'sponsorship_now'] };
    }
    if (authorized === 'no') return { value: 'no', factKeys: ['authorized_to_work_us'] };
    if (sponsorshipNow === 'yes') return { value: 'no', factKeys: ['sponsorship_now'] };
  }
  if (kind === 'authorization_without_sponsorship_now_or_future') {
    if (authorized === 'yes' && sponsorshipNow === 'no' && sponsorshipFuture === 'no') {
      return { value: 'yes', factKeys: ['authorized_to_work_us', 'sponsorship_now', 'sponsorship_future'] };
    }
    if (authorized === 'no') return { value: 'no', factKeys: ['authorized_to_work_us'] };
    if (sponsorshipNow === 'yes') return { value: 'no', factKeys: ['sponsorship_now'] };
    if (sponsorshipFuture === 'yes') return { value: 'no', factKeys: ['sponsorship_future'] };
  }
  return null;
};

export const validateModelSettings = (value) => {
  const settings = {
    generationModel: String(value?.generationModel || '').trim(),
    fallbackGenerationModel: String(value?.fallbackGenerationModel || '').trim(),
    embeddingModel: String(value?.embeddingModel || '').trim()
  };
  for (const [key, model] of Object.entries(settings)) {
    if (!ALLOWED_MODEL_PATTERN.test(model)) throw new Error(`${key} must be a valid local Ollama model name.`);
    assertLocalModelName(model);
  }
  return settings;
};

export const isCurrentPrivacyConsent = value => Boolean(
  value
  && value.version === PRIVACY_NOTICE_VERSION
  && typeof value.acceptedAt === 'string'
  && Number.isFinite(Date.parse(value.acceptedAt))
);

export const ollamaOriginCommand = extensionId => `OLLAMA_ORIGINS=chrome-extension://${String(extensionId || '').trim()}`;

export const proposalAnswerValue = (proposal) => {
  if (proposal?.value_type === 'selected_values') return [...(proposal.selected_values || [])];
  if (proposal?.value_type === 'checked') return Boolean(proposal.checked);
  return String(proposal?.value || '');
};

export const scanSnapshotKey = scan => [scan?.pageId, scan?.urlHash, scan?.domRevision].join(':');

const exactString = value => typeof value === 'string' ? value : '';

const DISCOVERY_MODES = new Set(['standard', 'free_format', 'limited']);
const DISCOVERY_CONTEXT_STATUSES = new Set(['unsupported', 'manual']);
const DISCOVERY_CONTEXT_PRESENTATIONS = Object.freeze({
  cross_origin_iframe: Object.freeze({
    title: 'Embedded application fields need their own tab',
    copy: 'Some application controls are inside a cross-origin frame. Open the employer application form in its own tab, then analyze that page.'
  }),
  closed_shadow_root: Object.freeze({
    title: 'Protected custom controls need manual entry',
    copy: 'Some controls are hidden inside a closed page component. Copilot cannot inspect or fill them; complete those controls manually.'
  }),
  custom_aria_widget: Object.freeze({
    title: 'Custom controls are copy-only',
    copy: 'Copilot can prepare answers for recognized questions, but unsupported custom widgets must be reviewed and copied manually.'
  }),
  account_gate: Object.freeze({
    title: 'Complete the account step first',
    copy: 'A sign-in or account-creation gate was detected. Complete it manually, then rescan the next application step.'
  })
});
const DISCOVERY_CONTEXT_KINDS = new Set(Object.keys(DISCOVERY_CONTEXT_PRESENTATIONS));
const DISCOVERY_EXCLUSION_KEYS = Object.freeze(['F0_EXCLUDED', 'F3_SENSITIVE', 'F4_CONSENT']);
const boundedCount = (value, fallback = 0) => Number.isSafeInteger(value) && value >= 0
  ? Math.min(value, 10_000)
  : fallback;

export const isCopyOnlyField = field => field?.fillMode === 'copy_only';

export const normalizeScanDiscovery = (scan = {}) => {
  const supplied = scan?.discovery && typeof scan.discovery === 'object' && !Array.isArray(scan.discovery)
    ? scan.discovery
    : null;
  const legacyExclusions = scan?.exclusionCounts && typeof scan.exclusionCounts === 'object'
    ? scan.exclusionCounts
    : {};
  const suppliedExclusions = supplied?.exclusionCounts && typeof supplied.exclusionCounts === 'object'
    ? supplied.exclusionCounts
    : legacyExclusions;
  const exclusionCounts = Object.fromEntries(DISCOVERY_EXCLUSION_KEYS.map(key => [
    key,
    boundedCount(suppliedExclusions[key])
  ]));
  const contexts = [];
  for (const rawContext of Array.isArray(supplied?.contexts) ? supplied.contexts : []) {
    const kind = exactString(rawContext?.kind);
    const status = exactString(rawContext?.status);
    const count = boundedCount(rawContext?.count);
    if (!DISCOVERY_CONTEXT_KINDS.has(kind) || !DISCOVERY_CONTEXT_STATUSES.has(status) || count < 1) continue;
    const existing = contexts.find(context => context.kind === kind && context.status === status);
    if (existing) existing.count = Math.min(10_000, existing.count + count);
    else contexts.push({ kind, count, status });
  }
  const recognizedFallback = Array.isArray(scan?.fields) ? scan.fields.length : 0;
  return {
    available: Boolean(supplied),
    mode: DISCOVERY_MODES.has(supplied?.mode) ? supplied.mode : 'standard',
    recognizedCount: boundedCount(supplied?.recognizedCount, recognizedFallback),
    unsupportedCount: boundedCount(supplied?.unsupportedCount, contexts.reduce((sum, context) => sum + context.count, 0)),
    exclusionCounts,
    contexts,
    truncated: Boolean(supplied?.truncated ?? scan?.truncated)
  };
};

export const hasAccountGateContext = scan => normalizeScanDiscovery(scan).contexts
  .some(context => context.kind === 'account_gate');

const discoveryExcludedTotal = discovery => Object.values(discovery.exclusionCounts)
  .reduce((sum, count) => sum + count, 0);

export const scanSupportPresentation = (scan) => {
  const discovery = normalizeScanDiscovery(scan);
  const fieldCount = Array.isArray(scan?.fields) ? scan.fields.length : 0;
  const copyOnlyCount = (scan?.fields || []).filter(isCopyOnlyField).length;
  const excludedCount = discoveryExcludedTotal(discovery);
  const primaryContext = discovery.contexts[0];
  const contextPresentation = primaryContext
    ? DISCOVERY_CONTEXT_PRESENTATIONS[primaryContext.kind]
    : null;
  const statusParts = [
    `${fieldCount} safe ${fieldCount === 1 ? 'field' : 'fields'} found`,
    copyOnlyCount ? `${copyOnlyCount} copy-only` : '',
    discovery.unsupportedCount ? `${discovery.unsupportedCount} unsupported` : '',
    excludedCount ? `${excludedCount} protected or non-application controls excluded` : '',
    discovery.truncated ? 'scan limit reached' : ''
  ].filter(Boolean);
  const status = `${statusParts.join('. ')}. Submission is never automated.`;
  if (contextPresentation) {
    const remaining = discovery.contexts.length - 1;
    return {
      showNotice: true,
      title: contextPresentation.title,
      copy: `${contextPresentation.copy}${remaining > 0 ? ` ${remaining} additional unsupported context ${remaining === 1 ? 'was' : 'were'} also detected.` : ''}`,
      status,
      emptyError: contextPresentation.copy
    };
  }
  if (!fieldCount && excludedCount) {
    const copy = `${excludedCount} protected, consent, navigation, or other non-application ${excludedCount === 1 ? 'control was' : 'controls were'} intentionally excluded. Complete any required manual step, then rescan.`;
    return {
      showNotice: true,
      title: 'No fillable fields on this step',
      copy,
      status,
      emptyError: copy
    };
  }
  if (discovery.truncated) {
    const copy = 'The page exceeded the bounded scan limit. Review the recognized fields and complete any remaining controls manually.';
    return {
      showNotice: true,
      title: 'Only part of this form was scanned',
      copy,
      status,
      emptyError: copy
    };
  }
  return {
    showNotice: false,
    title: '',
    copy: '',
    status,
    emptyError: 'No supported application fields were found on this page.'
  };
};

export const canonicalApplicationStructure = scan => canonicalJson({
  adapter: exactString(scan?.adapter),
  captchaPresent: Boolean(scan?.captchaPresent),
  domRevision: Number.isSafeInteger(scan?.domRevision) ? scan.domRevision : 0,
  fields: Array.isArray(scan?.fields) ? scan.fields.map(field => ({
    fieldId: exactString(field?.fieldId),
    fillMode: isCopyOnlyField(field) ? 'copy_only' : '',
    fingerprint: exactString(field?.fingerprint),
    label: exactString(field?.label),
    maxLength: Number.isSafeInteger(field?.maxLength) ? field.maxLength : null,
    nearbyText: exactString(field?.nearbyText),
    options: Array.isArray(field?.options) ? field.options.map(exactString) : [],
    required: Boolean(field?.required),
    riskClass: exactString(field?.riskClass),
    type: exactString(field?.type)
  })) : [],
  job: {
    description: exactString(scan?.job?.description),
    company: exactString(scan?.job?.company),
    jobUrl: exactString(scan?.job?.jobUrl),
    location: exactString(scan?.job?.location),
    source: exactString(scan?.job?.source),
    title: exactString(scan?.job?.title)
  },
  pageId: exactString(scan?.pageId),
  urlHash: exactString(scan?.urlHash)
});

export const applicationStructureSignature = scan => sha256Base64Url(canonicalApplicationStructure(scan));

const canonicalFillTarget = field => canonicalJson({
  fieldId: exactString(field?.fieldId),
  fillMode: isCopyOnlyField(field) ? 'copy_only' : '',
  fingerprint: exactString(field?.fingerprint),
  label: exactString(field?.label),
  maxLength: Number.isSafeInteger(field?.maxLength) ? field.maxLength : null,
  nearbyText: exactString(field?.nearbyText),
  options: Array.isArray(field?.options) ? field.options.map(exactString) : [],
  required: Boolean(field?.required),
  riskClass: exactString(field?.riskClass),
  type: exactString(field?.type)
});

const canonicalFillJobIdentity = scan => canonicalJson({
  company: exactString(scan?.job?.company),
  jobUrl: exactString(scan?.job?.jobUrl),
  title: exactString(scan?.job?.title)
});

export const canFillFieldAcrossRevision = ({
  analyzedScan,
  freshScan,
  field
} = {}) => {
  if (!analyzedScan || !freshScan || !field || freshScan.captchaPresent
    || hasAccountGateContext(analyzedScan) || hasAccountGateContext(freshScan)) return false;
  if (exactString(analyzedScan.pageId) !== exactString(freshScan.pageId)
    || exactString(analyzedScan.urlHash) !== exactString(freshScan.urlHash)
    || exactString(analyzedScan.adapter) !== exactString(freshScan.adapter)
    || canonicalFillJobIdentity(analyzedScan) !== canonicalFillJobIdentity(freshScan)) {
    return false;
  }
  const analyzedField = analyzedScan.fields?.find(candidate => candidate.fieldId === field.fieldId);
  const freshField = freshScan.fields?.find(candidate => candidate.fieldId === field.fieldId);
  if (!analyzedField || !freshField) return false;
  const expected = canonicalFillTarget(field);
  return canonicalFillTarget(analyzedField) === expected
    && canonicalFillTarget(freshField) === expected;
};

export const proposalsMatchFieldIds = (proposals, fields) => {
  if (!Array.isArray(proposals) || !Array.isArray(fields) || proposals.length !== fields.length) return false;
  const expected = new Set(fields.map(field => field?.fieldId));
  if (expected.size !== fields.length || expected.has(undefined) || expected.has('')) return false;
  const actual = proposals.map(proposal => proposal?.field_id);
  return actual.every(fieldId => typeof fieldId === 'string' && expected.has(fieldId))
    && new Set(actual).size === actual.length;
};

export const generationCacheSeed = ({
  structureSignature,
  sourceSignature,
  generationModel,
  fallbackGenerationModel,
  embeddingModel
}) => [
  structureSignature,
  sourceSignature,
  generationModel,
  fallbackGenerationModel,
  embeddingModel
].join('|');

export const sourceSelectionScope = scan => scan?.pageId && scan?.urlHash
  ? canonicalJson({
    adapter: exactString(scan.adapter),
    job: {
      company: exactString(scan.job?.company),
      jobUrl: exactString(scan.job?.jobUrl),
      location: exactString(scan.job?.location),
      source: exactString(scan.job?.source),
      title: exactString(scan.job?.title)
    },
    pageId: exactString(scan.pageId),
    urlHash: exactString(scan.urlHash)
  })
  : UNSCOPED_SOURCE_SELECTION;

export const isReusableSourceRecord = record => isReusableCatalogueSourceRecord(record);

export const carrySourceSelection = options => carryCatalogueSourceSelection(options);

export const retrievalCacheSeed = ({ applicationId, sourceSignature, embeddingModel }) => [
  applicationId,
  sourceSignature,
  embeddingModel
].join('|');

export const preparedChunkMode = ({ cacheHit, cachedValue }) => (
  cacheHit && cachedValue?.mode === 'hybrid' ? 'hybrid' : 'lexical'
);

export const validateImportFile = (file) => {
  const size = Number(file?.size);
  if (!Number.isSafeInteger(size) || size < 1) throw new Error('The selected source file is empty or has an invalid size.');
  if (size > MAX_SOURCE_FILE_BYTES) {
    throw new Error('Source files must be 10 MB or smaller. Compress or split this file before importing it.');
  }
  return file;
};

export const readDocumentBytes = async file => new Uint8Array(await validateImportFile(file).arrayBuffer());

export const runtimeAnswerValue = (proposal, field) => {
  const value = proposalAnswerValue(proposal);
  if (proposal?.value_type === 'selected_values'
    && ['radio', 'select-one'].includes(field?.type)) return value[0] || '';
  return value;
};

export const modelEligibleFields = fields => (fields || []).filter(field => field.riskClass === 'F2_REVIEW'
  && !eligibilityQuestionKind(field));
export const explicitFieldFillConfirmation = field => field?.riskClass === 'F2_REVIEW';

export const planBulkFill = ({
  fields = [],
  proposals = [],
  reviewedConsequential = false,
  verifiedOnly = false,
  alreadyFilledIds = []
} = {}) => {
  const proposalsByField = new Map(proposals.map(proposal => [proposal?.field_id, proposal]));
  const filled = alreadyFilledIds instanceof Set ? alreadyFilledIds : new Set(alreadyFilledIds);
  const items = [];
  for (const field of fields) {
    const proposal = proposalsByField.get(field?.fieldId);
    if (!proposal || proposal.action !== 'fill' || field.manual === true || isCopyOnlyField(field)) continue;
    if (!['F1_VERIFIED', 'F2_REVIEW'].includes(field.riskClass)) continue;
    if (proposal.risk_class !== field.riskClass || proposal.confidence === 'needs_input') continue;
    if (!Array.isArray(proposal.citation_ids) || proposal.citation_ids.length < 1) continue;
    if (filled.has(field.fieldId)) continue;
    if (field.riskClass === 'F1_VERIFIED' && proposal.confidence !== 'high') continue;
    if (verifiedOnly && field.riskClass !== 'F1_VERIFIED') continue;
    if (field.riskClass === 'F2_REVIEW' && !reviewedConsequential) continue;
    items.push({ field, proposal, confirmed: field.riskClass === 'F2_REVIEW' });
  }
  return {
    items,
    verifiedCount: items.filter(item => item.field.riskClass === 'F1_VERIFIED').length,
    consequentialCount: items.filter(item => item.field.riskClass === 'F2_REVIEW').length
  };
};

const BULK_FILL_STOP_REASONS = new Set([
  'captcha_present',
  'field_fingerprint_changed',
  'field_missing_after_rescan',
  'review_confirmation_required'
]);

export const executeBulkFillItems = async ({
  items = [],
  requestFill,
  onProgress = () => {},
  onFilled = () => {},
  onOccupied = () => {}
} = {}) => {
  if (!Array.isArray(items) || typeof requestFill !== 'function') {
    throw new TypeError('Bulk fill requires reviewed items and a field request handler.');
  }
  const result = { filled: 0, occupied: 0, copyOnly: 0, stopped: '' };
  for (let index = 0; index < items.length; index += 1) {
    const item = items[index];
    onProgress(item, index, items.length);
    let payload;
    try {
      payload = await requestFill(item, index);
    } catch (error) {
      result.stopped = readableError(error);
      break;
    }
    if (payload?.verified) {
      onFilled(item, payload);
      result.filled += 1;
      continue;
    }
    if (payload?.reason === 'field_already_has_value') {
      onOccupied(item, payload);
      result.occupied += 1;
      continue;
    }
    if (BULK_FILL_STOP_REASONS.has(payload?.reason)) {
      result.stopped = payload.reason.replaceAll('_', ' ');
      break;
    }
    result.copyOnly += 1;
  }
  return result;
};

export const bulkFillResultTone = ({ filled = 0, copyOnly = 0, stopped = '' } = {}) => (
  stopped || (filled < 1 && copyOnly > 0) ? 'error' : 'success'
);

export const canCommitGeneration = ({
  expectedScanKey,
  currentScan,
  expectedStructureSignature,
  currentStructureSignature,
  stale,
  aborted
}) => !stale
  && !aborted
  && typeof expectedStructureSignature === 'string'
  && Boolean(expectedStructureSignature)
  && expectedStructureSignature === currentStructureSignature
  && expectedScanKey === scanSnapshotKey(currentScan);

export const isModelInstalled = (requestedModel, installedModels) => {
  const requested = String(requestedModel || '').trim();
  const installed = new Set((installedModels || []).map(value => String(value || '').trim()));
  if (!requested) return false;
  if (installed.has(requested)) return true;
  return !requested.includes(':') && installed.has(`${requested}:latest`);
};

export const isClearlyUnsupportedOllamaVersion = (value, minimum = [0, 32, 0]) => {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:$|[-+])/u.exec(String(value || '').trim());
  if (!match) return false;
  const version = match.slice(1).map(Number);
  for (let index = 0; index < minimum.length; index += 1) {
    if (version[index] > minimum[index]) return false;
    if (version[index] < minimum[index]) return true;
  }
  return false;
};

export const parseTrackerTags = (value) => {
  const tags = String(value || '').split(',').map(tag => tag.trim()).filter(Boolean);
  const unique = [];
  const seen = new Set();
  for (const tag of tags) {
    if (tag.length > 36) throw new Error('Tracker tags must be 36 characters or fewer.');
    const normalized = tag.toLocaleLowerCase('en-US');
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    unique.push(tag);
  }
  if (unique.length > 12) throw new Error('Tracker handoff supports at most 12 tags.');
  return unique;
};

const defaultSettings = () => ({
  generationModel: DEFAULT_LOCAL_MODEL_CONFIG.generationModel,
  fallbackGenerationModel: DEFAULT_LOCAL_MODEL_CONFIG.fallbackGenerationModel,
  embeddingModel: DEFAULT_LOCAL_MODEL_CONFIG.embeddingModel
});

const initialSourceSelection = new Set();
const state = {
  vault: null,
  ollama: null,
  research: null,
  initialized: false,
  unlocked: false,
  documents: [],
  facts: [],
  applicationPackets: [],
  activeApplicationPacketId: '',
  profileDirty: false,
  selectedSourceIds: initialSourceSelection,
  sourceCatalogueInitialized: false,
  sourceSelectionScope: UNSCOPED_SOURCE_SELECTION,
  sourceSelectionsByScope: new Map([[UNSCOPED_SOURCE_SELECTION, initialSourceSelection]]),
  settings: defaultSettings(),
  privacyConsent: null,
  privacyReviewOpen: false,
  operationalStarted: false,
  activeTabId: null,
  scan: null,
  stale: false,
  evidence: null,
  proposals: [],
  filledFieldIds: new Set(),
  bulkBusy: false,
  batchStatus: '',
  answerFilter: 'all',
  selectedFieldId: null,
  retrievalMode: '',
  sourceSignature: '',
  analysisNotice: '',
  analysisController: null,
  generationController: null,
  ollamaController: null,
  busy: false,
  toastTimer: null
};

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

const createElement = (tagName, { className = '', text = '', attributes = {}, dataset = {} } = {}) => {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text !== '') element.textContent = text;
  Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
  Object.entries(dataset).forEach(([name, value]) => { element.dataset[name] = String(value); });
  return element;
};

const showToast = (message, tone = 'info') => {
  const toast = $('[data-toast]');
  if (!toast) return;
  clearTimeout(state.toastTimer);
  toast.textContent = String(message || '');
  toast.dataset.tone = tone;
  toast.hidden = false;
  state.toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
};

const readableError = error => String(error?.message || error || 'Something went wrong.');

export const clarificationRegenerationFailureMessage = ({ clarificationSaved = false, error } = {}) => {
  const detail = readableError(error);
  if (!clarificationSaved) return detail;
  return `Clarification was saved locally, but the regenerated answer was not applied. ${detail} Retry regeneration; you do not need to enter the clarification again.`;
};

export const ollamaHealthErrorPresentation = (error, extensionId) => {
  if (error?.status === 403) {
    return {
      title: 'Ollama origin not allowed',
      copy: `Ollama returned HTTP 403. Set ${ollamaOriginCommand(extensionId || '<extension-id>')}, fully quit Ollama, and restart it.`
    };
  }
  const blocked = error?.code === 'REMOTE_MODEL_NOT_ALLOWED';
  return {
    title: blocked ? 'Cloud or remote model blocked' : 'Ollama unavailable',
    copy: readableError(error)
  };
};

const setButtonBusy = (button, busy, label) => {
  if (!button) return;
  if (busy) {
    button.dataset.previousLabel = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.previousLabel || button.textContent;
    delete button.dataset.previousLabel;
    button.disabled = false;
  }
};

const activateTab = (name, { focus = false } = {}) => {
  $$('[data-tab]').forEach((button) => {
    const selected = button.dataset.tab === name;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  $$('[data-panel]').forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
};

const installTabs = () => {
  const tabs = $$('[role="tab"]');
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateTab(tab.dataset.tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      activateTab(tabs[next].dataset.tab, { focus: true });
    });
  });
};

const requirePrivacyConsent = () => {
  if (!isCurrentPrivacyConsent(state.privacyConsent)) {
    throw new Error('Review and accept the privacy notice before using Job Application Copilot.');
  }
};

const renderPrivacyExperience = () => {
  const consented = isCurrentPrivacyConsent(state.privacyConsent);
  const gateVisible = !consented || state.privacyReviewOpen;
  const onboarding = $('[data-privacy-onboarding]');
  const shell = $('[data-operational-shell]');
  const form = $('[data-privacy-consent-form]');
  const skipLink = $('[data-skip-link]');
  onboarding.hidden = !gateVisible;
  shell.hidden = gateVisible;
  shell.inert = gateVisible;
  form.elements.acknowledged.checked = consented;
  $('[data-privacy-accept]').textContent = consented ? 'Return to settings' : 'Accept and continue';
  const privacyVersion = $('[data-privacy-version]');
  if (privacyVersion) privacyVersion.textContent = `Version ${PRIVACY_NOTICE_VERSION}`;
  skipLink.href = gateVisible ? '#privacy-onboarding' : '#main';
  skipLink.textContent = gateVisible ? 'Skip to privacy notice' : 'Skip to workspace';
};

const installOperationalConsentGuard = () => {
  const guard = (event) => {
    if (isCurrentPrivacyConsent(state.privacyConsent)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    showToast('Review and accept the privacy notice before using Job Application Copilot.', 'error');
  };
  const shell = $('[data-operational-shell]');
  shell.addEventListener('click', guard, true);
  shell.addEventListener('submit', guard, true);
  shell.addEventListener('change', guard, true);
};

const installPrivacyActions = () => {
  $('[data-privacy-consent-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;
    const wasReviewing = state.privacyReviewOpen;
    try {
      if (!isCurrentPrivacyConsent(state.privacyConsent)) {
        const consent = { version: PRIVACY_NOTICE_VERSION, acceptedAt: new Date().toISOString() };
        await chrome.storage.local.set({ [PRIVACY_CONSENT_KEY]: consent });
        state.privacyConsent = consent;
      }
      state.privacyReviewOpen = false;
      renderPrivacyExperience();
      await startOperationalControllers();
      if (wasReviewing) activateTab('settings');
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
};

const setVaultIndicator = (unlocked) => {
  const indicator = $('[data-vault-indicator]');
  indicator.dataset.vaultIndicator = unlocked ? 'unlocked' : 'locked';
  indicator.textContent = unlocked ? 'Vault unlocked' : 'Vault locked';
};

const setOllamaIndicator = (status, text) => {
  const indicator = $('[data-ollama-indicator]');
  indicator.dataset.ollamaIndicator = status;
  indicator.textContent = text;
};

const resetSourceSelectionScopes = () => {
  const selection = new Set();
  state.selectedSourceIds = selection;
  state.sourceCatalogueInitialized = false;
  state.sourceSelectionScope = UNSCOPED_SOURCE_SELECTION;
  state.sourceSelectionsByScope = new Map([[UNSCOPED_SOURCE_SELECTION, selection]]);
};

const activateSourceSelectionScope = (scan) => {
  const nextScope = sourceSelectionScope(scan);
  if (nextScope === state.sourceSelectionScope) return;
  state.sourceSelectionsByScope.set(state.sourceSelectionScope, state.selectedSourceIds);
  let nextSelection = state.sourceSelectionsByScope.get(nextScope);
  if (!nextSelection) {
    nextSelection = carrySourceSelection({
      selectedIds: state.selectedSourceIds,
      records: state.documents,
      applicationId: scan?.pageId,
      fromUnscoped: state.sourceSelectionScope === UNSCOPED_SOURCE_SELECTION
    });
    state.sourceSelectionsByScope.set(nextScope, nextSelection);
  }
  state.sourceSelectionScope = nextScope;
  state.selectedSourceIds = nextSelection;
};

const clearDecryptedState = () => {
  state.documents = [];
  state.facts = [];
  state.applicationPackets = [];
  state.activeApplicationPacketId = '';
  state.profileDirty = false;
  resetSourceSelectionScopes();
  state.evidence = null;
  state.proposals = [];
  state.filledFieldIds.clear();
  state.batchStatus = '';
  state.answerFilter = 'all';
  state.bulkBusy = false;
  state.sourceSignature = '';
  state.analysisNotice = '';
  renderSources();
  renderApplication();
};

const today = () => new Date().toISOString().slice(0, 10);

export const repairDocumentMetadataFromRetainedBytes = async (record) => {
  const documentRecord = record?.value?.document;
  const size = Number(documentRecord?.size);
  const digest = String(documentRecord?.sha256 || '');
  const metadataValid = Number.isSafeInteger(size) && size > 0 && digest.length >= 43;
  if (!documentRecord || metadataValid || typeof record.value?.originalBytesBase64 !== 'string'
    || !record.value.originalBytesBase64) return null;
  let bytes;
  try {
    bytes = base64ToBytes(record.value.originalBytesBase64);
  } catch {
    return null;
  }
  if (!bytes.byteLength) return null;
  const sha256 = await sha256Base64Url(bytes);
  return {
    ...record.value,
    document: {
      ...documentRecord,
      id: `doc:${sha256.slice(0, 24)}`,
      size: bytes.byteLength,
      sha256
    }
  };
};

const repairLegacyDocumentRecords = async (records) => {
  let repairedCount = 0;
  for (const record of records || []) {
    const value = await repairDocumentMetadataFromRetainedBytes(record);
    if (!value) continue;
    await state.vault.putRecord({
      id: record.id,
      kind: record.kind,
      schemaVersion: record.schemaVersion,
      value
    });
    repairedCount += 1;
  }
  return repairedCount;
};

const isEligibleTrackerDocument = record => {
  const documentRecord = record?.value?.document;
  if (!documentRecord || !record.value.originalBytesBase64) return false;
  const lowerName = String(documentRecord.filename || '').toLocaleLowerCase('en-US');
  const isPdf = documentRecord.mimeType === 'application/pdf' && lowerName.endsWith('.pdf');
  const isDocx = documentRecord.mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    && lowerName.endsWith('.docx');
  return (isPdf || isDocx) && documentRecord.size > 0 && documentRecord.size <= MAX_TRACKER_FILE_BYTES;
};

export const filterSelectedDocumentRecords = (records, selectedIds, applicationId) => {
  const selected = selectedIds instanceof Set ? selectedIds : new Set(selectedIds || []);
  return (records || []).filter((record) => {
    if (!selected.has(record.id)) return false;
    const recordApplicationId = record.value?.document?.applicationId || null;
    return !recordApplicationId || recordApplicationId === applicationId;
  });
};

export const formatBytes = (bytes) => {
  const value = Number(bytes);
  if (!Number.isSafeInteger(value) || value < 1) return 'Size unavailable';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
};

const emptyInline = text => createElement('p', { className: 'empty-inline', text });

const renderFactList = () => {
  const list = $('[data-fact-list]');
  if (!list) return;
  list.replaceChildren();
  const verifiedFacts = state.facts.filter(record => !CUSTOM_PROFILE_CATEGORY_SET.has(record.value?.category));
  if (!verifiedFacts.length) {
    list.append(emptyInline('No verified facts yet.'));
    return;
  }
  verifiedFacts
    .slice()
    .sort((left, right) => String(left.value?.label).localeCompare(String(right.value?.label)))
    .forEach((record) => {
      const item = createElement('article', { className: 'record-item' });
      const copy = createElement('div');
      copy.append(createElement('strong', { text: record.value.label || FACT_DEFINITIONS[record.value.key]?.label || 'Verified fact' }));
      copy.append(createElement('span', { className: 'record-value', text: record.value.value }));
      copy.append(createElement('span', { className: 'record-meta', text: 'User verified · exact-match only' }));
      const remove = createElement('button', {
        className: 'button button-quiet button-small',
        text: 'Delete',
        attributes: { type: 'button', 'aria-label': `Delete ${record.value.label || 'verified fact'}` },
        dataset: { deleteRecord: record.id }
      });
      item.append(copy, remove);
      list.append(item);
    });
};

const renderCustomProfileRecords = () => {
  const presentations = [{
    category: CUSTOM_PROFILE_CATEGORIES.LINK,
    listSelector: '[data-custom-link-list]',
    countSelector: '[data-custom-link-count]',
    empty: 'No custom links saved.',
    meta: 'Exact custom link - deterministic match'
  }, {
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    listSelector: '[data-custom-thought-list]',
    countSelector: '[data-custom-thought-count]',
    empty: 'No custom thoughts saved.',
    meta: 'User-provided context - local Ollama'
  }];
  presentations.forEach((presentation) => {
    const list = $(presentation.listSelector);
    const count = $(presentation.countSelector);
    if (!list || !count) return;
    const reusableRecords = customProfileRecords(state.facts, presentation.category);
    const records = (presentation.category === CUSTOM_PROFILE_CATEGORIES.THOUGHT
      ? [...reusableRecords, ...state.facts.filter(isFieldClarificationRecord)]
      : reusableRecords)
      .sort((left, right) => String(left.value?.label || '').localeCompare(String(right.value?.label || '')));
    count.textContent = `${records.length} saved`;
    list.replaceChildren();
    if (!records.length) {
      list.append(emptyInline(presentation.empty));
      return;
    }
    records.forEach((record) => {
      const item = createElement('article', { className: 'record-item' });
      const copy = createElement('div');
      copy.append(createElement('strong', { text: record.value.label || 'Custom profile item' }));
      copy.append(createElement('span', { className: 'record-value', text: record.value.value }));
      copy.append(createElement('span', {
        className: 'record-meta',
        text: isFieldClarificationRecord(record)
          ? 'Application-only clarification - encrypted local evidence'
          : presentation.meta
      }));
      const remove = createElement('button', {
        className: 'button button-quiet button-small',
        text: 'Delete',
        attributes: { type: 'button', 'aria-label': `Delete ${record.value.label || 'custom profile item'}` },
        dataset: { deleteRecord: record.id }
      });
      item.append(copy, remove);
      list.append(item);
    });
  });
};

const renderVerifiedProfile = () => {
  const form = $('[data-profile-form]');
  const count = $('[data-profile-count]');
  const status = $('[data-profile-save-status]');
  if (!form || !count || !status) return;
  const values = verifiedProfileValues(state.facts);
  const savedCount = Object.values(values).filter(value => String(value).trim()).length;
  count.textContent = `${savedCount} of ${Object.keys(FACT_DEFINITIONS).length} saved`;
  if (!state.profileDirty) {
    for (const [key, value] of Object.entries(values)) {
      const input = form.elements.namedItem(key);
      if (input) input.value = value;
    }
  }
  status.textContent = state.profileDirty
    ? 'Unsaved changes. Save once when your profile is ready.'
    : savedCount
      ? 'Profile saved locally. Clearing a saved field removes it on the next save.'
      : 'Blank fields stay out of the vault. Add any details you want to reuse.';
};

const renderDocumentList = () => {
  const list = $('[data-document-list]');
  const count = $('[data-document-count]');
  if (!list || !count) return;
  const catalogue = sourceCatalogueSummary(state.documents, state.selectedSourceIds, state.scan?.pageId);
  const catalogueById = new Map(catalogue.entries.map(entry => [entry.id, entry]));
  count.textContent = `${catalogue.activeCount} in use / ${catalogue.totalCount} saved`;
  list.replaceChildren();
  if (!state.documents.length) {
    list.append(emptyInline('No imported documents or notes yet.'));
    return;
  }
  state.documents
    .slice()
    .sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt)))
    .forEach((record) => {
      const documentRecord = record.value.document || {};
      const catalogueEntry = catalogueById.get(record.id);
      const item = createElement('article', { className: 'record-item' });
      const copy = createElement('div');
      copy.append(createElement('strong', { text: documentRecord.filename || 'Untitled source' }));
      const meta = createElement('div', { className: 'record-meta' });
      meta.append(
        createElement('span', { text: SOURCE_ROLE_LABELS[documentRecord.sourceRole] || documentRecord.sourceRole || 'Source' }),
        createElement('span', { text: formatBytes(documentRecord.size) })
      );
      if (formatBytes(documentRecord.size) === 'Size unavailable') {
        meta.append(createElement('span', { text: 'Re-import to restore file metadata' }));
      }
      if (isEligibleTrackerDocument(record)) meta.append(createElement('span', { text: 'Attachment ready' }));
      if (catalogueEntry?.recommended) meta.append(createElement('span', { text: 'Recommended default' }));
      if (catalogueEntry?.applicationId) {
        meta.append(createElement('span', {
          text: catalogueEntry.availableForApplication ? 'This application' : 'Saved for another application'
        }));
      }
      if (record.value.warnings?.length) meta.append(createElement('span', { text: `${record.value.warnings.length} parser warning(s)` }));
      copy.append(meta);
      const useLabel = createElement('label', { className: 'source-toggle' });
      const useInput = createElement('input', {
        attributes: { type: 'checkbox', 'aria-label': `Use ${documentRecord.filename || 'source'} for this application` },
        dataset: { sourceSelection: record.id }
      });
      useInput.checked = Boolean(catalogueEntry?.selected && catalogueEntry.availableForApplication);
      useInput.disabled = catalogueEntry?.availableForApplication === false;
      const selectionLabel = catalogueEntry?.availableForApplication === false
        ? 'Saved for another application'
        : catalogueEntry?.recommended
          ? 'Recommended for this application'
          : 'Use for this application';
      useLabel.append(useInput, createElement('span', { text: selectionLabel }));
      copy.append(useLabel);
      const remove = createElement('button', {
        className: 'button button-quiet button-small',
        text: 'Delete',
        attributes: { type: 'button', 'aria-label': `Delete ${documentRecord.filename || 'source'}` },
        dataset: { deleteRecord: record.id }
      });
      item.append(copy, remove);
      list.append(item);
    });
};

const renderTrackerDocumentOptions = () => {
  const resume = $('[name="resumeRecordId"]');
  const cover = $('[name="coverRecordId"]');
  if (!resume || !cover) return;
  const selectedResume = resume.value;
  const selectedCover = cover.value;
  resume.replaceChildren(createElement('option', { text: 'No resume', attributes: { value: '' } }));
  cover.replaceChildren(createElement('option', { text: 'No cover letter', attributes: { value: '' } }));
  state.documents.filter(isEligibleTrackerDocument).forEach((record) => {
    const label = `${record.value.document.filename} (${formatBytes(record.value.document.size)})`;
    resume.append(createElement('option', { text: label, attributes: { value: record.id } }));
    cover.append(createElement('option', { text: label, attributes: { value: record.id } }));
  });
  if ([...resume.options].some(option => option.value === selectedResume)) resume.value = selectedResume;
  if ([...cover.options].some(option => option.value === selectedCover)) cover.value = selectedCover;
};

function renderSources() {
  const gate = $('[data-vault-gate]');
  const content = $('[data-vault-content]');
  if (!gate || !content) return;
  gate.hidden = state.unlocked;
  content.hidden = !state.unlocked;
  setVaultIndicator(state.unlocked);
  if (!state.unlocked) {
    const title = $('[data-vault-gate-title]');
    const copy = $('[data-vault-gate-copy]');
    const submit = $('[data-vault-submit]');
    title.textContent = state.initialized ? 'Unlock your vault' : 'Create your vault';
    copy.textContent = state.initialized
      ? 'Unlock to decrypt your local sources for this side-panel session. The passphrase is never stored.'
      : 'Your documents, notes, verified facts, retrieval data, and exact page caches are encrypted before IndexedDB storage. The passphrase is never stored.';
    submit.textContent = state.initialized ? 'Unlock encrypted vault' : 'Create encrypted vault';
  }
  renderVerifiedProfile();
  renderCustomProfileRecords();
  renderDocumentList();
  renderTrackerDocumentOptions();
};

const refreshVaultRecords = async () => {
  if (!state.unlocked) return;
  let [documents, facts, applicationPackets] = await Promise.all([
    state.vault.listRecords('document'),
    state.vault.listRecords('verified-fact'),
    state.vault.listRecords(APPLICATION_PACKET_RECORD_KIND)
  ]);
  const repairedCount = await repairLegacyDocumentRecords(documents);
  if (repairedCount) documents = await state.vault.listRecords('document');
  state.documents = documents;
  if (!state.sourceCatalogueInitialized) {
    const selection = recommendedSourceSelection(documents, state.scan?.pageId);
    state.selectedSourceIds = selection;
    state.sourceSelectionsByScope.set(state.sourceSelectionScope, selection);
    state.sourceCatalogueInitialized = true;
  }
  state.facts = facts;
  state.applicationPackets = applicationPackets;
  renderSources();
  renderApplication();
  if (repairedCount) {
    showToast(`${repairedCount} encrypted ${repairedCount === 1 ? 'source' : 'sources'} recovered from retained file bytes.`, 'success');
  }
};

const refreshVaultStatus = async () => {
  await state.vault.open();
  state.initialized = await state.vault.isInitialized();
  state.unlocked = state.initialized && await state.vault.isUnlocked();
  if (state.unlocked) await refreshVaultRecords();
  else clearDecryptedState();
  renderSources();
};

const parseDocumentInWorker = async (file) => {
  const originalBytes = await readDocumentBytes(file);
  const parserBytes = originalBytes.slice();
  return new Promise((resolve, reject) => {
    const worker = new Worker(chrome.runtime.getURL('workers/document-parser.worker.js'), { type: 'module' });
    const requestId = crypto.randomUUID();
    const cleanup = () => worker.terminate();
    worker.addEventListener('message', (event) => {
      if (event.data?.requestId !== requestId) return;
      cleanup();
      if (event.data.ok) resolve({ parsed: event.data.result, originalBytes });
      else reject(new Error(event.data?.error?.message || 'Unable to parse document.'));
    });
    worker.addEventListener('error', (event) => {
      cleanup();
      reject(new Error(event.message || 'The document parser worker failed.'));
    }, { once: true });
    worker.postMessage({
      requestId,
      document: {
        name: file.name,
        type: file.type,
        data: parserBytes,
        lastModified: file.lastModified
      }
    }, [parserBytes.buffer]);
  });
};

const installVaultActions = () => {
  $('[data-vault-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = $('[data-vault-submit]');
    const passphrase = String(new FormData(form).get('passphrase') || '');
    const wasInitialized = state.initialized;
    setButtonBusy(button, true, wasInitialized ? 'Unlocking...' : 'Creating...');
    try {
      if (state.initialized) await state.vault.unlock(passphrase);
      else await state.vault.initialize(passphrase);
      form.reset();
      await refreshVaultStatus();
      showToast(wasInitialized ? 'Vault unlocked for this browser session.' : 'Encrypted vault created.', 'success');
      if (wasInitialized) activateTab('application');
    } catch (error) {
      form.reset();
      showToast(readableError(error), 'error');
    } finally {
      setButtonBusy(button, false);
    }
  });

  const profileForm = $('[data-profile-form]');
  profileForm.addEventListener('input', () => {
    state.profileDirty = true;
    renderVerifiedProfile();
  });
  profileForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = $('[data-profile-submit]');
    setButtonBusy(button, true, 'Saving profile...');
    try {
      const changes = verifiedFactChanges({
        values: Object.fromEntries(new FormData(form)),
        existingFacts: state.facts,
        verifiedAt: new Date().toISOString()
      });
      if (!changes.changedCount) {
        state.profileDirty = false;
        renderVerifiedProfile();
        showToast('Verified profile is already up to date.', 'success');
        return;
      }
      for (const record of changes.upserts) await state.vault.putRecord(record);
      for (const recordId of changes.deletes) await state.vault.deleteRecord(recordId);
      state.profileDirty = false;
      await refreshVaultRecords();
      await invalidateGeneratedAnswers();
      const updated = changes.upserts.length;
      const removed = changes.deletes.length;
      const summary = [
        updated ? `${updated} ${updated === 1 ? 'field' : 'fields'} saved` : '',
        removed ? `${removed} removed` : ''
      ].filter(Boolean).join(', ');
      showToast(`Verified profile updated: ${summary}.`, 'success');
    } catch (error) {
      showToast(readableError(error), 'error');
      await refreshVaultRecords();
    } finally {
      setButtonBusy(button, false);
      renderVerifiedProfile();
    }
  });

  $$('[data-custom-profile-form]').forEach((customForm) => {
    customForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const button = form.querySelector('button[type="submit"]');
      const values = new FormData(form);
      const category = form.dataset.customProfileForm;
      setButtonBusy(button, true, 'Encrypting...');
      try {
        const record = createCustomProfileRecord({
          category,
          label: values.get('label'),
          value: values.get('value'),
          existingFacts: state.facts
        });
        await state.vault.putRecord(record);
        form.reset();
        await refreshVaultRecords();
        await invalidateGeneratedAnswers();
        showToast(category === CUSTOM_PROFILE_CATEGORIES.LINK
          ? 'Custom link encrypted and added.'
          : 'Custom thought encrypted and added.', 'success');
      } catch (error) {
        showToast(readableError(error), 'error');
      } finally {
        setButtonBusy(button, false);
      }
    });
  });

  $('[data-document-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const values = new FormData(form);
    const files = [...form.elements.documents.files];
    const sourceRole = String(values.get('sourceRole') || SOURCE_ROLES.CANDIDATE_EVIDENCE);
    const progress = $('[data-import-progress]');
    const progressLabel = $('[data-import-progress-label]');
    const progressBar = progress.querySelector('progress');
    setButtonBusy(button, true, 'Importing...');
    progress.hidden = false;
    progressBar.max = Math.max(1, files.length);
    progressBar.value = 0;
    try {
      let catalogueRecords = state.documents.slice();
      const applicationId = importedSourceApplicationId({
        sourceRole,
        applicationId: state.scan?.pageId
      });
      for (let index = 0; index < files.length; index += 1) {
        const file = files[index];
        progressLabel.textContent = `Parsing ${file.name} (${index + 1} of ${files.length})...`;
        const { parsed, originalBytes } = await parseDocumentInWorker(file);
        const retainOriginal = true;
        const value = createDocumentVaultRecord(parsed, {
          sourceRole,
          applicationId,
          retainOriginal,
          originalBytes
        });
        const importedRecord = { id: parsed.document.id, kind: 'document', value };
        await state.vault.putRecord(importedRecord);
        state.selectedSourceIds = selectImportedSource({
          selectedIds: state.selectedSourceIds,
          records: catalogueRecords,
          importedRecord
        });
        state.sourceSelectionsByScope.set(state.sourceSelectionScope, state.selectedSourceIds);
        catalogueRecords = [...catalogueRecords.filter(record => record.id !== importedRecord.id), importedRecord];
        progressBar.value = index + 1;
      }
      form.reset();
      await refreshVaultRecords();
      await invalidateGeneratedAnswers();
      showToast(`${files.length} ${files.length === 1 ? 'source' : 'sources'} imported and encrypted.`, 'success');
    } catch (error) {
      showToast(readableError(error), 'error');
      await refreshVaultRecords();
      await invalidateGeneratedAnswers();
    } finally {
      progress.hidden = true;
      setButtonBusy(button, false);
    }
  });

  $('[data-note-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const values = new FormData(form);
    setButtonBusy(button, true, 'Encrypting...');
    try {
      const snapshot = await state.research.createSnapshot({
        title: values.get('title'),
        url: values.get('url'),
        text: values.get('text')
      });
      const sourceRole = String(values.get('sourceRole') || SOURCE_ROLES.COMPANY_CONTEXT);
      snapshot.document.sourceRole = sourceRole;
      snapshot.document.applicationId = importedSourceApplicationId({
        sourceRole,
        applicationId: state.scan?.pageId
      });
      const importedRecord = { id: snapshot.document.id, kind: 'document', value: snapshot };
      await state.vault.putRecord(importedRecord);
      state.selectedSourceIds = selectImportedSource({
        selectedIds: state.selectedSourceIds,
        records: state.documents,
        importedRecord
      });
      state.sourceSelectionsByScope.set(state.sourceSelectionScope, state.selectedSourceIds);
      form.reset();
      await refreshVaultRecords();
      await invalidateGeneratedAnswers();
      showToast('Pasted note encrypted and added.', 'success');
    } catch (error) {
      showToast(readableError(error), 'error');
    } finally {
      setButtonBusy(button, false);
    }
  });

  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-delete-record]');
    if (!button) return;
    const recordId = button.dataset.deleteRecord;
    if (!confirm('Delete this encrypted source from the local vault?')) return;
    button.disabled = true;
    try {
      await state.vault.deleteRecord(recordId);
      state.sourceSelectionsByScope.forEach(selection => selection.delete(recordId));
      await refreshVaultRecords();
      await invalidateGeneratedAnswers();
      showToast('Encrypted source deleted.', 'success');
    } catch (error) {
      showToast(readableError(error), 'error');
      button.disabled = false;
    }
  });

  document.addEventListener('change', async (event) => {
    const input = event.target.closest('[data-source-selection]');
    if (!input) return;
    if (input.checked) state.selectedSourceIds.add(input.dataset.sourceSelection);
    else state.selectedSourceIds.delete(input.dataset.sourceSelection);
    await invalidateGeneratedAnswers();
    renderSources();
  });
};

const runtimeEnvelope = (type, payload = {}) => ({
  channel: RUNTIME_CHANNEL,
  version: RUNTIME_VERSION,
  type,
  requestId: `panel-${Date.now().toString(36)}-${crypto.randomUUID().slice(0, 8)}`,
  payload
});

export const activeHttpTab = async (tabsApi = chrome.tabs) => {
  const [tab] = await tabsApi.query({ active: true, currentWindow: true });
  if (!Number.isSafeInteger(tab?.id)) throw new Error('No active browser tab is available.');
  if (typeof tab.url !== 'string' || !tab.url.trim()) {
    throw new Error('Chrome has not granted access to this page. Click the Job Application Copilot toolbar icon while the job application page is active, then select Rescan.');
  }
  let url;
  try {
    url = new URL(tab.url);
  } catch {
    throw new Error('The active tab does not expose a supported web page.');
  }
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('Open an HTTP(S) job application page before analyzing.');
  }
  return tab;
};

const sendPageMessage = async (tabId, type, payload = {}) => {
  const request = runtimeEnvelope(type, payload);
  const response = await chrome.tabs.sendMessage(tabId, request);
  if (!response || response.channel !== RUNTIME_CHANNEL || response.version !== RUNTIME_VERSION
    || response.requestId !== request.requestId) {
    throw new Error('The page runtime returned an invalid response.');
  }
  return response;
};

const packetRecordByApplicationId = applicationId => state.applicationPackets.find(
  record => record.value?.applicationId === applicationId
) || null;

const latestPacketContext = packetRecord => {
  const snapshots = packetRecord?.value?.contextSnapshots || [];
  const lastContextId = packetRecord?.value?.lastContextId;
  return snapshots.find(snapshot => snapshot.contextId === lastContextId)
    || snapshots[0]
    || null;
};

const packetMatchForScan = scan => {
  const input = { job: scan?.job || {}, url: scan?.job?.jobUrl || '' };
  const direct = matchApplicationPacketRecord(state.applicationPackets, input);
  if (direct) return direct;
  const active = packetRecordByApplicationId(state.activeApplicationPacketId);
  const hasRoleIdentity = Boolean(String(scan?.job?.company || '').trim() && String(scan?.job?.title || '').trim());
  return active && Array.isArray(scan?.fields) && scan.fields.length > 0 && !hasRoleIdentity
    ? { record: active, reason: 'active_navigation' }
    : null;
};

export const mergeApplicationPacketContext = ({ scan, packetRecord, preferPacket = false } = {}) => {
  if (!scan || !packetRecord) return scan;
  const context = latestPacketContext(packetRecord);
  if (!context) return scan;
  const packetJob = context.job || {};
  const currentJob = scan.job || {};
  const preferred = (currentValue, packetValue) => preferPacket
    ? String(packetValue || currentValue || '')
    : String(currentValue || packetValue || '');
  return {
    ...scan,
    job: {
      ...currentJob,
      company: preferred(currentJob.company, packetJob.company || packetRecord.value?.identity?.company),
      title: preferred(currentJob.title, packetJob.title || packetRecord.value?.identity?.title),
      location: preferred(currentJob.location, packetJob.location || packetRecord.value?.identity?.location),
      source: preferred(currentJob.source, packetJob.source),
      jobUrl: preferred(currentJob.jobUrl, packetJob.jobUrl || context.source?.url),
      description: preferred(currentJob.description, context.content)
    }
  };
};

const applicationPacketContextForScan = (scan, overrides = {}) => {
  const origin = overrides.origin || APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE;
  const mode = overrides.mode || (origin === APPLICATION_CONTEXT_ORIGINS.MANUAL
    ? APPLICATION_CONTEXT_MODES.MANUAL
    : scan?.fields?.length
      ? APPLICATION_CONTEXT_MODES.APPLICATION
      : APPLICATION_CONTEXT_MODES.POSTING);
  const overrideJob = overrides.job || {};
  const description = String(overrides.content ?? overrideJob.description ?? scan?.job?.description ?? '').trim();
  const job = {
    ...(scan?.job || {}),
    ...overrideJob,
    description
  };
  return {
    origin,
    mode,
    capturedAt: new Date().toISOString(),
    label: overrides.label || job.source || job.title || 'Captured job context',
    url: overrides.url ?? job.jobUrl ?? '',
    content: description,
    job,
    ...(origin === APPLICATION_CONTEXT_ORIGINS.MANUAL ? {} : {
      page: {
        adapter: scan?.adapter || '',
        pageId: scan?.pageId || '',
        urlHash: scan?.urlHash || '',
        domRevision: Number.isSafeInteger(scan?.domRevision) ? scan.domRevision : 0
      }
    })
  };
};

const preflightForPacket = scan => summarizeApplicationPreflight({
  fields: scan?.fields || [],
  proposals: state.proposals,
  filledFieldIds: state.filledFieldIds,
  unsupportedCount: Math.max(0, Number(scan?.discovery?.unsupportedCount) || 0)
});

const persistApplicationPacket = async ({
  scan = state.scan,
  context = null,
  approvedAnswers = [],
  clarifications = []
} = {}) => {
  if (!state.unlocked || !scan) return null;
  const contextInput = context || applicationPacketContextForScan(scan);
  let existingRecord = packetMatchForScan(scan)?.record || null;
  if (!existingRecord && state.activeApplicationPacketId) {
    const active = packetRecordByApplicationId(state.activeApplicationPacketId);
    const lacksRoleIdentity = !String(contextInput.job?.company || '').trim()
      || !String(contextInput.job?.title || '').trim();
    if (active && lacksRoleIdentity) existingRecord = active;
  }
  const packet = await upsertApplicationPacketRecord({
    existingRecord,
    context: contextInput,
    approvedAnswers,
    clarifications,
    preflight: preflightForPacket(scan)
  });
  await state.vault.putRecord(packet);
  const localRecord = {
    ...packet,
    schemaVersion: 1,
    createdAt: packet.value.createdAt,
    updatedAt: packet.value.updatedAt
  };
  state.applicationPackets = [
    ...state.applicationPackets.filter(record => record.id !== localRecord.id),
    localRecord
  ];
  state.activeApplicationPacketId = packet.value.applicationId;
  return localRecord;
};

const scanActivePage = async () => {
  const tab = await activeHttpTab();
  await chrome.scripting.executeScript({
    target: { tabId: tab.id, frameIds: [0] },
    files: ['content/page-runtime.js']
  });
  const response = await sendPageMessage(tab.id, 'PAGE_SCAN_REQUEST');
  if (response.type !== 'PAGE_SCAN_RESULT' || !response.payload?.pageId || !Array.isArray(response.payload.fields)) {
    throw new Error('The page scan response was incomplete.');
  }
  const packetMatch = packetMatchForScan(response.payload);
  const scan = packetMatch
    ? mergeApplicationPacketContext({ scan: response.payload, packetRecord: packetMatch.record })
    : response.payload;
  if (packetMatch) state.activeApplicationPacketId = packetMatch.record.value.applicationId;
  activateSourceSelectionScope(scan);
  state.activeTabId = tab.id;
  state.scan = scan;
  state.stale = false;
  state.evidence = null;
  state.proposals = [];
  state.filledFieldIds.clear();
  state.batchStatus = '';
  state.answerFilter = 'all';
  state.analysisNotice = '';
  const batchReview = $('[data-batch-review]');
  if (batchReview) batchReview.checked = false;
  state.retrievalMode = '';
  renderSources();
  return scan;
};

const factDocumentRecord = (record, applicationId) => ({
  document: {
    id: record.id,
    filename: record.value.label,
    mimeType: 'text/plain',
    size: new TextEncoder().encode(record.value.value).byteLength,
    sha256: record.id,
    version: 1,
    sourceRole: SOURCE_ROLES.USER_VERIFIED,
    applicationId
  },
  text: `${record.value.label}: ${record.value.value}`,
  blocks: [{
    blockIndex: 0,
    paragraph: 1,
    section: record.value.category === CUSTOM_PROFILE_CATEGORIES.THOUGHT
      ? 'Custom profile thought'
      : record.value.category === CUSTOM_PROFILE_CATEGORIES.LINK
        ? 'Custom profile link' : 'Verified profile fact',
    page: null,
    text: `${record.value.label}: ${record.value.value}`
  }],
  factKey: record.value.key
});
const LIVE_JOB_DOCUMENT_PREFIX = 'live-job:';
const POSTED_SALARY_RANGE_PATTERN = /((?:[\$\u20ac\u00a3]\s*)?\d{2,3}(?:,\d{3})+(?:\.\d+)?\s*(?:USD|CAD|EUR|GBP)?\s*(?:-|to|\u2013|\u2014)\s*(?:[\$\u20ac\u00a3]\s*)?\d{2,3}(?:,\d{3})+(?:\.\d+)?\s*(?:USD|CAD|EUR|GBP)?)/iu;

export const createLiveJobRequirementRecord = (scan) => {
  const description = String(scan?.job?.description || '').replace(/\s+/gu, ' ').trim();
  const pageId = String(scan?.pageId || '').trim();
  if (!description || !pageId) return null;
  const title = String(scan?.job?.title || '').trim();
  const company = String(scan?.job?.company || '').trim();
  const location = String(scan?.job?.location || '').trim();
  const heading = [
    title ? 'Position: ' + title : '',
    company ? 'Company: ' + company : '',
    location ? 'Location: ' + location : ''
  ].filter(Boolean).join('\n');
  const text = [heading, description].filter(Boolean).join('\n\n');
  const documentId = LIVE_JOB_DOCUMENT_PREFIX + pageId;
  return {
    document: {
      id: documentId,
      filename: 'Live job posting',
      mimeType: 'text/plain',
      size: new TextEncoder().encode(text).byteLength,
      sha256: 'live:' + pageId + ':' + description.length,
      version: 1,
      sourceRole: SOURCE_ROLES.JOB_REQUIREMENT,
      applicationId: pageId,
      sourceUrl: String(scan?.job?.jobUrl || '')
    },
    text,
    blocks: [{
      blockIndex: 0,
      paragraph: 1,
      section: 'Live job posting',
      page: null,
      text
    }]
  };
};

export const postedSalaryRangeFromText = (value) => {
  const match = POSTED_SALARY_RANGE_PATTERN.exec(String(value || '').replace(/\s+/gu, ' '));
  return match?.[1]?.trim() || '';
};

export const createPostedSalaryProposal = ({ field, evidence }) => {
  if (!field || field.riskClass !== 'F2_REVIEW' || !isSalaryExpectationField(field)) return null;
  if (!['text', 'textarea', 'contenteditable'].includes(field.type)) return null;
  const citationsById = new Map((evidence?.citations || []).map(citation => [citation.citationId, citation]));
  const allowed = evidence?.byField?.[field.fieldId] || [];
  for (const citationId of allowed) {
    const citation = citationsById.get(citationId);
    if (citation?.sourceRole !== SOURCE_ROLES.JOB_REQUIREMENT) continue;
    const range = postedSalaryRangeFromText(citation.text);
    if (!range) continue;
    const maxLength = Number.isSafeInteger(field.maxLength) && field.maxLength > 0
      ? field.maxLength
      : Number.POSITIVE_INFINITY;
    const value = [
      'I am open to discussing compensation within the posted range of ' + range + ', taking the full compensation package into account.',
      'Within the posted range of ' + range + '.',
      range
    ].find(candidate => candidate.length <= maxLength);
    if (!value) continue;
    return {
      field_id: field.fieldId,
      action: 'fill',
      confidence: 'review',
      risk_class: field.riskClass,
      value_type: 'text',
      value,
      selected_values: [],
      checked: false,
      citation_ids: [citationId],
      short_rationale: 'Uses only the exact compensation range published in the live job posting.',
      abstain_reason: ''
    };
  }
  return null;
};

const EXPERIENCE_FIELD_PATTERN = /\bexperience\b/iu;
const EXPERIENCE_NARRATIVE_PATTERN = /\b(?:how many|years?|explain|describe|relevant|direct|background)\b/iu;
const EXPERIENCE_TRANSFER_QUERY = 'transferable responsibilities projects achievements skills employment analytics reporting';
const CANDIDATE_CONTEXT_ROLES = Object.freeze([
  SOURCE_ROLES.CANDIDATE_EVIDENCE,
  SOURCE_ROLES.USER_VERIFIED
]);

export const isNarrativeExperienceField = field => field?.riskClass === 'F2_REVIEW'
  && ['text', 'textarea', 'contenteditable'].includes(field?.type)
  && EXPERIENCE_FIELD_PATTERN.test(`${field?.label || ''} ${field?.nearbyText || ''}`)
  && EXPERIENCE_NARRATIVE_PATTERN.test(`${field?.label || ''} ${field?.nearbyText || ''}`);

const EXPERIENCE_RECOVERY_INSTRUCTION = [
  'Create the safest useful answer from the current evidence.',
  'Write the field value in natural first-person applicant voice. Never mention evidence, citations, sources, a resume, documents, supplied materials, the model, or the system in the answer.',
  'If the supplied evidence does not establish direct experience or a numeric duration in the requested area, do not invent either.',
  'Instead, candidly say in first person that the applicant does not have a specific duration of direct experience to report, then summarize only cited adjacent candidate experience.',
  'Return a review-only text value rather than abstaining when adjacent candidate evidence is available.'
].join(' ');

const EXPERIENCE_ACTION_PATTERN = /\b(?:analyzed|automated|built|coordinated|created|delivered|designed|developed|implemented|improved|launched|led|managed|optimized|owned|produced|reported|supported|worked)\b/iu;
const EXPERIENCE_ACTION_PREFIX_PATTERN = /^(?:analyzed|automated|built|coordinated|created|delivered|designed|developed|implemented|improved|launched|led|managed|optimized|owned|produced|reported|supported|worked)\b/iu;
const EVIDENCE_GAP_PATTERN = /\b(?:resume|materials?|documents?|evidence)\b[^.!?]{0,140}\b(?:do not|does not|did not|not|no)\b/iu;
const APPLICANT_UNCERTAINTY_PATTERN = /\bi\s+(?:(?:cannot|can not|can't)\s+(?:confidently\s+)?claim\b[^.!?]{0,120}\bdirect\b[^.!?]{0,80}\bexperience\b|(?:do not|don't)\s+have\b[^.!?]{0,80}\bspecific\s+number\b[^.!?]{0,80}\bdirect\b[^.!?]{0,80}\bexperience\b[^.!?]{0,40}\breport\b)/iu;
const EXPERIENCE_RECOVERY_RATIONALE_PREFIX = 'Best-effort applicant draft:';
const EXPERIENCE_TOPIC_SAFE_PATTERN = /^[\p{L}\p{N}+#&/ .-]{2,60}$/u;
const EXPERIENCE_TOPIC_BLOCKED_PATTERN = /\b(?:answer|claim|describe|explain|ignore|include|please|say|tell|write|you|your)\b/iu;
const FIELD_CLARIFICATION_EVIDENCE_PREFIX_PATTERN = /^Clarification - [^\r\n]*? \[[A-Za-z0-9_-]{1,128}\]:\s*/u;
const DIRECT_EXPERIENCE_GAP_UNIT_PATTERN = /^(?:i\s+(?:(?:do not|don't)\s+have|have\s+no|lack)|no)\b[^.!?]{0,180}\bdirect\b[^.!?]{0,120}\bexperience\b[^.!?]*[.!?]?$/iu;
const EXPERIENCE_LINK_ONLY_UNIT_PATTERN = /^(?:(?:github|linkedin|portfolio|website)(?:\s+(?:profile|url))?|url|link)\s*:\s*https?:\/\/\S+\s*$/iu;

const evidenceUnits = text => String(text || '')
  .replaceAll('\r\n', '\n')
  .replaceAll('\r', '\n')
  .split(/\n+|[\u2022\u25aa\u25e6]\s*|(?<=[.!?])\s+/u)
  .map(value => value.replace(/\s+/gu, ' ').trim())
  .filter(value => value.length >= 24);

const experienceEvidenceUnits = card => evidenceUnits(
  card?.sourceRole === SOURCE_ROLES.USER_VERIFIED
    ? String(card.text || '').replace(FIELD_CLARIFICATION_EVIDENCE_PREFIX_PATTERN, '')
    : card?.text
).filter(unit => !EXPERIENCE_LINK_ONLY_UNIT_PATTERN.test(unit))
  .filter(unit => !(DIRECT_EXPERIENCE_GAP_UNIT_PATTERN.test(unit)
    && !/[;:]/u.test(unit)
    && !/\b(?:although|but|however|while)\b/iu.test(unit)));

const clippedEvidenceUnit = (value, maxLength = 240) => {
  const normalized = String(value || '').replace(/\s+/gu, ' ').trim();
  if (normalized.length <= maxLength) return normalized;
  const clipped = normalized.slice(0, maxLength + 1);
  const lastSpace = clipped.lastIndexOf(' ');
  return (lastSpace >= Math.floor(maxLength * 0.7) ? clipped.slice(0, lastSpace) : clipped.slice(0, maxLength)).trim() + '...';
};

const applicantEvidenceUnit = (value) => {
  const normalized = String(value || '').replace(/\s+/gu, ' ').trim();
  if (!normalized || /^i\b/iu.test(normalized) || !EXPERIENCE_ACTION_PREFIX_PATTERN.test(normalized)) return normalized;
  return `I ${normalized[0].toLocaleLowerCase('en-US')}${normalized.slice(1)}`;
};

const requestedExperienceTopic = (field) => {
  const match = String(field?.label || '').match(/\b(?:years?|months?)\s+of\s+(.{2,80}?)\s+experience\b/iu);
  const topic = String(match?.[1] || '').replace(/\s+/gu, ' ').trim();
  if (!EXPERIENCE_TOPIC_SAFE_PATTERN.test(topic)
    || EXPERIENCE_TOPIC_BLOCKED_PATTERN.test(topic)
    || topic.split(/\s+/u).length > 6) return '';
  return topic;
};

const experienceCardsForField = ({ field, evidence }) => {
  const citationIds = evidence?.byField?.[field?.fieldId] || [];
  if (!citationIds.length) return [];
  try {
    return resolveCitationCards({ citationIds, fieldId: field.fieldId, evidence })
      .filter(card => CANDIDATE_CONTEXT_ROLES.includes(card.sourceRole) && String(card.text || '').trim());
  } catch {
    return [];
  }
};

export const proposalDisclosesEvidenceGap = proposal => EVIDENCE_GAP_PATTERN.test(String(proposal?.value || ''))
  || APPLICANT_UNCERTAINTY_PATTERN.test(String(proposal?.value || ''))
  || String(proposal?.short_rationale || '').startsWith(EXPERIENCE_RECOVERY_RATIONALE_PREFIX);

export const clarificationQuestionForProposal = ({ field, proposal, cards = [] } = {}) => {
  if (!field || field.manual || field.riskClass !== 'F2_REVIEW' || !proposal) return '';
  if (isNarrativeExperienceField(field)
    && proposal.action === 'fill'
    && proposal.confidence === 'review'
    && proposalDisclosesEvidenceGap(proposal)
    && cards.some(card => CANDIDATE_CONTEXT_ROLES.includes(card.sourceRole))) {
    return 'Do you have direct experience relevant to this question that is missing from this draft? If yes, provide the organization or project, dates, responsibilities, and outcome. If none, say that explicitly.';
  }
  if (proposal.action !== 'ask_user' && proposal.confidence !== 'needs_input') return '';
  const reason = String(proposal.abstain_reason || '').replace(/\s+/gu, ' ').trim();
  return reason && !/^review this field/iu.test(reason)
    ? reason
    : `What factual information should be used to answer "${String(field.label || 'this question').trim()}"?`;
};

export const createBestEffortExperienceProposal = ({ field, evidence } = {}) => {
  if (!isNarrativeExperienceField(field) || field?.manual) return null;
  const cards = experienceCardsForField({ field, evidence });
  if (!cards.length) return null;

  const rankedUnits = cards.flatMap((card, cardIndex) => experienceEvidenceUnits(card).map((text, unitIndex) => ({
    card,
    cardIndex,
    unitIndex,
    text,
    actionScore: EXPERIENCE_ACTION_PATTERN.test(text) ? 1 : 0
  }))).sort((left, right) => right.actionScore - left.actionScore
    || left.cardIndex - right.cardIndex
    || left.unitIndex - right.unitIndex
    || right.text.length - left.text.length);

  const durationUnit = /\byears?\b/iu.test(field.label || '')
    ? 'years'
    : /\bmonths?\b/iu.test(field.label || '') ? 'months' : '';
  const topic = requestedExperienceTopic(field);
  const directExperience = topic ? `direct ${topic} experience` : 'direct experience in this area';
  const opening = durationUnit
    ? `I don't have a specific number of ${durationUnit} of ${directExperience} to report, but my relevant experience includes:`
    : `I don't have a specific duration of ${directExperience} to report, but my relevant experience includes:`;
  const prefix = opening;
  const maxLength = Number.isSafeInteger(field.maxLength) && field.maxLength > 0
    ? Math.min(field.maxLength, 900)
    : 900;
  let value = prefix;
  const selectedCards = [];
  const selectedText = new Set();
  for (const entry of rankedUnits) {
    const excerpt = applicantEvidenceUnit(clippedEvidenceUnit(entry.text));
    if (!excerpt || selectedText.has(excerpt)) continue;
    const next = value + '\n- ' + excerpt;
    if (next.length > maxLength) continue;
    value = next;
    selectedText.add(excerpt);
    if (!selectedCards.some(card => card.citationId === entry.card.citationId)) selectedCards.push(entry.card);
    if (selectedText.size >= 2) break;
  }
  if (!selectedText.size) return null;

  const proposal = {
    field_id: field.fieldId,
    action: 'fill',
    confidence: 'review',
    risk_class: field.riskClass,
    value_type: 'text',
    value,
    selected_values: [],
    checked: false,
    citation_ids: selectedCards.map(card => card.citationId),
    short_rationale: EXPERIENCE_RECOVERY_RATIONALE_PREFIX + ' the exact direct-experience duration still needs clarification. Add any missing direct experience below, then regenerate.',
    abstain_reason: ''
  };
  const grounded = enforceGroundingAcceptance({
    output: { page_id: 'experience-recovery', proposals: [proposal] },
    request: { pageId: 'experience-recovery', fields: [{ ...field, manual: false }] },
    evidence
  }).proposals[0];
  return grounded.action === 'fill' ? grounded : null;
};
export const createFieldClarificationRecord = ({
  field,
  applicationId,
  text,
  existingFacts = [],
  recordToken = crypto.randomUUID(),
  verifiedAt = new Date().toISOString()
} = {}) => {
  if (!field || field?.manual || field.riskClass !== 'F2_REVIEW') {
    throw new Error('Only consequential, model-eligible fields can store clarification.');
  }
  const normalizedApplicationId = String(applicationId || '').trim();
  const normalizedFieldId = String(field.fieldId || '').trim();
  if (!normalizedApplicationId || !normalizedFieldId) throw new Error('Application and field scope are required.');
  const labelPrefix = 'Clarification - ';
  const labelSuffix = ' [' + String(recordToken).slice(0, 8) + ']';
  const labelSubject = String(field.label || 'Experience question').replace(/\s+/gu, ' ').trim()
    .slice(0, MAX_CUSTOM_LABEL_LENGTH - labelPrefix.length - labelSuffix.length);
  const record = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: labelPrefix + labelSubject + labelSuffix,
    value: text,
    existingFacts: [],
    recordToken,
    verifiedAt
  });
  return {
    ...record,
    value: {
      ...record.value,
      applicationId: normalizedApplicationId,
      fieldId: normalizedFieldId
    }
  };
};
export const ensureExperienceCandidateEvidence = ({
  field,
  results = [],
  chunks = [],
  queryText = '',
  queryEmbedding = null,
  applicationId,
  limit = 5
} = {}) => {
  if (!isNarrativeExperienceField(field)) return results.slice(0, limit);
  const direct = hybridRetrieve({
    queryText: `${queryText} ${EXPERIENCE_TRANSFER_QUERY}`,
    queryEmbedding,
    chunks,
    limit: Math.min(3, limit),
    filters: { applicationId, sourceRoles: CANDIDATE_CONTEXT_ROLES }
  });
  const fallback = hybridRetrieve({
    queryText: EXPERIENCE_TRANSFER_QUERY,
    queryEmbedding: null,
    chunks,
    limit: Math.min(2, limit),
    filters: { applicationId, sourceRoles: [SOURCE_ROLES.CANDIDATE_EVIDENCE] },
    minScore: 0,
    minVectorSimilarity: 0
  });
  const retrievedCandidateEvidence = [...direct, ...fallback]
    .some(result => result?.sourceRole === SOURCE_ROLES.CANDIDATE_EVIDENCE);
  const representative = retrievedCandidateEvidence
    ? null
    : chunks
      .filter(chunk => chunk?.applicationId === applicationId
        && chunk?.sourceRole === SOURCE_ROLES.CANDIDATE_EVIDENCE
        && typeof chunk?.text === 'string'
        && chunk.text.trim())
      .slice()
      .sort((left, right) => right.text.length - left.text.length || left.id.localeCompare(right.id))[0] || null;
  const representativeResult = representative
    ? {
        ...representative,
        retrieval: { score: 0.01, vectorScore: 0, lexicalScore: 0, roleScore: 0.95 }
      }
    : null;
  const merged = [];
  const seen = new Set();
  for (const result of [...(representativeResult ? [representativeResult] : []), ...direct, ...fallback, ...results]) {
    if (!result?.id || seen.has(result.id)) continue;
    seen.add(result.id);
    merged.push(result);
    if (merged.length >= limit) break;
  }
  return merged;
};

export const factAppliesToApplication = (record, applicationId) => {
  const scopedApplicationId = String(record?.value?.applicationId || '').trim();
  return !scopedApplicationId || scopedApplicationId === String(applicationId || '');
};

const sourceSignatureForRecords = async () => {
  const applicationId = String(state.scan?.pageId || '');
  return sha256Base64Url(JSON.stringify([
    ...filterSelectedDocumentRecords(state.documents, state.selectedSourceIds, applicationId).map(record => [
      record.id,
      record.updatedAt,
      record.value?.document?.sha256,
      record.value?.document?.sourceRole,
      record.value?.document?.applicationId || null
    ]),
    ...state.facts.filter(record => factAppliesToApplication(record, applicationId)).map(record => [
      record.id,
      record.updatedAt,
      record.value?.key,
      record.value?.value,
      record.value?.applicationId || null,
      record.value?.fieldId || null
    ]),
    [
      LIVE_JOB_DOCUMENT_PREFIX + applicationId,
      state.scan?.job?.description || '',
      state.scan?.job?.jobUrl || '',
      state.scan?.job?.title || '',
      state.scan?.job?.company || ''
    ]
  ].sort((left, right) => String(left[0]).localeCompare(String(right[0])))));
};

const buildSourceChunks = async (applicationId) => {
  const chunks = [];
  for (const record of filterSelectedDocumentRecords(state.documents, state.selectedSourceIds, applicationId)) {
    const scoped = {
      ...record.value,
      document: { ...record.value.document, applicationId }
    };
    const recordChunks = await chunkDocument(scoped);
    chunks.push(...recordChunks);
  }
  for (const record of state.facts) {
    if (ELIGIBILITY_FACT_KEY_SET.has(record.value?.key) || !factAppliesToApplication(record, applicationId)) continue;
    const factDocument = factDocumentRecord(record, applicationId);
    const recordChunks = await chunkDocument(factDocument);
    recordChunks.forEach(chunk => {
      chunk.factKey = record.value.key;
      if (record.value?.fieldId) chunk.fieldId = record.value.fieldId;
    });
    chunks.push(...recordChunks);
  }
  const liveJob = createLiveJobRequirementRecord(state.scan);
  if (liveJob && liveJob.document.applicationId === applicationId) {
    const recordChunks = await chunkDocument(liveJob);
    chunks.push(...recordChunks);
  }
  return chunks;
};
const cacheRecordId = async (prefix, value) => `${prefix}:${(await sha256Base64Url(value)).slice(0, 32)}`;

const trimEncryptedCaches = async (kind, keep = 10) => {
  const records = await state.vault.listRecords(kind);
  const obsolete = records.sort((left, right) => String(right.updatedAt).localeCompare(String(left.updatedAt))).slice(keep);
  await Promise.all(obsolete.map(record => state.vault.deleteRecord(record.id)));
};

const preparedSourceChunks = async ({ applicationId, sourceSignature, signal }) => {
  const id = await cacheRecordId('retrieval', retrievalCacheSeed({
    applicationId,
    sourceSignature,
    embeddingModel: state.settings.embeddingModel
  }));
  const cached = await state.vault.getRecord(id);
  const cachedValue = cached?.value;
  const cacheHit = cachedValue?.sourceSignature === sourceSignature
    && cachedValue.applicationId === applicationId
    && cachedValue.embeddingModel === state.settings.embeddingModel
    && Array.isArray(cachedValue.chunks);
  let chunks = cacheHit
    ? cachedValue.chunks
    : await buildSourceChunks(applicationId);
  if (!chunks.length) return { chunks: [], mode: 'lexical' };

  if (!chunks.every(chunk => Array.isArray(chunk.embedding) && chunk.embedding.length)) {
    const embedded = await embedChunksWithLexicalFallback({
      client: state.ollama,
      chunks,
      model: state.settings.embeddingModel,
      signal,
      onStatus: update => {
        if (update.code === GENERATION_STATUS_CODES.EMBEDDING_LEXICAL_FALLBACK) {
          updateAnalysisProgress('Embedding unavailable', 'Continuing with deterministic lexical retrieval.');
        }
      }
    });
    chunks = embedded.chunks;
    await state.vault.putRecord({
      id,
      kind: 'retrieval-cache',
      value: {
        schemaVersion: CACHE_SCHEMA_VERSION,
        applicationId,
        sourceSignature,
        embeddingModel: state.settings.embeddingModel,
        mode: embedded.mode,
        chunks,
        createdAt: new Date().toISOString()
      }
    });
    void trimEncryptedCaches('retrieval-cache', 6).catch(() => {});
    return { chunks, mode: embedded.mode };
  }
  return { chunks, mode: preparedChunkMode({ cacheHit, cachedValue }) };
};

export const prioritizeEvidenceResults = (entries = []) => {
  const ordered = Array.isArray(entries) ? entries : [];
  return [
    ...ordered.filter(entry => entry?.narrativeExperience),
    ...ordered.filter(entry => !entry?.narrativeExperience && entry?.riskClass === 'F1_VERIFIED'),
    ...ordered.filter(entry => !entry?.narrativeExperience && entry?.riskClass !== 'F1_VERIFIED')
  ];
};

const retrieveEvidence = async ({ scan, signal }) => {
  const sourceSignature = await sourceSignatureForRecords();
  state.sourceSignature = sourceSignature;
  const prepared = await preparedSourceChunks({ applicationId: scan.pageId, sourceSignature, signal });
  const fields = scan.fields.map(field => ({ ...field, manual: false }));
  if (!prepared.chunks.length) {
    return {
      request: { pageId: scan.pageId, urlHash: scan.urlHash, domRevision: scan.domRevision, fields },
      evidence: { citations: [], byField: Object.fromEntries(fields.map(field => [field.fieldId, []])) },
      mode: 'lexical',
      sourceSignature
    };
  }
  const queries = fields.map(field => [field.label, field.nearbyText, scan.job?.title, scan.job?.company]
    .filter(Boolean).join(' '));
  let queryEmbeddings = null;
  if (prepared.mode === 'hybrid') {
    try {
      const response = await state.ollama.embed({
        model: state.settings.embeddingModel,
        input: queries,
        keepAlive: DEFAULT_LOCAL_MODEL_CONFIG.keepAlive,
        signal
      });
      queryEmbeddings = response.embeddings;
    } catch {
      queryEmbeddings = null;
    }
  }
  const resultsByField = fields.map((field, index) => {
    const fieldChunks = prepared.chunks.filter(chunk => !chunk.fieldId || chunk.fieldId === field.fieldId);
    let results = hybridRetrieve({
      queryText: queries[index],
      queryEmbedding: queryEmbeddings?.[index] || null,
      chunks: fieldChunks,
      limit: 5,
      filters: { applicationId: scan.pageId }
    });
    const exactFactKey = field.riskClass === 'F1_VERIFIED' ? factKeyForFieldLabel(field.label, state.facts) : null;
    const exactFactChunk = exactFactKey ? fieldChunks.find(chunk => chunk.factKey === exactFactKey) : null;
    if (exactFactChunk && !results.some(result => result.id === exactFactChunk.id)) {
      results.unshift({
        ...exactFactChunk,
        retrieval: { score: 2, vectorScore: 1, lexicalScore: 1, roleScore: 1 }
      });
      results.splice(5);
    }
    results = ensureExperienceCandidateEvidence({
      field,
      results,
      chunks: fieldChunks,
      queryText: queries[index],
      queryEmbedding: queryEmbeddings?.[index] || null,
      applicationId: scan.pageId,
      limit: 5
    });
    return { fieldId: field.fieldId, riskClass: field.riskClass, narrativeExperience: isNarrativeExperienceField(field), results };
  });
  const prioritizedResults = prioritizeEvidenceResults(resultsByField);
  return {
    request: { pageId: scan.pageId, urlHash: scan.urlHash, domRevision: scan.domRevision, fields },
    evidence: buildEvidencePack(prioritizedResults),
    mode: queryEmbeddings ? 'hybrid' : 'lexical',
    sourceSignature
  };
};

const emptyProposal = field => ({
  field_id: field.fieldId,
  action: 'ask_user',
  confidence: 'needs_input',
  risk_class: field.riskClass,
  value_type: 'none',
  value: '',
  selected_values: [],
  checked: false,
  citation_ids: [],
  short_rationale: 'No verified answer is available.',
  abstain_reason: 'Review this field and enter the answer yourself.'
});

export const createVerifiedFactProposal = ({ field, fact, citationId, matchedKey = '' }) => {
  if (!field || field.riskClass !== 'F1_VERIFIED' || !fact || !citationId) return null;
  const expectedKey = matchedKey || factKeyForFieldLabel(field.label);
  if (!expectedKey || fact.key !== expectedKey || !String(fact.value || '').trim()) return null;
  const value = String(fact.value);
  const label = String(fact.label || FACT_DEFINITIONS[fact.key]?.label || 'verified fact');
  if (['select-one', 'select-multiple', 'radio'].includes(field.type)) {
    const matchingOption = (field.options || []).find(option => normalizeLabel(option) === normalizeLabel(value));
    if (!matchingOption) return null;
    return {
      field_id: field.fieldId,
      action: 'fill',
      confidence: 'high',
      risk_class: field.riskClass,
      value_type: 'selected_values',
      value: '',
      selected_values: [matchingOption],
      checked: false,
      citation_ids: [citationId],
      short_rationale: `Exact ${label.toLocaleLowerCase('en-US')} from a user-verified profile fact.`,
      abstain_reason: ''
    };
  }
  if (field.type === 'checkbox') return null;
  return {
    field_id: field.fieldId,
    action: 'fill',
    confidence: 'high',
    risk_class: field.riskClass,
    value_type: 'text',
    value,
    selected_values: [],
    checked: false,
    citation_ids: [citationId],
    short_rationale: `Exact ${label.toLocaleLowerCase('en-US')} from a user-verified profile fact.`,
    abstain_reason: ''
  };
};

export const withEligibilityEvidence = ({ evidence, fields = [], facts = [] } = {}) => {
  const citations = [...(evidence?.citations || [])];
  const byField = Object.fromEntries(Object.entries(evidence?.byField || {})
    .map(([fieldId, citationIds]) => [fieldId, [...citationIds]]));
  const recordsByKey = new Map((facts || [])
    .filter(record => ELIGIBILITY_FACT_KEY_SET.has(record?.value?.key))
    .map(record => [record.value.key, record]));
  const citationIdByKey = new Map();
  for (const [key, record] of recordsByKey) {
    const value = eligibilityValue(record.value?.value);
    if (!value) continue;
    const citationId = `profile-${key}`;
    citationIdByKey.set(key, citationId);
    if (citations.some(citation => citation.citationId === citationId)) continue;
    citations.push({
      citationId,
      documentId: record.id || `fact:${key}`,
      documentVersion: 1,
      chunkId: `verified-eligibility:${key}`,
      sourceRole: SOURCE_ROLES.USER_VERIFIED,
      locator: { section: 'Employment eligibility profile' },
      quoteHash: `verified:${record.updatedAt || record.value?.verifiedAt || key}`,
      text: `${FACT_DEFINITIONS[key].label}: ${value === 'yes' ? 'Yes' : 'No'}`
    });
  }
  const values = verifiedProfileValues(facts);
  for (const field of fields || []) {
    const kind = eligibilityQuestionKind(field);
    const derived = deriveEligibilityAnswer({ kind, values });
    if (!derived) continue;
    const citationIds = derived.factKeys.map(key => citationIdByKey.get(key)).filter(Boolean);
    if (citationIds.length !== derived.factKeys.length) continue;
    byField[field.fieldId] = [...new Set([...(byField[field.fieldId] || []), ...citationIds])];
  }
  return { citations, byField };
};

export const evidenceForModelFields = (evidence, fieldIds = []) => {
  const requestedIds = [...new Set((fieldIds || []).filter(fieldId => typeof fieldId === 'string' && fieldId))];
  const allowedByField = Object.fromEntries(requestedIds.map(fieldId => [
    fieldId,
    [...new Set(evidence?.byField?.[fieldId] || [])]
  ]));
  const allowed = new Set(Object.values(allowedByField).flat());
  const citations = (evidence?.citations || []).filter(citation => (
    allowed.has(citation.citationId)
    && !String(citation.chunkId || '').startsWith('verified-eligibility:')
  ));
  const available = new Set(citations.map(citation => citation.citationId));
  return {
    citations,
    byField: Object.fromEntries(requestedIds.map(fieldId => [
      fieldId,
      allowedByField[fieldId].filter(citationId => available.has(citationId))
    ]))
  };
};

export const evidenceForModelField = (evidence, fieldId) => evidenceForModelFields(evidence, [fieldId]);

const matchingEligibilityOption = (options, value) => {
  const expected = normalizeLabel(value);
  return (options || []).find(option => normalizeLabel(option) === expected) || null;
};

export const createEligibilityProposal = ({ field, facts = [], evidence }) => {
  if (!field || field.riskClass !== 'F2_REVIEW') return null;
  const kind = eligibilityQuestionKind(field);
  if (!kind) return null;
  const derived = deriveEligibilityAnswer({ kind, values: verifiedProfileValues(facts) });
  if (!derived) return null;
  const recordsByKey = new Map((facts || []).map(record => [record?.value?.key, record]));
  const allowed = new Set(evidence?.byField?.[field.fieldId] || []);
  const citationIds = derived.factKeys.map((key) => {
    const record = recordsByKey.get(key);
    return (evidence?.citations || []).find(citation => citation.documentId === record?.id
      && allowed.has(citation.citationId))?.citationId || '';
  }).filter(Boolean);
  if (citationIds.length !== derived.factKeys.length) return null;
  const displayValue = derived.value === 'yes' ? 'Yes' : 'No';
  if (['select-one', 'radio'].includes(field.type)) {
    const option = matchingEligibilityOption(field.options, displayValue);
    if (!option) return null;
    return {
      field_id: field.fieldId,
      action: 'fill',
      confidence: 'review',
      risk_class: field.riskClass,
      value_type: 'selected_values',
      value: '',
      selected_values: [option],
      checked: false,
      citation_ids: citationIds,
      short_rationale: 'Uses only explicit work-eligibility selections saved by the user.',
      abstain_reason: ''
    };
  }
  if (!['text', 'textarea', 'contenteditable'].includes(field.type)) return null;
  return {
    field_id: field.fieldId,
    action: 'fill',
    confidence: 'review',
    risk_class: field.riskClass,
    value_type: 'text',
    value: displayValue,
    selected_values: [],
    checked: false,
    citation_ids: citationIds,
    short_rationale: 'Uses only explicit work-eligibility selections saved by the user.',
    abstain_reason: ''
  };
};

const deterministicFactProposal = (field, evidence, facts = state.facts) => {
  if (field.riskClass !== 'F1_VERIFIED') return null;
  const key = factKeyForFieldLabel(field.label, facts);
  const fact = key ? facts.find(record => record.value.key === key) : null;
  if (!fact) return null;
  const citation = evidence.citations.find(candidate => candidate.documentId === fact.id
    && (evidence.byField[field.fieldId] || []).includes(candidate.citationId));
  if (!citation) return null;
  return createVerifiedFactProposal({ field, fact: fact.value, citationId: citation.citationId, matchedKey: key });
};

const deterministicEligibilityProposal = (field, evidence, facts = state.facts) => createEligibilityProposal({
  field, facts, evidence
});

export const mergeDeterministicFacts = (output, request, evidence, facts = state.facts) => ({
  ...output,
  proposals: request.fields.map((field) => {
    const generated = output.proposals.find(proposal => proposal.field_id === field.fieldId);
    return deterministicFactProposal(field, evidence, facts)
      || deterministicEligibilityProposal(field, evidence, facts)
      || (generated?.action === 'fill' ? generated : null)
      || createPostedSalaryProposal({ field, evidence })
      || generated
      || emptyProposal(field);
  })
});

const generationCacheContext = async ({ scan, sourceSignature }) => {
  const structureSignature = await applicationStructureSignature(scan);
  const seed = generationCacheSeed({
    structureSignature,
    sourceSignature,
    generationModel: state.settings.generationModel,
    fallbackGenerationModel: state.settings.fallbackGenerationModel,
    embeddingModel: state.settings.embeddingModel
  });
  return {
    id: await cacheRecordId('generation', seed),
    structureSignature
  };
};

const loadGenerationCache = async ({ scan, sourceSignature }) => {
  const { id, structureSignature } = await generationCacheContext({ scan, sourceSignature });
  const record = await state.vault.getRecord(id);
  if (!record || record.value?.schemaVersion !== CACHE_SCHEMA_VERSION
    || record.value.generationStatus !== 'complete'
    || record.value.structureSignature !== structureSignature
    || record.value.sourceSignature !== sourceSignature
    || !proposalsMatchFieldIds(record.value.proposals, scan.fields)
    || !record.value.evidence) return null;
  return record.value;
};

const storeGenerationCache = async ({ scan, sourceSignature, evidence, proposals, retrievalMode }) => {
  if (!proposalsMatchFieldIds(proposals, scan.fields)) {
    throw new Error('Refusing to cache proposals that do not exactly match the analyzed field IDs.');
  }
  if (proposals.some(isDegradedFallbackProposal)) return false;
  const { id, structureSignature } = await generationCacheContext({ scan, sourceSignature });
  await state.vault.putRecord({
    id,
    kind: 'generation-cache',
    value: {
      schemaVersion: CACHE_SCHEMA_VERSION,
      generationStatus: 'complete',
      structureSignature,
      sourceSignature,
      evidence,
      proposals,
      retrievalMode,
      createdAt: new Date().toISOString()
    }
  });
  void trimEncryptedCaches('generation-cache', 10).catch(() => {});
  return true;
};

const proposalForRuntime = proposal => ({ ...proposal, confirmed: false });

const publishProposals = async () => {
  if (!state.activeTabId || !state.scan || state.stale) return;
  await sendPageMessage(state.activeTabId, 'PAGE_PROPOSALS_UPDATE', {
    proposals: state.proposals.map(proposalForRuntime)
  });
};

async function invalidateGeneratedAnswers() {
  abortActiveOperations();
  state.evidence = null;
  state.proposals = [];
  state.filledFieldIds.clear();
  state.batchStatus = '';
  state.sourceSignature = '';
  state.retrievalMode = '';
  state.analysisNotice = '';
  const batchReview = $('[data-batch-review]');
  if (batchReview) batchReview.checked = false;
  try {
    await publishProposals();
  } catch {}
  renderApplication();
}

const assertFreshGenerationCommit = async ({
  tabId,
  expectedScan,
  expectedStructureSignature,
  controller
}) => {
  if (!Number.isSafeInteger(tabId) || state.activeTabId !== tabId) {
    throw new DOMException('Analysis moved to another browser tab.', 'AbortError');
  }
  const response = await sendPageMessage(tabId, 'PAGE_SCAN_REQUEST');
  if (response.type !== 'PAGE_SCAN_RESULT' || !response.payload?.pageId || !Array.isArray(response.payload.fields)) {
    throw new Error('The live page scan response was incomplete.');
  }
  const [liveStructureSignature, stateStructureSignature] = await Promise.all([
    applicationStructureSignature(response.payload),
    applicationStructureSignature(state.scan)
  ]);
  const expectedScanKey = scanSnapshotKey(expectedScan);
  const common = {
    expectedScanKey,
    expectedStructureSignature,
    stale: state.stale,
    aborted: controller?.signal?.aborted
  };
  const liveMatches = canCommitGeneration({
    ...common,
    currentScan: response.payload,
    currentStructureSignature: liveStructureSignature
  });
  const stateMatches = canCommitGeneration({
    ...common,
    currentScan: state.scan,
    currentStructureSignature: stateStructureSignature
  });
  if (!liveMatches || !stateMatches) {
    state.stale = true;
    renderApplication();
    throw new DOMException('Analysis was superseded by a changed application page.', 'AbortError');
  }
};

const updateAnalysisProgress = (title, copy) => {
  const panel = $('[data-analysis-progress]');
  panel.hidden = false;
  $('[data-analysis-progress-title]').textContent = title;
  $('[data-analysis-progress-copy]').textContent = copy;
};

const abortActiveOperations = () => {
  state.analysisController?.abort();
  if (state.generationController !== state.analysisController) state.generationController?.abort();
  state.ollamaController?.abort();
  state.analysisController = null;
  state.generationController = null;
  state.ollamaController = null;
};

const recoverNarrativeExperienceProposals = async ({
  output,
  request,
  evidence,
  generationError = null,
  signal
}) => {
  const proposals = [...(output?.proposals || [])];
  const errors = [];
  let modelRecoveredCount = 0;
  let safeFallbackCount = 0;
  const allowModelRecovery = !generationError || [
    GENERATION_ERROR_CODES.MALFORMED_OUTPUT,
    GENERATION_ERROR_CODES.FAILED
  ].includes(generationError.code);

  for (const field of request.fields.filter(isNarrativeExperienceField)) {
    const proposalIndex = proposals.findIndex(proposal => proposal.field_id === field.fieldId);
    const priorDraft = proposalIndex >= 0 ? proposals[proposalIndex] : emptyProposal(field);
    if (priorDraft.action === 'fill') continue;
    const safeFallback = createBestEffortExperienceProposal({ field, evidence });
    if (!safeFallback) continue;

    let recovered = null;
    if (allowModelRecovery) {
      updateAnalysisProgress('Recovering an experience answer', 'Drafting this question separately from the page batch...');
      try {
        const generated = await orchestrateFieldRegeneration({
          client: state.ollama,
          field: { ...field, manual: false },
          priorDraft,
          feedback: {
            preset: 'more_specific',
            text: EXPERIENCE_RECOVERY_INSTRUCTION,
            maxChars: field.maxLength || null,
            mustInclude: [],
            mustAvoid: []
          },
          evidence: evidenceForModelField(evidence, field.fieldId),
          primaryModel: state.settings.generationModel,
          fallbackModel: state.settings.fallbackGenerationModel,
          signal
        });
        const candidate = regenerationProposal(generated.output, field);
        if (candidate.action === 'fill') {
          recovered = candidate;
          modelRecoveredCount += 1;
        }
      } catch (error) {
        if (signal?.aborted || error?.name === 'AbortError' || error?.code === GENERATION_ERROR_CODES.CANCELLED) throw error;
        errors.push(error);
      }
    }

    if (!recovered) {
      recovered = safeFallback;
      safeFallbackCount += 1;
    }
    if (proposalIndex >= 0) proposals[proposalIndex] = recovered;
    else proposals.push(recovered);
  }

  return {
    output: { ...output, proposals },
    modelRecoveredCount,
    safeFallbackCount,
    errors
  };
};
const analyzeActivePage = async () => {
  if (state.busy) return;
  state.busy = true;
  abortActiveOperations();
  const controller = new AbortController();
  state.analysisController = controller;
  state.generationController = controller;
  const analyzeButton = $('[data-action="scan-active-page"]');
  setButtonBusy(analyzeButton, true, 'Analyzing...');
  updateAnalysisProgress('Scanning active page', 'Injecting the isolated field runtime...');
  try {
    const scan = await scanActivePage();
    const expectedTabId = state.activeTabId;
    const expectedStructureSignature = await applicationStructureSignature(scan);
    let capturedPacket = null;
    if (state.unlocked) capturedPacket = await persistApplicationPacket({ scan });
    renderApplication();
    if (hasAccountGateContext(scan)) throw new Error(scanSupportPresentation(scan).emptyError);
    if (!scan.fields.length) {
      if (!state.unlocked && scan.job?.description) {
        throw new Error('The job description was detected. Unlock the vault, then analyze again to save it before opening the application.');
      }
      if (capturedPacket && scan.job?.description) {
        showToast('Job context captured. Open the application form, then analyze again.', 'success');
        return;
      }
      throw new Error(scanSupportPresentation(scan).emptyError);
    }
    if (!state.unlocked) throw new Error('Unlock the encrypted vault before retrieving evidence and generating answers.');
    const activeSourceCount = filterSelectedDocumentRecords(state.documents, state.selectedSourceIds, scan.pageId).length;
    const activeFactCount = state.facts.filter(record => factAppliesToApplication(record, scan.pageId)).length;
    if (!activeSourceCount && !activeFactCount) throw new Error('Select at least one source or add a verified fact before generating answers.');
    updateAnalysisProgress('Retrieving evidence', 'Building application-scoped candidate evidence...');
    const retrieval = await retrieveEvidence({ scan, signal: controller.signal });
    const completeEvidence = withEligibilityEvidence({
      evidence: retrieval.evidence,
      fields: retrieval.request.fields,
      facts: state.facts
    });
    state.evidence = completeEvidence;
    state.retrievalMode = retrieval.mode;
    state.sourceSignature = retrieval.sourceSignature;
    const cached = await loadGenerationCache({ scan, sourceSignature: retrieval.sourceSignature });
    if (cached) {
      await assertFreshGenerationCommit({
        tabId: expectedTabId,
        expectedScan: scan,
        expectedStructureSignature,
        controller
      });
      state.evidence = cached.evidence;
      state.proposals = cached.proposals;
      state.retrievalMode = cached.retrievalMode;
      await persistApplicationPacket({ scan });
      await publishProposals();
      showToast('Loaded the exact encrypted cache for this page revision.', 'success');
      return;
    }
    const modelFields = modelEligibleFields(retrieval.request.fields);
    updateAnalysisProgress(
      modelFields.length ? 'Generating grounded answers' : 'Applying verified facts',
      modelFields.length ? `Using ${state.settings.generationModel} with schema validation...` : 'No consequential fields require local-model drafting.'
    );
    let output;
    let generationError = null;
    try {
      let generatedOutput = { page_id: retrieval.request.pageId, proposals: [] };
      if (modelFields.length) {
        const generated = await orchestratePageGeneration({
          client: state.ollama,
          request: { ...retrieval.request, fields: modelFields },
          evidence: evidenceForModelFields(retrieval.evidence, modelFields.map(field => field.fieldId)),
          primaryModel: state.settings.generationModel,
          fallbackModel: state.settings.fallbackGenerationModel,
          signal: controller.signal,
          onStatus: update => {
            if (update.code === GENERATION_STATUS_CODES.STRUCTURED_RETRY) {
              updateAnalysisProgress('Repairing model output', 'Retrying once with the same frozen evidence and schema...');
            } else if (update.code === GENERATION_STATUS_CODES.MODEL_FALLBACK) {
              updateAnalysisProgress('Using fallback model', 'Trying installed local model ' + update.model + '...');
            }
          }
        });
        generatedOutput = generated.output;
      }
      output = mergeDeterministicFacts(generatedOutput, retrieval.request, completeEvidence);
    } catch (error) {
      if (controller.signal.aborted || error?.name === 'AbortError' || error?.code === GENERATION_ERROR_CODES.CANCELLED) throw error;
      generationError = error;
      const deterministic = retrieval.request.fields.map(field => deterministicFactProposal(field, completeEvidence)
        || deterministicEligibilityProposal(field, completeEvidence));
      output = {
        page_id: retrieval.request.pageId,
        proposals: retrieval.request.fields.map((field, index) => deterministic[index]
          || createPostedSalaryProposal({ field, evidence: completeEvidence })
          || emptyProposal(field))
      };
    }

    const recovery = await recoverNarrativeExperienceProposals({
      output,
      request: retrieval.request,
      evidence: completeEvidence,
      generationError,
      signal: controller.signal
    });
    output = recovery.output;
    const recoveredCount = recovery.modelRecoveredCount + recovery.safeFallbackCount;
    if (generationError) {
      state.analysisNotice = 'The page-wide structured response failed. Experience questions were recovered where candidate evidence was available, and degraded placeholders were not cached. ' + readableError(generationError);
    } else if (recoveredCount) {
      state.analysisNotice = recoveredCount + ' experience answer' + (recoveredCount === 1 ? '' : 's') + ' required isolated evidence-grounded recovery. Review each low-confidence draft and add clarification if needed.';
    } else {
      state.analysisNotice = '';
    }

    await assertFreshGenerationCommit({
      tabId: expectedTabId,
      expectedScan: scan,
      expectedStructureSignature,
      controller
    });
    state.proposals = output.proposals;
    await persistApplicationPacket({ scan });
    if (!generationError) {
      await storeGenerationCache({
        scan,
        sourceSignature: retrieval.sourceSignature,
        evidence: completeEvidence,
        proposals: state.proposals,
        retrievalMode: retrieval.mode
      });
    }
    await publishProposals();
    showToast(
      generationError
        ? readableError(generationError) + (recoveredCount ? ' Recovered ' + recoveredCount + ' experience draft(s); degraded results were not cached.' : ' Degraded results were not cached.')
        : 'Prepared ' + state.proposals.length + ' reviewed proposals.',
      generationError ? 'error' : 'success'
    );
  } catch (error) {
    if (error?.name !== 'AbortError' && error?.code !== 'GENERATION_CANCELLED') showToast(readableError(error), 'error');
  } finally {
    if (state.analysisController === controller) state.analysisController = null;
    if (state.generationController === controller) state.generationController = null;
    state.busy = false;
    $('[data-analysis-progress]').hidden = true;
    setButtonBusy(analyzeButton, false);
    renderApplication();
  }
};

export const evidenceConfidenceForProposal = ({ field, proposal, cards = [] } = {}) => {
  if (!proposal || proposal.action !== 'fill' || proposal.confidence === 'needs_input') {
    return { label: 'Needs input', tone: 'risk', detail: 'The supplied evidence does not support a fillable answer.' };
  }
  if (field?.riskClass === 'F1_VERIFIED' && proposal.confidence === 'high') {
    return { label: 'Verified', tone: 'success', detail: 'Exact value from a user-verified profile fact.' };
  }
  const roles = new Set(cards.map(card => card.sourceRole));
  const documentedGap = proposalDisclosesEvidenceGap(proposal);
  if (documentedGap) {
    return { label: 'Low confidence', tone: 'review', detail: 'Direct tenure is not documented; the draft uses cited adjacent experience and requests clarification.' };
  }
  if (roles.has(SOURCE_ROLES.JOB_REQUIREMENT)
    && !roles.has(SOURCE_ROLES.CANDIDATE_EVIDENCE)
    && !roles.has(SOURCE_ROLES.USER_VERIFIED)) {
    return { label: 'Partial evidence', tone: 'review', detail: 'Grounded context is available, but this answer needs careful review.' };
  }
  if (roles.has(SOURCE_ROLES.CANDIDATE_EVIDENCE) || roles.has(SOURCE_ROLES.USER_VERIFIED)) {
    return { label: 'Strong evidence', tone: 'success', detail: 'The draft is grounded in supplied candidate evidence and still requires review.' };
  }
  return { label: 'Partial evidence', tone: 'review', detail: 'Only limited grounded context supports this draft.' };
};

const FIELD_TYPE_LABELS = Object.freeze({
  text: 'short answer',
  textarea: 'long answer',
  email: 'email',
  tel: 'phone',
  url: 'link',
  number: 'number',
  date: 'date',
  'select-one': 'dropdown',
  radio: 'choice',
  checkbox: 'checkbox',
  contenteditable: 'long answer'
});

export const fieldTypeLabel = value => FIELD_TYPE_LABELS[String(value || '').toLocaleLowerCase('en-US')] || 'custom field';
export const fieldMetaLabel = field => [field?.required ? 'Required' : 'Optional', fieldTypeLabel(field?.type)].join(' - ');

const sourceName = documentId => {
  if (String(documentId || '').startsWith(LIVE_JOB_DOCUMENT_PREFIX)) return 'Live job posting';
  const fact = state.facts.find(record => record.id === documentId);
  if (fact) return fact.value.label;
  const documentRecord = state.documents.find(record => record.value?.document?.id === documentId || record.id === documentId);
  return documentRecord?.value?.document?.filename || documentId;
};

const sourceAttributionUrl = documentId => {
  const record = state.documents.find(candidate => candidate.value?.document?.id === documentId || candidate.id === documentId);
  const rawUrl = String(documentId || '').startsWith(LIVE_JOB_DOCUMENT_PREFIX)
    ? state.scan?.job?.jobUrl
    : record?.value?.document?.sourceUrl;
  if (!rawUrl) return null;
  try {
    const url = new URL(rawUrl);
    return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null;
  } catch {
    return null;
  }
};

const locatorLabel = locator => {
  if (!locator || typeof locator !== 'object') return '';
  const parts = [];
  if (locator.section) parts.push(locator.section);
  if (locator.pageStart) parts.push(locator.pageEnd && locator.pageEnd !== locator.pageStart
    ? `pages ${locator.pageStart}-${locator.pageEnd}`
    : `page ${locator.pageStart}`);
  if (locator.paragraphStart) parts.push(`paragraph ${locator.paragraphStart}`);
  return parts.join(' · ');
};

const citationCardsForProposal = (proposal) => {
  if (!state.evidence || !proposal.citation_ids?.length) return [];
  try {
    return resolveCitationCards({
      citationIds: proposal.citation_ids,
      fieldId: proposal.field_id,
      evidence: state.evidence
    });
  } catch {
    return [];
  }
};

const renderCitationDetails = (proposal) => {
  const cards = citationCardsForProposal(proposal);
  if (!cards.length) return null;
  const details = createElement('details', { className: 'citations' });
  details.append(createElement('summary', { text: `${cards.length} exact ${cards.length === 1 ? 'citation' : 'citations'}` }));
  cards.forEach((card) => {
    const article = createElement('article', { className: 'citation-card' });
    article.append(createElement('strong', { text: `${card.citationId} · ${sourceName(card.documentId)}` }));
    const attributionUrl = sourceAttributionUrl(card.documentId);
    if (attributionUrl) article.append(createElement('a', {
      className: 'citation-locator',
      text: attributionUrl,
      attributes: { href: attributionUrl, target: '_blank', rel: 'noopener noreferrer' }
    }));
    const locator = locatorLabel(card.locator);
    if (locator) article.append(createElement('span', { className: 'citation-locator', text: locator }));
    article.append(createElement('blockquote', { text: card.text }));
    details.append(article);
  });
  return details;
};

const renderAnswerCard = (field, proposal) => {
  const card = createElement('details', {
    className: 'answer-card',
    dataset: {
      fieldCard: field.fieldId,
      selected: state.selectedFieldId === field.fieldId
    }
  });
  const header = createElement('summary', { className: 'answer-header' });
  const label = createElement('div', { className: 'answer-label' });
  card.open = state.selectedFieldId === field.fieldId;
  label.append(
    createElement('h3', { text: field.label }),
    createElement('span', { text: fieldMetaLabel(field) })
  );
  const citationCards = citationCardsForProposal(proposal);
  const confidence = evidenceConfidenceForProposal({
    field,
    proposal,
    cards: citationCards
  });
  const clarificationQuestion = clarificationQuestionForProposal({
    field,
    proposal,
    cards: citationCards
  });
  const badges = createElement('div', { className: 'badge-row' });
  badges.append(createElement('span', {
    className: 'badge badge-' + confidence.tone,
    text: confidence.label,
    attributes: { title: confidence.detail }
  }));
  if (field.riskClass === 'F2_REVIEW') {
    badges.append(createElement('span', { className: 'badge badge-review', text: 'Review' }));
  }
  if (isCopyOnlyField(field)) {
    badges.append(createElement('span', { className: 'badge badge-review', text: 'Copy only' }));
  }
  header.append(label, badges);

  const body = createElement('div', { className: 'answer-body' });
  const answer = proposal.action === 'ask_user'
    ? proposal.abstain_reason || 'Enter this answer yourself.'
    : proposalAnswerValue(proposal);
  const answerText = Array.isArray(answer) ? answer.join(', ') : String(answer);
  body.append(createElement('pre', { className: 'answer-value', text: answerText || 'No proposed value.' }));
  if (proposal.short_rationale) body.append(createElement('p', { className: 'answer-rationale', text: proposal.short_rationale }));
  if (clarificationQuestion) {
    const clarification = createElement('div', { className: 'notice notice-warning clarification-callout' });
    clarification.append(
      createElement('strong', { text: 'Clarification can strengthen this draft' }),
      createElement('span', { text: clarificationQuestion })
    );
    body.append(clarification);
  }
  const citations = renderCitationDetails(proposal);
  if (citations) body.append(citations);

  if (isCopyOnlyField(field) && proposal.action === 'fill') {
    const copyOnlyNotice = createElement('div', { className: 'notice notice-warning copy-only-notice' });
    copyOnlyNotice.append(
      createElement('strong', { text: 'This custom control is copy-only' }),
      createElement('span', { text: 'Review the cited answer, copy it, and enter it in the application manually. Bulk fill will skip this field.' })
    );
    body.append(copyOnlyNotice);
  }

  // A deliberate consequential-field button click is the field-level approval.

  const actions = createElement('div', { className: 'answer-actions' });
  const canFill = proposal.action === 'fill' && !isCopyOnlyField(field) && !state.scan?.captchaPresent
    && !hasAccountGateContext(state.scan) && !state.stale;
  actions.append(
    createElement('button', {
      className: 'button button-primary',
      text: field.riskClass === 'F2_REVIEW' ? 'Approve & fill' : 'Fill',
      attributes: { type: 'button', ...(canFill ? {} : { disabled: '' }), ...(proposal.action !== 'fill' || isCopyOnlyField(field) ? { hidden: '' } : {}) },
      dataset: { fillField: field.fieldId }
    }),
    createElement('button', {
      className: 'button button-secondary',
      text: 'Copy',
      attributes: { type: 'button', ...(proposal.action === 'fill' ? {} : { hidden: '' }) },
      dataset: { copyAnswer: field.fieldId }
    }),
    createElement('button', {
      className: 'button button-quiet',
      text: proposal.action === 'fill' ? 'Revise' : 'Try again',
      attributes: { type: 'button', 'aria-expanded': 'false', ...(!state.evidence || field.riskClass === 'F1_VERIFIED' ? { hidden: '' } : {}) },
      dataset: { toggleFeedback: field.fieldId }
    })
  );
  body.append(actions);

  const feedback = createElement('div', { className: 'feedback-panel', dataset: { feedbackPanel: field.fieldId } });
  feedback.hidden = true;
  const feedbackSelect = createElement('select', { dataset: { feedbackPreset: field.fieldId } });
  [
    ['', 'No additional feedback'],
    ['shorter', 'Make it shorter'],
    ['more_specific', 'Make it more specific'],
    ['tone', 'Adjust the tone'],
    ['other', 'Other instruction']
  ].forEach(([value, text]) => feedbackSelect.append(createElement('option', { text, attributes: { value } })));
  const feedbackText = createElement('textarea', {
    attributes: {
      rows: '4',
      maxlength: '2000',
      placeholder: clarificationQuestion
        ? 'Optional factual clarification: organization or project, dates, responsibilities, outcome, or explicitly confirm that you have no direct experience.'
        : 'Optional instruction grounded in the same frozen evidence. Leave blank to revise the current draft...',
      spellcheck: 'false',
      autocomplete: 'off'
    },
    dataset: { feedbackText: field.fieldId }
  });
  const regenerate = createElement('button', {
    className: 'button button-secondary',
    text: 'Regenerate answer',
    attributes: { type: 'button' },
    dataset: { regenerateField: field.fieldId }
  });
  feedback.append(feedbackSelect, feedbackText);
  if (clarificationQuestion) {
    feedback.append(createElement('p', {
      className: 'field-help',
      text: 'Optional. Nonblank factual context is encrypted and saved as user-verified evidence for this application. Leave blank to save nothing and regenerate from the current draft, field request, and frozen evidence.'
    }));
  } else {
    feedback.append(createElement('p', {
      className: 'field-help',
      text: 'Optional. Leave blank to regenerate from the current draft, field request, and frozen evidence.'
    }));
  }
  feedback.append(regenerate);
  card.append(header, body, feedback);
  return card;
};
const populateTrackerForm = () => {
  const form = $('[data-tracker-form]');
  if (!form || !state.scan) return;
  const key = scanSnapshotKey(state.scan);
  if (form.dataset.scanKey === key) return;
  form.dataset.scanKey = key;
  form.elements.company.value = state.scan.job?.company || '';
  form.elements.title.value = state.scan.job?.title || '';
  form.elements.jobUrl.value = state.scan.job?.jobUrl || '';
  form.elements.location.value = state.scan.job?.location || '';
  form.elements.source.value = state.scan.job?.source || state.scan.adapter || '';
  form.elements.postingDate.value = '';
  form.elements.appliedDate.value = today();
  form.elements.status.value = 'Applied';
  form.elements.tags.value = '';
  // No extra checkbox is needed; the handoff button is the explicit approval.
};

export const applicationPreflightSummary = ({ scan, proposals = [] } = {}) => {
  const fields = Array.isArray(scan?.fields) ? scan.fields : [];
  const proposalByField = new Map((proposals || []).map(proposal => [proposal?.field_id, proposal]));
  const counts = {
    verifiedReady: 0,
    reviewReady: 0,
    copyOnly: 0,
    needsInput: 0,
    needsClarification: 0,
    manual: 0,
    ready: 0
  };
  for (const field of fields) {
    if (field?.manual || !['F1_VERIFIED', 'F2_REVIEW'].includes(field?.riskClass)) {
      counts.manual += 1;
      continue;
    }
    const proposal = proposalByField.get(field.fieldId);
    if (!proposal) {
      counts.needsInput += 1;
      continue;
    }
    const cards = typeof state !== 'undefined' && state.evidence
      ? citationCardsForProposal(proposal)
      : [];
    if (clarificationQuestionForProposal({ field, proposal, cards })) counts.needsClarification += 1;
    if (proposal.action !== 'fill') {
      counts.needsInput += 1;
      continue;
    }
    if (isCopyOnlyField(field)) {
      counts.copyOnly += 1;
      continue;
    }
    if (field.riskClass === 'F1_VERIFIED') counts.verifiedReady += 1;
    else counts.reviewReady += 1;
  }
  const discovery = normalizeScanDiscovery(scan);
  const exclusions = scan?.exclusionCounts || discovery.exclusionCounts || {};
  counts.manual += Math.max(0, Number(discovery.unsupportedCount) || 0)
    + Math.max(0, Number(exclusions.F3_SENSITIVE) || 0)
    + Math.max(0, Number(exclusions.F4_CONSENT) || 0);
  counts.ready = counts.verifiedReady + counts.reviewReady;
  return counts;
};

const proposalNeedsAttention = (field, proposal) => proposal?.action !== 'fill'
  || isCopyOnlyField(field)
  || Boolean(clarificationQuestionForProposal({
    field,
    proposal,
    cards: citationCardsForProposal(proposal)
  }));

const renderClarificationBatch = () => {
  const details = $('[data-clarification-batch]');
  const summary = $('[data-clarification-summary]');
  const list = $('[data-batch-clarification-list]');
  if (!details || !summary || !list) return;
  const existingValues = new Map($$('[data-batch-clarification-field]').map(input => [
    input.dataset.batchClarificationField,
    input.value
  ]));
  const fieldsById = new Map((state.scan?.fields || []).map(field => [field.fieldId, field]));
  const items = state.proposals.map(proposal => {
    const field = fieldsById.get(proposal.field_id);
    if (!field) return null;
    const question = clarificationQuestionForProposal({
      field,
      proposal,
      cards: citationCardsForProposal(proposal)
    });
    return question ? { field, proposal, question } : null;
  }).filter(Boolean);
  details.hidden = items.length < 1;
  summary.textContent = items.length === 1
    ? 'Clarify 1 answer once'
    : `Clarify ${items.length} answers once`;
  list.replaceChildren();
  for (const item of items) {
    const label = createElement('label');
    label.append(
      createElement('span', { text: item.field.label }),
      createElement('small', { text: item.question })
    );
    const input = createElement('textarea', {
      attributes: {
        rows: '3',
        maxlength: '2000',
        placeholder: 'Add only facts or preferences you want cited in the regenerated answer.',
        autocomplete: 'off',
        spellcheck: 'false'
      },
      dataset: { batchClarificationField: item.field.fieldId }
    });
    input.value = existingValues.get(item.field.fieldId) || '';
    label.append(input);
    list.append(label);
  }
};

const renderBatchFillControls = () => {
  const container = $('[data-batch-fill]');
  const reviewCheckbox = $('[data-batch-review]');
  const fillAllButton = $('[data-action="fill-all-ready"]');
  const verifiedButton = $('[data-action="fill-verified-only"]');
  const filterButton = $('[data-action="toggle-unresolved"]');
  const status = $('[data-batch-fill-status]');
  if (!container || !reviewCheckbox || !fillAllButton || !verifiedButton || !status) return;
  const fields = state.scan?.fields || [];
  const preflight = applicationPreflightSummary({ scan: state.scan, proposals: state.proposals });
  const attentionCount = preflight.needsInput + preflight.copyOnly + preflight.manual;
  const reviewedPlan = planBulkFill({
    fields,
    proposals: state.proposals,
    reviewedConsequential: true,
    alreadyFilledIds: state.filledFieldIds
  });
  const verifiedPlan = planBulkFill({
    fields,
    proposals: state.proposals,
    verifiedOnly: true,
    alreadyFilledIds: state.filledFieldIds
  });
  const hasConsequential = reviewedPlan.consequentialCount > 0;
  const reviewCopy = $('[data-batch-review-copy]');
  if (reviewCopy) reviewCopy.textContent = `I reviewed ${reviewedPlan.consequentialCount} drafted answer${reviewedPlan.consequentialCount === 1 ? '' : 's'} and the sources.`;

  const blocked = state.bulkBusy || state.stale || state.scan?.captchaPresent || hasAccountGateContext(state.scan);
  container.hidden = state.proposals.length < 1;
  reviewCheckbox.closest('.batch-review').hidden = !hasConsequential;
  reviewCheckbox.disabled = state.bulkBusy || !hasConsequential;
  fillAllButton.disabled = blocked
    || reviewedPlan.items.length < 1
    || (hasConsequential && !reviewCheckbox.checked);
  verifiedButton.disabled = blocked || verifiedPlan.items.length < 1;
  fillAllButton.textContent = reviewedPlan.items.length
    ? 'Fill ' + reviewedPlan.items.length + (reviewedPlan.items.length === 1 ? ' field' : ' fields')
    : 'Fill ready fields';
  verifiedButton.textContent = verifiedPlan.items.length
    ? 'Verified only (' + verifiedPlan.items.length + ')'
    : 'Fill verified';
  if (filterButton) {
    filterButton.hidden = attentionCount < 1;
    filterButton.textContent = state.answerFilter === 'attention' ? 'Show all' : `Review issues (${attentionCount})`;
  }
  const heading = $('[data-preflight-heading]');
  const summary = $('[data-preflight-summary]');
  if (heading) heading.textContent = preflight.ready
    ? `${preflight.ready} ready`
    : 'Needs your input';
  if (summary) {
    const remaining = preflight.needsInput + preflight.copyOnly + preflight.manual;
    summary.textContent = remaining
      ? `${preflight.ready} ready. ${remaining} ${remaining === 1 ? 'item needs' : 'items need'} attention.`
      : 'Everything recognized is prepared. Nothing is submitted automatically.';
  }
  const stats = $('[data-preflight-stats]');
  if (stats) {
    stats.replaceChildren();
    [
      ['Ready', preflight.ready, 'success'],
      ['Needs attention', attentionCount, 'risk']
    ].filter(([, count]) => count > 0).forEach(([label, count, tone]) => {
      const item = createElement('span', { className: 'preflight-stat', dataset: { tone } });
      item.append(createElement('strong', { text: String(count) }), createElement('span', { text: label }));
      stats.append(item);
    });
  }
  renderClarificationBatch();
  if (state.batchStatus) {
    status.textContent = state.batchStatus;
    return;
  }
  status.textContent = '';
};
const renderJobContext = () => {
  const heading = $('[data-job-context-heading]');
  const copy = $('[data-job-context-copy]');
  const badge = $('[data-job-context-badge]');
  const preview = $('[data-job-context-preview]');
  const form = $('[data-job-context-form]');
  const pickerWrap = $('[data-saved-context-picker]');
  const picker = $('[data-job-context-picker]');
  if (!heading || !copy || !badge || !preview || !form || !pickerWrap || !picker) return;
  const packet = packetRecordByApplicationId(state.activeApplicationPacketId)
    || packetMatchForScan(state.scan)?.record
    || null;
  const context = latestPacketContext(packet);
  const description = String(state.scan?.job?.description || context?.content || '').trim();
  const contextOrigin = context?.source?.origin || (description ? APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE : '');
  if (description) {
    const originLabel = contextOrigin === APPLICATION_CONTEXT_ORIGINS.MANUAL
      ? 'Manual override'
      : context ? 'Encrypted capture' : 'Current page';
    heading.textContent = 'Job description captured';
    copy.textContent = `${description.length.toLocaleString('en-US')} characters - ${originLabel}`;
    badge.textContent = contextOrigin === APPLICATION_CONTEXT_ORIGINS.MANUAL ? 'Manual' : 'Automatic';
    badge.className = 'badge badge-success';
    preview.textContent = description;
  } else {
    heading.textContent = 'No job description captured';
    copy.textContent = 'Capture it automatically, use selected page text, or paste a correction.';
    badge.textContent = 'Missing';
    badge.className = 'badge badge-review';
    preview.textContent = 'No description is attached to this application yet.';
  }

  const contextKey = [
    packet?.value?.applicationId || '',
    context?.contextId || '',
    state.scan?.pageId || '',
    description.length
  ].join('|');
  if (form.dataset.contextKey !== contextKey) {
    form.dataset.contextKey = contextKey;
    form.elements.company.value = state.scan?.job?.company || context?.job?.company || '';
    form.elements.title.value = state.scan?.job?.title || context?.job?.title || '';
    form.elements.jobUrl.value = state.scan?.job?.jobUrl || context?.job?.jobUrl || context?.source?.url || '';
    form.elements.location.value = state.scan?.job?.location || context?.job?.location || '';
    form.elements.description.value = description;
  }

  const selectedApplicationId = packet?.value?.applicationId || state.activeApplicationPacketId;
  picker.replaceChildren();
  state.applicationPackets
    .slice()
    .sort((left, right) => String(right.value?.updatedAt || right.updatedAt || '')
      .localeCompare(String(left.value?.updatedAt || left.updatedAt || '')))
    .forEach((record) => {
      const label = [record.value?.identity?.title, record.value?.identity?.company]
        .filter(Boolean).join(' at ') || 'Captured application';
      picker.append(createElement('option', {
        text: label,
        attributes: { value: record.value.applicationId }
      }));
    });
  pickerWrap.hidden = state.applicationPackets.length < 1;
  if ([...picker.options].some(option => option.value === selectedApplicationId)) {
    picker.value = selectedApplicationId;
  }
};

function renderApplication() {
  const answerList = $('[data-answer-list]');
  if (!answerList) return;
  $('[data-captcha-notice]').hidden = !state.scan?.captchaPresent;
  $('[data-stale-notice]').hidden = !state.stale;
  const generationNotice = $('[data-generation-notice]');
  if (generationNotice) generationNotice.hidden = !state.analysisNotice;
  const generationNoticeCopy = $('[data-generation-notice-copy]');
  if (generationNoticeCopy) generationNoticeCopy.textContent = state.analysisNotice;
  const supportPresentation = state.scan ? scanSupportPresentation(state.scan) : null;
  const discoveryNotice = $('[data-discovery-notice]');
  if (discoveryNotice) discoveryNotice.hidden = !supportPresentation?.showNotice;
  const discoveryNoticeTitle = $('[data-discovery-notice-title]');
  if (discoveryNoticeTitle) discoveryNoticeTitle.textContent = supportPresentation?.title || '';
  const discoveryNoticeCopy = $('[data-discovery-notice-copy]');
  if (discoveryNoticeCopy) discoveryNoticeCopy.textContent = supportPresentation?.copy || '';
  const analyzeButton = $('[data-action="scan-active-page"]');
  if (analyzeButton && !state.busy) {
    analyzeButton.textContent = state.scan ? 'Refresh answers' : 'Analyze current page';
  }
  $('[data-job-summary]').hidden = !state.scan;
  $('[data-field-workspace]').hidden = !state.scan?.fields?.length;
  $('[data-tracker-card]').hidden = !state.scan?.fields?.length || !state.unlocked;
  const retrievalBadge = $('[data-retrieval-mode]');
  retrievalBadge.hidden = !state.retrievalMode;
  retrievalBadge.textContent = state.retrievalMode === 'hybrid' ? 'Local embeddings + lexical' : 'Lexical fallback';
  if (state.scan) {
    const discovery = normalizeScanDiscovery(state.scan);
    const adapterLabel = state.scan.adapter || 'generic';
    $('[data-adapter-badge]').textContent = discovery.mode === 'standard'
      ? adapterLabel
      : `${adapterLabel} - ${discovery.mode === 'free_format' ? 'free format' : 'limited'}`;
    $('[data-page-status]').textContent = supportPresentation.status;
    $('[data-field-count]').textContent = `${state.scan.fields.length} ${state.scan.fields.length === 1 ? 'field' : 'fields'}`;
    $('[data-job-heading]').textContent = [state.scan.job?.title, state.scan.job?.company].filter(Boolean).join(' at ') || 'Detected application';
    const metadata = $('[data-job-metadata]');
    metadata.replaceChildren();
    [['Company', state.scan.job?.company], ['Title', state.scan.job?.title], ['Location', state.scan.job?.location], ['Source', state.scan.job?.source], ['Page revision', String(state.scan.domRevision)]].forEach(([term, value]) => {
      if (!value) return;
      metadata.append(createElement('dt', { text: term }), createElement('dd', { text: value }));
    });
    renderJobContext();
    populateTrackerForm();
  } else {
    $('[data-adapter-badge]').textContent = 'Not scanned';
    $('[data-page-status]').textContent = 'Open a job application form, then analyze the active page. Nothing is submitted automatically.';
  }
  answerList.replaceChildren();
  const fieldsById = new Map((state.scan?.fields || []).map(field => [field.fieldId, field]));
  let visibleAnswerCount = 0;
  state.proposals.forEach((proposal) => {
    const field = fieldsById.get(proposal.field_id);
    if (!field) return;
    const card = renderAnswerCard(field, proposal);
    const filtered = state.answerFilter === 'attention'
      && state.selectedFieldId !== field.fieldId
      && !proposalNeedsAttention(field, proposal);
    card.dataset.filtered = String(filtered);
    if (!filtered) visibleAnswerCount += 1;
    answerList.append(card);
  });
  $('[data-answer-empty]').hidden = visibleAnswerCount > 0;
  renderBatchFillControls();
  renderTrackerDocumentOptions();
  if (state.selectedFieldId) {
    requestAnimationFrame(() => $('[data-field-card][data-selected="true"]')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
  }
};

const fieldAndProposal = (fieldId) => ({
  field: state.scan?.fields.find(candidate => candidate.fieldId === fieldId),
  proposal: state.proposals.find(candidate => candidate.field_id === fieldId)
});

const verifyFreshField = async (field) => {
  if (!state.activeTabId || !state.scan) throw new Error('Analyze the active page first.');
  const response = await sendPageMessage(state.activeTabId, 'PAGE_SCAN_REQUEST');
  if (response.type !== 'PAGE_SCAN_RESULT' || !response.payload?.pageId || !Array.isArray(response.payload.fields)) {
    throw new Error('The live page scan response was incomplete.');
  }
  const fresh = response.payload;
  if (!canFillFieldAcrossRevision({
    analyzedScan: state.scan,
    freshScan: fresh,
    field
  })) {
    state.stale = true;
    renderApplication();
    throw new Error('This question changed after analysis. Rescan before filling it.');
  }
  return fresh;
};

const requestFieldFill = async ({
  field,
  proposal,
  confirmed,
  skipIfPopulated = false
}) => {
  if (hasAccountGateContext(state.scan)) {
    throw new Error('Complete the account step manually, then rescan before filling.');
  }
  if (isCopyOnlyField(field)) {
    throw new Error('This custom application control is copy-only. Copy the reviewed answer and enter it manually.');
  }
  await verifyFreshField(field);
  const response = await sendPageMessage(state.activeTabId, 'FIELD_FILL_REQUEST', {
    fieldId: field.fieldId,
    fingerprint: field.fingerprint,
    value: runtimeAnswerValue(proposal, field),
    confirmed,
    skipIfPopulated
  });
  if (response.type !== 'FIELD_FILL_RESULT') {
    throw new Error('The page runtime did not confirm the fill attempt.');
  }
  return response.payload || {};
};

const persistApprovedAnswerItems = async (items = []) => {
  if (!items.length || !state.unlocked || !state.scan) return;
  let packet = packetRecordByApplicationId(state.activeApplicationPacketId)
    || packetMatchForScan(state.scan)?.record
    || await persistApplicationPacket({ scan: state.scan });
  if (!packet) return;
  const applicationId = packet.value.applicationId;
  const contextId = packet.value.lastContextId || '';
  const approvedAnswers = await Promise.all(items.map(item => createApprovedAnswerSnapshot({
    applicationId,
    field: item.field,
    proposal: item.proposal,
    approved: true,
    sourceSignature: state.sourceSignature,
    contextId
  })));
  packet = await persistApplicationPacket({ scan: state.scan, approvedAnswers });
  return packet;
};

const persistClarificationRecordSnapshot = async ({ field, record, text } = {}) => {
  let packet = packetRecordByApplicationId(state.activeApplicationPacketId)
    || packetMatchForScan(state.scan)?.record
    || await persistApplicationPacket({ scan: state.scan });
  if (!packet) return;
  const clarification = await createClarificationSnapshot({
    applicationId: packet.value.applicationId,
    field,
    text,
    verified: true,
    evidenceRecordId: record.id,
    contextId: packet.value.lastContextId || ''
  });
  await persistApplicationPacket({ scan: state.scan, clarifications: [clarification] });
};

const fillReadyFields = async ({ verifiedOnly = false } = {}) => {
  if (state.bulkBusy) return;
  if (hasAccountGateContext(state.scan)) throw new Error('Complete the account step manually, then rescan before filling.');
  if (state.scan?.captchaPresent) throw new Error('Interactive CAPTCHA challenge detected - filling paused. Complete it, then rescan.');
  const reviewedConsequential = Boolean($('[data-batch-review]')?.checked);
  const plan = planBulkFill({
    fields: state.scan?.fields || [],
    proposals: state.proposals,
    reviewedConsequential,
    verifiedOnly,
    alreadyFilledIds: state.filledFieldIds
  });
  if (!plan.items.length) {
    throw new Error(verifiedOnly
      ? 'No unfilled verified answers are ready.'
      : 'Review the consequential drafts and sources before filling all ready answers.');
  }

  state.bulkBusy = true;
  try {
    const approvedItems = [];
    const result = await executeBulkFillItems({
      items: plan.items,
      onProgress: (item, index, count) => {
        state.batchStatus = 'Checking field ' + (index + 1) + ' of ' + count + '...';
        renderBatchFillControls();
      },
      requestFill: item => requestFieldFill({
        field: item.field,
        proposal: item.proposal,
        confirmed: item.confirmed,
        skipIfPopulated: true
      }),
      onFilled: item => {
        state.filledFieldIds.add(item.field.fieldId);
        approvedItems.push(item);
      },
      onOccupied: item => state.filledFieldIds.add(item.field.fieldId)
    });
    if (approvedItems.length) {
      try {
        await persistApprovedAnswerItems(approvedItems);
      } catch (error) {
        state.analysisNotice = 'Fields were filled, but their encrypted application-packet snapshot could not be updated. ' + readableError(error);
      }
    }
    const summary = [
      'Filled ' + result.filled,
      result.occupied ? 'kept ' + result.occupied + ' existing' : '',
      result.copyOnly ? result.copyOnly + ' copy-only' : '',
      result.stopped ? 'stopped: ' + result.stopped : '',
      state.stale && !result.stopped ? 'form revealed or changed questions; analyze those questions next' : ''
    ].filter(Boolean).join(' | ');
    state.batchStatus = summary + '.';
    showToast(state.batchStatus, bulkFillResultTone(result));
  } finally {
    state.bulkBusy = false;
    renderApplication();
  }
};
const fillField = async (fieldId, button) => {
  const { field, proposal } = fieldAndProposal(fieldId);
  if (!field || !proposal || proposal.action !== 'fill') throw new Error('No fillable proposal is available for this field.');
  if (hasAccountGateContext(state.scan)) throw new Error('Complete the account step manually, then rescan before filling.');
  if (isCopyOnlyField(field)) throw new Error('This custom application control is copy-only. Copy the reviewed answer and enter it manually.');
  if (state.scan?.captchaPresent) throw new Error('Interactive CAPTCHA challenge detected—filling paused. Complete it, then rescan.');
  // Clicking the field-specific approval button is the explicit confirmation for this exact draft.
  const confirmed = explicitFieldFillConfirmation(field);
  setButtonBusy(button, true, 'Checking...');
  try {
    await verifyFreshField(field);
    button.textContent = 'Filling...';
    const response = await sendPageMessage(state.activeTabId, 'FIELD_FILL_REQUEST', {
      fieldId,
      fingerprint: field.fingerprint,
      value: runtimeAnswerValue(proposal, field),
      confirmed
    });
    if (response.type !== 'FIELD_FILL_RESULT') throw new Error('The page runtime did not confirm the fill attempt.');
    if (!response.payload?.verified) {
      throw new Error(`The page rejected controlled filling (${response.payload?.reason || 'copy only'}). Copy the reviewed answer instead.`);
    }
    state.filledFieldIds.add(fieldId);
    state.batchStatus = '';
    try {
      await persistApprovedAnswerItems([{ field, proposal }]);
    } catch (error) {
      state.analysisNotice = 'The field was filled, but its encrypted application-packet snapshot could not be updated. ' + readableError(error);
    }
    renderBatchFillControls();
    showToast('Field filled and verified after the page update.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const copyProposal = async (proposal) => {
  const value = proposalAnswerValue(proposal);
  const text = Array.isArray(value) ? value.join(', ') : typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
  await navigator.clipboard.writeText(text);
  showToast('Reviewed answer copied.', 'success');
};

export const isDegradedFallbackProposal = proposal => proposal?.action === 'ask_user'
  && proposal?.confidence === 'needs_input'
  && proposal?.short_rationale === 'No verified answer is available.'
  && proposal?.abstain_reason === 'Review this field and enter the answer yourself.';

export const appendFieldClarificationEvidence = async ({ evidence, field, record, applicationId } = {}) => {
  if (!evidence?.citations || !evidence?.byField || !field?.fieldId || !record?.id) {
    throw new Error('Existing field evidence and an encrypted clarification record are required.');
  }
  if (record.value?.applicationId !== applicationId || record.value?.fieldId !== field.fieldId) {
    throw new Error('Clarification scope does not match the active application field.');
  }
  const replacedIds = new Set(evidence.citations
    .filter(citation => citation.documentId === record.id)
    .map(citation => citation.citationId));
  const retainedCitations = evidence.citations.filter(citation => !replacedIds.has(citation.citationId));
  const existingIds = new Set(retainedCitations.map(citation => citation.citationId));
  const currentFieldIds = [...new Set(evidence.byField[field.fieldId] || [])]
    .filter(citationId => !replacedIds.has(citationId));
  const citationById = new Map(retainedCitations.map(citation => [citation.citationId, citation]));
  while (currentFieldIds.length >= VALIDATION_LIMITS.maxCitationsPerProposal) {
    const nonVerifiedIndex = currentFieldIds.findLastIndex(citationId => (
      citationById.get(citationId)?.sourceRole !== SOURCE_ROLES.USER_VERIFIED
    ));
    currentFieldIds.splice(nonVerifiedIndex >= 0 ? nonVerifiedIndex : currentFieldIds.length - 1, 1);
  }
  const availableSlots = Math.max(0, VALIDATION_LIMITS.maxCitationsPerProposal - currentFieldIds.length);
  const chunks = (await chunkDocument(factDocumentRecord(record, record.value.applicationId))).slice(0, availableSlots);
  const citations = [];
  for (const chunk of chunks) {
    const digest = await sha256Base64Url(record.id + '|' + chunk.id + '|' + chunk.quoteHash);
    let citationId = 'u-' + digest.slice(0, 18);
    let suffix = 1;
    while (existingIds.has(citationId)) {
      citationId = 'u-' + digest.slice(0, 14) + '-' + suffix;
      suffix += 1;
    }
    existingIds.add(citationId);
    citations.push({
      citationId,
      documentId: chunk.documentId,
      documentVersion: chunk.documentVersion,
      chunkId: chunk.id,
      sourceRole: SOURCE_ROLES.USER_VERIFIED,
      locator: chunk.locator,
      quoteHash: chunk.quoteHash,
      text: chunk.text
    });
  }
  if (!citations.length) throw new Error('The saved clarification could not be converted into cited evidence.');
  return {
    citations: [...retainedCitations, ...citations],
    byField: {
      ...evidence.byField,
      [field.fieldId]: [...currentFieldIds, ...citations.map(citation => citation.citationId)]
    }
  };
};
const regenerationProposal = (output, field) => {
  const fillLike = output.value_type !== 'none'
    && output.confidence !== 'needs_input'
    && !output.abstain_reason;
  return {
    field_id: output.field_id,
    action: fillLike ? 'fill' : 'ask_user',
    confidence: output.confidence,
    risk_class: output.risk_class,
    value_type: output.value_type,
    value: output.value,
    selected_values: output.selected_values,
    checked: output.checked,
    citation_ids: output.citation_ids,
    short_rationale: output.changes_summary || (fillLike ? 'Regenerated from the same frozen evidence.' : 'User input required.'),
    abstain_reason: output.abstain_reason || (fillLike ? '' : `Review ${field.label} yourself.`)
  };
};

const EXPLICIT_EXPERIENCE_DURATION_PATTERN = /\b\d+(?:\.\d+)?\s*(?:(?:\+)\s*|(?:[-\u2013\u2014]|to)\s*\d+(?:\.\d+)?\s*)?(?:years?|months?)\b/iu;

export const recoverAbstainedClarifiedExperienceProposal = ({
  field,
  proposal,
  evidence,
  clarificationSaved = false,
  facts = [],
  applicationId = ''
} = {}) => {
  const cards = experienceCardsForField({ field, evidence });
  const normalizedApplicationId = String(applicationId || '');
  const fieldClarificationDocumentIds = new Set((facts || [])
    .filter(record => isFieldClarificationRecord(record)
      && String(record.value.applicationId || '') === normalizedApplicationId
      && record.value.fieldId === field?.fieldId)
    .map(record => record.id));
  const persistedClarificationCards = cards.filter(card => card.sourceRole === SOURCE_ROLES.USER_VERIFIED
    && fieldClarificationDocumentIds.has(card.documentId));
  if ((!clarificationSaved && !persistedClarificationCards.length)
    || proposal?.action !== 'ask_user'
    || !isNarrativeExperienceField(field)) return proposal;
  const clarificationCards = persistedClarificationCards.length
    ? persistedClarificationCards
    : cards.filter(card => card.sourceRole === SOURCE_ROLES.USER_VERIFIED);
  if (clarificationCards.some(card => EXPLICIT_EXPERIENCE_DURATION_PATTERN.test(String(card.text || '')))) return proposal;
  const fallback = createBestEffortExperienceProposal({ field, evidence });
  if (!fallback) return proposal;
  const fallbackCitationIds = [...new Set(fallback.citation_ids || [])];
  const matchedClarificationIds = [...new Set(persistedClarificationCards
    .map(card => card.citationId)
    .filter(Boolean))];
  const availableClarificationSlots = Math.max(
    0,
    VALIDATION_LIMITS.maxCitationsPerProposal - fallbackCitationIds.length
  );
  const addedClarificationIds = matchedClarificationIds
    .filter(citationId => !fallbackCitationIds.includes(citationId))
    .slice(0, availableClarificationSlots);
  return {
    ...fallback,
    citation_ids: [...addedClarificationIds, ...fallbackCitationIds]
  };
};

export const resolveBlankFeedbackRegenerationProposal = ({
  priorProposal,
  regeneratedProposal,
  feedback
} = {}) => {
  const priorCitationIds = Array.isArray(priorProposal?.citation_ids)
    ? priorProposal.citation_ids.filter(Boolean)
    : [];
  const retained = priorProposal?.action === 'fill'
    && priorCitationIds.length > 0
    && regeneratedProposal?.action === 'ask_user'
    && !String(feedback?.text || '').trim();
  return {
    proposal: retained ? priorProposal : regeneratedProposal,
    retained
  };
};
const regenerateOneField = async (fieldId, button, feedbackOverride = null) => {
  const { field, proposal } = fieldAndProposal(fieldId);
  if (!field || !proposal || !state.evidence) throw new Error('Analyze this field before regenerating it.');
  if (field.riskClass === 'F1_VERIFIED') throw new Error('Verified fact fields are deterministic and cannot be regenerated.');
  if (eligibilityQuestionKind(field)) {
    const next = deterministicEligibilityProposal(field, state.evidence);
    if (!next) throw new Error('Complete the matching work-eligibility selections in the vault, then analyze again.');
    state.proposals = state.proposals.map(candidate => candidate.field_id === fieldId ? next : candidate);
    state.filledFieldIds.delete(fieldId);
    state.batchStatus = '';
    const batchReview = $('[data-batch-review]');
    if (batchReview) batchReview.checked = false;
    await storeGenerationCache({
      scan: state.scan,
      sourceSignature: state.sourceSignature,
      evidence: state.evidence,
      proposals: state.proposals,
      retrievalMode: state.retrievalMode
    });
    await publishProposals();
    renderApplication();
    showToast('Eligibility answer refreshed from your explicit saved selections.', 'success');
    return;
  }
  const feedbackPanel = $('[data-feedback-panel="' + CSS.escape(fieldId) + '"]');
  const preset = feedbackOverride?.preset ?? feedbackPanel?.querySelector('[data-feedback-preset]')?.value ?? '';
  const text = feedbackOverride?.text ?? feedbackPanel?.querySelector('[data-feedback-text]')?.value ?? '';
  const clarificationQuestion = clarificationQuestionForProposal({
    field,
    proposal,
    cards: citationCardsForProposal(proposal)
  });
  const clarificationMode = Boolean(clarificationQuestion);
  let feedback = normalizeFieldRegenerationFeedback({
    preset,
    text,
    maxChars: field.maxLength || null,
    mustInclude: [],
    mustAvoid: []
  });
  abortActiveOperations();
  const controller = new AbortController();
  state.generationController = controller;
  const expectedScan = state.scan;
  const expectedTabId = state.activeTabId;
  const expectedStructureSignature = await applicationStructureSignature(expectedScan);
  setButtonBusy(button, true, clarificationMode && text.trim() ? 'Saving & regenerating...' : 'Regenerating...');
  let clarificationSaved = false;
  try {
    if (clarificationMode && text.trim()) {
      const applicationId = String(state.scan?.pageId || '');
      const existing = state.facts.find(record => record.value?.applicationId === applicationId
        && record.value?.fieldId === field.fieldId);
      const recordToken = existing?.value?.key?.split(':').slice(1).join(':') || crypto.randomUUID();
      const record = createFieldClarificationRecord({
        field,
        applicationId,
        text,
        existingFacts: state.facts.filter(candidate => candidate.id !== existing?.id),
        recordToken
      });
      await state.vault.putRecord(record);
      clarificationSaved = true;
      state.facts = existing
        ? state.facts.map(candidate => candidate.id === existing.id ? record : candidate)
        : [...state.facts, record];
      state.evidence = await appendFieldClarificationEvidence({
        evidence: state.evidence,
        field,
        record,
        applicationId
      });
      try {
        await persistClarificationRecordSnapshot({ field, record, text });
      } catch (error) {
        state.analysisNotice = 'Clarification was saved and cited, but the application packet could not be updated. ' + readableError(error);
      }
      state.sourceSignature = await sourceSignatureForRecords();
      feedback = {
        preset: 'more_specific',
        text: 'Use the newly saved user-verified clarification citation to make this answer more specific. Do not add facts beyond the frozen citations.',
        maxChars: field.maxLength || null,
        mustInclude: [],
        mustAvoid: []
      };
      renderSources();
    }


    const generated = await orchestrateFieldRegeneration({
      client: state.ollama,
      field: { ...field, manual: false },
      priorDraft: proposal,
      feedback,
      evidence: evidenceForModelField(state.evidence, field.fieldId),
      primaryModel: state.settings.generationModel,
      fallbackModel: state.settings.fallbackGenerationModel,
      signal: controller.signal
    });
    await assertFreshGenerationCommit({
      tabId: expectedTabId,
      expectedScan,
      expectedStructureSignature,
      controller
    });
    const recovered = recoverAbstainedClarifiedExperienceProposal({
      field,
      proposal: regenerationProposal(generated.output, field),
      evidence: evidenceForModelField(state.evidence, field.fieldId),
      clarificationSaved,
      facts: state.facts,
      applicationId: String(state.scan?.pageId || '')
    });
    const resolution = resolveBlankFeedbackRegenerationProposal({
      priorProposal: proposal,
      regeneratedProposal: recovered,
      feedback
    });
    const next = resolution.proposal;
    state.proposals = state.proposals.map(candidate => candidate.field_id === fieldId ? next : candidate);
    try {
      await persistApplicationPacket({ scan: state.scan });
    } catch (error) {
      state.analysisNotice = 'The regenerated answer is available, but the encrypted application packet could not be updated. ' + readableError(error);
    }
    if (!resolution.retained) {
      state.filledFieldIds.delete(fieldId);
      state.batchStatus = '';
      const batchReview = $('[data-batch-review]');
      if (batchReview) batchReview.checked = false;
    }
    await storeGenerationCache({
      scan: state.scan,
      sourceSignature: state.sourceSignature,
      evidence: state.evidence,
      proposals: state.proposals,
      retrievalMode: state.retrievalMode
    });
    await publishProposals();
    renderApplication();
    showToast(
      resolution.retained
        ? 'The model could not produce a better supported revision, so the current cited answer was kept.'
        : clarificationSaved
          ? 'Clarification encrypted, cited, and used to regenerate this field.'
          : 'Field regenerated from the frozen evidence snapshot.',
      'success'
    );
  } catch (error) {
    if (clarificationSaved) {
      renderApplication();
      throw new Error(clarificationRegenerationFailureMessage({ clarificationSaved, error }), { cause: error });
    }
    throw error;
  } finally {
    if (state.generationController === controller) state.generationController = null;
    setButtonBusy(button, false);
  }
};
const trackerJobFromForm = (form) => {
  const values = new FormData(form);
  const jobUrl = String(values.get('jobUrl') || '').trim();
  if (jobUrl) {
    const parsed = new URL(jobUrl);
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Job URL must use HTTP or HTTPS.');
  }
  return {
    company: String(values.get('company') || '').trim(),
    title: String(values.get('title') || '').trim(),
    jobUrl,
    location: String(values.get('location') || '').trim(),
    source: String(values.get('source') || '').trim(),
    postingDate: String(values.get('postingDate') || ''),
    appliedDate: String(values.get('appliedDate') || ''),
    status: String(values.get('status') || 'Applied'),
    tags: parseTrackerTags(values.get('tags'))
  };
};

const beginTrackerHandoff = async (form, button) => {
  if (!form.reportValidity()) return;
  const values = new FormData(form);
  // Submitting this reviewed form is the explicit extension-side handoff approval.
  const recordSelections = [
    { recordId: String(values.get('resumeRecordId') || ''), kind: 'resume' },
    { recordId: String(values.get('coverRecordId') || ''), kind: 'cover-letter' }
  ].filter(selection => selection.recordId);
  if (new Set(recordSelections.map(selection => selection.recordId)).size !== recordSelections.length) {
    throw new Error('Choose different files for the resume and cover letter.');
  }
  recordSelections.forEach((selection) => {
    const record = state.documents.find(candidate => candidate.id === selection.recordId);
    if (!isEligibleTrackerDocument(record)) throw new Error('A selected tracker attachment is not an approved retained PDF or DOCX.');
  });
  const issuedAt = new Date();
  const expiresAt = new Date(issuedAt.getTime() + 15 * 60 * 1000);
  const message = {
    type: TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN,
    payload: {
      captureId: crypto.randomUUID(),
      issuedAt: issuedAt.toISOString(),
      expiresAt: expiresAt.toISOString(),
      job: trackerJobFromForm(form),
      files: recordSelections.map(selection => ({
        fileId: crypto.randomUUID(),
        recordId: selection.recordId,
        kind: selection.kind
      }))
    }
  };
  setButtonBusy(button, true, 'Preparing handoff...');
  try {
    const response = await chrome.runtime.sendMessage(message);
    if (!response?.ok || typeof response.trackerUrl !== 'string') {
      throw new Error(response?.error?.message || 'Unable to prepare the tracker handoff.');
    }
    await chrome.tabs.create({ url: response.trackerUrl });
    // The tracker page provides the final review and Save step.
    showToast('Tracker opened for sign-in and final review.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const captureCurrentJobContext = async (button) => {
  requirePrivacyConsent();
  if (!state.unlocked) throw new Error('Unlock the encrypted vault before saving job context.');
  setButtonBusy(button, true, 'Capturing...');
  try {
    const scan = await scanActivePage();
    if (!String(scan.job?.description || '').trim()) {
      renderApplication();
      throw new Error('No visible job description was found. Select the description on the page or paste it in Job context options.');
    }
    await persistApplicationPacket({ scan });
    renderApplication();
    showToast('Job description captured and encrypted for this application.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const captureSelectedJobText = async (button) => {
  requirePrivacyConsent();
  if (!state.unlocked) throw new Error('Unlock the encrypted vault before saving selected job text.');
  setButtonBusy(button, true, 'Capturing selection...');
  try {
    const tab = await activeHttpTab();
    const [selectionResult] = await chrome.scripting.executeScript({
      target: { tabId: tab.id, frameIds: [0] },
      func: () => String(globalThis.getSelection?.()?.toString?.() || '')
        .replace(/\r\n?/gu, '\n')
        .trim()
        .slice(0, 20_000)
    });
    const selectedText = String(selectionResult?.result || '').trim();
    if (!selectedText) throw new Error('Select the visible job description on the page, then try again.');
    const scan = await scanActivePage();
    const nextScan = {
      ...scan,
      job: { ...(scan.job || {}), description: selectedText }
    };
    state.scan = nextScan;
    await invalidateGeneratedAnswers();
    await persistApplicationPacket({
      scan: nextScan,
      context: applicationPacketContextForScan(nextScan, {
        content: selectedText,
        label: 'Selected page text'
      })
    });
    renderApplication();
    showToast('Selected job text captured and encrypted.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const saveManualJobContext = async (form, button) => {
  requirePrivacyConsent();
  if (!state.unlocked || !state.scan) throw new Error('Analyze a job page and unlock the vault first.');
  if (!form.reportValidity()) return;
  const values = new FormData(form);
  const description = String(values.get('description') || '').trim();
  const job = {
    ...(state.scan.job || {}),
    company: String(values.get('company') || '').trim(),
    title: String(values.get('title') || '').trim(),
    jobUrl: String(values.get('jobUrl') || '').trim(),
    location: String(values.get('location') || '').trim(),
    description
  };
  const nextScan = { ...state.scan, job };
  setButtonBusy(button, true, 'Encrypting...');
  try {
    state.scan = nextScan;
    await invalidateGeneratedAnswers();
    await persistApplicationPacket({
      scan: nextScan,
      context: applicationPacketContextForScan(nextScan, {
        origin: APPLICATION_CONTEXT_ORIGINS.MANUAL,
        mode: APPLICATION_CONTEXT_MODES.MANUAL,
        content: description,
        label: 'Manual job context',
        job
      })
    });
    delete form.dataset.contextKey;
    renderApplication();
    showToast('Manual job context encrypted and attached.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const useSavedJobContext = async (button) => {
  const picker = $('[data-job-context-picker]');
  const packet = packetRecordByApplicationId(picker?.value);
  if (!packet || !state.scan) throw new Error('Choose a saved job context first.');
  setButtonBusy(button, true, 'Attaching...');
  try {
    state.activeApplicationPacketId = packet.value.applicationId;
    state.scan = mergeApplicationPacketContext({
      scan: state.scan,
      packetRecord: packet,
      preferPacket: true
    });
    await invalidateGeneratedAnswers();
    await persistApplicationPacket({ scan: state.scan });
    renderApplication();
    showToast('Saved job context attached to this application page.', 'success');
  } finally {
    setButtonBusy(button, false);
  }
};

const installApplicationActions = () => {
  $('[data-action="scan-active-page"]').addEventListener('click', analyzeActivePage);
  $('[data-action="refresh-job-context"]').addEventListener('click', async (event) => {
    try {
      await captureCurrentJobContext(event.currentTarget);
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-action="capture-selected-job-text"]').addEventListener('click', async (event) => {
    try {
      await captureSelectedJobText(event.currentTarget);
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-action="use-saved-job-context"]').addEventListener('click', async (event) => {
    try {
      await useSavedJobContext(event.currentTarget);
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-job-context-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      await saveManualJobContext(event.currentTarget, event.submitter);
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-action="toggle-unresolved"]').addEventListener('click', () => {
    state.answerFilter = state.answerFilter === 'attention' ? 'all' : 'attention';
    renderApplication();
  });
  $('[data-batch-clarification-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const entries = [...form.querySelectorAll('[data-batch-clarification-field]')]
      .map(input => ({ fieldId: input.dataset.batchClarificationField, text: input.value.trim() }))
      .filter(entry => entry.text);
    if (!entries.length) {
      showToast('Add at least one factual clarification before regenerating.', 'error');
      return;
    }
    setButtonBusy(button, true, `Strengthening ${entries.length} answer${entries.length === 1 ? '' : 's'}...`);
    try {
      for (const entry of entries) {
        await regenerateOneField(entry.fieldId, null, { preset: 'more_specific', text: entry.text });
      }
      state.answerFilter = 'attention';
      renderApplication();
      showToast(`${entries.length} clarification${entries.length === 1 ? '' : 's'} saved and applied.`, 'success');
    } catch (error) {
      showToast(readableError(error), 'error');
    } finally {
      setButtonBusy(button, false);
    }
  });
  $('[data-batch-review]').addEventListener('change', () => {
    state.batchStatus = '';
    renderBatchFillControls();
  });
  $('[data-action="fill-all-ready"]').addEventListener('click', async () => {
    try {
      await fillReadyFields();
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-action="fill-verified-only"]').addEventListener('click', async () => {
    try {
      await fillReadyFields({ verifiedOnly: true });
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-tracker-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      await beginTrackerHandoff(event.currentTarget, event.submitter);
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-answer-list]').addEventListener('click', async (event) => {
    const fillButton = event.target.closest('[data-fill-field]');
    const copyButton = event.target.closest('[data-copy-answer]');
    const toggleButton = event.target.closest('[data-toggle-feedback]');
    const regenerateButton = event.target.closest('[data-regenerate-field]');
    try {
      if (fillButton) await fillField(fillButton.dataset.fillField, fillButton);
      else if (copyButton) {
        const proposal = state.proposals.find(candidate => candidate.field_id === copyButton.dataset.copyAnswer);
        if (proposal) await copyProposal(proposal);
      } else if (toggleButton) {
        const panel = $(`[data-feedback-panel="${CSS.escape(toggleButton.dataset.toggleFeedback)}"]`);
        panel.hidden = !panel.hidden;
        toggleButton.setAttribute('aria-expanded', String(!panel.hidden));
        if (!panel.hidden) panel.querySelector('textarea').focus();
      } else if (regenerateButton) {
        await regenerateOneField(regenerateButton.dataset.regenerateField, regenerateButton);
      }
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
};

const renderRuntimeConfiguration = () => {
  const command = $('[data-ollama-origin]');
  const extensionId = globalThis.chrome?.runtime?.id || '<extension-id>';
  if (command) command.textContent = ollamaOriginCommand(extensionId);
};

const renderSettings = () => {
  const form = $('[data-settings-form]');
  if (!form) return;
  form.elements.generationModel.value = state.settings.generationModel;
  form.elements.fallbackGenerationModel.value = state.settings.fallbackGenerationModel;
  form.elements.embeddingModel.value = state.settings.embeddingModel;
};

const updateOllamaStatus = (stateName, title, copy) => {
  const panel = $('[data-ollama-status]');
  panel.dataset.state = stateName;
  panel.querySelector('strong').textContent = title;
  panel.querySelector('span').textContent = copy;
  setOllamaIndicator(stateName === 'ready' ? 'ready' : stateName === 'error' ? 'error' : 'unknown', title);
};

const inspectInstalledModels = async (models, signal) => {
  const names = [...new Set((models.models || []).map(model => model.name || model.model).filter(Boolean))];
  const inspected = await Promise.all(names.map(async (name) => {
    try {
      await state.ollama.assertModelIsLocal(name, { signal });
      return { name, local: true };
    } catch (error) {
      if (error?.code === 'REMOTE_MODEL_NOT_ALLOWED') return { name, local: false };
      throw error;
    }
  }));
  return {
    localNames: inspected.filter(entry => entry.local).map(entry => entry.name),
    blockedNames: inspected.filter(entry => !entry.local).map(entry => entry.name)
  };
};

const checkOllama = async (button = $('[data-action="check-ollama"]')) => {
  requirePrivacyConsent();
  state.ollamaController?.abort();
  const controller = new AbortController();
  state.ollamaController = controller;
  setButtonBusy(button, true, 'Checking...');
  updateOllamaStatus('checking', 'Checking local-only Ollama', 'Connecting to the fixed loopback origin and inspecting model metadata...');
  try {
    const [version, models] = await Promise.all([
      state.ollama.getVersion({ signal: controller.signal }),
      state.ollama.listModels({ signal: controller.signal })
    ]);
    const { localNames, blockedNames } = await inspectInstalledModels(models, controller.signal);
    requirePrivacyConsent();
    const list = $('#ollama-model-list');
    list.replaceChildren(...localNames.map(name => createElement('option', { attributes: { value: name } })));
    const required = [state.settings.generationModel, state.settings.embeddingModel];
    const configured = [...required, state.settings.fallbackGenerationModel];
    const missing = required.filter(name => !isModelInstalled(name, localNames));
    const fallbackMissing = !isModelInstalled(state.settings.fallbackGenerationModel, localNames);
    const blockedConfigured = configured.filter(name => isModelInstalled(name, blockedNames));
    const versionUnsupported = isClearlyUnsupportedOllamaVersion(version.version);
    const detailParts = [
      String(version.version || 'Version unknown'),
      localNames.length + ' verified local ' + (localNames.length === 1 ? 'model' : 'models'),
      'OLLAMA_NO_CLOUD=1 required'
    ];
    if (blockedNames.length) detailParts.push(blockedNames.length + ' cloud or remote ' + (blockedNames.length === 1 ? 'model' : 'models') + ' blocked');
    if (missing.length) detailParts.push('install local ' + missing.join(', '));
    if (!missing.length && fallbackMissing) detailParts.push('optional local fallback ' + state.settings.fallbackGenerationModel + ' is not installed');
    if (versionUnsupported) detailParts.push('update Ollama to 0.32.0 or newer');
    const hasError = Boolean(missing.length || blockedConfigured.length || versionUnsupported);
    updateOllamaStatus(
      hasError ? 'error' : 'ready',
      blockedConfigured.length
        ? 'Cloud or remote model blocked'
        : versionUnsupported
          ? 'Ollama update required'
          : missing.length
            ? 'Ollama connected; local models missing'
            : 'Ollama ready',
      detailParts.join(' | ')
    );
  } catch (error) {
    if (isCurrentPrivacyConsent(state.privacyConsent)) {
      const presentation = ollamaHealthErrorPresentation(error, globalThis.chrome?.runtime?.id);
      updateOllamaStatus('error', presentation.title, presentation.copy);
    }
  } finally {
    if (state.ollamaController === controller) state.ollamaController = null;
    setButtonBusy(button, false);
  }
};

const fixtureContractError = () => new Error('The sanitized fixture failed its privacy contract.');
const isPlainRecord = value => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const assertFixtureKeys = (value, allowed, required = allowed) => {
  if (!isPlainRecord(value)) throw fixtureContractError();
  const keys = Object.keys(value);
  if (keys.some(key => !allowed.includes(key)) || required.some(key => !keys.includes(key))) {
    throw fixtureContractError();
  }
};
const safeFixtureString = (value, maxLength, pattern) => {
  if (typeof value !== 'string' || value.length > maxLength || (pattern && !pattern.test(value))) {
    throw fixtureContractError();
  }
  return value;
};
const fixtureCount = value => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 10_000) throw fixtureContractError();
  return value;
};

export const validateSanitizedFixture = (fixture) => {
  const topLevelKeys = ['adapter', 'captchaPresent', 'discovery', 'fields', 'schemaVersion'];
  assertFixtureKeys(fixture, topLevelKeys, ['adapter', 'captchaPresent', 'fields', 'schemaVersion']);
  if (!Number.isSafeInteger(fixture.schemaVersion) || fixture.schemaVersion < 1 || fixture.schemaVersion > RUNTIME_VERSION) {
    throw fixtureContractError();
  }
  if (typeof fixture.captchaPresent !== 'boolean' || !Array.isArray(fixture.fields)
    || fixture.fields.length > VALIDATION_LIMITS.maxFieldsPerPage) {
    throw fixtureContractError();
  }
  const fields = fixture.fields.map((field) => {
    const allowedKeys = ['fieldId', 'fillMode', 'label', 'options', 'riskClass', 'type'];
    assertFixtureKeys(field, allowedKeys, ['fieldId', 'label', 'options', 'riskClass', 'type']);
    if (!Array.isArray(field.options) || field.options.length > VALIDATION_LIMITS.maxOptionsPerField) {
      throw fixtureContractError();
    }
    const sanitized = {
      fieldId: safeFixtureString(field.fieldId, 80, /^[A-Za-z0-9._:-]+$/u),
      label: safeFixtureString(field.label, VALIDATION_LIMITS.maxLabelLength),
      type: safeFixtureString(field.type, 40, /^[a-z0-9-]+$/u),
      options: field.options.map(option => safeFixtureString(option, 1_000)),
      riskClass: safeFixtureString(field.riskClass, 20, /^(?:F1_VERIFIED|F2_REVIEW)$/u)
    };
    if (Object.hasOwn(field, 'fillMode')) {
      if (field.fillMode !== 'copy_only') throw fixtureContractError();
      sanitized.fillMode = 'copy_only';
    }
    return sanitized;
  });
  const validated = {
    schemaVersion: fixture.schemaVersion,
    adapter: safeFixtureString(fixture.adapter, 64, /^[a-z0-9._-]+$/iu),
    captchaPresent: fixture.captchaPresent,
    fields
  };
  if (Object.hasOwn(fixture, 'discovery')) {
    const discovery = fixture.discovery;
    const discoveryKeys = ['contexts', 'exclusionCounts', 'mode', 'recognizedCount', 'truncated', 'unsupportedCount'];
    assertFixtureKeys(discovery, discoveryKeys);
    if (!DISCOVERY_MODES.has(discovery.mode) || typeof discovery.truncated !== 'boolean'
      || !Array.isArray(discovery.contexts) || discovery.contexts.length > 16) {
      throw fixtureContractError();
    }
    assertFixtureKeys(discovery.exclusionCounts, DISCOVERY_EXCLUSION_KEYS);
    const exclusionCounts = Object.fromEntries(DISCOVERY_EXCLUSION_KEYS.map(key => [
      key,
      fixtureCount(discovery.exclusionCounts[key])
    ]));
    const seenContexts = new Set();
    const contexts = discovery.contexts.map((context) => {
      assertFixtureKeys(context, ['count', 'kind', 'status']);
      if (!DISCOVERY_CONTEXT_KINDS.has(context.kind) || !DISCOVERY_CONTEXT_STATUSES.has(context.status)) {
        throw fixtureContractError();
      }
      const identity = `${context.kind}:${context.status}`;
      if (seenContexts.has(identity)) throw fixtureContractError();
      seenContexts.add(identity);
      const count = fixtureCount(context.count);
      if (count < 1) throw fixtureContractError();
      return { kind: context.kind, count, status: context.status };
    });
    validated.discovery = {
      mode: discovery.mode,
      recognizedCount: fixtureCount(discovery.recognizedCount),
      unsupportedCount: fixtureCount(discovery.unsupportedCount),
      exclusionCounts,
      contexts,
      truncated: discovery.truncated
    };
  }
  return validated;
};

export const prepareSanitizedFixtureExport = async fixture => {
  const validated = validateSanitizedFixture(fixture);
  const structureSignature = await sha256Base64Url(canonicalJson(validated));
  return { ...validated, structureSignature };
};

const exportSanitizedFixture = async () => {
  if (!state.scan || !state.activeTabId) throw new Error('Analyze the active page before exporting a sanitized fixture.');
  const response = await sendPageMessage(state.activeTabId, 'SANITIZED_EXPORT_REQUEST');
  if (response.type !== 'SANITIZED_EXPORT_RESULT' || typeof response.payload?.json !== 'string') {
    throw new Error('The page runtime did not return a sanitized fixture.');
  }
  const fixture = await prepareSanitizedFixtureExport(JSON.parse(response.payload.json));
  const blob = new Blob([`${JSON.stringify(fixture, null, 2)}\n`], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = createElement('a', {
    attributes: { href: url, download: `job-copilot-${String(fixture.adapter || 'ats').replace(/[^a-z0-9-]+/giu, '-')}-fixture.json` }
  });
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  showToast('Sanitized ATS fixture downloaded.', 'success');
};

const clearTrustedSessionCapabilities = async () => {
  const values = await chrome.storage.session.get(null);
  const keys = Object.keys(values || {}).filter(key => key === SELECTED_FIELD_KEY
    || key.startsWith(TRACKER_PENDING_KEY_PREFIX)
    || key.startsWith(VAULT_SESSION_KEY_PREFIX));
  if (keys.length) await chrome.storage.session.remove(keys);
};

const withdrawPrivacyConsent = async () => {
  abortActiveOperations();
  await chrome.storage.local.remove(PRIVACY_CONSENT_KEY);
  state.privacyConsent = null;
  state.privacyReviewOpen = false;
  const cleanupResults = await Promise.allSettled([
    state.vault?.lock(),
    clearTrustedSessionCapabilities()
  ]);
  state.unlocked = false;
  state.activeTabId = null;
  state.scan = null;
  state.stale = false;
  state.selectedFieldId = null;
  state.retrievalMode = '';
  $('[data-operational-shell]').querySelectorAll('form').forEach(form => form.reset());
  clearDecryptedState();
  renderPrivacyExperience();
  const cleanupFailed = cleanupResults.some(result => result.status === 'rejected');
  showToast(
    cleanupFailed
      ? 'Consent withdrawn and features blocked. Restart Chrome to ensure all session keys are cleared.'
      : 'Consent withdrawn. The vault is locked and retained; Copilot features are blocked.',
    cleanupFailed ? 'error' : 'success'
  );
};

const installSettingsActions = () => {
  $('[data-action="check-ollama"]').addEventListener('click', event => void checkOllama(event.currentTarget));
  $('[data-action="copy-origin"]').addEventListener('click', async () => {
    await navigator.clipboard.writeText($('[data-ollama-origin]').textContent);
    showToast('Required Ollama origin copied.', 'success');
  });
  $('[data-action="copy-no-cloud"]').addEventListener('click', async () => {
    await navigator.clipboard.writeText($('[data-ollama-no-cloud]').textContent);
    showToast('Required local-only setting copied.', 'success');
  });
  $('[data-action="review-privacy"]').addEventListener('click', () => {
    state.privacyReviewOpen = true;
    renderPrivacyExperience();
    $('[data-privacy-onboarding]').focus?.();
  });
  $('[data-action="withdraw-privacy"]').addEventListener('click', async () => {
    if (!confirm('Withdraw consent and lock the vault? Encrypted vault data will be retained.')) return;
    await withdrawPrivacyConsent();
  });
  $('[data-settings-form]').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const values = Object.fromEntries(new FormData(event.currentTarget));
      state.settings = validateModelSettings(values);
      await chrome.storage.local.set({ [MODEL_SETTINGS_KEY]: state.settings });
      renderSettings();
      showToast('Local model settings saved.', 'success');
      void checkOllama();
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
  $('[data-action="lock-vault"]').addEventListener('click', async () => {
    abortActiveOperations();
    await state.vault.lock();
    state.unlocked = false;
    $$('[data-vault-form], [data-profile-form], [data-custom-profile-form], [data-document-form], [data-note-form]').forEach(form => form.reset());
    $('[data-import-progress]').hidden = true;
    $('[data-import-progress] progress').value = 0;
    clearDecryptedState();
    renderSources();
    showToast('Vault locked and decrypted panel state cleared.', 'success');
    activateTab('sources');
  });
  $('[data-action="reset-vault"]').addEventListener('click', async () => {
    if (!confirm('Permanently delete the encrypted local vault and all of its records? This cannot be undone.')) return;
    abortActiveOperations();
    await state.vault.reset();
    state.initialized = false;
    state.unlocked = false;
    state.scan = null;
    state.activeTabId = null;
    $$('form').forEach(form => form.reset());
    clearDecryptedState();
    renderSources();
    showToast('Encrypted local vault reset.', 'success');
    activateTab('sources');
  });
  $('[data-action="export-sanitized-fixture"]').addEventListener('click', async () => {
    try {
      await exportSanitizedFixture();
    } catch (error) {
      showToast(readableError(error), 'error');
    }
  });
};

const handledInlineSelectionRequests = new Set();

const handleInlineFieldSelection = async (selected) => {
  if (!selected?.fieldId) return;
  activateTab('application');
  if (state.activeTabId !== null && selected.tabId !== state.activeTabId) {
    showToast('This tab has not been analyzed. Rescan it before reviewing or regenerating answers.', 'error');
    return;
  }
  state.selectedFieldId = selected.fieldId;
  renderApplication();
  if (selected.intent !== 'regenerate') return;
  const requestKey = String(selected.requestId || selected.selectedAt || '');
  if (!requestKey || handledInlineSelectionRequests.has(requestKey)) return;
  handledInlineSelectionRequests.add(requestKey);
  if (handledInlineSelectionRequests.size > 200) {
    handledInlineSelectionRequests.delete(handledInlineSelectionRequests.values().next().value);
  }
  if (!state.unlocked) {
    showToast('Unlock the vault, then use regenerate again.', 'error');
    return;
  }
  try {
    await chrome.storage.session.set({
      [SELECTED_FIELD_KEY]: { ...selected, intent: 'review' }
    });
    await regenerateOneField(selected.fieldId, null);
  } catch (error) {
    showToast(readableError(error), 'error');
  }
};

const installRuntimeListeners = () => {
  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (!isCurrentPrivacyConsent(state.privacyConsent)) return;
    if (areaName !== 'session' || !changes[SELECTED_FIELD_KEY]?.newValue) return;
    void handleInlineFieldSelection(changes[SELECTED_FIELD_KEY].newValue);
  });
  chrome.runtime.onMessage.addListener((message, sender) => {
    if (!isCurrentPrivacyConsent(state.privacyConsent)) return false;
    if (message?.channel !== RUNTIME_CHANNEL || message?.type !== 'PAGE_SCAN_RESULT'
      || message?.payload?.stale !== true || sender?.tab?.id !== state.activeTabId) return false;
    if (state.scan && message.payload.pageId === state.scan.pageId) {
      state.stale = true;
      if (state.bulkBusy) {
        state.batchStatus = 'The form changed while filling. Finishing unchanged reviewed fields safely...';
        renderBatchFillControls();
        return false;
      }
      renderApplication();
      showToast('The application form changed. Rescan before filling.', 'error');
    }
    return false;
  });
};

const initPreview = () => {
  document.body.dataset.preview = 'true';
  state.privacyConsent = { version: PRIVACY_NOTICE_VERSION, acceptedAt: '2026-07-18T18:00:00.000Z' };
  state.privacyReviewOpen = false;
  renderPrivacyExperience();
  state.initialized = true;
  state.unlocked = true;
  state.settings = defaultSettings();
  state.facts = [{
    id: 'fact:email',
    updatedAt: '2026-07-18T18:00:00.000Z',
    value: { key: 'email', label: 'Email address', value: 'candidate@example.com' }
  }];
  state.documents = [{
    id: 'doc:preview-resume',
    updatedAt: '2026-07-18T18:00:00.000Z',
    value: {
      document: {
        id: 'doc:preview-resume',
        filename: 'Candidate-Resume.pdf',
        mimeType: 'application/pdf',
        size: 184320,
        sourceRole: SOURCE_ROLES.CANDIDATE_EVIDENCE
      },
      originalBytesBase64: 'preview-only'
    }
  }];
  state.selectedSourceIds.add('doc:preview-resume');
  state.scan = {
    pageId: 'page-preview',
    urlHash: 'preview',
    domRevision: 0,
    adapter: 'generic',
    captchaPresent: false,
    discovery: {
      mode: 'free_format',
      recognizedCount: 3,
      unsupportedCount: 1,
      exclusionCounts: { F0_EXCLUDED: 2, F3_SENSITIVE: 1, F4_CONSENT: 1 },
      contexts: [{ kind: 'custom_aria_widget', count: 1, status: 'manual' }],
      truncated: false
    },
    job: {
      company: 'Example Analytics',
      title: 'Data Analyst',
      location: 'Denver, CO',
      source: 'Custom application',
      jobUrl: 'https://jobs.example/apply',
      description: 'Example Analytics is hiring a Data Analyst to build trustworthy reporting, explain performance trends, and partner with cross-functional teams.'
    },
    fields: [{
      fieldId: 'field-preview-email',
      fingerprint: 'previewemail00001',
      label: 'Email address',
      type: 'email',
      options: [],
      nearbyText: '',
      required: true,
      riskClass: 'F1_VERIFIED'
    }, {
      fieldId: 'field-preview-summary',
      fingerprint: 'previewsummary01',
      label: 'Why are you interested in this role?',
      type: 'textarea',
      options: [],
      nearbyText: 'Share relevant experience.',
      required: true,
      riskClass: 'F2_REVIEW',
      fillMode: 'copy_only'
    }, {
      fieldId: 'field-preview-fintech',
      fingerprint: 'previewfintech01',
      label: 'How many years of fintech/payments experience do you have? Please explain.',
      type: 'textarea',
      options: [],
      nearbyText: 'Describe direct and transferable experience.',
      required: true,
      riskClass: 'F2_REVIEW',
      maxLength: 1000
    }]
  };
  state.evidence = {
    citations: [{
      citationId: 'c1',
      documentId: 'fact:email',
      documentVersion: 1,
      chunkId: 'chunk:preview:1',
      sourceRole: SOURCE_ROLES.USER_VERIFIED,
      locator: { section: 'Verified profile fact' },
      quoteHash: 'preview',
      text: 'Email address: candidate@example.com'
    }, {
      citationId: 'c2',
      documentId: 'doc:preview-resume',
      documentVersion: 1,
      chunkId: 'chunk:preview:2',
      sourceRole: SOURCE_ROLES.CANDIDATE_EVIDENCE,
      locator: { section: 'Professional summary', pageStart: 1, pageEnd: 1 },
      quoteHash: 'preview',
      text: 'Built transparent analytics tools and presented findings to cross-functional stakeholders.'
    }, {
      citationId: 'c3',
      documentId: 'doc:preview-resume',
      documentVersion: 1,
      chunkId: 'chunk:preview:3',
      sourceRole: SOURCE_ROLES.CANDIDATE_EVIDENCE,
      locator: { section: 'Analytics experience', pageStart: 1, pageEnd: 1 },
      quoteHash: 'preview',
      text: 'Built campaign performance dashboards, automated reporting workflows, and presented revenue insights to cross-functional stakeholders at Example Analytics.'
    }],
    byField: {
      'field-preview-email': ['c1'],
      'field-preview-summary': ['c2'],
      'field-preview-fintech': ['c3']
    }
  };
  const previewExperienceField = state.scan.fields.find(field => field.fieldId === 'field-preview-fintech');
  const previewExperienceProposal = createBestEffortExperienceProposal({
    field: previewExperienceField,
    evidence: state.evidence
  });
  if (!previewExperienceProposal) throw new Error('The experience recovery preview could not be grounded.');
  state.proposals = [{
    field_id: 'field-preview-email', action: 'fill', confidence: 'high', risk_class: 'F1_VERIFIED', value_type: 'text', value: 'candidate@example.com', selected_values: [], checked: false, citation_ids: ['c1'], short_rationale: 'Exact email address from a user-verified profile fact.', abstain_reason: ''
  }, {
    field_id: 'field-preview-summary', action: 'fill', confidence: 'review', risk_class: 'F2_REVIEW', value_type: 'text', value: 'I am interested in applying my experience building transparent analytics tools in a collaborative environment.', selected_values: [], checked: false, citation_ids: ['c2'], short_rationale: 'Drafted from candidate evidence; review the framing before filling.', abstain_reason: ''
  }, previewExperienceProposal];
  state.selectedFieldId = null;
  state.retrievalMode = 'hybrid';
  renderSources();
  renderRuntimeConfiguration();
  renderSettings();
  renderApplication();
  updateOllamaStatus('ready', 'Preview: Ollama ready', 'Synthetic preview state; no local service request was made.');
  activateTab('application');
};

const startOperationalControllers = async () => {
  requirePrivacyConsent();
  let initialFieldSelection = null;
  if (!state.operationalStarted) {
    state.vault = new EncryptedIndexedDbVault();
    state.ollama = new OllamaClient();
    state.research = new ManualResearchProvider();
    const sessionValues = await chrome.storage.session.get(SELECTED_FIELD_KEY);
    initialFieldSelection = sessionValues[SELECTED_FIELD_KEY] || null;
    state.selectedFieldId = initialFieldSelection?.fieldId || null;
    installOperationalConsentGuard();
    installVaultActions();
    installApplicationActions();
    installSettingsActions();
    installRuntimeListeners();
    state.operationalStarted = true;
  }
  renderRuntimeConfiguration();
  renderSettings();
  await refreshVaultStatus();
  activateTab(state.unlocked ? 'application' : 'sources');
  renderApplication();
  if (initialFieldSelection) void handleInlineFieldSelection(initialFieldSelection);
  void checkOllama();
};

const initialize = async () => {
  installTabs();
  window.addEventListener('pagehide', abortActiveOperations, { once: true });
  if (PREVIEW_MODE) {
    initPreview();
    return;
  }
  await Promise.all([
    chrome.storage.local.setAccessLevel?.({ accessLevel: 'TRUSTED_CONTEXTS' }),
    chrome.storage.session.setAccessLevel?.({ accessLevel: 'TRUSTED_CONTEXTS' })
  ]);
  const localValues = await chrome.storage.local.get([MODEL_SETTINGS_KEY, PRIVACY_CONSENT_KEY]);
  try {
    state.settings = localValues[MODEL_SETTINGS_KEY]
      ? validateModelSettings(localValues[MODEL_SETTINGS_KEY])
      : defaultSettings();
  } catch {
    state.settings = defaultSettings();
  }
  state.privacyConsent = isCurrentPrivacyConsent(localValues[PRIVACY_CONSENT_KEY])
    ? localValues[PRIVACY_CONSENT_KEY]
    : null;
  installPrivacyActions();
  renderPrivacyExperience();
  if (state.privacyConsent) await startOperationalControllers();
};

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    void initialize().catch((error) => {
      showToast(readableError(error), 'error');
    });
  }, { once: true });
}
