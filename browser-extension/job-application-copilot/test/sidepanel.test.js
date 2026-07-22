import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { IDBFactory } from 'fake-indexeddb';
import { createDocumentVaultRecord } from '../src/parsers/import-record.js';
import { buildEvidencePack } from '../src/rag/retrieval.js';
import {
  activeHttpTab,
  appendFieldClarificationEvidence,
  clarificationRegenerationFailureMessage,
  clarificationQuestionForProposal,
  CUSTOM_PROFILE_CATEGORIES,
  FACT_DEFINITIONS,
  applicationPreflightSummary,
  applicationStructureSignature,
  canCommitGeneration,
  canFillFieldAcrossRevision,
  canonicalApplicationStructure,
  bulkFillResultTone,
  carrySourceSelection,
  createBestEffortExperienceProposal,
  createCustomProfileRecord,
  createEligibilityProposal,
  createFieldClarificationRecord,
  createLiveJobRequirementRecord,
  createPostedSalaryProposal,
  customProfileRecords,
  deriveEligibilityAnswer,
  eligibilityQuestionKind,
  ensureExperienceCandidateEvidence,
  evidenceForModelField,
  evidenceForModelFields,
  evidenceConfidenceForProposal,
  explicitFieldFillConfirmation,
  fieldMetaLabel,
  fieldTypeLabel,
  executeBulkFillItems,
  formatBytes,
  hasAccountGateContext,
  planBulkFill,
  postedSalaryRangeFromText,
  createVerifiedFactProposal,
  factAppliesToApplication,
  factKeyForFieldLabel,
  filterSelectedDocumentRecords,
  generationCacheSeed,
  isClearlyUnsupportedOllamaVersion,
  isDegradedFallbackProposal,
  isModelInstalled,
  isNarrativeExperienceField,
  isCopyOnlyField,
  matchVerifiedFactFieldLabel,
  mergeApplicationPacketContext,
  mergeDeterministicFacts,
  modelEligibleFields,
  normalizeCustomLinkUrl,
  normalizeScanDiscovery,
  prioritizeEvidenceResults,
  parseTrackerTags,
  proposalDisclosesEvidenceGap,
  prepareSanitizedFixtureExport,
  preparedChunkMode,
  proposalsMatchFieldIds,
  readDocumentBytes,
  recoverAbstainedClarifiedExperienceProposal,
  resolveBlankFeedbackRegenerationProposal,
  repairDocumentMetadataFromRetainedBytes,
  retrievalCacheSeed,
  runtimeAnswerValue,
  scanSnapshotKey,
  scanSupportPresentation,
  sourceSelectionScope,
  withEligibilityEvidence,
  verifiedFactChanges,
  verifiedProfileValues,
  validateImportFile,
  validateSanitizedFixture,
  validateModelSettings
} from '../src/sidepanel/sidepanel.js';
import { bytesToBase64, sha256Base64Url } from '../src/vault/crypto.js';
import { EncryptedIndexedDbVault } from '../src/vault/indexeddb-vault.js';
import { MemorySessionKeyStore } from '../src/vault/session-key-store.js';

const packageDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const requestResult = request => new Promise((resolve, reject) => {
  request.addEventListener('success', () => resolve(request.result), { once: true });
  request.addEventListener('error', () => reject(request.error), { once: true });
});

const rawRecords = async (indexedDB, databaseName) => {
  const database = await requestResult(indexedDB.open(databaseName));
  const records = await requestResult(database.transaction('records', 'readonly').objectStore('records').getAll());
  database.close();
  return records;
};

test('sidepanel presents field metadata in plain language', () => {
  assert.equal(fieldTypeLabel('textarea'), 'long answer');
  assert.equal(fieldTypeLabel('select-one'), 'dropdown');
  assert.equal(fieldMetaLabel({ type: 'email', required: true }), 'Required - email');
  assert.equal(fieldMetaLabel({ type: 'vendor-widget', required: false }), 'Optional - custom field');
  assert.equal(explicitFieldFillConfirmation({ riskClass: 'F1_VERIFIED' }), false);
  assert.equal(explicitFieldFillConfirmation({ riskClass: 'F2_REVIEW' }), true);
});

test('sidepanel matches exact and safely qualified verified-fact labels', () => {
  assert.deepEqual(matchVerifiedFactFieldLabel('Email address *'), { key: 'email', matchKind: 'exact_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Candidate email'), { key: 'email', matchKind: 'exact_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Preferred First Name'), { key: 'first_name', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('What is your preferred first name?'), { key: 'first_name', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Current State'), { key: 'state_province', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Current state of residence'), { key: 'state_province', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('In which US State do you currently reside?'), { key: 'state_province', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('LinkedIn Profile (optional)'), { key: 'linkedin', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Website'), { key: 'website', matchKind: 'exact_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Portfolio Website'), { key: 'portfolio', matchKind: 'exact_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('Current city'), { key: 'city', matchKind: 'qualified_alias' });
  assert.deepEqual(matchVerifiedFactFieldLabel('What city do you currently reside in?'), { key: 'city', matchKind: 'qualified_alias' });
  assert.equal(factKeyForFieldLabel('Preferred First Name'), 'first_name');
  assert.equal(factKeyForFieldLabel('In which US State do you currently reside?'), 'state_province');
  assert.ok(Object.hasOwn(FACT_DEFINITIONS, 'linkedin'));
  assert.ok(Object.hasOwn(FACT_DEFINITIONS, 'postal_code'));
  assert.ok(Object.hasOwn(FACT_DEFINITIONS, 'website'));
});
test('sidepanel rejects ambiguous and third-party verified-fact labels', () => {
  [
    'Preferred Name',
    'Preferred State',
    'State of employment',
    'Supervisor first name',
    'Hiring manager email',
    'Reference phone number',
    'Employer city',
    'Work email',
    'Work location preference',
    'Billing city',
    'Shipping address line 1',
    'Country of citizenship'
  ].forEach(label => assert.equal(matchVerifiedFactFieldLabel(label), null, label));
  assert.equal(factKeyForFieldLabel('Supervisor email'), null);
  assert.equal(factKeyForFieldLabel('Reference phone number'), null);
  assert.equal(factKeyForFieldLabel('Work location preference'), null);
});

test('custom links match exactly while custom thoughts remain model-only evidence', () => {
  const link = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.LINK,
    label: 'Tableau Public profile',
    value: 'https://public.tableau.com/views/example',
    recordToken: 'tableau01',
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  const thought = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Salary expectation',
    value: 'I am targeting a base salary in the posted range.',
    existingFacts: [link],
    recordToken: 'salary001',
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  assert.equal(link.value.value, 'https://public.tableau.com/views/example');
  assert.deepEqual(
    matchVerifiedFactFieldLabel('Tableau Public profile (optional)', [link, thought]),
    { key: link.value.key, matchKind: 'custom_link_alias' }
  );
  assert.equal(factKeyForFieldLabel('Salary expectation', [link, thought]), null);
  assert.deepEqual(customProfileRecords([thought, link]).map(record => record.value.category), [
    CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    CUSTOM_PROFILE_CATEGORIES.LINK
  ]);
  assert.equal(normalizeCustomLinkUrl('https://example.com/profile'), 'https://example.com/profile');
  assert.throws(() => normalizeCustomLinkUrl('javascript:alert(1)'), /HTTP\(S\)/u);
  assert.throws(() => createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.LINK,
    label: 'Website',
    value: 'https://example.com',
    recordToken: 'website01'
  }), /built-in profile field/u);
  assert.throws(() => createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Visa status',
    value: 'Do not include this.',
    recordToken: 'sensitive01'
  }), /must remain manual/u);
  assert.throws(() => createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Other context',
    value: 'My visa status should never enter a model prompt.',
    recordToken: 'sensitive02'
  }), /content must remain manual/u);
  assert.throws(() => createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Sponsorship',
    value: 'A dedicated deterministic selector handles this.',
    recordToken: 'sensitive03'
  }), /must remain manual/u);
  const benignThought = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Industry context',
    value: 'I sponsored an analytics initiative for a health reporting team.',
    existingFacts: [link, thought],
    recordToken: 'benign001'
  });
  assert.match(benignThought.value.value, /health reporting team/u);
  const employerThought = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Payments experience at Visa',
    value: 'I have 3 years of payments analytics experience at Visa.',
    existingFacts: [link, thought, benignThought],
    recordToken: 'visaemployer01'
  });
  assert.match(employerThought.value.value, /experience at Visa/u);
  assert.throws(() => createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.LINK,
    label: 'Tableau Public profile',
    value: 'https://example.com/duplicate',
    existingFacts: [link],
    recordToken: 'duplicate01'
  }), /already exists/u);
});

test('live job posting becomes bounded application-scoped requirement evidence', () => {
  const record = createLiveJobRequirementRecord({
    pageId: 'page-live',
    job: {
      title: 'Data Analyst',
      company: 'Example',
      location: 'Denver, CO',
      jobUrl: 'https://jobs.example/123',
      description: 'Responsibilities include analytics. Salary range: $65,000 - $85,000 USD.'
    }
  });
  assert.equal(record.document.id, 'live-job:page-live');
  assert.equal(record.document.sourceRole, 'job_requirement');
  assert.equal(record.document.applicationId, 'page-live');
  assert.equal(record.document.sourceUrl, 'https://jobs.example/123');
  assert.match(record.text, /Salary range: \$65,000 - \$85,000 USD/u);
  assert.equal(postedSalaryRangeFromText(record.text), '$65,000 - $85,000 USD');
  assert.equal(postedSalaryRangeFromText('Worked from 2020 - 2024.'), '');
  assert.equal(createLiveJobRequirementRecord({ pageId: 'page-live', job: { description: '' } }), null);
});

test('posted salary fallback uses only field-scoped job evidence and remains review-only', () => {
  const field = {
    fieldId: 'salary',
    label: 'What are your salary and additional compensation expectations?',
    type: 'textarea',
    riskClass: 'F2_REVIEW'
  };
  const evidence = {
    citations: [{
      citationId: 'c-job',
      sourceRole: 'job_requirement',
      documentId: 'live-job:page-live',
      text: 'The posted salary range is $65,000 - $85,000 USD.'
    }, {
      citationId: 'c-other',
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      text: 'Analytics background.'
    }],
    byField: { salary: ['c-job', 'c-other'] }
  };
  const proposal = createPostedSalaryProposal({ field, evidence });
  assert.equal(proposal.action, 'fill');
  assert.equal(proposal.confidence, 'review');
  assert.deepEqual(proposal.citation_ids, ['c-job']);
  assert.match(proposal.value, /\$65,000 - \$85,000 USD/u);
  assert.equal(createPostedSalaryProposal({
    field,
    evidence: { ...evidence, byField: { salary: ['c-other'] } }
  }), null);
  assert.equal(createPostedSalaryProposal({ field: { ...field, type: 'number' }, evidence }), null);
  assert.equal(createPostedSalaryProposal({
    field: { ...field, maxLength: 36 },
    evidence
  }).value, '$65,000 - $85,000 USD');
  assert.equal(createPostedSalaryProposal({
    field: { ...field, maxLength: 10 },
    evidence
  }), null);
});


test('work eligibility is U.S.-scoped and excludes identity or compliance questions', () => {
  assert.equal(
    eligibilityQuestionKind('Are you authorized to work in the United States without sponsorship?'),
    'authorization_without_sponsorship'
  );
  assert.equal(
    eligibilityQuestionKind('Will you now or in the future require U.S. employment sponsorship?'),
    'sponsorship_now_or_future'
  );
  assert.equal(eligibilityQuestionKind('Are you authorized to work in the US?'), 'authorization');
  assert.equal(eligibilityQuestionKind('Are you authorized to work in the U.S.?'), 'authorization');
  assert.equal(eligibilityQuestionKind('Are you authorized to work in USA?'), 'authorization');
  assert.equal(eligibilityQuestionKind('Are you authorized to work in Canada?'), 'manual_scope');
  assert.equal(eligibilityQuestionKind('Will you require sponsorship in the future?'), 'manual_scope');
  assert.equal(eligibilityQuestionKind('What is your U.S. citizenship or visa status?'), null);
  assert.equal(eligibilityQuestionKind('Complete your U.S. Form I-9 or E-Verify information'), null);
  assert.equal(eligibilityQuestionKind('Are you a U.S. person under ITAR export-control rules?'), null);
});

test('work eligibility derivation fails closed when a required explicit selection is missing', () => {
  assert.deepEqual(deriveEligibilityAnswer({
    kind: 'authorization_without_sponsorship',
    values: { authorized_to_work_us: 'yes', sponsorship_now: 'no' }
  }), { value: 'yes', factKeys: ['authorized_to_work_us', 'sponsorship_now'] });
  assert.deepEqual(deriveEligibilityAnswer({
    kind: 'sponsorship_now_or_future',
    values: { sponsorship_now: 'no', sponsorship_future: 'yes' }
  }), { value: 'yes', factKeys: ['sponsorship_future'] });
  assert.equal(deriveEligibilityAnswer({
    kind: 'sponsorship_now_or_future',
    values: { sponsorship_now: 'no' }
  }), null);
  assert.deepEqual(deriveEligibilityAnswer({
    kind: 'authorization_without_sponsorship_now_or_future',
    values: {
      authorized_to_work_us: 'yes',
      sponsorship_now: 'no',
      sponsorship_future: 'no'
    }
  }), {
    value: 'yes',
    factKeys: ['authorized_to_work_us', 'sponsorship_now', 'sponsorship_future']
  });
  assert.equal(deriveEligibilityAnswer({
    kind: 'authorization_without_sponsorship_now_or_future',
    values: { authorized_to_work_us: 'yes', sponsorship_now: 'no' }
  }), null);
});

test('work eligibility proposals are deterministic, cited, and excluded from Ollama', () => {
  const field = {
    fieldId: 'work-auth',
    label: 'Are you authorized to work in the United States without sponsorship?',
    type: 'select-one',
    options: ['Please select', 'Yes', 'No'],
    riskClass: 'F2_REVIEW'
  };
  const facts = [{
    id: 'fact:authorized_to_work_us',
    updatedAt: '2026-07-19T12:00:00.000Z',
    value: { key: 'authorized_to_work_us', value: 'yes', verifiedAt: '2026-07-19T12:00:00.000Z' }
  }, {
    id: 'fact:sponsorship_now',
    updatedAt: '2026-07-19T12:00:00.000Z',
    value: { key: 'sponsorship_now', value: 'no', verifiedAt: '2026-07-19T12:00:00.000Z' }
  }];
  const evidence = withEligibilityEvidence({
    evidence: {
      citations: [{
        citationId: 'resume-citation',
        documentId: 'resume',
        chunkId: 'resume:1',
        sourceRole: 'candidate_evidence',
        text: 'Candidate evidence.'
      }],
      byField: { 'work-auth': ['resume-citation'] }
    },
    fields: [field],
    facts
  });
  const proposal = createEligibilityProposal({ field, facts, evidence });
  assert.equal(proposal.action, 'fill');
  assert.equal(proposal.confidence, 'review');
  assert.deepEqual(proposal.selected_values, ['Yes']);
  assert.equal(proposal.citation_ids.length, 2);
  assert.deepEqual(modelEligibleFields([field]), []);
  assert.deepEqual(modelEligibleFields([{
    ...field,
    fieldId: 'generic-sponsorship',
    label: 'Will you require sponsorship in the future?'
  }]), []);

  const modelEvidence = evidenceForModelField(evidence, field.fieldId);
  assert.deepEqual(modelEvidence.citations.map(citation => citation.citationId), ['resume-citation']);
  assert.deepEqual(modelEvidence.byField[field.fieldId], ['resume-citation']);

  const merged = mergeDeterministicFacts({
    page_id: 'page',
    proposals: []
  }, {
    pageId: 'page',
    fields: [field]
  }, evidence, facts);
  assert.deepEqual(merged.proposals[0].selected_values, ['Yes']);

  assert.equal(createEligibilityProposal({
    field: {
      ...field,
      options: ['Yes - I am a U.S. citizen or permanent resident', 'No']
    },
    facts,
    evidence
  }), null);
});

test('evidence confidence is deterministic and distinguishes verified, strong, partial, and missing', () => {
  assert.equal(evidenceConfidenceForProposal({
    field: { riskClass: 'F1_VERIFIED' },
    proposal: { action: 'fill', confidence: 'high' }
  }).label, 'Verified');
  assert.equal(evidenceConfidenceForProposal({
    field: { riskClass: 'F2_REVIEW' },
    proposal: { action: 'fill', confidence: 'review', value: 'Grounded analytics.' },
    cards: [{ sourceRole: 'candidate_evidence' }]
  }).label, 'Strong evidence');
  assert.equal(evidenceConfidenceForProposal({
    field: { riskClass: 'F2_REVIEW' },
    proposal: { action: 'fill', confidence: 'review', value: 'Within the posted range.' },
    cards: [{ sourceRole: 'job_requirement' }]
  }).label, 'Partial evidence');
  assert.equal(evidenceConfidenceForProposal({
    field: { riskClass: 'F2_REVIEW' },
    proposal: { action: 'ask_user', confidence: 'needs_input' }
  }).label, 'Needs input');
});

test('bulk fill planner needs no field selection and gates consequential drafts once', () => {
  const fields = [{
    fieldId: 'first',
    type: 'text',
    riskClass: 'F1_VERIFIED'
  }, {
    fieldId: 'experience',
    type: 'textarea',
    riskClass: 'F2_REVIEW'
  }, {
    fieldId: 'manual',
    type: 'text',
    riskClass: 'F2_REVIEW',
    manual: true
  }];
  const proposals = [{
    field_id: 'first',
    action: 'fill',
    confidence: 'high',
    risk_class: 'F1_VERIFIED',
    citation_ids: ['c1']
  }, {
    field_id: 'experience',
    action: 'fill',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    citation_ids: ['c2']
  }, {
    field_id: 'manual',
    action: 'fill',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    citation_ids: ['c3']
  }];

  const beforeReview = planBulkFill({ fields, proposals });
  assert.deepEqual(beforeReview.items.map(item => item.field.fieldId), ['first']);
  const afterReview = planBulkFill({ fields, proposals, reviewedConsequential: true });
  assert.deepEqual(afterReview.items.map(item => item.field.fieldId), ['first', 'experience']);
  assert.equal(afterReview.consequentialCount, 1);
  const verifiedOnly = planBulkFill({
    fields,
    proposals,
    reviewedConsequential: true,
    verifiedOnly: true
  });
  assert.deepEqual(verifiedOnly.items.map(item => item.field.fieldId), ['first']);
  const remaining = planBulkFill({
    fields,
    proposals,
    reviewedConsequential: true,
    alreadyFilledIds: new Set(['first'])
  });
  assert.deepEqual(remaining.items.map(item => item.field.fieldId), ['experience']);
});

test('application preflight groups ready, copy-only, missing, clarification, and manual work', () => {
  const scan = {
    fields: [{
      fieldId: 'verified',
      label: 'Email',
      type: 'email',
      riskClass: 'F1_VERIFIED'
    }, {
      fieldId: 'review',
      label: 'Why this role?',
      type: 'textarea',
      riskClass: 'F2_REVIEW'
    }, {
      fieldId: 'copy',
      label: 'Portfolio widget',
      type: 'text',
      riskClass: 'F2_REVIEW',
      fillMode: 'copy_only'
    }, {
      fieldId: 'ask',
      label: 'Salary expectation',
      type: 'text',
      riskClass: 'F2_REVIEW'
    }, {
      fieldId: 'missing',
      label: 'Preferred name',
      type: 'text',
      riskClass: 'F1_VERIFIED'
    }, {
      fieldId: 'manual',
      label: 'Attestation',
      type: 'checkbox',
      riskClass: 'F2_REVIEW',
      manual: true
    }],
    discovery: {
      unsupportedCount: 2,
      exclusionCounts: { F3_SENSITIVE: 1, F4_CONSENT: 1 }
    }
  };
  const proposals = [{
    field_id: 'verified',
    action: 'fill',
    confidence: 'high',
    citation_ids: []
  }, {
    field_id: 'review',
    action: 'fill',
    confidence: 'review',
    citation_ids: []
  }, {
    field_id: 'copy',
    action: 'fill',
    confidence: 'review',
    citation_ids: []
  }, {
    field_id: 'ask',
    action: 'ask_user',
    confidence: 'needs_input',
    abstain_reason: 'Add the minimum acceptable base salary.',
    citation_ids: []
  }];
  assert.deepEqual(applicationPreflightSummary({ scan, proposals }), {
    verifiedReady: 1,
    reviewReady: 1,
    copyOnly: 1,
    needsInput: 2,
    needsClarification: 1,
    manual: 5,
    ready: 2
  });
});

test('saved job context follows a posting into a separate application and manual context can override it', () => {
  const packetRecord = {
    value: {
      applicationId: 'local-app:example',
      lastContextId: 'context:posting',
      identity: {
        company: 'example analytics',
        title: 'data analyst',
        location: 'denver co'
      },
      contextSnapshots: [{
        contextId: 'context:posting',
        content: 'Build trustworthy reporting and explain performance trends.',
        source: {
          origin: 'live_page',
          mode: 'posting',
          url: 'https://jobs.example/roles/123'
        },
        job: {
          company: 'Example Analytics',
          title: 'Data Analyst',
          location: 'Denver, CO',
          source: 'Company site',
          jobUrl: 'https://jobs.example/roles/123'
        }
      }]
    }
  };
  const applicationScan = {
    pageId: 'ats-form',
    fields: [{ fieldId: 'email' }],
    job: {
      company: '',
      title: '',
      location: '',
      source: 'Greenhouse',
      jobUrl: 'https://boards.greenhouse.io/example/jobs/123'
    }
  };
  const carried = mergeApplicationPacketContext({ scan: applicationScan, packetRecord });
  assert.equal(carried.job.company, 'Example Analytics');
  assert.equal(carried.job.title, 'Data Analyst');
  assert.equal(carried.job.description, 'Build trustworthy reporting and explain performance trends.');
  assert.equal(carried.job.jobUrl, 'https://boards.greenhouse.io/example/jobs/123');

  const manualRecord = {
    value: {
      ...packetRecord.value,
      lastContextId: 'context:manual',
      contextSnapshots: [{
        contextId: 'context:manual',
        content: 'Corrected requirements selected by the applicant.',
        source: { origin: 'manual', mode: 'manual', url: 'https://jobs.example/roles/123' },
        job: {
          company: 'Corrected Company',
          title: 'Senior Data Analyst',
          location: 'Remote',
          source: 'Manual',
          jobUrl: 'https://jobs.example/roles/123'
        }
      }]
    }
  };
  const corrected = mergeApplicationPacketContext({
    scan: { ...applicationScan, job: { ...applicationScan.job, company: 'Page Company' } },
    packetRecord: manualRecord,
    preferPacket: true
  });
  assert.equal(corrected.job.company, 'Corrected Company');
  assert.equal(corrected.job.title, 'Senior Data Analyst');
  assert.equal(corrected.job.description, 'Corrected requirements selected by the applicant.');
});
test('one verified-profile save adds, edits, clears, trims, and skips unchanged facts', () => {
  const existingFacts = [{
    id: 'fact:email',
    value: { key: 'email', label: 'Email address', value: 'candidate@example.com', verifiedAt: 'older' }
  }, {
    id: 'fact:phone',
    value: { key: 'phone', label: 'Phone number', value: '555-0100', verifiedAt: 'older' }
  }, {
    id: 'fact:city',
    value: { key: 'city', label: 'City', value: 'Denver', verifiedAt: 'older' }
  }];
  const profile = verifiedProfileValues(existingFacts);
  assert.equal(Object.keys(profile).length, Object.keys(FACT_DEFINITIONS).length);
  assert.equal(profile.email, 'candidate@example.com');
  assert.equal(profile.github, '');

  const changes = verifiedFactChanges({
    values: {
      email: 'candidate@example.com',
      phone: '   ',
      first_name: '  Ada  ',
      city: 'Boulder',
      github: '  https://github.com/ada  ',
      unsupported_secret: 'must not be stored'
    },
    existingFacts,
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  assert.deepEqual(changes.upserts.map(record => record.id), ['fact:first_name', 'fact:city', 'fact:github']);
  assert.deepEqual(changes.deletes, ['fact:phone']);
  assert.equal(changes.savedCount, 4);
  assert.equal(changes.changedCount, 4);
  assert.equal(changes.upserts[0].value.value, 'Ada');
  assert.equal(changes.upserts[0].value.sourceRole, 'user_verified');
  assert.equal(changes.upserts[0].value.verifiedAt, '2026-07-19T12:00:00.000Z');
  assert.equal(changes.upserts.some(record => record.value.key === 'unsupported_secret'), false);

  const noOp = verifiedFactChanges({ values: verifiedProfileValues(existingFacts), existingFacts });
  assert.deepEqual(noOp.upserts, []);
  assert.deepEqual(noOp.deletes, []);
  assert.equal(noOp.changedCount, 0);
  assert.throws(() => verifiedFactChanges({
    values: { portfolio: 'x'.repeat(2001) },
    existingFacts
  }), /2000 characters or fewer/u);
});
test('employment-eligibility profile facts accept only explicit Yes or No selections', () => {
  const changes = verifiedFactChanges({
    values: {
      authorized_to_work_us: ' YES ',
      sponsorship_now: 'No',
      sponsorship_future: ''
    },
    existingFacts: [],
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  assert.deepEqual(changes.upserts.map(record => [
    record.value.key,
    record.value.value
  ]), [
    ['authorized_to_work_us', 'yes'],
    ['sponsorship_now', 'no']
  ]);
  assert.throws(() => verifiedFactChanges({
    values: { sponsorship_future: 'maybe' },
    existingFacts: []
  }), /must be Yes, No, or left unanswered/u);
});


test('source selection and application scope prevent old-document contamination', () => {
  const records = [{ id: 'current', value: { document: { applicationId: 'page-a' } } }, {
    id: 'global-selected', value: { document: { applicationId: null } }
  }, {
    id: 'old-resume', value: { document: { applicationId: 'page-old' } }
  }, {
    id: 'unselected', value: { document: { applicationId: null } }
  }];
  assert.deepEqual(
    filterSelectedDocumentRecords(records, new Set(['current', 'global-selected', 'old-resume']), 'page-a').map(record => record.id),
    ['current', 'global-selected']
  );
  assert.deepEqual(filterSelectedDocumentRecords(records, new Set(), 'page-a'), []);

  const common = { sourceSignature: 'selected-source-signature', embeddingModel: 'nomic-embed-text' };
  assert.notEqual(
    retrievalCacheSeed({ ...common, applicationId: 'page-a' }),
    retrievalCacheSeed({ ...common, applicationId: 'page-b' })
  );
});

test('navigation carries only selected reusable candidate material into a new application scope', () => {
  const records = [{ id: 'resume', value: { document: { sourceRole: 'candidate_evidence' } } }, {
    id: 'verified-material', value: { document: { sourceRole: 'user_verified' } }
  }, {
    id: 'cover-style', value: { document: { sourceRole: 'style_example' } }
  }, {
    id: 'position-a', value: { document: { sourceRole: 'job_requirement' } }
  }, {
    id: 'company-a', value: { document: { sourceRole: 'company_context' } }
  }];
  const selectedOnPageA = new Set(records.map(record => record.id));
  assert.deepEqual(
    [...carrySourceSelection({ selectedIds: selectedOnPageA, records, fromUnscoped: true })],
    ['resume', 'verified-material', 'cover-style', 'position-a', 'company-a']
  );
  assert.deepEqual(
    [...carrySourceSelection({ selectedIds: selectedOnPageA, records, fromUnscoped: false })].sort(),
    ['cover-style', 'resume', 'verified-material']
  );
  assert.deepEqual(
    [...carrySourceSelection({ selectedIds: new Set(['company-a']), records, fromUnscoped: false })].sort(),
    ['cover-style', 'resume', 'verified-material']
  );

  const common = { pageId: 'page-shared', urlHash: 'url-shared', adapter: 'generic' };
  const pageA = { ...common, domRevision: 1, job: { company: 'Acme', title: 'Analyst', jobUrl: 'https://jobs.example/a', location: '', source: 'Company site' } };
  const pageB = { ...common, domRevision: 1, job: { company: 'Beta', title: 'Analyst', jobUrl: 'https://jobs.example/b', location: '', source: 'Company site' } };
  assert.notEqual(sourceSelectionScope(pageA), sourceSelectionScope(pageB));
  assert.equal(sourceSelectionScope(pageA), sourceSelectionScope({ ...pageA, domRevision: 9, fields: [{ fieldId: 'changed' }] }));
});

test('generation cache binds to canonical exact job and field structure', async () => {
  const scan = {
    pageId: 'page-one',
    urlHash: 'url-one',
    domRevision: 4,
    adapter: 'applicantpro',
    captchaPresent: false,
    job: {
      company: 'Acme',
      title: 'Data Analyst',
      jobUrl: 'https://jobs.example/one',
      location: 'Denver, CO',
      source: 'ApplicantPro'
    },
    fields: [{
      fieldId: 'field-email',
      fingerprint: 'aaaaaaaaaaaaaaaa',
      label: 'Email address',
      type: 'email',
      options: [],
      nearbyText: 'Applicant contact',
      required: true,
      riskClass: 'F1_VERIFIED'
    }, {
      fieldId: 'field-summary',
      fingerprint: 'bbbbbbbbbbbbbbbb',
      label: 'Why this role?',
      type: 'textarea',
      options: [],
      nearbyText: 'Maximum 500 characters',
      required: true,
      riskClass: 'F2_REVIEW',
      maxLength: 500
    }]
  };
  const reorderedKeys = {
    fields: scan.fields.map(field => ({ ...field })),
    job: { source: 'ApplicantPro', title: 'Data Analyst', company: 'Acme', location: 'Denver, CO', jobUrl: 'https://jobs.example/one' },
    adapter: 'applicantpro',
    urlHash: 'url-one',
    pageId: 'page-one',
    captchaPresent: false,
    domRevision: 4
  };
  assert.equal(canonicalApplicationStructure(scan), canonicalApplicationStructure(reorderedKeys));
  const signature = await applicationStructureSignature(scan);
  assert.equal(signature, await applicationStructureSignature(reorderedKeys));
  assert.notEqual(signature, await applicationStructureSignature({
    ...scan,
    job: { ...scan.job, company: 'Different company' }
  }));
  assert.notEqual(signature, await applicationStructureSignature({
    ...scan,
    fields: scan.fields.map((field, index) => index ? { ...field, nearbyText: 'Changed constraint' } : field)
  }));

  const proposals = scan.fields.map(field => ({ field_id: field.fieldId }));
  assert.equal(proposalsMatchFieldIds(proposals, scan.fields), true);
  assert.equal(proposalsMatchFieldIds(proposals.slice(0, 1), scan.fields), false);
  assert.equal(proposalsMatchFieldIds([...proposals, { field_id: 'field-unknown' }], scan.fields), false);
  assert.equal(proposalsMatchFieldIds([{ field_id: 'field-email' }, { field_id: 'field-email' }], scan.fields), false);

  const baseSeed = {
    sourceSignature: 'sources',
    generationModel: 'qwen3.5:27b',
    fallbackGenerationModel: 'qwen3:8b',
    embeddingModel: 'nomic-embed-text'
  };
  assert.notEqual(
    generationCacheSeed({ ...baseSeed, structureSignature: signature }),
    generationCacheSeed({ ...baseSeed, structureSignature: await applicationStructureSignature({ ...scan, job: { ...scan.job, title: 'Senior Data Analyst' } }) })
  );
});

test('free-format scan diagnostics explain bounded unsupported contexts', () => {
  const legacy = normalizeScanDiscovery({
    fields: [{ fieldId: 'field-one' }],
    exclusionCounts: { F0_EXCLUDED: 2, F3_SENSITIVE: 1, F4_CONSENT: 3 },
    truncated: false
  });
  assert.equal(legacy.available, false);
  assert.equal(legacy.mode, 'standard');
  assert.equal(legacy.recognizedCount, 1);
  assert.equal(legacy.exclusionCounts.F3_SENSITIVE, 1);

  const expectedCopy = {
    cross_origin_iframe: /own tab/iu,
    closed_shadow_root: /closed page component/iu,
    custom_aria_widget: /cop(?:y|ied)/iu,
    account_gate: /sign-in or account-creation/iu
  };
  for (const [kind, copyPattern] of Object.entries(expectedCopy)) {
    const scan = {
      fields: kind === 'account_gate' ? [] : [{ fieldId: 'field-one', fillMode: kind === 'custom_aria_widget' ? 'copy_only' : undefined }],
      discovery: {
        mode: kind === 'custom_aria_widget' ? 'free_format' : 'limited',
        recognizedCount: kind === 'account_gate' ? 0 : 1,
        unsupportedCount: 1,
        exclusionCounts: { F0_EXCLUDED: 0, F3_SENSITIVE: 0, F4_CONSENT: 0 },
        contexts: [{ kind, count: 1, status: kind === 'account_gate' ? 'manual' : 'unsupported' }],
        truncated: false
      }
    };
    assert.equal(hasAccountGateContext(scan), kind === 'account_gate');
    const presentation = scanSupportPresentation(scan);
    assert.equal(presentation.showNotice, true);
    assert.match(presentation.copy, copyPattern);
    assert.match(presentation.status, /1 unsupported/iu);
    if (kind === 'custom_aria_widget') assert.match(presentation.status, /1 copy-only/iu);
  }
});

test('copy-only custom fields remain answerable but are excluded from all bulk fill plans', () => {
  const fields = [{
    fieldId: 'field-native',
    riskClass: 'F1_VERIFIED'
  }, {
    fieldId: 'field-custom',
    riskClass: 'F2_REVIEW',
    fillMode: 'copy_only'
  }];
  const proposals = [{
    field_id: 'field-native',
    action: 'fill',
    confidence: 'high',
    risk_class: 'F1_VERIFIED',
    citation_ids: ['candidate-email']
  }, {
    field_id: 'field-custom',
    action: 'fill',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    citation_ids: ['resume-project']
  }];
  assert.equal(isCopyOnlyField(fields[1]), true);
  assert.equal(isCopyOnlyField(fields[0]), false);
  assert.deepEqual(
    planBulkFill({ fields, proposals, reviewedConsequential: true }).items.map(item => item.field.fieldId),
    ['field-native']
  );
});

test('sanitized diagnostic fixtures whitelist structure and add an opaque stable signature', async () => {
  const legacyFixture = {
    schemaVersion: 1,
    adapter: 'generic',
    captchaPresent: false,
    fields: [{
      fieldId: 'field-0123456789abcdef',
      label: 'Portfolio URL',
      type: 'url',
      options: [],
      riskClass: 'F1_VERIFIED'
    }]
  };
  assert.deepEqual(validateSanitizedFixture(legacyFixture), legacyFixture);

  const fixture = structuredClone(legacyFixture);
  fixture.fields[0].fillMode = 'copy_only';
  fixture.discovery = {
    mode: 'free_format',
    recognizedCount: 1,
    unsupportedCount: 2,
    exclusionCounts: { F0_EXCLUDED: 4, F3_SENSITIVE: 1, F4_CONSENT: 2 },
    contexts: [
      { kind: 'custom_aria_widget', count: 1, status: 'manual' },
      { kind: 'cross_origin_iframe', count: 1, status: 'unsupported' }
    ],
    truncated: false
  };
  const first = await prepareSanitizedFixtureExport(fixture);
  const second = await prepareSanitizedFixtureExport(structuredClone(fixture));
  assert.match(first.structureSignature, /^[A-Za-z0-9_-]{43}$/u);
  assert.equal(first.structureSignature, second.structureSignature);
  assert.equal(first.fields[0].fillMode, 'copy_only');
  assert.notEqual(
    first.structureSignature,
    (await prepareSanitizedFixtureExport({
      ...fixture,
      discovery: { ...fixture.discovery, unsupportedCount: 3 }
    })).structureSignature
  );

  const invalidFixtures = [
    { ...fixture, pageUrl: 'https://jobs.example/private-token' },
    { ...fixture, fields: [{ ...fixture.fields[0], value: 'entered answer' }] },
    {
      ...fixture,
      discovery: {
        ...fixture.discovery,
        contexts: [{ kind: 'account_gate', count: 1, status: 'manual', pageText: 'private login prompt' }]
      }
    },
    {
      ...fixture,
      discovery: {
        ...fixture.discovery,
        exclusionCounts: { ...fixture.discovery.exclusionCounts, hiddenFieldNames: 2 }
      }
    }
  ];
  invalidFixtures.forEach(value => assert.throws(
    () => validateSanitizedFixture(value),
    /privacy contract/iu
  ));
});
test('prepared chunk mode preserves lexical fallback and tolerates cache misses', () => {
  assert.equal(preparedChunkMode({ cacheHit: false, cachedValue: undefined }), 'lexical');
  assert.equal(preparedChunkMode({ cacheHit: false, cachedValue: { mode: 'hybrid' } }), 'lexical');
  assert.equal(preparedChunkMode({ cacheHit: true, cachedValue: { mode: 'lexical' } }), 'lexical');
  assert.equal(preparedChunkMode({ cacheHit: true, cachedValue: { mode: 'unexpected' } }), 'lexical');
  assert.equal(preparedChunkMode({ cacheHit: true, cachedValue: { mode: 'hybrid' } }), 'hybrid');
});

test('panel helpers preserve field answer shapes, exact scan revisions, and tracker limits', () => {
  const selected = { value_type: 'selected_values', selected_values: ['Colorado'], value: '', checked: false };
  assert.equal(runtimeAnswerValue(selected, { type: 'select-one' }), 'Colorado');
  assert.deepEqual(runtimeAnswerValue(selected, { type: 'select-multiple' }), ['Colorado']);
  assert.equal(scanSnapshotKey({ pageId: 'p', urlHash: 'u', domRevision: 3 }), 'p:u:3');
  assert.deepEqual(parseTrackerTags('Analytics, Remote, analytics'), ['Analytics', 'Remote']);
  assert.throws(() => parseTrackerTags(Array.from({ length: 13 }, (_, index) => `tag-${index}`).join(',')), /at most 12/u);
  assert.deepEqual(validateModelSettings({
    generationModel: 'qwen3.5:27b',
    fallbackGenerationModel: 'qwen3:8b',
    embeddingModel: 'nomic-embed-text'
  }), {
    generationModel: 'qwen3.5:27b',
    fallbackGenerationModel: 'qwen3:8b',
    embeddingModel: 'nomic-embed-text'
  });
  assert.throws(() => validateModelSettings({
    generationModel: 'bad model name',
    fallbackGenerationModel: 'qwen3:8b',
    embeddingModel: 'nomic-embed-text'
  }), /valid local Ollama model/u);
});

test('narrative experience retrieval keeps candidate context even without domain keywords', () => {
  const field = {
    fieldId: 'experience',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    nearbyText: '',
    type: 'textarea',
    riskClass: 'F2_REVIEW'
  };
  const resume = {
    id: 'resume-context',
    documentId: 'resume',
    documentVersion: 1,
    applicationId: 'app-1',
    sourceRole: 'candidate_evidence',
    text: 'Built forecasting tools and executive dashboards at Example Organization.',
    terms: ['built', 'forecasting', 'tools', 'executive', 'dashboards', 'example', 'organization'],
    locator: { pageStart: 1 },
    quoteHash: 'resume-hash'
  };
  const job = {
    id: 'job-domain',
    documentId: 'job',
    documentVersion: 1,
    applicationId: 'app-1',
    sourceRole: 'job_requirement',
    text: 'Five years of fintech and payments experience requested.',
    terms: ['five', 'years', 'fintech', 'payments', 'experience'],
    locator: {},
    quoteHash: 'job-hash',
    retrieval: { score: 1, vectorScore: 0, lexicalScore: 1, roleScore: 0.85 }
  };
  assert.equal(isNarrativeExperienceField(field), true);
  const results = ensureExperienceCandidateEvidence({
    field,
    results: [job],
    chunks: [job, resume],
    queryText: field.label,
    applicationId: 'app-1'
  });
  assert.equal(results[0].sourceRole, 'candidate_evidence');
  assert.ok(results.some(result => result.id === 'job-domain'));
  const noOverlapResume = {
    ...resume,
    id: 'resume-no-overlap',
    text: 'Coordinated regional programs at Alpine Museum.',
    terms: ['coordinated', 'regional', 'programs', 'alpine', 'museum'],
    quoteHash: 'resume-no-overlap-hash'
  };
  const representativeFallback = ensureExperienceCandidateEvidence({
    field,
    results: [job],
    chunks: [job, noOverlapResume],
    queryText: field.label,
    applicationId: 'app-1'
  });
  assert.equal(representativeFallback[0].id, 'resume-no-overlap');
  assert.equal(representativeFallback[0].retrieval.lexicalScore, 0);
  assert.equal(isNarrativeExperienceField({ ...field, label: 'Salary expectation' }), false);
});

test('experience batch fallback returns a cited low-confidence draft and clarification instead of a placeholder', () => {
  const field = {
    fieldId: 'fintech',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    nearbyText: '',
    type: 'textarea',
    options: [],
    required: true,
    riskClass: 'F2_REVIEW',
    maxLength: 1000
  };
  const evidence = {
    citations: [{
      citationId: 'resume-adjacent',
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      documentVersion: 1,
      chunkId: 'resume-chunk',
      quoteHash: 'resume-hash',
      text: 'Built forecasting tools and executive dashboards for cross-functional leaders.',
      locator: { pageStart: 1 }
    }, {
      citationId: 'job-domain',
      sourceRole: 'job_requirement',
      documentId: 'live-job:fintech',
      documentVersion: 1,
      chunkId: 'job-chunk',
      quoteHash: 'job-hash',
      text: 'Fintech and payments experience is requested.',
      locator: { section: 'Live job posting' }
    }],
    byField: { fintech: ['resume-adjacent', 'job-domain'] }
  };

  const draft = createBestEffortExperienceProposal({ field, evidence });
  assert.equal(draft.action, 'fill');
  assert.equal(draft.confidence, 'review');
  assert.deepEqual(draft.citation_ids, ['resume-adjacent']);
  assert.match(draft.value, /^I don't have a specific number of years of direct fintech\/payments experience to report, but my relevant experience includes/iu);
  assert.match(draft.value, /I built forecasting tools and executive dashboards/iu);
  assert.doesNotMatch(draft.value, /\b(?:supplied materials?|documents?|evidence|resume|sources?|citations?|model|system)\b/iu);
  assert.doesNotMatch(draft.value, /\b\d+\s+years?\b/iu);
  assert.equal(createBestEffortExperienceProposal({
    field,
    evidence: {
      citations: [{
        citationId: 'profile-link',
        sourceRole: 'user_verified',
        documentId: 'profile',
        documentVersion: 1,
        chunkId: 'profile-link-chunk',
        quoteHash: 'profile-link-hash',
        text: 'Website URL: https://www.example.com/portfolio',
        locator: {}
      }],
      byField: { fintech: ['profile-link'] }
    }
  }), null, 'profile links alone are not experience evidence');

  assert.equal(proposalDisclosesEvidenceGap(draft), true);
  assert.equal(evidenceConfidenceForProposal({
    field,
    proposal: draft,
    cards: evidence.citations
  }).label, 'Low confidence');
  assert.match(clarificationQuestionForProposal({
    field,
    proposal: draft,
    cards: evidence.citations
  }), /organization or project, dates, responsibilities, and outcome/iu);
  assert.equal(createBestEffortExperienceProposal({
    field,
    evidence: {
      citations: [evidence.citations[1]],
      byField: { fintech: ['job-domain'] }
    }
  }), null);
  assert.equal(isDegradedFallbackProposal({
    action: 'ask_user',
    confidence: 'needs_input',
    short_rationale: 'No verified answer is available.',
    abstain_reason: 'Review this field and enter the answer yourself.'
  }), true);
});

test('clarified narrative experience abstentions recover without overriding fills or numeric durations', () => {
  const field = {
    fieldId: 'fintech',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    nearbyText: '',
    type: 'textarea',
    options: [],
    required: true,
    riskClass: 'F2_REVIEW',
    maxLength: 1000
  };
  const abstention = {
    field_id: field.fieldId,
    action: 'ask_user',
    confidence: 'needs_input',
    risk_class: field.riskClass,
    value_type: 'none',
    value: '',
    selected_values: [],
    checked: false,
    citation_ids: [],
    short_rationale: 'More information is required.',
    abstain_reason: 'Clarify the duration.'
  };
  const adjacentEvidence = {
    citations: [{
      citationId: 'clarification-no-direct',
      sourceRole: 'user_verified',
      documentId: 'clarification',
      documentVersion: 1,
      chunkId: 'clarification-chunk',
      quoteHash: 'clarification-hash',
      text: 'Clarification - fintech experience [synthetic]: I do not have direct fintech experience. I supported payments-adjacent analytics and reporting projects.',
      locator: {}
    }, {
      citationId: 'resume-adjacent',
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      documentVersion: 1,
      chunkId: 'resume-chunk',
      quoteHash: 'resume-hash',
      text: 'Built forecasting tools and executive dashboards for cross-functional leaders.',
      locator: { pageStart: 1 }
    }],
    byField: { fintech: ['clarification-no-direct', 'resume-adjacent'] }
  };

  const recovered = recoverAbstainedClarifiedExperienceProposal({
    field,
    proposal: abstention,
    evidence: adjacentEvidence,
    clarificationSaved: true
  });
  assert.equal(recovered.action, 'fill');
  assert.equal(recovered.confidence, 'review');
  assert.match(recovered.value, /^I don't have a specific number of years/iu);
  assert.match(recovered.value, /I built forecasting tools/iu);
  assert.match(recovered.value, /I supported payments-adjacent analytics and reporting projects/iu);
  assert.doesNotMatch(recovered.value, /Clarification -/u);
  assert.doesNotMatch(recovered.value, /\[[A-Za-z0-9_-]{1,128}\]/u);
  assert.doesNotMatch(recovered.value, /\n-\s+I (?:do not|don't) have[^.\n]*direct[^.\n]*experience/iu);
  assert.ok(recovered.citation_ids.includes('clarification-no-direct'));
  assert.ok(recovered.citation_ids.includes('resume-adjacent'));

  const successfulFill = { ...recovered, value: 'I have relevant adjacent analytics experience.' };
  assert.strictEqual(recoverAbstainedClarifiedExperienceProposal({
    field,
    proposal: successfulFill,
    evidence: adjacentEvidence,
    clarificationSaved: true
  }), successfulFill);

  for (const text of [
    'I have 3 years of direct fintech experience.',
    'I have 18 months of direct payments experience.'
  ]) {
    const numericEvidence = {
      ...adjacentEvidence,
      citations: adjacentEvidence.citations.map(citation => citation.citationId === 'clarification-no-direct'
        ? { ...citation, text }
        : citation)
    };
    assert.strictEqual(recoverAbstainedClarifiedExperienceProposal({
      field,
      proposal: abstention,
      evidence: numericEvidence,
      clarificationSaved: true
    }), abstention);
  }
});

test('a prior saved field clarification enables retry recovery without treating custom thoughts as clarification', () => {
  const field = {
    fieldId: 'fintech-retry',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    nearbyText: '',
    type: 'textarea',
    riskClass: 'F2_REVIEW',
    maxLength: 1000
  };
  const applicationId = 'application-retry';
  const clarification = createFieldClarificationRecord({
    field,
    applicationId,
    text: 'I do not have direct fintech or payments experience.',
    recordToken: 'clarifyretry01',
    verifiedAt: '2026-07-19T12:15:00.000Z'
  });
  const candidateCitation = {
    citationId: 'retry-resume',
    sourceRole: 'candidate_evidence',
    documentId: 'resume',
    documentVersion: 1,
    chunkId: 'resume-chunk',
    quoteHash: 'resume-hash',
    text: 'Built forecasting tools and executive dashboards for cross-functional leaders.',
    locator: { pageStart: 1 }
  };
  const evidence = {
    citations: [{
      citationId: 'retry-clarification',
      sourceRole: 'user_verified',
      documentId: clarification.id,
      documentVersion: 1,
      chunkId: 'clarification-chunk',
      quoteHash: 'clarification-hash',
      text: `${clarification.value.label}: ${clarification.value.value}`,
      locator: { section: 'Custom profile thought' }
    }, candidateCitation],
    byField: { [field.fieldId]: ['retry-clarification', candidateCitation.citationId] }
  };
  const abstention = {
    field_id: field.fieldId,
    action: 'ask_user',
    confidence: 'needs_input',
    risk_class: field.riskClass,
    value_type: 'none',
    value: '',
    selected_values: [],
    checked: false,
    citation_ids: [],
    short_rationale: 'More information is required.',
    abstain_reason: 'Clarify the duration.'
  };

  const recovered = recoverAbstainedClarifiedExperienceProposal({
    field,
    proposal: abstention,
    evidence,
    clarificationSaved: false,
    facts: [clarification],
    applicationId
  });
  assert.equal(recovered.action, 'fill');
  assert.match(recovered.value, /I built forecasting tools/iu);
  assert.doesNotMatch(recovered.value, /Clarification -/u);
  assert.doesNotMatch(recovered.value, /\n-\s+I (?:do not|don't) have[^.\n]*direct[^.\n]*experience/iu);
  assert.deepEqual(new Set(recovered.citation_ids), new Set(['retry-clarification', 'retry-resume']));

  const customThought = createCustomProfileRecord({
    category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
    label: 'Fintech context',
    value: 'My background includes adjacent analytics work.',
    recordToken: 'thoughtretry01',
    verifiedAt: '2026-07-19T12:16:00.000Z'
  });
  const customThoughtEvidence = {
    citations: evidence.citations.map(citation => citation.citationId === 'retry-clarification'
      ? {
          ...citation,
          documentId: customThought.id,
          text: `${customThought.value.label}: ${customThought.value.value}`
        }
      : citation),
    byField: evidence.byField
  };
  assert.strictEqual(recoverAbstainedClarifiedExperienceProposal({
    field,
    proposal: abstention,
    evidence: customThoughtEvidence,
    clarificationSaved: false,
    facts: [customThought],
    applicationId
  }), abstention);
});

test('blank-feedback abstention fallback retains only the exact current cited fill', () => {
  const prior = {
    field_id: 'fintech',
    action: 'fill',
    confidence: 'review',
    risk_class: 'F2_REVIEW',
    value_type: 'text',
    value: 'I do not have a specific number of years of direct fintech experience to report.',
    selected_values: [],
    checked: false,
    citation_ids: ['resume-adjacent'],
    short_rationale: 'Grounded current answer.',
    abstain_reason: ''
  };
  const abstention = {
    ...prior,
    action: 'ask_user',
    confidence: 'needs_input',
    value_type: 'none',
    value: '',
    citation_ids: [],
    short_rationale: 'User input required.',
    abstain_reason: 'More information is needed.'
  };
  const regeneratedFill = {
    ...prior,
    value: 'A newly supported answer.'
  };

  for (const feedback of [
    { preset: 'none', text: '' },
    { preset: 'shorter', text: '' },
    { preset: 'tone', text: '  \n\t ' }
  ]) {
    const resolved = resolveBlankFeedbackRegenerationProposal({
      priorProposal: prior,
      regeneratedProposal: abstention,
      feedback
    });
    assert.equal(resolved.retained, true);
    assert.equal(resolved.proposal, prior, 'the exact current proposal object must be retained');
  }

  const nonblank = resolveBlankFeedbackRegenerationProposal({
    priorProposal: prior,
    regeneratedProposal: abstention,
    feedback: { preset: 'other', text: 'Emphasize the analytics project.' }
  });
  assert.equal(nonblank.retained, false);
  assert.equal(nonblank.proposal, abstention);

  const priorNeedsInput = resolveBlankFeedbackRegenerationProposal({
    priorProposal: abstention,
    regeneratedProposal: abstention,
    feedback: { preset: 'none', text: '' }
  });
  assert.equal(priorNeedsInput.retained, false);
  assert.equal(priorNeedsInput.proposal, abstention);

  const successfulRevision = resolveBlankFeedbackRegenerationProposal({
    priorProposal: prior,
    regeneratedProposal: regeneratedFill,
    feedback: { preset: 'shorter', text: '' }
  });
  assert.equal(successfulRevision.retained, false);
  assert.equal(successfulRevision.proposal, regeneratedFill);

  const uncitedPrior = { ...prior, citation_ids: [] };
  const unsafeRetention = resolveBlankFeedbackRegenerationProposal({
    priorProposal: uncitedPrior,
    regeneratedProposal: abstention,
    feedback: { preset: 'none', text: '' }
  });
  assert.equal(unsafeRetention.retained, false);
  assert.equal(unsafeRetention.proposal, abstention);
});
test('field clarification is application scoped and becomes replaceable user-verified evidence', async () => {
  const field = {
    fieldId: 'fintech',
    label: 'How many years of fintech/payments experience do you have? Please explain.',
    nearbyText: '',
    type: 'textarea',
    riskClass: 'F2_REVIEW'
  };
  const applicationId = 'application-1';
  const record = createFieldClarificationRecord({
    field,
    applicationId,
    text: 'I do not have direct fintech or payments experience.',
    recordToken: 'clarify01',
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  assert.equal(record.value.applicationId, applicationId);
  assert.equal(record.value.fieldId, field.fieldId);
  assert.equal(factAppliesToApplication(record, applicationId), true);
  assert.equal(factAppliesToApplication(record, 'application-2'), false);
  assert.deepEqual(customProfileRecords([record], CUSTOM_PROFILE_CATEGORIES.THOUGHT), []);
  const saturatedThoughts = Array.from({ length: 24 }, (_, index) => ({
    id: `fact:thought:${index}`,
    value: {
      category: CUSTOM_PROFILE_CATEGORIES.THOUGHT,
      label: `Reusable thought ${index}`,
      value: `Context ${index}`
    }
  }));
  const employerClarification = createFieldClarificationRecord({
    field,
    applicationId,
    text: 'I have 3 years of payments analytics experience at Visa.',
    existingFacts: saturatedThoughts,
    recordToken: 'clarifyvisa01',
    verifiedAt: '2026-07-19T12:01:00.000Z'
  });
  assert.match(employerClarification.value.value, /experience at Visa/u);

  const baseEvidence = {
    citations: [{
      citationId: 'resume',
      sourceRole: 'candidate_evidence',
      documentId: 'resume',
      documentVersion: 1,
      chunkId: 'resume-chunk',
      quoteHash: 'resume-hash',
      text: 'Built analytics dashboards.',
      locator: { pageStart: 1 }
    }, {
      citationId: 'other',
      sourceRole: 'candidate_evidence',
      documentId: 'other',
      documentVersion: 1,
      chunkId: 'other-chunk',
      quoteHash: 'other-hash',
      text: 'Unrelated field evidence.',
      locator: {}
    }],
    byField: {
      fintech: ['resume'],
      otherField: ['other']
    }
  };
  const appended = await appendFieldClarificationEvidence({
    evidence: baseEvidence,
    field,
    record,
    applicationId
  });
  const clarificationCitation = appended.citations.find(citation => citation.documentId === record.id);
  assert.equal(clarificationCitation.sourceRole, 'user_verified');
  assert.match(clarificationCitation.text, /do not have direct fintech or payments experience/iu);
  assert.ok(appended.byField.fintech.includes(clarificationCitation.citationId));
  assert.deepEqual(evidenceForModelFields(appended, ['fintech']).byField.fintech, appended.byField.fintech);
  assert.deepEqual(evidenceForModelFields(appended, ['otherField']).citations.map(citation => citation.citationId), ['other']);

  const replacement = createFieldClarificationRecord({
    field,
    applicationId,
    text: 'I supported one payments-adjacent analytics project in 2025.',
    recordToken: 'clarify01',
    verifiedAt: '2026-07-19T12:05:00.000Z'
  });
  const replaced = await appendFieldClarificationEvidence({
    evidence: appended,
    field,
    record: replacement,
    applicationId
  });
  const replacementCitations = replaced.citations.filter(citation => citation.documentId === replacement.id);
  assert.equal(replacementCitations.length, 1);
  assert.match(replacementCitations[0].text, /payments-adjacent analytics project in 2025/iu);
  assert.doesNotMatch(replacementCitations[0].text, /do not have direct fintech/iu);
  await assert.rejects(() => appendFieldClarificationEvidence({
    evidence: baseEvidence,
    field,
    record,
    applicationId: 'application-2'
  }), /scope does not match/iu);
});

test('field clarification reserves a citation slot when existing evidence is saturated', async () => {
  const field = {
    fieldId: 'fintech',
    label: 'How many years of fintech/payments experience do you have?',
    nearbyText: '',
    type: 'textarea',
    riskClass: 'F2_REVIEW'
  };
  const applicationId = 'application-saturated';
  const record = createFieldClarificationRecord({
    field,
    applicationId,
    text: 'I do not have direct fintech experience, but I have adjacent analytics experience.',
    recordToken: 'clarifysaturated01',
    verifiedAt: '2026-07-19T12:10:00.000Z'
  });
  const citations = Array.from({ length: 12 }, (_, index) => ({
    citationId: `existing-${index}`,
    sourceRole: index === 11 ? 'user_verified' : 'candidate_evidence',
    documentId: `document-${index}`,
    documentVersion: 1,
    chunkId: `chunk-${index}`,
    quoteHash: `hash-${index}`,
    text: `Existing evidence ${index}.`,
    locator: {}
  }));
  const evidence = {
    citations,
    byField: { fintech: citations.map(citation => citation.citationId) }
  };

  const appended = await appendFieldClarificationEvidence({ evidence, field, record, applicationId });
  const fieldIds = appended.byField.fintech;
  const clarificationCitation = appended.citations.find(citation => citation.documentId === record.id);

  assert.equal(fieldIds.length, 12);
  assert.ok(fieldIds.includes(clarificationCitation.citationId));
  assert.ok(fieldIds.includes('existing-11'), 'existing user-verified evidence should be retained first');
  assert.equal(fieldIds.includes('existing-10'), false, 'a non-user-verified slot should be replaced first');
  assert.equal(evidenceForModelField(appended, field.fieldId).citations.length, 12);
});

test('clarification regeneration errors explain that saved input can be retried without re-entry', () => {
  assert.equal(
    clarificationRegenerationFailureMessage({ clarificationSaved: false, error: new Error('Ollama timed out.') }),
    'Ollama timed out.'
  );
  const message = clarificationRegenerationFailureMessage({
    clarificationSaved: true,
    error: new Error('Ollama timed out.')
  });
  assert.match(message, /^Clarification was saved locally/iu);
  assert.match(message, /regenerated answer was not applied/iu);
  assert.match(message, /Retry regeneration/iu);
  assert.match(message, /do not need to enter the clarification again/iu);
  assert.match(message, /Ollama timed out/iu);
});

test('evidence prioritization reserves citations for experience and exact verified links', () => {
  const resultFor = (id, sourceRole = 'candidate_evidence') => ({
    id: `chunk:${id}`,
    documentId: `doc:${id}`,
    documentVersion: 1,
    applicationId: 'app-priority',
    sourceRole,
    text: `Evidence for ${id}.`,
    locator: { section: id },
    quoteHash: `hash:${id}`,
    retrieval: { score: 1, vectorScore: 0, lexicalScore: 1, roleScore: 1 }
  });
  const ordinaryReview = Array.from({ length: 22 }, (_, index) => ({
    fieldId: `review-${index}`,
    riskClass: 'F2_REVIEW',
    narrativeExperience: false,
    results: [resultFor(`review-${index}`)]
  }));
  const website = {
    fieldId: 'website',
    riskClass: 'F1_VERIFIED',
    narrativeExperience: false,
    results: [resultFor('website', 'user_verified')]
  };
  const experience = {
    fieldId: 'experience',
    riskClass: 'F2_REVIEW',
    narrativeExperience: true,
    results: [resultFor('experience')]
  };
  const pack = buildEvidencePack(prioritizeEvidenceResults([...ordinaryReview, website, experience]));
  assert.equal(pack.citations.length, 20);
  assert.equal(pack.byField.website.length, 1);
  assert.equal(pack.byField.experience.length, 1);
});

test('F1 fields are deterministic-only and never enter model generation', () => {
  const emailField = {
    fieldId: 'field-email',
    label: 'Email address',
    type: 'email',
    options: [],
    riskClass: 'F1_VERIFIED'
  };
  const reviewField = { ...emailField, fieldId: 'field-summary', label: 'Why this role?', riskClass: 'F2_REVIEW' };
  assert.deepEqual(modelEligibleFields([emailField, reviewField]), [reviewField]);
  assert.deepEqual(modelEligibleFields([emailField]), []);
  assert.equal(createVerifiedFactProposal({ field: emailField, fact: null, citationId: 'c1' }), null);
  const proposal = createVerifiedFactProposal({
    field: emailField,
    fact: { key: 'email', label: 'Email address', value: 'candidate@example.com' },
    citationId: 'c1'
  });
  assert.equal(proposal.action, 'fill');
  assert.equal(proposal.confidence, 'high');
  assert.equal(proposal.value, 'candidate@example.com');
  assert.deepEqual(proposal.citation_ids, ['c1']);
  assert.equal(createVerifiedFactProposal({
    field: { ...emailField, label: 'Supervisor email' },
    fact: { key: 'email', label: 'Email address', value: 'candidate@example.com' },
    citationId: 'c1'
  }), null);
});

test('generation commits only to its unchanged live application structure', async () => {
  const currentScan = {
    pageId: 'page-a',
    urlHash: 'hash-a',
    domRevision: 2,
    adapter: 'generic',
    captchaPresent: false,
    job: { company: 'Acme', title: 'Analyst', jobUrl: 'https://jobs.example/a', location: 'Denver', source: 'Direct' },
    fields: [{
      fieldId: 'field-summary',
      fingerprint: 'aaaaaaaaaaaaaaaa',
      label: 'Why this role?',
      type: 'textarea',
      options: [],
      nearbyText: '',
      required: true,
      riskClass: 'F2_REVIEW',
      maxLength: 500
    }]
  };
  const expectedScanKey = scanSnapshotKey(currentScan);
  const expectedStructureSignature = await applicationStructureSignature(currentScan);
  const valid = { expectedScanKey, expectedStructureSignature, currentStructureSignature: expectedStructureSignature };
  assert.equal(canCommitGeneration({ ...valid, currentScan, stale: false, aborted: false }), true);
  assert.equal(canCommitGeneration({ ...valid, currentScan, stale: true, aborted: false }), false);
  assert.equal(canCommitGeneration({ ...valid, currentScan, stale: false, aborted: true }), false);
  assert.equal(canCommitGeneration({
    ...valid,
    currentScan: { ...currentScan, domRevision: 3 },
    stale: false,
    aborted: false
  }), false);
  const changedJob = { ...currentScan, job: { ...currentScan.job, title: 'Senior Analyst' } };
  assert.equal(canCommitGeneration({
    ...valid,
    currentScan: changedJob,
    currentStructureSignature: await applicationStructureSignature(changedJob),
    stale: false,
    aborted: false
  }), false);
  assert.equal(canCommitGeneration({ expectedScanKey, currentScan, stale: false, aborted: false }), false);
});

test('fill-time freshness permits unrelated validation changes but rejects a changed target', () => {
  const field = {
    fieldId: 'field-summary',
    fingerprint: 'aaaaaaaaaaaaaaaa',
    label: 'Why this role?',
    type: 'textarea',
    options: [],
    nearbyText: 'Maximum 500 characters',
    required: true,
    riskClass: 'F2_REVIEW',
    maxLength: 500
  };
  const analyzedScan = {
    pageId: 'page-a',
    urlHash: 'hash-a',
    domRevision: 2,
    adapter: 'generic',
    captchaPresent: false,
    job: {
      company: 'Acme',
      title: 'Analyst',
      jobUrl: 'https://jobs.example/a',
      location: 'Denver',
      source: 'Direct'
    },
    fields: [field]
  };
  const freshScan = {
    ...analyzedScan,
    domRevision: 3,
    fields: [{
      ...field
    }, {
      fieldId: 'field-new',
      fingerprint: 'bbbbbbbbbbbbbbbb',
      label: 'New conditional question',
      type: 'text',
      options: [],
      nearbyText: '',
      required: false,
      riskClass: 'F2_REVIEW'
    }]
  };
  assert.equal(canFillFieldAcrossRevision({ analyzedScan, freshScan, field }), true);
  assert.equal(canFillFieldAcrossRevision({
    analyzedScan,
    freshScan: {
      ...freshScan,
      fields: freshScan.fields.map(candidate => candidate.fieldId === field.fieldId
        ? { ...candidate, maxLength: 250 }
        : candidate)
    },
    field
  }), false);
  assert.equal(canFillFieldAcrossRevision({
    analyzedScan,
    freshScan: {
      ...freshScan,
      fields: freshScan.fields.map(candidate => candidate.fieldId === field.fieldId
        ? { ...candidate, nearbyText: 'If yes, explain the circumstances.' }
        : candidate)
    },
    field
  }), false);
  assert.equal(canFillFieldAcrossRevision({
    analyzedScan,
    freshScan: { ...freshScan, captchaPresent: true },
    field
  }), false);
  assert.equal(canFillFieldAcrossRevision({
    analyzedScan,
    freshScan: {
      ...freshScan,
      discovery: { contexts: [{ kind: 'account_gate', count: 1, status: 'manual' }] }
    },
    field
  }), false);
  assert.equal(canFillFieldAcrossRevision({
    analyzedScan,
    freshScan: { ...freshScan, job: { ...freshScan.job, title: 'Different role' } },
    field
  }), false);
});

test('one bulk action fills unchanged reviewed fields across an unrelated form revision', async () => {
  const first = {
    fieldId: 'field-first',
    fingerprint: 'aaaaaaaaaaaaaaaa',
    label: 'First name',
    type: 'text',
    options: [],
    nearbyText: '',
    required: true,
    riskClass: 'F1_VERIFIED'
  };
  const second = {
    fieldId: 'field-linkedin',
    fingerprint: 'bbbbbbbbbbbbbbbb',
    label: 'LinkedIn Profile',
    type: 'url',
    options: [],
    nearbyText: '',
    required: false,
    riskClass: 'F1_VERIFIED'
  };
  const analyzedScan = {
    pageId: 'page-a',
    urlHash: 'hash-a',
    domRevision: 0,
    adapter: 'generic',
    captchaPresent: false,
    job: {
      company: 'Acme',
      title: 'Analyst',
      jobUrl: 'https://jobs.example/a'
    },
    fields: [first, second]
  };
  let liveScan = analyzedScan;
  const requested = [];
  const completed = [];
  const result = await executeBulkFillItems({
    items: [first, second].map(field => ({ field, proposal: {}, confirmed: false })),
    requestFill: async item => {
      assert.equal(canFillFieldAcrossRevision({
        analyzedScan,
        freshScan: liveScan,
        field: item.field
      }), true);
      requested.push(item.field.fieldId);
      if (item.field.fieldId === first.fieldId) {
        liveScan = {
          ...liveScan,
          domRevision: 1,
          fields: [...liveScan.fields, {
            fieldId: 'field-conditional',
            fingerprint: 'cccccccccccccccc',
            label: 'New conditional question',
            type: 'text',
            options: [],
            nearbyText: '',
            required: false,
            riskClass: 'F2_REVIEW'
          }]
        };
      }
      return { verified: true, reason: 'verified_after_fill' };
    },
    onFilled: item => completed.push(item.field.fieldId)
  });
  assert.deepEqual(requested, [first.fieldId, second.fieldId]);
  assert.deepEqual(completed, requested);
  assert.deepEqual(result, { filled: 2, occupied: 0, copyOnly: 0, stopped: '' });
  assert.equal(bulkFillResultTone(result), 'success');
  assert.equal(bulkFillResultTone({ filled: 0, copyOnly: 2, stopped: '' }), 'error');
});

test('Ollama status accepts implicit latest tags and flags only clearly old versions', () => {
  assert.equal(isModelInstalled('nomic-embed-text', ['nomic-embed-text:latest']), true);
  assert.equal(isModelInstalled('qwen3:8b', ['qwen3:8b']), true);
  assert.equal(isModelInstalled('qwen3:8b', ['qwen3:latest']), false);
  assert.equal(isClearlyUnsupportedOllamaVersion('0.31.9'), true);
  assert.equal(isClearlyUnsupportedOllamaVersion('0.32.0'), false);
  assert.equal(isClearlyUnsupportedOllamaVersion('0.33.0-rc1'), false);
  assert.equal(isClearlyUnsupportedOllamaVersion('future-build'), false);
});

test('active page scanning explains how to restore an expired activeTab grant', async () => {
  await assert.rejects(
    activeHttpTab({ query: async () => [{ id: 42 }] }),
    {
      name: 'Error',
      message: 'Chrome has not granted access to this page. Click the Job Application Copilot toolbar icon while the job application page is active, then select Rescan.'
    }
  );
  const tab = { id: 42, url: 'https://job-boards.greenhouse.io/example/jobs/123' };
  assert.equal(await activeHttpTab({ query: async () => [tab] }), tab);
  await assert.rejects(
    activeHttpTab({ query: async () => [{ id: 42, url: 'not a URL' }] }),
    /supported web page/u
  );
});

test('document byte reads reject cleanly before a parser worker is created', async () => {
  await assert.rejects(readDocumentBytes({
    size: 1,
    async arrayBuffer() { throw new Error('simulated file read failure'); }
  }), /simulated file read failure/u);
  let readAttempted = false;
  const oversized = {
    size: (10 * 1024 * 1024) + 1,
    async arrayBuffer() { readAttempted = true; return new ArrayBuffer(1); }
  };
  assert.throws(() => validateImportFile(oversized), /10 MB or smaller/u);
  await assert.rejects(readDocumentBytes(oversized), /10 MB or smaller/u);
  assert.equal(readAttempted, false);
});

test('legacy zero-byte PDF metadata is repaired from encrypted retained bytes', async () => {
  const bytes = new TextEncoder().encode('%PDF retained candidate bytes');
  const record = {
    id: 'doc:legacy-empty-hash',
    kind: 'document',
    schemaVersion: 1,
    value: {
      document: {
        id: 'doc:legacy-empty-hash',
        filename: 'Resume.pdf',
        mimeType: 'application/pdf',
        size: 0,
        sha256: '47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU',
        version: 1
      },
      text: 'Candidate experience',
      blocks: [],
      originalBytesBase64: bytesToBase64(bytes)
    }
  };
  const repaired = await repairDocumentMetadataFromRetainedBytes(record);
  const digest = await sha256Base64Url(bytes);
  assert.equal(repaired.document.size, bytes.byteLength);
  assert.equal(repaired.document.sha256, digest);
  assert.equal(repaired.document.id, `doc:${digest.slice(0, 24)}`);
  assert.equal(repaired.originalBytesBase64, record.value.originalBytesBase64);
  assert.equal(formatBytes(0), 'Size unavailable');
  assert.equal(formatBytes(undefined), 'Size unavailable');
  assert.equal(formatBytes(1024), '1.0 KB');
  assert.equal(await repairDocumentMetadataFromRetainedBytes({
    ...record,
    value: { ...record.value, document: repaired.document }
  }), null);
});

test('TXT and HTML filenames, extracted text, and retained originals stay out of raw IndexedDB', async () => {
  const indexedDB = new IDBFactory();
  const databaseName = `sidepanel-raw-${crypto.randomUUID()}`;
  const vault = new EncryptedIndexedDbVault({
    indexedDB,
    databaseName,
    sessionKeyStore: new MemorySessionKeyStore()
  });
  await vault.initialize('strong local passphrase');
  const fixtures = [{
    id: 'doc:txt-secret',
    filename: 'private-candidate-notes.txt',
    mimeType: 'text/plain',
    phrase: 'UNIQUE_TXT_SOURCE_PHRASE_4729'
  }, {
    id: 'doc:html-secret',
    filename: 'private-company-research.html',
    mimeType: 'text/html',
    phrase: 'UNIQUE_HTML_SOURCE_PHRASE_9153'
  }];
  for (const fixture of fixtures) {
    const originalBytes = new TextEncoder().encode(fixture.phrase);
    const parsed = {
      document: {
        id: fixture.id,
        filename: fixture.filename,
        mimeType: fixture.mimeType,
        size: originalBytes.byteLength,
        lastModified: null,
        sha256: `hash-${fixture.id}`,
        version: 1
      },
      text: fixture.phrase,
      blocks: [{ blockIndex: 0, paragraph: 1, section: '', page: null, text: fixture.phrase }],
      warnings: []
    };
    const value = createDocumentVaultRecord(parsed, {
      retainOriginal: true,
      originalBytes
    });
    await vault.putRecord({ id: fixture.id, kind: 'document', value });
    const storedRows = await rawRecords(indexedDB, databaseName);
    storedRows.forEach(row => assert.deepEqual(Object.keys(row).sort(), ['encrypted', 'id']));
    const serialized = JSON.stringify(storedRows);
    assert.doesNotMatch(serialized, new RegExp(fixture.phrase, 'u'));
    assert.doesNotMatch(serialized, new RegExp(fixture.filename.replaceAll('.', '\\.'), 'u'));
    assert.doesNotMatch(serialized, new RegExp(fixture.id, 'u'));
    const decrypted = await vault.getRecord(fixture.id);
    assert.equal(decrypted.value.text, fixture.phrase);
    assert.ok(decrypted.value.originalBytesBase64);
  }
  await vault.close();
});

test('bulk verified-profile updates remain encrypted and upsert without delete-first', async () => {
  const indexedDB = new IDBFactory();
  const databaseName = `profile-raw-${crypto.randomUUID()}`;
  const vault = new EncryptedIndexedDbVault({
    indexedDB,
    databaseName,
    sessionKeyStore: new MemorySessionKeyStore()
  });
  await vault.initialize('strong local passphrase');
  await vault.putRecord({
    id: 'fact:email',
    kind: 'verified-fact',
    value: { key: 'email', label: 'Email address', value: 'old@example.com', sourceRole: 'user_verified', verifiedAt: 'older' }
  });
  const changes = verifiedFactChanges({
    values: { email: 'new@example.com', phone: '555-0199' },
    existingFacts: await vault.listRecords('verified-fact'),
    verifiedAt: '2026-07-19T12:00:00.000Z'
  });
  for (const record of changes.upserts) await vault.putRecord(record);
  for (const recordId of changes.deletes) await vault.deleteRecord(recordId);

  const facts = await vault.listRecords('verified-fact');
  assert.deepEqual(facts.map(record => record.value.value).sort(), ['555-0199', 'new@example.com']);
  const serialized = JSON.stringify(await rawRecords(indexedDB, databaseName));
  assert.doesNotMatch(serialized, /old@example\.com|new@example\.com|555-0199|fact:email|fact:phone/u);
  await vault.close();
});

test('sidepanel markup and controller retain the local-only safety contracts', async () => {
  const [html, css, source] = await Promise.all([
    readFile(path.join(packageDir, 'src/sidepanel/sidepanel.html'), 'utf8'),
    readFile(path.join(packageDir, 'src/sidepanel/sidepanel.css'), 'utf8'),
    readFile(path.join(packageDir, 'src/sidepanel/sidepanel.js'), 'utf8')
  ]);
  const regenerationSource = source.slice(
    source.indexOf('const regenerateOneField = async'),
    source.indexOf('const trackerJobFromForm =')
  );
  assert.match(html, /data-tab="sources"[\s\S]*data-tab="application"[\s\S]*data-tab="settings"/u);
  assert.match(html, /data-tab="sources"[^>]*><span class="tab-step"[^>]*>1<\/span> Setup/u);
  assert.match(html, /data-tab="application"[^>]*><span class="tab-step"[^>]*>2<\/span> Apply/u);
  assert.match(html, /autocomplete="off" spellcheck="false" required aria-describedby="vault-passphrase-help"/u);
  assert.match(html, /data-profile-form autocomplete="off"/u);
  assert.match(html, /<details class="disclosure profile-disclosure" open>[\s\S]*Identity and contact/u);
  assert.match(html, /<details class="disclosure profile-disclosure">[\s\S]*Address[\s\S]*Professional links[\s\S]*Work authorization/u);
  assert.match(html, /<details class="card reusable-context-card disclosure-card"/u);
  assert.match(html, /<details class="card supporting-sources-card disclosure-card"/u);
  assert.match(html, /name="website" type="url"/u);
  assert.match(html, /data-custom-profile-form="custom_link"/u);
  assert.match(html, /data-custom-profile-form="custom_thought"/u);
  assert.match(html, /Relevant thoughts are sent only to local Ollama and shown as citations\./u);
  assert.match(html, /Do not store passwords, government IDs, medical information, protected demographic information/u);
  assert.match(html, /data-profile-count/u);
  assert.match(html, /data-profile-save-status role="status"/u);
  assert.match(html, /data-batch-fill hidden[\s\S]*data-preflight-heading[\s\S]*data-preflight-stats[\s\S]*data-clarification-batch hidden[\s\S]*data-batch-review[\s\S]*data-action="fill-all-ready"[\s\S]*data-action="fill-verified-only"[\s\S]*data-action="toggle-unresolved"/u);
  assert.match(html, /data-batch-review-copy[\s\S]*One review unlocks those answers for this analysis/u);
  assert.match(html, /Opening the tracker transfers only the displayed metadata and selected files[\s\S]*>Review in tracker<\/button>/u);
  assert.doesNotMatch(html, /name="confirmed"/u);
  assert.doesNotMatch(source, /data-confirm-field|values\.get\('confirmed'\)/u);
  assert.match(source, /Approve & fill[\s\S]*text: 'Copy'[\s\S]*text: proposal\.action === 'fill' \? 'Revise' : 'Try again'/u);
  assert.match(source, /feedback\.hidden = true/u);
  assert.match(html, /data-job-summary hidden[\s\S]*data-job-context-heading[\s\S]*Job context options[\s\S]*data-action="refresh-job-context"[\s\S]*data-action="capture-selected-job-text"[\s\S]*data-job-context-form/u);
  assert.match(html, /Nothing is submitted automatically\./u);
  assert.match(html, /data-batch-fill-status role="status" aria-live="polite" aria-atomic="true"/u);
  assert.match(html, /data-action="scan-active-page">Analyze current page<\/button>[\s\S]*<details class="inline-disclosure job-summary" data-job-summary hidden>/u);
  assert.match(html, /data-field-workspace aria-labelledby="answers-heading" hidden/u);
  assert.doesNotMatch(html, /data-action="rescan"/u);
  assert.match(source, /state\.scan \? 'Refresh answers' : 'Analyze current page'/u);
  assert.match(source, /\$\('\[data-field-workspace\]'\)\.hidden = !state\.scan/u);
  assert.match(html, /<details class="disclosure">[\s\S]*Import documents[\s\S]*<details class="disclosure">[\s\S]*Add a pasted note/u);
  assert.match(html, /<details class="card tracker-card disclosure-card"/u);
  assert.match(html, /<details class="card disclosure-card">[\s\S]*Local model preferences/u);
  for (const key of Object.keys(FACT_DEFINITIONS)) assert.match(html, new RegExp(`name="${key}"`, 'u'));
  assert.doesNotMatch(html, /data-fact-form|Add verified fact/u);
  assert.match(html, /Interactive CAPTCHA challenge detected—filling paused\. Complete it, then rescan\./u);
  assert.match(html, /OLLAMA_ORIGINS=chrome-extension:\/\/[\s\S]*extension-id/u);
  assert.match(html, /OLLAMA_NO_CLOUD=1/u);
  assert.match(source, /globalThis\.chrome\?\.runtime\?\.id/u);
  assert.match(html, /data-action="export-sanitized-fixture"/u);
  assert.match(html, /data-discovery-notice hidden role="status" aria-live="polite"/u);
  assert.match(html, /opaque structure signature[\s\S]*never page URLs, entered values, answers, secrets, hidden-field details, source documents, or arbitrary page text/iu);
  assert.match(html, /value="user_verified">Verified material/u);
  assert.match(css, /--signal-blue: #005fed/u);
  assert.match(css, /\.batch-actions \{[^}]*display: flex[^}]*flex-wrap: wrap/u);
  assert.match(css, /@media \(max-width: 430px\)[\s\S]*\.batch-actions \{[^}]*display: grid;[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);[\s\S]*\[data-action="toggle-unresolved"\][\s\S]*grid-column: 1 \/ -1/u);
  assert.match(source, /orchestratePageGeneration/u);
  assert.match(source, /embedChunksWithLexicalFallback/u);
  assert.match(source, /files: \['content\/page-runtime\.js'\]/u);
  assert.match(source, /'PAGE_SCAN_REQUEST'/u);
  assert.match(source, /'FIELD_FILL_REQUEST'/u);
  assert.match(source, /fillMode === 'copy_only'/u);
  assert.match(source, /const existingValues = new Map\(\$\$\('\[data-batch-clarification-field\]'\)\.map/u);
  assert.match(source, /prepareSanitizedFixtureExport/u);
  assert.match(source, /TRACKER_INTERNAL_MESSAGE_TYPES\.BEGIN/u);
  assert.match(source, /const retainOriginal = true/u);
  assert.match(source, /\['', 'No additional feedback'\]/u);
  assert.match(source, /window\.addEventListener\('pagehide', abortActiveOperations/u);
  assert.match(source, /modelEligibleFields\(retrieval\.request\.fields\)/u);
  assert.match(source, /\[data-vault-form\], \[data-profile-form\], \[data-custom-profile-form\], \[data-document-form\], \[data-note-form\]/u);
  assert.match(source, /const CACHE_SCHEMA_VERSION = 10/u);
  assert.match(source, /filterSelectedDocumentRecords\(state\.documents, state\.selectedSourceIds, scan\.pageId\)/u);
  assert.match(source, /state\.facts\.filter\(record => factAppliesToApplication\(record, scan\.pageId\)\)/u);
  assert.match(source, /field-preview-fintech/u);
  assert.match(source, /mode: 'free_format'/u);
  assert.match(source, /field-preview-summary[^]*fillMode: 'copy_only'/u);
  assert.match(source, /text: 'Regenerate answer'/u);
  assert.match(source, /Leave blank to save nothing and regenerate from the current draft, field request, and frozen evidence/u);
  assert.doesNotMatch(source, /feedbackSelect\.value = 'more_specific'/u);
  assert.match(regenerationSource, /let feedback = normalizeFieldRegenerationFeedback\(\{[\s\S]*preset,[\s\S]*text,/u);
  assert.match(regenerationSource, /priorDraft: proposal/u);
  assert.match(regenerationSource, /if \(clarificationMode && text\.trim\(\)\)[\s\S]*Use the newly saved user-verified clarification citation/u);
  assert.doesNotMatch(regenerationSource, /EXPERIENCE_RECOVERY_INSTRUCTION/u);
  assert.match(regenerationSource, /const resolution = resolveBlankFeedbackRegenerationProposal\(\{/u);
  assert.match(regenerationSource, /if \(!resolution\.retained\) \{[\s\S]*state\.filledFieldIds\.delete\(fieldId\)/u);
  assert.match(regenerationSource, /model could not produce a better supported revision, so the current cited answer was kept/iu);
  assert.match(source, /Application-only clarification - encrypted local evidence/u);
  assert.match(source, /generationStatus: 'complete'/u);
  assert.match(source, /proposals\.some\(isDegradedFallbackProposal\)/u);
  assert.match(source, /if \(!generationError\) \{[\s\S]*await storeGenerationCache/u);
  assert.match(source, /repairDocumentMetadataFromRetainedBytes/u);
  assert.match(source, /ensureExperienceCandidateEvidence/u);
  assert.match(source, /createElement\('details',[\s\S]*className: 'answer-card'/u);
  assert.match(source, /PREVIEW_MODE[\s\S]*initPreview/u);
  assert.match(html, /This optional section applies only to U\.S\. employment/u);
  assert.match(source, /ELIGIBILITY_FACT_KEY_SET\.has\(record\.value\?\.key\) \|\| !factAppliesToApplication/u);
  assert.match(source, /evidence: evidenceForModelField\(state\.evidence, field\.fieldId\)/u);
  assert.match(html, /data-generation-notice/u);
  assert.ok((source.match(/await invalidateGeneratedAnswers\(\)/gu) || []).length >= 5);
  assert.match(source, /state\.activeTabId !== null && selected\.tabId !== state\.activeTabId[\s\S]*Rescan it before reviewing or regenerating/u);
  assert.doesNotMatch(`${html}\n${source}`, /OPENAI|api[_ -]?key/iu);
});
