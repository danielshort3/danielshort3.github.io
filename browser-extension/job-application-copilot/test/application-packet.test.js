import assert from 'node:assert/strict';
import test from 'node:test';
import { IDBFactory } from 'fake-indexeddb';
import {
  APPLICATION_CONTEXT_MODES,
  APPLICATION_CONTEXT_ORIGINS,
  APPLICATION_PACKET_RECORD_KIND,
  APPLICATION_PACKET_SCHEMA_VERSION,
  createApplicationContextSnapshot,
  createApprovedAnswerSnapshot,
  createClarificationSnapshot,
  matchApplicationPacketRecord,
  summarizeApplicationPreflight,
  upsertApplicationPacketRecord
} from '../src/application/application-packet.js';
import { sha256Base64Url } from '../src/vault/crypto.js';
import { EncryptedIndexedDbVault } from '../src/vault/indexeddb-vault.js';
import { MemorySessionKeyStore } from '../src/vault/session-key-store.js';

const CREATED_AT = '2026-07-20T18:00:00.000Z';
const UPDATED_AT = '2026-07-20T18:05:00.000Z';

const requestResult = (request) => new Promise((resolve, reject) => {
  request.addEventListener('success', () => resolve(request.result), { once: true });
  request.addEventListener('error', () => reject(request.error), { once: true });
});

const postingContext = ({
  url = 'https://careers.example.com/jobs/senior-analyst-12345?utm_source=board&jobId=REQ-12345',
  location = 'Remote - United States',
  capturedAt = CREATED_AT
} = {}) => ({
  origin: APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE,
  mode: APPLICATION_CONTEXT_MODES.POSTING,
  capturedAt,
  job: {
    company: 'Example Analytics, Inc.',
    title: 'Senior Data Analyst',
    location,
    source: 'Company careers',
    jobUrl: url,
    description: 'Build reliable analytics products and partner with business teams.'
  },
  page: {
    adapter: 'generic',
    pageId: 'posting-page',
    urlHash: 'posting-url-hash',
    domRevision: 0
  }
});

const atsContext = ({
  url = 'https://jobs.ats-example.com/example/987654/apply',
  capturedAt = UPDATED_AT
} = {}) => ({
  origin: APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE,
  mode: APPLICATION_CONTEXT_MODES.APPLICATION,
  capturedAt,
  job: {
    company: 'Example Analytics',
    title: 'Senior Data Analyst',
    source: 'ATS',
    jobUrl: url,
    description: 'Application form for the Senior Data Analyst role.'
  },
  page: {
    adapter: 'generic',
    pageId: 'application-page',
    urlHash: 'application-url-hash',
    domRevision: 3
  }
});

const answerField = {
  fieldId: 'why-role',
  fingerprint: 'field-fingerprint-1',
  label: 'Why are you interested in this role?',
  type: 'textarea',
  options: [],
  required: true,
  riskClass: 'F2_REVIEW'
};

const answerProposal = (value = 'I am interested in building reliable analytics products.') => ({
  field_id: answerField.fieldId,
  action: 'fill',
  confidence: 'review',
  risk_class: answerField.riskClass,
  value_type: 'text',
  value,
  selected_values: [],
  checked: false,
  citation_ids: ['candidate-citation-1'],
  short_rationale: 'Grounded in candidate evidence.',
  abstain_reason: ''
});

test('context snapshots retain bounded encrypted content and explicit source metadata', async () => {
  const longContent = `  First requirement.\r\n\r\n${'A'.repeat(20_500)}  `;
  const input = {
    ...postingContext(),
    content: longContent
  };
  const before = structuredClone(input);
  const snapshot = await createApplicationContextSnapshot(input);

  assert.equal(snapshot.source.origin, 'live_page');
  assert.equal(snapshot.source.mode, 'posting');
  assert.equal(snapshot.source.url, 'https://careers.example.com/jobs/senior-analyst-12345?jobId=REQ-12345');
  assert.equal(snapshot.source.label, 'Company careers');
  assert.equal(snapshot.page.adapter, 'generic');
  assert.equal(snapshot.content.length, 20_000);
  assert.equal(snapshot.preview.retainedLength, 20_000);
  assert.equal(snapshot.preview.sourceLength, 20_520);
  assert.equal(snapshot.preview.text.length, 240);
  assert.equal(snapshot.preview.truncated, true);
  assert.equal(snapshot.preview.contentTruncated, true);
  assert.equal(snapshot.preview.sha256, await sha256Base64Url(snapshot.content));
  assert.match(snapshot.contextId, /^context:[A-Za-z0-9_-]{32}$/u);
  assert.deepEqual(input, before);
});

test('manual context is explicit and invalid source combinations fail closed', async () => {
  const snapshot = await createApplicationContextSnapshot({
    origin: APPLICATION_CONTEXT_ORIGINS.MANUAL,
    mode: APPLICATION_CONTEXT_MODES.MANUAL,
    label: 'Reviewed recruiter note',
    content: 'The recruiter confirmed that this role is remote.',
    capturedAt: CREATED_AT,
    job: { company: 'Example Analytics', title: 'Senior Data Analyst' }
  });
  assert.equal(snapshot.source.origin, 'manual');
  assert.equal(snapshot.source.mode, 'manual');
  assert.equal(snapshot.page, null);
  assert.equal(snapshot.content, 'The recruiter confirmed that this role is remote.');

  await assert.rejects(createApplicationContextSnapshot({
    origin: APPLICATION_CONTEXT_ORIGINS.MANUAL,
    mode: APPLICATION_CONTEXT_MODES.APPLICATION,
    content: 'Invalid combination'
  }), /manual mode/iu);
  await assert.rejects(createApplicationContextSnapshot({
    origin: APPLICATION_CONTEXT_ORIGINS.LIVE_PAGE,
    mode: APPLICATION_CONTEXT_MODES.POSTING,
    url: 'file:///private/job.html'
  }), /HTTP\(S\)/u);
});

test('packet matching preserves identity from a posting through a different ATS URL', async () => {
  const postingPacket = await upsertApplicationPacketRecord({
    context: postingContext(),
    now: CREATED_AT
  });
  assert.equal(postingPacket.kind, APPLICATION_PACKET_RECORD_KIND);
  assert.equal(postingPacket.value.schemaVersion, APPLICATION_PACKET_SCHEMA_VERSION);
  assert.match(postingPacket.value.applicationId, /^local-app:[A-Za-z0-9_-]{32}$/u);

  const match = matchApplicationPacketRecord([postingPacket], atsContext());
  assert.equal(match.record.id, postingPacket.id);
  assert.equal(match.reason, 'company_title');

  const applicationPacket = await upsertApplicationPacketRecord({
    existingRecord: match.record,
    context: atsContext(),
    now: UPDATED_AT
  });
  assert.equal(applicationPacket.value.applicationId, postingPacket.value.applicationId);
  assert.equal(applicationPacket.id, postingPacket.id);
  assert.equal(applicationPacket.value.contextSnapshots.length, 2);
  assert.equal(applicationPacket.value.identity.aliases.urls.length, 2);
  assert.equal(Object.hasOwn(applicationPacket.value.contextSnapshots[0], 'identity'), false);
  assert.equal(applicationPacket.value.lastContextId, applicationPacket.value.contextSnapshots[0].contextId);
});

test('ambiguous company-title matches fail closed while an explicit packet hint resolves them', async () => {
  const first = await upsertApplicationPacketRecord({
    context: postingContext({ location: 'Denver, CO' }),
    now: CREATED_AT
  });
  const second = await upsertApplicationPacketRecord({
    context: postingContext({
      url: 'https://careers.example.com/jobs/senior-analyst-67890',
      location: 'Austin, TX'
    }),
    now: CREATED_AT
  });
  const nextPage = atsContext({ url: 'https://another-ats.example/apply/new-page' });
  assert.equal(matchApplicationPacketRecord([first, second], nextPage), null);
  const explicit = matchApplicationPacketRecord([first, second], {
    ...nextPage,
    applicationId: second.value.applicationId
  });
  assert.equal(explicit.record.id, second.id);
  assert.equal(explicit.reason, 'application_id');
});

test('approved answers and verified clarifications require explicit, cited user review', async () => {
  const packet = await upsertApplicationPacketRecord({ context: postingContext(), now: CREATED_AT });
  const applicationId = packet.value.applicationId;
  await assert.rejects(createApprovedAnswerSnapshot({
    applicationId,
    field: answerField,
    proposal: answerProposal(),
    approved: false
  }), /explicit user approval/iu);
  await assert.rejects(createApprovedAnswerSnapshot({
    applicationId,
    field: answerField,
    proposal: { ...answerProposal(), citation_ids: [] },
    approved: true
  }), /at least one citation/iu);

  const approved = await createApprovedAnswerSnapshot({
    applicationId,
    field: answerField,
    proposal: answerProposal(),
    approved: true,
    approvedAt: UPDATED_AT,
    sourceSignature: 'source-signature-1',
    contextId: packet.value.lastContextId
  });
  assert.equal(approved.approvedBy, 'user');
  assert.equal(approved.answer.valueType, 'text');
  assert.deepEqual(approved.citationIds, ['candidate-citation-1']);

  await assert.rejects(createClarificationSnapshot({
    applicationId,
    field: answerField,
    text: 'I led the analytics product rollout.',
    verified: false,
    evidenceRecordId: 'fact:clarification'
  }), /explicit user verification/iu);
  await assert.rejects(createClarificationSnapshot({
    applicationId,
    field: answerField,
    text: 'I led the analytics product rollout.',
    verified: true
  }), /citations or an encrypted evidence record/iu);

  const clarification = await createClarificationSnapshot({
    applicationId,
    field: answerField,
    text: 'I led the analytics product rollout.',
    verified: true,
    verifiedAt: UPDATED_AT,
    evidenceRecordId: 'fact:application-clarification',
    contextId: packet.value.lastContextId
  });
  assert.equal(clarification.verifiedBy, 'user');
  assert.equal(clarification.evidenceRecordId, 'fact:application-clarification');
});

test('preflight summary uses mutually exclusive compact workflow counts', () => {
  const fields = [
    { fieldId: 'filled', riskClass: 'F1_VERIFIED' },
    { fieldId: 'verified', riskClass: 'F1_VERIFIED' },
    { fieldId: 'review', riskClass: 'F2_REVIEW' },
    { fieldId: 'missing', riskClass: 'F2_REVIEW' },
    { fieldId: 'copy', riskClass: 'F2_REVIEW', fillMode: 'copy_only' },
    { fieldId: 'manual', riskClass: 'F2_REVIEW', manual: true },
    { fieldId: 'skip', riskClass: 'F2_REVIEW' }
  ];
  const proposal = (fieldId, action, confidence) => ({ field_id: fieldId, action, confidence });
  const summary = summarizeApplicationPreflight({
    fields,
    proposals: [
      proposal('filled', 'fill', 'high'),
      proposal('verified', 'fill', 'high'),
      proposal('review', 'fill', 'review'),
      proposal('missing', 'ask_user', 'needs_input'),
      proposal('copy', 'fill', 'review'),
      proposal('skip', 'skip', 'review')
    ],
    filledFieldIds: new Set(['filled']),
    unsupportedCount: 2
  });
  assert.deepEqual(summary, {
    total: 7,
    filled: 1,
    verifiedReady: 1,
    reviewRequired: 1,
    needsInput: 1,
    copyOnly: 1,
    manual: 1,
    skipped: 1,
    unsupported: 2,
    fillable: 2,
    requiresUserAction: 4,
    remaining: 5
  });
  assert.equal(Object.isFrozen(summary), true);
  assert.throws(() => summarizeApplicationPreflight({ filledFieldIds: 'filled' }), /array or Set/u);
});

test('packet upsert stores latest reviewed snapshots without weakening application scope', async () => {
  const initial = await upsertApplicationPacketRecord({ context: postingContext(), now: CREATED_AT });
  const applicationId = initial.value.applicationId;
  const firstAnswer = await createApprovedAnswerSnapshot({
    applicationId,
    field: answerField,
    proposal: answerProposal(),
    approved: true,
    approvedAt: CREATED_AT,
    contextId: initial.value.lastContextId
  });
  const revisedAnswer = await createApprovedAnswerSnapshot({
    applicationId,
    field: answerField,
    proposal: answerProposal('I am interested in building reliable analytics products with business partners.'),
    approved: true,
    approvedAt: UPDATED_AT,
    contextId: initial.value.lastContextId
  });
  const clarification = await createClarificationSnapshot({
    applicationId,
    field: answerField,
    text: 'I partnered with business teams on analytics product delivery.',
    verified: true,
    verifiedAt: UPDATED_AT,
    citationIds: ['user-verified-citation-1'],
    contextId: initial.value.lastContextId
  });
  const preflight = summarizeApplicationPreflight({
    fields: [answerField],
    proposals: [answerProposal()],
    filledFieldIds: [],
    unsupportedCount: 0
  });
  const withFirst = await upsertApplicationPacketRecord({
    existingRecord: initial,
    context: postingContext(),
    approvedAnswers: [firstAnswer],
    clarifications: [clarification],
    preflight,
    now: CREATED_AT
  });
  const withRevision = await upsertApplicationPacketRecord({
    existingRecord: withFirst,
    context: atsContext(),
    approvedAnswers: [revisedAnswer],
    now: UPDATED_AT
  });
  assert.equal(withRevision.value.approvedAnswers.length, 1);
  assert.equal(withRevision.value.approvedAnswers[0].answer.value, revisedAnswer.answer.value);
  assert.equal(withRevision.value.clarifications.length, 1);
  assert.deepEqual(withRevision.value.preflight, preflight);
  assert.equal(withRevision.value.createdAt, CREATED_AT);
  assert.equal(withRevision.value.updatedAt, UPDATED_AT);

  await assert.rejects(upsertApplicationPacketRecord({
    existingRecord: withRevision,
    context: atsContext(),
    approvedAnswers: [{ ...revisedAnswer, applicationId: 'local-app:different' }],
    now: UPDATED_AT
  }), /does not belong/iu);
  await assert.rejects(upsertApplicationPacketRecord({
    existingRecord: withRevision,
    context: atsContext(),
    preflight: { total: 1 },
    now: UPDATED_AT
  }), /summary returned by summarizeApplicationPreflight/iu);
});
test('application packet persists through the encrypted vault without plaintext metadata', async () => {
  const packet = await upsertApplicationPacketRecord({
    context: postingContext(),
    now: CREATED_AT
  });
  const indexedDB = new IDBFactory();
  const databaseName = `application-packet-${crypto.randomUUID()}`;
  const vault = new EncryptedIndexedDbVault({
    indexedDB,
    databaseName,
    sessionKeyStore: new MemorySessionKeyStore()
  });
  await vault.initialize('application packet test passphrase');
  await vault.putRecord(packet);

  const restored = await vault.getRecord(packet.id);
  assert.equal(restored.kind, APPLICATION_PACKET_RECORD_KIND);
  assert.equal(restored.value.applicationId, packet.value.applicationId);
  assert.equal(restored.value.contextSnapshots[0].content, packet.value.contextSnapshots[0].content);

  const database = await requestResult(indexedDB.open(databaseName));
  const transaction = database.transaction('records', 'readonly');
  const rawRows = await requestResult(transaction.objectStore('records').getAll());
  database.close();
  assert.equal(rawRows.length, 1);
  assert.deepEqual(Object.keys(rawRows[0]).sort(), ['encrypted', 'id']);
  assert.doesNotMatch(
    JSON.stringify(rawRows),
    /Example Analytics|Senior Data Analyst|application-packet|local-app|Build reliable analytics products/u
  );
  await vault.reset();
});