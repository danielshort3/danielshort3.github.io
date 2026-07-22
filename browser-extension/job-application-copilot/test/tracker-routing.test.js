import assert from 'node:assert/strict';
import test from 'node:test';
import { FieldSelectionRouter, validateFieldSelectedMessage } from '../src/background/field-selection-router.js';
import { TrackerCaptureRouter, TRACKER_PENDING_KEY_PREFIX } from '../src/background/tracker-capture-router.js';
import { installTrackerBridge } from '../src/content/tracker-bridge.js';
import {
  TRACKER_CHANNEL,
  TRACKER_EXTENSION_SOURCE,
  TRACKER_INTERNAL_MESSAGE_TYPES,
  TRACKER_MESSAGE_TYPES,
  TRACKER_PAGE_SOURCE,
  TRACKER_PROTOCOL_VERSION,
  TRACKER_TRANSFER_LIMITS,
  validateTrackerBeginMessage,
  validateTrackerWindowMessage
} from '../src/shared/tracker-protocol.js';
import { bytesToBase64, sha256Base64Url } from '../src/vault/crypto.js';

const EXTENSION_ID = 'jigajpmnbiofgmgcnmdeechgibpjlfop';
const CAPTURE_ID = '550e8400-e29b-41d4-a716-446655440000';
const FILE_ID = '0f51f5db-f044-4dce-bf42-bd4cc5dd7b2b';
const NONCE = 'tracker-generated-nonce-123456';
const RELOAD_NONCE = 'tracker-reload-nonce-987654321';
const DOCUMENT_ID = 'tracker-document-one';
const RELOAD_DOCUMENT_ID = 'tracker-document-two';
const NOW = new Date('2026-07-18T18:00:00.000Z');

class MemoryStorageArea {
  constructor() {
    this.values = {};
    this.accessLevels = [];
  }

  async setAccessLevel(value) {
    this.accessLevels.push(value);
  }

  async get(key) {
    if (typeof key === 'string') return { [key]: this.values[key] };
    return { ...this.values };
  }

  async set(values) {
    Object.assign(this.values, structuredClone(values));
  }

  async remove(key) {
    delete this.values[key];
  }
}

const job = {
  company: 'Acme',
  title: 'Data Analyst',
  jobUrl: 'https://acme.example/jobs/1',
  location: 'Denver, CO',
  source: 'Company site',
  postingDate: '2026-07-01',
  appliedDate: '2026-07-18',
  status: 'Applied',
  tags: ['Analytics']
};

const commonPageMessage = (type, requestId = crypto.randomUUID()) => ({
  channel: TRACKER_CHANNEL,
  source: TRACKER_PAGE_SOURCE,
  protocolVersion: TRACKER_PROTOCOL_VERSION,
  type,
  captureId: CAPTURE_ID,
  requestId,
  channelNonce: NONCE
});

test('tracker protocol rejects unknown keys, sources, and case-insensitive duplicate tags', () => {
  const acknowledgement = {
    channel: TRACKER_CHANNEL,
    source: TRACKER_EXTENSION_SOURCE,
    protocolVersion: TRACKER_PROTOCOL_VERSION,
    type: TRACKER_MESSAGE_TYPES.COMPLETE_ACK,
    captureId: CAPTURE_ID,
    requestId: crypto.randomUUID(),
    channelNonce: NONCE,
    payload: {
      accepted: true,
      applicationId: 'APP#CAPTURE#one',
      acknowledgedAt: NOW.toISOString(),
      reason: ''
    }
  };
  assert.equal(validateTrackerWindowMessage(acknowledgement, {
    expectedSource: TRACKER_EXTENSION_SOURCE
  }), acknowledgement);
  assert.throws(() => validateTrackerWindowMessage({ ...acknowledgement, source: TRACKER_PAGE_SOURCE }), /extension/u);
  const request = {
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST),
    requestedAt: NOW.toISOString()
  };
  assert.equal(validateTrackerWindowMessage(request), request);
  assert.throws(() => validateTrackerWindowMessage({ ...request, arbitrary: true }), /unexpected or missing keys/u);
  assert.throws(() => validateTrackerWindowMessage({ ...request, source: TRACKER_EXTENSION_SOURCE }), /tracker page/u);
  const invalidBegin = {
    type: TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN,
    payload: {
      captureId: CAPTURE_ID,
      issuedAt: NOW.toISOString(),
      expiresAt: '2026-07-18T18:10:00.000Z',
      job: { ...job, tags: ['Analytics', 'analytics'] },
      files: []
    }
  };
  assert.throws(() => validateTrackerWindowMessage({ ...request, requestId: 'not-a-uuid' }), /UUID/u);
  assert.throws(() => validateTrackerBeginMessage(invalidBegin), /unique without regard to case/u);
});

test('tracker router binds capture access to a main-frame tab, document, and nonce', async () => {
  const bytes = new Uint8Array(TRACKER_TRANSFER_LIMITS.chunkSize + 3).fill(65);
  const digest = await sha256Base64Url(bytes);
  const record = {
    kind: 'document',
    value: {
      document: {
        filename: 'Daniel-Short-Resume.pdf',
        mimeType: 'application/pdf',
        size: bytes.byteLength,
        sha256: digest
      },
      originalBytesBase64: bytesToBase64(bytes)
    }
  };
  const session = new MemoryStorageArea();
  const local = new MemoryStorageArea();
  const router = new TrackerCaptureRouter({
    storageSession: session,
    storageLocal: local,
    extensionId: EXTENSION_ID,
    now: () => NOW,
    vaultFactory: () => ({
      async open() {},
      async getRecord(id) { return id === 'doc:resume' ? record : null; },
      async close() {}
    })
  });
  await router.initialize();
  assert.deepEqual(session.accessLevels, [{ accessLevel: 'TRUSTED_CONTEXTS' }]);
  assert.deepEqual(local.accessLevels, [{ accessLevel: 'TRUSTED_CONTEXTS' }]);

  const begin = await router.handle({
    type: TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN,
    payload: {
      captureId: CAPTURE_ID,
      issuedAt: NOW.toISOString(),
      expiresAt: '2026-07-18T18:10:00.000Z',
      job,
      files: [{ fileId: FILE_ID, recordId: 'doc:resume', kind: 'resume' }]
    }
  }, { id: EXTENSION_ID, url: `chrome-extension://${EXTENSION_ID}/sidepanel/sidepanel.html` });
  assert.equal(begin.trackerUrl, `https://www.danielshort.me/tools/job-application-tracker?copilotCapture=${CAPTURE_ID}`);
  assert.equal('channelNonce' in begin, false);

  const pageUrl = begin.trackerUrl;
  const sender = {
    id: EXTENSION_ID,
    frameId: 0,
    documentId: DOCUMENT_ID,
    url: pageUrl,
    tab: { id: 8, url: pageUrl }
  };
  const manifestRequest = {
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST),
    requestedAt: NOW.toISOString()
  };
  await assert.rejects(router.handle({
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST),
    payload: { fileId: FILE_ID, chunkIndex: 0, chunkSize: TRACKER_TRANSFER_LIMITS.chunkSize }
  }, sender), /request its manifest/u);
  const manifest = await router.handle(manifestRequest, sender);
  assert.equal(manifest.type, TRACKER_MESSAGE_TYPES.MANIFEST);
  assert.equal(manifest.channelNonce, NONCE);
  assert.equal(manifest.payload.files[0].recordId, undefined);
  assert.equal(manifest.payload.files[0].totalChunks, 2);

  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: RELOAD_NONCE
  }, sender), /nonce does not match/u);

  const reloadSender = { ...sender, documentId: RELOAD_DOCUMENT_ID };
  const reboundManifest = await router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: RELOAD_NONCE
  }, reloadSender);
  assert.equal(reboundManifest.channelNonce, RELOAD_NONCE);
  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID()
  }, sender), /stale tracker document/u);
  await assert.rejects(router.handle({
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST),
    payload: { fileId: FILE_ID, chunkIndex: 0, chunkSize: TRACKER_TRANSFER_LIMITS.chunkSize }
  }, sender), /stale tracker document/u);
  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: 'different-tab-nonce-12345'
  }, {
    ...reloadSender,
    documentId: 'tracker-document-other-tab',
    tab: { ...sender.tab, id: 99 }
  }), /another tab/u);
  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: RELOAD_NONCE
  }, { ...reloadSender, frameId: 1 }), /main frame/u);
  const { documentId: ignoredDocumentId, ...senderWithoutDocument } = reloadSender;
  assert.equal(ignoredDocumentId, RELOAD_DOCUMENT_ID);
  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: RELOAD_NONCE
  }, senderWithoutDocument), /document identity/u);

  const reboundCommon = type => ({
    ...commonPageMessage(type),
    channelNonce: RELOAD_NONCE
  });

  const chunk = await router.handle({
    ...reboundCommon(TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST),
    payload: { fileId: FILE_ID, chunkIndex: 1, chunkSize: TRACKER_TRANSFER_LIMITS.chunkSize }
  }, reloadSender);
  assert.equal(chunk.type, TRACKER_MESSAGE_TYPES.FILE_CHUNK);
  assert.deepEqual(Buffer.from(chunk.payload.data, 'base64'), Buffer.from(bytes.subarray(TRACKER_TRANSFER_LIMITS.chunkSize)));

  await assert.rejects(router.handle({
    ...reboundCommon(TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST),
    payload: { fileId: crypto.randomUUID(), chunkIndex: 0, chunkSize: TRACKER_TRANSFER_LIMITS.chunkSize }
  }, reloadSender), /not part of this capture/u);

  await router.handle({
    ...reboundCommon(TRACKER_MESSAGE_TYPES.DISMISSED),
    payload: { dismissedAt: NOW.toISOString() }
  }, reloadSender);
  const storageKey = `${TRACKER_PENDING_KEY_PREFIX}${CAPTURE_ID}`;
  assert.ok(session.values[storageKey]);
  assert.equal(session.values[storageKey].trackerTabId, 8);
  assert.equal(session.values[storageKey].trackerDocumentId, RELOAD_DOCUMENT_ID);
  assert.deepEqual(session.values[storageKey].staleTrackerDocumentIds, [DOCUMENT_ID]);

  const completionRequest = {
    ...reboundCommon(TRACKER_MESSAGE_TYPES.COMPLETE),
    payload: { applicationId: 'APP#CAPTURE#one', completedAt: NOW.toISOString() }
  };
  const completion = await router.handle(completionRequest, reloadSender);
  assert.equal(completion.type, TRACKER_MESSAGE_TYPES.COMPLETE_ACK);
  assert.deepEqual(completion.payload, {
    accepted: true,
    applicationId: 'APP#CAPTURE#one',
    acknowledgedAt: NOW.toISOString(),
    reason: ''
  });
  const receipt = session.values[storageKey];
  assert.deepEqual(Object.keys(receipt).sort(), [
    'acknowledgedAt',
    'applicationId',
    'captureId',
    'channelNonce',
    'expiresAt',
    'state',
    'trackerDocumentId',
    'trackerTabId',
    'version'
  ]);
  assert.equal(receipt.state, 'completed');
  assert.equal(receipt.job, undefined);
  assert.equal(receipt.files, undefined);

  const retry = await router.handle({ ...completionRequest, requestId: crypto.randomUUID() }, reloadSender);
  assert.equal(retry.payload.accepted, true);
  assert.equal(retry.payload.acknowledgedAt, completion.payload.acknowledgedAt);
  const conflict = await router.handle({
    ...completionRequest,
    requestId: crypto.randomUUID(),
    payload: { ...completionRequest.payload, applicationId: 'APP#CAPTURE#two' }
  }, reloadSender);
  assert.equal(conflict.payload.accepted, false);
  assert.match(conflict.payload.reason, /different application/u);
  assert.equal(session.values[storageKey].applicationId, 'APP#CAPTURE#one');
  await assert.rejects(router.handle({
    ...completionRequest,
    requestId: crypto.randomUUID(),
    channelNonce: 'wrong-completion-nonce-1234'
  }, reloadSender), /nonce/u);
  await assert.rejects(router.handle({
    ...manifestRequest,
    requestId: crypto.randomUUID(),
    channelNonce: RELOAD_NONCE
  }, reloadSender), /already completed/u);
});

test('content bridge forwards only strict same-window tracker requests and matching responses', async () => {
  const listeners = new Map();
  const posted = [];
  const location = {
    href: `https://www.danielshort.me/tools/job-application-tracker?copilotCapture=${CAPTURE_ID}`,
    origin: 'https://www.danielshort.me'
  };
  const windowObject = {
    location,
    addEventListener(type, listener) { listeners.set(type, listener); },
    removeEventListener(type) { listeners.delete(type); },
    postMessage(message, origin) { posted.push({ message, origin }); }
  };
  const request = {
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST),
    requestedAt: NOW.toISOString()
  };
  const response = {
    channel: TRACKER_CHANNEL,
    source: TRACKER_EXTENSION_SOURCE,
    protocolVersion: TRACKER_PROTOCOL_VERSION,
    type: TRACKER_MESSAGE_TYPES.MANIFEST,
    captureId: CAPTURE_ID,
    requestId: request.requestId,
    channelNonce: NONCE,
    payload: {
      issuedAt: NOW.toISOString(),
      expiresAt: '2026-07-18T18:10:00.000Z',
      job,
      files: []
    }
  };
  let runtimeResponse = response;
  const uninstall = installTrackerBridge({ windowObject, runtime: { async sendMessage() { return runtimeResponse; } } });
  await listeners.get('message')({ source: windowObject, origin: location.origin, data: request });
  assert.deepEqual(posted, [{ message: response, origin: location.origin }]);
  await listeners.get('message')({ source: {}, origin: location.origin, data: request });
  const completionRequest = {
    ...commonPageMessage(TRACKER_MESSAGE_TYPES.COMPLETE),
    requestId: crypto.randomUUID(),
    payload: { applicationId: 'APP#CAPTURE#one', completedAt: NOW.toISOString() }
  };
  const completionResponse = {
    channel: TRACKER_CHANNEL,
    source: TRACKER_EXTENSION_SOURCE,
    protocolVersion: TRACKER_PROTOCOL_VERSION,
    type: TRACKER_MESSAGE_TYPES.COMPLETE_ACK,
    captureId: CAPTURE_ID,
    requestId: completionRequest.requestId,
    channelNonce: NONCE,
    payload: {
      accepted: true,
      applicationId: 'APP#CAPTURE#one',
      acknowledgedAt: NOW.toISOString(),
      reason: ''
    }
  };
  runtimeResponse = completionResponse;
  await listeners.get('message')({ source: windowObject, origin: location.origin, data: completionRequest });
  assert.deepEqual(posted[1], { message: completionResponse, origin: location.origin });
  runtimeResponse = { ...completionResponse, requestId: crypto.randomUUID() };
  await listeners.get('message')({
    source: windowObject,
    origin: location.origin,
    data: { ...completionRequest, requestId: crypto.randomUUID() }
  });
  assert.equal(posted.length, 2);
  uninstall();
  assert.equal(listeners.has('message'), false);
});

test('FIELD_SELECTED router accepts only a validated main-frame runtime event', async () => {
  const session = new MemoryStorageArea();
  const opened = [];
  const router = new FieldSelectionRouter({
    storageSession: session,
    sidePanel: { async open(value) { opened.push(value); } },
    extensionId: EXTENSION_ID,
    now: () => NOW
  });
  const message = {
    channel: 'job-application-copilot',
    version: 1,
    type: 'FIELD_SELECTED',
    requestId: 'overlay-request-1',
    payload: {
      field: {
        fieldId: 'field-0123456789abcdef',
        fingerprint: '0123456789abcdef',
        label: 'Email',
        type: 'email',
        options: [],
        nearbyText: '',
        required: true,
        riskClass: 'F1_VERIFIED'
      }
    }
  };
  assert.equal(validateFieldSelectedMessage(message), message);
  await router.handle(message, {
    id: EXTENSION_ID,
    frameId: 0,
    url: 'https://jobs.example/apply',
    tab: { id: 9 }
  });
  assert.deepEqual(opened, [{ tabId: 9 }]);
  assert.equal(session.values.jobCopilotSelectedField.fieldId, 'field-0123456789abcdef');
  await assert.rejects(router.handle(message, {
    id: EXTENSION_ID,
    frameId: 1,
    url: 'https://jobs.example/apply',
    tab: { id: 9 }
  }), /main-frame/u);
  assert.throws(() => validateFieldSelectedMessage({ ...message, arbitrary: true }), /Invalid FIELD_SELECTED/u);
});
