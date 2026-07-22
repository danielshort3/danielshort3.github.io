'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { randomUUID, webcrypto } = require('node:crypto');
const fs = require('node:fs');
const vm = require('node:vm');

const {
  CHANNEL,
  CHUNK_BYTES,
  EXTENSION_SOURCE,
  PROTOCOL_VERSION,
  WEBSITE_SOURCE,
  bytesToBase64Url,
  calculateSha256,
  decodeBase64Chunk,
  validateManifest
} = require('../js/tools/job-application-copilot-bridge.js');

const NOW = Date.parse('2026-07-18T18:00:00.000Z');

const validManifest = () => ({
  issuedAt: '2026-07-18T17:59:00.000Z',
  expiresAt: '2026-07-18T18:09:00.000Z',
  job: {
    company: 'Acme Corp',
    title: 'Data Analyst',
    jobUrl: 'https://acme.example/jobs/123',
    location: 'Remote',
    source: 'Company site',
    tags: ['remote', 'analytics']
  },
  files: [{
    fileId: '550e8400-e29b-41d4-a716-446655440000',
    sha256: 'A'.repeat(43),
    name: 'Daniel-Resume.pdf',
    type: 'application/pdf',
    kind: 'resume',
    size: 3,
    chunkSize: CHUNK_BYTES,
    totalChunks: 1
  }]
});

const waitFor = async (predicate, label = 'condition') => {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.fail(`Timed out waiting for ${label}.`);
};

const createFakeElement = () => ({
  checked: false,
  children: [],
  dataset: {},
  disabled: false,
  events: [],
  files: [],
  hidden: true,
  listeners: new Map(),
  open: false,
  textContent: '',
  value: '',
  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  },
  appendChild(child) {
    this.children.push(child);
    return child;
  },
  click() {
    return this.listeners.get('click')?.({ target: this });
  },
  dispatchEvent(event) {
    this.events.push(event.type);
    return this.listeners.get(event.type)?.(event) ?? true;
  },
  replaceChildren(...children) {
    this.children = children;
  },
  setAttribute(name, value) {
    this[name] = String(value);
  }
});

const runtimeManifest = ({ sha256 = 'A5BYxvLAy0ksUzsKTRTvd8wPeKvMztUofYShogEc-4E' } = {}) => {
  const now = Date.now();
  return {
    issuedAt: new Date(now - 1_000).toISOString(),
    expiresAt: new Date(now + 60_000).toISOString(),
    job: {
      company: 'Acme Corp',
      title: 'Data Analyst',
      jobUrl: 'https://acme.example/jobs/123',
      location: 'Remote',
      source: 'Company site',
      postingDate: new Date(now - 86_400_000).toISOString().slice(0, 10),
      appliedDate: new Date(now).toISOString().slice(0, 10),
      status: 'Applied',
      tags: ['analytics']
    },
    files: [{
      fileId: '550e8400-e29b-41d4-a716-446655440000',
      sha256,
      name: 'Daniel-Resume.pdf',
      type: 'application/pdf',
      kind: 'resume',
      size: 3,
      chunkSize: CHUNK_BYTES,
      totalChunks: 1
    }]
  };
};

const createBridgeHarness = ({
  manifest = runtimeManifest(),
  chunkMode = 'success',
  completionMode = 'success',
  manualTimers = false
} = {}) => {
  const source = fs.readFileSync('js/tools/job-application-copilot-bridge.js', 'utf8');
  const elements = {
    handoff: createFakeElement(),
    handoffStatus: createFakeElement(),
    retry: createFakeElement(),
    handoffDismiss: createFakeElement(),
    review: createFakeElement(),
    reviewTitle: createFakeElement(),
    reviewSummary: createFakeElement(),
    reviewFiles: createFakeElement(),
    reviewStatus: createFakeElement(),
    reviewAction: createFakeElement(),
    reviewDismiss: createFakeElement(),
    applicationType: createFakeElement(),
    prospectType: createFakeElement(),
    company: createFakeElement(),
    title: createFakeElement(),
    jobUrl: createFakeElement(),
    location: createFakeElement(),
    source: createFakeElement(),
    postingDate: createFakeElement(),
    appliedDate: createFakeElement(),
    status: createFakeElement(),
    tags: createFakeElement(),
    postingUnknown: createFakeElement(),
    resume: createFakeElement(),
    cover: createFakeElement(),
    details: createFakeElement()
  };
  const selectors = new Map([
    ['[data-jobtrack-copilot="handoff"]', elements.handoff],
    ['[data-jobtrack-copilot="handoff-status"]', elements.handoffStatus],
    ['[data-jobtrack-copilot="retry"]', elements.retry],
    ['[data-jobtrack-copilot="handoff-dismiss"]', elements.handoffDismiss],
    ['[data-jobtrack-copilot="review"]', elements.review],
    ['[data-jobtrack-copilot="review-title"]', elements.reviewTitle],
    ['[data-jobtrack-copilot="review-summary"]', elements.reviewSummary],
    ['[data-jobtrack-copilot="review-files"]', elements.reviewFiles],
    ['[data-jobtrack-copilot="review-status"]', elements.reviewStatus],
    ['[data-jobtrack-copilot="review-action"]', elements.reviewAction],
    ['[data-jobtrack-copilot="review-dismiss"]', elements.reviewDismiss],
    ['#jobtrack-entry-type-application', elements.applicationType],
    ['#jobtrack-entry-type-prospect', elements.prospectType],
    ['#jobtrack-company', elements.company],
    ['#jobtrack-title', elements.title],
    ['#jobtrack-job-url', elements.jobUrl],
    ['#jobtrack-location', elements.location],
    ['#jobtrack-source', elements.source],
    ['#jobtrack-posting-date', elements.postingDate],
    ['#jobtrack-date', elements.appliedDate],
    ['#jobtrack-status', elements.status],
    ['#jobtrack-tags', elements.tags],
    ['#jobtrack-posting-unknown', elements.postingUnknown],
    ['#jobtrack-resume', elements.resume],
    ['#jobtrack-cover', elements.cover],
    ['.jobtrack-form-details', elements.details]
  ]);
  const posted = [];
  const listeners = new Map();
  const timers = new Map();
  let timerSequence = 0;
  let context = null;

  class FakeFile {
    constructor(parts, name, options = {}) {
      this.name = name;
      this.type = options.type || '';
      this.lastModified = options.lastModified || 0;
      this.size = parts.reduce((total, part) => total + (part.byteLength ?? part.length ?? 0), 0);
    }
  }

  class FakeDataTransfer {
    constructor() {
      this.files = [];
      this.items = { add: file => this.files.push(file) };
    }
  }

  class FakeEvent {
    constructor(type) {
      this.type = type;
    }
  }

  const windowObject = {
    DataTransfer: FakeDataTransfer,
    Event: FakeEvent,
    File: FakeFile,
    atob: value => Buffer.from(value, 'base64').toString('binary'),
    btoa: value => Buffer.from(value, 'binary').toString('base64'),
    crypto: { randomUUID, subtle: webcrypto.subtle },
    document: {
      body: { dataset: { maxAttachmentBytes: String(10 * 1024 * 1024) } },
      createElement: () => createFakeElement(),
      querySelector: selector => selectors.get(selector) || null
    },
    history: {
      state: null,
      replaceState() {}
    },
    location: {
      href: 'https://www.danielshort.me/tools/job-application-tracker?copilotCapture=550e8400-e29b-41d4-a716-446655440000',
      origin: 'https://www.danielshort.me'
    },
    addEventListener(type, listener) {
      const values = listeners.get(type) || [];
      values.push(listener);
      listeners.set(type, values);
    },
    clearTimeout(timerId) {
      if (manualTimers) timers.delete(timerId);
      else clearTimeout(timerId);
    },
    setTimeout(callback, delay) {
      if (!manualTimers) return setTimeout(callback, delay);
      const timerId = ++timerSequence;
      timers.set(timerId, callback);
      return timerId;
    }
  };

  const dispatchResponse = (request, type, payload, overrides = {}) => {
    const responseData = {
      source: EXTENSION_SOURCE,
      channel: CHANNEL,
      protocolVersion: PROTOCOL_VERSION,
      type,
      captureId: request.captureId,
      requestId: request.requestId,
      channelNonce: request.channelNonce,
      payload,
      ...overrides
    };
    const response = vm.runInContext(
      `JSON.parse(${JSON.stringify(JSON.stringify(responseData))})`,
      context
    );
    (listeners.get('message') || []).forEach(listener => listener({
      source: windowObject,
      origin: windowObject.location.origin,
      data: response
    }));
  };

  windowObject.postMessage = (message) => {
    posted.push(message);
    if (message.type === 'capture-manifest-request') {
      Promise.resolve().then(() => dispatchResponse(message, 'capture-manifest', manifest));
      return;
    }
    if (message.type === 'capture-file-chunk-request' && chunkMode === 'success') {
      Promise.resolve().then(() => dispatchResponse(message, 'capture-file-chunk', {
        fileId: message.payload.fileId,
        chunkIndex: message.payload.chunkIndex,
        totalChunks: 1,
        data: 'AQID'
      }));
      return;
    }
    if (message.type === 'capture-complete' && completionMode !== 'none') {
      const accepted = completionMode !== 'reject';
      const payload = {
        accepted,
        applicationId: message.payload.applicationId,
        acknowledgedAt: new Date().toISOString(),
        reason: accepted ? '' : 'The extension rejected this completion.'
      };
      const overrides = completionMode === 'wrong-binding'
        ? { channelNonce: 'wrong-completion-channel-nonce' }
        : {};
      Promise.resolve().then(() => dispatchResponse(
        message,
        'capture-complete-ack',
        payload,
        overrides
      ));
    }
  };

  context = vm.createContext({
    Buffer,
    URL,
    Uint8Array,
    console,
    window: windowObject
  });
  vm.runInContext(source, context);

  return {
    bridge: windowObject.JobApplicationCopilotBridge,
    elements,
    posted,
    runTimers() {
      const callbacks = [...timers.values()];
      timers.clear();
      callbacks.forEach(callback => callback());
    }
  };
};

test('bridge manifest accepts only bounded tracker metadata and defaults appliedDate', () => {
  const manifest = validateManifest(validManifest(), { now: NOW, maxFileBytes: 10 * 1024 * 1024 });
  assert.equal(manifest.job.company, 'Acme Corp');
  assert.equal(manifest.job.appliedDate, '2026-07-18');
  assert.equal(manifest.files[0].chunkSize, 256 * 1024);
  assert.equal(manifest.files[0].sha256, 'A'.repeat(43));

  for (const forbiddenKey of [
    'answers',
    'batch',
    'captureDate',
    'followUpDate',
    'followUpNote',
    'jobDescription',
    'notes',
    'questions',
    'research',
    'sourceText'
  ]) {
    const payload = validManifest();
    payload.job[forbiddenKey] = 'must not cross the bridge';
    assert.throws(
      () => validateManifest(payload, { now: NOW }),
      error => /unsupported data/.test(error.message) && error.message.includes(forbiddenKey)
    );
  }

  const tooManyTags = validManifest();
  tooManyTags.job.tags = Array.from({ length: 13 }, (_, index) => `tag-${index}`);
  assert.throws(() => validateManifest(tooManyTags, { now: NOW }), /at most 12 labels/u);
});

test('bridge rejects expired, stale, oversized, malformed, and unapproved files', () => {
  const expired = validManifest();
  expired.expiresAt = '2026-07-18T17:59:59.000Z';
  assert.throws(() => validateManifest(expired, { now: NOW }), /expired/);

  const stale = validManifest();
  stale.issuedAt = '2026-07-18T17:40:00.000Z';
  assert.throws(() => validateManifest(stale, { now: NOW }), /stale/);

  const oversized = validManifest();
  oversized.files[0].size = 11 * 1024 * 1024;
  oversized.files[0].totalChunks = Math.ceil(oversized.files[0].size / CHUNK_BYTES);
  assert.throws(() => validateManifest(oversized, { now: NOW, maxFileBytes: 10 * 1024 * 1024 }), /size limit/);

  const wrongChunkSize = validManifest();
  wrongChunkSize.files[0].chunkSize = 64 * 1024;
  assert.throws(() => validateManifest(wrongChunkSize, { now: NOW }), /262144-byte chunks/);

  const wrongHash = validManifest();
  wrongHash.files[0].sha256 = 'not-a-sha256';
  assert.throws(() => validateManifest(wrongHash, { now: NOW }), /base64url SHA-256/);

  const unapproved = validManifest();
  unapproved.files[0].name = 'answers.txt';
  unapproved.files[0].type = 'text/plain';
  assert.throws(() => validateManifest(unapproved, { now: NOW }), /approved original document type/);

  const legacyDoc = validManifest();
  legacyDoc.files[0].name = 'legacy-resume.doc';
  legacyDoc.files[0].type = 'application/msword';
  assert.throws(() => validateManifest(legacyDoc, { now: NOW }), /approved original document type/);

  const docx = validManifest();
  docx.files[0].name = 'Daniel-Cover-Letter.docx';
  docx.files[0].type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
  docx.files[0].kind = 'cover-letter';
  assert.doesNotThrow(() => validateManifest(docx, { now: NOW }));
});

test('base64 chunks are exact-size decoded and SHA-256 digests use base64url', async () => {
  assert.deepEqual([...decodeBase64Chunk('AQID', 3)], [1, 2, 3]);
  assert.throws(() => decodeBase64Chunk('AQID', 2), /unexpected size/);
  assert.throws(() => decodeBase64Chunk('***=', 2), /valid base64/);
  assert.equal(bytesToBase64Url(Uint8Array.from([251, 255])), '-_8');
  assert.equal(
    await calculateSha256(Uint8Array.from([1, 2, 3])),
    'A5BYxvLAy0ksUzsKTRTvd8wPeKvMztUofYShogEc-4E'
  );
});

test('signed-out capture links make zero extension requests', async () => {
  const harness = createBridgeHarness();
  harness.bridge.setAuthenticated(false);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(harness.posted, []);
  assert.match(harness.elements.handoffStatus.textContent, /Sign in to load/u);
});

test('authenticated handoff validates descriptors first and transfers files only after review click', async () => {
  const harness = createBridgeHarness();
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'manifest review');

  assert.deepEqual(harness.posted.map(message => message.type), ['capture-manifest-request']);
  assert.equal(harness.elements.company.value, '');
  assert.equal(harness.elements.resume.files.length, 0);
  assert.match(harness.elements.reviewFiles.children[0].textContent, /Daniel-Resume\.pdf/u);
  assert.match(harness.elements.reviewStatus.textContent, /transfer and verify/u);

  await harness.elements.reviewAction.click();
  assert.deepEqual(harness.posted.map(message => message.type), [
    'capture-manifest-request',
    'capture-file-chunk-request'
  ]);
  assert.equal(harness.elements.company.value, 'Acme Corp');
  assert.equal(harness.elements.title.value, 'Data Analyst');
  assert.equal(harness.elements.resume.files.length, 1);
  assert.equal(harness.elements.resume.files[0].name, 'Daniel-Resume.pdf');
  assert.equal(harness.bridge.hasPendingCapture(), true);
});


test('completion remains pending until a strict extension acknowledgement arrives', async () => {
  const harness = createBridgeHarness();
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'manifest review');
  await harness.elements.reviewAction.click();
  const completion = harness.bridge.complete('APP#CAPTURE#one');
  assert.equal(harness.bridge.hasPendingCapture(), true);
  assert.equal(await completion, true);
  assert.equal(harness.bridge.hasPendingCapture(), false);
  const message = harness.posted.find(candidate => candidate.type === 'capture-complete');
  assert.equal(message.payload.applicationId, 'APP#CAPTURE#one');
  assert.equal(Object.hasOwn(message.payload, 'answers'), false);
});

test('rejected and invalid completion acknowledgements preserve the reviewed capture', async () => {
  const rejected = createBridgeHarness({ completionMode: 'reject' });
  rejected.bridge.setAuthenticated(true);
  await waitFor(() => rejected.elements.review.hidden === false, 'rejected manifest review');
  await rejected.elements.reviewAction.click();
  await assert.rejects(
    rejected.bridge.complete('APP#CAPTURE#one'),
    /extension rejected this completion/iu
  );
  assert.equal(rejected.bridge.hasPendingCapture(), true);

  const oversized = createBridgeHarness();
  oversized.bridge.setAuthenticated(true);
  await waitFor(() => oversized.elements.review.hidden === false, 'oversized manifest review');
  await oversized.elements.reviewAction.click();
  await assert.rejects(oversized.bridge.complete('x'.repeat(257)), /exceeds 256 characters/u);
  assert.equal(
    oversized.posted.some(message => message.type === 'capture-complete'),
    false
  );
  assert.equal(oversized.bridge.hasPendingCapture(), true);
});

test('completion timeout preserves the reviewed capture for retry', async () => {
  const harness = createBridgeHarness({ completionMode: 'none', manualTimers: true });
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'timeout manifest review');
  await harness.elements.reviewAction.click();
  const completion = harness.bridge.complete('APP#CAPTURE#one');
  await waitFor(() => harness.posted.some(message => message.type === 'capture-complete'), 'completion request');
  harness.runTimers();
  await assert.rejects(completion, /did not respond in time/u);
  assert.equal(harness.bridge.hasPendingCapture(), true);
});

test('wrong-bound completion acknowledgement is ignored and times out safely', async () => {
  const harness = createBridgeHarness({ completionMode: 'wrong-binding', manualTimers: true });
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'wrong-bound manifest review');
  await harness.elements.reviewAction.click();
  const completion = harness.bridge.complete('APP#CAPTURE#one');
  await waitFor(
    () => harness.posted.some(message => message.type === 'capture-complete'),
    'wrong-bound completion request'
  );
  await new Promise(resolve => setImmediate(resolve));
  harness.runTimers();
  await assert.rejects(completion, /did not respond in time/u);
  assert.equal(harness.bridge.hasPendingCapture(), true);
});
test('interrupted reviewed file transfer clears pending capture without prefilling', async () => {
  const harness = createBridgeHarness({ chunkMode: 'none' });
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'manifest review');
  const review = harness.elements.reviewAction.click();
  await waitFor(
    () => harness.posted.some(message => message.type === 'capture-file-chunk-request'),
    'first chunk request'
  );
  harness.bridge.setAuthenticated(false);
  await review;
  assert.equal(harness.elements.company.value, '');
  assert.equal(harness.elements.resume.files.length, 0);
  assert.equal(harness.bridge.hasPendingCapture(), false);
});

test('reviewed file transfer times out without prefilling and remains retryable', async () => {
  const harness = createBridgeHarness({ chunkMode: 'none', manualTimers: true });
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'manifest review');
  const review = harness.elements.reviewAction.click();
  await waitFor(
    () => harness.posted.some(message => message.type === 'capture-file-chunk-request'),
    'timed chunk request'
  );
  harness.runTimers();
  await review;
  assert.match(harness.elements.reviewStatus.textContent, /did not respond in time/u);
  assert.equal(harness.elements.company.value, '');
  assert.equal(harness.elements.reviewAction.disabled, false);
  assert.equal(harness.bridge.hasPendingCapture(), false);
});

test('hash mismatch rejects reviewed files before any form prefill', async () => {
  const harness = createBridgeHarness({ manifest: runtimeManifest({ sha256: 'B'.repeat(43) }) });
  harness.bridge.setAuthenticated(true);
  await waitFor(() => harness.elements.review.hidden === false, 'manifest review');
  await harness.elements.reviewAction.click();
  assert.match(harness.elements.reviewStatus.textContent, /failed its integrity check/u);
  assert.equal(harness.elements.company.value, '');
  assert.equal(harness.elements.resume.files.length, 0);
  assert.equal(harness.bridge.hasPendingCapture(), false);
});

test('page bridge contract is strict, nonce-bound, memory-only, and loaded before the tracker', () => {
  const bridge = fs.readFileSync('js/tools/job-application-copilot-bridge.js', 'utf8');
  const tracker = fs.readFileSync('js/tools/job-application-tracker.js', 'utf8');
  const html = fs.readFileSync('pages/job-application-tracker.html', 'utf8');

  assert.equal(PROTOCOL_VERSION, 1);
  assert.equal(CHANNEL, 'danielshort.job-application-copilot');
  assert.equal(WEBSITE_SOURCE, 'danielshort.job-tracker');
  assert.equal(EXTENSION_SOURCE, 'danielshort.job-application-copilot.extension');
  assert.match(bridge, /event\.source === global[\s\S]*event\.origin === global\.location\.origin/);
  assert.match(bridge, /data\.channelNonce === channelNonce/);
  assert.match(bridge, /global\.postMessage\(makeEnvelope\(type, fields\), global\.location\.origin\)/);
  assert.match(bridge, /postToExtension\('capture-manifest-request', \{ requestedAt: new Date\(\)\.toISOString\(\) \}\)/);
  assert.match(bridge, /assertEnvelopeKeys\(response, \['payload'\], 'capture manifest response'\)/);
  assert.match(bridge, /payload: \{[\s\S]*applicationId:[\s\S]*completedAt:/);
  assert.match(bridge, /payload: \{ dismissedAt: new Date\(\)\.toISOString\(\) \}/);
  assert.doesNotMatch(bridge, /localStorage|sessionStorage|indexedDB/);
  assert.doesNotMatch(bridge, /application\/msword|new Set\(\['\.doc'\]\)/u);
  assert.doesNotMatch(bridge, /#jobtrack-(?:batch|capture-date|notes|follow-up-date|follow-up-note)/u);
  assert.match(bridge, /applicationType\.checked = true/);
  assert.match(bridge, /new global\.Event\('input', \{ bubbles: true \}\)/);
  assert.match(bridge, /new global\.Event\('change', \{ bubbles: true \}\)/);
  assert.match(tracker, /pendingCapture \? '\/api\/applications\/capture' : '\/api\/applications'/);
  assert.doesNotMatch(tracker, /captureId: pendingCapture\.captureId, \.\.\.payload/u);
  assert.match(bridge, /waitForResponse\('capture-complete-ack'/u);
  assert.doesNotMatch(bridge, /applicationId: String\(applicationId\)\.slice/u);
  assert.match(tracker, /await copilotBridge\?\.complete\?\.\(applicationId\)/);
  const submitApplication = tracker.slice(
    tracker.indexOf('const submitApplication = async'),
    tracker.indexOf('const initEntryForm =')
  );
  for (const key of ['company', 'title', 'jobUrl', 'location', 'source', 'postingDate', 'appliedDate', 'status', 'tags']) {
    assert.match(submitApplication, new RegExp(`${key}: payload\\.${key}`), `capture body should explicitly map ${key}`);
  }
  const manifestTransfer = bridge.slice(
    bridge.indexOf('const startTransfer = async'),
    bridge.indexOf('const transferPendingFiles = async')
  );
  assert.doesNotMatch(manifestTransfer, /capture-file-chunk-request|requestFile\(/u);
  const reviewedTransferStart = bridge.indexOf("reviewButton.addEventListener('click'");
  const reviewedTransfer = bridge.slice(
    reviewedTransferStart,
    bridge.indexOf('if (retryButton)', reviewedTransferStart)
  );
  assert.ok(reviewedTransfer.indexOf('await transferPendingFiles()') < reviewedTransfer.indexOf('handlers.onReview()'));
  assert.ok(submitApplication.indexOf('const uploaded = await uploadAttachments') < submitApplication.indexOf('await copilotBridge?.complete?.(applicationId)'));
  assert.ok(submitApplication.indexOf('if (attachmentError)') < submitApplication.indexOf('await copilotBridge?.complete?.(applicationId)'));
  assert.ok(html.indexOf('js/tools/job-application-copilot-bridge.js') < html.indexOf('js/tools/job-application-tracker.js'));
  assert.match(html, /dist\/site-shell\.[a-f0-9]+\.js/u);
  assert.match(html, /dist\/site-tools-account\.[a-f0-9]+\.js/u);
  const trackerScriptSources = [...html.matchAll(/<script\b[^>]*\bsrc="([^"]+)"[^>]*>/giu)]
    .map(match => match[1]);
  assert.equal(
    trackerScriptSources.some(source => /site-consent|analytics|googletagmanager|gtm(?:\.min)?\.js/iu.test(source)),
    false
  );
  assert.match(html, /data-jobtrack-copilot="handoff"/);
  assert.match(html, /data-jobtrack-copilot="review"/);
});

test('tracker routes have isolated no-store headers and do not inherit the catchall CSP', () => {
  const config = JSON.parse(fs.readFileSync('vercel.json', 'utf8'));
  const routeSources = [
    '/tools/job-application-tracker',
    '/tools/job-application-tracker.html',
    '/pages/job-application-tracker',
    '/pages/job-application-tracker.html'
  ];
  const catchallIndex = config.headers.findIndex(entry => entry.source.startsWith('/:path((?!'));
  assert.ok(catchallIndex >= 0, 'expected the site catchall header rule');
  const catchall = config.headers[catchallIndex];

  routeSources.forEach((source) => {
    const routeIndex = config.headers.findIndex(entry => entry.source === source);
    assert.ok(routeIndex >= 0 && routeIndex < catchallIndex, `${source} needs a dedicated header rule before catchall`);
    const headers = Object.fromEntries(config.headers[routeIndex].headers.map(header => [header.key, header.value]));
    assert.equal(headers['Cache-Control'], 'no-store, max-age=0');
    assert.equal(headers['Referrer-Policy'], 'no-referrer');
    assert.equal(headers['X-Content-Type-Options'], 'nosniff');
    assert.equal(headers['X-Frame-Options'], 'DENY');
    assert.equal(headers['Cross-Origin-Resource-Policy'], 'same-origin');
    assert.match(headers['Content-Security-Policy'], /base-uri 'self';/u);
    assert.match(headers['Content-Security-Policy'], /frame-ancestors 'none';/u);
    assert.match(headers['Content-Security-Policy'], /script-src 'self'; script-src-elem 'self'; script-src-attr 'none';/u);
    assert.match(headers['Content-Security-Policy'], /style-src 'self'; style-src-elem 'self'; style-src-attr 'unsafe-inline';/u);
    assert.match(headers['Content-Security-Policy'], /connect-src 'self' https:\/\/job-tracker-auth-886623862678\.auth\.us-east-2\.amazoncognito\.com https:\/\/fhp2is6v8h\.execute-api\.us-east-2\.amazonaws\.com https:\/\/\*\.s3\.amazonaws\.com https:\/\/\*\.s3\.us-east-2\.amazonaws\.com;/u);
    assert.doesNotMatch(headers['Content-Security-Policy'], /googletagmanager|google-analytics|cdn\.jsdelivr|unsafe-eval|wasm-unsafe-eval/iu);
  });

  for (const fragment of [
    'tools/job-application-tracker$',
    'tools/job-application-tracker\\.html$',
    'pages/job-application-tracker$',
    'pages/job-application-tracker\\.html$'
  ]) {
    assert.ok(catchall.source.includes(fragment), `catchall must exclude ${fragment}`);
  }
});
