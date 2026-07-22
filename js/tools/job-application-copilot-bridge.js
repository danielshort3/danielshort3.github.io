((global) => {
  'use strict';

  const PROTOCOL_VERSION = 1;
  // Same-window protocol: the tracker posts WEBSITE_SOURCE messages and accepts
  // only EXTENSION_SOURCE replies on this channel, bound to captureId,
  // requestId, and a fresh per-page channelNonce.
  const CHANNEL = 'danielshort.job-application-copilot';
  const WEBSITE_SOURCE = 'danielshort.job-tracker';
  const EXTENSION_SOURCE = 'danielshort.job-application-copilot.extension';
  const CHUNK_BYTES = 256 * 1024;
  const MAX_FILES = 2;
  const MANIFEST_MAX_AGE_MS = 10 * 60 * 1000;
  const MANIFEST_FUTURE_SKEW_MS = 60 * 1000;
  const RESPONSE_TIMEOUT_MS = 10 * 1000;
  const TRANSFER_TIMEOUT_MS = 2 * 60 * 1000;
  const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  const SHA256_BASE64URL_PATTERN = /^[A-Za-z0-9_-]{43}$/;
  const APPROVED_FILE_TYPES = Object.freeze({
    'application/pdf': new Set(['.pdf']),
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': new Set(['.docx'])
  });
  const APPROVED_FILE_KINDS = new Set(['resume', 'cover-letter']);
  const APPLICATION_STATUSES = new Set(['Applied', 'Screening', 'Interview', 'Offer', 'Rejected', 'Withdrawn']);
  const JOB_KEYS = new Set([
    'company',
    'title',
    'jobUrl',
    'location',
    'source',
    'postingDate',
    'appliedDate',
    'status',
    'tags'
  ]);
  const JOB_LIMITS = Object.freeze({
    company: 200,
    title: 200,
    jobUrl: 2048,
    location: 240,
    source: 120
  });

  class BridgeValidationError extends Error {
    constructor(message, code = 'invalid-capture') {
      super(message);
      this.name = 'BridgeValidationError';
      this.code = code;
    }
  }

  const isPlainObject = (value) => Boolean(value)
    && typeof value === 'object'
    && !Array.isArray(value)
    && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);

  const assertExactKeys = (value, allowedKeys, label) => {
    if (!isPlainObject(value)) throw new BridgeValidationError(`${label} must be an object.`);
    const unknown = Object.keys(value).filter(key => !allowedKeys.has(key));
    if (unknown.length) {
      throw new BridgeValidationError(`${label} contains unsupported data: ${unknown.join(', ')}.`);
    }
  };

  const normalizeText = (value, field, { required = false, maxLength = 0 } = {}) => {
    if (value === undefined || value === null) {
      if (required) throw new BridgeValidationError(`${field} is required.`);
      return '';
    }
    if (typeof value !== 'string') throw new BridgeValidationError(`${field} must be text.`);
    const normalized = value.trim();
    if (required && !normalized) throw new BridgeValidationError(`${field} is required.`);
    if (maxLength && normalized.length > maxLength) {
      throw new BridgeValidationError(`${field} exceeds ${maxLength} characters.`);
    }
    return normalized;
  };

  const normalizeDate = (value, field) => {
    const normalized = normalizeText(value, field);
    if (!normalized) return '';
    if (!/^\d{4}-\d{2}-\d{2}$/.test(normalized)) {
      throw new BridgeValidationError(`${field} must use YYYY-MM-DD.`);
    }
    const parsed = new Date(`${normalized}T00:00:00Z`);
    if (Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== normalized) {
      throw new BridgeValidationError(`${field} is not a valid date.`);
    }
    return normalized;
  };

  const normalizeIsoTimestamp = (value, field) => {
    const normalized = normalizeText(value, field, { required: true, maxLength: 40 });
    const parsed = new Date(normalized);
    if (Number.isNaN(parsed.getTime()) || parsed.toISOString() !== normalized) {
      throw new BridgeValidationError(`${field} must be a canonical ISO timestamp.`, 'stale-capture');
    }
    return normalized;
  };

  const normalizeJobUrl = (value) => {
    const normalized = normalizeText(value, 'jobUrl', { maxLength: JOB_LIMITS.jobUrl });
    if (!normalized) return '';
    try {
      const parsed = new URL(normalized);
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Unsupported protocol');
      return parsed.toString();
    } catch {
      throw new BridgeValidationError('jobUrl must be a valid HTTP or HTTPS URL.');
    }
  };

  const normalizeJobMetadata = (job) => {
    assertExactKeys(job, JOB_KEYS, 'manifest.job');
    const normalized = {
      company: normalizeText(job.company, 'company', { required: true, maxLength: JOB_LIMITS.company }),
      title: normalizeText(job.title, 'title', { required: true, maxLength: JOB_LIMITS.title })
    };
    const jobUrl = normalizeJobUrl(job.jobUrl);
    if (jobUrl) normalized.jobUrl = jobUrl;
    ['location', 'source'].forEach((field) => {
      const value = normalizeText(job[field], field, { maxLength: JOB_LIMITS[field] });
      if (value) normalized[field] = value;
    });
    ['postingDate', 'appliedDate'].forEach((field) => {
      const value = normalizeDate(job[field], field);
      if (value) normalized[field] = value;
    });
    if (job.status !== undefined) {
      const status = normalizeText(job.status, 'status', { maxLength: 80 });
      if (!APPLICATION_STATUSES.has(status)) throw new BridgeValidationError('status is not supported.');
      normalized.status = status;
    }
    if (job.tags !== undefined) {
      if (!Array.isArray(job.tags) || job.tags.length > 12) {
        throw new BridgeValidationError('tags must contain at most 12 labels.');
      }
      const seen = new Set();
      normalized.tags = job.tags.map((tag) => {
        const value = normalizeText(tag, 'tag', { required: true, maxLength: 36 });
        const key = value.toLowerCase();
        if (seen.has(key)) throw new BridgeValidationError('tags must be unique.');
        seen.add(key);
        return value;
      });
    }
    return normalized;
  };

  const getFilenameExtension = (name) => {
    const match = /\.[^.]+$/.exec(name.toLowerCase());
    return match ? match[0] : '';
  };

  const normalizeFileDescriptor = (descriptor, maxFileBytes) => {
    const keys = new Set(['fileId', 'sha256', 'name', 'type', 'kind', 'size', 'chunkSize', 'totalChunks']);
    assertExactKeys(descriptor, keys, 'manifest file');
    const fileId = normalizeText(descriptor.fileId, 'fileId', { required: true, maxLength: 36 });
    if (!UUID_PATTERN.test(fileId)) throw new BridgeValidationError('fileId must be a UUID.');
    const name = normalizeText(descriptor.name, 'file name', { required: true, maxLength: 180 });
    if (name === '.' || name === '..' || /[\\/\u0000-\u001f]/.test(name)) {
      throw new BridgeValidationError('file name is invalid.');
    }
    const type = normalizeText(descriptor.type, 'file type', { required: true, maxLength: 160 }).toLowerCase();
    const extensions = APPROVED_FILE_TYPES[type];
    if (!extensions || !extensions.has(getFilenameExtension(name))) {
      throw new BridgeValidationError(`${name} is not an approved original document type.`, 'unsupported-file');
    }
    const kind = normalizeText(descriptor.kind, 'file kind', { required: true, maxLength: 40 });
    if (!APPROVED_FILE_KINDS.has(kind)) throw new BridgeValidationError('file kind is not supported.');
    if (!Number.isSafeInteger(descriptor.size) || descriptor.size <= 0 || descriptor.size > maxFileBytes) {
      throw new BridgeValidationError(`${name} exceeds the attachment size limit.`, 'oversized-file');
    }
    const expectedChunks = Math.ceil(descriptor.size / CHUNK_BYTES);
    if (descriptor.chunkSize !== CHUNK_BYTES) {
      throw new BridgeValidationError(`${name} must use ${CHUNK_BYTES}-byte chunks.`);
    }
    if (!Number.isSafeInteger(descriptor.totalChunks) || descriptor.totalChunks !== expectedChunks) {
      throw new BridgeValidationError(`${name} has an invalid chunk count.`);
    }
    const sha256 = normalizeText(descriptor.sha256, 'sha256', { required: true, maxLength: 43 });
    if (!SHA256_BASE64URL_PATTERN.test(sha256)) {
      throw new BridgeValidationError('sha256 must be an unpadded base64url SHA-256 digest.');
    }
    return {
      fileId,
      sha256,
      name,
      type,
      kind,
      size: descriptor.size,
      chunkSize: CHUNK_BYTES,
      totalChunks: expectedChunks
    };
  };

  const validateManifest = (manifest, {
    now = Date.now(),
    maxFileBytes = 10 * 1024 * 1024
  } = {}) => {
    assertExactKeys(manifest, new Set(['issuedAt', 'expiresAt', 'job', 'files']), 'manifest');
    const issuedAt = normalizeIsoTimestamp(manifest.issuedAt, 'issuedAt');
    const expiresAt = normalizeIsoTimestamp(manifest.expiresAt, 'expiresAt');
    const issuedAtMs = Date.parse(issuedAt);
    const expiresAtMs = Date.parse(expiresAt);
    if (!Number.isFinite(issuedAtMs)) throw new BridgeValidationError('issuedAt is invalid.', 'stale-capture');
    if (!Number.isFinite(expiresAtMs) || expiresAtMs <= issuedAtMs || expiresAtMs <= now) {
      throw new BridgeValidationError('This capture has expired. Return to the extension and reopen it.', 'stale-capture');
    }
    if (issuedAtMs < now - MANIFEST_MAX_AGE_MS || issuedAtMs > now + MANIFEST_FUTURE_SKEW_MS) {
      throw new BridgeValidationError('This capture is stale. Return to the extension and reopen it.', 'stale-capture');
    }
    const job = normalizeJobMetadata(manifest.job);
    if (!job.appliedDate) job.appliedDate = new Date(now).toISOString().slice(0, 10);
    if (!Array.isArray(manifest.files) || manifest.files.length > MAX_FILES) {
      throw new BridgeValidationError(`A capture can include at most ${MAX_FILES} approved files.`);
    }
    const files = manifest.files.map(file => normalizeFileDescriptor(file, maxFileBytes));
    const fileIds = new Set();
    const fileKinds = new Set();
    let totalBytes = 0;
    files.forEach((file) => {
      if (fileIds.has(file.fileId)) throw new BridgeValidationError('File IDs must be unique.');
      if (fileKinds.has(file.kind)) throw new BridgeValidationError('Only one file of each approved kind is allowed.');
      fileIds.add(file.fileId);
      fileKinds.add(file.kind);
      totalBytes += file.size;
    });
    if (totalBytes > maxFileBytes * MAX_FILES) {
      throw new BridgeValidationError('The capture files exceed the total attachment limit.', 'oversized-file');
    }
    return { issuedAt, expiresAt, job, files };
  };

  const decodeBase64Chunk = (value, expectedBytes) => {
    if (typeof value !== 'string' || !value || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
      throw new BridgeValidationError('A file chunk is not valid base64.');
    }
    const maxEncodedLength = 4 * Math.ceil(expectedBytes / 3);
    if (value.length !== maxEncodedLength) throw new BridgeValidationError('A file chunk has an unexpected size.');
    let binary;
    try {
      binary = global && typeof global.atob === 'function'
        ? global.atob(value)
        : Buffer.from(value, 'base64').toString('binary');
    } catch {
      throw new BridgeValidationError('A file chunk is not valid base64.');
    }
    if (binary.length !== expectedBytes) throw new BridgeValidationError('A file chunk has an unexpected size.');
    const bytes = new Uint8Array(expectedBytes);
    for (let index = 0; index < expectedBytes; index += 1) bytes[index] = binary.charCodeAt(index);
    return bytes;
  };

  const bytesToBase64Url = (bytes) => {
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    }
    const encoded = global && typeof global.btoa === 'function'
      ? global.btoa(binary)
      : Buffer.from(bytes).toString('base64');
    return encoded.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  };

  const calculateSha256 = async (bytes, cryptoProvider = global?.crypto || globalThis.crypto) => {
    if (!cryptoProvider?.subtle) throw new BridgeValidationError('File integrity checks are unavailable.');
    const digest = await cryptoProvider.subtle.digest('SHA-256', bytes);
    return bytesToBase64Url(new Uint8Array(digest));
  };

  const TEST_API = Object.freeze({
    APPLICATION_STATUSES,
    CHANNEL,
    CHUNK_BYTES,
    EXTENSION_SOURCE,
    PROTOCOL_VERSION,
    WEBSITE_SOURCE,
    BridgeValidationError,
    bytesToBase64Url,
    calculateSha256,
    decodeBase64Chunk,
    normalizeJobMetadata,
    validateManifest
  });

  if (!global || !global.document) {
    if (typeof module !== 'undefined' && module.exports) module.exports = TEST_API;
    return;
  }

  const document = global.document;
  const handoffBanner = document.querySelector('[data-jobtrack-copilot="handoff"]');
  const handoffStatus = document.querySelector('[data-jobtrack-copilot="handoff-status"]');
  const retryButton = document.querySelector('[data-jobtrack-copilot="retry"]');
  const handoffDismissButton = document.querySelector('[data-jobtrack-copilot="handoff-dismiss"]');
  const reviewBanner = document.querySelector('[data-jobtrack-copilot="review"]');
  const reviewTitle = document.querySelector('[data-jobtrack-copilot="review-title"]');
  const reviewSummary = document.querySelector('[data-jobtrack-copilot="review-summary"]');
  const reviewFiles = document.querySelector('[data-jobtrack-copilot="review-files"]');
  const reviewStatus = document.querySelector('[data-jobtrack-copilot="review-status"]');
  const reviewButton = document.querySelector('[data-jobtrack-copilot="review-action"]');
  const reviewDismissButton = document.querySelector('[data-jobtrack-copilot="review-dismiss"]');
  const maxFileBytes = parseInt(document.body?.dataset.maxAttachmentBytes || '10485760', 10) || 10 * 1024 * 1024;
  const locationUrl = new URL(global.location.href);
  const captureParameters = locationUrl.searchParams.getAll('copilotCapture');
  const rawCaptureId = captureParameters.length === 1 ? captureParameters[0].trim() : '';
  const captureId = UUID_PATTERN.test(rawCaptureId) ? rawCaptureId.toLowerCase() : '';
  const hasCaptureParam = captureParameters.length > 0;
  const handlers = { onReview: null, onDismiss: null };
  const bridgeState = {
    authenticated: false,
    active: false,
    generation: 0,
    requestId: '',
    pending: null,
    reviewed: false,
    waiter: null
  };

  const randomUuid = () => {
    if (!global.crypto || typeof global.crypto.randomUUID !== 'function') {
      throw new BridgeValidationError('Secure browser randomness is unavailable.');
    }
    return global.crypto.randomUUID();
  };

  let channelNonce = '';
  try {
    channelNonce = randomUuid();
  } catch {}

  const setVisible = (element, visible) => {
    if (!element) return;
    element.hidden = !visible;
  };

  const setStatus = (element, message, tone = '') => {
    if (!element) return;
    element.textContent = message;
    if (tone) element.dataset.tone = tone;
    else delete element.dataset.tone;
  };

  const formatBytes = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const removeCaptureParameter = () => {
    const next = new URL(global.location.href);
    next.searchParams.delete('copilotCapture');
    global.history.replaceState(global.history.state, '', `${next.pathname}${next.search}${next.hash}`);
  };

  const makeEnvelope = (type, fields = {}) => ({
    source: WEBSITE_SOURCE,
    channel: CHANNEL,
    protocolVersion: PROTOCOL_VERSION,
    type,
    captureId,
    requestId: bridgeState.requestId,
    channelNonce,
    ...fields
  });

  const postToExtension = (type, fields = {}) => {
    global.postMessage(makeEnvelope(type, fields), global.location.origin);
  };

  const isMatchingResponse = (event, type) => {
    const data = event.data;
    return event.source === global
      && event.origin === global.location.origin
      && isPlainObject(data)
      && data.source === EXTENSION_SOURCE
      && data.channel === CHANNEL
      && data.protocolVersion === PROTOCOL_VERSION
      && data.type === type
      && data.captureId === captureId
      && data.requestId === bridgeState.requestId
      && data.channelNonce === channelNonce;
  };

  const rejectWaiter = (error) => {
    const waiter = bridgeState.waiter;
    if (!waiter) return;
    global.clearTimeout(waiter.timer);
    bridgeState.waiter = null;
    waiter.reject(error);
  };

  global.addEventListener('message', (event) => {
    const waiter = bridgeState.waiter;
    if (!waiter || !isMatchingResponse(event, waiter.type)) return;
    global.clearTimeout(waiter.timer);
    bridgeState.waiter = null;
    waiter.resolve(event.data);
  });

  const waitForResponse = (type, deadline) => new Promise((resolve, reject) => {
    if (bridgeState.waiter) rejectWaiter(new BridgeValidationError('Capture request was superseded.'));
    const remaining = Math.max(0, deadline - Date.now());
    if (!remaining) {
      reject(new BridgeValidationError('The extension did not respond in time.', 'timeout'));
      return;
    }
    const timer = global.setTimeout(() => {
      bridgeState.waiter = null;
      reject(new BridgeValidationError('The extension did not respond in time.', 'timeout'));
    }, Math.min(RESPONSE_TIMEOUT_MS, remaining));
    bridgeState.waiter = { type, resolve, reject, timer };
  });

  const assertEnvelopeKeys = (message, extraKeys, label) => {
    assertExactKeys(message, new Set([
      'source',
      'channel',
      'protocolVersion',
      'type',
      'captureId',
      'requestId',
      'channelNonce',
      ...extraKeys
    ]), label);
  };

  const requestFile = async (descriptor, generation, deadline) => {
    const chunks = [];
    let receivedBytes = 0;
    for (let chunkIndex = 0; chunkIndex < descriptor.totalChunks; chunkIndex += 1) {
      if (generation !== bridgeState.generation || !bridgeState.authenticated) {
        throw new BridgeValidationError('Capture transfer was cancelled.');
      }
      const responsePromise = waitForResponse('capture-file-chunk', deadline);
      postToExtension('capture-file-chunk-request', {
        payload: {
          fileId: descriptor.fileId,
          chunkIndex,
          chunkSize: CHUNK_BYTES
        }
      });
      const response = await responsePromise;
      assertEnvelopeKeys(response, ['payload'], 'file chunk response');
      assertExactKeys(response.payload, new Set(['fileId', 'chunkIndex', 'totalChunks', 'data']), 'file chunk payload');
      const chunkPayload = response.payload;
      if (
        chunkPayload.fileId !== descriptor.fileId
        || chunkPayload.chunkIndex !== chunkIndex
        || chunkPayload.totalChunks !== descriptor.totalChunks
      ) {
        throw new BridgeValidationError('A file chunk does not match the requested file.');
      }
      const expectedBytes = chunkIndex === descriptor.totalChunks - 1
        ? descriptor.size - (chunkIndex * CHUNK_BYTES)
        : CHUNK_BYTES;
      const chunk = decodeBase64Chunk(chunkPayload.data, expectedBytes);
      chunks.push(chunk);
      receivedBytes += chunk.length;
    }
    if (receivedBytes !== descriptor.size) throw new BridgeValidationError(`${descriptor.name} has an invalid size.`);
    const bytes = new Uint8Array(descriptor.size);
    let offset = 0;
    chunks.forEach((chunk) => {
      bytes.set(chunk, offset);
      offset += chunk.length;
    });
    const digest = await calculateSha256(bytes);
    if (digest !== descriptor.sha256) {
      throw new BridgeValidationError(`${descriptor.name} failed its integrity check.`, 'hash-mismatch');
    }
    return {
      kind: descriptor.kind,
      file: new global.File([bytes], descriptor.name, { type: descriptor.type, lastModified: Date.now() })
    };
  };

  const renderReview = (pending) => {
    setVisible(handoffBanner, false);
    setVisible(reviewBanner, true);
    if (reviewTitle) reviewTitle.textContent = 'Review Copilot capture before importing';
    if (reviewSummary) {
      reviewSummary.textContent = `${pending.job.title} at ${pending.job.company}. Review the metadata below before any file bytes are requested from the extension.`;
    }
    if (reviewFiles) {
      reviewFiles.setAttribute('aria-label', 'Files listed for reviewed transfer');
      reviewFiles.replaceChildren();
      if (!pending.descriptors.length) {
        const item = document.createElement('li');
        item.textContent = 'No files included.';
        reviewFiles.appendChild(item);
      } else {
        pending.descriptors.forEach((descriptor) => {
          const item = document.createElement('li');
          item.textContent = `${descriptor.kind === 'resume' ? 'Resume' : 'Cover letter'}: ${descriptor.name} (${formatBytes(descriptor.size)})`;
          reviewFiles.appendChild(item);
        });
      }
    }
    if (reviewButton) {
      reviewButton.hidden = false;
      reviewButton.disabled = false;
    }
    setStatus(reviewStatus, 'Click Review in application form to transfer and verify the listed files, then prefill the form.', 'info');
  };

  const showHandoffState = (message, tone = 'info', { retry = false } = {}) => {
    setVisible(reviewBanner, false);
    setVisible(handoffBanner, true);
    setStatus(handoffStatus, message, tone);
    if (retryButton) retryButton.hidden = !retry;
  };

  const startTransfer = async () => {
    if (!captureId || !bridgeState.authenticated || bridgeState.active || bridgeState.pending) return;
    if (!channelNonce) {
      showHandoffState('This browser cannot create a secure Copilot handoff channel.', 'error');
      return;
    }
    const generation = bridgeState.generation + 1;
    bridgeState.generation = generation;
    bridgeState.requestId = randomUuid();
    bridgeState.active = true;
    showHandoffState('Securely requesting the Copilot capture metadata...', 'info');
    const deadline = Date.now() + TRANSFER_TIMEOUT_MS;
    try {
      const responsePromise = waitForResponse('capture-manifest', deadline);
      postToExtension('capture-manifest-request', { requestedAt: new Date().toISOString() });
      const response = await responsePromise;
      assertEnvelopeKeys(response, ['payload'], 'capture manifest response');
      const manifest = validateManifest(response.payload, { maxFileBytes });
      if (generation !== bridgeState.generation || !bridgeState.authenticated) return;
      bridgeState.pending = {
        captureId,
        job: manifest.job,
        descriptors: manifest.files,
        expiresAt: manifest.expiresAt,
        files: [],
        filesVerified: false
      };
      bridgeState.active = false;
      bridgeState.reviewed = false;
      renderReview(bridgeState.pending);
    } catch (error) {
      if (generation !== bridgeState.generation) return;
      bridgeState.active = false;
      bridgeState.pending = null;
      const message = error instanceof BridgeValidationError
        ? error.message
        : 'The Copilot capture could not be verified.';
      showHandoffState(message, 'error', { retry: bridgeState.authenticated });
    }
  };

  const transferPendingFiles = async () => {
    const pending = bridgeState.pending;
    if (!pending || !bridgeState.authenticated) {
      throw new BridgeValidationError('Sign in again before reviewing this capture.');
    }
    if (pending.filesVerified) return pending.files;
    if (bridgeState.active) throw new BridgeValidationError('The capture transfer is already in progress.');
    if (Date.parse(pending.expiresAt) <= Date.now()) {
      throw new BridgeValidationError('This capture expired before file transfer. Reopen it from the extension.', 'stale-capture');
    }

    const generation = bridgeState.generation + 1;
    bridgeState.generation = generation;
    bridgeState.requestId = randomUuid();
    bridgeState.active = true;
    pending.files = [];
    pending.filesVerified = false;
    if (reviewButton) reviewButton.disabled = true;
    setStatus(reviewStatus, pending.descriptors.length
      ? 'Transferring and verifying the approved original files...'
      : 'Preparing the reviewed capture...', 'info');
    const deadline = Math.min(Date.now() + TRANSFER_TIMEOUT_MS, Date.parse(pending.expiresAt));

    try {
      const files = [];
      for (const descriptor of pending.descriptors) {
        if (Date.parse(pending.expiresAt) <= Date.now()) {
          throw new BridgeValidationError('This capture expired during file transfer. Reopen it from the extension.', 'stale-capture');
        }
        files.push(await requestFile(descriptor, generation, deadline));
      }
      if (generation !== bridgeState.generation || !bridgeState.authenticated) {
        throw new BridgeValidationError('Capture transfer was cancelled.');
      }
      pending.files = files;
      pending.filesVerified = true;
      return files;
    } catch (error) {
      pending.files = [];
      pending.filesVerified = false;
      throw error;
    } finally {
      if (generation === bridgeState.generation) bridgeState.active = false;
      if (reviewButton) reviewButton.disabled = false;
    }
  };

  const clearPending = () => {
    bridgeState.generation += 1;
    bridgeState.active = false;
    bridgeState.pending = null;
    bridgeState.reviewed = false;
    rejectWaiter(new BridgeValidationError('Capture transfer was cancelled.'));
  };

  const dismiss = () => {
    if (!hasCaptureParam) return;
    const wasReviewed = bridgeState.reviewed;
    if (captureId && channelNonce && bridgeState.requestId) {
      postToExtension('capture-dismissed', { payload: { dismissedAt: new Date().toISOString() } });
    }
    clearPending();
    removeCaptureParameter();
    setVisible(handoffBanner, false);
    setVisible(reviewBanner, false);
    if (typeof handlers.onDismiss === 'function') handlers.onDismiss({ wasReviewed });
  };

  const applyPendingToForm = () => {
    const pending = bridgeState.pending;
    if (!pending || !pending.filesVerified || !bridgeState.authenticated) return false;
    const job = pending.job;
    const changedElements = new Set();
    const applicationType = document.querySelector('#jobtrack-entry-type-application');
    const prospectType = document.querySelector('#jobtrack-entry-type-prospect');
    if (!applicationType) throw new BridgeValidationError('The Application entry type is unavailable.');
    const fileAssignments = pending.files.map(({ file, kind }) => {
      if (typeof global.DataTransfer !== 'function') {
        throw new BridgeValidationError('This browser cannot place verified files into the form.');
      }
      const input = document.querySelector(kind === 'resume' ? '#jobtrack-resume' : '#jobtrack-cover');
      if (!input) throw new BridgeValidationError('An attachment field is unavailable.');
      const transfer = new global.DataTransfer();
      transfer.items.add(file);
      return { input, files: transfer.files };
    });
    const setValue = (selector, value) => {
      const element = document.querySelector(selector);
      if (element && value !== undefined) {
        element.value = value;
        changedElements.add(element);
      }
    };
    applicationType.checked = true;
    changedElements.add(applicationType);
    if (prospectType) prospectType.checked = false;
    setValue('#jobtrack-company', job.company);
    setValue('#jobtrack-title', job.title);
    setValue('#jobtrack-job-url', job.jobUrl || '');
    setValue('#jobtrack-location', job.location || '');
    setValue('#jobtrack-source', job.source || '');
    setValue('#jobtrack-posting-date', job.postingDate || '');
    if (job.appliedDate) setValue('#jobtrack-date', job.appliedDate);
    setValue('#jobtrack-status', job.status || 'Applied');
    setValue('#jobtrack-tags', Array.isArray(job.tags) ? job.tags.join(', ') : '');
    const postingUnknown = document.querySelector('#jobtrack-posting-unknown');
    if (postingUnknown) {
      postingUnknown.checked = !job.postingDate;
      changedElements.add(postingUnknown);
    }
    if (pending.files.length) {
      fileAssignments.forEach(({ input, files }) => {
        input.files = files;
        changedElements.add(input);
      });
      const details = document.querySelector('.jobtrack-form-details');
      if (details) details.open = true;
    }
    bridgeState.reviewed = true;
    changedElements.forEach((element) => {
      element.dispatchEvent(new global.Event('input', { bubbles: true }));
      element.dispatchEvent(new global.Event('change', { bubbles: true }));
    });
    if (reviewButton) reviewButton.hidden = true;
    if (reviewTitle) reviewTitle.textContent = 'Copilot capture ready for your final review';
    setStatus(reviewStatus, 'Review every field and selected file below, then click Save application. The extension is not acknowledged until the application and all attachments finish saving.', 'success');
    return true;
  };

  const getPendingCapture = () => {
    if (!bridgeState.reviewed || !bridgeState.pending) return null;
    return {
      captureId: bridgeState.pending.captureId,
      job: { ...bridgeState.pending.job },
      files: bridgeState.pending.files.map(item => ({ ...item }))
    };
  };

  const complete = async (applicationId) => {
    if (!bridgeState.reviewed || !bridgeState.pending) return false;
    if (bridgeState.active) throw new BridgeValidationError('Another capture request is already in progress.');
    const normalizedApplicationId = normalizeText(applicationId, 'applicationId', {
      required: true,
      maxLength: 256
    });
    const pending = bridgeState.pending;
    const expiresAt = Date.parse(pending.expiresAt);
    if (!Number.isFinite(expiresAt) || expiresAt <= Date.now()) {
      throw new BridgeValidationError('This capture expired before acknowledgement. Reopen it from the extension.', 'stale-capture');
    }
    const generation = bridgeState.generation + 1;
    bridgeState.generation = generation;
    bridgeState.requestId = randomUuid();
    bridgeState.active = true;
    const deadline = Math.min(Date.now() + RESPONSE_TIMEOUT_MS, expiresAt);
    try {
      const responsePromise = waitForResponse('capture-complete-ack', deadline);
      postToExtension('capture-complete', {
        payload: {
          applicationId: normalizedApplicationId,
          completedAt: new Date().toISOString()
        }
      });
      const response = await responsePromise;
      assertEnvelopeKeys(response, ['payload'], 'capture completion acknowledgement');
      assertExactKeys(
        response.payload,
        new Set(['accepted', 'applicationId', 'acknowledgedAt', 'reason']),
        'capture completion acknowledgement payload'
      );
      if (typeof response.payload.accepted !== 'boolean') {
        throw new BridgeValidationError('The extension returned an invalid completion decision.');
      }
      if (response.payload.applicationId !== normalizedApplicationId) {
        throw new BridgeValidationError('The extension acknowledged a different application.');
      }
      normalizeIsoTimestamp(response.payload.acknowledgedAt, 'acknowledgedAt');
      if (typeof response.payload.reason !== 'string' || response.payload.reason.length > 160) {
        throw new BridgeValidationError('The extension returned an invalid completion reason.');
      }
      if (response.payload.accepted ? response.payload.reason !== '' : !response.payload.reason.trim()) {
        throw new BridgeValidationError('The extension returned an inconsistent completion decision.');
      }
      if (!response.payload.accepted) throw new BridgeValidationError(response.payload.reason, 'completion-rejected');
      if (generation !== bridgeState.generation || bridgeState.pending !== pending || !bridgeState.reviewed) {
        throw new BridgeValidationError('Capture acknowledgement was superseded.');
      }
      clearPending();
      removeCaptureParameter();
      setVisible(handoffBanner, false);
      setVisible(reviewBanner, false);
      return true;
    } finally {
      if (generation === bridgeState.generation) bridgeState.active = false;
    }
  };

  const configure = ({ onReview, onDismiss } = {}) => {
    handlers.onReview = typeof onReview === 'function' ? onReview : null;
    handlers.onDismiss = typeof onDismiss === 'function' ? onDismiss : null;
  };

  const setAuthenticated = (authenticated) => {
    const next = authenticated === true;
    if (bridgeState.authenticated === next) {
      if (next && !bridgeState.active && !bridgeState.pending) void startTransfer();
      return;
    }
    bridgeState.authenticated = next;
    if (!next) {
      const wasReviewed = bridgeState.reviewed;
      clearPending();
      if (wasReviewed && typeof handlers.onDismiss === 'function') handlers.onDismiss({ wasReviewed: true });
      if (captureId) showHandoffState('Sign in to load this Copilot capture. No capture data has been requested yet.', 'info');
      return;
    }
    void startTransfer();
  };

  if (reviewButton) {
    reviewButton.addEventListener('click', async () => {
      if (!bridgeState.authenticated) {
        setStatus(reviewStatus, 'Sign in again before reviewing this capture.', 'error');
        return;
      }
      try {
        await transferPendingFiles();
        const accepted = typeof handlers.onReview === 'function'
          ? handlers.onReview()
          : applyPendingToForm();
        if (accepted === false) throw new BridgeValidationError('The application form is not ready.');
      } catch (error) {
        setStatus(reviewStatus, error?.message || 'Unable to place the capture into the form.', 'error');
      }
    });
  }
  if (retryButton) retryButton.addEventListener('click', () => void startTransfer());
  if (handoffDismissButton) handoffDismissButton.addEventListener('click', () => dismiss());
  if (reviewDismissButton) reviewDismissButton.addEventListener('click', () => dismiss());

  const api = Object.freeze({
    applyPendingToForm,
    complete,
    configure,
    dismiss,
    getPendingCapture,
    hasPendingCapture: () => Boolean(bridgeState.reviewed && bridgeState.pending),
    setAuthenticated,
    protocol: Object.freeze({
      channel: CHANNEL,
      extensionSource: EXTENSION_SOURCE,
      protocolVersion: PROTOCOL_VERSION,
      websiteSource: WEBSITE_SOURCE
    }),
    __test: TEST_API
  });
  global.JobApplicationCopilotBridge = api;

  if (!hasCaptureParam) return;
  if (!captureId) {
    showHandoffState('The Copilot capture link is malformed. Reopen the capture from the extension.', 'error');
    if (retryButton) retryButton.hidden = true;
    return;
  }
  showHandoffState('Sign in to load this Copilot capture. No capture data has been requested yet.', 'info');
})(typeof window !== 'undefined' ? window : null);
