import {
  TRACKER_CHANNEL,
  TRACKER_EXTENSION_SOURCE,
  TRACKER_INTERNAL_MESSAGE_TYPES,
  TRACKER_MESSAGE_TYPES,
  TRACKER_PAGE_SOURCE,
  TRACKER_PROTOCOL_VERSION,
  TRACKER_TRANSFER_LIMITS,
  isTrackerPageUrl,
  validateManifestPayload,
  validateTrackerBeginMessage,
  validateTrackerWindowMessage
} from '../shared/tracker-protocol.js';
import { base64ToBytes, bytesToBase64, sha256Base64Url } from '../vault/crypto.js';
import { EncryptedIndexedDbVault } from '../vault/indexeddb-vault.js';

const PENDING_KEY_PREFIX = 'jobCopilotPendingTrackerCapture:';
const TRACKER_URL = 'https://www.danielshort.me/tools/job-application-tracker';

const pendingKey = (captureId) => `${PENDING_KEY_PREFIX}${captureId}`;

const assertActive = (pending, now) => {
  if (!pending) throw new Error('No pending tracker capture matches this request.');
  if (new Date(pending.expiresAt).getTime() <= now.getTime()) {
    throw new Error('The pending tracker capture has expired.');
  }
};

const assertPageSender = (sender, message, extensionId) => {
  if (!sender?.tab || sender.id !== extensionId || !isTrackerPageUrl(sender.url || sender.tab.url)) {
    throw new Error('Tracker capture requests are only accepted from the installed tracker content script.');
  }
  if (!Number.isSafeInteger(sender.tab.id) || sender.tab.id < 0 || sender.frameId !== 0) {
    throw new Error('Tracker capture requests are only accepted from the main frame of a concrete tracker tab.');
  }
  if (typeof sender.documentId !== 'string'
    || sender.documentId.length < 1
    || sender.documentId.length > 128
    || !/^[A-Za-z0-9._:~-]+$/u.test(sender.documentId)) {
    throw new Error('Tracker capture requests require a valid document identity.');
  }
  const url = new URL(sender.url || sender.tab.url);
  if (url.searchParams.size !== 1
    || url.searchParams.get('copilotCapture') !== message.captureId
    || url.hash) {
    throw new Error('Tracker URL capability parameters do not match the request.');
  }
  return { tabId: sender.tab.id, documentId: sender.documentId };
};

const assertTrustedExtensionSender = (sender, extensionId) => {
  if (sender?.tab || sender?.id !== extensionId || typeof sender.url !== 'string') {
    throw new Error('Only a trusted extension page can begin a tracker capture.');
  }
  const url = new URL(sender.url);
  if (url.protocol !== 'chrome-extension:' || url.hostname !== extensionId) {
    throw new Error('Only a trusted extension page can begin a tracker capture.');
  }
};

const transferableRecord = async (record, requestedFile) => {
  if (!record || record.kind !== 'document' || !record.value?.document) {
    throw new Error(`Vault record ${requestedFile.recordId} is not an imported document.`);
  }
  const { document, originalBytesBase64 } = record.value;
  if (typeof originalBytesBase64 !== 'string' || !originalBytesBase64) {
    throw new Error(`Vault record ${requestedFile.recordId} did not retain its original bytes.`);
  }
  if (originalBytesBase64.length > Math.ceil(TRACKER_TRANSFER_LIMITS.maxFileBytes / 3) * 4 + 4) {
    throw new Error(`Vault record ${requestedFile.recordId} exceeds the transfer limit.`);
  }
  let bytes;
  try {
    bytes = base64ToBytes(originalBytesBase64);
  } catch {
    throw new Error(`Vault record ${requestedFile.recordId} contains invalid original bytes.`);
  }
  if (bytesToBase64(bytes) !== originalBytesBase64) {
    throw new Error(`Vault record ${requestedFile.recordId} does not use canonical base64.`);
  }
  if (bytes.byteLength < 1 || bytes.byteLength > TRACKER_TRANSFER_LIMITS.maxFileBytes) {
    throw new Error(`Vault record ${requestedFile.recordId} exceeds the transfer limit.`);
  }
  if (document.size !== bytes.byteLength) throw new Error(`Vault record ${requestedFile.recordId} has inconsistent size metadata.`);
  const digest = await sha256Base64Url(bytes);
  if (document.sha256 !== digest) throw new Error(`Vault record ${requestedFile.recordId} failed its SHA-256 check.`);
  return {
    bytes,
    descriptor: {
      fileId: requestedFile.fileId,
      recordId: requestedFile.recordId,
      name: document.filename,
      type: document.mimeType,
      kind: requestedFile.kind,
      size: bytes.byteLength,
      sha256: digest,
      totalChunks: Math.ceil(bytes.byteLength / TRACKER_TRANSFER_LIMITS.chunkSize),
      chunkSize: TRACKER_TRANSFER_LIMITS.chunkSize
    }
  };
};

export class TrackerCaptureRouter {
  constructor({
    storageSession = globalThis.chrome?.storage?.session,
    storageLocal = globalThis.chrome?.storage?.local,
    extensionId = globalThis.chrome?.runtime?.id,
    vaultFactory = () => new EncryptedIndexedDbVault(),
    now = () => new Date()
  } = {}) {
    if (!storageSession) throw new Error('chrome.storage.session is required for tracker capture routing.');
    if (!storageLocal) throw new Error('chrome.storage.local is required for tracker capture routing.');
    if (!extensionId) throw new Error('The extension runtime ID is required for tracker capture routing.');
    this.storageSession = storageSession;
    this.storageLocal = storageLocal;
    this.extensionId = extensionId;
    this.vaultFactory = vaultFactory;
    this.now = now;
  }

  ownsMessage(message) {
    return message?.type === TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN || message?.channel === TRACKER_CHANNEL;
  }

  async initialize() {
    if (typeof this.storageSession.setAccessLevel === 'function') {
      await this.storageSession.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' });
    }
    if (typeof this.storageLocal.setAccessLevel === 'function') {
      await this.storageLocal.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' });
    }
  }

  async handle(message, sender) {
    if (message?.type === TRACKER_INTERNAL_MESSAGE_TYPES.BEGIN) {
      assertTrustedExtensionSender(sender, this.extensionId);
      return this.#begin(validateTrackerBeginMessage(message));
    }

    validateTrackerWindowMessage(message, { expectedSource: TRACKER_PAGE_SOURCE });
    const senderBinding = assertPageSender(sender, message, this.extensionId);
    const pending = await this.#getPending(message.captureId);
    assertActive(pending, this.now());
    if (pending.state === 'completed') {
      return this.#completedResponse(message, pending, senderBinding);
    }
    if (pending.state && pending.state !== 'pending') throw new Error('Tracker capture state is invalid.');
    if (pending.trackerTabId !== null && pending.trackerTabId !== senderBinding.tabId) {
      throw new Error('The tracker capture is bound to another tab.');
    }
    const staleDocumentIds = Array.isArray(pending.staleTrackerDocumentIds)
      ? pending.staleTrackerDocumentIds.filter(value => typeof value === 'string')
      : [];
    if (pending.trackerTabId === null) {
      if (message.type !== TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST) {
        throw new Error('A tracker document must request its manifest before capture data is available.');
      }
      pending.trackerTabId = senderBinding.tabId;
      pending.trackerDocumentId = senderBinding.documentId;
      pending.channelNonce = message.channelNonce;
      pending.staleTrackerDocumentIds = staleDocumentIds;
      await this.#setPending(pending);
    } else if (!pending.trackerDocumentId) {
      if (message.type !== TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST) {
        throw new Error('A tracker document must request its manifest before capture data is available.');
      }
      pending.trackerDocumentId = senderBinding.documentId;
      pending.channelNonce = message.channelNonce;
      pending.staleTrackerDocumentIds = staleDocumentIds;
      await this.#setPending(pending);
    } else if (pending.trackerDocumentId !== senderBinding.documentId) {
      if (staleDocumentIds.includes(senderBinding.documentId)) {
        throw new Error('The tracker capture request came from a stale tracker document.');
      }
      if (message.type !== TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST) {
        throw new Error('A fresh tracker document must request its manifest before capture data is available.');
      }
      staleDocumentIds.push(pending.trackerDocumentId);
      pending.staleTrackerDocumentIds = staleDocumentIds.slice(-32);
      pending.trackerDocumentId = senderBinding.documentId;
      pending.channelNonce = message.channelNonce;
      await this.#setPending(pending);
    } else if (pending.channelNonce !== message.channelNonce) {
      throw new Error('Tracker channel nonce does not match.');
    }

    if (message.type === TRACKER_MESSAGE_TYPES.MANIFEST_REQUEST) {
      return this.#manifestResponse(message, pending);
    }
    if (message.type === TRACKER_MESSAGE_TYPES.FILE_CHUNK_REQUEST) {
      return this.#chunkResponse(message, pending);
    }
    if (message.type === TRACKER_MESSAGE_TYPES.COMPLETE) {
      const receipt = {
        version: 1,
        state: 'completed',
        captureId: message.captureId,
        expiresAt: pending.expiresAt,
        trackerTabId: pending.trackerTabId,
        trackerDocumentId: pending.trackerDocumentId,
        channelNonce: pending.channelNonce,
        applicationId: message.payload.applicationId,
        acknowledgedAt: this.now().toISOString()
      };
      await this.#setPending(receipt);
      return this.#completionResponse(message, receipt, {
        accepted: true,
        reason: ''
      });
    }
    if (message.type === TRACKER_MESSAGE_TYPES.DISMISSED) {
      pending.lastDismissedAt = message.payload.dismissedAt;
      await this.#setPending(pending);
      return { ok: true, captureId: message.captureId, retained: true };
    }
    throw new Error('Tracker page cannot send this message type.');
  }

  async #begin(message) {
    const now = this.now();
    if (new Date(message.payload.issuedAt).getTime() > now.getTime() + 5 * 60 * 1000) {
      throw new Error('Tracker capture issuedAt is too far in the future.');
    }
    if (new Date(message.payload.expiresAt).getTime() <= now.getTime()) {
      throw new Error('Tracker capture expiresAt must be in the future.');
    }
    const existing = await this.#getPending(message.payload.captureId);
    if (existing) {
      const existingExpiresAt = new Date(existing.expiresAt).getTime();
      if (Number.isFinite(existingExpiresAt) && existingExpiresAt <= now.getTime()) {
        await this.storageSession.remove(pendingKey(message.payload.captureId));
      } else {
        throw new Error('A pending or completed tracker capture already uses this captureId.');
      }
    }

    const files = [];
    for (const requestedFile of message.payload.files) {
      const record = await this.#getVaultRecord(requestedFile.recordId);
      files.push((await transferableRecord(record, requestedFile)).descriptor);
    }
    const manifest = {
      issuedAt: message.payload.issuedAt,
      expiresAt: message.payload.expiresAt,
      job: message.payload.job,
      files: files.map(({ recordId, ...file }) => file)
    };
    validateManifestPayload(manifest);
    const pending = {
      version: 1,
      state: 'pending',
      captureId: message.payload.captureId,
      channelNonce: null,
      issuedAt: message.payload.issuedAt,
      expiresAt: message.payload.expiresAt,
      job: message.payload.job,
      files,
      trackerTabId: null,
      trackerDocumentId: null,
      staleTrackerDocumentIds: [],
      lastDismissedAt: null
    };
    await this.#setPending(pending);
    const trackerUrl = new URL(TRACKER_URL);
    trackerUrl.searchParams.set('copilotCapture', pending.captureId);
    return {
      ok: true,
      captureId: pending.captureId,
      expiresAt: pending.expiresAt,
      trackerUrl: trackerUrl.toString()
    };
  }

  #completedResponse(request, receipt, senderBinding) {
    if (request.type !== TRACKER_MESSAGE_TYPES.COMPLETE) {
      throw new Error('This tracker capture has already completed.');
    }
    if (receipt.trackerTabId !== senderBinding.tabId) {
      throw new Error('The completed tracker capture is bound to another tab.');
    }
    if (receipt.trackerDocumentId !== senderBinding.documentId) {
      throw new Error('The completed tracker capture is bound to another tracker document.');
    }
    if (receipt.channelNonce !== request.channelNonce) {
      throw new Error('Tracker channel nonce does not match the completed capture.');
    }
    const accepted = receipt.applicationId === request.payload.applicationId;
    return this.#completionResponse(request, receipt, {
      accepted,
      reason: accepted ? '' : 'This capture was already acknowledged for a different application.'
    });
  }

  #completionResponse(request, receipt, { accepted, reason }) {
    return validateTrackerWindowMessage({
      channel: TRACKER_CHANNEL,
      source: TRACKER_EXTENSION_SOURCE,
      protocolVersion: TRACKER_PROTOCOL_VERSION,
      type: TRACKER_MESSAGE_TYPES.COMPLETE_ACK,
      captureId: request.captureId,
      requestId: request.requestId,
      channelNonce: request.channelNonce,
      payload: {
        accepted,
        applicationId: request.payload.applicationId,
        acknowledgedAt: receipt.acknowledgedAt,
        reason
      }
    }, { expectedSource: TRACKER_EXTENSION_SOURCE });
  }

  #manifestResponse(request, pending) {
    return validateTrackerWindowMessage({
      channel: TRACKER_CHANNEL,
      source: TRACKER_EXTENSION_SOURCE,
      protocolVersion: TRACKER_PROTOCOL_VERSION,
      type: TRACKER_MESSAGE_TYPES.MANIFEST,
      captureId: request.captureId,
      requestId: request.requestId,
      channelNonce: request.channelNonce,
      payload: {
        issuedAt: pending.issuedAt,
        expiresAt: pending.expiresAt,
        job: pending.job,
        files: pending.files.map(({ recordId, ...file }) => file)
      }
    }, { expectedSource: TRACKER_EXTENSION_SOURCE });
  }

  async #chunkResponse(request, pending) {
    const file = pending.files.find(candidate => candidate.fileId === request.payload.fileId);
    if (!file) throw new Error('The requested fileId is not part of this capture.');
    if (request.payload.chunkIndex >= file.totalChunks) throw new Error('The requested chunkIndex is outside the file.');
    const record = await this.#getVaultRecord(file.recordId);
    const current = await transferableRecord(record, file);
    const expected = current.descriptor;
    if (expected.name !== file.name
      || expected.type !== file.type
      || expected.size !== file.size
      || expected.sha256 !== file.sha256
      || expected.totalChunks !== file.totalChunks) {
      throw new Error('The encrypted vault record changed after the capture began.');
    }
    const start = request.payload.chunkIndex * TRACKER_TRANSFER_LIMITS.chunkSize;
    const chunk = current.bytes.subarray(start, Math.min(start + TRACKER_TRANSFER_LIMITS.chunkSize, current.bytes.length));
    return validateTrackerWindowMessage({
      channel: TRACKER_CHANNEL,
      source: TRACKER_EXTENSION_SOURCE,
      protocolVersion: TRACKER_PROTOCOL_VERSION,
      type: TRACKER_MESSAGE_TYPES.FILE_CHUNK,
      captureId: request.captureId,
      requestId: request.requestId,
      channelNonce: request.channelNonce,
      payload: {
        fileId: file.fileId,
        chunkIndex: request.payload.chunkIndex,
        totalChunks: file.totalChunks,
        data: bytesToBase64(chunk)
      }
    }, { expectedSource: TRACKER_EXTENSION_SOURCE });
  }

  async #getVaultRecord(recordId) {
    const vault = this.vaultFactory();
    try {
      await vault.open();
      return await vault.getRecord(recordId);
    } finally {
      await vault.close({ lock: false });
    }
  }

  async #getPending(captureId) {
    const key = pendingKey(captureId);
    const values = await this.storageSession.get(key);
    return values?.[key] || null;
  }

  async #setPending(pending) {
    await this.storageSession.set({ [pendingKey(pending.captureId)]: pending });
  }
}

export const TRACKER_PENDING_KEY_PREFIX = PENDING_KEY_PREFIX;
