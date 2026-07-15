(() => {
  'use strict';

  const TOOL_ID = 'transcribe';
  const API_BASE = '/api/tools/transcribe';
  const DEFAULT_CONFIG = {
    configured: false,
    service: 'Amazon Transcribe',
    region: 'us-east-2',
    languageCode: 'en-US',
    pricePerSecond: 0.0004,
    pricePerMinute: 0.024,
    minDurationSeconds: 15,
    minBillableSeconds: 15,
    maxFilesPerRun: 10,
    maxFileBytes: 500 * 1024 * 1024,
    maxTotalCostUsd: 10,
    supportedFormats: ['amr', 'flac', 'm4a', 'mp3', 'mp4', 'ogg', 'wav', 'webm']
  };
  const VIDEO_FORMATS = new Set(['mp4', 'webm']);
  const POLL_INTERVAL_MS = 5000;
  const POLL_MEDIUM_INTERVAL_MS = 15000;
  const POLL_MAX_INTERVAL_MS = 30000;
  const MAX_TRANSIENT_RETRIES = 5;
  const ACTIVE_RUNS_STORAGE_KEY = 'tools:transcribe:active-runs:v1';
  const ACTIVE_RUNS_MAX_ITEMS = 10;
  const ACTIVE_RUNS_MAX_TOKEN_CHARS = 8192;
  const DURATION_TIMEOUT_MS = 10000;
  const MAX_CONTAINER_SCAN_BYTES = 8 * 1024 * 1024;
  const MP4_SUSPICIOUS_STTS_DELTA_SECONDS = 120;
  const MP4_TIMELINE_OVERFLOW_SECONDS = 120;
  const MP4_CONTAINER_BOXES = new Set(['moov', 'trak', 'mdia', 'minf', 'dinf', 'stbl', 'edts', 'udta', 'meta']);
  const WEBM_CONTAINER_IDS = new Set([0x18538067, 0x1654AE6B, 0xAE]);

  const isMp4TimelineSuspicious = ({ maxDeltaSeconds, timelineSeconds, mediaDurationSeconds }) =>
    Number(maxDeltaSeconds) >= MP4_SUSPICIOUS_STTS_DELTA_SECONDS &&
    Number(timelineSeconds) - Number(mediaDurationSeconds) >= MP4_TIMELINE_OVERFLOW_SECONDS;

  if (typeof module !== 'undefined' && module.exports && typeof document === 'undefined') {
    module.exports = { isMp4TimelineSuspicious };
    return;
  }

  const $id = (id) => document.getElementById(id);

  const authPillEl = $id('transcribe-auth-pill');
  const authStatusEl = $id('transcribe-auth-status');
  const signInBtn = $id('transcribe-sign-in');
  const shellEl = $id('transcribe-shell');
  const uploadViewEl = $id('transcribe-upload-view');
  const processingViewEl = $id('transcribe-processing-view');
  const resultsViewEl = $id('transcribe-results-view');
  const dropzoneEl = $id('transcribe-dropzone');
  const serviceEl = $id('transcribe-stat-service');
  const priceEl = $id('transcribe-stat-price');
  const minimumEl = $id('transcribe-stat-minimum');
  const acceptedEl = $id('transcribe-stat-accepted');
  const formEl = $id('transcribe-form');
  const fileEl = $id('transcribe-files');
  const summaryEl = $id('transcribe-summary');
  const tableWrapEl = $id('transcribe-table-wrap');
  const fileRowsEl = $id('transcribe-file-rows');
  const totalEl = $id('transcribe-total');
  const approveEl = $id('transcribe-approve');
  const startBtn = $id('transcribe-start');
  const cancelBtn = $id('transcribe-cancel');
  const resetBtn = $id('transcribe-reset');
  const newBtn = $id('transcribe-new');
  const runStatusEl = $id('transcribe-run-status');
  const processingCopyEl = $id('transcribe-processing-copy');
  const processingRowsEl = $id('transcribe-processing-rows');
  const resultsSummaryEl = $id('transcribe-results-summary');
  const progressWrapEl = $id('transcribe-progress-wrap');
  const progressLabelEl = $id('transcribe-progress-label');
  const progressBarEl = $id('transcribe-progress');
  const resultsEl = $id('transcribe-results');

  if (!formEl || !fileEl || !startBtn || !fileRowsEl) return;

  const state = {
    config: { ...DEFAULT_CONFIG },
    files: [],
    busy: false,
    canceled: false,
    activeXhr: null,
    activeController: null,
    analyzing: false,
    view: 'upload'
  };

  const getAuthSub = () => {
    try {
      const authApi = window.ToolsAuth;
      const auth = authApi?.getAuth?.();
      if (!authApi?.authIsValid?.(auth)) return '';
      return String(authApi?.getUser?.(auth)?.sub || '').trim();
    } catch {
      return '';
    }
  };

  const tokenExpiryMs = (token) => {
    try {
      const body = String(token || '').split('.')[0];
      if (!body) return 0;
      const normalized = body.replace(/-/g, '+').replace(/_/g, '/');
      const decoded = atob(normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '='));
      const bytes = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
      const json = typeof TextDecoder === 'function' ? new TextDecoder().decode(bytes) : decoded;
      const payload = JSON.parse(json);
      const expiresAt = Number(payload?.exp) * 1000;
      return Number.isFinite(expiresAt) ? expiresAt : 0;
    } catch {
      return 0;
    }
  };

  const clearActiveRunRecovery = () => {
    try {
      window.sessionStorage.removeItem(ACTIVE_RUNS_STORAGE_KEY);
    } catch {}
  };

  const persistActiveRunRecovery = () => {
    const ownerSub = getAuthSub();
    if (!ownerSub) return;
    const now = Date.now();
    const items = state.files
      .filter((item) => {
        if (['complete', 'partial'].includes(item?.status) || item?.runErrorType === 'service') return false;
        return Boolean(item?.runToken) || (Boolean(item?.quoteToken) && item?.uploadComplete === true);
      })
      .slice(0, ACTIVE_RUNS_MAX_ITEMS)
      .map((item) => {
        const runToken = String(item.runToken || '').slice(0, ACTIVE_RUNS_MAX_TOKEN_CHARS);
        const quoteToken = String(item.quoteToken || '').slice(0, ACTIVE_RUNS_MAX_TOKEN_CHARS);
        const expiresAt = tokenExpiryMs(runToken || quoteToken);
        if (!expiresAt || expiresAt <= now) return null;
        return {
          id: String(item.id || '').slice(0, 120),
          name: String(item.name || 'media').slice(0, 180),
          extension: String(item.extension || '').slice(0, 16),
          contentType: String(item.contentType || '').slice(0, 120),
          bytes: Math.max(0, Number(item.bytes) || 0),
          durationSeconds: Math.max(0, Number(item.durationSeconds) || 0),
          billableSeconds: Math.max(0, Number(item.billableSeconds) || 0),
          estimatedCostUsd: Math.max(0, Number(item.estimatedCostUsd) || 0),
          runToken,
          quoteToken,
          uploadComplete: item.uploadComplete === true,
          pollStartedAt: Math.max(0, Number(item.pollStartedAt) || 0),
          expiresAt
        };
      })
      .filter(Boolean);
    if (!items.length) {
      clearActiveRunRecovery();
      return;
    }
    try {
      window.sessionStorage.setItem(ACTIVE_RUNS_STORAGE_KEY, JSON.stringify({ ownerSub, items }));
    } catch {}
  };

  const restoreActiveRunRecovery = () => {
    if (state.files.length) return 0;
    const ownerSub = getAuthSub();
    if (!ownerSub) return 0;
    let stored;
    try {
      stored = JSON.parse(window.sessionStorage.getItem(ACTIVE_RUNS_STORAGE_KEY) || 'null');
    } catch {
      clearActiveRunRecovery();
      return 0;
    }
    if (!stored || stored.ownerSub !== ownerSub || !Array.isArray(stored.items)) {
      if (stored?.ownerSub && stored.ownerSub !== ownerSub) clearActiveRunRecovery();
      return 0;
    }
    const now = Date.now();
    const restored = stored.items.slice(0, ACTIVE_RUNS_MAX_ITEMS).map((item, index) => {
      const runToken = String(item?.runToken || '').slice(0, ACTIVE_RUNS_MAX_TOKEN_CHARS);
      const quoteToken = String(item?.quoteToken || '').slice(0, ACTIVE_RUNS_MAX_TOKEN_CHARS);
      const expiresAt = tokenExpiryMs(runToken || quoteToken);
      if (!expiresAt || expiresAt <= now) return null;
      if (!runToken && (!quoteToken || item?.uploadComplete !== true)) return null;
      return {
        id: String(item?.id || `recovered-${index}-${Date.now()}`).slice(0, 120),
        fingerprint: '',
        file: null,
        name: String(item?.name || `Recovered file ${index + 1}`).slice(0, 180),
        extension: String(item?.extension || '').slice(0, 16),
        contentType: String(item?.contentType || 'application/octet-stream').slice(0, 120),
        bytes: Math.max(0, Number(item?.bytes) || 0),
        durationSeconds: Math.max(0, Number(item?.durationSeconds) || 0),
        billableSeconds: Math.max(0, Number(item?.billableSeconds) || 0),
        estimatedCostUsd: Math.max(0, Number(item?.estimatedCostUsd) || 0),
        costUsd: 0,
        progress: 0,
        status: 'failed',
        error: 'Recovered after this tab reloaded. Select Resume to continue the existing job.',
        runErrorType: 'network',
        transcript: '',
        runToken,
        quoteToken,
        uploadComplete: item?.uploadComplete === true,
        pollStartedAt: Math.max(0, Number(item?.pollStartedAt) || 0)
      };
    }).filter(Boolean);
    state.files = restored;
    if (!restored.length) clearActiveRunRecovery();
    else persistActiveRunRecovery();
    return restored.length;
  };

  const markSessionDirty = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:session-dirty', { detail: { toolId: TOOL_ID } }));
    } catch {}
  };

  const reportRunComplete = (resultBucket) => {
    try {
      document.dispatchEvent(new CustomEvent('tools:run-complete', {
        detail: { toolId: TOOL_ID, resultBucket }
      }));
    } catch {}
  };

  const reportRunError = (errorType) => {
    try {
      document.dispatchEvent(new CustomEvent('tools:run-error', {
        detail: { toolId: TOOL_ID, errorType }
      }));
    } catch {}
  };

  const reportRunCancel = () => {
    try {
      document.dispatchEvent(new CustomEvent('tools:run-cancel', {
        detail: { toolId: TOOL_ID }
      }));
    } catch {}
  };

  const escapeHtml = (value) => String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  const cleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();

  const classifyRunError = (error) => {
    const status = Number(error?.status || 0);
    const message = cleanText(error?.message).toLowerCase();
    const terminalStatus = cleanText(error?.data?.status).toUpperCase();
    if (terminalStatus === 'MISSING' || status === 404 || status === 410) return 'service';
    if (status === 401 || status === 403 || /sign in|not authorized|forbidden/.test(message)) return 'permission';
    if (status === 408 || status === 504 || /timed? out|timeout/.test(message)) return 'timeout';
    if (status >= 500 || /service|transcription failed|request failed/.test(message)) return 'service';
    if (/failed to fetch|network|connection|offline|upload failed|load failed/.test(message)) return 'network';
    return 'processing';
  };

  const setText = (el, value) => {
    if (el) el.textContent = value || '';
  };

  const setStatus = (el, message, tone) => {
    if (!el) return;
    el.textContent = message || '';
    el.dataset.tone = tone || '';
  };

  const setAuthPill = (stateName, label) => {
    if (!authPillEl) return;
    authPillEl.dataset.state = stateName || '';
    authPillEl.textContent = label || '';
  };

  const formatNumber = (value, digits = 2) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return '--';
    return num.toFixed(digits).replace(/\.00$/, '');
  };

  const formatBytes = (bytes) => {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return '--';
    const mb = value / (1024 * 1024);
    if (mb >= 1024) return `${formatNumber(mb / 1024, 2)} GB`;
    if (mb >= 1) return `${formatNumber(mb, 2)} MB`;
    return `${formatNumber(value / 1024, 1)} KB`;
  };

  const formatClock = (totalSeconds) => {
    const seconds = Math.max(0, Math.round(Number(totalSeconds) || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    const pad = (value) => String(value).padStart(2, '0');
    if (hours > 0) return `${hours}:${pad(minutes)}:${pad(secs)}`;
    return `${pad(minutes)}:${pad(secs)}`;
  };

  const formatUsd = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return '$0.00';
    const digits = num < 0.01 ? 4 : 2;
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }).format(num);
  };

  const getExtension = (name) => {
    const match = String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/);
    return match ? match[1] : '';
  };

  const safeDownloadName = (name) => {
    const base = String(name || 'transcript').replace(/\.[^.]+$/, '') || 'transcript';
    return `${base.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'transcript'}-transcript.txt`;
  };

  const fileFingerprint = (file) => [
    String(file?.name || ''),
    String(Number(file?.size) || 0),
    String(Number(file?.lastModified) || 0)
  ].join('\u0001');

  const createFileItem = (file, index) => ({
    id: `${Date.now()}-${index}-${Math.random().toString(36).slice(2)}`,
    fingerprint: fileFingerprint(file),
    file,
    name: file.name || `file-${index + 1}`,
    extension: getExtension(file.name || ''),
    contentType: file.type || 'application/octet-stream',
    bytes: Number(file.size) || 0,
    durationSeconds: null,
    billableSeconds: 0,
    estimatedCostUsd: 0,
    costUsd: 0,
    progress: 0,
    status: 'checking',
    skipReason: '',
    transcript: ''
  });

  const supportedFormats = () => new Set(
    Array.isArray(state.config.supportedFormats)
      ? state.config.supportedFormats.map((item) => String(item).toLowerCase())
      : DEFAULT_CONFIG.supportedFormats
  );

  const readAscii = (view, offset, length) => {
    if (!view || offset < 0 || offset + length > view.byteLength) return '';
    let value = '';
    for (let i = 0; i < length; i += 1) {
      value += String.fromCharCode(view.getUint8(offset + i));
    }
    return value;
  };

  const readSlice = async (file, start, length) => {
    const safeStart = Math.max(0, Number(start) || 0);
    const safeLength = Math.max(0, Math.min(Number(length) || 0, file.size - safeStart));
    if (!safeLength || typeof file.slice !== 'function' || typeof file.slice(safeStart, safeStart + safeLength).arrayBuffer !== 'function') {
      return null;
    }
    return file.slice(safeStart, safeStart + safeLength).arrayBuffer();
  };

  const readMp4BoxHeader = (view, offset) => {
    if (!view || offset + 8 > view.byteLength) return null;
    let size = view.getUint32(offset);
    const type = readAscii(view, offset + 4, 4);
    let headerSize = 8;
    if (size === 1) {
      if (offset + 16 > view.byteLength) return null;
      const high = view.getUint32(offset + 8);
      const low = view.getUint32(offset + 12);
      size = high * 4294967296 + low;
      headerSize = 16;
    } else if (size === 0) {
      size = view.byteLength - offset;
    }
    if (!type || !Number.isFinite(size) || size < headerSize) return null;
    return { type, size, headerSize };
  };

  const listMp4ChildBoxes = (view, start, end) => {
    const children = [];
    let offset = start;
    while (offset + 8 <= end && offset + 8 <= view.byteLength) {
      const box = readMp4BoxHeader(view, offset);
      if (!box || offset + box.size > end || offset + box.size > view.byteLength) break;
      children.push({
        ...box,
        offset,
        payloadStart: offset + box.headerSize,
        payloadEnd: offset + box.size
      });
      offset += box.size;
    }
    return children;
  };

  const findMp4ChildBox = (view, parent, type) => listMp4ChildBoxes(
    view,
    parent.payloadStart,
    parent.payloadEnd
  ).find((box) => box.type === type) || null;

  const readMp4AudioTimeline = (view, audioMdia) => {
    const mdhd = findMp4ChildBox(view, audioMdia, 'mdhd');
    const minf = findMp4ChildBox(view, audioMdia, 'minf');
    const stbl = minf ? findMp4ChildBox(view, minf, 'stbl') : null;
    const stts = stbl ? findMp4ChildBox(view, stbl, 'stts') : null;
    if (!mdhd || !stts || mdhd.payloadStart + 20 > mdhd.payloadEnd || stts.payloadStart + 8 > stts.payloadEnd) {
      return { checked: false, malformedTimeline: false };
    }

    const mdhdVersion = view.getUint8(mdhd.payloadStart);
    const timescaleOffset = mdhdVersion === 1 ? mdhd.payloadStart + 20 : mdhd.payloadStart + 12;
    const durationOffset = mdhdVersion === 1 ? mdhd.payloadStart + 24 : mdhd.payloadStart + 16;
    const durationBytes = mdhdVersion === 1 ? 8 : 4;
    if (timescaleOffset + 4 > mdhd.payloadEnd || durationOffset + durationBytes > mdhd.payloadEnd) {
      return { checked: false, malformedTimeline: false };
    }

    const timescale = view.getUint32(timescaleOffset);
    const mediaDurationTicks = mdhdVersion === 1
      ? view.getUint32(durationOffset) * 4294967296 + view.getUint32(durationOffset + 4)
      : view.getUint32(durationOffset);
    const entryCount = view.getUint32(stts.payloadStart + 4);
    const entriesStart = stts.payloadStart + 8;
    if (!timescale || entryCount > Math.floor((stts.payloadEnd - entriesStart) / 8)) {
      return { checked: false, malformedTimeline: false };
    }

    let timelineTicks = 0;
    let maxDeltaTicks = 0;
    for (let index = 0; index < entryCount; index += 1) {
      const entryOffset = entriesStart + index * 8;
      const sampleCount = view.getUint32(entryOffset);
      const sampleDelta = view.getUint32(entryOffset + 4);
      timelineTicks += sampleCount * sampleDelta;
      maxDeltaTicks = Math.max(maxDeltaTicks, sampleDelta);
    }

    const mediaDurationSeconds = mediaDurationTicks / timescale;
    const timelineSeconds = timelineTicks / timescale;
    const maxDeltaSeconds = maxDeltaTicks / timescale;
    const malformedTimeline = isMp4TimelineSuspicious({
      maxDeltaSeconds,
      timelineSeconds,
      mediaDurationSeconds
    });
    return {
      checked: true,
      malformedTimeline,
      mediaDurationSeconds,
      timelineSeconds,
      maxDeltaSeconds
    };
  };

  const inspectMp4Structure = (view) => {
    const moov = listMp4ChildBoxes(view, 0, view.byteLength).find((box) => box.type === 'moov');
    if (!moov) return { checked: false, hasAudio: true, malformedTimeline: false };
    const tracks = listMp4ChildBoxes(view, moov.payloadStart, moov.payloadEnd)
      .filter((box) => box.type === 'trak');
    let hasAudio = false;
    for (const track of tracks) {
      const mdia = findMp4ChildBox(view, track, 'mdia');
      const hdlr = mdia ? findMp4ChildBox(view, mdia, 'hdlr') : null;
      if (!mdia || !hdlr || readAscii(view, hdlr.payloadStart + 8, 4) !== 'soun') continue;
      hasAudio = true;
      const timeline = readMp4AudioTimeline(view, mdia);
      if (timeline.malformedTimeline) {
        return { checked: true, hasAudio: true, ...timeline };
      }
    }
    return { checked: true, hasAudio, malformedTimeline: false };
  };

  const mp4BoxesContainAudio = (view, start, end, depth = 0) => {
    if (!view || depth > 8) return false;
    let offset = start;
    while (offset + 8 <= end && offset + 8 <= view.byteLength) {
      const box = readMp4BoxHeader(view, offset);
      if (!box || offset + box.size > end || offset + box.size > view.byteLength) return false;
      const payloadStart = offset + box.headerSize;
      const payloadEnd = offset + box.size;
      if (box.type === 'hdlr') {
        const handlerType = readAscii(view, payloadStart + 8, 4);
        if (handlerType === 'soun') return true;
      }
      if (MP4_CONTAINER_BOXES.has(box.type)) {
        const childStart = box.type === 'meta' ? payloadStart + 4 : payloadStart;
        if (childStart < payloadEnd && mp4BoxesContainAudio(view, childStart, payloadEnd, depth + 1)) return true;
      }
      offset += box.size;
    }
    return false;
  };

  const inspectMp4AudioTrack = async (file) => {
    let offset = 0;
    while (offset + 8 <= file.size) {
      const headerBuffer = await readSlice(file, offset, 16);
      if (!headerBuffer) return { checked: false, hasAudio: true };
      const headerView = new DataView(headerBuffer);
      const box = readMp4BoxHeader(headerView, 0);
      if (!box) return { checked: false, hasAudio: true };
      if (box.type === 'moov') {
        if (box.size > MAX_CONTAINER_SCAN_BYTES) return { checked: false, hasAudio: true };
        const moovBuffer = await readSlice(file, offset, box.size);
        if (!moovBuffer) return { checked: false, hasAudio: true };
        const moovView = new DataView(moovBuffer);
        const structure = inspectMp4Structure(moovView);
        if (structure.checked) return structure;
        return { checked: true, hasAudio: mp4BoxesContainAudio(moovView, 0, moovView.byteLength) };
      }
      if (!Number.isFinite(box.size) || box.size <= 0) break;
      offset += box.size;
    }
    return { checked: false, hasAudio: true };
  };

  const readEbmlVint = (view, offset, stripMarker) => {
    if (!view || offset >= view.byteLength) return null;
    const first = view.getUint8(offset);
    let mask = 0x80;
    let length = 1;
    while (length <= 8 && !(first & mask)) {
      mask >>= 1;
      length += 1;
    }
    if (length > 8 || offset + length > view.byteLength) return null;
    let value = stripMarker ? first & (mask - 1) : first;
    for (let i = 1; i < length; i += 1) {
      value = value * 256 + view.getUint8(offset + i);
    }
    return { value, length };
  };

  const webmElementsContainAudio = (view, start, end, depth = 0) => {
    if (!view || depth > 6) return false;
    let offset = start;
    while (offset + 2 <= end && offset + 2 <= view.byteLength) {
      const id = readEbmlVint(view, offset, false);
      if (!id) return false;
      const size = readEbmlVint(view, offset + id.length, true);
      if (!size) return false;
      const payloadStart = offset + id.length + size.length;
      let payloadEnd = payloadStart + size.value;
      if (payloadEnd > end || payloadEnd > view.byteLength) {
        if (!WEBM_CONTAINER_IDS.has(id.value)) return false;
        payloadEnd = Math.min(end, view.byteLength);
      }
      if (id.value === 0x83 && size.value >= 1 && view.getUint8(payloadStart) === 2) return true;
      if (WEBM_CONTAINER_IDS.has(id.value) && webmElementsContainAudio(view, payloadStart, payloadEnd, depth + 1)) return true;
      offset = payloadEnd;
    }
    return false;
  };

  const inspectWebmAudioTrack = async (file) => {
    const buffer = await readSlice(file, 0, Math.min(file.size, MAX_CONTAINER_SCAN_BYTES));
    if (!buffer) return { checked: false, hasAudio: true };
    const view = new DataView(buffer);
    return { checked: true, hasAudio: webmElementsContainAudio(view, 0, view.byteLength) };
  };

  const inspectAudioTrack = async (file, extension) => {
    try {
      if (extension === 'mp4') return inspectMp4AudioTrack(file);
      if (extension === 'webm') return inspectWebmAudioTrack(file);
    } catch {}
    return { checked: false, hasAudio: true };
  };

  const friendlyTranscribeError = (message) => {
    const text = cleanText(message);
    if (/failed to parse audio file/i.test(text)) {
      return 'No readable audio track found. Upload a file that includes audio or export an audio-only file.';
    }
    return text || 'Transcription failed.';
  };

  const billableSeconds = (durationSeconds) => {
    const duration = Number(durationSeconds);
    if (!Number.isFinite(duration) || duration <= 0) return 0;
    return Math.max(Number(state.config.minBillableSeconds) || 15, Math.ceil(duration));
  };

  const estimatedCost = (durationSeconds) => {
    const cost = billableSeconds(durationSeconds) * Number(state.config.pricePerSecond || DEFAULT_CONFIG.pricePerSecond);
    return Number(cost.toFixed(6));
  };

  const acceptedFiles = () => state.files.filter((item) => item.status !== 'skipped');

  const completedFiles = () => state.files.filter((item) => item.status === 'complete');

  const partialFiles = () => state.files.filter((item) => item.status === 'partial');

  const estimatedTotal = () => acceptedFiles().reduce((sum, item) => sum + Number(item.estimatedCostUsd || 0), 0);

  const finalTotal = () => [...completedFiles(), ...partialFiles()]
    .reduce((sum, item) => sum + Number(item.costUsd || item.estimatedCostUsd || 0), 0);

  const countedForRunLimit = () => state.files.filter((item) => !['checking', 'skipped'].includes(item.status)).length;

  const authIsReady = () => {
    const authApi = window.ToolsAuth;
    if (!authApi || !authApi.getAuth || !authApi.authIsValid) return false;
    return authApi.authIsValid(authApi.getAuth());
  };

  const runConfigIsValid = () => {
    const numericValues = [
      state.config.pricePerSecond,
      state.config.minDurationSeconds,
      state.config.maxFilesPerRun,
      state.config.maxFileBytes,
      state.config.maxTotalCostUsd
    ].map(Number);
    return state.config.configured === true &&
      Boolean(cleanText(state.config.service)) &&
      Array.isArray(state.config.supportedFormats) &&
      state.config.supportedFormats.length > 0 &&
      numericValues.every((value) => Number.isFinite(value) && value > 0);
  };

  const setView = (view) => {
    const next = ['upload', 'processing', 'results'].includes(view) ? view : 'upload';
    state.view = next;
    if (shellEl) shellEl.dataset.transcribeViewState = next;
    if (uploadViewEl) uploadViewEl.hidden = next !== 'upload';
    if (processingViewEl) processingViewEl.hidden = next !== 'processing';
    if (resultsViewEl) resultsViewEl.hidden = next !== 'results';
  };

  const updateLayoutState = () => {
    const hasResults = state.files.some((item) =>
      item.transcript || ['complete', 'partial', 'failed'].includes(item.status));
    const hasFiles = state.files.length > 0;
    const nextState = state.view === 'processing' || state.busy
      ? 'working'
      : state.view === 'results' || hasResults
        ? 'results'
        : hasFiles
          ? 'ready'
          : 'empty';
    if (document.body) document.body.dataset.toolsState = nextState;
  };

  const setBusy = (busy) => {
    state.busy = Boolean(busy);
    if (fileEl) fileEl.disabled = state.busy || state.analyzing;
    if (dropzoneEl) dropzoneEl.dataset.disabled = state.busy || state.analyzing ? 'true' : 'false';
    if (resetBtn) resetBtn.disabled = state.busy || state.analyzing;
    if (newBtn) newBtn.disabled = state.busy || state.analyzing;
    if (cancelBtn) cancelBtn.disabled = !state.busy;
    if (startBtn) {
      startBtn.classList.toggle('is-busy', state.busy);
    }
    updateControls();
    updateLayoutState();
  };

  const updateProgress = ({ stateName = 'hidden', ratio = 0, label = '' } = {}) => {
    if (!progressWrapEl || !progressBarEl) return;
    progressWrapEl.dataset.state = stateName;
    if (stateName === 'hidden') {
      progressBarEl.value = 0;
      setText(progressLabelEl, 'Progress');
      return;
    }
    const safeRatio = Math.min(1, Math.max(0, Number(ratio) || 0));
    progressBarEl.value = safeRatio;
    setText(progressLabelEl, label || 'Progress');
  };

  const updateStats = () => {
    setText(serviceEl, state.config.service || 'Amazon Transcribe');
    setText(priceEl, `${formatUsd(state.config.pricePerMinute)} / min`);
    setText(minimumEl, `${Number(state.config.minDurationSeconds) || 15} sec`);
    setText(acceptedEl, String(acceptedFiles().length));
  };

  const updateAuthUi = () => {
    if (!window.ToolsAuth) {
      setAuthPill('warning', 'Loading auth');
      setStatus(authStatusEl, 'Loading tools account sign-in.', 'warning');
      if (signInBtn) signInBtn.disabled = true;
      updateControls();
      return;
    }

    const authed = authIsReady();
    if (authed) {
      const user = window.ToolsAuth.getUser ? window.ToolsAuth.getUser(window.ToolsAuth.getAuth()) : {};
      const label = cleanText(user.email || user.name) || 'Signed in';
      setAuthPill('ok', 'Signed in');
      setStatus(authStatusEl, `${label} can start transcription jobs.`, 'success');
      if (signInBtn) signInBtn.disabled = true;
    } else {
      setAuthPill('err', 'Sign in required');
      setStatus(authStatusEl, 'Sign in before uploading files or starting paid transcription jobs.', 'warning');
      if (signInBtn) signInBtn.disabled = false;
    }
    updateControls();
  };

  const updateSummary = () => {
    const totalCount = state.files.length;
    const readyCount = acceptedFiles().length;
    const skippedCount = state.files.filter((item) => item.status === 'skipped').length;
    const completedCount = completedFiles().length;
    const partialCount = partialFiles().length;
    const failedCount = state.files.filter((item) => item.status === 'failed').length;
    const totalCost = estimatedTotal();
    const runCost = finalTotal();

    if (!totalCount) {
      setStatus(summaryEl, 'No files selected.', '');
      setText(totalEl, 'Estimated total: $0.00');
      return;
    }

    const parts = [`${readyCount} ready`, `${skippedCount} skipped`];
    if (completedCount || partialCount || failedCount) {
      parts.push(`${completedCount} complete`, `${partialCount} partial`, `${failedCount} failed`);
    }
    setStatus(
      summaryEl,
      `${parts.join(' · ')}. Estimated total ${formatUsd(totalCost)}.`,
      skippedCount || partialCount || failedCount ? 'warning' : 'success'
    );
    setText(
      totalEl,
      completedCount || partialCount
        ? `Final run total: ${formatUsd(runCost)} · Estimated total: ${formatUsd(totalCost)}`
        : `Estimated total: ${formatUsd(totalCost)}`
    );
  };

  const statusLabel = (item) => {
    if (item.status === 'checking') return 'Checking';
    if (item.status === 'skipped') return `Skipped: ${item.skipReason || 'Not eligible'}`;
    if (item.status === 'ready') return 'Ready';
    if (item.status === 'presigning') return 'Preparing upload';
    if (item.status === 'uploading') return `Uploading ${Math.round((Number(item.progress) || 0) * 100)}%`;
    if (item.status === 'starting') return 'Starting job';
    if (item.status === 'transcribing') return 'Transcribing';
    if (item.status === 'complete') return 'Complete';
    if (item.status === 'partial') {
      const endedAt = Number(item.transcriptEndSeconds);
      return endedAt > 0 ? `Partial: ended near ${formatClock(endedAt)}` : 'Partial transcript';
    }
    if (item.status === 'failed') return `Failed: ${item.error || 'Transcription failed'}`;
    if (item.status === 'canceled') return 'Canceled';
    return cleanText(item.status) || 'Pending';
  };

  const rowTone = (item) => {
    if (item.status === 'skipped') return 'warning';
    if (item.status === 'partial') return 'warning';
    if (item.status === 'failed') return 'error';
    if (item.status === 'complete') return 'success';
    if (item.status === 'uploading' || item.status === 'transcribing') return 'active';
    return '';
  };

  const canRemoveItem = (item) => !state.busy &&
    !state.analyzing &&
    !['presigning', 'uploading', 'starting', 'transcribing'].includes(String(item?.status || ''));

  const canResumeItem = (item) => !state.busy &&
    !state.analyzing &&
    ['failed', 'canceled'].includes(item?.status) &&
    item?.runErrorType !== 'service' &&
    (Boolean(item.runToken) || (Boolean(item.quoteToken) && item.uploadComplete === true));

  const renderTable = () => {
    if (tableWrapEl) tableWrapEl.hidden = state.files.length === 0;
    fileRowsEl.innerHTML = state.files.map((item) => {
      const removable = canRemoveItem(item);
      const resumable = canResumeItem(item);
      return `
      <article class="transcribe-file-card" data-tone="${escapeHtml(rowTone(item))}">
        <div class="transcribe-file-main">
          <span class="transcribe-file-name">${escapeHtml(item.name)}</span>
          <span class="transcribe-file-meta">${escapeHtml(formatBytes(item.bytes))}${item.extension ? ` · ${escapeHtml(item.extension.toUpperCase())}` : ''}</span>
        </div>
        <div class="transcribe-file-facts" aria-label="File details">
          <span>${item.durationSeconds ? escapeHtml(formatClock(item.durationSeconds)) : '--'}</span>
          <span>${item.billableSeconds ? `${escapeHtml(String(item.billableSeconds))} sec` : '--'}</span>
          <span>${escapeHtml(formatUsd(item.estimatedCostUsd || 0))}</span>
        </div>
        <span class="transcribe-file-status">${escapeHtml(statusLabel(item))}</span>
        <div class="transcribe-file-actions">
          ${resumable ? `
            <button
              type="button"
              class="transcribe-file-resume"
              data-transcribe-file-resume
              data-id="${escapeHtml(item.id)}"
            >Resume</button>
          ` : ''}
          <button
            type="button"
            class="transcribe-file-remove"
            data-transcribe-file-remove
            data-id="${escapeHtml(item.id)}"
            aria-label="Remove ${escapeHtml(item.name)} from queue"
            ${removable ? '' : 'disabled'}
          >X</button>
        </div>
      </article>
    `;
    }).join('');
    updateStats();
    updateSummary();
    updateControls();
    updateLayoutState();
    renderProcessingList();
  };

  const renderProcessingList = () => {
    if (!processingRowsEl) return;
    const processItems = acceptedFiles().filter((item) => item.status !== 'skipped');
    if (!processItems.length) {
      processingRowsEl.innerHTML = '<p class="transcribe-empty">Waiting for eligible files.</p>';
      return;
    }
    processingRowsEl.innerHTML = processItems.map((item) => `
      <article class="transcribe-process-card" data-tone="${escapeHtml(rowTone(item))}">
        <div>
          <span class="transcribe-file-name">${escapeHtml(item.name)}</span>
          <span class="transcribe-file-meta">${escapeHtml(formatClock(item.durationSeconds))} · ${escapeHtml(formatUsd(item.estimatedCostUsd || item.costUsd || 0))}</span>
        </div>
        <span class="transcribe-file-status">${escapeHtml(statusLabel(item))}</span>
      </article>
    `).join('');
  };

  const renderResults = () => {
    if (!resultsEl) return;
    const resultItems = state.files.filter((item) =>
      item.transcript || ['complete', 'partial', 'failed'].includes(item.status));
    const completedCount = completedFiles().length;
    const partialCount = partialFiles().length;
    const failedCount = state.files.filter((item) => item.status === 'failed').length;
    const skippedCount = state.files.filter((item) => item.status === 'skipped').length;
    if (resultsSummaryEl) {
      setText(
        resultsSummaryEl,
        resultItems.length
          ? `${completedCount} complete · ${partialCount} partial · ${failedCount} failed · ${skippedCount} skipped · Final cost ${formatUsd(finalTotal())}`
          : 'Completed transcripts will appear below.'
      );
    }
    if (!resultItems.length) {
      resultsEl.innerHTML = '<p class="transcribe-empty">Completed transcripts will appear here.</p>';
      return;
    }
    resultsEl.innerHTML = resultItems.map((item) => {
      const transcript = String(item.transcript || '').trim();
      const isComplete = item.status === 'complete';
      const isPartial = item.status === 'partial';
      const resumable = canResumeItem(item);
      const status = isComplete
        ? `Cost: ${formatUsd(item.costUsd || item.estimatedCostUsd || 0)} · ${item.billableSeconds || 0} billable sec`
        : isPartial
          ? `${item.error || 'The transcript may have ended before the source media.'} Cost: ${formatUsd(item.costUsd || item.estimatedCostUsd || 0)}.`
          : item.error || 'Transcription failed.';
      return `
        <article class="transcribe-result" data-status="${escapeHtml(item.status)}" data-id="${escapeHtml(item.id)}">
          <div class="transcribe-result-header">
            <div>
              <h3>${escapeHtml(item.name)}</h3>
              <p>${escapeHtml(status)}</p>
            </div>
            ${transcript || resumable ? `
              <div class="transcribe-result-actions">
                ${resumable ? `<button type="button" class="btn-secondary" data-transcribe-action="resume" data-id="${escapeHtml(item.id)}">Resume</button>` : ''}
                ${transcript ? `<button type="button" class="btn-secondary" data-transcribe-action="copy" data-id="${escapeHtml(item.id)}">Copy</button>` : ''}
                ${transcript ? `<button type="button" class="btn-secondary" data-transcribe-action="download" data-id="${escapeHtml(item.id)}">Download</button>` : ''}
              </div>
            ` : ''}
          </div>
          ${transcript
            ? `<textarea readonly>${escapeHtml(transcript)}</textarea>`
            : '<p class="transcribe-result-error">No transcript was produced for this file.</p>'}
        </article>
      `;
    }).join('');
  };

  const updateControls = () => {
    const readyCount = acceptedFiles().filter((item) => item.status === 'ready').length;
    const approved = Boolean(approveEl && approveEl.checked);
    if (approveEl) approveEl.disabled = state.busy || state.analyzing || readyCount === 0;
    if (startBtn) {
      const ready = authIsReady() && runConfigIsValid() && approved && readyCount > 0;
      startBtn.disabled = state.busy || state.analyzing || !ready;
      startBtn.dataset.ready = ready ? 'true' : 'false';
    }
  };

  const readJson = async (res) => {
    let data = null;
    try {
      data = await res.json();
    } catch {
      data = null;
    }
    if (!res.ok || data?.ok === false) {
      const err = new Error(data?.error || data?.message || `Request failed (${res.status}).`);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  };

  const authFetchJson = async (url, options = {}) => {
    if (!window.ToolsAuth || !window.ToolsAuth.fetchWithAuth) {
      throw new Error('Sign in before starting transcription jobs.');
    }
    const res = await window.ToolsAuth.fetchWithAuth(url, options);
    return readJson(res);
  };

  const loadConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/config`, { method: 'GET' });
      const data = await readJson(res);
      state.config = { ...DEFAULT_CONFIG, ...data };
      if (state.config.configured !== true) {
        throw new Error('Transcribe is not fully configured on the server.');
      }
      updateStats();
      updateSummary();
    } catch (err) {
      setStatus(runStatusEl, err?.message || 'Transcribe configuration is unavailable.', 'warning');
      state.config = { ...DEFAULT_CONFIG, configured: false };
      updateStats();
      updateControls();
    }
  };

  const probeDuration = (file, extension) => new Promise((resolve) => {
    if (!file || !window.URL || typeof window.URL.createObjectURL !== 'function') {
      resolve(null);
      return;
    }

    const src = window.URL.createObjectURL(file);
    const tag = VIDEO_FORMATS.has(extension) || String(file.type || '').toLowerCase().startsWith('video/')
      ? 'video'
      : 'audio';
    const el = document.createElement(tag);
    let done = false;
    let timeoutId = null;

    const finish = (value) => {
      if (done) return;
      done = true;
      if (timeoutId) window.clearTimeout(timeoutId);
      try {
        el.removeAttribute('src');
        el.load();
      } catch {}
      try {
        window.URL.revokeObjectURL(src);
      } catch {}
      const duration = Number(value);
      resolve(Number.isFinite(duration) && duration > 0 ? duration : null);
    };

    timeoutId = window.setTimeout(() => finish(null), DURATION_TIMEOUT_MS);
    el.preload = 'metadata';
    el.muted = true;
    el.playsInline = true;
    el.onloadedmetadata = () => finish(el.duration);
    el.onerror = () => finish(null);
    el.src = src;
  });

  const analyzeSelectedFiles = async (selectedFiles) => {
    const files = Array.from(selectedFiles || fileEl.files || []);
    if (fileEl) fileEl.value = '';
    if (!files.length) {
      setStatus(runStatusEl, state.files.length ? 'No new files selected.' : '', '');
      updateControls();
      return;
    }

    const existingFingerprints = new Set(state.files.map((item) => item.fingerprint).filter(Boolean));
    const newItems = files.map((file, index) => {
      const item = createFileItem(file, state.files.length + index);
      if (existingFingerprints.has(item.fingerprint)) {
        item.status = 'skipped';
        item.skipReason = 'Already added.';
      } else {
        existingFingerprints.add(item.fingerprint);
      }
      return item;
    });

    setView('upload');
    state.analyzing = true;
    setBusy(false);
    updateControls();
    updateLayoutState();
    state.files = [...state.files, ...newItems];
    if (approveEl) approveEl.checked = false;
    renderTable();
    renderResults();
    setStatus(runStatusEl, `Checking ${newItems.length} added file${newItems.length === 1 ? '' : 's'}...`, '');

    const formats = supportedFormats();
    let acceptedCost = estimatedTotal();
    let acceptedCount = state.files.filter((item) => item.status === 'ready').length;
    let addedAcceptedCount = 0;

    for (let i = 0; i < newItems.length; i += 1) {
      const item = newItems[i];
      if (item.status === 'skipped') {
        renderTable();
        continue;
      }
      if (countedForRunLimit() >= Number(state.config.maxFilesPerRun || DEFAULT_CONFIG.maxFilesPerRun)) {
        item.status = 'skipped';
        item.skipReason = `Run limit is ${state.config.maxFilesPerRun} files.`;
        renderTable();
        continue;
      }
      if (!formats.has(item.extension)) {
        item.status = 'skipped';
        item.skipReason = 'Unsupported file type.';
        renderTable();
        continue;
      }
      if (!item.bytes) {
        item.status = 'skipped';
        item.skipReason = 'Empty file.';
        renderTable();
        continue;
      }
      if (item.bytes > Number(state.config.maxFileBytes || DEFAULT_CONFIG.maxFileBytes)) {
        item.status = 'skipped';
        item.skipReason = `File exceeds ${formatBytes(state.config.maxFileBytes)}.`;
        renderTable();
        continue;
      }

      item.durationSeconds = await probeDuration(item.file, item.extension);
      if (!Number.isFinite(Number(item.durationSeconds))) {
        item.status = 'skipped';
        item.skipReason = 'Unable to read duration before upload.';
        renderTable();
        continue;
      }
      if (item.durationSeconds < Number(state.config.minDurationSeconds || 15)) {
        item.status = 'skipped';
        item.skipReason = `Under ${state.config.minDurationSeconds || 15} seconds.`;
        renderTable();
        continue;
      }
      if (VIDEO_FORMATS.has(item.extension) || String(item.contentType || '').toLowerCase().startsWith('video/')) {
        const audioTrack = await inspectAudioTrack(item.file, item.extension);
        if (audioTrack.checked && !audioTrack.hasAudio) {
          item.status = 'skipped';
          item.skipReason = 'No audio track found.';
          renderTable();
          continue;
        }
        if (audioTrack.malformedTimeline) {
          item.status = 'skipped';
          item.skipReason = 'Malformed MP4 timing detected. Repair/remux the file or export audio-only before transcribing.';
          renderTable();
          continue;
        }
      }

      item.billableSeconds = billableSeconds(item.durationSeconds);
      item.estimatedCostUsd = estimatedCost(item.durationSeconds);
      if (acceptedCost + item.estimatedCostUsd > Number(state.config.maxTotalCostUsd || DEFAULT_CONFIG.maxTotalCostUsd)) {
        item.status = 'skipped';
        item.skipReason = `Total estimate cap is ${formatUsd(state.config.maxTotalCostUsd)}.`;
        renderTable();
        continue;
      }

      item.status = 'ready';
      acceptedCost += item.estimatedCostUsd;
      acceptedCount += 1;
      addedAcceptedCount += 1;
      renderTable();
    }

    state.analyzing = false;
    setBusy(false);
    setStatus(
      runStatusEl,
      addedAcceptedCount
        ? `Added ${addedAcceptedCount} file${addedAcceptedCount === 1 ? '' : 's'}. ${acceptedCount} ready total.`
        : 'No newly selected files are eligible for transcription.',
      addedAcceptedCount ? 'success' : 'warning'
    );
    renderTable();
    markSessionDirty();
  };

  const abortActive = () => {
    try {
      if (state.activeXhr && state.activeXhr.readyState !== 4) state.activeXhr.abort();
    } catch {}
    try {
      if (state.activeController) state.activeController.abort();
    } catch {}
    state.activeXhr = null;
    state.activeController = null;
  };

  const uploadFile = (item, presign = {}) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.activeXhr = xhr;
    const method = String(presign.method || 'PUT').toUpperCase();
    xhr.open(method, presign.uploadUrl, true);
    Object.entries(presign.headers || {}).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      try {
        xhr.setRequestHeader(key, String(value));
      } catch {}
    });
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      item.progress = event.total > 0 ? event.loaded / event.total : 0;
      renderTable();
      renderProcessingList();
      updateProgress({
        stateName: 'visible',
        ratio: item.progress,
        label: `Uploading ${item.name} (${Math.round(item.progress * 100)}%)`
      });
    };
    xhr.onload = () => {
      state.activeXhr = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
        return;
      }
      const error = new Error(`Upload failed (${xhr.status}).`);
      error.status = xhr.status;
      reject(error);
    };
    xhr.onerror = () => {
      state.activeXhr = null;
      reject(new Error('Upload failed.'));
    };
    xhr.onabort = () => {
      state.activeXhr = null;
      reject(new Error('Upload canceled.'));
    };
    if (method === 'POST' && presign.fields && typeof presign.fields === 'object') {
      const body = new FormData();
      Object.entries(presign.fields).forEach(([key, value]) => {
        if (value === undefined || value === null) return;
        body.append(key, String(value));
      });
      body.append('file', item.file, item.name || 'media');
      xhr.send(body);
      return;
    }
    xhr.send(item.file);
  });

  const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const isTransientRunError = (err) => {
    const status = Number(err?.status);
    if (status === 429 || status >= 500) return true;
    const message = String(err?.message || '').toLowerCase();
    return !status && /fetch|network|offline|connection|timeout|temporar/.test(message);
  };

  const pollIntervalFor = (startedAt) => {
    const elapsed = Math.max(0, Date.now() - Number(startedAt || Date.now()));
    if (elapsed < 60_000) return POLL_INTERVAL_MS;
    if (elapsed < 10 * 60_000) return POLL_MEDIUM_INTERVAL_MS;
    return POLL_MAX_INTERVAL_MS;
  };

  const pollRun = async (item, runToken) => {
    const pollStartedAt = Number(item.pollStartedAt) || Date.now();
    item.pollStartedAt = pollStartedAt;
    let transientFailures = 0;
    while (!state.canceled) {
      const controller = new AbortController();
      state.activeController = controller;
      let data;
      try {
        data = await authFetchJson(`${API_BASE}/status?run=${encodeURIComponent(runToken)}`, {
          method: 'GET',
          signal: controller.signal
        });
        transientFailures = 0;
      } catch (err) {
        if (!state.canceled && isTransientRunError(err) && transientFailures < MAX_TRANSIENT_RETRIES) {
          transientFailures += 1;
          const retryDelay = Math.min(POLL_MAX_INTERVAL_MS, 1000 * (2 ** transientFailures));
          item.status = 'transcribing';
          item.error = `Connection interrupted. Retrying (${transientFailures}/${MAX_TRANSIENT_RETRIES})...`;
          renderTable();
          renderProcessingList();
          updateProgress({ stateName: 'visible', ratio: 1, label: `Reconnecting to ${item.name}...` });
          await sleep(retryDelay);
          continue;
        }
        throw err;
      } finally {
        if (state.activeController === controller) state.activeController = null;
      }

      const status = String(data.status || '').toUpperCase();
      if (status === 'COMPLETED') {
        item.transcript = String(data.transcript || '').trim();
        item.costUsd = Number(data.costUsd || item.estimatedCostUsd || 0);
        item.billableSeconds = Number(data.billableSeconds || item.billableSeconds || 0);
        item.durationSeconds = Number(data.durationSeconds || item.durationSeconds || 0);
        item.transcriptEndSeconds = Math.max(0, Number(data.transcriptEndSeconds) || 0);
        item.transcriptGapSeconds = Math.max(0, Number(data.transcriptGapSeconds) || 0);
        if (String(data.coverageStatus || '').toUpperCase() === 'SUSPECTED_EARLY_END') {
          item.status = 'partial';
          item.runErrorType = 'service';
          item.error = friendlyTranscribeError(data.warning ||
            `Amazon Transcribe stopped near ${formatClock(item.transcriptEndSeconds)} of ${formatClock(item.durationSeconds)}. The transcript may be partial; remux the media or export audio-only, then retry.`);
        } else {
          item.status = 'complete';
          item.error = '';
          item.runErrorType = '';
        }
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        persistActiveRunRecovery();
        return;
      }
      if (status === 'FAILED') {
        item.status = 'failed';
        item.error = friendlyTranscribeError(data.error || 'Transcription failed.');
        item.runErrorType = 'service';
        item.costUsd = Number(data.costUsd ?? item.estimatedCostUsd ?? 0);
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        persistActiveRunRecovery();
        return;
      }

      item.status = 'transcribing';
      item.error = '';
      renderTable();
      renderProcessingList();
      updateProgress({ stateName: 'visible', ratio: 1, label: `Transcribing ${item.name}...` });
      await sleep(pollIntervalFor(pollStartedAt));
    }
    throw new Error('Canceled.');
  };

  const startReservedRun = async (item) => {
    let start;
    for (let attempt = 0; attempt <= 3; attempt += 1) {
      const startController = new AbortController();
      state.activeController = startController;
      try {
        start = await authFetchJson(`${API_BASE}/start`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ quoteToken: item.quoteToken }),
          signal: startController.signal
        });
        break;
      } catch (err) {
        if (state.canceled || !isTransientRunError(err) || attempt >= 3) throw err;
        await sleep(Math.min(8000, 1000 * (2 ** attempt)));
      } finally {
        if (state.activeController === startController) state.activeController = null;
      }
    }
    const runToken = String(start?.runToken || '');
    if (!runToken) throw new Error('The transcription job did not return a recovery token.');
    item.runToken = runToken;
    item.pollStartedAt = Number(item.pollStartedAt) || Date.now();
    persistActiveRunRecovery();
    return runToken;
  };

  const runFile = async (item) => {
    item.status = 'presigning';
    item.error = '';
    item.runErrorType = '';
    item.progress = 0;
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 0, label: `Preparing ${item.name}...` });

    const controller = new AbortController();
    state.activeController = controller;
    let presign;
    try {
      presign = await authFetchJson(`${API_BASE}/presign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: item.name,
          contentType: item.contentType,
          bytes: item.bytes,
          durationSeconds: item.durationSeconds
        }),
        signal: controller.signal
      });
    } finally {
      if (state.activeController === controller) state.activeController = null;
    }

    if (state.canceled) throw new Error('Canceled.');
    item.quoteToken = String(presign.quoteToken || '');

    item.status = 'uploading';
    renderTable();
    await uploadFile(item, presign);
    item.uploadComplete = true;
    persistActiveRunRecovery();
    if (state.canceled) throw new Error('Canceled.');

    item.status = 'starting';
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 1, label: `Starting ${item.name}...` });

    item.status = 'transcribing';
    const runToken = await startReservedRun(item);
    renderTable();
    await pollRun(item, runToken);
  };

  const resumeItem = async (item) => {
    if (!canResumeItem(item)) return;
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in again before resuming this transcription.', 'warning');
      updateAuthUi();
      return;
    }
    if (!runConfigIsValid()) {
      setStatus(runStatusEl, 'Transcription configuration is unavailable. Refresh and try again.', 'warning');
      return;
    }

    state.canceled = false;
    item.error = '';
    item.runErrorType = '';
    item.status = item.runToken ? 'transcribing' : 'starting';
    setView('processing');
    setBusy(true);
    renderTable();
    renderProcessingList();
    setStatus(runStatusEl, `Resuming ${item.name} with its existing job token...`, '');
    updateProgress({ stateName: 'visible', ratio: 1, label: `Resuming ${item.name}...` });

    try {
      const runToken = item.runToken || await startReservedRun(item);
      item.status = 'transcribing';
      renderTable();
      await pollRun(item, runToken);
      setStatus(
        runStatusEl,
        item.status === 'complete'
          ? `${item.name} is complete.`
          : item.status === 'partial'
            ? `${item.name} produced a partial transcript. Repair the media and submit it as a new file.`
            : `${item.name} could not be transcribed.`,
        item.status === 'complete' ? 'success' : 'warning'
      );
    } catch (err) {
      if (state.canceled || err?.name === 'AbortError' || err?.message === 'Canceled.') {
        item.status = 'canceled';
        item.error = 'Canceled.';
      } else {
        item.status = 'failed';
        item.error = friendlyTranscribeError(err?.message || 'Unable to resume transcription.');
        item.runErrorType = item.runToken && isTransientRunError(err) ? 'network' : classifyRunError(err);
      }
      setStatus(runStatusEl, item.error, 'warning');
    } finally {
      setBusy(false);
      updateProgress({ stateName: 'hidden' });
      renderTable();
      renderProcessingList();
      renderResults();
      setView('results');
      updateLayoutState();
      markSessionDirty();
      persistActiveRunRecovery();
    }
  };

  const runBatch = async ({ reportOutcome = false } = {}) => {
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in before starting transcription jobs.', 'warning');
      updateAuthUi();
      if (reportOutcome) reportRunError('permission');
      return;
    }

    if (!runConfigIsValid()) {
      setStatus(runStatusEl, 'Transcription configuration is unavailable. Refresh and try again.', 'warning');
      if (reportOutcome) reportRunError('validation');
      return;
    }

    const queue = acceptedFiles().filter((item) => item.status === 'ready');
    if (!queue.length) {
      setStatus(runStatusEl, 'No eligible files to transcribe.', 'warning');
      if (reportOutcome) reportRunError('validation');
      return;
    }
    if (!approveEl || !approveEl.checked) {
      setStatus(runStatusEl, 'Review and approve the estimated charge before starting.', 'warning');
      if (reportOutcome) reportRunError('validation');
      return;
    }

    state.canceled = false;
    setView('processing');
    setBusy(true);
    renderProcessingList();
    setStatus(runStatusEl, `Starting ${queue.length} transcription job${queue.length === 1 ? '' : 's'}...`, '');
    setText(processingCopyEl, `Processing ${queue.length} file${queue.length === 1 ? '' : 's'}. You can leave this tab open while Amazon Transcribe runs.`);
    updateProgress({ stateName: 'visible', ratio: 0, label: 'Starting batch...' });

    for (let i = 0; i < queue.length; i += 1) {
      const item = queue[i];
      if (state.canceled) break;
      try {
        setStatus(runStatusEl, `Processing ${i + 1} of ${queue.length}: ${item.name}`, '');
        setText(processingCopyEl, `Processing ${i + 1} of ${queue.length}: ${item.name}`);
        await runFile(item);
        updateProgress({
          stateName: 'visible',
          ratio: (i + 1) / queue.length,
          label: `${i + 1} of ${queue.length} files finished`
        });
      } catch (err) {
        if (state.canceled || err?.name === 'AbortError' || err?.message === 'Canceled.') {
          item.status = 'canceled';
          item.error = 'Canceled.';
          item.runErrorType = 'processing';
        } else {
          item.status = 'failed';
          item.error = friendlyTranscribeError(err?.message || 'Transcription failed.');
          item.runErrorType = item.runToken && isTransientRunError(err) ? 'network' : classifyRunError(err);
        }
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        persistActiveRunRecovery();
      }
    }

    const completed = queue.filter((item) => item.status === 'complete').length;
    const partial = queue.filter((item) => item.status === 'partial').length;
    const failed = queue.filter((item) => item.status === 'failed').length;
    const canceled = queue.filter((item) => item.status === 'canceled').length;
    const message = state.canceled
      ? `Canceled. ${completed} complete, ${failed} failed, ${canceled} canceled.`
      : `Done. ${completed} complete, ${partial} partial, ${failed} failed.`;
    setStatus(runStatusEl, message, partial || failed || canceled ? 'warning' : 'success');
    setBusy(false);
    updateProgress({ stateName: 'hidden' });
    renderTable();
    renderResults();
    setView('results');
    updateLayoutState();
    persistActiveRunRecovery();

    if (reportOutcome) {
      if (state.canceled || canceled) {
        reportRunCancel();
      } else if (completed > 0 || partial > 0) {
        reportRunComplete(partial || failed ? 'partial_success' : 'all_complete');
      } else {
        const errorTypes = queue.map((item) => item.runErrorType).filter(Boolean);
        const errorType = ['permission', 'network', 'timeout', 'service', 'processing']
          .find((candidate) => errorTypes.includes(candidate)) || 'service';
        reportRunError(errorType);
      }
    }
  };

  const reset = () => {
    state.canceled = true;
    state.analyzing = false;
    abortActive();
    state.files = [];
    clearActiveRunRecovery();
    if (fileEl) fileEl.value = '';
    if (approveEl) approveEl.checked = false;
    setView('upload');
    setBusy(false);
    setStatus(runStatusEl, '', '');
    updateProgress({ stateName: 'hidden' });
    renderTable();
    renderResults();
    markSessionDirty();
  };

  const signInReturnTo = () => `${window.location.pathname}${window.location.search}${window.location.hash}`;

  const signInWithPopup = () => {
    if (!window.ToolsAuth || !window.ToolsAuth.signIn) return;
    setStatus(runStatusEl, 'Opening sign-in in a new window. Your selected files will stay here.', '');
    window.ToolsAuth.signIn({ mode: 'popup', returnTo: signInReturnTo() })
      .then((result) => {
        if (result?.mode === 'popup') {
          setStatus(runStatusEl, 'Finish sign-in in the new window, then return here to start transcription.', '');
        }
      })
      .catch((err) => {
        setStatus(runStatusEl, err?.message || 'Unable to open sign-in.', 'warning');
      });
  };

  if (signInBtn) {
    signInBtn.addEventListener('click', () => {
      signInWithPopup();
    });
  }

  fileEl.addEventListener('change', () => {
    analyzeSelectedFiles(fileEl.files);
  });

  if (dropzoneEl) {
    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzoneEl.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (state.busy || state.analyzing) return;
        dropzoneEl.dataset.dragging = 'true';
      });
    });
    ['dragleave', 'drop'].forEach((eventName) => {
      dropzoneEl.addEventListener(eventName, (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (eventName === 'dragleave' && dropzoneEl.contains(event.relatedTarget)) return;
        dropzoneEl.dataset.dragging = 'false';
      });
    });
    dropzoneEl.addEventListener('drop', (event) => {
      if (state.busy || state.analyzing) return;
      const droppedFiles = event.dataTransfer?.files;
      if (!droppedFiles || !droppedFiles.length) return;
      if (fileEl) fileEl.value = '';
      analyzeSelectedFiles(droppedFiles);
    });
  }

  if (approveEl) {
    approveEl.addEventListener('change', updateControls);
  }

  if (fileRowsEl) {
    fileRowsEl.addEventListener('click', (event) => {
      const resumeBtn = event.target.closest('[data-transcribe-file-resume]');
      if (resumeBtn && !resumeBtn.disabled) {
        const id = resumeBtn.getAttribute('data-id');
        const item = state.files.find((entry) => entry.id === id);
        if (item && canResumeItem(item)) void resumeItem(item);
        return;
      }
      const removeBtn = event.target.closest('[data-transcribe-file-remove]');
      if (!removeBtn || removeBtn.disabled) return;
      const id = removeBtn.getAttribute('data-id');
      const item = state.files.find((entry) => entry.id === id);
      if (!item || !canRemoveItem(item)) return;
      state.files = state.files.filter((entry) => entry.id !== id);
      persistActiveRunRecovery();
      if (approveEl) approveEl.checked = false;
      setStatus(runStatusEl, `Removed ${item.name} from the queue. Review the updated estimate before starting.`, 'warning');
      renderTable();
      renderResults();
      markSessionDirty();
    });
  }

  formEl.addEventListener('submit', (event) => {
    event.preventDefault();
    void runBatch({ reportOutcome: true }).catch((error) => {
      reportRunError(classifyRunError(error));
    });
  });

  if (cancelBtn) {
    cancelBtn.addEventListener('click', () => {
      state.canceled = true;
      abortActive();
      setStatus(runStatusEl, 'Canceling. Already submitted AWS jobs may still incur cost.', 'warning');
      setBusy(false);
      updateProgress({ stateName: 'hidden' });
      persistActiveRunRecovery();
    });
  }

  if (resetBtn) resetBtn.addEventListener('click', reset);
  if (newBtn) newBtn.addEventListener('click', reset);

  if (resultsEl) {
    resultsEl.addEventListener('click', async (event) => {
      const actionBtn = event.target.closest('[data-transcribe-action]');
      if (!actionBtn) return;
      const action = actionBtn.getAttribute('data-transcribe-action');
      const id = actionBtn.getAttribute('data-id');
      const item = state.files.find((entry) => entry.id === id);
      if (!item) return;
      if (action === 'resume') {
        await resumeItem(item);
        return;
      }
      const transcript = String(item?.transcript || '').trim();
      if (!transcript) return;
      if (action === 'copy') {
        try {
          await navigator.clipboard.writeText(transcript);
          setStatus(runStatusEl, `Copied transcript for ${item.name}.`, 'success');
        } catch {
          setStatus(runStatusEl, 'Copy failed. Select the text manually.', 'error');
        }
      }
      if (action === 'download') {
        const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = safeDownloadName(item.name);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    });
  }

  document.addEventListener('tools:auth-changed', () => {
    updateAuthUi();
    const restoredCount = restoreActiveRunRecovery();
    if (!restoredCount) return;
    setView('results');
    renderTable();
    renderResults();
    setStatus(runStatusEl, `Recovered ${restoredCount} unfinished transcription job${restoredCount === 1 ? '' : 's'}. Select Resume to continue without uploading again.`, 'warning');
  });
  window.addEventListener('pagehide', persistActiveRunRecovery);

  document.addEventListener('tools:session-capture', (event) => {
    const detail = event?.detail;
    if (detail?.toolId !== TOOL_ID) return;
    const payload = detail?.payload;
    if (!payload || typeof payload !== 'object') return;

    const accepted = acceptedFiles();
    const skipped = state.files.filter((item) => item.status === 'skipped');
    const completed = completedFiles();
    const partial = partialFiles();
    const failed = state.files.filter((item) => item.status === 'failed');
    payload.inputs = {
      Files: `${state.files.length} selected`,
      Accepted: String(accepted.length),
      Skipped: String(skipped.length),
      'Estimated total': formatUsd(estimatedTotal())
    };
    payload.outputSummary = completed.length || partial.length || failed.length
      ? `${completed.length} complete · ${partial.length} partial · ${failed.length} failed · ${formatUsd(finalTotal())} run cost`
      : 'No transcripts saved in session history.';
  });

  loadConfig().finally(() => {
    const restoredCount = restoreActiveRunRecovery();
    setView(restoredCount ? 'results' : 'upload');
    updateAuthUi();
    renderTable();
    renderResults();
    if (restoredCount) {
      setStatus(runStatusEl, `Recovered ${restoredCount} unfinished transcription job${restoredCount === 1 ? '' : 's'}. Select Resume to continue without uploading again.`, 'warning');
    }
  });
  window.setTimeout(updateAuthUi, 250);
  window.setTimeout(updateAuthUi, 1000);
})();
