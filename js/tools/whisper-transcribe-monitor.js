(() => {
  'use strict';

  const TOOL_ID = 'transcribe';
  const API_BASE = '/api/tools/transcribe';
  const DEFAULT_CONFIG = {
    configured: false,
    service: 'Amazon Transcribe',
    region: 'us-east-2',
    languageCode: 'en-US',
    pricePerSecond: 0.0001,
    pricePerMinute: 0.006,
    minDurationSeconds: 15,
    minBillableSeconds: 15,
    maxFilesPerRun: 10,
    maxFileBytes: 500 * 1024 * 1024,
    maxTotalCostUsd: 100,
    maxServiceDurationSeconds: 8 * 60 * 60,
    supportedFormats: ['amr', 'flac', 'm4a', 'mp3', 'mp4', 'ogg', 'wav', 'webm']
  };
  const PROVIDER_AWS = 'aws';
  const PROVIDER_LOCAL = 'local';
  const LOCAL_POLL_INTERVAL_MS = 2000;
  const LOCAL_UPLOAD_RETRIES = 2;
  const DEFAULT_LOCAL_CHUNK_BYTES = 8 * 1024 * 1024;

  /*
   * Admin Local GPU integration contract:
   * - GET /api/tools/transcribe/local-config
   *   { ok, enabled, configured, service, workerOrigin, chunkBytes,
   *     maxFilesPerRun, maxFileBytes, maxServiceDurationSeconds,
   *     minDurationSeconds, supportedFormats, historyStored:false }
   * - POST /api/tools/transcribe/local-ticket, once per file, with
   *   { filename, format, contentType, bytes, durationSeconds }
   *   => { ok, enabled, configured, workerOrigin, chunkBytes, ticket,
   *        expiresAt, job:{ id, filename, format, contentType, bytes,
   *        durationSeconds } }
   * - GET /v1/health is an unauthenticated readiness check. Worker job
   *   routes receive Authorization: Bearer <ticket>:
   *   POST /v1/jobs with
   *     { id, filename, format, contentType, bytes, durationSeconds,
   *       chunkBytes, chunkCount }
   *   PUT /v1/jobs/{id}/chunks/{index} with Content-Range and
   *     X-Chunk-SHA256
   *   POST /v1/jobs/{id}/complete with { chunkCount }
   *   GET /v1/jobs/{id}
   *   POST /v1/jobs/{id}/cancel
   *   POST /v1/jobs/{id}/ack
   * - Worker job reads return top-level
   *   { ok, id, status, stage, progress, filename, error, transcript,
   *     durationSeconds, coverageStatus }.
   */
  const VIDEO_FORMATS = new Set(['mp4', 'webm']);
  const POLL_INTERVAL_MS = 5000;
  const POLL_MEDIUM_INTERVAL_MS = 15000;
  const POLL_MAX_INTERVAL_MS = 30000;
  const MAX_TRANSIENT_RETRIES = 5;
  const ACTIVE_RUNS_STORAGE_KEY = 'tools:transcribe:active-runs:v1';
  const ACTIVE_RUNS_MAX_ITEMS = 10;
  const ACTIVE_RUNS_MAX_TOKEN_CHARS = 8192;
  const HISTORY_PAGE_SIZE = 20;
  const NOTIFICATION_PREFERENCE_KEY = 'tools:transcribe:notifications:v1';
  const DURATION_TIMEOUT_MS = 10000;
  const MAX_CONTAINER_SCAN_BYTES = 8 * 1024 * 1024;
  const MP4_SUSPICIOUS_STTS_DELTA_SECONDS = 120;
  const MP4_TIMELINE_OVERFLOW_SECONDS = 120;
  const MP4_CONTAINER_BOXES = new Set(['moov', 'trak', 'mdia', 'minf', 'dinf', 'stbl', 'edts', 'udta', 'meta']);
  const WEBM_CONTAINER_IDS = new Set([0x18538067, 0x1654AE6B, 0xAE]);

  const isMp4TimelineSuspicious = ({ maxDeltaSeconds, timelineSeconds, mediaDurationSeconds }) =>
    Number(maxDeltaSeconds) >= MP4_SUSPICIOUS_STTS_DELTA_SECONDS &&
    Number(timelineSeconds) - Number(mediaDurationSeconds) >= MP4_TIMELINE_OVERFLOW_SECONDS;

  const isRecoverableItem = (item) =>
    item?.provider !== PROVIDER_LOCAL &&
    ['failed', 'canceled'].includes(String(item?.status || '')) &&
    item?.runErrorType !== 'service' &&
    (Boolean(item?.runToken) || (Boolean(item?.quoteToken) && item?.uploadComplete === true));

  if (typeof module !== 'undefined' && module.exports && typeof document === 'undefined') {
    module.exports = { isMp4TimelineSuspicious, isRecoverableItem };
    return;
  }

  const $id = (id) => document.getElementById(id);

  const shellEl = $id('transcribe-shell');
  const uploadViewEl = $id('transcribe-upload-view');
  const processingViewEl = $id('transcribe-processing-view');
  const resultsViewEl = $id('transcribe-results-view');
  const dropzoneEl = $id('transcribe-dropzone');
  const addFilesBtn = $id('transcribe-add-files');
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
  const resumeAllBtn = $id('transcribe-resume-all');
  const runStatusEl = $id('transcribe-run-status');
  const processingCopyEl = $id('transcribe-processing-copy');
  const processingRowsEl = $id('transcribe-processing-rows');
  const resultsSummaryEl = $id('transcribe-results-summary');
  const progressWrapEl = $id('transcribe-progress-wrap');
  const progressLabelEl = $id('transcribe-progress-label');
  const progressBarEl = $id('transcribe-progress');
  const resultsEl = $id('transcribe-results');
  const usageEl = $id('transcribe-usage');
  const usageValueEl = $id('transcribe-usage-value');
  const usageProgressEl = $id('transcribe-usage-progress');
  const usageTooltipEl = $id('transcribe-usage-tooltip');
  const historyOpenBtn = $id('transcribe-history-open');
  const historyDialogEl = $id('transcribe-history-dialog');
  const historyCloseBtn = $id('transcribe-history-close');
  const historyRefreshBtn = $id('transcribe-history-refresh');
  const historyStatusEl = $id('transcribe-history-status');
  const historyListViewEl = $id('transcribe-history-list-view');
  const historyListEl = $id('transcribe-history-list');
  const historyLoadMoreBtn = $id('transcribe-history-load-more');
  const historyDetailEl = $id('transcribe-history-detail');
  const historyBackBtn = $id('transcribe-history-back');
  const historyDetailNameEl = $id('transcribe-history-detail-name');
  const historyDetailMetaEl = $id('transcribe-history-detail-meta');
  const historyTranscriptEl = $id('transcribe-history-transcript');
  const historyCopyBtn = $id('transcribe-history-copy');
  const historyDownloadBtn = $id('transcribe-history-download');
  const historyDeleteBtn = $id('transcribe-history-delete');
  const historyPrivacyEl = $id('transcribe-history-privacy');
  const notificationsBtn = $id('transcribe-notifications');
  const methodEl = $id('transcribe-method');
  const methodHelpEl = $id('transcribe-method-help');
  const methodInputs = Array.from(document.querySelectorAll('input[name="transcribe-provider"]'));
  const localStateEl = $id('transcribe-local-state');
  const localStateCopyEl = $id('transcribe-local-state-copy');
  const localRefreshBtn = $id('transcribe-local-refresh');
  const providerKickerEl = $id('transcribe-provider-kicker');
  const uploadCopyEl = $id('transcribe-upload-copy');
  const costReviewEl = $id('transcribe-cost-review');
  const approvalCopyEl = $id('transcribe-approval-copy');
  const processingDetailsEl = $id('transcribe-processing-details');
  const detailsSummaryEl = $id('transcribe-details-summary');
  const detailServiceEl = $id('transcribe-stat-service');
  const detailMinimumEl = $id('transcribe-stat-minimum');
  const detailPriceEl = $id('transcribe-stat-price');
  const providerPanels = Array.from(document.querySelectorAll('[data-transcribe-provider-panel]'));
  const uploadActionsEl = startBtn?.closest('.transcribe-actions') || null;

  if (!formEl || !fileEl || !startBtn || !fileRowsEl) return;

  const state = {
    config: { ...DEFAULT_CONFIG },
    provider: PROVIDER_AWS,
    localConfig: null,
    localConfigLoading: false,
    localStatus: 'checking',
    localStatusMessage: 'Checking Home GPU...',
    admin: false,
    files: [],
    busy: false,
    canceled: false,
    activeXhr: null,
    activeController: null,
    analyzing: false,
    view: 'upload',
    accountSub: '',
    usage: null,
    usageLoading: false,
    usageRequestId: 0,
    historyItems: [],
    historyNextCursor: '',
    historyLoading: false,
    historyDetail: null,
    historyRequestId: 0,
    notificationsEnabled: false
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

  const isAdminUser = () => {
    try {
      const authApi = window.ToolsAuth;
      const auth = authApi?.getAuth?.();
      return Boolean(authApi?.authIsValid?.(auth) && authApi?.isAdmin?.(auth));
    } catch {
      return false;
    }
  };

  const isLocalProvider = () => state.provider === PROVIDER_LOCAL;

  const activeConfig = () => isLocalProvider()
    ? (state.localConfig || {})
    : state.config;

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
        if (item?.provider === PROVIDER_LOCAL) return false;
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
          provider: PROVIDER_AWS,
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
        provider: PROVIDER_AWS,
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
    else {
      state.provider = PROVIDER_AWS;
      persistActiveRunRecovery();
    }
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

  const setLocalStatus = (status, message) => {
    state.localStatus = status || 'offline';
    state.localStatusMessage = message || '';
    if (localStateEl) localStateEl.dataset.state = state.localStatus;
    setText(localStateCopyEl, state.localStatusMessage);
  };

  const normalizeWorkerOrigin = (value) => {
    const parsed = new URL(String(value || ''));
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      throw new Error('The home PC worker address must use HTTP or HTTPS.');
    }
    return parsed.origin;
  };

  const isLoopbackWorkerOrigin = (value) => {
    try {
      const hostname = new URL(value).hostname.toLowerCase().replace(/^\[|\]$/g, '');
      return hostname === 'localhost' || hostname === '::1' || /^127(?:\.\d{1,3}){3}$/.test(hostname);
    } catch {
      return false;
    }
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

  const formatUsageUsd = (value) => new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(Math.max(0, Number(value) || 0));

  const historyDate = (value) => {
    if (typeof value === 'number' || /^\d+$/.test(String(value || ''))) {
      const numeric = Number(value);
      return new Date(numeric > 10_000_000_000 ? numeric : numeric * 1000);
    }
    return new Date(String(value || ''));
  };

  const formatHistoryDate = (value) => {
    const date = historyDate(value);
    if (!Number.isFinite(date.getTime())) return 'Date unavailable';
    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short'
    }).format(date);
  };

  const formatUtcReset = (value) => {
    const date = historyDate(value);
    if (!Number.isFinite(date.getTime())) return 'Resets at 00:00 UTC.';
    return `Resets ${new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      timeZone: 'UTC',
      timeZoneName: 'short'
    }).format(date)}.`;
  };

  const getExtension = (name) => {
    const match = String(name || '').toLowerCase().match(/\.([a-z0-9]+)$/);
    return match ? match[1] : '';
  };

  const safeDownloadName = (name) => {
    const base = String(name || 'transcript').replace(/\.[^.]+$/, '') || 'transcript';
    return `${base.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'transcript'}-transcript.txt`;
  };

  const downloadTranscript = (name, transcript) => {
    const blob = new Blob([String(transcript || '')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = safeDownloadName(name);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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
    provider: state.provider,
    progress: 0,
    status: 'checking',
    skipReason: '',
    transcript: ''
  });

  const supportedFormats = () => new Set(
    Array.isArray(activeConfig().supportedFormats)
      ? activeConfig().supportedFormats.map((item) => String(item).toLowerCase())
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

  const responseErrorMessage = (data, fallback = '') => {
    const error = data?.error;
    if (typeof error === 'string') return cleanText(error) || cleanText(fallback);
    if (error && typeof error === 'object') {
      return cleanText(error.message || error.detail || error.code) || cleanText(fallback);
    }
    return cleanText(data?.message || fallback);
  };

  const billableSeconds = (durationSeconds) => {
    const duration = Number(durationSeconds);
    if (!Number.isFinite(duration) || duration <= 0) return 0;
    if (isLocalProvider()) return Math.ceil(duration);
    return Math.max(Number(state.config.minBillableSeconds) || 15, Math.ceil(duration));
  };

  const estimatedCost = (durationSeconds) => {
    if (isLocalProvider()) return 0;
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
    if (isLocalProvider()) {
      const config = state.localConfig || {};
      const numericValues = [
        config.chunkBytes,
        config.maxFilesPerRun,
        config.maxFileBytes,
        config.maxServiceDurationSeconds,
        config.minDurationSeconds
      ].map(Number);
      return isAdminUser() &&
        config.enabled === true &&
        config.configured === true &&
        state.localStatus === 'online' &&
        Boolean(cleanText(config.workerOrigin)) &&
        Array.isArray(config.supportedFormats) &&
        config.supportedFormats.length > 0 &&
        numericValues.every((value) => Number.isFinite(value) && value > 0);
    }
    const numericValues = [
      state.config.pricePerSecond,
      state.config.minDurationSeconds,
      state.config.maxFilesPerRun,
      state.config.maxFileBytes,
      state.config.maxTotalCostUsd,
      state.config.maxServiceDurationSeconds
    ].map(Number);
    return state.config.configured === true &&
      Boolean(cleanText(state.config.service)) &&
      Array.isArray(state.config.supportedFormats) &&
      state.config.supportedFormats.length > 0 &&
      numericValues.every((value) => Number.isFinite(value) && value > 0);
  };

  const syncProviderUi = () => {
    const admin = state.admin && isAdminUser();
    const localActive = admin && isLocalProvider();
    const hasFiles = state.files.length > 0;

    if (shellEl) {
      shellEl.dataset.transcribeProvider = state.provider;
      shellEl.dataset.transcribeHasFiles = hasFiles ? 'true' : 'false';
    }
    if (dropzoneEl) dropzoneEl.dataset.compact = hasFiles ? 'true' : 'false';

    providerPanels.forEach((panel) => {
      const provider = cleanText(panel.dataset.transcribeProviderPanel).toLowerCase();
      const providerMatches = provider === 'all' || provider === state.provider;
      const requiresFiles = panel.hasAttribute('data-transcribe-requires-files');
      panel.hidden = !providerMatches || (requiresFiles && !hasFiles);
    });

    if (localStateEl) {
      localStateEl.hidden = !localActive;
      localStateEl.setAttribute('aria-live', localActive ? 'polite' : 'off');
      localStateEl.setAttribute('aria-atomic', 'true');
    }
    if (localRefreshBtn) {
      localRefreshBtn.hidden = !localActive || !['offline', 'unavailable'].includes(state.localStatus);
    }
    if (usageEl) usageEl.hidden = isLocalProvider();
    if (historyOpenBtn) {
      historyOpenBtn.hidden = isLocalProvider() || !authIsReady();
      historyOpenBtn.disabled = isLocalProvider() || !authIsReady();
    }
    if (notificationsBtn) notificationsBtn.hidden = false;
    if (isLocalProvider() && historyDialogEl?.open) closeHistory();

    setText(detailsSummaryEl, isLocalProvider() ? 'Home GPU' : 'AWS');
    const config = activeConfig();
    const minimumSeconds = Math.max(0, Number(config.minDurationSeconds) || DEFAULT_CONFIG.minDurationSeconds);
    const serviceName = cleanText(config.service) || (isLocalProvider() ? 'Home GPU' : 'Amazon Transcribe');
    setText(detailServiceEl, serviceName);
    setText(detailMinimumEl, minimumSeconds < 60 ? `${Math.round(minimumSeconds)} sec` : formatClock(minimumSeconds));
    setText(detailPriceEl, `$${Number(state.config.pricePerMinute || DEFAULT_CONFIG.pricePerMinute).toFixed(3)} / min`);
    if (summaryEl) summaryEl.hidden = !hasFiles;
    if (costReviewEl) {
      const reviewProvider = cleanText(costReviewEl.dataset.transcribeProviderPanel).toLowerCase();
      const reviewMatches = !reviewProvider || reviewProvider === 'all' || reviewProvider === state.provider;
      costReviewEl.hidden = !hasFiles || !reviewMatches;
    }
    if (uploadActionsEl) uploadActionsEl.hidden = !hasFiles;
  };

  const updateMethodUi = () => {
    const admin = state.admin && isAdminUser();
    const locked = state.busy || state.analyzing || state.files.length > 0;
    if (methodEl) methodEl.hidden = !admin;
    methodInputs.forEach((input) => {
      const isLocal = input.value === PROVIDER_LOCAL;
      const localUnavailable = isLocal && (
        state.localConfigLoading ||
        (state.localConfig !== null && (
          state.localConfig?.enabled !== true ||
          state.localConfig?.configured !== true
        ))
      );
      input.disabled = !admin || locked || localUnavailable;
      input.checked = input.value === state.provider;
    });
    if (localRefreshBtn) localRefreshBtn.disabled = !admin || state.localConfigLoading || state.busy;
    setText(methodHelpEl, locked ? 'Remove all files to change the processing method.' : '');

    if (isLocalProvider()) {
      setText(providerKickerEl, 'Home GPU');
      setText(uploadCopyEl, 'Add audio or video to process on your home PC.');
      if (costReviewEl) costReviewEl.dataset.provider = PROVIDER_LOCAL;
      setText(approvalCopyEl, 'Send these files securely to my home PC for temporary processing.');
    } else {
      setText(providerKickerEl, 'Amazon Transcribe');
      setText(uploadCopyEl, 'Add audio or video, then review the estimate.');
      if (costReviewEl) costReviewEl.dataset.provider = PROVIDER_AWS;
      setText(approvalCopyEl, 'Approve the estimated charge shown above.');
    }
    syncProviderUi();
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
    if (addFilesBtn) addFilesBtn.disabled = state.busy || state.analyzing;
    if (dropzoneEl) dropzoneEl.dataset.disabled = state.busy || state.analyzing ? 'true' : 'false';
    if (resetBtn) resetBtn.disabled = state.busy || state.analyzing;
    if (newBtn) newBtn.disabled = state.busy || state.analyzing;
    if (cancelBtn) cancelBtn.disabled = !state.busy;
    if (startBtn) {
      startBtn.classList.toggle('is-busy', state.busy);
    }
    updateMethodUi();
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

  const updateAuthUi = () => {
    if (!window.ToolsAuth) {
      state.admin = false;
      if (isLocalProvider() && !state.files.length && !state.busy) {
        state.provider = PROVIDER_AWS;
      }
      updateMethodUi();
      renderUsage();
      updateControls();
      return;
    }

    const authed = authIsReady();
    const wasAdmin = state.admin;
    state.admin = authed && isAdminUser();
    if (!state.admin && isLocalProvider() && !state.files.length && !state.busy) {
      state.provider = PROVIDER_AWS;
      if (approveEl) approveEl.checked = false;
    }
    const accountSub = authed ? getAuthSub() : '';
    if (state.accountSub !== accountSub) {
      state.accountSub = accountSub;
      state.usageRequestId += 1;
      state.historyRequestId += 1;
      state.usage = null;
      state.usageLoading = false;
      state.historyItems = [];
      state.historyNextCursor = '';
      state.historyLoading = false;
      state.historyDetail = null;
      if (historyDialogEl?.open) closeHistory();
      showHistoryList();
    }

    const shouldPreferLocal = state.admin && !wasAdmin && !state.files.length && !state.busy;
    if (shouldPreferLocal) {
      const localKnownUnavailable = state.localConfig !== null && (
        state.localConfig?.enabled !== true || state.localConfig?.configured !== true
      );
      state.provider = localKnownUnavailable ? PROVIDER_AWS : PROVIDER_LOCAL;
      if (approveEl) approveEl.checked = false;
      if (processingDetailsEl) processingDetailsEl.open = false;
    }

    updateMethodUi();
    updateSummary();
    if (shouldPreferLocal && isLocalProvider()) void refreshLocalWorkerStatus();
    renderUsage();
    updateControls();
  };

  const updateSummary = () => {
    const totalCount = state.files.length;
    const readyCount = state.files.filter((item) => item.status === 'ready').length;
    const skippedCount = state.files.filter((item) => item.status === 'skipped').length;
    const completedCount = completedFiles().length;
    const partialCount = partialFiles().length;
    const failedCount = state.files.filter((item) => item.status === 'failed').length;
    const totalCost = estimatedTotal();
    const runCost = finalTotal();

    if (!totalCount) {
      setStatus(summaryEl, '', '');
      if (summaryEl) summaryEl.hidden = true;
      setText(totalEl, '');
      syncProviderUi();
      return;
    }

    const parts = [];
    if (readyCount) parts.push(`${readyCount} ready`);
    if (skippedCount) parts.push(`${skippedCount} skipped`);
    if (completedCount) parts.push(`${completedCount} complete`);
    if (partialCount) parts.push(`${partialCount} partial`);
    if (failedCount) parts.push(`${failedCount} failed`);
    if (!parts.length) parts.push(`${totalCount} selected`);
    if (!isLocalProvider()) parts.push(`${formatUsd(completedCount || partialCount ? runCost : totalCost)} estimated`);
    if (summaryEl) summaryEl.hidden = false;
    setStatus(
      summaryEl,
      parts.join(' · '),
      skippedCount || partialCount || failedCount ? 'warning' : 'success'
    );
    setText(
      totalEl,
      isLocalProvider()
        ? 'Home GPU processing'
        : completedCount || partialCount
          ? `Estimated charge: ${formatUsd(runCost)}`
          : `Estimated charge: ${formatUsd(totalCost)}`
    );
    syncProviderUi();
  };

  const statusLabel = (item) => {
    if (item.status === 'checking') return 'Checking';
    if (item.status === 'skipped') return `Skipped: ${item.skipReason || 'Not eligible'}`;
    if (item.status === 'ready') return 'Ready';
    if (item.status === 'presigning') return item.provider === PROVIDER_LOCAL ? 'Requesting secure ticket' : 'Preparing upload';
    if (item.status === 'uploading') return `Uploading ${Math.round((Number(item.progress) || 0) * 100)}%`;
    if (item.status === 'starting') return item.provider === PROVIDER_LOCAL ? 'Starting home PC job' : 'Starting job';
    if (item.status === 'transcribing') {
      if (item.provider === PROVIDER_LOCAL && item.localStage) {
        const progress = Math.round(Math.min(1, Math.max(0, Number(item.localProgress) || 0)) * 100);
        return `${item.localStage}${progress ? ` ${progress}%` : ''}`;
      }
      return 'Transcribing';
    }
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

  const canResumeItem = (item) => !state.busy && !state.analyzing && isRecoverableItem(item);

  const resumableFiles = () => state.files.filter(isRecoverableItem);

  const renderTable = () => {
    if (tableWrapEl) tableWrapEl.hidden = state.files.length === 0;
    fileRowsEl.innerHTML = state.files.map((item) => {
      const removable = canRemoveItem(item);
      const resumable = canResumeItem(item);
      const metadata = [
        formatBytes(item.bytes),
        item.extension ? item.extension.toUpperCase() : '',
        item.durationSeconds ? formatClock(item.durationSeconds) : ''
      ].filter(Boolean).join(' · ');
      const showStatus = item.status !== 'ready';
      return `
      <article class="transcribe-file-card" data-tone="${escapeHtml(rowTone(item))}">
        <div class="transcribe-file-main">
          <span class="transcribe-file-name">${escapeHtml(item.name)}</span>
          <span class="transcribe-file-meta">${escapeHtml(metadata)}</span>
        </div>
        ${showStatus ? `<span class="transcribe-file-status">${escapeHtml(statusLabel(item))}</span>` : ''}
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
          ><span aria-hidden="true">×</span></button>
        </div>
      </article>
    `;
    }).join('');
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
          <span class="transcribe-file-meta">${escapeHtml(formatClock(item.durationSeconds))}</span>
        </div>
        <span class="transcribe-file-status">${escapeHtml(statusLabel(item))}</span>
      </article>
    `).join('');
  };

  const renderResults = () => {
    if (!resultsEl) return;
    const resultItems = state.files.filter((item) =>
      item.transcript || ['complete', 'partial', 'failed', 'canceled'].includes(item.status));
    const completedCount = completedFiles().length;
    const partialCount = partialFiles().length;
    const failedCount = state.files.filter((item) => item.status === 'failed').length;
    const canceledCount = state.files.filter((item) => item.status === 'canceled').length;
    const skippedCount = state.files.filter((item) => item.status === 'skipped').length;
    const resumeCount = resumableFiles().length;
    if (resumeAllBtn) {
      resumeAllBtn.hidden = resumeCount === 0;
      resumeAllBtn.disabled = state.busy || state.analyzing || resumeCount === 0;
      setText(resumeAllBtn, `Resume all ${resumeCount} pending`);
    }
    if (resultsSummaryEl) {
      const summaryParts = [];
      if (completedCount) summaryParts.push(`${completedCount} complete`);
      if (partialCount) summaryParts.push(`${partialCount} partial`);
      if (failedCount) summaryParts.push(`${failedCount} failed`);
      if (canceledCount) summaryParts.push(`${canceledCount} canceled`);
      if (skippedCount) summaryParts.push(`${skippedCount} skipped`);
      if (!isLocalProvider() && resultItems.length) summaryParts.push(`${formatUsd(finalTotal())} estimated`);
      setText(
        resultsSummaryEl,
        resultItems.length
          ? summaryParts.join(' · ')
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
        ? item.provider === PROVIDER_LOCAL
          ? 'Completed on Home GPU'
          : `Estimated charge: ${formatUsd(item.costUsd || item.estimatedCostUsd || 0)} · ${item.billableSeconds || 0} billable sec`
        : isPartial
          ? item.provider === PROVIDER_LOCAL
            ? item.error || 'The transcript may have ended before the source media.'
            : `${item.error || 'The transcript may have ended before the source media.'} Estimated charge: ${formatUsd(item.costUsd || item.estimatedCostUsd || 0)}.`
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
    updateMethodUi();
    const readyCount = acceptedFiles().filter((item) => item.status === 'ready').length;
    const approved = Boolean(approveEl && approveEl.checked);
    const localConfigPending = isLocalProvider() && (
      state.localConfigLoading ||
      state.localConfig?.enabled !== true ||
      state.localConfig?.configured !== true
    );
    const pickerDisabled = state.busy || state.analyzing || localConfigPending;
    if (fileEl) fileEl.disabled = pickerDisabled;
    if (addFilesBtn) addFilesBtn.disabled = pickerDisabled;
    if (dropzoneEl) {
      dropzoneEl.dataset.disabled = pickerDisabled ? 'true' : 'false';
      dropzoneEl.setAttribute('aria-disabled', pickerDisabled ? 'true' : 'false');
    }
    if (approveEl) approveEl.disabled = state.busy || state.analyzing || readyCount === 0;
    if (startBtn) {
      const ready = authIsReady() && runConfigIsValid() && approved && readyCount > 0;
      startBtn.disabled = state.busy || state.analyzing || !ready;
      startBtn.dataset.ready = ready ? 'true' : 'false';
    }
    if (resumeAllBtn) {
      const count = resumableFiles().length;
      resumeAllBtn.hidden = count === 0;
      resumeAllBtn.disabled = state.busy || state.analyzing || count === 0;
      setText(resumeAllBtn, `Resume all ${count} pending`);
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
      const err = new Error(responseErrorMessage(data, `Request failed (${res.status}).`));
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

  const workerFetchJson = async (workerOrigin, path, ticket = '', options = {}) => {
    const origin = normalizeWorkerOrigin(workerOrigin);
    const headers = new Headers(options.headers || {});
    if (ticket) headers.set('Authorization', `Bearer ${ticket}`);
    const requestOptions = {
      ...options,
      headers,
      cache: options.cache || 'no-store'
    };
    if (isLoopbackWorkerOrigin(origin)) requestOptions.targetAddressSpace = 'loopback';
    const request = new Request(new URL(path, `${origin}/`), requestOptions);
    return readJson(await fetch(request));
  };

  const refreshLocalWorkerStatus = async () => {
    if (!isAdminUser() || !isLocalProvider() || state.localConfigLoading) return;
    state.localConfigLoading = true;
    setLocalStatus('checking', 'Checking Home GPU...');
    updateMethodUi();
    const fallbackToAws = (message) => {
      setLocalStatus('unavailable', message);
      if (!state.files.length && !state.busy) {
        state.provider = PROVIDER_AWS;
        if (approveEl) approveEl.checked = false;
        if (processingDetailsEl) processingDetailsEl.open = false;
      }
    };

    try {
      let config;
      try {
        config = await authFetchJson(`${API_BASE}/local-config`, { method: 'GET' });
      } catch {
        setLocalStatus('offline', 'Home GPU availability could not be checked. Check the connection and try again.');
        return;
      }

      state.localConfig = { ...config };
      if (config.enabled !== true || config.configured !== true) {
        const message = cleanText(config.disabledReason) || (config.enabled !== true
          ? 'Home GPU processing is disabled.'
          : 'Home GPU is not configured.');
        fallbackToAws(message);
        return;
      }
      if (!isLocalProvider()) return;

      let workerOrigin;
      try {
        workerOrigin = normalizeWorkerOrigin(config.workerOrigin);
      } catch {
        fallbackToAws('Home GPU is not configured correctly.');
        return;
      }
      state.localConfig.workerOrigin = workerOrigin;

      try {
        const health = await workerFetchJson(workerOrigin, '/v1/health', '', { method: 'GET' });
        const healthStatus = cleanText(health?.status).toLowerCase();
        if (health?.ready !== true || ['offline', 'unavailable', 'error', 'failed'].includes(healthStatus)) {
          throw new Error('Worker is not ready.');
        }
        setLocalStatus('online', 'Home GPU is online and ready.');
      } catch {
        setLocalStatus('offline', 'Home GPU is offline. Start the worker, then check again.');
      }
    } finally {
      state.localConfigLoading = false;
      updateMethodUi();
      updateSummary();
      updateControls();
      if (!isLocalProvider()) void loadUsage();
    }
  };

  const formatUsageLimitUsd = (value) => {
    const amount = Math.max(0, Number(value) || 0);
    return Number.isInteger(amount) ? `$${amount.toFixed(0)}` : formatUsageUsd(amount);
  };

  const renderUsage = () => {
    if (usageEl) usageEl.hidden = isLocalProvider();
    if (isLocalProvider()) return;
    const configuredLimit = Math.max(0, Number(state.config.dailyCostLimitUsd || 100) || 100);
    const formattedConfiguredLimit = formatUsageLimitUsd(configuredLimit);
    if (!authIsReady()) {
      setText(usageValueEl, `-- / ${formattedConfiguredLimit}`);
      if (usageProgressEl) {
        usageProgressEl.max = configuredLimit;
        usageProgressEl.value = 0;
        usageProgressEl.textContent = 'Reserved budget unavailable';
      }
      setText(usageTooltipEl, 'Sign in to view today\'s reserved AWS Transcribe safety budget. This is not billed spend.');
      if (usageEl) {
        usageEl.dataset.tone = '';
        usageEl.setAttribute('aria-label', 'Today\'s reserved AWS Transcribe safety budget is available after sign-in');
      }
      return;
    }
    if (state.usageLoading && !state.usage) {
      setText(usageValueEl, `Loading / ${formattedConfiguredLimit}`);
      setText(usageTooltipEl, 'Checking today\'s reserved Amazon Transcribe safety budget...');
      if (usageEl) usageEl.setAttribute('aria-label', 'Loading today\'s reserved AWS Transcribe safety budget');
      return;
    }
    if (!state.usage) {
      setText(usageValueEl, `-- / ${formattedConfiguredLimit}`);
      setText(usageTooltipEl, 'The reserved AWS Transcribe safety budget is temporarily unavailable.');
      if (usageEl) usageEl.setAttribute('aria-label', 'Today\'s reserved AWS Transcribe safety budget is temporarily unavailable');
      return;
    }
    const usedUsd = Math.max(0, Number(state.usage.usedUsd) || 0);
    const limitUsd = Math.max(0, Number(state.usage.limitUsd) || configuredLimit);
    const remainingUsd = Math.max(0, Number(state.usage.remainingUsd ?? (limitUsd - usedUsd)) || 0);
    const fileCount = Math.max(0, Number(state.usage.fileCount) || 0);
    const formattedUsage = `${formatUsageUsd(usedUsd)} / ${formatUsageLimitUsd(limitUsd)}`;
    setText(usageValueEl, formattedUsage);
    if (usageProgressEl) {
      usageProgressEl.max = limitUsd || 100;
      usageProgressEl.value = Math.min(usedUsd, limitUsd || usedUsd);
      usageProgressEl.textContent = `${formatUsageUsd(usedUsd)} reserved of ${formatUsageUsd(limitUsd)}`;
    }
    const usageDetails = `AWS reserved safety budget: ${formatUsageUsd(remainingUsd)} remaining · ${fileCount} file${fileCount === 1 ? '' : 's'} · ${formatUtcReset(state.usage.resetsAt)} · not billed spend`;
    setText(usageTooltipEl, usageDetails);
    if (usageEl) {
      const ratio = limitUsd > 0 ? usedUsd / limitUsd : 0;
      usageEl.dataset.tone = ratio >= 1 ? 'error' : ratio >= .8 ? 'warning' : '';
      usageEl.setAttribute('aria-label', `Today\'s reserved AWS Transcribe safety budget: ${formattedUsage}; this is not billed spend`);
    }
  };

  const loadUsage = async () => {
    if (isLocalProvider()) {
      state.usageRequestId += 1;
      state.usageLoading = false;
      renderUsage();
      return;
    }
    const sub = getAuthSub();
    const requestId = ++state.usageRequestId;
    if (!sub) {
      state.usage = null;
      state.usageLoading = false;
      renderUsage();
      return;
    }
    state.usageLoading = true;
    renderUsage();
    try {
      const data = await authFetchJson(`${API_BASE}/usage`, { method: 'GET' });
      if (requestId !== state.usageRequestId || sub !== getAuthSub()) return;
      state.usage = data?.usage && typeof data.usage === 'object' ? data.usage : null;
    } catch {
      if (requestId !== state.usageRequestId || sub !== getAuthSub()) return;
      state.usage = null;
    } finally {
      if (requestId === state.usageRequestId) {
        state.usageLoading = false;
        renderUsage();
      }
    }
  };

  const historyItemId = (item) => cleanText(item?.id || item?.jobName);

  const historyItemMeta = (item) => {
    const status = String(item?.status || '').toUpperCase() === 'PARTIAL' ? 'Needs review' : 'Complete';
    const cost = Number(item?.costUsd);
    const parts = [status, formatHistoryDate(item?.completedAt)];
    if (item?.costUsd !== null && item?.costUsd !== undefined && Number.isFinite(cost)) parts.push(`Est. AWS charge ${formatUsd(cost)}`);
    if (Number(item?.durationSeconds) > 0) parts.push(formatClock(item.durationSeconds));
    return parts.join(' · ');
  };

  const renderHistoryList = () => {
    if (!historyListEl) return;
    if (!state.historyItems.length) {
      historyListEl.innerHTML = state.historyLoading
        ? '<p class="transcribe-empty">Loading transcript history...</p>'
        : '<p class="transcribe-empty">No saved transcripts yet. Completed Amazon Transcribe results collected by this browser will appear here.</p>';
    } else {
      historyListEl.innerHTML = state.historyItems.map((item) => `
        <article class="transcribe-history-item">
          <div>
            <h3>${escapeHtml(item.filename || 'media')}</h3>
            <p>${escapeHtml(historyItemMeta(item))}</p>
          </div>
          <div class="transcribe-history-item-actions">
            <button type="button" class="btn-secondary" data-transcribe-history-action="view" data-id="${escapeHtml(historyItemId(item))}">View transcript</button>
          </div>
        </article>
      `).join('');
    }
    if (historyLoadMoreBtn) {
      historyLoadMoreBtn.hidden = !state.historyNextCursor;
      historyLoadMoreBtn.disabled = state.historyLoading;
    }
    if (historyRefreshBtn) historyRefreshBtn.disabled = state.historyLoading;
    setStatus(
      historyStatusEl,
      state.historyLoading
        ? 'Loading transcript history...'
        : state.historyItems.length
          ? `${state.historyItems.length} saved transcript${state.historyItems.length === 1 ? '' : 's'} loaded.`
          : 'No saved transcripts yet.',
      ''
    );
  };

  const showHistoryList = () => {
    state.historyDetail = null;
    if (historyDetailEl) historyDetailEl.hidden = true;
    if (historyListViewEl) historyListViewEl.hidden = false;
    if (historyTranscriptEl) historyTranscriptEl.value = '';
    renderHistoryList();
  };

  const renderHistoryDetail = () => {
    const item = state.historyDetail;
    if (!item) {
      showHistoryList();
      return;
    }
    if (historyListViewEl) historyListViewEl.hidden = true;
    if (historyDetailEl) historyDetailEl.hidden = false;
    setText(historyDetailNameEl, item.filename || 'Transcript');
    setText(historyDetailMetaEl, historyItemMeta(item));
    if (historyTranscriptEl) historyTranscriptEl.value = String(item.transcript || '');
  };

  const loadHistory = async ({ append = false } = {}) => {
    if (isLocalProvider()) return;
    const sub = getAuthSub();
    if (!sub || state.historyLoading) return;
    const cursor = append ? state.historyNextCursor : '';
    const requestId = ++state.historyRequestId;
    let loadError = '';
    state.historyLoading = true;
    if (!append) {
      state.historyItems = [];
      state.historyNextCursor = '';
    }
    renderHistoryList();
    const params = new URLSearchParams({ limit: String(HISTORY_PAGE_SIZE) });
    if (cursor) params.set('cursor', cursor);
    try {
      const data = await authFetchJson(`${API_BASE}/history?${params.toString()}`, { method: 'GET' });
      if (requestId !== state.historyRequestId || sub !== getAuthSub()) return;
      const items = Array.isArray(data?.items) ? data.items : [];
      const knownIds = new Set(state.historyItems.map(historyItemId));
      state.historyItems = append
        ? [...state.historyItems, ...items.filter((item) => !knownIds.has(historyItemId(item)))]
        : items;
      state.historyNextCursor = cleanText(data?.nextCursor);
    } catch (err) {
      if (requestId !== state.historyRequestId || sub !== getAuthSub()) return;
      loadError = err?.message || 'Unable to load transcript history.';
    } finally {
      if (requestId === state.historyRequestId) {
        state.historyLoading = false;
        renderHistoryList();
        if (loadError) setStatus(historyStatusEl, loadError, 'error');
      }
    }
  };

  const loadHistoryDetail = async (id) => {
    const safeId = cleanText(id);
    const sub = getAuthSub();
    if (!safeId || !sub) return;
    const requestId = ++state.historyRequestId;
    setStatus(historyStatusEl, 'Loading transcript...', '');
    try {
      const data = await authFetchJson(`${API_BASE}/history?job=${encodeURIComponent(safeId)}`, { method: 'GET' });
      if (requestId !== state.historyRequestId || sub !== getAuthSub()) return;
      state.historyDetail = data?.item && typeof data.item === 'object' ? data.item : null;
      if (!state.historyDetail) throw new Error('The saved transcript was not found.');
      renderHistoryDetail();
      setStatus(historyStatusEl, '', '');
      historyBackBtn?.focus();
    } catch (err) {
      if (requestId !== state.historyRequestId || sub !== getAuthSub()) return;
      setStatus(historyStatusEl, err?.message || 'Unable to load this transcript.', 'error');
    }
  };

  const openHistory = () => {
    if (isLocalProvider()) return;
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in to view your account\'s AWS transcript history.', 'warning');
      return;
    }
    showHistoryList();
    if (typeof historyDialogEl?.showModal === 'function') historyDialogEl.showModal();
    else historyDialogEl?.setAttribute('open', '');
    void loadHistory();
  };

  const closeHistory = () => {
    if (typeof historyDialogEl?.close === 'function' && historyDialogEl.open) historyDialogEl.close();
    else historyDialogEl?.removeAttribute('open');
  };

  const deleteHistoryDetail = async () => {
    const item = state.historyDetail;
    const id = historyItemId(item);
    if (!id || !window.confirm(`Delete the saved transcript for "${item.filename || 'this file'}"?`)) return;
    if (historyDeleteBtn) historyDeleteBtn.disabled = true;
    try {
      await authFetchJson(`${API_BASE}/history?job=${encodeURIComponent(id)}`, { method: 'DELETE' });
      state.historyItems = state.historyItems.filter((entry) => historyItemId(entry) !== id);
      showHistoryList();
      setStatus(historyStatusEl, 'Saved transcript deleted.', 'success');
    } catch (err) {
      setStatus(historyStatusEl, err?.message || 'Unable to delete this transcript.', 'error');
    } finally {
      if (historyDeleteBtn) historyDeleteBtn.disabled = false;
    }
  };

  const notificationApiAvailable = () =>
    typeof window.Notification === 'function' &&
    typeof window.Notification.requestPermission === 'function';

  const readNotificationPreference = () => {
    try {
      return window.localStorage.getItem(NOTIFICATION_PREFERENCE_KEY) === 'on';
    } catch {
      return false;
    }
  };

  const saveNotificationPreference = (enabled) => {
    try {
      window.localStorage.setItem(NOTIFICATION_PREFERENCE_KEY, enabled ? 'on' : 'off');
    } catch {}
  };

  const updateNotificationUi = () => {
    if (!notificationsBtn) return;
    notificationsBtn.title = 'Completion and failure alerts work while this page remains open.';
    if (!notificationApiAvailable()) {
      state.notificationsEnabled = false;
      notificationsBtn.disabled = true;
      notificationsBtn.dataset.state = 'unsupported';
      notificationsBtn.setAttribute('aria-pressed', 'false');
      setText(notificationsBtn, 'Notifications unavailable');
      return;
    }
    const permission = window.Notification.permission;
    if (permission === 'denied') {
      state.notificationsEnabled = false;
      notificationsBtn.disabled = true;
      notificationsBtn.dataset.state = 'denied';
      notificationsBtn.setAttribute('aria-pressed', 'false');
      setText(notificationsBtn, 'Notifications blocked');
      return;
    }
    state.notificationsEnabled = permission === 'granted' && readNotificationPreference();
    notificationsBtn.disabled = false;
    notificationsBtn.dataset.state = state.notificationsEnabled ? 'enabled' : 'off';
    notificationsBtn.setAttribute('aria-pressed', state.notificationsEnabled ? 'true' : 'false');
    setText(notificationsBtn, state.notificationsEnabled ? 'Notifications on' : 'Notifications off');
  };

  const toggleNotifications = async () => {
    if (!notificationApiAvailable()) return;
    if (window.Notification.permission === 'granted') {
      saveNotificationPreference(!state.notificationsEnabled);
      updateNotificationUi();
      setStatus(
        runStatusEl,
        state.notificationsEnabled
          ? 'Browser alerts are on while this page remains open.'
          : 'Browser alerts are off.',
        state.notificationsEnabled ? 'success' : ''
      );
      return;
    }
    let permission = window.Notification.permission;
    if (permission === 'default') {
      try {
        permission = await window.Notification.requestPermission();
      } catch {
        permission = 'denied';
      }
    }
    saveNotificationPreference(permission === 'granted');
    updateNotificationUi();
    setStatus(
      runStatusEl,
      permission === 'granted'
        ? 'Browser alerts are on while this page remains open.'
        : 'Notifications were not enabled. You can change this site permission in your browser settings.',
      permission === 'granted' ? 'success' : 'warning'
    );
  };

  const notifyItem = (item) => {
    const status = String(item?.status || '');
    if (!state.notificationsEnabled || !['complete', 'partial', 'failed'].includes(status)) return;
    if (item.notifiedStatus === status) return;
    item.notifiedStatus = status;
    const title = status === 'complete'
      ? 'Transcription complete'
      : status === 'partial'
        ? 'Transcript needs review'
        : 'Transcription failed';
    const body = status === 'complete'
      ? `${item.name} is ready.`
      : status === 'partial'
        ? `${item.name} may have ended early.`
        : `${item.name} could not be transcribed.`;
    try {
      const notification = new window.Notification(title, {
        body,
        tag: `transcribe-${item.id || item.name}-${status}`
      });
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    } catch {}
  };

  const loadConfig = async () => {
    try {
      const res = await fetch(`${API_BASE}/config`, { method: 'GET' });
      const data = await readJson(res);
      state.config = { ...DEFAULT_CONFIG, ...data };
      if (state.config.configured !== true) {
        throw new Error('Transcribe is not fully configured on the server.');
      }
      updateSummary();
      if (historyPrivacyEl) {
        const days = Math.max(1, Number(state.config.historyRetentionDays) || 90);
        setText(historyPrivacyEl, `Saved AWS transcripts, filenames, and job metadata are retained for ${days} days. Uploaded media is removed after processing; abandoned uploads use the configured S3 lifecycle.`);
      }
      renderUsage();
    } catch (err) {
      setStatus(runStatusEl, err?.message || 'Transcribe configuration is unavailable.', 'warning');
      state.config = { ...DEFAULT_CONFIG, configured: false };
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

    const config = activeConfig();
    const formats = supportedFormats();
    let acceptedCost = estimatedTotal();
    let addedAcceptedCount = 0;

    for (let i = 0; i < newItems.length; i += 1) {
      const item = newItems[i];
      if (item.status === 'skipped') {
        renderTable();
        continue;
      }
      if (countedForRunLimit() >= Number(config.maxFilesPerRun || DEFAULT_CONFIG.maxFilesPerRun)) {
        item.status = 'skipped';
        item.skipReason = `Run limit is ${config.maxFilesPerRun || DEFAULT_CONFIG.maxFilesPerRun} files.`;
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
      if (item.bytes > Number(config.maxFileBytes || DEFAULT_CONFIG.maxFileBytes)) {
        item.status = 'skipped';
        item.skipReason = `File exceeds ${formatBytes(config.maxFileBytes || DEFAULT_CONFIG.maxFileBytes)}.`;
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
      if (item.durationSeconds < Number(config.minDurationSeconds || 15)) {
        item.status = 'skipped';
        item.skipReason = `Under ${config.minDurationSeconds || 15} seconds.`;
        renderTable();
        continue;
      }
      const exceedsServiceDuration = isLocalProvider()
        ? item.durationSeconds > Number(config.maxServiceDurationSeconds || DEFAULT_CONFIG.maxServiceDurationSeconds)
        : item.durationSeconds > Number(state.config.maxServiceDurationSeconds || DEFAULT_CONFIG.maxServiceDurationSeconds);
      if (exceedsServiceDuration) {
        item.status = 'skipped';
        item.skipReason = `Exceeds ${(config.service || (isLocalProvider() ? 'Home GPU' : 'Amazon Transcribe'))}'s ${formatClock(config.maxServiceDurationSeconds || DEFAULT_CONFIG.maxServiceDurationSeconds)} media limit.`;
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
      if (!isLocalProvider() && acceptedCost + item.estimatedCostUsd > Number(state.config.maxTotalCostUsd || DEFAULT_CONFIG.maxTotalCostUsd)) {
        item.status = 'skipped';
        item.skipReason = `Total estimate cap is ${formatUsd(state.config.maxTotalCostUsd)}.`;
        renderTable();
        continue;
      }

      item.status = 'ready';
      acceptedCost += item.estimatedCostUsd;
      addedAcceptedCount += 1;
      renderTable();
    }

    state.analyzing = false;
    setBusy(false);
    const skippedAddedCount = newItems.filter((item) => item.status === 'skipped').length;
    if (!addedAcceptedCount) {
      setStatus(runStatusEl, 'No newly selected files are eligible for transcription.', 'warning');
    } else if (skippedAddedCount) {
      setStatus(runStatusEl, `${addedAcceptedCount} added · ${skippedAddedCount} skipped`, 'warning');
    } else {
      setStatus(runStatusEl, '', '');
    }
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
        notifyItem(item);
        void loadUsage();
        if (historyDialogEl?.open) void loadHistory();
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
        notifyItem(item);
        void loadUsage();
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
    void loadUsage();
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

  const sha256Hex = async (buffer) => {
    if (!window.crypto?.subtle) {
      throw new Error('Secure SHA-256 hashing is unavailable in this browser.');
    }
    const digest = await window.crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
  };

  const normalizeLocalProgress = (value) => {
    const progress = Number(value);
    if (!Number.isFinite(progress) || progress <= 0) return 0;
    return Math.min(1, progress > 1 ? progress / 100 : progress);
  };

  const localJobPath = (item, suffix = '') =>
    `/v1/jobs/${encodeURIComponent(String(item.localJobId || ''))}${suffix}`;

  const acknowledgeLocalItem = async (item) => {
    if (!item?.localWorkerOrigin || !item?.localJobId || !item?.localTicket) return;
    try {
      await workerFetchJson(item.localWorkerOrigin, localJobPath(item, '/ack'), item.localTicket, {
        method: 'POST'
      });
      item.localTicket = '';
    } catch {}
  };

  const cancelLocalItem = async (item) => {
    if (!item?.localWorkerOrigin || !item?.localJobId || !item?.localTicket) return false;
    try {
      await workerFetchJson(item.localWorkerOrigin, localJobPath(item, '/cancel'), item.localTicket, {
        method: 'POST'
      });
      return true;
    } catch {
      return false;
    }
  };

  const cancelActiveLocalJobs = async () => {
    const activeItems = state.files.filter((item) =>
      item.provider === PROVIDER_LOCAL &&
      item.localJobId &&
      item.localTicket &&
      ['presigning', 'uploading', 'starting', 'transcribing'].includes(String(item.status || '')));
    if (!activeItems.length) return 0;
    const results = await Promise.all(activeItems.map(cancelLocalItem));
    return results.filter(Boolean).length;
  };

  const uploadLocalChunks = async (item, chunkBytes) => {
    const safeChunkBytes = Math.max(1, Math.floor(Number(chunkBytes) || DEFAULT_LOCAL_CHUNK_BYTES));
    const chunkCount = Math.ceil(item.bytes / safeChunkBytes);
    for (let index = 0; index < chunkCount; index += 1) {
      if (state.canceled) throw new Error('Canceled.');
      const start = index * safeChunkBytes;
      const end = Math.min(item.bytes, start + safeChunkBytes);
      const chunkBuffer = await item.file.slice(start, end).arrayBuffer();
      const chunkHash = await sha256Hex(chunkBuffer);
      let uploaded = false;
      for (let attempt = 0; attempt <= LOCAL_UPLOAD_RETRIES; attempt += 1) {
        const controller = new AbortController();
        state.activeController = controller;
        try {
          await workerFetchJson(
            item.localWorkerOrigin,
            localJobPath(item, `/chunks/${index}`),
            item.localTicket,
            {
              method: 'PUT',
              headers: {
                'Content-Type': 'application/octet-stream',
                'Content-Range': `bytes ${start}-${end - 1}/${item.bytes}`,
                'X-Chunk-SHA256': chunkHash
              },
              body: chunkBuffer,
              signal: controller.signal
            }
          );
          uploaded = true;
          break;
        } catch (err) {
          if (state.canceled || err?.name === 'AbortError' || attempt >= LOCAL_UPLOAD_RETRIES || !isTransientRunError(err)) {
            throw err;
          }
          await sleep(500 * (2 ** attempt));
        } finally {
          if (state.activeController === controller) state.activeController = null;
        }
      }
      if (!uploaded) throw new Error(`Local upload failed on chunk ${index + 1}.`);
      item.progress = end / item.bytes;
      renderTable();
      renderProcessingList();
      updateProgress({
        stateName: 'visible',
        ratio: item.progress,
        label: `Sending ${item.name} to the home PC (${Math.round(item.progress * 100)}%)`
      });
    }
    return chunkCount;
  };

  const pollLocalRun = async (item) => {
    while (!state.canceled) {
      const controller = new AbortController();
      state.activeController = controller;
      let data;
      try {
        data = await workerFetchJson(item.localWorkerOrigin, localJobPath(item), item.localTicket, {
          method: 'GET',
          signal: controller.signal
        });
      } finally {
        if (state.activeController === controller) state.activeController = null;
      }

      const status = cleanText(data?.status).toUpperCase();
      item.localStage = cleanText(data?.stage || (status === 'QUEUED' ? 'Queued on home PC' : 'Transcribing on home PC'));
      item.localProgress = normalizeLocalProgress(data?.progress);
      if (['COMPLETED', 'COMPLETE', 'SUCCEEDED', 'DONE'].includes(status)) {
        item.transcript = String(data?.transcript || '').trim();
        item.durationSeconds = Number(data?.durationSeconds || item.durationSeconds || 0);
        item.costUsd = 0;
        const coverageStatus = cleanText(data?.coverageStatus).toUpperCase();
        if (['SUSPECTED_EARLY_END', 'PARTIAL', 'INCOMPLETE'].includes(coverageStatus)) {
          item.status = 'partial';
          item.runErrorType = 'service';
          item.error = friendlyTranscribeError(responseErrorMessage(
            data,
            'The local transcript may have ended before the source media.'
          ));
        } else {
          item.status = 'complete';
          item.runErrorType = '';
          item.error = '';
        }
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        notifyItem(item);
        await acknowledgeLocalItem(item);
        return;
      }
      if (['FAILED', 'ERROR'].includes(status)) {
        item.status = 'failed';
        item.error = friendlyTranscribeError(responseErrorMessage(
          data,
          'The home PC worker could not complete this transcription.'
        ));
        item.runErrorType = 'service';
        item.costUsd = 0;
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        notifyItem(item);
        await acknowledgeLocalItem(item);
        return;
      }
      if (['CANCELED', 'CANCELLED'].includes(status)) {
        item.status = 'canceled';
        item.error = 'Canceled.';
        item.runErrorType = 'processing';
        await acknowledgeLocalItem(item);
        return;
      }

      item.status = 'transcribing';
      item.error = '';
      renderTable();
      renderProcessingList();
      updateProgress({
        stateName: 'visible',
        ratio: item.localProgress,
        label: `${item.localStage}${item.localProgress ? ` (${Math.round(item.localProgress * 100)}%)` : ''}`
      });
      await sleep(LOCAL_POLL_INTERVAL_MS);
    }
    throw new Error('Canceled.');
  };

  const runLocalFile = async (item) => {
    item.status = 'presigning';
    item.error = '';
    item.runErrorType = '';
    item.progress = 0;
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 0, label: `Requesting a secure local ticket for ${item.name}...` });

    const ticketController = new AbortController();
    state.activeController = ticketController;
    let ticketData;
    try {
      ticketData = await authFetchJson(`${API_BASE}/local-ticket`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: item.name,
          format: item.extension,
          contentType: item.contentType,
          bytes: item.bytes,
          durationSeconds: item.durationSeconds
        }),
        signal: ticketController.signal
      });
    } finally {
      if (state.activeController === ticketController) state.activeController = null;
    }
    if (state.canceled) throw new Error('Canceled.');
    if (ticketData?.enabled !== true || ticketData?.configured !== true) {
      throw new Error('Home GPU processing is not available. The batch was not rerouted.');
    }

    const job = ticketData?.job && typeof ticketData.job === 'object' ? ticketData.job : {};
    item.localWorkerOrigin = normalizeWorkerOrigin(ticketData.workerOrigin || state.localConfig?.workerOrigin);
    item.localTicket = cleanText(ticketData.ticket);
    item.localJobId = cleanText(job.id);
    item.localTicketExpiresAt = cleanText(ticketData.expiresAt);
    const chunkBytes = Math.max(1, Math.floor(Number(ticketData.chunkBytes || state.localConfig?.chunkBytes) || DEFAULT_LOCAL_CHUNK_BYTES));
    const chunkCount = Math.ceil(item.bytes / chunkBytes);
    if (!item.localTicket || !item.localJobId) {
      throw new Error('The website did not return a complete Home GPU job ticket. The batch was not rerouted.');
    }

    item.status = 'starting';
    renderTable();
    const createController = new AbortController();
    state.activeController = createController;
    try {
      await workerFetchJson(item.localWorkerOrigin, '/v1/jobs', item.localTicket, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: item.localJobId,
          filename: cleanText(job.filename || item.name),
          format: cleanText(job.format || item.extension),
          contentType: cleanText(job.contentType || item.contentType),
          bytes: Number(job.bytes || item.bytes),
          durationSeconds: Number(job.durationSeconds || item.durationSeconds),
          chunkBytes,
          chunkCount
        }),
        signal: createController.signal
      });
    } finally {
      if (state.activeController === createController) state.activeController = null;
    }

    item.status = 'uploading';
    renderTable();
    await uploadLocalChunks(item, chunkBytes);
    if (state.canceled) throw new Error('Canceled.');

    item.status = 'starting';
    renderTable();
    updateProgress({ stateName: 'visible', ratio: 1, label: `Finalizing ${item.name} on the home PC...` });
    const completeController = new AbortController();
    state.activeController = completeController;
    try {
      await workerFetchJson(item.localWorkerOrigin, localJobPath(item, '/complete'), item.localTicket, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunkCount }),
        signal: completeController.signal
      });
    } finally {
      if (state.activeController === completeController) state.activeController = null;
    }

    item.status = 'transcribing';
    item.localStage = 'Queued on home PC';
    item.localProgress = 0;
    renderTable();
    await pollLocalRun(item);
  };

  const runSelectedFile = async (item) => {
    if (item.provider !== PROVIDER_LOCAL) return runFile(item);
    try {
      return await runLocalFile(item);
    } catch (err) {
      if (!state.canceled && isTransientRunError(err)) {
        setLocalStatus('offline', 'The Home GPU connection was interrupted. Check the worker before retrying.');
      }
      if (!state.canceled) await cancelLocalItem(item);
      throw err;
    }
  };

  const resumeQueue = async (items) => {
    const queue = Array.from(items || []).filter(isRecoverableItem);
    if (!queue.length) return;
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in again before resuming these transcriptions.', 'warning');
      updateAuthUi();
      return;
    }
    if (!runConfigIsValid()) {
      setStatus(runStatusEl, 'Transcription configuration is unavailable. Refresh and try again.', 'warning');
      return;
    }

    state.canceled = false;
    setView('processing');
    setBusy(true);
    renderProcessingList();
    setStatus(runStatusEl, `Resuming ${queue.length} pending transcription${queue.length === 1 ? '' : 's'}...`, '');
    setText(processingCopyEl, `Resuming ${queue.length} existing job${queue.length === 1 ? '' : 's'} in sequence.`);
    updateProgress({ stateName: 'visible', ratio: 0, label: 'Preparing recovered jobs...' });

    for (let index = 0; index < queue.length; index += 1) {
      const item = queue[index];
      if (state.canceled) break;
      item.error = '';
      item.runErrorType = '';
      item.status = item.runToken ? 'transcribing' : 'starting';
      setStatus(runStatusEl, `Resuming ${index + 1} of ${queue.length}: ${item.name}`, '');
      setText(processingCopyEl, `Resuming ${index + 1} of ${queue.length}: ${item.name}`);
      updateProgress({ stateName: 'visible', ratio: index / queue.length, label: `Resuming ${item.name}...` });
      renderTable();
      renderProcessingList();
      try {
        const runToken = item.runToken || await startReservedRun(item);
        item.status = 'transcribing';
        renderTable();
        await pollRun(item, runToken);
      } catch (err) {
        if (state.canceled || err?.name === 'AbortError' || err?.message === 'Canceled.') {
          item.status = 'canceled';
          item.error = 'Canceled.';
          item.runErrorType = 'processing';
        } else {
          item.status = 'failed';
          item.error = friendlyTranscribeError(err?.message || 'Unable to resume transcription.');
          item.runErrorType = item.runToken && isTransientRunError(err) ? 'network' : classifyRunError(err);
          notifyItem(item);
        }
        renderTable();
        renderProcessingList();
        renderResults();
        markSessionDirty();
        persistActiveRunRecovery();
      }
      updateProgress({
        stateName: 'visible',
        ratio: (index + 1) / queue.length,
        label: `${index + 1} of ${queue.length} recovered jobs finished`
      });
    }

    const completed = queue.filter((item) => item.status === 'complete').length;
    const partial = queue.filter((item) => item.status === 'partial').length;
    const failed = queue.filter((item) => item.status === 'failed').length;
    const canceled = queue.filter((item) => item.status === 'canceled').length;
    setStatus(
      runStatusEl,
      state.canceled
        ? `Resume canceled. ${completed} complete, ${partial} partial, ${failed} failed, ${canceled} canceled.`
        : `Resume finished. ${completed} complete, ${partial} partial, ${failed} failed.`,
      partial || failed || canceled ? 'warning' : 'success'
    );
    setBusy(false);
    updateProgress({ stateName: 'hidden' });
    renderTable();
    renderProcessingList();
    renderResults();
    setView('results');
    updateLayoutState();
    markSessionDirty();
    persistActiveRunRecovery();
    void loadUsage();
    if (historyDialogEl?.open) void loadHistory();
  };

  const resumeItem = async (item) => resumeQueue([item]);

  const runBatch = async ({ reportOutcome = false } = {}) => {
    if (!authIsReady()) {
      setStatus(runStatusEl, 'Sign in before starting transcription jobs.', 'warning');
      updateAuthUi();
      if (reportOutcome) reportRunError('permission');
      return;
    }

    if (isLocalProvider() && !isAdminUser()) {
      setStatus(runStatusEl, 'Home GPU processing is available only to the signed-in admin.', 'warning');
      if (reportOutcome) reportRunError('permission');
      return;
    }

    if (!runConfigIsValid()) {
      setStatus(
        runStatusEl,
        isLocalProvider()
          ? 'Home GPU is offline or unavailable. Check the connection and try again.'
          : 'Transcription configuration is unavailable. Refresh and try again.',
        'warning'
      );
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
      setStatus(
        runStatusEl,
        isLocalProvider()
          ? 'Confirm that these files can be sent to your home PC.'
          : 'Review and approve the estimated charge before starting.',
        'warning'
      );
      if (reportOutcome) reportRunError('validation');
      return;
    }

    state.canceled = false;
    setView('processing');
    setBusy(true);
    renderProcessingList();
    setStatus(runStatusEl, `Starting ${queue.length} ${isLocalProvider() ? 'Home GPU ' : ''}transcription job${queue.length === 1 ? '' : 's'}...`, '');
    setText(
      processingCopyEl,
      isLocalProvider()
        ? `Processing ${queue.length} file${queue.length === 1 ? '' : 's'} on the home PC. Keep this tab open while files upload and transcription runs.`
        : `Processing ${queue.length} file${queue.length === 1 ? '' : 's'}. Amazon Transcribe continues independently, but keep this tab open so the site can collect the result, clean up the upload, and save history.`
    );
    updateProgress({ stateName: 'visible', ratio: 0, label: 'Starting batch...' });

    for (let i = 0; i < queue.length; i += 1) {
      const item = queue[i];
      if (state.canceled) break;
      try {
        setStatus(runStatusEl, `Processing ${i + 1} of ${queue.length}: ${item.name}`, '');
        setText(processingCopyEl, `Processing ${i + 1} of ${queue.length}: ${item.name}`);
        await runSelectedFile(item);
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
          notifyItem(item);
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
    if (!isLocalProvider()) {
      void loadUsage();
      if (historyDialogEl?.open) void loadHistory();
    }

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

  if (notificationsBtn) {
    notificationsBtn.addEventListener('click', () => {
      void toggleNotifications();
    });
  }

  if (historyOpenBtn) historyOpenBtn.addEventListener('click', openHistory);
  if (historyCloseBtn) historyCloseBtn.addEventListener('click', closeHistory);
  if (historyRefreshBtn) historyRefreshBtn.addEventListener('click', () => {
    showHistoryList();
    void loadHistory();
  });
  if (historyLoadMoreBtn) historyLoadMoreBtn.addEventListener('click', () => {
    void loadHistory({ append: true });
  });
  if (historyBackBtn) historyBackBtn.addEventListener('click', showHistoryList);
  if (historyListEl) {
    historyListEl.addEventListener('click', (event) => {
      const button = event.target.closest('[data-transcribe-history-action="view"]');
      if (!button) return;
      void loadHistoryDetail(button.getAttribute('data-id'));
    });
  }
  if (historyCopyBtn) {
    historyCopyBtn.addEventListener('click', async () => {
      const transcript = String(state.historyDetail?.transcript || '');
      if (!transcript) return;
      try {
        await navigator.clipboard.writeText(transcript);
        setStatus(historyStatusEl, 'Transcript copied.', 'success');
      } catch {
        setStatus(historyStatusEl, 'Copy failed. Select the transcript text manually.', 'error');
      }
    });
  }
  if (historyDownloadBtn) {
    historyDownloadBtn.addEventListener('click', () => {
      const transcript = String(state.historyDetail?.transcript || '');
      if (!transcript) return;
      downloadTranscript(state.historyDetail?.filename, transcript);
    });
  }
  if (historyDeleteBtn) historyDeleteBtn.addEventListener('click', () => void deleteHistoryDetail());

  methodInputs.forEach((input) => {
    input.addEventListener('change', () => {
      if (!input.checked) return;
      if (state.files.length || state.busy || state.analyzing) {
        updateMethodUi();
        return;
      }
      const nextProvider = input.value === PROVIDER_LOCAL ? PROVIDER_LOCAL : PROVIDER_AWS;
      if (nextProvider === PROVIDER_LOCAL && !isAdminUser()) {
        state.provider = PROVIDER_AWS;
        updateMethodUi();
        return;
      }
      state.provider = nextProvider;
      if (approveEl) approveEl.checked = false;
      if (processingDetailsEl) processingDetailsEl.open = false;
      if (isLocalProvider()) {
        state.usageRequestId += 1;
        state.usageLoading = false;
      }
      updateMethodUi();
      updateSummary();
      updateControls();
      setStatus(runStatusEl, '', '');
      if (isLocalProvider()) void refreshLocalWorkerStatus();
      else void loadUsage();
      markSessionDirty();
    });
  });

  if (localRefreshBtn) {
    localRefreshBtn.addEventListener('click', () => {
      void refreshLocalWorkerStatus();
    });
  }

  if (addFilesBtn) {
    addFilesBtn.addEventListener('click', () => {
      if (addFilesBtn.disabled || fileEl.disabled || state.busy || state.analyzing) return;
      fileEl.click();
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
        if (fileEl.disabled || state.busy || state.analyzing) return;
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
      if (fileEl.disabled || state.busy || state.analyzing) return;
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
      const removedIndex = state.files.findIndex((entry) => entry.id === id);
      const item = removedIndex >= 0 ? state.files[removedIndex] : null;
      if (!item || !canRemoveItem(item)) return;
      state.files = state.files.filter((entry) => entry.id !== id);
      persistActiveRunRecovery();
      if (approveEl) approveEl.checked = false;
      setStatus(runStatusEl, '', '');
      renderTable();
      renderResults();
      markSessionDirty();
      window.requestAnimationFrame(() => {
        const removeButtons = Array.from(fileRowsEl.querySelectorAll('[data-transcribe-file-remove]:not(:disabled)'));
        const nextButton = removeButtons[Math.min(removedIndex, removeButtons.length - 1)] || addFilesBtn;
        nextButton?.focus();
      });
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
      const localRun = isLocalProvider();
      state.canceled = true;
      abortActive();
      if (localRun) {
        setStatus(runStatusEl, 'Canceling the home PC job.', 'warning');
        void cancelActiveLocalJobs().then((canceledCount) => {
          if (canceledCount > 0) {
            setStatus(runStatusEl, `Cancellation sent to ${canceledCount} home PC job${canceledCount === 1 ? '' : 's'}.`, 'warning');
          }
        });
      } else {
        setStatus(runStatusEl, 'Canceling. Already submitted AWS jobs may still incur cost.', 'warning');
      }
      setBusy(false);
      updateProgress({ stateName: 'hidden' });
      persistActiveRunRecovery();
    });
  }

  if (resetBtn) resetBtn.addEventListener('click', reset);
  if (newBtn) newBtn.addEventListener('click', reset);
  if (resumeAllBtn) {
    resumeAllBtn.addEventListener('click', () => {
      const queue = resumableFiles();
      void resumeQueue(queue);
    });
  }

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
        downloadTranscript(item.name, transcript);
      }
    });
  }

  document.addEventListener('tools:auth-changed', () => {
    const restoredCount = restoreActiveRunRecovery();
    updateAuthUi();
    updateNotificationUi();
    void loadUsage();
    if (!restoredCount) return;
    setView('results');
    renderTable();
    renderResults();
    setStatus(runStatusEl, `Recovered ${restoredCount} unfinished transcription job${restoredCount === 1 ? '' : 's'}. Resume one or continue all pending jobs without uploading again.`, 'warning');
  });
  window.addEventListener('pagehide', persistActiveRunRecovery);
  document.addEventListener('visibilitychange', updateNotificationUi);

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
      Method: isLocalProvider() ? 'Home GPU' : 'Amazon Transcribe',
      Files: `${state.files.length} selected`,
      Accepted: String(accepted.length),
      Skipped: String(skipped.length)
    };
    if (!isLocalProvider()) payload.inputs['Estimated total'] = formatUsd(estimatedTotal());
    payload.outputSummary = completed.length || partial.length || failed.length
      ? isLocalProvider()
        ? `${completed.length} complete · ${partial.length} partial · ${failed.length} failed`
        : `${completed.length} complete · ${partial.length} partial · ${failed.length} failed · ${formatUsd(finalTotal())} estimated AWS charge`
      : isLocalProvider()
        ? 'No Home GPU transcripts were produced in this browser session.'
        : 'No Amazon Transcribe results were collected in this browser session.';
  });

  loadConfig().finally(() => {
    const restoredCount = restoreActiveRunRecovery();
    setView(restoredCount ? 'results' : 'upload');
    updateAuthUi();
    updateNotificationUi();
    void loadUsage();
    renderTable();
    renderResults();
    if (restoredCount) {
      setStatus(runStatusEl, `Recovered ${restoredCount} unfinished transcription job${restoredCount === 1 ? '' : 's'}. Resume one or continue all pending jobs without uploading again.`, 'warning');
    }
  });
  window.setTimeout(updateAuthUi, 250);
  window.setTimeout(updateAuthUi, 1000);
})();
